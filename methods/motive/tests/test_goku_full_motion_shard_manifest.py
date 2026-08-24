from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from methods.motive.tests.test_goku_full_motion_finalize import (
    _FakeQwenApi,
    _make_qwen_run,
)
from motive import goku_full_motion_finalize as finalizer
from motive import goku_full_motion_shard_manifest as shard_manifest


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _object_sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _build_parent(root: Path) -> Path:
    families = [f"family-{index % 10}" for index in range(330)]
    candidate, outputs, _ = _make_qwen_run(
        root,
        count=330,
        multi_indices=set(range(100)),
        families=families,
    )
    parent = root / "finalizer"
    with mock.patch.object(
        finalizer, "_load_qwen_api", return_value=_FakeQwenApi
    ):
        finalizer.finalize_full_motion(
            candidate_manifest=candidate,
            qwen_outputs=outputs,
            output_dir=parent,
            primary_size=256,
            reserve_size=64,
            min_primary_multi_dynamic=64,
            target_signature_cap=32,
            family_cap=32,
            required_iids=["iid000"],
        )
    return parent


def _materialize(parent: Path, output: Path):
    with mock.patch.object(
        shard_manifest._finalizer, "DEFAULT_CANARY_IID", "iid000"
    ):
        return shard_manifest.materialize_full_motion_shards(
            finalizer_dir=parent, output_dir=output
        )


class GokuFullMotionShardManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.parent = _build_parent(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_contiguous_shards_jobs_and_done_closure(self) -> None:
        original_validator = finalizer.validate_generation_row
        with mock.patch.object(
            shard_manifest._finalizer,
            "DEFAULT_CANARY_IID",
            "iid000",
        ), mock.patch.object(
            shard_manifest._finalizer,
            "validate_generation_row",
            wraps=original_validator,
        ) as validator:
            summary = shard_manifest.materialize_full_motion_shards(
                finalizer_dir=self.parent,
                output_dir=self.root / "sharded",
            )
        self.assertEqual(validator.call_count, 256 + 64 + 330)
        self.assertEqual(summary["layout"]["shard_count"], 32)
        self.assertEqual(summary["layout"]["rows_per_shard"], 8)

        root_lines = (self.parent / "primary_256.jsonl").read_bytes().splitlines(
            keepends=True
        )
        reconstructed = b""
        for index, descriptor in enumerate(summary["shards"]):
            path = self.root / "sharded" / descriptor["path"]
            raw = path.read_bytes()
            self.assertEqual(raw, b"".join(root_lines[index * 8 : (index + 1) * 8]))
            self.assertEqual(
                descriptor["root_row_indices_zero_based"],
                list(range(index * 8, (index + 1) * 8)),
            )
            self.assertEqual(descriptor["sha256"], hashlib.sha256(raw).hexdigest())
            reconstructed += raw
        self.assertEqual(
            reconstructed, (self.parent / "primary_256.jsonl").read_bytes()
        )

        jobs = (self.root / "sharded" / "jobs.tsv").read_text().splitlines()
        self.assertEqual(len(jobs), 33)
        self.assertIn("root_row_start_zero_based", jobs[0])
        self.assertIn("ordered_iids_sha256", jobs[0])
        done = json.loads((self.root / "sharded" / "done.json").read_text())
        payload = dict(done)
        done_digest = payload.pop("done_digest")
        self.assertEqual(done_digest, _object_sha(payload))
        for relative, metadata in done["artifacts"].items():
            raw = (self.root / "sharded" / relative).read_bytes()
            self.assertEqual(metadata["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(metadata["bytes"], len(raw))

    def test_primary_tamper_is_rejected_before_publish(self) -> None:
        primary = self.parent / "primary_256.jsonl"
        primary.write_bytes(primary.read_bytes() + b"\n")
        with self.assertRaises(shard_manifest.GokuFullMotionShardManifestError):
            _materialize(self.parent, self.root / "sharded")
        self.assertFalse((self.root / "sharded").exists())

    def test_upstream_qwen_artifact_tamper_is_rejected(self) -> None:
        done = json.loads((self.parent / "done.json").read_text())
        output = Path(done["inputs"]["qwen_shards"][0]["output_path"])
        output.write_bytes(output.read_bytes() + b"\n")
        with self.assertRaises(shard_manifest.GokuFullMotionShardManifestError):
            _materialize(self.parent, self.root / "sharded")

    def test_terminal_done_tamper_is_rejected(self) -> None:
        done_path = self.parent / "done.json"
        done = json.loads(done_path.read_text())
        done["status"] = "tampered"
        done_path.write_text(json.dumps(done, sort_keys=True) + "\n")
        with self.assertRaises(shard_manifest.GokuFullMotionShardManifestError):
            _materialize(self.parent, self.root / "sharded")

    def test_generation_validator_failure_is_fail_closed(self) -> None:
        original = finalizer.validate_generation_row

        def reject_one(row):
            if row.get("iid") == "iid007":
                raise ValueError("synthetic row rejection")
            return original(row)

        with mock.patch.object(
            shard_manifest._finalizer,
            "DEFAULT_CANARY_IID",
            "iid000",
        ), mock.patch.object(
            shard_manifest._finalizer,
            "validate_generation_row",
            side_effect=reject_one,
        ):
            with self.assertRaisesRegex(
                shard_manifest.GokuFullMotionShardManifestError,
                "synthetic row rejection",
            ):
                shard_manifest.materialize_full_motion_shards(
                    finalizer_dir=self.parent,
                    output_dir=self.root / "sharded",
                )

    def test_determinism_and_create_only_publication(self) -> None:
        _materialize(self.parent, self.root / "first")
        _materialize(self.parent, self.root / "second")
        first_files = sorted(
            path.relative_to(self.root / "first")
            for path in (self.root / "first").rglob("*")
            if path.is_file()
        )
        second_files = sorted(
            path.relative_to(self.root / "second")
            for path in (self.root / "second").rglob("*")
            if path.is_file()
        )
        self.assertEqual(first_files, second_files)
        for relative in first_files:
            self.assertEqual(
                (self.root / "first" / relative).read_bytes(),
                (self.root / "second" / relative).read_bytes(),
            )
        with self.assertRaises(FileExistsError):
            _materialize(self.parent, self.root / "first")


if __name__ == "__main__":
    unittest.main()
