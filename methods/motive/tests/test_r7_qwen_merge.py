from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from motive.qwen_filter import _object_digest
from motive.r7_qwen_merge import (
    DONE_NAME,
    FUSED_NAME,
    R7_QWEN_DONE_SCHEMA,
    R7_QWEN_MERGE_SCHEMA,
    SUMMARY_NAME,
    merge_qwen_shards,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _input_row(
    index: int,
    *,
    existing_visual: bool = False,
) -> dict[str, object]:
    family = "walk" if index % 2 == 0 else "jump"
    row: dict[str, object] = {
        "iid": f"iid-{index:03d}",
        "prompt": f"make the person {family}",
        "src_video": f"{index}/source.mp4",
        "tgt_video": f"{index}/edited.mp4",
        "auto_rule": {"action_families": [family]},
        "r7_expansion_selection": {"primary_family": family},
    }
    if existing_visual:
        row["qwen_evidence"] = {"visual": {"old": True}}
    row["input_digest"] = _object_digest(row)
    return row


def _observation(*, moving: bool) -> dict[str, object]:
    if moving:
        return {
            "schema_version": "qwen-motion-observation-v2",
            "source_action": "no visible action",
            "target_action": "person walks from left to right",
            "source_actor_motion": "none",
            "target_actor_motion": "clear",
            "camera_dominance": "low",
            "background_dominance": "low",
            "artifact_level": "low",
            "preservation_quality": "acceptable",
            "temporal_evidence": [
                "The person advances between ordered target frames."
            ],
            "uncertainty_codes": [],
        }
    return {
        "schema_version": "qwen-motion-observation-v2",
        "source_action": "no visible action",
        "target_action": "no visible action",
        "source_actor_motion": "none",
        "target_actor_motion": "none",
        "camera_dominance": "low",
        "background_dominance": "low",
        "artifact_level": "low",
        "preservation_quality": "acceptable",
        "temporal_evidence": [
            "The actor remains stable across ordered target frames."
        ],
        "uncertainty_codes": [],
    }


def _result(*, moving: bool) -> dict[str, object]:
    if moving:
        return {
            "schema_version": "qwen-motion-judge-v4",
            "verdict": "valid_action",
            "edit_effect": "started",
            "action_signature": "walk right",
            "reason_codes": ["visible_target_motion"],
            "uncertainty_codes": [],
            "confidence": "high",
        }
    return {
        "schema_version": "qwen-motion-judge-v4",
        "verdict": "static",
        "edit_effect": "none",
        "action_signature": "unknown",
        "reason_codes": ["target_static"],
        "uncertainty_codes": [],
        "confidence": "high",
    }


def _materialize(
    root: Path,
    *,
    rows: int = 10,
    existing_visual: bool = False,
) -> tuple[Path, Path]:
    input_path = root / "selection.jsonl"
    qwen_root = root / "qwen"
    input_rows = [
        _input_row(index, existing_visual=existing_visual)
        for index in range(rows)
    ]
    _write_jsonl(input_path, input_rows)
    input_raw = input_path.read_bytes()
    source_sha256 = _sha256(input_raw)
    lines = input_raw.splitlines(keepends=True)
    for shard_index in range(8):
        tag = f"{shard_index:03d}"
        manifest = qwen_root / "manifests" / f"shard-{tag}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest_raw = b"".join(lines[shard_index::8])
        manifest.write_bytes(manifest_raw)
        marker = {
            "partition": "line_modulo_v1",
            "schema_version": "motive-qwen-shard-manifest-v2",
            "shard_count": 8,
            "shard_index": shard_index,
            "shard_rows": len(input_rows[shard_index::8]),
            "shard_sha256": _sha256(manifest_raw),
            "source_rows": len(input_rows),
            "source_sha256": source_sha256,
        }
        marker_path = Path(f"{manifest}.source")
        marker_path.write_text(
            json.dumps(marker, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_rows: list[dict[str, object]] = []
        for row in input_rows[shard_index::8]:
            iid = str(row["iid"])
            moving = int(iid.rsplit("-", 1)[1]) % 3 != 0
            observation = _observation(moving=moving)
            result = _result(moving=moving)
            output_rows.append(
                {
                    "iid": iid,
                    "input_digest": row["input_digest"],
                    "mode": "visual",
                    "status": "ok",
                    "observation": observation,
                    "observation_digest": _object_digest(observation),
                    "observation_repairs": [],
                    "observation_validated_from": "original",
                    "alignment_repairs": [],
                    "result": result,
                    "result_digest": _object_digest(result),
                    "result_validated_from": "original",
                    "visual_input_digest": hashlib.sha256(
                        f"visual-{iid}".encode()
                    ).hexdigest(),
                    "execution_shard_index": shard_index,
                    "execution_shard_count": 8,
                    "execution_manifest": str(manifest.resolve()),
                    "execution_manifest_sha256": _sha256(manifest_raw),
                    "run_config_digest": "a" * 64,
                    "config_digest": hashlib.sha256(
                        f"config-{shard_index}".encode()
                    ).hexdigest(),
                    "implementation_digest": "b" * 64,
                    "model_revision": "c" * 40,
                    "transformers_version": "4.51.3",
                }
            )
        _write_jsonl(
            qwen_root / "shards" / f"qwen-{tag}.jsonl",
            output_rows,
        )
    return input_path, qwen_root


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


class R7QwenMergeTests(unittest.TestCase):
    def test_fuses_in_original_order_and_binds_commit_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path, qwen_root = _materialize(root)
            output_dir = root / "fused"
            summary = merge_qwen_shards(
                input_path=input_path,
                qwen_root=qwen_root,
                output_dir=output_dir,
            )
            input_rows = _read_jsonl(input_path)
            fused_rows = _read_jsonl(output_dir / FUSED_NAME)
            self.assertEqual(
                [row["iid"] for row in fused_rows],
                [row["iid"] for row in input_rows],
            )
            self.assertTrue(
                all(row["qwen_evidence"]["visual"]["status"] == "ok"
                    for row in fused_rows)
            )
            persisted_summary = json.loads(
                (output_dir / SUMMARY_NAME).read_text(encoding="utf-8")
            )
            done = json.loads(
                (output_dir / DONE_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(
                persisted_summary["schema_version"],
                R7_QWEN_MERGE_SCHEMA,
            )
            self.assertEqual(done["schema_version"], R7_QWEN_DONE_SCHEMA)
            self.assertEqual(done["status"], "complete")
            self.assertEqual(
                done["fused_sha256"],
                _sha256((output_dir / FUSED_NAME).read_bytes()),
            )
            self.assertEqual(
                done["summary_sha256"],
                _sha256((output_dir / SUMMARY_NAME).read_bytes()),
            )
            self.assertEqual(
                persisted_summary["verdict_counts"],
                {"static": 4, "valid_action": 6},
            )
            self.assertEqual(
                persisted_summary["family_counts"],
                {"jump": 5, "walk": 5},
            )
            self.assertEqual(
                persisted_summary["validation_source_counts"],
                {"observation:original": 10, "result:original": 10},
            )
            self.assertEqual(
                persisted_summary["repair_generation_counts"],
                {},
            )
            self.assertEqual(
                persisted_summary["sanitization_counts"],
                {},
            )
            self.assertFalse(summary["resume_verified"])

    def test_reports_deterministic_sanitization_without_hiding_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path, qwen_root = _materialize(root)
            shard = qwen_root / "shards" / "qwen-000.jsonl"
            rows = _read_jsonl(shard)
            rows[0]["result_validated_from"] = "original_sanitized"
            rows[0]["alignment_repairs"] = [
                {
                    "attempt": 0,
                    "kind": "deterministic_original_sanitization",
                    "authoritative_context_digest": _object_digest(
                        rows[0]["observation"]
                    ),
                    "status": "ok",
                    "repair_generation_called": False,
                    "repair_sanitizations": [
                        {
                            "action": (
                                "downgrade_instruction_mismatch_to_static"
                            )
                        }
                    ],
                }
            ]
            _write_jsonl(shard, rows)
            summary = merge_qwen_shards(
                input_path=input_path,
                qwen_root=qwen_root,
                output_dir=root / "fused",
            )
            self.assertEqual(
                summary["validation_source_counts"][
                    "result:original_sanitized"
                ],
                1,
            )
            self.assertEqual(
                summary["repair_generation_counts"][
                    "alignment:deterministic"
                ],
                1,
            )
            self.assertEqual(
                summary["sanitization_counts"][
                    "alignment:"
                    "downgrade_instruction_mismatch_to_static"
                ],
                1,
            )

    def test_rejects_partition_and_marker_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path, qwen_root = _materialize(root)
            manifest = qwen_root / "manifests" / "shard-000.jsonl"
            original = manifest.read_bytes()
            manifest.write_bytes(
                b"".join(reversed(original.splitlines(keepends=True)))
            )
            with self.assertRaisesRegex(ValueError, "line-modulo"):
                merge_qwen_shards(
                    input_path=input_path,
                    qwen_root=qwen_root,
                    output_dir=root / "bad-partition",
                )
            manifest.write_bytes(original)
            marker = Path(f"{manifest}.source")
            value = json.loads(marker.read_text(encoding="utf-8"))
            value["source_rows"] += 1
            marker.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "marker v2"):
                merge_qwen_shards(
                    input_path=input_path,
                    qwen_root=qwen_root,
                    output_dir=root / "bad-marker",
                )

    def test_rejects_iid_digest_coverage_and_visual_schema_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path, qwen_root = _materialize(root)
            output = qwen_root / "shards" / "qwen-000.jsonl"
            rows = _read_jsonl(output)
            rows[0]["input_digest"] = "f" * 64
            _write_jsonl(output, rows)
            with self.assertRaisesRegex(ValueError, "input_digest mismatch"):
                merge_qwen_shards(
                    input_path=input_path,
                    qwen_root=qwen_root,
                    output_dir=root / "bad-digest",
                )

            input_path, qwen_root = _materialize(root / "schema")
            output = qwen_root / "shards" / "qwen-000.jsonl"
            rows = _read_jsonl(output)
            rows[0]["result"]["verdict"] = "invented"
            rows[0]["result_digest"] = _object_digest(rows[0]["result"])
            _write_jsonl(output, rows)
            with self.assertRaisesRegex(ValueError, "schema validation"):
                merge_qwen_shards(
                    input_path=input_path,
                    qwen_root=qwen_root,
                    output_dir=root / "bad-schema",
                )

            input_path, qwen_root = _materialize(root / "coverage")
            output = qwen_root / "shards" / "qwen-000.jsonl"
            rows = _read_jsonl(output)
            _write_jsonl(output, rows[:-1])
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                merge_qwen_shards(
                    input_path=input_path,
                    qwen_root=qwen_root,
                    output_dir=root / "bad-coverage",
                )

    def test_refuses_reuse_and_resume_is_exact_verification_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path, qwen_root = _materialize(root)
            output_dir = root / "fused"
            merge_qwen_shards(
                input_path=input_path,
                qwen_root=qwen_root,
                output_dir=output_dir,
            )
            with self.assertRaisesRegex(FileExistsError, "exists"):
                merge_qwen_shards(
                    input_path=input_path,
                    qwen_root=qwen_root,
                    output_dir=output_dir,
                )
            summary = merge_qwen_shards(
                input_path=input_path,
                qwen_root=qwen_root,
                output_dir=output_dir,
                resume=True,
            )
            self.assertTrue(summary["resume_verified"])
            with (output_dir / FUSED_NAME).open("ab") as handle:
                handle.write(b" ")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                merge_qwen_shards(
                    input_path=input_path,
                    qwen_root=qwen_root,
                    output_dir=output_dir,
                    resume=True,
                )

    def test_resume_requires_existing_output_before_input_or_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path, qwen_root = _materialize(root)
            output_dir = root / "missing-fused"

            # Both upstream inputs are deliberately unusable.  A strict
            # verification-only resume must reject the missing commit before
            # reading either of them and must not recreate any output.
            input_path.write_bytes(b"{not-json}\n")
            (qwen_root / "shards" / "qwen-000.jsonl").unlink()
            with self.assertRaisesRegex(
                FileNotFoundError,
                "--resume is verification-only.*existing output directory",
            ):
                merge_qwen_shards(
                    input_path=input_path,
                    qwen_root=qwen_root,
                    output_dir=output_dir,
                    resume=True,
                )

            self.assertFalse(output_dir.exists())
            self.assertEqual(
                [],
                list(root.glob(".missing-fused.*.tmp")),
            )

    def test_requires_exactly_eight_shards_and_does_not_overwrite_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path, qwen_root = _materialize(
                root,
                existing_visual=True,
            )
            with self.assertRaisesRegex(ValueError, "exactly 8"):
                merge_qwen_shards(
                    input_path=input_path,
                    qwen_root=qwen_root,
                    output_dir=root / "wrong-count",
                    shard_count=4,
                )
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                merge_qwen_shards(
                    input_path=input_path,
                    qwen_root=qwen_root,
                    output_dir=root / "existing-evidence",
                )


if __name__ == "__main__":
    unittest.main()
