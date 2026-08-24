#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import clean_source_visual_context_checkpoint_review_contract_v1 as contract  # noqa: E402


@dataclass(frozen=True)
class _Row:
    iid: str
    split: str
    action_family: str
    source_video_sha256: str
    source_posterior_path: str
    source_posterior_file_sha256: str


class _Manifest:
    def __init__(self, rows: list[_Row], digest: str) -> None:
        self.rows = tuple(rows)
        self.manifest_digest = digest

    def rows_for_split(self, split: str):
        return tuple(row for row in self.rows if row.split == split)


def _fixture(root: Path) -> tuple[Path, Path, _Manifest, dict[str, dict]]:
    digest = "a" * 64
    rows = []
    authored = []
    identities: dict[str, dict] = {}
    for index, sentinel in enumerate(contract.SENTINEL_ORDER):
        fixed = dict(contract.SENTINEL_IDENTITIES[sentinel])
        iid = fixed["iid"]
        video = root / f"{sentinel}.mp4"
        payload = f"raw-source-{sentinel}".encode()
        video.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        instructions = {
            branch: f"Full {branch} instruction for {sentinel}."
            for branch in contract.TEXT_BRANCHES
        }
        fixed["source_video_sha256"] = sha
        fixed["forward_instruction_sha256"] = hashlib.sha256(
            instructions["forward"].encode("utf-8")
        ).hexdigest()
        identities[sentinel] = fixed
        rows.append(
            _Row(
                iid=iid,
                split="heldout",
                action_family=fixed["action_family"],
                source_video_sha256=sha,
                source_posterior_path=str(root / f"{iid}.source-posterior-index0.pt"),
                source_posterior_file_sha256=hashlib.sha256(f"posterior-{iid}".encode()).hexdigest(),
            )
        )
        authored.append(
            {
                "sentinel_id": sentinel,
                "diversity_role": fixed["diversity_role"],
                "source_entity_type": fixed["source_entity_type"],
                "iid": iid,
                "action_family": fixed["action_family"],
                "source_video": str(video),
                "source_video_sha256": sha,
                "source_caption": f"Complete source description for {sentinel}.",
                "seed": fixed["seed"],
                "wrong_owner_iid": fixed["wrong_owner_iid"],
                "latent_shape": fixed["latent_shape"],
                "instructions": instructions,
            }
        )
    source_path = root / "source-only.json"
    source_path.write_text("{}\n", encoding="ascii")
    unsigned = {
        "schema_version": contract.AUTHORING_SCHEMA,
        "authoring_id": "fixed-four-unit-test",
        "source_only_manifest": {
            "path": str(source_path),
            "file_sha256": contract.file_sha256(source_path),
            "manifest_digest": digest,
            "selected_split": "heldout",
        },
        "raw_full644": {
            "path": str(contract.source_data.PINNED_RAW_PARQUET),
            "file_sha256": contract.source_data.PINNED_RAW_PARQUET_SHA256,
            "safe_columns_read": [
                "iid", "group_id", "family", "inputs", "source_video_path",
                "source_video_declared_path", "source_video_sha256",
                "edit_instruction_sha256", "selection_gates_json",
                "strict_selection_gates_all_true",
            ],
            "videos_column_read": False,
            "target_video_path_read": False,
            "target_video_bytes_read": False,
            "target_video_copied": False,
            "synthetic_target_semantics_used": False,
        },
        "sentinels": authored,
        "authority": {
            "fixed_before_checkpoint_decode": True,
            "quality_based_selection": False,
            "optimizer_access": False,
            "sentinel_rule": "fixed-actual-v3-heldout-diversity-four-v1",
            "forward_instruction_authority": "pinned-raw-full644-inputs-text",
            "typed_controls_manually_preregistered": True,
            "target_video_available_to_review": False,
        },
    }
    authoring = {**unsigned, "authoring_digest": contract.object_sha256(unsigned)}
    author_path = root / "authoring.json"
    author_path.write_text(json.dumps(authoring, sort_keys=True), encoding="utf-8")
    return source_path, author_path, _Manifest(rows, digest), identities


def _resign_authoring(path: Path) -> None:
    value = json.loads(path.read_text())
    value.pop("authoring_digest", None)
    value["authoring_digest"] = contract.object_sha256(value)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class ManifestMaterializerTests(unittest.TestCase):
    def test_binds_exact_four_to_real_heldout_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source_path, author_path, manifest, identities = _fixture(root)
            with mock.patch.object(
                contract.source_data,
                "load_source_only_split_manifest",
                return_value=manifest,
            ), mock.patch.object(contract, "SENTINEL_IDENTITIES", identities):
                value = contract.materialize_manifest_value(
                    source_only_manifest_path=source_path,
                    authoring_path=author_path,
                    verify_files=False,
                    verify_source_media=False,
                )
                self.assertEqual(value["sentinel_order"], list(contract.SENTINEL_ORDER))
                self.assertEqual(value["source_only_manifest"]["selected_split"], "heldout")
                self.assertEqual(value["source_only_manifest"]["train_overlap_count"], 0)
                self.assertEqual(len(value["sentinels"]), 4)
                self.assertEqual(value["checkpoint_steps"], [0, 20, 40, 60, 80])
                output = root / "manifest.json"
                contract.write_create_only_json(output, value)
                loaded = contract.load_manifest(output, verify_files=False)
                self.assertEqual(loaded["manifest_digest"], value["manifest_digest"])

    def test_rejects_historical_example_not_in_stage_b_heldout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source_path, author_path, manifest, identities = _fixture(root)
            moved = list(manifest.rows)
            moved[0] = _Row(**{**moved[0].__dict__, "split": "train"})
            nonheldout = _Manifest(moved, manifest.manifest_digest)
            with mock.patch.object(
                contract.source_data,
                "load_source_only_split_manifest",
                return_value=nonheldout,
            ), mock.patch.object(contract, "SENTINEL_IDENTITIES", identities):
                with self.assertRaisesRegex(
                    contract.CheckpointReviewContractError, "not in.*heldout"
                ):
                    contract.materialize_manifest_value(
                        source_only_manifest_path=source_path,
                        authoring_path=author_path,
                        verify_files=False,
                        verify_source_media=False,
                    )

    def test_rejects_unregistered_wrong_owner_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source_path, author_path, manifest, identities = _fixture(root)
            authoring = json.loads(author_path.read_text())
            authoring["sentinels"][0]["wrong_owner_iid"] = authoring["sentinels"][3]["iid"]
            author_path.write_text(json.dumps(authoring, sort_keys=True), encoding="utf-8")
            _resign_authoring(author_path)
            with mock.patch.object(
                contract.source_data,
                "load_source_only_split_manifest",
                return_value=manifest,
            ), mock.patch.object(contract, "SENTINEL_IDENTITIES", identities):
                with self.assertRaisesRegex(
                    contract.CheckpointReviewContractError, "fixed held-out identity"
                ):
                    contract.materialize_manifest_value(
                        source_only_manifest_path=source_path,
                        authoring_path=author_path,
                        verify_files=False,
                        verify_source_media=False,
                    )

    def test_rejects_incomplete_instruction_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source_path, author_path, manifest, identities = _fixture(root)
            authoring = json.loads(author_path.read_text())
            del authoring["sentinels"][3]["instructions"]["camera-only"]
            author_path.write_text(json.dumps(authoring, sort_keys=True), encoding="utf-8")
            _resign_authoring(author_path)
            with mock.patch.object(
                contract.source_data,
                "load_source_only_split_manifest",
                return_value=manifest,
            ), mock.patch.object(contract, "SENTINEL_IDENTITIES", identities):
                with self.assertRaisesRegex(
                    contract.CheckpointReviewContractError, "instruction closure"
                ):
                    contract.materialize_manifest_value(
                        source_only_manifest_path=source_path,
                        authoring_path=author_path,
                        verify_files=False,
                        verify_source_media=False,
                    )


if __name__ == "__main__":
    unittest.main()
