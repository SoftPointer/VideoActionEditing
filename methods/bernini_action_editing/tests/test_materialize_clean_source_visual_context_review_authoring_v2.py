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
TOOLS_ROOT = METHOD_ROOT / "tools"
for path in (METHOD_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import clean_source_visual_context_checkpoint_review_contract_v1 as contract  # noqa: E402
import clean_source_visual_context_training_v1 as source_data  # noqa: E402
import materialize_clean_source_visual_context_review_authoring_v2 as target  # noqa: E402


@dataclass(frozen=True)
class _Row:
    iid: str
    split: str
    group_id: str
    action_family: str
    source_video_sha256: str


class _Manifest:
    def __init__(self, rows: list[_Row]) -> None:
        self.rows = tuple(rows)
        self.manifest_digest = "d" * 64

    def rows_for_split(self, split: str):
        return tuple(row for row in self.rows if row.split == split)


def _fixture(root: Path):
    manifest_path = root / "source-only-v3.json"
    manifest_path.write_text("{}\n", encoding="ascii")
    raw_path = root / "raw.parquet"
    raw_path.write_bytes(b"raw-safe-projection-test")
    identities: dict[str, dict] = {}
    split_rows: list[_Row] = []
    raw_rows = []
    for index, sentinel_id in enumerate(contract.SENTINEL_ORDER):
        fixed = dict(contract.SENTINEL_IDENTITIES[sentinel_id])
        source = root / f"raw-{sentinel_id}.mp4"
        source.write_bytes(f"source-{sentinel_id}".encode())
        source_sha = contract.file_sha256(source)
        forward = f"Pinned raw forward instruction for {sentinel_id}."
        forward_sha = hashlib.sha256(forward.encode()).hexdigest()
        fixed["source_video_sha256"] = source_sha
        fixed["forward_instruction_sha256"] = forward_sha
        identities[sentinel_id] = fixed
        group_id = f"group-{index}"
        split_rows.append(
            _Row(
                iid=fixed["iid"],
                split="heldout",
                group_id=group_id,
                action_family=fixed["action_family"],
                source_video_sha256=source_sha,
            )
        )
        raw_rows.append(
            {
                "iid": fixed["iid"],
                "group_id": group_id,
                "family": fixed["action_family"],
                "inputs": json.dumps(
                    [
                        {"has_loss": 0, "type": "video"},
                        {"has_loss": 0, "text": forward, "type": "text"},
                        {"has_loss": 1, "type": "video_gen"},
                    ]
                ),
                "source_video_path": str(source),
                "source_video_declared_path": str(source),
                "source_video_sha256": source_sha,
                "edit_instruction_sha256": forward_sha,
                "selection_gates_json": json.dumps(
                    {"single_dynamic_actor": True}, sort_keys=True
                ),
                "strict_selection_gates_all_true": True,
            }
        )
    for index in range(4):
        split_rows.append(
            _Row(
                iid=f"extra-heldout-{index}",
                split="heldout",
                group_id=f"extra-group-{index}",
                action_family=f"extra-{index}",
                source_video_sha256=hashlib.sha256(f"extra-{index}".encode()).hexdigest(),
            )
        )
    return manifest_path, raw_path, _Manifest(split_rows), raw_rows, identities


class AuthoringMaterializerV2Tests(unittest.TestCase):
    def _patches(self, raw_path, manifest, raw_rows, identities):
        return (
            mock.patch.object(source_data, "PINNED_RAW_PARQUET", raw_path),
            mock.patch.object(
                source_data, "PINNED_RAW_PARQUET_SHA256", contract.file_sha256(raw_path)
            ),
            mock.patch.object(source_data, "FULL644_ROWS", 4),
            mock.patch.object(
                source_data, "load_source_only_split_manifest", return_value=manifest
            ),
            mock.patch.object(target, "_read_raw_rows", return_value=raw_rows),
            mock.patch.object(contract, "SENTINEL_IDENTITIES", identities),
            mock.patch.object(
                contract,
                "_ffprobe_exact81",
                return_value={"frame_count": 81, "fps": 25},
            ),
        )

    def test_materializes_real_diverse_four_and_exact_six_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest_path, raw_path, manifest, raw_rows, identities = _fixture(root)
            patches = self._patches(raw_path, manifest, raw_rows, identities)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
                output, value = target.materialize_authoring(
                    source_only_manifest_path=manifest_path,
                    expected_source_only_file_sha256=contract.file_sha256(manifest_path),
                    output_dir=root / "frozen-authoring",
                )
                self.assertEqual(output.name, "checkpoint_review_authoring_v2.json")
                self.assertEqual(
                    [row["sentinel_id"] for row in value["sentinels"]],
                    list(contract.SENTINEL_ORDER),
                )
                self.assertEqual(
                    {row["diversity_role"] for row in value["sentinels"]},
                    {"animal", "human", "hand-object-interaction", "physical-emitter"},
                )
                self.assertTrue(
                    all(set(row["instructions"]) == set(contract.TEXT_BRANCHES) for row in value["sentinels"])
                )
                self.assertEqual(len(list((output.parent / "sources").glob("*.mp4"))), 4)
                self.assertFalse(value["raw_full644"]["target_video_bytes_read"])
                self.assertFalse(value["raw_full644"]["target_video_copied"])

    def test_raw_projection_has_no_video_or_target_column(self) -> None:
        self.assertNotIn("videos", target.RAW_SAFE_COLUMNS)
        self.assertFalse(any("target" in column for column in target.RAW_SAFE_COLUMNS))
        source = Path(target.__file__).read_text(encoding="utf-8")
        self.assertIn("columns=list(RAW_SAFE_COLUMNS)", source)

    def test_rejects_manifest_file_sha_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest_path, raw_path, manifest, raw_rows, identities = _fixture(root)
            with self.assertRaisesRegex(
                target.ReviewAuthoringMaterializationError, "caller pin"
            ):
                target.materialize_authoring(
                    source_only_manifest_path=manifest_path,
                    expected_source_only_file_sha256="0" * 64,
                    output_dir=root / "must-not-exist",
                )
            self.assertFalse((root / "must-not-exist").exists())

    def test_rejects_forward_text_not_bound_to_pinned_raw_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest_path, raw_path, manifest, raw_rows, identities = _fixture(root)
            raw_rows[0] = {**raw_rows[0], "edit_instruction_sha256": "0" * 64}
            patches = self._patches(raw_path, manifest, raw_rows, identities)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
                with self.assertRaisesRegex(
                    target.ReviewAuthoringMaterializationError,
                    "forward instruction SHA",
                ):
                    target.materialize_authoring(
                        source_only_manifest_path=manifest_path,
                        expected_source_only_file_sha256=contract.file_sha256(manifest_path),
                        output_dir=root / "must-not-exist",
                    )
            self.assertFalse((root / "must-not-exist").exists())


if __name__ == "__main__":
    unittest.main()
