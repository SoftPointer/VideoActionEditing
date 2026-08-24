from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np

from motive import r7_visual_graph_input as graph_input
from motive.r7_preflight_extract import DINO_DIM, DINO_FRAMES
from motive.r7_visual_candidate_manifest import (
    ROW_SCHEMA as CANDIDATE_ROW_SCHEMA,
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dino_provenance() -> dict[str, Any]:
    return {
        "encoder_id": "facebook/dinov2-base",
        "encoder_revision": "a" * 40,
        "resolved_path": "/models/dinov2-base",
        "model_tree_sha256": "b" * 64,
        "weights_sha256": "c" * 64,
        "model_file_count": 14,
        "frame_sampling_version": "uniform-6-from-uniform-32-v1",
        "preprocessing_version":
            "transformers-auto-image-processor-local-v1",
        "pooling": "last-hidden-state-cls-token-v1",
        "embedding_dim": DINO_DIM,
        "dtype": "float32",
        "normalization": "l2-per-frame",
        "frozen_encoder": True,
        "local_files_only": True,
    }


def _feature(count: int, offset: int) -> np.ndarray:
    result = np.zeros((count, DINO_FRAMES, DINO_DIM), dtype=np.float32)
    for row in range(count):
        for frame in range(DINO_FRAMES):
            result[row, frame, (offset + row + frame) % DINO_DIM] = 1.0
    return result


def _candidate_row(iid: str, cohort: str) -> dict[str, Any]:
    token = hashlib.sha256(iid.encode("utf-8")).hexdigest()
    return {
        "schema_version": CANDIDATE_ROW_SCHEMA,
        "iid": iid,
        "input_digest": token,
        "prompt": f"edit {iid}",
        "src_video": f"{iid}/source.mp4",
        "tgt_video": f"{iid}/target.mp4",
        "cohort": cohort,
        "primary_family": "locomotion",
        "source_row_sha256": token,
        "source_artifact_digest": "d" * 64,
        "split_assigned": False,
        "human_label": False,
        "training_eligible": False,
    }


def _anchor_row(iid: str, label: str = "positive") -> dict[str, Any]:
    return {
        "iid": iid,
        "src_video": f"{iid}/source.mp4",
        "tgt_video": f"{iid}/target.mp4",
        "prompt": f"edit {iid}",
        "input_digest": f"digest-{iid}",
        "r5_pilot_label": {
            "class": label,
            "action_signature": "walking",
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(_canonical(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_source_tree(root: Path, label: str) -> Path:
    final = root / label / "final"
    final.mkdir(parents=True)
    for name in graph_input.SOURCE_ARTIFACT_NAMES:
        (final / name).write_bytes(f"{label}:final:{name}\n".encode())
    shards = final.parent / "shards"
    for rank in range(8):
        shard = shards / f"rank-{rank:03d}-of-008"
        shard.mkdir(parents=True)
        for name in graph_input.SOURCE_ARTIFACT_NAMES:
            (shard / name).write_bytes(
                f"{label}:shard:{rank}:{name}\n".encode()
            )
    return final


def _candidate_result(
    rows: list[dict[str, Any]],
    *,
    valid: bool = True,
    dino: dict[str, Any] | None = None,
) -> dict[str, Any]:
    count = len(rows)
    hashes = np.full((count, DINO_FRAMES), "0123456789abcdef")
    video_hashes = np.asarray(
        [
            hashlib.sha256(f"candidate-source-{index}".encode()).hexdigest()
            for index in range(count)
        ],
        dtype="<U64",
    )
    target_video_hashes = np.asarray(
        [
            hashlib.sha256(f"candidate-target-{index}".encode()).hexdigest()
            for index in range(count)
        ],
        dtype="<U64",
    )
    feature_rows = []
    for index, row in enumerate(rows):
        sides = {}
        for role, digest in (
            ("source", str(video_hashes[index])),
            ("target", str(target_video_hashes[index])),
        ):
            sides[role] = {
                "valid": valid,
                "video_sha256": digest,
                "decode": {
                    "difference_hashes":
                        hashes[index].astype(str).tolist(),
                },
            }
        feature_rows.append(
            {
                "input_index": index,
                "iid": row["iid"],
                "input_row_sha256": graph_input._object_digest(row),
                **sides,
            }
        )
    arrays = {
        "input_indices": np.arange(count, dtype=np.int64),
        "source_valid": np.full(count, valid, dtype=np.bool_),
        "target_valid": np.full(count, valid, dtype=np.bool_),
        "source_dino_cls": _feature(count, 0),
        "target_dino_cls": _feature(count, 20),
        "source_difference_hashes": hashes.copy(),
        "target_difference_hashes": hashes.copy(),
        "source_video_sha256": video_hashes,
        "target_video_sha256": target_video_hashes,
    }
    return {
        "summary": {
            "common_contract": {
                "dino": dict(dino or _dino_provenance()),
            },
        },
        "rows": feature_rows,
        "arrays": arrays,
        "source_shards": [{} for _ in range(8)],
    }


def _anchor_results(
    rows: list[dict[str, Any]],
    *,
    final: Path,
    input_manifest: Path,
    dino: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    count = len(rows)
    hashes = np.full((count, DINO_FRAMES), "fedcba9876543210")
    feature_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        sides = {}
        for role in ("source", "target"):
            sides[role] = {
                "dino_valid": True,
                "video_sha256": hashlib.sha256(
                    f"anchor-{role}-{index}".encode()
                ).hexdigest(),
                "decode": {
                    "sampling_version": "uniform-32-decoded-frames-v1",
                    "decoded_frames": 32,
                    "dino_frame_offsets": [0, 6, 12, 19, 25, 31],
                    "perceptual_hashes":
                        hashes[index].astype(str).tolist(),
                },
            }
        feature_rows.append(
            {
                "input_index": index,
                "merged_array_index": index,
                "iid": row["iid"],
                "input_row_sha256": graph_input._object_digest(row),
                **sides,
            }
        )
    arrays = {
        "input_indices": np.arange(count, dtype=np.int64),
        "source_dino_valid": np.ones(count, dtype=np.bool_),
        "target_dino_valid": np.ones(count, dtype=np.bool_),
        "source_dino_cls": _feature(count, 40),
        "target_dino_cls": _feature(count, 60),
        "source_perceptual_hashes": hashes.copy(),
        "target_perceptual_hashes": hashes.copy(),
    }
    final_result = {
        "summary": {
            "archive_sha256": _sha(final / graph_input.ARCHIVE_NAME),
            "manifest_sha256": _sha(final / graph_input.MANIFEST_NAME),
            "shard_done_sha256": [
                _sha(
                    final.parent
                    / "shards"
                    / f"rank-{rank:03d}-of-008"
                    / graph_input.DONE_NAME
                )
                for rank in range(8)
            ],
        },
        "rows": feature_rows,
        "arrays": arrays,
    }
    shards: list[dict[str, Any]] = []
    input_sha = _sha(input_manifest)
    for rank in range(8):
        indices = [index for index in range(count) if index % 8 == rank]
        shard_rows = []
        for index in indices:
            row = dict(feature_rows[index])
            row.pop("merged_array_index")
            shard_rows.append(row)
        shard_arrays = {
            name: np.asarray(value)[indices]
            for name, value in arrays.items()
        }
        shards.append(
            {
                "contract": {
                    "rank": rank,
                    "world_size": 8,
                    "device": f"cuda:{rank}",
                    "input_manifest_sha256": input_sha,
                    "video_sampling": {
                        "version": "uniform-32-decoded-frames-v1",
                        "frames": 32,
                        "maximum_side": 384,
                    },
                    "dino": dict(dino or _dino_provenance()),
                },
                "rows": shard_rows,
                "arrays": shard_arrays,
            }
        )
    return final_result, shards


class Fixture:
    def __init__(
        self,
        root: Path,
        *,
        candidate_iids: tuple[str, ...] = ("zeta", "alpha"),
        anchor_iids: tuple[str, ...] = ("middle",),
        candidate_valid: bool = True,
        candidate_dino: dict[str, Any] | None = None,
        anchor_dino: dict[str, Any] | None = None,
    ) -> None:
        self.candidate_rows = [
            _candidate_row(
                iid,
                "pseudo_positive" if index % 2 == 0
                else "pseudo_negative",
            )
            for index, iid in enumerate(candidate_iids)
        ]
        self.anchor_rows = [_anchor_row(iid) for iid in anchor_iids]
        self.candidate_manifest = root / "candidates.jsonl"
        self.anchor_manifest = root / "anchor.jsonl"
        _write_jsonl(self.candidate_manifest, self.candidate_rows)
        _write_jsonl(self.anchor_manifest, self.anchor_rows)
        self.candidate_final = _make_source_tree(root, "candidate")
        self.anchor_final = _make_source_tree(root, "anchor")
        self.candidate_result = _candidate_result(
            self.candidate_rows,
            valid=candidate_valid,
            dino=candidate_dino,
        )
        self.anchor_result, self.anchor_shards = _anchor_results(
            self.anchor_rows,
            final=self.anchor_final,
            input_manifest=self.anchor_manifest,
            dino=anchor_dino,
        )

    @contextmanager
    def patched(self) -> Iterator[None]:
        def shard_result(path: Path, **_: Any) -> dict[str, Any]:
            rank = int(path.name.split("-")[1])
            return self.anchor_shards[rank]

        with (
            patch(
                "motive.r7_visual_graph_input."
                "expansion_features.validate_final",
                return_value=self.candidate_result,
            ) as candidate_validator,
            patch(
                "motive.r7_visual_graph_input.preflight.validate_final",
                return_value=self.anchor_result,
            ) as anchor_validator,
            patch(
                "motive.r7_visual_graph_input.preflight.validate_shard",
                side_effect=shard_result,
            ) as shard_validator,
        ):
            yield
            self.candidate_validator = candidate_validator
            self.anchor_validator = anchor_validator
            self.shard_validator = shard_validator

    def build(self, output: Path, *, resume: bool = False) -> dict[str, Any]:
        return graph_input.build_graph_input(
            candidate_features_dir=self.candidate_final,
            candidate_manifest=self.candidate_manifest,
            anchor_features_dir=self.anchor_final,
            anchor_input_manifest=self.anchor_manifest,
            output_dir=output,
            resume=resume,
        )


class VisualGraphInputTests(unittest.TestCase):
    def test_merge_is_paired_canonical_and_has_no_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            output = root / "graph"
            with fixture.patched():
                done = fixture.build(output)
            self.assertEqual(fixture.candidate_validator.call_count, 1)
            self.assertEqual(fixture.anchor_validator.call_count, 1)
            self.assertEqual(fixture.shard_validator.call_count, 8)
            self.assertEqual(set(path.name for path in output.iterdir()), {
                "manifest.jsonl",
                "features.npz",
                "summary.json",
                "done.json",
            })
            rows = [
                json.loads(line)
                for line in (output / "manifest.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [(row["iid"], row["role"]) for row in rows],
                [
                    ("alpha", "source"),
                    ("alpha", "target"),
                    ("middle", "source"),
                    ("middle", "target"),
                    ("zeta", "source"),
                    ("zeta", "target"),
                ],
            )
            self.assertEqual(
                [row["asset_index"] for row in rows],
                list(range(6)),
            )
            self.assertEqual(
                [row["anchor"] for row in rows],
                [False, False, True, True, False, False],
            )
            with np.load(output / "features.npz", allow_pickle=False) as data:
                self.assertEqual(set(data.files), {
                    "asset_indices",
                    "dino_cls",
                })
                self.assertEqual(data["dino_cls"].shape, (6, 6, 768))
                self.assertEqual(data["dino_cls"].dtype, np.float32)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["candidate_iids"]["count"], 2)
            self.assertEqual(summary["anchor_iids"]["count"], 1)
            self.assertFalse(summary["split_assigned"])
            self.assertFalse(summary["human_labels_asserted"])
            self.assertFalse(summary["training_authorized"])
            self.assertFalse(done["training_authorized"])

    def test_commit_validator_is_self_contained_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            output = root / "graph"
            with fixture.patched():
                fixture.build(output)
            committed = graph_input.validate_graph_input_commit(output)
            self.assertEqual(
                set(committed["artifact_hashes"]),
                {"manifest", "archive", "summary", "done"},
            )
            self.assertEqual(
                committed["artifact_digest"],
                committed["done"]["artifact_digest"],
            )
            self.assertEqual(
                set(committed["dino_contract"]),
                set(graph_input.DINO_COMPARISON_FIELDS),
            )
            self.assertEqual(
                committed["arrays"]["dino_cls"].shape,
                (6, DINO_FRAMES, DINO_DIM),
            )
            archive = output / graph_input.ARCHIVE_NAME
            payload = bytearray(archive.read_bytes())
            payload[-1] ^= 1
            archive.write_bytes(bytes(payload))
            with self.assertRaisesRegex(ValueError, "archive digest differs"):
                graph_input.validate_graph_input_commit(output)

    def test_resume_only_validates_and_nonresume_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            output = root / "graph"
            with fixture.patched():
                fixture.build(output)
                before = {
                    path.name: (_sha(path), path.stat().st_mtime_ns)
                    for path in output.iterdir()
                }
                with self.assertRaises(FileExistsError):
                    fixture.build(output)
                resumed = fixture.build(output, resume=True)
            after = {
                path.name: (_sha(path), path.stat().st_mtime_ns)
                for path in output.iterdir()
            }
            self.assertEqual(before, after)
            self.assertEqual(resumed["status"], "complete")

    def test_dino_contract_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mismatch = _dino_provenance()
            mismatch["weights_sha256"] = "e" * 64
            fixture = Fixture(root, anchor_dino=mismatch)
            output = root / "graph"
            with fixture.patched(), self.assertRaisesRegex(
                ValueError,
                "DINO contracts differ.*weights_sha256",
            ):
                fixture.build(output)
            self.assertFalse(output.exists())

    def test_invalid_side_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root, candidate_valid=False)
            output = root / "graph"
            with fixture.patched(), self.assertRaisesRegex(
                ValueError,
                "not DINO-valid",
            ):
                fixture.build(output)
            self.assertFalse(output.exists())

    def test_duplicate_candidate_anchor_iid_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(
                root,
                candidate_iids=("same",),
                anchor_iids=("same",),
            )
            output = root / "graph"
            with fixture.patched(), self.assertRaisesRegex(
                ValueError,
                "IID sets overlap",
            ):
                fixture.build(output)
            self.assertFalse(output.exists())

    def test_tamper_is_rejected_by_verification_only_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            output = root / "graph"
            with fixture.patched():
                fixture.build(output)
            with (output / "manifest.jsonl").open("ab") as handle:
                handle.write(b"{}\n")
            with fixture.patched(), self.assertRaisesRegex(
                ValueError,
                "digest differs",
            ):
                fixture.build(output, resume=True)

    def test_anchor_final_must_be_bound_to_exact_shard_done_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            fixture.anchor_result["summary"]["shard_done_sha256"][3] = (
                "0" * 64
            )
            output = root / "graph"
            with fixture.patched(), self.assertRaisesRegex(
                ValueError,
                "not bound to the exact source shards",
            ):
                fixture.build(output)


if __name__ == "__main__":
    unittest.main()
