from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from motive.r7_expansion_visual_features import (
    FINAL_WORLD_SIZE,
    extract_rank,
    finalize_shards,
    load_candidate_manifest,
    rank_directory,
    validate_final,
    validate_shard,
)
from motive.r7_preflight_extract import (
    DINO_DIM,
    DINO_FRAMES,
    VIDEO_FRAMES,
    DecodedVideo,
    GlobalExtractionError,
    PerVideoError,
)
from motive.r7_visual_candidate_manifest import (
    ROW_SCHEMA as CANDIDATE_ROW_SCHEMA,
)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _candidate(index: int) -> dict[str, object]:
    iid = f"candidate-{index:03d}"
    return {
        "schema_version": CANDIDATE_ROW_SCHEMA,
        "iid": iid,
        "input_digest": hashlib.sha256(
            f"input:{index}".encode()
        ).hexdigest(),
        "prompt": f"make action {index}",
        "src_video": f"videos/{iid}/source.mp4",
        "tgt_video": f"videos/{iid}/edited.mp4",
        "cohort": (
            "pseudo_positive" if index % 2 == 0 else "pseudo_negative"
        ),
        "primary_family": f"family-{index % 3}",
        "source_row_sha256": hashlib.sha256(
            f"source:{index}".encode()
        ).hexdigest(),
        "source_artifact_digest": "f" * 64,
        "split_assigned": False,
        "human_label": False,
        "training_eligible": False,
    }


def _write_manifest(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    path.write_text(
        "".join(_canonical(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_videos(
    data_root: Path,
    rows: list[dict[str, object]],
) -> None:
    for row in rows:
        for field in ("src_video", "tgt_video"):
            path = data_root / str(row[field])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                f"{row['iid']}:{field}".encode("utf-8")
            )


def _decoded(path: Path) -> DecodedVideo:
    if "candidate-005" in str(path) and path.name == "edited.mp4":
        raise PerVideoError(
            "video_decode_failed",
            f"synthetic failure: {path}",
        )
    # A horizontal gradient makes the dHash deterministic and non-empty.
    width = 10
    base = np.arange(width, dtype=np.uint8)[None, :, None]
    frame = np.broadcast_to(base, (8, width, 3)).copy()
    frames = np.stack(
        [np.roll(frame, index % width, axis=1) for index in range(VIDEO_FRAMES)]
    )
    indices = np.arange(VIDEO_FRAMES, dtype=np.int64)
    return DecodedVideo(
        frames_rgb=frames,
        frame_times=indices.astype(np.float64) / 24.0,
        source_frame_indices=indices,
        source_fps=24.0,
        source_frame_count=48,
        source_size=(8, width),
        resized_size=(8, width),
    )


class FakeEncoder:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail
        self.provenance = {
            "encoder_id": "facebook/dinov2-base",
            "encoder_revision": "a" * 40,
            "resolved_path": "/models/dinov2-base",
            "model_tree_sha256": "b" * 64,
            "weights_sha256": "c" * 64,
            "model_file_count": 3,
            "embedding_dim": DINO_DIM,
            "dtype": "float32",
            "normalization": "l2-per-frame",
            "frozen_encoder": True,
            "local_files_only": True,
            "frame_sampling_version": "uniform-6-from-uniform-32-v1",
            "preprocessing_version":
                "transformers-auto-image-processor-local-v1",
            "pooling": "last-hidden-state-cls-token-v1",
        }

    def encode(self, frames_rgb: object) -> np.ndarray:
        self.calls += 1
        if self.fail:
            raise GlobalExtractionError("synthetic global DINO failure")
        frames = np.asarray(frames_rgb)
        if frames.shape[0] != DINO_FRAMES:
            raise AssertionError("test received a non-six-frame sample")
        matrix = np.zeros((DINO_FRAMES, DINO_DIM), dtype=np.float32)
        matrix[np.arange(DINO_FRAMES), np.arange(DINO_FRAMES)] = 1.0
        return matrix


class ExpansionVisualFeatureTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        count: int = 11,
    ) -> tuple[Path, Path, Path, list[dict[str, object]]]:
        rows = [_candidate(index) for index in range(count)]
        manifest = root / "candidates.jsonl"
        data_root = root / "data"
        data_root.mkdir()
        _write_manifest(manifest, rows)
        _write_videos(data_root, rows)
        return manifest, data_root, root / "features", rows

    def _extract_all(
        self,
        *,
        manifest: Path,
        data_root: Path,
        output_root: Path,
    ) -> list[FakeEncoder]:
        encoders: list[FakeEncoder] = []
        with patch(
            "motive.r7_expansion_visual_features.decode_video_fixed_frames",
            side_effect=_decoded,
        ):
            for rank in range(FINAL_WORLD_SIZE):
                encoder = FakeEncoder()
                encoders.append(encoder)
                extract_rank(
                    input_manifest=manifest,
                    data_root=data_root,
                    output_root=output_root,
                    rank=rank,
                    world_size=FINAL_WORLD_SIZE,
                    local_rank=rank,
                    encoder=encoder,
                )
        return encoders

    def test_eight_rank_finalize_preserves_order_and_fails_split_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, data_root, output, input_rows = self._fixture(root)
            encoders = self._extract_all(
                manifest=manifest,
                data_root=data_root,
                output_root=output,
            )
            # One target decode failure skips exactly one encoder call.
            self.assertEqual(
                sum(encoder.calls for encoder in encoders),
                2 * len(input_rows) - 1,
            )
            done = finalize_shards(
                input_manifest=manifest,
                output_root=output,
            )
            self.assertFalse(done["split_ready"])
            final = validate_final(
                output / "final",
                input_manifest=manifest,
                output_root=output,
            )
            self.assertEqual(
                [row["iid"] for row in final["rows"]],
                [row["iid"] for row in input_rows],
            )
            self.assertEqual(
                final["arrays"]["input_indices"].tolist(),
                list(range(len(input_rows))),
            )
            summary = final["summary"]
            self.assertEqual(summary["extraction_status"], "failed")
            self.assertFalse(summary["split_ready"])
            self.assertEqual(summary["statistics"]["failed_sides"], 1)
            self.assertEqual(
                summary["statistics"]["failures"],
                {"target:video_decode_failed": 1},
            )
            self.assertFalse(summary["cotracker_executed"])
            failed = final["rows"][5]["target"]
            self.assertFalse(failed["valid"])
            self.assertEqual(failed["failure_stage"], "decode")
            self.assertEqual(
                final["arrays"]["target_dino_cls"][5].sum(),
                0.0,
            )

    def test_hash_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, data_root, output, _rows = self._fixture(
                root,
                count=8,
            )
            with patch(
                "motive.r7_expansion_visual_features."
                "decode_video_fixed_frames",
                side_effect=_decoded,
            ):
                extract_rank(
                    input_manifest=manifest,
                    data_root=data_root,
                    output_root=output,
                    rank=0,
                    world_size=FINAL_WORLD_SIZE,
                    local_rank=0,
                    encoder=FakeEncoder(),
                )
            shard = rank_directory(output, 0, FINAL_WORLD_SIZE)
            with (shard / "features.npz").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "artifact digest mismatch"):
                validate_shard(shard, input_manifest=manifest)

    def test_finalize_rejects_missing_rank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, data_root, output, _rows = self._fixture(
                root,
                count=8,
            )
            with patch(
                "motive.r7_expansion_visual_features."
                "decode_video_fixed_frames",
                side_effect=_decoded,
            ):
                for rank in range(FINAL_WORLD_SIZE - 1):
                    extract_rank(
                        input_manifest=manifest,
                        data_root=data_root,
                        output_root=output,
                        rank=rank,
                        world_size=FINAL_WORLD_SIZE,
                        local_rank=rank,
                        encoder=FakeEncoder(),
                    )
            with self.assertRaisesRegex(ValueError, "missing=.*rank-007"):
                finalize_shards(
                    input_manifest=manifest,
                    output_root=output,
                )
            self.assertFalse((output / "final").exists())

    def test_resume_only_validates_and_rejects_tampered_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, data_root, output, _rows = self._fixture(
                root,
                count=8,
            )
            encoder = FakeEncoder()
            with patch(
                "motive.r7_expansion_visual_features."
                "decode_video_fixed_frames",
                side_effect=_decoded,
            ):
                original = extract_rank(
                    input_manifest=manifest,
                    data_root=data_root,
                    output_root=output,
                    rank=0,
                    world_size=FINAL_WORLD_SIZE,
                    local_rank=0,
                    encoder=encoder,
                )
            self.assertEqual(encoder.calls, 2)
            bomb = FakeEncoder(fail=True)
            with patch(
                "motive.r7_expansion_visual_features."
                "decode_video_fixed_frames",
                side_effect=AssertionError("resume decoded a video"),
            ):
                resumed = extract_rank(
                    input_manifest=manifest,
                    data_root=data_root,
                    output_root=output,
                    rank=0,
                    world_size=FINAL_WORLD_SIZE,
                    local_rank=0,
                    encoder=bomb,
                    resume=True,
                )
            self.assertEqual(resumed, original)
            self.assertEqual(bomb.calls, 0)
            shard = rank_directory(output, 0, FINAL_WORLD_SIZE)
            with (shard / "summary.json").open("ab") as handle:
                handle.write(b" ")
            with self.assertRaisesRegex(ValueError, "artifact digest mismatch"):
                extract_rank(
                    input_manifest=manifest,
                    data_root=data_root,
                    output_root=output,
                    rank=0,
                    world_size=FINAL_WORLD_SIZE,
                    local_rank=0,
                    encoder=FakeEncoder(fail=True),
                    resume=True,
                )

    def test_global_dino_failure_publishes_no_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, data_root, output, _rows = self._fixture(
                root,
                count=8,
            )
            with patch(
                "motive.r7_expansion_visual_features."
                "decode_video_fixed_frames",
                side_effect=_decoded,
            ):
                with self.assertRaisesRegex(
                    GlobalExtractionError,
                    "synthetic global DINO failure",
                ):
                    extract_rank(
                        input_manifest=manifest,
                        data_root=data_root,
                        output_root=output,
                        rank=0,
                        world_size=FINAL_WORLD_SIZE,
                        local_rank=0,
                        encoder=FakeEncoder(fail=True),
                    )
            self.assertFalse(
                rank_directory(output, 0, FINAL_WORLD_SIZE).exists()
            )

    def test_manifest_uniqueness_and_path_containment_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.jsonl"
            rows = [_candidate(0), _candidate(0)]
            _write_manifest(duplicate, rows)
            with self.assertRaisesRegex(ValueError, "duplicates iid"):
                load_candidate_manifest(duplicate)

            manifest, data_root, output, valid_rows = self._fixture(
                root,
                count=8,
            )
            split_rows = [dict(row) for row in valid_rows]
            split_rows[0]["split_assigned"] = True
            _write_manifest(manifest, split_rows)
            with self.assertRaisesRegex(ValueError, "split_assigned"):
                load_candidate_manifest(manifest)

            valid_rows[0]["src_video"] = "../outside.mp4"
            _write_manifest(manifest, valid_rows)
            with self.assertRaisesRegex(ValueError, "escapes data_root"):
                extract_rank(
                    input_manifest=manifest,
                    data_root=data_root,
                    output_root=output,
                    rank=0,
                    world_size=FINAL_WORLD_SIZE,
                    local_rank=0,
                    encoder=FakeEncoder(),
                )
            self.assertFalse(
                rank_directory(output, 0, FINAL_WORLD_SIZE).exists()
            )


if __name__ == "__main__":
    unittest.main()
