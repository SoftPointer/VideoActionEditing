from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from motive import goku_atomic_source_high_recall as high_recall
from motive.goku_action_anchor_qwen import validate_input_row


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object) -> str:
    return _sha(_canonical(value).encode())


def _json_bytes(value: dict[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _png() -> bytes:
    ok, encoded = cv2.imencode(
        ".png", np.full((12, 16, 3), 127, dtype=np.uint8)
    )
    assert ok
    return encoded.tobytes()


def _row(
    root: Path,
    iid: str,
    family: str,
    score: float,
    reasons: list[str],
    *,
    group: str | None = None,
    entropy: float = 0.99,
) -> dict[str, object]:
    source = root / f"{iid}.mp4"
    source.write_bytes(f"video-{iid}".encode())
    stat = source.stat()
    return {
        "schema_version": "motive-goku-atomic-source-expand-v1",
        "iid": iid,
        "group_id": group or f"g-{iid}",
        "family": family,
        "src_video": source.name,
        "resolved_src_video": str(source),
        "source_caption": "A visible subject moves.",
        "edited_caption": "The visible subject waves.",
        "prompt": "Make the visible subject wave.",
        "mother_rank": 0,
        "source_video_sha256": _sha(source.read_bytes()),
        "prefilter_score": score,
        "eligible": not reasons,
        "rejection_reasons": reasons,
        "selected": False,
        "selection_rank": None,
        "within_family_rank": None,
        "anchor_image": None,
        "resolved_anchor_image": None,
        "anchor_sha256": None,
        "media": {
            "width": 1280,
            "height": 720,
            "short_side": 720,
            "pixels": 1280 * 720,
            "fps": 25.0,
            "frame_count": 81,
            "duration_seconds": 3.24,
            "file_size_bytes": stat.st_size,
            "mtime_ns_at_analysis": stat.st_mtime_ns,
        },
        "motion": {
            "label": "dynamic_object",
            "residual_speed_p90": 0.02,
            "active_pixel_fraction": 0.08,
            "active_frame_fraction": 0.8,
            "scene_cut_ratio": 0.0,
        },
        "actor_motion": {
            "spatial_energy_entropy": entropy,
            "actor_likeness": 0.7,
            "temporal_coverage": 0.8,
            "largest_component_share": 0.5,
        },
    }


def _replace_source_with_decodable_video(
    row: dict[str, object], *, value: int
) -> None:
    """Replace the lightweight fixture with a real source for subprocess tests."""

    source = Path(str(row["resolved_src_video"]))
    source.unlink()
    writer = cv2.VideoWriter(
        str(source),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (32, 24),
    )
    if not writer.isOpened():
        raise RuntimeError("test OpenCV could not create an mp4v fixture")
    try:
        for frame_index in range(4):
            frame = np.full(
                (24, 32, 3),
                (value + frame_index) % 255,
                dtype=np.uint8,
            )
            writer.write(frame)
    finally:
        writer.release()
    stat = source.stat()
    row["source_video_sha256"] = _sha(source.read_bytes())
    media = row["media"]
    assert isinstance(media, dict)
    media.update(
        {
            "width": 32,
            "height": 24,
            "file_size_bytes": stat.st_size,
            "mtime_ns_at_analysis": stat.st_mtime_ns,
        }
    )


def _upstream(root: Path, rows: list[dict[str, object]]) -> tuple[Path, str]:
    final = root / "source-final"
    final.mkdir()
    for rank, row in enumerate(rows, start=1):
        row["mother_rank"] = rank
    evaluated_raw = "".join(_canonical(row) + "\n" for row in rows).encode()
    (final / "evaluated.jsonl").write_bytes(evaluated_raw)
    inputs = {
        "binding_digest": "b" * 64,
        "old_selected_sha256": "c" * 64,
    }
    summary = {
        "status": "complete",
        "inputs": inputs,
        "config_digest": "d" * 64,
        "semantics": {
            "fresh_media_geometry_motion_analysis": True,
            "old_iid_and_group_exclusion": True,
        },
        "counts": {"mother": len(rows)},
    }
    summary_raw = _json_bytes(summary)
    (final / "summary.json").write_bytes(summary_raw)
    artifacts = {
        "evaluated.jsonl": _sha(evaluated_raw),
        "summary.json": _sha(summary_raw),
    }
    done = {
        "status": "complete",
        "input_binding_digest": inputs["binding_digest"],
        "config_digest": summary["config_digest"],
        "implementation_bundle_digest": "e" * 64,
        "artifacts": artifacts,
        "artifact_digest": _digest(artifacts),
    }
    done_raw = _json_bytes(done)
    (final / "done.json").write_bytes(done_raw)
    return final, _sha(done_raw)


class HighRecallTest(unittest.TestCase):
    def test_waives_only_entropy_and_selects_score_family_round_robin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                _row(
                    root,
                    "walk-low",
                    "walk",
                    0.70,
                    [high_recall.ADVISORY_REASON],
                ),
                _row(
                    root,
                    "run-best",
                    "run",
                    0.99,
                    [high_recall.ADVISORY_REASON],
                ),
                _row(
                    root,
                    "walk-best",
                    "walk",
                    0.95,
                    [high_recall.ADVISORY_REASON],
                ),
                _row(
                    root,
                    "run-hard",
                    "run",
                    1.00,
                    [high_recall.ADVISORY_REASON, "scene_cut_nonzero"],
                ),
            ]
            source_final, done_sha = _upstream(root, rows)
            output = root / "high-recall"
            with patch.object(
                high_recall.prefilter,
                "_extract_anchor_png_bytes",
                return_value=(_png(), 16, 12),
            ):
                summary = high_recall.finalize(
                    source_final_dir=source_final,
                    expected_source_done_sha256=done_sha,
                    output_dir=output,
                    sample_size=3,
                )
            selected = [
                json.loads(line)
                for line in (output / high_recall.SELECTED_NAME).read_text().splitlines()
            ]
            self.assertEqual(
                [row["iid"] for row in selected],
                ["run-best", "walk-best", "walk-low"],
            )
            self.assertNotIn("run-hard", [row["iid"] for row in selected])
            for row in selected:
                self.assertIs(validate_input_row(row), row)
                policy = row["source_gate_policy"]
                self.assertIn(high_recall.ADVISORY_REASON, policy["advisory_waivers"])
                self.assertEqual(policy["hard_rejection_reasons"], [])
            self.assertEqual(summary["counts"]["entropy_advisory_present"], 4)
            self.assertEqual(
                summary["counts"]["entropy_advisory_waived_eligible"], 3
            )
            self.assertEqual(summary["counts"]["selected"], 3)
            self.assertEqual(
                {path.name for path in output.iterdir()}, high_recall.FINAL_ENTRIES
            )

    def test_group_unique_and_upstream_selection_annotation_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotated = _row(
                root,
                "annotated",
                "walk",
                0.90,
                [high_recall.UPSTREAM_SELECTION_ANNOTATION],
                group="shared",
                entropy=0.5,
            )
            annotated["eligible"] = True
            rows = [
                annotated,
                _row(root, "higher", "run", 0.95, [], group="shared", entropy=0.5),
                _row(root, "other", "jump", 0.80, [], entropy=0.5),
            ]
            source_final, done_sha = _upstream(root, rows)
            output = root / "high-recall"
            with patch.object(
                high_recall.prefilter,
                "_extract_anchor_png_bytes",
                return_value=(_png(), 16, 12),
            ):
                high_recall.finalize(
                    source_final_dir=source_final,
                    expected_source_done_sha256=done_sha,
                    output_dir=output,
                    sample_size=3,
                )
            selected = [
                json.loads(line)
                for line in (output / high_recall.SELECTED_NAME).read_text().splitlines()
            ]
            self.assertEqual([row["iid"] for row in selected], ["higher", "other"])
            self.assertEqual(len({row["group_id"] for row in selected}), len(selected))

    def test_done_sha_binding_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_final, unused_done_sha = _upstream(
                root, [_row(root, "a", "walk", 0.9, [], entropy=0.5)]
            )
            with self.assertRaisesRegex(high_recall.HighRecallError, "done SHA differs"):
                high_recall.finalize(
                    source_final_dir=source_final,
                    expected_source_done_sha256="0" * 64,
                    output_dir=root / "output",
                    sample_size=1,
                )

    def test_false_upstream_verdict_without_reason_cannot_be_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inconsistent = _row(root, "a", "walk", 0.9, [], entropy=0.5)
            inconsistent["eligible"] = False
            decision = high_recall._eligibility_decision(
                inconsistent, max_spatial_energy_entropy=1.0
            )
            self.assertFalse(decision["high_recall_eligible"])
            self.assertIn(
                "high_recall_upstream_eligibility_inconsistent",
                decision["hard_rejection_reasons"],
            )

    def test_parallel_and_serial_anchor_outputs_are_byte_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                _row(root, "walk-a", "walk", 0.91, [], entropy=0.5),
                _row(root, "run-a", "run", 0.99, [], entropy=0.5),
                _row(root, "walk-b", "walk", 0.89, [], entropy=0.5),
                _row(root, "jump-a", "jump", 0.95, [], entropy=0.5),
            ]
            for value, row in enumerate(rows, start=20):
                _replace_source_with_decodable_video(row, value=value * 5)
            source_final, done_sha = _upstream(root, rows)
            output = root / "high-recall"
            high_recall.finalize(
                source_final_dir=source_final,
                expected_source_done_sha256=done_sha,
                output_dir=output,
                sample_size=4,
                local_workers=1,
            )
            serial_selected = (output / high_recall.SELECTED_NAME).read_bytes()
            serial_eligibility = (output / high_recall.ELIGIBILITY_NAME).read_bytes()
            serial_anchors = {
                path.name: path.read_bytes()
                for path in sorted((output / high_recall.ANCHOR_DIR).iterdir())
            }
            shutil.rmtree(output)

            extract_anchor = high_recall._extract_bound_anchor

            def complete_out_of_order(payload):
                time.sleep(0.01 * (4 - int(payload["index"])))
                return extract_anchor(payload)

            # The sandbox used by this unit suite denies POSIX semaphore
            # sysconf.  Substitute only the executor transport while exercising
            # the identical bounded/dynamic scheduling and ordered-write path.
            with patch.object(
                high_recall, "ProcessPoolExecutor", ThreadPoolExecutor
            ), patch.object(
                high_recall, "_extract_bound_anchor", complete_out_of_order
            ):
                parallel_summary = high_recall.finalize(
                    source_final_dir=source_final,
                    expected_source_done_sha256=done_sha,
                    output_dir=output,
                    sample_size=4,
                    local_workers=2,
                )
            parallel_anchors = {
                path.name: path.read_bytes()
                for path in sorted((output / high_recall.ANCHOR_DIR).iterdir())
            }
            self.assertEqual(
                (output / high_recall.SELECTED_NAME).read_bytes(), serial_selected
            )
            self.assertEqual(
                (output / high_recall.ELIGIBILITY_NAME).read_bytes(),
                serial_eligibility,
            )
            self.assertEqual(parallel_anchors, serial_anchors)
            selected = [
                json.loads(line) for line in serial_selected.decode().splitlines()
            ]
            self.assertEqual(
                [row["selection_rank"] for row in selected], [1, 2, 3, 4]
            )
            self.assertEqual(
                parallel_summary["anchor_extraction"]["executor"],
                "bounded_dynamic_process_pool",
            )


if __name__ == "__main__":
    unittest.main()
