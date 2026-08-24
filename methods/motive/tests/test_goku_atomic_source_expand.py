from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from motive import goku_atomic_source_expand as expand
from motive.goku_action_anchor_qwen import validate_input_row


def _jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    raw = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _mother(iid: str, group: str) -> dict[str, object]:
    return {
        "iid": iid,
        "group_id": group,
        "prompt": "Make the person wave.",
        "source_caption": "A person moves in view.",
        "edited_caption": "The person waves.",
        "src_video": f"{iid}.mp4",
        "auto_rule": {"action_families": ["gesture"]},
    }


def _result(payload: dict[str, object]) -> dict[str, object]:
    source = Path(str(payload["video_path"]))
    stat = source.stat()
    return {
        "iid": payload["iid"],
        "ok": True,
        "media": {
            "width": 1280,
            "height": 720,
            "pixels": 1280 * 720,
            "short_side": 720,
            "fps": 25.0,
            "frame_count": 81,
            "duration_seconds": 3.2,
            "file_size_bytes": stat.st_size,
            "mtime_ns_at_analysis": stat.st_mtime_ns,
        },
        "motion": {
            "label": "dynamic_object",
            "raw_speed_mean": 0.02,
            "raw_speed_p90": 0.03,
            "residual_speed_mean": 0.015,
            "residual_speed_p90": 0.02,
            "residual_speed_p99": 0.04,
            "active_pixel_fraction": 0.08,
            "active_frame_fraction": 0.8,
            "camera_explained_ratio": 0.2,
            "affine_inlier_ratio": 0.8,
            "scene_cut_ratio": 0.0,
            "temporal_energy_cv": 0.3,
            "sampled_frames": 32,
        },
        "actor_motion": {
            "active_fraction": 0.08,
            "temporal_coverage": 0.8,
            "largest_component_share": 0.6,
            "support_bbox_fraction": 0.3,
            "spatial_energy_entropy": 0.5,
            "direction_consistency": 0.6,
            "centroid_path_length": 0.4,
            "centroid_acceleration": 0.02,
            "adjacent_energy_coherence": 0.8,
            "periodicity": 0.1,
            "actor_likeness": 0.75,
        },
    }


def _png() -> bytes:
    ok, encoded = cv2.imencode(
        ".png", np.full((12, 16, 3), 127, dtype=np.uint8)
    )
    assert ok
    return encoded.tobytes()


class SourceExpandTest(unittest.TestCase):
    def _fixture(self, root: Path):
        video_root = root / "videos"
        video_root.mkdir()
        mother_rows = [_mother("a", "ga"), _mother("b", "gb"), _mother("c", "gc")]
        for row in mother_rows:
            (video_root / str(row["src_video"])).write_bytes(
                f"video-{row['iid']}".encode()
            )
        mother = root / "mother.jsonl"
        mother_sha = _jsonl(mother, mother_rows)
        # Repeated legacy group IDs are legal and still form one exclusion set.
        old = root / "old.jsonl"
        old_sha = _jsonl(
            old,
            [
                {"iid": "old-a", "group_id": "same-old-group"},
                {"iid": "old-b", "group_id": "same-old-group"},
            ],
        )
        return video_root, mother, mother_sha, old, old_sha

    def test_worker_crash_keeps_prior_receipt_and_resume_skips_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_root, mother, mother_sha, old, old_sha = self._fixture(root)
            work = root / "work"
            calls = 0

            def crash_second(payload):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated worker crash")
                return _result(payload)

            with patch.object(expand.prefilter, "_analyze_payload", side_effect=crash_second):
                with self.assertRaisesRegex(RuntimeError, "simulated worker crash"):
                    expand.run_worker(
                        input_path=mother,
                        old_selected=old,
                        video_root=video_root,
                        work_dir=work,
                        expected_input_sha256=mother_sha,
                        expected_old_selected_sha256=old_sha,
                        worker_index=0,
                        num_workers=1,
                        local_workers=1,
                    )
            first = work / expand.RECEIPT_DIR / "a.json"
            self.assertTrue(first.is_file())
            self.assertFalse((work / expand.RECEIPT_DIR / "b.json").exists())
            frozen = first.read_bytes()

            with patch.object(
                expand.prefilter, "_analyze_payload", side_effect=_result
            ) as analyze:
                summary = expand.run_worker(
                    input_path=mother,
                    old_selected=old,
                    video_root=video_root,
                    work_dir=work,
                    expected_input_sha256=mother_sha,
                    expected_old_selected_sha256=old_sha,
                    worker_index=0,
                    num_workers=1,
                    local_workers=1,
                )
            self.assertEqual(summary["resumed"], 1)
            self.assertEqual(analyze.call_count, 2)
            self.assertEqual(first.read_bytes(), frozen)
            self.assertEqual(
                sorted(path.stem for path in (work / expand.RECEIPT_DIR).glob("*.json")),
                ["a", "b", "c"],
            )

    def test_finalize_follows_mother_rank_and_emits_atomic_compatible_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_root, mother, mother_sha, old, old_sha = self._fixture(root)
            work = root / "work"
            with patch.object(
                expand.prefilter, "_analyze_payload", side_effect=_result
            ):
                expand.run_worker(
                    input_path=mother,
                    old_selected=old,
                    video_root=video_root,
                    work_dir=work,
                    expected_input_sha256=mother_sha,
                    expected_old_selected_sha256=old_sha,
                    worker_index=0,
                    num_workers=1,
                    local_workers=1,
                )
            output = root / "final"
            with patch.object(
                expand.prefilter,
                "_extract_anchor_png_bytes",
                return_value=(_png(), 16, 12),
            ):
                summary = expand.finalize(
                    input_path=mother,
                    old_selected=old,
                    work_dir=work,
                    output_dir=output,
                    expected_input_sha256=mother_sha,
                    expected_old_selected_sha256=old_sha,
                    sample_size=2,
                )
            selected = [
                json.loads(line)
                for line in (output / expand.SELECTED_NAME).read_text().splitlines()
            ]
            self.assertEqual([row["iid"] for row in selected], ["a", "b"])
            self.assertEqual([row["selection_rank"] for row in selected], [1, 2])
            for row in selected:
                self.assertIs(validate_input_row(row), row)
                anchor = output / row["anchor_image"]
                self.assertEqual(
                    hashlib.sha256(anchor.read_bytes()).hexdigest(),
                    row["anchor_sha256"],
                )
                self.assertFalse(row["legacy_qwen_provenance"]["used_as_gate"])
            self.assertEqual(summary["counts"]["selected"], 2)
            self.assertEqual({path.name for path in output.iterdir()}, expand.FINAL_ENTRIES)

    def test_sha_binding_and_complete_receipt_closure_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_root, mother, mother_sha, old, old_sha = self._fixture(root)
            with self.assertRaisesRegex(expand.SourceExpandError, "mother input SHA differs"):
                expand.run_worker(
                    input_path=mother,
                    old_selected=old,
                    video_root=video_root,
                    work_dir=root / "work",
                    expected_input_sha256="0" * 64,
                    expected_old_selected_sha256=old_sha,
                    worker_index=0,
                    num_workers=1,
                    local_workers=1,
                )
            empty_work = root / "empty-work"
            empty_work.mkdir()
            with self.assertRaisesRegex(expand.SourceExpandError, "closure incomplete"):
                expand.finalize(
                    input_path=mother,
                    old_selected=old,
                    work_dir=empty_work,
                    output_dir=root / "final",
                    expected_input_sha256=mother_sha,
                    expected_old_selected_sha256=old_sha,
                    sample_size=1,
                )

    def test_wan_geometry_and_scene_cut_are_hard_gates(self) -> None:
        config = expand.SourceExpandConfig()
        result = _result({"iid": "x", "video_path": __file__})
        self.assertEqual(expand._media_reasons(result, config), [])
        mutations = (
            ("media", "frame_count", 80, "frame_count_not_wan81"),
            ("media", "fps", 24.0, "fps_not_wan25"),
            ("media", "short_side", 639, "resolution_short_side_too_small"),
            ("motion", "scene_cut_ratio", 0.1, "scene_cut_nonzero"),
        )
        for section, field, value, reason in mutations:
            with self.subTest(field=field):
                changed = json.loads(json.dumps(result))
                changed[section][field] = value
                self.assertIn(reason, expand._media_reasons(changed, config))

    def test_raw_directory_build_input_is_filename_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "combine_json"
            raw.mkdir()
            for name, iid in (("z_all.json", "z"), ("a_all.json", "a")):
                (raw / name).write_text(
                    json.dumps(
                        {
                            "case_id": iid,
                            "source_video": f"{iid}.mp4",
                            "instruction_en": "Make the person wave.",
                            "source_caption": "A person moves.",
                        }
                    )
                )
            output = root / "normalized.jsonl"
            summary = expand.build_input(input_raw_dir=raw, output_jsonl=output)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["iid"] for row in rows], ["a", "z"])
            self.assertEqual([row["mother_rank"] for row in rows], [1, 2])
            self.assertEqual(summary["rows"], 2)


if __name__ == "__main__":
    unittest.main()
