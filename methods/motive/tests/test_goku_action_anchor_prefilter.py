from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from motive import goku_action_anchor_prefilter as prefilter
from motive import qwen_filter
from motive.geometry import MotionAnalysis, MotionMetrics
from motive.motion_features import ActorMotionFeatures


def _object_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _visual(
    iid: str,
    input_digest: str,
    *,
    source_motion: str = "clear",
    preservation: str = "poor",
    camera_dominance: str = "low",
    background_dominance: str = "low",
    artifact_level: str = "low",
) -> dict[str, object]:
    observation = {
        "schema_version": qwen_filter.OBSERVATION_SCHEMA_VERSION,
        "source_action": "the dog runs from left to right",
        "target_action": "the dog jumps over the branch",
        "source_actor_motion": source_motion,
        "target_actor_motion": "clear",
        "camera_dominance": camera_dominance,
        "background_dominance": background_dominance,
        "artifact_level": artifact_level,
        "preservation_quality": preservation,
        "temporal_evidence": [
            "Across ordered source frames the dog changes position.",
            "Across ordered target frames the dog rises over the branch.",
        ],
        "uncertainty_codes": [],
    }
    result = {
        "schema_version": qwen_filter.VISUAL_SCHEMA_VERSION,
        "verdict": "valid_action",
        "edit_effect": "changed_action",
        "action_signature": "jump over branch",
        "reason_codes": ["visible_target_action"],
        "uncertainty_codes": [],
        "confidence": "high",
    }
    return {
        "iid": iid,
        "input_digest": input_digest,
        "status": "ok",
        "mode": "visual",
        "observation_validated_from": "original",
        "result_validated_from": "original",
        "observation": observation,
        "result": result,
        "observation_digest": _object_digest(observation),
        "result_digest": _object_digest(result),
        "visual_input_digest": "1" * 64,
        "run_config_digest": "2" * 64,
        "config_digest": "3" * 64,
        "implementation_digest": "4" * 64,
        "execution_manifest_sha256": "5" * 64,
        "model_revision": "test-model-revision",
        "transformers_version": "test-transformers-version",
        "execution_manifest": "test-execution-manifest.json",
        "execution_shard_index": 0,
        "execution_shard_count": 8,
        "observation_repairs": [],
        "alignment_repairs": [],
    }


def _row(
    iid: str,
    *,
    family: str = "jump",
    group_id: str | None = None,
    source_motion: str = "clear",
    camera_dominance: str = "low",
    background_dominance: str = "low",
    artifact_level: str = "low",
) -> dict[str, object]:
    input_digest = hashlib.sha256(f"input:{iid}".encode()).hexdigest()
    return {
        "iid": iid,
        "input_digest": input_digest,
        "group_id": group_id or f"group-{iid}",
        "prompt": "Make the dog jump over the nearby branch.",
        "source_caption": "A dog runs beside a nearby branch.",
        "edited_caption": "The same dog jumps over the nearby branch.",
        "src_video": f"videos/{iid}/source.mp4",
        "tgt_video": f"videos/{iid}/edited.mp4",
        "auto_rule": {"action_families": [family]},
        "qwen_evidence": {
            "visual": _visual(
                iid,
                input_digest,
                source_motion=source_motion,
                camera_dominance=camera_dominance,
                background_dominance=background_dominance,
                artifact_level=artifact_level,
            )
        },
    }


def _analysis(path: Path) -> MotionAnalysis:
    metrics = MotionMetrics(
        raw_speed_mean=0.018,
        raw_speed_p90=0.035,
        residual_speed_mean=0.014,
        residual_speed_p90=0.022,
        residual_speed_p99=0.050,
        active_pixel_fraction=0.080,
        active_frame_fraction=0.85,
        camera_explained_ratio=0.15,
        affine_inlier_ratio=0.80,
        scene_cut_ratio=0.0,
        temporal_energy_cv=0.30,
        sampled_frames=8,
        duration_seconds=4.0,
        source_fps=24.0,
        source_frame_count=96,
        source_width=1280,
        source_height=720,
    )
    gray = np.zeros((8, 8, 8), dtype=np.uint8)
    flow = np.zeros((7, 8, 8, 2), dtype=np.float32)
    return MotionAnalysis(
        path=path,
        label="dynamic_object",
        metrics=metrics,
        frames_gray=gray,
        frame_times=np.arange(8, dtype=np.float32) / 24.0,
        raw_flows=flow,
        global_flows=flow,
        residual_flows=flow,
    )


def _actor() -> ActorMotionFeatures:
    return ActorMotionFeatures(
        active_fraction=0.08,
        temporal_coverage=0.90,
        largest_component_share=0.70,
        support_bbox_fraction=0.20,
        spatial_energy_entropy=0.55,
        direction_consistency=0.65,
        centroid_path_length=0.50,
        centroid_acceleration=0.02,
        adjacent_energy_coherence=0.80,
        periodicity=0.20,
        actor_likeness=0.75,
    )


def _png() -> bytes:
    ok, encoded = cv2.imencode(
        ".png", np.full((12, 16, 3), 127, dtype=np.uint8)
    )
    assert ok
    return encoded.tobytes()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class QwenSourceGateTest(unittest.TestCase):
    def test_requires_clear_original_but_not_old_pair_quality(self) -> None:
        row = _row(
            "iid-1",
            camera_dominance="high",
            background_dominance="high",
            artifact_level="high",
        )
        gate, reasons = prefilter.qwen_source_gate(row)
        self.assertEqual(reasons, [])
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["evidence_integrity"])
        self.assertEqual(gate["camera_dominance"], "high")
        self.assertEqual(gate["background_dominance"], "high")
        self.assertEqual(gate["artifact_level"], "high")
        self.assertEqual(gate["preservation_quality"], "poor")

        weak = _row("iid-2", source_motion="weak")
        gate, reasons = prefilter.qwen_source_gate(weak)
        self.assertFalse(gate["passed"])
        self.assertIn("qwen_source_motion_not_clear", reasons)

        repaired = _row("iid-3")
        repaired["qwen_evidence"]["visual"][
            "observation_validated_from"
        ] = "repair_1"
        gate, reasons = prefilter.qwen_source_gate(repaired)
        self.assertFalse(gate["passed"])
        self.assertIn("qwen_observation_not_original", reasons)

    def test_source_evidence_identity_and_digest_mismatches_fail_closed(
        self,
    ) -> None:
        mutations = {
            "iid": lambda row: row["qwen_evidence"]["visual"].__setitem__(
                "iid", "different-iid"
            ),
            "input_digest": (
                lambda row: row["qwen_evidence"]["visual"].__setitem__(
                    "input_digest", "f" * 64
                )
            ),
            "observation_digest": (
                lambda row: row["qwen_evidence"]["visual"]["observation"]
                .__setitem__("source_action", "the dog is stationary")
            ),
            "implementation_digest": (
                lambda row: row["qwen_evidence"]["visual"].__setitem__(
                    "implementation_digest", "not-a-digest"
                )
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                row = copy.deepcopy(_row(f"mismatch-{name}"))
                mutate(row)
                gate, reasons = prefilter.qwen_source_gate(row)
                self.assertFalse(gate["passed"])
                self.assertFalse(gate["evidence_integrity"])
                self.assertIn("invalid_r7_qwen_evidence", reasons)
                self.assertIsNotNone(gate["evidence_error"])

    def test_corrupt_paired_result_is_ignored_by_source_gate(self) -> None:
        baseline = _row("source-only-result")
        baseline_gate, baseline_reasons = prefilter.qwen_source_gate(baseline)
        self.assertEqual(baseline_reasons, [])

        mutated = copy.deepcopy(baseline)
        visual = mutated["qwen_evidence"]["visual"]
        visual["result"] = {
            "corrupt_target_verdict": ["not", "the", "old", "schema"]
        }
        visual["result_digest"] = "definitely-not-a-digest"
        visual["result_validated_from"] = "broken"
        visual["alignment_repairs"] = "broken"
        gate, reasons = prefilter.qwen_source_gate(mutated)

        self.assertEqual(reasons, [])
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["evidence_integrity"])
        self.assertTrue(gate["legacy_result_ignored"])
        self.assertEqual(
            gate["legacy_observation_digest"],
            baseline_gate["legacy_observation_digest"],
        )
        self.assertEqual(
            gate["observed_source_action"],
            baseline_gate["observed_source_action"],
        )

    def test_observation_digest_field_mutation_fails_closed(self) -> None:
        row = _row("source-digest-field")
        row["qwen_evidence"]["visual"]["observation_digest"] = "f" * 64
        gate, reasons = prefilter.qwen_source_gate(row)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["evidence_integrity"])
        self.assertIn("invalid_r7_qwen_evidence", reasons)


class MediaMotionGateTest(unittest.TestCase):
    @staticmethod
    def _boundary_result(
        config: prefilter.PrefilterConfig,
    ) -> dict[str, object]:
        return {
            "media": {
                "short_side": config.min_short_side,
                "pixels": config.min_pixels,
                "fps": config.min_fps,
                "duration_seconds": config.min_duration_seconds,
                "frame_count": config.min_source_frames,
            },
            "motion": {
                "label": "dynamic_object",
                "residual_speed_p90": config.min_residual_speed_p90,
                "active_pixel_fraction": config.min_active_pixel_fraction,
                "active_frame_fraction": config.min_active_frame_fraction,
            },
            "actor_motion": {
                "actor_likeness": config.min_actor_likeness,
                "temporal_coverage": config.min_temporal_coverage,
                "largest_component_share": (
                    config.min_largest_component_share
                ),
                "spatial_energy_entropy": (
                    config.max_spatial_energy_entropy
                ),
            },
        }

    def test_exact_threshold_boundaries_pass(self) -> None:
        config = prefilter.PrefilterConfig()
        self.assertEqual(
            prefilter._media_motion_reasons(
                self._boundary_result(config),
                config,
            ),
            [],
        )

    def test_each_minimum_just_below_boundary_is_rejected(self) -> None:
        config = prefilter.PrefilterConfig()
        cases = (
            (
                ("media", "short_side"),
                config.min_short_side - 1,
                "resolution_short_side_too_small",
            ),
            (
                ("media", "pixels"),
                config.min_pixels - 1,
                "resolution_pixel_count_too_small",
            ),
            (
                ("media", "fps"),
                math.nextafter(config.min_fps, -math.inf),
                "fps_out_of_range",
            ),
            (
                ("media", "duration_seconds"),
                math.nextafter(config.min_duration_seconds, -math.inf),
                "duration_out_of_range",
            ),
            (
                ("media", "frame_count"),
                config.min_source_frames - 1,
                "too_few_source_frames",
            ),
            (
                ("motion", "residual_speed_p90"),
                math.nextafter(config.min_residual_speed_p90, -math.inf),
                "residual_motion_too_weak",
            ),
            (
                ("motion", "active_pixel_fraction"),
                math.nextafter(config.min_active_pixel_fraction, -math.inf),
                "active_pixel_fraction_too_small",
            ),
            (
                ("motion", "active_frame_fraction"),
                math.nextafter(config.min_active_frame_fraction, -math.inf),
                "active_frame_fraction_too_small",
            ),
            (
                ("actor_motion", "actor_likeness"),
                math.nextafter(config.min_actor_likeness, -math.inf),
                "actor_likeness_too_low",
            ),
            (
                ("actor_motion", "temporal_coverage"),
                math.nextafter(config.min_temporal_coverage, -math.inf),
                "actor_temporal_coverage_too_low",
            ),
            (
                ("actor_motion", "largest_component_share"),
                math.nextafter(
                    config.min_largest_component_share,
                    -math.inf,
                ),
                "actor_motion_too_diffuse",
            ),
        )
        for path, value, expected_reason in cases:
            with self.subTest(path=".".join(path)):
                result = self._boundary_result(config)
                result[path[0]][path[1]] = value
                reasons = prefilter._media_motion_reasons(result, config)
                self.assertIn(expected_reason, reasons)

    def test_upper_bound_violations_are_rejected(self) -> None:
        config = prefilter.PrefilterConfig()
        cases = (
            (
                ("media", "fps"),
                math.nextafter(config.max_fps, math.inf),
                "fps_out_of_range",
            ),
            (
                ("media", "duration_seconds"),
                math.nextafter(config.max_duration_seconds, math.inf),
                "duration_out_of_range",
            ),
            (
                ("actor_motion", "spatial_energy_entropy"),
                math.nextafter(
                    config.max_spatial_energy_entropy,
                    math.inf,
                ),
                "spatial_motion_entropy_too_high",
            ),
        )
        for path, value, expected_reason in cases:
            with self.subTest(path=".".join(path)):
                result = self._boundary_result(config)
                result[path[0]][path[1]] = value
                reasons = prefilter._media_motion_reasons(result, config)
                self.assertIn(expected_reason, reasons)


class ContainerMetadataTest(unittest.TestCase):
    @staticmethod
    def _capture(*, fps: float, opened: bool = True) -> MagicMock:
        capture = MagicMock()
        capture.isOpened.return_value = opened
        values = {
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_COUNT: 96.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
        }
        capture.get.side_effect = values.__getitem__
        return capture

    def test_invalid_or_unknown_fps_fails_closed(self) -> None:
        for fps in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(fps=fps):
                capture = self._capture(fps=fps)
                with patch.object(
                    prefilter.cv2,
                    "VideoCapture",
                    return_value=capture,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "invalid or unavailable container metadata: fps",
                    ):
                        prefilter._probe_container_metadata(
                            Path("/not/read/by/mock.mp4")
                        )
                capture.release.assert_called_once_with()

    def test_unopenable_video_fails_closed(self) -> None:
        capture = self._capture(fps=24.0, opened=False)
        with patch.object(
            prefilter.cv2,
            "VideoCapture",
            return_value=capture,
        ):
            with self.assertRaisesRegex(RuntimeError, "could not open"):
                prefilter._probe_container_metadata(Path("missing.mp4"))


class SourcePathResolutionTest(unittest.TestCase):
    def test_inside_absolute_relative_and_symlink_paths_are_confined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            video_root = root / "dataset"
            video_root.mkdir()
            inside = video_root / "inside.mp4"
            inside.write_bytes(b"inside")

            self.assertEqual(
                prefilter._resolve_source_video(
                    str(inside),
                    video_root,
                ),
                inside.resolve(),
            )
            self.assertEqual(
                prefilter._resolve_source_video(
                    "inside.mp4",
                    video_root,
                ),
                inside.resolve(),
            )

            inside_link = video_root / "inside-link.mp4"
            inside_link.symlink_to(inside)
            self.assertEqual(
                prefilter._resolve_source_video(
                    "inside-link.mp4",
                    video_root,
                ),
                inside.resolve(),
            )

    def test_absolute_relative_and_symlink_escapes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            video_root = root / "dataset"
            video_root.mkdir()
            outside = root / "outside.mp4"
            outside.write_bytes(b"outside")
            escape_link = video_root / "escape-link.mp4"
            escape_link.symlink_to(outside)

            for value in (
                str(outside),
                "../outside.mp4",
                "escape-link.mp4",
            ):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(
                        ValueError,
                        "escapes video_root",
                    ):
                        prefilter._resolve_source_video(value, video_root)


class ParserDefaultsTest(unittest.TestCase):
    def test_cli_defaults_equal_prefilter_config(self) -> None:
        expected = prefilter.PrefilterConfig()
        with patch.object(
            prefilter.os,
            "cpu_count",
            return_value=max(expected.workers, os.cpu_count() or 1),
        ):
            parser = prefilter.build_parser()
        args = parser.parse_args(
            [
                "--input-fused",
                "input.jsonl",
                "--video-root",
                "videos",
                "--output-dir",
                "output",
            ]
        )
        for field in fields(prefilter.PrefilterConfig):
            with self.subTest(field=field.name):
                self.assertEqual(
                    getattr(args, field.name),
                    getattr(expected, field.name),
                )


class DiversityTest(unittest.TestCase):
    @staticmethod
    def _candidate(
        iid: str,
        family: str,
        score: float,
        group: str,
    ) -> dict[str, object]:
        return {
            "iid": iid,
            "family": family,
            "prefilter_score": score,
            "group_id": group,
            "eligible": True,
        }

    def test_family_round_robin_is_order_independent_and_group_unique(self) -> None:
        rows = [
            self._candidate("a1", "a", 0.95, "g1"),
            self._candidate("a2", "a", 0.90, "shared"),
            self._candidate("b1", "b", 0.94, "g2"),
            self._candidate("b2", "b", 0.89, "shared"),
            self._candidate("c1", "c", 0.70, "g3"),
        ]
        first = prefilter.select_diverse(
            rows, sample_size=5, max_per_family=2
        )
        second = prefilter.select_diverse(
            list(reversed(rows)), sample_size=5, max_per_family=2
        )
        first_ids = [row["iid"] for row in first]
        self.assertEqual(first_ids, [row["iid"] for row in second])
        self.assertEqual(first_ids[:3], ["a1", "b1", "c1"])
        self.assertEqual(len(first_ids), 4)
        self.assertEqual(
            len({row["group_id"] for row in first}), len(first)
        )
        self.assertEqual(
            [row["selection_rank"] for row in first], [1, 2, 3, 4]
        )


class PrefilterIntegrationTest(unittest.TestCase):
    def _fixture(
        self, root: Path
    ) -> tuple[Path, Path, list[dict[str, object]]]:
        video_root = root / "dataset"
        rows = [
            _row("clear-b", family="run"),
            _row("weak", family="walk", source_motion="weak"),
            _row("clear-a", family="jump"),
        ]
        for row in rows:
            source = video_root / str(row["src_video"])
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"video-{row['iid']}".encode())
        fused = root / "fused.jsonl"
        _write_jsonl(fused, rows)
        return fused, video_root, rows

    def test_atomic_artifact_and_required_selected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fused, video_root, unused_rows = self._fixture(root)
            output = root / "prefilter"

            def analyze(path, unused_config):
                return _analysis(Path(path))

            with patch.object(
                prefilter,
                "analyze_video",
                side_effect=analyze,
            ), patch.object(
                prefilter,
                "_probe_container_metadata",
                return_value={
                    "fps": 24.0,
                    "frame_count": 96,
                    "width": 1280,
                    "height": 720,
                },
            ), patch.object(
                prefilter,
                "extract_actor_motion_features",
                return_value=_actor(),
            ), patch.object(
                prefilter,
                "_extract_anchor_png_bytes",
                return_value=(_png(), 16, 12),
            ):
                summary = prefilter.run_prefilter(
                    input_fused=fused,
                    video_root=video_root,
                    output_dir=output,
                    config=prefilter.PrefilterConfig(
                        sample_size=2,
                        workers=1,
                        max_per_family=2,
                    ),
                )

            self.assertEqual(
                {path.name for path in output.iterdir()},
                prefilter.OUTPUT_ENTRIES,
            )
            selected = [
                json.loads(line)
                for line in (output / prefilter.SELECTED_NAME)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(selected), 2)
            self.assertEqual(summary["counts"]["evaluated"], 3)
            self.assertEqual(summary["counts"]["eligible"], 2)
            required = {
                "iid",
                "group_id",
                "family",
                "src_video",
                "resolved_src_video",
                "source_caption",
                "edited_caption",
                "prompt",
                "anchor_image",
                "anchor_sha256",
                "source_video_sha256",
                "prefilter_score",
                "media",
                "motion",
                "actor_motion",
            }
            for row in selected:
                self.assertTrue(required.issubset(row))
                anchor = output / row["anchor_image"]
                self.assertTrue(anchor.is_file())
                self.assertEqual(
                    hashlib.sha256(anchor.read_bytes()).hexdigest(),
                    row["anchor_sha256"],
                )
                source = Path(row["resolved_src_video"])
                self.assertEqual(
                    hashlib.sha256(source.read_bytes()).hexdigest(),
                    row["source_video_sha256"],
                )

            evaluated = [
                json.loads(line)
                for line in (output / prefilter.EVALUATED_NAME)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [row["iid"] for row in evaluated],
                sorted(row["iid"] for row in evaluated),
            )
            weak = next(row for row in evaluated if row["iid"] == "weak")
            self.assertIn(
                "qwen_source_motion_not_clear",
                weak["rejection_reasons"],
            )
            self.assertIsNone(weak["media"])
            done = json.loads(
                (output / prefilter.DONE_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(done["status"], "complete")
            self.assertEqual(done["selected_rows"], 2)

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                prefilter.run_prefilter(
                    input_fused=fused,
                    video_root=video_root,
                    output_dir=output,
                    config=prefilter.PrefilterConfig(workers=1),
                )

    def test_anchor_failure_leaves_no_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fused, video_root, unused_rows = self._fixture(root)
            output = root / "prefilter"
            with patch.object(
                prefilter,
                "analyze_video",
                side_effect=lambda path, unused_config: _analysis(Path(path)),
            ), patch.object(
                prefilter,
                "_probe_container_metadata",
                return_value={
                    "fps": 24.0,
                    "frame_count": 96,
                    "width": 1280,
                    "height": 720,
                },
            ), patch.object(
                prefilter,
                "extract_actor_motion_features",
                return_value=_actor(),
            ), patch.object(
                prefilter,
                "_extract_anchor_png_bytes",
                side_effect=RuntimeError("decode failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "decode failed"):
                    prefilter.run_prefilter(
                        input_fused=fused,
                        video_root=video_root,
                        output_dir=output,
                        config=prefilter.PrefilterConfig(
                            sample_size=1,
                            workers=1,
                        ),
                    )
            self.assertFalse(output.exists())
            self.assertEqual(
                list(root.glob(".prefilter.staging-*")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
