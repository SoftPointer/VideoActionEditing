from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import saic_exact81_media_diagnostics_v1 as diagnostics  # noqa: E402


class _Metrics:
    def __init__(self, *, camera: float, temporal_cv: float) -> None:
        self.values = {
            "raw_speed_mean": 0.02,
            "raw_speed_p90": 0.03,
            "residual_speed_mean": 0.01,
            "residual_speed_p90": 0.02,
            "residual_speed_p99": 0.025,
            "active_pixel_fraction": 0.2,
            "active_frame_fraction": 0.5,
            "camera_explained_ratio": camera,
            "affine_inlier_ratio": 0.8,
            "scene_cut_ratio": 0.0,
            "temporal_energy_cv": temporal_cv,
            "sampled_frames": 81,
            "duration_seconds": 3.2,
            "source_fps": 25.0,
            "source_frame_count": 81,
            "source_width": 16,
            "source_height": 12,
        }

    def to_dict(self) -> dict[str, float | int]:
        return dict(self.values)


def _fake_analysis(path: str | Path, config: object) -> object:
    assert config.analysis_frames == 81
    candidate = Path(path).name.startswith("candidate")
    shape = (80, 6, 8, 2)
    raw = np.zeros(shape, dtype=np.float32)
    global_flow = np.zeros(shape, dtype=np.float32)
    residual = np.zeros(shape, dtype=np.float32)
    global_flow[..., 0] = 0.02 if candidate else 0.01
    raw[..., 0] = global_flow[..., 0] + 0.005
    residual[..., 0] = 0.005
    return SimpleNamespace(
        label="dynamic_object",
        metrics=_Metrics(camera=0.7 if candidate else 0.6, temporal_cv=0.3 if candidate else 0.2),
        frames_gray=np.zeros((81, 6, 8), dtype=np.uint8),
        frame_times=np.arange(81, dtype=np.float32) / 25.0,
        raw_flows=raw,
        global_flows=global_flow,
        residual_flows=residual,
    )


def _fake_decode(path: str | Path, *, ffmpeg: str, ffprobe: str):
    candidate = Path(path).name.startswith("candidate")
    frame_size = 6 * 8 * 3
    frames = tuple(
        bytes(((index * 3 + offset + int(candidate) * 7) % 256) for offset in range(frame_size))
        for index in range(81)
    )
    raw = b"".join(frames)
    return frames, {
        "frame_count": 81,
        "fps": 25,
        "width": 8,
        "height": 6,
        "decoded_rgb24_sha256": hashlib.sha256(raw).hexdigest(),
        "decoder_contract": "mock-exact81-rgb24",
    }


def _runtime() -> dict[str, object]:
    return {
        "python": "test",
        "numpy": "test",
        "opencv": "test",
        "implementation_path": str(Path(diagnostics.__file__).resolve()),
        "implementation_sha256": hashlib.sha256(Path(diagnostics.__file__).read_bytes()).hexdigest(),
        "geometry_path": "test/geometry.py",
        "geometry_sha256": "a" * 64,
        "decoded_evaluator_path": "test/decoded_temporal_event_evaluator_v1.py",
        "decoded_evaluator_sha256": "d" * 64,
        "ffmpeg": {"path": "/test/ffmpeg", "sha256": "b" * 64, "version_line": "test"},
        "ffprobe": {"path": "/test/ffprobe", "sha256": "c" * 64, "version_line": "test"},
    }


class SAICExact81MediaDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "source.mp4"
        self.candidate = self.root / "candidate.mp4"
        self.source.write_bytes(b"source exact81 fixture")
        self.candidate.write_bytes(b"candidate exact81 fixture")
        self.source_sha = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.candidate_sha = hashlib.sha256(self.candidate.read_bytes()).hexdigest()
        self.patches = (
            mock.patch.object(diagnostics, "_runtime_identity", side_effect=_runtime),
            mock.patch.object(diagnostics, "_postflight_runtime_identity", return_value=None),
            mock.patch.object(diagnostics, "decode_exact81_rgb24", side_effect=_fake_decode),
            mock.patch.object(diagnostics, "analyze_video", side_effect=_fake_analysis),
        )
        for patch in self.patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in reversed(self.patches):
            patch.stop()
        self.temporary.cleanup()

    def _build(self) -> dict[str, object]:
        return diagnostics.build_diagnostic(
            source_video=self.source,
            expected_source_sha256=self.source_sha,
            candidate_video=self.candidate,
            expected_candidate_sha256=self.candidate_sha,
        )

    def test_real_inputs_yield_full81_full80_diagnostic_only_evidence(self) -> None:
        value = self._build()
        checked = diagnostics.validate_diagnostic(value)
        self.assertEqual(checked["media"]["source"]["decode"]["frame_count"], 81)
        self.assertEqual(len(checked["source"]["transition_descriptors"]), 80)
        self.assertEqual(len(checked["candidate"]["transition_descriptors"]), 80)
        self.assertEqual(
            len(checked["candidate"]["technical_full81"]["sharpness_by_frame"]), 81
        )
        self.assertEqual(
            len(checked["candidate"]["technical_full81"]["mean_absolute_step_by_transition"]),
            80,
        )
        self.assertGreater(
            checked["comparisons"]["camera_trajectory"][
                "global_mean_xy_l2_difference_mean"
            ],
            0.0,
        )
        for axis in diagnostics.UNAVAILABLE_AXES:
            self.assertEqual(checked["availability"][axis], "unavailable")
        self.assertFalse(checked["authority"]["measurement_runtime_qualified"])
        self.assertFalse(checked["authority"]["optimizer_step_allowed"])
        self.assertFalse(checked["input_closure"]["external_flow_read"])
        self.assertTrue(
            checked["input_closure"]["internally_computed_optical_flow_diagnostic_only"]
        )

    def test_absolute_regular_path_and_pinned_hash_are_required(self) -> None:
        with self.assertRaisesRegex(diagnostics.SAICExact81DiagnosticError, "absolute"):
            diagnostics.build_diagnostic(
                source_video="source.mp4",
                expected_source_sha256=self.source_sha,
                candidate_video=self.candidate,
                expected_candidate_sha256=self.candidate_sha,
            )
        link = self.root / "source-link.mp4"
        link.symlink_to(self.source)
        with self.assertRaisesRegex(diagnostics.SAICExact81DiagnosticError, "non-symlink"):
            diagnostics.build_diagnostic(
                source_video=link,
                expected_source_sha256=self.source_sha,
                candidate_video=self.candidate,
                expected_candidate_sha256=self.candidate_sha,
            )
        with self.assertRaisesRegex(diagnostics.SAICExact81DiagnosticError, "hash differs"):
            diagnostics.build_diagnostic(
                source_video=self.source,
                expected_source_sha256="0" * 64,
                candidate_video=self.candidate,
                expected_candidate_sha256=self.candidate_sha,
            )

    def test_exact81_decode_and_full80_geometry_are_both_required(self) -> None:
        bad_frames, bad_metadata = _fake_decode(
            self.source, ffmpeg="unused", ffprobe="unused"
        )
        bad_metadata = dict(bad_metadata)
        bad_metadata["frame_count"] = 80
        with mock.patch.object(
            diagnostics,
            "decode_exact81_rgb24",
            return_value=(bad_frames[:-1], bad_metadata),
        ):
            with self.assertRaisesRegex(
                diagnostics.SAICExact81DiagnosticError, "not exact81"
            ):
                self._build()

        dishonest_metadata = dict(_fake_decode(
            self.source, ffmpeg="unused", ffprobe="unused"
        )[1])
        dishonest_metadata["decoded_rgb24_sha256"] = "0" * 64
        with mock.patch.object(
            diagnostics,
            "decode_exact81_rgb24",
            return_value=(bad_frames, dishonest_metadata),
        ):
            with self.assertRaisesRegex(
                diagnostics.SAICExact81DiagnosticError,
                "hash differs from decoded bytes",
            ):
                self._build()

        analysis = _fake_analysis(self.source, SimpleNamespace(analysis_frames=81))
        analysis.metrics.values["sampled_frames"] = 80
        with mock.patch.object(diagnostics, "analyze_video", return_value=analysis):
            with self.assertRaisesRegex(
                diagnostics.SAICExact81DiagnosticError, "did not analyze exact81"
            ):
                self._build()

    def test_tamper_and_resign_cannot_survive_media_replay(self) -> None:
        value = self._build()
        tampered = deepcopy(value)
        tampered["comparisons"]["camera_trajectory"][
            "global_mean_xy_l2_difference_mean"
        ] = 0.0
        body = {
            key: item for key, item in tampered.items() if key != "diagnostic_digest"
        }
        tampered["diagnostic_digest"] = diagnostics.object_sha256(body)
        with self.assertRaisesRegex(
            diagnostics.SAICExact81DiagnosticError, "differs from media replay"
        ):
            diagnostics.validate_diagnostic(tampered)

        self.source.write_bytes(b"mutated after diagnostic")
        with self.assertRaisesRegex(diagnostics.SAICExact81DiagnosticError, "hash differs"):
            diagnostics.validate_diagnostic(value)

    def test_no_resigned_caller_boolean_can_grant_optimizer_authority(self) -> None:
        value = self._build()
        tampered = deepcopy(value)
        tampered["authority"]["measurement_runtime_qualified"] = True
        tampered["authority"]["optimizer_step_allowed"] = True
        body = {
            key: item for key, item in tampered.items() if key != "diagnostic_digest"
        }
        tampered["diagnostic_digest"] = diagnostics.object_sha256(body)
        with self.assertRaisesRegex(
            diagnostics.SAICExact81DiagnosticError, "cannot acquire authority"
        ):
            diagnostics.validate_diagnostic(tampered)

    def test_rebinding_public_contract_globals_cannot_weaken_build_or_replay(self) -> None:
        value = self._build()
        hostile_authority = {
            "measurement_runtime_qualified": True,
            "candidate_selection_allowed": True,
            "training_allowed": True,
            "optimizer_step_allowed": True,
            "absolute_action_editing_success_claimed": True,
        }
        hostile_input = {
            "source_video_read": False,
            "candidate_video_read": False,
            "external_mask_read": True,
        }
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(diagnostics, "AUTHORITY", hostile_authority)
            )
            stack.enter_context(
                mock.patch.object(diagnostics, "INPUT_CLOSURE", hostile_input)
            )
            stack.enter_context(mock.patch.object(diagnostics, "UNAVAILABLE_AXES", ()))
            stack.enter_context(
                mock.patch.object(
                    diagnostics,
                "DIAGNOSTIC_AXES",
                ("identity", "event", "source_bind"),
                )
            )
            stack.enter_context(mock.patch.object(diagnostics, "SCHEMA_VERSION", "hostile-v9"))
            stack.enter_context(mock.patch.object(diagnostics, "FRAME_COUNT", 1))
            stack.enter_context(mock.patch.object(diagnostics, "FPS", 1))
            stack.enter_context(mock.patch.object(diagnostics, "TRANSITION_COUNT", 0))
            rebuilt = self._build()
            checked = diagnostics.validate_diagnostic(value)
            checked_rebuilt = diagnostics.validate_diagnostic(rebuilt)

        for checked_value in (checked, checked_rebuilt):
            self.assertFalse(checked_value["authority"]["optimizer_step_allowed"])
            self.assertFalse(checked_value["authority"]["training_allowed"])
            self.assertEqual(checked_value["availability"]["identity"], "unavailable")
            self.assertEqual(checked_value["availability"]["event"], "unavailable")
            self.assertFalse(checked_value["input_closure"]["external_mask_read"])
            self.assertEqual(
                checked_value["schema_version"],
                "bernini-saic-exact81-media-diagnostics-v1",
            )
            self.assertEqual(checked_value["media"]["source"]["decode"]["frame_count"], 81)
            self.assertEqual(len(checked_value["source"]["transition_descriptors"]), 80)

    def test_measurement_uses_hash_verified_private_snapshots(self) -> None:
        observed_paths: list[Path] = []

        def observe_decode(path: str | Path, *, ffmpeg: str, ffprobe: str):
            observed_paths.append(Path(path))
            self.assertNotEqual(Path(path), self.source)
            self.assertNotEqual(Path(path), self.candidate)
            return _fake_decode(path, ffmpeg=ffmpeg, ffprobe=ffprobe)

        def observe_analysis(path: str | Path, config: object) -> object:
            observed_paths.append(Path(path))
            self.assertNotEqual(Path(path), self.source)
            self.assertNotEqual(Path(path), self.candidate)
            return _fake_analysis(path, config)

        with mock.patch.object(
            diagnostics, "decode_exact81_rgb24", side_effect=observe_decode
        ), mock.patch.object(
            diagnostics, "analyze_video", side_effect=observe_analysis
        ):
            value = self._build()
        self.assertEqual(len(observed_paths), 4)
        self.assertTrue(all("saic-exact81-media-" in str(path) for path in observed_paths))
        self.assertEqual(value["media"]["source"]["path"], str(self.source))
        self.assertEqual(value["media"]["candidate"]["path"], str(self.candidate))

    def test_media_replacement_during_measurement_is_rejected_postflight(self) -> None:
        replaced = False

        def replace_during_decode(path: str | Path, *, ffmpeg: str, ffprobe: str):
            nonlocal replaced
            if not replaced:
                replaced = True
                self.source.write_bytes(b"hostile concurrent replacement")
            return _fake_decode(path, ffmpeg=ffmpeg, ffprobe=ffprobe)

        with mock.patch.object(
            diagnostics,
            "decode_exact81_rgb24",
            side_effect=replace_during_decode,
        ):
            with self.assertRaisesRegex(
                diagnostics.SAICExact81DiagnosticError,
                "source video postflight hash differs",
            ):
                self._build()

    def test_bool_numeric_geometry_metric_is_rejected(self) -> None:
        analysis = _fake_analysis(self.source, SimpleNamespace(analysis_frames=81))
        analysis.metrics.values["scene_cut_ratio"] = True
        with mock.patch.object(diagnostics, "analyze_video", return_value=analysis):
            with self.assertRaisesRegex(
                diagnostics.SAICExact81DiagnosticError, "numeric, not bool/text"
            ):
                self._build()

    def test_decoder_implementation_is_bound_and_replayed(self) -> None:
        value = self._build()
        self.assertEqual(
            value["runtime"]["decoded_evaluator_sha256"], "d" * 64
        )
        tampered = deepcopy(value)
        tampered["runtime"]["decoded_evaluator_sha256"] = "e" * 64
        body = {
            key: item for key, item in tampered.items() if key != "diagnostic_digest"
        }
        tampered["diagnostic_digest"] = diagnostics.object_sha256(body)
        with self.assertRaisesRegex(
            diagnostics.SAICExact81DiagnosticError, "differs from media replay"
        ):
            diagnostics.validate_diagnostic(tampered)

    def test_numeric_zero_cannot_replace_false_authority_even_if_resigned(self) -> None:
        value = self._build()
        tampered = deepcopy(value)
        tampered["authority"]["optimizer_step_allowed"] = 0
        body = {
            key: item for key, item in tampered.items() if key != "diagnostic_digest"
        }
        tampered["diagnostic_digest"] = diagnostics.object_sha256(body)
        with self.assertRaisesRegex(
            diagnostics.SAICExact81DiagnosticError, "cannot acquire authority"
        ):
            diagnostics.validate_diagnostic(tampered)

    def test_create_only_file_is_canonical_and_read_only(self) -> None:
        value = self._build()
        hostile = deepcopy(value)
        hostile["authority"]["optimizer_step_allowed"] = True
        hostile_body = {
            key: item for key, item in hostile.items() if key != "diagnostic_digest"
        }
        hostile["diagnostic_digest"] = diagnostics.object_sha256(hostile_body)
        with self.assertRaisesRegex(
            diagnostics.SAICExact81DiagnosticError, "cannot acquire authority"
        ):
            diagnostics.write_diagnostic_create_only(
                self.root / "hostile-diagnostic.json", hostile
            )
        self.assertFalse((self.root / "hostile-diagnostic.json").exists())

        output = self.root / "diagnostic.json"
        diagnostics.write_diagnostic_create_only(output, value)
        self.assertEqual(
            output.read_bytes(), diagnostics.canonical_json_bytes(value) + b"\n"
        )
        self.assertEqual(os.stat(output).st_mode & 0o777, 0o400)
        loaded = diagnostics.load_canonical_diagnostic(output)
        self.assertEqual(loaded, json.loads(diagnostics.canonical_json_bytes(value)))
        with self.assertRaisesRegex(
            diagnostics.SAICExact81DiagnosticError, "refusing to overwrite"
        ):
            diagnostics.write_diagnostic_create_only(output, value)


if __name__ == "__main__":
    unittest.main()
