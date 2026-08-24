from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from motive.r7_temporal_teacher import (
    LazyCoTrackerAdapter,
    TemporalTeacherConfig,
    TemporalTeacherError,
    TrackObservations,
    build_temporal_teacher,
    build_temporal_teacher_with_stability,
    deterministic_track_time_perturbation,
    robust_camera_compensation,
    select_event_window,
)


def _synthetic_tracks(
    *,
    actor_motion: bool = True,
    camera_motion: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Camera-affine sequence with eight coherent actor outliers."""

    frames = 24
    tracks = 40
    actor_tracks = 8
    frame_size = (120, 180)
    rng = np.random.default_rng(260108828)
    reference = np.stack(
        (
            rng.uniform(0.10, 0.82, size=tracks),
            rng.uniform(0.10, 0.82, size=tracks),
        ),
        axis=1,
    )
    actor_offset = np.zeros((frames, 2), dtype=np.float64)
    if actor_motion:
        for frame_index in range(1, frames):
            actor_offset[frame_index] = actor_offset[frame_index - 1]
            if 7 <= frame_index <= 15:
                actor_offset[frame_index, 0] += 0.012
                actor_offset[frame_index, 1] += 0.002

    normalized = np.empty((frames, tracks, 2), dtype=np.float64)
    center = np.asarray([0.5, 0.5])
    for frame_index in range(frames):
        angle = 0.006 * frame_index if camera_motion else 0.0
        scale = 1.0 + 0.0015 * frame_index if camera_motion else 1.0
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        content = reference.copy()
        content[-actor_tracks:] += actor_offset[frame_index]
        translation = (
            np.asarray([0.005 * frame_index, 0.002 * frame_index])
            if camera_motion
            else np.zeros(2, dtype=np.float64)
        )
        normalized[frame_index] = (
            (content - center) @ (scale * rotation).T
            + center
            + translation
        )
    pixel_tracks = normalized * np.asarray(
        [frame_size[1], frame_size[0]],
        dtype=np.float64,
    )
    visibility = np.ones((frames, tracks), dtype=np.float32)
    frame_times = np.arange(frames, dtype=np.float64) / 10.0
    return (
        pixel_tracks.astype(np.float32),
        visibility,
        frame_times,
        frame_size,
    )


def _config(**overrides: object) -> TemporalTeacherConfig:
    values: dict[str, object] = {
        "minimum_actor_speed": 0.02,
        "camera_inlier_threshold": 0.003,
        "perturb_coordinate_jitter": 0.00005,
    }
    values.update(overrides)
    return TemporalTeacherConfig(**values)


class ArrayContractTests(unittest.TestCase):
    def test_track_observations_reject_bad_time_and_nonfinite_values(self) -> None:
        tracks, visibility, times, size = _synthetic_tracks()
        bad_times = times.copy()
        bad_times[5] = bad_times[4]
        with self.assertRaisesRegex(
            TemporalTeacherError,
            "invalid_frame_times",
        ):
            TrackObservations.create(
                tracks=tracks,
                visibility=visibility,
                frame_times=bad_times,
                frame_size=size,
                backend="synthetic",
            )
        tracks[0, 0, 0] = np.nan
        with self.assertRaisesRegex(
            TemporalTeacherError,
            "non_finite_input",
        ):
            build_temporal_teacher(
                tracks,
                visibility,
                times,
                size,
                config=_config(),
            )

    def test_schema_fixes_trajectory_to_32_phases(self) -> None:
        with self.assertRaisesRegex(TemporalTeacherError, "schema-fixed"):
            TemporalTeacherConfig(phase_steps=16).validate()

    def test_static_tracks_fail_closed_instead_of_emitting_a_token(self) -> None:
        tracks, visibility, times, size = _synthetic_tracks(
            actor_motion=False
        )
        with self.assertRaisesRegex(
            TemporalTeacherError,
            "no_actor_tracks",
        ):
            build_temporal_teacher(
                tracks,
                visibility,
                times,
                size,
                config=_config(),
            )

    def test_missing_visibility_fails_camera_estimation(self) -> None:
        tracks, visibility, times, size = _synthetic_tracks()
        visibility[10:12] = 0.0
        with self.assertRaisesRegex(
            TemporalTeacherError,
            "insufficient_camera_tracks",
        ):
            robust_camera_compensation(
                tracks,
                visibility,
                size,
                config=_config(),
            )


class TemporalTeacherTests(unittest.TestCase):
    def test_robust_affine_rejects_actor_outliers(self) -> None:
        tracks, visibility, _, size = _synthetic_tracks()
        compensation = robust_camera_compensation(
            tracks,
            visibility,
            size,
            config=_config(),
        )
        self.assertEqual(compensation.transition_affines.shape, (23, 2, 3))
        self.assertEqual(compensation.cumulative_affines.shape, (24, 2, 3))
        self.assertGreater(compensation.background_residual_reduction, 0.95)
        self.assertGreater(compensation.camera_explained_ratio, 0.99)
        self.assertTrue(compensation.crossfit_valid)
        self.assertGreater(compensation.crossfit_raw_median, 0.002)
        self.assertGreater(
            compensation.crossfit_residual_reduction,
            0.95,
        )
        # During the action, the eight moving points are excluded from the
        # majority-camera fit.
        self.assertLess(
            float(compensation.transition_inlier_fraction[8]),
            0.90,
        )

    def test_static_camera_cannot_claim_perfect_crossfit_reduction(self) -> None:
        tracks, visibility, _, size = _synthetic_tracks(
            camera_motion=False
        )
        compensation = robust_camera_compensation(
            tracks,
            visibility,
            size,
            config=_config(),
        )
        self.assertTrue(compensation.crossfit_valid)
        self.assertLess(compensation.crossfit_raw_median, 1e-7)
        self.assertEqual(compensation.crossfit_residual_reduction, 0.0)

    def test_event_window_is_continuous_and_energy_localized(self) -> None:
        times = np.arange(21, dtype=np.float64) / 10.0
        energy = np.zeros(20, dtype=np.float64)
        energy[6:13] = 1.0
        event = select_event_window(energy, times, config=_config())
        self.assertLessEqual(event.transition_start, 6)
        self.assertGreaterEqual(event.transition_stop, 13)
        self.assertLess(event.transition_stop - event.transition_start, 15)
        self.assertGreaterEqual(event.captured_energy_fraction, 0.85)
        self.assertEqual(event.frame_stop, event.transition_stop + 1)

    def test_teacher_emits_factorized_32_phase_trajectory(self) -> None:
        tracks, visibility, times, size = _synthetic_tracks()
        teacher = build_temporal_teacher(
            tracks,
            visibility,
            times,
            size,
            config=_config(),
        )
        self.assertEqual(teacher.actor_trajectory.shape, (32, 2))
        self.assertEqual(teacher.actor_velocity.shape, (32, 2))
        self.assertEqual(teacher.actor_acceleration.shape, (32, 2))
        self.assertEqual(teacher.camera_trajectory.shape, (32, 4))
        self.assertEqual(teacher.actor_track_trajectories.shape, (8, 32, 2))
        self.assertEqual(teacher.phase_energy.shape, (32,))
        self.assertEqual(teacher.embedding().shape, (224,))
        self.assertEqual(set(teacher.active_track_indices), set(range(32, 40)))
        self.assertGreater(float(teacher.actor_trajectory[-1, 0]), 0.08)
        self.assertGreater(teacher.mean_visibility, 0.99)
        self.assertLessEqual(teacher.event_window.transition_start, 7)
        self.assertGreaterEqual(teacher.event_window.transition_stop, 15)

    def test_perturbation_is_deterministic_and_stable(self) -> None:
        tracks, visibility, times, size = _synthetic_tracks()
        first = deterministic_track_time_perturbation(
            tracks,
            visibility,
            times,
            size,
            seed=17,
            config=_config(),
        )
        second = deterministic_track_time_perturbation(
            tracks,
            visibility,
            times,
            size,
            seed=17,
            config=_config(),
        )
        for first_array, second_array in zip(first, second):
            np.testing.assert_array_equal(first_array, second_array)

        result = build_temporal_teacher_with_stability(
            tracks,
            visibility,
            times,
            size,
            seed=1,
            audit_seed=2,
            config=_config(),
        )
        self.assertTrue(result.diagnostic_ready)
        self.assertIsNotNone(result.perturbed)
        self.assertIsNotNone(result.stability)
        assert result.stability is not None
        self.assertGreaterEqual(result.stability.event_window_iou, 0.70)
        self.assertGreaterEqual(result.stability.embedding_cosine, 0.85)
        self.assertTrue(result.stability.passed)
        self.assertTrue(result.audit_available)
        self.assertTrue(result.audit_passed)
        self.assertIsNotNone(result.audit_stability)
        assert result.audit_stability is not None
        self.assertLessEqual(
            result.audit_stability.event_duration_relative_error,
            0.10,
        )
        self.assertLessEqual(
            result.audit_stability.embedding_norm_relative_error,
            0.10,
        )
        self.assertLessEqual(result.audit_stability.trajectory_rmse, 0.01)

    def test_failed_perturbation_cannot_pass_stability_gate(self) -> None:
        tracks, visibility, times, size = _synthetic_tracks()
        # Dropping half of the full-grid tracks is deliberately destructive:
        # some perturbations leave too little background/actor separation.
        result = build_temporal_teacher_with_stability(
            tracks,
            visibility,
            times,
            size,
            seed=4,
            config=_config(
                perturb_track_drop_fraction=0.50,
                perturb_visibility_drop_fraction=0.20,
            ),
        )
        if result.perturbed is None:
            self.assertFalse(result.diagnostic_ready)
            self.assertIsNotNone(result.failure_reason)
        else:
            self.assertEqual(
                result.diagnostic_ready,
                bool(result.stability and result.stability.passed),
            )

    def test_independent_audit_runs_after_screening_rejection(self) -> None:
        tracks, visibility, times, size = _synthetic_tracks()
        result = build_temporal_teacher_with_stability(
            tracks,
            visibility,
            times,
            size,
            seed=17,
            audit_seed=2,
            config=_config(),
        )
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(result.failure_reason, "unstable_teacher")
        self.assertTrue(result.audit_available)
        self.assertTrue(result.audit_passed)


class LazyAdapterTests(unittest.TestCase):
    def test_constructor_and_preflight_validation_do_not_import_torch(self) -> None:
        with patch(
            "motive.r7_temporal_teacher.importlib.import_module"
        ) as importer:
            adapter = LazyCoTrackerAdapter(device="cpu", grid_size=8)
            self.assertFalse(adapter.loaded)
            self.assertFalse(adapter.backward_tracking)
            importer.assert_not_called()
            with self.assertRaisesRegex(TemporalTeacherError, "invalid_shape"):
                adapter.track(np.zeros((4, 16, 16), dtype=np.uint8))
            importer.assert_not_called()

    def test_predictor_receives_explicit_zero_query_frame(self) -> None:
        class FakeVideo:
            def permute(self, *args):
                return self

            def __getitem__(self, key):
                return self

            def float(self):
                return self

            def to(self, device):
                return self

        class FakeOutput:
            def __init__(self, value):
                self.value = value

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class FakeContext:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, traceback):
                return False

        class FakeTorch:
            @staticmethod
            def from_numpy(value):
                return FakeVideo()

            @staticmethod
            def inference_mode():
                return FakeContext()

        class FakePredictor:
            def __init__(self):
                self.kwargs = None

            def __call__(self, video, **kwargs):
                self.kwargs = kwargs
                return (
                    FakeOutput(np.zeros((1, 4, 64, 2), dtype=np.float32)),
                    FakeOutput(np.ones((1, 4, 64), dtype=np.float32)),
                )

        predictor = FakePredictor()
        adapter = LazyCoTrackerAdapter(device="cpu", grid_size=8)
        adapter._torch = FakeTorch()
        adapter._predictor = predictor
        result = adapter.track(
            np.zeros((4, 16, 16, 3), dtype=np.uint8),
            frame_times=np.arange(4, dtype=np.float64),
        )
        self.assertEqual(
            predictor.kwargs,
            {
                "grid_size": 8,
                "grid_query_frame": 0,
                "backward_tracking": False,
            },
        )
        self.assertEqual(result.provenance["query_frame"], 0)


if __name__ == "__main__":
    unittest.main()
