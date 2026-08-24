from __future__ import annotations

import itertools
import unittest

import numpy as np

from motive.r8_stable_motion import (
    R8_SMOOTHED_CENTER_DEFAULT_DELTA,
    R8_SMOOTHED_CENTER_MAX_ITERATIONS,
    R8_SMOOTHED_CENTER_MAX_NORMALIZED_VALUE,
    R8_SMOOTHED_CENTER_MIN_NORMALIZED_VALUE,
    R8_STABLE_MOTION_SCHEMA,
    R8_STABLE_MOTION_SHAPE_DIM,
    StableMotionConfig,
    build_stable_motion_representation,
)


PHASES = 32


def _static_tracks(offsets: np.ndarray) -> np.ndarray:
    values = np.asarray(offsets, dtype=np.float64)
    return np.repeat(values[:, None, :], PHASES, axis=1)


def _full_mask(track_count: int) -> np.ndarray:
    return np.ones((track_count, PHASES), dtype=bool)


class StableSetAggregationTests(unittest.TestCase):
    def test_permutation_preserves_all_canonical_outputs(self) -> None:
        offsets = np.asarray(
            [
                [-0.30, -0.10],
                [-0.10, 0.20],
                [0.10, -0.20],
                [0.30, 0.10],
                [0.05, 0.35],
                [-0.05, -0.35],
            ],
            dtype=np.float64,
        )
        tracks = _static_tracks(offsets)
        phase = np.arange(PHASES, dtype=np.float64)
        translation = np.stack((0.004 * phase, -0.002 * phase), axis=1)
        tracks += translation[None]
        tracks[:3, :, 0] += 0.0004 * phase[None] ** 2
        tracks[3:, :, 0] -= 0.0004 * phase[None] ** 2
        mask = _full_mask(len(tracks))
        energy = np.linspace(0.0, 0.25, PHASES, dtype=np.float32)
        membership = np.asarray([11, 4, 29, 7, 18, 2], dtype=np.int64)

        baseline = build_stable_motion_representation(
            tracks,
            mask,
            energy,
            component_track_indices=membership,
        )
        permutation = np.asarray([3, 0, 5, 2, 1, 4])
        permuted = build_stable_motion_representation(
            tracks[permutation],
            mask[permutation],
            energy,
            component_track_indices=membership[permutation],
        )
        self.assertTrue(baseline.diagnostic_ready)
        self.assertTrue(permuted.diagnostic_ready)
        np.testing.assert_array_equal(
            permuted.component_track_indices,
            baseline.component_track_indices,
        )
        np.testing.assert_allclose(
            permuted.trajectory,
            baseline.trajectory,
            atol=0.0,
        )
        np.testing.assert_allclose(
            permuted.shape_tokens,
            baseline.shape_tokens,
            atol=0.0,
        )
        np.testing.assert_allclose(
            permuted.transition_displacement,
            baseline.transition_displacement,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            permuted.transition_support_count,
            baseline.transition_support_count,
        )
        np.testing.assert_allclose(
            permuted.anchored_track_trajectories,
            baseline.anchored_track_trajectories,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            permuted.track_phase_mask,
            baseline.track_phase_mask,
        )

    def test_static_shape_offsets_do_not_become_membership_motion(
        self,
    ) -> None:
        offsets = np.asarray(
            [
                [-0.60, 0.00],
                [-0.30, 0.20],
                [0.00, -0.20],
                [0.30, 0.10],
                [0.90, -0.10],
            ]
        )
        tracks = _static_tracks(offsets)
        mask = _full_mask(len(tracks))
        # Removing the two left tracks temporarily shifts an absolute-position
        # median even though every physical track is static.
        mask[:2, 12:20] = False
        naive = np.zeros((PHASES, 2), dtype=np.float64)
        for phase in range(PHASES):
            naive[phase] = np.median(
                tracks[mask[:, phase], phase],
                axis=0,
            )
        naive -= naive[0]
        self.assertGreater(float(np.max(np.abs(naive))), 0.20)

        result = build_stable_motion_representation(
            tracks,
            mask,
            np.zeros(PHASES, dtype=np.float32),
            component_track_indices=np.arange(100, 105),
        )
        self.assertTrue(result.diagnostic_ready, result.failure_detail)
        np.testing.assert_allclose(result.trajectory, 0.0, atol=0.0)
        np.testing.assert_allclose(result.shape_tokens, 0.0, atol=0.0)
        np.testing.assert_allclose(result.phase_support[12:20], 0.60)
        self.assertTrue((~result.track_phase_mask[:2, 12:20]).all())

    def test_whole_track_dropout_keeps_static_motion_zero(self) -> None:
        offsets = np.asarray(
            [
                [-0.50, 0.00],
                [-0.10, 0.20],
                [0.20, -0.10],
                [0.60, 0.10],
                [0.90, -0.20],
            ]
        )
        phase = np.arange(PHASES, dtype=np.float64)
        translation = np.stack(
            (
                0.003 * phase + 0.0001 * phase**2,
                -0.002 * phase,
            ),
            axis=1,
        )
        full = build_stable_motion_representation(
            _static_tracks(offsets) + translation[None],
            _full_mask(5),
            np.zeros(PHASES),
            component_track_indices=np.arange(5),
        )
        keep = np.asarray([0, 2, 4])
        dropped = build_stable_motion_representation(
            _static_tracks(offsets[keep]) + translation[None],
            _full_mask(3),
            np.zeros(PHASES),
            component_track_indices=keep,
        )
        self.assertTrue(full.diagnostic_ready)
        self.assertTrue(dropped.diagnostic_ready)
        np.testing.assert_allclose(
            full.trajectory,
            translation,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            dropped.trajectory,
            translation,
            atol=1e-7,
        )
        np.testing.assert_allclose(full.shape_tokens, 0.0, atol=1e-12)
        np.testing.assert_allclose(dropped.shape_tokens, 0.0, atol=1e-12)

    def test_first_valid_anchor_and_visibility_hole_are_preserved(self) -> None:
        offsets = np.asarray(
            [
                [-0.30, 0.10],
                [-0.10, -0.10],
                [0.10, 0.20],
                [0.30, -0.20],
            ]
        )
        tracks = _static_tracks(offsets)
        mask = _full_mask(4)
        mask[0, :5] = False
        mask[1, 13:17] = False
        result = build_stable_motion_representation(
            tracks,
            mask,
            np.zeros(PHASES),
            component_track_indices=np.asarray([9, 3, 8, 1]),
        )
        self.assertTrue(result.diagnostic_ready, result.failure_detail)
        np.testing.assert_allclose(result.trajectory, 0.0, atol=0.0)
        np.testing.assert_allclose(result.shape_tokens, 0.0, atol=0.0)
        # Canonical membership order is [1,3,8,9]; track 9 starts at phase 5.
        position = int(
            np.flatnonzero(result.component_track_indices == 9)[0]
        )
        self.assertEqual(int(result.track_anchor_phase[position]), 5)
        self.assertTrue((~result.track_phase_mask[position, :5]).all())
        hole_position = int(
            np.flatnonzero(result.component_track_indices == 3)[0]
        )
        self.assertTrue(
            (~result.track_phase_mask[hole_position, 13:17]).all()
        )

    def test_changing_first_visible_anchor_does_not_change_translation(
        self,
    ) -> None:
        offsets = np.asarray(
            [
                [-0.4, 0.0],
                [-0.2, 0.1],
                [0.0, -0.1],
                [0.2, 0.1],
                [0.4, -0.1],
            ]
        )
        phase = np.arange(PHASES, dtype=np.float64)
        translation = np.stack(
            (0.004 * phase, 0.002 * np.sin(phase / 5.0)),
            axis=1,
        )
        tracks = _static_tracks(offsets) + translation[None]
        full_mask = _full_mask(5)
        late_mask = full_mask.copy()
        late_mask[0, :6] = False
        baseline = build_stable_motion_representation(
            tracks,
            full_mask,
            np.zeros(PHASES),
            component_track_indices=np.arange(5),
        )
        late = build_stable_motion_representation(
            tracks,
            late_mask,
            np.zeros(PHASES),
            component_track_indices=np.arange(5),
        )
        self.assertTrue(baseline.diagnostic_ready)
        self.assertTrue(late.diagnostic_ready)
        np.testing.assert_allclose(
            late.trajectory,
            baseline.trajectory,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            late.trajectory,
            translation,
            atol=1e-7,
        )
        position = int(
            np.flatnonzero(late.component_track_indices == 0)[0]
        )
        self.assertEqual(int(late.track_anchor_phase[position]), 6)
        self.assertEqual(int(late.transition_support_count[0]), 0)
        self.assertTrue(
            (late.transition_support_count[1:7] == 4).all()
        )
        self.assertTrue(
            (late.transition_support_count[7:] == 5).all()
        )


class MotionModeTests(unittest.TestCase):
    def test_smoothed_center_handles_numeric_boundaries(
        self,
    ) -> None:
        cases = (
            (
                "near_collinear_vertex",
                np.asarray(
                    [
                        [-0.0090, -0.000006],
                        [-0.0013, -0.000003],
                        [0.0058, 0.000010],
                        [0.0089, -0.000008],
                    ],
                    dtype=np.float64,
                ),
            ),
            (
                "near_coincident_duplicate",
                np.asarray(
                    [
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [1e-10, -2e-10],
                        [0.01, 0.0],
                    ],
                    dtype=np.float64,
                ),
            ),
        )
        membership = np.asarray([17, 2, 11, 5], dtype=np.int64)
        phase = np.arange(PHASES, dtype=np.float64)
        angle = 0.713
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        for name, transition_vectors in cases:
            with self.subTest(case=name):
                tracks = (
                    transition_vectors[:, None, :]
                    * phase[None, :, None]
                )
                baseline = build_stable_motion_representation(
                    tracks,
                    _full_mask(4),
                    np.zeros(PHASES),
                    component_track_indices=membership,
                )
                self.assertTrue(
                    baseline.diagnostic_ready,
                    baseline.failure_detail,
                )
                for raw_permutation in itertools.permutations(range(4)):
                    permutation = np.asarray(raw_permutation)
                    permuted = build_stable_motion_representation(
                        tracks[permutation],
                        _full_mask(4)[permutation],
                        np.zeros(PHASES),
                        component_track_indices=membership[permutation],
                    )
                    self.assertTrue(
                        permuted.diagnostic_ready,
                        permuted.failure_detail,
                    )
                    np.testing.assert_array_equal(
                        permuted.transition_displacement,
                        baseline.transition_displacement,
                    )
                    np.testing.assert_array_equal(
                        permuted.trajectory,
                        baseline.trajectory,
                    )
                rotated = build_stable_motion_representation(
                    tracks @ rotation.T,
                    _full_mask(4),
                    np.zeros(PHASES),
                    component_track_indices=membership,
                )
                self.assertTrue(
                    rotated.diagnostic_ready,
                    rotated.failure_detail,
                )
                np.testing.assert_allclose(
                    rotated.transition_displacement,
                    baseline.transition_displacement @ rotation.T,
                    atol=2e-8,
                )
                np.testing.assert_allclose(
                    rotated.trajectory,
                    baseline.trajectory @ rotation.T,
                    atol=2e-8,
                )

    def test_smoothed_center_makes_collinear_case_unique(
        self,
    ) -> None:
        transition_vectors = np.asarray(
            [
                [-0.0090, 0.0],
                [-0.0013, 0.0],
                [0.0058, 0.0],
                [0.0089, 0.0],
            ],
            dtype=np.float64,
        )
        phase = np.arange(PHASES, dtype=np.float64)
        tracks = (
            transition_vectors[:, None, :]
            * phase[None, :, None]
        )
        angle = 0.713
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        baseline = build_stable_motion_representation(
            tracks,
            _full_mask(4),
            np.zeros(PHASES),
            component_track_indices=np.arange(4),
        )
        self.assertTrue(baseline.diagnostic_ready, baseline.failure_detail)
        for raw_permutation in itertools.permutations(range(4)):
            permutation = np.asarray(raw_permutation)
            permuted = build_stable_motion_representation(
                tracks[permutation],
                _full_mask(4)[permutation],
                np.zeros(PHASES),
                component_track_indices=permutation,
            )
            self.assertTrue(permuted.diagnostic_ready)
            np.testing.assert_array_equal(
                permuted.transition_displacement,
                baseline.transition_displacement,
            )
        rotated = build_stable_motion_representation(
            tracks @ rotation.T,
            _full_mask(4),
            np.zeros(PHASES),
            component_track_indices=np.arange(4),
        )
        self.assertTrue(rotated.diagnostic_ready, rotated.failure_detail)
        np.testing.assert_allclose(
            rotated.transition_displacement,
            baseline.transition_displacement @ rotation.T,
            atol=2e-8,
        )

    def test_nonvertex_iteration_exhaustion_still_fails_closed(
        self,
    ) -> None:
        transition_vectors = np.asarray(
            [
                [-0.4015348829485148, -0.42530156214896514],
                [-0.5065298585202478, -0.5365111457334341],
                [0.00017850547013013911, 0.00018907113310057789],
                [0.06941926925112026, 0.07352816631718363],
            ],
            dtype=np.float64,
        )
        tracks = (
            transition_vectors[:, None, :]
            * np.arange(PHASES, dtype=np.float64)[None, :, None]
        )
        result = build_stable_motion_representation(
            tracks,
            _full_mask(4),
            np.zeros(PHASES),
            config=StableMotionConfig(
                smoothed_center_max_iterations=8,
            ),
        )
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(
            result.failure_reason,
            "smoothed_median_uncertified",
        )

    def test_smoothed_center_replaces_false_vertex_certificate(
        self,
    ) -> None:
        # Synthetic convex quadrilateral.  Its unique median is the diagonal
        # intersection (-0.001, 0), but a 1e-9 relaxed KKT test incorrectly
        # certified the nearby input vertex (-0.003, 1.2e-7).
        transition_vectors = 0.01 * np.asarray(
            [
                [-1.0, 0.0],
                [-0.3, 1.2e-5],
                [0.3, -2.4e-5],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        )
        tracks = (
            transition_vectors[:, None, :]
            * np.arange(PHASES, dtype=np.float64)[None, :, None]
        )
        angle = 0.713
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        baseline = build_stable_motion_representation(
            tracks,
            _full_mask(4),
            np.zeros(PHASES),
            component_track_indices=np.arange(4),
        )
        self.assertTrue(baseline.diagnostic_ready, baseline.failure_detail)
        for raw_permutation in itertools.permutations(range(4)):
            permutation = np.asarray(raw_permutation)
            permuted = build_stable_motion_representation(
                tracks[permutation],
                _full_mask(4)[permutation],
                np.zeros(PHASES),
                component_track_indices=permutation,
            )
            self.assertTrue(permuted.diagnostic_ready)
            np.testing.assert_array_equal(
                permuted.transition_displacement,
                baseline.transition_displacement,
            )
        rotated = build_stable_motion_representation(
            tracks @ rotation.T,
            _full_mask(4),
            np.zeros(PHASES),
            component_track_indices=np.arange(4),
        )
        self.assertTrue(rotated.diagnostic_ready, rotated.failure_detail)
        np.testing.assert_allclose(
            rotated.transition_displacement,
            baseline.transition_displacement @ rotation.T,
            atol=2e-8,
        )

    def test_near_collinear_midpoint_false_return_fails_closed(
        self,
    ) -> None:
        # Fixed synthetic regression (seed 731992, trial 29884).  Numerical
        # rank regularization returned a midpoint 0.292 away from the unique
        # segment-intersection median, despite a tiny objective gap.
        transition_vectors = np.asarray(
            [
                [-0.4015348829485148, -0.42530156214896514],
                [-0.5065298585202478, -0.5365111457334341],
                [0.00017850547013013911, 0.00018907113310057789],
                [0.06941926925112026, 0.07352816631718363],
            ],
            dtype=np.float64,
        )
        tracks = (
            transition_vectors[:, None, :]
            * np.arange(PHASES, dtype=np.float64)[None, :, None]
        )
        angle = 0.713
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        for coordinate_transform in (np.eye(2), rotation):
            transformed = tracks @ coordinate_transform.T
            for raw_permutation in itertools.permutations(range(4)):
                permutation = np.asarray(raw_permutation)
                result = build_stable_motion_representation(
                    transformed[permutation],
                    _full_mask(4)[permutation],
                    np.zeros(PHASES),
                    component_track_indices=permutation,
                )
                self.assertFalse(result.diagnostic_ready)
                self.assertEqual(
                    result.failure_reason,
                    "smoothed_median_uncertified",
                )

    def test_step_small_false_return_is_replaced_by_certificate(
        self,
    ) -> None:
        transition_vectors = np.asarray(
            [
                [-3.7508392198767112e-12, -1.4718187085490178e-11],
                [-5.722772347372722e-06, -2.245603195589704e-05],
                [-2.1673997240806482e-07, -8.491642942653873e-07],
                [4.290151125575851e-05, 1.6834431760401377e-04],
                [8.23572654602769e-06, 3.231680858011634e-05],
                [9.047763479003036e-06, 3.544818061922885e-05],
            ],
            dtype=np.float64,
        )
        phase = np.arange(PHASES, dtype=np.float64)
        tracks = (
            transition_vectors[:, None, :]
            * phase[None, :, None]
        )
        baseline = build_stable_motion_representation(
            tracks,
            _full_mask(6),
            np.zeros(PHASES),
            component_track_indices=np.arange(6),
        )
        permutation = np.asarray([4, 1, 5, 0, 3, 2])
        permuted = build_stable_motion_representation(
            tracks[permutation],
            _full_mask(6)[permutation],
            np.zeros(PHASES),
            component_track_indices=permutation,
        )
        angle = 0.713
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        rotated = build_stable_motion_representation(
            tracks @ rotation.T,
            _full_mask(6),
            np.zeros(PHASES),
            component_track_indices=np.arange(6),
        )
        self.assertTrue(baseline.diagnostic_ready, baseline.failure_detail)
        self.assertTrue(permuted.diagnostic_ready, permuted.failure_detail)
        self.assertTrue(rotated.diagnostic_ready, rotated.failure_detail)
        np.testing.assert_array_equal(
            permuted.transition_displacement,
            baseline.transition_displacement,
        )
        np.testing.assert_allclose(
            rotated.transition_displacement,
            baseline.transition_displacement @ rotation.T,
            atol=2e-8,
        )
        self.assertTrue(
            (
                baseline.center_position_error_upper_bound[1:]
                <= StableMotionConfig().smoothed_center_absolute_position_tolerance
            ).all()
        )

    def test_fixed_delta_limits_repeated_outlier_bias(self) -> None:
        tolerance = (
            StableMotionConfig().smoothed_center_absolute_position_tolerance
        )
        for repeated_zero_count in (3, 4, 7):
            with self.subTest(repeated_zero_count=repeated_zero_count):
                transition_vectors = np.vstack(
                    (
                        np.zeros(
                            (repeated_zero_count, 2),
                            dtype=np.float64,
                        ),
                        np.asarray([[1.0, 0.0]], dtype=np.float64),
                    )
                )
                tracks = (
                    transition_vectors[:, None, :]
                    * np.arange(PHASES, dtype=np.float64)[None, :, None]
                )
                track_count = repeated_zero_count + 1
                result = build_stable_motion_representation(
                    tracks,
                    _full_mask(track_count),
                    np.zeros(PHASES),
                    component_track_indices=np.arange(track_count),
                )
                self.assertTrue(
                    result.diagnostic_ready,
                    result.failure_detail,
                )
                theoretical_limit = (
                    R8_SMOOTHED_CENTER_DEFAULT_DELTA
                    / np.sqrt(repeated_zero_count**2 - 1.0)
                )
                self.assertGreaterEqual(
                    float(result.transition_displacement[1, 0]),
                    -tolerance,
                )
                self.assertLessEqual(
                    float(result.transition_displacement[1, 0]),
                    theoretical_limit + tolerance,
                )
                self.assertLessEqual(
                    float(result.trajectory[-1, 0]),
                    (PHASES - 1) * (theoretical_limit + tolerance),
                )
                np.testing.assert_allclose(
                    result.transition_displacement[:, 1],
                    0.0,
                    atol=0.0,
                )
                self.assertTrue(
                    (
                        result.center_position_error_upper_bound[1:]
                        <= tolerance
                    ).all()
                )

    def test_near_coincident_slow_convergence_is_stable(self) -> None:
        # Fixed synthetic RNG regression (seed 88001, trial 3, mode 3).
        # The former 1e-10 default exceeded 512 Weiszfeld iterations on this
        # finite near-coincident set.
        transition_vectors = np.asarray(
            [
                [-0.00028268288588151336, -0.0036386747378855944],
                [-0.022739266976714134, 0.013050105422735214],
                [-0.00009942863835021853, -0.00017973597277887166],
                [-0.006589461583644152, -0.0062282998114824295],
                [-0.0058202012442052364, 0.01044323481619358],
                [-0.0003046761266887188, -0.0003343753924127668],
                [-0.0007470791460946202, -0.0008858003420755267],
                [-0.013058217242360115, -0.0011300460901111364],
                [0.001905636047013104, -0.0006895505357533693],
                [0.005144939757883549, -0.0001341227616649121],
                [-0.0005015117931179702, 0.0026944493874907494],
            ],
            dtype=np.float32,
        ).astype(np.float64)
        phase = np.arange(PHASES, dtype=np.float64)
        tracks = (
            transition_vectors[:, None, :]
            * phase[None, :, None]
        )
        membership = np.asarray(
            [31, 7, 45, 2, 19, 8, 24, 5, 38, 12, 1],
            dtype=np.int64,
        )
        baseline = build_stable_motion_representation(
            tracks,
            _full_mask(len(tracks)),
            np.zeros(PHASES),
            component_track_indices=membership,
        )
        permutation = np.asarray([7, 2, 10, 0, 5, 9, 1, 8, 3, 6, 4])
        permuted = build_stable_motion_representation(
            tracks[permutation],
            _full_mask(len(tracks))[permutation],
            np.zeros(PHASES),
            component_track_indices=membership[permutation],
        )
        angle = 0.713
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        rotated = build_stable_motion_representation(
            tracks @ rotation.T,
            _full_mask(len(tracks)),
            np.zeros(PHASES),
            component_track_indices=membership,
        )

        self.assertEqual(
            StableMotionConfig().smoothed_center_delta,
            R8_SMOOTHED_CENTER_DEFAULT_DELTA,
        )
        self.assertTrue(baseline.diagnostic_ready, baseline.failure_detail)
        self.assertTrue(permuted.diagnostic_ready, permuted.failure_detail)
        self.assertTrue(rotated.diagnostic_ready, rotated.failure_detail)
        np.testing.assert_array_equal(
            permuted.transition_displacement,
            baseline.transition_displacement,
        )
        np.testing.assert_array_equal(
            permuted.trajectory,
            baseline.trajectory,
        )
        np.testing.assert_allclose(
            rotated.transition_displacement,
            baseline.transition_displacement @ rotation.T,
            atol=2e-8,
        )
        np.testing.assert_allclose(
            rotated.trajectory,
            baseline.trajectory @ rotation.T,
            atol=2e-8,
        )
        np.testing.assert_array_equal(
            baseline.transition_displacement[1:],
            np.repeat(
                baseline.transition_displacement[1:2],
                PHASES - 1,
                axis=0,
            ),
        )
        self.assertTrue(
            (
                baseline.center_position_error_upper_bound[1:]
                <= StableMotionConfig().smoothed_center_absolute_position_tolerance
            ).all()
        )

    def test_global_trajectory_is_coordinate_rotation_equivariant(
        self,
    ) -> None:
        rng = np.random.default_rng(808)
        track_count = 7
        offsets = rng.uniform(-0.4, 0.4, size=(track_count, 2))
        directions = rng.normal(size=(track_count, 2))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        phase = np.arange(PHASES, dtype=np.float64)
        tracks = np.zeros((track_count, PHASES, 2), dtype=np.float64)
        global_motion = np.stack(
            (0.006 * phase, -0.003 * phase + 0.0001 * phase**2),
            axis=1,
        )
        for index in range(track_count):
            deformation = (
                0.012
                * np.sin(phase / 4.0 + 0.37 * index)
            )[:, None] * directions[index]
            tracks[index] = (
                offsets[index][None] + global_motion + deformation
            )
        angle = 0.713
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        energy = np.linspace(0.0, 0.2, PHASES)
        original = build_stable_motion_representation(
            tracks,
            _full_mask(track_count),
            energy,
            component_track_indices=np.arange(30, 37),
        )
        rotated = build_stable_motion_representation(
            tracks @ rotation.T,
            _full_mask(track_count),
            energy,
            component_track_indices=np.arange(30, 37),
        )
        self.assertTrue(original.diagnostic_ready, original.failure_detail)
        self.assertTrue(rotated.diagnostic_ready, rotated.failure_detail)
        np.testing.assert_allclose(
            rotated.trajectory,
            original.trajectory @ rotation.T,
            atol=3e-7,
        )
        np.testing.assert_allclose(
            rotated.transition_displacement,
            original.transition_displacement @ rotation.T,
            atol=3e-7,
        )
        np.testing.assert_allclose(
            rotated.shape_tokens[:, :4],
            original.shape_tokens[:, :4],
            atol=2e-7,
        )
        np.testing.assert_allclose(
            rotated.shape_tokens[:, 7],
            original.shape_tokens[:, 7],
            atol=2e-7,
        )
        for phase_index in range(1, PHASES):
            covariance = np.asarray(
                [
                    [
                        original.shape_tokens[phase_index, 4],
                        original.shape_tokens[phase_index, 5],
                    ],
                    [
                        original.shape_tokens[phase_index, 5],
                        original.shape_tokens[phase_index, 6],
                    ],
                ]
            )
            rotated_covariance = np.asarray(
                [
                    [
                        rotated.shape_tokens[phase_index, 4],
                        rotated.shape_tokens[phase_index, 5],
                    ],
                    [
                        rotated.shape_tokens[phase_index, 5],
                        rotated.shape_tokens[phase_index, 6],
                    ],
                ]
            )
            np.testing.assert_allclose(
                rotated_covariance,
                rotation @ covariance @ rotation.T,
                atol=2e-7,
            )

    def test_translation_is_trajectory_and_energy_is_preserved(self) -> None:
        offsets = np.asarray(
            [
                [-0.20, -0.10],
                [-0.10, 0.10],
                [0.10, -0.10],
                [0.20, 0.10],
            ]
        )
        phase = np.arange(PHASES, dtype=np.float64)
        translation = np.stack((0.01 * phase, -0.004 * phase), axis=1)
        tracks = _static_tracks(offsets) + translation[None]
        energy = np.linspace(0.0, 0.4, PHASES, dtype=np.float32)
        result = build_stable_motion_representation(
            tracks,
            _full_mask(4),
            energy,
            component_track_indices=np.asarray([6, 2, 8, 4]),
            phase_times=np.linspace(0.0, 2.0, PHASES),
            coordinate_space="normalized-max-side-isotropic",
        )
        self.assertTrue(result.diagnostic_ready)
        np.testing.assert_allclose(
            result.trajectory,
            translation,
            atol=1e-7,
        )
        np.testing.assert_allclose(result.shape_tokens, 0.0, atol=1e-12)
        expected_transition = np.zeros_like(translation)
        expected_transition[1:] = np.diff(translation, axis=0)
        np.testing.assert_allclose(
            result.transition_displacement,
            expected_transition,
            atol=1e-7,
        )
        self.assertEqual(int(result.transition_support_count[0]), 0)
        self.assertTrue(
            (result.transition_support_count[1:] == 4).all()
        )
        np.testing.assert_allclose(result.transition_support[0], 0.0)
        np.testing.assert_allclose(result.transition_support[1:], 1.0)
        np.testing.assert_array_equal(result.phase_energy, energy)
        self.assertEqual(
            result.coordinate_space,
            "normalized-max-side-isotropic",
        )

    def test_rotation_has_zero_translation_and_nonzero_shape_token(
        self,
    ) -> None:
        angles = np.arange(8, dtype=np.float64) * 2.0 * np.pi / 8.0
        theta = np.linspace(0.0, 1.2, PHASES)
        tracks = np.zeros((8, PHASES, 2), dtype=np.float64)
        for phase in range(PHASES):
            tracks[:, phase, 0] = 0.50 + 0.08 * np.cos(
                angles + theta[phase]
            )
            tracks[:, phase, 1] = 0.50 + 0.08 * np.sin(
                angles + theta[phase]
            )
        energy = np.abs(np.sin(theta)).astype(np.float32)
        result = build_stable_motion_representation(
            tracks,
            _full_mask(8),
            energy,
            component_track_indices=np.arange(20, 28),
        )
        self.assertTrue(result.diagnostic_ready)
        np.testing.assert_allclose(result.trajectory, 0.0, atol=1e-7)
        self.assertGreater(
            float(
                np.max(
                    result.shape_tokens[:, 0]
                )
            ),
            0.002,
        )
        self.assertGreater(
            float(np.max(result.shape_tokens[:, 4])),
            1e-6,
        )
        np.testing.assert_allclose(result.shape_tokens[0], 0.0, atol=0.0)
        np.testing.assert_array_equal(result.phase_energy, energy)

    def test_nonrigid_deformation_is_separate_from_translation(self) -> None:
        offsets = np.asarray(
            [
                [-0.30, 0.00],
                [-0.20, 0.10],
                [-0.10, -0.10],
                [0.10, 0.10],
                [0.20, -0.10],
                [0.30, 0.00],
            ]
        )
        tracks = np.zeros((6, PHASES, 2), dtype=np.float64)
        phase = np.arange(PHASES, dtype=np.float64)
        translation = np.stack((0.003 * phase, 0.001 * phase), axis=1)
        scale = 1.0 + 0.30 * np.sin(np.linspace(0.0, np.pi, PHASES))
        for index in range(6):
            tracks[index] = (
                offsets[index][None] * scale[:, None]
                + translation
                + np.asarray([0.5, 0.5])
            )
        result = build_stable_motion_representation(
            tracks,
            _full_mask(6),
            np.linspace(0.0, 0.3, PHASES),
        )
        self.assertTrue(result.diagnostic_ready)
        np.testing.assert_allclose(
            result.trajectory,
            translation,
            atol=1e-7,
        )
        self.assertGreater(
            float(np.max(result.shape_tokens[:, 0])),
            0.005,
        )
        self.assertGreater(
            float(np.max(result.shape_tokens[:, 7])),
            0.70,
        )


class ValidationTests(unittest.TestCase):
    def test_smoothed_center_config_fails_closed(self) -> None:
        tracks = _static_tracks(
            np.asarray([[-0.1, 0.0], [0.0, 0.1], [0.1, 0.0]])
        )
        configs = (
            StableMotionConfig(smoothed_center_delta=0.0),
            StableMotionConfig(
                smoothed_center_absolute_position_tolerance=0.0
            ),
            StableMotionConfig(smoothed_center_delta="invalid"),
            StableMotionConfig(smoothed_center_delta="1e-4"),
            StableMotionConfig(
                minimum_transition_track_fraction="invalid"
            ),
            StableMotionConfig(smoothed_center_delta=True),
        )
        for config in configs:
            result = build_stable_motion_representation(
                tracks,
                _full_mask(3),
                np.zeros(PHASES),
                config=config,
            )
            self.assertFalse(result.diagnostic_ready)
            self.assertEqual(result.failure_reason, "invalid_config")
        invalid_delta = build_stable_motion_representation(
            tracks,
            _full_mask(3),
            np.zeros(PHASES),
            config=configs[0],
        )
        self.assertIsNone(invalid_delta.smoothed_center_delta)
        self.assertIsNone(
            invalid_delta.to_summary()["smoothed_center_delta"]
        )
        invalid_tolerance = build_stable_motion_representation(
            tracks,
            _full_mask(3),
            np.zeros(PHASES),
            config=configs[1],
        )
        self.assertIsNone(
            invalid_tolerance.smoothed_center_absolute_position_tolerance
        )

    def test_extreme_center_config_fails_before_solver(self) -> None:
        tracks = _static_tracks(
            np.asarray([[-0.1, 0.0], [0.0, 0.1], [0.1, 0.0]])
        )
        extreme_values = (
            np.nextafter(0.0, 1.0),
            np.finfo(np.float64).tiny,
            1e-200,
            1e200,
            np.finfo(np.float64).max,
        )
        for field in (
            "smoothed_center_delta",
            "smoothed_center_absolute_position_tolerance",
        ):
            for value in extreme_values:
                with self.subTest(field=field, value=value):
                    config = StableMotionConfig(**{field: value})
                    result = build_stable_motion_representation(
                        tracks,
                        _full_mask(3),
                        np.zeros(PHASES),
                        config=config,
                    )
                    self.assertFalse(result.diagnostic_ready)
                    self.assertEqual(
                        result.failure_reason,
                        "invalid_config",
                    )
                    self.assertIsNone(getattr(result, field))
        for iterations in (
            R8_SMOOTHED_CENTER_MAX_ITERATIONS + 1,
            10**100,
        ):
            result = build_stable_motion_representation(
                tracks,
                _full_mask(3),
                np.zeros(PHASES),
                config=StableMotionConfig(
                    smoothed_center_max_iterations=iterations,
                ),
            )
            self.assertFalse(result.diagnostic_ready)
            self.assertEqual(result.failure_reason, "invalid_config")

    def test_normalized_center_config_boundaries_are_accepted(self) -> None:
        for field in (
            "smoothed_center_delta",
            "smoothed_center_absolute_position_tolerance",
        ):
            for value in (
                R8_SMOOTHED_CENTER_MIN_NORMALIZED_VALUE,
                R8_SMOOTHED_CENTER_MAX_NORMALIZED_VALUE,
            ):
                StableMotionConfig(**{field: value}).validate()
        StableMotionConfig(
            smoothed_center_max_iterations=(
                R8_SMOOTHED_CENTER_MAX_ITERATIONS
            )
        ).validate()

    def test_float32_energy_output_overflow_fails_closed(self) -> None:
        energy = np.zeros(PHASES, dtype=np.float64)
        energy[3] = 1e40
        result = build_stable_motion_representation(
            _static_tracks(
                np.asarray([[-0.1, 0.0], [0.0, 0.1], [0.1, 0.0]])
            ),
            _full_mask(3),
            energy,
        )
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(result.failure_reason, "non_finite_output")

    def test_failure_summary_preserves_valid_custom_center_config(
        self,
    ) -> None:
        transition_vectors = np.asarray(
            [
                [-0.4015348829485148, -0.42530156214896514],
                [-0.5065298585202478, -0.5365111457334341],
                [0.00017850547013013911, 0.00018907113310057789],
                [0.06941926925112026, 0.07352816631718363],
            ],
            dtype=np.float64,
        )
        tracks = (
            transition_vectors[:, None, :]
            * np.arange(PHASES, dtype=np.float64)[None, :, None]
        )
        config = StableMotionConfig(
            smoothed_center_max_iterations=8,
            smoothed_center_delta=2e-4,
            smoothed_center_absolute_position_tolerance=5e-8,
        )
        result = build_stable_motion_representation(
            tracks,
            _full_mask(4),
            np.zeros(PHASES),
            config=config,
        )
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(
            result.failure_reason,
            "smoothed_median_uncertified",
        )
        summary = result.to_summary()
        self.assertEqual(summary["smoothed_center_delta"], 2e-4)
        self.assertEqual(
            summary["smoothed_center_absolute_position_tolerance"],
            5e-8,
        )
        self.assertIsNone(
            summary["max_center_position_error_upper_bound"]
        )
        self.assertEqual(summary["center_certificate_kind_counts"], {})

    def test_nonfinite_and_negative_energy_fail_closed(self) -> None:
        tracks = _static_tracks(
            np.asarray([[-0.1, 0.0], [0.0, 0.1], [0.1, 0.0]])
        )
        tracks[0, 4, 0] = np.nan
        result = build_stable_motion_representation(
            tracks,
            _full_mask(3),
            np.zeros(PHASES),
        )
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(result.failure_reason, "non_finite_input")
        self.assertEqual(result.trajectory.shape, (32, 2))
        self.assertEqual(result.shape_tokens.shape, (32, 8))
        self.assertEqual(result.component_track_indices.shape, (0,))

        tracks[0, 4, 0] = 0.0
        energy = np.zeros(PHASES)
        energy[5] = -0.1
        result = build_stable_motion_representation(
            tracks,
            _full_mask(3),
            energy,
        )
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(result.failure_reason, "invalid_energy")

    def test_bad_mask_and_membership_fail_closed(self) -> None:
        tracks = _static_tracks(
            np.asarray([[-0.1, 0.0], [0.0, 0.1], [0.1, 0.0]])
        )
        numeric_mask = np.ones((3, PHASES), dtype=np.int64)
        result = build_stable_motion_representation(
            tracks,
            numeric_mask,
            np.zeros(PHASES),
        )
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(result.failure_reason, "invalid_mask_dtype")

        result = build_stable_motion_representation(
            tracks,
            _full_mask(3),
            np.zeros(PHASES),
            component_track_indices=np.asarray([2, 2, 4]),
        )
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(
            result.failure_reason,
            "invalid_component_membership",
        )

    def test_insufficient_phase_support_fails_closed(self) -> None:
        tracks = _static_tracks(
            np.asarray(
                [
                    [-0.2, 0.0],
                    [-0.1, 0.1],
                    [0.1, -0.1],
                    [0.2, 0.0],
                ]
            )
        )
        mask = _full_mask(4)
        mask[1:, 16] = False
        result = build_stable_motion_representation(
            tracks,
            mask,
            np.zeros(PHASES),
        )
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(
            result.failure_reason,
            "insufficient_phase_support",
        )

    def test_alternating_mask_fails_common_transition_support(
        self,
    ) -> None:
        tracks = _static_tracks(
            np.asarray(
                [
                    [-0.2, 0.0],
                    [-0.1, 0.1],
                    [0.1, -0.1],
                    [0.2, 0.0],
                ]
            )
        )
        mask = np.zeros((4, PHASES), dtype=bool)
        mask[:2, ::2] = True
        mask[2:, 1::2] = True
        self.assertTrue((np.sum(mask, axis=0) == 2).all())
        result = build_stable_motion_representation(
            tracks,
            mask,
            np.zeros(PHASES),
        )
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(
            result.failure_reason,
            "insufficient_transition_support",
        )
        self.assertEqual(result.transition_support.shape, (32,))
        self.assertEqual(
            result.transition_support_count.shape,
            (32,),
        )

    def test_summary_keeps_dev_only_semantic_limit(self) -> None:
        result = build_stable_motion_representation(
            _static_tracks(
                np.asarray([[-0.1, 0.0], [0.0, 0.1], [0.1, 0.0]])
            ),
            _full_mask(3),
            np.zeros(PHASES),
        )
        self.assertTrue(result.diagnostic_ready)
        summary = result.to_summary()
        self.assertEqual(summary["schema_version"], R8_STABLE_MOTION_SCHEMA)
        self.assertEqual(
            summary["shape_token_dim"],
            R8_STABLE_MOTION_SHAPE_DIM,
        )
        self.assertFalse(summary["semantic_actor_identified"])
        self.assertIn("development-only", summary["scope"])
        self.assertTrue(summary["energy_is_upstream_preserved"])
        self.assertEqual(summary["track_reanchor_role"], "evidence-only")
        self.assertFalse(
            summary["center_objective_is_exact_geometric_median"]
        )
        self.assertEqual(
            summary["smoothed_center_delta"],
            R8_SMOOTHED_CENTER_DEFAULT_DELTA,
        )
        self.assertEqual(summary["center_storage_dtype"], "float64")
        self.assertEqual(
            summary["smoothed_center_normalized_parameter_bounds"],
            {
                "minimum": R8_SMOOTHED_CENTER_MIN_NORMALIZED_VALUE,
                "maximum": R8_SMOOTHED_CENTER_MAX_NORMALIZED_VALUE,
            },
        )
        self.assertEqual(
            summary["smoothed_center_max_iterations_limit"],
            R8_SMOOTHED_CENTER_MAX_ITERATIONS,
        )
        self.assertEqual(result.trajectory.dtype, np.dtype(np.float64))
        self.assertEqual(
            result.transition_displacement.dtype,
            np.dtype(np.float64),
        )
        self.assertIn(
            "position certificates",
            summary["center_success_condition"],
        )
        self.assertLessEqual(
            summary["max_center_position_error_upper_bound"],
            summary["smoothed_center_absolute_position_tolerance"],
        )
        self.assertEqual(result.center_position_error_upper_bound.shape, (32,))
        self.assertEqual(len(result.center_certificate_kind), 32)
        self.assertTrue(summary["phase_zero_has_no_transition"])
        self.assertEqual(summary["formal_status"], "INSUFFICIENT")
        self.assertFalse(summary["production_decision"])
        self.assertFalse(summary["generation_authorized"])


if __name__ == "__main__":
    unittest.main()
