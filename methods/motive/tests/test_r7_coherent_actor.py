from __future__ import annotations

import unittest

import numpy as np

from motive.r7_coherent_actor import (
    CoherentActorConfig,
    R7_COHERENT_ACTOR_SCHEMA,
    select_coherent_actor,
)


def _scene(
    *,
    frames: int = 24,
    background_tracks: int = 30,
    actor_tracks: int = 6,
    seed: int = 260108828,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    background = np.stack(
        (
            rng.uniform(0.05, 0.95, size=background_tracks),
            rng.uniform(0.05, 0.95, size=background_tracks),
        ),
        axis=1,
    )
    # Actor points occupy a compact region, independent of the background
    # grid.  Inputs are already camera compensated and normalized.
    actor = np.stack(
        (
            rng.uniform(0.36, 0.46, size=actor_tracks),
            rng.uniform(0.42, 0.54, size=actor_tracks),
        ),
        axis=1,
    )
    reference = np.concatenate((background, actor), axis=0)
    tracks = np.repeat(reference[None], frames, axis=0)
    visibility = np.ones((frames, len(reference)), dtype=np.float64)
    times = np.arange(frames, dtype=np.float64) / 10.0
    return tracks, visibility, times


def _add_local_motion(
    tracks: np.ndarray,
    *,
    actor_tracks: int,
    start: int = 6,
    stop: int = 15,
    velocity: tuple[float, float] = (0.010, 0.002),
) -> None:
    offset = np.zeros(2, dtype=np.float64)
    for frame in range(1, len(tracks)):
        if start <= frame < stop:
            offset += np.asarray(velocity, dtype=np.float64)
        tracks[frame, -actor_tracks:] += offset


def _add_indexed_motion(
    tracks: np.ndarray,
    indices: np.ndarray,
    *,
    start: int = 6,
    stop: int = 15,
    velocity: tuple[float, float] = (0.010, 0.002),
) -> None:
    offset = np.zeros(2, dtype=np.float64)
    for frame in range(1, len(tracks)):
        if start <= frame < stop:
            offset += np.asarray(velocity, dtype=np.float64)
        tracks[frame, indices] += offset


def _config(**overrides: object) -> CoherentActorConfig:
    values: dict[str, object] = {}
    values.update(overrides)
    return CoherentActorConfig(**values)


class ValidationAndFailureTests(unittest.TestCase):
    def test_nonfinite_tracks_fail_closed_with_empty_actor(self) -> None:
        tracks, visibility, times = _scene()
        tracks[3, 2, 0] = np.nan
        result = select_coherent_actor(tracks, visibility, times)
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(result.failure_reason, "non_finite_input")
        self.assertEqual(int(result.actor_track_mask.sum()), 0)
        self.assertEqual(result.actor_track_indices.shape, (0,))
        self.assertEqual(result.actor_trajectory.shape, (32, 2))
        self.assertEqual(float(result.phase_energy.sum()), 0.0)

    def test_duplicate_time_and_invalid_visibility_fail_closed(self) -> None:
        tracks, visibility, times = _scene()
        duplicate = times.copy()
        duplicate[7] = duplicate[6]
        result = select_coherent_actor(tracks, visibility, duplicate)
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(result.failure_reason, "invalid_frame_times")

        visibility[0, 0] = 1.1
        result = select_coherent_actor(tracks, visibility, times)
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(result.failure_reason, "invalid_visibility")

    def test_bad_shape_and_bad_config_are_structured_failures(self) -> None:
        result = select_coherent_actor(
            np.zeros((8, 12)),
            np.ones((8, 12)),
            np.arange(8),
        )
        self.assertEqual(result.failure_reason, "invalid_shape")
        self.assertEqual(result.actor_track_mask.shape, (0,))

        tracks, visibility, times = _scene()
        result = select_coherent_actor(
            tracks,
            visibility,
            times,
            config=_config(maximum_component_fraction=0.9),
        )
        self.assertEqual(result.failure_reason, "invalid_config")

    def test_invalid_frame_size_fails_closed(self) -> None:
        tracks, visibility, times = _scene()
        result = select_coherent_actor(
            tracks,
            visibility,
            times,
            frame_size=(1080, 0),
        )
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(result.failure_reason, "invalid_frame_size")

    def test_static_jitter_does_not_emit_actor_representation(self) -> None:
        tracks, visibility, times = _scene()
        phase = np.arange(len(tracks), dtype=np.float64)
        jitter = 0.00015 * np.stack(
            (np.sin(phase), np.cos(phase)),
            axis=1,
        )
        tracks += jitter[:, None]
        result = select_coherent_actor(tracks, visibility, times)
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(result.failure_reason, "no_moving_tracks")
        self.assertEqual(int(result.actor_track_mask.sum()), 0)

    def test_fast_small_amplitude_global_jitter_fails_excursion_gate(
        self,
    ) -> None:
        tracks, visibility, times = _scene()
        alternating = np.where(
            np.arange(len(tracks)) % 2 == 0,
            -1.0,
            1.0,
        )
        tracks[:, :, 0] += 0.003 * alternating[:, None]
        result = select_coherent_actor(tracks, visibility, times)
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(result.failure_reason, "no_moving_tracks")

    def test_tiny_coherent_motion_fails_absolute_motion_gate(self) -> None:
        tracks, visibility, times = _scene()
        _add_local_motion(
            tracks,
            actor_tracks=6,
            velocity=(0.0003, 0.0001),
        )
        result = select_coherent_actor(tracks, visibility, times)
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(result.failure_reason, "no_moving_tracks")


class CoherenceAndLocalityTests(unittest.TestCase):
    def test_global_translation_is_rejected_as_camera_residual(self) -> None:
        tracks, visibility, times = _scene()
        for frame in range(1, len(tracks)):
            tracks[frame] += np.asarray([0.008, -0.002]) * frame
        result = select_coherent_actor(tracks, visibility, times)
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(result.failure_reason, "global_residual_motion")
        self.assertGreater(result.global_moving_fraction, 0.95)
        self.assertGreater(result.moving_spatial_coverage, 0.95)
        self.assertGreater(result.global_direction_coherence, 0.99)
        self.assertEqual(int(result.actor_track_mask.sum()), 0)

    def test_partial_framewide_comotion_is_rejected(self) -> None:
        tracks, visibility, times = _scene(
            background_tracks=34,
            actor_tracks=6,
        )
        # Forty percent is below the coarse 65% moving-track rejection gate,
        # but these tracks span the field and share one residual translation.
        moving_indices = np.linspace(0, 33, 16, dtype=np.int64)
        for frame in range(1, len(tracks)):
            tracks[frame, moving_indices] += (
                np.asarray([0.009, -0.002]) * frame
            )
        result = select_coherent_actor(tracks, visibility, times)
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(result.failure_reason, "global_residual_motion")
        self.assertAlmostEqual(result.global_moving_fraction, 0.4)
        self.assertGreater(result.moving_spatial_coverage, 0.70)
        self.assertGreater(result.global_direction_coherence, 0.99)

    def test_single_noisy_track_cannot_form_actor(self) -> None:
        tracks, visibility, times = _scene()
        for frame in range(6, 15):
            tracks[frame:, 0] += np.asarray([0.013, -0.004])
        result = select_coherent_actor(tracks, visibility, times)
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(result.failure_reason, "no_coherent_component")
        self.assertEqual(result.moving_track_count, 1)
        self.assertTrue(result.components)
        self.assertEqual(result.components[0].track_count, 1)

    def test_local_coherent_motion_selects_only_actor_tracks(self) -> None:
        tracks, visibility, times = _scene(actor_tracks=6)
        _add_local_motion(tracks, actor_tracks=6)
        result = select_coherent_actor(tracks, visibility, times)
        self.assertTrue(result.diagnostic_ready, result.failure_detail)
        np.testing.assert_array_equal(
            np.flatnonzero(result.actor_track_mask),
            np.arange(30, 36),
        )
        self.assertEqual(result.schema_version, R7_COHERENT_ACTOR_SCHEMA)
        self.assertEqual(result.actor_trajectory.shape, (32, 2))
        self.assertEqual(result.actor_track_trajectories.shape, (6, 32, 2))
        self.assertEqual(result.actor_track_phase_mask.shape, (6, 32))
        self.assertTrue(result.actor_track_phase_mask.all())
        self.assertGreater(float(result.actor_trajectory[-1, 0]), 0.07)
        self.assertGreater(float(np.max(result.phase_energy)), 0.08)
        self.assertLess(result.global_moving_fraction, 0.25)
        summary = result.to_summary()
        self.assertTrue(summary["diagnostic_ready"])
        self.assertEqual(
            sorted(summary["actor_track_indices"]),
            list(range(30, 36)),
        )
        self.assertFalse(summary["semantic_actor_identified"])
        self.assertIn("development-only", summary["scope"])

    def test_articulated_directions_connect_through_soft_coherence(self) -> None:
        tracks, visibility, times = _scene(actor_tracks=8)
        offset_a = np.zeros(2, dtype=np.float64)
        offset_b = np.zeros(2, dtype=np.float64)
        for frame in range(1, len(tracks)):
            if 6 <= frame < 15:
                offset_a += np.asarray([0.010, 0.001])
                offset_b += np.asarray([0.001, 0.010])
            tracks[frame, -8:-4] += offset_a
            tracks[frame, -4:] += offset_b
        result = select_coherent_actor(tracks, visibility, times)
        self.assertTrue(result.diagnostic_ready, result.failure_detail)
        np.testing.assert_array_equal(
            np.flatnonzero(result.actor_track_mask),
            np.arange(30, 38),
        )
        assert result.selected_component is not None
        component = result.components[result.selected_component]
        self.assertEqual(component.track_count, 8)
        self.assertGreater(component.edge_density, 0.30)
        self.assertGreater(float(result.actor_trajectory[-1].sum()), 0.05)

    def test_spatially_distant_synchronous_objects_stay_separate(self) -> None:
        tracks, visibility, times = _scene(actor_tracks=6)
        tracks[:, -3:, 0] += 0.42
        _add_local_motion(tracks, actor_tracks=6)
        result = select_coherent_actor(
            tracks,
            visibility,
            times,
            config=_config(minimum_component_tracks=3),
        )
        self.assertTrue(result.diagnostic_ready, result.failure_detail)
        self.assertEqual(int(result.actor_track_mask.sum()), 3)
        self.assertEqual(
            max(component.track_count for component in result.components),
            3,
        )

    def test_permuting_tracks_preserves_physical_selection_and_features(
        self,
    ) -> None:
        tracks, visibility, times = _scene(actor_tracks=8)
        _add_local_motion(tracks, actor_tracks=8)
        baseline = select_coherent_actor(tracks, visibility, times)
        self.assertTrue(baseline.diagnostic_ready)

        rng = np.random.default_rng(17)
        permutation = rng.permutation(tracks.shape[1])
        permuted = select_coherent_actor(
            tracks[:, permutation],
            visibility[:, permutation],
            times,
        )
        self.assertTrue(permuted.diagnostic_ready)
        physical_mask = np.zeros_like(permuted.actor_track_mask)
        physical_mask[permutation] = permuted.actor_track_mask
        np.testing.assert_array_equal(
            physical_mask,
            baseline.actor_track_mask,
        )
        np.testing.assert_allclose(
            permuted.actor_trajectory,
            baseline.actor_trajectory,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            permuted.phase_energy,
            baseline.phase_energy,
            atol=1e-7,
        )

    def test_equal_score_component_tie_break_is_permutation_invariant(
        self,
    ) -> None:
        tracks, visibility, times = _scene(actor_tracks=6)
        # Make two equally moving, disconnected three-track components.
        tracks[:, -3:, 0] += 0.42
        _add_local_motion(tracks, actor_tracks=6)
        baseline = select_coherent_actor(tracks, visibility, times)
        self.assertTrue(baseline.diagnostic_ready)
        self.assertEqual(int(baseline.actor_track_mask.sum()), 3)

        permutation = np.random.default_rng(99).permutation(tracks.shape[1])
        permuted = select_coherent_actor(
            tracks[:, permutation],
            visibility[:, permutation],
            times,
        )
        self.assertTrue(permuted.diagnostic_ready)
        physical_mask = np.zeros_like(permuted.actor_track_mask)
        physical_mask[permutation] = permuted.actor_track_mask
        np.testing.assert_array_equal(
            physical_mask,
            baseline.actor_track_mask,
        )


class AdversarialRegressionTests(unittest.TestCase):
    def test_thin_frame_spanning_motion_band_is_not_actor_local(self) -> None:
        tracks, visibility, times = _scene(
            background_tracks=30,
            actor_tracks=9,
        )
        tracks[:, -9:, 0] = np.linspace(0.10, 0.90, 9)[None]
        tracks[:, -9:, 1] = 0.50
        _add_local_motion(tracks, actor_tracks=9)
        result = select_coherent_actor(tracks, visibility, times)
        self.assertFalse(result.diagnostic_ready)
        self.assertGreater(result.moving_spatial_coverage_x, 0.65)

        # Relax graph-shape rejection to isolate the independent per-axis
        # locality gate: diagonal coverage alone would accept this thin band.
        coverage_only = select_coherent_actor(
            tracks,
            visibility,
            times,
            config=_config(
                minimum_component_degree_fraction=0.20,
                maximum_component_graph_diameter=8,
            ),
        )
        self.assertFalse(coverage_only.diagnostic_ready)
        self.assertEqual(
            coverage_only.failure_reason,
            "global_residual_motion",
        )
        self.assertIn(
            "global_residual_motion",
            {
                component.rejection_reason
                for component in coverage_only.components
            },
        )

    def test_opposite_parallax_patches_fail_without_net_direction(self) -> None:
        tracks, visibility, times = _scene(
            background_tracks=24,
            actor_tracks=16,
            seed=91,
        )
        rng = np.random.default_rng(92)
        tracks[:, -16:-8, 0] = rng.uniform(0.10, 0.20, 8)[None]
        tracks[:, -16:-8, 1] = rng.uniform(0.20, 0.30, 8)[None]
        tracks[:, -8:, 0] = rng.uniform(0.80, 0.90, 8)[None]
        tracks[:, -8:, 1] = rng.uniform(0.70, 0.80, 8)[None]
        _add_indexed_motion(
            tracks,
            np.arange(24, 32),
            velocity=(0.010, 0.0),
        )
        _add_indexed_motion(
            tracks,
            np.arange(32, 40),
            velocity=(-0.010, 0.0),
        )
        result = select_coherent_actor(tracks, visibility, times)
        self.assertFalse(result.diagnostic_ready)
        self.assertEqual(result.failure_reason, "global_residual_motion")
        self.assertAlmostEqual(result.global_moving_fraction, 0.40)
        self.assertLess(result.global_translation_coherence, 0.20)
        self.assertGreater(
            max(
                result.global_radial_coherence,
                result.global_rotation_coherence,
            ),
            0.50,
        )

    def test_stationary_centroid_rotation_keeps_deformation_energy(self) -> None:
        tracks, visibility, times = _scene(actor_tracks=8)
        angles = np.arange(8, dtype=np.float64) * 2.0 * np.pi / 8.0
        for frame in range(len(tracks)):
            theta = 0.18 * max(0, min(frame, 14) - 5)
            tracks[frame, -8:, 0] = 0.50 + 0.05 * np.cos(
                angles + theta
            )
            tracks[frame, -8:, 1] = 0.50 + 0.05 * np.sin(
                angles + theta
            )
        result = select_coherent_actor(tracks, visibility, times)
        self.assertTrue(result.diagnostic_ready, result.failure_detail)
        self.assertGreater(float(np.max(result.phase_energy)), 0.05)
        self.assertGreater(result.global_rotation_coherence, 0.80)
        centroid_path = float(
            np.sum(
                np.linalg.norm(
                    np.diff(result.actor_trajectory, axis=0),
                    axis=1,
                )
            )
        )
        self.assertLess(centroid_path, 1e-6)

    def test_internal_visibility_hole_is_not_interpolated_as_valid(
        self,
    ) -> None:
        tracks, visibility, times = _scene(actor_tracks=6, seed=111)
        _add_local_motion(tracks, actor_tracks=6)
        visibility[10:12, 30] = 0.0
        result = select_coherent_actor(tracks, visibility, times)
        self.assertTrue(result.diagnostic_ready, result.failure_detail)
        position = int(np.flatnonzero(result.actor_track_indices == 30)[0])
        self.assertTrue(result.actor_track_phase_mask[position].any())
        self.assertTrue((~result.actor_track_phase_mask[position]).any())
        self.assertGreater(
            int(np.sum(~result.actor_track_phase_mask[position])),
            1,
        )

    def test_single_link_chain_fails_minimum_degree(self) -> None:
        tracks, visibility, times = _scene(
            background_tracks=30,
            actor_tracks=8,
            seed=121,
        )
        tracks[:, -8:, 0] = np.linspace(0.15, 0.85, 8)[None]
        tracks[:, -8:, 1] = 0.50
        _add_local_motion(tracks, actor_tracks=8, velocity=(0.010, 0.0))
        result = select_coherent_actor(
            tracks,
            visibility,
            times,
            config=_config(
                spatial_neighbor_radius=0.11,
                articulated_neighbor_radius=0.05,
                maximum_component_axis_coverage=0.99,
                maximum_component_spatial_coverage=0.99,
                maximum_component_spatial_occupancy=1.0,
                maximum_component_graph_diameter=20,
            ),
        )
        self.assertFalse(result.diagnostic_ready)
        self.assertIn(
            "low_component_minimum_degree",
            {
                component.rejection_reason
                for component in result.components
            },
        )

    def test_collapsed_duplicate_tracks_are_not_independent_evidence(
        self,
    ) -> None:
        tracks, visibility, times = _scene(
            background_tracks=30,
            actor_tracks=6,
            seed=131,
        )
        tracks[:, -6:] = np.asarray([0.45, 0.45])
        _add_local_motion(tracks, actor_tracks=6, velocity=(0.010, 0.0))
        result = select_coherent_actor(tracks, visibility, times)
        self.assertFalse(result.diagnostic_ready)
        self.assertIn(
            "insufficient_distinct_component_tracks",
            {
                component.rejection_reason
                for component in result.components
            },
        )
        component = max(
            result.components,
            key=lambda value: value.track_count,
        )
        self.assertEqual(component.unique_track_count, 1)

    def test_frame_size_makes_equal_pixel_motion_isotropic(self) -> None:
        base, visibility, times = _scene(actor_tracks=6, seed=141)
        results = []
        for velocity in (
            (60.0 / 1920.0 / 9.0, 0.0),
            (0.0, 60.0 / 1080.0 / 9.0),
        ):
            tracks = base.copy()
            _add_local_motion(
                tracks,
                actor_tracks=6,
                velocity=velocity,
            )
            results.append(
                select_coherent_actor(
                    tracks,
                    visibility,
                    times,
                    frame_size=(1080, 1920),
                )
            )
        self.assertTrue(all(result.diagnostic_ready for result in results))
        self.assertEqual(
            results[0].coordinate_space,
            "normalized-max-side-isotropic",
        )
        np.testing.assert_allclose(
            results[0].isotropic_scale,
            (1.0, 1080.0 / 1920.0),
        )
        horizontal = float(np.linalg.norm(results[0].actor_trajectory[-1]))
        vertical = float(np.linalg.norm(results[1].actor_trajectory[-1]))
        self.assertAlmostEqual(horizontal, vertical, places=7)

    def test_canonical_per_track_tensor_survives_permutation(self) -> None:
        tracks, visibility, times = _scene(actor_tracks=8, seed=151)
        first = np.zeros(2, dtype=np.float64)
        second = np.zeros(2, dtype=np.float64)
        for frame in range(1, len(tracks)):
            if 6 <= frame < 15:
                first += np.asarray([0.010, 0.001])
                second += np.asarray([0.001, 0.010])
            tracks[frame, -8:-4] += first
            tracks[frame, -4:] += second
        baseline = select_coherent_actor(tracks, visibility, times)
        permutation = np.random.default_rng(152).permutation(
            tracks.shape[1]
        )
        permuted = select_coherent_actor(
            tracks[:, permutation],
            visibility[:, permutation],
            times,
        )
        self.assertTrue(baseline.diagnostic_ready)
        self.assertTrue(permuted.diagnostic_ready)
        np.testing.assert_allclose(
            permuted.actor_track_trajectories,
            baseline.actor_track_trajectories,
            atol=1e-7,
        )
        np.testing.assert_array_equal(
            permuted.actor_track_phase_mask,
            baseline.actor_track_phase_mask,
        )


if __name__ == "__main__":
    unittest.main()
