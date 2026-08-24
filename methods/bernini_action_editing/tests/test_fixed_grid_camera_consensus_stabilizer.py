from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import fixed_grid_camera_consensus as camera_core  # noqa: E402
import fixed_grid_camera_consensus_stabilizer as stabilizer  # noqa: E402


class FixedGridCameraConsensusStabilizerStaticTests(unittest.TestCase):
    def test_api_has_only_source_action_beta_config_and_geometry(self) -> None:
        parameters = inspect.signature(
            stabilizer.stabilize_camera_consensus
        ).parameters
        self.assertEqual(
            list(parameters),
            [
                "source_clean_field",
                "action_clean_field",
                "beta",
                "config",
                "precomputed_geometry",
            ],
        )
        forbidden = ("mask", "flow", "track", "pose", "box", "point", "trajectory")
        for name in parameters:
            self.assertFalse(any(token in name.lower() for token in forbidden), name)

    def test_receipt_states_robust_per_phase_estimator_without_orthogonal_claim(self) -> None:
        receipt = stabilizer.camera_consensus_stabilizer_contract_receipt()
        self.assertEqual(receipt["execution"], "Xa+beta*C_consensus(S-Xa)")
        self.assertIn("robust_consensus", receipt["estimator"])
        self.assertEqual(
            receipt["consensus_scope"],
            "independent_per_batch_and_latent_phase",
        )
        self.assertEqual(
            receipt["zero_beta"],
            "original_action_object_passthrough_without_geometry",
        )
        self.assertEqual(
            receipt["invalid_phase"], "exact_action_value_passthrough"
        )
        self.assertIn("inlier_tile_count", receipt["diagnostics"])
        self.assertIn("spatial_coverage_valid", receipt["diagnostics"])
        self.assertNotIn("orthogonal", json.dumps(receipt).lower())


try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class FixedGridCameraConsensusStabilizerTensorTests(unittest.TestCase):
    @staticmethod
    def textured_source(*, seed: int = 6203):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(
            1,
            2,
            21,
            20,
            24,
            generator=generator,
            dtype=torch.float32,
        )

    @staticmethod
    def normalized_gradients(source):
        gradient_x = torch.zeros_like(source)
        gradient_y = torch.zeros_like(source)
        height, width = int(source.shape[-2]), int(source.shape[-1])
        gradient_x[..., 0] = source[..., 1] - source[..., 0]
        gradient_x[..., -1] = source[..., -1] - source[..., -2]
        gradient_x[..., 1:-1] = 0.5 * (
            source[..., 2:] - source[..., :-2]
        )
        gradient_y[..., 0, :] = source[..., 1, :] - source[..., 0, :]
        gradient_y[..., -1, :] = source[..., -1, :] - source[..., -2, :]
        gradient_y[..., 1:-1, :] = 0.5 * (
            source[..., 2:, :] - source[..., :-2, :]
        )
        gradient_x *= 0.5 * float(width - 1)
        gradient_y *= 0.5 * float(height - 1)
        return gradient_x, gradient_y

    @classmethod
    def physical_homography_field(cls, source, coefficients):
        gradient_x, gradient_y = cls.normalized_gradients(source)
        height, width = int(source.shape[-2]), int(source.shape[-1])
        y_axis = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
        x_axis = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
        y_coord, x_coord = torch.meshgrid(y_axis, x_axis, indexing="ij")
        x_coord = x_coord.reshape(1, 1, 1, height, width)
        y_coord = y_coord.reshape(1, 1, 1, height, width)
        value = coefficients[:, None, :, None, None, :]
        dx = (
            value[..., 0] * x_coord
            + value[..., 1] * y_coord
            + value[..., 2]
            - value[..., 6] * x_coord.square()
            - value[..., 7] * x_coord * y_coord
        )
        dy = (
            value[..., 3] * x_coord
            + value[..., 4] * y_coord
            + value[..., 5]
            - value[..., 6] * x_coord * y_coord
            - value[..., 7] * y_coord.square()
        )
        return gradient_x * dx + gradient_y * dy

    @staticmethod
    def phase_coefficients():
        base = torch.tensor(
            [0.017, -0.010, 0.029, 0.008, -0.015, -0.022, 0.006, -0.005],
            dtype=torch.float32,
        )
        phase_scale = torch.linspace(0.70, 1.20, 21, dtype=torch.float32)
        return (phase_scale[:, None] * base[None, :]).unsqueeze(0)

    @classmethod
    def actor_fields(cls, source):
        gradient_x, gradient_y = cls.normalized_gradients(source)
        height, width = int(source.shape[-2]), int(source.shape[-1])
        y_coord, x_coord = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, width, dtype=torch.float32),
            indexing="ij",
        )
        x_coord = x_coord.reshape(1, 1, 1, height, width)
        y_coord = y_coord.reshape(1, 1, 1, height, width)
        support = torch.zeros(1, 1, 1, height, width, dtype=torch.float32)
        support[..., 6:14, 7:17] = 1.0
        return {
            "translation": support * (0.072 * gradient_x - 0.049 * gradient_y),
            "scale": support
            * (
                gradient_x * (0.082 * x_coord)
                + gradient_y * (0.082 * y_coord)
            ),
        }

    def test_beta_one_removes_a_global_camera_residual(self) -> None:
        source = self.textured_source()
        global_camera = self.physical_homography_field(
            source, self.phase_coefficients()
        )
        action = source - global_camera

        result = stabilizer.stabilize_camera_consensus(
            source, action, beta=1.0
        )

        self.assertTrue(bool(result.projection.consensus_valid.all()))
        relative_error = (
            (result.executed_clean_field - source).square().sum()
            / source.square().sum().clamp_min(1.0e-12)
        ).sqrt()
        self.assertLess(float(relative_error), 2.0e-6)
        self.assertTrue(result.trace.geometry_built)
        self.assertFalse(result.trace.geometry_reused)
        self.assertEqual(
            int(result.trace.geometry_valid_tile_count.min()),
            16,
        )
        self.assertEqual(int(result.trace.fit_valid_tile_count.min()), 16)
        self.assertGreaterEqual(int(result.trace.inlier_tile_count.min()), 10)
        self.assertTrue(bool(result.trace.spatial_coverage_valid.all()))
        self.assertTrue(
            bool(torch.isfinite(result.trace.consensus_coefficient_max_abs).all())
        )
        self.assertTrue(
            bool(torch.isfinite(result.trace.tile_relative_fit_residual_max).all())
        )

    def test_local_actor_translation_and_scale_are_preserved(self) -> None:
        source = self.textured_source(seed=6337)
        geometry = camera_core.build_fixed_grid_camera_geometry(source)
        for label, actor_field in self.actor_fields(source).items():
            with self.subTest(local_actor_transform=label):
                action = source + actor_field
                result = stabilizer.stabilize_camera_consensus(
                    source,
                    action,
                    beta=1.0,
                    precomputed_geometry=geometry,
                )
                self.assertTrue(
                    torch.equal(result.executed_clean_field, action)
                )
                self.assertTrue(result.trace.invalid_phases_exact_action)
                self.assertTrue(
                    torch.equal(
                        result.projection.camera_component,
                        torch.zeros_like(action),
                    )
                )

    def test_mixed_global_and_local_motion_removes_only_global_consensus(self) -> None:
        source = self.textured_source(seed=6451)
        global_camera = self.physical_homography_field(
            source, self.phase_coefficients()
        )
        geometry = camera_core.build_fixed_grid_camera_geometry(source)
        for label, actor_field in self.actor_fields(source).items():
            with self.subTest(local_actor_transform=label):
                action = source - global_camera + actor_field
                expected = source + actor_field
                result = stabilizer.stabilize_camera_consensus(
                    source,
                    action,
                    beta=1.0,
                    precomputed_geometry=geometry,
                )
                relative_error = (
                    (result.executed_clean_field - expected).square().sum()
                    / global_camera.square().sum().clamp_min(1.0e-12)
                ).sqrt()
                self.assertLess(float(relative_error), 5.0e-6)
                actor_change = (
                    result.executed_clean_field - source
                )
                actor_relative_error = (
                    (actor_change - actor_field).square().sum().sqrt()
                    / actor_field.square().sum().sqrt().clamp_min(1.0e-12)
                )
                self.assertLess(float(actor_relative_error), 1.0e-4)

    def test_zero_beta_returns_original_action_object_without_geometry(self) -> None:
        source = self.textured_source(seed=6571)
        action = torch.randn_like(source)
        with mock.patch.object(
            camera_core, "project_camera_consensus"
        ) as projector:
            result = stabilizer.stabilize_camera_consensus(
                source,
                action,
                beta=0.0,
                # The bypass must not even validate a supplied precomputation.
                precomputed_geometry=object(),
            )
        projector.assert_not_called()
        self.assertIs(result.executed_clean_field, action)
        self.assertIsNone(result.projection)
        self.assertTrue(result.trace.bypassed)
        self.assertEqual(result.trace.bypass_reason, "zero_beta")
        self.assertFalse(result.trace.geometry_built)
        self.assertFalse(result.trace.geometry_reused)
        self.assertIsNone(result.trace.geometry_valid_tile_count)
        self.assertIsNone(result.trace.fit_valid_tile_count)
        self.assertIsNone(result.trace.inlier_tile_count)
        self.assertIsNone(result.trace.spatial_coverage_valid)

    def test_beta_outside_closed_unit_interval_fails_closed(self) -> None:
        source = self.textured_source(seed=6629)
        action = torch.randn_like(source)
        invalid = (-0.01, 1.01, float("nan"), [0.5] * 20 + [1.1])
        for beta in invalid:
            with self.subTest(beta=beta), self.assertRaises(
                camera_core.CameraConsensusError
            ), mock.patch.object(
                camera_core, "project_camera_consensus"
            ) as projector:
                stabilizer.stabilize_camera_consensus(
                    source,
                    action,
                    beta=beta,
                )
            projector.assert_not_called()

    def test_nonconsensus_phase_is_bitwise_unchanged(self) -> None:
        source = self.textured_source(seed=6689)
        geometry = camera_core.build_fixed_grid_camera_geometry(source)
        batch, channels, phases, height, width = source.shape
        residual_flat = torch.zeros(
            batch,
            phases,
            channels * height * width,
            dtype=torch.float32,
        )
        negative = torch.tensor(
            [-0.025, 0.0, -0.075, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
        ).reshape(1, 8)
        positive = -negative
        phase_index = 9
        for tile_number, index in enumerate(geometry.tile_indices):
            coefficient = negative if tile_number < 8 else positive
            tile = torch.matmul(
                geometry.tangent_matrix[:, phase_index].index_select(-2, index),
                coefficient.unsqueeze(-1),
            ).squeeze(-1)
            residual_flat[:, phase_index, index] = tile
        residual = residual_flat.reshape(
            batch, phases, channels, height, width
        ).permute(0, 2, 1, 3, 4)
        action = source - residual

        result = stabilizer.stabilize_camera_consensus(
            source,
            action,
            beta=1.0,
            precomputed_geometry=geometry,
        )

        self.assertFalse(bool(result.projection.consensus_valid[0, phase_index]))
        self.assertTrue(
            torch.equal(
                result.executed_clean_field[:, :, phase_index],
                action[:, :, phase_index],
            )
        )
        self.assertTrue(result.trace.invalid_phases_exact_action)

    def test_geometry_build_and_strict_reuse_are_reported(self) -> None:
        source = self.textured_source(seed=6791)
        camera_field = self.physical_homography_field(
            source, self.phase_coefficients()
        )
        action = source - camera_field

        with mock.patch.object(
            camera_core,
            "build_fixed_grid_camera_geometry",
            wraps=camera_core.build_fixed_grid_camera_geometry,
        ) as builder:
            built = stabilizer.stabilize_camera_consensus(source, action)
            self.assertEqual(builder.call_count, 1)
        self.assertTrue(built.trace.geometry_built)
        self.assertFalse(built.trace.geometry_reused)

        geometry = camera_core.build_fixed_grid_camera_geometry(source)
        with mock.patch.object(
            camera_core,
            "build_fixed_grid_camera_geometry",
            wraps=camera_core.build_fixed_grid_camera_geometry,
        ) as builder:
            reused = stabilizer.stabilize_camera_consensus(
                source,
                action,
                precomputed_geometry=geometry,
            )
            builder.assert_not_called()
        self.assertFalse(reused.trace.geometry_built)
        self.assertTrue(reused.trace.geometry_reused)
        self.assertTrue(
            torch.allclose(
                built.executed_clean_field,
                reused.executed_clean_field,
                rtol=0.0,
                atol=0.0,
            )
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
