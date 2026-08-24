from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import fixed_grid_camera_consensus as consensus  # noqa: E402


class FixedGridCameraConsensusStaticTests(unittest.TestCase):
    def test_contract_is_fixed_global_physical_and_exact_21_phase_fp32(self) -> None:
        receipt = consensus.camera_consensus_contract_receipt()
        self.assertEqual(
            receipt["tensor_contract"],
            {"layout": "B,C,T,H,W", "dtype": "float32", "latent_phases": 21},
        )
        self.assertEqual(receipt["grid"]["rows"], 4)
        self.assertEqual(receipt["grid"]["columns"], 4)
        self.assertEqual(
            receipt["grid"]["coordinates"],
            "fixed_global_normalized_minus_one_to_one",
        )
        self.assertEqual(
            receipt["homography"]["physical_coefficients"],
            ["a", "b", "c", "d", "e", "f", "g", "h"],
        )
        self.assertEqual(
            receipt["aggregation"],
            ["coordinate_median", "scaled_MAD_rejection", "trimmed_mean"],
        )
        self.assertEqual(
            receipt["spatial_support"],
            {
                "minimum_inliers_per_row": 1,
                "minimum_inliers_per_column": 1,
                "minimum_inliers_per_quadrant": 2,
                "minimum_corner_inliers": 3,
            },
        )
        self.assertEqual(
            receipt["tile_evidence"]["maximum_relative_fit_residual"],
            0.5,
        )

    def test_runtime_projector_has_no_external_spatial_condition_surface(self) -> None:
        forbidden = ("mask", "flow", "track", "pose", "box", "point", "trajectory")
        parameters = inspect.signature(consensus.project_camera_consensus).parameters
        self.assertEqual(
            list(parameters),
            ["field", "reference_clean_field", "config", "precomputed_geometry"],
        )
        for name in parameters:
            self.assertFalse(any(token in name.lower() for token in forbidden), name)
        serialized = json.dumps(
            consensus.camera_consensus_contract_receipt()
        ).lower()
        self.assertFalse(any(f'"{token}"' in serialized for token in forbidden))

    def test_torch_import_is_lazy_and_invalid_configs_fail_without_it(self) -> None:
        tree = ast.parse(Path(consensus.__file__).read_text(encoding="utf-8"))
        eager = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                eager.extend(alias.name for alias in node.names if alias.name == "torch")
            elif isinstance(node, ast.ImportFrom) and node.module == "torch":
                eager.append(node.module)
        self.assertEqual(eager, [])
        invalid = (
            {"tile_rows": 0},
            {"tile_columns": True},
            {"minimum_valid_tile_fraction": 0.5},
            {"minimum_valid_tile_fraction": True},
            {"minimum_valid_tile_fraction": 1.1},
            {"trim_fraction": 0.5},
            {"mad_floor": -1.0},
            {"max_condition_number": 1.0},
            {"maximum_tile_relative_fit_residual": 0.0},
            {"fit_energy_floor": 0.0},
            {"minimum_inliers_per_row": 5},
            {"minimum_inliers_per_column": 5},
            {"minimum_inliers_per_quadrant": 5},
            {"minimum_corner_inliers": 5},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(consensus.CameraConsensusError):
                    consensus.CameraConsensusConfig(**kwargs).validate()


try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class FixedGridCameraConsensusTensorTests(unittest.TestCase):
    @staticmethod
    def textured_reference(*, seed: int = 1907):
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
    def normalized_gradients(reference):
        gradient_x = torch.zeros_like(reference)
        gradient_y = torch.zeros_like(reference)
        height, width = int(reference.shape[-2]), int(reference.shape[-1])
        gradient_x[..., 0] = reference[..., 1] - reference[..., 0]
        gradient_x[..., -1] = reference[..., -1] - reference[..., -2]
        gradient_x[..., 1:-1] = 0.5 * (
            reference[..., 2:] - reference[..., :-2]
        )
        gradient_y[..., 0, :] = reference[..., 1, :] - reference[..., 0, :]
        gradient_y[..., -1, :] = reference[..., -1, :] - reference[..., -2, :]
        gradient_y[..., 1:-1, :] = 0.5 * (
            reference[..., 2:, :] - reference[..., :-2, :]
        )
        gradient_x *= 0.5 * float(width - 1)
        gradient_y *= 0.5 * float(height - 1)
        return gradient_x, gradient_y

    @classmethod
    def physical_homography_field(cls, reference, coefficients):
        gradient_x, gradient_y = cls.normalized_gradients(reference)
        height, width = int(reference.shape[-2]), int(reference.shape[-1])
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
            [0.018, -0.011, 0.031, 0.009, -0.016, -0.024, 0.007, -0.006],
            dtype=torch.float32,
        )
        scale = torch.linspace(0.65, 1.15, 21, dtype=torch.float32)
        return (scale[:, None] * base[None, :]).unsqueeze(0)

    def test_global_homography_is_recovered_in_physical_coefficients(self) -> None:
        reference = self.textured_reference()
        coefficients = self.phase_coefficients()
        field = self.physical_homography_field(reference, coefficients)
        geometry = consensus.build_fixed_grid_camera_geometry(reference)

        result = consensus.project_camera_consensus(
            field,
            reference,
            precomputed_geometry=geometry,
        )

        self.assertTrue(bool(geometry.valid_tiles.all()))
        self.assertTrue(bool(result.consensus_valid.all()))
        maximum_coefficient_error = (
            result.consensus_coefficients - coefficients
        ).abs().max()
        self.assertLess(float(maximum_coefficient_error), 2.0e-5)
        relative_field_error = (
            (result.camera_component - field).square().sum()
            / field.square().sum().clamp_min(1.0e-12)
        ).sqrt()
        self.assertLess(float(relative_field_error), 2.0e-5)
        tile_error = (
            result.per_tile_coefficients - coefficients.unsqueeze(2)
        ).abs().amax()
        self.assertLess(float(tile_error), 5.0e-5)

    def test_autocast_cannot_reduce_geometry_or_projection_below_fp32(self) -> None:
        reference = self.textured_reference(seed=1979)
        coefficients = self.phase_coefficients()
        field = self.physical_homography_field(reference, coefficients)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            geometry = consensus.build_fixed_grid_camera_geometry(reference)
            result = consensus.project_camera_consensus(
                field,
                reference,
                precomputed_geometry=geometry,
            )

        self.assertEqual(geometry.tangent_matrix.dtype, torch.float32)
        self.assertTrue(
            all(
                value.dtype == torch.float32
                for value in geometry.tile_pseudoinverses
            )
        )
        self.assertEqual(result.per_tile_coefficients.dtype, torch.float32)
        self.assertEqual(result.camera_component.dtype, torch.float32)
        relative_error = (
            (result.camera_component - field).square().sum()
            / field.square().sum().clamp_min(1.0e-12)
        ).sqrt()
        self.assertLess(float(relative_error), 2.0e-5)

    def test_nonsquare_perspective_matches_grid_sample_finite_difference(self) -> None:
        import torch.nn.functional as functional

        generator = torch.Generator().manual_seed(2017)
        height, width = 20, 28
        source_phase = torch.randn(
            1,
            2,
            height,
            width,
            generator=generator,
            dtype=torch.float32,
        )
        source_phase[..., 0, :] = source_phase[..., 1, :]
        source_phase[..., -1, :] = source_phase[..., -2, :]
        source_phase[..., :, 0] = source_phase[..., :, 1]
        source_phase[..., :, -1] = source_phase[..., :, -2]
        source = source_phase.unsqueeze(2).repeat(1, 1, 21, 1, 1)
        # This test isolates the independent coordinate oracle.  A relaxed
        # MAD floor prevents finite-difference interpolation noise in the
        # seven nominally-zero coefficients from becoming a support test.
        config = consensus.CameraConsensusConfig(mad_floor=5.0e-3)
        geometry = consensus.build_fixed_grid_camera_geometry(source, config)
        y_axis = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
        x_axis = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
        y_coord, x_coord = torch.meshgrid(y_axis, x_axis, indexing="ij")
        identity = torch.stack((x_coord, y_coord), dim=-1).unsqueeze(0)
        epsilon = 1.0e-3
        vector_fields = (
            0.10 * torch.stack((-x_coord.square(), -x_coord * y_coord), dim=-1),
            0.10 * torch.stack((-x_coord * y_coord, -y_coord.square()), dim=-1),
        )
        for vector_field in vector_fields:
            plus = functional.grid_sample(
                source_phase,
                identity + epsilon * vector_field.unsqueeze(0),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            minus = functional.grid_sample(
                source_phase,
                identity - epsilon * vector_field.unsqueeze(0),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            derivative = ((plus - minus) / (2.0 * epsilon)).unsqueeze(2).repeat(
                1, 1, 21, 1, 1
            )
            result = consensus.project_camera_consensus(
                derivative,
                source,
                config=config,
                precomputed_geometry=geometry,
            )
            self.assertTrue(bool(result.consensus_valid.all()))
            relative_error = (
                (result.camera_component - derivative).square().sum()
                / derivative.square().sum().clamp_min(1.0e-12)
            ).sqrt()
            self.assertLess(float(relative_error), 1.0e-2)

    def test_local_actor_translation_and_scale_leakage_is_at_most_point_one(self) -> None:
        reference = self.textured_reference(seed=2027)
        coefficients = self.phase_coefficients()
        global_field = self.physical_homography_field(reference, coefficients)
        gradient_x, gradient_y = self.normalized_gradients(reference)
        height, width = int(reference.shape[-2]), int(reference.shape[-1])
        y_axis = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
        x_axis = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
        y_coord, x_coord = torch.meshgrid(y_axis, x_axis, indexing="ij")
        x_coord = x_coord.reshape(1, 1, 1, height, width)
        y_coord = y_coord.reshape(1, 1, 1, height, width)
        actor_support = torch.zeros(
            1, 1, 1, height, width, dtype=torch.float32
        )
        # Exactly four of sixteen fixed tiles: a strict global majority remains.
        # Deliberately not aligned to tile boundaries; it still contaminates
        # only the central four tiles and cannot supply a global majority.
        actor_support[..., 6:14, 7:17] = 1.0
        geometry = consensus.build_fixed_grid_camera_geometry(reference)

        local_fields = {
            "translation": actor_support
            * (0.070 * gradient_x - 0.052 * gradient_y),
            "scale": actor_support
            * (
                gradient_x * (0.085 * x_coord)
                + gradient_y * (0.085 * y_coord)
            ),
        }
        for label, actor_field in local_fields.items():
            with self.subTest(local_actor_transform=label):
                mixed = global_field + actor_field
                result = consensus.project_camera_consensus(
                    mixed,
                    reference,
                    precomputed_geometry=geometry,
                )
                self.assertTrue(bool(result.consensus_valid.all()))
                correction_leakage = (
                    (result.camera_component - global_field).square().sum().sqrt()
                    / actor_field.square().sum().sqrt().clamp_min(1.0e-12)
                )
                self.assertLessEqual(float(correction_leakage), 0.10)
                # The four actor tiles are rejected while twelve globally
                # agreeing tiles remain as evidence on every latent phase.
                self.assertEqual(
                    int(result.inlier_tiles.sum(dim=-1).min()),
                    12,
                )

    def test_half_frame_bimodal_motion_has_no_consensus_and_fails_exact_zero(self) -> None:
        reference = self.textured_reference(seed=3301)
        geometry = consensus.build_fixed_grid_camera_geometry(reference)
        batch, channels, phases, height, width = reference.shape
        flattened = torch.zeros(
            batch,
            phases,
            channels * height * width,
            dtype=torch.float32,
        )
        negative = torch.tensor(
            [-0.03, 0.0, -0.08, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
        ).reshape(1, 1, 8)
        positive = -negative
        for tile_index, index in enumerate(geometry.tile_indices):
            coefficient = negative if tile_index < 8 else positive
            tile = torch.matmul(
                geometry.tangent_matrix.index_select(-2, index),
                coefficient.unsqueeze(-1),
            ).squeeze(-1)
            flattened[..., index] = tile
        field = flattened.reshape(
            batch, phases, channels, height, width
        ).permute(0, 2, 1, 3, 4)

        result = consensus.project_camera_consensus(
            field,
            reference,
            precomputed_geometry=geometry,
        )

        self.assertFalse(bool(result.consensus_valid.any()))
        self.assertTrue(
            torch.equal(
                result.consensus_coefficients,
                torch.zeros_like(result.consensus_coefficients),
            )
        )
        self.assertTrue(
            torch.equal(
                result.camera_component,
                torch.zeros_like(result.camera_component),
            )
        )

    def test_three_quarter_frame_local_translation_fails_spatial_coverage(self) -> None:
        reference = torch.randn(
            1,
            16,
            21,
            20,
            24,
            generator=torch.Generator().manual_seed(123),
            dtype=torch.float32,
        )
        gradient_x, gradient_y = self.normalized_gradients(reference)
        geometry = consensus.build_fixed_grid_camera_geometry(reference)
        supports = {
            "top_three_quarters": (slice(0, 15), slice(0, 24)),
            "left_three_quarters": (slice(0, 20), slice(0, 18)),
        }
        for label, (rows, columns) in supports.items():
            with self.subTest(local_support=label):
                support = torch.zeros(
                    1, 1, 1, 20, 24, dtype=torch.float32
                )
                support[..., rows, columns] = 1.0
                field = support * (0.070 * gradient_x - 0.052 * gradient_y)
                result = consensus.project_camera_consensus(
                    field,
                    reference,
                    precomputed_geometry=geometry,
                )

                self.assertEqual(
                    int(result.inlier_tiles.sum(dim=-1).min()),
                    12,
                )
                self.assertFalse(bool(result.spatial_coverage_valid.any()))
                self.assertFalse(bool(result.consensus_valid.any()))
                self.assertTrue(
                    torch.equal(
                        result.camera_component,
                        torch.zeros_like(result.camera_component),
                    )
                )

    def test_poor_per_tile_homography_fit_cannot_vote(self) -> None:
        reference = self.textured_reference(seed=3907)
        geometry = consensus.build_fixed_grid_camera_geometry(reference)
        batch, channels, phases, height, width = reference.shape
        generator = torch.Generator().manual_seed(3911)
        flattened = torch.zeros(
            batch,
            phases,
            channels * height * width,
            dtype=torch.float32,
        )
        for index, pseudoinverse in zip(
            geometry.tile_indices,
            geometry.tile_pseudoinverses,
        ):
            candidate = torch.randn(
                batch,
                phases,
                int(index.numel()),
                generator=generator,
                dtype=torch.float32,
            )
            tile_matrix = geometry.tangent_matrix.index_select(-2, index)
            fitted = torch.matmul(
                tile_matrix,
                torch.matmul(
                    pseudoinverse,
                    candidate.unsqueeze(-1),
                ),
            ).squeeze(-1)
            flattened[..., index] = candidate - fitted
        field = flattened.reshape(
            batch, phases, channels, height, width
        ).permute(0, 2, 1, 3, 4)

        result = consensus.project_camera_consensus(
            field,
            reference,
            precomputed_geometry=geometry,
        )

        self.assertGreater(
            float(result.per_tile_relative_fit_residual.min()),
            0.99,
        )
        self.assertFalse(bool(result.fit_valid_tiles.any()))
        self.assertFalse(bool(result.consensus_valid.any()))
        self.assertTrue(
            torch.equal(
                result.camera_component,
                torch.zeros_like(result.camera_component),
            )
        )

    def test_texture_degeneracy_and_invalid_inputs_fail_closed(self) -> None:
        reference = torch.ones(1, 2, 21, 20, 24, dtype=torch.float32)
        field = torch.randn_like(reference)
        result = consensus.project_camera_consensus(field, reference)
        self.assertFalse(bool(result.geometry_valid_tiles.any()))
        self.assertFalse(bool(result.consensus_valid.any()))
        self.assertTrue(
            torch.equal(result.camera_component, torch.zeros_like(field))
        )

        with self.assertRaises(consensus.CameraConsensusError):
            consensus.project_camera_consensus(field.to(torch.float64), reference)
        malformed = torch.randn(1, 2, 20, 20, 24, dtype=torch.float32)
        with self.assertRaises(consensus.CameraConsensusError):
            consensus.project_camera_consensus(malformed, malformed)
        nonfinite = field.clone()
        nonfinite[..., 0, 0] = float("nan")
        with self.assertRaises(consensus.CameraConsensusError):
            consensus.project_camera_consensus(nonfinite, reference)

    def test_precomputed_geometry_rejects_clones_and_in_place_mutation(self) -> None:
        reference = self.textured_reference(seed=4513)
        geometry = consensus.build_fixed_grid_camera_geometry(reference)
        field = torch.zeros_like(reference)
        with self.assertRaises(consensus.CameraConsensusError):
            consensus.project_camera_consensus(
                field,
                reference.clone(),
                precomputed_geometry=geometry,
            )
        reference.add_(0.01)
        with self.assertRaises(consensus.CameraConsensusError):
            consensus.project_camera_consensus(
                field,
                reference,
                precomputed_geometry=geometry,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
