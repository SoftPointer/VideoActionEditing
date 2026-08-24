from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import generator_native_camera_stabilizer as camera  # noqa: E402


class CameraTangentStaticContractTests(unittest.TestCase):
    def test_contract_is_tri_branch_float32_exact_21_phase(self) -> None:
        receipt = camera.camera_stabilizer_contract_receipt()
        self.assertEqual(
            receipt["tensor_contract"],
            {
                "layout": "B,C,T,H,W",
                "dtype": "float32",
                "latent_phases": 21,
                "branches": [
                    "source_clean_field",
                    "action_clean_field",
                    "noop_clean_field",
                ],
            },
        )
        self.assertEqual(receipt["basis"]["degrees_of_freedom"], 8)
        self.assertEqual(
            receipt["basis"]["parameters"],
            ["a", "b", "c", "d", "e", "f", "g", "h"],
        )
        self.assertIn("each_phase", receipt["basis"]["source_policy"])
        self.assertIn("thin_svd", receipt["basis"]["factorization"])

    def test_runtime_api_and_receipt_have_no_side_input_surface(self) -> None:
        forbidden = ("target", "mask", "flow", "track", "pose", "trajectory")
        parameters = inspect.signature(camera.stabilize_camera_tangent).parameters
        self.assertEqual(
            list(parameters),
            [
                "source_clean_field",
                "action_clean_field",
                "noop_clean_field",
                "beta",
                "enabled",
                "camera_edit_requested",
                "config",
                "precomputed_basis",
            ],
        )
        for name in parameters:
            self.assertFalse(any(token in name for token in forbidden), name)
        serialized = json.dumps(camera.camera_stabilizer_contract_receipt()).lower()
        self.assertFalse(any(token in serialized for token in forbidden))

    def test_torch_is_a_genuinely_lazy_import(self) -> None:
        tree = ast.parse(Path(camera.__file__).read_text(encoding="utf-8"))
        eager = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                eager.extend(alias.name for alias in node.names if alias.name == "torch")
            elif isinstance(node, ast.ImportFrom) and node.module == "torch":
                eager.append(node.module)
        self.assertEqual(eager, [])
        config = camera.CameraTangentConfig()
        config.validate()
        self.assertEqual(camera.EXPECTED_LATENT_PHASES, 21)

    def test_invalid_numerical_policies_fail_closed_without_torch(self) -> None:
        invalid = (
            {"relative_rank_cutoff": -1.0},
            {"absolute_rank_cutoff": float("inf")},
            {"max_condition_number": 1.0},
            {"minimum_rank": 0},
            {"minimum_rank": 9},
            {"invariance_atol": -1.0},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(camera.CameraTangentError):
                    camera.CameraTangentConfig(**kwargs).validate()


try:
    import torch
except ImportError:  # pragma: no cover - host dependent
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class CameraTangentTensorTests(unittest.TestCase):
    @staticmethod
    def textured_source(*, batch: int = 1, channels: int = 2):
        generator = torch.Generator(device="cpu").manual_seed(1701)
        return torch.randn(
            batch,
            channels,
            21,
            7,
            8,
            generator=generator,
            dtype=torch.float32,
        )

    def test_basis_is_fp32_orthonormal_rank_conditioned_and_detached(self) -> None:
        source = self.textured_source().requires_grad_(True)
        basis = camera.build_camera_tangent_basis(source)
        self.assertEqual(tuple(basis.orthonormal_vectors.shape), (1, 21, 112, 8))
        self.assertEqual(basis.orthonormal_vectors.dtype, torch.float32)
        self.assertFalse(basis.orthonormal_vectors.requires_grad)
        self.assertTrue(basis.source_was_detached)
        self.assertTrue(bool(basis.valid_phase.all()))
        self.assertTrue(bool((basis.retained_rank >= 1).all()))
        self.assertTrue(
            bool(
                (
                    basis.condition_number[basis.valid_phase]
                    <= camera.DEFAULT_MAX_CONDITION_NUMBER
                ).all()
            )
        )
        vectors = basis.orthonormal_vectors[0, 3]
        gram = vectors.transpose(0, 1) @ vectors
        keep = basis.retained_modes[0, 3]
        expected = torch.diag(keep.to(dtype=torch.float32))
        self.assertTrue(torch.allclose(gram, expected, rtol=0.0, atol=2.0e-5))
        self.assertIsNone(source.grad)

    def test_spatial_gradients_use_normalized_coordinates_on_nonsquare_grid(self) -> None:
        height, width = 5, 11
        y_axis = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
        x_axis = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
        y_coord, x_coord = torch.meshgrid(y_axis, x_axis, indexing="ij")
        image = x_coord + 2.0 * y_coord
        source = image.reshape(1, 1, 1, height, width).repeat(1, 1, 21, 1, 1)

        gradient_x, gradient_y = camera._spatial_gradients(source)

        self.assertTrue(
            torch.allclose(gradient_x, torch.ones_like(gradient_x), atol=1.0e-6)
        )
        self.assertTrue(
            torch.allclose(
                gradient_y,
                torch.full_like(gradient_y, 2.0),
                atol=1.0e-6,
            )
        )

    def test_nonsquare_perspective_tangent_matches_grid_sample_derivative(self) -> None:
        import torch.nn.functional as functional

        torch.manual_seed(1776)
        height, width = 9, 15
        source_phase = torch.randn(1, 2, height, width, dtype=torch.float32)
        # Replicated borders avoid padding derivatives contaminating the
        # central finite-difference comparison at the image boundary.
        source_phase[..., 0, :] = source_phase[..., 1, :]
        source_phase[..., -1, :] = source_phase[..., -2, :]
        source_phase[..., :, 0] = source_phase[..., :, 1]
        source_phase[..., :, -1] = source_phase[..., :, -2]
        source = source_phase.unsqueeze(2).repeat(1, 1, 21, 1, 1)
        basis = camera.build_camera_tangent_basis(source)

        y_axis = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
        x_axis = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
        y_coord, x_coord = torch.meshgrid(y_axis, x_axis, indexing="ij")
        identity = torch.stack((x_coord, y_coord), dim=-1).unsqueeze(0)
        epsilon = 1.0e-3
        vector_fields = (
            torch.stack((-x_coord.square(), -x_coord * y_coord), dim=-1),
            torch.stack((-x_coord * y_coord, -y_coord.square()), dim=-1),
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
            projected = camera.project_camera_tangent(derivative, basis)
            relative_error = (
                (projected - derivative).square().sum()
                / derivative.square().sum().clamp_min(1.0e-12)
            ).sqrt()
            self.assertLess(float(relative_error), 2.5e-3)

    def test_formula_replaces_camera_component_and_preserves_local_residual(self) -> None:
        torch.manual_seed(91)
        source = self.textured_source()
        noop = 0.2 * torch.randn_like(source)
        basis = camera.build_camera_tangent_basis(source)
        source_camera = camera.project_camera_tangent(source - noop, basis)

        raw_local = torch.zeros_like(source)
        raw_local[:, :, :, 2:5, 3:6] = torch.randn_like(
            raw_local[:, :, :, 2:5, 3:6]
        )
        local_nonrigid = raw_local - camera.project_camera_tangent(raw_local, basis)
        extra_camera = camera.project_camera_tangent(torch.randn_like(source), basis)
        action = noop + source_camera + 0.75 * extra_camera + local_nonrigid

        result = camera.stabilize_camera_tangent(
            source, action, noop, beta=1.0
        )
        expected = action + (
            camera.project_camera_tangent(source - noop, basis)
            - camera.project_camera_tangent(action - noop, basis)
        )
        self.assertTrue(
            torch.allclose(result.executed_clean_field, expected, rtol=0.0, atol=4.0e-6)
        )
        executed_camera = camera.project_camera_tangent(
            result.executed_clean_field - noop, basis
        )
        self.assertTrue(
            torch.allclose(executed_camera, source_camera, rtol=2.0e-5, atol=2.0e-5)
        )
        action_noncamera = (action - noop) - camera.project_camera_tangent(
            action - noop, basis
        )
        executed_noncamera = (result.executed_clean_field - noop) - executed_camera
        self.assertTrue(
            torch.allclose(
                executed_noncamera,
                action_noncamera,
                rtol=camera.DEFAULT_INVARIANCE_RTOL,
                atol=camera.DEFAULT_INVARIANCE_ATOL,
            )
        )
        self.assertGreater(float(local_nonrigid.square().mean().sqrt()), 0.0)
        self.assertFalse(result.trace.bypassed)
        self.assertTrue(result.trace.invariant_satisfied)
        self.assertLessEqual(
            result.trace.noncamera_invariance_max_abs,
            result.trace.noncamera_invariance_tolerance,
        )

    def test_scalar_and_per_phase_beta_implement_the_same_equation(self) -> None:
        torch.manual_seed(108)
        source = self.textured_source(batch=2, channels=1)
        noop = torch.randn_like(source)
        action = torch.randn_like(source)
        scalar = camera.stabilize_camera_tangent(
            source, action, noop, beta=0.25
        )
        shared = camera.stabilize_camera_tangent(
            source, action, noop, beta=[0.25] * 21
        )
        per_batch = camera.stabilize_camera_tangent(
            source,
            action,
            noop,
            beta=torch.full((2, 21), 0.25, dtype=torch.float32),
        )
        self.assertTrue(torch.equal(scalar.executed_clean_field, shared.executed_clean_field))
        self.assertTrue(
            torch.equal(scalar.executed_clean_field, per_batch.executed_clean_field)
        )
        self.assertEqual(scalar.trace.beta_mode, "scalar")
        self.assertEqual(shared.trace.beta_mode, "shared_per_phase")
        self.assertEqual(per_batch.trace.beta_mode, "per_batch_phase")

        phase_beta = torch.zeros(21, dtype=torch.float32)
        phase_beta[6] = 1.0
        selective = camera.stabilize_camera_tangent(
            source, action, noop, beta=phase_beta
        ).executed_clean_field
        untouched = torch.ones(21, dtype=torch.bool)
        untouched[6] = False
        self.assertTrue(torch.equal(selective[:, :, untouched], action[:, :, untouched]))

    def test_one_build_supports_40_strictly_equivalent_reuses(self) -> None:
        from unittest import mock

        torch.manual_seed(120)
        source = self.textured_source(channels=1)
        noop = torch.randn_like(source)
        action = torch.randn_like(source)
        config = camera.CameraTangentConfig(relative_rank_cutoff=2.0e-5)

        original_builder = camera.build_camera_tangent_basis
        with mock.patch.object(
            camera,
            "build_camera_tangent_basis",
            wraps=original_builder,
        ) as counted_builder:
            basis = camera.build_camera_tangent_basis(source, config)
            first = None
            for _ in range(40):
                result = camera.stabilize_camera_tangent(
                    source,
                    action,
                    noop,
                    beta=0.7,
                    config=config,
                    precomputed_basis=basis,
                )
                self.assertTrue(result.trace.basis_reused)
                self.assertTrue(result.trace.basis_built)
                if first is None:
                    first = result.executed_clean_field
                else:
                    self.assertTrue(torch.equal(result.executed_clean_field, first))
            self.assertEqual(counted_builder.call_count, 1)

        # Rebuilding through the original path and reusing the explicit basis
        # execute the identical tensor program.
        fresh = camera.stabilize_camera_tangent(
            source, action, noop, beta=0.7, config=config
        )
        self.assertTrue(torch.equal(first, fresh.executed_clean_field))
        self.assertFalse(fresh.trace.basis_reused)

    def test_precomputed_basis_mismatch_fails_closed(self) -> None:
        source = self.textured_source(channels=1)
        noop = torch.zeros_like(source)
        action = torch.randn_like(source)
        config = camera.CameraTangentConfig()
        basis = camera.build_camera_tangent_basis(source, config)

        mismatches = (
            (
                source.clone(),
                basis,
                config,
                "source identity",
            ),
            (
                source,
                replace(basis, source_shape=(1, 1, 21, 7, 7)),
                config,
                "source shape",
            ),
            (
                source,
                replace(basis, source_device=torch.device("meta")),
                config,
                "source device",
            ),
            (
                source,
                replace(basis, source_dtype=torch.float64),
                config,
                "source dtype",
            ),
            (
                source,
                basis,
                camera.CameraTangentConfig(relative_rank_cutoff=9.0e-5),
                "config signature",
            ),
        )
        for supplied_source, supplied_basis, supplied_config, message in mismatches:
            with self.subTest(message=message):
                with self.assertRaisesRegex(camera.CameraTangentError, message):
                    camera.stabilize_camera_tangent(
                        supplied_source,
                        action,
                        noop,
                        config=supplied_config,
                        precomputed_basis=supplied_basis,
                    )

        mutated_source = self.textured_source(channels=1)
        mutated_basis = camera.build_camera_tangent_basis(mutated_source)
        mutated_source.add_(0.125)
        with self.assertRaisesRegex(camera.CameraTangentError, "modified in place"):
            camera.stabilize_camera_tangent(
                mutated_source,
                action,
                noop,
                precomputed_basis=mutated_basis,
            )

    def test_each_phase_uses_its_own_source_and_partial_degeneracy_is_local(self) -> None:
        source = self.textured_source(channels=1)
        # A flat phase has no observable camera tangent, while the neighboring
        # phases remain textured and valid.  This rules out an I0-only basis.
        source[:, :, 0] = 3.0
        basis = camera.build_camera_tangent_basis(source)
        self.assertFalse(bool(basis.valid_phase[0, 0]))
        self.assertTrue(bool(basis.valid_phase[0, 1:].all()))

        noop = torch.zeros_like(source)
        action = camera.project_camera_tangent(torch.randn_like(source), basis)
        action[:, :, 0] = 7.0
        result = camera.stabilize_camera_tangent(source, action, noop, beta=1.0)
        self.assertFalse(result.trace.bypassed)
        self.assertTrue(torch.equal(result.executed_clean_field[:, :, 0], action[:, :, 0]))
        self.assertGreater(
            float((result.executed_clean_field[:, :, 1:] - action[:, :, 1:]).abs().max()),
            0.0,
        )

    def test_all_exact_bypasses_return_the_original_action_object(self) -> None:
        source = self.textured_source(channels=1)
        noop = torch.zeros_like(source)
        action = torch.randn_like(source)
        invocations = (
            (
                {"enabled": False, "precomputed_basis": object()},
                "disabled",
            ),
            ({"beta": 0.0, "precomputed_basis": object()}, "zero_beta"),
            (
                {"beta": [0.0] * 21, "precomputed_basis": object()},
                "zero_beta",
            ),
            (
                {
                    "camera_edit_requested": True,
                    "precomputed_basis": object(),
                },
                "camera_edit_requested",
            ),
        )
        for kwargs, reason in invocations:
            with self.subTest(reason=reason, kwargs=kwargs):
                result = camera.stabilize_camera_tangent(
                    source, action, noop, **kwargs
                )
                self.assertIs(result.executed_clean_field, action)
                self.assertTrue(result.trace.bypassed)
                self.assertEqual(result.trace.bypass_reason, reason)

        parity_action = noop.clone()
        parity = camera.stabilize_camera_tangent(
            source, parity_action, noop, precomputed_basis=object()
        )
        self.assertIs(parity.executed_clean_field, parity_action)
        self.assertEqual(parity.trace.bypass_reason, "action_noop_exact_parity")

        flat_source = torch.full_like(source, 2.5)
        degenerate = camera.stabilize_camera_tangent(
            flat_source, action, noop
        )
        self.assertIs(degenerate.executed_clean_field, action)
        self.assertEqual(degenerate.trace.bypass_reason, "all_phases_degenerate")
        self.assertTrue(degenerate.trace.basis_built)
        self.assertFalse(bool(degenerate.trace.valid_phase.any()))

    def test_source_is_detached_but_action_path_remains_differentiable(self) -> None:
        source = self.textured_source(channels=1).requires_grad_(True)
        noop = torch.zeros_like(source)
        action = torch.randn_like(source).requires_grad_(True)
        result = camera.stabilize_camera_tangent(
            source, action, noop, beta=0.6
        )
        result.executed_clean_field.square().mean().backward()
        self.assertIsNone(source.grad)
        self.assertIsNotNone(action.grad)
        self.assertTrue(bool(torch.isfinite(action.grad).all()))
        self.assertGreater(float(action.grad.abs().sum()), 0.0)

    def test_trace_receipt_is_json_safe_and_explicit(self) -> None:
        source = self.textured_source(channels=1)
        noop = torch.zeros_like(source)
        action = torch.randn_like(source)
        receipt = camera.stabilize_camera_tangent(
            source, action, noop, beta=0.5
        ).trace.to_receipt()
        serialized = json.dumps(receipt, allow_nan=False)
        self.assertIn("retained_rank", receipt)
        self.assertIn("condition_number", receipt)
        self.assertIn("noncamera_invariance_max_abs", receipt)
        self.assertTrue(receipt["source_basis_detached"])
        self.assertFalse(receipt["basis_reused"])
        self.assertTrue(receipt["invariant_satisfied"])
        forbidden = ("target", "mask", "flow", "track", "pose", "trajectory")
        self.assertFalse(any(token in serialized.lower() for token in forbidden))

    def test_invalid_tensor_and_control_contracts_fail_closed(self) -> None:
        source = self.textured_source(channels=1)
        noop = torch.zeros_like(source)
        action = torch.randn_like(source)
        invalid_calls = (
            lambda: camera.stabilize_camera_tangent(
                source.double(), action, noop
            ),
            lambda: camera.stabilize_camera_tangent(
                source[:, :, :20], action[:, :, :20], noop[:, :, :20]
            ),
            lambda: camera.stabilize_camera_tangent(
                source, action[:, :, :, :, :-1], noop
            ),
            lambda: camera.stabilize_camera_tangent(
                source, action, noop, beta=torch.zeros(20)
            ),
            lambda: camera.stabilize_camera_tangent(
                source, action, noop, enabled=1
            ),
            lambda: camera.stabilize_camera_tangent(
                source, action, noop, camera_edit_requested=1
            ),
        )
        for invocation in invalid_calls:
            with self.subTest(invocation=invocation):
                with self.assertRaises(camera.CameraTangentError):
                    invocation()


if __name__ == "__main__":
    unittest.main()
