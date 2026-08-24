from __future__ import annotations

import ast
from dataclasses import fields
import importlib.util
import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import dmiq_factorized_marginal_composer as composer


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class FactorizedMarginalStaticContractTests(unittest.TestCase):
    def test_public_api_has_no_target_mask_or_proposal_carrier(self) -> None:
        signature = inspect.signature(
            composer.compose_factorized_action_marginal
        )
        names = set(signature.parameters)
        forbidden_fragments = {
            "target",
            "mask",
            "proposal",
            "rgb",
            "latent",
            "flow",
            "pose",
            "track",
            "trajectory",
        }
        for name in names:
            for fragment in forbidden_fragments:
                self.assertNotIn(fragment, name)

    def test_contract_is_generic_noop_factorization_and_fail_closed(self) -> None:
        receipt = composer.factorized_marginal_composer_contract_receipt()
        self.assertEqual(receipt["formula"], "h=h0+U_A@alpha")
        self.assertTrue(receipt["generic_source_invariant_action_basis"])
        self.assertTrue(receipt["same_current_state_frozen_noop_costate"])
        self.assertFalse(
            receipt["correct_source_specific_raw_action_residual_required"]
        )
        self.assertEqual(
            receipt["orthogonal_complement"], "P_perp(h)==P_perp(h0)"
        )
        self.assertEqual(receipt["loss_space"], "action_coefficients_only")
        self.assertFalse(receipt["optimizer_updates_authorized"])
        self.assertIn("proposal_rgb", receipt["forbidden_inputs"])
        self.assertIn("proposal_latent", receipt["forbidden_inputs"])
        self.assertIn("segmentation_mask", receipt["forbidden_inputs"])

    def test_action_and_identity_diagnostics_have_no_scalar_compensation(
        self,
    ) -> None:
        action_fields = {field.name for field in fields(
            composer.ActionMarginalDiagnostics
        )}
        identity_fields = {field.name for field in fields(
            composer.IdentityMarginalDiagnostics
        )}
        self.assertNotIn("identity_feasible", action_fields)
        self.assertNotIn("coefficient_mse", identity_fields)
        self.assertNotIn("combined_score", action_fields | identity_fields)
        self.assertIn("scalar_compensation_used", identity_fields)

    def test_module_imports_torch_lazily(self) -> None:
        source = Path(composer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        eager_torch_imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                eager_torch_imports.extend(
                    alias.name for alias in node.names if alias.name == "torch"
                )
            elif isinstance(node, ast.ImportFrom) and node.module == "torch":
                eager_torch_imports.append(node.module)
        self.assertEqual(eager_torch_imports, [])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class FactorizedMarginalTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch

    def _randn(self, shape: tuple[int, ...], seed: int):
        generator = self.torch.Generator(device="cpu").manual_seed(seed)
        return self.torch.randn(
            shape,
            generator=generator,
            dtype=self.torch.float32,
        ).contiguous()

    def _fixture(self) -> dict[str, object]:
        batch, layers, sites, phases, positions, hidden, rank = (
            2,
            2,
            2,
            3,
            4,
            6,
            2,
        )
        noop = self._randn(
            (batch, layers, sites, phases, positions, hidden),
            11,
        )
        basis = self.torch.zeros(
            (layers, sites, phases, positions, hidden, rank),
            dtype=self.torch.float32,
        )
        basis[..., 0, 0] = 1.0
        basis[..., 1, 1] = 1.0
        basis = basis.contiguous()
        pass_a = (
            0.03
            * self._randn(
                (batch, layers, sites, phases, positions, rank),
                13,
            )
        ).contiguous()
        predicted = (pass_a.clone() + 0.01).contiguous().requires_grad_(True)
        prefix = self._randn((batch, 5, hidden), 17)
        coordinates = composer.MarginalCoordinateBinding(
            schema_version=composer.COORDINATE_SCHEMA,
            layer_ids=(2, 7),
            site_ids=("self_attn", "ffn"),
            phase_ids=(0, 1, 2),
            spatial_ids=(10, 11, 12, 13),
        )
        config = composer.MarginalComposerConfig()
        return {
            "noop": noop,
            "basis": basis,
            "pass_a": pass_a,
            "predicted": predicted,
            "prefix": prefix,
            "prefix_runtime": prefix.clone().contiguous(),
            "coordinates": coordinates,
            "config": config,
        }

    def _bind(
        self,
        values: dict[str, object],
        **overrides: bool,
    ) -> composer.MarginalComposerProvenance:
        flags = {
            "same_current_state_verified": True,
            "basis_evidence_valid": True,
            "action_evidence_valid": True,
            "identity_feasible": True,
        }
        flags.update(overrides)
        return composer.bind_marginal_composer_provenance(
            values["noop"],
            values["basis"],
            values["pass_a"],
            values["prefix"],
            values["coordinates"],
            checkpoint_tree_sha256="1" * 64,
            noop_query_receipt_sha256="2" * 64,
            action_basis_evidence_receipt_sha256="3" * 64,
            identity_feasibility_receipt_sha256="4" * 64,
            config=values["config"],
            **flags,
        )

    def _run(
        self,
        values: dict[str, object],
        *,
        provenance: composer.MarginalComposerProvenance | None = None,
    ) -> composer.FactorizedMarginalResult:
        return composer.compose_factorized_action_marginal(
            values["noop"],
            values["basis"],
            values["predicted"],
            values["pass_a"],
            values["prefix"],
            values["prefix_runtime"],
            values["coordinates"],
            self._bind(values) if provenance is None else provenance,
            config=values["config"],
        )

    def test_exact_complement_and_action_coefficient_recovery(self) -> None:
        values = self._fixture()
        result = self._run(values)
        update = result.composed_tail_hidden - values["noop"]
        recovered = self.torch.einsum(
            "lstpdr,blstpd->blstpr",
            values["basis"],
            update,
        )
        reconstructed = self.torch.einsum(
            "lstpdr,blstpr->blstpd",
            values["basis"],
            recovered,
        )
        self.assertTrue(
            self.torch.allclose(
                recovered,
                result.effective_coefficients,
                atol=5.0e-6,
                rtol=0.0,
            )
        )
        self.assertTrue(
            self.torch.allclose(update, reconstructed, atol=5.0e-6, rtol=0.0)
        )

        noop_projection = self.torch.einsum(
            "lstpdr,blstpd->blstpr",
            values["basis"],
            values["noop"],
        )
        output_projection = self.torch.einsum(
            "lstpdr,blstpd->blstpr",
            values["basis"],
            result.composed_tail_hidden,
        )
        noop_complement = values["noop"] - self.torch.einsum(
            "lstpdr,blstpr->blstpd",
            values["basis"],
            noop_projection,
        )
        output_complement = result.composed_tail_hidden - self.torch.einsum(
            "lstpdr,blstpr->blstpd",
            values["basis"],
            output_projection,
        )
        self.assertTrue(
            self.torch.allclose(
                noop_complement,
                output_complement,
                atol=5.0e-6,
                rtol=0.0,
            )
        )
        self.assertTrue(result.local_composition_valid)
        self.assertTrue(
            result.identity_diagnostics.orthogonal_complement_preserved
        )
        self.assertTrue(result.action_diagnostics.coefficient_recovery_passed)
        self.assertFalse(result.optimizer_updates_authorized)
        self.assertNotEqual(
            result.composed_tail_hidden.untyped_storage().data_ptr(),
            values["noop"].untyped_storage().data_ptr(),
        )

    def test_loss_is_coefficient_only_and_gradient_reaches_prediction(
        self,
    ) -> None:
        values = self._fixture()
        result = self._run(values)
        gradient = self.torch.autograd.grad(
            result.coefficient_loss,
            values["predicted"],
        )[0]
        expected = (
            2.0
            * (values["predicted"].detach() - values["pass_a"])
            / values["predicted"].numel()
        )
        self.assertTrue(
            self.torch.allclose(gradient, expected, atol=1.0e-7, rtol=0.0)
        )
        for name in ("noop", "basis", "pass_a", "prefix"):
            self.assertFalse(values[name].requires_grad)
            self.assertIsNone(values[name].grad_fn)
        self.assertTrue(result.composed_tail_hidden.requires_grad)
        self.assertFalse(result.identity_diagnostics.scalar_compensation_used)
        self.assertFalse(
            result.receipt["training"][
                "identity_action_scalar_compensation_present"
            ]
        )

    def test_prediction_is_radially_capped_but_pass_a_overflow_nulls(
        self,
    ) -> None:
        values = self._fixture()
        values["predicted"] = self.torch.full_like(
            values["pass_a"],
            100.0,
            requires_grad=True,
        ).contiguous()
        result = self._run(values)
        effective_norm = self.torch.linalg.vector_norm(
            result.effective_coefficients,
            dim=-1,
        )
        self.assertTrue(result.action_diagnostics.coefficient_cap_applied)
        self.assertLessEqual(
            float(effective_norm.max().item()),
            result.action_diagnostics.trust_radius_max + 2.0e-6,
        )
        self.assertTrue(result.local_composition_valid)

        invalid = self._fixture()
        invalid["pass_a"] = self.torch.full_like(
            invalid["pass_a"],
            100.0,
        ).contiguous()
        invalid["predicted"] = (
            0.01 * self.torch.ones_like(invalid["pass_a"])
        ).contiguous().requires_grad_(True)
        null_result = self._run(invalid)
        self.assertTrue(
            self.torch.equal(
                null_result.composed_tail_hidden,
                invalid["noop"],
            )
        )
        self.assertTrue(
            self.torch.equal(
                null_result.effective_coefficients,
                self.torch.zeros_like(invalid["predicted"]),
            )
        )
        null_gradient = self.torch.autograd.grad(
            null_result.coefficient_loss,
            invalid["predicted"],
        )[0]
        self.assertTrue(
            self.torch.equal(null_gradient, self.torch.zeros_like(null_gradient))
        )
        self.assertFalse(null_result.local_composition_valid)
        self.assertFalse(
            null_result.action_diagnostics.pass_a_within_trust_cap
        )
        self.assertTrue(null_result.action_diagnostics.null_action_update)
        self.assertFalse(null_result.optimizer_updates_authorized)

    def test_each_invalid_scientific_gate_returns_exact_null_update(self) -> None:
        for flag in (
            "same_current_state_verified",
            "basis_evidence_valid",
            "action_evidence_valid",
            "identity_feasible",
        ):
            with self.subTest(flag=flag):
                values = self._fixture()
                provenance = self._bind(values, **{flag: False})
                result = self._run(values, provenance=provenance)
                self.assertTrue(
                    self.torch.equal(
                        result.composed_tail_hidden,
                        values["noop"],
                    )
                )
                self.assertFalse(result.local_composition_valid)
                self.assertTrue(result.action_diagnostics.null_action_update)
                self.assertFalse(result.optimizer_updates_authorized)

    def test_basis_tamper_raises_and_nonorthogonal_basis_nulls(self) -> None:
        values = self._fixture()
        provenance = self._bind(values)
        tampered_basis = values["basis"].clone().contiguous()
        tampered_basis[0, 0, 0, 0, 0, 0] = 0.5
        values["basis"] = tampered_basis
        with self.assertRaisesRegex(
            composer.DMIQFactorizedMarginalError,
            "action_basis_digest",
        ):
            self._run(values, provenance=provenance)

        nonorthogonal = self._fixture()
        bad_basis = nonorthogonal["basis"].clone().contiguous()
        bad_basis[..., 0, 0] = 2.0
        nonorthogonal["basis"] = bad_basis
        result = self._run(nonorthogonal)
        self.assertTrue(
            self.torch.equal(
                result.composed_tail_hidden,
                nonorthogonal["noop"],
            )
        )
        self.assertFalse(
            result.action_diagnostics.basis_orthogonality_passed
        )
        self.assertFalse(result.local_composition_valid)
        self.assertFalse(result.optimizer_updates_authorized)

    def test_prefix_mutation_and_alias_are_rejected(self) -> None:
        mutated = self._fixture()
        mutated_runtime = mutated["prefix_runtime"].clone().contiguous()
        mutated_runtime[0, 0, 0] += 1.0
        mutated["prefix_runtime"] = mutated_runtime
        with self.assertRaisesRegex(
            composer.DMIQFactorizedMarginalError,
            "source prefix changed",
        ):
            self._run(mutated)

        aliased = self._fixture()
        aliased["prefix_runtime"] = aliased["prefix"]
        with self.assertRaisesRegex(
            composer.DMIQFactorizedMarginalError,
            "aliases storage",
        ):
            self._run(aliased)

    def test_frozen_trainable_and_alias_boundaries(self) -> None:
        aliased = self._fixture()
        predicted_alias = aliased["pass_a"].detach()
        predicted_alias.requires_grad_(True)
        aliased["predicted"] = predicted_alias
        with self.assertRaisesRegex(
            composer.DMIQFactorizedMarginalError,
            "aliases storage",
        ):
            self._run(aliased)

        trainable_carrier = self._fixture()
        trainable_carrier["noop"] = (
            trainable_carrier["noop"].clone().requires_grad_(True)
        )
        with self.assertRaisesRegex(
            composer.DMIQFactorizedMarginalError,
            "detached and frozen",
        ):
            self._bind(trainable_carrier)

        frozen_prediction = self._fixture()
        frozen_prediction["predicted"] = (
            frozen_prediction["predicted"].detach().clone().contiguous()
        )
        with self.assertRaisesRegex(
            composer.DMIQFactorizedMarginalError,
            "trainable predicted tensor",
        ):
            self._run(frozen_prediction)

        nonleaf = self._fixture()
        raw = nonleaf["predicted"].detach().clone().requires_grad_(True)
        nonleaf["predicted"] = (raw * 1.0).contiguous()
        result = self._run(nonleaf)
        result.coefficient_loss.backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(result.local_composition_valid)

    def test_runtime_config_and_coordinate_provenance_are_exact(self) -> None:
        changed_config = self._fixture()
        provenance = self._bind(changed_config)
        changed_config["config"] = composer.MarginalComposerConfig(
            max_coefficient_norm=3.0
        )
        with self.assertRaisesRegex(
            composer.DMIQFactorizedMarginalError,
            "runtime trust/tolerance config",
        ):
            self._run(changed_config, provenance=provenance)

        changed_coordinates = self._fixture()
        provenance = self._bind(changed_coordinates)
        changed_coordinates["coordinates"] = composer.MarginalCoordinateBinding(
            schema_version=composer.COORDINATE_SCHEMA,
            layer_ids=(3, 8),
            site_ids=("self_attn", "ffn"),
            phase_ids=(0, 1, 2),
            spatial_ids=(10, 11, 12, 13),
        )
        with self.assertRaisesRegex(
            composer.DMIQFactorizedMarginalError,
            "coordinates_digest",
        ):
            self._run(changed_coordinates, provenance=provenance)


if __name__ == "__main__":
    unittest.main()
