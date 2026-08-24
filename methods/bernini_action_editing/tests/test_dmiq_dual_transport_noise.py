from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import dmiq_dual_transport_noise as dual


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class DMIQDualTransportNoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch

    def _randn(self, shape: tuple[int, ...], seed: int):
        generator = self.torch.Generator(device="cpu").manual_seed(seed)
        return self.torch.randn(
            shape, generator=generator, dtype=self.torch.float32
        ).contiguous()

    def _inputs(
        self,
        *,
        batch: int = 3,
        channels: int = 2,
        positions: int = 3,
        gate_value: float | None = 0.35,
    ) -> dict[str, object]:
        shape = (batch, channels, dual.LATENT_PHASES, positions)
        identity = self.torch.eye(positions, dtype=self.torch.float32).repeat(
            dual.LATENT_PHASES, 1, 1
        )
        if gate_value is None:
            gate = self.torch.rand(
                (batch, 1, dual.LATENT_PHASES, positions),
                generator=self.torch.Generator(device="cpu").manual_seed(97),
                dtype=self.torch.float32,
            )
        else:
            gate = self.torch.full(
                (batch, 1, dual.LATENT_PHASES, positions),
                gate_value,
                dtype=self.torch.float32,
            )
        return {
            "source_base": self._randn((batch, channels, positions), 11),
            "source_innovations": self._randn(shape, 13),
            "action_base": self._randn((batch, channels, positions), 17),
            "action_innovations": self._randn(shape, 19),
            "source_transport": identity.clone().contiguous(),
            "action_transport": identity.clone().contiguous(),
            "pre_noise_gate": gate.contiguous(),
        }

    def _run(
        self,
        values: dict[str, object],
        *,
        k_source: float = 0.6,
        k_action: float = 0.4,
    ):
        return dual.dmiq_dual_transport_noise(
            values["source_base"],
            values["source_innovations"],
            values["action_base"],
            values["action_innovations"],
            values["source_transport"],
            values["action_transport"],
            values["pre_noise_gate"],
            k_source=k_source,
            k_action=k_action,
        )

    def _signed_controls(self, positions: int) -> dict[str, object]:
        indices = self.torch.arange(positions, dtype=self.torch.int64)
        source_permutation = self.torch.stack(
            [
                self.torch.roll(indices, shifts=phase)
                for phase in range(dual.LATENT_PHASES)
            ],
            dim=0,
        ).contiguous()
        action_permutation = self.torch.stack(
            [
                self.torch.roll(indices.flip(0), shifts=2 * phase)
                for phase in range(dual.LATENT_PHASES)
            ],
            dim=0,
        ).contiguous()
        phase = self.torch.arange(dual.LATENT_PHASES).view(-1, 1)
        position = self.torch.arange(positions).view(1, -1)
        source_signs = self.torch.where(
            (phase + position) % 2 == 0,
            self.torch.tensor(1.0),
            self.torch.tensor(-1.0),
        ).to(dtype=self.torch.float32).contiguous()
        action_signs = self.torch.where(
            (2 * phase + position) % 3 == 0,
            self.torch.tensor(-1.0),
            self.torch.tensor(1.0),
        ).to(dtype=self.torch.float32).contiguous()
        return {
            "source_permutation": source_permutation,
            "source_signs": source_signs,
            "action_permutation": action_permutation,
            "action_signs": action_signs,
        }

    def _signed_run(
        self,
        values: dict[str, object],
        controls: dict[str, object],
        *,
        k_source: float = 0.6,
        k_action: float = 0.4,
    ):
        return dual.dmiq_dual_signed_permutation_noise(
            values["source_base"],
            values["source_innovations"],
            values["action_base"],
            values["action_innovations"],
            controls["source_permutation"],
            controls["source_signs"],
            controls["action_permutation"],
            controls["action_signs"],
            values["pre_noise_gate"],
            k_source=k_source,
            k_action=k_action,
        )

    def _explicit_signed_permutation(
        self, permutation: object, signs: object
    ):
        phases, positions = (int(item) for item in permutation.shape)
        matrix = self.torch.zeros(
            (phases, positions, positions), dtype=self.torch.float32
        )
        return matrix.scatter_(
            dim=2,
            index=permutation.unsqueeze(-1),
            src=signs.unsqueeze(-1),
        ).contiguous()

    def test_gate_endpoints_select_source_and_action_marginals(self) -> None:
        source_values = self._inputs(gate_value=0.0)
        source_result = self._run(source_values, k_source=0.6, k_action=0.4)
        source_expected = (
            0.6**0.5
            * self.torch.einsum(
                "tpq,bcq->bctp",
                source_values["source_transport"],
                source_values["source_base"],
            )
            + 0.4**0.5 * source_values["source_innovations"]
        )
        self.assertTrue(
            self.torch.equal(source_result.initial_noise, source_expected)
        )

        action_values = self._inputs(gate_value=1.0)
        action_result = self._run(action_values, k_source=0.6, k_action=0.4)
        action_expected = (
            0.4**0.5
            * self.torch.einsum(
                "tpq,bcq->bctp",
                action_values["action_transport"],
                action_values["action_base"],
            )
            + 0.6**0.5 * action_values["action_innovations"]
        )
        self.assertTrue(
            self.torch.equal(action_result.initial_noise, action_expected)
        )
        self.assertTrue(
            source_result.diagnostics.joint_covariance_non_iid_rank_certificate
        )
        self.assertTrue(
            action_result.diagnostics.joint_covariance_non_iid_rank_certificate
        )
        self.assertTrue(
            all(
                source_result.diagnostics.source_shared_carrier_effective_per_batch
            )
        )
        self.assertFalse(
            any(
                source_result.diagnostics.action_shared_carrier_effective_per_batch
            )
        )
        self.assertFalse(
            any(
                action_result.diagnostics.source_shared_carrier_effective_per_batch
            )
        )
        self.assertTrue(
            all(
                action_result.diagnostics.action_shared_carrier_effective_per_batch
            )
        )

    def test_phase_specific_rotation_is_applied_to_base_field(self) -> None:
        values = self._inputs(
            batch=1, channels=1, positions=2, gate_value=0.0
        )
        values["source_base"] = self.torch.tensor(
            [[[1.0, 0.0]]], dtype=self.torch.float32
        )
        values["source_innovations"] = self.torch.zeros(
            (1, 1, dual.LATENT_PHASES, 2), dtype=self.torch.float32
        )
        rotations = []
        expected = []
        for phase in range(dual.LATENT_PHASES):
            angle = phase * 0.07
            cosine = float(self.torch.cos(self.torch.tensor(angle)).item())
            sine = float(self.torch.sin(self.torch.tensor(angle)).item())
            rotations.append([[cosine, -sine], [sine, cosine]])
            expected.append([cosine, sine])
        values["source_transport"] = self.torch.tensor(
            rotations, dtype=self.torch.float32
        ).contiguous()
        result = self._run(values, k_source=1.0, k_action=0.5)
        expected_tensor = self.torch.tensor(
            expected, dtype=self.torch.float32
        ).view(1, 1, dual.LATENT_PHASES, 2)
        self.assertTrue(
            self.torch.allclose(
                result.initial_noise, expected_tensor, atol=1.0e-6, rtol=0.0
            )
        )

    def test_variance_identity_is_analytic_not_a_sample_statistic(self) -> None:
        values = self._inputs(positions=4, gate_value=None)
        for label, seed in (
            ("source_transport", 101),
            ("action_transport", 103),
        ):
            matrices = self._randn((dual.LATENT_PHASES, 4, 4), seed)
            orthogonal, _ = self.torch.linalg.qr(matrices)
            values[label] = orthogonal.contiguous()
        result = self._run(values, k_source=0.73, k_action=0.29)
        diagnostics = result.diagnostics
        self.assertTrue(diagnostics.analytic_per_coordinate_variance_passed)
        self.assertLess(
            diagnostics.output_variance_coefficient_max_abs_error, 1.0e-5
        )
        self.assertAlmostEqual(
            diagnostics.output_variance_coefficient_min, 1.0, places=5
        )
        self.assertAlmostEqual(
            diagnostics.output_variance_coefficient_max, 1.0, places=5
        )
        distribution = result.receipt["distribution"]
        self.assertTrue(distribution["analytic_per_coordinate_variance_one"])
        self.assertFalse(
            distribution["variance_verified_from_realized_sample_statistics"]
        )

    def test_shared_base_produces_expected_empirical_temporal_covariance(self) -> None:
        # Diagnostic test only: the implementation's correctness gate above is
        # analytic and never uses this realized covariance estimate.
        batch = 10_000
        values = self._inputs(
            batch=batch, channels=1, positions=1, gate_value=0.0
        )
        result = self._run(values, k_source=0.64, k_action=0.5)
        first = result.initial_noise[:, 0, 0, 0]
        second = result.initial_noise[:, 0, 1, 0]
        covariance = ((first - first.mean()) * (second - second.mean())).mean()
        self.assertAlmostEqual(float(covariance.item()), 0.64, delta=0.05)
        self.assertTrue(result.receipt["distribution"]["joint_covariance_non_iid"])
        self.assertFalse(result.receipt["distribution"]["native_iid_gaussian_claim"])

    def test_implicit_signed_permutation_matches_explicit_dense_small_p(self) -> None:
        parameters = inspect.signature(
            dual.dmiq_dual_signed_permutation_noise
        ).parameters
        self.assertIn("source_permutation_indices", parameters)
        self.assertIn("action_permutation_indices", parameters)
        positions = 5
        values = self._inputs(positions=positions, gate_value=None)
        controls = self._signed_controls(positions)
        signed = self._signed_run(
            values, controls, k_source=0.67, k_action=0.31
        )

        dense_values = dict(values)
        dense_values["source_transport"] = self._explicit_signed_permutation(
            controls["source_permutation"], controls["source_signs"]
        )
        dense_values["action_transport"] = self._explicit_signed_permutation(
            controls["action_permutation"], controls["action_signs"]
        )
        dense = self._run(
            dense_values, k_source=0.67, k_action=0.31
        )
        self.assertTrue(
            self.torch.allclose(
                signed.initial_noise,
                dense.initial_noise,
                atol=1.0e-6,
                rtol=0.0,
            )
        )
        self.assertEqual(
            signed.receipt["transport_representation"],
            "implicit_signed_permutation",
        )
        self.assertEqual(
            signed.receipt["transport"]["representation"],
            "implicit_signed_permutation",
        )
        self.assertFalse(
            signed.receipt["transport"]["dense_transport_materialized"]
        )
        self.assertEqual(
            dense.receipt["transport"]["representation"],
            "explicit_dense_matrix",
        )
        self.assertTrue(
            dense.receipt["transport"]["dense_validation_or_small_p_control_only"]
        )
        self.assertEqual(signed.receipt["transport"]["current_P"], positions)
        self.assertEqual(
            signed.receipt["transport"]["practical_latent_grid"]["P"],
            3720,
        )
        self.assertFalse(
            signed.receipt["transport"]["current_P_must_equal_practical_P"]
        )

    def test_practical_62x60_grid_uses_only_linear_size_controls(self) -> None:
        positions = 62 * 60
        values = {
            "source_base": self._randn((1, 1, positions), 211),
            "source_innovations": self._randn(
                (1, 1, dual.LATENT_PHASES, positions), 223
            ),
            "action_base": self._randn((1, 1, positions), 227),
            "action_innovations": self._randn(
                (1, 1, dual.LATENT_PHASES, positions), 229
            ),
            "pre_noise_gate": self.torch.full(
                (1, 1, dual.LATENT_PHASES, positions),
                0.4,
                dtype=self.torch.float32,
            ),
        }
        controls = self._signed_controls(positions)
        result = self._signed_run(values, controls)
        self.assertEqual(
            tuple(result.initial_noise.shape),
            (1, 1, dual.LATENT_PHASES, positions),
        )
        self.assertEqual(
            result.receipt["transport"]["storage_complexity"], "O(21P)"
        )
        self.assertFalse(
            result.receipt["transport"]["dense_transport_materialized"]
        )
        self.assertEqual(
            controls["source_permutation"].numel(),
            dual.LATENT_PHASES * positions,
        )

    def test_signed_permutation_indices_signs_and_alias_fail_closed(self) -> None:
        values = self._inputs(positions=4)

        duplicate = self._signed_controls(4)
        duplicate["source_permutation"][0, 0] = duplicate[
            "source_permutation"
        ][0, 1]
        with self.assertRaisesRegex(
            dual.DMIQDualTransportNoiseError, "exactly once per phase"
        ):
            self._signed_run(values, duplicate)

        out_of_range = self._signed_controls(4)
        out_of_range["action_permutation"][2, 3] = 4
        with self.assertRaisesRegex(
            dual.DMIQDualTransportNoiseError, "exactly once per phase"
        ):
            self._signed_run(values, out_of_range)

        wrong_index_dtype = self._signed_controls(4)
        wrong_index_dtype["source_permutation"] = wrong_index_dtype[
            "source_permutation"
        ].to(dtype=self.torch.int32)
        with self.assertRaisesRegex(
            dual.DMIQDualTransportNoiseError, "torch.int64"
        ):
            self._signed_run(values, wrong_index_dtype)

        invalid_sign = self._signed_controls(4)
        invalid_sign["action_signs"][0, 0] = 0.0
        with self.assertRaisesRegex(
            dual.DMIQDualTransportNoiseError,
            r"only exact elementwise -1 or \+1",
        ):
            self._signed_run(values, invalid_sign)

        aliased = self._signed_controls(4)
        aliased["action_permutation"] = aliased["source_permutation"]
        with self.assertRaisesRegex(
            dual.DMIQDualTransportNoiseError, "aliases storage"
        ):
            self._signed_run(values, aliased)

    def test_receipt_limits_itself_to_numeric_not_provenance_closure(self) -> None:
        result = self._run(self._inputs(gate_value=0.25))
        receipt = result.receipt
        self.assertTrue(receipt["ablation_only"])
        self.assertFalse(receipt["scientific_claim_authorized"])
        self.assertTrue(receipt["pure_tensor_operator"])
        self.assertFalse(receipt["gate"]["external_mask"])
        self.assertFalse(
            receipt["gate"]["origin_mechanically_inferable_from_tensor_values"]
        )
        self.assertTrue(receipt["transport"]["origin_is_caller_contract"])
        closure = receipt["condition_closure_limit"]
        self.assertFalse(
            closure["direct_media_or_control_tensor_arguments_accepted"]
        )
        self.assertFalse(
            closure[
                "absence_of_target_mask_flow_pose_or_track_derivation_proved"
            ]
        )
        self.assertFalse(
            closure["numeric_operator_receipt_is_condition_closure_evidence"]
        )
        self.assertIn(
            "pass_a_action_plan_receipt",
            closure["runtime_provenance_binding_still_required"],
        )
        self.assertFalse(
            receipt["gate"][
                "train_inference_same_builder_identity_mechanically_verified"
            ]
        )
        serialized = json.dumps(receipt, sort_keys=True)
        self.assertIn('"joint_covariance_non_iid": true', serialized)
        self.assertNotIn('"segmentation_mask_injected": false', serialized)
        self.assertIn(
            '"upstream_gate_provenance_mechanically_verified": false',
            serialized,
        )

    def test_alias_dtype_gradient_shape_gate_and_k_fail_closed(self) -> None:
        cases: list[tuple[str, dict[str, object], dict[str, float], str]] = []

        aliased = self._inputs()
        aliased["action_base"] = aliased["source_base"]
        cases.append(("alias", aliased, {}, "aliases storage"))

        wrong_dtype = self._inputs()
        wrong_dtype["action_base"] = wrong_dtype["action_base"].double()
        cases.append(("dtype", wrong_dtype, {}, "torch.float32"))

        gradient = self._inputs()
        gradient["source_base"] = gradient["source_base"].requires_grad_(True)
        cases.append(("gradient", gradient, {}, "detached/no-grad"))

        nonfinite = self._inputs()
        nonfinite["action_innovations"][0, 0, 0, 0] = float("nan")
        cases.append(("finite", nonfinite, {}, "must be finite"))

        noncontiguous = self._inputs()
        noncontiguous["action_transport"] = noncontiguous[
            "action_transport"
        ].transpose(-1, -2)
        cases.append(("contiguous", noncontiguous, {}, "must be contiguous"))

        wrong_shape = self._inputs()
        wrong_shape["source_innovations"] = self._randn((3, 2, 20, 3), 31)
        cases.append(("shape", wrong_shape, {}, "must have shape"))

        bad_gate = self._inputs()
        bad_gate["pre_noise_gate"][0, 0, 0, 0] = 1.01
        cases.append(("gate", bad_gate, {}, r"elementwise in \[0,1\]"))

        nonorthogonal = self._inputs()
        nonorthogonal["source_transport"][0, 0, 0] = 2.0
        cases.append(("orthogonal", nonorthogonal, {}, "not orthogonal"))

        no_carrier = self._inputs()
        cases.append(
            (
                "non_iid",
                no_carrier,
                {"k_source": 0.0, "k_action": 0.0},
                "lacks the conservative carrier-rank certificate",
            )
        )

        for name, values, kwargs, pattern in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    dual.DMIQDualTransportNoiseError, pattern
                ):
                    self._run(values, **kwargs)

        with self.assertRaisesRegex(
            dual.DMIQDualTransportNoiseError, r"k_source.*\[0,1\]"
        ):
            self._run(self._inputs(), k_source=-0.1)

    def test_offline_float64_cpu_polar_projection(self) -> None:
        base = self.torch.tensor(
            [[1.0, 0.25], [-0.15, 0.8]], dtype=self.torch.float32
        )
        candidate = base.repeat(dual.LATENT_PHASES, 1, 1).contiguous()
        result = dual.polar_orthogonalize_cpu(candidate)
        transport = result.transport
        identity = self.torch.eye(2, dtype=self.torch.float32).repeat(
            dual.LATENT_PHASES, 1, 1
        )
        self.assertEqual(transport.dtype, self.torch.float32)
        self.assertEqual(transport.device.type, "cpu")
        self.assertFalse(transport.requires_grad)
        self.assertNotEqual(
            transport.untyped_storage().data_ptr(),
            candidate.untyped_storage().data_ptr(),
        )
        self.assertTrue(
            self.torch.allclose(
                transport @ transport.transpose(-1, -2),
                identity,
                atol=3.0e-5,
                rtol=0.0,
            )
        )
        self.assertEqual(result.receipt["computation_dtype"], "torch.float64")
        self.assertEqual(result.receipt["computation_device"], "cpu")

        singular = self.torch.zeros(
            (dual.LATENT_PHASES, 2, 2), dtype=self.torch.float32
        )
        with self.assertRaisesRegex(
            dual.DMIQDualTransportNoiseError, "singular"
        ):
            dual.polar_orthogonalize_cpu(singular)


if __name__ == "__main__":
    unittest.main()
