from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch  # noqa: E402
except ImportError:
    torch = None

if torch is not None:
    import fewshot_privileged_motion_code as epmc  # noqa: E402
else:  # pragma: no cover - only exercised in dependency-light environments
    epmc = None


@unittest.skipIf(torch is None, "torch is unavailable")
class MotionCodeContractTests(unittest.TestCase):
    def test_closed_shapes_bounds_and_receipt(self) -> None:
        self.assertEqual(epmc.LATENT_PHASES, 21)
        self.assertEqual(epmc.SCHEMA_VERSION, "bernini-epmc-core-v2")
        self.assertEqual(epmc.MOTION_BLOCKS, 16)
        self.assertEqual(epmc.ATTENTION_HEADS, 12)
        self.assertEqual(epmc.HEAD_DIM, 128)
        self.assertEqual(epmc.HIDDEN_SIZE, 1536)
        receipt = epmc.build_contract_receipt()
        self.assertEqual(receipt, epmc.CONTRACT_RECEIPT)
        self.assertEqual(
            receipt["code_shape"],
            {"phase_gates": [21], "block_head_gates": [16, 12]},
        )
        self.assertEqual(
            receipt["inference_arguments"],
            ["source_descriptor", "text_descriptor"],
        )
        self.assertFalse(receipt["target_available_at_inference"])
        self.assertFalse(receipt["support_available_at_inference"])
        self.assertEqual(
            receipt["gating_input_shape"], ["B", "Q", 12, 128]
        )
        self.assertEqual(receipt["query_phase_semantics"]["source"], -1)
        self.assertTrue(
            receipt["preprojection_1536_channel_chunk_gating_forbidden"]
        )
        self.assertFalse(receipt["coordinate_input_or_claim"])
        self.assertFalse(hasattr(epmc, "modulate_cpmr_content_carrier"))
        self.assertFalse(hasattr(epmc, "MotionModulationResult"))
        self.assertEqual(len(receipt["receipt_sha256"]), 64)
        self.assertEqual(receipt["receipt_sha256"], epmc.CONTRACT_RECEIPT_SHA256)
        int(receipt["receipt_sha256"], 16)
        self.assertEqual(epmc.validate_contract_receipt(receipt), receipt)

    def test_inference_signature_excludes_all_privileged_or_oracle_inputs(self) -> None:
        parameters = tuple(
            name
            for name in inspect.signature(
                epmc.AmortizedMotionCodePredictor.forward
            ).parameters
            if name != "self"
        )
        self.assertEqual(parameters, epmc.INFERENCE_ARGUMENTS)
        for forbidden in epmc.FORBIDDEN_INFERENCE_ARGUMENTS:
            self.assertNotIn(forbidden, parameters)

    def test_receipt_tampering_fails_closed_even_with_rehashed_payload(self) -> None:
        tampered = deepcopy(epmc.CONTRACT_RECEIPT)
        tampered["support_available_at_inference"] = True
        with self.assertRaisesRegex(
            epmc.PrivilegedMotionCodeContractError, "digest"
        ):
            epmc.validate_contract_receipt(tampered)

        # A caller cannot legitimize a changed scientific contract by rehashing it.
        tampered.pop("receipt_sha256")
        tampered["receipt_sha256"] = epmc._canonical_json_sha256(tampered)
        with self.assertRaisesRegex(
            epmc.PrivilegedMotionCodeContractError, "frozen"
        ):
            epmc.validate_contract_receipt(tampered)

    def test_bounded_decode_phase_zero_and_canonical_noop_are_exact(self) -> None:
        phase_logits = torch.full((2, 20), 1000.0, dtype=torch.float32)
        head_logits = torch.full((2, 16, 12), -1000.0, dtype=torch.float32)
        code = epmc.decode_bounded_motion_code(phase_logits, head_logits)
        self.assertTrue((code.phase_gates[:, 1:] <= 1.0).all().item())
        self.assertTrue((code.phase_gates[:, 1:] >= -1.0).all().item())
        self.assertTrue((code.block_head_gates <= 1.0).all().item())
        self.assertTrue((code.block_head_gates >= -1.0).all().item())
        self.assertEqual(torch.count_nonzero(code.phase_gates[:, 0]).item(), 0)
        self.assertFalse(torch.signbit(code.phase_gates[:, 0]).any().item())

        noop = epmc.canonical_noop_motion_code(2)
        noop.validate(require_noop=True)
        self.assertEqual(torch.count_nonzero(noop.flattened()).item(), 0)
        self.assertEqual(
            torch.count_nonzero(noop.flattened().view(torch.uint8)).item(), 0
        )

    def test_zero_initialized_oracle_has_gradients_for_both_gate_families(self) -> None:
        oracle = epmc.LearnableEpisodicMotionCode()
        code = oracle()
        phase_target = torch.linspace(-0.8, 0.8, 20).reshape(1, 20)
        head_target = torch.linspace(-0.5, 0.5, 16 * 12).reshape(1, 16, 12)
        teacher = epmc.MotionCode(
            torch.cat((torch.zeros(1, 1), phase_target), dim=1), head_target
        )
        loss = torch.nn.functional.smooth_l1_loss(
            code.flattened(), teacher.flattened()
        )
        loss.backward()
        self.assertGreater(oracle.phase_logits_nonzero.grad.abs().sum().item(), 0.0)
        self.assertGreater(oracle.block_head_logits.grad.abs().sum().item(), 0.0)

    def test_invalid_shapes_values_signed_zero_and_dtype_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            epmc.PrivilegedMotionCodeContractError, r"\[B,20\]"
        ):
            epmc.decode_bounded_motion_code(
                torch.zeros(1, 21), torch.zeros(1, 16, 12)
            )
        with self.assertRaisesRegex(
            epmc.PrivilegedMotionCodeContractError, "float32"
        ):
            epmc.decode_bounded_motion_code(
                torch.zeros(1, 20, dtype=torch.float16),
                torch.zeros(1, 16, 12, dtype=torch.float16),
            )

        phase = torch.zeros(1, 21)
        phase[:, 0] = 0.1
        with self.assertRaisesRegex(
            epmc.PrivilegedMotionCodeContractError, "phase_gates"
        ):
            epmc.MotionCode(phase, torch.zeros(1, 16, 12))

        signed = torch.cat(
            (torch.full((1, 1), -0.0), torch.zeros(1, 20)), dim=1
        )
        self.assertNotEqual(
            torch.count_nonzero(
                signed[:, 0].reshape(-1).repeat(1).view(torch.uint8)
            ).item(),
            0,
        )
        with self.assertRaisesRegex(
            epmc.PrivilegedMotionCodeContractError, "signed zero"
        ):
            epmc.MotionCode(signed, torch.zeros(1, 16, 12))

        unbounded = torch.zeros(1, 16, 12)
        unbounded[:, 0, 0] = 1.01
        with self.assertRaisesRegex(
            epmc.PrivilegedMotionCodeContractError, "escaped"
        ):
            epmc.MotionCode(torch.zeros(1, 21), unbounded)


@unittest.skipIf(torch is None, "torch is unavailable")
class RobustPrototypeTests(unittest.TestCase):
    @staticmethod
    def _codes(values: list[float], *, requires_grad: bool = False) -> epmc.MotionCode:
        phase = torch.zeros(len(values), 21, dtype=torch.float32)
        head = torch.zeros(len(values), 16, 12, dtype=torch.float32)
        for index, value in enumerate(values):
            phase[index, 1:] = value
            head[index] = value
        phase.requires_grad_(requires_grad)
        head.requires_grad_(requires_grad)
        return epmc.MotionCode(phase, head)

    def test_k1_is_identity_and_training_only_receipt_is_explicit(self) -> None:
        support = self._codes([0.3], requires_grad=True)
        result = epmc.build_training_support_prototype(support)
        self.assertEqual(result.rule, "single_support_identity")
        self.assertEqual(result.support_count, 1)
        self.assertTrue(torch.equal(result.code.phase_gates, support.phase_gates))
        self.assertFalse(result.code.phase_gates.requires_grad)
        receipt = result.audit_receipt()
        self.assertTrue(receipt["training_only"])
        self.assertFalse(receipt["support_available_at_inference"])
        self.assertEqual(receipt["contract_sha256"], epmc.CONTRACT_RECEIPT_SHA256)

    def test_k2_has_the_documented_exact_midpoint_degeneracy_rule(self) -> None:
        support = self._codes([-0.2, 0.6])
        result = epmc.build_training_support_prototype(support)
        self.assertEqual(result.rule, "exact_arithmetic_midpoint")
        self.assertEqual(result.iterations, 0)
        expected = support.flattened().mean(dim=0, keepdim=True)
        self.assertTrue(torch.equal(result.code.flattened(), expected))
        self.assertTrue(
            torch.allclose(
                result.code.phase_gates[:, 1:],
                torch.full((1, 20), 0.2),
            )
        )
        self.assertEqual(
            result.audit_receipt()["k_equals_two_rule"],
            "exact_arithmetic_midpoint",
        )

    def test_huber_irls_reduces_a_single_outliers_influence(self) -> None:
        support = self._codes([0.08, 0.10, 0.12, 1.0])
        result = epmc.build_training_support_prototype(support)
        plain_mean = sum([0.08, 0.10, 0.12, 1.0]) / 4.0
        robust_value = result.code.phase_gates[0, 1].item()
        self.assertEqual(result.rule, "spatial_huber_irls")
        self.assertTrue(result.converged)
        self.assertLess(abs(robust_value - 0.10), abs(plain_mean - 0.10))
        self.assertLess(robust_value, plain_mean)
        self.assertLessEqual(result.iterations, epmc.PROTOTYPE_MAX_ITERATIONS)


@unittest.skipIf(torch is None, "torch is unavailable")
class ProjectedHeadGatingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.phase_ids = torch.tensor([-1, -1, 0, 1, 3, 3, 20, 2])
        cls.heads = torch.linspace(
            -2.0,
            2.0,
            len(cls.phase_ids) * 12 * 128,
            dtype=torch.float32,
        ).reshape(1, len(cls.phase_ids), 12, 128)

    def test_real_projected_heads_receive_exact_phase_and_head_gate(self) -> None:
        phase = torch.zeros(1, 21)
        phase[:, 1] = 0.4
        phase[:, 3] = 0.6
        heads = torch.zeros(1, 16, 12)
        heads[:, 4, 0] = 0.2
        heads[:, 4, 1] = -0.2
        code = epmc.MotionCode(phase, heads)
        result = epmc.gate_projected_motion_heads(
            self.heads,
            self.phase_ids,
            code,
            block_index=4,
        )
        self.assertEqual(
            tuple(result.projected_motion_heads_fp32.shape), (1, 8, 12, 128)
        )
        # Query 4 is target phase 3.  These are genuine post-attention heads,
        # not adjacent chunks of a pre-projection 1536-vector.
        self.assertAlmostEqual(result.effective_head_gates[0, 4, 0].item(), 0.4)
        self.assertAlmostEqual(result.effective_head_gates[0, 4, 1].item(), 0.2)
        self.assertTrue(
            torch.allclose(
                result.projected_motion_heads_fp32[0, 4, 0],
                self.heads[0, 4, 0] * 0.4,
            )
        )
        self.assertTrue(
            torch.allclose(
                result.projected_motion_heads_fp32[0, 4, 1],
                self.heads[0, 4, 1] * 0.2,
            )
        )
        self.assertTrue(
            torch.equal(
                result.flattened_output(),
                result.projected_motion_heads_fp32.reshape(1, 8, 1536),
            )
        )
        receipt = result.audit_receipt()
        self.assertEqual(
            receipt["gating_point"],
            "post_attention_projected_heads_before_output_merge",
        )
        self.assertFalse(receipt["preprojection_channel_chunk_gating"])
        self.assertEqual(receipt["contract_sha256"], epmc.CONTRACT_RECEIPT_SHA256)

    def test_source_and_phase0_are_positive_zero_even_for_active_code(self) -> None:
        phase = torch.full((1, 21), 0.8)
        phase[:, 0].zero_()
        code = epmc.MotionCode(phase, torch.full((1, 16, 12), -0.4))
        result = epmc.gate_projected_motion_heads(
            self.heads, self.phase_ids, code, block_index=0
        )
        disabled = self.phase_ids <= 0
        disabled_output = result.projected_motion_heads_fp32[:, disabled]
        self.assertEqual(torch.count_nonzero(disabled_output).item(), 0)
        self.assertEqual(
            torch.count_nonzero(disabled_output.contiguous().view(torch.uint8)).item(),
            0,
        )
        self.assertEqual(
            torch.count_nonzero(result.effective_head_gates[:, disabled]).item(), 0
        )
        self.assertEqual(
            torch.count_nonzero(
                result.effective_head_gates[:, disabled]
                .contiguous()
                .view(torch.uint8)
            ).item(),
            0,
        )

    def test_canonical_noop_yields_byte_exact_positive_zero(self) -> None:
        result = epmc.gate_projected_motion_heads(
            self.heads,
            self.phase_ids,
            epmc.canonical_noop_motion_code(1),
            block_index=0,
        )
        self.assertEqual(
            torch.count_nonzero(result.projected_motion_heads_fp32).item(), 0
        )
        self.assertEqual(
            torch.count_nonzero(
                result.projected_motion_heads_fp32.contiguous().view(torch.uint8)
            ).item(),
            0,
        )
        self.assertEqual(torch.count_nonzero(result.effective_head_gates).item(), 0)

    def test_preprojection_shape_and_bad_phase_contracts_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            epmc.PrivilegedMotionCodeContractError, r"\[B,Q,12,128\]"
        ):
            epmc.gate_projected_motion_heads(
                torch.zeros(1, 8, 1536),
                self.phase_ids,
                epmc.canonical_noop_motion_code(1),
                block_index=0,
            )
        with self.assertRaisesRegex(
            epmc.PrivilegedMotionCodeContractError, "torch.int64"
        ):
            epmc.gate_projected_motion_heads(
                self.heads,
                self.phase_ids.float(),
                epmc.canonical_noop_motion_code(1),
                block_index=0,
            )
        with self.assertRaisesRegex(
            epmc.PrivilegedMotionCodeContractError, r"\[Q\]"
        ):
            epmc.gate_projected_motion_heads(
                self.heads,
                self.phase_ids[:-1],
                epmc.canonical_noop_motion_code(1),
                block_index=0,
            )
        for bad_id in (-2, 21):
            bad_phase_ids = self.phase_ids.clone()
            bad_phase_ids[0] = bad_id
            with self.subTest(bad_id=bad_id):
                with self.assertRaisesRegex(
                    epmc.PrivilegedMotionCodeContractError, "source=-1"
                ):
                    epmc.gate_projected_motion_heads(
                        self.heads,
                        bad_phase_ids,
                        epmc.canonical_noop_motion_code(1),
                        block_index=0,
                    )

    def test_bad_batch_block_and_nonfinite_fail_closed(self) -> None:
        with self.assertRaisesRegex(epmc.PrivilegedMotionCodeContractError, "batch"):
            epmc.gate_projected_motion_heads(
                self.heads.expand(2, -1, -1, -1),
                self.phase_ids,
                epmc.canonical_noop_motion_code(1),
                block_index=0,
            )
        with self.assertRaisesRegex(
            epmc.PrivilegedMotionCodeContractError, r"\[0,15\]"
        ):
            epmc.gate_projected_motion_heads(
                self.heads,
                self.phase_ids,
                epmc.canonical_noop_motion_code(1),
                block_index=16,
            )
        nonfinite = self.heads.clone()
        nonfinite[:, 2, 0, 0] = float("nan")
        with self.assertRaisesRegex(epmc.PrivilegedMotionCodeContractError, "NaN"):
            epmc.gate_projected_motion_heads(
                nonfinite,
                self.phase_ids,
                epmc.canonical_noop_motion_code(1),
                block_index=0,
            )

    def test_zero_initialization_preserves_both_gate_gradients(self) -> None:
        oracle = epmc.LearnableEpisodicMotionCode()
        result = epmc.gate_projected_motion_heads(
            self.heads, self.phase_ids, oracle(), block_index=7
        )
        self.assertEqual(
            torch.count_nonzero(result.projected_motion_heads_fp32).item(), 0
        )
        result.projected_motion_heads_fp32.mean().backward()
        self.assertGreater(oracle.phase_logits_nonzero.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            oracle.block_head_logits.grad[:, 7].abs().sum().item(), 0.0
        )
        self.assertEqual(
            oracle.block_head_logits.grad[:, :7].abs().sum().item(), 0.0
        )

    def test_source_and_phase0_queries_cannot_train_the_code(self) -> None:
        oracle = epmc.LearnableEpisodicMotionCode()
        phase_ids = torch.tensor([-1, 0, -1, 0])
        heads = torch.randn(1, 4, 12, 128)
        result = epmc.gate_projected_motion_heads(
            heads, phase_ids, oracle(), block_index=5
        )
        result.projected_motion_heads_fp32.sum().backward()
        self.assertEqual(oracle.phase_logits_nonzero.grad.abs().sum().item(), 0.0)
        self.assertEqual(oracle.block_head_logits.grad.abs().sum().item(), 0.0)


@unittest.skipIf(torch is None, "torch is unavailable")
class PredictorAndLossTests(unittest.TestCase):
    @staticmethod
    def _teacher(batch_size: int = 1) -> epmc.MotionCode:
        nonzero = torch.linspace(-0.9, 0.8, 20).repeat(batch_size, 1)
        phase = torch.cat((torch.zeros(batch_size, 1), nonzero), dim=1)
        heads = torch.full((batch_size, 16, 12), 0.05)
        return epmc.MotionCode(phase, heads)

    @staticmethod
    def _negative(code: epmc.MotionCode) -> epmc.MotionCode:
        phase = torch.cat(
            (torch.zeros_like(code.phase_gates[:, :1]), -code.phase_gates[:, 1:]),
            dim=1,
        )
        return epmc.MotionCode(phase, -code.block_head_gates)

    def test_predictor_accepts_only_source_and_text_and_outputs_bounded_code(self) -> None:
        predictor = epmc.AmortizedMotionCodePredictor(7, 5, hidden_dim=16)
        source = torch.randn(3, 21, 7)
        text = torch.randn(3, 5)
        code = predictor(source, text)
        code.validate(require_noop=True)  # zero-initialized safe behavior
        self.assertEqual(tuple(code.phase_gates.shape), (3, 21))
        self.assertEqual(tuple(code.block_head_gates.shape), (3, 16, 12))
        with self.assertRaisesRegex(
            epmc.PrivilegedMotionCodeContractError, "source_descriptor"
        ):
            predictor(torch.randn(3, 21, 6), text)
        with self.assertRaisesRegex(
            epmc.PrivilegedMotionCodeContractError, "batches"
        ):
            predictor(source, torch.randn(2, 5))
        with self.assertRaisesRegex(
            epmc.PrivilegedMotionCodeContractError, ">= 2"
        ):
            epmc.AmortizedMotionCodePredictor(1, 5)

    def test_phase_predictor_is_equivariant_to_frozen_temporal_controls(self) -> None:
        predictor = epmc.AmortizedMotionCodePredictor(2, 2, hidden_dim=1)
        with torch.no_grad():
            predictor.phase_network[0].weight.zero_()
            predictor.phase_network[0].bias.fill_(0.3)
            predictor.phase_network[0].weight[0, 0] = 1.0
            predictor.phase_network[-1].weight.fill_(1.0)
            predictor.phase_network[-1].bias.zero_()
        values = torch.linspace(-1.0, 1.0, 21)
        source = torch.stack((values, -values), dim=1).unsqueeze(0)
        text = torch.tensor([[1.0, -1.0]])
        original = predictor(source, text)
        for indices in (
            epmc.REVERSE_PHASE_INDICES,
            epmc.SHUFFLE_PHASE_INDICES,
        ):
            index = torch.tensor(indices)
            controlled = predictor(source.index_select(1, index), text)
            expected = epmc.permute_motion_code_phases(original, indices)
            self.assertTrue(
                torch.allclose(
                    controlled.phase_gates,
                    expected.phase_gates,
                    atol=1.0e-6,
                    rtol=1.0e-6,
                )
            )
            self.assertTrue(
                torch.equal(
                    controlled.block_head_gates,
                    expected.block_head_gates,
                )
            )

    def test_perfect_codes_zero_every_loss_component(self) -> None:
        teacher = self._teacher()
        reverse = epmc.permute_motion_code_phases(
            teacher, epmc.REVERSE_PHASE_INDICES
        )
        shuffle = epmc.permute_motion_code_phases(
            teacher, epmc.SHUFFLE_PHASE_INDICES
        )
        result = epmc.teacher_amortization_losses(
            teacher,
            teacher,
            epmc.canonical_noop_motion_code(1),
            self._negative(teacher),
            reverse,
            shuffle,
        )
        receipt = result.detached_receipt()
        for name, value in receipt.items():
            if name not in ("schema_version", "contract_sha256"):
                self.assertLess(abs(value), 2.0e-6, msg=name)
        self.assertEqual(receipt["contract_sha256"], epmc.CONTRACT_RECEIPT_SHA256)

    def test_invariant_wrong_and_temporal_codes_are_penalized(self) -> None:
        teacher = self._teacher()
        result = epmc.teacher_amortization_losses(
            teacher,
            teacher,
            teacher,
            teacher,
            teacher,
            teacher,
        )
        self.assertGreater(result.noop_zero.item(), 0.0)
        self.assertGreaterEqual(result.wrong_action_margin.item(), 0.19)
        self.assertGreater(result.reverse_sensitivity.item(), 0.0)
        self.assertGreater(result.shuffle_sensitivity.item(), 0.0)
        self.assertGreater(result.total.item(), 0.0)

    def test_loss_backpropagates_and_zero_action_teacher_fails_closed(self) -> None:
        predictor = epmc.AmortizedMotionCodePredictor(4, 3, hidden_dim=12)
        source = torch.randn(2, 21, 4)
        text = torch.randn(2, 3)
        predicted = predictor(source, text)
        noop_predicted = predictor(source, torch.zeros_like(text))
        wrong = predictor(source, -text)
        reverse_index = torch.tensor(epmc.REVERSE_PHASE_INDICES)
        shuffle_index = torch.tensor(epmc.SHUFFLE_PHASE_INDICES)
        reverse = predictor(source.index_select(1, reverse_index), text)
        shuffle = predictor(source.index_select(1, shuffle_index), text)
        loss = epmc.teacher_amortization_losses(
            predicted,
            self._teacher(),
            noop_predicted,
            wrong,
            reverse,
            shuffle,
        )
        loss.total.backward()
        for last_layer in (
            predictor.phase_network[-1],
            predictor.block_head_network[-1],
        ):
            self.assertIsNotNone(last_layer.weight.grad)
            self.assertGreater(last_layer.weight.grad.abs().sum().item(), 0.0)
            self.assertGreater(last_layer.bias.grad.abs().sum().item(), 0.0)

        with self.assertRaisesRegex(
            epmc.PrivilegedMotionCodeContractError, "canonical no-op"
        ):
            epmc.teacher_amortization_losses(
                predicted,
                epmc.canonical_noop_motion_code(1),
                noop_predicted,
                wrong,
                reverse,
                shuffle,
            )


if __name__ == "__main__":
    unittest.main()
