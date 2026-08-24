#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest


try:
    import torch
    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # local macOS workspace intentionally has no torch
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

if TORCH_AVAILABLE:
    import factorial_target_temporal_plan_adapter_v1 as fact  # noqa: E402
else:
    fact = None  # type: ignore[assignment]


AUTH_SHA = "1" * 64
CHECKPOINT_SHA = "2" * 64
QUERY_SHA = "3" * 64
SOURCE_SHA = "4" * 64
ACTION_SHA = "5" * 64
MATERIALIZER_SHA = "6" * 64
SCORER_RECEIPT_SHA = "7" * 64
SCORER_SOURCE_SHA = "8" * 64


def _energies(
    *, batch: int = 1, failed_indices: tuple[int, ...] = ()
) -> fact.PureTActionEnergies:
    action = torch.full((batch,), 0.2, dtype=torch.float32)
    noop = torch.full((batch,), 1.0, dtype=torch.float32)
    reverse = torch.full((batch,), 1.2, dtype=torch.float32)
    incomplete = torch.full((batch,), 0.9, dtype=torch.float32)
    for index in failed_indices:
        reverse[index] = 0.0
    return fact.PureTActionEnergies(action, noop, reverse, incomplete)


def _source_scores(
    *, batch: int = 1, failed_indices: tuple[int, ...] = ()
) -> fact.VXIFactorialSourceScores:
    correct = torch.tensor(
        [[[0.0, 1.0], [1.0, 4.0]]], dtype=torch.float32
    ).repeat(batch, 1, 1)
    wrong = torch.tensor(
        [[[0.0, 1.0], [1.0, 1.0]]], dtype=torch.float32
    ).repeat(batch, 1, 1)
    for index in failed_indices:
        wrong[index, 1, 1] = 100.0
    return fact.VXIFactorialSourceScores(correct, wrong)


def _evidence() -> fact.FactPlanGateEvidence:
    return fact.make_gate_evidence(
        scorer_receipt_sha256=SCORER_RECEIPT_SHA,
        scorer_source_digest=SCORER_SOURCE_SHA,
        checkpoint_digest=CHECKPOINT_SHA,
        query_digest=QUERY_SHA,
        correct_source_digest=SOURCE_SHA,
        pure_t_action_evidence_digest=ACTION_SHA,
    )


def _decision(
    *, batch: int = 1, energy_fail: tuple[int, ...] = (), source_fail: tuple[int, ...] = ()
) -> fact.FactPlanHardGateDecision:
    return fact.evaluate_fact_plan_hard_gate(
        _energies(batch=batch, failed_indices=energy_fail),
        _source_scores(batch=batch, failed_indices=source_fail),
        evidence=_evidence(),
    )


def _hidden(
    *, batch: int = 1, patches: int = 2, dtype: torch.dtype = None
) -> torch.Tensor:
    if dtype is None:
        dtype = torch.float32
    generator = torch.Generator(device="cpu").manual_seed(17)
    return torch.randn(
        batch,
        fact.LATENT_PHASES,
        patches,
        fact.HIDDEN_SIZE,
        generator=generator,
        dtype=torch.float32,
    ).to(dtype=dtype)


def _plan(*, batch: int = 1, patches: int = 2) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(29)
    return torch.randn(
        batch,
        fact.PLAN_SEGMENTS,
        patches,
        fact.PLAN_WIDTH,
        generator=generator,
        dtype=torch.float32,
    ).contiguous()


def _packs(
    hidden: torch.Tensor, plan: torch.Tensor
) -> tuple[fact.AuthenticatedTargetPack, fact.AuthenticatedActionPlanPack]:
    target_provenance = fact.make_target_pack_provenance(
        hidden,
        authentication_receipt_sha256=AUTH_SHA,
        checkpoint_digest=CHECKPOINT_SHA,
        query_digest=QUERY_SHA,
        correct_source_digest=SOURCE_SHA,
    )
    target_pack = fact.authenticate_target_pack(hidden, target_provenance)
    plan_provenance = fact.make_action_plan_pack_provenance(
        plan,
        materializer_receipt_sha256=MATERIALIZER_SHA,
        checkpoint_digest=CHECKPOINT_SHA,
        query_digest=QUERY_SHA,
        pure_t_action_evidence_digest=ACTION_SHA,
        target_pack_provenance_digest=target_provenance.digest,
    )
    plan_pack = fact.authenticate_action_plan_pack(plan, plan_provenance)
    return target_pack, plan_pack


def _activate_first_axis(adapter: fact.FactorialTargetTemporalPlanAdapter) -> None:
    with torch.no_grad():
        adapter.U.zero_()
        adapter.U[0, 0] = 1.0


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable in this environment")
class FactorialTargetTemporalPlanAdapterTests(unittest.TestCase):
    def test_authenticated_target_and_action_plan_are_required_and_rehashed(self) -> None:
        adapter = fact.FactorialTargetTemporalPlanAdapter()
        hidden = _hidden()
        plan = _plan()
        target_pack, plan_pack = _packs(hidden, plan)
        result = adapter(
            target_pack, plan_pack, sigma=0.8, decision=_decision()
        )
        self.assertEqual(tuple(result.shape), (1, 21, 2, 1536))
        self.assertEqual(
            target_pack.provenance.slice_semantics, fact.TARGET_SLICE_SEMANTICS
        )
        self.assertEqual(
            plan_pack.provenance.tensor_semantics, fact.PLAN_TENSOR_SEMANTICS
        )

        with self.assertRaises(fact.FactPlanError):
            adapter(hidden, plan_pack, sigma=0.8, decision=_decision())
        with self.assertRaises(fact.FactPlanError):
            adapter(target_pack, plan, sigma=0.8, decision=_decision())

        with torch.no_grad():
            hidden[0, 0, 0, 0].add_(1.0)
        with self.assertRaises(fact.FactPlanError):
            adapter(target_pack, plan_pack, sigma=0.8, decision=_decision())

        fresh_hidden = _hidden()
        fresh_plan = _plan()
        fresh_target, fresh_plan_pack = _packs(fresh_hidden, fresh_plan)
        with torch.no_grad():
            fresh_plan[0, 0, 0, 0].add_(1.0)
        with self.assertRaises(fact.FactPlanError):
            adapter(fresh_target, fresh_plan_pack, sigma=0.8, decision=_decision())

    def test_provenance_semantics_and_cross_binding_fail_closed(self) -> None:
        hidden = _hidden()
        provenance = fact.make_target_pack_provenance(
            hidden,
            authentication_receipt_sha256=AUTH_SHA,
            checkpoint_digest=CHECKPOINT_SHA,
            query_digest=QUERY_SHA,
            correct_source_digest=SOURCE_SHA,
        )
        with self.assertRaises(fact.FactPlanError):
            fact.authenticate_target_pack(
                hidden, replace(provenance, target_suffix_only=False)
            )

        target_pack, plan_pack = _packs(hidden, _plan())
        bad_plan_provenance = replace(
            plan_pack.provenance, action_only=False
        )
        with self.assertRaises(fact.FactPlanError):
            fact.authenticate_action_plan_pack(plan_pack.tensor, bad_plan_provenance)

        other_query = "9" * 64
        mismatched = fact.make_action_plan_pack_provenance(
            plan_pack.tensor,
            materializer_receipt_sha256=MATERIALIZER_SHA,
            checkpoint_digest=CHECKPOINT_SHA,
            query_digest=other_query,
            pure_t_action_evidence_digest=ACTION_SHA,
            target_pack_provenance_digest=target_pack.provenance.digest,
        )
        mismatched_pack = fact.authenticate_action_plan_pack(
            plan_pack.tensor, mismatched
        )
        with self.assertRaises(fact.FactPlanError):
            fact.FactorialTargetTemporalPlanAdapter()(
                target_pack, mismatched_pack, sigma=0.8, decision=_decision()
            )

    def test_zero_init_identity_and_receipt_checks_live_u_scope(self) -> None:
        adapter = fact.FactorialTargetTemporalPlanAdapter()
        hidden = _hidden()
        target_pack, plan_pack = _packs(hidden, _plan())
        result = adapter(target_pack, plan_pack, sigma=0.8, decision=_decision())
        self.assertTrue(torch.equal(result, hidden))
        receipt = adapter.receipt()
        self.assertTrue(receipt["u_requires_grad"])
        self.assertTrue(receipt["only_u_trainable"])
        self.assertTrue(receipt["u_zero_at_receipt_time"])
        self.assertNotIn("optimizer_constructed", receipt)
        self.assertNotIn("bernini_model_forward_performed", receipt)
        adapter.U.requires_grad_(False)
        self.assertFalse(adapter.receipt()["only_u_trainable"])
        with self.assertRaises(fact.FactPlanError):
            adapter(target_pack, plan_pack, sigma=0.8, decision=_decision())

    def test_each_of_four_segments_has_independent_support_and_zero_dc(self) -> None:
        adapter = fact.FactorialTargetTemporalPlanAdapter()
        _activate_first_axis(adapter)
        hidden = torch.zeros(1, 21, 1, 1536, dtype=torch.float32)
        for segment, (begin, stop) in enumerate(fact.SEGMENT_RANGES):
            with self.subTest(segment=segment):
                plan = torch.zeros(1, 4, 1, 32, dtype=torch.float32)
                plan[:, segment, :, 0] = float(segment + 1)
                target_pack, plan_pack = _packs(hidden, plan.contiguous())
                delta = adapter.adapter_delta(
                    target_pack, plan_pack, sigma=0.8, decision=_decision()
                )
                outside = torch.cat((delta[:, :begin], delta[:, stop:]), dim=1)
                self.assertEqual(int(torch.count_nonzero(outside).item()), 0)
                self.assertGreater(
                    int(torch.count_nonzero(delta[:, begin:stop]).item()), 0
                )
                self.assertTrue(
                    torch.allclose(
                        delta.sum(dim=1),
                        torch.zeros_like(delta[:, 0]),
                        rtol=0.0,
                        atol=2.0e-6,
                    )
                )

    def test_per_segment_patch_vector_norm_is_capped(self) -> None:
        cap = 0.75
        adapter = fact.FactorialTargetTemporalPlanAdapter(
            max_segment_vector_norm=cap
        )
        with torch.no_grad():
            adapter.U.fill_(100.0)
        hidden = torch.zeros(1, 21, 2, 1536, dtype=torch.float32)
        plan = torch.full((1, 4, 2, 32), 100.0, dtype=torch.float32)
        target_pack, plan_pack = _packs(hidden, plan.contiguous())
        delta = adapter.adapter_delta(
            target_pack, plan_pack, sigma=0.8, decision=_decision()
        )
        for begin, stop in fact.SEGMENT_RANGES:
            segment_norm = torch.linalg.vector_norm(
                delta[:, begin:stop].permute(0, 2, 1, 3).reshape(1, 2, -1),
                dim=-1,
            )
            self.assertTrue(torch.all(segment_norm <= cap + 2.0e-5))
            self.assertTrue(torch.all(segment_norm > 0.0))

    def test_gate_detaches_vxi_graph_owns_clone_and_rejects_mutation(self) -> None:
        correct_leaf = torch.tensor(
            [[[0.0, 1.0], [1.0, 4.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        wrong_leaf = torch.tensor(
            [[[0.0, 1.0], [1.0, 1.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        correct_graph = correct_leaf * 1.0
        wrong_graph = wrong_leaf * 1.0
        decision = fact.evaluate_fact_plan_hard_gate(
            _energies(),
            fact.VXIFactorialSourceScores(correct_graph, wrong_graph),
            evidence=_evidence(),
        )
        before = decision.source_margins
        self.assertEqual(before.dtype, torch.float64)
        self.assertEqual(before.device.type, "cpu")
        self.assertFalse(before.requires_grad)
        self.assertIsNone(before.grad_fn)
        with torch.no_grad():
            correct_graph.add_(100.0)
            wrong_graph.sub_(100.0)
        decision.validate()
        self.assertTrue(torch.equal(decision.source_margins, before))
        self.assertNotEqual(
            decision.source_margins.untyped_storage().data_ptr(),
            before.untyped_storage().data_ptr(),
        )

    def test_hard_axes_do_not_compensate_and_wrong_source_swap_reverses_sign(self) -> None:
        bad_energy = fact.PureTActionEnergies(
            action=torch.tensor([0.2]),
            noop=torch.tensor([1000.0]),
            reverse=torch.tensor([0.0]),
            incomplete=torch.tensor([1000.0]),
        )
        decision = fact.evaluate_fact_plan_hard_gate(
            bad_energy, _source_scores(), evidence=_evidence()
        )
        self.assertFalse(bool(decision.pure_t_pass.item()))
        self.assertTrue(bool(decision.source_pass.item()))
        self.assertFalse(decision.all_examples_pass)

        good = _source_scores()
        swapped = fact.VXIFactorialSourceScores(
            good.wrong_source, good.correct_source
        )
        swapped_decision = fact.evaluate_fact_plan_hard_gate(
            _energies(), swapped, evidence=_evidence()
        )
        self.assertTrue(torch.all(swapped_decision.source_margins < 0.0))
        self.assertFalse(swapped_decision.all_examples_pass)

    def test_evidence_off_and_low_sigma_both_return_same_input_object(self) -> None:
        adapter = fact.FactorialTargetTemporalPlanAdapter()
        with torch.no_grad():
            adapter.U.fill_(0.25)
        hidden = _hidden()
        target_pack, plan_pack = _packs(hidden, _plan())
        evidence_off = _decision(energy_fail=(0,))
        off_result = adapter(
            target_pack, plan_pack, sigma=0.8, decision=evidence_off
        )
        self.assertIs(off_result, hidden)
        self.assertEqual(off_result.data_ptr(), hidden.data_ptr())
        low_result = adapter(
            target_pack, plan_pack, sigma=0.1, decision=_decision()
        )
        self.assertIs(low_result, hidden)
        self.assertEqual(low_result.data_ptr(), hidden.data_ptr())

    def test_mixed_batch_keeps_failed_sample_exact_and_updates_passed_sample(self) -> None:
        adapter = fact.FactorialTargetTemporalPlanAdapter()
        _activate_first_axis(adapter)
        hidden = _hidden(batch=2, patches=1)
        plan = torch.zeros(2, 4, 1, 32, dtype=torch.float32)
        plan[:, :, :, 0] = 1.0
        target_pack, plan_pack = _packs(hidden, plan.contiguous())
        decision = _decision(batch=2, source_fail=(1,))
        self.assertEqual(decision.per_example_pass.tolist(), [True, False])
        result = adapter(target_pack, plan_pack, sigma=0.8, decision=decision)
        self.assertFalse(torch.equal(result[0], hidden[0]))
        self.assertTrue(torch.equal(result[1], hidden[1]))

    def test_sigma_boundaries_and_mid_amplitude(self) -> None:
        self.assertEqual(fact.sigma_gate(0.0), ("low_exact_base", 0.0))
        self.assertEqual(fact.sigma_gate(0.249999), ("low_exact_base", 0.0))
        self.assertEqual(fact.sigma_gate(0.25), ("mid", 0.5))
        self.assertEqual(fact.sigma_gate(0.549999), ("mid", 0.5))
        self.assertEqual(fact.sigma_gate(0.55), ("high", 1.0))
        self.assertEqual(fact.sigma_gate(1.0), ("high", 1.0))

        adapter = fact.FactorialTargetTemporalPlanAdapter()
        _activate_first_axis(adapter)
        hidden = torch.zeros(1, 21, 1, 1536, dtype=torch.float32)
        plan = torch.zeros(1, 4, 1, 32, dtype=torch.float32)
        plan[:, :, :, 0] = 1.0
        target_pack, plan_pack = _packs(hidden, plan.contiguous())
        high = adapter.adapter_delta(
            target_pack, plan_pack, sigma=0.55, decision=_decision()
        )
        mid = adapter.adapter_delta(
            target_pack, plan_pack, sigma=0.25, decision=_decision()
        )
        self.assertTrue(torch.equal(mid * 2.0, high))

    def test_fp16_and_bf16_residuals_keep_dtype_and_zero_dc(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                adapter = fact.FactorialTargetTemporalPlanAdapter()
                _activate_first_axis(adapter)
                hidden = torch.zeros(1, 21, 1, 1536, dtype=dtype)
                plan = torch.zeros(1, 4, 1, 32, dtype=torch.float32)
                plan[:, :, :, 0] = 0.75
                target_pack, plan_pack = _packs(hidden, plan.contiguous())
                delta = adapter.adapter_delta(
                    target_pack, plan_pack, sigma=0.8, decision=_decision()
                )
                output = adapter(
                    target_pack, plan_pack, sigma=0.8, decision=_decision()
                )
                self.assertEqual(delta.dtype, dtype)
                self.assertEqual(output.dtype, dtype)
                self.assertTrue(
                    torch.allclose(
                        delta.float().sum(dim=1),
                        torch.zeros_like(delta[:, 0].float()),
                        rtol=0.0,
                        atol=2.0e-3,
                    )
                )

    def test_public_or_tampered_all_true_gate_fails_closed(self) -> None:
        with self.assertRaises(fact.FactPlanError):
            fact.FactPlanHardGateDecision(
                per_example_pass=torch.ones(1, dtype=torch.bool)
            )
        forged = object.__new__(fact.FactPlanHardGateDecision)
        with self.assertRaises(fact.FactPlanError):
            forged.validate()

        decision = _decision(energy_fail=(0,))
        object.__setattr__(
            decision, "_per_example_pass", torch.ones(1, dtype=torch.bool)
        )
        with self.assertRaises(fact.FactPlanError):
            decision.validate()
        target_pack, plan_pack = _packs(_hidden(), _plan())
        with self.assertRaises(fact.FactPlanError):
            fact.FactorialTargetTemporalPlanAdapter()(
                target_pack, plan_pack, sigma=0.8, decision=decision
            )

    def test_invalid_shapes_nans_grad_plan_and_sigma_fail_closed(self) -> None:
        with self.assertRaises(fact.FactPlanError):
            fact.make_target_pack_provenance(
                torch.zeros(1, 22, 1, 1536),
                authentication_receipt_sha256=AUTH_SHA,
                checkpoint_digest=CHECKPOINT_SHA,
                query_digest=QUERY_SHA,
                correct_source_digest=SOURCE_SHA,
            ).validate()
        hidden = _hidden()
        target_provenance = fact.make_target_pack_provenance(
            hidden,
            authentication_receipt_sha256=AUTH_SHA,
            checkpoint_digest=CHECKPOINT_SHA,
            query_digest=QUERY_SHA,
            correct_source_digest=SOURCE_SHA,
        )
        plan = _plan().requires_grad_(True)
        provenance = fact.make_action_plan_pack_provenance(
            plan.detach(),
            materializer_receipt_sha256=MATERIALIZER_SHA,
            checkpoint_digest=CHECKPOINT_SHA,
            query_digest=QUERY_SHA,
            pure_t_action_evidence_digest=ACTION_SHA,
            target_pack_provenance_digest=target_provenance.digest,
        )
        with self.assertRaises(fact.FactPlanError):
            fact.authenticate_action_plan_pack(plan, provenance)
        wrong_plan = torch.zeros(1, 5, 2, 32, dtype=torch.float32)
        with self.assertRaises(fact.FactPlanError):
            fact.make_action_plan_pack_provenance(
                wrong_plan,
                materializer_receipt_sha256=MATERIALIZER_SHA,
                checkpoint_digest=CHECKPOINT_SHA,
                query_digest=QUERY_SHA,
                pure_t_action_evidence_digest=ACTION_SHA,
                target_pack_provenance_digest=target_provenance.digest,
            )
        with self.assertRaises(fact.FactPlanError):
            fact.evaluate_fact_plan_hard_gate(
                fact.PureTActionEnergies(
                    torch.zeros(1), torch.zeros(2), torch.zeros(1), torch.zeros(1)
                ),
                _source_scores(),
                evidence=_evidence(),
            )
        bad = _source_scores().correct_source.clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaises(fact.FactPlanError):
            fact.evaluate_fact_plan_hard_gate(
                _energies(),
                fact.VXIFactorialSourceScores(
                    bad, _source_scores().wrong_source
                ),
                evidence=_evidence(),
            )
        for invalid in (True, -0.1, 1.01, float("nan")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(fact.FactPlanError):
                    fact.sigma_gate(invalid)


if __name__ == "__main__":
    unittest.main()
