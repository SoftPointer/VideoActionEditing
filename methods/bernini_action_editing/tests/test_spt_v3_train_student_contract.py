from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
SPT_ROOT = METHOD_ROOT / "spt_v2"
for root in (METHOD_ROOT, SPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import grounded_phase_planner as grounded  # noqa: E402
import phase_transport as spt  # noqa: E402
import train_student as train  # noqa: E402


def _args(**overrides):
    parser = train.build_parser()
    values = parser.parse_args(
        [
            "--bernini-root", "/b",
            "--veomni-root", "/v",
            "--checkpoint", "/c",
            "--preprocessed-parquet-dir", "/d",
            "--dataset-summary", "/s",
            "--output", "/o",
            "--planner-architecture", grounded.ARCHITECTURE_NAME,
        ]
    )
    for name, value in overrides.items():
        setattr(values, name, value)
    return values


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class GroundedStudentLossTests(unittest.TestCase):
    def _plans(self):
        source = torch.zeros(1, 21, 1, 10, 4)
        teacher_gates = torch.zeros(1, 3, 21, 1, 10)
        teacher_gates[:, spt.GATE_PRESERVE, ..., :8] = 1.0
        teacher_gates[:, spt.GATE_TRANSPORT, ..., 8] = 1.0
        teacher_gates[:, spt.GATE_GENERATE, ..., 9] = 1.0
        teacher_offsets = torch.zeros(1, 3, 21, 1, 10)
        teacher = spt.PhasePlan(
            offsets=teacher_offsets,
            gate_probs=teacher_gates,
            provenance="oracle_pair_proxy",
            diagnostics={
                "prebudget_generate_fraction": 0.1,
                "postbudget_generate_fraction": 0.1,
                "budget_reject_fraction": 0.0,
                "max_generate_fraction_per_phase": 0.12,
                "observed_max_postbudget_generate_fraction_per_phase": 0.1,
            },
        )

        change_logits = torch.full((1, 1, 21, 1, 10), -20.0)
        change_logits[..., 8:] = 20.0
        novelty_logits = torch.full_like(change_logits, -20.0)
        novelty_logits[..., 9] = 20.0
        gates, raw_generate, scale = grounded.budgeted_factorized_gates(
            change_logits, novelty_logits
        )
        candidates = torch.tensor(grounded.candidate_lattice(), dtype=torch.float32)
        zero_index = grounded.candidate_lattice().index((0, 0, 0))
        candidate_logits = torch.full((1, 125, 21, 1, 10), -20.0)
        candidate_logits[:, zero_index] = 20.0
        diagnostics = {
            "architecture": grounded.ARCHITECTURE_NAME,
            "change_logits": change_logits,
            "novelty_logits": novelty_logits,
            "prebudget_generate_probs": raw_generate,
            "generate_budget_scale": scale,
            "offset_candidate_logits": candidate_logits,
            "offset_candidates": candidates,
            "soft_offsets": teacher_offsets.clone(),
            "coarse_change_logits": change_logits.clone(),
            "mid_change_logits": change_logits.clone(),
        }
        action = spt.PhasePlan(
            offsets=teacher_offsets.clone(),
            gate_probs=gates,
            provenance="student",
            diagnostics=diagnostics,
        )

        noop_change = torch.full_like(change_logits, -20.0)
        noop_novelty = torch.zeros_like(noop_change)
        noop_gates, noop_raw, noop_scale = grounded.budgeted_factorized_gates(
            noop_change, noop_novelty
        )
        noop = spt.PhasePlan(
            offsets=torch.zeros_like(teacher_offsets),
            gate_probs=noop_gates,
            provenance="student",
            diagnostics={
                "architecture": grounded.ARCHITECTURE_NAME,
                "change_logits": noop_change,
                "novelty_logits": noop_novelty,
                "prebudget_generate_probs": noop_raw,
                "generate_budget_scale": noop_scale,
                "offset_candidate_logits": candidate_logits.clone(),
                "offset_candidates": candidates,
                "soft_offsets": torch.zeros_like(teacher_offsets),
                "coarse_change_logits": noop_change.clone(),
                "mid_change_logits": noop_change.clone(),
            },
        )
        return source, action, teacher, noop

    def test_perfect_grounded_plan_has_aligned_losses_metrics_and_budget(self) -> None:
        source, action, teacher, noop = self._plans()
        loss, parts = train._planner_loss(
            action, teacher, noop, source, _args()
        )
        self.assertLess(float(loss), 1.0e-4)
        self.assertEqual(float(parts["change_head_iou"]), 1.0)
        self.assertEqual(float(parts["conditional_tg_f1"]), 1.0)
        self.assertEqual(float(parts["offset_candidate_top1_accuracy"]), 1.0)
        self.assertEqual(float(parts["transport_cell_offset_mae"]), 0.0)
        self.assertLessEqual(
            float(parts["student_observed_max_generate_fraction_per_phase"]),
            0.120002,
        )

    def test_novelty_supervision_is_conditional_generate_ratio(self) -> None:
        source, action, teacher, noop = self._plans()
        teacher.gate_probs[:, :, :, :, 9] = torch.tensor(
            [0.7, 0.1, 0.2]
        ).view(1, 3, 1, 1)
        logits = action.diagnostics["novelty_logits"].detach().clone()
        logits.zero_()
        logits.requires_grad_(True)
        action.diagnostics["novelty_logits"] = logits
        loss, _ = train._planner_loss(action, teacher, noop, source, _args())
        loss.backward()
        # On this soft teacher cell, absolute G is .2 while conditional
        # G/(T+G) is 2/3.  At q=.5 the correct conditional target must push q
        # upward; the incorrect absolute-G target would push it downward.
        self.assertLess(float(logits.grad[0, 0, 0, 0, 9]), 0.0)
        self.assertGreater(float(logits.grad[0, 0, 0, 0, 8]), 0.0)

    def test_prebudget_generate_mass_keeps_gradient_when_cap_is_saturated(self) -> None:
        source, action, teacher, noop = self._plans()
        change_logits = torch.full_like(action.diagnostics["change_logits"], 20.0)
        novelty_logits = torch.full_like(change_logits, 2.0, requires_grad=True)
        gates, raw, scale = grounded.budgeted_factorized_gates(
            change_logits, novelty_logits
        )
        self.assertAlmostEqual(
            float(gates[:, spt.GATE_GENERATE].mean()), 0.12, places=5
        )
        action.gate_probs = gates
        action.diagnostics["change_logits"] = change_logits
        action.diagnostics["novelty_logits"] = novelty_logits
        action.diagnostics["prebudget_generate_probs"] = raw
        action.diagnostics["generate_budget_scale"] = scale
        args = _args(
            conditional_gate_loss_weight=0.0,
            phase_change_mass_weight=0.0,
            phase_generate_mass_weight=1.0,
            mid_change_loss_weight=0.0,
            coarse_change_loss_weight=0.0,
            offset_loss_weight=0.0,
            expected_offset_loss_weight=0.0,
            noop_loss_weight=0.0,
        )
        loss, _ = train._planner_loss(action, teacher, noop, source, args)
        loss.backward()
        self.assertGreater(float(novelty_logits.grad.abs().sum()), 0.0)

    def test_noop_offset_candidate_loss_pushes_toward_exact_zero(self) -> None:
        source, action, teacher, noop = self._plans()
        candidates = noop.diagnostics["offset_candidates"]
        zero_index = grounded.candidate_lattice().index((0, 0, 0))
        wrong_index = grounded.candidate_lattice().index((1, 0, 0))
        logits = torch.zeros_like(
            noop.diagnostics["offset_candidate_logits"], requires_grad=True
        )
        logits.data[:, wrong_index] = 2.0
        noop.diagnostics["offset_candidate_logits"] = logits
        args = _args(
            gate_loss_weight=0.0,
            offset_loss_weight=0.0,
            expected_offset_loss_weight=0.0,
            noop_loss_weight=1.0,
            noop_generate_weight=0.0,
            noop_offset_weight=1.0,
        )
        loss, _ = train._planner_loss(action, teacher, noop, source, args)
        loss.backward()
        self.assertLess(float(logits.grad[:, zero_index].mean()), 0.0)
        self.assertGreater(float(logits.grad[:, wrong_index].mean()), 0.0)
        self.assertEqual(tuple(candidates.shape), (125, 3))

    def test_categorical_loss_rejects_transport_offset_outside_lattice(self) -> None:
        logits = torch.zeros(1, 125, 21, 1, 1)
        candidates = torch.tensor(grounded.candidate_lattice(), dtype=torch.float32)
        offsets = torch.zeros(1, 3, 21, 1, 1)
        offsets[:, 0] = 3.0
        gates = torch.zeros(1, 3, 21, 1, 1)
        gates[:, spt.GATE_TRANSPORT] = 1.0
        with self.assertRaisesRegex(train.StudentTrainingError, "outside"):
            train.categorical_offset_loss(logits, candidates, offsets, gates)

    def test_phase_balanced_bce_matches_exact_per_bt_normalized_formula(self) -> None:
        import torch.nn.functional as functional

        logits = torch.tensor(
            [[[[[-1.0, 0.5, 1.5, -0.25, 0.75]], [[0.2, -0.4, 1.0, -1.5, 0.0]]]]]
        ).reshape(1, 1, 2, 1, 5)
        target = torch.tensor(
            [[[[[1.0, 0.0, 0.0, 0.0, 0.0]], [[1.0, 1.0, 0.0, 0.0, 0.0]]]]]
        ).reshape_as(logits)
        result = train.phase_balanced_bce_with_logits(logits, target)
        expected_weights = torch.tensor([[4.0, 1.5]])
        weight = expected_weights[:, None, :, None, None]
        numerator = (
            weight * target * functional.softplus(-logits)
            + (1.0 - target) * functional.softplus(logits)
        ).sum(dim=(-2, -1)).squeeze(1)
        denominator = (
            weight * target + (1.0 - target)
        ).sum(dim=(-2, -1)).squeeze(1)
        expected = (numerator / denominator).mean()
        self.assertTrue(torch.equal(result["pos_weight"], expected_weights))
        self.assertTrue(torch.allclose(result["loss"], expected, atol=1e-7, rtol=0.0))

    def test_balanced_bce_clamps_rare_positive_weight_and_gradient_ratio(self) -> None:
        logits = torch.zeros(1, 1, 1, 1, 10, requires_grad=True)
        target = torch.zeros_like(logits)
        target[..., 0] = 1.0
        result = train.phase_balanced_bce_with_logits(logits, target)
        self.assertEqual(float(result["pos_weight"]), 4.0)
        result["loss"].backward()
        positive_gradient = float(logits.grad[..., 0])
        negative_gradient = float(logits.grad[..., 1])
        self.assertLess(positive_gradient, 0.0)
        self.assertGreater(negative_gradient, 0.0)
        self.assertAlmostEqual(abs(positive_gradient) / negative_gradient, 4.0, places=5)

    def test_pooled_mid_change_uses_the_same_balanced_formula(self) -> None:
        import torch.nn.functional as functional

        full = torch.zeros(1, 1, 2, 4, 4)
        full[:, :, :, :2, :2] = 1.0
        logits = torch.randn(1, 1, 2, 2, 2)
        pooled = functional.adaptive_avg_pool3d(full, (2, 2, 2))
        expected = train.phase_balanced_bce_with_logits(logits, pooled)
        actual = train._pooled_change_bce(logits, full)
        self.assertTrue(torch.equal(actual["pos_weight"], expected["pos_weight"]))
        self.assertTrue(torch.equal(actual["loss"], expected["loss"]))

    def test_counterfactual_delta_separates_change_and_restores_preserve_invariance(self) -> None:
        action = torch.tensor(
            [[[[[0.4, 0.0, 0.0, 0.0]]]]], requires_grad=True
        )
        noop = torch.zeros_like(action, requires_grad=True)
        teacher = torch.zeros_like(action)
        teacher[..., 3] = 1.0
        result = train.counterfactual_change_loss(action, noop, teacher)
        result["loss"].backward()
        # Changed cells require the action logit to exceed the no-op logit.
        self.assertLess(float(action.grad[..., 3]), 0.0)
        self.assertGreater(float(noop.grad[..., 3]), 0.0)
        # A nonzero preserve delta is pulled back toward exact invariance.
        self.assertGreater(float(action.grad[..., 0]), 0.0)
        self.assertLess(float(noop.grad[..., 0]), 0.0)
        # Equal action/no-op logits on preserve cells are already the optimum.
        self.assertEqual(float(action.grad[..., 1]), 0.0)
        self.assertEqual(float(noop.grad[..., 1]), 0.0)

    def test_counterfactual_formula_matches_balanced_margin_smooth_l1(self) -> None:
        import torch.nn.functional as functional

        action = torch.tensor([[[[[0.25, -0.5, 0.75, 0.0]]]]])
        noop = torch.tensor([[[[[-0.25, 0.25, 0.25, 0.0]]]]])
        teacher = torch.tensor([[[[[0.0, 0.0, 1.0, 0.0]]]]])
        result = train.counterfactual_change_loss(action, noop, teacher)
        delta = action - noop
        pos_weight = torch.tensor(3.0)
        effective_weight = pos_weight * teacher + (1.0 - teacher)
        numerator = (
            effective_weight
            * functional.smooth_l1_loss(delta, teacher, reduction="none", beta=1.0)
        ).sum()
        denominator = (pos_weight * teacher + (1.0 - teacher)).sum()
        self.assertEqual(float(result["pos_weight"]), 3.0)
        self.assertTrue(
            torch.allclose(result["loss"], numerator / denominator, atol=1e-7, rtol=0.0)
        )
        self.assertTrue(torch.equal(result["target_delta"], teacher))

    def test_noop_fine_mid_coarse_receive_exact_zero_bce(self) -> None:
        source, action, teacher, noop = self._plans()
        fine = torch.zeros_like(noop.diagnostics["change_logits"], requires_grad=True)
        mid = torch.zeros_like(noop.diagnostics["mid_change_logits"], requires_grad=True)
        coarse = torch.zeros_like(
            noop.diagnostics["coarse_change_logits"], requires_grad=True
        )
        noop.diagnostics["change_logits"] = fine
        noop.diagnostics["mid_change_logits"] = mid
        noop.diagnostics["coarse_change_logits"] = coarse
        args = _args(
            gate_loss_weight=0.0,
            counterfactual_change_loss_weight=0.0,
            change_polarization_loss_weight=0.0,
            offset_loss_weight=0.0,
            expected_offset_loss_weight=0.0,
            noop_loss_weight=1.0,
            noop_generate_weight=0.0,
            noop_offset_weight=0.0,
            mid_change_loss_weight=1.0,
            coarse_change_loss_weight=1.0,
        )
        loss, parts = train._planner_loss(action, teacher, noop, source, args)
        self.assertAlmostEqual(float(loss), 3.0 * torch.log(torch.tensor(2.0)).item(), places=6)
        self.assertAlmostEqual(float(parts["noop_fine_change_bce"]), torch.log(torch.tensor(2.0)).item(), places=6)
        self.assertAlmostEqual(float(parts["noop_mid_change_bce"]), torch.log(torch.tensor(2.0)).item(), places=6)
        self.assertAlmostEqual(float(parts["noop_coarse_change_bce"]), torch.log(torch.tensor(2.0)).item(), places=6)
        loss.backward()
        for gradient in (fine.grad, mid.grad, coarse.grad):
            self.assertTrue(bool((gradient > 0.0).all()))

    def test_fresh_routing_recipe_backward_contract_and_delayed_semantic_gradient(self) -> None:
        torch.manual_seed(17)
        planner = grounded.GroundedPhasePlanner(
            grounded.GroundedPhasePlannerConfig(
                latent_channels=8,
                text_channels=12,
                hidden_channels=32,
                attention_heads=4,
                match_channels=8,
                edit_slots=4,
                dense_query_chunk_size=64,
            )
        )
        source = torch.randn(1, 21, 5, 5, 8)
        action_tokens = torch.randn(1, 5, 12)
        noop_tokens = torch.randn(1, 5, 12)
        teacher_gates = torch.zeros(1, 3, 21, 5, 5)
        teacher_gates[:, spt.GATE_PRESERVE] = 1.0
        teacher_gates[:, spt.GATE_PRESERVE, :, 2, 2] = 0.0
        teacher_gates[:, spt.GATE_TRANSPORT, :, 2, 2] = 1.0
        teacher_gates[:, spt.GATE_PRESERVE, :, 1, 3] = 0.0
        teacher_gates[:, spt.GATE_GENERATE, :, 1, 3] = 1.0
        teacher = spt.PhasePlan(
            offsets=torch.zeros(1, 3, 21, 5, 5),
            gate_probs=teacher_gates,
            provenance="oracle_pair_proxy",
            diagnostics={
                "prebudget_generate_fraction": 0.04,
                "postbudget_generate_fraction": 0.04,
                "budget_reject_fraction": 0.0,
                "max_generate_fraction_per_phase": 0.12,
                "observed_max_postbudget_generate_fraction_per_phase": 0.04,
            },
        )
        args = _args(
            gate_loss_weight=1.0,
            conditional_gate_loss_weight=0.25,
            change_tversky_weight=1.0,
            phase_change_mass_weight=0.1,
            phase_generate_mass_weight=0.1,
            mid_change_loss_weight=0.25,
            coarse_change_loss_weight=0.125,
            counterfactual_change_loss_weight=1.0,
            change_polarization_loss_weight=0.0,
            offset_loss_weight=0.0,
            expected_offset_loss_weight=0.0,
            noop_loss_weight=1.0,
            noop_generate_weight=0.2,
            noop_offset_weight=0.0,
        )

        def backward_once():
            action = planner(source, action_tokens)
            noop = planner(source, noop_tokens)
            loss, _ = train._planner_loss(action, teacher, noop, source, args)
            loss.backward()
            return loss

        first_loss = backward_once()
        self.assertTrue(bool(torch.isfinite(first_loss)))
        named = dict(planner.named_parameters())
        self.assertTrue(all(parameter.grad is not None for parameter in named.values()))
        self.assertTrue(
            all(bool(torch.isfinite(parameter.grad).all()) for parameter in named.values())
        )
        offset_only_prefixes = (
            "offset_query.",
            "offset_key.",
            "offset_residual.",
            "offset_correlation_logit",
        )
        offset_only = {
            name: parameter
            for name, parameter in named.items()
            if name.startswith(offset_only_prefixes)
        }
        self.assertTrue(offset_only)
        self.assertTrue(
            all(bool((parameter.grad == 0.0).all()) for parameter in offset_only.values())
        )
        for name in (
            "change_head.weight",
            "change_head.bias",
            "mid_change_head.weight",
            "coarse_change_head.weight",
            "novelty_head.weight",
        ):
            self.assertGreater(float(named[name].grad.abs().sum()), 0.0)
        # Safe zero heads intentionally delay deep semantic gradients by one
        # update even though their forward residual routes start nonzero.
        self.assertEqual(float(named["text_in.weight"].grad.abs().sum()), 0.0)

        with torch.no_grad():
            for name, parameter in named.items():
                if name.startswith(
                    (
                        "change_head.",
                        "mid_change_head.",
                        "coarse_change_head.",
                        "novelty_head.",
                    )
                ):
                    parameter.add_(parameter.grad, alpha=-1.0e-3)
        planner.zero_grad(set_to_none=True)
        second_loss = backward_once()
        self.assertTrue(bool(torch.isfinite(second_loss)))
        self.assertGreater(float(named["text_in.weight"].grad.abs().sum()), 0.0)
        self.assertTrue(
            all(bool((parameter.grad == 0.0).all()) for parameter in offset_only.values())
        )

    def test_change_polarization_pushes_each_nonzero_logit_toward_its_pole(self) -> None:
        logits = torch.tensor([[[[[-2.0, 2.0]]]]], requires_grad=True)
        loss = train.change_polarization_loss(logits)
        expected = (4.0 * torch.sigmoid(logits) * (1.0 - torch.sigmoid(logits))).mean()
        self.assertTrue(torch.equal(loss, expected))
        loss.backward()
        self.assertGreater(float(logits.grad[..., 0]), 0.0)
        self.assertLess(float(logits.grad[..., 1]), 0.0)


class GroundedStudentReceiptTests(unittest.TestCase):
    def test_routing_only_cli_is_nonvacuous_and_offset_free(self) -> None:
        args = _args(
            offset_loss_weight=0.0,
            expected_offset_loss_weight=0.0,
            noop_offset_weight=0.0,
        )
        train.validate_cli(args)
        self.assertEqual(args.counterfactual_change_loss_weight, 0.0)
        self.assertEqual(args.change_polarization_loss_weight, 0.0)
        args.gate_loss_weight = 0.0
        args.noop_loss_weight = 1.0
        with self.assertRaisesRegex(train.StudentTrainingError, "action routing"):
            train.validate_cli(args)

    def test_student_and_counterfactual_apis_cannot_receive_paired_target(self) -> None:
        self.assertEqual(
            list(inspect.signature(train.student_plan).parameters),
            ["planner", "source", "instruction_tokens"],
        )
        self.assertEqual(
            list(inspect.signature(train.counterfactual_change_loss).parameters),
            ["action_change_logits", "noop_change_logits", "teacher_change"],
        )
        self.assertNotIn("paired_target", inspect.getsource(train.counterfactual_change_loss))

    def test_sbatch_exposes_both_optional_v3p1_routing_weights(self) -> None:
        source = (
            SPT_ROOT / "scripts" / "auh_train_student.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("BERNINI_SPT_COUNTERFACTUAL_CHANGE_LOSS_WEIGHT", source)
        self.assertIn("--counterfactual-change-loss-weight", source)
        self.assertIn("BERNINI_SPT_CHANGE_POLARIZATION_LOSS_WEIGHT", source)
        self.assertIn("--change-polarization-loss-weight", source)

    def test_immutable_contract_binds_v3p1_formula_weights_and_initialization(self) -> None:
        class Dataset:
            signature = "dataset-signature"

            def __len__(self):
                return 13

        args = _args(
            hidden_channels=32,
            attention_heads=4,
            match_channels=8,
            offset_loss_weight=0.0,
            expected_offset_loss_weight=0.0,
            noop_offset_weight=0.0,
            counterfactual_change_loss_weight=0.75,
            change_polarization_loss_weight=0.05,
        )
        planner_config = grounded.GroundedPhasePlannerConfig(
            latent_channels=8,
            text_channels=12,
            hidden_channels=32,
            attention_heads=4,
            match_channels=8,
        )
        teacher_config = spt.PhaseTransportConfig(
            latent_channels=8,
            text_channels=12,
            hidden_channels=32,
        )
        immutable = train._immutable(
            args=args,
            dataset=Dataset(),
            dataset_summary={
                "sha256": "a" * 64,
                "index_sha256": "b" * 64,
            },
            planner_config=planner_config,
            teacher_config=teacher_config,
            training_membership={
                "selection": "teacher_trust_membership",
                "training_rows": 13,
                "members": [],
            },
            world_size=4,
        )
        value = immutable["value"]
        self.assertEqual(immutable["digest"], train.legacy.object_sha256(value))
        self.assertEqual(value["counterfactual_change_loss_weight"], 0.75)
        self.assertEqual(value["change_polarization_loss_weight"], 0.05)
        self.assertEqual(value["change_class_balance"]["scope"], "per_sample_per_latent_phase")
        self.assertEqual(value["change_class_balance"]["maximum_positive_weight"], 4.0)
        self.assertEqual(value["grounded_initialization"]["temporal_residual_initial_scale"], 0.05)
        self.assertEqual(value["grounded_initialization"]["slot_self_residual_initial_scale"], 0.01)
        self.assertTrue(value["routing_only_offset_supervision_disabled"])

    def test_receipt_and_resume_are_architecture_bound(self) -> None:
        class Dataset:
            root = Path("/data")
            signature = "signature"

            def __len__(self):
                return 13

        class Planner:
            config = grounded.GroundedPhasePlannerConfig(
                latent_channels=8,
                text_channels=12,
                hidden_channels=32,
                attention_heads=4,
                match_channels=8,
            )

        class Distributed:
            world_size = 4

        receipt = train._receipt(
            args=_args(hidden_channels=32, attention_heads=4, match_channels=8),
            global_step=1,
            metrics={"total": 1.0},
            immutable={"value": {}, "digest": "digest"},
            dataset=Dataset(),
            dataset_summary={},
            training_membership={
                "selection": "teacher_trust_membership",
                "full_dataset_rows": 644,
                "training_rows": 13,
                "diagnostic_subset": True,
                "members": [],
                "membership_sha256": "membership",
            },
            planner=Planner(),
            named=[],
            initialization_digest="init",
            distributed=Distributed(),
            backend="nccl/rccl",
            resumed_from=None,
        )
        self.assertEqual(receipt["schema_version"], train.GROUNDED_RECEIPT_SCHEMA)
        self.assertEqual(receipt["method"], train.GROUNDED_METHOD_NAME)
        self.assertEqual(receipt["planner"]["architecture"], grounded.ARCHITECTURE_NAME)
        self.assertTrue(receipt["supervision"]["generate_budget_is_structural"])
        self.assertEqual(len(receipt["supervision"]["transport_candidate_lattice"]), 125)
        self.assertEqual(
            receipt["supervision"]["action_change_loss"],
            "per_bt_normalized_pos_weight_1_to_4_bce_plus_tversky_plus_per_phase_mass",
        )
        self.assertEqual(
            receipt["supervision"]["counterfactual_change_loss_weight"], 0.0
        )
        self.assertEqual(
            receipt["supervision"]["change_polarization_loss_weight"], 0.0
        )
        self.assertEqual(
            receipt["supervision"]["change_class_balance"][
                "maximum_positive_weight"
            ],
            4.0,
        )
        self.assertEqual(
            receipt["supervision"]["noop_change_supervision"],
            "exact_zero_bce_at_fine_mid_coarse_with_shared_deep_weights",
        )
        self.assertEqual(
            receipt["supervision"]["grounded_initialization"],
            {
                "semantic_cross_attention_residual_initial_scale": 0.05,
                "slot_text_residual_initial_scale": 0.05,
                "slot_source_residual_initial_scale": 0.05,
                "cell_text_residual_initial_scale": 0.05,
                "cell_slot_residual_initial_scale": 0.05,
                "temporal_residual_initial_scale": 0.05,
                "slot_self_residual_initial_scale": 0.01,
                "semantic_cross_attention_shared_coarse_and_fine": True,
                "zero_fusion_and_safe_routing_head_initialization_unchanged": True,
                "zero_output_heads_may_delay_semantic_backbone_nonzero_gradient": True,
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            (root / "planner_config.json").write_text(
                json.dumps({"architecture": grounded.ARCHITECTURE_NAME}),
                encoding="utf-8",
            )
            (root / "planner.safetensors").touch()
            (root / "optimizer.pt").touch()
            loaded, _ = train._load_resume(root, grounded.ARCHITECTURE_NAME)
            self.assertEqual(loaded["receipt_digest"], receipt["receipt_digest"])
            with self.assertRaises(train.StudentTrainingError):
                train._load_resume(root)


if __name__ == "__main__":
    unittest.main()
