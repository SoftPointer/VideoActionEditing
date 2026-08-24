#!/usr/bin/env python3
"""Contracts for the source-grounded SPT-v3 planner."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
SPT_ROOT = METHOD_ROOT / "spt_v2"
for root in (METHOD_ROOT, SPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import grounded_phase_planner as grounded  # noqa: E402
import phase_transport as spt  # noqa: E402


class GroundedPlannerPureTests(unittest.TestCase):
    def test_candidate_lattice_is_exact_oracle_lattice(self) -> None:
        candidates = grounded.candidate_lattice()
        self.assertEqual(len(candidates), 125)
        self.assertEqual(len(set(candidates)), 125)
        self.assertIn((0, 0, 0), candidates)
        self.assertEqual(
            {candidate[0] for candidate in candidates},
            set(grounded.TEMPORAL_CANDIDATES),
        )
        self.assertEqual(
            {candidate[1] for candidate in candidates},
            set(grounded.SPATIAL_CANDIDATES),
        )

    def test_config_locks_exact21_and_point12_budget(self) -> None:
        grounded.GroundedPhasePlannerConfig().validate()
        for config in (
            grounded.GroundedPhasePlannerConfig(latent_phases=20),
            grounded.GroundedPhasePlannerConfig(max_generate_fraction_per_phase=0.13),
            grounded.GroundedPhasePlannerConfig(source_bank_detach=False),
        ):
            with self.assertRaises(grounded.GroundedPlannerError):
                config.validate()

    def test_forward_api_has_no_privileged_condition(self) -> None:
        self.assertEqual(
            list(inspect.signature(grounded.GroundedPhasePlanner.forward).parameters),
            ["self", "source", "instruction_tokens"],
        )
        source = inspect.getsource(grounded.GroundedPhasePlanner.forward)
        for forbidden in ("target", "mask", "flow", "pose", "track", "trajectory"):
            self.assertNotIn(forbidden, source)
        module_source = Path(grounded.__file__).read_text(encoding="utf-8")
        self.assertEqual(module_source.count("self._dense_grounding("), 2)


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class GroundedPlannerTensorTests(unittest.TestCase):
    def _config(self, **overrides):
        values = {
            "latent_channels": 8,
            "text_channels": 12,
            "hidden_channels": 32,
            "attention_heads": 4,
            "match_channels": 8,
            "edit_slots": 4,
            "dense_query_chunk_size": 64,
        }
        values.update(overrides)
        return grounded.GroundedPhasePlannerConfig(**values)

    def test_budget_is_per_sample_and_phase_with_reject_to_preserve(self) -> None:
        change = torch.full((2, 1, 21, 2, 3), 12.0)
        novelty = torch.full_like(change, 12.0)
        gates, raw, scale = grounded.budgeted_factorized_gates(change, novelty)
        self.assertGreater(float(raw.mean()), 0.99)
        self.assertLess(float(scale.max()), 0.13)
        self.assertTrue(
            bool(
                (
                    gates[:, spt.GATE_GENERATE].mean(dim=(-2, -1))
                    <= 0.120002
                ).all()
            )
        )
        self.assertTrue(
            torch.allclose(gates.sum(dim=1), torch.ones_like(change[:, 0]), atol=2e-6)
        )
        self.assertGreater(float(gates[:, spt.GATE_PRESERVE].mean()), 0.87)

    def test_dense_plan_is_dynamic_source_grounded_and_initially_safe(self) -> None:
        torch.manual_seed(4)
        planner = grounded.GroundedPhasePlanner(self._config())
        source = torch.randn(1, 21, 5, 7, 8)
        tokens = torch.randn(1, 9, 12)
        plan = planner(source, tokens)
        self.assertEqual(tuple(plan.offsets.shape), (1, 3, 21, 5, 7))
        self.assertEqual(tuple(plan.gate_probs.shape), (1, 3, 21, 5, 7))
        self.assertEqual(plan.diagnostics["architecture"], grounded.ARCHITECTURE_NAME)
        self.assertEqual(
            tuple(plan.diagnostics["offset_candidate_logits"].shape),
            (1, 125, 21, 5, 7),
        )
        self.assertEqual(
            tuple(plan.diagnostics["offset_candidates"].shape), (125, 3)
        )
        self.assertEqual(
            tuple(plan.diagnostics["change_logits"].shape), (1, 1, 21, 5, 7)
        )
        self.assertEqual(tuple(plan.diagnostics["generate_budget_scale"].shape), (1, 1, 21, 1, 1))
        self.assertTrue(bool((plan.offsets == 0).all()))
        self.assertTrue(
            bool(
                (
                    plan.gate_probs[:, spt.GATE_GENERATE].mean(dim=(-2, -1))
                    <= 0.120002
                ).all()
            )
        )
        self.assertTrue(bool((plan.gate_probs.argmax(dim=1) == spt.GATE_PRESERVE).all()))
        self.assertTrue(
            all(parameter.ndim > 0 for parameter in planner.parameters())
        )

    def test_semantic_paths_start_open_but_execution_heads_remain_safe(self) -> None:
        planner = grounded.GroundedPhasePlanner(self._config())
        for module in (
            planner.slot_text,
            planner.slot_source,
            planner.cell_text,
            planner.cell_slots,
        ):
            self.assertTrue(
                torch.equal(
                    module.attention_scale.detach(),
                    torch.tensor([grounded.SEMANTIC_RESIDUAL_INITIAL_SCALE]),
                )
            )
            self.assertTrue(
                torch.equal(
                    module.feedforward_scale.detach(),
                    torch.tensor([grounded.SEMANTIC_RESIDUAL_INITIAL_SCALE]),
                )
            )
        self.assertTrue(
            torch.equal(
                planner.temporal.scale.detach(),
                torch.tensor([grounded.TEMPORAL_RESIDUAL_INITIAL_SCALE]),
            )
        )
        self.assertTrue(
            torch.equal(
                planner.slot_self_scale.detach(),
                torch.tensor([grounded.SLOT_SELF_INITIAL_SCALE]),
            )
        )
        for head in (
            planner.coarse_change_head,
            planner.mid_change_head,
            planner.change_head,
            planner.novelty_head,
        ):
            self.assertTrue(bool((head.weight.detach() == 0).all()))
        self.assertTrue(bool((planner.mid_fusion.delta[-1].weight.detach() == 0).all()))
        self.assertTrue(bool((planner.fine_fusion.delta[-1].weight.detach() == 0).all()))

    def test_odd_and_even_spatial_shapes_preserve_exact21(self) -> None:
        planner = grounded.GroundedPhasePlanner(self._config())
        tokens = torch.randn(1, 5, 12)
        for height, width in ((4, 6), (5, 7)):
            with self.subTest(shape=(height, width)):
                source = torch.randn(1, 21, height, width, 8)
                plan = planner(source, tokens)
                plan.validate(source)
                self.assertEqual(tuple(plan.offsets.shape[2:]), (21, height, width))


if __name__ == "__main__":
    unittest.main()
