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

import phase_query_planner as planner_module


class PhaseQueryPureContractTests(unittest.TestCase):
    def test_architecture_is_exactly_phase_query_v2(self) -> None:
        config = planner_module.PhaseQueryPlannerConfig()
        config.validate()
        self.assertEqual(config.architecture, "phase_query_v2")
        self.assertEqual(config.latent_phases, 21)
        self.assertEqual(config.cross_attention_layers, 2)

    def test_invalid_architecture_phase_count_or_attention_depth_fails(self) -> None:
        invalid = (
            {"architecture": "global_pool_v1"},
            {"latent_phases": 20},
            {"cross_attention_layers": 1},
            {"hidden_channels": 10, "attention_heads": 4},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(planner_module.PhaseQueryPlannerError):
                    planner_module.PhaseQueryPlannerConfig(**values).validate()

    def test_student_signature_has_only_source_and_instruction_tokens(self) -> None:
        self.assertEqual(
            list(inspect.signature(planner_module.PhaseQueryPlanner.forward).parameters),
            ["self", "source", "instruction_tokens"],
        )


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class PhaseQueryTensorContractTests(unittest.TestCase):
    def _config(self):
        return planner_module.PhaseQueryPlannerConfig(
            latent_channels=4,
            text_channels=12,
            hidden_channels=8,
            attention_heads=2,
            feedforward_multiplier=2,
        )

    def _source(self):
        return torch.randn(1, 21, 3, 5, 4)

    def test_normalized_tyx_position_channels_are_explicit(self) -> None:
        source = self._source()
        coordinates = planner_module.normalized_position_channels(source)
        self.assertEqual(tuple(coordinates.shape), (1, 3, 21, 3, 5))
        self.assertEqual(float(coordinates[0, 0, 0, 0, 0]), -1.0)
        self.assertEqual(float(coordinates[0, 0, -1, 0, 0]), 1.0)
        self.assertEqual(float(coordinates[0, 1, 0, 0, 0]), -1.0)
        self.assertEqual(float(coordinates[0, 2, 0, 0, -1]), 1.0)
        planner = planner_module.PhaseQueryPlanner(self._config())
        self.assertEqual(planner.source_in.in_channels, 4 + planner_module.POSITION_CHANNELS)

    def test_all_tokens_reach_both_cross_attention_layers(self) -> None:
        planner = planner_module.PhaseQueryPlanner(self._config())
        source = self._source()
        instruction_tokens = torch.randn(1, 7, 12)
        observed = []

        def capture(_module, inputs):
            observed.append((tuple(inputs[0].shape), tuple(inputs[1].shape)))

        hooks = [
            block.cross_attention.register_forward_pre_hook(capture)
            for block in planner.cross_attention_blocks
        ]
        try:
            planner(source, instruction_tokens)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(len(observed), 2)
        self.assertEqual(observed, [((1, 21, 8), (1, 7, 8))] * 2)

    def test_initial_plan_is_zero_transport_and_preserve_biased(self) -> None:
        planner = planner_module.PhaseQueryPlanner(self._config())
        plan = planner(self._source(), torch.randn(1, 9, 12))
        self.assertEqual(tuple(plan.offsets.shape), (1, 3, 21, 3, 5))
        self.assertEqual(tuple(plan.gate_probs.shape), (1, 3, 21, 3, 5))
        self.assertTrue(torch.equal(plan.offsets, torch.zeros_like(plan.offsets)))
        self.assertGreater(float(plan.gate_probs[:, 0].min()), 0.99)
        self.assertTrue(
            torch.allclose(
                plan.gate_probs.sum(dim=1),
                torch.ones(1, 21, 3, 5),
                atol=2e-6,
                rtol=0.0,
            )
        )

    def test_phase_queries_and_explicit_time_encoding_are_distinct(self) -> None:
        planner = planner_module.PhaseQueryPlanner(self._config())
        self.assertEqual(tuple(planner.phase_queries.shape), (21, 8))
        self.assertTrue(planner.phase_queries.requires_grad)
        self.assertEqual(tuple(planner.phase_time_encoding.shape), (21, 8))
        self.assertFalse(
            torch.allclose(planner.phase_time_encoding[0], planner.phase_time_encoding[-1])
        )

    def test_padded_instruction_rows_fail_closed(self) -> None:
        planner = planner_module.PhaseQueryPlanner(self._config())
        tokens = torch.randn(1, 5, 12)
        tokens[:, -1].zero_()
        with self.assertRaisesRegex(planner_module.PhaseQueryPlannerError, "padded"):
            planner(self._source(), tokens)


if __name__ == "__main__":
    unittest.main()
