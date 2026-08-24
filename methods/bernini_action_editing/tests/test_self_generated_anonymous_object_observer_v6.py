from pathlib import Path
import sys
import unittest

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import anonymous_visual_projection_hook_v6 as hook
import self_generated_anonymous_object_observer_v6 as observer
import self_generated_anonymous_object_registry_v6 as registry


class AnonymousObjectObserverV6Test(unittest.TestCase):
    def _evaluated(self, phase, x, y, descriptor=(1.0, 0.0), local_id=0):
        return observer.EvaluatedComponentV6(
            proposal_phase=max(0, phase - 1),
            evaluation_phase=phase,
            local_id=local_id,
            mass=0.8,
            centroid=(x, y),
            descriptor=descriptor,
            neutral_visual_cosine_margin=0.5,
            top_vs_median_margin=0.5,
            top10_mass_fraction=0.8,
        )

    def test_soft_component_variable_cardinality_and_no_forced_component(self):
        delta = torch.zeros((25, 4), dtype=torch.float32)
        delta[6:8] = 10.0
        delta[18:20] = 8.0
        neutral = torch.randn((25, 4), generator=torch.Generator().manual_seed(4))
        rows = observer.discover_soft_components_v6(
            delta, neutral, phase=0, height=5, width=5
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.soft_mass >= 1.5 for row in rows))
        empty = observer.discover_soft_components_v6(
            torch.zeros_like(delta), neutral, phase=0, height=5, width=5
        )
        self.assertEqual(empty, ())

    def test_constant_neutral_tokens_cannot_be_rescued_by_geometry(self):
        component = observer.ProposalComponentV6(
            0, 0, (6, 7), (1.0, 1.0), 2.0, (-0.5, -0.5), (1.0, 0.0)
        )
        neutral = torch.tensor([[1.0, 0.0]] * 25, dtype=torch.float32)
        result = observer.evaluate_component_with_neutral_tokens_v6(
            component, neutral, evaluation_phase=1, height=5, width=5
        )
        self.assertIsNone(result)

    def test_informative_neutral_visual_descriptor_can_correspond(self):
        component = observer.ProposalComponentV6(
            0, 0, (6, 7), (1.0, 1.0), 2.0, (-0.5, -0.5), (1.0, 0.0)
        )
        neutral = torch.tensor([[0.0, 1.0]] * 25, dtype=torch.float32)
        neutral[6:8] = torch.tensor([1.0, 0.0])
        result = observer.evaluate_component_with_neutral_tokens_v6(
            component, neutral, evaluation_phase=1, height=5, width=5
        )
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.neutral_visual_cosine_margin, 0.03)

    def test_unbalanced_ot_has_explicit_dustbin(self):
        previous = [self._evaluated(1, -0.5, 0.0)]
        current = [
            self._evaluated(3, -0.45, 0.0),
            self._evaluated(3, 0.8, 0.8, descriptor=(0.0, 1.0), local_id=1),
        ]
        plan, cost = observer.unbalanced_ot_with_dustbin_v6(previous, current)
        self.assertEqual(tuple(plan.shape), (2, 3))
        self.assertEqual(tuple(cost.shape), (2, 3))
        self.assertGreater(plan[-1, 1].item(), 0.0)

    def test_tracking_birth_occlusion_reentry_death(self):
        rows = {
            1: (self._evaluated(1, -0.5, 0.0),),
            3: (),
            5: (self._evaluated(5, -0.45, 0.0),),
        }
        tracks, events = observer.track_hypotheses_v6(rows)
        kinds = [row["event"] for row in events]
        self.assertIn("birth", kinds)
        self.assertIn("occlusion", kinds)
        self.assertIn("reentry", kinds)
        self.assertIn("death", kinds)
        self.assertEqual(len(tracks), 1)

    def test_dynamic_edge_lifecycle(self):
        left = observer.TrackStateV6(
            0,
            [
                self._evaluated(1, 0.0, 0.0),
                self._evaluated(3, 0.0, 0.0),
                self._evaluated(5, 0.0, 0.0),
            ],
        )
        right = observer.TrackStateV6(
            1,
            [
                self._evaluated(1, 0.05, 0.0, local_id=1),
                self._evaluated(3, 1.0, 1.0, local_id=1),
                self._evaluated(5, 1.0, 1.0, local_id=1),
            ],
        )
        rows = observer.dynamic_edge_lifecycle_v6((left, right))
        kinds = [row["event"] for row in rows]
        self.assertIn("activate", kinds)
        self.assertIn("deactivate", kinds)
        ephemeral = (
            observer.TrackStateV6(2, [self._evaluated(1, 0.0, 0.0)]),
            observer.TrackStateV6(
                3, [self._evaluated(1, 0.01, 0.0, local_id=1)]
            ),
        )
        self.assertEqual(observer.qualified_tracks_v6(ephemeral), ())
        self.assertEqual(observer.dynamic_edge_lifecycle_v6(ephemeral), ())
        gate_kwargs = self._passing_gate_kwargs()
        gate_kwargs.update(
            primary_track_count=len(observer.qualified_tracks_v6(ephemeral)),
            primary_coverage=0.0,
            primary_lifecycle_count=len(
                observer.dynamic_edge_lifecycle_v6(ephemeral)
            ),
        )
        self.assertFalse(
            observer.branch_gate_decision_v6(**gate_kwargs)["primary_graph_valid"]
        )

    def test_phase_shuffle_absolute_floor_semantics(self):
        self.assertEqual(observer.phase_shuffle_gate_v6(0.0, 0.0), (False, None))
        self.assertEqual(observer.phase_shuffle_gate_v6(0.0, 0.03), (True, None))
        passed, ratio = observer.phase_shuffle_gate_v6(0.02, 0.03)
        self.assertTrue(passed)
        self.assertAlmostEqual(ratio, 1.5)

    def _passing_gate_kwargs(self):
        return {
            "primary_track_count": 2,
            "primary_coverage": 0.8,
            "primary_lifecycle_count": 1,
            "component_counts": {"noop": 0},
            "static_ratio": 0.0,
            "reverse_cosine": -0.5,
            "phase_shuffle_pass": True,
            "paraphrase_iou": 0.8,
            "paraphrase_cosine": 0.8,
            "lexical_ratio": 0.1,
            "source_swap_iou": 0.1,
            "source_swap_coverage": 0.0,
            "source_swap_lifecycle_count": 0,
        }

    def test_zero_static_track_definition_passes(self):
        gates = observer.branch_gate_decision_v6(**self._passing_gate_kwargs())
        self.assertTrue(gates["static_pass"])
        self.assertTrue(all(gates.values()))

    def test_source_swap_stable_neutral_graph_cannot_pass_on_low_iou_alone(self):
        kwargs = self._passing_gate_kwargs()
        kwargs["source_swap_coverage"] = 0.9
        kwargs["source_swap_lifecycle_count"] = 2
        gates = observer.branch_gate_decision_v6(**kwargs)
        self.assertFalse(gates["source_swap_pass"])

    def test_graph_abstention_cannot_be_compensated(self):
        kwargs = self._passing_gate_kwargs()
        kwargs["primary_lifecycle_count"] = 0
        gates = observer.branch_gate_decision_v6(**kwargs)
        self.assertFalse(gates["primary_graph_valid"])
        self.assertFalse(all(gates.values()))

    def test_crossfit_phase_pairs_have_no_overwrite_or_unused_claim(self):
        self.assertEqual(
            observer.CROSS_FIT_PHASE_PAIRS["A_to_B"],
            tuple((phase, phase + 1) for phase in range(0, 20, 2)),
        )
        self.assertEqual(
            observer.CROSS_FIT_PHASE_PAIRS["B_to_A"],
            tuple((phase, phase + 1) for phase in range(1, 20, 2)),
        )
        for rows in observer.CROSS_FIT_PHASE_PAIRS.values():
            self.assertEqual(len({left for left, _ in rows}), 10)
            self.assertEqual(len({right for _, right in rows}), 10)

    def _capture(self, arm="action", block=6, timestep="b", rotary="c"):
        identity = hook.AnonymousCaptureIdentityV6(
            "appearance_0",
            arm,
            "high",
            18,
            "a" * 64,
            timestep * 64,
            rotary * 64,
            37,
            25,
        )
        return hook.ProjectedVisualCaptureV6(
            identity,
            block,
            "d" * 64,
            torch.ones((1, 21, 925, 16), dtype=torch.float32),
            torch.ones((1, 21, 925, 16), dtype=torch.float32),
        )

    def test_capture_ownership_transfer_is_single_take(self):
        capture = self._capture()
        arm = observer.AnonymousProjectedArmV6.from_capture(capture)
        self.assertTrue(capture.consumed)
        with self.assertRaises(hook.AnonymousVisualProjectionHookV6Error):
            observer.AnonymousProjectedArmV6.from_capture(capture)
        arm.validate()
        arm.zeroize()
        self.assertEqual(torch.count_nonzero(capture.query_sketch).item(), 0)

    def test_nontext_timestep_mismatch_fails_and_scrubs_whole_cell(self):
        rows = {}
        raw = []
        for arm in registry.ARMS:
            rows[arm] = {}
            for block in registry.BLOCKS:
                capture = self._capture(
                    arm=arm,
                    block=block,
                    timestep="e" if (arm == "static" and block == 18) else "b",
                )
                item = observer.AnonymousProjectedArmV6.from_capture(capture)
                rows[arm][block] = item
                raw.extend((item.query_sketch, item.hidden_sketch))
        with self.assertRaisesRegex(observer.AnonymousObjectObserverV6Error, "timestep_sha256"):
            observer.reduce_anonymous_cell_v6(rows)
        self.assertTrue(all(torch.count_nonzero(value).item() == 0 for value in raw))

    def _reduced_cell(self, appearance, sigma, admitted):
        branch = {
            "control_executed": {name: True for name in observer.CONTROL_ARMS},
            "branch_pass": admitted,
        }
        return observer.ReducedAnonymousCellV6(
            appearance,
            sigma,
            "a" * 64,
            "b" * 64,
            "c" * 64,
            {name: dict(branch) for name in observer.BRANCHES},
            admitted,
            True,
        )

    def test_overall_requires_nine_of_nine(self):
        stream = observer.AnonymousObjectObserverV6()
        index = 0
        for appearance in registry.APPEARANCE_IDS:
            for sigma in registry.SIGMA_CELL_INDICES:
                stream.add(self._reduced_cell(appearance, sigma, admitted=index != 8))
                index += 1
        receipt = stream.finalize()
        self.assertEqual(receipt["branchwise_diagnostic_admitted_cell_count"], 8)
        self.assertFalse(receipt["diagnostic_component_admitted"])
        self.assertEqual(receipt["diagnostic_component_status"], "REJECTED")

    def test_nine_of_nine_can_only_admit_diagnostic_not_representation(self):
        stream = observer.AnonymousObjectObserverV6()
        for appearance in registry.APPEARANCE_IDS:
            for sigma in registry.SIGMA_CELL_INDICES:
                stream.add(self._reduced_cell(appearance, sigma, admitted=True))
        receipt = stream.finalize()
        self.assertTrue(receipt["diagnostic_component_admitted"])
        self.assertFalse(receipt["representation_admitted"])
        self.assertFalse(receipt["stable_transferable_action_representation_claimed"])


if __name__ == "__main__":
    unittest.main()
