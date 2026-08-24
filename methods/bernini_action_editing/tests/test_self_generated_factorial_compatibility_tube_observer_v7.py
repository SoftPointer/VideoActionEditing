from pathlib import Path
import sys
import unittest

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import self_generated_anonymous_object_observer_v6 as v6_observer
import self_generated_factorial_compatibility_registry_v7 as registry
import self_generated_factorial_compatibility_tube_observer_v7 as observer


class FactorialCompatibilityTubeObserverV7Test(unittest.TestCase):
    def _additive_factorial(self, width=2):
        shape = (registry.PHASES, registry.PATCHES, width)
        neutral = {}
        factorial = {}
        for state_index, state_id in enumerate(registry.APPEARANCE_IDS):
            state = torch.full(shape, float(state_index + 1), dtype=torch.float32)
            neutral[state_id] = state
            factorial[state_id] = {}
            for caption_index, caption_id in enumerate(registry.APPEARANCE_IDS):
                caption = torch.full(
                    shape, float((caption_index + 1) * 10), dtype=torch.float32
                )
                factorial[state_id][caption_id] = state + caption
        return factorial, neutral

    def test_interaction_residual_exactly_removes_additive_appearance_and_caption(self):
        factorial, neutral = self._additive_factorial()
        for state_id in registry.APPEARANCE_IDS:
            for branch in observer.BRANCHES:
                result = observer.factorial_interaction_residual_v7(
                    factorial, neutral, state_id=state_id, branch=branch
                )
                self.assertEqual(torch.count_nonzero(result).item(), 0)

    def test_interaction_retains_matched_compatibility_and_uses_disjoint_baselines(self):
        factorial, neutral = self._additive_factorial(width=1)
        state = "appearance_0"
        factorial[state][state] = factorial[state][state] + 10.0
        # A->B nuisance for caption_0 is state_2; B->A nuisance is state_1.
        factorial["appearance_2"][state] = factorial["appearance_2"][state] + 3.0
        factorial["appearance_1"][state] = factorial["appearance_1"][state] + 7.0
        a = observer.factorial_interaction_residual_v7(
            factorial, neutral, state_id=state, branch="A_to_B"
        )
        b = observer.factorial_interaction_residual_v7(
            factorial, neutral, state_id=state, branch="B_to_A"
        )
        self.assertTrue(torch.equal(a, torch.full_like(a, 7.0)))
        self.assertTrue(torch.equal(b, torch.full_like(b, 3.0)))
        self.assertNotEqual(
            registry.nuisance_state_for_caption("A_to_B", state),
            registry.nuisance_state_for_caption("B_to_A", state),
        )

    def test_missing_factorial_entry_fails_closed(self):
        factorial, neutral = self._additive_factorial(width=1)
        del factorial["appearance_0"]["appearance_2"]
        with self.assertRaisesRegex(
            observer.FactorialCompatibilityTubeV7Error,
            "full three by three",
        ):
            observer.factorial_interaction_residual_v7(
                factorial,
                neutral,
                state_id="appearance_0",
                branch="A_to_B",
            )

    @staticmethod
    def _small_neutral(height=5, width=5, feature_width=2):
        value = torch.zeros(
            (registry.PHASES, height * width, feature_width), dtype=torch.float32
        )
        value[..., 1] = 1.0
        return value

    def test_joint_space_time_tube_connects_motion_before_finalization(self):
        height = width = 5
        delta = torch.zeros((registry.PHASES, 25, 2), dtype=torch.float32)
        neutral = self._small_neutral()
        supports = {0: (6, 7), 2: (7, 8), 4: (8, 9)}
        for phase, patches in supports.items():
            delta[phase, list(patches)] = 10.0
            neutral[phase, list(patches)] = torch.tensor([1.0, 0.0])
        result = observer.construct_space_time_tubes_v7(
            delta,
            neutral,
            active_phases=(0, 2, 4),
            height=height,
            width=width,
        )
        self.assertEqual(len(result.tubes), 1)
        self.assertEqual(result.tubes[0].observed_proposal_phases, (0, 2, 4))
        self.assertGreater(result.temporal_edge_count, 0)
        self.assertEqual(result.dustbin_voxel_count, 0)

    def test_exact_21_by_37_by_25_domain_is_jointly_consumed_by_disjoint_folds(self):
        delta = torch.zeros(
            (registry.PHASES, registry.PATCHES, 2), dtype=torch.float32
        )
        neutral = torch.zeros_like(delta)
        neutral[..., 1] = 1.0
        for phase in observer.TIME_FOLDS["A"]:
            delta[phase, [26, 27]] = 10.0
            neutral[phase, [26, 27]] = torch.tensor([1.0, 0.0])
        result = observer.construct_space_time_tubes_v7(
            delta,
            neutral,
            active_phases=observer.TIME_FOLDS["A"],
            height=registry.PATCH_HEIGHT,
            width=registry.PATCH_WIDTH,
        )
        self.assertEqual(result.tubes[0].observed_proposal_phases, observer.TIME_FOLDS["A"])
        self.assertEqual(result.public_row()["construction_domain"], [21, 37, 25])
        self.assertFalse(set(observer.TIME_FOLDS["A"]) & set(observer.TIME_FOLDS["B"]))
        self.assertEqual(
            set(observer.TIME_FOLDS["A"]) | set(observer.TIME_FOLDS["B"]),
            set(range(21)),
        )

    def test_variable_cardinality_and_explicit_voxel_dustbin(self):
        delta = torch.zeros((registry.PHASES, 25, 2), dtype=torch.float32)
        neutral = self._small_neutral()
        delta[0, [6, 7, 18, 19, 0]] = 10.0
        neutral[0, [6, 7]] = torch.tensor([1.0, 0.0])
        neutral[0, [18, 19]] = torch.tensor([-1.0, 0.0])
        result = observer.construct_space_time_tubes_v7(
            delta,
            neutral,
            active_phases=(0,),
            height=5,
            width=5,
        )
        self.assertEqual(len(result.tubes), 2)
        self.assertEqual(result.eligible_voxel_count, 5)
        self.assertEqual(result.dustbin_voxel_count, 1)

    def test_neutral_unbalanced_ot_rejects_incompatible_temporal_match_to_dustbin(self):
        delta = torch.zeros((registry.PHASES, 25, 2), dtype=torch.float32)
        neutral = self._small_neutral()
        delta[0, [6, 7]] = 10.0
        delta[2, [18, 19]] = 10.0
        neutral[0, [6, 7]] = torch.tensor([1.0, 0.0])
        neutral[2, [18, 19]] = torch.tensor([-1.0, 0.0])
        result = observer.construct_space_time_tubes_v7(
            delta,
            neutral,
            active_phases=(0, 2),
            height=5,
            width=5,
        )
        self.assertEqual(len(result.tubes), 2)
        self.assertEqual(result.temporal_edge_count, 0)
        self.assertEqual(result.temporal_dustbin_assignment_count, 2)
        self.assertIn("unbalanced OT", result.public_row()["temporal_correspondence"])

    def test_zero_field_never_forces_a_tube(self):
        delta = torch.zeros((registry.PHASES, 25, 2), dtype=torch.float32)
        result = observer.construct_space_time_tubes_v7(
            delta,
            self._small_neutral(),
            active_phases=(0, 2, 4),
            height=5,
            width=5,
        )
        self.assertEqual(result.tubes, ())
        self.assertEqual(result.eligible_voxel_count, 0)

    def test_tube_lifecycle_includes_birth_occlusion_reentry_death(self):
        delta = torch.zeros((registry.PHASES, 25, 2), dtype=torch.float32)
        proposer_neutral = self._small_neutral()
        for phase in (0, 4):
            delta[phase, [6, 7]] = 10.0
            proposer_neutral[phase, [6, 7]] = torch.tensor([1.0, 0.0])
        construction = observer.construct_space_time_tubes_v7(
            delta,
            proposer_neutral,
            active_phases=(0, 2, 4),
            height=5,
            width=5,
        )
        evaluator = self._small_neutral()
        for phase in (1, 5):
            evaluator[phase, [6, 7]] = torch.tensor([1.0, 0.0])
        tracks, events, dustbin = observer._evaluate_tubes(
            construction,
            evaluator,
            phase_pairs=((0, 1), (2, 3), (4, 5)),
            height=5,
            width=5,
            prereg=registry.load_preregistration(),
        )
        self.assertEqual(len(tracks), 1)
        kinds = [row["event"] for row in events]
        self.assertEqual(kinds, ["birth", "occlusion", "reentry", "death"])
        self.assertEqual(dustbin, 0)

    def _zero_feature_slab(self):
        zero = torch.zeros(
            (registry.PHASES, registry.PATCHES, 1), dtype=torch.float32
        )
        factorial = {
            state: {
                caption: {block: zero for block in registry.BLOCKS}
                for caption in registry.APPEARANCE_IDS
            }
            for state in registry.APPEARANCE_IDS
        }
        controls = {
            state: {
                arm: {block: zero for block in registry.BLOCKS}
                for arm in registry.CONTROL_ARMS
            }
            for state in registry.APPEARANCE_IDS
        }
        return factorial, controls

    def test_zero_feature_slab_reduces_three_cells_and_abstains(self):
        factorial, controls = self._zero_feature_slab()
        rows = observer.reduce_factorial_feature_slab_v7(
            factorial, controls, sigma_band="high"
        )
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertFalse(row.branchwise_diagnostic_admitted)
            self.assertEqual(set(row.branch_receipts), set(observer.BRANCHES))
            self.assertTrue(
                all(receipt["graph_abstained"] for receipt in row.branch_receipts.values())
            )
            self.assertTrue(
                all(
                    receipt["off_diagonal_folds_disjoint"]
                    for receipt in row.branch_receipts.values()
                )
            )

    def _projected_capture(self, state, arm, block, timestep="b"):
        query = self._shared_query
        hidden = self._shared_hidden
        return v6_observer.AnonymousProjectedArmV6(
            state,
            arm,
            "high",
            block,
            (str(registry.APPEARANCE_IDS.index(state) + 1) * 64)[:64],
            timestep * 64,
            "c" * 64,
            "d" * 64,
            query,
            hidden,
        )

    def test_capture_storage_alias_failure_scrubs_all_owned_v6_projected_rows(self):
        self._shared_query = torch.ones(
            (1, registry.PHASES, registry.PATCHES, 16), dtype=torch.float32
        )
        self._shared_hidden = torch.ones_like(self._shared_query)
        factorial = {}
        controls = {}
        for state in registry.APPEARANCE_IDS:
            factorial[state] = {}
            for caption in registry.APPEARANCE_IDS:
                arm = "action" if state == caption else "source_swap"
                factorial[state][caption] = {
                    block: self._projected_capture(state, arm, block)
                    for block in registry.BLOCKS
                }
            controls[state] = {}
            for arm in registry.CONTROL_ARMS:
                controls[state][arm] = {
                    block: self._projected_capture(
                        state,
                        arm,
                        block,
                        timestep=(
                            "e"
                            if state == "appearance_2"
                            and arm == "lexical_placebo"
                            and block == 24
                            else "b"
                        ),
                    )
                    for block in registry.BLOCKS
                }
        with self.assertRaisesRegex(
            observer.FactorialCompatibilityTubeV7Error,
            "storage ownership is aliased",
        ):
            observer.reduce_factorial_capture_slab_v7(factorial, controls)
        self.assertEqual(torch.count_nonzero(self._shared_query).item(), 0)
        self.assertEqual(torch.count_nonzero(self._shared_hidden).item(), 0)

    def test_malformed_outer_slab_still_scrubs_every_supplied_v6_capture(self):
        self._shared_query = torch.ones(
            (1, registry.PHASES, registry.PATCHES, 16), dtype=torch.float32
        )
        self._shared_hidden = torch.ones_like(self._shared_query)
        row = self._projected_capture("appearance_0", "action", 6)
        malformed = {"appearance_0": {"appearance_0": {6: row}}}
        with self.assertRaisesRegex(
            observer.FactorialCompatibilityTubeV7Error,
            "capture slab states",
        ):
            observer.reduce_factorial_capture_slab_v7(malformed, {})
        self.assertTrue(row.consumed)
        self.assertEqual(torch.count_nonzero(self._shared_query).item(), 0)
        self.assertEqual(torch.count_nonzero(self._shared_hidden).item(), 0)

    def _reduced_cell(self, appearance, sigma, admitted=True):
        branch = {
            "control_executed": {name: True for name in observer.CONTROL_NAMES},
            "branch_pass": admitted,
        }
        return observer.ReducedFactorialTubeCellV7(
            appearance,
            sigma,
            "a" * 64,
            "b" * 64,
            "c" * 64,
            {name: dict(branch) for name in observer.BRANCHES},
            admitted,
            True,
        )

    def test_even_nine_of_nine_diagnostic_keeps_all_scientific_boundaries_false(self):
        stream = observer.FactorialCompatibilityTubeObserverV7()
        for appearance in registry.APPEARANCE_IDS:
            for sigma in registry.SIGMA_CELL_INDICES:
                stream.add(self._reduced_cell(appearance, sigma))
        receipt = stream.finalize()
        self.assertTrue(receipt["diagnostic_component_admitted"])
        for key in (
            "representation_admitted",
            "stable_transferable_action_representation_claimed",
            "scientific_claim_authorized",
            "training_or_parameter_updates_authorized",
            "renderer_or_decoder_authorized",
            "route_or_injection_authorized",
            "prompt_shuffle_control_executed",
            "heldout_transfer_control_executed",
            "gpu_launch_authorized",
            "renderer_called",
            "decoder_called",
            "optimizer_created",
            "route_or_injection_called",
        ):
            self.assertFalse(receipt[key])
        self.assertEqual(receipt["parameter_updates"], 0)
        self.assertFalse(receipt["gpu_runner_implemented"])
        self.assertFalse(
            receipt["factorial_prompt_embedding_runtime_binding_implemented"]
        )
        self.assertTrue(receipt["launch_blocked_pending_independent_audit"])


if __name__ == "__main__":
    unittest.main()
