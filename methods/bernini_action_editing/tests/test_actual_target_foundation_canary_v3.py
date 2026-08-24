#!/usr/bin/env python3
from __future__ import annotations

import copy
import inspect
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import actual_target_foundation_canary_v3 as authority
import actual_target_foundation_graph_v3 as graph

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def valid_evidence(family: str, pair_id: str) -> authority.CaseEvidenceV3:
    margins = {name: 0.1 for name in authority.INPUT_CONTROLS}
    return authority.CaseEvidenceV3(
        family=family,
        pair_id=pair_id,
        branches={
            "frozen_base": {
                "all_models_eval_frozen": True,
                "source_and_weight_closure_unchanged": True,
                "parameter_updates": 0,
                "generator_forward_calls": 0,
                "actual_forward_hook_delta": {"sam2_image_encoder": 24, "dinov2": 24, "cotracker": 5, "vjepa2": 5},
                "full_model_closure_deferred_to_run_receipt": True,
            },
            "node": {
                "dustbin_used": True,
                "unbalanced_phase_pair_count": 7,
                "dustbin_unmatched_count": 2,
                "dustbin_transport_mass": 0.3,
                "forced_nonempty_slot_used": False,
                "anonymous_slot_relabel_invariant": True,
                "phase_cardinalities": [2] * 8,
                "mechanically_valid_phases": 8,
                "positive_similarity": 0.9,
                "input_margins": margins,
                "mask_descriptor_binding_break_margin": 0.1,
            },
            "track": {
                "assigned_track_count": 2,
                "assigned_point_count": 6,
                "minimum_same_track_member_phases_observed": 3,
                "visible_and_member_fraction": 0.8,
                "per_phase_visible_member_counts": [5] * 8,
                "ambiguous_overlap_observation_count": 1,
                "out_of_bounds_observation_count": 1,
                "nonfinite_observation_count": 0,
                "vote_tie_abstain_count": 1,
                "insufficient_membership_abstain_count": 2,
                "state_counts": {"ABSENT": 1, "VISIBLE_MEMBER": 12, "OCCLUDED": 1, "VISIBLE_OUTSIDE_MASK": 2},
                "lifecycle_counts": {"entry": 2, "occlusion": 1, "membership_loss": 1, "reentry": 2, "death": 1},
                "dynamic_nonentry_lifecycle_observed": True,
                "valid_adjacent_velocity_count": 5,
                "positive_similarity": 0.8,
                "input_margins": margins,
                "cross_phase_track_identity_break_margin": 0.1,
            },
            "edge": {
                "per_phase_active_counts": [0, 1, 1, 0, 0, 0, 0, 0],
                "per_phase_birth_counts": [0, 1, 0, 0, 0, 0, 0, 0],
                "per_phase_persist_counts": [0, 0, 1, 0, 0, 0, 0, 0],
                "per_phase_death_counts": [0, 0, 0, 1, 0, 0, 0, 0],
                "per_phase_valid_velocity_counts": [0, 1, 1, 0, 0, 0, 0, 0],
                "per_phase_qualified_lifecycle_counts": [0, 0, 1, 1, 0, 0, 0, 0],
                "evaluated_pairwise_edge_count": 2,
                "real_per_phase_lifecycle_channels": True,
                "positive_similarity": 0.8,
                "input_margins": margins,
                "drop_edge_margin": 0.2,
                "drop_edge_removed_count": 2,
                "drop_edge_control_norm": 1.0,
                "drop_edge_control_similarity": 0.6,
                "drop_edge_positive_l2_distance": 0.5,
            },
            "ordered_phase": {"input_margins": {name: 0.02 for name in authority.INPUT_CONTROLS}},
        },
    )


class AuthorityTests(unittest.TestCase):
    def test_authority_is_development_only_and_launch_authorized(self) -> None:
        value = authority.load_authority()
        self.assertTrue(value["boundaries"]["real_gpu_launch_authorized"])
        self.assertTrue(authority.REAL_GPU_LAUNCH_AUTHORIZED)
        self.assertTrue(value["boundaries"]["representation_admission_hard_false"])
        self.assertFalse(value["scope"]["scientific_representation_claim_permitted"])
        self.assertIn("not physical-contact truth", value["edge_contract"]["claim_boundary"])
        self.assertTrue(value["fixed_paths"]["fresh_formal_run_root"].endswith("v3r4"))
        self.assertTrue(
            value["v3r2_failed_engineering_attempt"][
                "immutable_preservation_required"
            ]
        )
        self.assertFalse(
            value["v3r2_failed_engineering_attempt"]["candidate_present"]
        )
        self.assertTrue(
            value["v3r3_failed_engineering_attempt"][
                "immutable_preservation_required"
            ]
        )
        self.assertFalse(
            value["v3r3_failed_engineering_attempt"]["candidate_present"]
        )
        self.assertTrue(
            value["v3r4_engineering_repair_contract"][
                "sequence_truthiness_forbidden"
            ]
        )
        self.assertTrue(
            value["v3r4_engineering_repair_contract"][
                "full_numpy_node_track_edge_drop_edge_phase_case_path_test_required"
            ]
        )
        self.assertTrue(
            value["v3r3_engineering_repair_contract"][
                "sam_external_mask_full_backing_storage_span_required"
            ]
        )
        self.assertTrue(
            value["v3r3_engineering_repair_contract"][
                "sam_per_mask_claim_copy_release_order_required"
            ]
        )
        self.assertTrue(
            value["v3r3_engineering_repair_contract"][
                "sam_arbitrary_strided_and_partial_base_views_rejected"
            ]
        )
        source_evidence = value["v3r3_engineering_repair_contract"][
            "sam_pinned_binary_mask_source_evidence"
        ]
        self.assertIn("not inferred from the V3R2", source_evidence["claim_boundary"])
        self.assertEqual(
            {row["role"]: row["sha256"] for row in source_evidence["sources"]},
            {
                "uncompressed_rle_to_mask": "b7b33090e2af72e04dbb815c8f32aff41a4ed1abf9668f62b59f1bdd640ca5d8",
                "automatic_binary_mask_return": "66df266dbe14412305ae3398f0ec1bb21b303a93216b102d767e6c4ee5d4c3d7",
            },
        )
        self.assertEqual(
            value["v3r2_failed_engineering_attempt"][
                "failure_closure_receipt"
            ]["self_sha256"],
            "dfed283e4a9f1716dac887070f9f704c47110a5d7cf6e364173f828c89cbd128",
        )
        self.assertEqual(
            value["v3r3_failed_engineering_attempt"][
                "failure_closure_receipt"
            ]["self_sha256"],
            "b9b8841ff85f9ff74588d6ba7b29f14362815269c776a9ff631ae49e2f21ec25",
        )
        self.assertNotEqual(
            value["fixed_paths"]["fresh_formal_run_root"],
            value["prior_failed_engineering_attempt"]["run_root"],
        )
        self.assertEqual(
            value["prior_failed_engineering_attempt"]["formal_log"]["sha256"],
            "a4d10d94b465ce32955c976d6b86d519be788e2fd27e9403c9c27da6e53c293b",
        )
        self.assertEqual(
            value["prior_failed_engineering_attempt"]["attempt_ledger"]["sha256"],
            "05db949ca238195f951ecd3944eccf27ea889b4d5e987c9e76c9504a93d0bc59",
        )
        self.assertIn(
            "sam_mask_c_contiguous_copy",
            value["raw_inventory_required_categories"],
        )

    def test_strict_json_rejects_duplicate_and_nonfinite(self) -> None:
        with self.assertRaises(authority.CanaryV3Error):
            authority.strict_json_bytes(b'{"a":1,"a":2}')
        for token in (b"NaN", b"Infinity", b"-Infinity"):
            with self.assertRaises(authority.CanaryV3Error):
                authority.strict_json_bytes(b'{"a":' + token + b"}")

    def test_case_schema_and_all_branches_pass(self) -> None:
        pair = authority.load_preregistration()["pairs"][0]
        evidence = valid_evidence(pair["family"], pair["pair_id"])
        row = authority.evaluate_case(evidence)
        self.assertTrue(row["case_pass"])
        self.assertEqual(set(row["scalar_metrics"]), set(authority.BRANCHES))
        rebuilt = authority.CaseEvidenceV3.from_mapping(evidence.to_mapping())
        self.assertEqual(authority.evaluate_case(rebuilt), row)

    def test_each_branch_is_noncompensable(self) -> None:
        pair = authority.load_preregistration()["pairs"][0]
        for branch in authority.BRANCHES:
            value = valid_evidence(pair["family"], pair["pair_id"]).to_mapping()
            if branch == "frozen_base":
                value["branches"][branch]["all_models_eval_frozen"] = False
            elif branch == "node":
                value["branches"][branch]["dustbin_used"] = False
            elif branch == "track":
                value["branches"][branch]["minimum_same_track_member_phases_observed"] = 2
            elif branch == "edge":
                value["branches"][branch]["drop_edge_positive_l2_distance"] = 0.0
            else:
                value["branches"][branch]["input_margins"]["target_reverse"] = 0.0
            row = authority.evaluate_case(authority.CaseEvidenceV3.from_mapping(value))
            self.assertFalse(row["branch_pass"][branch])
            self.assertFalse(row["case_pass"])

    def test_null_abstentions_are_json_safe_and_fail_without_exception(self) -> None:
        pair = authority.load_preregistration()["pairs"][0]
        value = valid_evidence(pair["family"], pair["pair_id"]).to_mapping()
        value["branches"]["node"]["positive_similarity"] = None
        value["branches"]["node"]["input_margins"] = {
            name: None for name in authority.INPUT_CONTROLS
        }
        value["branches"]["track"]["visible_and_member_fraction"] = None
        value["branches"]["track"]["positive_similarity"] = None
        value["branches"]["edge"]["positive_similarity"] = None
        value["branches"]["edge"]["drop_edge_margin"] = None
        value["branches"]["edge"]["drop_edge_control_similarity"] = None
        value["branches"]["ordered_phase"]["input_margins"] = {
            name: None for name in authority.INPUT_CONTROLS
        }
        evidence = authority.CaseEvidenceV3.from_mapping(value)
        row = authority.evaluate_case(evidence)
        self.assertFalse(row["case_pass"])
        self.assertFalse(row["branch_pass"]["node"])
        self.assertFalse(row["branch_pass"]["track"])
        self.assertFalse(row["branch_pass"]["edge"])
        self.assertFalse(row["branch_pass"]["ordered_phase"])
        authority.canonical_json_bytes(row)

    def test_aggregate_recomputes_rows_and_rejects_forgery(self) -> None:
        evidences = [valid_evidence(pair["family"], pair["pair_id"]) for pair in authority.load_preregistration()["pairs"]]
        rows = [authority.evaluate_case(row) for row in evidences]
        aggregate = authority.aggregate_canary(rows, evidences)
        self.assertTrue(aggregate["diagnostic_canary_pass"])
        forged = copy.deepcopy(rows)
        forged[0]["case_pass"] = False
        with self.assertRaises(authority.CanaryV3Error):
            authority.aggregate_canary(forged, evidences)


@unittest.skipIf(np is None, "numpy unavailable")
class GraphAdversarialTests(unittest.TestCase):
    def test_cotracker_inputs_require_exact_no_copy_float64_bool_buffers(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[4:12, 4:12] = True
        phases = tuple((self.node(mask, track_id=0),) for _ in range(8))
        xy = np.full((8, 1, 2), (6.0, 6.0), dtype=np.float64)
        visible = np.ones((8, 1), dtype=bool)
        result = graph.assign_points_with_same_track_membership(
            phases, xy, visible
        )
        self.assertEqual(len(result.memberships), 1)
        with self.assertRaisesRegex(graph.GraphV3Error, "float64"):
            graph.assign_points_with_same_track_membership(
                phases, xy.astype(np.float32), visible
            )
        with self.assertRaisesRegex(graph.GraphV3Error, "C-contiguous"):
            graph.assign_points_with_same_track_membership(
                phases,
                np.broadcast_to(xy[:, :1], (8, 2, 2)),
                np.ones((8, 2), dtype=bool),
            )
        source = inspect.getsource(graph.assign_points_with_same_track_membership)
        self.assertIn("xy = coordinates_xy", source)
        self.assertIn("vis = visible", source)
        self.assertNotIn("asarray(coordinates_xy", source)

    def test_node_key_hashes_owned_bool_buffer_without_copy_or_tobytes(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:5, 1:4] = True
        node = graph.AnonymousNodeV3(mask, np.ones(8), float(mask.mean()), (0.3, 0.4))
        self.assertEqual(len(graph._node_key(node)), 32)
        with self.assertRaises(graph.GraphV3Error):
            graph._node_key(
                graph.AnonymousNodeV3(
                    mask[:, ::2], np.ones(8), float(mask[:, ::2].mean()), (0.3, 0.4)
                )
            )
        with self.assertRaises(graph.GraphV3Error):
            graph._node_key(
                graph.AnonymousNodeV3(
                    mask.astype(np.uint8), np.ones(8), float(mask.mean()), (0.3, 0.4)
                )
            )
        source = inspect.getsource(graph._node_key)
        self.assertNotIn("tobytes", source)
        self.assertNotIn("ascontiguousarray", source)

    @staticmethod
    def node(mask, descriptor=(1.0, 0.0), track_id=-1):
        ys, xs = np.asarray(mask).nonzero()
        return graph.AnonymousNodeV3(
            np.asarray(mask, dtype=bool),
            np.asarray(descriptor, dtype=np.float64),
            float(np.asarray(mask).mean()),
            (float(xs.mean() / 19), float(ys.mean() / 19)),
            track_id,
        )

    def test_dormant_track_reentry_across_two_absent_phases(self) -> None:
        mask = np.zeros((20, 20), bool); mask[5:9, 5:9] = True
        phases = [(self.node(mask),), (), ()] + [(self.node(mask),)] * 5
        tracked = graph.assign_anonymous_tracks(phases, max_absent_gap_phases=2)
        self.assertEqual(tracked[0][0].track_id, tracked[3][0].track_id)
        xy = np.full((8, 1, 2), (6.0, 6.0)); visible = np.ones((8, 1), bool)
        assignment = graph.assign_points_with_same_track_membership(tracked, xy, visible)
        self.assertEqual(len(assignment.memberships), 1)
        lifecycle = assignment.memberships[0].lifecycle
        self.assertEqual(lifecycle["entry"], 1)
        self.assertEqual(lifecycle["death"], 1)
        self.assertEqual(lifecycle["reentry"], 1)

    def test_out_of_bounds_is_not_clamped_into_mask(self) -> None:
        mask = np.zeros((20, 20), bool); mask[:, 0] = True
        phases = tuple((self.node(mask, track_id=0),) for _ in range(8))
        xy = np.full((8, 1, 2), (-50.0, 5.0)); visible = np.ones((8, 1), bool)
        result = graph.assign_points_with_same_track_membership(phases, xy, visible)
        self.assertEqual(result.memberships, ())
        self.assertEqual(result.out_of_bounds_observation_count, 8)

    def test_overlap_and_vote_tie_abstain(self) -> None:
        mask = np.zeros((20, 20), bool); mask[4:12, 4:12] = True
        overlap_phases = tuple((self.node(mask, track_id=0), self.node(mask, (0.0, 1.0), 1)) for _ in range(8))
        xy = np.full((8, 1, 2), (6.0, 6.0)); visible = np.ones((8, 1), bool)
        overlap = graph.assign_points_with_same_track_membership(overlap_phases, xy, visible)
        self.assertEqual(overlap.memberships, ())
        self.assertEqual(overlap.ambiguous_overlap_observation_count, 8)

        left = np.zeros((20, 20), bool); left[4:10, 4:10] = True
        right = np.zeros((20, 20), bool); right[4:10, 12:18] = True
        phases = tuple((self.node(left, track_id=0), self.node(right, (0.0, 1.0), 1)) for _ in range(8))
        tie_xy = np.asarray([[[6.0, 6.0]]] * 4 + [[[14.0, 6.0]]] * 4)
        tie = graph.assign_points_with_same_track_membership(phases, tie_xy, visible)
        self.assertEqual(tie.memberships, ())
        self.assertEqual(tie.vote_tie_abstain_count, 1)

    def test_singleton_binding_negative_preserves_exact_eight_phase_shape(self) -> None:
        mask = np.zeros((20, 20), bool); mask[4:12, 4:12] = True
        phases = tuple((self.node(mask, track_id=phase),) for phase in range(8))
        broken = graph.break_mask_descriptor_binding(phases)
        self.assertEqual(len(broken), 8)
        self.assertEqual(broken, ((),) * 8)

    def test_full_transition_table_and_right_censor(self) -> None:
        self.assertEqual(
            set(graph.TRANSITION_EVENTS),
            {(left, right) for left in graph.TRACK_STATES for right in graph.TRACK_STATES},
        )
        mask = np.zeros((20, 20), bool); mask[4:12, 4:12] = True
        phases = tuple((self.node(mask, track_id=0),) for _ in range(8))
        xy = np.full((8, 1, 2), (6.0, 6.0)); xy[1, 0] = (16.0, 16.0)
        visible = np.ones((8, 1), bool); visible[3, 0] = False
        result = graph.assign_points_with_same_track_membership(phases, xy, visible)
        row = result.memberships[0]
        self.assertEqual(row.lifecycle["membership_loss"], 1)
        self.assertEqual(row.lifecycle["occlusion"], 1)
        self.assertEqual(row.lifecycle["reentry"], 2)
        self.assertEqual(row.lifecycle["death"], 0)
        self.assertEqual(int(row.velocity_valid.sum()), 3)

    @staticmethod
    def membership(track_id: int, centers) -> graph.TrackMembershipV3:
        centers = np.asarray(centers, dtype=np.float64)
        return graph.TrackMembershipV3(
            track_id=track_id,
            point_indices=(track_id,),
            member_phase_counts=(8,),
            phase_member_counts=(1,) * 8,
            phase_visible_counts=(1,) * 8,
            phase_states=("VISIBLE_MEMBER",) * 8,
            centers_xy=centers,
            center_valid=np.ones(8, bool),
            velocities_xy=np.zeros((8, 2), np.float64),
            velocity_valid=np.asarray([False] + [True] * 7),
            lifecycle={"entry": 1, "occlusion": 0, "membership_loss": 0, "reentry": 0, "death": 0},
        )

    def test_flexible_interaction_activate_persist_deactivate_and_real_drop(self) -> None:
        far_l = np.zeros((20, 20), bool); far_l[8:11, 1:4] = True
        far_r = np.zeros((20, 20), bool); far_r[8:11, 16:19] = True
        close_l = np.zeros((20, 20), bool); close_l[8:11, 7:10] = True
        close_r = np.zeros((20, 20), bool); close_r[8:11, 10:13] = True
        phases = []
        for phase in range(8):
            lhs, rhs = (close_l, close_r) if phase in (1, 2) else (far_l, far_r)
            phases.append((self.node(lhs, track_id=0), self.node(rhs, (0.0, 1.0), 1)))
        centers = np.zeros((8, 2))
        edge = graph.per_phase_edge_signatures(tuple(phases), (self.membership(0, centers), self.membership(1, centers)))
        self.assertEqual(edge.per_phase_active_counts, (0, 1, 1, 0, 0, 0, 0, 0))
        self.assertEqual(edge.per_phase_birth_counts[1], 1)
        self.assertEqual(edge.per_phase_persist_counts[2], 1)
        self.assertEqual(edge.per_phase_death_counts[3], 1)
        self.assertGreaterEqual(edge.per_phase_qualified_lifecycle_counts[2], 1)
        self.assertEqual(edge.per_phase_death_counts[-1], 0)
        self.assertEqual(edge.dropped_signature[1 * 5 + 3], 0.0)
        self.assertNotEqual(edge.signature, edge.dropped_signature)
        self.assertGreater(sum(value * value for value in edge.dropped_signature), 0.0)

    def test_track_and_edge_signatures_ignore_arbitrary_track_id_relabel(self) -> None:
        mask_a = np.zeros((20, 20), bool); mask_a[8:11, 7:10] = True
        mask_b = np.zeros((20, 20), bool); mask_b[8:11, 10:13] = True
        phases = tuple((self.node(mask_a, track_id=2), self.node(mask_b, (0.0, 1.0), 9)) for _ in range(8))
        centers = np.zeros((8, 2))
        memberships = (self.membership(2, centers), self.membership(9, centers))
        signature = graph.canonical_track_signature(memberships, {2: (1.0, 0.0), 9: (0.0, 1.0)})
        edge = graph.per_phase_edge_signatures(phases, memberships)
        relabeled_phases = tuple(
            tuple(graph.AnonymousNodeV3(node.mask, node.descriptor, node.area_fraction, node.centroid_xy, {2: 101, 9: 4}[node.track_id]) for node in phase)
            for phase in phases
        )
        relabeled_memberships = (
            graph.TrackMembershipV3(**{**memberships[1].__dict__, "track_id": 4}),
            graph.TrackMembershipV3(**{**memberships[0].__dict__, "track_id": 101}),
        )
        relabeled_signature = graph.canonical_track_signature(relabeled_memberships, {101: (1.0, 0.0), 4: (0.0, 1.0)})
        relabeled_edge = graph.per_phase_edge_signatures(relabeled_phases, relabeled_memberships)
        self.assertEqual(signature, relabeled_signature)
        self.assertEqual(edge.signature, relabeled_edge.signature)
        self.assertEqual(edge.dropped_signature, relabeled_edge.dropped_signature)

    def test_three_tied_motion_tracks_keep_drop_edge_exact_under_global_relabel_and_reorder(self) -> None:
        mask_a = np.zeros((20, 20), bool); mask_a[8:12, 1:5] = True
        mask_b = np.zeros((20, 20), bool); mask_b[8:12, 5:9] = True
        mask_c = np.zeros((20, 20), bool); mask_c[8:12, 7:11] = True
        nodes = (
            self.node(mask_a, (1.0, 0.0, 0.0), 2),
            self.node(mask_b, (0.0, 1.0, 0.0), 9),
            self.node(mask_c, (0.0, 0.0, 1.0), 15),
        )
        phases = tuple(nodes for _ in range(8))
        centers = np.zeros((8, 2))
        memberships = tuple(self.membership(track_id, centers) for track_id in (2, 9, 15))
        edge = graph.per_phase_edge_signatures(phases, memberships)
        self.assertEqual(edge.per_phase_active_counts, (2,) * 8)

        relabel = {2: 101, 9: 4, 15: 77}
        relabeled_phases = tuple(
            tuple(
                graph.AnonymousNodeV3(
                    node.mask,
                    node.descriptor,
                    node.area_fraction,
                    node.centroid_xy,
                    relabel[node.track_id],
                )
                for node in reversed(phase)
            )
            for phase in phases
        )
        relabeled_memberships = tuple(
            self.membership(track_id, centers) for track_id in (77, 4, 101)
        )
        relabeled_edge = graph.per_phase_edge_signatures(
            relabeled_phases, relabeled_memberships
        )
        self.assertEqual(edge.signature, relabeled_edge.signature)
        self.assertEqual(edge.dropped_signature, relabeled_edge.dropped_signature)
        self.assertEqual(
            edge.per_phase_birth_counts, relabeled_edge.per_phase_birth_counts
        )
        self.assertEqual(
            edge.per_phase_persist_counts, relabeled_edge.per_phase_persist_counts
        )
        self.assertEqual(
            edge.per_phase_death_counts, relabeled_edge.per_phase_death_counts
        )


if __name__ == "__main__":
    unittest.main()
