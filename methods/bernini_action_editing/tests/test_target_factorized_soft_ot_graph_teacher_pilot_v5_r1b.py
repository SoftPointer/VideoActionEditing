#!/usr/bin/env python3
import importlib
import json
from pathlib import Path
import unittest

import numpy as np

from methods.bernini_action_editing import target_factorized_soft_ot_graph_teacher_pilot_v5_r1b as pilot


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "methods/bernini_action_editing/assets/target_factorized_soft_ot_graph_teacher_manifest_v5_r1b.json"
PREREG = ROOT / "methods/bernini_action_editing/assets/target_factorized_soft_ot_graph_teacher_prereg_v5_r1b.json"


def component(phase, index, translation=(0.0, 0.0), perturb=0.0):
    centers = (
        (-0.75 + 0.16 * phase, -0.25 + 0.02 * phase),
        (0.75 - 0.13 * phase, 0.24 - 0.015 * phase),
        (-0.15 + 0.04 * phase, 0.72 - 0.10 * phase),
    )
    motion = np.zeros(pilot.MOTION_DIM, dtype=np.float64)
    motion[(index * 5 + phase // 3) % pilot.MOTION_DIM] = 1.0
    motion[(index * 5 + 1) % pilot.MOTION_DIM] = 0.35 + perturb
    motion /= np.linalg.norm(motion)
    center = np.asarray(centers[index], dtype=np.float64) + np.asarray(translation) + perturb
    return pilot.SoftComponent(
        mass=0.15 + 0.015 * index + 0.004 * phase,
        center=center,
        motion=motion,
        energy=0.25 + 0.04 * index + 0.01 * phase,
        spread=np.asarray([0.18 + 0.01 * index, 0.08 + 0.005 * phase]),
        entropy=0.45 + 0.03 * index,
    )


def phases(translation=(0.0, 0.0), perturb=0.0):
    return tuple(tuple(component(phase, index, translation, perturb) for index in range(3))
                 for phase in range(pilot.PHASES))


class FactorizedSoftOTGraphTest(unittest.TestCase):
    def aggregate_fixture(self):
        pairs = []
        rows = []
        ordinal = 0
        for family in pilot.FAMILIES:
            for local in range(4):
                split = "locked_validation" if local == 3 else "development_report"
                pair_id = f"{ordinal:064x}"
                pairs.append(pilot.PairRow(
                    ordinal, pair_id, f"uuid-{ordinal}", family, split, "instruction",
                    Path("/source"), Path("/target"), {}, {}, {},
                ))
                margins = {name: 0.1 for name in
                           ("target_reverse", "target_deterministic_shuffle", "source_noop")}
                rows.append({
                    "pair_id": pair_id,
                    "metrics": {
                        "pair_pass": True,
                        "all_seven_graphs_mechanically_valid": True,
                        "phase_trunk": {"pass": True, "margins": dict(margins)},
                        "object_trunk": {"pass": True, "input_margins": dict(margins)},
                    },
                })
                ordinal += 1
        return pairs, rows

    def test_exact_r0_phase_trunk(self):
        r0 = importlib.import_module("methods.action_anchor_target_gap_audit.representation_eval")
        hidden = np.arange(3 * 32 * 256 * 1024, dtype=np.float32).reshape(3, 32, 256, 1024)
        hidden %= np.float32(97.0)
        expected = r0.ordered_residual_descriptor(hidden[2].mean(axis=1, dtype=np.float32))
        observed = pilot.phase_descriptor(hidden, r0)
        np.testing.assert_array_equal(observed, expected)

    def test_r0_phase_is_order_sensitive(self):
        r0 = importlib.import_module("methods.action_anchor_target_gap_audit.representation_eval")
        tokens = np.stack([np.sin(np.arange(16) * 0.2 + time * 0.3) for time in range(32)]).astype(np.float32)
        self.assertLess(r0.cosine(r0.ordered_residual_descriptor(tokens),
                                  r0.ordered_residual_descriptor(tokens[::-1])), 0.95)

    def test_soft_component_zero_abstains(self):
        self.assertEqual(pilot.extract_soft_components(np.zeros((pilot.PATCHES, pilot.MOTION_DIM))), ())

    def test_soft_component_one_slot_is_not_promoted(self):
        motion = np.zeros((pilot.PATCHES, pilot.MOTION_DIM), dtype=np.float64)
        motion[17, 3] = 10.0
        self.assertEqual(len(pilot.extract_soft_components(motion)), 1)

    def test_soft_component_variable_cardinality(self):
        motion = np.zeros((pilot.PATCHES, pilot.MOTION_DIM), dtype=np.float64)
        coords = pilot.grid_coordinates()
        motion[:, 0] = np.exp(-4.0 * np.sum((coords - [-0.5, 0.0]) ** 2, axis=1))
        motion[:, 1] = np.exp(-4.0 * np.sum((coords - [0.5, 0.0]) ** 2, axis=1))
        self.assertGreaterEqual(len(pilot.extract_soft_components(motion)), 2)

    def test_empty_transition_cases(self):
        two = phases()[0][:2]
        zero_to_two = pilot.track_transition((), two)
        two_to_zero = pilot.track_transition(two, ())
        zero_to_zero = pilot.track_transition((), ())
        self.assertEqual(zero_to_two.plan.shape, (0, 2))
        self.assertEqual(two_to_zero.plan.shape, (2, 0))
        self.assertEqual(zero_to_zero.plan.shape, (0, 0))
        self.assertEqual((zero_to_two.matched_ratio, two_to_zero.matched_ratio,
                          zero_to_zero.matched_ratio), (0.0, 0.0, 0.0))

    def test_zero_and_one_slot_phases_mask_without_phantoms(self):
        rows = list(phases())
        rows[1] = ()
        rows[3] = rows[3][:1]
        graph = pilot.assemble_soft_graph(tuple(rows))
        self.assertTrue(graph.mechanically_valid)
        self.assertEqual(graph.diagnostics["phase_component_counts"][1], 0)
        self.assertEqual(graph.diagnostics["phase_component_counts"][3], 1)
        self.assertGreaterEqual(graph.diagnostics["valid_phase_count"], 4)
        # 0->3 produces full birth and 3->0 full death on node features.
        self.assertTrue(np.any(graph.node_features[:, 5] > 0.99))
        self.assertTrue(np.any(graph.node_features[:, 6] > 0.99))

    def test_under_four_valid_phases_fails_closed(self):
        rows = list(phases())
        for index in range(5):
            rows[index] = () if index % 2 == 0 else rows[index][:1]
        self.assertFalse(pilot.assemble_soft_graph(tuple(rows)).mechanically_valid)

    def test_translation_invariance(self):
        base = pilot.assemble_soft_graph(phases())
        moved = pilot.assemble_soft_graph(phases(translation=(7.0, -4.0)))
        observed = pilot.graph_similarity(base, moved)
        reference = pilot.graph_similarity(base, base)
        for key in ("node", "edge", "tracking"):
            self.assertAlmostEqual(observed[key], reference[key], places=10)

    def test_consistent_component_relabel_invariance(self):
        base_phases = phases()
        relabeled = tuple(tuple((rows[2], rows[0], rows[1])) for rows in base_phases)
        base = pilot.assemble_soft_graph(base_phases)
        other = pilot.assemble_soft_graph(relabeled)
        observed = pilot.graph_similarity(base, other)
        reference = pilot.graph_similarity(base, base)
        for key in ("node", "edge", "tracking"):
            self.assertAlmostEqual(observed[key], reference[key], places=10)

    def test_reverse_and_shuffle_are_object_sensitive(self):
        base_phases = phases()
        base = pilot.assemble_soft_graph(base_phases)
        positive = pilot.assemble_soft_graph(phases(perturb=0.002))
        reverse = pilot.assemble_soft_graph(tuple(reversed(base_phases)))
        shuffle = pilot.assemble_soft_graph(tuple(base_phases[i] for i in (0, 1, 6, 7, 2, 3, 4, 5)))
        pos = pilot.graph_similarity(base, positive)["lexicographic_min"]
        self.assertGreater(pos, pilot.graph_similarity(base, reverse)["lexicographic_min"])
        self.assertGreater(pos, pilot.graph_similarity(base, shuffle)["lexicographic_min"])

    def test_object_only_counterfactuals_have_incremental_effect(self):
        base = pilot.assemble_soft_graph(phases())
        positive = pilot.assemble_soft_graph(phases(perturb=0.002))
        slot = pilot.spatial_slot_permutation_control(base)
        dropped = pilot.drop_edge_control(base)
        pos = pilot.graph_similarity(base, positive)
        slot_score = pilot.graph_similarity(base, slot)
        drop_score = pilot.graph_similarity(base, dropped)
        self.assertGreater(pos["tracking"], slot_score["tracking"])
        self.assertGreater(pos["edge"], slot_score["edge"])
        self.assertGreater(pos["edge"], drop_score["edge"])

    def test_phase_cannot_compensate_object_failure(self):
        graph = pilot.assemble_soft_graph(phases())
        invalid = pilot.assemble_soft_graph(tuple(() for _ in range(pilot.PHASES)))
        phase_scores = {name: -1.0 for name in pilot.VIEW_ORDER}
        phase_scores["target_forward_eval"] = 1.0
        graphs = {name: graph for name in pilot.VIEW_ORDER}
        graphs["source_noop"] = invalid
        metrics = pilot.evaluate_pair(phase_scores, graphs, graph, graph)
        self.assertTrue(metrics["phase_trunk"]["pass"])
        self.assertFalse(metrics["object_trunk"]["pass"])
        self.assertFalse(metrics["pair_pass"])

    def test_independent_null_bank_is_exact_and_video_free(self):
        bank = pilot.independent_null_calibration_bank()
        self.assertEqual(bank["summary_sha256"],
                         "4efdbb5b038a6970239dc76e86beee8f7d522ac35eea002ac6469639b8fa0d9d")
        self.assertFalse(bank["real_video_or_model_features_used"])
        self.assertEqual(bank["fixed_point_scale"], 1_000_000_000)
        self.assertGreater(bank["positive_minus_null_margin_p05_q1e9"]["tracking"],
                           int(pilot.THRESHOLDS["object_margin_each_input_negative_min"]
                               * bank["fixed_point_scale"]))

    def test_sampling_and_shuffle_controls(self):
        for count in (48, 67, 422):
            reference = pilot.reference_indices(count)
            evaluation = pilot.eval_indices(count)
            self.assertFalse(np.array_equal(reference, evaluation))
            self.assertEqual((reference[0], reference[-1]), (0, count - 1))
            self.assertEqual((evaluation[0], evaluation[-1]), (0, count - 1))
        permutation = pilot.shuffle_permutation("a" * 64)
        self.assertEqual(set(permutation), set(range(8)))
        self.assertNotEqual(permutation, tuple(range(8)))
        self.assertNotEqual(permutation, tuple(reversed(range(8))))

    def test_manifest_authority_fresh16_and_family_split(self):
        value = pilot.load_authority(MANIFEST, pilot.MANIFEST_FILE_SHA256)
        rows = pilot.validate_manifest(value)
        self.assertEqual(len(rows), 16)
        excluded = set(value["selection"]["excluded_prior_uuid"])
        self.assertEqual(len(excluded), 28)
        self.assertFalse(excluded & {row.uuid for row in rows})
        for family in pilot.FAMILIES:
            self.assertEqual(sum(row.family == family and row.split == "development_report"
                                 for row in rows), 3)
            self.assertEqual(sum(row.family == family and row.split == "locked_validation"
                                 for row in rows), 1)

    def test_prereg_authority_and_runtime_contract(self):
        value = pilot.load_authority(PREREG, pilot.PREREG_FILE_SHA256)
        pilot.validate_prereg(value)
        self.assertEqual(value["admission_formula"]["final"],
                         "all global, per-control, and every-family gates must pass")
        self.assertFalse(value["admission_formula"]["weighted_sum_or_cross_branch_compensation"])

    def test_aggregate_fails_when_one_control_is_only_11_of_16(self):
        pairs, rows = self.aggregate_fixture()
        for row in rows[:5]:
            row["metrics"]["phase_trunk"]["margins"]["target_reverse"] = 0.0
        result = pilot.aggregate(rows, pairs)
        self.assertEqual(result["summary"]["phase_forward_above_each_control"]["target_reverse"], 11)
        self.assertFalse(result["admitted"])

    def test_aggregate_family_locked_validation_is_fail_closed(self):
        pairs, rows = self.aggregate_fixture()
        target = next(pair for pair in pairs if pair.family == pilot.FAMILIES[0]
                      and pair.split == "locked_validation")
        next(row for row in rows if row["pair_id"] == target.pair_id)["metrics"]["pair_pass"] = False
        result = pilot.aggregate(rows, pairs)
        self.assertEqual(result["summary"]["locked_validation_pair_pass"], 3)
        self.assertEqual(result["family"][pilot.FAMILIES[0]]["locked_validation_pass"], 0)
        self.assertFalse(result["admitted"])

    def test_aggregate_never_compensates_phase_and_object_branches(self):
        pairs, rows = self.aggregate_fixture()
        for index, row in enumerate(rows):
            if index < 8:
                row["metrics"]["phase_trunk"]["pass"] = False
            else:
                row["metrics"]["object_trunk"]["pass"] = False
            row["metrics"]["pair_pass"] = False
        result = pilot.aggregate(rows, pairs)
        self.assertTrue(all(value == 16 for value in
                            result["summary"]["phase_forward_above_each_control"].values()))
        self.assertTrue(all(value == 16 for value in
                            result["summary"]["object_forward_above_each_control"].values()))
        self.assertFalse(result["admitted"])


if __name__ == "__main__":
    unittest.main()
