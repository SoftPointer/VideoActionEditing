#!/usr/bin/env python3
"""Semantic and ownership mutations for v15c-r9 four-role authority."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "source_role_authority_v15c_r9.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module loader unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load_path("source_role_authority_v15c_r9_test", MODULE_PATH)


def run_statistic_unit(*, tracks, affinity):
    """Exercise math only; the official r9 entry rejects future=true."""
    if affinity.four_role_joint_null_available:
        return M._run_source_four_role_statistic_mechanical_unit_v15c_r9(
            tracks=tracks, affinity=affinity
        )
    return M.run_source_four_role_authority_v15c_r9(
        tracks=tracks, affinity=affinity
    )


def proposal_id(label: str) -> str:
    return "sam2-f000-" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def rectangle(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    value = np.zeros((M.GRID_HEIGHT, M.GRID_WIDTH), dtype=np.float32)
    value[y0:y1, x0:x1] = 1.0
    return value


def good_geometry() -> M.WholeTrackGeometryV15CR9:
    return M.WholeTrackGeometryV15CR9(
        all_81_frames_visible=True,
        area_p95_to_p05_ratio=1.2,
        median_adjacent_iou=0.82,
        p10_area_pixels=80000.0,
        median_largest_component_fraction=0.99,
        median_bbox_fill_fraction=0.55,
        p10_bbox_diagonal_frame_fraction=0.48,
    )


def synthetic_inputs(*, human_signal_phases: int = 21):
    generator = np.random.default_rng(20260821)
    phase_masks = np.stack(
        [
            rectangle(3, 31, 2, 9),
            rectangle(8, 13, 11, 15),
            rectangle(17, 23, 15, 20),
            rectangle(28, 33, 3, 7),
        ],
        axis=0,
    )
    coverage = np.repeat(phase_masks[:, None], M.PHASE_COUNT, axis=1)
    real = generator.normal(
        0.0,
        1.0,
        size=(len(M.BLOCK_INDICES), len(M.ROLE_NAMES), M.PHASE_COUNT, M.GRID_HEIGHT, M.GRID_WIDTH),
    ).astype(np.float32)
    shuffled = generator.normal(0.0, 1.0, size=real.shape).astype(np.float32)
    null = generator.normal(
        0.0,
        1.0,
        size=(
            len(M.BLOCK_INDICES),
            len(M.ROLE_NAMES),
            M.NULL_COUNT,
            M.PHASE_COUNT,
            M.GRID_HEIGHT,
            M.GRID_WIDTH,
        ),
    ).astype(np.float32)
    for block in range(len(M.BLOCK_INDICES)):
        for role in range(len(M.ROLE_NAMES)):
            phases = human_signal_phases if role == 0 else M.PHASE_COUNT
            for phase in range(phases):
                real[block, role, phase] += np.float32(12.0) * phase_masks[role]
    tracks = M.ProposalTrackInputV15CR9(
        proposal_ids=tuple(proposal_id(role) for role in M.ROLE_NAMES),
        phase_coverage=np.ascontiguousarray(coverage),
        track_gate_pass=(True,) * len(M.ROLE_NAMES),
        geometry=(good_geometry(),) * len(M.ROLE_NAMES),
    )
    null = np.ascontiguousarray(null)
    role_tensor_sha = tuple(
        M.array_sha256(null[:, role]) for role in range(len(M.ROLE_NAMES))
    )
    upstream = {
        "schema_version": "bernini-four-role-joint-null-observer-v15c-r10-local",
        "validation_sha256": "7" * 64,
        "capture_channel_registry_sha256": "8" * 64,
        "capture_channel_value_binding_sha256": "9" * 64,
        "independent_capture_channel_value_binding_pinned": False,
        "actual_sp4_rank_shard_files_replayed": False,
        "official_r10_runner_present": False,
        "role_assignment_mechanical_candidate_qualified": False,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
    }
    binding = M.joint_null_binding_payload_v15c_r9(
        real=np.ascontiguousarray(real),
        shuffled=np.ascontiguousarray(shuffled),
        null_bank=null,
        null_registry_sha256=M.V15C_R10_JOINT_NULL_REGISTRY_SHA256,
        role_null_registry_sha256=M.V15C_R10_ROLE_NULL_REGISTRY_SHA256,
        role_null_tensor_sha256=role_tensor_sha,
        upstream_validation=upstream,
    )
    affinity = M.R6AffinityInputV15CR9(
        real=np.ascontiguousarray(real),
        shuffled=np.ascontiguousarray(shuffled),
        null_bank=null,
        null_registry_sha256=M.V15C_R10_JOINT_NULL_REGISTRY_SHA256,
        null_index_alignment_verified=True,
        four_role_joint_null_available=True,
        role_null_registry_sha256=M.V15C_R10_ROLE_NULL_REGISTRY_SHA256,
        role_null_tensor_sha256=role_tensor_sha,
        joint_null_upstream_validation=upstream,
        joint_null_binding_sha256=M.object_sha256(binding),
    )
    return tracks, affinity


class FourRoleAssignmentTests(unittest.TestCase):
    def test_official_r9_entry_rejects_every_future_true_affinity(self):
        tracks, affinity = synthetic_inputs()
        with self.assertRaisesRegex(
            M.SourceRoleAuthorityV15CR9Error, "fresh r10 runner/postflight absent"
        ):
            M.run_source_four_role_authority_v15c_r9(
                tracks=tracks, affinity=affinity
            )

    def test_clear_four_roles_pass_assignment_but_not_ownership_or_science(self):
        tracks, affinity = synthetic_inputs()
        result = run_statistic_unit(tracks=tracks, affinity=affinity)
        self.assertTrue(result["role_assignment_mechanical_candidate_qualified"])
        self.assertFalse(result["ownership_partition_mechanical_candidate_qualified"])
        self.assertFalse(result["mechanical_candidate_qualified"])
        self.assertEqual(
            result["assignments"],
            {role: proposal_id(role) for role in M.ROLE_NAMES},
        )
        self.assertEqual(
            result["multiple_comparison_control"]["minimum_attainable_global_fwer_p"],
            1.0 / 65.0,
        )
        for key in (
            "remote_worker_execution_verified",
            "observer_execution_authorized",
            "localization_semantically_certified",
            "scientific_claim_authorized",
            "route_authorized",
            "decode_authorized",
            "training_authorized",
        ):
            self.assertIs(result[key], False)

    def test_global_null_alignment_missing_is_no_go_for_every_role(self):
        tracks, affinity = synthetic_inputs()
        with self.assertRaisesRegex(
            M.SourceRoleAuthorityV15CR9Error, "lacks index alignment"
        ):
            M.R6AffinityInputV15CR9(
                real=affinity.real,
                shuffled=affinity.shuffled,
                null_bank=affinity.null_bank,
                null_registry_sha256=affinity.null_registry_sha256,
                null_index_alignment_verified=False,
                four_role_joint_null_available=True,
                role_null_registry_sha256=affinity.role_null_registry_sha256,
                role_null_tensor_sha256=affinity.role_null_tensor_sha256,
                joint_null_upstream_validation=(
                    affinity.joint_null_upstream_validation
                ),
                joint_null_binding_sha256=affinity.joint_null_binding_sha256,
            )

    def test_existing_r6_common_null_without_role_axis_is_diagnostic_no_go(self):
        tracks, affinity = synthetic_inputs()
        common = affinity.null_bank[:, 0:1]
        diagnostic_only = M.R6AffinityInputV15CR9(
            real=affinity.real,
            shuffled=affinity.shuffled,
            null_bank=np.ascontiguousarray(
                np.broadcast_to(common, affinity.null_bank.shape)
            ),
            null_registry_sha256=affinity.null_registry_sha256,
            null_index_alignment_verified=True,
            four_role_joint_null_available=False,
        )
        result = run_statistic_unit(
            tracks=tracks, affinity=diagnostic_only
        )
        control = result["multiple_comparison_control"]
        self.assertEqual(
            control["method"],
            "NO_GO_existing_r6_common_null_lacks_four_role_joint_axis",
        )
        self.assertFalse(control["global_four_role_fwer_certified"])
        self.assertFalse(control["common_null_broadcast_used_for_certification"])
        self.assertTrue(all(value is None for value in result["assignments"].values()))
        self.assertFalse(result["role_assignment_mechanical_candidate_qualified"])

    def test_global_max_t_controls_all_four_roles_and_vessel_extra_gate(self):
        tracks, affinity = synthetic_inputs()
        result = run_statistic_unit(tracks=tracks, affinity=affinity)
        for role in M.ROLE_NAMES:
            row = next(item for item in result["evidence"][role] if item["proposal_id"] == proposal_id(role))
            self.assertAlmostEqual(row["global_max_t_empirical_upper_p"], 1.0 / 65.0)
            self.assertTrue(row["gates"]["global_four_role_max_t_fwer"])
            if role in M.VESSEL_ROLE_NAMES:
                self.assertAlmostEqual(row["vessel_three_role_bonferroni_fwer_upper_p"], 3.0 / 65.0)
                self.assertTrue(row["gates"]["vessel_three_role_bonferroni_extra_gate"])

    def test_human_missing_affinity_is_unassigned_without_forcing(self):
        tracks, affinity = synthetic_inputs(human_signal_phases=0)
        result = run_statistic_unit(tracks=tracks, affinity=affinity)
        self.assertIsNone(result["assignments"]["human_agent"])
        self.assertFalse(result["forced_assignment"])
        self.assertFalse(result["role_assignment_mechanical_candidate_qualified"])

    def test_human_fragment_geometry_is_unassigned(self):
        tracks, affinity = synthetic_inputs()
        fragment = M.WholeTrackGeometryV15CR9(
            all_81_frames_visible=True,
            area_p95_to_p05_ratio=8.0,
            median_adjacent_iou=0.1,
            p10_area_pixels=100.0,
            median_largest_component_fraction=0.4,
            median_bbox_fill_fraction=0.02,
            p10_bbox_diagonal_frame_fraction=0.01,
        )
        mutated = M.ProposalTrackInputV15CR9(
            proposal_ids=tracks.proposal_ids,
            phase_coverage=tracks.phase_coverage,
            track_gate_pass=tracks.track_gate_pass,
            geometry=(fragment, *tracks.geometry[1:]),
        )
        result = run_statistic_unit(tracks=mutated, affinity=affinity)
        self.assertIsNone(result["assignments"]["human_agent"])
        row = next(item for item in result["evidence"]["human_agent"] if item["proposal_id"] == tracks.proposal_ids[0])
        self.assertFalse(row["gates"]["human_four_connected_component_support"])
        self.assertFalse(row["gates"]["human_whole_person_area_support"])

    def test_human_duplicate_or_nested_family_unassigns_all_members(self):
        tracks, affinity = synthetic_inputs()
        coverage = np.concatenate([tracks.phase_coverage, tracks.phase_coverage[0:1]], axis=0)
        duplicate_id = proposal_id("human-duplicate")
        mutated = M.ProposalTrackInputV15CR9(
            proposal_ids=(*tracks.proposal_ids, duplicate_id),
            phase_coverage=np.ascontiguousarray(coverage),
            track_gate_pass=(*tracks.track_gate_pass, True),
            geometry=(*tracks.geometry, good_geometry()),
        )
        result = run_statistic_unit(tracks=mutated, affinity=affinity)
        self.assertIsNone(result["assignments"]["human_agent"])
        pairs = result["same_role_duplicate_nesting_families"]["human_agent"]
        self.assertTrue(any(duplicate_id in (row["left"], row["right"]) for row in pairs))

    def test_limited_human_vessel_contact_is_relation_not_conflict(self):
        tracks, affinity = synthetic_inputs()
        coverage = tracks.phase_coverage.copy()
        # One low-resolution contact pixel over all phases: limited relative to
        # both whole person and old vessel, but still explicit relation evidence.
        coverage[1, :, 12, 8] = 1.0
        mutated = M.ProposalTrackInputV15CR9(
            proposal_ids=tracks.proposal_ids,
            phase_coverage=coverage,
            track_gate_pass=tracks.track_gate_pass,
            geometry=tracks.geometry,
        )
        result = run_statistic_unit(tracks=mutated, affinity=affinity)
        relation = next(row for row in result["human_vessel_contact_or_occlusion_evidence"] if row["roles"][1] == "old_actor")
        self.assertTrue(relation["limited_overlap_gate"])
        self.assertGreater(relation["metrics"]["overlap_phase_count"], 0)
        self.assertNotIn("old_actor", {role for row in result["cross_role_conflicts"] for role in row["roles"]})

    def test_heavy_human_vessel_overlap_is_fail_closed(self):
        tracks, affinity = synthetic_inputs()
        coverage = tracks.phase_coverage.copy()
        coverage[1] = coverage[0]
        mutated = M.ProposalTrackInputV15CR9(
            proposal_ids=tracks.proposal_ids,
            phase_coverage=coverage,
            track_gate_pass=tracks.track_gate_pass,
            geometry=tracks.geometry,
        )
        result = run_statistic_unit(tracks=mutated, affinity=affinity)
        self.assertIsNone(result["assignments"]["human_agent"])
        self.assertIsNone(result["assignments"]["old_actor"])
        self.assertTrue(any(row["kind"] == "human_vessel_overlap_not_safely_limited" for row in result["cross_role_conflicts"]))

    def test_vessel_roles_are_strictly_mutually_exclusive(self):
        tracks, affinity = synthetic_inputs()
        # Make old_actor and new_actor text channels independently choose the
        # same source proposal.  The cross-role family must clear both.
        real = affinity.real.copy()
        real[:, 2] = affinity.shuffled[:, 2]
        for block in range(len(M.BLOCK_INDICES)):
            for phase in range(M.PHASE_COUNT):
                real[block, 2, phase] += np.float32(20.0) * tracks.phase_coverage[1, phase]
        real = np.ascontiguousarray(real)
        binding = M.joint_null_binding_payload_v15c_r9(
            real=real,
            shuffled=affinity.shuffled,
            null_bank=affinity.null_bank,
            null_registry_sha256=affinity.null_registry_sha256,
            role_null_registry_sha256=affinity.role_null_registry_sha256,
            role_null_tensor_sha256=affinity.role_null_tensor_sha256,
            upstream_validation=affinity.joint_null_upstream_validation,
        )
        conflict_affinity = M.R6AffinityInputV15CR9(
            real=real,
            shuffled=affinity.shuffled,
            null_bank=affinity.null_bank,
            null_registry_sha256=affinity.null_registry_sha256,
            null_index_alignment_verified=True,
            four_role_joint_null_available=True,
            role_null_registry_sha256=affinity.role_null_registry_sha256,
            role_null_tensor_sha256=affinity.role_null_tensor_sha256,
            joint_null_upstream_validation=(
                affinity.joint_null_upstream_validation
            ),
            joint_null_binding_sha256=M.object_sha256(binding),
        )
        result = run_statistic_unit(tracks=tracks, affinity=conflict_affinity)
        self.assertIsNone(result["assignments"]["old_actor"])
        self.assertIsNone(result["assignments"]["new_actor"])
        self.assertTrue(any(row["kind"] == "vessel_role_tube_conflict" for row in result["cross_role_conflicts"]))


def ownership_inputs(size: int = 24):
    shape = (81, size, size)
    masks = {role: np.zeros(shape, dtype=np.bool_) for role in M.ROLE_NAMES}
    masks["human_agent"][:, 2:18, 2:9] = True
    masks["old_actor"][:, 17:24, 8:15] = True  # one-pixel corner contact
    masks["new_actor"][:, 2:9, 15:22] = True
    masks["new_actor"][:, 4:7, 17:20] = False  # stable handle/interior hole
    masks["recipient"][:, 11:16, 16:21] = True
    logits = {}
    for index, role in enumerate(M.ROLE_NAMES):
        values = np.full(shape, -2.0, dtype=np.float32)
        values[masks[role]] = np.float32(1.0 + index * 0.1)
        logits[role] = values
    # Human is the strict foreground owner at its single contact pixel.
    logits["human_agent"][:, 17, 8] = 3.0
    return masks, logits


class OwnershipPartitionTests(unittest.TestCase):
    def test_contact_overlap_is_preserved_but_final_ownership_is_exclusive(self):
        masks, logits = ownership_inputs()
        result = M.partition_source_role_ownership_v15c_r9(
            proposal_masks=masks,
            replayed_raw_signed_valued_logits=logits,
        )
        receipt = result["receipt"]
        self.assertTrue(receipt["all_four_roles_ownership_qualified"])
        self.assertTrue(receipt["pairwise_exclusive_final_ownership"])
        self.assertGreater(int(result["human_vessel_contact_masks"]["old_actor"].sum()), 0)
        self.assertFalse(bool((result["final_ownership_masks"].sum(axis=0) > 1).any()))
        self.assertFalse(receipt["morphological_repair_applied"])
        self.assertFalse(receipt["route_authorized"])
        self.assertFalse(receipt["training_authorized"])
        self.assertFalse(receipt["scientific_claim_authorized"])

    def test_equal_overlap_logits_become_unassigned_pixels(self):
        masks, logits = ownership_inputs()
        logits["human_agent"][:, 17, 8] = 1.0
        logits["old_actor"][:, 17, 8] = 1.0
        result = M.partition_source_role_ownership_v15c_r9(
            proposal_masks=masks,
            replayed_raw_signed_valued_logits=logits,
        )
        self.assertTrue(bool(result["unassigned_occlusion_mask"][:, 17, 8].all()))
        self.assertFalse(bool(result["final_ownership_masks"][:, :, 17, 8].any()))

    def test_partition_created_fragment_clears_entire_role(self):
        masks, logits = ownership_inputs()
        old = np.zeros_like(masks["old_actor"])
        old[:, 18:22, 8:12] = True
        old[:, 18:22, 14:18] = True
        old[:, 19, 12:15] = True
        masks["old_actor"] = old
        logits["old_actor"] = np.where(old, 1.0, -2.0).astype(np.float32)
        # A connected human finger takes the only bridge pixel.
        masks["human_agent"][:, 2:20, 12] = True
        logits["human_agent"] = np.where(masks["human_agent"], 3.0, -2.0).astype(np.float32)
        result = M.partition_source_role_ownership_v15c_r9(
            proposal_masks=masks,
            replayed_raw_signed_valued_logits=logits,
        )
        self.assertIn("old_actor", result["receipt"]["failed_roles"])
        self.assertFalse(result["receipt"]["role_gates"]["old_actor"]["ownership_is_single_4_connected_component_every_frame"])
        self.assertEqual(int(result["final_ownership_masks"][1].sum()), 0)

    def test_partition_created_hole_clears_entire_role(self):
        masks, logits = ownership_inputs()
        masks["old_actor"][:] = False
        masks["old_actor"][:, 8:17, 8:17] = True
        logits["old_actor"] = np.where(masks["old_actor"], 1.0, -2.0).astype(np.float32)
        masks["human_agent"][:] = False
        masks["human_agent"][:, 12, 12] = True
        logits["human_agent"] = np.where(masks["human_agent"], 3.0, -2.0).astype(np.float32)
        result = M.partition_source_role_ownership_v15c_r9(
            proposal_masks=masks,
            replayed_raw_signed_valued_logits=logits,
        )
        self.assertIn("old_actor", result["receipt"]["failed_roles"])
        self.assertFalse(result["receipt"]["role_gates"]["old_actor"]["ownership_hole_and_component_topology_matches_proposal_every_frame"])

    def test_excessive_overlap_area_loss_clears_role_without_repair(self):
        masks, logits = ownership_inputs()
        masks["human_agent"][:, 17:24, 8:11] = True
        logits["human_agent"] = np.where(masks["human_agent"], 3.0, -2.0).astype(np.float32)
        result = M.partition_source_role_ownership_v15c_r9(
            proposal_masks=masks,
            replayed_raw_signed_valued_logits=logits,
        )
        self.assertIn("old_actor", result["receipt"]["failed_roles"])
        self.assertFalse(result["receipt"]["role_gates"]["old_actor"]["ownership_minimum_area_retention"])
        self.assertFalse(result["receipt"]["morphological_repair_applied"])

    def test_mask_must_be_exact_positive_logit_replay(self):
        masks, logits = ownership_inputs()
        logits["recipient"][:, 12, 17] = -0.1
        with self.assertRaisesRegex(M.SourceRoleAuthorityV15CR9Error, "mask/logit"):
            M.partition_source_role_ownership_v15c_r9(
                proposal_masks=masks,
                replayed_raw_signed_valued_logits=logits,
            )

    def test_v15b_adapter_never_receives_raw_overlap(self):
        masks, logits = ownership_inputs()
        partition = M.partition_source_role_ownership_v15c_r9(
            proposal_masks=masks,
            replayed_raw_signed_valued_logits=logits,
        )
        adapted = M.adapt_qualified_ownership_to_v15b_v15c_r9(
            final_ownership_masks=partition["final_ownership_masks"],
            human_vessel_contact_masks=partition["human_vessel_contact_masks"],
        )
        self.assertFalse(bool((adapted["role_masks"].sum(axis=0) > 1).any()))
        self.assertTrue(adapted["receipt"]["raw_overlapping_proposals_passed_to_v15b"] is False)
        self.assertTrue(adapted["receipt"]["contact_relation_mask_is_independent"])
        self.assertFalse(adapted["receipt"]["v15b_source_role_mask_set_creation_authorized"])
        self.assertFalse(adapted["receipt"]["route_authorized"])

    def test_v15b_adapter_fragment_is_whole_role_no_go(self):
        masks, _logits = ownership_inputs()
        final = np.stack([masks[role] for role in M.ROLE_NAMES], axis=0)
        # Remove the normal old actor and replace it by two disjoint tubes.
        final[1] = False
        final[1, :, 2:6, 10:14] = True
        final[1, :, 18:22, 10:14] = True
        contacts = {
            role: np.zeros(final.shape[1:], dtype=np.bool_)
            for role in M.VESSEL_ROLE_NAMES
        }
        adapted = M.adapt_qualified_ownership_to_v15b_v15c_r9(
            final_ownership_masks=np.ascontiguousarray(final),
            human_vessel_contact_masks=contacts,
        )
        self.assertIn("old_actor", adapted["receipt"]["failed_roles"])
        self.assertEqual(int(adapted["role_masks"][1].sum()), 0)
        self.assertFalse(adapted["receipt"]["mechanical_candidate_qualified"])


if __name__ == "__main__":
    unittest.main()
