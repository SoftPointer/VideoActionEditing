from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_graft_phase_a_short_trajectory_diagnostic_gpu_v1 as trajectory


AUTHORITY = trajectory.AUTHORITY_FIELDS


def false_authority() -> dict[str, bool]:
    return {name: False for name in AUTHORITY}


def sealed(value: dict) -> dict:
    return dict(trajectory.seal_mapping(value))


def fake_metrics(*, arm: int, stage: str, index: int, passed: bool) -> dict:
    correct_loss = 0.5 if passed else 1.0
    wrong_loss = dropped_loss = 1.0
    wrong_gain = (wrong_loss - correct_loss) / wrong_loss
    dropped_gain = (dropped_loss - correct_loss) / dropped_loss
    correct_norm, dropped_norm, cosine = 2.0, 1.0, 1.0
    ratio = correct_norm / dropped_norm
    gates = {
        "correct_vs_wrong_noop_relative_gain": (
            wrong_gain >= trajectory.short_trainer.MIN_CONFIRMATION_RELATIVE_GAIN
        ),
        "correct_vs_drop_noop_relative_gain": (
            dropped_gain >= trajectory.short_trainer.MIN_CONFIRMATION_RELATIVE_GAIN
        ),
        "action_delta_correct_drop_norm_ratio": (
            ratio
            >= trajectory.short_trainer.MIN_ACTION_DELTA_CORRECT_DROP_NORM_RATIO
        ),
        "action_delta_correct_drop_cosine": (
            cosine >= trajectory.short_trainer.MIN_ACTION_DELTA_COSINE
        ),
    }
    roles = tuple(trajectory.short_trainer.CONFIRMATION_FIELD_ROLES)
    return sealed(
        {
            "schema_version": "bernini-graft-phase-a-confirmation-metrics-v1",
            "schedule_index": index,
            "field_roles": list(roles),
            "field_shape": [2, 3],
            "field_dtype": "torch.float32",
            "field_device_type": "cuda",
            "field_tensor_sha256": {
                role: trajectory.object_sha256(
                    {"field": role, "arm": arm, "stage": stage, "index": index}
                )
                for role in roles
            },
            "noop_fm_loss_float64_hex": {
                "correct_atlas": correct_loss.hex(),
                "wrong_atlas": wrong_loss.hex(),
                "dropped_atlas": dropped_loss.hex(),
            },
            "relative_gain_formula": (
                "(L_control-L_correct)/max(L_control,float64_tiny)"
            ),
            "relative_gain_float64_hex": {
                "correct_vs_wrong": wrong_gain.hex(),
                "correct_vs_drop": dropped_gain.hex(),
            },
            "minimum_relative_gain_float64_hex": (
                trajectory.short_trainer.MIN_CONFIRMATION_RELATIVE_GAIN.hex()
            ),
            "action_delta_formula": "v_action-v_noop",
            "action_delta_norm_float64_hex": {
                "correct_atlas": correct_norm.hex(),
                "dropped_atlas": dropped_norm.hex(),
            },
            "action_delta_correct_drop_norm_ratio_formula": (
                "norm(delta_correct)/max(norm(delta_drop),float64_tiny)"
            ),
            "action_delta_correct_drop_norm_ratio_float64_hex": ratio.hex(),
            "minimum_action_delta_correct_drop_norm_ratio_float64_hex": (
                trajectory.short_trainer.MIN_ACTION_DELTA_CORRECT_DROP_NORM_RATIO.hex()
            ),
            "action_delta_correct_drop_cosine_float64_hex": cosine.hex(),
            "minimum_action_delta_cosine_float64_hex": (
                trajectory.short_trainer.MIN_ACTION_DELTA_COSINE.hex()
            ),
            "float64_tiny_hex": trajectory.short_trainer.FLOAT64_TINY.hex(),
            "noncompensating_gates": gates,
            "noncompensating_all_pass": all(gates.values()),
            "metrics_computed_from_six_detached_fields_by_this_core": True,
            "field_origin_same_noise_state_coordinate_verified_by_this_core": False,
            **false_authority(),
        }
    )


def fake_provenance(*, arm: int, sp_rank: int, stage: str, index: int) -> dict:
    roles = tuple(trajectory.short_trainer.CONFIRMATION_FIELD_ROLES)
    same_state = {
        name: {"content_sha256": trajectory.object_sha256({"same": name, "arm": arm})}
        for name in trajectory._PRODUCTION_SAME_STATE_IDENTITY_FIELDS
    }
    return sealed(
        {
            "schema_version": trajectory.CONFIRMATION_FIELDS_SCHEMA_VERSION,
            "schedule_index": index,
            "confirmation_iid": trajectory.CONFIRMATION_IID_BY_DP_ARM[arm],
            "confirmation_source_sha256": trajectory.object_sha256(
                {"confirmation-source": arm}
            ),
            "wrong_owner_iid": trajectory.FIT_IID_BY_DP_ARM[arm],
            "wrong_owner_source_sha256": trajectory.object_sha256(
                {"wrong-source": arm}
            ),
            "field_roles": list(roles),
            "field_tensor_identities": {
                role: {
                    "content_sha256": trajectory.object_sha256(
                        {
                            "identity": role,
                            "arm": arm,
                            "sp": sp_rank,
                            "stage": stage,
                            "index": index,
                        }
                    )
                }
                for role in roles
            },
            "same_state_identities_before_model_fields": same_state,
            "same_state_identities_after_all_fields": copy.deepcopy(same_state),
            "same_state_tensor_identities_recomputed_byte_equal": True,
            "wrong_route_receipts_differ_only_in_atlas_memory": True,
            "drop_route_receipts_retain_v_branch_disable_only_rebinder": True,
            "action_noop_route_receipts_equal_with_negative_raw_reuse": True,
            **{name: True for name in trajectory._CONFIRMATION_TRUE_FLAGS},
            **{name: False for name in trajectory._CONFIRMATION_FALSE_FLAGS},
            **false_authority(),
        }
    )


def make_packets(
    *,
    no_go: bool = True,
    arm_passes: tuple[bool, bool] | None = None,
) -> list[dict]:
    if arm_passes is None:
        arm_passes = (not no_go, not no_go)
    matrices_by_rank: list[list[dict]] = [[] for _ in range(8)]
    for rank in range(8):
        arm, sp_rank = divmod(rank, 4)
        for stage in trajectory.TRAJECTORY_STAGES:
            parameter = trajectory.object_sha256({"parameter-stage": stage})
            cells = []
            for index in trajectory.CONFIRMATION_INDICES:
                passed = not (
                    stage == "after_update_38"
                    and index == 29
                    and not arm_passes[arm]
                )
                metrics = fake_metrics(
                    arm=arm, stage=stage, index=index, passed=passed
                )
                provenances = [
                    fake_provenance(
                        arm=arm,
                        sp_rank=item,
                        stage=stage,
                        index=index,
                    )
                    for item in range(4)
                ]
                owner = trajectory.object_sha256(
                    {"owner": arm, "stage": stage, "index": index}
                )
                atlas = trajectory.object_sha256(
                    {"atlas": arm, "stage": stage, "index": index}
                )
                state = trajectory.object_sha256(
                    {"state": arm, "stage": stage, "index": index}
                )
                route_digests = [
                    trajectory.object_sha256(
                        {"route": item, "arm": arm, "stage": stage, "index": index}
                    )
                    for item in range(4)
                ]
                digests = sealed(
                    {
                        "schema_version": (
                            "bernini-graft-phase-a-short-trajectory-owner-atlas-route-v1"
                        ),
                        "owner_digest": owner,
                        "atlas_digest": atlas,
                        "same_state_digest": state,
                        "route_digest": route_digests[sp_rank],
                        **false_authority(),
                    }
                )
                sp_rows = [
                    {
                        "sp_rank": item,
                        "metrics_digest": metrics["digest"],
                        "provenance_digest": provenances[item]["digest"],
                        "owner_digest": owner,
                        "atlas_digest": atlas,
                        "route_digest": route_digests[item],
                        "same_state_digest": state,
                    }
                    for item in range(4)
                ]
                manifest = sealed(
                    {
                        "schema_version": (
                            "bernini-graft-phase-a-short-trajectory-sp4-consensus-v1"
                        ),
                        "dp_arm": arm,
                        "stage": stage,
                        "schedule_index": index,
                        "ordered_local_evidence": sp_rows,
                        "metrics_owner_atlas_same_state_equal_across_sp4": True,
                        "rank_local_route_digests_preserved": True,
                    }
                )
                record = {
                    "stage": stage,
                    "schedule_index": index,
                    "dp_arm": arm,
                    "confirmation_iid": trajectory.CONFIRMATION_IID_BY_DP_ARM[arm],
                    "wrong_owner_iid": trajectory.FIT_IID_BY_DP_ARM[arm],
                    "parameter_digest": parameter,
                    "metrics_digest": metrics["digest"],
                    "sp4_evidence_manifest_digest": manifest["digest"],
                }
                cells.append(
                    sealed(
                        {
                            "schema_version": (
                                "bernini-graft-phase-a-short-trajectory-cell-v1"
                            ),
                            **record,
                            "metrics": metrics,
                            "provenance": provenances[sp_rank],
                            "owner_atlas_route": digests,
                            "sp4_evidence_manifest": manifest,
                            "sp4_consensus_digest": trajectory.object_sha256(record),
                            "sp4_exact_consensus": True,
                            "same_noise_within_correct_wrong_drop": True,
                            "original_thresholds_used_without_change": True,
                            "optimizer_update_performed_by_measurement": False,
                            "checkpoint_written": False,
                            **false_authority(),
                        }
                    )
                )
            matrices_by_rank[rank].append(
                sealed(
                    {
                        "schema_version": trajectory.TRAJECTORY_MATRIX_SCHEMA_VERSION,
                        "stage": stage,
                        "parameter_digest_before": parameter,
                        "parameter_digest_after": parameter,
                        "parameter_bytes_unchanged_by_measurement": True,
                        "schedule_indices": list(trajectory.CONFIRMATION_INDICES),
                        "cells": cells,
                        "all_cells_sp4_exact_consensus": True,
                        "checkpoint_written": False,
                        **false_authority(),
                    }
                )
            )

    gate_packets = []
    for rank, matrices in enumerate(matrices_by_rank):
        arm, sp_rank = divmod(rank, 4)
        vector = {
            str(cell["schedule_index"]): cell["metrics"][
                "noncompensating_all_pass"
            ]
            for cell in matrices[-1]["cells"]
        }
        metric_digests = {
            str(cell["schedule_index"]): cell["metrics"]["digest"]
            for cell in matrices[-1]["cells"]
        }
        gate_packets.append(
            sealed(
                {
                    "schema_version": trajectory.TERMINAL_GATE_PACKET_SCHEMA_VERSION,
                    "global_rank": rank,
                    "dp_arm": arm,
                    "sp_rank": sp_rank,
                    "family": trajectory.FAMILY_BY_DP_ARM[arm],
                    "final_gate_vector": vector,
                    "final_metrics_digests": metric_digests,
                    "local_gate_pass": all(vector.values()),
                    "gate_observed_before_original_admission": True,
                    "checkpoint_written": False,
                    "publication_performed": False,
                    **false_authority(),
                }
            )
        )
    gate_manifest = dict(
        trajectory._admit_world8_terminal_gate_packets(gate_packets)
    )
    world_no_go = gate_manifest["world_any_no_go"]
    initial = matrices_by_rank[0][0]["parameter_digest_before"]
    terminal = (
        initial
        if world_no_go
        else matrices_by_rank[0][-1]["parameter_digest_before"]
    )
    failures: list[dict | None] = []
    successes: list[dict | None] = []
    paths = []
    for rank, matrices in enumerate(matrices_by_rank):
        arm, sp_rank = divmod(rank, 4)
        local_pass = arm_passes[arm]
        if world_no_go:
            path = (
                "local_gate_pass_global_no_go_rolled_back"
                if local_pass
                else "local_gate_fail_global_no_go_rolled_back"
            )
            failed_metrics = None if local_pass else matrices[-1]["cells"][0]["metrics"]
            failure_reason = (
                "_World8ScientificNoGoRollback:"
                + trajectory._GLOBAL_NO_GO_ROLLBACK_SENTINEL
                if local_pass
                else (
                    "GraftPhaseAShortTrainingError:confirmation index 29 "
                    "failed a noncompensating gate"
                )
            )
            failures.append(
                sealed(
                    {
                        "schema_version": (
                            trajectory.short_trainer.FAILURE_SCHEMA_VERSION
                        ),
                        "status": "failed_rolled_back_no_checkpoint",
                        "rank": rank,
                        "dp_arm": arm,
                        "sp_rank": sp_rank,
                        "completed_optimizer_steps_before_failure": 2,
                        "failure_reason": failure_reason,
                        "failed_confirmation_metrics": failed_metrics,
                        "trainable_parameters_restored_to_initial_snapshot": True,
                        "checkpoint_written": False,
                        "checkpoint_payload_returned": False,
                        "publication_performed": False,
                        **false_authority(),
                    }
                )
            )
            successes.append(None)
        else:
            path = "local_gate_pass_world_all_pass_finished"
            failures.append(None)
            successes.append(
                sealed(
                    {
                        "schema_version": trajectory.short_trainer.SCHEMA_VERSION,
                        "status": "completed_in_memory_orchestration",
                        "topology": {"rank": rank, "dp_arm": arm, "sp_rank": sp_rank},
                        "checkpoint_written": False,
                        "checkpoint_payload_returned": False,
                        "publication_performed": False,
                        **false_authority(),
                    }
                )
            )
        paths.append(path)

    rollback_manifest = None
    if world_no_go:
        rollback_packets = []
        for rank, failure in enumerate(failures):
            arm, sp_rank = divmod(rank, 4)
            rollback_packets.append(
                sealed(
                    {
                        "schema_version": (
                            trajectory.TERMINAL_ROLLBACK_PACKET_SCHEMA_VERSION
                        ),
                        "global_rank": rank,
                        "dp_arm": arm,
                        "sp_rank": sp_rank,
                        "family": trajectory.FAMILY_BY_DP_ARM[arm],
                        "local_gate_pass": arm_passes[arm],
                        "world_any_no_go": True,
                        "local_terminal_path": paths[rank],
                        "initial_parameter_digest": initial,
                        "terminal_parameter_digest": terminal,
                        "trainer_failure_receipt_digest": failure["digest"],
                        "session_phase_after_rollback": "failed",
                        "finish_called": False,
                        "checkpoint_written": False,
                        "publication_performed": False,
                        **false_authority(),
                    }
                )
            )
        rollback_manifest = dict(
            trajectory._admit_world8_terminal_rollback_packets(
                rollback_packets, gate_manifest=gate_manifest
            )
        )

    packets = []
    for rank, matrices in enumerate(matrices_by_rank):
        arm, sp_rank = divmod(rank, 4)
        local_pass = arm_passes[arm]
        final_vector = dict(gate_packets[rank]["final_gate_vector"])
        final_metrics_digests = dict(gate_packets[rank]["final_metrics_digests"])
        admitted_indices = list(trajectory.CONFIRMATION_INDICES)
        if world_no_go and not local_pass:
            admitted_indices = []
        admissions = []
        for index in admitted_indices:
            metrics = matrices[-1]["cells"][
                list(trajectory.CONFIRMATION_INDICES).index(index)
            ]["metrics"]
            record = {
                "row_iid": trajectory.CONFIRMATION_IID_BY_DP_ARM[arm],
                "wrong_owner_iid": trajectory.FIT_IID_BY_DP_ARM[arm],
                "schedule_index": index,
                "metrics_digest": metrics["digest"],
                "parameter_digest": trajectory.object_sha256(
                    {"admission-parameter": rank}
                ),
                "base_digest": trajectory.object_sha256({"base": rank}),
                "optimizer_digest": trajectory.object_sha256({"optimizer": rank}),
            }
            admissions.append(
                sealed(
                    {
                        "schema_version": (
                            "bernini-graft-phase-a-confirmation-field-admission-v1"
                        ),
                        **record,
                        "metrics": metrics,
                        "sp4_consensus_digest": trajectory.object_sha256(record),
                        "checkpoint_written": False,
                        **false_authority(),
                    }
                )
            )
        status = (
            (
                "completed_local_gate_pass_global_no_go_rolled_back"
                if local_pass
                else "completed_local_gate_fail_global_no_go_rolled_back"
            )
            if world_no_go
            else "completed_original_confirmation_pass_no_checkpoint"
        )
        update_routes = [
            sealed(
                {
                    "schema_version": "bernini-graft-phase-a-short-update-route-v1",
                    "update_number": ordinal + 1,
                    "schedule_index": trajectory.UPDATE_INDICES[ordinal],
                    "row_iid": trajectory.FIT_IID_BY_DP_ARM[arm],
                    "fit_row_only": True,
                    "exact_four_native_forwards": True,
                    "forward_order": [
                        ["measurement", "negative"],
                        ["measurement", "positive"],
                        ["replay", "negative"],
                        ["replay", "positive"],
                    ],
                    "fresh_atlas_per_forward": True,
                    "measurement_atlas_detached": True,
                    "replay_atlas_graph_bearing_only_on_target_owner": True,
                    "checkpoint_written": False,
                    **false_authority(),
                }
            )
            for ordinal in range(2)
        ]
        local = sealed(
            {
                "schema_version": trajectory.SCHEMA_VERSION,
                "status": status,
                "complete": True,
                "scientific_outcome": (
                    "NO_GO" if world_no_go else "ORIGINAL_GATE_PASS"
                ),
                "local_scientific_outcome": (
                    "LOCAL_GATE_PASS" if local_pass else "LOCAL_GATE_NO_GO"
                ),
                "topology": {
                    "world_size": 8,
                    "data_parallel_size": 2,
                    "sequence_parallel_size": 4,
                    "rank": rank,
                    "dp_arm": arm,
                    "sp_rank": sp_rank,
                    "family": trajectory.FAMILY_BY_DP_ARM[arm],
                },
                "single_preregistered_optimizer_arm": {},
                "source_routing": {
                    "fit_iid": trajectory.FIT_IID_BY_DP_ARM[arm],
                    "confirmation_iid": trajectory.CONFIRMATION_IID_BY_DP_ARM[arm],
                },
                "trajectory_stage_order": list(trajectory.TRAJECTORY_STAGES),
                "trajectory_schedule_indices": list(trajectory.CONFIRMATION_INDICES),
                "trajectory_matrices": matrices,
                "trajectory_cell_count": 6,
                "update_route_receipts": update_routes,
                "world8_terminal_coordination": gate_manifest,
                "world8_terminal_rollback_manifest": rollback_manifest,
                "original_confirmation_gate": {
                    "thresholds_unchanged": True,
                    "final_stage_gate_vector": final_vector,
                    "final_stage_metrics_digests": final_metrics_digests,
                    "all_pass": local_pass,
                    "local_all_pass": local_pass,
                    "world_all_pass": not world_no_go,
                    "global_no_go": world_no_go,
                    "local_terminal_path": paths[rank],
                    "finish_called": not world_no_go,
                    "admissions_before_terminal_outcome": admissions,
                    "trainer_failure_receipt": failures[rank],
                    "trainer_success_receipt": successes[rank],
                    "rollback_to_initial_trainables": world_no_go,
                },
                "initial_parameter_digest": initial,
                "terminal_parameter_digest": terminal,
                "terminal_equals_initial_after_no_go": world_no_go,
                "training_updates_executed_for_diagnostic": 2,
                "checkpoint_written": False,
                "publication_performed": False,
                **false_authority(),
            }
        )
        packets.append(
            {"global_rank": rank, "result_digest": local["digest"], "local_result": local}
        )
    return packets


class TrajectoryDiagnosticTests(unittest.TestCase):
    def test_preregistered_order_is_executed_in_official_function(self) -> None:
        source = inspect.getsource(trajectory.execute_authenticated_trajectory_diagnostic)
        initial = source.index('stage="initial"')
        update_loop = source.index("for update_number, schedule_index")
        stage_measure = source.index("stage=stage")
        original_gate = source.index("session.confirmation_plan()")
        self.assertLess(initial, update_loop)
        self.assertLess(update_loop, stage_measure)
        self.assertLess(stage_measure, original_gate)
        coordination = source.index("_coordinate_world8_terminal_gate(")
        admission = source.index("session.record_confirmation_fields(")
        self.assertLess(coordination, original_gate)
        self.assertLess(coordination, admission)
        self.assertIn("if not world_any_no_go:", source)
        self.assertIn("_coordinate_world8_terminal_rollback(", source)

    def test_world8_scientific_no_go_is_admitted(self) -> None:
        receipt = trajectory.assemble_trajectory_world8_results(make_packets())
        self.assertEqual(receipt["scientific_outcome"], "NO_GO")
        self.assertEqual(len(receipt["dog_human_trajectory_matrices"]), 2)
        self.assertFalse(receipt["training_authority"])

    def test_world8_pass_is_diagnostic_only(self) -> None:
        receipt = trajectory.assemble_trajectory_world8_results(
            make_packets(no_go=False)
        )
        self.assertEqual(receipt["scientific_outcome"], "ORIGINAL_GATE_PASS")
        self.assertFalse(receipt["scientific_success_claimed"])

    def test_all_terminal_outcome_families_share_one_world_protocol(self) -> None:
        cases = (
            ((True, True), "ORIGINAL_GATE_PASS"),
            ((False, False), "NO_GO"),
            ((True, False), "NO_GO"),
            ((False, True), "NO_GO"),
        )
        for arm_passes, outcome in cases:
            with self.subTest(arm_passes=arm_passes):
                receipt = trajectory.assemble_trajectory_world8_results(
                    make_packets(arm_passes=arm_passes)
                )
                self.assertEqual(receipt["scientific_outcome"], outcome)
                coordination = receipt["world8_terminal_coordination"]
                self.assertEqual(
                    coordination["terminal_collective_protocol"],
                    (
                        "all_pass_original_finish_all_ranks"
                        if all(arm_passes)
                        else "global_no_go_all_rank_rollback_no_finish"
                    ),
                )
                self.assertTrue(
                    receipt["terminal_collective_sequence_unified_across_world8"]
                )
                families = receipt["dog_human_trajectory_matrices"]
                self.assertEqual(
                    [row["local_scientific_outcome"] for row in families],
                    [
                        "LOCAL_GATE_PASS" if value else "LOCAL_GATE_NO_GO"
                        for value in arm_passes
                    ],
                )
                if not all(arm_passes):
                    rollback = receipt["world8_terminal_rollback_manifest"]
                    self.assertTrue(rollback["all_eight_skipped_finish"])
                    self.assertTrue(rollback["all_eight_terminal_equal_initial"])
                    self.assertEqual(
                        len(rollback["ordered_rollback_packets"]), 8
                    )

    def test_self_consistent_terminal_gate_manifest_tamper_fails(self) -> None:
        packets = make_packets(arm_passes=(True, False))
        original = packets[0]["local_result"]["world8_terminal_coordination"]
        manifest = dict(original)
        manifest.pop("digest")
        rows = list(manifest["ordered_gate_packets"])
        packet = dict(rows[4])
        packet.pop("digest")
        packet["local_gate_pass"] = True
        rows[4] = sealed(packet)
        manifest["ordered_gate_packets"] = rows
        forged = sealed(manifest)
        for rank in range(8):
            local = dict(packets[rank]["local_result"])
            local.pop("digest")
            local["world8_terminal_coordination"] = forged
            packets[rank]["local_result"] = sealed(local)
            packets[rank]["result_digest"] = packets[rank]["local_result"]["digest"]
        with self.assertRaises(trajectory.GraftPhaseAShortGPUError):
            trajectory.assemble_trajectory_world8_results(packets)

    def test_self_consistent_terminal_rollback_manifest_tamper_fails(self) -> None:
        packets = make_packets(arm_passes=(False, True))
        original = packets[0]["local_result"]["world8_terminal_rollback_manifest"]
        manifest = dict(original)
        manifest.pop("digest")
        rows = list(manifest["ordered_rollback_packets"])
        packet = dict(rows[0])
        packet.pop("digest")
        packet["terminal_parameter_digest"] = trajectory.object_sha256(
            {"forged-terminal": True}
        )
        rows[0] = sealed(packet)
        manifest["ordered_rollback_packets"] = rows
        forged = sealed(manifest)
        for rank in range(8):
            local = dict(packets[rank]["local_result"])
            local.pop("digest")
            local["world8_terminal_rollback_manifest"] = forged
            packets[rank]["local_result"] = sealed(local)
            packets[rank]["result_digest"] = packets[rank]["local_result"]["digest"]
        with self.assertRaises(trajectory.GraftPhaseAShortGPUError):
            trajectory.assemble_trajectory_world8_results(packets)

    def test_allpass_terminal_must_equal_after_update_38(self) -> None:
        packets = make_packets(arm_passes=(True, True))
        local = dict(packets[0]["local_result"])
        local.pop("digest")
        local["terminal_parameter_digest"] = trajectory.object_sha256(
            {"not-after-update-38": True}
        )
        packets[0]["local_result"] = sealed(local)
        packets[0]["result_digest"] = packets[0]["local_result"]["digest"]
        with self.assertRaises(trajectory.GraftPhaseAShortGPUError):
            trajectory.assemble_trajectory_world8_results(packets)

    def test_missing_stage_fails_closed(self) -> None:
        packets = make_packets()
        value = dict(packets[0]["local_result"])
        value.pop("digest")
        value["trajectory_matrices"] = value["trajectory_matrices"][:2]
        packets[0]["local_result"] = sealed(value)
        packets[0]["result_digest"] = packets[0]["local_result"]["digest"]
        with self.assertRaises(trajectory.GraftPhaseAShortGPUError):
            trajectory.assemble_trajectory_world8_results(packets)

    def test_elevated_authority_fails_closed(self) -> None:
        packets = make_packets()
        value = dict(packets[0]["local_result"])
        value.pop("digest")
        value["training_authority"] = True
        packets[0]["local_result"] = sealed(value)
        packets[0]["result_digest"] = packets[0]["local_result"]["digest"]
        with self.assertRaises(trajectory.GraftPhaseAShortGPUError):
            trajectory.assemble_trajectory_world8_results(packets)

    def test_self_consistent_after_digest_tamper_fails_closed(self) -> None:
        packets = make_packets()
        local = dict(packets[0]["local_result"])
        local.pop("digest")
        matrices = list(local["trajectory_matrices"])
        matrix = dict(matrices[0])
        matrix.pop("digest")
        matrix["parameter_digest_after"] = trajectory.object_sha256(
            {"tampered": True}
        )
        matrices[0] = sealed(matrix)
        local["trajectory_matrices"] = matrices
        packets[0]["local_result"] = sealed(local)
        packets[0]["result_digest"] = packets[0]["local_result"]["digest"]
        with self.assertRaises(trajectory.GraftPhaseAShortGPUError):
            trajectory.assemble_trajectory_world8_results(packets)

    def test_self_consistent_manifest_flag_tamper_fails_closed(self) -> None:
        packets = make_packets()
        local = dict(packets[0]["local_result"])
        local.pop("digest")
        matrices = list(local["trajectory_matrices"])
        matrix = dict(matrices[0])
        matrix.pop("digest")
        cells = list(matrix["cells"])
        cell = dict(cells[0])
        cell.pop("digest")
        manifest = dict(cell["sp4_evidence_manifest"])
        manifest.pop("digest")
        manifest["rank_local_route_digests_preserved"] = False
        manifest = sealed(manifest)
        cell["sp4_evidence_manifest"] = manifest
        cell["sp4_evidence_manifest_digest"] = manifest["digest"]
        consensus_record = {
            "stage": cell["stage"],
            "schedule_index": cell["schedule_index"],
            "dp_arm": cell["dp_arm"],
            "confirmation_iid": cell["confirmation_iid"],
            "wrong_owner_iid": cell["wrong_owner_iid"],
            "parameter_digest": cell["parameter_digest"],
            "metrics_digest": cell["metrics_digest"],
            "sp4_evidence_manifest_digest": manifest["digest"],
        }
        cell["sp4_consensus_digest"] = trajectory.object_sha256(consensus_record)
        cells[0] = sealed(cell)
        matrix["cells"] = cells
        matrices[0] = sealed(matrix)
        local["trajectory_matrices"] = matrices
        packets[0]["local_result"] = sealed(local)
        packets[0]["result_digest"] = packets[0]["local_result"]["digest"]
        with self.assertRaises(trajectory.GraftPhaseAShortGPUError):
            trajectory.assemble_trajectory_world8_results(packets)

    def test_plan_and_launchers_are_nonhold_fresh_world8(self) -> None:
        plan = json.loads(
            (ROOT / "assets/graft_phase_a_short_trajectory_world8_plan_v1.json").read_text()
        )
        self.assertEqual(plan["trajectory_stages"], list(trajectory.TRAJECTORY_STAGES))
        self.assertEqual(plan["measurement_indices"], [29, 38])
        self.assertFalse(plan["optimizer_arm"]["lr_or_step_sweep"])
        launcher = (
            ROOT / "scripts/auh_run_graft_phase_a_short_trajectory_world8_v1.sbatch"
        ).read_text()
        submitter = (
            ROOT / "scripts/auh_submit_graft_phase_a_short_trajectory_world8_v1.sh"
        ).read_text()
        self.assertIn("#SBATCH --gres=gpu:mi210:8", launcher)
        self.assertIn("#SBATCH --time=24:00:00", launcher)
        self.assertNotIn("#SBATCH --hold", launcher)
        self.assertNotIn("--dependency", submitter)
        self.assertIn('"hold":False,"dependency":None', submitter)
        self.assertIn("failure.receipt.json", launcher)

    def test_submitter_reserves_one_inode_before_only_sbatch(self) -> None:
        submitter = (
            ROOT / "scripts/auh_submit_graft_phase_a_short_trajectory_world8_v1.sh"
        ).read_text()
        reservation = submitter.index(
            "fd=os.open(receipt_path,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)"
        )
        sbatch = submitter.index("completed=subprocess.run(argv")
        finalize = submitter.index("os.fchmod(fd,0o444)")
        self.assertLess(reservation, sbatch)
        self.assertLess(sbatch, finalize)
        self.assertEqual(submitter.count("completed=subprocess.run(argv"), 1)
        self.assertIn("same_inode_retained_across_sbatch", submitter)
        self.assertNotIn("--dependency", submitter)
        self.assertNotIn('"--hold"', submitter)

    def test_submitter_failed_sbatch_leaves_one_0600_non_success_reservation(self) -> None:
        self._exercise_submitter_hostile(sbatch_success=False)

    def test_submitter_success_finalizes_same_reservation_0444(self) -> None:
        self._exercise_submitter_hostile(sbatch_success=True)

    def _exercise_submitter_hostile(self, *, sbatch_success: bool) -> None:
        submitter = (
            ROOT / "scripts/auh_submit_graft_phase_a_short_trajectory_world8_v1.sh"
        ).read_text()
        with tempfile.TemporaryDirectory(prefix="graft-traj-submitter-") as raw:
            root = Path(raw)
            marker = root / "sbatch.calls"
            fake_sbatch = root / "sbatch"
            fake_sbatch.write_text(
                "#!/bin/bash\nprintf x >> "
                + str(marker)
                + ("\nprintf '246810;cluster\\n'\nexit 0\n" if sbatch_success else "\nexit 17\n")
            )
            fake_sbatch.chmod(0o755)
            transformed = root / "submit.sh"
            transformed.write_text(
                submitter.replace("/usr/bin/sbatch", str(fake_sbatch))
            )
            transformed.chmod(0o755)
            inputs = {}
            for name in ("archive", "closure", "checkpoint_manifest", "plan", "terminal", "launcher"):
                path = root / name
                path.write_bytes((name + "\n").encode("ascii"))
                inputs[name] = path
            python_bin = Path(sys.executable).resolve(strict=True)
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            output = root / "runs" / "fresh"
            output.parent.mkdir()
            environment = {
                "GRAFT_TRAJ_SOURCE_ARCHIVE": str(inputs["archive"]),
                "GRAFT_TRAJ_SOURCE_ARCHIVE_SHA256": sha(inputs["archive"]),
                "GRAFT_TRAJ_RUNTIME_CLOSURE": str(inputs["closure"]),
                "GRAFT_TRAJ_RUNTIME_CLOSURE_SHA256": sha(inputs["closure"]),
                "GRAFT_TRAJ_PYTHON_BIN": str(python_bin),
                "GRAFT_TRAJ_PYTHON_SHA256": sha(python_bin),
                "BERNINI_OFFICIAL_ROOT": str(root / "bernini"),
                "BERNINI_VEOMNI_ROOT": str(root / "veomni"),
                "BERNINI_ACTION_CHECKPOINT": str(root / "checkpoint"),
                "BERNINI_CHECKPOINT_CONTENT_MANIFEST": str(inputs["checkpoint_manifest"]),
                "BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256": sha(inputs["checkpoint_manifest"]),
                "GRAFT_TRAJ_PLAN": str(inputs["plan"]),
                "GRAFT_TRAJ_PLAN_SHA256": sha(inputs["plan"]),
                "GRAFT_TRAJ_TERMINAL_ADMISSION": str(inputs["terminal"]),
                "GRAFT_TRAJ_TERMINAL_ADMISSION_SHA256": sha(inputs["terminal"]),
                "GRAFT_TRAJ_TERMINAL_MATERIALIZER_RUNTIME_SHA256": "1" * 64,
                "GRAFT_TRAJ_OUTPUT_ROOT": str(output),
                "GRAFT_TRAJ_LAUNCHER_SOURCE": str(inputs["launcher"]),
                "GRAFT_TRAJ_LAUNCHER_SHA256": sha(inputs["launcher"]),
                "GRAFT_TRAJ_RUNNER_SHA256": "2" * 64,
            }
            completed = subprocess.run(
                ["/usr/bin/bash", "-p", str(transformed)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C", **environment},
                check=False,
                timeout=30,
            )
            receipt = Path(str(output) + ".submission.receipt.json")
            self.assertTrue(receipt.is_file())
            self.assertEqual(marker.read_text(), "x")
            if sbatch_success:
                self.assertEqual(completed.returncode, 0, completed.stderr.decode())
                self.assertEqual(stat.S_IMODE(receipt.stat().st_mode), 0o444)
                value = json.loads(receipt.read_text())
                self.assertEqual(value["submitted_job"]["job_id"], "246810")
                self.assertIsNone(value["request"]["dependency"])
                self.assertFalse(value["request"]["hold"])
                self.assertTrue(value["outputs"]["same_inode_retained_across_sbatch"])
            else:
                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(stat.S_IMODE(receipt.stat().st_mode), 0o600)
                self.assertEqual(receipt.stat().st_size, 0)
                retry = subprocess.run(
                    ["/usr/bin/bash", "-p", str(transformed)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C", **environment},
                    check=False,
                    timeout=30,
                )
                self.assertNotEqual(retry.returncode, 0)
                self.assertEqual(marker.read_text(), "x")


if __name__ == "__main__":
    unittest.main()
