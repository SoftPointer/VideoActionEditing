#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import inspect
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
for root in (METHOD_ROOT, TEST_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import graft_source_conditioned_proposal_selector_v1 as selector  # noqa: E402
import test_graft_action_first_source_guided_aggregation_v1 as asga_test  # noqa: E402


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _seal(payload: dict) -> dict:
    owned = copy.deepcopy(payload)
    owned.pop("receipt_digest", None)
    owned["receipt_digest"] = selector.object_sha256(owned)
    return owned


def _entry(receipt: dict) -> dict:
    raw = selector.canonical_json_bytes(receipt)
    return {"canonical_json": raw.decode("ascii"), "sha256": hashlib.sha256(raw).hexdigest()}


def _entry_sha(entry: dict) -> str:
    return entry["sha256"]


def _calibration(threshold: float = 0.05) -> dict:
    registry = []
    for axis_index, axis_name in enumerate(selector.RAW_AXIS_NAMES):
        registry.append(
            {
                "axis_index": axis_index,
                "axis_name": axis_name,
                "score_direction": selector.RAW_AXIS_DIRECTIONS[axis_index],
                "score_domain": "closed_unit_interval",
                "gate_name": selector.GATE_NAMES[axis_index],
                "gate_direction": "strictly_greater_than_threshold",
                "threshold_exact_fp32": dict(selector.fp32_encoding(threshold)),
                "producer_code_sha256": _sha(f"producer-{axis_name}"),
                "model_artifact_sha256": _sha(f"model-{axis_name}"),
                "model_config_sha256": _sha(f"config-{axis_name}"),
                "evaluator_runtime_sha256": _sha(f"runtime-{axis_name}"),
                "preprocess_config_sha256": _sha(f"preprocess-{axis_name}"),
                "temporal_exact81_protocol_sha256": _sha(f"temporal81-{axis_name}"),
                "temporal_frame_count": selector.FRAME_COUNT,
                "score_aggregation_sha256": _sha(f"aggregation-{axis_name}"),
                "prompt_counterfactual_sha256": _sha(f"prompt-counterfactual-{axis_name}"),
            }
        )
    return _seal(
        {
            "schema_version": selector.CALIBRATION_SCHEMA_VERSION,
            "selector_method": selector.METHOD,
            "raw_axis_names": list(selector.RAW_AXIS_NAMES),
            "gate_names": list(selector.GATE_NAMES),
            "axis_registry": registry,
            "calibration_dataset_receipt_sha256": _sha("calibration-dataset"),
            "calibration_run_receipt_sha256": _sha("calibration-run"),
            "frozen_before_trial_execution": True,
            "disjoint_from_trial_sources": True,
            "trial_source_read": False,
            "proposal_bank_read": False,
            "target_video_read": False,
            "proposal_media_read": False,
            "mask_pose_flow_track_read": False,
            "semantic_correctness_authority": False,
            "optimizer_authority": False,
            "same_process_security_boundary": False,
        }
    )


def _intervention(bank, kind: str = "baseline_source_and_retelling") -> dict:
    effective = bank.retelling.digest
    visual, retelling, wrong = True, True, False
    if kind == "wrong_retelling":
        effective, wrong = _sha("intentionally-wrong-retelling"), True
    elif kind == "drop_visual":
        visual = False
    elif kind == "drop_retelling":
        effective, retelling = selector.ABSENT_RETELLING_DIGEST, False
    return _seal(
        {
            "schema_version": selector.INTERVENTION_SCHEMA_VERSION,
            "kind": kind,
            "proposal_bank_digest": bank.provenance.digest,
            "original_source_video_sha256": bank.retelling.source_video_sha256,
            "original_retelling_digest": bank.retelling.digest,
            "effective_retelling_digest": effective,
            "source_visual_condition_present": visual,
            "source_retelling_condition_present": retelling,
            "wrong_retelling_intentional": wrong,
            "semantic_interpretation_authority": False,
            "optimizer_authority": False,
        }
    )


def _observations() -> list[list[float]]:
    return [
        [0.75, 0.20, 0.20, 0.20, 0.65, 0.70, 0.70, 0.70, 0.70, 0.70],
        [0.76, 0.20, 0.20, 0.20, 0.66, 0.71, 0.71, 0.71, 0.71, 0.71],
        [0.90, 0.10, 0.10, 0.10, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90],
        [0.78, 0.20, 0.20, 0.20, 0.68, 0.73, 0.73, 0.73, 0.73, 0.73],
        [0.77, 0.20, 0.20, 0.20, 0.67, 0.72, 0.72, 0.72, 0.72, 0.72],
    ]


def _trial_receipt(bank, index: int, intervention_entry: dict, calibration_entry: dict) -> dict:
    intervention = _receipt(intervention_entry)
    return _seal(
        {
            "schema_version": selector.TRIAL_SCHEMA_VERSION,
            "selector_method": selector.METHOD,
            "execution_kind": "trial_candidate",
            "execution_id": _sha(f"trial-execution-id-{index}"),
            "candidate_index": index,
            "proposal_bank_digest": bank.provenance.digest,
            "source_video_sha256": bank.retelling.source_video_sha256,
            "original_retelling_digest": bank.retelling.digest,
            "effective_retelling_digest": intervention["effective_retelling_digest"],
            "intervention_receipt_sha256": _entry_sha(intervention_entry),
            "calibration_receipt_sha256": _entry_sha(calibration_entry),
            "instruction_sha256": bank.retelling.instruction_sha256,
            "candidate_gaussian_raw_sha256": bank.provenance.branch_gaussian_raw_sha256s[index][0],
            "schedule_digest": bank.provenance.branch_schedule_digests[index][0],
            "program_slice_sha256": bank.provenance.candidate_slice_sha256s[index],
            "counterfactual_execution_receipt_sha256s": list(
                bank.provenance.branch_execution_receipt_sha256s[index]
            ),
            "matched_runtime_config_digest": _sha("one-matched-runtime"),
            "frame_count": selector.FRAME_COUNT,
            "output_artifact_sha256": _sha(f"trial-output-{index}"),
            "output_artifact_byte_size": selector.FRAME_COUNT * 8 * 8 * 3,
            "output_artifact_shape": [selector.FRAME_COUNT, 8, 8, 3],
            "output_artifact_dtype": "uint8",
            "output_artifact_layout": "THWC_RGB",
            "frame81_digest": _sha(f"trial-frame81-{index}"),
            "source_visual_condition_present": intervention["source_visual_condition_present"],
            "source_retelling_condition_present": intervention["source_retelling_condition_present"],
            "proposal_rgb_read_by_selector": False,
            "proposal_latent_read_by_selector": False,
            "raw_velocity_read_by_selector": False,
            "target_video_read_by_selector": False,
            "mask_pose_flow_track_read_by_selector": False,
            "semantic_correctness_authority": False,
            "optimizer_authority": False,
            "same_process_security_boundary": False,
        }
    )


def _post_receipt(bank, selected: int, trial_entry: dict, intervention_entry: dict, calibration_entry: dict) -> dict:
    intervention = _receipt(intervention_entry)
    return _seal(
        {
            "schema_version": selector.POST_COMMIT_SCHEMA_VERSION,
            "selector_method": selector.METHOD,
            "execution_kind": "post_commit_selected_candidate",
            "execution_id": _sha(f"independent-post-execution-id-{selected}"),
            "proposal_bank_digest": bank.provenance.digest,
            "selected_candidate_index": selected,
            "selected_program_slice_sha256": bank.provenance.candidate_slice_sha256s[selected],
            "selected_trial_execution_receipt_sha256": _entry_sha(trial_entry),
            "source_video_sha256": bank.retelling.source_video_sha256,
            "original_retelling_digest": bank.retelling.digest,
            "effective_retelling_digest": intervention["effective_retelling_digest"],
            "intervention_receipt_sha256": _entry_sha(intervention_entry),
            "calibration_receipt_sha256": _entry_sha(calibration_entry),
            "instruction_sha256": bank.retelling.instruction_sha256,
            "gaussian_raw_sha256": bank.provenance.branch_gaussian_raw_sha256s[selected][0],
            "schedule_digest": bank.provenance.branch_schedule_digests[selected][0],
            "matched_runtime_config_digest": _sha("one-matched-runtime"),
            "frame_count": selector.FRAME_COUNT,
            "output_artifact_sha256": _sha(f"post-output-{selected}"),
            "output_artifact_byte_size": selector.FRAME_COUNT * 8 * 8 * 3,
            "output_artifact_shape": [selector.FRAME_COUNT, 8, 8, 3],
            "output_artifact_dtype": "uint8",
            "output_artifact_layout": "THWC_RGB",
            "frame81_digest": _sha(f"post-frame81-{selected}"),
            "program_executed_without_mutation": True,
            "source_visual_condition_present": intervention["source_visual_condition_present"],
            "source_retelling_condition_present": intervention["source_retelling_condition_present"],
            "proposal_rgb_read_by_selector": False,
            "proposal_latent_read_by_selector": False,
            "raw_velocity_read_by_selector": False,
            "target_video_read_by_selector": False,
            "mask_pose_flow_track_read_by_selector": False,
            "semantic_correctness_authority": False,
            "optimizer_authority": False,
            "same_process_security_boundary": False,
        }
    )


def _axis_entries(
    stage: str,
    candidate: int,
    execution_entry: dict,
    values: list[float],
    axis_registry: list[dict],
) -> list[dict]:
    entries = []
    execution = _receipt(execution_entry)
    for axis_index, (axis_name, value) in enumerate(zip(selector.RAW_AXIS_NAMES, values)):
        registry = axis_registry[axis_index]
        receipt = _seal(
            {
                "schema_version": selector.EVALUATOR_SCHEMA_VERSION,
                "selector_method": selector.METHOD,
                "evaluation_stage": stage,
                "candidate_index": candidate,
                "axis_index": axis_index,
                "axis_name": axis_name,
                "score_direction": registry["score_direction"],
                "score_domain": registry["score_domain"],
                "gate_name": registry["gate_name"],
                "gate_direction": registry["gate_direction"],
                "threshold_exact_fp32": registry["threshold_exact_fp32"],
                "execution_receipt_sha256": _entry_sha(execution_entry),
                "execution_id": execution["execution_id"],
                "output_artifact_sha256": execution["output_artifact_sha256"],
                "output_artifact_frame81_digest": execution["frame81_digest"],
                "axis_registry_entry_sha256": selector.object_sha256(registry),
                "producer_code_sha256": registry["producer_code_sha256"],
                "model_artifact_sha256": registry["model_artifact_sha256"],
                "model_config_sha256": registry["model_config_sha256"],
                "evaluator_runtime_sha256": registry["evaluator_runtime_sha256"],
                "preprocess_config_sha256": registry["preprocess_config_sha256"],
                "temporal_exact81_protocol_sha256": registry["temporal_exact81_protocol_sha256"],
                "temporal_frame_count": registry["temporal_frame_count"],
                "score_aggregation_sha256": registry["score_aggregation_sha256"],
                "prompt_counterfactual_sha256": registry["prompt_counterfactual_sha256"],
                "value_exact_fp32": dict(selector.fp32_encoding(value)),
                "frame_count": selector.FRAME_COUNT,
                "semantic_correctness_authority": False,
                "optimizer_authority": False,
            }
        )
        entries.append(_entry(receipt))
    return entries


def _receipt(entry: dict) -> dict:
    import json
    return json.loads(entry["canonical_json"])


def _build_envelope(*, bank=None, kind="baseline_source_and_retelling", observations=None) -> tuple:
    if bank is None:
        bank = asga_test._bank()
    if observations is None:
        observations = _observations()
    calibration_entry = _entry(_calibration())
    axis_registry = _receipt(calibration_entry)["axis_registry"]
    intervention_entry = _entry(_intervention(bank, kind))
    trials = []
    trial_receipt_entries = []
    for index in range(selector.CANDIDATE_COUNT):
        execution_entry = _entry(_trial_receipt(bank, index, intervention_entry, calibration_entry))
        trial_receipt_entries.append(execution_entry)
        trials.append(
            {
                "execution_receipt": execution_entry,
                "evaluator_receipts": _axis_entries(
                    "trial", index, execution_entry, observations[index], axis_registry
                ),
            }
        )
    # Helper inputs select candidate 2 unless a test deliberately changes them.
    gate_rows = torch.tensor(observations, dtype=torch.float32)
    action = gate_rows[:, 0]
    gate_rows = torch.stack(
        (action, action-gate_rows[:, 1], action-gate_rows[:, 2], action-gate_rows[:, 3],
         gate_rows[:, 4], gate_rows[:, 5], gate_rows[:, 6], gate_rows[:, 7], gate_rows[:, 8], gate_rows[:, 9]),
        dim=1,
    )
    threshold = torch.tensor(0.05, dtype=torch.float32)
    feasible = [i for i in range(5) if bool(torch.all(gate_rows[i] - threshold > 0).item())]
    # Mirror fixed rule sufficiently for fixtures: standard cases are strict
    # dominance; exact-tie cases intentionally choose the lower index.
    selected = None if not feasible else max(
        feasible,
        key=lambda i: (
            float(torch.min(gate_rows[i]-threshold)),
            tuple((gate_rows[i]-threshold).tolist()),
            -i,
        ),
    )
    post = None
    if selected is not None:
        post_execution_entry = _entry(
            _post_receipt(
                bank, selected, trial_receipt_entries[selected], intervention_entry, calibration_entry
            )
        )
        post = {
            "execution_receipt": post_execution_entry,
            "evaluator_receipts": _axis_entries(
                "post", selected, post_execution_entry, observations[selected], axis_registry
            ),
        }
    envelope = _seal(
        {
            "schema_version": selector.ENVELOPE_SCHEMA_VERSION,
            "selector_method": selector.METHOD,
            "proposal_bank_digest": bank.provenance.digest,
            "calibration_receipt": calibration_entry,
            "intervention_receipt": intervention_entry,
            "trial_executions": trials,
            "post_commit_execution": post,
            "proposal_rgb_present": False,
            "proposal_latent_present": False,
            "raw_velocity_present": False,
            "target_video_present": False,
            "mask_pose_flow_track_present": False,
            "semantic_correctness_authority": False,
            "optimizer_authority": False,
            "same_process_security_boundary": False,
        }
    )
    return envelope, bank


def _raw(envelope: dict) -> bytes:
    return selector.canonical_json_bytes(envelope)


def _reseal_entry(receipt: dict) -> dict:
    return _entry(_seal(receipt))


def _run_file(envelope: dict, bank):
    raw = _raw(envelope)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sealed-envelope.json"
        path.write_bytes(raw)
        path.chmod(0o444)
        result = selector.release_source_conditioned_proposal_from_file(
            bank, path.resolve(), expected_file_sha256=hashlib.sha256(raw).hexdigest()
        )
    return result


class SourceConditionedProposalSelectorTests(unittest.TestCase):
    def test_canonical_bytes_are_structure_only_and_never_compute_selection(self) -> None:
        envelope, bank = _build_envelope()
        result = selector.inspect_self_attested_execution_envelope(_raw(envelope))
        payload = result.payload()
        self.assertIsInstance(result, selector.SelfAttestedEnvelopeInspection)
        self.assertFalse(hasattr(result, "selected_program"))
        self.assertEqual(payload["trial_execution_count"], 5)
        self.assertEqual(payload["trial_axis_receipt_counts"], [10] * 5)
        self.assertEqual(payload["post_axis_receipt_count"], 10)
        self.assertFalse(payload["score_values_parsed"])
        self.assertFalse(payload["score_derived_outcome_computed"])
        forbidden = {
            "feasible_candidate_indices", "pareto_frontier_candidate_indices",
            "observed_candidate_index", "accepted_candidate_index",
            "selected_program_slice_sha256", "gaussian_raw_sha256", "schedule_digest",
        }
        self.assertTrue(forbidden.isdisjoint(payload))
        with self.assertRaisesRegex(selector.GraftSelectorError, "sealed-loader"):
            selector.select_source_conditioned_proposal(bank, _raw(envelope))

    def test_self_attested_inspection_never_enters_score_parser_or_choice(self) -> None:
        envelope, bank = _build_envelope()
        # Even structurally present evaluator payloads are opaque to this path.
        opaque = copy.deepcopy(envelope)
        opaque["trial_executions"][0]["evaluator_receipts"][0]["canonical_json"] = "opaque"
        opaque["trial_executions"][0]["evaluator_receipts"][0]["sha256"] = _sha("opaque")
        opaque = _seal(opaque)
        with mock.patch.object(selector, "_decode_fp32", side_effect=AssertionError("score parsed")), mock.patch.object(
            selector, "_discrete_choice", side_effect=AssertionError("choice executed")
        ):
            inspection = selector.inspect_self_attested_execution_envelope(_raw(opaque))
        payload = inspection.payload()
        self.assertFalse(payload["embedded_receipt_payloads_parsed"])
        self.assertFalse(payload["score_derived_outcome_computed"])

    def test_pinned_plain0444_loader_releases_exact_discrete_bank_slice(self) -> None:
        envelope, bank = _build_envelope()
        result = _run_file(envelope, bank)
        payload = result.provenance.payload()
        self.assertEqual(payload["accepted_candidate_index"], 2)
        self.assertEqual(payload["pareto_frontier_candidate_indices"], [2])
        self.assertTrue(torch.equal(result.selected_program, bank.tensor[2]))
        self.assertTrue(payload["trial_observations_recomputed_from_axis_receipts"])
        self.assertTrue(payload["post_observation_recomputed_from_axis_receipts"])
        self.assertFalse(payload["program_aggregation_used"])
        self.assertFalse(payload["optimizer_authority"])

    def test_resign_all_cannot_cross_external_file_sha_anchor(self) -> None:
        envelope, bank = _build_envelope()
        original = _raw(envelope)
        forged = copy.deepcopy(envelope)
        # This is an otherwise schema-valid score forgery: the attacker changes
        # one post evaluator value, recomputes that receipt's self digest and
        # byte SHA, then recomputes the outer self digest.  None of those
        # caller-controlled hashes can replace the external frozen file SHA.
        evaluator = _receipt(
            forged["post_commit_execution"]["evaluator_receipts"][9]
        )
        evaluator["value_exact_fp32"] = dict(selector.fp32_encoding(0.91))
        forged["post_commit_execution"]["evaluator_receipts"][9] = _reseal_entry(
            evaluator
        )
        forged = _seal(forged)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sealed.json"
            path.write_bytes(_raw(forged))
            path.chmod(0o444)
            with self.assertRaisesRegex(selector.GraftSelectorError, "external pinned file SHA"):
                selector.release_source_conditioned_proposal_from_file(
                    bank, path.resolve(), expected_file_sha256=hashlib.sha256(original).hexdigest()
                )

    def test_opaque_sha_swap_is_rejected_from_actual_canonical_bytes(self) -> None:
        envelope, bank = _build_envelope()
        forged = copy.deepcopy(envelope)
        forged["post_commit_execution"]["execution_receipt"]["sha256"] = _sha("opaque-swap")
        forged = _seal(forged)
        with self.assertRaisesRegex(selector.GraftSelectorError, "canonical byte hash differs"):
            _run_file(forged, bank)

    def test_trial_receipt_cannot_be_replayed_as_post_execution(self) -> None:
        envelope, bank = _build_envelope()
        forged = copy.deepcopy(envelope)
        forged["post_commit_execution"]["execution_receipt"] = copy.deepcopy(
            forged["trial_executions"][2]["execution_receipt"]
        )
        forged = _seal(forged)
        with self.assertRaisesRegex(selector.GraftSelectorError, "exact81 post execution receipt keys differ"):
            _run_file(forged, bank)

    def test_missing_and_swapped_evaluator_receipts_fail_closed(self) -> None:
        envelope, bank = _build_envelope()
        missing = copy.deepcopy(envelope)
        missing["trial_executions"][0]["evaluator_receipts"].pop()
        missing = _seal(missing)
        with self.assertRaisesRegex(selector.GraftSelectorError, "exactly ten"):
            _run_file(missing, bank)

        swapped = copy.deepcopy(envelope)
        receipts = swapped["post_commit_execution"]["evaluator_receipts"]
        receipts[0], receipts[1] = receipts[1], receipts[0]
        swapped = _seal(swapped)
        with self.assertRaisesRegex(selector.GraftSelectorError, "axis_index"):
            _run_file(swapped, bank)

    def test_trial_evaluator_receipts_cannot_be_reused_for_post(self) -> None:
        envelope, bank = _build_envelope()
        forged = copy.deepcopy(envelope)
        forged["post_commit_execution"]["evaluator_receipts"] = copy.deepcopy(
            forged["trial_executions"][2]["evaluator_receipts"]
        )
        forged = _seal(forged)
        with self.assertRaisesRegex(selector.GraftSelectorError, "evaluation_stage"):
            _run_file(forged, bank)

    def test_every_axis_must_exactly_match_frozen_calibration_registry(self) -> None:
        envelope, bank = _build_envelope()
        forged = copy.deepcopy(envelope)
        evaluator = _receipt(forged["trial_executions"][3]["evaluator_receipts"][6])
        evaluator["producer_code_sha256"] = _sha("attacker-replacement-producer")
        forged["trial_executions"][3]["evaluator_receipts"][6] = _reseal_entry(evaluator)
        forged = _seal(forged)
        with self.assertRaisesRegex(selector.GraftSelectorError, "producer_code_sha256"):
            _run_file(forged, bank)

        prompt_swap, bank = _build_envelope()
        evaluator = _receipt(prompt_swap["post_commit_execution"]["evaluator_receipts"][4])
        evaluator["prompt_counterfactual_sha256"] = _sha("wrong-counterfactual-prompt")
        prompt_swap["post_commit_execution"]["evaluator_receipts"][4] = _reseal_entry(evaluator)
        prompt_swap = _seal(prompt_swap)
        with self.assertRaisesRegex(selector.GraftSelectorError, "prompt_counterfactual_sha256"):
            _run_file(prompt_swap, bank)

    def test_post_axes_bind_real_post_execution_and_output_artifact(self) -> None:
        envelope, bank = _build_envelope()
        forged = copy.deepcopy(envelope)
        evaluator = _receipt(forged["post_commit_execution"]["evaluator_receipts"][5])
        evaluator["output_artifact_sha256"] = _sha("different-output-artifact")
        forged["post_commit_execution"]["evaluator_receipts"][5] = _reseal_entry(evaluator)
        forged = _seal(forged)
        with self.assertRaisesRegex(selector.GraftSelectorError, "output_artifact_sha256"):
            _run_file(forged, bank)

        wrong_frame, bank = _build_envelope()
        evaluator = _receipt(wrong_frame["post_commit_execution"]["evaluator_receipts"][5])
        evaluator["output_artifact_frame81_digest"] = _sha("different-frame81")
        wrong_frame["post_commit_execution"]["evaluator_receipts"][5] = _reseal_entry(evaluator)
        wrong_frame = _seal(wrong_frame)
        with self.assertRaisesRegex(selector.GraftSelectorError, "output_artifact_frame81_digest"):
            _run_file(wrong_frame, bank)

    def test_post_execution_id_is_independent_and_output_shape_size_is_exact(self) -> None:
        envelope, bank = _build_envelope()
        forged = copy.deepcopy(envelope)
        post = _receipt(forged["post_commit_execution"]["execution_receipt"])
        trial = _receipt(forged["trial_executions"][2]["execution_receipt"])
        post["execution_id"] = trial["execution_id"]
        forged["post_commit_execution"]["execution_receipt"] = _reseal_entry(post)
        # Rebind all post evaluators so this specifically reaches the global
        # independent-execution-id check rather than failing a stale receipt SHA.
        post_entry = forged["post_commit_execution"]["execution_receipt"]
        registry = _receipt(forged["calibration_receipt"])["axis_registry"]
        forged["post_commit_execution"]["evaluator_receipts"] = _axis_entries(
            "post", 2, post_entry, _observations()[2], registry
        )
        forged = _seal(forged)
        with self.assertRaisesRegex(selector.GraftSelectorError, "independent"):
            _run_file(forged, bank)

        bad_size, bank = _build_envelope()
        post = _receipt(bad_size["post_commit_execution"]["execution_receipt"])
        post["output_artifact_byte_size"] += 1
        bad_size["post_commit_execution"]["execution_receipt"] = _reseal_entry(post)
        bad_size = _seal(bad_size)
        with self.assertRaisesRegex(selector.GraftSelectorError, "size/shape"):
            _run_file(bad_size, bank)

    def test_post_execution_parent_is_exact_selected_trial_receipt(self) -> None:
        envelope, bank = _build_envelope()
        forged = copy.deepcopy(envelope)
        post = _receipt(forged["post_commit_execution"]["execution_receipt"])
        post["selected_trial_execution_receipt_sha256"] = forged["trial_executions"][1][
            "execution_receipt"
        ]["sha256"]
        forged["post_commit_execution"]["execution_receipt"] = _reseal_entry(post)
        forged = _seal(forged)
        with self.assertRaisesRegex(selector.GraftSelectorError, "selected_trial_execution_receipt_sha256"):
            _run_file(forged, bank)

    def test_equal_values_do_not_allow_evaluator_receipt_reuse(self) -> None:
        tied = [[0.8, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8] for _ in range(5)]
        envelope, bank = _build_envelope(observations=tied)
        forged = copy.deepcopy(envelope)
        forged["trial_executions"][1]["evaluator_receipts"][0] = copy.deepcopy(
            forged["trial_executions"][0]["evaluator_receipts"][0]
        )
        forged = _seal(forged)
        with self.assertRaisesRegex(selector.GraftSelectorError, "candidate_index"):
            _run_file(forged, bank)

    def test_underflow_and_nonexact_threshold_encodings_are_rejected(self) -> None:
        envelope, bank = _build_envelope()
        underflow = copy.deepcopy(envelope)
        calibration = _receipt(underflow["calibration_receipt"])
        calibration["axis_registry"][0]["threshold_exact_fp32"] = {
            "fp32_bits": "00000001", "fp32_hex": float(struct.unpack(">f", bytes.fromhex("00000001"))[0]).hex()
        }
        underflow["calibration_receipt"] = _reseal_entry(calibration)
        underflow = _seal(underflow)
        with self.assertRaisesRegex(selector.GraftSelectorError, "normal exact FP32"):
            _run_file(underflow, bank)

        nonexact, bank = _build_envelope()
        calibration = _receipt(nonexact["calibration_receipt"])
        calibration["axis_registry"][0]["threshold_exact_fp32"]["fp32_hex"] = (0.0500001).hex()
        nonexact["calibration_receipt"] = _reseal_entry(calibration)
        nonexact = _seal(nonexact)
        with self.assertRaisesRegex(selector.GraftSelectorError, "round-trip exactly"):
            _run_file(nonexact, bank)

    def test_drop_retelling_has_explicit_absent_digest_and_remains_low_authority(self) -> None:
        envelope, bank = _build_envelope(kind="drop_retelling")
        result = _run_file(envelope, bank)
        payload = result.provenance.payload()
        self.assertEqual(payload["intervention_kind"], "drop_retelling")
        self.assertEqual(payload["effective_retelling_digest"], selector.ABSENT_RETELLING_DIGEST)
        self.assertFalse(payload["source_retelling_condition_present"])
        self.assertFalse(payload["semantic_correctness_authority"])

    def test_empty_feasible_set_abstains_without_post_or_fallback(self) -> None:
        rows = _observations()
        for row in rows:
            row[0] = row[1]
        envelope, bank = _build_envelope(observations=rows)
        result = _run_file(envelope, bank)
        payload = result.provenance.payload()
        self.assertEqual(payload["feasible_candidate_indices"], [])
        self.assertIsNone(payload["observed_candidate_index"])
        self.assertIsNone(payload["accepted_candidate_index"])
        self.assertEqual(payload["selection_status"], "empty_feasible_set")
        self.assertIsNone(result.selected_program)

    def test_exact_ties_and_one_ulp_near_ties_are_deterministic(self) -> None:
        tied_row = [0.8, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
        envelope, bank = _build_envelope(observations=[list(tied_row) for _ in range(5)])
        result = _run_file(envelope, bank)
        self.assertEqual(result.provenance.payload()["accepted_candidate_index"], 0)

        bits = struct.unpack(">I", struct.pack(">f", 0.8))[0]
        one_ulp_up = struct.unpack(">f", struct.pack(">I", bits + 1))[0]
        rows = [list(tied_row) for _ in range(5)]
        rows[1] = [one_ulp_up if i != 1 and i != 2 and i != 3 else 0.2 for i in range(10)]
        # Raising action and all six absolute preservation axes by exactly one
        # FP32 ULP makes candidate 1 strictly dominate without decimal rounding.
        envelope, bank = _build_envelope(observations=rows)
        result = _run_file(envelope, bank)
        self.assertEqual(result.provenance.payload()["accepted_candidate_index"], 1)

    def test_downstream_revalidation_detects_program_and_provenance_mutation(self) -> None:
        envelope, bank = _build_envelope()
        result = _run_file(envelope, bank)
        result.selected_program[0, 0] += 1.0
        with self.assertRaisesRegex(selector.GraftSelectorError, "mutated or interpolated"):
            result.validate()

        clean = _run_file(envelope, bank)
        clean.provenance = selector.SelectionProvenance(
            clean.provenance.payload_json, _sha("forged-provenance")
        )
        with self.assertRaisesRegex(selector.GraftSelectorError, "provenance digest"):
            clean.validate()

    def test_loader_rejects_writable_file_and_symlink(self) -> None:
        envelope, _ = _build_envelope()
        raw = _raw(envelope)
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "envelope.json"
            path.write_bytes(raw)
            path.chmod(0o644)
            with self.assertRaisesRegex(selector.GraftSelectorError, "exact plain 0444"):
                selector.load_sealed_execution_envelope(path.resolve(), expected_file_sha256=digest)
            path.chmod(0o444)
            link = Path(directory) / "link.json"
            link.symlink_to(path)
            with self.assertRaisesRegex(selector.GraftSelectorError, "open failed"):
                selector.load_sealed_execution_envelope(os.fspath(link.absolute()), expected_file_sha256=digest)

    def test_public_api_has_no_scores_media_or_caller_sha_pairs(self) -> None:
        parameters = set(inspect.signature(selector.select_source_conditioned_proposal).parameters)
        forbidden = {
            "raw_observations", "post_commit_raw_observation", "proposal_rgb",
            "proposal_latent", "raw_velocity", "target_video", "mask", "pose",
            "flow", "track", "aggregation_weights", "averaged_program",
            "frozen_trial_receipt_sha256s", "frozen_post_commit_receipt_sha256",
        }
        self.assertTrue(forbidden.isdisjoint(parameters))
        source = inspect.getsource(selector.select_source_conditioned_proposal)
        self.assertNotIn("torch.mean", source)
        self.assertNotIn("torch.softmax", source)


if __name__ == "__main__":
    unittest.main()
