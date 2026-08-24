#!/usr/bin/env python3
"""Externally admit one terminal SAIC top-up detached-review-v2 packet.

Run only after the review submission receipt exists and Slurm reports a
terminal state.  This postflight performs no media generation and submits no
job.  It re-observes the exact review Job with root-owned ``sacct``, verifies
the sealed runtime automation receipt and immutable packet closure, and creates
one fresh terminal admission receipt only for COMPLETED/0:0.

The admission exposes a machine-artifact ingestion interface but preserves the
blind protocol: technical HTML is not released to human observers before two
independent external human response seals, and no human label is synthesized.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Any, Mapping, Sequence
from urllib.parse import quote, unquote


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_SACCT = Path("/usr/bin/sacct")
EXPECTED_SACCT_SHA256 = (
    "fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e"
)
FORMAL_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809"
)
RELEASE_ROOT = (
    FORMAL_ROOT
    / "releases/saic-t2v-topup-r6-formal-v2-r2-retfd-20260812-r1-review-v2-r1"
)
INPUTS = RELEASE_ROOT / "inputs"
EXPECTED_LAUNCHER = (
    INPUTS / "auh_build_saic_t2v_event_bank_topup_detached_review_v2_cpu.sbatch"
)
EXPECTED_ADAPTER = (
    INPUTS / "build_saic_t2v_event_bank_topup_detached_review_v2.py"
)
EXPECTED_POSTFLIGHT = (
    INPUTS / "postflight_saic_t2v_event_bank_topup_detached_review_v2.py"
)
EXPECTED_SUBMITTER = (
    INPUTS / "submit_saic_t2v_event_bank_topup_detached_review_v2.py"
)
EXPECTED_HOSTILE = (
    INPUTS / "test_saic_t2v_topup_detached_review_v2_release_auh.py"
)
EXPECTED_MANIFEST_MATERIALIZER = (
    INPUTS / "materialize_saic_t2v_topup_review_v2_release_manifest_v1.py"
)
EXPECTED_RELEASE_MANIFEST = RELEASE_ROOT / "release-manifest.json"
EXPECTED_ADAPTER_SHA256 = "__REVIEW_ADAPTER_SHA256__"
EXPECTED_LAUNCHER_SHA256 = "__REVIEW_LAUNCHER_SHA256__"
EXPECTED_COMPUTE_BASH_SHA256 = "__REVIEW_COMPUTE_BASH_SHA256__"
EXPECTED_SOURCE_ARCHIVE = INPUTS / "videoedit-saic-20c2193-methods.tar"
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b"
)
EXPECTED_SOURCE_REVISION = "20c2193954e780e9654347754b1485f3492fbea5"
EXPECTED_FORMAL_ROOT = (
    FORMAL_ROOT / "runs/t2v-events-topup-r6-formal-v2-r2-retfd-20260812-r1"
)
EXPECTED_FORMAL_MASTER = (
    EXPECTED_FORMAL_ROOT / "saic-pure-t2v-event-bank-topup-receipt.json"
)
EXPECTED_FORMAL_SUBMISSION = Path(
    str(EXPECTED_FORMAL_ROOT) + ".submission.receipt.json"
)
EXPECTED_PYTHON = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
EXPECTED_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
EXPECTED_FFMPEG = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
    "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)
EXPECTED_FFMPEG_SHA256 = (
    "e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99"
)
EXPECTED_FFPROBE_WRAPPER_SHA256 = (
    "06a5698bdaa8069e541e804e2f0e945665b2d16409633aa92d37f76ed7506710"
)
EXPECTED_COMPUTE_BASH = Path("/usr/bin/bash")
EXPECTED_COMPUTE_BASH_VERSION_STDOUT_SHA256 = (
    "51bd40ffa4710175920033d329a0e9e1667e6b7f56178e302432ff4610d554a7"
)
EXPECTED_COMPUTE_BASH_VERSION_FIRST_LINE = (
    "GNU bash, version 5.1.16(1)-release (x86_64-pc-linux-gnu)"
)
PACKET_ID = (
    "t2v-events-topup-r6-formal-v2-r2-retfd-20260812-r1-"
    "detached-review-v2-r1"
)
OUTPUT_ROOT = FORMAL_ROOT / "reviews" / PACKET_ID
SUBMISSION_RECEIPT = Path(str(OUTPUT_ROOT) + ".submission.receipt.json")
AUTOMATION_RECEIPT = Path(str(OUTPUT_ROOT) + ".automation.receipt.json")
TERMINAL_ADMISSION = Path(str(OUTPUT_ROOT) + ".terminal.admission.receipt.json")
EXPECTED_SLURM_LOG_DIR = (
    FORMAL_ROOT
    / "slurm/saic-t2v-topup-r6-formal-v2-r2-retfd-20260812-r1-review-v2-r1"
)
SACCT_FIELDS = [
    "JobIDRaw", "State", "ExitCode", "AllocTRES%512", "NodeList",
    "ElapsedRaw", "Start", "End", "SubmitLine%8192",
]
EXPORT_NAMES = [
    "SAIC_T2V_TOPUP_REVIEW_RELEASE_ROOT",
    "SAIC_T2V_TOPUP_REVIEW_SOURCE_ARCHIVE",
    "SAIC_T2V_TOPUP_REVIEW_SOURCE_ARCHIVE_SHA256",
    "SAIC_T2V_TOPUP_REVIEW_SOURCE_REVISION",
    "SAIC_T2V_TOPUP_REVIEW_ADAPTER",
    "SAIC_T2V_TOPUP_REVIEW_ADAPTER_SHA256",
    "SAIC_T2V_TOPUP_REVIEW_INPUT_ROOT",
    "SAIC_T2V_TOPUP_REVIEW_FORMAL_MASTER",
    "SAIC_T2V_TOPUP_REVIEW_FORMAL_MASTER_SHA256",
    "SAIC_T2V_TOPUP_REVIEW_FORMAL_MASTER_DIGEST",
    "SAIC_T2V_TOPUP_REVIEW_FORMAL_SUBMISSION",
    "SAIC_T2V_TOPUP_REVIEW_FORMAL_SUBMISSION_SHA256",
    "SAIC_T2V_TOPUP_REVIEW_FORMAL_SUBMISSION_DIGEST",
    "SAIC_T2V_TOPUP_REVIEW_FORMAL_JOB_ID",
    "SAIC_T2V_TOPUP_REVIEW_OUTPUT_ROOT",
    "SAIC_T2V_TOPUP_REVIEW_AUTOMATION_RECEIPT",
    "SAIC_T2V_TOPUP_REVIEW_PYTHON_BIN",
    "SAIC_T2V_TOPUP_REVIEW_PYTHON_SHA256",
    "SAIC_T2V_TOPUP_REVIEW_FFMPEG_BIN",
    "SAIC_T2V_TOPUP_REVIEW_FFMPEG_SHA256",
    "SAIC_T2V_TOPUP_REVIEW_FFPROBE_WRAPPER_SHA256",
    "SAIC_T2V_TOPUP_REVIEW_COMPUTE_BASH",
    "SAIC_T2V_TOPUP_REVIEW_COMPUTE_BASH_SHA256",
    "SAIC_T2V_TOPUP_REVIEW_COMPUTE_BASH_VERSION_STDOUT_SHA256",
    "SAIC_T2V_TOPUP_REVIEW_WORKERS",
]
FORMAL_SACCT_FIELDS = [
    "JobIDRaw", "State", "ExitCode", "AllocTRES%512", "NodeList",
    "ElapsedRaw", "Start", "End",
]
EXPECTED_FORMAL_ALLOC_TRES = {
    "billing": "32", "cpu": "32", "gres/gpu:mi210": "8",
    "gres/gpu": "8", "mem": "256G", "node": "1",
}
EXPECTED_REVIEW_ALLOC_TRES = {
    "billing": "32", "cpu": "32", "mem": "192G", "node": "1",
}
BLANK_RESPONSE_FIELDS = {
    "review_item_id",
    "registered_q0_state_correct_frames_0_15",
    "registered_q0_state_held_full81",
    "registered_target_terminal_absent_full81",
    "incomplete_directional_progress_visible",
    "incomplete_partial_state_held_frames_73_80",
    "registered_smooth_camera_motion_visible",
    "camera_cut_or_discontinuity_absent",
    "registered_appearance_only_change_visible",
    "camera_locked_full81",
    "nonregistered_subject_appearance_or_geometry_change_absent",
    "event_branch_pass",
    "identity_preserved_full81",
    "technical_quality_acceptable_full81",
    "observer_notes",
}
BLANK_TEMPLATE_FIELDS = {
    "schema_version", "packet_id", "observer_slot", "template_only",
    "semantic_status", "observer_id", "observer_kind",
    "observer_authority_artifact", "observer_protocol_artifact",
    "completed_at", "independent_observer_required",
    "same_person_must_not_fill_both_slots",
    "copy_outside_sealed_packet_before_completion",
    "blindness_or_independence_established_by_template",
    "review_item_set_digest", "responses", "authority", "template_digest",
}
FALSE_AUTHORITY = {
    "machine_diagnostics_have_semantic_authority": False,
    "event_verified": False,
    "identity_preservation_verified": False,
    "candidate_selection_allowed": False,
    "seed_selection_allowed": False,
    "training_target_allowed": False,
    "training_allowed": False,
    "optimizer_step_allowed": False,
    "parameter_update_allowed": False,
    "absolute_action_editing_success_claimed": False,
}
RELEASE_AUTHORITY = {
    "scientific": False, "human_review": False, "event_verified": False,
    "identity_preservation_verified": False, "candidate_selection": False,
    "seed_selection": False, "training_target": False, "training": False,
    "optimizer_step": False, "parameter_update": False,
}
DIAGNOSTIC_AUTHORITY = {
    "measurement_runtime_qualified": False,
    "candidate_selection_allowed": False,
    "training_allowed": False,
    "optimizer_step_allowed": False,
    "absolute_action_editing_success_claimed": False,
}
REVIEW_RECEIPT_FIELDS = {
    "schema_version", "packet_id", "job_id", "input_bank_receipt",
    "review_manifest", "html_review", "blind_human_review",
    "observer_protocol", "row_count", "source_count", "seed_cell_count",
    "candidate_count", "machine_diagnostic_count",
    "exact81_machine_diagnostics_complete",
    "full80_transition_machine_diagnostics_complete",
    "machine_diagnostics_zero_authority",
    "detached_full81_event_review_complete", "semantic_status",
    "event_verified", "identity_preservation_verified",
    "candidate_ranking_or_selection_performed", "seed_selection_authorized",
    "training_target_authorized", "training_performed", "optimizer_created",
    "optimizer_step_authorized", "parameter_update_authorized",
    "observer_template_count", "observer_labels_present", "authority",
    "receipt_digest",
}
REVIEW_MANIFEST_FIELDS = {
    "schema_version", "packet_id", "job_id", "input_root",
    "input_bank_receipt_digest", "input_event_spec_raw_sha256",
    "input_bindings", "row_count", "source_count", "seed_cell_count",
    "candidate_count", "machine_diagnostic_count", "frame_count_per_media",
    "transition_count_per_media", "fps", "branch_order",
    "machine_diagnostic_axes", "machine_diagnostic_authority",
    "semantic_status", "detached_human_review_complete", "observer_protocol",
    "observer_template_count", "observer_templates",
    "candidate_ranking_or_selection_performed", "authority", "items",
    "manifest_digest",
}
OBSERVER_PROTOCOL_FIELDS = {
    "schema_version", "packet_id", "protocol_timing",
    "post_generation_protocol_cannot_claim_pre_generation_preregistration",
    "review_item_set_digest", "media_contract", "stage_order",
    "branch_specific_full81_criteria", "shared_full81_axes",
    "observer_contract", "aggregation_rule", "machine_diagnostic_contract",
    "authority", "protocol_digest",
}
REVIEW_ITEM_FIELDS = {
    "registered_candidate_index", "candidate_id", "row_id", "iid",
    "analysis_split", "actor_family", "action_family_id", "branch", "seed",
    "initial_state_type", "terminal_state_type", "branch_start_state_caption",
    "branch_instruction", "full_t2v_caption", "source_input_path",
    "source_sha256", "candidate_input_path", "candidate_sha256",
    "attempt_receipt_input_path", "attempt_receipt_sha256",
    "attempt_receipt_digest", "semantic_status", "event_verified",
    "identity_preservation_verified", "assessor_private_candidate_id",
    "assessor_private_source_row_id", "review_item_id", "portable_source",
    "portable_candidate", "portable_attempt_receipt", "portable_diagnostic",
    "portable_source_bytes", "portable_candidate_bytes",
    "portable_attempt_receipt_bytes", "diagnostic_digest",
    "diagnostic_file_sha256", "diagnostic_summary",
}
FORMAL_MASTER_FIELDS = {
    "schema_version", "bank_id", "top_up_only", "root_spec_raw_sha256",
    "base_v1_spec_raw_sha256", "base_v1_spec_content_sha256",
    "source_manifest_content_sha256", "topology", "sampling_contract",
    "semantic_input_closure", "geometry_proxy_contract",
    "artifact_authority", "attempt_count", "row_count", "seed_cell_count",
    "branch_order", "merged_branch_order", "six_branch_spec_merge_cell_count",
    "same_seed_official_gaussian_proofs", "attempts",
    "detached_full81_event_review_complete", "event_verified",
    "identity_preservation_verified", "seed_selection_authorized",
    "training_target_authorized", "optimizer_or_parameter_update_authorized",
    "receipt_digest",
}
FORMAL_MASTER_ATTEMPT_FIELDS = {
    "candidate_id", "row_id", "iid", "analysis_split", "branch", "seed",
    "receipt_path", "receipt_sha256", "receipt_digest", "mp4_path",
    "mp4_sha256", "event_audit_status",
}
FORMAL_GAUSSIAN_PROOF_FIELDS = {
    "iid", "seed", "branch_order",
    "official_gaussian_tensor_values_byte_equal",
    "official_gaussian_identity_digest",
}
FORMAL_SUBMISSION_FIELDS = {
    "schema_version", "status", "submission_success", "job_success",
    "submitted_job", "request", "submission_boundary", "inputs",
    "canary_admission", "outputs", "authority", "threat_model",
    "receipt_digest",
}
FORMAL_SUBMISSION_INPUT_FIELDS = {
    "wrapper", "wrapper_sha256", "base_launcher", "base_launcher_sha256",
    "materializer", "materializer_sha256", "effective_launcher",
    "effective_launcher_sha256", "gate", "gate_sha256",
    "rendezvous_guard", "rendezvous_guard_sha256",
    "retained_fd_canary_admission", "retained_fd_canary_admission_sha256",
    "retained_fd_canary_admission_digest", "retained_fd_canary_job_id",
    "probe_validator", "probe_validator_sha256",
    "compute_bash_probe_admission", "compute_bash_probe_admission_sha256",
    "compute_bash_probe_admission_digest", "source_archive",
    "source_archive_sha256", "generation_runtime_sha256",
    "archived_rendezvous_guard_v1_sha256", "base_v1_spec_sha256",
    "contract_runtime_sha256", "effective_dynamic_plan_schema_version",
    "scientific_spec_changed_for_rendezvous_guard_v2", "source_revision",
    "source_manifest", "source_manifest_sha256", "event_spec",
    "event_spec_sha256", "checkpoint_manifest", "checkpoint_manifest_sha256",
    "python", "python_sha256", "static_ffmpeg", "static_ffmpeg_sha256",
    "static_ffmpeg_version_stdout_sha256", "static_ffmpeg_version_first_line",
    "compute_bash", "compute_bash_sha256",
    "compute_bash_version_stdout_sha256", "compute_bash_version_first_line",
    "bernini_root", "veomni_root", "checkpoint",
}
FORMAL_EXPORT_NAMES = [
    "SAIC_T2V_FV2_BASE_LAUNCHER", "SAIC_T2V_FV2_BASE_LAUNCHER_SHA256",
    "SAIC_T2V_FV2_MATERIALIZER", "SAIC_T2V_FV2_MATERIALIZER_SHA256",
    "SAIC_T2V_FV2_EFFECTIVE_LAUNCHER",
    "SAIC_T2V_FV2_EFFECTIVE_LAUNCHER_SHA256", "SAIC_T2V_FV2_WRAPPER",
    "SAIC_T2V_FV2_WRAPPER_SHA256", "SAIC_T2V_FV2_GATE",
    "SAIC_T2V_FV2_GATE_SHA256", "SAIC_T2V_V4_EXTERNAL_RENDEZVOUS_GUARD",
    "SAIC_T2V_V3_RENDEZVOUS_GUARD_SHA256",
    "SAIC_T2V_FV2_CANARY_RECEIPT",
    "SAIC_T2V_FV2_CANARY_SUBMISSION_RECEIPT",
    "SAIC_T2V_FV2_RETAINED_FD_CANARY_ADMISSION",
    "SAIC_T2V_FV2_PROBE_VALIDATOR", "SAIC_T2V_FV2_PROBE_VALIDATOR_SHA256",
    "SAIC_T2V_FV2_COMPUTE_BASH_PROBE_ADMISSION",
    "SAIC_T2V_FV2_COMPUTE_BASH_PROBE_ADMISSION_SHA256",
    "SAIC_T2V_FV2_COMPUTE_BASH_PROBE_ADMISSION_DIGEST",
    "SAIC_T2V_FV2_OWN_SUBMISSION_RECEIPT", "SAIC_T2V_V3_SOURCE_ARCHIVE",
    "SAIC_T2V_V3_SOURCE_ARCHIVE_SHA256", "SAIC_T2V_V3_SOURCE_REVISION",
    "SAIC_T2V_V3_SOURCE_MANIFEST", "SAIC_T2V_V3_SOURCE_MANIFEST_SHA256",
    "SAIC_T2V_V3_EVENT_SPEC", "SAIC_T2V_V3_EVENT_SPEC_SHA256",
    "BERNINI_OFFICIAL_ROOT", "BERNINI_VEOMNI_ROOT",
    "BERNINI_ACTION_CHECKPOINT", "BERNINI_CHECKPOINT_CONTENT_MANIFEST",
    "SAIC_T2V_FV2_CHECKPOINT_MANIFEST_SHA256", "SAIC_T2V_V3_OUTPUT_ROOT",
    "SAIC_T2V_V3_PYTHON_BIN", "SAIC_T2V_FV2_PYTHON_SHA256",
    "SAIC_T2V_V3_STATIC_FFMPEG", "SAIC_T2V_FV2_STATIC_FFMPEG_SHA256",
    "SAIC_T2V_FV2_STATIC_FFMPEG_VERSION_STDOUT_SHA256",
    "SAIC_T2V_FV2_STATIC_FFMPEG_VERSION_FIRST_LINE",
    "SAIC_T2V_FV2_COMPUTE_BASH", "SAIC_T2V_FV2_COMPUTE_BASH_SHA256",
    "SAIC_T2V_FV2_COMPUTE_BASH_VERSION_STDOUT_SHA256",
    "SAIC_T2V_FV2_SLURM_LOG_DIR",
]
FORMAL_THREAT_MODEL = {
    "pathname_replacement_rename_symlink_leaf_swap_resistance": True,
    "retained_fd_admission_roots": [
        "formal_gate", "effective_launcher", "rendezvous_guard_v2",
        "compute_bash_probe_validator",
    ],
    "shared_science_paths_assumed_not_concurrently_replaced_by_same_uid": [
        "source_archive", "source_manifest", "event_spec",
        "checkpoint_manifest", "python", "static_ffmpeg", "bernini_root",
        "veomni_root", "checkpoint",
    ],
    "same_inode_in_place_mutation_resistance_claimed": False,
    "sealed_release_permissions_are_not_claimed_as_same_uid_immutability": True,
    "three_independent_operational_proof_objects_required": True,
    "exact60_lifecycle_probe_world8_transport_non_substitutability": True,
    "compute_bash_exact_identity_pinned": True,
}
FORMAL_CANARY_ADMISSION_FIELDS = {
    "job_id", "terminal_receipt_path", "terminal_receipt_sha256",
    "terminal_receipt_digest", "submission_receipt_path",
    "submission_receipt_sha256", "submission_receipt_digest",
    "slurm_state_required", "slurm_exit_code_required",
    "allocated_gpu_resource_required", "sacct_observation",
    "compute_bash_probe_admission", "retained_fd_world8",
}
FORMAL_PROBE_BINDING_FIELDS = {
    "path", "sha256", "receipt_digest", "schema_version", "status",
    "slurm_job_id", "compute_bash", "submission_receipt_sha256",
    "submission_receipt_digest", "submission_receipt_path",
    "operational_evidence_sha256", "operational_evidence_digest",
    "operational_evidence_path", "release_manifest_path",
    "release_manifest_file_sha256", "release_manifest_digest",
    "wrapper_sha256", "postflight_sha256", "authority",
}
FORMAL_RETAINED_WORLD8_FIELDS = {
    "job_id", "admission_path", "admission_sha256", "admission_digest",
    "operational_evidence_path", "operational_evidence_sha256",
    "operational_evidence_digest", "wrapper_sha256", "payload_sha256",
    "guard_sha256", "runtime_sha256", "probe_validator_sha256",
    "probe_admission_binding", "compute_bash",
    "external_postflight_sacct_observation", "slurm_state_required",
    "slurm_exit_code_required", "allocated_gpu_resource_required",
    "science_generation_entered",
    "formal_submission_authorized_by_canary_alone",
    "submitter_sacct_observation",
}
FORMAL_PROBE_AUTHORITY = {
    "scientific": False, "generation": False, "training": False,
    "publication": False, "formal_job_authorized": False,
}
EXPECTED_FORMAL_SLURM_LOG_DIR = (
    FORMAL_ROOT
    / "slurm/saic-t2v-topup-r6-formal-v2-r2-retfd-20260812-r1"
)
EXPECTED_BASE_V1_SPEC_RAW_SHA256 = (
    "623a7ed8a2ce2d327247c541b59aa2d39f1fbfe4a480f7351d042c7ef7a47927"
)
EXPECTED_BASE_V1_SPEC_CONTENT_SHA256 = (
    "3920d5c121b75c6bbf984c24440c9773dfb49006778c61a671ae50963bb5456a"
)
EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "9c2a3d6841951ea0ed050dc230630a1176460e25a979ec199eab575ad22f3c6f"
)
EXPECTED_EVENT_SPEC_RAW_SHA256 = (
    "d693d0784530f007888e2825d15db3db808fdf4f1d111b5d080d968c894ff145"
)


def die(message: str) -> None:
    raise SystemExit(f"postflight-saic-t2v-topup-review-v2: {message}")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def expected_observer_protocol(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    review_projection = [
        {
            "review_item_id": item["review_item_id"],
            "candidate_media_sha256": item["candidate_sha256"],
            "source_media_sha256": item["source_sha256"],
            "branch": item["branch"],
            "seed": item["seed"],
        }
        for item in items
    ]
    body = {
        "schema_version":
            "bernini-saic-t2v-event-bank-topup-independent-full81-"
            "observer-protocol-v2",
        "packet_id": PACKET_ID,
        "protocol_timing":
            "SEALED_AFTER_GENERATION_BEFORE_ANY_DETACHED_HUMAN_LABELS",
        "post_generation_protocol_cannot_claim_pre_generation_preregistration":
            True,
        "review_item_set_digest": sha_bytes(canonical(review_projection)),
        "media_contract": {
            "source_and_candidate_shown_side_by_side": True,
            "candidate_frame_count": 81, "source_frame_count": 81,
            "fps": 25,
            "all_frames_must_be_viewed_at_normal_speed_at_least_once": True,
            "frame_scrubbing_and_replay_after_first_view_allowed": True,
            "candidate_id_seed_and_machine_metrics_hidden_in_human_stage": True,
        },
        "stage_order": [
            {
                "stage": 1, "artifact": "blind-review.html",
                "rule": "Each observer independently seals all 60 responses "
                        "before opening any machine diagnostic artifact.",
            },
            {
                "stage": 2, "artifact": "index.html and diagnostics/*.json",
                "rule": "Machine diagnostics may be inspected only after both "
                        "human response artifacts are immutable and cannot revise "
                        "labels.",
            },
        ],
        "branch_specific_full81_criteria": {
            "incomplete": {
                "start": "The registered q0 start state is visibly correct "
                         "throughout frames 0-15.",
                "progress": "A directionally correct partial progression begins "
                            "and is visibly sustained, but remains incomplete.",
                "terminal_absence": "The registered target terminal state is "
                                    "never reached at any frame 0-80.",
                "end_hold": "A visibly partial, nonterminal state remains held "
                            "through every frame 73-80.",
                "camera_lock": "The registered locked camera and framing remain "
                               "fixed throughout frames 0-80.",
                "event_pass": "start AND progress AND terminal_absence AND "
                              "end_hold AND camera_lock must all be true.",
            },
            "camera_only": {
                "state_hold": "The registered q0 actor/action state remains held "
                              "throughout frames 0-80 and the target event never "
                              "occurs.",
                "camera_motion": "The registered conspicuous, smooth camera "
                                 "motion occurs without a cut or discontinuity.",
                "appearance_hold": "The subject appearance and geometry remain "
                                   "materially unchanged apart from viewpoint "
                                   "effects.",
                "event_pass": "state_hold AND camera_motion AND appearance_hold "
                              "must all be true.",
            },
            "appearance_only": {
                "state_hold": "The registered q0 actor/action state remains held "
                              "throughout frames 0-80 and the target event never "
                              "occurs.",
                "appearance_change": "The registered appearance-only change is "
                                     "clearly visible without creating the target "
                                     "action event.",
                "camera_lock": "The camera and framing remain locked throughout "
                               "frames 0-80.",
                "geometry_hold": "Subject geometry and scene layout remain "
                                 "materially fixed while only the registered "
                                 "appearance attributes change.",
                "event_pass": "state_hold AND appearance_change AND camera_lock "
                              "AND geometry_hold must all be true.",
            },
        },
        "shared_full81_axes": {
            "identity_preserved_full81": "The same source subject remains "
                                         "recognizable with no swap, replacement, "
                                         "or material identity morph at every "
                                         "frame 0-80.",
            "technical_quality_acceptable_full81": "No ghosting, duplication, "
                                                   "tearing, disappearance, or "
                                                   "corruption obscures the branch "
                                                   "judgment at any frame.",
        },
        "observer_contract": {
            "minimum_independent_observers": 2,
            "observer_kind": "independent_human_full81_review",
            "different_people_required": True,
            "communication_or_label_sharing_before_seal_forbidden": True,
            "distinct_observer_identity_and_authority_artifacts_required": True,
            "same_preparer_must_not_act_as_either_observer": True,
            "one_person_filling_both_templates_forbidden": True,
        },
        "aggregation_rule": {
            "majority_vote_allowed": False,
            "tie_break_or_adjudication_inside_v2_allowed": False,
            "missing_response_result": "UNASSESSED",
            "observer_disagreement_result": "UNASSESSED",
            "agreed_positive_result":
                "AGREED_POSITIVE_PENDING_SEPARATE_EXTERNAL_SEAL",
            "agreed_negative_result":
                "AGREED_NEGATIVE_PENDING_SEPARATE_EXTERNAL_SEAL",
            "event_verified_may_be_set_by_this_packet": False,
            "identity_verified_may_be_set_by_this_packet": False,
            "separate_versioned_aggregator_required": True,
        },
        "machine_diagnostic_contract": {
            "human_labels_must_precede_machine_diagnostic_access": True,
            "machine_camera_or_technical_thresholds_calibrated": False,
            "machine_diagnostics_may_fill_or_change_human_labels": False,
            "machine_diagnostics_have_semantic_authority": False,
            "machine_diagnostics_may_select_seed_or_training_target": False,
        },
        "authority": FALSE_AUTHORITY,
    }
    return {**body, "protocol_digest": sha_bytes(canonical(body))}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retained_plain_bytes(
    path: Path, *, label: str, exact_mode: int | None = None,
) -> tuple[int, bytes, os.stat_result]:
    if not path.is_absolute():
        die(f"{label} path is not absolute")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        public = path.lstat()
        mode = stat.S_IMODE(before.st_mode)
        if (
            path.resolve(strict=True) != path
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or (exact_mode is not None and mode != exact_mode)
            or (exact_mode is None and mode & 0o022)
            or not stat.S_ISREG(public.st_mode)
            or stat.S_ISLNK(public.st_mode)
            or (public.st_dev, public.st_ino) != (before.st_dev, before.st_ino)
        ):
            die(f"{label} retained identity differs")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        public_after = path.lstat()
        identity = lambda item: (
            item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
        )
        if (
            identity(before) != identity(after)
            or identity(after) != identity(public_after)
            or len(raw) != after.st_size
            or not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or after.st_uid != os.getuid()
            or stat.S_IMODE(after.st_mode) != mode
            or not stat.S_ISREG(public_after.st_mode)
            or stat.S_ISLNK(public_after.st_mode)
            or public_after.st_nlink != 1
            or public_after.st_uid != os.getuid()
            or stat.S_IMODE(public_after.st_mode) != mode
        ):
            die(f"{label} bytes changed while retained")
        return descriptor, raw, after
    except BaseException:
        os.close(descriptor)
        raise


def reread_retained(
    descriptor: int, path: Path, raw: bytes, initial: os.stat_result, *, label: str,
) -> None:
    before = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
    public = path.lstat()
    identity = lambda item: (
        item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
    )
    if (
        b"".join(chunks) != raw
        or identity(before) != identity(after)
        or identity(after) != identity(public)
        or before.st_uid != initial.st_uid
        or after.st_uid != initial.st_uid
        or public.st_uid != initial.st_uid
        or initial.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != stat.S_IMODE(initial.st_mode)
        or stat.S_IMODE(after.st_mode) != stat.S_IMODE(initial.st_mode)
        or stat.S_IMODE(public.st_mode) != stat.S_IMODE(initial.st_mode)
        or not stat.S_ISREG(public.st_mode)
        or stat.S_ISLNK(public.st_mode)
        or public.st_nlink != 1
    ):
        die(f"{label} terminal retained reread differs")


def no_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            die(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def plain_file(path: Path, *, mode: int = 0o444, label: str) -> Path:
    info = path.lstat()
    if (
        path.resolve(strict=True) != path
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != mode
    ):
        die(f"{label} identity differs")
    return path


def packet_path(relative: Any, *, label: str) -> Path:
    if type(relative) is not str:
        die(f"{label} portable path differs")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != relative
    ):
        die(f"{label} portable path differs")
    path = OUTPUT_ROOT.joinpath(*pure.parts)
    if OUTPUT_ROOT not in path.parents:
        die(f"{label} escaped review packet")
    return path


def validate_copy_binding(
    binding: Any, *, portable_key: str, label: str
) -> Path:
    allowed = {"path", "portable_path", "sha256", "bytes"}
    if label == "event_spec evidence":
        allowed.add("base_v1_spec")
    if (
        type(binding) is not dict
        or set(binding) != allowed
        or type(binding.get("bytes")) is not int
        or binding["bytes"] < 0
        or SHA256.fullmatch(str(binding.get("sha256"))) is None
    ):
        die(f"{label} binding differs")
    path = packet_path(binding.get(portable_key), label=label)
    plain_file(path, label=label)
    if (
        binding.get("path") != str(path)
        or path.stat().st_size != binding["bytes"]
        or sha_file(path) != binding["sha256"]
    ):
        die(f"{label} binding differs")
    return path


class _BlindSurfaceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []
        self.sources: list[str] = []
        self.video_sources: list[str] = []
        self.video_count = 0
        self.video_end_count = 0
        self.video_depth = 0
        self.tags: list[str] = []
        self.attribute_values: list[str] = []
        self.text: list[str] = []
        self.forbidden_transport = False

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        lowered_tag = tag.lower()
        self.tags.append(lowered_tag)
        src_values: list[str] = []
        for name, value in attrs:
            lowered = name.lower()
            decoded = "" if value is None else unquote(value)
            self.attribute_values.append(decoded)
            if lowered == "href":
                self.hrefs.append(decoded)
            elif lowered == "src":
                self.sources.append(decoded)
                src_values.append(decoded)
            if (
                lowered.startswith("on")
                or lowered in {
                    "action", "background", "cite", "data", "formaction",
                    "http-equiv", "poster", "srcset",
                }
            ):
                self.forbidden_transport = True
        if lowered_tag == "video":
            self.video_count += 1
            self.video_depth += 1
            if self.video_depth != 1:
                self.forbidden_transport = True
            if len(src_values) != 1:
                self.forbidden_transport = True
            else:
                self.video_sources.append(src_values[0])
        elif src_values:
            self.forbidden_transport = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "video":
            self.video_end_count += 1
            self.video_depth -= 1
            if self.video_depth != 0:
                self.forbidden_transport = True

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def expected_blind_surface(
    items: Sequence[dict[str, Any]], *, job_id: str, protocol_digest: str,
) -> str:
    """Independently rebuild the one allowed stage-1 browser surface."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(str(item["row_id"]), []).append(item)
    sections: list[str] = []
    for row_index, row_items in enumerate(grouped.values(), start=1):
        first = row_items[0]
        cards: list[str] = []
        for item in row_items:
            instruction = str(item["branch_instruction"])
            prefix = {
                "incomplete": "",
                "camera_only": "Counterfactual camera-only negative: ",
                "appearance_only": "Counterfactual appearance-only negative: ",
            }[str(item["branch"])]
            if not instruction.startswith(prefix):
                die("blind registered criterion prefix differs")
            instruction = instruction[len(prefix):]
            if instruction:
                instruction = instruction[0].upper() + instruction[1:]
            cards.append(
                f'''<article class="card"><header><span class="eyebrow">{html.escape(str(item['review_item_id']))}</span><h3>Registered evaluation criterion</h3><p>{html.escape(str(item['branch_start_state_caption']))}</p><p>{html.escape(instruction)}</p></header><video controls muted playsinline preload="metadata" src="{quote(str(item['portable_candidate']), safe='/:._-')}"></video></article>'''
            )
        sections.append(
            f'''<section class="sample"><h2>Blind source set {row_index:02d}</h2><p class="muted">Registered identifiers, sampling metadata, and machine measurements are absent from this page.</p><div class="source"><article class="card source-card"><header><span class="eyebrow">SOURCE REFERENCE</span><h3>Hash-bound exact81 source</h3></header><video controls muted playsinline preload="metadata" src="{quote(str(first['portable_source']), safe='/:._-')}"></video></article></div><div class="grid">{''.join(cards)}</div></section>'''
        )
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SAIC T2V blind human review · Job {html.escape(job_id)}</title><style>
:root{{--ink:#18211d;--paper:#f3efe7;--panel:#fffdf7;--line:#cdc6b8;--muted:#68716c;--accent:#166953;--warn:#873d1b;--warnbg:#ffefdc}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}main{{max-width:1880px;margin:auto;padding:24px}}.hero,.sample{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px;margin-bottom:17px}}h1{{font-size:clamp(29px,4vw,52px);line-height:1.03;margin:7px 0 13px}}h2,h3,p{{margin-top:0}}.eyebrow{{font-size:11px;font-weight:800;color:var(--accent);letter-spacing:.1em}}.warning{{background:var(--warnbg);border:1px solid #dda171;border-radius:10px;padding:13px;color:var(--warn)}}.muted,.card p{{color:var(--muted)}}.source{{max-width:330px;margin:12px 0}}.grid{{display:grid;grid-template-columns:repeat(6,minmax(195px,1fr));gap:9px}}.card{{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#faf8f2}}.source-card{{border-color:#6b9f8b;background:#edf7f2}}.card header{{padding:9px;min-height:150px}}video{{display:block;width:100%;aspect-ratio:1/1;object-fit:contain;background:#0c0e0d}}@media(max-width:1200px){{.grid{{grid-template-columns:repeat(3,1fr)}}}}@media(max-width:700px){{main{{padding:8px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><section class="hero"><span class="eyebrow">INDEPENDENT HUMAN STAGE 1 · AUH {html.escape(job_id)}</span><h1>Blind full81 review</h1><p class="warning"><strong>Use only this stage-1 page and its embedded media.</strong> Watch every source/proposal pair through all 81 frames at normal speed, complete one assigned observer template independently, and seal that response outside this packet. Do not access any technical or assessor-private artifact before both response seals exist. Machine diagnostics cannot revise a human label.</p><p>Protocol digest: <code>{html.escape(protocol_digest)}</code>. Registered identifiers, sampling metadata, and all machine metrics are absent from this page.</p></section>{''.join(sections)}</main></body></html>'''


def validate_blind_surface(
    items: Sequence[dict[str, Any]], *, job_id: str, protocol_digest: str,
) -> None:
    blind_path = OUTPUT_ROOT / "blind-review.html"
    plain_file(blind_path, label="blind human review HTML")
    try:
        rendered = blind_path.read_text(encoding="utf-8")
    except UnicodeError as error:
        die(f"blind human review HTML encoding differs: {error}")
    parser = _BlindSurfaceParser()
    parser.feed(rendered)
    parser.close()
    expected_rendered = expected_blind_surface(
        items, job_id=job_id, protocol_digest=protocol_digest,
    )
    source_paths = {str(item.get("portable_source")) for item in items}
    candidate_paths = [str(item.get("portable_candidate")) for item in items]
    if (
        len(source_paths) != 8
        or len(candidate_paths) != 60
        or rendered != expected_rendered
        or any(
            re.fullmatch(r"media/sources/source-[0-9]{4}\.mp4", path) is None
            for path in source_paths
        )
        or any(
            re.fullmatch(
                r"media/candidates/[0-9]{4}-candidate-[0-9]{4}\.mp4",
                path,
            ) is None
            for path in candidate_paths
        )
        or any(tag in {"base", "embed", "form", "iframe", "link", "object", "script"}
               for tag in parser.tags)
        or parser.forbidden_transport
        or parser.hrefs
        or parser.video_count != 68
        or parser.video_end_count != 68
        or parser.video_depth != 0
        or len(parser.video_sources) != 68
        or parser.sources != parser.video_sources
        or Counter(parser.video_sources)
        != Counter([*sorted(source_paths), *candidate_paths])
    ):
        die("blind HTML href/src opaque namespace differs")
    lower = "\n".join(
        [rendered, *parser.attribute_values, *parser.text]
    ).lower()
    forbidden_identifiers = {
        str(item.get(field))
        for item in items
        for field in (
            "assessor_private_candidate_id",
            "assessor_private_source_row_id",
            "iid",
        )
    } | {
        str(item.get(field))
        for item in items
        for field in ("branch",)
    } | {
        f"s{item.get('seed')}" for item in items
    } | {
        "incomplete", "camera_only", "appearance_only",
        "camera-only", "appearance-only",
    }
    if (
        any(token and token.lower() in lower for token in forbidden_identifiers)
        or "seed" in lower
        or "candidate_id" in lower
        or "url(" in lower
        or "@import" in lower
        or any(
            token in lower
            for token in (
                "index.html", "review-manifest.json", "diagnostics/", "evidence/",
                "observer-protocol.json", "detached-review-receipt.json",
                "observer-templates/",
            )
        )
    ):
        die("blind HTML leaks assessor-private identifiers or artifacts")


def load_canonical(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    plain_file(path, label=label)
    raw = path.read_bytes()
    return decode_canonical(raw, label=label), raw


def decode_canonical(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=no_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        die(f"cannot decode {label}: {error}")
    if type(value) is not dict or raw != canonical(value) + b"\n":
        die(f"{label} is not canonical JSON")
    return value


def verify_seal(
    value: dict[str, Any], *, field: str, expected: str | None, label: str
) -> str:
    unsigned = dict(value)
    declared = unsigned.pop(field, None)
    if (
        type(declared) is not str
        or SHA256.fullmatch(declared) is None
        or (expected is not None and declared != expected)
        or sha_bytes(canonical(unsigned)) != declared
    ):
        die(f"{label} object seal differs")
    return declared


def validate_release_manifest(
    raw: bytes, *, expected_sha256: str, expected_digest: str,
    postflight_raw: bytes,
) -> dict[str, Any]:
    if (
        SHA256.fullmatch(expected_sha256) is None
        or SHA256.fullmatch(expected_digest) is None
        or sha_bytes(raw) != expected_sha256
    ):
        die("review release manifest external pin differs")
    value = decode_canonical(raw, label="review release manifest")
    digest = verify_seal(
        value, field="receipt_digest", expected=expected_digest,
        label="review release manifest",
    )
    inputs = value.get("inputs")
    formal_inputs = value.get("formal_inputs")
    executables = value.get("executables")
    expected_paths = {
        "manifest_materializer": EXPECTED_MANIFEST_MATERIALIZER,
        "adapter": EXPECTED_ADAPTER,
        "launcher": EXPECTED_LAUNCHER,
        "submitter": EXPECTED_SUBMITTER,
        "postflight": EXPECTED_POSTFLIGHT,
        "hostile": EXPECTED_HOSTILE,
        "source_archive": EXPECTED_SOURCE_ARCHIVE,
    }
    executable_paths = {
        "python": EXPECTED_PYTHON, "ffmpeg": EXPECTED_FFMPEG,
        "compute_bash": EXPECTED_COMPUTE_BASH, "sacct": EXPECTED_SACCT,
    }
    formal_paths = {
        "master_receipt": EXPECTED_FORMAL_MASTER,
        "submission_receipt": EXPECTED_FORMAL_SUBMISSION,
    }
    release_info = RELEASE_ROOT.lstat()
    inputs_info = INPUTS.lstat()
    if (
        set(value) != {
            "schema_version", "status", "release_root", "inputs",
            "formal_inputs", "executables", "authority", "receipt_digest",
        }
        or value.get("schema_version")
        != "saic-t2v-topup-review-v2-release-manifest-v1"
        or value.get("status") != "sealed_before_review_submission"
        or value.get("release_root") != str(RELEASE_ROOT)
        or value.get("authority") != RELEASE_AUTHORITY
        or digest != expected_digest
        or type(inputs) is not dict
        or set(inputs) != set(expected_paths)
        or type(executables) is not dict
        or set(executables) != set(executable_paths)
        or type(formal_inputs) is not dict
        or set(formal_inputs) != set(formal_paths)
        or RELEASE_ROOT.resolve(strict=True) != RELEASE_ROOT
        or INPUTS.resolve(strict=True) != INPUTS
        or not stat.S_ISDIR(release_info.st_mode)
        or not stat.S_ISDIR(inputs_info.st_mode)
        or stat.S_ISLNK(release_info.st_mode)
        or stat.S_ISLNK(inputs_info.st_mode)
        or release_info.st_uid != os.getuid()
        or inputs_info.st_uid != os.getuid()
        or stat.S_IMODE(release_info.st_mode) & 0o022
        or stat.S_IMODE(inputs_info.st_mode) & 0o022
        or set(INPUTS.iterdir()) != set(expected_paths.values())
        or set(RELEASE_ROOT.iterdir()) != {INPUTS, EXPECTED_RELEASE_MANIFEST}
    ):
        die("review release manifest schema differs")
    for name, path in expected_paths.items():
        binding = inputs[name]
        if (
            type(binding) is not dict
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != str(path)
            or SHA256.fullmatch(str(binding.get("sha256"))) is None
        ):
            die(f"review release {name} binding differs")
        plain_file(path, label=f"review release {name}")
        observed = sha_bytes(postflight_raw) if name == "postflight" else sha_file(path)
        if observed != binding["sha256"]:
            die(f"review release {name} bytes differ")
    for name, path in executable_paths.items():
        binding = executables[name]
        info = path.lstat()
        if (
            type(binding) is not dict
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != str(path)
            or SHA256.fullmatch(str(binding.get("sha256"))) is None
            or path.resolve(strict=True) != path
            or not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) & 0o022
            or not os.access(path, os.X_OK)
            or sha_file(path) != binding["sha256"]
        ):
            die(f"review release executable {name} differs")
    for name, path in formal_paths.items():
        binding = formal_inputs[name]
        if (
            type(binding) is not dict
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != str(path)
            or SHA256.fullmatch(str(binding.get("sha256"))) is None
        ):
            die(f"review release formal input {name} binding differs")
        plain_file(path, label=f"review release formal input {name}")
        if sha_file(path) != binding["sha256"]:
            die(f"review release formal input {name} bytes differ")
    if (
        inputs["adapter"]["sha256"] != EXPECTED_ADAPTER_SHA256
        or inputs["launcher"]["sha256"] != EXPECTED_LAUNCHER_SHA256
        or inputs["source_archive"]["sha256"]
        != EXPECTED_SOURCE_ARCHIVE_SHA256
        or executables["python"]["sha256"] != EXPECTED_PYTHON_SHA256
        or executables["ffmpeg"]["sha256"] != EXPECTED_FFMPEG_SHA256
        or executables["compute_bash"]["sha256"]
        != EXPECTED_COMPUTE_BASH_SHA256
        or executables["sacct"]["sha256"] != EXPECTED_SACCT_SHA256
    ):
        die("review release independently pinned identities differ")
    return value


def validate_blank_template(
    value: Any,
    *,
    slot: int,
    expected_review_ids: Sequence[str],
    protocol_binding: Mapping[str, Any],
) -> None:
    if type(value) is not dict or set(value) != BLANK_TEMPLATE_FIELDS:
        die("blank observer template schema differs")
    responses = value.get("responses")
    if (
        value.get("schema_version")
        != "bernini-saic-t2v-event-bank-topup-independent-observer-blank-template-v2"
        or value.get("packet_id") != PACKET_ID
        or value.get("observer_slot") != slot
        or value.get("template_only") is not True
        or value.get("semantic_status") != "UNASSESSED"
        or value.get("observer_id") is not None
        or value.get("observer_kind") is not None
        or value.get("observer_authority_artifact") is not None
        or value.get("observer_protocol_artifact") != protocol_binding
        or value.get("completed_at") is not None
        or value.get("independent_observer_required") is not True
        or value.get("same_person_must_not_fill_both_slots") is not True
        or value.get("copy_outside_sealed_packet_before_completion") is not True
        or value.get("blindness_or_independence_established_by_template")
        is not False
        or SHA256.fullmatch(str(value.get("review_item_set_digest"))) is None
        or value.get("review_item_set_digest")
        != protocol_binding.get("review_item_set_digest")
        or value.get("authority") != FALSE_AUTHORITY
        or type(responses) is not list
        or len(responses) != 60
    ):
        die("blank observer template contract differs")
    for expected_id, response in zip(expected_review_ids, responses, strict=True):
        if (
            type(response) is not dict
            or set(response) != BLANK_RESPONSE_FIELDS
            or response.get("review_item_id") != expected_id
            or any(
                response[field] is not None
                for field in BLANK_RESPONSE_FIELDS - {"review_item_id"}
            )
        ):
            die("blank observer response differs")


def validate_formal_receipts(
    master_raw: bytes,
    submission_raw: bytes,
    *,
    formal_bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        sha_bytes(master_raw) != formal_bundle.get("master_receipt_sha256")
        or sha_bytes(submission_raw)
        != formal_bundle.get("submission_receipt_sha256")
    ):
        die("formal retained receipt file hash differs")
    master = decode_canonical(master_raw, label="formal master receipt")
    submission = decode_canonical(
        submission_raw, label="formal submission receipt"
    )
    verify_seal(
        master, field="receipt_digest",
        expected=formal_bundle.get("master_receipt_digest"),
        label="formal master receipt",
    )
    verify_seal(
        submission, field="receipt_digest",
        expected=formal_bundle.get("submission_receipt_digest"),
        label="formal submission receipt",
    )
    attempts = master.get("attempts")
    proofs = master.get("same_seed_official_gaussian_proofs")
    if (
        set(master) != FORMAL_MASTER_FIELDS
        or master.get("schema_version")
        != "bernini-saic-pure-t2v-event-bank-topup-receipt-v2"
        or master.get("bank_id")
        != "saic-text-only-hard-negative-topup-exact81-v2"
        or master.get("top_up_only") is not True
        or master.get("root_spec_raw_sha256")
        != EXPECTED_EVENT_SPEC_RAW_SHA256
        or master.get("base_v1_spec_raw_sha256")
        != EXPECTED_BASE_V1_SPEC_RAW_SHA256
        or master.get("base_v1_spec_content_sha256")
        != EXPECTED_BASE_V1_SPEC_CONTENT_SHA256
        or master.get("source_manifest_content_sha256")
        != EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256
        or master.get("topology")
        != "two_concurrent_world4_sp4_groups_on_one_8gpu_node"
        or master.get("attempt_count") != 60
        or master.get("row_count") != 8
        or master.get("seed_cell_count") != 20
        or master.get("branch_order")
        != ["incomplete", "camera_only", "appearance_only"]
        or master.get("merged_branch_order")
        != [
            "forward", "reverse", "noop", "incomplete", "camera_only",
            "appearance_only",
        ]
        or master.get("six_branch_spec_merge_cell_count") != 20
        or any(
            master.get(field) is not False
            for field in (
                "detached_full81_event_review_complete", "event_verified",
                "identity_preservation_verified", "seed_selection_authorized",
                "training_target_authorized",
                "optimizer_or_parameter_update_authorized",
            )
        )
        or type(attempts) is not list
        or len(attempts) != 60
        or type(proofs) is not list
        or len(proofs) != 20
        or any(
            type(master.get(field)) is not dict
            for field in (
                "sampling_contract", "semantic_input_closure",
                "geometry_proxy_contract", "artifact_authority",
            )
        )
    ):
        die("formal master exact schema/header differs")
    candidate_ids: set[str] = set()
    attempt_cells: set[tuple[str, int]] = set()
    cell_branches: dict[tuple[str, int], set[str]] = {}
    source_rows: set[str] = set()
    for attempt in attempts:
        if type(attempt) is not dict:
            die("formal master attempt closure differs")
        candidate_id = str(attempt.get("candidate_id", ""))
        if (
            set(attempt) != FORMAL_MASTER_ATTEMPT_FIELDS
            or re.fullmatch(
                r"saic-topup-v2-[0-9a-f]{16}-"
                r"(?:incomplete|camera_only|appearance_only)-s[0-9]+",
                candidate_id,
            ) is None
            or candidate_id in candidate_ids
            or re.fullmatch(
                r"(?:fit|confirmation)-(?:dog|human)-0[01]-[0-9a-f]{16}",
                str(attempt.get("row_id", "")),
            ) is None
            or not str(attempt.get("row_id", "")).endswith(
                "-" + str(attempt.get("iid", ""))
            )
            or attempt.get("analysis_split") not in {"fit", "confirmation"}
            or attempt.get("branch") not in {
                "incomplete", "camera_only", "appearance_only",
            }
            or type(attempt.get("seed")) is not int
            or attempt.get("seed") < 0
            or Path(str(attempt.get("receipt_path")))
            != EXPECTED_FORMAL_ROOT / "attempts" / candidate_id
                / "saic-event-topup-generation-receipt.json"
            or Path(str(attempt.get("mp4_path")))
            != EXPECTED_FORMAL_ROOT / "attempts" / candidate_id / "t2v.mp4"
            or any(
                SHA256.fullmatch(str(attempt.get(field))) is None
                for field in (
                    "receipt_sha256", "receipt_digest", "mp4_sha256",
                )
            )
            or attempt.get("event_audit_status")
            != "pending_detached_full81_review"
            or not candidate_id.endswith(
                f"-{attempt.get('branch')}-s{attempt.get('seed')}"
            )
        ):
            die("formal master attempt closure differs")
        candidate_ids.add(candidate_id)
        attempt_cells.add((str(attempt["iid"]), int(attempt["seed"])))
        cell_branches.setdefault(
            (str(attempt["iid"]), int(attempt["seed"])), set()
        ).add(str(attempt["branch"]))
        source_rows.add(str(attempt["row_id"]))
    proof_cells: set[tuple[str, int]] = set()
    for proof in proofs:
        if type(proof) is not dict or type(proof.get("seed")) is not int:
            die("formal master same-Gaussian proof closure differs")
        cell = (str(proof.get("iid", "")), proof["seed"])
        if (
            set(proof) != FORMAL_GAUSSIAN_PROOF_FIELDS
            or cell in proof_cells
            or proof.get("branch_order")
            != ["incomplete", "camera_only", "appearance_only"]
            or proof.get("official_gaussian_tensor_values_byte_equal") is not True
            or SHA256.fullmatch(
                str(proof.get("official_gaussian_identity_digest"))
            ) is None
        ):
            die("formal master same-Gaussian proof closure differs")
        proof_cells.add(cell)
    if (
        attempt_cells != proof_cells
        or len(source_rows) != 8
        or any(
            branches != {"incomplete", "camera_only", "appearance_only"}
            for branches in cell_branches.values()
        )
    ):
        die("formal master seed-cell proof mapping differs")

    submitted = submission.get("submitted_job")
    boundary = submission.get("submission_boundary")
    inputs = submission.get("inputs")
    if (
        set(submission) != FORMAL_SUBMISSION_FIELDS
        or submission.get("schema_version")
        != "saic-t2v-topup-r6-formal-v2-r2-submission-v1"
        or submission.get("status") != "submitted"
        or submission.get("submission_success") is not True
        or submission.get("job_success") is not None
        or type(submitted) is not dict
        or set(submitted) != {
            "job_id", "cluster", "stdout_sha256", "stderr_sha256",
        }
        or submitted.get("job_id") != formal_bundle.get("job_id")
        or (
            submitted.get("cluster") is not None
            and (
                type(submitted.get("cluster")) is not str
                or not submitted["cluster"]
                or "\n" in submitted["cluster"]
                or ";" in submitted["cluster"]
            )
        )
        or any(
            SHA256.fullmatch(str(submitted.get(field))) is None
            for field in ("stdout_sha256", "stderr_sha256")
        )
        or submission.get("request") != {
            "job_name": "saic-t2v-topup-r6-v2-r2",
            "partition": "faculty", "qos": "bgqos", "nodes": 1,
            "ntasks": 1, "cpus_per_task": 32, "memory": "256G",
            "walltime": "24:00:00",
            "gpu_resource_requested": "gpu:mi210:8",
            "world_topology": "two_concurrent_world4_sp4",
            "candidate_count": 60,
            "dynamic_plan_schema_version":
                "saic-t2v-topup-rendezvous-dynamic-plan-v2",
            "fixed_candidate_set_and_order": True,
            "scientific_spec_changed_for_rendezvous_guard_v2": False,
            "hold": False, "dependency": None,
        }
        or type(boundary) is not dict
        or set(boundary) != {
            "environment_replaced", "exact_job_export_names", "export_all",
            "reservation_created_before_sbatch", "same_inode_retained",
            "launcher_submitted_from_retained_fd",
            "runtime_retained_fd_admission_roots",
            "pathname_replacement_resistant_admission_handoff",
            "compute_bash_exact_identity_pinned",
            "varredir_close_option_required", "reservation_device",
            "reservation_inode", "success_mode",
        }
        or boundary.get("environment_replaced") is not True
        or boundary.get("exact_job_export_names") != FORMAL_EXPORT_NAMES
        or boundary.get("export_all") is not False
        or boundary.get("reservation_created_before_sbatch") is not True
        or boundary.get("same_inode_retained") is not True
        or boundary.get("launcher_submitted_from_retained_fd") is not True
        or boundary.get("runtime_retained_fd_admission_roots") != [
            "formal_gate", "effective_launcher", "rendezvous_guard_v2",
            "compute_bash_probe_validator",
        ]
        or boundary.get("pathname_replacement_resistant_admission_handoff")
        is not True
        or boundary.get("compute_bash_exact_identity_pinned") is not True
        or boundary.get("varredir_close_option_required") is not False
        or type(boundary.get("reservation_device")) is not int
        or boundary["reservation_device"] < 0
        or type(boundary.get("reservation_inode")) is not int
        or boundary["reservation_inode"] <= 0
        or boundary.get("success_mode") != "0444"
        or type(inputs) is not dict
        or set(inputs) != FORMAL_SUBMISSION_INPUT_FIELDS
        or inputs.get("effective_dynamic_plan_schema_version")
        != "saic-t2v-topup-rendezvous-dynamic-plan-v2"
        or inputs.get("scientific_spec_changed_for_rendezvous_guard_v2") is not False
        or inputs.get("event_spec_sha256") != EXPECTED_EVENT_SPEC_RAW_SHA256
        or inputs.get("event_spec_sha256") != master.get("root_spec_raw_sha256")
        or inputs.get("source_manifest_sha256")
        != "899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9"
        or submission.get("outputs") != {
            "output_root": str(EXPECTED_FORMAL_ROOT),
            "submission_receipt": str(EXPECTED_FORMAL_SUBMISSION),
            "slurm_log_dir": str(EXPECTED_FORMAL_SLURM_LOG_DIR),
            "fresh_before_submission": True,
        }
        or submission.get("authority") != {
            "diagnostic_event_bank_execution_authorized": True,
            "training": False, "checkpoint": False,
            "scientific_success_claimed": False,
            "action_edit_success_claimed": False,
            "job_success_claimed": False,
        }
        or type(submission.get("canary_admission")) is not dict
        or submission.get("threat_model") != FORMAL_THREAT_MODEL
    ):
        die("formal submission exact closure differs")
    validate_formal_three_gate_bundle(submission)
    return master, submission


def validate_formal_three_gate_bundle(submission: Mapping[str, Any]) -> None:
    inputs = submission["inputs"]
    canary = submission.get("canary_admission")
    if type(canary) is not dict or set(canary) != FORMAL_CANARY_ADMISSION_FIELDS:
        die("formal three-gate admission schema differs")
    probe = canary.get("compute_bash_probe_admission")
    retained = canary.get("retained_fd_world8")
    compute_bash = probe.get("compute_bash") if type(probe) is dict else None
    expected_compute_bash = {
        "path": "/usr/bin/bash",
        "sha256": inputs.get("compute_bash_sha256"),
        "version_stdout_sha256":
            inputs.get("compute_bash_version_stdout_sha256"),
        "version_first_line": inputs.get("compute_bash_version_first_line"),
        "brace_fd_redirection_supported": True,
        "retained_fd_survives_bash_script_handoff": True,
        "varredir_close_option_required": False,
    }
    if (
        canary.get("job_id") != "134393"
        or canary.get("terminal_receipt_sha256")
        != "6927a2945fac87622beb167b96bc6e04b2d26d1bc0d957bfd130f379380cbc8d"
        or canary.get("terminal_receipt_digest")
        != "773bb9df35add9319d9dd8ff0d39c7402bdfbbd2a3f2c52a3551bb00c2a5e52b"
        or canary.get("submission_receipt_sha256")
        != "1840f3be3d96573e341c153d498e8447eac9c7250657bdbcae3aa8b1da1246af"
        or canary.get("submission_receipt_digest")
        != "be0ac76378d7bd199c01f8e02fb080ab8db3013be66bcd978b62d18ffed38796"
        or canary.get("slurm_state_required") != "COMPLETED"
        or canary.get("slurm_exit_code_required") != "0:0"
        or canary.get("allocated_gpu_resource_required")
        != "gres/gpu:mi210=8"
        or inputs.get("compute_bash") != "/usr/bin/bash"
        or inputs.get("compute_bash_sha256") != EXPECTED_COMPUTE_BASH_SHA256
        or inputs.get("compute_bash_version_stdout_sha256")
        != EXPECTED_COMPUTE_BASH_VERSION_STDOUT_SHA256
        or inputs.get("compute_bash_version_first_line")
        != EXPECTED_COMPUTE_BASH_VERSION_FIRST_LINE
        or inputs.get("probe_validator_sha256")
        != "3b5e78a0815fbfdee0404293ad800c640133deacc6b1bfafec12a02ca40ef85b"
        or inputs.get("compute_bash_probe_admission")
        != (
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
            "VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/"
            "canaries/compute-bash-retained-fd-probe-8283e73d-r1/"
            "probe-admission.json"
        )
        or inputs.get("compute_bash_probe_admission_sha256")
        != "d51ebf1f894d63483943042faaa2c6ccbf812c0f93769980084bae72f8ab84d8"
        or inputs.get("compute_bash_probe_admission_digest")
        != "a37e44c12f935a4f4e11ab08364b019799809fd13299af6459ca6797a8333fb7"
        or type(probe) is not dict
        or set(probe) != FORMAL_PROBE_BINDING_FIELDS
        or probe.get("path") != inputs.get("compute_bash_probe_admission")
        or probe.get("sha256")
        != inputs.get("compute_bash_probe_admission_sha256")
        or probe.get("receipt_digest")
        != inputs.get("compute_bash_probe_admission_digest")
        or probe.get("schema_version")
        != "saic-compute-bash-retained-fd-probe-admission-v1"
        or probe.get("status")
        != "terminal_completed_compute_bash_retained_fd_admitted"
        or probe.get("slurm_job_id") != "134647"
        or probe.get("sha256")
        != "d51ebf1f894d63483943042faaa2c6ccbf812c0f93769980084bae72f8ab84d8"
        or probe.get("receipt_digest")
        != "a37e44c12f935a4f4e11ab08364b019799809fd13299af6459ca6797a8333fb7"
        or probe.get("compute_bash") != expected_compute_bash
        or probe.get("authority") != FORMAL_PROBE_AUTHORITY
        or any(
            SHA256.fullmatch(str(probe.get(field))) is None
            for field in (
                "submission_receipt_sha256", "submission_receipt_digest",
                "operational_evidence_sha256", "operational_evidence_digest",
                "release_manifest_file_sha256", "release_manifest_digest",
                "wrapper_sha256", "postflight_sha256",
            )
        )
        or type(retained) is not dict
        or set(retained) != FORMAL_RETAINED_WORLD8_FIELDS
        or retained.get("job_id") != inputs.get("retained_fd_canary_job_id")
        or retained.get("admission_path")
        != inputs.get("retained_fd_canary_admission")
        or retained.get("admission_sha256")
        != inputs.get("retained_fd_canary_admission_sha256")
        or retained.get("admission_digest")
        != inputs.get("retained_fd_canary_admission_digest")
        or retained.get("guard_sha256")
        != inputs.get("rendezvous_guard_sha256")
        or retained.get("runtime_sha256")
        != inputs.get("generation_runtime_sha256")
        or retained.get("probe_validator_sha256")
        != inputs.get("probe_validator_sha256")
        or retained.get("probe_admission_binding") != probe
        or retained.get("compute_bash") != compute_bash
        or retained.get("slurm_state_required") != "COMPLETED"
        or retained.get("slurm_exit_code_required") != "0:0"
        or retained.get("allocated_gpu_resource_required")
        != "gres/gpu:mi210=8"
        or retained.get("science_generation_entered") is not False
        or retained.get("formal_submission_authorized_by_canary_alone")
        is not False
    ):
        die("formal three-gate admission binding differs")
    validate_formal_gate_sacct(
        canary.get("sacct_observation"), job_id="134393",
        alloc_tres=(
            "billing=32,cpu=32,gres/gpu:mi210=8,"
            "gres/gpu=8,mem=64G,node=1"
        ),
        phase="submitter_before_formal_sbatch", precheck=True,
    )
    validate_formal_gate_sacct(
        retained.get("external_postflight_sacct_observation"),
        job_id=str(retained["job_id"]),
        alloc_tres=(
            "billing=16,cpu=16,gres/gpu:mi210=8,"
            "gres/gpu=8,mem=32G,node=1"
        ),
        phase="external_postflight_after_canary_terminal", precheck=False,
        submit_line_required=True,
    )
    validate_formal_gate_sacct(
        retained.get("submitter_sacct_observation"),
        job_id=str(retained["job_id"]),
        alloc_tres=(
            "billing=16,cpu=16,gres/gpu:mi210=8,"
            "gres/gpu=8,mem=32G,node=1"
        ),
        phase="submitter_before_formal_sbatch", precheck=False,
        submit_line_required=True,
    )
    external_observation = retained["external_postflight_sacct_observation"]
    submitter_observation = retained["submitter_sacct_observation"]
    if (
        external_observation.get("stdout_sha256")
        != submitter_observation.get("stdout_sha256")
        or external_observation.get("parsed_row")
        != submitter_observation.get("parsed_row")
        or external_observation.get("submit_line_sha256")
        != submitter_observation.get("submit_line_sha256")
        or external_observation.get("retained_wrapper_fd")
        != submitter_observation.get("retained_wrapper_fd")
        or external_observation.get("exact_submit_line")
        != submitter_observation.get("exact_submit_line")
    ):
        die("formal retained-WORLD8 sacct observations diverge")


def validate_formal_gate_sacct(
    value: Any, *, job_id: str, alloc_tres: str, phase: str,
    precheck: bool, submit_line_required: bool = False,
) -> None:
    expected_fields = {
        "executable", "executable_sha256", "argv", "query_fields",
        "returncode", "stdout_sha256", "stderr_sha256", "parsed_row",
        "exact_single_row", "observation_phase",
    }
    if precheck:
        expected_fields.add("precheck_passed_before_only_sbatch")
    if submit_line_required:
        expected_fields.update({
            "submit_line_sha256", "retained_wrapper_fd", "exact_submit_line",
        })
    row = value.get("parsed_row") if type(value) is dict else None
    expected_row_fields = {
        "JobIDRaw", "State", "ExitCode", "AllocTRES", "NodeList",
        "Start", "End", "Elapsed", "SubmitLine",
    } if submit_line_required else {
        "JobIDRaw", "State", "ExitCode", "AllocTRES", "NodeList",
        "ElapsedRaw", "Start", "End",
    }
    query_fields = (
        [
            "JobIDRaw", "State", "ExitCode", "AllocTRES%512", "NodeList",
            "Start", "End", "Elapsed", "SubmitLine%8192",
        ]
        if submit_line_required else
        [
            "JobIDRaw", "State", "ExitCode", "AllocTRES%512", "NodeList",
            "ElapsedRaw", "Start", "End",
        ]
    )
    argv = [
        "/usr/bin/sacct", "-j", job_id, "-X", "--noheader", "-n", "-P",
        "-o", ",".join(query_fields),
    ]
    submit_line = str(row.get("SubmitLine", "")) if type(row) is dict else ""
    retained_match = re.fullmatch(
        r".* /proc/self/fd/([0-9]+)", submit_line,
    )
    if (
        type(value) is not dict
        or set(value) != expected_fields
        or value.get("executable") != "/usr/bin/sacct"
        or value.get("executable_sha256") != EXPECTED_SACCT_SHA256
        or value.get("argv") != argv
        or value.get("query_fields") != query_fields
        or value.get("returncode") != 0
        or value.get("stderr_sha256") != sha_bytes(b"")
        or SHA256.fullmatch(str(value.get("stdout_sha256"))) is None
        or value.get("exact_single_row") is not True
        or value.get("observation_phase") != phase
        or (precheck and value.get("precheck_passed_before_only_sbatch") is not True)
        or type(row) is not dict
        or set(row) != expected_row_fields
        or row.get("JobIDRaw") != job_id
        or row.get("State") != "COMPLETED"
        or row.get("ExitCode") != "0:0"
        or row.get("AllocTRES") != alloc_tres
        or not row.get("NodeList")
        or row.get("NodeList") in {"None assigned", "Unknown"}
        or not row.get("Start")
        or not row.get("End")
        or (
            submit_line_required
            and (
                not row.get("Elapsed")
                or "/proc/self/fd/" not in str(row.get("SubmitLine", ""))
                or "--export=NONE," not in str(row.get("SubmitLine", ""))
                or retained_match is None
                or str(int(retained_match.group(1))) != retained_match.group(1)
                or int(retained_match.group(1)) < 3
                or value.get("submit_line_sha256")
                != sha_bytes(submit_line.encode("ascii"))
                or value.get("retained_wrapper_fd")
                != int(retained_match.group(1))
                or value.get("exact_submit_line") is not True
            )
        )
        or (
            not submit_line_required
            and not str(row.get("ElapsedRaw", "")).isdigit()
        )
    ):
        die("formal three-gate sacct observation differs")


def expected_exports(submission: dict[str, Any]) -> dict[str, str]:
    formal = submission["formal_terminal_input_bundle"]
    inputs = submission["inputs"]
    exports = {
        "SAIC_T2V_TOPUP_REVIEW_RELEASE_ROOT": str(RELEASE_ROOT),
        "SAIC_T2V_TOPUP_REVIEW_SOURCE_ARCHIVE": str(EXPECTED_SOURCE_ARCHIVE),
        "SAIC_T2V_TOPUP_REVIEW_SOURCE_ARCHIVE_SHA256":
            EXPECTED_SOURCE_ARCHIVE_SHA256,
        "SAIC_T2V_TOPUP_REVIEW_SOURCE_REVISION": EXPECTED_SOURCE_REVISION,
        "SAIC_T2V_TOPUP_REVIEW_ADAPTER": str(EXPECTED_ADAPTER),
        "SAIC_T2V_TOPUP_REVIEW_ADAPTER_SHA256": inputs["adapter_sha256"],
        "SAIC_T2V_TOPUP_REVIEW_INPUT_ROOT": str(EXPECTED_FORMAL_ROOT),
        "SAIC_T2V_TOPUP_REVIEW_FORMAL_MASTER": str(EXPECTED_FORMAL_MASTER),
        "SAIC_T2V_TOPUP_REVIEW_FORMAL_MASTER_SHA256":
            formal["master_receipt_sha256"],
        "SAIC_T2V_TOPUP_REVIEW_FORMAL_MASTER_DIGEST":
            formal["master_receipt_digest"],
        "SAIC_T2V_TOPUP_REVIEW_FORMAL_SUBMISSION":
            str(EXPECTED_FORMAL_SUBMISSION),
        "SAIC_T2V_TOPUP_REVIEW_FORMAL_SUBMISSION_SHA256":
            formal["submission_receipt_sha256"],
        "SAIC_T2V_TOPUP_REVIEW_FORMAL_SUBMISSION_DIGEST":
            formal["submission_receipt_digest"],
        "SAIC_T2V_TOPUP_REVIEW_FORMAL_JOB_ID": formal["job_id"],
        "SAIC_T2V_TOPUP_REVIEW_OUTPUT_ROOT": str(OUTPUT_ROOT),
        "SAIC_T2V_TOPUP_REVIEW_AUTOMATION_RECEIPT": str(AUTOMATION_RECEIPT),
        "SAIC_T2V_TOPUP_REVIEW_PYTHON_BIN": str(EXPECTED_PYTHON),
        "SAIC_T2V_TOPUP_REVIEW_PYTHON_SHA256": EXPECTED_PYTHON_SHA256,
        "SAIC_T2V_TOPUP_REVIEW_FFMPEG_BIN": str(EXPECTED_FFMPEG),
        "SAIC_T2V_TOPUP_REVIEW_FFMPEG_SHA256": EXPECTED_FFMPEG_SHA256,
        "SAIC_T2V_TOPUP_REVIEW_FFPROBE_WRAPPER_SHA256":
            EXPECTED_FFPROBE_WRAPPER_SHA256,
        "SAIC_T2V_TOPUP_REVIEW_COMPUTE_BASH": str(EXPECTED_COMPUTE_BASH),
        "SAIC_T2V_TOPUP_REVIEW_COMPUTE_BASH_SHA256":
            inputs["compute_bash_sha256"],
        "SAIC_T2V_TOPUP_REVIEW_COMPUTE_BASH_VERSION_STDOUT_SHA256":
            inputs["compute_bash_version_stdout_sha256"],
        "SAIC_T2V_TOPUP_REVIEW_WORKERS": "16",
    }
    if list(exports) != EXPORT_NAMES or any(
        "," in key or "," in value or "\n" in key or "\n" in value
        for key, value in exports.items()
    ):
        die("review exact export transport differs")
    return exports


def expected_submit_line(submission: dict[str, Any]) -> str:
    descriptor = submission["submission_boundary"]["retained_launcher_fd"]
    exports = expected_exports(submission)
    return " ".join(
        [
            "/usr/bin/sbatch",
            "--parsable",
            f"--output={EXPECTED_SLURM_LOG_DIR}/"
            "saic-t2v-topup-review-v2-%j.out",
            f"--error={EXPECTED_SLURM_LOG_DIR}/"
            "saic-t2v-topup-review-v2-%j.err",
            "--export=NONE," + ",".join(
                f"{name}={value}" for name, value in exports.items()
            ),
            f"/proc/self/fd/{descriptor}",
        ]
    )


def validate_formal_sacct_observation(value: Any, *, job_id: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "executable", "executable_sha256", "argv", "query_fields",
        "returncode", "stdout_sha256", "stderr_sha256", "parsed_row",
        "exact_single_row", "observation_phase",
    }:
        die("formal submitter sacct observation closure differs")
    row = value.get("parsed_row")
    if type(row) is not dict or set(row) != {
        "JobIDRaw", "State", "ExitCode", "AllocTRES", "NodeList",
        "ElapsedRaw", "Start", "End",
    }:
        die("formal submitter sacct row closure differs")
    alloc: dict[str, str] = {}
    for token in str(row.get("AllocTRES", "")).split(","):
        if "=" not in token:
            die("formal submitter AllocTRES differs")
        key, item = token.split("=", 1)
        if not key or key in alloc:
            die("formal submitter AllocTRES closure differs")
        alloc[key] = item
    expected_argv = [
        str(EXPECTED_SACCT), "-j", job_id, "-X", "--noheader", "-n",
        "-P", "-o", ",".join(FORMAL_SACCT_FIELDS),
    ]
    if (
        value.get("executable") != str(EXPECTED_SACCT)
        or value.get("executable_sha256") != EXPECTED_SACCT_SHA256
        or value.get("argv") != expected_argv
        or value.get("query_fields") != FORMAL_SACCT_FIELDS
        or value.get("returncode") != 0
        or SHA256.fullmatch(str(value.get("stdout_sha256"))) is None
        or value.get("stderr_sha256") != sha_bytes(b"")
        or value.get("exact_single_row") is not True
        or value.get("observation_phase")
        != "review_submitter_after_formal_terminal_before_only_sbatch"
        or row.get("JobIDRaw") != job_id
        or row.get("State") != "COMPLETED"
        or row.get("ExitCode") != "0:0"
        or alloc != EXPECTED_FORMAL_ALLOC_TRES
        or not row.get("NodeList")
        or row.get("NodeList") in {"None assigned", "Unknown"}
        or not str(row.get("ElapsedRaw", "")).isdigit()
        or not row.get("Start")
        or row.get("Start") == "Unknown"
        or not row.get("End")
        or row.get("End") == "Unknown"
    ):
        die("formal submitter sacct observation differs")
    return value


def exact_sacct(job_id: str, submission: dict[str, Any]) -> dict[str, Any]:
    info = EXPECTED_SACCT.lstat()
    if (
        EXPECTED_SACCT.resolve(strict=True) != EXPECTED_SACCT
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != 0
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) & 0o022
        or not os.access(EXPECTED_SACCT, os.X_OK)
        or sha_file(EXPECTED_SACCT) != EXPECTED_SACCT_SHA256
    ):
        die("root-owned sacct executable differs")
    argv = [
        str(EXPECTED_SACCT), "-j", job_id, "-X", "--noheader", "-n",
        "-P", "-o", ",".join(SACCT_FIELDS),
    ]
    completed = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    try:
        stdout = completed.stdout.decode("ascii")
    except UnicodeDecodeError:
        die("review sacct stdout is not ASCII")
    lines = stdout.splitlines()
    fields = lines[0].split("|") if len(lines) == 1 else []
    keys = [field.split("%", 1)[0] for field in SACCT_FIELDS]
    row = dict(zip(keys, fields, strict=True)) if len(fields) == len(keys) else {}
    alloc: dict[str, str] = {}
    for token in str(row.get("AllocTRES", "")).split(","):
        if "=" not in token:
            die("review AllocTRES token differs")
        key, value = token.split("=", 1)
        if not key or key in alloc:
            die("review AllocTRES closure differs")
        alloc[key] = value
    submit_line = str(row.get("SubmitLine", ""))
    exact_submit = expected_submit_line(submission)
    if (
        completed.returncode != 0
        or completed.stderr
        or len(lines) != 1
        or row.get("JobIDRaw") != job_id
        or row.get("State") != "COMPLETED"
        or row.get("ExitCode") != "0:0"
        or alloc != EXPECTED_REVIEW_ALLOC_TRES
        or not row.get("NodeList")
        or row.get("NodeList") in {"None assigned", "Unknown"}
        or not str(row.get("ElapsedRaw", "")).isdigit()
        or not row.get("Start")
        or row.get("Start") == "Unknown"
        or not row.get("End")
        or row.get("End") == "Unknown"
        or submit_line != exact_submit
    ):
        die("review job is not exact terminal success")
    return {
        "executable": str(EXPECTED_SACCT),
        "executable_sha256": EXPECTED_SACCT_SHA256,
        "argv": argv,
        "query_fields": SACCT_FIELDS,
        "returncode": completed.returncode,
        "stdout_sha256": sha_bytes(completed.stdout),
        "stderr_sha256": sha_bytes(completed.stderr),
        "parsed_row": row,
        "submit_line_sha256": sha_bytes(submit_line.encode("ascii")),
        "retained_launcher_fd": submission["submission_boundary"][
            "retained_launcher_fd"
        ],
        "exact_submit_line": True,
        "exact_single_row": True,
        "observation_phase": "external_postflight_after_review_terminal",
    }


def validate_slurm_logs(
    job_id: str,
) -> tuple[
    dict[str, Any], dict[str, tuple[int, Path, bytes, os.stat_result]],
]:
    log_info = EXPECTED_SLURM_LOG_DIR.lstat()
    if (
        EXPECTED_SLURM_LOG_DIR.resolve(strict=True) != EXPECTED_SLURM_LOG_DIR
        or not stat.S_ISDIR(log_info.st_mode)
        or stat.S_ISLNK(log_info.st_mode)
        or log_info.st_uid != os.getuid()
        or stat.S_IMODE(log_info.st_mode) & 0o022
    ):
        die("review Slurm log directory differs")
    paths = {
        "stdout": EXPECTED_SLURM_LOG_DIR
        / f"saic-t2v-topup-review-v2-{job_id}.out",
        "stderr": EXPECTED_SLURM_LOG_DIR
        / f"saic-t2v-topup-review-v2-{job_id}.err",
    }
    if set(EXPECTED_SLURM_LOG_DIR.iterdir()) != set(paths.values()):
        die("review Slurm log exact namespace differs")
    result: dict[str, Any] = {}
    retained: dict[str, tuple[int, Path, bytes, os.stat_result]] = {}
    try:
        for label, path in paths.items():
            descriptor, raw, info = retained_plain_bytes(path, label=f"Slurm {label}")
            retained[label] = (descriptor, path, raw, info)
            result[label] = {
                "path": str(path), "sha256": sha_bytes(raw), "size": info.st_size,
            }
        if result["stderr"]["size"] != 0 or retained["stderr"][2] != b"":
            die("review Slurm stderr is not exactly empty")
        return result, retained
    except BaseException:
        for descriptor, _, _, _ in retained.values():
            os.close(descriptor)
        raise


def validate_submission(
    raw: bytes, retained_info: os.stat_result,
    *, release_manifest: Mapping[str, Any], release_manifest_sha256: str,
    release_manifest_digest: str, formal_master_raw: bytes,
    formal_submission_raw: bytes,
) -> tuple[
    dict[str, Any], bytes, str, dict[str, Any], dict[str, Any],
]:
    value = decode_canonical(raw, label="review submission receipt")
    digest = verify_seal(
        value, field="receipt_digest", expected=None, label="review submission receipt"
    )
    job = value.get("submitted_job", {})
    request = value.get("request", {})
    outputs = value.get("outputs", {})
    boundary = value.get("submission_boundary", {})
    formal = value.get("formal_terminal_input_bundle", {})
    inputs = value.get("inputs", {})
    backfill = value.get("automatic_backfill_contract", {})
    authority = value.get("authority", {})
    if (
        set(value) != {
            "schema_version", "status", "submission_success", "job_success",
            "submitted_job", "request", "submission_boundary",
            "formal_terminal_input_bundle", "inputs", "outputs",
            "automatic_backfill_contract", "authority", "receipt_digest",
        }
        or value.get("schema_version")
        != "saic-t2v-topup-detached-review-v2-submission-v1"
        or value.get("status") != "submitted"
        or value.get("submission_success") is not True
        or value.get("job_success") is not None
        or type(job) is not dict
        or set(job) != {"job_id", "cluster", "stdout_sha256", "stderr_sha256"}
        or type(job.get("job_id")) is not str
        or not job["job_id"].isdigit()
        or SHA256.fullmatch(str(job.get("stdout_sha256"))) is None
        or SHA256.fullmatch(str(job.get("stderr_sha256"))) is None
        or request != {
            "job_name": "saic-t2v-topup-review-v2",
            "partition": "faculty", "qos": "bgqos", "nodes": 1,
            "ntasks": 1, "cpus_per_task": 32, "memory": "192G",
            "walltime": "08:00:00", "gpu_resource_requested": None,
            "candidate_count": 60, "technical_diagnostic_count": 60,
            "hold": False, "dependency": None,
        }
        or type(boundary) is not dict
        or set(boundary) != {
            "environment_replaced", "exact_job_export_names",
            "comma_bearing_compute_bash_first_line_not_exported",
            "export_all", "reservation_created_before_sbatch",
            "same_inode_retained", "launcher_submitted_from_retained_fd",
            "retained_launcher_fd", "reservation_device",
            "reservation_inode", "success_mode",
        }
        or boundary.get("environment_replaced") is not True
        or boundary.get("exact_job_export_names") != EXPORT_NAMES
        or boundary.get("comma_bearing_compute_bash_first_line_not_exported")
        is not True
        or boundary.get("export_all") is not False
        or boundary.get("reservation_created_before_sbatch") is not True
        or boundary.get("same_inode_retained") is not True
        or boundary.get("launcher_submitted_from_retained_fd") is not True
        or type(boundary.get("retained_launcher_fd")) is not int
        or boundary["retained_launcher_fd"] < 3
        or type(boundary.get("reservation_device")) is not int
        or boundary["reservation_device"] < 0
        or type(boundary.get("reservation_inode")) is not int
        or boundary["reservation_inode"] <= 0
        or boundary.get("success_mode") != "0444"
        or boundary.get("reservation_device") != retained_info.st_dev
        or boundary.get("reservation_inode") != retained_info.st_ino
        or type(formal) is not dict
        or set(formal) != {
            "bundle_is_not_a_separate_formal_terminal_admission",
            "job_id", "master_receipt", "master_receipt_sha256",
            "master_receipt_digest", "submission_receipt",
            "submission_receipt_sha256", "submission_receipt_digest",
            "master_attempt_count", "master_branch_order",
            "live_sacct_observation",
        }
        or formal.get("bundle_is_not_a_separate_formal_terminal_admission")
        is not True
        or not str(formal.get("job_id", "")).isdigit()
        or formal.get("master_receipt") != str(EXPECTED_FORMAL_MASTER)
        or formal.get("submission_receipt") != str(EXPECTED_FORMAL_SUBMISSION)
        or formal.get("master_receipt_sha256")
        != release_manifest["formal_inputs"]["master_receipt"]["sha256"]
        or formal.get("submission_receipt_sha256")
        != release_manifest["formal_inputs"]["submission_receipt"]["sha256"]
        or formal.get("master_attempt_count") != 60
        or formal.get("master_branch_order")
        != ["incomplete", "camera_only", "appearance_only"]
        or any(
            SHA256.fullmatch(str(formal.get(field))) is None
            for field in (
                "master_receipt_sha256", "master_receipt_digest",
                "submission_receipt_sha256", "submission_receipt_digest",
            )
        )
        or type(inputs) is not dict
        or set(inputs) != {
            "release_root", "launcher", "launcher_sha256", "adapter",
            "adapter_sha256", "terminal_postflight",
            "terminal_postflight_sha256", "source_archive",
            "source_archive_sha256", "source_revision",
            "formal_submission_receipt_digest", "python", "python_sha256",
            "ffmpeg", "ffmpeg_sha256", "ffprobe_wrapper_sha256",
            "compute_bash", "compute_bash_sha256",
            "compute_bash_version_stdout_sha256",
            "compute_bash_version_first_line",
            "release_manifest", "release_manifest_sha256",
            "release_manifest_digest", "submitter", "submitter_sha256",
            "hostile", "hostile_sha256",
        }
        or inputs.get("release_root") != str(RELEASE_ROOT)
        or inputs.get("launcher") != str(EXPECTED_LAUNCHER)
        or inputs.get("adapter") != str(EXPECTED_ADAPTER)
        or inputs.get("terminal_postflight") != str(EXPECTED_POSTFLIGHT)
        or inputs.get("launcher_sha256")
        != release_manifest["inputs"]["launcher"]["sha256"]
        or inputs.get("adapter_sha256")
        != release_manifest["inputs"]["adapter"]["sha256"]
        or inputs.get("terminal_postflight_sha256")
        != release_manifest["inputs"]["postflight"]["sha256"]
        or inputs.get("source_archive") != str(EXPECTED_SOURCE_ARCHIVE)
        or inputs.get("source_archive_sha256")
        != EXPECTED_SOURCE_ARCHIVE_SHA256
        or inputs.get("source_revision") != EXPECTED_SOURCE_REVISION
        or inputs.get("formal_submission_receipt_digest")
        != formal.get("submission_receipt_digest")
        or inputs.get("python") != str(EXPECTED_PYTHON)
        or inputs.get("python_sha256") != EXPECTED_PYTHON_SHA256
        or inputs.get("ffmpeg") != str(EXPECTED_FFMPEG)
        or inputs.get("ffmpeg_sha256") != EXPECTED_FFMPEG_SHA256
        or inputs.get("ffprobe_wrapper_sha256")
        != EXPECTED_FFPROBE_WRAPPER_SHA256
        or inputs.get("compute_bash") != str(EXPECTED_COMPUTE_BASH)
        or inputs.get("compute_bash_sha256") != EXPECTED_COMPUTE_BASH_SHA256
        or inputs.get("compute_bash_version_stdout_sha256")
        != EXPECTED_COMPUTE_BASH_VERSION_STDOUT_SHA256
        or inputs.get("compute_bash_version_first_line")
        != EXPECTED_COMPUTE_BASH_VERSION_FIRST_LINE
        or inputs.get("release_manifest") != str(EXPECTED_RELEASE_MANIFEST)
        or inputs.get("release_manifest_sha256") != release_manifest_sha256
        or inputs.get("release_manifest_digest") != release_manifest_digest
        or inputs.get("submitter") != str(EXPECTED_SUBMITTER)
        or inputs.get("submitter_sha256")
        != release_manifest["inputs"]["submitter"]["sha256"]
        or inputs.get("hostile") != str(EXPECTED_HOSTILE)
        or inputs.get("hostile_sha256")
        != release_manifest["inputs"]["hostile"]["sha256"]
        or any(
            SHA256.fullmatch(str(inputs.get(field))) is None
            for field in (
                "launcher_sha256", "adapter_sha256",
                "terminal_postflight_sha256", "compute_bash_sha256",
                "compute_bash_version_stdout_sha256", "release_manifest_sha256",
                "release_manifest_digest", "submitter_sha256", "hostile_sha256",
            )
        )
        or outputs != {
            "packet_root": str(OUTPUT_ROOT),
            "submission_receipt": str(SUBMISSION_RECEIPT),
            "automation_receipt": str(AUTOMATION_RECEIPT),
            "terminal_admission": str(TERMINAL_ADMISSION),
            "technical_diagnostics_glob": str(OUTPUT_ROOT / "diagnostics/*.json"),
            "technical_html": str(OUTPUT_ROOT / "index.html"),
            "blind_human_review_html": str(OUTPUT_ROOT / "blind-review.html"),
            "observer_template_glob": str(
                OUTPUT_ROOT / "observer-templates/*.json"
            ),
            "slurm_log_dir": str(EXPECTED_SLURM_LOG_DIR),
            "fresh_before_submission": True,
        }
        or backfill != {
            "technical_packet_may_materialize_automatically": True,
            "blind_stage_public_surface_only": True,
            "technical_html_may_publish_before_two_external_human_seals": False,
            "assessor_private_mapping_may_be_copied_to_stage1": False,
            "automation_receipt_is_runtime_candidate_not_terminal_admission": True,
            "external_review_job_terminal_postflight_required": True,
            "blind_review_is_only_prelabel_ui": True,
            "technical_html_visibility_requires_two_external_human_seals": True,
            "human_templates_must_remain_blank": True,
            "human_labels_may_be_autofilled": False,
            "human_label_ingest_ready_at_submission": False,
            "machine_ingest_ready_at_submission": False,
            "terminal_machine_ingest_scope": "assessor_private_only",
            "human_visible_machine_backfill_ready_at_submission": False,
        }
        or authority != {
            "technical_diagnostic_execution_authorized": True,
            "machine_diagnostics_have_semantic_authority": False,
            "human_review_claimed": False, "event_verified": False,
            "identity_preservation_verified": False,
            "seed_selection_allowed": False, "training_target_allowed": False,
            "training_allowed": False, "optimizer_step_allowed": False,
            "parameter_update_allowed": False,
            "review_job_success_claimed": False,
        }
    ):
        die("review submission receipt closure differs")
    validate_formal_sacct_observation(
        formal["live_sacct_observation"], job_id=formal["job_id"]
    )
    for path, expected_sha, label in (
        (EXPECTED_LAUNCHER, release_manifest["inputs"]["launcher"]["sha256"],
         "review launcher"),
        (EXPECTED_ADAPTER, inputs["adapter_sha256"], "review adapter"),
        (EXPECTED_POSTFLIGHT, release_manifest["inputs"]["postflight"]["sha256"],
         "review terminal postflight"),
        (EXPECTED_SUBMITTER, inputs["submitter_sha256"], "review submitter"),
        (EXPECTED_HOSTILE, inputs["hostile_sha256"], "review hostile"),
        (EXPECTED_SOURCE_ARCHIVE, EXPECTED_SOURCE_ARCHIVE_SHA256,
         "review source archive"),
        (EXPECTED_FORMAL_MASTER, formal["master_receipt_sha256"],
         "formal master receipt"),
        (EXPECTED_FORMAL_SUBMISSION, formal["submission_receipt_sha256"],
         "formal submission receipt"),
    ):
        plain_file(path, label=label)
        if sha_file(path) != expected_sha:
            die(f"{label} SHA-256 differs")
    master_value, formal_submission_value = validate_formal_receipts(
        formal_master_raw,
        formal_submission_raw,
        formal_bundle=formal,
    )
    return value, raw, digest, master_value, formal_submission_value


def validate_packet(
    *, review_job_id: str, review_submission: dict[str, Any],
    retained_automation_raw: bytes, formal_master: Mapping[str, Any],
    formal_submission: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes, str, dict[str, Any]]:
    root_info = OUTPUT_ROOT.lstat()
    if (
        OUTPUT_ROOT.resolve(strict=True) != OUTPUT_ROOT
        or not stat.S_ISDIR(root_info.st_mode)
        or stat.S_ISLNK(root_info.st_mode)
        or root_info.st_uid != os.getuid()
        or stat.S_IMODE(root_info.st_mode) != 0o555
    ):
        die("review packet root differs")
    for path in OUTPUT_ROOT.rglob("*"):
        info = path.lstat()
        expected_mode = 0o555 if stat.S_ISDIR(info.st_mode) else 0o444
        if (
            stat.S_ISLNK(info.st_mode)
            or (not stat.S_ISDIR(info.st_mode) and not stat.S_ISREG(info.st_mode))
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != expected_mode
        ):
            die(f"review packet entry differs: {path}")

    automation_raw = retained_automation_raw
    automation = decode_canonical(
        automation_raw, label="review runtime automation receipt"
    )
    automation_digest = verify_seal(
        automation,
        field="receipt_digest",
        expected=None,
        label="review runtime automation receipt",
    )
    review_receipt_path = OUTPUT_ROOT / "detached-review-receipt.json"
    manifest_path = OUTPUT_ROOT / "review-manifest.json"
    protocol_path = OUTPUT_ROOT / "observer-protocol.json"
    review_receipt, review_raw = load_canonical(
        review_receipt_path, label="detached review receipt"
    )
    review_digest = verify_seal(
        review_receipt,
        field="receipt_digest",
        expected=None,
        label="detached review receipt",
    )
    manifest, manifest_raw = load_canonical(manifest_path, label="review manifest")
    manifest_digest = verify_seal(
        manifest, field="manifest_digest", expected=None, label="review manifest"
    )
    protocol, protocol_raw = load_canonical(protocol_path, label="observer protocol")
    protocol_digest = verify_seal(
        protocol, field="protocol_digest", expected=None, label="observer protocol"
    )
    interface = automation.get("automatic_backfill_interface", {})
    authority = automation.get("authority", {})
    packet = automation.get("packet", {})
    formal_input = automation.get("formal_input", {})
    formal_bundle = review_submission.get("formal_terminal_input_bundle", {})
    formal_sacct = formal_bundle.get("live_sacct_observation", {})
    if (
        type(automation) is not dict
        or set(automation) != {
            "schema_version", "status", "formal_input", "review_job_id",
            "packet", "automatic_backfill_interface", "authority",
            "receipt_digest",
        }
        or automation.get("schema_version")
        != "saic-t2v-topup-detached-review-v2-automation-receipt-v1"
        or automation.get("status")
        != "technical_packet_materialized_runtime_postflight_pending"
        or automation.get("review_job_id") != review_job_id
        or type(packet) is not dict
        or set(packet) != {
            "output_root", "packet_id", "review_receipt",
            "review_receipt_sha256", "review_receipt_digest",
            "review_manifest", "review_manifest_sha256",
            "review_manifest_digest", "candidate_count",
            "technical_diagnostic_count",
        }
        or packet.get("output_root") != str(OUTPUT_ROOT)
        or packet.get("packet_id") != PACKET_ID
        or packet.get("review_receipt")
        != str(OUTPUT_ROOT / "detached-review-receipt.json")
        or packet.get("review_manifest")
        != str(OUTPUT_ROOT / "review-manifest.json")
        or packet.get("review_receipt_sha256") != sha_bytes(review_raw)
        or packet.get("review_receipt_digest") != review_digest
        or packet.get("review_manifest_sha256") != sha_bytes(manifest_raw)
        or packet.get("review_manifest_digest") != manifest_digest
        or packet.get("candidate_count") != 60
        or packet.get("technical_diagnostic_count") != 60
        or type(formal_input) is not dict
        or set(formal_input) != {
            "job_id", "terminal_state", "exit_code", "master_receipt",
            "master_receipt_sha256", "master_receipt_digest",
            "submission_receipt", "submission_receipt_sha256",
            "submission_receipt_digest", "sacct_executable",
            "sacct_executable_sha256", "sacct_stdout_sha256",
            "sacct_stderr_sha256",
        }
        or formal_input.get("job_id") != formal_bundle.get("job_id")
        or formal_input.get("terminal_state") != "COMPLETED"
        or formal_input.get("exit_code") != "0:0"
        or formal_input.get("master_receipt")
        != formal_bundle.get("master_receipt")
        or formal_input.get("master_receipt_sha256")
        != formal_bundle.get("master_receipt_sha256")
        or formal_input.get("master_receipt_digest")
        != formal_bundle.get("master_receipt_digest")
        or formal_input.get("submission_receipt")
        != formal_bundle.get("submission_receipt")
        or formal_input.get("submission_receipt_sha256")
        != formal_bundle.get("submission_receipt_sha256")
        or formal_input.get("submission_receipt_digest")
        != formal_bundle.get("submission_receipt_digest")
        or formal_input.get("sacct_executable") != str(EXPECTED_SACCT)
        or formal_input.get("sacct_executable_sha256") != EXPECTED_SACCT_SHA256
        or formal_input.get("sacct_stdout_sha256")
        != formal_sacct.get("stdout_sha256")
        or formal_input.get("sacct_stderr_sha256")
        != formal_sacct.get("stderr_sha256")
        or type(interface) is not dict
        or interface != {
            "technical_diagnostics_glob": str(OUTPUT_ROOT / "diagnostics/*.json"),
            "technical_html": str(OUTPUT_ROOT / "index.html"),
            "technical_html_sha256": sha_file(OUTPUT_ROOT / "index.html"),
            "blind_human_review_html": str(OUTPUT_ROOT / "blind-review.html"),
            "blind_human_review_html_sha256":
                sha_file(OUTPUT_ROOT / "blind-review.html"),
            "observer_protocol": str(OUTPUT_ROOT / "observer-protocol.json"),
            "observer_protocol_sha256": sha_bytes(protocol_raw),
            "observer_templates": interface.get("observer_templates"),
            "machine_artifacts_materialized": True,
            "blind_stage_public_surface_only": True,
            "technical_html_may_publish_before_two_external_human_seals": False,
            "assessor_private_mapping_may_be_copied_to_stage1": False,
            "machine_ingest_ready": False,
            "terminal_machine_ingest_scope": "assessor_private_only",
            "human_visible_machine_backfill_ready": False,
            "external_terminal_postflight_required": True,
            "human_label_ingest_ready": False,
            "human_template_copyout_required": True,
            "blind_review_is_only_prelabel_ui": True,
            "technical_html_visibility_requires_two_external_human_seals": True,
        }
        or authority != {
            "machine_diagnostics_have_semantic_authority": False,
            "machine_backfill_may_precede_human_label_seal": False,
            "assessor_private_machine_ingest_may_precede_human_seals": True,
            "review_job_terminal_success_claimed": False,
            "human_labels_present": False, "event_verified": False,
            "identity_preservation_verified": False,
            "seed_selection_allowed": False, "training_target_allowed": False,
            "training_allowed": False, "optimizer_step_allowed": False,
            "parameter_update_allowed": False,
        }
    ):
        die("review runtime automation closure differs")
    if (
        set(review_receipt) != REVIEW_RECEIPT_FIELDS
        or set(manifest) != REVIEW_MANIFEST_FIELDS
        or set(protocol) != OBSERVER_PROTOCOL_FIELDS
        or review_receipt.get("schema_version")
        != "bernini-saic-t2v-event-bank-topup-detached-review-receipt-v2"
        or review_receipt.get("packet_id") != PACKET_ID
        or review_receipt.get("candidate_count") != 60
        or review_receipt.get("machine_diagnostic_count") != 60
        or review_receipt.get("machine_diagnostics_zero_authority") is not True
        or review_receipt.get("semantic_status") != "UNASSESSED"
        or review_receipt.get("observer_labels_present") is not False
        or any(
            review_receipt.get(field) is not False
            for field in (
                "detached_full81_event_review_complete", "event_verified",
                "identity_preservation_verified",
                "candidate_ranking_or_selection_performed",
                "seed_selection_authorized", "training_target_authorized",
                "training_performed", "optimizer_created",
                "optimizer_step_authorized", "parameter_update_authorized",
            )
        )
        or review_receipt.get("authority") != FALSE_AUTHORITY
        or manifest.get("schema_version")
        != "bernini-saic-t2v-event-bank-topup-detached-review-manifest-v2"
        or manifest.get("packet_id") != PACKET_ID
        or manifest.get("candidate_count") != 60
        or manifest.get("machine_diagnostic_count") != 60
        or manifest.get("semantic_status") != "UNASSESSED"
        or manifest.get("detached_human_review_complete") is not False
        or manifest.get("candidate_ranking_or_selection_performed") is not False
        or manifest.get("machine_diagnostic_authority")
        != "ZERO_AUTHORITY_DIAGNOSTIC_ONLY"
        or manifest.get("authority") != FALSE_AUTHORITY
        or protocol.get("packet_id") != PACKET_ID
        or protocol.get("schema_version")
        != "bernini-saic-t2v-event-bank-topup-independent-full81-observer-protocol-v2"
        or protocol.get("protocol_digest") != protocol_digest
        or protocol.get("authority") != FALSE_AUTHORITY
        or protocol.get("aggregation_rule") != {
            "majority_vote_allowed": False,
            "tie_break_or_adjudication_inside_v2_allowed": False,
            "missing_response_result": "UNASSESSED",
            "observer_disagreement_result": "UNASSESSED",
            "agreed_positive_result":
                "AGREED_POSITIVE_PENDING_SEPARATE_EXTERNAL_SEAL",
            "agreed_negative_result":
                "AGREED_NEGATIVE_PENDING_SEPARATE_EXTERNAL_SEAL",
            "event_verified_may_be_set_by_this_packet": False,
            "identity_verified_may_be_set_by_this_packet": False,
            "separate_versioned_aggregator_required": True,
        }
        or protocol.get("machine_diagnostic_contract") != {
            "human_labels_must_precede_machine_diagnostic_access": True,
            "machine_camera_or_technical_thresholds_calibrated": False,
            "machine_diagnostics_may_fill_or_change_human_labels": False,
            "machine_diagnostics_have_semantic_authority": False,
            "machine_diagnostics_may_select_seed_or_training_target": False,
        }
    ):
        die("review packet receipt/manifest/protocol closure differs")

    items = manifest.get("items")
    bindings = manifest.get("input_bindings")
    if (
        type(items) is not list
        or len(items) != 60
        or any(type(item) is not dict for item in items)
        or len({item.get("review_item_id") for item in items}) != 60
        or len({item.get("candidate_id") for item in items}) != 60
        or {item.get("row_id") for item in items}
        != {f"source-{index:04d}" for index in range(1, 9)}
        or type(bindings) is not dict
        or set(bindings) != {"master_receipt", "source_manifest", "event_spec"}
    ):
        die("review manifest item/evidence closure differs")
    private_candidate_ids: set[str] = set()
    private_source_rows: set[str] = set()
    source_alias_pairs: set[tuple[str, str]] = set()
    for alias_index, item in enumerate(items, start=1):
        candidate_alias = f"candidate-{alias_index:04d}"
        private_candidate_id = str(item.get("assessor_private_candidate_id", ""))
        private_source_row = str(item.get("assessor_private_source_row_id", ""))
        if (
            set(item) != REVIEW_ITEM_FIELDS
            or item.get("registered_candidate_index") != alias_index
            or item.get("candidate_id") != candidate_alias
            or item.get("review_item_id") != f"review-{alias_index:04d}"
            or re.fullmatch(r"source-[0-9]{4}", str(item.get("row_id")))
            is None
            or item.get("branch") not in {
                "incomplete", "camera_only", "appearance_only",
            }
            or type(item.get("seed")) is not int
            or item.get("portable_source")
            != f"media/sources/{item.get('row_id')}.mp4"
            or item.get("portable_candidate")
            != f"media/candidates/{alias_index:04d}-{candidate_alias}.mp4"
            or item.get("portable_attempt_receipt")
            != f"evidence/attempts/{candidate_alias}.json"
            or item.get("portable_diagnostic")
            != f"diagnostics/{candidate_alias}.json"
            or item.get("semantic_status") != "UNASSESSED"
            or item.get("event_verified") is not False
            or item.get("identity_preservation_verified") is not False
            or item.get("diagnostic_summary", {}).get("semantic_status")
            != "UNASSESSED"
            or item.get("diagnostic_summary", {}).get("authority")
            != "diagnostic_only"
            or re.fullmatch(
                r"saic-topup-v2-[0-9a-f]{16}-"
                r"(?:incomplete|camera_only|appearance_only)-s[0-9]+",
                private_candidate_id,
            ) is None
            or Path(str(item.get("candidate_input_path"))).parent.name
            != private_candidate_id
            or re.fullmatch(
                r"(?:fit|confirmation)-(?:dog|human)-0[01]-[0-9a-f]{16}",
                private_source_row,
            ) is None
            or not private_source_row.endswith("-" + str(item.get("iid")))
        ):
            die("review opaque alias registration differs")
        private_candidate_ids.add(private_candidate_id)
        private_source_rows.add(private_source_row)
        source_alias_pairs.add((private_source_row, str(item.get("row_id"))))
    if (
        len(private_candidate_ids) != 60
        or len(private_source_rows) != 8
        or len(source_alias_pairs) != 8
        or len({alias for _, alias in source_alias_pairs}) != 8
    ):
        die("review assessor-private alias mapping differs")
    expected_protocol = expected_observer_protocol(items)
    expected_protocol_binding = {
        "portable_path": "observer-protocol.json",
        "file_sha256": sha_bytes(protocol_raw),
        "protocol_digest": protocol_digest,
        "review_item_set_digest": expected_protocol["review_item_set_digest"],
    }
    if (
        protocol != expected_protocol
        or protocol_digest != expected_protocol["protocol_digest"]
        or review_receipt.get("job_id") != review_job_id
        or review_receipt.get("input_bank_receipt") != {
            "path": str(EXPECTED_FORMAL_MASTER),
            "file_sha256": formal_bundle.get("master_receipt_sha256"),
            "receipt_digest": formal_bundle.get("master_receipt_digest"),
        }
        or review_receipt.get("review_manifest") != {
            "path": str(manifest_path),
            "file_sha256": sha_bytes(manifest_raw),
            "manifest_digest": manifest_digest,
        }
        or review_receipt.get("html_review") != {
            "path": str(OUTPUT_ROOT / "index.html"),
            "file_sha256": sha_file(OUTPUT_ROOT / "index.html"),
        }
        or review_receipt.get("blind_human_review") != {
            "path": str(OUTPUT_ROOT / "blind-review.html"),
            "file_sha256": sha_file(OUTPUT_ROOT / "blind-review.html"),
            "machine_diagnostics_exposed": False,
        }
        or review_receipt.get("observer_protocol") != {
            "path": str(protocol_path),
            "file_sha256": sha_bytes(protocol_raw),
            "protocol_digest": protocol_digest,
        }
        or review_receipt.get("row_count") != 8
        or review_receipt.get("source_count") != 8
        or review_receipt.get("seed_cell_count") != 20
        or review_receipt.get("exact81_machine_diagnostics_complete") is not True
        or review_receipt.get("full80_transition_machine_diagnostics_complete")
        is not True
        or review_receipt.get("observer_template_count") != 2
        or manifest.get("job_id") != review_job_id
        or manifest.get("input_root") != str(EXPECTED_FORMAL_ROOT)
        or manifest.get("input_bank_receipt_digest")
        != formal_bundle.get("master_receipt_digest")
        or manifest.get("input_event_spec_raw_sha256")
        != bindings.get("event_spec", {}).get("sha256")
        or manifest.get("row_count") != 8
        or manifest.get("source_count") != 8
        or manifest.get("seed_cell_count") != 20
        or manifest.get("frame_count_per_media") != 81
        or manifest.get("transition_count_per_media") != 80
        or manifest.get("fps") != 25
        or manifest.get("branch_order")
        != ["incomplete", "camera_only", "appearance_only"]
        or manifest.get("machine_diagnostic_axes")
        != ["camera", "technical", "temporal_consistency"]
        or manifest.get("observer_protocol") != expected_protocol_binding
        or manifest.get("observer_template_count") != 2
        or type(manifest.get("observer_templates")) is not list
        or len(manifest["observer_templates"]) != 2
    ):
        die("review receipt/manifest/protocol deep binding differs")

    formal_attempts = {
        str(attempt["candidate_id"]): attempt
        for attempt in formal_master["attempts"]
    }
    if set(formal_attempts) != private_candidate_ids:
        die("formal master/review candidate set differs")
    for item in items:
        attempt = formal_attempts[str(item["assessor_private_candidate_id"])]
        if (
            attempt.get("row_id") != item.get("assessor_private_source_row_id")
            or attempt.get("iid") != item.get("iid")
            or attempt.get("analysis_split") != item.get("analysis_split")
            or attempt.get("branch") != item.get("branch")
            or attempt.get("seed") != item.get("seed")
            or attempt.get("receipt_path")
            != item.get("attempt_receipt_input_path")
            or attempt.get("receipt_sha256")
            != item.get("attempt_receipt_sha256")
            or attempt.get("receipt_digest")
            != item.get("attempt_receipt_digest")
            or attempt.get("mp4_path") != item.get("candidate_input_path")
            or attempt.get("mp4_sha256") != item.get("candidate_sha256")
        ):
            die("formal master/review item cross-binding differs")
    validate_blind_surface(
        items, job_id=review_job_id, protocol_digest=protocol_digest,
    )
    expected_directories = {
        OUTPUT_ROOT / "evidence",
        OUTPUT_ROOT / "evidence/attempts",
        OUTPUT_ROOT / "media",
        OUTPUT_ROOT / "media/sources",
        OUTPUT_ROOT / "media/candidates",
        OUTPUT_ROOT / "diagnostics",
        OUTPUT_ROOT / "observer-templates",
    }
    expected_files = {
        review_receipt_path,
        manifest_path,
        protocol_path,
        OUTPUT_ROOT / "index.html",
        OUTPUT_ROOT / "blind-review.html",
        OUTPUT_ROOT / "observer-templates/observer-1-blank.json",
        OUTPUT_ROOT / "observer-templates/observer-2-blank.json",
    }
    for label, binding in bindings.items():
        expected_files.add(validate_copy_binding(
            binding, portable_key="portable_path", label=f"{label} evidence"
        ))
    event_binding = bindings["event_spec"]
    base_binding = event_binding.get("base_v1_spec")
    if type(base_binding) is not dict:
        die("review base-v1 evidence binding differs")
    expected_files.add(validate_copy_binding(
        base_binding, portable_key="portable_path", label="base-v1 evidence"
    ))
    master_binding = bindings["master_receipt"]
    source_binding = bindings["source_manifest"]
    event_path = packet_path(
        event_binding.get("portable_path"), label="event_spec evidence"
    )
    source_manifest_path = packet_path(
        source_binding.get("portable_path"), label="source_manifest evidence"
    )
    event_spec, event_spec_raw = load_canonical(
        event_path, label="portable formal event spec"
    )
    source_manifest, source_manifest_raw = load_canonical(
        source_manifest_path, label="portable formal source manifest"
    )
    if (
        master_binding.get("sha256")
        != formal_bundle.get("master_receipt_sha256")
        or master_binding.get("bytes")
        != len(canonical(formal_master) + b"\n")
        or event_binding.get("sha256")
        != formal_master.get("root_spec_raw_sha256")
        or event_binding.get("sha256")
        != formal_submission.get("inputs", {}).get("event_spec_sha256")
        or sha_bytes(event_spec_raw) != event_binding.get("sha256")
        or set(event_spec) != {
            "schema_version", "bank_id", "top_up_only",
            "base_v1_spec_raw_sha256", "base_v1_spec_content_sha256",
            "source_manifest_content_sha256", "source_manifest_file_sha256",
            "sampling_contract", "semantic_input_closure",
            "geometry_proxy_contract", "artifact_authority", "branch_order",
            "merged_branch_order", "groups",
        }
        or event_spec.get("schema_version")
        != "bernini-saic-pure-t2v-event-bank-topup-spec-v2"
        or event_spec.get("bank_id") != formal_master.get("bank_id")
        or event_spec.get("top_up_only") is not True
        or event_spec.get("base_v1_spec_raw_sha256")
        != formal_master.get("base_v1_spec_raw_sha256")
        or event_spec.get("base_v1_spec_content_sha256")
        != formal_master.get("base_v1_spec_content_sha256")
        or event_spec.get("source_manifest_content_sha256")
        != formal_master.get("source_manifest_content_sha256")
        or event_spec.get("source_manifest_file_sha256")
        != source_binding.get("sha256")
        or source_binding.get("sha256")
        != formal_submission.get("inputs", {}).get("source_manifest_sha256")
        or sha_bytes(source_manifest_raw) != source_binding.get("sha256")
        or sha_bytes(canonical(source_manifest))
        != EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256
        or any(
            event_spec.get(field) != formal_master.get(field)
            for field in (
                "sampling_contract", "semantic_input_closure",
                "geometry_proxy_contract", "artifact_authority",
                "branch_order", "merged_branch_order",
            )
        )
        or base_binding.get("sha256") != EXPECTED_BASE_V1_SPEC_RAW_SHA256
    ):
        die("formal evidence/master/spec cross-binding differs")
    event_groups = event_spec.get("groups")
    source_rows = source_manifest.get("rows")
    if (
        type(event_groups) is not list
        or len(event_groups) != 2
        or type(source_rows) is not list
        or len(source_rows) != 8
    ):
        die("formal event/source registry cardinality differs")
    spec_candidates: dict[str, Mapping[str, Any]] = {}
    for group in event_groups:
        candidates = group.get("candidates") if type(group) is dict else None
        if type(candidates) is not list or len(candidates) != 30:
            die("formal event candidate group differs")
        for candidate in candidates:
            candidate_id = (
                str(candidate.get("candidate_id", ""))
                if type(candidate) is dict else ""
            )
            if not candidate_id or candidate_id in spec_candidates:
                die("formal event candidate registry differs")
            spec_candidates[candidate_id] = candidate
    sources_by_iid: dict[str, Mapping[str, Any]] = {}
    for source in source_rows:
        iid = str(source.get("iid", "")) if type(source) is dict else ""
        if not iid or iid in sources_by_iid:
            die("formal source registry differs")
        sources_by_iid[iid] = source
    if (
        set(spec_candidates) != private_candidate_ids
        or set(sources_by_iid) != {str(item.get("iid", "")) for item in items}
    ):
        die("formal event/source review registry differs")
    candidate_item_fields = (
        "iid", "analysis_split", "actor_family", "action_family_id",
        "branch", "seed", "initial_state_type", "terminal_state_type",
        "branch_start_state_caption", "branch_instruction", "full_t2v_caption",
    )
    source_item_fields = (
        "iid", "analysis_split", "actor_family", "action_family_id",
        "initial_state_type", "terminal_state_type",
    )
    for item in items:
        candidate = spec_candidates[
            str(item["assessor_private_candidate_id"])
        ]
        source = sources_by_iid[str(item["iid"])]
        if (
            candidate.get("row_id")
            != item.get("assessor_private_source_row_id")
            or any(
                candidate.get(field) != item.get(field)
                for field in candidate_item_fields
            )
            or source.get("row_id")
            != item.get("assessor_private_source_row_id")
            or any(
                source.get(field) != item.get(field)
                for field in source_item_fields
            )
            or source.get("source_video") != item.get("source_input_path")
            or source.get("source_video_sha256") != item.get("source_sha256")
            or candidate.get("source_media_sha256_for_nonuse_audit")
            != item.get("source_sha256")
        ):
            die("formal event/source/review item cross-binding differs")
    source_files: set[Path] = set()
    for item in items:
        if type(item) is not dict:
            die("review manifest item differs")
        source_path = packet_path(item.get("portable_source"), label="portable source")
        plain_file(source_path, label="portable source")
        if (
            sha_file(source_path) != item.get("source_sha256")
            or source_path.stat().st_size != item.get("portable_source_bytes")
        ):
            die("portable source binding differs")
        source_files.add(source_path)
        for field, label in (
            ("portable_candidate", "portable candidate"),
            ("portable_attempt_receipt", "portable attempt receipt"),
            ("portable_diagnostic", "portable diagnostic"),
        ):
            path = packet_path(item.get(field), label=label)
            plain_file(path, label=label)
            sha_field = {
                "portable_candidate": "candidate_sha256",
                "portable_attempt_receipt": "attempt_receipt_sha256",
                "portable_diagnostic": "diagnostic_file_sha256",
            }[field]
            if sha_file(path) != item.get(sha_field):
                die(f"{label} binding differs")
            bytes_field = {
                "portable_candidate": "portable_candidate_bytes",
                "portable_attempt_receipt": "portable_attempt_receipt_bytes",
                "portable_diagnostic": None,
            }[field]
            if bytes_field is not None and path.stat().st_size != item.get(bytes_field):
                die(f"{label} byte count differs")
            if field == "portable_attempt_receipt":
                attempt_receipt, _ = load_canonical(path, label=label)
                verify_seal(
                    attempt_receipt,
                    field="receipt_digest",
                    expected=item.get("attempt_receipt_digest"),
                    label=label,
                )
            if field == "portable_diagnostic":
                diagnostic, _ = load_canonical(path, label=label)
                diagnostic_digest = verify_seal(
                    diagnostic,
                    field="diagnostic_digest",
                    expected=item.get("diagnostic_digest"),
                    label=label,
                )
                if (
                    set(diagnostic) != {
                        "schema_version", "media", "runtime", "input_closure",
                        "availability", "source", "candidate", "comparisons",
                        "authority", "remaining_gaps", "diagnostic_digest",
                    }
                    or diagnostic_digest != item.get("diagnostic_digest")
                    or diagnostic.get("authority") != DIAGNOSTIC_AUTHORITY
                    or diagnostic.get("availability") != {
                        "identity": "unavailable", "appearance": "unavailable",
                        "background": "unavailable", "non_target": "unavailable",
                        "event": "unavailable", "source_bind": "unavailable",
                        "inverse": "unavailable", "camera": "diagnostic_only",
                        "technical": "diagnostic_only",
                        "temporal_consistency": "diagnostic_only",
                    }
                    or item.get("diagnostic_summary") != {
                        "camera": diagnostic.get("comparisons", {}).get(
                            "camera_trajectory"
                        ),
                        "technical": diagnostic.get("comparisons", {}).get(
                            "technical"
                        ),
                        "scene_cut_ratio_absolute_difference":
                            diagnostic.get("comparisons", {}).get(
                                "scene_cut_ratio_absolute_difference"
                            ),
                        "temporal_energy_cv_absolute_difference":
                            diagnostic.get("comparisons", {}).get(
                                "temporal_energy_cv_absolute_difference"
                            ),
                        "semantic_status": "UNASSESSED",
                        "authority": "diagnostic_only",
                    }
                ):
                    die("portable diagnostic zero-authority closure differs")
            expected_files.add(path)
    expected_files.update(source_files)
    actual_entries = set(OUTPUT_ROOT.rglob("*"))
    if (
        len(source_files) != 8
        or len(expected_files) != 199
        or actual_entries != expected_directories | expected_files
    ):
        die("review packet exact namespace closure differs")
    if (
        interface.get("technical_html_sha256")
        != sha_file(OUTPUT_ROOT / "index.html")
        or interface.get("blind_human_review_html_sha256")
        != sha_file(OUTPUT_ROOT / "blind-review.html")
        or interface.get("observer_protocol_sha256") != sha_bytes(protocol_raw)
        or len(list((OUTPUT_ROOT / "diagnostics").glob("*.json"))) != 60
        or len(list((OUTPUT_ROOT / "observer-templates").glob("*.json"))) != 2
    ):
        die("review automatic artifact binding differs")
    templates = interface.get("observer_templates")
    if type(templates) is not list or len(templates) != 2:
        die("review observer-template binding differs")
    protocol_binding = manifest.get("observer_protocol")
    expected_review_ids = [str(item["review_item_id"]) for item in items]
    for slot, binding in enumerate(templates, start=1):
        path = OUTPUT_ROOT / f"observer-templates/observer-{slot}-blank.json"
        template, raw = load_canonical(path, label="blank observer template")
        verify_seal(template, field="template_digest", expected=None, label="blank observer template")
        validate_blank_template(
            template,
            slot=slot,
            expected_review_ids=expected_review_ids,
            protocol_binding=protocol_binding,
        )
        manifest_template_binding = manifest["observer_templates"][slot - 1]
        if (
            type(binding) is not dict
            or set(binding) != {
                "observer_slot", "path", "sha256", "template_digest",
                "template_only",
            }
            or binding.get("observer_slot") != slot
            or binding.get("path") != str(path)
            or binding.get("sha256") != sha_bytes(raw)
            or binding.get("template_digest") != template.get("template_digest")
            or binding.get("template_only") is not True
            or template.get("template_only") is not True
            or manifest_template_binding != {
                "observer_slot": slot,
                "portable_path":
                    f"observer-templates/observer-{slot}-blank.json",
                "file_sha256": sha_bytes(raw),
                "template_digest": template.get("template_digest"),
                "semantic_status": "UNASSESSED",
                "template_only": True,
            }
        ):
            die("observer template is not exactly blank")
    return automation, automation_raw, automation_digest, {
        "review_receipt_sha256": sha_bytes(review_raw),
        "review_receipt_digest": review_digest,
        "manifest_sha256": sha_bytes(manifest_raw),
        "manifest_digest": manifest_digest,
        "protocol_sha256": sha_bytes(protocol_raw),
        "protocol_digest": protocol_digest,
    }


def write_admission(value: dict[str, Any]) -> None:
    if TERMINAL_ADMISSION.exists() or TERMINAL_ADMISSION.is_symlink():
        die("terminal admission target is not fresh")
    parent = TERMINAL_ADMISSION.parent
    parent_info = parent.lstat()
    if (
        parent.resolve(strict=True) != parent
        or not stat.S_ISDIR(parent_info.st_mode)
        or stat.S_ISLNK(parent_info.st_mode)
        or parent_info.st_uid != os.getuid()
        or stat.S_IMODE(parent_info.st_mode) & 0o022
    ):
        die("terminal admission parent differs")
    payload = canonical(value) + b"\n"
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened_parent = os.fstat(parent_descriptor)
        if (
            not stat.S_ISDIR(opened_parent.st_mode)
            or (opened_parent.st_dev, opened_parent.st_ino)
            != (parent_info.st_dev, parent_info.st_ino)
        ):
            die("opened terminal admission parent differs")
        try:
            os.stat(
                TERMINAL_ADMISSION.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            die("terminal admission target appeared before create")
        flags = (
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(
            TERMINAL_ADMISSION.name, flags, 0o600, dir_fd=parent_descriptor
        )
        try:
            reserved = os.fstat(descriptor)
            public = os.stat(
                TERMINAL_ADMISSION.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            identity = (reserved.st_dev, reserved.st_ino)
            if (
                not stat.S_ISREG(reserved.st_mode)
                or reserved.st_nlink != 1
                or reserved.st_uid != os.getuid()
                or stat.S_IMODE(reserved.st_mode) != 0o600
                or reserved.st_size != 0
                or (public.st_dev, public.st_ino) != identity
            ):
                die("terminal admission reservation differs")
            offset = 0
            while offset < len(payload):
                wrote = os.write(descriptor, payload[offset:])
                if wrote <= 0:
                    die("terminal admission write stalled")
                offset += wrote
            os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.read(descriptor, len(payload) + 1) != payload:
                die("terminal admission same-FD reread differs")
            written = os.fstat(descriptor)
            public = os.stat(
                TERMINAL_ADMISSION.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(written.st_mode)
                or written.st_nlink != 1
                or written.st_uid != os.getuid()
                or stat.S_IMODE(written.st_mode) != 0o600
                or written.st_size != len(payload)
                or (written.st_dev, written.st_ino) != identity
                or (public.st_dev, public.st_ino) != identity
                or stat.S_IMODE(public.st_mode) != 0o600
                or public.st_size != len(payload)
            ):
                die("terminal admission written identity differs")
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.read(descriptor, len(payload) + 1) != payload:
                die("sealed terminal admission same-FD reread differs")
            sealed = os.fstat(descriptor)
            public = os.stat(
                TERMINAL_ADMISSION.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            public_path = TERMINAL_ADMISSION.lstat()
            if (
                not stat.S_ISREG(sealed.st_mode)
                or sealed.st_nlink != 1
                or sealed.st_uid != os.getuid()
                or stat.S_IMODE(sealed.st_mode) != 0o444
                or sealed.st_size != len(payload)
                or (sealed.st_dev, sealed.st_ino) != identity
                or (public.st_dev, public.st_ino) != identity
                or (public_path.st_dev, public_path.st_ino) != identity
                or stat.S_IMODE(public.st_mode) != 0o444
                or stat.S_IMODE(public_path.st_mode) != 0o444
                or public.st_size != len(payload)
                or public_path.st_size != len(payload)
            ):
                die("sealed terminal admission identity differs")
        finally:
            os.close(descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", required=True)
    parser.add_argument("--release-manifest-sha256", required=True)
    parser.add_argument("--release-manifest-digest", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if TERMINAL_ADMISSION.exists() or TERMINAL_ADMISSION.is_symlink():
        die("terminal admission already exists")
    retained_descriptors: list[
        tuple[int, Path, bytes, os.stat_result]
    ] = []
    try:
        if Path(args.release_manifest) != EXPECTED_RELEASE_MANIFEST:
            die("review release manifest path differs")
        manifest_fd, manifest_raw, manifest_info = retained_plain_bytes(
            EXPECTED_RELEASE_MANIFEST,
            label="review release manifest",
            exact_mode=0o444,
        )
        retained_descriptors.append(
            (manifest_fd, EXPECTED_RELEASE_MANIFEST, manifest_raw, manifest_info)
        )
        if Path(__file__) != EXPECTED_POSTFLIGHT:
            die("review postflight execution path differs")
        self_fd, self_raw, self_info = retained_plain_bytes(
            EXPECTED_POSTFLIGHT,
            label="review terminal postflight",
            exact_mode=0o444,
        )
        retained_descriptors.append(
            (self_fd, EXPECTED_POSTFLIGHT, self_raw, self_info)
        )
        release_manifest = validate_release_manifest(
            manifest_raw,
            expected_sha256=args.release_manifest_sha256,
            expected_digest=args.release_manifest_digest,
            postflight_raw=self_raw,
        )
        formal_master_fd, formal_master_raw, formal_master_info = \
            retained_plain_bytes(
                EXPECTED_FORMAL_MASTER,
                label="formal master receipt",
                exact_mode=0o444,
            )
        retained_descriptors.append(
            (
                formal_master_fd, EXPECTED_FORMAL_MASTER, formal_master_raw,
                formal_master_info,
            )
        )
        formal_submission_fd, formal_submission_raw, formal_submission_info = \
            retained_plain_bytes(
                EXPECTED_FORMAL_SUBMISSION,
                label="formal submission receipt",
                exact_mode=0o444,
            )
        retained_descriptors.append(
            (
                formal_submission_fd, EXPECTED_FORMAL_SUBMISSION,
                formal_submission_raw, formal_submission_info,
            )
        )
        submission_fd, submission_raw, submission_info = retained_plain_bytes(
            SUBMISSION_RECEIPT,
            label="review submission receipt",
            exact_mode=0o444,
        )
        retained_descriptors.append(
            (submission_fd, SUBMISSION_RECEIPT, submission_raw, submission_info)
        )
        submission, submission_raw, submission_digest, formal_master, \
            formal_submission = validate_submission(
            submission_raw, submission_info,
            release_manifest=release_manifest,
            release_manifest_sha256=args.release_manifest_sha256,
            release_manifest_digest=args.release_manifest_digest,
            formal_master_raw=formal_master_raw,
            formal_submission_raw=formal_submission_raw,
        )
        review_job_id = submission["submitted_job"]["job_id"]
        automation_fd, automation_raw, automation_info = retained_plain_bytes(
            AUTOMATION_RECEIPT,
            label="review runtime automation receipt",
            exact_mode=0o444,
        )
        retained_descriptors.append(
            (automation_fd, AUTOMATION_RECEIPT, automation_raw, automation_info)
        )
        sacct = exact_sacct(review_job_id, submission)
        slurm_logs, retained_logs = validate_slurm_logs(review_job_id)
        retained_descriptors.extend(retained_logs.values())
        automation, automation_raw, automation_digest, packet_seals = validate_packet(
            review_job_id=review_job_id,
            review_submission=submission,
            retained_automation_raw=automation_raw,
            formal_master=formal_master,
            formal_submission=formal_submission,
        )
        body = {
        "schema_version": "saic-t2v-topup-detached-review-v2-terminal-admission-v1",
        "status": "terminal_technical_packet_admitted_human_review_unassessed",
        "review_job_id": review_job_id,
        "terminal_sacct_observation": sacct,
        "slurm_logs": slurm_logs,
        "release_manifest": {
            "path": str(EXPECTED_RELEASE_MANIFEST),
            "sha256": args.release_manifest_sha256,
            "receipt_digest": args.release_manifest_digest,
            "postflight_sha256":
                release_manifest["inputs"]["postflight"]["sha256"],
            "submitter_sha256":
                release_manifest["inputs"]["submitter"]["sha256"],
        },
        "formal_inputs": {
            "job_id": submission["formal_terminal_input_bundle"]["job_id"],
            "master_receipt": str(EXPECTED_FORMAL_MASTER),
            "master_receipt_sha256": sha_bytes(formal_master_raw),
            "master_receipt_digest": formal_master["receipt_digest"],
            "submission_receipt": str(EXPECTED_FORMAL_SUBMISSION),
            "submission_receipt_sha256": sha_bytes(formal_submission_raw),
            "submission_receipt_digest": formal_submission["receipt_digest"],
            "event_spec_raw_sha256": formal_master["root_spec_raw_sha256"],
            "candidate_count": formal_master["attempt_count"],
        },
        "submission_receipt": {
            "path": str(SUBMISSION_RECEIPT),
            "sha256": sha_bytes(submission_raw),
            "receipt_digest": submission_digest,
        },
        "runtime_automation_receipt": {
            "path": str(AUTOMATION_RECEIPT),
            "sha256": sha_bytes(automation_raw),
            "receipt_digest": automation_digest,
            "prior_machine_ingest_ready": automation[
                "automatic_backfill_interface"
            ]["machine_ingest_ready"],
        },
        "packet": {
            "output_root": str(OUTPUT_ROOT),
            "packet_id": PACKET_ID,
            **packet_seals,
            "candidate_count": 60,
            "technical_diagnostic_count": 60,
        },
        "automatic_backfill_admission": {
            "machine_artifact_ingest_ready": True,
            "machine_artifact_ingest_scope": "assessor_private_only",
            "human_visible_machine_backfill_ready": False,
            "technical_diagnostics_glob": str(OUTPUT_ROOT / "diagnostics/*.json"),
            "technical_html": str(OUTPUT_ROOT / "index.html"),
            "blind_human_review_html": str(OUTPUT_ROOT / "blind-review.html"),
            "observer_template_glob": str(
                OUTPUT_ROOT / "observer-templates/*.json"
            ),
            "human_label_ingest_ready": False,
            "blind_review_is_only_prelabel_ui": True,
            "blind_stage_public_surface_only": True,
            "technical_html_may_publish_before_two_external_human_seals": False,
            "assessor_private_mapping_may_be_copied_to_stage1": False,
            "technical_html_visibility_to_human_observers_ready": False,
            "two_external_human_response_seals_required_before_technical_visibility":
                True,
        },
        "authority": {
            "review_job_terminal_success": True,
            "machine_diagnostics_have_semantic_authority": False,
            "human_labels_present": False,
            "human_review_complete": False,
            "event_verified": False,
            "identity_preservation_verified": False,
            "seed_selection_allowed": False,
            "training_target_allowed": False,
            "training_allowed": False,
            "optimizer_step_allowed": False,
            "parameter_update_allowed": False,
        },
        }
        value = {**body, "admission_digest": sha_bytes(canonical(body))}
        for descriptor, path, raw, initial in retained_descriptors:
            reread_retained(
                descriptor, path, raw, initial, label=path.name,
            )
        if exact_sacct(review_job_id, submission) != sacct:
            die("review terminal Slurm accounting changed before publication")
        reread_logs, closing_logs = validate_slurm_logs(review_job_id)
        try:
            if reread_logs != slurm_logs:
                die("review Slurm log bytes changed before publication")
        finally:
            for descriptor, _, _, _ in closing_logs.values():
                os.close(descriptor)
        # Packet closure is deliberately the final external read before the
        # create-only terminal admission publication.
        closing_automation, closing_automation_raw, closing_automation_digest, \
            closing_packet_seals = validate_packet(
                review_job_id=review_job_id,
                review_submission=submission,
                retained_automation_raw=automation_raw,
                formal_master=formal_master,
                formal_submission=formal_submission,
            )
        if (
            closing_automation != automation
            or closing_automation_raw != automation_raw
            or closing_automation_digest != automation_digest
            or closing_packet_seals != packet_seals
        ):
            die("review packet changed before terminal publication")
        write_admission(value)
    finally:
        for descriptor, _, _, _ in retained_descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
    print(
        canonical(
            {
                "review_job_id": review_job_id,
                "terminal_admission": str(TERMINAL_ADMISSION),
                "admission_digest": value["admission_digest"],
                "machine_artifact_ingest_ready": True,
                "machine_artifact_ingest_scope": "assessor_private_only",
                "human_label_ingest_ready": False,
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
