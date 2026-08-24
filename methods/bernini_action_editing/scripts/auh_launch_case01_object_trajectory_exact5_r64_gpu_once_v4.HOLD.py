#!/usr/bin/env python3
"""Fresh-v4 receipt-gated one-shot GPU controller for exact-five.

This checked-in copy is deliberately HOLD.  A READY copy may be derived only
after the fresh package controller and composite four-rank CPU admission have
published immutable receipts and every dynamic pin below has been reviewed.

The controller consumes the sealed package; it never edits a package source,
the HOLD plan, the HOLD launch input, or the named HOLD payload.  Instead it
derives the only permitted READY-plan transform in memory.  On execution the
immutable attempt is the first mutation, followed by a create-only READY plan,
then exactly one ``srun`` whose stdin is an in-memory production payload.  The
frozen exact5 runner owns all five serial arms, four-rank Torchrun, media
publication, report, and attestation.
"""

from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


SCHEMA = "case01-object-trajectory-exact5-r64-gpu-controller-v4"
ATTEMPT_SCHEMA = SCHEMA + "-attempt"
DISPATCH_SCHEMA = SCHEMA + "-dispatch"
EVIDENCE_SCHEMA = SCHEMA + "-evidence"
RUNTIME_SCHEMA = SCHEMA + "-runtime"
RANK_CACHE_SCHEMA = SCHEMA + "-rank-cache"
COMPUTE_RESULT_SCHEMA = SCHEMA + "-compute-result"
READY_PLAN_SCHEMA = "case01-object-trajectory-exact5-plan-v3"
CONTROLLER_STATE = "HOLD_PENDING_FRESH_V4_COMPOSITE_CPU_PINS"
READY_STATE = "READY_EXPLICIT_SINGLE_SRUN_EXACT_FIVE_NO_RETRY"

HOLDER_JOB_ID = "143808"
NODE = "auh7-1b-gpu-292"
GPU_COUNT = 8
CPUS_PER_TASK = 64
MEMORY = "64G"
SRUN_TIMEOUT_SECONDS = 10_800
TERMINATE_GRACE_SECONDS = 10.0
REMOTE_UID = 2012
REMOTE_GID = 2000
FILE_MODE = 0o444
RECEIPT_MODE = 0o400
DIRECTORY_MODE = 0o700
MAX_JSON_SIZE = 64 * 1024 * 1024
MAX_SOURCE_SIZE = 4 * 1024 * 1024
MAX_RUNTIME_EXECUTABLE_SIZE = 128 * 1024 * 1024
MAX_VIDEO_SIZE = 4 * 1024 * 1024 * 1024
# AUH's allocation host reported this exact execve(2) limit.  The outer srun
# argv is fixed-width and deliberately gets a much smaller reviewed ceiling;
# the generated Bash stdin and its nested Python argv have independent bounds.
OBSERVED_AUH_ARG_MAX = 2_097_152
MAX_EXACT_SRUN_ARGV_BYTES = 32_768
MAX_HELD_STDIN_BYTES = 131_072
MAX_NESTED_PYTHON_ARGV_BYTES = 131_072
MIN_EXECVE_HEADROOM_BYTES = 1_048_576
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")

EXPERIMENTS = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments"
)
PACKAGE_ROOT = (
    EXPERIMENTS / "bernini_case01_object_trajectory_exact5_r64_canary_v3"
)
SOURCE_OVERLAY_ROOT = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r5f_v4_source_overlay_6_20260822_r1"
)
SOURCE_OVERLAY_RECEIPT_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r5f_v4_source_overlay_6_20260822_r1."
    "receipt_v1.json"
)
PACKAGE_PUBLICATION_RECEIPT_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r64_canary_v3."
    "publication_receipt_v4.json"
)
MATERIALIZATION_REPORT_PATH = (
    PACKAGE_ROOT / "authority/package_materialization_receipt_v4.json"
)
PACKAGE_CONTROLLER_EVIDENCE_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r64_canary_v3."
    "materialize_controller_evidence_v3.json"
)
COMPOSITE_CPU_RECEIPT_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r64_canary_v3."
    "composite_cpu_admission_receipt_v2.json"
)
COMPOSITE_CPU_EVIDENCE_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r64_canary_v3."
    "composite_cpu_admission_controller_evidence_v2.json"
)
HOLD_PLAN_PATH = (
    PACKAGE_ROOT / "plan/case01_object_trajectory_exact5_r64_HOLD_plan_v3.json"
)
LAUNCH_INPUT_PATH = PACKAGE_ROOT / "launch/root_launch_input_HOLD_v3.json"
HOLD_PAYLOAD_PATH = PACKAGE_ROOT / "launch/root_launch_payload_HOLD_v3.sh"

READY_PLAN_PATH = (
    PACKAGE_ROOT / "runtime/case01_object_trajectory_exact5_r64_READY_plan_v3.json"
)
RUNTIME_RECEIPT_PATH = PACKAGE_ROOT / "runtime/exact5_gpu_runtime_v4.json"
RANK_CACHE_RECEIPT_PATH = (
    PACKAGE_ROOT / "runtime/exact5_gpu_rank_cache_receipt_v4.json"
)
ATTEMPT_PATH = PACKAGE_ROOT / "evidence/exact5_gpu_attempt_v4.json"
DISPATCH_PATH = PACKAGE_ROOT / "evidence/exact5_gpu_dispatch_v4.json"
EVIDENCE_PATH = PACKAGE_ROOT / "evidence/exact5_gpu_controller_receipt_v4.json"
STDOUT_PATH = PACKAGE_ROOT / "logs/exact5_gpu_srun_v4.stdout.log"
STDERR_PATH = PACKAGE_ROOT / "logs/exact5_gpu_srun_v4.stderr.log"
OUTPUT_REPORT_PATH = (
    PACKAGE_ROOT / "final/object_trajectory_exact5_report_v3.json"
)
RUNNER_ATTESTATION_PATH = (
    PACKAGE_ROOT / "final/object_trajectory_exact5_runner_attestation_v3.json"
)
OUTPUT_ROOT = PACKAGE_ROOT / "outputs/media_v3"
AUTHORITY_ROOT = PACKAGE_ROOT / "runtime/model-authority-v3"
RANK_CACHE_ROOT = Path(
    "/tmp/bernini-case01-object-trajectory-exact5-r64-job143808-node292-r3-"
    "rank-cache"
)
SITE_PACKAGES_ROOT = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
    "site-packages"
)
TORCH_PACKAGE_INIT_PATH = SITE_PACKAGES_ROOT / "torch/__init__.py"

PACKAGE_PUBLICATION_SCHEMA = (
    "case01-object-trajectory-exact5-r64-package-publication-v4-receipt"
)
MATERIALIZATION_SCHEMA = (
    "case01-object-trajectory-exact5-r64-hold-materialization-v4"
)
PACKAGE_CONTROLLER_EVIDENCE_SCHEMA = (
    "case01-object-trajectory-exact5-r64-overlay-package-controller-v3-evidence"
)
COMPOSITE_CPU_SCHEMA = (
    "case01-object-trajectory-exact5-r5f-v4-composite-cpu-admission-v2"
)
COMPOSITE_CPU_EVIDENCE_SCHEMA = (
    "case01-object-trajectory-exact5-r5f-v4-composite-cpu-controller-v2-evidence"
)
PACKAGE_RECEIPT_FIELDS = frozenset({
    "schema_version", "status", "target_root", "receipt_path",
    "materialization_receipt_path", "materialization_receipt_sha256",
    "materialization_receipt_digest", "source_snapshot_manifest_sha256",
    "source_snapshot_manifest_digest", "source_staging_receipt_sha256",
    "source_staging_receipt_digest", "source_overlay_receipt_sha256",
    "source_overlay_receipt_digest", "source_overlay_root_identity",
    "publication_protocol", "rename_noreplace", "cooperative_writer_exclusion",
    "target_absent_rechecked_before_rename", "ordinary_posix_rename_performed",
    "publication_observation", "whole_tree_atomically_visible",
    "uncooperative_same_uid_race_out_of_scope", "retry_allowed",
    "target_root_identity", "receipt_mode", "receipt_is_consumption_gate",
    "receipt_is_admission", "launch_allowed", "receipt_inode_anchor",
    "receipt_digest",
})
MATERIALIZATION_FIELDS = frozenset({
    "schema_version", "status", "launch_allowed", "root",
    "source_snapshot_root", "source_snapshot", "source_overlay_root",
    "source_overlay", "source_provenance", "source_staging_receipt_authority",
    "package_publication_receipt_path", "publication_protocol",
    "rename_noreplace", "cooperative_writer_exclusion",
    "uncooperative_same_uid_race_out_of_scope", "retry_allowed",
    "release_file_count", "production_identity_count", "release", "production",
    "condition_and_admission_authority_count", "plan", "launch", "admission",
    "slurm_step_launched", "gpu_attempt_claimed", "artifacts", "receipt_digest",
})
HOLD_LAUNCH_FIELDS = frozenset({
    "schema_version", "status", "launch_allowed", "slurm_step_launched",
    "gpu_attempt_claimed", "input", "release", "payload_path",
    "payload_sha256", "payload_size", "receipt_digest",
})
HOLD_LAUNCH_RELEASE_FIELDS = frozenset({
    "schema_version", "status", "launch_allowed", "campaign_mode",
    "selected_task_ids", "identity_roles", "identities", "input_sha256",
    "ready_overlay_required", "named_payload_execution_forbidden",
    "release_digest",
})
REPORT_RELEASE_FIELDS = frozenset({"files", "manifest_digest"})
REPORT_RELEASE_ROW_FIELDS = frozenset({
    "path", "sha256", "size", "provenance",
})
PRODUCTION_FIELDS = frozenset({
    "identity_roles", "identities", "identity_set_digest",
    "inner_outer_crosslink",
})
INNER_OUTER_CROSSLINK_FIELDS = frozenset({
    "adapter", "object_wrapper_inner", "producer_adapter",
    "producer_object_wrapper_inner", "distinct_paths",
    "outer_calls_pinned_inner_contract",
})
DIAGNOSTIC_ARTIFACT_SHA256 = {
    "diagnostics/case01_object_trajectory_exact5_static_probe_v1.py":
        "071256da47635fc3481f51b48e7e5eddddc963a5345b1dda405473744d2c01a9",
    "diagnostics/case01_object_trajectory_exact5_root_fake_runner_v1.py":
        "0d73fdaa4a4f1817f572eea471661850098ffe5aa54f54a9927c37a7e3f2a872",
    "diagnostics/case01_object_trajectory_exact5_world4_probe_v1.py":
        "71c52ea3d7b36f07fdf5f9af3c9ecadf2020123795d9a4e10888a67eb0c7536b",
}
HOLD_LAUNCH_RECEIPT_SCHEMA = (
    "case01-object-trajectory-exact5-hold-launch-receipt-auh-v3"
)
HOLD_LAUNCH_RELEASE_SCHEMA = (
    "case01-object-trajectory-exact5-hold-launch-release-auh-v3"
)
RUNNER_SCHEMA = "case01-object-trajectory-exact5-runner-attestation-v3"
REPORT_SCHEMA = "case01-object-trajectory-exact5-report-v3"
CAMPAIGN = "case01-object-trajectory-exact5-r64-engineering-oracle-v3"
EXPECTED_TORCH_VERSION = "2.7.1+rocm6.3"
EXPECTED_HIP_VERSION = "6.3.42131-fa1d09cbd"
EXPECTED_GPU_NAME = "AMD Instinct MI210"

RUNTIME_RECEIPT_FIELDS = frozenset({
    "schema_version", "status", "holder_job_id", "node", "slurm_step_id",
    "production_release_digest", "ready_plan_digest", "torch_version",
    "hip_version", "device_count", "device_names",
    "r64_checkpoint_manifest_sha256", "exact26_identity_set_digest",
    "site_packages_root", "site_packages_identity",
    "torch_package_init_authority", "torch_module_path",
    "torch_module_sha256", "torch_package_entry_compiled_from_held_source",
    "torch_source_loader", "isolated_flag", "no_site_flag",
    "dont_write_bytecode",
    "site_packages_activated_explicitly_under_isolated_no_site",
    "renderer_started", "receipt_digest",
})
RANK_CACHE_RECEIPT_FIELDS = frozenset({
    "schema_version", "status", "holder_job_id", "node", "hostname",
    "uname_nodename", "slurm_step_id", "production_release_digest",
    "runtime_receipt_sha256", "runtime_receipt_digest",
    "runner_attestation_sha256", "runner_attestation_digest",
    "rank_cache_root", "rank_cache_root_identity", "rank_cache_parent",
    "rank_cache_parent_identity", "package_root", "package_root_identity",
    "cache_device_matches_tmp", "cache_device_differs_from_package",
    "cache_mount", "compute_node_observation", "retained_compute_local",
    "fresh_unique_attempt_path",
    "freshness_enforced_by_frozen_runner_before_mkdir",
    "cleanup_performed", "absent_claimed", "non_scientific_cache",
    "cache_is_not_output_or_result", "task_count", "rank_process_count",
    "coordinator_process_count", "task_topology", "inventory_bound",
    "inventory", "inventory_entry_count", "inventory_total_file_bytes",
    "inventory_digest", "terminal_inventory_replayed_exactly",
    "internal_artifact_inventory", "internal_artifact_count",
    "internal_artifact_inventory_digest",
    "internal_artifact_fds_held_and_terminal_replayed",
    "model_authority_root", "model_authority_root_identity",
    "model_authority_root_empty",
    "model_authority_root_held_and_terminal_replayed",
    "rank_processes_zero", "torchrun_processes_zero", "cgroup_baseline",
    "cgroup_terminal", "cgroup_returned_exactly_to_single_root_process",
    "process_scan_performed", "process_scan_sources", "process_scan",
    "matched_residual_pids", "receipt_digest",
})
COMPUTE_RESULT_FIELDS = frozenset({
    "schema_version", "status", "holder_job_id", "node", "slurm_step_id",
    "runner_returncode", "production_release_digest", "ready_plan_digest",
    "runtime_receipt_sha256", "runtime_receipt_digest",
    "rank_cache_receipt_sha256", "rank_cache_receipt_digest",
    "runner_attestation_sha256", "runner_attestation_digest",
    "all_five_arms_attempted_exactly_once", "retry_count",
    "rank_processes_zero", "rank_process_scan_performed",
    "rank_cache_compute_state", "rank_cache_inventory_digest",
    "internal_artifact_count", "internal_artifact_inventory_digest",
    "model_authority_root_identity", "model_authority_root_empty",
    "model_authority_root_held_and_terminal_replayed",
    "local_srun_process_group_observed", "result_digest",
})
COMPOSITE_CPU_RECEIPT_FIELDS = frozenset({
    "schema_version", "status", "holder_job_id", "node", "slurm_step_id",
    "package", "world_size", "rank_count", "rank_rows",
    "isolated_runtime", "private_parent_fd", "shared_ofd_pread",
    "module_binding", "activation_import", "side_effects", "cache_lifecycle",
    "process_cleanup", "launch_allowed", "receipt_digest",
})
COMPOSITE_CPU_RANK_ROW_FIELDS = frozenset({
    "rank", "pid", "private_parent_fd_number",
    "private_parent_replacement_inode", "pread_bytes_sha256",
    "pread_offset_before", "pread_offset_after",
    "activation_callback_import_module",
    "activation_import_before_callback_return",
    "captured_vendor_finder_preinstalled",
    "captured_vendor_finder_count", "captured_vendor_loader_type",
    "captured_vendor_spec_loader_type",
    "captured_vendor_loader_is_spec_loader",
    "captured_vendor_cached_is_none", "rank_digest",
})
COMPOSITE_CPU_EVIDENCE_FIELDS = frozenset({
    "schema_version", "status", "holder_job_id", "node", "slurm_step_id",
    "single_srun_attempt", "retry_allowed", "srun_count", "srun_ntasks",
    "real_rank_process_count", "cpus_per_task", "gpu_count",
    "srun_returncode", "receipt", "receipt_digest", "stdout", "stderr",
    "stderr_empty", "process_group_zero", "launch_allowed",
    "renderer_or_vae_loaded", "publication_performed", "evidence_digest",
})

ARM_ORDER = (
    "null_before", "route_off", "trajectory_bone_only",
    "trajectory_dog_bone", "null_after",
)
TASK_IDS = tuple(
    f"case01-object-trajectory-{arm}-full644" for arm in ARM_ORDER
)
IDENTITY_ROLES = (
    "runner", "legacy_exact5_runner", "object_eval", "legacy_exact5_eval",
    "frozen_runner", "bridge", "adapter", "object_wrapper_inner",
    "legacy_infer_alias",
    "trajectory_projection", "trajectory_scaffold_module", "base_adapter",
    "eval_v1", "eval_v2", "model_authority", "torchrun_source",
    "torchrun_handler_source", "torch_local_agent_source",
    "torch_dynamic_rendezvous_source", "torch_multiprocessing_api_source",
    "base_model_manifest", "r64_checkpoint_manifest", "python", "ffmpeg",
    "ffprobe", "plan",
)
EXECUTABLE_ROLES = frozenset({"python", "ffmpeg", "ffprobe"})

CORE4_RELEASE_PINS = {
    "release/methods/bernini_action_editing/"
    "full644_exploratory_matched_infer_adapter_v3.py": (
        "7b72e8dc88d95daa34d93604dddacf6dcf4f75a2f92f356f743183cf06fa7120",
        124_612,
    ),
    "release/methods/bernini_action_editing/"
    "infer_case01_object_trajectory_oracle_auh_r5f_v4.py": (
        "797c5d1e7cb8bbfda1f2e4cc3825702c248d3ce64770ddc1520155f5635c3557",
        42_184,
    ),
    "release/methods/bernini_action_editing/"
    "case01_object_trajectory_exact5_eval_v4.py": (
        "381ba375147bec7580b451226b07b3d1cab9125866978602de05fbba4f16aaa3",
        116_371,
    ),
    "release/methods/bernini_action_editing/"
    "case01_object_trajectory_exact5_runner_v4.py": (
        "326ccfff1a09d6db8c93d02cfe6018e465e127263547f325cc7f18e7d16a7148",
        21_712,
    ),
    "release/methods/bernini_action_editing/"
    "case01_object_trajectory_exact5_spooled_launcher_auh_v4.py": (
        "0315a8630f77e816c3fc5fc9139b8fb72323db59d5d155f85b039ba132cc9b5a",
        27_878,
    ),
}
CORE4_IDENTITY_BINDINGS = {
    "base_adapter": (
        "release/methods/bernini_action_editing/"
        "full644_exploratory_matched_infer_adapter_v3.py"
    ),
    "adapter": (
        "release/methods/bernini_action_editing/"
        "infer_case01_object_trajectory_oracle_auh_r5f_v4.py"
    ),
    "object_eval": (
        "release/methods/bernini_action_editing/"
        "case01_object_trajectory_exact5_eval_v4.py"
    ),
    "runner": (
        "release/methods/bernini_action_editing/"
        "case01_object_trajectory_exact5_runner_v4.py"
    ),
}
SOURCE_OVERLAY_RECEIPT_SHA256 = (
    "5827df34b30496b3b768f26e4be91de71b4a54dda0da000f0b00373549146be4"
)
SOURCE_OVERLAY_RECEIPT_SIZE = 3_915
SOURCE_OVERLAY_RECEIPT_DIGEST = (
    "463bf54d848730ac8b13a625c8a66e436c5523b7595dcda29095fe194986baa2"
)
SOURCE_OVERLAY_ROOT_IDENTITY = [
    48, 5704346356003806783, 2012, 2000, 16749, 2, 0, 4096, 0,
    1787377341057464968, 1787377456971845705,
]
SOURCE_OVERLAY_MATERIALIZER_PATH = SOURCE_OVERLAY_ROOT / (
    "methods/bernini_action_editing/tools/"
    "materialize_case01_object_trajectory_exact5_r64_overlay_package_v3.py"
)
SOURCE_OVERLAY_MATERIALIZER_SHA256 = (
    "63196562f8f036cae8c54ba7799dd3695ee6805f01a3dd9756b1167eaa7f3d13"
)
SOURCE_OVERLAY_MATERIALIZER_SIZE = 119_970

PACKAGE_PUBLICATION_RECEIPT_SHA256 = (
    "ffb74c4cf70ced6491cde23a37d9389b3f8c65431e354194d96842dd6a494871"
)
PACKAGE_PUBLICATION_RECEIPT_SIZE: int | str = 2_528
PACKAGE_PUBLICATION_RECEIPT_DIGEST = (
    "82533716cd0286182fc731e3ffdf46cf8b95ed1ae0bb0a421d26aa77685bf720"
)
MATERIALIZATION_REPORT_SHA256 = (
    "c60c28ab1418914fd61480507c7c2e284ea58a1132fb265a40e3a5aa2ec56c95"
)
MATERIALIZATION_REPORT_SIZE: int | str = 41_726
MATERIALIZATION_REPORT_DIGEST = (
    "d0790a3618539d918d7deaa07a066961b08a19e4973de5e11f8abca9cd52d7be"
)
PACKAGE_CONTROLLER_EVIDENCE_SHA256 = (
    "bed59791557b9cdebd8280edbd3a68976c4588984815045eeea1f45b864ea0c7"
)
PACKAGE_CONTROLLER_EVIDENCE_SIZE: int | str = 8_099
PACKAGE_CONTROLLER_EVIDENCE_DIGEST = (
    "39dbc033a2845cb7a73d759334df76411521cebd20a455620d37f5db5339236e"
)
PACKAGE_ROOT_IDENTITY: list[int] | str = [
    48, 3113453814725663979, 2012, 2000, 16832, 2, 0, 4096, 0,
    1787378196307021665, 1787378196629068696,
]

# Final package-local plan/input/named-HOLD-payload facts certified by the
# immutable materialization report.  They are intentionally independent of
# the downstream admission pins and make width regression use the real sealed
# package, rather than a short synthetic plan or launcher fixture.
SEALED_HOLD_PLAN_SHA256 = (
    "d9dadcd5a293e2313e4e5381bd095380f2da730add4c29afba2dd38f9b2e7483"
)
SEALED_HOLD_PLAN_SIZE: int | str = 32_050
SEALED_HOLD_PLAN_DIGEST = (
    "e4485a73c1988a3560378000cdee2182266c81aff5f0318b83ac21d9ee787d24"
)
SEALED_LAUNCH_INPUT_SHA256 = (
    "7973c9311e5a539a106a938bf452d5405f492fdd8f7f41a9038b5da09279d347"
)
SEALED_LAUNCH_INPUT_SIZE: int | str = 9_788
NAMED_HOLD_PAYLOAD_SHA256 = (
    "f5eb7add48c521d01893e64e4d12963401b0aa2986e25ed06f66afb4fdaa1ccf"
)
NAMED_HOLD_PAYLOAD_SIZE: int | str = 12_783

# Static producer join for the import-order fix.  The fresh package must bind
# the active outer adapter to this exact base-v3 leaf; the CPU P0 gate further
# proves that ``bernini.pipeline`` is imported by the captured finder while
# the original activation callback is still running.
BASE_ADAPTER_BASENAME = "full644_exploratory_matched_infer_adapter_v3.py"
BASE_ADAPTER_SHA256 = (
    "7b72e8dc88d95daa34d93604dddacf6dcf4f75a2f92f356f743183cf06fa7120"
)
BASE_ADAPTER_PATH = PACKAGE_ROOT / (
    "release/methods/bernini_action_editing/"
    "full644_exploratory_matched_infer_adapter_v3.py"
)


def expected_composite_cpu_activation_import() -> dict[str, Any]:
    return {
        "module": "bernini.pipeline",
        "callback_phase": "inside_original_activate_before_return",
        "finder_installed_before_callback": True,
        "finder_count_per_rank": [1, 1, 1, 1],
        "loader_type": "_CapturedVendorLoader",
        "spec_loader_type": "_CapturedVendorLoader",
        "loader_is_spec_loader": True,
        "cached_is_none": True,
        "base_adapter_role": "base_adapter",
        "base_adapter_path": str(BASE_ADAPTER_PATH),
        "base_adapter_sha256": BASE_ADAPTER_SHA256,
        "rank_count": 4,
    }

# Learned only from the fresh four-rank composite CPU gate.  The checked-in
# controller remains HOLD while even one literal is unresolved.
COMPOSITE_CPU_RECEIPT_SHA256 = (
    "BLOCKED_PENDING_FRESH_V4_COMPOSITE_CPU_RECEIPT_SHA256"
)
COMPOSITE_CPU_RECEIPT_SIZE: int | str = (
    "BLOCKED_PENDING_FRESH_V4_COMPOSITE_CPU_RECEIPT_SIZE"
)
COMPOSITE_CPU_RECEIPT_DIGEST = (
    "BLOCKED_PENDING_FRESH_V4_COMPOSITE_CPU_RECEIPT_DIGEST"
)
COMPOSITE_CPU_EVIDENCE_SHA256 = (
    "BLOCKED_PENDING_FRESH_V4_COMPOSITE_CPU_EVIDENCE_SHA256"
)
COMPOSITE_CPU_EVIDENCE_SIZE: int | str = (
    "BLOCKED_PENDING_FRESH_V4_COMPOSITE_CPU_EVIDENCE_SIZE"
)
COMPOSITE_CPU_EVIDENCE_DIGEST = (
    "BLOCKED_PENDING_FRESH_V4_COMPOSITE_CPU_EVIDENCE_DIGEST"
)

SRUN_AUTHORITY = {
    "path": "/usr/bin/srun",
    "sha256": "2b8f60b30edf7efed35bb00864651da1b0bec68e75f942ce58b5ff82bc43cd9e",
    "size": 164_720,
}


class GPUControllerError(RuntimeError):
    """The reviewed one-shot GPU contract differs."""


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise GPUControllerError("value is not canonical JSON") from error


def object_digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev), int(info.st_ino), int(info.st_uid), int(info.st_gid),
        int(info.st_mode), int(info.st_nlink), int(info.st_rdev),
        int(info.st_size), int(getattr(info, "st_blocks", 0)),
        int(info.st_mtime_ns), int(info.st_ctime_ns),
    )


IDENTITY_FIELD_NAMES = (
    "device", "inode", "uid", "gid", "mode", "nlink", "rdev", "size",
    "blocks", "mtime_ns", "ctime_ns",
)


def held_publication_identity(authority: HeldAuthority) -> dict[str, int]:
    """Describe the inode actually held, never an earlier named-path sample."""

    authority.replay()
    if len(authority.held_identity) != len(IDENTITY_FIELD_NAMES):
        raise GPUControllerError("held publication identity width differs")
    return dict(zip(IDENTITY_FIELD_NAMES, authority.held_identity))


def object_anchor(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev), int(info.st_ino), int(info.st_uid), int(info.st_gid),
        int(stat.S_IFMT(info.st_mode)), int(stat.S_IMODE(info.st_mode)),
        int(info.st_rdev),
    )


def read_fd(descriptor: int, size: int) -> bytes:
    if type(size) is not int or size < 0:
        raise GPUControllerError("held read size differs")
    blocks: list[bytes] = []
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1_048_576, size - offset), offset)
        if not block:
            break
        blocks.append(block); offset += len(block)
    raw = b"".join(blocks)
    if len(raw) != size or os.pread(descriptor, 1, size) != b"":
        raise GPUControllerError("held read is incomplete")
    return raw


def strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise GPUControllerError(f"duplicate key in {label}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise GPUControllerError(f"invalid JSON authority: {label}") from error
    if type(value) is not dict or raw != canonical(value) + b"\n":
        raise GPUControllerError(f"noncanonical JSON authority: {label}")
    return value


class HeldAuthority:
    def __init__(
        self, path: Path, descriptor: int, held_identity: tuple[int, ...],
        raw: bytes,
    ) -> None:
        self.path = path
        self.descriptor = descriptor
        self.held_identity = held_identity
        self.raw = raw

    def replay(self) -> None:
        opened = os.fstat(self.descriptor)
        named = os.lstat(self.path)
        if (
            identity(opened) != self.held_identity
            or identity(named) != self.held_identity
            or read_fd(self.descriptor, opened.st_size) != self.raw
        ):
            raise GPUControllerError(f"held authority changed: {self.path}")

    def row(self) -> dict[str, Any]:
        info = os.fstat(self.descriptor)
        return {
            "path": str(self.path), "sha256": hashlib.sha256(self.raw).hexdigest(),
            "size": len(self.raw), "identity": list(identity(info)),
            "mode": stat.S_IMODE(info.st_mode), "nlink": info.st_nlink,
        }

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor); self.descriptor = -1


class HeldDirectory:
    def __init__(self, path: Path, descriptor: int, held_identity: tuple[int, ...]):
        self.path = path; self.descriptor = descriptor
        self.held_identity = held_identity

    def replay(self) -> None:
        if (
            object_anchor(os.fstat(self.descriptor))
            != object_anchor_from_identity(self.held_identity)
            or object_anchor(os.lstat(self.path))
            != object_anchor_from_identity(self.held_identity)
        ):
            raise GPUControllerError(f"held directory changed: {self.path}")

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor); self.descriptor = -1


def object_anchor_from_identity(value: Sequence[int]) -> tuple[int, ...]:
    if len(value) != 11:
        raise GPUControllerError("directory identity width differs")
    return (
        int(value[0]), int(value[1]), int(value[2]), int(value[3]),
        int(stat.S_IFMT(value[4])), int(stat.S_IMODE(value[4])), int(value[6]),
    )


def open_authority(
    path: Path, *, expected_sha256: str, expected_size: int,
    expected_mode: int | None, maximum_size: int = MAX_SOURCE_SIZE,
    executable: bool = False, expected_uid: int | None = REMOTE_UID,
    expected_gid: int | None = REMOTE_GID,
) -> HeldAuthority:
    if (
        not path.is_absolute() or os.path.normpath(str(path)) != str(path)
        or SHA_RE.fullmatch(str(expected_sha256)) is None
        or type(expected_size) is not int or not (0 <= expected_size <= maximum_size)
    ):
        raise GPUControllerError(f"noncanonical authority pin: {path}")
    named = os.lstat(path)
    if (
        not stat.S_ISREG(named.st_mode) or named.st_nlink != 1
        or (expected_uid is not None and named.st_uid != expected_uid)
        or (expected_gid is not None and named.st_gid != expected_gid)
        or (expected_mode is not None
            and stat.S_IMODE(named.st_mode) != expected_mode)
        or named.st_size != expected_size or path.resolve(strict=True) != path
        or (executable and not named.st_mode & 0o111)
    ):
        raise GPUControllerError(f"named authority differs: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor); raw = read_fd(descriptor, before.st_size)
        replay = read_fd(descriptor, before.st_size); after = os.fstat(descriptor)
        if (
            identity(before) != identity(named)
            or identity(before) != identity(after)
            or identity(before) != identity(os.lstat(path))
            or raw != replay or hashlib.sha256(raw).hexdigest() != expected_sha256
        ):
            raise GPUControllerError(f"authority replay differs: {path}")
        return HeldAuthority(path, descriptor, identity(before), raw)
    except BaseException:
        os.close(descriptor); raise


def open_observed_authority(
    path: Path, *, expected_mode: int, maximum_size: int,
    executable: bool = False,
) -> HeldAuthority:
    named = os.lstat(path)
    if (
        not path.is_absolute() or os.path.normpath(str(path)) != str(path)
        or not stat.S_ISREG(named.st_mode) or named.st_nlink != 1
        or named.st_uid != REMOTE_UID or named.st_gid != REMOTE_GID
        or stat.S_IMODE(named.st_mode) != expected_mode
        or not (0 <= named.st_size <= maximum_size)
        or path.resolve(strict=True) != path
        or (executable and not named.st_mode & 0o111)
    ):
        raise GPUControllerError(f"observed authority differs: {path}")
    return open_authority(
        path, expected_sha256=_sha_file(path), expected_size=named.st_size,
        expected_mode=expected_mode, maximum_size=maximum_size,
        executable=executable,
    )


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        while True:
            block = os.read(descriptor, 1_048_576)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def open_directory(path: Path, *, expected_identity: Sequence[int] | None = None) -> HeldDirectory:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor); named = os.lstat(path)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened.st_uid != REMOTE_UID or opened.st_gid != REMOTE_GID
            or identity(opened) != identity(named) or path.resolve(strict=True) != path
            or (
                expected_identity is not None
                and identity(opened) != tuple(expected_identity)
            )
        ):
            raise GPUControllerError(f"held directory differs: {path}")
        return HeldDirectory(path, descriptor, identity(opened))
    except BaseException:
        os.close(descriptor); raise


def dynamic_pin_values() -> dict[str, Any]:
    return {
        "package_publication_receipt_sha256": PACKAGE_PUBLICATION_RECEIPT_SHA256,
        "package_publication_receipt_size": PACKAGE_PUBLICATION_RECEIPT_SIZE,
        "package_publication_receipt_digest": PACKAGE_PUBLICATION_RECEIPT_DIGEST,
        "materialization_report_sha256": MATERIALIZATION_REPORT_SHA256,
        "materialization_report_size": MATERIALIZATION_REPORT_SIZE,
        "materialization_report_digest": MATERIALIZATION_REPORT_DIGEST,
        "package_controller_evidence_sha256": PACKAGE_CONTROLLER_EVIDENCE_SHA256,
        "package_controller_evidence_size": PACKAGE_CONTROLLER_EVIDENCE_SIZE,
        "package_controller_evidence_digest": PACKAGE_CONTROLLER_EVIDENCE_DIGEST,
        "sealed_hold_plan_sha256": SEALED_HOLD_PLAN_SHA256,
        "sealed_hold_plan_size": SEALED_HOLD_PLAN_SIZE,
        "sealed_hold_plan_digest": SEALED_HOLD_PLAN_DIGEST,
        "sealed_launch_input_sha256": SEALED_LAUNCH_INPUT_SHA256,
        "sealed_launch_input_size": SEALED_LAUNCH_INPUT_SIZE,
        "named_hold_payload_sha256": NAMED_HOLD_PAYLOAD_SHA256,
        "named_hold_payload_size": NAMED_HOLD_PAYLOAD_SIZE,
        "composite_cpu_receipt_sha256": COMPOSITE_CPU_RECEIPT_SHA256,
        "composite_cpu_receipt_size": COMPOSITE_CPU_RECEIPT_SIZE,
        "composite_cpu_receipt_digest": COMPOSITE_CPU_RECEIPT_DIGEST,
        "composite_cpu_evidence_sha256": COMPOSITE_CPU_EVIDENCE_SHA256,
        "composite_cpu_evidence_size": COMPOSITE_CPU_EVIDENCE_SIZE,
        "composite_cpu_evidence_digest": COMPOSITE_CPU_EVIDENCE_DIGEST,
        "package_root_identity": PACKAGE_ROOT_IDENTITY,
    }


def blocked_dynamic_pins() -> tuple[str, ...]:
    blocked: list[str] = []
    for key, value in dynamic_pin_values().items():
        if key == "package_root_identity":
            valid = (
                type(value) is list and len(value) == 11
                and all(type(item) is int for item in value)
            )
        elif key.endswith("_size"):
            valid = type(value) is int and 0 < value <= MAX_JSON_SIZE
        else:
            valid = type(value) is str and SHA_RE.fullmatch(value) is not None
        if not valid:
            blocked.append(key)
    return tuple(blocked)


def authorization_token() -> str:
    return object_digest({
        "schema_version": SCHEMA + "-authorization",
        "state": CONTROLLER_STATE, "job_id": HOLDER_JOB_ID, "node": NODE,
        "dynamic_pins": dynamic_pin_values(),
        "outputs": {
            "attempt": str(ATTEMPT_PATH), "dispatch": str(DISPATCH_PATH),
            "ready_plan": str(READY_PLAN_PATH),
            "runtime": str(RUNTIME_RECEIPT_PATH), "evidence": str(EVIDENCE_PATH),
            "rank_cache_receipt": str(RANK_CACHE_RECEIPT_PATH),
            "stdout": str(STDOUT_PATH), "stderr": str(STDERR_PATH),
        },
        "single_srun": True, "retry_allowed": False,
    })


def _self_digested(value: Mapping[str, Any], field: str, expected: str) -> None:
    unsigned = dict(value); claimed = unsigned.pop(field, None)
    if claimed != expected or claimed != object_digest(unsigned):
        raise GPUControllerError(f"{field} closure differs")


def validate_publication_receipt(held: HeldAuthority) -> dict[str, Any]:
    value = strict_json(held.raw, label="package publication receipt")
    _self_digested(value, "receipt_digest", PACKAGE_PUBLICATION_RECEIPT_DIGEST)
    if (
        set(value) != PACKAGE_RECEIPT_FIELDS
        or value.get("schema_version") != PACKAGE_PUBLICATION_SCHEMA
        or value.get("status") != "PUBLISHED_RECEIPT_GATED"
        or value.get("target_root") != str(PACKAGE_ROOT)
        or value.get("receipt_path") != str(PACKAGE_PUBLICATION_RECEIPT_PATH)
        or value.get("materialization_receipt_path")
        != str(MATERIALIZATION_REPORT_PATH)
        or value.get("materialization_receipt_sha256")
        != MATERIALIZATION_REPORT_SHA256
        or value.get("materialization_receipt_digest")
        != MATERIALIZATION_REPORT_DIGEST
        or value.get("source_overlay_receipt_sha256")
        != SOURCE_OVERLAY_RECEIPT_SHA256
        or value.get("source_overlay_receipt_digest")
        != SOURCE_OVERLAY_RECEIPT_DIGEST
        or value.get("source_overlay_root_identity")
        != SOURCE_OVERLAY_ROOT_IDENTITY
        or value.get("target_root_identity") != PACKAGE_ROOT_IDENTITY
        or value.get("rename_noreplace") is not False
        or value.get("cooperative_writer_exclusion") is not True
        or value.get("target_absent_rechecked_before_rename") is not True
        or value.get("ordinary_posix_rename_performed") is not True
        or value.get("publication_observation") != {
            "namespace_state": "target_same_inode_source_absent",
            "rename_returned_zero": True, "rename_error_errno": None,
            "parent_fsync_returned_zero": True,
            "parent_fsync_error_errno": None,
        }
        or value.get("whole_tree_atomically_visible") is not True
        or value.get("uncooperative_same_uid_race_out_of_scope") is not True
        or value.get("retry_allowed") is not False
        or value.get("receipt_mode") != RECEIPT_MODE
        or value.get("receipt_is_consumption_gate") is not True
        or value.get("receipt_is_admission") is not True
        or value.get("launch_allowed") is not False
    ):
        raise GPUControllerError("package publication receipt semantics differ")
    return value


def validate_materialization_report(held: HeldAuthority) -> dict[str, Any]:
    value = strict_json(held.raw, label="package materialization report")
    _self_digested(value, "receipt_digest", MATERIALIZATION_REPORT_DIGEST)
    launch = value.get("launch")
    plan = value.get("plan")
    admission = value.get("admission")
    input_row = launch.get("input") if type(launch) is dict else None
    production = value.get("production")
    release_block = value.get("release")
    source_overlay = value.get("source_overlay")
    if (
        set(value) != MATERIALIZATION_FIELDS
        or value.get("schema_version") != MATERIALIZATION_SCHEMA
        or value.get("status") != "MATERIALIZED_HOLD_NOT_SUBMITTED"
        or value.get("launch_allowed") is not False
        or value.get("root") != str(PACKAGE_ROOT)
        or value.get("source_overlay_root") != str(SOURCE_OVERLAY_ROOT)
        or type(source_overlay) is not dict
        or set(source_overlay) != {
            "root", "root_identity", "receipt", "source_file_count",
            "materializer", "files",
        }
        or source_overlay.get("root") != str(SOURCE_OVERLAY_ROOT)
        or source_overlay.get("root_identity") != SOURCE_OVERLAY_ROOT_IDENTITY
        or source_overlay.get("source_file_count") != 6
        or type(source_overlay.get("receipt")) is not dict
        or set(source_overlay["receipt"]) != {
            "path", "sha256", "size", "receipt_digest",
        }
        or source_overlay["receipt"].get("path")
        != str(SOURCE_OVERLAY_RECEIPT_PATH)
        or source_overlay["receipt"].get("sha256")
        != SOURCE_OVERLAY_RECEIPT_SHA256
        or source_overlay["receipt"].get("size")
        != SOURCE_OVERLAY_RECEIPT_SIZE
        or source_overlay["receipt"].get("receipt_digest")
        != SOURCE_OVERLAY_RECEIPT_DIGEST
        or source_overlay.get("materializer") != {
            "path": str(SOURCE_OVERLAY_MATERIALIZER_PATH),
            "sha256": SOURCE_OVERLAY_MATERIALIZER_SHA256,
            "size": SOURCE_OVERLAY_MATERIALIZER_SIZE,
        }
        or value.get("source_provenance") != {
            "base_release_leaf_count": 20,
            "overlay_release_leaf_count": 5,
            "overlay_nonrelease_materializer_leaf_count": 1,
            "burned_canary_v1_consumed": False,
            "failed_canary_v2_consumed": False,
        }
        or value.get("package_publication_receipt_path")
        != str(PACKAGE_PUBLICATION_RECEIPT_PATH)
        or value.get("retry_allowed") is not False
        or value.get("release_file_count") != 25
        or value.get("production_identity_count") != 26
        or value.get("condition_and_admission_authority_count") != 6
        or value.get("slurm_step_launched") is not False
        or value.get("gpu_attempt_claimed") is not False
        or admission != {
            "static_executed": False, "root_fake_executed": False,
            "world4_executed": False,
        }
        or type(plan) is not dict
        or set(plan) != {"path", "sha256", "plan_digest"}
        or plan.get("path") != str(HOLD_PLAN_PATH)
        or plan.get("sha256") != SEALED_HOLD_PLAN_SHA256
        or plan.get("plan_digest") != SEALED_HOLD_PLAN_DIGEST
        or type(launch) is not dict
        or set(launch) != HOLD_LAUNCH_FIELDS
        or launch.get("schema_version") != HOLD_LAUNCH_RECEIPT_SCHEMA
        or launch.get("status") != "MATERIALIZED_HOLD_NOT_SUBMITTED"
        or launch.get("launch_allowed") is not False
        or launch.get("slurm_step_launched") is not False
        or launch.get("gpu_attempt_claimed") is not False
        or launch.get("payload_path") != str(HOLD_PAYLOAD_PATH)
        or launch.get("payload_sha256") != NAMED_HOLD_PAYLOAD_SHA256
        or launch.get("payload_size") != NAMED_HOLD_PAYLOAD_SIZE
        or type(input_row) is not dict
        or set(input_row) != {"path", "sha256", "size", "mode", "nlink"}
        or input_row.get("path") != str(LAUNCH_INPUT_PATH)
        or input_row.get("sha256") != SEALED_LAUNCH_INPUT_SHA256
        or input_row.get("size") != SEALED_LAUNCH_INPUT_SIZE
        or input_row.get("mode") != FILE_MODE
        or input_row.get("nlink") != 1
    ):
        raise GPUControllerError("package materialization semantics differ")
    release = launch.get("release")
    identities = release.get("identities") if type(release) is dict else None
    production_identities = (
        production.get("identities") if type(production) is dict else None
    )
    crosslink = (
        production.get("inner_outer_crosslink")
        if type(production) is dict else None
    )
    unsigned_launch = dict(launch); launch_digest = unsigned_launch.pop(
        "receipt_digest", None,
    )
    unsigned_release = dict(release) if type(release) is dict else {}
    release_digest = unsigned_release.pop("release_digest", None)
    if (
        launch_digest != object_digest(unsigned_launch)
        or type(release) is not dict
        or set(release) != HOLD_LAUNCH_RELEASE_FIELDS
        or release.get("schema_version") != HOLD_LAUNCH_RELEASE_SCHEMA
        or release.get("status") != "HOLD_NOT_LAUNCHABLE"
        or release.get("launch_allowed") is not False
        or release.get("campaign_mode") != CAMPAIGN
        or release.get("selected_task_ids") != list(TASK_IDS)
        or release.get("identity_roles") != list(IDENTITY_ROLES)
        or release.get("input_sha256") != SEALED_LAUNCH_INPUT_SHA256
        or release.get("ready_overlay_required") is not True
        or release.get("named_payload_execution_forbidden") is not True
        or type(identities) is not dict
        or set(identities) != set(IDENTITY_ROLES)
        or len(identities) != 26
        or release_digest != object_digest(unsigned_release)
        or type(release_block) is not dict
        or set(release_block) != REPORT_RELEASE_FIELDS
        or type(release_block.get("files")) is not list
        or len(release_block["files"]) != 25
        or release_block.get("manifest_digest")
        != object_digest(release_block["files"])
        or type(production) is not dict
        or set(production) != PRODUCTION_FIELDS
        or production.get("identity_roles") != list(IDENTITY_ROLES)
        or production_identities != identities
        or production.get("identity_set_digest") != object_digest(identities)
        or type(crosslink) is not dict
        or set(crosslink) != INNER_OUTER_CROSSLINK_FIELDS
        or crosslink.get("adapter") != identities.get("adapter")
        or crosslink.get("object_wrapper_inner")
        != identities.get("object_wrapper_inner")
        or crosslink.get("producer_adapter") != identities.get("adapter")
        or crosslink.get("producer_object_wrapper_inner")
        != identities.get("object_wrapper_inner")
        or crosslink.get("distinct_paths") is not True
        or crosslink.get("outer_calls_pinned_inner_contract") is not True
    ):
        raise GPUControllerError("materialized HOLD launch closure differs")
    release_paths: list[str] = []
    for row in release_block["files"]:
        if (
            type(row) is not dict or set(row) != REPORT_RELEASE_ROW_FIELDS
            or type(row.get("path")) is not str
            or not row["path"].startswith("release/")
            or Path(row["path"]).is_absolute()
            or os.path.normpath(row["path"]) != row["path"]
            or SHA_RE.fullmatch(str(row.get("sha256"))) is None
            or type(row.get("size")) is not int or row["size"] <= 0
            or row.get("provenance") not in {
                "receipt_gated_exact5_overlay",
                "receipt_gated_exact35_snapshot",
            }
        ):
            raise GPUControllerError("materialized release25 row differs")
        release_paths.append(row["path"])
    if release_paths != sorted(release_paths) or len(set(release_paths)) != 25:
        raise GPUControllerError("materialized release25 path closure differs")
    if (
        "release/methods/bernini_action_editing/"
        "full644_exploratory_matched_infer_adapter_v2.py"
    ) in release_paths:
        raise GPUControllerError("revoked base-v2 adapter reappeared")
    release_by_path = {row["path"]: row for row in release_block["files"]}
    for relative, (expected_sha256, expected_size) in CORE4_RELEASE_PINS.items():
        if release_by_path.get(relative) != {
            "path": relative,
            "sha256": expected_sha256,
            "size": expected_size,
            "provenance": "receipt_gated_exact5_overlay",
        }:
            raise GPUControllerError(f"core-v4 release pin differs: {relative}")
    artifacts = value.get("artifacts")
    expected_release_artifacts = {
        row["path"]: {"sha256": row["sha256"], "size": row["size"]}
        for row in release_block["files"]
    }
    if (
        type(artifacts) is not dict or len(artifacts) != 28
        or not set(expected_release_artifacts) <= set(artifacts)
        or set(artifacts) - set(expected_release_artifacts)
        != set(DIAGNOSTIC_ARTIFACT_SHA256)
    ):
        raise GPUControllerError("materialized artifact map differs")
    for path, expected in expected_release_artifacts.items():
        if artifacts.get(path) != expected:
            raise GPUControllerError("materialized release artifact differs")
    for path, digest in DIAGNOSTIC_ARTIFACT_SHA256.items():
        row = artifacts.get(path)
        if (
            type(row) is not dict or set(row) != {"sha256", "size"}
            or row.get("sha256") != digest
            or type(row.get("size")) is not int or row["size"] <= 0
        ):
            raise GPUControllerError("materialized diagnostic artifact differs")
    for role, row in identities.items():
        if (
            type(row) is not dict or set(row) != {"path", "sha256", "size"}
            or type(row.get("path")) is not str
            or not Path(row["path"]).is_absolute()
            or os.path.normpath(row["path"]) != row["path"]
            or SHA_RE.fullmatch(str(row.get("sha256"))) is None
            or type(row.get("size")) is not int or row["size"] <= 0
        ):
            raise GPUControllerError(f"materialized identity differs: {role}")
    for role, relative in CORE4_IDENTITY_BINDINGS.items():
        expected_sha256, expected_size = CORE4_RELEASE_PINS[relative]
        if identities.get(role) != {
            "path": str(PACKAGE_ROOT / relative),
            "sha256": expected_sha256,
            "size": expected_size,
        }:
            raise GPUControllerError(f"core-v4 identity join differs: {role}")
    if (
        identities["base_adapter"].get("path") != str(BASE_ADAPTER_PATH)
        or identities["base_adapter"].get("sha256") != BASE_ADAPTER_SHA256
        or Path(identities["base_adapter"]["path"]).name
        != BASE_ADAPTER_BASENAME
    ):
        raise GPUControllerError("base-v3 adapter identity differs")
    if identities["plan"] != {
        "path": plan["path"], "sha256": plan["sha256"],
        "size": SEALED_HOLD_PLAN_SIZE,
    }:
        raise GPUControllerError("materialized plan identity differs")
    return value


def _row_matches_held(
    row: Any, held: HeldAuthority, *, label: str,
) -> None:
    if (
        type(row) is not dict
        or set(row) != {"path", "sha256", "size", "identity"}
        or row.get("path") != str(held.path)
        or row.get("sha256") != hashlib.sha256(held.raw).hexdigest()
        or row.get("size") != len(held.raw)
        or row.get("identity") != list(held.held_identity)
    ):
        raise GPUControllerError(f"{label} held-authority binding differs")


def validate_package_controller_evidence(
    held: HeldAuthority, publication_held: HeldAuthority,
    materialization_held: HeldAuthority, report: Mapping[str, Any],
) -> dict[str, Any]:
    value = strict_json(held.raw, label="fresh package controller evidence")
    _self_digested(value, "evidence_digest", PACKAGE_CONTROLLER_EVIDENCE_DIGEST)
    expected_fields = {
        "schema_version", "status", "single_attempt", "retry_allowed",
        "launch_allowed", "attempt", "snapshot", "overlay", "controller",
        "python", "materializer", "child", "publication", "ssh_performed",
        "slurm_performed", "srun_performed", "gpu_attempt_claimed",
        "renderer_invoked", "evidence_digest",
    }
    package = value.get("publication")
    if (
        set(value) != expected_fields
        or value.get("schema_version") != PACKAGE_CONTROLLER_EVIDENCE_SCHEMA
        or value.get("status") != "PASS_R64_HOLD_PACKAGE_RECEIPT_GATED"
        or value.get("single_attempt") is not True
        or value.get("retry_allowed") is not False
        or value.get("launch_allowed") is not False
        or value.get("ssh_performed") is not False
        or value.get("slurm_performed") is not False
        or value.get("srun_performed") is not False
        or value.get("gpu_attempt_claimed") is not False
        or value.get("renderer_invoked") is not False
        or type(package) is not dict
        or set(package) != {
            "publication_receipt", "publication_receipt_digest",
            "materialization_receipt", "materialization_receipt_digest",
            "package_root", "release", "production", "gpu_attempt_claimed",
            "srun_performed", "file_count", "directory_count",
        }
    ):
        raise GPUControllerError("fresh package controller semantics differ")
    _row_matches_held(
        package.get("publication_receipt"), publication_held,
        label="package publication receipt",
    )
    _row_matches_held(
        package.get("materialization_receipt"), materialization_held,
        label="package materialization receipt",
    )
    production = report["production"]
    if (
        package.get("publication_receipt_digest")
        != PACKAGE_PUBLICATION_RECEIPT_DIGEST
        or package.get("materialization_receipt_digest")
        != MATERIALIZATION_REPORT_DIGEST
        or package.get("package_root") != {
            "path": str(PACKAGE_ROOT), "identity": PACKAGE_ROOT_IDENTITY,
        }
        or package.get("release") != {
            "file_count": 25,
            "manifest_digest": report["release"]["manifest_digest"],
        }
        or package.get("production") != {
            "identity_count": 26,
            "identity_roles": list(IDENTITY_ROLES),
            "identity_set_digest": production["identity_set_digest"],
            "inner_outer_crosslink": production["inner_outer_crosslink"],
        }
        or package.get("gpu_attempt_claimed") is not False
        or package.get("srun_performed") is not False
        or package.get("file_count") != 39
        or type(package.get("directory_count")) is not int
        or package["directory_count"] <= 0
    ):
        raise GPUControllerError("fresh package controller crosslink differs")
    return value


def validate_composite_cpu_admission(
    receipt_held: HeldAuthority, evidence_held: HeldAuthority,
    publication_held: HeldAuthority, materialization_held: HeldAuthority,
    package_controller_held: HeldAuthority, report: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = strict_json(
        receipt_held.raw, label="fresh composite CPU admission receipt",
    )
    _self_digested(receipt, "receipt_digest", COMPOSITE_CPU_RECEIPT_DIGEST)
    package = receipt.get("package")
    rows = receipt.get("rank_rows")
    production = report["production"]
    if (
        set(receipt) != COMPOSITE_CPU_RECEIPT_FIELDS
        or receipt.get("schema_version") != COMPOSITE_CPU_SCHEMA
        or receipt.get("status")
        != "PASS_COMPOSITE_CPU_EXACT26_ACTIVATION_IMPORT_HOLD"
        or receipt.get("holder_job_id") != HOLDER_JOB_ID
        or receipt.get("node") != NODE
        or type(receipt.get("slurm_step_id")) is not str
        or not receipt["slurm_step_id"].isascii()
        or not receipt["slurm_step_id"].isdecimal()
        or str(int(receipt["slurm_step_id"])) != receipt["slurm_step_id"]
        or int(receipt["slurm_step_id"]) <= 0
        or receipt.get("world_size") != 4
        or receipt.get("rank_count") != 4
        or type(rows) is not list or len(rows) != 4
        or [row.get("rank") for row in rows if type(row) is dict]
        != [0, 1, 2, 3]
        or len({row.get("pid") for row in rows if type(row) is dict}) != 4
        or type(package) is not dict
        or package != {
            "root": str(PACKAGE_ROOT),
            "root_identity": PACKAGE_ROOT_IDENTITY,
            "publication_receipt_sha256": hashlib.sha256(
                publication_held.raw
            ).hexdigest(),
            "publication_receipt_digest": PACKAGE_PUBLICATION_RECEIPT_DIGEST,
            "materialization_receipt_sha256": hashlib.sha256(
                materialization_held.raw
            ).hexdigest(),
            "materialization_receipt_digest": MATERIALIZATION_REPORT_DIGEST,
            "package_controller_evidence_sha256": hashlib.sha256(
                package_controller_held.raw
            ).hexdigest(),
            "package_controller_evidence_digest":
            PACKAGE_CONTROLLER_EVIDENCE_DIGEST,
            "release_file_count": 25,
            "release_manifest_digest": report["release"]["manifest_digest"],
            "production_identity_count": 26,
            "identity_roles": list(IDENTITY_ROLES),
            "identity_set_digest": production["identity_set_digest"],
            "inner_outer_crosslink": production["inner_outer_crosslink"],
        }
        or receipt.get("isolated_runtime") != {
            "python_flags": ["-I", "-S", "-B"], "isolated": 1,
            "no_site": 1, "dont_write_bytecode": True,
            "entry_via_proc_self_fd": True,
        }
        or receipt.get("private_parent_fd") != {
            "synthetic_model_capture": True,
            "captured_parent_omitted": True,
            "captured_parent_closed_or_reused": True,
            "frozen_validator_rejected": True,
            "r5f_validator_accepted": True,
            "r5f_pread_path_exercised": True,
        }
        or receipt.get("shared_ofd_pread") != {
            "rank_count": 4, "all_reads_exact": True,
            "offsets_unchanged": True,
        }
        or receipt.get("module_binding") != {
            "module_name": "infer_lora",
            "base_infer_lora_same_object": True,
            "object_cli_applied_to_base_module": True,
            "translated_publication_applied_to_base_module": True,
            "legacy_module_instance_count": 1,
            "duplicate_legacy_module_loaded": False,
        }
        or receipt.get("activation_import")
        != expected_composite_cpu_activation_import()
        or receipt.get("side_effects") != {
            "gpu_requested": False, "torch_imported": False,
            "renderer_or_vae_loaded": False, "publication_performed": False,
        }
        or receipt.get("cache_lifecycle") != {
            "admission_cache_root": (
                "/tmp/bernini-case01-object-trajectory-r5f-v4-composite-cpu-"
                f"job{HOLDER_JOB_ID}-step{receipt.get('slurm_step_id')}-cache"
            ),
            "admission_cache_fresh": True,
            "admission_cache_cleanup_performed": True,
            "admission_cache_absent_terminal": True,
            "production_rank_cache": str(RANK_CACHE_ROOT),
            "production_rank_cache_untouched": True,
            "production_rank_cache_absent_before_and_after": True,
        }
        or receipt.get("process_cleanup") != {
            "all_rank_returncodes_zero": True, "rank_processes_zero": True,
            "torchrun_processes_zero": True, "child_processes_terminal": True,
        }
        or receipt.get("launch_allowed") is not False
    ):
        raise GPUControllerError("fresh composite CPU receipt differs")
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != COMPOSITE_CPU_RANK_ROW_FIELDS:
            raise GPUControllerError("composite CPU rank row differs")
        unsigned = dict(row); claimed = unsigned.pop("rank_digest", None)
        if (
            row.get("rank") != index
            or type(row.get("pid")) is not int or row["pid"] <= 1
            or type(row.get("private_parent_fd_number")) is not int
            or row["private_parent_fd_number"] < 3
            or type(row.get("private_parent_replacement_inode")) is not int
            or row["private_parent_replacement_inode"] <= 0
            or row.get("pread_bytes_sha256")
            != "08e33aedf25337c87eb15e08c32a58f6f4caa21fe073d00b53014c57f8d148e0"
            or row.get("pread_offset_before") != 13
            or row.get("pread_offset_after") != 13
            or row.get("activation_callback_import_module")
            != "bernini.pipeline"
            or row.get("activation_import_before_callback_return") is not True
            or row.get("captured_vendor_finder_preinstalled") is not True
            or row.get("captured_vendor_finder_count") != 1
            or row.get("captured_vendor_loader_type")
            != "_CapturedVendorLoader"
            or row.get("captured_vendor_spec_loader_type")
            != "_CapturedVendorLoader"
            or row.get("captured_vendor_loader_is_spec_loader") is not True
            or row.get("captured_vendor_cached_is_none") is not True
            or claimed != object_digest(unsigned)
        ):
            raise GPUControllerError("composite CPU rank proof differs")

    evidence = strict_json(
        evidence_held.raw, label="fresh composite CPU controller evidence",
    )
    _self_digested(evidence, "evidence_digest", COMPOSITE_CPU_EVIDENCE_DIGEST)
    if (
        set(evidence) != COMPOSITE_CPU_EVIDENCE_FIELDS
        or evidence.get("schema_version") != COMPOSITE_CPU_EVIDENCE_SCHEMA
        or evidence.get("status")
        != "PASS_FRESH_CANARY_V3_COMPOSITE_CPU_CONTROLLER"
        or evidence.get("holder_job_id") != HOLDER_JOB_ID
        or evidence.get("node") != NODE
        or evidence.get("slurm_step_id") != receipt["slurm_step_id"]
        or evidence.get("single_srun_attempt") is not True
        or evidence.get("retry_allowed") is not False
        or evidence.get("srun_count") != 1
        or evidence.get("srun_ntasks") != 1
        or evidence.get("real_rank_process_count") != 4
        or evidence.get("cpus_per_task") != 8
        or evidence.get("gpu_count") != 0
        or evidence.get("srun_returncode") != 0
        or evidence.get("receipt_digest") != receipt["receipt_digest"]
        or evidence.get("stderr_empty") is not True
        or evidence.get("process_group_zero") is not True
        or evidence.get("launch_allowed") is not False
        or evidence.get("renderer_or_vae_loaded") is not False
        or evidence.get("publication_performed") is not False
    ):
        raise GPUControllerError("fresh composite CPU evidence differs")
    _row_matches_held(evidence.get("receipt"), receipt_held, label="composite CPU")
    for label in ("stdout", "stderr"):
        row = evidence.get(label)
        if (
            type(row) is not dict
            or set(row) != {"path", "sha256", "size", "identity"}
            or type(row.get("path")) is not str
            or not Path(row["path"]).is_absolute()
            or SHA_RE.fullmatch(str(row.get("sha256"))) is None
            or type(row.get("size")) is not int or row["size"] < 0
            or type(row.get("identity")) is not list
            or len(row["identity"]) != 11
            or any(type(part) is not int for part in row["identity"])
        ):
            raise GPUControllerError(f"composite CPU {label} authority differs")
    if evidence["stderr"]["size"] != 0:
        raise GPUControllerError("composite CPU stderr is not empty")
    return receipt, evidence


class PackageGate:
    def __init__(
        self, authorities: Sequence[HeldAuthority], root: HeldDirectory,
        values: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.authorities = list(authorities); self.root = root
        self.values = {key: dict(value) for key, value in values.items()}

    def replay(self) -> None:
        # Preserve receipt-first replay order on every boundary.
        for authority in self.authorities:
            authority.replay()
        self.root.replay()

    def evidence(self) -> dict[str, Any]:
        return {
            "receipt_first_order": [str(row.path) for row in self.authorities],
            "authorities": [row.row() for row in self.authorities],
            "package_root_identity": list(self.root.held_identity),
            "receipt_first_before_package_root": True,
        }

    def close(self) -> None:
        for authority in self.authorities:
            authority.close()
        self.root.close()


def open_package_gate() -> PackageGate:
    """Open every final receipt before the package root or any target."""

    authorities: list[HeldAuthority] = []
    root: HeldDirectory | None = None
    try:
        publication = open_authority(
            PACKAGE_PUBLICATION_RECEIPT_PATH,
            expected_sha256=PACKAGE_PUBLICATION_RECEIPT_SHA256,
            expected_size=PACKAGE_PUBLICATION_RECEIPT_SIZE,
            expected_mode=RECEIPT_MODE, maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(publication)
        publication_value = validate_publication_receipt(publication)
        materialization = open_authority(
            MATERIALIZATION_REPORT_PATH,
            expected_sha256=MATERIALIZATION_REPORT_SHA256,
            expected_size=MATERIALIZATION_REPORT_SIZE,
            expected_mode=RECEIPT_MODE, maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(materialization)
        report = validate_materialization_report(materialization)
        package_controller = open_authority(
            PACKAGE_CONTROLLER_EVIDENCE_PATH,
            expected_sha256=PACKAGE_CONTROLLER_EVIDENCE_SHA256,
            expected_size=PACKAGE_CONTROLLER_EVIDENCE_SIZE,
            expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(package_controller)
        package_controller_value = validate_package_controller_evidence(
            package_controller, publication, materialization, report,
        )
        composite_cpu = open_authority(
            COMPOSITE_CPU_RECEIPT_PATH,
            expected_sha256=COMPOSITE_CPU_RECEIPT_SHA256,
            expected_size=COMPOSITE_CPU_RECEIPT_SIZE,
            expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(composite_cpu)
        composite_cpu_evidence = open_authority(
            COMPOSITE_CPU_EVIDENCE_PATH,
            expected_sha256=COMPOSITE_CPU_EVIDENCE_SHA256,
            expected_size=COMPOSITE_CPU_EVIDENCE_SIZE,
            expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        )
        authorities.append(composite_cpu_evidence)
        composite_cpu_value, composite_cpu_evidence_value = (
            validate_composite_cpu_admission(
                composite_cpu, composite_cpu_evidence, publication,
                materialization, package_controller, report,
            )
        )
        root = open_directory(
            PACKAGE_ROOT, expected_identity=PACKAGE_ROOT_IDENTITY,
        )
        if publication_value["target_root_identity"] != list(root.held_identity):
            raise GPUControllerError("publication/package root binding differs")
        gate = PackageGate(authorities, root, {
            "publication": publication_value, "materialization": report,
            "package_controller": package_controller_value,
            "composite_cpu": composite_cpu_value,
            "composite_cpu_evidence": composite_cpu_evidence_value,
        })
        gate.replay(); return gate
    except BaseException:
        for authority in authorities:
            authority.close()
        if root is not None:
            root.close()
        raise


def _exact_names(path: Path, expected: set[str], *, label: str) -> None:
    info = os.lstat(path)
    if (
        not stat.S_ISDIR(info.st_mode) or info.st_uid != REMOTE_UID
        or info.st_gid != REMOTE_GID or path.resolve(strict=True) != path
    ):
        raise GPUControllerError(f"{label} directory differs")
    with os.scandir(path) as entries:
        names = {entry.name for entry in entries}
    if names != expected:
        raise GPUControllerError(f"{label} exact names differ")


def require_fresh_outputs() -> None:
    # Fresh admissions are immutable siblings, never package-local outputs.
    _exact_names(PACKAGE_ROOT / "evidence", set(), label="pre-GPU evidence")
    _exact_names(PACKAGE_ROOT / "logs", set(), label="pre-GPU logs")
    _exact_names(OUTPUT_ROOT, set(), label="pre-GPU output")
    _exact_names(PACKAGE_ROOT / "final", set(), label="pre-GPU final")
    _exact_names(PACKAGE_ROOT / "runtime", set(), label="pre-GPU runtime")
    for path in (
        ATTEMPT_PATH, DISPATCH_PATH, EVIDENCE_PATH, READY_PLAN_PATH,
        RUNTIME_RECEIPT_PATH, RANK_CACHE_RECEIPT_PATH,
        STDOUT_PATH, STDERR_PATH, OUTPUT_REPORT_PATH, RUNNER_ATTESTATION_PATH,
    ):
        if os.path.lexists(path):
            raise GPUControllerError(f"fresh GPU target differs: {path}")


def _open_package_identities(
    report: Mapping[str, Any],
) -> tuple[dict[str, HeldAuthority], dict[str, Any]]:
    input_row = report["launch"]["input"]
    input_authority = open_authority(
        LAUNCH_INPUT_PATH, expected_sha256=input_row["sha256"],
        expected_size=input_row["size"], expected_mode=FILE_MODE,
        maximum_size=MAX_JSON_SIZE,
    )
    try:
        launch_input = strict_json(input_authority.raw, label="HOLD launch input")
        if (
            launch_input.get("schema_version")
            != "case01-object-trajectory-exact5-hold-launch-input-auh-v3"
            or launch_input.get("entry_mode") != "trusted_stdin"
            or launch_input.get("campaign_mode") != CAMPAIGN
            or launch_input.get("holder_job_id") != HOLDER_JOB_ID
            or launch_input.get("expected_node") != NODE
            or launch_input.get("expected_allocation_gpu_count") != GPU_COUNT
            or launch_input.get("output_report") != str(OUTPUT_REPORT_PATH)
            or launch_input.get("runner_attestation") != str(RUNNER_ATTESTATION_PATH)
            or launch_input.get("authority_root") != str(AUTHORITY_ROOT)
            or launch_input.get("rank_cache_root") != str(RANK_CACHE_ROOT)
            or launch_input.get("identities")
            != report["launch"]["release"]["identities"]
        ):
            raise GPUControllerError("HOLD launch input semantics differ")
        result: dict[str, HeldAuthority] = {"launch_input": input_authority}
        for role in IDENTITY_ROLES:
            row = launch_input["identities"][role]
            mode = FILE_MODE
            authority = open_authority(
                Path(row["path"]), expected_sha256=row["sha256"],
                expected_size=row["size"], expected_mode=None,
                maximum_size=(
                    MAX_RUNTIME_EXECUTABLE_SIZE
                    if role in {"python", "ffmpeg"} else MAX_SOURCE_SIZE
                ),
                executable=role in EXECUTABLE_ROLES,
                expected_uid=None, expected_gid=None,
            )
            result[role] = authority
        if tuple(key for key in result if key != "launch_input") != IDENTITY_ROLES:
            raise GPUControllerError("held exact26 order differs")
        return result, launch_input
    except BaseException:
        if "result" in locals():
            for authority in result.values():
                authority.close()
        else:
            input_authority.close()
        raise


def validate_site_packages_layout(
    launch_input: Mapping[str, Any],
) -> Path:
    identities = launch_input.get("identities")
    if type(identities) is not dict:
        raise GPUControllerError("Torch identity layout is absent")
    expected = {
        "torchrun_source": "torch/distributed/run.py",
        "torchrun_handler_source": (
            "torch/distributed/elastic/multiprocessing/"
            "subprocess_handler/subprocess_handler.py"
        ),
        "torch_local_agent_source": (
            "torch/distributed/elastic/agent/server/local_elastic_agent.py"
        ),
        "torch_dynamic_rendezvous_source": (
            "torch/distributed/elastic/rendezvous/dynamic_rendezvous.py"
        ),
        "torch_multiprocessing_api_source": (
            "torch/distributed/elastic/multiprocessing/api.py"
        ),
    }
    if any(
        type(identities.get(role)) is not dict
        or identities[role].get("path") != str(SITE_PACKAGES_ROOT / relative)
        for role, relative in expected.items()
    ):
        raise GPUControllerError("Torch source/site-packages layout differs")
    return SITE_PACKAGES_ROOT


def derive_ready_plan(hold_raw: bytes) -> tuple[dict[str, Any], bytes]:
    hold = strict_json(hold_raw, label="sealed HOLD plan")
    unsigned = dict(hold); claimed = unsigned.pop("plan_digest", None)
    tasks = hold.get("tasks")
    if (
        claimed != object_digest(unsigned)
        or hold.get("schema_version") != READY_PLAN_SCHEMA
        or hold.get("status") != "HOLD_INCOMPLETE_PRODUCER_OR_AUTHORITY"
        or hold.get("production_ready") is not False
        or hold.get("launch_allowed") is not False
        or hold.get("hold_reasons") != ["explicit_launch_release_not_granted"]
        or hold.get("arms") != list(ARM_ORDER)
        or hold.get("task_count") != 5
        or type(tasks) is not list
        or [row.get("task_id") for row in tasks] != list(TASK_IDS)
        or [row.get("oracle_arm") for row in tasks] != list(ARM_ORDER)
        or any(row.get("source_onset_policy") != "hard1_every_step" for row in tasks)
        or any(row.get("output", {}).get("create_only") is not True for row in tasks)
        or any(
            row.get("output", {}).get("video_path")
            != str(OUTPUT_ROOT / f"{task_id}.mp4")
            for row, task_id in zip(tasks, TASK_IDS)
        )
    ):
        raise GPUControllerError("sealed HOLD plan is not the exact release base")
    ready = dict(hold)
    ready["status"] = "READY_FOR_EXPLICIT_LOCAL_LAUNCH"
    ready["launch_allowed"] = True
    ready["hold_reasons"] = []
    ready.pop("plan_digest")
    ready["plan_digest"] = object_digest(ready)
    raw = canonical(ready) + b"\n"
    # Exact transform: apart from the digest, precisely three semantic leaves.
    changed = {
        key for key in ready
        if key != "plan_digest" and ready.get(key) != hold.get(key)
    }
    if changed != {"status", "launch_allowed", "hold_reasons"}:
        raise GPUControllerError("READY plan transform differs")
    return ready, raw


RUNNER_ARGUMENT_FLAGS = (
    "--campaign-mode", "--plan", "--plan-sha256", "--output-report",
    "--runner-attestation", "--runner-sha256", "--bridge-script",
    "--bridge-script-sha256", "--adapter-script", "--adapter-script-sha256",
    "--eval-v1-source", "--eval-v1-source-sha256", "--eval-v2-source",
    "--eval-v2-source-sha256", "--model-authority-source",
    "--model-authority-source-sha256", "--python", "--python-sha256",
    "--ffmpeg-executable", "--ffmpeg-executable-sha256", "--torchrun-source",
    "--torchrun-source-sha256", "--torchrun-handler-source",
    "--torchrun-handler-source-sha256", "--torch-local-agent-source",
    "--torch-local-agent-source-sha256", "--torch-dynamic-rendezvous-source",
    "--torch-dynamic-rendezvous-source-sha256",
    "--torch-multiprocessing-api-source",
    "--torch-multiprocessing-api-source-sha256", "--model-root",
    "--model-manifest", "--model-manifest-sha256", "--bernini-root",
    "--veomni-root", "--authority-root", "--rank-cache-root",
    "--holder-job-id", "--expected-node", "--expected-allocation-gpu-count",
)


def build_runner_arguments(
    launch_input: Mapping[str, Any], production_rows: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    rows = production_rows
    pairs = (
        ("--campaign-mode", CAMPAIGN),
        ("--plan", rows["plan"]["path"]),
        ("--plan-sha256", rows["plan"]["sha256"]),
        ("--output-report", launch_input["output_report"]),
        ("--runner-attestation", launch_input["runner_attestation"]),
        ("--runner-sha256", rows["runner"]["sha256"]),
        ("--bridge-script", rows["bridge"]["path"]),
        ("--bridge-script-sha256", rows["bridge"]["sha256"]),
        ("--adapter-script", rows["adapter"]["path"]),
        ("--adapter-script-sha256", rows["adapter"]["sha256"]),
        ("--eval-v1-source", rows["eval_v1"]["path"]),
        ("--eval-v1-source-sha256", rows["eval_v1"]["sha256"]),
        ("--eval-v2-source", rows["eval_v2"]["path"]),
        ("--eval-v2-source-sha256", rows["eval_v2"]["sha256"]),
        ("--model-authority-source", rows["model_authority"]["path"]),
        ("--model-authority-source-sha256", rows["model_authority"]["sha256"]),
        ("--python", rows["python"]["path"]),
        ("--python-sha256", rows["python"]["sha256"]),
        ("--ffmpeg-executable", rows["ffmpeg"]["path"]),
        ("--ffmpeg-executable-sha256", rows["ffmpeg"]["sha256"]),
        ("--torchrun-source", rows["torchrun_source"]["path"]),
        ("--torchrun-source-sha256", rows["torchrun_source"]["sha256"]),
        ("--torchrun-handler-source", rows["torchrun_handler_source"]["path"]),
        ("--torchrun-handler-source-sha256", rows["torchrun_handler_source"]["sha256"]),
        ("--torch-local-agent-source", rows["torch_local_agent_source"]["path"]),
        ("--torch-local-agent-source-sha256", rows["torch_local_agent_source"]["sha256"]),
        ("--torch-dynamic-rendezvous-source", rows["torch_dynamic_rendezvous_source"]["path"]),
        ("--torch-dynamic-rendezvous-source-sha256", rows["torch_dynamic_rendezvous_source"]["sha256"]),
        ("--torch-multiprocessing-api-source", rows["torch_multiprocessing_api_source"]["path"]),
        ("--torch-multiprocessing-api-source-sha256", rows["torch_multiprocessing_api_source"]["sha256"]),
        ("--model-root", launch_input["model_root"]),
        ("--model-manifest", rows["base_model_manifest"]["path"]),
        ("--model-manifest-sha256", rows["base_model_manifest"]["sha256"]),
        ("--bernini-root", launch_input["bernini_root"]),
        ("--veomni-root", launch_input["veomni_root"]),
        ("--authority-root", launch_input["authority_root"]),
        ("--rank-cache-root", launch_input["rank_cache_root"]),
        ("--holder-job-id", HOLDER_JOB_ID),
        ("--expected-node", NODE),
        ("--expected-allocation-gpu-count", str(GPU_COUNT)),
    )
    if tuple(key for key, _ in pairs) != RUNNER_ARGUMENT_FLAGS:
        raise GPUControllerError("runner argument flag order differs")
    result = [item for pair in pairs for item in pair]
    if any(type(item) is not str or not item or "\x00" in item for item in result):
        raise GPUControllerError("runner argument value differs")
    return result


_WITHDRAWN_PRE_CACHE_AUDIT_ROOT_BOOTSTRAP = r'''import base64,hashlib,io,json,os,stat,sys,types
if len(sys.argv)!=15: raise RuntimeError("GPU root bootstrap argv differs")
python_fd_raw,release_b64,release_digest,bootstrap_sha,max_gate_step,job_id,step_id,gpus_on_node,gpus_per_node,step_gpus,node_count,step_node_count,job_nodelist,step_nodelist=sys.argv[1:]
if not step_id.isascii() or not step_id.isdecimal() or str(int(step_id))!=step_id or int(step_id)<=int(max_gate_step): raise RuntimeError("GPU step ordering differs")
try: python_fd=int(python_fd_raw)
except ValueError as error: raise RuntimeError("held Python FD differs") from error
if python_fd<3 or str(python_fd)!=python_fd_raw or os.get_inheritable(python_fd) is not True: raise RuntimeError("held Python FD entry differs")
python_before=os.fstat(python_fd); os.set_inheritable(python_fd,False)
if os.get_inheritable(python_fd): raise RuntimeError("held Python FD remained inheritable")
def pairs(items):
 out={}
 for key,value in items:
  if key in out: raise RuntimeError("duplicate root JSON key")
  out[key]=value
 return out
def canonical(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def sha_ok(value): return type(value) is str and len(value)==64 and all(character in "0123456789abcdef" for character in value)
def ident(value): return [value.st_dev,value.st_ino,value.st_uid,value.st_gid,value.st_mode,value.st_nlink,value.st_rdev,value.st_size,getattr(value,"st_blocks",0),value.st_mtime_ns,value.st_ctime_ns]
def read_fd(fd,size):
 chunks=[];offset=0
 while offset<size:
  block=os.pread(fd,min(1048576,size-offset),offset)
  if not block: break
  chunks.append(block);offset+=len(block)
 raw=b"".join(chunks)
 if len(raw)!=size or os.pread(fd,1,size)!=b"": raise RuntimeError("root held read differs")
 return raw
def replay(role,row,executable=False):
 if type(row) is not dict or set(row)!={"path","sha256","size","identity","mode","nlink"}: raise RuntimeError("root identity row differs: "+role)
 path=row.get("path")
 if type(path) is not str or not os.path.isabs(path) or os.path.normpath(path)!=path or os.path.realpath(path)!=path: raise RuntimeError("root identity path differs: "+role)
 fd=os.open(path,os.O_RDONLY|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_NONBLOCK",0))
 try:
  before=os.fstat(fd);first=read_fd(fd,before.st_size);middle=os.fstat(fd);second=read_fd(fd,before.st_size);after=os.fstat(fd);named=os.lstat(path)
 except BaseException:
  os.close(fd);raise
 if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_uid!=2012 or before.st_gid!=2000 or (executable and not before.st_mode&0o111) or ident(before)!=ident(middle) or ident(before)!=ident(after) or ident(before)!=ident(named) or ident(before)!=row.get("identity") or stat.S_IMODE(before.st_mode)!=row.get("mode") or before.st_nlink!=row.get("nlink") or first!=second or len(first)!=row.get("size") or hashlib.sha256(first).hexdigest()!=row.get("sha256"): os.close(fd);raise RuntimeError("root identity replay differs: "+role)
 return fd,first,before
try: release_raw=base64.b64decode(release_b64.encode("ascii"),validate=True)
except Exception as error: raise RuntimeError("release base64 differs") from error
release=json.loads(release_raw.decode("utf-8","strict"),object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
roles=("runner","legacy_exact5_runner","object_eval","legacy_exact5_eval","frozen_runner","bridge","adapter","object_wrapper_inner","legacy_infer_alias","trajectory_projection","trajectory_scaffold_module","base_adapter","eval_v1","eval_v2","model_authority","torchrun_source","torchrun_handler_source","torch_local_agent_source","torch_dynamic_rendezvous_source","torch_multiprocessing_api_source","base_model_manifest","r64_checkpoint_manifest","python","ffmpeg","ffprobe","plan")
tasks=["case01-object-trajectory-null_before-full644","case01-object-trajectory-route_off-full644","case01-object-trajectory-trajectory_bone_only-full644","case01-object-trajectory-trajectory_dog_bone-full644","case01-object-trajectory-null_after-full644"]
arms=["null_before","route_off","trajectory_bone_only","trajectory_dog_bone","null_after"]
fields={"schema_version","entry_mode","external_root_of_trust","bash_privileged_mode","slurm_export_none","named_hold_payload_executed","ready_plan_is_derived_overlay","python_is_executed_from_held_fd","runner_is_compiled_from_captured_fd_bytes","all_exact26_named_identities_replayed_before_runner","expected_allocation_gpu_count","campaign_mode","task_count","selected_task_ids","arm_order","all_arms_attempted_exactly_once_by_runner","retry_allowed","partial_outputs_are_not_results","expected_runtime_versions","expected_gpu_name","runtime_receipt_path","rank_cache_receipt_path","rank_cache_root","runner_attestation_path","torch_package_init_authority","directory_authorities","holder_job_id","expected_node","identities","runner_arguments"}
if (type(release) is not dict or set(release)!=fields or release_raw!=canonical(release) or digest(release)!=release_digest
 or release.get("schema_version")!="case01-object-trajectory-exact5-r64-gpu-release-v4" or release.get("entry_mode")!="trusted_controller_streamed_stdin"
 or release.get("external_root_of_trust")!="receipt-gated-controller-held-bytes" or release.get("bash_privileged_mode") is not True
 or release.get("slurm_export_none") is not True or release.get("named_hold_payload_executed") is not False
 or release.get("ready_plan_is_derived_overlay") is not True or release.get("python_is_executed_from_held_fd") is not True
 or release.get("runner_is_compiled_from_captured_fd_bytes") is not True or release.get("all_exact26_named_identities_replayed_before_runner") is not True
 or release.get("expected_allocation_gpu_count")!=8 or release.get("campaign_mode")!="case01-object-trajectory-exact5-r64-engineering-oracle-v3"
 or release.get("task_count")!=5 or release.get("selected_task_ids")!=tasks or release.get("arm_order")!=arms
 or release.get("all_arms_attempted_exactly_once_by_runner") is not True or release.get("retry_allowed") is not False
 or release.get("partial_outputs_are_not_results") is not True or release.get("expected_runtime_versions")!={"hip":"6.3.42131-fa1d09cbd","torch":"2.7.1+rocm6.3"}
 or release.get("expected_gpu_name")!="AMD Instinct MI210" or release.get("holder_job_id")!=job_id
 or release.get("expected_node")!=step_nodelist or job_nodelist!=step_nodelist or gpus_on_node!="8" or gpus_per_node!="8"
 or step_gpus!="0,1,2,3,4,5,6,7" or node_count!="1" or step_node_count!="1"
 or release.get("rank_cache_root")!="/tmp/bernini-case01-object-trajectory-exact5-r64-job143808-node292-r3-rank-cache"):
 raise RuntimeError("GPU release differs")
identities=release.get("identities");arguments=release.get("runner_arguments")
if type(identities) is not dict or set(identities)!=set(roles) or len(identities)!=26 or len({row.get("path") for row in identities.values() if type(row) is dict})!=26: raise RuntimeError("exact26 production identity closure differs")
if type(arguments) is not list or len(arguments)!=80 or any(type(value) is not str or not value for value in arguments): raise RuntimeError("runner argument vector differs")
argmap=dict(zip(arguments[::2],arguments[1::2]))
if len(argmap)!=40 or argmap.get("--campaign-mode")!=release["campaign_mode"] or argmap.get("--holder-job-id")!=job_id or argmap.get("--expected-node")!=release["expected_node"] or argmap.get("--expected-allocation-gpu-count")!="8" or argmap.get("--plan")!=identities["plan"]["path"] or argmap.get("--plan-sha256")!=identities["plan"]["sha256"] or argmap.get("--runner-sha256")!=identities["runner"]["sha256"]: raise RuntimeError("runner argument binding differs")
held={};raw={};info={}
try:
 for role in roles:
  held[role],raw[role],info[role]=replay(role,identities[role],role in {"python","ffmpeg","ffprobe"})
 if ident(python_before)!=ident(info["python"]) or ident(python_before)!=ident(os.stat("/proc/self/exe")) or hashlib.sha256(read_fd(python_fd,python_before.st_size)).hexdigest()!=identities["python"]["sha256"]: raise RuntimeError("held process Python differs")
 plan=json.loads(raw["plan"].decode("utf-8","strict"),object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
 unsigned=dict(plan);claimed=unsigned.pop("plan_digest",None);rows=plan.get("tasks") if type(plan) is dict else None
 if raw["plan"]!=canonical(plan)+b"\n" or claimed!=digest(unsigned) or plan.get("schema_version")!="case01-object-trajectory-exact5-plan-v3" or plan.get("status")!="READY_FOR_EXPLICIT_LOCAL_LAUNCH" or plan.get("production_ready") is not False or plan.get("launch_allowed") is not True or plan.get("hold_reasons")!=[] or type(rows) is not list or len(rows)!=5 or [row.get("task_id") for row in rows]!=tasks or [row.get("oracle_arm") for row in rows]!=arms or any(row.get("source_onset_policy")!="hard1_every_step" for row in rows): raise RuntimeError("READY exact-five plan differs")
 checkpoint=plan.get("checkpoint_manifest")
 if type(checkpoint) is not dict or checkpoint.get("path")!=identities["r64_checkpoint_manifest"]["path"] or checkpoint.get("sha256")!=identities["r64_checkpoint_manifest"]["sha256"] or checkpoint.get("pin_complete") is not True: raise RuntimeError("R64 authority closure differs")
 runner_fd=held["runner"];os.set_inheritable(runner_fd,False)
 sys.path.insert(0,release["directory_authorities"]["site_packages"]["path"]);import torch
 names=[torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
 if torch.__version__!="2.7.1+rocm6.3" or torch.version.hip!="6.3.42131-fa1d09cbd" or not torch.cuda.is_available() or torch.cuda.device_count()!=8 or names!=["AMD Instinct MI210"]*8: raise RuntimeError("Torch/HIP/GPU runtime differs")
 runtime={"schema_version":"case01-object-trajectory-exact5-r64-gpu-controller-v4-runtime","status":"PASS_GPU_RUNTIME_BEFORE_RUNNER","holder_job_id":job_id,"node":step_nodelist,"slurm_step_id":step_id,"torch_version":torch.__version__,"hip_version":torch.version.hip,"device_count":torch.cuda.device_count(),"device_names":names,"r64_checkpoint_manifest_sha256":identities["r64_checkpoint_manifest"]["sha256"],"exact26_identity_set_digest":digest(identities),"renderer_started":False}
 runtime["receipt_digest"]=digest(runtime);runtime_raw=canonical(runtime)+b"\n";runtime_path=release["runtime_receipt_path"]
 if type(runtime_path) is not str or not os.path.isabs(runtime_path) or os.path.normpath(runtime_path)!=runtime_path or os.path.lexists(runtime_path) or os.path.realpath(os.path.dirname(runtime_path))!=os.path.dirname(runtime_path): raise RuntimeError("runtime receipt target differs")
 runtime_parent=os.open(os.path.dirname(runtime_path),os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_CLOEXEC",0))
 runtime_fd=None
 try:
  parent_info=os.fstat(runtime_parent)
  if not stat.S_ISDIR(parent_info.st_mode) or parent_info.st_uid!=2012 or parent_info.st_gid!=2000 or stat.S_IMODE(parent_info.st_mode)!=0o700: raise RuntimeError("runtime receipt parent differs")
  runtime_fd=os.open(os.path.basename(runtime_path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0),0,dir_fd=runtime_parent)
  offset=0
  while offset<len(runtime_raw):
   count=os.write(runtime_fd,runtime_raw[offset:])
   if count<=0: raise RuntimeError("runtime receipt write made no progress")
   offset+=count
  os.fsync(runtime_fd);os.fchmod(runtime_fd,0o400);os.fsync(runtime_fd);os.fsync(runtime_parent)
 finally:
  if runtime_fd is not None: os.close(runtime_fd)
  os.close(runtime_parent)
 entry={"schema_version":"full644-exploratory-matched-captured-runner-entry-authority-v1","runner_fd":runner_fd,"runner_path":identities["runner"]["path"],"runner_sha256":identities["runner"]["sha256"],"runner_identity":{key:value for key,value in zip(("device","inode","uid","gid","mode","nlink","rdev","size","blocks","mtime_ns","ctime_ns"),ident(info["runner"]))},"python_fd":python_fd,"python_path":identities["python"]["path"],"python_sha256":identities["python"]["sha256"],"python_identity":{key:value for key,value in zip(("device","inode","uid","gid","mode","nlink","rdev","size","blocks","mtime_ns","ctime_ns"),ident(python_before))},"release_digest":release_digest,"bootstrap_sha256":bootstrap_sha,"entry_method":"slurm-spooled-or-trusted-stdin-held-python-fd-v1","slurm_export_none_required":True,"bash_privileged_startup_required":True,"captured_source_entry":True}
 entry["authority_digest"]=digest(entry)
 os.environ.clear();os.environ.update({"FULL644_MATCHED_CAPTURED_RUNNER_ENTRY_AUTHORITY":canonical(entry).decode("utf-8"),"SLURM_JOB_ID":job_id,"SLURM_STEP_ID":step_id,"SLURM_GPUS_ON_NODE":gpus_on_node,"SLURM_GPUS_PER_NODE":gpus_per_node,"SLURM_STEP_GPUS":step_gpus,"SLURM_NNODES":node_count,"SLURM_STEP_NUM_NODES":step_node_count,"SLURM_JOB_NODELIST":job_nodelist,"SLURM_STEP_NODELIST":step_nodelist})
 runner_source=raw["runner"].decode("utf-8","strict");runner_path=identities["runner"]["path"]
 sys.argv=[runner_path,*arguments]
 module=types.ModuleType("__main__");module.__file__=runner_path;module.__package__=None;module.__loader__=None;module.__spec__=None;module.__cached__=None;module.__builtins__=__builtins__;sys.modules["__main__"]=module
 exec(compile(runner_source,runner_path,"exec",dont_inherit=True),module.__dict__)
finally:
 for role,fd in held.items():
  if role!="runner":
   try: os.close(fd)
   except OSError: pass
'''


# This is the only bootstrap consumed by ``build_production_release``.  It is
# intentionally self-contained because ``-I -S`` removes the environment's
# site-packages from sys.path.  The compute process holds and replays all four
# directory anchors, activates the site root derived from the five pinned
# Torch sources, captures the frozen runner's stdout, and publishes a truthful
# compute-local rank-cache inventory rather than asking the login host to stat
# a compute-node /tmp path.
ROOT_BOOTSTRAP = r'''import base64,hashlib,importlib.machinery,importlib.util,io,json,os,socket,stat,sys,types
if len(sys.argv)!=15: raise RuntimeError("GPU root bootstrap argv differs")
python_fd_raw,release_b64,release_digest,bootstrap_sha,max_gate_step,job_id,step_id,gpus_on_node,gpus_per_node,step_gpus,node_count,step_node_count,job_nodelist,step_nodelist=sys.argv[1:]
if not step_id.isascii() or not step_id.isdecimal() or str(int(step_id))!=step_id or int(step_id)<=int(max_gate_step): raise RuntimeError("GPU step ordering differs")
try: python_fd=int(python_fd_raw)
except ValueError as error: raise RuntimeError("held Python FD differs") from error
if python_fd<3 or str(python_fd)!=python_fd_raw or os.get_inheritable(python_fd) is not True: raise RuntimeError("held Python FD entry differs")
python_before=os.fstat(python_fd);os.set_inheritable(python_fd,False)
if os.get_inheritable(python_fd): raise RuntimeError("held Python FD remained inheritable")
def pairs(items):
 out={}
 for key,value in items:
  if key in out: raise RuntimeError("duplicate root JSON key")
  out[key]=value
 return out
def canonical(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def sha_ok(value): return type(value) is str and len(value)==64 and all(character in "0123456789abcdef" for character in value)
def ident(value): return [value.st_dev,value.st_ino,value.st_uid,value.st_gid,value.st_mode,value.st_nlink,value.st_rdev,value.st_size,getattr(value,"st_blocks",0),value.st_mtime_ns,value.st_ctime_ns]
def anchor(value): return [value.st_dev,value.st_ino,value.st_uid,value.st_gid,stat.S_IFMT(value.st_mode),stat.S_IMODE(value.st_mode),value.st_rdev]
def anchor_row(row):
 value=row.get("identity") if type(row) is dict else None
 if type(value) is not list or len(value)!=11: raise RuntimeError("directory identity width differs")
 return [value[0],value[1],value[2],value[3],stat.S_IFMT(value[4]),stat.S_IMODE(value[4]),value[6]]
def read_fd(fd,size,maximum=4294967296):
 if type(size) is not int or size<0 or size>maximum: raise RuntimeError("root held size differs")
 chunks=[];offset=0
 while offset<size:
  block=os.pread(fd,min(1048576,size-offset),offset)
  if not block: break
  chunks.append(block);offset+=len(block)
 raw=b"".join(chunks)
 if len(raw)!=size or os.pread(fd,1,size)!=b"": raise RuntimeError("root held read differs")
 return raw
def replay(role,row,executable=False):
 if type(row) is not dict or set(row)!={"path","sha256","size","identity","mode","nlink"}: raise RuntimeError("root identity row differs: "+role)
 path=row.get("path")
 if type(path) is not str or not os.path.isabs(path) or os.path.normpath(path)!=path or os.path.realpath(path)!=path: raise RuntimeError("root identity path differs: "+role)
 fd=os.open(path,os.O_RDONLY|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_NONBLOCK",0))
 try:
  before=os.fstat(fd);first=read_fd(fd,before.st_size);middle=os.fstat(fd);second=read_fd(fd,before.st_size);after=os.fstat(fd);named=os.lstat(path)
 except BaseException:
  os.close(fd);raise
 if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or (executable and not before.st_mode&0o111) or ident(before)!=ident(middle) or ident(before)!=ident(after) or ident(before)!=ident(named) or ident(before)!=row.get("identity") or stat.S_IMODE(before.st_mode)!=row.get("mode") or before.st_nlink!=row.get("nlink") or first!=second or len(first)!=row.get("size") or hashlib.sha256(first).hexdigest()!=row.get("sha256"): os.close(fd);raise RuntimeError("root identity replay differs: "+role)
 return fd,first,before
def open_bound_directory(label,row,expected_path):
 if type(row) is not dict or set(row)!={"path","identity"} or row.get("path")!=expected_path or not os.path.isabs(expected_path) or os.path.normpath(expected_path)!=expected_path or os.path.realpath(expected_path)!=expected_path: raise RuntimeError("root directory authority differs: "+label)
 fd=os.open(expected_path,os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_CLOEXEC",0))
 opened=os.fstat(fd);named=os.lstat(expected_path)
 if not stat.S_ISDIR(opened.st_mode) or anchor(opened)!=anchor(named) or anchor(opened)!=anchor_row(row): os.close(fd);raise RuntimeError("root directory replay differs: "+label)
 return fd
def replay_directory(label,fd,row):
 if anchor(os.fstat(fd))!=anchor_row(row) or anchor(os.lstat(row["path"]))!=anchor_row(row): raise RuntimeError("held root directory changed: "+label)
def replay_empty_directory(label,fd,path,expected_identity,expected_owner):
 opened=os.fstat(fd);named=os.lstat(path)
 with os.scandir(fd) as entries: names=sorted(entry.name for entry in entries)
 if not stat.S_ISDIR(opened.st_mode) or ident(opened)!=expected_identity or ident(named)!=expected_identity or stat.S_IMODE(opened.st_mode)!=0o700 or (opened.st_uid,opened.st_gid)!=expected_owner or os.path.realpath(path)!=path or names!=[]: raise RuntimeError("held empty directory changed: "+label)
def parse_json(raw,label,newline=True):
 try: value=json.loads(raw.decode("utf-8","strict"),object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
 except Exception as error: raise RuntimeError("invalid root JSON: "+label) from error
 if type(value) is not dict or raw!=(canonical(value)+b"\n" if newline else canonical(value)): raise RuntimeError("noncanonical root JSON: "+label)
 return value
def publish_json(path,value,digest_field,runtime_fd):
 unsigned=dict(value);unsigned.pop(digest_field,None);value[digest_field]=digest(unsigned);raw=canonical(value)+b"\n"
 if os.path.dirname(path)!=directory_rows["runtime"]["path"] or os.path.basename(path) in {"",".",".."} or os.path.lexists(path): raise RuntimeError("compute receipt target differs")
 write_fd=None
 try:
  write_fd=os.open(os.path.basename(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0),0,dir_fd=runtime_fd)
  offset=0
  while offset<len(raw):
   count=os.write(write_fd,raw[offset:])
   if count<=0: raise RuntimeError("compute receipt write made no progress")
   offset+=count
  os.fsync(write_fd);os.fchmod(write_fd,0o400);os.fsync(write_fd);os.fsync(runtime_fd)
 finally:
  if write_fd is not None: os.close(write_fd)
 held_fd=os.open(os.path.basename(path),os.O_RDONLY|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0),dir_fd=runtime_fd)
 before=os.fstat(held_fd);named=os.lstat(path);observed=read_fd(held_fd,before.st_size,67108864)
 if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or stat.S_IMODE(before.st_mode)!=0o400 or ident(before)!=ident(named) or observed!=raw: os.close(held_fd);raise RuntimeError("published compute receipt differs")
 return held_fd,raw
def capture_json(path,parent_fd,mode,label):
 if os.path.dirname(path) not in {directory_rows["final"]["path"],directory_rows["runtime"]["path"]}: raise RuntimeError(label+" parent differs")
 fd=os.open(os.path.basename(path),os.O_RDONLY|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0),dir_fd=parent_fd)
 before=os.fstat(fd);raw=read_fd(fd,before.st_size,67108864);after=os.fstat(fd);named=os.lstat(path)
 if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or stat.S_IMODE(before.st_mode)!=mode or ident(before)!=ident(after) or ident(before)!=ident(named): os.close(fd);raise RuntimeError(label+" identity differs")
 return fd,raw,parse_json(raw,label)
def capture_leaf_at(parent_fd,parent_path,basename,maximum,label):
 if type(basename) is not str or basename in {"",".",".."} or "/" in basename: raise RuntimeError(label+" basename differs")
 path=parent_path+"/"+basename
 fd=os.open(basename,os.O_RDONLY|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_NONBLOCK",0),dir_fd=parent_fd)
 try:
  before=os.fstat(fd);first=read_fd(fd,before.st_size,maximum);middle=os.fstat(fd);second=read_fd(fd,before.st_size,maximum);after=os.fstat(fd);named=os.lstat(path)
 except BaseException:
  os.close(fd);raise
 if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or stat.S_IMODE(before.st_mode)!=0o400 or ident(before)!=ident(middle) or ident(before)!=ident(after) or ident(before)!=ident(named) or first!=second: os.close(fd);raise RuntimeError(label+" held leaf differs")
 return fd,first,before
def cache_inventory(root,expected_uid,expected_gid):
 rows=[];file_bytes=0
 def walk(parent_fd,name,relative,depth):
  nonlocal file_bytes
  if depth>32 or len(rows)>=50000: raise RuntimeError("rank-cache inventory bound differs")
  named=os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
  flags=os.O_RDONLY|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_NONBLOCK",0)
  if stat.S_ISDIR(named.st_mode): flags|=getattr(os,"O_DIRECTORY",0)
  try: fd=os.open(name,flags,dir_fd=parent_fd)
  except OSError as error: raise RuntimeError("rank-cache openat differs") from error
  try:
   before=os.fstat(fd)
   if ident(before)!=ident(named) or before.st_uid!=expected_uid or before.st_gid!=expected_gid: raise RuntimeError("rank-cache held identity differs")
   if stat.S_ISDIR(before.st_mode):
    if stat.S_IMODE(before.st_mode)!=0o700: raise RuntimeError("rank-cache directory mode differs")
    rows.append({"kind":"directory","path":relative,"mode":0o700,"identity":ident(before)})
    with os.scandir(fd) as entries: children=sorted(entry.name for entry in entries)
    for child in children:
     if child in {"",".",".."} or "/" in child: raise RuntimeError("rank-cache basename differs")
     walk(fd,child,child if not relative else relative+"/"+child,depth+1)
   elif stat.S_ISREG(before.st_mode):
    mode=stat.S_IMODE(before.st_mode)
    if before.st_nlink!=1 or mode&0o077 or before.st_size>1073741824 or file_bytes+before.st_size>4294967296: raise RuntimeError("rank-cache file policy differs")
    raw=read_fd(fd,before.st_size,1073741824);file_bytes+=len(raw)
    rows.append({"kind":"file","path":relative,"mode":mode,"identity":ident(before),"size":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
   else: raise RuntimeError("rank-cache special leaf differs")
   after=os.fstat(fd);renamed=os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
   if ident(before)!=ident(after) or ident(before)!=ident(renamed): raise RuntimeError("rank-cache entry changed during held traversal")
  finally: os.close(fd)
 parent=os.path.dirname(root);basename=os.path.basename(root)
 parent_fd=os.open(parent,os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_CLOEXEC",0))
 try: walk(parent_fd,basename,"",0)
 finally: os.close(parent_fd)
 return rows,file_bytes
def residual_rank_pids(cache_root,step_id):
 needles=[cache_root.encode("utf-8"),("SLURM_STEP_ID="+step_id).encode("ascii")];matched=[];scanned=0;vanished=[]
 for name in os.listdir("/proc"):
  if not name.isdecimal() or int(name)==os.getpid(): continue
  try:
   proc_info=os.stat("/proc/"+name)
   if proc_info.st_uid!=os.getuid(): continue
   scanned+=1
   raw=b""
   for leaf in ("environ","cmdline"):
    with open("/proc/"+name+"/"+leaf,"rb",buffering=0) as stream:
     block=stream.read(16777216)
     if len(block)>=16777216: raise RuntimeError("process scan bound differs")
     raw+=block
   if any(needle in raw for needle in needles): matched.append(int(name))
  except (FileNotFoundError,ProcessLookupError): vanished.append(int(name))
  except PermissionError as error: raise RuntimeError("same-UID process scan is unreadable: "+name) from error
 return {"matched_pids":sorted(matched),"same_uid_processes_scanned":scanned,"vanished_pids":sorted(set(vanished)),"unreadable_pids":[]}
def mount_authority(path):
 def unescape(value):
  return value.replace("\\040"," ").replace("\\011","\t").replace("\\012","\n").replace("\\134","\\")
 with open("/proc/self/mountinfo","rb",buffering=0) as stream: mount_raw=stream.read(16777216)
 if not mount_raw or len(mount_raw)>=16777216: raise RuntimeError("mountinfo bound differs")
 candidates=[]
 for raw_line in mount_raw.splitlines():
  fields=raw_line.decode("utf-8","strict").split(" ")
  try: separator=fields.index("-")
  except ValueError: raise RuntimeError("mountinfo syntax differs")
  if separator<6 or len(fields)<separator+4: raise RuntimeError("mountinfo row differs")
  mountpoint=unescape(fields[4])
  if path==mountpoint or path.startswith(mountpoint.rstrip("/")+"/"):
   candidates.append((len(mountpoint),fields,separator,mountpoint))
 if not candidates: raise RuntimeError("cache mountpoint is absent")
 _,fields,separator,mountpoint=max(candidates,key=lambda row:row[0])
 result={"mount_id":fields[0],"parent_mount_id":fields[1],"major_minor":fields[2],"mount_root":unescape(fields[3]),"mount_point":mountpoint,"mount_options":fields[5].split(","),"optional_fields":fields[6:separator],"fs_type":fields[separator+1],"mount_source":unescape(fields[separator+2]),"super_options":fields[separator+3].split(","),"mountinfo_sha256":hashlib.sha256(mount_raw).hexdigest()}
 if result["mount_point"]!="/" or result["fs_type"]!="ext4" or result["mount_source"]!="/dev/mapper/vgroot-lvroot": raise RuntimeError("rank cache node-local mount authority differs")
 return result
def cgroup_snapshot(job_id,step_id):
 expected="/system.slice/slurmstepd.scope/job_"+job_id+"/step_"+step_id+"/user/task_0"
 with open("/proc/self/cgroup","rb",buffering=0) as stream: membership_raw=stream.read(1048576)
 if membership_raw!=(("0::"+expected+"\n").encode("ascii")): raise RuntimeError("compute cgroup-v2 membership differs")
 procs_path="/sys/fs/cgroup"+expected+"/cgroup.procs"
 fd=os.open(procs_path,os.O_RDONLY|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0))
 try:
  before=os.fstat(fd);raw=os.read(fd,1048576);after=os.fstat(fd);named=os.lstat(procs_path)
 finally: os.close(fd)
 if len(raw)>=1048576 or ident(before)!=ident(after) or ident(before)!=ident(named): raise RuntimeError("compute cgroup.procs identity differs")
 try: pids=sorted(int(line) for line in raw.decode("ascii","strict").splitlines())
 except (UnicodeError,ValueError) as error: raise RuntimeError("compute cgroup.procs payload differs") from error
 if not pids or len(set(pids))!=len(pids) or os.getpid() not in pids: raise RuntimeError("compute cgroup PID closure differs")
 return {"version":2,"membership_path":expected,"membership_sha256":hashlib.sha256(membership_raw).hexdigest(),"cgroup_procs_path":procs_path,"cgroup_procs_identity":ident(before),"cgroup_procs_anchor":anchor(before),"pids":pids,"self_pid":os.getpid()}
try: release_raw=base64.b64decode(release_b64.encode("ascii"),validate=True)
except Exception as error: raise RuntimeError("release base64 differs") from error
release=parse_json(release_raw,"release",newline=False)
roles=("runner","legacy_exact5_runner","object_eval","legacy_exact5_eval","frozen_runner","bridge","adapter","object_wrapper_inner","legacy_infer_alias","trajectory_projection","trajectory_scaffold_module","base_adapter","eval_v1","eval_v2","model_authority","torchrun_source","torchrun_handler_source","torch_local_agent_source","torch_dynamic_rendezvous_source","torch_multiprocessing_api_source","base_model_manifest","r64_checkpoint_manifest","python","ffmpeg","ffprobe","plan")
tasks=["case01-object-trajectory-null_before-full644","case01-object-trajectory-route_off-full644","case01-object-trajectory-trajectory_bone_only-full644","case01-object-trajectory-trajectory_dog_bone-full644","case01-object-trajectory-null_after-full644"]
arms=["null_before","route_off","trajectory_bone_only","trajectory_dog_bone","null_after"]
fields={"schema_version","entry_mode","external_root_of_trust","bash_privileged_mode","slurm_export_none","named_hold_payload_executed","ready_plan_is_derived_overlay","python_is_executed_from_held_fd","runner_is_compiled_from_captured_fd_bytes","all_exact26_named_identities_replayed_before_runner","expected_allocation_gpu_count","campaign_mode","task_count","selected_task_ids","arm_order","all_arms_attempted_exactly_once_by_runner","retry_allowed","partial_outputs_are_not_results","expected_runtime_versions","expected_gpu_name","runtime_receipt_path","rank_cache_receipt_path","rank_cache_root","runner_attestation_path","torch_package_init_authority","directory_authorities","holder_job_id","expected_node","identities","runner_arguments"}
if (set(release)!=fields or digest(release)!=release_digest or release.get("schema_version")!="case01-object-trajectory-exact5-r64-gpu-release-v4" or release.get("entry_mode")!="trusted_controller_streamed_stdin" or release.get("external_root_of_trust")!="receipt-gated-controller-held-bytes" or release.get("bash_privileged_mode") is not True or release.get("slurm_export_none") is not True or release.get("named_hold_payload_executed") is not False or release.get("ready_plan_is_derived_overlay") is not True or release.get("python_is_executed_from_held_fd") is not True or release.get("runner_is_compiled_from_captured_fd_bytes") is not True or release.get("all_exact26_named_identities_replayed_before_runner") is not True or release.get("expected_allocation_gpu_count")!=8 or release.get("campaign_mode")!="case01-object-trajectory-exact5-r64-engineering-oracle-v3" or release.get("task_count")!=5 or release.get("selected_task_ids")!=tasks or release.get("arm_order")!=arms or release.get("all_arms_attempted_exactly_once_by_runner") is not True or release.get("retry_allowed") is not False or release.get("partial_outputs_are_not_results") is not True or release.get("expected_runtime_versions")!={"hip":"6.3.42131-fa1d09cbd","torch":"2.7.1+rocm6.3"} or release.get("expected_gpu_name")!="AMD Instinct MI210" or release.get("holder_job_id")!=job_id or release.get("expected_node")!=step_nodelist or job_nodelist!=step_nodelist or gpus_on_node!="8" or gpus_per_node!="8" or step_gpus!="0,1,2,3,4,5,6,7" or node_count!="1" or step_node_count!="1" or release.get("rank_cache_root")!="/tmp/bernini-case01-object-trajectory-exact5-r64-job143808-node292-r3-rank-cache"): raise RuntimeError("GPU release differs")
identities=release.get("identities");arguments=release.get("runner_arguments");directory_rows=release.get("directory_authorities")
if type(identities) is not dict or set(identities)!=set(roles) or len(identities)!=26 or len({row.get("path") for row in identities.values() if type(row) is dict})!=26: raise RuntimeError("exact26 production identity closure differs")
if type(arguments) is not list or len(arguments)!=80 or any(type(value) is not str or not value for value in arguments): raise RuntimeError("runner argument vector differs")
argmap=dict(zip(arguments[::2],arguments[1::2]))
if len(argmap)!=40 or argmap.get("--campaign-mode")!=release["campaign_mode"] or argmap.get("--holder-job-id")!=job_id or argmap.get("--expected-node")!=release["expected_node"] or argmap.get("--expected-allocation-gpu-count")!="8" or argmap.get("--plan")!=identities["plan"]["path"] or argmap.get("--plan-sha256")!=identities["plan"]["sha256"] or argmap.get("--runner-sha256")!=identities["runner"]["sha256"] or argmap.get("--rank-cache-root")!=release["rank_cache_root"]: raise RuntimeError("runner argument binding differs")
package_root=os.path.dirname(os.path.dirname(argmap["--output-report"]))
expected_dirs={"output":os.path.join(package_root,"outputs","media_v3"),"final":os.path.join(package_root,"final"),"runtime":os.path.join(package_root,"runtime"),"site_packages":"/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages"}
if argmap["--output-report"]!=os.path.join(expected_dirs["final"],"object_trajectory_exact5_report_v3.json") or release["runner_attestation_path"]!=os.path.join(expected_dirs["final"],"object_trajectory_exact5_runner_attestation_v3.json") or release["runtime_receipt_path"]!=os.path.join(expected_dirs["runtime"],"exact5_gpu_runtime_v4.json") or release["rank_cache_receipt_path"]!=os.path.join(expected_dirs["runtime"],"exact5_gpu_rank_cache_receipt_v4.json"): raise RuntimeError("compute target path closure differs")
directory_fds={};held={};raw={};info={};extra_fds=[]
try:
 if type(directory_rows) is not dict or set(directory_rows)!={"output","final","runtime","site_packages"}: raise RuntimeError("directory authority set differs")
 for label in ("output","final","runtime","site_packages"):
  directory_fds[label]=open_bound_directory(label,directory_rows[label],expected_dirs[label])
 expected_owner=(anchor_row(directory_rows["output"])[2],anchor_row(directory_rows["output"])[3])
 if any((anchor_row(row)[2],anchor_row(row)[3])!=expected_owner for row in directory_rows.values()): raise RuntimeError("directory owner closure differs")
 for role in roles:
  held[role],raw[role],info[role]=replay(role,identities[role],role in {"python","ffmpeg","ffprobe"})
 if ident(python_before)!=ident(info["python"]) or ident(python_before)!=ident(os.stat("/proc/self/exe")) or hashlib.sha256(read_fd(python_fd,python_before.st_size)).hexdigest()!=identities["python"]["sha256"]: raise RuntimeError("held process Python differs")
 plan=parse_json(raw["plan"],"READY exact-five plan");unsigned=dict(plan);claimed=unsigned.pop("plan_digest",None);plan_rows=plan.get("tasks")
 if claimed!=digest(unsigned) or plan.get("schema_version")!="case01-object-trajectory-exact5-plan-v3" or plan.get("status")!="READY_FOR_EXPLICIT_LOCAL_LAUNCH" or plan.get("production_ready") is not False or plan.get("launch_allowed") is not True or plan.get("hold_reasons")!=[] or type(plan_rows) is not list or len(plan_rows)!=5 or [row.get("task_id") for row in plan_rows]!=tasks or [row.get("oracle_arm") for row in plan_rows]!=arms or any(row.get("source_onset_policy")!="hard1_every_step" for row in plan_rows): raise RuntimeError("READY exact-five plan differs")
 checkpoint=plan.get("checkpoint_manifest")
 if type(checkpoint) is not dict or checkpoint.get("path")!=identities["r64_checkpoint_manifest"]["path"] or checkpoint.get("sha256")!=identities["r64_checkpoint_manifest"]["sha256"] or checkpoint.get("pin_complete") is not True: raise RuntimeError("R64 authority closure differs")
 site=directory_rows["site_packages"]["path"]
 torch_paths={"torchrun_source":"torch/distributed/run.py","torchrun_handler_source":"torch/distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py","torch_local_agent_source":"torch/distributed/elastic/agent/server/local_elastic_agent.py","torch_dynamic_rendezvous_source":"torch/distributed/elastic/rendezvous/dynamic_rendezvous.py","torch_multiprocessing_api_source":"torch/distributed/elastic/multiprocessing/api.py"}
 if any(identities[role]["path"]!=site+"/"+relative for role,relative in torch_paths.items()) or site in sys.path or any(name=="torch" or name.startswith("torch.") for name in sys.modules): raise RuntimeError("isolated Torch site binding differs")
 torch_init=site+"/torch/__init__.py";torch_init_row=release["torch_package_init_authority"]
 torch_init_fd,torch_init_raw,torch_init_info=replay("torch-package-init",torch_init_row);extra_fds.append(torch_init_fd)
 if torch_init_row["path"]!=torch_init or sys.flags.isolated!=1 or sys.flags.no_site!=1 or not sys.dont_write_bytecode: raise RuntimeError("Torch package entry/isolation differs")
 sys.path.insert(0,site);os.environ["FULL644_MATCHED_SITE_PACKAGES_ROOT"]=site
 torch_loader=importlib.machinery.SourceFileLoader("torch",torch_init)
 torch_spec=importlib.util.spec_from_file_location("torch",torch_init,loader=torch_loader,submodule_search_locations=[site+"/torch"])
 if torch_spec is None or torch_spec.loader is not torch_loader or torch_spec.origin!=torch_init: raise RuntimeError("held Torch source spec differs")
 torch=importlib.util.module_from_spec(torch_spec);sys.modules["torch"]=torch
 try: exec(compile(torch_init_raw,torch_init,"exec",dont_inherit=True),torch.__dict__)
 except BaseException:
  sys.modules.pop("torch",None);raise
 if os.path.realpath(torch.__file__)!=torch_init or torch.__spec__.origin!=torch_init or torch.__loader__ is not torch_loader: raise RuntimeError("Torch loaded outside held source authority")
 if read_fd(torch_init_fd,torch_init_info.st_size,16777216)!=torch_init_raw or ident(os.fstat(torch_init_fd))!=ident(torch_init_info) or ident(os.lstat(torch_init))!=ident(torch_init_info): raise RuntimeError("held Torch source changed after import")
 names=[torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
 if torch.__version__!="2.7.1+rocm6.3" or torch.version.hip!="6.3.42131-fa1d09cbd" or not torch.cuda.is_available() or torch.cuda.device_count()!=8 or names!=["AMD Instinct MI210"]*8: raise RuntimeError("Torch/HIP/GPU runtime differs")
 for label in directory_fds: replay_directory(label,directory_fds[label],directory_rows[label])
 runtime={"schema_version":"case01-object-trajectory-exact5-r64-gpu-controller-v4-runtime","status":"PASS_GPU_RUNTIME_BEFORE_RUNNER","holder_job_id":job_id,"node":step_nodelist,"slurm_step_id":step_id,"production_release_digest":release_digest,"ready_plan_digest":plan["plan_digest"],"torch_version":torch.__version__,"hip_version":torch.version.hip,"device_count":torch.cuda.device_count(),"device_names":names,"r64_checkpoint_manifest_sha256":identities["r64_checkpoint_manifest"]["sha256"],"exact26_identity_set_digest":digest(identities),"site_packages_root":site,"site_packages_identity":directory_rows["site_packages"]["identity"],"torch_package_init_authority":torch_init_row,"torch_module_path":torch_init,"torch_module_sha256":hashlib.sha256(torch_init_raw).hexdigest(),"torch_package_entry_compiled_from_held_source":True,"torch_source_loader":"SourceFileLoader","isolated_flag":sys.flags.isolated,"no_site_flag":sys.flags.no_site,"dont_write_bytecode":sys.dont_write_bytecode,"site_packages_activated_explicitly_under_isolated_no_site":True,"renderer_started":False}
 runtime_fd,runtime_raw=publish_json(release["runtime_receipt_path"],runtime,"receipt_digest",directory_fds["runtime"]);extra_fds.append(runtime_fd)
 cgroup_before=cgroup_snapshot(job_id,step_id)
 if cgroup_before["pids"]!=[os.getpid()]: raise RuntimeError("compute baseline cgroup is not single-process")
 runner_fd=held["runner"];os.set_inheritable(runner_fd,False)
 entry={"schema_version":"full644-exploratory-matched-captured-runner-entry-authority-v1","runner_fd":runner_fd,"runner_path":identities["runner"]["path"],"runner_sha256":identities["runner"]["sha256"],"runner_identity":{key:value for key,value in zip(("device","inode","uid","gid","mode","nlink","rdev","size","blocks","mtime_ns","ctime_ns"),ident(info["runner"]))},"python_fd":python_fd,"python_path":identities["python"]["path"],"python_sha256":identities["python"]["sha256"],"python_identity":{key:value for key,value in zip(("device","inode","uid","gid","mode","nlink","rdev","size","blocks","mtime_ns","ctime_ns"),ident(python_before))},"release_digest":release_digest,"bootstrap_sha256":bootstrap_sha,"entry_method":"slurm-spooled-or-trusted-stdin-held-python-fd-v1","slurm_export_none_required":True,"bash_privileged_startup_required":True,"captured_source_entry":True}
 entry["authority_digest"]=digest(entry)
 os.environ.clear();os.environ.update({"FULL644_MATCHED_CAPTURED_RUNNER_ENTRY_AUTHORITY":canonical(entry).decode("utf-8"),"SLURM_JOB_ID":job_id,"SLURM_STEP_ID":step_id,"SLURM_GPUS_ON_NODE":gpus_on_node,"SLURM_GPUS_PER_NODE":gpus_per_node,"SLURM_STEP_GPUS":step_gpus,"SLURM_NNODES":node_count,"SLURM_STEP_NUM_NODES":step_node_count,"SLURM_JOB_NODELIST":job_nodelist,"SLURM_STEP_NODELIST":step_nodelist})
 runner_source=raw["runner"].decode("utf-8","strict");runner_path=identities["runner"]["path"];sys.argv=[runner_path,*arguments]
 module=types.ModuleType("__main__");module.__file__=runner_path;module.__package__=None;module.__loader__=None;module.__spec__=None;module.__cached__=None;module.__builtins__=__builtins__;sys.modules["__main__"]=module
 capture=io.StringIO();old_stdout=sys.stdout;exit_zero=False
 try:
  sys.stdout=capture
  try: exec(compile(runner_source,runner_path,"exec",dont_inherit=True),module.__dict__)
  except SystemExit as exited:
   code=0 if exited.code is None else exited.code
   if code!=0: raise RuntimeError("exact-five runner exited nonzero")
   exit_zero=True
 finally: sys.stdout=old_stdout
 if not exit_zero: raise RuntimeError("exact-five runner did not own terminal exit")
 runner_stdout=capture.getvalue().encode("utf-8","strict")
 if runner_stdout.count(b"\n")!=1: raise RuntimeError("exact-five runner stdout cardinality differs")
 stdout_attestation=parse_json(runner_stdout,"captured runner stdout")
 attestation_fd,attestation_raw,attestation=capture_json(release["runner_attestation_path"],directory_fds["final"],0o444,"runner attestation");extra_fds.append(attestation_fd)
 task_results=attestation.get("task_results")
 if attestation!=stdout_attestation or attestation.get("schema_version")!="case01-object-trajectory-exact5-runner-attestation-v3" or attestation.get("status")!="EXACT5_COMPLETE_AWAITING_BLIND_REVIEW" or attestation.get("campaign_mode")!=release["campaign_mode"] or attestation.get("task_count")!=5 or attestation.get("task_ids")!=tasks or attestation.get("all_exact5_tasks_attempted_exactly_once") is not True or attestation.get("all_exact5_tasks_succeeded") is not True or attestation.get("retry_count")!=0 or type(task_results) is not list or len(task_results)!=5 or [(row.get("task_index"),row.get("task_id"),row.get("return_code")) for row in task_results]!=[(index,task,0) for index,task in enumerate(tasks)] or attestation.get("task_environment_digests")!=[row.get("environment_digest") for row in task_results]: raise RuntimeError("compute runner attestation differs")
 artifact_suffixes={"model_capture":"-model-capture.json","model_pre_use":"-model-pre-use.json","consumption_input":"-consumption-input.json","model_post_use":"-model-post-use.json","eval_consumption_chain":"-eval-consumption-chain.json","adapter_capture":"-adapter-capture.json","adapter_pre_use":"-adapter-pre-use.json","adapter_post_use":"-adapter-post-use.json","adapter_final":"-adapter-final.json"}
 artifact_replays=attestation.get("task_artifact_replays")
 if type(artifact_replays) is not list or len(artifact_replays)!=5 or [row.get("task_id") for row in artifact_replays]!=tasks or attestation.get("task_result_digests")!=[row.get("task_result_digest") for row in task_results]: raise RuntimeError("runner artifact replay closure differs")
 internal_inventory=[];internal_held=[]
 for index,task in enumerate(tasks):
  result=task_results[index];unsigned_result=dict(result);task_result_digest=unsigned_result.pop("task_result_digest",None);refs=result.get("authority_artifacts");replayed=artifact_replays[index];prefix=".matched-v2-%02d-%s"%(index,task)
  if task_result_digest!=digest(unsigned_result) or type(refs) is not dict or set(refs)!=set(artifact_suffixes) or result.get("attempt_count")!=1 or result.get("retry_allowed") is not False or result.get("log_basename")!=prefix+".log" or replayed.get("task_result_digest")!=task_result_digest or replayed.get("artifact_count")!=9 or not sha_ok(replayed.get("artifact_rows_digest")): raise RuntimeError("task internal artifact authority differs")
  for role,suffix in artifact_suffixes.items():
   reference=refs[role];basename=prefix+suffix
   if type(reference) is not dict or set(reference)!={"basename","sha256"} or reference.get("basename")!=basename or not sha_ok(reference.get("sha256")): raise RuntimeError("task artifact reference differs")
   artifact_fd,artifact_raw,artifact_info=capture_leaf_at(directory_fds["output"],directory_rows["output"]["path"],basename,67108864,"task artifact");extra_fds.append(artifact_fd);internal_held.append((artifact_fd,artifact_raw,ident(artifact_info),basename))
   if hashlib.sha256(artifact_raw).hexdigest()!=reference["sha256"]: raise RuntimeError("task artifact hash differs")
   parse_json(artifact_raw,"task artifact")
   internal_inventory.append({"task_index":index,"task_id":task,"role":role,"basename":basename,"sha256":reference["sha256"],"size":len(artifact_raw),"identity":ident(artifact_info)})
  runner_task_basename=prefix+"-runner-task.json";runner_task_fd,runner_task_raw,runner_task_info=capture_leaf_at(directory_fds["output"],directory_rows["output"]["path"],runner_task_basename,67108864,"runner-task artifact");extra_fds.append(runner_task_fd);internal_held.append((runner_task_fd,runner_task_raw,ident(runner_task_info),runner_task_basename))
  if hashlib.sha256(runner_task_raw).hexdigest()!=replayed.get("runner_task_file_sha256") or parse_json(runner_task_raw,"runner-task artifact")!=result: raise RuntimeError("runner-task artifact differs")
  internal_inventory.append({"task_index":index,"task_id":task,"role":"runner_task","basename":runner_task_basename,"sha256":hashlib.sha256(runner_task_raw).hexdigest(),"size":len(runner_task_raw),"identity":ident(runner_task_info)})
  log_basename=prefix+".log";log_fd,log_raw,log_info=capture_leaf_at(directory_fds["output"],directory_rows["output"]["path"],log_basename,1073741824,"task log artifact");extra_fds.append(log_fd);internal_held.append((log_fd,log_raw,ident(log_info),log_basename))
  internal_inventory.append({"task_index":index,"task_id":task,"role":"task_log","basename":log_basename,"sha256":hashlib.sha256(log_raw).hexdigest(),"size":len(log_raw),"identity":ident(log_info)})
 internal_inventory=sorted(internal_inventory,key=lambda row:row["basename"])
 if len(internal_inventory)!=55 or len({row["basename"] for row in internal_inventory})!=55: raise RuntimeError("exact55 internal artifact closure differs")
 cache_root=release["rank_cache_root"]
 root_info=os.lstat(cache_root)
 if not stat.S_ISDIR(root_info.st_mode) or stat.S_IMODE(root_info.st_mode)!=0o700 or (root_info.st_uid,root_info.st_gid)!=expected_owner or os.path.realpath(cache_root)!=cache_root: raise RuntimeError("compute rank-cache root differs")
 if socket.gethostname()!=step_nodelist or os.uname().nodename!=step_nodelist: raise RuntimeError("compute hostname binding differs")
 tmp_info=os.lstat("/tmp");package_info=os.lstat(package_root);cache_mount=mount_authority(cache_root)
 if not stat.S_ISDIR(tmp_info.st_mode) or not stat.S_ISDIR(package_info.st_mode) or tmp_info.st_dev!=64768 or tmp_info.st_ino!=28311553 or tmp_info.st_uid!=0 or tmp_info.st_gid!=0 or stat.S_IMODE(tmp_info.st_mode)!=0o1777 or root_info.st_dev!=tmp_info.st_dev or package_info.st_dev!=48 or cache_mount["major_minor"]!="253:0" or cache_mount["major_minor"]!="%d:%d"%(os.major(root_info.st_dev),os.minor(root_info.st_dev)): raise RuntimeError("compute-local rank-cache device differs")
 task_topology=[]
 expected_task_names=["task-%02d-%s"%(index,task) for index,task in enumerate(tasks)]
 if sorted(os.listdir(cache_root))!=sorted(expected_task_names): raise RuntimeError("compute rank-cache task closure differs")
 rank_names=["rank-0","rank-1","rank-2","rank-3"]
 rank_children=["extensions","hf","home","inductor","miopen-custom","miopen-user","pycache","tmp","torch","triton","xdg"]
 coordinator_children=["hf","home","pycache","tmp","torch","xdg"]
 for index,task in enumerate(tasks):
  task_path=os.path.join(cache_root,expected_task_names[index]);task_info=os.lstat(task_path)
  if not stat.S_ISDIR(task_info.st_mode) or stat.S_IMODE(task_info.st_mode)!=0o700 or (task_info.st_uid,task_info.st_gid)!=expected_owner or sorted(os.listdir(task_path))!=sorted(["coordinator"]+rank_names): raise RuntimeError("compute task-cache topology differs")
  coordinator=os.path.join(task_path,"coordinator")
  if sorted(os.listdir(coordinator))!=coordinator_children: raise RuntimeError("compute coordinator-cache topology differs")
  if os.listdir(os.path.join(coordinator,"pycache"))!=[]: raise RuntimeError("compute coordinator pycache differs")
  for rank_name in rank_names:
   rank_path=os.path.join(task_path,rank_name)
   if sorted(os.listdir(rank_path))!=rank_children or os.listdir(os.path.join(rank_path,"pycache"))!=[]: raise RuntimeError("compute rank-cache topology differs")
  task_topology.append({"task_index":index,"task_id":task,"task_directory":expected_task_names[index],"coordinator_directory":"coordinator","rank_directories":rank_names,"environment_digest":task_results[index]["environment_digest"]})
 inventory,total_cache_bytes=cache_inventory(cache_root,*expected_owner)
 if [row["path"] for row in inventory]!=sorted(row["path"] for row in inventory) or len({row["path"] for row in inventory})!=len(inventory): raise RuntimeError("canonical rank-cache inventory differs")
 residual_scan=residual_rank_pids(cache_root,step_id)
 if residual_scan["matched_pids"] or residual_scan["unreadable_pids"]: raise RuntimeError("compute rank processes remain")
 cgroup_after=cgroup_snapshot(job_id,step_id)
 if cgroup_after["membership_path"]!=cgroup_before["membership_path"] or cgroup_after["membership_sha256"]!=cgroup_before["membership_sha256"] or cgroup_after["cgroup_procs_path"]!=cgroup_before["cgroup_procs_path"] or cgroup_after["cgroup_procs_anchor"]!=cgroup_before["cgroup_procs_anchor"] or cgroup_after["cgroup_procs_identity"]!=cgroup_before["cgroup_procs_identity"] or cgroup_after["pids"]!=[os.getpid()]: raise RuntimeError("compute cgroup did not return to baseline")
 authority_path=argmap["--authority-root"]
 authority_fd=os.open(authority_path,os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_CLOEXEC",0));extra_fds.append(authority_fd)
 authority_identity=ident(os.fstat(authority_fd));replay_empty_directory("model authority root",authority_fd,authority_path,authority_identity,expected_owner)
 inventory_replay,total_cache_bytes_replay=cache_inventory(cache_root,*expected_owner)
 if inventory_replay!=inventory or total_cache_bytes_replay!=total_cache_bytes: raise RuntimeError("terminal rank-cache inventory replay differs")
 for artifact_fd,artifact_raw,artifact_identity,artifact_basename in internal_held:
  if ident(os.fstat(artifact_fd))!=artifact_identity or read_fd(artifact_fd,len(artifact_raw),1073741824)!=artifact_raw or ident(os.stat(artifact_basename,dir_fd=directory_fds["output"],follow_symlinks=False))!=artifact_identity: raise RuntimeError("terminal internal artifact replay differs")
 root_after=os.lstat(cache_root)
 if ident(root_after)!=ident(root_info) or read_fd(torch_init_fd,torch_init_info.st_size,16777216)!=torch_init_raw or ident(os.fstat(torch_init_fd))!=ident(torch_init_info) or ident(os.lstat(torch_init))!=ident(torch_init_info): raise RuntimeError("terminal compute authority replay differs")
 replay_empty_directory("model authority root",authority_fd,authority_path,authority_identity,expected_owner)
 for label in directory_fds: replay_directory(label,directory_fds[label],directory_rows[label])
 cache_receipt={"schema_version":"case01-object-trajectory-exact5-r64-gpu-controller-v4-rank-cache","status":"PASS_RETAINED_COMPUTE_LOCAL_RANK_CACHE_AUTHORITY","holder_job_id":job_id,"node":step_nodelist,"hostname":socket.gethostname(),"uname_nodename":os.uname().nodename,"slurm_step_id":step_id,"production_release_digest":release_digest,"runtime_receipt_sha256":hashlib.sha256(runtime_raw).hexdigest(),"runtime_receipt_digest":runtime["receipt_digest"],"runner_attestation_sha256":hashlib.sha256(attestation_raw).hexdigest(),"runner_attestation_digest":attestation["attestation_digest"],"rank_cache_root":cache_root,"rank_cache_root_identity":ident(root_info),"rank_cache_parent":"/tmp","rank_cache_parent_identity":ident(tmp_info),"package_root":package_root,"package_root_identity":ident(package_info),"cache_device_matches_tmp":True,"cache_device_differs_from_package":True,"cache_mount":cache_mount,"compute_node_observation":True,"retained_compute_local":True,"fresh_unique_attempt_path":True,"freshness_enforced_by_frozen_runner_before_mkdir":True,"cleanup_performed":False,"absent_claimed":False,"non_scientific_cache":True,"cache_is_not_output_or_result":True,"task_count":5,"rank_process_count":20,"coordinator_process_count":5,"task_topology":task_topology,"inventory_bound":{"maximum_depth":32,"maximum_entries":50000,"maximum_file_bytes":4294967296,"maximum_single_file_bytes":1073741824},"inventory":inventory,"inventory_entry_count":len(inventory),"inventory_total_file_bytes":total_cache_bytes,"inventory_digest":digest(inventory),"terminal_inventory_replayed_exactly":True,"internal_artifact_inventory":internal_inventory,"internal_artifact_count":55,"internal_artifact_inventory_digest":digest(internal_inventory),"internal_artifact_fds_held_and_terminal_replayed":True,"model_authority_root":authority_path,"model_authority_root_identity":authority_identity,"model_authority_root_empty":True,"model_authority_root_held_and_terminal_replayed":True,"rank_processes_zero":True,"torchrun_processes_zero":True,"cgroup_baseline":cgroup_before,"cgroup_terminal":cgroup_after,"cgroup_returned_exactly_to_single_root_process":True,"process_scan_performed":True,"process_scan_sources":["/proc/*/environ","/proc/*/cmdline"],"process_scan":residual_scan,"matched_residual_pids":[]}
 cache_fd,cache_raw=publish_json(release["rank_cache_receipt_path"],cache_receipt,"receipt_digest",directory_fds["runtime"]);extra_fds.append(cache_fd)
 replay_empty_directory("model authority root",authority_fd,authority_path,authority_identity,expected_owner)
 compute={"schema_version":"case01-object-trajectory-exact5-r64-gpu-controller-v4-compute-result","status":"PASS_EXACT_FIVE_COMPUTE_POSTFLIGHT","holder_job_id":job_id,"node":step_nodelist,"slurm_step_id":step_id,"runner_returncode":0,"production_release_digest":release_digest,"ready_plan_digest":plan["plan_digest"],"runtime_receipt_sha256":hashlib.sha256(runtime_raw).hexdigest(),"runtime_receipt_digest":runtime["receipt_digest"],"rank_cache_receipt_sha256":hashlib.sha256(cache_raw).hexdigest(),"rank_cache_receipt_digest":cache_receipt["receipt_digest"],"runner_attestation_sha256":hashlib.sha256(attestation_raw).hexdigest(),"runner_attestation_digest":attestation["attestation_digest"],"all_five_arms_attempted_exactly_once":True,"retry_count":0,"rank_processes_zero":True,"rank_process_scan_performed":True,"rank_cache_compute_state":"RETAINED_COMPUTE_LOCAL_POSTFLIGHT_AUTHORITY","rank_cache_inventory_digest":cache_receipt["inventory_digest"],"internal_artifact_count":55,"internal_artifact_inventory_digest":cache_receipt["internal_artifact_inventory_digest"],"model_authority_root_identity":authority_identity,"model_authority_root_empty":True,"model_authority_root_held_and_terminal_replayed":True,"local_srun_process_group_observed":False}
 compute["result_digest"]=digest(compute);sys.stdout.write(canonical(compute).decode("utf-8")+"\n");sys.stdout.flush()
finally:
 for fd in extra_fds:
  try: os.close(fd)
  except OSError: pass
 for role,fd in held.items():
  if role!="runner":
   try: os.close(fd)
   except OSError: pass
 for fd in directory_fds.values():
  try: os.close(fd)
  except OSError: pass
'''


def _shell_quote(value: str) -> str:
    if "\x00" in value:
        raise GPUControllerError("shell literal contains NUL")
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_production_release(
    launch_input: Mapping[str, Any], authorities: Mapping[str, HeldAuthority],
    ready_plan_authority: HeldAuthority,
    site_packages_authority: HeldDirectory,
    torch_package_init_authority: HeldAuthority,
    target_directories: Mapping[str, HeldDirectory],
) -> tuple[dict[str, Any], bytes]:
    rows = {
        role: (
            ready_plan_authority.row() if role == "plan" else authorities[role].row()
        )
        for role in IDENTITY_ROLES
    }
    arguments = build_runner_arguments(launch_input, rows)
    expected_directory_paths = {
        "output": OUTPUT_ROOT,
        "final": PACKAGE_ROOT / "final",
        "runtime": PACKAGE_ROOT / "runtime",
        "site_packages": SITE_PACKAGES_ROOT,
    }
    supplied_directories = {
        "output": target_directories.get("output"),
        "final": target_directories.get("final"),
        "runtime": target_directories.get("runtime"),
        "site_packages": site_packages_authority,
    }
    if any(
        directory is None or directory.path != expected_directory_paths[label]
        or len(directory.held_identity) != 11
        for label, directory in supplied_directories.items()
    ):
        raise GPUControllerError("production directory authority differs")
    directory_authorities = {
        label: {
            "path": str(expected_directory_paths[label]),
            "identity": list(supplied_directories[label].held_identity),
        }
        for label in ("output", "final", "runtime", "site_packages")
    }
    torch_init_row = torch_package_init_authority.row()
    if (
        torch_package_init_authority.path != TORCH_PACKAGE_INIT_PATH
        or torch_init_row.get("path") != str(TORCH_PACKAGE_INIT_PATH)
        or torch_init_row.get("mode") != 0o644
        or torch_init_row.get("nlink") != 1
        or SHA_RE.fullmatch(str(torch_init_row.get("sha256"))) is None
        or type(torch_init_row.get("size")) is not int
        or not (0 < torch_init_row["size"] <= MAX_SOURCE_SIZE)
        or type(torch_init_row.get("identity")) is not list
        or len(torch_init_row["identity"]) != 11
    ):
        raise GPUControllerError("held Torch package init authority differs")
    release: dict[str, Any] = {
        "schema_version": "case01-object-trajectory-exact5-r64-gpu-release-v4",
        "entry_mode": "trusted_controller_streamed_stdin",
        "external_root_of_trust": "receipt-gated-controller-held-bytes",
        "bash_privileged_mode": True, "slurm_export_none": True,
        "named_hold_payload_executed": False,
        "ready_plan_is_derived_overlay": True,
        "python_is_executed_from_held_fd": True,
        "runner_is_compiled_from_captured_fd_bytes": True,
        "all_exact26_named_identities_replayed_before_runner": True,
        "expected_allocation_gpu_count": GPU_COUNT,
        "campaign_mode": CAMPAIGN, "task_count": 5,
        "selected_task_ids": list(TASK_IDS), "arm_order": list(ARM_ORDER),
        "all_arms_attempted_exactly_once_by_runner": True,
        "retry_allowed": False, "partial_outputs_are_not_results": True,
        "expected_runtime_versions": {
            "torch": EXPECTED_TORCH_VERSION, "hip": EXPECTED_HIP_VERSION,
        },
        "expected_gpu_name": EXPECTED_GPU_NAME,
        "runtime_receipt_path": str(RUNTIME_RECEIPT_PATH),
        "rank_cache_receipt_path": str(RANK_CACHE_RECEIPT_PATH),
        "rank_cache_root": str(RANK_CACHE_ROOT),
        "runner_attestation_path": str(RUNNER_ATTESTATION_PATH),
        "torch_package_init_authority": torch_init_row,
        "directory_authorities": directory_authorities,
        "holder_job_id": HOLDER_JOB_ID, "expected_node": NODE,
        "identities": rows, "runner_arguments": arguments,
    }
    release_digest = object_digest(release)
    bootstrap_sha = hashlib.sha256(ROOT_BOOTSTRAP.encode("utf-8")).hexdigest()
    release_b64 = base64.b64encode(canonical(release)).decode("ascii")
    script = f'''#!/bin/bash -p
set -Eeuo pipefail
umask 077
[[ "$-" == *p* ]] || exit 91
[[ "$0" == /bin/bash || "$0" == bash || "$0" == -bash ]] || exit 92
[[ "${{SLURM_JOB_ID-}}" == {HOLDER_JOB_ID} ]] || exit 93
[[ "${{SLURM_STEP_ID-}}" =~ ^[1-9][0-9]*$ ]] || exit 94
[[ "${{SLURM_GPUS_ON_NODE-}}" == 8 && "${{SLURM_GPUS_PER_NODE-}}" == 8 && "${{SLURM_STEP_GPUS-}}" == 0,1,2,3,4,5,6,7 ]] || exit 95
[[ "${{SLURM_NNODES-}}" == 1 && "${{SLURM_STEP_NUM_NODES-}}" == 1 && "${{SLURM_JOB_NODELIST-}}" == {NODE} && "${{SLURM_STEP_NODELIST-}}" == {NODE} ]] || exit 96
[[ -z "${{SLURM_JOB_GPUS+x}}" && -z "${{SLURM_JOB_NUM_NODES+x}}" ]] || exit 97
if shopt -q varredir_close 2>/dev/null; then shopt -u varredir_close; fi
readonly PINNED_PYTHON={_shell_quote(rows['python']['path'])}
exec {{PINNED_PYTHON_FD}}<"$PINNED_PYTHON"
[[ "$PINNED_PYTHON_FD" =~ ^[0-9]+$ && -r "/proc/self/fd/$PINNED_PYTHON_FD" ]] || exit 98
exec -c "/proc/self/fd/$PINNED_PYTHON_FD" -I -S -B -c {_shell_quote(ROOT_BOOTSTRAP)} "$PINNED_PYTHON_FD" {_shell_quote(release_b64)} {_shell_quote(release_digest)} {_shell_quote(bootstrap_sha)} "$1" "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$SLURM_GPUS_ON_NODE" "$SLURM_GPUS_PER_NODE" "$SLURM_STEP_GPUS" "$SLURM_NNODES" "$SLURM_STEP_NUM_NODES" "$SLURM_JOB_NODELIST" "$SLURM_STEP_NODELIST"
'''.encode("utf-8")
    return release, script


def build_srun_argv(max_gate_step: int) -> list[str]:
    if type(max_gate_step) is not int or max_gate_step < 0:
        raise GPUControllerError("composite CPU step floor differs")
    return [
        SRUN_AUTHORITY["path"], f"--jobid={HOLDER_JOB_ID}",
        "--job-name=case01-object-trajectory-exact5-r64-gpu-v4",
        "--exclusive", "--exact", "--immediate=10", "--kill-on-bad-exit=1",
        "--nodes=1", "--ntasks=1", f"--nodelist={NODE}",
        f"--cpus-per-task={CPUS_PER_TASK}", f"--mem={MEMORY}",
        f"--gpus-per-node={GPU_COUNT}", "--export=NONE", "--time=03:00:00",
        "/bin/bash", "-p", "-s", "--", str(max_gate_step),
    ]


def srun_environment() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin", "HOME": "/vast/users/guangyi.chen",
        "LANG": "C", "LC_ALL": "C", "BASH_ENV": "/dev/null",
    }


def _ascii_vector_width(values: Sequence[str], *, label: str) -> int:
    if type(values) not in {list, tuple} or not values:
        raise GPUControllerError(f"{label} vector differs")
    width = 0
    for value in values:
        if type(value) is not str or not value or "\x00" in value:
            raise GPUControllerError(f"{label} element differs")
        try:
            width += len(value.encode("ascii", "strict")) + 1
        except UnicodeError as error:
            raise GPUControllerError(f"{label} is not ASCII") from error
    return width


def _nested_python_argv_upper_bound(
    release: Mapping[str, Any], max_gate_step: int,
) -> int:
    # Bash starts in privileged mode and uses ``exec -c`` (empty environment).
    # The FD and Slurm step are runtime values; twenty decimal characters is a
    # conservative signed-64-bit-width reservation for each.
    if type(max_gate_step) is not int or max_gate_step < 0:
        raise GPUControllerError("nested composite CPU step floor differs")
    release_raw = canonical(release)
    release_b64 = base64.b64encode(release_raw).decode("ascii")
    values = [
        "/proc/self/fd/" + "9" * 20,
        "-I", "-S", "-B", "-c", ROOT_BOOTSTRAP, "9" * 20,
        release_b64, object_digest(release),
        hashlib.sha256(ROOT_BOOTSTRAP.encode("utf-8")).hexdigest(),
        "9" * max(20, len(str(max_gate_step))),
        HOLDER_JOB_ID, "9" * 20, "8", "8",
        "0,1,2,3,4,5,6,7", "1", "1", NODE, NODE,
    ]
    return _ascii_vector_width(values, label="nested Python argv upper bound")


def validate_srun_transport(
    command: Sequence[str], *, max_gate_step: int,
    payload: bytes | None = None,
    release: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected = build_srun_argv(max_gate_step)
    if list(command) != expected:
        raise GPUControllerError("exact fixed GPU srun argv differs")
    if not (
        0 < MAX_EXACT_SRUN_ARGV_BYTES < OBSERVED_AUH_ARG_MAX
        and 0 < MAX_HELD_STDIN_BYTES < OBSERVED_AUH_ARG_MAX
        and 0 < MAX_NESTED_PYTHON_ARGV_BYTES < OBSERVED_AUH_ARG_MAX
        and 0 < MIN_EXECVE_HEADROOM_BYTES < OBSERVED_AUH_ARG_MAX
    ):
        raise GPUControllerError("AUH transport bounds differ")
    argv_width = _ascii_vector_width(expected, label="exact srun argv")
    joined_width = len(" ".join(expected).encode("ascii", "strict"))
    environment = srun_environment()
    environment_width = _ascii_vector_width(
        [f"{key}={value}" for key, value in sorted(environment.items())],
        label="srun environment",
    )
    execve_width = argv_width + environment_width
    headroom = OBSERVED_AUH_ARG_MAX - execve_width
    if (
        argv_width >= MAX_EXACT_SRUN_ARGV_BYTES
        or joined_width >= MAX_EXACT_SRUN_ARGV_BYTES
        or headroom < MIN_EXECVE_HEADROOM_BYTES
    ):
        raise GPUControllerError("exact GPU srun argv exceeds reviewed bound")
    result: dict[str, Any] = {
        "transport": "fixed_srun_argv_plus_held_stdin_pipe",
        "observed_auh_arg_max": OBSERVED_AUH_ARG_MAX,
        "max_exact_srun_argv_bytes": MAX_EXACT_SRUN_ARGV_BYTES,
        "exact_srun_joined_width": joined_width,
        "exact_srun_execve_argv_width": argv_width,
        "exact_srun_environment_width": environment_width,
        "exact_srun_execve_total_width": execve_width,
        "exact_srun_execve_headroom": headroom,
        "composite_cpu_admission_step_floor": max_gate_step,
        "max_held_stdin_bytes": MAX_HELD_STDIN_BYTES,
        "max_nested_python_argv_bytes": MAX_NESTED_PYTHON_ARGV_BYTES,
        "held_stdin_sha256": None, "held_stdin_size": None,
        "nested_python_argv_upper_bound": None,
    }
    if payload is None:
        if release is not None:
            raise GPUControllerError("release supplied without held stdin")
        return result
    if type(payload) is not bytes or not (0 < len(payload) < MAX_HELD_STDIN_BYTES):
        raise GPUControllerError("held GPU stdin exceeds reviewed bound")
    if type(release) is not dict:
        raise GPUControllerError("held GPU stdin release is absent")
    nested_width = _nested_python_argv_upper_bound(release, max_gate_step)
    if (
        nested_width >= MAX_NESTED_PYTHON_ARGV_BYTES
        or OBSERVED_AUH_ARG_MAX - nested_width < MIN_EXECVE_HEADROOM_BYTES
    ):
        raise GPUControllerError("nested Python argv exceeds reviewed bound")
    result.update({
        "held_stdin_sha256": hashlib.sha256(payload).hexdigest(),
        "held_stdin_size": len(payload),
        "nested_python_argv_upper_bound": nested_width,
    })
    return result


def create_immutable(
    directory: HeldDirectory, path: Path, raw: bytes, mode: int,
) -> bytes:
    if path.parent != directory.path or os.path.lexists(path):
        raise GPUControllerError(f"create-only target differs: {path}")
    directory.replay()
    descriptor = os.open(
        path.name, os.O_RDWR | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0, dir_fd=directory.descriptor,
    )
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise GPUControllerError("create-only write made no progress")
            offset += count
        os.fsync(descriptor)
        staged = os.fstat(descriptor)
        named = os.stat(
            path.name, dir_fd=directory.descriptor, follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(staged.st_mode) or stat.S_IMODE(staged.st_mode) != 0
            or staged.st_nlink != 1 or identity(staged) != identity(named)
            or read_fd(descriptor, len(raw)) != raw
        ):
            raise GPUControllerError("create-only staging replay differs")
        os.fchmod(descriptor, mode); os.fsync(descriptor)
        committed = os.fstat(descriptor)
        if (
            stat.S_IMODE(committed.st_mode) != mode
            or identity(committed) != identity(os.stat(
                path.name, dir_fd=directory.descriptor, follow_symlinks=False,
            ))
            or read_fd(descriptor, len(raw)) != raw
        ):
            raise GPUControllerError("create-only commit replay differs")
        # The attempt/dispatch claims must survive a controller or allocation
        # crash before srun.  Sync both the committed inode and its directory
        # entry before returning to the caller.
        os.fsync(directory.descriptor)
    finally:
        os.close(descriptor)
    directory.replay(); return raw


def create_immutable_json(
    directory: HeldDirectory, path: Path, value: Mapping[str, Any], mode: int,
) -> bytes:
    return create_immutable(directory, path, canonical(value) + b"\n", mode)


def _process_group_absent(process_group: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if not _process_group_absent(process_group, TERMINATE_GRACE_SECONDS):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        try:
            process.wait(timeout=TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired as error:
            raise GPUControllerError("srun leader survived SIGKILL") from error
    if not _process_group_absent(process_group, TERMINATE_GRACE_SECONDS):
        raise GPUControllerError("srun descendants survived SIGKILL")


def _close_process_pipes(process: subprocess.Popen[bytes]) -> None:
    failures: list[BaseException] = []
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except BaseException as error:
                failures.append(error)
    if failures:
        raise GPUControllerError("srun pipe cleanup differs") from failures[0]


def run_single_srun(
    command: Sequence[str], payload: bytes, release: Mapping[str, Any],
    max_gate_step: int,
) -> tuple[int, bytes, bytes, int]:
    # Repeat the complete transport admission immediately at the dispatch
    # boundary so a caller cannot substitute a wider argv or different stdin
    # after the immutable dispatch claim.
    validate_srun_transport(
        command, max_gate_step=max_gate_step, payload=payload, release=release,
    )
    environment = srun_environment()
    process = subprocess.Popen(
        list(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=environment, close_fds=True,
        start_new_session=True,
    )
    process_group = process.pid
    try:
        stdout, stderr = process.communicate(
            input=payload, timeout=SRUN_TIMEOUT_SECONDS,
        )
    except BaseException as primary_error:
        try:
            _terminate_process_group(process)
        except BaseException as cleanup_error:
            try:
                _close_process_pipes(process)
            except BaseException:
                pass
            raise GPUControllerError("srun process group cleanup failed") from cleanup_error
        try:
            _close_process_pipes(process)
        except BaseException as cleanup_error:
            raise GPUControllerError("srun pipe cleanup failed") from cleanup_error
        raise primary_error
    if process.poll() is None:
        _terminate_process_group(process)
        raise GPUControllerError("srun remained live after communicate")
    if not _process_group_absent(process_group, TERMINATE_GRACE_SECONDS):
        _terminate_process_group(process)
        raise GPUControllerError("terminal srun process group required cleanup")
    _close_process_pipes(process)
    return int(process.returncode), stdout, stderr, process_group


def _load_final_json(path: Path, *, mode: int, label: str) -> tuple[HeldAuthority, dict[str, Any]]:
    named = os.lstat(path)
    held = open_authority(
        path, expected_sha256=_sha_file(path), expected_size=named.st_size,
        expected_mode=mode, maximum_size=MAX_JSON_SIZE,
    )
    try:
        return held, strict_json(held.raw, label=label)
    except BaseException:
        held.close(); raise


class PostflightClosure:
    """Postflight facts whose source FDs stay held through evidence commit."""

    def __init__(
        self, evidence: Mapping[str, Any], authorities: Sequence[HeldAuthority],
        directory_contracts: Sequence[tuple[HeldDirectory, set[str]]],
        owned_directories: Sequence[HeldDirectory] = (),
    ) -> None:
        self.evidence = dict(evidence)
        self.authorities = list(authorities)
        self.directory_contracts = [
            (directory, set(names)) for directory, names in directory_contracts
        ]
        self.owned_directories = list(owned_directories)

    def replay(self) -> None:
        for authority in self.authorities:
            authority.replay()
        for directory, expected_names in self.directory_contracts:
            directory.replay()
            with os.scandir(directory.descriptor) as entries:
                before = {
                    entry.name: identity(os.stat(
                        entry.name, dir_fd=directory.descriptor,
                        follow_symlinks=False,
                    ))
                    for entry in entries
                }
            if set(before) != expected_names:
                raise GPUControllerError(
                    f"held directory member closure changed: {directory.path}"
                )
            for authority in self.authorities:
                if authority.path.parent == directory.path:
                    authority.replay()
                    relative = os.stat(
                        authority.path.name, dir_fd=directory.descriptor,
                        follow_symlinks=False,
                    )
                    if identity(relative) != authority.held_identity:
                        raise GPUControllerError(
                            f"held directory leaf changed: {authority.path}"
                        )
            directory.replay()
            with os.scandir(directory.descriptor) as entries:
                after = {
                    entry.name: identity(os.stat(
                        entry.name, dir_fd=directory.descriptor,
                        follow_symlinks=False,
                    ))
                    for entry in entries
                }
            if after != before:
                raise GPUControllerError(
                    f"held directory identity map changed: {directory.path}"
                )
        for authority in self.authorities:
            authority.replay()

    def close(self) -> None:
        while self.authorities:
            self.authorities.pop().close()
        while self.owned_directories:
            self.owned_directories.pop().close()


def validate_compute_binding_closure(
    *, runtime: Mapping[str, Any], runtime_raw: bytes,
    rank_cache: Mapping[str, Any], rank_cache_raw: bytes,
    compute: Mapping[str, Any], attestation: Mapping[str, Any],
    attestation_raw: bytes, ready_plan: Mapping[str, Any],
    release: Mapping[str, Any], dispatch: Mapping[str, Any],
) -> None:
    """Reject any break in the step/release/runtime/cache/runner join."""

    release_digest = object_digest(release)
    identities = release.get("identities")
    if type(identities) is not dict:
        raise GPUControllerError("compute identity binding differs")
    if (
        strict_json(runtime_raw, label="bound runtime receipt") != dict(runtime)
        or strict_json(rank_cache_raw, label="bound rank-cache receipt")
        != dict(rank_cache)
        or strict_json(attestation_raw, label="bound runner attestation")
        != dict(attestation)
    ):
        raise GPUControllerError("compute raw/value binding differs")
    _self_digested(runtime, "receipt_digest", runtime.get("receipt_digest"))
    _self_digested(
        rank_cache, "receipt_digest", rank_cache.get("receipt_digest"),
    )
    _self_digested(compute, "result_digest", compute.get("result_digest"))
    _self_digested(
        attestation, "attestation_digest", attestation.get("attestation_digest"),
    )
    step = runtime.get("slurm_step_id")
    if (
        type(step) is not str or not step.isdecimal()
        or str(int(step)) != step or int(step) <= 0
        or runtime.get("production_release_digest") != release_digest
        or runtime.get("ready_plan_digest") != ready_plan.get("plan_digest")
        or runtime.get("r64_checkpoint_manifest_sha256")
        != identities.get("r64_checkpoint_manifest", {}).get("sha256")
        or runtime.get("exact26_identity_set_digest")
        != object_digest(identities)
        or rank_cache.get("slurm_step_id") != step
        or rank_cache.get("production_release_digest") != release_digest
        or rank_cache.get("runtime_receipt_sha256")
        != hashlib.sha256(runtime_raw).hexdigest()
        or rank_cache.get("runtime_receipt_digest")
        != runtime.get("receipt_digest")
        or rank_cache.get("runner_attestation_sha256")
        != hashlib.sha256(attestation_raw).hexdigest()
        or rank_cache.get("runner_attestation_digest")
        != attestation.get("attestation_digest")
        or compute.get("slurm_step_id") != step
        or compute.get("production_release_digest") != release_digest
        or compute.get("ready_plan_digest") != ready_plan.get("plan_digest")
        or compute.get("runtime_receipt_sha256")
        != hashlib.sha256(runtime_raw).hexdigest()
        or compute.get("runtime_receipt_digest")
        != runtime.get("receipt_digest")
        or compute.get("rank_cache_receipt_sha256")
        != hashlib.sha256(rank_cache_raw).hexdigest()
        or compute.get("rank_cache_receipt_digest")
        != rank_cache.get("receipt_digest")
        or compute.get("internal_artifact_count") != 55
        or compute.get("internal_artifact_inventory_digest")
        != rank_cache.get("internal_artifact_inventory_digest")
        or compute.get("model_authority_root_identity")
        != rank_cache.get("model_authority_root_identity")
        or compute.get("model_authority_root_empty") is not True
        or compute.get("model_authority_root_held_and_terminal_replayed")
        is not True
        or compute.get("runner_attestation_sha256")
        != hashlib.sha256(attestation_raw).hexdigest()
        or compute.get("runner_attestation_digest")
        != attestation.get("attestation_digest")
        or compute.get("rank_processes_zero") is not True
        or compute.get("rank_cache_compute_state")
        != "RETAINED_COMPUTE_LOCAL_POSTFLIGHT_AUTHORITY"
        or dispatch.get("production_release_digest") != release_digest
        or dispatch.get("ready_plan_digest") != ready_plan.get("plan_digest")
    ):
        raise GPUControllerError("compute authority binding closure differs")


def validate_compute_package_root_identity(value: Any) -> None:
    """Bind the compute-node package observation to the publication inode."""

    if (
        type(value) is not list or len(value) != 11
        or any(type(part) is not int for part in value)
        or value != PACKAGE_ROOT_IDENTITY
    ):
        raise GPUControllerError("compute package-root identity differs")


def validate_internal_artifact_bindings(
    attestation: Mapping[str, Any],
    internal_inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    suffixes = {
        "model_capture": "-model-capture.json",
        "model_pre_use": "-model-pre-use.json",
        "consumption_input": "-consumption-input.json",
        "model_post_use": "-model-post-use.json",
        "eval_consumption_chain": "-eval-consumption-chain.json",
        "adapter_capture": "-adapter-capture.json",
        "adapter_pre_use": "-adapter-pre-use.json",
        "adapter_post_use": "-adapter-post-use.json",
        "adapter_final": "-adapter-final.json",
    }
    task_results = attestation.get("task_results")
    artifact_replays = attestation.get("task_artifact_replays")
    if (
        type(task_results) is not list or len(task_results) != 5
        or type(artifact_replays) is not list or len(artifact_replays) != 5
        or [row.get("task_id") for row in task_results] != list(TASK_IDS)
        or [row.get("task_id") for row in artifact_replays] != list(TASK_IDS)
        or len(internal_inventory) != 55
    ):
        raise GPUControllerError("internal artifact cardinality differs")
    by_name = {row.get("basename"): row for row in internal_inventory}
    if len(by_name) != 55 or None in by_name:
        raise GPUControllerError("internal artifact basename closure differs")
    for index, task_id in enumerate(TASK_IDS):
        result = task_results[index]
        unsigned = dict(result)
        task_digest = unsigned.pop("task_result_digest", None)
        refs = result.get("authority_artifacts")
        replayed = artifact_replays[index]
        prefix = f".matched-v2-{index:02d}-{task_id}"
        if (
            task_digest != object_digest(unsigned)
            or type(refs) is not dict or set(refs) != set(suffixes)
            or result.get("log_basename") != prefix + ".log"
            or replayed.get("task_result_digest") != task_digest
            or replayed.get("artifact_count") != 9
        ):
            raise GPUControllerError(f"internal task authority differs: {task_id}")
        for role, suffix in suffixes.items():
            basename = prefix + suffix
            ref = refs[role]
            row = by_name.get(basename)
            if (
                type(ref) is not dict or set(ref) != {"basename", "sha256"}
                or ref.get("basename") != basename
                or row is None or row.get("role") != role
                or row.get("task_index") != index or row.get("task_id") != task_id
                or row.get("sha256") != ref.get("sha256")
            ):
                raise GPUControllerError(f"internal JSON authority differs: {basename}")
        runner_row = by_name.get(prefix + "-runner-task.json")
        log_row = by_name.get(prefix + ".log")
        if (
            runner_row is None or runner_row.get("role") != "runner_task"
            or runner_row.get("sha256")
            != replayed.get("runner_task_file_sha256")
            or log_row is None or log_row.get("role") != "task_log"
            or log_row.get("task_index") != index
            or log_row.get("task_id") != task_id
        ):
            raise GPUControllerError(f"internal runner/log authority differs: {task_id}")
    return by_name


def validate_postflight(
    *, ready_plan: Mapping[str, Any], ready_plan_raw: bytes,
    stdout_raw: bytes, stderr_raw: bytes, release: Mapping[str, Any],
    dispatch: Mapping[str, Any],
    target_directories: Mapping[str, HeldDirectory],
) -> PostflightClosure:
    if stderr_raw != b"" or stdout_raw.count(b"\n") != 1:
        raise GPUControllerError("GPU terminal streams differ")
    stdout_value = strict_json(stdout_raw, label="GPU runner stdout")
    runtime_held: HeldAuthority | None = None
    rank_cache_held: HeldAuthority | None = None
    report_held: HeldAuthority | None = None
    attestation_held: HeldAuthority | None = None
    media_held: list[HeldAuthority] = []
    authority_directory: HeldDirectory | None = None
    completed = False
    try:
        runtime_held, runtime = _load_final_json(
            RUNTIME_RECEIPT_PATH, mode=RECEIPT_MODE,
            label="GPU runtime receipt",
        )
        rank_cache_held, rank_cache = _load_final_json(
            RANK_CACHE_RECEIPT_PATH, mode=RECEIPT_MODE,
            label="compute rank-cache receipt",
        )
        report_held, report = _load_final_json(
            OUTPUT_REPORT_PATH, mode=FILE_MODE, label="exact5 report",
        )
        attestation_held, attestation = _load_final_json(
            RUNNER_ATTESTATION_PATH, mode=FILE_MODE,
            label="runner attestation",
        )
        _self_digested(runtime, "receipt_digest", runtime.get("receipt_digest"))
        release_digest = object_digest(release)
        release_identities = release.get("identities")
        site_row = release.get("directory_authorities", {}).get("site_packages")
        if (
            set(runtime) != RUNTIME_RECEIPT_FIELDS
            or runtime.get("schema_version") != RUNTIME_SCHEMA
            or runtime.get("status") != "PASS_GPU_RUNTIME_BEFORE_RUNNER"
            or runtime.get("holder_job_id") != HOLDER_JOB_ID
            or runtime.get("node") != NODE
            or type(runtime.get("slurm_step_id")) is not str
            or not runtime["slurm_step_id"].isdecimal()
            or str(int(runtime["slurm_step_id"])) != runtime["slurm_step_id"]
            or int(runtime["slurm_step_id"]) <= 0
            or runtime.get("production_release_digest") != release_digest
            or runtime.get("ready_plan_digest") != ready_plan["plan_digest"]
            or runtime.get("torch_version") != EXPECTED_TORCH_VERSION
            or runtime.get("hip_version") != EXPECTED_HIP_VERSION
            or runtime.get("device_count") != GPU_COUNT
            or runtime.get("device_names") != [EXPECTED_GPU_NAME] * GPU_COUNT
            or runtime.get("r64_checkpoint_manifest_sha256")
            != release_identities["r64_checkpoint_manifest"]["sha256"]
            or runtime.get("exact26_identity_set_digest")
            != object_digest(release_identities)
            or runtime.get("site_packages_root") != str(SITE_PACKAGES_ROOT)
            or runtime.get("site_packages_identity") != site_row["identity"]
            or runtime.get("torch_package_init_authority")
            != release.get("torch_package_init_authority")
            or runtime.get("torch_module_path")
            != str(TORCH_PACKAGE_INIT_PATH)
            or runtime.get("torch_module_sha256")
            != release.get("torch_package_init_authority", {}).get("sha256")
            or runtime.get("torch_package_entry_compiled_from_held_source")
            is not True
            or runtime.get("torch_source_loader") != "SourceFileLoader"
            or runtime.get("isolated_flag") != 1
            or runtime.get("no_site_flag") != 1
            or runtime.get("dont_write_bytecode") is not True
            or runtime.get(
                "site_packages_activated_explicitly_under_isolated_no_site"
            ) is not True
            or runtime.get("renderer_started") is not False
        ):
            raise GPUControllerError("GPU runtime receipt semantics differ")
        _self_digested(
            rank_cache, "receipt_digest", rank_cache.get("receipt_digest"),
        )
        topology = rank_cache.get("task_topology")
        inventory = rank_cache.get("inventory")
        internal_inventory = rank_cache.get("internal_artifact_inventory")
        cache_identity = rank_cache.get("rank_cache_root_identity")
        tmp_identity = rank_cache.get("rank_cache_parent_identity")
        package_identity = rank_cache.get("package_root_identity")
        validate_compute_package_root_identity(package_identity)
        authority_root_identity = rank_cache.get("model_authority_root_identity")
        mount = rank_cache.get("cache_mount")
        if (
            set(rank_cache) != RANK_CACHE_RECEIPT_FIELDS
            or rank_cache.get("schema_version") != RANK_CACHE_SCHEMA
            or rank_cache.get("status")
            != "PASS_RETAINED_COMPUTE_LOCAL_RANK_CACHE_AUTHORITY"
            or rank_cache.get("holder_job_id") != HOLDER_JOB_ID
            or rank_cache.get("node") != NODE
            or rank_cache.get("hostname") != NODE
            or rank_cache.get("uname_nodename") != NODE
            or rank_cache.get("slurm_step_id") != runtime["slurm_step_id"]
            or rank_cache.get("production_release_digest") != release_digest
            or rank_cache.get("runtime_receipt_sha256")
            != hashlib.sha256(runtime_held.raw).hexdigest()
            or rank_cache.get("runtime_receipt_digest")
            != runtime["receipt_digest"]
            or rank_cache.get("rank_cache_root") != str(RANK_CACHE_ROOT)
            or type(cache_identity) is not list or len(cache_identity) != 11
            or type(tmp_identity) is not list or len(tmp_identity) != 11
            or type(package_identity) is not list or len(package_identity) != 11
            or type(authority_root_identity) is not list
            or len(authority_root_identity) != 11
            or rank_cache.get("rank_cache_parent") != "/tmp"
            or rank_cache.get("package_root") != str(PACKAGE_ROOT)
            or cache_identity[0] != 64_768
            or tmp_identity[0] != 64_768
            or tmp_identity[1] != 28_311_553
            or tmp_identity[2:4] != [0, 0]
            or stat.S_IMODE(tmp_identity[4]) != 0o1777
            or package_identity != PACKAGE_ROOT_IDENTITY
            or rank_cache.get("model_authority_root") != str(AUTHORITY_ROOT)
            or authority_root_identity[2:4] != [REMOTE_UID, REMOTE_GID]
            or not stat.S_ISDIR(authority_root_identity[4])
            or stat.S_IMODE(authority_root_identity[4]) != 0o700
            or rank_cache.get("model_authority_root_empty") is not True
            or rank_cache.get(
                "model_authority_root_held_and_terminal_replayed"
            ) is not True
            or rank_cache.get("cache_device_matches_tmp") is not True
            or rank_cache.get("cache_device_differs_from_package") is not True
            or type(mount) is not dict
            or mount.get("major_minor") != "253:0"
            or mount.get("fs_type") != "ext4"
            or mount.get("mount_point") != "/"
            or mount.get("mount_source") != "/dev/mapper/vgroot-lvroot"
            or SHA_RE.fullmatch(str(mount.get("mountinfo_sha256"))) is None
            or rank_cache.get("compute_node_observation") is not True
            or rank_cache.get("retained_compute_local") is not True
            or rank_cache.get("fresh_unique_attempt_path") is not True
            or rank_cache.get("freshness_enforced_by_frozen_runner_before_mkdir")
            is not True
            or rank_cache.get("cleanup_performed") is not False
            or rank_cache.get("absent_claimed") is not False
            or rank_cache.get("non_scientific_cache") is not True
            or rank_cache.get("cache_is_not_output_or_result") is not True
            or rank_cache.get("task_count") != 5
            or rank_cache.get("rank_process_count") != 20
            or rank_cache.get("coordinator_process_count") != 5
            or type(topology) is not list or len(topology) != 5
            or [row.get("task_index") for row in topology] != list(range(5))
            or [row.get("task_id") for row in topology] != list(TASK_IDS)
            or any(
                row.get("task_directory") != f"task-{index:02d}-{TASK_IDS[index]}"
                or row.get("coordinator_directory") != "coordinator"
                or row.get("rank_directories")
                != ["rank-0", "rank-1", "rank-2", "rank-3"]
                for index, row in enumerate(topology)
            )
            or type(rank_cache.get("inventory_entry_count")) is not int
            or rank_cache["inventory_entry_count"] < 281
            or type(inventory) is not list
            or len(inventory) != rank_cache["inventory_entry_count"]
            or any(
                type(row) is not dict or type(row.get("path")) is not str
                or row.get("kind") not in {"directory", "file"}
                or (
                    row.get("kind") == "directory"
                    and (set(row) != {"kind", "path", "mode", "identity"}
                         or row.get("mode") != 0o700
                         or type(row.get("identity")) is not list
                         or len(row["identity"]) != 11)
                )
                or (
                    row.get("kind") == "file"
                    and (
                        set(row)
                        != {"kind", "path", "mode", "identity", "size", "sha256"}
                        or type(row.get("mode")) is not int
                        or row["mode"] & 0o077
                        or type(row.get("identity")) is not list
                        or len(row["identity"]) != 11
                        or type(row.get("size")) is not int or row["size"] < 0
                        or SHA_RE.fullmatch(str(row.get("sha256"))) is None
                    )
                )
                for row in inventory
            )
            or [row.get("path") for row in inventory]
            != sorted(row.get("path") for row in inventory)
            or len({row.get("path") for row in inventory}) != len(inventory)
            or object_digest(inventory) != rank_cache.get("inventory_digest")
            or rank_cache.get("inventory_bound") != {
                "maximum_depth": 32, "maximum_entries": 50_000,
                "maximum_file_bytes": 4_294_967_296,
                "maximum_single_file_bytes": 1_073_741_824,
            }
            or type(rank_cache.get("inventory_total_file_bytes")) is not int
            or rank_cache["inventory_total_file_bytes"] < 0
            or SHA_RE.fullmatch(str(rank_cache.get("inventory_digest"))) is None
            or type(internal_inventory) is not list
            or len(internal_inventory) != 55
            or rank_cache.get("internal_artifact_count") != 55
            or object_digest(internal_inventory)
            != rank_cache.get("internal_artifact_inventory_digest")
            or rank_cache.get("internal_artifact_fds_held_and_terminal_replayed")
            is not True
            or any(
                type(row) is not dict
                or set(row) != {
                    "task_index", "task_id", "role", "basename", "sha256",
                    "size", "identity",
                }
                or type(row.get("task_index")) is not int
                or row["task_index"] not in range(5)
                or row.get("task_id") != TASK_IDS[row["task_index"]]
                or type(row.get("role")) is not str or not row["role"]
                or type(row.get("basename")) is not str
                or "/" in row["basename"]
                or SHA_RE.fullmatch(str(row.get("sha256"))) is None
                or type(row.get("size")) is not int or row["size"] < 0
                or type(row.get("identity")) is not list
                or len(row["identity"]) != 11
                or stat.S_IMODE(row["identity"][4]) != RECEIPT_MODE
                or row["identity"][5] != 1
                or row["identity"][7] != row["size"]
                for row in internal_inventory
            )
            or [row.get("basename") for row in internal_inventory]
            != sorted(row.get("basename") for row in internal_inventory)
            or len({row.get("basename") for row in internal_inventory}) != 55
            or rank_cache.get("rank_processes_zero") is not True
            or rank_cache.get("torchrun_processes_zero") is not True
            or rank_cache.get("terminal_inventory_replayed_exactly") is not True
            or rank_cache.get("cgroup_returned_exactly_to_single_root_process")
            is not True
            or type(rank_cache.get("cgroup_baseline")) is not dict
            or type(rank_cache.get("cgroup_terminal")) is not dict
            or rank_cache["cgroup_baseline"].get("version") != 2
            or rank_cache["cgroup_terminal"].get("version") != 2
            or rank_cache["cgroup_baseline"].get("membership_path")
            != (
                f"/system.slice/slurmstepd.scope/job_{HOLDER_JOB_ID}/"
                f"step_{runtime['slurm_step_id']}/user/task_0"
            )
            or rank_cache["cgroup_terminal"].get("membership_path")
            != rank_cache["cgroup_baseline"].get("membership_path")
            or rank_cache["cgroup_terminal"].get("membership_sha256")
            != rank_cache["cgroup_baseline"].get("membership_sha256")
            or rank_cache["cgroup_terminal"].get("cgroup_procs_path")
            != rank_cache["cgroup_baseline"].get("cgroup_procs_path")
            or type(rank_cache["cgroup_baseline"].get("cgroup_procs_identity"))
            is not list
            or len(rank_cache["cgroup_baseline"]["cgroup_procs_identity"]) != 11
            or rank_cache["cgroup_terminal"].get("cgroup_procs_identity")
            != rank_cache["cgroup_baseline"].get("cgroup_procs_identity")
            or rank_cache["cgroup_baseline"]["cgroup_procs_identity"][2:4]
            != [0, 0]
            or not stat.S_ISREG(
                rank_cache["cgroup_baseline"]["cgroup_procs_identity"][4]
            )
            or stat.S_IMODE(
                rank_cache["cgroup_baseline"]["cgroup_procs_identity"][4]
            ) != 0o644
            or rank_cache["cgroup_terminal"].get("cgroup_procs_anchor")
            != rank_cache["cgroup_baseline"].get("cgroup_procs_anchor")
            or rank_cache["cgroup_baseline"].get("pids")
            != [rank_cache["cgroup_baseline"].get("self_pid")]
            or rank_cache["cgroup_terminal"].get("pids")
            != [rank_cache["cgroup_terminal"].get("self_pid")]
            or rank_cache["cgroup_terminal"].get("self_pid")
            != rank_cache["cgroup_baseline"].get("self_pid")
            or rank_cache.get("process_scan_performed") is not True
            or rank_cache.get("process_scan_sources")
            != ["/proc/*/environ", "/proc/*/cmdline"]
            or rank_cache.get("matched_residual_pids") != []
            or type(rank_cache.get("process_scan")) is not dict
            or rank_cache["process_scan"].get("matched_pids") != []
            or rank_cache["process_scan"].get("unreadable_pids") != []
            or type(rank_cache["process_scan"].get("same_uid_processes_scanned"))
            is not int
        ):
            raise GPUControllerError("compute rank-cache receipt semantics differ")
        _self_digested(report, "report_digest", report.get("report_digest"))
        results = report.get("results")
        if (
            report.get("schema_version") != REPORT_SCHEMA
            or report.get("status")
            != "ENGINEERING_ORACLE_COMPLETE_AWAITING_MANUAL_REVIEW"
            or report.get("campaign_mode") != CAMPAIGN
            or report.get("plan_schema_version") != READY_PLAN_SCHEMA
            or report.get("plan_digest") != ready_plan["plan_digest"]
            or report.get("task_count") != 5
            or report.get("task_ids") != list(TASK_IDS)
            or report.get("variant_order") != list(ARM_ORDER)
            or report.get("all_exact5_tasks_verified_no_cherry_pick") is not True
            or report.get("same_model_capture_all_tasks") is not True
            or report.get("manual_blind_review_required") is not True
            or report.get("formal_full16_report") is not False
            or type(results) is not list or len(results) != 5
            or any(type(row) is not dict for row in results)
            or [row.get("task_id") for row in results]
            != list(TASK_IDS)
        ):
            raise GPUControllerError("exact5 report semantics differ")
        _self_digested(
            attestation, "attestation_digest", attestation.get("attestation_digest"),
        )
        if (
            attestation.get("schema_version") != RUNNER_SCHEMA
            or attestation.get("status") != "EXACT5_COMPLETE_AWAITING_BLIND_REVIEW"
            or attestation.get("campaign_mode") != CAMPAIGN
            or attestation.get("plan", {}).get("path") != str(READY_PLAN_PATH)
            or attestation.get("plan", {}).get("sha256")
            != hashlib.sha256(ready_plan_raw).hexdigest()
            or attestation.get("plan", {}).get("plan_digest")
            != ready_plan["plan_digest"]
            or attestation.get("task_count") != 5
            or attestation.get("task_ids") != list(TASK_IDS)
            or attestation.get("all_exact5_tasks_attempted_exactly_once") is not True
            or attestation.get("all_exact5_tasks_succeeded") is not True
            or attestation.get("retry_count") != 0
            or attestation.get("same_model_capture_all_exact5_tasks") is not True
            or attestation.get("native_receipts_replayed_0400_single_link") is not True
            or attestation.get("scientific_claim_authorized") is not False
            or attestation.get("formal_claim_authorized") is not False
            or attestation.get("verified_report", {}).get("sha256")
            != hashlib.sha256(report_held.raw).hexdigest()
            or attestation.get("verified_report", {}).get("report_digest")
            != report["report_digest"]
        ):
            raise GPUControllerError("runner attestation semantics differ")
        task_results = attestation.get("task_results")
        if (
            type(task_results) is not list or len(task_results) != 5
            or [row.get("task_index") for row in task_results] != list(range(5))
            or [row.get("task_id") for row in task_results] != list(TASK_IDS)
            or any(row.get("return_code") != 0 for row in task_results)
            or topology is None
            or [row.get("environment_digest") for row in topology]
            != [row.get("environment_digest") for row in task_results]
            or rank_cache.get("runner_attestation_sha256")
            != hashlib.sha256(attestation_held.raw).hexdigest()
            or rank_cache.get("runner_attestation_digest")
            != attestation["attestation_digest"]
        ):
            raise GPUControllerError("runner/cache task closure differs")
        artifact_suffixes = {
            "model_capture": "-model-capture.json",
            "model_pre_use": "-model-pre-use.json",
            "consumption_input": "-consumption-input.json",
            "model_post_use": "-model-post-use.json",
            "eval_consumption_chain": "-eval-consumption-chain.json",
            "adapter_capture": "-adapter-capture.json",
            "adapter_pre_use": "-adapter-pre-use.json",
            "adapter_post_use": "-adapter-post-use.json",
            "adapter_final": "-adapter-final.json",
        }
        artifact_replays = attestation.get("task_artifact_replays")
        internal_by_name = validate_internal_artifact_bindings(
            attestation, internal_inventory,
        )
        if (
            type(artifact_replays) is not list or len(artifact_replays) != 5
            or [row.get("task_id") for row in artifact_replays] != list(TASK_IDS)
            or attestation.get("task_result_digests")
            != [row.get("task_result_digest") for row in task_results]
        ):
            raise GPUControllerError("attested internal replay set differs")
        for index, task_id in enumerate(TASK_IDS):
            result = task_results[index]
            unsigned_result = dict(result)
            task_result_digest = unsigned_result.pop("task_result_digest", None)
            refs = result.get("authority_artifacts")
            replayed = artifact_replays[index]
            prefix = f".matched-v2-{index:02d}-{task_id}"
            if (
                task_result_digest != object_digest(unsigned_result)
                or type(refs) is not dict or set(refs) != set(artifact_suffixes)
                or result.get("attempt_count") != 1
                or result.get("retry_allowed") is not False
                or result.get("log_basename") != prefix + ".log"
                or replayed.get("task_result_digest") != task_result_digest
                or replayed.get("artifact_count") != 9
                or SHA_RE.fullmatch(
                    str(replayed.get("artifact_rows_digest"))
                ) is None
            ):
                raise GPUControllerError(f"task internal replay differs: {task_id}")
            for role, suffix in artifact_suffixes.items():
                basename = prefix + suffix
                reference = refs[role]
                observed = internal_by_name.get(basename)
                if (
                    type(reference) is not dict
                    or set(reference) != {"basename", "sha256"}
                    or reference.get("basename") != basename
                    or observed is None or observed.get("role") != role
                    or observed.get("task_index") != index
                    or observed.get("task_id") != task_id
                    or observed.get("sha256") != reference.get("sha256")
                ):
                    raise GPUControllerError(
                        f"attested artifact binding differs: {basename}"
                    )
            runner_basename = prefix + "-runner-task.json"
            runner_row = internal_by_name.get(runner_basename)
            log_row = internal_by_name.get(prefix + ".log")
            if (
                runner_row is None or runner_row.get("role") != "runner_task"
                or runner_row.get("sha256")
                != replayed.get("runner_task_file_sha256")
                or log_row is None or log_row.get("role") != "task_log"
                or log_row.get("task_index") != index
                or log_row.get("task_id") != task_id
            ):
                raise GPUControllerError(
                    f"attested runner/log binding differs: {task_id}"
                )

        _self_digested(
            stdout_value, "result_digest", stdout_value.get("result_digest"),
        )
        if (
            set(stdout_value) != COMPUTE_RESULT_FIELDS
            or stdout_value.get("schema_version") != COMPUTE_RESULT_SCHEMA
            or stdout_value.get("status")
            != "PASS_EXACT_FIVE_COMPUTE_POSTFLIGHT"
            or stdout_value.get("holder_job_id") != HOLDER_JOB_ID
            or stdout_value.get("node") != NODE
            or stdout_value.get("slurm_step_id") != runtime["slurm_step_id"]
            or stdout_value.get("runner_returncode") != 0
            or stdout_value.get("production_release_digest") != release_digest
            or stdout_value.get("ready_plan_digest")
            != ready_plan["plan_digest"]
            or stdout_value.get("runtime_receipt_sha256")
            != hashlib.sha256(runtime_held.raw).hexdigest()
            or stdout_value.get("runtime_receipt_digest")
            != runtime["receipt_digest"]
            or stdout_value.get("rank_cache_receipt_sha256")
            != hashlib.sha256(rank_cache_held.raw).hexdigest()
            or stdout_value.get("rank_cache_receipt_digest")
            != rank_cache["receipt_digest"]
            or stdout_value.get("runner_attestation_sha256")
            != hashlib.sha256(attestation_held.raw).hexdigest()
            or stdout_value.get("runner_attestation_digest")
            != attestation["attestation_digest"]
            or stdout_value.get("all_five_arms_attempted_exactly_once")
            is not True
            or stdout_value.get("retry_count") != 0
            or stdout_value.get("rank_processes_zero") is not True
            or stdout_value.get("rank_process_scan_performed") is not True
            or stdout_value.get("rank_cache_compute_state")
            != "RETAINED_COMPUTE_LOCAL_POSTFLIGHT_AUTHORITY"
            or stdout_value.get("rank_cache_inventory_digest")
            != rank_cache["inventory_digest"]
            or stdout_value.get("internal_artifact_count") != 55
            or stdout_value.get("internal_artifact_inventory_digest")
            != rank_cache["internal_artifact_inventory_digest"]
            or stdout_value.get("model_authority_root_identity")
            != authority_root_identity
            or stdout_value.get("model_authority_root_empty") is not True
            or stdout_value.get(
                "model_authority_root_held_and_terminal_replayed"
            ) is not True
            or stdout_value.get("local_srun_process_group_observed") is not False
            or dispatch.get("production_release_digest") != release_digest
            or dispatch.get("ready_plan_digest") != ready_plan["plan_digest"]
        ):
            raise GPUControllerError("compute result cross-link differs")
        validate_compute_binding_closure(
            runtime=runtime, runtime_raw=runtime_held.raw,
            rank_cache=rank_cache, rank_cache_raw=rank_cache_held.raw,
            compute=stdout_value, attestation=attestation,
            attestation_raw=attestation_held.raw, ready_plan=ready_plan,
            release=release, dispatch=dispatch,
        )

        suffixes = (
            "-model-capture.json", "-model-pre-use.json",
            "-consumption-input.json", "-model-post-use.json",
            "-eval-consumption-chain.json", "-adapter-capture.json",
            "-adapter-pre-use.json", "-adapter-post-use.json",
            "-adapter-final.json", ".log", "-runner-task.json",
        )
        expected_names = {
            *(f"{task_id}.mp4" for task_id in TASK_IDS),
            *(f"{task_id}.mp4.receipt.json" for task_id in TASK_IDS),
            *(
                f".matched-v2-{index:02d}-{task_id}{suffix}"
                for index, task_id in enumerate(TASK_IDS) for suffix in suffixes
            ),
        }
        _exact_names(OUTPUT_ROOT, expected_names, label="post-GPU media")
        for index, task_id in enumerate(TASK_IDS):
            video = OUTPUT_ROOT / f"{task_id}.mp4"
            receipt_path = OUTPUT_ROOT / f"{task_id}.mp4.receipt.json"
            video_named = os.lstat(video)
            video_held = open_authority(
                video, expected_sha256=_sha_file(video),
                expected_size=video_named.st_size, expected_mode=FILE_MODE,
                maximum_size=MAX_VIDEO_SIZE,
            )
            media_held.append(video_held)
            receipt_held, receipt = _load_final_json(
                receipt_path, mode=RECEIPT_MODE,
                label=f"native receipt {task_id}",
            )
            media_held.append(receipt_held)
            unsigned = dict(receipt); claimed = unsigned.pop("receipt_digest", None)
            output = receipt.get("output")
            verified = results[index]
            publication_identity = held_publication_identity(video_held)
            expected_receipt_schema = ready_plan.get("producer", {}).get(
                "inference_receipt_schemas", {},
            ).get(
                "off" if index in {0, 4} else "route_or_active",
            )
            if (
                claimed != object_digest(unsigned)
                or type(output) is not dict
                or receipt.get("schema_version") != expected_receipt_schema
                or receipt.get("production_claim_forbidden") is not True
                or receipt.get("scientific_claim_authorized") is not False
                or output.get("path") != str(video)
                or output.get("sha256")
                != hashlib.sha256(video_held.raw).hexdigest()
                or output.get("size") != len(video_held.raw)
                or output.get("publication_identity") != publication_identity
                or verified.get("task_id") != task_id
                or verified.get("receipt_path") != str(receipt_path)
                or verified.get("output_path") != str(video)
                or verified.get("receipt_file_sha256")
                != hashlib.sha256(receipt_held.raw).hexdigest()
                or verified.get("receipt_digest") != claimed
                or verified.get("output_sha256")
                != hashlib.sha256(video_held.raw).hexdigest()
                or verified.get("output_size") != len(video_held.raw)
            ):
                raise GPUControllerError(f"native media receipt differs: {task_id}")
        for name in expected_names:
            info = os.lstat(OUTPUT_ROOT / name)
            if (
                not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != REMOTE_UID or info.st_gid != REMOTE_GID
                or stat.S_IMODE(info.st_mode) not in {FILE_MODE, RECEIPT_MODE}
            ):
                raise GPUControllerError(f"post-GPU media leaf differs: {name}")
        already_held = {
            *(f"{task_id}.mp4" for task_id in TASK_IDS),
            *(f"{task_id}.mp4.receipt.json" for task_id in TASK_IDS),
        }
        for name in sorted(expected_names - already_held):
            path = OUTPUT_ROOT / name
            expected_internal = internal_by_name.get(name)
            if expected_internal is None:
                raise GPUControllerError(f"compute internal row absent: {name}")
            internal_held = open_authority(
                path, expected_sha256=expected_internal["sha256"],
                expected_size=expected_internal["size"],
                expected_mode=RECEIPT_MODE,
                maximum_size=(MAX_VIDEO_SIZE if name.endswith(".log") else MAX_JSON_SIZE),
            )
            if internal_held.held_identity != tuple(expected_internal["identity"]):
                internal_held.close()
                raise GPUControllerError(
                    f"compute internal identity changed: {name}"
                )
            media_held.append(internal_held)
        final_names = {OUTPUT_REPORT_PATH.name, RUNNER_ATTESTATION_PATH.name}
        runtime_names = {
            READY_PLAN_PATH.name, RUNTIME_RECEIPT_PATH.name,
            RANK_CACHE_RECEIPT_PATH.name, AUTHORITY_ROOT.name,
        }
        log_names = {
            STDOUT_PATH.name, STDERR_PATH.name,
        }
        expected_directory_paths = {
            "output": OUTPUT_ROOT, "final": PACKAGE_ROOT / "final",
            "runtime": PACKAGE_ROOT / "runtime", "logs": PACKAGE_ROOT / "logs",
        }
        if (
            not set(expected_directory_paths).issubset(target_directories)
            or any(
                target_directories[label].path != path
                for label, path in expected_directory_paths.items()
            )
        ):
            raise GPUControllerError("postflight held directory set differs")
        _exact_names(PACKAGE_ROOT / "final", final_names, label="post-GPU final")
        _exact_names(PACKAGE_ROOT / "runtime", runtime_names, label="post-GPU runtime")
        _exact_names(AUTHORITY_ROOT, set(), label="post-GPU model authority")
        _exact_names(PACKAGE_ROOT / "logs", log_names, label="post-GPU logs")
        authority_directory = open_directory(
            AUTHORITY_ROOT, expected_identity=authority_root_identity,
        )
        if stat.S_IMODE(authority_directory.held_identity[4]) != 0o700:
            raise GPUControllerError("model authority directory mode differs")
        evidence = {
            "runtime": runtime_held.row(), "report": report_held.row(),
            "rank_cache": rank_cache_held.row(),
            "attestation": attestation_held.row(),
            "compute_result_digest": stdout_value["result_digest"],
            "slurm_step_id": runtime["slurm_step_id"],
            "rank_cache_compute_state":
            "RETAINED_COMPUTE_LOCAL_POSTFLIGHT_AUTHORITY",
            "rank_cache_inventory_digest": rank_cache["inventory_digest"],
            "compute_rank_processes_zero": True,
            "media_file_count": 65,
            "video_count": 5, "native_receipt_count": 5,
            "internal_artifact_count": 55,
            "all_native_and_internal_leaf_fds_held": True,
            "all_five_arms_exactly_once": True,
        }
        authorities = [
            runtime_held, rank_cache_held, report_held, attestation_held,
            *media_held,
        ]
        directory_contracts = [
            (target_directories["output"], expected_names),
            (target_directories["final"], final_names),
            (target_directories["runtime"], runtime_names),
            (target_directories["logs"], log_names),
            (authority_directory, set()),
        ]
        closure = PostflightClosure(
            evidence, authorities, directory_contracts, [authority_directory],
        )
        closure.replay()
        completed = True
        return closure
    finally:
        if not completed:
            for held in media_held:
                held.close()
            if runtime_held is not None:
                runtime_held.close()
            if rank_cache_held is not None:
                rank_cache_held.close()
            if report_held is not None:
                report_held.close()
            if attestation_held is not None:
                attestation_held.close()
            if authority_directory is not None:
                authority_directory.close()


def controller() -> dict[str, Any]:
    gate: PackageGate | None = None
    authorities: dict[str, HeldAuthority] = {}
    attempt_authority: HeldAuthority | None = None
    dispatch_authority: HeldAuthority | None = None
    ready_plan_authority: HeldAuthority | None = None
    site_packages_authority: HeldDirectory | None = None
    torch_package_init_authority: HeldAuthority | None = None
    stdout_authority: HeldAuthority | None = None
    stderr_authority: HeldAuthority | None = None
    postflight_closure: PostflightClosure | None = None
    self_authority: HeldAuthority | None = None
    srun_authority: HeldAuthority | None = None
    directories: dict[str, HeldDirectory] = {}
    try:
        # Literal admission boundary: no package root, source, target, self, or
        # executable is observed before all upstream receipt authorities.
        gate = open_package_gate()
        report = gate.values["materialization"]
        authorities, launch_input = _open_package_identities(report)
        site_packages_path = validate_site_packages_layout(launch_input)
        site_packages_authority = open_directory(site_packages_path)
        torch_package_init_authority = open_observed_authority(
            TORCH_PACKAGE_INIT_PATH, expected_mode=0o644,
            maximum_size=MAX_SOURCE_SIZE,
        )
        gate.replay()

        # The named HOLD payload is held only to prove it remains unchanged; it
        # is never passed to Bash or srun.
        hold_payload_row = report["launch"]
        if (
            hashlib.sha256(authorities["plan"].raw).hexdigest()
            != report["plan"]["sha256"]
            or len(authorities["plan"].raw)
            != launch_input["identities"]["plan"]["size"]
        ):
            raise GPUControllerError("held HOLD plan/report binding differs")
        hold_payload = open_authority(
            HOLD_PAYLOAD_PATH, expected_sha256=hold_payload_row["payload_sha256"],
            expected_size=hold_payload_row["payload_size"],
            expected_mode=FILE_MODE, maximum_size=MAX_SOURCE_SIZE,
        )
        authorities["hold_payload"] = hold_payload
        if b"exit 88" not in hold_payload.raw or b"srun" in hold_payload.raw:
            raise GPUControllerError("named HOLD payload semantics differ")

        srun_authority = open_authority(
            Path(SRUN_AUTHORITY["path"]),
            expected_sha256=SRUN_AUTHORITY["sha256"],
            expected_size=SRUN_AUTHORITY["size"], expected_mode=0o755,
            maximum_size=MAX_SOURCE_SIZE, executable=True,
            expected_uid=0, expected_gid=0,
        )
        self_authority = open_observed_authority(
            Path(__file__).resolve(strict=True), expected_mode=FILE_MODE,
            maximum_size=MAX_SOURCE_SIZE,
        )
        for label, path in (
            ("evidence", PACKAGE_ROOT / "evidence"),
            ("runtime", PACKAGE_ROOT / "runtime"),
            ("logs", PACKAGE_ROOT / "logs"),
            ("output", OUTPUT_ROOT), ("final", PACKAGE_ROOT / "final"),
        ):
            directories[label] = open_directory(path)
        require_fresh_outputs()
        ready_plan, ready_plan_raw = derive_ready_plan(authorities["plan"].raw)
        ready_plan_sha = hashlib.sha256(ready_plan_raw).hexdigest()
        max_gate_step = int(
            gate.values["composite_cpu_evidence"]["slurm_step_id"]
        )
        command = build_srun_argv(max_gate_step)
        if (
            command.count(SRUN_AUTHORITY["path"]) != 1
            or command.count("--gpus-per-node=8") != 1
            or command.count("--ntasks=1") != 1
            or command.count("--nodes=1") != 1
            or command.count("--export=NONE") != 1
        ):
            raise GPUControllerError("single GPU srun argv differs")
        # This is the complete, real outer argv.  It contains no plan or
        # payload fixture; those bytes travel through the separately bounded
        # held stdin channel.
        transport_preflight = validate_srun_transport(
            command, max_gate_step=max_gate_step,
        )

        attempt: dict[str, Any] = {
            "schema_version": ATTEMPT_SCHEMA,
            "status": "ATTEMPT_CLAIMED_BEFORE_READY_PLAN_AND_SINGLE_SRUN",
            "holder_job_id": HOLDER_JOB_ID, "node": NODE,
            "campaign_mode": CAMPAIGN, "task_count": 5,
            "task_ids": list(TASK_IDS), "arm_order": list(ARM_ORDER),
            "single_srun_attempt": True, "srun_invocation_count": 1,
            "runner_owns_all_five_arms": True,
            "each_arm_exactly_once": True, "retry_allowed": False,
            "partial_outputs_are_not_results": True,
            "named_hold_payload_executed": False,
            "composite_cpu_admission_step_floor": max_gate_step,
            "package_gate": gate.evidence(),
            "controller": self_authority.row(),
            "srun_authority": srun_authority.row(),
            "ready_plan": {
                "path": str(READY_PLAN_PATH), "sha256": ready_plan_sha,
                "size": len(ready_plan_raw),
                "plan_digest": ready_plan["plan_digest"],
                "exact_semantic_changes": [
                    "hold_reasons", "launch_allowed", "status",
                ],
            },
            "held_exact26_identity_set_digest": object_digest({
                role: authorities[role].row() for role in IDENTITY_ROLES
            }),
            "exact_srun_argv": command,
            "exact_srun_argv_digest": object_digest(command),
            "exact_srun_transport_preflight": transport_preflight,
            "held_stdin_lock_path": str(DISPATCH_PATH),
            "sealed_hold_plan_sha256": SEALED_HOLD_PLAN_SHA256,
            "sealed_hold_plan_size": SEALED_HOLD_PLAN_SIZE,
            "named_hold_payload_sha256": NAMED_HOLD_PAYLOAD_SHA256,
            "named_hold_payload_size": NAMED_HOLD_PAYLOAD_SIZE,
            "runtime_receipt_path": str(RUNTIME_RECEIPT_PATH),
            "output_report_path": str(OUTPUT_REPORT_PATH),
            "runner_attestation_path": str(RUNNER_ATTESTATION_PATH),
        }
        attempt["attempt_digest"] = object_digest(attempt)
        # First mutation.  Every later failure is terminal because this path is
        # immutable/create-only and there is no retry branch.
        attempt_raw = create_immutable_json(
            directories["evidence"], ATTEMPT_PATH, attempt, RECEIPT_MODE,
        )
        attempt_authority = open_authority(
            ATTEMPT_PATH,
            expected_sha256=hashlib.sha256(attempt_raw).hexdigest(),
            expected_size=len(attempt_raw), expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        )
        attempt_authority.replay()

        create_immutable(
            directories["runtime"], READY_PLAN_PATH, ready_plan_raw, FILE_MODE,
        )
        ready_plan_authority = open_authority(
            READY_PLAN_PATH, expected_sha256=ready_plan_sha,
            expected_size=len(ready_plan_raw), expected_mode=FILE_MODE,
            maximum_size=MAX_JSON_SIZE,
        )
        release, payload = build_production_release(
            launch_input, authorities, ready_plan_authority,
            site_packages_authority, torch_package_init_authority,
            directories,
        )
        if b"root_launch_payload_HOLD" in payload or b"exit 88" in payload:
            raise GPUControllerError("production payload overlaps named HOLD payload")
        payload_sha = hashlib.sha256(payload).hexdigest()
        transport = validate_srun_transport(
            command, max_gate_step=max_gate_step,
            payload=payload, release=release,
        )
        dispatch: dict[str, Any] = {
            "schema_version": DISPATCH_SCHEMA,
            "status": "PAYLOAD_LOCKED_BEFORE_SINGLE_SRUN",
            "holder_job_id": HOLDER_JOB_ID, "node": NODE,
            "single_srun_attempt": True, "srun_invocation_count": 1,
            "retry_allowed": False,
            "attempt_sha256": hashlib.sha256(attempt_raw).hexdigest(),
            "attempt_digest": attempt["attempt_digest"],
            "ready_plan_sha256": ready_plan_sha,
            "ready_plan_digest": ready_plan["plan_digest"],
            "production_release_digest": object_digest(release),
            "composite_cpu_admission_step_floor": max_gate_step,
            "exact_srun_argv": command,
            "exact_srun_argv_digest": object_digest(command),
            "transport": transport,
            "named_hold_payload_executed": False,
        }
        dispatch["dispatch_digest"] = object_digest(dispatch)
        dispatch_raw = create_immutable_json(
            directories["evidence"], DISPATCH_PATH, dispatch, RECEIPT_MODE,
        )
        dispatch_authority = open_authority(
            DISPATCH_PATH,
            expected_sha256=hashlib.sha256(dispatch_raw).hexdigest(),
            expected_size=len(dispatch_raw), expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        )
        gate.replay(); ready_plan_authority.replay()
        site_packages_authority.replay()
        torch_package_init_authority.replay()
        attempt_authority.replay(); dispatch_authority.replay()
        for authority in authorities.values():
            authority.replay()
        self_authority.replay(); srun_authority.replay()

        returncode, stdout_raw, stderr_raw, process_group = run_single_srun(
            command, payload, release, max_gate_step,
        )
        create_immutable(
            directories["logs"], STDOUT_PATH, stdout_raw, RECEIPT_MODE,
        )
        create_immutable(
            directories["logs"], STDERR_PATH, stderr_raw, RECEIPT_MODE,
        )
        stdout_authority = open_authority(
            STDOUT_PATH, expected_sha256=hashlib.sha256(stdout_raw).hexdigest(),
            expected_size=len(stdout_raw), expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        )
        stderr_authority = open_authority(
            STDERR_PATH, expected_sha256=hashlib.sha256(stderr_raw).hexdigest(),
            expected_size=len(stderr_raw), expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        )
        gate.replay(); ready_plan_authority.replay()
        site_packages_authority.replay()
        torch_package_init_authority.replay()
        attempt_authority.replay(); dispatch_authority.replay()
        for authority in authorities.values():
            authority.replay()
        self_authority.replay(); srun_authority.replay()
        stdout_authority.replay(); stderr_authority.replay()
        for directory in directories.values():
            directory.replay()
        if returncode != 0:
            raise GPUControllerError(f"single GPU srun failed: {returncode}")
        postflight_closure = validate_postflight(
            ready_plan=ready_plan, ready_plan_raw=ready_plan_raw,
            stdout_raw=stdout_raw, stderr_raw=stderr_raw,
            release=release, dispatch=dispatch,
            target_directories=directories,
        )
        postflight_closure.replay()

        evidence: dict[str, Any] = {
            "schema_version": EVIDENCE_SCHEMA,
            "status": "PASS_EXACT_FIVE_AWAITING_MANUAL_REVIEW",
            "holder_job_id": HOLDER_JOB_ID, "node": NODE,
            "campaign_mode": CAMPAIGN, "task_count": 5,
            "task_ids": list(TASK_IDS), "arm_order": list(ARM_ORDER),
            "single_srun_attempt": True, "srun_invocation_count": 1,
            "runner_owns_all_five_arms": True,
            "all_five_arms_attempted_exactly_once": True,
            "all_five_arms_succeeded": True, "retry_allowed": False,
            "srun_returncode": returncode,
            "local_srun_process_group": process_group,
            "local_srun_process_group_zero": True,
            "compute_rank_processes_zero": True,
            "attempt_sha256": hashlib.sha256(attempt_raw).hexdigest(),
            "attempt_digest": attempt["attempt_digest"],
            "dispatch_sha256": hashlib.sha256(dispatch_raw).hexdigest(),
            "dispatch_digest": dispatch["dispatch_digest"],
            "ready_plan_sha256": ready_plan_sha,
            "ready_plan_digest": ready_plan["plan_digest"],
            "production_release_digest": object_digest(release),
            "composite_cpu_admission_step_floor": max_gate_step,
            "in_memory_payload_sha256": payload_sha,
            "in_memory_payload_size": len(payload),
            "srun_transport": transport,
            "named_hold_payload_executed": False,
            "torch_version": EXPECTED_TORCH_VERSION,
            "hip_version": EXPECTED_HIP_VERSION,
            "gpu_count": GPU_COUNT, "gpu_name": EXPECTED_GPU_NAME,
            "stdout_sha256": hashlib.sha256(stdout_raw).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr_raw).hexdigest(),
            "stderr_empty": True,
            "rank_cache_compute_state":
            "RETAINED_COMPUTE_LOCAL_POSTFLIGHT_AUTHORITY",
            "rank_cache_absent_claimed": False,
            "package_gate": gate.evidence(),
            "postflight": postflight_closure.evidence,
            "manual_review_required": True,
            "engineering_oracle_only": True,
            "scientific_claim_authorized": False,
            "formal_claim_authorized": False,
        }
        evidence["evidence_digest"] = object_digest(evidence)
        evidence_names_before = {
            ATTEMPT_PATH.name, DISPATCH_PATH.name,
        }
        _exact_names(
            PACKAGE_ROOT / "evidence", evidence_names_before,
            label="pre-controller-evidence closure",
        )
        gate.replay(); ready_plan_authority.replay()
        site_packages_authority.replay()
        torch_package_init_authority.replay()
        attempt_authority.replay(); dispatch_authority.replay()
        self_authority.replay(); srun_authority.replay()
        stdout_authority.replay(); stderr_authority.replay()
        postflight_closure.replay()
        for directory in directories.values():
            directory.replay()
        for authority in authorities.values():
            authority.replay()
        evidence_raw = create_immutable_json(
            directories["evidence"], EVIDENCE_PATH, evidence, RECEIPT_MODE,
        )
        evidence_held = open_authority(
            EVIDENCE_PATH, expected_sha256=hashlib.sha256(evidence_raw).hexdigest(),
            expected_size=len(evidence_raw), expected_mode=RECEIPT_MODE,
            maximum_size=MAX_JSON_SIZE,
        )
        try:
            _exact_names(
                PACKAGE_ROOT / "evidence",
                evidence_names_before | {EVIDENCE_PATH.name},
                label="final controller evidence closure",
            )
            gate.replay(); ready_plan_authority.replay(); evidence_held.replay()
            site_packages_authority.replay()
            torch_package_init_authority.replay()
            attempt_authority.replay(); dispatch_authority.replay()
            self_authority.replay(); srun_authority.replay()
            stdout_authority.replay(); stderr_authority.replay()
            postflight_closure.replay()
            for directory in directories.values():
                directory.replay()
            for authority in authorities.values():
                authority.replay()
        finally:
            evidence_held.close()
        return evidence
    finally:
        for authority in authorities.values():
            authority.close()
        if attempt_authority is not None:
            attempt_authority.close()
        if dispatch_authority is not None:
            dispatch_authority.close()
        if ready_plan_authority is not None:
            ready_plan_authority.close()
        if site_packages_authority is not None:
            site_packages_authority.close()
        if torch_package_init_authority is not None:
            torch_package_init_authority.close()
        if stdout_authority is not None:
            stdout_authority.close()
        if stderr_authority is not None:
            stderr_authority.close()
        if postflight_closure is not None:
            postflight_closure.close()
        if self_authority is not None:
            self_authority.close()
        if srun_authority is not None:
            srun_authority.close()
        for directory in directories.values():
            directory.close()
        if gate is not None:
            gate.close()


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    # HOLD-before-I/O: state and placeholders are inspected without touching
    # any filesystem path, subprocess, network, Slurm, or output target.
    if CONTROLLER_STATE != READY_STATE:
        print("HOLD: GPU controller state is not READY", file=sys.stderr)
        return 88
    blocked = blocked_dynamic_pins()
    if blocked:
        print("HOLD: blocked dynamic pins: " + ",".join(blocked), file=sys.stderr)
        return 88
    if values != ["--execute", authorization_token()]:
        print("HOLD: exact one-shot authorization token required", file=sys.stderr)
        return 88
    try:
        result = controller()
    except (OSError, ValueError, GPUControllerError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 96
    print(canonical(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
