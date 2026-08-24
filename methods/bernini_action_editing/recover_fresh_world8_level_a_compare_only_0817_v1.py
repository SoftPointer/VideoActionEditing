#!/usr/bin/env python3
"""Fail-closed, GPU-free recovery of the frozen Level-A A/B parity receipt.

The original ``fresh-world8-level-a-r2-p2-launchbound-v2`` campaign completed
two independent WORLD8 Slurm steps, then deliberately failed closed because its
final comparator compared NFS-server mtimes with Slurm-controller timestamps as
if they shared one clock.  This module does not reopen or modify that campaign.
It authenticates the exact old release, launch authority, evidence files, and
terminal sacct rows against a compiled trust root, applies an explicit bounded
cross-clock policy, and can create exactly one recovery receipt in a fresh root.

It imports no old driver code, starts no subprocess, and has no GPU/Slurm control
surface.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any, Mapping, NoReturn, Sequence


SCHEMA_VERSION = (
    "bernini-action-edit-fresh-world8-level-a-compare-only-recovery-receipt-v1"
)
METHOD = "bernini-action-edit-fresh-world8-level-a-compare-only-recovery-0817-v1"
AUTHORITY = "PRE_D0_ENGINEERING_ONLY"
AUTHORITY_SCHEMA = (
    "bernini-action-edit-fresh-world8-level-a-compare-only-recovery-authority-v1"
)
AUTHORITY_STATUS = "FROZEN_COMPARE_ONLY_RECOVERY_AUTHORITY"
AUTHORITY_FILENAME = "fresh_world8_level_a_compare_only_recovery_v1_AUTHORITY.json"
AUTHORITY_SHA256 = "d8ca84e783ca7127cec09c14d6f46ef1427bd37292948fccc5c94bd0429c665e"
OUTPUT_FILENAME = "fresh_world8_a_b.compare_only_recovery_receipt"

OLD_CAMPAIGN_TAG = "fresh-world8-level-a-r2-p2-launchbound-v2"
OLD_METHOD = "bernini-action-edit-fresh-world8-level-a-driver-0817-v1"
OLD_AUTHORITY = "PRE_D0_ENGINEERING_ONLY"
PARENT_JOB_ID = 140846
PINNED_NODE = "auh7-1b-gpu-279"
ATTEMPT_STEPS = {"A": "140846.367", "B": "140846.368"}
ATTEMPT_JOB_NAMES = {
    "A": "bernini0817-level-a-launchbound-v2-A",
    "B": "bernini0817-level-a-launchbound-v2-B",
}
EXPECTED_PARENT_STATE = "RUNNING|auh7-1b-gpu-[246-248,279]|gres/gpu:mi210:8"
EXPECTED_COMPARE_FAILURE = (
    "Level-A driver refused: terminal sacct validation timed out: terminal "
    "sacct authority differs or predates attempt claim\n"
)
MAX_CLOCK_SKEW_SECONDS = 60.0

POST_CONSENSUS_ONLY_KEYS = {
    "world8_consensus",
    "world8_consumer_complete",
    "fresh_world8_process_forward_exact_consensus_verified",
    "fresh_world8_process_forward_scope",
    "full_bernini_renderer_forward_executed",
    "checkpoint_bytes_conditioner_exact30_fresh_consumer_go",
}
FALSE_RECEIPT_CLAIMS = {
    "promotable",
    "promotion_authorized",
    "formal_training_started",
    "counts_as_d0",
    "scientific_claim_authorized",
    "action_quality_claim_authorized",
    "full_bernini_renderer_forward_executed",
    "offline_product_inference_completed",
    "full40_denoise_executed",
    "mp4_emitted",
    "training_to_fresh_forward_parity_verified",
    "conditioner_cell_training_to_fresh_forward_parity_verified",
    "full_bernini_renderer_training_to_fresh_forward_parity_verified",
    "fresh_a_b_parity_verified",
    "training_gradient_checkpoint_hooks_installed",
}
SACCT_KEYS = {
    "JobIDRaw",
    "JobName",
    "State",
    "ExitCode",
    "NodeList",
    "NNodes",
    "NTasks",
    "AllocTRES",
    "Start",
    "End",
}
EXPECTED_TRES = "cpu=32,gres/gpu:mi210=8,gres/gpu=8,mem=60G,node=1"


class RecoveryError(RuntimeError):
    """Raised when any old authority or recovery invariant differs."""


def fail(message: str) -> NoReturn:
    raise RecoveryError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def object_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _reject_constant(value: str) -> NoReturn:
    fail(f"non-finite JSON constant is forbidden: {value}")


def _object_pairs(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def strict_json_bytes(
    payload: bytes,
    *,
    label: str,
    allow_one_trailing_newline: bool = False,
) -> Any:
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryError(f"{label} is not strict UTF-8 JSON") from error
    rendered = canonical_json_bytes(value)
    allowed = {rendered}
    if allow_one_trailing_newline:
        allowed.add(rendered + b"\n")
    if payload not in allowed:
        fail(f"{label} is not canonical JSON")
    return value


def _absolute_plain_path(value: str | Path, *, directory: bool, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        fail(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=True)
        info = path.lstat()
    except OSError as error:
        raise RecoveryError(f"{label} is absent") from error
    if resolved != path or stat.S_ISLNK(info.st_mode):
        fail(f"{label} is symlinked or non-canonical")
    if directory and not stat.S_ISDIR(info.st_mode):
        fail(f"{label} is not a directory")
    if not directory and not stat.S_ISREG(info.st_mode):
        fail(f"{label} is not a regular file")
    return path


def _plain_file(value: str | Path, *, label: str) -> tuple[Path, bytes, os.stat_result]:
    path = _absolute_plain_path(value, directory=False, label=label)
    info = path.lstat()
    if info.st_nlink != 1:
        fail(f"{label} link count differs")
    payload = path.read_bytes()
    after = path.lstat()
    if (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_mode,
        info.st_nlink,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
        after.st_nlink,
    ):
        fail(f"{label} changed while it was read")
    return path, payload, info


def _exact_keys(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        fail(f"{label} key closure differs")
    return value


def _self_digest(value: Mapping[str, Any], key: str, *, label: str) -> str:
    claimed = value.get(key)
    if not isinstance(claimed, str) or len(claimed) != 64:
        fail(f"{label} {key} is absent")
    unsigned = dict(value)
    del unsigned[key]
    if object_sha256(unsigned) != claimed:
        fail(f"{label} {key} does not recompute")
    return claimed


def load_frozen_authority() -> tuple[Mapping[str, Any], str]:
    path = Path(__file__).resolve().parent / "audits" / AUTHORITY_FILENAME
    _, payload, _ = _plain_file(path, label="compiled recovery authority")
    observed_sha = sha256_bytes(payload)
    if observed_sha != AUTHORITY_SHA256:
        fail("compiled recovery authority SHA differs")
    value = strict_json_bytes(
        payload,
        label="compiled recovery authority",
        allow_one_trailing_newline=True,
    )
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != AUTHORITY_SCHEMA
        or value.get("method") != METHOD
        or value.get("status") != AUTHORITY_STATUS
        or value.get("clock_domain_policy", {}).get(
            "maximum_absolute_start_claim_skew_seconds"
        ) != 60
        or value.get("clock_domain_policy", {}).get("recovery_reason")
        != "cross_clock_domain"
    ):
        fail("compiled recovery authority content differs")
    return value, observed_sha


def validate_old_trust_roots(
    release_manifest_path: str | Path,
    launch_authority_path: str | Path,
    deployment_pins_path: str | Path,
    *,
    authority: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    trust = authority["old_trust_roots"]
    _, release_payload, _ = _plain_file(
        release_manifest_path, label="old frozen release manifest"
    )
    _, core_payload, _ = _plain_file(
        launch_authority_path, label="old frozen launch authority core"
    )
    _, pins_payload, _ = _plain_file(
        deployment_pins_path, label="old frozen deployment pins"
    )
    if sha256_bytes(release_payload) != trust["release_manifest_sha256"]:
        fail("old frozen release manifest SHA differs")
    if sha256_bytes(core_payload) != trust["launch_authority_core_sha256"]:
        fail("old frozen launch authority core SHA differs")
    if sha256_bytes(pins_payload) != trust["deployment_pins_sha256"]:
        fail("old frozen deployment pins SHA differs")
    release = strict_json_bytes(
        release_payload,
        label="old frozen release manifest",
        allow_one_trailing_newline=True,
    )
    core = strict_json_bytes(
        core_payload,
        label="old frozen launch authority core",
        allow_one_trailing_newline=True,
    )
    pins = strict_json_bytes(
        pins_payload,
        label="old frozen deployment pins",
        allow_one_trailing_newline=True,
    )
    members = {row.get("path"): row for row in release.get("files", [])}
    required_member_hashes = {
        "action_edit_checkpoint_consumer_0817_v1.py": trust["consumer_sha256"],
        "action_edit_fresh_world8_level_a_driver_0817_v1.py": trust["driver_sha256"],
        "infer_action_edit_product_abi_0817_v1.py": trust["product_bridge_sha256"],
    }
    if (
        release.get("schema_version")
        != "bernini-action-edit-fresh-world8-level-a-release-manifest-v1"
        or release.get("member_root") != "methods/bernini_action_editing"
        or len(members) != 11
        or any(
            not isinstance(members.get(name), Mapping)
            or members[name].get("sha256") != digest
            or members[name].get("mode") != 0o444
            for name, digest in required_member_hashes.items()
        )
    ):
        fail("old frozen release manifest content differs")
    if (
        core.get("schema_version")
        != "bernini-action-edit-fresh-world8-level-a-launch-authority-core-v2"
        or core.get("status") != "FROZEN_ONE_SHOT_LAUNCH_AUTHORITY"
        or core.get("method") != OLD_METHOD
        or core.get("authority") != OLD_AUTHORITY
        or core.get("parent_allocation")
        != {"control_authorized": False, "job_id": PARENT_JOB_ID, "node": PINNED_NODE}
        or core.get("topology", {}).get("attempt_order") != ["A", "B"]
        or core.get("topology", {}).get("world_size") != 8
        or core.get("topology", {}).get("dp_size") != 2
        or core.get("topology", {}).get("sp_size") != 4
        or core.get("topology", {}).get("max_restarts") != 0
        or core.get("release", {}).get("manifest_sha256")
        != trust["release_manifest_sha256"]
        or core.get("release", {}).get("driver_sha256") != trust["driver_sha256"]
        or core.get("release", {}).get("consumer_sha256") != trust["consumer_sha256"]
        or core.get("release", {}).get("product_bridge_sha256")
        != trust["product_bridge_sha256"]
        or core.get("launcher_hash_chain", {}).get("step_payload_sha256")
        != trust["step_payload_sha256"]
        or core.get("launcher_hash_chain", {}).get("rank_exec_sha256")
        != trust["rank_exec_sha256"]
        or core.get("checkpoint")
        != {"parameter_sha256": trust["checkpoint_parameter_sha256"], "step": 2}
    ):
        fail("old frozen launch authority core content differs")
    for label in ("A", "B"):
        expected_intent = trust[f"intent_{label.lower()}_sha256"]
        if (
            core.get("attempts", {}).get(label, {}).get("intent_sha256")
            != expected_intent
            or core.get("attempts", {}).get(label, {}).get("job_name")
            != ATTEMPT_JOB_NAMES[label]
        ):
            fail(f"old launch authority attempt {label} differs")
    launchers = {
        row.get("path"): row
        for row in pins.get("launchers", [])
        if isinstance(row, Mapping)
    }
    if (
        pins.get("schema_version")
        != "bernini-action-edit-fresh-world8-level-a-deployment-pins-v2"
        or pins.get("tag") != OLD_CAMPAIGN_TAG
        or pins.get("authority") != OLD_AUTHORITY
        or pins.get("launch_authority_core", {}).get("sha256")
        != trust["launch_authority_core_sha256"]
        or pins.get("release", {}).get("manifest_sha256")
        != trust["release_manifest_sha256"]
        or pins.get("release", {}).get("driver_sha256") != trust["driver_sha256"]
        or pins.get("release", {}).get("consumer_sha256") != trust["consumer_sha256"]
        or pins.get("release", {}).get("product_bridge_sha256")
        != trust["product_bridge_sha256"]
        or launchers.get("auh_launch_fresh_world8_level_a_r2_p2_node279_job140846_v1.sh", {}).get("sha256")
        != trust["controller_sha256"]
        or launchers.get("auh_fresh_world8_level_a_r2_p2_node279_step_v1.sh", {}).get("sha256")
        != trust["step_payload_sha256"]
        or launchers.get("auh_fresh_world8_level_a_r2_p2_node279_rank_exec_v1.sh", {}).get("sha256")
        != trust["rank_exec_sha256"]
        or pins.get("hash_chain", {}).get("outer_pins_bind_controller") is not True
        or pins.get("parent_allocation")
        != {"control_authorized": False, "job_id": PARENT_JOB_ID, "node": PINNED_NODE}
    ):
        fail("old frozen deployment pins content differs")
    return release, core


@dataclass(frozen=True)
class AttemptEvidence:
    label: str
    intent: Mapping[str, Any]
    receipt: Mapping[str, Any]
    validation: Mapping[str, Any]
    controller: Mapping[str, Any]
    terminal: Mapping[str, Any]
    success: Mapping[str, Any]
    file_sha256: Mapping[str, str]


def validate_evidence_files(
    evidence_root_value: str | Path,
    *,
    authority: Mapping[str, Any],
) -> tuple[Path, Mapping[str, tuple[bytes, os.stat_result]]]:
    root = _absolute_plain_path(
        evidence_root_value, directory=True, label="old evidence root"
    )
    expected_rows = {row["path"]: row for row in authority["evidence_files"]}
    observed_files: set[str] = set()
    observed_dirs: set[str] = set()
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        current_info = current_path.lstat()
        if stat.S_ISLNK(current_info.st_mode) or not stat.S_ISDIR(current_info.st_mode):
            fail("old evidence tree contains a non-plain directory")
        for name in directory_names:
            path = current_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                fail("old evidence tree contains a symlinked directory")
            observed_dirs.add(str(path.relative_to(root)))
        for name in file_names:
            path = current_path / name
            observed_files.add(str(path.relative_to(root)))
    expected_dirs = {"A", "A/STARTED", "B", "B/STARTED", "run_A", "run_B"}
    if observed_dirs != expected_dirs or observed_files != set(expected_rows):
        fail("old evidence root closure differs")
    loaded: dict[str, tuple[bytes, os.stat_result]] = {}
    for relative, expected in expected_rows.items():
        _, payload, info = _plain_file(root / relative, label=f"old evidence {relative}")
        if (
            stat.S_IMODE(info.st_mode) != expected["mode"]
            or info.st_size != expected["size"]
            or sha256_bytes(payload) != expected["sha256"]
        ):
            fail(f"old evidence identity differs: {relative}")
        loaded[relative] = (payload, info)
    if loaded["compare.log"][0] != EXPECTED_COMPARE_FAILURE.encode("utf-8"):
        fail("old fail-closed compare log differs")
    return root, loaded


def _load_evidence_json(
    loaded: Mapping[str, tuple[bytes, os.stat_result]], relative: str
) -> Mapping[str, Any]:
    value = strict_json_bytes(loaded[relative][0], label=f"old evidence {relative}")
    if not isinstance(value, Mapping):
        fail(f"old evidence {relative} is not an object")
    return value


def _validate_intent(
    label: str,
    intent: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    core: Mapping[str, Any],
) -> None:
    trust = authority["old_trust_roots"]
    expected_sha = trust[f"intent_{label.lower()}_sha256"]
    expected = core["attempts"][label]
    if (
        intent.get("schema_version")
        != "bernini-action-edit-fresh-world8-level-a-attempt-intent-v2"
        or intent.get("method") != OLD_METHOD
        or intent.get("authority") != OLD_AUTHORITY
        or intent.get("attempt") != label
        or intent.get("parent_job_id") != PARENT_JOB_ID
        or intent.get("node") != PINNED_NODE
        or intent.get("job_name") != ATTEMPT_JOB_NAMES[label]
        or intent.get("attempt_root") != expected["attempt_root"]
        or intent.get("output_root") != expected["output_root"]
        or intent.get("checkpoint_step") != 2
        or intent.get("world_size") != 8
        or intent.get("dp_size") != 2
        or intent.get("sp_size") != 4
        or intent.get("release_manifest_sha256") != trust["release_manifest_sha256"]
        or intent.get("driver_sha256") != trust["driver_sha256"]
        or intent.get("consumer_sha256") != trust["consumer_sha256"]
        or intent.get("product_bridge_sha256") != trust["product_bridge_sha256"]
        or intent.get("step_payload_sha256") != trust["step_payload_sha256"]
        or intent.get("rank_exec_sha256") != trust["rank_exec_sha256"]
        or intent.get("automatic_relaunch_authorized") is not False
        or intent.get("parent_control_authorized") is not False
        or object_sha256(intent) != expected_sha
    ):
        fail(f"attempt {label} intent content differs")


def _validate_receipt(
    label: str,
    receipt: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> tuple[str, str, list[str]]:
    receipt_digest = _self_digest(receipt, "receipt_digest", label=f"attempt {label} receipt")
    trust = authority["old_trust_roots"]
    binding = receipt.get("launch_binding")
    if not isinstance(binding, Mapping):
        fail(f"attempt {label} receipt launch binding is absent")
    intent_sha = trust[f"intent_{label.lower()}_sha256"]
    expected_intent_mtime = authority["nfs_mtime_ns"][label]["intent"]
    if (
        binding.get("schema_version")
        != "bernini-action-edit-fresh-world8-level-a-launch-binding-v2"
        or binding.get("attempt") != label
        or binding.get("attempt_intent_sha256") != intent_sha
        or binding.get("attempt_claim_mtime_ns") != expected_intent_mtime
        or binding.get("launch_authority_core_sha256")
        != trust["launch_authority_core_sha256"]
        or binding.get("parent_job_id") != PARENT_JOB_ID
        or binding.get("slurm_numeric_step") != ATTEMPT_STEPS[label]
        or binding.get("node") != PINNED_NODE
        or binding.get("job_name") != ATTEMPT_JOB_NAMES[label]
        or binding.get("release_manifest_sha256") != trust["release_manifest_sha256"]
        or binding.get("driver_source_sha256") != trust["driver_sha256"]
        or binding.get("launcher_hash_chain")
        != {
            "rank_exec_sha256": trust["rank_exec_sha256"],
            "step_payload_sha256": trust["step_payload_sha256"],
        }
    ):
        fail(f"attempt {label} receipt launch binding differs")
    if (
        receipt.get("authority") != OLD_AUTHORITY
        or receipt.get("complete") is not True
        or receipt.get("checkpoint_step") != 2
        or receipt.get("checkpoint_parameter_sha256")
        != trust["checkpoint_parameter_sha256"]
        or receipt.get("loaded_parameter_sha256") != trust["checkpoint_parameter_sha256"]
        or receipt.get("consumer_source_sha256") != trust["consumer_sha256"]
        or receipt.get("product_bridge_source_sha256") != trust["product_bridge_sha256"]
        or receipt.get("world8_consumer_complete") is not True
        or receipt.get("fresh_world8_process_forward_exact_consensus_verified")
        is not True
        or receipt.get("fresh_world8_process_forward_scope")
        != "conditioner_predictor_plus_exact30_cell_only_not_bernini_renderer"
        or receipt.get("checkpoint_bytes_conditioner_exact30_fresh_consumer_go")
        is not True
        or any(receipt.get(key) is not False for key in FALSE_RECEIPT_CLAIMS)
    ):
        fail(f"attempt {label} receipt claims differ")
    consensus = receipt.get("world8_consensus")
    if not isinstance(consensus, Mapping):
        fail(f"attempt {label} WORLD8 consensus is absent")
    sessions = consensus.get("rank_local_fresh_process_sessions")
    if (
        consensus.get("world_size") != 8
        or consensus.get("rank_order") != list(range(8))
        or consensus.get("all8_exact_consensus") is not True
        or consensus.get("eight_distinct_fresh_process_sessions") is not True
        or not isinstance(sessions, list)
        or len(sessions) != 8
        or len(set(sessions)) != 8
        or any(not isinstance(item, str) or len(item) != 64 for item in sessions)
        or receipt.get("fresh_process_session_id") not in sessions
    ):
        fail(f"attempt {label} WORLD8 session evidence differs")
    raw = dict(receipt)
    del raw["receipt_digest"]
    del raw["launch_binding"]
    common = dict(raw)
    for key in POST_CONSENSUS_ONLY_KEYS:
        common.pop(key, None)
    common.pop("fresh_process_session_id", None)
    consensus_digest = consensus.get("consumer_receipt_sha256")
    if not isinstance(consensus_digest, str) or object_sha256(common) != consensus_digest:
        fail(f"attempt {label} WORLD8 consensus digest does not recompute")
    if binding["attempt_intent_sha256"] != object_sha256(intent):
        fail(f"attempt {label} receipt does not bind its intent bytes")
    return receipt_digest, consensus_digest, list(sessions)


def _validate_validation(
    label: str,
    validation: Mapping[str, Any],
    *,
    receipt_file_sha: str,
    receipt_digest: str,
    consensus_digest: str,
    authority: Mapping[str, Any],
) -> str:
    digest = _self_digest(
        validation, "validation_digest", label=f"attempt {label} validation"
    )
    trust = authority["old_trust_roots"]
    if (
        validation.get("schema_version")
        != "bernini-action-edit-fresh-world8-level-a-receipt-validation-v2"
        or validation.get("method") != OLD_METHOD
        or validation.get("authority") != OLD_AUTHORITY
        or validation.get("attempt") != label
        or validation.get("attempt_intent_sha256")
        != trust[f"intent_{label.lower()}_sha256"]
        or validation.get("launch_authority_core_sha256")
        != trust["launch_authority_core_sha256"]
        or validation.get("receipt_file_sha256") != receipt_file_sha
        or validation.get("receipt_digest") != receipt_digest
        or validation.get("world8_consensus_sha256") != consensus_digest
        or validation.get("slurm_numeric_step") != ATTEMPT_STEPS[label]
        or validation.get("node") != PINNED_NODE
        or validation.get("job_name") != ATTEMPT_JOB_NAMES[label]
        or validation.get("receipt_validated") is not True
        or validation.get("full_bernini_renderer_forward_executed") is not False
        or validation.get("offline_product_inference_completed") is not False
        or validation.get("full40_denoise_executed") is not False
        or validation.get("mp4_emitted") is not False
        or validation.get("promotion_authorized") is not False
    ):
        fail(f"attempt {label} receipt validation differs")
    return digest


def _validate_controller(
    label: str,
    controller: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
) -> str:
    digest = _self_digest(
        controller, "status_digest", label=f"attempt {label} controller status"
    )
    trust = authority["old_trust_roots"]
    if (
        controller.get("schema_version")
        != "bernini-action-edit-fresh-world8-level-a-controller-status-v2"
        or controller.get("method") != OLD_METHOD
        or controller.get("authority") != OLD_AUTHORITY
        or controller.get("attempt") != label
        or controller.get("attempt_intent_sha256")
        != trust[f"intent_{label.lower()}_sha256"]
        or controller.get("launch_authority_core_sha256")
        != trust["launch_authority_core_sha256"]
        or controller.get("parent_job_id") != PARENT_JOB_ID
        or controller.get("node") != PINNED_NODE
        or controller.get("child_exit") != 0
        or controller.get("attempt_claimed") is not True
        or controller.get("parent_state_before") != EXPECTED_PARENT_STATE
        or controller.get("parent_state_after") != EXPECTED_PARENT_STATE
        or controller.get("parent_cancelled") is not False
        or controller.get("parent_released") is not False
        or controller.get("parent_requeued") is not False
        or controller.get("parent_signalled") is not False
        or controller.get("automatic_relaunch_authorized") is not False
        or controller.get("receipt_validated") is not False
    ):
        fail(f"attempt {label} controller status differs")
    return digest


def _validate_terminal_and_success(
    label: str,
    terminal: Mapping[str, Any],
    success: Mapping[str, Any],
    *,
    receipt_file_sha: str,
    receipt_digest: str,
    consensus_digest: str,
    validation_file_sha: str,
    controller_file_sha: str,
    terminal_file_sha: str,
    authority: Mapping[str, Any],
) -> None:
    _self_digest(terminal, "terminal_digest", label=f"attempt {label} terminal")
    _self_digest(success, "success_digest", label=f"attempt {label} SUCCESS")
    trust = authority["old_trust_roots"]
    common = {
        "attempt": label,
        "attempt_intent_sha256": trust[f"intent_{label.lower()}_sha256"],
        "launch_authority_core_sha256": trust["launch_authority_core_sha256"],
        "receipt_file_sha256": receipt_file_sha,
        "receipt_digest": receipt_digest,
        "world8_consensus_sha256": consensus_digest,
        "slurm_numeric_step": ATTEMPT_STEPS[label],
        "node": PINNED_NODE,
        "job_name": ATTEMPT_JOB_NAMES[label],
    }
    if (
        terminal.get("schema_version")
        != "bernini-action-edit-fresh-world8-level-a-terminal-authority-v2"
        or terminal.get("method") != OLD_METHOD
        or terminal.get("authority") != OLD_AUTHORITY
        or terminal.get("status") != "SUCCESS"
        or any(terminal.get(key) != value for key, value in common.items())
        or terminal.get("controller_status_sha256") != controller_file_sha
        or terminal.get("receipt_validation_sha256") != validation_file_sha
        or terminal.get("consecutive_full_validations") != 2
        or terminal.get("receipt_validated") is not True
        or terminal.get("parent_untouched") is not True
        or terminal.get("automatic_relaunch_authorized") is not False
        or terminal.get("promotion_authorized") is not False
    ):
        fail(f"attempt {label} terminal authority differs")
    if (
        success.get("schema_version")
        != "bernini-action-edit-fresh-world8-level-a-terminal-success-v2"
        or success.get("method") != OLD_METHOD
        or success.get("authority") != OLD_AUTHORITY
        or success.get("status") != "SUCCESS"
        or any(success.get(key) != value for key, value in common.items())
        or success.get("terminal_authority_sha256") != terminal_file_sha
        or success.get("parent_untouched") is not True
        or success.get("automatic_relaunch_authorized") is not False
        or success.get("promotion_authorized") is not False
    ):
        fail(f"attempt {label} SUCCESS authority differs")


def load_and_validate_attempts(
    loaded: Mapping[str, tuple[bytes, os.stat_result]],
    *,
    authority: Mapping[str, Any],
    core: Mapping[str, Any],
) -> Mapping[str, AttemptEvidence]:
    result: dict[str, AttemptEvidence] = {}
    expected_files = {row["path"]: row for row in authority["evidence_files"]}
    for label in ("A", "B"):
        receipt_relative = f"run_{label}/bundle.consumer_receipt"
        paths = {
            "intent": f"{label}/STARTED/intent.json",
            "validation1": f"{label}/STARTED/receipt-validation-1.json",
            "validation2": f"{label}/STARTED/receipt-validation-2.json",
            "controller": f"{label}/controller.status.json",
            "terminal": f"{label}/terminal.authority.json",
            "success": f"{label}/SUCCESS",
            "receipt": receipt_relative,
        }
        values = {name: _load_evidence_json(loaded, path) for name, path in paths.items()}
        _validate_intent(label, values["intent"], authority=authority, core=core)
        receipt_digest, consensus_digest, _ = _validate_receipt(
            label,
            values["receipt"],
            authority=authority,
            intent=values["intent"],
        )
        receipt_file_sha = expected_files[receipt_relative]["sha256"]
        validation_digest_1 = _validate_validation(
            label,
            values["validation1"],
            receipt_file_sha=receipt_file_sha,
            receipt_digest=receipt_digest,
            consensus_digest=consensus_digest,
            authority=authority,
        )
        validation_digest_2 = _validate_validation(
            label,
            values["validation2"],
            receipt_file_sha=receipt_file_sha,
            receipt_digest=receipt_digest,
            consensus_digest=consensus_digest,
            authority=authority,
        )
        validation_sha_1 = expected_files[paths["validation1"]]["sha256"]
        validation_sha_2 = expected_files[paths["validation2"]]["sha256"]
        if (
            values["validation1"] != values["validation2"]
            or validation_digest_1 != validation_digest_2
            or validation_sha_1 != validation_sha_2
        ):
            fail(f"attempt {label} consecutive full validations differ")
        _validate_controller(label, values["controller"], authority=authority)
        _validate_terminal_and_success(
            label,
            values["terminal"],
            values["success"],
            receipt_file_sha=receipt_file_sha,
            receipt_digest=receipt_digest,
            consensus_digest=consensus_digest,
            validation_file_sha=validation_sha_1,
            controller_file_sha=expected_files[paths["controller"]]["sha256"],
            terminal_file_sha=expected_files[paths["terminal"]]["sha256"],
            authority=authority,
        )
        result[label] = AttemptEvidence(
            label=label,
            intent=values["intent"],
            receipt=values["receipt"],
            validation=values["validation1"],
            controller=values["controller"],
            terminal=values["terminal"],
            success=values["success"],
            file_sha256={name: expected_files[path]["sha256"] for name, path in paths.items()},
        )
    return result


def allowed_parity_projection(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    projected = json.loads(canonical_json_bytes(receipt).decode("utf-8"))
    for key in ("fresh_process_session_id", "launch_binding", "receipt_digest"):
        if key not in projected:
            fail(f"receipt parity projection key is absent: {key}")
        del projected[key]
    consensus = projected.get("world8_consensus")
    if not isinstance(consensus, dict) or "rank_local_fresh_process_sessions" not in consensus:
        fail("receipt parity projection sessions are absent")
    del consensus["rank_local_fresh_process_sessions"]
    return projected


def validate_cross_attempt_parity(
    attempts: Mapping[str, AttemptEvidence],
) -> Mapping[str, Any]:
    if set(attempts) != {"A", "B"}:
        fail("A/B attempt closure differs")
    a = attempts["A"].receipt
    b = attempts["B"].receipt
    projection_a = allowed_parity_projection(a)
    projection_b = allowed_parity_projection(b)
    if canonical_json_bytes(projection_a) != canonical_json_bytes(projection_b):
        fail("A/B allowed receipt projection differs")
    sessions_a = a["world8_consensus"]["rank_local_fresh_process_sessions"]
    sessions_b = b["world8_consensus"]["rank_local_fresh_process_sessions"]
    if len(set(sessions_a + sessions_b)) != 16:
        fail("A/B fresh process sessions are not exactly 16 distinct sessions")
    if (
        a["world8_consensus"]["consumer_receipt_sha256"]
        != b["world8_consensus"]["consumer_receipt_sha256"]
        or a["launch_binding"]["slurm_numeric_step"]
        == b["launch_binding"]["slurm_numeric_step"]
    ):
        fail("A/B consensus or distinct step evidence differs")
    return {
        "exact_parity": True,
        "allowed_projection_sha256": object_sha256(projection_a),
        "world8_consensus_sha256": a["world8_consensus"]["consumer_receipt_sha256"],
        "fresh_fixed_forward_tensor_set_sha256": a[
            "fresh_loaded_fixed_forward_fingerprint"
        ]["tensor_set_sha256"],
        "world8_launches": 2,
        "distinct_fresh_process_sessions": 16,
    }


def validate_sacct_rows(
    rows: Any,
    *,
    authority: Mapping[str, Any],
) -> list[Mapping[str, str]]:
    expected = authority["sacct"]["rows"]
    if not isinstance(rows, list) or len(rows) != 2 or rows != expected:
        fail("terminal sacct rows differ from frozen external observation")
    result: list[Mapping[str, str]] = []
    for label, row in zip(("A", "B"), rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != SACCT_KEYS
            or row.get("JobIDRaw") != ATTEMPT_STEPS[label]
            or row.get("JobName") != ATTEMPT_JOB_NAMES[label]
            or row.get("State") != "COMPLETED"
            or row.get("ExitCode") != "0:0"
            or row.get("NodeList") != PINNED_NODE
            or row.get("NNodes") != "1"
            or row.get("NTasks") != "1"
            or row.get("AllocTRES") != EXPECTED_TRES
        ):
            fail(f"terminal sacct attempt {label} differs")
        result.append(dict(row))
    return result


def load_and_validate_sacct(
    value: str | Path,
    *,
    authority: Mapping[str, Any],
) -> list[Mapping[str, str]]:
    _, payload, _ = _plain_file(value, label="pinned terminal sacct rows")
    if sha256_bytes(payload) != authority["sacct"]["rows_file_sha256"]:
        fail("terminal sacct rows file SHA differs")
    rows = strict_json_bytes(
        payload,
        label="pinned terminal sacct rows",
        allow_one_trailing_newline=True,
    )
    return validate_sacct_rows(rows, authority=authority)


def _utc_epoch(value: Any, *, label: str) -> float:
    if not isinstance(value, str):
        fail(f"{label} is not a timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise RecoveryError(f"{label} encoding differs") from error
    if parsed.tzinfo is not None:
        fail(f"{label} unexpectedly carries a timezone")
    return parsed.replace(tzinfo=timezone.utc).timestamp()


def validate_clock_policy(
    rows: Sequence[Mapping[str, str]],
    nfs_mtime_ns: Mapping[str, Mapping[str, int]],
    *,
    maximum_skew_seconds: float = MAX_CLOCK_SKEW_SECONDS,
) -> Mapping[str, Any]:
    if len(rows) != 2 or set(nfs_mtime_ns) != {"A", "B"}:
        fail("clock policy A/B closure differs")
    observations: dict[str, Any] = {}
    slurm_times: dict[str, tuple[float, float]] = {}
    for label, row in zip(("A", "B"), rows):
        start = _utc_epoch(row.get("Start"), label=f"attempt {label} sacct Start")
        end = _utc_epoch(row.get("End"), label=f"attempt {label} sacct End")
        times = nfs_mtime_ns[label]
        if set(times) != {"intent", "receipt", "terminal", "SUCCESS"}:
            fail(f"attempt {label} NFS timestamp closure differs")
        if any(type(times[key]) is not int or times[key] <= 0 for key in times):
            fail(f"attempt {label} NFS timestamp encoding differs")
        intent_epoch = times["intent"] / 1_000_000_000
        skew = abs(start - intent_epoch)
        if skew > maximum_skew_seconds:
            fail(f"attempt {label} cross-clock start/claim skew exceeds bound")
        if end < start:
            fail(f"attempt {label} sacct End predates Start")
        ordered = [times[key] for key in ("intent", "receipt", "terminal", "SUCCESS")]
        if ordered != sorted(ordered):
            fail(f"attempt {label} NFS evidence order differs")
        slurm_times[label] = (start, end)
        observations[label] = {
            "slurm_step": row["JobIDRaw"],
            "slurm_start": row["Start"],
            "slurm_end": row["End"],
            "intent_nfs_mtime_ns": times["intent"],
            "receipt_nfs_mtime_ns": times["receipt"],
            "terminal_nfs_mtime_ns": times["terminal"],
            "success_nfs_mtime_ns": times["SUCCESS"],
            "absolute_start_claim_skew_seconds": round(skew, 9),
        }
    if slurm_times["A"][1] > slurm_times["B"][0]:
        fail("attempt A sacct End is after attempt B sacct Start")
    if nfs_mtime_ns["A"]["SUCCESS"] > nfs_mtime_ns["B"]["intent"]:
        fail("attempt A SUCCESS is after attempt B intent in the NFS clock domain")
    return {
        "policy": "bounded_NFSv3_server_mtime_vs_Slurm_controller_sacct_v1",
        "different_clock_domains": True,
        "maximum_absolute_start_claim_skew_seconds": maximum_skew_seconds,
        "nfs_domain_order_verified": True,
        "slurm_domain_order_verified": True,
        "attempts": observations,
    }


def build_recovery_receipt(
    *,
    authority: Mapping[str, Any],
    authority_sha: str,
    attempts: Mapping[str, AttemptEvidence],
    rows: Sequence[Mapping[str, str]],
) -> Mapping[str, Any]:
    parity = validate_cross_attempt_parity(attempts)
    clock = validate_clock_policy(
        rows,
        authority["nfs_mtime_ns"],
        maximum_skew_seconds=float(
            authority["clock_domain_policy"][
                "maximum_absolute_start_claim_skew_seconds"
            ]
        ),
    )
    output_claims = authority["output_claims"]
    if output_claims != {
        "counts_as_d0": False,
        "formal_training_started": False,
        "full40_denoise_executed": False,
        "full_bernini_renderer_forward_executed": False,
        "gpu_relaunched": False,
        "mp4_emitted": False,
        "offline_product_inference_completed": False,
        "old_campaign_failed": True,
        "promotion_authorized": False,
        "recovered_parity_only": True,
        "scientific_claim_authorized": False,
    }:
        fail("recovery output claims differ")
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "authority": AUTHORITY,
        "status": "RECOVERED_PARITY_ONLY",
        "recovery_authority_sha256": authority_sha,
        "recovery_reason": "cross_clock_domain",
        "old_campaign": {
            "tag": OLD_CAMPAIGN_TAG,
            "status": "FAILED_FAIL_CLOSED",
            "compare_log_sha256": next(
                row["sha256"]
                for row in authority["evidence_files"]
                if row["path"] == "compare.log"
            ),
            "parity_receipt_published": False,
            "root_success_published": False,
        },
        "old_trust_roots": dict(authority["old_trust_roots"]),
        "terminal_sacct_rows_sha256": authority["sacct"]["rows_file_sha256"],
        "terminal_sacct_rows": [dict(row) for row in rows],
        "source_attempts": {
            label: {
                "intent_sha256": evidence.file_sha256["intent"],
                "receipt_file_sha256": evidence.file_sha256["receipt"],
                "receipt_digest": evidence.receipt["receipt_digest"],
                "validation_file_sha256": evidence.file_sha256["validation1"],
                "controller_status_sha256": evidence.file_sha256["controller"],
                "terminal_authority_sha256": evidence.file_sha256["terminal"],
                "success_sha256": evidence.file_sha256["success"],
                "slurm_numeric_step": evidence.receipt["launch_binding"][
                    "slurm_numeric_step"
                ],
                "parent_untouched": True,
            }
            for label, evidence in attempts.items()
        },
        "recovered_parity": dict(parity),
        "clock_domain_evidence": dict(clock),
        **dict(output_claims),
    }
    if (
        unsigned["gpu_relaunched"] is not False
        or unsigned["promotion_authorized"] is not False
        or unsigned["old_campaign_failed"] is not True
        or unsigned["recovered_parity_only"] is not True
    ):
        fail("recovery boundary claims differ")
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def validate_inputs(
    *,
    old_release_manifest: str | Path,
    old_launch_authority_core: str | Path,
    old_deployment_pins: str | Path,
    evidence_root: str | Path,
    sacct_rows_json: str | Path,
) -> Mapping[str, Any]:
    authority, authority_sha = load_frozen_authority()
    _, core = validate_old_trust_roots(
        old_release_manifest,
        old_launch_authority_core,
        old_deployment_pins,
        authority=authority,
    )
    _, loaded = validate_evidence_files(evidence_root, authority=authority)
    attempts = load_and_validate_attempts(loaded, authority=authority, core=core)
    rows = load_and_validate_sacct(sacct_rows_json, authority=authority)
    return build_recovery_receipt(
        authority=authority,
        authority_sha=authority_sha,
        attempts=attempts,
        rows=rows,
    )


def publish_create_only(output_root_value: str | Path, receipt: Mapping[str, Any]) -> Path:
    output_root = _absolute_plain_path(
        output_root_value, directory=True, label="fresh recovery output root"
    )
    root_info = output_root.lstat()
    if stat.S_IMODE(root_info.st_mode) != 0o700:
        fail("fresh recovery output root mode differs")
    if any(output_root.iterdir()):
        fail("fresh recovery output root is not empty")
    destination = output_root / OUTPUT_FILENAME
    payload = canonical_json_bytes(receipt)
    temporary = output_root / f".{OUTPUT_FILENAME}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    created = False
    linked = False
    file_descriptor = -1
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        file_descriptor = os.open(temporary, flags, 0o400)
        created = True
        offset = 0
        while offset < len(payload):
            written = os.write(file_descriptor, payload[offset:])
            if written <= 0:
                fail("recovery provisional write made no progress")
            offset += written
        os.fsync(file_descriptor)
        os.fchmod(file_descriptor, 0o444)
        written_info = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(written_info.st_mode)
            or stat.S_IMODE(written_info.st_mode) != 0o444
            or written_info.st_nlink != 1
            or written_info.st_size != len(payload)
        ):
            fail("recovery provisional topology differs")
        os.close(file_descriptor)
        file_descriptor = -1
        os.link(temporary, destination, follow_symlinks=False)
        linked = True
        temporary.unlink()
        created = False
        directory_fd = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise RecoveryError("recovery receipt already exists") from error
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if created:
            try:
                temporary.unlink()
            except OSError:
                pass
    if not linked:
        fail("recovery receipt was not published")
    _, observed, info = _plain_file(destination, label="published recovery receipt")
    if (
        stat.S_IMODE(info.st_mode) != 0o444
        or observed != payload
        or sha256_bytes(observed) != sha256_bytes(payload)
        or {item.name for item in output_root.iterdir()} != {OUTPUT_FILENAME}
    ):
        fail("published recovery receipt differs")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-release-manifest", required=True)
    parser.add_argument("--old-launch-authority-core", required=True)
    parser.add_argument("--old-deployment-pins", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--sacct-rows-json", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    recover = subparsers.add_parser("recover")
    recover.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = validate_inputs(
            old_release_manifest=args.old_release_manifest,
            old_launch_authority_core=args.old_launch_authority_core,
            old_deployment_pins=args.old_deployment_pins,
            evidence_root=args.evidence_root,
            sacct_rows_json=args.sacct_rows_json,
        )
        if args.command == "validate":
            sys.stdout.buffer.write(canonical_json_bytes(receipt) + b"\n")
            return 0
        destination = publish_create_only(args.output_root, receipt)
        print(
            "PASS Level-A compare-only recovery "
            f"receipt_sha256={sha256_bytes(destination.read_bytes())} "
            "old_campaign_failed=true gpu_relaunched=false promotion=false",
            flush=True,
        )
        return 0
    except RecoveryError as error:
        print(f"Level-A compare-only recovery refused: {error}", file=sys.stderr)
        return 98


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_SHA256",
    "AttemptEvidence",
    "MAX_CLOCK_SKEW_SECONDS",
    "OUTPUT_FILENAME",
    "RecoveryError",
    "allowed_parity_projection",
    "build_recovery_receipt",
    "canonical_json_bytes",
    "load_frozen_authority",
    "object_sha256",
    "publish_create_only",
    "sha256_bytes",
    "strict_json_bytes",
    "validate_clock_policy",
    "validate_cross_attempt_parity",
    "validate_inputs",
    "validate_sacct_rows",
]
