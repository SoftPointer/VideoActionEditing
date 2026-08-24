#!/usr/bin/env python3
"""Fail-closed postflight and strict manual evaluation for case01 trajectory exact5.

The producer accepts only one completed five-arm run.  It replays the launch
plan, the runner report and attestation, all five native inference receipts,
and all five 81-frame outputs *before* it creates the requested portable
bundle.  It then copies the verified bytes and makes deterministic 9x9
all-frame sheets.  Missing arms are never filled and historical outputs are
never substituted.

The second stage consumes a separately authored, digest-sealed all-81-frame
observation document.  Per-arm decisions reuse the exact identity,
``bone#1`` lineage/conservation, and ordered
``approach -> contact -> grip -> lift -> hold`` gates from
``case01_source_object_strict_eval_v1``.  The gates are conjunctive: no score,
arm, or cue can compensate for a failed gate.

This is an engineering-oracle canary.  It does not claim a learned
object-centric representation or authorize a scientific/causal conclusion.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

try:
    from methods.bernini_action_editing import (
        case01_object_trajectory_exact5_eval_v3 as trajectory_eval,
    )
    from methods.bernini_action_editing import (
        case01_source_object_strict_eval_v1 as strict_eval,
    )
except ModuleNotFoundError:  # Direct execution from this tools directory.
    repository_root = Path(__file__).resolve().parents[3]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from methods.bernini_action_editing import (
        case01_object_trajectory_exact5_eval_v3 as trajectory_eval,
    )
    from methods.bernini_action_editing import (
        case01_source_object_strict_eval_v1 as strict_eval,
    )


POSTFLIGHT_SCHEMA = "case01-object-trajectory-exact5-postflight-bundle-v3"
OBSERVATION_SCHEMA = (
    "case01-object-trajectory-exact5-strict-observations-v2"
)
OBSERVATION_SKELETON_SCHEMA = (
    "case01-object-trajectory-exact5-strict-observation-skeleton-v1"
)
STRICT_REPORT_SCHEMA = (
    "case01-object-trajectory-exact5-strict-manual-report-v3"
)
REVIEW_DESIGN = "independent_nonblind"
OBSERVATION_STATUS = "COMPLETE_INDEPENDENT_NONBLIND_ALL81_REVIEW"
OBSERVATION_ROLE = "independent-nonblind-all81-strict-observations"
CASE_ID = "case01"
IID = trajectory_eval.IID
INSTRUCTION = trajectory_eval.INSTRUCTION
ARM_ORDER = tuple(trajectory_eval.ARM_ORDER)
TASK_IDS = tuple(trajectory_eval.TASK_IDS)
PRIMARY_ARM = "trajectory_dog_bone"
EXPECTED_OUTPUT_PROBE = {
    "frame_count": 81,
    "fps_num": 25,
    "fps_den": 1,
    "width": 480,
    "height": 496,
    "stream_count": 1,
}
EXPECTED_SOURCE_PROBE = {
    "frame_count": 81,
    "fps_num": 25,
    "fps_den": 1,
    "width": 704,
    "height": 736,
    "stream_count": 1,
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

# This is the exact imageio-ffmpeg binary already admitted by the execution
# package.  Postflight sheets are evidence, so accepting an arbitrary PATH
# ffmpeg would leave their byte-level producer unbound.
EXPECTED_FFMPEG_SHA256 = (
    "e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99"
)
EXPECTED_FFMPEG_SIZE = 79_826_272
ALL81_FILTERGRAPH = (
    "select=between(n\\,0\\,80),scale=160:-2:flags=lanczos,"
    "tile=9x9:nb_frames=81:padding=3:margin=3"
)
ALL81_FRAME_INDICES = list(range(81))
SHEET_GRID_COLUMNS = 9
SHEET_GRID_ROWS = 9
SHEET_TILE_WIDTH = 160
SHEET_PADDING = 3
SHEET_MARGIN = 3

STRICT_SUCCESS_RULE = (
    "per arm: all81 coverage AND same dog identity AND same source bone#1 "
    "reuse/conservation AND approach->contact->grip->lift->hold; no compensation"
)
POSTFLIGHT_CLAIM_LIMITS = {
    "engineering_oracle_only": True,
    "hand_authored_trajectory_scaffold": True,
    "learned_object_centric_method_claim_authorized": False,
    "automatic_output_identity_or_lineage_claimed": False,
    "causal_claim_authorized": False,
    "scientific_claim_authorized": False,
}
POSTFLIGHT_COMPLETION_GATES = {
    "real_output_count": 5,
    "native_receipt_count": 5,
    "runner_report_replayed": True,
    "runner_attestation_replayed": True,
    "all_outputs_exact_81_frames": True,
    "all81_sheets_materialized": True,
    "strict_manual_review_complete": False,
    "html_publication_allowed": False,
}

PLAN_BUNDLE_REL = Path("run/plan.json")
RUNNER_REPORT_BUNDLE_REL = Path("run/runner-report.json")
RUNNER_ATTESTATION_BUNDLE_REL = Path("run/runner-attestation.json")
SOURCE_BUNDLE_REL = Path("source/exact_original.mp4")
MANIFEST_REL = Path("evidence/postflight-manifest.json")
PUBLICATION_MARKER_REL = Path(".publication-complete.json")
PUBLICATION_MARKER_SCHEMA = (
    "case01-object-trajectory-exact5-directory-publication-marker-v1"
)

RUNNER_REPORT_FIELDS = {
    "schema_version", "status", "campaign_mode", "plan_schema_version",
    "plan_digest", "task_count", "task_ids", "variant_order",
    "all_exact5_tasks_verified_no_cherry_pick", "same_model_capture_all_tasks",
    "null_envelope", "retained_publication_root_fd_replayed",
    "retained_ffprobe_executable_fd_replayed",
    "retained_publication_leaf_fds_replayed", "manual_blind_review_required",
    "formal_full16_report", "results", "claim_limits", "report_digest",
}
RUNNER_RESULT_FIELDS = {
    "task_id", "arm", "oracle_arm", "receipt_path", "output_path",
    "receipt_file_sha256", "receipt_digest", "output_sha256", "output_size",
    "media_probe",
}
RUNNER_MEDIA_PROBE_FIELDS = {
    "ffprobe_path", "ffprobe_sha256", "ffprobe_size", "frame_count",
    "fps_num", "fps_den", "width", "height", "stream_count",
}
RUNNER_ATTESTATION_FIELDS = {
    "schema_version", "status", "campaign_mode", "formal_full16_report",
    "manual_blind_review_required", "plan", "physical_bindings",
    "captured_runner_entry", "retained_publication_root",
    "retained_ffprobe_executable", "retained_task_publications",
    "retained_child_publication_handoffs", "retained_final_parents",
    "task_count", "task_ids", "unselected_task_ids", "unselected_task_count",
    "all_exact5_tasks_attempted_exactly_once", "all_exact5_tasks_succeeded",
    "retry_count", "task_result_digests", "task_environment_digests",
    "ffmpeg_exec_authority_digest",
    "all_rank0_encoders_used_retained_ffmpeg_executable", "task_results",
    "task_artifact_replays", "runner_task_json_replayed_for_all_tasks",
    "native_publication_before_parent_post_use_replay",
    "all_model_adapter_post_use_replays_complete",
    "native_receipts_replayed_0400_single_link", "model_capture_digest",
    "same_model_capture_all_exact5_tasks", "model_final", "verified_report",
    "reused_frozen_execution_contract", "exploratory_only",
    "scientific_claim_authorized", "formal_claim_authorized",
    "attestation_digest",
}

TASK_RESULT_FIELDS = {
    "schema_version", "task_index", "task_id", "arm", "plan_digest",
    "task_input_digest", "argv_digest", "environment_digest",
    "ffmpeg_exec_authority_digest", "publication_handoff_authority_digest",
    "publication_handoff_payload_digest", "return_code", "attempt_count",
    "retry_allowed", "model_capture_digest", "adapter_capture_digest",
    "consumption_input_digest", "consumption_digest", "native_receipt_digest",
    "native_receipt_file_sha256", "native_output_sha256", "native_output_size",
    "native_receipt_identity", "native_output_identity", "output_path",
    "receipt_path", "log_basename", "authority_artifacts",
    "native_publication_completed_before_parent_post_use_replay",
    "parent_post_use_closed_before_native_publication", "post_use_replay_complete",
    "task_result_digest",
}
ARTIFACT_REPLAY_FIELDS = {
    "task_id", "artifact_count", "artifact_rows_digest", "consumption_digest",
    "task_result_digest", "runner_task_file_sha256",
    "native_receipt_file_sha256", "native_receipt_mode", "native_receipt_nlink",
    "native_output_sha256", "publication_authority_digest",
    "publication_handoff_authority_digest", "publication_handoff_payload_digest",
    "retained_receipt_and_output_fds_replayed", "v2_verified_result_cross_linked",
    "all_post_use_artifacts_replayed",
}
MODEL_FINAL_FIELDS = {
    "schema_version", "model_capture_digest", "task_count",
    "task_consumption_digests", "task_consumption_set_digest",
    "final_rehash_digest", "private_parent_current_identity",
    "all_model_bytes_rehashed_after_last_task",
    "all_model_file_and_directory_fds_retained_through_final_rehash",
    "model_final_digest",
}
STAT_IDENTITY_FIELDS = {
    "device", "inode", "uid", "gid", "mode", "nlink", "rdev", "size",
    "blocks", "mtime_ns", "ctime_ns",
}
DIRECTORY_IDENTITY_FIELDS = {"device", "inode", "uid", "gid", "mode", "rdev"}
PINNED_FILE_IDENTITY_FIELDS = {
    "path", "sha256", "size", "mode", "device", "inode", "uid", "gid",
    "nlink",
}
PHYSICAL_BINDINGS_FIELDS = {
    "schema_version", "plan_path", "plan_sha256", "plan_digest",
    "authority_binding_digest", "source_authority_digest",
    "condition_authority_digests", "admission_authority_digests",
    "producer_roles_distinct", "allocation", "identities",
    "captured_runner_entry", "captured_runner_entry_required",
    "exec_authority", "exec_authority_retained_source_and_python_fds",
    "ffprobe_authority", "ffprobe_retained_executable_fd",
    "isolated_child_interpreters", "child_environment_exact_allowlist",
    "model_root", "bernini_root", "veomni_root", "campaign_mode",
    "formal_full16_report", "task_count", "task_ids", "retry_allowed",
    "final_artifacts", "physical_bindings_digest",
}
ALLOCATION_FIELDS = {
    "holder_job_id", "node", "slurm_step_id",
    "slurm_environment_source_names", "slurm_environment_raw_values",
    "slurm_observed_absent_fields", "normalized_slurm_authority",
    "world_size", "ulysses_size", "reserved_gpu_count", "visible_gpu_indices",
}
PHYSICAL_IDENTITY_ROLES = {
    "runner", "frozen_runner", "exact5_eval", "bridge", "adapter",
    "object_wrapper_inner", "legacy_infer_lora", "trajectory_projection",
    "trajectory_scaffold",
    "eval_v1", "eval_v2", "model_authority", "python", "torchrun_source",
    "torchrun_handler_source", "torch_local_agent_source",
    "torch_dynamic_rendezvous_source", "torch_multiprocessing_api_source",
    "model_manifest", "ffmpeg", "ffprobe",
}
CAPTURED_ENTRY_FIELDS = {
    "schema_version", "runner_fd", "runner_path", "runner_sha256",
    "runner_identity", "python_fd", "python_path", "python_sha256",
    "python_identity", "release_digest", "bootstrap_sha256", "entry_method",
    "slurm_export_none_required", "bash_privileged_startup_required",
    "captured_source_entry", "authority_digest",
}
CAPTURED_ENTRY_SUMMARY_FIELDS = {
    "authority_digest", "release_digest", "bootstrap_sha256",
    "captured_source_entry", "held_through_attestation_publication",
}
EXEC_AUTHORITY_FIELDS = {"schema_version", "rows", "rows_digest", "binding_digest"}
EXEC_AUTHORITY_ROLES = (
    "python_executable", "bridge_source", "adapter_source", "ffmpeg_executable",
)
FFPROBE_AUTHORITY_FIELDS = {
    "schema_version", "fd", "source_path", "sha256", "identity",
    "authority_digest",
}
AUTHORITY_ARTIFACT_ROLES = (
    "model_capture", "model_pre_use", "consumption_input", "adapter_capture",
    "adapter_pre_use", "adapter_post_use", "adapter_final", "model_post_use",
    "eval_consumption_chain",
)
AUTHORITY_ARTIFACT_SUFFIXES = {
    "model_capture": "-model-capture.json",
    "model_pre_use": "-model-pre-use.json",
    "consumption_input": "-consumption-input.json",
    "adapter_capture": "-adapter-capture.json",
    "adapter_pre_use": "-adapter-pre-use.json",
    "adapter_post_use": "-adapter-post-use.json",
    "adapter_final": "-adapter-final.json",
    "model_post_use": "-model-post-use.json",
    "eval_consumption_chain": "-eval-consumption-chain.json",
}

# The nested attestation is itself evidence.  These source pins are the exact
# object-trajectory execution chain retained by the runner, not workspace-head
# guesses made by postflight.
PINNED_PHYSICAL_SHA256 = {
    "runner": "02207e64a129444b26adf8bd92307102c4a91e85d2a029fa60030a7e9e6f45c8",
    "frozen_runner": "847b91a267fe55cfbfa793027548f82beb5ec9630efab329878576ae6c5a9223",
    "exact5_eval": "cfdfc5fec04243265b6c122649fed9144d89510d17184a77782c0ec0ddc5ed8a",
    "bridge": "c91de7eb821a05c61f66349c02f9232ede27c49e54659f351f72930fb071d136",
    "eval_v1": "d6ef0939a67598e66ccf2652d22520ae3a87a068789f70f921522ba86046138d",
    "eval_v2": "b675b84fd5f1b95a21f6454c9eb8c53b0965d7dbdd3fedf1ea92b6ad153ac982",
    "model_authority": "b9457e434b8000e5368056c925edd0227b4dd3d8a439090494af088817d51ecf",
    "torchrun_source": "1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c",
    "torchrun_handler_source": "9871ee801f346c4952fcaf2cc87965f3c997d974b550df70e1fc7f4534c66e87",
    "torch_local_agent_source": "71f390071316417643aa91514ebb170b3adb7eca5c1fe8286d03fe2eef21e497",
    "torch_dynamic_rendezvous_source": "adc34f683614cdc6de5f5cc64e34ee7201b0671609a7ee574b9731f4266e5cec",
    "torch_multiprocessing_api_source": "f815c915fd857bbff12b4d00530c7c1ffb0badfcd48c41e7f378c65828192ef7",
    "model_manifest": "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
    "ffmpeg": EXPECTED_FFMPEG_SHA256,
}

# Sizes for the migration-critical v3 sources are part of the authority, not
# merely informative metadata.  Historical frozen dependencies remain bound
# by SHA plus their attested positive, single-link identities.
PINNED_PHYSICAL_SIZE = {
    "runner": 21_716,
    "frozen_runner": 144_676,
    "exact5_eval": 116_374,
    "ffmpeg": EXPECTED_FFMPEG_SIZE,
}

EXPECTED_TRAJECTORY_EVAL_SHA256 = PINNED_PHYSICAL_SHA256["exact5_eval"]
EXPECTED_TRAJECTORY_EVAL_SIZE = PINNED_PHYSICAL_SIZE["exact5_eval"]
EXPECTED_STRICT_EVAL_SHA256 = (
    "93cacb2e092d8e07ae365a85fff00d72980a4df674e2a684859845e547e8aaf7"
)
EXPECTED_STRICT_EVAL_SIZE = 40_106


class PostflightError(RuntimeError):
    """The completed-run, portable-bundle, or review closure differs."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise PostflightError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_exact_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_exact_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_exact_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PostflightError(message)


def _require_nonempty_text(value: Any, *, label: str) -> str:
    _require(
        isinstance(value, str) and value.strip() != "",
        f"{label} must be non-empty text",
    )
    return value


def _require_exact_bool(value: Any, expected: bool, *, label: str) -> None:
    _require(type(value) is bool and value is expected, f"{label} boolean differs")


def _require_exact_int(value: Any, expected: int, *, label: str) -> None:
    _require(type(value) is int and value == expected, f"{label} integer differs")


def _require_exact_bool_map(
    value: Any,
    expected: Mapping[str, bool],
    *,
    label: str,
) -> None:
    _require(isinstance(value, Mapping) and set(value) == set(expected), f"{label} schema differs")
    for key, expected_value in expected.items():
        _require_exact_bool(value.get(key), expected_value, label=f"{label}.{key}")


def _require_canonical_declared_absolute_path(value: Any, *, label: str) -> Path:
    _require(isinstance(value, str) and value != "", f"{label} path is absent")
    path = Path(value)
    _require(
        path.is_absolute() and os.path.normpath(value) == value,
        f"{label} path is not canonical absolute",
    )
    return path


def _ensure_plain_parent(path: Path) -> None:
    _require(path.is_absolute(), f"output path is not absolute: {path}")
    _require(os.path.normpath(str(path)) == str(path), f"output path is not canonical: {path}")
    missing: list[Path] = []
    cursor = path.parent
    while not os.path.lexists(cursor):
        _require(cursor.parent != cursor, f"output parent has no existing ancestor: {path}")
        missing.append(cursor)
        cursor = cursor.parent
    # Reject a symlink ancestor before creating even the first descendant.
    _plain_directory(cursor, label=f"output ancestor {cursor}")
    for directory in reversed(missing):
        try:
            os.mkdir(directory, 0o700)
        except FileExistsError:
            pass
        _plain_directory(directory, label=f"created output parent {directory}")
    _plain_directory(path.parent, label=f"output parent {path.parent}")


def _owned_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    # Mode is intentionally excluded: every owned file is captured at 0600
    # and then sealed with fchmod before write-after replay.  Ownership must
    # remain recognizable if a later replay/fsync fails after that transition.
    return (info.st_dev, info.st_ino, info.st_uid, info.st_gid)


def _cleanup_owned_file(path: Path, owned: tuple[int, int, int, int]) -> None:
    """Quarantine, verify, then remove only the inode created by this process.

    Moving the name into a fresh mode-0700 directory closes the old
    ``lstat(path) -> unlink(path)`` replacement race.  If the atomically moved
    inode is not ours, it is linked back create-only and never deleted.
    """

    try:
        current = os.lstat(path)
    except FileNotFoundError:
        return
    if not (
        _owned_identity(current) == owned
        and stat.S_ISREG(current.st_mode)
        and not stat.S_ISLNK(current.st_mode)
        and current.st_uid == os.getuid()
    ):
        return
    _plain_directory(path.parent, label="owned-file cleanup parent")
    quarantine_root = Path(
        tempfile.mkdtemp(prefix=f".{path.name}.cleanup-", dir=path.parent)
    )
    quarantine = quarantine_root / "owned-file"
    restored = False
    try:
        try:
            os.rename(path, quarantine)
        except FileNotFoundError:
            return
        moved = os.lstat(quarantine)
        if (
            _owned_identity(moved) != owned
            or not stat.S_ISREG(moved.st_mode)
            or stat.S_ISLNK(moved.st_mode)
        ):
            # A replacement won the race before rename.  Hard-linking it back
            # is create-only and preserves its inode before removing the
            # quarantine name.
            os.link(quarantine, path, follow_symlinks=False)
            restored = True
        os.unlink(quarantine)
    finally:
        if quarantine.exists() and not restored:
            # Never delete an unrecognized replacement on a recovery failure.
            pass
        else:
            try:
                os.rmdir(quarantine_root)
            except OSError:
                pass


def _cleanup_owned_directory(path: Path, owned: tuple[int, int, int, int]) -> None:
    """Atomically quarantine an owned tree before recursive deletion."""

    try:
        current = os.lstat(path)
    except FileNotFoundError:
        return
    if not (
        _owned_identity(current) == owned
        and stat.S_ISDIR(current.st_mode)
        and not stat.S_ISLNK(current.st_mode)
        and current.st_uid == os.getuid()
    ):
        return
    _plain_directory(path.parent, label="owned-directory cleanup parent")
    quarantine_root = Path(
        tempfile.mkdtemp(prefix=f".{path.name}.cleanup-", dir=path.parent)
    )
    quarantine = quarantine_root / "owned-directory"
    try:
        try:
            os.rename(path, quarantine)
        except FileNotFoundError:
            return
        moved = os.lstat(quarantine)
        if (
            _owned_identity(moved) != owned
            or not stat.S_ISDIR(moved.st_mode)
            or stat.S_ISLNK(moved.st_mode)
        ):
            # The source name was replaced just before rename.  Restore that
            # directory; do not traverse or delete it.
            os.rename(quarantine, path)
            return
        shutil.rmtree(quarantine)
    finally:
        try:
            os.rmdir(quarantine_root)
        except OSError:
            pass


def _fsync_directory(path: Path, *, label: str) -> None:
    _plain_directory(path, label=label)
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes, *, label: str) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        _require(type(written) is int and written > 0, f"short write made no progress: {label}")
        offset += written


def _write_create_only_bytes(
    path: Path,
    payload: bytes,
    *,
    final_mode: int = 0o400,
    label: str,
) -> dict[str, Any]:
    _ensure_plain_parent(path)
    _require(not os.path.lexists(path), f"{label} already exists: {path}")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    owned = _owned_identity(os.fstat(descriptor))
    try:
        try:
            _write_all(descriptor, payload, label=label)
            os.fchmod(descriptor, final_mode)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        row = _stable_file(
            path,
            label=label,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size=len(payload),
            allowed_modes={final_mode},
            return_bytes=True,
        )
        _require(row["bytes"] == payload, f"{label} write-after replay differs")
        row.pop("bytes")
        _fsync_directory(path.parent, label=f"{label} parent")
        return row
    except BaseException:
        _cleanup_owned_file(path, owned)
        raise


def _round_scaled_height_to_even(*, width: int, height: int) -> int:
    """Reproduce ``scale=160:-2`` for the two admitted media geometries."""

    _require(width > 0 and height > 0, "sheet input geometry is invalid")
    # nearest integer multiple of two, with half values rounded upward
    numerator = height * SHEET_TILE_WIDTH
    return 2 * ((numerator + width) // (2 * width))


def _sheet_contract(media_probe: Mapping[str, Any]) -> dict[str, Any]:
    if _json_exact_equal(media_probe, EXPECTED_OUTPUT_PROBE):
        admitted_probe = EXPECTED_OUTPUT_PROBE
    elif _json_exact_equal(media_probe, EXPECTED_SOURCE_PROBE):
        admitted_probe = EXPECTED_SOURCE_PROBE
    else:
        raise PostflightError("sheet input media probe differs")
    _validate_media_probe_exact(
        media_probe, admitted_probe, label="sheet input media probe"
    )
    tile_height = _round_scaled_height_to_even(
        width=media_probe["width"],
        height=media_probe["height"],
    )
    image_width = (
        SHEET_GRID_COLUMNS * SHEET_TILE_WIDTH
        + (SHEET_GRID_COLUMNS - 1) * SHEET_PADDING
        + 2 * SHEET_MARGIN
    )
    image_height = (
        SHEET_GRID_ROWS * tile_height
        + (SHEET_GRID_ROWS - 1) * SHEET_PADDING
        + 2 * SHEET_MARGIN
    )
    return {
        "layout": "9x9-row-major",
        "filtergraph": ALL81_FILTERGRAPH,
        "frame_indices": list(ALL81_FRAME_INDICES),
        "frame_count": 81,
        "tile_count": 81,
        "grid": {"columns": SHEET_GRID_COLUMNS, "rows": SHEET_GRID_ROWS},
        "tile": {"width": SHEET_TILE_WIDTH, "height": tile_height},
        "padding": SHEET_PADDING,
        "margin": SHEET_MARGIN,
        "image": {"width": image_width, "height": image_height},
    }


def _validate_media_probe_exact(
    value: Any,
    expected: Mapping[str, int],
    *,
    label: str,
) -> dict[str, int]:
    _require(isinstance(value, Mapping) and set(value) == set(expected), f"{label} schema differs")
    for key, expected_value in expected.items():
        _require_exact_int(value.get(key), expected_value, label=f"{label}.{key}")
    return dict(value)


def _validate_sheet_contract_exact(
    value: Any,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    expected_keys = {
        "layout", "filtergraph", "frame_indices", "frame_count", "tile_count",
        "grid", "tile", "padding", "margin", "image",
    }
    _require(isinstance(value, Mapping) and set(value) == expected_keys, f"{label} schema differs")
    _require(
        value.get("layout") == "9x9-row-major"
        and type(value.get("layout")) is str
        and value.get("filtergraph") == ALL81_FILTERGRAPH
        and type(value.get("filtergraph")) is str,
        f"{label} identity differs",
    )
    indices = value.get("frame_indices")
    _require(
        isinstance(indices, list)
        and len(indices) == 81
        and all(type(item) is int for item in indices)
        and indices == list(range(81)),
        f"{label} frame coverage differs",
    )
    for key in ("frame_count", "tile_count", "padding", "margin"):
        _require_exact_int(value.get(key), int(expected[key]), label=f"{label}.{key}")
    for object_key, field_keys in (
        ("grid", ("columns", "rows")),
        ("tile", ("width", "height")),
        ("image", ("width", "height")),
    ):
        child = value.get(object_key)
        _require(
            isinstance(child, Mapping) and set(child) == set(field_keys),
            f"{label}.{object_key} schema differs",
        )
        for field in field_keys:
            _require_exact_int(
                child.get(field),
                int(expected[object_key][field]),
                label=f"{label}.{object_key}.{field}",
            )
    _require(dict(value) == dict(expected), f"{label} value differs")
    return dict(value)


def _jpeg_dimensions(payload: bytes, *, label: str) -> tuple[int, int]:
    """Read JPEG SOF dimensions without accepting metadata sidecars as truth."""

    _require(len(payload) >= 4 and payload[:2] == b"\xff\xd8", f"{label} is not a JPEG")
    _require(
        payload[-2:] == b"\xff\xd9",
        f"{label} JPEG EOI/trailing-byte closure differs",
    )
    _require(
        b"\xff\xda" in payload[2:-2],
        f"{label} JPEG lacks a start-of-scan marker",
    )
    offset = 2
    sof_markers = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
    while offset < len(payload):
        while offset < len(payload) and payload[offset] != 0xFF:
            offset += 1
        while offset < len(payload) and payload[offset] == 0xFF:
            offset += 1
        _require(offset < len(payload), f"{label} JPEG marker is truncated")
        marker = payload[offset]
        offset += 1
        if marker in (0x01, 0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        _require(offset + 2 <= len(payload), f"{label} JPEG segment is truncated")
        segment_length = int.from_bytes(payload[offset : offset + 2], "big")
        _require(segment_length >= 2, f"{label} JPEG segment length differs")
        segment_end = offset + segment_length
        _require(segment_end <= len(payload), f"{label} JPEG segment exceeds file")
        if marker in sof_markers:
            _require(segment_length >= 7, f"{label} JPEG SOF is truncated")
            height = int.from_bytes(payload[offset + 3 : offset + 5], "big")
            width = int.from_bytes(payload[offset + 5 : offset + 7], "big")
            _require(width > 0 and height > 0, f"{label} JPEG dimensions differ")
            return width, height
        offset = segment_end
    raise PostflightError(f"{label} JPEG lacks a supported SOF marker")


def _strict_digest(value: Mapping[str, Any], field: str, *, label: str) -> str:
    unsigned = dict(value)
    claimed = unsigned.pop(field, None)
    observed = object_sha256(unsigned)
    _require(
        isinstance(claimed, str) and claimed == observed,
        f"{label} digest differs",
    )
    return claimed


def _require_sha256(value: Any, *, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} SHA-256 differs",
    )
    return value


def _require_absolute_path(value: Any, *, label: str) -> Path:
    return _require_canonical_declared_absolute_path(value, label=label)


def _validate_stat_identity(
    value: Any,
    *,
    label: str,
    permissions: int | None = None,
    nlink: int | None = None,
    size: int | None = None,
    directory: bool = False,
) -> dict[str, int]:
    _require(
        isinstance(value, Mapping) and set(value) == STAT_IDENTITY_FIELDS,
        f"{label} identity schema differs",
    )
    _require(
        all(type(value.get(field)) is int for field in STAT_IDENTITY_FIELDS),
        f"{label} identity integer type differs",
    )
    mode = value["mode"]
    _require(
        stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode),
        f"{label} inode type differs",
    )
    if permissions is not None:
        _require(stat.S_IMODE(mode) == permissions, f"{label} permissions differ")
    if nlink is not None:
        _require_exact_int(value.get("nlink"), nlink, label=f"{label}.nlink")
    if size is not None:
        _require_exact_int(value.get("size"), size, label=f"{label}.size")
    return dict(value)


def _validate_directory_identity(value: Any, *, label: str) -> dict[str, int]:
    _require(
        isinstance(value, Mapping) and set(value) == DIRECTORY_IDENTITY_FIELDS,
        f"{label} directory identity schema differs",
    )
    _require(
        all(type(value.get(field)) is int for field in DIRECTORY_IDENTITY_FIELDS),
        f"{label} directory identity integer type differs",
    )
    _require(stat.S_ISDIR(value["mode"]), f"{label} is not a directory identity")
    return dict(value)


def _portable_plan_replay(
    plan: Mapping[str, Any], *, validation_root: Path,
) -> dict[str, Any]:
    """Run the complete plan validator without reopening native absolute paths.

    The upstream validator has one non-semantic check: its publication parent
    must still exist.  A portable bundle must remain valid after the native
    package is gone, so a canonical clone is pointed at this already-existing
    bundle directory for that check.  Every original path is checked below;
    no original authority is opened and the sealed original is never mutated.
    """

    _plain_directory(validation_root, label="portable plan validation root")
    try:
        clone = json.loads(canonical_json_bytes(plan).decode("utf-8"))
    except (UnicodeError, ValueError, TypeError) as error:
        raise PostflightError("portable plan clone failed") from error
    tasks = clone.get("tasks") if isinstance(clone, Mapping) else None
    _require(isinstance(tasks, list) and len(tasks) == 5, "portable plan task count differs")
    original_paths: list[Path] = []
    for task, task_id in zip(tasks, TASK_IDS):
        output = task.get("output") if isinstance(task, Mapping) else None
        _require(isinstance(output, Mapping), f"portable plan output is absent: {task_id}")
        native_video = _require_absolute_path(
            output.get("video_path"), label=f"portable plan {task_id} output"
        )
        native_receipt = _require_absolute_path(
            output.get("receipt_path"), label=f"portable plan {task_id} receipt"
        )
        _require(
            native_video.name == f"{task_id}.mp4"
            and native_receipt == native_video.with_name(native_video.name + ".receipt.json"),
            f"portable plan output/receipt identity differs: {task_id}",
        )
        original_paths.extend((native_video, native_receipt))
        proxy_video = validation_root / f"{task_id}.mp4"
        output["video_path"] = str(proxy_video)
        output["receipt_path"] = str(proxy_video.with_name(proxy_video.name + ".receipt.json"))
    _require(
        len(set(original_paths)) == 10
        and len({path.parent for path in original_paths}) == 1,
        "portable plan publication leaf closure differs",
    )
    clone.pop("plan_digest", None)
    clone["plan_digest"] = trajectory_eval.object_sha256(clone)
    try:
        trajectory_eval.validate_plan(
            clone,
            reopen_sources=False,
            require_fresh_outputs=False,
            require_launchable=True,
        )
    except trajectory_eval.ObjectTrajectoryEvalError as error:
        raise PostflightError(f"portable trajectory plan replay failed: {error}") from error
    return dict(plan)


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_gid,
        info.st_mode,
        info.st_nlink,
        info.st_rdev,
        info.st_size,
        getattr(info, "st_blocks", 0),
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _stable_file(
    path_value: str | Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    allowed_modes: set[int] | frozenset[int] | None = None,
    return_bytes: bool = False,
) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    _require(path.is_absolute(), f"{label} path is not absolute")
    _require(os.path.normpath(str(path)) == str(path), f"{label} path is not canonical")
    try:
        named_before = os.lstat(path)
    except OSError as error:
        raise PostflightError(f"missing {label}: {path}") from error
    _require(stat.S_ISREG(named_before.st_mode), f"{label} is not a regular file")
    _require(not stat.S_ISLNK(named_before.st_mode), f"{label} is a symlink")
    _require(named_before.st_nlink == 1, f"{label} is not single-link")
    try:
        _require(path.resolve(strict=True) == path, f"{label} resolves elsewhere")
    except OSError as error:
        raise PostflightError(f"unavailable {label}: {path}") from error
    observed_mode = stat.S_IMODE(named_before.st_mode)
    if allowed_modes is not None:
        _require(observed_mode in allowed_modes, f"{label} mode differs")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
            if return_bytes:
                chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named_after = os.lstat(path)
    _require(
        _identity(before) == _identity(after) == _identity(named_before) == _identity(named_after),
        f"{label} changed while hashing",
    )
    sha256 = digest.hexdigest()
    if expected_sha256 is not None:
        _require(sha256 == expected_sha256, f"{label} SHA-256 differs")
    if expected_size is not None:
        _require(size == expected_size, f"{label} size differs")
    return {
        "path": path,
        "sha256": sha256,
        "size": size,
        "mode": observed_mode,
        "bytes": b"".join(chunks) if return_bytes else None,
    }


def _evaluation_authority() -> dict[str, dict[str, Any]]:
    """Bind both semantic evaluators to fixed local source bytes."""

    declarations = (
        (
            "trajectory_eval",
            "trajectory-plan-report-receipt-evaluator",
            Path(trajectory_eval.__file__).resolve(strict=True),
            EXPECTED_TRAJECTORY_EVAL_SHA256,
            EXPECTED_TRAJECTORY_EVAL_SIZE,
        ),
        (
            "strict_eval",
            "strict-source-object-visual-gate-evaluator",
            Path(strict_eval.__file__).resolve(strict=True),
            EXPECTED_STRICT_EVAL_SHA256,
            EXPECTED_STRICT_EVAL_SIZE,
        ),
    )
    result: dict[str, dict[str, Any]] = {}
    for key, role, path, sha256, size in declarations:
        observed = _stable_file(
            path,
            label=role,
            expected_sha256=sha256,
            expected_size=size,
            allowed_modes={0o444, 0o644},
        )
        result[key] = {
            "role": role,
            "sha256": observed["sha256"],
            "size": observed["size"],
        }
    return result


def _pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_existing_path(path_value: str | Path, *, label: str) -> Path:
    path = Path(path_value).expanduser()
    _require(path.is_absolute(), f"{label} path is not absolute")
    _require(os.path.normpath(str(path)) == str(path), f"{label} path is not canonical")
    try:
        info = os.lstat(path)
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PostflightError(f"missing {label}: {path}") from error
    _require(not stat.S_ISLNK(info.st_mode) and resolved == path, f"{label} path resolves elsewhere")
    return path


def _load_canonical_json(
    path_value: str | Path,
    *,
    label: str,
    allowed_modes: set[int] | frozenset[int] | None = None,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = _stable_file(
        path_value,
        label=label,
        expected_sha256=expected_sha256,
        allowed_modes=allowed_modes,
        return_bytes=True,
    )
    raw = row["bytes"]
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise PostflightError(f"{label} is not strict JSON") from error
    _require(isinstance(value, dict), f"{label} root is not an object")
    _require(raw == canonical_json_bytes(value) + b"\n", f"{label} is not canonical JSON plus LF")
    row = dict(row)
    row.pop("bytes")
    return value, row


def _probe_video(path: Path, ffprobe: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                str(ffprobe),
                "-v", "error",
                "-count_frames",
                "-show_entries",
                "stream=codec_type,width,height,avg_frame_rate,nb_read_frames",
                "-of", "json",
                str(path),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            env={"LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PostflightError(f"ffprobe could not decode {path}") from error
    _require(
        completed.returncode == 0,
        f"ffprobe rejected {path}: {completed.stderr.decode('utf-8', 'replace')[:300]}",
    )
    try:
        value = json.loads(completed.stdout)
        streams = value["streams"]
        stream = streams[0]
        rate = Fraction(stream["avg_frame_rate"])
        frame_count = int(stream["nb_read_frames"])
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as error:
        raise PostflightError(f"ffprobe metadata differs: {path}") from error
    _require(
        isinstance(streams, list)
        and len(streams) == 1
        and isinstance(stream, dict)
        and stream.get("codec_type") == "video",
        f"media stream closure differs: {path}",
    )
    return {
        "frame_count": frame_count,
        "fps_num": rate.numerator,
        "fps_den": rate.denominator,
        "width": stream.get("width"),
        "height": stream.get("height"),
        "stream_count": 1,
    }


def _critical_probe(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.get(key) for key in EXPECTED_OUTPUT_PROBE}


def _validate_runner_report(
    report: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _require(set(report) == RUNNER_REPORT_FIELDS, "runner report schema differs")
    _strict_digest(report, "report_digest", label="runner report")
    expected_runner_claim_limits = {
        "engineering_oracle_only": True,
        "learned_object_centric_method_claim_authorized": False,
        "causal_claim_authorized": False,
        "scientific_claim_authorized": False,
        "formal_claim_authorized": False,
        "manual_review_required": True,
    }
    _require_exact_bool_map(
        plan.get("claim_limits"),
        expected_runner_claim_limits,
        label="trajectory plan claim limits",
    )
    _require_exact_bool_map(
        report.get("claim_limits"),
        expected_runner_claim_limits,
        label="runner report claim limits",
    )
    _require_exact_int(report.get("task_count"), 5, label="runner report.task_count")
    for field in (
        "all_exact5_tasks_verified_no_cherry_pick",
        "same_model_capture_all_tasks",
        "retained_publication_root_fd_replayed",
        "retained_ffprobe_executable_fd_replayed",
        "retained_publication_leaf_fds_replayed",
        "manual_blind_review_required",
    ):
        _require_exact_bool(report.get(field), True, label=f"runner report.{field}")
    _require_exact_bool(
        report.get("formal_full16_report"), False, label="runner report.formal_full16_report"
    )
    _require(
        report.get("schema_version") == trajectory_eval.REPORT_SCHEMA
        and report.get("status") == "ENGINEERING_ORACLE_COMPLETE_AWAITING_MANUAL_REVIEW"
        and report.get("campaign_mode") == trajectory_eval.CAMPAIGN
        and report.get("plan_schema_version") == trajectory_eval.SCHEMA_VERSION
        and report.get("plan_digest") == plan["plan_digest"]
        and report.get("task_count") == 5
        and report.get("task_ids") == list(TASK_IDS)
        and report.get("variant_order") == list(ARM_ORDER)
        and report.get("all_exact5_tasks_verified_no_cherry_pick") is True
        and report.get("same_model_capture_all_tasks") is True
        and report.get("retained_publication_root_fd_replayed") is True
        and report.get("retained_ffprobe_executable_fd_replayed") is True
        and report.get("retained_publication_leaf_fds_replayed") is True
        and report.get("manual_blind_review_required") is True
        and report.get("formal_full16_report") is False
        and report.get("claim_limits") == plan["claim_limits"],
        "runner report identity/claim closure differs",
    )
    null_envelope = report.get("null_envelope")
    _require(
        isinstance(null_envelope, Mapping)
        and set(null_envelope)
        == {
            "same_source_prompt_seed_steps_sampler_coordinates",
            "output_byte_equality_required",
            "observed_output_sha256_equal",
            "historical_exact_sha_gate_applied",
        },
        "runner null envelope differs",
    )
    _require_exact_bool(
        null_envelope.get("same_source_prompt_seed_steps_sampler_coordinates"),
        True,
        label="runner null envelope.coordinates",
    )
    _require_exact_bool(
        null_envelope.get("output_byte_equality_required"),
        False,
        label="runner null envelope.byte equality required",
    )
    _require(
        type(null_envelope.get("observed_output_sha256_equal")) is bool,
        "runner null envelope observed equality boolean differs",
    )
    _require_exact_bool(
        null_envelope.get("historical_exact_sha_gate_applied"),
        False,
        label="runner null envelope historical gate",
    )
    results = report.get("results")
    _require(isinstance(results, list) and len(results) == 5, "runner result count differs")
    normalized: list[dict[str, Any]] = []
    for task, expected_arm, result in zip(plan["tasks"], ARM_ORDER, results):
        _require(isinstance(result, Mapping), "runner result row is not an object")
        _require(set(result) == RUNNER_RESULT_FIELDS, "runner result row schema differs")
        _require(
            result.get("task_id") == task["task_id"]
            and result.get("arm") == "full644"
            and result.get("oracle_arm") == expected_arm
            and result.get("receipt_path") == task["output"]["receipt_path"]
            and result.get("output_path") == task["output"]["video_path"]
            and isinstance(result.get("receipt_file_sha256"), str)
            and SHA256_RE.fullmatch(result["receipt_file_sha256"]) is not None
            and isinstance(result.get("receipt_digest"), str)
            and SHA256_RE.fullmatch(result["receipt_digest"]) is not None
            and isinstance(result.get("output_sha256"), str)
            and SHA256_RE.fullmatch(result["output_sha256"]) is not None
            and type(result.get("output_size")) is int
            and result["output_size"] > 0
            and isinstance(result.get("media_probe"), Mapping)
            and set(result["media_probe"]) == RUNNER_MEDIA_PROBE_FIELDS,
            f"runner result closure differs: {expected_arm}",
        )
        declared_probe = result["media_probe"]
        _require(
            declared_probe.get("ffprobe_path") == plan["producer"]["ffprobe_path"]
            and declared_probe.get("ffprobe_sha256")
            == plan["producer"]["ffprobe_sha256"],
            f"runner result ffprobe authority differs: {expected_arm}",
        )
        _require_exact_int(
            declared_probe.get("ffprobe_size"),
            plan["producer"]["ffprobe_size"],
            label=f"runner result ffprobe size {expected_arm}",
        )
        _validate_media_probe_exact(
            _critical_probe(declared_probe),
            EXPECTED_OUTPUT_PROBE,
            label=f"runner result media probe {expected_arm}",
        )
        normalized.append(dict(result))
    return normalized


def _validate_embedded_rows(value: Any, *, label: str) -> list[dict[str, Any]]:
    _require(isinstance(value, list) and len(value) == 5, f"{label} count differs")
    rows: list[dict[str, Any]] = []
    for task_id, row in zip(TASK_IDS, value):
        _require(isinstance(row, Mapping), f"{label} row is not an object")
        _require(row.get("task_id") == task_id, f"{label} task order differs")
        rows.append(dict(row))
    return rows


def _validate_physical_bindings(
    value: Any,
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping) and set(value) == PHYSICAL_BINDINGS_FIELDS,
        "physical bindings root schema differs",
    )
    _strict_digest(value, "physical_bindings_digest", label="physical bindings")
    _require(
        value.get("schema_version")
        == "case01-object-trajectory-exact5-physical-bindings-v3"
        and value.get("plan_sha256") == plan_sha256
        and value.get("plan_digest") == plan.get("plan_digest")
        and value.get("campaign_mode") == trajectory_eval.CAMPAIGN
        and value.get("isolated_child_interpreters") == "-I -S -B"
        and value.get("task_ids") == list(TASK_IDS),
        "physical bindings identity closure differs",
    )
    _require_exact_int(value.get("task_count"), 5, label="physical task_count")
    for field, expected in (
        ("captured_runner_entry_required", True),
        ("exec_authority_retained_source_and_python_fds", True),
        ("ffprobe_retained_executable_fd", True),
        ("child_environment_exact_allowlist", True),
        ("formal_full16_report", False),
        ("retry_allowed", False),
    ):
        _require_exact_bool(value.get(field), expected, label=f"physical {field}")
    plan_path = _require_absolute_path(value.get("plan_path"), label="physical plan path")
    for field in ("model_root", "bernini_root", "veomni_root"):
        _require_absolute_path(value.get(field), label=f"physical {field}")

    authority_binding = {
        "source_authority": plan.get("source_authority"),
        "condition_authorities": plan.get("condition_authorities"),
        "admission_authorities": plan.get("admission_authorities"),
    }
    source_authority = plan.get("source_authority")
    conditions = plan.get("condition_authorities")
    admissions = plan.get("admission_authorities")
    _require(
        isinstance(source_authority, Mapping)
        and isinstance(conditions, Mapping)
        and isinstance(admissions, Mapping)
        and value.get("authority_binding_digest") == object_sha256(authority_binding)
        and value.get("source_authority_digest")
        == source_authority.get("authority_digest")
        and value.get("condition_authority_digests")
        == {
            key: row.get("authority_digest")
            for key, row in sorted(conditions.items())
            if isinstance(row, Mapping)
        }
        and value.get("admission_authority_digests")
        == {
            key: row.get("authority_digest")
            for key, row in sorted(admissions.items())
            if isinstance(row, Mapping)
        },
        "physical condition/admission authority closure differs",
    )
    producer = plan.get("producer")
    _require(isinstance(producer, Mapping), "plan producer is absent")
    _require(
        _json_exact_equal(
            value.get("producer_roles_distinct"),
            {
                "invoked_adapter_source": "r5f_composite_inference_wrapper",
                "object_wrapper_inner_source": "source_loaded_support_only",
                "single_frozen_legacy_module": "base_adapter_infer_lora",
                "composite_inner_and_legacy_hashes_distinct": True,
            },
        )
        and len(
            {
                producer.get("inference_wrapper_sha256"),
                producer.get("object_wrapper_inner_sha256"),
                producer.get("infer_lora_sha256"),
            }
        )
        == 3,
        "physical producer role separation differs",
    )

    final_artifacts = value.get("final_artifacts")
    _require(
        isinstance(final_artifacts, Mapping)
        and set(final_artifacts) == {"output_report", "runner_attestation"},
        "physical final artifact schema differs",
    )
    final_paths = {
        role: _require_absolute_path(path, label=f"physical final {role}")
        for role, path in final_artifacts.items()
    }
    _require(
        len(set(final_paths.values())) == 2,
        "physical final artifact path reuse differs",
    )

    allocation = value.get("allocation")
    _require(
        isinstance(allocation, Mapping) and set(allocation) == ALLOCATION_FIELDS,
        "physical allocation schema differs",
    )
    source_names = {
        "job_id": "SLURM_JOB_ID", "step_id": "SLURM_STEP_ID",
        "gpu_count": "SLURM_GPUS_ON_NODE",
        "gpus_per_node": "SLURM_GPUS_PER_NODE",
        "step_gpu_indices": "SLURM_STEP_GPUS",
        "job_node_count": "SLURM_NNODES",
        "step_node_count": "SLURM_STEP_NUM_NODES",
        "job_nodelist": "SLURM_JOB_NODELIST",
        "step_nodelist": "SLURM_STEP_NODELIST",
    }
    raw = allocation.get("slurm_environment_raw_values")
    holder = allocation.get("holder_job_id")
    step = allocation.get("slurm_step_id")
    node = allocation.get("node")
    normalized = allocation.get("normalized_slurm_authority")
    _require(
        allocation.get("slurm_environment_source_names") == source_names
        and isinstance(raw, Mapping) and set(raw) == set(source_names.values())
        and isinstance(holder, str) and holder.isdecimal() and str(int(holder)) == holder
        and isinstance(step, str) and step.isdecimal() and int(step) > 0
        and str(int(step)) == step
        and isinstance(node, str) and node.strip() == node and node != ""
        and raw.get("SLURM_JOB_ID") == holder
        and raw.get("SLURM_STEP_ID") == step
        and raw.get("SLURM_GPUS_ON_NODE") == "8"
        and raw.get("SLURM_GPUS_PER_NODE") == "8"
        and raw.get("SLURM_STEP_GPUS") == "0,1,2,3,4,5,6,7"
        and raw.get("SLURM_NNODES") == "1"
        and raw.get("SLURM_STEP_NUM_NODES") == "1"
        and raw.get("SLURM_JOB_NODELIST") == node
        and raw.get("SLURM_STEP_NODELIST") == node
        and allocation.get("slurm_observed_absent_fields")
        == ["SLURM_JOB_GPUS", "SLURM_JOB_NUM_NODES"]
        and _json_exact_equal(
            normalized,
            {
                "job_node_count": 1, "step_node_count": 1,
                "gpu_count_on_node": 8, "gpus_per_node": 8,
                "step_gpu_indices": list(range(8)), "job_node": node,
                "step_node": node,
            },
        ),
        "physical allocation authority differs",
    )
    for field, expected in (
        ("world_size", 4), ("ulysses_size", 4),
        ("reserved_gpu_count", 8),
    ):
        _require_exact_int(allocation.get(field), expected, label=f"allocation {field}")
    _require(
        allocation.get("visible_gpu_indices") == [0, 1, 2, 3]
        and all(type(item) is int for item in allocation["visible_gpu_indices"]),
        "physical visible GPU indices differ",
    )

    identities = value.get("identities")
    _require(
        isinstance(identities, Mapping) and set(identities) == PHYSICAL_IDENTITY_ROLES,
        "physical identity role closure differs",
    )
    dynamic_sha = {
        "adapter": producer.get("inference_wrapper_sha256"),
        "object_wrapper_inner": producer.get("object_wrapper_inner_sha256"),
        "legacy_infer_lora": producer.get("infer_lora_sha256"),
        "trajectory_projection": producer.get("trajectory_projection_module_sha256"),
        "trajectory_scaffold": producer.get("trajectory_scaffold_module_sha256"),
        "ffprobe": producer.get("ffprobe_sha256"),
    }
    dynamic_path = {
        "adapter": producer.get("inference_wrapper_path"),
        "object_wrapper_inner": producer.get("object_wrapper_inner_path"),
        "legacy_infer_lora": producer.get("infer_lora_path"),
        "trajectory_projection": producer.get("trajectory_projection_module_path"),
        "trajectory_scaffold": producer.get("trajectory_scaffold_module_path"),
        "ffprobe": producer.get("ffprobe_path"),
    }
    dynamic_size = {
        "adapter": producer.get("inference_wrapper_size"),
        "object_wrapper_inner": producer.get("object_wrapper_inner_size"),
        "legacy_infer_lora": producer.get("infer_lora_size"),
        "trajectory_projection": producer.get("trajectory_projection_module_size"),
        "trajectory_scaffold": producer.get("trajectory_scaffold_module_size"),
        "ffprobe": producer.get("ffprobe_size"),
    }
    checked_identities: dict[str, dict[str, Any]] = {}
    for role, row in identities.items():
        _require(
            isinstance(row, Mapping) and set(row) == PINNED_FILE_IDENTITY_FIELDS,
            f"physical pinned identity schema differs: {role}",
        )
        _require_absolute_path(row.get("path"), label=f"physical {role} path")
        _require_sha256(row.get("sha256"), label=f"physical {role}")
        for field in ("size", "mode", "device", "inode", "uid", "gid", "nlink"):
            _require(type(row.get(field)) is int, f"physical {role}.{field} type differs")
        _require(row["size"] > 0 and row["nlink"] == 1, f"physical {role} size/link differs")
        expected_sha = dynamic_sha.get(role, PINNED_PHYSICAL_SHA256.get(role))
        if expected_sha is not None:
            _require(row["sha256"] == expected_sha, f"physical source pin differs: {role}")
        expected_size = dynamic_size.get(role, PINNED_PHYSICAL_SIZE.get(role))
        if expected_size is not None:
            _require_exact_int(
                row.get("size"), expected_size,
                label=f"physical source size {role}",
            )
        if role in dynamic_path:
            _require(row["path"] == dynamic_path[role], f"physical producer path differs: {role}")
        checked_identities[role] = dict(row)

    captured = value.get("captured_runner_entry")
    _require(
        isinstance(captured, Mapping) and set(captured) == CAPTURED_ENTRY_FIELDS,
        "captured runner entry schema differs",
    )
    _strict_digest(captured, "authority_digest", label="captured runner entry")
    runner_identity = _validate_stat_identity(
        captured.get("runner_identity"), label="captured runner identity",
        permissions=0o444, nlink=1,
    )
    python_identity = _validate_stat_identity(
        captured.get("python_identity"), label="captured Python identity", nlink=1,
    )
    for field in ("release_digest", "bootstrap_sha256"):
        _require_sha256(captured.get(field), label=f"captured entry {field}")
    _require(
        captured.get("schema_version")
        == "full644-exploratory-matched-captured-runner-entry-authority-v1"
        and type(captured.get("runner_fd")) is int and captured["runner_fd"] >= 3
        and type(captured.get("python_fd")) is int and captured["python_fd"] >= 3
        and captured["runner_fd"] != captured["python_fd"]
        and captured.get("runner_path") == checked_identities["runner"]["path"]
        and captured.get("runner_sha256") == checked_identities["runner"]["sha256"]
        and captured.get("python_path") == checked_identities["python"]["path"]
        and captured.get("python_sha256") == checked_identities["python"]["sha256"]
        and captured.get("entry_method")
        == "slurm-spooled-or-trusted-stdin-held-python-fd-v1",
        "captured runner entry value differs",
    )
    for field in (
        "slurm_export_none_required", "bash_privileged_startup_required",
        "captured_source_entry",
    ):
        _require_exact_bool(captured.get(field), True, label=f"captured entry {field}")
    for identity, role in ((runner_identity, "runner"), (python_identity, "python")):
        pinned = checked_identities[role]
        _require(
            all(identity[field] == pinned[field] for field in (
                "device", "inode", "uid", "gid", "nlink", "size"
            ))
            and stat.S_IMODE(identity["mode"]) == pinned["mode"],
            f"captured {role} identity cross-link differs",
        )
    _require(python_identity["mode"] & 0o111 != 0, "captured Python is not executable")

    exec_authority = value.get("exec_authority")
    _require(
        isinstance(exec_authority, Mapping) and set(exec_authority) == EXEC_AUTHORITY_FIELDS,
        "retained exec authority schema differs",
    )
    _strict_digest(exec_authority, "binding_digest", label="retained exec authority")
    exec_rows = exec_authority.get("rows")
    _require(
        exec_authority.get("schema_version")
        == "full644-exploratory-matched-exec-authority-v2"
        and isinstance(exec_rows, list)
        and [row.get("role") if isinstance(row, Mapping) else None for row in exec_rows]
        == list(EXEC_AUTHORITY_ROLES)
        and exec_authority.get("rows_digest") == object_sha256(exec_rows),
        "retained exec authority row closure differs",
    )
    exec_physical_roles = ("python", "bridge", "adapter", "ffmpeg")
    exec_fds: list[int] = []
    for row, role in zip(exec_rows, exec_physical_roles):
        _require(
            isinstance(row, Mapping)
            and set(row) == {"role", "fd", "source_path", "sha256", "identity"},
            f"retained exec row schema differs: {role}",
        )
        identity = _validate_stat_identity(
            row.get("identity"), label=f"retained exec {role}", nlink=1,
        )
        fd = row.get("fd")
        pinned = checked_identities[role]
        _require(
            type(fd) is int and fd >= 3
            and row.get("source_path") == pinned["path"]
            and row.get("sha256") == pinned["sha256"]
            and all(identity[field] == pinned[field] for field in (
                "device", "inode", "uid", "gid", "nlink", "size"
            ))
            and stat.S_IMODE(identity["mode"]) == pinned["mode"],
            f"retained exec cross-link differs: {role}",
        )
        if role in {"python", "ffmpeg"}:
            _require(identity["mode"] & 0o111 != 0, f"retained {role} is not executable")
        exec_fds.append(fd)
    _require(
        exec_fds == sorted(exec_fds) and len(set(exec_fds)) == 4,
        "retained exec FD order differs",
    )

    ffprobe = value.get("ffprobe_authority")
    _require(
        isinstance(ffprobe, Mapping) and set(ffprobe) == FFPROBE_AUTHORITY_FIELDS,
        "retained ffprobe authority schema differs",
    )
    _strict_digest(ffprobe, "authority_digest", label="retained ffprobe authority")
    ffprobe_identity = _validate_stat_identity(
        ffprobe.get("identity"), label="retained ffprobe identity", nlink=1,
    )
    pinned_ffprobe = checked_identities["ffprobe"]
    _require(
        ffprobe.get("schema_version")
        == "bernini-full644-exploratory-matched-ffprobe-exec-authority-v1"
        and type(ffprobe.get("fd")) is int and ffprobe["fd"] >= 3
        and ffprobe.get("source_path") == pinned_ffprobe["path"]
        and ffprobe.get("sha256") == pinned_ffprobe["sha256"]
        and all(ffprobe_identity[field] == pinned_ffprobe[field] for field in (
            "device", "inode", "uid", "gid", "nlink", "size"
        ))
        and stat.S_IMODE(ffprobe_identity["mode"]) == pinned_ffprobe["mode"]
        and ffprobe_identity["mode"] & 0o111 != 0,
        "retained ffprobe authority differs",
    )
    return {
        "plan_path": plan_path,
        "final_artifacts": final_paths,
        "identities": checked_identities,
        "captured_runner_entry": dict(captured),
        "ffprobe_authority": dict(ffprobe),
        "ffmpeg_exec_authority_digest": object_sha256(exec_rows[3]),
    }


def _validate_runner_attestation(
    attestation: Mapping[str, Any],
    *,
    attestation_path: Path,
    plan_path: Path,
    plan_row: Mapping[str, Any],
    plan: Mapping[str, Any],
    report_path: Path,
    report_row: Mapping[str, Any],
    report: Mapping[str, Any],
    expected_native_attestation_path: Path | None = None,
) -> dict[str, Any]:
    _require(set(attestation) == RUNNER_ATTESTATION_FIELDS, "runner attestation schema differs")
    _strict_digest(attestation, "attestation_digest", label="runner attestation")
    plan_link = attestation.get("plan")
    report_link = attestation.get("verified_report")
    task_results = _validate_embedded_rows(attestation.get("task_results"), label="attested task result")
    artifact_replays = _validate_embedded_rows(
        attestation.get("task_artifact_replays"), label="attested artifact replay"
    )
    _require_exact_int(attestation.get("task_count"), 5, label="runner attestation.task_count")
    _require_exact_int(
        attestation.get("unselected_task_count"),
        0,
        label="runner attestation.unselected_task_count",
    )
    _require_exact_int(attestation.get("retry_count"), 0, label="runner attestation.retry_count")
    for field in (
        "manual_blind_review_required", "all_exact5_tasks_attempted_exactly_once",
        "all_exact5_tasks_succeeded",
        "all_rank0_encoders_used_retained_ffmpeg_executable",
        "runner_task_json_replayed_for_all_tasks",
        "native_publication_before_parent_post_use_replay",
        "all_model_adapter_post_use_replays_complete",
        "native_receipts_replayed_0400_single_link",
        "same_model_capture_all_exact5_tasks", "exploratory_only",
    ):
        _require_exact_bool(attestation.get(field), True, label=f"runner attestation.{field}")
    for field in ("formal_full16_report", "scientific_claim_authorized", "formal_claim_authorized"):
        _require_exact_bool(attestation.get(field), False, label=f"runner attestation.{field}")
    _require(
        attestation.get("schema_version")
        == "case01-object-trajectory-exact5-runner-attestation-v3"
        and attestation.get("status") == "EXACT5_COMPLETE_AWAITING_BLIND_REVIEW"
        and attestation.get("campaign_mode") == trajectory_eval.CAMPAIGN
        and attestation.get("formal_full16_report") is False
        and attestation.get("manual_blind_review_required") is True
        and attestation.get("task_count") == 5
        and attestation.get("task_ids") == list(TASK_IDS)
        and attestation.get("unselected_task_ids") == []
        and attestation.get("unselected_task_count") == 0
        and attestation.get("all_exact5_tasks_attempted_exactly_once") is True
        and attestation.get("all_exact5_tasks_succeeded") is True
        and attestation.get("retry_count") == 0
        and attestation.get("runner_task_json_replayed_for_all_tasks") is True
        and attestation.get("native_publication_before_parent_post_use_replay") is True
        and attestation.get("all_model_adapter_post_use_replays_complete") is True
        and attestation.get("native_receipts_replayed_0400_single_link") is True
        and attestation.get("same_model_capture_all_exact5_tasks") is True
        and attestation.get("all_rank0_encoders_used_retained_ffmpeg_executable") is True
        and attestation.get("exploratory_only") is True
        and attestation.get("scientific_claim_authorized") is False
        and attestation.get("formal_claim_authorized") is False,
        "runner attestation completion/claim closure differs",
    )
    _require(
        isinstance(plan_link, Mapping)
        and set(plan_link) == {"path", "sha256", "plan_digest"}
        and plan_link.get("path") == str(plan_path)
        and plan_link.get("sha256") == plan_row["sha256"]
        and plan_link.get("plan_digest") == plan["plan_digest"],
        "runner attestation plan cross-link differs",
    )
    _require(
        isinstance(report_link, Mapping)
        and set(report_link) == {"path", "sha256", "report_digest", "verified_task_count"}
        and report_link.get("path") == str(report_path)
        and report_link.get("sha256") == report_row["sha256"]
        and report_link.get("report_digest") == report["report_digest"]
        and report_link.get("verified_task_count") == 5,
        "runner attestation report cross-link differs",
    )
    _require_exact_int(
        report_link.get("verified_task_count"),
        5,
        label="runner attestation verified task count",
    )
    physical = _validate_physical_bindings(
        attestation.get("physical_bindings"), plan=plan, plan_sha256=plan_row["sha256"]
    )
    captured_summary = attestation.get("captured_runner_entry")
    captured = physical["captured_runner_entry"]
    _require(
        isinstance(captured_summary, Mapping)
        and set(captured_summary) == CAPTURED_ENTRY_SUMMARY_FIELDS
        and captured_summary.get("authority_digest") == captured["authority_digest"]
        and captured_summary.get("release_digest") == captured["release_digest"]
        and captured_summary.get("bootstrap_sha256") == captured["bootstrap_sha256"],
        "runner captured entry summary differs",
    )
    for field in ("captured_source_entry", "held_through_attestation_publication"):
        _require_exact_bool(
            captured_summary.get(field), True, label=f"captured summary.{field}"
        )
    _require(
        plan_link.get("path") == str(physical["plan_path"])
        and report_link.get("path") == str(physical["final_artifacts"]["output_report"])
        and attestation.get("ffmpeg_exec_authority_digest")
        == physical["ffmpeg_exec_authority_digest"],
        "runner physical/final binding differs",
    )
    if expected_native_attestation_path is not None:
        _require(
            physical["final_artifacts"]["runner_attestation"]
            == expected_native_attestation_path,
            "runner physical/native attestation path differs",
        )
    physical_ffprobe = physical["ffprobe_authority"]
    for result in report["results"]:
        probe = result["media_probe"]
        _require(
            probe.get("ffprobe_path") == physical_ffprobe["source_path"]
            and probe.get("ffprobe_sha256") == physical_ffprobe["sha256"]
            and probe.get("ffprobe_size") == physical_ffprobe["identity"]["size"],
            "report/physical ffprobe binding differs",
        )

    retained_root = attestation.get("retained_publication_root")
    retained_ffprobe = attestation.get("retained_ffprobe_executable")
    retained_tasks = attestation.get("retained_task_publications")
    retained_handoffs = attestation.get("retained_child_publication_handoffs")
    retained_final = attestation.get("retained_final_parents")
    publication_parent = str(Path(plan["tasks"][0]["output"]["video_path"]).parent)
    _require(
        isinstance(retained_root, Mapping)
        and set(retained_root)
        == {"path", "fd", "immutable_identity", "held_through_attestation_publication"}
        and retained_root.get("path") == publication_parent
        and type(retained_root.get("fd")) is int and retained_root["fd"] >= 3,
        "retained publication root schema differs",
    )
    retained_root_identity = _validate_directory_identity(
        retained_root.get("immutable_identity"), label="retained publication root"
    )
    _require_exact_bool(
        retained_root.get("held_through_attestation_publication"), True,
        label="retained publication root held",
    )
    _require(
        isinstance(retained_ffprobe, Mapping)
        and set(retained_ffprobe)
        == {"authority_digest", "fd", "source_path", "sha256", "held_through_result_verification"}
        and retained_ffprobe.get("authority_digest") == physical_ffprobe["authority_digest"]
        and retained_ffprobe.get("fd") == physical_ffprobe["fd"]
        and retained_ffprobe.get("source_path") == physical_ffprobe["source_path"]
        and retained_ffprobe.get("sha256") == physical_ffprobe["sha256"],
        "retained ffprobe summary differs",
    )
    _require_exact_bool(
        retained_ffprobe.get("held_through_result_verification"), True,
        label="retained ffprobe held",
    )
    _require(
        isinstance(retained_tasks, Mapping) and set(retained_tasks) == set(TASK_IDS),
        "retained task publication set differs",
    )
    _require(
        isinstance(retained_handoffs, Mapping)
        and set(retained_handoffs) == set(TASK_IDS),
        "retained handoff set differs",
    )
    for task_id in TASK_IDS:
        retained = retained_tasks[task_id]
        handoff = retained_handoffs[task_id]
        _require(
            isinstance(retained, Mapping)
            and set(retained)
            == {"authority_digest", "receipt_fd", "output_fd", "held_through_result_verification"}
            and type(retained.get("receipt_fd")) is int and retained["receipt_fd"] >= 3
            and type(retained.get("output_fd")) is int and retained["output_fd"] >= 3
            and retained["receipt_fd"] != retained["output_fd"],
            f"retained task publication differs: {task_id}",
        )
        _require_sha256(retained.get("authority_digest"), label=f"{task_id} publication")
        _require_exact_bool(
            retained.get("held_through_result_verification"), True,
            label=f"{task_id} retained publication held",
        )
        _require(
            isinstance(handoff, Mapping)
            and set(handoff)
            == {"authority_digest", "fd", "payload_digest", "held_sealed_through_attestation"}
            and type(handoff.get("fd")) is int and handoff["fd"] >= 3,
            f"retained handoff differs: {task_id}",
        )
        _require_sha256(handoff.get("authority_digest"), label=f"{task_id} handoff")
        _require_sha256(handoff.get("payload_digest"), label=f"{task_id} handoff payload")
        _require_exact_bool(
            handoff.get("held_sealed_through_attestation"), True,
            label=f"{task_id} sealed handoff held",
        )
    _require(
        isinstance(retained_final, Mapping)
        and set(retained_final) == {"output_report", "runner_attestation"},
        "retained final parent set differs",
    )
    for role, row in retained_final.items():
        _require(
            isinstance(row, Mapping)
            and set(row) == {"path", "fd", "immutable_identity"}
            and row.get("path") == str(physical["final_artifacts"][role].parent)
            and type(row.get("fd")) is int and row["fd"] >= 3,
            f"retained final parent differs: {role}",
        )
        _validate_directory_identity(
            row.get("immutable_identity"), label=f"retained final parent {role}"
        )

    task_digests = attestation.get("task_result_digests")
    environment_digests = attestation.get("task_environment_digests")
    _require(
        isinstance(task_digests, list)
        and task_digests == [row.get("task_result_digest") for row in task_results],
        "attested task result digest list differs",
    )
    _require(
        isinstance(environment_digests, list)
        and environment_digests == [row.get("environment_digest") for row in task_results],
        "attested task environment digest list differs",
    )
    model_capture = _require_sha256(
        attestation.get("model_capture_digest"), label="runner model capture"
    )
    for index, (task_id, row, replay, result, task) in enumerate(
        zip(TASK_IDS, task_results, artifact_replays, report["results"], plan["tasks"])
    ):
        _require(set(row) == TASK_RESULT_FIELDS, f"task result schema differs: {task_id}")
        _require(set(replay) == ARTIFACT_REPLAY_FIELDS, f"artifact replay schema differs: {task_id}")
        authority_artifacts = row.get("authority_artifacts")
        _require(
            isinstance(authority_artifacts, Mapping)
            and set(authority_artifacts) == set(AUTHORITY_ARTIFACT_ROLES),
            f"task artifact role closure differs: {task_id}",
        )
        prefix = f".matched-v2-{index:02d}-{task_id}"
        for role, suffix in AUTHORITY_ARTIFACT_SUFFIXES.items():
            reference = authority_artifacts[role]
            _require(
                isinstance(reference, Mapping)
                and set(reference) == {"basename", "sha256"}
                and reference.get("basename") == prefix + suffix,
                f"task artifact reference differs: {task_id}:{role}",
            )
            _require_sha256(reference.get("sha256"), label=f"{task_id} {role} artifact")
        replay_rows = [
            {
                "role": role,
                "basename": authority_artifacts[role]["basename"],
                "sha256": authority_artifacts[role]["sha256"],
            }
            for role in sorted(authority_artifacts)
        ]
        expected_task_input = object_sha256(
            {
                "schema_version": "full644-exploratory-matched-task-input-v2",
                "plan_digest": plan["plan_digest"],
                "task": task,
            }
        )
        for field in (
            "task_input_digest", "argv_digest", "environment_digest",
            "ffmpeg_exec_authority_digest", "publication_handoff_authority_digest",
            "publication_handoff_payload_digest", "model_capture_digest",
            "adapter_capture_digest", "consumption_input_digest", "consumption_digest",
            "native_receipt_digest", "native_receipt_file_sha256", "native_output_sha256",
        ):
            _require_sha256(row.get(field), label=f"{task_id} {field}")
        receipt_identity = _validate_stat_identity(
            row.get("native_receipt_identity"), label=f"{task_id} native receipt",
            permissions=0o400, nlink=1,
        )
        output_identity = _validate_stat_identity(
            row.get("native_output_identity"), label=f"{task_id} native output",
            permissions=0o444, nlink=1, size=result["output_size"],
        )
        _require(
            receipt_identity["size"] > 0
            and receipt_identity["uid"] == output_identity["uid"]
            and receipt_identity["gid"] == output_identity["gid"]
            and output_identity["uid"] == retained_root_identity["uid"]
            and output_identity["gid"] == retained_root_identity["gid"],
            f"task publication owner closure differs: {task_id}",
        )
        retained = retained_tasks[task_id]
        handoff = retained_handoffs[task_id]
        publication_authority = {
            "schema_version": "bernini-full644-exploratory-matched-publication-authority-v1",
            "task_id": task_id,
            "output_path": row["output_path"],
            "output_fd": retained["output_fd"],
            "output_identity": output_identity,
            "output_sha256": row["native_output_sha256"],
            "output_size": row["native_output_size"],
            "receipt_path": row["receipt_path"],
            "receipt_fd": retained["receipt_fd"],
            "receipt_identity": receipt_identity,
            "receipt_sha256": row["native_receipt_file_sha256"],
            "receipt_size": receipt_identity["size"],
        }
        handoff_payload = {
            "schema_version": "full644-exploratory-matched-publication-handoff-payload-v1",
            "task_id": task_id,
            "output_path": row["output_path"],
            "output_identity": output_identity,
            "output_sha256": row["native_output_sha256"],
            "output_size": row["native_output_size"],
            "receipt_path": row["receipt_path"],
            "receipt_identity": receipt_identity,
            "receipt_sha256": row["native_receipt_file_sha256"],
            "receipt_size": receipt_identity["size"],
            "receipt_digest": row["native_receipt_digest"],
        }
        _require(
            row.get("schema_version") == "full644-exploratory-matched-runner-task-auh-r5"
            and row.get("task_index") == index and type(row.get("task_index")) is int
            and row.get("task_id") == task_id
            and row.get("arm") == "full644"
            and row.get("plan_digest") == plan["plan_digest"]
            and row.get("task_input_digest") == expected_task_input
            and row.get("return_code") == 0 and type(row.get("return_code")) is int
            and row.get("attempt_count") == 1 and type(row.get("attempt_count")) is int
            and row.get("retry_allowed") is False
            and row.get("model_capture_digest") == model_capture
            and row.get("native_receipt_digest") == result["receipt_digest"]
            and row.get("native_receipt_file_sha256") == result["receipt_file_sha256"]
            and row.get("native_output_sha256") == result["output_sha256"]
            and row.get("native_output_size") == result["output_size"]
            and type(row.get("native_output_size")) is int
            and row.get("output_path") == result["output_path"]
            and row.get("receipt_path") == result["receipt_path"]
            and row.get("log_basename") == prefix + ".log",
            f"task result/report/plan closure differs: {task_id}",
        )
        for field, expected in (
            ("native_publication_completed_before_parent_post_use_replay", True),
            ("parent_post_use_closed_before_native_publication", False),
            ("post_use_replay_complete", True),
        ):
            _require_exact_bool(row.get(field), expected, label=f"{task_id}.{field}")
        _require(
            row.get("environment_digest") == environment_digests[index]
            and row.get("ffmpeg_exec_authority_digest")
            == attestation.get("ffmpeg_exec_authority_digest")
            and row.get("publication_handoff_authority_digest")
            == handoff["authority_digest"]
            and row.get("publication_handoff_payload_digest") == handoff["payload_digest"]
            and _strict_digest(row, "task_result_digest", label=f"task result {task_id}")
            == task_digests[index],
            f"task result attestation closure differs: {task_id}",
        )
        _require(
            replay.get("task_id") == task_id
            and replay.get("task_result_digest") == task_digests[index]
            and replay.get("artifact_count") == 9
            and type(replay.get("artifact_count")) is int
            and replay.get("artifact_rows_digest") == object_sha256(replay_rows)
            and replay.get("runner_task_file_sha256")
            == hashlib.sha256(canonical_json_bytes(row) + b"\n").hexdigest()
            and replay.get("consumption_digest") == row["consumption_digest"]
            and replay.get("native_receipt_file_sha256") == row["native_receipt_file_sha256"]
            and replay.get("native_output_sha256") == row["native_output_sha256"]
            and replay.get("native_receipt_mode") == 0o400
            and type(replay.get("native_receipt_mode")) is int
            and replay.get("native_receipt_nlink") == 1
            and type(replay.get("native_receipt_nlink")) is int
            and replay.get("publication_authority_digest") == retained["authority_digest"]
            and replay.get("publication_authority_digest") == object_sha256(publication_authority)
            and replay.get("publication_handoff_authority_digest") == handoff["authority_digest"]
            and replay.get("publication_handoff_payload_digest") == handoff["payload_digest"]
            and replay.get("publication_handoff_payload_digest") == object_sha256(handoff_payload),
            f"task artifact/publication replay differs: {task_id}",
        )
        for field in (
            "retained_receipt_and_output_fds_replayed",
            "v2_verified_result_cross_linked", "all_post_use_artifacts_replayed",
        ):
            _require_exact_bool(replay.get(field), True, label=f"{task_id} replay.{field}")

    _require(
        len(
            {
                row["authority_artifacts"]["model_capture"]["sha256"]
                for row in task_results
            }
        )
        == 1,
        "runner model-capture artifact bytes differ across exact5",
    )

    model_final = attestation.get("model_final")
    _require(
        isinstance(model_final, Mapping) and set(model_final) == MODEL_FINAL_FIELDS,
        "model final schema differs",
    )
    _strict_digest(model_final, "model_final_digest", label="model final")
    _validate_stat_identity(
        model_final.get("private_parent_current_identity"),
        label="model final private parent", directory=True,
    )
    consumptions = [row["consumption_digest"] for row in task_results]
    _require(
        model_final.get("schema_version")
        == "bernini-action-preservation-model-held-fd-final-v3"
        and model_final.get("model_capture_digest") == model_capture
        and model_final.get("task_count") == 5
        and type(model_final.get("task_count")) is int
        and model_final.get("task_consumption_digests") == consumptions
        and model_final.get("task_consumption_set_digest") == object_sha256(consumptions),
        "model final task-consumption closure differs",
    )
    _require_sha256(model_final.get("final_rehash_digest"), label="model final rehash")
    for field in (
        "all_model_bytes_rehashed_after_last_task",
        "all_model_file_and_directory_fds_retained_through_final_rehash",
    ):
        _require_exact_bool(model_final.get(field), True, label=f"model final.{field}")
    _require(
        _json_exact_equal(
            attestation.get("reused_frozen_execution_contract"),
            {
            "frozen_runner_sha256": PINNED_PHYSICAL_SHA256["frozen_runner"],
            "retained_model_adapter_fd_closure": True,
            "sealed_publication_handoff": True,
            "four_rank_torchrun": True,
            "post_use_replay": True,
            },
        ),
        "reused frozen execution contract differs",
    )
    _require(attestation_path == attestation_path.resolve(strict=True), "attestation path resolves elsewhere")
    return {
        "task_results": task_results,
        "artifact_replays": artifact_replays,
        "physical": physical,
    }


def _validate_receipt_task_closure(
    receipt: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    result: Mapping[str, Any],
    task_result: Mapping[str, Any],
    receipt_file: Mapping[str, Any],
) -> None:
    task_id = task["task_id"]
    expected_task_input = object_sha256(
        {
            "schema_version": "full644-exploratory-matched-task-input-v2",
            "plan_digest": task_result["plan_digest"],
            "task": task,
        }
    )
    consumption = receipt.get("model_consumption")
    output = receipt.get("output")
    adapter = receipt.get("adapter")
    _require(
        isinstance(consumption, Mapping)
        and isinstance(output, Mapping)
        and isinstance(adapter, Mapping),
        f"{task_id} receipt task/model closure is absent",
    )
    _require(
        receipt.get("task_input_digest") == expected_task_input
        and receipt.get("task_input_digest") == task_result["task_input_digest"]
        and receipt.get("consumption_input_digest")
        == task_result["consumption_input_digest"]
        and consumption.get("task_input_digest") == task_result["task_input_digest"]
        and consumption.get("consumption_input_digest")
        == task_result["consumption_input_digest"]
        and consumption.get("model_capture_digest") == task_result["model_capture_digest"]
        and consumption.get("adapter_capture_digest")
        == task_result["adapter_capture_digest"]
        and receipt.get("receipt_digest") == task_result["native_receipt_digest"]
        and receipt.get("receipt_digest") == result["receipt_digest"]
        and receipt_file.get("sha256") == task_result["native_receipt_file_sha256"]
        and receipt_file.get("size") == task_result["native_receipt_identity"]["size"]
        and output.get("path") == task_result["output_path"]
        and output.get("sha256") == task_result["native_output_sha256"]
        and output.get("size") == task_result["native_output_size"],
        f"{task_id} receipt/task-result/consumption closure differs",
    )
    publication_identity = output.get("publication_identity")
    _require(
        isinstance(publication_identity, Mapping)
        and _json_exact_equal(
            publication_identity, task_result["native_output_identity"]
        ),
        f"{task_id} receipt/output physical identity closure differs",
    )
    _require(
        adapter.get("profile") == trajectory_eval.FULL644_PROFILE
        and adapter.get("adapter_model_sha256")
        == task["adapter"]["adapter_model_sha256"],
        f"{task_id} receipt/plan adapter closure differs",
    )


def inspect_completed_run(
    *,
    plan_path: str | Path,
    report_path: str | Path,
    attestation_path: str | Path,
    probe: Callable[[Path, Path], dict[str, Any]] = _probe_video,
) -> dict[str, Any]:
    """Replay a completed run without writing any file or directory."""

    evaluation_authority = _evaluation_authority()
    plan_abs = _canonical_existing_path(plan_path, label="trajectory plan")
    report_abs = _canonical_existing_path(report_path, label="runner report")
    attestation_abs = _canonical_existing_path(attestation_path, label="runner attestation")
    _require(len({plan_abs, report_abs, attestation_abs}) == 3, "final document paths overlap")
    plan, plan_file = _load_canonical_json(
        plan_abs, label="trajectory plan", allowed_modes={0o444}
    )
    try:
        plan = trajectory_eval.validate_plan(
            plan,
            reopen_sources=True,
            require_fresh_outputs=False,
            require_launchable=True,
        )
    except trajectory_eval.ObjectTrajectoryEvalError as error:
        raise PostflightError(f"trajectory plan replay failed: {error}") from error
    report, report_file = _load_canonical_json(
        report_abs, label="runner report", allowed_modes={0o444}
    )
    results = _validate_runner_report(report, plan=plan)
    attestation, attestation_file = _load_canonical_json(
        attestation_abs, label="runner attestation", allowed_modes={0o444}
    )
    attested = _validate_runner_attestation(
        attestation,
        attestation_path=attestation_abs,
        plan_path=plan_abs,
        plan_row=plan_file,
        plan=plan,
        report_path=report_abs,
        report_row=report_file,
        report=report,
        expected_native_attestation_path=attestation_abs,
    )

    producer = plan["producer"]
    ffprobe = Path(producer["ffprobe_path"])
    _stable_file(
        ffprobe,
        label="pinned ffprobe",
        expected_sha256=producer["ffprobe_sha256"],
        expected_size=producer["ffprobe_size"],
    )
    source_authority = plan["source_authority"]
    source = _stable_file(
        source_authority["path"],
        label="exact original source",
        expected_sha256=source_authority["sha256"],
        expected_size=source_authority["size"],
        allowed_modes={0o444},
    )
    source_probe = probe(source["path"], ffprobe)
    source_probe = _validate_media_probe_exact(
        source_probe, EXPECTED_SOURCE_PROBE, label="source video probe"
    )

    arm_rows: list[dict[str, Any]] = []
    null_receipts: dict[str, dict[str, Any]] = {}
    native_authority_paths = [
        plan_abs, report_abs, attestation_abs, Path(source["path"]),
    ]
    for task, result, arm, task_result in zip(
        plan["tasks"], results, ARM_ORDER, attested["task_results"]
    ):
        output_path = Path(task["output"]["video_path"])
        receipt_path = Path(task["output"]["receipt_path"])
        native_authority_paths.extend((output_path, receipt_path))
        output = _stable_file(
            output_path,
            label=f"{arm} native output",
            expected_sha256=result["output_sha256"],
            expected_size=result["output_size"],
            allowed_modes={0o444},
        )
        receipt, receipt_file = _load_canonical_json(
            receipt_path,
            label=f"{arm} native receipt",
            allowed_modes={0o400},
            expected_sha256=result["receipt_file_sha256"],
        )
        try:
            if arm in {"null_before", "null_after"}:
                trajectory_eval.validate_off_inference_receipt(receipt, task, producer)
            else:
                trajectory_eval.validate_custom_inference_receipt(receipt, task, producer)
        except trajectory_eval.ObjectTrajectoryEvalError as error:
            raise PostflightError(f"{arm} native receipt replay failed: {error}") from error
        _require(
            receipt.get("receipt_digest") == result["receipt_digest"]
            and isinstance(receipt.get("output"), Mapping)
            and receipt["output"].get("path") == str(output_path)
            and receipt["output"].get("sha256") == output["sha256"]
            and receipt["output"].get("size") == output["size"],
            f"{arm} receipt/output cross-link differs",
        )
        _validate_receipt_task_closure(
            receipt,
            task=task,
            result=result,
            task_result=task_result,
            receipt_file=receipt_file,
        )
        if arm in {"null_before", "null_after"}:
            null_receipts[arm] = receipt
        media_probe = probe(output_path, ffprobe)
        media_probe = _validate_media_probe_exact(
            media_probe, EXPECTED_OUTPUT_PROBE, label=f"{arm} output probe"
        )
        _require(
            _critical_probe(result["media_probe"]) == media_probe,
            f"{arm} report/media probe cross-link differs",
        )
        arm_rows.append(
            {
                "arm": arm,
                "task_id": task["task_id"],
                "output": output,
                "receipt": receipt_file,
                "receipt_digest": receipt["receipt_digest"],
                "media_probe": media_probe,
            }
        )
    try:
        recomputed_null_envelope = trajectory_eval.validate_null_envelope_receipts(
            null_receipts["null_before"], null_receipts["null_after"]
        )
    except (KeyError, trajectory_eval.ObjectTrajectoryEvalError) as error:
        raise PostflightError(f"null-envelope receipt replay failed: {error}") from error
    _require(
        _json_exact_equal(recomputed_null_envelope, report["null_envelope"]),
        "runner null envelope is not recomputed from null receipts",
    )
    _require(
        len(set(native_authority_paths)) == len(native_authority_paths),
        "native authority path reuse/cross-type alias differs",
    )
    _require(len({row["output"]["sha256"] for row in arm_rows}) >= 1, "output closure is empty")
    return {
        "evaluation_authority": evaluation_authority,
        "plan": plan,
        "plan_file": plan_file,
        "runner_report": report,
        "runner_report_file": report_file,
        "runner_attestation": attestation,
        "runner_attestation_file": attestation_file,
        "source": source,
        "source_probe": source_probe,
        "arms": arm_rows,
    }


def _copy_verified(
    source: Path, target: Path, *, sha256: str, size: int, final_mode: int = 0o444,
) -> None:
    _ensure_plain_parent(target)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    owned = _owned_identity(os.fstat(descriptor))
    digest = hashlib.sha256()
    written = 0
    try:
        try:
            source_descriptor = os.open(
                source,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                before = os.fstat(source_descriptor)
                while True:
                    block = os.read(source_descriptor, 1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    written += len(block)
                    _write_all(descriptor, block, label=f"copy {target}")
                after = os.fstat(source_descriptor)
            finally:
                os.close(source_descriptor)
            os.fchmod(descriptor, final_mode)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _require(_identity(before) == _identity(after), f"source changed while copying: {source}")
        _require(digest.hexdigest() == sha256 and written == size, f"copy source bytes differ: {source}")
        _stable_file(
            target,
            label=f"copied {target.name}",
            expected_sha256=sha256,
            expected_size=size,
            allowed_modes={final_mode},
        )
        _fsync_directory(target.parent, label=f"copied {target.name} parent")
    except BaseException:
        _cleanup_owned_file(target, owned)
        raise


def make_all81_sheet(
    video: str | Path,
    output: str | Path,
    *,
    ffmpeg: str | Path,
    input_probe: Mapping[str, Any],
) -> dict[str, Any]:
    """Fully decode, render twice, and publish one byte-exact 9x9 JPEG."""

    video_path = _canonical_existing_path(video, label="all81 input video")
    output_path = Path(output)
    expected_contract = _sheet_contract(input_probe)
    _require(not os.path.lexists(output_path), f"sheet target already exists: {output_path}")
    _ensure_plain_parent(output_path)
    ffmpeg_path = _canonical_existing_path(ffmpeg, label="all81 ffmpeg")
    decoder = _stable_file(
        ffmpeg_path,
        label="all81 pinned decoder",
        expected_sha256=EXPECTED_FFMPEG_SHA256,
        expected_size=EXPECTED_FFMPEG_SIZE,
    )
    _require(decoder["mode"] & 0o111 != 0, "all81 decoder executable mode differs")
    render_root = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.render-", dir=output_path.parent)
    )
    render_owned = _owned_identity(os.lstat(render_root))
    environment = {"LC_ALL": "C", "LANG": "C"}

    def run(arguments: list[str], *, label: str) -> subprocess.CompletedProcess[bytes]:
        try:
            completed = subprocess.run(
                arguments,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise PostflightError(f"{label} failed: {video_path}") from error
        _require(
            completed.returncode == 0,
            f"{label} failed: {completed.stderr.decode('utf-8', 'replace')[:300]}",
        )
        return completed

    def seal_render(render_path: Path, *, label: str) -> dict[str, Any]:
        _require(os.path.lexists(render_path), f"{label} created no output")
        created = os.lstat(render_path)
        _require(
            stat.S_ISREG(created.st_mode)
            and not stat.S_ISLNK(created.st_mode)
            and created.st_uid == os.getuid(),
            f"{label} output authority differs",
        )
        owned = _owned_identity(created)
        descriptor = os.open(
            render_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            _require(
                _owned_identity(os.fstat(descriptor)) == owned,
                f"{label} inode changed before sealing",
            )
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        row = _stable_file(
            render_path,
            label=label,
            allowed_modes={0o444},
            return_bytes=True,
        )
        _require(row["size"] > 0, f"{label} is empty")
        width, height = _jpeg_dimensions(row["bytes"], label=label)
        _require(
            {"width": width, "height": height} == expected_contract["image"],
            f"all81 sheet dimensions differ: {output_path.name}",
        )
        return row

    try:
        decoded = run(
            [
                str(ffmpeg_path), "-hide_banner", "-loglevel", "error",
                "-xerror", "-i", str(video_path), "-map", "0:v:0",
                "-vsync", "0", "-f", "framehash", "-hash", "sha256", "-",
            ],
            label="all81 full decode",
        )
        try:
            framehash_text = decoded.stdout.decode("ascii", "strict")
        except UnicodeDecodeError as error:
            raise PostflightError("all81 framehash output is not ASCII") from error
        frame_rows = [
            line for line in framehash_text.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        _require(len(frame_rows) == 81, "all81 pinned decoder frame count differs")
        for frame_index, line in enumerate(frame_rows):
            fields = [field.strip() for field in line.split(",")]
            _require(
                len(fields) == 6,
                "all81 pinned decoder framehash row schema differs",
            )
            try:
                stream_index, dts, pts, duration, frame_size = (
                    int(field) for field in fields[:5]
                )
            except ValueError as error:
                raise PostflightError(
                    "all81 pinned decoder framehash integer differs"
                ) from error
            _require(
                stream_index == 0
                and dts == frame_index
                and pts == frame_index
                and duration > 0
                and frame_size > 0
                and SHA256_RE.fullmatch(fields[5].lower()) is not None,
                "all81 pinned decoder framehash sequence differs",
            )
        rendered: list[dict[str, Any]] = []
        for index in (1, 2):
            render_path = render_root / f"render-{index}.jpg"
            run(
                [
                    str(ffmpeg_path), "-hide_banner", "-loglevel", "error", "-n",
                    "-i", str(video_path), "-vf", ALL81_FILTERGRAPH,
                    "-frames:v", "1", str(render_path),
                ],
                label=f"all81 deterministic render {index}",
            )
            rendered.append(
                seal_render(render_path, label=f"all81 deterministic render {index}")
            )
        _require(
            rendered[0]["bytes"] == rendered[1]["bytes"],
            "all81 deterministic rerender bytes differ",
        )
        published = _write_create_only_bytes(
            output_path,
            rendered[0]["bytes"],
            final_mode=0o444,
            label=f"all81 sheet {output_path.name}",
        )
        published["sheet_contract"] = expected_contract
        published["decode_replay"] = {
            "decoder_sha256": decoder["sha256"],
            "decoder_size": decoder["size"],
            "decoded_frame_count": 81,
            "framehash_sha256": hashlib.sha256(decoded.stdout).hexdigest(),
            "deterministic_render_count": 2,
            "deterministic_rerender_byte_equal": True,
            "render_sha256": published["sha256"],
        }
        return published
    finally:
        _cleanup_owned_directory(render_root, render_owned)


def _portable_file_row(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    row = _stable_file(path, label=role, allowed_modes={0o444})
    return {
        "role": role,
        "path": path.relative_to(root).as_posix(),
        "sha256": row["sha256"],
        "size": row["size"],
        "mode": "0444",
    }


def build_publication_marker(
    *, kind: str, authority_role: str, authority_path: str,
    authority_sha256: str, authority_digest: str,
) -> dict[str, Any]:
    _require_nonempty_text(kind, label="publication marker kind")
    _require_nonempty_text(authority_role, label="publication marker authority role")
    _require(
        isinstance(authority_path, str)
        and authority_path != ""
        and not Path(authority_path).is_absolute()
        and ".." not in Path(authority_path).parts
        and Path(authority_path).as_posix() == authority_path,
        "publication marker authority path differs",
    )
    _require_sha256(authority_sha256, label="publication marker authority")
    _require_sha256(authority_digest, label="publication marker authority digest")
    marker: dict[str, Any] = {
        "schema_version": PUBLICATION_MARKER_SCHEMA,
        "status": "COMPLETE",
        "kind": kind,
        "authority": {
            "role": authority_role,
            "path": authority_path,
            "sha256": authority_sha256,
            "digest": authority_digest,
        },
    }
    marker["marker_digest"] = object_sha256(marker)
    return marker


def validate_publication_marker(
    root: Path,
    *,
    kind: str,
    authority_role: str,
    authority_path: str,
    authority_sha256: str,
    authority_digest: str,
) -> dict[str, Any]:
    marker, _ = _load_canonical_json(
        root / PUBLICATION_MARKER_REL,
        label="directory publication marker",
        allowed_modes={0o400},
    )
    _strict_digest(marker, "marker_digest", label="directory publication marker")
    expected = build_publication_marker(
        kind=kind,
        authority_role=authority_role,
        authority_path=authority_path,
        authority_sha256=authority_sha256,
        authority_digest=authority_digest,
    )
    _require(
        _json_exact_equal(marker, expected),
        "directory publication marker authority differs",
    )
    return marker


def _publish_directory_create_only(
    staging: Path, destination: Path, *, marker: Mapping[str, Any],
) -> tuple[int, int, int, int]:
    """Publish without POSIX rename-over-empty-directory replacement.

    ``rename(staging, final)`` is not create-only on POSIX: it may replace an
    already-existing empty directory.  Reserve the final name atomically with
    ``mkdir`` and then transfer the already-validated top-level members.  A
    same-UID actor mutating the freshly reserved directory concurrently is
    outside this local publication threat model; any such mutation is still
    caught by the immediate exact-name/content revalidation.
    """

    _plain_directory(staging, label="publication staging root")
    _plain_directory(destination.parent, label="publication parent")
    _require(staging.parent == destination.parent, "publication roots do not share a parent")
    _require(not os.path.lexists(destination), f"bundle target already exists: {destination}")
    try:
        os.mkdir(destination, 0o700)
    except FileExistsError as error:
        raise PostflightError(f"bundle target was concurrently occupied: {destination}") from error
    reserved = os.lstat(destination)
    _require(
        stat.S_ISDIR(reserved.st_mode)
        and not stat.S_ISLNK(reserved.st_mode)
        and reserved.st_uid == os.getuid(),
        "reserved publication root authority differs",
    )
    owned = _owned_identity(reserved)
    try:
        for member in sorted(staging.iterdir(), key=lambda item: item.name):
            target = destination / member.name
            _require(not os.path.lexists(target), f"publication member became occupied: {target}")
            os.rename(member, target)
        _write_create_only_json(
            destination / PUBLICATION_MARKER_REL,
            marker,
            digest_field="marker_digest",
        )
        _fsync_directory(destination, label="published directory")
        _fsync_directory(destination.parent, label="publication parent")
        staging.rmdir()
        return owned
    except BaseException:
        _cleanup_owned_directory(destination, owned)
        raise


def produce_bundle(
    *,
    plan_path: str | Path,
    report_path: str | Path,
    attestation_path: str | Path,
    bundle_root: str | Path,
    ffmpeg: str | Path,
    probe: Callable[[Path, Path], dict[str, Any]] = _probe_video,
    sheet_builder: Callable[..., dict[str, Any]] = make_all81_sheet,
) -> dict[str, Any]:
    """Create a portable bundle only after the full run validates."""

    destination = Path(bundle_root).expanduser()
    _require(destination.is_absolute(), "bundle root is not absolute")
    _require(os.path.normpath(str(destination)) == str(destination), "bundle root is not canonical")
    _require(not os.path.lexists(destination), "bundle root already exists")

    # Deliberately first: missing output/receipt/report/attestation must not
    # create the bundle root or any sibling staging directory.
    observed = inspect_completed_run(
        plan_path=plan_path,
        report_path=report_path,
        attestation_path=attestation_path,
        probe=probe,
    )
    ffmpeg_path = Path(ffmpeg).expanduser().resolve(strict=True)
    ffmpeg_row = _stable_file(
        ffmpeg_path,
        label="ffmpeg executable",
        expected_sha256=EXPECTED_FFMPEG_SHA256,
        expected_size=EXPECTED_FFMPEG_SIZE,
    )
    _require(ffmpeg_row["mode"] & 0o111 != 0, "ffmpeg executable mode differs")
    _ensure_plain_parent(destination)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    staging_owned = _owned_identity(os.lstat(staging))
    try:
        copy_rows = (
            (observed["plan_file"], PLAN_BUNDLE_REL),
            (observed["runner_report_file"], RUNNER_REPORT_BUNDLE_REL),
            (observed["runner_attestation_file"], RUNNER_ATTESTATION_BUNDLE_REL),
            (observed["source"], SOURCE_BUNDLE_REL),
        )
        for source_row, relative in copy_rows:
            _copy_verified(
                source_row["path"], staging / relative,
                sha256=source_row["sha256"], size=source_row["size"],
            )
        arm_artifacts: list[dict[str, Any]] = []
        for arm_row in observed["arms"]:
            arm = arm_row["arm"]
            output_rel = Path("media") / f"{arm}.mp4"
            receipt_rel = Path("receipts") / f"{arm}.receipt.json"
            _copy_verified(
                arm_row["output"]["path"], staging / output_rel,
                sha256=arm_row["output"]["sha256"], size=arm_row["output"]["size"],
            )
            _copy_verified(
                arm_row["receipt"]["path"], staging / receipt_rel,
                sha256=arm_row["receipt"]["sha256"], size=arm_row["receipt"]["size"],
            )
            sheet_rel = Path("sheets") / f"{arm}-all81.jpg"
            sheet_contract = _sheet_contract(arm_row["media_probe"])
            sheet = sheet_builder(
                staging / output_rel,
                staging / sheet_rel,
                ffmpeg=ffmpeg_path,
                input_probe=arm_row["media_probe"],
            )
            _require(
                _json_exact_equal(sheet.get("sheet_contract"), sheet_contract),
                f"all81 sheet builder contract differs: {arm}",
            )
            arm_artifacts.append(
                {
                    "arm": arm,
                    "task_id": arm_row["task_id"],
                    "output": _portable_file_row(staging, staging / output_rel, role=f"{arm}-output"),
                    "receipt": _portable_file_row(staging, staging / receipt_rel, role=f"{arm}-receipt"),
                    "receipt_digest": arm_row["receipt_digest"],
                    "media_probe": arm_row["media_probe"],
                    "all81_sheet": {
                        **_portable_file_row(staging, staging / sheet_rel, role=f"{arm}-all81-sheet"),
                        "sheet_contract": sheet_contract,
                        "decode_replay": sheet.get("decode_replay"),
                    },
                }
            )
        source_sheet_rel = Path("sheets/source-exact_original-all81.jpg")
        source_sheet = sheet_builder(
            staging / SOURCE_BUNDLE_REL,
            staging / source_sheet_rel,
            ffmpeg=ffmpeg_path,
            input_probe=observed["source_probe"],
        )
        source_sheet_contract = _sheet_contract(observed["source_probe"])
        _require(
            _json_exact_equal(
                source_sheet.get("sheet_contract"), source_sheet_contract
            ),
            "source all81 sheet builder contract differs",
        )
        manifest: dict[str, Any] = {
            "schema_version": POSTFLIGHT_SCHEMA,
            "status": "POSTFLIGHT_COMPLETE_AWAITING_INDEPENDENT_ALL81_REVIEW",
            "case_id": CASE_ID,
            "iid": IID,
            "instruction": INSTRUCTION,
            "arm_order": list(ARM_ORDER),
            "primary_arm": PRIMARY_ARM,
            "runner_documents": {
                "plan": _portable_file_row(staging, staging / PLAN_BUNDLE_REL, role="launch-plan"),
                "report": _portable_file_row(staging, staging / RUNNER_REPORT_BUNDLE_REL, role="runner-report"),
                "report_digest": observed["runner_report"]["report_digest"],
                "attestation": _portable_file_row(
                    staging, staging / RUNNER_ATTESTATION_BUNDLE_REL, role="runner-attestation"
                ),
                "attestation_digest": observed["runner_attestation"]["attestation_digest"],
            },
            "source": {
                "video": _portable_file_row(staging, staging / SOURCE_BUNDLE_REL, role="source-video"),
                "media_probe": observed["source_probe"],
                "all81_sheet": {
                    **_portable_file_row(staging, staging / source_sheet_rel, role="source-all81-sheet"),
                    "sheet_contract": source_sheet_contract,
                    "decode_replay": source_sheet.get("decode_replay"),
                },
            },
            "arms": arm_artifacts,
            "render_authority": {
                "ffmpeg": {
                    "sha256": ffmpeg_row["sha256"],
                    "size": ffmpeg_row["size"],
                },
                "ffprobe": {
                    "role": "runner-media-probe",
                    "sha256": observed["plan"]["producer"]["ffprobe_sha256"],
                    "size": observed["plan"]["producer"]["ffprobe_size"],
                },
                "all81_filtergraph": ALL81_FILTERGRAPH,
            },
            "evaluation_authority": observed["evaluation_authority"],
            "completion_gates": dict(POSTFLIGHT_COMPLETION_GATES),
            "strict_success_rule": STRICT_SUCCESS_RULE,
            "claim_limits": dict(POSTFLIGHT_CLAIM_LIMITS),
        }
        manifest["manifest_digest"] = object_sha256(manifest)
        manifest_path = staging / MANIFEST_REL
        manifest_file = _write_create_only_json(
            manifest_path,
            manifest,
            digest_field="manifest_digest",
        )
        validate_bundle(
            staging,
            ffmpeg=ffmpeg_path,
            require_publication_marker=False,
        )
        marker = build_publication_marker(
            kind="postflight-bundle",
            authority_role="postflight-manifest",
            authority_path=MANIFEST_REL.as_posix(),
            authority_sha256=manifest_file["sha256"],
            authority_digest=manifest["manifest_digest"],
        )
        published_owned = _publish_directory_create_only(
            staging, destination, marker=marker
        )
        try:
            return validate_bundle(destination, ffmpeg=ffmpeg_path)
        except BaseException:
            _cleanup_owned_directory(destination, published_owned)
            raise
    except BaseException:
        _cleanup_owned_directory(staging, staging_owned)
        raise


def _resolve_member(root: Path, relative: Any, *, label: str) -> Path:
    _require(isinstance(relative, str) and relative != "", f"{label} path is absent")
    pure = Path(relative)
    _require(not pure.is_absolute() and ".." not in pure.parts, f"{label} path escapes bundle")
    path = root / pure
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PostflightError(f"missing {label}: {relative}") from error
    _require(resolved == path, f"{label} path resolves elsewhere")
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise PostflightError(f"{label} path escapes bundle") from error
    return path


def _validate_member(
    root: Path,
    value: Any,
    *,
    label: str,
    expected_role: str | None = None,
    expected_path: str | None = None,
    expected_mode: int = 0o444,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} authority is absent")
    _require(
        set(value) == {"role", "path", "sha256", "size", "mode"},
        f"{label} authority schema differs",
    )
    _require_nonempty_text(value.get("role"), label=f"{label} role")
    _require_nonempty_text(value.get("path"), label=f"{label} path")
    _require(
        isinstance(value.get("sha256"), str)
        and SHA256_RE.fullmatch(value["sha256"]) is not None,
        f"{label} SHA-256 declaration differs",
    )
    _require(
        type(value.get("size")) is int and value["size"] > 0,
        f"{label} size declaration differs",
    )
    if expected_role is not None:
        _require(value.get("role") == expected_role, f"{label} role differs")
    if expected_path is not None:
        _require(value.get("path") == expected_path, f"{label} fixed path differs")
    path = _resolve_member(root, value["path"], label=label)
    _require(
        value.get("mode") == f"0{expected_mode:03o}",
        f"{label} bundle mode declaration differs",
    )
    row = _stable_file(
        path,
        label=label,
        expected_sha256=value.get("sha256"),
        expected_size=value.get("size"),
        allowed_modes={expected_mode},
    )
    return {**dict(value), "absolute_path": path, "observed": row}


def _validate_authority_declaration(
    value: Any,
    *,
    role: str,
    path: str,
    label: str,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} authority is absent")
    _require(
        set(value) == {"role", "path", "sha256", "size", "mode"},
        f"{label} authority schema differs",
    )
    _require(
        value.get("role") == role
        and value.get("path") == path
        and value.get("mode") == "0444"
        and isinstance(value.get("sha256"), str)
        and SHA256_RE.fullmatch(value["sha256"]) is not None
        and type(value.get("size")) is int
        and value["size"] > 0,
        f"{label} authority declaration differs",
    )
    return dict(value)


def _validate_sheet_declaration(
    value: Any,
    *,
    role: str,
    path: str,
    media_probe: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} sheet authority is absent")
    _require(
        set(value)
        == {
            "role", "path", "sha256", "size", "mode", "sheet_contract",
            "decode_replay",
        },
        f"{label} sheet schema differs",
    )
    base = _validate_authority_declaration(
        {key: value[key] for key in ("role", "path", "sha256", "size", "mode")},
        role=role,
        path=path,
        label=label,
    )
    contract = _validate_sheet_contract_exact(
        value.get("sheet_contract"),
        _sheet_contract(media_probe),
        label=f"{label} contract",
    )
    replay = value.get("decode_replay")
    _require(
        isinstance(replay, Mapping)
        and set(replay)
        == {
            "decoder_sha256", "decoder_size", "decoded_frame_count",
            "framehash_sha256", "deterministic_render_count",
            "deterministic_rerender_byte_equal", "render_sha256",
        }
        and replay.get("decoder_sha256") == EXPECTED_FFMPEG_SHA256
        and isinstance(replay.get("framehash_sha256"), str)
        and SHA256_RE.fullmatch(replay["framehash_sha256"]) is not None
        and replay.get("render_sha256") == value.get("sha256"),
        f"{label} decode/rerender authority differs",
    )
    _require_exact_int(
        replay.get("decoder_size"), EXPECTED_FFMPEG_SIZE,
        label=f"{label} decoder size",
    )
    _require_exact_int(
        replay.get("decoded_frame_count"), 81,
        label=f"{label} decoded frame count",
    )
    _require_exact_int(
        replay.get("deterministic_render_count"), 2,
        label=f"{label} deterministic render count",
    )
    _require_exact_bool(
        replay.get("deterministic_rerender_byte_equal"), True,
        label=f"{label} deterministic rerender equality",
    )
    return {**base, "sheet_contract": contract, "decode_replay": dict(replay)}


def _plain_directory(path: Path, *, label: str) -> None:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise PostflightError(f"missing {label}: {path}") from error
    _require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"{label} is not a plain directory")
    _require(path.resolve(strict=True) == path, f"{label} resolves elsewhere")


def _exact_names(path: Path, expected: set[str], *, label: str) -> None:
    _plain_directory(path, label=label)
    observed = {item.name for item in path.iterdir()}
    _require(
        observed == expected,
        f"{label} name closure differs: missing={sorted(expected - observed)} extra={sorted(observed - expected)}",
    )


def _validate_sheet(
    root: Path,
    value: Any,
    *,
    label: str,
    media_probe: Mapping[str, Any],
    expected_role: str,
    expected_path: str,
    video_path: Path,
    video_sha256: str,
    ffmpeg_path: Path,
    replay_cache: dict[tuple[str, str, str, str], bool],
) -> dict[str, Any]:
    declaration = _validate_sheet_declaration(
        value,
        role=expected_role,
        path=expected_path,
        media_probe=media_probe,
        label=label,
    )
    expected_contract = _sheet_contract(media_probe)
    _validate_sheet_contract_exact(
        value.get("sheet_contract"), expected_contract, label=f"{label} sheet coverage"
    )
    base = {key: value[key] for key in ("role", "path", "sha256", "size", "mode")}
    member = _validate_member(
        root,
        base,
        label=label,
        expected_role=expected_role,
        expected_path=expected_path,
    )
    payload = _stable_file(
        member["absolute_path"],
        label=label,
        expected_sha256=member["sha256"],
        expected_size=member["size"],
        allowed_modes={0o444},
        return_bytes=True,
    )["bytes"]
    width, height = _jpeg_dimensions(payload, label=label)
    _require(
        {"width": width, "height": height} == expected_contract["image"],
        f"{label} sheet dimensions differ",
    )
    replay_key = (
        video_sha256,
        member["sha256"],
        object_sha256(declaration["decode_replay"]),
        object_sha256(dict(media_probe)),
    )
    # Reuse is content-addressed, never path- or arm-addressed: two names may
    # share one replay only after both embedded video bytes, embedded sheet
    # bytes, the complete replay declaration, and the media contract match.
    if replay_key in replay_cache:
        return {**declaration, **member}
    replay_root = Path(
        tempfile.mkdtemp(prefix="case01-all81-validator-replay-")
    ).resolve(strict=True)
    replay_owned = _owned_identity(os.lstat(replay_root))
    try:
        replay_path = replay_root / "all81-replay.jpg"
        replayed = make_all81_sheet(
            video_path,
            replay_path,
            ffmpeg=ffmpeg_path,
            input_probe=media_probe,
        )
        replay_file = _stable_file(
            replay_path,
            label=f"{label} deterministic replay",
            expected_sha256=member["sha256"],
            expected_size=member["size"],
            allowed_modes={0o444},
            return_bytes=True,
        )
        _require(
            replay_file["bytes"] == payload,
            f"{label} bytes differ from pinned-decoder MP4 replay",
        )
        _require(
            _json_exact_equal(
                replayed.get("decode_replay"),
                declaration.get("decode_replay"),
            ),
            f"{label} decode replay differs from pinned-decoder MP4 replay",
        )
        replay_cache[replay_key] = True
    finally:
        _cleanup_owned_directory(replay_root, replay_owned)
    return {**declaration, **member}


def _validate_manifest_semantics(manifest: Mapping[str, Any]) -> None:
    expected_fields = {
        "schema_version", "status", "case_id", "iid", "instruction",
        "arm_order", "primary_arm", "runner_documents", "source", "arms",
        "render_authority", "evaluation_authority", "completion_gates",
        "strict_success_rule", "claim_limits", "manifest_digest",
    }
    _require(set(manifest) == expected_fields, "postflight manifest schema differs")
    _require(
        manifest.get("schema_version") == POSTFLIGHT_SCHEMA
        and manifest.get("status") == "POSTFLIGHT_COMPLETE_AWAITING_INDEPENDENT_ALL81_REVIEW"
        and manifest.get("case_id") == CASE_ID
        and manifest.get("iid") == IID
        and manifest.get("instruction") == INSTRUCTION
        and manifest.get("arm_order") == list(ARM_ORDER)
        and manifest.get("primary_arm") == PRIMARY_ARM
        and manifest.get("strict_success_rule") == STRICT_SUCCESS_RULE
        and type(manifest.get("strict_success_rule")) is str,
        "postflight manifest identity/rule differs",
    )

    completion = manifest.get("completion_gates")
    _require(isinstance(completion, Mapping) and set(completion) == set(POSTFLIGHT_COMPLETION_GATES), "postflight completion schema differs")
    for key, expected in POSTFLIGHT_COMPLETION_GATES.items():
        if type(expected) is int:
            _require_exact_int(completion.get(key), expected, label=f"postflight completion.{key}")
        else:
            _require_exact_bool(completion.get(key), expected, label=f"postflight completion.{key}")
    _require_exact_bool_map(
        manifest.get("claim_limits"), POSTFLIGHT_CLAIM_LIMITS, label="postflight claim limits"
    )
    _require(
        _json_exact_equal(
            manifest.get("evaluation_authority"), _evaluation_authority()
        ),
        "postflight evaluation authority differs",
    )

    render = manifest.get("render_authority")
    _require(
        isinstance(render, Mapping)
        and set(render) == {"ffmpeg", "ffprobe", "all81_filtergraph"}
        and type(render.get("all81_filtergraph")) is str
        and render.get("all81_filtergraph") == ALL81_FILTERGRAPH,
        "postflight render authority schema differs",
    )
    ffmpeg = render.get("ffmpeg")
    _require(
        isinstance(ffmpeg, Mapping)
        and set(ffmpeg) == {"sha256", "size"}
        and ffmpeg.get("sha256") == EXPECTED_FFMPEG_SHA256,
        "postflight ffmpeg authority differs",
    )
    _require_exact_int(ffmpeg.get("size"), EXPECTED_FFMPEG_SIZE, label="postflight ffmpeg size")
    ffprobe = render.get("ffprobe")
    _require(
        isinstance(ffprobe, Mapping)
        and set(ffprobe) == {"role", "sha256", "size"}
        and ffprobe.get("role") == "runner-media-probe",
        "postflight ffprobe authority schema differs",
    )
    _require_sha256(ffprobe.get("sha256"), label="postflight ffprobe")
    _require(
        type(ffprobe.get("size")) is int and ffprobe["size"] > 0,
        "postflight ffprobe size differs",
    )

    runner = manifest.get("runner_documents")
    _require(
        isinstance(runner, Mapping)
        and set(runner) == {"plan", "report", "report_digest", "attestation", "attestation_digest"},
        "runner document schema differs",
    )
    _validate_authority_declaration(
        runner.get("plan"), role="launch-plan", path=PLAN_BUNDLE_REL.as_posix(), label="bundled plan"
    )
    _validate_authority_declaration(
        runner.get("report"), role="runner-report", path=RUNNER_REPORT_BUNDLE_REL.as_posix(), label="bundled runner report"
    )
    _validate_authority_declaration(
        runner.get("attestation"),
        role="runner-attestation",
        path=RUNNER_ATTESTATION_BUNDLE_REL.as_posix(),
        label="bundled runner attestation",
    )
    for field in ("report_digest", "attestation_digest"):
        _require(
            isinstance(runner.get(field), str)
            and SHA256_RE.fullmatch(runner[field]) is not None,
            f"runner document {field} differs",
        )

    source = manifest.get("source")
    _require(
        isinstance(source, Mapping)
        and set(source) == {"video", "media_probe", "all81_sheet"},
        "source schema differs",
    )
    _validate_authority_declaration(
        source.get("video"),
        role="source-video",
        path=SOURCE_BUNDLE_REL.as_posix(),
        label="bundled source video",
    )
    _validate_media_probe_exact(source.get("media_probe"), EXPECTED_SOURCE_PROBE, label="bundled source probe")
    _validate_sheet_declaration(
        source.get("all81_sheet"),
        role="source-all81-sheet",
        path="sheets/source-exact_original-all81.jpg",
        media_probe=EXPECTED_SOURCE_PROBE,
        label="bundled source sheet",
    )

    arms = manifest.get("arms")
    _require(isinstance(arms, list) and len(arms) == 5, "bundled arm count differs")
    for arm, task_id, row in zip(ARM_ORDER, TASK_IDS, arms):
        _require(
            isinstance(row, Mapping)
            and set(row)
            == {"arm", "task_id", "output", "receipt", "receipt_digest", "media_probe", "all81_sheet"}
            and row.get("arm") == arm
            and row.get("task_id") == task_id,
            f"bundled arm identity differs: {arm}",
        )
        _validate_authority_declaration(
            row.get("output"),
            role=f"{arm}-output",
            path=f"media/{arm}.mp4",
            label=f"{arm} output",
        )
        _validate_authority_declaration(
            row.get("receipt"),
            role=f"{arm}-receipt",
            path=f"receipts/{arm}.receipt.json",
            label=f"{arm} receipt",
        )
        _require(
            isinstance(row.get("receipt_digest"), str)
            and SHA256_RE.fullmatch(row["receipt_digest"]) is not None,
            f"{arm} receipt digest differs",
        )
        _validate_media_probe_exact(row.get("media_probe"), EXPECTED_OUTPUT_PROBE, label=f"{arm} media probe")
        _validate_sheet_declaration(
            row.get("all81_sheet"),
            role=f"{arm}-all81-sheet",
            path=f"sheets/{arm}-all81.jpg",
            media_probe=EXPECTED_OUTPUT_PROBE,
            label=f"{arm} sheet",
        )

    declared = [
        runner["plan"], runner["report"], runner["attestation"],
        source["video"], source["all81_sheet"],
        *(item for row in arms for item in (row["output"], row["receipt"], row["all81_sheet"])),
    ]
    _require(
        len({item["path"] for item in declared}) == len(declared)
        and len({item["role"] for item in declared}) == len(declared),
        "postflight authority path/role reuse differs",
    )


def validate_bundle(
    bundle_root: str | Path, *, ffmpeg: str | Path,
    require_publication_marker: bool = True,
) -> dict[str, Any]:
    root = _canonical_existing_path(bundle_root, label="bundle root")
    _require(root.is_dir() and not root.is_symlink(), "bundle root is not a plain directory")
    ffmpeg_path = _canonical_existing_path(ffmpeg, label="bundle validation ffmpeg")
    decoder = _stable_file(
        ffmpeg_path,
        label="bundle validation pinned decoder",
        expected_sha256=EXPECTED_FFMPEG_SHA256,
        expected_size=EXPECTED_FFMPEG_SIZE,
    )
    _require(
        decoder["mode"] & 0o111 != 0,
        "bundle validation decoder executable mode differs",
    )
    _exact_names(
        root,
        {
            "evidence", "media", "receipts", "run", "sheets", "source",
            *({PUBLICATION_MARKER_REL.name} if require_publication_marker else set()),
        },
        label="bundle root",
    )
    _exact_names(root / "evidence", {MANIFEST_REL.name}, label="bundle evidence")
    _exact_names(
        root / "run",
        {PLAN_BUNDLE_REL.name, RUNNER_REPORT_BUNDLE_REL.name, RUNNER_ATTESTATION_BUNDLE_REL.name},
        label="bundle runner documents",
    )
    _exact_names(root / "source", {SOURCE_BUNDLE_REL.name}, label="bundle source")
    _exact_names(
        root / "media", {f"{arm}.mp4" for arm in ARM_ORDER}, label="bundle media"
    )
    _exact_names(
        root / "receipts", {f"{arm}.receipt.json" for arm in ARM_ORDER}, label="bundle receipts"
    )
    _exact_names(
        root / "sheets",
        {"source-exact_original-all81.jpg", *(f"{arm}-all81.jpg" for arm in ARM_ORDER)},
        label="bundle sheets",
    )
    manifest, manifest_file = _load_canonical_json(
        root / MANIFEST_REL,
        label="postflight manifest",
        allowed_modes={0o400},
    )
    _strict_digest(manifest, "manifest_digest", label="postflight manifest")
    _validate_manifest_semantics(manifest)
    runner = manifest.get("runner_documents")
    plan_file = _validate_member(
        root,
        runner.get("plan"),
        label="bundled plan",
        expected_role="launch-plan",
        expected_path=PLAN_BUNDLE_REL.as_posix(),
    )
    report_file = _validate_member(
        root,
        runner.get("report"),
        label="bundled runner report",
        expected_role="runner-report",
        expected_path=RUNNER_REPORT_BUNDLE_REL.as_posix(),
    )
    attestation_file = _validate_member(
        root,
        runner.get("attestation"),
        label="bundled runner attestation",
        expected_role="runner-attestation",
        expected_path=RUNNER_ATTESTATION_BUNDLE_REL.as_posix(),
    )
    plan, _ = _load_canonical_json(
        plan_file["absolute_path"],
        label="bundled plan",
        allowed_modes={0o444},
        expected_sha256=plan_file["sha256"],
    )
    _strict_digest(plan, "plan_digest", label="bundled plan")
    _portable_plan_replay(plan, validation_root=root)
    tasks = plan["tasks"]
    manifest_ffprobe = manifest["render_authority"]["ffprobe"]
    _require(
        manifest_ffprobe.get("sha256") == plan["producer"]["ffprobe_sha256"]
        and manifest_ffprobe.get("size") == plan["producer"]["ffprobe_size"],
        "postflight manifest/plan ffprobe binding differs",
    )
    report, _ = _load_canonical_json(
        report_file["absolute_path"],
        label="bundled runner report",
        allowed_modes={0o444},
        expected_sha256=report_file["sha256"],
    )
    results = _validate_runner_report(report, plan=plan)
    attestation, _ = _load_canonical_json(
        attestation_file["absolute_path"],
        label="bundled runner attestation",
        allowed_modes={0o444},
        expected_sha256=attestation_file["sha256"],
    )
    attested_plan_path = _require_canonical_declared_absolute_path(
        attestation.get("plan", {}).get("path")
        if isinstance(attestation.get("plan"), Mapping) else None,
        label="attested native plan",
    )
    attested_report_path = _require_canonical_declared_absolute_path(
        attestation.get("verified_report", {}).get("path")
        if isinstance(attestation.get("verified_report"), Mapping) else None,
        label="attested native report",
    )
    _require(attested_plan_path != attested_report_path, "attested plan/report path reuse differs")
    attested = _validate_runner_attestation(
        attestation,
        attestation_path=attestation_file["absolute_path"],
        plan_path=attested_plan_path,
        plan_row=plan_file,
        plan=plan,
        report_path=attested_report_path,
        report_row=report_file,
        report=report,
    )
    runner_ffmpeg = attested["physical"]["identities"]["ffmpeg"]
    manifest_ffmpeg = manifest["render_authority"]["ffmpeg"]
    _require(
        runner_ffmpeg.get("sha256") == manifest_ffmpeg["sha256"]
        and type(runner_ffmpeg.get("size")) is int
        and runner_ffmpeg.get("size") == manifest_ffmpeg["size"],
        "runner/postflight ffmpeg authority differs",
    )
    _require(
        runner.get("report_digest") == report.get("report_digest")
        and runner.get("attestation_digest") == attestation.get("attestation_digest"),
        "bundled runner digest cross-link differs",
    )
    source = manifest.get("source")
    sheet_replay_cache: dict[tuple[str, str, str, str], bool] = {}
    source_video = _validate_member(
        root,
        source.get("video"),
        label="bundled source video",
        expected_role="source-video",
        expected_path=SOURCE_BUNDLE_REL.as_posix(),
    )
    source_sheet = _validate_sheet(
        root,
        source.get("all81_sheet"),
        label="bundled source",
        media_probe=EXPECTED_SOURCE_PROBE,
        expected_role="source-all81-sheet",
        expected_path="sheets/source-exact_original-all81.jpg",
        video_path=source_video["absolute_path"],
        video_sha256=source_video["sha256"],
        ffmpeg_path=ffmpeg_path,
        replay_cache=sheet_replay_cache,
    )
    _validate_media_probe_exact(source.get("media_probe"), EXPECTED_SOURCE_PROBE, label="bundled source probe")
    plan_source = plan.get("source_authority")
    _require(
        isinstance(plan_source, Mapping)
        and plan_source.get("sha256") == source_video["sha256"]
        and type(plan_source.get("size")) is int
        and plan_source.get("size") == source_video["size"],
        "bundled source/plan authority cross-link differs",
    )
    native_source_path = _require_canonical_declared_absolute_path(
        plan_source.get("path"), label="native source"
    )
    arms = manifest.get("arms")
    arm_rows: list[dict[str, Any]] = []
    null_receipts: dict[str, dict[str, Any]] = {}
    native_publication_paths: list[Path] = []
    for arm, task_id, row, task, result in zip(ARM_ORDER, TASK_IDS, arms, tasks, results):
        _require(
            isinstance(task, Mapping)
            and task.get("task_id") == task_id
            and task.get("oracle_arm") == arm
            and task.get("arm") == "full644"
            and isinstance(task.get("output"), Mapping),
            f"bundled task/arm closure differs: {arm}",
        )
        task_output = task["output"]
        _require_exact_bool(task_output.get("create_only"), True, label=f"{arm} task create_only")
        native_output = _require_canonical_declared_absolute_path(
            task_output.get("video_path"), label=f"{arm} native output"
        )
        native_receipt = _require_canonical_declared_absolute_path(
            task_output.get("receipt_path"), label=f"{arm} native receipt"
        )
        _require(
            native_receipt == native_output.with_name(native_output.name + ".receipt.json"),
            f"{arm} native output/receipt pairing differs",
        )
        native_publication_paths.extend((native_output, native_receipt))
        output_authority = _validate_member(
            root,
            row["output"],
            label=f"{arm} output",
            expected_role=f"{arm}-output",
            expected_path=f"media/{arm}.mp4",
        )
        receipt_authority = _validate_member(
            root,
            row["receipt"],
            label=f"{arm} receipt",
            expected_role=f"{arm}-receipt",
            expected_path=f"receipts/{arm}.receipt.json",
        )
        receipt, _ = _load_canonical_json(
            receipt_authority["absolute_path"],
            label=f"{arm} bundled native receipt",
            allowed_modes={0o444},
            expected_sha256=receipt_authority["sha256"],
        )
        _strict_digest(receipt, "receipt_digest", label=f"{arm} bundled native receipt")
        try:
            if arm in {"null_before", "null_after"}:
                trajectory_eval.validate_off_inference_receipt(receipt, task, plan["producer"])
            else:
                trajectory_eval.validate_custom_inference_receipt(receipt, task, plan["producer"])
        except trajectory_eval.ObjectTrajectoryEvalError as error:
            raise PostflightError(f"{arm} bundled native receipt replay failed: {error}") from error
        _validate_receipt_task_closure(
            receipt,
            task=task,
            result=result,
            task_result=attested["task_results"][len(arm_rows)],
            receipt_file=receipt_authority["observed"],
        )
        if arm in {"null_before", "null_after"}:
            null_receipts[arm] = receipt
        receipt_output = receipt.get("output")
        _require(
            isinstance(receipt_output, Mapping)
            and receipt.get("receipt_digest") == row.get("receipt_digest") == result.get("receipt_digest")
            and result.get("task_id") == task_id
            and result.get("oracle_arm") == arm
            and result.get("output_path") == str(native_output)
            and result.get("receipt_path") == str(native_receipt)
            and result.get("receipt_file_sha256") == receipt_authority["sha256"]
            and result.get("output_sha256") == output_authority["sha256"]
            and type(result.get("output_size")) is int
            and result.get("output_size") == output_authority["size"]
            and receipt_output.get("path") == str(native_output)
            and receipt_output.get("sha256") == output_authority["sha256"]
            and type(receipt_output.get("size")) is int
            and receipt_output.get("size") == output_authority["size"],
            f"{arm} task/report/receipt/output cross-link differs",
        )
        arm_rows.append(
            {
                **dict(row),
                "output_authority": output_authority,
                "receipt_authority": receipt_authority,
                "receipt": receipt,
                "sheet_authority": _validate_sheet(
                    root,
                    row["all81_sheet"],
                    label=f"{arm} all81",
                    media_probe=EXPECTED_OUTPUT_PROBE,
                    expected_role=f"{arm}-all81-sheet",
                    expected_path=f"sheets/{arm}-all81.jpg",
                    video_path=output_authority["absolute_path"],
                    video_sha256=output_authority["sha256"],
                    ffmpeg_path=ffmpeg_path,
                    replay_cache=sheet_replay_cache,
                ),
            }
        )
    try:
        recomputed_null_envelope = trajectory_eval.validate_null_envelope_receipts(
            null_receipts["null_before"], null_receipts["null_after"]
        )
    except (KeyError, trajectory_eval.ObjectTrajectoryEvalError) as error:
        raise PostflightError(f"bundled null-envelope receipt replay failed: {error}") from error
    _require(
        _json_exact_equal(recomputed_null_envelope, report["null_envelope"]),
        "bundled null envelope is not recomputed from null receipts",
    )
    all_native_paths = [
        native_source_path, attested_plan_path, attested_report_path,
        attested["physical"]["final_artifacts"]["runner_attestation"],
        *native_publication_paths,
    ]
    _require(
        len(set(all_native_paths)) == len(all_native_paths),
        "native authority path reuse/cross-type alias differs",
    )
    if require_publication_marker:
        validate_publication_marker(
            root,
            kind="postflight-bundle",
            authority_role="postflight-manifest",
            authority_path=MANIFEST_REL.as_posix(),
            authority_sha256=manifest_file["sha256"],
            authority_digest=manifest["manifest_digest"],
        )
    return {
        "root": root,
        "manifest": manifest,
        "manifest_path": root / MANIFEST_REL,
        "manifest_sha256": manifest_file["sha256"],
        "manifest_size": manifest_file["size"],
        "plan": plan,
        "plan_authority": plan_file,
        "runner_report": report,
        "runner_report_authority": report_file,
        "runner_attestation": attestation,
        "runner_attestation_authority": attestation_file,
        "source_video_authority": source_video,
        "source_sheet_authority": source_sheet,
        "arms": arm_rows,
    }


def _evaluate_arm_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Run the legacy strict gates without weakening or copying their logic."""

    arm = observation.get("variant")
    _require(arm in ARM_ORDER, f"unknown observation arm: {arm}")
    mapped = dict(observation)
    mapped["variant"] = strict_eval.VARIANT_ORDER[0]
    try:
        result = dict(strict_eval.evaluate_variant(mapped))
    except strict_eval.StrictEvalError as error:
        raise PostflightError(f"strict observation differs for {arm}: {error}") from error
    result["variant"] = arm
    result["conjunction"] = (
        "all81_review_coverage AND dog_identity_retention AND "
        "same_source_bone_reuse AND ordered_source_bone_action"
    )
    return result


def _validate_observation_text_and_timing(observation: Mapping[str, Any]) -> None:
    """Tighten legacy gates without changing their conjunctive semantics."""

    arm = observation.get("variant")
    _require(
        set(observation)
        == {"variant", "review_coverage", "dog_identity", "source_bone", "action_trace"},
        f"observation arm schema differs: {arm}",
    )
    coverage = observation.get("review_coverage")
    _require(
        isinstance(coverage, Mapping)
        and set(coverage)
        == {"all_81_decoded_frames_reviewed", "source_and_output_pair_reviewed", "frame_range", "frame_count"},
        f"review coverage schema differs: {arm}",
    )
    _require(
        type(coverage.get("all_81_decoded_frames_reviewed")) is bool
        and type(coverage.get("source_and_output_pair_reviewed")) is bool,
        f"review coverage boolean type differs: {arm}",
    )
    frame_range = coverage.get("frame_range")
    _require(
        isinstance(frame_range, list)
        and len(frame_range) == 2
        and all(type(item) is int for item in frame_range)
        and frame_range == [0, 80],
        f"review coverage frame range differs: {arm}",
    )
    _require_exact_int(coverage.get("frame_count"), 81, label=f"review frame count {arm}")
    identity = observation.get("dog_identity")
    _require(
        isinstance(identity, Mapping)
        and set(identity) == {"subject_track_id", "identity_switch_observed", "first_mismatch_frame", "cues"},
        f"dog identity schema differs: {arm}",
    )
    switch = identity.get("identity_switch_observed")
    mismatch = identity.get("first_mismatch_frame")
    _require(type(switch) is bool, f"identity switch flag differs: {arm}")
    if switch:
        _require(
            type(mismatch) is int and 0 <= mismatch <= 80,
            f"first mismatch frame/range differs: {arm}",
        )
    else:
        _require(mismatch is None, f"first mismatch frame/flag coupling differs: {arm}")
    cues = identity.get("cues")
    _require(isinstance(cues, list) and len(cues) == len(strict_eval.IDENTITY_CUES), f"identity cues are absent: {arm}")
    for index, cue in enumerate(cues):
        _require(
            isinstance(cue, Mapping)
            and set(cue) == {"name", "source", "output", "preserved"}
            and cue.get("name") == strict_eval.IDENTITY_CUES[index]
            and type(cue.get("preserved")) is bool,
            f"identity cue schema/type differs: {arm}:{index}",
        )
        name = cue.get("name", index)
        _require_nonempty_text(cue.get("source"), label=f"identity cue source {arm}:{name}")
        _require_nonempty_text(cue.get("output"), label=f"identity cue output {arm}:{name}")

    bone = observation.get("source_bone")
    _require(
        isinstance(bone, Mapping)
        and set(bone)
        == {
            "patient_track_id", "input_patient_available", "same_instance_continuity",
            "left_initial_support", "entered_effector_region", "terminal_hold",
            "source_instance_remains_in_background", "duplicate_or_substitute_prop",
            "observed_state",
        },
        f"source bone schema differs: {arm}",
    )
    for field in (
        "input_patient_available", "left_initial_support", "entered_effector_region",
        "terminal_hold", "source_instance_remains_in_background",
    ):
        _require(type(bone.get(field)) is bool, f"source bone boolean type differs: {arm}:{field}")
    _require_nonempty_text(bone.get("observed_state"), label=f"observed state {arm}")
    duplicate = bone.get("duplicate_or_substitute_prop")
    _require(
        isinstance(duplicate, Mapping)
        and set(duplicate) == {"observed", "frame_interval", "description"}
        and type(duplicate.get("observed")) is bool,
        f"duplicate observation schema/type differs: {arm}",
    )
    duplicate_interval = duplicate.get("frame_interval")
    _require(
        duplicate_interval is None
        or (
            isinstance(duplicate_interval, list)
            and len(duplicate_interval) == 2
            and all(type(item) is int for item in duplicate_interval)
            and 0 <= duplicate_interval[0] <= duplicate_interval[1] <= 80
        ),
        f"duplicate interval type/range differs: {arm}",
    )
    _require(
        (duplicate_interval is not None) == duplicate["observed"],
        f"duplicate interval/flag coupling differs: {arm}",
    )
    _require_nonempty_text(
        duplicate.get("description"), label=f"duplicate description {arm}"
    )

    action = observation.get("action_trace")
    _require(
        isinstance(action, Mapping)
        and set(action) == {"patient_track_id", "effector_region_id", "minimum_hold_frames", "stages"},
        f"action trace schema differs: {arm}",
    )
    _require(
        type(action.get("minimum_hold_frames")) is int
        and action.get("minimum_hold_frames") == 10,
        f"minimum_hold_frames must equal 10: {arm}",
    )
    stages = action.get("stages")
    _require(isinstance(stages, list) and len(stages) == len(strict_eval.ACTION_STAGES), f"action stages are absent: {arm}")
    hold: Mapping[str, Any] | None = None
    for index, stage in enumerate(stages):
        _require(
            isinstance(stage, Mapping)
            and set(stage) == {"name", "observed", "frame_interval", "evidence"}
            and stage.get("name") == strict_eval.ACTION_STAGES[index]
            and type(stage.get("observed")) is bool,
            f"action stage schema/type differs: {arm}:{index}",
        )
        name = stage.get("name", index)
        _require_nonempty_text(stage.get("evidence"), label=f"action stage evidence {arm}:{name}")
        interval = stage.get("frame_interval")
        _require(
            interval is None
            or (
                isinstance(interval, list)
                and len(interval) == 2
                and all(type(item) is int for item in interval)
                and 0 <= interval[0] <= interval[1] <= 80
            ),
            f"action stage interval type/range differs: {arm}:{name}",
        )
        _require(
            (interval is not None) == stage["observed"],
            f"action stage interval/flag coupling differs: {arm}:{name}",
        )
        if name == "hold":
            hold = stage
    _require(hold is not None, f"hold stage is absent: {arm}")
    if hold.get("observed") is True:
        interval = hold.get("frame_interval")
        _require(
            isinstance(interval, list)
            and len(interval) == 2
            and all(type(value) is int for value in interval)
            and 0 <= interval[0] <= interval[1] == 80
            and interval[1] - interval[0] + 1 >= 10,
            f"observed hold must end at frame 80 and cover at least 10 frames: {arm}",
        )


def _observation_bindings(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "postflight_manifest_sha256": bundle["manifest_sha256"],
        "postflight_manifest_digest": bundle["manifest"]["manifest_digest"],
        "runner_report_digest": bundle["runner_report"]["report_digest"],
        "runner_attestation_digest": bundle["runner_attestation"]["attestation_digest"],
        "source_video_sha256": bundle["source_video_authority"]["sha256"],
        "source_all81_sheet_sha256": bundle["source_sheet_authority"]["sha256"],
        "arms": {
            row["arm"]: {
                "output_sha256": row["output_authority"]["sha256"],
                "receipt_sha256": row["receipt_authority"]["sha256"],
                "all81_sheet_sha256": row["sheet_authority"]["sha256"],
            }
            for row in bundle["arms"]
        },
    }


def _observation_claim_limits() -> dict[str, bool]:
    return {
        "automatic_identity_metric_claimed": False,
        "automatic_source_object_correspondence_claimed": False,
        "learned_object_centric_method_claimed": False,
        "formal_causal_claim_authorized": False,
        "scientific_claim_authorized": False,
    }


def validate_observations(
    bundle: Mapping[str, Any], observations_path: str | Path,
) -> dict[str, Any]:
    observations, observations_file = _load_canonical_json(
        _canonical_existing_path(observations_path, label="strict observations"),
        label="strict observations",
        allowed_modes={0o444, 0o600},
    )
    _strict_digest(observations, "observations_digest", label="strict observations")
    expected_fields = {
        "schema_version", "status", "case_id", "iid", "instruction", "arm_order",
        "review_method", "evidence_bindings", "claim_limits", "arms",
        "observations_digest",
    }
    _require(set(observations) == expected_fields, "strict observation root schema differs")
    _require(
        observations.get("schema_version") == OBSERVATION_SCHEMA
        and observations.get("status") == OBSERVATION_STATUS
        and observations.get("case_id") == CASE_ID
        and observations.get("iid") == IID
        and observations.get("instruction") == INSTRUCTION
        and observations.get("arm_order") == list(ARM_ORDER),
        "strict observation identity differs",
    )
    method = observations.get("review_method")
    _require(
        isinstance(method, Mapping)
        and set(method)
        == {
            "reviewer_role", "review_design", "randomized_arm_aliases_used",
            "sealed_alias_key_used", "all81_sheet_layout", "decoded_videos_reviewed",
            "structured_after_review", "automatic_output_tracking",
        },
        "strict observation review method schema differs",
    )
    for field, expected in (
        ("randomized_arm_aliases_used", False),
        ("sealed_alias_key_used", False),
        ("decoded_videos_reviewed", True),
        ("structured_after_review", True),
        ("automatic_output_tracking", False),
    ):
        _require_exact_bool(method.get(field), expected, label=f"strict review method.{field}")
    _require(
        method == {
            "reviewer_role": "independent-all81-visual-auditor",
            "review_design": REVIEW_DESIGN,
            "randomized_arm_aliases_used": False,
            "sealed_alias_key_used": False,
            "all81_sheet_layout": "9x9-row-major",
            "decoded_videos_reviewed": True,
            "structured_after_review": True,
            "automatic_output_tracking": False,
        },
        "strict observation review method differs",
    )
    limits = observations.get("claim_limits")
    _require(
        limits == _observation_claim_limits(),
        "strict observation claim limits differ",
    )
    _require_exact_bool_map(
        limits, _observation_claim_limits(), label="strict observation claim limits"
    )
    bindings = observations.get("evidence_bindings")
    expected_bindings = _observation_bindings(bundle)
    _require(bindings == expected_bindings, "strict observation evidence binding differs")
    arms = observations.get("arms")
    _require(isinstance(arms, list) and len(arms) == 5, "strict observation arm count differs")
    for row in arms:
        _require(isinstance(row, Mapping), "strict observation arm is not an object")
    _require([row.get("variant") for row in arms] == list(ARM_ORDER), "strict observation arm order differs")
    for row in arms:
        _validate_observation_text_and_timing(row)
    reports = [_evaluate_arm_observation(row) for row in arms]
    return {
        "observations": observations,
        "observations_path": observations_file["path"],
        "observations_sha256": observations_file["sha256"],
        "observations_size": observations_file["size"],
        "arm_reports": reports,
    }


def build_observation_skeleton(
    *, bundle_root: str | Path, ffmpeg: str | Path,
) -> dict[str, Any]:
    """Bind real evidence into an explicitly unfilled reviewer template.

    The skeleton deliberately cannot pass :func:`validate_observations`.  All
    visual facts remain ``null`` until a human has reviewed the source/output
    pairs and all 81 decoded frames.  Reviewers should write a *new* canonical
    observation file rather than editing this sealed template in place.
    """

    bundle = validate_bundle(bundle_root, ffmpeg=ffmpeg)
    arms: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        arms.append(
            {
                "variant": arm,
                "review_coverage": {
                    "all_81_decoded_frames_reviewed": None,
                    "source_and_output_pair_reviewed": None,
                    "frame_range": [0, 80],
                    "frame_count": 81,
                },
                "dog_identity": {
                    "subject_track_id": "dog#1",
                    "identity_switch_observed": None,
                    "first_mismatch_frame": None,
                    "cues": [
                        {
                            "name": name,
                            "source": None,
                            "output": None,
                            "preserved": None,
                        }
                        for name in strict_eval.IDENTITY_CUES
                    ],
                },
                "source_bone": {
                    "patient_track_id": "bone#1",
                    "input_patient_available": None,
                    "same_instance_continuity": None,
                    "left_initial_support": None,
                    "entered_effector_region": None,
                    "terminal_hold": None,
                    "source_instance_remains_in_background": None,
                    "duplicate_or_substitute_prop": {
                        "observed": None,
                        "frame_interval": None,
                        "description": None,
                    },
                    "observed_state": None,
                },
                "action_trace": {
                    "patient_track_id": "bone#1",
                    "effector_region_id": "dog#1.mouth",
                    "minimum_hold_frames": 10,
                    "stages": [
                        {
                            "name": name,
                            "observed": None,
                            "frame_interval": None,
                            "evidence": None,
                        }
                        for name in strict_eval.ACTION_STAGES
                    ],
                },
            }
        )
    skeleton: dict[str, Any] = {
        "schema_version": OBSERVATION_SKELETON_SCHEMA,
        "status": "UNFILLED_REQUIRES_INDEPENDENT_NONBLIND_ALL81_REVIEW",
        "target_observation_schema": OBSERVATION_SCHEMA,
        "case_id": CASE_ID,
        "iid": IID,
        "instruction": INSTRUCTION,
        "arm_order": list(ARM_ORDER),
        "review_method": {
            "reviewer_role": "independent-all81-visual-auditor",
            "review_design": REVIEW_DESIGN,
            "randomized_arm_aliases_used": False,
            "sealed_alias_key_used": False,
            "all81_sheet_layout": "9x9-row-major",
            "decoded_videos_reviewed": None,
            "structured_after_review": None,
            "automatic_output_tracking": False,
        },
        "evidence_bindings": _observation_bindings(bundle),
        "claim_limits": _observation_claim_limits(),
        "arms": arms,
        "instructions": [
            "This sealed file is a binding skeleton, not a completed visual audit.",
            "Review source/output decoded videos and every frame 0 through 80 before authoring observations.",
            "Copy bindings into a fresh canonical target-schema JSON; replace every null visual fact with an honest observation.",
            "Do not infer identity, bone lineage, or action stages from the prompt or trajectory scaffold.",
        ],
    }
    skeleton["skeleton_digest"] = object_sha256(skeleton)
    return skeleton


def _build_strict_report_from_validated(
    bundle: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    arm_reports = review["arm_reports"]
    by_arm = {row["variant"]: row for row in arm_reports}
    pass_arms = [arm for arm in ARM_ORDER if by_arm[arm]["status"] == "PASS"]
    primary_pass = by_arm[PRIMARY_ARM]["status"] == "PASS"
    gate_module = Path(strict_eval.__file__).resolve(strict=True)
    gate_module_authority = _stable_file(
        gate_module,
        label="strict gate implementation",
        expected_sha256=EXPECTED_STRICT_EVAL_SHA256,
        expected_size=EXPECTED_STRICT_EVAL_SIZE,
        allowed_modes={0o444, 0o644},
    )
    report: dict[str, Any] = {
        "schema_version": STRICT_REPORT_SCHEMA,
        "status": "COMPLETE_FAIL_CLOSED",
        "case_id": CASE_ID,
        "iid": IID,
        "instruction": INSTRUCTION,
        "arm_order": list(ARM_ORDER),
        "primary_arm": PRIMARY_ARM,
        "review_design": REVIEW_DESIGN,
        "primary_canary_status": "PASS" if primary_pass else "FAIL",
        "success_rule": (
            "Each arm passes iff all81 review coverage, source-dog identity, same "
            "source bone#1 reuse/conservation, and ordered approach/contact/grip/lift/hold "
            "all pass; no averaging, cue compensation, or cross-arm compensation."
        ),
        "counts": {
            "arm_count": 5,
            "pass_count": len(pass_arms),
            "fail_count": 5 - len(pass_arms),
        },
        "passing_arms": pass_arms,
        "postflight_authority": {
            "role": "postflight-manifest",
            "path": MANIFEST_REL.as_posix(),
            "sha256": bundle["manifest_sha256"],
            "manifest_digest": bundle["manifest"]["manifest_digest"],
        },
        "observation_authority": {
            "role": OBSERVATION_ROLE,
            "sha256": review["observations_sha256"],
            "size": review["observations_size"],
            "observations_digest": review["observations"]["observations_digest"],
        },
        "gate_implementation": {
            "module": "methods/bernini_action_editing/case01_source_object_strict_eval_v1.py",
            "reused_entry": "evaluate_variant",
            "sha256": gate_module_authority["sha256"],
            "size": gate_module_authority["size"],
            "identity_cues": list(strict_eval.IDENTITY_CUES),
            "action_stages": list(strict_eval.ACTION_STAGES),
        },
        "arms": arm_reports,
        "claim_limits": {
            "engineering_oracle_only": True,
            "diagnostic_success_or_failure_report_only": True,
            "hand_authored_trajectory_scaffold": True,
            "learned_object_centric_method_claim_authorized": False,
            "automatic_output_identity_or_lineage_claimed": False,
            "causal_claim_authorized": False,
            "scientific_claim_authorized": False,
        },
        "limitations": [
            "Output identity and bone lineage are structured by an independent non-blind all-81-frame human audit, not an automatic output tracker.",
            "No randomized arm aliases or sealed alias key are implemented; this report must not be described as blind.",
            "The upstream runner field manual_blind_review_required records a requested review gate, not proof that this actual review was blinded.",
            "The trajectory scaffold is hand-authored and zero-training; this canary cannot establish that slot attention, CTRL-O, or another learned object representation works.",
            "There is one case and one seed per arm; arm contrasts are engineering diagnostics, not a scientific causal estimate.",
        ],
    }
    report["report_digest"] = object_sha256(report)
    return report


def build_strict_report(
    *,
    bundle_root: str | Path,
    observations_path: str | Path,
    ffmpeg: str | Path,
) -> dict[str, Any]:
    bundle = validate_bundle(bundle_root, ffmpeg=ffmpeg)
    review = validate_observations(bundle, observations_path)
    return _build_strict_report_from_validated(bundle, review)


def _write_create_only_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    digest_field: str | None = None,
) -> dict[str, Any]:
    payload = canonical_json_bytes(value) + b"\n"
    row = _write_create_only_bytes(
        path,
        payload,
        final_mode=0o400,
        label=f"create-only JSON {path.name}",
    )
    replay, replay_file = _load_canonical_json(
        path,
        label=f"create-only JSON {path.name}",
        allowed_modes={0o400},
        expected_sha256=row["sha256"],
    )
    _require(
        _json_exact_equal(replay, dict(value)),
        f"create-only JSON replay differs: {path}",
    )
    if digest_field is not None:
        _strict_digest(replay, digest_field, label=f"create-only JSON {path.name}")
    return replay_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--plan", required=True)
    inspect_parser.add_argument("--report", required=True)
    inspect_parser.add_argument("--attestation", required=True)
    produce_parser = subparsers.add_parser("produce")
    produce_parser.add_argument("--plan", required=True)
    produce_parser.add_argument("--report", required=True)
    produce_parser.add_argument("--attestation", required=True)
    produce_parser.add_argument("--bundle", required=True)
    produce_parser.add_argument("--ffmpeg", required=True)
    verify_parser = subparsers.add_parser("verify-bundle")
    verify_parser.add_argument("--bundle", required=True)
    verify_parser.add_argument("--ffmpeg", required=True)
    prepare_parser = subparsers.add_parser("prepare-observations")
    prepare_parser.add_argument("--bundle", required=True)
    prepare_parser.add_argument("--ffmpeg", required=True)
    prepare_parser.add_argument("--output", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--bundle", required=True)
    evaluate_parser.add_argument("--ffmpeg", required=True)
    evaluate_parser.add_argument("--observations", required=True)
    evaluate_parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            value = inspect_completed_run(
                plan_path=args.plan,
                report_path=args.report,
                attestation_path=args.attestation,
            )
            summary = {
                "status": "COMPLETED_RUN_VALID",
                "output_count": len(value["arms"]),
                "report_digest": value["runner_report"]["report_digest"],
                "attestation_digest": value["runner_attestation"]["attestation_digest"],
            }
        elif args.command == "produce":
            _require(args.ffmpeg is not None, "ffmpeg is unavailable")
            value = produce_bundle(
                plan_path=args.plan,
                report_path=args.report,
                attestation_path=args.attestation,
                bundle_root=args.bundle,
                ffmpeg=args.ffmpeg,
            )
            summary = {
                "status": "POSTFLIGHT_BUNDLE_CREATED",
                "bundle": str(value["root"]),
                "manifest_sha256": value["manifest_sha256"],
                "manifest_digest": value["manifest"]["manifest_digest"],
            }
        elif args.command == "verify-bundle":
            value = validate_bundle(args.bundle, ffmpeg=args.ffmpeg)
            summary = {
                "status": "POSTFLIGHT_BUNDLE_VALID",
                "bundle": str(value["root"]),
                "manifest_sha256": value["manifest_sha256"],
                "manifest_digest": value["manifest"]["manifest_digest"],
            }
        elif args.command == "prepare-observations":
            output = Path(args.output).expanduser()
            _require(output.is_absolute(), "observation skeleton output is not absolute")
            skeleton = build_observation_skeleton(
                bundle_root=args.bundle, ffmpeg=args.ffmpeg
            )
            _write_create_only_json(output, skeleton, digest_field="skeleton_digest")
            summary = {
                "status": skeleton["status"],
                "output": str(output),
                "target_observation_schema": skeleton["target_observation_schema"],
                "skeleton_digest": skeleton["skeleton_digest"],
            }
        else:
            output = Path(args.output).expanduser()
            _require(output.is_absolute(), "strict report output is not absolute")
            # Evaluation and every observation binding complete before output
            # parent creation or file reservation.
            report = build_strict_report(
                bundle_root=args.bundle,
                observations_path=args.observations,
                ffmpeg=args.ffmpeg,
            )
            _write_create_only_json(output, report, digest_field="report_digest")
            summary = {
                "status": report["status"],
                "primary_canary_status": report["primary_canary_status"],
                "pass_count": report["counts"]["pass_count"],
                "fail_count": report["counts"]["fail_count"],
                "report_digest": report["report_digest"],
            }
    except (OSError, PostflightError) as error:
        print(json.dumps({"status": "EVIDENCE_INVALID", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
