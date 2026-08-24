#!/usr/bin/env python3
"""Exactly-once submitter template for the formal-v2-r2 full60 CPU review.

The template refuses to run while any ``__REVIEW_*__`` pin remains.  The final
submitter is materialized only after the formal full60 job is terminal success
and its canonical master/submission receipts are available.  It performs a
fresh authoritative sacct observation, then queues one CPU review job directly
with no hold and no dependency.

The resulting job may automatically materialize assessor-private technical
diagnostics and HTML, but only the opaque blind surface may be copied to stage
1 observers before both external response seals exist.  Human observer files
remain blank; this submitter cannot claim event,
identity, seed-selection, training-target, optimizer, or parameter authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
from typing import Any, Sequence


SHA256 = re.compile(r"[0-9a-f]{64}\Z")

EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b"
)
EXPECTED_SOURCE_REVISION = "20c2193954e780e9654347754b1485f3492fbea5"
EXPECTED_ADAPTER_SHA256 = "__REVIEW_ADAPTER_SHA256__"
EXPECTED_LAUNCHER_SHA256 = "__REVIEW_LAUNCHER_SHA256__"
EXPECTED_POSTFLIGHT_SHA256 = (
    "__REVIEW_POSTFLIGHT_SHA256__"
)
EXPECTED_V1_BUILDER_SHA256 = (
    "eac827c2122d8eeb176f4dc2fcf9b95e1fe1c9cbe709c2542a4103424ff8d02b"
)
EXPECTED_FFPROBE_WRAPPER_SHA256 = (
    "06a5698bdaa8069e541e804e2f0e945665b2d16409633aa92d37f76ed7506710"
)
EXPECTED_DIAGNOSTICS_SHA256 = (
    "3658056640b0adc3411c04c029ce99efd5a4d9388be638f659bd8eb472399e0a"
)
EXPECTED_GENERATION_RUNTIME_SHA256 = (
    "3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36"
)
EXPECTED_TOPUP_CONTRACT_SHA256 = (
    "508dde8d995dcc8deeccb47b35be71b9915a86964626383660d8eed952ef5278"
)
EXPECTED_FORMAL_JOB_ID = "__REVIEW_FORMAL_JOB_ID__"
EXPECTED_FORMAL_MASTER_SHA256 = "__REVIEW_FORMAL_MASTER_SHA256__"
EXPECTED_FORMAL_MASTER_DIGEST = "__REVIEW_FORMAL_MASTER_DIGEST__"
EXPECTED_FORMAL_SUBMISSION_SHA256 = "__REVIEW_FORMAL_SUBMISSION_SHA256__"
EXPECTED_FORMAL_SUBMISSION_DIGEST = "__REVIEW_FORMAL_SUBMISSION_DIGEST__"
EXPECTED_COMPUTE_BASH_SHA256 = "__REVIEW_COMPUTE_BASH_SHA256__"
EXPECTED_COMPUTE_BASH_VERSION_STDOUT_SHA256 = (
    "__REVIEW_COMPUTE_BASH_VERSION_STDOUT_SHA256__"
)
# Embedded submitter/release evidence only.  This comma-bearing text must not be
# serialized into Slurm's comma-delimited --export argument.
EXPECTED_COMPUTE_BASH_VERSION_FIRST_LINE = (
    "__REVIEW_COMPUTE_BASH_VERSION_FIRST_LINE__"
)
EXPECTED_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
EXPECTED_FFMPEG_SHA256 = (
    "e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99"
)
EXPECTED_FFMPEG_VERSION_STDOUT_SHA256 = (
    "389368da4bcd4e22d7bf9134f3a8c24dd36027de7d963015230969a87c9e3339"
)
EXPECTED_FFMPEG_VERSION_FIRST_LINE = (
    "ffmpeg version 7.0.2-static https://johnvansickle.com/ffmpeg/  "
    "Copyright (c) 2000-2024 the FFmpeg developers"
)
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
EXPECTED_SOURCE_ARCHIVE = INPUTS / "videoedit-saic-20c2193-methods.tar"
EXPECTED_FORMAL_ROOT = (
    FORMAL_ROOT / "runs/t2v-events-topup-r6-formal-v2-r2-retfd-20260812-r1"
)
EXPECTED_FORMAL_MASTER = (
    EXPECTED_FORMAL_ROOT / "saic-pure-t2v-event-bank-topup-receipt.json"
)
EXPECTED_FORMAL_SUBMISSION = Path(
    str(EXPECTED_FORMAL_ROOT) + ".submission.receipt.json"
)
PACKET_ID = (
    "t2v-events-topup-r6-formal-v2-r2-retfd-20260812-r1-"
    "detached-review-v2-r1"
)
EXPECTED_OUTPUT_ROOT = FORMAL_ROOT / "reviews" / PACKET_ID
EXPECTED_AUTOMATION_RECEIPT = Path(
    str(EXPECTED_OUTPUT_ROOT) + ".automation.receipt.json"
)
EXPECTED_SUBMISSION_RECEIPT = Path(
    str(EXPECTED_OUTPUT_ROOT) + ".submission.receipt.json"
)
EXPECTED_TERMINAL_ADMISSION = Path(
    str(EXPECTED_OUTPUT_ROOT) + ".terminal.admission.receipt.json"
)
EXPECTED_SLURM_LOG_DIR = (
    FORMAL_ROOT
    / "slurm/saic-t2v-topup-r6-formal-v2-r2-retfd-20260812-r1-review-v2-r1"
)
EXPECTED_PYTHON = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
EXPECTED_FFMPEG = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
    "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)
EXPECTED_COMPUTE_BASH = Path("/usr/bin/bash")
EXPECTED_SACCT = Path("/usr/bin/sacct")

ARCHIVE_MEMBERS = {
    "methods/bernini_action_editing/tools/"
    "build_saic_t2v_event_bank_detached_review_v1.py":
        EXPECTED_V1_BUILDER_SHA256,
    "methods/bernini_action_editing/tools/"
    "ffprobe_pyav_exact81_diagnostic_v1.py":
        EXPECTED_FFPROBE_WRAPPER_SHA256,
    "methods/bernini_action_editing/saic_exact81_media_diagnostics_v1.py":
        EXPECTED_DIAGNOSTICS_SHA256,
    "methods/bernini_action_editing/"
    "generate_saic_pure_t2v_event_bank_topup_v2.py":
        EXPECTED_GENERATION_RUNTIME_SHA256,
    "methods/bernini_action_editing/saic_pure_t2v_event_bank_topup_v2.py":
        EXPECTED_TOPUP_CONTRACT_SHA256,
}
RELEASE_AUTHORITY = {
    "scientific": False, "human_review": False, "event_verified": False,
    "identity_preservation_verified": False, "candidate_selection": False,
    "seed_selection": False, "training_target": False, "training": False,
    "optimizer_step": False, "parameter_update": False,
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
SACCT_FIELDS = [
    "JobIDRaw", "State", "ExitCode", "AllocTRES%512", "NodeList",
    "ElapsedRaw", "Start", "End",
]
EXPECTED_FORMAL_ALLOC_TRES = {
    "billing": "32", "cpu": "32", "gres/gpu:mi210": "8",
    "gres/gpu": "8", "mem": "256G", "node": "1",
}
EXPECTED_EVENT_SPEC_RAW_SHA256 = (
    "d693d0784530f007888e2825d15db3db808fdf4f1d111b5d080d968c894ff145"
)
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


def die(message: str) -> None:
    raise SystemExit(f"submit-saic-t2v-topup-review-v2: {message}")


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


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def no_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            die(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def ensure_pins_resolved() -> None:
    values = {
        "adapter SHA": EXPECTED_ADAPTER_SHA256,
        "launcher SHA": EXPECTED_LAUNCHER_SHA256,
        "postflight SHA": EXPECTED_POSTFLIGHT_SHA256,
        "formal JobID": EXPECTED_FORMAL_JOB_ID,
        "formal master SHA": EXPECTED_FORMAL_MASTER_SHA256,
        "formal master digest": EXPECTED_FORMAL_MASTER_DIGEST,
        "formal submission SHA": EXPECTED_FORMAL_SUBMISSION_SHA256,
        "formal submission digest": EXPECTED_FORMAL_SUBMISSION_DIGEST,
        "compute Bash SHA": EXPECTED_COMPUTE_BASH_SHA256,
        "compute Bash version SHA": EXPECTED_COMPUTE_BASH_VERSION_STDOUT_SHA256,
        "compute Bash first line": EXPECTED_COMPUTE_BASH_VERSION_FIRST_LINE,
    }
    if any("__REVIEW_" in value for value in values.values()):
        die("review release retains unresolved placeholder pins")
    for label, value in values.items():
        if label == "formal JobID" or label == "compute Bash first line":
            continue
        if SHA256.fullmatch(value) is None:
            die(f"{label} differs")
    if not EXPECTED_FORMAL_JOB_ID.isdigit():
        die("formal JobID differs")


def exact_dir(path: Path, *, label: str) -> Path:
    info = path.lstat()
    if (
        path.resolve(strict=True) != path
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink < 2
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o022
    ):
        die(f"{label} is not a sealed plain directory")
    return path


def exact_file(path: Path, expected_sha256: str, *, label: str) -> Path:
    if SHA256.fullmatch(expected_sha256) is None:
        die(f"{label} expected SHA differs")
    info = path.lstat()
    if (
        path.resolve(strict=True) != path
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o444
        or sha_file(path) != expected_sha256
    ):
        die(f"{label} identity differs")
    return path


def exact_executable(path: Path, expected_sha256: str, *, label: str) -> Path:
    info = path.lstat()
    if (
        path.resolve(strict=True) != path
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) & 0o022
        or not os.access(path, os.X_OK)
        or sha_file(path) != expected_sha256
    ):
        die(f"{label} executable differs")
    return path


def safe_mutable_dir(path: Path, *, label: str) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o755)
    info = path.lstat()
    if (
        path.resolve(strict=True) != path
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o022
    ):
        die(f"{label} differs")
    return path


def directory_identity(path: Path) -> tuple[int, int]:
    info = path.lstat()
    return info.st_dev, info.st_ino


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        wrote = os.write(descriptor, payload[offset:])
        if wrote <= 0:
            die("submission receipt write stalled")
        offset += wrote


def load_sealed(
    path: Path, expected_sha256: str, expected_digest: str, *, label: str
) -> dict[str, Any]:
    if SHA256.fullmatch(expected_sha256) is None:
        die(f"{label} expected SHA differs")
    descriptor = os.open(
        path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        public = path.lstat()
        if (
            path.resolve(strict=True) != path
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o444
            or not stat.S_ISREG(public.st_mode)
            or stat.S_ISLNK(public.st_mode)
            or (before.st_dev, before.st_ino) != (public.st_dev, public.st_ino)
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
            stat.S_IMODE(item.st_mode), item.st_nlink, item.st_uid,
        )
        if (
            identity(before) != identity(after)
            or identity(after) != identity(public_after)
            or len(raw) != after.st_size
            or sha_bytes(raw) != expected_sha256
        ):
            die(f"{label} bytes changed while retained")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=no_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        die(f"cannot decode {label}: {error}")
    if type(value) is not dict or raw != canonical(value) + b"\n":
        die(f"{label} is not canonical JSON")
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    if (
        declared != expected_digest
        or SHA256.fullmatch(str(declared)) is None
        or sha_bytes(canonical(unsigned)) != declared
    ):
        die(f"{label} object seal differs")
    return value


def validate_release_manifest(
    *, expected_sha256: str, expected_digest: str,
) -> dict[str, Any]:
    manifest = load_sealed(
        EXPECTED_RELEASE_MANIFEST,
        expected_sha256,
        expected_digest,
        label="review release manifest",
    )
    inputs = manifest.get("inputs")
    formal_inputs = manifest.get("formal_inputs")
    executables = manifest.get("executables")
    expected_inputs = {
        "manifest_materializer": EXPECTED_MANIFEST_MATERIALIZER,
        "adapter": EXPECTED_ADAPTER, "launcher": EXPECTED_LAUNCHER,
        "submitter": EXPECTED_SUBMITTER, "postflight": EXPECTED_POSTFLIGHT,
        "hostile": EXPECTED_HOSTILE, "source_archive": EXPECTED_SOURCE_ARCHIVE,
    }
    expected_executables = {
        "python": EXPECTED_PYTHON, "ffmpeg": EXPECTED_FFMPEG,
        "compute_bash": EXPECTED_COMPUTE_BASH, "sacct": EXPECTED_SACCT,
    }
    expected_formal_inputs = {
        "master_receipt": EXPECTED_FORMAL_MASTER,
        "submission_receipt": EXPECTED_FORMAL_SUBMISSION,
    }
    release_info = RELEASE_ROOT.lstat()
    inputs_info = INPUTS.lstat()
    if (
        set(manifest) != {
            "schema_version", "status", "release_root", "inputs",
            "formal_inputs", "executables", "authority", "receipt_digest",
        }
        or manifest.get("schema_version")
        != "saic-t2v-topup-review-v2-release-manifest-v1"
        or manifest.get("status") != "sealed_before_review_submission"
        or manifest.get("release_root") != str(RELEASE_ROOT)
        or manifest.get("authority") != RELEASE_AUTHORITY
        or type(inputs) is not dict
        or set(inputs) != set(expected_inputs)
        or type(executables) is not dict
        or set(executables) != set(expected_executables)
        or type(formal_inputs) is not dict
        or set(formal_inputs) != set(expected_formal_inputs)
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
        or set(INPUTS.iterdir()) != set(expected_inputs.values())
        or set(RELEASE_ROOT.iterdir()) != {INPUTS, EXPECTED_RELEASE_MANIFEST}
    ):
        die("review release manifest closure differs")
    for name, path in expected_inputs.items():
        binding = inputs[name]
        if (
            type(binding) is not dict
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != str(path)
            or SHA256.fullmatch(str(binding.get("sha256"))) is None
        ):
            die(f"review release {name} binding differs")
        exact_file(path, binding["sha256"], label=f"review release {name}")
    for name, path in expected_executables.items():
        binding = executables[name]
        if (
            type(binding) is not dict
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != str(path)
            or SHA256.fullmatch(str(binding.get("sha256"))) is None
        ):
            die(f"review release executable {name} binding differs")
        exact_executable(
            path, binding["sha256"], label=f"review release executable {name}"
        )
    for name, path in expected_formal_inputs.items():
        binding = formal_inputs[name]
        if (
            type(binding) is not dict
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != str(path)
            or SHA256.fullmatch(str(binding.get("sha256"))) is None
        ):
            die(f"review release formal input {name} binding differs")
        exact_file(
            path, binding["sha256"], label=f"review release formal input {name}"
        )
    if (
        inputs["adapter"]["sha256"] != EXPECTED_ADAPTER_SHA256
        or inputs["launcher"]["sha256"] != EXPECTED_LAUNCHER_SHA256
        or inputs["postflight"]["sha256"] != EXPECTED_POSTFLIGHT_SHA256
        or inputs["source_archive"]["sha256"] != EXPECTED_SOURCE_ARCHIVE_SHA256
        or formal_inputs["master_receipt"]["sha256"]
        != EXPECTED_FORMAL_MASTER_SHA256
        or formal_inputs["submission_receipt"]["sha256"]
        != EXPECTED_FORMAL_SUBMISSION_SHA256
        or executables["python"]["sha256"] != EXPECTED_PYTHON_SHA256
        or executables["ffmpeg"]["sha256"] != EXPECTED_FFMPEG_SHA256
        or executables["compute_bash"]["sha256"] != EXPECTED_COMPUTE_BASH_SHA256
        or executables["sacct"]["sha256"] != EXPECTED_SACCT_SHA256
    ):
        die("review release hard pins differ")
    return manifest


def validate_archive(path: Path) -> None:
    values: dict[str, bytes] = {}
    with tarfile.open(path, "r:*") as handle:
        if handle.pax_headers.get("comment") != EXPECTED_SOURCE_REVISION:
            die("source archive revision differs")
        for member in handle.getmembers():
            normalized = PurePosixPath(member.name)
            if normalized.is_absolute() or ".." in normalized.parts:
                die("source archive member escaped")
            name = normalized.as_posix()
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                die("source archive contains non-plain entry")
            if name in ARCHIVE_MEMBERS:
                if name in values or not member.isfile():
                    die("source archive required member differs")
                extracted = handle.extractfile(member)
                if extracted is None:
                    die("source archive member is unreadable")
                values[name] = extracted.read()
    if set(values) != set(ARCHIVE_MEMBERS):
        die("source archive review closure differs")
    for name, expected_sha in ARCHIVE_MEMBERS.items():
        if sha_bytes(values[name]) != expected_sha:
            die(f"source archive member SHA differs: {name}")


def validate_formal_receipts() -> tuple[dict[str, Any], dict[str, Any]]:
    master = load_sealed(
        EXPECTED_FORMAL_MASTER,
        EXPECTED_FORMAL_MASTER_SHA256,
        EXPECTED_FORMAL_MASTER_DIGEST,
        label="formal master receipt",
    )
    submission = load_sealed(
        EXPECTED_FORMAL_SUBMISSION,
        EXPECTED_FORMAL_SUBMISSION_SHA256,
        EXPECTED_FORMAL_SUBMISSION_DIGEST,
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
        != "623a7ed8a2ce2d327247c541b59aa2d39f1fbfe4a480f7351d042c7ef7a47927"
        or master.get("base_v1_spec_content_sha256")
        != "3920d5c121b75c6bbf984c24440c9773dfb49006778c61a671ae50963bb5456a"
        or master.get("source_manifest_content_sha256")
        != "9c2a3d6841951ea0ed050dc230630a1176460e25a979ec199eab575ad22f3c6f"
        or master.get("topology")
        != "two_concurrent_world4_sp4_groups_on_one_8gpu_node"
        or master.get("attempt_count") != 60
        or master.get("row_count") != 8
        or master.get("seed_cell_count") != 20
        or master.get("branch_order")
        != ["incomplete", "camera_only", "appearance_only"]
        or master.get("merged_branch_order") != [
            "forward", "reverse", "noop", "incomplete", "camera_only",
            "appearance_only",
        ]
        or master.get("six_branch_spec_merge_cell_count") != 20
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
        die("formal master contract/cardinality differs")
    for field in (
        "detached_full81_event_review_complete",
        "event_verified",
        "identity_preservation_verified",
        "seed_selection_authorized",
        "training_target_authorized",
        "optimizer_or_parameter_update_authorized",
    ):
        if master.get(field) is not False:
            die(f"formal master unexpectedly authorizes {field}")
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
    submitted = submission.get("submitted_job", {})
    request = submission.get("request", {})
    outputs = submission.get("outputs", {})
    boundary = submission.get("submission_boundary", {})
    inputs = submission.get("inputs", {})
    if (
        set(submission) != FORMAL_SUBMISSION_FIELDS
        or submission.get("schema_version")
        != "saic-t2v-topup-r6-formal-v2-r2-submission-v1"
        or submission.get("status") != "submitted"
        or submission.get("submission_success") is not True
        or submission.get("job_success") is not None
        or set(submitted) != {
            "job_id", "cluster", "stdout_sha256", "stderr_sha256",
        }
        or submitted.get("job_id") != EXPECTED_FORMAL_JOB_ID
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
        or request != {
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
        or set(inputs) != FORMAL_SUBMISSION_INPUT_FIELDS
        or inputs.get("event_spec_sha256") != EXPECTED_EVENT_SPEC_RAW_SHA256
        or inputs.get("event_spec_sha256") != master.get("root_spec_raw_sha256")
        or inputs.get("source_manifest_sha256")
        != "899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9"
        or outputs != {
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
        die("formal submission receipt binding differs")
    validate_formal_three_gate_bundle(submission)
    return master, submission


def validate_formal_three_gate_bundle(submission: dict[str, Any]) -> None:
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


def observe_formal_sacct() -> dict[str, Any]:
    exact_executable(EXPECTED_SACCT, EXPECTED_SACCT_SHA256, label="sacct")
    argv = [
        str(EXPECTED_SACCT), "-j", EXPECTED_FORMAL_JOB_ID, "-X",
        "--noheader", "-n", "-P", "-o", ",".join(SACCT_FIELDS),
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
        die("formal sacct stdout is not ASCII")
    lines = stdout.splitlines()
    fields = lines[0].split("|") if len(lines) == 1 else []
    keys = [field.split("%", 1)[0] for field in SACCT_FIELDS]
    row = dict(zip(keys, fields, strict=True)) if len(fields) == len(keys) else {}
    alloc_tokens: dict[str, str] = {}
    for token in str(row.get("AllocTRES", "")).split(","):
        if "=" not in token:
            die("formal AllocTRES token differs")
        key, value = token.split("=", 1)
        if not key or key in alloc_tokens:
            die("formal AllocTRES closure differs")
        alloc_tokens[key] = value
    if (
        completed.returncode != 0
        or completed.stderr
        or len(lines) != 1
        or row.get("JobIDRaw") != EXPECTED_FORMAL_JOB_ID
        or row.get("State") != "COMPLETED"
        or row.get("ExitCode") != "0:0"
        or alloc_tokens != EXPECTED_FORMAL_ALLOC_TRES
        or not row.get("NodeList")
        or row.get("NodeList") in {"None assigned", "Unknown"}
        or not str(row.get("ElapsedRaw", "")).isdigit()
        or not row.get("Start")
        or row.get("Start") == "Unknown"
        or not row.get("End")
        or row.get("End") == "Unknown"
    ):
        die("formal job is not an exact terminal success")
    return {
        "executable": str(EXPECTED_SACCT),
        "executable_sha256": EXPECTED_SACCT_SHA256,
        "argv": argv,
        "query_fields": SACCT_FIELDS,
        "returncode": completed.returncode,
        "stdout_sha256": sha_bytes(completed.stdout),
        "stderr_sha256": sha_bytes(completed.stderr),
        "parsed_row": row,
        "exact_single_row": True,
        "observation_phase": "review_submitter_after_formal_terminal_before_only_sbatch",
    }


def validate_runtime_executables() -> None:
    exact_executable(EXPECTED_PYTHON, EXPECTED_PYTHON_SHA256, label="Python")
    exact_executable(EXPECTED_FFMPEG, EXPECTED_FFMPEG_SHA256, label="ffmpeg")
    exact_executable(
        EXPECTED_COMPUTE_BASH, EXPECTED_COMPUTE_BASH_SHA256, label="compute Bash"
    )
    for executable, expected_sha, expected_first_line, label in (
        (
            EXPECTED_FFMPEG,
            EXPECTED_FFMPEG_VERSION_STDOUT_SHA256,
            EXPECTED_FFMPEG_VERSION_FIRST_LINE,
            "ffmpeg",
        ),
        (
            EXPECTED_COMPUTE_BASH,
            EXPECTED_COMPUTE_BASH_VERSION_STDOUT_SHA256,
            EXPECTED_COMPUTE_BASH_VERSION_FIRST_LINE,
            "compute Bash",
        ),
    ):
        completed = subprocess.run(
            [str(executable), "-version" if label == "ffmpeg" else "--version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
        try:
            lines = completed.stdout.decode("ascii").splitlines()
        except UnicodeDecodeError:
            die(f"{label} version output is not ASCII")
        if (
            completed.returncode != 0
            or completed.stderr
            or sha_bytes(completed.stdout) != expected_sha
            or not lines
            or lines[0] != expected_first_line
        ):
            die(f"{label} version contract differs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", required=True)
    parser.add_argument("--release-manifest-sha256", required=True)
    parser.add_argument("--release-manifest-digest", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ensure_pins_resolved()
    if (
        Path(__file__) != EXPECTED_SUBMITTER
        or Path(__file__).resolve(strict=True) != EXPECTED_SUBMITTER
    ):
        die("review submitter execution path differs")
    exact_dir(RELEASE_ROOT, label="review release root")
    exact_dir(INPUTS, label="review release inputs")
    if Path(args.release_manifest) != EXPECTED_RELEASE_MANIFEST:
        die("review release manifest path differs")
    release_manifest = validate_release_manifest(
        expected_sha256=args.release_manifest_sha256,
        expected_digest=args.release_manifest_digest,
    )
    validate_archive(EXPECTED_SOURCE_ARCHIVE)
    master, formal_submission = validate_formal_receipts()
    formal_sacct = observe_formal_sacct()
    validate_runtime_executables()

    output_parent = safe_mutable_dir(EXPECTED_OUTPUT_ROOT.parent, label="review parent")
    slurm_log_dir = safe_mutable_dir(EXPECTED_SLURM_LOG_DIR, label="review log dir")
    if any(slurm_log_dir.iterdir()):
        die("review Slurm log namespace is not fresh")
    if (
        EXPECTED_OUTPUT_ROOT.exists()
        or EXPECTED_OUTPUT_ROOT.is_symlink()
        or EXPECTED_AUTOMATION_RECEIPT.exists()
        or EXPECTED_AUTOMATION_RECEIPT.is_symlink()
        or EXPECTED_SUBMISSION_RECEIPT.exists()
        or EXPECTED_SUBMISSION_RECEIPT.is_symlink()
        or EXPECTED_TERMINAL_ADMISSION.exists()
        or EXPECTED_TERMINAL_ADMISSION.is_symlink()
    ):
        die("review output/automation/submission/admission targets are not jointly fresh")
    output_parent_identity = directory_identity(output_parent)
    slurm_log_identity = directory_identity(slurm_log_dir)
    pre_reservation_siblings = {entry.name for entry in os.scandir(output_parent)}

    exports = {
        "SAIC_T2V_TOPUP_REVIEW_RELEASE_ROOT": str(RELEASE_ROOT),
        "SAIC_T2V_TOPUP_REVIEW_SOURCE_ARCHIVE": str(EXPECTED_SOURCE_ARCHIVE),
        "SAIC_T2V_TOPUP_REVIEW_SOURCE_ARCHIVE_SHA256":
            EXPECTED_SOURCE_ARCHIVE_SHA256,
        "SAIC_T2V_TOPUP_REVIEW_SOURCE_REVISION": EXPECTED_SOURCE_REVISION,
        "SAIC_T2V_TOPUP_REVIEW_ADAPTER": str(EXPECTED_ADAPTER),
        "SAIC_T2V_TOPUP_REVIEW_ADAPTER_SHA256": EXPECTED_ADAPTER_SHA256,
        "SAIC_T2V_TOPUP_REVIEW_INPUT_ROOT": str(EXPECTED_FORMAL_ROOT),
        "SAIC_T2V_TOPUP_REVIEW_FORMAL_MASTER": str(EXPECTED_FORMAL_MASTER),
        "SAIC_T2V_TOPUP_REVIEW_FORMAL_MASTER_SHA256":
            EXPECTED_FORMAL_MASTER_SHA256,
        "SAIC_T2V_TOPUP_REVIEW_FORMAL_MASTER_DIGEST":
            EXPECTED_FORMAL_MASTER_DIGEST,
        "SAIC_T2V_TOPUP_REVIEW_FORMAL_SUBMISSION":
            str(EXPECTED_FORMAL_SUBMISSION),
        "SAIC_T2V_TOPUP_REVIEW_FORMAL_SUBMISSION_SHA256":
            EXPECTED_FORMAL_SUBMISSION_SHA256,
        "SAIC_T2V_TOPUP_REVIEW_FORMAL_SUBMISSION_DIGEST":
            EXPECTED_FORMAL_SUBMISSION_DIGEST,
        "SAIC_T2V_TOPUP_REVIEW_FORMAL_JOB_ID": EXPECTED_FORMAL_JOB_ID,
        "SAIC_T2V_TOPUP_REVIEW_OUTPUT_ROOT": str(EXPECTED_OUTPUT_ROOT),
        "SAIC_T2V_TOPUP_REVIEW_AUTOMATION_RECEIPT":
            str(EXPECTED_AUTOMATION_RECEIPT),
        "SAIC_T2V_TOPUP_REVIEW_PYTHON_BIN": str(EXPECTED_PYTHON),
        "SAIC_T2V_TOPUP_REVIEW_PYTHON_SHA256": EXPECTED_PYTHON_SHA256,
        "SAIC_T2V_TOPUP_REVIEW_FFMPEG_BIN": str(EXPECTED_FFMPEG),
        "SAIC_T2V_TOPUP_REVIEW_FFMPEG_SHA256": EXPECTED_FFMPEG_SHA256,
        "SAIC_T2V_TOPUP_REVIEW_FFPROBE_WRAPPER_SHA256":
            EXPECTED_FFPROBE_WRAPPER_SHA256,
        "SAIC_T2V_TOPUP_REVIEW_COMPUTE_BASH": str(EXPECTED_COMPUTE_BASH),
        "SAIC_T2V_TOPUP_REVIEW_COMPUTE_BASH_SHA256":
            EXPECTED_COMPUTE_BASH_SHA256,
        "SAIC_T2V_TOPUP_REVIEW_COMPUTE_BASH_VERSION_STDOUT_SHA256":
            EXPECTED_COMPUTE_BASH_VERSION_STDOUT_SHA256,
        "SAIC_T2V_TOPUP_REVIEW_WORKERS": "16",
    }
    if list(exports) != EXPORT_NAMES:
        die("review export name/order differs")
    if any("," in key or "," in value or "\n" in key or "\n" in value for key, value in exports.items()):
        die("review sbatch export transport differs")

    if not Path("/proc/self/fd").is_dir():
        die("Linux retained-fd launcher transport is unavailable")
    launcher_descriptor = os.open(
        EXPECTED_LAUNCHER, os.O_RDONLY | os.O_NOFOLLOW
    )
    launcher_info = os.fstat(launcher_descriptor)
    launcher_identity = (launcher_info.st_dev, launcher_info.st_ino)
    if (
        not stat.S_ISREG(launcher_info.st_mode)
        or launcher_info.st_nlink != 1
        or stat.S_IMODE(launcher_info.st_mode) != 0o444
        or sha_file(Path(f"/proc/self/fd/{launcher_descriptor}"))
        != EXPECTED_LAUNCHER_SHA256
    ):
        os.close(launcher_descriptor)
        die("retained review launcher differs")

    output_parent_descriptor = os.open(
        output_parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    opened_output_parent = os.fstat(output_parent_descriptor)
    if (
        (opened_output_parent.st_dev, opened_output_parent.st_ino)
        != output_parent_identity
    ):
        os.close(launcher_descriptor)
        os.close(output_parent_descriptor)
        die("retained review output parent differs")
    receipt_descriptor = os.open(
        EXPECTED_SUBMISSION_RECEIPT.name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=output_parent_descriptor,
    )
    reservation = os.fstat(receipt_descriptor)
    reservation_identity = (reservation.st_dev, reservation.st_ino)
    if (
        not stat.S_ISREG(reservation.st_mode)
        or reservation.st_nlink != 1
        or reservation.st_uid != os.getuid()
        or stat.S_IMODE(reservation.st_mode) != 0o600
    ):
        os.close(launcher_descriptor)
        os.close(receipt_descriptor)
        os.close(output_parent_descriptor)
        die("review submission reservation differs")
    provisional = {
        "schema_version": "saic-t2v-topup-detached-review-v2-submission-v1",
        "status": "reserved_before_sbatch_not_submission_success",
        "submission_success": False,
        "job_success": None,
        "formal_job_id": EXPECTED_FORMAL_JOB_ID,
        "formal_master_sha256": EXPECTED_FORMAL_MASTER_SHA256,
        "launcher_sha256": EXPECTED_LAUNCHER_SHA256,
        "adapter_sha256": EXPECTED_ADAPTER_SHA256,
    }
    write_all(receipt_descriptor, canonical(provisional) + b"\n")
    os.fsync(receipt_descriptor)
    os.fsync(output_parent_descriptor)
    current_siblings = {entry.name for entry in os.scandir(output_parent)}
    retained_before = os.fstat(launcher_descriptor)
    public_before = EXPECTED_SUBMISSION_RECEIPT.lstat()
    if (
        current_siblings != pre_reservation_siblings | {EXPECTED_SUBMISSION_RECEIPT.name}
        or EXPECTED_OUTPUT_ROOT.exists()
        or EXPECTED_AUTOMATION_RECEIPT.exists()
        or EXPECTED_TERMINAL_ADMISSION.exists()
        or directory_identity(output_parent) != output_parent_identity
        or directory_identity(slurm_log_dir) != slurm_log_identity
        or not stat.S_ISREG(public_before.st_mode)
        or stat.S_ISLNK(public_before.st_mode)
        or public_before.st_nlink != 1
        or public_before.st_uid != os.getuid()
        or stat.S_IMODE(public_before.st_mode) != 0o600
        or (public_before.st_dev, public_before.st_ino) != reservation_identity
        or not stat.S_ISREG(retained_before.st_mode)
        or retained_before.st_nlink != 1
        or retained_before.st_uid != os.getuid()
        or stat.S_IMODE(retained_before.st_mode) != 0o444
        or (retained_before.st_dev, retained_before.st_ino) != launcher_identity
    ):
        os.close(launcher_descriptor)
        os.close(receipt_descriptor)
        die("review pre-sbatch boundary differs")

    command = [
        "/usr/bin/sbatch",
        "--parsable",
        f"--output={slurm_log_dir}/saic-t2v-topup-review-v2-%j.out",
        f"--error={slurm_log_dir}/saic-t2v-topup-review-v2-%j.err",
        "--export=NONE," + ",".join(
            f"{name}={value}" for name, value in exports.items()
        ),
        f"/proc/self/fd/{launcher_descriptor}",
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
            pass_fds=(launcher_descriptor,),
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
    finally:
        try:
            os.close(launcher_descriptor)
        except OSError:
            pass
    try:
        stdout_text = completed.stdout.decode("ascii")
    except UnicodeDecodeError:
        os.close(receipt_descriptor)
        die("sbatch stdout is not ASCII")
    match = re.fullmatch(r"([0-9]+)(?:;([^\n;]+))?\n?", stdout_text)
    if completed.returncode != 0 or match is None:
        os.close(receipt_descriptor)
        die(
            "sbatch failed; 0600 reservation retained: "
            f"exit={completed.returncode} stderr_sha256={sha_bytes(completed.stderr)}"
        )
    review_job_id = match.group(1)

    submitted_provisional = {
        **provisional,
        "status": "sbatch_returned_job_id_receipt_not_terminal",
        "submitted_job_id": review_job_id,
        "sbatch_stdout_sha256": sha_bytes(completed.stdout),
        "sbatch_stderr_sha256": sha_bytes(completed.stderr),
    }
    staged = canonical(submitted_provisional) + b"\n"
    os.lseek(receipt_descriptor, 0, os.SEEK_SET)
    os.ftruncate(receipt_descriptor, 0)
    write_all(receipt_descriptor, staged)
    os.fsync(receipt_descriptor)

    public = EXPECTED_SUBMISSION_RECEIPT.lstat()
    if (
        directory_identity(output_parent) != output_parent_identity
        or directory_identity(slurm_log_dir) != slurm_log_identity
        or EXPECTED_SUBMISSION_RECEIPT.resolve(strict=True)
        != EXPECTED_SUBMISSION_RECEIPT
        or not stat.S_ISREG(public.st_mode)
        or stat.S_ISLNK(public.st_mode)
        or public.st_nlink != 1
        or public.st_uid != os.getuid()
        or stat.S_IMODE(public.st_mode) != 0o600
        or (public.st_dev, public.st_ino) != reservation_identity
    ):
        os.close(receipt_descriptor)
        die("review post-sbatch reservation differs")

    core = {
        "schema_version": "saic-t2v-topup-detached-review-v2-submission-v1",
        "status": "submitted",
        "submission_success": True,
        "job_success": None,
        "submitted_job": {
            "job_id": review_job_id,
            "cluster": match.group(2),
            "stdout_sha256": sha_bytes(completed.stdout),
            "stderr_sha256": sha_bytes(completed.stderr),
        },
        "request": {
            "job_name": "saic-t2v-topup-review-v2",
            "partition": "faculty",
            "qos": "bgqos",
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 32,
            "memory": "192G",
            "walltime": "08:00:00",
            "gpu_resource_requested": None,
            "candidate_count": 60,
            "technical_diagnostic_count": 60,
            "hold": False,
            "dependency": None,
        },
        "submission_boundary": {
            "environment_replaced": True,
            "exact_job_export_names": list(exports),
            "comma_bearing_compute_bash_first_line_not_exported": True,
            "export_all": False,
            "reservation_created_before_sbatch": True,
            "same_inode_retained": True,
            "launcher_submitted_from_retained_fd": True,
            "retained_launcher_fd": launcher_descriptor,
            "reservation_device": reservation.st_dev,
            "reservation_inode": reservation.st_ino,
            "success_mode": "0444",
        },
        "formal_terminal_input_bundle": {
            "bundle_is_not_a_separate_formal_terminal_admission": True,
            "job_id": EXPECTED_FORMAL_JOB_ID,
            "master_receipt": str(EXPECTED_FORMAL_MASTER),
            "master_receipt_sha256": EXPECTED_FORMAL_MASTER_SHA256,
            "master_receipt_digest": EXPECTED_FORMAL_MASTER_DIGEST,
            "submission_receipt": str(EXPECTED_FORMAL_SUBMISSION),
            "submission_receipt_sha256": EXPECTED_FORMAL_SUBMISSION_SHA256,
            "submission_receipt_digest": EXPECTED_FORMAL_SUBMISSION_DIGEST,
            "master_attempt_count": master["attempt_count"],
            "master_branch_order": master["branch_order"],
            "live_sacct_observation": formal_sacct,
        },
        "inputs": {
            "release_root": str(RELEASE_ROOT),
            "launcher": str(EXPECTED_LAUNCHER),
            "launcher_sha256": EXPECTED_LAUNCHER_SHA256,
            "adapter": str(EXPECTED_ADAPTER),
            "adapter_sha256": EXPECTED_ADAPTER_SHA256,
            "terminal_postflight": str(EXPECTED_POSTFLIGHT),
            "terminal_postflight_sha256": EXPECTED_POSTFLIGHT_SHA256,
            "source_archive": str(EXPECTED_SOURCE_ARCHIVE),
            "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
            "source_revision": EXPECTED_SOURCE_REVISION,
            "formal_submission_receipt_digest": formal_submission["receipt_digest"],
            "python": str(EXPECTED_PYTHON),
            "python_sha256": EXPECTED_PYTHON_SHA256,
            "ffmpeg": str(EXPECTED_FFMPEG),
            "ffmpeg_sha256": EXPECTED_FFMPEG_SHA256,
            "ffprobe_wrapper_sha256": EXPECTED_FFPROBE_WRAPPER_SHA256,
            "compute_bash": str(EXPECTED_COMPUTE_BASH),
            "compute_bash_sha256": EXPECTED_COMPUTE_BASH_SHA256,
            "compute_bash_version_stdout_sha256":
                EXPECTED_COMPUTE_BASH_VERSION_STDOUT_SHA256,
            "compute_bash_version_first_line":
                EXPECTED_COMPUTE_BASH_VERSION_FIRST_LINE,
            "release_manifest": str(EXPECTED_RELEASE_MANIFEST),
            "release_manifest_sha256": args.release_manifest_sha256,
            "release_manifest_digest": args.release_manifest_digest,
            "submitter": str(EXPECTED_SUBMITTER),
            "submitter_sha256":
                release_manifest["inputs"]["submitter"]["sha256"],
            "hostile": str(EXPECTED_HOSTILE),
            "hostile_sha256": release_manifest["inputs"]["hostile"]["sha256"],
        },
        "outputs": {
            "packet_root": str(EXPECTED_OUTPUT_ROOT),
            "submission_receipt": str(EXPECTED_SUBMISSION_RECEIPT),
            "automation_receipt": str(EXPECTED_AUTOMATION_RECEIPT),
            "terminal_admission": str(EXPECTED_TERMINAL_ADMISSION),
            "technical_diagnostics_glob":
                str(EXPECTED_OUTPUT_ROOT / "diagnostics/*.json"),
            "technical_html": str(EXPECTED_OUTPUT_ROOT / "index.html"),
            "blind_human_review_html":
                str(EXPECTED_OUTPUT_ROOT / "blind-review.html"),
            "observer_template_glob":
                str(EXPECTED_OUTPUT_ROOT / "observer-templates/*.json"),
            "slurm_log_dir": str(EXPECTED_SLURM_LOG_DIR),
            "fresh_before_submission": True,
        },
        "automatic_backfill_contract": {
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
        },
        "authority": {
            "technical_diagnostic_execution_authorized": True,
            "machine_diagnostics_have_semantic_authority": False,
            "human_review_claimed": False,
            "event_verified": False,
            "identity_preservation_verified": False,
            "seed_selection_allowed": False,
            "training_target_allowed": False,
            "training_allowed": False,
            "optimizer_step_allowed": False,
            "parameter_update_allowed": False,
            "review_job_success_claimed": False,
        },
    }
    value = {**core, "receipt_digest": sha_bytes(canonical(core))}
    payload = canonical(value) + b"\n"
    os.lseek(receipt_descriptor, 0, os.SEEK_SET)
    os.ftruncate(receipt_descriptor, 0)
    write_all(receipt_descriptor, payload)
    os.fsync(receipt_descriptor)
    os.lseek(receipt_descriptor, 0, os.SEEK_SET)
    if os.read(receipt_descriptor, len(payload) + 1) != payload:
        os.close(receipt_descriptor)
        die("review submission receipt reread differs")
    public = EXPECTED_SUBMISSION_RECEIPT.lstat()
    if (
        not stat.S_ISREG(public.st_mode)
        or stat.S_ISLNK(public.st_mode)
        or public.st_nlink != 1
        or stat.S_IMODE(public.st_mode) != 0o600
        or public.st_size != len(payload)
        or (public.st_dev, public.st_ino) != reservation_identity
    ):
        os.close(receipt_descriptor)
        die("review submission receipt publication differs")
    os.fchmod(receipt_descriptor, 0o444)
    os.fsync(receipt_descriptor)
    os.lseek(receipt_descriptor, 0, os.SEEK_SET)
    if os.read(receipt_descriptor, len(payload) + 1) != payload:
        os.close(receipt_descriptor)
        os.close(output_parent_descriptor)
        die("sealed review submission receipt same-FD reread differs")
    sealed = os.fstat(receipt_descriptor)
    public = os.stat(
        EXPECTED_SUBMISSION_RECEIPT.name,
        dir_fd=output_parent_descriptor,
        follow_symlinks=False,
    )
    public_path = EXPECTED_SUBMISSION_RECEIPT.lstat()
    if (
        not stat.S_ISREG(sealed.st_mode)
        or sealed.st_nlink != 1
        or sealed.st_uid != os.getuid()
        or stat.S_IMODE(sealed.st_mode) != 0o444
        or sealed.st_size != len(payload)
        or (sealed.st_dev, sealed.st_ino) != reservation_identity
        or (public.st_dev, public.st_ino) != reservation_identity
        or (public_path.st_dev, public_path.st_ino) != reservation_identity
        or public.st_uid != os.getuid()
        or public_path.st_uid != os.getuid()
        or stat.S_IMODE(public.st_mode) != 0o444
        or stat.S_IMODE(public_path.st_mode) != 0o444
        or public.st_size != len(payload)
        or public_path.st_size != len(payload)
    ):
        os.close(receipt_descriptor)
        die("review sealed submission receipt differs")
    os.fsync(output_parent_descriptor)
    os.close(receipt_descriptor)
    os.close(output_parent_descriptor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
