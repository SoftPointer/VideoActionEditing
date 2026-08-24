#!/usr/bin/env python3
"""Admit a terminal retained-FD WORLD8 canary after authoritative Slurm accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types
from typing import Any, Sequence


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_WRAPPER_SHA256 = "fb3f1ac4e8f87f4833d45ff6be184ae863df0becbb8998dced22ed28ba240bd2"
EXPECTED_PAYLOAD_SHA256 = "96335bf5f5a0896fdbf7e88dbe774bd5f4c03e0e44ecab96757834b9413d6750"
EXPECTED_GUARD_SHA256 = "1a38b2ac18f46e818b1596db884b883dff8e0612fcd5b0cf1ed78aca377ac965"
EXPECTED_RUNTIME_SHA256 = "3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36"
EXPECTED_SOURCE_ARCHIVE_SHA256 = "3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b"
EXPECTED_ARCHIVE_MEMBER_MANIFEST_SHA256 = "1f3c8af23f5b4d416cea04476900c5d479ad3000338746e11f0e655b995b0fcc"
EXPECTED_RUNTIME_ORIGIN_MANIFEST_SHA256 = "2e9360581b21b56e6998e1e5db8df98e4cc66acf95fbb7819baffd1161eb98ba"
EXPECTED_PROBE_VALIDATOR_SHA256 = "3b5e78a0815fbfdee0404293ad800c640133deacc6b1bfafec12a02ca40ef85b"
PROBE_ADMISSION = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/canaries/"
    "compute-bash-retained-fd-probe-8283e73d-r1/probe-admission.json"
)
PROBE_ADMISSION_SHA256 = "d51ebf1f894d63483943042faaa2c6ccbf812c0f93769980084bae72f8ab84d8"
PROBE_ADMISSION_DIGEST = "a37e44c12f935a4f4e11ab08364b019799809fd13299af6459ca6797a8333fb7"
EXPECTED_PYTHON = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
EXPECTED_PYTHON_SHA256 = "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
EXPECTED_SACCT = Path("/usr/bin/sacct")
EXPECTED_SACCT_SHA256 = "fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809"
)
STEM = "saic-formal-v2-retained-fd-world8-canary-96335bf5-fb3f1ac4-r10"
EXPECTED_OUTPUT_PARENT = ROOT / "canaries" / STEM
EXPECTED_SUBMISSION_RECEIPT = EXPECTED_OUTPUT_PARENT / "submission-receipt.json"
EXPECTED_ADMISSION = EXPECTED_OUTPUT_PARENT / "canary-admission.json"
EXPECTED_LOG_DIR = ROOT / "slurm" / STEM
EXPECTED_RELEASE = ROOT / "releases" / STEM
EXPECTED_INPUTS = EXPECTED_RELEASE / "inputs"
EXPECTED_WRAPPER = EXPECTED_INPUTS / "auh_canary_saic_formal_v2_retained_fd_world8_v1.sbatch"
EXPECTED_PAYLOAD = EXPECTED_INPUTS / "auh_canary_saic_formal_v2_retained_fd_world8_payload_v1.sh"
EXPECTED_GUARD = EXPECTED_INPUTS / "saic_t2v_rendezvous_guard_v2.py"
EXPECTED_RUNTIME = EXPECTED_INPUTS / "generate_saic_pure_t2v_event_bank_topup_v2.py"
EXPECTED_SOURCE_ARCHIVE = EXPECTED_INPUTS / "videoedit-saic-20c2193-methods.tar"
EXPECTED_PROBE_VALIDATOR = EXPECTED_INPUTS / "probe_admission_binding_v1.py"
EXPECTED_POSTFLIGHT = (
    EXPECTED_RELEASE / "postflight"
    / "postflight_saic_formal_v2_retained_fd_world8_canary_v1.py"
)
EXPECTED_RELEASE_MANIFEST = EXPECTED_RELEASE / "release-manifest.json"
AUTHORITY = {
    "scientific": False,
    "generation": False,
    "training": False,
    "publication": False,
    "formal_job_authorized": False,
}
ADMISSION_AUTHORITY = {
    **AUTHORITY,
    "operational_gate": "exact_saic_t2v_topup_r6_formal_v2_release_only",
    "reusable_for_other_release": False,
    "authorizes_formal_submission_by_itself": False,
}
SACCT_FIELD_SPECS = [
    "JobIDRaw", "State", "ExitCode", "AllocTRES%512", "NodeList",
    "Start", "End", "Elapsed", "SubmitLine%8192",
]
SACCT_FIELD_KEYS = [item.split("%", 1)[0] for item in SACCT_FIELD_SPECS]
EXPECTED_EXPORTS = [
    "SAIC_FV2_FD_CANARY_PAYLOAD", "SAIC_FV2_FD_CANARY_PAYLOAD_SHA256",
    "SAIC_FV2_FD_CANARY_GUARD", "SAIC_FV2_FD_CANARY_GUARD_SHA256",
    "SAIC_FV2_FD_CANARY_RUNTIME", "SAIC_FV2_FD_CANARY_RUNTIME_SHA256",
    "SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE",
    "SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE_SHA256",
    "SAIC_FV2_FD_CANARY_PYTHON", "SAIC_FV2_FD_CANARY_PYTHON_SHA256",
    "SAIC_FV2_FD_CANARY_OUTPUT_PARENT", "SAIC_FV2_FD_CANARY_OUTPUT_PARENT_DEVICE",
    "SAIC_FV2_FD_CANARY_OUTPUT_PARENT_INODE", "SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT",
    "SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT_DEVICE",
    "SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT_INODE",
    "SAIC_FV2_FD_CANARY_WRAPPER", "SAIC_FV2_FD_CANARY_WRAPPER_SHA256",
    "SAIC_FV2_FD_CANARY_POSTFLIGHT", "SAIC_FV2_FD_CANARY_POSTFLIGHT_SHA256",
    "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST",
    "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_SHA256",
    "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_DIGEST",
    "SAIC_FV2_FD_CANARY_PROBE_VALIDATOR",
    "SAIC_FV2_FD_CANARY_PROBE_VALIDATOR_SHA256",
    "SAIC_FV2_FD_CANARY_PROBE_ADMISSION",
    "SAIC_FV2_FD_CANARY_PROBE_ADMISSION_SHA256",
    "SAIC_FV2_FD_CANARY_PROBE_ADMISSION_DIGEST",
]
SUBMISSION_FIELDS = {
    "schema_version", "status", "submission_success", "job_success",
    "submitted_job", "request", "submission_boundary", "inputs", "outputs",
    "authority", "receipt_digest",
}
EVIDENCE_FIELDS = {
    "schema_version", "status", "slurm_job_id", "job_success",
    "slurm_terminal_verified", "formal_admission", "topology",
    "requested_gpu_count", "world_size_total", "group_count",
    "rank_packet_count", "unique_actual_master_port_count",
    "collision_receipt_count", "all_launch_rdzv_id_count",
    "all_launch_rdzv_ids_unique", "guard_sha256", "payload_sha256",
    "runtime_sha256", "source_archive_path", "source_archive_sha256",
    "source_archive_retained_fd_number",
    "source_archive_read_from_stage0_retained_fd",
    "archive_member_manifest_sha256", "extracted_tree_manifest_sha256",
    "extracted_tree_manifest_source", "extracted_tree_mode_policy",
    "extracted_tree_entry_set_exact",
    "archive_member_count", "archive_regular_file_count",
    "archive_directory_count", "archive_manifest_canonical_json_bytes",
    "archive_binding_receipt_digest", "runtime_origin_schema_version",
    "runtime_origin_project_module_count", "runtime_origin_project_module_rows",
    "runtime_origin_manifest_sha256", "runtime_origin_receipt_digest",
    "runtime_import_origins_all_from_extracted_archive",
    "runtime_import_checker_isolated_python",
    "runtime_imported_from_pinned_source_archive",
    "runtime_import_scratch_removed_before_evidence",
    "guard_fd_path", "payload_fd_path",
    "replacement_fixture", "guard_retained_leaf", "guard_decoy_leaf",
    "payload_retained_leaf", "payload_decoy_leaf",
    "spooled_wrapper_retained_fd_submission_required",
    "payload_exec_from_parent_retained_fd",
    "guard_read_by_two_background_groups_and_all_workers_from_parent_retained_fd",
    "logical_leaf_replacement_after_fd_open_observed",
    "scratch_cleaned_before_operational_evidence_publication",
    "external_terminal_postflight_admission_required",
    "submission_receipt_path", "submission_receipt_file_sha256",
    "submission_receipt_digest", "output_parent_identity",
    "submission_receipt_identity", "compute_output_parent_identity",
    "compute_submission_receipt_identity",
    "submission_receipt_retained_fd_number",
    "submission_receipt_read_from_retained_fd",
    "stage0_o_nofollow_source_open", "stage0_python_execve_bash_handoff",
    "stage0_inherited_source_fd_numbers", "stage0_source_fds_distinct",
    "stage0_output_directory_fd_number", "stage0_output_directory_identity",
    "stage0_output_directory_o_nofollow",
    "wrapper_sha256",
    "probe_admission_binding",
    "scientific_generation_entered", "scientific_output_created",
    "formal_full60_result_claimed", "candidate_rows", "collision_rows",
    "authority", "receipt_digest",
}
COMPLETION_FIELDS = {
    "schema_version", "status", "slurm_job_id", "group_id", "candidate_index",
    "candidate_id", "launch_ordinal", "rdzv_id", "actual_master_port",
    "claim_path", "claim_digest", "admission_digest", "rank_packet_digests",
    "guard_observed_via_parent_proc_fd", "scientific_generation_entered",
    "source_archive_sha256", "archive_member_manifest_sha256",
    "extracted_tree_manifest_sha256", "extracted_tree_manifest_source",
    "extracted_tree_mode_policy", "extracted_tree_entry_set_exact",
    "archive_member_count",
    "archive_regular_file_count", "archive_directory_count",
    "archive_binding_receipt_digest", "runtime_origin_manifest_sha256",
    "runtime_origin_project_module_count", "runtime_origin_receipt_digest",
    "runtime_import_origins_all_from_extracted_archive",
    "authority", "receipt_digest",
}
RELEASE_MANIFEST_FIELDS = {
    "schema_version", "status", "stem", "release_root", "output_parent",
    "inputs", "postflight", "executables", "probe_admission", "authority",
    "receipt_digest",
}


def die(message: str) -> None:
    raise SystemExit(f"postflight-saic-formal-v2-fd-world8-canary-v1: {message}")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
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


def exact_file(path: Path, expected: Path, expected_sha: str, label: str) -> Path:
    if path != expected or not path.is_absolute() or path.resolve(strict=True) != path:
        die(f"{label} identity differs")
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o444
        or sha_file(path) != expected_sha
    ):
        die(f"{label} bytes/mode differ")
    return path


def exact_directory(path: Path, expected: Path, label: str, mode: int) -> Path:
    if path != expected or not path.is_absolute() or path.resolve(strict=True) != path:
        die(f"{label} identity differs")
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != mode
    ):
        die(f"{label} mode/owner differs")
    return path


def directory_identity(path: Path) -> tuple[int, int]:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        die("directory identity differs")
    return info.st_dev, info.st_ino


def close_before_publication(descriptors: Sequence[int]) -> None:
    first_error: OSError | None = None
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise RuntimeError("retained descriptor close failed before publication") from first_error


def reread_descriptor_exact(descriptor: int, expected: bytes, label: str) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    if b"".join(chunks) != expected:
        die(f"{label} retained bytes changed")
    os.lseek(descriptor, 0, os.SEEK_SET)


def exact_executable(path: Path, expected: Path, expected_sha: str, label: str) -> Path:
    if path != expected or not path.is_absolute() or path.resolve(strict=True) != path:
        die(f"{label} identity differs")
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) & 0o022
        or not os.access(path, os.X_OK) or sha_file(path) != expected_sha
    ):
        die(f"{label} executable differs")
    return path


def retained_exact_bytes(
    path: Path, expected: Path, expected_sha: str, label: str
) -> tuple[int, bytes]:
    if path != expected or not path.is_absolute():
        die(f"{label} identity differs")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        fd_info = os.fstat(descriptor)
        leaf_info = path.lstat()
        if (
            path.resolve(strict=True) != path
            or not stat.S_ISREG(fd_info.st_mode) or fd_info.st_nlink != 1
            or stat.S_IMODE(fd_info.st_mode) != 0o444
            or not stat.S_ISREG(leaf_info.st_mode) or stat.S_ISLNK(leaf_info.st_mode)
            or leaf_info.st_nlink != 1
            or (leaf_info.st_dev, leaf_info.st_ino) != (fd_info.st_dev, fd_info.st_ino)
        ):
            die(f"{label} retained identity differs")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if sha_bytes(raw) != expected_sha:
            die(f"{label} retained bytes differ")
        return descriptor, raw
    except BaseException:
        os.close(descriptor)
        raise


def load_guard() -> tuple[types.ModuleType, int, bytes]:
    descriptor, raw = retained_exact_bytes(
        EXPECTED_GUARD, EXPECTED_GUARD, EXPECTED_GUARD_SHA256, "guard"
    )
    module = types.ModuleType("fd_canary_postflight_guard")
    exec(compile(raw, "guard-v2", "exec"), module.__dict__)
    return module, descriptor, raw


def observe_sacct(job_id: str, submission: dict[str, Any]) -> dict[str, Any]:
    info = EXPECTED_SACCT.lstat()
    if (
        EXPECTED_SACCT.resolve(strict=True) != EXPECTED_SACCT
        or not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
        or info.st_uid != 0 or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) & 0o022
        or not os.access(EXPECTED_SACCT, os.X_OK)
        or sha_file(EXPECTED_SACCT) != EXPECTED_SACCT_SHA256
    ):
        die("root-owned sacct executable differs")
    argv = [
        str(EXPECTED_SACCT), "-j", job_id, "-X", "--noheader", "-n",
        "-P", "-o", ",".join(SACCT_FIELD_SPECS),
    ]
    completed = subprocess.run(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False, timeout=60,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    try:
        stdout = completed.stdout.decode("ascii")
    except UnicodeDecodeError as error:
        raise SystemExit("sacct stdout is not ASCII") from error
    lines = stdout.splitlines()
    fields = lines[0].split("|") if len(lines) == 1 else []
    row = (
        dict(zip(SACCT_FIELD_KEYS, fields, strict=True))
        if len(fields) == len(SACCT_FIELD_KEYS) else {}
    )
    alloc_tokens: dict[str, str] = {}
    for token in str(row.get("AllocTRES", "")).split(","):
        if "=" not in token:
            die("terminal AllocTRES token differs")
        key, value = token.split("=", 1)
        if not key or key in alloc_tokens:
            die("terminal AllocTRES token closure differs")
        alloc_tokens[key] = value
    submit_line = str(row.get("SubmitLine", ""))
    boundary = submission["submission_boundary"]
    inputs = submission["inputs"]
    expected_env = {
        "SAIC_FV2_FD_CANARY_PAYLOAD": str(EXPECTED_PAYLOAD),
        "SAIC_FV2_FD_CANARY_PAYLOAD_SHA256": EXPECTED_PAYLOAD_SHA256,
        "SAIC_FV2_FD_CANARY_GUARD": str(EXPECTED_GUARD),
        "SAIC_FV2_FD_CANARY_GUARD_SHA256": EXPECTED_GUARD_SHA256,
        "SAIC_FV2_FD_CANARY_RUNTIME": str(EXPECTED_RUNTIME),
        "SAIC_FV2_FD_CANARY_RUNTIME_SHA256": EXPECTED_RUNTIME_SHA256,
        "SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE": str(EXPECTED_SOURCE_ARCHIVE),
        "SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE_SHA256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        "SAIC_FV2_FD_CANARY_PYTHON": str(EXPECTED_PYTHON),
        "SAIC_FV2_FD_CANARY_PYTHON_SHA256": EXPECTED_PYTHON_SHA256,
        "SAIC_FV2_FD_CANARY_OUTPUT_PARENT": str(EXPECTED_OUTPUT_PARENT),
        "SAIC_FV2_FD_CANARY_OUTPUT_PARENT_DEVICE": str(boundary["output_parent_device"]),
        "SAIC_FV2_FD_CANARY_OUTPUT_PARENT_INODE": str(boundary["output_parent_inode"]),
        "SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT": str(EXPECTED_SUBMISSION_RECEIPT),
        "SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT_DEVICE": str(boundary["reservation_device"]),
        "SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT_INODE": str(boundary["reservation_inode"]),
        "SAIC_FV2_FD_CANARY_WRAPPER": str(EXPECTED_WRAPPER),
        "SAIC_FV2_FD_CANARY_WRAPPER_SHA256": EXPECTED_WRAPPER_SHA256,
        "SAIC_FV2_FD_CANARY_POSTFLIGHT": str(EXPECTED_POSTFLIGHT),
        "SAIC_FV2_FD_CANARY_POSTFLIGHT_SHA256": inputs["postflight_sha256"],
        "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST": str(EXPECTED_RELEASE_MANIFEST),
        "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_SHA256": inputs["release_manifest_file_sha256"],
        "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_DIGEST": inputs["release_manifest_digest"],
        "SAIC_FV2_FD_CANARY_PROBE_VALIDATOR": str(EXPECTED_PROBE_VALIDATOR),
        "SAIC_FV2_FD_CANARY_PROBE_VALIDATOR_SHA256": EXPECTED_PROBE_VALIDATOR_SHA256,
        "SAIC_FV2_FD_CANARY_PROBE_ADMISSION": str(PROBE_ADMISSION),
        "SAIC_FV2_FD_CANARY_PROBE_ADMISSION_SHA256": PROBE_ADMISSION_SHA256,
        "SAIC_FV2_FD_CANARY_PROBE_ADMISSION_DIGEST": PROBE_ADMISSION_DIGEST,
    }
    fixed_submit_line = " ".join([
        "/usr/bin/sbatch", "--parsable",
        f"--output={EXPECTED_LOG_DIR}/saic-fv2-fd-w8-cny1-%j.out",
        f"--error={EXPECTED_LOG_DIR}/saic-fv2-fd-w8-cny1-%j.err",
        "--export=NONE," + ",".join(
            f"{name}={expected_env[name]}" for name in EXPECTED_EXPORTS
        ),
    ])
    retained_match = re.fullmatch(
        re.escape(fixed_submit_line + " /proc/self/fd/") + r"([0-9]+)",
        submit_line,
    )
    if (
        completed.returncode != 0 or completed.stderr
        or len(lines) != 1 or len(fields) != len(SACCT_FIELD_KEYS)
        or row.get("JobIDRaw") != job_id or row.get("State") != "COMPLETED"
        or row.get("ExitCode") != "0:0"
        or alloc_tokens != {
            "billing": "16", "cpu": "16", "gres/gpu:mi210": "8",
            "gres/gpu": "8", "mem": "32G", "node": "1",
        }
        or not row.get("NodeList") or row.get("NodeList") in {"None assigned", "Unknown"}
        or not row.get("Start") or row.get("Start") == "Unknown"
        or not row.get("End") or row.get("End") == "Unknown"
        or not re.fullmatch(r"[0-9]+-[0-9]{2}:[0-9]{2}:[0-9]{2}|[0-9]{2}:[0-9]{2}:[0-9]{2}", str(row.get("Elapsed", "")))
        or list(expected_env) != EXPECTED_EXPORTS
        or retained_match is None
        or str(int(retained_match.group(1))) != retained_match.group(1)
        or int(retained_match.group(1)) < 3
    ):
        die("terminal Slurm accounting differs")
    return {
        "executable": str(EXPECTED_SACCT),
        "executable_sha256": EXPECTED_SACCT_SHA256,
        "argv": argv, "query_fields": SACCT_FIELD_SPECS,
        "returncode": completed.returncode,
        "stdout_sha256": sha_bytes(completed.stdout),
        "stderr_sha256": sha_bytes(completed.stderr),
        "parsed_row": row, "exact_single_row": True,
        "submit_line_sha256": sha_bytes(submit_line.encode("ascii")),
        "retained_wrapper_fd": int(retained_match.group(1)),
        "exact_submit_line": True,
        "observation_phase": "external_postflight_after_canary_terminal",
    }


def validate_release_manifest(
    guard: types.ModuleType,
    *,
    raw: bytes,
    manifest_sha256: str,
    postflight_sha256: str,
    probe_binding: dict[str, Any],
) -> dict[str, Any]:
    manifest = guard._decode_sealed(
        raw,
        schema_version="saic-formal-v2-retained-fd-world8-release-manifest-v1",
        exact_fields=RELEASE_MANIFEST_FIELDS,
    )
    expected_inputs = {
        "wrapper": str(EXPECTED_WRAPPER),
        "wrapper_sha256": EXPECTED_WRAPPER_SHA256,
        "payload": str(EXPECTED_PAYLOAD),
        "payload_sha256": EXPECTED_PAYLOAD_SHA256,
        "guard": str(EXPECTED_GUARD),
        "guard_sha256": EXPECTED_GUARD_SHA256,
        "runtime": str(EXPECTED_RUNTIME),
        "runtime_sha256": EXPECTED_RUNTIME_SHA256,
        "source_archive": str(EXPECTED_SOURCE_ARCHIVE),
        "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        "probe_validator": str(EXPECTED_PROBE_VALIDATOR),
        "probe_validator_sha256": EXPECTED_PROBE_VALIDATOR_SHA256,
    }
    if (
        sha_bytes(raw) != manifest_sha256
        or manifest.get("status") != "sealed_before_canary_submission"
        or manifest.get("stem") != STEM
        or manifest.get("release_root") != str(EXPECTED_RELEASE)
        or manifest.get("output_parent") != str(EXPECTED_OUTPUT_PARENT)
        or manifest.get("inputs") != expected_inputs
        or manifest.get("postflight") != {
            "path": str(EXPECTED_POSTFLIGHT),
            "sha256": postflight_sha256,
            "sha256_pinned_outside_postflight_source": True,
        }
        or manifest.get("executables") != {
            "python": "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12",
            "python_sha256": EXPECTED_PYTHON_SHA256,
            "sacct": str(EXPECTED_SACCT),
            "sacct_sha256": EXPECTED_SACCT_SHA256,
        }
        or manifest.get("probe_admission") != probe_binding
        or manifest.get("authority") != AUTHORITY
    ):
        die("sealed release manifest differs")
    return manifest


def validate_slurm_logs(job_id: str) -> dict[str, Any]:
    log_dir = exact_directory(EXPECTED_LOG_DIR, EXPECTED_LOG_DIR, "Slurm log root", 0o700)
    stdout_path = log_dir / f"saic-fv2-fd-w8-cny1-{job_id}.out"
    stderr_path = log_dir / f"saic-fv2-fd-w8-cny1-{job_id}.err"
    if set(log_dir.iterdir()) != {stdout_path, stderr_path}:
        die("Slurm log namespace differs")
    result: dict[str, Any] = {}
    for label, path in (("stdout", stdout_path), ("stderr", stderr_path)):
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o022
        ):
            die(f"Slurm {label} log identity differs")
        result[label] = {
            "path": str(path),
            "sha256": sha_file(path),
            "size": info.st_size,
        }
    if result["stderr"]["size"] != 0:
        die("Slurm stderr is not empty")
    return result


def validate_submission(
    submission: dict[str, Any], *, job_id: str, job_root: Path,
    submission_path: Path, submission_identity: tuple[int, int],
    postflight_sha256: str, release_manifest_sha256: str,
    release_manifest_digest: str,
    probe_binding: dict[str, Any],
) -> None:
    submitted = submission.get("submitted_job")
    boundary = submission.get("submission_boundary", {})
    output_info = EXPECTED_OUTPUT_PARENT.lstat()
    if (
        submission.get("status") != "submitted"
        or submission.get("submission_success") is not True
        or submission.get("job_success") is not None
        or not isinstance(submitted, dict)
        or set(submitted) != {"job_id", "cluster", "stdout_sha256", "stderr_sha256"}
        or submitted.get("job_id") != job_id
        or (submitted.get("cluster") is not None
            and (not isinstance(submitted.get("cluster"), str)
                 or not submitted["cluster"] or "\n" in submitted["cluster"]
                 or ";" in submitted["cluster"]))
        or SHA256.fullmatch(str(submitted.get("stdout_sha256", ""))) is None
        or SHA256.fullmatch(str(submitted.get("stderr_sha256", ""))) is None
        or submission.get("request") != {
            "job_name": "saic-fv2-fd-w8-cny1", "partition": "faculty",
            "qos": "bgqos", "nodes": 1, "ntasks": 1, "cpus_per_task": 16,
            "memory": "32G", "walltime": "00:15:00",
            "gpu_resource_requested": "gpu:mi210:8",
            "world_topology": "two_concurrent_world4", "hold": False,
            "dependency": None, "scientific_generation": False,
        }
        or boundary != {
            "environment_replaced": True, "exact_job_export_names": EXPECTED_EXPORTS,
            "export_all": False, "reservation_created_before_sbatch": True,
            "same_inode_retained": True, "launcher_submitted_from_retained_fd": True,
            "reservation_device": boundary.get("reservation_device"),
            "reservation_inode": boundary.get("reservation_inode"),
            "output_parent_device": boundary.get("output_parent_device"),
            "output_parent_inode": boundary.get("output_parent_inode"),
            "success_mode": "0444",
        }
        or not isinstance(boundary.get("reservation_device"), int)
        or boundary["reservation_device"] < 0
        or not isinstance(boundary.get("reservation_inode"), int)
        or boundary["reservation_inode"] <= 0
        or boundary.get("reservation_inode") != submission_identity[1]
        or boundary.get("output_parent_inode") != output_info.st_ino
        or submission.get("inputs") != {
            "wrapper": str(EXPECTED_WRAPPER),
            "wrapper_sha256": EXPECTED_WRAPPER_SHA256,
            "payload": str(EXPECTED_PAYLOAD),
            "payload_sha256": EXPECTED_PAYLOAD_SHA256,
            "guard": str(EXPECTED_GUARD),
            "guard_sha256": EXPECTED_GUARD_SHA256,
            "runtime": str(EXPECTED_RUNTIME),
            "runtime_sha256": EXPECTED_RUNTIME_SHA256,
            "source_archive": str(EXPECTED_SOURCE_ARCHIVE),
            "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
            "python": "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12",
            "python_sha256": EXPECTED_PYTHON_SHA256,
            "postflight": str(EXPECTED_POSTFLIGHT),
            "postflight_sha256": postflight_sha256,
            "release_manifest": str(EXPECTED_RELEASE_MANIFEST),
            "release_manifest_file_sha256": release_manifest_sha256,
            "release_manifest_digest": release_manifest_digest,
            "probe_validator": str(EXPECTED_PROBE_VALIDATOR),
            "probe_validator_sha256": EXPECTED_PROBE_VALIDATOR_SHA256,
            "probe_admission": str(PROBE_ADMISSION),
            "probe_admission_sha256": PROBE_ADMISSION_SHA256,
            "probe_admission_digest": PROBE_ADMISSION_DIGEST,
            "probe_admission_binding": probe_binding,
        }
        or submission.get("outputs") != {
            "output_parent": str(EXPECTED_OUTPUT_PARENT),
            "job_output_root": str(job_root),
            "submission_receipt": str(submission_path),
            "fresh_before_submission": True,
        }
        or submission.get("authority") != AUTHORITY
    ):
        die("submission receipt deep admission differs")


def validate_operational_closure(
    guard: types.ModuleType, *, job_id: str, job_root: Path,
    fixture: Path, evidence: dict[str, Any], submission_raw: bytes,
    submission: dict[str, Any], submission_path: Path,
    probe_binding: dict[str, Any],
) -> None:
    root = guard.exact_directory(job_root, label="postflight output root")
    if sorted(item.name for item in root.iterdir()) != [
        "forbidden-attempts", "logs", "operational-evidence.json", "rendezvous"
    ]:
        die("postflight output root closure differs")
    forbidden = guard.exact_directory(
        root / "forbidden-attempts", label="postflight forbidden root"
    )
    if any(forbidden.iterdir()):
        die("postflight observed scientific output")
    rendezvous = guard.exact_directory(root / "rendezvous", label="postflight rendezvous root")
    if sorted(item.name for item in rendezvous.iterdir()) != ["port-claims", "sp4-a", "sp4-b"]:
        die("postflight rendezvous root closure differs")
    claim_root = guard.exact_directory(rendezvous / "port-claims", label="postflight claim root")
    rows: list[dict[str, Any]] = []
    collision_rows: list[dict[str, Any]] = []
    ports: set[int] = set()
    claims: set[Path] = set()
    rdzv_ids: set[str] = set()
    for group in ("sp4-a", "sp4-b"):
        candidate_id = f"fd-canary-{group}-candidate-00"
        candidate_digest = hashlib.sha256(candidate_id.encode("ascii")).hexdigest()
        group_root = guard.exact_directory(rendezvous / group, label="postflight group root")
        candidate_root = group_root / f"candidate-00-{candidate_digest[:16]}"
        if sorted(item.name for item in group_root.iterdir()) != [candidate_root.name]:
            die("postflight candidate root closure differs")
        candidate_root = guard.exact_directory(candidate_root, label="postflight candidate root")
        launches = sorted(candidate_root.iterdir(), key=lambda item: item.name)
        if (
            not launches or len(launches) > 16
            or [item.name for item in launches]
            != [f"launch-{ordinal:02d}" for ordinal in range(1, len(launches) + 1)]
        ):
            die("postflight launch closure differs")
        for ordinal, life in enumerate(launches[:-1], start=1):
            life = guard.exact_directory(life, label="postflight collision lifecycle")
            if sorted(item.name for item in life.iterdir()) != ["collision.json", "torchrun.log"]:
                die("postflight collision lifecycle closure differs")
            collision = guard._validate_collision_for_job_audit(
                life / "collision.json", claim_root=claim_root, slurm_job_id=job_id
            )
            rdzv_id = f"saic-{job_id}-{group}-c00-{candidate_digest[:16]}-l{ordinal:02d}"
            if (
                collision.get("group_id") != group
                or collision.get("candidate_index") != 0
                or collision.get("candidate_id") != candidate_id
                or collision.get("launch_ordinal") != ordinal
                or collision.get("rdzv_id") != rdzv_id
                or rdzv_id in rdzv_ids
            ):
                die("postflight collision identity differs")
            rdzv_ids.add(rdzv_id)
            collision_rows.append({
                "group_id": group, "launch_ordinal": ordinal,
                "rdzv_id": rdzv_id,
                "collision_digest": collision["receipt_digest"],
            })
        ordinal = len(launches)
        life = guard.exact_directory(launches[-1], label="postflight successful lifecycle")
        if sorted(item.name for item in life.iterdir()) != [
            "admission.json", "fd-completion.json", "rank-0.json", "rank-1.json",
            "rank-2.json", "rank-3.json", "torchrun.log",
        ]:
            die("postflight successful lifecycle closure differs")
        for current in launches:
            info = (current / "torchrun.log").lstat()
            if (
                not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o444
            ):
                die("postflight torchrun log differs")
        completion = guard.wait_load_sealed(
            life / "fd-completion.json",
            schema_version="saic-formal-v2-retained-fd-world4-completion-v1",
            exact_fields=COMPLETION_FIELDS, label="postflight group completion",
        )
        decision = guard.wait_load_sealed(
            life / "admission.json", schema_version=guard.DECISION_SCHEMA_VERSION,
            exact_fields=guard.DECISION_FIELDS, label="postflight group admission",
        )
        rdzv_id = f"saic-{job_id}-{group}-c00-{candidate_digest[:16]}-l{ordinal:02d}"
        claim_path = claim_root / f"port-{decision['actual_master_port']}.json"
        claim = guard.wait_load_sealed(
            claim_path, schema_version=guard.CLAIM_SCHEMA_VERSION,
            exact_fields=guard.CLAIM_FIELDS, label="postflight group claim",
        )
        guard._validate_claim(
            claim, port=decision["actual_master_port"]
        )
        guard._validate_generic_admission(claim, require_rank_packets=True)
        if (
            completion.get("status") != "retained_guard_fd_world4_runtime_help_completed"
            or completion.get("slurm_job_id") != job_id
            or completion.get("group_id") != group
            or completion.get("candidate_index") != 0
            or completion.get("candidate_id") != candidate_id
            or completion.get("launch_ordinal") != ordinal
            or completion.get("rdzv_id") != rdzv_id
            or completion.get("actual_master_port") != decision.get("actual_master_port")
            or completion.get("claim_path") != str(claim_path)
            or completion.get("claim_digest") != claim.get("receipt_digest")
            or completion.get("admission_digest") != decision.get("receipt_digest")
            or completion.get("rank_packet_digests") != decision.get("rank_packet_digests")
            or completion.get("source_archive_sha256")
            != EXPECTED_SOURCE_ARCHIVE_SHA256
            or completion.get("archive_member_manifest_sha256")
            != EXPECTED_ARCHIVE_MEMBER_MANIFEST_SHA256
            or SHA256.fullmatch(str(
                completion.get("extracted_tree_manifest_sha256", ""))) is None
            or completion.get("extracted_tree_manifest_source")
            != "actual_lstat_after_extraction"
            or completion.get("extracted_tree_mode_policy")
            != "directories_0700_files_0400"
            or completion.get("extracted_tree_entry_set_exact") is not True
            or completion.get("archive_member_count") != 864
            or completion.get("archive_regular_file_count") != 853
            or completion.get("archive_directory_count") != 11
            or SHA256.fullmatch(str(
                completion.get("archive_binding_receipt_digest", ""))) is None
            or completion.get("runtime_origin_manifest_sha256")
            != EXPECTED_RUNTIME_ORIGIN_MANIFEST_SHA256
            or completion.get("runtime_origin_project_module_count") != 14
            or SHA256.fullmatch(str(
                completion.get("runtime_origin_receipt_digest", ""))) is None
            or completion.get("runtime_import_origins_all_from_extracted_archive")
            is not True
            or completion.get("guard_observed_via_parent_proc_fd") is not True
            or completion.get("scientific_generation_entered") is not False
            or completion.get("authority") != AUTHORITY
            or any(decision.get(key) != expected for key, expected in {
                "slurm_job_id": job_id, "group_id": group, "candidate_index": 0,
                "candidate_id": candidate_id, "launch_ordinal": ordinal,
                "rdzv_id": rdzv_id,
            }.items())
            or rdzv_id in rdzv_ids
        ):
            die("postflight group linkage differs")
        rdzv_ids.add(rdzv_id)
        ports.add(completion["actual_master_port"])
        claims.add(claim_path)
        rows.append({
            "group_id": group, "candidate_id": candidate_id,
            "successful_launch_ordinal": ordinal, "rdzv_id": rdzv_id,
            "actual_master_port": completion["actual_master_port"],
            "claim_path": str(claim_path), "claim_digest": completion["claim_digest"],
            "admission_digest": completion["admission_digest"],
            "rank_packet_digests": completion["rank_packet_digests"],
            "archive_binding_receipt_digest": completion[
                "archive_binding_receipt_digest"],
            "archive_member_manifest_sha256": completion[
                "archive_member_manifest_sha256"],
            "extracted_tree_manifest_sha256": completion[
                "extracted_tree_manifest_sha256"],
            "runtime_origin_manifest_sha256": completion[
                "runtime_origin_manifest_sha256"],
            "runtime_origin_receipt_digest": completion[
                "runtime_origin_receipt_digest"],
            "completion_digest": completion["receipt_digest"],
        })
    if set(claim_root.iterdir()) != claims or len(claims) != 2 or len(ports) != 2:
        die("postflight WORLD8 claim/port closure differs")
    logs = guard.exact_directory(root / "logs", label="postflight logs root")
    if {item.name for item in logs.iterdir()} != {"sp4-a.group.log", "sp4-b.group.log"}:
        die("postflight log closure differs")
    for log in logs.iterdir():
        info = log.lstat()
        if (
            not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o444
        ):
            die("postflight group log differs")
    fixture = guard.exact_directory(fixture, label="postflight replacement fixture")
    if sorted(item.name for item in fixture.iterdir()) != [
        "guard.logical", "guard.original", "payload.logical", "payload.original"
    ]:
        die("postflight fixture closure differs")
    guard_original = fixture / "guard.original"
    guard_decoy = fixture / "guard.logical"
    payload_original = fixture / "payload.original"
    payload_decoy = fixture / "payload.logical"
    for path in (guard_original, guard_decoy, payload_original, payload_decoy):
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o444
        ):
            die("postflight fixture identity differs")
    archive_binding_core = {
        "schema_version": "saic-formal-v2-source-archive-extraction-binding-v1",
        "status": "exact_formal_source_archive_extracted_and_verified",
        "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        "archive_member_manifest_sha256": EXPECTED_ARCHIVE_MEMBER_MANIFEST_SHA256,
        "extracted_tree_manifest_sha256": evidence.get(
            "extracted_tree_manifest_sha256"),
        "extracted_tree_manifest_source": "actual_lstat_after_extraction",
        "extracted_tree_mode_policy": "directories_0700_files_0400",
        "extracted_tree_entry_set_exact": True,
        "archive_member_count": 864,
        "archive_regular_file_count": 853,
        "archive_directory_count": 11,
        "archive_manifest_canonical_json_bytes": 173053,
        "source_archive_read_from_stage0_retained_fd": True,
        "authority": AUTHORITY,
    }
    archive_binding_digest = sha_bytes(canonical(archive_binding_core))
    runtime_origin_core = {
        "schema_version": "saic-formal-v2-runtime-import-origin-closure-v1",
        "status": "isolated_runtime_help_import_closure_verified",
        "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        "archive_member_manifest_sha256": EXPECTED_ARCHIVE_MEMBER_MANIFEST_SHA256,
        "runtime_sha256": EXPECTED_RUNTIME_SHA256,
        "runtime_relative_path": (
            "methods/bernini_action_editing/"
            "generate_saic_pure_t2v_event_bank_topup_v2.py"
        ),
        "project_module_count": 14,
        "project_module_rows": evidence.get("runtime_origin_project_module_rows"),
        "project_module_manifest_sha256": EXPECTED_RUNTIME_ORIGIN_MANIFEST_SHA256,
        "all_project_module_origins_from_extracted_archive": True,
        "isolated_python_no_environment_path": True,
        "runtime_help_exit_status": 0,
        "authority": AUTHORITY,
    }
    runtime_origin_digest = sha_bytes(canonical(runtime_origin_core))
    if (
        sha_file(guard_original) != EXPECTED_GUARD_SHA256
        or sha_file(payload_original) != EXPECTED_PAYLOAD_SHA256
        or sha_file(guard_decoy) == EXPECTED_GUARD_SHA256
        or sha_file(payload_decoy) == EXPECTED_PAYLOAD_SHA256
        or (guard_original.stat().st_dev, guard_original.stat().st_ino)
        == (guard_decoy.stat().st_dev, guard_decoy.stat().st_ino)
        or (payload_original.stat().st_dev, payload_original.stat().st_ino)
        == (payload_decoy.stat().st_dev, payload_decoy.stat().st_ino)
        or evidence.get("status") != "retained_fd_operational_evidence_complete"
        or evidence.get("slurm_job_id") != job_id
        or evidence.get("job_success") is not None
        or evidence.get("slurm_terminal_verified") is not False
        or evidence.get("formal_admission") is not False
        or evidence.get("topology")
        != "two_concurrent_world4_on_one_requested_8mi210_node"
        or evidence.get("requested_gpu_count") != 8
        or evidence.get("world_size_total") != 8
        or evidence.get("group_count") != 2
        or evidence.get("rank_packet_count") != 8
        or evidence.get("unique_actual_master_port_count") != 2
        or evidence.get("all_launch_rdzv_id_count") != len(rdzv_ids)
        or evidence.get("all_launch_rdzv_ids_unique") is not True
        or evidence.get("guard_sha256") != EXPECTED_GUARD_SHA256
        or evidence.get("payload_sha256") != EXPECTED_PAYLOAD_SHA256
        or evidence.get("runtime_sha256") != EXPECTED_RUNTIME_SHA256
        or evidence.get("source_archive_path") != str(EXPECTED_SOURCE_ARCHIVE)
        or evidence.get("source_archive_sha256") != EXPECTED_SOURCE_ARCHIVE_SHA256
        or not isinstance(evidence.get("source_archive_retained_fd_number"), int)
        or evidence["source_archive_retained_fd_number"] < 3
        or evidence.get("source_archive_read_from_stage0_retained_fd") is not True
        or evidence.get("archive_member_manifest_sha256")
        != EXPECTED_ARCHIVE_MEMBER_MANIFEST_SHA256
        or SHA256.fullmatch(str(
            evidence.get("extracted_tree_manifest_sha256", ""))) is None
        or evidence.get("extracted_tree_manifest_source")
        != "actual_lstat_after_extraction"
        or evidence.get("extracted_tree_mode_policy")
        != "directories_0700_files_0400"
        or evidence.get("extracted_tree_entry_set_exact") is not True
        or evidence.get("archive_member_count") != 864
        or evidence.get("archive_regular_file_count") != 853
        or evidence.get("archive_directory_count") != 11
        or evidence.get("archive_manifest_canonical_json_bytes") != 173053
        or SHA256.fullmatch(str(
            evidence.get("archive_binding_receipt_digest", ""))) is None
        or evidence.get("archive_binding_receipt_digest") != archive_binding_digest
        or evidence.get("runtime_origin_schema_version")
        != "saic-formal-v2-runtime-import-origin-closure-v1"
        or evidence.get("runtime_origin_project_module_count") != 14
        or not isinstance(evidence.get("runtime_origin_project_module_rows"), list)
        or len(evidence["runtime_origin_project_module_rows"]) != 14
        or any(
            not isinstance(row, dict)
            or set(row) != {"module", "relative_path", "sha256"}
            or row.get("relative_path") != f"{row.get('module')}.py"
            or SHA256.fullmatch(str(row.get("sha256", ""))) is None
            for row in evidence["runtime_origin_project_module_rows"]
        )
        or sha_bytes(canonical(evidence["runtime_origin_project_module_rows"]))
        != EXPECTED_RUNTIME_ORIGIN_MANIFEST_SHA256
        or evidence.get("runtime_origin_manifest_sha256")
        != EXPECTED_RUNTIME_ORIGIN_MANIFEST_SHA256
        or SHA256.fullmatch(str(
            evidence.get("runtime_origin_receipt_digest", ""))) is None
        or evidence.get("runtime_origin_receipt_digest") != runtime_origin_digest
        or evidence.get("runtime_import_origins_all_from_extracted_archive") is not True
        or evidence.get("runtime_import_checker_isolated_python") is not True
        or evidence.get("runtime_imported_from_pinned_source_archive") is not True
        or evidence.get("runtime_import_scratch_removed_before_evidence") is not True
        or re.fullmatch(
            r"/proc/[0-9]+/fd/[0-9]+", str(evidence.get("guard_fd_path", ""))
        ) is None
        or re.fullmatch(
            r"/proc/[0-9]+/fd/[0-9]+", str(evidence.get("payload_fd_path", ""))
        ) is None
        or evidence.get("spooled_wrapper_retained_fd_submission_required") is not True
        or evidence.get("payload_exec_from_parent_retained_fd") is not True
        or evidence.get(
            "guard_read_by_two_background_groups_and_all_workers_from_parent_retained_fd"
        ) is not True
        or evidence.get("logical_leaf_replacement_after_fd_open_observed") is not True
        or evidence.get("scratch_cleaned_before_operational_evidence_publication")
        is not True
        or evidence.get("external_terminal_postflight_admission_required") is not True
        or evidence.get("scientific_generation_entered") is not False
        or evidence.get("scientific_output_created") is not False
        or evidence.get("formal_full60_result_claimed") is not False
        or evidence.get("wrapper_sha256") != EXPECTED_WRAPPER_SHA256
        or evidence.get("authority") != AUTHORITY
        or evidence.get("candidate_rows") != rows
        or any(
            row.get("archive_binding_receipt_digest") != archive_binding_digest
            or row.get("archive_member_manifest_sha256")
            != EXPECTED_ARCHIVE_MEMBER_MANIFEST_SHA256
            or row.get("extracted_tree_manifest_sha256")
            != evidence.get("extracted_tree_manifest_sha256")
            or row.get("runtime_origin_manifest_sha256")
            != EXPECTED_RUNTIME_ORIGIN_MANIFEST_SHA256
            or row.get("runtime_origin_receipt_digest") != runtime_origin_digest
            for row in rows
        )
        or evidence.get("collision_rows") != collision_rows
        or evidence.get("collision_receipt_count") != len(collision_rows)
        or evidence.get("replacement_fixture") != str(fixture)
        or evidence.get("guard_retained_leaf") != str(guard_original)
        or evidence.get("guard_decoy_leaf") != str(guard_decoy)
        or evidence.get("payload_retained_leaf") != str(payload_original)
        or evidence.get("payload_decoy_leaf") != str(payload_decoy)
        or evidence.get("submission_receipt_path") != str(submission_path)
        or evidence.get("submission_receipt_file_sha256") != sha_bytes(submission_raw)
        or evidence.get("submission_receipt_digest") != submission["receipt_digest"]
        or evidence.get("output_parent_identity")
        != (
            f"{submission['submission_boundary']['output_parent_device']}:"
            f"{submission['submission_boundary']['output_parent_inode']}"
        )
        or evidence.get("submission_receipt_identity")
        != (
            f"{submission['submission_boundary']['reservation_device']}:"
            f"{submission['submission_boundary']['reservation_inode']}"
        )
        or re.fullmatch(
            r"[0-9]+:[1-9][0-9]*",
            str(evidence.get("compute_output_parent_identity", "")),
        ) is None
        or re.fullmatch(
            r"[0-9]+:[1-9][0-9]*",
            str(evidence.get("compute_submission_receipt_identity", "")),
        ) is None
        or evidence["compute_output_parent_identity"].split(":")[-1]
        != str(submission["submission_boundary"]["output_parent_inode"])
        or evidence["compute_submission_receipt_identity"].split(":")[-1]
        != str(submission["submission_boundary"]["reservation_inode"])
        or not isinstance(evidence.get("submission_receipt_retained_fd_number"), int)
        or evidence["submission_receipt_retained_fd_number"] < 3
        or evidence.get("submission_receipt_read_from_retained_fd") is not True
        or evidence.get("stage0_o_nofollow_source_open") is not True
        or evidence.get("stage0_python_execve_bash_handoff") is not True
        or evidence.get("stage0_source_fds_distinct") is not True
        or not isinstance(evidence.get("stage0_inherited_source_fd_numbers"), dict)
        or set(evidence["stage0_inherited_source_fd_numbers"]) != {
            "guard", "spooled_wrapper", "payload", "probe_validator",
            "source_archive", "submission_receipt",
        }
        or any(
            not isinstance(number, int) or number < 3
            for number in evidence["stage0_inherited_source_fd_numbers"].values()
        )
        or len(set(evidence["stage0_inherited_source_fd_numbers"].values())) != 6
        or evidence["stage0_inherited_source_fd_numbers"]["source_archive"]
        != evidence["source_archive_retained_fd_number"]
        or evidence["stage0_inherited_source_fd_numbers"]["submission_receipt"]
        != evidence["submission_receipt_retained_fd_number"]
        or not isinstance(evidence.get("stage0_output_directory_fd_number"), int)
        or evidence["stage0_output_directory_fd_number"] < 3
        or evidence["stage0_output_directory_fd_number"]
        in set(evidence["stage0_inherited_source_fd_numbers"].values())
        or evidence.get("stage0_output_directory_identity")
        != evidence.get("compute_output_parent_identity")
        or evidence.get("stage0_output_directory_o_nofollow") is not True
        or evidence.get("probe_admission_binding") != probe_binding
    ):
        die("postflight evidence deep closure differs")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--submission-receipt", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--wrapper", required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--guard", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--release-manifest", required=True)
    parser.add_argument("--release-manifest-sha256", required=True)
    parser.add_argument("--postflight", required=True)
    parser.add_argument("--postflight-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        not re.fullmatch(r"[0-9]+", args.job_id)
        or SHA256.fullmatch(args.release_manifest_sha256) is None
        or SHA256.fullmatch(args.postflight_sha256) is None
    ):
        die("job ID differs")

    output_parent = exact_directory(
        Path(args.output_parent), EXPECTED_OUTPUT_PARENT, "output parent", 0o700
    )
    output_identity = directory_identity(output_parent)
    release = exact_directory(EXPECTED_RELEASE, EXPECTED_RELEASE, "release root", 0o555)
    inputs = exact_directory(EXPECTED_INPUTS, EXPECTED_INPUTS, "release inputs", 0o555)
    postflight_root = exact_directory(
        EXPECTED_POSTFLIGHT.parent, EXPECTED_POSTFLIGHT.parent,
        "release postflight root", 0o555,
    )
    if (
        set(release.iterdir())
        != {EXPECTED_INPUTS, EXPECTED_POSTFLIGHT.parent, EXPECTED_RELEASE_MANIFEST}
        or set(inputs.iterdir())
        != {
            EXPECTED_WRAPPER, EXPECTED_PAYLOAD, EXPECTED_GUARD,
            EXPECTED_RUNTIME, EXPECTED_SOURCE_ARCHIVE, EXPECTED_PROBE_VALIDATOR,
        }
        or set(postflight_root.iterdir()) != {EXPECTED_POSTFLIGHT}
    ):
        die("sealed release namespace differs")

    submission_path = Path(args.submission_receipt)
    if submission_path != EXPECTED_SUBMISSION_RECEIPT:
        die("submission receipt path differs")
    submission_info = submission_path.lstat()
    if (
        not stat.S_ISREG(submission_info.st_mode)
        or stat.S_ISLNK(submission_info.st_mode)
        or submission_info.st_nlink != 1
        or stat.S_IMODE(submission_info.st_mode) != 0o444
    ):
        die("submission receipt identity differs")
    submission_identity = (submission_info.st_dev, submission_info.st_ino)
    admission_path = Path(args.admission)
    if (
        admission_path != EXPECTED_ADMISSION or not admission_path.is_absolute()
        or admission_path.exists() or admission_path.is_symlink()
    ):
        die("combined admission is not fresh")
    exact_file(Path(args.wrapper), EXPECTED_WRAPPER, EXPECTED_WRAPPER_SHA256, "wrapper")
    exact_file(Path(args.payload), EXPECTED_PAYLOAD, EXPECTED_PAYLOAD_SHA256, "payload")
    exact_file(Path(args.guard), EXPECTED_GUARD, EXPECTED_GUARD_SHA256, "guard")
    exact_file(Path(args.runtime), EXPECTED_RUNTIME, EXPECTED_RUNTIME_SHA256, "runtime")
    exact_file(
        Path(args.source_archive), EXPECTED_SOURCE_ARCHIVE,
        EXPECTED_SOURCE_ARCHIVE_SHA256, "source archive",
    )
    exact_file(
        EXPECTED_PROBE_VALIDATOR, EXPECTED_PROBE_VALIDATOR,
        EXPECTED_PROBE_VALIDATOR_SHA256, "probe validator",
    )
    exact_executable(
        Path(sys.executable).resolve(strict=True), EXPECTED_PYTHON,
        EXPECTED_PYTHON_SHA256, "running Python",
    )

    invoked_postflight = Path(__file__)
    if (
        invoked_postflight != EXPECTED_POSTFLIGHT
        or Path(args.postflight) != EXPECTED_POSTFLIGHT
        or Path(args.release_manifest) != EXPECTED_RELEASE_MANIFEST
    ):
        die("postflight invocation path differs")

    job_root = output_parent / f"job-{args.job_id}"
    failure_path = output_parent / f"job-{args.job_id}.failure.json"
    fixture = output_parent / f"fd-fixture-job-{args.job_id}"
    evidence_path = job_root / "operational-evidence.json"

    expected_parent_entries = {submission_path, job_root, fixture}
    if (
        failure_path.exists() or failure_path.is_symlink()
        or set(output_parent.iterdir()) != expected_parent_entries
    ):
        die("pre-admission output namespace differs")

    descriptors: list[int] = []
    try:
        postflight_descriptor, postflight_raw = retained_exact_bytes(
            Path(args.postflight), EXPECTED_POSTFLIGHT,
            args.postflight_sha256, "postflight",
        )
        descriptors.append(postflight_descriptor)
        manifest_descriptor, manifest_raw = retained_exact_bytes(
            Path(args.release_manifest), EXPECTED_RELEASE_MANIFEST,
            args.release_manifest_sha256, "release manifest",
        )
        descriptors.append(manifest_descriptor)
        guard, guard_descriptor, guard_raw = load_guard()
        descriptors.append(guard_descriptor)
        source_archive_descriptor, source_archive_raw = retained_exact_bytes(
            EXPECTED_SOURCE_ARCHIVE, EXPECTED_SOURCE_ARCHIVE,
            EXPECTED_SOURCE_ARCHIVE_SHA256, "source archive",
        )
        descriptors.append(source_archive_descriptor)
        validator_descriptor, validator_raw = retained_exact_bytes(
            EXPECTED_PROBE_VALIDATOR, EXPECTED_PROBE_VALIDATOR,
            EXPECTED_PROBE_VALIDATOR_SHA256, "probe validator",
        )
        descriptors.append(validator_descriptor)
        validator = types.ModuleType("sealed_probe_admission_binding_v1")
        exec(
            compile(validator_raw, str(EXPECTED_PROBE_VALIDATOR), "exec"),
            validator.__dict__,
        )
        probe_binding = validator.validate_probe_admission(
            PROBE_ADMISSION,
            expected_sha=PROBE_ADMISSION_SHA256,
            expected_digest=PROBE_ADMISSION_DIGEST,
        )

        manifest = validate_release_manifest(
            guard, raw=manifest_raw,
            manifest_sha256=args.release_manifest_sha256,
            postflight_sha256=args.postflight_sha256,
            probe_binding=probe_binding,
        )
        submission_raw = guard.wait_ready_bytes(
            submission_path, label="postflight submission receipt",
            expected_identity=submission_identity,
        )
        submission = guard._decode_sealed(
            submission_raw,
            schema_version="saic-formal-v2-retained-fd-world8-submission-v2",
            exact_fields=SUBMISSION_FIELDS,
        )
        validate_submission(
            submission, job_id=args.job_id, job_root=job_root,
            submission_path=submission_path,
            submission_identity=submission_identity,
            postflight_sha256=args.postflight_sha256,
            release_manifest_sha256=args.release_manifest_sha256,
            release_manifest_digest=manifest["receipt_digest"],
            probe_binding=probe_binding,
        )
        evidence_raw = guard.wait_ready_bytes(
            evidence_path, label="postflight operational evidence"
        )
        evidence = guard._decode_sealed(
            evidence_raw,
            schema_version=(
                "saic-formal-v2-retained-fd-world8-operational-evidence-v1"
            ),
            exact_fields=EVIDENCE_FIELDS,
        )
        validate_operational_closure(
            guard, job_id=args.job_id, job_root=job_root, fixture=fixture,
            evidence=evidence, submission_raw=submission_raw,
            submission=submission, submission_path=submission_path,
            probe_binding=probe_binding,
        )

        guard_fd_match = re.fullmatch(
            r"/proc/([0-9]+)/fd/([0-9]+)", str(evidence["guard_fd_path"])
        )
        payload_fd_match = re.fullmatch(
            r"/proc/([0-9]+)/fd/([0-9]+)", str(evidence["payload_fd_path"])
        )
        if (
            guard_fd_match is None or payload_fd_match is None
            or guard_fd_match.group(1) != payload_fd_match.group(1)
            or guard_fd_match.group(2) == payload_fd_match.group(2)
        ):
            die("retained child descriptor path linkage differs")

        sacct = observe_sacct(args.job_id, submission)
        slurm_logs = validate_slurm_logs(args.job_id)

        if (
            guard.wait_ready_bytes(
                submission_path, label="terminal submission receipt reread",
                expected_identity=submission_identity,
            ) != submission_raw
            or guard.wait_ready_bytes(
                evidence_path, label="terminal operational evidence reread"
            ) != evidence_raw
        ):
            die("terminal evidence bytes changed")
        reread_descriptor_exact(
            postflight_descriptor, postflight_raw, "postflight"
        )
        reread_descriptor_exact(
            manifest_descriptor, manifest_raw, "release manifest"
        )
        reread_descriptor_exact(guard_descriptor, guard_raw, "guard")
        reread_descriptor_exact(
            source_archive_descriptor, source_archive_raw, "source archive"
        )
        reread_descriptor_exact(
            validator_descriptor, validator_raw, "probe validator"
        )
        if validator.validate_probe_admission(
            PROBE_ADMISSION,
            expected_sha=PROBE_ADMISSION_SHA256,
            expected_digest=PROBE_ADMISSION_DIGEST,
        ) != probe_binding:
            die("terminal probe admission bytes changed")
        if sha_bytes(guard_raw) != EXPECTED_GUARD_SHA256:
            die("guard retained bytes differ")

        retained_identities = {
            "postflight": {
                "device": os.fstat(postflight_descriptor).st_dev,
                "inode": os.fstat(postflight_descriptor).st_ino,
            },
            "release_manifest": {
                "device": os.fstat(manifest_descriptor).st_dev,
                "inode": os.fstat(manifest_descriptor).st_ino,
            },
            "guard": {
                "device": os.fstat(guard_descriptor).st_dev,
                "inode": os.fstat(guard_descriptor).st_ino,
            },
            "source_archive": {
                "device": os.fstat(source_archive_descriptor).st_dev,
                "inode": os.fstat(source_archive_descriptor).st_ino,
            },
            "probe_validator": {
                "device": os.fstat(validator_descriptor).st_dev,
                "inode": os.fstat(validator_descriptor).st_ino,
            },
        }
    except BaseException:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise

    core = {
        "schema_version": "saic-formal-v2-retained-fd-world8-canary-admission-v1",
        "status": "terminal_completed_retained_fd_world8_operational_admitted",
        "job_id": args.job_id, "job_success": True,
        "slurm_terminal_verified": True,
        "operational_canary_admitted": True,
        "formal_admission": False,
        "external_formal_submitter_exact_pin_required": True,
        "submission_receipt_path": str(submission_path),
        "submission_receipt_sha256": sha_bytes(submission_raw),
        "submission_receipt_digest": submission["receipt_digest"],
        "output_parent_identity": {
            "device": output_identity[0], "inode": output_identity[1],
        },
        "submission_receipt_identity": {
            "device": submission_identity[0], "inode": submission_identity[1],
        },
        "operational_evidence_path": str(evidence_path),
        "operational_evidence_sha256": sha_bytes(evidence_raw),
        "operational_evidence_digest": evidence["receipt_digest"],
        "failure_receipt_absent": True,
        "wrapper_sha256": EXPECTED_WRAPPER_SHA256,
        "payload_sha256": EXPECTED_PAYLOAD_SHA256,
        "guard_sha256": EXPECTED_GUARD_SHA256,
        "runtime_sha256": EXPECTED_RUNTIME_SHA256,
        "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        "archive_member_manifest_sha256": evidence[
            "archive_member_manifest_sha256"],
        "extracted_tree_manifest_sha256": evidence[
            "extracted_tree_manifest_sha256"],
        "extracted_tree_manifest_source": evidence[
            "extracted_tree_manifest_source"],
        "extracted_tree_mode_policy": evidence["extracted_tree_mode_policy"],
        "extracted_tree_entry_set_exact": evidence[
            "extracted_tree_entry_set_exact"],
        "archive_member_count": evidence["archive_member_count"],
        "archive_regular_file_count": evidence["archive_regular_file_count"],
        "archive_directory_count": evidence["archive_directory_count"],
        "archive_binding_receipt_digest": evidence[
            "archive_binding_receipt_digest"],
        "runtime_origin_schema_version": evidence["runtime_origin_schema_version"],
        "runtime_origin_project_module_count": evidence[
            "runtime_origin_project_module_count"],
        "runtime_origin_project_module_rows": evidence[
            "runtime_origin_project_module_rows"],
        "runtime_origin_manifest_sha256": evidence[
            "runtime_origin_manifest_sha256"],
        "runtime_origin_receipt_digest": evidence[
            "runtime_origin_receipt_digest"],
        "runtime_import_origins_all_from_extracted_archive": True,
        "probe_validator_sha256": EXPECTED_PROBE_VALIDATOR_SHA256,
        "probe_admission_binding": probe_binding,
        "python_sha256": EXPECTED_PYTHON_SHA256,
        "release_manifest_path": str(EXPECTED_RELEASE_MANIFEST),
        "release_manifest_sha256": sha_bytes(manifest_raw),
        "release_manifest_digest": manifest["receipt_digest"],
        "postflight_path": str(EXPECTED_POSTFLIGHT),
        "postflight_sha256": sha_bytes(postflight_raw),
        "postflight_sha256_pinned_by_release_manifest": True,
        "retained_verification_identities": retained_identities,
        "sacct_terminal_observation": sacct,
        "slurm_logs": slurm_logs,
        "output_namespace_deep_closed": True,
        "underlying_world8_closure_deep_validated": True,
        "science_generation_entered": False,
        "authority": ADMISSION_AUTHORITY,
    }
    sealed_admission = guard.seal(core)

    if (
        exact_directory(
            output_parent, EXPECTED_OUTPUT_PARENT, "terminal output parent", 0o700
        ) != output_parent
        or directory_identity(output_parent) != output_identity
        or set(output_parent.iterdir()) != expected_parent_entries
        or failure_path.exists() or failure_path.is_symlink()
        or admission_path.exists() or admission_path.is_symlink()
        or guard.wait_ready_bytes(
            submission_path, label="final submission receipt",
            expected_identity=submission_identity,
        ) != submission_raw
        or guard.wait_ready_bytes(
            evidence_path, label="final operational evidence"
        ) != evidence_raw
    ):
        die("terminal admission boundary differs")
    validate_operational_closure(
        guard, job_id=args.job_id, job_root=job_root, fixture=fixture,
        evidence=evidence, submission_raw=submission_raw,
        submission=submission, submission_path=submission_path,
        probe_binding=probe_binding,
    )
    if validate_slurm_logs(args.job_id) != slurm_logs:
        die("terminal Slurm log bytes changed")
    if observe_sacct(args.job_id, submission) != sacct:
        die("terminal Slurm accounting changed")
    exact_file(EXPECTED_WRAPPER, EXPECTED_WRAPPER, EXPECTED_WRAPPER_SHA256, "wrapper")
    exact_file(EXPECTED_PAYLOAD, EXPECTED_PAYLOAD, EXPECTED_PAYLOAD_SHA256, "payload")
    exact_file(EXPECTED_GUARD, EXPECTED_GUARD, EXPECTED_GUARD_SHA256, "guard")
    exact_file(EXPECTED_RUNTIME, EXPECTED_RUNTIME, EXPECTED_RUNTIME_SHA256, "runtime")
    exact_file(
        EXPECTED_SOURCE_ARCHIVE, EXPECTED_SOURCE_ARCHIVE,
        EXPECTED_SOURCE_ARCHIVE_SHA256, "source archive",
    )
    exact_file(
        EXPECTED_PROBE_VALIDATOR, EXPECTED_PROBE_VALIDATOR,
        EXPECTED_PROBE_VALIDATOR_SHA256, "probe validator",
    )
    exact_file(
        EXPECTED_RELEASE_MANIFEST, EXPECTED_RELEASE_MANIFEST,
        args.release_manifest_sha256, "release manifest",
    )
    exact_file(
        EXPECTED_POSTFLIGHT, EXPECTED_POSTFLIGHT,
        args.postflight_sha256, "postflight",
    )
    for label, path in (
        ("postflight", EXPECTED_POSTFLIGHT),
        ("release_manifest", EXPECTED_RELEASE_MANIFEST),
        ("guard", EXPECTED_GUARD),
        ("source_archive", EXPECTED_SOURCE_ARCHIVE),
        ("probe_validator", EXPECTED_PROBE_VALIDATOR),
    ):
        info = path.lstat()
        expected_identity = retained_identities[label]
        if (info.st_dev, info.st_ino) != (
            expected_identity["device"], expected_identity["inode"]
        ):
            die(f"{label} retained pathname identity changed")
    close_before_publication(descriptors)
    guard.write_create_only(admission_path, sealed_admission)
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
