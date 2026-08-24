#!/usr/bin/env python3
"""Admit exactly one completed in-allocation WORLD8 step, never its parent job."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Sequence


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
PARENT_ALLOCATION_JOB_ID = "134936"
EXPECTED_NODE = "auh7-1b-gpu-185"
STEP_RECEIPT_SCHEMA = "saic-formal-v2-retained-fd-world8-inallocation-r14-step-launch-v1"
STEM = "saic-formal-v2-retained-fd-world8-canary-inallocation-r14"
ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809"
)
RELEASE = ROOT / "releases" / STEM
OUTPUT_PARENT = ROOT / "canaries" / STEM
LOG_DIR = ROOT / "slurm" / STEM
STEP_RECEIPT = OUTPUT_PARENT / "inallocation-r14-step-launch-receipt.json"
CLIENT_RECEIPT = OUTPUT_PARENT / "inallocation-r14-srun-client-receipt.json"
ADMISSION = OUTPUT_PARENT / "inallocation-r14-step-admission.json"
RELEASE_MANIFEST = RELEASE / "release-manifest.json"
EXPECTED_GUARD = RELEASE / "inputs/saic_t2v_rendezvous_guard_v2.py"
EXPECTED_GUARD_SHA256 = "1a38b2ac18f46e818b1596db884b883dff8e0612fcd5b0cf1ed78aca377ac965"
EXPECTED_WRAPPER_SHA256 = "67c160e045d361a9e95604c660a918cc1ec41b7a6a7cbaf24618712c37118f42"
EXPECTED_PAYLOAD_SHA256 = "a7a161e03c807a7ba006adb102caf3a2b209dfa20d030fd96a57d823d9568fb8"
EXPECTED_RUNTIME_SHA256 = "3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36"
EXPECTED_ARCHIVE_SHA256 = "3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b"
EXPECTED_ARCHIVE_MEMBER_MANIFEST_SHA256 = "1f3c8af23f5b4d416cea04476900c5d479ad3000338746e11f0e655b995b0fcc"
EXPECTED_RUNTIME_ORIGIN_MANIFEST_SHA256 = "2e9360581b21b56e6998e1e5db8df98e4cc66acf95fbb7819baffd1161eb98ba"
EXPECTED_SACCT = Path("/usr/bin/sacct")
EXPECTED_SACCT_SHA256 = "fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e"
SACCT_FIELDS = [
    "JobIDRaw", "JobName", "State", "ExitCode", "AllocTRES", "NodeList",
    "Start", "End", "SubmitLine%8192",
]
AUTHORITY = {
    "scientific": False,
    "generation": False,
    "training": False,
    "publication": False,
    "formal_job_authorized": False,
}
RELEASE_MANIFEST_FIELDS = {
    "schema_version", "status", "stem", "release_root", "output_parent",
    "log_directory", "parent_allocation_job_id", "expected_node", "inputs",
    "postflight", "immutable_ancestor", "executables", "probe_admission",
    "authority", "receipt_digest",
}


def die(message: str) -> None:
    raise SystemExit(f"postflight-saic-fv2-fd-world8-inallocation-r14: {message}")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def retained_bytes(path: Path, label: str) -> tuple[bytes, os.stat_result]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        leaf = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or not stat.S_ISREG(leaf.st_mode) or stat.S_ISLNK(leaf.st_mode)
            or (before.st_dev, before.st_ino) != (leaf.st_dev, leaf.st_ino)
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
        leaf_after = path.lstat()
        identity = lambda info: (
            info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            stat.S_IMODE(info.st_mode), info.st_nlink, info.st_uid,
        )
        if (
            identity(before) != identity(after) or after.st_size != len(raw)
            or (after.st_dev, after.st_ino)
            != (leaf_after.st_dev, leaf_after.st_ino)
        ):
            die(f"{label} changed during retained read")
        return raw, after
    finally:
        os.close(descriptor)


def sha_file(path: Path) -> str:
    raw, _ = retained_bytes(path, str(path))
    return hashlib.sha256(raw).hexdigest()


def exact_file(path: Path, expected_sha: str, label: str, mode: int = 0o444) -> Path:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        die(f"{label} identity differs")
    raw, info = retained_bytes(path, label)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != mode
        or hashlib.sha256(raw).hexdigest() != expected_sha
    ):
        die(f"{label} bytes/mode differ")
    return path


def retained_directory_identity(path: Path, label: str) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        leaf = path.lstat()
        if (
            not stat.S_ISDIR(before.st_mode) or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o700
            or not stat.S_ISDIR(leaf.st_mode) or stat.S_ISLNK(leaf.st_mode)
            or (before.st_dev, before.st_ino) != (leaf.st_dev, leaf.st_ino)
        ):
            die(f"{label} retained identity differs")
        after = os.fstat(descriptor)
        leaf_after = path.lstat()
        if (
            (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or (after.st_dev, after.st_ino)
            != (leaf_after.st_dev, leaf_after.st_ino)
        ):
            die(f"{label} changed during retained check")
        return f"{after.st_dev}:{after.st_ino}"
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sealed(path: Path, schema: str, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        die(f"{label} identity differs")
    raw, info = retained_bytes(path, label)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o444
    ):
        die(f"{label} mode differs")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        die(f"{label} encoding differs: {error}")
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        die(f"{label} schema differs")
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_digest", None)
    if (
        not isinstance(claimed, str) or SHA256.fullmatch(claimed) is None
        or claimed != object_sha(unsigned) or raw != canonical(value) + b"\n"
    ):
        die(f"{label} seal differs")
    return value, raw


def parse_tres(value: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for item in value.split(","):
        key, separator, amount = item.partition("=")
        if not separator or not key or key in rows:
            die("AllocTRES encoding differs")
        rows[key] = amount
    return rows


def run_sacct(job_step_id: str) -> tuple[dict[str, str], bytes, bytes]:
    command = [
        str(EXPECTED_SACCT), "-j", job_step_id, "--noheader", "-n", "-P",
        "-o", ",".join(SACCT_FIELDS),
    ]
    if "-X" in command:
        die("whole-job-only accounting option is forbidden")
    completed = subprocess.run(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False, timeout=60,
        env={
            "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
            "SLURM_BITSTR_LEN": "8192",
        },
    )
    if completed.returncode != 0 or completed.stderr:
        die(
            "sacct query failed: "
            f"exit={completed.returncode} stderr_sha256={hashlib.sha256(completed.stderr).hexdigest()}"
        )
    try:
        text = completed.stdout.decode("ascii")
    except UnicodeDecodeError:
        die("sacct output is not ASCII")
    lines = text.splitlines()
    if len(lines) != 1:
        die("sacct did not return exactly one step row")
    columns = lines[0].split("|")
    if columns and columns[-1] == "":
        columns.pop()
    if len(columns) != len(SACCT_FIELDS):
        die("sacct field count differs")
    keys = [field.split("%", 1)[0] for field in SACCT_FIELDS]
    return dict(zip(keys, columns)), completed.stdout, completed.stderr


def write_create_only(path: Path, value: dict[str, Any]) -> None:
    raw = canonical(value) + b"\n"
    parent_fd = os.open(
        path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    parent_before = os.fstat(parent_fd)
    parent_leaf = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or stat.S_IMODE(parent_before.st_mode) != 0o700
        or (parent_before.st_dev, parent_before.st_ino)
        != (parent_leaf.st_dev, parent_leaf.st_ino)
    ):
        os.close(parent_fd)
        die("admission parent retained identity differs")
    descriptor = os.open(
        path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600, dir_fd=parent_fd,
    )
    try:
        view = memoryview(raw)
        while view:
            wrote = os.write(descriptor, view)
            if wrote <= 0:
                die("admission write stalled")
            view = view[wrote:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        info = os.fstat(descriptor)
        public = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        parent_after = os.fstat(parent_fd)
        if (
            not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o444 or info.st_size != len(raw)
            or (info.st_dev, info.st_ino) != (public.st_dev, public.st_ino)
            or (parent_after.st_dev, parent_after.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
        ):
            die("admission retained seal differs")
        os.fsync(parent_fd)
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def retain_exact_log(
    path: Path, *, label: str, expected_payload: bytes,
) -> tuple[str, int, str]:
    descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        leaf_before = path.lstat()
        identity = lambda info: (
            info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            stat.S_IMODE(info.st_mode), info.st_nlink, info.st_uid,
        )
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or not stat.S_ISREG(leaf_before.st_mode)
            or stat.S_ISLNK(leaf_before.st_mode)
            or (before.st_dev, before.st_ino)
            != (leaf_before.st_dev, leaf_before.st_ino)
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
        leaf_after = path.lstat()
        if (
            identity(before) != identity(after)
            or (after.st_dev, after.st_ino)
            != (leaf_after.st_dev, leaf_after.st_ino)
            or after.st_size != len(raw) or raw != expected_payload
        ):
            die(f"{label} changed or content differs")
        final_before = after
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        final_raw = os.read(descriptor, final_before.st_size + 1)
        final_after = os.fstat(descriptor)
        public = path.lstat()
        if (
            final_raw != expected_payload or final_after.st_size != len(final_raw)
            or stat.S_IMODE(final_after.st_mode) != 0o400
            or final_after.st_nlink != 1 or final_after.st_uid != os.getuid()
            or (final_after.st_dev, final_after.st_ino)
            != (final_before.st_dev, final_before.st_ino)
            or (public.st_dev, public.st_ino)
            != (final_after.st_dev, final_after.st_ino)
        ):
            die(f"{label} final seal differs")
        return (
            hashlib.sha256(final_raw).hexdigest(), len(final_raw),
            f"{final_after.st_dev}:{final_after.st_ino}",
        )
    finally:
        os.close(descriptor)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--release-manifest-sha256", required=True)
    value.add_argument("--release-manifest-digest", required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not SHA256.fullmatch(args.release_manifest_sha256):
        die("release manifest SHA differs")
    if not SHA256.fullmatch(args.release_manifest_digest):
        die("release manifest digest differs")
    exact_file(EXPECTED_SACCT, EXPECTED_SACCT_SHA256, "sacct", mode=0o755)
    exact_file(EXPECTED_GUARD, EXPECTED_GUARD_SHA256, "guard")
    output_parent_identity = retained_directory_identity(
        OUTPUT_PARENT, "output parent"
    )
    log_directory_identity = retained_directory_identity(LOG_DIR, "log directory")
    receipt, receipt_raw = sealed(STEP_RECEIPT, STEP_RECEIPT_SCHEMA, "step receipt")
    expected_receipt_fields = {
        "schema_version", "status", "parent_allocation_job_id", "step_id",
        "job_step_id", "node", "exact_srun_argv", "exact_srun_argv_digest",
        "output_parent_identity", "log_directory_identity",
        "compute_bootstrap_sha256",
        "compute_bootstrap_size_bytes",
        "release_manifest", "release_manifest_file_sha256",
        "release_manifest_digest", "step_success", "parent_job_success",
        "bootstrap_boundary", "authority", "receipt_digest",
    }
    step_id = receipt.get("step_id")
    job_step_id = receipt.get("job_step_id")
    if (
        set(receipt) != expected_receipt_fields
        or receipt.get("status") != "compute_step_bootstrap_admitted"
        or receipt.get("parent_allocation_job_id") != PARENT_ALLOCATION_JOB_ID
        or not isinstance(step_id, str) or re.fullmatch(r"[0-9]+", step_id) is None
        or job_step_id != f"{PARENT_ALLOCATION_JOB_ID}.{step_id}"
        or receipt.get("node") != EXPECTED_NODE
        or receipt.get("output_parent_identity") != output_parent_identity
        or receipt.get("log_directory_identity") != log_directory_identity
        or receipt.get("step_success") is not None
        or receipt.get("parent_job_success") is not None
        or receipt.get("release_manifest_file_sha256")
        != args.release_manifest_sha256
        or receipt.get("release_manifest_digest") != args.release_manifest_digest
        or receipt.get("exact_srun_argv_digest")
        != object_sha(receipt.get("exact_srun_argv"))
        or receipt.get("bootstrap_boundary") != {
            "receipt_reserved_before_srun": True,
            "receipt_same_inode": True,
            "receipt_opened_o_nofollow_inside_step": True,
            "wrapper_opened_o_nofollow_inside_step": True,
            "wrapper_executed_from_retained_fd": True,
            "compute_bootstrap_transported_over_srun_stdin": True,
            "compute_bootstrap_stdin_sha256_verified_inside_step": True,
            "compute_bootstrap_pathname_execution": False,
            "compute_bootstrap_interpreter": "/usr/bin/python3",
            "compute_bootstrap_interpreter_trust": "host_os_absolute_path",
            "science_python_opened_o_nofollow_inside_step": True,
            "science_python_retained_fd_prepared_for_wrapper": True,
            "receipt_success_mode": "0444",
        }
        or receipt.get("authority") != AUTHORITY
    ):
        die("step receipt content differs")
    client_receipt, client_receipt_raw = sealed(
        CLIENT_RECEIPT,
        "saic-formal-v2-retained-fd-world8-inallocation-r14-srun-client-v1",
        "srun client receipt",
    )
    client_unsigned = dict(client_receipt)
    client_unsigned.pop("receipt_digest", None)
    if (
        set(client_receipt) != {
            "schema_version", "status", "parent_allocation_job_id",
            "job_step_id", "exact_srun_argv_digest",
            "step_launch_receipt_digest", "srun_client_returncode",
            "srun_client_stdout_sha256", "srun_client_stdout_size",
            "srun_client_stderr_sha256", "srun_client_stderr_size",
            "parent_job_success", "authority", "receipt_digest",
        }
        or client_receipt.get("status") != "srun_client_terminal_success"
        or client_receipt.get("parent_allocation_job_id")
        != PARENT_ALLOCATION_JOB_ID
        or client_receipt.get("job_step_id") != job_step_id
        or client_receipt.get("exact_srun_argv_digest")
        != receipt.get("exact_srun_argv_digest")
        or client_receipt.get("step_launch_receipt_digest")
        != receipt.get("receipt_digest")
        or client_receipt.get("srun_client_returncode") != 0
        or client_receipt.get("srun_client_stdout_sha256")
        != hashlib.sha256(b"").hexdigest()
        or client_receipt.get("srun_client_stdout_size") != 0
        or client_receipt.get("srun_client_stderr_sha256")
        != hashlib.sha256(b"").hexdigest()
        or client_receipt.get("srun_client_stderr_size") != 0
        or client_receipt.get("parent_job_success") is not None
        or client_receipt.get("authority") != AUTHORITY
    ):
        die("srun client receipt content differs")
    release_manifest_path = Path(receipt["release_manifest"])
    if (
        release_manifest_path != RELEASE_MANIFEST
        or sha_file(release_manifest_path)
        != receipt["release_manifest_file_sha256"]
    ):
        die("release manifest file binding differs")
    manifest, _ = sealed(
        release_manifest_path,
        "saic-formal-v2-retained-fd-world8-inallocation-r14-release-manifest-v1",
        "release manifest",
    )
    if (
        set(manifest) != RELEASE_MANIFEST_FIELDS
        or manifest.get("status") != "sealed_before_inallocation_step"
        or manifest.get("stem") != STEM
        or manifest.get("release_root") != str(RELEASE)
        or manifest.get("output_parent") != str(OUTPUT_PARENT)
        or manifest.get("log_directory") != str(LOG_DIR)
        or manifest.get("parent_allocation_job_id") != PARENT_ALLOCATION_JOB_ID
        or manifest.get("expected_node") != EXPECTED_NODE
        or manifest.get("receipt_digest") != receipt["release_manifest_digest"]
        or manifest.get("inputs", {}).get("guard") != {
            "path": str(EXPECTED_GUARD), "sha256": EXPECTED_GUARD_SHA256,
        }
        or manifest.get("inputs", {}).get("payload", {}).get("sha256")
        != EXPECTED_PAYLOAD_SHA256
        or manifest.get("inputs", {}).get("wrapper", {}).get("sha256")
        != EXPECTED_WRAPPER_SHA256
        or manifest.get("inputs", {}).get("source_archive", {}).get("sha256")
        != EXPECTED_ARCHIVE_SHA256
        or set(manifest.get("inputs", {})) != {
            "guard", "launcher", "payload", "probe_validator", "runtime",
            "source_archive", "wrapper",
        }
        or manifest.get("immutable_ancestor", {}).get("guard_source_path")
        != str(ROOT / "releases/saic-formal-v2-retained-fd-world8-canary-96335bf5-fb3f1ac4-r10/inputs/saic_t2v_rendezvous_guard_v2.py")
        or manifest.get("immutable_ancestor", {}).get("guard_sha256")
        != EXPECTED_GUARD_SHA256
        or manifest.get("immutable_ancestor", {}).get(
            "guard_copied_from_external_immutable_release"
        ) is not True
        or manifest.get("immutable_ancestor", {}).get(
            "local_guard_source_forbidden"
        ) is not True
        or manifest.get("executables") != {
            "python": "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12",
            "python_sha256": "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a",
            "compute_bootstrap_python": "/usr/bin/python3",
            "compute_bootstrap_python_trust": "host_os_absolute_path",
            "sacct": "/usr/bin/sacct", "sacct_sha256": EXPECTED_SACCT_SHA256,
        }
        or manifest.get("probe_admission") != {
            "path": str(ROOT / "canaries/compute-bash-retained-fd-probe-8283e73d-r1/probe-admission.json"),
            "file_sha256": "d51ebf1f894d63483943042faaa2c6ccbf812c0f93769980084bae72f8ab84d8",
            "receipt_digest": "a37e44c12f935a4f4e11ab08364b019799809fd13299af6459ca6797a8333fb7",
        }
        or manifest.get("authority") != AUTHORITY
    ):
        die("release manifest closure differs")
    for label, binding in manifest["inputs"].items():
        if (
            not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
            or not isinstance(binding.get("sha256"), str)
            or SHA256.fullmatch(binding["sha256"]) is None
            or sha_file(Path(binding["path"])) != binding["sha256"]
        ):
            die(f"release input {label} closure differs")
    launcher_path = Path(manifest["inputs"]["launcher"]["path"])
    launcher_raw, _ = retained_bytes(launcher_path, "release launcher")
    try:
        launcher_source = launcher_raw.decode("utf-8")
    except UnicodeDecodeError:
        die("release launcher encoding differs")
    launcher_tree = ast.parse(launcher_source)
    bootstrap_values = [
        ast.literal_eval(node.value)
        for node in launcher_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "BOOTSTRAP"
                for target in node.targets)
    ]
    if (
        len(bootstrap_values) != 1
        or not isinstance(bootstrap_values[0], str)
        or hashlib.sha256(bootstrap_values[0].encode("ascii")).hexdigest()
        != receipt.get("compute_bootstrap_sha256")
        or len(bootstrap_values[0].encode("ascii"))
        != receipt.get("compute_bootstrap_size_bytes")
    ):
        die("compute bootstrap stdin bytes differ")
    output_root = OUTPUT_PARENT / f"step-{job_step_id}"
    evidence_path = output_root / "operational-evidence.json"
    evidence, evidence_raw = sealed(
        evidence_path,
        "saic-formal-v2-retained-fd-world8-inallocation-r14-operational-evidence-v1",
        "operational evidence",
    )
    if (
        evidence.get("status") != "retained_fd_operational_evidence_complete"
        or evidence.get("parent_allocation_job_id") != PARENT_ALLOCATION_JOB_ID
        or evidence.get("step_id") != step_id
        or evidence.get("job_step_id") != job_step_id
        or evidence.get("node") != EXPECTED_NODE
        or evidence.get("step_success") is not None
        or evidence.get("parent_job_success") is not None
        or evidence.get("step_terminal_verified") is not False
        or evidence.get("formal_admission") is not False
        or evidence.get("topology")
        != "two_concurrent_world4_on_one_requested_8mi210_node"
        or evidence.get("requested_gpu_count") != 8
        or evidence.get("world_size_total") != 8
        or evidence.get("group_count") != 2
        or evidence.get("rank_packet_count") != 8
        or evidence.get("guard_sha256") != EXPECTED_GUARD_SHA256
        or evidence.get("payload_sha256") != EXPECTED_PAYLOAD_SHA256
        or evidence.get("wrapper_sha256") != EXPECTED_WRAPPER_SHA256
        or evidence.get("source_archive_sha256") != EXPECTED_ARCHIVE_SHA256
        or evidence.get("archive_member_manifest_sha256")
        != EXPECTED_ARCHIVE_MEMBER_MANIFEST_SHA256
        or evidence.get("archive_member_count") != 864
        or evidence.get("archive_regular_file_count") != 853
        or evidence.get("archive_directory_count") != 11
        or evidence.get("runtime_origin_manifest_sha256")
        != EXPECTED_RUNTIME_ORIGIN_MANIFEST_SHA256
        or evidence.get("runtime_origin_project_module_count") != 14
        or evidence.get("canonical_runtime_origin_verified") is not True
        or evidence.get("canonical_runtime_origin_sha256")
        != EXPECTED_RUNTIME_SHA256
        or evidence.get("canonical_runtime_origin_path")
        != "methods/bernini_action_editing/generate_saic_pure_t2v_event_bank_topup_v2.py"
        or evidence.get("science_python_opened_o_nofollow_inside_step") is not True
        or evidence.get(
            "science_python_controller_invocations_executed_from_retained_fd"
        ) is not True
        or evidence.get("science_python_fd_exec_argv0_preserved") is not True
        or evidence.get("torchrun_no_python_worker_entrypoint_retained_fd")
        is not True
        or evidence.get("stage0_inherited_source_fd_numbers", {}).get(
            "science_python"
        ) is None
        or evidence.get("inallocation_step_wrapper_executed_from_retained_fd")
        is not True
        or evidence.get("scientific_generation_entered") is not False
        or evidence.get("scientific_output_created") is not False
        or evidence.get("formal_full60_result_claimed") is not False
        or evidence.get("authority") != AUTHORITY
        or evidence.get("step_launch_receipt_file_sha256")
        != hashlib.sha256(receipt_raw).hexdigest()
        or evidence.get("step_launch_receipt_digest") != receipt["receipt_digest"]
        or evidence.get("exact_srun_argv_digest")
        != receipt["exact_srun_argv_digest"]
        or evidence.get("compute_bootstrap_sha256")
        != receipt["compute_bootstrap_sha256"]
        or evidence.get("compute_bootstrap_size_bytes")
        != receipt["compute_bootstrap_size_bytes"]
    ):
        die("operational evidence content differs")
    rows = evidence.get("candidate_rows")
    if (
        not isinstance(rows, list) or len(rows) != 2
        or {row.get("group_id") for row in rows if isinstance(row, dict)}
        != {"sp4-a", "sp4-b"}
    ):
        die("two-WORLD4 evidence differs")

    accounting, sacct_stdout, sacct_stderr = run_sacct(job_step_id)
    tres = parse_tres(accounting["AllocTRES"])
    if (
        accounting["JobIDRaw"] != job_step_id
        or accounting["JobName"] != "saic-fv2-fd-w8-inalloc-r14"
        or accounting["State"] != "COMPLETED"
        or accounting["ExitCode"] != "0:0"
        or accounting["NodeList"] != EXPECTED_NODE
        or tres.get("cpu") != "16"
        or tres.get("mem") != "32G"
        or tres.get("node") != "1"
        or tres.get("gres/gpu") != "8"
        or tres.get("gres/gpu:mi210") != "8"
        or set(tres) != {"cpu", "gres/gpu", "gres/gpu:mi210", "mem", "node"}
        or not accounting["Start"] or accounting["Start"] == "Unknown"
        or not accounting["End"] or accounting["End"] == "Unknown"
    ):
        die("authoritative step accounting differs")
    expected_command = receipt["exact_srun_argv"]
    if (
        not isinstance(expected_command, list) or len(expected_command) <= 14
        or expected_command[0] != "/usr/bin/srun"
        or expected_command[1] != "--jobid=134936"
        or expected_command.count("--job-name=saic-fv2-fd-w8-inalloc-r14") != 1
        or expected_command.count("--nodes=1") != 1
        or expected_command.count("--ntasks=1") != 1
        or expected_command.count("--cpus-per-task=16") != 1
        or expected_command.count("--mem=32G") != 1
        or expected_command.count("--gres=gpu:mi210:8") != 1
        or expected_command.count(f"--nodelist={EXPECTED_NODE}") != 1
        or expected_command.count("--overlap") != 1
        or expected_command.count("--exact") != 1
        or expected_command.count("--input=0") != 1
        or expected_command.count("--open-mode=truncate") != 1
        or expected_command.count(
            f"--output={LOG_DIR}/saic-fv2-fd-w8-inalloc-r14-%J.out"
        ) != 1
        or expected_command.count(
            f"--error={LOG_DIR}/saic-fv2-fd-w8-inalloc-r14-%J.err"
        ) != 1
        or "/usr/bin/python3" not in expected_command
    ):
        die("recorded srun command structure differs")
    expected_submit_line = " ".join(expected_command)
    expected_basename_submit_line = " ".join(["srun", *expected_command[1:]])
    observed_submit_line = accounting["SubmitLine"]
    if observed_submit_line not in {
        expected_submit_line, expected_basename_submit_line,
    }:
        die("step SubmitLine differs from the exact srun argv")
    stdout_log = LOG_DIR / f"saic-fv2-fd-w8-inalloc-r14-{job_step_id}.out"
    stderr_log = LOG_DIR / f"saic-fv2-fd-w8-inalloc-r14-{job_step_id}.err"
    sentinel = f"SAIC_FV2_FD_WORLD8_INALLOCATION_R14_PASS {job_step_id}\n".encode(
        "ascii"
    )
    stdout_sha, stdout_size, stdout_identity = retain_exact_log(
        stdout_log, label="stdout log", expected_payload=sentinel,
    )
    stderr_sha, stderr_size, stderr_identity = retain_exact_log(
        stderr_log, label="stderr log", expected_payload=b"",
    )
    fsync_directory(LOG_DIR)

    core = {
        "schema_version": (
            "saic-formal-v2-retained-fd-world8-inallocation-r14-step-admission-v1"
        ),
        "status": "inallocation_step_admitted",
        "parent_allocation_job_id": PARENT_ALLOCATION_JOB_ID,
        "step_id": step_id,
        "job_step_id": job_step_id,
        "node": EXPECTED_NODE,
        "step_success": True,
        "parent_job_success": None,
        "parent_job_terminal_state_not_claimed": True,
        "accounting_query_exact_step_only": True,
        "accounting_query_forbidden_dash_x_absent": True,
        "sacct_fields": SACCT_FIELDS,
        "sacct_row": accounting,
        "sacct_stdout_sha256": hashlib.sha256(sacct_stdout).hexdigest(),
        "sacct_stderr_sha256": hashlib.sha256(sacct_stderr).hexdigest(),
        "exact_srun_argv_digest": receipt["exact_srun_argv_digest"],
        "submit_line_exact": True,
        "alloc_tres_exact": tres,
        "logs": {
            "stdout": str(stdout_log), "stdout_sha256": stdout_sha,
            "stdout_size": stdout_size, "stdout_identity": stdout_identity,
            "stdout_exact_pass_sentinel": True,
            "stderr": str(stderr_log), "stderr_sha256": stderr_sha,
            "stderr_size": stderr_size, "stderr_identity": stderr_identity,
            "stderr_exact_empty": True,
            "both_logs_sealed_0400_after_retained_read": True,
        },
        "step_launch_receipt": str(STEP_RECEIPT),
        "step_launch_receipt_file_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "step_launch_receipt_digest": receipt["receipt_digest"],
        "srun_client_receipt": str(CLIENT_RECEIPT),
        "srun_client_receipt_file_sha256": hashlib.sha256(
            client_receipt_raw
        ).hexdigest(),
        "srun_client_receipt_digest": client_receipt["receipt_digest"],
        "operational_evidence": str(evidence_path),
        "operational_evidence_file_sha256": hashlib.sha256(evidence_raw).hexdigest(),
        "operational_evidence_digest": evidence["receipt_digest"],
        "scientific_generation_entered": False,
        "formal_full60_result_claimed": False,
        "authority": AUTHORITY,
    }
    value = dict(core)
    value["receipt_digest"] = object_sha(core)
    if ADMISSION.exists() or ADMISSION.is_symlink():
        die("step admission is not fresh")
    write_create_only(ADMISSION, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
