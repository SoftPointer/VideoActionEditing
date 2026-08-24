#!/bin/bash -p
set -euo pipefail
umask 077

# Frozen against the fresh r5g materialization receipt and exact package bytes.
readonly R5G_CONTROLLER_STATE=READY
[[ "$R5G_CONTROLLER_STATE" == READY ]] || { printf '%s\n' 'r5g full16 CPU gate controller is HOLD pending frozen materialization pins' >&2; exit 88; }

[[ "$-" == *p* ]] || { printf '%s\n' 'controller must be streamed to /bin/bash -p -s' >&2; exit 90; }
[[ "$0" == "bash" || "$0" == "/bin/bash" || "$0" == "-bash" ]] || { printf '%s\n' 'named controller execution forbidden' >&2; exit 91; }
[[ -z "${BASH_ENV-}" && -z "${ENV-}" ]] || { printf '%s\n' 'ambient shell startup variable present' >&2; exit 92; }
if shopt -q varredir_close 2>/dev/null; then shopt -u varredir_close; fi
readonly ROOT_PYTHON=/usr/bin/python3.10
exec {ROOT_PYTHON_FD}<"$ROOT_PYTHON"
[[ "$ROOT_PYTHON_FD" =~ ^[0-9]+$ ]] || exit 93
exec -c "/proc/self/fd/$ROOT_PYTHON_FD" -I -S -B - "$ROOT_PYTHON_FD" bootstrap <<'PY'
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time

ROOT = "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_matched_eval_auh_r5f_job143812_node293_full16_847b91a2_c91de7eb_d70eac5c_r1"
JOB_ID = "143812"
NODE = "auh7-1b-gpu-293"
ROOT_PYTHON = "/usr/bin/python3.10"
ROOT_PYTHON_SHA256 = "11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96"
ROOT_PYTHON_SIZE = 5937800
SRUN = "/usr/bin/srun"
SRUN_SHA256 = "2b8f60b30edf7efed35bb00864651da1b0bec68e75f942ce58b5ff82bc43cd9e"
SRUN_SIZE = 164720
SACCT = "/usr/bin/sacct"
SACCT_SHA256 = "fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e"
SACCT_SIZE = 85952
PLAN_PATH = ROOT + "/plan/full644_exploratory_matched_plan_auh_r5d.json"
PLAN_SHA256 = "097b601d180ee7122230fa7d98dcac9c7102489195c065a6d03eb7e38131dfbe"
PLAN_DIGEST = "2136926b734796333788ab9f296e6cca076989bfc25b7a589eae773d42b61a00"
LAUNCH_INPUT_SHA256 = "913b64846383f329d586d6fbc737b8d2898f4eb468aaa4fc9ae85d65f0679d8a"
LAUNCH_RECEIPT_SHA256 = "3c46cebe92fd4585132d8cad5eaba1cb0189338b078730ae98d56b2571570cae"
LAUNCH_RECEIPT_DIGEST = "b0a9a6717b10b927414730777284f1f8c60c156a6c82a611fda876577939fde3"
RELEASE_DIGEST = "7678fdaf13223403093b810aeb1440f96d96f9238cf4e55caa01896623edee14"
BOOTSTRAP_RELEASE_DIGEST = "65e410335c0d25aef2458c776f5041b5dbcb4ea2aebf35d744df6830cff3e74b"
PRODUCTION_PAYLOAD_SHA256 = "c1a7154fa4f7347fd01cb57fcbb3b0a5d5765c3bfc835be6b434d1abede2d115"
PRODUCTION_PAYLOAD_SIZE = 26721
STATIC_PAYLOAD_SIZE = 6263
BOOTSTRAP_PAYLOAD_SIZE = 26934
SELECTED = [f"shared8-{index:02d}-{arm}" for index in range(8) for arm in ("base", "full644")]
CONFIGS = {
    "static": {
        "step_name": "f644-r5g-full16-static-v1",
        "payload": ROOT + "/diagnostics/static_nomodel_probe_payload_r5d.sh",
        "payload_sha256": "90eeaf8fae75231a730b38dc197b122675918045f939f449254299b8eb2bcb3b",
        "payload_size": STATIC_PAYLOAD_SIZE,
        "parent": ROOT + "/evidence",
        "result": "static_nomodel_probe_receipt_r5d.json",
        "stdout": "r5g_full16_static_nomodel_probe.stdout.log",
        "stderr": "r5g_full16_static_nomodel_probe.stderr.log",
        "evidence": "r5g_full16_static_nomodel_probe.sacct-and-replay.json",
        "sentinel": "R5G_FULL16_STATIC_NOMODEL_PASS",
    },
    "bootstrap": {
        "step_name": "f644-r5g-full16-bootstrap-v1",
        "payload": ROOT + "/diagnostics/root_bootstrap_probe_payload_r5d.sh",
        "payload_sha256": "8eabdc7ccff1e51a35da7deaf2323bd3874dcd8a64a5ee85d573b4059143e61e",
        "payload_size": BOOTSTRAP_PAYLOAD_SIZE,
        "parent": ROOT + "/evidence",
        "result": "root_bootstrap_cpu_probe_receipt_r5d.json",
        "stdout": "r5g_full16_root_bootstrap_cpu_probe.stdout.log",
        "stderr": "r5g_full16_root_bootstrap_cpu_probe.stderr.log",
        "evidence": "r5g_full16_root_bootstrap_cpu_probe.sacct-and-replay.json",
        "sentinel": "R5G_FULL16_ROOT_BOOTSTRAP_CPU_PROBE_PASS",
    },
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def identity(value):
    return {
        "device": value.st_dev, "inode": value.st_ino, "uid": value.st_uid,
        "gid": value.st_gid, "mode": value.st_mode, "nlink": value.st_nlink,
        "rdev": value.st_rdev, "size": value.st_size,
        "blocks": getattr(value, "st_blocks", 0), "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def object_identity(value):
    return (value.st_dev, value.st_ino, value.st_uid, value.st_gid, value.st_mode, value.st_nlink, value.st_rdev)


def pread_exact(descriptor, size):
    chunks = []
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not block:
            break
        chunks.append(block)
        offset += len(block)
    raw = b"".join(chunks)
    if len(raw) != size:
        raise RuntimeError("short pread")
    return raw


def verify_self(descriptor):
    if descriptor < 3 or not os.get_inheritable(descriptor) or os.lseek(descriptor, 0, os.SEEK_CUR) != 0:
        raise RuntimeError("root Python inherited FD or offset differs")
    before = os.fstat(descriptor)
    named = os.lstat(ROOT_PYTHON)
    process = os.stat("/proc/self/exe")
    raw = pread_exact(descriptor, before.st_size)
    after = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o755
        or before.st_uid != 0 or before.st_gid != 0 or before.st_nlink != 1
        or before.st_size != ROOT_PYTHON_SIZE or identity(before) != identity(named)
        or identity(before) != identity(process) or identity(before) != identity(after)
        or hashlib.sha256(raw).hexdigest() != ROOT_PYTHON_SHA256
        or os.lseek(descriptor, 0, os.SEEK_CUR) != 0
    ):
        raise RuntimeError("root Python authority differs")
    os.set_inheritable(descriptor, False)
    if os.get_inheritable(descriptor):
        raise RuntimeError("root Python FD remained inheritable")
    return before


def open_pinned(path, expected_sha256, expected_size, expected_uid, expected_gid, expected_mode):
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    before = os.fstat(descriptor)
    named = os.lstat(path)
    raw = pread_exact(descriptor, before.st_size)
    after = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != expected_mode
        or before.st_uid != expected_uid or before.st_gid != expected_gid or before.st_nlink != 1
        or before.st_size != expected_size or identity(before) != identity(named)
        or identity(before) != identity(after) or hashlib.sha256(raw).hexdigest() != expected_sha256
        or os.lseek(descriptor, 0, os.SEEK_CUR) != 0
    ):
        os.close(descriptor)
        raise RuntimeError("pinned file authority differs: " + path)
    return descriptor, before, raw


def replay_pinned(descriptor, path, expected_before, expected_sha256):
    before = os.fstat(descriptor)
    named = os.lstat(path)
    raw = pread_exact(descriptor, before.st_size)
    after = os.fstat(descriptor)
    if identity(before) != identity(expected_before) or identity(before) != identity(named) or identity(before) != identity(after) or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError("pinned file post replay differs: " + path)


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def strict_json(raw):
    try:
        value = json.loads(raw, object_pairs_hook=no_duplicates, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (UnicodeError, ValueError, TypeError) as error:
        raise RuntimeError("strict JSON differs") from error
    if type(value) is not dict or raw != canonical(value) + b"\n":
        raise RuntimeError("canonical JSON bytes differ")
    return value


def fresh_at(parent_fd, name):
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise RuntimeError("fresh path gate failed: " + name)


def stable_read_at(parent_fd, name, expected_mode, allow_empty=False):
    descriptor = os.open(name, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
    try:
        before = os.fstat(descriptor)
        raw = pread_exact(descriptor, before.st_size)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != expected_mode
            or before.st_uid != 2012 or before.st_gid != 2000 or before.st_nlink != 1
            or (not raw and not allow_empty) or identity(before) != identity(after)
            or identity(before) != identity(named)
        ):
            raise RuntimeError("stable artifact replay differs: " + name)
        return raw, identity(before)
    finally:
        os.close(descriptor)


def write_create_only_at(parent_fd, name, raw):
    descriptor = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o000, dir_fd=parent_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0 or before.st_nlink != 1 or before.st_size != 0:
            raise RuntimeError("create-only staging identity differs")
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise RuntimeError("create-only write differs")
            offset += written
        os.fsync(descriptor)
        staged = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if pread_exact(descriptor, len(raw)) != raw or staged.st_size != len(raw) or staged.st_dev != before.st_dev or staged.st_ino != before.st_ino or identity(staged) != identity(named):
            raise RuntimeError("create-only staging replay differs")
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parse_memory(raw):
    match = re.fullmatch(r"([1-9][0-9]*)([KMGT])", raw)
    if match is None:
        raise RuntimeError("memory TRES syntax differs")
    return int(match.group(1)) * {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[match.group(2)]


def parse_tres(raw):
    result = {}
    for token in raw.split(","):
        if token.count("=") != 1:
            raise RuntimeError("TRES token differs")
        key, value = token.split("=", 1)
        if not key or not value or key in result:
            raise RuntimeError("TRES closure or duplicate differs")
        result[key] = value
    allowed = {"cpu", "mem", "node", "billing", "gres/gpu", "gres/gpu:mi210"}
    if not set(result).issubset(allowed):
        raise RuntimeError("unexpected TRES key")
    if (
        result.get("cpu") != "4" or result.get("node") != "1"
        or parse_memory(result.get("mem", "")) != 8 * 1024**3
        or result.get("gres/gpu") != "8" or result.get("gres/gpu:mi210") != "8"
        or ("billing" in result and result["billing"] != "4")
    ):
        raise RuntimeError("allocated TRES differs")
    return result


def parse_sacct_row(line, expected_step):
    columns = line.split("|")
    if not columns or columns[0] != expected_step:
        return None
    if len(columns) != 9:
        raise RuntimeError("sacct row field closure differs")
    return columns


def validate_sacct_raw_fixture():
    fixture = (
        "143812.179|f644-r5g-full16-static-v1|COMPLETED|0:0|1|4|"
        "cpu=4,gres/gpu:mi210=8,gres/gpu=8,mem=8192M,node=1|"
        "auh7-1b-gpu-293|85286912"
    )
    parsed = parse_sacct_row(fixture, "143812.179")
    if parsed is None or len(parsed) != 9 or parsed[-1] != "85286912":
        raise RuntimeError("AUH sacct exact-nine raw fixture differs")
    for hostile in (fixture + "|", fixture + "|extra", fixture.rsplit("|", 1)[0]):
        try:
            parse_sacct_row(hostile, "143812.179")
        except RuntimeError:
            continue
        raise RuntimeError("AUH sacct hostile raw fixture accepted")


def validate_receipt(variant, value, expected_step):
    body = dict(value)
    claimed = body.pop("receipt_digest", None)
    if type(claimed) is not str or not re.fullmatch(r"[0-9a-f]{64}", claimed) or digest(body) != claimed:
        raise RuntimeError("receipt digest differs")
    slurm_names = sorted([
        "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE", "SLURM_STEP_GPUS",
        "SLURM_NNODES", "SLURM_STEP_NUM_NODES", "SLURM_JOB_NODELIST", "SLURM_STEP_NODELIST",
    ])
    if variant == "static":
        expected_keys = {
            "schema_version", "status", "campaign_mode", "selected_task_ids", "unselected_task_count",
            "formal_full16_report", "canary_stops_after_pair_for_manual_visual_review", "root", "holder_job_id",
            "expected_node", "slurm_step_id", "slurm_environment_source_names", "slurm_fields_observed_absent",
            "plan_sha256", "plan_digest", "launch_input_sha256", "launch_receipt_sha256",
            "launch_receipt_digest", "release_digest", "release_file_count", "release_files_digest",
            "production_launch_identity_count", "all_production_launch_identity_bytes_replayed", "payload_sha256",
            "payload_identity", "probe_sha256", "pure_metadata_only", "torch_imported",
            "checkpoint_member_opened_by_probe", "model_weight_or_source_video_opened_by_probe",
            "gpu_device_fd_observed_at_probe_end", "formal_report_generated", "html_generated",
        }
        payload_identity = body.get("payload_identity")
        if (
            set(body) != expected_keys
            or body.get("schema_version") != "full644-exploratory-matched-r5g-full16-static-nomodel-probe-v1"
            or body.get("status") != "PASS" or body.get("campaign_mode") != "full16-production"
            or body.get("selected_task_ids") != SELECTED or body.get("unselected_task_count") != 0
            or body.get("formal_full16_report") is not True
            or body.get("canary_stops_after_pair_for_manual_visual_review") is not False
            or body.get("root") != ROOT or body.get("holder_job_id") != JOB_ID or body.get("expected_node") != NODE
            or body.get("slurm_step_id") != expected_step or body.get("slurm_environment_source_names") != slurm_names
            or body.get("slurm_fields_observed_absent") != ["SLURM_JOB_GPUS", "SLURM_JOB_NUM_NODES"]
            or body.get("plan_sha256") != PLAN_SHA256 or body.get("plan_digest") != PLAN_DIGEST
            or body.get("launch_input_sha256") != LAUNCH_INPUT_SHA256
            or body.get("launch_receipt_sha256") != LAUNCH_RECEIPT_SHA256
            or body.get("launch_receipt_digest") != LAUNCH_RECEIPT_DIGEST
            or body.get("release_digest") != RELEASE_DIGEST or body.get("release_file_count") != 17
            or not re.fullmatch(r"[0-9a-f]{64}", body.get("release_files_digest", ""))
            or body.get("production_launch_identity_count") != 16
            or body.get("all_production_launch_identity_bytes_replayed") is not True
            or body.get("payload_sha256") != PRODUCTION_PAYLOAD_SHA256
            or type(payload_identity) is not dict
            or set(payload_identity) != {"device", "inode", "uid", "gid", "mode", "nlink", "rdev", "size", "blocks", "mtime_ns", "ctime_ns"}
            or payload_identity.get("uid") != 2012 or payload_identity.get("gid") != 2000
            or payload_identity.get("mode") != 0o100444 or payload_identity.get("nlink") != 1
            or payload_identity.get("size") != PRODUCTION_PAYLOAD_SIZE
            or body.get("probe_sha256") != "e3a603ad1d94f4d53c7a26fcb29f9313592c571b4483431bdd04dfae5ba68d12"
            or body.get("pure_metadata_only") is not True or body.get("torch_imported") is not False
            or body.get("checkpoint_member_opened_by_probe") is not False
            or body.get("model_weight_or_source_video_opened_by_probe") is not False
            or body.get("gpu_device_fd_observed_at_probe_end") is not False
            or body.get("formal_report_generated") is not False or body.get("html_generated") is not False
        ):
            raise RuntimeError("static receipt semantic contract differs")
    else:
        expected = {
            "schema_version": "full644-exploratory-matched-r5g-full16-root-bootstrap-cpu-probe-v1", "status": "PASS",
            "campaign_mode": "full16-production", "selected_task_ids": SELECTED, "unselected_task_count": 0,
            "plan_path": PLAN_PATH, "plan_sha256": PLAN_SHA256, "slurm_step_id": expected_step,
            "slurm_environment_source_names": slurm_names,
            "slurm_fields_observed_absent": ["SLURM_JOB_GPUS", "SLURM_JOB_NUM_NODES"],
            "captured_source_entry": True,
            "isolated_python": True, "torch_imported": False,
            "gpu_or_model_payload_fd_targets_observed_at_probe_end": [],
            "probe_runner_opened_model_or_checkpoint_payload": False,
            "probe_runner_imported_or_executed_torch": False,
            "gpu_device_fd_observed_at_probe_end": False,
            "formal_report_generated": False, "html_generated": False,
            "formal_full16_report": True, "canary_stops_after_pair_for_manual_visual_review": False,
        }
        if set(body) != set(expected) | {"entry_authority_digest", "release_digest"}:
            raise RuntimeError("bootstrap receipt field closure differs")
        dynamic = {key: body.pop(key) for key in ("entry_authority_digest", "release_digest")}
        if (
            body != expected
            or dynamic.get("release_digest") != BOOTSTRAP_RELEASE_DIGEST
            or type(dynamic.get("entry_authority_digest")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", dynamic["entry_authority_digest"]) is None
        ):
            raise RuntimeError("bootstrap receipt semantic contract differs")
    return claimed


if len(sys.argv) != 3:
    raise RuntimeError("controller argv differs")
root_fd_raw, variant = sys.argv[1:]
if not root_fd_raw.isascii() or not root_fd_raw.isdecimal() or str(int(root_fd_raw)) != root_fd_raw:
    raise RuntimeError("root Python FD syntax differs")
root_fd = int(root_fd_raw)
if variant not in CONFIGS:
    raise RuntimeError("controller variant differs")
if set(os.environ) not in (set(), {"LC_CTYPE"}) or ("LC_CTYPE" in os.environ and os.environ["LC_CTYPE"] != "C.UTF-8"):
    raise RuntimeError("controller bootstrap environment differs")
os.environ.clear()
validate_sacct_raw_fixture()
root_python_before = verify_self(root_fd)
config = CONFIGS[variant]

parent_fd = os.open(config["parent"], os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
parent_before = os.fstat(parent_fd)
parent_named = os.lstat(config["parent"])
if (
    not stat.S_ISDIR(parent_before.st_mode) or stat.S_IMODE(parent_before.st_mode) != 0o755
    or parent_before.st_uid != 2012 or parent_before.st_gid != 2000
    or object_identity(parent_before) != object_identity(parent_named)
):
    raise RuntimeError("held result parent differs")
for leaf in (config["result"], config["stdout"], config["stderr"], config["evidence"]):
    fresh_at(parent_fd, leaf)

payload_fd, payload_before, payload_raw = open_pinned(config["payload"], config["payload_sha256"], config["payload_size"], 2012, 2000, 0o444)
srun_fd, srun_before, _ = open_pinned(SRUN, SRUN_SHA256, SRUN_SIZE, 0, 0, 0o755)
sacct_fd, sacct_before, _ = open_pinned(SACCT, SACCT_SHA256, SACCT_SIZE, 0, 0, 0o755)

if not hasattr(os, "memfd_create") or any(not hasattr(os, key) for key in ("MFD_CLOEXEC", "MFD_ALLOW_SEALING")) or any(not hasattr(fcntl, key) for key in ("F_ADD_SEALS", "F_GET_SEALS", "F_SEAL_SEAL", "F_SEAL_SHRINK", "F_SEAL_GROW", "F_SEAL_WRITE")):
    raise RuntimeError("Linux sealed memfd API unavailable")
memfd = os.memfd_create("full644-r5g-full16-" + variant + "-payload", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
written = 0
while written < len(payload_raw):
    count = os.write(memfd, payload_raw[written:])
    if count <= 0:
        raise RuntimeError("memfd write differs")
    written += count
os.fchmod(memfd, 0o400)
os.fsync(memfd)
seal_mask = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
fcntl.fcntl(memfd, fcntl.F_ADD_SEALS, seal_mask)
memfd_info = os.fstat(memfd)
if (
    not stat.S_ISREG(memfd_info.st_mode) or stat.S_IMODE(memfd_info.st_mode) != 0o400
    or memfd_info.st_nlink != 0 or memfd_info.st_size != len(payload_raw)
    or fcntl.fcntl(memfd, fcntl.F_GET_SEALS) != seal_mask or pread_exact(memfd, len(payload_raw)) != payload_raw
):
    raise RuntimeError("sealed payload memfd differs")
os.lseek(memfd, 0, os.SEEK_SET)
if os.lseek(memfd, 0, os.SEEK_CUR) != 0:
    raise RuntimeError("sealed payload initial offset differs")

# The mode-000 create-only leaf is the irreversible single-attempt marker.
attempt_fd = os.open(config["evidence"], os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o000, dir_fd=parent_fd)
attempt_before = os.fstat(attempt_fd)
if (
    not stat.S_ISREG(attempt_before.st_mode) or stat.S_IMODE(attempt_before.st_mode) != 0
    or attempt_before.st_uid != 2012 or attempt_before.st_gid != 2000
    or attempt_before.st_nlink != 1 or attempt_before.st_size != 0
):
    raise RuntimeError("single-attempt tombstone differs")

remote_wrapper = r'''set -euo pipefail
[[ "$-" == *p* ]] || exit 81
[[ -z "${BASH_ENV-}" && -z "${ENV-}" ]] || exit 82
[[ "${SLURM_JOB_ID-}" == "143812" ]] || exit 83
[[ "${SLURM_STEP_ID-}" =~ ^[1-9][0-9]*$ ]] || exit 84
printf 'CONTROLLER_STEP={"job_id":"143812","step_id":"%s"}\n' "$SLURM_STEP_ID"
exec /bin/bash -p -s
'''
srun_argv = [
    SRUN, "--jobid=" + JOB_ID, "--exclusive", "--exact", "--kill-on-bad-exit=1", "--nodes=1", "--ntasks=1",
    "--nodelist=" + NODE, "--cpus-per-task=4", "--mem=8G", "--gpus-per-node=8", "--export=NONE",
    "--time=00:05:00", "--job-name=" + config["step_name"], "/bin/bash", "-p", "-c", remote_wrapper, "/bin/bash",
]
exact_subprocess_environment = {"LANG": "C", "LC_ALL": "C"}
completed = subprocess.run(
    srun_argv, executable="/proc/self/fd/" + str(srun_fd), stdin=memfd, stdout=subprocess.PIPE,
    stderr=subprocess.PIPE, close_fds=True, pass_fds=(srun_fd,), cwd="/", env=exact_subprocess_environment, check=False,
)
print("srun_rc=" + str(completed.returncode), flush=True)
write_create_only_at(parent_fd, config["stdout"], completed.stdout)
write_create_only_at(parent_fd, config["stderr"], completed.stderr, )
if completed.returncode != 0:
    raise RuntimeError("single srun attempt failed: " + str(completed.returncode))
if completed.stderr != b"":
    raise RuntimeError("srun/probe stderr is not empty")
memfd_after = os.fstat(memfd)
if (
    fcntl.fcntl(memfd, fcntl.F_GET_SEALS) != seal_mask
    or not stat.S_ISREG(memfd_after.st_mode) or stat.S_IMODE(memfd_after.st_mode) != 0o400
    or memfd_after.st_uid != 2012 or memfd_after.st_gid != 2000
    or memfd_after.st_nlink != 0 or memfd_after.st_size != len(payload_raw)
    or pread_exact(memfd, len(payload_raw)) != payload_raw
    or os.lseek(memfd, 0, os.SEEK_CUR) != len(payload_raw)
):
    raise RuntimeError("sealed payload post-consumption replay differs")

lines = completed.stdout.splitlines(keepends=True)
if len(lines) != 2:
    raise RuntimeError("probe stdout line closure differs")
step_match = re.fullmatch(rb'CONTROLLER_STEP=\{"job_id":"143812","step_id":"([1-9][0-9]*)"\}\n', lines[0])
if step_match is None:
    raise RuntimeError("controller step handoff differs")
step_id = step_match.group(1).decode("ascii")
full_step = JOB_ID + "." + step_id
receipt_raw, receipt_identity = stable_read_at(parent_fd, config["result"], 0o400)
receipt = strict_json(receipt_raw)
receipt_digest = validate_receipt(variant, receipt, step_id)
if lines[1] != (config["sentinel"] + " " + receipt_digest + "\n").encode("ascii"):
    raise RuntimeError("probe sentinel differs")

sacct_argv = [
    SACCT, "--jobs=" + full_step, "--noheader", "--parsable2", "--noconvert",
    "--format=JobIDRaw,JobName%64,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES%256,NodeList%64,MaxRSS",
]
sacct_attempts = 0
sacct_stdout = b""
sacct_stderr = b""
sacct_columns = None
for sacct_attempts in range(1, 31):
    observed = subprocess.run(
        sacct_argv, executable="/proc/self/fd/" + str(sacct_fd), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True, pass_fds=(sacct_fd,),
        cwd="/", env=exact_subprocess_environment, check=False,
    )
    if observed.returncode != 0 or observed.stderr != b"":
        raise RuntimeError("held sacct query failed")
    sacct_stdout = observed.stdout
    sacct_stderr = observed.stderr
    exact_rows = []
    for line in sacct_stdout.decode("utf-8", "strict").splitlines():
        columns = parse_sacct_row(line, full_step)
        if columns is not None:
            exact_rows.append(columns)
    if len(exact_rows) > 1:
        raise RuntimeError("duplicate exact sacct row")
    if len(exact_rows) == 1:
        candidate = exact_rows[0]
        if candidate[2] not in {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "RESIZING"}:
            sacct_columns = candidate
            break
    time.sleep(1)
if sacct_columns is None:
    raise RuntimeError("terminal sacct row unavailable")
row_step, row_name, row_state, row_exit, row_elapsed, row_cpus, row_tres, row_node, row_maxrss = sacct_columns
tres = parse_tres(row_tres)
if (
    row_step != full_step or row_name != config["step_name"] or row_state != "COMPLETED" or row_exit != "0:0"
    or not row_elapsed.isascii() or not row_elapsed.isdecimal() or row_cpus != "4" or row_node != NODE
):
    raise RuntimeError("terminal sacct semantics differ")

replay_pinned(payload_fd, config["payload"], payload_before, config["payload_sha256"])
replay_pinned(srun_fd, SRUN, srun_before, SRUN_SHA256)
replay_pinned(sacct_fd, SACCT, sacct_before, SACCT_SHA256)
root_python_after = os.fstat(root_fd)
root_python_raw_after = pread_exact(root_fd, root_python_after.st_size)
if (
    identity(root_python_after) != identity(root_python_before)
    or identity(root_python_after) != identity(os.lstat(ROOT_PYTHON))
    or identity(root_python_after) != identity(os.stat("/proc/self/exe"))
    or hashlib.sha256(root_python_raw_after).hexdigest() != ROOT_PYTHON_SHA256
    or os.lseek(root_fd, 0, os.SEEK_CUR) != 0
):
    raise RuntimeError("root Python post replay differs")
parent_after = os.fstat(parent_fd)
if object_identity(parent_after) != object_identity(parent_before) or object_identity(parent_after) != object_identity(os.lstat(config["parent"])):
    raise RuntimeError("held result parent post replay differs")

stdout_written, stdout_identity = stable_read_at(parent_fd, config["stdout"], 0o400)
stderr_written, stderr_identity = stable_read_at(parent_fd, config["stderr"], 0o400, allow_empty=True)
if stdout_written != completed.stdout or stderr_written != completed.stderr:
    raise RuntimeError("controller log publication replay differs")

evidence = {
    "schema_version": "full644-r5g-full16-" + variant + "-heldfd-controller-evidence-v1", "status": "PASS",
    "campaign_mode": "full16-production", "holder_job_id": JOB_ID, "node": NODE, "numeric_step": full_step,
    "single_srun_attempt": True, "srun_returncode": completed.returncode, "srun_argv": srun_argv,
    "srun_environment": exact_subprocess_environment,
    "srun_executable": {"path": SRUN, "sha256": SRUN_SHA256, "executed_via_retained_fd": True},
    "sacct_executable": {"path": SACCT, "sha256": SACCT_SHA256, "executed_via_retained_fd": True},
    "root_python": {"path": ROOT_PYTHON, "sha256": ROOT_PYTHON_SHA256, "executed_via_retained_fd": True},
    "payload": {
        "path": config["payload"], "sha256": config["payload_sha256"], "size": len(payload_raw),
        "executed_from_sealed_memfd_stdin": True, "seal_mask": seal_mask, "final_offset": os.lseek(memfd, 0, os.SEEK_CUR),
    },
    "step_handoff": {"job_id": JOB_ID, "step_id": step_id, "canonical_stdout_line": lines[0].decode("ascii").rstrip("\n")},
    "requested_resources": {"cpus_per_task": 4, "memory": "8G", "gpus_per_node": 8, "exclusive": True, "exact": True, "overlap": False},
    "sacct": {
        "query_argv": sacct_argv, "query_attempts": sacct_attempts,
        "raw_stdout_sha256": hashlib.sha256(sacct_stdout).hexdigest(), "raw_stderr_sha256": hashlib.sha256(sacct_stderr).hexdigest(),
        "row": {"JobIDRaw": row_step, "JobName": row_name, "State": row_state, "ExitCode": row_exit,
                "ElapsedRaw": row_elapsed, "AllocCPUS": row_cpus, "AllocTRES": tres, "NodeList": row_node, "MaxRSS": row_maxrss},
    },
    "receipt_replay": {
        "path": config["parent"] + "/" + config["result"], "file_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "receipt_digest": receipt_digest, "identity": receipt_identity, "canonical_json_plus_lf": True,
        "semantic_contract_replayed": True,
    },
    "stdout": {"path": config["parent"] + "/" + config["stdout"], "sha256": hashlib.sha256(completed.stdout).hexdigest(), "identity": stdout_identity},
    "stderr": {"path": config["parent"] + "/" + config["stderr"], "sha256": hashlib.sha256(completed.stderr).hexdigest(), "identity": stderr_identity, "empty": True},
    "formal_report_generated": False, "html_generated": False,
    "external_trust_boundary": "root-owned default Slurm config, dynamic loader, shared libraries, plugins, kernel, and /bin/bash",
}
evidence["evidence_digest"] = digest(evidence)
evidence_raw = canonical(evidence) + b"\n"
attempt_current = os.fstat(attempt_fd)
attempt_named = os.stat(config["evidence"], dir_fd=parent_fd, follow_symlinks=False)
if (
    attempt_current.st_dev != attempt_before.st_dev or attempt_current.st_ino != attempt_before.st_ino
    or identity(attempt_current) != identity(attempt_named) or stat.S_IMODE(attempt_current.st_mode) != 0 or attempt_current.st_size != 0
):
    raise RuntimeError("single-attempt tombstone replay differs")
offset = 0
while offset < len(evidence_raw):
    count = os.write(attempt_fd, evidence_raw[offset:])
    if count <= 0:
        raise RuntimeError("evidence write differs")
    offset += count
os.fsync(attempt_fd)
staged = os.fstat(attempt_fd)
staged_named = os.stat(config["evidence"], dir_fd=parent_fd, follow_symlinks=False)
if pread_exact(attempt_fd, len(evidence_raw)) != evidence_raw or staged.st_size != len(evidence_raw) or identity(staged) != identity(staged_named):
    raise RuntimeError("evidence staged replay differs")
# Acceptance commit: no fallible provenance checks follow this chmod.
os.fchmod(attempt_fd, 0o400)
pass_line = (
    ("R5G_FULL16_STATIC_NOMODEL" if variant == "static" else "R5G_FULL16_ROOT_BOOTSTRAP")
    + "_HELDFD_GATE_PASS step=" + full_step + " receipt_digest=" + receipt_digest
    + " evidence_digest=" + evidence["evidence_digest"] + "\n"
).encode("ascii")
try:
    os.write(1, pass_line)
except OSError:
    pass
os._exit(0)
PY
