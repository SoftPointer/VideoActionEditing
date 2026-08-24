#!/bin/bash -p
# One-shot exact8 Shared8 full16 production launch.
#
# This deliberately reuses the already GPU-proven r5f execution release while
# requiring a fresh package root and a fresh node-local rank cache.  Stream it
# to the AUH login node through the exact minimal environment documented below;
# named execution and a second invocation are both refused by freshness gates.

set -euo pipefail
umask 077

# Frozen after all three gate receipts and all three held-FD controller
# evidence files passed and were replayed byte-for-byte.
readonly R5G_GPU_CONTROLLER_STATE=READY
[[ "$R5G_GPU_CONTROLLER_STATE" == READY ]] || {
  /usr/bin/printf 'r5g full16 GPU v2 is HOLD pending exact CPU gate pins\n' >&2
  exit 88
}

[[ "$0" == /bin/bash && "$#" -eq 0 && "$-" == *p* && "$-" != *i* ]] || {
  /usr/bin/printf 'r5g full16 launch refused: shell entry differs\n' >&2
  exit 96
}
[[ "${PATH:-}" == /usr/bin:/bin && "${LC_ALL:-}" == C \
  && "${LANG:-}" == C && "${HOME:-}" == /vast/users/guangyi.chen \
  && "${BASH_ENV:-}" == /dev/null ]] || {
  /usr/bin/printf 'r5g full16 launch refused: entry environment differs\n' >&2
  exit 96
}
[[ -z "${ENV:-}" && -z "${LD_PRELOAD:-}" && -z "${LD_LIBRARY_PATH:-}" \
  && -z "${PYTHONPATH:-}" && -z "${PYTHONHOME:-}" \
  && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${ROCR_VISIBLE_DEVICES:-}" \
  && -z "${HIP_VISIBLE_DEVICES:-}" && -z "${GPU_DEVICE_ORDINAL:-}" ]] || {
  /usr/bin/printf 'r5g full16 launch refused: loader or GPU environment differs\n' >&2
  exit 96
}
if builtin declare -F | /usr/bin/grep . >/dev/null; then
  /usr/bin/printf 'r5g full16 launch refused: preloaded Bash function exists\n' >&2
  exit 96
fi
if shopt -q varredir_close 2>/dev/null; then
  shopt -u varredir_close
fi

readonly R5G_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_matched_eval_auh_r5f_job143812_node293_full16_847b91a2_c91de7eb_d70eac5c_r1
readonly R5G_PAYLOAD="$R5G_ROOT/launch/root_launch_payload_auh_r5d.sh"
readonly R5G_STATIC="$R5G_ROOT/diagnostics/full644_exploratory_matched_r5g_static_nomodel_probe_v1.py"
readonly R5G_CACHE=/tmp/bernini-full644-r5g-job143812-node293-full16-r1-rank-cache
readonly ROOT_PYTHON=/usr/bin/python3.10

[[ -f "$R5G_PAYLOAD" && ! -L "$R5G_PAYLOAD" \
  && -f "$R5G_STATIC" && ! -L "$R5G_STATIC" \
  && -f "$ROOT_PYTHON" && ! -L "$ROOT_PYTHON" ]] || {
  /usr/bin/printf 'r5g full16 launch refused: held entry path differs\n' >&2
  exit 96
}
[[ ! -e "$R5G_CACHE" && ! -L "$R5G_CACHE" ]] || {
  /usr/bin/printf 'r5g full16 launch refused: login-visible rank cache is not fresh\n' >&2
  exit 96
}

exec {ROOT_PYTHON_FD}<"$ROOT_PYTHON"
exec {R5G_PAYLOAD_FD}<"$R5G_PAYLOAD"
[[ "$ROOT_PYTHON_FD" =~ ^[0-9]+$ && "$R5G_PAYLOAD_FD" =~ ^[0-9]+$ \
  && "$ROOT_PYTHON_FD" -ge 3 && "$R5G_PAYLOAD_FD" -ge 3 \
  && "$ROOT_PYTHON_FD" != "$R5G_PAYLOAD_FD" ]] || exit 97

"/proc/self/fd/$ROOT_PYTHON_FD" -I -S -B - \
  "$ROOT_PYTHON_FD" "$R5G_PAYLOAD_FD" "$R5G_ROOT" "$R5G_STATIC" "$R5G_CACHE" <<'PY'
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import stat
import sys
import types

ROOT_PYTHON = "/usr/bin/python3.10"
ROOT_PYTHON_SHA256 = "11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96"
ROOT_PYTHON_SIZE = 5_937_800
STATIC_SHA256 = "e3a603ad1d94f4d53c7a26fcb29f9313592c571b4483431bdd04dfae5ba68d12"
JOB_ID = "143812"
NODE = "auh7-1b-gpu-293"
CAMPAIGN = "full16-production"
TASKS = [
    f"shared8-{index:02d}-{arm}"
    for index in range(8)
    for arm in ("base", "full644")
]
# Exact post-gate pins; any byte or identity change stops before the GPU marker.
GATE_PINS = {
    "static_receipt": {
        "relative": "evidence/static_nomodel_probe_receipt_r5d.json",
        "sha256": "4403602ae1e3e7c50f5c22fdc2c397b5a16e78358ce7c84f88f4f759fa886c43",
        "receipt_digest": "5b1e08d7814b26afd98a475067e938fe56145e616d21103893e558b7f08c015e",
    },
    "static_evidence": {
        "relative": "evidence/r5g_full16_static_nomodel_probe.sacct-and-replay.json",
        "sha256": "1ba054cdd23c638d09ceca1923be84bcfdcf54ee09f1c7fcc91550087cd5d3b1",
        "evidence_digest": "63938e0edb3fa273964ac7ba5b8b73b0fc97a60ee38e39c5bc4ac245fd8e1b05",
    },
    "bootstrap_receipt": {
        "relative": "evidence/root_bootstrap_cpu_probe_receipt_r5d.json",
        "sha256": "69b4a8bb4e7d66224560f1df911745770519f2f7209d3c7f4a4e075e9f32e3b0",
        "receipt_digest": "74507c60534016a0d998b48c6ab85650b338ab3abfb1763cf478154a20a91f58",
    },
    "bootstrap_evidence": {
        "relative": "evidence/r5g_full16_root_bootstrap_cpu_probe.sacct-and-replay.json",
        "sha256": "4b3dad51a6d73221a6ad9952543d7b04c418625eb83a509d5c119b751b5d4869",
        "evidence_digest": "5fb90052c58ac1a6c0b3ada6ea46895cd1ec900d5d35ca5121fa862fe1698f51",
    },
    "consumption_receipt": {
        "relative": "diagnostics/cpu_consumption_probe_work_r1/r5d-cpu-consumption-probe.json",
        "sha256": "9829c803e25c4ac0d85db3c915789c4d20de2855b957caa9c2484f9bab441936",
        "receipt_digest": "ca5be8045b753ede8d2481e07792f9e67ab09c5f202476cd0e3d4aa17cedd824",
    },
    "consumption_evidence": {
        "relative": "diagnostics/r5g_full16_cpu_consumption_probe.sacct-and-replay.json",
        "sha256": "e061be50f6f92c0d5f4795fc4ee5c5ded3f1438c6c3bd93525e57d199165d76c",
        "evidence_digest": "2570f96d70171843c67d4d3d5cab34a7d4195e430419e903e34abef61bc2272e",
    },
}
GPU_ATTEMPT_RELATIVE = "evidence/r5g_full16_gpu_attempt_v2.json"


def ident(value):
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "uid": value.st_uid,
        "gid": value.st_gid,
        "mode": value.st_mode,
        "nlink": value.st_nlink,
        "rdev": value.st_rdev,
        "size": value.st_size,
        "blocks": getattr(value, "st_blocks", 0),
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def pread(descriptor, size):
    blocks = []
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1_048_576, size - offset), offset)
        if not block:
            break
        blocks.append(block)
        offset += len(block)
    raw = b"".join(blocks)
    if len(raw) != size:
        raise RuntimeError("captured file short read")
    return raw


def held(descriptor, path, digest, size, mode, uid, gid, *, process=False):
    before = os.fstat(descriptor)
    raw = pread(descriptor, before.st_size)
    after = os.fstat(descriptor)
    named = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != mode
        or before.st_uid != uid
        or before.st_gid != gid
        or before.st_nlink != 1
        or before.st_size != size
        or ident(before) != ident(after)
        or ident(before) != ident(named)
        or hashlib.sha256(raw).hexdigest() != digest
        or os.lseek(descriptor, 0, os.SEEK_CUR) != 0
        or (process and ident(before) != ident(os.stat("/proc/self/exe")))
    ):
        raise RuntimeError("held authority differs: " + path)
    return raw, ident(before)


def stable_blob(path, expected_sha256, *, expected_mode, allow_empty=False):
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.is_symlink()
    ):
        raise RuntimeError("gate artifact path differs: " + str(path))
    descriptor = os.open(
        path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        before = os.fstat(descriptor)
        raw = pread(descriptor, before.st_size)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != expected_mode
        or before.st_uid != 2012
        or before.st_gid != 2000
        or before.st_nlink != 1
        or (not allow_empty and not raw)
        or ident(before) != ident(after)
        or ident(before) != ident(named)
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise RuntimeError("gate artifact authority differs: " + str(path))
    return raw, ident(before)


def stable_gate_json(root, key, digest_field):
    pin = GATE_PINS[key]
    path = root / pin["relative"]
    raw, identity = stable_blob(
        path, pin["sha256"], expected_mode=0o400, allow_empty=False
    )
    value = module.strict_json(raw, label=key)
    unsigned = dict(value)
    claimed = unsigned.pop(digest_field, None)
    if (
        claimed != pin[digest_field]
        or claimed != module.object_sha256(unsigned)
    ):
        raise RuntimeError("gate object digest differs: " + key)
    return path, raw, identity, value


def exact_directory(path, expected_names, *, expected_mode):
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) != expected_mode
        or info.st_uid != 2012
        or info.st_gid != 2000
        or path.is_symlink()
    ):
        raise RuntimeError("gate directory identity differs: " + str(path))
    with os.scandir(path) as entries:
        names = {entry.name for entry in entries}
    if names != set(expected_names):
        raise RuntimeError("gate directory closure differs: " + str(path))


def validate_log_replay(root, evidence):
    for label, allow_empty in (("stdout", False), ("stderr", True)):
        row = evidence.get(label)
        if (
            not isinstance(row, dict)
            or set(row) != (
                {"path", "sha256", "identity"}
                if label == "stdout"
                else {"path", "sha256", "identity", "empty"}
            )
            or (label == "stderr" and row.get("empty") is not True)
        ):
            raise RuntimeError("gate log row differs")
        path = Path(row["path"])
        if root not in path.parents:
            raise RuntimeError("gate log escaped package root")
        raw, identity = stable_blob(
            path, row["sha256"], expected_mode=0o400, allow_empty=allow_empty
        )
        if identity != row["identity"] or (label == "stderr" and raw != b""):
            raise RuntimeError("gate log replay differs")


def validate_sidecar(root, key, receipt_key, expected_schema, receipt):
    path, raw, identity, value = stable_gate_json(
        root, key, "evidence_digest"
    )
    numeric_step = value.get("numeric_step")
    if (
        value.get("schema_version") != expected_schema
        or value.get("status") != "PASS"
        or value.get("campaign_mode") != CAMPAIGN
        or value.get("holder_job_id") != JOB_ID
        or value.get("node") != NODE
        or value.get("single_srun_attempt") is not True
        or value.get("srun_returncode") != 0
        or type(numeric_step) is not str
        or not numeric_step.startswith(JOB_ID + ".")
        or not numeric_step[len(JOB_ID) + 1 :].isascii()
        or not numeric_step[len(JOB_ID) + 1 :].isdecimal()
        or int(numeric_step[len(JOB_ID) + 1 :]) <= 178
        or value.get("sacct", {}).get("row", {}).get("JobIDRaw") != numeric_step
        or value.get("sacct", {}).get("row", {}).get("State") != "COMPLETED"
        or value.get("sacct", {}).get("row", {}).get("ExitCode") != "0:0"
    ):
        raise RuntimeError("gate evidence semantics differ: " + key)
    receipt_row = value.get("receipt_replay")
    receipt_pin = GATE_PINS[receipt_key]
    if (
        not isinstance(receipt_row, dict)
        or receipt_row.get("path") != str(root / receipt_pin["relative"])
        or receipt_row.get("file_sha256") != receipt_pin["sha256"]
        or receipt_row.get("receipt_digest") != receipt_pin["receipt_digest"]
        or receipt_row.get("canonical_json_plus_lf") is not True
        or receipt_row.get("semantic_contract_replayed") is not True
        or receipt_row.get("identity") != receipt["_identity"]
    ):
        raise RuntimeError("gate receipt/evidence binding differs: " + key)
    validate_log_replay(root, value)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "identity": identity,
        "numeric_step": numeric_step,
        "evidence_digest": value["evidence_digest"],
    }


def create_attempt_marker(path, value):
    value = dict(value)
    value["attempt_digest"] = module.object_sha256(value)
    raw = module.canonical_json_bytes(value) + b"\n"
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY
        | os.O_CLOEXEC
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        parent = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(parent.st_mode)
            or stat.S_IMODE(parent.st_mode) != 0o755
            or parent.st_uid != 2012
            or parent.st_gid != 2000
        ):
            raise RuntimeError("GPU attempt parent differs")
        descriptor = os.open(
            path.name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0,
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0
            or before.st_uid != 2012
            or before.st_gid != 2000
            or before.st_nlink != 1
            or before.st_size != 0
        ):
            raise RuntimeError("GPU single-attempt tombstone differs")
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise RuntimeError("GPU attempt marker write differs")
            offset += count
        os.fsync(descriptor)
        staged = os.fstat(descriptor)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            pread(descriptor, len(raw)) != raw
            or staged.st_size != len(raw)
            or ident(staged) != ident(named)
            or staged.st_dev != before.st_dev
            or staged.st_ino != before.st_ino
        ):
            raise RuntimeError("GPU attempt marker staging replay differs")
        # Irreversible acceptance commit: a failed srun remains a burned attempt.
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)
    return value["attempt_digest"]


if len(sys.argv) != 6:
    raise RuntimeError("preflight argv differs")
python_fd, payload_fd = (int(sys.argv[1]), int(sys.argv[2]))
root = Path(sys.argv[3])
static_path = Path(sys.argv[4])
cache = Path(sys.argv[5])
if (
    root != Path(
        "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
        "VideoEdit_experiments/bernini_full644_exploratory_matched_eval_auh_"
        "r5f_job143812_node293_full16_847b91a2_c91de7eb_d70eac5c_r1"
    )
    or cache != Path(
        "/tmp/bernini-full644-r5g-job143812-node293-full16-r1-rank-cache"
    )
    or os.path.lexists(cache)
):
    raise RuntimeError("fresh root/cache binding differs")
held(
    python_fd,
    ROOT_PYTHON,
    ROOT_PYTHON_SHA256,
    ROOT_PYTHON_SIZE,
    0o755,
    0,
    0,
    process=True,
)
static_fd = os.open(
    static_path,
    os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
)
try:
    static_raw, _ = held(
        static_fd, str(static_path), STATIC_SHA256, 45_606, 0o444, 2012, 2000
    )
finally:
    os.close(static_fd)
module = types.ModuleType("_r5g_full16_static_preflight")
module.__file__ = str(static_path)
module.__package__ = None
module.__loader__ = None
module.__cached__ = None
module.__spec__ = importlib.machinery.ModuleSpec(
    module.__name__, loader=None, origin=str(static_path)
)
sys.modules[module.__name__] = module
exec(
    compile(static_raw.decode("utf-8", "strict"), str(static_path), "exec", dont_inherit=True),
    module.__dict__,
)
if module.CAMPAIGN != CAMPAIGN or list(module.SELECTED) != TASKS:
    raise RuntimeError("full16 static source semantics differ")

plan_path = root / "plan/full644_exploratory_matched_plan_auh_r5d.json"
input_path = root / "launch/root_launch_input_auh_r5d.json"
receipt_path = root / "launch/root_launch_receipt_auh_r5d.json"
payload_path = root / "launch/root_launch_payload_auh_r5d.sh"
module.validate_release_tree(root / "release")
for relative, label in (
    ("outputs/media", "media output"),
    ("final", "final artifact"),
    ("runtime", "runtime authority"),
):
    module.require_empty_directory(root / relative, label=label)
exact_directory(
    root / "evidence",
    {
        "static_nomodel_probe_receipt_r5d.json",
        "r5g_full16_static_nomodel_probe.stdout.log",
        "r5g_full16_static_nomodel_probe.stderr.log",
        "r5g_full16_static_nomodel_probe.sacct-and-replay.json",
        "root_bootstrap_cpu_probe_receipt_r5d.json",
        "r5g_full16_root_bootstrap_cpu_probe.stdout.log",
        "r5g_full16_root_bootstrap_cpu_probe.stderr.log",
        "r5g_full16_root_bootstrap_cpu_probe.sacct-and-replay.json",
    },
    expected_mode=0o755,
)
exact_directory(
    root / "diagnostics",
    {
        "full644_exploratory_matched_r5g_root_bootstrap_probe_runner_v1.py",
        "full644_exploratory_matched_r5g_static_nomodel_probe_v1.py",
        "full644_exploratory_matched_r5f_cpu_consumption_probe_v1.py",
        "root_bootstrap_probe_input_r5d.json",
        "root_bootstrap_probe_payload_r5d.sh",
        "root_bootstrap_probe_materialization_receipt_r5d.json",
        "static_nomodel_probe_payload_r5d.sh",
        "cpu_consumption_probe_work_r1",
        "r5g_full16_cpu_consumption_probe.stdout.log",
        "r5g_full16_cpu_consumption_probe.stderr.log",
        "r5g_full16_cpu_consumption_probe.sacct-and-replay.json",
    },
    expected_mode=0o755,
)
exact_directory(
    root / "diagnostics/cpu_consumption_probe_work_r1",
    {"r5d-cpu-consumption-probe.json"},
    expected_mode=0o700,
)
plan_raw, plan_identity = module.stable_file(plan_path, expected_mode=0o444)
plan_sha = hashlib.sha256(plan_raw).hexdigest()
plan = module.strict_json(plan_raw, label="plan")
module.validate_plan(plan, root)
input_raw, input_identity = module.stable_file(input_path, expected_mode=0o444)
input_sha = hashlib.sha256(input_raw).hexdigest()
launch_input = module.strict_json(input_raw, label="launch input")
saved_job = os.environ.get("SLURM_JOB_ID")
saved_nodes = os.environ.get("SLURM_JOB_NODELIST")
os.environ["SLURM_JOB_ID"] = JOB_ID
os.environ["SLURM_JOB_NODELIST"] = NODE
try:
    module.validate_input(launch_input, root, plan_path)
finally:
    if saved_job is None:
        os.environ.pop("SLURM_JOB_ID", None)
    else:
        os.environ["SLURM_JOB_ID"] = saved_job
    if saved_nodes is None:
        os.environ.pop("SLURM_JOB_NODELIST", None)
    else:
        os.environ["SLURM_JOB_NODELIST"] = saved_nodes
receipt_raw, _ = module.stable_file(receipt_path, expected_mode=0o400)
receipt = module.strict_json(receipt_raw, label="launch receipt")
module.validate_launch_receipt(receipt, launch_input, input_identity, plan_sha)
module.validate_launch_publication_binding(
    receipt,
    input_path=input_path,
    input_sha256=input_sha,
    input_identity=input_identity,
    receipt_path=receipt_path,
)
release = receipt["release"]
if (
    release.get("campaign_mode") != CAMPAIGN
    or release.get("selected_task_ids") != TASKS
    or release.get("formal_full16_report") is not True
    or release.get("canary_stops_after_pair_for_manual_visual_review") is not False
):
    raise RuntimeError("full16 production release differs")
for row in release["identities"].values():
    module.replay_identity_row(row)
payload_raw, payload_identity = held(
    payload_fd,
    str(payload_path),
    receipt["payload_sha256"],
    receipt["payload_size"],
    0o444,
    2012,
    2000,
)
if (
    receipt.get("payload_path") != str(payload_path)
    or receipt.get("payload_mode") != 0o444
    or release["identities"]["plan"]["identity"] != plan_identity
    or hashlib.sha256(payload_raw).hexdigest() != receipt["payload_sha256"]
    or ident(os.fstat(payload_fd)) != payload_identity
):
    raise RuntimeError("held full16 payload binding differs")
if any(
    type(value) is not str
    or module.SHA256_RE.fullmatch(value) is None
    for pin in GATE_PINS.values()
    for key, value in pin.items()
    if key.endswith("sha256") or key.endswith("digest")
):
    raise RuntimeError("CPU gate pins remain HOLD placeholders")

gate_objects = {}
for key, digest_field in (
    ("static_receipt", "receipt_digest"),
    ("bootstrap_receipt", "receipt_digest"),
    ("consumption_receipt", "receipt_digest"),
):
    gate_path, gate_raw, gate_identity, gate_value = stable_gate_json(
        root, key, digest_field
    )
    gate_objects[key] = {
        "path": gate_path,
        "raw": gate_raw,
        "identity": gate_identity,
        "value": gate_value,
    }

static_receipt = gate_objects["static_receipt"]["value"]
bootstrap_receipt = gate_objects["bootstrap_receipt"]["value"]
consumption_receipt = gate_objects["consumption_receipt"]["value"]
for key in gate_objects:
    gate_objects[key]["value"] = {
        **gate_objects[key]["value"],
        "_identity": gate_objects[key]["identity"],
    }
if (
    static_receipt.get("schema_version")
    != "full644-exploratory-matched-r5g-full16-static-nomodel-probe-v1"
    or static_receipt.get("status") != "PASS"
    or static_receipt.get("campaign_mode") != CAMPAIGN
    or static_receipt.get("selected_task_ids") != TASKS
    or static_receipt.get("unselected_task_count") != 0
    or static_receipt.get("formal_full16_report") is not True
    or static_receipt.get("canary_stops_after_pair_for_manual_visual_review")
    is not False
    or static_receipt.get("root") != str(root)
    or static_receipt.get("holder_job_id") != JOB_ID
    or static_receipt.get("expected_node") != NODE
    or static_receipt.get("plan_sha256") != plan_sha
    or static_receipt.get("plan_digest") != plan["plan_digest"]
    or static_receipt.get("launch_input_sha256") != input_sha
    or static_receipt.get("launch_receipt_digest") != receipt["receipt_digest"]
    or static_receipt.get("release_digest") != receipt["release_digest"]
    or static_receipt.get("payload_sha256") != receipt["payload_sha256"]
    or static_receipt.get("pure_metadata_only") is not True
    or static_receipt.get("torch_imported") is not False
):
    raise RuntimeError("static CPU gate receipt differs")
if (
    bootstrap_receipt.get("schema_version")
    != "full644-exploratory-matched-r5g-full16-root-bootstrap-cpu-probe-v1"
    or bootstrap_receipt.get("status") != "PASS"
    or bootstrap_receipt.get("campaign_mode") != CAMPAIGN
    or bootstrap_receipt.get("selected_task_ids") != TASKS
    or bootstrap_receipt.get("unselected_task_count") != 0
    or bootstrap_receipt.get("plan_path") != str(plan_path)
    or bootstrap_receipt.get("plan_sha256") != plan_sha
    or bootstrap_receipt.get("formal_full16_report") is not True
    or bootstrap_receipt.get("canary_stops_after_pair_for_manual_visual_review")
    is not False
    or bootstrap_receipt.get("captured_source_entry") is not True
    or bootstrap_receipt.get("isolated_python") is not True
    or bootstrap_receipt.get("torch_imported") is not False
):
    raise RuntimeError("bootstrap CPU gate receipt differs")
if (
    consumption_receipt.get("schema_version")
    != "full644-exploratory-matched-r5f-cpu-consumption-probe-v1"
    or consumption_receipt.get("status") != "PASS"
    or consumption_receipt.get("summary")
    != {
        "successful_arms": ["base", "full"],
        "rejected_hostiles": [
            "digest",
            "task",
            "adapter_namespace",
            "adapter_leaf",
        ],
        "success_count": 2,
        "hostile_rejection_count": 4,
    }
):
    raise RuntimeError("consumption CPU gate receipt differs")

static_evidence = validate_sidecar(
    root,
    "static_evidence",
    "static_receipt",
    "full644-r5g-full16-static-heldfd-controller-evidence-v1",
    gate_objects["static_receipt"]["value"],
)
bootstrap_evidence = validate_sidecar(
    root,
    "bootstrap_evidence",
    "bootstrap_receipt",
    "full644-r5g-full16-bootstrap-heldfd-controller-evidence-v1",
    gate_objects["bootstrap_receipt"]["value"],
)
consumption_evidence = validate_sidecar(
    root,
    "consumption_evidence",
    "consumption_receipt",
    "full644-r5g-full16-consumption-heldfd-controller-evidence-v1",
    gate_objects["consumption_receipt"]["value"],
)
if (
    static_receipt.get("slurm_step_id")
    != static_evidence["numeric_step"].split(".", 1)[1]
    or bootstrap_receipt.get("slurm_step_id")
    != bootstrap_evidence["numeric_step"].split(".", 1)[1]
    or len(
        {
            static_evidence["numeric_step"],
            bootstrap_evidence["numeric_step"],
            consumption_evidence["numeric_step"],
        }
    )
    != 3
):
    raise RuntimeError("CPU gate step binding differs")

attempt_digest = create_attempt_marker(
    root / GPU_ATTEMPT_RELATIVE,
    {
        "schema_version": "full644-r5g-full16-gpu-attempt-v2",
        "status": "ATTEMPT_CLAIMED_BEFORE_SRUN",
        "holder_job_id": JOB_ID,
        "node": NODE,
        "campaign_mode": CAMPAIGN,
        "selected_task_ids": TASKS,
        "root": str(root),
        "rank_cache_root": str(cache),
        "plan_sha256": plan_sha,
        "plan_digest": plan["plan_digest"],
        "production_payload_sha256": receipt["payload_sha256"],
        "gate_pins": GATE_PINS,
        "gate_numeric_steps": [
            static_evidence["numeric_step"],
            bootstrap_evidence["numeric_step"],
            consumption_evidence["numeric_step"],
        ],
        "retry_allowed": False,
        "formal_report_generated": False,
        "html_generated": False,
    },
)
print(
    "R5G_FULL16_PREFLIGHT_PASS "
    + receipt["payload_sha256"]
    + " "
    + plan["plan_digest"]
    + " "
    + attempt_digest,
    flush=True,
)
PY

exec {ROOT_PYTHON_FD}<&-
exec /usr/bin/srun \
  --jobid=143812 \
  --job-name=f644-r5g-full16-gpu \
  --exclusive --exact --immediate=10 --kill-on-bad-exit=1 \
  --nodes=1 --ntasks=1 --nodelist=auh7-1b-gpu-293 \
  --cpus-per-task=64 --mem=64G --gpus-per-node=8 \
  --export=NONE --time=03:00:00 \
  /bin/bash -p -c '
[[ "${SLURM_JOB_ID-}" == 143812 && "${SLURM_STEP_ID-}" =~ ^[1-9][0-9]*$ ]] || exit 81
(( 10#$SLURM_STEP_ID > 211 )) || exit 82
[[ "${SLURM_GPUS_ON_NODE-}" == 8 && "${SLURM_GPUS_PER_NODE-}" == 8 \
  && "${SLURM_STEP_GPUS-}" == 0,1,2,3,4,5,6,7 ]] || exit 83
[[ ! -e /tmp/bernini-full644-r5g-job143812-node293-full16-r1-rank-cache \
  && ! -L /tmp/bernini-full644-r5g-job143812-node293-full16-r1-rank-cache ]] || exit 84
exec /bin/bash -p -s
' /bin/bash <&"$R5G_PAYLOAD_FD"
