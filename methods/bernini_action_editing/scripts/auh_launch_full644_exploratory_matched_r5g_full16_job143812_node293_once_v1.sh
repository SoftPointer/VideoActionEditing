#!/bin/bash -p
# One-shot exact8 Shared8 full16 production launch.
#
# This deliberately reuses the already GPU-proven r5f execution release while
# requiring a fresh package root and a fresh node-local rank cache.  Stream it
# to the AUH login node through the exact minimal environment documented below;
# named execution and a second invocation are both refused by freshness gates.

set -euo pipefail
umask 077

# HOLD: superseded by v2, which requires all three CPU gate receipts/evidence
# and commits a create-only GPU attempt marker before the only srun.
readonly R5G_GPU_CONTROLLER_STATE=HOLD_SUPERSEDED_BY_V2
[[ "$R5G_GPU_CONTROLLER_STATE" == READY ]] || {
  /usr/bin/printf 'r5g full16 GPU v1 is HOLD; use frozen v2 only\n' >&2
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
    ("evidence", "evidence"),
    ("outputs/media", "media output"),
    ("final", "final artifact"),
    ("runtime", "runtime authority"),
):
    module.require_empty_directory(root / relative, label=label)
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
print(
    "R5G_FULL16_PREFLIGHT_PASS "
    + receipt["payload_sha256"]
    + " "
    + plan["plan_digest"],
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
(( 10#$SLURM_STEP_ID > 178 )) || exit 82
[[ "${SLURM_GPUS_ON_NODE-}" == 8 && "${SLURM_GPUS_PER_NODE-}" == 8 \
  && "${SLURM_STEP_GPUS-}" == 0,1,2,3,4,5,6,7 ]] || exit 83
[[ ! -e /tmp/bernini-full644-r5g-job143812-node293-full16-r1-rank-cache \
  && ! -L /tmp/bernini-full644-r5g-job143812-node293-full16-r1-rank-cache ]] || exit 84
exec /bin/bash -p -s
' /bin/bash <&"$R5G_PAYLOAD_FD"
