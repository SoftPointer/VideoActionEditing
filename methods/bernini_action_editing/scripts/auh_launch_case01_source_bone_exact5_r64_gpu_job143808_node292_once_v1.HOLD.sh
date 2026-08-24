#!/bin/bash -p
# One-shot exact5 R64 GPU controller template.
#
# This file is intentionally non-runnable.  A reviewer must replace every
# HOLD_* pin from the two completed CPU gates, then change CONTROLLER_STATE to
# READY in a separately hashed, reviewed controller.  Merely changing the
# state while a placeholder remains still fails before the attempt marker.

set -Eeuo pipefail
umask 077

readonly CONTROLLER_STATE=HOLD_PENDING_EXACT_CPU_GATE_PINS
[[ "$CONTROLLER_STATE" == READY ]] || {
  /usr/bin/printf '%s\n' \
    'case01 exact5 GPU controller is HOLD pending exact CPU receipt/evidence pins' >&2
  exit 88
}

[[ "$0" == /bin/bash && "$#" -eq 0 && "$-" == *p* && "$-" != *i* ]] || exit 96
[[ "${PATH:-}" == /usr/bin:/bin && "${LC_ALL:-}" == C \
  && "${LANG:-}" == C && "${HOME:-}" == /vast/users/guangyi.chen \
  && "${BASH_ENV:-}" == /dev/null ]] || exit 96
[[ -z "${ENV:-}" && -z "${LD_PRELOAD:-}" && -z "${LD_LIBRARY_PATH:-}" \
  && -z "${PYTHONPATH:-}" && -z "${PYTHONHOME:-}" \
  && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${ROCR_VISIBLE_DEVICES:-}" \
  && -z "${HIP_VISIBLE_DEVICES:-}" && -z "${GPU_DEVICE_ORDINAL:-}" ]] || exit 96
if builtin declare -F | /usr/bin/grep . >/dev/null; then exit 96; fi
if shopt -q varredir_close 2>/dev/null; then shopt -u varredir_close; fi

readonly ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_object_grounded_case01_0821_exact5_r64_canary_v1
readonly PACKAGE="$ROOT/authority/package_materialization_receipt_v1.json"
readonly LAUNCH_RECEIPT="$ROOT/launch/root_launch_receipt_exact5_v1.json"
readonly PAYLOAD="$ROOT/launch/root_launch_payload_exact5_v1.sh"
readonly ATTEMPT="$ROOT/evidence/case01_source_bone_exact5_r64_gpu_attempt_v1.json"
readonly CACHE=/tmp/bernini-case01-exact5-r64-job143808-node292-r1-rank-cache
readonly ROOT_PYTHON=/usr/bin/python3.10

[[ -f "$ROOT_PYTHON" && -x "$ROOT_PYTHON" && ! -L "$ROOT_PYTHON" \
  && -f "$PACKAGE" && ! -L "$PACKAGE" \
  && -f "$LAUNCH_RECEIPT" && ! -L "$LAUNCH_RECEIPT" \
  && -f "$PAYLOAD" && ! -L "$PAYLOAD" ]] || exit 96
[[ ! -e "$ATTEMPT" && ! -L "$ATTEMPT" \
  && ! -e "$CACHE" && ! -L "$CACHE" ]] || exit 96

# These are the same held package/payload descriptors used by preflight and
# postflight; the production payload descriptor is also stdin to the only srun.
exec {ROOT_PYTHON_FD}<"$ROOT_PYTHON"
exec {PACKAGE_FD}<"$PACKAGE"
exec {PAYLOAD_FD}<"$PAYLOAD"
exec {EVIDENCE_DIR_FD}<"$ROOT/evidence"
[[ "$ROOT_PYTHON_FD" =~ ^[0-9]+$ && "$PACKAGE_FD" =~ ^[0-9]+$ \
  && "$PAYLOAD_FD" =~ ^[0-9]+$ && "$EVIDENCE_DIR_FD" =~ ^[0-9]+$ \
  && "$ROOT_PYTHON_FD" -ge 3 \
  && "$PACKAGE_FD" -ge 3 && "$PAYLOAD_FD" -ge 3 \
  && "$EVIDENCE_DIR_FD" -ge 3 \
  && "$ROOT_PYTHON_FD" != "$PACKAGE_FD" \
  && "$ROOT_PYTHON_FD" != "$PAYLOAD_FD" \
  && "$ROOT_PYTHON_FD" != "$EVIDENCE_DIR_FD" \
  && "$PACKAGE_FD" != "$PAYLOAD_FD" \
  && "$PACKAGE_FD" != "$EVIDENCE_DIR_FD" \
  && "$PAYLOAD_FD" != "$EVIDENCE_DIR_FD" ]] || exit 97

readonly MAX_GATE_STEP="$(
"/proc/self/fd/$ROOT_PYTHON_FD" -I -S -B - \
  "$ROOT_PYTHON_FD" "$PACKAGE_FD" "$PAYLOAD_FD" "$EVIDENCE_DIR_FD" \
  "$ROOT" "$PACKAGE" "$LAUNCH_RECEIPT" "$PAYLOAD" "$ATTEMPT" "$CACHE" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

ROOT_PYTHON = "/usr/bin/python3.10"
ROOT_PYTHON_SHA256 = "11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96"
ROOT_PYTHON_SIZE = 5_937_800
JOB = "143808"
NODE = "auh7-1b-gpu-292"
CAMPAIGN = "case01-source-bone-exact5-r64-canary"
TASKS = [
    "case01-exact_original-full644",
    "case01-codec_only_present-full644",
    "case01-bone_removed-full644",
    "case01-bone_translated_up150-full644",
    "case01-sham_control_up150-full644",
]
PINS = {
    "package": {
        "sha256": "HOLD_PACKAGE_RECEIPT_SHA256",
        "receipt_digest": "HOLD_PACKAGE_RECEIPT_DIGEST",
    },
    "launch_receipt": {
        "sha256": "HOLD_PRODUCTION_LAUNCH_RECEIPT_SHA256",
        "receipt_digest": "HOLD_PRODUCTION_LAUNCH_RECEIPT_DIGEST",
    },
    "payload": {
        "sha256": "HOLD_PRODUCTION_PAYLOAD_SHA256",
        "size": "HOLD_PRODUCTION_PAYLOAD_SIZE",
    },
    "static_receipt": {
        "sha256": "HOLD_STATIC_RECEIPT_SHA256",
        "receipt_digest": "HOLD_STATIC_RECEIPT_DIGEST",
    },
    "static_evidence": {
        "sha256": "HOLD_STATIC_EVIDENCE_SHA256",
        "evidence_digest": "HOLD_STATIC_EVIDENCE_DIGEST",
    },
    "root_fake_receipt": {
        "sha256": "HOLD_ROOT_FAKE_RECEIPT_SHA256",
        "receipt_digest": "HOLD_ROOT_FAKE_RECEIPT_DIGEST",
    },
    "root_fake_evidence": {
        "sha256": "HOLD_ROOT_FAKE_EVIDENCE_SHA256",
        "evidence_digest": "HOLD_ROOT_FAKE_EVIDENCE_DIGEST",
    },
}
SHA = re.compile(r"[0-9a-f]{64}")


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise RuntimeError("duplicate JSON key")
        value[key] = item
    return value


def strict(raw):
    value = json.loads(
        raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if type(value) is not dict or raw != canonical(value) + b"\n":
        raise RuntimeError("canonical JSON plus LF differs")
    return value


def ident(value):
    return {
        "device": value.st_dev, "inode": value.st_ino,
        "uid": value.st_uid, "gid": value.st_gid, "mode": value.st_mode,
        "nlink": value.st_nlink, "rdev": value.st_rdev,
        "size": value.st_size, "blocks": getattr(value, "st_blocks", 0),
        "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns,
    }


def object_id(value):
    return (
        value.st_dev, value.st_ino, value.st_uid, value.st_gid,
        value.st_mode, value.st_nlink, value.st_rdev,
    )


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
        raise RuntimeError("held file short read")
    return raw


def held(descriptor, path, expected, mode, uid, gid, size=None, process=False):
    before = os.fstat(descriptor)
    raw = pread(descriptor, before.st_size)
    after = os.fstat(descriptor)
    named = os.lstat(path)
    if (
        os.path.realpath(path) != path or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != mode
        or before.st_uid != uid or before.st_gid != gid or before.st_nlink != 1
        or (size is not None and before.st_size != size)
        or ident(before) != ident(after) or ident(before) != ident(named)
        or hashlib.sha256(raw).hexdigest() != expected
        or (process and ident(before) != ident(os.stat("/proc/self/exe")))
    ):
        raise RuntimeError("held authority differs: " + path)
    return raw, ident(before)


def stable(path, mode, allow_empty=False):
    if (
        not os.path.isabs(path) or os.path.normpath(path) != path
        or os.path.realpath(path) != path
    ):
        raise RuntimeError("noncanonical path")
    descriptor = os.open(
        path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        before = os.fstat(descriptor)
        raw = pread(descriptor, before.st_size)
        after = os.fstat(descriptor)
        named = os.lstat(path)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != mode
        or before.st_uid != 2012 or before.st_gid != 2000 or before.st_nlink != 1
        or ident(before) != ident(after) or ident(before) != ident(named)
        or (not allow_empty and not raw)
    ):
        raise RuntimeError("stable authority differs: " + path)
    return raw, ident(before)


def stable_json(path, mode, pin, digest_field):
    raw, identity = stable(path, mode)
    value = strict(raw)
    unsigned = dict(value)
    claimed = unsigned.pop(digest_field, None)
    if (
        hashlib.sha256(raw).hexdigest() != pin["sha256"]
        or claimed != pin[digest_field] or claimed != digest(unsigned)
    ):
        raise RuntimeError("pinned JSON differs: " + path)
    return raw, identity, value


def exact_directory(path, expected, mode=0o755):
    descriptor = os.open(
        path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        before = os.fstat(descriptor)
        actual = set(os.listdir(descriptor))
        after = os.fstat(descriptor)
        named = os.lstat(path)
    finally:
        os.close(descriptor)
    if (
        os.path.realpath(path) != path or not stat.S_ISDIR(before.st_mode)
        or stat.S_IMODE(before.st_mode) != mode or before.st_uid != 2012
        or before.st_gid != 2000 or ident(before) != ident(after)
        or ident(before) != ident(named) or actual != set(expected)
    ):
        raise RuntimeError("directory closure differs: " + path)


def verify_replay_row(row, path, raw, identity, digest_value=None):
    if (
        type(row) is not dict or row.get("path") != path
        or row.get("sha256") != hashlib.sha256(raw).hexdigest()
        or row.get("identity") != identity
        or (digest_value is not None and row.get("receipt_digest") != digest_value)
    ):
        raise RuntimeError("evidence replay row differs: " + path)


if len(sys.argv) != 11:
    raise RuntimeError("GPU preflight argv differs")
python_fd, package_fd, payload_fd, evidence_dir_fd = map(int, sys.argv[1:5])
root, package_path, launch_path, payload_path, attempt_path, cache = sys.argv[5:]
if (
    root != "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/"
    "bernini_object_grounded_case01_0821_exact5_r64_canary_v1"
    or package_path != root + "/authority/package_materialization_receipt_v1.json"
    or launch_path != root + "/launch/root_launch_receipt_exact5_v1.json"
    or payload_path != root + "/launch/root_launch_payload_exact5_v1.sh"
    or attempt_path != root + "/evidence/case01_source_bone_exact5_r64_gpu_attempt_v1.json"
    or cache != "/tmp/bernini-case01-exact5-r64-job143808-node292-r1-rank-cache"
):
    raise RuntimeError("GPU path binding differs")
evidence_path = root + "/evidence"
evidence_before = os.fstat(evidence_dir_fd)
evidence_named = os.lstat(evidence_path)
if (
    os.path.realpath(evidence_path) != evidence_path
    or not stat.S_ISDIR(evidence_before.st_mode)
    or stat.S_IMODE(evidence_before.st_mode) != 0o755
    or evidence_before.st_uid != 2012 or evidence_before.st_gid != 2000
    or ident(evidence_before) != ident(evidence_named)
):
    raise RuntimeError("held evidence directory differs")
for name, pin in PINS.items():
    for field, value in pin.items():
        if field == "size":
            if type(value) is not int or value <= 0:
                raise RuntimeError("GPU pin remains HOLD: " + name + "." + field)
        elif type(value) is not str or SHA.fullmatch(value) is None:
            raise RuntimeError("GPU pin remains HOLD: " + name + "." + field)

held(
    python_fd, ROOT_PYTHON, ROOT_PYTHON_SHA256, 0o755, 0, 0,
    ROOT_PYTHON_SIZE, process=True,
)
package_raw, package_identity = held(
    package_fd, package_path, PINS["package"]["sha256"], 0o400, 2012, 2000,
)
package = strict(package_raw)
package_unsigned = dict(package)
package_digest = package_unsigned.pop("receipt_digest", None)
package_fields = {
    "schema_version", "status", "root", "holder_job_id", "expected_node",
    "campaign_mode", "selected_task_ids", "task_count",
    "physical_release_file_count", "production_identity_count",
    "production_identity_decomposition", "sealed_r5f_infer_lora_reused",
    "working_tree_infer_lora_read", "captured_materializer_sha256",
    "input_root", "independent_audit", "plan", "launch", "cpu_admission",
    "rank_cache_root", "fresh_outputs", "fresh_final", "fresh_runtime",
    "publication_final_internal_paths_pairwise_disjoint",
    "slurm_step_launched", "gpu_attempt_claimed",
    "retry_allowed_after_gpu_attempt", "artifacts_before_materialization_receipt",
    "receipt_digest",
}
if (
    set(package) != package_fields
    or package.get("schema_version") != "case01-source-bone-exact5-r64-materialization-v1"
    or package.get("status") != "MATERIALIZED_NOT_SUBMITTED"
    or package.get("root") != root or package.get("holder_job_id") != JOB
    or package.get("expected_node") != NODE or package.get("campaign_mode") != CAMPAIGN
    or package.get("selected_task_ids") != TASKS or package.get("task_count") != 5
    or package.get("physical_release_file_count") != 19
    or package.get("production_identity_count") != 18
    or package.get("sealed_r5f_infer_lora_reused") is not True
    or package.get("working_tree_infer_lora_read") is not False
    or package.get("fresh_outputs") is not True
    or package.get("fresh_final") is not True
    or package.get("fresh_runtime") is not True
    or package.get("publication_final_internal_paths_pairwise_disjoint") is not True
    or package.get("slurm_step_launched") is not False
    or package.get("gpu_attempt_claimed") is not False
    or package.get("retry_allowed_after_gpu_attempt") is not False
    or package.get("rank_cache_root") != cache
    or package_digest != PINS["package"]["receipt_digest"]
    or package_digest != digest(package_unsigned)
):
    raise RuntimeError("package semantic replay differs")

launch_raw, launch_identity, launch = stable_json(
    launch_path, 0o400, PINS["launch_receipt"], "receipt_digest"
)
launch_fields = {
    "schema_version", "status", "launch_input", "release", "release_digest",
    "root_bootstrap_sha256", "payload_path", "payload_sha256", "payload_size",
    "payload_mode", "receipt_path", "required_entry",
    "named_payload_execution_forbidden", "submission_or_execution_performed",
    "remote_execution_authorized_by_this_receipt", "receipt_digest",
}
release = launch.get("release", {})
identities = release.get("identities", {}) if type(release) is dict else {}
if (
    set(launch) != launch_fields
    or launch.get("schema_version") != "case01-source-bone-exact5-root-launch-receipt-auh-v1"
    or launch.get("status") != "MATERIALIZED_NOT_SUBMITTED"
    or launch.get("payload_path") != payload_path
    or launch.get("payload_sha256") != PINS["payload"]["sha256"]
    or launch.get("payload_size") != PINS["payload"]["size"]
    or launch.get("payload_mode") != 0o444
    or launch.get("named_payload_execution_forbidden") is not True
    or launch.get("submission_or_execution_performed") is not False
    or launch.get("remote_execution_authorized_by_this_receipt") is not True
    or launch.get("release_digest") != digest(release)
    or release.get("campaign_mode") != CAMPAIGN
    or release.get("task_count") != 5
    or release.get("selected_task_ids") != TASKS
    or release.get("retry_allowed") is not False
    or release.get("partial_outputs_are_not_results") is not True
    or release.get("all_exact18_named_identities_replayed_before_runner") is not True
    or type(identities) is not dict or len(identities) != 18
):
    raise RuntimeError("production launch receipt differs")
payload_raw, payload_identity = held(
    payload_fd, payload_path, PINS["payload"]["sha256"], 0o444, 2012, 2000,
    PINS["payload"]["size"],
)
if (
    package.get("launch", {}).get("receipt") != launch_path
    or package.get("launch", {}).get("receipt_sha256") != hashlib.sha256(launch_raw).hexdigest()
    or package.get("launch", {}).get("receipt_digest") != launch["receipt_digest"]
    or package.get("launch", {}).get("payload") != payload_path
    or package.get("launch", {}).get("payload_sha256") != hashlib.sha256(payload_raw).hexdigest()
    or package.get("launch", {}).get("payload_size") != len(payload_raw)
    or launch.get("payload_sha256") != hashlib.sha256(payload_raw).hexdigest()
):
    raise RuntimeError("package/launch/held payload binding differs")

# Replay the plan and independent audit pinned by the package.
for row_name, digest_field in (("plan", "plan_digest"), ("independent_audit", "audit_digest")):
    row = package.get(row_name, {})
    raw, _ = stable(row.get("path", ""), 0o444)
    value = strict(raw)
    unsigned = dict(value)
    claimed = unsigned.pop(digest_field, None)
    if (
        hashlib.sha256(raw).hexdigest() != row.get("sha256")
        or len(raw) != row.get("size") or claimed != row.get(digest_field)
        or claimed != digest(unsigned)
    ):
        raise RuntimeError(row_name + " authority differs")
    if row_name == "plan" and (
        value.get("task_count") != 5
        or [task.get("task_id") for task in value.get("tasks", [])] != TASKS
    ):
        raise RuntimeError("exact5 plan task closure differs")

paths = {
    "static_receipt": root + "/evidence/exact5_static_probe_receipt_v1.json",
    "static_attempt": root + "/evidence/exact5_static_probe_attempt_v1.json",
    "static_evidence": root + "/evidence/exact5_static_probe_controller_evidence_v1.json",
    "static_payload": root + "/diagnostics/exact5_static_probe_payload_v1.sh",
    "static_stdout": root + "/logs/exact5_static_probe.stdout.log",
    "static_stderr": root + "/logs/exact5_static_probe.stderr.log",
    "root_fake_receipt": root + "/evidence/exact5_root_fake_runner_probe_receipt_v1.json",
    "root_fake_attempt": root + "/evidence/exact5_root_fake_runner_attempt_v1.json",
    "root_fake_evidence": root + "/evidence/exact5_root_fake_runner_controller_evidence_v1.json",
    "root_fake_payload": root + "/diagnostics/root_fake_launch_payload_v1.sh",
    "root_fake_materialization": root + "/diagnostics/root_fake_launch_materialization_receipt_v1.json",
    "root_fake_stdout": root + "/logs/exact5_root_fake_runner.stdout.log",
    "root_fake_stderr": root + "/logs/exact5_root_fake_runner.stderr.log",
}
static_receipt_raw, static_receipt_identity, static_receipt = stable_json(
    paths["static_receipt"], 0o400, PINS["static_receipt"], "receipt_digest"
)
static_evidence_raw, _, static_evidence = stable_json(
    paths["static_evidence"], 0o400, PINS["static_evidence"], "evidence_digest"
)
root_receipt_raw, root_receipt_identity, root_receipt = stable_json(
    paths["root_fake_receipt"], 0o400, PINS["root_fake_receipt"], "receipt_digest"
)
root_evidence_raw, _, root_evidence = stable_json(
    paths["root_fake_evidence"], 0o400, PINS["root_fake_evidence"], "evidence_digest"
)

static_receipt_fields = {
    "schema_version", "status", "campaign_mode", "holder_job_id",
    "expected_node", "slurm_step_id", "task_count", "selected_task_ids",
    "release_file_count", "launch_identity_count", "plan_sha256", "plan_digest",
    "independent_audit_sha256", "independent_audit_digest",
    "checkpoint_manifest_sha256", "launch_receipt_sha256",
    "launch_receipt_digest", "payload_sha256", "ffprobe_path",
    "ffprobe_sha256", "rank_cache_root", "production_outputs_fresh",
    "rank_cache_fresh", "pure_metadata_only", "torch_imported",
    "renderer_executed", "receipt_digest",
}
root_receipt_fields = {
    "schema_version", "status", "campaign_mode", "holder_job_id",
    "expected_node", "slurm_step_id", "task_count", "selected_task_ids",
    "plan_sha256", "plan_digest", "runner_sha256", "entry_authority_digest",
    "release_digest", "captured_source_entry", "held_python_fd_entry",
    "all_exact18_named_identities_replayed_by_root_bootstrap",
    "slurm_environment_from_step", "torch_imported", "renderer_executed",
    "receipt_digest",
}
if (
    set(static_receipt) != static_receipt_fields
    or static_receipt.get("schema_version") != "case01-source-bone-exact5-static-probe-v1"
    or static_receipt.get("status") != "PASS"
    or static_receipt.get("campaign_mode") != CAMPAIGN
    or static_receipt.get("holder_job_id") != JOB
    or static_receipt.get("expected_node") != NODE
    or static_receipt.get("task_count") != 5
    or static_receipt.get("selected_task_ids") != TASKS
    or static_receipt.get("release_file_count") != 19
    or static_receipt.get("launch_identity_count") != 18
    or static_receipt.get("plan_sha256") != package["plan"]["sha256"]
    or static_receipt.get("plan_digest") != package["plan"]["plan_digest"]
    or static_receipt.get("independent_audit_sha256") != package["independent_audit"]["sha256"]
    or static_receipt.get("independent_audit_digest") != package["independent_audit"]["audit_digest"]
    or static_receipt.get("checkpoint_manifest_sha256")
    != "7a4864a3ffa50c12af91f8d2b88610a6cd8f994aa68eef8d27b95bcc2d73d3b2"
    or static_receipt.get("launch_receipt_sha256") != package["launch"]["receipt_sha256"]
    or static_receipt.get("launch_receipt_digest") != package["launch"]["receipt_digest"]
    or static_receipt.get("payload_sha256") != package["launch"]["payload_sha256"]
    or static_receipt.get("rank_cache_root") != cache
    or static_receipt.get("ffprobe_path")
    != "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe"
    or static_receipt.get("ffprobe_sha256")
    != "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5"
    or static_receipt.get("production_outputs_fresh") is not True
    or static_receipt.get("rank_cache_fresh") is not True
    or static_receipt.get("pure_metadata_only") is not True
    or static_receipt.get("torch_imported") is not False
    or static_receipt.get("renderer_executed") is not False
):
    raise RuntimeError("static gate receipt semantics differ")
if (
    set(root_receipt) != root_receipt_fields
    or root_receipt.get("schema_version") != "case01-source-bone-exact5-root-fake-runner-probe-v1"
    or root_receipt.get("status") != "PASS"
    or root_receipt.get("campaign_mode") != CAMPAIGN
    or root_receipt.get("holder_job_id") != JOB
    or root_receipt.get("expected_node") != NODE
    or root_receipt.get("task_count") != 5
    or root_receipt.get("selected_task_ids") != TASKS
    or root_receipt.get("plan_sha256") != package["plan"]["sha256"]
    or root_receipt.get("plan_digest") != package["plan"]["plan_digest"]
    or root_receipt.get("runner_sha256")
    != package["cpu_admission"]["captured_root_fake_runner_probe"]["runner_sha256"]
    or root_receipt.get("captured_source_entry") is not True
    or root_receipt.get("held_python_fd_entry") is not True
    or root_receipt.get("all_exact18_named_identities_replayed_by_root_bootstrap") is not True
    or root_receipt.get("slurm_environment_from_step") is not True
    or root_receipt.get("torch_imported") is not False
    or root_receipt.get("renderer_executed") is not False
):
    raise RuntimeError("root-fake gate receipt semantics differ")

attempt_fields = {
    "schema_version", "status", "holder_job_id", "node",
    "package_receipt_sha256", "package_receipt_digest", "payload_path",
    "payload_sha256", "receipt_path", "single_srun_attempt", "retry_allowed",
    "renderer_executed", "attempt_digest",
}
root_attempt_fields = attempt_fields | {"materialization_receipt_sha256"}
static_attempt_raw, static_attempt_identity = stable(paths["static_attempt"], 0o400)
static_attempt = strict(static_attempt_raw)
static_attempt_unsigned = dict(static_attempt)
static_attempt_digest = static_attempt_unsigned.pop("attempt_digest", None)
root_attempt_raw, root_attempt_identity = stable(paths["root_fake_attempt"], 0o400)
root_attempt = strict(root_attempt_raw)
root_attempt_unsigned = dict(root_attempt)
root_attempt_digest = root_attempt_unsigned.pop("attempt_digest", None)
if (
    set(static_attempt) != attempt_fields
    or static_attempt_digest != digest(static_attempt_unsigned)
    or static_attempt.get("schema_version") != "case01-source-bone-exact5-static-attempt-v1"
    or set(root_attempt) != root_attempt_fields
    or root_attempt_digest != digest(root_attempt_unsigned)
    or root_attempt.get("schema_version") != "case01-source-bone-exact5-root-fake-attempt-v1"
):
    raise RuntimeError("CPU gate attempt schema/digest differs")
for attempt, receipt_path in (
    (static_attempt, paths["static_receipt"]),
    (root_attempt, paths["root_fake_receipt"]),
):
    if (
        attempt.get("status") != "ATTEMPT_CLAIMED_BEFORE_SRUN"
        or attempt.get("holder_job_id") != JOB or attempt.get("node") != NODE
        or attempt.get("package_receipt_sha256") != hashlib.sha256(package_raw).hexdigest()
        or attempt.get("package_receipt_digest") != package_digest
        or attempt.get("receipt_path") != receipt_path
        or attempt.get("single_srun_attempt") is not True
        or attempt.get("retry_allowed") is not False
        or attempt.get("renderer_executed") is not False
    ):
        raise RuntimeError("CPU gate attempt semantics differ")
if (
    static_attempt.get("payload_path") != paths["static_payload"]
    or static_attempt.get("payload_sha256")
    != package["cpu_admission"]["static_probe"]["payload_sha256"]
    or root_attempt.get("payload_path") != paths["root_fake_payload"]
    or root_attempt.get("payload_sha256")
    != package["cpu_admission"]["captured_root_fake_runner_probe"]["payload_sha256"]
):
    raise RuntimeError("CPU gate attempt/payload binding differs")

evidence_fields = {
    "schema_version", "status", "holder_job_id", "node", "numeric_step",
    "single_srun_attempt", "srun_returncode", "package_replay",
    "payload_replay", "receipt_replay", "attempt_replay", "stdout", "stderr",
    "retry_allowed", "renderer_executed", "evidence_digest",
}
root_evidence_fields = evidence_fields | {"materialization_replay"}
if set(static_evidence) != evidence_fields or set(root_evidence) != root_evidence_fields:
    raise RuntimeError("CPU gate evidence schema differs")

gate_rows = []
for label, evidence, receipt_raw, receipt_identity, receipt, attempt_raw, attempt_identity, attempt in (
    (
        "static", static_evidence, static_receipt_raw, static_receipt_identity,
        static_receipt, static_attempt_raw, static_attempt_identity, static_attempt,
    ),
    (
        "root_fake", root_evidence, root_receipt_raw, root_receipt_identity,
        root_receipt, root_attempt_raw, root_attempt_identity, root_attempt,
    ),
):
    step = evidence.get("numeric_step", "")
    if (
        evidence.get("status") != "PASS" or evidence.get("holder_job_id") != JOB
        or evidence.get("node") != NODE or evidence.get("single_srun_attempt") is not True
        or evidence.get("srun_returncode") != 0
        or evidence.get("retry_allowed") is not False
        or evidence.get("renderer_executed") is not False
        or type(step) is not str or not step.startswith(JOB + ".")
        or not step[len(JOB) + 1:].isascii()
        or not step[len(JOB) + 1:].isdecimal()
        or int(step[len(JOB) + 1:]) <= 394
        or receipt.get("slurm_step_id") != step[len(JOB) + 1:]
    ):
        raise RuntimeError(label + " gate terminal semantics differ")
    verify_replay_row(
        evidence.get("package_replay"), package_path, package_raw,
        package_identity, package_digest,
    )
    verify_replay_row(
        evidence.get("receipt_replay"), paths[label + "_receipt"],
        receipt_raw, receipt_identity, receipt["receipt_digest"],
    )
    attempt_row = evidence.get("attempt_replay")
    if (
        type(attempt_row) is not dict
        or attempt_row.get("path") != paths[label + "_attempt"]
        or attempt_row.get("sha256") != hashlib.sha256(attempt_raw).hexdigest()
        or attempt_row.get("attempt_digest") != attempt["attempt_digest"]
        or attempt_row.get("identity") != attempt_identity
    ):
        raise RuntimeError(label + " attempt evidence differs")
    gate_rows.append(int(step[len(JOB) + 1:]))

static_payload_raw, static_payload_identity = stable(paths["static_payload"], 0o444)
root_payload_raw, root_payload_identity = stable(paths["root_fake_payload"], 0o444)
verify_replay_row(
    static_evidence.get("payload_replay"), paths["static_payload"],
    static_payload_raw, static_payload_identity,
)
verify_replay_row(
    root_evidence.get("payload_replay"), paths["root_fake_payload"],
    root_payload_raw, root_payload_identity,
)
root_materialization_raw, root_materialization_identity = stable(
    paths["root_fake_materialization"], 0o400
)
root_materialization = strict(root_materialization_raw)
root_materialization_unsigned = dict(root_materialization)
root_materialization_digest = root_materialization_unsigned.pop("receipt_digest", None)
root_materialization_fields = {
    "schema_version", "status", "launch_input", "release", "release_digest",
    "root_bootstrap_sha256", "payload_path", "payload_sha256", "payload_size",
    "payload_mode", "receipt_path", "required_entry",
    "named_payload_execution_forbidden", "submission_or_execution_performed",
    "remote_execution_authorized_by_this_receipt", "receipt_digest",
}
root_fake_release = root_materialization.get("release", {})
root_fake_identities = (
    root_fake_release.get("identities", {}) if type(root_fake_release) is dict else {}
)
if (
    set(root_materialization) != root_materialization_fields
    or root_materialization_digest != digest(root_materialization_unsigned)
    or root_materialization.get("schema_version")
    != "case01-source-bone-exact5-root-launch-receipt-auh-v1"
    or root_materialization.get("status") != "MATERIALIZED_NOT_SUBMITTED"
    or root_materialization.get("payload_path") != paths["root_fake_payload"]
    or root_materialization.get("payload_sha256")
    != hashlib.sha256(root_payload_raw).hexdigest()
    or root_materialization.get("payload_size") != len(root_payload_raw)
    or root_materialization.get("payload_mode") != 0o444
    or root_materialization.get("release_digest") != digest(root_fake_release)
    or root_fake_release.get("campaign_mode") != CAMPAIGN
    or root_fake_release.get("task_count") != 5
    or root_fake_release.get("selected_task_ids") != TASKS
    or type(root_fake_identities) is not dict or len(root_fake_identities) != 18
):
    raise RuntimeError("root-fake materialization digest differs")
if root_receipt.get("release_digest") != root_materialization.get("release_digest"):
    raise RuntimeError("root-fake receipt/materialization release differs")
verify_replay_row(
    root_evidence.get("materialization_replay"),
    paths["root_fake_materialization"], root_materialization_raw,
    root_materialization_identity, root_materialization_digest,
)
if (
    root_attempt.get("materialization_receipt_sha256")
    != hashlib.sha256(root_materialization_raw).hexdigest()
):
    raise RuntimeError("root-fake attempt/materialization binding differs")

for label, evidence, receipt in (
    ("static", static_evidence, static_receipt),
    ("root_fake", root_evidence, root_receipt),
):
    stdout_raw, stdout_identity = stable(paths[label + "_stdout"], 0o400)
    stderr_raw, stderr_identity = stable(paths[label + "_stderr"], 0o400, True)
    expected_stdout = (
        ("CASE01_EXACT5_STATIC_PASS " if label == "static" else "CASE01_EXACT5_ROOT_FAKE_PASS ")
        + receipt["receipt_digest"] + "\n"
    ).encode("ascii")
    stdout_row = evidence.get("stdout")
    stderr_row = evidence.get("stderr")
    if (
        stdout_raw != expected_stdout or stderr_raw != b""
        or stdout_row.get("path") != paths[label + "_stdout"]
        or stdout_row.get("sha256") != hashlib.sha256(stdout_raw).hexdigest()
        or stdout_row.get("identity") != stdout_identity
        or stderr_row.get("path") != paths[label + "_stderr"]
        or stderr_row.get("sha256") != hashlib.sha256(stderr_raw).hexdigest()
        or stderr_row.get("identity") != stderr_identity
        or stderr_row.get("empty") is not True
    ):
        raise RuntimeError(label + " log replay differs")

if len(set(gate_rows)) != 2:
    raise RuntimeError("CPU gate numeric steps are not distinct")
exact_directory(root + "/outputs", {"media"})
for relative in ("outputs/media", "final", "runtime"):
    exact_directory(root + "/" + relative, set())
exact_directory(
    root + "/evidence",
    {
        "exact5_static_probe_receipt_v1.json",
        "exact5_static_probe_attempt_v1.json",
        "exact5_static_probe_controller_evidence_v1.json",
        "exact5_root_fake_runner_probe_receipt_v1.json",
        "exact5_root_fake_runner_attempt_v1.json",
        "exact5_root_fake_runner_controller_evidence_v1.json",
    },
)
exact_directory(
    root + "/logs",
    {
        "exact5_static_probe.stdout.log", "exact5_static_probe.stderr.log",
        "exact5_root_fake_runner.stdout.log",
        "exact5_root_fake_runner.stderr.log",
    },
)
if os.path.lexists(cache) or os.path.lexists(attempt_path):
    raise RuntimeError("GPU fresh target differs")

# The attempt parent is held by directory FD.  No named chmod is used.
attempt_parent = str(Path(attempt_path).parent)
if attempt_parent != evidence_path or ident(os.fstat(evidence_dir_fd)) != ident(evidence_before):
    raise RuntimeError("GPU attempt held parent binding differs")
parent_fd = evidence_dir_fd
descriptor = -1
try:
    parent_before = os.fstat(parent_fd)
    parent_named = os.lstat(attempt_parent)
    if (
        os.path.realpath(attempt_parent) != attempt_parent
        or not stat.S_ISDIR(parent_before.st_mode)
        or stat.S_IMODE(parent_before.st_mode) != 0o755
        or parent_before.st_uid != 2012 or parent_before.st_gid != 2000
        or ident(parent_before) != ident(parent_named)
    ):
        raise RuntimeError("GPU attempt parent differs")
    attempt = {
        "schema_version": "case01-source-bone-exact5-r64-gpu-attempt-v1",
        "status": "ATTEMPT_CLAIMED_BEFORE_SRUN",
        "holder_job_id": JOB, "node": NODE, "campaign_mode": CAMPAIGN,
        "selected_task_ids": TASKS, "task_count": 5, "root": root,
        "rank_cache_root": cache,
        "package_receipt_sha256": hashlib.sha256(package_raw).hexdigest(),
        "package_receipt_digest": package_digest,
        "package_identity": package_identity,
        "launch_receipt_sha256": hashlib.sha256(launch_raw).hexdigest(),
        "launch_receipt_digest": launch["receipt_digest"],
        "launch_receipt_identity": launch_identity,
        "production_payload_sha256": hashlib.sha256(payload_raw).hexdigest(),
        "production_payload_size": len(payload_raw),
        "production_payload_identity": payload_identity,
        "static_receipt_sha256": hashlib.sha256(static_receipt_raw).hexdigest(),
        "static_receipt_digest": static_receipt["receipt_digest"],
        "static_evidence_sha256": hashlib.sha256(static_evidence_raw).hexdigest(),
        "static_evidence_digest": static_evidence["evidence_digest"],
        "root_fake_receipt_sha256": hashlib.sha256(root_receipt_raw).hexdigest(),
        "root_fake_receipt_digest": root_receipt["receipt_digest"],
        "root_fake_evidence_sha256": hashlib.sha256(root_evidence_raw).hexdigest(),
        "root_fake_evidence_digest": root_evidence["evidence_digest"],
        "cpu_gate_numeric_steps": [JOB + "." + str(value) for value in gate_rows],
        "single_srun_attempt": True, "retry_allowed": False,
        "partial_outputs_are_not_results": True,
    }
    attempt["attempt_digest"] = digest(attempt)
    attempt_raw = canonical(attempt) + b"\n"
    descriptor = os.open(
        Path(attempt_path).name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0, dir_fd=parent_fd,
    )
    offset = 0
    while offset < len(attempt_raw):
        count = os.write(descriptor, attempt_raw[offset:])
        if count <= 0:
            raise RuntimeError("GPU attempt write made no progress")
        offset += count
    os.fsync(descriptor)
    staged = os.fstat(descriptor)
    named = os.stat(Path(attempt_path).name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(staged.st_mode) or stat.S_IMODE(staged.st_mode) != 0
        or staged.st_nlink != 1 or ident(staged) != ident(named)
        or pread(descriptor, len(attempt_raw)) != attempt_raw
        or object_id(os.fstat(parent_fd)) != object_id(parent_before)
    ):
        raise RuntimeError("GPU attempt staging replay differs")
    os.fchmod(descriptor, 0o400)
    os.fsync(descriptor)
    committed = os.fstat(descriptor)
    committed_named = os.stat(
        Path(attempt_path).name, dir_fd=parent_fd, follow_symlinks=False,
    )
    if (
        stat.S_IMODE(committed.st_mode) != 0o400
        or ident(committed) != ident(committed_named)
        or pread(descriptor, len(attempt_raw)) != attempt_raw
        or object_id(os.fstat(parent_fd)) != object_id(parent_before)
        or ident(os.lstat(attempt_parent)) != ident(os.fstat(parent_fd))
    ):
        raise RuntimeError("GPU attempt commit replay differs")
finally:
    if descriptor >= 0:
        os.close(descriptor)

# Recheck the same held FDs after all named gate reads and before returning.
if (
    ident(os.fstat(package_fd)) != package_identity
    or hashlib.sha256(pread(package_fd, os.fstat(package_fd).st_size)).hexdigest()
    != PINS["package"]["sha256"]
    or ident(os.fstat(payload_fd)) != payload_identity
    or hashlib.sha256(pread(payload_fd, os.fstat(payload_fd).st_size)).hexdigest()
    != PINS["payload"]["sha256"]
):
    raise RuntimeError("held production authority changed before srun")
print(max(gate_rows))
PY
)"
[[ "$MAX_GATE_STEP" =~ ^[1-9][0-9]*$ ]] || exit 98

set +e
/usr/bin/srun --jobid=143808 --job-name=case01-exact5-r64-gpu-v1 \
  --exclusive --exact --immediate=10 --kill-on-bad-exit=1 \
  --nodes=1 --ntasks=1 --nodelist=auh7-1b-gpu-292 \
  --cpus-per-task=64 --mem=64G --gpus-per-node=8 \
  --export=NONE --time=03:00:00 \
  /bin/bash -p -c '
[[ "${SLURM_JOB_ID-}" == 143808 && "${SLURM_STEP_ID-}" =~ ^[1-9][0-9]*$ ]] || exit 81
[[ "$1" =~ ^[1-9][0-9]*$ ]] || exit 82
(( 10#$SLURM_STEP_ID > 10#$1 )) || exit 83
[[ "${SLURM_GPUS_ON_NODE-}" == 8 && "${SLURM_GPUS_PER_NODE-}" == 8 \
  && "${SLURM_STEP_GPUS-}" == 0,1,2,3,4,5,6,7 ]] || exit 84
[[ "${SLURM_NNODES-}" == 1 && "${SLURM_STEP_NUM_NODES-}" == 1 \
  && "${SLURM_JOB_NODELIST-}" == auh7-1b-gpu-292 \
  && "${SLURM_STEP_NODELIST-}" == auh7-1b-gpu-292 ]] || exit 85
[[ -z "${SLURM_JOB_GPUS+x}" && -z "${SLURM_JOB_NUM_NODES+x}" ]] || exit 86
[[ ! -e /tmp/bernini-case01-exact5-r64-job143808-node292-r1-rank-cache \
  && ! -L /tmp/bernini-case01-exact5-r64-job143808-node292-r1-rank-cache ]] || exit 87
exec /bin/bash -p -s
' /bin/bash "$MAX_GATE_STEP" <&"$PAYLOAD_FD"
readonly SRUN_RC=$?
set -e

# Post-use replay uses the very same held package/payload FDs and the immutable
# pre-srun marker.  It performs no named chmod and creates no second attempt.
"/proc/self/fd/$ROOT_PYTHON_FD" -I -S -B - \
  "$PACKAGE_FD" "$PAYLOAD_FD" "$EVIDENCE_DIR_FD" \
  "$PACKAGE" "$PAYLOAD" "$ATTEMPT" <<'PY'
import hashlib,json,os,stat,sys
def canonical(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def ident(v): return {"device":v.st_dev,"inode":v.st_ino,"uid":v.st_uid,"gid":v.st_gid,"mode":v.st_mode,"nlink":v.st_nlink,"rdev":v.st_rdev,"size":v.st_size,"blocks":getattr(v,"st_blocks",0),"mtime_ns":v.st_mtime_ns,"ctime_ns":v.st_ctime_ns}
def pread(fd,size):
 out=[];off=0
 while off<size:
  block=os.pread(fd,min(1048576,size-off),off)
  if not block: break
  out.append(block);off+=len(block)
 raw=b"".join(out)
 if len(raw)!=size: raise RuntimeError("post-use short read")
 return raw
package_fd,payload_fd,evidence_dir_fd=map(int,sys.argv[1:4]);package_path,payload_path,attempt_path=sys.argv[4:]
evidence_path=os.path.dirname(attempt_path);attempt_name=os.path.basename(attempt_path)
evidence_info=os.fstat(evidence_dir_fd);evidence_named=os.lstat(evidence_path)
attempt_fd=os.open(attempt_name,os.O_RDONLY|os.O_CLOEXEC|getattr(os,"O_NOFOLLOW",0),dir_fd=evidence_dir_fd)
try:
 attempt_info=os.fstat(attempt_fd);attempt_raw=pread(attempt_fd,attempt_info.st_size);attempt_named=os.stat(attempt_name,dir_fd=evidence_dir_fd,follow_symlinks=False)
finally: os.close(attempt_fd)
attempt=json.loads(attempt_raw)
unsigned=dict(attempt);claimed=unsigned.pop("attempt_digest",None)
package_info=os.fstat(package_fd);payload_info=os.fstat(payload_fd)
package_raw=pread(package_fd,package_info.st_size);payload_raw=pread(payload_fd,payload_info.st_size)
if (os.path.realpath(evidence_path)!=evidence_path or not stat.S_ISDIR(evidence_info.st_mode)
 or stat.S_IMODE(evidence_info.st_mode)!=0o755 or evidence_info.st_uid!=2012 or evidence_info.st_gid!=2000
 or ident(evidence_info)!=ident(evidence_named) or not stat.S_ISREG(attempt_info.st_mode)
 or stat.S_IMODE(attempt_info.st_mode)!=0o400 or attempt_info.st_uid!=2012 or attempt_info.st_gid!=2000
 or attempt_info.st_nlink!=1 or ident(attempt_info)!=ident(attempt_named)
 or attempt_raw!=canonical(attempt)+b"\n" or claimed!=hashlib.sha256(canonical(unsigned)).hexdigest()
 or attempt.get("status")!="ATTEMPT_CLAIMED_BEFORE_SRUN"
 or attempt.get("single_srun_attempt") is not True or attempt.get("retry_allowed") is not False
 or ident(package_info)!=attempt.get("package_identity") or ident(package_info)!=ident(os.lstat(package_path))
 or hashlib.sha256(package_raw).hexdigest()!=attempt.get("package_receipt_sha256")
 or ident(payload_info)!=attempt.get("production_payload_identity") or ident(payload_info)!=ident(os.lstat(payload_path))
 or hashlib.sha256(payload_raw).hexdigest()!=attempt.get("production_payload_sha256")):
 raise RuntimeError("post-use held authority replay differs")
print("CASE01_EXACT5_GPU_POST_USE_REPLAY_PASS "+claimed)
PY

[[ "$SRUN_RC" -eq 0 ]] || exit "$SRUN_RC"
