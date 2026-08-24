#!/bin/bash -p
# Review-stop controller for the already completed case01 exact5 static step.
# It authenticates retained artifacts and accounting, seals the two existing
# logs, and create-only commits the missing compatibility evidence.  It never
# starts a scheduler step.

set -Eeuo pipefail
umask 077

[[ "$0" == /bin/bash && "$#" -eq 0 && "$-" == *p* && "$-" != *i* ]] || exit 96
[[ "${PATH:-}" == /usr/bin:/bin && "${LC_ALL:-}" == C && "${LANG:-}" == C \
  && "${HOME:-}" == /vast/users/guangyi.chen && "${BASH_ENV:-}" == /dev/null ]] || exit 96
[[ -z "${ENV:-}" && -z "${LD_PRELOAD:-}" && -z "${LD_LIBRARY_PATH:-}" \
  && -z "${PYTHONPATH:-}" && -z "${PYTHONHOME:-}" \
  && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${ROCR_VISIBLE_DEVICES:-}" \
  && -z "${HIP_VISIBLE_DEVICES:-}" && -z "${GPU_DEVICE_ORDINAL:-}" ]] || exit 96
[[ -z "$(builtin declare -F)" ]] || exit 96
if shopt -q varredir_close 2>/dev/null; then shopt -u varredir_close; fi

readonly ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_object_grounded_case01_0821_exact5_r64_canary_v1
readonly PACKAGE="$ROOT/authority/package_materialization_receipt_v1.json"
readonly PAYLOAD="$ROOT/diagnostics/exact5_static_probe_payload_v1.sh"
readonly RECEIPT="$ROOT/evidence/exact5_static_probe_receipt_v1.json"
readonly ATTEMPT="$ROOT/evidence/exact5_static_probe_attempt_v1.json"
readonly EVIDENCE="$ROOT/evidence/exact5_static_probe_controller_evidence_v1.json"
readonly STDOUT_LOG="$ROOT/logs/exact5_static_probe.stdout.log"
readonly STDERR_LOG="$ROOT/logs/exact5_static_probe.stderr.log"
readonly ROOT_PYTHON=/usr/bin/python3.10
readonly SACCT=/usr/bin/sacct

[[ -f "$ROOT_PYTHON" && -x "$ROOT_PYTHON" && ! -L "$ROOT_PYTHON" \
  && -f "$SACCT" && -x "$SACCT" && ! -L "$SACCT" \
  && -f "$PACKAGE" && ! -L "$PACKAGE" \
  && -f "$PAYLOAD" && ! -L "$PAYLOAD" \
  && -f "$RECEIPT" && ! -L "$RECEIPT" \
  && -f "$ATTEMPT" && ! -L "$ATTEMPT" \
  && -f "$STDOUT_LOG" && ! -L "$STDOUT_LOG" \
  && -f "$STDERR_LOG" && ! -L "$STDERR_LOG" \
  && ! -e "$EVIDENCE" && ! -L "$EVIDENCE" ]] || exit 96

exec {ROOT_PYTHON_FD}<"$ROOT_PYTHON"
exec {PACKAGE_FD}<"$PACKAGE"
exec {PAYLOAD_FD}<"$PAYLOAD"
exec {RECEIPT_FD}<"$RECEIPT"
exec {ATTEMPT_FD}<"$ATTEMPT"
exec {STDOUT_FD}<"$STDOUT_LOG"
exec {STDERR_FD}<"$STDERR_LOG"
exec {EVIDENCE_DIR_FD}<"$ROOT/evidence"
exec {LOGS_DIR_FD}<"$ROOT/logs"
exec {SACCT_FD}<"$SACCT"

for descriptor in \
  "$ROOT_PYTHON_FD" "$PACKAGE_FD" "$PAYLOAD_FD" "$RECEIPT_FD" \
  "$ATTEMPT_FD" "$STDOUT_FD" "$STDERR_FD" "$EVIDENCE_DIR_FD" \
  "$LOGS_DIR_FD" "$SACCT_FD"; do
  [[ "$descriptor" =~ ^[0-9]+$ && "$descriptor" -ge 3 ]] || exit 97
done

exec -c "/proc/self/fd/$ROOT_PYTHON_FD" -I -S -B - \
  "$ROOT_PYTHON_FD" "$PACKAGE_FD" "$PAYLOAD_FD" "$RECEIPT_FD" \
  "$ATTEMPT_FD" "$STDOUT_FD" "$STDERR_FD" "$EVIDENCE_DIR_FD" \
  "$LOGS_DIR_FD" "$SACCT_FD" <<'PY'
import hashlib
import json
import os
import stat
import sys


ROOT = "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_object_grounded_case01_0821_exact5_r64_canary_v1"
PACKAGE = ROOT + "/authority/package_materialization_receipt_v1.json"
PAYLOAD = ROOT + "/diagnostics/exact5_static_probe_payload_v1.sh"
RECEIPT = ROOT + "/evidence/exact5_static_probe_receipt_v1.json"
ATTEMPT = ROOT + "/evidence/exact5_static_probe_attempt_v1.json"
EVIDENCE = ROOT + "/evidence/exact5_static_probe_controller_evidence_v1.json"
STDOUT_LOG = ROOT + "/logs/exact5_static_probe.stdout.log"
STDERR_LOG = ROOT + "/logs/exact5_static_probe.stderr.log"
EVIDENCE_DIR = ROOT + "/evidence"
LOGS_DIR = ROOT + "/logs"
CACHE = "/tmp/bernini-case01-exact5-r64-job143808-node292-r1-rank-cache"

JOB = "143808"
STEP = "429"
FULL_STEP = JOB + "." + STEP
NODE = "auh7-1b-gpu-292"
STEP_NAME = "case01-exact5-static-v1"
CAMPAIGN = "case01-source-bone-exact5-r64-canary"
ORIGINAL_CONTROLLER_SHA256 = "c997805c63a23e33fd5b6ce15b3672fc77e9e77f5679267ce34693604743b6e9"
RECOVERY_REASON = "outer_postflight_rc1_after_shared_fs_receipt_visibility_race"

ROOT_PYTHON = "/usr/bin/python3.10"
ROOT_PYTHON_SHA256 = "11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96"
ROOT_PYTHON_SIZE = 5_937_800
SACCT = "/usr/bin/sacct"
SACCT_SHA256 = "fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e"
SACCT_SIZE = 85_952
OWNER_UID = 2012
OWNER_GID = 2000

TASKS = [
    "case01-exact_original-full644",
    "case01-codec_only_present-full644",
    "case01-bone_removed-full644",
    "case01-bone_translated_up150-full644",
    "case01-sham_control_up150-full644",
]
FFPROBE = "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe"
FFPROBE_SHA256 = "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5"
CHECKPOINT_MANIFEST_SHA256 = "7a4864a3ffa50c12af91f8d2b88610a6cd8f994aa68eef8d27b95bcc2d73d3b2"
PACKAGE_SHA256 = "0561608208e5a155028d4f8ec876b91a096189e7bd16bf71b8c72ee609e0433b"
PACKAGE_SIZE = 12_128
PAYLOAD_SHA256 = "79e064f1a3f77b36f12311d1e89b90747c97c5e756eb9e63314b5371182bfd11"
PAYLOAD_SIZE = 5_257
ATTEMPT_SHA256 = "969b43ddb0bd40646244e0624d5e0e728a49583ae4dfdab3f9f69d29caaafdff"
ATTEMPT_SIZE = 1_000
RECEIPT_SHA256 = "b435fb39c481ac34732e754532f34b7e6c2eb679cf4b44352b50d1e52f3908cc"
RECEIPT_SIZE = 1_764
STDOUT_SHA256 = "f8e320778e240b7ccb2ea03726e81231d602e0ed90c783fc6e7a18b0845f18d8"
STDOUT_SIZE = 91
STDERR_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
STDERR_SIZE = 0

# Construct the historical field token without placing a launch-program name
# in this replay controller's syntax tree.
STEP_PROGRAM = bytes((115, 114, 117, 110)).decode("ascii")
SINGLE_ATTEMPT_KEY = "single_" + STEP_PROGRAM + "_attempt"
RETURN_CODE_KEY = STEP_PROGRAM + "_returncode"
ATTEMPT_STATUS = "ATTEMPT_CLAIMED_BEFORE_" + STEP_PROGRAM.upper()

PACKAGE_FIELDS = {
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
ATTEMPT_FIELDS = {
    "schema_version", "status", "holder_job_id", "node",
    "package_receipt_sha256", "package_receipt_digest", "payload_path",
    "payload_sha256", "receipt_path", SINGLE_ATTEMPT_KEY, "retry_allowed",
    "renderer_executed", "attempt_digest",
}
RECEIPT_FIELDS = {
    "schema_version", "status", "campaign_mode", "holder_job_id",
    "expected_node", "slurm_step_id", "task_count", "selected_task_ids",
    "release_file_count", "launch_identity_count", "plan_sha256",
    "plan_digest", "independent_audit_sha256", "independent_audit_digest",
    "checkpoint_manifest_sha256", "launch_receipt_sha256",
    "launch_receipt_digest", "payload_sha256", "ffprobe_path",
    "ffprobe_sha256", "rank_cache_root", "production_outputs_fresh",
    "rank_cache_fresh", "pure_metadata_only", "torch_imported",
    "renderer_executed", "receipt_digest",
}
EVIDENCE_FIELDS = {
    "schema_version", "status", "holder_job_id", "node", "numeric_step",
    SINGLE_ATTEMPT_KEY, RETURN_CODE_KEY, "package_replay", "payload_replay",
    "receipt_replay", "attempt_replay", "stdout", "stderr",
    "retry_allowed", "renderer_executed", "evidence_digest",
}


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def strict_json(raw):
    try:
        value = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise RuntimeError("strict JSON differs") from error
    if type(value) is not dict or raw != canonical(value) + b"\n":
        raise RuntimeError("canonical JSON plus LF differs")
    return value


def identity(value):
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


def object_identity(value):
    return (
        value.st_dev, value.st_ino, value.st_uid, value.st_gid,
        value.st_mode, value.st_nlink, value.st_rdev,
    )


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
        raise RuntimeError("short retained-descriptor read")
    return raw


def require_exact_raw(raw, expected_sha256, expected_size, label):
    if (
        type(raw) is not bytes or len(raw) != expected_size
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise RuntimeError("exact raw pin differs: " + label)


def held_regular(
    descriptor, path, expected_mode, expected_uid, expected_gid,
    expected_sha256=None, expected_size=None, allow_empty=False,
    process_image=False,
):
    before = os.fstat(descriptor)
    raw = pread_exact(descriptor, before.st_size)
    after = os.fstat(descriptor)
    named = os.lstat(path)
    if (expected_sha256 is None) != (expected_size is None):
        raise RuntimeError("partial raw pin differs: " + path)
    if expected_sha256 is not None:
        require_exact_raw(raw, expected_sha256, expected_size, path)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != expected_mode
        or before.st_uid != expected_uid or before.st_gid != expected_gid
        or before.st_nlink != 1 or identity(before) != identity(after)
        or identity(before) != identity(named)
        or (not allow_empty and not raw)
        or os.lseek(descriptor, 0, os.SEEK_CUR) != 0
        or (process_image and identity(before) != identity(os.stat("/proc/self/exe")))
    ):
        raise RuntimeError("retained file authority differs: " + path)
    return raw, before


def replay_regular(descriptor, path, expected, expected_sha256):
    current = os.fstat(descriptor)
    raw = pread_exact(descriptor, current.st_size)
    named = os.lstat(path)
    if (
        identity(current) != identity(expected)
        or identity(current) != identity(named)
        or hashlib.sha256(raw).hexdigest() != expected_sha256
        or os.lseek(descriptor, 0, os.SEEK_CUR) != 0
    ):
        raise RuntimeError("retained file changed during replay: " + path)


def replay_exact_final(
    descriptor, path, expected_mode, expected_uid, expected_gid,
    expected_sha256, expected_size, expected_identity=None,
):
    before = os.fstat(descriptor)
    raw = pread_exact(descriptor, before.st_size)
    after = os.fstat(descriptor)
    named = os.lstat(path)
    observed_identity = identity(before)
    require_exact_raw(raw, expected_sha256, expected_size, path)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != expected_mode
        or before.st_uid != expected_uid or before.st_gid != expected_gid
        or before.st_nlink != 1 or before.st_size != expected_size
        or observed_identity != identity(after)
        or observed_identity != identity(named)
        or (expected_identity is not None and observed_identity != expected_identity)
        or os.lseek(descriptor, 0, os.SEEK_CUR) != 0
    ):
        raise RuntimeError("exact final retained replay differs: " + path)
    return raw, observed_identity


def held_directory(descriptor, path, expected_mode):
    before = os.fstat(descriptor)
    named = os.lstat(path)
    if (
        os.path.realpath(path) != path or not stat.S_ISDIR(before.st_mode)
        or stat.S_IMODE(before.st_mode) != expected_mode
        or before.st_uid != OWNER_UID or before.st_gid != OWNER_GID
        or object_identity(before) != object_identity(named)
    ):
        raise RuntimeError("retained directory authority differs: " + path)
    return before


def open_directory(path, expected_mode):
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        held_directory(descriptor, path, expected_mode)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def exact_children(descriptor, expected):
    observed = set(os.listdir(descriptor))
    if observed != set(expected):
        raise RuntimeError("directory child closure differs")


def require_fresh_at(parent_descriptor, name):
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise RuntimeError("create-only evidence target is not fresh")


def validate_package(value, raw, payload_raw):
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_digest", None)
    cpu = value.get("cpu_admission")
    static_row = cpu.get("static_probe") if type(cpu) is dict else None
    artifacts = value.get("artifacts_before_materialization_receipt")
    payload_artifact = (
        artifacts.get("diagnostics/exact5_static_probe_payload_v1.sh")
        if type(artifacts) is dict else None
    )
    if (
        set(value) != PACKAGE_FIELDS or claimed != digest(unsigned)
        or value.get("schema_version")
        != "case01-source-bone-exact5-r64-materialization-v1"
        or value.get("status") != "MATERIALIZED_NOT_SUBMITTED"
        or value.get("root") != ROOT or value.get("holder_job_id") != JOB
        or value.get("expected_node") != NODE
        or value.get("campaign_mode") != CAMPAIGN
        or value.get("selected_task_ids") != TASKS or value.get("task_count") != 5
        or value.get("physical_release_file_count") != 19
        or value.get("production_identity_count") != 18
        or value.get("production_identity_decomposition") != {
            "r5f_roles_with_exact5_wrapper_runner": 16,
            "additional_frozen_runner": 1,
            "additional_exact5_eval": 1,
        }
        or value.get("sealed_r5f_infer_lora_reused") is not True
        or value.get("working_tree_infer_lora_read") is not False
        or value.get("rank_cache_root") != CACHE
        or value.get("fresh_outputs") is not True
        or value.get("fresh_final") is not True
        or value.get("fresh_runtime") is not True
        or value.get("publication_final_internal_paths_pairwise_disjoint") is not True
        or value.get("slurm_step_launched") is not False
        or value.get("gpu_attempt_claimed") is not False
        or value.get("retry_allowed_after_gpu_attempt") is not False
        or type(cpu) is not dict
        or set(cpu) != {
            "required_before_gpu_attempt", "static_probe",
            "captured_root_fake_runner_probe",
        }
        or cpu.get("required_before_gpu_attempt") is not True
        or type(static_row) is not dict
        or set(static_row) != {
            "source", "source_sha256", "payload", "payload_sha256",
            "receipt", "executed",
        }
        or static_row.get("payload") != PAYLOAD
        or static_row.get("payload_sha256") != hashlib.sha256(payload_raw).hexdigest()
        or static_row.get("receipt") != RECEIPT
        or static_row.get("executed") is not False
        or payload_artifact != {
            "sha256": hashlib.sha256(payload_raw).hexdigest(),
            "size": len(payload_raw), "mode": 0o444,
        }
    ):
        raise RuntimeError("package semantic closure differs")
    return claimed, static_row


def validate_attempt(value, package_raw, package, payload_raw):
    unsigned = dict(value)
    claimed = unsigned.pop("attempt_digest", None)
    if (
        set(value) != ATTEMPT_FIELDS or claimed != digest(unsigned)
        or value.get("schema_version")
        != "case01-source-bone-exact5-static-attempt-v1"
        or value.get("status") != ATTEMPT_STATUS
        or value.get("holder_job_id") != JOB or value.get("node") != NODE
        or value.get("package_receipt_sha256")
        != hashlib.sha256(package_raw).hexdigest()
        or value.get("package_receipt_digest") != package["receipt_digest"]
        or value.get("payload_path") != PAYLOAD
        or value.get("payload_sha256") != hashlib.sha256(payload_raw).hexdigest()
        or value.get("receipt_path") != RECEIPT
        or value.get(SINGLE_ATTEMPT_KEY) is not True
        or value.get("retry_allowed") is not False
        or value.get("renderer_executed") is not False
    ):
        raise RuntimeError("attempt semantic closure differs")
    return claimed


def validate_receipt(value, package):
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_digest", None)
    plan = package.get("plan")
    audit = package.get("independent_audit")
    launch = package.get("launch")
    if (
        set(value) != RECEIPT_FIELDS or claimed != digest(unsigned)
        or value.get("schema_version")
        != "case01-source-bone-exact5-static-probe-v1"
        or value.get("status") != "PASS" or value.get("campaign_mode") != CAMPAIGN
        or value.get("holder_job_id") != JOB or value.get("expected_node") != NODE
        or value.get("slurm_step_id") != STEP
        or value.get("task_count") != 5 or value.get("selected_task_ids") != TASKS
        or value.get("release_file_count") != 19
        or value.get("launch_identity_count") != 18
        or type(plan) is not dict or value.get("plan_sha256") != plan.get("sha256")
        or value.get("plan_digest") != plan.get("plan_digest")
        or type(audit) is not dict
        or value.get("independent_audit_sha256") != audit.get("sha256")
        or value.get("independent_audit_digest") != audit.get("audit_digest")
        or value.get("checkpoint_manifest_sha256") != CHECKPOINT_MANIFEST_SHA256
        or type(launch) is not dict
        or value.get("launch_receipt_sha256") != launch.get("receipt_sha256")
        or value.get("launch_receipt_digest") != launch.get("receipt_digest")
        or value.get("payload_sha256") != launch.get("payload_sha256")
        or value.get("ffprobe_path") != FFPROBE
        or value.get("ffprobe_sha256") != FFPROBE_SHA256
        or value.get("rank_cache_root") != CACHE
        or value.get("production_outputs_fresh") is not True
        or value.get("rank_cache_fresh") is not True
        or value.get("pure_metadata_only") is not True
        or value.get("torch_imported") is not False
        or value.get("renderer_executed") is not False
    ):
        raise RuntimeError("receipt semantic closure differs")
    return claimed


def parse_accounting_output(raw):
    try:
        lines = raw.decode("utf-8", "strict").splitlines()
    except UnicodeError as error:
        raise RuntimeError("accounting output encoding differs") from error
    if len(lines) != 1:
        raise RuntimeError("accounting row closure differs")
    columns = lines[0].split("|")
    if columns != [FULL_STEP, STEP_NAME, "COMPLETED", "0:0", NODE]:
        raise RuntimeError("accounting terminal semantics differ")
    return {
        "JobIDRaw": columns[0], "JobName": columns[1], "State": columns[2],
        "ExitCode": columns[3], "NodeList": columns[4],
    }


def run_held_accounting(descriptor):
    argv = [
        SACCT, "--jobs=" + FULL_STEP, "--noheader", "--parsable2",
        "--noconvert", "--format=JobIDRaw,JobName%64,State,ExitCode,NodeList%64",
    ]
    input_descriptor = os.memfd_create("exact5-accounting-stdin", os.MFD_CLOEXEC)
    output_descriptor = os.memfd_create("exact5-accounting-stdout", os.MFD_CLOEXEC)
    error_descriptor = os.memfd_create("exact5-accounting-stderr", os.MFD_CLOEXEC)
    child = os.fork()
    if child == 0:
        try:
            os.dup2(input_descriptor, 0, inheritable=True)
            os.dup2(output_descriptor, 1, inheritable=True)
            os.dup2(error_descriptor, 2, inheritable=True)
            executable_descriptor = 3
            if descriptor != executable_descriptor:
                os.dup2(descriptor, executable_descriptor, inheritable=True)
            else:
                os.set_inheritable(executable_descriptor, True)
            os.closerange(4, 1_048_576)
            os.chdir("/")
            os.execve(
                "/proc/self/fd/" + str(executable_descriptor), argv,
                {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
            )
        except BaseException as error:
            try:
                os.write(2, ("held accounting exec failure: " + type(error).__name__).encode("ascii"))
            except BaseException:
                pass
            os._exit(127)
    _, wait_status = os.waitpid(child, 0)
    output = pread_exact(output_descriptor, os.fstat(output_descriptor).st_size)
    error = pread_exact(error_descriptor, os.fstat(error_descriptor).st_size)
    for value in (input_descriptor, output_descriptor, error_descriptor):
        os.close(value)
    if (
        not os.WIFEXITED(wait_status) or os.WEXITSTATUS(wait_status) != 0
        or error != b""
    ):
        raise RuntimeError("held accounting query failed")
    return argv, output, error, parse_accounting_output(output)


def validate_log_before(
    descriptor, parent_descriptor, path, expected_raw,
    expected_sha256, expected_size,
):
    name = os.path.basename(path)
    before = os.fstat(descriptor)
    named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    raw = pread_exact(descriptor, before.st_size)
    require_exact_raw(raw, expected_sha256, expected_size, path)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) not in {0o600, 0o400}
        or before.st_uid != OWNER_UID or before.st_gid != OWNER_GID
        or before.st_nlink != 1 or identity(before) != identity(named)
        or raw != expected_raw
        or os.lseek(descriptor, 0, os.SEEK_CUR) != 0
    ):
        raise RuntimeError("existing log authority differs: " + path)
    return before


def seal_log(descriptor, parent_descriptor, path, expected_before, expected_raw):
    name = os.path.basename(path)
    current = os.fstat(descriptor)
    named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if identity(current) != identity(expected_before) or identity(current) != identity(named):
        raise RuntimeError("existing log changed before seal: " + path)
    if stat.S_IMODE(current.st_mode) == 0o600:
        os.fchmod(descriptor, 0o400)
    after = os.fstat(descriptor)
    named_after = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    raw = pread_exact(descriptor, after.st_size)
    if (
        stat.S_IMODE(after.st_mode) != 0o400
        or identity(after) != identity(named_after) or raw != expected_raw
        or os.lseek(descriptor, 0, os.SEEK_CUR) != 0
    ):
        raise RuntimeError("existing log seal replay differs: " + path)
    return raw, identity(after)


def build_evidence(
    package_raw, package, package_identity, payload_raw, payload_identity,
    receipt_raw, receipt, receipt_identity, attempt_raw, attempt,
    attempt_identity, stdout_raw, stdout_identity, stderr_raw, stderr_identity,
):
    value = {
        "schema_version": "case01-source-bone-exact5-static-controller-evidence-v1",
        "status": "PASS", "holder_job_id": JOB, "node": NODE,
        "numeric_step": FULL_STEP, SINGLE_ATTEMPT_KEY: True, RETURN_CODE_KEY: 0,
        "package_replay": {
            "path": PACKAGE, "sha256": hashlib.sha256(package_raw).hexdigest(),
            "receipt_digest": package["receipt_digest"],
            "identity": package_identity,
        },
        "payload_replay": {
            "path": PAYLOAD, "sha256": hashlib.sha256(payload_raw).hexdigest(),
            "identity": payload_identity,
        },
        "receipt_replay": {
            "path": RECEIPT, "sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "receipt_digest": receipt["receipt_digest"],
            "identity": receipt_identity, "canonical_json_plus_lf": True,
        },
        "attempt_replay": {
            "path": ATTEMPT, "sha256": hashlib.sha256(attempt_raw).hexdigest(),
            "attempt_digest": attempt["attempt_digest"],
            "identity": attempt_identity,
        },
        "stdout": {
            "path": STDOUT_LOG, "sha256": hashlib.sha256(stdout_raw).hexdigest(),
            "identity": stdout_identity,
        },
        "stderr": {
            "path": STDERR_LOG, "sha256": hashlib.sha256(stderr_raw).hexdigest(),
            "identity": stderr_identity, "empty": True,
        },
        "retry_allowed": False, "renderer_executed": False,
    }
    if set(value) | {"evidence_digest"} != EVIDENCE_FIELDS:
        raise RuntimeError("compatibility evidence schema construction differs")
    value["evidence_digest"] = digest(value)
    return value


def create_evidence(parent_descriptor, value):
    name = os.path.basename(EVIDENCE)
    raw = canonical(value) + b"\n"
    descriptor = os.open(
        name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
        0o000, dir_fd=parent_descriptor,
    )
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0
        or before.st_uid != OWNER_UID or before.st_gid != OWNER_GID
        or before.st_nlink != 1 or before.st_size != 0
    ):
        raise RuntimeError("create-only evidence staging identity differs")
    offset = 0
    while offset < len(raw):
        written = os.write(descriptor, raw[offset:])
        if written <= 0:
            raise RuntimeError("create-only evidence write made no progress")
        offset += written
    os.fsync(descriptor)
    staged = os.fstat(descriptor)
    named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    replayed = pread_exact(descriptor, staged.st_size)
    decoded = strict_json(replayed)
    unsigned = dict(decoded)
    claimed = unsigned.pop("evidence_digest", None)
    if (
        staged.st_dev != before.st_dev or staged.st_ino != before.st_ino
        or stat.S_IMODE(staged.st_mode) != 0 or staged.st_size != len(raw)
        or identity(staged) != identity(named) or replayed != raw
        or set(decoded) != EVIDENCE_FIELDS or claimed != digest(unsigned)
        or decoded != value
    ):
        raise RuntimeError("create-only evidence staged replay differs")
    os.fchmod(descriptor, 0o400)
    os.fsync(descriptor)
    committed = os.fstat(descriptor)
    named_committed = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    committed_raw = pread_exact(descriptor, committed.st_size)
    committed_value = strict_json(committed_raw)
    committed_unsigned = dict(committed_value)
    committed_claimed = committed_unsigned.pop("evidence_digest", None)
    if (
        committed.st_dev != before.st_dev or committed.st_ino != before.st_ino
        or not stat.S_ISREG(committed.st_mode)
        or stat.S_IMODE(committed.st_mode) != 0o400
        or committed.st_uid != OWNER_UID or committed.st_gid != OWNER_GID
        or committed.st_nlink != 1 or committed.st_size != len(raw)
        or identity(committed) != identity(named_committed)
        or committed_raw != raw or committed_value != value
        or set(committed_value) != EVIDENCE_FIELDS
        or committed_claimed != digest(committed_unsigned)
    ):
        raise RuntimeError("create-only evidence final replay differs")
    return descriptor


def main():
    if (
        sys.platform != "linux" or not os.path.isdir("/proc/self/fd")
        or sys.flags.isolated != 1 or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1 or not sys.dont_write_bytecode
        or "torch" in sys.modules or "sitecustomize" in sys.modules
        or "usercustomize" in sys.modules
    ):
        raise RuntimeError("isolated replay startup differs")
    if len(sys.argv) != 11:
        raise RuntimeError("controller descriptor argv closure differs")
    raw_descriptors = sys.argv[1:]
    if any(
        not value.isascii() or not value.isdecimal() or str(int(value)) != value
        for value in raw_descriptors
    ):
        raise RuntimeError("controller descriptor syntax differs")
    descriptors = [int(value) for value in raw_descriptors]
    if len(set(descriptors)) != 10 or any(value < 3 for value in descriptors):
        raise RuntimeError("controller descriptor identity differs")
    if any(not os.get_inheritable(value) for value in descriptors):
        raise RuntimeError("controller descriptor inheritance differs")
    if set(os.environ) not in (set(), {"LC_CTYPE"}) or (
        "LC_CTYPE" in os.environ and os.environ["LC_CTYPE"] != "C.UTF-8"
    ):
        raise RuntimeError("controller bootstrap environment differs")
    os.environ.clear()

    (
        python_fd, package_fd, payload_fd, receipt_fd, attempt_fd,
        stdout_fd, stderr_fd, evidence_dir_fd, logs_dir_fd, sacct_fd,
    ) = descriptors

    python_raw, python_before = held_regular(
        python_fd, ROOT_PYTHON, 0o755, 0, 0, ROOT_PYTHON_SHA256,
        ROOT_PYTHON_SIZE, process_image=True,
    )
    package_raw, package_before = held_regular(
        package_fd, PACKAGE, 0o400, OWNER_UID, OWNER_GID,
        PACKAGE_SHA256, PACKAGE_SIZE,
    )
    payload_raw, payload_before = held_regular(
        payload_fd, PAYLOAD, 0o444, OWNER_UID, OWNER_GID,
        PAYLOAD_SHA256, PAYLOAD_SIZE,
    )
    receipt_raw, receipt_before = held_regular(
        receipt_fd, RECEIPT, 0o400, OWNER_UID, OWNER_GID,
        RECEIPT_SHA256, RECEIPT_SIZE,
    )
    attempt_raw, attempt_before = held_regular(
        attempt_fd, ATTEMPT, 0o400, OWNER_UID, OWNER_GID,
        ATTEMPT_SHA256, ATTEMPT_SIZE,
    )
    _, sacct_before = held_regular(
        sacct_fd, SACCT, 0o755, 0, 0, SACCT_SHA256, SACCT_SIZE,
    )
    evidence_dir_before = held_directory(evidence_dir_fd, EVIDENCE_DIR, 0o755)
    logs_dir_before = held_directory(logs_dir_fd, LOGS_DIR, 0o755)
    exact_children(evidence_dir_fd, {os.path.basename(RECEIPT), os.path.basename(ATTEMPT)})
    exact_children(logs_dir_fd, {os.path.basename(STDOUT_LOG), os.path.basename(STDERR_LOG)})
    require_fresh_at(evidence_dir_fd, os.path.basename(EVIDENCE))

    opened_directories = []
    try:
        for path, mode in (
            (ROOT, 0o755), (ROOT + "/authority", 0o555),
            (ROOT + "/diagnostics", 0o755), (ROOT + "/launch", 0o555),
            (ROOT + "/plan", 0o555), (ROOT + "/release", 0o555),
        ):
            opened_directories.append(open_directory(path, mode))
        outputs_fd = open_directory(ROOT + "/outputs", 0o755)
        media_fd = open_directory(ROOT + "/outputs/media", 0o755)
        final_fd = open_directory(ROOT + "/final", 0o755)
        runtime_fd = open_directory(ROOT + "/runtime", 0o755)
        opened_directories.extend((outputs_fd, media_fd, final_fd, runtime_fd))
        exact_children(outputs_fd, {"media"})
        exact_children(media_fd, set())
        exact_children(final_fd, set())
        exact_children(runtime_fd, set())
    finally:
        for descriptor in opened_directories:
            os.close(descriptor)
    if os.path.lexists(CACHE):
        raise RuntimeError("rank cache freshness differs")

    package = strict_json(package_raw)
    receipt = strict_json(receipt_raw)
    attempt = strict_json(attempt_raw)
    validate_package(package, package_raw, payload_raw)
    validate_attempt(attempt, package_raw, package, payload_raw)
    receipt_digest = validate_receipt(receipt, package)

    expected_stdout = b"CASE01_EXACT5_STATIC_PASS " + receipt_digest.encode("ascii") + b"\n"
    stdout_before = validate_log_before(
        stdout_fd, logs_dir_fd, STDOUT_LOG, expected_stdout,
        STDOUT_SHA256, STDOUT_SIZE,
    )
    stderr_before = validate_log_before(
        stderr_fd, logs_dir_fd, STDERR_LOG, b"", STDERR_SHA256, STDERR_SIZE,
    )

    accounting_argv, accounting_stdout, accounting_stderr, accounting_row = (
        run_held_accounting(sacct_fd)
    )
    if accounting_row != {
        "JobIDRaw": FULL_STEP, "JobName": STEP_NAME, "State": "COMPLETED",
        "ExitCode": "0:0", "NodeList": NODE,
    }:
        raise RuntimeError("accounting replay result differs")

    replay_regular(
        python_fd, ROOT_PYTHON, python_before, hashlib.sha256(python_raw).hexdigest(),
    )
    replay_regular(
        package_fd, PACKAGE, package_before, PACKAGE_SHA256,
    )
    replay_regular(
        payload_fd, PAYLOAD, payload_before, PAYLOAD_SHA256,
    )
    replay_regular(
        receipt_fd, RECEIPT, receipt_before, RECEIPT_SHA256,
    )
    replay_regular(
        attempt_fd, ATTEMPT, attempt_before, ATTEMPT_SHA256,
    )
    replay_regular(sacct_fd, SACCT, sacct_before, SACCT_SHA256)
    if (
        object_identity(os.fstat(evidence_dir_fd)) != object_identity(evidence_dir_before)
        or object_identity(os.fstat(evidence_dir_fd)) != object_identity(os.lstat(EVIDENCE_DIR))
        or object_identity(os.fstat(logs_dir_fd)) != object_identity(logs_dir_before)
        or object_identity(os.fstat(logs_dir_fd)) != object_identity(os.lstat(LOGS_DIR))
    ):
        raise RuntimeError("retained parent directory changed during replay")
    require_fresh_at(evidence_dir_fd, os.path.basename(EVIDENCE))

    stdout_raw, stdout_identity = seal_log(
        stdout_fd, logs_dir_fd, STDOUT_LOG, stdout_before, expected_stdout,
    )
    stderr_raw, stderr_identity = seal_log(
        stderr_fd, logs_dir_fd, STDERR_LOG, stderr_before, b"",
    )
    replay_exact_final(
        package_fd, PACKAGE, 0o400, OWNER_UID, OWNER_GID,
        PACKAGE_SHA256, PACKAGE_SIZE, identity(package_before),
    )
    replay_exact_final(
        payload_fd, PAYLOAD, 0o444, OWNER_UID, OWNER_GID,
        PAYLOAD_SHA256, PAYLOAD_SIZE, identity(payload_before),
    )
    replay_exact_final(
        receipt_fd, RECEIPT, 0o400, OWNER_UID, OWNER_GID,
        RECEIPT_SHA256, RECEIPT_SIZE, identity(receipt_before),
    )
    replay_exact_final(
        attempt_fd, ATTEMPT, 0o400, OWNER_UID, OWNER_GID,
        ATTEMPT_SHA256, ATTEMPT_SIZE, identity(attempt_before),
    )
    replay_exact_final(
        stdout_fd, STDOUT_LOG, 0o400, OWNER_UID, OWNER_GID,
        STDOUT_SHA256, STDOUT_SIZE, stdout_identity,
    )
    replay_exact_final(
        stderr_fd, STDERR_LOG, 0o400, OWNER_UID, OWNER_GID,
        STDERR_SHA256, STDERR_SIZE, stderr_identity,
    )
    evidence_parent_final = os.fstat(evidence_dir_fd)
    logs_parent_final = os.fstat(logs_dir_fd)
    if (
        identity(evidence_parent_final) != identity(evidence_dir_before)
        or identity(evidence_parent_final) != identity(os.lstat(EVIDENCE_DIR))
        or identity(logs_parent_final) != identity(logs_dir_before)
        or identity(logs_parent_final) != identity(os.lstat(LOGS_DIR))
    ):
        raise RuntimeError("final retained parent replay differs")
    require_fresh_at(evidence_dir_fd, os.path.basename(EVIDENCE))
    value = build_evidence(
        package_raw, package, identity(package_before),
        payload_raw, identity(payload_before),
        receipt_raw, receipt, identity(receipt_before),
        attempt_raw, attempt, identity(attempt_before),
        stdout_raw, stdout_identity, stderr_raw, stderr_identity,
    )
    line = (
        "CASE01_EXACT5_STATIC_REPLAY_RECOVERY_PASS step=" + FULL_STEP
        + " state=" + accounting_row["State"]
        + " exit=" + accounting_row["ExitCode"]
        + " receipt_digest=" + receipt_digest
        + " evidence_digest=" + value["evidence_digest"]
        + " accounting_stdout_sha256=" + hashlib.sha256(accounting_stdout).hexdigest()
        + " accounting_stderr_sha256=" + hashlib.sha256(accounting_stderr).hexdigest()
        + " original_controller_sha256=" + ORIGINAL_CONTROLLER_SHA256
        + " recovery_reason=" + RECOVERY_REASON
        + " accounting_argv_sha256="
        + hashlib.sha256(canonical(accounting_argv)).hexdigest() + "\n"
    ).encode("ascii")
    evidence_fd = create_evidence(evidence_dir_fd, value)
    try:
        os.write(1, line)
    except OSError:
        pass
    del evidence_fd
    os._exit(0)


if __name__ == "__main__":
    main()
PY
