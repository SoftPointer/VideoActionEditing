#!/bin/bash -p
# BOX-EXP-013 r6: two arms incomplete-only calls on retained 136140/gpu215.
# The passing action clips are read-only external authority.  This launcher
# creates no diagnostic, Q, a_min, training, or optimizer state.

[[ $- == *p* ]] || { builtin printf '%s\n' '[arms-incomplete-exact2-r6] ERROR: privileged canonical Bash entry is required' >&2; builtin exit 126; }
# Privileged Bash ignores BASH_ENV/ENV, imported functions, SHELLOPTS,
# BASHOPTS, CDPATH, and GLOBIGNORE before the first script line.  Replace the
# caller PATH using shell builtins before any external authority command.
export PATH=/usr/bin:/bin
export LC_ALL=C LANG=C
unset BASH_ENV ENV CDPATH GLOBIGNORE
IFS=$' \t\n'
while IFS= read -r inherited_environment_name; do
  case "${inherited_environment_name}" in
    LD_*|DYLD_*|GLIBC_*|GCONV_PATH|LOCPATH|MALLOC_*|BASH_FUNC_*|SHELLOPTS|BASHOPTS)
      builtin printf '%s\n' "[arms-incomplete-exact2-r6] ERROR: unsafe process-loader/shell environment is forbidden: ${inherited_environment_name}" >&2
      builtin exit 126
      ;;
  esac
done < <(compgen -e)
[[ "${HOME:-}" == /vast/users/guangyi.chen && "${USER:-}" == guangyi.chen && "${LOGNAME:-}" == guangyi.chen ]] || {
  builtin printf '%s\n' '[arms-incomplete-exact2-r6] ERROR: canonical user environment differs' >&2
  builtin exit 126
}
set -Eeuo pipefail
umask 077

fail() { echo "[arms-incomplete-exact2-r6] ERROR: $*" >&2; exit 2; }
fail_status() {
  local status="$1"
  shift
  [[ "${status}" =~ ^[0-9]+$ ]] && (( status >= 1 && status <= 255 )) || status=2
  echo "[arms-incomplete-exact2-r6] ERROR: $* (status=${status})" >&2
  exit "${status}"
}
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

readonly confirm="${F13_CONFIRM:?explicit launch confirmation required}"
readonly entry_environment="${F13_ENTRY_ENVIRONMENT:?explicit clean-entry environment authority required}"
readonly run_root="${F13_RUN_ROOT:?set fresh run root}"
readonly release_root="${F13_RELEASE_ROOT:?set exact four-entry release root}"
readonly method_root="${F13_METHOD_ROOT:?set extracted release method root}"
readonly method_archive="${F13_METHOD_ARCHIVE:?set release archive}"
readonly method_manifest="${F13_METHOD_MANIFEST:?set release manifest}"
readonly deployment_envelope="${F13_DEPLOYMENT_ENVELOPE:?set deployment envelope}"
readonly deployment_envelope_sha="${F13_DEPLOYMENT_ENVELOPE_SHA256:?external live preflight must pin envelope SHA}"
readonly expected_detached_launcher_sha="${F13_DETACHED_LAUNCHER_SHA256:?external live preflight must pin detached launcher SHA}"
readonly seed1_spec="${F13_SEED1_SPEC:?set seed1 reserve4 spec}"
readonly seed2_spec="${F13_SEED2_SPEC:?set seed2 reserve4 spec}"
readonly external_root="${F13_EXTERNAL_EVIDENCE_ROOT:?stage sealed BOX-EXP-011 arms4 evidence}"
readonly external_key="${F13_EXTERNAL_KEY:?set byte-original sealed key}"
readonly external_review="${F13_EXTERNAL_REVIEW_RECEIPT:?set blind reviewer receipt}"
readonly python_bin="${F13_PYTHON_BIN:?set frozen Python}"
readonly frozen_python_path=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly frozen_python_realpath=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly frozen_python_file_type='regular file'
readonly frozen_python_mode=755
readonly frozen_python_uid=2012
readonly frozen_python_size=31490256
readonly frozen_python_nlink=1
readonly frozen_python_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly rejected_python_symlink=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python
readonly root_bootstrap_python=/usr/bin/python3.10
readonly root_bootstrap_python_sha=11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96
readonly root_bootstrap_python_size=5937800
readonly srun_bin=/usr/bin/srun
readonly srun_bin_sha=2b8f60b30edf7efed35bb00864651da1b0bec68e75f942ce58b5ff82bc43cd9e
readonly srun_bin_size=164720
readonly ffprobe_bin=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe
readonly ffprobe_sha=356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5
readonly bernini_root="${F13_BERNINI_ROOT:?set Bernini root}"
readonly veomni_root="${F13_VEOMNI_ROOT:?set VeOmni root}"
readonly checkpoint="${F13_CHECKPOINT:?set Bernini checkpoint}"
readonly checkpoint_manifest="${F13_CHECKPOINT_MANIFEST:?set checkpoint manifest}"
readonly master_port="${F13_MASTER_PORT:?set one port}"
readonly r10_receipt="${F13_R10_COMPILE_SMOKE_RECEIPT:?bind r10 receipt}"
readonly r10_receipt_sha="${F13_R10_COMPILE_SMOKE_RECEIPT_SHA256:?pin r10 receipt}"
readonly r10_log="${F13_R10_GENERATION_LOG:?bind r10 log}"
readonly r10_log_sha="${F13_R10_GENERATION_LOG_SHA256:?pin r10 log}"

readonly holder_job=136140
readonly holder_node=auh7-1b-gpu-215
readonly holder_login_host=auh-1b-cpu-login-002
readonly launch_confirmation=launch-approved-BOX-EXP-013-arms-incomplete-repair-exact2-r6-136140
readonly revoked_release_leaf=full30-action-arms-incomplete-exact2-rfinal-96c4095e-a59fb218
readonly revoked_run_leaf=full30-action-arms-incomplete-exact2-rfinal-96c4095e-j136140-r1
readonly revoked_live_log_sha=6df1462415b72bbf966f8b41c125932e7fcd39353c58a8172121577c41f9285a
readonly revoked_run_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/data_prep/full30-action-arms-incomplete-exact2-rfinal-96c4095e-j136140-r1
readonly revoked_portable_r2_archive_sha=dc945cc658934050501d327d5f81afb97de93f43bd39a2e00472e515f502484c
readonly revoked_portable_r2_manifest_sha=5a9f069bfc1f1452e12f11d81fdf61a43ad8f5a716b2a166126cac54bfef6d83
readonly revoked_terminal_r3_archive_sha=ce1a5ff5cd8ed4458e2a704ea3850a08f57630d8d7f8b9139c02fdbe0701f5fb
readonly revoked_terminal_r3_manifest_sha=e146c3c426980b9e299514036af58fa9675d43a3b466c1fe78076401d40b6f6c
readonly revoked_terminal_r3_launcher_sha=3917a978ec4d6c42da139349b3553e9f18606ad27e8c9dfa8f3ce9869d6c30de
readonly revoked_terminal_r3_envelope_sha=5412ed07234b7e60eeb68d46f8dbf9271b16a95c03837e56cd71ad5cf1f7b41a
readonly revoked_resource_reuse_r4_archive_sha=29b2f5002363673b3957bfcc8135859bfac304434a590220919fcbc1051f0a23
readonly revoked_resource_reuse_r4_manifest_sha=fafd3d250f4b489cc5be5451279957aa95fd9607528911c4e888a8e2f3984330
readonly revoked_resource_reuse_r4_launcher_sha=8524f21373476147f905c081339f6281ebfc7ebfc38cbbb641faa542d7b420d6
readonly revoked_resource_reuse_r4_envelope_sha=31c6c949403cf3816176ecbff8da95686a717f35eb8e5e93e9d3373c3ad77b24
readonly revoked_resource_reuse_r4_release_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/releases/full30-action-arms-incomplete-exact2-r4-233bb4be-29b2f500
readonly revoked_resource_reuse_r4_materialization_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/release_materializations/full30-action-arms-incomplete-exact2-r4-233bb4be-29b2f500
readonly revoked_resource_reuse_r4_run_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/data_prep/full30-action-arms-incomplete-exact2-r4-233bb4be-j136140-r1
# r5 was authorized but failed closed before any run root, numbered child, or GPU
# effect.  Its bytes and all three proposed remote namespaces remain permanent
# NO-GO even if copied, renamed, nested, or presented under a later approval.
readonly revoked_canonical_python_r5_archive_sha=3db741464fa8e5bd258d4d0b7a3f90c5ee9c9eeb78695cab094dcea919fb94b5
readonly revoked_canonical_python_r5_manifest_sha=b0129020fd4134e50e6840a2fdf8e61d0cd32f267bc2dadda8ba17e241d92208
readonly revoked_canonical_python_r5_launcher_sha=018ddaf9f4ab8c423dd8d081fd7303139d3256eb32c815f8c9d4a8494b4f2e6d
readonly revoked_canonical_python_r5_envelope_sha=517e31a381fb7e8fe626a1cc71829e59aea4cbe62febb4a68aeb8f39b49c3154
readonly revoked_canonical_python_r5_release_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/releases/full30-action-arms-incomplete-exact2-r5-3b59480b-3db74146
readonly revoked_canonical_python_r5_materialization_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/release_materializations/full30-action-arms-incomplete-exact2-r5-3b59480b-3db74146
readonly revoked_canonical_python_r5_run_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/data_prep/full30-action-arms-incomplete-exact2-r5-3b59480b-j136140-r1
readonly core_archive_sha=48e3ebc03a1912f5630237f43444924d4b2382f4701c3252db5149f5dc39201b
readonly core_manifest_sha=2a5a830e2801e7786a6209c3c99a0582a7d72fa7195e8ee6573bc84d107a4590
readonly core_manifest_digest=b904a4ed2230b1a7ca7fc09e43d1ddb40b9299918cb2460f3ed8a4eb1b258a37
readonly method_revision=9773201f8b89f1c413b903f70c568a4ba4f7c179
readonly owned_plan_sha=84451d103dbf9ddf72d9b3fff65a95b8aa55fdf9a032372fe4b9ce19238bfc26
readonly owned_generator_sha=8021fe6f6393b98ac7c1bd5d30c2b3b8cadd70ad820c301ba31ea14109fc8129
readonly owned_controller_sha=d7873ca59ef2a9992dcf15eee1d026186438b9b043963027a285b1d511de1d14
readonly owned_builder_sha=1c947066f21c77560f9e03425cc52d50ed5f3d5148e94c28429f48154b78b1a1
readonly owned_rank_wrapper_sha=e01c91aae3a602a373b8f8d3206c3bfe38933526ce32e8fece33f5ca79eae863

case "${1:-}" in
  "") readonly role=parent ;;
  __child) [[ $# == 1 ]] || fail "child takes no extra arguments"; readonly role=child ;;
  *) fail "launcher arguments differ" ;;
esac

# The parent deliberately scrubs these four names at the srun boundary.  A
# direct/replayed child invocation carrying even an explicitly empty value is
# rejected before any unset, scratch preparation, or other compute effect.
if [[ "${role}" == child ]]; then
  for caller_scratch_name in SLURM_TMPDIR TMPDIR GADP_NODE_LOCAL_SCRATCH GADP_NODE_LOCAL_SCRATCH_FSTYPE; do
    [[ ! -v "${caller_scratch_name}" ]] || fail "caller scratch environment presence is forbidden, including empty values: ${caller_scratch_name}"
  done
fi

[[ "${confirm}" == "${launch_confirmation}" ]] || fail "launch confirmation differs"
[[ "${entry_environment}" == clean-env-i-bash-p-v1 ]] || fail "clean-entry environment authority differs"
[[ "${master_port}" =~ ^[0-9]+$ ]] && (( master_port >= 1024 && master_port <= 65535 )) || fail "port differs"
[[ "${method_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "method revision differs"
for digest in core_archive_sha core_manifest_sha core_manifest_digest owned_plan_sha owned_generator_sha owned_controller_sha owned_builder_sha owned_rank_wrapper_sha deployment_envelope_sha expected_detached_launcher_sha r10_receipt_sha r10_log_sha; do
  [[ "${!digest}" =~ ^[0-9a-f]{64}$ ]] || fail "${digest} is not SHA-256"
done
[[ "${expected_detached_launcher_sha}" != "${revoked_terminal_r3_launcher_sha}" && "${deployment_envelope_sha}" != "${revoked_terminal_r3_envelope_sha}" ]] || fail "revoked terminal-physical r3 launcher/envelope is permanent NO-GO"
[[ "${expected_detached_launcher_sha}" != "${revoked_resource_reuse_r4_launcher_sha}" && "${deployment_envelope_sha}" != "${revoked_resource_reuse_r4_envelope_sha}" ]] || fail "revoked resource-reuse r4 launcher/envelope is permanent NO-GO"
[[ "${expected_detached_launcher_sha}" != "${revoked_canonical_python_r5_launcher_sha}" && "${deployment_envelope_sha}" != "${revoked_canonical_python_r5_envelope_sha}" ]] || fail "revoked canonical-Python r5 launcher/envelope is permanent NO-GO"
for name in run_root release_root method_root method_archive method_manifest deployment_envelope seed1_spec seed2_spec external_root external_key external_review python_bin bernini_root veomni_root checkpoint checkpoint_manifest r10_receipt r10_log; do
  value="${!name}"
  [[ "${value}" == /* && "${value}" != / ]] || fail "${name} must be scoped and absolute"
done

deny_revoked_deployed_path() {
  local label="$1" value="$2" canonical
  canonical="$(readlink -m -- "${value}")" || fail "${label} cannot be canonicalized"
  [[ "${canonical}" == "${value}" ]] || fail "${label} is not a canonical path"
  case "${canonical}" in
    "${revoked_resource_reuse_r4_release_root}"|"${revoked_resource_reuse_r4_release_root}"/*|\
    "${revoked_resource_reuse_r4_materialization_root}"|"${revoked_resource_reuse_r4_materialization_root}"/*|\
    "${revoked_resource_reuse_r4_run_root}"|"${revoked_resource_reuse_r4_run_root}"/*)
      fail "revoked resource-reuse r4 release/materialization/run subtree is permanent NO-GO"
      ;;
    "${revoked_canonical_python_r5_release_root}"|"${revoked_canonical_python_r5_release_root}"/*|\
    "${revoked_canonical_python_r5_materialization_root}"|"${revoked_canonical_python_r5_materialization_root}"/*|\
    "${revoked_canonical_python_r5_run_root}"|"${revoked_canonical_python_r5_run_root}"/*)
      fail "revoked canonical-Python r5 release/materialization/run subtree is permanent NO-GO"
      ;;
  esac
}

validate_frozen_python_before_run_root_or_srun() {
  local observed_realpath observed_metadata observed_sha bootstrap_realpath bootstrap_metadata bootstrap_sha
  [[ -f "${root_bootstrap_python}" && ! -L "${root_bootstrap_python}" ]] || fail "root-owned bootstrap Python is not a canonical non-symlink regular file"
  bootstrap_realpath="$(readlink -f -- "${root_bootstrap_python}")" || fail "root-owned bootstrap Python realpath is unavailable"
  [[ "${bootstrap_realpath}" == "${root_bootstrap_python}" ]] || fail "root-owned bootstrap Python realpath differs"
  bootstrap_metadata="$(stat -c '%F|%a|%u|%g|%s|%h' -- "${root_bootstrap_python}")" || fail "root-owned bootstrap Python metadata is unavailable"
  [[ "${bootstrap_metadata}" == "regular file|755|0|0|${root_bootstrap_python_size}|1" ]] || fail "root-owned bootstrap Python type/mode/owner/size/link-count differs"
  bootstrap_sha="$(sha256_file "${root_bootstrap_python}")" || fail "root-owned bootstrap Python SHA-256 is unavailable"
  [[ "${bootstrap_sha}" == "${root_bootstrap_python_sha}" ]] || fail "root-owned bootstrap Python SHA-256 differs"
  [[ "${python_bin}" == "${frozen_python_path}" ]] || fail "alternate Python interpreter is forbidden (including ${rejected_python_symlink})"
  [[ -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "frozen Python is not a canonical non-symlink regular file"
  observed_realpath="$(readlink -f -- "${python_bin}")" || fail "frozen Python realpath is unavailable"
  [[ "${observed_realpath}" == "${frozen_python_realpath}" && "${observed_realpath}" == "${python_bin}" ]] || fail "frozen Python realpath differs"
  observed_metadata="$(stat -c '%F|%a|%u|%s|%h' -- "${python_bin}")" || fail "frozen Python metadata is unavailable"
  [[ "${observed_metadata}" == "${frozen_python_file_type}|${frozen_python_mode}|${frozen_python_uid}|${frozen_python_size}|${frozen_python_nlink}" ]] || fail "frozen Python type/mode/uid/size/link-count differs"
  observed_sha="$(sha256_file "${python_bin}")" || fail "frozen Python SHA-256 is unavailable"
  [[ "${observed_sha}" == "${frozen_python_sha}" ]] || fail "frozen Python SHA-256 differs"
}

validate_frozen_python_before_run_root_or_srun

for name in run_root release_root method_root method_archive method_manifest deployment_envelope; do
  deny_revoked_deployed_path "${name}" "${!name}"
done

run_root_canonical_value="$(readlink -m -- "${run_root}")" || fail "run root cannot be canonicalized"
readonly run_root_canonical="${run_root_canonical_value}"
[[ "${run_root_canonical}" == "${run_root}" ]] || fail "run root is not canonical or traverses a symlink/dot component"
case "${run_root_canonical}" in "${revoked_run_root}"|"${revoked_run_root}"/*) fail "revoked live run root or descendant is permanent NO-GO (${revoked_live_log_sha})" ;; esac

detached_launcher_value="$(readlink -f -- "${BASH_SOURCE[0]}")" || fail "detached launcher realpath is unavailable"
readonly detached_launcher="${detached_launcher_value}"
[[ "${BASH_SOURCE[0]}" == "${detached_launcher}" && -f "${detached_launcher}" && ! -L "${detached_launcher}" && "$(stat -c '%a' -- "${detached_launcher}")" == 555 ]] || fail "detached launcher path/mode differs"
detached_launcher_sha_value="$(sha256_file "${detached_launcher}")" || fail "detached launcher SHA-256 is unavailable"
readonly detached_launcher_sha="${detached_launcher_sha_value}"
[[ "${detached_launcher_sha}" == "${expected_detached_launcher_sha}" ]] || fail "detached launcher SHA differs from external live preflight"
if [[ "${role}" == child ]]; then
  [[ "${F13_PARENT_DETACHED_LAUNCHER_PATH:?parent detached launcher path required}" == "${detached_launcher}" ]] || fail "child did not execute the parent's exact detached launcher path"
  [[ "${F13_PARENT_DETACHED_LAUNCHER_SHA256:?parent detached launcher SHA required}" == "${detached_launcher_sha}" ]] || fail "child detached launcher SHA differs from parent"
fi

readonly controller="${method_root}/full30_action_arms_incomplete_repair_exact2_controller_v1.py"
readonly plan_tool="${method_root}/full30_action_arms_incomplete_repair_exact2_plan_v1.py"
readonly generator="${method_root}/full30_action_arms_incomplete_repair_exact2_generator_v1.py"
readonly resource="${method_root}/tools/reserve4_fixed_generation_sp4_136140_specialized_v1.py"
readonly release_builder="${method_root}/tools/build_full30_action_arms_incomplete_repair_exact2_release_v1.py"
readonly rank_wrapper="${method_root}/scripts/auh_generic_action_data_prep_rank_exec_v1.sh"
readonly launcher="${detached_launcher}"
readonly external_action_mp4_seed1="${external_root}/source_media/formal_00.mp4"
readonly external_action_mp4_seed2="${external_root}/source_media/formal_02.mp4"

# No caller-controlled Python startup or import-path variable may reach either
# the parent or the child.  The three deterministic variables below are set by
# this launcher itself and are scrubbed again at the srun boundary so the child
# repeats this presence check before recreating them.
while IFS='=' read -r caller_environment_name _; do
  case "${caller_environment_name}" in
    PYTHON*) fail "caller Python environment presence is forbidden: ${caller_environment_name}" ;;
  esac
done < <(env)
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0

# Every execution of the user-owned frozen interpreter is bootstrapped by the
# root-owned system Python.  The bootstrap verifies its own executable, holds
# the frozen interpreter by file descriptor, and then uses fd-exec.  Thus no
# application or gate can be started from a same-UID replacement of the
# frozen interpreter between a path check and exec.
read -r -d '' frozen_python_fd_exec_bootstrap_py <<'PY' || true
import hashlib
import os
import signal
import stat
import sys


def reject(message):
    raise SystemExit("frozen Python fd-exec bootstrap rejected: " + message)


def identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_gid,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def stable_open(path, expected_sha, expected_size, uid, gid):
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    before = os.fstat(descriptor)
    first = bytearray()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        first.extend(chunk)
    middle = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    second = bytearray()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        second.extend(chunk)
    after = os.fstat(descriptor)
    named = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o755
        or before.st_uid != uid
        or before.st_gid != gid
        or before.st_nlink != 1
        or before.st_size != expected_size
        or identity(before) != identity(middle)
        or identity(before) != identity(after)
        or identity(before) != identity(named)
        or first != second
        or hashlib.sha256(first).hexdigest() != expected_sha
    ):
        os.close(descriptor)
        reject("physical executable identity differs: " + path)
    return descriptor


if (
    sys.flags.isolated != 1
    or sys.flags.ignore_environment != 1
    or sys.flags.no_site != 1
    or sys.flags.no_user_site != 1
    or sys.flags.dont_write_bytecode != 1
    or len(sys.argv) < 9
):
    reject("isolated root-bootstrap flags or arguments differ")
bootstrap_path, bootstrap_sha, bootstrap_size_text = sys.argv[1:4]
target_path, target_sha, target_size_text = sys.argv[4:7]
if sys.argv[7] != "--":
    reject("argument separator differs")
target_arguments = sys.argv[8:]
if (
    bootstrap_path != "/usr/bin/python3.10"
    or target_path != "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
    or not bootstrap_size_text.isdecimal()
    or not target_size_text.isdecimal()
    or len(bootstrap_sha) != 64
    or len(target_sha) != 64
    or not target_arguments
):
    reject("literal executable authority differs")
if os.path.realpath(bootstrap_path) != bootstrap_path or os.path.realpath(target_path) != target_path:
    reject("executable path is not canonical")
try:
    running_path = os.readlink("/proc/self/exe")
except OSError as error:
    reject("running executable identity is unavailable: " + str(error))
if running_path != bootstrap_path:
    reject("running executable is not the root-owned bootstrap")

signal.alarm(30)
bootstrap_fd = stable_open(
    bootstrap_path, bootstrap_sha, int(bootstrap_size_text), 0, 0
)
try:
    if identity(os.fstat(bootstrap_fd)) != identity(os.stat("/proc/self/exe")):
        reject("running bootstrap inode differs")
finally:
    os.close(bootstrap_fd)
target_fd = stable_open(
    target_path, target_sha, int(target_size_text), 2012, 2000
)

allowed_environment = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "SLURM_JOB_ID",
    "SLURM_STEP_ID",
    "TMPDIR",
    "ROCR_VISIBLE_DEVICES",
    "GADP_NODE_LOCAL_SCRATCH",
    "GADP_NODE_LOCAL_SCRATCH_FSTYPE",
    "GADP_HOST_MEMORY_SAMPLE_JOURNAL",
    "GADP_HOST_MEMORY_MONITOR_START_RECEIPT",
    "GADP_HOST_MEMORY_MONITOR_PID",
    "GADP_HOST_MEMORY_MONITOR_SUPERVISOR_PID",
    "NATIVE_SERIALIZED_HOST_LOAD_REQUIRED",
    "NATIVE_V_AXIS_LOAD_LOCK",
    "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED",
    "F13_METHOD_MANIFEST",
    "F13_METHOD_MANIFEST_SHA256",
    "F13_VERIFIED_RUNNER_PATH",
    "F13_VERIFIED_RUNNER_SHA256",
    "F13_RANK_WRAPPER_SHA256",
}
unsafe_exact = {
    "BASH_ENV",
    "ENV",
    "SHELLOPTS",
    "BASHOPTS",
    "CDPATH",
    "GLOBIGNORE",
    "GCONV_PATH",
    "LOCPATH",
}
for name in os.environ:
    if name in unsafe_exact or name.startswith(
        ("LD_", "DYLD_", "GLIBC_", "MALLOC_", "BASH_FUNC_")
    ):
        os.close(target_fd)
        reject("unsafe loader or shell environment is present: " + name)
safe_environment = {
    name: os.environ[name] for name in allowed_environment if name in os.environ
}
safe_environment.update(
    {
        "PATH": "/usr/bin:/bin",
        "HOME": "/vast/users/guangyi.chen",
        "USER": "guangyi.chen",
        "LOGNAME": "guangyi.chen",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
    }
)
signal.alarm(0)
if os.execve not in os.supports_fd:
    os.close(target_fd)
    reject("held-fd execve is unavailable")
try:
    os.execve(target_fd, [target_path, *target_arguments], safe_environment)
finally:
    os.close(target_fd)
PY
readonly frozen_python_fd_exec_bootstrap_py
readonly -a frozen_python_exec_prefix=(
  "${root_bootstrap_python}" -I -S -s -B -c "${frozen_python_fd_exec_bootstrap_py}"
  "${root_bootstrap_python}" "${root_bootstrap_python_sha}" "${root_bootstrap_python_size}"
  "${python_bin}" "${frozen_python_sha}" "${frozen_python_size}" --
)

run_frozen_python() {
  "${frozen_python_exec_prefix[@]}" "$@"
}

# This stdlib-only verifier is the first Python payload executed by both the
# login-side parent and the compute child.  It does not import any code from
# the materialized release.  Instead it binds every materialized byte to the
# detached launcher's literal manifest identity before an application module
# can execute.  The verifier is intentionally repeated before every child
# application command so a same-path replacement after an earlier check is
# not inherited by a later command.
read -r -d '' materialization_bootstrap_py <<'PY' || true
import hashlib
import importlib.abc
import importlib.util
import json
import os
import stat
import sys


def reject(message):
    raise SystemExit("materialized method bootstrap rejected: " + message)


if (
    sys.flags.no_site != 1
    or sys.flags.no_user_site != 1
    or getattr(sys.flags, "safe_path", 0) != 1
    or sys.flags.dont_write_bytecode != 1
    or sys.flags.hash_randomization != 0
    or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
    or os.environ.get("PYTHONNOUSERSITE") != "1"
    or os.environ.get("PYTHONHASHSEED") != "0"
):
    reject("isolated deterministic interpreter flags differ")


def canonical_json_bytes(value):
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        reject("manifest is not canonical finite JSON: " + str(error))


def metadata_tuple(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_gid,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def stable_plain_bytes(path, expected_mode, expected_uid):
    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        reject("plain file open failed: %s: %s" % (path, error))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            reject("member is not regular: " + path)
        if before.st_uid != expected_uid or before.st_nlink != 1:
            reject("member owner/link-count differs: " + path)
        if stat.S_IMODE(before.st_mode) != expected_mode:
            reject("member mode differs: " + path)
        first_chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            first_chunks.append(chunk)
        middle = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        second_chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            second_chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    first = b"".join(first_chunks)
    second = b"".join(second_chunks)
    if first != second:
        reject("member changed between same-fd reads: " + path)
    if not (
        metadata_tuple(before)
        == metadata_tuple(middle)
        == metadata_tuple(after)
    ):
        reject("member metadata changed during same-fd reads: " + path)
    if len(first) != before.st_size:
        reject("member byte count differs from stat size: " + path)
    return first, metadata_tuple(before)


def directory_snapshot(root, expected_uid):
    directories = {}
    files = {}
    pending = [("", root)]
    while pending:
        relative_parent, absolute_parent = pending.pop()
        try:
            parent_metadata = os.lstat(absolute_parent)
        except OSError as error:
            reject("directory lstat failed: %s: %s" % (absolute_parent, error))
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or stat.S_ISLNK(parent_metadata.st_mode)
            or parent_metadata.st_uid != expected_uid
            or parent_metadata.st_nlink < 2
            or stat.S_IMODE(parent_metadata.st_mode) & 0o022
        ):
            reject("directory identity/mode differs: " + absolute_parent)
        directories[relative_parent] = metadata_tuple(parent_metadata)
        try:
            entries = sorted(os.scandir(absolute_parent), key=lambda item: item.name)
        except OSError as error:
            reject("directory enumeration failed: %s: %s" % (absolute_parent, error))
        for entry in entries:
            if entry.name in (".", "..") or "/" in entry.name:
                reject("directory entry name differs: " + entry.name)
            relative = entry.name if not relative_parent else relative_parent + "/" + entry.name
            try:
                observed = entry.stat(follow_symlinks=False)
            except OSError as error:
                reject("entry lstat failed: %s: %s" % (relative, error))
            if stat.S_ISLNK(observed.st_mode):
                reject("symlink is forbidden: " + relative)
            if observed.st_uid != expected_uid:
                reject("entry owner differs: " + relative)
            if stat.S_ISDIR(observed.st_mode):
                pending.append((relative, entry.path))
            elif stat.S_ISREG(observed.st_mode):
                files[relative] = metadata_tuple(observed)
            else:
                reject("non-regular materialization entry is forbidden: " + relative)
    return directories, files


if len(sys.argv) < 6:
    reject("bootstrap argument count differs")
root, manifest_path, expected_manifest_sha, expected_manifest_digest, expected_revision = sys.argv[1:6]
application_arguments = sys.argv[6:]
await_parent_commit = False
if application_arguments[:1] == ["--f13-await-parent-commit"]:
    await_parent_commit = True
    application_arguments = application_arguments[1:]
if not (os.path.isabs(root) and os.path.isabs(manifest_path)):
    reject("paths must be absolute")
if os.path.realpath(root) != root or os.path.realpath(manifest_path) != manifest_path:
    reject("root or manifest is not canonical")
uid = os.getuid()
try:
    root_metadata = os.lstat(root)
except OSError as error:
    reject("method root is unavailable: " + str(error))
if (
    not stat.S_ISDIR(root_metadata.st_mode)
    or stat.S_ISLNK(root_metadata.st_mode)
    or root_metadata.st_uid != uid
    or root_metadata.st_nlink < 2
    or stat.S_IMODE(root_metadata.st_mode) & 0o022
):
    reject("method root identity differs")

manifest_raw, _ = stable_plain_bytes(manifest_path, 0o444, uid)
if hashlib.sha256(manifest_raw).hexdigest() != expected_manifest_sha:
    reject("manifest file SHA-256 differs")
try:
    manifest = json.loads(manifest_raw)
except (json.JSONDecodeError, UnicodeDecodeError) as error:
    reject("manifest JSON is invalid: " + str(error))
if canonical_json_bytes(manifest) + b"\n" != manifest_raw:
    reject("manifest bytes are not canonical")
if not isinstance(manifest, dict):
    reject("manifest top level is not an object")
unsigned = dict(manifest)
declared_digest = unsigned.pop("manifest_digest", None)
if (
    declared_digest != expected_manifest_digest
    or hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != declared_digest
):
    reject("manifest digest differs")
rows = manifest.get("files")
if (
    manifest.get("schema_version")
    != "bernini-full30-action-arms-incomplete-repair-exact2-core-release-v6"
    or manifest.get("release_generation") != "r6"
    or manifest.get("member_root") != "methods/bernini_action_editing"
    or manifest.get("file_count") != 25
    or manifest.get("exact_member_closure") is not True
    or manifest.get("content_closure_sha1") != expected_revision
    or not isinstance(rows, list)
    or len(rows) != 25
):
    reject("manifest release identity/closure differs")
if not root.endswith("/" + manifest["member_root"]):
    reject("method root suffix differs from manifest member root")
if hashlib.sha1(
    canonical_json_bytes({"member_root": manifest["member_root"], "files": rows})
).hexdigest() != expected_revision:
    reject("content closure SHA-1 differs")

expected_files = {}
expected_directories = {""}
previous_path = None
for row in rows:
    if not isinstance(row, dict) or set(row) != {"mode", "path", "sha256", "size"}:
        reject("manifest file row shape differs")
    relative = row.get("path")
    mode = row.get("mode")
    size = row.get("size")
    digest = row.get("sha256")
    if (
        not isinstance(relative, str)
        or not relative
        or relative.startswith("/")
        or "//" in relative
        or relative.split("/")[-1] in ("", ".", "..")
        or any(part in ("", ".", "..") for part in relative.split("/"))
        or previous_path is not None and relative <= previous_path
        or mode not in (0o444, 0o555)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        reject("manifest file row value differs")
    previous_path = relative
    expected_files[relative] = (mode, size, digest)
    parts = relative.split("/")[:-1]
    for index in range(1, len(parts) + 1):
        expected_directories.add("/".join(parts[:index]))

before_directories, before_files = directory_snapshot(root, uid)
if set(before_directories) != expected_directories or set(before_files) != set(expected_files):
    reject("materialization exact path inventory differs")
first_pass = {}
verified_payloads = {}
for relative in sorted(expected_files):
    mode, size, digest = expected_files[relative]
    raw, metadata = stable_plain_bytes(os.path.join(root, relative), mode, uid)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        reject("materialized member bytes differ: " + relative)
    if metadata != before_files[relative]:
        reject("materialized member path/fd identity differs: " + relative)
    first_pass[relative] = (metadata, digest)
    verified_payloads[relative] = raw

middle_directories, middle_files = directory_snapshot(root, uid)
if (
    middle_directories != before_directories
    or middle_files != before_files
):
    reject("materialization tree changed after first pass")
for relative in sorted(expected_files):
    mode, size, digest = expected_files[relative]
    raw, metadata = stable_plain_bytes(os.path.join(root, relative), mode, uid)
    if (
        len(raw) != size
        or hashlib.sha256(raw).hexdigest() != digest
        or (metadata, digest) != first_pass[relative]
    ):
        reject("materialized member changed on second pass: " + relative)
after_directories, after_files = directory_snapshot(root, uid)
if after_directories != before_directories or after_files != before_files:
    reject("materialization tree changed after second pass")

if application_arguments:
    target = application_arguments[0]
    if not os.path.isabs(target) or os.path.realpath(target) != target:
        reject("application target must be one canonical absolute path")
    try:
        target_relative = os.path.relpath(target, root)
    except ValueError as error:
        reject("application target cannot be relativized: " + str(error))
    if (
        target_relative not in verified_payloads
        or not target_relative.endswith(".py")
        or os.path.join(root, target_relative) != target
    ):
        reject("application target is not one verified Python member")

    module_rows = {}
    for relative, raw in verified_payloads.items():
        if not relative.endswith(".py"):
            continue
        module_name = os.path.basename(relative)[:-3]
        if not module_name.isidentifier() or module_name in module_rows:
            reject("verified Python module basename is ambiguous: " + relative)
        module_rows[module_name] = (relative, raw)

    class VerifiedReleaseLoader(importlib.abc.Loader):
        def create_module(self, specification):
            return None

        def exec_module(self, module):
            relative, raw = module_rows[module.__spec__.name]
            filename = os.path.join(root, relative)
            module.__file__ = filename
            module.__cached__ = None
            module.__loader__ = self
            code = compile(raw, filename, "exec", dont_inherit=True)
            exec(code, module.__dict__)

    class VerifiedReleaseFinder(importlib.abc.MetaPathFinder):
        def __init__(self):
            self.loader = VerifiedReleaseLoader()

        def find_spec(self, fullname, path=None, target=None):
            # Released modules insert their source directory into sys.path.
            # Remove every release-local search root before *all* delegation,
            # so a file created after capture can never satisfy a later import.
            forbidden = {root, os.path.join(root, "tools"), ""}
            sys.path[:] = [entry for entry in sys.path if entry not in forbidden]
            if fullname not in module_rows:
                return None
            relative, _ = module_rows[fullname]
            return importlib.util.spec_from_loader(
                fullname,
                self.loader,
                origin=os.path.join(root, relative),
                is_package=False,
            )

    if any(name in sys.modules for name in module_rows):
        reject("release-local module was imported before verified closure")
    forbidden_search_roots = {root, os.path.join(root, "tools"), ""}
    sys.path[:] = [
        entry for entry in sys.path if entry not in forbidden_search_roots
    ]
    sys.meta_path.insert(0, VerifiedReleaseFinder())
    sys.dont_write_bytecode = True
    # -S -s -P -B removes automatic site/sitecustomize loading, user site,
    # unsafe cwd/script paths, and bytecode writes while still honoring the
    # launcher's explicitly sealed PYTHONHASHSEED=0 environment.  The
    # frozen environment's dependency directory is added explicitly only
    # after the release sources have been verified and captured in memory.
    frozen_site_packages = (
        "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages"
    )
    if os.path.isdir(frozen_site_packages) and not os.path.islink(frozen_site_packages):
        if os.path.realpath(frozen_site_packages) != frozen_site_packages:
            reject("frozen site-packages path is not canonical")
        site_metadata = os.lstat(frozen_site_packages)
        if (
            not stat.S_ISDIR(site_metadata.st_mode)
            or site_metadata.st_uid != uid
            or stat.S_IMODE(site_metadata.st_mode) & 0o022
        ):
            reject("frozen site-packages identity/mode differs")
        sys.path.append(frozen_site_packages)
    if await_parent_commit:
        sys.stdout.write("F13_VERIFIED_RUNNER_READY\n")
        sys.stdout.flush()
        if sys.stdin.buffer.readline() != b"F13_PARENT_COMMIT\n":
            reject("parent publication commit token differs")
        devnull = os.open(os.devnull, os.O_WRONLY | os.O_CLOEXEC)
        try:
            os.dup2(devnull, 1)
        finally:
            os.close(devnull)
    sys.argv = [target, *application_arguments[1:]]
    main_globals = {
        "__name__": "__main__",
        "__file__": target,
        "__package__": None,
        "__cached__": None,
        "__loader__": None,
        "__spec__": None,
        "__builtins__": __builtins__,
    }
    exec(
        compile(verified_payloads[target_relative], target, "exec", dont_inherit=True),
        main_globals,
    )
PY
readonly materialization_bootstrap_py

# A fixed-Python, stdlib-only exec gate ensures that an srun process cannot
# exist until the parent has independently bound the gate PID and Linux start
# time.  The gate PID is preserved by execve, so the same token identifies the
# local srun for its entire lifetime.  A pre-exec alarm guarantees that an
# identity-unknown gate closes by itself without any unsafe numeric signal.
read -r -d '' exact_srun_exec_gate_py <<'PY' || true
import hashlib
import os
import signal
import stat
import sys


def reject(message):
    raise SystemExit("exact srun exec gate rejected: " + message)


if (
    sys.flags.no_site != 1
    or sys.flags.no_user_site != 1
    or getattr(sys.flags, "safe_path", 0) != 1
    or sys.flags.dont_write_bytecode != 1
    or sys.flags.hash_randomization != 0
):
    reject("isolated deterministic interpreter flags differ")
if len(sys.argv) < 6:
    reject("argument count differs")
srun_path, expected_sha, expected_size_text, log_path = sys.argv[1:5]
srun_arguments = sys.argv[5:]
if (
    srun_path != "/usr/bin/srun"
    or not expected_size_text.isdecimal()
    or not os.path.isabs(log_path)
    or os.path.realpath(log_path) != log_path
    or os.path.basename(log_path) != "arms-incomplete-exact2-generation.log"
    or os.path.basename(os.path.dirname(log_path)) != "logs"
):
    reject("pinned path/size shape differs")

signal.alarm(30)
flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
try:
    descriptor = os.open(srun_path, flags)
except OSError as error:
    reject("srun open failed: " + str(error))
before = os.fstat(descriptor)
first = bytearray()
while True:
    chunk = os.read(descriptor, 1024 * 1024)
    if not chunk:
        break
    first.extend(chunk)
middle = os.fstat(descriptor)
os.lseek(descriptor, 0, os.SEEK_SET)
second = bytearray()
while True:
    chunk = os.read(descriptor, 1024 * 1024)
    if not chunk:
        break
    second.extend(chunk)
after = os.fstat(descriptor)
identity = lambda value: (
    value.st_dev,
    value.st_ino,
    value.st_uid,
    value.st_gid,
    value.st_mode,
    value.st_nlink,
    value.st_size,
    value.st_mtime_ns,
    value.st_ctime_ns,
)
if (
    not stat.S_ISREG(before.st_mode)
    or stat.S_IMODE(before.st_mode) != 0o755
    or before.st_uid != 0
    or before.st_gid != 0
    or before.st_nlink != 1
    or before.st_size != int(expected_size_text)
    or identity(before) != identity(middle)
    or identity(before) != identity(after)
    or first != second
    or hashlib.sha256(first).hexdigest() != expected_sha
):
    reject("srun physical identity differs")

try:
    stat_line = open("/proc/self/stat", "r", encoding="ascii").read()
except (OSError, UnicodeError) as error:
    reject("self start-time is unavailable: " + str(error))
separator = stat_line.rfind(") ")
if separator < 0:
    reject("self stat shape differs")
fields = stat_line[separator + 2 :].split()
if len(fields) < 20 or fields[0] == "Z" or not fields[19].isdecimal() or int(fields[19]) <= 0:
    reject("self process identity differs")
pid = os.getpid()
start_time = fields[19]
sys.stdout.write("F13_SRUN_GATE_READY|%d|%s\n" % (pid, start_time))
sys.stdout.flush()
if sys.stdin.buffer.readline() != b"F13_SRUN_GATE_COMMIT\n":
    reject("parent commit token differs")

logs_path = os.path.dirname(log_path)
run_path = os.path.dirname(logs_path)
directory_flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
directory_flags |= getattr(os, "O_NOFOLLOW", 0)
run_fd = os.open(run_path, directory_flags)
logs_fd = os.open("logs", directory_flags, dir_fd=run_fd)
try:
    run_stat = os.fstat(run_fd)
    logs_stat = os.fstat(logs_fd)
    if (
        not stat.S_ISDIR(run_stat.st_mode)
        or not stat.S_ISDIR(logs_stat.st_mode)
        or run_stat.st_uid != os.getuid()
        or logs_stat.st_uid != os.getuid()
        or stat.S_IMODE(run_stat.st_mode) != 0o700
        or stat.S_IMODE(logs_stat.st_mode) != 0o700
    ):
        reject("run/log directory identity differs")
    log_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    log_flags |= getattr(os, "O_NOFOLLOW", 0)
    log_fd = os.open(os.path.basename(log_path), log_flags, 0o600, dir_fd=logs_fd)
    log_stat = os.fstat(log_fd)
    if (
        not stat.S_ISREG(log_stat.st_mode)
        or log_stat.st_uid != os.getuid()
        or log_stat.st_nlink != 1
        or stat.S_IMODE(log_stat.st_mode) != 0o600
        or log_stat.st_size != 0
    ):
        reject("generation log creation identity differs")
    os.fsync(log_fd)
    os.fsync(logs_fd)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)
finally:
    os.close(logs_fd)
    os.close(run_fd)
signal.alarm(0)
safe_names = {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL"}
safe_environment = {
    name: value
    for name, value in os.environ.items()
    if name in safe_names
    or name.startswith("LC_")
}
safe_environment["PATH"] = "/usr/bin:/bin"
for name in safe_environment:
    if name.startswith(("LD_", "DYLD_", "GLIBC_", "MALLOC_", "BASH_FUNC_")):
        reject("unsafe environment survived allowlist")
if os.execve not in os.supports_fd:
    reject("held-fd execve is unavailable")
try:
    os.execve(descriptor, [srun_path, *srun_arguments], safe_environment)
finally:
    os.close(descriptor)
PY
readonly exact_srun_exec_gate_py

read -r -d '' exact_parent_command_exec_gate_py <<'PY' || true
import hashlib
import os
import signal
import stat
import sys


def reject(message):
    raise SystemExit("exact parent command exec gate rejected: " + message)


if (
    sys.flags.no_site != 1
    or sys.flags.no_user_site != 1
    or getattr(sys.flags, "safe_path", 0) != 1
    or sys.flags.dont_write_bytecode != 1
    or sys.flags.hash_randomization != 0
    or len(sys.argv) < 5
):
    reject("interpreter flags or arguments differ")
target, expected_sha, expected_size_text = sys.argv[1:4]
target_arguments = sys.argv[4:]
preserve_stdio = False
if target_arguments[:1] == ["--f13-preserve-stdio"]:
    preserve_stdio = True
    target_arguments = target_arguments[1:]
if (
    target != "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
    or not expected_size_text.isdecimal()
):
    reject("target identity shape differs")
signal.alarm(30)
flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(target, flags)
before = os.fstat(descriptor)
first = bytearray()
while True:
    chunk = os.read(descriptor, 1024 * 1024)
    if not chunk:
        break
    first.extend(chunk)
middle = os.fstat(descriptor)
os.lseek(descriptor, 0, os.SEEK_SET)
second = bytearray()
while True:
    chunk = os.read(descriptor, 1024 * 1024)
    if not chunk:
        break
    second.extend(chunk)
after = os.fstat(descriptor)
identity = lambda value: (
    value.st_dev,
    value.st_ino,
    value.st_uid,
    value.st_gid,
    value.st_mode,
    value.st_nlink,
    value.st_size,
    value.st_mtime_ns,
    value.st_ctime_ns,
)
if (
    not stat.S_ISREG(before.st_mode)
    or stat.S_IMODE(before.st_mode) != 0o755
    or before.st_uid != 2012
    or before.st_nlink != 1
    or before.st_size != int(expected_size_text)
    or identity(before) != identity(middle)
    or identity(before) != identity(after)
    or first != second
    or hashlib.sha256(first).hexdigest() != expected_sha
):
    reject("target physical identity differs")
stat_line = open("/proc/self/stat", "r", encoding="ascii").read()
separator = stat_line.rfind(") ")
fields = stat_line[separator + 2 :].split() if separator >= 0 else []
if len(fields) < 20 or fields[0] == "Z" or not fields[19].isdecimal() or int(fields[19]) <= 0:
    reject("self process identity differs")
sys.stdout.write("F13_PARENT_COMMAND_GATE_READY|%d|%s\n" % (os.getpid(), fields[19]))
sys.stdout.flush()
if sys.stdin.buffer.readline() != b"F13_PARENT_COMMAND_GATE_COMMIT\n":
    reject("parent commit token differs")
if not preserve_stdio:
    devnull = os.open(os.devnull, os.O_WRONLY | os.O_CLOEXEC)
    try:
        os.dup2(devnull, 1)
    finally:
        os.close(devnull)
signal.alarm(0)
safe_names = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONNOUSERSITE",
    "PYTHONHASHSEED",
}
safe_environment = {
    name: value
    for name, value in os.environ.items()
    if name in safe_names or name.startswith("LC_")
}
safe_environment["PATH"] = "/usr/bin:/bin"
safe_environment["PYTHONDONTWRITEBYTECODE"] = "1"
safe_environment["PYTHONNOUSERSITE"] = "1"
safe_environment["PYTHONHASHSEED"] = "0"
if os.execve not in os.supports_fd:
    reject("held-fd execve is unavailable")
try:
    os.execve(descriptor, [target, *target_arguments], safe_environment)
finally:
    os.close(descriptor)
PY
readonly exact_parent_command_exec_gate_py

read -r -d '' stable_json_field_py <<'PY' || true
import hashlib
import json
import os
import stat
import sys


def reject(message):
    raise SystemExit("stable JSON field rejected: " + message)


if (
    sys.flags.no_site != 1
    or sys.flags.no_user_site != 1
    or getattr(sys.flags, "safe_path", 0) != 1
    or sys.flags.dont_write_bytecode != 1
    or sys.flags.hash_randomization != 0
    or len(sys.argv) < 4
):
    reject("interpreter flags or arguments differ")
path, expected_sha = sys.argv[1:3]
keys = sys.argv[3:]
if not os.path.isabs(path) or os.path.realpath(path) != path:
    reject("path is not canonical absolute")
flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(path, flags)
try:
    before = os.fstat(descriptor)
    first = bytearray()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        first.extend(chunk)
    middle = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    second = bytearray()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        second.extend(chunk)
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
identity = lambda value: (
    value.st_dev,
    value.st_ino,
    value.st_uid,
    value.st_gid,
    value.st_mode,
    value.st_nlink,
    value.st_size,
    value.st_mtime_ns,
    value.st_ctime_ns,
)
if (
    not stat.S_ISREG(before.st_mode)
    or before.st_uid != os.getuid()
    or before.st_nlink != 1
    or stat.S_IMODE(before.st_mode) not in (0o400, 0o444)
    or identity(before) != identity(middle)
    or identity(before) != identity(after)
    or first != second
    or hashlib.sha256(first).hexdigest() != expected_sha
):
    reject("file identity/bytes/SHA differ")
try:
    value = json.loads(first)
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    reject("JSON differs: " + str(error))
for key in keys:
    if type(value) is not dict or key not in value:
        reject("field path differs")
    value = value[key]
if type(value) not in (str, int) or isinstance(value, bool):
    reject("field scalar type differs")
sys.stdout.write(str(value) + "\n")
PY
readonly stable_json_field_py

stable_json_field() {
  local path="$1" expected_sha="$2"
  shift 2
  run_frozen_python -S -s -P -B -c "${stable_json_field_py}" \
    "${path}" "${expected_sha}" "$@"
}

verify_materialized_method_before_app_import() {
  run_frozen_python -S -s -P -B -c "${materialization_bootstrap_py}" \
    "${method_root}" "${method_manifest}" "${core_manifest_sha}" \
    "${core_manifest_digest}" "${method_revision}"
}

run_verified_materialized_app() {
  local target="$1"
  shift
  run_frozen_python -S -s -P -B -c "${materialization_bootstrap_py}" \
    "${method_root}" "${method_manifest}" "${core_manifest_sha}" \
    "${core_manifest_digest}" "${method_revision}" "${target}" "$@"
}

# State used by the child and parent signal/exit machines.  No handler ever
# signals the retained holder job id, its batch shell, or its extern step.
child_phase=entry
child_failure_phase=prepare
child_terminal_ready_committed=false
child_spawn_in_progress=false
child_spawn_deferred_signal_status=0
child_spawn_deferred_signal_phase=""
child_owned_pid=""
child_owned_starttime=""
host_memory_monitor_pid=""
host_memory_monitor_starttime=""
task_scratch=""
task_scratch_binding=""
task_scratch_binding_sha=""
scratch_prepare=""
scratch_prepare_sha=""
compute_preflight=""
compute_preflight_sha=""
owned_step_cgroup_path=""
owned_step_cgroup_procs=""
owned_step_cgroup_membership=()
owned_step_cgroup_tokens=()
parent_srun_pid=""
parent_srun_starttime=""
parent_srun_gate_read_fd=""
parent_srun_gate_write_fd=""
parent_owned_pid=""
parent_owned_starttime=""
parent_command_gate_read_fd=""
parent_command_gate_write_fd=""
parent_signal_status=0
parent_spawn_in_progress=false
parent_spawn_deferred_signal_status=0
parent_success_commit_active=false
parent_success_commit_deferred_signal_status=0
parent_success_committed=false
parent_publisher_read_fd=""
parent_publisher_write_fd=""

pid_starttime() {
  local pid="$1" line rest value
  [[ "${pid}" =~ ^[1-9][0-9]*$ && -r "/proc/${pid}/stat" ]] || return 1
  IFS= read -r line <"/proc/${pid}/stat" || return 1
  [[ "${line}" == *') '* ]] || return 1
  rest="${line##*) }"
  value="$(awk '{print $20}' <<<"${rest}")" || return 1
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s\n' "${value}"
}

pid_state() {
  local pid="$1" line rest state
  [[ "${pid}" =~ ^[1-9][0-9]*$ && -r "/proc/${pid}/stat" ]] || return 1
  IFS= read -r line <"/proc/${pid}/stat" || return 1
  [[ "${line}" == *') '* ]] || return 1
  rest="${line##*) }"
  state="${rest%% *}"
  [[ "${state}" =~ ^[A-Z]$ ]] || return 1
  printf '%s\n' "${state}"
}

pid_is_same_process() {
  local pid="$1" expected_start="$2" observed_start
  [[ -n "${expected_start}" ]] || return 1
  observed_start="$(pid_starttime "${pid}")" || return 1
  [[ "${observed_start}" == "${expected_start}" ]]
}

terminate_owned_pid_bounded() {
  local pid="$1" expected_start="$2" label="$3" attempt state
  pid_is_same_process "${pid}" "${expected_start}" || return 1
  kill -TERM "${pid}" 2>/dev/null || true
  for attempt in $(seq 1 50); do
    pid_is_same_process "${pid}" "${expected_start}" || { wait "${pid}" 2>/dev/null || true; return 0; }
    state="$(pid_state "${pid}" 2>/dev/null)" || state=""
    [[ "${state}" != Z ]] || { wait "${pid}" 2>/dev/null || true; return 0; }
    sleep 0.1
  done
  pid_is_same_process "${pid}" "${expected_start}" && kill -KILL "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  if pid_is_same_process "${pid}" "${expected_start}"; then
    echo "[arms-incomplete-exact2-r6] ERROR: owned ${label} survived bounded TERM/KILL" >&2
    return 1
  fi
  return 0
}

step_cgroup_is_exact_owned() {
  local line cgroup_path="" v2_count=0
  [[ "${role}" == child && "${SLURM_JOB_ID:-}" == "${holder_job}" && "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || return 1
  [[ -r /proc/self/cgroup ]] || return 1
  while IFS= read -r line; do
    case "${line}" in
      0::*)
        v2_count=$((v2_count + 1))
        cgroup_path="${line#0::}"
        ;;
    esac
  done </proc/self/cgroup || return 1
  [[ ${v2_count} -eq 1 && "${cgroup_path}" == /* && "${cgroup_path}" != *'//'* ]] || return 1
  [[ "${cgroup_path}" != *'/./'* && "${cgroup_path}" != *'/../'* && "${cgroup_path}" != */. && "${cgroup_path}" != */.. ]] || return 1
  [[ "${cgroup_path}" =~ (^|/)job_${holder_job}(/|$) ]] || return 1
  [[ "${cgroup_path}" =~ (^|/)step_${SLURM_STEP_ID}(/|$) ]] || return 1
  owned_step_cgroup_path="${cgroup_path}"
}

resolve_owned_step_cgroup_procs() {
  local separator_index root mount_point relative candidate match_count=0 field_index
  local -a fields
  step_cgroup_is_exact_owned || return 1
  [[ -r /proc/self/mountinfo ]] || return 1
  while IFS=' ' read -r -a fields; do
    separator_index=-1
    for (( field_index=6; field_index<${#fields[@]}; field_index++ )); do
      if [[ "${fields[field_index]}" == - ]]; then separator_index="${field_index}"; break; fi
    done
    (( separator_index >= 0 && separator_index + 1 < ${#fields[@]} )) || continue
    [[ "${fields[separator_index + 1]}" == cgroup2 ]] || continue
    root="${fields[3]}"
    mount_point="${fields[4]}"
    [[ "${root}" == /* && "${mount_point}" == /* && "${root}" != *'\'* && "${mount_point}" != *'\'* ]] || return 1
    if [[ "${root}" == / ]]; then
      relative="${owned_step_cgroup_path}"
    elif [[ "${owned_step_cgroup_path}" == "${root}" ]]; then
      relative=""
    elif [[ "${owned_step_cgroup_path}" == "${root}"/* ]]; then
      relative="${owned_step_cgroup_path#"${root}"}"
    else
      continue
    fi
    candidate="${mount_point%/}${relative}/cgroup.procs"
    match_count=$((match_count + 1))
    owned_step_cgroup_procs="${candidate}"
  done </proc/self/mountinfo || return 1
  [[ ${match_count} -eq 1 && -r "${owned_step_cgroup_procs}" && ! -L "${owned_step_cgroup_procs}" ]] || return 1
}

read_owned_step_cgroup_membership() {
  local line existing
  owned_step_cgroup_membership=()
  [[ -n "${owned_step_cgroup_procs}" && -r "${owned_step_cgroup_procs}" && ! -L "${owned_step_cgroup_procs}" ]] || return 1
  while IFS= read -r line; do
    [[ "${line}" =~ ^[1-9][0-9]*$ && ${line} -gt 1 ]] || return 1
    for existing in "${owned_step_cgroup_membership[@]}"; do
      [[ "${existing}" != "${line}" ]] || return 1
    done
    owned_step_cgroup_membership+=("${line}")
  done <"${owned_step_cgroup_procs}" || return 1
}

owned_step_membership_contains() {
  local expected="$1" member
  for member in "${owned_step_cgroup_membership[@]}"; do
    [[ "${member}" == "${expected}" ]] && return 0
  done
  return 1
}

collect_owned_step_cgroup_pids() {
  local attempt pid uid self_uid starttime first_key second_key retry
  local -a first_members candidate_tokens
  resolve_owned_step_cgroup_procs || return 1
  self_uid="$(id -u)" || return 1
  for attempt in $(seq 1 5); do
    read_owned_step_cgroup_membership || return 1
    first_members=("${owned_step_cgroup_membership[@]}")
    first_key="${first_members[*]}"
    candidate_tokens=()
    retry=false
    for pid in "${first_members[@]}"; do
      [[ "${pid}" != "$$" && "${pid}" != "${BASHPID}" ]] || continue
      if ! uid="$(stat -c '%u' -- "/proc/${pid}" 2>/dev/null)"; then
        read_owned_step_cgroup_membership || return 1
        owned_step_membership_contains "${pid}" && return 1
        retry=true
        break
      fi
      [[ "${uid}" == "${self_uid}" ]] || return 1
      if ! starttime="$(pid_starttime "${pid}" 2>/dev/null)"; then
        read_owned_step_cgroup_membership || return 1
        owned_step_membership_contains "${pid}" && return 1
        retry=true
        break
      fi
      candidate_tokens+=("${pid}:${starttime}")
    done
    [[ "${retry}" == false ]] || { sleep 0.01; continue; }
    read_owned_step_cgroup_membership || return 1
    second_key="${owned_step_cgroup_membership[*]}"
    [[ "${second_key}" == "${first_key}" ]] || { sleep 0.01; continue; }
    owned_step_cgroup_tokens=("${candidate_tokens[@]}")
    return 0
  done
  return 1
}

terminate_owned_step_cgroup_bounded() {
  local attempt pid starttime token
  collect_owned_step_cgroup_pids || return 1
  for token in "${owned_step_cgroup_tokens[@]}"; do
    pid="${token%%:*}"; starttime="${token#*:}"
    pid_is_same_process "${pid}" "${starttime}" && kill -TERM "${pid}" 2>/dev/null || true
  done
  for attempt in $(seq 1 50); do
    collect_owned_step_cgroup_pids || return 1
    (( ${#owned_step_cgroup_tokens[@]} == 0 )) && return 0
    sleep 0.1
  done
  for token in "${owned_step_cgroup_tokens[@]}"; do
    pid="${token%%:*}"; starttime="${token#*:}"
    pid_is_same_process "${pid}" "${starttime}" && kill -KILL "${pid}" 2>/dev/null || true
  done
  for attempt in $(seq 1 20); do
    collect_owned_step_cgroup_pids || return 1
    (( ${#owned_step_cgroup_tokens[@]} == 0 )) && return 0
    sleep 0.1
  done
  echo "[arms-incomplete-exact2-r6] ERROR: owned numbered-step cgroup descendants survived bounded TERM/KILL" >&2
  return 1
}

run_child_owned_command() {
  local status starttime_value deferred_status deferred_phase restore_errexit
  restore_errexit=false
  [[ $- == *e* ]] && restore_errexit=true
  [[ "${role}" == child && -z "${child_owned_pid}" ]] || return 71
  child_spawn_in_progress=true
  "$@" &
  child_owned_pid=$!
  starttime_value="$(pid_starttime "${child_owned_pid}")"
  status=$?
  if (( status == 0 )); then child_owned_starttime="${starttime_value}"; else child_owned_starttime=""; fi
  deferred_status="${child_spawn_deferred_signal_status}"
  deferred_phase="${child_spawn_deferred_signal_phase}"
  if (( deferred_status != 0 )); then
    trap '' INT TERM HUP
    child_spawn_in_progress=false
    child_spawn_deferred_signal_status=0
    child_spawn_deferred_signal_phase=""
    child_signal_handler "${deferred_status}" "${deferred_phase}"
  fi
  child_spawn_in_progress=false
  # A signal can latch after the first snapshot but before the spawning state
  # becomes idle.  Re-read once in idle; any later signal dispatches directly.
  deferred_status="${child_spawn_deferred_signal_status}"
  deferred_phase="${child_spawn_deferred_signal_phase}"
  if (( deferred_status != 0 )); then
    trap '' INT TERM HUP
    child_spawn_deferred_signal_status=0
    child_spawn_deferred_signal_phase=""
    child_signal_handler "${deferred_status}" "${deferred_phase}"
  fi
  if (( status != 0 )); then
    if kill -0 "${child_owned_pid}" 2>/dev/null; then
      # A PID without its original start-time never authorizes a numeric
      # signal.  The exact Slurm step cgroup is the only safe fallback.
      terminate_owned_step_cgroup_bounded || true
      child_owned_pid=""
      child_owned_starttime=""
      return 71
    fi
    set +e
    wait "${child_owned_pid}"
    status=$?
    if [[ "${restore_errexit}" == true ]]; then set -e; else set +e; fi
    child_owned_pid=""
    child_owned_starttime=""
    return "${status}"
  fi
  while true; do
    set +e
    wait "${child_owned_pid}"
    status=$?
    if [[ "${restore_errexit}" == true ]]; then set -e; else set +e; fi
    pid_is_same_process "${child_owned_pid}" "${child_owned_starttime}" || break
    terminate_owned_pid_bounded "${child_owned_pid}" "${child_owned_starttime}" child-foreground-command
    status=71
    break
  done
  child_owned_pid=""
  child_owned_starttime=""
  return "${status}"
}

run_child_required() {
  local label status restore_errexit target
  [[ $# -ge 2 ]] || fail "child application arguments differ"
  label="$1"
  target="$2"
  shift 2
  [[ "${target}" == "${method_root}"/*.py || "${target}" == "${method_root}"/tools/*.py ]] || fail "child application target is not runner-bound"
  restore_errexit=false
  [[ $- == *e* ]] && restore_errexit=true
  set +e
  run_child_owned_command "${frozen_python_exec_prefix[@]}" -S -s -P -B -c "${materialization_bootstrap_py}" \
    "${method_root}" "${method_manifest}" "${core_manifest_sha}" \
    "${core_manifest_digest}" "${method_revision}" "${target}" "$@"
  status=$?
  if [[ "${restore_errexit}" == true ]]; then set -e; else set +e; fi
  (( status == 0 )) || fail_status "${status}" "${label}"
}

write_child_launcher_failure_receipt() {
  local status="$1" receipt signed_path signed_sha
  [[ -d "${run_root}/logs" && ! -L "${run_root}/logs" ]] || return 0
  receipt="${run_root}/logs/child-launcher-failure.status"
  [[ ! -e "${receipt}" && ! -L "${receipt}" ]] || return 0
  signed_path="${run_root}/logs/child-scratch-failure.json"
  signed_sha=unavailable
  [[ -f "${signed_path}" && ! -L "${signed_path}" ]] && signed_sha="$(sha256_file "${signed_path}" 2>/dev/null || printf unavailable)"
  (
    set -o noclobber
    umask 077
    printf '%s\n' \
      'schema=bernini-full30-action-arms-incomplete-exact2-child-failure-v1' \
      "phase=${child_failure_phase}" \
      "exit_status=${status}" \
      'retention_claim=see-signed-receipt' \
      'retention_not_claimed=true' \
      'scratch_reusable=false' \
      'manual_scratch_cleanup_authorized=false' \
      "signed_failure_receipt=${signed_path}" \
      "signed_failure_receipt_sha256=${signed_sha}" >"${receipt}"
    chmod 0400 "${receipt}"
  ) 2>/dev/null || true
}

seal_child_failure_best_effort() {
  local status="$1" signed_failure failure_output_status failure_pid failure_start failure_state attempt
  [[ -n "${scratch_prepare}" && -f "${scratch_prepare}" && "${scratch_prepare_sha}" =~ ^[0-9a-f]{64}$ ]] || { write_child_launcher_failure_receipt "${status}"; return 0; }
  signed_failure="${run_root}/logs/child-scratch-failure.json"
  [[ ! -e "${signed_failure}" && ! -L "${signed_failure}" ]] || { write_child_launcher_failure_receipt "${status}"; return 0; }
  verify_materialized_method_before_app_import >/dev/null 2>&1 || { write_child_launcher_failure_receipt "${status}"; return 0; }
  failure_args=(
    "${frozen_python_exec_prefix[@]}" -S -s -P -B -c "${materialization_bootstrap_py}"
    "${method_root}" "${method_manifest}" "${core_manifest_sha}"
    "${core_manifest_digest}" "${method_revision}"
    "${controller}" seal-child-scratch-failure
    --scratch-prepare "${scratch_prepare}"
    --expected-scratch-prepare-sha256 "${scratch_prepare_sha}"
    --failure-phase "${child_failure_phase}"
    --exit-status "${status}"
    --output "${signed_failure}"
  )
  if [[ -n "${compute_preflight}" && -f "${compute_preflight}" && "${compute_preflight_sha}" =~ ^[0-9a-f]{64}$ ]]; then
    failure_args+=(--compute-preflight "${compute_preflight}" --expected-compute-preflight-sha256 "${compute_preflight_sha}")
  fi
  if [[ -n "${task_scratch_binding}" && -f "${task_scratch_binding}" && "${task_scratch_binding_sha}" =~ ^[0-9a-f]{64}$ ]]; then
    failure_args+=(--task-scratch-bind "${task_scratch_binding}" --expected-task-scratch-bind-sha256 "${task_scratch_binding_sha}")
  fi
  "${failure_args[@]}" >/dev/null 2>&1 &
  failure_pid=$!
  failure_start="$(pid_starttime "${failure_pid}" 2>/dev/null)" || failure_start=""
  failure_output_status=124
  for attempt in $(seq 1 100); do
    if [[ -z "${failure_start}" ]] || ! pid_is_same_process "${failure_pid}" "${failure_start}"; then
      set +e; wait "${failure_pid}" 2>/dev/null; failure_output_status=$?; set -e
      break
    fi
    failure_state="$(pid_state "${failure_pid}" 2>/dev/null)" || failure_state=""
    if [[ "${failure_state}" == Z ]]; then
      set +e; wait "${failure_pid}" 2>/dev/null; failure_output_status=$?; set -e
      break
    fi
    sleep 0.1
  done
  if (( failure_output_status == 124 )); then
    if [[ -n "${failure_start}" ]]; then
      terminate_owned_pid_bounded "${failure_pid}" "${failure_start}" child-failure-receipt || true
    else
      terminate_owned_step_cgroup_bounded || true
    fi
  fi
  (( failure_output_status == 0 )) || true
  write_child_launcher_failure_receipt "${status}"
}

child_exit_handler() {
  local status=$?
  trap - EXIT
  trap '' INT TERM HUP
  if (( status == 0 )) && [[ "${child_terminal_ready_committed}" != true ]]; then
    status=70
  fi
  if (( status != 0 )); then
    if [[ -n "${child_owned_pid}" ]]; then
      if [[ -n "${child_owned_starttime}" ]]; then
        terminate_owned_pid_bounded "${child_owned_pid}" "${child_owned_starttime}" child-foreground-command || true
      else
        echo "[arms-incomplete-exact2-r6] ERROR: child command PID identity unavailable; numeric signal forbidden" >&2
      fi
    fi
    if [[ -n "${host_memory_monitor_pid}" ]]; then
      if [[ -n "${host_memory_monitor_starttime}" ]]; then
        terminate_owned_pid_bounded "${host_memory_monitor_pid}" "${host_memory_monitor_starttime}" host-memory-monitor || true
      else
        echo "[arms-incomplete-exact2-r6] ERROR: host monitor PID identity unavailable; numeric signal forbidden" >&2
      fi
    fi
    terminate_owned_step_cgroup_bounded || true
    seal_child_failure_best_effort "${status}"
  fi
  exit "${status}"
}

child_signal_handler() {
  local status="$1" phase="$2"
  if [[ "${child_spawn_in_progress}" == true ]]; then
    if (( child_spawn_deferred_signal_status == 0 )); then
      child_spawn_deferred_signal_status="${status}"
      child_spawn_deferred_signal_phase="${phase}"
    fi
    return 0
  fi
  child_failure_phase="${phase}"
  trap '' INT TERM HUP
  if [[ -n "${child_owned_pid}" ]]; then
    if [[ -n "${child_owned_starttime}" ]]; then
      terminate_owned_pid_bounded "${child_owned_pid}" "${child_owned_starttime}" child-foreground-command || true
    else
      echo "[arms-incomplete-exact2-r6] ERROR: child command PID identity unavailable; numeric signal forbidden" >&2
    fi
  fi
  if [[ -n "${host_memory_monitor_pid}" ]]; then
    if [[ -n "${host_memory_monitor_starttime}" ]]; then
      terminate_owned_pid_bounded "${host_memory_monitor_pid}" "${host_memory_monitor_starttime}" host-memory-monitor || true
    else
      echo "[arms-incomplete-exact2-r6] ERROR: host monitor PID identity unavailable; numeric signal forbidden" >&2
    fi
  fi
  terminate_owned_step_cgroup_bounded || true
  exit "${status}"
}

child_signal_int() { child_signal_handler 130 signal-int; }
child_signal_term() { child_signal_handler 143 signal-term; }
child_signal_hup() { child_signal_handler 129 signal-hup; }

if [[ "${role}" == child ]]; then
  trap child_exit_handler EXIT
  trap child_signal_int INT
  trap child_signal_term TERM
  trap child_signal_hup HUP
fi

case "${method_root}" in */"${revoked_release_leaf}"|*/"${revoked_release_leaf}"/*) fail "revoked live release is permanent NO-GO (${revoked_live_log_sha})" ;; esac
case "${method_archive}" in */"${revoked_release_leaf}"/*) fail "revoked live release archive is permanent NO-GO" ;; esac
case "${method_manifest}" in */"${revoked_release_leaf}"/*) fail "revoked live release manifest is permanent NO-GO" ;; esac

validate_sealed_inputs_after_compute_preflight() {
  verify_materialized_method_before_app_import || fail "materialized method bootstrap closure differs"
  [[ "$(readlink -f -- "${release_root}")" == "${release_root}" && -d "${release_root}" && ! -L "${release_root}" ]] || fail "release root is not canonical"
  [[ "${method_archive}" == "${release_root}/source.tar" && "${method_manifest}" == "${release_root}/source.manifest.json" && "${deployment_envelope}" == "${release_root}/deployment-envelope.json" && "${detached_launcher}" == "${release_root}/$(basename -- "${detached_launcher}")" ]] || fail "release exact-four paths differ"
  local release_entries expected_release_entries observed_core_archive_sha observed_core_manifest_sha
  local observed_manifest_digest observed_method_revision detached_launcher_basename
  detached_launcher_basename="$(basename -- "${detached_launcher}")" || fail "detached launcher basename is unavailable"
  release_entries="$(find "${release_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)" || fail "release entries cannot be enumerated"
  expected_release_entries="$(printf '%s\n' deployment-envelope.json source.manifest.json source.tar "${detached_launcher_basename}" | LC_ALL=C sort)" || fail "expected release entries cannot be built"
  [[ "${release_entries}" == "${expected_release_entries}" ]] || fail "release root does not contain exact four entries"
  [[ "$(stat -c '%a' -- "${method_archive}")" == 444 && "$(stat -c '%a' -- "${method_manifest}")" == 444 && "$(stat -c '%a' -- "${deployment_envelope}")" == 444 ]] || fail "deployed core/envelope modes differ"
  for path in "${method_archive}" "${method_manifest}" "${deployment_envelope}" "${seed1_spec}" "${seed2_spec}" "${external_key}" "${external_review}" "${python_bin}" "${ffprobe_bin}" "${checkpoint_manifest}" "${r10_receipt}" "${r10_log}" "${controller}" "${plan_tool}" "${generator}" "${resource}" "${release_builder}" "${rank_wrapper}" "${launcher}"; do
    [[ -f "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "sealed input differs: ${path}"
  done
  for path in "${release_root}" "${method_root}" "${external_root}" "${bernini_root}" "${veomni_root}" "${checkpoint}"; do
    [[ -d "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "sealed directory differs: ${path}"
  done
  [[ -x "${python_bin}" ]] || fail "Python is not executable"
  [[ -x "${ffprobe_bin}" && "$(sha256_file "${ffprobe_bin}")" == "${ffprobe_sha}" ]] || fail "portable ffprobe identity differs"
  observed_core_archive_sha="$(sha256_file "${method_archive}")" || fail "core archive SHA-256 is unavailable"
  observed_core_manifest_sha="$(sha256_file "${method_manifest}")" || fail "core manifest SHA-256 is unavailable"
  [[ "${observed_core_archive_sha}" != "${revoked_portable_r2_archive_sha}" && "${observed_core_manifest_sha}" != "${revoked_portable_r2_manifest_sha}" ]] || fail "revoked portable r2 core is permanent NO-GO even if renamed"
  [[ "${observed_core_archive_sha}" != "${revoked_terminal_r3_archive_sha}" && "${observed_core_manifest_sha}" != "${revoked_terminal_r3_manifest_sha}" ]] || fail "revoked terminal-physical r3 core is permanent NO-GO even if renamed"
  [[ "${observed_core_archive_sha}" != "${revoked_resource_reuse_r4_archive_sha}" && "${observed_core_manifest_sha}" != "${revoked_resource_reuse_r4_manifest_sha}" ]] || fail "revoked resource-reuse r4 core is permanent NO-GO even if renamed"
  [[ "${observed_core_archive_sha}" != "${revoked_canonical_python_r5_archive_sha}" && "${observed_core_manifest_sha}" != "${revoked_canonical_python_r5_manifest_sha}" ]] || fail "revoked canonical-Python r5 core is permanent NO-GO even if renamed"
  [[ "${observed_core_archive_sha}" == "${core_archive_sha}" ]] || fail "literal-pinned core archive SHA differs"
  [[ "${observed_core_manifest_sha}" == "${core_manifest_sha}" ]] || fail "literal-pinned core manifest SHA differs"
  observed_manifest_digest="$(stable_json_field "${method_manifest}" "${core_manifest_sha}" manifest_digest)" || fail "manifest digest cannot be read"
  observed_method_revision="$(stable_json_field "${method_manifest}" "${core_manifest_sha}" content_closure_sha1)" || fail "manifest revision cannot be read"
  [[ "${observed_manifest_digest}" == "${core_manifest_digest}" && "${observed_method_revision}" == "${method_revision}" ]] || fail "literal-pinned manifest digest/revision differs"
  [[ "$(sha256_file "${deployment_envelope}")" == "${deployment_envelope_sha}" ]] || fail "deployment envelope SHA differs"
  [[ "$(sha256_file "${plan_tool}")" == "${owned_plan_sha}" && "$(sha256_file "${generator}")" == "${owned_generator_sha}" && "$(sha256_file "${controller}")" == "${owned_controller_sha}" && "$(sha256_file "${release_builder}")" == "${owned_builder_sha}" ]] || fail "literal-pinned owned business component differs"
  [[ "$(sha256_file "${rank_wrapper}")" == "${owned_rank_wrapper_sha}" && -x "${rank_wrapper}" ]] || fail "literal-pinned rank wrapper differs"
  [[ "$(sha256_file "${external_key}")" == 4c0864c7018b28b284a49d7134bce574e8d8fe47d5d795a71497b78fff446f8c ]] || fail "external sealed key differs"
  [[ "$(sha256_file "${external_review}")" == 1b40da8dde07f348c2501adf3fd62fb528062053cde6e99c62f6d02e3ad8a4bc ]] || fail "external blind review differs"
  [[ "$(sha256_file "${resource}")" == f85ab3c5072d4f0a9cbf2084ac724589f6346430422197a719e4968394c5987c ]] || fail "frozen resource differs"
  [[ "$(sha256_file "${seed1_spec}")" == 2861b1021531896d387b0dccb945b9fc2516bf01472982c0fe2f7c1377ca7bab ]] || fail "seed1 source spec differs"
  [[ "$(sha256_file "${seed2_spec}")" == 0578cd6c39cdb625e69cf04164ffce29487b81e852974619f0ee43325e49398e ]] || fail "seed2 source spec differs"
  [[ "$(sha256_file "${r10_receipt}")" == "${r10_receipt_sha}" ]] || fail "r10 receipt SHA differs"
  [[ "$(sha256_file "${r10_log}")" == "${r10_log_sha}" ]] || fail "r10 log SHA differs"
  run_verified_materialized_app "${release_builder}" audit \
    --archive "${method_archive}" --expected-archive-sha256 "${core_archive_sha}" \
    --manifest "${method_manifest}" --expected-manifest-sha256 "${core_manifest_sha}" \
    --detached-launcher "${detached_launcher}" --expected-detached-launcher-sha256 "${detached_launcher_sha}" \
    --deployment-envelope "${deployment_envelope}" --expected-deployment-envelope-sha256 "${deployment_envelope_sha}" >/dev/null || fail "release deployment audit failed"
  validate_frozen_python_before_run_root_or_srun
}

if [[ "${role}" == parent ]]; then
  validate_sealed_inputs_after_compute_preflight
fi

all_holder_step_ids() {
  local raw line output=""
  raw="$(/usr/bin/squeue -s -j "${holder_job}" -h -o '%i')" || return 1
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    case "${line}" in
      "${holder_job}.batch"|"${holder_job}.extern") ;;
      *) [[ "${line}" =~ ^${holder_job}[.][0-9]+$ ]] || return 1 ;;
    esac
    output+="${line}"$'\n'
  done <<<"${raw}"
  printf '%s' "${output}"
}

numbered_steps() {
  local raw line
  raw="$(all_holder_step_ids)" || return 1
  while IFS= read -r line; do
    [[ "${line}" =~ ^${holder_job}[.][0-9]+$ ]] && printf '%s\n' "${line}"
  done <<<"${raw}"
}

assert_holder_batch_extern_exact() {
  local raw nonnumbered=""
  raw="$(all_holder_step_ids)" || return 1
  while IFS= read -r line; do
    case "${line}" in
      "${holder_job}.batch"|"${holder_job}.extern") nonnumbered+="${line}"$'\n' ;;
    esac
  done <<<"${raw}"
  [[ "$(printf '%s' "${nonnumbered}" | /usr/bin/sort)" == "$(printf '%s\n' "${holder_job}.batch" "${holder_job}.extern" | /usr/bin/sort)" ]]
}

assert_all8_idle_local() {
  local snapshot count memory_count busy
  snapshot="$(rocm-smi --showuse --showmemuse --showpids)"
  count="$(awk '/GPU use \(%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  memory_count="$(awk '/GPU Memory Allocated \(VRAM%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  busy="$(awk '/GPU use \(%\)/ || /GPU Memory Allocated \(VRAM%\)/ {v=$NF; gsub(/[^0-9]/,"",v); if((v+0)!=0)print}' <<<"${snapshot}")"
  [[ "${count}" == 8 && "${memory_count}" == 8 && -z "${busy}" ]] || fail "all8 physical GPUs are not idle"
}

if [[ "${role}" == child ]]; then
  [[ "${SLURM_JOB_ID:?Slurm child required}" == "${holder_job}" ]] || fail "child holder differs"
  [[ "$(hostname -s)" == "${holder_node}" ]] || fail "child node differs"
  [[ "${SLURM_STEP_ID:?numbered step required}" =~ ^[0-9]+$ ]] || fail "child step differs"
  [[ "$(numbered_steps)" == "${holder_job}.${SLURM_STEP_ID}" ]] || fail "child is not sole numbered step"
  unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL ROCR_VISIBLE_DEVICES
  [[ -f "${ffprobe_bin}" && ! -L "${ffprobe_bin}" && "$(readlink -f -- "${ffprobe_bin}")" == "${ffprobe_bin}" && -x "${ffprobe_bin}" ]] || fail "portable ffprobe canonical executable differs"
  [[ "$(sha256_file "${ffprobe_bin}")" == "${ffprobe_sha}" ]] || fail "portable ffprobe SHA differs"

  child_phase=prepare
  child_failure_phase=prepare
  scratch_prepare="${run_root}/logs/child-scratch-prepare.json"
  run_child_required "signed child scratch prepare failed" "${controller}" prepare-child-scratch \
    --controller-plan "${F13_CONTROLLER_PLAN}" \
    --expected-controller-plan-sha256 "${F13_CONTROLLER_PLAN_SHA256}" \
    --output "${scratch_prepare}" >/dev/null
  scratch_prepare_sha_value="$(sha256_file "${scratch_prepare}")" || fail "child scratch prepare SHA-256 is unavailable"
  [[ "${scratch_prepare_sha_value}" =~ ^[0-9a-f]{64}$ ]] || fail "child scratch prepare SHA-256 differs"
  scratch_prepare_sha="${scratch_prepare_sha_value}"

  child_phase=compute
  child_failure_phase=compute
  compute_preflight="${run_root}/logs/compute-preflight.json"
  run_child_required "compute portable-ffprobe/scratch preflight failed" "${controller}" seal-compute-preflight \
    --controller-plan "${F13_CONTROLLER_PLAN}" \
    --expected-controller-plan-sha256 "${F13_CONTROLLER_PLAN_SHA256}" \
    --ffprobe-bin "${ffprobe_bin}" --expected-ffprobe-sha256 "${ffprobe_sha}" \
    --external-action-mp4-seed1 "${external_action_mp4_seed1}" \
    --external-action-mp4-seed2 "${external_action_mp4_seed2}" \
    --scratch-prepare "${scratch_prepare}" \
    --expected-scratch-prepare-sha256 "${scratch_prepare_sha}" \
    --output "${compute_preflight}" >/dev/null
  compute_preflight_sha_value="$(sha256_file "${compute_preflight}")" || fail "compute preflight SHA-256 is unavailable"
  [[ "${compute_preflight_sha_value}" =~ ^[0-9a-f]{64}$ ]] || fail "compute preflight SHA-256 differs"
  compute_preflight_sha="${compute_preflight_sha_value}"
  scratch_parent="$(stable_json_field "${scratch_prepare}" "${scratch_prepare_sha}" scratch_root path)" || fail "prepared scratch path cannot be read"
  receipt_scratch_fstype="$(stable_json_field "${scratch_prepare}" "${scratch_prepare_sha}" filesystem raw_filesystem_type)" || fail "prepared scratch filesystem cannot be read"
  [[ "${scratch_parent}" == "/tmp/BOX-EXP-013-r6-${SLURM_JOB_ID}-${SLURM_STEP_ID}" ]] || fail "prepared fixed outer scratch path differs"
  observed_scratch_fstype="$(stat -f -c '%T' -- "${scratch_parent}")" || fail "prepared scratch statfs is unavailable"
  [[ "${observed_scratch_fstype}" == "${receipt_scratch_fstype}" ]] || fail "scratch filesystem changed after preflight receipt"
  validate_sealed_inputs_after_compute_preflight
  assert_all8_idle_local

  child_phase=inner
  child_failure_phase=inner
  task_scratch_binding="${run_root}/logs/child-task-scratch-bind.json"
  run_child_required "create-and-bind child task scratch failed" "${controller}" create-and-bind-child-task-scratch \
    --scratch-prepare "${scratch_prepare}" \
    --expected-scratch-prepare-sha256 "${scratch_prepare_sha}" \
    --compute-preflight "${compute_preflight}" \
    --expected-compute-preflight-sha256 "${compute_preflight_sha}" \
    --output "${task_scratch_binding}" >/dev/null
  task_scratch_binding_sha_value="$(sha256_file "${task_scratch_binding}")" || fail "task scratch binding SHA-256 is unavailable"
  [[ "${task_scratch_binding_sha_value}" =~ ^[0-9a-f]{64}$ ]] || fail "task scratch binding SHA-256 differs"
  task_scratch_binding_sha="${task_scratch_binding_sha_value}"
  task_scratch="$(stable_json_field "${task_scratch_binding}" "${task_scratch_binding_sha}" scratch_inner path)" || fail "task scratch path cannot be read"
  [[ "${task_scratch}" == "${scratch_parent}"/arms-incomplete-exact2-"${SLURM_JOB_ID}"-"${SLURM_STEP_ID}".* ]] || fail "bound inner scratch path differs"
  run_child_required "initial task scratch binding replay failed" "${controller}" validate-child-task-scratch-bind \
    --scratch-prepare "${scratch_prepare}" \
    --expected-scratch-prepare-sha256 "${scratch_prepare_sha}" \
    --task-scratch-bind "${task_scratch_binding}" \
    --expected-task-scratch-bind-sha256 "${task_scratch_binding_sha}" >/dev/null
  export GADP_NODE_LOCAL_SCRATCH="${task_scratch}" GADP_NODE_LOCAL_SCRATCH_FSTYPE="${receipt_scratch_fstype}" TMPDIR="${task_scratch}"
  model_load_lock="$(stable_json_field "${task_scratch_binding}" "${task_scratch_binding_sha}" renderer_load_lock path)" || fail "renderer load lock binding path cannot be read"
  [[ "${model_load_lock}" == "${task_scratch}/renderer-load.lock" ]] || fail "renderer load lock binding path differs"
  export NATIVE_SERIALIZED_HOST_LOAD_REQUIRED=1 NATIVE_V_AXIS_LOAD_LOCK="${model_load_lock}" NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED=1

  child_phase=monitor
  child_failure_phase=monitor
  journal="${run_root}/logs/host-cgroup-memory-current-samples.bin"
  monitor_start="${run_root}/logs/host-cgroup-memory-monitor-start.json"
  monitor_stop="${task_scratch}/host-memory-monitor-stop"
  child_spawn_in_progress=true
  "${frozen_python_exec_prefix[@]}" -S -s -P -B -c "${materialization_bootstrap_py}" \
    "${method_root}" "${method_manifest}" "${core_manifest_sha}" \
    "${core_manifest_digest}" "${method_revision}" "${resource}" host-memory-monitor \
    --sample-journal "${journal}" --start-receipt-output "${monitor_start}" \
    --stop-path "${monitor_stop}" --supervisor-pid "$$" \
    --slurm-job-id "${SLURM_JOB_ID}" --slurm-step-id "${SLURM_STEP_ID}" &
  host_memory_monitor_pid=$!
  host_memory_monitor_starttime_value="$(pid_starttime "${host_memory_monitor_pid}")"
  host_memory_monitor_starttime_status=$?
  if (( host_memory_monitor_starttime_status == 0 )); then host_memory_monitor_starttime="${host_memory_monitor_starttime_value}"; else host_memory_monitor_starttime=""; fi
  deferred_status="${child_spawn_deferred_signal_status}"
  deferred_phase="${child_spawn_deferred_signal_phase}"
  if (( deferred_status != 0 )); then
    trap '' INT TERM HUP
    child_spawn_in_progress=false
    child_spawn_deferred_signal_status=0
    child_spawn_deferred_signal_phase=""
    child_signal_handler "${deferred_status}" "${deferred_phase}"
  fi
  child_spawn_in_progress=false
  deferred_status="${child_spawn_deferred_signal_status}"
  deferred_phase="${child_spawn_deferred_signal_phase}"
  if (( child_spawn_deferred_signal_status != 0 )); then
    trap '' INT TERM HUP
    child_spawn_deferred_signal_status=0
    child_spawn_deferred_signal_phase=""
    child_signal_handler "${deferred_status}" "${deferred_phase}"
  fi
  if (( host_memory_monitor_starttime_status != 0 )); then
    if kill -0 "${host_memory_monitor_pid}" 2>/dev/null; then
      terminate_owned_step_cgroup_bounded || true
    else
      wait "${host_memory_monitor_pid}" 2>/dev/null || true
    fi
    host_memory_monitor_pid=""
    fail "host monitor PID identity is unavailable"
  fi
  export GADP_HOST_MEMORY_SAMPLE_JOURNAL="${journal}" GADP_HOST_MEMORY_MONITOR_START_RECEIPT="${monitor_start}"
  export GADP_HOST_MEMORY_MONITOR_PID="${host_memory_monitor_pid}" GADP_HOST_MEMORY_MONITOR_SUPERVISOR_PID="$$"
  monitor_ready=false
  for _ in $(seq 1 2000); do
    kill -0 "${host_memory_monitor_pid}" 2>/dev/null || fail "host monitor exited before readiness"
    if [[ -f "${monitor_start}" ]] \
      && run_child_owned_command "${frozen_python_exec_prefix[@]}" -S -s -P -B \
        -c "${materialization_bootstrap_py}" \
        "${method_root}" "${method_manifest}" "${core_manifest_sha}" \
        "${core_manifest_digest}" "${method_revision}" \
        "${resource}" assert-host-memory-monitor-live >/dev/null 2>&1; then
      monitor_ready=true
      break
    fi
    sleep 0.01
  done
  [[ "${monitor_ready}" == true ]] || fail "10ms host monitor did not become live"

  child_phase=smoke
  child_failure_phase=smoke
  run_child_required "pre-smoke task scratch binding replay failed" "${controller}" validate-child-task-scratch-bind \
    --scratch-prepare "${scratch_prepare}" --expected-scratch-prepare-sha256 "${scratch_prepare_sha}" \
    --task-scratch-bind "${task_scratch_binding}" --expected-task-scratch-bind-sha256 "${task_scratch_binding_sha}" >/dev/null
  smoke_plan_root="${run_root}/resource-smoke-plan"
  smoke_plan_stdout="${run_root}/logs/resource-smoke-plan.stdout.json"
  [[ ! -e "${smoke_plan_stdout}" && ! -L "${smoke_plan_stdout}" ]] || fail "resource smoke plan stdout path is not fresh"
  run_child_required "resource smoke plan build failed" "${resource}" build-plan --seed1-spec "${seed1_spec}" --seed2-spec "${seed2_spec}" --split fit --output-dir "${smoke_plan_root}" >"${smoke_plan_stdout}"
  [[ -f "${smoke_plan_stdout}" && ! -L "${smoke_plan_stdout}" ]] || fail "resource smoke plan stdout is unavailable"
  smoke_plan_json="$(< "${smoke_plan_stdout}")" || fail "resource smoke plan stdout cannot be read"
  smoke_plan="$(run_frozen_python -S -s -P -B -c 'import json,sys; print(json.loads(sys.argv[1])["plan_path"])' "${smoke_plan_json}")" || fail "resource smoke plan path cannot be read"
  smoke_plan_sha="$(run_frozen_python -S -s -P -B -c 'import json,sys; print(json.loads(sys.argv[1])["plan_file_sha256"])' "${smoke_plan_json}")" || fail "resource smoke plan SHA-256 cannot be read"
  smoke_receipt="${run_root}/logs/resource-compile-smoke-receipt.json"
  smoke_host_gate="${run_root}/logs/resource-compile-smoke-host-gate.json"
  export ROCR_VISIBLE_DEVICES=0,1,2,3
  run_child_required "resource smoke failed" "${resource}" smoke-sp4 \
    --plan "${smoke_plan}" --expected-plan-sha256 "${smoke_plan_sha}" \
    --python "${python_bin}" --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
    --method-source-revision "${method_revision}" --method-source-archive-sha256 "${core_archive_sha}" \
    --master-port "${master_port}" --receipt-output "${smoke_receipt}" \
    --host-memory-gate-output "${smoke_host_gate}" \
    --r10-compile-smoke-receipt "${r10_receipt}" --expected-r10-compile-smoke-receipt-sha256 "${r10_receipt_sha}" \
    --r10-generation-log "${r10_log}" --expected-r10-generation-log-sha256 "${r10_log_sha}"
  smoke_receipt_sha="$(sha256_file "${smoke_receipt}")" || fail "resource smoke receipt SHA-256 is unavailable"

  child_phase=generation
  child_failure_phase=generation
  assert_all8_idle_local
  run_child_required "pre-generation task scratch binding replay failed" "${controller}" validate-child-task-scratch-bind \
    --scratch-prepare "${scratch_prepare}" --expected-scratch-prepare-sha256 "${scratch_prepare_sha}" \
    --task-scratch-bind "${task_scratch_binding}" --expected-task-scratch-bind-sha256 "${task_scratch_binding_sha}" >/dev/null
  export ROCR_VISIBLE_DEVICES=0,1,2,3
  run_child_required "exact2 generation failed" "${generator}" run-sp4 \
    --controller-plan "${F13_CONTROLLER_PLAN}" --expected-controller-plan-sha256 "${F13_CONTROLLER_PLAN_SHA256}" \
    --scratch-prepare "${scratch_prepare}" --expected-scratch-prepare-sha256 "${scratch_prepare_sha}" \
    --plan "${F13_EXACT2_PLAN}" --expected-plan-sha256 "${F13_EXACT2_PLAN_SHA256}" \
    --resource-contract "${resource}" --resource-compile-smoke-receipt "${smoke_receipt}" \
    --expected-resource-compile-smoke-receipt-sha256 "${smoke_receipt_sha}" \
    --group-id sp4-a --python "${python_bin}" --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" --checkpoint "${checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --method-source-revision "${method_revision}" --method-source-archive-sha256 "${core_archive_sha}" \
    --master-port "${master_port}" \
    --compute-preflight "${compute_preflight}" --expected-compute-preflight-sha256 "${compute_preflight_sha}" \
    --task-scratch-bind "${task_scratch_binding}" --expected-task-scratch-bind-sha256 "${task_scratch_binding_sha}" \
    --output-dir "${run_root}/generation/sp4-a"
  unset ROCR_VISIBLE_DEVICES
  child_phase=audit
  child_failure_phase=audit
  run_child_required "formal exact2 audit failed" "${generator}" audit-exact2 \
    --controller-plan "${F13_CONTROLLER_PLAN}" --expected-controller-plan-sha256 "${F13_CONTROLLER_PLAN_SHA256}" \
    --scratch-prepare "${scratch_prepare}" --expected-scratch-prepare-sha256 "${scratch_prepare_sha}" \
    --plan "${F13_EXACT2_PLAN}" --expected-plan-sha256 "${F13_EXACT2_PLAN_SHA256}" \
    --generation-root "${run_root}/generation/sp4-a" \
    --compute-preflight "${compute_preflight}" --expected-compute-preflight-sha256 "${compute_preflight_sha}" \
    --task-scratch-bind "${task_scratch_binding}" --expected-task-scratch-bind-sha256 "${task_scratch_binding_sha}" \
    --output "${run_root}/generation-exact2-audit.json" \
    --gap-output "${run_root}/generation-exact2-gap.json" >/dev/null
  generation_audit_sha="$(sha256_file "${run_root}/generation-exact2-audit.json")" || fail "generation audit SHA-256 is unavailable"
  child_phase=blind
  child_failure_phase=blind
  run_child_required "blind review input sealing failed" "${controller}" seal-blind-review-input \
    --controller-plan "${F13_CONTROLLER_PLAN}" \
    --expected-controller-plan-sha256 "${F13_CONTROLLER_PLAN_SHA256}" \
    --generation-audit "${run_root}/generation-exact2-audit.json" \
    --expected-generation-audit-sha256 "${generation_audit_sha}" \
    --ffprobe-bin "${ffprobe_bin}" --expected-ffprobe-sha256 "${ffprobe_sha}" \
    --output-dir "${run_root}/blind-review-packet" >/dev/null

  mkdir -m 0700 "${monitor_stop}"
  set +e
  wait "${host_memory_monitor_pid}"
  monitor_status=$?
  set -e
  host_memory_monitor_pid_before_clear="${host_memory_monitor_pid}"
  host_memory_monitor_pid=""
  host_memory_monitor_starttime=""
  [[ "${monitor_status}" == 0 ]] || fail_status "${monitor_status}" "host monitor failed"
  child_phase=terminal
  child_failure_phase=terminal
  monitor_start_sha="$(sha256_file "${monitor_start}")" || fail "monitor-start SHA-256 is unavailable"
  terminal_host_gate="${run_root}/logs/terminal-arms-incomplete-exact2-host-gate.json"
  run_child_required "terminal host gate failed" "${controller}" seal-terminal-host-gate \
    --compute-preflight "${compute_preflight}" \
    --expected-compute-preflight-sha256 "${compute_preflight_sha}" \
    --resource-contract "${resource}" \
    --expected-resource-contract-sha256 f85ab3c5072d4f0a9cbf2084ac724589f6346430422197a719e4968394c5987c \
    --monitor-start-receipt "${monitor_start}" \
    --expected-monitor-start-receipt-sha256 "${monitor_start_sha}" \
    --monitor-exit-status "${monitor_status}" \
    --output "${terminal_host_gate}" >/dev/null
  terminal_host_gate_sha="$(sha256_file "${terminal_host_gate}")" || fail "terminal host-gate SHA-256 is unavailable"
  remaining_jobs="$(jobs -pr)" || fail "background job census failed"
  [[ -z "${remaining_jobs}" ]] || fail "background process remained"
  collect_owned_step_cgroup_pids || fail "numbered-step cgroup census failed"
  (( ${#owned_step_cgroup_tokens[@]} == 0 )) || fail "numbered-step cgroup descendant remained before terminal attestation"

  child_phase=attestation
  child_failure_phase=attestation
  blind_manifest="${run_root}/blind-review-packet/reviewer/review-manifest.json"
  blind_key="${run_root}/blind-review-packet/sealed-key.json"
  blind_manifest_sha="$(sha256_file "${blind_manifest}")" || fail "blind manifest SHA-256 is unavailable"
  blind_key_sha="$(sha256_file "${blind_key}")" || fail "blind key SHA-256 is unavailable"
  physical_attestation="${run_root}/logs/child-terminal-physical-attestation.json"
  run_child_required "terminal physical attestation failed" "${controller}" seal-child-terminal-physical-attestation \
    --controller-plan "${F13_CONTROLLER_PLAN}" --expected-controller-plan-sha256 "${F13_CONTROLLER_PLAN_SHA256}" \
    --generation-audit "${run_root}/generation-exact2-audit.json" --expected-generation-audit-sha256 "${generation_audit_sha}" \
    --terminal-host-gate "${terminal_host_gate}" --expected-terminal-host-gate-sha256 "${terminal_host_gate_sha}" \
    --blind-review-manifest "${blind_manifest}" --expected-blind-review-manifest-sha256 "${blind_manifest_sha}" \
    --blind-review-key "${blind_key}" --expected-blind-review-key-sha256 "${blind_key_sha}" \
    --scratch-prepare "${scratch_prepare}" --expected-scratch-prepare-sha256 "${scratch_prepare_sha}" \
    --compute-preflight "${compute_preflight}" --expected-compute-preflight-sha256 "${compute_preflight_sha}" \
    --task-scratch-bind "${task_scratch_binding}" --expected-task-scratch-bind-sha256 "${task_scratch_binding_sha}" \
    --ffprobe-bin "${ffprobe_bin}" --expected-ffprobe-sha256 "${ffprobe_sha}" \
    --supervisor-pid "$$" --output "${physical_attestation}" >/dev/null
  physical_attestation_sha="$(sha256_file "${physical_attestation}")" || fail "terminal physical attestation SHA-256 is unavailable"

  child_phase=retained-terminal
  child_failure_phase=retained-terminal
  scratch_retained_terminal="${run_root}/logs/child-scratch-retained-terminal.json"
  run_child_required "terminal retained-scratch sealing failed" "${controller}" seal-child-scratch-retained-terminal \
    --controller-plan "${F13_CONTROLLER_PLAN}" --expected-controller-plan-sha256 "${F13_CONTROLLER_PLAN_SHA256}" \
    --scratch-prepare "${scratch_prepare}" --expected-scratch-prepare-sha256 "${scratch_prepare_sha}" \
    --compute-preflight "${compute_preflight}" --expected-compute-preflight-sha256 "${compute_preflight_sha}" \
    --task-scratch-bind "${task_scratch_binding}" --expected-task-scratch-bind-sha256 "${task_scratch_binding_sha}" \
    --generation-audit "${run_root}/generation-exact2-audit.json" --expected-generation-audit-sha256 "${generation_audit_sha}" \
    --terminal-host-gate "${terminal_host_gate}" --expected-terminal-host-gate-sha256 "${terminal_host_gate_sha}" \
    --physical-attestation "${physical_attestation}" --expected-physical-attestation-sha256 "${physical_attestation_sha}" \
    --supervisor-pid "$$" \
    --output "${scratch_retained_terminal}" >/dev/null
  scratch_retained_terminal_sha="$(sha256_file "${scratch_retained_terminal}")" || exit 70
  [[ "${scratch_retained_terminal_sha}" =~ ^[0-9a-f]{64}$ ]] || exit 70
  terminal_ready_marker="${run_root}/logs/child-terminal-ready.status"
  run_child_required "durable child terminal-ready sealing failed" "${controller}" seal-child-terminal-ready \
    --controller-plan "${F13_CONTROLLER_PLAN}" --expected-controller-plan-sha256 "${F13_CONTROLLER_PLAN_SHA256}" \
    --generation-audit "${run_root}/generation-exact2-audit.json" --expected-generation-audit-sha256 "${generation_audit_sha}" \
    --terminal-host-gate "${terminal_host_gate}" --expected-terminal-host-gate-sha256 "${terminal_host_gate_sha}" \
    --physical-attestation "${physical_attestation}" --expected-physical-attestation-sha256 "${physical_attestation_sha}" \
    --scratch-retained-terminal "${scratch_retained_terminal}" --expected-scratch-retained-terminal-sha256 "${scratch_retained_terminal_sha}" \
    --blind-review-manifest "${blind_manifest}" --expected-blind-review-manifest-sha256 "${blind_manifest_sha}" \
    --blind-review-key "${blind_key}" --expected-blind-review-key-sha256 "${blind_key_sha}" \
    --output "${terminal_ready_marker}" >/dev/null
  terminal_ready_sha="$(sha256_file "${terminal_ready_marker}")" || exit 70
  [[ "${terminal_ready_sha}" =~ ^[0-9a-f]{64}$ ]] || exit 70
  child_terminal_ready_committed=true
  exit 0
fi

[[ $# == 0 ]] || fail "parent launcher takes no arguments"
[[ ! -e "${run_root}" && ! -L "${run_root}" && -d "$(dirname -- "${run_root}")" ]] || fail "run root must be fresh"
parent_state="$(squeue -j "${holder_job}" -h -o '%T|%N|%u' | sort -u)" || fail "retained parent state is unavailable"
[[ "${parent_state}" == "RUNNING|${holder_node}|guangyi.chen" ]] || fail "retained parent differs"
initial_numbered_steps="$(numbered_steps)" || fail "initial numbered-step census failed"
[[ -z "${initial_numbered_steps}" ]] || fail "holder already has a numbered child"
assert_holder_batch_extern_exact || fail "holder batch/extern baseline differs"
mkdir -m 0700 "${run_root}" "${run_root}/logs" "${run_root}/generation"

verify_materialized_method_before_app_import || fail "materialized method changed before exact2 plan build"
exact2_json="$(run_verified_materialized_app "${plan_tool}" build-plan \
  --seed1-spec "${seed1_spec}" --expected-seed1-spec-sha256 2861b1021531896d387b0dccb945b9fc2516bf01472982c0fe2f7c1377ca7bab \
  --seed2-spec "${seed2_spec}" --expected-seed2-spec-sha256 0578cd6c39cdb625e69cf04164ffce29487b81e852974619f0ee43325e49398e \
  --external-key "${external_key}" --external-review-receipt "${external_review}" \
  --external-evidence-root "${external_root}" --output-dir "${run_root}/exact2-plan")" || fail "exact2 plan build failed"
exact2_plan="$(run_frozen_python -S -s -P -B -c 'import json,sys; print(json.loads(sys.argv[1])["plan_path"])' "${exact2_json}")" || fail "exact2 plan path cannot be read"
exact2_plan_sha="$(run_frozen_python -S -s -P -B -c 'import json,sys; print(json.loads(sys.argv[1])["plan_file_sha256"])' "${exact2_json}")" || fail "exact2 plan SHA-256 cannot be read"
readonly controller_plan="${run_root}/controller-plan.json"
verify_materialized_method_before_app_import || fail "materialized method changed before controller plan build"
run_verified_materialized_app "${controller}" plan \
  --method-root "${method_root}" --release-manifest "${method_manifest}" \
  --expected-release-manifest-sha256 "${core_manifest_sha}" \
  --exact2-plan "${exact2_plan}" --expected-exact2-plan-sha256 "${exact2_plan_sha}" \
  --output "${controller_plan}" >/dev/null || fail "controller plan failed"
controller_plan_sha="$(sha256_file "${controller_plan}")" || fail "controller plan SHA-256 is unavailable"

sole_numbered_step() {
  local capture status
  local -a steps=()
  capture="$(numbered_steps)"
  status=$?
  (( status == 0 )) || return 1
  [[ -n "${capture}" ]] || return 3
  mapfile -t steps <<<"${capture}"
  (( ${#steps[@]} == 1 )) || return 1
  [[ "${steps[0]}" =~ ^${holder_job}[.][0-9]+$ ]] || return 1
  printf '%s\n' "${steps[0]}"
}

validate_exact_numbered_step_authority() {
  local step="$1" record expected_name field tres="" observed_tres expected_tres
  [[ "${step}" =~ ^${holder_job}[.][0-9]+$ ]] || return 1
  expected_name="BOX-EXP-013-r6-${controller_plan_sha}"
  record="$(/usr/bin/scontrol show step -o "${step}")" || return 1
  [[ -n "${record}" && "${record}" != *$'\n'* ]] || return 1
  [[ " ${record} " == *" StepId=${step} "* ]] || return 1
  [[ " ${record} " == *" UserId=2012 "* ]] || return 1
  [[ " ${record} " == *" NodeList=${holder_node} "* ]] || return 1
  [[ " ${record} " == *" Nodes=1 "* && " ${record} " == *" CPUs=32 "* && " ${record} " == *" Tasks=1 "* ]] || return 1
  [[ " ${record} " == *" Name=${expected_name} "* ]] || return 1
  [[ " ${record} " == *" SrunHost:Pid=${holder_login_host}:${parent_srun_pid} "* ]] || return 1
  [[ " ${record} " == *" State=RUNNING "* \
     || " ${record} " == *" State=COMPLETING "* \
     || " ${record} " == *" State=CANCELLED "* \
     || " ${record} " == *" State=FAILED "* \
     || " ${record} " == *" State=OUT_OF_MEMORY "* ]] || return 1
  for field in ${record}; do
    case "${field}" in TRES=*) [[ -z "${tres}" ]] || return 1; tres="${field#TRES=}" ;; esac
  done
  [[ -n "${tres}" ]] || return 1
  observed_tres="$(printf '%s' "${tres}" | /usr/bin/tr ',' '\n' | /usr/bin/sort)" || return 1
  expected_tres="$(printf '%s\n' cpu=32 gres/gpu:mi210=8 gres/gpu=8 mem=60G node=1 | /usr/bin/sort)" || return 1
  [[ "${observed_tres}" == "${expected_tres}" ]] || return 1
}

cancel_exact_numbered_step_if_present() {
  local step observed attempt
  observed="$(numbered_steps)" || return 1
  [[ -n "${observed}" ]] || { assert_holder_batch_extern_exact; return; }
  step="$(sole_numbered_step)" || return 1
  validate_exact_numbered_step_authority "${step}" || return 1
  # Reopen the exact authority immediately before each narrowly scoped action.
  validate_exact_numbered_step_authority "${step}" || return 1
  /usr/bin/scancel --signal=TERM -- "${step}" || return 1
  for attempt in $(seq 1 50); do
    observed="$(numbered_steps)" || return 1
    if [[ -z "${observed}" ]]; then assert_holder_batch_extern_exact; return; fi
    [[ "${observed}" == "${step}" ]] || return 1
    sleep 0.1
  done
  validate_exact_numbered_step_authority "${step}" || return 1
  /usr/bin/scancel --signal=KILL -- "${step}" || return 1
  for attempt in $(seq 1 50); do
    observed="$(numbered_steps)" || return 1
    if [[ -z "${observed}" ]]; then assert_holder_batch_extern_exact; return; fi
    [[ "${observed}" == "${step}" ]] || return 1
    sleep 0.1
  done
  return 1
}

assert_retained_parent_unchanged() {
  local observed
  observed="$(/usr/bin/squeue -j "${holder_job}" -h -o '%T|%N|%u' | /usr/bin/sort -u)" || return 1
  [[ "${observed}" == "RUNNING|${holder_node}|guangyi.chen" ]] || return 1
  assert_holder_batch_extern_exact
}

run_parent_owned_command() {
  local status starttime_value starttime_status deferred_status gate_ready gate_ready_status
  local gate_tag gate_pid gate_start gate_extra
  [[ "${role}" == parent && -z "${parent_owned_pid}" ]] || return 71
  parent_spawn_in_progress=true
  coproc F13_PARENT_COMMAND_GATE {
    exec "${frozen_python_exec_prefix[@]}" -S -s -P -B -c "${exact_parent_command_exec_gate_py}" \
      "${python_bin}" "${frozen_python_sha}" "${frozen_python_size}" "$@"
  }
  parent_owned_pid="${F13_PARENT_COMMAND_GATE_PID}"
  parent_command_gate_read_fd="${F13_PARENT_COMMAND_GATE[0]}"
  parent_command_gate_write_fd="${F13_PARENT_COMMAND_GATE[1]}"
  gate_ready=""
  set +e
  IFS= read -r -t 35 -u "${parent_command_gate_read_fd}" gate_ready
  gate_ready_status=$?
  set -e
  gate_tag=""; gate_pid=""; gate_start=""; gate_extra=""
  IFS='|' read -r gate_tag gate_pid gate_start gate_extra <<<"${gate_ready}"
  starttime_value=""
  if starttime_value="$(pid_starttime "${parent_owned_pid}" 2>/dev/null)"; then starttime_status=0; else starttime_status=1; fi
  if (( gate_ready_status == 0 && starttime_status == 0 )) \
    && [[ "${gate_tag}" == F13_PARENT_COMMAND_GATE_READY \
          && "${gate_pid}" == "${parent_owned_pid}" \
          && "${gate_start}" == "${starttime_value}" \
          && -z "${gate_extra}" ]] \
    && pid_is_same_process "${parent_owned_pid}" "${starttime_value}"; then
    parent_owned_starttime="${starttime_value}"
  else
    parent_owned_starttime=""
  fi
  deferred_status="${parent_spawn_deferred_signal_status}"
  parent_spawn_in_progress=false
  if (( deferred_status == 0 )); then deferred_status="${parent_spawn_deferred_signal_status}"; fi
  parent_spawn_deferred_signal_status=0
  if [[ -z "${parent_owned_starttime}" ]] || (( deferred_status != 0 )); then
    exec {parent_command_gate_write_fd}>&- || true
    exec {parent_command_gate_read_fd}<&- || true
    parent_command_gate_write_fd=""
    parent_command_gate_read_fd=""
    if [[ -n "${parent_owned_starttime}" ]]; then
      terminate_owned_pid_bounded "${parent_owned_pid}" "${parent_owned_starttime}" parent-command-pre-exec-gate || true
    else
      # No token was sent; the frozen gate's pre-exec alarm bounds this wait.
      set +e; wait "${parent_owned_pid}"; status=$?; set -e
    fi
    parent_owned_pid=""
    parent_owned_starttime=""
    if (( deferred_status != 0 )); then exit "${deferred_status}"; fi
    return 71
  fi
  printf 'F13_PARENT_COMMAND_GATE_COMMIT\n' >&"${parent_command_gate_write_fd}" || return 71
  exec {parent_command_gate_write_fd}>&-
  exec {parent_command_gate_read_fd}<&-
  parent_command_gate_write_fd=""
  parent_command_gate_read_fd=""
  if (( deferred_status != 0 )); then
    trap '' INT TERM HUP
    terminate_owned_pid_bounded "${parent_owned_pid}" "${parent_owned_starttime}" parent-owned-command || true
    parent_owned_pid=""
    parent_owned_starttime=""
    exit "${deferred_status}"
  fi
  set +e
  wait "${parent_owned_pid}"
  status=$?
  set -e
  parent_owned_pid=""
  parent_owned_starttime=""
  if (( deferred_status != 0 )); then
    exit "${deferred_status}"
  fi
  return "${status}"
}

parent_exit_handler() {
  local status=$?
  trap - EXIT
  trap '' INT TERM HUP
  if [[ -n "${parent_srun_gate_write_fd}" ]]; then
    exec {parent_srun_gate_write_fd}>&- || true
  fi
  if [[ -n "${parent_srun_gate_read_fd}" ]]; then
    exec {parent_srun_gate_read_fd}<&- || true
  fi
  if [[ -n "${parent_command_gate_write_fd}" ]]; then
    exec {parent_command_gate_write_fd}>&- || true
  fi
  if [[ -n "${parent_command_gate_read_fd}" ]]; then
    exec {parent_command_gate_read_fd}<&- || true
  fi
  if [[ -n "${parent_publisher_write_fd}" ]]; then
    exec {parent_publisher_write_fd}>&- || true
  fi
  if [[ -n "${parent_publisher_read_fd}" ]]; then
    exec {parent_publisher_read_fd}<&- || true
  fi
  if [[ -n "${parent_owned_pid}" ]]; then
    if [[ -n "${parent_owned_starttime}" ]]; then
      if ! terminate_owned_pid_bounded "${parent_owned_pid}" "${parent_owned_starttime}" parent-owned-command; then
        (( status != 0 )) || status=70
      fi
    else
      echo "[arms-incomplete-exact2-r6] ERROR: parent-owned command identity unavailable; numeric signal forbidden" >&2
      (( status != 0 )) || status=70
    fi
  fi
  if [[ -n "${parent_srun_pid}" ]]; then
    if [[ -n "${parent_srun_starttime}" ]]; then
      terminate_owned_pid_bounded "${parent_srun_pid}" "${parent_srun_starttime}" local-srun || true
    else
      echo "[arms-incomplete-exact2-r6] ERROR: local srun PID identity unavailable; numeric signal forbidden" >&2
      (( status != 0 )) || status=70
    fi
    if ! cancel_exact_numbered_step_if_present; then
      echo "[arms-incomplete-exact2-r6] ERROR: exact owned numbered step could not be proven gone" >&2
      (( status != 0 )) || status=70
    fi
    if ! assert_retained_parent_unchanged; then
      echo "[arms-incomplete-exact2-r6] ERROR: retained parent changed after local srun" >&2
      (( status != 0 )) || status=70
    fi
  fi
  exit "${status}"
}

parent_signal_handler() {
  local status="$1"
  # Commit readiness takes precedence over the earlier spawn latch.  Once the
  # resident publisher is marked commit-active, every first signal is retained
  # in the commit latch even while the spawn flag is being retired.
  if [[ "${parent_success_commit_active}" == true ]]; then
    if (( parent_success_commit_deferred_signal_status == 0 )); then parent_success_commit_deferred_signal_status="${status}"; fi
    return 0
  fi
  if [[ "${parent_spawn_in_progress}" == true ]]; then
    if (( parent_spawn_deferred_signal_status == 0 )); then parent_spawn_deferred_signal_status="${status}"; fi
    return 0
  fi
  parent_signal_status="${status}"
  trap '' INT TERM HUP
  exit "${status}"
}

parent_signal_int() { parent_signal_handler 130; }
parent_signal_term() { parent_signal_handler 143; }
parent_signal_hup() { parent_signal_handler 129; }

trap parent_exit_handler EXIT
trap parent_signal_int INT
trap parent_signal_term TERM
trap parent_signal_hup HUP
set +e
parent_spawn_in_progress=true
generation_log="${run_root}/logs/arms-incomplete-exact2-generation.log"
[[ ! -e "${generation_log}" && ! -L "${generation_log}" ]] || fail "generation log path is not fresh"
(( BASH_VERSINFO[0] >= 4 )) || fail "exact srun exec gate requires Bash 4 or newer"
coproc F13_SRUN_GATE {
  exec "${frozen_python_exec_prefix[@]}" -S -s -P -B -c "${exact_srun_exec_gate_py}" \
    "${srun_bin}" "${srun_bin_sha}" "${srun_bin_size}" "${generation_log}" \
    --jobid="${holder_job}" --nodelist="${holder_node}" --nodes=1 --ntasks=1 \
    --exclusive --exact --kill-on-bad-exit=1 --immediate=5 --export=NONE \
    --job-name="BOX-EXP-013-r6-${controller_plan_sha}" \
    --cpus-per-task=32 --mem=60G --gpus-per-task=8 --gpu-bind=none --gres-flags=enforce-binding \
    /usr/bin/env -u SLURM_TMPDIR -u TMPDIR -u GADP_NODE_LOCAL_SCRATCH -u GADP_NODE_LOCAL_SCRATCH_FSTYPE \
      -u PYTHONDONTWRITEBYTECODE -u PYTHONNOUSERSITE -u PYTHONHASHSEED \
      -u BASH_ENV -u ENV -u CDPATH -u GLOBIGNORE \
      PATH=/usr/bin:/bin \
      F13_CONFIRM="${confirm}" F13_ENTRY_ENVIRONMENT="${entry_environment}" F13_RUN_ROOT="${run_root}" \
      F13_RELEASE_ROOT="${release_root}" F13_METHOD_ROOT="${method_root}" \
      F13_METHOD_ARCHIVE="${method_archive}" F13_METHOD_MANIFEST="${method_manifest}" \
      F13_DEPLOYMENT_ENVELOPE="${deployment_envelope}" F13_DEPLOYMENT_ENVELOPE_SHA256="${deployment_envelope_sha}" \
      F13_DETACHED_LAUNCHER_SHA256="${expected_detached_launcher_sha}" \
      F13_SEED1_SPEC="${seed1_spec}" F13_SEED2_SPEC="${seed2_spec}" \
      F13_EXTERNAL_EVIDENCE_ROOT="${external_root}" F13_EXTERNAL_KEY="${external_key}" \
      F13_EXTERNAL_REVIEW_RECEIPT="${external_review}" F13_PYTHON_BIN="${python_bin}" \
      F13_BERNINI_ROOT="${bernini_root}" F13_VEOMNI_ROOT="${veomni_root}" \
      F13_CHECKPOINT="${checkpoint}" F13_CHECKPOINT_MANIFEST="${checkpoint_manifest}" \
      F13_MASTER_PORT="${master_port}" \
      F13_R10_COMPILE_SMOKE_RECEIPT="${r10_receipt}" F13_R10_COMPILE_SMOKE_RECEIPT_SHA256="${r10_receipt_sha}" \
      F13_R10_GENERATION_LOG="${r10_log}" F13_R10_GENERATION_LOG_SHA256="${r10_log_sha}" \
      F13_CONTROLLER_PLAN="${controller_plan}" F13_CONTROLLER_PLAN_SHA256="${controller_plan_sha}" \
      F13_EXACT2_PLAN="${exact2_plan}" F13_EXACT2_PLAN_SHA256="${exact2_plan_sha}" \
      F13_METHOD_MANIFEST_SHA256="${core_manifest_sha}" \
      F13_VERIFIED_RUNNER_PATH="${release_builder}" F13_VERIFIED_RUNNER_SHA256="${owned_builder_sha}" \
      F13_RANK_WRAPPER_SHA256="${owned_rank_wrapper_sha}" \
      F13_PARENT_DETACHED_LAUNCHER_PATH="${detached_launcher}" F13_PARENT_DETACHED_LAUNCHER_SHA256="${detached_launcher_sha}" \
      /bin/bash --noprofile --norc -p "${launcher}" __child
}
parent_srun_pid="${F13_SRUN_GATE_PID}"
parent_srun_gate_read_fd="${F13_SRUN_GATE[0]}"
parent_srun_gate_write_fd="${F13_SRUN_GATE[1]}"
gate_ready=""
IFS= read -r -t 35 -u "${parent_srun_gate_read_fd}" gate_ready
gate_ready_status=$?
gate_tag=""; gate_pid=""; gate_start=""; gate_extra=""
IFS='|' read -r gate_tag gate_pid gate_start gate_extra <<<"${gate_ready}"
observed_gate_start=""
if observed_gate_start="$(pid_starttime "${parent_srun_pid}" 2>/dev/null)"; then
  starttime_status=0
else
  starttime_status=1
fi
gate_identity_valid=false
if (( gate_ready_status == 0 && starttime_status == 0 )) \
  && [[ "${gate_tag}" == F13_SRUN_GATE_READY && "${gate_pid}" == "${parent_srun_pid}" \
        && "${gate_start}" == "${observed_gate_start}" && -z "${gate_extra}" ]] \
  && pid_is_same_process "${parent_srun_pid}" "${observed_gate_start}"; then
  gate_identity_valid=true
  parent_srun_starttime="${observed_gate_start}"
fi
deferred_parent_signal_status="${parent_spawn_deferred_signal_status}"
parent_spawn_in_progress=false
if (( deferred_parent_signal_status == 0 )); then
  deferred_parent_signal_status="${parent_spawn_deferred_signal_status}"
fi
parent_spawn_deferred_signal_status=0
if [[ "${gate_identity_valid}" != true ]] || (( deferred_parent_signal_status != 0 )); then
  exec {parent_srun_gate_write_fd}>&- || true
  parent_srun_gate_write_fd=""
  exec {parent_srun_gate_read_fd}<&- || true
  parent_srun_gate_read_fd=""
  if [[ "${gate_identity_valid}" == true ]]; then
    terminate_owned_pid_bounded "${parent_srun_pid}" "${parent_srun_starttime}" srun-pre-exec-gate || true
  else
    # No commit token was sent.  The exact gate self-terminates under its
    # pre-exec alarm and therefore cannot create a numbered step.
    wait "${parent_srun_pid}"
    status=$?
  fi
  parent_srun_pid=""
  parent_srun_starttime=""
  set -e
  if (( deferred_parent_signal_status != 0 )); then exit "${deferred_parent_signal_status}"; fi
  fail "exact local srun pre-exec gate did not become ready status=${status:-71}"
fi
[[ -z "$(numbered_steps)" ]] || fail "numbered step appeared before exact srun gate commit"
assert_retained_parent_unchanged || fail "retained parent changed before exact srun gate commit"
printf 'F13_SRUN_GATE_COMMIT\n' >&"${parent_srun_gate_write_fd}" || fail "exact srun gate commit token failed"
exec {parent_srun_gate_write_fd}>&-
parent_srun_gate_write_fd=""
exec {parent_srun_gate_read_fd}<&-
parent_srun_gate_read_fd=""
wait "${parent_srun_pid}"
status=$?
set -e
residual_numbered_steps="$(numbered_steps)" || fail "post-srun numbered-step census failed"
if [[ -n "${residual_numbered_steps}" ]]; then
  cancel_exact_numbered_step_if_present || fail "residual exact numbered step could not be terminated"
  (( status != 0 )) || status=70
fi
assert_retained_parent_unchanged || fail "retained parent changed after local srun returned"
parent_srun_pid=""
parent_srun_starttime=""
if [[ -f "${generation_log}" && ! -L "${generation_log}" ]]; then chmod 0400 "${generation_log}" || fail "generation log mode seal failed"; fi
if (( status != 0 )); then tail -n 240 "${generation_log}" >&2 || true; fail_status "${status}" "exact2 child failed"; fi

# From this boundary onward the parent replays only shared receipt/status bytes
# and Slurm parent/step facts.  It never resolves, stats, scans, or deletes child
# /tmp paths and never inspects child /proc state.
blind_manifest="${run_root}/blind-review-packet/reviewer/review-manifest.json"
blind_key="${run_root}/blind-review-packet/sealed-key.json"
parent_scratch_prepare="${run_root}/logs/child-scratch-prepare.json"
parent_compute_preflight="${run_root}/logs/compute-preflight.json"
parent_task_scratch_binding="${run_root}/logs/child-task-scratch-bind.json"
parent_terminal_host_gate="${run_root}/logs/terminal-arms-incomplete-exact2-host-gate.json"
parent_physical_attestation="${run_root}/logs/child-terminal-physical-attestation.json"
parent_generation_audit="${run_root}/generation-exact2-audit.json"
parent_scratch_retained_terminal="${run_root}/logs/child-scratch-retained-terminal.json"
parent_terminal_ready="${run_root}/logs/child-terminal-ready.status"
for required_receipt in "${parent_scratch_prepare}" "${parent_compute_preflight}" "${parent_task_scratch_binding}" "${parent_generation_audit}" "${parent_terminal_host_gate}" "${parent_physical_attestation}" "${parent_scratch_retained_terminal}" "${blind_manifest}" "${blind_key}" "${parent_terminal_ready}"; do
  [[ -f "${required_receipt}" && ! -L "${required_receipt}" ]] || fail "post-srun receipt chain is incomplete: ${required_receipt}"
done
parent_scratch_prepare_sha="$(sha256_file "${parent_scratch_prepare}")" || fail "scratch prepare SHA-256 is unavailable"
parent_compute_preflight_sha="$(sha256_file "${parent_compute_preflight}")" || fail "compute preflight SHA-256 is unavailable"
parent_task_scratch_binding_sha="$(sha256_file "${parent_task_scratch_binding}")" || fail "task scratch binding SHA-256 is unavailable"
parent_generation_audit_sha="$(sha256_file "${parent_generation_audit}")" || fail "generation audit SHA-256 is unavailable"
parent_terminal_host_gate_sha="$(sha256_file "${parent_terminal_host_gate}")" || fail "terminal host-gate SHA-256 is unavailable"
parent_physical_attestation_sha="$(sha256_file "${parent_physical_attestation}")" || fail "physical attestation SHA-256 is unavailable"
parent_scratch_retained_terminal_sha="$(sha256_file "${parent_scratch_retained_terminal}")" || fail "retained-terminal scratch receipt SHA-256 is unavailable"
blind_manifest_sha="$(sha256_file "${blind_manifest}")" || fail "blind manifest SHA-256 is unavailable"
blind_key_sha="$(sha256_file "${blind_key}")" || fail "blind key SHA-256 is unavailable"
parent_terminal_ready_sha="$(sha256_file "${parent_terminal_ready}")" || fail "child terminal-ready SHA-256 is unavailable"
run_verified_materialized_app "${controller}" validate-child-scratch-retained-terminal \
  --controller-plan "${controller_plan}" --expected-controller-plan-sha256 "${controller_plan_sha}" \
  --scratch-prepare "${parent_scratch_prepare}" --expected-scratch-prepare-sha256 "${parent_scratch_prepare_sha}" \
  --compute-preflight "${parent_compute_preflight}" --expected-compute-preflight-sha256 "${parent_compute_preflight_sha}" \
  --task-scratch-bind "${parent_task_scratch_binding}" --expected-task-scratch-bind-sha256 "${parent_task_scratch_binding_sha}" \
  --generation-audit "${parent_generation_audit}" --expected-generation-audit-sha256 "${parent_generation_audit_sha}" \
  --terminal-host-gate "${parent_terminal_host_gate}" --expected-terminal-host-gate-sha256 "${parent_terminal_host_gate_sha}" \
  --physical-attestation "${parent_physical_attestation}" --expected-physical-attestation-sha256 "${parent_physical_attestation_sha}" \
  --blind-review-manifest "${blind_manifest}" --expected-blind-review-manifest-sha256 "${blind_manifest_sha}" \
  --blind-review-key "${blind_key}" --expected-blind-review-key-sha256 "${blind_key_sha}" \
  --scratch-retained-terminal "${parent_scratch_retained_terminal}" --expected-scratch-retained-terminal-sha256 "${parent_scratch_retained_terminal_sha}" >/dev/null || fail "post-srun receipt-only retained-scratch validation failed"
echo 'BOX_EXP_013_ARMS_INCOMPLETE_EXACT2_PENDING_REVIEW candidates=2 diagnostics=0 optimizer=false' || fail "parent terminal notice failed"

# The expensive shared-media and holder/step replay remains fully catchable.
# It produces a durable precommit but no success marker.
parent_generation_precommit="${run_root}/logs/parent-generation.precommit.json"
run_parent_owned_command "${python_bin}" -S -s -P -B -c "${materialization_bootstrap_py}" \
  "${method_root}" "${method_manifest}" "${core_manifest_sha}" \
  "${core_manifest_digest}" "${method_revision}" "${controller}" \
  prepare-parent-generation-status \
  --controller-plan "${controller_plan}" --expected-controller-plan-sha256 "${controller_plan_sha}" \
  --generation-audit "${parent_generation_audit}" --expected-generation-audit-sha256 "${parent_generation_audit_sha}" \
  --terminal-host-gate "${parent_terminal_host_gate}" --expected-terminal-host-gate-sha256 "${parent_terminal_host_gate_sha}" \
  --physical-attestation "${parent_physical_attestation}" --expected-physical-attestation-sha256 "${parent_physical_attestation_sha}" \
  --scratch-retained-terminal "${parent_scratch_retained_terminal}" --expected-scratch-retained-terminal-sha256 "${parent_scratch_retained_terminal_sha}" \
  --blind-review-manifest "${blind_manifest}" --expected-blind-review-manifest-sha256 "${blind_manifest_sha}" \
  --blind-review-key "${blind_key}" --expected-blind-review-key-sha256 "${blind_key_sha}" \
  --child-terminal-ready "${parent_terminal_ready}" --expected-child-terminal-ready-sha256 "${parent_terminal_ready_sha}" \
  --srun-exit-status 0 --output "${parent_generation_precommit}" >/dev/null || fail "durable parent generation precommit failed"
parent_generation_precommit_sha="$(sha256_file "${parent_generation_precommit}")" || fail "parent generation precommit SHA-256 is unavailable"
[[ "${parent_generation_precommit_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "parent generation precommit SHA-256 differs"

launcher_status="${run_root}/logs/parent-generation.status"
parent_generation_publisher() {
  exec "${frozen_python_exec_prefix[@]}" -S -s -P -B -c "${exact_parent_command_exec_gate_py}" \
    "${python_bin}" "${frozen_python_sha}" "${frozen_python_size}" \
    --f13-preserve-stdio -S -s -P -B -c "${materialization_bootstrap_py}" \
      "${method_root}" "${method_manifest}" "${core_manifest_sha}" \
      "${core_manifest_digest}" "${method_revision}" "${controller}" \
      resident-publish-parent-generation-status \
      --parent-generation-precommit "${parent_generation_precommit}" \
      --expected-parent-generation-precommit-sha256 "${parent_generation_precommit_sha}" \
      --output "${launcher_status}"
}

# The verified in-memory controller fully validates/captures the precommit,
# precomputes status bytes, and opens/validates the shared run/log dirfds before
# it emits READY.  Only held-fd O_EXCL/write/fsync publication remains after
# the exact parent commit token.
(( BASH_VERSINFO[0] >= 4 )) || fail "resident publisher requires Bash 4 or newer"
parent_spawn_in_progress=true
coproc F13_PUBLISHER { parent_generation_publisher; }
parent_owned_pid="${F13_PUBLISHER_PID}"
parent_publisher_read_fd="${F13_PUBLISHER[0]}"
parent_publisher_write_fd="${F13_PUBLISHER[1]}"
publisher_start_status=1
publisher_start_value=""
for _ in $(seq 1 500); do
  if publisher_start_value="$(pid_starttime "${parent_owned_pid}" 2>/dev/null)"; then
    publisher_start_status=0
    break
  fi
  kill -0 "${parent_owned_pid}" 2>/dev/null || break
  sleep 0.01
done
if (( publisher_start_status == 0 )); then parent_owned_starttime="${publisher_start_value}"; fi
publisher_gate_ready=""
set +e
IFS= read -r -t 35 -u "${parent_publisher_read_fd}" publisher_gate_ready
publisher_gate_ready_status=$?
set -e
publisher_gate_tag=""; publisher_gate_pid=""; publisher_gate_start=""; publisher_gate_extra=""
IFS='|' read -r publisher_gate_tag publisher_gate_pid publisher_gate_start publisher_gate_extra <<<"${publisher_gate_ready}"
publisher_gate_valid=false
if (( publisher_gate_ready_status == 0 && publisher_start_status == 0 )) \
  && [[ "${publisher_gate_tag}" == F13_PARENT_COMMAND_GATE_READY \
        && "${publisher_gate_pid}" == "${parent_owned_pid}" \
        && "${publisher_gate_start}" == "${parent_owned_starttime}" \
        && -z "${publisher_gate_extra}" ]] \
  && pid_is_same_process "${parent_owned_pid}" "${parent_owned_starttime}"; then
  publisher_gate_valid=true
fi
deferred_parent_publisher_spawn_status="${parent_spawn_deferred_signal_status}"
parent_spawn_in_progress=false
if (( deferred_parent_publisher_spawn_status == 0 )); then
  deferred_parent_publisher_spawn_status="${parent_spawn_deferred_signal_status}"
fi
parent_spawn_deferred_signal_status=0
if [[ "${publisher_gate_valid}" != true ]] || (( deferred_parent_publisher_spawn_status != 0 )); then
  exec {parent_publisher_write_fd}>&- || true
  exec {parent_publisher_read_fd}<&- || true
  if [[ "${publisher_gate_valid}" == true ]]; then
    terminate_owned_pid_bounded "${parent_owned_pid}" "${parent_owned_starttime}" parent-publisher-pre-exec-gate || true
  else
    set +e; wait "${parent_owned_pid}"; publisher_status=$?; set -e
  fi
  parent_owned_pid=""
  parent_owned_starttime=""
  if (( deferred_parent_publisher_spawn_status != 0 )); then exit "${deferred_parent_publisher_spawn_status}"; fi
  fail "resident publisher exact pre-exec gate did not become ready status=${publisher_status:-71}"
fi
printf 'F13_PARENT_COMMAND_GATE_COMMIT\n' >&"${parent_publisher_write_fd}" || fail "resident publisher pre-exec gate commit failed"

# The same verified PID now executes the in-memory release runner.  Signals
# remain fully catchable during its expensive precommit/dirfd preparation.
publisher_ready=""
set +e
IFS= read -r -t 120 -u "${parent_publisher_read_fd}" publisher_ready
publisher_ready_status=$?
set -e
publisher_ready_tag=""; publisher_prepared_status_sha=""; publisher_ready_extra=""
IFS=' ' read -r publisher_ready_tag publisher_prepared_status_sha publisher_ready_extra <<<"${publisher_ready}"
# The controller has blocked terminal signals and holds all required fds before
# READY.  From here the parent records (rather than acts on) the first signal.
parent_success_commit_active=true
deferred_parent_commit_status=0
if (( publisher_ready_status != 0 )) \
  || [[ "${publisher_ready_tag}" != BOX-EXP-013-r6-PARENT-PUBLISH-READY \
        || ! "${publisher_prepared_status_sha}" =~ ^[0-9a-f]{64}$ \
        || -n "${publisher_ready_extra}" ]] \
  || (( publisher_start_status != 0 )) \
  || ! pid_is_same_process "${parent_owned_pid}" "${parent_owned_starttime}"; then
  exec {parent_publisher_write_fd}>&- || true
  set +e
  wait "${parent_owned_pid}"
  publisher_status=$?
  set -e
  trap '' INT TERM HUP
  if (( deferred_parent_commit_status == 0 && parent_success_commit_deferred_signal_status != 0 )); then
    deferred_parent_commit_status="${parent_success_commit_deferred_signal_status}"
  fi
  parent_owned_pid=""
  parent_owned_starttime=""
  exec {parent_publisher_read_fd}<&- || true
  if (( deferred_parent_commit_status != 0 )); then exit "${deferred_parent_commit_status}"; fi
  fail "resident parent generation-status publisher did not become ready status=${publisher_status}"
fi
if (( deferred_parent_commit_status != 0 || parent_success_commit_deferred_signal_status != 0 )); then
  [[ ${deferred_parent_commit_status} -ne 0 ]] || deferred_parent_commit_status="${parent_success_commit_deferred_signal_status}"
  trap '' INT TERM HUP
  exec {parent_publisher_write_fd}>&- || true
  set +e
  wait "${parent_owned_pid}"
  set -e
  parent_owned_pid=""
  parent_owned_starttime=""
  exec {parent_publisher_read_fd}<&- || true
  exit "${deferred_parent_commit_status}"
fi
trap '' INT TERM HUP
if (( parent_success_commit_deferred_signal_status != 0 )); then
  deferred_parent_commit_status="${parent_success_commit_deferred_signal_status}"
  exec {parent_publisher_write_fd}>&- || true
  set +e
  wait "${parent_owned_pid}"
  set -e
  parent_owned_pid=""
  parent_owned_starttime=""
  exec {parent_publisher_read_fd}<&- || true
  exit "${deferred_parent_commit_status}"
fi
printf 'BOX-EXP-013-r6-PARENT-PUBLISH-COMMIT\n' >&"${parent_publisher_write_fd}" || fail "resident publisher commit token failed"
exec {parent_publisher_write_fd}>&-
publisher_ack=""
set +e
IFS= read -r -t 30 -u "${parent_publisher_read_fd}" publisher_ack
publisher_ack_status=$?
set -e
set +e
wait "${parent_owned_pid}"
publisher_status=$?
set -e
parent_owned_pid=""
parent_owned_starttime=""
exec {parent_publisher_read_fd}<&-
(( publisher_status == 0 )) || fail_status "${publisher_status}" "durable parent generation-status publication failed"
[[ ${publisher_ack_status} -eq 0 && "${publisher_ack}" == "BOX-EXP-013-r6-PARENT-PUBLISH-ACK ${publisher_prepared_status_sha}" ]] || fail "resident publisher acknowledgement differs"
# ACK is emitted only after the verified resident controller has completed its
# held-fd double-read/hash/metadata/name replay and directory fsync.  That
# durable marker is the semantic commit.  Catchable signals remain ignored
# through the immediately following in-process state assignment and exit, so a
# post-commit signal cannot turn an authoritative success into rc129/130/143.
parent_success_committed=true
trap - EXIT
exit 0
