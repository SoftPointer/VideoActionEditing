#!/usr/bin/env bash
# Per-rank node-local cache wrapper for the fit40 native renderer.
#
# AMD COMGR must not place its temporary hipfatbin/unbundle objects on the
# shared NFS experiment tree.  torchrun invokes this wrapper with --no_python;
# each rank creates a private directory below the launcher's authenticated
# node-local scratch root, then starts the frozen Python worker inside it.

set -Eeuo pipefail
umask 077

fail() { echo "[generic-action-rank-cache-v1] ERROR: $*" >&2; exit 2; }

readonly cache_token="${GADP_RANK_CACHE_TOKEN:?set rank cache token}"
readonly scratch_parent="${GADP_NODE_LOCAL_SCRATCH:?set node-local scratch}"
readonly expected_fstype="${GADP_NODE_LOCAL_SCRATCH_FSTYPE:?set scratch filesystem}"
readonly python_bin="${GADP_RANK_PYTHON_BIN:?set frozen Python executable}"
readonly frozen_python_path=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly root_bootstrap_python=/usr/bin/python3.10
readonly method_root="${GADP_METHOD_ROOT:?set released method root}"
readonly release_manifest="${F13_METHOD_MANIFEST:?set release manifest}"
readonly release_manifest_sha="${F13_METHOD_MANIFEST_SHA256:?pin release manifest}"
readonly verified_runner="${F13_VERIFIED_RUNNER_PATH:?set verified release runner}"
readonly verified_runner_sha="${F13_VERIFIED_RUNNER_SHA256:?pin verified release runner}"
readonly rank_wrapper_sha="${F13_RANK_WRAPPER_SHA256:?pin captured rank wrapper}"
readonly local_rank="${LOCAL_RANK:?torchrun LOCAL_RANK required}"
readonly global_rank="${RANK:?torchrun RANK required}"
readonly world_size="${WORLD_SIZE:?torchrun WORLD_SIZE required}"
readonly job_id="${SLURM_JOB_ID:?Slurm job required}"
readonly step_id="${SLURM_STEP_ID:?numbered Slurm step required}"

[[ $# -ge 1 ]] || fail "worker path is required"
[[ "${cache_token}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$ ]] || fail "cache token differs"
[[ "${world_size}" == 4 && "${local_rank}" =~ ^[0-3]$ && "${global_rank}" =~ ^[0-3]$ ]] || fail "WORLD4 rank identity differs"
[[ "${global_rank}" == "${local_rank}" ]] || fail "single-node WORLD4 rank mapping differs"
[[ "${scratch_parent}" == /* && "${scratch_parent}" != / && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "node-local scratch root differs"
[[ "${method_root}" == /* && -d "${method_root}" && ! -L "${method_root}" && "$(readlink -f -- "${method_root}")" == "${method_root}" ]] || fail "released method root differs"
[[ "${python_bin}" == "${frozen_python_path}" && -x "${python_bin}" && ! -L "${python_bin}" && "$(readlink -f -- "${python_bin}")" == "${python_bin}" ]] || fail "frozen Python differs"
[[ -x "${root_bootstrap_python}" && ! -L "${root_bootstrap_python}" && "$(readlink -f -- "${root_bootstrap_python}")" == "${root_bootstrap_python}" ]] || fail "root-owned bootstrap Python differs"
readonly worker="$1"
shift
readonly worker_relative=infer_pair_v5_t2v_calibration_bank.py
[[ "${worker}" == "${method_root}/${worker_relative}" && -f "${worker}" && ! -L "${worker}" && "$(readlink -f -- "${worker}")" == "${worker}" ]] || fail "worker is not the exact released generation target"
[[ "${release_manifest}" == /* && -f "${release_manifest}" && ! -L "${release_manifest}" && "${release_manifest_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "release manifest runner binding differs"
[[ "${verified_runner}" == "${method_root}/tools/build_full30_action_arms_incomplete_repair_exact2_release_v1.py" && -f "${verified_runner}" && ! -L "${verified_runner}" && "${verified_runner_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "verified release runner binding differs"
[[ "${rank_wrapper_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "captured rank wrapper SHA binding differs"

scratch_real="$(readlink -f -- "${scratch_parent}")" || fail "node-local scratch canonicalization failed"
[[ -n "${scratch_real}" ]] || fail "node-local scratch canonical path is empty"
readonly scratch_real
[[ "${scratch_real}" == "${scratch_parent}" ]] || fail "node-local scratch canonical path differs"
observed_fstype="$(stat -f -c '%T' -- "${scratch_real}")" || fail "node-local scratch statfs failed"
[[ -n "${observed_fstype}" ]] || fail "node-local scratch filesystem type is empty"
readonly observed_fstype
[[ "${observed_fstype}" == "${expected_fstype}" ]] || fail "node-local scratch filesystem changed"
case "${observed_fstype}" in
  ext2/ext3|xfs|tmpfs) ;;
  *) fail "COMGR scratch filesystem is not an allowed node-local type: ${observed_fstype}" ;;
esac

rank_root="$(mktemp -d -- "${scratch_real}/gadp-${job_id}-${step_id}-${cache_token}-r${global_rank}.XXXXXXXX")" || fail "rank cache create-only mkdir failed"
[[ -n "${rank_root}" && "${rank_root}" == "${scratch_real}/"* ]] || fail "rank cache path differs"
rank_root_real="$(readlink -f -- "${rank_root}")" || fail "rank cache canonicalization failed"
[[ "${rank_root_real}" == "${rank_root}" ]] || fail "rank cache is not canonical"
readonly rank_root_real
readonly rank_root
scratch_device="$(stat -c '%d' -- "${scratch_real}")" || fail "rank cache parent device stat failed"
[[ "${scratch_device}" =~ ^[0-9]+$ ]] || fail "rank cache parent device differs"
readonly scratch_device
rank_identity="$(stat -c '%d:%i:%u:%g:%a:%h' -- "${rank_root}")" || fail "rank cache identity stat failed"
[[ "${rank_identity}" =~ ^${scratch_device}:[1-9][0-9]*:2012:2000:700:2$ ]] || fail "rank cache identity format differs"
readonly rank_identity
for leaf in tmp xdg hf torch-extensions triton torchinductor pycache miopen-user miopen-custom; do
  mkdir -m 0700 "${rank_root}/${leaf}"
done

export TMPDIR="${rank_root}/tmp"
export TMP="${rank_root}/tmp"
export TEMP="${rank_root}/tmp"
export XDG_CACHE_HOME="${rank_root}/xdg"
export HF_HOME="${rank_root}/hf"
export TORCH_EXTENSIONS_DIR="${rank_root}/torch-extensions"
export TRITON_CACHE_DIR="${rank_root}/triton"
export TORCHINDUCTOR_CACHE_DIR="${rank_root}/torchinductor"
export PYTHONPYCACHEPREFIX="${rank_root}/pycache"
export MIOPEN_USER_DB_PATH="${rank_root}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${rank_root}/miopen-custom"
export GADP_RANK_CACHE_FILESYSTEM="${observed_fstype}"
export GADP_RANK_CACHE_IS_NODE_LOCAL=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
export LC_ALL=C LANG=C

child_pid=""
child_start_ticks=""
pending_signal_status=0
spawn_identity_in_progress=false
gate_write_fd=""

proc_start_ticks() {
  local pid="$1"
  local raw rest state
  [[ "${pid}" =~ ^[1-9][0-9]*$ && -r "/proc/${pid}/stat" ]] || return 1
  IFS= read -r raw < "/proc/${pid}/stat" || return 1
  [[ "${raw}" == *") "* ]] || return 1
  rest="${raw##*) }"
  set -- ${rest}
  state="$1"
  [[ $# -ge 20 && "${state}" =~ ^[A-Z]$ && "${20}" =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s:%s\n' "${20}" "${state}"
}

owned_child_observation() {
  local observed_identity observed_ticks observed_state
  [[ -n "${child_pid}" && -n "${child_start_ticks}" ]] || return 2
  # UNKNOWN/malformed and start-tick mismatch never authorize a signal.  The
  # outer launcher owns exact Slurm-cgroup teardown if this wrapper returns70.
  observed_identity="$(proc_start_ticks "${child_pid}")" || return 2
  observed_ticks="${observed_identity%%:*}"
  observed_state="${observed_identity#*:}"
  [[ "${observed_ticks}" == "${child_start_ticks}" ]] || return 2
  [[ "${observed_state}" != Z ]] || return 1
  return 0
}

signal_owned_child_exact() {
  local signal_name="$1" observation_status
  if owned_child_observation; then
    # This exact read is the authorization immediately preceding the signal.
    kill "-${signal_name}" "${child_pid}" 2>/dev/null || true
    return 0
  else
    observation_status=$?
  fi
  [[ "${observation_status}" -eq 1 ]] && return 1
  return 70
}

bounded_reap_owned_child() {
  local attempt observation_status signal_status
  [[ -n "${child_pid}" ]] || return 0
  [[ -n "${child_start_ticks}" ]] || return 70
  if owned_child_observation; then
    if signal_owned_child_exact TERM; then
      :
    else
      signal_status=$?
      [[ "${signal_status}" -eq 1 ]] || return 70
    fi
  else
    observation_status=$?
    if [[ "${observation_status}" -eq 1 ]]; then
      wait "${child_pid}" 2>/dev/null || true
      child_pid=""
      child_start_ticks=""
      return 0
    fi
    return 70
  fi
  for attempt in {1..50}; do
    if owned_child_observation; then
      sleep 0.1
      continue
    fi
    observation_status=$?
    if [[ "${observation_status}" -eq 1 ]]; then
      wait "${child_pid}" 2>/dev/null || true
      child_pid=""
      child_start_ticks=""
      return 0
    fi
    return 70
  done
  if signal_owned_child_exact KILL; then
    :
  else
    signal_status=$?
    if [[ "${signal_status}" -eq 1 ]]; then
      wait "${child_pid}" 2>/dev/null || true
      child_pid=""
      child_start_ticks=""
      return 0
    fi
    return 70
  fi
  for attempt in {1..50}; do
    if owned_child_observation; then
      sleep 0.1
      continue
    fi
    observation_status=$?
    if [[ "${observation_status}" -eq 1 ]]; then
      wait "${child_pid}" 2>/dev/null || true
      child_pid=""
      child_start_ticks=""
      return 0
    fi
    return 70
  done
  return 70
}

record_signal() {
  local explicit_status="$1"
  if [[ "${pending_signal_status}" -eq 0 ]]; then
    pending_signal_status="${explicit_status}"
    trap '' INT TERM HUP
  fi
}

on_int() { record_signal 130; }
on_term() { record_signal 143; }
on_hup() { record_signal 129; }

finish() {
  local explicit_status="$1"
  trap - EXIT
  trap '' INT TERM HUP
  if [[ -n "${gate_write_fd}" ]]; then
    exec {gate_write_fd}>&- || true
    gate_write_fd=""
  fi
  if [[ "${pending_signal_status}" -ne 0 ]]; then
    explicit_status="${pending_signal_status}"
  fi
  if [[ -n "${child_pid}" ]]; then
    if ! bounded_reap_owned_child; then
      if [[ "${pending_signal_status}" -eq 0 ]]; then
        explicit_status=70
      fi
    fi
  fi
  # Every rank subtree, including failed and successful worker output, is
  # retained at child terminal seal.  It is nonreusable, this release grants
  # no physical or manual cleanup authority, and future persistence is not
  # guaranteed after the Slurm step or a host reboot.
  exit "${explicit_status}"
}

on_exit() {
  local explicit_status=$?
  finish "${explicit_status}"
}

trap on_exit EXIT
trap on_int INT
trap on_term TERM
trap on_hup HUP

[[ "${pending_signal_status}" -eq 0 ]] || exit "${pending_signal_status}"
readonly verified_runner_bootstrap='import hashlib,os,stat,sys
p,e=sys.argv[1:3]
if not os.path.isabs(p) or len(e)!=64 or any(c not in "0123456789abcdef" for c in e): raise SystemExit(70)
f=os.open(p,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_CLOEXEC",0))
def fields(s): return (s.st_dev,s.st_ino,s.st_uid,s.st_gid,s.st_mode,s.st_nlink,s.st_size,s.st_blocks,s.st_mtime_ns,s.st_ctime_ns)
def readall():
 b=[]
 while True:
  c=os.read(f,1048576)
  if not c:return b"".join(b)
  b.append(c)
a=os.fstat(f);x=readall();b=os.fstat(f);os.lseek(f,0,os.SEEK_SET);y=readall();c=os.fstat(f);os.close(f);n=os.lstat(p)
if not (os.path.realpath(p)==p and stat.S_ISREG(a.st_mode) and not stat.S_ISLNK(n.st_mode) and a.st_nlink==1 and stat.S_IMODE(a.st_mode)==0o444 and fields(a)==fields(b)==fields(c)==fields(n) and x==y and len(x)==a.st_size and hashlib.sha256(x).hexdigest()==e): raise SystemExit(70)
g={"__name__":"_box_exp_013_verified_runner_bootstrap","__file__":p,"__package__":None,"__spec__":None,"__builtins__":__builtins__}
exec(compile(x,p,"exec",dont_inherit=True),g)
raise SystemExit(g["main"](sys.argv[3:]))'
spawn_identity_in_progress=true
exec 3>&1
coproc RANK_WORKER {
  trap - EXIT INT TERM HUP
  exec 1>&3
  IFS= read -r gate_value || exit 70
  [[ "${gate_value}" == go ]] || exit 70
  exec "${root_bootstrap_python}" -I -S -s -B -c "${verified_runner_bootstrap}" \
    "${verified_runner}" "${verified_runner_sha}" \
    held-fd-exec-frozen-python -- \
    -I -S -B -c "${verified_runner_bootstrap}" \
    "${verified_runner}" "${verified_runner_sha}" \
    verified-run-module \
    --method-root "${method_root}" \
    --manifest "${release_manifest}" \
    --expected-manifest-sha256 "${release_manifest_sha}" \
    --expected-runner-sha256 "${verified_runner_sha}" \
    --target "${worker_relative}" -- "$@"
}
exec 3>&-
gate_read_fd="${RANK_WORKER[0]}"
gate_write_fd="${RANK_WORKER[1]}"
spawned_pid="${RANK_WORKER_PID:-$!}"
exec {gate_read_fd}<&-
[[ "${spawned_pid}" =~ ^[1-9][0-9]*$ ]] || fail "rank worker PID differs"
child_pid="${spawned_pid}"
child_identity="$(proc_start_ticks "${child_pid}")" || {
  spawn_identity_in_progress=false
  if [[ "${pending_signal_status}" -ne 0 ]]; then
    exit "${pending_signal_status}"
  fi
  echo "[generic-action-rank-cache-v1] ERROR: rank worker start-ticks identity unavailable" >&2
  exit 70
}
child_start_ticks="${child_identity%%:*}"
child_initial_state="${child_identity#*:}"
spawn_identity_in_progress=false
if [[ "${child_initial_state}" == Z ]]; then
  wait "${child_pid}" 2>/dev/null || true
  child_pid=""
  child_start_ticks=""
  [[ "${pending_signal_status}" -eq 0 ]] || exit "${pending_signal_status}"
  exit 70
fi
if [[ "${pending_signal_status}" -ne 0 ]]; then
  bounded_reap_owned_child || true
  exit "${pending_signal_status}"
fi
printf 'go\n' >&"${gate_write_fd}" || {
  bounded_reap_owned_child || true
  exit 70
}
exec {gate_write_fd}>&-
gate_write_fd=""
if [[ "${pending_signal_status}" -ne 0 ]]; then
  bounded_reap_owned_child || true
  exit "${pending_signal_status}"
fi
set +e
wait "${child_pid}"
status=$?
set -e
if [[ "${pending_signal_status}" -eq 0 ]]; then
  # An uninterrupted wait has positively reaped this exact waitable child.
  # Clear immediately; a later /proc lookup could observe an unrelated reused PID.
  child_pid=""
  child_start_ticks=""
else
  bounded_reap_owned_child || exit "${pending_signal_status}"
fi
if [[ "${pending_signal_status}" -ne 0 ]]; then
  exit "${pending_signal_status}"
fi
exit "${status}"
