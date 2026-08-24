#!/usr/bin/env bash
# Hostile shell closure for the BOX-EXP-013 r6 detached launcher.
#
# This suite is workstation-safe: it never calls real Slurm or a GPU.  Its
# integration cases execute a real parent -> stubbed srun -> child process
# boundary.  Linux/AUH-only physical cgroup and /tmp identities remain the
# controller's responsibility; this suite checks the launcher's shell state
# machine, signal codes, ownership rules, and failure-retention semantics.

set -Eeuo pipefail
umask 077

test_root="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
launcher="${test_root}/methods/bernini_action_editing/scripts/auh_full30_action_arms_incomplete_repair_exact2_136140_world4_v1.sh"
controller="${test_root}/methods/bernini_action_editing/full30_action_arms_incomplete_repair_exact2_controller_v1.py"
[[ -f "${launcher}" ]] || { echo "missing launcher: ${launcher}" >&2; exit 2; }
[[ -f "${controller}" ]] || { echo "missing controller: ${controller}" >&2; exit 2; }

work="$(mktemp -d "${TMPDIR:-/tmp}/f13-r6-hostile.XXXXXXXX")"
cleanup_test_work() { rm -rf -- "${work}"; }
trap cleanup_test_work EXIT

pass_count=0
fail_test() { echo "not ok $((pass_count + 1)) - $*" >&2; exit 1; }
pass_test() { pass_count=$((pass_count + 1)); echo "ok ${pass_count} - $*"; }
require_source() { rg -F --quiet -- "$1" "${launcher}" || fail_test "missing source anchor: $1"; }
forbid_source() { ! rg -F --quiet -- "$1" "${launcher}" || fail_test "forbidden source anchor: $1"; }

# Extract the exact production helpers used by the process tests.  The AUH
# /proc start-time primitive is replaced below only because this suite also
# runs on Darwin; ownership remains an unreaped direct-child invariant here.
functions_file="${work}/launcher-state-functions.sh"
for function_name in \
  terminate_owned_pid_bounded \
  step_cgroup_is_exact_owned resolve_owned_step_cgroup_procs \
  read_owned_step_cgroup_membership owned_step_membership_contains \
  collect_owned_step_cgroup_pids \
  run_child_owned_command run_child_required child_exit_handler \
  child_signal_handler; do
  awk -v name="${function_name}" '
    $0 ~ "^" name "\\(\\) \\{" {copy=1}
    copy {print}
    copy && /^}/ {copy=0; found=1}
    END {if (!found) exit 9}
  ' "${launcher}" >>"${functions_file}" || fail_test "cannot extract ${function_name}"
done

parent_functions="${work}/launcher-parent-functions.sh"
for function_name in parent_exit_handler parent_signal_handler; do
  awk -v name="${function_name}" '
    $0 ~ "^" name "\\(\\) \\{" {copy=1}
    copy {print}
    copy && /^}/ {copy=0; found=1}
    END {if (!found) exit 9}
  ' "${launcher}" >>"${parent_functions}" || fail_test "cannot extract ${function_name}"
done

common="${work}/child-common.sh"
cat >"${common}" <<'EOF'
set -Eeuo pipefail
umask 077
source "${F13_TEST_FUNCTIONS}"
fail_status() { local status="$1"; shift; echo "$*" >&2; exit "${status}"; }
# The production wrapper accepts only verified release Python targets.  These
# state-machine fixtures deliberately run tiny local commands, so retain the
# production owned-PID primitive while replacing only that target-policy shim.
run_child_required() {
  local label="$1" status
  shift
  set +e
  run_child_owned_command "$@"
  status=$?
  set -e
  (( status == 0 )) || fail_status "${status}" "${label}"
}
pid_starttime() {
  kill -0 "$1" 2>/dev/null || return 1
  if [[ "${F13_TEST_MODE:-}" == signal-spawn-deferred && ! -e "${F13_TEST_ROOT}/spawn-delay.used" ]]; then
    : >"${F13_TEST_ROOT}/spawn-delay.used"
    sleep 0.2
  fi
  printf '1\n'
}
pid_is_same_process() { kill -0 "$1" 2>/dev/null; }
terminate_owned_step_cgroup_bounded() {
  local pid attempt survivors
  test_pid_live() { local state; state="$(ps -o stat= -p "$1" 2>/dev/null | tr -d '[:space:]')" || return 1; [[ -n "${state}" && "${state:0:1}" != Z ]]; }
  [[ -f "${F13_TEST_ROOT}/detached.pids" ]] || return 0
  while IFS= read -r pid; do [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -TERM "${pid}" 2>/dev/null || true; done <"${F13_TEST_ROOT}/detached.pids"
  for attempt in $(seq 1 50); do
    survivors=""
    while IFS= read -r pid; do test_pid_live "${pid}" && survivors+="${pid} "; done <"${F13_TEST_ROOT}/detached.pids"
    [[ -z "${survivors}" ]] && return 0
    sleep 0.1
  done
  while IFS= read -r pid; do test_pid_live "${pid}" && kill -KILL "${pid}" 2>/dev/null || true; done <"${F13_TEST_ROOT}/detached.pids"
  for attempt in $(seq 1 50); do
    survivors=""
    while IFS= read -r pid; do test_pid_live "${pid}" && survivors+="${pid} "; done <"${F13_TEST_ROOT}/detached.pids"
    [[ -z "${survivors}" ]] && return 0
    sleep 0.1
  done
  return 1
}
write_child_launcher_failure_receipt() {
  printf '%s\n' "phase=${child_failure_phase}" "status=$1" \
    'retention_claim=see-signed-receipt' 'retention_not_claimed=true' >"${F13_TEST_ROOT}/failure.receipt"
}
seal_child_failure_best_effort() {
  [[ "${F13_TEST_MODE:-}" != outer-drift-failure ]] || printf failed >"${F13_TEST_ROOT}/failure-seal.failed"
  write_child_launcher_failure_receipt "$1"
}
child_spawn_in_progress=false
child_spawn_deferred_signal_status=0
child_spawn_deferred_signal_phase=""
child_owned_pid=""
child_owned_starttime=""
owned_step_cgroup_path=""
owned_step_cgroup_procs=""
owned_step_cgroup_membership=()
owned_step_cgroup_tokens=()
child_terminal_ready_committed=false
child_failure_phase=prepare
host_memory_monitor_pid=""
host_memory_monitor_starttime=""
role=child
child_signal_int() { child_signal_handler 130 signal-int; }
child_signal_term() { child_signal_handler 143 signal-term; }
child_signal_hup() { child_signal_handler 129 signal-hup; }
trap child_exit_handler EXIT
trap child_signal_int INT
trap child_signal_term TERM
trap child_signal_hup HUP
EOF

child="${work}/stub-child.sh"
cat >"${child}" <<'EOF'
#!/usr/bin/env bash
source "${F13_TEST_COMMON}"
for name in SLURM_TMPDIR TMPDIR GADP_NODE_LOCAL_SCRATCH GADP_NODE_LOCAL_SCRATCH_FSTYPE; do
  if [[ -v "${name}" ]]; then
    child_failure_phase=prepare
    exit 42
  fi
done
mkdir -m 0700 "${F13_TEST_ROOT}/outer" "${F13_TEST_ROOT}/outer/inner"
mkdir -m 0700 "${F13_TEST_ROOT}/outer/inner/rank" "${F13_TEST_ROOT}/outer/inner/rank/cache"
printf 'forensic-bytes\n' >"${F13_TEST_ROOT}/outer/inner/rank/cache/evidence.bin"
printf 'prepared\n' >"${F13_TEST_ROOT}/prepared"
delete_primitive_forbidden() { printf attempted >"${F13_TEST_ROOT}/deletion.attempt"; return 97; }
rm() { delete_primitive_forbidden; }
rmdir() { delete_primitive_forbidden; }
unlink() { delete_primitive_forbidden; }
rmtree() { delete_primitive_forbidden; }
export -f delete_primitive_forbidden rm rmdir unlink rmtree
case "${F13_TEST_MODE}" in
  success|success-plan-replaced|success-blind-replaced)
    run_child_required success-command /bin/sh -c 'exit 0'
    printf 'blind-original\n' >"${F13_TEST_ROOT}/blind.manifest"
    shasum -a 256 "${F13_TEST_ROOT}/blind.manifest" | awk '{print $1}' >"${F13_TEST_ROOT}/attested-blind.sha256"
    printf 'retained-terminal\n' >"${F13_TEST_ROOT}/retained-terminal.receipt"
    printf 'terminal-ready\n' >"${F13_TEST_ROOT}/terminal-ready.marker"
    child_terminal_ready_committed=true
    exit 0
    ;;
  nested-failure)
    child_failure_phase=generation
    run_child_required nested-failure /bin/sh -c 'exit 23'
    ;;
  outer-drift-failure)
    child_failure_phase=generation
    mv "${F13_TEST_ROOT}/outer" "${F13_TEST_ROOT}/outer.identity-drifted"
    exit 74
    ;;
  uid-census-failure)
    child_failure_phase=terminal
    printf '%s\n' "$$" >"${F13_TEST_ROOT}/cgroup.procs"
    resolve_owned_step_cgroup_procs() { owned_step_cgroup_procs="${F13_TEST_ROOT}/cgroup.procs"; return 0; }
    id() { return 44; }
    if collect_owned_step_cgroup_pids >/dev/null; then printf fail-open >"${F13_TEST_ROOT}/census.failopen"; fi
    exit 95
    ;;
  identity-census-failure)
    child_failure_phase=terminal
    printf '99999999\n' >"${F13_TEST_ROOT}/cgroup.procs"
    resolve_owned_step_cgroup_procs() { owned_step_cgroup_procs="${F13_TEST_ROOT}/cgroup.procs"; return 0; }
    if collect_owned_step_cgroup_pids >/dev/null; then printf fail-open >"${F13_TEST_ROOT}/census.failopen"; fi
    exit 95
    ;;
  malformed-census-failure)
    child_failure_phase=terminal
    printf 'not-a-pid\n' >"${F13_TEST_ROOT}/cgroup.procs"
    resolve_owned_step_cgroup_procs() { owned_step_cgroup_procs="${F13_TEST_ROOT}/cgroup.procs"; return 0; }
    if collect_owned_step_cgroup_pids >/dev/null; then printf fail-open >"${F13_TEST_ROOT}/census.failopen"; fi
    exit 95
    ;;
  unreadable-census-failure)
    child_failure_phase=terminal
    printf '%s\n' "$$" >"${F13_TEST_ROOT}/cgroup.procs"
    chmod 000 "${F13_TEST_ROOT}/cgroup.procs"
    resolve_owned_step_cgroup_procs() { owned_step_cgroup_procs="${F13_TEST_ROOT}/cgroup.procs"; [[ -r "${owned_step_cgroup_procs}" && ! -L "${owned_step_cgroup_procs}" ]]; }
    if collect_owned_step_cgroup_pids >/dev/null; then printf fail-open >"${F13_TEST_ROOT}/census.failopen"; fi
    exit 95
    ;;
  live-census-detected)
    child_failure_phase=terminal
    /bin/sh -c 'exec sleep 60' &
    census_live_pid=$!
    printf '%s\n%s\n' "$$" "${census_live_pid}" >"${F13_TEST_ROOT}/cgroup.procs"
    resolve_owned_step_cgroup_procs() { owned_step_cgroup_procs="${F13_TEST_ROOT}/cgroup.procs"; return 0; }
    stat() { printf '%s\n' "$(id -u)"; }
    pid_starttime() { kill -0 "$1" 2>/dev/null || return 1; printf '1\n'; }
    collect_owned_step_cgroup_pids
    [[ ${#owned_step_cgroup_tokens[@]} -eq 1 && "${owned_step_cgroup_tokens[0]}" == "${census_live_pid}:1" ]] || exit 96
    kill -TERM "${census_live_pid}" 2>/dev/null || true
    wait "${census_live_pid}" 2>/dev/null || true
    exit 76
    ;;
  signal-unset)
    printf 'window\n' >"${F13_TEST_ROOT}/window.ready"
    while :; do sleep 0.1; done
    ;;
  signal-assigned)
    child_failure_phase=generation
    run_child_required long-command /bin/sh -c 'printf ready >"$F13_TEST_ROOT/window.ready"; exec sleep 60'
    ;;
  signal-spawn-deferred)
    child_failure_phase=generation
    run_child_required spawn-deferred /bin/sh -c 'kill -TERM "$PPID"; exec sleep 60'
    ;;
  signal-ignore-term)
    child_failure_phase=generation
    run_child_required ignore-term /bin/sh -c 'trap "" TERM; printf ready >"$F13_TEST_ROOT/window.ready"; while :; do sleep 1; done'
    ;;
  retained-terminal-signal)
    child_failure_phase=retained-terminal
    run_child_required retained-terminal-seal /bin/sh -c 'printf ready >"$F13_TEST_ROOT/window.ready"; sleep 60'
    ;;
  signal-double-fork)
    child_failure_phase=generation
    run_child_required double-fork python3 -c '
import os, signal, time
pid = os.fork()
if pid == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with open(os.environ["F13_TEST_ROOT"] + "/detached.pids", "x", encoding="ascii") as handle:
        handle.write(str(os.getpid()) + "\n")
    while True: time.sleep(1)
with open(os.environ["F13_TEST_ROOT"] + "/window.ready", "x", encoding="ascii") as handle:
    handle.write("ready\n")
while True: time.sleep(1)
'
    ;;
  signal-cleared)
    run_child_required cleared-command /bin/sh -c 'exit 0'
    printf 'window\n' >"${F13_TEST_ROOT}/window.ready"
    while :; do sleep 0.1; done
    ;;
  *) exit 98 ;;
esac
EOF
chmod 0700 "${child}"

parent_race="${work}/stub-parent-race.sh"
cat >"${parent_race}" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
source "${F13_TEST_FUNCTIONS}"
source "${F13_TEST_PARENT_FUNCTIONS}"
pid_starttime() { kill -0 "$1" 2>/dev/null || return 1; printf '1\n'; }
pid_is_same_process() { kill -0 "$1" 2>/dev/null; }
pid_state() { local state; state="$(ps -o stat= -p "$1" 2>/dev/null | tr -d '[:space:]')" || return 1; [[ -n "${state}" ]] || return 1; printf '%s\n' "${state:0:1}"; }
wait_numbered_step_gone() {
  [[ -z "${parent_srun_pid}" ]] || {
    for _ in $(seq 1 100); do kill -0 "${parent_srun_pid}" 2>/dev/null || break; sleep 0.01; done
  }
  printf gone >"${F13_TEST_ROOT}/step.gone"
}
cancel_exact_numbered_step_if_present() {
  printf gone >"${F13_TEST_ROOT}/step.gone"
  return 0
}
assert_retained_parent_unchanged() { [[ "$(cat "${F13_TEST_ROOT}/retained-parent")" == 'RUNNING|auh7-1b-gpu-215|guangyi.chen' ]]; }
parent_srun_pid=""
parent_srun_starttime=""
parent_srun_gate_read_fd=""
parent_srun_gate_write_fd=""
parent_owned_pid=""
parent_owned_starttime=""
parent_command_gate_read_fd=""
parent_command_gate_write_fd=""
parent_publisher_read_fd=""
parent_publisher_write_fd=""
parent_signal_status=0
parent_spawn_in_progress=false
parent_spawn_deferred_signal_status=0
parent_success_commit_active=false
parent_success_commit_deferred_signal_status=0
parent_success_committed=false
parent_signal_int() { parent_signal_handler 130; }
parent_signal_term() { parent_signal_handler 143; }
parent_signal_hup() { parent_signal_handler 129; }
trap parent_exit_handler EXIT
trap parent_signal_int INT
trap parent_signal_term TERM
trap parent_signal_hup HUP
case "${F13_TEST_MODE}" in
  parent-spawn)
    parent_spawn_in_progress=true
    kill -TERM "$$"
    kill -HUP "$$"
    /bin/sh -c 'trap "" TERM; while :; do sleep 1; done' &
    parent_srun_pid=$!
    parent_srun_starttime="$(pid_starttime "${parent_srun_pid}")"
    deferred="${parent_spawn_deferred_signal_status}"
    if (( deferred != 0 )); then
      trap '' INT TERM HUP
      parent_spawn_in_progress=false
      parent_spawn_deferred_signal_status=0
      parent_signal_handler "${deferred}"
    fi
    parent_spawn_in_progress=false
    deferred="${parent_spawn_deferred_signal_status}"
    if (( deferred != 0 )); then
      trap '' INT TERM HUP
      parent_spawn_deferred_signal_status=0
      parent_signal_handler "${deferred}"
    fi
    exit 97
    ;;
  parent-assigned)
    /bin/sh -c 'trap "" TERM; while :; do sleep 1; done' &
    parent_srun_pid=$!
    parent_srun_starttime="$(pid_starttime "${parent_srun_pid}")"
    printf ready >"${F13_TEST_ROOT}/window.ready"
    wait "${parent_srun_pid}"
    ;;
  parent-cleared)
    /bin/sh -c 'exit 0' &
    parent_srun_pid=$!
    parent_srun_starttime="$(pid_starttime "${parent_srun_pid}")" || parent_srun_starttime=""
    wait "${parent_srun_pid}" || true
    printf gone >"${F13_TEST_ROOT}/step.gone"
    parent_srun_pid=""
    parent_srun_starttime=""
    printf ready >"${F13_TEST_ROOT}/window.ready"
    while :; do sleep 0.1; done
    ;;
  parent-commit-before)
    parent_success_commit_active=true
    kill -TERM "$$"
    trap '' INT TERM HUP
    if (( parent_success_commit_deferred_signal_status != 0 )); then
      status="${parent_success_commit_deferred_signal_status}"
      parent_success_commit_active=false
      exit "${status}"
    fi
    exit 96
    ;;
  parent-commit-mid)
    parent_success_commit_active=true
    trap '' INT TERM HUP
    printf 'valid-' >"${F13_TEST_ROOT}/launcher.status"
    kill -TERM "$$"
    printf 'success\n' >>"${F13_TEST_ROOT}/launcher.status"
    chmod 0400 "${F13_TEST_ROOT}/launcher.status"
    parent_success_committed=true
    trap - EXIT
    exit 0
    ;;
  parent-commit-after)
    parent_success_commit_active=true
    trap '' INT TERM HUP
    printf 'valid-success\n' >"${F13_TEST_ROOT}/launcher.status"
    chmod 0400 "${F13_TEST_ROOT}/launcher.status"
    kill -TERM "$$"
    parent_success_committed=true
    trap - EXIT
    exit 0
    ;;
  *) exit 98 ;;
esac
EOF
chmod 0700 "${parent_race}"

fake_bin="${work}/bin"
mkdir -m 0700 "${fake_bin}"
cat >"${fake_bin}/srun" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
: >"${F13_TEST_ROOT}/srun.called"
exec "$@"
EOF
chmod 0700 "${fake_bin}/srun"

signal_spawner="${work}/signal-spawner.py"
cat >"${signal_spawner}" <<'EOF'
import os
import signal
import subprocess
import sys

pid_path, command = sys.argv[1], sys.argv[2:]

def reset_signals():
    for value in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(value, signal.SIG_DFL)

child = subprocess.Popen(command, preexec_fn=reset_signals)
descriptor = os.open(pid_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="ascii") as handle:
    handle.write(str(child.pid) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
status = child.wait()
raise SystemExit(status if status >= 0 else 128 - status)
EOF

durable_marker_hostile="${work}/durable-marker-hostile.py"
cat >"${durable_marker_hostile}" <<'EOF'
import importlib.util
import os
from pathlib import Path
import sys

controller_path = Path(sys.argv[1]).resolve(strict=True)
root = Path(sys.argv[2]).resolve(strict=True)
spec = importlib.util.spec_from_file_location("exp013_r6_controller_hostile", controller_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
value = {"schema_version": "hostile-durable-writer-v1", "payload": "x" * 4096}
shared = root / "shared-data-prep"
run_root = shared / "full30-action-arms-incomplete-exact2-r6-deadbeef-j136140-r1"
logs = run_root / "logs"
logs.mkdir(parents=True, mode=0o700)
module.SHARED_DATA_PREP_ROOT = shared

class NfsProbe:
    stdout = "nfs\n"

module.subprocess.run = lambda *_args, **_kwargs: NfsProbe()
real_write = module.os.write
real_fsync = module.os.fsync
real_open = module.os.open
real_fsync_directory = module._fsync_directory
real_unlink = os.unlink
real_close = os.close
real_fchmod = os.fchmod

def exercise(label, *, write=None, fsync=None, open_call=None, fsync_directory=None, expect_success=False):
    path = logs / f"{label}.json"
    success = root / f"{label}.success"
    module.os.write = write or real_write
    module.os.fsync = fsync or real_fsync
    module.os.open = open_call or real_open
    module._fsync_directory = fsync_directory or real_fsync_directory
    returned = False
    try:
        module._write_shared_terminal_marker(path, value)
        returned = True
        success.write_text("caller-observed-return\n", encoding="ascii")
    except Exception:
        returned = False
    finally:
        module.os.write = real_write
        module.os.fsync = real_fsync
        module.os.open = real_open
        module._fsync_directory = real_fsync_directory
    assert returned is expect_success, (label, returned)
    assert success.exists() is expect_success, (label, success)
    if expect_success:
        assert path.read_bytes() == module.canonical_json_bytes(value) + b"\n"
        assert path.stat().st_mode & 0o777 == 0o400

def short_write(fd, data):
    return real_write(fd, data[: max(1, min(7, len(data)))])

exercise("short-write-loop", write=short_write, expect_success=True)
exercise("zero-write", write=lambda _fd, _data: 0)
exercise("file-fsync-failure", fsync=lambda _fd: (_ for _ in ()).throw(OSError("fsync fault")))

open_count = 0
def reopen_failure(path, flags, mode=0o777):
    global open_count
    open_count += 1
    # create=1, parent-directory fsync open=2, same-path replay open=3.
    if open_count == 3:
        raise OSError("reopen fault")
    return real_open(path, flags, mode)
exercise("reopen-failure", open_call=reopen_failure)

replacement_path = logs / "replacement-race.json"
def replace_after_directory_fsync(parent):
    real_fsync_directory(parent)
    real_unlink(replacement_path)
    descriptor = real_open(replacement_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        real_fchmod(descriptor, 0o400)
        real_write(descriptor, b"{}\n")
        real_fsync(descriptor)
    finally:
        real_close(descriptor)
exercise("replacement-race", fsync_directory=replace_after_directory_fsync)

preexisting = logs / "preexisting.json"
preexisting.write_bytes(b"sentinel\n")
try:
    module._write_shared_terminal_marker(preexisting, value)
except Exception:
    pass
else:
    raise AssertionError("preexisting output was accepted")
assert preexisting.read_bytes() == b"sentinel\n"

target = root / "symlink-target"
target.write_bytes(b"target\n")
symlink = logs / "symlink.json"
symlink.symlink_to(target)
try:
    module._write_shared_terminal_marker(symlink, value)
except Exception:
    pass
else:
    raise AssertionError("symlink output was accepted")
assert target.read_bytes() == b"target\n"

print("DURABLE_MARKER_HOSTILE_PASS cases=7")
EOF

run_stub_parent() {
  local root="$1" mode="$2" child_status restore_errexit=false expected_plan_sha observed_plan_sha expected_blind_sha observed_blind_sha
  [[ $- == *e* ]] && restore_errexit=true
  mkdir -m 0700 "${root}"
  printf 'RUNNING|auh7-1b-gpu-215|guangyi.chen\n' >"${root}/retained-parent"
  printf 'controller-plan-original\n' >"${root}/controller-plan.json"
  expected_plan_sha="$(shasum -a 256 "${root}/controller-plan.json" | awk '{print $1}')"
  set +e
  F13_TEST_FUNCTIONS="${functions_file}" F13_TEST_COMMON="${common}" \
  F13_TEST_ROOT="${root}" F13_TEST_MODE="${mode}" \
    "${fake_bin}/srun" env -u SLURM_TMPDIR -u TMPDIR \
      -u GADP_NODE_LOCAL_SCRATCH -u GADP_NODE_LOCAL_SCRATCH_FSTYPE \
      bash "${child}"
  child_status=$?
  if [[ "${restore_errexit}" == true ]]; then set -e; else set +e; fi
  (( child_status == 0 )) || return "${child_status}"
  [[ "${mode}" != success-plan-replaced ]] || printf 'controller-plan-resigned-replacement\n' >"${root}/controller-plan.json"
  [[ "${mode}" != success-blind-replaced ]] || printf 'blind-replaced-after-child-exit\n' >"${root}/blind.manifest"
  [[ -f "${root}/terminal-ready.marker" && -f "${root}/retained-terminal.receipt" ]] || return 72
  observed_plan_sha="$(shasum -a 256 "${root}/controller-plan.json" | awk '{print $1}')" || return 75
  [[ "${observed_plan_sha}" == "${expected_plan_sha}" ]] || return 75
  expected_blind_sha="$(tr -d '[:space:]' <"${root}/attested-blind.sha256")" || return 76
  observed_blind_sha="$(shasum -a 256 "${root}/blind.manifest" | awk '{print $1}')" || return 76
  [[ "${observed_blind_sha}" == "${expected_blind_sha}" ]] || return 76
  [[ "$(cat "${root}/retained-parent")" == 'RUNNING|auh7-1b-gpu-215|guangyi.chen' ]] || return 73
  [[ -f "${root}/outer/inner/rank/cache/evidence.bin" && ! -e "${root}/deletion.attempt" ]] || return 74
  printf 'parent-success-after-srun-and-retained-validation\n' >"${root}/launcher.status"
}

wait_path() {
  local path="$1" attempt
  for attempt in $(seq 1 200); do [[ -e "${path}" ]] && return 0; sleep 0.01; done
  return 1
}

spawn_signal_child() {
  local root="$1" mode="$2"
  F13_TEST_FUNCTIONS="${functions_file}" F13_TEST_COMMON="${common}" F13_TEST_ROOT="${root}" F13_TEST_MODE="${mode}" \
    python3 "${signal_spawner}" "${root}/child.pid" \
      env -u SLURM_TMPDIR -u TMPDIR -u GADP_NODE_LOCAL_SCRATCH -u GADP_NODE_LOCAL_SCRATCH_FSTYPE bash "${child}" &
  spawned_wrapper_pid=$!
  wait_path "${root}/child.pid" || return 1
  spawned_child_pid="$(tr -d '[:space:]' <"${root}/child.pid")"
  [[ "${spawned_child_pid}" =~ ^[1-9][0-9]*$ ]]
}

spawn_parent_race() {
  local root="$1" mode="$2"
  printf 'RUNNING|auh7-1b-gpu-215|guangyi.chen\n' >"${root}/retained-parent"
  F13_TEST_FUNCTIONS="${functions_file}" F13_TEST_PARENT_FUNCTIONS="${parent_functions}" F13_TEST_ROOT="${root}" F13_TEST_MODE="${mode}" \
    python3 "${signal_spawner}" "${root}/parent.pid" bash "${parent_race}" &
  spawned_wrapper_pid=$!
  wait_path "${root}/parent.pid" || return 1
  spawned_child_pid="$(tr -d '[:space:]' <"${root}/parent.pid")"
  [[ "${spawned_child_pid}" =~ ^[1-9][0-9]*$ ]]
}

# H1: direct child replay with any caller scratch name (including empty) fails
# before the scratch prepare marker.
h1="${work}/h1"
mkdir -m 0700 "${h1}"
for scratch_name in SLURM_TMPDIR TMPDIR GADP_NODE_LOCAL_SCRATCH GADP_NODE_LOCAL_SCRATCH_FSTYPE; do
  for value_case in empty nonempty; do
    case_root="${h1}/${scratch_name}-${value_case}"
    mkdir -m 0700 "${case_root}"
    scratch_value=""
    [[ "${value_case}" == empty ]] || scratch_value=caller-value
    set +e
    F13_TEST_FUNCTIONS="${functions_file}" F13_TEST_COMMON="${common}" F13_TEST_ROOT="${case_root}" F13_TEST_MODE=success \
      env -u SLURM_TMPDIR -u TMPDIR -u GADP_NODE_LOCAL_SCRATCH -u GADP_NODE_LOCAL_SCRATCH_FSTYPE \
        "${scratch_name}=${scratch_value}" bash "${child}" >/dev/null 2>&1
    h1_status=$?
    set -e
    [[ ${h1_status} -eq 42 && ! -e "${case_root}/prepared" && ! -e "${case_root}/srun.called" ]] || fail_test "H1 ${scratch_name} ${value_case} rejection"
  done
done
require_source '[[ ! -v "${caller_scratch_name}" ]]'
pass_test "H1 caller scratch presence, including empty, rejects before prepare"

# H2: a real stub-srun boundary scrubs all four names.  The child retains the
# complete forensic tree and publishes only terminal-ready; the parent writes
# the sole success marker after the child exits zero and receipt replay passes.
h2="${work}/h2"
run_stub_parent "${h2}" success
[[ -f "${h2}/srun.called" && -f "${h2}/terminal-ready.marker" && -f "${h2}/launcher.status" ]] || fail_test "H2 normal parent-child retained commit"
[[ "$(cat "${h2}/outer/inner/rank/cache/evidence.bin")" == forensic-bytes && ! -e "${h2}/deletion.attempt" ]] || fail_test "H2 success tree was not retained byte-exact"
for replacement_mode in success-plan-replaced success-blind-replaced; do
  replacement_root="${work}/h2-${replacement_mode}"
  set +e
  run_stub_parent "${replacement_root}" "${replacement_mode}" >/dev/null 2>&1
  replacement_status=$?
  set -e
  [[ ${replacement_status} -ne 0 && ! -e "${replacement_root}/launcher.status" && -f "${replacement_root}/outer/inner/rank/cache/evidence.bin" ]] || fail_test "H2 ${replacement_mode} external-anchor replacement"
done
require_source 'env -u SLURM_TMPDIR -u TMPDIR -u GADP_NODE_LOCAL_SCRATCH -u GADP_NODE_LOCAL_SCRATCH_FSTYPE'
pass_test "H2 parent-to-stubbed-srun success retains tree and rejects post-child anchor replacement"

# H3: INT/TERM/HUP retain their conventional exact statuses across the
# unset/assigned/cleared PID windows.
for signal_case in 'INT:130:signal-unset' 'TERM:143:signal-assigned' 'HUP:129:signal-cleared'; do
  IFS=: read -r signal_name expected_status mode <<<"${signal_case}"
  root="${work}/h3-${signal_name}"
  mkdir -m 0700 "${root}"
  set +e
  spawn_signal_child "${root}" "${mode}" || { set -e; fail_test "H3 ${signal_name} spawn"; }
  wait_path "${root}/window.ready" || { kill -KILL "${spawned_child_pid}" 2>/dev/null || true; wait "${spawned_wrapper_pid}" 2>/dev/null || true; set -e; fail_test "H3 ${signal_name} readiness"; }
  kill -s "${signal_name}" "${spawned_child_pid}"
  wait "${spawned_wrapper_pid}"
  observed_status=$?
  set -e
  [[ ${observed_status} -eq ${expected_status} ]] || fail_test "H3 ${signal_name} expected ${expected_status}, got ${observed_status}"
done
spawn_deferred_root="${work}/h3-child-spawn-deferred"
mkdir -m 0700 "${spawn_deferred_root}"
set +e
spawn_signal_child "${spawn_deferred_root}" signal-spawn-deferred || { set -e; fail_test "H3 child spawn-deferred spawn"; }
wait "${spawned_wrapper_pid}"
spawn_deferred_status=$?
set -e
[[ ${spawn_deferred_status} -eq 143 && -f "${spawn_deferred_root}/spawn-delay.used" ]] || fail_test "H3 child spawn-deferred first-signal latch"
for parent_mode in parent-spawn parent-assigned parent-cleared; do
  parent_root="${work}/h3-${parent_mode}"
  mkdir -m 0700 "${parent_root}"
  set +e
  spawn_parent_race "${parent_root}" "${parent_mode}" || { set -e; fail_test "H3 ${parent_mode} spawn"; }
  if [[ "${parent_mode}" != parent-spawn ]]; then
    wait_path "${parent_root}/window.ready" || { kill -KILL "${spawned_child_pid}" 2>/dev/null || true; set -e; fail_test "H3 ${parent_mode} readiness"; }
    kill -TERM "${spawned_child_pid}"
  fi
  wait "${spawned_wrapper_pid}"
  parent_status=$?
  set -e
  [[ ${parent_status} -eq 143 && -f "${parent_root}/step.gone" && "$(cat "${parent_root}/retained-parent")" == 'RUNNING|auh7-1b-gpu-215|guangyi.chen' ]] || fail_test "H3 ${parent_mode} state machine"
done
pass_test "H3 child codes and first-signal spawn/assigned/cleared windows"

# H4: a foreground command that ignores TERM is boundedly escalated to KILL;
# the launcher exits with the original signal status and writes failure state.
h4="${work}/h4"
mkdir -m 0700 "${h4}"
set +e
spawn_signal_child "${h4}" signal-ignore-term || fail_test "H4 spawn"
wait_path "${h4}/window.ready" || fail_test "H4 readiness"
kill -TERM "${spawned_child_pid}"
wait "${spawned_wrapper_pid}"
h4_status=$?
set -e
[[ ${h4_status} -eq 143 && -f "${h4}/failure.receipt" ]] || fail_test "H4 bounded ignored TERM"
pass_test "H4 ignored TERM foreground is boundedly killed with status143"

# H5: the exact production EXIT handler converts an uncommitted status0 into
# 70 and emits failure evidence.
h5="${work}/h5"
mkdir -m 0700 "${h5}"
set +e
F13_TEST_FUNCTIONS="${functions_file}" F13_TEST_COMMON="${common}" F13_TEST_ROOT="${h5}" bash -c 'source "$F13_TEST_COMMON"; exit 0' >/dev/null 2>&1
h5_status=$?
set -e
[[ ${h5_status} -eq 70 && -f "${h5}/failure.receipt" ]] || fail_test "H5 uncommitted zero"
pass_test "H5 uncommitted child status0 is converted to fail-closed 70"

# H6: a second catchable signal cannot interrupt the first handler.  A signal
# during retained-terminal sealing is fail-closed.  At the parent's sole
# success boundary, pre-commit signals prevent a marker while mid/after-commit
# signals are ignored and the completed marker remains paired with status0.
h6="${work}/h6"
mkdir -m 0700 "${h6}"
set +e
spawn_signal_child "${h6}" signal-ignore-term || fail_test "H6 spawn"
wait_path "${h6}/window.ready" || fail_test "H6 readiness"
kill -TERM "${spawned_child_pid}"
sleep 0.05
kill -HUP "${spawned_child_pid}" 2>/dev/null || true
wait "${spawned_wrapper_pid}"
h6_status=$?
set -e
[[ ${h6_status} -eq 143 && -s "${h6}/failure.receipt" ]] || fail_test "H6 second signal latch"
h6_terminal="${work}/h6-retained-terminal"
mkdir -m 0700 "${h6_terminal}"
set +e
spawn_signal_child "${h6_terminal}" retained-terminal-signal || { set -e; fail_test "H6 retained-terminal spawn"; }
wait_path "${h6_terminal}/window.ready" || fail_test "H6 retained-terminal readiness"
kill -TERM "${spawned_child_pid}"
wait "${spawned_wrapper_pid}"
h6_terminal_status=$?
set -e
[[ ${h6_terminal_status} -eq 143 && -s "${h6_terminal}/failure.receipt" && ! -e "${h6_terminal}/terminal-ready.marker" && -f "${h6_terminal}/outer/inner/rank/cache/evidence.bin" ]] || fail_test "H6 retained-terminal signal"
for commit_case in parent-commit-before parent-commit-mid parent-commit-after; do
  commit_root="${work}/h6-${commit_case}"
  mkdir -m 0700 "${commit_root}"
  set +e
  spawn_parent_race "${commit_root}" "${commit_case}" || { set -e; fail_test "H6 ${commit_case} spawn"; }
  wait "${spawned_wrapper_pid}"
  commit_status=$?
  set -e
  if [[ "${commit_case}" == parent-commit-before ]]; then
    [[ ${commit_status} -eq 143 && ! -e "${commit_root}/launcher.status" ]] || fail_test "H6 pre-commit signal"
  else
    [[ ${commit_status} -eq 0 && "$(cat "${commit_root}/launcher.status")" == valid-success ]] || fail_test "H6 ${commit_case}"
  fi
done
require_source 'seal-child-terminal-ready'
require_source 'prepare-parent-generation-status'
require_source 'resident-publish-parent-generation-status'
require_source 'BOX-EXP-013-r6-PARENT-PUBLISH-READY'
require_source 'BOX-EXP-013-r6-PARENT-PUBLISH-COMMIT'
require_source 'BOX-EXP-013-r6-PARENT-PUBLISH-ACK'
pass_test "H6 signal commit boundary and resident durable-publication protocol are coherent"

# H7: critical command substitutions are never hidden in readonly/local,
# every parent srun PID has an explicit start-time/error branch, and a failed
# cgroup UID-authority lookup is not misreported as an empty census.
if rg -n '(^|[[:space:]])(readonly|local)[[:space:]]+[A-Za-z_][A-Za-z0-9_]*(=.*)?\$\(' "${launcher}" >/dev/null; then
  fail_test "H7 readonly/local masks command substitution status"
fi
require_source 'starttime_status=$?'
require_source 'local srun PID identity unavailable'
require_source 'self_uid="$(id -u)" || return 1'
for census_mode in uid-census-failure identity-census-failure malformed-census-failure unreadable-census-failure live-census-detected; do
  h7_census="${work}/h7-${census_mode}"
  set +e
  run_stub_parent "${h7_census}" "${census_mode}" >/dev/null 2>&1
  h7_census_status=$?
  set -e
  [[ ${h7_census_status} -ne 0 && -f "${h7_census}/outer/inner/rank/cache/evidence.bin" && -f "${h7_census}/failure.receipt" && ! -e "${h7_census}/census.failopen" && ! -e "${h7_census}/terminal-ready.marker" && ! -e "${h7_census}/launcher.status" ]] || fail_test "H7 ${census_mode} fail-close"
done
if rg -n '=\$\(collect_owned_step_cgroup_pids|collect_owned_step_cgroup_pids[[:space:]]+[|][[:space:]]|[|][[:space:]]+collect_owned_step_cgroup_pids' "${launcher}" >/dev/null; then
  fail_test "H7 cgroup collector runs in a subshell"
fi
pass_test "H7 substitution/srun identity and stable cgroup authority fail closed"

# H8: the launcher performs no path-based create/chmod of the renderer lock.
# It accepts only the controller's fd-created, signed task-bind path; the
# controller suite owns the O_EXCL/O_NOFOLLOW collision hostiles.
require_source 'renderer_load_lock path)'
require_source '[[ "${model_load_lock}" == "${task_scratch}/renderer-load.lock" ]]'
forbid_source ': >"${model_load_lock}"'
forbid_source 'chmod 0400 "${model_load_lock}"'
pass_test "H8 renderer lock mutation is delegated to signed fd authority"

# H9: inner creation is controller-owned and immediately signed; every later
# generator/audit/attestation/retained-terminal phase consumes that binding.
# No shell mktemp, deletion command, or obsolete cleanup receipt remains.
for anchor in create-and-bind-child-task-scratch validate-child-task-scratch-bind seal-child-terminal-physical-attestation seal-child-scratch-retained-terminal seal-child-terminal-ready validate-child-scratch-retained-terminal prepare-parent-generation-status resident-publish-parent-generation-status; do require_source "${anchor}"; done
[[ $(rg -F --count -- '--task-scratch-bind "${task_scratch_binding}"' "${launcher}") -ge 6 ]] || fail_test "H9 task bind chain count"
rg -U --quiet 'run-sp4 \\\n[[:space:]]+--controller-plan .*\\\n[[:space:]]+--scratch-prepare ' "${launcher}" || fail_test "H9 run-sp4 missing strict plan/prepare gate"
rg -U --quiet 'audit-exact2 \\\n[[:space:]]+--controller-plan .*\\\n[[:space:]]+--scratch-prepare ' "${launcher}" || fail_test "H9 audit-exact2 missing strict plan/prepare gate"
forbid_source 'mktemp -d -- "${scratch_parent}'
for obsolete in cleanup-child-scratch validate-child-scratch-cleanup child-scratch-cleanup.json; do forbid_source "${obsolete}"; done
for obsolete_marker in child-success.status child_success_committed_by_parent_after_srun experiment_completion=true; do forbid_source "${obsolete_marker}"; done
if rg -n '^[[:space:]]*(command[[:space:]]+)?(rm|rmdir|unlink|rmtree)([[:space:]]|$)|os[.](unlink|remove)|shutil[.]rmtree' "${launcher}" >/dev/null; then
  fail_test "H9 launcher contains a scratch deletion primitive"
fi
pass_test "H9 signed inode chain retains scratch and contains no deletion path"

# H10: nested failure keeps the whole forensic subtree and its bytes, writes
# failure evidence, and creates no terminal-ready or parent success marker.
# A real double-forked, TERM-ignoring descendant is terminated through the
# fixture cgroup census without modifying scratch bytes.
h10="${work}/h10"
set +e
run_stub_parent "${h10}" nested-failure >/dev/null 2>&1
h10_status=$?
set -e
[[ ${h10_status} -eq 23 && "$(cat "${h10}/outer/inner/rank/cache/evidence.bin")" == forensic-bytes && -f "${h10}/failure.receipt" && ! -e "${h10}/terminal-ready.marker" && ! -e "${h10}/launcher.status" ]] || fail_test "H10 nested failure retention"
h10_detached="${work}/h10-detached"
mkdir -m 0700 "${h10_detached}"
set +e
spawn_signal_child "${h10_detached}" signal-double-fork || { set -e; fail_test "H10 double-fork spawn"; }
wait_path "${h10_detached}/window.ready" || fail_test "H10 double-fork readiness"
wait_path "${h10_detached}/detached.pids" || fail_test "H10 detached PID publication"
kill -TERM "${spawned_child_pid}"
wait "${spawned_wrapper_pid}"
h10_detached_status=$?
set -e
h10_detached_pid="$(tr -d '[:space:]' <"${h10_detached}/detached.pids")"
h10_detached_state="$(ps -o stat= -p "${h10_detached_pid}" 2>/dev/null | tr -d '[:space:]')" || h10_detached_state=""
[[ ${h10_detached_status} -eq 143 && ( -z "${h10_detached_state}" || "${h10_detached_state:0:1}" == Z ) ]] || fail_test "H10 detached cgroup teardown"
pass_test "H10 nested fixture retention and detached descendant teardown"

# H11: an external identity drift plus failed signed-failure sealing never
# produces a plain retained=true claim.  The original bytes remain under the
# drifted name, no child-ready/parent-success marker exists, and delete
# primitives are hostile in the normal success fixture.
h11="${work}/h11"
set +e
run_stub_parent "${h11}" outer-drift-failure >/dev/null 2>&1
h11_status=$?
set -e
[[ ${h11_status} -eq 74 && -f "${h11}/failure-seal.failed" && -f "${h11}/failure.receipt" ]] || fail_test "H11 drift failure receipt"
[[ "$(cat "${h11}/outer.identity-drifted/inner/rank/cache/evidence.bin")" == forensic-bytes ]] || fail_test "H11 drifted forensic bytes"
rg -Fx --quiet 'retention_claim=see-signed-receipt' "${h11}/failure.receipt" || fail_test "H11 signed-receipt indirection absent"
rg -Fx --quiet 'retention_not_claimed=true' "${h11}/failure.receipt" || fail_test "H11 no-claim flag absent"
! rg -n '^(scratch_retained_for_forensics|retained_forensics|retention_claim)=(true|present)$' "${h11}/failure.receipt" >/dev/null || fail_test "H11 false retained claim"
[[ ! -e "${h11}/terminal-ready.marker" && ! -e "${h11}/launcher.status" ]] || fail_test "H11 false success publication"
[[ "$(cat "${h11}/retained-parent")" == 'RUNNING|auh7-1b-gpu-215|guangyi.chen' ]] || fail_test "H11 retained parent changed"
[[ ! -e "${h2}/deletion.attempt" && -f "${h2}/outer/inner/rank/cache/evidence.bin" ]] || fail_test "H11 delete monkeypatch was invoked on success"
require_source '/usr/bin/scancel --signal=TERM -- "${step}"'
require_source '/usr/bin/scancel --signal=KILL -- "${step}"'
forbid_source '/usr/bin/scancel --signal=TERM -- "${holder_job}"'
forbid_source '/usr/bin/scancel --signal=KILL -- "${holder_job}"'
pass_test "H11 drift/failure-seal cannot overclaim retention or delete scratch"

[[ ${pass_count} -eq 11 ]] || fail_test "hostile test count differs"
printf 'EXP013_R6_HOSTILE_PASS count=%s\n' "${pass_count}"
