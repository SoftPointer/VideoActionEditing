#!/usr/bin/env bash
# Run the four-stratum Stage-B source-noised-carrier pretext across holders
# 135407 and 135411 using an already sealed source-self dataset.  Holder 135412
# is retained and checked but never receives a child step.  This controller
# creates only identity-bound numbered child steps and never releases, cancels,
# or requeues any holder allocation.

set -Eeuo pipefail
umask 077

fail() { echo "[snc-stage-b-two-holder] ERROR: $*" >&2; exit 2; }

readonly controller_requested="${BASH_SOURCE[0]}"
[[ "${controller_requested}" == /* ]] || fail "controller must be invoked by absolute path"
[[ -f "${controller_requested}" && ! -L "${controller_requested}" ]] || fail "controller must be an absolute plain file"
readonly controller_source="$(readlink -f -- "${controller_requested}")"
readonly holder_user="guangyi.chen"
readonly retained_job=135412
readonly retained_node=auh7-1b-gpu-293
readonly train_job0=135407
readonly train_node0=auh7-1b-gpu-260
readonly train_job1=135411
readonly train_node1=auh7-1b-gpu-214
readonly expected_rank_cache_exec_sha256=f416f8f687a61788d4f687e500dae04af677072ec205fb560016cb5249b4f9c5
readonly expected_checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly expected_checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly expected_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly expected_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly memory_peak_limit_bytes=55834574848

sha256_file() { sha256sum "$1" | awk '{print $1}'; }

is_allowed_holder() {
  case "$1" in 135407|135411) return 0 ;; *) return 1 ;; esac
}

holder_node() {
  case "$1" in
    135407) printf '%s\n' auh7-1b-gpu-260 ;;
    135411) printf '%s\n' auh7-1b-gpu-214 ;;
    135412) printf '%s\n' auh7-1b-gpu-293 ;;
    *) return 2 ;;
  esac
}

assert_rocm_snapshot_idle() {
  local snapshot="$1" label="$2" expected_count="$3" use_count memory_count busy
  use_count="$(awk '/GPU use \(%\)/ {count += 1} END {print count + 0}' <<<"${snapshot}")"
  memory_count="$(awk '/GPU Memory Allocated \(VRAM%\)/ {count += 1} END {print count + 0}' <<<"${snapshot}")"
  busy="$(awk '
    /GPU use \(%\)/ || /GPU Memory Allocated \(VRAM%\)/ {
      value=$NF; gsub(/[^0-9]/,"",value); if ((value + 0) != 0) print
    }
  ' <<<"${snapshot}")"
  if [[ "${label}" == child-visible* ]]; then
    [[ "${use_count}" -ge "${expected_count}" && "${use_count}" == "${memory_count}" ]] || \
      fail "${label}: ROCm inventory exposes fewer than ${expected_count} devices"
  else
    [[ "${use_count}" == "${expected_count}" && "${memory_count}" == "${expected_count}" ]] || \
      fail "${label}: ROCm inventory differs from exact${expected_count}"
  fi
  [[ -z "${busy}" ]] || fail "${label}: a GPU is already active"
}

child_preflight() {
  local expected_gpus="$1" python_bin="$2" expected_sha snapshot
  shift 2
  expected_sha="${BERNINI_SNC_STAGE_B_TWO_HOLDER_CONTROLLER_SHA256:?set controller SHA}"
  [[ "${expected_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "node controller pin differs"
  [[ "$(sha256_file "${controller_source}")" == "${expected_sha}" ]] || fail "node controller bytes differ"
  is_allowed_holder "${SLURM_JOB_ID:?child requires holder job}" || fail "child holder is outside allowlist"
  [[ "$(holder_node "${SLURM_JOB_ID}")" == "$(hostname -s)" ]] || fail "child holder/node binding differs"
  [[ "${SLURM_STEP_ID:?child requires numbered step}" =~ ^[0-9]+$ ]] || fail "child step identity differs"
  [[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "node Python differs"
  snapshot="$(rocm-smi --showuse --showmemuse --showpids)"
  assert_rocm_snapshot_idle "${snapshot}" "child-visible preflight" "${expected_gpus}"
  "${python_bin}" -B - "${expected_gpus}" <<'PY'
import sys, torch
n = int(sys.argv[1])
assert torch.version.hip and torch.cuda.is_available()
assert torch.cuda.device_count() == n
assert all(torch.cuda.memory_allocated(i) == 0 for i in range(n))
PY
  "$@"
}

resolve_cgroup2_memory_current() {
  local cgroup_path="" mount_root="" mount_point="" suffix="" candidate="" value=""
  [[ -r /proc/self/cgroup && -r /proc/self/mountinfo ]] || return 70
  cgroup_path="$(awk -F: '$1=="0" && $2=="" {print $3; exit}' /proc/self/cgroup)"
  read -r mount_root mount_point < <(awk '$0 ~ / - cgroup2 / {print $4, $5; exit}' /proc/self/mountinfo)
  [[ "${cgroup_path}" == /* && "${mount_root}" == /* && "${mount_point}" == /* ]] || return 70
  if [[ "${mount_root}" == / ]]; then
    suffix="${cgroup_path}"
  elif [[ "${cgroup_path}" == "${mount_root}" ]]; then
    suffix=""
  elif [[ "${cgroup_path}" == "${mount_root}/"* ]]; then
    suffix="/${cgroup_path#"${mount_root}/"}"
  else
    return 70
  fi
  candidate="${mount_point%/}${suffix}/memory.current"
  [[ "${candidate}" == /sys/fs/cgroup/* && -f "${candidate}" && ! -L "${candidate}" && -r "${candidate}" ]] || return 70
  value="$(tr -d '[:space:]' <"${candidate}" 2>/dev/null || true)"
  [[ "${value}" =~ ^[0-9]+$ ]] || return 70
  printf '%s\n' "${candidate}"
}

write_memory_evidence() {
  local evidence="$1" node_rank="$2" status="$3" peak="$4" samples="$5" source="$6"
  printf 'schema=bernini-snc-stage-b-child-cgroup-memory-sampled-v1 job_id=%s step_id=%s node_rank=%s status=%s sampled_peak_bytes=%s samples=%s interval_seconds=0.1 limit_bytes=%s source=%s\n' \
    "${SLURM_JOB_ID:?}" "${SLURM_STEP_ID:?}" "${node_rank}" "${status}" "${peak}" "${samples}" "${memory_peak_limit_bytes}" "${source}" \
    >"${evidence}"
  chmod 0400 "${evidence}"
}

sample_cgroup_memory_current() {
  local evidence="$1" node_rank="$2" child_pid="$3" candidate="$4" current="" peak=0 samples=0 status=unavailable
  while kill -0 "${child_pid}" 2>/dev/null; do
    current="$(tr -d '[:space:]' <"${candidate}" 2>/dev/null || true)"
    if [[ ! "${current}" =~ ^[0-9]+$ ]]; then
      write_memory_evidence "${evidence}" "${node_rank}" unavailable unavailable "${samples}" "${candidate}"
      return 70
    fi
    (( current > peak )) && peak="${current}"
    (( samples += 1 ))
    sleep 0.1
  done
  [[ "${peak}" =~ ^[0-9]+$ && "${samples}" -gt 0 ]] && status=available
  write_memory_evidence "${evidence}" "${node_rank}" "${status}" "${peak:-unavailable}" "${samples}" "${candidate}"
  [[ "${status}" == available && "${peak}" =~ ^[0-9]+$ && "${peak}" -lt "${memory_peak_limit_bytes}" ]]
}

if [[ "${1:-}" == __train_exec ]]; then
  shift
  evidence="${1:?train child needs peak evidence}"
  node_rank="${2:?train child needs node rank}"
  shift 2
  python_bin="$1"
  shift
  child_preflight 2 "${python_bin}" /usr/bin/true
  cgroup_counter="$(resolve_cgroup2_memory_current || true)"
  if [[ -z "${cgroup_counter}" ]]; then
    write_memory_evidence "${evidence}" "${node_rank}" unavailable unavailable 0 unresolved-cgroup2-memory.current
    exit 70
  fi
  set +e
  "$@" &
  training_pid=$!
  sample_cgroup_memory_current "${evidence}" "${node_rank}" "${training_pid}" "${cgroup_counter}"
  peak_status=$?
  wait "${training_pid}"
  child_status=$?
  set -e
  (( child_status == 0 && peak_status == 0 )) || exit 70
  exit 0
fi

usage() {
  cat >&2 <<'EOF'
usage: auh_train_source_noised_carrier_stage_b_two_holder_v1.sh run

Required environment:
  BERNINI_SNC_STAGE_B_RUN_ROOT, BERNINI_SNC_STAGE_B_SPEC_SHA256
  BERNINI_SNC_STAGE_B_EXISTING_MATERIALIZED (mandatory sealed two-file dataset)
  BERNINI_SNC_STAGE_B_EXISTING_PARQUET_SHA256
  BERNINI_SNC_STAGE_B_EXISTING_RECEIPT_SHA256
  BERNINI_SNC_STAGE_B_EXISTING_RECEIPT_DIGEST
  BERNINI_SNC_STAGE_B_SOURCE_ARCHIVE, BERNINI_SNC_STAGE_B_SOURCE_ARCHIVE_SHA256
  BERNINI_SNC_STAGE_B_SOURCE_MANIFEST, BERNINI_SNC_STAGE_B_SOURCE_MANIFEST_SHA256
  BERNINI_SNC_STAGE_B_SOURCE_REVISION
  BERNINI_SNC_STAGE_B_TWO_HOLDER_CONTROLLER_SHA256
  BERNINI_SNC_STAGE_B_RANK_CACHE_EXEC, BERNINI_OFFICIAL_ROOT, BERNINI_VEOMNI_ROOT
  BERNINI_CHECKPOINT, BERNINI_CHECKPOINT_CONTENT_MANIFEST
  BERNINI_SNC_STAGE_B_PYTHON_BIN
EOF
  exit 2
}

[[ "${1:-}" == run && $# == 1 ]] || usage

readonly run_root="${BERNINI_SNC_STAGE_B_RUN_ROOT:?set fresh absolute run root}"
readonly spec_sha="${BERNINI_SNC_STAGE_B_SPEC_SHA256:?set materialization spec SHA}"
readonly source_archive="${BERNINI_SNC_STAGE_B_SOURCE_ARCHIVE:?set source archive}"
readonly source_archive_sha="${BERNINI_SNC_STAGE_B_SOURCE_ARCHIVE_SHA256:?set source archive SHA}"
readonly source_manifest="${BERNINI_SNC_STAGE_B_SOURCE_MANIFEST:?set source manifest}"
readonly source_manifest_sha="${BERNINI_SNC_STAGE_B_SOURCE_MANIFEST_SHA256:?set source manifest SHA}"
readonly source_revision="${BERNINI_SNC_STAGE_B_SOURCE_REVISION:?set content-closure SHA1}"
readonly controller_sha="${BERNINI_SNC_STAGE_B_TWO_HOLDER_CONTROLLER_SHA256:?set controller SHA}"
readonly rank_cache_exec="${BERNINI_SNC_STAGE_B_RANK_CACHE_EXEC:?set rank-cache worker}"
readonly bernini_root="${BERNINI_OFFICIAL_ROOT:?set Bernini root}"
readonly veomni_root="${BERNINI_VEOMNI_ROOT:?set VeOmni root}"
readonly checkpoint="${BERNINI_CHECKPOINT:?set checkpoint}"
readonly checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
readonly python_bin="${BERNINI_SNC_STAGE_B_PYTHON_BIN:?set vace Python}"
readonly existing_materialized="${BERNINI_SNC_STAGE_B_EXISTING_MATERIALIZED:?set sealed existing materialized dataset}"
readonly existing_parquet_sha="${BERNINI_SNC_STAGE_B_EXISTING_PARQUET_SHA256:?set existing parquet SHA}"
readonly existing_receipt_sha="${BERNINI_SNC_STAGE_B_EXISTING_RECEIPT_SHA256:?set existing receipt SHA}"
readonly existing_receipt_digest="${BERNINI_SNC_STAGE_B_EXISTING_RECEIPT_DIGEST:?set existing receipt digest}"
readonly master_port="${BERNINI_SNC_STAGE_B_MASTER_PORT:-29843}"
readonly train_timeout="${BERNINI_SNC_STAGE_B_TRAIN_TIMEOUT_SECONDS:-21600}"

for name in run_root existing_materialized source_archive source_manifest rank_cache_exec bernini_root veomni_root checkpoint checkpoint_manifest python_bin; do
  value="${!name}"
  [[ "${value}" == /* ]] || fail "${name} must be absolute"
done
[[ -d "${existing_materialized}" && ! -L "${existing_materialized}" ]] || fail "existing materialized dataset differs"
[[ "$(readlink -f -- "${existing_materialized}")" == "${existing_materialized}" ]] || \
  fail "existing materialized path is not canonical"
for digest in existing_parquet_sha existing_receipt_sha existing_receipt_digest; do
  [[ "${!digest}" =~ ^[0-9a-f]{64}$ ]] || fail "${digest} differs"
done
for digest in spec_sha source_archive_sha source_manifest_sha controller_sha; do
  [[ "${!digest}" =~ ^[0-9a-f]{64}$ ]] || fail "${digest} differs"
done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${master_port}" =~ ^[0-9]+$ ]] && (( master_port >= 1024 && master_port <= 65535 )) || fail "master port differs"
[[ "${train_timeout}" =~ ^[1-9][0-9]*$ ]] || fail "timeout differs"
for path in "${source_archive}" "${source_manifest}" "${checkpoint_manifest}" "${controller_source}"; do
  [[ -f "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "sealed file differs: ${path}"
done
[[ -x "${rank_cache_exec}" && ! -L "${rank_cache_exec}" ]] || fail "rank-cache worker differs"
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"
for path in "${bernini_root}" "${veomni_root}" "${checkpoint}"; do [[ -d "${path}" && ! -L "${path}" ]] || fail "runtime root differs: ${path}"; done
[[ "$(sha256_file "${controller_source}")" == "${controller_sha}" ]] || fail "controller bytes differ"
[[ "$(sha256_file "${rank_cache_exec}")" == "${expected_rank_cache_exec_sha256}" ]] || fail "rank-cache worker bytes differ"
[[ "$(sha256_file "${source_archive}")" == "${source_archive_sha}" ]] || fail "source archive bytes differ"
[[ "$(sha256_file "${source_manifest}")" == "${source_manifest_sha}" ]] || fail "source manifest bytes differ"
[[ "$(sha256_file "${checkpoint_manifest}")" == "${expected_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest bytes differ"
[[ "${run_root}" != / && "$(realpath -m -- "${run_root}")" == "${run_root}" ]] || fail "run root must be safe and canonical"
[[ ! -e "${run_root}" && ! -L "${run_root}" ]] || fail "run root must be fresh"

assert_parent_running() {
  local job="$1" expected_node record
  expected_node="$(holder_node "${job}")"
  record="$(scontrol show job -o "${job}")"
  [[ "${record}" == *"JobId=${job} "* && "${record}" == *"JobState=RUNNING"* ]] || fail "holder ${job} is not RUNNING"
  [[ "${record}" == *"UserId=${holder_user}"* ]] || fail "holder ${job} owner differs"
  [[ "${record}" == *"NodeList=${expected_node}"* && "${record}" == *"NumCPUs=64"* ]] || fail "holder ${job} topology differs"
  [[ "${record}" == *"AllocTRES=cpu=64,mem=64G,"* && "${record}" == *"gres/gpu:mi210=8"* ]] || fail "holder ${job} resources differ"
}

assert_all_parents_running() {
  assert_parent_running "${retained_job}"
  assert_parent_running "${train_job0}"
  assert_parent_running "${train_job1}"
}

numbered_steps() { squeue -s -j "$1" -h -o '%i' | awk '/[.][0-9]+$/ {print}'; }
assert_no_numbered_steps() { local steps; steps="$(numbered_steps "$1")"; [[ -z "${steps}" ]] || fail "holder $1 has numbered step: ${steps}"; }

assert_remote_idle_once() {
  local job="$1" node="$2" processes hidden snapshot
  assert_parent_running "${job}"
  assert_no_numbered_steps "${job}"
  processes="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${node}" "ps -u ${holder_user} -ww -o pid=,ppid=,comm=,args=")"
  hidden="$(awk -v job="${job}" '
    { line=$0; comm=$3
      if (index(line,"/var/spool/slurmd/job" job "/slurm_script")) next
      if (comm=="sleep" && line ~ /sleep infinity[[:space:]]*$/) next
      if ((comm=="bash" || comm=="sh") && index(line,"holding allocation across nodes:") && index(line,"sleep infinity")) next
      if ((comm=="bash" || comm=="sh") && index(line,"ps -u guangyi.chen -ww -o")) next
      if (comm=="systemd" || comm=="(sd-pam)" || comm=="podman" || comm=="dbus-daemon" || comm=="sshd" || comm=="ps") next
      print }
  ' <<<"${processes}")"
  [[ -z "${hidden}" ]] || fail "holder ${job}/${node} has a hidden user process"
  snapshot="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${node}" 'rocm-smi --showuse --showmemuse --showpids')"
  assert_rocm_snapshot_idle "${snapshot}" "${node} outer preflight" 8
}

assert_idle_twice() {
  local label="$1"; shift
  local pair job node
  for pair in "$@"; do job="${pair%%:*}"; node="${pair#*:}"; assert_remote_idle_once "${job}" "${node}"; done
  sleep 2
  for pair in "$@"; do job="${pair%%:*}"; node="${pair#*:}"; assert_remote_idle_once "${job}" "${node}"; done
  echo "IDLE_TWICE label=${label}"
}

assert_master_port_free() {
  local listeners
  listeners="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${train_node0}" "ss -H -ltn 'sport = :${master_port}'")"
  [[ -z "${listeners}" ]] || fail "master port is occupied"
}

registered_child_pids=()
declare -A child_starttime=() child_cmdline_sha=() child_exe=() child_job=() child_node=()
launch_critical=0
pending_signal=""

proc_field() { awk -v field="$2" '{print $field}' "/proc/$1/stat" 2>/dev/null; }
child_identity_matches() {
  local pid="$1" ppid start exe cmd_sha cmdline job node
  [[ "${pid}" =~ ^[1-9][0-9]*$ && -n "${child_starttime[${pid}]-}" ]] || return 1
  ppid="$(proc_field "${pid}" 4)"; start="$(proc_field "${pid}" 22)"
  exe="$(readlink -f -- "/proc/${pid}/exe" 2>/dev/null || true)"
  cmd_sha="$(sha256_file "/proc/${pid}/cmdline" 2>/dev/null || true)"
  cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
  job="${child_job[${pid}]}"; node="${child_node[${pid}]}"
  [[ "${ppid}" == "$$" && "${start}" == "${child_starttime[${pid}]}" && "${exe}" == "${child_exe[${pid}]}" ]] || return 1
  [[ "$(basename -- "${exe}")" == srun && "${cmd_sha}" == "${child_cmdline_sha[${pid}]}" ]] || return 1
  [[ " ${cmdline} " == *" --jobid=${job} "* && " ${cmdline} " == *" --nodelist=${node} "* ]]
}

register_child_pid() {
  local pid="$1" job="$2" node="$3" ppid start exe cmdline
  for _ in {1..100}; do
    if [[ -r "/proc/${pid}/cmdline" ]]; then
      ppid="$(proc_field "${pid}" 4)"; start="$(proc_field "${pid}" 22)"
      exe="$(readlink -f -- "/proc/${pid}/exe" 2>/dev/null || true)"
      cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
      if [[ "${ppid}" == "$$" && "$(basename -- "${exe:-missing}")" == srun && " ${cmdline} " == *" --jobid=${job} "* && " ${cmdline} " == *" --nodelist=${node} "* ]]; then
        child_starttime["${pid}"]="${start}"; child_cmdline_sha["${pid}"]="$(sha256_file "/proc/${pid}/cmdline")"
        child_exe["${pid}"]="${exe}"; child_job["${pid}"]="${job}"; child_node["${pid}"]="${node}"
        registered_child_pids+=("${pid}"); return 0
      fi
    else break; fi
    sleep 0.02
  done
  return 1
}

safe_signal_child() {
  if child_identity_matches "$1"; then kill -"$2" "$1" 2>/dev/null || true
  elif [[ -e "/proc/$1" ]]; then echo "[snc-stage-b-two-holder] REFUSE_SIGNAL identity mismatch pid=$1 signal=$2" >&2; fi
}

unregister_child_pid() {
  local retired="$1" pid kept=()
  for pid in "${registered_child_pids[@]:-}"; do [[ "${pid}" == "${retired}" ]] || kept+=("${pid}"); done
  registered_child_pids=("${kept[@]}")
  unset 'child_starttime['"${retired}"']' 'child_cmdline_sha['"${retired}"']' 'child_exe['"${retired}"']' 'child_job['"${retired}"']' 'child_node['"${retired}"']'
}

terminate_registered_children() {
  local pid
  for pid in "${registered_child_pids[@]:-}"; do safe_signal_child "${pid}" TERM; done
  for pid in "${registered_child_pids[@]:-}"; do
    for _ in {1..30}; do kill -0 "${pid}" 2>/dev/null || break; sleep 1; done
    kill -0 "${pid}" 2>/dev/null && safe_signal_child "${pid}" KILL
    wait "${pid}" 2>/dev/null || true; unregister_child_pid "${pid}"
  done
}

wait_for_steps_gone() {
  local job steps jobs=("${train_job0}" "${train_job1}")
  for job in "${jobs[@]}"; do
    for _ in {1..60}; do steps="$(numbered_steps "${job}")"; [[ -z "${steps}" ]] && break; sleep 1; done
    [[ -z "${steps:-}" ]] || return 1
  done
}

cleanup_on_exit() {
  local status=$?
  trap '' INT TERM HUP
  trap - EXIT
  terminate_registered_children
  wait_for_steps_gone || status=70
  exit "${status}"
}
signal_exit() { pending_signal="$1"; (( launch_critical == 1 )) || exit 130; }
trap 'signal_exit INT' INT
trap 'signal_exit TERM' TERM
trap 'signal_exit HUP' HUP
trap cleanup_on_exit EXIT

assert_all_parents_running
assert_idle_twice startup-existing-materialized "${train_job0}:${train_node0}" "${train_job1}:${train_node1}"
mkdir -m 0700 "${run_root}" "${run_root}/logs" "${run_root}/runtime-source"

"${python_bin}" -I -S -B - "${source_archive}" "${source_manifest}" "${run_root}/runtime-source" "${source_revision}" <<'PY'
import hashlib, json, stat, sys, tarfile
from pathlib import Path
archive, manifest_path, out = map(Path, sys.argv[1:4]); revision = sys.argv[4]
raw = manifest_path.read_bytes(); manifest = json.loads(raw.decode("ascii"))
unsigned = dict(manifest); declared = unsigned.pop("manifest_digest", None)
canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
assert declared == hashlib.sha256(canonical).hexdigest()
assert manifest["schema_version"] == "bernini-source-noised-carrier-stage-b-release-v1"
assert manifest["release_generation"] == "r4" and manifest["file_count"] == 12
assert manifest["revision_kind"] == "content-closure-sha1" and manifest["content_closure_sha1"] == revision
assert manifest["git_commit_claimed"] is False and manifest["exact_member_closure"] is True
rows = manifest["files"]
expected_relatives = [
    "source_self_role_repaint.py",
    "source_self_runtime.py",
    "train_source_self_role_repaint.py",
    "train_lora.py",
    "assets/source_self_role_repaint_canary_spec_v2.json",
    "tools/materialize_source_self_role_repaint.py",
    "tools/materialize_ramp_motion_analogy_vae.py",
    "tools/materialize_vae.py",
    "tools/build_renderer_dataset.py",
    "inference_sigma_strata.py",
    "source_noised_ladder_v1.py",
    "train_source_noised_carrier_strata_v1.py",
]
assert [row["path"] for row in rows] == expected_relatives
expected = ["methods/bernini_action_editing/" + row["path"] for row in rows]
with tarfile.open(archive, "r:") as handle:
    members = handle.getmembers()
    assert [item.name for item in members] == expected
    for member, row in zip(members, rows):
        stream = handle.extractfile(member)
        assert member.isfile() and not member.issym() and not member.islnk() and stream is not None
        payload = stream.read()
        assert member.uid == member.gid == member.mtime == 0 and stat.S_IMODE(member.mode) == 0o444
        assert len(payload) == row["size"] == member.size and hashlib.sha256(payload).hexdigest() == row["sha256"]
    handle.extractall(out, filter="data")
PY

readonly method_root="${run_root}/runtime-source/methods/bernini_action_editing"
readonly materialized="${existing_materialized}"
readonly training="${run_root}/stage_b_four_strata"
find "${method_root}" -type f -exec chmod 0400 {} +

launch_child() {
  local job="$1" node="$2" log="$3"; shift 3
  local pid
  launch_critical=1
  [[ -z "${pending_signal}" ]] || { launch_critical=0; exit 130; }
  srun --jobid="${job}" --nodelist="${node}" --nodes=1 --exclusive --exact --kill-on-bad-exit=1 \
    "$@" >"${log}" 2>&1 &
  pid=$!
  register_child_pid "${pid}" "${job}" "${node}" || { launch_critical=0; wait "${pid}" 2>/dev/null || true; fail "cannot bind child srun PID"; }
  launched_pid="${pid}"
  launch_critical=0
  echo "CHILD_SRUN_REGISTERED pid=${pid} job=${job} node=${node}"
  [[ -z "${pending_signal}" ]] || exit 130
}

printf 'reused_materialized=%s\n' "${materialized}" >"${run_root}/logs/materialize-reuse.txt"
chmod 0400 "${run_root}/logs/materialize-reuse.txt"
[[ -f "${materialized}/dataset.parquet" && -f "${materialized}/receipt.json" ]] || fail "sealed dataset closure differs"
"${python_bin}" -I -B - "${materialized}" "${spec_sha}" \
  "${existing_parquet_sha}" "${existing_receipt_sha}" "${existing_receipt_digest}" <<'PY'
import hashlib, json, sys
from pathlib import Path
root=Path(sys.argv[1]); expected=sys.argv[2]
parquet_pin, receipt_pin, digest_pin = sys.argv[3:6]
assert {p.name for p in root.iterdir()} == {"dataset.parquet","receipt.json"}
for path in root.iterdir(): assert path.is_file() and not path.is_symlink()
receipt_raw=(root/"receipt.json").read_bytes()
r=json.loads(receipt_raw); d=r.pop("receipt_digest")
raw=json.dumps(r,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
assert hashlib.sha256(raw).hexdigest()==d and r["complete"] is True and r["spec"]["sha256"]==expected
assert r["dataset"]["rows"] == 2 and r["independent_vae_encode_calls_per_row"] == 6
assert r["paired_dataset_accessed"] is False and r["prior_posterior_accessed"] is False
parquet_sha=hashlib.sha256((root/"dataset.parquet").read_bytes()).hexdigest()
assert parquet_sha == r["dataset"]["sha256"]
assert parquet_sha == parquet_pin
assert hashlib.sha256(receipt_raw).hexdigest() == receipt_pin
assert d == digest_pin
PY
assert_all_parents_running

assert_idle_twice pre-training "${train_job0}:${train_node0}" "${train_job1}:${train_node1}"
assert_master_port_free
sleep 1
assert_master_port_free

launch_train_node() {
  local job="$1" node="$2" node_rank="$3" log="$4" peak="$5"
  launch_child "${job}" "${node}" "${log}" --ntasks=1 --cpus-per-task=16 --mem=56G --gres=gpu:mi210:2 \
    env BERNINI_SNC_STAGE_B_TWO_HOLDER_CONTROLLER_SHA256="${controller_sha}" \
      BERNINI_HELDOUT_RANK_CACHE_TOKEN="snc-stage-b-world4-${source_revision:0:12}" BERNINI_HELDOUT_PYTHON_BIN="${python_bin}" \
      PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 \
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH="${method_root}" \
      "${controller_source}" __train_exec "${peak}" "${node_rank}" "${python_bin}" \
      "${python_bin}" -B -m torch.distributed.run --nnodes=2 --nproc_per_node=2 --node_rank="${node_rank}" \
        --master_addr="${train_node0}" --master_port="${master_port}" --no_python "${rank_cache_exec}" \
        "${method_root}/train_source_noised_carrier_strata_v1.py" --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
        --checkpoint "${checkpoint}" --dataset-root "${materialized}" --expected-materialization-spec-sha256 "${spec_sha}" \
        --output "${training}" --mode source-carrier-strata-v1 --parallel-topology world4-dp1-sp4 \
        --adapter-block-scope early-mid-0-22 --expected-bernini-commit "${expected_bernini_commit}" \
        --expected-veomni-commit "${expected_veomni_commit}" --expected-checkpoint-tree-sha256 "${expected_checkpoint_tree_sha256}" \
        --method-source-revision "${source_revision}" --method-source-revision-kind content-closure-sha1 \
        --method-source-archive-sha256 "${source_archive_sha}" --method-source-manifest-sha256 "${source_manifest_sha}" \
        --ack-upstream-training-use-forbidden --ack-forward-noising-is-not-inversion
}

launch_train_node "${train_job0}" "${train_node0}" 0 "${run_root}/logs/train-node0.log" "${run_root}/logs/train-node0-memory-peak.txt"
train_pid0="${launched_pid}"
launch_train_node "${train_job1}" "${train_node1}" 1 "${run_root}/logs/train-node1.log" "${run_root}/logs/train-node1-memory-peak.txt"
train_pid1="${launched_pid}"
rc0=0; rc1=0; done0=0; done1=0; failure_ticks=0; started_at="${SECONDS}"
while (( done0 == 0 || done1 == 0 )); do
  if (( done0 == 0 )) && ! kill -0 "${train_pid0}" 2>/dev/null; then wait "${train_pid0}" || rc0=$?; unregister_child_pid "${train_pid0}"; done0=1; fi
  if (( done1 == 0 )) && ! kill -0 "${train_pid1}" 2>/dev/null; then wait "${train_pid1}" || rc1=$?; unregister_child_pid "${train_pid1}"; done1=1; fi
  if (( SECONDS - started_at >= train_timeout )); then rc0=124; rc1=124; fi
  if (( rc0 != 0 || rc1 != 0 )); then
    (( done0 == 0 )) && safe_signal_child "${train_pid0}" TERM
    (( done1 == 0 )) && safe_signal_child "${train_pid1}" TERM
    (( failure_ticks += 1 ))
    if (( failure_ticks >= 30 )); then (( done0 == 0 )) && safe_signal_child "${train_pid0}" KILL; (( done1 == 0 )) && safe_signal_child "${train_pid1}" KILL; fi
  fi
  (( done0 == 0 || done1 == 0 )) && sleep 1
done
(( rc0 == 0 && rc1 == 0 )) || { tail -n 160 "${run_root}/logs/train-node0.log" >&2 || true; tail -n 160 "${run_root}/logs/train-node1.log" >&2 || true; fail "WORLD4 training failed"; }
wait_for_steps_gone || fail "training step remained"
assert_all_parents_running

verify_step_memory() {
  local evidence="$1" expected_job="$2" expected_node_rank="$3" accounting="$4" step_id="" row=""
  step_id="$(sed -n 's/.* step_id=\([0-9][0-9]*\) .*/\1/p' "${evidence}")"
  [[ "${step_id}" =~ ^[0-9]+$ ]] || fail "sampled memory step identity differs"
  for _ in {1..30}; do
    row="$(sacct -j "${expected_job}.${step_id}" -n -P -o JobIDRaw,State,ExitCode,MaxRSS | awk -F'|' -v id="${expected_job}.${step_id}" '$1==id {print; exit}')"
    [[ -n "${row}" && "${row}" == *"|COMPLETED|0:0|"* && -n "${row##*|}" ]] && break
    sleep 1
  done
  [[ -n "${row}" && "${row}" == *"|COMPLETED|0:0|"* ]] || fail "exact child accounting differs"
  "${python_bin}" -I -B - "${evidence}" "${row}" "${expected_job}" "${expected_node_rank}" "${step_id}" "${memory_peak_limit_bytes}" "${accounting}" <<'PY'
import json, os, re, sys
from pathlib import Path
evidence, row, expected_job, expected_node_rank, expected_step, limit, output = sys.argv[1:]
fields = {}
for token in Path(evidence).read_text(encoding="ascii").strip().split():
    if "=" in token:
        key, value = token.split("=", 1); fields[key] = value
assert set(fields) == {"schema","job_id","step_id","node_rank","status","sampled_peak_bytes","samples","interval_seconds","limit_bytes","source"}
assert fields["schema"] == "bernini-snc-stage-b-child-cgroup-memory-sampled-v1"
assert fields["job_id"] == expected_job and fields["step_id"] == expected_step
assert fields["node_rank"] == expected_node_rank and fields["status"] == "available"
assert fields["interval_seconds"] == "0.1" and fields["limit_bytes"] == limit
assert fields["source"].startswith("/sys/fs/cgroup/") and fields["source"].endswith("/memory.current")
sampled = int(fields["sampled_peak_bytes"]); assert int(fields["samples"]) > 0
job_id, state, exit_code, max_rss = row.split("|")[:4]
assert job_id == f"{expected_job}.{expected_step}" and state == "COMPLETED" and exit_code == "0:0"
m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?)", max_rss); assert m
scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}[m.group(2)]
sacct_peak = int(float(m.group(1)) * scale); bound = int(limit)
assert sampled < bound and sacct_peak < bound
payload = {"schema_version":"bernini-snc-stage-b-child-memory-crosscheck-v1","job_id":expected_job,
           "step_id":expected_step,"sampled_memory_current_peak_bytes":sampled,
           "sacct_max_rss_raw":max_rss,"sacct_max_rss_bytes":sacct_peak,"limit_bytes":bound,
           "both_below_limit":True}
tmp = Path(output + ".tmp")
tmp.write_text(json.dumps(payload, sort_keys=True, separators=(",",":")) + "\n", encoding="ascii")
os.chmod(tmp, 0o400); os.link(tmp, output); tmp.unlink()
PY
}
verify_step_memory "${run_root}/logs/train-node0-memory-peak.txt" "${train_job0}" 0 "${run_root}/logs/train-node0-memory-crosscheck.json"
verify_step_memory "${run_root}/logs/train-node1-memory-peak.txt" "${train_job1}" 1 "${run_root}/logs/train-node1-memory-crosscheck.json"

for _ in {1..120}; do
  [[ -f "${training}/receipt.json" ]] && break
  sleep 1
done
[[ -f "${training}/receipt.json" ]] || fail "published training receipt did not become visible to controller"

"${python_bin}" -I -B - "${training}" "${source_revision}" "${source_archive_sha}" "${source_manifest_sha}" \
  "${existing_parquet_sha}" "${existing_receipt_sha}" "${existing_receipt_digest}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); revision,archive_sha,manifest_sha=sys.argv[2:5]
parquet_pin, materialization_receipt_pin, materialization_digest_pin=sys.argv[5:8]
assert {p.name for p in root.iterdir()} == {"adapter.safetensors","optimizer.pt","history.json","receipt.json"}
r=json.loads((root/"receipt.json").read_text(encoding="utf-8")); declared=r.pop("receipt_digest")
raw=json.dumps(r,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
assert hashlib.sha256(raw).hexdigest()==declared and r["complete"] is True
assert r["schema_version"]=="bernini-source-noised-carrier-training-receipt-v1"
assert r["mode"]=="source-carrier-strata-v1" and r["optimizer_steps"]==4
assert r["positive_gradient_steps"]==4 and r["registered_schedule_indices"]==[16,29,35,38]
assert r["optimizer_step_per_registered_sigma"] is True
assert r["all_registered_strata_optimizer_authorized"] is True
assert r["late_or_low_sigma_zero_update_gate_present"] is False
assert r["exact40_schedule_sha256"]=="3e5ad4473d133318026cc9e8f32399782bf06313691b58870c89d9c4c87c3d03"
assert r["distributed"]["profile"]=="world4-dp1-sp4" and r["distributed"]["world_size"]==4
assert r["distributed"]["physical_data_parallel_size"]==1 and r["distributed"]["ulysses_sequence_parallel_size"]==4
placement=r["distributed"]["placement"]
assert placement=={"nodes":2,"local_world_size":2,"ranks_per_node":2,"sp4_crosses_nodes":True,"preferred_world4_placement":True}
assert r["distributed"]["checkpoint_recomputation_route_context_replayed"] is True
assert r["method_source_revision"]==revision and r["method_source_revision_kind"]=="content-closure-sha1"
assert r["method_source_archive_sha256"]==archive_sha and r["method_source_manifest_sha256"]==manifest_sha
assert r["dataset"]["parquet_sha256"]==parquet_pin
assert r["dataset"]["receipt_sha256"]==materialization_receipt_pin
assert r["dataset"]["receipt_digest"]==materialization_digest_pin
assert r["forward_noising"]["forward_noising_only"] is True
assert r["forward_noising"]["same_epsilon_target_and_donor_verified_every_logical_record"] is True
assert r["forward_noising"]["same_sigma_target_and_donor_verified_every_logical_record"] is True
assert r["forward_noising"]["different_epsilon_across_eight_logical_step_samples"] is True
assert r["forward_noising"]["clean_source_references_routed_every_logical_record"] is True
assert r["forward_noising"]["inversion_claimed"] is False
assert r["forward_noising"]["reverse_ode_executed"] is False
assert r["forward_noising"]["solver_state_replayed"] is False
assert r["forward_noising"]["exact_roundtrip_claimed"] is False
assert r["upstream_training_use_forbidden_acknowledged"] is True
assert r["forward_noising_not_inversion_acknowledged"] is True
for key in ("semantic_motion_preservation_claimed","natural_semantic_action_learned","action_editing_claim_authorized","video_quality_claim_authorized","scientific_claim_authorized","method_success_claimed","long_training_scientific_gate_passed","long_training_automatically_submitted"):
    assert r[key] is False
assert r["initial_adapter_sha256"] != r["final_adapter_sha256"]
assert r["adapter_changed_each_optimizer_step"] is True
for name,digest in r["artifacts"].items(): assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
h=json.loads((root/"history.json").read_text(encoding="utf-8"))
assert h["optimizer_steps"]==4 and h["registered_schedule_indices"]==[16,29,35,38]
assert len(h["steps"])==4 and [step["schedule_index"] for step in h["steps"]]==[16,29,35,38]
assert all(step["optimizer_step_executed"] is True and len(step["logical_records"])==2 for step in h["steps"])
before=r["initial_adapter_sha256"]
records=[]
for expected_step,step in enumerate(h["steps"]):
    assert step["optimizer_step_zero_based"]==expected_step
    assert step["parameter_sha256_before_step"]==before
    after=step["parameter_sha256_after_step"]
    assert after != before
    before=after
    assert [record["logical_arm"] for record in step["logical_records"]]==[0,1]
    records.extend(step["logical_records"])
assert before==r["final_adapter_sha256"] and len(records)==8
assert len({record["epsilon_sha256"] for record in records})==8
for record in records:
    binding=dict(record["shared_noise_binding"]); digest=binding.pop("digest")
    canonical=json.dumps(binding,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
    assert hashlib.sha256(canonical).hexdigest()==digest
    assert binding["epsilon_sha256"]==record["epsilon_sha256"]==record["tensor_identities"]["epsilon"]
    for key in ("same_epsilon_object_reused_during_target_and_donor_construction","target_formula_recomputed_and_equal","donor_formula_recomputed_and_equal","same_sigma_registered_coordinate_reused","clean_source_references_routed","references_independently_encoded_from_source_rgb","forward_noising_only"):
        assert binding[key] is True
    for key in ("reference_from_video_posterior_slice","inversion_claimed","reverse_ode_executed","solver_state_replayed","exact_roundtrip_claimed"):
        assert binding[key] is False
PY

assert_all_parents_running
printf 'COMPLETE_STAGE_B_PRETEXT_ONLY topology=world4-dp1-sp4 optimizer_steps=4 schedule_indices=16,29,35,38 materialization_mode=reused_sealed_existing training_holders=%s,%s parents_retained=135407,135411,135412 method_success_claimed=false inversion_claimed=false\n' \
  "${train_job0}" "${train_job1}" >"${run_root}/controller.COMPLETE"
chmod 0400 "${run_root}/controller.COMPLETE"
trap - EXIT INT TERM HUP
echo "COMPLETE_STAGE_B_PRETEXT_ONLY output=${run_root}"
