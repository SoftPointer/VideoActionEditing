#!/usr/bin/env bash
# Display-only pure-T2V anchor generation for sealed PAIR-v5 reserve IID00435.
# Parent allocations are immutable.  This controller owns and may signal only
# the exact numbered child srun PIDs it creates; it never invokes Stage-B.
set -Eeuo pipefail
umask 077

fail() { echo "[pair-v5-display-anchor-two-holder] ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

readonly controller_requested="${BASH_SOURCE[0]}"
[[ "${controller_requested}" == /* && -f "${controller_requested}" && ! -L "${controller_requested}" ]] || fail "controller must be an absolute plain file"
readonly controller_source="$(readlink -f -- "${controller_requested}")"
readonly holder_user=guangyi.chen
readonly work_job0="${BERNINI_PAIR_V5_ANCHOR_WORK_JOB0:?set first work holder}"
readonly work_job1="${BERNINI_PAIR_V5_ANCHOR_WORK_JOB1:?set second work holder}"

holder_node() {
  case "$1" in
    135407) printf '%s\n' auh7-1b-gpu-260 ;;
    135411) printf '%s\n' auh7-1b-gpu-214 ;;
    135412) printf '%s\n' auh7-1b-gpu-293 ;;
    *) return 2 ;;
  esac
}
[[ "${work_job0}" != "${work_job1}" ]] || fail "work holders must be distinct"
holder_node "${work_job0}" >/dev/null || fail "first work holder is outside exact allowlist"
holder_node "${work_job1}" >/dev/null || fail "second work holder is outside exact allowlist"
case "${work_job0}:${work_job1}" in
  135407:135411|135411:135407) retained_job=135412 ;;
  135407:135412|135412:135407) retained_job=135411 ;;
  135411:135412|135412:135411) retained_job=135407 ;;
  *) fail "work-holder pair is outside exact pair closure" ;;
esac
readonly retained_job
readonly work_node0="$(holder_node "${work_job0}")"
readonly work_node1="$(holder_node "${work_job1}")"
readonly retained_node="$(holder_node "${retained_job}")"
readonly expected_source_archive_sha=f9360fcef6bdcb9e37345515fb85d18e4c444fd2b100de35aeb0c1a55a98ac55
readonly expected_source_revision=17cc2c73d774e14cdd10bd2ceea4afbaf4b0be26
readonly expected_overlay_sha=e4d2f998ce61c007ca61c7d66278cbcbc0576bf152010d35d543162cf264064d
readonly expected_native_generator_sha=a60c37591c40206c6130185f1a2d2a7a8e473f5af4425205e268ae4a8b58f334
readonly expected_bank_generator_sha=e19e353d7e83ce7a7fe37bc958dd67e58ae6ae772fafaba8cc40bfb2097e3db6
readonly expected_selection_sha=a4baa1aea27f6497ca2dd615cc09b2b90eee37173f506e60ae7d630c41886be6
readonly expected_registry_sha=204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c
readonly expected_rank_cache_sha=f416f8f687a61788d4f687e500dae04af677072ec205fb560016cb5249b4f9c5
readonly expected_checkpoint_manifest_sha=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly source_sha=b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1
readonly memory_peak_limit_bytes=55834574848

is_work_holder() { [[ "$1" == "${work_job0}" || "$1" == "${work_job1}" ]]; }

assert_rocm_idle() {
  local snapshot="$1" expected="$2" label="$3" uses mems busy
  uses="$(awk '/GPU use \(%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  mems="$(awk '/GPU Memory Allocated \(VRAM%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  busy="$(awk '/GPU use \(%\)/ || /GPU Memory Allocated \(VRAM%\)/ {v=$NF; gsub(/[^0-9]/,"",v); if(v+0!=0)print}' <<<"${snapshot}")"
  if [[ "${label}" == child-visible ]]; then
    [[ "${uses}" -ge "${expected}" && "${uses}" == "${mems}" ]] || fail "${label}: GPU inventory differs"
  else
    [[ "${uses}" == "${expected}" && "${mems}" == "${expected}" ]] || fail "${label}: GPU inventory differs"
  fi
  [[ -z "${busy}" ]] || fail "${label}: GPU already active"
}

child_preflight() {
  local python_bin="$1" expected_sha snapshot
  expected_sha="${BERNINI_PAIR_V5_ANCHOR_CONTROLLER_SHA256:?set controller SHA}"
  [[ "$(sha256_file "${controller_source}")" == "${expected_sha}" ]] || fail "child controller bytes differ"
  is_work_holder "${SLURM_JOB_ID:?child holder required}" || fail "child holder outside work allowlist"
  [[ "$(holder_node "${SLURM_JOB_ID}")" == "$(hostname -s)" ]] || fail "child holder/node binding differs"
  [[ "${SLURM_STEP_ID:?numbered child step required}" =~ ^[0-9]+$ ]] || fail "child step identity differs"
  [[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "child Python differs"
  snapshot="$(rocm-smi --showuse --showmemuse --showpids)"
  assert_rocm_idle "${snapshot}" 2 child-visible
  "${python_bin}" -B - <<'PY'
import torch
assert torch.version.hip and torch.cuda.is_available()
assert torch.cuda.device_count() == 2
assert all(torch.cuda.memory_allocated(i) == 0 for i in range(2))
PY
}

resolve_memory_counter() {
  local cg root mount suffix candidate value
  cg="$(awk -F: '$1=="0"&&$2==""{print $3;exit}' /proc/self/cgroup)"
  read -r root mount < <(awk '$0~/ - cgroup2 /{print $4,$5;exit}' /proc/self/mountinfo)
  [[ "${cg}" == /* && "${root}" == /* && "${mount}" == /* ]] || return 70
  if [[ "${root}" == / ]]; then suffix="${cg}"; elif [[ "${cg}" == "${root}" ]]; then suffix="";
  elif [[ "${cg}" == "${root}/"* ]]; then suffix="/${cg#"${root}/"}"; else return 70; fi
  candidate="${mount%/}${suffix}/memory.current"
  [[ "${candidate}" == /sys/fs/cgroup/* && -f "${candidate}" && ! -L "${candidate}" && -r "${candidate}" ]] || return 70
  value="$(tr -d '[:space:]' <"${candidate}" 2>/dev/null || true)"; [[ "${value}" =~ ^[0-9]+$ ]] || return 70
  printf '%s\n' "${candidate}"
}

write_memory_evidence() {
  printf 'schema=bernini-pair-v5-display-anchor-child-cgroup-memory-v1 job_id=%s step_id=%s node_rank=%s status=%s sampled_peak_bytes=%s samples=%s interval_seconds=0.1 limit_bytes=%s source=%s\n' \
    "${SLURM_JOB_ID:?}" "${SLURM_STEP_ID:?}" "$2" "$3" "$4" "$5" "${memory_peak_limit_bytes}" "$6" >"$1"
  chmod 0400 "$1"
}

sample_memory() {
  local evidence="$1" rank="$2" pid="$3" counter="$4" current peak=0 samples=0
  while kill -0 "${pid}" 2>/dev/null; do
    current="$(tr -d '[:space:]' <"${counter}" 2>/dev/null || true)"
    [[ "${current}" =~ ^[0-9]+$ ]] || { write_memory_evidence "${evidence}" "${rank}" unavailable unavailable "${samples}" "${counter}"; return 70; }
    (( current > peak )) && peak="${current}"; (( samples += 1 )); sleep 0.1
  done
  write_memory_evidence "${evidence}" "${rank}" available "${peak}" "${samples}" "${counter}"
  (( samples > 0 && peak < memory_peak_limit_bytes ))
}

if [[ "${1:-}" == __anchor_exec ]]; then
  shift; evidence="${1:?memory evidence}"; rank="${2:?node rank}"; python_bin="${3:?Python}"; shift 3
  child_preflight "${python_bin}"
  counter="$(resolve_memory_counter || true)"
  [[ -n "${counter}" ]] || { write_memory_evidence "${evidence}" "${rank}" unavailable unavailable 0 unresolved-cgroup2-memory.current; exit 70; }
  set +e; "$@" & payload_pid=$!; sample_memory "${evidence}" "${rank}" "${payload_pid}" "${counter}"; peak_rc=$?
  wait "${payload_pid}"; payload_rc=$?; set -e
  (( payload_rc == 0 && peak_rc == 0 )) || exit 70
  exit 0
fi

usage() {
  cat >&2 <<'EOF'
usage: auh_generate_pair_v5_reserve_display_anchor_two_holder_v1.sh run
Required environment: BERNINI_PAIR_V5_ANCHOR_RUN_ROOT, _SOURCE_ARCHIVE,
_SOURCE_ARCHIVE_SHA256, _SOURCE_REVISION, _OVERLAY, _OVERLAY_SHA256,
_CONTROLLER_SHA256, _RANK_CACHE_EXEC, _SOURCE_VIDEO, _PYTHON_BIN,
_WORK_JOB0, _WORK_JOB1, optional _PROFILE={action-only|family4},
BERNINI_OFFICIAL_ROOT, BERNINI_VEOMNI_ROOT, BERNINI_CHECKPOINT,
BERNINI_CHECKPOINT_CONTENT_MANIFEST.
EOF
  exit 2
}
[[ "${1:-}" == run && $# == 1 ]] || usage

readonly run_root="${BERNINI_PAIR_V5_ANCHOR_RUN_ROOT:?}"
readonly source_archive="${BERNINI_PAIR_V5_ANCHOR_SOURCE_ARCHIVE:?}"
readonly source_archive_sha="${BERNINI_PAIR_V5_ANCHOR_SOURCE_ARCHIVE_SHA256:?}"
readonly source_revision="${BERNINI_PAIR_V5_ANCHOR_SOURCE_REVISION:?}"
readonly overlay="${BERNINI_PAIR_V5_ANCHOR_OVERLAY:?}"
readonly overlay_sha="${BERNINI_PAIR_V5_ANCHOR_OVERLAY_SHA256:?}"
readonly controller_sha="${BERNINI_PAIR_V5_ANCHOR_CONTROLLER_SHA256:?}"
readonly rank_cache_exec="${BERNINI_PAIR_V5_ANCHOR_RANK_CACHE_EXEC:?}"
readonly source_video="${BERNINI_PAIR_V5_ANCHOR_SOURCE_VIDEO:?}"
readonly python_bin="${BERNINI_PAIR_V5_ANCHOR_PYTHON_BIN:?}"
readonly profile="${BERNINI_PAIR_V5_ANCHOR_PROFILE:-action-only}"
readonly bernini_root="${BERNINI_OFFICIAL_ROOT:?}"
readonly veomni_root="${BERNINI_VEOMNI_ROOT:?}"
readonly checkpoint="${BERNINI_CHECKPOINT:?}"
readonly checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?}"
readonly timeout_seconds="${BERNINI_PAIR_V5_ANCHOR_TIMEOUT_SECONDS:-21600}"
readonly first_port="${BERNINI_PAIR_V5_ANCHOR_FIRST_PORT:-29961}"

for name in run_root source_archive overlay rank_cache_exec source_video python_bin bernini_root veomni_root checkpoint checkpoint_manifest; do
  [[ "${!name}" == /* ]] || fail "${name} must be absolute"
done
for digest in source_archive_sha overlay_sha controller_sha; do [[ "${!digest}" =~ ^[0-9a-f]{64}$ ]] || fail "${digest} differs"; done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${source_archive_sha}" == "${expected_source_archive_sha}" && "${source_revision}" == "${expected_source_revision}" ]] || fail "PAIR-v5 source authority differs"
[[ "${overlay_sha}" == "${expected_overlay_sha}" ]] || fail "display overlay authority differs"
[[ "${profile}" == action-only || "${profile}" == family4 ]] || fail "anchor profile differs"
[[ "${timeout_seconds}" =~ ^[1-9][0-9]*$ && "${first_port}" =~ ^[0-9]+$ ]] || fail "timeout/port differs"
for path in "${source_archive}" "${overlay}" "${source_video}" "${checkpoint_manifest}" "${controller_source}"; do
  [[ -f "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "sealed file differs: ${path}"
done
for path in "${rank_cache_exec}" "${python_bin}"; do [[ -x "${path}" && ! -L "${path}" ]] || fail "executable differs: ${path}"; done
for path in "${bernini_root}" "${veomni_root}" "${checkpoint}"; do
  [[ -d "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "runtime root differs: ${path}"
done
[[ "${run_root}" != / && "$(realpath -m -- "${run_root}")" == "${run_root}" && ! -e "${run_root}" && ! -L "${run_root}" ]] || fail "run root must be fresh canonical"
[[ "$(sha256_file "${controller_source}")" == "${controller_sha}" ]] || fail "controller bytes differ"
[[ "$(sha256_file "${rank_cache_exec}")" == "${expected_rank_cache_sha}" ]] || fail "rank cache bytes differ"
[[ "$(sha256_file "${source_archive}")" == "${source_archive_sha}" ]] || fail "source archive bytes differ"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "$(sha256_file "${overlay}")" == "${overlay_sha}" ]] || fail "overlay bytes differ"
[[ "$(sha256_file "${source_video}")" == "${source_sha}" ]] || fail "source video bytes differ"
[[ "$(sha256_file "${checkpoint_manifest}")" == "${expected_checkpoint_manifest_sha}" ]] || fail "checkpoint manifest differs"

assert_parent_running() {
  local job="$1" node record; node="$(holder_node "${job}")"; record="$(scontrol show job -o "${job}")"
  [[ "${record}" == *"JobId=${job} "* && "${record}" == *"JobState=RUNNING"* && "${record}" == *"UserId=${holder_user}"* ]] || fail "parent ${job} state/owner differs"
  [[ "${record}" == *"NodeList=${node}"* && "${record}" == *"NumCPUs=64"* && "${record}" == *"AllocTRES=cpu=64,mem=64G,"* && "${record}" == *"gres/gpu:mi210=8"* ]] || fail "parent ${job} topology differs"
}
assert_all_parents_running() { assert_parent_running "${work_job0}"; assert_parent_running "${work_job1}"; assert_parent_running "${retained_job}"; }
numbered_steps() { squeue -s -j "$1" -h -o '%i' | awk '/[.][0-9]+$/{print}'; }
assert_remote_idle_once() {
  local job="$1" node="$2" steps processes hidden snapshot
  assert_parent_running "${job}"; steps="$(numbered_steps "${job}")"; [[ -z "${steps}" ]] || fail "foreign/existing child on ${job}: ${steps}"
  processes="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${node}" "ps -u ${holder_user} -ww -o pid=,ppid=,comm=,args=")"
  hidden="$(awk -v job="${job}" '{line=$0;c=$3;if(index(line,"/var/spool/slurmd/job"job"/slurm_script"))next;if(c=="sleep"&&line~/sleep infinity[[:space:]]*$/)next;if((c=="bash"||c=="sh")&&index(line,"holding allocation across nodes:")&&index(line,"sleep infinity"))next;if((c=="bash"||c=="sh")&&index(line,"ps -u guangyi.chen -ww -o"))next;if(c=="systemd"||c=="(sd-pam)"||c=="podman"||c=="dbus-daemon"||c=="sshd"||c=="ps")next;print}' <<<"${processes}")"
  [[ -z "${hidden}" ]] || fail "hidden user process on ${job}/${node}"
  snapshot="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${node}" 'rocm-smi --showuse --showmemuse --showpids')"; assert_rocm_idle "${snapshot}" 8 outer
}
assert_idle_twice() {
  local label="$1"; assert_remote_idle_once "${work_job0}" "${work_node0}"; assert_remote_idle_once "${work_job1}" "${work_node1}"; sleep 2
  assert_remote_idle_once "${work_job0}" "${work_node0}"; assert_remote_idle_once "${work_job1}" "${work_node1}"; echo "IDLE_TWICE ${label}"
}
assert_port_free() {
  local port="$1" found; found="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${work_node0}" "ss -H -ltn 'sport = :${port}'")"; [[ -z "${found}" ]] || fail "master port ${port} occupied"
}

registered_pids=()
declare -A pid_start=() pid_cmd_sha=() pid_exe=() pid_job=() pid_node=()
proc_field() { awk -v f="$2" '{print $f}' "/proc/$1/stat" 2>/dev/null; }
pid_identity_matches() {
  local pid="$1" ppid start exe cmdline
  [[ "${pid}" =~ ^[1-9][0-9]*$ && -n "${pid_start[${pid}]-}" ]] || return 1
  ppid="$(proc_field "${pid}" 4)"; start="$(proc_field "${pid}" 22)"; exe="$(readlink -f -- "/proc/${pid}/exe" 2>/dev/null || true)"
  cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
  [[ "${ppid}" == "$$" && "${start}" == "${pid_start[${pid}]}" && "${exe}" == "${pid_exe[${pid}]}" ]] || return 1
  [[ "$(basename -- "${exe}")" == srun && "$(sha256_file "/proc/${pid}/cmdline" 2>/dev/null || true)" == "${pid_cmd_sha[${pid}]}" ]] || return 1
  [[ " ${cmdline} " == *" --jobid=${pid_job[${pid}]} "* && " ${cmdline} " == *" --nodelist=${pid_node[${pid}]} "* ]]
}
register_pid() {
  local pid="$1" job="$2" node="$3" ppid start exe cmdline
  for _ in {1..100}; do
    if [[ -r "/proc/${pid}/cmdline" ]]; then
      ppid="$(proc_field "${pid}" 4)"; start="$(proc_field "${pid}" 22)"; exe="$(readlink -f -- "/proc/${pid}/exe" 2>/dev/null || true)"
      cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
      if [[ "${ppid}" == "$$" && "$(basename -- "${exe:-missing}")" == srun && " ${cmdline} " == *" --jobid=${job} "* && " ${cmdline} " == *" --nodelist=${node} "* ]]; then
        pid_start["${pid}"]="${start}"; pid_cmd_sha["${pid}"]="$(sha256_file "/proc/${pid}/cmdline")"; pid_exe["${pid}"]="${exe}"
        pid_job["${pid}"]="${job}"; pid_node["${pid}"]="${node}"; registered_pids+=("${pid}"); return 0
      fi
    else break; fi
    sleep 0.02
  done
  return 1
}
signal_owned_pid() { if pid_identity_matches "$1"; then kill -"$2" "$1" 2>/dev/null || true; elif [[ -e "/proc/$1" ]]; then echo "REFUSE_SIGNAL pid=$1" >&2; fi; }
unregister_pid() {
  local dead="$1" pid kept=(); for pid in "${registered_pids[@]:-}"; do [[ "${pid}" == "${dead}" ]] || kept+=("${pid}"); done
  registered_pids=("${kept[@]}"); unset 'pid_start['"${dead}"']' 'pid_cmd_sha['"${dead}"']' 'pid_exe['"${dead}"']' 'pid_job['"${dead}"']' 'pid_node['"${dead}"']'
}
terminate_owned_children() {
  local pid; for pid in "${registered_pids[@]:-}"; do signal_owned_pid "${pid}" TERM; done
  for pid in "${registered_pids[@]:-}"; do
    for _ in {1..30}; do kill -0 "${pid}" 2>/dev/null || break; sleep 1; done
    kill -0 "${pid}" 2>/dev/null && signal_owned_pid "${pid}" KILL
    wait "${pid}" 2>/dev/null || true; unregister_pid "${pid}"
  done
}
cleanup() { local rc=$?; trap '' INT TERM HUP; trap - EXIT; terminate_owned_children; exit "${rc}"; }
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

mkdir -m 0700 "${run_root}" "${run_root}/logs" "${run_root}/runtime-source" "${run_root}/outputs" "${run_root}/topology"
"${python_bin}" -I -S -B - "${source_archive}" "${run_root}/runtime-source" <<'PY'
import sys,tarfile
from pathlib import Path,PurePosixPath
archive,out=map(Path,sys.argv[1:]); seen=set()
with tarfile.open(archive,"r:*") as handle:
    for member in handle.getmembers():
        path=PurePosixPath(member.name)
        normalized=path.as_posix().lstrip("./")
        assert normalized and not path.is_absolute() and ".." not in path.parts and normalized not in seen
        assert not member.issym() and not member.islnk() and not member.isdev() and not member.isfifo()
        assert member.isfile() or member.isdir()
        seen.add(normalized)
    handle.extractall(out,filter="data")
PY
readonly method_root="${run_root}/runtime-source/methods/bernini_action_editing"
readonly selection="${method_root}/assets/pair_v5_t2v_calibration_reserve4_selection_v1.json"
readonly registry="${method_root}/assets/pair_v5_t2v_calibration_first8_authoring_v1.json"
find "${run_root}/runtime-source" -type d -exec chmod 0700 {} +
find "${run_root}/runtime-source" -type f -exec chmod 0400 {} +
[[ "$(sha256_file "${method_root}/infer_native_identity_generation_canary.py")" == "${expected_native_generator_sha}" ]] || fail "native generator bytes differ"
[[ "$(sha256_file "${method_root}/infer_pair_v5_t2v_calibration_bank.py")" == "${expected_bank_generator_sha}" ]] || fail "PAIR bank generator bytes differ"
[[ "$(sha256_file "${selection}")" == "${expected_selection_sha}" ]] || fail "reserve selection bytes differ"
[[ "$(sha256_file "${registry}")" == "${expected_registry_sha}" ]] || fail "authoring registry bytes differ"

"${python_bin}" -I -B - "${overlay}" "${selection}" "${registry}" "${run_root}/authoring-preflight.json" <<'PY'
import hashlib,importlib.util,json,os,sys
from pathlib import Path
overlay,selection,registry,out=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location("display_anchor_overlay",overlay); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
row=module.load_authoring(selection,registry)
rows=[module.candidate_from_authoring(row,b) for b in module.BRANCH_ORDER]
unsigned={"schema_version":"bernini-pair-v5-reserve-display-anchor-authoring-preflight-v1","complete":True,"iid":module.IID,"branch_order":list(module.BRANCH_ORDER),"seed":module.SEED,"source_sha256":module.SOURCE_VIDEO_SHA256,"prompt_sha256_by_branch":{r["semantic_branch"]:r["full_t2v_caption_utf8_sha256"] for r in rows},"display_only":True,"stage_b_condition":False,"old40_bank_audit_claimed":False}
unsigned["receipt_digest"]=hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("ascii")).hexdigest()
fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,"w",encoding="ascii") as handle: handle.write(json.dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=True)+"\n")
PY
assert_all_parents_running

launch_critical=0; pending_signal=""; launched_pid=""
signal_exit() { pending_signal="$1"; (( launch_critical == 1 )) || exit 130; }
trap 'signal_exit INT' INT
trap 'signal_exit TERM' TERM
trap 'signal_exit HUP' HUP

launch_child() {
  local job="$1" node="$2" log="$3" pid; shift 3
  launch_critical=1; [[ -z "${pending_signal}" ]] || { launch_critical=0; exit 130; }
  srun --jobid="${job}" --nodelist="${node}" --nodes=1 --exclusive --exact --kill-on-bad-exit=1 "$@" >"${log}" 2>&1 &
  pid=$!; register_pid "${pid}" "${job}" "${node}" || { launch_critical=0; wait "${pid}" 2>/dev/null || true; fail "cannot identity-bind child srun"; }
  launched_pid="${pid}"; launch_critical=0; [[ -z "${pending_signal}" ]] || exit 130
}

wait_world4_pair() {
  local branch="$1" p0="$2" p1="$3" started="${SECONDS}" done0=0 done1=0 rc0=0 rc1=0 failure_ticks=0
  while (( done0 == 0 || done1 == 0 )); do
    if (( done0 == 0 )) && ! kill -0 "${p0}" 2>/dev/null; then wait "${p0}" || rc0=$?; unregister_pid "${p0}"; done0=1; fi
    if (( done1 == 0 )) && ! kill -0 "${p1}" 2>/dev/null; then wait "${p1}" || rc1=$?; unregister_pid "${p1}"; done1=1; fi
    if (( SECONDS - started >= timeout_seconds )); then rc0=124; rc1=124; fi
    if (( rc0 != 0 || rc1 != 0 )); then
      (( done0 == 0 )) && signal_owned_pid "${p0}" TERM; (( done1 == 0 )) && signal_owned_pid "${p1}" TERM; (( failure_ticks += 1 ))
      if (( failure_ticks >= 30 )); then (( done0 == 0 )) && signal_owned_pid "${p0}" KILL; (( done1 == 0 )) && signal_owned_pid "${p1}" KILL; fi
    fi
    (( done0 == 0 || done1 == 0 )) && sleep 1
  done
  (( rc0 == 0 && rc1 == 0 )) || { tail -n 120 "${run_root}/logs/${branch}-node0.log" >&2 || true; tail -n 120 "${run_root}/logs/${branch}-node1.log" >&2 || true; fail "WORLD4 branch ${branch} failed"; }
}

verify_memory() {
  local evidence="$1" job="$2" rank="$3" output="$4" step row
  step="$(sed -n 's/.* step_id=\([0-9][0-9]*\) .*/\1/p' "${evidence}")"; [[ "${step}" =~ ^[0-9]+$ ]] || fail "memory evidence step differs"
  for _ in {1..30}; do
    row="$(sacct -j "${job}.${step}" -n -P -o JobIDRaw,State,ExitCode,MaxRSS | awk -F'|' -v id="${job}.${step}" '$1==id{print;exit}')"
    [[ "${row}" == *"|COMPLETED|0:0|"* && -n "${row##*|}" ]] && break; sleep 1
  done
  "${python_bin}" -I -B - "${evidence}" "${row}" "${job}" "${rank}" "${step}" "${memory_peak_limit_bytes}" "${output}" <<'PY'
import json,os,re,sys
from pathlib import Path
evidence,row,job,rank,step,limit,out=sys.argv[1:]; fields={}
for token in Path(evidence).read_text().strip().split():
    if "=" in token: key,value=token.split("=",1); fields[key]=value
assert set(fields)=={"schema","job_id","step_id","node_rank","status","sampled_peak_bytes","samples","interval_seconds","limit_bytes","source"}
assert fields["schema"]=="bernini-pair-v5-display-anchor-child-cgroup-memory-v1" and fields["job_id"]==job and fields["step_id"]==step
assert fields["node_rank"]==rank and fields["status"]=="available" and fields["limit_bytes"]==limit and fields["interval_seconds"]=="0.1"
assert fields["source"].startswith("/sys/fs/cgroup/") and fields["source"].endswith("/memory.current")
sampled=int(fields["sampled_peak_bytes"]); assert int(fields["samples"])>0 and sampled<int(limit)
parts=row.split("|"); assert parts[:3]==[f"{job}.{step}","COMPLETED","0:0"]
match=re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?)",parts[3]); assert match
scale={"":1,"K":1024,"M":1024**2,"G":1024**3,"T":1024**4,"P":1024**5}[match.group(2)]; sacct=int(float(match.group(1))*scale); assert sacct<int(limit)
value={"schema_version":"bernini-pair-v5-display-anchor-memory-crosscheck-v1","job_id":job,"step_id":step,"sampled_memory_current_peak_bytes":sampled,"sacct_max_rss_raw":parts[3],"sacct_max_rss_bytes":sacct,"limit_bytes":int(limit),"both_below_limit":True}
fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,"w",encoding="ascii") as handle: handle.write(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n")
PY
}

run_branch() {
  local branch="$1" port="$2" output topology pid0 pid1
  output="${run_root}/outputs/${branch}"; topology="${run_root}/topology/${branch}"
  [[ ! -e "${output}" && ! -L "${output}" && ! -e "${topology}" && ! -L "${topology}" ]] || fail "branch paths are not fresh: ${branch}"
  mkdir -m 0700 "${topology}"
  assert_idle_twice "pre-${branch}"; assert_port_free "${port}"; sleep 1; assert_port_free "${port}"
  local -a overlay_args=(
    "${overlay}" run --method-root "${method_root}" --selection "${selection}" --registry "${registry}"
    --branch "${branch}" --source-video "${source_video}" --output-dir "${output}" --topology-dir "${topology}"
    --expected-job0 "${work_job0}" --expected-node0 "${work_node0}" --expected-job1 "${work_job1}" --expected-node1 "${work_node1}"
    --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" --checkpoint "${checkpoint}"
    --checkpoint-content-manifest "${checkpoint_manifest}" --method-source-revision "${source_revision}"
    --method-source-archive-sha256 "${source_archive_sha}"
  )
  launch_node() {
    local job="$1" node="$2" rank="$3" log="$4" peak="$5"
    launch_child "${job}" "${node}" "${log}" --ntasks=1 --cpus-per-task=16 --mem=56G --gres=gpu:mi210:2 \
      env BERNINI_PAIR_V5_ANCHOR_CONTROLLER_SHA256="${controller_sha}" \
        BERNINI_PAIR_V5_ANCHOR_WORK_JOB0="${work_job0}" BERNINI_PAIR_V5_ANCHOR_WORK_JOB1="${work_job1}" \
        BERNINI_HELDOUT_RANK_CACHE_TOKEN="pair5-anchor-${branch}-${source_revision:0:10}" BERNINI_HELDOUT_PYTHON_BIN="${python_bin}" \
        PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 \
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH="${method_root}" \
        "${controller_source}" __anchor_exec "${peak}" "${rank}" "${python_bin}" \
        "${python_bin}" -B -m torch.distributed.run --nnodes=2 --nproc_per_node=2 --node_rank="${rank}" \
          --master_addr="${work_node0}" --master_port="${port}" --no_python "${rank_cache_exec}" "${overlay_args[@]}"
  }
  launch_node "${work_job0}" "${work_node0}" 0 "${run_root}/logs/${branch}-node0.log" "${run_root}/logs/${branch}-node0-memory.txt"; pid0="${launched_pid}"
  launch_node "${work_job1}" "${work_node1}" 1 "${run_root}/logs/${branch}-node1.log" "${run_root}/logs/${branch}-node1-memory.txt"; pid1="${launched_pid}"
  wait_world4_pair "${branch}" "${pid0}" "${pid1}"
  assert_all_parents_running
  verify_memory "${run_root}/logs/${branch}-node0-memory.txt" "${work_job0}" 0 "${run_root}/logs/${branch}-node0-memory.json"
  verify_memory "${run_root}/logs/${branch}-node1-memory.txt" "${work_job1}" 1 "${run_root}/logs/${branch}-node1-memory.json"
  [[ -f "${output}/t2v.mp4" && ! -L "${output}/t2v.mp4" && -f "${output}/display-anchor-receipt.json" && ! -L "${output}/display-anchor-receipt.json" ]] || fail "display anchor publication absent: ${branch}"
}

run_branch action "$(( first_port + 0 ))"
if [[ "${profile}" == family4 ]]; then
  run_branch noop "$(( first_port + 1 ))"
  run_branch incomplete "$(( first_port + 2 ))"
  run_branch reverse "$(( first_port + 3 ))"
fi
"${python_bin}" -I -B "${overlay}" verify-set --root "${run_root}" --profile "${profile}"

"${python_bin}" -I -B - "${run_root}" "${profile}" "${controller_sha}" "${overlay_sha}" "${source_archive_sha}" "${source_revision}" "${work_job0}" "${work_node0}" "${work_job1}" "${work_node1}" "${retained_job}" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
root=Path(sys.argv[1]); profile,controller_sha,overlay_sha,archive_sha,revision,job0,node0,job1,node1,retained=sys.argv[2:]
set_path=root/"display-anchor-set-receipt.json"; raw=set_path.read_bytes(); value=json.loads(raw)
unsigned=dict(value); declared=unsigned.pop("receipt_digest")
canonical=json.dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
assert hashlib.sha256(canonical).hexdigest()==declared and value["complete"] is True and value["requested_profile"]==profile
branches=["action"] if profile=="action-only" else ["action","noop","incomplete","reverse"]
memory=[]
for branch in branches:
    for rank,job in ((0,job0),(1,job1)):
        path=root/"logs"/f"{branch}-node{rank}-memory.json"; row=json.loads(path.read_text())
        assert row["both_below_limit"] is True and row["job_id"]==job
        memory.append({"branch":branch,"node_rank":rank,"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
receipt={"schema_version":"bernini-pair-v5-reserve-display-anchor-controller-receipt-v1","complete":True,"iid":"00435ad621c44fac","profile":profile,"generation_order":branches,"action_canary_first":True,"set_receipt":{"path":str(set_path),"sha256":hashlib.sha256(raw).hexdigest(),"receipt_digest":declared},"method":{"source_archive_sha256":archive_sha,"source_revision":revision,"overlay_sha256":overlay_sha,"controller_sha256":controller_sha},"physical_holders":{"work_jobs":[job0,job1],"node_order":[node0,node1],"retained_only_job":retained,"world_size":4,"local_ranks_per_node":2,"gpus_per_node":2,"requested_memory_per_node":"56G","strict_peak_limit_bytes":55834574848,"all_memory_gates_passed":True},"memory_evidence":memory,"use_contract":{"display_only":True,"stage_b_condition":False,"passed_to_stage_b_runtime":False,"old40_bank_audit_claimed":False,"action_success_claimed":False,"scientific_claim_authorized":False},"parent_allocations_released":False,"parent_allocations_signaled":False}
receipt["receipt_digest"]=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")).hexdigest()
out=root/"controller-receipt.json"; fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,"w",encoding="ascii") as handle: handle.write(json.dumps(receipt,sort_keys=True,separators=(",",":"),ensure_ascii=True)+"\n")
PY

assert_idle_twice final
assert_all_parents_running
printf 'COMPLETE_PAIR_V5_DISPLAY_ANCHOR iid=%s profile=%s seed=%s display_only=true stage_b_condition=false old40_bank_audit=false action_success=false scientific_claim=false work_holders=%s,%s retained_only=%s parents_retained=135407,135411,135412\n' \
  "00435ad621c44fac" "${profile}" "2026080821" "${work_job0}" "${work_job1}" "${retained_job}" >"${run_root}/controller.COMPLETE"
chmod 0400 "${run_root}/controller.COMPLETE"
trap - EXIT INT TERM HUP
echo "COMPLETE_PAIR_V5_DISPLAY_ANCHOR output=${run_root}"
