#!/usr/bin/env bash
# Read-only Stage-A schedule x block decoded causal localization on two
# protected holders.  One consolidated WORLD4 child invocation loads the model
# once, executes the fixed C0 smoke gate, and (only on engineering C0 PASS)
# continues the preregistered full grid without human or visual selection.
# Parent allocations are immutable; only identity-bound direct child srun PIDs
# created by this controller may be signaled.
set -Eeuo pipefail
umask 077

fail() { echo "[schedule-block-stage-a-two-holder] ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

readonly controller_requested="${BASH_SOURCE[0]}"
[[ "${controller_requested}" == /* && -f "${controller_requested}" && ! -L "${controller_requested}" ]] || \
  fail "controller must be invoked as an absolute plain file"
readonly controller_source="$(readlink -f -- "${controller_requested}")"
readonly holder_user=guangyi.chen
readonly work_job0="${BERNINI_SBCL_STAGE_A_WORK_JOB0:?set first work holder}"
readonly work_job1="${BERNINI_SBCL_STAGE_A_WORK_JOB1:?set second work holder}"

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

# Frozen only after the independently-owned runtime/core, deterministic release,
# AUH dynamic gates, and independent hostile audit all passed.
readonly expected_runtime_sha=31a32125c11a36104b233a6ab271026add82478cdcb3144331fef6ad1e5f3b05
readonly expected_core_sha=385cc2321da888f75d5aff5017175b85acf06174969aaa39210b802cc14695c5
readonly expected_release_archive_sha=1d9d4eb37aedffc13d0e1aaf0663561ae989aeb15a42e52437ea4a0dd9287a9f
readonly expected_release_manifest_sha=c5226bd3b77630352938451ada7c68b8a5dbf51d95b6c1182b2f47c2aaee237a
readonly expected_release_revision=7ced6fc99f00c728af477e07cdd58a9e239e973c
readonly expected_policy_sha=1be281b0419a23254d51556a41eda0d014ecd75cb044caaf5e3ceb96f7c54998
readonly expected_grid_sha=992dc6e59399216f7556c8a0db7faa7e8bb98d81e6b6a37d8340284232267de8
readonly expected_prompt_authority_sha=204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c
readonly expected_orbit_authority_sha=dc2d83322357196cec84418ddf4318d9fc7d1eb41269cb216739bae7c6169651
readonly expected_rank_cache_sha=f416f8f687a61788d4f687e500dae04af677072ec205fb560016cb5249b4f9c5
readonly expected_checkpoint_manifest_sha=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly checkpoint_tree_sha=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly iid=00435ad621c44fac
readonly style_id=2
readonly seed=2026081401
readonly source_sha=b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1
readonly source_spec_sha=62468b24d4a57ec03d42ce8c006a707cbcf56588ef62d10632089eb5ad457920
readonly source_spec_digest=de2f92f314da538f8af322a8f1db23cbdf1feab4b28d2da66248d25309a25595
readonly source_parquet_sha=77d89b3ec2e563f624bab62451b49b616ffa7f7890db6105c4458617aac0d106
readonly source_receipt_sha=6ed77cf7d98391c2074e5938ab50d0688d457bddfd688f9a5825d455447a20bb
readonly source_receipt_digest=12ede44ebab03215e19574967a9afec3c634f246f2cfd2634a48ce0e3dea8738
readonly expected_wrong_owner_dataset_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_counterfactual_identity_orbit_v5_20260808_c099c6f/datasets/portrait2_rv2v4_exact81_v1
readonly wrong_owner_spec_sha=72c0f104b123a1b7ad69f32697a0b7f7e8c2fdf766c951f3c0bed7518f0f564f
readonly wrong_owner_spec_digest=25522068a18893afbc21f54a7851dbf641bc10ea7229653cdfe0c772be1f934e
readonly wrong_owner_reference_encoding_digest=181e93b1620cafce7de3806b334b6bfdd8e24aa633119cbd6506f3761175a269
readonly wrong_owner_parquet_sha=845727b8e9c461b9cf1f8bb98c0e27519599ffffcd6619bd8895250a2e075baf
readonly wrong_owner_receipt_sha=c088eb0128c3c807941f60eb3e763d0e71f4c8dbb190c60b9c0dad6caeca0230
readonly wrong_owner_receipt_digest=9000dd9dace16501587196ac8459b620529301508ee6c98662f266b3b29b8982
readonly wrong_owner_variant=variant_a
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
  expected_sha="${BERNINI_SBCL_STAGE_A_CONTROLLER_SHA256:?set controller SHA}"
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
assert all(torch.cuda.memory_allocated(index) == 0 for index in range(2))
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
  printf 'schema=bernini-schedule-block-stage-a-child-cgroup-memory-v1 job_id=%s step_id=%s node_rank=%s status=%s sampled_peak_bytes=%s samples=%s interval_seconds=0.1 limit_bytes=%s source=%s\n' \
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

if [[ "${1:-}" == __localize_exec ]]; then
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
usage: auh_infer_schedule_block_causal_localization_two_holder_v2.sh run
Required environment: BERNINI_SBCL_STAGE_A_RUN_ROOT, _SOURCE_ARCHIVE,
_SOURCE_ARCHIVE_SHA256, _SOURCE_MANIFEST, _SOURCE_MANIFEST_SHA256,
_SOURCE_REVISION, _CONTROLLER_SHA256, _RANK_CACHE_EXEC, _SOURCE_DATASET_ROOT,
_WRONG_OWNER_DATASET_ROOT, _SOURCE_VIDEO, _PYTHON_BIN, _WORK_JOB0, _WORK_JOB1,
BERNINI_OFFICIAL_ROOT, BERNINI_VEOMNI_ROOT, BERNINI_CHECKPOINT,
BERNINI_CHECKPOINT_CONTENT_MANIFEST. Optional _PROFILE is
smoke-then-full-fixed (formal default) or smoke-only (debug only).
EOF
  exit 2
}
[[ "${1:-}" == run && $# == 1 ]] || usage

readonly run_root="${BERNINI_SBCL_STAGE_A_RUN_ROOT:?}"
readonly source_archive="${BERNINI_SBCL_STAGE_A_SOURCE_ARCHIVE:?}"
readonly source_archive_sha="${BERNINI_SBCL_STAGE_A_SOURCE_ARCHIVE_SHA256:?}"
readonly source_manifest="${BERNINI_SBCL_STAGE_A_SOURCE_MANIFEST:?}"
readonly source_manifest_sha="${BERNINI_SBCL_STAGE_A_SOURCE_MANIFEST_SHA256:?}"
readonly source_revision="${BERNINI_SBCL_STAGE_A_SOURCE_REVISION:?}"
readonly controller_sha="${BERNINI_SBCL_STAGE_A_CONTROLLER_SHA256:?}"
readonly rank_cache_exec="${BERNINI_SBCL_STAGE_A_RANK_CACHE_EXEC:?}"
readonly source_dataset_root="${BERNINI_SBCL_STAGE_A_SOURCE_DATASET_ROOT:?}"
readonly wrong_owner_dataset_root="${BERNINI_SBCL_STAGE_A_WRONG_OWNER_DATASET_ROOT:?}"
readonly source_video="${BERNINI_SBCL_STAGE_A_SOURCE_VIDEO:?}"
readonly python_bin="${BERNINI_SBCL_STAGE_A_PYTHON_BIN:?}"
readonly profile="${BERNINI_SBCL_STAGE_A_PROFILE:-smoke-then-full-fixed}"
readonly bernini_root="${BERNINI_OFFICIAL_ROOT:?}"
readonly veomni_root="${BERNINI_VEOMNI_ROOT:?}"
readonly checkpoint="${BERNINI_CHECKPOINT:?}"
readonly checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?}"
readonly timeout_seconds="${BERNINI_SBCL_STAGE_A_TIMEOUT_SECONDS:-43200}"
readonly master_port="${BERNINI_SBCL_STAGE_A_MASTER_PORT:-30031}"

for name in run_root source_archive source_manifest rank_cache_exec source_dataset_root wrong_owner_dataset_root source_video python_bin bernini_root veomni_root checkpoint checkpoint_manifest; do
  [[ "${!name}" == /* && "${!name}" =~ ^/[A-Za-z0-9._/-]+$ ]] || fail "${name} must be an absolute shell-safe path"
done
for digest in source_archive_sha source_manifest_sha controller_sha; do [[ "${!digest}" =~ ^[0-9a-f]{64}$ ]] || fail "${digest} differs"; done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${source_archive_sha}" == "${expected_release_archive_sha}" && "${source_manifest_sha}" == "${expected_release_manifest_sha}" && "${source_revision}" == "${expected_release_revision}" ]] || fail "Stage-A release authority differs"
[[ "${wrong_owner_dataset_root}" == "${expected_wrong_owner_dataset_root}" ]] || fail "wrong-owner dataset root differs"
[[ "${profile}" == smoke-then-full-fixed || "${profile}" == smoke-only ]] || fail "profile differs"
[[ "${timeout_seconds}" =~ ^[1-9][0-9]*$ && "${master_port}" =~ ^[0-9]+$ && "${master_port}" -ge 1024 && "${master_port}" -le 65535 ]] || fail "timeout/port differs"
for path in "${source_archive}" "${source_manifest}" "${source_video}" "${checkpoint_manifest}" "${controller_source}"; do
  [[ -f "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "sealed file differs: ${path}"
done
for path in "${rank_cache_exec}" "${python_bin}"; do [[ -x "${path}" && ! -L "${path}" ]] || fail "executable differs: ${path}"; done
for path in "${source_dataset_root}" "${wrong_owner_dataset_root}" "${bernini_root}" "${veomni_root}" "${checkpoint}"; do
  [[ -d "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "runtime root differs: ${path}"
done
[[ "${run_root}" != / && "$(realpath -m -- "${run_root}")" == "${run_root}" && ! -e "${run_root}" && ! -L "${run_root}" ]] || fail "run root must be fresh canonical"
[[ "$(sha256_file "${controller_source}")" == "${controller_sha}" ]] || fail "controller bytes differ"
[[ "$(sha256_file "${rank_cache_exec}")" == "${expected_rank_cache_sha}" ]] || fail "rank cache bytes differ"
[[ "$(sha256_file "${source_archive}")" == "${source_archive_sha}" && "$(sha256_file "${source_manifest}")" == "${source_manifest_sha}" ]] || fail "release bytes differ"
[[ "$(sha256_file "${source_video}")" == "${source_sha}" ]] || fail "source video bytes differ"
[[ "$(sha256_file "${checkpoint_manifest}")" == "${expected_checkpoint_manifest_sha}" ]] || fail "checkpoint manifest differs"

"${python_bin}" -I -S -B - \
  "${source_dataset_root}" "${wrong_owner_dataset_root}" "${iid}" \
  "${source_spec_sha}" "${source_spec_digest}" "${source_parquet_sha}" \
  "${source_receipt_sha}" "${source_receipt_digest}" \
  "${wrong_owner_spec_sha}" "${wrong_owner_spec_digest}" \
  "${wrong_owner_reference_encoding_digest}" \
  "${wrong_owner_parquet_sha}" "${wrong_owner_receipt_sha}" \
  "${wrong_owner_receipt_digest}" <<'PY'
import hashlib,json,sys
from pathlib import Path
source,wrong=map(Path,sys.argv[1:3]); expected_iid=sys.argv[3]
(source_spec_sha,source_spec_digest,source_parquet_sha,source_receipt_sha,
 source_receipt_digest,wrong_spec_sha,wrong_spec_digest,
 wrong_reference_digest,wrong_parquet_sha,wrong_receipt_sha,
 wrong_receipt_digest)=sys.argv[4:]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def receipt(root, *, schema, spec_sha_key, spec_sha, spec_digest,
            parquet_sha, receipt_sha, digest, reference_digest=None):
    assert {path.name for path in root.iterdir()} == {"dataset.parquet", "receipt.json"}
    assert sha(root/"dataset.parquet") == parquet_sha
    raw=(root/"receipt.json").read_bytes(); assert sha(root/"receipt.json") == receipt_sha
    value=json.loads(raw)
    assert value["schema_version"] == schema and value["complete"] is True
    assert value["dataset"]["sha256"] == parquet_sha
    assert Path(value["dataset"]["path"]) == root/"dataset.parquet"
    assert value["dataset"]["iids"].count(expected_iid) == 1
    assert value["spec"][spec_sha_key] == spec_sha
    assert value["spec"]["digest"] == spec_digest
    if reference_digest is not None:
        assert value["spec"]["reference_encoding_contract_digest"] == reference_digest
    unsigned=dict(value); declared=unsigned.pop("receipt_digest")
    canonical=json.dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
    assert hashlib.sha256(canonical).hexdigest() == declared == digest
receipt(source,
    schema="bernini-source-self-role-repaint-dataset-receipt-v2",
    spec_sha_key="sha256",spec_sha=source_spec_sha,
    spec_digest=source_spec_digest,parquet_sha=source_parquet_sha,
    receipt_sha=source_receipt_sha,digest=source_receipt_digest)
receipt(wrong,
    schema="bernini-appearance-counterfactual-identity-orbit-dataset-receipt-v3",
    spec_sha_key="file_sha256",spec_sha=wrong_spec_sha,
    spec_digest=wrong_spec_digest,parquet_sha=wrong_parquet_sha,
    receipt_sha=wrong_receipt_sha,digest=wrong_receipt_digest,
    reference_digest=wrong_reference_digest)
PY

assert_parent_running() {
  local job="$1" node record
  node="$(holder_node "${job}")"; record="$(scontrol show job -o "${job}")"
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
  local port="$1" found
  found="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${work_node0}" "ss -H -ltn 'sport = :${port}'")"; [[ -z "${found}" ]] || fail "master port ${port} occupied"
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

mkdir -m 0700 "${run_root}" "${run_root}/logs" "${run_root}/runtime-source" "${run_root}/topology"
"${python_bin}" -I -S -B - "${source_archive}" "${source_manifest}" "${run_root}/runtime-source" "${source_revision}" <<'PY'
import hashlib,json,stat,sys,tarfile
from pathlib import Path
archive,manifest_path,out=map(Path,sys.argv[1:4]); revision=sys.argv[4]
raw=manifest_path.read_bytes(); value=json.loads(raw.decode("ascii")); unsigned=dict(value); declared=unsigned.pop("manifest_digest")
canonical=json.dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
assert hashlib.sha256(canonical).hexdigest()==declared
assert value["schema_version"]=="bernini-schedule-block-causal-localization-release-v2"
assert value["release_generation"]=="r2" and value["content_closure_sha1"]==revision
assert value["file_count"]==len(value["files"]) and value["exact_member_closure"] is True and value["git_commit_claimed"] is False
assert value["formal_profile"]=="smoke-then-full-fixed" and value["single_model_load_required"] is True
assert value["engineering_c0_decoded_output_count"]==6 and value["preregistered_full_grid_decoded_output_count"]==112
assert value["engineering_c0_plan_digest"]=="d11dbd0cfca34f26ea5f72bdd2f5ed8b21c512387410b659ade9f217d866c923"
assert value["preregistered_full_grid_plan_digest"]=="6fd3299a1af84968bebe12cd6f1b2a84feb0fb28a07d29619fbcfac66bf4d2e8"
assert value["formal_total_decoded_output_count"]==118
assert value["formal_full_continuation_automatic_after_c0_pass"] is True
assert value["c0_failure_forbids_full_grid"] is True and value["c0_gate_engineering_only"] is True
assert value["engineering_c0_has_no_visual_or_scientific_selection"] is True
assert value["prompt_calibration_action_reverse_direction_passed"] is True and value["prompt_calibration_noop_incomplete_semantics_passed"] is False
assert value["negative_cluster_semantically_validated"] is False and value["negative_cluster_scientific_veto_authorized"] is False
assert value["full_grid_cells_retained_without_deletion"] is True
assert value["diagnostic_only"] is True and value["optimizer_authorized"] is False
assert value["parameter_update_authorized"] is False and value["scientific_selection_authorized"] is False
expected=["methods/bernini_action_editing/"+row["path"] for row in value["files"]]
with tarfile.open(archive,"r:") as handle:
    members=handle.getmembers(); assert [member.name for member in members]==expected
    for member,row in zip(members,value["files"]):
        stream=handle.extractfile(member); assert member.isfile() and not member.issym() and not member.islnk() and stream is not None
        payload=stream.read(); assert member.uid==member.gid==member.mtime==0 and stat.S_IMODE(member.mode)==0o444
        assert len(payload)==row["size"]==member.size and row["mode"]=="0444" and hashlib.sha256(payload).hexdigest()==row["sha256"]
    handle.extractall(out,filter="data")
PY
readonly method_root="${run_root}/runtime-source/methods/bernini_action_editing"
readonly runtime_entry="${method_root}/infer_schedule_block_causal_localization_v1.py"
readonly core_entry="${method_root}/schedule_block_target_row_prompt_swap_v1.py"
readonly policy_entry="${method_root}/schedule_block_causal_policy_v1.py"
readonly prompt_authority="${method_root}/assets/pair_v5_t2v_calibration_first8_authoring_v1.json"
readonly orbit_authority="${method_root}/assets/appearance_identity_orbit_portrait2_review_v1.json"
find "${run_root}/runtime-source" -type d -exec chmod 0700 {} +
find "${run_root}/runtime-source" -type f -exec chmod 0400 {} +
[[ "$(sha256_file "${runtime_entry}")" == "${expected_runtime_sha}" ]] || fail "frozen Stage-A runtime differs"
[[ "$(sha256_file "${core_entry}")" == "${expected_core_sha}" ]] || fail "frozen Stage-A core differs"
[[ "$(sha256_file "${policy_entry}")" == "${expected_policy_sha}" ]] || fail "frozen Stage-A policy differs"
[[ "$(sha256_file "${prompt_authority}")" == "${expected_prompt_authority_sha}" ]] || fail "prompt authority differs"
[[ "$(sha256_file "${orbit_authority}")" == "${expected_orbit_authority_sha}" ]] || fail "orbit authority differs"

# A single create-only visibility specification is checked locally and from
# both work nodes with bounded retry.  This makes NFS propagation explicit.
readonly visibility_spec="${run_root}/nfs-visibility.spec"
readonly visibility_checker="${run_root}/nfs-visibility-check.py"
"${python_bin}" -I -S -B - "${visibility_spec}" "${visibility_checker}" \
  "${source_manifest}" "${method_root}" \
  "${controller_source}" "${controller_sha}" \
  "${source_video}" "${source_sha}" "${source_dataset_root}/dataset.parquet" "${source_parquet_sha}" \
  "${source_dataset_root}/receipt.json" "${source_receipt_sha}" "${wrong_owner_dataset_root}/dataset.parquet" "${wrong_owner_parquet_sha}" \
  "${wrong_owner_dataset_root}/receipt.json" "${wrong_owner_receipt_sha}" "${checkpoint_manifest}" "${expected_checkpoint_manifest_sha}" \
  "${rank_cache_exec}" "${expected_rank_cache_sha}" <<'PY'
import json,os,sys
from pathlib import Path
spec,checker,manifest_path,method_root=map(Path,sys.argv[1:5])
manifest=json.loads(manifest_path.read_text(encoding="ascii"))
release_pairs=[(str(method_root/row["path"]),row["sha256"]) for row in manifest["files"]]
external_pairs=list(zip(sys.argv[5::2],sys.argv[6::2]))
pairs=release_pairs+external_pairs
assert pairs and len({path for path,_ in pairs})==len(pairs)
spec_raw="".join(f"{digest} {path}\n" for path,digest in pairs).encode("ascii")
checker_raw=b'''import hashlib,sys\nfrom pathlib import Path\nroot=Path(sys.argv[1]); spec=Path(sys.argv[2])\nassert root.is_absolute() and root.resolve(strict=True)==root and root.is_dir() and not root.is_symlink()\nrows=[]\nfor line in spec.read_text(encoding="ascii").splitlines():\n    digest,name=line.split(" ",1); path=Path(name)\n    assert len(digest)==64 and path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and not path.is_symlink()\n    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest; rows.append(line)\nprint(hashlib.sha256(("\\n".join(rows)+"\\n").encode("ascii")).hexdigest())\n'''
for path,raw in ((spec,spec_raw),(checker,checker_raw)):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
    with os.fdopen(fd,"wb") as handle: handle.write(raw); handle.flush(); os.fsync(handle.fileno())
PY
readonly visibility_digest="$("${python_bin}" -I -S -B "${visibility_checker}" "${run_root}" "${visibility_spec}")"
[[ "${visibility_digest}" =~ ^[0-9a-f]{64}$ ]] || fail "local NFS visibility digest differs"
visibility_attempt0=0; visibility_attempt1=0
assert_remote_visibility() {
  local node="$1" rank="$2" attempt output
  for attempt in {1..30}; do
    output="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${node}" "${python_bin} -I -S -B ${visibility_checker} ${run_root} ${visibility_spec}" 2>/dev/null || true)"
    if [[ "${output}" == "${visibility_digest}" ]]; then printf -v "visibility_attempt${rank}" '%s' "${attempt}"; return 0; fi
    sleep 1
  done
  fail "shared filesystem not visible on ${node}"
}
assert_remote_visibility "${work_node0}" 0
assert_remote_visibility "${work_node1}" 1
"${python_bin}" -I -S -B - "${run_root}/nfs-visibility-receipt.json" "${visibility_spec}" "${visibility_digest}" "${work_node0}" "${visibility_attempt0}" "${work_node1}" "${visibility_attempt1}" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
out,spec=map(Path,sys.argv[1:3]); digest,node0,attempt0,node1,attempt1=sys.argv[3:]
unsigned={"schema_version":"bernini-schedule-block-stage-a-nfs-visibility-v1","complete":True,"spec_path":str(spec),"spec_sha256":hashlib.sha256(spec.read_bytes()).hexdigest(),"visibility_digest":digest,"nodes":[{"node":node0,"attempt":int(attempt0)},{"node":node1,"attempt":int(attempt1)}],"bounded_retry_attempts":30}
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
  local label="$1" p0="$2" p1="$3" started="${SECONDS}" done0=0 done1=0 rc0=0 rc1=0 failure_ticks=0
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
  (( rc0 == 0 && rc1 == 0 )) || { tail -n 160 "${run_root}/logs/stage-a-node0.log" >&2 || true; tail -n 160 "${run_root}/logs/stage-a-node1.log" >&2 || true; fail "WORLD4 Stage-A invocation failed"; }
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
assert fields["schema"]=="bernini-schedule-block-stage-a-child-cgroup-memory-v1" and fields["job_id"]==job and fields["step_id"]==step
assert fields["node_rank"]==rank and fields["status"]=="available" and fields["limit_bytes"]==limit and fields["interval_seconds"]=="0.1"
assert fields["source"].startswith("/sys/fs/cgroup/") and fields["source"].endswith("/memory.current")
sampled=int(fields["sampled_peak_bytes"]); assert int(fields["samples"])>0 and sampled<int(limit)
parts=row.split("|"); assert parts[:3]==[f"{job}.{step}","COMPLETED","0:0"]
match=re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?)",parts[3]); assert match
scale={"":1,"K":1024,"M":1024**2,"G":1024**3,"T":1024**4,"P":1024**5}[match.group(2)]; sacct=int(float(match.group(1))*scale); assert sacct<int(limit)
value={"schema_version":"bernini-schedule-block-stage-a-memory-crosscheck-v1","job_id":job,"step_id":step,"sampled_memory_current_peak_bytes":sampled,"sacct_max_rss_raw":parts[3],"sacct_max_rss_bytes":sacct,"limit_bytes":int(limit),"both_below_limit":True}
fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,"w",encoding="ascii") as handle: handle.write(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n")
PY
}

run_localization() {
  local output topology pid0 pid1
  output="${run_root}/localization"; topology="${run_root}/topology/stage-a"
  [[ ! -e "${output}" && ! -L "${output}" && ! -e "${topology}" && ! -L "${topology}" ]] || fail "Stage-A output paths are not fresh"
  mkdir -m 0700 "${topology}"
  assert_idle_twice pre-stage-a; assert_port_free "${master_port}"; sleep 1; assert_port_free "${master_port}"

  # Frozen runtime CLI: one distributed invocation and one model load per rank.
  local -a runtime_args=(
    "${runtime_entry}" run --profile "${profile}" --output-dir "${output}" --topology-dir "${topology}"
    --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" --checkpoint "${checkpoint}"
    --checkpoint-content-manifest "${checkpoint_manifest}" --source-dataset-root "${source_dataset_root}"
    --wrong-owner-dataset-root "${wrong_owner_dataset_root}" --source-video "${source_video}"
    --method-source-revision "${source_revision}" --method-source-archive-sha256 "${source_archive_sha}"
    --seed "${seed}"
  )
  launch_node() {
    local job="$1" node="$2" rank="$3" log="$4" peak="$5"
    launch_child "${job}" "${node}" "${log}" --ntasks=1 --cpus-per-task=16 --mem=56G --gres=gpu:mi210:2 \
      env BERNINI_SBCL_STAGE_A_CONTROLLER_SHA256="${controller_sha}" \
        BERNINI_SBCL_STAGE_A_WORK_JOB0="${work_job0}" BERNINI_SBCL_STAGE_A_WORK_JOB1="${work_job1}" \
        BERNINI_HELDOUT_RANK_CACHE_TOKEN="stage-a-${profile}-${source_revision:0:10}" BERNINI_HELDOUT_PYTHON_BIN="${python_bin}" \
        PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 \
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH="${method_root}" \
        "${controller_source}" __localize_exec "${peak}" "${rank}" "${python_bin}" \
        "${python_bin}" -B -m torch.distributed.run --nnodes=2 --nproc_per_node=2 --node_rank="${rank}" \
          --master_addr="${work_node0}" --master_port="${master_port}" --no_python "${rank_cache_exec}" "${runtime_args[@]}"
  }
  launch_node "${work_job0}" "${work_node0}" 0 "${run_root}/logs/stage-a-node0.log" "${run_root}/logs/stage-a-node0-memory.txt"; pid0="${launched_pid}"
  launch_node "${work_job1}" "${work_node1}" 1 "${run_root}/logs/stage-a-node1.log" "${run_root}/logs/stage-a-node1-memory.txt"; pid1="${launched_pid}"
  wait_world4_pair stage-a "${pid0}" "${pid1}"
  assert_all_parents_running
  verify_memory "${run_root}/logs/stage-a-node0-memory.txt" "${work_job0}" 0 "${run_root}/logs/stage-a-node0-memory.json"
  verify_memory "${run_root}/logs/stage-a-node1-memory.txt" "${work_job1}" 1 "${run_root}/logs/stage-a-node1-memory.json"
  [[ -f "${output}/receipt.json" && ! -L "${output}/receipt.json" ]] || fail "Stage-A receipt absent"
}

run_localization

# Re-run the frozen model-free verifier on the published bundle.  This is not a
# second inference invocation and cannot load the renderer; it reopens every
# authority, receipt, plan, artifact, and all 6+112 decoded videos.
"${python_bin}" -I -B "${runtime_entry}" verify \
  --output-dir "${run_root}/localization" --profile "${profile}"
assert_idle_twice final
assert_all_parents_running

# Bind the frozen runtime verifier result to the physical holder, memory, NFS,
# release, and controller authorities.  Runtime validation above is the deep
# receipt validator; these checks add the controller-specific exact bindings.
"${python_bin}" -I -B - \
  "${run_root}" "${profile}" "${controller_sha}" "${source_archive_sha}" \
  "${source_manifest_sha}" "${source_revision}" "${expected_runtime_sha}" \
  "${expected_core_sha}" "${expected_policy_sha}" "${source_dataset_root}" \
  "${wrong_owner_dataset_root}" "${source_video}" "${bernini_root}" \
  "${veomni_root}" "${checkpoint}" "${checkpoint_manifest}" \
  "${work_job0}" "${work_node0}" "${work_job1}" "${work_node1}" \
  "${retained_job}" "${method_root}" "${wrong_owner_variant}" <<'PY'
import hashlib,json,os,re,stat,sys
from pathlib import Path

(root_value,profile,controller_sha,archive_sha,manifest_sha,revision,
 runtime_sha,core_sha,policy_sha,source_root,orbit_root,source_video,
 bernini_root,veomni_root,checkpoint,checkpoint_manifest,job0,node0,
 job1,node1,retained,method_root,wrong_variant)=sys.argv[1:]
root=Path(root_value); method_root=Path(method_root)
sha256_re=re.compile(r"[0-9a-f]{64}\Z")

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")

def object_sha(value): return hashlib.sha256(canonical(value)).hexdigest()

def stable_bytes(path):
    path=Path(path)
    assert path.is_absolute() and not path.is_symlink() and path.resolve(strict=True)==path
    before=path.stat(); assert stat.S_ISREG(before.st_mode)
    with path.open("rb") as handle:
        opened=os.fstat(handle.fileno()); raw=handle.read(); after=os.fstat(handle.fileno())
    named=path.stat()
    identity=lambda item:(item.st_dev,item.st_ino,item.st_size,item.st_mtime_ns,stat.S_IMODE(item.st_mode))
    assert identity(before)==identity(opened)==identity(after)==identity(named) and len(raw)==before.st_size
    return raw

def strict_json(path):
    raw=stable_bytes(path)
    def reject_constant(value): raise ValueError(f"non-finite JSON constant: {value}")
    def reject_duplicates(pairs):
        value={}
        for key,item in pairs:
            assert type(key) is str and key not in value
            value[key]=item
        return value
    value=json.loads(raw.decode("ascii"),parse_constant=reject_constant,object_pairs_hook=reject_duplicates)
    assert type(value) is dict and raw==canonical(value)+b"\n"
    return value,raw

def embedded_digest(value,key="digest"):
    unsigned=dict(value); declared=unsigned.pop(key)
    assert sha256_re.fullmatch(str(declared)) and object_sha(unsigned)==declared
    return declared

assert root.is_absolute() and not root.is_symlink() and root.resolve(strict=True)==root
formal=profile=="smoke-then-full-fixed"
assert formal or profile=="smoke-only"
runtime_path=root/"localization"/"receipt.json"
runtime,raw=strict_json(runtime_path)
top_keys={"schema_version","complete","profile","model_load_count","one_process_one_model_load","same_load_c0_then_full","seed","datasets","source_video","prompt_authority","text_runtime","orbit_review_authority","model","distributed","first_forward_consensus","owner_pairs","input_invariants","method_source","execution","c0","full","outputs","artifacts","model_integrity","no_update","processor_patch","calibration","terminal_authority_audit","interpretation","receipt_digest"}
assert set(runtime)==top_keys
declared=embedded_digest(runtime,"receipt_digest")
assert runtime["schema_version"]=="bernini-schedule-block-causal-localization-runtime-v1"
assert runtime["complete"] is True and runtime["profile"]==profile and runtime["seed"]==2026081401
assert runtime["model_load_count"]==1 and runtime["one_process_one_model_load"] is True
assert runtime["same_load_c0_then_full"] is formal

method=runtime["method_source"]
assert method=={"revision":revision,"archive_sha256":archive_sha}
datasets=runtime["datasets"]; source= datasets["source_self_cross_authority"]; orbit=datasets["orbit_model_inputs"]
assert source["root"]==source_root and source["iid"]=="00435ad621c44fac"
assert source["parquet_sha256"]=="77d89b3ec2e563f624bab62451b49b616ffa7f7890db6105c4458617aac0d106"
assert source["receipt_sha256"]=="6ed77cf7d98391c2074e5938ab50d0688d457bddfd688f9a5825d455447a20bb"
assert source["receipt_digest"]=="12ede44ebab03215e19574967a9afec3c634f246f2cfd2634a48ce0e3dea8738"
assert source["materialization_spec_sha256"]=="62468b24d4a57ec03d42ce8c006a707cbcf56588ef62d10632089eb5ad457920"
assert source["materialization_spec_digest"]=="de2f92f314da538f8af322a8f1db23cbdf1feab4b28d2da66248d25309a25595"
assert source["projected_columns"]==["iid","source_video_sha256","row_digest"]
assert source["posterior_blob_columns_read"]==[] and source["latents_consumed"] is False
assert orbit["root"]==orbit_root and orbit["iid"]=="00435ad621c44fac"
assert orbit["parquet_sha256"]=="845727b8e9c461b9cf1f8bb98c0e27519599ffffcd6619bd8895250a2e075baf"
assert orbit["receipt_sha256"]=="c088eb0128c3c807941f60eb3e763d0e71f4c8dbb190c60b9c0dad6caeca0230"
assert orbit["receipt_digest"]=="9000dd9dace16501587196ac8459b620529301508ee6c98662f266b3b29b8982"
assert orbit["materialization_spec_sha256"]=="72c0f104b123a1b7ad69f32697a0b7f7e8c2fdf766c951f3c0bed7518f0f564f"
assert orbit["materialization_spec_digest"]=="25522068a18893afbc21f54a7851dbf641bc10ea7229653cdfe0c772be1f934e"
assert orbit["reference_encoding_contract_digest"]=="181e93b1620cafce7de3806b334b6bfdd8e24aa633119cbd6506f3761175a269"
assert orbit["all_target_and_owner_latents_from_orbit_row"] is True
assert orbit["orbit_tensor_broadcast"]["world4_consensus"] is True
assert runtime["source_video"]=={"path":source_video,"sha256":"b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1","cross_authority_only":True,"model_condition_consumed":False}
prompt=runtime["prompt_authority"]
assert prompt["path"]==str(method_root/"assets/pair_v5_t2v_calibration_first8_authoring_v1.json")
assert prompt["sha256"]=="204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c" and prompt["iid"]=="00435ad621c44fac"
review=runtime["orbit_review_authority"]
assert review["path"]==str(method_root/"assets/appearance_identity_orbit_portrait2_review_v1.json")
assert review["sha256"]=="dc2d83322357196cec84418ddf4318d9fc7d1eb41269cb216739bae7c6169651"
assert review["iid"]=="00435ad621c44fac" and review["wrong_owner_variant"]==wrong_variant=="variant_a"
assert review["all_qualification_gates_true"] is True

model=runtime["model"]
assert model["bernini_root"]==bernini_root and model["veomni_root"]==veomni_root
assert model["checkpoint"]==checkpoint and model["checkpoint_content_manifest"]==checkpoint_manifest
assert model["bernini_commit"]=="2d2b4591ac053ec25c6371b01a5a6746679e5793"
assert model["veomni_commit"]=="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
assert model["checkpoint_tree_sha256"]=="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
assert model["checkpoint_manifest_sha256"]=="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
assert model["checkpoint_verified_file_count"]==23 and model["transformer_count"]==1 and model["transformer_block_count"]==30

distributed=runtime["distributed"]
assert distributed["world_size"]==4 and distributed["local_world_size"]==2
assert distributed["nodes"]==2 and distributed["ranks_per_node"]==2 and distributed["ulysses_sp_size"]==4
assert distributed["sp4_crosses_nodes"] is True and distributed["topology_admitted_collectively_before_output"] is True
placement=[(row["rank"],row["local_rank"],row["hostname"]) for row in distributed["rank_hostname_local_rank"]]
assert placement==[(0,0,node0),(1,1,node0),(2,0,node1),(3,1,node1)]
topology=distributed["topology_admission"]
assert topology["path"]==str(root/"topology/stage-a") and topology["world_size"]==4
assert topology["empty_on_every_rank"] is True and topology["collective_before_output_reservation"] is True
first=runtime["first_forward_consensus"]
assert first["passed"] is True and first["before_any_real_forward"] is True
assert first["actual_model_text_input_tensor_hashes"] is True and first["first_output_world4_consensus"] is True

expected_execution={
 "distributed_invocation_count":1,"model_load_count":1,"vae_load_count":1,
 "c0_model_forward_count":10,"c0_capture_forward_count":2,
 "c0_internal_parity_forward_count":2,"c0_decoded_output_count":6,
 "full_model_forward_count":136 if formal else 0,
 "full_capture_forward_count":24 if formal else 0,
 "full_decoded_output_count":112 if formal else 0,
 "total_model_forward_count":146 if formal else 10,
 "scheduler_instance_count":0,"scheduler_step_count":0,"optimizer_instance_count":0,
}
assert runtime["execution"]==expected_execution
c0=runtime["c0"]
assert c0["schema_version"]=="bernini-schedule-block-causal-localization-c0-engineering-gate-v1"
assert c0["engineering_pass"] is True and c0["scientific_pass_claimed"] is False and c0["visual_selection_performed"] is False
assert c0["decoded_output_count"]==6 and c0["internal_noop_parity_decoded_output_count"]==0 and c0["media_complete"] is True
assert len(c0["processor_audits"])==10 and len(c0["cache_audits"])==2 and len(c0["noop_parity"])==2
if formal:
    full=runtime["full"]
    assert full["started_after_c0_pass"] is True and full["same_model_load"] is True
    assert full["fixed_plan_no_adaptation"] is True and full["decoded_output_count"]==112 and full["completed"] is True
    assert full["plan_digest"]=="6fd3299a1af84968bebe12cd6f1b2a84feb0fb28a07d29619fbcfac66bf4d2e8"
    assert len(full["processor_audits"])==136 and len(full["cache_audits"])==24
else:
    assert runtime["full"] is None

outputs=runtime["outputs"]; artifacts=runtime["artifacts"]
assert len(outputs)==(118 if formal else 6) and len(artifacts)==(120 if formal else 7)
assert [row["phase"] for row in outputs]==(["c0"]*6+(["full"]*112 if formal else []))
names=[row["name"] for row in outputs]
assert c0["decoded_output_names"]==names[:6] and len(names)==len(set(names))
assert set(artifacts)==set(names)|({"c0-plan.json","full-plan.json"} if formal else {"c0-plan.json"})
for row in outputs:
    assert row["actual_object_binding_used_for_forward"] is True and row["global_prompt_branch"]=="noop"
    assert row["result_world4_consensus"] is True and row["vae_frozen_eval"] is True
    assert row["frames"]==81 and float(row["fps"])==25.0 and row["hw"]==[592,400]
    assert row["decode_input_latent_shape"]==[1,16,21,74,50]
    assert row["decode_input_dtype"]=="torch.float32" and row["decode_input_device_type"]=="cuda"
    assert row["decode_input_contiguous"] is True and row["decode_input_finite"] is True
    assert artifacts[row["name"]]==row["sha256"] and sha256_re.fullmatch(row["sha256"])

pairs=runtime["owner_pairs"]
assert len(pairs["c0"])==1 and pairs["c0"][0]["schedule_index"]==29
if formal:
    assert [row["schedule_index"] for row in pairs["full"]]==[16,29,35,38]
    assert pairs["cross_schedule_closure"]["c0_s29_and_full_s29_binding_raw_equal"] is True
else:
    assert pairs["full"] is None and pairs["cross_schedule_closure"] is None
invariants=runtime["input_invariants"]; pre=invariants["pre_c0_snapshot"]["digest"]
assert invariants["schema_version"]=="stage-a-actual-model-input-invariants-v1"
assert invariants["actual_objects_rehashed_at_each_phase"] is True and invariants["all_actual_input_bytes_unchanged"] is True
assert invariants["full_phase_executed"] is formal
expected_stage={"pre_c0":pre,"post_c0":pre,"post_full":pre if formal else None,"terminal":pre}
assert invariants["stage_snapshot_digests"]==expected_stage
assert invariants["stage_world4_rank_digests"]=={key:([value]*4 if value is not None else None) for key,value in expected_stage.items()}
integrity=runtime["model_integrity"]
assert integrity["certificate_schema"]=="torch-module-parameters-buffers-raw-sha256-v1"
assert integrity["pre_sha256"]==integrity["post_c0_sha256"]==integrity["post_sha256"]
assert integrity["bytes_unchanged"] is True and integrity["all_parameters_frozen"] is True and integrity["all_parameter_gradients_absent"] is True
assert runtime["no_update"]=={"gradient_enabled":False,"optimizer_present":False,"scheduler_present":False,"scheduler_steps":0,"parameter_gradients_present":False,"parameter_updates":0,"torch_inference_mode_all_forwards":True}
patch=runtime["processor_patch"]
assert patch["optimizer_present"] is False and patch["parameter_update_authorized"] is False and patch["restored"] is True
assert patch["installation_and_restore_transactional"] is True and patch["owner_binding_required_by_every_context"] is True
calibration=runtime["calibration"]; branches=calibration["branch_authority"]
assert set(branches)=={"noop","forward","reverse","incomplete","camera_only","appearance_only"}
assert all(row["run_in_fixed_grid"] is True and row["scientific_veto_authorized"] is False for row in branches.values())
assert branches["forward"]["calibration"]==branches["reverse"]["calibration"]=="direction_pass"
assert branches["noop"]["semantic_negative_authorized"] is False
assert branches["incomplete"]["calibration"]=="failed_delayed_action_reaches_hands_on_hips"
assert calibration["family4_noise_consumed_by_stage_a"] is False and calibration["family4_and_stage_a_seed_distinct"] is True
assert runtime["interpretation"]=={"automatic_scientific_claim":False,"c0_engineering_only":True,"camera_and_appearance_role":"exploratory_confounded_nuisance_controls","incomplete_role":"calibration_failed_exploratory_not_scientific_veto","method_success_claimed":False,"negative_cluster_semantically_validated":False,"noop_role":"numerical_baseline_not_semantic_negative","reverse_role":"directional_negative_candidate_only","scientific_selection_performed":False,"scientific_veto_authorized":False,"visual_adaptive_cell_selection":False}
terminal=runtime["terminal_authority_audit"]
assert terminal["all_live_authorities_reopened_and_stable"] is True
assert terminal["source_video_sha256"]=="b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1"

nfs_path=root/"nfs-visibility-receipt.json"; nfs,nfs_raw=strict_json(nfs_path)
embedded_digest(nfs,"receipt_digest")
assert set(nfs)=={"schema_version","complete","spec_path","spec_sha256","visibility_digest","nodes","bounded_retry_attempts","receipt_digest"}
assert nfs["schema_version"]=="bernini-schedule-block-stage-a-nfs-visibility-v1" and nfs["complete"] is True
assert nfs["spec_path"]==str(root/"nfs-visibility.spec") and nfs["spec_sha256"]==hashlib.sha256(stable_bytes(root/"nfs-visibility.spec")).hexdigest()
assert sha256_re.fullmatch(nfs["visibility_digest"]) and nfs["bounded_retry_attempts"]==30
assert [row["node"] for row in nfs["nodes"]]==[node0,node1]
assert all(type(row["attempt"]) is int and 1<=row["attempt"]<=30 for row in nfs["nodes"])
memory=[]
memory_keys={"schema_version","job_id","step_id","sampled_memory_current_peak_bytes","sacct_max_rss_raw","sacct_max_rss_bytes","limit_bytes","both_below_limit"}
for rank,job in ((0,job0),(1,job1)):
    path=root/"logs"/f"stage-a-node{rank}-memory.json"; row,memory_raw=strict_json(path)
    assert set(row)==memory_keys and row["schema_version"]=="bernini-schedule-block-stage-a-memory-crosscheck-v1"
    assert row["both_below_limit"] is True and row["job_id"]==job and re.fullmatch(r"[0-9]+",row["step_id"])
    assert row["limit_bytes"]==55834574848 and row["sampled_memory_current_peak_bytes"]<row["limit_bytes"] and row["sacct_max_rss_bytes"]<row["limit_bytes"]
    memory.append({"node_rank":rank,"path":str(path),"sha256":hashlib.sha256(memory_raw).hexdigest(),"job_id":job,"step_id":row["step_id"]})

receipt={
 "schema_version":"bernini-schedule-block-stage-a-controller-receipt-v1",
 "complete":True,"profile":profile,"single_distributed_invocation":True,
 "single_model_load_required":True,"runtime_one_process_one_model_load":True,
 "runtime_verifier_completed":True,
 "runtime_receipt":{"path":str(runtime_path),"sha256":hashlib.sha256(raw).hexdigest(),"receipt_digest":declared,"schema_version":runtime["schema_version"]},
 "runtime_verified_output_closure":{"decoded_output_count":len(outputs),"engineering_c0_decoded_output_count":6,"preregistered_full_decoded_output_count":112 if formal else 0,"artifact_count":len(artifacts),"output_records_digest":object_sha(outputs),"artifact_map_digest":object_sha(artifacts),"formal_full_continuation_automatic_after_c0_pass":formal,"same_model_load_c0_then_full":formal,"fixed_plan_no_adaptation":formal},
 "nfs_visibility_receipt":{"path":str(nfs_path),"sha256":hashlib.sha256(nfs_raw).hexdigest(),"receipt_digest":nfs["receipt_digest"]},
 "method":{"source_archive_sha256":archive_sha,"source_manifest_sha256":manifest_sha,"source_revision":revision,"controller_sha256":controller_sha,"runtime_sha256":runtime_sha,"core_sha256":core_sha,"policy_sha256":policy_sha},
 "physical_holders":{"work_jobs":[job0,job1],"node_order":[node0,node1],"retained_only_job":retained,"world_size":4,"local_ranks_per_node":2,"gpus_per_node":2,"cpus_per_node":16,"requested_memory_per_node":"56G","strict_peak_limit_bytes":55834574848,"all_memory_gates_passed":True},
 "memory_evidence":memory,
 "failure_semantics":{"native_shared_step_internal_rccl_failure_recoverable":False,"native_shared_step_internal_rccl_recovery_claimed":False,"controller_signaling_scope":"identity-bound-direct-child-srun-only"},
 "authority":{"diagnostic_only":True,"optimizer_authorized":False,"parameter_update_authorized":False,"automatic_scientific_cell_selection":False,"route_update_authorized":False,"prompt_calibration_action_reverse_direction_passed":True,"prompt_calibration_noop_incomplete_semantics_passed":False,"negative_cluster_semantically_validated":False,"full_grid_cells_retained_without_deletion":True,"negative_cluster_scientific_veto_authorized":False,"method_success_claimed":False,"scientific_claim_authorized":False},
 "parent_allocations_released":False,"parent_allocations_signaled":False,
}
receipt["receipt_digest"]=object_sha(receipt)
out=root/"controller-receipt.json"; assert not out.exists() and not out.is_symlink()
fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,"wb") as handle:
    handle.write(canonical(receipt)+b"\n"); handle.flush(); os.fsync(handle.fileno())
PY

printf 'COMPLETE_STAGE_A_SCHEDULE_BLOCK_CAUSAL_LOCALIZATION iid=%s profile=%s diagnostic_only=true optimizer=false parameter_update=false auto_scientific_selection=false negative_cluster_validated=false negative_cluster_veto=false method_success=false scientific_claim=false work_holders=%s,%s retained_only=%s parents_retained=135407,135411,135412\n' \
  "${iid}" "${profile}" "${work_job0}" "${work_job1}" "${retained_job}" >"${run_root}/controller.COMPLETE"
chmod 0400 "${run_root}/controller.COMPLETE"
trap - EXIT INT TERM HUP
echo "COMPLETE_STAGE_A_SCHEDULE_BLOCK_CAUSAL_LOCALIZATION output=${run_root}"
