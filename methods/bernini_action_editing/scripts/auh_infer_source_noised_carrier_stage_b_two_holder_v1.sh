#!/usr/bin/env bash
# Four serialized, fit-source/fresh-noise Stage-B engineering decodes on two
# protected holder allocations.  Parent allocations are immutable; this file
# owns and may signal only the exact numbered child srun PIDs it creates.
set -Eeuo pipefail
umask 077

fail() { echo "[stage-b-infer-two-holder] ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

readonly controller_requested="${BASH_SOURCE[0]}"
[[ "${controller_requested}" == /* && -f "${controller_requested}" && ! -L "${controller_requested}" ]] || \
  fail "controller must be invoked as an absolute plain file"
readonly controller_source="$(readlink -f -- "${controller_requested}")"
readonly holder_user=guangyi.chen
readonly work_job0=135407 work_node0=auh7-1b-gpu-260
readonly work_job1=135411 work_node1=auh7-1b-gpu-214
readonly retained_job=135412 retained_node=auh7-1b-gpu-293
readonly expected_runtime_sha=b21f7f85531fd7f41f1a9741894b26b564b25054da418d7989f2f7a588a6f84f
readonly expected_dependency_sha=62df125ac130697b03aaea167b17a02d7fcb9d766a72f0bef71037924114e59f
readonly expected_release_archive_sha=646d5a9c73364db689b2592d3b1a4a486c2e9e032031f09e5be3a003845f3698
readonly expected_release_manifest_sha=50f87120b08e4a576acd3ff44efdda699848db9c2d9d13336f5006431f418639
readonly expected_release_revision=63eafa1b10f083eedf6bec316ad92fb3bedea17b
readonly expected_rank_cache_sha=f416f8f687a61788d4f687e500dae04af677072ec205fb560016cb5249b4f9c5
readonly expected_checkpoint_manifest_sha=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly checkpoint_tree_sha=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly iid=00435ad621c44fac style_id=2 seed=2026081401
readonly source_sha=b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1
readonly spec_sha=62468b24d4a57ec03d42ce8c006a707cbcf56588ef62d10632089eb5ad457920
readonly parquet_sha=77d89b3ec2e563f624bab62451b49b616ffa7f7890db6105c4458617aac0d106
readonly dataset_receipt_sha=6ed77cf7d98391c2074e5938ab50d0688d457bddfd688f9a5825d455447a20bb
readonly dataset_receipt_digest=12ede44ebab03215e19574967a9afec3c634f246f2cfd2634a48ce0e3dea8738
readonly adapter_sha=f85de3518ec88ac86a33fcb574c328ae3bca581ea3a1f86a648ef051b14ec16c
readonly training_receipt_sha=7c61d23d00e442a7fada318ade279db1131db57da88d068233325f004cd1dca9
readonly memory_peak_limit_bytes=55834574848
readonly instruction="Restore the original video appearance from the clean source references while following the ordered donor's temporal evolution and camera path."

holder_node() {
  case "$1" in
    135407) printf '%s\n' "${work_node0}" ;;
    135411) printf '%s\n' "${work_node1}" ;;
    135412) printf '%s\n' "${retained_node}" ;;
    *) return 2 ;;
  esac
}
is_work_holder() { [[ "$1" == "${work_job0}" || "$1" == "${work_job1}" ]]; }

assert_rocm_idle() {
  local snapshot="$1" expected="$2" label="$3" uses mems busy
  uses="$(awk '/GPU use \(%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  mems="$(awk '/GPU Memory Allocated \(VRAM%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  busy="$(awk '/GPU use \(%\)/ || /GPU Memory Allocated \(VRAM%\)/ {v=$NF; gsub(/[^0-9]/,"",v); if (v+0 != 0) print}' <<<"${snapshot}")"
  if [[ "${label}" == child-visible ]]; then
    [[ "${uses}" -ge "${expected}" && "${uses}" == "${mems}" ]] || fail "${label}: GPU inventory differs"
  else
    [[ "${uses}" == "${expected}" && "${mems}" == "${expected}" ]] || fail "${label}: GPU inventory differs"
  fi
  [[ -z "${busy}" ]] || fail "${label}: GPU already active"
}

child_preflight() {
  local python_bin="$1" expected_sha snapshot
  expected_sha="${BERNINI_SNC_STAGE_B_INFER_CONTROLLER_SHA256:?set controller SHA}"
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
  cg="$(awk -F: '$1=="0" && $2=="" {print $3; exit}' /proc/self/cgroup)"
  read -r root mount < <(awk '$0 ~ / - cgroup2 / {print $4,$5; exit}' /proc/self/mountinfo)
  [[ "${cg}" == /* && "${root}" == /* && "${mount}" == /* ]] || return 70
  if [[ "${root}" == / ]]; then suffix="${cg}"; elif [[ "${cg}" == "${root}" ]]; then suffix="";
  elif [[ "${cg}" == "${root}/"* ]]; then suffix="/${cg#"${root}/"}"; else return 70; fi
  candidate="${mount%/}${suffix}/memory.current"
  [[ "${candidate}" == /sys/fs/cgroup/* && -f "${candidate}" && ! -L "${candidate}" && -r "${candidate}" ]] || return 70
  value="$(tr -d '[:space:]' <"${candidate}" 2>/dev/null || true)"; [[ "${value}" =~ ^[0-9]+$ ]] || return 70
  printf '%s\n' "${candidate}"
}

write_memory_evidence() {
  printf 'schema=bernini-stage-b-infer-child-cgroup-memory-v1 job_id=%s step_id=%s node_rank=%s status=%s sampled_peak_bytes=%s samples=%s interval_seconds=0.1 limit_bytes=%s source=%s\n' \
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

if [[ "${1:-}" == __infer_exec ]]; then
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
usage: auh_infer_source_noised_carrier_stage_b_two_holder_v1.sh run
Required environment: BERNINI_SNC_STAGE_B_INFER_RUN_ROOT, _SOURCE_ARCHIVE,
_SOURCE_ARCHIVE_SHA256, _SOURCE_MANIFEST, _SOURCE_MANIFEST_SHA256,
_SOURCE_REVISION, _CONTROLLER_SHA256, _RANK_CACHE_EXEC, _DATASET_ROOT,
_TRAINING_OUTPUT, _SOURCE_VIDEO, _PYTHON_BIN, BERNINI_OFFICIAL_ROOT,
BERNINI_VEOMNI_ROOT, BERNINI_CHECKPOINT, BERNINI_CHECKPOINT_CONTENT_MANIFEST.
EOF
  exit 2
}
[[ "${1:-}" == run && $# == 1 ]] || usage

readonly run_root="${BERNINI_SNC_STAGE_B_INFER_RUN_ROOT:?}"
readonly source_archive="${BERNINI_SNC_STAGE_B_INFER_SOURCE_ARCHIVE:?}"
readonly source_archive_sha="${BERNINI_SNC_STAGE_B_INFER_SOURCE_ARCHIVE_SHA256:?}"
readonly source_manifest="${BERNINI_SNC_STAGE_B_INFER_SOURCE_MANIFEST:?}"
readonly source_manifest_sha="${BERNINI_SNC_STAGE_B_INFER_SOURCE_MANIFEST_SHA256:?}"
readonly source_revision="${BERNINI_SNC_STAGE_B_INFER_SOURCE_REVISION:?}"
readonly controller_sha="${BERNINI_SNC_STAGE_B_INFER_CONTROLLER_SHA256:?}"
readonly rank_cache_exec="${BERNINI_SNC_STAGE_B_INFER_RANK_CACHE_EXEC:?}"
readonly dataset_root="${BERNINI_SNC_STAGE_B_INFER_DATASET_ROOT:?}"
readonly training_output="${BERNINI_SNC_STAGE_B_INFER_TRAINING_OUTPUT:?}"
readonly source_video="${BERNINI_SNC_STAGE_B_INFER_SOURCE_VIDEO:?}"
readonly python_bin="${BERNINI_SNC_STAGE_B_INFER_PYTHON_BIN:?}"
readonly bernini_root="${BERNINI_OFFICIAL_ROOT:?}"
readonly veomni_root="${BERNINI_VEOMNI_ROOT:?}"
readonly checkpoint="${BERNINI_CHECKPOINT:?}"
readonly checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?}"
readonly timeout_seconds="${BERNINI_SNC_STAGE_B_INFER_TIMEOUT_SECONDS:-21600}"
readonly first_port="${BERNINI_SNC_STAGE_B_INFER_FIRST_PORT:-29851}"

for name in run_root source_archive source_manifest rank_cache_exec dataset_root training_output source_video python_bin bernini_root veomni_root checkpoint checkpoint_manifest; do
  [[ "${!name}" == /* ]] || fail "${name} must be absolute"
done
for digest in source_archive_sha source_manifest_sha controller_sha; do [[ "${!digest}" =~ ^[0-9a-f]{64}$ ]] || fail "${digest} differs"; done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${source_archive_sha}" == "${expected_release_archive_sha}" && "${source_manifest_sha}" == "${expected_release_manifest_sha}" && "${source_revision}" == "${expected_release_revision}" ]] || fail "inference release authority differs"
[[ "${timeout_seconds}" =~ ^[1-9][0-9]*$ && "${first_port}" =~ ^[0-9]+$ ]] || fail "timeout/port differs"
for path in "${source_archive}" "${source_manifest}" "${source_video}" "${checkpoint_manifest}" "${controller_source}"; do
  [[ -f "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "sealed file differs: ${path}"
done
for path in "${rank_cache_exec}" "${python_bin}"; do [[ -x "${path}" && ! -L "${path}" ]] || fail "executable differs: ${path}"; done
for path in "${dataset_root}" "${training_output}" "${bernini_root}" "${veomni_root}" "${checkpoint}"; do
  [[ -d "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "runtime root differs: ${path}"
done
[[ "${run_root}" != / && "$(realpath -m -- "${run_root}")" == "${run_root}" && ! -e "${run_root}" && ! -L "${run_root}" ]] || fail "run root must be fresh canonical"
[[ "$(sha256_file "${controller_source}")" == "${controller_sha}" ]] || fail "controller bytes differ"
[[ "$(sha256_file "${rank_cache_exec}")" == "${expected_rank_cache_sha}" ]] || fail "rank cache bytes differ"
[[ "$(sha256_file "${source_archive}")" == "${source_archive_sha}" && "$(sha256_file "${source_manifest}")" == "${source_manifest_sha}" ]] || fail "release bytes differ"
[[ "$(sha256_file "${source_video}")" == "${source_sha}" ]] || fail "source video bytes differ"
[[ "$(sha256_file "${checkpoint_manifest}")" == "${expected_checkpoint_manifest_sha}" ]] || fail "checkpoint manifest differs"

assert_parent_running() {
  local job="$1" node record; node="$(holder_node "${job}")"; record="$(scontrol show job -o "${job}")"
  [[ "${record}" == *"JobId=${job} "* && "${record}" == *"JobState=RUNNING"* && "${record}" == *"UserId=${holder_user}"* ]] || fail "parent ${job} state/owner differs"
  [[ "${record}" == *"NodeList=${node}"* && "${record}" == *"NumCPUs=64"* && "${record}" == *"AllocTRES=cpu=64,mem=64G,"* && "${record}" == *"gres/gpu:mi210=8"* ]] || fail "parent ${job} topology differs"
}
assert_all_parents_running() { assert_parent_running "${work_job0}"; assert_parent_running "${work_job1}"; assert_parent_running "${retained_job}"; }
numbered_steps() { squeue -s -j "$1" -h -o '%i' | awk '/[.][0-9]+$/ {print}'; }
assert_remote_idle_once() {
  local job="$1" node="$2" steps processes hidden snapshot
  assert_parent_running "${job}"; steps="$(numbered_steps "${job}")"; [[ -z "${steps}" ]] || fail "foreign/existing child on ${job}: ${steps}"
  processes="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${node}" "ps -u ${holder_user} -ww -o pid=,ppid=,comm=,args=")"
  hidden="$(awk -v job="${job}" '{line=$0; c=$3; if(index(line,"/var/spool/slurmd/job" job "/slurm_script"))next; if(c=="sleep"&&line~/sleep infinity[[:space:]]*$/)next; if((c=="bash"||c=="sh")&&index(line,"holding allocation across nodes:")&&index(line,"sleep infinity"))next; if((c=="bash"||c=="sh")&&index(line,"ps -u guangyi.chen -ww -o"))next; if(c=="systemd"||c=="(sd-pam)"||c=="podman"||c=="dbus-daemon"||c=="sshd"||c=="ps")next; print}' <<<"${processes}")"
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

mkdir -m 0700 "${run_root}" "${run_root}/logs" "${run_root}/runtime-source"
"${python_bin}" -I -S -B - "${source_archive}" "${source_manifest}" "${run_root}/runtime-source" "${source_revision}" <<'PY'
import hashlib, json, stat, sys, tarfile
from pathlib import Path
archive, manifest_path, out = map(Path, sys.argv[1:4]); revision=sys.argv[4]
raw=manifest_path.read_bytes(); m=json.loads(raw.decode("ascii")); unsigned=dict(m); declared=unsigned.pop("manifest_digest")
canon=json.dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
assert hashlib.sha256(canon).hexdigest()==declared
assert m["schema_version"]=="bernini-source-noised-carrier-stage-b-inference-release-v1"
assert m["release_generation"]=="r1" and m["file_count"]==14 and m["content_closure_sha1"]==revision
assert m["exact_member_closure"] is True and m["git_commit_claimed"] is False
expected_rel=["source_self_role_repaint.py","source_self_runtime.py","train_source_self_role_repaint.py","train_lora.py","assets/source_self_role_repaint_canary_spec_v2.json","tools/materialize_source_self_role_repaint.py","tools/materialize_ramp_motion_analogy_vae.py","tools/materialize_vae.py","tools/build_renderer_dataset.py","inference_sigma_strata.py","source_noised_ladder_v1.py","train_source_noised_carrier_strata_v1.py","infer_lora.py","infer_source_noised_carrier_stage_b_v1.py"]
expected_sha=["bf212ac4effcd5b3975eefc61e01c71cba366969ec92cf2ff186765ddec43f2e","62df125ac130697b03aaea167b17a02d7fcb9d766a72f0bef71037924114e59f","357ba5310a297c042e1c1bd10bef35bb69e483e18ff15b5ba4cc2bd65a07c80d","630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5","62468b24d4a57ec03d42ce8c006a707cbcf56588ef62d10632089eb5ad457920","8065cafc34c15d7e8e6fc8e3abb13551b2cbe20c925ab8415267be5b3993cc80","ca9b4620ad7dc6cd03e70b180f68d83aad05c21cef574fe6467bdaa1202bb93a","a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0","afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5","e3782a22130c09a48dc3ea27fa219af6caca445e1fce2c8f3bca7cde6058afd3","eb8653a5e98d0744c9fd7066f3aefc4c5e0dfcd8f70320e86a2e669a376fef98","39c3fad7e8d710eedd453e75b1acf7fb35f30c0ccba4dee71d336efec5274704","babd6d63287723ccd14b2bbe43bd4550c30b4feaa794d17c66f5a5ddefe979fe","b21f7f85531fd7f41f1a9741894b26b564b25054da418d7989f2f7a588a6f84f"]
rows=m["files"]; assert [r["path"] for r in rows]==expected_rel and [r["sha256"] for r in rows]==expected_sha
expected=["methods/bernini_action_editing/"+x for x in expected_rel]
with tarfile.open(archive,"r:") as t:
    members=t.getmembers(); assert [x.name for x in members]==expected
    for member,row in zip(members,rows):
        stream=t.extractfile(member); assert member.isfile() and not member.issym() and not member.islnk() and stream is not None
        payload=stream.read(); assert member.uid==member.gid==member.mtime==0 and stat.S_IMODE(member.mode)==0o444
        assert len(payload)==row["size"]==member.size and hashlib.sha256(payload).hexdigest()==row["sha256"]
    t.extractall(out,filter="data")
PY
readonly method_root="${run_root}/runtime-source/methods/bernini_action_editing"
readonly runtime_entry="${method_root}/infer_source_noised_carrier_stage_b_v1.py"
find "${run_root}/runtime-source" -type f -exec chmod 0400 {} +
[[ "$(sha256_file "${runtime_entry}")" == "${expected_runtime_sha}" ]] || fail "frozen inference runtime differs"
[[ "$(sha256_file "${method_root}/source_self_runtime.py")" == "${expected_dependency_sha}" ]] || fail "audited shared runtime differs"

"${python_bin}" -I -B - "${dataset_root}" "${training_output}" "${seed}" <<'PY'
import hashlib,json,sys
from pathlib import Path
dataset,training=map(Path,sys.argv[1:3]); seed=int(sys.argv[3])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert {p.name for p in dataset.iterdir()}=={"dataset.parquet","receipt.json"}
assert sha(dataset/"dataset.parquet")=="77d89b3ec2e563f624bab62451b49b616ffa7f7890db6105c4458617aac0d106"
raw=(dataset/"receipt.json").read_bytes(); assert hashlib.sha256(raw).hexdigest()=="6ed77cf7d98391c2074e5938ab50d0688d457bddfd688f9a5825d455447a20bb"
r=json.loads(raw); digest=r.pop("receipt_digest"); canonical=json.dumps(r,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
assert hashlib.sha256(canonical).hexdigest()==digest=="12ede44ebab03215e19574967a9afec3c634f246f2cfd2634a48ce0e3dea8738"
assert {p.name for p in training.iterdir()}=={"adapter.safetensors","optimizer.pt","history.json","receipt.json"}
pins={"adapter.safetensors":"f85de3518ec88ac86a33fcb574c328ae3bca581ea3a1f86a648ef051b14ec16c","optimizer.pt":"cf9113c74dce8f3f0c1e4a8c2a93e021f46e4346926c3d6009260b29330c8161","history.json":"fdac2070d5ed64028c18eef7453528c5140d81e70305d432319d9c59fb41e92a","receipt.json":"7c61d23d00e442a7fada318ade279db1131db57da88d068233325f004cd1dca9"}
assert all(sha(training/name)==value for name,value in pins.items())
h=json.loads((training/"history.json").read_text()); seen={int(rec["noise_seed"]) for step in h["steps"] for rec in step["logical_records"]}
assert len(seen)==8 and seed not in seen
PY
assert_all_parents_running

launch_critical=0
pending_signal=""
signal_exit() { pending_signal="$1"; (( launch_critical == 1 )) || exit 130; }
trap 'signal_exit INT' INT
trap 'signal_exit TERM' TERM
trap 'signal_exit HUP' HUP

launch_child() {
  local job="$1" node="$2" log="$3" pid; shift 3
  launch_critical=1
  [[ -z "${pending_signal}" ]] || { launch_critical=0; exit 130; }
  srun --jobid="${job}" --nodelist="${node}" --nodes=1 --exclusive --exact --kill-on-bad-exit=1 \
    "$@" >"${log}" 2>&1 &
  pid=$!
  register_pid "${pid}" "${job}" "${node}" || { launch_critical=0; wait "${pid}" 2>/dev/null || true; fail "cannot identity-bind child srun"; }
  launched_pid="${pid}"; launch_critical=0
  [[ -z "${pending_signal}" ]] || exit 130
}

wait_world4_pair() {
  local branch="$1" p0="$2" p1="$3" started="${SECONDS}" done0=0 done1=0 rc0=0 rc1=0 failure_ticks=0
  while (( done0 == 0 || done1 == 0 )); do
    if (( done0 == 0 )) && ! kill -0 "${p0}" 2>/dev/null; then wait "${p0}" || rc0=$?; unregister_pid "${p0}"; done0=1; fi
    if (( done1 == 0 )) && ! kill -0 "${p1}" 2>/dev/null; then wait "${p1}" || rc1=$?; unregister_pid "${p1}"; done1=1; fi
    if (( SECONDS - started >= timeout_seconds )); then rc0=124; rc1=124; fi
    if (( rc0 != 0 || rc1 != 0 )); then
      (( done0 == 0 )) && signal_owned_pid "${p0}" TERM
      (( done1 == 0 )) && signal_owned_pid "${p1}" TERM
      (( failure_ticks += 1 ))
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
    row="$(sacct -j "${job}.${step}" -n -P -o JobIDRaw,State,ExitCode,MaxRSS | awk -F'|' -v id="${job}.${step}" '$1==id {print;exit}')"
    [[ "${row}" == *"|COMPLETED|0:0|"* && -n "${row##*|}" ]] && break; sleep 1
  done
  "${python_bin}" -I -B - "${evidence}" "${row}" "${job}" "${rank}" "${step}" "${memory_peak_limit_bytes}" "${output}" <<'PY'
import json,os,re,sys
from pathlib import Path
evidence,row,job,rank,step,limit,out=sys.argv[1:]; fields={}
for token in Path(evidence).read_text().strip().split():
    if "=" in token: k,v=token.split("=",1); fields[k]=v
assert set(fields)=={"schema","job_id","step_id","node_rank","status","sampled_peak_bytes","samples","interval_seconds","limit_bytes","source"}
assert fields["schema"]=="bernini-stage-b-infer-child-cgroup-memory-v1" and fields["job_id"]==job and fields["step_id"]==step
assert fields["node_rank"]==rank and fields["status"]=="available" and fields["limit_bytes"]==limit and fields["interval_seconds"]=="0.1"
assert fields["source"].startswith("/sys/fs/cgroup/") and fields["source"].endswith("/memory.current")
sampled=int(fields["sampled_peak_bytes"]); assert int(fields["samples"])>0 and sampled<int(limit)
parts=row.split("|"); assert parts[:3]==[f"{job}.{step}","COMPLETED","0:0"]
m=re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?)",parts[3]); assert m
scale={"":1,"K":1024,"M":1024**2,"G":1024**3,"T":1024**4,"P":1024**5}[m.group(2)]; sacct=int(float(m.group(1))*scale); assert sacct<int(limit)
p={"schema_version":"bernini-stage-b-infer-memory-crosscheck-v1","job_id":job,"step_id":step,"sampled_memory_current_peak_bytes":sampled,"sacct_max_rss_raw":parts[3],"sacct_max_rss_bytes":sacct,"limit_bytes":int(limit),"both_below_limit":True}
fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,"w",encoding="ascii") as f: f.write(json.dumps(p,sort_keys=True,separators=(",",":"))+"\n")
PY
}

run_branch() {
  local branch="$1" mode="$2" arm="$3" port="$4" output="${run_root}/${branch}" pid0 pid1
  [[ ! -e "${output}" && ! -L "${output}" ]] || fail "branch output is not fresh: ${branch}"
  assert_idle_twice "pre-${branch}"; assert_port_free "${port}"; sleep 1; assert_port_free "${port}"
  local -a runtime_args=(
    "${runtime_entry}" --bernini-root "${bernini_root}" --veomni-root "${veomni_root}"
    --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}"
    --expected-checkpoint-content-manifest-sha256 "${expected_checkpoint_manifest_sha}"
    --dataset-root "${dataset_root}" --expected-materialization-spec-sha256 "${spec_sha}"
    --expected-dataset-parquet-sha256 "${parquet_sha}" --expected-dataset-receipt-sha256 "${dataset_receipt_sha}"
    --expected-dataset-receipt-digest "${dataset_receipt_digest}" --iid "${iid}" --style-id "${style_id}"
    --source-video "${source_video}" --expected-source-sha256 "${source_sha}" --instruction "${instruction}"
    --arm "${arm}" --mode "${mode}" --seed "${seed}" --output "${output}"
    --expected-bernini-commit "${bernini_commit}" --expected-veomni-commit "${veomni_commit}"
    --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha}" --method-source-revision "${source_revision}"
    --method-source-archive-sha256 "${source_archive_sha}"
  )
  if [[ "${arm}" == trained ]]; then runtime_args+=(--adapter-checkpoint "${training_output}" --expected-adapter-sha256 "${adapter_sha}" --expected-training-receipt-sha256 "${training_receipt_sha}")
  elif [[ "${arm}" != frozen_base ]]; then fail "arm differs"; fi
  launch_node() {
    local job="$1" node="$2" rank="$3" log="$4" peak="$5"
    launch_child "${job}" "${node}" "${log}" --ntasks=1 --cpus-per-task=16 --mem=56G --gres=gpu:mi210:2 \
      env BERNINI_SNC_STAGE_B_INFER_CONTROLLER_SHA256="${controller_sha}" \
        BERNINI_HELDOUT_RANK_CACHE_TOKEN="stage-b-${branch}-${source_revision:0:10}" BERNINI_HELDOUT_PYTHON_BIN="${python_bin}" \
        PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 \
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH="${method_root}" \
        "${controller_source}" __infer_exec "${peak}" "${rank}" "${python_bin}" \
        "${python_bin}" -B -m torch.distributed.run --nnodes=2 --nproc_per_node=2 --node_rank="${rank}" \
          --master_addr="${work_node0}" --master_port="${port}" --no_python "${rank_cache_exec}" "${runtime_args[@]}"
  }
  launch_node "${work_job0}" "${work_node0}" 0 "${run_root}/logs/${branch}-node0.log" "${run_root}/logs/${branch}-node0-memory.txt"; pid0="${launched_pid}"
  launch_node "${work_job1}" "${work_node1}" 1 "${run_root}/logs/${branch}-node1.log" "${run_root}/logs/${branch}-node1-memory.txt"; pid1="${launched_pid}"
  wait_world4_pair "${branch}" "${pid0}" "${pid1}"
  assert_all_parents_running
  verify_memory "${run_root}/logs/${branch}-node0-memory.txt" "${work_job0}" 0 "${run_root}/logs/${branch}-node0-memory.json"
  verify_memory "${run_root}/logs/${branch}-node1-memory.txt" "${work_job1}" 1 "${run_root}/logs/${branch}-node1-memory.json"
  [[ -f "${output}/receipt.json" && ! -L "${output}/receipt.json" ]] || fail "branch receipt absent"
}

verify_pair() {
  local base="$1" trained="$2" mode="$3" output="$4"
  [[ ! -e "${output}" && ! -L "${output}" ]] || fail "pair receipt output not fresh"
  "${python_bin}" -I -B "${runtime_entry}" verify-pair --base-dir "${base}" --trained-dir "${trained}" --mode "${mode}" --output "${output}"
}

run_branch registered_probes_frozen_base registered-probes frozen_base "$(( first_port + 0 ))"
run_branch registered_probes_trained registered-probes trained "$(( first_port + 1 ))"
verify_pair "${run_root}/registered_probes_frozen_base" "${run_root}/registered_probes_trained" registered-probes "${run_root}/registered_probes_pair.json"
run_branch full40_evolved_target_frozen_base full40-evolved-target-all40-route-extrapolation frozen_base "$(( first_port + 2 ))"
run_branch full40_evolved_target_trained full40-evolved-target-all40-route-extrapolation trained "$(( first_port + 3 ))"
verify_pair "${run_root}/full40_evolved_target_frozen_base" "${run_root}/full40_evolved_target_trained" full40-evolved-target-all40-route-extrapolation "${run_root}/full40_evolved_target_pair.json"

"${python_bin}" -I -B - "${run_root}" "${source_revision}" "${source_archive_sha}" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
root=Path(sys.argv[1]); revision,archive_sha=sys.argv[2:]
def load(path): return json.loads(path.read_text(encoding="ascii"))
def digest_ok(value):
    unsigned=dict(value); declared=unsigned.pop("receipt_digest")
    raw=json.dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()==declared
pairs=[
    ("registered_probes_pair.json","registered-probes","registered_probes_frozen_base","registered_probes_trained"),
    ("full40_evolved_target_pair.json","full40-evolved-target-all40-route-extrapolation","full40_evolved_target_frozen_base","full40_evolved_target_trained"),
]
for name,mode,base,trained in pairs:
    p=load(root/name); assert digest_ok(p)
    assert p["schema_version"]=="bernini-source-noised-carrier-stage-b-pair-receipt-v1" and p["complete"] is True and p["mode"]==mode
    assert p["base"]["directory"]==str(root/base) and p["trained"]["directory"]==str(root/trained)
    assert p["same_sealed_source_dataset_prompt_seed_epsilon_scheduler_route"] is True
    assert p["only_adapter_parameter_values_intentionally_differ"] is True
    for key in ("anchor_used_as_condition","inversion_claimed","method_success_claimed","scientific_claim_authorized"): assert p[key] is False
    for arm,dirname in (("frozen_base",base),("trained",trained)):
        r=load(root/dirname/"receipt.json"); assert digest_ok(r) and r["complete"] is True and r["arm"]==arm
        assert r["dataset"]["iid"]=="00435ad621c44fac" and r["dataset"]["style_id"]==2
        assert r["dataset"]["row_digest"]=="1a04a38e06060d9ad29790a1185705ac8ba7401a30e8c5f738b1dcc0cecf6b44"
        assert r["dataset"]["reference_order"]==[0,80,40]
        assert r["dataset"]["clean_posterior_blob_sha256"]=="f9135728c18d32d5304bf2d7f8f5e9fbcda08421707d78331526ea62627edb39"
        assert r["dataset"]["style_posterior_blob_sha256"]=="b5892364716808b690289338a64fe1d8182c3b74621fad6ba1921a35f602cfab"
        assert r["dataset"]["reference_posterior_blob_sha256_in_order"]==["ece7e686348528604c86839d1bf1dad3eed9a024b6834808eacc229e494aa886","56b45906611576eaef0bb3e416ff876d6917e3721437c3405c799e02d292afc3","5eb4d7301aa61ff40a187bd29ba86d4a5bb86ee6c8b60e0bc9f0d79e0ee4cf4f"]
        assert r["source"]["sha256"]=="b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1"
        assert r["tensor_binding"]["seed"]==2026081401
        assert r["method_source_revision"]==revision and r["method_source_archive_sha256"]==archive_sha
        a=r["anchor_action_display"]
        assert a=={"present":False,"path":None,"sha256":None,"full_video_must_be_embedded_by_web_report":False,"used_as_model_condition":False,"opened_for_hash_binding_only":False,"decoded_by_model_runtime":False,"vae_encoded_by_model_runtime":False,"routed_to_transformer":False,"latent_or_rgb_transplanted":False}
        assert r["inversion_claimed"] is False and r["method_success_claimed"] is False and r["scientific_claim_authorized"] is False
binding={"schema_version":"bernini-stage-b-anchor-display-binding-v1","iid":"00435ad621c44fac","present":False,"unavailable_after_exact_remote_search":True,"passed_to_model_runtime":False,"used_as_model_condition":False,"stage_b_condition":False,"opened":False,"decoded":False,"vae_encoded":False,"routed":False,"latent_or_rgb_transplanted":False,"stage_b_results_must_later_embed_full_anchor_family":True}
binding["receipt_digest"]=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("ascii")).hexdigest()
out=root/"anchor-display-binding.json"; fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,"w",encoding="ascii") as f: f.write(json.dumps(binding,sort_keys=True,separators=(",",":"))+"\n")
PY

assert_idle_twice final
assert_all_parents_running
printf 'COMPLETE_STAGE_B_TRAIN_FIT_FRESH_NOISE_ENGINEERING_DECODE_ONLY iid=%s style_id=%s seed=%s modes=registered-probes,full40-evolved-target-all40-route-extrapolation heldout=false action_success=false method_success=false scientific_claim=false anchor_condition=false parents_retained=135407,135411,135412\n' \
  "${iid}" "${style_id}" "${seed}" >"${run_root}/controller.COMPLETE"
chmod 0400 "${run_root}/controller.COMPLETE"
trap - EXIT INT TERM HUP
echo "COMPLETE_STAGE_B_TRAIN_FIT_FRESH_NOISE_ENGINEERING_DECODE_ONLY output=${run_root}"
