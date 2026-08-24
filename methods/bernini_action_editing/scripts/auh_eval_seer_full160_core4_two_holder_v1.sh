#!/usr/bin/env bash
# Run the sealed SEER full160 held-out decode in two existing 64-GiB holders.
# This controller creates numbered child steps only; the three holder jobs stay
# alive after success, failure, or signal handling.

set -Eeuo pipefail
umask 077

fail() { echo "[seer160-two-holder] ERROR: $*" >&2; exit 2; }

readonly controller_requested="${BASH_SOURCE[0]}"
[[ "${controller_requested}" == /* ]] || fail "controller must be invoked by absolute path"
[[ -f "${controller_requested}" && ! -L "${controller_requested}" ]] || fail "controller must be an absolute plain file"
readonly controller_source="$(readlink -f -- "${controller_requested}")"
readonly expected_rank_cache_exec_sha256="f416f8f687a61788d4f687e500dae04af677072ec205fb560016cb5249b4f9c5"
readonly holder_user="guangyi.chen"

sha256_file() { sha256sum "$1" | awk '{print $1}'; }

is_allowed_holder() {
  case "$1" in
    135407|135411|135412) return 0 ;;
    *) return 1 ;;
  esac
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
  if [[ "${expected_count}" == physical-exact8 ]]; then
    [[ "${use_count}" == 8 && "${memory_count}" == 8 ]] || \
      fail "${label}: physical ROCm inventory is not exact8"
  else
    [[ "${use_count}" -ge 2 && "${use_count}" == "${memory_count}" ]] || \
      fail "${label}: child-visible ROCm inventory is incomplete"
  fi
  [[ -z "${busy}" ]] || fail "${label}: a GPU is already active"
}

node_preflight_exec() {
  local expected_controller_sha python_bin snapshot
  expected_controller_sha="${SEER_FULL160_TWO_HOLDER_CONTROLLER_SHA256:?set controller SHA}"
  python_bin="${1:?node preflight requires Python}"
  shift
  [[ "${expected_controller_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "node controller pin differs"
  [[ -f "${controller_source}" && ! -L "${controller_source}" ]] || fail "node controller file differs"
  [[ "$(sha256_file "${controller_source}")" == "${expected_controller_sha}" ]] || fail "node controller bytes differ"
  is_allowed_holder "${SLURM_JOB_ID:?node worker requires holder job}" || fail "node worker holder is outside allowlist"
  [[ "${SLURM_STEP_ID:?node worker requires numbered step}" =~ ^[0-9]+$ ]] || fail "node worker step identity differs"
  [[ -x "${python_bin}" ]] || fail "node Python differs"
  snapshot="$(rocm-smi --showuse --showmemuse --showpids)"
  assert_rocm_snapshot_idle "${snapshot}" "child-visible preflight" child-visible
  "${python_bin}" -B - <<'PY'
import torch
assert torch.version.hip
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
assert all(torch.cuda.memory_allocated(i) == 0 for i in range(2))
PY
  exec "$@"
}

if [[ "${1:-}" == "__node_preflight_exec" ]]; then
  shift
  node_preflight_exec "$@"
fi

usage() {
  cat >&2 <<'EOF'
usage:
  auh_eval_seer_full160_core4_two_holder_v1.sh canary <heldout-iid>
  auh_eval_seer_full160_core4_two_holder_v1.sh core4

Required environment:
  SEER_FULL160_HOLDER_JOB0, SEER_FULL160_HOLDER_JOB1
  SEER_FULL160_EVAL_RUN_ID
  SEER_FULL160_TWO_HOLDER_CONTROLLER_SHA256
  SEER_FULL160_RANK_CACHE_EXEC
EOF
  exit 2
}

mode="${1:-}"
shift || true
case "${mode}" in
  canary)
    [[ $# == 1 ]] || usage
    case "$1" in
      99cde432839f4240|6ea45d35943742bb|311c82f83eca4a7f|6d346c38cf504493) iids=("$1") ;;
      *) fail "canary IID is outside sealed core4" ;;
    esac
    ;;
  core4)
    [[ $# == 0 ]] || usage
    iids=(99cde432839f4240 6ea45d35943742bb 311c82f83eca4a7f 6d346c38cf504493)
    ;;
  *) usage ;;
esac

readonly job0="${SEER_FULL160_HOLDER_JOB0:?set first existing holder}"
readonly job1="${SEER_FULL160_HOLDER_JOB1:?set second existing holder}"
readonly run_id="${SEER_FULL160_EVAL_RUN_ID:?set fresh run ID}"
readonly controller_sha="${SEER_FULL160_TWO_HOLDER_CONTROLLER_SHA256:?set controller SHA}"
readonly rank_cache_exec="${SEER_FULL160_RANK_CACHE_EXEC:?set rank cache worker}"
readonly master_port_base="${SEER_FULL160_MASTER_PORT_BASE:-29740}"
readonly arm_timeout_seconds="${SEER_FULL160_ARM_TIMEOUT_SECONDS:-7200}"

is_allowed_holder "${job0}" || fail "first holder is outside allowlist"
is_allowed_holder "${job1}" || fail "second holder is outside allowlist"
[[ "${job0}" != "${job1}" ]] || fail "two distinct holders are required"
readonly node0="$(holder_node "${job0}")"
readonly node1="$(holder_node "${job1}")"
[[ "${node0}" != "${node1}" ]] || fail "two distinct nodes are required"
[[ "${run_id}" =~ ^[a-z0-9][a-z0-9-]{0,31}$ ]] || fail "run ID differs"
[[ "${controller_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "controller pin differs"
[[ "${master_port_base}" =~ ^[0-9]+$ ]] || fail "master port base differs"
(( master_port_base >= 1024 && master_port_base <= 65520 )) || fail "master port base is outside safe range"
[[ "${arm_timeout_seconds}" =~ ^[1-9][0-9]*$ ]] || fail "arm timeout differs"
(( arm_timeout_seconds >= 600 && arm_timeout_seconds <= 21600 )) || fail "arm timeout is outside safe range"
[[ -f "${controller_source}" && ! -L "${controller_source}" ]] || fail "controller must be a plain file"
[[ "$(sha256_file "${controller_source}")" == "${controller_sha}" ]] || fail "controller bytes differ"
[[ "${rank_cache_exec}" == /* ]] || fail "rank cache worker must be absolute"
[[ -x "${rank_cache_exec}" && ! -L "${rank_cache_exec}" ]] || fail "rank cache worker differs"
[[ "$(sha256_file "${rank_cache_exec}")" == "${expected_rank_cache_exec_sha256}" ]] || fail "rank cache worker bytes differ"

readonly exp=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_seer_event_erasure_v1_20260813
readonly stage="${exp}/staging/seer-full160-eval-overlay-v2"
readonly stage_closure="${stage}/stage-closure.json"
readonly stage_closure_sha=16d9429ab1bc456a0d3faac5310efc1f0301678f1058dee4de74912f17ab0c19
readonly stage_closure_digest=da277a895cd86d09697d0e7e2db1e8952c1aaff46b214cc537c61f5a94429453
readonly training_archive="${exp}/staging/seer-event-erasure-method-r1.tar"
readonly training_archive_sha=ec4064949b1cc8f4dba5d3c15fab375c9392a1407f9610ad01cee24c173bf822
readonly training_revision=6bae2f9a70fb851fd1bf87f5a01a1064a8fe8a6a
readonly overlay_archive="${stage}/seer-full160-eval-overlay-v1.tar"
readonly overlay_archive_sha=2eaa2f38b7a2cb220a3a4ecafe1ead9c53667a9108d20c9d8cc22dabbdb2c2f4
readonly overlay_manifest="${stage}/seer-full160-eval-overlay-v1.manifest.json"
readonly overlay_manifest_sha=001000a79ca69d7f6addb482424a7466e261e7f1b9f6f9d02f5ab7b2edac83b4
readonly source_binder="${stage}/bind_seer_full160_eval_source_v2.py"
readonly source_binder_sha=54271026c3a779e3a2b346d8fa5a5a18126a7fea7c330dfb5ce91e6ee4f6c1cc
readonly recovery_verifier="${stage}/tools/recover_seer_fm160_job135313_v1.py"
readonly recovery_verifier_sha=7949746fade65e0f1f63379429e58a3e3b193a0e2328f080c5f864cce6c7ec03
readonly recovery_receipt="${exp}/staging/seer-fm160-job135313-recovery-v1.json"
readonly recovery_receipt_sha=c8aacbb931cb0b1fa8ffa1067ceb2ba908973a5496b47da1e50133725faee537
readonly recovery_receipt_digest=acafda85f45db31ba299b89db597f4bdb4d1def1c23c90c06c785ba4407c0ab3
readonly adapter="${exp}/runs/seer-same-state-fm-160step-r1/checkpoint-00000160"
readonly adapter_model_sha=3dadbd4a1f2551c34942c52bcae2694bb5a695e88b9a6d471f2720f4fc074c5d
readonly bernini=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
readonly veomni=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
readonly checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly spec_sha=82fbe0f042d86f8d54aa254ce72a384e70aa5bdc3c1ac66d5422037cd4b4051c
readonly output_root="${exp}/heldout/full160-two-holder-${mode}-${run_id}"
readonly results_root="${output_root}/results"

[[ "$(sha256_file "${stage_closure}")" == "${stage_closure_sha}" ]] || fail "stage closure raw SHA differs"
[[ "$(sha256_file "${training_archive}")" == "${training_archive_sha}" ]] || fail "training archive differs"
[[ "$(sha256_file "${overlay_archive}")" == "${overlay_archive_sha}" ]] || fail "overlay archive differs"
[[ "$(sha256_file "${overlay_manifest}")" == "${overlay_manifest_sha}" ]] || fail "overlay manifest differs"
[[ "$(sha256_file "${source_binder}")" == "${source_binder_sha}" ]] || fail "source binder differs"
[[ "$(sha256_file "${recovery_verifier}")" == "${recovery_verifier_sha}" ]] || fail "recovery verifier differs"
[[ "$(sha256_file "${recovery_receipt}")" == "${recovery_receipt_sha}" ]] || fail "recovery receipt differs"
[[ "$(sha256_file "${adapter}/adapter/adapter_model.safetensors")" == "${adapter_model_sha}" ]] || fail "full160 adapter model differs"
[[ -x "${python_bin}" && -d "${bernini}" && -d "${veomni}" && -d "${checkpoint}" ]] || fail "runtime roots differ"

"${python_bin}" -I -B - "${stage_closure}" "${stage}" "${stage_closure_digest}" <<'PY'
import hashlib,json,os,stat,sys
from pathlib import Path
p=Path(sys.argv[1]); stage=Path(sys.argv[2]).resolve(strict=True); expected=sys.argv[3]
r=json.loads(p.read_text(encoding="utf-8")); unsigned=dict(r); declared=unsigned.pop("receipt_digest",None)
canonical=json.dumps(unsigned,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
assert declared == expected == hashlib.sha256(canonical).hexdigest()
assert Path(r["stage_root"]).resolve(strict=True) == stage
assert r["status"] == "sealed_eval_runtime_pending_independent_stage_audit_no_job_submitted"
assert r["eval_job_submitted"] is False and r["method_success_claimed"] is False
assert r["final_plain_file_count"] == 12 and r["payload_file_count"] == 11
assert r["training_method_source"] == {"archive_sha256":"ec4064949b1cc8f4dba5d3c15fab375c9392a1407f9610ad01cee24c173bf822","revision":"6bae2f9a70fb851fd1bf87f5a01a1064a8fe8a6a"}
assert r["inference_runtime_overlay"]["archive_sha256"] == "2eaa2f38b7a2cb220a3a4ecafe1ead9c53667a9108d20c9d8cc22dabbdb2c2f4"
assert r["inference_runtime_overlay"]["manifest_sha256"] == "001000a79ca69d7f6addb482424a7466e261e7f1b9f6f9d02f5ab7b2edac83b4"
for row in r["files"]:
    q=stage/row["path"]; m=q.lstat(); assert stat.S_ISREG(m.st_mode) and not q.is_symlink()
    assert stat.S_IMODE(m.st_mode) == int(row["mode"],8)
    assert q.stat().st_size == row["size"]
    assert hashlib.sha256(q.read_bytes()).hexdigest() == row["sha256"]
PY

"${python_bin}" -I -B "${recovery_verifier}" verify --receipt "${recovery_receipt}" >/dev/null
"${python_bin}" -I -B - "${recovery_receipt}" "${recovery_receipt_digest}" "${adapter}" <<'PY'
import json,sys
from pathlib import Path
r=json.load(open(sys.argv[1],encoding="utf-8"))
assert r["receipt_digest"] == sys.argv[2]
assert r["checkpoint_heldout_eligible"] is True and r["slurm_job_success"] is False
assert Path(r["final_checkpoint"]["path"]).resolve(strict=True) == Path(sys.argv[3]).resolve(strict=True)
PY
"${python_bin}" -I -B - "${adapter}/receipt.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding="utf-8"))
assert r["global_step"] == r["max_steps"] == 160
v=r["immutable_contract"]["value"]
assert v["method_source_revision"] == "6bae2f9a70fb851fd1bf87f5a01a1064a8fe8a6a"
assert v["method_source_archive_sha256"] == "ec4064949b1cc8f4dba5d3c15fab375c9392a1407f9610ad01cee24c173bf822"
PY

assert_parent_running() {
  local job="$1" expected_node record
  expected_node="$(holder_node "${job}")"
  record="$(scontrol show job -o "${job}")"
  [[ "${record}" == *"JobState=RUNNING"* ]] || fail "holder ${job} is not RUNNING"
  [[ "${record}" == *"NodeList=${expected_node}"* ]] || fail "holder ${job} node differs"
  [[ "${record}" == *"NumCPUs=64"* ]] || fail "holder ${job} CPU allocation differs"
  [[ "${record}" == *"AllocTRES=cpu=64,mem=64G,"* ]] || fail "holder ${job} memory allocation differs"
  [[ "${record}" == *"gres/gpu:mi210=8"* ]] || fail "holder ${job} GPU allocation differs"
}

assert_all_parents_running() {
  assert_parent_running 135407
  assert_parent_running 135411
  assert_parent_running 135412
}

numbered_steps() {
  squeue -s -j "$1" -h -o '%i' | awk '/[.][0-9]+$/ {print}'
}

assert_no_numbered_steps() {
  local job="$1" steps
  steps="$(numbered_steps "${job}")"
  [[ -z "${steps}" ]] || fail "holder ${job} already has a numbered step: ${steps}"
}

remote_process_snapshot() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" \
    "ps -u ${holder_user} -ww -o pid=,ppid=,comm=,args="
}

assert_remote_process_idle() {
  local job="$1" node="$2" snapshot hidden
  snapshot="$(remote_process_snapshot "${node}")"
  hidden="$(awk -v job="${job}" '
    {
      line=$0; comm=$3
      if (index(line,"/var/spool/slurmd/job" job "/slurm_script")) next
      if (comm=="sleep" && line ~ /sleep infinity[[:space:]]*$/) next
      if ((comm=="bash" || comm=="sh") && index(line,"holding allocation across nodes:") && index(line,"sleep infinity")) next
      if ((comm=="bash" || comm=="sh") && index(line,"ps -u guangyi.chen -ww -o")) next
      if (comm=="systemd" || comm=="(sd-pam)" || comm=="podman" || comm=="dbus-daemon" || comm=="sshd" || comm=="ps") next
      print
    }
  ' <<<"${snapshot}")"
  [[ -z "${hidden}" ]] || { printf '%s\n' "${hidden}" >&2; fail "holder ${job}/${node} has a hidden user process"; }
}

assert_remote_gpu_idle() {
  local node="$1" snapshot
  snapshot="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${node}" 'rocm-smi --showuse --showmemuse --showpids')"
  assert_rocm_snapshot_idle "${snapshot}" "${node} outer preflight" physical-exact8
}

assert_remote_idle_once() {
  local job="$1" node="$2"
  assert_parent_running "${job}"
  assert_no_numbered_steps "${job}"
  assert_remote_process_idle "${job}" "${node}"
  assert_remote_gpu_idle "${node}"
}

assert_pair_idle_twice() {
  local label="$1"
  assert_remote_idle_once "${job0}" "${node0}"
  assert_remote_idle_once "${job1}" "${node1}"
  sleep 2
  assert_remote_idle_once "${job0}" "${node0}"
  assert_remote_idle_once "${job1}" "${node1}"
  echo "IDLE_TWICE label=${label} jobs=${job0},${job1} nodes=${node0},${node1}"
}

assert_master_port_free() {
  local node="$1" port="$2" listeners
  listeners="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${node}" "ss -H -ltn 'sport = :${port}'")"
  [[ -z "${listeners}" ]] || fail "master port ${node}:${port} is occupied"
}

registered_child_pids=()
declare -A child_starttime=()
declare -A child_cmdline_sha=()
declare -A child_exe=()
declare -A child_job=()
declare -A child_node=()
launch_critical=0
pending_signal=""

proc_field() {
  local pid="$1" field="$2"
  awk -v field="${field}" '{print $field}' "/proc/${pid}/stat" 2>/dev/null
}

child_identity_matches() {
  local pid="$1" ppid start exe cmd_sha job node cmdline
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -n "${child_starttime[${pid}]-}" && -n "${child_job[${pid}]-}" ]] || return 1
  [[ -r "/proc/${pid}/stat" && -r "/proc/${pid}/cmdline" ]] || return 1
  ppid="$(proc_field "${pid}" 4)"
  start="$(proc_field "${pid}" 22)"
  exe="$(readlink -f -- "/proc/${pid}/exe" 2>/dev/null || true)"
  cmd_sha="$(sha256_file "/proc/${pid}/cmdline" 2>/dev/null || true)"
  job="${child_job[${pid}]}"
  node="${child_node[${pid}]}"
  cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
  [[ "${ppid}" == "$$" ]] || return 1
  [[ "${start}" == "${child_starttime[${pid}]}" ]] || return 1
  [[ "${exe}" == "${child_exe[${pid}]}" && "$(basename -- "${exe}")" == srun ]] || return 1
  [[ "${cmd_sha}" == "${child_cmdline_sha[${pid}]}" ]] || return 1
  [[ " ${cmdline} " == *" --jobid=${job} "* ]] || return 1
  [[ " ${cmdline} " == *" --nodelist=${node} "* ]] || return 1
}

register_child_pid() {
  local pid="$1" job="$2" node="$3" attempt ppid start exe cmd_sha cmdline
  for attempt in {1..100}; do
    if [[ -r "/proc/${pid}/stat" && -r "/proc/${pid}/cmdline" ]]; then
      ppid="$(proc_field "${pid}" 4)"
      start="$(proc_field "${pid}" 22)"
      exe="$(readlink -f -- "/proc/${pid}/exe" 2>/dev/null || true)"
      cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
      if [[ "${ppid}" == "$$" && "$(basename -- "${exe:-missing}")" == srun \
        && " ${cmdline} " == *" --jobid=${job} "* \
        && " ${cmdline} " == *" --nodelist=${node} "* ]]; then
        cmd_sha="$(sha256_file "/proc/${pid}/cmdline")"
        child_starttime["${pid}"]="${start}"
        child_cmdline_sha["${pid}"]="${cmd_sha}"
        child_exe["${pid}"]="${exe}"
        child_job["${pid}"]="${job}"
        child_node["${pid}"]="${node}"
        registered_child_pids+=("${pid}")
        return 0
      fi
    else
      break
    fi
    sleep 0.02
  done
  return 1
}

safe_signal_child() {
  local pid="$1" signal="$2"
  if child_identity_matches "${pid}"; then
    kill -"${signal}" "${pid}" 2>/dev/null || true
  elif [[ -e "/proc/${pid}" ]]; then
    echo "[seer160-two-holder] REFUSE_SIGNAL identity mismatch pid=${pid} signal=${signal}" >&2
  fi
}

unregister_child_pid() {
  local retired="$1" pid
  local kept=()
  for pid in "${registered_child_pids[@]:-}"; do
    [[ "${pid}" == "${retired}" ]] || kept+=("${pid}")
  done
  registered_child_pids=("${kept[@]}")
  unset 'child_starttime['"${retired}"']' 'child_cmdline_sha['"${retired}"']' \
    'child_exe['"${retired}"']' 'child_job['"${retired}"']' 'child_node['"${retired}"']'
}

terminate_registered_children() {
  local pid attempt
  for pid in "${registered_child_pids[@]:-}"; do
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || continue
    safe_signal_child "${pid}" TERM
  done
  for pid in "${registered_child_pids[@]:-}"; do
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || continue
    for attempt in {1..30}; do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${pid}" 2>/dev/null; then safe_signal_child "${pid}" KILL; fi
    wait "${pid}" 2>/dev/null || true
    unregister_child_pid "${pid}"
  done
}

wait_for_steps_gone() {
  local job steps
  for job in "${job0}" "${job1}"; do
    for _ in {1..60}; do
      steps="$(numbered_steps "${job}")"
      [[ -z "${steps}" ]] && break
      sleep 1
    done
    [[ -z "${steps:-}" ]] || return 1
  done
}

cleanup_on_exit() {
  local status=$?
  trap - EXIT INT TERM HUP
  terminate_registered_children
  if ! wait_for_steps_gone; then status=70; fi
  exit "${status}"
}

signal_exit() {
  pending_signal="${1:-signal}"
  (( launch_critical == 1 )) && return 0
  exit 130
}
trap 'signal_exit INT' INT
trap 'signal_exit TERM' TERM
trap 'signal_exit HUP' HUP
trap cleanup_on_exit EXIT

assert_all_parents_running
assert_pair_idle_twice startup
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"
mkdir -m 0700 "${output_root}" "${results_root}"
printf 'mode=%s run_id=%s job0=%s node0=%s job1=%s node1=%s controller_sha256=%s rank_cache_exec_sha256=%s stage_closure_sha256=%s stage_closure_digest=%s\n' \
  "${mode}" "${run_id}" "${job0}" "${node0}" "${job1}" "${node1}" \
  "${controller_sha}" "${expected_rank_cache_exec_sha256}" \
  "${stage_closure_sha}" "${stage_closure_digest}" \
  >"${output_root}/controller-binding.txt"
chmod 0400 "${output_root}/controller-binding.txt"

launch_node() {
  local iid="$1" arm="$2" port="$3" job="$4" node="$5" node_rank="$6"
  local method_root="$7" runner="$8" results="$9" log="${10}"
  local cache_token="${run_id}-${iid}-${arm}-n${node_rank}" pid
  local adaptation=()
  [[ "${arm}" == trained_adapter ]] && adaptation=(--adapter-checkpoint "${adapter}")
  launch_critical=1
  [[ -z "${pending_signal}" ]] || { launch_critical=0; exit 130; }
  srun --jobid="${job}" --nodelist="${node}" --nodes=1 --exclusive --exact \
    --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task=16 --mem=56G \
    --gres=gpu:mi210:2 \
    env \
      SEER_FULL160_TWO_HOLDER_CONTROLLER_SHA256="${controller_sha}" \
      BERNINI_HELDOUT_RANK_CACHE_TOKEN="${cache_token}" \
      BERNINI_HELDOUT_PYTHON_BIN="${python_bin}" \
      PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=bond0 \
      GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 \
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      "${controller_source}" __node_preflight_exec "${python_bin}" \
      "${python_bin}" -B "${runner}" \
        --spec "${method_root}/assets/self_generated_action_lora_heldout_core4_v1.json" \
        --expected-spec-sha256 "${spec_sha}" run-arm \
        --iid "${iid}" --arm "${arm}" --method-root "${method_root}" \
        --python-bin "${python_bin}" --bernini-root "${bernini}" \
        --veomni-root "${veomni}" --checkpoint "${checkpoint}" \
        "${adaptation[@]}" --trained-infer-runner infer_seer_same_state_full160_lora.py \
        --output-root "${results}" --master-port "${port}" \
        --torchrun-nnodes 2 --torchrun-nproc-per-node 2 \
        --torchrun-node-rank "${node_rank}" --torchrun-master-addr "${node0}" \
        --torchrun-worker-prefix "${rank_cache_exec}" \
        --method-source-revision "${training_revision}" \
        --method-source-archive-sha256 "${training_archive_sha}" \
    >"${log}" 2>&1 &
  pid=$!
  if ! register_child_pid "${pid}" "${job}" "${node}"; then
    launch_critical=0
    wait "${pid}" 2>/dev/null || true
    fail "could not bind child srun process identity: pid=${pid} job=${job}"
  fi
  current_arm_pids+=("${pid}")
  launch_critical=0
  echo "CHILD_SRUN_REGISTERED pid=${pid} job=${job} node=${node} iid=${iid} arm=${arm}" 
  [[ -z "${pending_signal}" ]] || exit 130
}

wait_current_arm() {
  local p0="${current_arm_pids[0]}" p1="${current_arm_pids[1]}"
  local rc0=0 rc1=0 done0=0 done1=0 failure_ticks=0 timed_out=0
  local started_at="${SECONDS}"
  while (( done0 == 0 || done1 == 0 )); do
    if (( done0 == 0 )) && ! kill -0 "${p0}" 2>/dev/null; then
      wait "${p0}" || rc0=$?
      unregister_child_pid "${p0}"
      done0=1
    fi
    if (( done1 == 0 )) && ! kill -0 "${p1}" 2>/dev/null; then
      wait "${p1}" || rc1=$?
      unregister_child_pid "${p1}"
      done1=1
    fi
    if (( timed_out == 0 && SECONDS - started_at >= arm_timeout_seconds )); then
      timed_out=1
      rc0=124
      rc1=124
      echo "[seer160-two-holder] ARM_TIMEOUT seconds=${arm_timeout_seconds} pids=${p0},${p1}" >&2
    fi
    if (( rc0 != 0 || rc1 != 0 )); then
      (( done0 == 0 )) && safe_signal_child "${p0}" TERM
      (( done1 == 0 )) && safe_signal_child "${p1}" TERM
      (( failure_ticks += 1 ))
      if (( failure_ticks >= 30 )); then
        (( done0 == 0 )) && safe_signal_child "${p0}" KILL
        (( done1 == 0 )) && safe_signal_child "${p1}" KILL
      fi
    fi
    (( done0 == 0 || done1 == 0 )) && sleep 1
  done
  (( rc0 == 0 && rc1 == 0 ))
}

run_arm_pair() {
  local iid="$1" arm="$2" port="$3" method_root="$4" runner="$5" case_root="$6"
  [[ "${#registered_child_pids[@]}" == 0 ]] || fail "child PID registry was not empty before arm launch"
  assert_all_parents_running
  assert_pair_idle_twice "pre-${iid}-${arm}"
  assert_master_port_free "${node0}" "${port}"
  sleep 1
  assert_master_port_free "${node0}" "${port}"
  current_arm_pids=()
  launch_node "${iid}" "${arm}" "${port}" "${job0}" "${node0}" 0 \
    "${method_root}" "${runner}" "${results_root}" "${case_root}/${arm}-node0.log"
  launch_node "${iid}" "${arm}" "${port}" "${job1}" "${node1}" 1 \
    "${method_root}" "${runner}" "${results_root}" "${case_root}/${arm}-node1.log"
  if ! wait_current_arm; then
    tail -n 160 "${case_root}/${arm}-node0.log" >&2 || true
    tail -n 160 "${case_root}/${arm}-node1.log" >&2 || true
    fail "two-node arm failed: ${iid}/${arm}"
  fi
  [[ "${#registered_child_pids[@]}" == 0 ]] || fail "child PID registry was not empty after arm reap"
  wait_for_steps_gone || fail "numbered child step remained after ${iid}/${arm}"
  assert_all_parents_running
}

first_runner=""
first_spec=""
case_index=0
for iid in "${iids[@]}"; do
  case_root="${output_root}/${iid}"
  base_root="${case_root}/base-source"
  runtime_root="${case_root}/runtime-source"
  method_root="${runtime_root}/methods/bernini_action_editing"
  runner="${method_root}/run_self_generated_action_lora_heldout_core4_v1.py"
  spec="${method_root}/assets/self_generated_action_lora_heldout_core4_v1.json"
  binding_receipt="${case_root}/eval-source-binding.json"
  case_binding_receipt="${case_root}/eval-execution-binding.json"
  [[ ! -e "${case_root}" ]] || fail "case output must be fresh: ${iid}"
  mkdir -m 0700 "${case_root}" "${base_root}"
  tar --delay-directory-restore --no-same-owner --no-same-permissions -xf "${training_archive}" -C "${base_root}"
  cp -R "${base_root}" "${runtime_root}"
  chmod u+w "${method_root}"
  chmod u+w "${method_root}/infer_seer_same_state_lora.py" \
    "${method_root}/run_self_generated_action_lora_heldout_core4_v1.py"
  tar --no-same-owner --no-same-permissions -xf "${overlay_archive}" -C "${runtime_root}"
  chmod 0400 "${method_root}/infer_seer_same_state_full160_lora.py" \
    "${method_root}/infer_seer_same_state_lora.py" \
    "${method_root}/run_self_generated_action_lora_heldout_core4_v1.py"
  chmod u-w "${method_root}"
  "${python_bin}" -I -B "${source_binder}" bind \
    --training-archive "${training_archive}" --overlay-archive "${overlay_archive}" \
    --overlay-manifest "${overlay_manifest}" \
    --expected-overlay-archive-sha256 "${overlay_archive_sha}" \
    --expected-overlay-manifest-sha256 "${overlay_manifest_sha}" \
    --recovery-receipt "${recovery_receipt}" \
    --expected-recovery-receipt-sha256 "${recovery_receipt_sha}" \
    --expected-recovery-receipt-digest "${recovery_receipt_digest}" \
    --base-root "${base_root}" --runtime-root "${runtime_root}" \
    --output "${binding_receipt}" >"${case_root}/eval-source-binding.stdout.json"
  "${python_bin}" -I -B "${source_binder}" verify-receipt --receipt "${binding_receipt}" >/dev/null
  [[ "$(sha256_file "${spec}")" == "${spec_sha}" ]] || fail "heldout spec differs"
  "${python_bin}" -I -B "${runner}" --spec "${spec}" \
    --expected-spec-sha256 "${spec_sha}" inspect --verify-files \
    >"${case_root}/inspect.json"
  (( base_port = master_port_base + case_index * 2 ))
  (( trained_port = base_port + 1 ))
  run_arm_pair "${iid}" frozen_base "${base_port}" "${method_root}" "${runner}" "${case_root}"
  run_arm_pair "${iid}" trained_adapter "${trained_port}" "${method_root}" "${runner}" "${case_root}"
  verify_args=(--spec "${spec}" --expected-spec-sha256 "${spec_sha}" verify-pair \
    --iid "${iid}" --adapter-checkpoint "${adapter}" --output-root "${results_root}")
  command -v ffmpeg >/dev/null 2>&1 && verify_args+=(--ffmpeg "$(command -v ffmpeg)")
  "${python_bin}" -I -B "${runner}" "${verify_args[@]}" >"${case_root}/paired-verification.json"
  "${python_bin}" -I -B "${source_binder}" finalize-case \
    --source-binding "${binding_receipt}" \
    --paired-receipt "${results_root}/${iid}/paired-receipt.json" \
    --output "${case_binding_receipt}" >"${case_root}/eval-execution-binding.stdout.json"
  "${python_bin}" -I -B "${source_binder}" verify-case --receipt "${case_binding_receipt}" >/dev/null
  [[ -f "${results_root}/${iid}/paired-receipt.json" ]] || fail "paired evidence is incomplete"
  [[ -n "${first_runner}" ]] || { first_runner="${runner}"; first_spec="${spec}"; }
  echo "PASS_GENERATION_NOT_METHOD_SUCCESS mode=${mode} iid=${iid}"
  (( case_index += 1 ))
done

if [[ "${mode}" == core4 ]]; then
  "${python_bin}" -I -B "${first_runner}" --spec "${first_spec}" \
    --expected-spec-sha256 "${spec_sha}" verify-core4 --output-root "${results_root}" \
    >"${output_root}/core4-verification.json"
fi

assert_all_parents_running
assert_no_numbered_steps "${job0}"
assert_no_numbered_steps "${job1}"
printf 'COMPLETE_GENERATION_NOT_METHOD_SUCCESS mode=%s cases=%s jobs=%s,%s parents_retained=135407,135411,135412\n' \
  "${mode}" "${#iids[@]}" "${job0}" "${job1}" >"${output_root}/controller.COMPLETE"
chmod 0400 "${output_root}/controller.COMPLETE"
trap - EXIT INT TERM HUP
echo "COMPLETE_GENERATION_NOT_METHOD_SUCCESS mode=${mode} output=${output_root}"
