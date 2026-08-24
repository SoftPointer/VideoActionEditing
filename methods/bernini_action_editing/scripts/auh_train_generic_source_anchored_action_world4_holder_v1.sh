#!/usr/bin/env bash
# One numbered WORLD4=DP1xSP4 child in a retained 8xMI210 holder.  The child
# consumes exactly one four-GPU XGMI island and trains one shared model.  This
# launcher never cancels, releases, requeues, or signals the retained parent.

set -Eeuo pipefail
umask 077

fail() { echo "[generic-action-world4-holder] ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

readonly confirm_child="${GSA_CONFIRM_CHILD:?controller confirmation is required}"
readonly arm_id="${GSA_ARM_ID:?bind registered arm}"
readonly holder_job="${GSA_HOLDER_JOB:?bind holder job}"
readonly holder_node="${GSA_HOLDER_NODE:?bind holder node}"
readonly execution_profile="${GSA_EXECUTION_PROFILE:?bind execution profile}"
readonly carrier_policy="${GSA_CARRIER_POLICY:?bind carrier policy}"
readonly run_root="${GSA_RUN_ROOT:?set fresh run root}"
readonly master_port="${GSA_MASTER_PORT:?set master port}"
readonly authority_plan="${GSA_AUTHORITY_PLAN:?set sealed arm plan}"
readonly authority_plan_sha="${GSA_AUTHORITY_PLAN_SHA256:?set arm plan SHA-256}"
readonly method_root="${GSA_METHOD_ROOT:?set sealed method root}"
readonly trainer_sha="${GSA_TRAINER_SHA256:?pin trainer SHA-256}"
readonly core_sha="${GSA_CORE_SHA256:?pin core SHA-256}"
readonly launcher_sha="${GSA_LAUNCHER_SHA256:?pin launcher SHA-256}"
readonly manifest_validator_sha="${GSA_MANIFEST_VALIDATOR_SHA256:-}"
readonly method_archive="${GSA_METHOD_ARCHIVE:?set sealed method archive}"
readonly method_archive_sha="${GSA_METHOD_ARCHIVE_SHA256:?pin method archive SHA-256}"
readonly method_manifest="${GSA_METHOD_MANIFEST:?set sealed method manifest}"
readonly method_manifest_sha="${GSA_METHOD_MANIFEST_SHA256:?pin method manifest SHA-256}"
readonly source_manifest="${GSA_SOURCE_MANIFEST:?set source 64/16/8 manifest}"
readonly source_manifest_sha="${GSA_SOURCE_MANIFEST_SHA256:?pin source manifest SHA-256}"
readonly representation_manifest="${GSA_REPRESENTATION_MANIFEST:-}"
readonly representation_manifest_sha="${GSA_REPRESENTATION_MANIFEST_SHA256:-}"
readonly source_pair_manifest="${GSA_SOURCE_PAIR_MANIFEST:-}"
readonly source_pair_manifest_sha="${GSA_SOURCE_PAIR_MANIFEST_SHA256:-}"
readonly resume_checkpoint="${GSA_RESUME_CHECKPOINT:-}"
readonly resume_checkpoint_sha="${GSA_RESUME_CHECKPOINT_SHA256:-}"
readonly resume_receipt="${GSA_RESUME_RECEIPT:-}"
readonly resume_receipt_sha="${GSA_RESUME_RECEIPT_SHA256:-}"
readonly python_bin="${GSA_PYTHON_BIN:?set frozen Python executable}"

readonly holder_user=guangyi.chen
readonly launch_confirmation=launch-approved-generic-pair-136309-136141
readonly source_authority_sha=128064fd335c4e48c567217c6e7bae43555a904875625c9d1e21178e6f7fcc3d
readonly gpu_memory_limit_gib=52
readonly host_memory_limit_gib=60
readonly bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/Bernini-2d2b4591
readonly veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/VeOmni-f90b3dc6
readonly checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256
readonly expected_checkpoint_manifest_sha=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831

case "${1:-}" in
  "") readonly launcher_role=parent ;;
  __child) readonly launcher_role=child ;;
  *) fail "launcher takes no positional arguments except internal __child" ;;
esac

[[ "${confirm_child}" == "${launch_confirmation}" ]] || fail "launch confirmation differs"
requires_action=false
requires_resume=false
expected_steps=0
complete_action_result=false
experiment=joint_source_anchored_v1
case "${holder_job}:${holder_node}:${arm_id}:${execution_profile}:${carrier_policy}" in
  136309:auh7-1b-gpu-280:joint_stage_r64:stage-r64:installed_trainable)
    expected_steps=64 ;;
  136309:auh7-1b-gpu-280:joint_resume_po40:resume-po40:resume_frozen_stage_r64)
    expected_steps=40; requires_action=true; requires_resume=true; complete_action_result=true ;;
  136141:auh7-1b-gpu-299:action_only_no_carrier:action-only40:not_installed_or_exact_zero_frozen)
    expected_steps=40; requires_action=true; complete_action_result=true; experiment=action_only_no_carrier_v1 ;;
  136309:auh7-1b-gpu-280:smoke_r:smoke-r:installed_trainable_disposable)
    expected_steps=1 ;;
  136309:auh7-1b-gpu-280:smoke_p:smoke-p:inactive_exact_zero_disposable)
    expected_steps=1; requires_action=true ;;
  136309:auh7-1b-gpu-280:smoke_o:smoke-o:inactive_exact_zero_disposable)
    expected_steps=1; requires_action=true ;;
  *) fail "holder/node/profile/carrier binding is outside the registered experiment" ;;
esac
readonly requires_action requires_resume expected_steps complete_action_result experiment

[[ "${master_port}" =~ ^[1-9][0-9]*$ ]] && (( master_port >= 1024 && master_port <= 65535 )) || \
  fail "master port differs"
[[ "${source_manifest_sha}" == "${source_authority_sha}" ]] || fail "source authority pin differs"
for digest_name in authority_plan_sha trainer_sha core_sha launcher_sha method_archive_sha method_manifest_sha source_manifest_sha; do
  [[ "${!digest_name}" =~ ^[0-9a-f]{64}$ ]] || fail "${digest_name} is not a frozen SHA-256"
done
for path_name in run_root authority_plan method_root method_archive method_manifest source_manifest python_bin; do
  value="${!path_name}"
  [[ "${value}" == /vast/users/guangyi.chen/* && "${value}" != / ]] || fail "${path_name} path differs"
done
[[ -d "${method_root}" && ! -L "${method_root}" && "$(readlink -f -- "${method_root}")" == "${method_root}" ]] || \
  fail "method root differs"
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python executable differs"
if [[ "${launcher_role}" == parent ]]; then
  [[ ! -e "${run_root}" && ! -L "${run_root}" && "$(realpath -m -- "${run_root}")" == "${run_root}" ]] || \
    fail "parent run root must be fresh and canonical"
else
  [[ -d "${run_root}" && ! -L "${run_root}" && "$(readlink -f -- "${run_root}")" == "${run_root}" ]] || \
    fail "child run root must be the canonical parent-created directory"
  [[ -d "${run_root}/logs" && ! -L "${run_root}/logs" && "$(readlink -f -- "${run_root}/logs")" == "${run_root}/logs" ]] || \
    fail "child log root must be the canonical parent-created directory"
fi

readonly trainer="${method_root}/train_generic_source_anchored_action_v1.py"
readonly core="${method_root}/generic_source_anchored_action_v1.py"
readonly controller="${method_root}/generic_source_anchored_action_pair_controller_v1.py"
readonly validator="${method_root}/tools/generic_action_manifest_v1.py"
readonly launcher="${method_root}/scripts/auh_train_generic_source_anchored_action_world4_holder_v1.sh"
readonly rank_exec_source="${method_root}/scripts/auh_generic_source_anchored_action_rank_exec_v1.sh"
for path in "${authority_plan}" "${method_archive}" "${method_manifest}" "${source_manifest}" "${trainer}" "${core}" "${controller}" "${launcher}" "${rank_exec_source}" "${checkpoint_manifest}"; do
  [[ -f "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || \
    fail "sealed input differs: ${path}"
done
[[ "$(sha256_file "${authority_plan}")" == "${authority_plan_sha}" ]] || fail "authority plan SHA differs"
[[ "$(sha256_file "${trainer}")" == "${trainer_sha}" ]] || fail "trainer SHA differs"
[[ "$(sha256_file "${core}")" == "${core_sha}" ]] || fail "core SHA differs"
[[ "$(sha256_file "${launcher}")" == "${launcher_sha}" ]] || fail "launcher SHA differs"
[[ "$(sha256_file "${method_archive}")" == "${method_archive_sha}" ]] || fail "method archive SHA differs"
[[ "$(sha256_file "${method_manifest}")" == "${method_manifest_sha}" ]] || fail "method manifest SHA differs"
[[ "$(sha256_file "${source_manifest}")" == "${source_manifest_sha}" ]] || fail "source manifest SHA differs"
[[ "$(sha256_file "${checkpoint_manifest}")" == "${expected_checkpoint_manifest_sha}" ]] || \
  fail "checkpoint manifest SHA differs"

action_args=()
if [[ "${requires_action}" == true ]]; then
  [[ "${manifest_validator_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "action profile has no frozen validator SHA-256"
  [[ -f "${validator}" && ! -L "${validator}" && "$(readlink -f -- "${validator}")" == "${validator}" ]] || \
    fail "action-manifest validator is absent or not canonical"
  [[ "$(sha256_file "${validator}")" == "${manifest_validator_sha}" ]] || fail "manifest validator SHA differs"
  for value in "${representation_manifest}" "${source_pair_manifest}"; do
    [[ "${value}" == /vast/users/guangyi.chen/* && "${value}" != / ]] || fail "action manifest path differs"
    [[ -f "${value}" && ! -L "${value}" && "$(readlink -f -- "${value}")" == "${value}" ]] || \
      fail "action manifest is absent or not canonical"
  done
  for digest in "${representation_manifest_sha}" "${source_pair_manifest_sha}"; do
    [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || fail "action manifest has no frozen SHA-256"
  done
  [[ "$(sha256_file "${representation_manifest}")" == "${representation_manifest_sha}" ]] || \
    fail "representation manifest SHA differs"
  [[ "$(sha256_file "${source_pair_manifest}")" == "${source_pair_manifest_sha}" ]] || \
    fail "source-pair manifest SHA differs"
  "${python_bin}" -B "${validator}" validate \
    --representation "${representation_manifest}" \
    --pairs "${source_pair_manifest}" >/dev/null || fail "action-manifest validator rejected authority"
  action_args=(
    --representation-manifest "${representation_manifest}"
    --expected-representation-manifest-sha256 "${representation_manifest_sha}"
    --source-pair-manifest "${source_pair_manifest}"
    --expected-source-pair-manifest-sha256 "${source_pair_manifest_sha}"
  )
else
  [[ -z "${manifest_validator_sha}" ]] || fail "R-only profile must not consume an action validator pin"
  [[ -z "${representation_manifest}${representation_manifest_sha}${source_pair_manifest}${source_pair_manifest_sha}" ]] || \
    fail "R-only profile must not consume action manifests"
fi

resume_args=()
if [[ "${requires_resume}" == true ]]; then
  for value in "${resume_checkpoint}" "${resume_receipt}"; do
    [[ "${value}" == /vast/users/guangyi.chen/* && "${value}" != / ]] || fail "resume path differs"
    [[ -f "${value}" && ! -L "${value}" && "$(readlink -f -- "${value}")" == "${value}" ]] || \
      fail "resume input is absent or not canonical"
  done
  for digest in "${resume_checkpoint_sha}" "${resume_receipt_sha}"; do
    [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || fail "resume input has no frozen SHA-256"
  done
  [[ "$(sha256_file "${resume_checkpoint}")" == "${resume_checkpoint_sha}" ]] || fail "R64 checkpoint SHA differs"
  [[ "$(sha256_file "${resume_receipt}")" == "${resume_receipt_sha}" ]] || fail "R64 receipt SHA differs"
  resume_args=(
    --resume-checkpoint "${resume_checkpoint}"
    --expected-resume-checkpoint-sha256 "${resume_checkpoint_sha}"
    --resume-receipt "${resume_receipt}"
    --expected-resume-receipt-sha256 "${resume_receipt_sha}"
  )
else
  [[ -z "${resume_checkpoint}${resume_checkpoint_sha}${resume_receipt}${resume_receipt_sha}" ]] || \
    fail "fresh profile must not consume an R64 checkpoint"
fi

assert_idle() {
  local snapshot count memory_count busy
  snapshot="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" 'rocm-smi --showuse --showmemuse --showpids')"
  count="$(awk '/GPU use \(%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  memory_count="$(awk '/GPU Memory Allocated \(VRAM%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  busy="$(awk '/GPU use \(%\)/ || /GPU Memory Allocated \(VRAM%\)/ {v=$NF; gsub(/[^0-9]/,"",v); if ((v+0)!=0) print}' <<<"${snapshot}")"
  [[ "${count}" == 8 && "${memory_count}" == 8 && -z "${busy}" ]] || \
    fail "holder GPU inventory is not idle exact8"
}

assert_two_xgmi4_islands() {
  local topology rows row_count xgmi_count pcie_count numa_zero_count numa_one_count
  topology="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" 'rocm-smi --showtopo')"
  rows="$(awk '$1 ~ /^GPU[0-7]$/ && /XGMI/ {print}' <<<"${topology}")"
  row_count="$(awk 'NF {n++} END {print n+0}' <<<"${rows}")"
  xgmi_count="$(awk '{for(i=1;i<=NF;i++) if($i=="XGMI") n++} END {print n+0}' <<<"${rows}")"
  pcie_count="$(awk '{for(i=1;i<=NF;i++) if($i=="PCIE") n++} END {print n+0}' <<<"${rows}")"
  numa_zero_count="$(awk '/GPU\[[0-3]\].*Topology.*Numa Node:/ && $NF==0 {n++} END {print n+0}' <<<"${topology}")"
  numa_one_count="$(awk '/GPU\[[4-7]\].*Topology.*Numa Node:/ && $NF==1 {n++} END {print n+0}' <<<"${topology}")"
  [[ "${row_count}" == 8 && "${xgmi_count}" == 24 && "${pcie_count}" == 32 && "${numa_zero_count}" == 4 && "${numa_one_count}" == 4 ]] || \
    fail "physical GPUs are not two exact same-XGMI4/NUMA islands"
}

if [[ "${launcher_role}" == child ]]; then
  shift
  [[ $# == 0 ]] || fail "unexpected child arguments"
  [[ "${SLURM_JOB_ID:?Slurm child required}" == "${holder_job}" ]] || fail "child holder job differs"
  [[ "$(hostname -s)" == "${holder_node}" ]] || fail "child node differs"
  [[ "${SLURM_STEP_ID:?numbered child step required}" =~ ^[0-9]+$ ]] || fail "child step differs"
  child_numbered_steps="$(squeue -s -j "${holder_job}" -h -o '%i' | awk '/[.][0-9]+$/ {print}')"
  [[ "${child_numbered_steps}" == "${holder_job}.${SLURM_STEP_ID}" ]] || \
    fail "child is not the holder's only numbered step: ${child_numbered_steps}"
  physical_gpus="${SLURM_STEP_GPUS:-}"
  case "${physical_gpus}" in
    0,1,2,3|4,5,6,7) ;;
    *) fail "Slurm did not grant one contiguous same-XGMI four-GPU island" ;;
  esac
  unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${physical_gpus}"
  visible_snapshot="$(rocm-smi --showuse --showmemuse)"
  visible_count="$(awk '/GPU use \(%\)/ {n++} END {print n+0}' <<<"${visible_snapshot}")"
  [[ "${visible_count}" == 4 ]] || fail "child-visible GPU inventory is not exact4"

  export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=bond0
  export GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  export PYTHONPATH="${method_root}"
  export GSA_RANK_CACHE_TOKEN="${arm_id}-${trainer_sha:0:10}"

  exec "${python_bin}" -B -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=4 \
    --master_addr=127.0.0.1 --master_port="${master_port}" \
    --no_python bash "${rank_exec_source}" "${trainer}" \
    --source-manifest "${source_manifest}" \
    --expected-source-manifest-sha256 "${source_manifest_sha}" \
    "${action_args[@]}" \
    "${resume_args[@]}" \
    --output "${run_root}/training" \
    --experiment "${experiment}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" \
    --execution-profile "${execution_profile}" \
    --parallel-topology world4-dp1-sp4 \
    --gpu-memory-limit-gib "${gpu_memory_limit_gib}" \
    --host-memory-limit-gib "${host_memory_limit_gib}" \
    --ack-upstream-training-use-forbidden \
    --ack-user-authorized-exploratory-training
fi

[[ $# == 0 ]] || fail "launcher takes no positional arguments"
job_record="$(scontrol show job -o "${holder_job}")"
[[ "${job_record}" == *"JobId=${holder_job} "* && "${job_record}" == *"JobState=RUNNING"* ]] || \
  fail "holder is not RUNNING"
[[ "${job_record}" == *"UserId=${holder_user}"* && "${job_record}" == *"NodeList=${holder_node}"* ]] || \
  fail "holder owner/node differs"
numbered_steps="$(squeue -s -j "${holder_job}" -h -o '%i' | awk '/[.][0-9]+$/ {print}')"
[[ -z "${numbered_steps}" ]] || fail "holder already has a numbered child: ${numbered_steps}"
assert_idle
sleep 2
assert_idle
assert_two_xgmi4_islands
[[ -z "$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" "ss -H -ltn 'sport = :${master_port}'")" ]] || \
  fail "master port is occupied"

numbered_steps="$(squeue -s -j "${holder_job}" -h -o '%i' | awk '/[.][0-9]+$/ {print}')"
[[ -z "${numbered_steps}" ]] || fail "holder acquired a numbered child before run-root creation: ${numbered_steps}"
mkdir -m 0700 "${run_root}" "${run_root}/logs"
child_pid=""
cleanup_child() {
  local status=$?
  trap - EXIT INT TERM HUP
  if [[ -n "${child_pid}" ]]; then
    kill -TERM "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup_child EXIT INT TERM HUP

set +e
srun --jobid="${holder_job}" --nodelist="${holder_node}" --nodes=1 --ntasks=1 \
  --exclusive --exact --kill-on-bad-exit=1 --immediate=5 \
  --cpus-per-task=32 --mem=60G --gpus-per-task=4 --gpu-bind=none --gres-flags=enforce-binding \
  env \
    GSA_CONFIRM_CHILD="${confirm_child}" \
    GSA_ARM_ID="${arm_id}" GSA_HOLDER_JOB="${holder_job}" GSA_HOLDER_NODE="${holder_node}" \
    GSA_EXECUTION_PROFILE="${execution_profile}" GSA_CARRIER_POLICY="${carrier_policy}" \
    GSA_RUN_ROOT="${run_root}" GSA_MASTER_PORT="${master_port}" \
    GSA_AUTHORITY_PLAN="${authority_plan}" GSA_AUTHORITY_PLAN_SHA256="${authority_plan_sha}" \
    GSA_METHOD_ROOT="${method_root}" GSA_TRAINER_SHA256="${trainer_sha}" GSA_CORE_SHA256="${core_sha}" \
    GSA_LAUNCHER_SHA256="${launcher_sha}" GSA_MANIFEST_VALIDATOR_SHA256="${manifest_validator_sha}" \
    GSA_METHOD_ARCHIVE="${method_archive}" GSA_METHOD_ARCHIVE_SHA256="${method_archive_sha}" \
    GSA_METHOD_MANIFEST="${method_manifest}" GSA_METHOD_MANIFEST_SHA256="${method_manifest_sha}" \
    GSA_SOURCE_MANIFEST="${source_manifest}" GSA_SOURCE_MANIFEST_SHA256="${source_manifest_sha}" \
    GSA_REPRESENTATION_MANIFEST="${representation_manifest}" GSA_REPRESENTATION_MANIFEST_SHA256="${representation_manifest_sha}" \
    GSA_SOURCE_PAIR_MANIFEST="${source_pair_manifest}" GSA_SOURCE_PAIR_MANIFEST_SHA256="${source_pair_manifest_sha}" \
    GSA_RESUME_CHECKPOINT="${resume_checkpoint}" GSA_RESUME_CHECKPOINT_SHA256="${resume_checkpoint_sha}" \
    GSA_RESUME_RECEIPT="${resume_receipt}" GSA_RESUME_RECEIPT_SHA256="${resume_receipt_sha}" \
    GSA_PYTHON_BIN="${python_bin}" \
    bash "${launcher}" __child \
  >"${run_root}/logs/train.log" 2>&1 &
child_pid=$!
wait "${child_pid}"
status=$?
child_pid=""
set -e

printf 'holder_job=%s\nholder_node=%s\narm_id=%s\nexecution_profile=%s\noptimizer_steps=%s\nchild_exit=%s\nparent_not_released=true\n' \
  "${holder_job}" "${holder_node}" "${arm_id}" "${execution_profile}" "${expected_steps}" "${status}" \
  >"${run_root}/controller.status"
if (( status != 0 )); then
  tail -n 240 "${run_root}/logs/train.log" >&2 || true
  exit "${status}"
fi

readonly receipt="${run_root}/training/run_receipt.json"
[[ -f "${receipt}" && ! -L "${receipt}" ]] || fail "training receipt missing"
PYTHONPATH="${method_root}" "${python_bin}" -B -c \
  'from pathlib import Path; import sys; import generic_source_anchored_action_pair_controller_v1 as c; c.validate_training_receipt(Path(sys.argv[1]), expected_profile=sys.argv[2], expected_steps=int(sys.argv[3]), expected_complete_action_result=sys.argv[4]=="true")' \
  "${receipt}" "${execution_profile}" "${expected_steps}" "${complete_action_result}" || \
  fail "training receipt failed profile/topology/memory validation"
case "${execution_profile}" in
  stage-r64)
    printf 'retain_stage_complete=true\ncomplete_action_result=false\nresume_checkpoint_required_for_po=true\ndecoded_review_complete=false\n' >>"${run_root}/controller.status"
    printf 'RETAIN_STAGE_COMPLETE holder=%s node=%s steps=64 complete_action_result=false parent_retained=true\n' \
      "${holder_job}" "${holder_node}" >"${run_root}/controller.RETAIN_STAGE_COMPLETE"
    ;;
  smoke-*)
    printf 'disposable_smoke=true\nformal_checkpoint=false\ncomplete_action_result=false\n' >>"${run_root}/controller.status"
    printf 'SMOKE_COMPLETE holder=%s node=%s profile=%s disposable=true parent_retained=true\n' \
      "${holder_job}" "${holder_node}" "${execution_profile}" >"${run_root}/controller.SMOKE_COMPLETE"
    ;;
  resume-po40|action-only40)
    printf 'training_complete=true\ncomplete_action_result=true\ndecoded_review_complete=false\n' >>"${run_root}/controller.status"
    printf 'TRAINING_COMPLETE holder=%s node=%s profile=%s decoded_review=false parent_retained=true\n' \
      "${holder_job}" "${holder_node}" "${execution_profile}" >"${run_root}/controller.TRAINING_COMPLETE"
    ;;
esac
echo "GENERIC_ACTION_CHILD_FINISHED arm=${arm_id} profile=${execution_profile} output=${run_root} review_complete=false"
