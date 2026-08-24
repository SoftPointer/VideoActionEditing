#!/usr/bin/env bash
#SBATCH --job-name=seer-eval256
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --exclusive
#SBATCH --exclude=auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 077

method="${SEER_EVAL_METHOD:?set constrained or direct}"
run_id="${SEER_EVAL_RUN_ID:-fresh256-exclusive-r5}"
[[ "${run_id}" =~ ^[a-z0-9][a-z0-9-]{0,63}$ ]] || {
  echo "invalid SEER_EVAL_RUN_ID=${run_id}" >&2
  exit 2
}
case "${method}" in
  constrained)
    trained_runner=infer_seer_same_state_lora.py
    ;;
  direct)
    trained_runner=infer_seer_scoped_lora.py
    ;;
  *)
    echo "unsupported SEER_EVAL_METHOD=${method}" >&2
    exit 2
    ;;
esac

exp=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_seer_event_erasure_v1_20260813
stage="${exp}/staging/seer-background-r4"
archive="${exp}/staging/seer-event-erasure-method-r1.tar"
archive_sha=ec4064949b1cc8f4dba5d3c15fab375c9392a1407f9610ad01cee24c173bf822
revision=6bae2f9a70fb851fd1bf87f5a01a1064a8fe8a6a
dispatcher="${stage}/run_self_generated_action_lora_heldout_core4_dispatch_20260813.py"
dispatcher_sha=e92af8ad0fc1b9399c3547cded143462f6099c1fd87bfed6b44cf2a85fd39f9b
spec_sha=82fbe0f042d86f8d54aa254ce72a384e70aa5bdc3c1ac66d5422037cd4b4051c
bernini=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python
output_root="${exp}/heldout/${method}-core4-${run_id}"
logs_root="${exp}/logs/eval-${method}-core4-${run_id}"
node_cache_root="${SLURM_TMPDIR:-/tmp}/seer-eval-${method}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

if [[ "${method}" == constrained ]]; then
  adapter="${exp}/runs/seer-same-state-fm-4step-r3/checkpoint-00000004"
else
  adapter="${exp}/runs/seer-direct-fm-4step-single4-r3/checkpoint-00000004"
fi

test "${SLURM_NNODES:?}" = 1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_ARRAY_TASK_COUNT:?}" = 2
test "$(sha256sum "${archive}" | awk '{print $1}')" = "${archive_sha}"
test "$(sha256sum "${dispatcher}" | awk '{print $1}')" = "${dispatcher_sha}"
test -f "${adapter}/adapter/adapter_model.safetensors"
mkdir -m 0700 -p "${output_root}" "${logs_root}"
test ! -e "${node_cache_root}"
mkdir -m 0700 "${node_cache_root}"
cleanup() {
  find "${node_cache_root}" -xdev -mindepth 1 -delete 2>/dev/null || true
  rmdir "${node_cache_root}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

case "${SLURM_ARRAY_TASK_ID}" in
  0) iids=(99cde432839f4240 311c82f83eca4a7f) ;;
  1) iids=(6ea45d35943742bb 6d346c38cf504493) ;;
  *) echo "unexpected array index" >&2; exit 2 ;;
esac

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"${python_bin}" -B -c 'import torch; assert torch.cuda.is_available(); assert torch.version.hip; assert torch.cuda.device_count() >= 4; print({"rocm_visible_devices": torch.cuda.device_count()})'

run_case() {
  local iid="$1"
  local case_root="${output_root}/${iid}"
  local source_root="${case_root}/source"
  local method_root="${source_root}/methods/bernini_action_editing"
  local runner="${method_root}/run_self_generated_action_lora_heldout_core4_dispatch_20260813.py"
  local spec="${method_root}/assets/self_generated_action_lora_heldout_core4_v1.json"
  local results="${case_root}/results"
  local cache_root="${node_cache_root}/${iid}"
  test ! -e "${case_root}"
  mkdir -m 0700 "${case_root}" "${source_root}" "${results}" "${cache_root}"
  tar --no-same-owner --no-same-permissions -xf "${archive}" -C "${source_root}"
  chmod u+w "${method_root}"
  cp -- "${dispatcher}" "${runner}"
  chmod 0400 "${runner}"
  chmod u-w "${method_root}"
  test "$(sha256sum "${runner}" | awk '{print $1}')" = "${dispatcher_sha}"
  test "$(sha256sum "${spec}" | awk '{print $1}')" = "${spec_sha}"
  "${python_bin}" -B "${runner}" --spec "${spec}" \
    --expected-spec-sha256 "${spec_sha}" inspect --verify-files

  run_arm() {
    local arm="$1" port="$2"
    local cache="${cache_root}/${arm}"
    local adaptation=()
    mkdir -m 0700 -p "${cache}/tmp" "${cache}/xdg" \
      "${cache}/torch-extensions" "${cache}/triton" \
      "${cache}/miopen-user" "${cache}/miopen-custom"
    if [[ "${arm}" == trained_adapter ]]; then
      adaptation=(--adapter-checkpoint "${adapter}")
    fi
    env \
      TMPDIR="${cache}/tmp" XDG_CACHE_HOME="${cache}/xdg" \
      TORCH_EXTENSIONS_DIR="${cache}/torch-extensions" \
      TRITON_CACHE_DIR="${cache}/triton" \
      MIOPEN_USER_DB_PATH="${cache}/miopen-user" \
      MIOPEN_CUSTOM_CACHE_DIR="${cache}/miopen-custom" \
      "${python_bin}" -B "${runner}" \
        --spec "${spec}" --expected-spec-sha256 "${spec_sha}" \
        run-arm --iid "${iid}" --arm "${arm}" \
        --method-root "${method_root}" --python-bin "${python_bin}" \
        --bernini-root "${bernini}" --veomni-root "${veomni}" \
        --checkpoint "${checkpoint}" "${adaptation[@]}" \
        --trained-infer-runner "${trained_runner}" \
        --output-root "${results}" --master-port "${port}" \
        --method-source-revision "${revision}" \
        --method-source-archive-sha256 "${archive_sha}"
  }

  echo "START method=${method} iid=${iid} arm=frozen_base"
  run_arm frozen_base 29511 >"${case_root}/frozen-base.log" 2>&1
  echo "START method=${method} iid=${iid} arm=trained_adapter"
  run_arm trained_adapter 29512 >"${case_root}/trained-adapter.log" 2>&1
  verify_args=(
    --spec "${spec}" --expected-spec-sha256 "${spec_sha}"
    verify-pair --iid "${iid}" --adapter-checkpoint "${adapter}"
    --output-root "${results}"
  )
  if command -v ffmpeg >/dev/null 2>&1; then
    verify_args+=(--ffmpeg "$(command -v ffmpeg)")
  fi
  "${python_bin}" -B "${runner}" "${verify_args[@]}" \
    >"${case_root}/paired-verification.json"
  echo "PASS_GENERATION_NOT_METHOD_SUCCESS method=${method} iid=${iid}"
}

for iid in "${iids[@]}"; do
  run_case "${iid}"
done
printf 'COMPLETE method=%s array_task=%s node=%s\n' \
  "${method}" "${SLURM_ARRAY_TASK_ID}" "${SLURMD_NODENAME}" \
  >"${output_root}/task-${SLURM_ARRAY_TASK_ID}.COMPLETE"
