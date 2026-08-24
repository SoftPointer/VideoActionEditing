#!/usr/bin/env bash
#SBATCH --job-name=motive-action-infer
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=08:00:00
#SBATCH --array=0-3%2

set -Eeuo pipefail

inference_root="${MOTIVE_INFERENCE_ROOT:?MOTIVE_INFERENCE_ROOT is required}"
code_root="${MOTIVE_CODE_ROOT:?MOTIVE_CODE_ROOT is required}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?MOTIVE_SOURCE_TREE_SHA256 is required}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
task_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

arms=(
  e1_plain_lora
  e2_fixed_random
  e2_random_router
  e3_motive_frozen
)
if (( task_index < 0 || task_index >= ${#arms[@]} )); then
  echo "[motive-infer] invalid task index: ${task_index}" >&2
  exit 2
fi
arm="${arms[task_index]}"
contract="${inference_root}/contract.json"
arm_root="${inference_root}/arms/${arm}"
log_root="${inference_root}/logs/${arm}"

for required in \
  "${contract}" \
  "${code_root}/lucy/infer_compare.py" \
  "${code_root}/methods/motive/motive/action_inference.py" \
  "${code_root}/methods/motive/scripts/action_source_snapshot.py" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[motive-infer] missing required path: ${required}" >&2
    exit 2
  fi
done
if [[ -e "${inference_root}/status/${arm}.json" ]]; then
  echo "[motive-infer] refusing completed/reused arm: ${arm}" >&2
  exit 2
fi
if [[ -d "${arm_root}" ]] && find "${arm_root}" -mindepth 1 -print -quit | grep -q .; then
  echo "[motive-infer] refusing non-empty arm output: ${arm_root}" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${code_root}/methods/motive:${code_root}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"${python_bin}" \
  "${code_root}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${code_root}" \
  --expected-tree-sha256 "${source_tree_sha256}"
"${python_bin}" -c \
  'import torch; n=torch.cuda.device_count(); print(f"[motive-infer] torch={torch.__version__} hip={torch.version.hip} visible_gpus={n}", flush=True); assert n == 8, f"expected 8 visible GPUs, got {n}"; assert torch.version.hip, "expected ROCm"'

manifest="$(jq -r '.selected_manifest.path' "${contract}")"
model_path="$(jq -r '.model_path' "${contract}")"
checkpoint="$(jq -r --arg arm "${arm}" '.checkpoints[$arm].path' "${contract}")"
if [[ "${arm}" == "e1_plain_lora" ]]; then
  config="$(jq -r '.plain_config.path' "${contract}")"
else
  config="$(jq -r '.router_config.path' "${contract}")"
fi
for required in "${manifest}" "${model_path}" "${checkpoint}" "${config}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[motive-infer] contract path is missing: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${arm_root}" "${log_root}"
common_args=(
  --config "${config}"
  --model-path "${model_path}"
  --manifest-path "${manifest}"
  --checkpoints "${checkpoint}"
  --output-dir "${arm_root}"
  --num-samples 8
  --sample-offset 0
  --num-shards 8
  --seed 2026
  --width 832
  --height 480
  --num-frames 81
  --sample-mode uniform
  --fps 25
  --guidance-scale 5
  --num-inference-steps 50
  --device cuda
  --dtype bfloat16
  --vae-dtype bfloat16
  --vae-enable-tiling
  --vae-enable-slicing
  --adapter-scale 0
  --lora-rank 64
  --lora-alpha 64
  --lora-scale 1
  --skip-base-generation
)
if [[ "${arm}" != "e1_plain_lora" ]]; then
  common_args+=(
    --v10-sparse-lora
    --v10-lora-num-groups 8
    --v10-lora-gate-floor 0.05
    --v10-lora-gate-scale 1
    --v10-lora-gate-mode learned_residual_components
    --v10-lora-learned-feature-dim 512
    --v10-lora-learned-hidden-dim 128
    --v10-lora-learned-topk 2
    --v10-lora-learned-temperature 0.7
    --v10-lora-residual-gate-scale 0.05
  )
fi

pids=()
for shard in 0 1 2 3 4 5 6 7; do
  cache_root="${SLURM_TMPDIR:-/tmp}/motive-action-infer-${SLURM_JOB_ID}-${arm}-${shard}"
  mkdir -p \
    "${cache_root}/tmp" \
    "${cache_root}/xdg" \
    "${cache_root}/huggingface" \
    "${cache_root}/torch" \
    "${cache_root}/kernels" \
    "${cache_root}/inductor" \
    "${cache_root}/triton" \
    "${cache_root}/miopen"
  env \
    HIP_VISIBLE_DEVICES="${shard}" \
    CUDA_VISIBLE_DEVICES="${shard}" \
    TMPDIR="${cache_root}/tmp" \
    XDG_CACHE_HOME="${cache_root}/xdg" \
    HF_HOME="${cache_root}/huggingface" \
    TORCH_HOME="${cache_root}/torch" \
    PYTORCH_KERNEL_CACHE_PATH="${cache_root}/kernels" \
    TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor" \
    TRITON_CACHE_DIR="${cache_root}/triton" \
    MIOPEN_USER_DB_PATH="${cache_root}/miopen" \
    "${python_bin}" -u -m lucy.infer_compare \
    "${common_args[@]}" \
    --shard-index "${shard}" \
    > "${log_root}/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "[motive-infer] one or more Python inference shards failed for ${arm}" >&2
  exit 1
fi

"${python_bin}" -m motive.action_inference validate-arm \
  --output-root "${inference_root}" \
  --arm "${arm}"
echo "[motive-infer] complete arm=${arm} outputs=${arm_root}"
