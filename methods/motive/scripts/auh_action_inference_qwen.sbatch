#!/usr/bin/env bash
#SBATCH --job-name=motive-action-qwen
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=06:00:00

set -Eeuo pipefail

inference_root="${MOTIVE_INFERENCE_ROOT:?MOTIVE_INFERENCE_ROOT is required}"
code_root="${MOTIVE_CODE_ROOT:?MOTIVE_CODE_ROOT is required}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?MOTIVE_SOURCE_TREE_SHA256 is required}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
model_path="${QWEN_MODEL_PATH:-/vast/users/guangyi.chen/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
input_manifest="${inference_root}/qwen/input.jsonl"
output_root="${inference_root}/qwen/shards"
manifest_root="${inference_root}/qwen/manifests"
log_root="${inference_root}/logs/qwen"

for required in \
  "${input_manifest}" \
  "${model_path}/config.json" \
  "${code_root}/methods/motive/motive/qwen_filter.py" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[motive-qwen-eval] missing required path: ${required}" >&2
    exit 2
  fi
done
for fresh_root in "${output_root}" "${manifest_root}"; do
  if [[ -d "${fresh_root}" ]] && find "${fresh_root}" -mindepth 1 -print -quit | grep -q .; then
    echo "[motive-qwen-eval] refusing non-empty output: ${fresh_root}" >&2
    exit 2
  fi
done

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
  'import torch; n=torch.cuda.device_count(); print(f"[motive-qwen-eval] torch={torch.__version__} hip={torch.version.hip} visible_gpus={n}", flush=True); assert n == 8, f"expected 8 visible GPUs, got {n}"; assert torch.version.hip, "expected ROCm"'

mkdir -p "${output_root}" "${manifest_root}" "${log_root}"
pids=()
for shard in 0 1 2 3 4 5 6 7; do
  tag="$(printf '%03d' "${shard}")"
  shard_manifest="${manifest_root}/shard-${tag}.jsonl"
  awk -v shard="${shard}" \
    '((NR - 1) % 8) == shard' "${input_manifest}" > "${shard_manifest}"
  shard_rows="$(awk 'END {print NR + 0}' "${shard_manifest}")"
  if [[ "${shard_rows}" != "4" ]]; then
    echo "[motive-qwen-eval] expected 4 rows in shard=${shard}, got ${shard_rows}" >&2
    exit 2
  fi
  cache_root="${SLURM_TMPDIR:-/tmp}/motive-action-qwen-${SLURM_JOB_ID}-${tag}"
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
    "${python_bin}" -u -m motive.qwen_filter \
    --input "${shard_manifest}" \
    --output "${output_root}/qwen-${tag}.jsonl" \
    --model "${model_path}" \
    --mode visual \
    --root / \
    --max-new-tokens 512 \
    --repair-attempts 1 \
    --schema-failure-policy uncertain \
    --nframes 6 \
    --max-pixels 147456 \
    --visual-input mosaic \
    --attn-implementation sdpa \
    --shard-index 0 \
    --num-shards 1 \
    --execution-shard-index "${shard}" \
    --execution-shard-count 8 \
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
  echo "[motive-qwen-eval] one or more Qwen shards failed" >&2
  exit 1
fi
echo "[motive-qwen-eval] complete outputs=${output_root}"
