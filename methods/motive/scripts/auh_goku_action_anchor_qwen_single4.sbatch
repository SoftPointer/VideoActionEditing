#!/usr/bin/env bash
# Four-GPU fallback for small Qwen3-VL screening batches.  The model is loaded
# once and all eight deterministic logical shards are consumed sequentially.
#SBATCH --job-name=goku-anchor-qwen4
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=02:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 077

source_snapshot="${MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT:?set MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT}"
selected="${MOTIVE_GOKU_ACTION_SELECTED:?set MOTIVE_GOKU_ACTION_SELECTED}"
qwen_model="${MOTIVE_GOKU_ACTION_QWEN_MODEL:?set MOTIVE_GOKU_ACTION_QWEN_MODEL}"
qwen_root="${MOTIVE_GOKU_ACTION_QWEN_OUTPUT:?set MOTIVE_GOKU_ACTION_QWEN_OUTPUT}"
final_output="${MOTIVE_GOKU_ACTION_FINAL_OUTPUT:?set MOTIVE_GOKU_ACTION_FINAL_OUTPUT}"
python_bin="${MOTIVE_GOKU_ACTION_PYTHON_BIN:?set MOTIVE_GOKU_ACTION_PYTHON_BIN}"
nframes="${MOTIVE_GOKU_ACTION_QWEN_NFRAMES:-12}"
max_pixels="${MOTIVE_GOKU_ACTION_QWEN_MAX_PIXELS:-589824}"
max_new_tokens="${MOTIVE_GOKU_ACTION_QWEN_MAX_NEW_TOKENS:-1536}"
final_seed="${MOTIVE_GOKU_ACTION_FINAL_SEED:-260730}"
allow_partial="${MOTIVE_GOKU_ACTION_ALLOW_PARTIAL:-1}"
code_root="${source_snapshot}/methods/motive"

for path in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${code_root}/motive/goku_action_anchor_qwen.py" \
  "${code_root}/motive/goku_action_anchor_finalize.py" \
  "${selected}" \
  "${qwen_model}/config.json" \
  "${python_bin}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[goku-anchor-qwen4] missing required path: ${path}" >&2
    exit 2
  fi
done
if [[ ! -x "${python_bin}" ]]; then
  echo "[goku-anchor-qwen4] Python is not executable: ${python_bin}" >&2
  exit 2
fi
for output in "${qwen_root}" "${final_output}"; do
  if [[ "${output}" != /* || "${output}" == "/" || -e "${output}" || -L "${output}" ]]; then
    echo "[goku-anchor-qwen4] output must be a new non-root absolute path: ${output}" >&2
    exit 2
  fi
  if [[ ! -d "${output%/*}" || ! -w "${output%/*}" ]]; then
    echo "[goku-anchor-qwen4] output parent is unavailable: ${output%/*}" >&2
    exit 2
  fi
done
if [[ "${allow_partial}" != "0" && "${allow_partial}" != "1" ]]; then
  echo "[goku-anchor-qwen4] allow-partial must be 0 or 1" >&2
  exit 2
fi

mkdir "${qwen_root}"
cache_root="${SLURM_TMPDIR:-/tmp}/motive-goku-anchor-qwen4-${SLURM_JOB_ID}"
mkdir -p \
  "${cache_root}/tmp" \
  "${cache_root}/xdg" \
  "${cache_root}/torch" \
  "${cache_root}/torch-kernels" \
  "${cache_root}/miopen" \
  "${cache_root}/miopen-custom"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHONPATH="${code_root}:${source_snapshot}"
export SLURM_EXPORT_ENV=ALL
cd "${code_root}"

echo "[goku-anchor-qwen4] host=$(hostname) job=${SLURM_JOB_ID} input=${selected}"
# The frozen implementation deliberately binds a four-GPU process to exactly
# four logical shards.  Reuse the same four allocated GPUs for the two halves
# in sequence; this is slower than dual-4GPU, but can backfill a half node.
for sequential_shards in "0,1,2,3" "4,5,6,7"; do
  echo "[goku-anchor-qwen4] logical_shards=${sequential_shards}"
  srun \
    --exclusive \
    --exact \
    --kill-on-bad-exit=1 \
    --ntasks=1 \
    --cpus-per-task=60 \
    --mem=450G \
    --gpus-per-task=4 \
    --gpu-bind=none \
    env \
      TMPDIR="${cache_root}/tmp" \
      XDG_CACHE_HOME="${cache_root}/xdg" \
      TORCH_HOME="${cache_root}/torch" \
      PYTORCH_KERNEL_CACHE_PATH="${cache_root}/torch-kernels" \
      MIOPEN_USER_DB_PATH="${cache_root}/miopen" \
      MIOPEN_CUSTOM_CACHE_DIR="${cache_root}/miopen-custom" \
      "${python_bin}" \
        -m motive.goku_action_anchor_qwen \
        --input "${selected}" \
        --output "${qwen_root}" \
        --model "${qwen_model}" \
        --all-shards-sequential \
        --sequential-shards "${sequential_shards}" \
        --shard-index 0 \
        --num-shards 8 \
        --nframes "${nframes}" \
        --max-pixels "${max_pixels}" \
        --max-new-tokens "${max_new_tokens}" \
        --attn-implementation sdpa \
        --allow-errors \
        --resume
done

finalizer_args=(
  --input "${selected}"
  --qwen-root "${qwen_root}"
  --output-dir "${final_output}"
  --seed "${final_seed}"
)
if [[ "${allow_partial}" == "1" ]]; then
  finalizer_args+=(--allow-partial)
fi
"${python_bin}" -m motive.goku_action_anchor_finalize "${finalizer_args[@]}"

for artifact in review_candidates.jsonl proposed_128.jsonl reserve_32.jsonl generation_manifest.jsonl summary.json done.json; do
  if [[ -L "${final_output}/${artifact}" || ! -f "${final_output}/${artifact}" ]]; then
    echo "[goku-anchor-qwen4] missing final artifact: ${artifact}" >&2
    exit 2
  fi
done
sha256sum "${final_output}/generation_manifest.jsonl" "${final_output}/summary.json" "${final_output}/done.json"
echo "[goku-anchor-qwen4] complete"
