#!/usr/bin/env bash
#SBATCH --job-name=goku-anchor-qwen8
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1024G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=12:00:00
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
allow_partial="${MOTIVE_GOKU_ACTION_ALLOW_PARTIAL:-0}"
code_root="${source_snapshot}/methods/motive"
shard_count=8

require_absolute() {
  local label="$1"
  local value="$2"
  if [[ "${value}" != /* || "${value}" == "/" ]]; then
    echo "[goku-anchor-qwen] ${label} must be a non-root absolute path: ${value}" >&2
    exit 2
  fi
}

require_positive_integer() {
  local label="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[goku-anchor-qwen] ${label} must be a positive integer: ${value}" >&2
    exit 2
  fi
}

for binding in \
  "source_snapshot:${source_snapshot}" \
  "selected:${selected}" \
  "qwen_model:${qwen_model}" \
  "qwen_root:${qwen_root}" \
  "final_output:${final_output}" \
  "python_bin:${python_bin}"; do
  require_absolute "${binding%%:*}" "${binding#*:}"
done
require_positive_integer "MOTIVE_GOKU_ACTION_QWEN_NFRAMES" "${nframes}"
require_positive_integer "MOTIVE_GOKU_ACTION_QWEN_MAX_PIXELS" "${max_pixels}"
require_positive_integer \
  "MOTIVE_GOKU_ACTION_QWEN_MAX_NEW_TOKENS" \
  "${max_new_tokens}"
if (( nframes < 2 )); then
  echo "[goku-anchor-qwen] Qwen nframes must be at least two" >&2
  exit 2
fi
if [[ ! "${final_seed}" =~ ^[0-9]+$ ]]; then
  echo "[goku-anchor-qwen] final seed must be a non-negative integer" >&2
  exit 2
fi
if [[ "${allow_partial}" != "0" && "${allow_partial}" != "1" ]]; then
  echo "[goku-anchor-qwen] allow-partial must be 0 or 1" >&2
  exit 2
fi
if [[ "${SLURM_NTASKS:-}" != "1" ]]; then
  echo "[goku-anchor-qwen] allocation must expose exactly one Slurm task" >&2
  exit 2
fi

for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${code_root}/motive/goku_action_anchor_qwen.py" \
  "${code_root}/motive/goku_action_anchor_finalize.py" \
  "${selected}" \
  "${qwen_model}/config.json" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[goku-anchor-qwen] missing required path: ${required}" >&2
    exit 2
  fi
done
if [[ ! -x "${python_bin}" ]]; then
  echo "[goku-anchor-qwen] Python is not executable: ${python_bin}" >&2
  exit 2
fi
if [[ -L "${qwen_root}" ]]; then
  echo "[goku-anchor-qwen] Qwen output must not be a symlink: ${qwen_root}" >&2
  exit 2
fi
if [[ -e "${qwen_root}" && ! -d "${qwen_root}" ]]; then
  echo "[goku-anchor-qwen] Qwen output is not a directory: ${qwen_root}" >&2
  exit 2
fi
if [[ -e "${final_output}" || -L "${final_output}" ]]; then
  echo "[goku-anchor-qwen] refusing existing final output: ${final_output}" >&2
  exit 2
fi
qwen_parent="${qwen_root%/*}"
final_parent="${final_output%/*}"
if [[ ! -d "${qwen_parent}" || ! -w "${qwen_parent}" ]]; then
  echo "[goku-anchor-qwen] Qwen output parent is unavailable: ${qwen_parent}" >&2
  exit 2
fi
if [[ ! -d "${final_parent}" || ! -w "${final_parent}" ]]; then
  echo "[goku-anchor-qwen] final output parent is unavailable: ${final_parent}" >&2
  exit 2
fi
if [[ ! -d "${qwen_root}" ]]; then
  mkdir "${qwen_root}"
fi

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
# The allocation may intentionally start with a closed --export list.  Ask
# nested srun steps to inherit exports created inside that sanitized job.
export SLURM_EXPORT_ENV=ALL
cd "${code_root}"

echo "[goku-anchor-qwen] host=$(hostname) job=${SLURM_JOB_ID} physical_tasks=${SLURM_NTASKS} logical_shards=${shard_count}"
echo "[goku-anchor-qwen] input=${selected} output=${qwen_root}"

# Qwen3-VL-32B is loaded exactly once per worker.  Two isolated workers each
# see four GPUs and own one disjoint half of the eight deterministic SHA-256
# IID shards.  This keeps the frozen qwen_shard_NNN.jsonl/receipt contract
# while cutting wall-clock latency without creating eight model replicas.
cache_parent="${SLURM_TMPDIR:-/tmp}"
worker_a_cache="${cache_parent}/motive-goku-anchor-qwen-${SLURM_JOB_ID}-a"
worker_b_cache="${cache_parent}/motive-goku-anchor-qwen-${SLURM_JOB_ID}-b"
for worker_cache in "${worker_a_cache}" "${worker_b_cache}"; do
  mkdir -p \
    "${worker_cache}/tmp" \
    "${worker_cache}/xdg" \
    "${worker_cache}/torch" \
    "${worker_cache}/torch-kernels" \
    "${worker_cache}/miopen" \
    "${worker_cache}/miopen-custom"
done

run_worker() {
  local worker_cache="$1"
  local sequential_shards="$2"
  srun \
    --exclusive \
    --exact \
    --kill-on-bad-exit=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=60 \
    --mem=450G \
    --gpus-per-task=4 \
    --gpu-bind=none \
    env \
      TMPDIR="${worker_cache}/tmp" \
      XDG_CACHE_HOME="${worker_cache}/xdg" \
      TORCH_HOME="${worker_cache}/torch" \
      PYTORCH_KERNEL_CACHE_PATH="${worker_cache}/torch-kernels" \
      MIOPEN_USER_DB_PATH="${worker_cache}/miopen" \
      MIOPEN_CUSTOM_CACHE_DIR="${worker_cache}/miopen-custom" \
      "${python_bin}" \
        -m motive.goku_action_anchor_qwen \
        --input "${selected}" \
        --output "${qwen_root}" \
        --model "${qwen_model}" \
        --all-shards-sequential \
        --sequential-shards "${sequential_shards}" \
        --shard-index 0 \
        --num-shards "${shard_count}" \
        --nframes "${nframes}" \
        --max-pixels "${max_pixels}" \
        --max-new-tokens "${max_new_tokens}" \
        --attn-implementation sdpa \
        --allow-errors \
        --resume
}

run_worker "${worker_a_cache}" "0,1,2,3" &
worker_a_pid=$!
run_worker "${worker_b_cache}" "4,5,6,7" &
worker_b_pid=$!

worker_status=0
wait "${worker_a_pid}" || worker_status=$?
wait "${worker_b_pid}" || worker_status=$?
if (( worker_status != 0 )); then
  echo "[goku-anchor-qwen] a dual-4GPU worker failed: ${worker_status}" >&2
  exit "${worker_status}"
fi

echo "[goku-anchor-qwen] all eight logical inference shards completed"

finalizer_args=(
  --input "${selected}"
  --qwen-root "${qwen_root}"
  --output-dir "${final_output}"
  --seed "${final_seed}"
)
if [[ "${allow_partial}" == "1" ]]; then
  finalizer_args+=(--allow-partial)
fi
"${python_bin}" -m motive.goku_action_anchor_finalize \
  "${finalizer_args[@]}"

for artifact in \
  review_candidates.jsonl \
  proposed_128.jsonl \
  reserve_32.jsonl \
  generation_manifest.jsonl \
  summary.json \
  done.json; do
  if [[ -L "${final_output}/${artifact}" || ! -f "${final_output}/${artifact}" ]]; then
    echo "[goku-anchor-qwen] missing final artifact: ${artifact}" >&2
    exit 2
  fi
done

sha256sum \
  "${final_output}/review_candidates.jsonl" \
  "${final_output}/proposed_128.jsonl" \
  "${final_output}/reserve_32.jsonl" \
  "${final_output}/generation_manifest.jsonl" \
  "${final_output}/summary.json" \
  "${final_output}/done.json"
echo "[goku-anchor-qwen] strict merge and finalization complete"
