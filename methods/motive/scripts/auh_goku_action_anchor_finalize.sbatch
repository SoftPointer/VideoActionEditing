#!/usr/bin/env bash
#SBATCH --job-name=goku-anchor-finalize
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# AUH bgqos requires at least one GPU even for this CPU-only merge.
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=01:00:00

# Recover a completed eight-shard Qwen audit without loading Qwen or rerunning
# inference.  The strict finalizer itself validates complete IID coverage,
# shard placement, input/config/provenance digests, and the Qwen implementation
# digest against the frozen source snapshot before atomically publishing.

set -Eeuo pipefail
umask 077

source_snapshot="${MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT:?set MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT}"
selected="${MOTIVE_GOKU_ACTION_SELECTED:?set MOTIVE_GOKU_ACTION_SELECTED}"
qwen_root="${MOTIVE_GOKU_ACTION_QWEN_OUTPUT:?set MOTIVE_GOKU_ACTION_QWEN_OUTPUT}"
final_output="${MOTIVE_GOKU_ACTION_FINAL_OUTPUT:?set MOTIVE_GOKU_ACTION_FINAL_OUTPUT}"
python_bin="${MOTIVE_GOKU_ACTION_PYTHON_BIN:?set MOTIVE_GOKU_ACTION_PYTHON_BIN}"
final_seed="${MOTIVE_GOKU_ACTION_FINAL_SEED:-260730}"
allow_partial="${MOTIVE_GOKU_ACTION_ALLOW_PARTIAL:-0}"
approval_path="${MOTIVE_GOKU_ACTION_APPROVAL_PATH:-}"
code_root="${source_snapshot}/methods/motive"
shard_count=8

fail() {
  echo "[goku-anchor-finalize] $*" >&2
  exit 2
}

require_absolute() {
  local label="$1"
  local value="$2"
  if [[ "${value}" != /* || "${value}" == "/" ]]; then
    fail "${label} must be a non-root absolute path: ${value}"
  fi
}

for binding in \
  "source_snapshot:${source_snapshot}" \
  "selected:${selected}" \
  "qwen_root:${qwen_root}" \
  "final_output:${final_output}" \
  "python_bin:${python_bin}"; do
  require_absolute "${binding%%:*}" "${binding#*:}"
done
if [[ -n "${approval_path}" ]]; then
  require_absolute "approval_path" "${approval_path}"
fi
if [[ ! "${final_seed}" =~ ^[0-9]+$ ]]; then
  fail "MOTIVE_GOKU_ACTION_FINAL_SEED must be a non-negative integer"
fi
if [[ "${allow_partial}" != "0" && "${allow_partial}" != "1" ]]; then
  fail "MOTIVE_GOKU_ACTION_ALLOW_PARTIAL must be 0 or 1"
fi

for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${code_root}/motive/goku_action_anchor_qwen.py" \
  "${code_root}/motive/goku_action_anchor_finalize.py" \
  "${selected}" \
  "${python_bin}"; do
  if [[ -L "${required}" || ! -f "${required}" ]]; then
    fail "missing regular non-symlink runtime file: ${required}"
  fi
done
if [[ ! -s "${selected}" ]]; then
  fail "selected input is empty: ${selected}"
fi
if [[ ! -x "${python_bin}" ]]; then
  fail "Python is not executable: ${python_bin}"
fi
if [[ -n "${approval_path}" ]] && {
  [[ -L "${approval_path}" ]] || [[ ! -s "${approval_path}" ]];
}; then
  fail "approval must be a non-empty regular non-symlink file: ${approval_path}"
fi
if [[ -L "${qwen_root}" || ! -d "${qwen_root}" ]]; then
  fail "Qwen root must be a non-symlink directory: ${qwen_root}"
fi

shopt -s nullglob
actual_shards=("${qwen_root}"/qwen_shard_*.jsonl)
if (( ${#actual_shards[@]} != shard_count )); then
  fail "Qwen root must contain exactly eight qwen_shard_*.jsonl files"
fi
for (( shard_index=0; shard_index<shard_count; shard_index++ )); do
  printf -v shard_name 'qwen_shard_%03d.jsonl' "${shard_index}"
  shard_path="${qwen_root}/${shard_name}"
  if [[ -L "${shard_path}" || ! -f "${shard_path}" ]]; then
    fail "missing regular non-symlink Qwen shard: ${shard_path}"
  fi
done

if [[ -e "${final_output}" || -L "${final_output}" ]]; then
  fail "refusing existing final output: ${final_output}"
fi
final_parent="${final_output%/*}"
if [[ ! -d "${final_parent}" || ! -w "${final_parent}" ]]; then
  fail "final output parent is unavailable: ${final_parent}"
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
cd "${code_root}"

echo "[goku-anchor-finalize] host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[goku-anchor-finalize] selected=${selected}"
echo "[goku-anchor-finalize] qwen_root=${qwen_root}"
echo "[goku-anchor-finalize] output=${final_output}"
echo "[goku-anchor-finalize] approval=${approval_path:-none-proposal-only}"

finalizer_args=(
  --input "${selected}"
  --qwen-root "${qwen_root}"
  --output-dir "${final_output}"
  --seed "${final_seed}"
)
if [[ "${allow_partial}" == "1" ]]; then
  finalizer_args+=(--allow-partial)
fi
if [[ -n "${approval_path}" ]]; then
  finalizer_args+=(--approval "${approval_path}")
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
    fail "missing finalized artifact: ${artifact}"
  fi
done
for nonempty_artifact in \
  review_candidates.jsonl \
  proposed_128.jsonl \
  generation_manifest.jsonl \
  summary.json \
  done.json; do
  if [[ ! -s "${final_output}/${nonempty_artifact}" ]]; then
    fail "finalized artifact is unexpectedly empty: ${nonempty_artifact}"
  fi
done
if [[ "${allow_partial}" == "0" && ! -s "${final_output}/reserve_32.jsonl" ]]; then
  fail "strict finalization requires a non-empty reserve_32.jsonl"
fi

sha256sum \
  "${final_output}/review_candidates.jsonl" \
  "${final_output}/proposed_128.jsonl" \
  "${final_output}/reserve_32.jsonl" \
  "${final_output}/generation_manifest.jsonl" \
  "${final_output}/summary.json" \
  "${final_output}/done.json"
echo "[goku-anchor-finalize] standalone recovery complete"
