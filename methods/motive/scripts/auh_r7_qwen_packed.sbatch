#!/usr/bin/env bash
#SBATCH --job-name=m7-qwen-new
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=1024G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=12:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

repo_root="${MOTIVE_REPO_ROOT:?set MOTIVE_REPO_ROOT}"
source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?set MOTIVE_SOURCE_SNAPSHOT}"
input_manifest="${MOTIVE_R7_QWEN_INPUT:?set MOTIVE_R7_QWEN_INPUT}"
output_root="${MOTIVE_R7_QWEN_OUTPUT_ROOT:?set MOTIVE_R7_QWEN_OUTPUT_ROOT}"
fused_output="${MOTIVE_R7_QWEN_FUSED_OUTPUT:?set MOTIVE_R7_QWEN_FUSED_OUTPUT}"
data_root="${MOTIVE_DATA_ROOT:?set MOTIVE_DATA_ROOT}"
python_bin="${QWEN_PYTHON_BIN:?set QWEN_PYTHON_BIN}"
model_path="${QWEN_MODEL_PATH:?set QWEN_MODEL_PATH}"
worker_script="${source_snapshot}/methods/motive/scripts/auh_qwen_visual_array.sbatch"
code_root="${source_snapshot}/methods/motive"

for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${code_root}/motive/qwen_filter.py" \
  "${code_root}/motive/r7_qwen_merge.py" \
  "${worker_script}" \
  "${input_manifest}" \
  "${data_root}" \
  "${python_bin}" \
  "${model_path}/config.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "[r7-qwen-packed] missing required path: ${required}" >&2
    exit 2
  fi
done

if [[ "${SLURM_NTASKS:-}" != "8" ]]; then
  echo "[r7-qwen-packed] expected exactly 8 real GPU tasks" >&2
  exit 2
fi

export MOTIVE_REPO_ROOT="${repo_root}"
export MOTIVE_CODE_ROOT="${code_root}"
export MOTIVE_DATA_ROOT="${data_root}"
export MOTIVE_OUTPUT_ROOT="${output_root}"
export MOTIVE_QWEN_INPUT="${input_manifest}"
export MOTIVE_QWEN_NUM_SHARDS=8
export MOTIVE_QWEN_ALLOW_LEGACY_PROVENANCE=0
export MOTIVE_QWEN_SCHEMA_FAILURE_POLICY=uncertain
export MOTIVE_QWEN_REPAIR_ATTEMPTS=1
export MOTIVE_QWEN_MAX_NEW_TOKENS=512
export MOTIVE_QWEN_MAX_PIXELS=147456
export QWEN_PYTHON_BIN="${python_bin}"
export QWEN_MODEL_PATH="${model_path}"
export MOTIVE_R7_QWEN_WORKER_SCRIPT="${worker_script}"
export PYTHONDONTWRITEBYTECODE=1
# Bind both worker and post-worker merge imports to the frozen snapshot.
# Without an explicit code root, all eight inference ranks can finish while
# the short parent-side merge fails to resolve ``motive.r7_qwen_merge``.
export PYTHONPATH="${code_root}:${source_snapshot}"
cd "${code_root}"

echo "[r7-qwen-packed] host=$(hostname) job=${SLURM_JOB_ID} tasks=${SLURM_NTASKS}"
echo "[r7-qwen-packed] input=${input_manifest} output=${output_root}"

# This is one packed eight-GPU allocation.  Every Slurm task executes real
# Qwen inference; there is no dependency/controller allocation holding GPUs.
srun \
  --kill-on-bad-exit=1 \
  --ntasks=8 \
  --ntasks-per-node=8 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  bash -lc '
    set -Eeuo pipefail
    export MOTIVE_QWEN_SHARD_INDEX="${SLURM_PROCID}"
    export MOTIVE_QWEN_NUM_SHARDS="${SLURM_NTASKS}"
    exec bash "${MOTIVE_R7_QWEN_WORKER_SCRIPT}"
  '

echo "[r7-qwen-packed] all eight inference shards completed"

# Fuse only after every real GPU worker exits successfully.  Keeping this
# short CPU validation in the same allocation avoids an empty dependency job
# and ensures a successful packed job always has a complete, hash-bound
# eight-shard artifact.
"${python_bin}" -m motive.r7_qwen_merge \
  --input "${input_manifest}" \
  --qwen-root "${output_root}/qwen_visual" \
  --output-dir "${fused_output}"

echo "[r7-qwen-packed] strict eight-shard merge completed: ${fused_output}"
