#!/usr/bin/env bash
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=02:00:00
#SBATCH --job-name=motive-r10b-lucy-smoke

set -euo pipefail

source_snapshot="${MOTIVE_R10B_SOURCE_SNAPSHOT:?set MOTIVE_R10B_SOURCE_SNAPSHOT}"
source_tree_sha256="${MOTIVE_R10B_SOURCE_TREE_SHA256:?set MOTIVE_R10B_SOURCE_TREE_SHA256}"
manifest="${MOTIVE_R10B_MANIFEST:?set MOTIVE_R10B_MANIFEST}"
track_cache="${MOTIVE_R10B_TRACK_CACHE:?set MOTIVE_R10B_TRACK_CACHE}"
output_dir="${MOTIVE_R10B_OUTPUT_DIR:?set MOTIVE_R10B_OUTPUT_DIR}"
model_path="${MOTIVE_R10B_MODEL_PATH:?set MOTIVE_R10B_MODEL_PATH}"
python_bin="${MOTIVE_R10B_PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

for path in \
  "${source_snapshot}" \
  "${manifest}" \
  "${track_cache}" \
  "${model_path}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required R10B path: ${path}" >&2
    exit 2
  fi
done
if [[ ! "${source_tree_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid source tree SHA-256" >&2
  exit 2
fi
if [[ -e "${output_dir}" ]]; then
  echo "refusing to reuse R10B output: ${output_dir}" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${source_snapshot}:${source_snapshot}/methods/motive"

task_scratch="${SLURM_TMPDIR:-/tmp}/motive-r10b-${SLURM_JOB_ID}"
mkdir -p "${task_scratch}/miopen-user" "${task_scratch}/miopen-custom"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

"${python_bin}" -m motive.r10b_lucy_tangent_extract \
  --model-path "${model_path}" \
  --manifest "${manifest}" \
  --track-cache "${track_cache}" \
  --output-dir "${output_dir}" \
  --width 256 \
  --height 128 \
  --num-frames 17 \
  --max-samples 2 \
  --projection-dim 512 \
  --projection-seeds 260108847 260108848 \
  --diffusion-noise-seed 260108849 \
  --torch-seed 260108850 \
  --scheduler-steps 50 \
  --source-tree-sha256 "${source_tree_sha256}"

"${python_bin}" -m motive.r10b_lucy_tangent_extract \
  --output-dir "${output_dir}" \
  --validate-only
