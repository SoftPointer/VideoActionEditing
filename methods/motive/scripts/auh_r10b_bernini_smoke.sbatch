#!/usr/bin/env bash
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=02:00:00
#SBATCH --job-name=motive-r10b-bernini-smoke

set -euo pipefail

source_snapshot="${MOTIVE_R10B_SOURCE_SNAPSHOT:?set MOTIVE_R10B_SOURCE_SNAPSHOT}"
source_tree_sha256="${MOTIVE_R10B_SOURCE_TREE_SHA256:?set MOTIVE_R10B_SOURCE_TREE_SHA256}"
manifest="${MOTIVE_R10B_MANIFEST:?set MOTIVE_R10B_MANIFEST}"
manifest_sha256="${MOTIVE_R10B_MANIFEST_SHA256:?set MOTIVE_R10B_MANIFEST_SHA256}"
track_cache="${MOTIVE_R10B_TRACK_CACHE:?set MOTIVE_R10B_TRACK_CACHE}"
track_cache_sha256="${MOTIVE_R10B_TRACK_CACHE_SHA256:?set MOTIVE_R10B_TRACK_CACHE_SHA256}"
output_dir="${MOTIVE_R10B_OUTPUT_DIR:?set MOTIVE_R10B_OUTPUT_DIR}"
model_path="${MOTIVE_R10B_MODEL_PATH:?set MOTIVE_R10B_MODEL_PATH}"
bernini_repo="${MOTIVE_R10B_BERNINI_REPO:?set MOTIVE_R10B_BERNINI_REPO}"
bernini_source_commit="${MOTIVE_R10B_BERNINI_SOURCE_COMMIT:?set MOTIVE_R10B_BERNINI_SOURCE_COMMIT}"
python_bin="${MOTIVE_R10B_PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

for path in \
  "${source_snapshot}" \
  "${manifest}" \
  "${track_cache}" \
  "${model_path}" \
  "${bernini_repo}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required R10B path: ${path}" >&2
    exit 2
  fi
done
for digest in \
  "${source_tree_sha256}" \
  "${manifest_sha256}" \
  "${track_cache_sha256}"; do
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "invalid SHA-256: ${digest}" >&2
    exit 2
  fi
done
if [[ ! "${bernini_source_commit}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "invalid Bernini source commit" >&2
  exit 2
fi
if [[ -e "${output_dir}" ]]; then
  echo "refusing to reuse R10B output: ${output_dir}" >&2
  exit 2
fi

verify_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "${path}" | cut -d ' ' -f 1)"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA-256 mismatch: ${path}" >&2
    exit 3
  fi
}

verify_sha256 "${manifest}" "${manifest_sha256}"
verify_sha256 "${track_cache}" "${track_cache_sha256}"

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${source_snapshot}:${source_snapshot}/methods/motive"

task_scratch="${SLURM_TMPDIR:-/tmp}/motive-r10b-bernini-${SLURM_JOB_ID}"
mkdir -p "${task_scratch}/miopen-user" "${task_scratch}/miopen-custom"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

"${python_bin}" -m motive.r10b_bernini_tangent_extract \
  --model-path "${model_path}" \
  --bernini-repo "${bernini_repo}" \
  --bernini-source-commit "${bernini_source_commit}" \
  --manifest "${manifest}" \
  --track-cache "${track_cache}" \
  --output-dir "${output_dir}" \
  --width 256 \
  --height 256 \
  --num-frames 17 \
  --max-samples 2 \
  --projection-dim 512 \
  --projection-seeds 260108851 260108852 \
  --diffusion-noise-seed 260108853 \
  --torch-seed 260108854 \
  --scheduler-steps 50 \
  --source-tree-sha256 "${source_tree_sha256}"

"${python_bin}" -m motive.r10b_bernini_tangent_extract \
  --output-dir "${output_dir}" \
  --validate-only

verify_sha256 "${manifest}" "${manifest_sha256}"
verify_sha256 "${track_cache}" "${track_cache_sha256}"
