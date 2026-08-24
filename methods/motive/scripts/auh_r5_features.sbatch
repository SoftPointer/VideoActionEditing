#!/usr/bin/env bash
# Submit explicitly with: sbatch --array=0-1%2 auh_r5_features.sbatch
# No embedded array directive: a plain sbatch invocation fails closed below.
#SBATCH --job-name=motive-r5-features
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# AUH faculty QOS requires one GPU even though endpoint extraction is CPU work.
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=1-00:00:00

set -Eeuo pipefail

repo_root="${MOTIVE_REPO_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit}"
code_root="${MOTIVE_CODE_ROOT:-${repo_root}/methods/motive}"
data_root="${MOTIVE_DATA_ROOT:-${repo_root}/data/goku/subject_movement/extracted}"
input_manifest="${MOTIVE_R5_INPUT:-}"
output_root="${MOTIVE_R5_OUTPUT:-}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
task_count="${MOTIVE_R5_TASK_COUNT:-2}"
workers="${MOTIVE_R5_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"

if [[ -z "${input_manifest}" ]]; then
  echo "[r5-features] MOTIVE_R5_INPUT must name this run's prep/pilot_manifest.jsonl" >&2
  exit 2
fi
if [[ -z "${output_root}" ]]; then
  echo "[r5-features] MOTIVE_R5_OUTPUT must name this run's unique feature directory" >&2
  exit 2
fi
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "[r5-features] submit with --array=0-1%2; refusing a non-array job" >&2
  exit 2
fi
if [[ "${task_count}" != "2" ]]; then
  echo "[r5-features] production contract requires exactly two array tasks" >&2
  exit 2
fi
if [[ "${SLURM_ARRAY_TASK_ID}" != "0" && "${SLURM_ARRAY_TASK_ID}" != "1" ]]; then
  echo "[r5-features] invalid array task: ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
for required in \
  "${code_root}/motive/source_aware_features.py" \
  "${input_manifest}" \
  "${data_root}" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[r5-features] missing required path: ${required}" >&2
    exit 2
  fi
done

export PYTHONPATH="${code_root}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "${repo_root}"

"${python_bin}" -m motive.source_aware_features extract \
  --input "${input_manifest}" \
  --data-root "${data_root}" \
  --output-dir "${output_root}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --task-count "${task_count}" \
  --workers "${workers}" \
  --analysis-frames "${MOTIVE_R5_ANALYSIS_FRAMES:-20}" \
  --resize-width "${MOTIVE_R5_RESIZE_WIDTH:-160}" \
  --active-speed-threshold "${MOTIVE_R5_ACTIVE_SPEED_THRESHOLD:-0.005}" \
  --perceptual-hash-frames "${MOTIVE_R5_PHASH_FRAMES:-6}" \
  --instruction-dim "${MOTIVE_R5_INSTRUCTION_DIM:-512}" \
  --resume
