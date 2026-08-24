#!/usr/bin/env bash
# Submit with an afterok dependency on the two-task extraction array.
#SBATCH --job-name=motive-r5-finalize
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# AUH faculty QOS requires one GPU although global DSU/finalization is CPU work.
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=02:00:00

set -Eeuo pipefail

repo_root="${MOTIVE_REPO_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit}"
code_root="${MOTIVE_CODE_ROOT:-${repo_root}/methods/motive}"
output_root="${MOTIVE_R5_OUTPUT:-}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
task_count="${MOTIVE_R5_TASK_COUNT:-2}"

if [[ -z "${output_root}" ]]; then
  echo "[r5-features-finalize] MOTIVE_R5_OUTPUT must name this run's unique feature directory" >&2
  exit 2
fi
if [[ "${task_count}" != "2" ]]; then
  echo "[r5-features-finalize] production contract requires exactly two tasks" >&2
  exit 2
fi
for required in \
  "${code_root}/motive/source_aware_features.py" \
  "${output_root}/tasks/task-000/done.json" \
  "${output_root}/tasks/task-001/done.json" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[r5-features-finalize] missing required path: ${required}" >&2
    exit 2
  fi
done

export PYTHONPATH="${code_root}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "${repo_root}"

"${python_bin}" -m motive.source_aware_features finalize \
  --output-dir "${output_root}" \
  --task-count "${task_count}" \
  --data-seed "${MOTIVE_R5_DATA_SEED:-260108828}" \
  --train-fraction "${MOTIVE_R5_TRAIN_FRACTION:-0.8}" \
  --validation-fraction "${MOTIVE_R5_VALIDATION_FRACTION:-0.1}" \
  --maximum-hamming-fraction "${MOTIVE_R5_MAX_HAMMING_FRACTION:-0.10}" \
  --minimum-positive-train "${MOTIVE_R5_MIN_POSITIVE_TRAIN:-2}" \
  --minimum-positive-validation "${MOTIVE_R5_MIN_POSITIVE_VALIDATION:-1}" \
  --minimum-positive-test "${MOTIVE_R5_MIN_POSITIVE_TEST:-1}" \
  --resume
