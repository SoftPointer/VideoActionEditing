#!/usr/bin/env bash
#SBATCH --job-name=lucy-action-screen
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=12:00:00
#SBATCH --array=0-2%2

set -Eeuo pipefail

repo_root="${MOTIVE_REPO_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit}"
run_root="${MOTIVE_ACTION_RUN_ROOT:-}"
source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:-}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:-}"
task_index="${SLURM_ARRAY_TASK_ID:-}"
if ! [[ "${task_index}" =~ ^[0-2]$ ]]; then
  echo "[motive-lucy] expected Slurm array index 0, 1, or 2" >&2
  exit 2
fi
if [[ -z "${run_root}" || -z "${source_snapshot}" || -z "${source_tree_sha256}" ]]; then
  echo "[motive-lucy] MOTIVE_ACTION_RUN_ROOT, MOTIVE_SOURCE_SNAPSHOT, and MOTIVE_SOURCE_TREE_SHA256 are required" >&2
  exit 2
fi
arms=(e1_plain_lora e2_random_router e3_motive_frozen)
arm="${arms[task_index]}"

export MOTIVE_REPO_ROOT="${repo_root}"
export PYTHON_BIN="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
export MOTIVE_ACTION_EXPERIMENT_ROOT="${MOTIVE_ACTION_EXPERIMENT_ROOT:-${run_root}/prep}"
export MOTIVE_LUCY_OUTPUT_ROOT="${MOTIVE_LUCY_OUTPUT_ROOT:-${run_root}/lucy}"
export MOTIVE_ACTION_STATUS_ROOT="${MOTIVE_ACTION_STATUS_ROOT:-${run_root}/status}"
export MOTIVE_TRAIN_SEED="${MOTIVE_TRAIN_SEED:-2026}"
export MOTIVE_LUCY_NUM_PROCESSES="${MOTIVE_LUCY_NUM_PROCESSES:-8}"
export MOTIVE_LUCY_GRAD_ACCUM="${MOTIVE_LUCY_GRAD_ACCUM:-1}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONDONTWRITEBYTECODE=1

"${PYTHON_BIN}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

"${PYTHON_BIN}" -c \
  'import torch; n=torch.cuda.device_count(); print(f"[motive-lucy] torch={torch.__version__} hip={torch.version.hip} visible_gpus={n}", flush=True); assert n == 8, f"expected 8 visible GPUs, got {n}"; assert torch.version.hip, "expected a ROCm build"'

cd "${repo_root}"
exec bash lucy/scripts/motive/run_goku_action_screen.sh "${arm}"
