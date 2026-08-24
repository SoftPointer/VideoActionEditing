#!/usr/bin/env bash
#SBATCH --job-name=motive-action-finalize
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# AUH faculty QOS requires one MI210 for controller/finalizer jobs.
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=00:15:00

set -Eeuo pipefail

repo_root="${MOTIVE_REPO_ROOT:?MOTIVE_REPO_ROOT is required}"
run_root="${MOTIVE_ACTION_RUN_ROOT:?MOTIVE_ACTION_RUN_ROOT is required}"
source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?MOTIVE_SOURCE_SNAPSHOT is required}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?MOTIVE_SOURCE_TREE_SHA256 is required}"
repr_job_id="${MOTIVE_REPR_JOB_ID:?MOTIVE_REPR_JOB_ID is required}"
lucy_job_id="${MOTIVE_LUCY_JOB_ID:?MOTIVE_LUCY_JOB_ID is required}"
controller_job_id="${MOTIVE_CONTROLLER_JOB_ID:-}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

mkdir -p "${run_root}/analysis"
job_query="${repr_job_id},${lucy_job_id}"
if [[ -n "${controller_job_id}" ]]; then
  job_query="${job_query},${controller_job_id}"
fi
sacct \
  -j "${job_query}" \
  --format=JobID,JobName%36,State,ExitCode,Elapsed,AllocTRES%60,NodeList%40 \
  -P > "${run_root}/analysis/slurm_sacct.tsv"

PYTHONPATH="${repo_root}/methods/motive" \
  "${python_bin}" -m motive.finalize_action_experiment \
  --run-root "${run_root}" \
  --allow-incomplete

echo "[motive-finalizer] report=${run_root}/analysis/final_summary.json"
