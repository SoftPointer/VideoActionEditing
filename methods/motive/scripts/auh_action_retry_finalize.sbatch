#!/usr/bin/env bash
#SBATCH --job-name=motive-retry-finalize
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=00:20:00

set -Eeuo pipefail

run_root="${MOTIVE_ACTION_RUN_ROOT:?MOTIVE_ACTION_RUN_ROOT is required}"
runtime_repo="${MOTIVE_RUNTIME_REPO:?MOTIVE_RUNTIME_REPO is required}"
source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?MOTIVE_SOURCE_SNAPSHOT is required}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?MOTIVE_SOURCE_TREE_SHA256 is required}"
smoke_job_id="${MOTIVE_SMOKE_JOB_ID:?MOTIVE_SMOKE_JOB_ID is required}"
full_job_id="${MOTIVE_FULL_JOB_ID:?MOTIVE_FULL_JOB_ID is required}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

mkdir -p "${run_root}/analysis"
sacct \
  -j "${smoke_job_id},${full_job_id}" \
  --format=JobID,JobName%36,State,ExitCode,Elapsed,AllocTRES%60,NodeList%40 \
  -P > "${run_root}/analysis/slurm_sacct.tsv"

PYTHONPATH="${runtime_repo}/methods/motive" \
  "${python_bin}" -m motive.finalize_action_experiment \
  --run-root "${run_root}" \
  --allow-incomplete

submission_path="${run_root}/submission.json"
if [[ -s "${submission_path}" ]]; then
  final_state="incomplete"
  if jq -e '.complete == true' \
    "${run_root}/analysis/final_summary.json" >/dev/null; then
    final_state="complete"
  fi
  submission_tmp="${submission_path}.tmp.$$"
  jq --arg state "${final_state}" '.state = $state' \
    "${submission_path}" > "${submission_tmp}"
  mv "${submission_tmp}" "${submission_path}"
fi
echo "[motive-retry-finalizer] report=${run_root}/analysis/final_summary.json"
