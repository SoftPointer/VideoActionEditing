#!/usr/bin/env bash
#SBATCH --job-name=motive-retry-controller
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
prep_root="${MOTIVE_ACTION_PREP_ROOT:?MOTIVE_ACTION_PREP_ROOT is required}"
runtime_repo="${MOTIVE_RUNTIME_REPO:?MOTIVE_RUNTIME_REPO is required}"
source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?MOTIVE_SOURCE_SNAPSHOT is required}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?MOTIVE_SOURCE_TREE_SHA256 is required}"
smoke_job_id="${MOTIVE_SMOKE_JOB_ID:?MOTIVE_SMOKE_JOB_ID is required}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
status_path="${run_root}/controller_status.json"

export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

smoke_labels=(
  s0_identity_fixed
  s1_identity_random_linear_noreg
  s2_identity_random_linear_reg
  s3_fixed_random
  s4_random_linear
  s5_motive_linear
)
failed=()
for label in "${smoke_labels[@]}"; do
  arm_status="${run_root}/smoke/status/${label}_seed_2026.json"
  validation="${run_root}/smoke/lucy/${label}/seed_2026/training_validation.json"
  if ! jq -e '.state == "succeeded" and .exit_code == 0' "${arm_status}" >/dev/null 2>&1; then
    failed+=("${label}:status")
    continue
  fi
  if ! jq -e '.complete == true' "${validation}" >/dev/null 2>&1; then
    failed+=("${label}:validation")
  fi
done

mkdir -p "${run_root}/analysis"
if (( ${#failed[@]} > 0 )); then
  sacct \
    -j "${smoke_job_id}" \
    --format=JobID,JobName%36,State,ExitCode,Elapsed,AllocTRES%60,NodeList%40 \
    -P > "${run_root}/analysis/slurm_sacct.tsv"
  PYTHONPATH="${runtime_repo}/methods/motive" \
    "${python_bin}" -m motive.finalize_action_experiment \
    --run-root "${run_root}" \
    --allow-incomplete
  jq -n \
    --arg schema "motive-action-retry-controller-v1" \
    --arg state "smoke_failed" \
    --arg smoke_job_id "${smoke_job_id}" \
    --argjson failed "$(printf '%s\n' "${failed[@]}" | jq -R . | jq -s .)" \
    '{
      schema: $schema,
      state: $state,
      smoke_job_id: $smoke_job_id,
      failed: $failed,
      full_job_id: null,
      finalizer_job_id: null
    }' > "${status_path}"
  submission_path="${run_root}/submission.json"
  if [[ -s "${submission_path}" ]]; then
    submission_tmp="${submission_path}.tmp.$$"
    jq '.state = "smoke_failed"' \
      "${submission_path}" > "${submission_tmp}"
    mv "${submission_tmp}" "${submission_path}"
  fi
  echo "[motive-retry-controller] refusing full run; failed=${failed[*]}" >&2
  exit 1
fi

common_export="ALL,MOTIVE_ACTION_RUN_ROOT=${run_root},MOTIVE_ACTION_PREP_ROOT=${prep_root},MOTIVE_RUNTIME_REPO=${runtime_repo},MOTIVE_SOURCE_SNAPSHOT=${source_snapshot},MOTIVE_SOURCE_TREE_SHA256=${source_tree_sha256},PYTHON_BIN=${python_bin}"
full_job_id="$(
  sbatch \
    --parsable \
    --array=0-2%2 \
    --output="${run_root}/slurm/%x_%A_%a.out" \
    --export="${common_export},MOTIVE_RETRY_PHASE=full" \
    "${source_snapshot}/methods/motive/scripts/auh_lucy_action_retry.sbatch"
)"
finalizer_job_id="$(
  sbatch \
    --parsable \
    --dependency="afterany:${full_job_id}" \
    --output="${run_root}/slurm/%x_%j.out" \
    --export="${common_export},MOTIVE_SMOKE_JOB_ID=${smoke_job_id},MOTIVE_FULL_JOB_ID=${full_job_id}" \
    "${source_snapshot}/methods/motive/scripts/auh_action_retry_finalize.sbatch"
)"

status_tmp="${status_path}.tmp.$$"
jq -n \
  --arg schema "motive-action-retry-controller-v1" \
  --arg state "full_submitted" \
  --arg smoke_job_id "${smoke_job_id}" \
  --arg full_job_id "${full_job_id}" \
  --arg finalizer_job_id "${finalizer_job_id}" \
  '{
    schema: $schema,
    state: $state,
    smoke_job_id: $smoke_job_id,
    failed: [],
    full_job_id: $full_job_id,
    finalizer_job_id: $finalizer_job_id
  }' > "${status_tmp}"
mv "${status_tmp}" "${status_path}"
submission_path="${run_root}/submission.json"
if [[ -s "${submission_path}" ]]; then
  submission_tmp="${submission_path}.tmp.$$"
  jq \
    --arg full_job_id "${full_job_id}" \
    --arg finalizer_job_id "${finalizer_job_id}" \
    '.full_job_id = $full_job_id
     | .finalizer_job_id = $finalizer_job_id
     | .state = "full_submitted"' \
    "${submission_path}" > "${submission_tmp}"
  mv "${submission_tmp}" "${submission_path}"
fi
echo "[motive-retry-controller] full_job_id=${full_job_id} finalizer_job_id=${finalizer_job_id}"
