#!/usr/bin/env bash
set -Eeuo pipefail

run_root="${MOTIVE_ACTION_RUN_ROOT:?MOTIVE_ACTION_RUN_ROOT is required}"
prep_root="${MOTIVE_ACTION_PREP_ROOT:?MOTIVE_ACTION_PREP_ROOT is required}"
runtime_repo="${MOTIVE_RUNTIME_REPO:?MOTIVE_RUNTIME_REPO is required}"
source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?MOTIVE_SOURCE_SNAPSHOT is required}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?MOTIVE_SOURCE_TREE_SHA256 is required}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
submission_path="${run_root}/submission.json"

for required in \
  "${run_root}/contracts/retry_contract.json" \
  "${run_root}/lucy/e1_plain_lora/seed_2026/training_validation.json" \
  "${source_snapshot}/methods/motive/scripts/auh_lucy_action_retry.sbatch" \
  "${source_snapshot}/methods/motive/scripts/auh_action_retry_controller.sbatch"; do
  if [[ ! -s "${required}" ]]; then
    echo "[motive-retry-submit] missing required artifact: ${required}" >&2
    exit 2
  fi
done
if [[ -e "${submission_path}" ]]; then
  echo "[motive-retry-submit] refusing to overwrite ${submission_path}" >&2
  exit 2
fi
jq -e '.complete == true' \
  "${run_root}/lucy/e1_plain_lora/seed_2026/training_validation.json" \
  >/dev/null

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

common_export="ALL,MOTIVE_ACTION_RUN_ROOT=${run_root},MOTIVE_ACTION_PREP_ROOT=${prep_root},MOTIVE_RUNTIME_REPO=${runtime_repo},MOTIVE_SOURCE_SNAPSHOT=${source_snapshot},MOTIVE_SOURCE_TREE_SHA256=${source_tree_sha256},PYTHON_BIN=${python_bin}"
smoke_job_id="$(
  sbatch \
    --parsable \
    --array=0-5%2 \
    --output="${run_root}/slurm/%x_%A_%a.out" \
    --export="${common_export},MOTIVE_RETRY_PHASE=smoke" \
    "${source_snapshot}/methods/motive/scripts/auh_lucy_action_retry.sbatch"
)"
controller_job_id="$(
  sbatch \
    --parsable \
    --dependency="afterany:${smoke_job_id}" \
    --output="${run_root}/slurm/%x_%j.out" \
    --export="${common_export},MOTIVE_SMOKE_JOB_ID=${smoke_job_id}" \
    "${source_snapshot}/methods/motive/scripts/auh_action_retry_controller.sbatch"
)"

submission_tmp="${submission_path}.tmp.$$"
jq -n \
  --arg schema "motive-action-retry-submission-v1" \
  --arg submitted_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg smoke_job_id "${smoke_job_id}" \
  --arg controller_job_id "${controller_job_id}" \
  --arg source_tree_sha256 "${source_tree_sha256}" \
  '{
    schema: $schema,
    submitted_at_utc: $submitted_at_utc,
    smoke_job_id: $smoke_job_id,
    controller_job_id: $controller_job_id,
    source_tree_sha256: $source_tree_sha256,
    state: "smoke_submitted",
    max_concurrent_nodes: 2,
    gpus_per_node: 8,
    full_job_id: null,
    finalizer_job_id: null
  }' > "${submission_tmp}"
mv "${submission_tmp}" "${submission_path}"

echo "smoke_job_id=${smoke_job_id}"
echo "controller_job_id=${controller_job_id}"
