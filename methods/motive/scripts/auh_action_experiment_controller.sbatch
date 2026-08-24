#!/usr/bin/env bash
#SBATCH --job-name=motive-action-controller
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
run_id="${MOTIVE_RUN_ID:?MOTIVE_RUN_ID is required}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
status_path="${run_root}/status/controller.json"
submission_path="${run_root}/contracts/submission.json"

export MOTIVE_REPO_ROOT="${repo_root}"
export MOTIVE_ACTION_RUN_ROOT="${run_root}"
export MOTIVE_SOURCE_SNAPSHOT="${source_snapshot}"
export MOTIVE_SOURCE_TREE_SHA256="${source_tree_sha256}"
export MOTIVE_ACTION_EXPERIMENT_ROOT="${run_root}/prep"
export MOTIVE_LUCY_OUTPUT_ROOT="${run_root}/lucy"
export MOTIVE_ACTION_STATUS_ROOT="${run_root}/status"
export PYTHON_BIN="${python_bin}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${run_root}/status" "${run_root}/logs" "${run_root}/contracts"
if [[ -e "${status_path}" ]]; then
  echo "[motive-controller] refusing to reuse status path: ${status_path}" >&2
  exit 2
fi

started_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
on_exit() {
  exit_code=$?
  trap - EXIT
  state="failed"
  if [[ "${exit_code}" -eq 0 ]]; then
    state="succeeded"
  fi
  temporary="${status_path}.tmp.$$"
  jq -n \
    --arg schema "motive-action-controller-status-v1" \
    --arg state "${state}" \
    --arg started_at_utc "${started_at_utc}" \
    --arg ended_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg slurm_job_id "${SLURM_JOB_ID:-}" \
    --arg repr_job_id "${repr_job_id}" \
    --argjson exit_code "${exit_code}" \
    '{
      schema: $schema,
      state: $state,
      started_at_utc: $started_at_utc,
      ended_at_utc: $ended_at_utc,
      slurm_job_id: $slurm_job_id,
      repr_job_id: $repr_job_id,
      exit_code: $exit_code
    }' > "${temporary}"
  mv "${temporary}" "${status_path}"
  exit "${exit_code}"
}
trap on_exit EXIT

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

repr_state="$(
  sacct -X -n -P -j "${repr_job_id}" --format=State \
    | head -n 1 \
    | cut -d'|' -f1
)"
repr_status="${run_root}/status/representation.json"
repr_status_state="$(
  jq -r '.state // empty' "${repr_status}" 2>/dev/null || true
)"
# afterany can release the controller a few seconds before slurmdbd exposes the
# parent in sacct.  The atomic status is written only after all three
# checkpoints and sidecars pass, so it is the authoritative hand-off.
if [[ "${repr_status_state}" != "succeeded" ]]; then
  echo "[motive-controller] representation failed or incomplete: state=${repr_state}" >&2
  PYTHONPATH="${repo_root}/methods/motive" \
    "${python_bin}" -m motive.finalize_action_experiment \
    --run-root "${run_root}" \
    --allow-incomplete
  exit 3
fi

for seed in 2026 2027 2028; do
  for artifact in \
    "${run_root}/prep/repr_seed_${seed}/metrics.json" \
    "${run_root}/prep/repr_seed_${seed}/prompt_action_encoder.pt" \
    "${run_root}/prep/repr_seed_${seed}/prompt_action_encoder.pt.json"; do
    if [[ ! -s "${artifact}" ]]; then
      echo "[motive-controller] missing representation artifact: ${artifact}" >&2
      PYTHONPATH="${repo_root}/methods/motive" \
        "${python_bin}" -m motive.finalize_action_experiment \
        --run-root "${run_root}" \
        --allow-incomplete
      exit 3
    fi
  done
done

lucy_job_id="$(
  sbatch \
    --parsable \
    --job-name="motive-lucy-${run_id}" \
    --output="${run_root}/logs/lucy-%A_%a.out" \
    --error="${run_root}/logs/lucy-%A_%a.err" \
    "${repo_root}/methods/motive/scripts/auh_lucy_action_screen.sbatch"
)"
lucy_job_id="${lucy_job_id%%;*}"
if ! [[ "${lucy_job_id}" =~ ^[0-9]+$ ]]; then
  echo "[motive-controller] invalid Lucy job id: ${lucy_job_id}" >&2
  exit 4
fi
export MOTIVE_LUCY_JOB_ID="${lucy_job_id}"
export MOTIVE_CONTROLLER_JOB_ID="${SLURM_JOB_ID:-}"

finalizer_job_id="$(
  sbatch \
    --parsable \
    --dependency="afterany:${lucy_job_id}" \
    --job-name="motive-final-${run_id}" \
    --output="${run_root}/logs/finalizer-%j.out" \
    --error="${run_root}/logs/finalizer-%j.err" \
    "${repo_root}/methods/motive/scripts/auh_action_experiment_finalize.sbatch"
)"
finalizer_job_id="${finalizer_job_id%%;*}"
if ! [[ "${finalizer_job_id}" =~ ^[0-9]+$ ]]; then
  echo "[motive-controller] invalid finalizer job id: ${finalizer_job_id}" >&2
  exit 4
fi

temporary="${submission_path}.tmp.$$"
jq \
  --arg controller_job_id "${SLURM_JOB_ID:-}" \
  --arg lucy_array_job_id "${lucy_job_id}" \
  --arg finalizer_job_id "${finalizer_job_id}" \
  --arg updated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '. + {
    controller_job_id: $controller_job_id,
    lucy_array_job_id: $lucy_array_job_id,
    finalizer_job_id: $finalizer_job_id,
    updated_at_utc: $updated_at_utc
  }' "${submission_path}" > "${temporary}"
mv "${temporary}" "${submission_path}"

echo "[motive-controller] submitted Lucy array=${lucy_job_id} finalizer=${finalizer_job_id}"
