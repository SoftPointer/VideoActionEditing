#!/usr/bin/env bash
#SBATCH --job-name=motive-action-repr-pilot
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# AUH faculty QOS requires one MI210 even though this stage runs on CPU.
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=00:30:00

set -Eeuo pipefail

repo_root="${MOTIVE_REPO_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit}"
code_root="${MOTIVE_CODE_ROOT:-${repo_root}/methods/motive}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
source_run="${MOTIVE_SOURCE_RUN:-${repo_root}/methods/motive/runs/goku_action_rules_v2/pilot2k/qwen_full2k_calibration_v6}"
run_root="${MOTIVE_ACTION_RUN_ROOT:-}"
source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:-}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:-}"
if [[ -z "${run_root}" || -z "${source_snapshot}" || -z "${source_tree_sha256}" ]]; then
  echo "[motive-action-repr] MOTIVE_ACTION_RUN_ROOT, MOTIVE_SOURCE_SNAPSHOT, and MOTIVE_SOURCE_TREE_SHA256 are required" >&2
  exit 2
fi
experiment_root="${MOTIVE_ACTION_EXPERIMENT_ROOT:-${run_root}/prep}"
source_manifest="${source_run}/fused/all.jsonl"
feature_dir="${source_run}/feature_snapshot"

for required in \
  "${code_root}/motive/experiment_prep.py" \
  "${code_root}/motive/train_action_repr.py" \
  "${python_bin}" \
  "${source_manifest}" \
  "${feature_dir}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[motive-action-repr] missing required path: ${required}" >&2
    exit 2
  fi
done
if [[ -e "${experiment_root}" ]]; then
  echo "[motive-action-repr] refusing to reuse experiment root: ${experiment_root}" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "[motive-action-repr] jq is required for atomic run status records" >&2
  exit 2
fi
mkdir -p "${run_root}/status"
status_path="${run_root}/status/representation.json"
if [[ -e "${status_path}" ]]; then
  echo "[motive-action-repr] refusing to reuse status path: ${status_path}" >&2
  exit 2
fi
started_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
status_tmp="${status_path}.tmp.$$"
jq -n \
  --arg schema "motive-action-representation-status-v1" \
  --arg state "running" \
  --arg started_at_utc "${started_at_utc}" \
  --arg hostname "$(hostname)" \
  --arg slurm_job_id "${SLURM_JOB_ID:-}" \
  --arg source_tree_sha256 "${source_tree_sha256}" \
  '{
    schema: $schema,
    state: $state,
    started_at_utc: $started_at_utc,
    hostname: $hostname,
    slurm_job_id: $slurm_job_id,
    source_tree_sha256: $source_tree_sha256,
    exit_code: null
  }' > "${status_tmp}"
mv "${status_tmp}" "${status_path}"

on_exit() {
  exit_code=$?
  trap - EXIT
  state="failed"
  if [[ "${exit_code}" -eq 0 ]]; then
    state="succeeded"
  fi
  status_tmp="${status_path}.tmp.$$"
  jq -n \
    --arg schema "motive-action-representation-status-v1" \
    --arg state "${state}" \
    --arg started_at_utc "${started_at_utc}" \
    --arg ended_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg hostname "$(hostname)" \
    --arg slurm_job_id "${SLURM_JOB_ID:-}" \
    --arg source_tree_sha256 "${source_tree_sha256}" \
    --argjson exit_code "${exit_code}" \
    '{
      schema: $schema,
      state: $state,
      started_at_utc: $started_at_utc,
      ended_at_utc: $ended_at_utc,
      hostname: $hostname,
      slurm_job_id: $slurm_job_id,
      source_tree_sha256: $source_tree_sha256,
      exit_code: $exit_code
    }' > "${status_tmp}"
  mv "${status_tmp}" "${status_path}"
  exit "${exit_code}"
}
trap on_exit EXIT

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="${code_root}:${PYTHONPATH:-}"

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

"${python_bin}" -m motive.experiment_prep \
  legacy-qwen-original-valid-action-v1 \
  --input "${source_manifest}" \
  --output-dir "${experiment_root}" \
  --max-lucy-train 64 \
  --seed 260108828

for seed in 2026 2027 2028; do
  "${python_bin}" -m motive.train_action_repr \
    --feature-dir "${feature_dir}" \
    --manifest "${experiment_root}/representation_manifest.jsonl" \
    --output-dir "${experiment_root}/repr_seed_${seed}" \
    --decisions review \
    --allow-unreviewed-pseudo-labels \
    --allow-non-content-splits \
    --text-feature-dim 512 \
    --action-dim 128 \
    --camera-dims 8 \
    --epochs 30 \
    --batch-size 32 \
    --seed "${seed}" \
    --device cpu
done

for seed in 2026 2027 2028; do
  for artifact in \
    "${experiment_root}/repr_seed_${seed}/metrics.json" \
    "${experiment_root}/repr_seed_${seed}/prompt_action_encoder.pt" \
    "${experiment_root}/repr_seed_${seed}/prompt_action_encoder.pt.json"; do
    if [[ ! -s "${artifact}" ]]; then
      echo "[motive-action-repr] missing final artifact: ${artifact}" >&2
      exit 2
    fi
  done
done

echo "[motive-action-repr] completed pseudo-label interface pilot: ${experiment_root}"
