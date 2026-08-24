#!/usr/bin/env bash

set -Eeuo pipefail

inference_root="${MOTIVE_INFERENCE_ROOT:?MOTIVE_INFERENCE_ROOT is required}"
code_root="${MOTIVE_CODE_ROOT:?MOTIVE_CODE_ROOT is required}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?MOTIVE_SOURCE_TREE_SHA256 is required}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
submission_path="${inference_root}/submission.json"

for required in \
  "${inference_root}/contract.json" \
  "${code_root}/methods/motive/scripts/auh_action_inference.sbatch" \
  "${code_root}/methods/motive/scripts/auh_action_inference_metrics.sbatch" \
  "${code_root}/methods/motive/scripts/auh_action_inference_qwen.sbatch" \
  "${code_root}/methods/motive/scripts/auh_action_inference_finalize.sbatch"; do
  if [[ ! -s "${required}" ]]; then
    echo "[motive-infer-submit] missing required artifact: ${required}" >&2
    exit 2
  fi
done
if [[ -e "${submission_path}" ]]; then
  echo "[motive-infer-submit] refusing to overwrite ${submission_path}" >&2
  exit 2
fi

"${python_bin}" \
  "${code_root}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${code_root}" \
  --expected-tree-sha256 "${source_tree_sha256}"

mkdir -p "${inference_root}/slurm"
common_export="ALL,MOTIVE_INFERENCE_ROOT=${inference_root},MOTIVE_CODE_ROOT=${code_root},MOTIVE_SOURCE_TREE_SHA256=${source_tree_sha256},PYTHON_BIN=${python_bin}"
inference_job="$(
  sbatch \
    --parsable \
    --array=0-3%2 \
    --output="${inference_root}/slurm/%x_%A_%a.out" \
    --export="${common_export}" \
    "${code_root}/methods/motive/scripts/auh_action_inference.sbatch"
)"
metrics_job="$(
  sbatch \
    --parsable \
    --dependency="afterok:${inference_job}" \
    --output="${inference_root}/slurm/%x_%j.out" \
    --export="${common_export}" \
    "${code_root}/methods/motive/scripts/auh_action_inference_metrics.sbatch"
)"
qwen_job="$(
  sbatch \
    --parsable \
    --dependency="afterok:${metrics_job}" \
    --output="${inference_root}/slurm/%x_%j.out" \
    --export="${common_export}" \
    "${code_root}/methods/motive/scripts/auh_action_inference_qwen.sbatch"
)"
finalize_job="$(
  sbatch \
    --parsable \
    --dependency="afterok:${qwen_job}" \
    --output="${inference_root}/slurm/%x_%j.out" \
    --export="${common_export}" \
    "${code_root}/methods/motive/scripts/auh_action_inference_finalize.sbatch"
)"

temporary="${submission_path}.tmp.$$"
jq -n \
  --arg schema "motive-action-inference-submission-v1" \
  --arg submitted_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg inference_job "${inference_job}" \
  --arg metrics_job "${metrics_job}" \
  --arg qwen_job "${qwen_job}" \
  --arg finalize_job "${finalize_job}" \
  --arg source_tree_sha256 "${source_tree_sha256}" \
  '{
    schema: $schema,
    submitted_at_utc: $submitted_at_utc,
    inference_job: $inference_job,
    metrics_job: $metrics_job,
    qwen_job: $qwen_job,
    finalize_job: $finalize_job,
    source_tree_sha256: $source_tree_sha256,
    max_concurrent_nodes: 2,
    inference_gpus_per_node: 8,
    qwen_gpus_per_node: 8
  }' > "${temporary}"
mv "${temporary}" "${submission_path}"

echo "inference_job=${inference_job}"
echo "metrics_job=${metrics_job}"
echo "qwen_job=${qwen_job}"
echo "finalize_job=${finalize_job}"
