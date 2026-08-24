#!/usr/bin/env bash
set -euo pipefail

# Run this script from the AUH checkout. It is intentionally read-only with
# respect to source videos; only manifests/descriptors are written below
# methods/motive/outputs.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
data_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/data/goku/subject_movement/extracted"
output_root="${1:-${repo_root}/methods/motive/outputs/goku_subject_movement_sample}"
sample_limit="${MOTIVE_SAMPLE_LIMIT:-500}"
worker_count="${MOTIVE_WORKERS:-8}"
python_bin="${PYTHON_BIN:-python}"

cd "${repo_root}"
mkdir -p "${output_root}"
PYTHONPATH="${repo_root}/methods/motive:${PYTHONPATH:-}" \
  "${python_bin}" -m motive.goku_manifest \
  --dataset-root "${data_root}" \
  --output "${output_root}/sample_manifest.jsonl" \
  --sample-size "${sample_limit}" \
  --seed 260108828 \
  --semantic-classes continuous_action motion_suppression

PYTHONPATH="${repo_root}/methods/motive:${PYTHONPATH:-}" \
  "${python_bin}" -m motive.audit \
  --input "${output_root}/sample_manifest.jsonl" \
  --root "${data_root}" \
  --output-dir "${output_root}/audit" \
  --source-key src_video \
  --target-key tgt_video \
  --workers "${worker_count}" \
  --analysis-frames 32 \
  --resize-width 256 \
  --semantic-classes continuous_action motion_suppression \
  --min-descriptor-delta 0.35 \
  --min-action-residual-p90 0.005 \
  --min-suppression-residual-p90 0.003 \
  --min-suppression-motion-ratio 1.10
