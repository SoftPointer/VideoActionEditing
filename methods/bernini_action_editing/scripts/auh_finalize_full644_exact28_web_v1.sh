#!/usr/bin/env bash
set -Eeuo pipefail
root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_self_generated_anchor_v1_20260818
eval_root="${root}/evaluation/exact28_matched_seed3_v1"
release="${root}/release_2ad469e5_9ea76025"
main_done="${eval_root}/PARTIAL_FROZEN_SEED20260820_COMPLETE"
replicas_done="${eval_root}/PARTIAL_SEED20260821_SEED20260822_COMPLETE"
while [[ ! -f "${main_done}" || ! -f "${replicas_done}" ]]; do sleep 30; done

generated="$(find "${eval_root}/media" -type f -name '*.mp4' | wc -l | tr -d ' ')"
receipts="$(find "${eval_root}/media" -type f -name '*.mp4.receipt.json' | wc -l | tr -d ' ')"
[[ "${generated}" == 112 && "${receipts}" == 112 ]]
find "${eval_root}/media" -type f \( -name '*.mp4' -o -name '*.mp4.receipt.json' \) -exec chmod 0444 {} +

python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
"${python_bin}" "${release}/evaluate_and_build_full644_exact28_web_v1.py" \
  --selection "${root}/data/exact28_eval_selection_v1.json" \
  --eval-root "${eval_root}" \
  --output-html "${eval_root}/index.html" \
  --output-metrics "${eval_root}/metrics.json" \
  >"${eval_root}/web_build_receipt.json"
chmod 0444 "${eval_root}/index.html" "${eval_root}/metrics.json" "${eval_root}/web_build_receipt.json"
printf 'training_coverage=644\nevaluation_rows=28\naction_families=28\ngenerated_videos=112\nweb_media_videos=168\nhtml_sha256=%s\nmetrics_sha256=%s\n' \
  "$(sha256sum "${eval_root}/index.html" | awk '{print $1}')" \
  "$(sha256sum "${eval_root}/metrics.json" | awk '{print $1}')" \
  >"${eval_root}/WEB_COMPLETE"
chmod 0444 "${eval_root}/WEB_COMPLETE"
