#!/usr/bin/env bash
set -euo pipefail

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_repr_target_selfgen_20260824_v1
source_root="$root/source"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
builder="$source_root/build_mev_target_selfgen_flow_calibration_review_v1.py"
manifest="$source_root/experiment_manifest.json"
inference_root="$root/inference_serial_r3"
review_root="$root/review_serial_r3"

test -x "$python_bin"
test -f "$builder"
test -f "$manifest"
test ! -e "$review_root"
test ! -e "$root/REVIEW_SERIAL_R3_COMPLETE"
test ! -e "$root/REVIEW_SERIAL_R3_FAILED"

attempts=0
while [ ! -f "$root/INFERENCE_SERIAL_r3_ALL_COMPLETE" ]; do
  if [ -e "$root/INFERENCE_SERIAL_r3_fit_canary_FAILED" ] \
    || [ -e "$root/INFERENCE_SERIAL_r3_heldout_canary_FAILED" ] \
    || [ -e "$root/INFERENCE_SERIAL_r3_fit_rest_FAILED" ] \
    || [ -e "$root/INFERENCE_SERIAL_r3_heldout_rest_FAILED" ]; then
    echo "an r3 inference batch failed; review publication is blocked" >&2
    touch "$root/REVIEW_SERIAL_R3_FAILED"
    exit 1
  fi
  attempts=$((attempts + 1))
  if [ "$attempts" -gt 3600 ]; then
    echo "timed out waiting for complete r3 inference" >&2
    touch "$root/REVIEW_SERIAL_R3_FAILED"
    exit 1
  fi
  sleep 10
done

if ! "$python_bin" -B "$builder" \
  --manifest "$manifest" \
  --inference-root "$inference_root" \
  --output-dir "$review_root" \
  --copy-mode copy; then
  touch "$root/REVIEW_SERIAL_R3_FAILED"
  exit 1
fi

test -s "$review_root/index.html"
test -s "$review_root/review_manifest.json"
test "$(find "$review_root/media" -type f -name '*.mp4' | wc -l | tr -d ' ')" = 64
touch "$root/REVIEW_SERIAL_R3_COMPLETE"

