#!/usr/bin/env bash
set -euo pipefail

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_repr_target_selfgen_20260824_v1
launcher="$root/source/auh_launch_mev_target_selfgen_flow_calibration_20260824_serial_r3.sh"
infer_root="$root/inference_serial_r3"
controller_logs="$root/logs/controllers"

fit_canary_complete="$root/INFERENCE_SERIAL_r3_fit_canary_COMPLETE"
held_canary_complete="$root/INFERENCE_SERIAL_r3_heldout_canary_COMPLETE"
fit_canary_failed="$root/INFERENCE_SERIAL_r3_fit_canary_FAILED"
held_canary_failed="$root/INFERENCE_SERIAL_r3_heldout_canary_FAILED"

test -x "$launcher"
test -d "$infer_root"
mkdir -p "$controller_logs"
test ! -e "$root/INFERENCE_SERIAL_r3_fit_rest_COMPLETE"
test ! -e "$root/INFERENCE_SERIAL_r3_fit_rest_FAILED"
test ! -e "$root/INFERENCE_SERIAL_r3_heldout_rest_COMPLETE"
test ! -e "$root/INFERENCE_SERIAL_r3_heldout_rest_FAILED"
test ! -e "$root/INFERENCE_SERIAL_r3_ALL_COMPLETE"

attempts=0
while [ ! -f "$fit_canary_complete" ] || [ ! -f "$held_canary_complete" ]; do
  if [ -e "$fit_canary_failed" ] || [ -e "$held_canary_failed" ]; then
    echo "a canary failed; remaining cases are not authorized" >&2
    exit 1
  fi
  attempts=$((attempts + 1))
  if [ "$attempts" -gt 720 ]; then
    echo "timed out waiting for both r3 canaries" >&2
    exit 1
  fi
  sleep 10
done

srun \
  --jobid=147881 \
  --exclusive \
  --exact \
  --nodes=1 \
  --ntasks=1 \
  --nodelist=auh7-1b-gpu-213 \
  --cpus-per-task=32 \
  --gres=gpu:mi210:4 \
  --mem=0 \
  env \
  INFER_ROOT_OVERRIDE="$infer_root" \
  SERIAL_REVISION=r3 \
  SERIAL_BATCH_TAG=fit_rest \
  SERIAL_CASE_SET=fb7725c351ba,40712e1341dc,8b05aaf463db \
  bash "$launcher" infer-fit-serial \
  >"$controller_logs/infer-fit-rest-serial-r3-job147881.log" 2>&1 & fit_pid=$!

srun \
  --jobid=147871 \
  --exclusive \
  --exact \
  --nodes=1 \
  --ntasks=1 \
  --nodelist=auh7-1b-gpu-232 \
  --cpus-per-task=32 \
  --gres=gpu:mi210:4 \
  --mem=0 \
  env \
  INFER_ROOT_OVERRIDE="$infer_root" \
  SERIAL_REVISION=r3 \
  SERIAL_BATCH_TAG=heldout_rest \
  SERIAL_CASE_SET=81533c9e56ec,5e83a9279951,840b214afead \
  bash "$launcher" infer-heldout-serial \
  >"$controller_logs/infer-heldout-rest-serial-r3-job147871.log" 2>&1 & held_pid=$!

status=0
if ! wait "$fit_pid"; then status=1; fi
if ! wait "$held_pid"; then status=1; fi
if [ "$status" -ne 0 ]; then
  echo "one or both remaining-case steps failed" >&2
  exit 1
fi

test -f "$root/INFERENCE_SERIAL_r3_fit_rest_COMPLETE"
test -f "$root/INFERENCE_SERIAL_r3_heldout_rest_COMPLETE"
test "$(find "$infer_root" -type f -name output.mp4 | wc -l | tr -d ' ')" = 32
test "$(find "$infer_root" -type f -name calibration_receipt.json | wc -l | tr -d ' ')" = 32
touch "$root/INFERENCE_SERIAL_r3_ALL_COMPLETE"

