#!/usr/bin/env bash
set -euo pipefail

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
runner="$stage/anchor_qk_symmetry_v1/auh_dynaedit_final_projection_event01_v91.sh"
output_root="$stage/dynaedit_final_projection_event01_v91"
test -x "$runner"
mkdir -p "$output_root/logs"

ssh auh7-1b-gpu-246 "'$runner' 10 015" >"$output_root/logs/246_k10_rf015.log" 2>&1 &
ssh auh7-1b-gpu-246 "'$runner' 10 020" >"$output_root/logs/246_k10_rf020.log" 2>&1 &
ssh auh7-1b-gpu-247 "'$runner' 20 015" >"$output_root/logs/247_k20_rf015.log" 2>&1 &
ssh auh7-1b-gpu-247 "'$runner' 20 020" >"$output_root/logs/247_k20_rf020.log" 2>&1 &
wait

for expected in \
  NATIVE_OBSERVER_FINALSTRONG_K10_RF015 \
  NATIVE_OBSERVER_FINALSTRONG_K10_RF020 \
  NATIVE_OBSERVER_FINALSTRONG_K20_RF015 \
  NATIVE_OBSERVER_FINALSTRONG_K20_RF020
do
  test -s "$output_root/$expected.mp4"
  test -s "$output_root/$expected.mp4.receipt.json"
done
printf 'complete\n' > "$output_root/SWEEP_COMPLETE"
