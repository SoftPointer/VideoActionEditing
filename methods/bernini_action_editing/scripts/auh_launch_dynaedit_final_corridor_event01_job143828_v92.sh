#!/usr/bin/env bash
set -euo pipefail

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
runner="$stage/anchor_qk_symmetry_v1/auh_dynaedit_final_corridor_event01_v92.sh"
output_root="$stage/dynaedit_final_corridor_event01_v92"
test -x "$runner"
mkdir -p "$output_root/logs"

ssh auh7-1b-gpu-246 "'$runner' observer 000" >"$output_root/logs/246_observer_id000.log" 2>&1 &
ssh auh7-1b-gpu-246 "'$runner' observer 025" >"$output_root/logs/246_observer_id025.log" 2>&1 &
ssh auh7-1b-gpu-247 "'$runner' explicit 000" >"$output_root/logs/247_explicit_id000.log" 2>&1 &
ssh auh7-1b-gpu-247 "'$runner' explicit 025" >"$output_root/logs/247_explicit_id025.log" 2>&1 &
wait

for expected in \
  NATIVE_FINALCORRIDOR_OBSERVER_ID000 \
  NATIVE_FINALCORRIDOR_OBSERVER_ID025 \
  NATIVE_FINALCORRIDOR_EXPLICIT_ID000 \
  NATIVE_FINALCORRIDOR_EXPLICIT_ID025
do
  test -s "$output_root/$expected.mp4"
  test -s "$output_root/$expected.mp4.receipt.json"
done
printf 'complete\n' > "$output_root/SWEEP_COMPLETE"
