#!/usr/bin/env bash
set -euo pipefail

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
runner="$stage/anchor_qk_symmetry_v1/auh_dynaedit_snapshot_residual_event01_v90.sh"
output_root="$stage/dynaedit_snapshot_residual_event01_v90"
test -x "$runner"
mkdir -p "$output_root/logs"

# Direct node shells are required here: nested overlapping Slurm steps expose
# only logical devices 0-3 on these MI210 nodes.  Each runner still validates
# its host and binds an explicit physical four-GPU slot.
ssh auh7-1b-gpu-246 "'$runner' 32 015" >"$output_root/logs/246_p32_rf015.log" 2>&1 &
ssh auh7-1b-gpu-246 "'$runner' 36 015" >"$output_root/logs/246_p36_rf015.log" 2>&1 &
ssh auh7-1b-gpu-247 "'$runner' 32 020" >"$output_root/logs/247_p32_rf020.log" 2>&1 &
ssh auh7-1b-gpu-247 "'$runner' 36 020" >"$output_root/logs/247_p36_rf020.log" 2>&1 &
wait

for expected in \
  NATIVE_OBSERVER_SNAPSHOT_P32_RF015_R8 \
  NATIVE_OBSERVER_SNAPSHOT_P36_RF015_R8 \
  NATIVE_OBSERVER_SNAPSHOT_P32_RF020_R8 \
  NATIVE_OBSERVER_SNAPSHOT_P36_RF020_R8
do
  test -s "$output_root/$expected.mp4"
  test -s "$output_root/$expected.mp4.receipt.json"
done
printf 'complete\n' > "$output_root/SWEEP_COMPLETE"
