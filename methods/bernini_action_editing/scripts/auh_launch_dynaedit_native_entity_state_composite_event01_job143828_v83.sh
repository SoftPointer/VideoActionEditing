#!/usr/bin/env bash
set -euo pipefail

job_id="${ROUND83_JOB_ID:-143828}"
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
runner="$stage/anchor_qk_sourcepatch_v1/auh_dynaedit_native_entity_state_composite_event01_v83.sh"
output_root="$stage/dynaedit_entity_state_composite_event01_v83"

test -x "$runner"
mkdir -p "$output_root/logs"

launch_node() {
  local node=$1
  local log="$output_root/logs/${node}.log"
  srun --jobid="$job_id" --overlap --nodes=1 --ntasks=1 \
    --nodelist="$node" --gres=gpu:8 --cpus-per-task=64 --mem=0 \
    bash -lc "'$runner' 0 && '$runner' 1" >"$log" 2>&1 &
}

launch_node auh7-1b-gpu-246
launch_node auh7-1b-gpu-247
wait

for expected in \
  NATIVE_ENTITYSTATE_ID025_PATCHV_A100_ALL40_B4_9_P24_R8 \
  NATIVE_ENTITYSTATE_ID075_PATCHV_A100_ALL40_B4_9_P24_R8 \
  NATIVE_ENTITYSTATE_ID025_PATCHV_A100_ALL40_B8_13_P24_R8 \
  NATIVE_ENTITYSTATE_ID075_PATCHV_A100_ALL40_B8_13_P24_R8
do
  test -s "$output_root/$expected.mp4"
  test -s "$output_root/$expected.mp4.receipt.json"
done
printf 'complete\n' > "$output_root/SWEEP_COMPLETE"
