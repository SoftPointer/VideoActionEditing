#!/usr/bin/env bash
set -euo pipefail

job_id="${ROUND81_JOB_ID:-143828}"
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
runner="$stage/anchor_qk_sourcepatch_v1/auh_dynaedit_native_source_patch_move_event01_v81.sh"
output_root="$stage/dynaedit_source_patch_move_event01_v81"

test -x "$runner"
mkdir -p "$output_root/logs"

launch_node() {
  local node=$1
  local log="$output_root/logs/${node}.log"
  srun --jobid="$job_id" --overlap --nodes=1 --ntasks=1 \
    --nodelist="$node" --gres=gpu:8 --cpus-per-task=64 --mem=0 \
    bash -lc "'$runner' 0 && '$runner' 1" >"$log" 2>&1 &
}

# One 4-rank model at a time per host.  Each completed process releases its
# roughly 32.5-GiB host footprint before the second GPU group starts.
launch_node auh7-1b-gpu-246
launch_node auh7-1b-gpu-247
wait

for expected in \
  NATIVE_SOURCEPATCH_MOVE_A100_ALL40_B4_9_ACTOROBJ_P24_R8 \
  NATIVE_SOURCEPATCH_MOVE_A050_ALL40_B4_9_ACTOROBJ_P24_R8 \
  NATIVE_SOURCEPATCH_MOVE_A100_ALL40_B8_13_ACTOROBJ_P24_R8 \
  NATIVE_SOURCEPATCH_MOVE_A050_ALL40_B8_13_ACTOROBJ_P24_R8
do
  test -s "$output_root/$expected.mp4"
  test -s "$output_root/$expected.mp4.receipt.json"
done
printf 'complete\n' > "$output_root/SWEEP_COMPLETE"
