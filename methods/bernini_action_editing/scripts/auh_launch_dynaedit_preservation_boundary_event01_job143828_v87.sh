#!/usr/bin/env bash
set -euo pipefail

job_id="${ROUND87_JOB_ID:-143828}"
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
runner="$stage/anchor_qk_symmetry_v1/auh_dynaedit_preservation_boundary_event01_v87.sh"
output_root="$stage/dynaedit_preservation_boundary_event01_v87"

test -x "$runner"
mkdir -p "$output_root/logs"

launch_node() {
  local node=$1
  local mode=$2
  local log="$output_root/logs/${node}.log"
  srun --jobid="$job_id" --overlap --nodes=1 --ntasks=1 \
    --nodelist="$node" --gres=gpu:8 --cpus-per-task=64 --mem=0 \
    bash -lc "'$runner' '$mode' 32 && '$runner' '$mode' 36" >"$log" 2>&1 &
}

launch_node auh7-1b-gpu-246 observer
launch_node auh7-1b-gpu-247 explicit
wait

for expected in \
  NATIVE_OBSERVER_ENVELOPE_B4_9_P32_R8 \
  NATIVE_OBSERVER_ENVELOPE_B4_9_P36_R8 \
  NATIVE_EXPLSRC_ANCHORREL_PATCHV_A100_ALL40_B4_9_P32_R8 \
  NATIVE_EXPLSRC_ANCHORREL_PATCHV_A100_ALL40_B4_9_P36_R8
do
  test -s "$output_root/$expected.mp4"
  test -s "$output_root/$expected.mp4.receipt.json"
done
printf 'complete\n' > "$output_root/SWEEP_COMPLETE"
