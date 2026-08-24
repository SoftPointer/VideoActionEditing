#!/usr/bin/env bash
set -euo pipefail

job_id="${ROUND89_JOB_ID:-143828}"
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
runner="$stage/anchor_qk_symmetry_v1/auh_dynaedit_complex4_observer_p39_job143828_v89.sh"
output_root="$stage/dynaedit_complex4_observer_p39_event_v89"

test -x "$runner"
mkdir -p "$output_root/logs"

launch_event() {
  local node=$1
  local event=$2
  local log="$output_root/logs/${node}_e${event}.log"
  srun --jobid="$job_id" --overlap --nodes=1 --ntasks=1 \
    --nodelist="$node" --gres=gpu:4 --cpus-per-task=32 --mem=64G \
    bash -lc "'$runner' '$event'" >"$log" 2>&1 &
}

# Two independent four-GPU probes per node fill all eight allocated GPUs while
# preserving the already-audited four-rank inference topology.
launch_event auh7-1b-gpu-246 0
launch_event auh7-1b-gpu-246 4
launch_event auh7-1b-gpu-247 2
launch_event auh7-1b-gpu-247 7
wait

for event in 0 2 4 7; do
  test "$(find "$output_root" -maxdepth 1 -type f -name "COMPLEX4_E${event}_*.mp4" | wc -l)" -eq 1
  test "$(find "$output_root" -maxdepth 1 -type f -name "COMPLEX4_E${event}_*.mp4.receipt.json" | wc -l)" -eq 1
done
printf 'complete\n' > "$output_root/SWEEP_COMPLETE"
