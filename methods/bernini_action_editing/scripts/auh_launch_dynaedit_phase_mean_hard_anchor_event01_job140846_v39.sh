#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then echo "usage: $0" >&2; exit 2; fi
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
payload="$stage/anchor_qk_dev_v1/auh_dynaedit_phase_mean_hard_anchor_event01_job140846_v39.sh"
output_root="$stage/dynaedit_phase_mean_hard_anchor_event01_v39"
test -x "$payload"
mkdir -p "$output_root"

launch_pair() {
  group="$1"
  left_label="$2"
  right_label="$3"
  srun --overlap --exact --jobid=140846 -w auh7-1b-gpu-246 -N1 -n1 -c64 \
    --mem=64G --gres=gpu:mi210:8 bash "$payload" "$group" \
    >"$output_root/launch_${left_label}.log" 2>&1 &
  left_pid=$!
  srun --overlap --exact --jobid=140846 -w auh7-1b-gpu-247 -N1 -n1 -c64 \
    --mem=64G --gres=gpu:mi210:8 bash "$payload" "$group" \
    >"$output_root/launch_${right_label}.log" 2>&1 &
  right_pid=$!
  status=0
  wait "$left_pid" || status=$?
  wait "$right_pid" || status=$?
  return "$status"
}

# One SP4 arm per node at a time: two such models exceed the node's host RAM.
launch_pair 0 qk_b0_5 qk_b4_9
launch_pair 1 attn_b0_5 attn_b4_9
