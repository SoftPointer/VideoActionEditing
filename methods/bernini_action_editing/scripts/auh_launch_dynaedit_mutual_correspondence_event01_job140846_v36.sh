#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then echo "usage: $0" >&2; exit 2; fi
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
payload="$stage/anchor_qk_dev_v1/auh_dynaedit_mutual_correspondence_event01_job140846_v36.sh"
output_root="$stage/dynaedit_mutual_correspondence_event01_v36"
test -x "$payload"
mkdir -p "$output_root"

launch() {
  node="$1"
  group="$2"
  label="$3"
  srun --overlap --exact --jobid=140846 \
    -w "$node" -N1 -n1 -c64 --mem=64G \
    --gres=gpu:mi210:8 \
    bash "$payload" "$group" \
    >"$output_root/launch_${label}.log" 2>&1 &
  launched_pid=$!
}

launch auh7-1b-gpu-246 0 qk_b4_9
pid_qk_b4_9=$launched_pid
launch auh7-1b-gpu-246 1 attn_b4_9
pid_attn_b4_9=$launched_pid
launch auh7-1b-gpu-247 0 qk_b10_15
pid_qk_b10_15=$launched_pid
launch auh7-1b-gpu-247 1 attn_b10_15
pid_attn_b10_15=$launched_pid

status=0
for pid in "$pid_qk_b4_9" "$pid_attn_b4_9" "$pid_qk_b10_15" "$pid_attn_b10_15"; do
  wait "$pid" || status=$?
done
exit "$status"
