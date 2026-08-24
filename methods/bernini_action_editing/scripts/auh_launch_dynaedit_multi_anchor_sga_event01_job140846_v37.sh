#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then echo "usage: $0" >&2; exit 2; fi
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
payload="$stage/anchor_qk_dev_v1/auh_dynaedit_multi_anchor_sga_event01_job140846_v37.sh"
output_root="$stage/dynaedit_multi_anchor_sga_event01_v37"
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

launch auh7-1b-gpu-246 0 sga_bg_a010
pid_sga_bg_a010=$launched_pid
launch auh7-1b-gpu-246 1 avg_a010
pid_avg_a010=$launched_pid
launch auh7-1b-gpu-247 0 sga_global_a010
pid_sga_global_a010=$launched_pid
launch auh7-1b-gpu-247 1 sga_bg_a025
pid_sga_bg_a025=$launched_pid

status=0
for pid in "$pid_sga_bg_a010" "$pid_avg_a010" "$pid_sga_global_a010" "$pid_sga_bg_a025"; do
  wait "$pid" || status=$?
done
exit "$status"
