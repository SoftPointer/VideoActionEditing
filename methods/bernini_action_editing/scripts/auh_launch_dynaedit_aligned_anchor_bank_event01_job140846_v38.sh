#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then echo "usage: $0" >&2; exit 2; fi
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
payload="$stage/anchor_qk_dev_v1/auh_dynaedit_aligned_anchor_bank_event01_job140846_v38.sh"
output_root="$stage/dynaedit_aligned_anchor_bank_event01_v38"
test -x "$payload"
mkdir -p "$output_root"

srun --overlap --exact --jobid=140846 \
  -w auh7-1b-gpu-246 -N1 -n1 -c64 --mem=64G --gres=gpu:mi210:8 \
  bash "$payload" 0 >"$output_root/launch_aligned_a010.log" 2>&1 &
pid_a010=$!
srun --overlap --exact --jobid=140846 \
  -w auh7-1b-gpu-247 -N1 -n1 -c64 --mem=64G --gres=gpu:mi210:8 \
  bash "$payload" 0 >"$output_root/launch_aligned_a025.log" 2>&1 &
pid_a025=$!

status=0
wait "$pid_a010" || status=$?
wait "$pid_a025" || status=$?
exit "$status"
