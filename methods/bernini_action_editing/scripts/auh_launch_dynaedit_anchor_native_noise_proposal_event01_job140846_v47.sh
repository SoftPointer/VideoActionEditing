#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then echo "usage: $0" >&2; exit 2; fi
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
payload="$stage/anchor_qk_dev_v1/auh_dynaedit_anchor_native_noise_proposal_event01_job140846_v47.sh"
root="$stage/dynaedit_anchor_native_noise_proposal_event01_v47"
test -x "$payload"
mkdir -p "$root"

srun --overlap --exact --jobid=140846 -w auh7-1b-gpu-246 -N1 -n1 -c64 \
  --mem=64G --gres=gpu:mi210:8 bash "$payload" 0 \
  >"$root/launch_sga.log" 2>&1 &
left=$!
srun --overlap --exact --jobid=140846 -w auh7-1b-gpu-247 -N1 -n1 -c64 \
  --mem=64G --gres=gpu:mi210:8 bash "$payload" 0 \
  >"$root/launch_forced.log" 2>&1 &
right=$!
status=0
wait "$left" || status=$?
wait "$right" || status=$?
exit "$status"
