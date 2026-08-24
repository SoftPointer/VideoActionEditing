#!/usr/bin/env bash
set -euo pipefail

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
payload="$stage/anchor_qk_dev_v1/auh_dynaedit_native_t2v_trajectory_event01_job140846_v48.sh"
test -x "$payload"

srun --jobid=140846 --overlap --exact -w auh7-1b-gpu-246 -N1 -n1 -c32 \
  --mem=40G --gres=gpu:mi210:4 "$payload" 1 \
  >"$stage/dynaedit_native_t2v_trajectory_event01_v48_246_1.log" 2>&1 &
left=$!
srun --jobid=140846 --overlap --exact -w auh7-1b-gpu-247 -N1 -n1 -c32 \
  --mem=40G --gres=gpu:mi210:4 "$payload" 0 \
  >"$stage/dynaedit_native_t2v_trajectory_event01_v48_247_0.log" 2>&1 &
right=$!
status=0
wait "$left" || status=$?
wait "$right" || status=$?
exit "$status"
