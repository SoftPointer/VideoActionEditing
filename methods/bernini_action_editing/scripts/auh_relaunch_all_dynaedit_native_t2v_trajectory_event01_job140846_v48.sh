#!/usr/bin/env bash
set -euo pipefail

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
payload="$stage/anchor_qk_dev_v1/auh_dynaedit_native_t2v_trajectory_event01_job140846_v48.sh"
test -x "$payload"

pids=()
for node in 246 247; do
  for arm in 0 1; do
    srun --jobid=140846 --overlap --exact -w "auh7-1b-gpu-$node" -N1 -n1 -c32 \
      --mem=40G --gres=gpu:mi210:4 "$payload" "$arm" \
      >"$stage/dynaedit_native_t2v_trajectory_event01_v48_${node}_${arm}.log" 2>&1 &
    pids+=("$!")
  done
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
