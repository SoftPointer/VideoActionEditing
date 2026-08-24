#!/usr/bin/env bash
set -euo pipefail

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
runner="$stage/anchor_qk_dev_v1/auh_dynaedit_native_observer_reward_event01_job140846_v67.sh"

launch() {
  local node=$1
  local mode=$2
  srun --jobid=140846 --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
    --gres=gpu:4 --cpus-per-task=32 --mem=0 \
    bash "$runner" "$mode" \
    >"$stage/dynaedit_native_observer_reward_event01_v67_${node}_${mode}.log" 2>&1 &
}

launch auh7-1b-gpu-246 action
launch auh7-1b-gpu-247 envelope
wait
