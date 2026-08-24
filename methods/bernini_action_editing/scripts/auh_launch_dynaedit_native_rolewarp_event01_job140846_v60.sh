#!/usr/bin/env bash
set -euo pipefail

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
runner="$stage/anchor_qk_dev_v1/auh_dynaedit_native_rolewarp_event01_job140846_v60.sh"
test -x "$runner"

launch() {
  local node=$1
  local mode=$2
  srun --jobid=140846 --overlap --exact -w "auh7-1b-gpu-$node" -N1 -n1 -c64 \
    --mem=64G --gres=gpu:mi210:8 "$runner" "$mode" \
    >"$stage/dynaedit_native_rolewarp_event01_v60_${node}_${mode}.log" 2>&1 &
}

launch 246 sga_full
launch 246 avg_full
launch 247 sga_sparse25
launch 247 avg_sparse25
wait
