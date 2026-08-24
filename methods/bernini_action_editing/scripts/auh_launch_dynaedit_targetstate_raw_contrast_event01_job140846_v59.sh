#!/usr/bin/env bash
set -euo pipefail

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
dev="$stage/anchor_qk_dev_v1"
runner="$dev/auh_dynaedit_targetstate_raw_contrast_event01_job140846_v59.sh"
test -x "$runner"

launch() {
  local node=$1
  local mode=$2
  # AUH cgroups expose only the GPUs requested by a child.  Request the full
  # parent-node allocation, then let the runner bind one disjoint SP4 island.
  # --overlap is intentional: group 0 and group 1 share no physical GPU.
  srun --jobid=140846 --overlap --exact --nodes=1 --ntasks=1 \
    --nodelist="auh7-1b-gpu-$node" --cpus-per-task=64 --mem=64G \
    --gres=gpu:mi210:8 \
    "$runner" "$mode" \
    >"$stage/dynaedit_targetstate_raw_contrast_event01_v59_${node}_${mode}.log" 2>&1 &
}

launch 246 raw_full
launch 246 native_full
launch 247 raw_sparse25
launch 247 native_sparse25
wait
