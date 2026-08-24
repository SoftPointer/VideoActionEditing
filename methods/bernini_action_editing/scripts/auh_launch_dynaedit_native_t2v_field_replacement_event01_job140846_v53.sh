#!/usr/bin/env bash
set -euo pipefail

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
dev="$stage/anchor_qk_dev_v1"
launcher="$dev/auh_dynaedit_native_t2v_field_replacement_event01_job140846_v53.sh"
test -x "$launcher"

for node in 246 247; do
  srun --jobid=140846 --overlap --exact --nodes=1 --ntasks=1 \
    --nodelist="auh7-1b-gpu-$node" --cpus-per-task=32 --mem=40G \
    --gres=gpu:mi210:4 \
    "$launcher" \
    >"$stage/dynaedit_native_t2v_field_replacement_event01_v53_${node}.log" 2>&1 &
done
wait
