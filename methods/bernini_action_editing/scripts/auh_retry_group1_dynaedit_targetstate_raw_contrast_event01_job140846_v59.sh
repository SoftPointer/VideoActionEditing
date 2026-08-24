#!/usr/bin/env bash
set -euo pipefail

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
runner="$stage/anchor_qk_dev_v1/auh_dynaedit_targetstate_raw_contrast_event01_job140846_v59.sh"
test -x "$runner"

srun --jobid=140846 --overlap --exact -w auh7-1b-gpu-246 -N1 -n1 -c64 \
  --mem=64G --gres=gpu:mi210:8 "$runner" native_full \
  >"$stage/dynaedit_targetstate_raw_contrast_event01_v59_246_native_full_r2.log" 2>&1 &
left=$!
srun --jobid=140846 --overlap --exact -w auh7-1b-gpu-247 -N1 -n1 -c64 \
  --mem=64G --gres=gpu:mi210:8 "$runner" native_sparse25 \
  >"$stage/dynaedit_targetstate_raw_contrast_event01_v59_247_native_sparse25_r2.log" 2>&1 &
right=$!

status=0
wait "$left" || status=$?
wait "$right" || status=$?
exit "$status"
