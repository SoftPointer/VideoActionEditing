#!/usr/bin/env bash
set -euo pipefail

node="$(hostname -s)"
case "$node" in
  auh7-1b-gpu-246) profiles=(sgaanc uniform_anc) ;;
  auh7-1b-gpu-247) profiles=(hard_sga no_gain) ;;
  *) echo "dual SGA/ANC training is restricted to nodes 246/247" >&2; exit 3 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
runner="$stage/source-sga-anc-training-v1/methods/bernini_action_editing/scripts/auh_train_complex8_sga_anc_v1.sh"
log_root="$stage/complex8_sga_anc_training_v1/logs"
mkdir -p "$log_root"

SGA_ANC_PROFILE="${profiles[0]}" SGA_ANC_VISIBLE_DEVICES=0,1,2,3 \
  bash "$runner" >"$log_root/${node}-${profiles[0]}.log" 2>&1 &
pid0=$!
SGA_ANC_PROFILE="${profiles[1]}" SGA_ANC_VISIBLE_DEVICES=4,5,6,7 \
  bash "$runner" >"$log_root/${node}-${profiles[1]}.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"
touch "$stage/complex8_sga_anc_training_v1/TRAIN_${node}_COMPLETE"
