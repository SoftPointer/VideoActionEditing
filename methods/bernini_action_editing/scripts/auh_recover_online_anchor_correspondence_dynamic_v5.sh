#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this controller must run on the AUH login host" >&2; exit 2 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
train_runner="$stage/source-online-anchor-attention-training-v1/methods/bernini_action_editing/scripts/auh_train_online_anchor_attention_fullgrid_existing_allocations_v2.sh"
decode_runner="$stage/source-online-anchor-attention-training-v1/methods/bernini_action_editing/scripts/auh_decode_online_anchor_attention_dynaedit_v1.sh"
archive="$release/online-anchor-correspondence-kernel-v5.tar"
train_logs="$release/logs/correspondence_kernel_v5_train"
decode_logs="$release/logs/correspondence_kernel_v5_decode"
experiment=corr25_dynamicstatic_r025_replay025_lr5e6_s96_v5
output="$release/train_$experiment"
train_log="$train_logs/${experiment}_auh7-1b-gpu-226.log"
decode_log="$decode_logs/node226_dynamic_recovery.log"

mkdir -p "$train_logs" "$decode_logs"
test -f "$train_runner"
test -f "$decode_runner"
test -f "$archive"
test ! -e "$output"
test ! -e "$train_log"
test ! -e "$decode_log"

nohup srun --jobid=141620 --overlap --nodes=1 --ntasks=1 --nodelist=auh7-1b-gpu-226 \
  env ONLINE_ANCHOR_PROFILE=dynamic_static \
      ONLINE_ANCHOR_VISIBLE_DEVICES=0,1,2,3 \
      ONLINE_ANCHOR_EXPERIMENT="$experiment" \
      ONLINE_ANCHOR_MAX_STEPS=96 \
      ONLINE_ANCHOR_ROUTE_OPERATOR=self_correspondence_kernel25 \
      ONLINE_ANCHOR_ROUTE_STRENGTH=0.25 \
      ONLINE_ANCHOR_REPLAY_WEIGHT=0.25 \
      ONLINE_ANCHOR_LEARNING_RATE=5e-6 \
      ONLINE_ANCHOR_METHOD_ARCHIVE="$archive" \
      bash "$train_runner" >"$train_log" 2>&1 &
train_pid=$!

nohup bash -lc "
  set -euo pipefail
  while [ ! -f '$output/TRAINING_COMPLETE' ]; do sleep 10; done
  for event in 0 4; do
    srun --jobid=141620 --overlap --nodes=1 --ntasks=1 --nodelist=auh7-1b-gpu-226 \
      env ONLINE_ANCHOR_DECODE_EXPERIMENT='$experiment' \
          ONLINE_ANCHOR_DECODE_STEP=96 \
          ONLINE_ANCHOR_DECODE_TRANSPORT=correspondence_gated_hard_kernel_top25_attn_output \
          ONLINE_ANCHOR_DECODE_STRENGTH=0.25 \
          bash '$decode_runner' \"\$event\" dynamic_static 0
  done
" >"$decode_log" 2>&1 &
decode_pid=$!

echo "train_pid=$train_pid decode_watcher_pid=$decode_pid experiment=$experiment"
