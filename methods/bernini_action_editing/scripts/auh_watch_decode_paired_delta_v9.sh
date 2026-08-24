#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this watcher must run on the AUH login host" >&2; exit 2 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
runner="$stage/source-online-anchor-attention-training-v1/methods/bernini_action_editing/scripts/auh_decode_online_anchor_attention_dynaedit_v1.sh"
logs="$release/logs/paired_delta_v9_decode"
transport=correspondence_gated_hard_kernel_top25_attn_output
mkdir -p "$logs"
test -f "$runner"

wait_complete() {
  local experiment="$1"
  while [ ! -f "$release/train_${experiment}/TRAINING_COMPLETE" ]; do sleep 10; done
}

wait_receipt() {
  local experiment="$1" step="$2"
  local receipt="$release/train_${experiment}/checkpoint-$(printf '%08d' "$step")/receipt.json"
  while [ ! -f "$receipt" ]; do sleep 10; done
}

decode_one() {
  local job="$1" node="$2" experiment="$3" tag="$4" step="$5" event="$6"
  srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --cpus-per-task=64 \
    --gres=gpu:4 --mem=0 --nodelist="$node" \
    env ONLINE_ANCHOR_DECODE_EXPERIMENT="$experiment" \
        ONLINE_ANCHOR_OUTPUT_EXPERIMENT="$tag" \
        ONLINE_ANCHOR_DECODE_STEP="$step" \
        ONLINE_ANCHOR_DECODE_TRANSPORT="$transport" \
        ONLINE_ANCHOR_DECODE_TRANSPORT_STEPS=3 \
        ONLINE_ANCHOR_DECODE_STRENGTH=0.25 \
        ONLINE_ANCHOR_DECODE_CFG_SCOPE=target_conditional_only \
        bash "$runner" "$event" action_noop 0
}

run_trend() {
  local job="$1" node="$2" experiment="$3" tag="$4"
  for step in 32 64; do
    wait_receipt "$experiment" "$step"
    for event in 0 4; do
      decode_one "$job" "$node" "$experiment" "$tag" "$step" "$event"
    done
  done
}

run_final() {
  local job="$1" node="$2" experiment="$3" tag="$4"
  wait_complete "$experiment"
  for event in 0 4; do
    decode_one "$job" "$node" "$experiment" "$tag" 64 "$event"
  done
}

run_probe() {
  local job="$1" node="$2" experiment="$3" tag="$4"
  wait_receipt "$experiment" 8
  decode_one "$job" "$node" "$experiment" "$tag" 8 0
}

launch() {
  local label="$1"; shift
  local log="$logs/${label}.log"
  test ! -e "$log"
  nohup bash "$0" --chain "$@" >"$log" 2>&1 &
  echo "$! $label"
}

if [ "${1:-}" = --chain ]; then
  shift
  mode="$1"; shift
  case "$mode" in
    trend) run_trend "$@" ;;
    final) run_final "$@" ;;
    probe) run_probe "$@" ;;
    *) exit 3 ;;
  esac
  exit 0
fi

launch primary_trend trend 143812 auh7-1b-gpu-293 \
  corr25_paireddelta_cf4_tf025_replay025_s64_v9 \
  corr25_paireddelta_cf4_tf025_replay025_early3_v9
launch pure_delta_trend trend 143811 auh7-1b-gpu-306 \
  corr25_paireddelta_cf4_puredelta_replay025_s64_v9 \
  corr25_paireddelta_cf4_puredelta_replay025_early3_v9
launch noanchor_final final 143808 auh7-1b-gpu-292 \
  noanchor_paireddelta_cf4_tf025_replay025_s64_v9 \
  noanchor_paireddelta_cf4_tf025_replay025_early3_v9
launch mixed_final final 143808 auh7-1b-gpu-315 \
  corr25_paireddelta_mixed_tf025_replay025_s64_v9 \
  corr25_paireddelta_mixed_tf025_replay025_early3_v9
