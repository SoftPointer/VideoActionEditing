#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this controller must run on the AUH login host" >&2; exit 2 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
runner="$stage/source-online-anchor-attention-training-v1/methods/bernini_action_editing/scripts/auh_decode_online_anchor_attention_dynaedit_v1.sh"
logs="$release/logs/paired_delta_v9_decode_recovery"
transport=correspondence_gated_hard_kernel_top25_attn_output
mkdir -p "$logs"
test -f "$runner"

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

run_chain() {
  local job="$1" node="$2" experiment="$3" tag="$4" mode="$5"
  local steps
  case "$mode" in
    final) steps="64" ;;
    trend) steps="32 64" ;;
    *) echo "unknown mode: $mode" >&2; exit 3 ;;
  esac
  for step in $steps; do
    for event in 0 4; do
      decode_one "$job" "$node" "$experiment" "$tag" "$step" "$event"
    done
  done
}

if [ "${1:-}" = --chain ]; then
  shift
  run_chain "$@"
  exit 0
fi

launch() {
  local label="$1"; shift
  local log="$logs/${label}.log"
  test ! -e "$log"
  nohup bash "$0" --chain "$@" >"$log" 2>&1 &
  echo "$! $label"
}

# One four-GPU decoder per node; each chain is sequential to stay below the
# 64-GiB host-memory cgroup limit observed when two replicas overlapped.
launch noanchor_final_recover 143812 auh7-1b-gpu-293 \
  noanchor_paireddelta_cf4_tf025_replay025_s64_v9 \
  noanchor_paireddelta_cf4_tf025_replay025_recover_v9 final
launch pure_delta_trend_recover 143811 auh7-1b-gpu-306 \
  corr25_paireddelta_cf4_puredelta_replay025_s64_v9 \
  corr25_paireddelta_cf4_puredelta_replay025_recover_v9 trend
