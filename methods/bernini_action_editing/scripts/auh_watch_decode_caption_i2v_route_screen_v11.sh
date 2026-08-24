#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this watcher must run on the AUH login host" >&2; exit 2 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
runner="$stage/source-online-anchor-attention-training-v1/methods/bernini_action_editing/scripts/auh_decode_online_anchor_attention_dynaedit_v1.sh"
logs="$release/logs/caption_i2v_route_screen_v11_decode"
mkdir -p "$logs"
test -f "$runner"

wait_complete() {
  local experiment="$1"
  while [ ! -f "$release/train_${experiment}/TRAINING_COMPLETE" ]; do sleep 10; done
}

decode_one() {
  local job="$1" node="$2" experiment="$3" tag="$4" transport="$5" step="$6" event="$7"
  local event_token
  printf -v event_token 'e%02d' "$event"
  local output_dir="$release/dynaedit_fullgrid_v2/$tag/step_$(printf '%08d' "$step")/$event_token"
  if find "$output_dir" -maxdepth 1 -type f -name '*.mp4' -print -quit 2>/dev/null | grep -q .; then
    echo "skip existing decode: tag=$tag step=$step event=$event_token"
    return 0
  fi
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

run_arm() {
  local job="$1" node="$2" experiment="$3" tag="$4" transport="$5"
  wait_complete "$experiment"
  # Decode both an early checkpoint and the one-full-target-bank checkpoint.
  # E00 exposes actor/recipient leakage; E04 exposes ordered articulation.
  for step in 8 32; do
    for event in 0 4; do
      decode_one "$job" "$node" "$experiment" "$tag" "$transport" "$step" "$event"
    done
  done
}

if [ "${1:-}" = --chain ]; then
  shift
  run_arm "$@"
  exit 0
fi

launch() {
  local label="$1"; shift
  local log="$logs/${label}.log"
  test ! -e "$log"
  nohup bash "$0" --chain "$@" >"$log" 2>&1 &
  echo "$! $label"
}

# The watcher consumes no GPU until its matching S32 training has naturally
# completed.  Each arm then decodes on that same node, so no node hosts two
# model replicas at once.
launch cross_tf025 143808 auh7-1b-gpu-233 \
  cross_captioni2v_delta_tf025_replay025_s32_v11 \
  cross_captioni2v_delta_tf025_replay025_early3_v11 \
  temporal_contrast_cross_attn_output
launch cross_pure 143808 auh7-1b-gpu-268 \
  cross_captioni2v_delta_pure_replay025_s32_v11 \
  cross_captioni2v_delta_pure_replay025_early3_v11 \
  temporal_contrast_cross_attn_output
launch temporal_kernel 143808 auh7-1b-gpu-315 \
  temporalkernel_captioni2v_delta_tf025_replay025_s32_v11 \
  temporalkernel_captioni2v_delta_tf025_replay025_early3_v11 \
  temporal_kernel_contrast_attn_output
launch target_gate 141620 auh7-1b-gpu-226 \
  targetgate_captioni2v_delta_tf025_replay025_s32_v11 \
  targetgate_captioni2v_delta_tf025_replay025_early3_v11 \
  target_gated_hard_kernel_top25_attn_output
