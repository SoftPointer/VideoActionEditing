#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this watcher must run on the AUH login host" >&2; exit 2 ;;
esac

job="${ONLINE_ANCHOR_JOB:-143808}"
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
source_tree="$stage/source-online-anchor-targetcoord-routed-teacher-v14"
dev="$source_tree/methods/bernini_action_editing"
runner="$dev/scripts/auh_decode_online_anchor_attention_dynaedit_v1.sh"
bridge_runner="$dev/scripts/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh"
watcher="$dev/scripts/auh_watch_decode_real_source_routed_teacher_delta_v14.sh"
logs="$release/logs/real_source_routed_teacher_delta_v14_decode"
mkdir -p "$logs"
test -d "$source_tree"
test ! -L "$source_tree"
test -f "$runner"
test ! -L "$runner"
test -f "$bridge_runner"
test ! -L "$bridge_runner"

wait_complete() {
  local experiment="$1"
  while [ ! -f "$release/train_${experiment}/TRAINING_COMPLETE" ]; do
    sleep 10
  done
}

decode_transport_for_route() {
  case "$1" in
    self_temporal_kernel) echo temporal_kernel_contrast_attn_output ;;
    self_target_gated_kernel25) echo target_gated_hard_kernel_top25_attn_output ;;
    self_correspondence_kernel25) echo correspondence_gated_hard_kernel_top25_attn_output ;;
    *) echo "unsupported v14 training route: $1" >&2; return 2 ;;
  esac
}

decode_one() {
  local node="$1" experiment="$2" tag="$3" route="$4" step="$5" event="$6"
  local transport_steps="$7" allow_route_off="$8" preservation="$9" sga_score="${10}"
  local transport checkpoint receipt adapter_model event_token output_dir
  local adapter_sha_line receipt_sha_line adapter_sha receipt_sha
  transport="$(decode_transport_for_route "$route")"
  checkpoint="$release/train_${experiment}/checkpoint-$(printf '%08d' "$step")"
  receipt="$checkpoint/receipt.json"
  adapter_model="$checkpoint/adapter/adapter_model.safetensors"
  test -f "$receipt"
  test -f "$checkpoint/adapter/adapter_config.json"
  test -f "$adapter_model"
  jq -e --argjson step "$step" --arg route "$route" '
    .complete == true and
    .global_step == $step and
    .training_contract.training_objective == "real_source_routed_teacher_delta" and
    .training_contract.route_operator == $route
  ' "$receipt" >/dev/null
  adapter_sha_line="$(sha256sum -- "$adapter_model")"
  receipt_sha_line="$(sha256sum -- "$receipt")"
  adapter_sha="${adapter_sha_line%% *}"
  receipt_sha="${receipt_sha_line%% *}"
  printf -v event_token 'e%02d' "$event"
  output_dir="$release/dynaedit_fullgrid_v2/$tag/step_$(printf '%08d' "$step")/$event_token"
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
        ONLINE_ANCHOR_DECODE_TRANSPORT_STEPS="$transport_steps" \
        ONLINE_ANCHOR_DECODE_STRENGTH=0.25 \
        ONLINE_ANCHOR_DECODE_CFG_SCOPE=shared \
        ONLINE_ANCHOR_DECODE_STATE_MODE=clean_noised \
        ONLINE_ANCHOR_DECODE_SOURCE_CFG_SCALE=4.5 \
        ONLINE_ANCHOR_DECODE_TARGET_CFG_SCALE=4.5 \
        ONLINE_ANCHOR_DECODE_PRESERVATION_MODE="$preservation" \
        ONLINE_ANCHOR_DECODE_SGA_SCORE_MODE="$sga_score" \
        ONLINE_ANCHOR_ALLOW_TRAINED_ROUTE_OFF_CONTROL="$allow_route_off" \
        ONLINE_ANCHOR_EXPECTED_TRAINING_OBJECTIVE=real_source_routed_teacher_delta \
        ONLINE_ANCHOR_EXPECTED_TRAINING_ROUTE_OPERATOR="$route" \
        ONLINE_ANCHOR_EXPECTED_ADAPTER_SHA256="$adapter_sha" \
        ONLINE_ANCHOR_EXPECTED_RECEIPT_SHA256="$receipt_sha" \
        ONLINE_ANCHOR_DECODE_DEV_OVERRIDE="$dev" \
        ONLINE_ANCHOR_DECODE_BRIDGE_RUNNER="$bridge_runner" \
        bash "$runner" "$event" action_noop 0
}

run_arm() {
  local node="$1" experiment="$2" route="$3" primary="$4"
  local route_on_tag="${experiment}_routeon_clean40_v14"
  local route_off_tag="${experiment}_sameckpt_routeoff_control_v14"
  local preservation_tag="${experiment}_preserve_motion_actionreward_v14"
  wait_complete "$experiment"

  # One srun is awaited before the next begins: one four-GPU decoder per node.
  for step in 8 32; do
    for event in 0 2 4 7; do
      decode_one "$node" "$experiment" "$route_on_tag" "$route" \
        "$step" "$event" 40 0 none global_source_cosine
    done
  done
  for event in 0 4; do
    decode_one "$node" "$experiment" "$route_off_tag" "$route" \
      32 "$event" 0 1 none global_source_cosine
  done
  if [ "$primary" = 1 ]; then
    for event in 0 4; do
      decode_one "$node" "$experiment" "$preservation_tag" "$route" \
        32 "$event" 40 0 source_motion_support background_plus_anchor_action_002
    done
  elif [ "$primary" != 0 ]; then
    echo "primary flag must be 0 or 1" >&2
    exit 3
  fi
}

if [ "${1:-}" = --chain ]; then
  shift
  run_arm "$@"
  exit 0
fi

test -f "$watcher"
test ! -L "$watcher"
launch() {
  local label="$1"; shift
  local log="$logs/${label}.log"
  if [ -e "$log" ]; then
    echo "skip existing watcher log: $log"
    return 0
  fi
  nohup bash "$watcher" --chain "$@" >"$log" 2>&1 &
  echo "$! $job $label"
}

launch temporal_t050 auh7-1b-gpu-233 \
  temporalkernel_targetcoord_t050_s025_replay0025_s32_v14 \
  self_temporal_kernel 1
launch temporal_t100 auh7-1b-gpu-268 \
  temporalkernel_targetcoord_t100_s025_replay0025_s32_v14 \
  self_temporal_kernel 0
launch target_gate_t050 auh7-1b-gpu-292 \
  targetgate_targetcoord_t050_s025_replay0025_s32_v14 \
  self_target_gated_kernel25 0
launch correspondence_t050 auh7-1b-gpu-315 \
  corr25_targetcoord_t050_s025_replay0025_s32_v14 \
  self_correspondence_kernel25 0
