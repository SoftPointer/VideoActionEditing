#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this watcher must run on the AUH login host" >&2; exit 2 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
runner="$stage/source-online-anchor-attention-training-v1/methods/bernini_action_editing/scripts/auh_decode_online_anchor_attention_dynaedit_v1.sh"
logs="$release/logs/counterfactual4_identity_v8_decode"
corr=correspondence_gated_hard_kernel_top25_attn_output
kernel=temporal_kernel_contrast_attn_output
mkdir -p "$logs"

wait_receipt() {
  local experiment="$1" step="$2"
  local receipt="$release/train_${experiment}/checkpoint-$(printf '%08d' "$step")/receipt.json"
  while [[ ! -f "$receipt" ]]; do sleep 10; done
}

wait_training_complete() {
  local experiment="$1"
  local sentinel="$release/train_${experiment}/TRAINING_COMPLETE"
  while [[ ! -f "$sentinel" ]]; do sleep 10; done
}

decode_one() {
  local job="$1" node="$2" experiment="$3" output_tag="$4"
  local step="$5" event="$6" transport="$7" strength="$8"
  local cfg_scope="${9:-target_conditional_only}"
  srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --cpus-per-task=64 \
    --gres=gpu:4 --mem=0 --nodelist="$node" \
    env ONLINE_ANCHOR_DECODE_EXPERIMENT="$experiment" \
        ONLINE_ANCHOR_OUTPUT_EXPERIMENT="$output_tag" \
        ONLINE_ANCHOR_DECODE_STEP="$step" \
        ONLINE_ANCHOR_DECODE_TRANSPORT="$transport" \
        ONLINE_ANCHOR_DECODE_TRANSPORT_STEPS=3 \
        ONLINE_ANCHOR_DECODE_STRENGTH="$strength" \
        ONLINE_ANCHOR_DECODE_CFG_SCOPE="$cfg_scope" \
        bash "$runner" "$event" action_noop 0
}

run_primary_trend() {
  local experiment=corr25_counterfactual4_identity_r025_s96_v8
  local tag=corr25_counterfactual4_identity_primary_early3_v8
  while squeue -h -s -j 141620 | grep -q '^ *141620\.348 '; do sleep 10; done
  for step in 32 64 96; do
    wait_receipt "$experiment" "$step"
    decode_one 141620 auh7-1b-gpu-226 "$experiment" "$tag" "$step" 0 "$corr" 0.25
    decode_one 141620 auh7-1b-gpu-226 "$experiment" "$tag" "$step" 4 "$corr" 0.25
  done
  for event in 2 7; do
    decode_one 141620 auh7-1b-gpu-226 "$experiment" "$tag" 96 "$event" "$corr" 0.25
  done
}

run_pair() {
  local job="$1" node="$2" experiment_a="$3" tag_a="$4" transport_a="$5"
  local experiment_b="$6" tag_b="$7" transport_b="$8"
  wait_training_complete "$experiment_a"
  wait_training_complete "$experiment_b"
  for event in 0 4; do
    decode_one "$job" "$node" "$experiment_a" "$tag_a" 96 "$event" "$transport_a" 0.25
  done
  for event in 0 4; do
    decode_one "$job" "$node" "$experiment_b" "$tag_b" 96 "$event" "$transport_b" 0.25
  done
}

run_single() {
  local job="$1" node="$2" experiment="$3" tag="$4" transport="$5"
  wait_training_complete "$experiment"
  for event in 0 4; do
    decode_one "$job" "$node" "$experiment" "$tag" 96 "$event" "$transport" 0.25
  done
}

run_primary_shared() {
  local experiment=corr25_counterfactual4_identity_r025_s96_v8
  local tag=corr25_counterfactual4_identity_primary_sharedcfg_early3_v8
  wait_training_complete corr25_deterministic_noopreplay_r025_s96_v7
  for step in 32 64 96; do
    wait_receipt "$experiment" "$step"
    for event in 0 4; do
      decode_one 143808 auh7-1b-gpu-315 "$experiment" "$tag" "$step" "$event" "$corr" 0.25 shared
    done
  done
}

launch() {
  local label="$1"; shift
  local log="$logs/${label}.log"
  test ! -e "$log"
  nohup bash "$0" --chain "$@" >"$log" 2>&1 &
  echo "$! $label"
}

if [[ "${1:-}" == --chain ]]; then
  shift
  mode="$1"; shift
  case "$mode" in
    primary) run_primary_trend "$@" ;;
    primary_shared) run_primary_shared "$@" ;;
    pair) run_pair "$@" ;;
    single) run_single "$@" ;;
    *) exit 3 ;;
  esac
  exit 0
fi

launch primary_trend primary
launch node268_pair pair 143808 auh7-1b-gpu-268 \
  noanchor_counterfactual4_identity_s96_v8 noanchor_counterfactual4_identity_early3_v8 "$corr" \
  corr25_counterfactual4_identity_r010_s96_v8_g47 corr25_counterfactual4_identity_replay010_early3_v8 "$corr"
launch node292_pair pair 143808 auh7-1b-gpu-292 \
  kernel_counterfactual4_identity_r025_s96_v8 kernel_counterfactual4_identity_early3_v8 "$kernel" \
  corr25_counterfactual4_identity_r025_lr5e6_s96_v8_map3456 corr25_counterfactual4_identity_lr5e6_early3_v8 "$corr"
launch node306_pair pair 143811 auh7-1b-gpu-306 \
  corr25_counterfactual4_identity_r0125_s96_v8 corr25_counterfactual4_identity_trainstrength0125_early3_v8 "$corr" \
  corr25_mixed_identity_r025_s96_v8_g47 corr25_mixed_identity_early3_v8 "$corr"
launch replay050 single 143812 auh7-1b-gpu-293 \
  corr25_counterfactual4_identity_r050_s96_v8 corr25_counterfactual4_identity_replay050_early3_v8 "$corr"
launch primary_shared primary_shared
