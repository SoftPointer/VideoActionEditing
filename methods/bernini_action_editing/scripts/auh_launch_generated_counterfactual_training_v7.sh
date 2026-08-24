#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this controller must run on the AUH login host" >&2; exit 2 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
source_tree="$stage/source-online-anchor-attention-training-v1"
train_runner="$source_tree/methods/bernini_action_editing/scripts/auh_train_online_anchor_attention_fullgrid_existing_allocations_v2.sh"
decode_runner="$source_tree/methods/bernini_action_editing/scripts/auh_decode_online_anchor_attention_dynaedit_v1.sh"
archive="$release/online-anchor-generated-counterfactual-v7.tar"
revision="$release/online-anchor-generated-counterfactual-v7.revision"
native_manifest="$stage/complex8_crossappearance_motion_v1/pairs_generated_counterfactual_v2/manifest_cross.json"
old_manifest="$stage/complex8_crossappearance_motion_v1/pairs_exact/manifest_cross.json"
logs="$release/logs/generated_counterfactual_v7"
transport=correspondence_gated_hard_kernel_top25_attn_output
mkdir -p "$logs"
test -f "$train_runner"
test -f "$decode_runner"
test -f "$archive"
test -f "$revision"
test -f "$old_manifest"

run_chain() {
  local job="$1" node="$2" sentinel="$3" experiment="$4" output_tag="$5"
  local pair_manifest="$6" replay_prompt="$7" route_strength="$8"
  while [[ ! -f "$sentinel" || ! -f "$pair_manifest" ]]; do sleep 10; done
  srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
    env ONLINE_ANCHOR_PROFILE=action_noop \
        ONLINE_ANCHOR_VISIBLE_DEVICES=0,1,2,3 \
        ONLINE_ANCHOR_EXPERIMENT="$experiment" \
        ONLINE_ANCHOR_MAX_STEPS=96 \
        ONLINE_ANCHOR_ROUTE_OPERATOR=self_correspondence_kernel25 \
        ONLINE_ANCHOR_ROUTE_STRENGTH="$route_strength" \
        ONLINE_ANCHOR_REPLAY_WEIGHT=0.25 \
        ONLINE_ANCHOR_REPLAY_PROMPT="$replay_prompt" \
        ONLINE_ANCHOR_PAIR_MANIFEST="$pair_manifest" \
        ONLINE_ANCHOR_LEARNING_RATE=1e-5 \
        ONLINE_ANCHOR_METHOD_ARCHIVE="$archive" \
        ONLINE_ANCHOR_METHOD_REVISION_FILE="$revision" \
        bash "$train_runner"
  for event in 0 4; do
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
      env ONLINE_ANCHOR_DECODE_EXPERIMENT="$experiment" \
          ONLINE_ANCHOR_OUTPUT_EXPERIMENT="$output_tag" \
          ONLINE_ANCHOR_DECODE_STEP=96 \
          ONLINE_ANCHOR_DECODE_TRANSPORT="$transport" \
          ONLINE_ANCHOR_DECODE_TRANSPORT_STEPS=3 \
          ONLINE_ANCHOR_DECODE_STRENGTH=0.25 \
          bash "$decode_runner" "$event" action_noop 0
  done
}

launch() {
  local label="$1" job="$2" node="$3" sentinel="$4" experiment="$5"
  local output_tag="$6" pair_manifest="$7" replay_prompt="$8" route_strength="$9"
  local log="$logs/${label}.log"
  test ! -e "$log"
  test ! -e "$release/train_$experiment"
  test ! -e "$release/dynaedit_fullgrid_v2/$output_tag"
  nohup bash "$0" --chain "$job" "$node" "$sentinel" "$experiment" \
    "$output_tag" "$pair_manifest" "$replay_prompt" "$route_strength" \
    >"$log" 2>&1 &
  echo "$! $node $experiment waits for $(basename "$sentinel") and $(basename "$pair_manifest")"
}

if [[ "${1:-}" == "--chain" ]]; then
  shift
  run_chain "$@"
  exit 0
fi

ready_268="$release/dynaedit_fullgrid_v2/corr25_primary_s96_early3_r050_v6/step_00000096/e04/E04_close-door-then-drawer_corr25_primary_s96_early3_r050_v6_S96_ONLINE_ANCHOR_REAL_SGA_ANC.mp4.receipt.json"
ready_315="$release/dynaedit_fullgrid_v2/corr25_primary_s96_early3_r025_v6/step_00000096/e04/E04_close-door-then-drawer_corr25_primary_s96_early3_r025_v6_S96_ONLINE_ANCHOR_REAL_SGA_ANC.mp4.receipt.json"
ready_292="$release/dynaedit_fullgrid_v2/corr25_noanchor_s64_early3_r025_v6/step_00000064/e04/E04_close-door-then-drawer_corr25_noanchor_s64_early3_r025_v6_S64_ONLINE_ANCHOR_REAL_SGA_ANC.mp4.receipt.json"

launch native_noop_r025 143808 auh7-1b-gpu-233 "$native_manifest" \
  corr25_nativecf_noopreplay_r025_s96_v7 corr25_nativecf_noopreplay_r025_s96_early3_v7 \
  "$native_manifest" noop 0.25
launch native_action_r025 143808 auh7-1b-gpu-268 "$ready_268" \
  corr25_nativecf_actionreplay_r025_s96_v7 corr25_nativecf_actionreplay_r025_s96_early3_v7 \
  "$native_manifest" action 0.25
launch deterministic_noop_r025 143808 auh7-1b-gpu-315 "$ready_315" \
  corr25_deterministic_noopreplay_r025_s96_v7 corr25_deterministic_noopreplay_r025_s96_early3_v7 \
  "$old_manifest" noop 0.25
launch native_noop_r050 143808 auh7-1b-gpu-292 "$ready_292" \
  corr25_nativecf_noopreplay_r050_s96_v7 corr25_nativecf_noopreplay_r050_s96_early3_v7 \
  "$native_manifest" noop 0.50
