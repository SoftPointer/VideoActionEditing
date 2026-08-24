#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this controller must run on the AUH login host" >&2; exit 2 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
runner="$stage/source-online-anchor-attention-training-v1/methods/bernini_action_editing/scripts/auh_decode_online_anchor_attention_dynaedit_v1.sh"
logs="$release/logs/correspondence_kernel_v6_earlyroute_decode"
primary=corr25_actionnoop_r050_replay025_lr1e5_s96_v5
transport=correspondence_gated_hard_kernel_top25_attn_output
mkdir -p "$logs"

run_one() {
  local job="$1" node="$2" event="$3" tag="$4" steps="$5" strength="$6"
  srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
    env ONLINE_ANCHOR_DECODE_EXPERIMENT="$primary" \
        ONLINE_ANCHOR_OUTPUT_EXPERIMENT="$tag" \
        ONLINE_ANCHOR_DECODE_STEP=96 \
        ONLINE_ANCHOR_DECODE_TRANSPORT="$transport" \
        ONLINE_ANCHOR_DECODE_TRANSPORT_STEPS="$steps" \
        ONLINE_ANCHOR_DECODE_STRENGTH="$strength" \
        bash "$runner" "$event" action_noop 0
}

case "${1:-}" in
  --chain)
    job="$2" node="$3" sentinel="$4" tag="$5" steps="$6" strength="$7"
    while [[ ! -f "$sentinel" ]]; do sleep 10; done
    run_one "$job" "$node" 0 "$tag" "$steps" "$strength"
    run_one "$job" "$node" 4 "$tag" "$steps" "$strength"
    exit 0
    ;;
esac

launch() {
  local label="$1" job="$2" node="$3" sentinel="$4" tag="$5" steps="$6" strength="$7"
  local log="$logs/${label}.log"
  test ! -e "$log"
  test ! -e "$release/dynaedit_fullgrid_v2/$tag"
  nohup bash "$0" --chain "$job" "$node" "$sentinel" "$tag" "$steps" "$strength" >"$log" 2>&1 &
  echo "$! $node $tag waits for $(basename "$sentinel")"
}

root="$release/dynaedit_fullgrid_v2"
launch early3_r025 143808 auh7-1b-gpu-315 \
  "$root/corr25_actionnoop_r050_replay025_lr2e5_s96_v5/step_00000096/e04/E04_close-door-then-drawer_corr25_actionnoop_r050_replay025_lr2e5_s96_v5_S96_ONLINE_ANCHOR_REAL_SGA_ANC.mp4.receipt.json" \
  corr25_primary_s96_early3_r025_v6 3 0.25
launch early3_r050 143808 auh7-1b-gpu-268 \
  "$root/corr25_hybrid_r050_replay025_lr1e5_s96_v5/step_00000096/e04/E04_close-door-then-drawer_corr25_hybrid_r050_replay025_lr1e5_s96_v5_S96_ONLINE_ANCHOR_REAL_SGA_ANC.mp4.receipt.json" \
  corr25_primary_s96_early3_r050_v6 3 0.50
launch early8_r025 143812 auh7-1b-gpu-293 \
  "$root/$primary/step_00000096/e04/E04_close-door-then-drawer_${primary}_S96_ONLINE_ANCHOR_REAL_SGA_ANC.mp4.receipt.json" \
  corr25_primary_s96_early8_r025_v6 8 0.25
launch early8_r050 143811 auh7-1b-gpu-306 \
  "$root/corr25_actionnoop_r025_replay025_lr1e5_s96_v5/step_00000096/e00/E00_pour-liquid-into-cup_corr25_actionnoop_r025_replay025_lr1e5_s96_v5_S96_ONLINE_ANCHOR_REAL_SGA_ANC.mp4.receipt.json" \
  corr25_primary_s96_early8_r050_v6 8 0.50
