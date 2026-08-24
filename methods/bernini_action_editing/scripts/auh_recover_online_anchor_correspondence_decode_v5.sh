#!/usr/bin/env bash
set -euo pipefail

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
runner="$stage/source-online-anchor-attention-training-v1/methods/bernini_action_editing/scripts/auh_decode_online_anchor_attention_dynaedit_v1.sh"
logs="$release/logs/correspondence_kernel_v5_decode"
transport=correspondence_gated_hard_kernel_top25_attn_output
primary=corr25_actionnoop_r050_replay025_lr1e5_s96_v5
low=corr25_actionnoop_r025_replay025_lr1e5_s96_v5
replay50=corr25_actionnoop_r050_replay050_lr1e5_s96_v5
replay10=corr25_actionnoop_r050_replay010_lr1e5_s96_v5

run_one() {
  local job="$1" node="$2" event="$3" experiment="$4" step="$5"
  srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
    env ONLINE_ANCHOR_DECODE_EXPERIMENT="$experiment" \
        ONLINE_ANCHOR_DECODE_STEP="$step" \
        ONLINE_ANCHOR_DECODE_TRANSPORT="$transport" \
        ONLINE_ANCHOR_DECODE_STRENGTH=0.50 \
        bash "$runner" "$event" action_noop 0
}

case "${1:-}" in
  --node293)
    run_one 143812 auh7-1b-gpu-293 0 "$replay50" 96
    run_one 143812 auh7-1b-gpu-293 4 "$replay50" 96
    while [[ ! -f "$release/train_$primary/TRAINING_COMPLETE" ]]; do sleep 10; done
    run_one 143812 auh7-1b-gpu-293 4 "$primary" 96
    exit 0
    ;;
  --node306)
    run_one 143811 auh7-1b-gpu-306 0 "$replay10" 96
    run_one 143811 auh7-1b-gpu-306 4 "$replay10" 96
    while [[ ! -f "$release/train_$low/TRAINING_COMPLETE" ]]; do sleep 10; done
    run_one 143811 auh7-1b-gpu-306 0 "$low" 96
    exit 0
    ;;
esac

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this controller must run on the AUH login host" >&2; exit 2 ;;
esac

mkdir -p "$logs"
for item in \
  "$replay50/step_00000096/e00/E00_pour-liquid-into-cup_${replay50}_S96_ONLINE_ANCHOR_REAL_SGA_ANC.mp4" \
  "$replay50/step_00000096/e04/E04_close-door-then-drawer_${replay50}_S96_ONLINE_ANCHOR_REAL_SGA_ANC.mp4" \
  "$primary/step_00000096/e04/E04_close-door-then-drawer_${primary}_S96_ONLINE_ANCHOR_REAL_SGA_ANC.mp4" \
  "$replay10/step_00000096/e00/E00_pour-liquid-into-cup_${replay10}_S96_ONLINE_ANCHOR_REAL_SGA_ANC.mp4" \
  "$replay10/step_00000096/e04/E04_close-door-then-drawer_${replay10}_S96_ONLINE_ANCHOR_REAL_SGA_ANC.mp4" \
  "$low/step_00000096/e00/E00_pour-liquid-into-cup_${low}_S96_ONLINE_ANCHOR_REAL_SGA_ANC.mp4"; do
  test ! -e "$release/dynaedit_fullgrid_v2/$item"
done

log293="$logs/node293_replay50_primarye04_recovery.log"
log306="$logs/node306_replay10_lowe00_recovery.log"
test ! -e "$log293"
test ! -e "$log306"
nohup bash "$0" --node293 >"$log293" 2>&1 &
pid293=$!
nohup bash "$0" --node306 >"$log306" 2>&1 &
pid306=$!
echo "node293_pid=$pid293 node306_pid=$pid306"
