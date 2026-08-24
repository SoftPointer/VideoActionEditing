#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this controller must run on the AUH login host" >&2; exit 2 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
runner="$stage/source-online-anchor-attention-training-v1/methods/bernini_action_editing/scripts/auh_decode_online_anchor_attention_dynaedit_v1.sh"
logs="$release/logs/correspondence_kernel_v5_decode"
transport=correspondence_gated_hard_kernel_top25_attn_output
mkdir -p "$logs"
test -f "$runner"

run_one() {
  local job="$1" node="$2" event="$3" experiment="$4" step="$5"
  local profile="$6" strength="$7"
  srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
    env ONLINE_ANCHOR_DECODE_EXPERIMENT="$experiment" \
        ONLINE_ANCHOR_DECODE_STEP="$step" \
        ONLINE_ANCHOR_DECODE_TRANSPORT="$transport" \
        ONLINE_ANCHOR_DECODE_STRENGTH="$strength" \
        bash "$runner" "$event" "$profile" 0
}

if [[ "${1:-}" == --functions-only ]]; then
  return 0 2>/dev/null || exit 0
fi

primary=corr25_actionnoop_r050_replay025_lr1e5_s96_v5
low=corr25_actionnoop_r025_replay025_lr1e5_s96_v5
high=corr25_actionnoop_r100_replay025_lr1e5_s96_v5
replay50=corr25_actionnoop_r050_replay050_lr1e5_s96_v5
replay10=corr25_actionnoop_r050_replay010_lr1e5_s96_v5
dynamic=corr25_dynamicstatic_r050_replay025_lr1e5_s96_v5
hybrid=corr25_hybrid_r050_replay025_lr1e5_s96_v5
lr2=corr25_actionnoop_r050_replay025_lr2e5_s96_v5

launch_chain() {
  local label="$1" job="$2" node="$3" sentinel="$4" body="$5"
  local log="$logs/${label}.log"
  test ! -e "$log"
  nohup bash -lc "
    set -euo pipefail
    source '$0' --functions-only
    while [ ! -f '$sentinel' ]; do sleep 10; done
    $body
  " >"$log" 2>&1 &
  echo "$! $node waits for $(basename "$(dirname "$sentinel")") then $label"
}

launch_chain node233_steptrend 143808 auh7-1b-gpu-233 \
  "$release/train_$primary/TRAINING_COMPLETE" \
  "while [ ! -f '$release/train_$low/TRAINING_COMPLETE' ]; do sleep 10; done
   run_one 143808 auh7-1b-gpu-233 0 '$primary' 16 action_noop 0.50
   run_one 143808 auh7-1b-gpu-233 0 '$primary' 32 action_noop 0.50
   run_one 143808 auh7-1b-gpu-233 0 '$primary' 64 action_noop 0.50"

launch_chain node292_primary96_high 143808 auh7-1b-gpu-292 \
  "$release/train_$high/TRAINING_COMPLETE" \
  "while [ ! -f '$release/train_$primary/TRAINING_COMPLETE' ]; do sleep 10; done
   run_one 143808 auh7-1b-gpu-292 0 '$primary' 96 action_noop 0.50
   run_one 143808 auh7-1b-gpu-292 0 '$high' 96 action_noop 1.00
   run_one 143808 auh7-1b-gpu-292 4 '$high' 96 action_noop 1.00"

launch_chain node293_replay50_primarye04 143812 auh7-1b-gpu-293 \
  "$release/train_$replay50/TRAINING_COMPLETE" \
  "run_one 143812 auh7-1b-gpu-293 0 '$replay50' 96 action_noop 0.50
   run_one 143812 auh7-1b-gpu-293 4 '$replay50' 96 action_noop 0.50
   while [ ! -f '$release/train_$primary/TRAINING_COMPLETE' ]; do sleep 10; done
   run_one 143812 auh7-1b-gpu-293 4 '$primary' 96 action_noop 0.50"

launch_chain node306_replay10_lowe00 143811 auh7-1b-gpu-306 \
  "$release/train_$replay10/TRAINING_COMPLETE" \
  "run_one 143811 auh7-1b-gpu-306 0 '$replay10' 96 action_noop 0.50
   run_one 143811 auh7-1b-gpu-306 4 '$replay10' 96 action_noop 0.50
   while [ ! -f '$release/train_$low/TRAINING_COMPLETE' ]; do sleep 10; done
   run_one 143811 auh7-1b-gpu-306 0 '$low' 96 action_noop 0.25"

launch_chain node226_low_e04_dynamic 141620 auh7-1b-gpu-226 \
  "$release/train_$dynamic/TRAINING_COMPLETE" \
  "while [ ! -f '$release/train_$low/TRAINING_COMPLETE' ]; do sleep 10; done
   run_one 141620 auh7-1b-gpu-226 4 '$low' 96 action_noop 0.25
   run_one 141620 auh7-1b-gpu-226 0 '$dynamic' 96 dynamic_static 0.50
   run_one 141620 auh7-1b-gpu-226 4 '$dynamic' 96 dynamic_static 0.50"

launch_chain node268_hybrid 143808 auh7-1b-gpu-268 \
  "$release/train_$hybrid/TRAINING_COMPLETE" \
  "run_one 143808 auh7-1b-gpu-268 0 '$hybrid' 96 hybrid 0.50
   run_one 143808 auh7-1b-gpu-268 4 '$hybrid' 96 hybrid 0.50"

launch_chain node315_lr2 143808 auh7-1b-gpu-315 \
  "$release/train_$lr2/TRAINING_COMPLETE" \
  "run_one 143808 auh7-1b-gpu-315 0 '$lr2' 96 action_noop 0.50
   run_one 143808 auh7-1b-gpu-315 4 '$lr2' 96 action_noop 0.50"
