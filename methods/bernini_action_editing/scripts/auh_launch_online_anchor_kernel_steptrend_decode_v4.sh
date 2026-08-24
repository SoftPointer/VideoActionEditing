#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this controller must run on the AUH login host" >&2; exit 2 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
training="$stage/online_anchor_attention_training_v1"
runner="$stage/source-online-anchor-attention-training-v1/methods/bernini_action_editing/scripts/auh_decode_online_anchor_attention_dynaedit_v1.sh"
log_root="$training/logs/kernel_v4_steptrend_decode"
mkdir -p "$log_root"
test -f "$runner"

launch_when_ready() {
  local job="$1" node="$2" experiment="$3" step="$4" transport="$5" strength="$6"
  local checkpoint="$training/train_${experiment}/checkpoint-$(printf '%08d' "$step")/receipt.json"
  local output="$training/dynaedit_fullgrid_v2/$experiment/step_$(printf '%08d' "$step")/e00"
  local log="$log_root/${experiment}_s${step}_e00_${node}.log"
  test ! -e "$output"
  test ! -e "$log"
  nohup bash -lc "
    set -euo pipefail
    while [ ! -f '$checkpoint' ]; do sleep 10; done
    exec srun --jobid='$job' --overlap --nodes=1 --ntasks=1 --nodelist='$node' \\
      env ONLINE_ANCHOR_DECODE_EXPERIMENT='$experiment' \\
          ONLINE_ANCHOR_DECODE_STEP='$step' \\
          ONLINE_ANCHOR_DECODE_TRANSPORT='$transport' \\
          ONLINE_ANCHOR_DECODE_STRENGTH='$strength' \\
          bash '$runner' 0 action_noop 1
  " >"$log" 2>&1 &
  echo "$! waits for $experiment step=$step on $node/GPU4-7"
}

primary=kernel_actionnoop_r025_replay025_lr1e5_s64_v4
hard=hard25_actionnoop_r050_replay025_lr1e5_s64_v4
launch_when_ready 141620 auh7-1b-gpu-226 "$primary" 16 temporal_kernel_contrast_attn_output 0.25
launch_when_ready 143812 auh7-1b-gpu-293 "$primary" 32 temporal_kernel_contrast_attn_output 0.25
launch_when_ready 143811 auh7-1b-gpu-306 "$primary" 64 temporal_kernel_contrast_attn_output 0.25
launch_when_ready 143808 auh7-1b-gpu-268 "$hard" 16 target_gated_hard_kernel_top25_attn_output 0.50
launch_when_ready 143808 auh7-1b-gpu-292 "$hard" 32 target_gated_hard_kernel_top25_attn_output 0.50
launch_when_ready 143808 auh7-1b-gpu-315 "$hard" 64 target_gated_hard_kernel_top25_attn_output 0.50
