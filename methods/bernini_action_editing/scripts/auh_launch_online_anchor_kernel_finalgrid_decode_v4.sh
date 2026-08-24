#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this controller must run on the AUH login host" >&2; exit 2 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
training="$stage/online_anchor_attention_training_v1"
runner="$stage/source-online-anchor-attention-training-v1/methods/bernini_action_editing/scripts/auh_decode_online_anchor_attention_dynaedit_v1.sh"
log_root="$training/logs/kernel_v4_finalgrid_decode"
mkdir -p "$log_root"
test -f "$runner"

launch() {
  local job="$1" node="$2" slot="$3" event="$4" experiment="$5"
  local arm="$6" transport="$7" strength="$8"
  local receipt="$training/train_${experiment}/checkpoint-00000064/receipt.json"
  local output="$training/dynaedit_fullgrid_v2/$experiment/step_00000064/e$(printf '%02d' "$event")"
  local log="$log_root/${experiment}_s64_e$(printf '%02d' "$event")_${node}_g${slot}.log"
  test -f "$receipt"
  test ! -e "$output"
  test ! -e "$log"
  nohup srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
    env ONLINE_ANCHOR_DECODE_EXPERIMENT="$experiment" \
        ONLINE_ANCHOR_DECODE_STEP=64 \
        ONLINE_ANCHOR_DECODE_TRANSPORT="$transport" \
        ONLINE_ANCHOR_DECODE_STRENGTH="$strength" \
        bash "$runner" "$event" "$arm" "$slot" >"$log" 2>&1 &
  echo "$! $node/GPU-group-$slot E$event $experiment"
}

launch_chain() {
  local job="$1" node="$2" slot="$3"
  local hard=hard25_actionnoop_r050_replay025_lr1e5_s64_v4
  local hybrid=hard25_hybrid_r050_replay025_lr1e5_s64_v4
  local log="$log_root/chain_hard32_e00_then_hybrid_e04_${node}_g${slot}.log"
  test -f "$training/train_${hard}/checkpoint-00000032/receipt.json"
  test -f "$training/train_${hybrid}/checkpoint-00000064/receipt.json"
  test ! -e "$training/dynaedit_fullgrid_v2/$hard/step_00000032/e00"
  test ! -e "$training/dynaedit_fullgrid_v2/$hybrid/step_00000064/e04"
  test ! -e "$log"
  nohup bash -lc "
    set -euo pipefail
    srun --jobid='$job' --overlap --nodes=1 --ntasks=1 --nodelist='$node' \\
      env ONLINE_ANCHOR_DECODE_EXPERIMENT='$hard' ONLINE_ANCHOR_DECODE_STEP=32 \\
          ONLINE_ANCHOR_DECODE_TRANSPORT=target_gated_hard_kernel_top25_attn_output \\
          ONLINE_ANCHOR_DECODE_STRENGTH=0.50 bash '$runner' 0 action_noop '$slot'
    srun --jobid='$job' --overlap --nodes=1 --ntasks=1 --nodelist='$node' \\
      env ONLINE_ANCHOR_DECODE_EXPERIMENT='$hybrid' ONLINE_ANCHOR_DECODE_STEP=64 \\
          ONLINE_ANCHOR_DECODE_TRANSPORT=target_gated_hard_kernel_top25_attn_output \\
          ONLINE_ANCHOR_DECODE_STRENGTH=0.50 bash '$runner' 4 hybrid '$slot'
  " >"$log" 2>&1 &
  echo "$! $node/GPU-group-$slot hard32-E00 then hard-hybrid-E04"
}

kernel=temporal_kernel_contrast_attn_output
hard=target_gated_hard_kernel_top25_attn_output

launch_chain 143808 auh7-1b-gpu-233 0
launch 143808 auh7-1b-gpu-233 1 4 kernel_actionnoop_r025_replay025_lr1e5_s64_v4 action_noop "$kernel" 0.25
launch 143808 auh7-1b-gpu-268 0 0 kernel_actionnoop_r050_replay025_lr1e5_s64_v4 action_noop "$kernel" 0.50
launch 143808 auh7-1b-gpu-268 1 4 kernel_actionnoop_r050_replay025_lr1e5_s64_v4 action_noop "$kernel" 0.50
launch 143808 auh7-1b-gpu-292 0 0 kernel_actionnoop_r025_replay050_lr1e5_s64_v4 action_noop "$kernel" 0.25
launch 143808 auh7-1b-gpu-292 1 4 kernel_actionnoop_r025_replay050_lr1e5_s64_v4 action_noop "$kernel" 0.25
launch 143808 auh7-1b-gpu-315 0 0 kernel_dynamicstatic_r025_replay025_lr1e5_s64_v4 dynamic_static "$kernel" 0.25
launch 143808 auh7-1b-gpu-315 1 4 kernel_dynamicstatic_r025_replay025_lr1e5_s64_v4 dynamic_static "$kernel" 0.25
launch 141620 auh7-1b-gpu-226 0 0 hard25_actionnoop_r025_replay025_lr1e5_s64_v4 action_noop "$hard" 0.25
launch 141620 auh7-1b-gpu-226 1 4 hard25_actionnoop_r025_replay025_lr1e5_s64_v4 action_noop "$hard" 0.25
launch 143812 auh7-1b-gpu-293 0 4 hard25_actionnoop_r050_replay025_lr1e5_s64_v4 action_noop "$hard" 0.50
launch 143812 auh7-1b-gpu-293 1 0 hard25_actionnoop_r100_replay025_lr1e5_s64_v4 action_noop "$hard" 1.00
launch 143811 auh7-1b-gpu-306 0 4 hard25_actionnoop_r100_replay025_lr1e5_s64_v4 action_noop "$hard" 1.00
launch 143811 auh7-1b-gpu-306 1 0 hard25_hybrid_r050_replay025_lr1e5_s64_v4 hybrid "$hard" 0.50
