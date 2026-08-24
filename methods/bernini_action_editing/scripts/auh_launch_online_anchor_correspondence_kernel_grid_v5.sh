#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this controller must run on the AUH login host" >&2; exit 2 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
runner="$stage/source-online-anchor-attention-training-v1/methods/bernini_action_editing/scripts/auh_train_online_anchor_attention_fullgrid_existing_allocations_v2.sh"
archive="$release/online-anchor-correspondence-kernel-v5.tar"
logs="$release/logs/correspondence_kernel_v5_train"
mkdir -p "$logs"
test -f "$runner"
test -f "$archive"

launch() {
  local job="$1" node="$2" devices="$3" experiment="$4" profile="$5"
  local strength="$6" replay="$7" lr="$8"
  local output="$release/train_$experiment"
  local log="$logs/${experiment}_${node}.log"
  test ! -e "$output"
  test ! -e "$log"
  nohup srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
    env ONLINE_ANCHOR_PROFILE="$profile" \
        ONLINE_ANCHOR_VISIBLE_DEVICES="$devices" \
        ONLINE_ANCHOR_EXPERIMENT="$experiment" \
        ONLINE_ANCHOR_MAX_STEPS=96 \
        ONLINE_ANCHOR_ROUTE_OPERATOR=self_correspondence_kernel25 \
        ONLINE_ANCHOR_ROUTE_STRENGTH="$strength" \
        ONLINE_ANCHOR_REPLAY_WEIGHT="$replay" \
        ONLINE_ANCHOR_LEARNING_RATE="$lr" \
        ONLINE_ANCHOR_METHOD_ARCHIVE="$archive" \
        bash "$runner" >"$log" 2>&1 &
  echo "$! $node/$devices $experiment"
}

launch_after() {
  local sentinel="$1"; shift
  local job="$1" node="$2" devices="$3" experiment="$4" profile="$5"
  local strength="$6" replay="$7" lr="$8"
  local output="$release/train_$experiment"
  local log="$logs/${experiment}_${node}.log"
  test ! -e "$output"
  test ! -e "$log"
  nohup bash -lc "
    set -euo pipefail
    while [ ! -f '$sentinel' ]; do sleep 10; done
    exec srun --jobid='$job' --overlap --nodes=1 --ntasks=1 --nodelist='$node' \\
      env ONLINE_ANCHOR_PROFILE='$profile' \\
          ONLINE_ANCHOR_VISIBLE_DEVICES='$devices' \\
          ONLINE_ANCHOR_EXPERIMENT='$experiment' \\
          ONLINE_ANCHOR_MAX_STEPS=96 \\
          ONLINE_ANCHOR_ROUTE_OPERATOR=self_correspondence_kernel25 \\
          ONLINE_ANCHOR_ROUTE_STRENGTH='$strength' \\
          ONLINE_ANCHOR_REPLAY_WEIGHT='$replay' \\
          ONLINE_ANCHOR_LEARNING_RATE='$lr' \\
          ONLINE_ANCHOR_METHOD_ARCHIVE='$archive' \\
          bash '$runner'
  " >"$log" 2>&1 &
  echo "$! waits on $sentinel then $node/$devices $experiment"
}

launch 143808 auh7-1b-gpu-233 0,1,2,3 corr25_actionnoop_r025_replay025_lr1e5_s96_v5 action_noop 0.25 0.25 1e-5
launch 143808 auh7-1b-gpu-233 4,5,6,7 corr25_actionnoop_r050_replay025_lr1e5_s96_v5 action_noop 0.50 0.25 1e-5
launch 143808 auh7-1b-gpu-292 0,1,2,3 corr25_actionnoop_r100_replay025_lr1e5_s96_v5 action_noop 1.00 0.25 1e-5
launch 143812 auh7-1b-gpu-293 0,1,2,3 corr25_actionnoop_r050_replay050_lr1e5_s96_v5 action_noop 0.50 0.50 1e-5
launch 143811 auh7-1b-gpu-306 0,1,2,3 corr25_actionnoop_r050_replay010_lr1e5_s96_v5 action_noop 0.50 0.10 1e-5

decode_root="$release/dynaedit_fullgrid_v2"
launch_after "$decode_root/hard25_actionnoop_r025_replay025_lr1e5_s64_v4/step_00000064/e04/E04_close-door-then-drawer_hard25_actionnoop_r025_replay025_lr1e5_s64_v4_S64_ONLINE_ANCHOR_REAL_SGA_ANC.mp4" \
  141620 auh7-1b-gpu-226 0,1,2,3 corr25_dynamicstatic_r050_replay025_lr1e5_s96_v5 dynamic_static 0.50 0.25 1e-5
launch_after "$decode_root/kernel_actionnoop_r025_replay025_lr1e5_s64_v4/step_00000064/e04/E04_close-door-then-drawer_kernel_actionnoop_r025_replay025_lr1e5_s64_v4_S64_ONLINE_ANCHOR_REAL_SGA_ANC.mp4" \
  143808 auh7-1b-gpu-268 0,1,2,3 corr25_hybrid_r050_replay025_lr1e5_s96_v5 hybrid 0.50 0.25 1e-5
launch_after "$decode_root/kernel_dynamicstatic_r025_replay025_lr1e5_s64_v4/step_00000064/e04/E04_close-door-then-drawer_kernel_dynamicstatic_r025_replay025_lr1e5_s64_v4_S64_ONLINE_ANCHOR_REAL_SGA_ANC.mp4" \
  143808 auh7-1b-gpu-315 0,1,2,3 corr25_actionnoop_r050_replay025_lr2e5_s96_v5 action_noop 0.50 0.25 2e-5
