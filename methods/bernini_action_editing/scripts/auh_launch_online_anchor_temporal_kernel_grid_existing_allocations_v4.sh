#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this controller must run on the AUH login host" >&2; exit 2 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
runner="$stage/source-online-anchor-attention-training-v1/methods/bernini_action_editing/scripts/auh_train_online_anchor_attention_fullgrid_existing_allocations_v2.sh"
archive="$release/online-anchor-temporal-kernel-v4.tar"
revision="$release/online-anchor-temporal-kernel-v4.revision"
log_root="$release/logs/kernel_v4_grid"
mkdir -p "$log_root"
test -f "$runner"
test -f "$archive"
test -f "$revision"

launch() {
  local job="$1" node="$2" devices="$3" experiment="$4"
  local operator="$5" strength="$6" replay="$7" profile="$8"
  local output="$release/train_${experiment}"
  local log="$log_root/${experiment}_${node}.log"
  test ! -e "$output"
  test ! -e "$log"
  nohup srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
    env \
      ONLINE_ANCHOR_PROFILE="$profile" \
      ONLINE_ANCHOR_ROUTE_OPERATOR="$operator" \
      ONLINE_ANCHOR_VISIBLE_DEVICES="$devices" \
      ONLINE_ANCHOR_EXPERIMENT="$experiment" \
      ONLINE_ANCHOR_MAX_STEPS=64 \
      ONLINE_ANCHOR_ROUTE_STRENGTH="$strength" \
      ONLINE_ANCHOR_REPLAY_WEIGHT="$replay" \
      ONLINE_ANCHOR_LEARNING_RATE=1e-5 \
      ONLINE_ANCHOR_METHOD_ARCHIVE="$archive" \
      ONLINE_ANCHOR_METHOD_REVISION_FILE="$revision" \
      bash "$runner" >"$log" 2>&1 &
  echo "$! $job $node $devices $experiment $operator strength=$strength replay=$replay profile=$profile"
}

launch 143808 auh7-1b-gpu-233 0,1,2,3 kernel_actionnoop_r025_replay025_lr1e5_s64_v4 self_temporal_kernel 0.25 0.25 action_noop
launch 143808 auh7-1b-gpu-268 0,1,2,3 kernel_actionnoop_r050_replay025_lr1e5_s64_v4 self_temporal_kernel 0.50 0.25 action_noop
launch 143808 auh7-1b-gpu-292 0,1,2,3 kernel_actionnoop_r025_replay050_lr1e5_s64_v4 self_temporal_kernel 0.25 0.50 action_noop
launch 143808 auh7-1b-gpu-315 0,1,2,3 kernel_dynamicstatic_r025_replay025_lr1e5_s64_v4 self_temporal_kernel 0.25 0.25 dynamic_static
launch 141620 auh7-1b-gpu-226 0,1,2,3 hard25_actionnoop_r025_replay025_lr1e5_s64_v4 self_target_gated_kernel25 0.25 0.25 action_noop
launch 143812 auh7-1b-gpu-293 0,1,2,3 hard25_actionnoop_r050_replay025_lr1e5_s64_v4 self_target_gated_kernel25 0.50 0.25 action_noop
launch 143811 auh7-1b-gpu-306 0,1,2,3 hard25_actionnoop_r100_replay025_lr1e5_s64_v4 self_target_gated_kernel25 1.00 0.25 action_noop
launch 143808 auh7-1b-gpu-233 4,5,6,7 hard25_hybrid_r050_replay025_lr1e5_s64_v4 self_target_gated_kernel25 0.50 0.25 hybrid
