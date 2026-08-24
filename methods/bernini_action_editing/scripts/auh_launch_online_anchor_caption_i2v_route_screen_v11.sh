#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this controller must run on the AUH login host" >&2; exit 2 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
source_tree="$stage/source-online-anchor-attention-training-v1"
runner="$source_tree/methods/bernini_action_editing/scripts/auh_train_online_anchor_attention_fullgrid_existing_allocations_v2.sh"
authoring="$source_tree/methods/bernini_action_editing/assets/interaction_complex8_multianchor_authoring_v2.json"
archive="$release/online-anchor-caption-i2v-route-screen-v11.tar"
revision="$release/online-anchor-caption-i2v-route-screen-v11.revision"
log_root="$release/logs/caption_i2v_route_screen_v11"
mkdir -p "$log_root"
test -f "$runner"
test -f "$authoring"
test -f "$archive"
test -f "$revision"

launch() {
  local job="$1" node="$2" experiment="$3" route_operator="$4"
  local target_weight="$5" max_steps="$6"
  local output="$release/train_${experiment}"
  local log="$log_root/${experiment}_${node}.log"
  test ! -e "$output"
  test ! -e "$log"
  srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
    env \
      ONLINE_ANCHOR_PROFILE=action_noop \
      ONLINE_ANCHOR_ROUTE_OPERATOR="$route_operator" \
      ONLINE_ANCHOR_VISIBLE_DEVICES=0,1,2,3 \
      ONLINE_ANCHOR_EXPERIMENT="$experiment" \
      ONLINE_ANCHOR_MAX_STEPS="$max_steps" \
      ONLINE_ANCHOR_SOURCE_VARIANT=mixed \
      ONLINE_ANCHOR_ROUTE_STRENGTH=0.25 \
      ONLINE_ANCHOR_TRAINING_OBJECTIVE=paired_delta_fm \
      ONLINE_ANCHOR_TRAINING_INTERFACE=first_phase_caption_i2v \
      ONLINE_ANCHOR_AUTHORING="$authoring" \
      ONLINE_ANCHOR_PAIRED_TARGET_FM_WEIGHT="$target_weight" \
      ONLINE_ANCHOR_REPLAY_WEIGHT=0.25 \
      ONLINE_ANCHOR_REPLAY_PROMPT=identity \
      ONLINE_ANCHOR_LEARNING_RATE=1e-5 \
      ONLINE_ANCHOR_METHOD_ARCHIVE="$archive" \
      ONLINE_ANCHOR_METHOD_REVISION_FILE="$revision" \
      bash "$runner" >"$log" 2>&1
}

if [ "${1:-}" = _one ]; then
  launch "$2" "$3" "$4" "$5" "$6" "$7"
  exit 0
fi

smoke=cross_captioni2v_delta_tf025_replay025_s1_smoke_v11
launch 141620 auh7-1b-gpu-226 "$smoke" cross_sparse 0.25 1
receipt="$release/train_${smoke}/checkpoint-00000001/receipt.json"
jq -e '
  .complete == true and
  .training_contract.training_interface == "first_phase_caption_i2v" and
  .training_contract.route_operator == "cross_sparse" and
  .training_contract.renderer_training_source_name == "t2v$action_editing_81f" and
  .training_contract.online_anchor_uses_t2v_system_prompt == true and
  .training_contract.true_training_memory_fraction_strictly_above_half == true and
  .training_contract.dummy_or_padding_allocations == false and
  .gradient_coverage.nonzero_tensor_count > 0
' "$receipt" >/dev/null

launch_background() {
  local job="$1" node="$2" experiment="$3" route_operator="$4"
  local target_weight="$5" max_steps="$6"
  nohup bash -lc "
    set -euo pipefail
    bash '$(realpath "$0")' _one '$job' '$node' '$experiment' '$route_operator' '$target_weight' '$max_steps'
  " >"$log_root/controller_${experiment}_${node}.log" 2>&1 &
  echo "$! $job $node $experiment $route_operator"
}

# This is an operator screen, not another weight sweep.  S32 covers every one
# of the 32 Complex8 action targets exactly once; promote only a visually
# credible operator to a longer run.
launch_background 143808 auh7-1b-gpu-233 cross_captioni2v_delta_tf025_replay025_s32_v11 cross_sparse 0.25 32
launch_background 143808 auh7-1b-gpu-268 cross_captioni2v_delta_pure_replay025_s32_v11 cross_sparse 0.0 32
launch_background 143808 auh7-1b-gpu-315 temporalkernel_captioni2v_delta_tf025_replay025_s32_v11 self_temporal_kernel 0.25 32
launch_background 141620 auh7-1b-gpu-226 targetgate_captioni2v_delta_tf025_replay025_s32_v11 self_target_gated_kernel25 0.25 32
