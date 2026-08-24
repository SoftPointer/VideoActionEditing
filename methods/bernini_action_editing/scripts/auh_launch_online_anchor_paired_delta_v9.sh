#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this controller must run on the AUH login host" >&2; exit 2 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
runner="$stage/source-online-anchor-attention-training-v1/methods/bernini_action_editing/scripts/auh_train_online_anchor_attention_fullgrid_existing_allocations_v2.sh"
archive="$release/online-anchor-paired-delta-fullfield-v9.tar"
revision="$release/online-anchor-paired-delta-fullfield-v9.revision"
log_root="$release/logs/paired_delta_v9"
mkdir -p "$log_root"
test -f "$runner"
test -f "$archive"
test -f "$revision"

launch() {
  local job="$1" node="$2" experiment="$3" profile="$4"
  local source_variant="$5" target_weight="$6" max_steps="$7"
  local output="$release/train_${experiment}"
  local log="$log_root/${experiment}_${node}.log"
  test ! -e "$output"
  test ! -e "$log"
  srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
    env \
      ONLINE_ANCHOR_PROFILE="$profile" \
      ONLINE_ANCHOR_ROUTE_OPERATOR=self_correspondence_kernel25 \
      ONLINE_ANCHOR_VISIBLE_DEVICES=0,1,2,3 \
      ONLINE_ANCHOR_EXPERIMENT="$experiment" \
      ONLINE_ANCHOR_MAX_STEPS="$max_steps" \
      ONLINE_ANCHOR_SOURCE_VARIANT="$source_variant" \
      ONLINE_ANCHOR_ROUTE_STRENGTH=0.25 \
      ONLINE_ANCHOR_TRAINING_OBJECTIVE=paired_delta_fm \
      ONLINE_ANCHOR_PAIRED_TARGET_FM_WEIGHT="$target_weight" \
      ONLINE_ANCHOR_REPLAY_WEIGHT=0.25 \
      ONLINE_ANCHOR_REPLAY_PROMPT=identity \
      ONLINE_ANCHOR_LEARNING_RATE=1e-5 \
      ONLINE_ANCHOR_METHOD_ARCHIVE="$archive" \
      ONLINE_ANCHOR_METHOD_REVISION_FILE="$revision" \
      bash "$runner" >"$log" 2>&1
}

if [ "${1:-}" = _one ]; then
  launch "$2" "$3" "$4" "$5" "$6" "$7" 64
  exit 0
fi

smoke=corr25_paireddelta_r025_tf025_replay025_s1_smoke_v9
smoke_receipt="$release/train_${smoke}/checkpoint-00000001/receipt.json"
if [ "${ONLINE_ANCHOR_SKIP_SMOKE:-0}" != 1 ]; then
  launch 141620 auh7-1b-gpu-226 "$smoke" action_noop counterfactual4 0.25 1
fi
test -f "$smoke_receipt"
jq -e '
  .complete == true and
  .training_contract.training_objective == "paired_delta_fm" and
  .training_contract.objective_tensor_scope == "complete_spatiotemporal_video_velocity_field" and
  .training_contract.paired_delta_has_gradients_through_both_model_queries == true and
  .training_contract.true_training_memory_fraction_strictly_above_half == true and
  .training_contract.dummy_or_padding_allocations == false and
  .gradient_coverage.nonzero_tensor_count > 0
' "$smoke_receipt" >/dev/null

launch_background() {
  local job="$1" node="$2" experiment="$3" profile="$4"
  local source_variant="$5" target_weight="$6"
  nohup bash -lc "
    set -euo pipefail
    bash '$(realpath "$0")' _one '$job' '$node' '$experiment' '$profile' '$source_variant' '$target_weight'
  " >"$log_root/controller_${experiment}_${node}.log" 2>&1 &
  echo "$! $job $node $experiment $profile $source_variant target_fm_weight=$target_weight"
}

launch_background 143808 auh7-1b-gpu-233 corr25_paireddelta_cf4_tf025_replay025_s64_v9 action_noop counterfactual4 0.25
launch_background 143808 auh7-1b-gpu-268 noanchor_paireddelta_cf4_tf025_replay025_s64_v9 no_anchor counterfactual4 0.25
launch_background 143808 auh7-1b-gpu-315 corr25_paireddelta_mixed_tf025_replay025_s64_v9 action_noop mixed 0.25
launch_background 141620 auh7-1b-gpu-226 corr25_paireddelta_cf4_puredelta_replay025_s64_v9 action_noop counterfactual4 0.0
