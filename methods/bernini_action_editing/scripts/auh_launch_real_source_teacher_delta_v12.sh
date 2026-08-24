#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this controller must run on the AUH login host" >&2; exit 2 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
source_tree="$stage/source-online-anchor-real-source-teacher-delta-v12"
runner="$source_tree/methods/bernini_action_editing/scripts/auh_train_online_anchor_attention_fullgrid_existing_allocations_v2.sh"
authoring="$source_tree/methods/bernini_action_editing/assets/interaction_complex8_multianchor_authoring_v2.json"
real_source_manifest="$release/complex8_real_source_latents_v12/manifest.json"
archive="$release/online-anchor-real-source-teacher-delta-v12.tar"
revision="$release/online-anchor-real-source-teacher-delta-v12.revision"
log_root="$release/logs/real_source_teacher_delta_v12"
mkdir -p "$log_root"
test -f "$runner"
test -f "$authoring"
test -f "$real_source_manifest"
test -f "$archive"
test -f "$revision"
real_source_sha="$(sha256sum "$real_source_manifest" | awk '{print $1}')"

launch() {
  local job="$1" node="$2" experiment="$3" route_operator="$4"
  local teacher_mode="$5" max_steps="$6"
  local output="$release/train_${experiment}"
  local log="$log_root/${experiment}_${node}.log"
  test ! -e "$output"
  test ! -e "$log"
  srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --cpus-per-task=64 \
    --gres=gpu:4 --mem=0 --nodelist="$node" \
    env \
      ONLINE_ANCHOR_SOURCE_TREE="$source_tree" \
      ONLINE_ANCHOR_PROFILE=action_noop \
      ONLINE_ANCHOR_ROUTE_OPERATOR="$route_operator" \
      ONLINE_ANCHOR_VISIBLE_DEVICES=0,1,2,3 \
      ONLINE_ANCHOR_EXPERIMENT="$experiment" \
      ONLINE_ANCHOR_MAX_STEPS="$max_steps" \
      ONLINE_ANCHOR_SOURCE_VARIANT=mixed \
      ONLINE_ANCHOR_ROUTE_STRENGTH=0.25 \
      ONLINE_ANCHOR_TRAINING_OBJECTIVE=real_source_teacher_delta \
      ONLINE_ANCHOR_TRAINING_INTERFACE=first_phase_caption_i2v \
      ONLINE_ANCHOR_AUTHORING="$authoring" \
      ONLINE_ANCHOR_REAL_SOURCE_MANIFEST="$real_source_manifest" \
      ONLINE_ANCHOR_REAL_SOURCE_MANIFEST_SHA256="$real_source_sha" \
      ONLINE_ANCHOR_TEACHER_DELTA_MODE="$teacher_mode" \
      ONLINE_ANCHOR_PAIRED_TARGET_FM_WEIGHT=0 \
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

smoke=cross_realteacher_p0rel_replay025_s1_smoke_v12
launch 143808 auh7-1b-gpu-268 "$smoke" cross_sparse phase0_relative 1
receipt="$release/train_${smoke}/checkpoint-00000001/receipt.json"
jq -e '
  .complete == true and
  .training_contract.training_objective == "real_source_teacher_delta" and
  .training_contract.student_clean_target_is_complete_real_source == true and
  .training_contract.synthetic_clean_target_flow_matching_weight == 0 and
  .training_contract.t2v_teacher_is_same_noisy_state_action_minus_noop_full_tensor == true and
  .training_contract.teacher_tensor_reduced_to_low_dimensional_statistic == false and
  .training_contract.true_training_memory_fraction_strictly_above_half == true and
  .training_contract.dummy_or_padding_allocations == false and
  .gradient_coverage.nonzero_tensor_count > 0
' "$receipt" >/dev/null

launch_background() {
  local job="$1" node="$2" experiment="$3" route_operator="$4"
  local teacher_mode="$5" max_steps="$6"
  local controller_log="$log_root/controller_${experiment}_${node}.log"
  test ! -e "$controller_log"
  nohup bash -lc "
    set -euo pipefail
    bash '$(realpath "$0")' _one '$job' '$node' '$experiment' '$route_operator' '$teacher_mode' '$max_steps'
  " >"$controller_log" 2>&1 &
  echo "$! $job $node $experiment $route_operator $teacher_mode"
}

launch_background 143808 auh7-1b-gpu-233 \
  cross_realteacher_raw_replay025_s32_v12 cross_sparse raw 32
launch_background 143808 auh7-1b-gpu-268 \
  cross_realteacher_p0rel_replay025_s32_v12 cross_sparse phase0_relative 32
launch_background 143808 auh7-1b-gpu-292 \
  targetgate_realteacher_p0rel_replay025_s32_v12 self_target_gated_kernel25 phase0_relative 32
launch_background 143808 auh7-1b-gpu-315 \
  temporalkernel_realteacher_p0rel_replay025_s32_v12 self_temporal_kernel phase0_relative 32
