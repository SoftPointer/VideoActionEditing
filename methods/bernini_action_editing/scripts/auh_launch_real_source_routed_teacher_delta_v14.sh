#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this controller must run on the AUH login host" >&2; exit 2 ;;
esac

job="${ONLINE_ANCHOR_JOB:-143808}"
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
source_tree="$stage/source-online-anchor-targetcoord-routed-teacher-v14"
runner="$source_tree/methods/bernini_action_editing/scripts/auh_train_online_anchor_attention_fullgrid_existing_allocations_v2.sh"
authoring="$source_tree/methods/bernini_action_editing/assets/interaction_complex8_multianchor_authoring_v2.json"
real_source_manifest="$release/complex8_real_source_latents_v13/manifest.json"
archive="$release/online-anchor-targetcoord-routed-teacher-v14.tar"
revision="$release/online-anchor-targetcoord-routed-teacher-v14.revision"
log_root="$release/logs/real_source_routed_teacher_delta_v14"
mkdir -p "$log_root"
test -f "$runner"
test -f "$authoring"
test -f "$real_source_manifest"
test -f "$archive"
test -f "$revision"
real_source_sha="$(sha256sum "$real_source_manifest" | awk '{print $1}')"
test "$real_source_sha" = 8b0a9d7fd8ccc9d8b555a66f8efe6d2e5d91880f81f5e7b892e123d88235bc63

launch() {
  local node="$1" experiment="$2" route_operator="$3"
  local teacher_strength="$4" max_steps="$5"
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
      ONLINE_ANCHOR_TEACHER_ROUTE_STRENGTH="$teacher_strength" \
      ONLINE_ANCHOR_TRAINING_OBJECTIVE=real_source_routed_teacher_delta \
      ONLINE_ANCHOR_TRAINING_INTERFACE=first_phase_caption_i2v \
      ONLINE_ANCHOR_AUTHORING="$authoring" \
      ONLINE_ANCHOR_REAL_SOURCE_MANIFEST="$real_source_manifest" \
      ONLINE_ANCHOR_REAL_SOURCE_MANIFEST_SHA256="$real_source_sha" \
      ONLINE_ANCHOR_TEACHER_DELTA_MODE=raw \
      ONLINE_ANCHOR_PAIRED_TARGET_FM_WEIGHT=0 \
      ONLINE_ANCHOR_REPLAY_WEIGHT=0.025 \
      ONLINE_ANCHOR_REPLAY_PROMPT=action \
      ONLINE_ANCHOR_LEARNING_RATE=1e-5 \
      ONLINE_ANCHOR_METHOD_ARCHIVE="$archive" \
      ONLINE_ANCHOR_METHOD_REVISION_FILE="$revision" \
      bash "$runner" >"$log" 2>&1
}

if [ "${1:-}" = _one ]; then
  launch "$2" "$3" "$4" "$5" "$6"
  exit 0
fi

smoke=temporalkernel_targetcoord_t050_s025_replay0025_s1_smoke_v14
launch auh7-1b-gpu-268 "$smoke" self_temporal_kernel 0.50 1
receipt="$release/train_${smoke}/checkpoint-00000001/receipt.json"
jq -e '
  .complete == true and
  .training_contract.training_objective == "real_source_routed_teacher_delta" and
  .training_contract.route_operator == "self_temporal_kernel" and
  .training_contract.target_coordinate_routed_teacher == true and
  .training_contract.anchor_only_captures_route == true and
  .training_contract.anchor_model_velocity_used_as_supervision == false and
  .training_contract.anchor_value_pixel_latent_or_spatial_coordinate_copied == false and
  .training_contract.target_value_stream_is_sole_routed_content == true and
  .training_contract.student_route_strength == 0.25 and
  .training_contract.teacher_route_strength == 0.5 and
  .training_contract.anchor_route_replay_uses_per_capture == 2 and
  .training_contract.teacher_delta_mode == "raw" and
  .training_contract.synthetic_clean_target_flow_matching_weight == 0 and
  .training_contract.source_reconstruction_weight == 0.025 and
  .training_contract.source_reconstruction_prompt == "source_caption" and
  .training_contract.real_source_action_and_source_share_exact_noisy_tensor == true and
  .training_contract.anchor_and_real_source_noise_deliberately_unbound == true and
  .training_contract.true_training_memory_fraction_strictly_above_half == true and
  .training_contract.dummy_or_padding_allocations == false and
  .gradient_coverage.nonzero_tensor_count > 0 and
  .anchor_cache.pending_entries == 0 and
  .anchor_cache.replay_count == (2 * .anchor_cache.capture_count)
' "$receipt" >/dev/null

launch_background() {
  local node="$1" experiment="$2" route_operator="$3"
  local teacher_strength="$4" max_steps="$5"
  local controller_log="$log_root/controller_${experiment}_${node}.log"
  test ! -e "$controller_log"
  nohup bash -lc "
    set -euo pipefail
    bash '$(realpath "$0")' _one '$node' '$experiment' '$route_operator' '$teacher_strength' '$max_steps'
  " >"$controller_log" 2>&1 &
  echo "$! $job $node $experiment $route_operator $teacher_strength"
}

launch_background auh7-1b-gpu-233 \
  temporalkernel_targetcoord_t050_s025_replay0025_s32_v14 \
  self_temporal_kernel 0.50 32
launch_background auh7-1b-gpu-268 \
  temporalkernel_targetcoord_t100_s025_replay0025_s32_v14 \
  self_temporal_kernel 1.00 32
launch_background auh7-1b-gpu-292 \
  targetgate_targetcoord_t050_s025_replay0025_s32_v14 \
  self_target_gated_kernel25 0.50 32
launch_background auh7-1b-gpu-315 \
  corr25_targetcoord_t050_s025_replay0025_s32_v14 \
  self_correspondence_kernel25 0.50 32
