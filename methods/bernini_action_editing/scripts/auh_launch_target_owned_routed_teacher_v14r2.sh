#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this controller must run on the AUH login host" >&2; exit 2 ;;
esac

job="${ONLINE_ANCHOR_JOB:-143808}"
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
source_tree="$stage/source-online-anchor-targetowned-qk-routed-teacher-v14r3-gradgeom"
runner="$source_tree/methods/bernini_action_editing/scripts/auh_train_online_anchor_attention_fullgrid_existing_allocations_v2.sh"
authoring="$source_tree/methods/bernini_action_editing/assets/interaction_complex8_multianchor_authoring_v2.json"
real_source_manifest="$release/complex8_real_source_latents_v13/manifest.json"
archive="$release/online-anchor-targetowned-qk-routed-teacher-v14r3-gradgeom.tar"
revision="$release/online-anchor-targetowned-qk-routed-teacher-v14r3-gradgeom.revision"
content_manifest="$release/online-anchor-targetowned-qk-routed-teacher-v14r3-gradgeom.content.json"
deployment_marker="$release/DEPLOYMENT_TESTS_PASS_targetcoord_v14r3_gradgeom.json"
deployment_validator="$source_tree/methods/bernini_action_editing/validate_v14r2_deployment_marker.py"
validator_python=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
log_root="$release/logs/target_owned_routed_teacher_v14r3_gradgeom"

test -f "$runner"
test -f "$authoring"
test -f "$real_source_manifest"
test -f "$archive"
test -f "$revision"
test -f "$content_manifest"
test -f "$deployment_marker"
test -f "$deployment_validator"
test ! -L "$deployment_validator"
test -x "$validator_python"
"$validator_python" -B "$deployment_validator" \
  --marker "$deployment_marker" --role training \
  --source-tree "$source_tree" --archive "$archive" --revision "$revision" \
  --content-manifest "$content_manifest" --min-test-count 129 \
  --required-file methods/bernini_action_editing/train_online_anchor_attention_v1.py \
  --required-file methods/bernini_action_editing/anchor_qk_transport.py \
  --required-file methods/bernini_action_editing/anchor_cross_attention_transport.py \
  --required-file methods/bernini_action_editing/anchor_sga_anc_controller.py \
  --required-file methods/bernini_action_editing/infer_anchor_sga_anc_event_v1.py \
  --required-file methods/bernini_action_editing/materialize_complex8_real_source_latents_v1.py \
  --required-file methods/bernini_action_editing/scripts/auh_train_online_anchor_attention_fullgrid_existing_allocations_v2.sh \
  --required-file methods/bernini_action_editing/scripts/auh_launch_target_owned_routed_teacher_v14r2.sh \
  --required-file methods/bernini_action_editing/validate_v14r2_deployment_marker.py \
  --required-file methods/bernini_action_editing/tests/test_train_online_anchor_attention_v1.py \
  --required-file methods/bernini_action_editing/tests/test_anchor_qk_transport.py \
  --required-file methods/bernini_action_editing/tests/test_infer_anchor_sga_anc_event_rank0_prompt_bank_v1.py \
  --required-file methods/bernini_action_editing/tests/test_validate_v14r2_deployment_marker.py
real_source_sha="$(sha256sum "$real_source_manifest" | awk '{print $1}')"
test "$real_source_sha" = 8b0a9d7fd8ccc9d8b555a66f8efe6d2e5d91880f81f5e7b892e123d88235bc63
mkdir -p "$log_root"

launch() {
  local node="$1" experiment="$2" route_operator="$3"
  local teacher_mode="$4" combine_mode="$5" max_steps="$6"
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
      ONLINE_ANCHOR_ROUTE_STRENGTH=0.25 \
      ONLINE_ANCHOR_TEACHER_ROUTE_STRENGTH=0.50 \
      ONLINE_ANCHOR_TRAINING_OBJECTIVE=real_source_target_owned_routed_teacher_delta_v14r2 \
      ONLINE_ANCHOR_TRAINING_INTERFACE=first_phase_caption_i2v \
      ONLINE_ANCHOR_ROUTED_TEACHER_MODE="$teacher_mode" \
      ONLINE_ANCHOR_REPLAY_COMBINE_MODE="$combine_mode" \
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

gate_smoke() {
  local receipt="$1" expected_step="$2" route_operator="$3"
  local teacher_mode="$4" combine_mode="$5"
  local expected_transport
  case "$route_operator" in
    self_target_owned_temporal_kernel_v14r2)
      expected_transport=self_target_owned_temporal_kernel_attn_output_v14r2 ;;
    self_target_owned_activity_kernel10_v14r2)
      expected_transport=self_target_owned_activity_kernel10_attn_output_v14r2 ;;
    self_target_owned_activity_kernel25_v14r2)
      expected_transport=self_target_owned_activity_kernel25_attn_output_v14r2 ;;
    *) return 31 ;;
  esac
  jq -e \
    --arg route "$route_operator" \
    --arg transport "$expected_transport" \
    --arg teacher_mode "$teacher_mode" \
    --arg combine_mode "$combine_mode" \
    --argjson expected_step "$expected_step" '
    def packed_state_diagnostic_ok:
      .schema_version == "bernini-real-source-prebind-packed-state-v1" and
      .raw_same_seed_state_exact == true and
      .raw_same_seed_unequal_fields == [] and
      .raw_same_seed_exact_by_field == {
        "input_vae_latents": true,
        "input_vae_rope": true,
        "target_lens": true,
        "target_velocity": true,
        "timesteps": true,
        "vae_latents_mask": true,
        "vae_seqlen": true
      } and
      (.packed_geometry as $g |
        ($g.unpacked_spatial_shape | length) == 5 and
        $g.unpacked_spatial_shape[0] == 1 and
        $g.unpacked_spatial_shape[1] == 16 and
        $g.unpacked_spatial_shape[2] == 21 and
        $g.unpacked_spatial_shape[3] > 0 and
        $g.unpacked_spatial_shape[4] > 0 and
        ($g.unpacked_spatial_shape[3] % 2) == 0 and
        ($g.unpacked_spatial_shape[4] % 2) == 0 and
        $g.latent_phases == 21 and
        $g.tokens_per_video == (
          21 * ($g.unpacked_spatial_shape[3] / 2) *
          ($g.unpacked_spatial_shape[4] / 2)
        ) and
        $g.packed_total_tokens == (2 * $g.tokens_per_video) and
        $g.source_token_count == $g.tokens_per_video and
        $g.target_token_count == $g.tokens_per_video and
        $g.input_vae_latents_shape == [$g.packed_total_tokens, 16, 1, 2, 2] and
        $g.input_vae_rope_shape == [$g.packed_total_tokens, 1, 64] and
        $g.vae_latents_mask_shape == [1, $g.packed_total_tokens] and
        $g.selector_transition_count == 1 and
        $g.vae_seqlen == $g.packed_total_tokens and
        $g.timestep_range_inclusive == [0, 1000] and
        $g.timestep_value >= 0 and $g.timestep_value <= 1000 and
        $g.target_velocity_shape == [$g.tokens_per_video, 16, 1, 2, 2] and
        $g.target_lens == $g.tokens_per_video and
        $g.state_fields.input_vae_latents == {
          "shape": [$g.packed_total_tokens, 16, 1, 2, 2], "dtype": "float32"
        } and
        $g.state_fields.input_vae_rope == {
          "shape": [$g.packed_total_tokens, 1, 64], "dtype": "complex128"
        } and
        $g.state_fields.vae_latents_mask == {
          "shape": [1, $g.packed_total_tokens], "dtype": "bool"
        } and
        $g.state_fields.vae_seqlen == {
          "shape": [1, 1], "dtype": "int64"
        } and
        $g.state_fields.timesteps == {
          "shape": [1, 1], "dtype": "bfloat16"
        } and
        $g.state_fields.target_velocity == {
          "shape": [$g.tokens_per_video, 16, 1, 2, 2], "dtype": "float32"
        } and
        $g.state_fields.target_lens == {
          "shape": [1, 1], "dtype": "int64"
        });
    def near($actual; $expected; $tolerance):
      (($actual - $expected) | abs) <= $tolerance;
    def combine_geometry_ok($mode):
      . as $i |
      $i.replay_combine_mode == $mode and
      $i.action_l2_norm_fp64 > 1e-12 and
      $i.raw_replay_l2_norm_fp64 > 1e-12 and
      $i.combined_l2_norm_fp64 > 1e-12 and
      $i.action_gradient_dot_combined_gradient_fp64 > 0 and
      $i.action_alignment_ratio > 0 and
      $i.effective_replay_scale >= 0 and
      $i.correction_ratio_q >= 0 and
      $i.weighted_replay_gradient_fraction >= 0 and
      $i.weighted_replay_gradient_fraction < 1 and
      (if $mode == "action_only" then
         near($i.effective_replay_scale; 0; 1e-12) and
         near($i.correction_ratio_q; 0; 1e-12) and
         near($i.weighted_replay_gradient_fraction; 0; 1e-12) and
         near($i.action_alignment_ratio; 1; 1e-5) and
         $i.replay_projection_applied == false
       elif $mode == "norm_balanced_025" then
         $i.effective_replay_scale > 0 and
         near($i.correction_ratio_q; 0.25; 1e-5) and
         near($i.weighted_replay_gradient_fraction; 0.20; 1e-5) and
         $i.action_alignment_ratio >= 0.75 and
         $i.first_order_source_fm_preserved == true and
         $i.raw_replay_gradient_dot_combined_gradient_fp64 >= -1e-8
       elif $mode == "action_priority_pcgrad_010" then
         $i.effective_replay_scale >= 0 and
         near($i.correction_ratio_q; 0.10; 1e-5) and
         near($i.weighted_replay_gradient_fraction; (1 / 11); 1e-5) and
         $i.action_alignment_ratio >= 0.99 and
         $i.action_priority_conflict_control_not_source_preservation == true and
         (if $i.replay_projection_applied then
            ($i.processed_replay_action_cosine | abs) <= 1e-5 and
            $i.processed_replay_retained_raw_norm_fraction >= 0.20
          else
            $i.action_replay_cosine >= 0
          end)
       elif $mode == "source_halfspace_001" then
         $i.effective_replay_scale > 0 and
         $i.correction_ratio_q >= 0.01 and
         $i.correction_ratio_q <= 1.01 and
         $i.action_alignment_ratio >= 0.10 and
         $i.first_order_source_fm_preserved == true and
         $i.raw_replay_combined_alignment_over_action_replay_norms >= 0.009
       else false
       end);
    def actual_optimizer_update_ok($mode; $step):
      .schema_version == "bernini-actual-optimizer-update-probe-v1" and
      .step == $step and
      .replay_combine_mode == $mode and
      .gradient_scope == "separately_allreduced_global_action_and_raw_replay" and
      .optimizer_semantics_observed_not_modified == true and
      .parameter_snapshot_native_dtype == true and
      (.parameter_snapshot_dtypes | length) >= 1 and
      .tensor_count == 480 and
      .parameter_element_count == 188743680 and
      .changed_element_count > 0 and
      .delta_theta_l2_norm_fp64 > 0 and
      .action_descent_required == true and
      .action_descent_passed == true and
      .action_descent_fp64 > 0 and
      (($step == 1 and .changed_tensor_count == 240) or
       ($step == 2 and .changed_tensor_count == 480)) and
      (if ($mode == "norm_balanced_025" or $mode == "source_halfspace_001") then
         .source_descent_required == true and
         .source_descent_passed == true and
         .source_descent_fp64 >= .minimum_allowed_source_descent_fp64
       else
         .source_descent_required == false
       end);
    .complete == true and
    .schema_version == "bernini-online-anchor-attention-training-receipt-v3" and
    .global_step == $expected_step and
    .training_contract.training_objective == "real_source_target_owned_routed_teacher_delta_v14r2" and
    .training_contract.route_operator == $route and
    .training_contract.route_transport == $transport and
    .training_contract.routed_teacher_mode == $teacher_mode and
    .training_contract.replay_combine_mode == $combine_mode and
    .training_contract.target_owned_qk_route_v14r2 == true and
    .training_contract.target_coordinate_routed_teacher == true and
    .training_contract.anchor_only_captures_route == true and
    .training_contract.anchor_model_velocity_used_as_supervision == false and
    .training_contract.anchor_donor_cached_fields == ["query", "key"] and
    .training_contract.anchor_donor_value_cached_or_used_by_route == false and
    .training_contract.anchor_donor_hidden_or_attention_output_cached_or_used_by_route == false and
    .training_contract.anchor_donor_rgb_latent_or_absolute_spatial_coordinate_used_by_route == false and
    .training_contract.anchor_qk_time_constant_caption_offset_removed_before_support_and_kernel == true and
    .training_contract.anchor_qk_phase0_only_difference_produces_zero_route == true and
    .training_contract.anchor_to_target_appearance_correspondence_used == false and
    .training_contract.target_value_stream_is_sole_routed_content == true and
    .training_contract.student_route_strength == 0.25 and
    .training_contract.teacher_route_strength == 0.5 and
    .training_contract.anchor_route_replay_uses_per_capture == 2 and
    .training_contract.teacher_delta_mode == "raw" and
    .training_contract.synthetic_clean_target_flow_matching_weight == 0 and
    .training_contract.source_reconstruction_weight == null and
    .training_contract.source_reconstruction_weight_argument == 0.025 and
    .training_contract.base_replay_scale == 0.025 and
    .training_contract.effective_replay_scale == .component_gradient_probes.interaction.effective_replay_scale and
    .training_contract.source_reconstruction_prompt == "source_caption" and
    .training_contract.source_variant_argument == "not_applicable" and
    .training_contract.real_source_variant_schedule == "complete_real_source" and
    .training_contract.micro_semantics == "different_seed_and_cross_appearance_donor" and
    .training_contract.anchor_and_real_source_noise_deliberately_unbound == true and
    .training_contract.real_source_exact_state_binding == "action_transform_vae_state_alias_source_caption_text_only" and
    .training_contract.real_source_raw_same_seed_exact_required_before_alias == true and
    .training_contract.real_source_packed_state_contract == "tokens_c_pt_ph_pw_with_21_phases_verified_from_unpacked_clean_shape" and
    .training_contract.true_training_memory_fraction_strictly_above_half == true and
    .training_contract.training_memory_gate_capture_phase == "after_two_real_component_backwards_before_actual_update_audit_clones" and
    .training_contract.actual_update_audit_allocations_excluded_from_training_memory_gate == true and
    .training_contract.dummy_or_padding_allocations == false and
    .memory_gate.capture_phase == "after_two_real_component_backwards_before_actual_update_audit_clones" and
    .memory_gate.actual_update_audit_allocations_excluded == true and
    .memory_gate.true_training_tensors_only == true and
    .memory_gate.dummy_or_padding_allocations == false and
    .memory_gate.passed == true and
    .memory_gate.minimum_reserved_fraction > 0.5 and
    all(.memory_gate.per_rank[]; .reserved_fraction > 0.5) and
    .last_loss == null and
    .last_reporting_scalar_is_not_a_joint_backpropagated_objective == true and
    .last_objective_components.action_objective_unweighted > 1e-12 and
    .last_objective_components.base_replay_scale == 0.025 and
    .last_objective_components.effective_replay_scale == .component_gradient_probes.interaction.effective_replay_scale and
    (($combine_mode == "action_only" and
      .last_objective_components.effective_source_replay_scalar_for_reporting == 0 and
      .last_objective_components.effective_source_replay_reporting_fraction == 0) or
     ($combine_mode != "action_only" and
      .last_objective_components.effective_source_replay_scalar_for_reporting > 0)) and
    .last_objective_components.effective_source_replay_reporting_fraction >= 0 and
    .last_objective_components.effective_source_replay_reporting_fraction <= 1 and
    .real_source_prebind_state.schema_version == "bernini-real-source-prebind-packed-update-v1" and
    .real_source_prebind_state.micro_count == 2 and
    .real_source_prebind_state.all_raw_same_seed_state_exact == true and
    (.real_source_prebind_state.action_branches | length) == 2 and
    (.real_source_prebind_state.replay_branches | length) == 2 and
    all(.real_source_prebind_state.action_branches[]; packed_state_diagnostic_ok) and
    all(.real_source_prebind_state.replay_branches[]; packed_state_diagnostic_ok) and
    .component_gradient_probes.action_objective.tensor_count == 480 and
    .component_gradient_probes.action_objective.l2_norm_fp64 > 1e-12 and
    .component_gradient_probes.raw_source_caption_trajectory_replay.tensor_count == 480 and
    .component_gradient_probes.raw_source_caption_trajectory_replay.l2_norm_fp64 > 1e-12 and
    (.component_gradient_probes.interaction | combine_geometry_ok($combine_mode)) and
    (.actual_optimizer_update_probe |
      actual_optimizer_update_ok($combine_mode; $expected_step)) and
    (($expected_step == 1 and
      .component_gradient_probes.action_objective.nonzero_tensor_count == 240 and
      .component_gradient_probes.action_objective.epsilon_active_tensor_count == 240 and
      .component_gradient_probes.action_objective.adapter_sides.lora_A.nonzero_tensor_count == 0 and
      .component_gradient_probes.action_objective.adapter_sides.lora_A.epsilon_active_tensor_count == 0 and
      .component_gradient_probes.action_objective.adapter_sides.lora_B.nonzero_tensor_count == 240 and
      .component_gradient_probes.action_objective.adapter_sides.lora_B.epsilon_active_tensor_count == 240) or
     ($expected_step == 2 and
      .component_gradient_probes.action_objective.nonzero_tensor_count == 480 and
      .component_gradient_probes.action_objective.epsilon_active_tensor_count == 480 and
      .component_gradient_probes.action_objective.adapter_sides.lora_A.nonzero_tensor_count == 240 and
      .component_gradient_probes.action_objective.adapter_sides.lora_A.epsilon_active_tensor_count == 240 and
      .component_gradient_probes.action_objective.adapter_sides.lora_B.nonzero_tensor_count == 240 and
      .component_gradient_probes.action_objective.adapter_sides.lora_B.epsilon_active_tensor_count == 240 and
      .component_gradient_probes.raw_source_caption_trajectory_replay.nonzero_tensor_count == 480 and
      .component_gradient_probes.raw_source_caption_trajectory_replay.epsilon_active_tensor_count == 480 and
      .component_gradient_probes.raw_source_caption_trajectory_replay.adapter_sides.lora_A.nonzero_tensor_count == 240 and
      .component_gradient_probes.raw_source_caption_trajectory_replay.adapter_sides.lora_A.epsilon_active_tensor_count == 240 and
      .component_gradient_probes.raw_source_caption_trajectory_replay.adapter_sides.lora_B.nonzero_tensor_count == 240 and
      .component_gradient_probes.raw_source_caption_trajectory_replay.adapter_sides.lora_B.epsilon_active_tensor_count == 240 and
      .gradient_coverage.tensor_count == 480 and
      .gradient_coverage.nonzero_tensor_count == 480)) and
    .anchor_cache.pending_entries == 0 and
    .anchor_cache.qk_only_capture_count == .anchor_cache.capture_count and
    .anchor_cache.qk_only_replay_count == .anchor_cache.replay_count and
    .anchor_cache.qk_only_cached_fields == ["query", "key"] and
    .anchor_cache.replay_count == (2 * .anchor_cache.capture_count) and
    (($teacher_mode == "same_action_route_only" and .source_absorption_diagnostic.applicable == false) or
     ($teacher_mode == "cross_caption_two_sided" and .source_absorption_diagnostic.applicable == true))
  ' "$receipt" >/dev/null
}

run_arm() {
  local node="$1" arm="$2" route_operator="$3"
  local teacher_mode="$4" combine_mode="$5"
  local smoke="${arm}_s2_smoke_v14r3_gradgeom"
  local short="${arm}_s8_v14r3_gradgeom"
  launch "$node" "$smoke" "$route_operator" "$teacher_mode" "$combine_mode" 2
  gate_smoke \
    "$release/train_${smoke}/checkpoint-00000001/receipt.json" 1 \
    "$route_operator" "$teacher_mode" "$combine_mode"
  gate_smoke \
    "$release/train_${smoke}/checkpoint-00000002/receipt.json" \
    2 "$route_operator" "$teacher_mode" "$combine_mode"
  launch "$node" "$short" "$route_operator" "$teacher_mode" "$combine_mode" 8
}

if [ "${1:-}" = _arm ]; then
  run_arm "$2" "$3" "$4" "$5" "$6"
  exit 0
fi

launch_arm_background() {
  local node="$1" arm="$2" route_operator="$3"
  local teacher_mode="$4" combine_mode="$5"
  local controller_log="$log_root/controller_${arm}_${node}.log"
  test ! -e "$controller_log"
  nohup bash -lc "
    set -euo pipefail
    bash '$(realpath "$0")' _arm '$node' '$arm' '$route_operator' '$teacher_mode' '$combine_mode'
  " >"$controller_log" 2>&1 &
  echo "$! $job $node $arm $route_operator $teacher_mode $combine_mode"
}

launch_arm_background auh7-1b-gpu-233 \
  sameaction_global_actiononly \
  self_target_owned_temporal_kernel_v14r2 \
  same_action_route_only action_only
launch_arm_background auh7-1b-gpu-268 \
  sameaction_global_norm025 \
  self_target_owned_temporal_kernel_v14r2 \
  same_action_route_only norm_balanced_025
launch_arm_background auh7-1b-gpu-292 \
  sameaction_gate25_pcgrad010 \
  self_target_owned_activity_kernel25_v14r2 \
  same_action_route_only action_priority_pcgrad_010
launch_arm_background auh7-1b-gpu-315 \
  sameaction_gate25_halfspace001 \
  self_target_owned_activity_kernel25_v14r2 \
  same_action_route_only source_halfspace_001
