#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this watcher must run on the AUH login host" >&2; exit 2 ;;
esac

job="${ONLINE_ANCHOR_JOB:-143808}"
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
source_tree="$stage/source-online-anchor-targetowned-qk-decode-v14r3-gradgeom-dfix2"
dev="$source_tree/methods/bernini_action_editing"
runner="$dev/scripts/auh_decode_online_anchor_attention_dynaedit_v1.sh"
bridge_runner="$dev/scripts/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh"
watcher="$dev/scripts/auh_watch_decode_target_owned_routed_teacher_v14r2.sh"
sidecar_validator="$dev/validate_v14r2_decode_sidecar.py"
validator_python=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
deployment_validator="$dev/validate_v14r2_deployment_marker.py"
archive="$release/online-anchor-targetowned-qk-decode-v14r3-gradgeom-dfix2.tar"
revision="$release/online-anchor-targetowned-qk-decode-v14r3-gradgeom-dfix2.revision"
content_manifest="$release/online-anchor-targetowned-qk-decode-v14r3-gradgeom-dfix2.content.json"
deployment_marker="$release/DEPLOYMENT_TESTS_PASS_decode_targetcoord_v14r3_gradgeom_dfix2.json"
training_deployment_marker="$release/DEPLOYMENT_TESTS_PASS_targetcoord_v14r3_gradgeom.json"
logs="$release/logs/target_owned_routed_teacher_v14r3_gradgeom_decode_dfix2"
abort_authority="$dev/assets/v14r3_last_valid_checkpoint_abort_authority_v1.json"
training_source_tree="$stage/source-online-anchor-targetowned-qk-routed-teacher-v14r3-gradgeom"

test -d "$source_tree"
test ! -L "$source_tree"
test -f "$runner"
test ! -L "$runner"
test -f "$bridge_runner"
test ! -L "$bridge_runner"
test -f "$sidecar_validator"
test ! -L "$sidecar_validator"
test -f "$deployment_validator"
test ! -L "$deployment_validator"
test -x "$validator_python"
test -f "$archive"
test -f "$revision"
test -f "$content_manifest"
test -f "$deployment_marker"
test -f "$training_deployment_marker"
test -f "$abort_authority"
test ! -L "$abort_authority"
"$validator_python" -B "$deployment_validator" \
  --marker "$deployment_marker" --role decode \
  --source-tree "$source_tree" --archive "$archive" --revision "$revision" \
  --content-manifest "$content_manifest" --min-test-count 144 \
  --training-marker "$training_deployment_marker" \
  --shared-core methods/bernini_action_editing/anchor_qk_transport.py \
  --shared-core methods/bernini_action_editing/anchor_sga_anc_controller.py \
  --shared-core methods/bernini_action_editing/infer_anchor_sga_anc_event_v1.py \
  --required-file methods/bernini_action_editing/validate_v14r2_deployment_marker.py \
  --required-file methods/bernini_action_editing/validate_v14r2_decode_sidecar.py \
  --required-file methods/bernini_action_editing/anchor_qk_transport.py \
  --required-file methods/bernini_action_editing/anchor_cross_attention_transport.py \
  --required-file methods/bernini_action_editing/anchor_sga_anc_controller.py \
  --required-file methods/bernini_action_editing/exact_local_video_materializer_v1.py \
  --required-file methods/bernini_action_editing/infer_anchor_sga_anc_event_v1.py \
  --required-file methods/bernini_action_editing/infer_anchor_sga_anc_trained_editor_decode_v1.py \
  --required-file methods/bernini_action_editing/infer_lora.py \
  --required-file methods/bernini_action_editing/tools/build_renderer_dataset.py \
  --required-file methods/bernini_action_editing/tools/materialize_vae.py \
  --required-file methods/bernini_action_editing/assets/v14r3_last_valid_checkpoint_abort_authority_v1.json \
  --required-file methods/bernini_action_editing/scripts/auh_decode_online_anchor_attention_dynaedit_v1.sh \
  --required-file methods/bernini_action_editing/scripts/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh \
  --required-file methods/bernini_action_editing/scripts/auh_watch_decode_target_owned_routed_teacher_v14r2.sh \
  --required-file methods/bernini_action_editing/tests/test_anchor_qk_transport.py \
  --required-file methods/bernini_action_editing/tests/test_anchor_sga_anc_controller.py \
  --required-file methods/bernini_action_editing/tests/test_infer_anchor_sga_anc_event_rank0_prompt_bank_v1.py \
  --required-file methods/bernini_action_editing/tests/test_infer_anchor_sga_anc_trained_editor_decode_v1.py \
  --required-file methods/bernini_action_editing/tests/test_validate_v14r2_deployment_marker.py
mkdir -p "$logs"

wait_complete() {
  local experiment="$1"
  while [ ! -f "$release/train_${experiment}/TRAINING_COMPLETE" ]; do
    sleep 10
  done
}

validate_aborted_last_valid() {
  local experiment="$1" step="$2" node="$3" expected_combine_mode="$4"
  local run_dir="$release/train_${experiment}"
  local checkpoint="$run_dir/checkpoint-$(printf '%08d' "$step")"
  local log_path log_sha log_size failure_message rank_count step_id
  local expected_receipt_sha expected_adapter_sha expected_config_sha
  local accounting expected_entries actual_entries

  jq -e --arg experiment "$experiment" --arg node "$node" \
    --arg mode "$expected_combine_mode" --argjson step "$step" '
      .schema_version == "bernini-v14r3-last-valid-preoptimizer-abort-authority-v1" and
      .diagnostic_decode_only == true and
      .promotion_authorized == false and
      .checkpoint_artifact_complete == true and
      .requested_max_steps == 8 and
      (.arms[$experiment] | type == "object") and
      .arms[$experiment].training_run_complete == false and
      .arms[$experiment].node == $node and
      .arms[$experiment].replay_combine_mode == $mode and
      .arms[$experiment].checkpoint.last_completed_optimizer_step == $step and
      .arms[$experiment].failure.failed_attempt_step == ($step + 1) and
      .arms[$experiment].failure.stage == "before_parameter_gradient_mutation_and_optimizer_step" and
      .arms[$experiment].failure.rank_error_count == 4 and
      .arms[$experiment].slurm.state == "FAILED" and
      .arms[$experiment].slurm.exit_code == "1:0" and
      .training_authority.archive_sha256 == "7a8acbf18a03df4740927bf98019875bf07cc9665857f519b261f1fda33eedec" and
      .training_authority.deployment_marker_sha256 == "8cd78c38792e143cb5c4e068d93e80de0845605ef0f5b17ff61a67f49bee2b1e" and
      .training_authority.trainer_sha256 == "fd8c5b6d8d7fb94de9cb8d2811a953b116643216162b5cfe758f50ef0b55626c"
    ' "$abort_authority" >/dev/null

  log_path="$(jq -er --arg experiment "$experiment" '.arms[$experiment].log.path' "$abort_authority")"
  log_sha="$(jq -er --arg experiment "$experiment" '.arms[$experiment].log.sha256' "$abort_authority")"
  log_size="$(jq -er --arg experiment "$experiment" '.arms[$experiment].log.size_bytes' "$abort_authority")"
  failure_message="$(jq -er --arg experiment "$experiment" '.arms[$experiment].failure.message_exact' "$abort_authority")"
  rank_count="$(jq -er --arg experiment "$experiment" '.arms[$experiment].failure.rank_error_count' "$abort_authority")"
  step_id="$(jq -er --arg experiment "$experiment" '.arms[$experiment].slurm.step_id' "$abort_authority")"
  expected_receipt_sha="$(jq -er --arg experiment "$experiment" '.arms[$experiment].checkpoint.receipt_sha256' "$abort_authority")"
  expected_adapter_sha="$(jq -er --arg experiment "$experiment" '.arms[$experiment].checkpoint.adapter_model_sha256' "$abort_authority")"
  expected_config_sha="$(jq -er --arg experiment "$experiment" '.arms[$experiment].checkpoint.adapter_config_sha256' "$abort_authority")"
  test -f "$log_path"
  test ! -L "$log_path"
  test "$(stat -c %s -- "$log_path")" = "$log_size"
  test "$(sha256sum -- "$log_path" | awk '{print $1}')" = "$log_sha"
  test "$(grep -F -c -- "OnlineAnchorTrainingError: $failure_message" "$log_path")" = "$rank_count"

  test -d "$run_dir"
  test ! -L "$run_dir"
  test ! -e "$run_dir/TRAINING_COMPLETE"
  expected_entries="$(jq -r --arg experiment "$experiment" '.arms[$experiment].checkpoint.expected_run_entries[]' "$abort_authority")"
  actual_entries="$(find "$run_dir" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)"
  test "$actual_entries" = "$expected_entries"
  test -f "$checkpoint/receipt.json"
  test -f "$checkpoint/adapter/adapter_model.safetensors"
  test -f "$checkpoint/adapter/adapter_config.json"
  test "$(sha256sum -- "$checkpoint/receipt.json" | awk '{print $1}')" = "$expected_receipt_sha"
  test "$(sha256sum -- "$checkpoint/adapter/adapter_model.safetensors" | awk '{print $1}')" = "$expected_adapter_sha"
  test "$(sha256sum -- "$checkpoint/adapter/adapter_config.json" | awk '{print $1}')" = "$expected_config_sha"
  test "$(sha256sum -- "$training_source_tree/methods/bernini_action_editing/train_online_anchor_attention_v1.py" | awk '{print $1}')" = "fd8c5b6d8d7fb94de9cb8d2811a953b116643216162b5cfe758f50ef0b55626c"
  test "$(sha256sum -- "$release/online-anchor-targetowned-qk-routed-teacher-v14r3-gradgeom.tar" | awk '{print $1}')" = "7a8acbf18a03df4740927bf98019875bf07cc9665857f519b261f1fda33eedec"
  test "$(sha256sum -- "$training_deployment_marker" | awk '{print $1}')" = "8cd78c38792e143cb5c4e068d93e80de0845605ef0f5b17ff61a67f49bee2b1e"
  accounting="$(sacct -j "$step_id" --format=JobIDRaw,State,ExitCode,NodeList -n -P | head -n 1)"
  test "$accounting" = "$step_id|FAILED|1:0|$node"
}

validate_training_run_state() {
  local experiment="$1" step="$2" node="$3" expected_combine_mode="$4" policy="$5"
  case "$policy" in
    completed_s8)
      test "$step" = 8
      wait_complete "$experiment"
      ;;
    aborted_preoptimizer_last_valid)
      validate_aborted_last_valid "$experiment" "$step" "$node" "$expected_combine_mode"
      ;;
    *)
      echo "unsupported training-run decode policy: $policy" >&2
      return 2
      ;;
  esac
}

decode_transport_for_route() {
  case "$1" in
    self_target_owned_temporal_kernel_v14r2)
      echo self_target_owned_temporal_kernel_attn_output_v14r2 ;;
    self_target_owned_activity_kernel10_v14r2)
      echo self_target_owned_activity_kernel10_attn_output_v14r2 ;;
    self_target_owned_activity_kernel25_v14r2)
      echo self_target_owned_activity_kernel25_attn_output_v14r2 ;;
    *) echo "unsupported v14r2 training route: $1" >&2; return 2 ;;
  esac
}

event_slug() {
  case "$1" in
    0) echo pour-liquid-into-cup ;;
    2) echo twist-pull-mushroom ;;
    4) echo close-door-then-drawer ;;
    7) echo players-contact-then-separate ;;
    *) return 2 ;;
  esac
}

source_sha_for_event() {
  case "$1" in
    0) echo 888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de ;;
    2) echo 63dd620627e18d0f6836058bf725ffba7ab1f9b0b455e784c6a71dc68a8de46c ;;
    4) echo d699a8d5e35a57f09ae4ba5fc5124e733be9ed18a2bddb2ee90a1ba0232c53f5 ;;
    7) echo 1164531fd34d3d1273d56930aed139eb1a5d8db708ac3cdc4f7434abc0080799 ;;
    *) return 2 ;;
  esac
}

anchor_sha_for_event() {
  case "$1" in
    0) echo e485b2963d5ba191d8fc98158b7ab100bd1bd7584735221077c014ff0c3a19aa ;;
    2) echo a1076cbe83c9dae4a4fddc25f73077288f6a3240324fd2d5e1854aa842b07b63 ;;
    4) echo c6d6a4e2835972609fcde8a8fbc2357eb36396f4a54aef7366adf809d6593f5e ;;
    7) echo 7b128ed47a7f6122d40a12711cf31535e39bcd5b92ce97d031cc2ff49424f4fc ;;
    *) return 2 ;;
  esac
}

validate_existing_decode() {
  local video="$1" sidecar="$2" checkpoint="$3" step="$4" route="$5"
  local transport="$6" transport_steps="$7" adapter_sha="$8" receipt_sha="$9"
  local config_sha="${10}"
  local source_sha="${11}" anchor_sha="${12}" preservation="${13}"
  local sga_score="${14}" route_off="${15}"
  test -f "$video"
  test ! -L "$video"
  test -f "$sidecar"
  test ! -L "$sidecar"
  local video_sha
  video_sha="$(sha256sum -- "$video" | awk '{print $1}')"
  jq -e \
    --arg video "$video" --arg video_sha "$video_sha" \
    --arg checkpoint "$checkpoint" --argjson step "$step" \
    --arg route "$route" --arg transport "$transport" \
    --argjson transport_steps "$transport_steps" \
    --arg adapter_sha "$adapter_sha" --arg receipt_sha "$receipt_sha" \
    --arg config_sha "$config_sha" '
    .complete == true and
    .loaded_trained_attention_checkpoint == true and
    .trained_attention_checkpoint.path == $checkpoint and
    .trained_attention_checkpoint.global_step == $step and
    .trained_attention_checkpoint.training_objective == "real_source_target_owned_routed_teacher_delta_v14r2" and
    .trained_attention_checkpoint.route_operator == $route and
    .trained_attention_checkpoint.required_decode_transport == $transport and
    .trained_attention_checkpoint.adapter_model_sha256 == $adapter_sha and
    .trained_attention_checkpoint.adapter_config_sha256 == $config_sha and
    .trained_attention_checkpoint.receipt_sha256 == $receipt_sha and
    .trained_attention_checkpoint.checkpoint_binding.global_step == $step and
    .trained_attention_checkpoint.checkpoint_binding.training_objective == "real_source_target_owned_routed_teacher_delta_v14r2" and
    .trained_attention_checkpoint.checkpoint_binding.route_operator == $route and
    .trained_attention_checkpoint.checkpoint_binding.required_decode_transport == $transport and
    .trained_attention_checkpoint.checkpoint_binding.adapter_model_sha256 == $adapter_sha and
    .trained_attention_checkpoint.checkpoint_binding.adapter_config_sha256 == $config_sha and
    .trained_attention_checkpoint.checkpoint_binding.receipt_sha256 == $receipt_sha and
    .mechanism.transport == $transport and
    .mechanism.transport_steps == $transport_steps and
    .mechanism.decode_audit_contract.transport_steps == $transport_steps and
    .output.path == $video and
    .output.sha256 == $video_sha and
    .output.frames == 81 and
    .output.fps == 25
  ' "$sidecar" >/dev/null
  "$validator_python" -B "$sidecar_validator" \
    --sidecar "$sidecar" --video "$video" --checkpoint "$checkpoint" \
    --step "$step" --route "$route" --transport "$transport" \
    --transport-steps "$transport_steps" --adapter-sha256 "$adapter_sha" \
    --adapter-config-sha256 "$config_sha" --receipt-sha256 "$receipt_sha" \
    --source-sha256 "$source_sha" --anchor-sha256 "$anchor_sha" \
    --preservation-mode "$preservation" --sga-score-mode "$sga_score" \
    --route-off "$route_off"
  ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=nb_read_frames,avg_frame_rate -of json "$video" \
    | jq -e '
        .streams | length == 1 and
        .[0].nb_read_frames == "81" and
        .[0].avg_frame_rate == "25/1"
      ' >/dev/null
}

decode_one() {
  local node="$1" experiment="$2" tag="$3" route="$4" step="$5" event="$6"
  local transport_steps="$7" allow_route_off="$8" preservation="$9" sga_score="${10}"
  local expected_teacher_mode="${11}" expected_combine_mode="${12}"
  local transport checkpoint receipt adapter_model adapter_config
  local adapter_sha config_sha receipt_sha
  local slug event_token output_dir label video sidecar source_sha anchor_sha
  transport="$(decode_transport_for_route "$route")"
  checkpoint="$release/train_${experiment}/checkpoint-$(printf '%08d' "$step")"
  receipt="$checkpoint/receipt.json"
  adapter_model="$checkpoint/adapter/adapter_model.safetensors"
  adapter_config="$checkpoint/adapter/adapter_config.json"
  test -f "$receipt"
  test -f "$adapter_config"
  test -f "$adapter_model"
  adapter_sha="$(sha256sum -- "$adapter_model" | awk '{print $1}')"
  config_sha="$(sha256sum -- "$adapter_config" | awk '{print $1}')"
  source_sha="$(source_sha_for_event "$event")"
  anchor_sha="$(anchor_sha_for_event "$event")"
  jq -e --argjson step "$step" --arg route "$route" --arg transport "$transport" \
    --arg teacher_mode "$expected_teacher_mode" --arg combine_mode "$expected_combine_mode" \
    --arg adapter_sha "$adapter_sha" --arg config_sha "$config_sha" '
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
    def actual_optimizer_update_ok($mode; $expected_step):
      .schema_version == "bernini-actual-optimizer-update-probe-v1" and
      .step == $expected_step and
      .replay_combine_mode == $mode and
      .gradient_scope == "separately_allreduced_global_action_and_raw_replay" and
      .optimizer_semantics_observed_not_modified == true and
      .parameter_snapshot_native_dtype == true and
      (.parameter_snapshot_dtypes | length) >= 1 and
      .tensor_count == 480 and
      .parameter_element_count == 188743680 and
      .changed_tensor_count == 480 and
      .changed_element_count > 0 and
      .delta_theta_l2_norm_fp64 > 0 and
      .action_descent_required == true and
      .action_descent_passed == true and
      .action_descent_fp64 > 0 and
      (if ($mode == "norm_balanced_025" or $mode == "source_halfspace_001") then
         .source_descent_required == true and
         .source_descent_passed == true and
         .source_descent_fp64 >= .minimum_allowed_source_descent_fp64
       else
         .source_descent_required == false
       end);
    .complete == true and
    .schema_version == "bernini-online-anchor-attention-training-receipt-v3" and
    .global_step == $step and
    .training_contract.training_objective == "real_source_target_owned_routed_teacher_delta_v14r2" and
    .training_contract.route_operator == $route and
    .training_contract.route_transport == $transport and
    .training_contract.routed_teacher_mode == $teacher_mode and
    .training_contract.replay_combine_mode == $combine_mode and
    .adapter_model_sha256 == $adapter_sha and
    .adapter_config_sha256 == $config_sha and
    .training_contract.target_owned_qk_route_v14r2 == true and
    .training_contract.anchor_donor_cached_fields == ["query", "key"] and
    .training_contract.anchor_donor_value_cached_or_used_by_route == false and
    .training_contract.anchor_to_target_appearance_correspondence_used == false and
    .training_contract.anchor_qk_time_constant_caption_offset_removed_before_support_and_kernel == true and
    .training_contract.anchor_qk_phase0_only_difference_produces_zero_route == true and
    .training_contract.real_source_variant_schedule == "complete_real_source" and
    .training_contract.source_variant_argument == "not_applicable" and
    .training_contract.micro_semantics == "different_seed_and_cross_appearance_donor" and
    .training_contract.base_replay_scale == 0.025 and
    .training_contract.true_training_memory_fraction_strictly_above_half == true and
    .training_contract.training_memory_gate_capture_phase == "after_two_real_component_backwards_before_actual_update_audit_clones" and
    .training_contract.actual_update_audit_allocations_excluded_from_training_memory_gate == true and
    .memory_gate.capture_phase == "after_two_real_component_backwards_before_actual_update_audit_clones" and
    .memory_gate.actual_update_audit_allocations_excluded == true and
    .memory_gate.true_training_tensors_only == true and
    .memory_gate.dummy_or_padding_allocations == false and
    .memory_gate.passed == true and
    .memory_gate.minimum_reserved_fraction > 0.5 and
    all(.memory_gate.per_rank[]; .reserved_fraction > 0.5) and
    .training_contract.effective_replay_scale == .component_gradient_probes.interaction.effective_replay_scale and
    .last_loss == null and
    .last_reporting_scalar_is_not_a_joint_backpropagated_objective == true and
    .component_gradient_probes.action_objective.tensor_count == 480 and
    .component_gradient_probes.action_objective.nonzero_tensor_count == 480 and
    .component_gradient_probes.action_objective.epsilon_active_tensor_count == 480 and
    .component_gradient_probes.action_objective.adapter_sides.lora_A.nonzero_tensor_count == 240 and
    .component_gradient_probes.action_objective.adapter_sides.lora_A.epsilon_active_tensor_count == 240 and
    .component_gradient_probes.action_objective.adapter_sides.lora_B.nonzero_tensor_count == 240 and
    .component_gradient_probes.action_objective.adapter_sides.lora_B.epsilon_active_tensor_count == 240 and
    .component_gradient_probes.raw_source_caption_trajectory_replay.nonzero_tensor_count == 480 and
    .component_gradient_probes.raw_source_caption_trajectory_replay.epsilon_active_tensor_count == 480 and
    (.component_gradient_probes.interaction | combine_geometry_ok($combine_mode)) and
    (.actual_optimizer_update_probe |
      actual_optimizer_update_ok($combine_mode; $step)) and
    .gradient_coverage.tensor_count == 480 and
    .gradient_coverage.nonzero_tensor_count == 480 and
    .anchor_cache.pending_entries == 0 and
    .anchor_cache.qk_only_cached_fields == ["query", "key"] and
    .anchor_cache.replay_count == (2 * .anchor_cache.capture_count)
  ' "$receipt" >/dev/null
  receipt_sha="$(sha256sum -- "$receipt" | awk '{print $1}')"
  slug="$(event_slug "$event")"
  printf -v event_token 'e%02d' "$event"
  output_dir="$release/dynaedit_fullgrid_v2/$tag/step_$(printf '%08d' "$step")/$event_token"
  label="E$(printf '%02d' "$event")_${slug}_${tag}_S${step}_ONLINE_ANCHOR_REAL_SGA_ANC"
  video="$output_dir/$label.mp4"
  sidecar="$video.receipt.json"

  if [ -e "$video" ] || [ -L "$video" ] || [ -e "$sidecar" ] || [ -L "$sidecar" ]; then
    validate_existing_decode "$video" "$sidecar" "$checkpoint" "$step" "$route" \
      "$transport" "$transport_steps" "$adapter_sha" "$receipt_sha" "$config_sha" \
      "$source_sha" "$anchor_sha" "$preservation" "$sga_score" "$allow_route_off"
    echo "validated exact existing decode: tag=$tag step=$step event=$event_token"
    return 0
  fi
  if [ -d "$output_dir" ] && find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "fresh v14r2 tag contains an unexpected partial artifact: $output_dir" >&2
    return 23
  fi

  srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --cpus-per-task=64 \
    --gres=gpu:4 --mem=0 --nodelist="$node" \
    env ONLINE_ANCHOR_DECODE_EXPERIMENT="$experiment" \
        ONLINE_ANCHOR_OUTPUT_EXPERIMENT="$tag" \
        ONLINE_ANCHOR_DECODE_STEP="$step" \
        ONLINE_ANCHOR_DECODE_TRANSPORT="$transport" \
        ONLINE_ANCHOR_DECODE_TRANSPORT_STEPS="$transport_steps" \
        ONLINE_ANCHOR_DECODE_STRENGTH=0.25 \
        ONLINE_ANCHOR_DECODE_CFG_SCOPE=shared \
        ONLINE_ANCHOR_DECODE_STATE_MODE=clean_noised \
        ONLINE_ANCHOR_DECODE_SOURCE_CFG_SCALE=4.5 \
        ONLINE_ANCHOR_DECODE_TARGET_CFG_SCALE=4.5 \
        ONLINE_ANCHOR_DECODE_PRESERVATION_MODE="$preservation" \
        ONLINE_ANCHOR_DECODE_SGA_SCORE_MODE="$sga_score" \
        ONLINE_ANCHOR_ALLOW_TRAINED_ROUTE_OFF_CONTROL="$allow_route_off" \
        ONLINE_ANCHOR_EXPECTED_TRAINING_OBJECTIVE=real_source_target_owned_routed_teacher_delta_v14r2 \
        ONLINE_ANCHOR_EXPECTED_TRAINING_ROUTE_OPERATOR="$route" \
        ONLINE_ANCHOR_EXPECTED_ADAPTER_SHA256="$adapter_sha" \
        ONLINE_ANCHOR_EXPECTED_ADAPTER_CONFIG_SHA256="$config_sha" \
        ONLINE_ANCHOR_EXPECTED_RECEIPT_SHA256="$receipt_sha" \
        ONLINE_ANCHOR_DECODE_DEV_OVERRIDE="$dev" \
        ONLINE_ANCHOR_DECODE_BRIDGE_RUNNER="$bridge_runner" \
        bash "$runner" "$event" action_noop 0
  validate_existing_decode "$video" "$sidecar" "$checkpoint" "$step" "$route" \
    "$transport" "$transport_steps" "$adapter_sha" "$receipt_sha" "$config_sha" \
    "$source_sha" "$anchor_sha" "$preservation" "$sga_score" "$allow_route_off"
}

run_arm() {
  local node="$1" experiment="$2" route="$3" primary="$4"
  local expected_teacher_mode="$5" expected_combine_mode="$6"
  local step="$7" run_policy="$8" tag_prefix
  if [ "$run_policy" = aborted_preoptimizer_last_valid ]; then
    tag_prefix="${experiment}_LASTVALID_S${step}_ABORTED_PREOPT_S$((step + 1))_DIAG"
  else
    tag_prefix="${experiment}"
  fi
  local route_on_tag="${tag_prefix}_routeon_clean40_v14r3d2"
  local route_off_tag="${tag_prefix}_sameckpt_routeoff_control_v14r3d2"
  local preservation_tag="${tag_prefix}_preserve_motion_actionreward_v14r3d2"
  validate_training_run_state "$experiment" "$step" "$node" \
    "$expected_combine_mode" "$run_policy"
  for event in 0 2 4 7; do
    validate_training_run_state "$experiment" "$step" "$node" \
      "$expected_combine_mode" "$run_policy"
    decode_one "$node" "$experiment" "$route_on_tag" "$route" \
      "$step" "$event" 40 0 none global_source_cosine \
      "$expected_teacher_mode" "$expected_combine_mode"
  done
  for event in 0 2 4 7; do
    validate_training_run_state "$experiment" "$step" "$node" \
      "$expected_combine_mode" "$run_policy"
    decode_one "$node" "$experiment" "$route_off_tag" "$route" \
      "$step" "$event" 0 1 none global_source_cosine \
      "$expected_teacher_mode" "$expected_combine_mode"
  done
  if [ "$primary" = 1 ]; then
    for event in 0 4; do
      validate_training_run_state "$experiment" "$step" "$node" \
        "$expected_combine_mode" "$run_policy"
      decode_one "$node" "$experiment" "$preservation_tag" "$route" \
        "$step" "$event" 40 0 source_motion_support background_plus_anchor_action_002 \
        "$expected_teacher_mode" "$expected_combine_mode"
    done
  elif [ "$primary" != 0 ]; then
    return 3
  fi
}

if [ "${1:-}" = --validate-aborted-last-valid ]; then
  if [ "$#" -ne 5 ]; then
    echo "usage: $0 --validate-aborted-last-valid EXPERIMENT STEP NODE COMBINE_MODE" >&2
    exit 2
  fi
  validate_aborted_last_valid "$2" "$3" "$4" "$5"
  exit 0
fi

if [ "${1:-}" = --chain ]; then
  shift
  run_arm "$@"
  exit 0
fi

test -f "$watcher"
test ! -L "$watcher"
launch() {
  local label="$1"; shift
  local log="$logs/${label}.log"
  test ! -e "$log"
  nohup bash "$watcher" --chain "$@" >"$log" 2>&1 &
  echo "$! $job $label"
}

launch sameaction_global_actiononly auh7-1b-gpu-233 \
  sameaction_global_actiononly_s8_v14r3_gradgeom \
  self_target_owned_temporal_kernel_v14r2 0 \
  same_action_route_only action_only 4 aborted_preoptimizer_last_valid
launch sameaction_global_norm025 auh7-1b-gpu-268 \
  sameaction_global_norm025_s8_v14r3_gradgeom \
  self_target_owned_temporal_kernel_v14r2 1 \
  same_action_route_only norm_balanced_025 4 aborted_preoptimizer_last_valid
launch sameaction_gate25_pcgrad010 auh7-1b-gpu-292 \
  sameaction_gate25_pcgrad010_s8_v14r3_gradgeom \
  self_target_owned_activity_kernel25_v14r2 0 \
  same_action_route_only action_priority_pcgrad_010 8 completed_s8
