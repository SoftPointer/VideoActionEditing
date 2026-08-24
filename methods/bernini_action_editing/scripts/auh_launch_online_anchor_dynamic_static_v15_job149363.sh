#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "v15 controller must run on the AUH login host" >&2; exit 2 ;;
esac

job=149363
node=auh7-1b-gpu-312
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
source_tree="${ONLINE_ANCHOR_SOURCE_TREE:-$stage/source-online-anchor-targetowned-qk-routed-teacher-v15-dynamicstatic-20260823}"
release="${ONLINE_ANCHOR_RELEASE:-$stage/online_anchor_dynamic_static_v15_20260823}"
worker="$source_tree/methods/bernini_action_editing/scripts/auh_train_online_anchor_dynamic_static_v15.sh"
log_root="$release/logs/job149363_node312_s2_s8_s32"

test -f "$worker"
test ! -L "$worker"
test -f "$source_tree/methods/bernini_action_editing/train_online_anchor_attention_dynamic_static_v15.py"
mkdir -p "$log_root"

gate_receipt() {
  local receipt="$1" expected_step="$2"
  test -f "$receipt"
  test ! -L "$receipt"
  jq -e --argjson expected_step "$expected_step" '
    .complete == true and
    .schema_version == "bernini-online-anchor-dynamic-static-routed-teacher-receipt-v15" and
    .global_step == $expected_step and
    .scientific_claim_authorized == false and
    .claim_scope == "engineering_training_run_only_non_scientific_until_held_out_evaluation" and
    .training_contract.method == "bernini-online-anchor-dynamic-static-routed-teacher-v15" and
    .training_contract.profile == "dynamic_static" and
    .training_contract.training_objective == "real_source_target_owned_routed_teacher_delta_v14r2" and
    .training_contract.route_operator == "self_target_owned_activity_kernel25_v14r2" and
    .training_contract.routed_teacher_mode == "same_action_route_only" and
    .training_contract.replay_combine_mode == "action_priority_pcgrad_010" and
    .training_contract.student_route_strength == 0.25 and
    .training_contract.teacher_route_strength == 0.5 and
    .training_contract.anchor_contrast_profile_is_state_not_caption == true and
    .training_contract.anchor_action_caption_equals_static_caption == true and
    .training_contract.caption_difference_used_as_anchor_supervision == false and
    .training_contract.anchor_phase0_is_exactly_donor_owned_on_both_branches == true and
    .training_contract.anchor_dynamic_and_static_post_phase0_clean_states_differ == true and
    .training_contract.anchor_dynamic_and_static_share_exact_noise_rng_seed_scheduler_and_timestep == true and
    .training_contract.anchor_recovered_gaussian_agrees_within_fp32_tolerance == true and
    .training_contract.anchor_recovered_gaussian_fp32_max_abs_error <=
      .training_contract.anchor_recovered_gaussian_fp32_tolerance and
    .training_contract.anchor_dynamic_static_pairs_audited == (2 * $expected_step) and
    .training_contract.training_manifest_order == "variant_major_event_interleaved_v15" and
    .training_contract.actual_distinct_target_iid_count == $expected_step and
    (.training_contract.actual_distinct_target_iids | length) == $expected_step and
    .training_contract.actual_distinct_target_event_count ==
      (if $expected_step < 8 then $expected_step else 8 end) and
    (.training_contract.actual_distinct_target_events | length) ==
      (if $expected_step < 8 then $expected_step else 8 end) and
    .training_contract.actual_distinct_cross_appearance_donor_iid_count >= $expected_step and
    .training_contract.anchor_qk_support_uses_phase0_relative_action_noop_contrast == false and
    .training_contract.anchor_qk_support_uses_phase0_relative_same_caption_dynamic_static_contrast == true and
    .training_contract.self_generated_intermediate_supervision ==
      "detached_frozen_donor_qk_temporal_route_support" and
    .training_contract.self_generated_rgb_or_latent_used_as_flow_matching_target == false and
    .training_contract.student_supervision_is_target_coordinate_routed_teacher_delta == true and
    .training_contract.starts_from_frozen_base_checkpoint_not_prior_adapter == true and
    .training_contract.scientific_claim_authorized == false and
    .training_contract.target_owned_qk_route_v14r2 == true and
    .training_contract.target_value_stream_is_sole_routed_content == true and
    .training_contract.action_objective_backpropagates_only_routed_student_query == true and
    .training_contract.true_training_memory_fraction_strictly_above_half == true and
    .actual_optimizer_update_probe.action_descent_passed == true
  ' "$receipt" >/dev/null
}

run_stage() {
  local steps="$1"
  local experiment="dynamic_static_activity25_pcgrad_s${steps}_job149363"
  local output="$release/train_${experiment}"
  local log="$log_root/${experiment}.log"
  test ! -e "$output"
  test ! -e "$log"
  srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --cpus-per-task=64 \
    --gres=gpu:4 --mem=0 --nodelist="$node" \
    env \
      ONLINE_ANCHOR_SOURCE_TREE="$source_tree" \
      ONLINE_ANCHOR_RELEASE="$release" \
      ONLINE_ANCHOR_EXPERIMENT="$experiment" \
      ONLINE_ANCHOR_MAX_STEPS="$steps" \
      bash "$worker" >"$log" 2>&1
  test -f "$output/TRAINING_COMPLETE"
  gate_receipt "$output/checkpoint-$(printf '%08d' "$steps")/receipt.json" "$steps"
}

# These are three independent fresh-from-base runs.  An earlier checkpoint is
# never passed to a later stage; the gates increase exposure only after the
# preceding engineering run has completed and emitted a valid receipt.
run_stage 2
run_stage 8
run_stage 32
