#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "v16r3 controller must run on the AUH login host" >&2; exit 2 ;;
esac

job="${ONLINE_ANCHOR_JOB_ID:?Slurm allocation ID is required}"
gpu_devices="${ONLINE_ANCHOR_GPU_DEVICES:-0,1,2,3}"
source_tree="${ONLINE_ANCHOR_SOURCE_TREE:?fresh frozen v16r3 source tree is required}"
release="${ONLINE_ANCHOR_RELEASE:?fresh v16r3 release root is required}"
case "$job" in *[!0-9]*|'') echo "invalid Slurm allocation ID" >&2; exit 3 ;; esac
mapfile -t allocated_nodes < <(
  squeue --noheader --jobs="$job" --states=RUNNING --format='%N'
)
if [ "${#allocated_nodes[@]}" -ne 1 ]; then
  echo "v16r3 allocation $job must be RUNNING on exactly one resolved node" >&2
  exit 3
fi
node="${allocated_nodes[0]}"
case "$node" in *[!A-Za-z0-9_.-]*|'') echo "invalid resolved AUH node name" >&2; exit 3 ;; esac

worker="$source_tree/methods/bernini_action_editing/scripts/auh_train_online_anchor_full644_dynamic_static_v16r3.sh"
trainer="$source_tree/methods/bernini_action_editing/train_online_anchor_attention_full644_dynamic_static_v16r3.py"
log_root="$release/logs/job${job}_${node}_continuous_s644"
test -f "$worker"
test ! -L "$worker"
test -f "$trainer"
test ! -L "$trainer"
mkdir -p "$log_root"

gate_receipt() {
  local receipt="$1" expected_step="$2"
  test -f "$receipt"
  test ! -L "$receipt"
  jq -e --argjson expected_step "$expected_step" '
    .complete == true and
    .schema_version == "bernini-online-anchor-full644-dynamic-static-routed-teacher-receipt-v16r3" and
    .global_step == $expected_step and
    .max_steps == 644 and
    .scientific_claim_authorized == false and
    .claim_scope == "engineering_training_run_only_non_scientific_until_held_out_evaluation" and
    .training_contract.method == "bernini-online-anchor-full644-dynamic-static-routed-teacher-v16r3" and
    .training_contract.profile == "dynamic_static" and
    .training_contract.full644_manifest_row_count == 644 and
    .training_contract.full644_manifest_family_count == 28 and
    .training_contract.full644_manifest_sha256 ==
      "61da995eb680b9fba7ab3b7d3b6041c7b51c7e95253c74e607ddab6fdd6a61aa" and
    .training_contract.full644_manifest_digest ==
      "96fe6188ad0f5ee72dcd89fbc018835f3f2995e45ff116f07449e863fa9b51d5" and
    .training_contract.full644_optimizer_schedule == "exact644_unique_rows_once" and
    .training_contract.training_manifest_order == "family_round_robin_manifest_iid_stable_exact644_once_v16" and
    .training_contract.actual_distinct_target_iid_count == $expected_step and
    (.training_contract.actual_distinct_target_iids | length) == $expected_step and
    .training_contract.anchor_dynamic_static_pairs_audited == (2 * $expected_step) and
    .training_contract.anchor_source_and_donor_share_iid == true and
    .training_contract.anchor_cross_appearance == false and
    .training_contract.source_posterior_used_as_complete_target_coordinate_state == true and
    .training_contract.self_generated_action_anchor_used_as_flow_matching_target == false and
    .training_contract.two_independently_seeded_anchor_captures_per_target_update == true and
    .training_contract.starts_from_frozen_base_checkpoint_not_prior_adapter == true and
    .training_contract.manual_or_visual_review_controls_optimizer_admission == false and
    .training_contract.qwen_or_other_verifier_controls_optimizer_admission == false and
    .training_contract.strict_selection_flag_filters_optimizer_rows == false and
    .training_contract.broad_and_strict_rows_are_both_optimizer_admitted == true and
    .training_contract.actual_action_descent_gate_relaxed == false and
    .training_contract.actual_action_descent_failed_candidates_committed == false and
    .training_contract.actual_action_descent_fallback_parameter_values_exactly_restored == true and
    .training_contract.actual_action_descent_fallback_optimizer_state_restored == false and
    .training_contract.actual_action_descent_fallback_uses_primary_action_only == true and
    .training_contract.actual_action_descent_fallback_retry_limit == 1 and
    .training_contract.actual_action_descent_fallback_reprobes_frozen_authority == true and
    .training_contract.qk_only_zero_rms_backward_policy ==
      "exact_forward_zero_rms_zero_subgradient_v1" and
    .training_contract.qk_only_zero_rms_backward_scope ==
      ["current_temporal_rms", "route_rms"] and
    .training_contract.qk_only_zero_rms_forward_values_changed == false and
    .training_contract.qk_only_zero_rms_zero_subgradient == 0 and
    .training_contract.loss_scale_changed_for_v16r3 == false and
    .training_contract.seed_or_timestep_changed_for_v16r3 == false and
    .training_contract.sample_retry_or_skip_for_v16r3 == false and
    .training_contract.component_preallreduce_finite_gate_relaxed == false and
    .training_contract.nonfinite_gradient_committed == false and
    .anchor_cache.qk_only_zero_rms_backward_policy ==
      "exact_forward_zero_rms_zero_subgradient_v1" and
    .v16r3_zero_rms_backward_summary.policy ==
      "exact_forward_zero_rms_zero_subgradient_v1" and
    .v16r3_zero_rms_backward_summary.scope ==
      ["current_temporal_rms", "route_rms"] and
    .v16r3_zero_rms_backward_summary.finite_nonnegative_forward_values_bit_exact == true and
    .v16r3_zero_rms_backward_summary.zero_forward_value == 0 and
    .v16r3_zero_rms_backward_summary.zero_backward_subgradient == 0 and
    .v16r3_zero_rms_backward_summary.positive_backward_matches_standard_sqrt == true and
    .v16r3_zero_rms_backward_summary.negative_or_nonfinite_values_masked == false and
    .v16r3_zero_rms_backward_summary.loss_scale_changed == false and
    .v16r3_zero_rms_backward_summary.seed_or_timestep_changed == false and
    .v16r3_zero_rms_backward_summary.sample_retry_or_skip == false and
    .v16r3_zero_rms_backward_summary.component_preallreduce_finite_gate_relaxed == false and
    .v16r3_zero_rms_backward_summary.nonfinite_gradient_committed == false and
    .v16r3_zero_rms_backward_summary.policy_fixed_from_step_one == true and
    .v16r3_zero_rms_backward_summary.single_continuous_fresh_from_base_exact644 == true and
    .v16r3_zero_rms_backward_summary.s279_endpoint_canary.step == 279 and
    .v16r3_zero_rms_backward_summary.s279_endpoint_canary.target_iid ==
      "4aeb0557a94b4db3" and
    .v16r3_zero_rms_backward_summary.s279_endpoint_canary.target_family == "fall" and
    .v16r3_zero_rms_backward_summary.s279_endpoint_canary.expected_calls == [
      {"role":"action_micro_0","seed":1656484053,"timestep":1000},
      {"role":"raw_replay_micro_0","seed":1657484056,"timestep":580},
      {"role":"action_micro_1","seed":718898016,"timestep":764},
      {"role":"raw_replay_micro_1","seed":719898019,"timestep":880}
    ] and
    (if $expected_step >= 279 then
      .training_contract.s279_endpoint_canary_covered == true and
      .v16r3_zero_rms_backward_summary.s279_endpoint_canary.covered_by_checkpoint == true and
      .v16r3_zero_rms_backward_summary.s279_endpoint_canary.observed_calls ==
        .v16r3_zero_rms_backward_summary.s279_endpoint_canary.expected_calls
    else
      .training_contract.s279_endpoint_canary_covered == false and
      .v16r3_zero_rms_backward_summary.s279_endpoint_canary.covered_by_checkpoint == false and
      (.v16r3_zero_rms_backward_summary.s279_endpoint_canary.observed_calls | length) == 0
    end) and
    (if $expected_step >= 22 then
      .training_contract.actual_action_descent_fallback_count >= 1 and
      (.training_contract.actual_action_descent_fallback_steps | index(22) != null) and
      .training_contract.optimizer_history_matches_uninterrupted_adamw == false
    else
      .training_contract.actual_action_descent_fallback_count == 0 and
      .training_contract.optimizer_history_matches_uninterrupted_adamw == true
    end) and
    .v16r2_actual_action_descent_fallback_summary.failed_candidates_committed == false and
    .v16r2_actual_action_descent_fallback_summary.optimizer_state_restored == false and
    .v16r2_actual_action_descent_fallback_summary.action_descent_gate_relaxed == false and
    .actual_optimizer_update_probe.action_descent_passed == true and
    .actual_optimizer_update_probe.v16r2_action_descent_gate_relaxed == false and
    .v16_full644_summary.target_prefix_row_count == $expected_step and
    .v16_full644_summary.target_prefix_exact_once == true and
    .v16_full644_summary.donor_selection_count == (2 * $expected_step) and
    .v16_full644_summary.same_iid_role1_donor_count == (2 * $expected_step) and
    .v16_full644_summary.anchor_cross_appearance == false and
    .v16_full644_summary.manual_or_visual_review_controls_optimizer_admission == false and
    .v16_full644_summary.all_rows_admitted_from_sealed_manifest_without_per_sample_review == true and
    .v16_full644_summary.scientific_claim_authorized == false and
    (if $expected_step == 644 then
      .training_contract.all_full644_rows_targeted_exactly_once == true and
      .training_contract.actual_distinct_target_family_count == 28 and
      .training_contract.observed_latent_geometry_count == 20 and
      .training_contract.lazy_pair_cache_max_rows == 1 and
      .v16_full644_summary.actual_strict_target_count == 359 and
      .v16_full644_summary.actual_broad_target_count == 285 and
      .v16_full644_summary.pair_decode_count == 644 and
      .v16_full644_summary.all_full644_rows_targeted_exactly_once == true
    else true end)
  ' "$receipt" >/dev/null
}

run_exact644() {
  local steps=644
  local experiment="full644_dynamic_static_activity25_pcgrad_zerormssafe_v16r3_s${steps}_job${job}"
  local output="$release/train_${experiment}"
  local log="$log_root/${experiment}.log"
  test ! -e "$output"
  test ! -e "$log"
  srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --cpus-per-task=64 \
    --gres=gpu:4 --mem=0 --nodelist="$node" \
    env \
      ONLINE_ANCHOR_JOB_ID="$job" \
      ONLINE_ANCHOR_NODE="$node" \
      ONLINE_ANCHOR_SOURCE_TREE="$source_tree" \
      ONLINE_ANCHOR_RELEASE="$release" \
      ONLINE_ANCHOR_GPU_DEVICES="$gpu_devices" \
      ONLINE_ANCHOR_EXPERIMENT="$experiment" \
      ONLINE_ANCHOR_MAX_STEPS="$steps" \
      bash "$worker" >"$log" 2>&1
  test -f "$output/TRAINING_COMPLETE"
  local saved_step receipt
  for saved_step in 1 4 8 16 28 32 64 128 256 359 512 644; do
    receipt="$output/checkpoint-$(printf '%08d' "$saved_step")/receipt.json"
    gate_receipt "$receipt" "$saved_step"
  done
}

# One fixed-method, continuous family-round-robin trajectory, fresh from base.
run_exact644
