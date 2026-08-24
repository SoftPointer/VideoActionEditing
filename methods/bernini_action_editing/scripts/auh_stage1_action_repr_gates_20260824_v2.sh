#!/usr/bin/env bash
# Strict zero-update Stage-1 launcher for the 2026-08-24 action representation.
#
# This file can extract frozen representations and evaluate G0/G1.  It has no
# training command.  The production WORLD4 G2a audit is reachable only after a
# passed target G1 receipt; it performs six step-zero native no-op routes and
# still creates no optimizer, backward, checkpoint, or parameter update.
set -euo pipefail
umask 077

experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_repr_target_selfgen_middle_g1_20260824_v2
g0_source_root="$experiment_root/source"
source_root="$experiment_root/source_v2_7"
method_root="$source_root/methods/bernini_action_editing"
test_root="$source_root/tests"
manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_repr_target_selfgen_20260824_v1/source/experiment_manifest.json
prereg="$g0_source_root/stage1_v2_preregistration.json"
prereg_addendum="$source_root/stage1_v2_source_lock_addendum.json"
posterior_identity_addendum="$source_root/stage1_v2_posterior_identity_addendum.json"
matched_noise_addendum="$source_root/stage1_v2_matched_noise_addendum.json"
g2a_six_route_addendum="$source_root/stage1_v2_g2a_six_route_addendum.json"
explicit_gaussian_authority_addendum="$source_root/stage1_v2_explicit_gaussian_authority_addendum.json"
g1_authority_fixture_addendum="$source_root/stage1_v2_g1_authority_fixture_addendum.json"
deterministic_vae_authority_addendum="$source_root/stage1_v2_deterministic_vae_authority_addendum.json"
quantized_energy_match_addendum="$source_root/stage1_v2_quantized_energy_match_addendum.json"
stage_root="$experiment_root/stage1_v2_7"
log_root="$experiment_root/logs/stage1_v2_7"

python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4

expected_manifest_sha=c78e42f0661e5905407505037ce322d32d67ffec0b70b1cab466f895dc8d0632
expected_prereg_sha=294168e596212bd61e8d555e72702ceeeb993fb18c7fa7536a43d0b00ad592b3
expected_addendum_sha=1b1b4736a0925a080c7423ebb3f5358e0be3b5719df6b403840f6c221a570985
expected_posterior_identity_addendum_sha=62e3ed49084a85d8d969dbd68f28c1659aceacedc19f62abcd4f7324dcd05228
expected_matched_noise_addendum_sha=5cb3ab6350f8122b84860b0a99264334d2343137921196049e71d832588fe70c
expected_g2a_six_route_addendum_sha=35944dbad37d38148c8305c818207c2586960e9b0692fc94f8ee8952608d11d2
expected_explicit_gaussian_authority_addendum_sha=dc52025c8bbfe349d8917e3711dadc688f0dec5b121d239a2765edbe3f566444
expected_g1_authority_fixture_addendum_sha=fea432a543760a2771c529f1d7ff59798dea27425a6a7e617e46defb9b975266
expected_deterministic_vae_authority_addendum_sha=20f9c7a138a30d742113eca4c68c7b0c3a815d40d5753bb016b6b6c30959731f
expected_quantized_energy_match_addendum_sha=39a2879c35bdc0fc87c67f05adc11e5766f7dae61792c75f44653450b7ee04da
expected_bernini_data_sha=29aa4f89579c7771cb9f78706fde4f0dca0de954fdb2f5e2de1abacd8a0d6c65
expected_pack_vae_latents_source_sha=445893fee2cca1f745265cea857740937f338a04b67e9f895fef943948c49c9f
expected_process_renderer_sample_source_sha=9e8532898267ea167f0776a71a30233cbfada4f94132e0b546f1740115ee372e
expected_raft_weights_sha=ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322
expected_scorer_sha=04cbb9093fc6e59b9db996fed05ba8aa121652f81bfbca6fd0b63f0c3802c041

flow_extractor="$method_root/extract_anchor_raft_flow_v1.py"
g0_middle_extractor="$method_root/materialize_decoded_middle_action_repr_v1.py"
middle_extractor="$method_root/materialize_decoded_middle_action_repr_v1.py"
video_controls="$method_root/materialize_exact81_video_controls_v1.py"
flow_controls="$method_root/materialize_g1_flow_control_cohort_v1.py"
middle_controls="$method_root/materialize_g1_middle_control_cohort_v1.py"
g1_evaluator="$method_root/evaluate_g1_action_repr_selectivity_v1.py"
g1_scorer="$method_root/score_g1_joint_action_repr_admission_v1.py"
g2a_test="$test_root/test_action_repr_g2a_adapter_v1.py"
g2a_world4_test="$test_root/test_audit_action_repr_g2a_world4_v1.py"
g2a_world4_runner="$method_root/audit_action_repr_g2a_world4_v1.py"

fail() {
  echo "[stage1-v2] ERROR: $*" >&2
  exit 1
}

usage() {
  cat >&2 <<'EOF'
usage:
  auh_stage1_action_repr_gates_20260824_v2.sh preflight
  auh_stage1_action_repr_gates_20260824_v2.sh launch-target-canary [147881]
  auh_stage1_action_repr_gates_20260824_v2.sh target-controls CASE_ID
  auh_stage1_action_repr_gates_20260824_v2.sh launch-target-repr CASE_ID correct|temporal_shuffle|reverse [JOB_ID]
  auh_stage1_action_repr_gates_20260824_v2.sh target-g1 CASE_ID
  auh_stage1_action_repr_gates_20260824_v2.sh target-eval CASE_ID
  auh_stage1_action_repr_gates_20260824_v2.sh target-admission
  auh_stage1_action_repr_gates_20260824_v2.sh g2a-cpu-api-audit
  auh_stage1_action_repr_gates_20260824_v2.sh launch-g2a-production [147881]
  auh_stage1_action_repr_gates_20260824_v2.sh g2a-production-status

worker-target-repr and worker-g2a-production are internal srun entrypoints.
There is intentionally no training, optimizer, decode, checkpoint, LoRA, or
automatic expansion mode.
EOF
  exit 2
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_sha() {
  local path="$1" expected="$2" actual
  test -f "$path" || fail "missing regular file: $path"
  test ! -L "$path" || fail "symlink is forbidden: $path"
  actual="$(sha256_file "$path")"
  test "$actual" = "$expected" || fail "SHA-256 mismatch: $path expected=$expected actual=$actual"
}

manifest_value() {
  local iid="$1" expression="$2"
  jq -er --arg iid "$iid" ".cases[] | select(.case_id == \$iid) | $expression" "$manifest"
}

require_case() {
  local iid="$1"
  [[ "$iid" =~ ^[0-9a-f]{12}$ ]] || fail "unsafe case id: $iid"
  test "$(jq -r --arg iid "$iid" '[.cases[] | select(.case_id == $iid)] | length' "$manifest")" = 1 \
    || fail "case is not uniquely sealed in the v1 manifest: $iid"
}

source_pin_path() {
  local name="$1"
  case "$name" in
    test_action_repr_g2a_adapter_v1.py) printf '%s/%s\n' "$test_root" "$name" ;;
    test_*.py) printf '%s/tests/%s\n' "$method_root" "$name" ;;
    *) printf '%s/%s\n' "$method_root" "$name" ;;
  esac
}

common_preflight() {
  local name expected path relative_path corrected corrected_old vendor_relative_path vendor_data_path
  test -x "$python_bin" || fail "vace Python is unavailable"
  test -x "$(command -v jq)" || fail "jq is unavailable"
  test -x "$(command -v sha256sum)" || fail "sha256sum is unavailable"
  test -d "$bernini_root" || fail "pinned Bernini tree is unavailable"
  test -d "$veomni_root" || fail "pinned VeOmni tree is unavailable"
  test -d "$base_checkpoint" || fail "pinned Bernini base checkpoint is unavailable"
  require_sha "$manifest" "$expected_manifest_sha"
  require_sha "$prereg" "$expected_prereg_sha"
  require_sha "$prereg_addendum" "$expected_addendum_sha"
  require_sha "$posterior_identity_addendum" "$expected_posterior_identity_addendum_sha"
  require_sha "$matched_noise_addendum" "$expected_matched_noise_addendum_sha"
  require_sha "$g2a_six_route_addendum" "$expected_g2a_six_route_addendum_sha"
  require_sha "$explicit_gaussian_authority_addendum" "$expected_explicit_gaussian_authority_addendum_sha"
  require_sha "$g1_authority_fixture_addendum" "$expected_g1_authority_fixture_addendum_sha"
  require_sha "$deterministic_vae_authority_addendum" "$expected_deterministic_vae_authority_addendum_sha"
  require_sha "$quantized_energy_match_addendum" "$expected_quantized_energy_match_addendum_sha"
  require_sha "$g0_middle_extractor" "$(jq -er '.corrected_source_hash_pins["materialize_decoded_middle_action_repr_v1.py"].new_sha256' "$deterministic_vae_authority_addendum")"
  test "$(jq -r '.optimizer_creation_authorized' "$prereg")" = false \
    || fail "prereg unexpectedly authorizes an optimizer"
  test "$(jq -r '.optimization_steps' "$prereg")" = 0 \
    || fail "prereg optimization-step closure differs"
  test "$(jq -r '.parameter_updates_authorized' "$prereg")" = false \
    || fail "prereg unexpectedly authorizes parameter updates"
  test "$(jq -r '.input_authority.manifest_sha256' "$prereg")" = "$expected_manifest_sha" \
    || fail "prereg input-manifest pin differs"
  test "$(jq -r '.canary.case_id' "$prereg")" = 0be6494dfac3 \
    || fail "preregistered initial canary case differs"
  test "$(jq -r '.canary.input_role' "$prereg")" = real_forward \
    || fail "preregistered initial canary role differs"
  test "$(jq -r '.canary.slurm_job_id' "$prereg")" = 147881 \
    || fail "preregistered initial canary job differs"
  test "$(jq -r '.canary.node' "$prereg")" = auh7-1b-gpu-213 \
    || fail "preregistered initial canary node differs"
  test "$(jq -r '.g2a_current_boundary.production_WORLD4_renderer_integration_closed' "$prereg")" = false \
    || fail "prereg unexpectedly closes production G2a"
  test "$(jq -r '.source_preregistration_sha256' "$prereg_addendum")" = "$expected_prereg_sha" \
    || fail "source-lock addendum is not bound to the frozen prereg"
  test "$(jq -r '.canonical_preregistration.sha256' "$posterior_identity_addendum")" = "$expected_prereg_sha" \
    || fail "posterior-identity addendum is not bound to the frozen prereg"
  test "$(jq -r '.prior_source_lock_addendum.sha256' "$posterior_identity_addendum")" = "$expected_addendum_sha" \
    || fail "posterior-identity addendum is not chained to the v2.1 source lock"
  test "$(jq -r '.prior_source_lock_addendum.sha256' "$matched_noise_addendum")" = "$expected_addendum_sha" \
    || fail "matched-noise addendum is not bound to the v2.1 source lock"
  test "$(jq -r '.prior_posterior_identity_addendum.sha256' "$matched_noise_addendum")" = "$expected_posterior_identity_addendum_sha" \
    || fail "matched-noise addendum is not chained to posterior identity"
  test "$(jq -r '.prior_matched_noise_addendum.sha256' "$g2a_six_route_addendum")" = "$expected_matched_noise_addendum_sha" \
    || fail "six-route G2a addendum is not chained to matched-noise authority"
  test "$(jq -r '.canonical_preregistration.sha256' "$g2a_six_route_addendum")" = "$expected_prereg_sha" \
    || fail "six-route G2a addendum is not bound to the frozen prereg"
  test "$(jq -r '.canonical_preregistration.sha256' "$explicit_gaussian_authority_addendum")" = "$expected_prereg_sha" \
    || fail "explicit-Gaussian addendum is not bound to the frozen prereg"
  test "$(jq -r '.prior_source_lock_addendum.sha256' "$explicit_gaussian_authority_addendum")" = "$expected_addendum_sha" \
    || fail "explicit-Gaussian addendum is not chained to the v2.1 source lock"
  test "$(jq -r '.prior_posterior_identity_addendum.sha256' "$explicit_gaussian_authority_addendum")" = "$expected_posterior_identity_addendum_sha" \
    || fail "explicit-Gaussian addendum is not chained to posterior identity"
  test "$(jq -r '.prior_matched_noise_addendum.sha256' "$explicit_gaussian_authority_addendum")" = "$expected_matched_noise_addendum_sha" \
    || fail "explicit-Gaussian addendum is not chained to matched-noise authority"
  test "$(jq -r '.prior_g2a_six_route_addendum.sha256' "$explicit_gaussian_authority_addendum")" = "$expected_g2a_six_route_addendum_sha" \
    || fail "explicit-Gaussian addendum is not chained to six-route G2a"
  test "$(jq -r '.canonical_preregistration.sha256' "$g1_authority_fixture_addendum")" = "$expected_prereg_sha" \
    || fail "G1-authority fixture addendum is not bound to the frozen prereg"
  test "$(jq -r '.prior_explicit_gaussian_authority_addendum.sha256' "$g1_authority_fixture_addendum")" = "$expected_explicit_gaussian_authority_addendum_sha" \
    || fail "G1-authority fixture addendum is not chained to explicit Gaussian authority"
  test "$(jq -r '.canonical_preregistration.sha256' "$deterministic_vae_authority_addendum")" = "$expected_prereg_sha" \
    || fail "deterministic-VAE addendum is not bound to the frozen prereg"
  test "$(jq -r '.prior_explicit_gaussian_authority_addendum.sha256' "$deterministic_vae_authority_addendum")" = "$expected_explicit_gaussian_authority_addendum_sha" \
    || fail "deterministic-VAE addendum is not chained to explicit Gaussian authority"
  test "$(jq -r '.prior_g1_authority_fixture_addendum.sha256' "$deterministic_vae_authority_addendum")" = "$expected_g1_authority_fixture_addendum_sha" \
    || fail "deterministic-VAE addendum is not chained to the G1-authority fixture correction"
  test "$(jq -r '.canonical_preregistration.sha256' "$quantized_energy_match_addendum")" = "$expected_prereg_sha" \
    || fail "quantized-energy addendum is not bound to the frozen prereg"
  test "$(jq -r '.prior_deterministic_vae_authority_addendum.sha256' "$quantized_energy_match_addendum")" = "$expected_deterministic_vae_authority_addendum_sha" \
    || fail "quantized-energy addendum is not chained to deterministic VAE authority"
  test "$(jq -r '.runtime_boundary.optimizer_created' "$posterior_identity_addendum")" = false \
    || fail "posterior-identity correction unexpectedly created an optimizer"
  test "$(jq -r '.runtime_boundary.optimization_steps' "$posterior_identity_addendum")" = 0 \
    || fail "posterior-identity correction step count differs"
  test "$(jq -r '.runtime_boundary.optimizer_created' "$matched_noise_addendum")" = false \
    || fail "matched-noise correction unexpectedly created an optimizer"
  test "$(jq -r '.runtime_boundary.optimization_steps' "$matched_noise_addendum")" = 0 \
    || fail "matched-noise correction step count differs"
  test "$(jq -r '.zero_optimizer_contract.optimizer_created' "$g2a_six_route_addendum")" = false \
    || fail "six-route G2a correction unexpectedly created an optimizer"
  test "$(jq -r '.zero_optimizer_contract.optimization_steps' "$g2a_six_route_addendum")" = 0 \
    || fail "six-route G2a correction step count differs"
  test "$(jq -r '.g2a_six_route_step0_contract.required_routes_in_order | join(",")' "$g2a_six_route_addendum")" = correct,zero,temporal_shuffle,reverse,incomplete,wrong_action \
    || fail "six-route G2a order differs"
  test "$(jq -r '.production_world4_boundary.production_WORLD4_G2a_status' "$g2a_six_route_addendum")" = PENDING_UNTIL_G1_TARGET_PASS \
    || fail "production G2a pre-G1 boundary differs"
  test "$(jq -r '.zero_optimizer_contract.optimizer_created' "$explicit_gaussian_authority_addendum")" = false \
    || fail "explicit-Gaussian correction unexpectedly created an optimizer"
  test "$(jq -r '.zero_optimizer_contract.optimizer_creation_authorized_by_this_addendum' "$explicit_gaussian_authority_addendum")" = false \
    || fail "explicit-Gaussian correction unexpectedly authorizes an optimizer"
  test "$(jq -r '.zero_optimizer_contract.optimization_steps' "$explicit_gaussian_authority_addendum")" = 0 \
    || fail "explicit-Gaussian correction step count differs"
  test "$(jq -r '.zero_optimizer_contract.parameter_updates' "$explicit_gaussian_authority_addendum")" = 0 \
    || fail "explicit-Gaussian correction update count differs"
  test "$(jq -r '.explicit_prepack_gaussian_contract.seed_binding' "$explicit_gaussian_authority_addendum")" = domain_plus_base_seed_plus_case_id_plus_instruction_sha256_not_control_video_sha256 \
    || fail "explicit Gaussian seed binding differs"
  test "$(jq -r '.explicit_prepack_gaussian_contract.same_case_correct_shuffle_reverse_authority_required' "$explicit_gaussian_authority_addendum")" = true \
    || fail "same-case temporal controls do not share explicit Gaussian authority"
  test "$(jq -r '.explicit_prepack_gaussian_contract.recovered_from_x_or_velocity' "$explicit_gaussian_authority_addendum")" = false \
    || fail "explicit Gaussian authority unexpectedly permits inverse recovery"
  test "$(jq -r '.ordered_execution_boundary.fresh_source_root' "$explicit_gaussian_authority_addendum")" = source_v2_4 \
    || fail "explicit-Gaussian source root differs"
  test "$(jq -r '.ordered_execution_boundary.fresh_stage_root' "$explicit_gaussian_authority_addendum")" = stage1_v2_4 \
    || fail "explicit-Gaussian stage root differs"
  test "$(jq -r '.ordered_execution_boundary.required_order | join(",")' "$explicit_gaussian_authority_addendum")" = AUH_source_tests_and_preflight,single_case_target_G0,eight_case_G1_target,production_WORLD4_six_route_G2a,target_T0_then_TP_optimizer_experiment,independent_G1_selfgen_before_any_selfgen_optimizer_experiment \
    || fail "explicit-Gaussian ordered execution boundary differs"
  test "$(jq -r '.ordered_execution_boundary.expand_to_G1_target_only_after_G0_pass' "$explicit_gaussian_authority_addendum")" = true \
    || fail "G0-to-G1 target boundary differs"
  test "$(jq -r '.ordered_execution_boundary.run_production_G2a_only_after_G1_target_pass' "$explicit_gaussian_authority_addendum")" = true \
    || fail "G1-target-to-G2a boundary differs"
  test "$(jq -r '.ordered_execution_boundary.create_target_optimizer_only_after_G0_G1_target_and_G2a_pass' "$explicit_gaussian_authority_addendum")" = true \
    || fail "target optimizer boundary differs"
  test "$(jq -r '.runtime_boundary.optimizer_created' "$g1_authority_fixture_addendum")" = false \
    || fail "G1-authority fixture correction unexpectedly created an optimizer"
  test "$(jq -r '.runtime_boundary.optimization_steps' "$g1_authority_fixture_addendum")" = 0 \
    || fail "G1-authority fixture correction step count differs"
  test "$(jq -r '.runtime_boundary.parameter_updates' "$g1_authority_fixture_addendum")" = 0 \
    || fail "G1-authority fixture correction update count differs"
  test "$(jq -r '.runtime_boundary.test_fixture_only' "$g1_authority_fixture_addendum")" = true \
    || fail "G1-authority correction is not fixture-only"
  test "$(jq -r '.test_rerun_contract.fresh_source_root' "$g1_authority_fixture_addendum")" = source_v2_5 \
    || fail "G1-authority fixture rerun source root differs"
  test "$(jq -r '.test_rerun_contract.fresh_stage_root' "$g1_authority_fixture_addendum")" = stage1_v2_5 \
    || fail "G1-authority fixture rerun stage root differs"
  test "$(jq -r '.test_rerun_contract.pythonpath_must_include_source_root' "$g1_authority_fixture_addendum")" = true \
    || fail "G1-authority fixture rerun does not require the source root in PYTHONPATH"
  test "$(jq -r '.test_rerun_contract.pythonpath_must_include_method_root' "$g1_authority_fixture_addendum")" = true \
    || fail "G1-authority fixture rerun does not require the method root in PYTHONPATH"
  test "$(jq -r '.test_rerun_contract.full_methods_suite_required' "$g1_authority_fixture_addendum")" = true \
    || fail "G1-authority fixture rerun does not require the full methods suite"
  test "$(jq -r '.test_rerun_contract.root_G2a_and_launcher_static_suite_required' "$g1_authority_fixture_addendum")" = true \
    || fail "G1-authority fixture rerun does not require root G2a and launcher tests"
  test "$(jq -r '.test_rerun_contract.launcher_preflight_required' "$g1_authority_fixture_addendum")" = true \
    || fail "G1-authority fixture rerun does not require launcher preflight"
  test "$(jq -r '.test_rerun_contract.single_case_target_G0_only_after_all_tests_pass' "$g1_authority_fixture_addendum")" = true \
    || fail "G1-authority fixture rerun does not keep G0 behind all source tests"
  test "$(jq -r '.deterministic_vae_authority_contract.authority_kind' "$deterministic_vae_authority_addendum")" = rank0_local_strict_deterministic_vae_encode \
    || fail "deterministic-VAE authority kind differs"
  test "$(jq -r '.deterministic_vae_authority_contract.policy' "$deterministic_vae_authority_addendum")" = rank0_two_branch_vae_encode_in_local_strict_deterministic_scope_with_exact_flag_restoration_v1 \
    || fail "deterministic-VAE policy differs"
  test "$(jq -r '.deterministic_vae_authority_contract.producer_rank' "$deterministic_vae_authority_addendum")" = 0 \
    || fail "deterministic-VAE producer rank differs"
  test "$(jq -r '.deterministic_vae_authority_contract.encode_call_count' "$deterministic_vae_authority_addendum")" = 2 \
    || fail "deterministic-VAE encode call count differs"
  test "$(jq -r '.deterministic_vae_authority_contract.scope' "$deterministic_vae_authority_addendum")" = action_and_first_frame_repeat_encode_calls_only \
    || fail "deterministic-VAE scope differs"
  test "$(jq -r '.deterministic_vae_authority_contract.during_flags.deterministic_algorithms_enabled' "$deterministic_vae_authority_addendum")" = true \
    || fail "deterministic algorithms are not enabled inside the VAE authority scope"
  test "$(jq -r '.deterministic_vae_authority_contract.during_flags.deterministic_algorithms_warn_only' "$deterministic_vae_authority_addendum")" = false \
    || fail "deterministic algorithms unexpectedly use warn-only mode"
  test "$(jq -r '.deterministic_vae_authority_contract.during_flags.cudnn_deterministic' "$deterministic_vae_authority_addendum")" = true \
    || fail "deterministic-VAE cuDNN deterministic flag differs"
  test "$(jq -r '.deterministic_vae_authority_contract.during_flags.cudnn_benchmark' "$deterministic_vae_authority_addendum")" = false \
    || fail "deterministic-VAE cuDNN benchmark flag differs"
  test "$(jq -r '.deterministic_vae_authority_contract.restore_preexisting_flags_on_success_and_exception' "$deterministic_vae_authority_addendum")" = true \
    || fail "deterministic-VAE scope does not restore preexisting flags"
  test "$(jq -r '.deterministic_vae_authority_contract.raw_action_noop_posterior_phase0_bit_exact_required' "$deterministic_vae_authority_addendum")" = true \
    || fail "deterministic-VAE raw posterior phase0 gate differs"
  test "$(jq -r '.deterministic_vae_authority_contract.sampled_clean_phase0_bit_exact_required' "$deterministic_vae_authority_addendum")" = true \
    || fail "deterministic-VAE sampled-clean phase0 gate differs"
  jq -e '.deterministic_vae_authority_contract.phase0_match_atol == 0' \
    "$deterministic_vae_authority_addendum" >/dev/null \
    || fail "deterministic-VAE phase0 tolerance is not exact"
  test "$(jq -r '.deterministic_vae_authority_contract.posterior_modified_after_encode' "$deterministic_vae_authority_addendum")" = false \
    || fail "deterministic-VAE authority unexpectedly modifies the posterior"
  test "$(jq -r '.deterministic_vae_authority_contract.posterior_copy_or_splice_used' "$deterministic_vae_authority_addendum")" = false \
    || fail "deterministic-VAE authority unexpectedly copies or splices the posterior"
  test "$(jq -r '.deterministic_vae_authority_contract.posterior_or_clean_latent_received_by_trainer' "$deterministic_vae_authority_addendum")" = false \
    || fail "deterministic-VAE authority leaks posterior or clean latent to the trainer"
  test "$(jq -r '.deterministic_vae_authority_contract.no_posterior_or_absolute_clean_latent_persisted' "$deterministic_vae_authority_addendum")" = true \
    || fail "deterministic-VAE persistence firewall differs"
  test "$(jq -r '.deterministic_vae_authority_contract.explicit_prepack_gaussian_contract_unchanged' "$deterministic_vae_authority_addendum")" = true \
    || fail "deterministic-VAE addendum unexpectedly changes explicit Gaussian authority"
  test "$(jq -r '.zero_optimizer_contract.optimizer_created' "$deterministic_vae_authority_addendum")" = false \
    || fail "deterministic-VAE correction unexpectedly created an optimizer"
  test "$(jq -r '.zero_optimizer_contract.optimizer_creation_authorized_by_this_addendum' "$deterministic_vae_authority_addendum")" = false \
    || fail "deterministic-VAE correction unexpectedly authorizes optimizer creation"
  test "$(jq -r '.zero_optimizer_contract.optimization_steps' "$deterministic_vae_authority_addendum")" = 0 \
    || fail "deterministic-VAE correction step count differs"
  test "$(jq -r '.zero_optimizer_contract.parameter_updates' "$deterministic_vae_authority_addendum")" = 0 \
    || fail "deterministic-VAE correction update count differs"
  test "$(jq -r '.zero_optimizer_contract.checkpoint_or_lora_created' "$deterministic_vae_authority_addendum")" = false \
    || fail "deterministic-VAE correction unexpectedly created a checkpoint or LoRA"
  test "$(jq -r '.ordered_execution_boundary.fresh_source_root' "$deterministic_vae_authority_addendum")" = source_v2_6 \
    || fail "deterministic-VAE source root differs"
  test "$(jq -r '.ordered_execution_boundary.fresh_stage_root' "$deterministic_vae_authority_addendum")" = stage1_v2_6 \
    || fail "deterministic-VAE stage root differs"
  test "$(jq -r '.ordered_execution_boundary.fresh_log_root' "$deterministic_vae_authority_addendum")" = logs/stage1_v2_6 \
    || fail "deterministic-VAE log root differs"
  test "$(jq -r '.ordered_execution_boundary.required_order | join(",")' "$deterministic_vae_authority_addendum")" = AUH_source_tests_and_preflight,single_case_target_G0,eight_case_G1_target,production_WORLD4_six_route_G2a,target_T0_then_TP_optimizer_experiment,independent_G1_selfgen_before_any_selfgen_optimizer_experiment \
    || fail "deterministic-VAE ordered execution boundary differs"
  test "$(jq -r '.ordered_execution_boundary.pythonpath_must_include_source_root' "$deterministic_vae_authority_addendum")" = true \
    || fail "deterministic-VAE rerun does not require the source root in PYTHONPATH"
  test "$(jq -r '.ordered_execution_boundary.pythonpath_must_include_method_root' "$deterministic_vae_authority_addendum")" = true \
    || fail "deterministic-VAE rerun does not require the method root in PYTHONPATH"
  test "$(jq -r '.ordered_execution_boundary.full_methods_suite_required' "$deterministic_vae_authority_addendum")" = true \
    || fail "deterministic-VAE rerun does not require the full methods suite"
  test "$(jq -r '.ordered_execution_boundary.root_G2a_and_launcher_static_suite_required' "$deterministic_vae_authority_addendum")" = true \
    || fail "deterministic-VAE rerun does not require root G2a and launcher tests"
  test "$(jq -r '.ordered_execution_boundary.launcher_preflight_required' "$deterministic_vae_authority_addendum")" = true \
    || fail "deterministic-VAE rerun does not require launcher preflight"
  test "$(jq -r '.ordered_execution_boundary.single_case_target_G0_only_after_all_tests_pass' "$deterministic_vae_authority_addendum")" = true \
    || fail "deterministic-VAE rerun does not keep G0 behind all source tests"
  test "$(jq -r '.ordered_execution_boundary.expand_to_G1_target_only_after_G0_pass' "$deterministic_vae_authority_addendum")" = true \
    || fail "deterministic-VAE G0-to-G1 target boundary differs"
  test "$(jq -r '.ordered_execution_boundary.run_production_G2a_only_after_G1_target_pass' "$deterministic_vae_authority_addendum")" = true \
    || fail "deterministic-VAE G1-target-to-G2a boundary differs"
  test "$(jq -r '.ordered_execution_boundary.create_target_optimizer_only_after_G0_G1_target_and_G2a_pass' "$deterministic_vae_authority_addendum")" = true \
    || fail "deterministic-VAE target optimizer boundary differs"
  test "$(jq -r '.quantized_energy_match_contract.energy_definition' "$quantized_energy_match_addendum")" = 'sqrt(sum(all_block_residual_squared)/number_of_residual_scalars)' \
    || fail "quantized-energy definition differs"
  test "$(jq -r '.quantized_energy_match_contract.analytic_scale' "$quantized_energy_match_addendum")" = correct_energy_divided_by_wrong_action_donor_energy \
    || fail "quantized-energy analytic scale differs"
  jq -e '.quantized_energy_match_contract.scale_lower_bound == 0.01' \
    "$quantized_energy_match_addendum" >/dev/null \
    || fail "quantized-energy lower scale bound differs"
  jq -e '.quantized_energy_match_contract.scale_upper_bound == 100.0' \
    "$quantized_energy_match_addendum" >/dev/null \
    || fail "quantized-energy upper scale bound differs"
  test "$(jq -r '.quantized_energy_match_contract.bracket_lower' "$quantized_energy_match_addendum")" = 'max(scale_lower_bound,analytic_scale/2)' \
    || fail "quantized-energy lower bracket differs"
  test "$(jq -r '.quantized_energy_match_contract.bracket_upper' "$quantized_energy_match_addendum")" = 'min(scale_upper_bound,analytic_scale*2)' \
    || fail "quantized-energy upper bracket differs"
  test "$(jq -r '.quantized_energy_match_contract.quantized_bracket_must_straddle_target' "$quantized_energy_match_addendum")" = true \
    || fail "quantized-energy bracket need not straddle the target"
  test "$(jq -r '.quantized_energy_match_contract.calibration_compute_dtype' "$quantized_energy_match_addendum")" = torch.float64_on_cpu \
    || fail "quantized-energy calibration is not FP64 on CPU"
  test "$(jq -r '.quantized_energy_match_contract.candidate_publication_dtype' "$quantized_energy_match_addendum")" = exact_original_donor_tensor_dtype \
    || fail "quantized-energy candidate publication dtype differs"
  test "$(jq -r '.quantized_energy_match_contract.calibration_iterations' "$quantized_energy_match_addendum")" = 32 \
    || fail "quantized-energy calibration iteration count differs"
  test "$(jq -r '.quantized_energy_match_contract.candidate_selection_order | join(",")' "$quantized_energy_match_addendum")" = absolute_relative_energy_error,absolute_scale_distance_from_analytic_scale,scale \
    || fail "quantized-energy candidate selection order differs"
  test "$(jq -r '.quantized_energy_match_contract.final_candidate_requantization_and_exact_energy_replay_required' "$quantized_energy_match_addendum")" = true \
    || fail "quantized-energy final candidate replay is not required"
  jq -e '.quantized_energy_match_contract.energy_match_rtol == 0.00002' \
    "$quantized_energy_match_addendum" >/dev/null \
    || fail "quantized-energy tolerance differs from 2e-5"
  test "$(jq -r '.quantized_energy_match_contract.energy_match_rtol_changed' "$quantized_energy_match_addendum")" = false \
    || fail "quantized-energy tolerance was widened"
  test "$(jq -r '.quantized_energy_match_contract.maximum_energy_scale_changed' "$quantized_energy_match_addendum")" = false \
    || fail "quantized-energy maximum scale was changed"
  test "$(jq -r '.quantized_energy_match_contract.wrong_action_donor_cycle_changed' "$quantized_energy_match_addendum")" = false \
    || fail "quantized-energy wrong-action donor cycle was changed"
  test "$(jq -r '.quantized_energy_match_contract.output_dtype_promotion_authorized' "$quantized_energy_match_addendum")" = false \
    || fail "quantized-energy output dtype promotion is unexpectedly authorized"
  test "$(jq -r '.quantized_energy_match_contract.optimizer_or_trainer_access' "$quantized_energy_match_addendum")" = false \
    || fail "quantized-energy calibration unexpectedly permits optimizer or trainer access"
  test "$(jq -r '.zero_optimizer_contract.optimizer_created' "$quantized_energy_match_addendum")" = false \
    || fail "quantized-energy correction unexpectedly created an optimizer"
  test "$(jq -r '.zero_optimizer_contract.optimizer_creation_authorized_by_this_addendum' "$quantized_energy_match_addendum")" = false \
    || fail "quantized-energy correction unexpectedly authorizes optimizer creation"
  test "$(jq -r '.zero_optimizer_contract.optimization_steps' "$quantized_energy_match_addendum")" = 0 \
    || fail "quantized-energy correction step count differs"
  test "$(jq -r '.zero_optimizer_contract.parameter_updates' "$quantized_energy_match_addendum")" = 0 \
    || fail "quantized-energy correction update count differs"
  test "$(jq -r '.zero_optimizer_contract.checkpoint_or_lora_created' "$quantized_energy_match_addendum")" = false \
    || fail "quantized-energy correction unexpectedly created a checkpoint or LoRA"
  test "$(jq -r '.ordered_execution_boundary.fresh_source_root' "$quantized_energy_match_addendum")" = source_v2_7 \
    || fail "quantized-energy source root differs"
  test "$(jq -r '.ordered_execution_boundary.fresh_stage_root' "$quantized_energy_match_addendum")" = stage1_v2_7 \
    || fail "quantized-energy stage root differs"
  test "$(jq -r '.ordered_execution_boundary.fresh_log_root' "$quantized_energy_match_addendum")" = logs/stage1_v2_7 \
    || fail "quantized-energy log root differs"
  test "$(jq -r '.ordered_execution_boundary.required_order | join(",")' "$quantized_energy_match_addendum")" = AUH_source_tests_and_preflight,single_case_target_G0,eight_case_G1_target,production_WORLD4_six_route_G2a,target_T0_then_TP_optimizer_experiment,independent_G1_selfgen_before_any_selfgen_optimizer_experiment \
    || fail "quantized-energy ordered execution boundary differs"
  test "$(jq -r '.ordered_execution_boundary.pythonpath_must_include_source_root' "$quantized_energy_match_addendum")" = true \
    || fail "quantized-energy rerun does not require the source root in PYTHONPATH"
  test "$(jq -r '.ordered_execution_boundary.pythonpath_must_include_method_root' "$quantized_energy_match_addendum")" = true \
    || fail "quantized-energy rerun does not require the method root in PYTHONPATH"
  test "$(jq -r '.ordered_execution_boundary.full_methods_suite_required' "$quantized_energy_match_addendum")" = true \
    || fail "quantized-energy rerun does not require the full methods suite"
  test "$(jq -r '.ordered_execution_boundary.root_G2a_and_launcher_static_suite_required' "$quantized_energy_match_addendum")" = true \
    || fail "quantized-energy rerun does not require root G2a and launcher tests"
  test "$(jq -r '.ordered_execution_boundary.launcher_preflight_required' "$quantized_energy_match_addendum")" = true \
    || fail "quantized-energy rerun does not require launcher preflight"
  test "$(jq -r '.ordered_execution_boundary.single_case_target_G0_only_after_all_tests_pass' "$quantized_energy_match_addendum")" = true \
    || fail "quantized-energy rerun does not keep G0 behind all source tests"
  test "$(jq -r '.ordered_execution_boundary.expand_to_G1_target_only_after_G0_pass' "$quantized_energy_match_addendum")" = true \
    || fail "quantized-energy G0-to-G1 target boundary differs"
  test "$(jq -r '.ordered_execution_boundary.run_production_G2a_only_after_G1_target_pass' "$quantized_energy_match_addendum")" = true \
    || fail "quantized-energy G1-target-to-G2a boundary differs"
  test "$(jq -r '.ordered_execution_boundary.create_target_optimizer_only_after_G0_G1_target_and_G2a_pass' "$quantized_energy_match_addendum")" = true \
    || fail "quantized-energy target optimizer boundary differs"
  test "$(jq -r '.ordered_execution_boundary.reuse_source_v2_6_representation_outputs_as_canonical_v2_7_gate_evidence' "$quantized_energy_match_addendum")" = false \
    || fail "quantized-energy rerun unexpectedly reuses v2.6 representation outputs as canonical evidence"
  vendor_relative_path="$(jq -er '.pinned_vendor_authority.relative_path' "$explicit_gaussian_authority_addendum")"
  test "$vendor_relative_path" = bernini/training/data.py \
    || fail "pinned Bernini authority path differs"
  test "$(jq -r '.pinned_vendor_authority.bernini_revision' "$explicit_gaussian_authority_addendum")" = 2d2b4591 \
    || fail "pinned Bernini authority revision differs"
  test "$(jq -r '.pinned_vendor_authority.file_sha256' "$explicit_gaussian_authority_addendum")" = "$expected_bernini_data_sha" \
    || fail "pinned Bernini data.py hash differs"
  test "$(jq -r '.pinned_vendor_authority.pack_vae_latents_source_sha256' "$explicit_gaussian_authority_addendum")" = "$expected_pack_vae_latents_source_sha" \
    || fail "pinned pack_vae_latents source hash differs"
  test "$(jq -r '.pinned_vendor_authority.process_renderer_sample_source_sha256' "$explicit_gaussian_authority_addendum")" = "$expected_process_renderer_sample_source_sha" \
    || fail "pinned process_renderer_sample source hash differs"
  vendor_data_path="$bernini_root/$vendor_relative_path"
  require_sha "$vendor_data_path" "$expected_bernini_data_sha"
  while IFS=$'\t' read -r name expected; do
    corrected="$(jq -r --arg name "$name" '((.corrected_source_hash_pins // {})[$name].new_sha256) // empty' "$prereg_addendum")"
    if [ -n "$corrected" ]; then
      expected="$corrected"
    fi
    corrected="$(jq -r --arg name "$name" '((.corrected_source_hash_pins // {})[$name].new_sha256) // empty' "$posterior_identity_addendum")"
    if [ -n "$corrected" ]; then
      corrected_old="$(jq -er --arg name "$name" '.corrected_source_hash_pins[$name].old_sha256' "$posterior_identity_addendum")"
      test "$expected" = "$corrected_old" \
        || fail "posterior-identity source-lock chain differs for $name"
      expected="$corrected"
    fi
    path="$(source_pin_path "$name")"
    corrected="$(jq -r --arg name "$name" '((.corrected_source_hash_pins // {})[$name].new_sha256) // empty' "$matched_noise_addendum")"
    if [ -n "$corrected" ]; then
      corrected_old="$(jq -er --arg name "$name" '.corrected_source_hash_pins[$name].old_sha256' "$matched_noise_addendum")"
      test "$expected" = "$corrected_old" \
        || fail "matched-noise source-lock chain differs for $name"
      expected="$corrected"
    fi
    relative_path="${path#"$source_root/"}"
    corrected="$(jq -r --arg name "$relative_path" '((.corrected_source_hash_pins // {})[$name].new_sha256) // empty' "$g2a_six_route_addendum")"
    if [ -n "$corrected" ]; then
      corrected_old="$(jq -er --arg name "$relative_path" '.corrected_source_hash_pins[$name].old_sha256' "$g2a_six_route_addendum")"
      test "$expected" = "$corrected_old" \
        || fail "six-route G2a source-lock chain differs for $relative_path"
      expected="$corrected"
    fi
    corrected="$(jq -r --arg name "$name" '((.corrected_source_hash_pins // {})[$name].new_sha256) // empty' "$explicit_gaussian_authority_addendum")"
    if [ -n "$corrected" ]; then
      corrected_old="$(jq -er --arg name "$name" '.corrected_source_hash_pins[$name].old_sha256' "$explicit_gaussian_authority_addendum")"
      test "$expected" = "$corrected_old" \
        || fail "explicit-Gaussian source-lock chain differs for $name"
      expected="$corrected"
    fi
    corrected="$(jq -r --arg name "$name" '((.corrected_source_hash_pins // {})[$name].new_sha256) // empty' "$g1_authority_fixture_addendum")"
    if [ -n "$corrected" ]; then
      corrected_old="$(jq -er --arg name "$name" '.corrected_source_hash_pins[$name].old_sha256' "$g1_authority_fixture_addendum")"
      test "$expected" = "$corrected_old" \
        || fail "G1-authority fixture source-lock chain differs for $name"
      expected="$corrected"
    fi
    corrected="$(jq -r --arg name "$name" '((.corrected_source_hash_pins // {})[$name].new_sha256) // empty' "$deterministic_vae_authority_addendum")"
    if [ -n "$corrected" ]; then
      corrected_old="$(jq -er --arg name "$name" '.corrected_source_hash_pins[$name].old_sha256' "$deterministic_vae_authority_addendum")"
      test "$expected" = "$corrected_old" \
        || fail "deterministic-VAE source-lock chain differs for $name"
      expected="$corrected"
    fi
    corrected="$(jq -r --arg name "$name" '((.corrected_source_hash_pins // {})[$name].new_sha256) // empty' "$quantized_energy_match_addendum")"
    if [ -n "$corrected" ]; then
      corrected_old="$(jq -er --arg name "$name" '.corrected_source_hash_pins[$name].old_sha256' "$quantized_energy_match_addendum")"
      test "$expected" = "$corrected_old" \
        || fail "quantized-energy source-lock chain differs for $name"
      expected="$corrected"
    fi
    require_sha "$path" "$expected"
  done < <(jq -r '.source_pins | to_entries[] | [.key,.value] | @tsv' "$prereg")
  while IFS=$'\t' read -r name expected; do
    require_sha "$method_root/$name" "$expected"
  done < <(jq -r '.middle_runtime_dependency_pins | to_entries[] | [.key,.value] | @tsv' "$prereg")
  require_sha "$flow_extractor" "$(jq -er '.additional_source_hash_pins["extract_anchor_raft_flow_v1.py"]' "$prereg_addendum")"
  while IFS=$'\t' read -r name expected; do
    require_sha "$source_root/$name" "$expected"
  done < <(jq -r '.additional_source_hash_pins | to_entries[] | [.key,.value] | @tsv' "$g2a_six_route_addendum")
  require_sha "$g1_scorer" "$expected_scorer_sha"
  require_sha "$g2a_test" "$(jq -er '.corrected_source_hash_pins["tests/test_action_repr_g2a_adapter_v1.py"].new_sha256' "$g2a_six_route_addendum")"
  require_sha "$g2a_world4_test" "$(jq -er '.additional_source_hash_pins["tests/test_audit_action_repr_g2a_world4_v1.py"]' "$g2a_six_route_addendum")"
  require_sha "$g2a_world4_runner" "$(jq -er '.additional_source_hash_pins["methods/bernini_action_editing/audit_action_repr_g2a_world4_v1.py"]' "$g2a_six_route_addendum")"
  test "$(jq -r '(.input_authority.fit_cases + .input_authority.heldout_cases) | length' "$prereg")" = 8 \
    || fail "preregistered case count differs"
  test "$(jq -r '.cases | length' "$manifest")" = 8 || fail "v1 manifest case count differs"
  test "$(jq -r '.current_experiment_optimization_steps' "$manifest")" = 0 \
    || fail "v1 manifest current-step count differs"
}

media_preflight() {
  local iid="$1" role path expected
  require_case "$iid"
  for role in source real_forward; do
    path="$(manifest_value "$iid" ".$role.path")"
    expected="$(manifest_value "$iid" ".$role.sha256")"
    require_sha "$path" "$expected"
  done
}

canary_receipt() {
  printf '%s\n' "$stage_root/representations/target/0be6494dfac3/correct/g0_receipt.json"
}

require_canary_complete() {
  local receipt
  receipt="$(canary_receipt)"
  test -f "$receipt" || fail "initial 0be6494dfac3 target real_forward WORLD4 canary is not complete"
  test "$(jq -r '.complete' "$receipt")" = true || fail "initial canary receipt is incomplete"
  test "$(jq -r '.case_id' "$receipt")" = 0be6494dfac3 || fail "initial canary case differs"
  test "$(jq -r '.video_role' "$receipt")" = real_forward || fail "initial canary role differs"
  test "$(jq -r '.optimization_steps' "$receipt")" = 0 || fail "initial canary step count differs"
  test "$(jq -r '.source_lock.matched_noise_addendum_sha256' "$receipt")" = "$expected_matched_noise_addendum_sha" \
    || fail "initial canary predates the matched-noise source lock"
  test "$(jq -r '.source_lock.g2a_six_route_addendum_sha256' "$receipt")" = "$expected_g2a_six_route_addendum_sha" \
    || fail "initial canary predates the six-route source lock"
  test "$(jq -r '.source_lock.explicit_gaussian_authority_addendum_sha256' "$receipt")" = "$expected_explicit_gaussian_authority_addendum_sha" \
    || fail "initial canary predates the explicit-Gaussian source lock"
  test "$(jq -r '.source_lock.g1_authority_fixture_addendum_sha256' "$receipt")" = "$expected_g1_authority_fixture_addendum_sha" \
    || fail "initial canary predates the G1-authority fixture source lock"
  test "$(jq -r '.source_lock.deterministic_vae_authority_addendum_sha256' "$receipt")" = "$expected_deterministic_vae_authority_addendum_sha" \
    || fail "initial canary predates the deterministic-VAE source lock"
  test "$(jq -r '.source_lock.quantized_energy_match_addendum_sha256' "$receipt")" = "$expected_quantized_energy_match_addendum_sha" \
    || fail "initial canary predates the quantized-energy source lock"
}

expected_node_for_job() {
  case "$1" in
    147871) printf '%s\n' auh7-1b-gpu-232 ;;
    147873) printf '%s\n' auh7-1b-gpu-284 ;;
    147881) printf '%s\n' auh7-1b-gpu-213 ;;
    *) fail "parent job is not preregistered: $1" ;;
  esac
}

validate_parent_job() {
  local job="$1" node row
  node="$(expected_node_for_job "$job")"
  row="$(scontrol show job "$job" -o)"
  [[ " $row " == *" JobState=RUNNING "* ]] || fail "parent job is not RUNNING: $job"
  [[ " $row " == *" NodeList=$node "* ]] || fail "parent job/node binding differs: $job/$node"
  if [[ " $row " != *" MinMemoryNode=64G "* && " $row " != *" MinMemoryNode=65536M "* ]]; then
    fail "parent job does not advertise the preregistered 64G host memory: $job"
  fi
}

validate_worker_allocation() {
  local expected_job="$1" expected_node
  expected_node="$(expected_node_for_job "$expected_job")"
  test "${SLURM_JOB_ID:-}" = "$expected_job" || fail "worker SLURM job differs"
  test "$(hostname -s)" = "$expected_node" || fail "worker host differs"
  validate_parent_job "$expected_job"
  export ROCR_VISIBLE_DEVICES=0,1,2,3
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  test "$($python_bin -c 'import torch; print(torch.cuda.device_count())')" = 4 \
    || fail "WORLD4 worker does not see exactly four MI210 devices"
}

cgroup_memory_files() {
  local unified relative base current maximum
  unified="$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)"
  if [ -n "$unified" ]; then
    base="/sys/fs/cgroup${unified}"
    current="$base/memory.current"
    maximum="$base/memory.max"
    if [ -r "$current" ] && [ -r "$maximum" ]; then
      printf '%s\t%s\n' "$current" "$maximum"
      return 0
    fi
  fi
  relative="$(awk -F: '$2 ~ /(^|,)memory(,|$)/ {print $3}' /proc/self/cgroup)"
  base="/sys/fs/cgroup/memory${relative}"
  current="$base/memory.usage_in_bytes"
  maximum="$base/memory.limit_in_bytes"
  [ -r "$current" ] && [ -r "$maximum" ] || return 1
  printf '%s\t%s\n' "$current" "$maximum"
}

memory_guard_preflight() {
  local files current_file max_file current maximum_raw maximum available gib proof_source
  files="$(cgroup_memory_files)" || fail "cannot prove the Slurm memory cgroup"
  IFS=$'\t' read -r current_file max_file <<<"$files"
  current="$(cat "$current_file")"
  maximum_raw="$(cat "$max_file")"
  [[ "$current" =~ ^[0-9]+$ ]] || fail "Slurm memory current value is nonnumeric"
  gib=$((1024 * 1024 * 1024))
  if [[ "$maximum_raw" =~ ^[0-9]+$ ]]; then
    maximum="$maximum_raw"
    proof_source=cgroup_memory_max
  elif [ "$maximum_raw" = max ] && [ "${SLURM_MEM_PER_NODE:-}" = 65536 ]; then
    # This AUH cgroup-v2 deployment exposes memory.max=max even when Slurm
    # assigns the full 64 GiB parent allocation.  Keep a software watchdog at
    # the sealed 64 GiB TRES instead of silently dropping the guard.
    maximum=$((SLURM_MEM_PER_NODE * 1024 * 1024))
    proof_source=slurm_mem_per_node_65536_cgroup_max_unbounded
  else
    fail "Slurm memory limit is neither numeric nor the sealed 65536 MiB allocation"
  fi
  [ "$maximum" -ge $((60 * gib)) ] && [ "$maximum" -le $((70 * gib)) ] \
    || fail "proved cgroup is not the preregistered 64G class: $maximum bytes"
  available=$((maximum - current))
  [ "$available" -ge $((48 * gib)) ] \
    || fail "less than 48 GiB remains before WORLD4 middle extraction: $available bytes"
  printf '%s\t%s\t%s\n' "$current_file" "$maximum" "$proof_source"
}

memory_watchdog() {
  local child="$1" current_file="$2" maximum="$3" marker="$4"
  local threshold current gib
  gib=$((1024 * 1024 * 1024))
  [[ "$maximum" =~ ^[0-9]+$ ]] || fail "memory watchdog limit is nonnumeric"
  threshold=$((maximum - 2 * gib))
  while kill -0 "$child" 2>/dev/null; do
    current="$(cat "$current_file")"
    if [[ "$current" =~ ^[0-9]+$ ]] && [ "$current" -ge "$threshold" ]; then
      printf 'memory_watchdog_limit_bytes=%s\nobserved_bytes=%s\n' "$threshold" "$current" >"$marker"
      kill -TERM "$child" 2>/dev/null || true
      return 0
    fi
    sleep 2
  done
}

run_middle_world4() {
  local log="$1" current_file="$2" maximum="$3"
  shift 3
  local marker child watchdog status
  marker="${log}.memory-guard"
  test ! -e "$marker" || fail "memory guard marker already exists: $marker"
  "$@" >"$log" 2>&1 &
  child=$!
  memory_watchdog "$child" "$current_file" "$maximum" "$marker" &
  watchdog=$!
  status=0
  wait "$child" || status=$?
  kill -TERM "$watchdog" 2>/dev/null || true
  wait "$watchdog" 2>/dev/null || true
  if [ -e "$marker" ]; then
    fail "64G memory watchdog stopped WORLD4 extraction; engineering failure only: $marker"
  fi
  [ "$status" -eq 0 ] || fail "WORLD4 middle extraction failed with status $status"
}

target_control_dir() {
  printf '%s\n' "$stage_root/video_controls/target/$1"
}

target_repr_dir() {
  printf '%s\n' "$stage_root/representations/target/$1/$2"
}

target_video_binding() {
  local iid="$1" role="$2" control_dir receipt path expected
  case "$role" in
    correct)
      manifest_value "$iid" '.real_forward.path'
      manifest_value "$iid" '.real_forward.sha256'
      printf '%s\n' real_forward
      ;;
    temporal_shuffle|reverse)
      control_dir="$(target_control_dir "$iid")"
      receipt="$control_dir/receipt.json"
      test -f "$receipt" || fail "exact81 target controls are missing for $iid"
      test "$(jq -r '.schema_version' "$receipt")" = bernini-exact81-temporal-video-controls-v1 \
        || fail "exact81 target control receipt schema differs"
      test "$(jq -r '.case_id' "$receipt")" = "$iid" || fail "target control case differs"
      test "$(jq -r '.anchor_kind' "$receipt")" = target || fail "target control anchor kind differs"
      path="$control_dir/$role.mp4"
      expected="$(jq -er --arg role "$role" '.controls[$role].sha256' "$receipt")"
      require_sha "$path" "$expected"
      printf '%s\n%s\n%s\n' "$path" "$expected" "$role"
      ;;
    *) fail "target representation role is not sealed: $role" ;;
  esac
}

write_g0_receipt() {
  local iid="$1" role="$2" video_role="$3" video_sha="$4" out="$5" job="$6" cgroup_max="$7" memory_proof_source="$8" selected_middle="$9"
  local flow="$out/flow.safetensors" middle="$out/middle/middle_repr.safetensors"
  local middle_receipt="$out/middle/receipt.json" receipt="$out/g0_receipt.json" temporary
  test ! -e "$receipt" || fail "G0 receipt already exists"
  temporary="$(mktemp "$out/.g0_receipt.XXXXXX")"
  jq -n \
    --arg schema_version bernini-action-repr-stage1-g0-receipt-v2 \
    --arg experiment_id action_repr_target_selfgen_middle_g1_20260824_v2 \
    --arg case_id "$iid" \
    --arg anchor_kind target \
    --arg representation_role "$role" \
    --arg video_role "$video_role" \
    --arg video_sha256 "$video_sha" \
    --arg flow_path "$flow" \
    --arg flow_sha256 "$(sha256_file "$flow")" \
    --arg flow_sidecar_sha256 "$(sha256_file "$out/flow.json")" \
    --arg middle_path "$middle" \
    --arg middle_sha256 "$(sha256_file "$middle")" \
    --arg middle_receipt_sha256 "$(sha256_file "$middle_receipt")" \
    --arg source_prereg_sha256 "$expected_prereg_sha" \
    --arg source_lock_addendum_sha256 "$expected_addendum_sha" \
    --arg posterior_identity_addendum_sha256 "$expected_posterior_identity_addendum_sha" \
    --arg matched_noise_addendum_sha256 "$expected_matched_noise_addendum_sha" \
    --arg g2a_six_route_addendum_sha256 "$expected_g2a_six_route_addendum_sha" \
    --arg explicit_gaussian_authority_addendum_sha256 "$expected_explicit_gaussian_authority_addendum_sha" \
    --arg g1_authority_fixture_addendum_sha256 "$expected_g1_authority_fixture_addendum_sha" \
    --arg deterministic_vae_authority_addendum_sha256 "$expected_deterministic_vae_authority_addendum_sha" \
    --arg quantized_energy_match_addendum_sha256 "$expected_quantized_energy_match_addendum_sha" \
    --arg middle_extractor_source "$selected_middle" \
    --arg middle_extractor_source_sha256 "$(sha256_file "$selected_middle")" \
    --arg parent_job_id "$job" \
    --arg host "$(hostname -s)" \
    --argjson cgroup_memory_limit_bytes "$cgroup_max" \
    --arg memory_limit_proof_source "$memory_proof_source" \
    '{
      schema_version: $schema_version,
      experiment_id: $experiment_id,
      complete: true,
      case_id: $case_id,
      anchor_kind: $anchor_kind,
      representation_role: $representation_role,
      video_role: $video_role,
      input_video_sha256: $video_sha256,
      representations: {
        flow: {path: $flow_path, sha256: $flow_sha256, sidecar_sha256: $flow_sidecar_sha256},
        middle: {path: $middle_path, sha256: $middle_sha256, receipt_sha256: $middle_receipt_sha256}
      },
      runtime: {
        parent_job_id: $parent_job_id,
        host: $host,
        world_size: 4,
        ulysses_size: 4,
        advertised_host_memory_gib: 64,
        cgroup_memory_limit_bytes: $cgroup_memory_limit_bytes,
        memory_limit_proof_source: $memory_limit_proof_source,
        memory_watchdog_headroom_gib: 2
      },
      source_lock: {
        manifest_sha256: "c78e42f0661e5905407505037ce322d32d67ffec0b70b1cab466f895dc8d0632",
        preregistration_sha256: $source_prereg_sha256,
        addendum_sha256: $source_lock_addendum_sha256,
        posterior_identity_addendum_sha256: $posterior_identity_addendum_sha256,
        matched_noise_addendum_sha256: $matched_noise_addendum_sha256,
        g2a_six_route_addendum_sha256: $g2a_six_route_addendum_sha256,
        explicit_gaussian_authority_addendum_sha256: $explicit_gaussian_authority_addendum_sha256,
        g1_authority_fixture_addendum_sha256: $g1_authority_fixture_addendum_sha256,
        deterministic_vae_authority_addendum_sha256: $deterministic_vae_authority_addendum_sha256,
        quantized_energy_match_addendum_sha256: $quantized_energy_match_addendum_sha256,
        middle_extractor_source: $middle_extractor_source,
        middle_extractor_source_sha256: $middle_extractor_source_sha256
      },
      information_firewall: {
        target_video_accessed_by_frozen_extractors_only: true,
        target_video_accessed_by_trainer: false,
        target_rgb_vae_clean_latent_or_absolute_hidden_to_trainer: false,
        detached_flow_and_projected_middle_cache_only: true
      },
      optimizer_created: false,
      optimization_steps: 0,
      parameter_updates: 0,
      training_authority: false,
      method_success_claimed: false
    }' >"$temporary"
  ln "$temporary" "$receipt"
  rm -f "$temporary"
}

run_target_repr_worker() {
  local iid="$1" role="$2" expected_job="$3"
  local binding video video_sha video_role source source_sha out scratch files current_file cgroup_max memory_proof_source selected_middle
  common_preflight
  require_case "$iid"
  media_preflight "$iid"
  validate_worker_allocation "$expected_job"
  if [ "$iid" = 0be6494dfac3 ] && [ "$role" = correct ] && [ "$expected_job" != 147881 ]; then
    fail "initial canary is preregistered on parent job 147881 only"
  fi
  selected_middle="$middle_extractor"
  if [ "$iid" = 0be6494dfac3 ] && [ "$role" = correct ]; then
    selected_middle="$g0_middle_extractor"
  fi
  if [ "$iid" != 0be6494dfac3 ] || [ "$role" != correct ]; then
    require_canary_complete
  fi
  binding="$(target_video_binding "$iid" "$role")"
  video="$(sed -n '1p' <<<"$binding")"
  video_sha="$(sed -n '2p' <<<"$binding")"
  video_role="$(sed -n '3p' <<<"$binding")"
  require_sha "$video" "$video_sha"
  source="$(manifest_value "$iid" '.source.path')"
  source_sha="$(manifest_value "$iid" '.source.sha256')"
  require_sha "$source" "$source_sha"
  out="$(target_repr_dir "$iid" "$role")"
  test ! -e "$out" || fail "representation output must be fresh: $out"
  mkdir -p "$(dirname "$out")"
  mkdir "$out"
  scratch="/tmp/action-repr-stage1-v2-${iid}-${role}-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
  test ! -e "$scratch" || fail "worker scratch already exists: $scratch"
  mkdir -p "$scratch/xdg" "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
  export XDG_CACHE_HOME="$scratch/xdg"
  export MIOPEN_USER_DB_PATH="$scratch/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
  export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions"
  export TRITON_CACHE_DIR="$scratch/triton"
  "$python_bin" -B "$flow_extractor" \
    --source "$source" \
    --anchor "$video" \
    --output "$out/flow.safetensors" \
    --latent-height 46 \
    --latent-width 82 \
    >"$out/flow.log" 2>&1
  test "$(jq -r '.source_sha256' "$out/flow.json")" = "$source_sha" || fail "flow source SHA binding differs"
  test "$(jq -r '.anchor_sha256' "$out/flow.json")" = "$video_sha" || fail "flow anchor SHA binding differs"
  test "$(jq -r '.raft_weights_sha256' "$out/flow.json")" = "$expected_raft_weights_sha" \
    || fail "RAFT weight pin differs"
  files="$(memory_guard_preflight)"
  IFS=$'\t' read -r current_file cgroup_max memory_proof_source <<<"$files"
  run_middle_world4 "$out/middle.log" "$current_file" "$cgroup_max" \
    "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
      "$selected_middle" \
      --bernini-root "$bernini_root" \
      --veomni-root "$veomni_root" \
      --checkpoint "$base_checkpoint" \
      --video "$video" \
      --video-sha256 "$video_sha" \
      --input-role "$video_role" \
      --case-id "$iid" \
      --instruction "$(manifest_value "$iid" '.instruction')" \
      --output "$out/middle" \
      --seed "$(manifest_value "$iid" '.seed')"
  "$python_bin" -B -c \
    'from pathlib import Path; import sys; from materialize_decoded_middle_action_repr_v1 import load_middle_representation_cache; load_middle_representation_cache(Path(sys.argv[1]), Path(sys.argv[2]), expected_role=sys.argv[3])' \
    "$out/middle/middle_repr.safetensors" "$out/middle/receipt.json" "$video_role"
  test "$(jq -r '.training_authority.optimizer_created' "$out/middle/receipt.json")" = false \
    || fail "middle extractor receipt unexpectedly owns an optimizer"
  test "$(jq -r '.training_authority.optimization_steps' "$out/middle/receipt.json")" = 0 \
    || fail "middle extractor receipt step count differs"
  write_g0_receipt "$iid" "$role" "$video_role" "$video_sha" "$out" "$expected_job" "$cgroup_max" "$memory_proof_source" "$selected_middle"
  echo "[stage1-v2] COMPLETE G0 target case=$iid role=$role WORLD4=true optimizer=false output=$out"
}

launch_target_repr() {
  local iid="$1" role="$2" job="$3" log node
  common_preflight
  require_case "$iid"
  validate_parent_job "$job"
  node="$(expected_node_for_job "$job")"
  if [ "$iid" = 0be6494dfac3 ] && [ "$role" = correct ] && [ "$job" != 147881 ]; then
    fail "initial canary is preregistered on parent job 147881 only"
  fi
  if [ "$iid" != 0be6494dfac3 ] || [ "$role" != correct ]; then
    require_canary_complete
  fi
  target_video_binding "$iid" "$role" >/dev/null
  test ! -e "$(target_repr_dir "$iid" "$role")" || fail "representation output is not fresh"
  mkdir -p "$log_root"
  log="$log_root/target-${iid}-${role}-job${job}.log"
  test ! -e "$log" || fail "launch log is not fresh: $log"
  srun --jobid="$job" --exclusive --exact --nodelist="$node" --nodes=1 --ntasks=1 \
    --gres=gpu:mi210:4 --cpus-per-task=16 --mem=0 \
    --export="ALL,STAGE1_EXPECTED_PARENT_JOB=$job" \
    "$method_root/scripts/auh_stage1_action_repr_gates_20260824_v2.sh" \
      worker-target-repr "$iid" "$role" "$job" >"$log" 2>&1
  test -f "$(target_repr_dir "$iid" "$role")/g0_receipt.json" \
    || fail "srun returned without a G0 receipt"
  echo "[stage1-v2] srun complete case=$iid role=$role job=$job log=$log"
}

materialize_target_controls() {
  local iid="$1" input input_sha output
  common_preflight
  require_canary_complete
  media_preflight "$iid"
  input="$(manifest_value "$iid" '.real_forward.path')"
  input_sha="$(manifest_value "$iid" '.real_forward.sha256')"
  output="$(target_control_dir "$iid")"
  test ! -e "$output" || fail "target video-control output must be fresh: $output"
  "$python_bin" -B "$video_controls" \
    --input-video "$input" \
    --input-sha256 "$input_sha" \
    --output "$output" \
    --case-id "$iid" \
    --anchor-kind target \
    --seed "$(manifest_value "$iid" '.seed')"
  test "$(jq -r '.authority.optimizer_created' "$output/receipt.json")" = false \
    || fail "video-control materializer unexpectedly owns an optimizer"
  echo "[stage1-v2] COMPLETE exact81 controls target case=$iid optimizer=false output=$output"
}

materialize_target_g1() {
  local iid="$1" wrong action_family wrong_family correct shuffle reverse donor output
  common_preflight
  require_canary_complete
  require_case "$iid"
  wrong="$(jq -er --arg iid "$iid" '.input_authority.wrong_action_donor_cycle[$iid]' "$prereg")"
  require_case "$wrong"
  action_family="$(manifest_value "$iid" '.action_family')"
  wrong_family="$(manifest_value "$wrong" '.action_family')"
  test "$action_family" != "$wrong_family" || fail "wrong-action donor has the same action family"
  correct="$(target_repr_dir "$iid" correct)"
  shuffle="$(target_repr_dir "$iid" temporal_shuffle)"
  reverse="$(target_repr_dir "$iid" reverse)"
  donor="$(target_repr_dir "$wrong" correct)"
  for output in "$correct" "$shuffle" "$reverse" "$donor"; do
    test -f "$output/g0_receipt.json" || fail "G0 representation is missing: $output"
  done
  output="$stage_root/g1_cohorts/target/$iid"
  test ! -e "$output" || fail "target G1 cohort output must be fresh: $output"
  mkdir -p "$output"
  "$python_bin" -B "$flow_controls" materialize \
    --correct "$correct/flow.safetensors" \
    --temporal-shuffle "$shuffle/flow.safetensors" \
    --reverse "$reverse/flow.safetensors" \
    --wrong-action "$donor/flow.safetensors" \
    --output-dir "$output/flow" \
    --case-id "$iid" \
    --anchor-kind target \
    --action-family "$action_family" \
    --wrong-case-id "$wrong" \
    --wrong-action-family "$wrong_family" \
    --incomplete-transitions 10 \
    >"$output/flow.log" 2>&1
  "$python_bin" -B "$middle_controls" materialize \
    --correct-cache "$correct/middle/middle_repr.safetensors" \
    --correct-receipt "$correct/middle/receipt.json" \
    --temporal-shuffle-cache "$shuffle/middle/middle_repr.safetensors" \
    --temporal-shuffle-receipt "$shuffle/middle/receipt.json" \
    --reverse-cache "$reverse/middle/middle_repr.safetensors" \
    --reverse-receipt "$reverse/middle/receipt.json" \
    --wrong-action-cache "$donor/middle/middle_repr.safetensors" \
    --wrong-action-receipt "$donor/middle/receipt.json" \
    --output-dir "$output/middle" \
    --case-id "$iid" \
    --anchor-kind target \
    --action-family "$action_family" \
    --wrong-case-id "$wrong" \
    --wrong-action-family "$wrong_family" \
    --incomplete-action-phases 10 \
    >"$output/middle.log" 2>&1
  "$python_bin" -B "$flow_controls" verify --receipt "$output/flow/cohort_receipt.json" >/dev/null
  "$python_bin" -B "$middle_controls" verify --receipt "$output/middle/cohort_receipt.json" >/dev/null
  echo "[stage1-v2] COMPLETE G1 controls target case=$iid wrong_case=$wrong optimizer=false output=$output"
}

evaluate_target_g1() {
  local iid="$1" cohort output
  common_preflight
  require_canary_complete
  require_case "$iid"
  cohort="$stage_root/g1_cohorts/target/$iid"
  test -f "$cohort/flow/cohort_receipt.json" || fail "target G1 flow cohort is missing"
  test -f "$cohort/middle/cohort_receipt.json" || fail "target G1 middle cohort is missing"
  output="$stage_root/g1_evaluations/target/$iid.json"
  test ! -e "$output" || fail "target G1 evaluation output must be fresh: $output"
  "$python_bin" -B "$g1_evaluator" evaluate \
    --target-flow-receipt "$cohort/flow/cohort_receipt.json" \
    --target-middle-receipt "$cohort/middle/cohort_receipt.json" \
    --subject-flow-receipt "$cohort/flow/cohort_receipt.json" \
    --subject-middle-receipt "$cohort/middle/cohort_receipt.json" \
    --output "$output"
  "$python_bin" -B "$g1_evaluator" verify --receipt "$output" >/dev/null
  echo "[stage1-v2] COMPLETE deterministic G1 evaluation target case=$iid optimizer=false output=$output"
}

publish_target_evidence_manifest() {
  local output="$1" jsonl temporary iid eval_path split family
  test ! -e "$output" || fail "target G1 evidence manifest must be fresh"
  mkdir -p "$(dirname "$output")"
  jsonl="$(mktemp "$(dirname "$output")/.target-evaluations.XXXXXX")"
  while IFS= read -r iid; do
    eval_path="$stage_root/g1_evaluations/target/$iid.json"
    test -f "$eval_path" || fail "target evaluation is missing: $iid"
    split="$(manifest_value "$iid" '.split')"
    family="$(manifest_value "$iid" '.action_family')"
    jq -cn \
      --arg case_id "$iid" \
      --arg split "$split" \
      --arg action_family "$family" \
      --arg anchor_kind target \
      --arg path "$eval_path" \
      --arg sha "$(sha256_file "$eval_path")" \
      '{case_id:$case_id,split:$split,action_family:$action_family,anchor_kind:$anchor_kind,evaluation_receipt_path:$path,evaluation_receipt_sha256:$sha}' \
      >>"$jsonl"
  done < <(jq -r '.cases[].case_id' "$manifest")
  temporary="$(mktemp "$(dirname "$output")/.target-evidence.XXXXXX")"
  jq -s \
    --slurpfile source_manifest "$manifest" \
    '{
      schema_version:"bernini-g1-joint-action-repr-evidence-manifest-v2",
      experiment_id:"action_repr_target_selfgen_middle_g1_20260824_v2",
      admission_scope:"target",
      expected_cases:($source_manifest[0].cases | map({case_id,split,action_family})),
      evaluations:.
    }' "$jsonl" >"$temporary"
  ln "$temporary" "$output"
  rm -f "$temporary" "$jsonl"
}

score_target_admission() {
  local output_root evidence receipt
  common_preflight
  require_canary_complete
  output_root="$stage_root/g1_admission/target"
  test ! -e "$output_root" || fail "target G1 admission output must be fresh"
  mkdir -p "$output_root"
  evidence="$output_root/evidence_manifest.json"
  receipt="$output_root/receipt.json"
  publish_target_evidence_manifest "$evidence"
  set +e
  "$python_bin" -B "$g1_scorer" score \
    --evidence-manifest "$evidence" \
    --output "$receipt" \
    --admission-scope target \
    >"$output_root/score.log" 2>&1
  local status=$?
  set -e
  [ "$status" -eq 0 ] || {
    echo "[stage1-v2] G1_TARGET_NO_GO scorer_status=$status receipt=${receipt}" >&2
    return "$status"
  }
  test "$(jq -r '.g1_target_status' "$receipt")" = passed || fail "target G1 receipt did not pass"
  test "$(jq -r '.g1_selfgen_status' "$receipt")" = not_evaluated || fail "target scope misrepresented selfgen"
  test "$(jq -r '.g1_selfgen_passed' "$receipt")" = null || fail "target scope misrepresented selfgen pass"
  test "$(jq -r '.optimizer_creation_authorized_by_this_receipt' "$receipt")" = false \
    || fail "G1 receipt unexpectedly authorizes optimizer creation"
  echo "[stage1-v2] G1_TARGET_PASS selfgen=not_evaluated optimizer=false receipt=$receipt"
}

target_g1_receipt() {
  printf '%s\n' "$stage_root/g1_admission/target/receipt.json"
}

require_target_g1_pass() {
  local receipt
  receipt="$(target_g1_receipt)"
  test -f "$receipt" || fail "G1_target admission has not closed"
  test "$(jq -r '.g1_target_status' "$receipt")" = passed || fail "G1_target did not pass"
  test "$(jq -r '.g1_selfgen_status' "$receipt")" = not_evaluated \
    || fail "target admission misrepresents self-generated status"
  test "$(jq -r '.optimizer_creation_authorized_by_this_receipt' "$receipt")" = false \
    || fail "G1_target receipt unexpectedly authorizes an optimizer"
}

run_g2a_cpu_api_audit() {
  local g1_receipt output log receipt temporary status
  common_preflight
  require_target_g1_pass
  g1_receipt="$(target_g1_receipt)"
  output="$stage_root/g2a/cpu_api_audit"
  test ! -e "$output" || fail "G2a CPU API audit output must be fresh"
  mkdir -p "$output"
  log="$output/unittest.log"
  set +e
  PYTHONPATH="$source_root:$method_root" "$python_bin" -B "$g2a_test" >"$log" 2>&1
  status=$?
  set -e
  [ "$status" -eq 0 ] || fail "G2a CPU API tests failed: $log"
  receipt="$output/receipt.json"
  temporary="$(mktemp "$output/.receipt.XXXXXX")"
  jq -n \
    --arg g1_target_receipt "$g1_receipt" \
    --arg g1_target_receipt_sha256 "$(sha256_file "$g1_receipt")" \
    --arg test_source_sha256 "$(sha256_file "$g2a_test")" \
    --arg module_source_sha256 "$(sha256_file "$method_root/action_repr_g2a_adapter_v1.py")" \
    --arg unittest_log_sha256 "$(sha256_file "$log")" \
    '{
      schema_version:"bernini-action-repr-g2a-cpu-api-audit-receipt-v1",
      complete:true,
      g1_target_receipt:$g1_target_receipt,
      g1_target_receipt_sha256:$g1_target_receipt_sha256,
      module_source_sha256:$module_source_sha256,
      test_source_sha256:$test_source_sha256,
      unittest_log_sha256:$unittest_log_sha256,
      cpu_api_tests_passed:true,
      required_routes:["correct","zero","temporal_shuffle","reverse","incomplete","wrong_action"],
      implementation_audit_only:true,
      production_WORLD4_integration_available:false,
      production_G2a_closed:false,
      optimizer_created:false,
      optimization_steps:0,
      optimizer_authorized:false,
      method_success_claimed:false
    }' >"$temporary"
  ln "$temporary" "$receipt"
  rm -f "$temporary"
  echo "[stage1-v2] G2A_CPU_API_PASS production_WORLD4=pending production_G2a_closed=false optimizer=false receipt=$receipt"
}

g2a_production_case_id() {
  printf '%s\n' 0be6494dfac3
}

g2a_production_receipt() {
  printf '%s\n' "$stage_root/g2a/production_world4/$(g2a_production_case_id)/receipt.json"
}

run_g2a_production_worker() {
  local expected_job="$1" iid source source_sha instruction receipt output_root scratch files current_file cgroup_max memory_proof_source log
  common_preflight
  require_canary_complete
  require_target_g1_pass
  validate_worker_allocation "$expected_job"
  iid="$(g2a_production_case_id)"
  require_case "$iid"
  source="$(manifest_value "$iid" '.source.path')"
  source_sha="$(manifest_value "$iid" '.source.sha256')"
  instruction="$(manifest_value "$iid" '.instruction')"
  require_sha "$source" "$source_sha"
  receipt="$(g2a_production_receipt)"
  output_root="$(dirname "$receipt")"
  test ! -e "$output_root" || fail "production G2a output must be fresh: $output_root"
  mkdir -p "$(dirname "$output_root")"
  mkdir "$output_root"
  log="$output_root/world4.log"
  scratch="/tmp/action-repr-stage1-v2-g2a-${iid}-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
  test ! -e "$scratch" || fail "production G2a scratch already exists: $scratch"
  mkdir -p "$scratch/xdg" "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
  export XDG_CACHE_HOME="$scratch/xdg"
  export MIOPEN_USER_DB_PATH="$scratch/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
  export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions"
  export TRITON_CACHE_DIR="$scratch/triton"
  files="$(memory_guard_preflight)"
  IFS=$'\t' read -r current_file cgroup_max memory_proof_source <<<"$files"
  run_middle_world4 "$log" "$current_file" "$cgroup_max" \
    "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
      "$g2a_world4_runner" \
      --bernini-root "$bernini_root" \
      --veomni-root "$veomni_root" \
      --checkpoint "$base_checkpoint" \
      --source-video "$source" \
      --g1-admission-receipt "$(target_g1_receipt)" \
      --case-id "$iid" \
      --instruction "$instruction" \
      --output "$receipt" \
      --sigma-index 1 \
      --seed "$(manifest_value "$iid" '.seed')" \
      --adapter-seed 2026082403
  test -f "$receipt" || fail "production WORLD4 G2a returned without a receipt"
  "$python_bin" -B -c \
    'import json,sys; from audit_action_repr_g2a_world4_v1 import validate_world4_receipt; validate_world4_receipt(json.load(open(sys.argv[1],"r",encoding="ascii")))' \
    "$receipt"
  test "$(jq -r '.passed' "$receipt")" = true || fail "production WORLD4 G2a did not pass"
  test "$(jq -r '.representation_routes.step0_required_routes | join(",")' "$receipt")" = correct,zero,temporal_shuffle,reverse,incomplete,wrong_action \
    || fail "production WORLD4 G2a omitted a required route"
  test "$(jq -r '.parameter_firewall.native_forward_count' "$receipt")" = 8 \
    || fail "production WORLD4 G2a native forward count differs"
  test "$(jq -r '.training_authority.optimizer_created' "$receipt")" = false \
    || fail "production WORLD4 G2a unexpectedly created an optimizer"
  test "$(jq -r '.training_authority.optimization_steps' "$receipt")" = 0 \
    || fail "production WORLD4 G2a step count differs"
  echo "[stage1-v2] G2A_PRODUCTION_PASS case=$iid WORLD4=true six_routes=true optimizer=false receipt=$receipt memory_proof=$memory_proof_source"
}

launch_g2a_production() {
  local job="$1" node iid receipt outer_log
  common_preflight
  require_canary_complete
  require_target_g1_pass
  validate_parent_job "$job"
  node="$(expected_node_for_job "$job")"
  iid="$(g2a_production_case_id)"
  receipt="$(g2a_production_receipt)"
  test ! -e "$(dirname "$receipt")" || fail "production G2a output is not fresh"
  mkdir -p "$log_root"
  outer_log="$log_root/g2a-production-${iid}-job${job}.log"
  test ! -e "$outer_log" || fail "production G2a launch log is not fresh: $outer_log"
  srun --jobid="$job" --exclusive --exact --nodelist="$node" --nodes=1 --ntasks=1 \
    --gres=gpu:mi210:4 --cpus-per-task=16 --mem=0 \
    --export="ALL,STAGE1_EXPECTED_PARENT_JOB=$job" \
    "$method_root/scripts/auh_stage1_action_repr_gates_20260824_v2.sh" \
      worker-g2a-production "$job" >"$outer_log" 2>&1
  test -f "$receipt" || fail "production G2a srun returned without a receipt"
  echo "[stage1-v2] production G2a srun complete case=$iid job=$job log=$outer_log receipt=$receipt"
}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TORCH_HOME=/vast/users/guangyi.chen/.cache/torch
export PYTHONPATH="$source_root:$method_root"

[ "$#" -ge 1 ] || usage
command="$1"
shift
case "$command" in
  preflight)
    [ "$#" -eq 0 ] || usage
    common_preflight
    echo "[stage1-v2.7] PREFLIGHT_PASS cases=8 rank0_posterior=true explicit_prepack_gaussian=true deterministic_vae=true quantized_energy=true control_matched_gaussian=true six_route_G2a_code=true optimizer=false production_G2a=pending_until_G1_target"
    ;;
  launch-target-canary)
    [ "$#" -le 1 ] || usage
    launch_target_repr 0be6494dfac3 correct "${1:-147881}"
    ;;
  target-controls)
    [ "$#" -eq 1 ] || usage
    materialize_target_controls "$1"
    ;;
  launch-target-repr)
    [ "$#" -ge 2 ] && [ "$#" -le 3 ] || usage
    launch_target_repr "$1" "$2" "${3:-147871}"
    ;;
  worker-target-repr)
    [ "$#" -eq 3 ] || usage
    run_target_repr_worker "$1" "$2" "$3"
    ;;
  target-g1)
    [ "$#" -eq 1 ] || usage
    materialize_target_g1 "$1"
    ;;
  target-eval)
    [ "$#" -eq 1 ] || usage
    evaluate_target_g1 "$1"
    ;;
  target-admission)
    [ "$#" -eq 0 ] || usage
    score_target_admission
    ;;
  g2a-cpu-api-audit)
    [ "$#" -eq 0 ] || usage
    run_g2a_cpu_api_audit
    ;;
  launch-g2a-production)
    [ "$#" -le 1 ] || usage
    launch_g2a_production "${1:-147881}"
    ;;
  worker-g2a-production)
    [ "$#" -eq 1 ] || usage
    run_g2a_production_worker "$1"
    ;;
  g2a-production-status)
    [ "$#" -eq 0 ] || usage
    receipt="$(g2a_production_receipt)"
    if [ ! -f "$receipt" ]; then
      echo "[stage1-v2] PENDING: production WORLD4 G2a requires passed G1_target and a fresh six-route receipt; optimizer remains forbidden" >&2
      exit 64
    fi
    "$python_bin" -B -c \
      'import json,sys; from audit_action_repr_g2a_world4_v1 import validate_world4_receipt; validate_world4_receipt(json.load(open(sys.argv[1],"r",encoding="ascii")))' \
      "$receipt"
    echo "[stage1-v2] G2A_PRODUCTION_PASS optimizer=false receipt=$receipt"
    ;;
  *) usage ;;
esac
