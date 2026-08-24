#!/usr/bin/env bash
set -Eeuo pipefail

fail() {
  echo "[v16r6-ab-debug32-worker] ERROR: $*" >&2
  exit 3
}

readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
readonly veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
readonly base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly full644_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_self_generated_anchor_v1_20260818/data/full644_action_anchor_manifest_v1.json
readonly heldout_sha=c05c4e5b5bf85de882bde32c71a984d736247733e586ed91d40026b12aaaf701
readonly full644_sha=61da995eb680b9fba7ab3b7d3b6041c7b51c7e95253c74e607ddab6fdd6a61aa

[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_STEP_ID:-}" ]] || fail "must run inside a Slurm step"
[[ "${SLURM_JOB_ID}" =~ ^[0-9]+$ && "${SLURM_STEP_ID}" =~ ^[0-9]+$ ]] || fail "Slurm identity syntax differs"
[[ -n "${V16R6_EXPECTED_JOB:-}" && "${SLURM_JOB_ID}" == "${V16R6_EXPECTED_JOB}" ]] || fail "allocation differs"
[[ -n "${V16R6_EXPECTED_NODE:-}" && "$(hostname -s)" == "${V16R6_EXPECTED_NODE}" ]] || fail "node differs"
[[ "${V16R6_VARIANT:-}" == "a" || "${V16R6_VARIANT:-}" == "b" || "${V16R6_VARIANT:-}" == "c" || "${V16R6_VARIANT:-}" == "d" ]] || fail "variant must be a, b, c, or d"
[[ -n "${V16R6_RELEASE_ROOT:-}" && -d "${V16R6_RELEASE_ROOT}" && ! -L "${V16R6_RELEASE_ROOT}" ]] || fail "release root differs"
[[ -n "${V16R6_PLAN:-}" && -f "${V16R6_PLAN}" && ! -L "${V16R6_PLAN}" ]] || fail "run plan differs"
[[ -n "${V16R6_PLAN_SHA256:-}" && "${V16R6_PLAN_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "run plan SHA is absent"
[[ "${ROCR_VISIBLE_DEVICES:-}" == "0,1,2,3" ]] || fail "WORLD4 GPU visibility differs"

readonly release="${V16R6_RELEASE_ROOT}"
readonly release_manifest="${release}/v16r6ab-release.json"
readonly source_archive="${release}/v16r6ab-source.tar"
readonly source_manifest="${release}/v16r6ab-source.manifest.json"
readonly plan="${V16R6_PLAN}"

for path in "${release_manifest}" "${source_archive}" "${source_manifest}" "${plan}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "release input is absent: ${path}"
done
builder_leaf="$(jq -er '.source_release.builder' "${release_manifest}")"
builder="${release}/${builder_leaf}"
[[ -f "${builder}" && ! -L "${builder}" ]] || fail "release verifier is absent"
[[ -x "${python_bin}" ]] || fail "vace Python is absent"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini source tree is absent"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni source tree is absent"
[[ -d "${base_checkpoint}" && ! -L "${base_checkpoint}" ]] || fail "base checkpoint is absent"
[[ -f "${full644_manifest}" && ! -L "${full644_manifest}" ]] || fail "full644 manifest is absent"
[[ "$(sha256sum -- "${full644_manifest}" | awk '{print $1}')" == "${full644_sha}" ]] || fail "full644 manifest SHA differs"
[[ "$(sha256sum -- "${plan}" | awk '{print $1}')" == "${V16R6_PLAN_SHA256}" ]] || fail "run plan SHA differs"

archive_sha="$(jq -er '.source_release.archive_sha256' "${release_manifest}")"
manifest_sha="$(jq -er '.source_release.manifest_sha256' "${release_manifest}")"
method_revision="$(jq -er '.source_release.content_closure_sha256' "${release_manifest}")"
trainer_member="$(jq -er --arg variant "${V16R6_VARIANT}" '.variants[$variant].trainer_member' "${release_manifest}")"
method="$(jq -er --arg variant "${V16R6_VARIANT}" '.variants[$variant].method' "${release_manifest}")"
learning_rate="$(jq -er --arg variant "${V16R6_VARIANT}" '.variants[$variant].learning_rate' "${release_manifest}")"
target_count="$(jq -er --arg variant "${V16R6_VARIANT}" '.variants[$variant].target_module_count' "${release_manifest}")"
target_sha="$(jq -er --arg variant "${V16R6_VARIANT}" '.variants[$variant].target_modules_sha256' "${release_manifest}")"
output="$(jq -er '.output' "${plan}")"

jq -e \
  --arg variant "${V16R6_VARIANT}" \
  --argjson job "${SLURM_JOB_ID}" \
  --arg node "$(hostname -s)" \
  --arg output "${output}" \
  --arg archive_sha "${archive_sha}" \
  --arg manifest_sha "${manifest_sha}" \
  --arg revision "${method_revision}" \
  '.schema_version == "bernini-v16r6-ab-debug32-run-plan-v1" and
   .variant == $variant and .job_id == $job and .node == $node and
   .output == $output and .max_steps == 32 and
   .exact644_training_complete == false and
   .terminal_full644_checkpoint == false and
   .scientific_claim_authorized == false and
   .source_archive_sha256 == $archive_sha and
   .source_manifest_sha256 == $manifest_sha and
   .method_source_revision == $revision' \
  "${plan}" >/dev/null || fail "run plan semantic contract differs"

[[ "$(sha256sum -- "${source_archive}" | awk '{print $1}')" == "${archive_sha}" ]] || fail "source archive SHA differs"
[[ "$(sha256sum -- "${source_manifest}" | awk '{print $1}')" == "${manifest_sha}" ]] || fail "source manifest SHA differs"
[[ ! -e "${output}" && ! -L "${output}" ]] || fail "training output is not fresh"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ -d "${scratch_parent}" && ! -L "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="${scratch_parent}/v16r6${V16R6_VARIANT}-debug32-${SLURM_JOB_ID}-${SLURM_STEP_ID}-$$"
[[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || fail "task scratch is not fresh"
mkdir -m 0700 "${task_scratch}"
cleanup() {
  local status="$?"
  set +e
  trap - EXIT
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" && "${task_scratch}" == "${scratch_parent}/v16r6${V16R6_VARIANT}-debug32-${SLURM_JOB_ID}-${SLURM_STEP_ID}-"* ]]; then
    rm -rf -- "${task_scratch}"
  fi
  exit "${status}"
}
trap cleanup EXIT

"${python_bin}" -B "${builder}" verify \
  --archive "${source_archive}" \
  --manifest "${source_manifest}" \
  --expected-archive-sha256 "${archive_sha}" \
  --expected-manifest-sha256 "${manifest_sha}"
mkdir -m 0700 "${task_scratch}/source" "${task_scratch}/rank-caches"
tar -xf "${source_archive}" -C "${task_scratch}/source"
source_tree="${task_scratch}/source"
method_root="${source_tree}/methods/bernini_action_editing"
trainer="${source_tree}/${trainer_member}"
rank_wrapper="${method_root}/scripts/auh_v16r4_rank_wrapper.sh"
heldout_manifest="${source_tree}/methods/action_editing_baselines/manifests/goku_legacy_heldout8_inputs.jsonl"
for path in "${trainer}" "${rank_wrapper}" "${heldout_manifest}" "${method_root}/source_kv_replay.py"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "required source member is absent: ${path}"
done
[[ "$(sha256sum -- "${heldout_manifest}" | awk '{print $1}')" == "${heldout_sha}" ]] || fail "Heldout8 manifest SHA differs"
trainer_sha="$(jq -er --arg member "${trainer_member}" '.files[] | select(.path == $member) | .sha256' "${source_manifest}")"
[[ "$(sha256sum -- "${trainer}" | awk '{print $1}')" == "${trainer_sha}" ]] || fail "trainer SHA differs"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export ROCR_VISIBLE_DEVICES=0,1,2,3
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export PYTHONPATH="${method_root}"
export V16R4_RANK_CACHE_ROOT="${task_scratch}/rank-caches"
export V16R5_RANK_CACHE_ROOT="${task_scratch}/rank-caches"

"${python_bin}" -B -m torch.distributed.run \
  --standalone --nproc_per_node=4 --no-python \
  bash "${rank_wrapper}" "${python_bin}" -B "${trainer}" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${base_checkpoint}" \
  --pair-manifest "${full644_manifest}" \
  --authoring "${full644_manifest}" \
  --output "${output}" \
  --profile dynamic_static \
  --route-operator self_target_owned_activity_kernel25_v14r2 \
  --max-steps 32 \
  --micro-records 2 \
  --source-variant not_applicable \
  --route-strength 0.25 \
  --teacher-route-strength 0.50 \
  --training-objective real_source_target_owned_routed_teacher_delta_v14r2 \
  --training-interface first_phase_caption_i2v \
  --paired-target-fm-weight 0 \
  --real-source-manifest "${full644_manifest}" \
  --real-source-manifest-sha256 "${full644_sha}" \
  --full644-manifest-sha256 "${full644_sha}" \
  --teacher-delta-mode raw \
  --routed-teacher-mode same_action_route_only \
  --source-reconstruction-weight 0.025 \
  --replay-combine-mode source_halfspace_001 \
  --source-reconstruction-prompt action \
  --learning-rate "${learning_rate}" \
  --seed 2026082302 \
  --max-grad-norm 10 \
  --decoded-canary-manifest "${heldout_manifest}" \
  --decoded-canary-manifest-sha256 "${heldout_sha}" \
  --method-source-revision "${method_revision}" \
  --method-source-archive-sha256 "${archive_sha}"

completion="${output}/TRAINING_COMPLETE"
receipt="${output}/checkpoint-00000032/receipt.json"
adapter_config="${output}/checkpoint-00000032/adapter/adapter_config.json"
for path in "${completion}" "${receipt}" "${adapter_config}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "debug terminal artifact is absent: ${path}"
done
[[ "$(sha256sum -- "${completion}" | awk '{print $1}')" == "37a40f08d8548dba289b9b0bb35bcf63b359f6d37ee86044ebc6b6da080b9ec1" ]] || fail "process-completion marker differs"
jq -e \
  --arg method "${method}" \
  --arg target_sha "${target_sha}" \
  --argjson target_count "${target_count}" \
  '.global_step == 32 and .max_steps == 32 and .complete == false and
   .exact644_training_complete == false and
   .terminal_full644_checkpoint == false and
   .scientific_claim_authorized == false and
   .training_contract.method == $method and
   .training_contract.lora_target_module_count == $target_count and
   .training_contract.lora_target_modules_sha256 == $target_sha and
   .training_contract.all_full644_rows_targeted_exactly_once == false and
   .v16r6_debug_contract.debug_optimizer_step_budget == 32 and
   .v16r6_debug_contract.debug_run_complete == true and
   .v16r6_debug_contract.exact644_training_complete == false' \
  "${receipt}" >/dev/null || fail "debug receipt closure differs"

if [[ "${V16R6_VARIANT}" == "b" ]]; then
  jq -e \
    '.v16r6b_lora_scope_contract.target_module_count == 44 and
     .v16r6b_lora_scope_contract.trainable_tensor_count == 88 and
     .v16r6b_lora_scope_contract.trainable_parameter_count == 34603008 and
     .training_contract.lora_nonroute_blocks_trainable == false and
     .training_contract.lora_attn2_trainable == false and
     .training_contract.lora_value_or_output_trainable == false' \
    "${receipt}" >/dev/null || fail "variant-B scope receipt differs"
elif [[ "${V16R6_VARIANT}" == "a" ]]; then
  jq -e \
    '.v16r6a_learning_rate_contract.active_coordinate_rms_learning_rate == 1e-7 and
     .v16r6a_learning_rate_contract.lora_target_module_count == 240 and
     .v16r6a_learning_rate_contract.trainable_parameter_count == 188743680 and
     .v16r6a_scale_equivalent_gradient_audit.v16r5_absolute_epsilon == 1e-12 and
     .v16r6a_scale_equivalent_gradient_audit.learning_rate_ratio_to_v16r5 == 0.1 and
     .v16r6a_scale_equivalent_gradient_audit.effective_absolute_epsilon == 1e-13 and
     .v16r6a_scale_equivalent_gradient_audit.required_action_tensor_count_step2_plus == 480 and
     .v16r6a_scale_equivalent_gradient_audit.required_raw_replay_tensor_count_step2_plus == 480 and
     .v16r6a_scale_equivalent_gradient_audit.training_gradient_loss_optimizer_or_data_changed_by_audit == false and
     .training_contract.lora_scope_changed_from_v16r5 == false' \
    "${receipt}" >/dev/null || fail "variant-A learning-rate receipt differs"
elif [[ "${V16R6_VARIANT}" == "c" ]]; then
  jq -e \
    '.v16r6c_two_sided_delta_contract.gradient_mode == "two_sided_sequential_j_on_minus_j_off_v16r6c" and
     .v16r6c_two_sided_delta_contract.student_delta_jacobian == "J_route_on_minus_J_route_off" and
     .v16r6c_two_sided_delta_contract.route_off_forward_has_grad == true and
     .v16r6c_two_sided_delta_contract.sequential_backward == true and
     .v16r6c_two_sided_delta_contract.simultaneous_two_30_block_graph_retention == false and
     .training_contract.same_action_route_off_gradient_enabled == true and
     .training_contract.learning_rate_changed_from_v16r5 == false and
     .training_contract.lora_scope_changed_from_v16r5 == false' \
    "${receipt}" >/dev/null || fail "variant-C two-sided gradient receipt differs"
else
  jq -e \
    '.v16r6d_absolute_route_off_anchor_contract.mode == "same_state_route_off_frozen_base_fm_weight0025_v16r6d" and
     .v16r6d_absolute_route_off_anchor_contract.weight == 0.025 and
     .v16r6d_absolute_route_off_anchor_contract.student_delta_jacobian == "J_route_on_only_legacy" and
     .v16r6d_absolute_route_off_anchor_contract.teacher_detached == true and
     .v16r6d_absolute_route_off_anchor_contract.student_route_off_forward_has_grad == true and
     .v16r6d_absolute_route_off_anchor_contract.sequential_backward == true and
     .v16r6d_absolute_route_off_anchor_contract.simultaneous_two_30_block_graph_retention == false and
     .v16r6d_absolute_route_off_anchor_contract.decoded_source_preservation_claimed == false and
     .route_off_absolute_anchor_diagnostic.applicable == true and
     .route_off_absolute_anchor_diagnostic.micro_count == 2 and
     .route_off_absolute_anchor_diagnostic.weight == 0.025 and
     .training_contract.same_action_route_off_gradient_enabled == false and
     .training_contract.same_action_student_delta_gradient_mode == "route_on_only_legacy" and
     .training_contract.same_action_route_off_absolute_anchor_enabled == true and
     .training_contract.learning_rate_changed_from_v16r5 == false and
     .training_contract.lora_scope_changed_from_v16r5 == false' \
    "${receipt}" >/dev/null || fail "variant-D absolute anchor receipt differs"
fi

echo "[v16r6-ab-debug32-worker] variant ${V16R6_VARIANT} exact32 debug completed; this is not exact644"
