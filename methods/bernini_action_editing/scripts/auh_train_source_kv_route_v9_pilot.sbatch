#!/usr/bin/env bash
#SBATCH --job-name=bern-csvart-v9
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=08:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 077

source_archive="${BERNINI_V9_SOURCE_ARCHIVE:?set BERNINI_V9_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_V9_SOURCE_ARCHIVE_SHA256:?set BERNINI_V9_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_V9_SOURCE_REVISION:?set BERNINI_V9_SOURCE_REVISION}"
source_repository="${BERNINI_V9_SOURCE_REPOSITORY:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
output_dir="${BERNINI_V9_TRAIN_OUTPUT:?set BERNINI_V9_TRAIN_OUTPUT}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

# Formal full644 / strict359 inputs are literals, not caller-controlled aliases.
data_dir="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/data/vae_full_81f_4d41e4c/shards"
dataset_summary="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/data/vae_full_81f_4d41e4c/dataset_summary.json"
dataset_index="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/data/vae_full_81f_4d41e4c/dataset_index.jsonl"
routing_jsonl="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_c2fr_20260806/routing/strict359.jsonl"
dataset_summary_sha256="5dc45b4a6d700b3cd0108e941242ae364396458f20f41249744e74e00acc02dd"
dataset_index_sha256="d36fb5de3487ba5bf494589948430a60e214851d29776cc4f439e4e2d54ee52b"
routing_sha256="0da09787889687726d9161b0c74b8df5d58226f6e431632b317891d630ef49eb"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
checkpoint_content_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
checkpoint_content_file_count=23

fail() { echo "[bernini-csvart-v9] ERROR: $*" >&2; exit 2; }

validate_checkpoint_content() {
  local manifest="$1"
  local actual_count
  [[ -f "${manifest}" && ! -L "${manifest}" ]] || fail "checkpoint manifest is not a plain file"
  [[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${checkpoint_content_manifest_sha256}" ]] || fail "checkpoint manifest digest mismatch"
  [[ "$(wc -l < "${manifest}")" -eq "${checkpoint_content_file_count}" ]] || fail "checkpoint manifest file count differs"
  [[ -z "$(find "${checkpoint}" -path "${checkpoint}/.cache" -prune -o -type l -print -quit)" ]] || fail "checkpoint contains a non-cache symlink"
  actual_count="$(find "${checkpoint}" -path "${checkpoint}/.cache" -prune -o -type f -print | wc -l)"
  [[ "${actual_count}" -eq "${checkpoint_content_file_count}" ]] || fail "checkpoint non-cache file set differs"
  (cd "${checkpoint}" && sha256sum --strict --status -c "${manifest}") || fail "checkpoint content hash differs"
}

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid source archive"
[[ -d "${source_repository}/.git" ]] || fail "invalid source repository"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini root"
[[ -d "${veomni_root}/.git" ]] || fail "invalid VeOmni root"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid checkpoint"
[[ -d "${data_dir}" && ! -L "${data_dir}" ]] || fail "invalid pinned parquet directory"
[[ -f "${dataset_summary}" && ! -L "${dataset_summary}" ]] || fail "invalid pinned dataset summary"
[[ -f "${dataset_index}" && ! -L "${dataset_index}" ]] || fail "invalid pinned dataset index"
[[ -f "${routing_jsonl}" && ! -L "${routing_jsonl}" ]] || fail "invalid pinned routing JSONL"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid archive SHA-256"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ "${output_dir}" == /* ]] || fail "output must be absolute"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing to reuse output"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash mismatch"
[[ "$(sha256sum "${dataset_summary}" | awk '{print $1}')" == "${dataset_summary_sha256}" ]] || fail "dataset summary hash mismatch"
[[ "$(sha256sum "${dataset_index}" | awk '{print $1}')" == "${dataset_index_sha256}" ]] || fail "dataset index hash mismatch"
[[ "$(sha256sum "${routing_jsonl}" | awk '{print $1}')" == "${routing_sha256}" ]] || fail "routing hash mismatch"
[[ "$(wc -l < "${dataset_index}")" -eq 644 ]] || fail "dataset index row count differs from full644"
[[ "$(wc -l < "${routing_jsonl}")" -eq 644 ]] || fail "routing row count differs from full644"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_commit}" ]] || fail "VeOmni commit mismatch"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source is dirty"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-csvart-v9-${SLURM_JOB_ID}.XXXXXX")"
archive_copy="${task_scratch}/method.tar"
archive_from_revision="${task_scratch}/method-from-revision.tar"
source_root="${task_scratch}/source"
mkdir -p \
  "${source_root}" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton" \
  "${output_dir%/*}"
cp -- "${source_archive}" "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "private archive hash mismatch"
[[ "$(git -C "${source_repository}" cat-file -t "${source_revision}^{commit}")" == "commit" ]] || fail "source revision is not a repository commit"
git -C "${source_repository}" archive --format=tar \
  -o "${archive_from_revision}" \
  "${source_revision}" methods/bernini_action_editing
[[ "$(sha256sum "${archive_from_revision}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "archive bytes are not the declared git revision"
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${source_root}"
[[ -z "$(find "${source_root}" -type l -print -quit)" ]] || fail "source archive contains a symlink"

method_root="${source_root}/methods/bernini_action_editing"
for required in \
  train_source_kv_route_auh.py \
  source_kv_replay.py \
  source_kv_route_batches.py \
  source_kv_route_objective.py \
  source_kv_route_scope.py \
  train_cross_mode_cmsg_auh.py \
  train_cross_mode_cmsg_lora.py \
  cross_mode_branches.py \
  cross_mode_motion_spectrum.py \
  train_prior_tangent_lora.py \
  prior_guided_tangent.py \
  tri_branch_unipc.py \
  train_delta_lora.py \
  train_lora.py \
  motion_residual.py \
  inference_sigma_strata.py \
  tests/test_source_kv_replay.py \
  tests/test_source_kv_route_batches.py \
  tests/test_source_kv_route_objective.py \
  tests/test_source_kv_route_scope.py \
  tests/test_train_source_kv_route_auh_contract.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive lacks plain ${required}"
done
checkpoint_manifest="${method_root}/audits/bernini_r13_ff4c5d4_checkpoint.sha256"
validate_checkpoint_content "${checkpoint_manifest}"

export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-csvart-v9] source_revision=${source_revision} archive_sha256=${source_archive_sha256}"
echo "[bernini-csvart-v9] full644_index_sha256=${dataset_index_sha256} strict359_route_sha256=${routing_sha256}"
echo "[bernini-csvart-v9] frames=81 phases=21 forwards=6 graph_forwards=2 carrier=all30 scope=exact92 tensors=184 rank=8 steps=40"
echo "[bernini-csvart-v9] query_state=offline_source_tangent_beta0 exposure_gap=declared production_claim=forbidden"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

"${python_bin}" -m py_compile \
  "${method_root}/source_kv_replay.py" \
  "${method_root}/source_kv_route_batches.py" \
  "${method_root}/source_kv_route_objective.py" \
  "${method_root}/source_kv_route_scope.py" \
  "${method_root}/train_source_kv_route_auh.py"
for pattern in \
  'test_source_kv_replay.py' \
  'test_source_kv_route_batches.py' \
  'test_source_kv_route_objective.py' \
  'test_source_kv_route_scope.py' \
  'test_train_source_kv_route_auh_contract.py'
do
  PYTHONPATH="${method_root}" "${python_bin}" -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${method_root}/train_source_kv_route_auh.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --preprocessed-parquet-dir "${data_dir}" \
  --dataset-summary "${dataset_summary}" \
  --routing-jsonl "${routing_jsonl}" \
  --expected-routing-jsonl-sha256 "${routing_sha256}" \
  --output "${output_dir}" \
  --num-frames 81 \
  --max-steps 40 \
  --save-every 40 \
  --learning-rate 1e-5 \
  --weight-decay 0 \
  --max-grad-norm 1 \
  --block-selection all \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}"

pilot_checkpoint="${output_dir}/checkpoint-00000040"
receipt="${pilot_checkpoint}/receipt.json"
[[ -f "${receipt}" && ! -L "${receipt}" ]] || fail "pilot receipt missing"
[[ -f "${pilot_checkpoint}/adapter/adapter_model.safetensors" && ! -L "${pilot_checkpoint}/adapter/adapter_model.safetensors" ]] || fail "pilot adapter missing"
validate_checkpoint_content "${checkpoint_manifest}"
jq -e '
  .schema_version == "bernini-source-kv-route-auh-training-receipt-v9" and
  .method == "bernini-frozen-source-kv-csv-art-v9-auh" and
  .global_step == 40 and
  .formal_exact40_complete == true and
  .exact40_audit.validated == true and
  .exact40_audit.all30_main == true and
  .exact40_audit.selected_block_count == 30 and
  .adapter.target_module_count == 92 and
  .adapter.adapter_tensor_count == 184 and
  .adapter.trainable_parameter_count == 2260992 and
  .optimizer.state_parameter_count == 184 and
  .optimizer.state_step_values == [40] and
  .optimizer.no_moment_reset == true and
  .paired_target_model_forward_access == false and
  .external_mask_track_flow_pose_trajectory == false and
  .first_frame_anchor == false and
  .query_state_policy.name == "offline_source_tangent_beta0" and
  .query_state_policy.query_state_train_test_matched == false and
  .query_state_policy.paired_target_tail_used == false and
  .production_claim_forbidden == true and
  .scientific_claim_authorized == false and
  .artifact_validation.verified == true and
  .artifact_validation.status == "post_save_adapter_tensor_file_roundtrip_and_optimizer_load_state_dict_complete" and
  .artifact_validation.adapter_tensor_file_roundtrip_verified == true and
  .artifact_validation.adapter_tensor_file_runtime_equality == true and
  .artifact_validation.torch_deserialize_verified == true and
  .artifact_validation.fresh_optimizer_load_state_dict_verified == true and
  .artifact_validation.optimizer_state_logical_equality_verified == true and
  .artifact_validation.state_parameter_count == 184 and
  .artifact_validation.state_step_values == [40] and
  .artifact_validation.runtime_adapter_loader_verified == false and
  .artifact_validation.fresh_base_peft_from_pretrained_verified == false and
  .artifact_validation.deployment_loader_claim_forbidden == true and
  .dataset.input_integrity.validated == true and
  .dataset.input_integrity.access_count == 40 and
  .dataset.input_integrity.dataset_summary_final_sha256 == "5dc45b4a6d700b3cd0108e941242ae364396458f20f41249744e74e00acc02dd" and
  .dataset.input_integrity.dataset_index_final_sha256 == "d36fb5de3487ba5bf494589948430a60e214851d29776cc4f439e4e2d54ee52b" and
  .dataset.input_integrity.routing_final_sha256 == "0da09787889687726d9161b0c74b8df5d58226f6e431632b317891d630ef49eb" and
  (all(.step_audit[];
    .forwards_per_candidate == 6 and
    .graph_forwards_per_candidate == 2 and
    .gradient_audit.trainable_tensor_count == 184 and
    .gradient_audit.all_gradients_present == true and
    .gradient_audit.all_gradients_finite == true and
    .gradient_audit.positive_global_l2_norm == true and
    .gradient_audit.global_l2_norm > 0 and
    .cache_after_backward_audit.capture_calls_delta == 30 and
    .cache_after_backward_audit.replay_lookups_delta == 210 and
    .cache_after_backward_audit.replay_phase_count_delta.eager == 90 and
    .cache_after_backward_audit.replay_phase_count_delta.checkpoint_forward == 60 and
    .cache_after_backward_audit.replay_phase_count_delta.checkpoint_recompute == 60 and
    .cache_after_backward_audit.checkpoint_context_count_delta.checkpoint_forward == 60 and
    .cache_after_backward_audit.checkpoint_context_count_delta.checkpoint_recompute == 60 and
    .cache_after_clear_audit.cleared_after_backward == true and
    .cache_after_clear_audit.identity_after_clear == null and
    .input_shard_integrity.hash_closed_read == true and
    .input_shard_integrity.cache_invalidated_before_read == true and
    .input_shard_integrity.before_read_sha256 == .input_shard_integrity.expected_sha256 and
    .input_shard_integrity.after_read_sha256 == .input_shard_integrity.expected_sha256 and
    .target_energy_retention == 1 and
    .target_clipped_fraction == 0
  ))
' "${receipt}" >/dev/null

adapter_model="${pilot_checkpoint}/adapter/adapter_model.safetensors"
adapter_config="${pilot_checkpoint}/adapter/adapter_config.json"
optimizer_checkpoint="${pilot_checkpoint}/optimizer.pt"
[[ "$(sha256sum "${adapter_model}" | awk '{print $1}')" == "$(jq -r '.artifact_validation.adapter_model_sha256' "${receipt}")" ]] || fail "adapter model SHA differs from receipt"
[[ "$(sha256sum "${adapter_config}" | awk '{print $1}')" == "$(jq -r '.artifact_validation.adapter_config_sha256' "${receipt}")" ]] || fail "adapter config SHA differs from receipt"
[[ "$(sha256sum "${optimizer_checkpoint}" | awk '{print $1}')" == "$(jq -r '.artifact_validation.optimizer_checkpoint_sha256' "${receipt}")" ]] || fail "optimizer SHA differs from receipt"
PYTHONPATH="${method_root}" "${python_bin}" -c 'import json,sys; import source_kv_route_scope as scope; value=json.load(open(sys.argv[1], encoding="utf-8")); declared=value.pop("receipt_digest", None); assert declared == scope.object_sha256(value), "canonical receipt digest mismatch"' "${receipt}"
[[ "$(jq -r '.receipt_digest' "${receipt}")" == "$(jq -r '.receipt_digest' "${output_dir}/latest.json")" ]] || fail "latest.json receipt digest differs"
[[ "$(sha256sum "${dataset_summary}" | awk '{print $1}')" == "${dataset_summary_sha256}" ]] || fail "dataset summary changed after training"
[[ "$(sha256sum "${dataset_index}" | awk '{print $1}')" == "${dataset_index_sha256}" ]] || fail "dataset index changed after training"
[[ "$(sha256sum "${routing_jsonl}" | awk '{print $1}')" == "${routing_sha256}" ]] || fail "routing changed after training"

rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-csvart-v9] PASS engineering_exact40=true deployment_efficacy_unproven=true"
