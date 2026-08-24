#!/usr/bin/env bash
#SBATCH --job-name=bern-rsfqt-v8-tr
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=04:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 077

source_archive="${BERNINI_V8_SOURCE_ARCHIVE:?set BERNINI_V8_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_V8_SOURCE_ARCHIVE_SHA256:?set BERNINI_V8_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_V8_SOURCE_REVISION:?set BERNINI_V8_SOURCE_REVISION}"
source_repository="${BERNINI_V8_SOURCE_REPOSITORY:?set BERNINI_V8_SOURCE_REPOSITORY}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
data_dir="${BERNINI_ACTION_PARQUET_DIR:?set BERNINI_ACTION_PARQUET_DIR}"
dataset_summary="${BERNINI_ACTION_DATASET_SUMMARY:?set BERNINI_ACTION_DATASET_SUMMARY}"
routing_jsonl="${BERNINI_V8_ROUTING_JSONL:?set BERNINI_V8_ROUTING_JSONL}"
routing_sha256="${BERNINI_V8_ROUTING_JSONL_SHA256:?set BERNINI_V8_ROUTING_JSONL_SHA256}"
output_dir="${BERNINI_V8_TRAIN_OUTPUT:?set BERNINI_V8_TRAIN_OUTPUT}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
checkpoint_content_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
checkpoint_content_file_count=23

fail() { echo "[bernini-rsfqt-v8-train] ERROR: $*" >&2; exit 2; }

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
[[ -d "${data_dir}" && ! -L "${data_dir}" ]] || fail "invalid parquet directory"
[[ -f "${dataset_summary}" && ! -L "${dataset_summary}" ]] || fail "invalid dataset summary"
[[ -f "${routing_jsonl}" && ! -L "${routing_jsonl}" ]] || fail "invalid routing JSONL"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid archive SHA-256"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ "${routing_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid routing SHA-256"
[[ "${output_dir}" == /* ]] || fail "output must be absolute"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing to reuse output"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash mismatch"
[[ "$(sha256sum "${routing_jsonl}" | awk '{print $1}')" == "${routing_sha256}" ]] || fail "routing hash mismatch"
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

task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-rsfqt-v8-train-${SLURM_JOB_ID}.XXXXXX")"
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
  feasible_quotient_objective.py \
  gauge_anchored_commutator.py \
  motion_commutator.py \
  train_feasible_quotient_auh.py \
  train_relational_motion_commutator_auh.py \
  train_cross_mode_cmsg_lora.py \
  train_cross_mode_cmsg_auh.py \
  train_prior_tangent_lora.py \
  train_delta_lora.py \
  train_lora.py \
  motion_residual.py \
  inference_sigma_strata.py \
  infer_feasible_quotient_lora.py \
  infer_relational_motion_commutator.py \
  finalize_feasible_quotient_checkpoint.py \
  run_relational_motion_commutator_inference.py \
  tests/test_feasible_quotient_objective.py \
  tests/test_train_feasible_quotient_auh.py \
  tests/test_infer_feasible_quotient_lora.py \
  tests/test_finalize_feasible_quotient_checkpoint.py \
  tests/test_feasible_quotient_sbatch_contract.py \
  tests/test_gauge_anchored_commutator.py \
  tests/test_train_prior_tangent_lora_contract.py \
  tests/test_infer_relational_motion_commutator.py \
  tests/test_run_relational_motion_commutator_inference_contract.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive lacks plain ${required}"
done
checkpoint_manifest="${method_root}/audits/bernini_r13_ff4c5d4_checkpoint.sha256"
validate_checkpoint_content "${checkpoint_manifest}"
echo "[bernini-rsfqt-v8-train] checkpoint_content_manifest_sha256=${checkpoint_content_manifest_sha256} files=${checkpoint_content_file_count}"

export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-rsfqt-v8-train] source_revision=${source_revision} archive_sha256=${source_archive_sha256}"
echo "[bernini-rsfqt-v8-train] routing_sha256=${routing_sha256} output=${output_dir}"
echo "[bernini-rsfqt-v8-train] frames=81 phases=21 forwards=5 graph_forwards=2 generator_forwards=0 scope=exact46 rank=8 lr=1e-5 steps=40"
echo "[bernini-rsfqt-v8-train] artifact_release=pending-v8-post-save-loader"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

"${python_bin}" -m py_compile \
  "${method_root}/feasible_quotient_objective.py" \
  "${method_root}/gauge_anchored_commutator.py" \
  "${method_root}/train_feasible_quotient_auh.py"
for pattern in \
  'test_feasible_quotient_objective.py' \
  'test_train_feasible_quotient_auh.py' \
  'test_infer_feasible_quotient_lora.py' \
  'test_finalize_feasible_quotient_checkpoint.py' \
  'test_feasible_quotient_sbatch_contract.py' \
  'test_gauge_anchored_commutator.py' \
  'test_train_prior_tangent_lora_contract.py' \
  'test_infer_relational_motion_commutator.py' \
  'test_run_relational_motion_commutator_inference_contract.py'
do
  PYTHONPATH="${method_root}" "${python_bin}" -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${method_root}/train_feasible_quotient_auh.py" \
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
  --teacher-mode paired_displacement_only \
  --relational-auxiliary-weight 0.0 \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  --inference-loader-parity-verified

pilot_checkpoint="${output_dir}/checkpoint-00000040"
[[ -f "${pilot_checkpoint}/receipt.json" && ! -L "${pilot_checkpoint}/receipt.json" ]] || fail "pilot receipt missing"
[[ -f "${pilot_checkpoint}/adapter/adapter_model.safetensors" && ! -L "${pilot_checkpoint}/adapter/adapter_model.safetensors" ]] || fail "pilot adapter missing"
validate_checkpoint_content "${checkpoint_manifest}"
"${python_bin}" "${method_root}/finalize_feasible_quotient_checkpoint.py" \
  --formal-checkpoint-root "${pilot_checkpoint}" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --base-checkpoint "${checkpoint}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}"
jq -e '
  .schema_version == "bernini-rs-fqt-auh-training-receipt-v8" and
  .method == "bernini-reconstruction-section-feasible-quotient-lora-v8-auh" and
  .global_step == 40 and
  .accepted_sigma_schedule_indices == ([range(0;40)]) and
  .training_generator_forwards == 0 and
  .inference_loader_parity_pending == false and
  .artifact_validation.verified == true and
  .artifact_validation.status == "post_save_strict_reload_complete"
' "${pilot_checkpoint}/receipt.json" >/dev/null

jq -c '
  .step_audit[0:31] as $motion |
  {
    motion_update_count: ($motion | length),
    mean_target_energy_retention: (
      [$motion[].target_energy_retention_mean_active] | add / length
    ),
    mean_target_clipped_fraction: (
      [$motion[].target_clipped_fraction_active] | add / length
    ),
    mean_radius_floor_dominated_fraction: (
      [$motion[].radius_floor_dominated_fraction_active] | add / length
    ),
    mean_required_radius_multiplier_p50: (
      [$motion[].target_required_radius_multiplier_p50] | add / length
    ),
    mean_required_radius_multiplier_p90: (
      [$motion[].target_required_radius_multiplier_p90] | add / length
    ),
    max_required_radius_multiplier: (
      [$motion[].target_required_radius_multiplier_max] | max
    ),
    mean_target_frozen_prior_cosine: (
      [$motion[].target_frozen_prior_cosine_mean_active] | add / length
    ),
    mean_target_frozen_prior_positive_fraction: (
      [$motion[].target_frozen_prior_positive_cosine_fraction_active] | add / length
    ),
    mean_target_high_frequency_fraction: (
      [$motion[].target_high_frequency_fraction_mean_active] | add / length
    ),
    mean_frozen_noop_target_over_source_error_ratio: (
      [$motion[].frozen_noop_target_over_source_error_ratio] | add / length
    )
  }
' "${pilot_checkpoint}/receipt.json"

rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-rsfqt-v8-train] PASS artifact_release=ready"
