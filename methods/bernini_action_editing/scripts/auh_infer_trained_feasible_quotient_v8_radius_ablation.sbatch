#!/usr/bin/env bash
#SBATCH --job-name=bern-v8-radius-ab
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=12:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 077

# This launcher is intentionally not a main-arm launcher.  It accepts only the
# two non-unit trained-V8 feasible-radius diagnostics.
runtime_source_archive="${BERNINI_V8_SOURCE_ARCHIVE:?set BERNINI_V8_SOURCE_ARCHIVE}"
runtime_source_archive_sha256="${BERNINI_V8_SOURCE_ARCHIVE_SHA256:?set BERNINI_V8_SOURCE_ARCHIVE_SHA256}"
runtime_source_revision="${BERNINI_V8_SOURCE_REVISION:?set BERNINI_V8_SOURCE_REVISION}"
runtime_source_repository="${BERNINI_V8_SOURCE_REPOSITORY:?set BERNINI_V8_SOURCE_REPOSITORY}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
adapter_checkpoint="${BERNINI_V8_ADAPTER_CHECKPOINT:?set BERNINI_V8_ADAPTER_CHECKPOINT}"
source_video="${BERNINI_V8_SOURCE_VIDEO:?set BERNINI_V8_SOURCE_VIDEO}"
instruction="${BERNINI_V8_INSTRUCTION:?set BERNINI_V8_INSTRUCTION}"
output_video="${BERNINI_V8_RADIUS_INFERENCE_OUTPUT:?set BERNINI_V8_RADIUS_INFERENCE_OUTPUT}"
radius_scale="${BERNINI_V8_RADIUS_SCALE:?set BERNINI_V8_RADIUS_SCALE to 2.5 or 4.0}"
seed="${BERNINI_V8_INFERENCE_SEED:-2027}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
checkpoint_content_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
checkpoint_content_file_count=23
operator_mode="v8_reconstruction_section_feasible_quotient_transport"
evaluation_arm="v8_trained_feasible_radius_scale_inference_only_ablation"

fail() { echo "[bernini-v8-radius-ablation] ERROR: $*" >&2; exit 2; }

case "${radius_scale}" in
  2.5|4.0) ;;
  *) fail "radius ablation launcher accepts only 2.5 or 4.0, never main scale 1.0" ;;
esac

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
[[ -f "${runtime_source_archive}" && ! -L "${runtime_source_archive}" ]] || fail "invalid runtime source archive"
[[ -d "${runtime_source_repository}/.git" ]] || fail "invalid runtime source repository"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini root"
[[ -d "${veomni_root}/.git" ]] || fail "invalid VeOmni root"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid checkpoint"
[[ -d "${adapter_checkpoint}" && ! -L "${adapter_checkpoint}" ]] || fail "invalid adapter checkpoint"
[[ -f "${adapter_checkpoint}/adapter/adapter_config.json" && ! -L "${adapter_checkpoint}/adapter/adapter_config.json" ]] || fail "adapter config missing"
[[ -f "${adapter_checkpoint}/adapter/adapter_model.safetensors" && ! -L "${adapter_checkpoint}/adapter/adapter_model.safetensors" ]] || fail "adapter tensors missing"
[[ -f "${adapter_checkpoint}/receipt.json" && ! -L "${adapter_checkpoint}/receipt.json" ]] || fail "training receipt missing"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "invalid source video"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ -n "${instruction//[[:space:]]/}" ]] || fail "instruction is empty"
[[ "${runtime_source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid runtime archive SHA-256"
[[ "${runtime_source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid runtime source revision"
[[ "${seed}" =~ ^[0-9]+$ ]] || fail "seed must be non-negative"
[[ "${output_video}" == /*.mp4 ]] || fail "output must be an absolute MP4 path"
[[ ! -e "${output_video}" && ! -L "${output_video}" ]] || fail "refusing to overwrite output"
[[ ! -e "${output_video}.receipt.json" && ! -L "${output_video}.receipt.json" ]] || fail "refusing to overwrite receipt"
[[ "$(sha256sum "${runtime_source_archive}" | awk '{print $1}')" == "${runtime_source_archive_sha256}" ]] || fail "runtime source archive hash mismatch"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_commit}" ]] || fail "VeOmni commit mismatch"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source is dirty"

training_source_revision="$(jq -er '
  select(
    .schema_version == "bernini-rs-fqt-auh-training-receipt-v8" and
    .method == "bernini-reconstruction-section-feasible-quotient-lora-v8-auh" and
    .global_step == 40 and
    .formal_40_sigma_cycle_complete == true and
    .inference_loader_parity_pending == false and
    .artifact_validation.verified == true and
    .artifact_validation.status == "post_save_strict_reload_complete"
  ) |
  .immutable_contract.value.method_source_revision
' "${adapter_checkpoint}/receipt.json")" || fail "adapter is not a finalized trained V8 checkpoint"
training_source_archive_sha256="$(jq -er '.immutable_contract.value.method_source_archive_sha256' "${adapter_checkpoint}/receipt.json")" || fail "training archive identity missing"
[[ "${training_source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid training source revision"
[[ "${training_source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid training archive SHA-256"

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

task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-v8-radius-${SLURM_JOB_ID}.XXXXXX")"
archive_copy="${task_scratch}/runtime-method.tar"
archive_from_revision="${task_scratch}/runtime-method-from-revision.tar"
source_root="${task_scratch}/source"
mkdir -p \
  "${source_root}" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton" \
  "${task_scratch}/rank-cache" \
  "${output_video%/*}"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
export BERNINI_C2FR_RANK_CACHE_ROOT="${task_scratch}/rank-cache"

cp -- "${runtime_source_archive}" "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${runtime_source_archive_sha256}" ]] || fail "private runtime archive hash mismatch"
[[ "$(git -C "${runtime_source_repository}" cat-file -t "${runtime_source_revision}^{commit}")" == "commit" ]] || fail "runtime source revision is not a repository commit"
git -C "${runtime_source_repository}" archive --format=tar \
  -o "${archive_from_revision}" \
  "${runtime_source_revision}" methods/bernini_action_editing
[[ "$(sha256sum "${archive_from_revision}" | awk '{print $1}')" == "${runtime_source_archive_sha256}" ]] || fail "runtime archive bytes are not the declared git revision"
"${python_bin}" - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

with tarfile.open(sys.argv[1], mode="r:*") as handle:
    members = handle.getmembers()
    if not members:
        raise SystemExit("empty runtime method source archive")
    for member in members:
        path = PurePosixPath(member.name)
        if not member.name or path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe archive path: {member.name!r}")
        if member.issym() or member.islnk() or member.isfifo() or member.isdev():
            raise SystemExit(f"unsafe archive member: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported archive member: {member.name!r}")
PY
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${source_root}"
method_root="${source_root}/methods/bernini_action_editing"
for required in \
  run_relational_motion_commutator_inference.py \
  infer_relational_motion_commutator.py \
  infer_feasible_quotient_lora.py \
  finalize_feasible_quotient_checkpoint.py \
  feasible_quotient_objective.py \
  gauge_anchored_commutator.py \
  motion_commutator.py \
  train_feasible_quotient_auh.py \
  train_relational_motion_commutator_auh.py \
  infer_prior_tangent_lora.py \
  train_prior_tangent_lora.py \
  train_cross_mode_cmsg_auh.py \
  train_delta_lora.py \
  train_lora.py \
  motion_residual.py \
  tri_branch_unipc.py \
  inference_sigma_strata.py \
  infer_delta_lora.py \
  tests/test_feasible_quotient_objective.py \
  tests/test_infer_feasible_quotient_lora.py \
  tests/test_finalize_feasible_quotient_checkpoint.py \
  tests/test_feasible_quotient_sbatch_contract.py \
  tests/test_gauge_anchored_commutator.py \
  tests/test_train_prior_tangent_lora_contract.py \
  tests/test_infer_relational_motion_commutator.py \
  tests/test_run_relational_motion_commutator_inference_contract.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "runtime archive lacks plain ${required}"
done
checkpoint_manifest="${method_root}/audits/bernini_r13_ff4c5d4_checkpoint.sha256"
validate_checkpoint_content "${checkpoint_manifest}"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-v8-radius-ablation] training_revision=${training_source_revision} training_archive_sha256=${training_source_archive_sha256}"
echo "[bernini-v8-radius-ablation] runtime_revision=${runtime_source_revision} runtime_archive_sha256=${runtime_source_archive_sha256}"
echo "[bernini-v8-radius-ablation] operator=${operator_mode} radius_scale=${radius_scale} arm=${evaluation_arm}"
echo "[bernini-v8-radius-ablation] frames=81 phases=21 steps=40 editor_forwards_per_step=5 generator_forwards=0 ulysses=4"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

for pattern in \
  'test_feasible_quotient_objective.py' \
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
  "${method_root}/run_relational_motion_commutator_inference.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --adapter-checkpoint "${adapter_checkpoint}" \
  --source-video "${source_video}" \
  --instruction "${instruction}" \
  --output "${output_video}" \
  --num-inference-steps 40 \
  --kappa 0.25 \
  --operator-mode "${operator_mode}" \
  --v8-radius-scale "${radius_scale}" \
  --seed "${seed}" \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${training_source_revision}" \
  --method-source-archive-sha256 "${training_source_archive_sha256}" \
  --runtime-method-source-revision "${runtime_source_revision}" \
  --runtime-method-source-archive-sha256 "${runtime_source_archive_sha256}" \
  --runtime-method-source-archive "${archive_copy}"

[[ -f "${output_video}" && ! -L "${output_video}" ]] || fail "inference output missing"
[[ -f "${output_video}.receipt.json" && ! -L "${output_video}.receipt.json" ]] || fail "inference receipt missing"
jq -e \
  --arg arm "${evaluation_arm}" \
  --argjson scale "${radius_scale}" \
  --arg training_archive "${training_source_archive_sha256}" \
  --arg runtime_archive "${runtime_source_archive_sha256}" '
  .schema_version == "bernini-rs-fqt-inference-receipt-v8" and
  .evaluation_arm == $arm and
  .inference_only_ablation == true and
  .training_matched == false and
  .training_matched_main_arm == false and
  .training_inference_alignment.operator_training_matched == false and
  .training_inference_alignment.overall_arm_training_matched == false and
  .training_inference_alignment.v8_radius_scale_inference_only_ablation == true and
  .training_inference_alignment.v8_runtime_radius_scale == $scale and
  .sampling.relational_motion_commutator_contract.training_matched == false and
  .sampling.relational_motion_commutator_contract.inference_only_ablation == true and
  .sampling.relational_motion_commutator_contract.main_operator == false and
  .sampling.relational_motion_commutator_contract.experimental_operator_ablation == true and
  .sampling.relational_motion_commutator_contract.operator_training_matched == false and
  .sampling.v8_radius_scale == $scale and
  .sampling.feasible_quotient_config.frozen_quotient_radius_ratio == $scale and
  .sampling.feasible_quotient_config.noop_dynamics_radius_ratio == ($scale * 0.25) and
  .sampling.feasible_quotient_config.radius_floor == 0.001 and
  .v8_radius_scale_diagnostic.enabled == true and
  .v8_radius_scale_diagnostic.inference_only_ablation == true and
  .v8_radius_scale_diagnostic.training_matched == false and
  .v8_radius_scale_diagnostic.training_method_source_archive_sha256 == $training_archive and
  .v8_radius_scale_diagnostic.runtime_method_source_archive_sha256 == $runtime_archive and
  .adapter.training_method_source_archive_sha256 == $training_archive and
  .runtime_method_source.archive_sha256 == $runtime_archive and
  .runtime_method_source.differs_from_training_source == true and
  .runtime_method_source.matches_training_source == false and
  .runtime_method_source.archive_hash_verified_by_runner == true and
  .runtime_method_source.executing_method_files_verified_against_archive == true
' "${output_video}.receipt.json" >/dev/null
rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-v8-radius-ablation] PASS inference_only_ablation=true training_matched=false"
