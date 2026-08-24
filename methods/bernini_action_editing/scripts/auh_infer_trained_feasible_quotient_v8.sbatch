#!/usr/bin/env bash
#SBATCH --job-name=bern-rsfqt-v8-main
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

source_archive="${BERNINI_V8_SOURCE_ARCHIVE:?set BERNINI_V8_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_V8_SOURCE_ARCHIVE_SHA256:?set BERNINI_V8_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_V8_SOURCE_REVISION:?set BERNINI_V8_SOURCE_REVISION}"
source_repository="${BERNINI_V8_SOURCE_REPOSITORY:?set BERNINI_V8_SOURCE_REPOSITORY}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
adapter_checkpoint="${BERNINI_V8_ADAPTER_CHECKPOINT:?set BERNINI_V8_ADAPTER_CHECKPOINT}"
source_video="${BERNINI_V8_SOURCE_VIDEO:?set BERNINI_V8_SOURCE_VIDEO}"
instruction="${BERNINI_V8_INSTRUCTION:?set BERNINI_V8_INSTRUCTION}"
output_video="${BERNINI_V8_INFERENCE_OUTPUT:?set BERNINI_V8_INFERENCE_OUTPUT}"
seed="${BERNINI_V8_INFERENCE_SEED:-2027}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
checkpoint_content_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
checkpoint_content_file_count=23
operator_mode="v8_reconstruction_section_feasible_quotient_transport"

fail() { echo "[bernini-rsfqt-v8-main] ERROR: $*" >&2; exit 2; }

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
[[ -d "${adapter_checkpoint}" && ! -L "${adapter_checkpoint}" ]] || fail "invalid adapter checkpoint"
[[ -f "${adapter_checkpoint}/adapter/adapter_config.json" && ! -L "${adapter_checkpoint}/adapter/adapter_config.json" ]] || fail "adapter config missing"
[[ -f "${adapter_checkpoint}/adapter/adapter_model.safetensors" && ! -L "${adapter_checkpoint}/adapter/adapter_model.safetensors" ]] || fail "adapter tensors missing"
[[ -f "${adapter_checkpoint}/receipt.json" && ! -L "${adapter_checkpoint}/receipt.json" ]] || fail "training receipt missing"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "invalid source video"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ -n "${instruction//[[:space:]]/}" ]] || fail "instruction is empty"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid archive SHA-256"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ "${seed}" =~ ^[0-9]+$ ]] || fail "seed must be non-negative"
[[ "${output_video}" == /*.mp4 ]] || fail "output must be an absolute MP4 path"
[[ ! -e "${output_video}" && ! -L "${output_video}" ]] || fail "refusing to overwrite output"
[[ ! -e "${output_video}.receipt.json" && ! -L "${output_video}.receipt.json" ]] || fail "refusing to overwrite receipt"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash mismatch"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_commit}" ]] || fail "VeOmni commit mismatch"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source is dirty"
jq -e --arg revision "${source_revision}" --arg archive "${source_archive_sha256}" '
  .schema_version == "bernini-rs-fqt-auh-training-receipt-v8" and
  .method == "bernini-reconstruction-section-feasible-quotient-lora-v8-auh" and
  .global_step == 40 and
  .formal_40_sigma_cycle_complete == true and
  .inference_loader_parity_pending == false and
  .artifact_validation.verified == true and
  .artifact_validation.status == "post_save_strict_reload_complete" and
  .immutable_contract.value.method_source_revision == $revision and
  .immutable_contract.value.method_source_archive_sha256 == $archive
' "${adapter_checkpoint}/receipt.json" >/dev/null

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

task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-rsfqt-v8-main-${SLURM_JOB_ID}.XXXXXX")"
archive_copy="${task_scratch}/method.tar"
archive_from_revision="${task_scratch}/method-from-revision.tar"
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

cp -- "${source_archive}" "${archive_copy}"
[[ "$(git -C "${source_repository}" cat-file -t "${source_revision}^{commit}")" == "commit" ]] || fail "source revision is not a repository commit"
git -C "${source_repository}" archive --format=tar \
  -o "${archive_from_revision}" \
  "${source_revision}" methods/bernini_action_editing
[[ "$(sha256sum "${archive_from_revision}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "archive bytes are not the declared git revision"
"${python_bin}" - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

with tarfile.open(sys.argv[1], mode="r:*") as handle:
    members = handle.getmembers()
    if not members:
        raise SystemExit("empty method source archive")
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
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive lacks plain ${required}"
done

checkpoint_manifest="${method_root}/audits/bernini_r13_ff4c5d4_checkpoint.sha256"
validate_checkpoint_content "${checkpoint_manifest}"
echo "[bernini-rsfqt-v8-main] checkpoint_content_manifest_sha256=${checkpoint_content_manifest_sha256} files=${checkpoint_content_file_count}"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-rsfqt-v8-main] source_revision=${source_revision} archive_sha256=${source_archive_sha256}"
echo "[bernini-rsfqt-v8-main] source_sha256=$(sha256sum "${source_video}" | awk '{print $1}')"
echo "[bernini-rsfqt-v8-main] adapter_sha256=$(sha256sum "${adapter_checkpoint}/adapter/adapter_model.safetensors" | awk '{print $1}')"
echo "[bernini-rsfqt-v8-main] frames=81 phases=21 steps=40 editor_forwards_per_step=5 generator_forwards=0 ulysses=4"
echo "[bernini-rsfqt-v8-main] operator=${operator_mode} arm=projection-consistent-training-matched-main"
echo "[bernini-rsfqt-v8-main] output=${output_video} seed=${seed}"
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
  --seed "${seed}" \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  --runtime-method-source-revision "${source_revision}" \
  --runtime-method-source-archive-sha256 "${source_archive_sha256}" \
  --runtime-method-source-archive "${archive_copy}"

[[ -f "${output_video}" && ! -L "${output_video}" ]] || fail "inference output missing"
[[ -f "${output_video}.receipt.json" && ! -L "${output_video}.receipt.json" ]] || fail "inference receipt missing"
jq -e '
  .schema_version == "bernini-rs-fqt-inference-receipt-v8" and
  .evaluation_arm == "v8_projection_consistent_training_matched_main" and
  .training_matched_main_arm == true and
  .inference_only_ablation == false and
  .training_inference_alignment.operator_training_matched == true and
  .training_inference_alignment.v8_objective_is_projection_consistent == true and
  .runtime_method_source.matches_training_source == true
' "${output_video}.receipt.json" >/dev/null
rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-rsfqt-v8-main] PASS"
