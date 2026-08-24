#!/usr/bin/env bash
#SBATCH --job-name=bern-rmc-v7-81
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

source_archive="${BERNINI_V7_SOURCE_ARCHIVE:?set BERNINI_V7_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_V7_SOURCE_ARCHIVE_SHA256:?set BERNINI_V7_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_V7_SOURCE_REVISION:?set BERNINI_V7_SOURCE_REVISION}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
data_dir="${BERNINI_ACTION_PARQUET_DIR:?set BERNINI_ACTION_PARQUET_DIR}"
dataset_summary="${BERNINI_ACTION_DATASET_SUMMARY:?set BERNINI_ACTION_DATASET_SUMMARY}"
routing_jsonl="${BERNINI_V7_ROUTING_JSONL:?set BERNINI_V7_ROUTING_JSONL}"
routing_sha256="${BERNINI_V7_ROUTING_JSONL_SHA256:?set BERNINI_V7_ROUTING_JSONL_SHA256}"
output_dir="${BERNINI_V7_TRAIN_OUTPUT:?set BERNINI_V7_TRAIN_OUTPUT}"
max_steps="${BERNINI_V7_MAX_STEPS:-40}"
save_every="${BERNINI_V7_SAVE_EVERY:-40}"
teacher_mode="${BERNINI_V7_TEACHER_MODE:-target_only}"
relational_weight="${BERNINI_V7_RELATIONAL_WEIGHT:-0.0}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"

fail() { echo "[bernini-rmc-v7-81] ERROR: $*" >&2; exit 2; }

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid source archive"
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
[[ "${max_steps}" =~ ^[1-9][0-9]*$ ]] || fail "max steps must be positive"
[[ "${save_every}" =~ ^[0-9]+$ ]] || fail "save interval must be nonnegative"
[[ "${teacher_mode}" == target_only || "${teacher_mode}" == relational_auxiliary ]] || fail "invalid teacher mode"
if [[ "${teacher_mode}" == target_only && "${relational_weight}" != 0.0 && "${relational_weight}" != 0 ]]; then
  fail "target_only requires relational weight zero"
fi
if [[ "${teacher_mode}" == relational_auxiliary && "${relational_weight}" == 0.0 ]]; then
  fail "relational_auxiliary requires positive weight"
fi
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

task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-rmc-v7-81-${SLURM_JOB_ID}.XXXXXX")"
archive_copy="${task_scratch}/method.tar"
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
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${source_root}"
[[ -z "$(find "${source_root}" -type l -print -quit)" ]] || fail "source archive contains a symlink"

method_root="${source_root}/methods/bernini_action_editing"
for required in \
  cross_mode_branches.py \
  cross_mode_motion_kernel.py \
  cross_mode_motion_spectrum.py \
  motion_commutator.py \
  relational_commutator_objective.py \
  infer_relational_motion_commutator.py \
  run_relational_motion_commutator_inference.py \
  finalize_relational_motion_commutator_checkpoint.py \
  infer_prior_tangent_lora.py \
  infer_delta_lora.py \
  train_relational_motion_commutator_auh.py \
  train_cross_mode_cmsg_lora.py \
  train_cross_mode_cmsg_auh.py \
  train_prior_tangent_lora.py \
  train_delta_lora.py \
  train_lora.py \
  motion_residual.py \
  inference_sigma_strata.py \
  tri_branch_unipc.py \
  tests/test_cross_mode_motion_kernel.py \
  tests/test_motion_commutator.py \
  tests/test_relational_commutator_objective.py \
  tests/test_train_relational_motion_commutator_auh_contract.py \
  tests/test_infer_relational_motion_commutator.py \
  tests/test_run_relational_motion_commutator_inference_contract.py \
  tests/test_finalize_relational_motion_commutator_checkpoint.py \
  tests/test_infer_delta_lora_contract.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive lacks plain ${required}"
done

export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-rmc-v7-81] source_revision=${source_revision} archive_sha256=${source_archive_sha256}"
echo "[bernini-rmc-v7-81] routing_sha256=${routing_sha256} output=${output_dir}"
echo "[bernini-rmc-v7-81] frames=81 phases=21 beta=0 forwards=7 graph_forwards=2 scope=exact46 rank=8 lr=2e-5"
echo "[bernini-rmc-v7-81] teacher_mode=${teacher_mode} relational_weight=${relational_weight} max_steps=${max_steps} save_every=${save_every}"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

"${python_bin}" -m py_compile \
  "${method_root}/cross_mode_motion_kernel.py" \
  "${method_root}/motion_commutator.py" \
  "${method_root}/relational_commutator_objective.py" \
  "${method_root}/train_relational_motion_commutator_auh.py" \
  "${method_root}/infer_relational_motion_commutator.py" \
  "${method_root}/finalize_relational_motion_commutator_checkpoint.py"
for pattern in \
  'test_cross_mode_motion_kernel.py' \
  'test_motion_commutator.py' \
  'test_relational_commutator_objective.py' \
  'test_train_relational_motion_commutator_auh_contract.py' \
  'test_infer_relational_motion_commutator.py' \
  'test_run_relational_motion_commutator_inference_contract.py' \
  'test_finalize_relational_motion_commutator_checkpoint.py' \
  'test_infer_delta_lora_contract.py'
do
  PYTHONPATH="${method_root}" "${python_bin}" -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${method_root}/train_relational_motion_commutator_auh.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --preprocessed-parquet-dir "${data_dir}" \
  --dataset-summary "${dataset_summary}" \
  --routing-jsonl "${routing_jsonl}" \
  --expected-routing-jsonl-sha256 "${routing_sha256}" \
  --output "${output_dir}" \
  --num-frames 81 \
  --max-steps "${max_steps}" \
  --save-every "${save_every}" \
  --learning-rate 2e-5 \
  --teacher-mode "${teacher_mode}" \
  --relational-auxiliary-weight "${relational_weight}" \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  --inference-loader-parity-verified

if [[ \
  "${max_steps}" == 40 && \
  "${save_every}" == 40 && \
  "${teacher_mode}" == target_only && \
  ( "${relational_weight}" == 0.0 || "${relational_weight}" == 0 ) \
]]; then
  formal_checkpoint="${output_dir}/checkpoint-00000040"
  echo "[bernini-rmc-v7-81] post_save_strict_reload=${formal_checkpoint}"
  "${python_bin}" \
    "${method_root}/finalize_relational_motion_commutator_checkpoint.py" \
    --formal-checkpoint-root "${formal_checkpoint}" \
    --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" \
    --base-checkpoint "${checkpoint}" \
    --method-source-revision "${source_revision}" \
    --method-source-archive-sha256 "${source_archive_sha256}" \
    --expected-bernini-commit "${bernini_commit}" \
    --expected-veomni-commit "${veomni_commit}" \
    --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}"
  jq -e '
    .inference_loader_parity_pending == false and
    .artifact_validation.verified == true and
    .artifact_validation.status == "post_save_strict_reload_complete" and
    .artifact_validation.expanded_target_module_count == 46 and
    .artifact_validation.adapter_tensor_count == 92 and
    .artifact_validation.active_lora_module_count == 46
  ' "${formal_checkpoint}/receipt.json" >/dev/null
  echo "[bernini-rmc-v7-81] artifact_release=ready"
else
  echo "[bernini-rmc-v7-81] artifact_release=pending (non-formal arm)"
fi

rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-rmc-v7-81] PASS"
