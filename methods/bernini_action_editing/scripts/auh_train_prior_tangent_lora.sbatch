#!/usr/bin/env bash
#SBATCH --job-name=bern-pgt-v5-81
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

source_archive="${BERNINI_V5_SOURCE_ARCHIVE:?set BERNINI_V5_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_V5_SOURCE_ARCHIVE_SHA256:?set BERNINI_V5_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_V5_SOURCE_REVISION:?set BERNINI_V5_SOURCE_REVISION}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
data_dir="${BERNINI_ACTION_PARQUET_DIR:?set BERNINI_ACTION_PARQUET_DIR}"
dataset_summary="${BERNINI_ACTION_DATASET_SUMMARY:?set BERNINI_ACTION_DATASET_SUMMARY}"
routing_jsonl="${BERNINI_V5_ROUTING_JSONL:?set BERNINI_V5_ROUTING_JSONL}"
routing_sha256="${BERNINI_V5_ROUTING_JSONL_SHA256:?set BERNINI_V5_ROUTING_JSONL_SHA256}"
output_dir="${BERNINI_V5_TRAIN_OUTPUT:?set BERNINI_V5_TRAIN_OUTPUT}"
max_steps="${BERNINI_V5_MAX_STEPS:?set BERNINI_V5_MAX_STEPS}"
save_every="${BERNINI_V5_SAVE_EVERY:?set BERNINI_V5_SAVE_EVERY}"
resume="${BERNINI_V5_RESUME:-}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
inverse_sigma_weight_floor="0.11765105277299881"

fail() {
  echo "[bernini-pgt-v5-81] ERROR: $*" >&2
  exit 2
}

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent must be an absolute writable plain directory"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid method archive"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini root"
[[ -d "${veomni_root}/.git" ]] || fail "VeOmni root must be a Git checkout"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid checkpoint"
[[ -d "${data_dir}" && ! -L "${data_dir}" ]] || fail "invalid parquet directory"
[[ -f "${dataset_summary}" && ! -L "${dataset_summary}" ]] || fail "invalid dataset summary"
[[ -f "${routing_jsonl}" && ! -L "${routing_jsonl}" ]] || fail "invalid routing JSONL"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid archive SHA-256"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ "${routing_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid routing SHA-256"
[[ "${max_steps}" =~ ^[1-9][0-9]*$ ]] || fail "max steps must be positive"
[[ "${save_every}" =~ ^[0-9]+$ ]] || fail "save interval must be non-negative"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive hash mismatch"
[[ "$(sha256sum "${routing_jsonl}" | awk '{print $1}')" == "${routing_sha256}" ]] || fail "routing hash mismatch"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_commit}" ]] || fail "VeOmni commit mismatch"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source is dirty"
if [[ -n "${resume}" ]]; then
  [[ -d "${resume}" && ! -L "${resume}" ]] || fail "invalid resume checkpoint"
fi

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

task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-pgt-v5-81-${SLURM_JOB_ID}.XXXXXX")"
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
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "private archive copy hash mismatch"
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${source_root}"
[[ -z "$(find "${source_root}" -type l -print -quit)" ]] || fail "method archive contains a symbolic link"

method_root="${source_root}/methods/bernini_action_editing"
for required in \
  prior_guided_tangent.py \
  train_prior_tangent_lora.py \
  inference_sigma_strata.py \
  motion_residual.py \
  train_delta_lora.py \
  train_lora.py \
  tri_branch_unipc.py \
  tests/test_prior_guided_tangent.py \
  tests/test_train_prior_tangent_lora_contract.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive lacks plain ${required}"
done

export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-pgt-v5-81] source_revision=${source_revision}"
echo "[bernini-pgt-v5-81] source_archive_sha256=${source_archive_sha256}"
echo "[bernini-pgt-v5-81] data=${data_dir} routing_sha256=${routing_sha256}"
echo "[bernini-pgt-v5-81] output=${output_dir} max_steps=${max_steps} save_every=${save_every}"
echo "[bernini-pgt-v5-81] frames=81 phases=21 forwards_per_step=8 scope=cross_q rank=8 lr=2e-5 gamma=24-full+11-cosine+5-frozen"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

"${python_bin}" -m py_compile \
  "${method_root}/prior_guided_tangent.py" \
  "${method_root}/train_prior_tangent_lora.py"
for pattern in \
  test_prior_guided_tangent.py \
  test_train_prior_tangent_lora_contract.py \
  test_inference_sigma_strata.py \
  test_motion_residual.py \
  test_train_delta_lora_contract.py
do
  PYTHONPATH="${method_root}" "${python_bin}" -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

command=(
  "${python_bin}" -m torch.distributed.run
  --standalone
  --nproc_per_node=4
  "${method_root}/train_prior_tangent_lora.py"
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --preprocessed-parquet-dir "${data_dir}"
  --dataset-summary "${dataset_summary}"
  --routing-jsonl "${routing_jsonl}"
  --expected-routing-jsonl-sha256 "${routing_sha256}"
  --unreviewed-tier reject
  --output "${output_dir}"
  --num-frames 81
  --max-steps "${max_steps}"
  --save-every "${save_every}"
  --learning-rate 2e-5
  --lora-scope cross_q
  --branch-state-mode source_target_bridge_clean_field
  --minimum-training-sigma 0.1
  --inverse-sigma-weight-floor "${inverse_sigma_weight_floor}"
  --motion-loss-weight 1.0
  --copy-loss-weight 0.0
  --boundary-gauge-loss-weight 0.0
  --anchor-loss-weight 0.0
  --motion-objective causal_boundary_charbonnier
  --bridge-consistency-weight 0.05
  --late-replay-loss-weight 0.10
  --charbonnier-scale 0.1
  --high-noise-floor 1.0
  --expected-bernini-commit "${bernini_commit}"
  --expected-veomni-commit "${veomni_commit}"
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}"
  --method-source-revision "${source_revision}"
  --method-source-archive-sha256 "${source_archive_sha256}"
)
if [[ -n "${resume}" ]]; then
  command+=(--resume "${resume}")
fi
"${command[@]}"

rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-pgt-v5-81] PASS"
