#!/usr/bin/env bash
#SBATCH --job-name=bern-r13-lora81
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=1-00:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:?set BERNINI_ACTION_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:?set BERNINI_ACTION_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_ACTION_SOURCE_REVISION:?set BERNINI_ACTION_SOURCE_REVISION}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
bernini_archive="${BERNINI_OFFICIAL_ARCHIVE:?set BERNINI_OFFICIAL_ARCHIVE}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
data_dir="${BERNINI_ACTION_PARQUET_DIR:?set BERNINI_ACTION_PARQUET_DIR}"
dataset_summary="${BERNINI_ACTION_DATASET_SUMMARY:?set BERNINI_ACTION_DATASET_SUMMARY}"
output_dir="${BERNINI_ACTION_TRAIN_OUTPUT:?set BERNINI_ACTION_TRAIN_OUTPUT}"
max_steps="${BERNINI_ACTION_MAX_STEPS:-1}"
save_every="${BERNINI_ACTION_SAVE_EVERY:-1}"
learning_rate="${BERNINI_ACTION_LR:-1e-4}"
resume="${BERNINI_ACTION_RESUME:-}"
allow_incomplete_dataset="${BERNINI_ACTION_ALLOW_INCOMPLETE_DATASET:-0}"
allow_reward_selected_targets="${BERNINI_ACTION_ALLOW_REWARD_SELECTED_TARGETS:-0}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
bernini_archive_sha256="08a331958560544efb5e842666c236d819bfdc36d34b6eb9c1cdcee1546ae670"

fail() {
  echo "[bernini-lora81] ERROR: $*" >&2
  exit 2
}

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid source archive"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini source root"
[[ -f "${bernini_archive}" && ! -L "${bernini_archive}" ]] || fail "invalid Bernini source archive"
[[ -d "${veomni_root}/.git" ]] || fail "VeOmni root must be a Git checkout"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid checkpoint"
[[ -d "${data_dir}" && ! -L "${data_dir}" ]] || fail "invalid parquet directory"
[[ -f "${dataset_summary}" && ! -L "${dataset_summary}" ]] || fail "invalid dataset summary"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid source archive hash"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ "${max_steps}" =~ ^[1-9][0-9]*$ ]] || fail "max steps must be positive"
[[ "${save_every}" =~ ^[0-9]+$ ]] || fail "save interval must be non-negative"
[[ "${allow_incomplete_dataset}" =~ ^[01]$ ]] || fail "allow-incomplete flag must be 0 or 1"
[[ "${allow_reward_selected_targets}" =~ ^[01]$ ]] || fail "allow-reward-selected flag must be 0 or 1"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash mismatch"
[[ "$(sha256sum "${bernini_archive}" | awk '{print $1}')" == "${bernini_archive_sha256}" ]] || fail "Bernini source archive hash mismatch"
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

task_scratch="${SLURM_TMPDIR:-/tmp}/bernini-lora81-${SLURM_JOB_ID}"
mkdir -p \
  "${task_scratch}/source" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"

tar -xf "${source_archive}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
[[ -f "${method_root}/train_lora.py" ]] || fail "archive lacks LoRA trainer"

mkdir -p "${output_dir%/*}"
hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-lora81] source_revision=${source_revision}"
echo "[bernini-lora81] source_archive_sha256=${source_archive_sha256}"
echo "[bernini-lora81] max_steps=${max_steps}"
echo "[bernini-lora81] data_dir=${data_dir}"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

PYTHONPATH="${method_root}" "${python_bin}" -m unittest discover \
  -s "${method_root}/tests" \
  -p 'test_train_lora_contract.py' \
  -v

command=(
  "${python_bin}" -m torch.distributed.run
  --standalone
  --nproc_per_node=4
  "${method_root}/train_lora.py"
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --preprocessed-parquet-dir "${data_dir}"
  --dataset-summary "${dataset_summary}"
  --output "${output_dir}"
  --num-frames 81
  --max-steps "${max_steps}"
  --save-every "${save_every}"
  --learning-rate "${learning_rate}"
  --expected-bernini-commit "${bernini_commit}"
  --expected-veomni-commit "${veomni_commit}"
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}"
  --method-source-revision "${source_revision}"
  --method-source-archive-sha256 "${source_archive_sha256}"
)
if [[ "${allow_incomplete_dataset}" == 1 ]]; then
  command+=(--allow-incomplete-dataset)
fi
if [[ "${allow_reward_selected_targets}" == 1 ]]; then
  command+=(--allow-reward-selected-synthetic-targets)
fi
if [[ -n "${resume}" ]]; then
  command+=(--resume "${resume}")
fi
"${command[@]}"

rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-lora81] PASS"
