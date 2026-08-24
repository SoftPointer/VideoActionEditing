#!/usr/bin/env bash
#SBATCH --job-name=bern-r13-vae81
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=03:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:?set BERNINI_ACTION_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:?set BERNINI_ACTION_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_ACTION_SOURCE_REVISION:?set BERNINI_ACTION_SOURCE_REVISION}"
raw_parquet="${BERNINI_ACTION_RAW_PARQUET:?set BERNINI_ACTION_RAW_PARQUET}"
raw_receipt="${BERNINI_ACTION_RAW_RECEIPT:?set BERNINI_ACTION_RAW_RECEIPT}"
raw_job_done="${BERNINI_ACTION_RAW_JOB_DONE:?set BERNINI_ACTION_RAW_JOB_DONE}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
output_root="${BERNINI_ACTION_VAE_OUTPUT:?set BERNINI_ACTION_VAE_OUTPUT}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

fail() {
  echo "[bernini-vae81-smoke] ERROR: $*" >&2
  exit 2
}

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid source archive"
[[ -f "${raw_parquet}" && ! -L "${raw_parquet}" ]] || fail "invalid raw parquet"
[[ -f "${raw_receipt}" && ! -L "${raw_receipt}" ]] || fail "invalid raw receipt"
[[ -f "${raw_job_done}" && ! -L "${raw_job_done}" ]] || fail "invalid raw job-done receipt"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid checkpoint"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid source archive hash"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash mismatch"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

task_scratch="${SLURM_TMPDIR:-/tmp}/bernini-vae81-${SLURM_JOB_ID}"
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
[[ -f "${method_root}/tools/materialize_vae.py" ]] || fail "archive lacks VAE materializer"

mkdir -p "${output_root}"
hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-vae81-smoke] source_revision=${source_revision}"
echo "[bernini-vae81-smoke] source_archive_sha256=${source_archive_sha256}"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

PYTHONPATH="${method_root}" "${python_bin}" -m unittest discover \
  -s "${method_root}/tests" \
  -p 'test_*.py' \
  -v

"${python_bin}" "${method_root}/tools/materialize_vae.py" \
  --raw-parquet "${raw_parquet}" \
  --raw-receipt "${raw_receipt}" \
  --raw-job-done "${raw_job_done}" \
  --checkpoint "${checkpoint}" \
  --output-root "${output_root}" \
  --device cuda:0 \
  --max-rows 1 \
  --max-pixels 245760 \
  --stride 16 \
  --min-target-retention 0.98

"${python_bin}" "${method_root}/tools/finalize_vae_dataset.py" \
  --raw-receipt "${raw_receipt}" \
  --raw-job-done "${raw_job_done}" \
  --materialized-root "${output_root}" \
  --output-index "${output_root}/smoke_index.jsonl" \
  --output-summary "${output_root}/smoke_summary.json" \
  --allow-missing

echo "[bernini-vae81-smoke] PASS"
