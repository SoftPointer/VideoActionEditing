#!/usr/bin/env bash
#SBATCH --job-name=bern-spt-audit8
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=01:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_root="${BERNINI_SPT_SOURCE_ROOT:?set BERNINI_SPT_SOURCE_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
data_dir="${BERNINI_ACTION_PARQUET_DIR:?set BERNINI_ACTION_PARQUET_DIR}"
dataset_summary="${BERNINI_ACTION_DATASET_SUMMARY:?set BERNINI_ACTION_DATASET_SUMMARY}"
output="${BERNINI_SPT_ORACLE_PREFIX_OUTPUT:?set BERNINI_SPT_ORACLE_PREFIX_OUTPUT}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

fail() { echo "[bernini-spt-audit8] ERROR: $*" >&2; exit 2; }
[[ -f "${source_root}/spt_v2/audit_oracle_prefix.py" ]] || fail "missing prefix audit"
[[ -f "${source_root}/train_lora.py" ]] || fail "missing latent helpers"
[[ -d "${checkpoint}" ]] || fail "missing checkpoint"
[[ -d "${data_dir}" ]] || fail "missing parquet directory"
[[ -f "${dataset_summary}" ]] || fail "missing dataset summary"
[[ ! -e "${output}" ]] || fail "refusing existing output"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

task_scratch="${SLURM_TMPDIR:-/tmp}/bernini-spt-audit8-${SLURM_JOB_ID}"
mkdir -p "${task_scratch}/method"
cp -a "${source_root}/." "${task_scratch}/method/"
method_root="${task_scratch}/method"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

PYTHONPATH="${method_root}:${method_root}/spt_v2" "${python_bin}" \
  -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${method_root}/spt_v2/audit_oracle_prefix.py" \
  --checkpoint "${checkpoint}" \
  --preprocessed-parquet-dir "${data_dir}" \
  --dataset-summary "${dataset_summary}" \
  --prefix-rows 8 \
  --teacher-feature-channels 64 \
  --output "${output}"

echo "[bernini-spt-audit8] PASS output=${output}"
