#!/usr/bin/env bash
#SBATCH --job-name=bern-spt-oracle
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=01:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

spt_root="${BERNINI_SPT_ROOT:?set BERNINI_SPT_ROOT}"
legacy_root="${BERNINI_SPT_LEGACY_ROOT:?set BERNINI_SPT_LEGACY_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
data_dir="${BERNINI_ACTION_PARQUET_DIR:?set BERNINI_ACTION_PARQUET_DIR}"
output="${BERNINI_SPT_ORACLE_OUTPUT:?set BERNINI_SPT_ORACLE_OUTPUT}"
row_index="${BERNINI_SPT_ROW_INDEX:-0}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

[[ -f "${spt_root}/oracle_diagnostic.py" ]] || { echo "missing SPT diagnostic" >&2; exit 2; }
[[ -f "${legacy_root}/motion_residual.py" ]] || { echo "missing legacy latent helper" >&2; exit 2; }
[[ -d "${checkpoint}" ]] || { echo "missing checkpoint" >&2; exit 2; }
[[ -d "${data_dir}" ]] || { echo "missing data" >&2; exit 2; }
[[ "${row_index}" =~ ^[0-9]+$ ]] || { echo "invalid row index" >&2; exit 2; }
[[ ! -e "${output}" ]] || { echo "refusing existing output" >&2; exit 2; }

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

PYTHONPATH="${legacy_root}:${spt_root%/spt_v2}" "${python_bin}" \
  "${spt_root}/oracle_diagnostic.py" \
  --checkpoint "${checkpoint}" \
  --preprocessed-parquet-dir "${data_dir}" \
  --row-index "${row_index}" \
  --output "${output}"

echo "[bernini-spt-oracle] PASS output=${output}"
