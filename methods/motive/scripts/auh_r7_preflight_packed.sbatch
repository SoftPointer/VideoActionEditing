#!/usr/bin/env bash
#SBATCH --job-name=m7-p0-track
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1024G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=04:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?set MOTIVE_SOURCE_SNAPSHOT}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?set MOTIVE_SOURCE_TREE_SHA256}"
input_manifest="${MOTIVE_R7_P0_INPUT:?set MOTIVE_R7_P0_INPUT}"
data_root="${MOTIVE_DATA_ROOT:?set MOTIVE_DATA_ROOT}"
output_root="${MOTIVE_R7_P0_OUTPUT:?set MOTIVE_R7_P0_OUTPUT}"
python_bin="${PYTHON_BIN:?set PYTHON_BIN}"
cotracker_root="${MOTIVE_COTRACKER_ROOT:?set MOTIVE_COTRACKER_ROOT}"
cotracker_checkpoint="${MOTIVE_COTRACKER_CHECKPOINT:?set MOTIVE_COTRACKER_CHECKPOINT}"
dinov2_model="${MOTIVE_DINOV2_MODEL:?set MOTIVE_DINOV2_MODEL}"
tracker_grid_size="${MOTIVE_R7_TRACKER_GRID_SIZE:-10}"
seed="${MOTIVE_R7_DATA_SEED:-260108828}"

for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${source_snapshot}/methods/motive/motive/r7_preflight_extract.py" \
  "${source_snapshot}/methods/motive/motive/r7_temporal_teacher.py" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  "${input_manifest}" \
  "${data_root}" \
  "${python_bin}" \
  "${cotracker_root}/cotracker/predictor.py" \
  "${cotracker_checkpoint}" \
  "${dinov2_model}/config.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "[r7-p0] missing required path: ${required}" >&2
    exit 2
  fi
done

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

visible_devices="$("${python_bin}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${visible_devices}" != "8" ]]; then
  echo "[r7-p0] expected exactly eight Slurm-visible GPUs, got ${visible_devices}" >&2
  exit 2
fi

cache_parent="${SLURM_TMPDIR:-/tmp}"
cache_root="${cache_parent}/motive-r7-p0-${SLURM_JOB_ID}"
mkdir -p \
  "${cache_root}/tmp" \
  "${cache_root}/xdg" \
  "${cache_root}/huggingface" \
  "${cache_root}/torch" \
  "${cache_root}/torch-kernels" \
  "${cache_root}/inductor" \
  "${cache_root}/triton" \
  "${cache_root}/miopen" \
  "${cache_root}/miopen-custom"

export TMPDIR="${cache_root}/tmp"
export XDG_CACHE_HOME="${cache_root}/xdg"
export HF_HOME="${cache_root}/huggingface"
export TORCH_HOME="${cache_root}/torch"
export PYTORCH_KERNEL_CACHE_PATH="${cache_root}/torch-kernels"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor"
export TRITON_CACHE_DIR="${cache_root}/triton"
export MIOPEN_USER_DB_PATH="${cache_root}/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="${cache_root}/miopen-custom"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH="${source_snapshot}/methods/motive:${cotracker_root}:${PYTHONPATH:-}"

mkdir -p "${output_root}"
echo "[r7-p0] host=$(hostname) job=${SLURM_JOB_ID} GPUs=${visible_devices}"
echo "[r7-p0] input=${input_manifest} output=${output_root}"

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  -m motive.r7_preflight_extract extract \
  --input-manifest "${input_manifest}" \
  --data-root "${data_root}" \
  --output-dir "${output_root}" \
  --cotracker-checkpoint "${cotracker_checkpoint}" \
  --dinov2-model "${dinov2_model}" \
  --tracker-grid-size "${tracker_grid_size}" \
  --seed "${seed}" \
  --resume

"${python_bin}" -m motive.r7_preflight_extract finalize \
  --input-manifest "${input_manifest}" \
  --output-dir "${output_root}" \
  --resume

echo "[r7-p0] extraction and diagnostic gate completed"
