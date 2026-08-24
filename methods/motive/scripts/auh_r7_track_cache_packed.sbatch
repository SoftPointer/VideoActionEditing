#!/usr/bin/env bash
#SBATCH --job-name=m7-p1-cache
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1024G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=02:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?set MOTIVE_SOURCE_SNAPSHOT}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?set MOTIVE_SOURCE_TREE_SHA256}"
input_manifest="${MOTIVE_R7_P1_INPUT:?set MOTIVE_R7_P1_INPUT}"
data_root="${MOTIVE_DATA_ROOT:?set MOTIVE_DATA_ROOT}"
output_root="${MOTIVE_R7_P1_CACHE_OUTPUT:?set MOTIVE_R7_P1_CACHE_OUTPUT}"
python_bin="${PYTHON_BIN:?set PYTHON_BIN}"
cotracker_root="${MOTIVE_COTRACKER_ROOT:?set MOTIVE_COTRACKER_ROOT}"
cotracker_checkpoint="${MOTIVE_COTRACKER_CHECKPOINT:?set MOTIVE_COTRACKER_CHECKPOINT}"
tracker_grid_size="${MOTIVE_R7_TRACKER_GRID_SIZE:-16}"
seed="${MOTIVE_R7_DATA_SEED:-260108828}"

for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${source_snapshot}/methods/motive/motive/r7_track_cache.py" \
  "${source_snapshot}/methods/motive/motive/r7_preflight_extract.py" \
  "${source_snapshot}/methods/motive/motive/r7_temporal_teacher.py" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  "${input_manifest}" \
  "${data_root}" \
  "${python_bin}" \
  "${cotracker_root}/cotracker/predictor.py" \
  "${cotracker_checkpoint}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[r7-p1-cache] missing required path: ${required}" >&2
    exit 2
  fi
done

if [[ ! -d "${source_snapshot}" || ! -d "${data_root}" || ! -d "${cotracker_root}" ]]; then
  echo "[r7-p1-cache] source/data/CoTracker roots must be directories" >&2
  exit 2
fi
if [[ ! -f "${input_manifest}" || ! -f "${cotracker_checkpoint}" || ! -x "${python_bin}" ]]; then
  echo "[r7-p1-cache] manifest/checkpoint must be files and Python executable" >&2
  exit 2
fi
if [[ ! "${tracker_grid_size}" =~ ^[0-9]+$ ]] || (( tracker_grid_size < 2 )); then
  echo "[r7-p1-cache] tracker grid size must be an integer >= 2" >&2
  exit 2
fi
if [[ -e "${output_root}/final/done.json" ]]; then
  echo "[r7-p1-cache] completed final exists; refusing an 8-GPU validation-only resume" >&2
  exit 3
fi

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

visible_devices="$("${python_bin}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${visible_devices}" != "8" ]]; then
  echo "[r7-p1-cache] expected 8 Slurm-visible GPUs, got ${visible_devices}" >&2
  exit 2
fi

cache_parent="${SLURM_TMPDIR:-/tmp}"
cache_root="${cache_parent}/motive-r7-p1-cache-${SLURM_JOB_ID}"
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
export PYTHONHASHSEED="${seed}"

"${python_bin}" -m motive.r7_track_cache preflight \
  --input-manifest "${input_manifest}" \
  --data-root "${data_root}" \
  --output-dir "${output_root}" \
  --cotracker-checkpoint "${cotracker_checkpoint}" \
  --cotracker-root "${cotracker_root}" \
  --source-snapshot "${source_snapshot}"

echo "[r7-p1-cache] host=$(hostname) job=${SLURM_JOB_ID} GPUs=${visible_devices}"
echo "[r7-p1-cache] input=${input_manifest} output=${output_root}"
echo "[r7-p1-cache] grid=${tracker_grid_size} (tracks=$((tracker_grid_size * tracker_grid_size)))"

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  --max_restarts=0 \
  -m motive.r7_track_cache extract \
  --input-manifest "${input_manifest}" \
  --data-root "${data_root}" \
  --output-dir "${output_root}" \
  --cotracker-checkpoint "${cotracker_checkpoint}" \
  --cotracker-root "${cotracker_root}" \
  --source-snapshot "${source_snapshot}" \
  --tracker-grid-size "${tracker_grid_size}" \
  --seed "${seed}" \
  --resume

"${python_bin}" -m motive.r7_track_cache finalize \
  --input-manifest "${input_manifest}" \
  --output-dir "${output_root}" \
  --resume

echo "[r7-p1-cache] immutable 8-shard cache completed"
