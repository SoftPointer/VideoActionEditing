#!/usr/bin/env bash
#SBATCH --job-name=m7-candtrk
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
input_dir="${MOTIVE_R7_CANDIDATE_TEMPORAL_INPUT:?set MOTIVE_R7_CANDIDATE_TEMPORAL_INPUT}"
input_done_sha256="${MOTIVE_R7_CANDIDATE_TEMPORAL_DONE_SHA256:?set MOTIVE_R7_CANDIDATE_TEMPORAL_DONE_SHA256}"
data_root="${MOTIVE_DATA_ROOT:?set MOTIVE_DATA_ROOT}"
output_root="${MOTIVE_R7_CANDIDATE_TRACK_CACHE_OUTPUT:?set MOTIVE_R7_CANDIDATE_TRACK_CACHE_OUTPUT}"
python_bin="${PYTHON_BIN:?set PYTHON_BIN}"
cotracker_root="${MOTIVE_COTRACKER_ROOT:?set MOTIVE_COTRACKER_ROOT}"
cotracker_checkpoint="${MOTIVE_COTRACKER_CHECKPOINT:?set MOTIVE_COTRACKER_CHECKPOINT}"
tracker_grid_size="${MOTIVE_R7_TRACKER_GRID_SIZE:-16}"
seed="${MOTIVE_R7_DATA_SEED:-260108828}"

for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${source_snapshot}/methods/motive/motive/r7_candidate_temporal_manifest.py" \
  "${source_snapshot}/methods/motive/motive/r7_candidate_track_cache.py" \
  "${source_snapshot}/methods/motive/motive/r7_track_cache.py" \
  "${source_snapshot}/methods/motive/motive/r7_preflight_extract.py" \
  "${source_snapshot}/methods/motive/motive/r7_temporal_teacher.py" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  "${input_dir}/manifest.jsonl" \
  "${input_dir}/summary.json" \
  "${input_dir}/done.json" \
  "${data_root}" \
  "${python_bin}" \
  "${cotracker_root}/cotracker/predictor.py" \
  "${cotracker_checkpoint}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[r7-candidate-cache] missing required path: ${required}" >&2
    exit 2
  fi
done

if [[ ! -d "${source_snapshot}" || ! -d "${input_dir}" || ! -d "${data_root}" || ! -d "${cotracker_root}" ]]; then
  echo "[r7-candidate-cache] source/input/data/CoTracker roots must be directories" >&2
  exit 2
fi
if [[ ! -f "${input_dir}/done.json" || ! -f "${cotracker_checkpoint}" || ! -x "${python_bin}" ]]; then
  echo "[r7-candidate-cache] done/checkpoint must be files and Python executable" >&2
  exit 2
fi
if [[ ! "${input_done_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "[r7-candidate-cache] expected input done SHA-256 is invalid" >&2
  exit 2
fi
if [[ ! "${tracker_grid_size}" =~ ^[0-9]+$ ]] || (( tracker_grid_size < 4 )); then
  echo "[r7-candidate-cache] tracker grid size must be an integer >= 4" >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "[r7-candidate-cache] flock is required for single-writer publication" >&2
  exit 2
fi
mkdir -p "$(dirname "${output_root}")"
exec 9>"${output_root}.slurm.lock"
if ! flock -n 9; then
  echo "[r7-candidate-cache] another writer holds ${output_root}.slurm.lock" >&2
  exit 4
fi

actual_input_done_sha256="$(sha256sum "${input_dir}/done.json" | awk '{print $1}')"
if [[ "${actual_input_done_sha256}" != "${input_done_sha256}" ]]; then
  echo "[r7-candidate-cache] input done SHA-256 differs" >&2
  exit 2
fi

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

visible_devices="$("${python_bin}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${visible_devices}" != "8" ]]; then
  echo "[r7-candidate-cache] expected 8 Slurm-visible GPUs, got ${visible_devices}" >&2
  exit 2
fi

cache_parent="${SLURM_TMPDIR:-/tmp}"
cache_root="${cache_parent}/motive-r7-candidate-cache-${SLURM_JOB_ID}"
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
cd "${source_snapshot}"

readarray -t imported_paths < <(
  "${python_bin}" -c '
from pathlib import Path
import motive.r7_candidate_track_cache as candidate_cache
import cotracker.predictor as predictor
print(Path(candidate_cache.__file__).resolve())
print(Path(predictor.__file__).resolve())
'
)
expected_cache_module="${source_snapshot}/methods/motive/motive/r7_candidate_track_cache.py"
expected_predictor_module="${cotracker_root}/cotracker/predictor.py"
if [[ "${#imported_paths[@]}" != "2" ]] \
  || [[ "${imported_paths[0]}" != "${expected_cache_module}" ]] \
  || [[ "${imported_paths[1]}" != "${expected_predictor_module}" ]]; then
  echo "[r7-candidate-cache] imported module origins differ from frozen inputs" >&2
  printf "cache=%s\npredictor=%s\n" "${imported_paths[@]:-missing}" >&2
  exit 2
fi

"${python_bin}" -m motive.r7_candidate_temporal_manifest validate \
  --input-dir "${input_dir}"

if [[ -e "${output_root}/final/done.json" ]]; then
  "${python_bin}" -m motive.r7_candidate_track_cache finalize \
    --input-dir "${input_dir}" \
    --expected-input-done-sha256 "${input_done_sha256}" \
    --output-dir "${output_root}" \
    --resume
  "${python_bin}" -m motive.r7_candidate_track_cache validate \
    --input-dir "${input_dir}" \
    --expected-input-done-sha256 "${input_done_sha256}" \
    --output-dir "${output_root}" \
    --final
  echo "[r7-candidate-cache] existing final repaired/validated; no extraction run"
  exit 0
fi

"${python_bin}" -m motive.r7_candidate_track_cache preflight \
  --input-dir "${input_dir}" \
  --expected-input-done-sha256 "${input_done_sha256}" \
  --data-root "${data_root}" \
  --output-dir "${output_root}" \
  --cotracker-checkpoint "${cotracker_checkpoint}" \
  --cotracker-root "${cotracker_root}" \
  --source-snapshot "${source_snapshot}"

"${python_bin}" -m motive.r7_candidate_track_cache runtime-preflight \
  --input-dir "${input_dir}" \
  --expected-input-done-sha256 "${input_done_sha256}" \
  --data-root "${data_root}" \
  --output-dir "${output_root}" \
  --cotracker-checkpoint "${cotracker_checkpoint}" \
  --cotracker-root "${cotracker_root}" \
  --source-snapshot "${source_snapshot}" \
  --tracker-grid-size "${tracker_grid_size}" \
  --seed "${seed}"

echo "[r7-candidate-cache] host=$(hostname) job=${SLURM_JOB_ID} GPUs=${visible_devices}"
echo "[r7-candidate-cache] input=${input_dir} output=${output_root}"
echo "[r7-candidate-cache] rows=1187 grid=${tracker_grid_size} tracks=$((tracker_grid_size * tracker_grid_size))"

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  --max_restarts=0 \
  -m motive.r7_candidate_track_cache extract \
  --input-dir "${input_dir}" \
  --expected-input-done-sha256 "${input_done_sha256}" \
  --data-root "${data_root}" \
  --output-dir "${output_root}" \
  --cotracker-checkpoint "${cotracker_checkpoint}" \
  --cotracker-root "${cotracker_root}" \
  --source-snapshot "${source_snapshot}" \
  --tracker-grid-size "${tracker_grid_size}" \
  --seed "${seed}" \
  --resume

"${python_bin}" -m motive.r7_candidate_track_cache finalize \
  --input-dir "${input_dir}" \
  --expected-input-done-sha256 "${input_done_sha256}" \
  --output-dir "${output_root}" \
  --resume

"${python_bin}" -m motive.r7_candidate_track_cache validate \
  --input-dir "${input_dir}" \
  --expected-input-done-sha256 "${input_done_sha256}" \
  --output-dir "${output_root}" \
  --final

echo "[r7-candidate-cache] immutable exact-eight-shard cache completed"
