#!/usr/bin/env bash
#SBATCH --job-name=m7-vis3167
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
input_manifest="${MOTIVE_R7_VISUAL_CANDIDATES:?set MOTIVE_R7_VISUAL_CANDIDATES}"
input_sha256="${MOTIVE_R7_VISUAL_CANDIDATES_SHA256:?set MOTIVE_R7_VISUAL_CANDIDATES_SHA256}"
data_root="${MOTIVE_DATA_ROOT:?set MOTIVE_DATA_ROOT}"
output_root="${MOTIVE_R7_VISUAL_FEATURE_OUTPUT:?set MOTIVE_R7_VISUAL_FEATURE_OUTPUT}"
python_bin="${PYTHON_BIN:?set PYTHON_BIN}"
dinov2_model="${MOTIVE_DINOV2_MODEL:?set MOTIVE_DINOV2_MODEL}"
dinov2_revision="${MOTIVE_DINOV2_REVISION:?set MOTIVE_DINOV2_REVISION}"

for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${source_snapshot}/methods/motive/motive/r7_expansion_visual_features.py" \
  "${source_snapshot}/methods/motive/motive/r7_visual_candidate_manifest.py" \
  "${source_snapshot}/methods/motive/motive/r7_preflight_extract.py" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  "${input_manifest}" \
  "${data_root}" \
  "${python_bin}" \
  "${dinov2_model}/config.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "[r7-visual-features] missing required path: ${required}" >&2
    exit 2
  fi
done

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

actual_input_sha256="$("${python_bin}" -c '
import hashlib, pathlib, sys
path = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
' "${input_manifest}")"
if [[ "${actual_input_sha256}" != "${input_sha256}" ]]; then
  echo "[r7-visual-features] candidate manifest SHA differs" >&2
  exit 2
fi

visible_devices="$("${python_bin}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${visible_devices}" != "8" ]]; then
  echo "[r7-visual-features] expected 8 visible GPUs, got ${visible_devices}" >&2
  exit 2
fi

cache_parent="${SLURM_TMPDIR:-/tmp}"
cache_root="${cache_parent}/motive-r7-visual-${SLURM_JOB_ID}"
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
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH="${source_snapshot}/methods/motive"

mkdir -p "${output_root}"
shards_root="${output_root}/shards"
shard_count=0
if [[ -d "${shards_root}" ]]; then
  shard_count="$(find "${shards_root}" -mindepth 1 -maxdepth 1 -type d -name 'rank-???-of-008' | wc -l | tr -d ' ')"
fi
extract_resume=()
if [[ "${shard_count}" == "8" ]]; then
  extract_resume=(--resume)
elif [[ "${shard_count}" != "0" ]]; then
  echo "[r7-visual-features] refusing partial shard set: ${shard_count}/8" >&2
  exit 2
fi

echo "[r7-visual-features] host=$(hostname) job=${SLURM_JOB_ID} GPUs=${visible_devices}"
echo "[r7-visual-features] rows_manifest=${input_manifest} output=${output_root}"

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  -m motive.r7_expansion_visual_features extract \
  --input-manifest "${input_manifest}" \
  --data-root "${data_root}" \
  --output-dir "${output_root}" \
  --dinov2-model "${dinov2_model}" \
  --dinov2-revision "${dinov2_revision}" \
  "${extract_resume[@]}"

final_resume=()
if [[ -d "${output_root}/final" ]]; then
  final_resume=(--resume)
fi
"${python_bin}" -m motive.r7_expansion_visual_features finalize \
  --input-manifest "${input_manifest}" \
  --output-dir "${output_root}" \
  "${final_resume[@]}"

"${python_bin}" - "${output_root}/final/done.json" <<'PY'
import json
import pathlib
import sys

done = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if done.get("split_ready") is not True:
    raise SystemExit(
        "[r7-visual-features] final commit is not split-ready; "
        "inspect failed video sides"
    )
print(
    "[r7-visual-features] split-ready rows="
    f"{done.get('rows')} world_size={done.get('world_size')}",
    flush=True,
)
PY

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

echo "[r7-visual-features] completed"
