#!/usr/bin/env bash
#SBATCH --job-name=goku-anchor-prefilter
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
# AUH's bgqos has MinTRES=gres/gpu=1, including CPU-heavy jobs.
# Reserve the minimum legal device even though media probing remains on CPU.
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=04:00:00

set -Eeuo pipefail
umask 077

source_snapshot="${MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT:?set MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT}"
input_fused="${MOTIVE_GOKU_ACTION_INPUT_FUSED:?set MOTIVE_GOKU_ACTION_INPUT_FUSED}"
video_root="${MOTIVE_GOKU_ACTION_VIDEO_ROOT:?set MOTIVE_GOKU_ACTION_VIDEO_ROOT}"
output_dir="${MOTIVE_GOKU_ACTION_PREFILTER_OUTPUT:?set MOTIVE_GOKU_ACTION_PREFILTER_OUTPUT}"
python_bin="${MOTIVE_GOKU_ACTION_PYTHON_BIN:?set MOTIVE_GOKU_ACTION_PYTHON_BIN}"
sample_size="${MOTIVE_GOKU_ACTION_PREFILTER_SIZE:-768}"
workers="${MOTIVE_GOKU_ACTION_PREFILTER_WORKERS:-${SLURM_CPUS_PER_TASK:-32}}"
code_root="${source_snapshot}/methods/motive"

require_absolute() {
  local label="$1"
  local value="$2"
  if [[ "${value}" != /* || "${value}" == "/" ]]; then
    echo "[goku-anchor-prefilter] ${label} must be a non-root absolute path: ${value}" >&2
    exit 2
  fi
}

require_positive_integer() {
  local label="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[goku-anchor-prefilter] ${label} must be a positive integer: ${value}" >&2
    exit 2
  fi
}

for binding in \
  "source_snapshot:${source_snapshot}" \
  "input_fused:${input_fused}" \
  "video_root:${video_root}" \
  "output_dir:${output_dir}" \
  "python_bin:${python_bin}"; do
  require_absolute "${binding%%:*}" "${binding#*:}"
done
require_positive_integer "MOTIVE_GOKU_ACTION_PREFILTER_SIZE" "${sample_size}"
require_positive_integer "MOTIVE_GOKU_ACTION_PREFILTER_WORKERS" "${workers}"

for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${code_root}/motive/goku_action_anchor_prefilter.py" \
  "${input_fused}" \
  "${video_root}" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[goku-anchor-prefilter] missing required path: ${required}" >&2
    exit 2
  fi
done
if [[ ! -x "${python_bin}" ]]; then
  echo "[goku-anchor-prefilter] Python is not executable: ${python_bin}" >&2
  exit 2
fi
if [[ -e "${output_dir}" || -L "${output_dir}" ]]; then
  echo "[goku-anchor-prefilter] refusing existing output: ${output_dir}" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${code_root}:${source_snapshot}"
export TOKENIZERS_PARALLELISM=false
cd "${code_root}"

echo "[goku-anchor-prefilter] host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[goku-anchor-prefilter] input=${input_fused} output=${output_dir}"

"${python_bin}" -m motive.goku_action_anchor_prefilter \
  --input-fused "${input_fused}" \
  --video-root "${video_root}" \
  --output-dir "${output_dir}" \
  --sample-size "${sample_size}" \
  --workers "${workers}"

if [[ -L "${output_dir}/selected.jsonl" || ! -s "${output_dir}/selected.jsonl" ]]; then
  echo "[goku-anchor-prefilter] selected.jsonl was not committed" >&2
  exit 2
fi

sha256sum "${output_dir}/selected.jsonl"
echo "[goku-anchor-prefilter] complete"
