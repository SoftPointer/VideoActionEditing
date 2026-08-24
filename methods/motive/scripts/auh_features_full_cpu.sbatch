#!/usr/bin/env bash
#SBATCH --job-name=motive-features-full
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
# AUH faculty QOS enforces MinTRES=gres/gpu:1 for every job.
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=2-00:00:00

set -Eeuo pipefail

repo_root="${MOTIVE_REPO_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit}"
data_root="${MOTIVE_DATA_ROOT:-${repo_root}/data/goku/subject_movement/extracted}"
output_root="${MOTIVE_OUTPUT_ROOT:-${repo_root}/methods/motive/outputs/goku_subject_movement_full}"
code_root="${MOTIVE_CODE_ROOT:-${repo_root}/methods/motive}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
input_manifest="${MOTIVE_FEATURE_INPUT:-${output_root}/rules/candidates.jsonl}"
worker_count="${MOTIVE_FEATURE_WORKERS:-${SLURM_CPUS_PER_TASK:-32}}"
internal_shard_size="${MOTIVE_FEATURE_SHARD_SIZE:-256}"

for required in \
  "${code_root}/motive/cascade.py" \
  "${data_root}" \
  "${input_manifest}" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[motive-features] missing required path: ${required}" >&2
    exit 2
  fi
done

make_modulo_shard() {
  local source_path="$1"
  local destination_path="$2"
  local shard_index="$3"
  local shard_count="$4"
  local marker_path="${destination_path}.source"
  local source_digest
  local expected_marker
  local temporary_path
  local temporary_marker

  source_digest="$(sha256sum "${source_path}" | awk '{print $1}')"
  expected_marker="${source_digest}|${shard_count}|${shard_index}"
  if [[ -s "${destination_path}" && -s "${marker_path}" ]]; then
    if [[ "$(<"${marker_path}")" != "${expected_marker}" ]]; then
      echo "[motive-features] existing shard provenance changed: ${destination_path}" >&2
      exit 2
    fi
    return
  fi
  if [[ -e "${destination_path}" || -e "${marker_path}" ]]; then
    echo "[motive-features] incomplete shard input exists: ${destination_path}" >&2
    exit 2
  fi

  mkdir -p "$(dirname "${destination_path}")"
  temporary_path="${destination_path}.tmp.${SLURM_JOB_ID:-manual}.${shard_index}"
  temporary_marker="${marker_path}.tmp.${SLURM_JOB_ID:-manual}.${shard_index}"
  # GNU awk reserves `index` as a builtin function name.
  awk -v shard="${shard_index}" -v count="${shard_count}" \
    '((NR - 1) % count) == shard' "${source_path}" > "${temporary_path}"
  printf '%s\n' "${expected_marker}" > "${temporary_marker}"
  mv "${temporary_path}" "${destination_path}"
  mv "${temporary_marker}" "${marker_path}"
}

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  task_index="${SLURM_ARRAY_TASK_ID}"
  task_count="${MOTIVE_FEATURE_ARRAY_COUNT:-${SLURM_ARRAY_TASK_COUNT:-8}}"
  if ! [[ "${task_index}" =~ ^[0-9]+$ && "${task_count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[motive-features] invalid array index/count: ${task_index}/${task_count}" >&2
    exit 2
  fi
  if (( task_index >= task_count )); then
    echo "[motive-features] array index is outside count: ${task_index}/${task_count}" >&2
    exit 2
  fi
  task_tag="$(printf '%03d' "${task_index}")"
  task_root="${output_root}/features_array/tasks/task-${task_tag}"
  task_input="${output_root}/features_array/manifests/task-${task_tag}.jsonl"
  make_modulo_shard "${input_manifest}" "${task_input}" "${task_index}" "${task_count}"
else
  task_index=0
  task_count=1
  task_root="${output_root}/features_single"
  task_input="${input_manifest}"
fi

stage_output="${task_root}/stage"
export_output="${task_root}/export"
mkdir -p "${task_root}"
cd "${repo_root}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${code_root}:${PYTHONPATH:-}"

echo "[motive-features] task=${task_index}/${task_count} input=${task_input}"
echo "[motive-features] stage_output=${stage_output} workers=${worker_count}"
feature_error_args=()
if [[ "${MOTIVE_FEATURE_ALLOW_ERRORS:-0}" == "1" ]]; then
  feature_error_args+=(--allow-errors)
fi
"${python_bin}" -m motive.cascade features \
  --input "${task_input}" \
  --root "${data_root}" \
  --output-dir "${stage_output}" \
  --workers "${worker_count}" \
  --shard-size "${internal_shard_size}" \
  --resume \
  --analysis-frames "${MOTIVE_ANALYSIS_FRAMES:-20}" \
  --resize-width "${MOTIVE_RESIZE_WIDTH:-160}" \
  --active-speed-threshold "${MOTIVE_ACTIVE_SPEED_THRESHOLD:-0.005}" \
  --static-residual-p90 "${MOTIVE_STATIC_RESIDUAL_P90:-0.003}" \
  --static-active-fraction "${MOTIVE_STATIC_ACTIVE_FRACTION:-0.025}" \
  --max-scene-cuts "${MOTIVE_MAX_SCENE_CUTS:-0}" \
  "${feature_error_args[@]}"

"${python_bin}" -m motive.cascade export \
  --feature-dir "${stage_output}/features" \
  --output-dir "${export_output}" \
  --keep-decisions auto_keep review \
  --qwen-budget "${MOTIVE_QWEN_BUDGET:-8000}" \
  --qwen-decisions auto_keep review \
  --qwen-auto-keep-fraction "${MOTIVE_QWEN_AUTO_KEEP_FRACTION:-0.20}"

echo "[motive-features] completed task=${task_index}/${task_count} export=${export_output}"
