#!/usr/bin/env bash
#SBATCH --job-name=a8-omni2-base
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=08:00:00
#SBATCH --array=0-7%2
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

launcher_root="${ACTION_BASELINE_LAUNCHER_ROOT:?set ACTION_BASELINE_LAUNCHER_ROOT}"
script_dir="${launcher_root}/methods/action_editing_baselines/scripts"
[[ -d "${script_dir}" && ! -L "${script_dir}" ]] || {
  printf '[shared8-launch] invalid frozen launcher directory: %s\n' "${script_dir}" >&2
  exit 2
}
# shellcheck source=shared8_auh_common.sh
source "${script_dir}/shared8_auh_common.sh"

sample_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
[[ "${sample_index}" =~ ^[0-7]$ ]] || shared8_fail "array index must be in 0..7"

omni_root="${ACTION_BASELINE_OMNI_ROOT:?set ACTION_BASELINE_OMNI_ROOT}"
omni_checkpoint="${ACTION_BASELINE_OMNI_CHECKPOINT:?set ACTION_BASELINE_OMNI_CHECKPOINT}"
qwen_checkpoint="${ACTION_BASELINE_QWEN_CHECKPOINT:?set ACTION_BASELINE_QWEN_CHECKPOINT}"
omni_max_context_len="${ACTION_BASELINE_OMNI_MAX_CONTEXT_LEN:-9216}"
shared8_require_directory "${omni_root}" "OmniVideo2 official source"
shared8_require_directory "${omni_checkpoint}" "OmniVideo2 checkpoint"
shared8_require_directory "${qwen_checkpoint}" "Qwen checkpoint"
[[ "${omni_max_context_len}" == 9216 ]] || shared8_fail "Omni context length must equal 9216"
shared8_init_common "omni2"
shared8_configure_worker_cache "${SHARED8_SCRATCH_ROOT}/worker-${sample_index}"

worker_log="${SHARED8_LAUNCHER_LOG_ROOT}/omni2-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}-${SLURM_JOB_ID}-sample-${sample_index}.log"
[[ ! -e "${worker_log}" && ! -L "${worker_log}" ]] || shared8_fail "worker log already exists: ${worker_log}"

"${SHARED8_PYTHON_BIN}" "${SHARED8_RUNNER}" \
  --model-id omnivideo2_official_base \
  --manifest "${SHARED8_MANIFEST}" \
  --manifest-sha256 "${SHARED8_MANIFEST_SHA256}" \
  --index "${sample_index}" \
  --output-root "${SHARED8_OUTPUT_ROOT}" \
  --python-bin "${SHARED8_PYTHON_BIN}" \
  --code-root "${SHARED8_SOURCE_ROOT}" \
  --source-revision "${SHARED8_SOURCE_REVISION}" \
  --source-archive-sha256 "${SHARED8_SOURCE_ARCHIVE_SHA256}" \
  --ffprobe "${SHARED8_FFPROBE_BIN}" \
  --omni-root "${omni_root}" \
  --omni-checkpoint "${omni_checkpoint}" \
  --qwen-checkpoint "${qwen_checkpoint}" \
  --omni-max-context-len "${omni_max_context_len}" \
  >"${worker_log}" 2>&1

printf '[shared8-omni2] PASS sample=%s log=%s\n' "${sample_index}" "${worker_log}"
