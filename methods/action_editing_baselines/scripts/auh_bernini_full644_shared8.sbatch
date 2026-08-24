#!/usr/bin/env bash
#SBATCH --job-name=a8-bern644
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=12:00:00
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

bernini_root="${ACTION_BASELINE_BERNINI_ROOT:?set ACTION_BASELINE_BERNINI_ROOT}"
veomni_root="${ACTION_BASELINE_VEOMNI_ROOT:?set ACTION_BASELINE_VEOMNI_ROOT}"
bernini_checkpoint="${ACTION_BASELINE_BERNINI_CHECKPOINT:?set ACTION_BASELINE_BERNINI_CHECKPOINT}"
bernini_adapter="${ACTION_BASELINE_BERNINI_ADAPTER:?set ACTION_BASELINE_BERNINI_ADAPTER}"
bernini_inference_archive="${ACTION_BASELINE_BERNINI_INFERENCE_ARCHIVE:?set ACTION_BASELINE_BERNINI_INFERENCE_ARCHIVE}"
shared8_require_directory "${bernini_root}" "Bernini official source"
shared8_require_directory "${veomni_root}" "VeOmni source"
shared8_require_directory "${bernini_checkpoint}" "Bernini checkpoint"
shared8_require_directory "${bernini_adapter}" "Bernini adapter"
shared8_require_plain_file "${bernini_inference_archive}" "Bernini frozen inference archive"
shared8_init_common "bernini"
shared8_configure_worker_cache "${SHARED8_SCRATCH_ROOT}/worker-${sample_index}"
export MODELING_BACKEND=hf

worker_log="${SHARED8_LAUNCHER_LOG_ROOT}/bernini-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}-${SLURM_JOB_ID}-sample-${sample_index}.log"
[[ ! -e "${worker_log}" && ! -L "${worker_log}" ]] || shared8_fail "worker log already exists: ${worker_log}"

"${SHARED8_PYTHON_BIN}" "${SHARED8_RUNNER}" \
  --model-id bernini_full644_lora_step644 \
  --manifest "${SHARED8_MANIFEST}" \
  --manifest-sha256 "${SHARED8_MANIFEST_SHA256}" \
  --index "${sample_index}" \
  --output-root "${SHARED8_OUTPUT_ROOT}" \
  --python-bin "${SHARED8_PYTHON_BIN}" \
  --code-root "${SHARED8_SOURCE_ROOT}" \
  --source-revision "${SHARED8_SOURCE_REVISION}" \
  --source-archive-sha256 "${SHARED8_SOURCE_ARCHIVE_SHA256}" \
  --ffprobe "${SHARED8_FFPROBE_BIN}" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --bernini-checkpoint "${bernini_checkpoint}" \
  --bernini-adapter "${bernini_adapter}" \
  --bernini-inference-archive "${bernini_inference_archive}" \
  >"${worker_log}" 2>&1

printf '[shared8-bernini] PASS sample=%s log=%s\n' "${sample_index}" "${worker_log}"
