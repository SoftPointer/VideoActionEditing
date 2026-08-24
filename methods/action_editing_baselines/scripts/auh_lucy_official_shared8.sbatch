#!/usr/bin/env bash
#SBATCH --job-name=a8-lucy-base
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=12:00:00
#SBATCH --array=0-7%8
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

lucy_checkpoint="${ACTION_BASELINE_LUCY_CHECKPOINT:?set ACTION_BASELINE_LUCY_CHECKPOINT}"
lucy_checkpoint_tree_sha256="${ACTION_BASELINE_LUCY_CHECKPOINT_TREE_SHA256:?set ACTION_BASELINE_LUCY_CHECKPOINT_TREE_SHA256}"
sample_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
[[ "${sample_index}" =~ ^[0-7]$ ]] || shared8_fail "array index must be in 0..7"
shared8_require_directory "${lucy_checkpoint}" "Lucy checkpoint"
shared8_require_sha256 "${lucy_checkpoint_tree_sha256}" "Lucy checkpoint-tree hash"
shared8_init_common "lucy"
shared8_configure_worker_cache "${SHARED8_SCRATCH_ROOT}/worker-${sample_index}"

worker_log="${SHARED8_LAUNCHER_LOG_ROOT}/lucy-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}-${SLURM_JOB_ID}-sample-${sample_index}.log"
[[ ! -e "${worker_log}" && ! -L "${worker_log}" ]] || shared8_fail "worker log already exists: ${worker_log}"

"${SHARED8_PYTHON_BIN}" "${SHARED8_RUNNER}" \
  --model-id lucy_official_base \
  --manifest "${SHARED8_MANIFEST}" \
  --manifest-sha256 "${SHARED8_MANIFEST_SHA256}" \
  --index "${sample_index}" \
  --output-root "${SHARED8_OUTPUT_ROOT}" \
  --python-bin "${SHARED8_PYTHON_BIN}" \
  --code-root "${SHARED8_SOURCE_ROOT}" \
  --source-revision "${SHARED8_SOURCE_REVISION}" \
  --source-archive-sha256 "${SHARED8_SOURCE_ARCHIVE_SHA256}" \
  --ffprobe "${SHARED8_FFPROBE_BIN}" \
  --lucy-checkpoint "${lucy_checkpoint}" \
  --lucy-checkpoint-tree-sha256 "${lucy_checkpoint_tree_sha256}" \
  >"${worker_log}" 2>&1

printf '[shared8-lucy] PASS sample=%s log=%s\n' "${sample_index}" "${worker_log}"
