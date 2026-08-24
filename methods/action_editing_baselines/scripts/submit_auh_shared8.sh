#!/usr/bin/env bash

# Submit one baseline or the complete shared-8 suite. OmniVideo2 is always
# staged as sample-0 canary followed by a dependent sample-1..7 array.

set -Eeuo pipefail

fail() {
  printf '[shared8-submit] ERROR: %s\n' "$*" >&2
  exit 2
}

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "set ${name}"
}

require_common_env() {
  local name
  for name in \
    ACTION_BASELINE_SOURCE_ARCHIVE \
    ACTION_BASELINE_SOURCE_ARCHIVE_SHA256 \
    ACTION_BASELINE_SOURCE_REVISION \
    ACTION_BASELINE_MANIFEST \
    ACTION_BASELINE_MANIFEST_SHA256 \
    ACTION_BASELINE_OUTPUT_ROOT \
    ACTION_BASELINE_PYTHON_BIN; do
    require_env "${name}"
  done
  [[ "${ACTION_BASELINE_SOURCE_ARCHIVE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid source archive hash"
  [[ "${ACTION_BASELINE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid manifest hash"
  [[ "${ACTION_BASELINE_SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
  [[ -f "${ACTION_BASELINE_SOURCE_ARCHIVE}" && ! -L "${ACTION_BASELINE_SOURCE_ARCHIVE}" ]] || fail "invalid source archive"
  [[ -f "${ACTION_BASELINE_MANIFEST}" && ! -L "${ACTION_BASELINE_MANIFEST}" ]] || fail "invalid input manifest"
  [[ -x "${ACTION_BASELINE_PYTHON_BIN}" ]] || fail "Python is not executable"
  [[ "${ACTION_BASELINE_OUTPUT_ROOT}" == /* ]] || fail "output root must be absolute"
}

require_lucy_env() {
  require_env ACTION_BASELINE_LUCY_CHECKPOINT
  require_env ACTION_BASELINE_LUCY_CHECKPOINT_TREE_SHA256
  [[ -d "${ACTION_BASELINE_LUCY_CHECKPOINT}" ]] || fail "invalid Lucy checkpoint"
  [[ "${ACTION_BASELINE_LUCY_CHECKPOINT_TREE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid Lucy checkpoint-tree hash"
}

require_bernini_env() {
  local name
  for name in \
    ACTION_BASELINE_BERNINI_ROOT \
    ACTION_BASELINE_VEOMNI_ROOT \
    ACTION_BASELINE_BERNINI_CHECKPOINT \
    ACTION_BASELINE_BERNINI_ADAPTER; do
    require_env "${name}"
    [[ -d "${!name}" ]] || fail "invalid directory in ${name}"
  done
  require_env ACTION_BASELINE_BERNINI_INFERENCE_ARCHIVE
  [[ -f "${ACTION_BASELINE_BERNINI_INFERENCE_ARCHIVE}" && ! -L "${ACTION_BASELINE_BERNINI_INFERENCE_ARCHIVE}" ]] || fail "invalid Bernini inference archive"
}

require_omni_env() {
  local name
  for name in \
    ACTION_BASELINE_OMNI_ROOT \
    ACTION_BASELINE_OMNI_CHECKPOINT \
    ACTION_BASELINE_QWEN_CHECKPOINT; do
    require_env "${name}"
    [[ -d "${!name}" ]] || fail "invalid directory in ${name}"
  done
}

submit_job() {
  local label="$1"
  shift
  local raw_job_id
  raw_job_id="$(sbatch --parsable --export=ALL "$@")"
  SUBMITTED_JOB_ID="${raw_job_id%%;*}"
  [[ "${SUBMITTED_JOB_ID}" =~ ^[0-9]+$ ]] || fail "sbatch returned an invalid job ID for ${label}: ${raw_job_id}"
  printf '[shared8-submit] %s job_id=%s\n' "${label}" "${SUBMITTED_JOB_ID}"
}

submit_lucy() {
  require_lucy_env
  submit_job lucy-canary --array=0 "${script_dir}/auh_lucy_official_shared8.sbatch"
  local canary_job_id="${SUBMITTED_JOB_ID}"
  submit_job lucy-rest --array=1-7%8 --dependency="afterok:${canary_job_id}" \
    "${script_dir}/auh_lucy_official_shared8.sbatch"
}

submit_bernini() {
  require_bernini_env
  submit_job bernini "${script_dir}/auh_bernini_full644_shared8.sbatch"
}

submit_omni() {
  require_omni_env
  submit_job omni2-canary --array=0 "${script_dir}/auh_omnivideo2_official_shared8.sbatch"
  local canary_job_id="${SUBMITTED_JOB_ID}"
  submit_job omni2-rest --array=1-7%2 --dependency="afterok:${canary_job_id}" "${script_dir}/auh_omnivideo2_official_shared8.sbatch"
}

command -v sbatch >/dev/null 2>&1 || fail "sbatch is not available"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ACTION_BASELINE_LAUNCHER_ROOT="$(cd -- "${script_dir}/../../.." && pwd -P)"
export ACTION_BASELINE_LAUNCHER_ROOT
mode="${1:-all}"
[[ "$#" -le 1 ]] || fail "usage: $0 [lucy|bernini|omni|all]"
case "${mode}" in
  lucy|bernini|omni|all) ;;
  *) fail "usage: $0 [lucy|bernini|omni|all]" ;;
esac

require_common_env
case "${mode}" in
  lucy)
    submit_lucy
    ;;
  bernini)
    submit_bernini
    ;;
  omni)
    submit_omni
    ;;
  all)
    submit_lucy
    submit_bernini
    submit_omni
    ;;
esac
