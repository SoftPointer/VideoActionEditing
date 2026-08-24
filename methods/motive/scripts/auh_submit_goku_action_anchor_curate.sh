#!/usr/bin/env bash

# Submit one disconnect-safe dependency chain:
#   CPU video prefilter -> packed 8-GPU Qwen audit -> in-allocation finalizer.

set -Eeuo pipefail
umask 077

source_snapshot="${MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT:?set MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT}"
input_fused="${MOTIVE_GOKU_ACTION_INPUT_FUSED:?set MOTIVE_GOKU_ACTION_INPUT_FUSED}"
video_root="${MOTIVE_GOKU_ACTION_VIDEO_ROOT:?set MOTIVE_GOKU_ACTION_VIDEO_ROOT}"
run_root="${MOTIVE_GOKU_ACTION_RUN_ROOT:?set MOTIVE_GOKU_ACTION_RUN_ROOT}"
qwen_model="${MOTIVE_GOKU_ACTION_QWEN_MODEL:?set MOTIVE_GOKU_ACTION_QWEN_MODEL}"
python_bin="${MOTIVE_GOKU_ACTION_PYTHON_BIN:?set MOTIVE_GOKU_ACTION_PYTHON_BIN}"
qwen_nframes="${MOTIVE_GOKU_ACTION_QWEN_NFRAMES:-12}"
qwen_max_pixels="${MOTIVE_GOKU_ACTION_QWEN_MAX_PIXELS:-589824}"
final_seed="${MOTIVE_GOKU_ACTION_FINAL_SEED:-260730}"
allow_partial="${MOTIVE_GOKU_ACTION_ALLOW_PARTIAL:-0}"

prefilter_script="${source_snapshot}/methods/motive/scripts/auh_goku_action_anchor_prefilter.sbatch"
qwen_script="${source_snapshot}/methods/motive/scripts/auh_goku_action_anchor_qwen.sbatch"
prefilter_output="${run_root}/prefilter"
qwen_output="${run_root}/qwen8"
final_output="${run_root}/final"
log_dir="${run_root}/logs"
retry_script="${run_root}/retry_qwen.sh"

fail() {
  echo "[goku-anchor-submit] $*" >&2
  exit 2
}

require_absolute() {
  local label="$1"
  local value="$2"
  if [[ "${value}" != /* || "${value}" == "/" ]]; then
    fail "${label} must be a non-root absolute path: ${value}"
  fi
}

reject_exported_sbatch_controls() {
  local entry=""
  local name=""
  while IFS= read -r -d '' entry; do
    name="${entry%%=*}"
    if [[ "${name}" == SBATCH_* ]]; then
      fail "exported ${name} is forbidden; resources are fixed in sbatch files"
    fi
  done < <(env -0)
}

for binding in \
  "source_snapshot:${source_snapshot}" \
  "input_fused:${input_fused}" \
  "video_root:${video_root}" \
  "run_root:${run_root}" \
  "qwen_model:${qwen_model}" \
  "python_bin:${python_bin}"; do
  require_absolute "${binding%%:*}" "${binding#*:}"
done
reject_exported_sbatch_controls

if [[ ! "${qwen_nframes}" =~ ^[1-9][0-9]*$ ]] || (( qwen_nframes < 2 )); then
  fail "MOTIVE_GOKU_ACTION_QWEN_NFRAMES must be an integer of at least two"
fi
if [[ ! "${qwen_max_pixels}" =~ ^[1-9][0-9]*$ ]]; then
  fail "MOTIVE_GOKU_ACTION_QWEN_MAX_PIXELS must be a positive integer"
fi
if [[ ! "${final_seed}" =~ ^[0-9]+$ ]]; then
  fail "MOTIVE_GOKU_ACTION_FINAL_SEED must be a non-negative integer"
fi
if [[ "${allow_partial}" != "0" && "${allow_partial}" != "1" ]]; then
  fail "MOTIVE_GOKU_ACTION_ALLOW_PARTIAL must be 0 or 1"
fi

for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${prefilter_script}" \
  "${qwen_script}" \
  "${input_fused}" \
  "${video_root}" \
  "${qwen_model}/config.json" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    fail "missing required path: ${required}"
  fi
done
if [[ ! -x "${python_bin}" ]]; then
  fail "Python is not executable: ${python_bin}"
fi
if [[ -e "${run_root}" || -L "${run_root}" ]]; then
  fail "run root must be fresh: ${run_root}"
fi
run_parent="${run_root%/*}"
if [[ ! -d "${run_parent}" || ! -w "${run_parent}" ]]; then
  fail "run-root parent is unavailable: ${run_parent}"
fi

mkdir "${run_root}"
mkdir "${log_dir}"

export MOTIVE_GOKU_ACTION_PREFILTER_OUTPUT="${prefilter_output}"
export MOTIVE_GOKU_ACTION_SELECTED="${prefilter_output}/selected.jsonl"
export MOTIVE_GOKU_ACTION_QWEN_OUTPUT="${qwen_output}"
export MOTIVE_GOKU_ACTION_FINAL_OUTPUT="${final_output}"
export MOTIVE_GOKU_ACTION_QWEN_NFRAMES="${qwen_nframes}"
export MOTIVE_GOKU_ACTION_QWEN_MAX_PIXELS="${qwen_max_pixels}"
export MOTIVE_GOKU_ACTION_FINAL_SEED="${final_seed}"
export MOTIVE_GOKU_ACTION_ALLOW_PARTIAL="${allow_partial}"

# This durable helper re-submits only the resumable packed Qwen/finalize
# stage. It remains usable after a login disconnect or a preempted allocation.
{
  printf '%s\n' '#!/usr/bin/env bash'
  printf '%s\n' 'set -Eeuo pipefail'
  printf 'export MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT=%q\n' "${source_snapshot}"
  printf 'export MOTIVE_GOKU_ACTION_SELECTED=%q\n' "${prefilter_output}/selected.jsonl"
  printf 'export MOTIVE_GOKU_ACTION_QWEN_MODEL=%q\n' "${qwen_model}"
  printf 'export MOTIVE_GOKU_ACTION_QWEN_OUTPUT=%q\n' "${qwen_output}"
  printf 'export MOTIVE_GOKU_ACTION_FINAL_OUTPUT=%q\n' "${final_output}"
  printf 'export MOTIVE_GOKU_ACTION_PYTHON_BIN=%q\n' "${python_bin}"
  printf 'export MOTIVE_GOKU_ACTION_QWEN_NFRAMES=%q\n' "${qwen_nframes}"
  printf 'export MOTIVE_GOKU_ACTION_QWEN_MAX_PIXELS=%q\n' "${qwen_max_pixels}"
  printf 'export MOTIVE_GOKU_ACTION_FINAL_SEED=%q\n' "${final_seed}"
  printf 'export MOTIVE_GOKU_ACTION_ALLOW_PARTIAL=%q\n' "${allow_partial}"
  printf 'exec sbatch --parsable --export=ALL --output=%q --error=%q %q\n' \
    "${log_dir}/qwen-finalize-retry-%j.out" \
    "${log_dir}/qwen-finalize-retry-%j.err" \
    "${qwen_script}"
} > "${retry_script}"
chmod 0700 "${retry_script}"

prefilter_submission="$(
  sbatch \
    --parsable \
    --export=ALL \
    --output="${log_dir}/prefilter-%j.out" \
    --error="${log_dir}/prefilter-%j.err" \
    "${prefilter_script}"
)"
prefilter_job="${prefilter_submission%%;*}"
if [[ ! "${prefilter_job}" =~ ^[1-9][0-9]*$ ]]; then
  fail "could not parse prefilter job ID: ${prefilter_submission}"
fi

qwen_submission="$(
  sbatch \
    --parsable \
    --kill-on-invalid-dep=yes \
    --dependency="afterok:${prefilter_job}" \
    --export=ALL \
    --output="${log_dir}/qwen-finalize-%j.out" \
    --error="${log_dir}/qwen-finalize-%j.err" \
    "${qwen_script}"
)"
qwen_job="${qwen_submission%%;*}"
if [[ ! "${qwen_job}" =~ ^[1-9][0-9]*$ ]]; then
  fail "could not parse Qwen job ID: ${qwen_submission}"
fi

printf 'prefilter_job_id=%s\n' "${prefilter_job}"
printf 'qwen_finalize_job_id=%s\n' "${qwen_job}"
printf 'dependency=afterok:%s\n' "${prefilter_job}"
printf 'run_root=%s\n' "${run_root}"
printf 'final_output=%s\n' "${final_output}"
printf 'qwen_retry_script=%s\n' "${retry_script}"
