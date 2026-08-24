#!/usr/bin/env bash

# Submit only the strict finalizer for one already-complete eight-shard Qwen
# audit.  The returned Slurm job ID is durably recorded and is the dependency
# passed to the existing Wan2.2 generation-chain submitter.

set -Eeuo pipefail
umask 077

source_snapshot="${MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT:?set MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT}"
selected="${MOTIVE_GOKU_ACTION_SELECTED:?set MOTIVE_GOKU_ACTION_SELECTED}"
qwen_root="${MOTIVE_GOKU_ACTION_QWEN_OUTPUT:?set MOTIVE_GOKU_ACTION_QWEN_OUTPUT}"
run_root="${MOTIVE_GOKU_ACTION_RECOVERY_RUN_ROOT:?set MOTIVE_GOKU_ACTION_RECOVERY_RUN_ROOT}"
python_bin="${MOTIVE_GOKU_ACTION_PYTHON_BIN:?set MOTIVE_GOKU_ACTION_PYTHON_BIN}"
approval_path="${MOTIVE_GOKU_ACTION_APPROVAL_PATH:?set the proposal-bound human approval JSON}"
final_seed="${MOTIVE_GOKU_ACTION_FINAL_SEED:-260730}"
allow_partial="${MOTIVE_GOKU_ACTION_ALLOW_PARTIAL:-0}"

finalize_script="${source_snapshot}/methods/motive/scripts/auh_goku_action_anchor_finalize.sbatch"
wan_submit_script="${source_snapshot}/methods/motive/scripts/auh_submit_wan22_i2v_chain.sh"
final_output="${run_root}/final"
generation_manifest="${final_output}/generation_manifest.jsonl"
log_dir="${run_root}/logs"
jobs_file="${run_root}/jobs.tsv"
submission_receipt="${run_root}/finalize_submission.raw"
retry_script="${run_root}/retry_finalize.sh"
wan_helper="${run_root}/submit_wan_chain.sh"
shard_count=8

fail() {
  echo "[goku-anchor-finalize-submit] $*" >&2
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
      fail "exported ${name} is forbidden; resources are fixed in the sbatch file"
    fi
  done < <(env -0)
}

for binding in \
  "source_snapshot:${source_snapshot}" \
  "selected:${selected}" \
  "qwen_root:${qwen_root}" \
  "run_root:${run_root}" \
  "python_bin:${python_bin}" \
  "approval_path:${approval_path}"; do
  require_absolute "${binding%%:*}" "${binding#*:}"
done
reject_exported_sbatch_controls
if [[ ! "${final_seed}" =~ ^[0-9]+$ ]]; then
  fail "MOTIVE_GOKU_ACTION_FINAL_SEED must be a non-negative integer"
fi
if [[ "${allow_partial}" != "0" && "${allow_partial}" != "1" ]]; then
  fail "MOTIVE_GOKU_ACTION_ALLOW_PARTIAL must be 0 or 1"
fi

for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${source_snapshot}/methods/motive/motive/goku_action_anchor_qwen.py" \
  "${source_snapshot}/methods/motive/motive/goku_action_anchor_finalize.py" \
  "${finalize_script}" \
  "${wan_submit_script}" \
  "${selected}" \
  "${approval_path}" \
  "${python_bin}"; do
  if [[ -L "${required}" || ! -f "${required}" ]]; then
    fail "missing regular non-symlink input: ${required}"
  fi
done
if [[ ! -s "${selected}" ]]; then
  fail "selected input is empty: ${selected}"
fi
if [[ ! -s "${approval_path}" ]]; then
  fail "proposal-bound approval is empty: ${approval_path}"
fi
if [[ ! -x "${python_bin}" ]]; then
  fail "Python is not executable: ${python_bin}"
fi
if [[ -L "${qwen_root}" || ! -d "${qwen_root}" ]]; then
  fail "Qwen root must be a non-symlink directory: ${qwen_root}"
fi

shopt -s nullglob
actual_shards=("${qwen_root}"/qwen_shard_*.jsonl)
if (( ${#actual_shards[@]} != shard_count )); then
  fail "Qwen root must contain exactly eight qwen_shard_*.jsonl files"
fi
for (( shard_index=0; shard_index<shard_count; shard_index++ )); do
  printf -v shard_name 'qwen_shard_%03d.jsonl' "${shard_index}"
  shard_path="${qwen_root}/${shard_name}"
  if [[ -L "${shard_path}" || ! -f "${shard_path}" ]]; then
    fail "missing regular non-symlink Qwen shard: ${shard_path}"
  fi
done

if [[ -e "${run_root}" || -L "${run_root}" ]]; then
  fail "recovery run root must be fresh: ${run_root}"
fi
run_parent="${run_root%/*}"
if [[ ! -d "${run_parent}" || ! -w "${run_parent}" ]]; then
  fail "recovery run-root parent is unavailable: ${run_parent}"
fi

mkdir "${run_root}"
mkdir "${log_dir}"
printf 'stage\tjob_id\tdependency\toutput_root\n' > "${jobs_file}"

export MOTIVE_GOKU_ACTION_FINAL_OUTPUT="${final_output}"
export MOTIVE_GOKU_ACTION_FINAL_SEED="${final_seed}"
export MOTIVE_GOKU_ACTION_ALLOW_PARTIAL="${allow_partial}"
export MOTIVE_GOKU_ACTION_APPROVAL_PATH="${approval_path}"

# This helper is intentionally reusable if the short finalizer allocation is
# preempted before its atomic output directory is published.
{
  printf '%s\n' '#!/usr/bin/env bash'
  printf '%s\n' 'set -Eeuo pipefail'
  printf 'export MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT=%q\n' "${source_snapshot}"
  printf 'export MOTIVE_GOKU_ACTION_SELECTED=%q\n' "${selected}"
  printf 'export MOTIVE_GOKU_ACTION_QWEN_OUTPUT=%q\n' "${qwen_root}"
  printf 'export MOTIVE_GOKU_ACTION_FINAL_OUTPUT=%q\n' "${final_output}"
  printf 'export MOTIVE_GOKU_ACTION_PYTHON_BIN=%q\n' "${python_bin}"
  printf 'export MOTIVE_GOKU_ACTION_FINAL_SEED=%q\n' "${final_seed}"
  printf 'export MOTIVE_GOKU_ACTION_ALLOW_PARTIAL=%q\n' "${allow_partial}"
  printf 'export MOTIVE_GOKU_ACTION_APPROVAL_PATH=%q\n' "${approval_path}"
  printf 'exec sbatch --parsable --export=ALL --output=%q --error=%q %q\n' \
    "${log_dir}/finalize-retry-%j.out" \
    "${log_dir}/finalize-retry-%j.err" \
    "${finalize_script}"
} > "${retry_script}"
chmod 0700 "${retry_script}"

sbatch \
  --parsable \
  --export=ALL \
  --output="${log_dir}/finalize-%j.out" \
  --error="${log_dir}/finalize-%j.err" \
  "${finalize_script}" > "${submission_receipt}"

IFS= read -r finalize_submission < "${submission_receipt}" || true
finalize_job="${finalize_submission%%;*}"
if [[ ! "${finalize_job}" =~ ^[1-9][0-9]*$ ]]; then
  fail "could not parse standalone finalizer job ID: ${finalize_submission}"
fi
printf 'standalone_finalize\t%s\tnone\t%s\n' \
  "${finalize_job}" "${final_output}" >> "${jobs_file}"
chmod 0600 "${jobs_file}" "${submission_receipt}"

# This helper performs no generation now.  When invoked with the remaining
# MOTIVE_WAN22_* bindings, the existing submitter creates a fresh Wan run root
# and submits every stage afterok:<standalone-finalizer-job>.
{
  printf '%s\n' '#!/usr/bin/env bash'
  printf '%s\n' 'set -Eeuo pipefail'
  printf 'export MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT=%q\n' "${source_snapshot}"
  printf 'export MOTIVE_GOKU_ACTION_CURATION_JOB_ID=%q\n' "${finalize_job}"
  printf 'export MOTIVE_GOKU_ACTION_GENERATION_MANIFEST=%q\n' \
    "${generation_manifest}"
  printf 'exec bash %q\n' "${wan_submit_script}"
} > "${wan_helper}"
chmod 0700 "${wan_helper}"

printf 'standalone_finalize_job_id=%s\n' "${finalize_job}"
printf 'final_output=%s\n' "${final_output}"
printf 'generation_manifest=%s\n' "${generation_manifest}"
printf 'jobs_file=%s\n' "${jobs_file}"
printf 'submission_receipt=%s\n' "${submission_receipt}"
printf 'retry_finalize_script=%s\n' "${retry_script}"
printf 'submit_wan_chain_script=%s\n' "${wan_helper}"
