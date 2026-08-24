#!/usr/bin/env bash

# Submit a disconnect-safe generation chain:
#   action-anchor curation
#     -> one source-length / 4-step technical smoke
#     -> one 81-frame / 720p / 1-step full-geometry smoke
#     -> eight serial full-quality chunks, at most 16 new samples each.
#
# Every full chunk shares one output root.  The Python runner validates and
# skips atomically committed samples, so a requeued allocation or a later
# chunk never regenerates a valid target.

set -Eeuo pipefail
umask 077

curation_job_id="${MOTIVE_GOKU_ACTION_CURATION_JOB_ID:?set MOTIVE_GOKU_ACTION_CURATION_JOB_ID}"
curation_dependency_mode="${MOTIVE_WAN22_CURATION_DEPENDENCY_MODE:-afterok}"
source_snapshot="${MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT:?set MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT}"
generation_manifest="${MOTIVE_GOKU_ACTION_GENERATION_MANIFEST:?set MOTIVE_GOKU_ACTION_GENERATION_MANIFEST}"
signed_release="${MOTIVE_WAN22_SIGNED_RELEASE:?set MOTIVE_WAN22_SIGNED_RELEASE}"
wan_code_root="${MOTIVE_WAN22_CODE_ROOT:?set MOTIVE_WAN22_CODE_ROOT}"
checkpoint_dir="${MOTIVE_WAN22_CKPT_DIR:?set MOTIVE_WAN22_CKPT_DIR}"
python_bin="${MOTIVE_WAN22_PYTHON_BIN:?set MOTIVE_WAN22_PYTHON_BIN}"
ffprobe_bin="${MOTIVE_WAN22_FFPROBE_BIN:?set MOTIVE_WAN22_FFPROBE_BIN}"
run_root="${MOTIVE_WAN22_RUN_ROOT:?set MOTIVE_WAN22_RUN_ROOT}"
allow_pending_review="${MOTIVE_WAN22_ALLOW_PENDING_REVIEW:?set explicitly to 0}"
data_root="${MOTIVE_WAN22_DATA_ROOT:-}"
chunk_count="${MOTIVE_WAN22_CHUNK_COUNT:-8}"
chunk_size="${MOTIVE_WAN22_CHUNK_SIZE:-16}"

smoke_script="${source_snapshot}/methods/motive/scripts/auh_wan22_i2v_smoke.sbatch"
full_script="${source_snapshot}/methods/motive/scripts/auh_wan22_i2v_full.sbatch"
smoke_output="${run_root}/smoke"
full_geometry_smoke_output="${run_root}/full_geometry_smoke"
full_output="${run_root}/full"
log_dir="${run_root}/logs"
jobs_file="${run_root}/jobs.tsv"

fail() {
  echo "[wan22-chain-submit] $*" >&2
  exit 2
}

require_absolute() {
  local label="$1"
  local value="$2"
  if [[ "${value}" != /* || "${value}" == "/" ]]; then
    fail "${label} must be a non-root absolute path: ${value}"
  fi
}

require_positive_integer() {
  local label="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    fail "${label} must be a positive integer: ${value}"
  fi
}

for binding in \
  "source_snapshot:${source_snapshot}" \
  "generation_manifest:${generation_manifest}" \
  "signed_release:${signed_release}" \
  "wan_code_root:${wan_code_root}" \
  "checkpoint_dir:${checkpoint_dir}" \
  "python_bin:${python_bin}" \
  "ffprobe_bin:${ffprobe_bin}" \
  "run_root:${run_root}"; do
  require_absolute "${binding%%:*}" "${binding#*:}"
done
if [[ -n "${data_root}" ]]; then
  require_absolute "data_root" "${data_root}"
fi
require_positive_integer "MOTIVE_GOKU_ACTION_CURATION_JOB_ID" "${curation_job_id}"
require_positive_integer "MOTIVE_WAN22_CHUNK_COUNT" "${chunk_count}"
require_positive_integer "MOTIVE_WAN22_CHUNK_SIZE" "${chunk_size}"
if [[ "${curation_dependency_mode}" != "afterok" && "${curation_dependency_mode}" != "published" ]]; then
  fail "MOTIVE_WAN22_CURATION_DEPENDENCY_MODE must be afterok or published"
fi
if [[ "${allow_pending_review}" != "0" ]]; then
  fail "MOTIVE_WAN22_ALLOW_PENDING_REVIEW must be 0; only the signed exact-eight-row release authorizes generation"
fi
for required in \
  "${smoke_script}" \
  "${full_script}" \
  "${signed_release}" \
  "${wan_code_root}/generate.py" \
  "${checkpoint_dir}/Wan2.1_VAE.pth" \
  "${ffprobe_bin}" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    fail "missing required path: ${required}"
  fi
done
if [[ ! -x "${python_bin}" ]]; then
  fail "Python is not executable: ${python_bin}"
fi
if [[ -L "${ffprobe_bin}" || ! -f "${ffprobe_bin}" || ! -x "${ffprobe_bin}" ]]; then
  fail "ffprobe must be an executable regular non-symlink file: ${ffprobe_bin}"
fi
# A published partial curation can legitimately contain fewer than 128 rows.
# When it already exists at submission time, fail before allocating GPUs if
# the requested chunk capacity cannot cover every row.
if [[ -f "${generation_manifest}" && ! -L "${generation_manifest}" ]]; then
  manifest_rows="$(
    awk 'NF { rows += 1 } END { print rows + 0 }' "${generation_manifest}"
  )"
  if (( manifest_rows <= 0 )); then
    fail "generation manifest is empty: ${generation_manifest}"
  fi
  if (( manifest_rows != 8 )); then
    fail "signed release generation manifest must contain exactly 8 rows"
  fi
  if (( chunk_count * chunk_size < manifest_rows )); then
    fail "full chunk capacity is too small: rows=${manifest_rows} capacity=$(( chunk_count * chunk_size ))"
  fi
fi
if [[ "${curation_dependency_mode}" == "published" ]]; then
  if [[ -L "${generation_manifest}" || ! -s "${generation_manifest}" ]]; then
    fail "published generation manifest must be a non-empty regular non-symlink file"
  fi
  curation_done="${generation_manifest%/*}/done.json"
  if [[ -L "${curation_done}" || ! -s "${curation_done}" ]]; then
    fail "published curation done receipt is missing: ${curation_done}"
  fi
  expected_manifest_sha="$(
    "${python_bin}" -c \
      'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); assert p.get("status") == "complete"; print(p["output_sha256"]["generation_manifest.jsonl"])' \
      "${curation_done}"
  )"
  actual_manifest_sha="$(sha256sum "${generation_manifest}" | cut -d' ' -f1)"
  if [[ ! "${expected_manifest_sha}" =~ ^[0-9a-f]{64}$ || "${actual_manifest_sha}" != "${expected_manifest_sha}" ]]; then
    fail "published generation manifest SHA-256 differs from curation done receipt"
  fi
fi
# The manifest and even its immediate parent are intentionally allowed to be
# absent at submission time: the curation dependency atomically publishes
# both before the smoke allocation can start.
if [[ -e "${run_root}" || -L "${run_root}" ]]; then
  fail "run root must be fresh: ${run_root}"
fi
run_parent="${run_root%/*}"
if [[ ! -d "${run_parent}" || ! -w "${run_parent}" ]]; then
  fail "run-root parent is unavailable: ${run_parent}"
fi

mkdir "${run_root}"
mkdir "${log_dir}"
printf 'stage\tjob_id\tdependency\toutput_root\n' > "${jobs_file}"

export MOTIVE_WAN22_OUTPUT_ROOT="${smoke_output}"
export MOTIVE_WAN22_MAX_SAMPLES=1
export MOTIVE_WAN22_MAX_NEW_SAMPLES=1
export MOTIVE_WAN22_FRAME_NUM=81
export MOTIVE_WAN22_SAMPLE_STEPS=4
export MOTIVE_WAN22_SAMPLE_SHIFT=3.0
export MOTIVE_WAN22_SIZE='480*832'
smoke_submit_args=(
  --parsable
  --export=ALL
  --output="${log_dir}/smoke-%j.out"
  --error="${log_dir}/smoke-%j.err"
)
if [[ "${curation_dependency_mode}" == "afterok" ]]; then
  smoke_dependency="afterok:${curation_job_id}"
  smoke_submit_args+=(
    --kill-on-invalid-dep=yes
    --dependency="${smoke_dependency}"
  )
else
  smoke_dependency="published:${curation_job_id}"
fi
smoke_submission="$(sbatch "${smoke_submit_args[@]}" "${smoke_script}")"
smoke_job="${smoke_submission%%;*}"
require_positive_integer "smoke job ID" "${smoke_job}"
printf 'smoke\t%s\t%s\t%s\n' \
  "${smoke_job}" "${smoke_dependency}" "${smoke_output}" >> "${jobs_file}"

export MOTIVE_WAN22_OUTPUT_ROOT="${full_geometry_smoke_output}"
export MOTIVE_WAN22_MAX_SAMPLES=1
export MOTIVE_WAN22_MAX_NEW_SAMPLES=1
export MOTIVE_WAN22_FRAME_NUM=81
export MOTIVE_WAN22_SAMPLE_STEPS=1
export MOTIVE_WAN22_SAMPLE_SHIFT=5.0
export MOTIVE_WAN22_SIZE='1280*720'
full_geometry_smoke_submission="$(
  sbatch \
    --parsable \
    --kill-on-invalid-dep=yes \
    --dependency="afterok:${smoke_job}" \
    --export=ALL \
    --output="${log_dir}/full-geometry-smoke-%j.out" \
    --error="${log_dir}/full-geometry-smoke-%j.err" \
    "${smoke_script}"
)"
full_geometry_smoke_job="${full_geometry_smoke_submission%%;*}"
require_positive_integer \
  "full-geometry smoke job ID" \
  "${full_geometry_smoke_job}"
printf 'full_geometry_smoke\t%s\tafterok:%s\t%s\n' \
  "${full_geometry_smoke_job}" \
  "${smoke_job}" \
  "${full_geometry_smoke_output}" >> "${jobs_file}"

export MOTIVE_WAN22_OUTPUT_ROOT="${full_output}"
export MOTIVE_WAN22_MAX_NEW_SAMPLES="${chunk_size}"
export MOTIVE_WAN22_FRAME_NUM=81
export MOTIVE_WAN22_SAMPLE_STEPS=40
export MOTIVE_WAN22_SAMPLE_SHIFT=5.0
export MOTIVE_WAN22_SIZE='1280*720'
unset MOTIVE_WAN22_MAX_SAMPLES
previous_job="${full_geometry_smoke_job}"
for (( chunk_index=0; chunk_index<chunk_count; chunk_index++ )); do
  submission="$(
    sbatch \
      --parsable \
      --kill-on-invalid-dep=yes \
      --dependency="afterok:${previous_job}" \
      --export=ALL \
      --output="${log_dir}/full-chunk-${chunk_index}-%j.out" \
      --error="${log_dir}/full-chunk-${chunk_index}-%j.err" \
      "${full_script}"
  )"
  job_id="${submission%%;*}"
  require_positive_integer "full chunk ${chunk_index} job ID" "${job_id}"
  printf 'full_chunk_%02d\t%s\tafterok:%s\t%s\n' \
    "${chunk_index}" "${job_id}" "${previous_job}" "${full_output}" \
    >> "${jobs_file}"
  previous_job="${job_id}"
done

chmod 0600 "${jobs_file}"
printf 'curation_job_id=%s\n' "${curation_job_id}"
printf 'curation_dependency_mode=%s\n' "${curation_dependency_mode}"
printf 'smoke_job_id=%s\n' "${smoke_job}"
printf 'full_geometry_smoke_job_id=%s\n' "${full_geometry_smoke_job}"
printf 'last_full_chunk_job_id=%s\n' "${previous_job}"
printf 'smoke_output=%s\n' "${smoke_output}"
printf 'full_geometry_smoke_output=%s\n' "${full_geometry_smoke_output}"
printf 'full_output=%s\n' "${full_output}"
printf 'jobs_file=%s\n' "${jobs_file}"
