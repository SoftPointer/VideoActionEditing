#!/usr/bin/env bash

# Prepare contiguous manifest shards, submit one independent eight-GPU Wan
# job per shard after a successful full-geometry smoke, and submit one
# deterministic CPU aggregate finalizer after every shard succeeds.

set -Eeuo pipefail
umask 077

source_snapshot="${MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT:?set MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT}"
generation_manifest="${MOTIVE_GOKU_ACTION_GENERATION_MANIFEST:?set MOTIVE_GOKU_ACTION_GENERATION_MANIFEST}"
signed_release="${MOTIVE_WAN22_SIGNED_RELEASE:-}"
wan_code_root="${MOTIVE_WAN22_CODE_ROOT:?set MOTIVE_WAN22_CODE_ROOT}"
checkpoint_dir="${MOTIVE_WAN22_CKPT_DIR:?set MOTIVE_WAN22_CKPT_DIR}"
python_bin="${MOTIVE_WAN22_PYTHON_BIN:?set MOTIVE_WAN22_PYTHON_BIN}"
ffprobe_bin="${MOTIVE_WAN22_FFPROBE_BIN:?set MOTIVE_WAN22_FFPROBE_BIN}"
parallel_root="${MOTIVE_WAN22_PARALLEL_ROOT:?set MOTIVE_WAN22_PARALLEL_ROOT}"
geometry_job_id="${MOTIVE_WAN22_GEOMETRY_JOB_ID:?set MOTIVE_WAN22_GEOMETRY_JOB_ID}"
allow_pending_review="${MOTIVE_WAN22_ALLOW_PENDING_REVIEW:?set explicitly to 0}"
shard_count="${MOTIVE_WAN22_PARALLEL_SHARD_COUNT:-3}"
expected_row_count="${MOTIVE_WAN22_EXPECTED_ROW_COUNT:-8}"
data_root="${MOTIVE_WAN22_DATA_ROOT:-}"

code_root="${source_snapshot}/methods/motive"
prepare_module="${code_root}/motive/wan22_parallel_shards.py"
full_script="${code_root}/scripts/auh_wan22_i2v_parallel_full.sbatch"
finalize_script="${code_root}/scripts/auh_wan22_i2v_parallel_finalize.sbatch"

fail() {
  echo "[wan22-parallel-submit] $*" >&2
  exit 2
}

if [[ -z "${signed_release}" ]]; then
  fail "signed generation release gate is unavailable for unsigned inputs"
fi

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
  "parallel_root:${parallel_root}"; do
  require_absolute "${binding%%:*}" "${binding#*:}"
done
if [[ -n "${data_root}" ]]; then
  require_absolute "data_root" "${data_root}"
fi
require_positive_integer "MOTIVE_WAN22_GEOMETRY_JOB_ID" "${geometry_job_id}"
require_positive_integer "MOTIVE_WAN22_PARALLEL_SHARD_COUNT" "${shard_count}"
require_positive_integer "MOTIVE_WAN22_EXPECTED_ROW_COUNT" "${expected_row_count}"
if [[ "${allow_pending_review}" != "0" ]]; then
  fail "MOTIVE_WAN22_ALLOW_PENDING_REVIEW must be 0; only the signed exact-eight-row release authorizes generation"
fi
if [[ -e "${parallel_root}" || -L "${parallel_root}" ]]; then
  fail "parallel root must be fresh: ${parallel_root}"
fi
for required in \
  "${generation_manifest}" \
  "${signed_release}" \
  "${wan_code_root}/generate.py" \
  "${checkpoint_dir}/Wan2.1_VAE.pth" \
  "${ffprobe_bin}" \
  "${prepare_module}" \
  "${full_script}" \
  "${finalize_script}"; do
  if [[ -L "${required}" || ! -f "${required}" ]]; then
    fail "missing regular non-symlink file: ${required}"
  fi
done
if [[ ! -e "${python_bin}" || ! -x "${python_bin}" ]]; then
  fail "Python must resolve to an executable file: ${python_bin}"
fi
if [[ ! -x "${ffprobe_bin}" ]]; then
  fail "ffprobe is not executable: ${ffprobe_bin}"
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${code_root}"

prepare_args=(
  "${prepare_module}" prepare
  --manifest "${generation_manifest}"
  --signed-release "${signed_release}"
  --parallel-root "${parallel_root}"
  --geometry-job-id "${geometry_job_id}"
  --shard-count "${shard_count}"
  --expected-row-count "${expected_row_count}"
)
"${python_bin}" "${prepare_args[@]}"

shards_tsv="${parallel_root}/shards.tsv"
jobs_file="${parallel_root}/parallel_jobs.tsv"
if [[ -L "${shards_tsv}" || ! -s "${shards_tsv}" ]]; then
  fail "prepare did not publish a regular non-empty shard table"
fi
printf 'stage\tjob_id\tdependency\tmanifest\toutput_root\n' > "${jobs_file}"
chmod 0600 "${jobs_file}"

# Freeze the full-quality contract and the socket-only RCCL transport.  Each
# sbatch receives a complete copy through --export=ALL.
export MOTIVE_WAN22_FRAME_NUM=81
export MOTIVE_WAN22_SAMPLE_STEPS=40
export MOTIVE_WAN22_SAMPLE_SHIFT=5.0
export MOTIVE_WAN22_SIZE='1280*720'
export MOTIVE_WAN22_BASE_SEED=260730
unset MOTIVE_WAN22_MAX_SAMPLES
export NCCL_IB_DISABLE=1
unset NCCL_IB_HCA NCCL_IB_GID_INDEX
export NCCL_SOCKET_IFNAME=bond0
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=bond0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,NET

shard_jobs=()
while IFS=$'\t' read -r \
  shard_id row_start row_stop row_count manifest_sha shard_manifest \
  shard_output stdout_pattern stderr_pattern; do
  if [[ "${shard_id}" == "shard_id" ]]; then
    continue
  fi
  require_positive_integer "${shard_id} row_count" "${row_count}"
  if [[ ! "${manifest_sha}" =~ ^[0-9a-f]{64}$ ]]; then
    fail "${shard_id} manifest SHA-256 is invalid"
  fi
  require_absolute "${shard_id} manifest" "${shard_manifest}"
  require_absolute "${shard_id} output" "${shard_output}"
  if [[ -e "${shard_output}" || -L "${shard_output}" ]]; then
    fail "${shard_id} output root must be fresh: ${shard_output}"
  fi
  if [[ "$(sha256sum "${shard_manifest}" | cut -d' ' -f1)" != "${manifest_sha}" ]]; then
    fail "${shard_id} manifest differs from the prepared plan"
  fi

  export MOTIVE_GOKU_ACTION_GENERATION_MANIFEST="${shard_manifest}"
  export MOTIVE_WAN22_OUTPUT_ROOT="${shard_output}"
  export MOTIVE_WAN22_MAX_NEW_SAMPLES="${row_count}"
  raw_submission="$(
    sbatch \
      --parsable \
      --job-name="wan22-${shard_id}" \
      --kill-on-invalid-dep=yes \
      --dependency="afterok:${geometry_job_id}" \
      --export=ALL \
      --output="${stdout_pattern}" \
      --error="${stderr_pattern}" \
      "${full_script}"
  )"
  printf '%s\n' "${raw_submission}" \
    > "${parallel_root}/submissions/${shard_id}.raw"
  job_id="${raw_submission%%;*}"
  require_positive_integer "${shard_id} job ID" "${job_id}"
  shard_jobs+=("${job_id}")
  printf '%s\t%s\tafterok:%s\t%s\t%s\n' \
    "${shard_id}" \
    "${job_id}" \
    "${geometry_job_id}" \
    "${shard_manifest}" \
    "${shard_output}" >> "${jobs_file}"
done < "${shards_tsv}"

if (( ${#shard_jobs[@]} != shard_count )); then
  fail "submitted shard count differs: expected=${shard_count} actual=${#shard_jobs[@]}"
fi
dependency="afterok"
for job_id in "${shard_jobs[@]}"; do
  dependency+=":${job_id}"
done

export MOTIVE_WAN22_PARALLEL_ROOT="${parallel_root}"
raw_finalize="$(
  sbatch \
    --parsable \
    --job-name=wan22-parallel-finalize \
    --kill-on-invalid-dep=yes \
    --dependency="${dependency}" \
    --export=ALL \
    --output="${parallel_root}/logs/finalize-%j.out" \
    --error="${parallel_root}/logs/finalize-%j.err" \
    "${finalize_script}"
)"
printf '%s\n' "${raw_finalize}" \
  > "${parallel_root}/submissions/finalize.raw"
finalize_job="${raw_finalize%%;*}"
require_positive_integer "finalize job ID" "${finalize_job}"
printf 'finalize\t%s\t%s\t%s\t%s\n' \
  "${finalize_job}" \
  "${dependency}" \
  "${generation_manifest}" \
  "${parallel_root}/final" >> "${jobs_file}"

chmod 0600 "${jobs_file}" "${parallel_root}"/submissions/*.raw
printf 'parallel_root=%s\n' "${parallel_root}"
printf 'geometry_job_id=%s\n' "${geometry_job_id}"
printf 'shard_job_ids=%s\n' "${shard_jobs[*]}"
printf 'finalize_job_id=%s\n' "${finalize_job}"
printf 'jobs_file=%s\n' "${jobs_file}"
