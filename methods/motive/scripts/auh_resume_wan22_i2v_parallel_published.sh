#!/usr/bin/env bash

# Resume a prepared parallel run after Slurm has already purged the successful
# geometry job from its live dependency table.  The published geometry
# receipts are verified before any independent shard is submitted.

set -Eeuo pipefail
umask 077

source_snapshot="${MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT:?set MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT}"
generation_manifest="${MOTIVE_GOKU_ACTION_GENERATION_MANIFEST:?set MOTIVE_GOKU_ACTION_GENERATION_MANIFEST}"
signed_release="${MOTIVE_WAN22_SIGNED_RELEASE:?set MOTIVE_WAN22_SIGNED_RELEASE}"
wan_code_root="${MOTIVE_WAN22_CODE_ROOT:?set MOTIVE_WAN22_CODE_ROOT}"
checkpoint_dir="${MOTIVE_WAN22_CKPT_DIR:?set MOTIVE_WAN22_CKPT_DIR}"
python_bin="${MOTIVE_WAN22_PYTHON_BIN:?set MOTIVE_WAN22_PYTHON_BIN}"
ffprobe_bin="${MOTIVE_WAN22_FFPROBE_BIN:?set MOTIVE_WAN22_FFPROBE_BIN}"
parallel_root="${MOTIVE_WAN22_PARALLEL_ROOT:?set MOTIVE_WAN22_PARALLEL_ROOT}"
geometry_output_root="${MOTIVE_WAN22_GEOMETRY_OUTPUT_ROOT:?set MOTIVE_WAN22_GEOMETRY_OUTPUT_ROOT}"
geometry_job_id="${MOTIVE_WAN22_GEOMETRY_JOB_ID:?set MOTIVE_WAN22_GEOMETRY_JOB_ID}"
allow_pending_review="${MOTIVE_WAN22_ALLOW_PENDING_REVIEW:?set explicitly to 0}"
expected_row_count="${MOTIVE_WAN22_EXPECTED_ROW_COUNT:-8}"
data_root="${MOTIVE_WAN22_DATA_ROOT:-}"

full_script="${source_snapshot}/methods/motive/scripts/auh_wan22_i2v_parallel_full.sbatch"
finalize_script="${source_snapshot}/methods/motive/scripts/auh_wan22_i2v_parallel_finalize.sbatch"
plan="${parallel_root}/parallel_plan.json"
shards_tsv="${parallel_root}/shards.tsv"
jobs_file="${parallel_root}/parallel_jobs.tsv"
submissions="${parallel_root}/submissions"

fail() {
  echo "[wan22-parallel-published-resume] $*" >&2
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
  "parallel_root:${parallel_root}" \
  "geometry_output_root:${geometry_output_root}"; do
  require_absolute "${binding%%:*}" "${binding#*:}"
done
if [[ -n "${data_root}" ]]; then
  require_absolute "data_root" "${data_root}"
fi
require_positive_integer "MOTIVE_WAN22_GEOMETRY_JOB_ID" "${geometry_job_id}"
require_positive_integer "MOTIVE_WAN22_EXPECTED_ROW_COUNT" "${expected_row_count}"
if [[ "${allow_pending_review}" != "0" ]]; then
  fail "MOTIVE_WAN22_ALLOW_PENDING_REVIEW must be 0; only the signed exact-eight-row release authorizes generation"
fi
if [[ ! -e "${python_bin}" || ! -x "${python_bin}" ]]; then
  fail "Python must resolve to an executable file: ${python_bin}"
fi
for required in \
  "${generation_manifest}" \
  "${signed_release}" \
  "${ffprobe_bin}" \
  "${full_script}" \
  "${finalize_script}" \
  "${plan}" \
  "${shards_tsv}" \
  "${jobs_file}" \
  "${geometry_output_root}/run_contract.json" \
  "${geometry_output_root}/run_complete.json" \
  "${geometry_output_root}/generated_manifest.jsonl"; do
  if [[ -L "${required}" || ! -s "${required}" ]]; then
    fail "missing regular non-empty non-symlink file: ${required}"
  fi
done
if [[ -L "${submissions}" || ! -d "${submissions}" ]]; then
  fail "submissions must be a non-symlink directory: ${submissions}"
fi
if [[ "$(wc -l < "${jobs_file}" | tr -d '[:space:]')" != "1" ]]; then
  fail "jobs table is not an untouched prepared header: ${jobs_file}"
fi
if [[ -n "$(find "${submissions}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  fail "submission evidence already exists: ${submissions}"
fi

manifest_sha="$(sha256sum "${generation_manifest}" | cut -d' ' -f1)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
"${python_bin}" - \
  "${geometry_output_root}" \
  "${generation_manifest}" \
  "${manifest_sha}" \
  "${parallel_root}" \
  "${geometry_job_id}" \
  "${expected_row_count}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

geometry = Path(sys.argv[1])
manifest = Path(sys.argv[2]).resolve(strict=True)
manifest_sha = sys.argv[3]
parallel_root = Path(sys.argv[4]).resolve(strict=True)
geometry_job_id = int(sys.argv[5])
expected_row_count = int(sys.argv[6])

contract = json.loads((geometry / "run_contract.json").read_text("utf-8"))
completion = json.loads((geometry / "run_complete.json").read_text("utf-8"))
generated = (geometry / "generated_manifest.jsonl").read_bytes()
plan = json.loads((parallel_root / "parallel_plan.json").read_text("utf-8"))
parameters = contract["generation_parameters"]
distributed = contract["distributed_execution"]

assert contract["manifest"]["path"] == str(manifest)
assert contract["manifest"]["sha256"] == manifest_sha
assert parameters["size"] == "1280*720"
assert parameters["frame_num"] == 81
assert parameters["sample_steps"] == 1
assert parameters["sample_shift"] == 5.0
assert distributed["world_size"] == 8
assert completion["contract_digest"] == contract["contract_digest"]
assert completion["selected_sample_count"] == 1
assert completion["completed_sample_count"] == 1
assert completion["generated_manifest_sha256"] == hashlib.sha256(
    generated
).hexdigest()

assert plan["parallel_root"] == str(parallel_root)
assert plan["geometry_job_id"] == geometry_job_id
assert plan["source_manifest"]["path"] == str(manifest)
assert plan["source_manifest"]["sha256"] == manifest_sha
assert plan["source_manifest"]["row_count"] == expected_row_count
assert plan["expected_source_row_count"] == expected_row_count
assert len(plan["shards"]) == plan["shard_count"]
assert sum(shard["row_count"] for shard in plan["shards"]) == expected_row_count

print(
    "published_geometry_verified",
    completion["complete_digest"],
    flush=True,
)
PY

# Freeze the same full-quality and socket-only transport contract used by the
# successful technical and full-geometry smoke jobs.
export MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT="${source_snapshot}"
export MOTIVE_WAN22_CODE_ROOT="${wan_code_root}"
export MOTIVE_WAN22_CKPT_DIR="${checkpoint_dir}"
export MOTIVE_WAN22_PYTHON_BIN="${python_bin}"
export MOTIVE_WAN22_FFPROBE_BIN="${ffprobe_bin}"
export MOTIVE_WAN22_ALLOW_PENDING_REVIEW="${allow_pending_review}"
export MOTIVE_WAN22_FRAME_NUM=81
export MOTIVE_WAN22_SAMPLE_STEPS=40
export MOTIVE_WAN22_SAMPLE_SHIFT=5.0
export MOTIVE_WAN22_SIZE='1280*720'
export MOTIVE_WAN22_BASE_SEED=260730
export NCCL_IB_DISABLE=1
unset NCCL_IB_HCA NCCL_IB_GID_INDEX
export NCCL_SOCKET_IFNAME=bond0
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=bond0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,NET
unset MOTIVE_WAN22_MAX_SAMPLES

shard_jobs=()
while IFS=$'\t' read -r \
  shard_id row_start row_stop row_count manifest_digest shard_manifest \
  shard_output stdout_pattern stderr_pattern; do
  if [[ "${shard_id}" == "shard_id" ]]; then
    continue
  fi
  require_positive_integer "${shard_id} row_count" "${row_count}"
  if [[ "$(sha256sum "${shard_manifest}" | cut -d' ' -f1)" != "${manifest_digest}" ]]; then
    fail "${shard_id} manifest differs from the prepared plan"
  fi
  if [[ -e "${shard_output}" || -L "${shard_output}" ]]; then
    fail "${shard_id} output root must be fresh: ${shard_output}"
  fi

  export MOTIVE_GOKU_ACTION_GENERATION_MANIFEST="${shard_manifest}"
  export MOTIVE_WAN22_OUTPUT_ROOT="${shard_output}"
  export MOTIVE_WAN22_MAX_NEW_SAMPLES="${row_count}"
  raw_submission="$(
    sbatch \
      --parsable \
      --job-name="wan22-${shard_id}" \
      --export=ALL \
      --output="${stdout_pattern}" \
      --error="${stderr_pattern}" \
      "${full_script}"
  )"
  job_id="${raw_submission%%;*}"
  require_positive_integer "${shard_id} job ID" "${job_id}"
  printf '%s\n' "${raw_submission}" > "${submissions}/${shard_id}.raw"
  printf '%s\t%s\tpublished:%s\t%s\t%s\n' \
    "${shard_id}" \
    "${job_id}" \
    "${geometry_job_id}" \
    "${shard_manifest}" \
    "${shard_output}" >> "${jobs_file}"
  shard_jobs+=("${job_id}")
done < "${shards_tsv}"

if (( ${#shard_jobs[@]} != 3 )); then
  fail "expected exactly three shard submissions, got ${#shard_jobs[@]}"
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
finalize_job="${raw_finalize%%;*}"
require_positive_integer "finalize job ID" "${finalize_job}"
printf '%s\n' "${raw_finalize}" > "${submissions}/finalize.raw"
printf 'finalize\t%s\t%s\t%s\t%s\n' \
  "${finalize_job}" \
  "${dependency}" \
  "${generation_manifest}" \
  "${parallel_root}/final" >> "${jobs_file}"

chmod 0600 "${jobs_file}" "${submissions}"/*.raw
printf 'published_geometry_job_id=%s\n' "${geometry_job_id}"
printf 'shard_job_ids=%s\n' "${shard_jobs[*]}"
printf 'finalize_job_id=%s\n' "${finalize_job}"
printf 'jobs_file=%s\n' "${jobs_file}"
