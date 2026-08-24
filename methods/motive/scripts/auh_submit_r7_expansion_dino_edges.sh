#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# Run this wrapper on an AUH login node.  It performs every check that can be
# completed without a GPU, reserves one fresh output root, and only then calls
# sbatch.  The compute-node script repeats the output-state check to fail
# closed if anything changes while the job is queued.

source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?set MOTIVE_SOURCE_SNAPSHOT}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?set MOTIVE_SOURCE_TREE_SHA256}"
input_directory="${MOTIVE_R7_GRAPH_INPUT:?set MOTIVE_R7_GRAPH_INPUT}"
input_artifact_digest="${MOTIVE_R7_GRAPH_INPUT_DIGEST:?set MOTIVE_R7_GRAPH_INPUT_DIGEST}"
output_root="${MOTIVE_R7_DINO_EDGE_OUTPUT:?set MOTIVE_R7_DINO_EDGE_OUTPUT}"
python_bin="${PYTHON_BIN:?set PYTHON_BIN}"
block_size="${MOTIVE_R7_DINO_BLOCK_SIZE:-256}"
audit_top_k="${MOTIVE_R7_DINO_AUDIT_TOP_K:-20}"
calibration_per_stratum="${MOTIVE_R7_DINO_CALIBRATION_PER_STRATUM:-256}"
log_dir="${MOTIVE_R7_DINO_LOG_DIR:-${output_root}.logs}"
job_name="${MOTIVE_R7_DINO_JOB_NAME:-m7-dinoedge}"
sbatch_bin="sbatch"

readonly slurm_partition="faculty"
readonly slurm_account="test-acc"
readonly slurm_qos="bgqos"
readonly slurm_nodes="1"
readonly slurm_ntasks="1"
readonly slurm_cpus_per_task="32"
readonly slurm_memory="256G"
readonly slurm_gres="gpu:mi210:8"
readonly slurm_time_limit="01:00:00"
readonly slurm_exclude="auh7-1b-gpu-185,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318"

reservation_dir=""
reservation_owned=0
job_submitted=0
login_validation_cwd=""
atomic_temp=""

fail() {
  echo "[r7-dino-submit] $*" >&2
  exit 2
}

cleanup() {
  if [[ -n "${atomic_temp}" && -f "${atomic_temp}" ]]; then
    rm -f "${atomic_temp}"
  fi
  if [[
    "${reservation_owned}" == "1" &&
    "${job_submitted}" == "0" &&
    -n "${reservation_dir}" &&
    -d "${reservation_dir}" &&
    ! -L "${reservation_dir}"
  ]]; then
    rm -f "${reservation_dir}/token"
    rmdir "${reservation_dir}" 2>/dev/null || true
  fi
  if [[
    -n "${login_validation_cwd}" &&
    -d "${login_validation_cwd}" &&
    ! -L "${login_validation_cwd}"
  ]]; then
    rmdir "${login_validation_cwd}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

reject_exported_sbatch_controls() {
  local entry=""
  local name=""
  while IFS= read -r -d '' entry; do
    name="${entry%%=*}"
    if [[ "${name}" == SBATCH_* ]]; then
      fail "exported ${name} is forbidden; Slurm resources are fixed"
    fi
  done < <(env -0)
}

canonical_existing_path() {
  local raw="$1"
  local label="$2"
  local resolved=""
  if [[ ! -e "${raw}" ]]; then
    fail "${label} does not exist: ${raw}"
  fi
  if ! resolved="$(realpath "${raw}")"; then
    fail "${label} cannot be resolved: ${raw}"
  fi
  if [[ "${resolved}" != /* ]]; then
    fail "${label} did not resolve to an absolute path: ${raw}"
  fi
  printf '%s\n' "${resolved}"
}

canonical_target_directory() {
  local raw="$1"
  local label="$2"
  local probe=""
  local component=""
  local suffix=""
  local resolved=""

  while [[ "${raw}" != "/" && "${raw}" == */ ]]; do
    raw="${raw%/}"
  done
  if [[ -L "${raw}" ]]; then
    fail "${label} must not be a symlink: ${raw}"
  fi
  if [[ -e "${raw}" ]]; then
    if [[ ! -d "${raw}" ]]; then
      fail "${label} is not a directory: ${raw}"
    fi
    canonical_existing_path "${raw}" "${label}"
    return
  fi

  probe="${raw}"
  while [[ ! -e "${probe}" && ! -L "${probe}" ]]; do
    component="${probe##*/}"
    if [[
      -z "${component}" ||
      "${component}" == "." ||
      "${component}" == ".."
    ]]; then
      fail "${label} has a non-canonical missing component: ${raw}"
    fi
    suffix="/${component}${suffix}"
    probe="${probe%/*}"
    if [[ -z "${probe}" ]]; then
      probe="/"
    fi
  done
  if [[ -L "${probe}" && ! -e "${probe}" ]]; then
    fail "${label} has a dangling symlink ancestor: ${probe}"
  fi
  if ! resolved="$(realpath "${probe}")"; then
    fail "${label} ancestor cannot be resolved: ${probe}"
  fi
  if [[ ! -d "${resolved}" ]]; then
    fail "${label} ancestor is not a directory: ${resolved}"
  fi
  if [[ "${resolved}" == "/" ]]; then
    printf '/%s\n' "${suffix#/}"
  else
    printf '%s%s\n' "${resolved}" "${suffix}"
  fi
}

path_is_within() {
  local child="$1"
  local parent="$2"
  if [[ "${child}" == "${parent}" ]]; then
    return 0
  fi
  if [[ "${parent}" == "/" ]]; then
    [[ "${child}" == /* ]]
    return
  fi
  [[ "${child}" == "${parent}/"* ]]
}

reject_path_overlap() {
  local first="$1"
  local first_label="$2"
  local second="$3"
  local second_label="$4"
  if path_is_within "${first}" "${second}" ||
    path_is_within "${second}" "${first}"; then
    fail \
      "${first_label} and ${second_label} overlap after normalization: " \
      "${first} <-> ${second}"
  fi
}

check_fresh_output() {
  local root="$1"
  local final_path="${root}/final"
  local shards_path="${root}/shards"
  local first_entry=""

  if [[ -L "${root}" ]]; then
    fail "output root must not be a symlink: ${root}"
  fi
  if [[ -e "${root}" && ! -d "${root}" ]]; then
    fail "output root is not a directory: ${root}"
  fi
  if [[ -e "${final_path}" || -L "${final_path}" ]]; then
    fail "final output already exists: ${final_path}"
  fi
  if [[ -L "${shards_path}" ]]; then
    fail "shards path must not be a symlink: ${shards_path}"
  fi
  if [[ -e "${shards_path}" && ! -d "${shards_path}" ]]; then
    fail "shards path is not a directory: ${shards_path}"
  fi
  if [[ -d "${shards_path}" ]]; then
    first_entry="$(
      find "${shards_path}" \
        -mindepth 1 \
        -maxdepth 1 \
        -print \
        -quit
    )"
    if [[ -n "${first_entry}" ]]; then
      fail "shards directory is not empty: ${first_entry}"
    fi
  fi
}

write_atomic_readonly() {
  local directory="$1"
  local filename="$2"
  local value="$3"
  local target="${directory}/${filename}"

  if [[ -e "${target}" || -L "${target}" ]]; then
    fail "reservation record already exists: ${target}"
  fi
  atomic_temp="$(
    mktemp "${directory}/.${filename}.tmp.XXXXXX"
  )"
  printf '%s\n' "${value}" > "${atomic_temp}"
  chmod 0444 "${atomic_temp}"
  mv "${atomic_temp}" "${target}"
  atomic_temp=""
}

run_login_python() {
  (
    cd "${login_validation_cwd}"
    "${python_bin}" "$@"
  )
}

reject_exported_sbatch_controls

if [[
  ! "${source_tree_sha256}" =~ ^[0-9a-f]{64}$ ||
  ! "${input_artifact_digest}" =~ ^[0-9a-f]{64}$
]]; then
  fail "source tree and graph-input digests must be lowercase SHA-256"
fi
if [[ ! "${block_size}" =~ ^[1-9][0-9]*$ ]]; then
  fail "MOTIVE_R7_DINO_BLOCK_SIZE must be a positive integer"
fi
if [[ ! "${audit_top_k}" =~ ^[1-9][0-9]*$ ]]; then
  fail "MOTIVE_R7_DINO_AUDIT_TOP_K must be a positive integer"
fi
if [[ ! "${calibration_per_stratum}" =~ ^[1-9][0-9]*$ ]]; then
  fail "MOTIVE_R7_DINO_CALIBRATION_PER_STRATUM must be a positive integer"
fi
if [[ ! "${job_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  fail "MOTIVE_R7_DINO_JOB_NAME is not a safe Slurm job name"
fi
for absolute_path in \
  "${source_snapshot}" \
  "${input_directory}" \
  "${output_root}" \
  "${python_bin}" \
  "${log_dir}"; do
  if [[ "${absolute_path}" != /* ]]; then
    fail "all paths must be absolute: ${absolute_path}"
  fi
  if [[
    "${absolute_path}" == *$'\n'* ||
    "${absolute_path}" == *$'\r'*
  ]]; then
    fail "paths must not contain control newlines"
  fi
done

source_snapshot="$(
  canonical_existing_path "${source_snapshot}" "source snapshot"
)"
input_directory="$(
  canonical_existing_path "${input_directory}" "graph input"
)"
python_bin="$(
  canonical_existing_path "${python_bin}" "Python executable"
)"
output_root="$(
  canonical_target_directory "${output_root}" "output root"
)"
log_dir="$(
  canonical_target_directory "${log_dir}" "log directory"
)"
reservation_dir="${output_root}/.r7-dino-edge-submission.lock"

if [[ ! -d "${source_snapshot}" ]]; then
  fail "source snapshot is not a directory: ${source_snapshot}"
fi
if [[ ! -d "${input_directory}" ]]; then
  fail "graph input is not a directory: ${input_directory}"
fi
if [[ ! -x "${python_bin}" || ! -f "${python_bin}" ]]; then
  fail "PYTHON_BIN is not an executable regular file: ${python_bin}"
fi

reject_path_overlap \
  "${output_root}" "output root" \
  "${log_dir}" "log directory"
mutable_paths=("${output_root}" "${log_dir}")
mutable_labels=("output root" "log directory")
for mutable_index in "${!mutable_paths[@]}"; do
  mutable_path="${mutable_paths[mutable_index]}"
  mutable_label="${mutable_labels[mutable_index]}"
  reject_path_overlap \
    "${mutable_path}" "${mutable_label}" \
    "${source_snapshot}" "source snapshot"
  reject_path_overlap \
    "${mutable_path}" "${mutable_label}" \
    "${input_directory}" "graph input"
  reject_path_overlap \
    "${mutable_path}" "${mutable_label}" \
    "${python_bin}" "Python executable"
done

for export_value in \
  "${source_snapshot}" \
  "${source_tree_sha256}" \
  "${input_directory}" \
  "${input_artifact_digest}" \
  "${output_root}" \
  "${python_bin}" \
  "${block_size}" \
  "${audit_top_k}" \
  "${calibration_per_stratum}"; do
  if [[ "${export_value}" == *","* || "${export_value}" == *$'\n'* ]]; then
    fail "Slurm export values must not contain commas or newlines"
  fi
done

for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${source_snapshot}/methods/motive/motive/r7_expansion_dino_edges.py" \
  "${source_snapshot}/methods/motive/motive/r7_dino_quotient_calibration.py" \
  "${source_snapshot}/methods/motive/motive/r7_visual_graph_input.py" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  "${source_snapshot}/methods/motive/scripts/auh_r7_expansion_dino_edges.sbatch" \
  "${input_directory}/manifest.jsonl" \
  "${input_directory}/features.npz" \
  "${input_directory}/summary.json" \
  "${input_directory}/done.json"; do
  if [[ ! -e "${required}" ]]; then
    fail "missing required path: ${required}"
  fi
done
if ! command -v "${sbatch_bin}" >/dev/null 2>&1; then
  fail "sbatch command is unavailable"
fi
if [[ -e "${reservation_dir}" || -L "${reservation_dir}" ]]; then
  fail "output root is already reserved: ${reservation_dir}"
fi
check_fresh_output "${output_root}"

login_cwd_parent="$(realpath /tmp)"
comparison_paths=(
  "${source_snapshot}"
  "${input_directory}"
  "${output_root}"
  "${log_dir}"
  "${python_bin}"
)
comparison_labels=(
  "source snapshot"
  "graph input"
  "output root"
  "log directory"
  "Python executable"
)
for comparison_index in "${!comparison_paths[@]}"; do
  comparison_path="${comparison_paths[comparison_index]}"
  comparison_label="${comparison_labels[comparison_index]}"
  if path_is_within \
    "${login_cwd_parent}" \
    "${comparison_path}"; then
    fail \
      "${comparison_label} contains the isolated login cwd parent: " \
      "${comparison_path}"
  fi
done
login_validation_cwd="$(
  mktemp -d "${login_cwd_parent}/motive-r7-dino-submit.XXXXXX"
)"
login_validation_cwd="$(
  canonical_existing_path \
    "${login_validation_cwd}" \
    "login validation directory"
)"
reject_path_overlap \
  "${login_validation_cwd}" "login validation directory" \
  "${source_snapshot}" "source snapshot"
reject_path_overlap \
  "${login_validation_cwd}" "login validation directory" \
  "${input_directory}" "graph input"
reject_path_overlap \
  "${login_validation_cwd}" "login validation directory" \
  "${output_root}" "output root"
reject_path_overlap \
  "${login_validation_cwd}" "login validation directory" \
  "${log_dir}" "log directory"
reject_path_overlap \
  "${login_validation_cwd}" "login validation directory" \
  "${python_bin}" "Python executable"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${source_snapshot}/methods/motive"
unset PYTHONHOME PYTHONSTARTUP PYTHONINSPECT

run_login_python \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

run_login_python - \
  "${source_snapshot}" \
  "${input_directory}" \
  "${input_artifact_digest}" \
  "${login_validation_cwd}" <<'PY'
import sys
from pathlib import Path

import motive.r7_expansion_dino_edges as module

snapshot = Path(sys.argv[1]).resolve(strict=True)
expected_module = (
    snapshot
    / "methods"
    / "motive"
    / "motive"
    / "r7_expansion_dino_edges.py"
).resolve(strict=True)
actual_module = Path(module.__file__).resolve(strict=True)
if actual_module != expected_module:
    raise SystemExit(
        "[r7-dino-submit] motive module origin differs: "
        f"expected={expected_module} actual={actual_module}"
    )
expected_cwd = Path(sys.argv[4]).resolve(strict=True)
actual_cwd = Path.cwd().resolve(strict=True)
if actual_cwd != expected_cwd:
    raise SystemExit("[r7-dino-submit] login validation cwd differs")
try:
    actual_cwd.relative_to(snapshot)
except ValueError:
    pass
else:
    raise SystemExit(
        "[r7-dino-submit] login validation cwd is inside snapshot"
    )

validated = module.validate_graph_input(Path(sys.argv[2]))
expected = sys.argv[3]
actual = validated["artifact_digest"]
if actual != expected:
    raise SystemExit(
        "[r7-dino-submit] graph-input artifact digest differs: "
        f"expected={expected} actual={actual}"
    )
print(
    "[r7-dino-submit] graph input verified "
    f"rows={len(validated['rows'])} artifact_digest={actual}",
    flush=True,
)
PY

# Recheck after the potentially long graph validation, then claim this exact
# output root with an atomic directory creation.  Cooperative submissions
# cannot race each other into the same immutable shard namespace.
check_fresh_output "${output_root}"
if [[ ! -e "${output_root}" ]]; then
  mkdir -p "${output_root}"
fi
if [[
  -L "${output_root}" ||
  ! -d "${output_root}" ||
  "$(realpath "${output_root}")" != "${output_root}"
]]; then
  fail "output root changed identity during reservation: ${output_root}"
fi
submission_token="$(
  run_login_python -c \
    'import secrets; print(secrets.token_hex(32))'
)"
if [[ ! "${submission_token}" =~ ^[0-9a-f]{64}$ ]]; then
  fail "failed to generate a submission token"
fi
if ! mkdir "${reservation_dir}"; then
  fail "another submission reserved this output root"
fi
chmod 0700 "${reservation_dir}"
reservation_owned=1
write_atomic_readonly "${reservation_dir}" "token" "${submission_token}"
check_fresh_output "${output_root}"

if [[ ! -e "${log_dir}" ]]; then
  mkdir -p "${log_dir}"
fi
if [[
  -L "${log_dir}" ||
  ! -d "${log_dir}" ||
  "$(realpath "${log_dir}")" != "${log_dir}"
]]; then
  fail "log directory changed identity: ${log_dir}"
fi

common_export="ALL,MOTIVE_SOURCE_SNAPSHOT=${source_snapshot},MOTIVE_SOURCE_TREE_SHA256=${source_tree_sha256},MOTIVE_R7_GRAPH_INPUT=${input_directory},MOTIVE_R7_GRAPH_INPUT_DIGEST=${input_artifact_digest},MOTIVE_R7_DINO_EDGE_OUTPUT=${output_root},MOTIVE_R7_DINO_SUBMISSION_TOKEN=${submission_token},MOTIVE_R7_DINO_BLOCK_SIZE=${block_size},MOTIVE_R7_DINO_AUDIT_TOP_K=${audit_top_k},MOTIVE_R7_DINO_CALIBRATION_PER_STRATUM=${calibration_per_stratum},PYTHON_BIN=${python_bin}"
job_id="$(
  "${sbatch_bin}" \
    --parsable \
    --partition="${slurm_partition}" \
    --account="${slurm_account}" \
    --qos="${slurm_qos}" \
    --nodes="${slurm_nodes}" \
    --ntasks="${slurm_ntasks}" \
    --cpus-per-task="${slurm_cpus_per_task}" \
    --mem="${slurm_memory}" \
    --gres="${slurm_gres}" \
    --time="${slurm_time_limit}" \
    --exclude="${slurm_exclude}" \
    --job-name="${job_name}" \
    --output="${log_dir}/%x_%j.out" \
    --error="${log_dir}/%x_%j.err" \
    --export="${common_export}" \
    "${source_snapshot}/methods/motive/scripts/auh_r7_expansion_dino_edges.sbatch"
)"
job_submitted=1
if [[ ! "${job_id}" =~ ^[1-9][0-9]*$ ]]; then
  fail "sbatch returned an invalid numeric job id"
fi

# Publish the accepted job id atomically inside the reservation before any
# result is printed.  A failed provenance write leaves the private reservation
# in place instead of making the running job look unreserved.
write_atomic_readonly "${reservation_dir}" "job_id" "${job_id}"
reservation_owned=0

echo "job_id=${job_id}"
echo "output_root=${output_root}"
echo "stdout=${log_dir}/${job_name}_${job_id}.out"
echo "stderr=${log_dir}/${job_name}_${job_id}.err"
