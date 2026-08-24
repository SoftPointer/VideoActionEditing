#!/usr/bin/env bash
#SBATCH --job-name=m7-dinoedge
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=01:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 077

source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?set MOTIVE_SOURCE_SNAPSHOT}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?set MOTIVE_SOURCE_TREE_SHA256}"
input_directory="${MOTIVE_R7_GRAPH_INPUT:?set MOTIVE_R7_GRAPH_INPUT}"
input_artifact_digest="${MOTIVE_R7_GRAPH_INPUT_DIGEST:?set MOTIVE_R7_GRAPH_INPUT_DIGEST}"
output_root="${MOTIVE_R7_DINO_EDGE_OUTPUT:?set MOTIVE_R7_DINO_EDGE_OUTPUT}"
submission_token="${MOTIVE_R7_DINO_SUBMISSION_TOKEN:?use auh_submit_r7_expansion_dino_edges.sh}"
python_bin="${PYTHON_BIN:?set PYTHON_BIN}"
block_size="${MOTIVE_R7_DINO_BLOCK_SIZE:-256}"
audit_top_k="${MOTIVE_R7_DINO_AUDIT_TOP_K:-20}"
calibration_per_stratum="${MOTIVE_R7_DINO_CALIBRATION_PER_STRATUM:-256}"
reservation_dir="${output_root}/.r7-dino-edge-submission.lock"
active_child=""

fail() {
  echo "[r7-dino-edges] $*" >&2
  exit 2
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
  if [[ "${resolved}" != "${raw}" ]]; then
    fail "${label} is not the canonical path: ${raw} -> ${resolved}"
  fi
  printf '%s\n' "${resolved}"
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
      "${first_label} and ${second_label} overlap: " \
      "${first} <-> ${second}"
  fi
}

forward_signal() {
  local signal_name="$1"
  local exit_code="$2"
  trap - "${signal_name}"
  if [[ -n "${active_child}" ]] &&
    kill -0 "${active_child}" 2>/dev/null; then
    kill -s "${signal_name}" "${active_child}" 2>/dev/null || true
    set +e
    wait "${active_child}"
    set -e
  fi
  exit "${exit_code}"
}

run_checked() {
  local status=0
  "$@" &
  active_child="$!"
  set +e
  wait "${active_child}"
  status="$?"
  set -e
  active_child=""
  return "${status}"
}

trap 'forward_signal TERM 143' TERM
trap 'forward_signal INT 130' INT
trap 'forward_signal HUP 129' HUP

check_fresh_output() {
  local root="$1"
  local final_path="${root}/final"
  local shards_path="${root}/shards"
  local first_entry=""

  if [[ -L "${root}" || ! -d "${root}" ]]; then
    fail "output root must be a real directory: ${root}"
  fi
  if [[ -e "${final_path}" || -L "${final_path}" ]]; then
    fail "final output already exists; refusing GPU work: ${final_path}"
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
      fail "shards directory is not empty; refusing GPU work: ${first_entry}"
    fi
  fi
}

if [[ ! "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]]; then
  fail "SLURM_JOB_ID must be a positive numeric identifier"
fi
for absolute_path in \
  "${source_snapshot}" \
  "${input_directory}" \
  "${output_root}" \
  "${python_bin}"; do
  if [[ "${absolute_path}" != /* ]]; then
    fail "all compute paths must be absolute: ${absolute_path}"
  fi
done
canonical_existing_path "${source_snapshot}" "source snapshot" >/dev/null
canonical_existing_path "${input_directory}" "graph input" >/dev/null
canonical_existing_path "${output_root}" "output root" >/dev/null
canonical_existing_path "${python_bin}" "Python executable" >/dev/null
if [[
  ! -d "${source_snapshot}" ||
  ! -d "${input_directory}" ||
  ! -d "${output_root}" ||
  ! -f "${python_bin}" ||
  ! -x "${python_bin}"
]]; then
  fail "compute source/input/output/Python path type differs"
fi

for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${source_snapshot}/methods/motive/motive/r7_expansion_dino_edges.py" \
  "${source_snapshot}/methods/motive/motive/r7_dino_quotient_calibration.py" \
  "${source_snapshot}/methods/motive/motive/r7_visual_graph_input.py" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  "${input_directory}/manifest.jsonl" \
  "${input_directory}/features.npz" \
  "${input_directory}/summary.json" \
  "${input_directory}/done.json" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[r7-dino-edges] missing required path: ${required}" >&2
    exit 2
  fi
done
if [[ ! -x "${python_bin}" ]]; then
  fail "PYTHON_BIN is not executable: ${python_bin}"
fi
if [[
  ! "${source_tree_sha256}" =~ ^[0-9a-f]{64}$ ||
  ! "${input_artifact_digest}" =~ ^[0-9a-f]{64}$ ||
  ! "${submission_token}" =~ ^[0-9a-f]{64}$
]]; then
  fail "source, graph-input, and submission digests are invalid"
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
if [[ -L "${reservation_dir}" || ! -d "${reservation_dir}" ]]; then
  fail "missing immutable submission reservation: ${reservation_dir}"
fi
if [[
  -L "${reservation_dir}/token" ||
  ! -f "${reservation_dir}/token"
]]; then
  fail "invalid submission reservation token file"
fi
actual_submission_token="$(<"${reservation_dir}/token")"
if [[ "${actual_submission_token}" != "${submission_token}" ]]; then
  fail "submission reservation token differs"
fi

# This repeats the login-side submit-wrapper check after queueing.  It is a
# strict closure check: every file, directory, or symlink under shards is
# stale state, and even a dangling final symlink is an existing final target.
check_fresh_output "${output_root}"

cache_parent="${SLURM_TMPDIR:-/tmp}"
if [[ "${cache_parent}" != /* || ! -d "${cache_parent}" ]]; then
  fail "SLURM_TMPDIR/cache parent must be an absolute directory"
fi
cache_parent="$(realpath "${cache_parent}")"
cache_root="${cache_parent}/motive-r7-dino-edges-${SLURM_JOB_ID}"
reject_path_overlap \
  "${cache_root}" "compute cache" \
  "${source_snapshot}" "source snapshot"
reject_path_overlap \
  "${cache_root}" "compute cache" \
  "${input_directory}" "graph input"
reject_path_overlap \
  "${cache_root}" "compute cache" \
  "${output_root}" "output root"
reject_path_overlap \
  "${cache_root}" "compute cache" \
  "${python_bin}" "Python executable"
if [[ -e "${cache_root}" || -L "${cache_root}" ]]; then
  fail "isolated compute cache already exists: ${cache_root}"
fi
mkdir -m 0700 "${cache_root}"
mkdir \
  "${cache_root}/tmp" \
  "${cache_root}/xdg" \
  "${cache_root}/torch" \
  "${cache_root}/torch-kernels" \
  "${cache_root}/inductor" \
  "${cache_root}/triton" \
  "${cache_root}/miopen" \
  "${cache_root}/miopen-custom"
cache_root="$(realpath "${cache_root}")"

export TMPDIR="${cache_root}/tmp"
export XDG_CACHE_HOME="${cache_root}/xdg"
export TORCH_HOME="${cache_root}/torch"
export PYTORCH_KERNEL_CACHE_PATH="${cache_root}/torch-kernels"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor"
export TRITON_CACHE_DIR="${cache_root}/triton"
export MIOPEN_USER_DB_PATH="${cache_root}/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="${cache_root}/miopen-custom"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONSAFEPATH=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH="${source_snapshot}/methods/motive"
unset PYTHONHOME PYTHONSTARTUP PYTHONINSPECT
cd "${cache_root}"

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

actual_input_digest="$(
  "${python_bin}" - \
    "${source_snapshot}" \
    "${input_directory}" \
    "${cache_root}" <<'PY'
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
        "[r7-dino-edges] motive module origin differs: "
        f"expected={expected_module} actual={actual_module}"
    )
expected_cwd = Path(sys.argv[3]).resolve(strict=True)
actual_cwd = Path.cwd().resolve(strict=True)
if actual_cwd != expected_cwd:
    raise SystemExit("[r7-dino-edges] compute cwd differs")
try:
    actual_cwd.relative_to(snapshot)
except ValueError:
    pass
else:
    raise SystemExit("[r7-dino-edges] compute cwd is inside snapshot")
print(
    module.validate_graph_input(Path(sys.argv[2]))[
        "artifact_digest"
    ]
)
PY
)"
if [[ "${actual_input_digest}" != "${input_artifact_digest}" ]]; then
  echo "[r7-dino-edges] graph-input artifact digest differs" >&2
  exit 2
fi

visible_devices="$("${python_bin}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${visible_devices}" != "8" ]]; then
  echo "[r7-dino-edges] expected 8 visible GPUs, got ${visible_devices}" >&2
  exit 2
fi
check_fresh_output "${output_root}"

echo "[r7-dino-edges] host=$(hostname) job=${SLURM_JOB_ID} GPUs=${visible_devices}"
echo "[r7-dino-edges] input=${input_directory} output=${output_root}"

run_checked "${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  -m motive.r7_expansion_dino_edges extract \
  --input-directory "${input_directory}" \
  --output-root "${output_root}" \
  --block-size "${block_size}" \
  --audit-top-k "${audit_top_k}" \
  --calibration-per-stratum "${calibration_per_stratum}"

run_checked "${python_bin}" -m motive.r7_expansion_dino_edges finalize \
  --input-directory "${input_directory}" \
  --output-root "${output_root}"

"${python_bin}" - "${output_root}/final/summary.json" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
coverage = summary.get("coverage_proof", {})
assets = summary.get("input_rows")
expected = assets * (assets - 1) // 2 if isinstance(assets, int) else None
iids = assets // 2 if isinstance(assets, int) else None
expected_iid_pairs = iids * (iids - 1) // 2 if iids is not None else None
expected_quotient_partials = (
    2 * expected_iid_pairs if expected_iid_pairs is not None else None
)
if (
    summary.get("status") != "complete"
    or summary.get("rank_count") != 8
    or coverage.get("expected_compared_pairs") != expected
    or coverage.get("observed_compared_pairs") != expected
    or coverage.get("complete_upper_triangle") is not True
    or coverage.get("calibration_population_complete") is not True
    or coverage.get("thresholds_human_calibrated") is not False
    or coverage.get("training_authorized") is not False
    or summary.get("calibration_sampling", {}).get("population_count")
    != expected
    or coverage.get("expected_iid_pairs") != expected_iid_pairs
    or coverage.get("observed_iid_pairs") != expected_iid_pairs
    or coverage.get("quotient_partials_per_iid_pair") != 2
    or coverage.get("quotient_observed_partials")
    != expected_quotient_partials
    or coverage.get("quotient_coverage_complete") is not True
    or summary.get("iid_pair_maxima_rows") != expected_iid_pairs
):
    raise SystemExit("[r7-dino-edges] final coverage proof differs")
print(
    "[r7-dino-edges] complete "
    f"assets={assets} hard={summary.get('hard_edges')} "
    f"audit={summary.get('audit_edges')} "
    f"calibration={summary.get('calibration_edges')} "
    f"iid_pair_maxima={summary.get('iid_pair_maxima_rows')}",
    flush=True,
)
PY

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

echo "[r7-dino-edges] completed"
