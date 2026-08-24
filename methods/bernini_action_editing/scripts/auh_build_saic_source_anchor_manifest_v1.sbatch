#!/usr/bin/env bash
#SBATCH --job-name=saic-anchor-manifest
#SBATCH --partition=faculty
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# CPU-only pre-step.  Choose QOS at submission time; no GPU is requested.
# Example: sbatch --qos=gtqos .../auh_build_saic_source_anchor_manifest_v1.sbatch

set -Eeuo pipefail
umask 027

fail() { echo "[saic-anchor-manifest] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }

source_archive="${SAIC_SOURCE_ARCHIVE:?set SAIC_SOURCE_ARCHIVE}"
source_archive_sha256="${SAIC_SOURCE_ARCHIVE_SHA256:?set SAIC_SOURCE_ARCHIVE_SHA256}"
source_revision="${SAIC_SOURCE_REVISION:?set SAIC_SOURCE_REVISION}"
samples_root="${SAIC_V17_SAMPLES_ROOT:?set SAIC_V17_SAMPLES_ROOT to wan_next1000_v17/samples}"
output="${SAIC_SOURCE_ANCHOR_MANIFEST_OUTPUT:?set a fresh manifest JSON path}"
python_bin="${SAIC_PYTHON_BIN:?set SAIC_PYTHON_BIN to vace Python}"
seed="${SAIC_SOURCE_ANCHOR_SELECTION_SEED:-20260809}"

for name in source_archive samples_root output python_bin; do require_absolute "${!name}" "${name}"; done
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source archive SHA-256 differs"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${seed}" =~ ^[0-9]+$ ]] || fail "selection seed must be decimal"
source_archive="$(realpath -e -- "${source_archive}")"
samples_root="$(realpath -e -- "${samples_root}")"
python_bin="$(realpath -e -- "${python_bin}")"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive differs"
[[ -d "${samples_root}" && ! -L "${samples_root}" ]] || fail "samples root differs"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "Python differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "${output}" != / && "${output}" == *.json ]] || fail "output must be a non-root JSON path"
[[ "$(realpath -m -- "${output}")" == "${output}" ]] || fail "output must be canonical"
[[ ! -e "${output}" && ! -L "${output}" ]] || fail "manifest output is create-only"
mkdir -p -- "$(dirname -- "${output}")"
[[ -d "$(dirname -- "${output}")" && ! -L "$(dirname -- "${output}")" ]] || fail "output parent differs"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES=""
[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/saic-anchor-manifest-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in "${scratch_parent%/}/saic-anchor-manifest-${SLURM_JOB_ID}."*) ;; *) exit 2 ;; esac
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || status=2
    rm -rf -- "${task_scratch}" || status=2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -- "${task_scratch}/source"
tar --no-same-owner --no-same-permissions -xf "${source_archive}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
for required in \
  infer_lora.py \
  train_lora.py \
  tools/build_saic_source_anchor_manifest_v1.py \
  tools/materialize_vae.py \
  tests/test_build_saic_source_anchor_manifest_v1.py; do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive lacks ${required}"
done
find "${method_root}" -type f -exec chmod a-w -- {} +

PYTHONPATH="${method_root}" "${python_bin}" -B \
  "${method_root}/tests/test_build_saic_source_anchor_manifest_v1.py"
PYTHONPATH="${method_root}" "${python_bin}" -B \
  "${method_root}/tools/build_saic_source_anchor_manifest_v1.py" \
  --samples-root "${samples_root}" \
  --output "${output}" \
  --seed "${seed}"

[[ -f "${output}" && ! -L "${output}" ]] || fail "manifest was not created"
chmod 0444 -- "${output}"
manifest_sha256="$(sha256sum "${output}" | awk '{print $1}')"
"${python_bin}" -B - "${output}" <<'PY'
import json
from pathlib import Path
import sys
value = json.loads(Path(sys.argv[1]).read_text("ascii"))
if (
    value.get("train_count") != 64
    or value.get("holdout_count") != 16
    or value.get("holdout_used_by_optimizer") is not False
    or value.get("optimizer_authorized") is not False
):
    raise SystemExit("manifest postflight closure differs")
PY
echo "[saic-anchor-manifest] PASS output=${output}"
echo "export SAIC_SOURCE_ANCHOR_MANIFEST=${output}"
echo "export SAIC_SOURCE_ANCHOR_MANIFEST_SHA256=${manifest_sha256}"
