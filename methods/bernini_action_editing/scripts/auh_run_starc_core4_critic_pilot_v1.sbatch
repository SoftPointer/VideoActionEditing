#!/usr/bin/env bash
#SBATCH --job-name=starc-critic-v1
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=00:45:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# Fit exactly the preregistered small STARC critic head on the two fit
# episodes, save/reload/freeze step 200, and evaluate the two confirmation
# episodes once.  This launcher never loads or updates the Bernini editor.

source_archive="${STARC_CRITIC_SOURCE_ARCHIVE:?set STARC_CRITIC_SOURCE_ARCHIVE}"
source_archive_sha256="${STARC_CRITIC_SOURCE_ARCHIVE_SHA256:?set STARC_CRITIC_SOURCE_ARCHIVE_SHA256}"
source_revision="${STARC_CRITIC_SOURCE_REVISION:?set STARC_CRITIC_SOURCE_REVISION}"
master_manifest="${STARC_CRITIC_MASTER_MANIFEST:?set STARC_CRITIC_MASTER_MANIFEST}"
master_manifest_sha256="${STARC_CRITIC_MASTER_MANIFEST_SHA256:?set STARC_CRITIC_MASTER_MANIFEST_SHA256}"
output_dir="${STARC_CRITIC_OUTPUT_DIR:?set STARC_CRITIC_OUTPUT_DIR}"
python_bin="${STARC_CRITIC_PYTHON_BIN:?set STARC_CRITIC_PYTHON_BIN}"

fail() { echo "[starc-critic-v1] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }
require_sha1() { [[ "$1" =~ ^[0-9a-f]{40}$ ]] || fail "$2 must be lowercase SHA-1"; }

for name in source_archive master_manifest output_dir python_bin; do
  require_absolute "${!name}" "${name}"
done
require_sha256 "${source_archive_sha256}" source_archive_sha256
require_sha256 "${master_manifest_sha256}" master_manifest_sha256
require_sha1 "${source_revision}" source_revision

source_archive="$(realpath -e -- "${source_archive}")"
master_manifest="$(realpath -e -- "${master_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive must be a plain file"
[[ -f "${master_manifest}" && ! -L "${master_manifest}" ]] || fail "master manifest must be a plain file"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python executable differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive changed"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "$(sha256sum "${master_manifest}" | awk '{print $1}')" == "${master_manifest_sha256}" ]] || fail "master manifest changed"
[[ "${output_dir}" != / && "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output dir must be canonical non-root"
[[ "${output_dir##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename differs"
[[ -d "$(dirname -- "${output_dir}")" && ! -L "$(dirname -- "${output_dir}")" ]] || fail "output parent differs"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "output dir must be fresh"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/starc-critic-v1-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/starc-critic-v1-${SLURM_JOB_ID}."*) ;;
    *) cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[starc-critic-v1] PASS fit_steps=200 confirmation_once=true editor_loaded=false output=${output_dir}"
  else
    echo "[starc-critic-v1] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -- "${task_scratch}/source-tree"
archive_copy="${task_scratch}/source.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod 0444 -- "${archive_copy}"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile

required = {
    "methods/bernini_action_editing/run_starc_core4_critic_pilot_v1.py",
    "methods/bernini_action_editing/latent_temporal_event_critic_dataset.py",
    "methods/bernini_action_editing/scripts/auh_run_starc_core4_critic_pilot_v1.sbatch",
}
seen = set()
with tarfile.open(sys.argv[1], "r:*") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        normalized = path.as_posix().lstrip("./")
        if not normalized.startswith("methods/bernini_action_editing/"):
            continue
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise SystemExit("source archive contains an unsafe method member")
        seen.add(normalized)
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"source archive lacks STARC critic closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree" \
  methods/bernini_action_editing
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
runner="${method_root}/run_starc_core4_critic_pilot_v1.py"
launcher="${method_root}/scripts/auh_run_starc_core4_critic_pilot_v1.sbatch"
[[ -f "${runner}" && ! -L "${runner}" ]] || fail "runner source differs"
[[ -f "${launcher}" && ! -L "${launcher}" ]] || fail "launcher source differs"
[[ "$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')" == "$(sha256sum "${launcher}" | awk '{print $1}')" ]] || fail "running launcher differs from source archive"
find "${task_scratch}/source-tree" -type f -exec chmod a-w -- {} +

runner_sha256="$(sha256sum "${runner}" | awk '{print $1}')"
echo "[starc-critic-v1] source_revision=${source_revision} runner_sha256=${runner_sha256} master_sha256=${master_manifest_sha256}"
echo "[starc-critic-v1] fit_episodes=2 confirmation_episodes=2 fixed_steps=200 final_checkpoint_only=true editor_loaded=false"

PYTHONPATH="${method_root}" "${python_bin}" -B "${runner}" fit-evaluate \
  --master-manifest "${master_manifest}" \
  --expected-master-sha256 "${master_manifest_sha256}" \
  --output-dir "${output_dir}" \
  --device cuda:0

for expected in \
  starc-core4-critic-config-v1.json \
  starc-core4-critic-fit-trace-v1.json \
  starc-core4-critic-final-step-0200.safetensors \
  starc-core4-critic-checkpoint-receipt-v1.json \
  starc-core4-heldout-provisional-gate-receipt-v1.json; do
  [[ -f "${output_dir}/${expected}" && ! -L "${output_dir}/${expected}" ]] || fail "missing output ${expected}"
done
output_file_count="$(find "${output_dir}" -maxdepth 1 -type f | wc -l)"
(( output_file_count == 5 )) || fail "critic output file count differs"
sha256sum "${output_dir}"/*
