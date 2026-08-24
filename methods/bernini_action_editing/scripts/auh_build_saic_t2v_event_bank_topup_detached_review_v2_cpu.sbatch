#!/usr/bin/env bash
#SBATCH --job-name=saic-topup-review-v2
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=08:00:00

# AUH's available QOS requires one GPU even though this job is CPU-only.  The
# runtime clears every GPU visibility variable before importing review code.

set -Eeuo pipefail
umask 027

fail() { echo "[saic-topup-review-v2] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }

source_archive="${SAIC_T2V_TOPUP_REVIEW_SOURCE_ARCHIVE:?}"
source_archive_sha256="${SAIC_T2V_TOPUP_REVIEW_SOURCE_ARCHIVE_SHA256:?}"
source_revision="${SAIC_T2V_TOPUP_REVIEW_SOURCE_REVISION:?}"
adapter="${SAIC_T2V_TOPUP_REVIEW_ADAPTER:?}"
adapter_sha256="${SAIC_T2V_TOPUP_REVIEW_ADAPTER_SHA256:?}"
input_root="${SAIC_T2V_TOPUP_REVIEW_INPUT_ROOT:?}"
output_root="${SAIC_T2V_TOPUP_REVIEW_OUTPUT_ROOT:?}"
python_bin="${SAIC_T2V_TOPUP_REVIEW_PYTHON_BIN:?}"
ffmpeg_bin="${SAIC_T2V_TOPUP_REVIEW_FFMPEG_BIN:?}"
ffmpeg_sha256="${SAIC_T2V_TOPUP_REVIEW_FFMPEG_SHA256:?}"
ffprobe_wrapper_sha256="${SAIC_T2V_TOPUP_REVIEW_FFPROBE_WRAPPER_SHA256:?}"
workers="${SAIC_T2V_TOPUP_REVIEW_WORKERS:-16}"

for name in source_archive adapter input_root output_root python_bin ffmpeg_bin; do
  require_absolute "${!name}" "${name}"
done
for value in "${source_archive_sha256}" "${adapter_sha256}" \
  "${ffmpeg_sha256}" "${ffprobe_wrapper_sha256}"; do
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || fail "SHA-256 input differs"
done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${workers}" =~ ^[0-9]+$ ]] && (( workers >= 1 && workers <= 32 )) || \
  fail "workers must be in [1,32]"

source_archive="$(realpath -e -- "${source_archive}")"
adapter="$(realpath -e -- "${adapter}")"
input_root="$(realpath -e -- "${input_root}")"
python_bin="$(realpath -e -- "${python_bin}")"
ffmpeg_bin="$(realpath -e -- "${ffmpeg_bin}")"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive differs"
[[ -f "${adapter}" && ! -L "${adapter}" ]] || fail "review adapter differs"
[[ -d "${input_root}" && ! -L "${input_root}" ]] || fail "input root differs"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "Python differs"
[[ -x "${ffmpeg_bin}" && -f "${ffmpeg_bin}" && ! -L "${ffmpeg_bin}" ]] || fail "ffmpeg differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "$(sha256sum "${adapter}" | awk '{print $1}')" == "${adapter_sha256}" ]] || fail "review adapter hash differs"
[[ "$(sha256sum "${ffmpeg_bin}" | awk '{print $1}')" == "${ffmpeg_sha256}" ]] || fail "ffmpeg hash differs"
[[ -f "${input_root}/saic-pure-t2v-event-bank-topup-receipt.json" ]] || fail "top-up master receipt is absent"

[[ "${output_root}" != / && ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"
output_parent="$(dirname -- "${output_root}")"
mkdir -p -- "${output_parent}"
output_parent="$(realpath -e -- "${output_parent}")"
[[ "${output_root}" == "${output_parent}/${output_root##*/}" ]] || fail "output root must be canonical"

[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "Slurm JobID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/saic-topup-review-v2-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/saic-topup-review-v2-${SLURM_JOB_ID}."*) ;;
    *) cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[saic-topup-review-v2] PASS candidates=60 diagnostics=60 semantic=UNASSESSED machine_authority=zero packet=${output_root}"
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -- "${task_scratch}/source-tree"
cp -- "${source_archive}" "${task_scratch}/source.tar"
chmod 0444 -- "${task_scratch}/source.tar"
tar --no-same-owner --no-same-permissions -xf "${task_scratch}/source.tar" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
motive_root="${task_scratch}/source-tree/methods/motive"
runtime_adapter="${method_root}/tools/build_saic_t2v_event_bank_topup_detached_review_v2.py"
[[ ! -e "${runtime_adapter}" && ! -L "${runtime_adapter}" ]] || fail "archive unexpectedly contains review adapter"
cp -- "${adapter}" "${runtime_adapter}"
chmod 0444 -- "${runtime_adapter}"

required=(
  "${runtime_adapter}"
  "${method_root}/tools/build_saic_t2v_event_bank_detached_review_v1.py"
  "${method_root}/tools/ffprobe_pyav_exact81_diagnostic_v1.py"
  "${method_root}/saic_exact81_media_diagnostics_v1.py"
  "${method_root}/decoded_temporal_event_evaluator_v1.py"
  "${method_root}/generate_saic_pure_t2v_event_bank_topup_v2.py"
  "${method_root}/generate_saic_pure_t2v_event_bank_v1.py"
  "${method_root}/saic_pure_t2v_event_bank_topup_v2.py"
  "${method_root}/saic_pure_t2v_event_bank_v1.py"
  "${method_root}/build_saic_reversible_source_set_v1.py"
  "${method_root}/assets/saic_reversible_source_set_v1.json"
  "${method_root}/assets/saic_pure_t2v_event_bank_v1.json"
  "${method_root}/assets/saic_pure_t2v_event_bank_topup_v2.json"
  "${motive_root}/motive/geometry.py"
  "${motive_root}/motive/descriptor.py"
  "${motive_root}/motive/selection.py"
  "${motive_root}/motive/semantics.py"
)
for path in "${required[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "review runtime closure absent: ${path}"
done
find "${task_scratch}/source-tree" -type f -exec chmod a-w -- {} +

tool_bin="${task_scratch}/tool-bin"
mkdir -- "${tool_bin}"
ln -s -- "${ffmpeg_bin}" "${tool_bin}/ffmpeg"
ffprobe_source="${method_root}/tools/ffprobe_pyav_exact81_diagnostic_v1.py"
[[ "$(sha256sum "${ffprobe_source}" | awk '{print $1}')" == "${ffprobe_wrapper_sha256}" ]] || fail "ffprobe wrapper differs"
cp -- "${ffprobe_source}" "${tool_bin}/ffprobe"
chmod 0555 -- "${tool_bin}/ffprobe"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="${task_scratch}/pycache"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${method_root}/tools:${method_root}:${motive_root}"
export SAIC_T2V_REVIEW_PYTHON_BIN="${python_bin}"
export PATH="${tool_bin}:/usr/bin:/bin"
export SAIC_T2V_REVIEW_EXPECTED_FFMPEG="${ffmpeg_bin}"
export SAIC_T2V_REVIEW_EXPECTED_FFPROBE="${tool_bin}/ffprobe"
export SAIC_T2V_REVIEW_FFMPEG_SHA256="${ffmpeg_sha256}"
export SAIC_T2V_REVIEW_FFPROBE_SHA256="${ffprobe_wrapper_sha256}"
export SAIC_T2V_REVIEW_CLEAN_PATH="${PATH}"
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES=""

"${python_bin}" -B "${method_root}/tests/test_build_saic_t2v_event_bank_detached_review_v1.py"
"${python_bin}" -B "${method_root}/tests/test_saic_exact81_media_diagnostics_v1.py"
"${python_bin}" -B "${runtime_adapter}" build \
  --input-root "${input_root}" \
  --output-root "${output_root}" \
  --job-id "${SLURM_JOB_ID}" \
  --workers "${workers}"
"${python_bin}" -B "${runtime_adapter}" validate --output-root "${output_root}"

[[ -f "${output_root}/detached-review-receipt.json" ]] || fail "review receipt absent"
[[ -f "${output_root}/review-manifest.json" ]] || fail "review manifest absent"
[[ -f "${output_root}/index.html" && -f "${output_root}/blind-review.html" ]] || fail "review HTML absent"
[[ "$(find "${output_root}/diagnostics" -maxdepth 1 -type f -name '*.json' | wc -l)" == 60 ]] || fail "diagnostic count differs"
