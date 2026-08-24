#!/usr/bin/env bash
#SBATCH --job-name=saic-t2v-review60
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/logs/saic-t2v-review60-%j.out
#SBATCH --error=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/logs/saic-t2v-review60-%j.err

set -Eeuo pipefail

source_archive="${SAIC_T2V_REVIEW_SOURCE_ARCHIVE:?set frozen git source archive}"
source_archive_sha256="${SAIC_T2V_REVIEW_SOURCE_ARCHIVE_SHA256:?set archive SHA-256}"
source_revision="${SAIC_T2V_REVIEW_SOURCE_REVISION:?set archive git revision}"
input_root="${SAIC_T2V_REVIEW_INPUT_ROOT:?set completed Job132254 root}"
output_root="${SAIC_T2V_REVIEW_OUTPUT_ROOT:?set fresh detached audit root}"
python_bin="${SAIC_T2V_REVIEW_PYTHON_BIN:?set vace Python executable}"
ffmpeg_bin="${SAIC_T2V_REVIEW_FFMPEG_BIN:?set AUH static ffmpeg executable}"
ffmpeg_sha256="${SAIC_T2V_REVIEW_FFMPEG_SHA256:?set ffmpeg SHA-256}"
ffprobe_wrapper_sha256="${SAIC_T2V_REVIEW_FFPROBE_WRAPPER_SHA256:?set archive-bound ffprobe adapter SHA-256}"
workers="${SAIC_T2V_REVIEW_WORKERS:-16}"

fail() {
  echo "[saic-t2v-review60] ERROR: $*" >&2
  exit 2
}

for value in "${source_archive}" "${input_root}" "${output_root}" "${python_bin}" "${ffmpeg_bin}"; do
  [[ "${value}" == /* ]] || fail "all paths must be absolute"
done
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "archive SHA-256 differs"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "archive revision differs"
[[ "${ffmpeg_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "ffmpeg SHA-256 differs"
[[ "${ffprobe_wrapper_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "ffprobe adapter SHA-256 differs"
[[ "${workers}" =~ ^[0-9]+$ ]] && (( workers >= 1 && workers <= 32 )) || fail "workers must be in [1,32]"

source_archive="$(realpath -e -- "${source_archive}")"
input_root="$(realpath -e -- "${input_root}")"
python_bin="$(realpath -e -- "${python_bin}")"
ffmpeg_bin="$(realpath -e -- "${ffmpeg_bin}")"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "archive must be plain"
[[ -d "${input_root}" && ! -L "${input_root}" ]] || fail "input root must be plain"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "Python executable differs"
[[ -x "${ffmpeg_bin}" && -f "${ffmpeg_bin}" && ! -L "${ffmpeg_bin}" ]] || fail "ffmpeg executable differs"
[[ "$(sha256sum "${ffmpeg_bin}" | awk '{print $1}')" == "${ffmpeg_sha256}" ]] || fail "ffmpeg changed"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "archive changed"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "archive revision changed"
[[ -f "${input_root}/saic-pure-t2v-event-bank-receipt.json" && ! -L "${input_root}/saic-pure-t2v-event-bank-receipt.json" ]] || fail "Job132254 master receipt absent"

[[ "${output_root}" != / && ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"
output_parent="$(dirname -- "${output_root}")"
mkdir -p -- "${output_parent}"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"
output_parent="$(realpath -e -- "${output_parent}")"
[[ "${output_root}" == "${output_parent}/${output_root##*/}" ]] || fail "output root must be canonical"
[[ "${output_root##*/}" == "t2v-events-v1-533074b-r3-detached-review-v1" ]] || fail "output packet id differs"

[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "Slurm job id differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/saic-t2v-review60-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/saic-t2v-review60-${SLURM_JOB_ID}."*) ;;
    *) cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[saic-t2v-review60] PASS candidates=60 exact81=true transitions=80 machine_authority=zero semantic=UNASSESSED event_verified=false optimizer=false output=${output_root}"
  else
    echo "[saic-t2v-review60] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false) output=${output_root}" >&2
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
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
motive_root="${task_scratch}/source-tree/methods/motive"

required=(
  "${method_root}/tools/build_saic_t2v_event_bank_detached_review_v1.py"
  "${method_root}/tools/ffprobe_pyav_exact81_diagnostic_v1.py"
  "${method_root}/saic_exact81_media_diagnostics_v1.py"
  "${method_root}/decoded_temporal_event_evaluator_v1.py"
  "${method_root}/saic_pure_t2v_event_bank_v1.py"
  "${method_root}/build_saic_reversible_source_set_v1.py"
  "${method_root}/assets/saic_reversible_source_set_v1.json"
  "${method_root}/assets/saic_pure_t2v_event_bank_v1.json"
  "${method_root}/tests/test_build_saic_t2v_event_bank_detached_review_v1.py"
  "${method_root}/tests/test_saic_exact81_media_diagnostics_v1.py"
  "${method_root}/tests/test_auh_build_saic_t2v_event_bank_detached_review_v1_cpu_launcher.py"
  "${method_root}/tests/test_saic_detached_review_clean_tool_visibility_v1.py"
  "${motive_root}/motive/geometry.py"
  "${motive_root}/motive/descriptor.py"
  "${motive_root}/motive/selection.py"
  "${motive_root}/motive/semantics.py"
)
for path in "${required[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "archive closure absent: ${path}"
done
find "${task_scratch}/source-tree" -type f -exec chmod a-w -- {} +

tool_bin="${task_scratch}/tool-bin"
mkdir -- "${tool_bin}"
ln -s -- "${ffmpeg_bin}" "${tool_bin}/ffmpeg"
[[ -L "${tool_bin}/ffmpeg" && "$(realpath -e -- "${tool_bin}/ffmpeg")" == "${ffmpeg_bin}" ]] || fail "runtime ffmpeg binding differs"
ffprobe_source="${method_root}/tools/ffprobe_pyav_exact81_diagnostic_v1.py"
[[ "$(sha256sum "${ffprobe_source}" | awk '{print $1}')" == "${ffprobe_wrapper_sha256}" ]] || fail "archive-bound ffprobe adapter changed"
cp -- "${ffprobe_source}" "${tool_bin}/ffprobe"
chmod 0555 -- "${tool_bin}/ffprobe"
ffprobe_bin="$(realpath -e -- "${tool_bin}/ffprobe")"
[[ "$(sha256sum "${ffprobe_bin}" | awk '{print $1}')" == "${ffprobe_wrapper_sha256}" ]] || fail "runtime ffprobe adapter changed"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${method_root}:${motive_root}"
export SAIC_T2V_REVIEW_PYTHON_BIN="${python_bin}"
export PATH="${tool_bin}:/usr/bin:/bin"
export SAIC_T2V_REVIEW_EXPECTED_FFMPEG="${ffmpeg_bin}"
export SAIC_T2V_REVIEW_EXPECTED_FFPROBE="${ffprobe_bin}"
export SAIC_T2V_REVIEW_FFMPEG_SHA256="${ffmpeg_sha256}"
export SAIC_T2V_REVIEW_FFPROBE_SHA256="${ffprobe_wrapper_sha256}"
export SAIC_T2V_REVIEW_CLEAN_PATH="${PATH}"
[[ "$(realpath -e -- "$(command -v ffmpeg)")" == "${ffmpeg_bin}" ]] || fail "clean PATH ffmpeg identity differs"
[[ "$(command -v ffprobe)" == "${ffprobe_bin}" ]] || fail "clean PATH ffprobe identity differs"
ffmpeg -version >/dev/null
ffprobe -version >/dev/null

"${python_bin}" -B "${method_root}/tests/test_build_saic_t2v_event_bank_detached_review_v1.py"
"${python_bin}" -B "${method_root}/tests/test_saic_exact81_media_diagnostics_v1.py"
"${python_bin}" -B "${method_root}/tests/test_auh_build_saic_t2v_event_bank_detached_review_v1_cpu_launcher.py"
"${python_bin}" -B "${method_root}/tests/test_saic_detached_review_clean_tool_visibility_v1.py"

"${python_bin}" -B "${method_root}/tools/build_saic_t2v_event_bank_detached_review_v1.py" build \
  --input-root "${input_root}" \
  --output-root "${output_root}" \
  --job-id "${SLURM_JOB_ID}" \
  --workers "${workers}"

[[ -f "${output_root}/detached-review-receipt.json" && ! -L "${output_root}/detached-review-receipt.json" ]] || fail "final receipt absent"
[[ -f "${output_root}/review-manifest.json" && ! -L "${output_root}/review-manifest.json" ]] || fail "sealed review manifest absent"
[[ -f "${output_root}/index.html" && ! -L "${output_root}/index.html" ]] || fail "HTML review absent"
[[ -f "${output_root}/blind-review.html" && ! -L "${output_root}/blind-review.html" ]] || fail "blind human review page absent"
[[ -f "${output_root}/observer-protocol.json" && ! -L "${output_root}/observer-protocol.json" ]] || fail "observer protocol absent"
[[ "$(find "${output_root}/diagnostics" -maxdepth 1 -type f -name '*.json' | wc -l)" -eq 60 ]] || fail "diagnostic closure differs"
[[ "$(find "${output_root}/observer-templates" -maxdepth 1 -type f -name '*.json' | wc -l)" -eq 2 ]] || fail "observer template closure differs"
"${python_bin}" -B "${method_root}/tools/build_saic_t2v_event_bank_detached_review_v1.py" validate \
  --output-root "${output_root}"
