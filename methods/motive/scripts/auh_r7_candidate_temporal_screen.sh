#!/usr/bin/env bash
set -Eeuo pipefail

source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?set MOTIVE_SOURCE_SNAPSHOT}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?set MOTIVE_SOURCE_TREE_SHA256}"
candidate_manifest_dir="${MOTIVE_R7_CANDIDATE_TEMPORAL_INPUT:?set MOTIVE_R7_CANDIDATE_TEMPORAL_INPUT}"
candidate_done_sha256="${MOTIVE_R7_CANDIDATE_TEMPORAL_DONE_SHA256:?set MOTIVE_R7_CANDIDATE_TEMPORAL_DONE_SHA256}"
track_cache_final="${MOTIVE_R7_CANDIDATE_TRACK_CACHE_FINAL:?set MOTIVE_R7_CANDIDATE_TRACK_CACHE_FINAL}"
track_cache_done_sha256="${MOTIVE_R7_CANDIDATE_TRACK_CACHE_DONE_SHA256:?set MOTIVE_R7_CANDIDATE_TRACK_CACHE_DONE_SHA256}"
visual_features_final="${MOTIVE_R7_VISUAL_FEATURES_FINAL:?set MOTIVE_R7_VISUAL_FEATURES_FINAL}"
visual_features_done_sha256="${MOTIVE_R7_VISUAL_FEATURES_DONE_SHA256:?set MOTIVE_R7_VISUAL_FEATURES_DONE_SHA256}"
visual_candidates_manifest="${MOTIVE_R7_VISUAL_CANDIDATES_MANIFEST:?set MOTIVE_R7_VISUAL_CANDIDATES_MANIFEST}"
visual_candidates_sha256="${MOTIVE_R7_VISUAL_CANDIDATES_SHA256:?set MOTIVE_R7_VISUAL_CANDIDATES_SHA256}"
output_dir="${MOTIVE_R7_CANDIDATE_TEMPORAL_SCREEN_OUTPUT:?set MOTIVE_R7_CANDIDATE_TEMPORAL_SCREEN_OUTPUT}"
python_bin="${PYTHON_BIN:?set PYTHON_BIN}"
seed="${MOTIVE_R7_SCREEN_SEED:-260108835}"
resume="${MOTIVE_R7_SCREEN_RESUME:-0}"

for digest in \
  "${source_tree_sha256}" \
  "${candidate_done_sha256}" \
  "${track_cache_done_sha256}" \
  "${visual_features_done_sha256}" \
  "${visual_candidates_sha256}"; do
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[r7-candidate-screen] invalid required SHA-256: ${digest}" >&2
    exit 2
  fi
done
if [[ ! "${seed}" =~ ^[0-9]+$ ]] || (( seed >= 4294967296 )); then
  echo "[r7-candidate-screen] seed must be an integer in [0,2**32)" >&2
  exit 2
fi
if [[ "${resume}" != "0" ]] && [[ "${resume}" != "1" ]]; then
  echo "[r7-candidate-screen] MOTIVE_R7_SCREEN_RESUME must be 0 or 1" >&2
  exit 2
fi
if [[ ! -f "${python_bin}" ]] || [[ ! -x "${python_bin}" ]]; then
  echo "[r7-candidate-screen] Python is not executable: ${python_bin}" >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "[r7-candidate-screen] flock is required" >&2
  exit 2
fi

for required_dir in \
  "${source_snapshot}" \
  "${candidate_manifest_dir}" \
  "${track_cache_final}" \
  "${visual_features_final}"; do
  if [[ ! -d "${required_dir}" ]] || [[ -L "${required_dir}" ]]; then
    echo "[r7-candidate-screen] required directory is missing/symlinked: ${required_dir}" >&2
    exit 2
  fi
done
for required_file in \
  "${candidate_manifest_dir}/done.json" \
  "${track_cache_final}/done.json" \
  "${visual_features_final}/done.json" \
  "${visual_candidates_manifest}" \
  "${source_snapshot}/methods/motive/motive/r7_candidate_temporal_screen.py"; do
  if [[ ! -f "${required_file}" ]] || [[ ! -s "${required_file}" ]] || [[ -L "${required_file}" ]]; then
    echo "[r7-candidate-screen] required file is missing/empty/symlinked: ${required_file}" >&2
    exit 2
  fi
done

if [[ "${resume}" == "0" ]]; then
  if [[ -e "${output_dir}" ]] || [[ -L "${output_dir}" ]]; then
    echo "[r7-candidate-screen] create-only output already exists: ${output_dir}" >&2
    exit 3
  fi
elif [[ ! -d "${output_dir}" ]] || [[ -L "${output_dir}" ]]; then
  echo "[r7-candidate-screen] strict resume requires a real output directory" >&2
  exit 3
fi

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

actual_candidate_done_sha256="$(sha256sum "${candidate_manifest_dir}/done.json" | awk '{print $1}')"
actual_track_cache_done_sha256="$(sha256sum "${track_cache_final}/done.json" | awk '{print $1}')"
actual_visual_features_done_sha256="$(sha256sum "${visual_features_final}/done.json" | awk '{print $1}')"
actual_visual_candidates_sha256="$(sha256sum "${visual_candidates_manifest}" | awk '{print $1}')"
if [[ "${actual_candidate_done_sha256}" != "${candidate_done_sha256}" ]] \
  || [[ "${actual_track_cache_done_sha256}" != "${track_cache_done_sha256}" ]] \
  || [[ "${actual_visual_features_done_sha256}" != "${visual_features_done_sha256}" ]] \
  || [[ "${actual_visual_candidates_sha256}" != "${visual_candidates_sha256}" ]]; then
  echo "[r7-candidate-screen] an external input SHA-256 differs" >&2
  exit 2
fi

export PYTHONPATH="${source_snapshot}/methods/motive"
export PYTHONDONTWRITEBYTECODE=1
actual_module="$(
  "${python_bin}" -c '
from pathlib import Path
import motive.r7_candidate_temporal_screen as screen
print(Path(screen.__file__).resolve(strict=True))
'
)"
expected_module="${source_snapshot}/methods/motive/motive/r7_candidate_temporal_screen.py"
if [[ "${actual_module}" != "${expected_module}" ]]; then
  echo "[r7-candidate-screen] imported screen module is not frozen" >&2
  exit 2
fi

mkdir -p "$(dirname "${output_dir}")"
exec 9>"${output_dir}.screen.lock"
if ! flock -n 9; then
  echo "[r7-candidate-screen] another screen writer holds the output lock" >&2
  exit 4
fi

arguments=(
  --candidate-manifest-dir "${candidate_manifest_dir}"
  --expected-candidate-manifest-done-sha256 "${candidate_done_sha256}"
  --track-cache-final "${track_cache_final}"
  --expected-track-cache-done-sha256 "${track_cache_done_sha256}"
  --visual-features-final "${visual_features_final}"
  --expected-visual-features-done-sha256 "${visual_features_done_sha256}"
  --visual-candidates-manifest "${visual_candidates_manifest}"
  --expected-visual-candidates-sha256 "${visual_candidates_sha256}"
  --output-dir "${output_dir}"
  --seed "${seed}"
)
if [[ "${resume}" == "1" ]]; then
  arguments+=(--resume)
fi

"${python_bin}" -m motive.r7_candidate_temporal_screen "${arguments[@]}"
