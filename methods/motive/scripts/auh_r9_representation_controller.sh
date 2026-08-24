#!/usr/bin/env bash
set -Eeuo pipefail

source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?set MOTIVE_SOURCE_SNAPSHOT}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?set MOTIVE_SOURCE_TREE_SHA256}"
run_root="${MOTIVE_R9_RUN_ROOT:?set MOTIVE_R9_RUN_ROOT}"
parent_run="${MOTIVE_R7_PARENT_RUN_ROOT:?set MOTIVE_R7_PARENT_RUN_ROOT}"
model_workspace="${MOTIVE_MODEL_WORKSPACE:?set MOTIVE_MODEL_WORKSPACE}"
python_bin="${PYTHON_BIN:?set PYTHON_BIN}"
seed="${MOTIVE_R9_SEED:-260108835}"

candidate_done_sha256="${MOTIVE_R7_CANDIDATE_TEMPORAL_DONE_SHA256:?set MOTIVE_R7_CANDIDATE_TEMPORAL_DONE_SHA256}"
track_cache_done_sha256="${MOTIVE_R7_CANDIDATE_TRACK_CACHE_DONE_SHA256:?set MOTIVE_R7_CANDIDATE_TRACK_CACHE_DONE_SHA256}"
visual_features_done_sha256="${MOTIVE_R7_VISUAL_FEATURES_DONE_SHA256:?set MOTIVE_R7_VISUAL_FEATURES_DONE_SHA256}"
visual_candidates_sha256="${MOTIVE_R7_VISUAL_CANDIDATES_SHA256:?set MOTIVE_R7_VISUAL_CANDIDATES_SHA256}"
screen_inputs_receipt_sha256="${MOTIVE_R7_SCREEN_INPUTS_RECEIPT_SHA256:?set MOTIVE_R7_SCREEN_INPUTS_RECEIPT_SHA256}"

for digest in \
  "${source_tree_sha256}" \
  "${candidate_done_sha256}" \
  "${track_cache_done_sha256}" \
  "${visual_features_done_sha256}" \
  "${visual_candidates_sha256}" \
  "${screen_inputs_receipt_sha256}"; do
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[r9-controller] invalid required SHA-256: ${digest}" >&2
    exit 2
  fi
done
if [[ ! "${seed}" =~ ^[0-9]+$ ]] || (( seed >= 4294967296 )); then
  echo "[r9-controller] seed must be an integer in [0,2**32)" >&2
  exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
  echo "[r9-controller] Python is not executable: ${python_bin}" >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "[r9-controller] flock is required" >&2
  exit 2
fi

candidate_manifest_dir="${parent_run}/expansion/candidate_temporal_screen_v1"
track_cache_final="${parent_run}/expansion/candidate_track_cache_v1/final"
screen_input_bundle="${parent_run}/expansion/screen_inputs_v14"
visual_features_final="${screen_input_bundle}/visual_features_v1/final"
visual_candidates_manifest="${screen_input_bundle}/visual_candidates_v1/candidates.jsonl"
screen_inputs_receipt="${parent_run}/provenance/screen_inputs_v14.receipt.txt"
baseline_output="${run_root}/representation/baseline_screen"
search_output="${run_root}/representation/search_seed_${seed}"
registry_path="${source_snapshot}/methods/motive/configs/instruction_video_editor_registry_v1.json"
availability_output="${run_root}/renderer_registry/availability_before_representation_gate.json"

for required_dir in \
  "${source_snapshot}" \
  "${candidate_manifest_dir}" \
  "${track_cache_final}" \
  "${screen_input_bundle}" \
  "${visual_features_final}" \
  "$(dirname "${visual_candidates_manifest}")" \
  "${model_workspace}"; do
  if [[ ! -d "${required_dir}" ]] || [[ -L "${required_dir}" ]]; then
    echo "[r9-controller] missing/symlinked directory: ${required_dir}" >&2
    exit 2
  fi
done
for required_file in \
  "${candidate_manifest_dir}/done.json" \
  "${track_cache_final}/done.json" \
  "${visual_features_final}/done.json" \
  "${visual_candidates_manifest}" \
  "${screen_inputs_receipt}" \
  "${registry_path}" \
  "${source_snapshot}/methods/motive/motive/r9_automated_representation_search.py"; do
  if [[ ! -s "${required_file}" ]] || [[ -L "${required_file}" ]]; then
    echo "[r9-controller] missing/empty/symlinked file: ${required_file}" >&2
    exit 2
  fi
done
if [[ "$(stat -c '%h' "${screen_inputs_receipt}")" != "1" ]] \
  || [[ "$(stat -c '%a' "${screen_inputs_receipt}")" != "444" ]]; then
  echo "[r9-controller] sealed screen-input receipt mode differs" >&2
  exit 2
fi

mkdir -p \
  "${run_root}/representation" \
  "${run_root}/renderer_registry" \
  "${run_root}/logs" \
  "${run_root}/provenance"
exec 9>"${run_root}/.controller.lock"
if ! flock -n 9; then
  echo "[r9-controller] another controller owns this run" >&2
  exit 4
fi

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

actual_candidate_done_sha256="$(sha256sum "${candidate_manifest_dir}/done.json" | awk '{print $1}')"
actual_track_cache_done_sha256="$(sha256sum "${track_cache_final}/done.json" | awk '{print $1}')"
actual_visual_features_done_sha256="$(sha256sum "${visual_features_final}/done.json" | awk '{print $1}')"
actual_visual_candidates_sha256="$(
  sha256sum "${visual_candidates_manifest}" | awk '{print $1}'
)"
actual_screen_inputs_receipt_sha256="$(
  sha256sum "${screen_inputs_receipt}" | awk '{print $1}'
)"
if [[ "${actual_candidate_done_sha256}" != "${candidate_done_sha256}" ]] \
  || [[ "${actual_track_cache_done_sha256}" != "${track_cache_done_sha256}" ]] \
  || [[ "${actual_visual_features_done_sha256}" != "${visual_features_done_sha256}" ]] \
  || [[ "${actual_visual_candidates_sha256}" != "${visual_candidates_sha256}" ]] \
  || [[ "${actual_screen_inputs_receipt_sha256}" != "${screen_inputs_receipt_sha256}" ]]; then
  echo "[r9-controller] an upstream commit SHA-256 differs" >&2
  exit 2
fi

export PYTHONPATH="${source_snapshot}/methods/motive"
export PYTHONDONTWRITEBYTECODE=1
actual_module="$(
  "${python_bin}" -c '
from pathlib import Path
import motive.r9_automated_representation_search as module
print(Path(module.__file__).resolve(strict=True))
'
)"
expected_module="${source_snapshot}/methods/motive/motive/r9_automated_representation_search.py"
if [[ "${actual_module}" != "${expected_module}" ]]; then
  echo "[r9-controller] imported search module is not frozen" >&2
  exit 2
fi

"${python_bin}" -c '
from pathlib import Path
from motive import r7_artifact_permissions as permissions
import sys
for raw in sys.argv[1:]:
    permissions.assert_sealed_tree(Path(raw))
' \
  "${screen_input_bundle}" \
  "${visual_features_final}" \
  "$(dirname "${visual_candidates_manifest}")"

if [[ ! -e "${availability_output}" ]]; then
  "${python_bin}" -m motive.instruction_model_registry \
    --registry "${registry_path}" \
    --workspace "${model_workspace}" \
    --output "${availability_output}"
fi

common_arguments=(
  --candidate-manifest-dir "${candidate_manifest_dir}"
  --expected-candidate-manifest-done-sha256 "${candidate_done_sha256}"
  --track-cache-final "${track_cache_final}"
  --expected-track-cache-done-sha256 "${track_cache_done_sha256}"
  --visual-features-final "${visual_features_final}"
  --expected-visual-features-done-sha256 "${visual_features_done_sha256}"
  --visual-candidates-manifest "${visual_candidates_manifest}"
  --expected-visual-candidates-sha256 "${visual_candidates_sha256}"
)

if [[ ! -e "${search_output}" ]] && [[ ! -L "${search_output}" ]]; then
  "${python_bin}" -m motive.r9_automated_representation_search \
    "${common_arguments[@]}" \
    --output-dir "${search_output}" \
    --seed "${seed}"
fi
"${python_bin}" -c '
from pathlib import Path
from motive.r9_automated_representation_search import validate_published_search
import sys
validate_published_search(Path(sys.argv[1]))
' "${search_output}"

if [[ -e "${baseline_output}" ]] || [[ -L "${baseline_output}" ]]; then
  "${python_bin}" -m motive.r7_candidate_temporal_screen \
    "${common_arguments[@]}" \
    --output-dir "${baseline_output}" \
    --seed "${seed}" \
    --resume
else
  "${python_bin}" -m motive.r7_candidate_temporal_screen \
    "${common_arguments[@]}" \
    --output-dir "${baseline_output}" \
    --seed "${seed}"
fi

echo "[r9-controller] completed search=${search_output} baseline=${baseline_output}"
