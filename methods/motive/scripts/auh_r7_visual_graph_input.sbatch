#!/usr/bin/env bash
#SBATCH --job-name=m7-graphinput
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -Eeuo pipefail

source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?set MOTIVE_SOURCE_SNAPSHOT}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?set MOTIVE_SOURCE_TREE_SHA256}"
candidate_features="${MOTIVE_R7_CANDIDATE_FEATURES:?set MOTIVE_R7_CANDIDATE_FEATURES}"
candidate_feature_done_sha256="${MOTIVE_R7_CANDIDATE_FEATURE_DONE_SHA256:?set MOTIVE_R7_CANDIDATE_FEATURE_DONE_SHA256}"
candidate_manifest="${MOTIVE_R7_CANDIDATE_MANIFEST:?set MOTIVE_R7_CANDIDATE_MANIFEST}"
candidate_manifest_sha256="${MOTIVE_R7_CANDIDATE_MANIFEST_SHA256:?set MOTIVE_R7_CANDIDATE_MANIFEST_SHA256}"
anchor_features="${MOTIVE_R7_ANCHOR_FEATURES:?set MOTIVE_R7_ANCHOR_FEATURES}"
anchor_feature_done_sha256="${MOTIVE_R7_ANCHOR_FEATURE_DONE_SHA256:?set MOTIVE_R7_ANCHOR_FEATURE_DONE_SHA256}"
anchor_manifest="${MOTIVE_R7_ANCHOR_MANIFEST:?set MOTIVE_R7_ANCHOR_MANIFEST}"
anchor_manifest_sha256="${MOTIVE_R7_ANCHOR_MANIFEST_SHA256:?set MOTIVE_R7_ANCHOR_MANIFEST_SHA256}"
output_dir="${MOTIVE_R7_GRAPH_INPUT_OUTPUT:?set MOTIVE_R7_GRAPH_INPUT_OUTPUT}"
expected_assets="${MOTIVE_R7_GRAPH_INPUT_EXPECTED_ASSETS:?set MOTIVE_R7_GRAPH_INPUT_EXPECTED_ASSETS}"
expected_iids="${MOTIVE_R7_GRAPH_INPUT_EXPECTED_IIDS:?set MOTIVE_R7_GRAPH_INPUT_EXPECTED_IIDS}"
python_bin="${PYTHON_BIN:?set PYTHON_BIN}"

for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${source_snapshot}/methods/motive/motive/r7_visual_graph_input.py" \
  "${source_snapshot}/methods/motive/motive/r7_expansion_visual_features.py" \
  "${source_snapshot}/methods/motive/motive/r7_preflight_extract.py" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  "${candidate_features}/done.json" \
  "${candidate_manifest}" \
  "${anchor_features}/done.json" \
  "${anchor_manifest}" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[r7-graph-input] missing required path: ${required}" >&2
    exit 2
  fi
done

if [[ -e "${output_dir}" || -L "${output_dir}" ]]; then
  echo "[r7-graph-input] refusing existing output: ${output_dir}" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${source_snapshot}/methods/motive"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

"${python_bin}" - \
  "${candidate_features}/done.json" \
  "${candidate_feature_done_sha256}" \
  "${candidate_manifest}" \
  "${candidate_manifest_sha256}" \
  "${anchor_features}/done.json" \
  "${anchor_feature_done_sha256}" \
  "${anchor_manifest}" \
  "${anchor_manifest_sha256}" <<'PY'
import hashlib
import pathlib
import sys

arguments = sys.argv[1:]
for offset in range(0, len(arguments), 2):
    path = pathlib.Path(arguments[offset]).resolve(strict=True)
    expected = arguments[offset + 1]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise SystemExit(
            f"[r7-graph-input] input SHA differs: {path}; "
            f"expected={expected} actual={actual}"
        )
print("[r7-graph-input] four external SHA anchors verified", flush=True)
PY

echo "[r7-graph-input] host=$(hostname) job=${SLURM_JOB_ID}"
echo "[r7-graph-input] output=${output_dir}"

"${python_bin}" -m motive.r7_visual_graph_input \
  --candidate-features-dir "${candidate_features}" \
  --candidate-manifest "${candidate_manifest}" \
  --anchor-features-dir "${anchor_features}" \
  --anchor-input-manifest "${anchor_manifest}" \
  --output-dir "${output_dir}"

# Strict resume is validation-only and freshly rederives every source binding.
"${python_bin}" -m motive.r7_visual_graph_input \
  --candidate-features-dir "${candidate_features}" \
  --candidate-manifest "${candidate_manifest}" \
  --anchor-features-dir "${anchor_features}" \
  --anchor-input-manifest "${anchor_manifest}" \
  --output-dir "${output_dir}" \
  --resume

"${python_bin}" - \
  "${output_dir}/done.json" \
  "${expected_assets}" \
  "${expected_iids}" <<'PY'
import json
import pathlib
import sys

done = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_assets = int(sys.argv[2])
expected_iids = int(sys.argv[3])
if (
    done.get("status") != "complete"
    or done.get("assets") != expected_assets
    or done.get("iids") != expected_iids
    or done.get("split_assigned") is not False
    or done.get("human_labels_asserted") is not False
    or done.get("training_authorized") is not False
):
    raise SystemExit("[r7-graph-input] final fail-closed contract differs")
print(
    "[r7-graph-input] complete "
    f"assets={expected_assets} iids={expected_iids} "
    f"artifact_digest={done.get('artifact_digest')}",
    flush=True,
)
PY

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

sha256sum \
  "${output_dir}/manifest.jsonl" \
  "${output_dir}/features.npz" \
  "${output_dir}/summary.json" \
  "${output_dir}/done.json"

echo "[r7-graph-input] completed"
