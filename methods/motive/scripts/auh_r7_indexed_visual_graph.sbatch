#!/usr/bin/env bash
#SBATCH --job-name=m7-indexgraph
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00

set -Eeuo pipefail

source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?set MOTIVE_SOURCE_SNAPSHOT}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?set MOTIVE_SOURCE_TREE_SHA256}"
graph_input="${MOTIVE_R7_GRAPH_INPUT:?set MOTIVE_R7_GRAPH_INPUT}"
graph_input_digest="${MOTIVE_R7_GRAPH_INPUT_DIGEST:?set MOTIVE_R7_GRAPH_INPUT_DIGEST}"
dino_edges="${MOTIVE_R7_DINO_EDGE_FINAL:?set MOTIVE_R7_DINO_EDGE_FINAL}"
dino_edge_done_sha256="${MOTIVE_R7_DINO_EDGE_DONE_SHA256:?set MOTIVE_R7_DINO_EDGE_DONE_SHA256}"
output_dir="${MOTIVE_R7_INDEXED_GRAPH_OUTPUT:?set MOTIVE_R7_INDEXED_GRAPH_OUTPUT}"
expected_iids="${MOTIVE_R7_INDEXED_GRAPH_EXPECTED_IIDS:?set MOTIVE_R7_INDEXED_GRAPH_EXPECTED_IIDS}"
python_bin="${PYTHON_BIN:?set PYTHON_BIN}"

for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${source_snapshot}/methods/motive/motive/r7_indexed_visual_graph.py" \
  "${source_snapshot}/methods/motive/motive/r7_indexed_visual_graph_io.py" \
  "${source_snapshot}/methods/motive/motive/r7_expansion_dino_edges.py" \
  "${source_snapshot}/methods/motive/motive/r7_visual_graph_input.py" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  "${graph_input}/done.json" \
  "${dino_edges}/hard_edges.jsonl" \
  "${dino_edges}/audit_edges.jsonl" \
  "${dino_edges}/summary.json" \
  "${dino_edges}/done.json" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[r7-indexed-graph] missing required path: ${required}" >&2
    exit 2
  fi
done

if [[ -e "${output_dir}" || -L "${output_dir}" ]]; then
  echo "[r7-indexed-graph] refusing existing output: ${output_dir}" >&2
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
  "${graph_input}/done.json" \
  "${graph_input_digest}" \
  "${dino_edges}/done.json" \
  "${dino_edge_done_sha256}" <<'PY'
import hashlib
import json
import pathlib
import sys

graph_done = pathlib.Path(sys.argv[1]).resolve(strict=True)
expected_graph_digest = sys.argv[2]
payload = json.loads(graph_done.read_text(encoding="utf-8"))
if payload.get("artifact_digest") != expected_graph_digest:
    raise SystemExit("[r7-indexed-graph] graph input digest differs")

dino_done = pathlib.Path(sys.argv[3]).resolve(strict=True)
expected_dino_done_sha = sys.argv[4]
digest = hashlib.sha256()
with dino_done.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != expected_dino_done_sha:
    raise SystemExit("[r7-indexed-graph] DINO final done SHA differs")
print("[r7-indexed-graph] external input anchors verified", flush=True)
PY

echo "[r7-indexed-graph] host=$(hostname) job=${SLURM_JOB_ID}"
echo "[r7-indexed-graph] output=${output_dir}"

"${python_bin}" -m motive.r7_indexed_visual_graph_io \
  --graph-input-dir "${graph_input}" \
  --dino-edges-dir "${dino_edges}" \
  --output-dir "${output_dir}"

# Strict resume is validation-only and fully recomputes the deterministic graph.
"${python_bin}" -m motive.r7_indexed_visual_graph_io \
  --graph-input-dir "${graph_input}" \
  --dino-edges-dir "${dino_edges}" \
  --output-dir "${output_dir}" \
  --resume

"${python_bin}" - \
  "${output_dir}/done.json" \
  "${expected_iids}" <<'PY'
import json
import pathlib
import sys

done = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_iids = int(sys.argv[2])
if (
    done.get("status") != "complete"
    or done.get("iids") != expected_iids
    or done.get("thresholds_human_calibrated") is not False
    or done.get("formal_split") is not False
    or done.get("training_authorized") is not False
):
    raise SystemExit("[r7-indexed-graph] fail-closed decision fields differ")
print(
    "[r7-indexed-graph] complete "
    f"iids={expected_iids} components={done.get('components')} "
    f"giant_warning={done.get('giant_component_warning')} "
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
  "${output_dir}/assignments.jsonl" \
  "${output_dir}/components.jsonl" \
  "${output_dir}/spanning_edges.jsonl" \
  "${output_dir}/summary.json" \
  "${output_dir}/done.json"

echo "[r7-indexed-graph] completed"
