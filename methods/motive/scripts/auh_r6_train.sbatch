#!/usr/bin/env bash
#SBATCH --job-name=motive-r6
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2

set -Eeuo pipefail

# Exactly two real tasks are pre-registered: CLIP and UMT5.  Each task runs a
# GPU probe followed by five-seed PyTorch training; neither task is a wrapper
# placeholder and test metrics are never used to pick an encoder winner.
task_id="${SLURM_ARRAY_TASK_ID:?submit as --array=0-1%2}"
if [[ "${task_id}" == "0" ]]; then
  encoder="clip"
elif [[ "${task_id}" == "1" ]]; then
  encoder="umt5"
else
  echo "[motive-r6] invalid array task ${task_id}; expected 0 or 1" >&2
  exit 2
fi

feature_root="${MOTIVE_R6_FEATURE_ROOT:?MOTIVE_R6_FEATURE_ROOT is required}"
semantic_root="${MOTIVE_R6_SEMANTIC_ROOT:?MOTIVE_R6_SEMANTIC_ROOT is required}"
output_base="${MOTIVE_R6_OUTPUT_ROOT:?MOTIVE_R6_OUTPUT_ROOT is required}"
source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?MOTIVE_SOURCE_SNAPSHOT is required}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?MOTIVE_SOURCE_TREE_SHA256 is required}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
code_root="${source_snapshot}/methods/motive"
label_mode="${MOTIVE_R6_LABEL_MODE:-strict_legacy_qwen}"
output_root="${output_base}/${encoder}"

for required in \
  "${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${code_root}/motive/train_source_aware_r6.py" \
  "${code_root}/motive/source_aware_repr_r6.py" \
  "${code_root}/motive/r6_gate.py" \
  "${code_root}/motive/r6_semantic_features.py" \
  "${feature_root}/final.npz" \
  "${feature_root}/manifest.jsonl" \
  "${feature_root}/summary.json" \
  "${feature_root}/done.json" \
  "${semantic_root}/semantic_features.npz" \
  "${semantic_root}/manifest.jsonl" \
  "${semantic_root}/metadata.json" \
  "${semantic_root}/summary.json" \
  "${semantic_root}/done.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "[motive-r6] missing required path: ${required}" >&2
    exit 2
  fi
done

if [[ -e "${output_root}" ]]; then
  echo "[motive-r6] refusing to reuse output root: ${output_root}" >&2
  exit 2
fi
if [[ "${label_mode}" != "human" && "${label_mode}" != "strict_legacy_qwen" ]]; then
  echo "[motive-r6] invalid MOTIVE_R6_LABEL_MODE=${label_mode}" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${code_root}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export MOTIVE_SOURCE_SNAPSHOT="${source_snapshot}"

# Prevent Slurm's inherited submit directory from shadowing the immutable
# snapshot with a mutable top-level motive package.
cd "${source_snapshot}/methods/motive"

source_files_sha256="$("${python_bin}" - "${source_snapshot}/SOURCE_FILES.jsonl" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
)"

"${python_bin}" - <<'PY'
import json
import os
from pathlib import Path

import motive
import torch

result = {
    "torch": torch.__version__,
    "torch_hip": torch.version.hip,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
    "motive_module": str(Path(motive.__file__).resolve()),
}
expected = (
    Path(os.environ["MOTIVE_SOURCE_SNAPSHOT"]).resolve()
    / "methods"
    / "motive"
)
if expected not in Path(motive.__file__).resolve().parents:
    raise SystemExit(
        f"[motive-r6] imported motive outside snapshot: {motive.__file__}"
    )
if not result["cuda_available"] or result["device_count"] != 1:
    print(json.dumps(result, sort_keys=True), flush=True)
    raise SystemExit("[motive-r6] expected exactly one Slurm-visible GPU")
result["device_name"] = torch.cuda.get_device_name(0)
result["probe_sum"] = float(torch.ones(1024, device="cuda").sum().item())
print("[motive-r6-gpu] " + json.dumps(result, sort_keys=True), flush=True)
PY

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

# Full semantic directory validation checks done.json, every sidecar digest,
# original input-manifest binding, exact IID/text order, and rejects synthetic
# artifacts before the trainer allocates its output directory.
"${python_bin}" - "${semantic_root}" <<'PY'
import sys
from pathlib import Path

from motive.r6_semantic_features import validate_artifact

result = validate_artifact(Path(sys.argv[1]))
if result["synthetic_test_artifact"]:
    raise SystemExit("synthetic semantic artifact cannot enter R6 training")
print(
    f"[motive-r6-semantic] validated rows={result['rows']} "
    f"synthetic={result['synthetic_test_artifact']}",
    flush=True,
)
PY

"${python_bin}" -m motive.train_source_aware_r6 \
  --features "${feature_root}/final.npz" \
  --manifest "${feature_root}/manifest.jsonl" \
  --semantic-features "${semantic_root}/semantic_features.npz" \
  --source-snapshot "${source_snapshot}" \
  --source-tree-sha256 "${source_tree_sha256}" \
  --source-files-sha256 "${source_files_sha256}" \
  --encoder "${encoder}" \
  --output-dir "${output_root}" \
  --label-mode "${label_mode}" \
  --data-seed "${MOTIVE_R6_DATA_SEED:-260108828}" \
  --model-seeds "${MOTIVE_R6_MODEL_SEEDS:-2026,2027,2028,2029,2030}" \
  --steps "${MOTIVE_R6_STEPS:-3000}" \
  --batch-size "${MOTIVE_R6_BATCH_SIZE:-128}" \
  --eval-batch-size "${MOTIVE_R6_EVAL_BATCH_SIZE:-512}" \
  --hidden-dim "${MOTIVE_R6_HIDDEN_DIM:-128}" \
  --semantic-condition-dim "${MOTIVE_R6_SEMANTIC_CONDITION_DIM:-64}" \
  --condition-dim "${MOTIVE_R6_MOTION_CONDITION_DIM:-16}" \
  --references-per-query "${MOTIVE_R6_REFERENCES_PER_QUERY:-3}" \
  --learning-rate "${MOTIVE_R6_LEARNING_RATE:-0.0003}" \
  --weight-decay "${MOTIVE_R6_WEIGHT_DECAY:-0.0001}" \
  --reference-dropout "${MOTIVE_R6_REFERENCE_DROPOUT:-0.5}" \
  --magnitude-weight "${MOTIVE_R6_MAGNITUDE_WEIGHT:-0.25}" \
  --compatibility-weight "${MOTIVE_R6_COMPATIBILITY_WEIGHT:-1.0}" \
  --active-threshold 0.0001 \
  --bootstrap-samples "${MOTIVE_R6_BOOTSTRAP_SAMPLES:-5000}" \
  --device cuda

for artifact in \
  "${output_root}/reference_pairs.jsonl" \
  "${output_root}/reference_audit.json" \
  "${output_root}/contract.json" \
  "${output_root}/transform.json" \
  "${output_root}/per_query.jsonl" \
  "${output_root}/gate_summary.json" \
  "${output_root}/summary.json" \
  "${output_root}/done.json"; do
  if [[ ! -s "${artifact}" ]]; then
    echo "[motive-r6] missing final artifact: ${artifact}" >&2
    exit 2
  fi
done

"${python_bin}" - "${output_root}/gate_summary.json" "${encoder}" <<'PY'
import json
import sys

path, encoder = sys.argv[1:]
with open(path, "r", encoding="utf-8") as handle:
    gate = json.load(handle)
if gate.get("status") != "INSUFFICIENT":
    raise SystemExit(
        f"R6 formal gate must remain INSUFFICIENT, got {gate.get('status')!r}"
    )
if gate.get("production_decision") is not False:
    raise SystemExit("R6 cannot make a production decision")
if gate.get("generation_authorized") is not False:
    raise SystemExit("R6 cannot authorize generation")
print(
    f"[motive-r6] encoder={encoder} formal={gate['status']} "
    f"pilot={gate['pilot_diagnostic']['status']}",
    flush=True,
)
PY

echo "[motive-r6] completed encoder=${encoder} output=${output_root}"
