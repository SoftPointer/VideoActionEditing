#!/usr/bin/env bash
#SBATCH --job-name=motive-r5-lite
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=12:00:00

set -Eeuo pipefail

# This job runs one real GPU trainer sequentially over the five registered model
# seeds.  It is intentionally not an eight-task placeholder allocation.
feature_root="${MOTIVE_R5_FEATURE_ROOT:?MOTIVE_R5_FEATURE_ROOT is required}"
output_root="${MOTIVE_R5_OUTPUT_ROOT:?MOTIVE_R5_OUTPUT_ROOT is required}"
source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?MOTIVE_SOURCE_SNAPSHOT is required}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?MOTIVE_SOURCE_TREE_SHA256 is required}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
code_root="${source_snapshot}/methods/motive"
label_mode="${MOTIVE_R5_LABEL_MODE:-strict_legacy_qwen}"

for required in \
  "${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  "${code_root}/motive/train_source_aware_repr.py" \
  "${code_root}/motive/r5_gate.py" \
  "${feature_root}/final.npz" \
  "${feature_root}/manifest.jsonl" \
  "${feature_root}/summary.json" \
  "${feature_root}/done.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "[motive-r5] missing required path: ${required}" >&2
    exit 2
  fi
done

if [[ -e "${output_root}" ]]; then
  echo "[motive-r5] refusing to reuse output root: ${output_root}" >&2
  exit 2
fi
if [[ "${label_mode}" != "human" && "${label_mode}" != "strict_legacy_qwen" ]]; then
  echo "[motive-r5] invalid MOTIVE_R5_LABEL_MODE=${label_mode}" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${code_root}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

human_label_args=()
if [[ "${label_mode}" == "human" && -n "${MOTIVE_R5_HUMAN_LABEL_SOURCE:-}" ]]; then
  if [[ ! -f "${MOTIVE_R5_HUMAN_LABEL_SOURCE}" ]]; then
    echo "[motive-r5] missing human label source: ${MOTIVE_R5_HUMAN_LABEL_SOURCE}" >&2
    exit 2
  fi
  human_label_args+=(--human-label-source "${MOTIVE_R5_HUMAN_LABEL_SOURCE}")
fi

"${python_bin}" - <<'PY'
import json
import os

import torch

result = {
    "torch": torch.__version__,
    "torch_hip": torch.version.hip,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
    "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
}
if not result["cuda_available"] or result["device_count"] != 1:
    print(json.dumps(result, sort_keys=True), flush=True)
    raise SystemExit(
        "[motive-r5] expected exactly one usable Slurm-visible GPU"
    )
result["device_name"] = torch.cuda.get_device_name(0)
result["probe_sum"] = float(torch.ones(1024, device="cuda").sum().item())
print("[motive-r5-gpu] " + json.dumps(result, sort_keys=True), flush=True)
PY

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

"${python_bin}" -m motive.train_source_aware_repr \
  --features "${feature_root}/final.npz" \
  --manifest "${feature_root}/manifest.jsonl" \
  --output-dir "${output_root}" \
  --label-mode "${label_mode}" \
  "${human_label_args[@]}" \
  --data-seed "${MOTIVE_R5_DATA_SEED:-260108828}" \
  --model-seeds "${MOTIVE_R5_MODEL_SEEDS:-2026,2027,2028,2029,2030}" \
  --steps "${MOTIVE_R5_STEPS:-3000}" \
  --batch-size "${MOTIVE_R5_BATCH_SIZE:-128}" \
  --eval-batch-size "${MOTIVE_R5_EVAL_BATCH_SIZE:-512}" \
  --hidden-dim "${MOTIVE_R5_HIDDEN_DIM:-128}" \
  --condition-dim "${MOTIVE_R5_CONDITION_DIM:-128}" \
  --learning-rate "${MOTIVE_R5_LEARNING_RATE:-0.0003}" \
  --weight-decay "${MOTIVE_R5_WEIGHT_DECAY:-0.0001}" \
  --reference-dropout "${MOTIVE_R5_REFERENCE_DROPOUT:-0.5}" \
  --magnitude-weight "${MOTIVE_R5_MAGNITUDE_WEIGHT:-0.25}" \
  --bootstrap-samples "${MOTIVE_R5_BOOTSTRAP_SAMPLES:-20000}" \
  --signflip-samples "${MOTIVE_R5_SIGNFLIP_SAMPLES:-50000}" \
  --device cuda

for artifact in \
  "${output_root}/contract.json" \
  "${output_root}/transform.json" \
  "${output_root}/per_query.jsonl" \
  "${output_root}/gate_summary.json" \
  "${output_root}/summary.json" \
  "${output_root}/done.json"; do
  if [[ ! -s "${artifact}" ]]; then
    echo "[motive-r5] missing final artifact: ${artifact}" >&2
    exit 2
  fi
done

"${python_bin}" - "${output_root}/gate_summary.json" "${label_mode}" <<'PY'
import json
import sys

path, label_mode = sys.argv[1:]
with open(path, "r", encoding="utf-8") as handle:
    gate = json.load(handle)
status = gate.get("status")
if status not in {"PASS", "NO_GO", "INSUFFICIENT"}:
    raise SystemExit(f"invalid/incomplete R5 gate status: {status!r}")
if label_mode != "human" and status == "PASS":
    raise SystemExit("pseudo-label R5 run must never return PASS")
print(f"[motive-r5] validated gate status={status}")
PY

echo "[motive-r5] completed output=${output_root}"
