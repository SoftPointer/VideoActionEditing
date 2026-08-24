#!/usr/bin/env bash
#SBATCH --job-name=motive-action-metrics
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
# AUH faculty QOS enforces MinTRES=gres/gpu:1 even for CPU analysis.
# The node runs the motion-evaluation Python process; this is not a placeholder.
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=03:00:00

set -Eeuo pipefail

inference_root="${MOTIVE_INFERENCE_ROOT:?MOTIVE_INFERENCE_ROOT is required}"
code_root="${MOTIVE_CODE_ROOT:?MOTIVE_CODE_ROOT is required}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?MOTIVE_SOURCE_TREE_SHA256 is required}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${code_root}/methods/motive:${code_root}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
"${python_bin}" \
  "${code_root}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${code_root}" \
  --expected-tree-sha256 "${source_tree_sha256}"
"${python_bin}" -m motive.action_inference validate-all \
  --output-root "${inference_root}"

arms_root="${inference_root}/arms"
"${python_bin}" -m motive.eval_action_inference \
  --samples-json "${arms_root}/e1_plain_lora/samples.json" \
  --arm "e1_plain_lora=${arms_root}/e1_plain_lora" \
  --arm "e2_fixed_random=${arms_root}/e2_fixed_random" \
  --arm "e2_random_router=${arms_root}/e2_random_router" \
  --arm "e3_motive_frozen=${arms_root}/e3_motive_frozen" \
  --output-dir "${inference_root}/analysis/motion" \
  --prediction-name step_000100.mp4 \
  --split-key split \
  --analysis-frames 32 \
  --resize-width 256 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 2026 \
  --reference-arm e1_plain_lora

"${python_bin}" -m motive.action_inference build-qwen-manifest \
  --output-root "${inference_root}"
echo "[motive-metrics] complete root=${inference_root}"
