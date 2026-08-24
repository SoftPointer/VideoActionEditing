#!/usr/bin/env bash
#SBATCH --job-name=motive-action-finalize
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# AUH faculty QOS enforces MinTRES=gres/gpu:1 even for finalization.
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=01:00:00

set -Eeuo pipefail

inference_root="${MOTIVE_INFERENCE_ROOT:?MOTIVE_INFERENCE_ROOT is required}"
code_root="${MOTIVE_CODE_ROOT:?MOTIVE_CODE_ROOT is required}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?MOTIVE_SOURCE_TREE_SHA256 is required}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${code_root}/methods/motive:${code_root}:${PYTHONPATH:-}"
"${python_bin}" \
  "${code_root}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${code_root}" \
  --expected-tree-sha256 "${source_tree_sha256}"
"${python_bin}" -m motive.action_inference finalize \
  --output-root "${inference_root}"
echo "[motive-finalize] complete root=${inference_root}"
