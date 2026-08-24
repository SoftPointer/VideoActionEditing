#!/usr/bin/env bash
#SBATCH --job-name=motive-action-validate
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# AUH faculty QOS enforces MinTRES=gres/gpu:1 for validation jobs.
# The validator runs Python/OpenCV and salvages already-generated outputs.
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

for arm in \
  e1_plain_lora \
  e2_fixed_random \
  e2_random_router \
  e3_motive_frozen; do
  "${python_bin}" -m motive.action_inference validate-arm \
    --output-root "${inference_root}" \
    --arm "${arm}"
done
"${python_bin}" -m motive.action_inference validate-all \
  --output-root "${inference_root}"
echo "[motive-validate] all four generated arms are complete"
