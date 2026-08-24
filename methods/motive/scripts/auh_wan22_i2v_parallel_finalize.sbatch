#!/usr/bin/env bash
#SBATCH --job-name=wan22-i2v-merge
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=02:00:00

# This validation job runs only after every independent eight-GPU shard
# succeeds.  Its code is CPU-only, but bgqos enforces MinTRES=gres/gpu=1.

set -Eeuo pipefail
umask 077

source_snapshot="${MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT:?set MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT}"
python_bin="${MOTIVE_WAN22_PYTHON_BIN:?set MOTIVE_WAN22_PYTHON_BIN}"
parallel_root="${MOTIVE_WAN22_PARALLEL_ROOT:?set MOTIVE_WAN22_PARALLEL_ROOT}"
code_root="${source_snapshot}/methods/motive"
module="${code_root}/motive/wan22_parallel_shards.py"
plan="${parallel_root}/parallel_plan.json"

for required in "${module}" "${plan}"; do
  if [[ -L "${required}" || ! -f "${required}" ]]; then
    echo "[wan22-parallel-finalize] missing regular file: ${required}" >&2
    exit 2
  fi
done
if [[ ! -e "${python_bin}" || ! -x "${python_bin}" ]]; then
  echo "[wan22-parallel-finalize] Python must resolve to an executable file: ${python_bin}" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${code_root}"
export PYTHONUNBUFFERED=1

"${python_bin}" "${module}" finalize \
  --parallel-root "${parallel_root}"

test -s "${parallel_root}/final/generated_manifest.jsonl"
test -s "${parallel_root}/final/aggregate_complete.json"
sha256sum \
  "${parallel_root}/final/generated_manifest.jsonl" \
  "${parallel_root}/final/aggregate_complete.json"
echo "[wan22-parallel-finalize] complete"
