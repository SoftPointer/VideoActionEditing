#!/usr/bin/env bash
# Run one recoverable eight-row Wan2.2 shard, then materialize its source video
# and edit instruction beside every generated preview.
#SBATCH --job-name=wan22-512-shard
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1024G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=12:00:00
#SBATCH --requeue

set -Eeuo pipefail
umask 077

source_snapshot="${MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT:?set MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT}"
manifest="${MOTIVE_GOKU_ACTION_GENERATION_MANIFEST:?set MOTIVE_GOKU_ACTION_GENERATION_MANIFEST}"
output_root="${MOTIVE_WAN22_OUTPUT_ROOT:?set MOTIVE_WAN22_OUTPUT_ROOT}"
expected_rows="${MOTIVE_WAN22_MAX_NEW_SAMPLES:?set MOTIVE_WAN22_MAX_NEW_SAMPLES}"
full_script="${source_snapshot}/methods/motive/scripts/auh_wan22_i2v_full.sbatch"
materializer="${source_snapshot}/methods/motive/scripts/materialize_wan22_sources.py"
materializer_python_requested="${MOTIVE_MATERIALIZE_PYTHON_BIN:-/usr/bin/python3}"

fail() {
  echo "[wan22-scale512-shard] $*" >&2
  exit 2
}

[[ "${expected_rows}" =~ ^[1-9][0-9]*$ ]] || fail "invalid expected row count"
for path in "${manifest}" "${full_script}" "${materializer}"; do
  [[ ! -L "${path}" && -f "${path}" ]] || fail "missing regular file: ${path}"
done
materializer_python="$(readlink -f -- "${materializer_python_requested}")"
[[ -n "${materializer_python}" && ! -L "${materializer_python}" && -f "${materializer_python}" ]] \
  || fail "could not resolve materializer Python: ${materializer_python_requested}"
[[ -x "${materializer_python}" ]] || fail "materializer Python is not executable"

export NCCL_IB_DISABLE=1
unset NCCL_IB_HCA NCCL_IB_GID_INDEX
export NCCL_SOCKET_IFNAME=bond0
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=bond0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,NET

echo "[wan22-scale512-shard] generation_start job=${SLURM_JOB_ID:-manual} host=$(hostname)"
bash "${full_script}"

manifest_sha256="$(sha256sum "${manifest}" | awk '{print $1}')"
"${materializer_python}" "${materializer}" \
  --manifest "${manifest}" \
  --output-root "${output_root}" \
  --expected-manifest-sha256 "${manifest_sha256}" \
  --require-all

source_count="$(find "${output_root}/samples" -mindepth 2 -maxdepth 2 -type f -name source.mp4 | wc -l)"
instruction_count="$(find "${output_root}/samples" -mindepth 2 -maxdepth 2 -type f -name edit_instruction.txt | wc -l)"
[[ "${source_count}" == "${expected_rows}" ]] \
  || fail "source copy count differs: ${source_count}/${expected_rows}"
[[ "${instruction_count}" == "${expected_rows}" ]] \
  || fail "instruction file count differs: ${instruction_count}/${expected_rows}"

sha256sum \
  "${output_root}/run_complete.json" \
  "${output_root}/generated_manifest.jsonl" \
  "${output_root}/source_target_pairs.jsonl" \
  "${output_root}/source_materialization_status.json"
echo "[wan22-scale512-shard] complete rows=${expected_rows}"
