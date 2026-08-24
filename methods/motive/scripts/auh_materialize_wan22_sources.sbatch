#!/usr/bin/env bash
# Copy each verified source video beside its committed Wan2.2 result.
#SBATCH --job-name=wan22-src-copy
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=00:15:00

set -Eeuo pipefail
umask 022

python_bin="${MOTIVE_MATERIALIZE_PYTHON_BIN:-/usr/bin/python3}"
materializer="${MOTIVE_SOURCE_MATERIALIZER:?set MOTIVE_SOURCE_MATERIALIZER}"
manifest="${MOTIVE_GOKU_ACTION_GENERATION_MANIFEST:?set MOTIVE_GOKU_ACTION_GENERATION_MANIFEST}"
manifest_sha256="${MOTIVE_GOKU_ACTION_GENERATION_MANIFEST_SHA256:?set MOTIVE_GOKU_ACTION_GENERATION_MANIFEST_SHA256}"
output_root="${MOTIVE_WAN22_OUTPUT_ROOT:?set MOTIVE_WAN22_OUTPUT_ROOT}"

for path in "${python_bin}" "${materializer}" "${manifest}"; do
  if [[ -L "${path}" || ! -f "${path}" ]]; then
    echo "required input must be a regular non-symlink file: ${path}" >&2
    exit 2
  fi
done
if [[ ! -x "${python_bin}" ]]; then
  echo "Python is not executable: ${python_bin}" >&2
  exit 2
fi
if [[ ! "${manifest_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "manifest SHA-256 must be 64 lowercase hexadecimal characters" >&2
  exit 2
fi

"${python_bin}" "${materializer}" \
  --manifest "${manifest}" \
  --output-root "${output_root}" \
  --expected-manifest-sha256 "${manifest_sha256}" \
  --require-all

sha256sum \
  "${output_root}/source_target_pairs.jsonl" \
  "${output_root}/source_materialization_status.json"
