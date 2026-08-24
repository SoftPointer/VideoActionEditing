#!/usr/bin/env bash
# Reserve a correctly sized Wan2.2 node, wait for an independently verified
# exact-eight release marker, then enter the ordinary fail-closed full runner.
#SBATCH --job-name=wan22-exact8-wait
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
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-192,auh7-1b-gpu-195,auh7-1b-gpu-201,auh7-1b-gpu-233,auh7-1b-gpu-258,auh7-1b-gpu-305,auh7-1b-gpu-318

set -Eeuo pipefail
umask 077

ready_marker="${MOTIVE_WAN22_READY_MARKER:?set MOTIVE_WAN22_READY_MARKER}"
source_snapshot="${MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT:?set MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT}"
ffprobe_bin="${MOTIVE_WAN22_FFPROBE_BIN:?set MOTIVE_WAN22_FFPROBE_BIN}"
ffprobe_sha256="${MOTIVE_WAN22_FFPROBE_SHA256:?set MOTIVE_WAN22_FFPROBE_SHA256}"
timeout_seconds="${MOTIVE_WAN22_READY_TIMEOUT_SECONDS:-21600}"
poll_seconds="${MOTIVE_WAN22_READY_POLL_SECONDS:-20}"

for value in "${timeout_seconds}" "${poll_seconds}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "wait timeout and poll interval must be positive integers" >&2
    exit 2
  fi
done
for path in "${ready_marker}" "${source_snapshot}" "${ffprobe_bin}"; do
  if [[ "${path}" != /* || "${path}" == "/" ]]; then
    echo "ready marker, source snapshot, and ffprobe must be non-root absolute paths" >&2
    exit 2
  fi
done
if [[ ! "${ffprobe_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "ffprobe SHA-256 must be lowercase hexadecimal" >&2
  exit 2
fi
if [[ -L "${ffprobe_bin}" || ! -f "${ffprobe_bin}" || ! -x "${ffprobe_bin}" ]]; then
  echo "ffprobe must be an executable regular non-symlink file: ${ffprobe_bin}" >&2
  exit 2
fi
actual_ffprobe_sha256="$(sha256sum "${ffprobe_bin}")"
actual_ffprobe_sha256="${actual_ffprobe_sha256%% *}"
if [[ "${actual_ffprobe_sha256}" != "${ffprobe_sha256}" ]]; then
  echo "ffprobe SHA-256 mismatch: ${actual_ffprobe_sha256}" >&2
  exit 2
fi

deadline=$(( $(date +%s) + timeout_seconds ))
echo "[wan22-wait] job=${SLURM_JOB_ID} host=$(hostname) marker=${ready_marker}"
echo "[wan22-wait] ffprobe=${ffprobe_bin} sha256=${ffprobe_sha256}"
while [[ ! -f "${ready_marker}" || -L "${ready_marker}" ]]; do
  if (( $(date +%s) >= deadline )); then
    echo "[wan22-wait] timed out before exact-eight release became ready" >&2
    exit 3
  fi
  sleep "${poll_seconds}"
done

if [[ ! -s "${ready_marker}" ]]; then
  echo "[wan22-wait] ready marker is empty" >&2
  exit 2
fi
full_wrapper="${source_snapshot}/methods/motive/scripts/auh_wan22_i2v_parallel_full.sbatch"
for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${full_wrapper}" \
  "${MOTIVE_GOKU_ACTION_GENERATION_MANIFEST:?set MOTIVE_GOKU_ACTION_GENERATION_MANIFEST}" \
  "${MOTIVE_WAN22_SIGNED_RELEASE:?set MOTIVE_WAN22_SIGNED_RELEASE}"; do
  if [[ -L "${required}" || ! -f "${required}" ]]; then
    echo "[wan22-wait] required release artifact is unavailable: ${required}" >&2
    exit 2
  fi
done

echo "[wan22-wait] release marker observed; entering fail-closed full runner"
exec bash "${full_wrapper}"
