#!/usr/bin/env bash
#SBATCH --job-name=braid-ref4fa-all8
#SBATCH --partition=faculty
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=04:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# Explicitly partial Stage-0 audit.  This wrapper authenticates itself through
# the shared launcher, then runs exactly two concurrent fresh WORLD4 groups:
# dog on physical GPUs 0..3 and human on 4..7.  It publishes only
# reference-4f-a-only-all8.receipt.json with partial_stage0=true and no Stage-A
# authority.  It never falls through to the unimplemented five arms.

wrapper_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
launcher_path="$(realpath -e -- "$(dirname -- "${wrapper_path}")/auh_run_braid_stage0_canaries_all8_v1.sbatch")"
[[ -f "${launcher_path}" && ! -L "${launcher_path}" ]] || {
  echo "[braid-ref4fa-all8] ERROR: shared authenticated launcher differs" >&2
  exit 2
}

export BRAID_STAGE0_SCOPE=reference-4f-a-only
export BRAID_STAGE0_SCOPE_LAUNCHER="${wrapper_path}"
exec /bin/bash "${launcher_path}"
