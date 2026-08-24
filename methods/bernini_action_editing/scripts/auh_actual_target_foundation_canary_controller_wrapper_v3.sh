#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 IMMUTABLE_SNAPSHOT_ROOT CONTROLLER_ARGUMENT..." >&2
  exit 64
fi

snapshot_root=$1
shift
expected_wrapper="${snapshot_root}/scripts/auh_actual_target_foundation_canary_controller_wrapper_v3.sh"
controller="${snapshot_root}/actual_target_foundation_controller_v3.py"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

if [[ "$0" != "$expected_wrapper" || ! -f "$controller" || -L "$controller" ]]; then
  echo "controller wrapper/source snapshot binding differs" >&2
  exit 65
fi
if [[ -z ${SLURM_JOB_ID:-} || -z ${SLURM_STEP_ID:-} ]]; then
  echo "controller wrapper requires one real Slurm task" >&2
  exit 66
fi

# Slurm's NoDevFiles plugin may remove ROCR_VISIBLE_DEVICES entirely even when
# --export requested an empty value.  Reset all three masks inside the task,
# after Slurm has finished mutating the environment and before Python starts.
export CUDA_VISIBLE_DEVICES=''
export ROCR_VISIBLE_DEVICES=''
export HIP_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1

exec "$python_bin" -B "$controller" "$@"
