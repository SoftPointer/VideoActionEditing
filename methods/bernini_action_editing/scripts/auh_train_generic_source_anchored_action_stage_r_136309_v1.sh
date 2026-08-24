#!/usr/bin/env bash
# Bound manual fallback for formal R64 on the currently idle main holder.
set -Eeuo pipefail
export GSA_ARM_ID=joint_stage_r64
export GSA_HOLDER_JOB=136309
export GSA_HOLDER_NODE=auh7-1b-gpu-280
export GSA_EXECUTION_PROFILE=stage-r64
export GSA_CARRIER_POLICY=installed_trainable
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "${script_dir}/auh_train_generic_source_anchored_action_world4_holder_v1.sh" "$@"

