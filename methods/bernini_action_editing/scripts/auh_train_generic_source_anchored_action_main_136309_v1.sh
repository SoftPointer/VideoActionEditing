#!/usr/bin/env bash
# Bound manual fallback for the staged main continuation on holder 136309.
set -Eeuo pipefail
export GSA_ARM_ID=joint_resume_po40
export GSA_HOLDER_JOB=136309
export GSA_HOLDER_NODE=auh7-1b-gpu-280
export GSA_EXECUTION_PROFILE=resume-po40
export GSA_CARRIER_POLICY=resume_frozen_stage_r64
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "${script_dir}/auh_train_generic_source_anchored_action_world4_holder_v1.sh" "$@"

