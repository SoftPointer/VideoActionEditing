#!/usr/bin/env bash
# Bound manual fallback for the no-carrier action-only control on holder 136141.
set -Eeuo pipefail
export GSA_ARM_ID=action_only_no_carrier
export GSA_HOLDER_JOB=136141
export GSA_HOLDER_NODE=auh7-1b-gpu-299
export GSA_EXECUTION_PROFILE=action-only40
export GSA_CARRIER_POLICY=not_installed_or_exact_zero_frozen
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "${script_dir}/auh_train_generic_source_anchored_action_world4_holder_v1.sh" "$@"

