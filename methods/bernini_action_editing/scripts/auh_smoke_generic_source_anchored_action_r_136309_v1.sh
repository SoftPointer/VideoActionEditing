#!/usr/bin/env bash
# Bound manual fallback for the disposable one-update R-path smoke.
set -Eeuo pipefail
export GSA_ARM_ID=smoke_r
export GSA_HOLDER_JOB=136309
export GSA_HOLDER_NODE=auh7-1b-gpu-280
export GSA_EXECUTION_PROFILE=smoke-r
export GSA_CARRIER_POLICY=installed_trainable_disposable
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "${script_dir}/auh_train_generic_source_anchored_action_world4_holder_v1.sh" "$@"

