#!/usr/bin/env bash
set -Eeuo pipefail
export CSVC_HOLDER_JOB=135981
export CSVC_HOLDER_NODE=auh7-1b-gpu-234
export CSVC_MEMORY_INPUT_KIND=same_noise_forward_noised_source
export CSVC_EXECUTION_SCOPE=structural-parity-preflight
unset CSVC_STAGE_A_ADMISSION CSVC_STAGE_A_ADMISSION_SHA256 CSVC_FORMAL_PAIR_ADMISSION CSVC_FORMAL_PAIR_ADMISSION_SHA256 CSVC_EXPECTED_INITIAL_PARAMETER_DIGEST
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "${script_dir}/auh_train_clean_source_visual_context_stage_b_holder_v1.sh" "$@"
