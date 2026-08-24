#!/usr/bin/env bash
# Optional single-command chain: exact80 training, then fixed checkpoint decode.
# The existing training holder remains independently owned and unmodified.

set -Eeuo pipefail

fail() { echo "[csvc-train-then-review] ERROR: $*" >&2; exit 2; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
training="${script_dir}/auh_train_clean_source_visual_context_stage_b_holder_v1.sh"
review="${script_dir}/auh_decode_clean_source_visual_context_checkpoint_review_holder_v1.sh"
[[ -f "${training}" && ! -L "${training}" ]] || fail "training holder script differs"
[[ -f "${review}" && ! -L "${review}" ]] || fail "checkpoint review holder script differs"
[[ "${CSVC_EXECUTION_SCOPE:?bind execution scope}" == formal-exact80 ]] || fail "decoded review requires formal exact80"
bash "${training}"
[[ -f "${CSVC_RUN_ROOT:?}/controller.TRAINING_COMPLETE" ]] || fail "training did not publish exact80 handoff"
exec bash "${review}"
