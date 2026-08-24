#!/bin/bash -p
set -Eeuo pipefail
umask 077
readonly CONTROLLER_STATE=HOLD_PENDING_EXPLICIT_READY_OVERLAY
readonly STATIC_RECEIPT_SHA256=__BLOCKED_PENDING_STATIC_RECEIPT_SHA256__
readonly ROOT_FAKE_RECEIPT_SHA256=__BLOCKED_PENDING_ROOT_FAKE_RECEIPT_SHA256__
readonly WORLD4_RECEIPT_SHA256=__BLOCKED_PENDING_WORLD4_RECEIPT_SHA256__
readonly READY_PLAN_SHA256=__BLOCKED_PENDING_CREATE_ONLY_READY_PLAN_SHA256__
[[ "$CONTROLLER_STATE" == READY \
  && "$STATIC_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ \
  && "$ROOT_FAKE_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ \
  && "$WORLD4_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ \
  && "$READY_PLAN_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  /usr/bin/printf '%s\n' 'HOLD: no launch without three admissions and a fresh READY overlay' >&2
  exit 88
}
/usr/bin/printf '%s\n' 'HOLD: this artifact intentionally contains no srun/sbatch/torchrun command' >&2
exit 88
