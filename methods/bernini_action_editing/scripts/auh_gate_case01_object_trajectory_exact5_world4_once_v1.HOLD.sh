#!/bin/bash -p
set -Eeuo pipefail
umask 077
readonly CONTROLLER_STATE=HOLD_PENDING_ROOT_FAKE_ADMISSION_PIN
readonly ROOT_FAKE_RECEIPT_SHA256=__BLOCKED_PENDING_ROOT_FAKE_RECEIPT_SHA256__
[[ "$CONTROLLER_STATE" == READY && "$ROOT_FAKE_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  /usr/bin/printf '%s\n' 'HOLD: world4 admission awaits root-fake receipt' >&2
  exit 88
}
/usr/bin/printf '%s\n' 'HOLD: torchrun is not enabled in this controller' >&2
exit 88
