#!/bin/bash -p
set -Eeuo pipefail
umask 077
readonly CONTROLLER_STATE=HOLD_PENDING_STATIC_ADMISSION_PIN
readonly STATIC_RECEIPT_SHA256=__BLOCKED_PENDING_STATIC_RECEIPT_SHA256__
[[ "$CONTROLLER_STATE" == READY && "$STATIC_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  /usr/bin/printf '%s\n' 'HOLD: root-fake admission awaits static receipt' >&2
  exit 88
}
/usr/bin/printf '%s\n' 'HOLD: no captured-root command is enabled in this controller' >&2
exit 88
