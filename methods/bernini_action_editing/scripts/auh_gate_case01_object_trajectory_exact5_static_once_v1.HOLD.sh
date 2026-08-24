#!/bin/bash -p
set -Eeuo pipefail
umask 077
readonly CONTROLLER_STATE=HOLD_PENDING_PACKAGE_RECEIPT_PIN
readonly PACKAGE_RECEIPT_SHA256=__BLOCKED_PENDING_PACKAGE_RECEIPT_SHA256__
[[ "$CONTROLLER_STATE" == READY && "$PACKAGE_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  /usr/bin/printf '%s\n' 'HOLD: static admission awaits sealed package receipt' >&2
  exit 88
}
/usr/bin/printf '%s\n' 'HOLD: no admission command is enabled in this controller' >&2
exit 88
