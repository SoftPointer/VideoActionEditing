#!/bin/bash -p
set -Eeuo pipefail
umask 077
readonly CONTROLLER_STATE=HOLD_PENDING_RECEIPT_GATED_EXACT35_SNAPSHOT_PIN
readonly SNAPSHOT_MANIFEST_SHA256=__BLOCKED_PENDING_EXACT35_SNAPSHOT_MANIFEST_SHA256__
[[ "$CONTROLLER_STATE" == READY && "$SNAPSHOT_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  /usr/bin/printf '%s\n' 'HOLD: package materialization awaits receipt-gated exact35 snapshot' >&2
  exit 88
}
/usr/bin/printf '%s\n' 'HOLD: this reviewed controller has no remote execution body' >&2
exit 88
