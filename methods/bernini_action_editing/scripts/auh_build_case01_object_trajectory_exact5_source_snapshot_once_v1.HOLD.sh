#!/bin/bash -p
set -Eeuo pipefail
umask 077
readonly CONTROLLER_STATE=HOLD_FORMAL_V3_PINS_NO_REVIEWED_REMOTE_BODY
readonly WRAPPER_SHA256=20ee1447148cfc60c6cb745316ce972180070d50b6431a8f4d254ee5dfff7db9
readonly RUNNER_SHA256=e47b81643c1d17e5099a9b33f16ca75521001ad52d2df2305b46b7e8c4d5ac4c
readonly EVAL_SHA256=47cc871b82b8cf7762db9183997440eeabd287b1c702d9cd7421fd43e0a555e0
[[ "$CONTROLLER_STATE" == READY && "$RUNNER_SHA256" =~ ^[0-9a-f]{64}$ \
  && "$EVAL_SHA256" =~ ^[0-9a-f]{64}$ \
  && "$WRAPPER_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  /usr/bin/printf '%s\n' 'HOLD: formal v3 pins are present; reviewed remote body is absent' >&2
  exit 88
}
/usr/bin/printf '%s\n' 'HOLD: this reviewed controller has no remote execution body' >&2
exit 88
