#!/usr/bin/env bash
# Exact per-rank interpreter boundary for the disposable 0817 PRE_D0 smoke.

set -Eeuo pipefail

readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

fail() {
  printf 'PRE_D0 rank wrapper refused: %s\n' "$*" >&2
  exit 97
}

[[ $# -ge 1 ]] || fail "missing absolute runner argv"
[[ "$1" == /* ]] || fail "runner path is not absolute"
[[ "${WORLD_SIZE:-}" == 8 ]] || fail "WORLD_SIZE is not 8"
[[ "${LOCAL_WORLD_SIZE:-}" == 8 ]] || fail "LOCAL_WORLD_SIZE is not 8"
[[ "${LOCAL_RANK:-}" =~ ^[0-7]$ ]] || fail "LOCAL_RANK is outside 0..7"
[[ "${RANK:-}" =~ ^[0-7]$ ]] || fail "RANK is outside 0..7"
[[ "${TORCHELASTIC_MAX_RESTARTS:-}" == 0 ]] || fail "elastic max restarts is not zero"
[[ "${TORCHELASTIC_RESTART_COUNT:-}" == 0 ]] || fail "elastic restart count is not zero"
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "pinned Python differs"

exec "${python_bin}" -I -B "$@"
