#!/usr/bin/env bash
set -euo pipefail

# CPU/read-only preflight only.  There is intentionally no srun/torchrun path
# until a new reviewed code version adds a compiled-public-key signature
# verifier, independent trust anchors/execution receipts, a review policy, and
# a frozen ABI-attested runner. Exact authority-SHA pinning cannot unlock this
# component-pinned graph.
if [ "$#" -ne 4 ]; then
  echo "usage: $0 e02|e03 native_local_r2v4|flowedit_step0_noise|connected /absolute/release-authority.json expected-authority-sha256" >&2
  exit 2
fi

case_id="$1"
execution="$2"
release_authority="$3"
release_authority_sha256="$4"
case "$case_id" in e02|e03) ;; *) exit 2 ;; esac
case "$execution" in native_local_r2v4|flowedit_step0_noise|connected) ;; *) exit 2 ;; esac
case "$release_authority" in /*) ;; *) exit 2 ;; esac
if [ "${#release_authority_sha256}" -ne 64 ]; then exit 2; fi
case "$release_authority_sha256" in *[!0-9a-f]*) exit 2 ;; esac

repo_root="${VIDEOEDIT_REPO_ROOT:?set VIDEOEDIT_REPO_ROOT to the absolute repository root}"
python_bin="${ORACLE_PREFLIGHT_PYTHON:-python3}"
spec="$repo_root/methods/bernini_action_editing/assets/oracle_regeneration_e02_e03_canary_v1.json"
tool="$repo_root/methods/bernini_action_editing/tools/preflight_oracle_regeneration_e02_e03_canary_v1.py"

test -f "$spec"
test -f "$tool"
exec "$python_bin" -B "$tool" \
  --spec "$spec" \
  --case "$case_id" \
  --execution "$execution" \
  --release-authority "$release_authority" \
  --expected-release-authority-sha256 "$release_authority_sha256" \
  --require-launch-ready
