#!/usr/bin/env bash
set -euo pipefail

# Static review template only.  This does not authorize A, B, or C.  The R6
# package diagnoses whether the old QK route amplifies whitening; it is not a
# property-preservation fix and must not be presented as one.
if [ "$#" -ne 2 ]; then
  echo "usage: $0 PACKAGE_ROOT A|BC" >&2
  exit 2
fi
package_root="$(cd -- "$1" && pwd -P)"
phase="$2"
bootstrap="$package_root/methods/bernini_action_editing/tools/e00_three_vessel_clean_diag_r6_external_bootstrap.py"
root="$package_root/methods/bernini_action_editing/assets/e00_three_vessel_clean_diag_r6_EXTERNAL_ROOT.json"
bootstrap_sha256=278cdf295a9f95d5edf581d21b1c777f6789d21d87797ccf7cfab3735a4fc912
root_sha256=1139f83a4b27d633d83dc79276d4c8c4cf19c592fde63ffc1e6b911a6d70d4af
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

for file in "$bootstrap" "$root"; do test -f "$file"; test ! -L "$file"; done
actual_bootstrap_sha256="$(sha256sum -- "$bootstrap")"
actual_bootstrap_sha256="${actual_bootstrap_sha256%% *}"
test "$actual_bootstrap_sha256" = "$bootstrap_sha256"

case "$phase" in
  A)
    consumer=methods/bernini_action_editing/scripts/auh_launch_e00_three_vessel_clean_diag_r6_phase_a_only_node292.sh
    ;;
  BC)
    consumer=methods/bernini_action_editing/scripts/auh_launch_e00_three_vessel_clean_diag_r6_phase_bc_node292.sh
    ;;
  *)
    echo "phase must be A or BC" >&2
    exit 2
    ;;
esac

# The bootstrap independently checks the root SHA, every one-way pin, and the
# absence of cache bytecode before it executes the selected launcher from the
# already verified open file descriptor.
exec "$python_bin" -I -S -B "$bootstrap" \
  --package-root "$package_root" \
  --root "$root" \
  --expected-root-sha256 "$root_sha256" \
  --consumer-relative "$consumer" --
