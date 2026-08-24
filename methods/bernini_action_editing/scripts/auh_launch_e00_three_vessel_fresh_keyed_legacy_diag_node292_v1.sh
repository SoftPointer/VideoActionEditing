#!/usr/bin/env bash
set -euo pipefail

# Login-host launcher for the reviewed immutable three-arm diagnostic package.
# This file is intentionally inert until an external authorization receipt and
# an explicit environment acknowledgement are both present.  The three srun
# steps are serial and never overlap on node292.
if [ "$#" -ne 0 ]; then echo "usage: $0" >&2; exit 2; fi
case "$(hostname -s)" in auh-1b-cpu-login-002|auh7-ib|auh) ;; *) echo "run on AUH login host" >&2; exit 2 ;; esac
: "${E00_DIAG_PACKAGE_ROOT:?missing E00_DIAG_PACKAGE_ROOT}"
: "${E00_DIAG_AUTHORIZATION:?missing E00_DIAG_AUTHORIZATION}"
: "${E00_DIAG_OUTPUT_ROOT:?missing E00_DIAG_OUTPUT_ROOT}"
if [ "${ALLOW_E00_THREE_VESSEL_DIAGNOSTIC:-}" != I_REVIEWED_THE_IMMUTABLE_THREE_ARM_PACKAGE ]; then
  echo "execution remains disabled pending explicit package review" >&2
  exit 3
fi

job=143808
node=auh7-1b-gpu-292
package_root="$(cd -- "$E00_DIAG_PACKAGE_ROOT" && pwd -P)"
method_root="$package_root/methods/bernini_action_editing"
manifest="$package_root/e00-fresh-keyed-legacy-package.manifest.json"
spec="$method_root/assets/e00_three_vessel_fresh_keyed_legacy_diagnostic_v1.json"
bridge="$method_root/scripts/auh_e00_three_vessel_fresh_keyed_legacy_bridge_v1.sh"
validator="$method_root/validate_e00_three_vessel_fresh_keyed_legacy_diagnostic_v1.py"
builder="$method_root/tools/build_e00_three_vessel_fresh_keyed_legacy_package_v1.py"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

for command_name in squeue srun sha256sum jq; do command -v "$command_name" >/dev/null; done
for file in "$manifest" "$spec" "$bridge" "$validator" "$builder" "$E00_DIAG_AUTHORIZATION"; do test -f "$file"; test ! -L "$file"; done
"$python_bin" -B "$builder" verify --package-root "$package_root" --manifest "$manifest" >/dev/null
"$python_bin" -B "$validator" spec --spec "$spec" >/dev/null
manifest_sha="$(sha256sum -- "$manifest")"; manifest_sha="${manifest_sha%% *}"
jq -e --arg sha "$manifest_sha" --arg job "$job" --arg node "$node" '
  .schema_version == "bernini-e00-three-vessel-execution-authorization-v1" and
  .execution_authorized == true and
  .package_manifest_sha256 == $sha and
  .parent_job_id == $job and .compute_node == $node and
  .serial_three_arm_order == [
    "pure_noobserver_output_routeoff",
    "observer_matched_output_routeoff",
    "old_pureqk_temporal_routeon"
  ] and
  .sp4_observer_released_node292 == true and
  (.authorized_by | type == "string" and length > 0)
' "$E00_DIAG_AUTHORIZATION" >/dev/null
state="$(squeue -h -j "$job" -w "$node" -o '%T' | head -n 1)"
test "$state" = RUNNING
test ! -e "$E00_DIAG_OUTPUT_ROOT"; test ! -L "$E00_DIAG_OUTPUT_ROOT"
mkdir -p "$E00_DIAG_OUTPUT_ROOT"
lock="$E00_DIAG_OUTPUT_ROOT/.serial-launch-lock"
mkdir "$lock"
cleanup_lock() { rmdir "$lock" 2>/dev/null || true; }

authorization_sha="$(sha256sum -- "$E00_DIAG_AUTHORIZATION")"; authorization_sha="${authorization_sha%% *}"
started_receipt="$E00_DIAG_OUTPUT_ROOT/e00-three-arm-launch.started.json"
failed_receipt="$E00_DIAG_OUTPUT_ROOT/e00-three-arm-launch.failed.json"
completed_receipt="$E00_DIAG_OUTPUT_ROOT/e00-three-arm-launch.completed.json"
jq -n -S \
  --arg job "$job" --arg node "$node" --arg package "$package_root" \
  --arg manifest "$manifest" --arg manifest_sha "$manifest_sha" \
  --arg auth "$E00_DIAG_AUTHORIZATION" --arg auth_sha "$authorization_sha" \
  --arg output "$E00_DIAG_OUTPUT_ROOT" '
  {
    schema_version:"bernini-e00-three-vessel-serial-launch-receipt-v1",
    state:"started", complete:false, training_performed:false, optimization_steps:0,
    parent_job_id:$job, compute_node:$node, package_root:$package,
    package_manifest:{path:$manifest,sha256:$manifest_sha},
    authorization:{path:$auth,sha256:$auth_sha}, output_root:$output,
    serial_order:[
      "pure_noobserver_output_routeoff",
      "observer_matched_output_routeoff",
      "old_pureqk_temporal_routeon"
    ]
  }
' > "$started_receipt"

finalize_launch() {
  status="$?"
  trap - EXIT
  cleanup_lock
  if [ "$status" -ne 0 ]; then
    jq -S --argjson exit_code "$status" \
      '.state="failed" | .complete=false | .exit_code=$exit_code' \
      "$started_receipt" > "$failed_receipt"
  fi
  exit "$status"
}
trap finalize_launch EXIT

run_arm() {
  local role="$1"
  srun --jobid="$job" --exclusive --nodes=1 --ntasks=1 --cpus-per-task=64 \
    --gres=gpu:4 --mem=0 --nodelist="$node" \
    env E00_DIAG_ARM_ROLE="$role" E00_DIAG_OUTPUT_ROOT="$E00_DIAG_OUTPUT_ROOT" \
      E00_DIAG_PACKAGE_MANIFEST="$manifest" \
      EXPECTED_COMPUTE_NODE="$node" EXPECTED_PARENT_JOB="$job" \
      bash "$bridge"
}

run_arm pure_noobserver_output_routeoff
run_arm observer_matched_output_routeoff
run_arm old_pureqk_temporal_routeon

label_a="$(jq -er '.arms[] | select(.arm_role == "pure_noobserver_output_routeoff") | .label' "$spec")"
label_b="$(jq -er '.arms[] | select(.arm_role == "observer_matched_output_routeoff") | .label' "$spec")"
label_c="$(jq -er '.arms[] | select(.arm_role == "old_pureqk_temporal_routeon") | .label' "$spec")"
audit_a="$E00_DIAG_OUTPUT_ROOT/$label_a.mp4.e00-legacy-audit.json"
audit_b="$E00_DIAG_OUTPUT_ROOT/$label_b.mp4.e00-legacy-audit.json"
audit_c="$E00_DIAG_OUTPUT_ROOT/$label_c.mp4.e00-legacy-audit.json"
pair_audit="$E00_DIAG_OUTPUT_ROOT/e00-three-arm-matched-pair.audit.json"
"$python_bin" -B "$validator" pair \
  --pure-noobserver-audit "$audit_a" \
  --observer-routeoff-audit "$audit_b" \
  --observer-routeon-audit "$audit_c" \
  --audit-output "$pair_audit"
jq -S --arg pair_audit "$pair_audit" \
  '.state="completed" | .complete=true | .pair_audit=$pair_audit' \
  "$started_receipt" > "$completed_receipt"
touch "$E00_DIAG_OUTPUT_ROOT/E00_THREE_ARM_DIAGNOSTIC_COMPLETE"
printf 'E00_THREE_ARM_DIAGNOSTIC_COMPLETE %s\n' "$pair_audit"
