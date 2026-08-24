#!/usr/bin/env bash
set -euo pipefail

# R6 phase A launcher; diagnostic of the old QK route, not a preservation fix.
if [ "$#" -ne 0 ]; then echo "usage: $0" >&2; exit 2; fi
case "$(hostname -s)" in auh-1b-cpu-login-002|auh7-ib|auh) ;; *) echo "run on AUH login host" >&2; exit 2 ;; esac
: "${E00_R6_BOOTSTRAP_VERIFIED:?missing external bootstrap proof}"
: "${E00_R6_PACKAGE_ROOT_VERIFIED:?missing verified package root}"
: "${E00_R6_ROOT_PATH_VERIFIED:?missing verified root path}"
: "${E00_R6_EXPECTED_ROOT_SHA256:?missing verified root SHA}"
: "${E00_R6_PHASE_A_AUTHORIZATION:?missing E00_R6_PHASE_A_AUTHORIZATION}"
: "${E00_R6_PHASE_A_TOKEN:?missing E00_R6_PHASE_A_TOKEN}"
: "${E00_R6_OUTPUT_ROOT:?missing E00_R6_OUTPUT_ROOT}"
test "$E00_R6_BOOTSTRAP_VERIFIED" = 1
if [ "${ALLOW_E00_R6_PHASE_A_ONLY:-}" != I_REVIEWED_R6_AND_AUTHORIZE_A_ONLY_THEN_STOP ]; then echo "R6 phase A remains disabled" >&2; exit 3; fi

job=143808; node=auh7-1b-gpu-292; role_a=pure_noobserver_output_routeoff
package_root="$(cd -- "$E00_R6_PACKAGE_ROOT_VERIFIED" && pwd -P)"
method_root="$package_root/methods/bernini_action_editing"
root="$method_root/assets/e00_three_vessel_clean_diag_r6_EXTERNAL_ROOT.json"
test "$root" = "$E00_R6_ROOT_PATH_VERIFIED"
manifest="$package_root/e00-clean-diagnostic-r6-package.manifest.json"
protocol="$method_root/assets/e00_three_vessel_clean_diag_r6_protocol_20260821.json"
base_spec="$method_root/assets/e00_three_vessel_fresh_keyed_legacy_diagnostic_v1.json"
validator="$method_root/validate_e00_three_vessel_clean_diag_r6.py"
builder="$method_root/tools/build_e00_three_vessel_clean_diag_r6_package.py"
bootstrap="$method_root/tools/e00_three_vessel_clean_diag_r6_external_bootstrap.py"
bootstrap_sha256=278cdf295a9f95d5edf581d21b1c777f6789d21d87797ccf7cfab3735a4fc912
bridge_relative=methods/bernini_action_editing/scripts/auh_e00_three_vessel_clean_diag_r6_bridge.sh
marker_writer="$method_root/e00_r6_atomic_marker.py"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

for command_name in squeue srun sha256sum jq find; do command -v "$command_name" >/dev/null; done
for file in "$root" "$manifest" "$protocol" "$base_spec" "$validator" "$builder" "$bootstrap" "$marker_writer" "$E00_R6_PHASE_A_AUTHORIZATION"; do test -f "$file"; test ! -L "$file"; done
verify_sha() { local file="$1" expected="$2" actual; actual="$(sha256sum -- "$file")"; actual="${actual%% *}"; test "$actual" = "$expected"; }
check_cache_free() { local d f; d="$(find "$package_root" -type d -name __pycache__ -print -quit)"; f="$(find "$package_root" -type f -name '*.pyc' -print -quit)"; test -z "$d"; test -z "$f"; }
write_marker() { local output="$1"; "$python_bin" -B "$marker_writer" write --output "$output" >/dev/null; }
verify_sha "$bootstrap" "$bootstrap_sha256"
check_cache_free
"$python_bin" -B "$builder" verify --package-root "$package_root" --manifest "$manifest" --expected-root-sha256 "$E00_R6_EXPECTED_ROOT_SHA256" >/dev/null
"$python_bin" -B "$validator" protocol --protocol "$protocol" >/dev/null
token_sha="$(printf '%s' "$E00_R6_PHASE_A_TOKEN" | sha256sum)"; token_sha="${token_sha%% *}"
"$python_bin" -B "$validator" bridge-capability --protocol "$protocol" --phase A --arm-role "$role_a" --package-manifest "$manifest" \
  --authorization "$E00_R6_PHASE_A_AUTHORIZATION" --capability-token-sha256 "$token_sha" >/dev/null
check_cache_free
state="$(squeue -h -j "$job" -w "$node" -o '%T' | head -n 1)"; test "$state" = RUNNING
test ! -e "$E00_R6_OUTPUT_ROOT"; test ! -L "$E00_R6_OUTPUT_ROOT"; mkdir -p "$E00_R6_OUTPUT_ROOT"
started="$E00_R6_OUTPUT_ROOT/e00-r6-phase-a.started.json"; failed="$E00_R6_OUTPUT_ROOT/e00-r6-phase-a.failed.json"; stopped="$E00_R6_OUTPUT_ROOT/E00_R6_PHASE_A_STOPPED_REVIEW_REQUIRED.json"
lock="$E00_R6_OUTPUT_ROOT/.r6-phase-a-lock"
cleanup_lock() { rmdir "$lock" 2>/dev/null || true; }
finalize() {
  status="$?"; trap - EXIT; cleanup_lock
  if [ "$status" -ne 0 ]; then
    if [ -f "$started" ]; then jq -S --argjson code "$status" '.state="a_failed"|.exit_code=$code|.terminal_marker_atomic_write_verified=true' "$started" | write_marker "$failed"
    else jq -n -S --argjson code "$status" '{schema_version:"bernini-e00-clean-diagnostic-r6-phase-a-failed-v6",state:"a_failed",exit_code:$code,terminal_marker_atomic_write_verified:true}' | write_marker "$failed"; fi
  fi
  exit "$status"
}
mkdir "$lock"
trap finalize EXIT

manifest_sha="$(sha256sum -- "$manifest")"; manifest_sha="${manifest_sha%% *}"; protocol_sha="$(sha256sum -- "$protocol")"; protocol_sha="${protocol_sha%% *}"
protocol_canonical="$("$python_bin" -B "$validator" protocol --protocol "$protocol" | jq -er '.canonical_sha256')"; auth_sha="$(sha256sum -- "$E00_R6_PHASE_A_AUTHORIZATION")"; auth_sha="${auth_sha%% *}"
jq -n -S --arg job "$job" --arg node "$node" --arg manifest_sha "$manifest_sha" --arg root_sha "$E00_R6_EXPECTED_ROOT_SHA256" \
  --arg auth "$E00_R6_PHASE_A_AUTHORIZATION" --arg auth_sha "$auth_sha" --arg token_sha "$token_sha" --arg role "$role_a" '
  {schema_version:"bernini-e00-clean-diagnostic-r6-phase-a-launch-v6",revision_tag:"E00_DFIX2_CLEAN_DIAG_R6_EXTERNAL_BOOTSTRAP_20260821",
   state:"a_started",complete:false,training_performed:false,optimization_steps:0,parent_job_id:$job,compute_node:$node,
   package_manifest_sha256:$manifest_sha,external_root_sha256:$root_sha,authorization:{path:$auth,sha256:$auth_sha},capability_token_sha256:$token_sha,
   authorized_arm_order:[$role],must_stop_after_a:true,bc_execution_authorized:false,old_qk_route_white_leakage_diagnostic_only:true,
   property_preservation_fix_claimed:false,terminal_marker_atomic_write_verified:true}' | write_marker "$started"

check_cache_free
srun --jobid="$job" --exclusive --nodes=1 --ntasks=1 --cpus-per-task=64 --gres=gpu:4 --mem=0 --nodelist="$node" \
  env E00_R6_ARM_ROLE="$role_a" E00_R6_PHASE=A E00_R6_OUTPUT_ROOT="$E00_R6_OUTPUT_ROOT" E00_R6_PACKAGE_MANIFEST="$manifest" \
    E00_R6_AUTHORIZATION="$E00_R6_PHASE_A_AUTHORIZATION" E00_R6_CAPABILITY_TOKEN="$E00_R6_PHASE_A_TOKEN" \
    "$python_bin" -I -S -B "$bootstrap" --package-root "$package_root" --root "$root" --expected-root-sha256 "$E00_R6_EXPECTED_ROOT_SHA256" \
    --consumer-relative "$bridge_relative" --
check_cache_free

label_a="$(jq -er --arg role "$role_a" '.arms[]|select(.arm_role==$role)|.label' "$base_spec")"; video_a="$E00_R6_OUTPUT_ROOT/$label_a.mp4"; audit_a="$video_a.e00-r6-arm-audit.json"
for path in "$video_a" "$audit_a"; do test -f "$path"; test ! -L "$path"; done
video_sha="$(sha256sum -- "$video_a")"; video_sha="${video_sha%% *}"; audit_sha="$(sha256sum -- "$audit_a")"; audit_sha="${audit_sha%% *}"
jq -n -S --arg job "$job" --arg node "$node" --arg manifest_sha "$manifest_sha" --arg protocol_sha "$protocol_sha" --arg protocol_canonical "$protocol_canonical" \
  --arg auth "$E00_R6_PHASE_A_AUTHORIZATION" --arg auth_sha "$auth_sha" --arg token_sha "$token_sha" --arg role "$role_a" \
  --arg video "$video_a" --arg video_sha "$video_sha" --arg audit "$audit_a" --arg audit_sha "$audit_sha" '
  {schema_version:"bernini-e00-clean-diagnostic-r6-phase-a-stopped-marker-v6",revision_tag:"E00_DFIX2_CLEAN_DIAG_R6_EXTERNAL_BOOTSTRAP_20260821",
   state:"A_STOPPED_REVIEW_REQUIRED",complete:true,training_performed:false,optimization_steps:0,parent_job_id:$job,compute_node:$node,
   package_manifest_sha256:$manifest_sha,protocol_file_sha256:$protocol_sha,protocol_canonical_sha256:$protocol_canonical,
   phase_a_authorization:{path:$auth,sha256:$auth_sha},phase_a_capability_token_sha256:$token_sha,observed_arm_order:[$role],
   phase_a_video:{path:$video,sha256:$video_sha},phase_a_arm_audit:{path:$audit,sha256:$audit_sha},must_stop_after_a:true,
   phase_bc_execution_authorized:false,old_qk_route_white_leakage_diagnostic_only:true,property_preservation_fix_claimed:false,
   terminal_marker_atomic_write_verified:true,next_required_state:"INDEPENDENT_A_PHASE_REVIEW_ONLY"}' | write_marker "$stopped"
jq -e '.terminal_marker_atomic_write_verified==true and .state=="A_STOPPED_REVIEW_REQUIRED"' "$stopped" >/dev/null
check_cache_free; cleanup_lock; trap - EXIT
printf 'E00_R6_PHASE_A_STOPPED_REVIEW_REQUIRED %s\n' "$stopped"
