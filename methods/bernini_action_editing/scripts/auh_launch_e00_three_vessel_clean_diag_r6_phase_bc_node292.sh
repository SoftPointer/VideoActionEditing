#!/usr/bin/env bash
set -euo pipefail

# R6 B/C launcher; unavailable unless A is independently reviewed and bit-exact.
if [ "$#" -ne 0 ]; then echo "usage: $0" >&2; exit 2; fi
case "$(hostname -s)" in auh-1b-cpu-login-002|auh7-ib|auh) ;; *) echo "run on AUH login host" >&2; exit 2 ;; esac
: "${E00_R6_BOOTSTRAP_VERIFIED:?missing external bootstrap proof}"
: "${E00_R6_PACKAGE_ROOT_VERIFIED:?missing verified package root}"
: "${E00_R6_ROOT_PATH_VERIFIED:?missing verified root path}"
: "${E00_R6_EXPECTED_ROOT_SHA256:?missing verified root SHA}"
: "${E00_R6_PHASE_BC_AUTHORIZATION:?missing E00_R6_PHASE_BC_AUTHORIZATION}"
: "${E00_R6_PHASE_BC_TOKEN:?missing E00_R6_PHASE_BC_TOKEN}"
: "${E00_R6_OUTPUT_ROOT:?missing E00_R6_OUTPUT_ROOT}"
test "$E00_R6_BOOTSTRAP_VERIFIED" = 1
if [ "${ALLOW_E00_R6_PHASE_BC:-}" != I_REVIEWED_R6_A_BITEXACT_AND_AUTHORIZE_B_GATE_C ]; then echo "R6 phase BC remains disabled" >&2; exit 3; fi

job=143808; node=auh7-1b-gpu-292; role_a=pure_noobserver_output_routeoff; role_b=observer_matched_output_routeoff; role_c=old_pureqk_temporal_routeon
package_root="$(cd -- "$E00_R6_PACKAGE_ROOT_VERIFIED" && pwd -P)"; method_root="$package_root/methods/bernini_action_editing"
root="$method_root/assets/e00_three_vessel_clean_diag_r6_EXTERNAL_ROOT.json"; test "$root" = "$E00_R6_ROOT_PATH_VERIFIED"
manifest="$package_root/e00-clean-diagnostic-r6-package.manifest.json"; protocol="$method_root/assets/e00_three_vessel_clean_diag_r6_protocol_20260821.json"
base_spec="$method_root/assets/e00_three_vessel_fresh_keyed_legacy_diagnostic_v1.json"; validator="$method_root/validate_e00_three_vessel_clean_diag_r6.py"
builder="$method_root/tools/build_e00_three_vessel_clean_diag_r6_package.py"; bootstrap="$method_root/tools/e00_three_vessel_clean_diag_r6_external_bootstrap.py"
bootstrap_sha256=278cdf295a9f95d5edf581d21b1c777f6789d21d87797ccf7cfab3735a4fc912
bridge_relative=methods/bernini_action_editing/scripts/auh_e00_three_vessel_clean_diag_r6_bridge.sh; marker_writer="$method_root/e00_r6_atomic_marker.py"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12; phase_a_marker="$E00_R6_OUTPUT_ROOT/E00_R6_PHASE_A_STOPPED_REVIEW_REQUIRED.json"
for command_name in squeue srun sha256sum jq find; do command -v "$command_name" >/dev/null; done
for file in "$root" "$manifest" "$protocol" "$base_spec" "$validator" "$builder" "$bootstrap" "$marker_writer" "$E00_R6_PHASE_BC_AUTHORIZATION" "$phase_a_marker"; do test -f "$file"; test ! -L "$file"; done
verify_sha() { local file="$1" expected="$2" actual; actual="$(sha256sum -- "$file")"; actual="${actual%% *}"; test "$actual" = "$expected"; }
check_cache_free() { local d f; d="$(find "$package_root" -type d -name __pycache__ -print -quit)"; f="$(find "$package_root" -type f -name '*.pyc' -print -quit)"; test -z "$d"; test -z "$f"; }
write_marker() { local output="$1"; "$python_bin" -B "$marker_writer" write --output "$output" >/dev/null; }
verify_sha "$bootstrap" "$bootstrap_sha256"; check_cache_free
"$python_bin" -B "$builder" verify --package-root "$package_root" --manifest "$manifest" --expected-root-sha256 "$E00_R6_EXPECTED_ROOT_SHA256" >/dev/null
"$python_bin" -B "$validator" protocol --protocol "$protocol" >/dev/null
token_sha="$(printf '%s' "$E00_R6_PHASE_BC_TOKEN" | sha256sum)"; token_sha="${token_sha%% *}"
"$python_bin" -B "$validator" bridge-capability --protocol "$protocol" --phase B --arm-role "$role_b" --package-manifest "$manifest" \
  --authorization "$E00_R6_PHASE_BC_AUTHORIZATION" --capability-token-sha256 "$token_sha" --phase-a-marker "$phase_a_marker" >/dev/null
check_cache_free
label_a="$(jq -er --arg role "$role_a" '.arms[]|select(.arm_role==$role)|.label' "$base_spec")"; label_b="$(jq -er --arg role "$role_b" '.arms[]|select(.arm_role==$role)|.label' "$base_spec")"; label_c="$(jq -er --arg role "$role_c" '.arms[]|select(.arm_role==$role)|.label' "$base_spec")"
video_a="$E00_R6_OUTPUT_ROOT/$label_a.mp4"; audit_a="$video_a.e00-r6-arm-audit.json"; video_b="$E00_R6_OUTPUT_ROOT/$label_b.mp4"; audit_b="$video_b.e00-r6-arm-audit.json"; video_c="$E00_R6_OUTPUT_ROOT/$label_c.mp4"; audit_c="$video_c.e00-r6-arm-audit.json"
for path in "$video_a" "$audit_a"; do test -f "$path"; test ! -L "$path"; done; for path in "$video_b" "$audit_b" "$video_c" "$audit_c"; do test ! -e "$path"; test ! -L "$path"; done
state="$(squeue -h -j "$job" -w "$node" -o '%T' | head -n 1)"; test "$state" = RUNNING
started="$E00_R6_OUTPUT_ROOT/e00-r6-phase-bc.started.json"; failed="$E00_R6_OUTPUT_ROOT/e00-r6-phase-bc.failed.json"; gate="$E00_R6_OUTPUT_ROOT/e00-r6-ab-current-bit-exact.gate.json"; final_audit="$E00_R6_OUTPUT_ROOT/e00-r6-final-current-artifact.audit.json"; completed="$E00_R6_OUTPUT_ROOT/E00_R6_ABC_DIAGNOSTIC_COMPLETE.json"
for path in "$started" "$failed" "$gate" "$final_audit" "$completed"; do test ! -e "$path"; test ! -L "$path"; done
lock="$E00_R6_OUTPUT_ROOT/.r6-phase-bc-lock"; cleanup_lock() { rmdir "$lock" 2>/dev/null || true; }
finalize() { status="$?"; trap - EXIT; cleanup_lock; if [ "$status" -ne 0 ]; then if [ -f "$started" ]; then jq -S --argjson code "$status" '.state="bc_failed_c_not_admitted_or_incomplete"|.exit_code=$code|.terminal_marker_atomic_write_verified=true' "$started" | write_marker "$failed"; else jq -n -S --argjson code "$status" '{schema_version:"bernini-e00-clean-diagnostic-r6-phase-bc-failed-v6",state:"bc_failed_c_not_admitted_or_incomplete",exit_code:$code,terminal_marker_atomic_write_verified:true}' | write_marker "$failed"; fi; fi; exit "$status"; }
mkdir "$lock"
trap finalize EXIT
manifest_sha="$(sha256sum -- "$manifest")"; manifest_sha="${manifest_sha%% *}"; auth_sha="$(sha256sum -- "$E00_R6_PHASE_BC_AUTHORIZATION")"; auth_sha="${auth_sha%% *}"; marker_sha="$(sha256sum -- "$phase_a_marker")"; marker_sha="${marker_sha%% *}"
jq -n -S --arg job "$job" --arg node "$node" --arg manifest_sha "$manifest_sha" --arg auth "$E00_R6_PHASE_BC_AUTHORIZATION" --arg auth_sha "$auth_sha" --arg token_sha "$token_sha" --arg marker "$phase_a_marker" --arg marker_sha "$marker_sha" --arg role_b "$role_b" --arg role_c "$role_c" '
 {schema_version:"bernini-e00-clean-diagnostic-r6-phase-bc-launch-v6",revision_tag:"E00_DFIX2_CLEAN_DIAG_R6_EXTERNAL_BOOTSTRAP_20260821",state:"b_started_c_forbidden",complete:false,training_performed:false,optimization_steps:0,parent_job_id:$job,compute_node:$node,package_manifest_sha256:$manifest_sha,authorization:{path:$auth,sha256:$auth_sha},capability_token_sha256:$token_sha,phase_a_marker:{path:$marker,sha256:$marker_sha},authorized_arm_order:[$role_b,$role_c],a_b_bit_exact_or_c_forbidden:true,old_qk_route_white_leakage_diagnostic_only:true,property_preservation_fix_claimed:false,terminal_marker_atomic_write_verified:true}' | write_marker "$started"
run_arm() { local phase="$1" role="$2"; shift 2; check_cache_free; srun --jobid="$job" --exclusive --nodes=1 --ntasks=1 --cpus-per-task=64 --gres=gpu:4 --mem=0 --nodelist="$node" env E00_R6_ARM_ROLE="$role" E00_R6_PHASE="$phase" E00_R6_OUTPUT_ROOT="$E00_R6_OUTPUT_ROOT" E00_R6_PACKAGE_MANIFEST="$manifest" E00_R6_AUTHORIZATION="$E00_R6_PHASE_BC_AUTHORIZATION" E00_R6_CAPABILITY_TOKEN="$E00_R6_PHASE_BC_TOKEN" E00_R6_PHASE_A_MARKER="$phase_a_marker" "$@" "$python_bin" -I -S -B "$bootstrap" --package-root "$package_root" --root "$root" --expected-root-sha256 "$E00_R6_EXPECTED_ROOT_SHA256" --consumer-relative "$bridge_relative" --; check_cache_free; }
run_arm B "$role_b"
"$python_bin" -B "$validator" ab-gate --protocol "$protocol" --a-audit "$audit_a" --b-audit "$audit_b" --gate-output "$gate"
"$python_bin" -B "$validator" verify-ab-gate --protocol "$protocol" --ab-gate "$gate" >/dev/null
# Any A/B latent or MP4 mismatch exits here; C is never invoked.
run_arm C "$role_c" E00_R6_AB_GATE="$gate"
"$python_bin" -B "$validator" abc-final --protocol "$protocol" --a-audit "$audit_a" --b-audit "$audit_b" --c-audit "$audit_c" --ab-gate "$gate" --audit-output "$final_audit"
jq -e '.old_qk_route_white_leakage_diagnostic_only==true and .property_preservation_fix_claimed==false and .a_b_bit_exact_or_c_forbidden==true and (.retained_three_object_instruction_and_source_anchor|type=="object")' "$final_audit" >/dev/null
gate_sha="$(sha256sum -- "$gate")"; gate_sha="${gate_sha%% *}"; final_sha="$(sha256sum -- "$final_audit")"; final_sha="${final_sha%% *}"
jq -S --arg gate "$gate" --arg gate_sha "$gate_sha" --arg final "$final_audit" --arg final_sha "$final_sha" '.state="abc_completed"|.complete=true|.ab_gate={path:$gate,sha256:$gate_sha}|.final_audit={path:$final,sha256:$final_sha}|.terminal_marker_atomic_write_verified=true' "$started" | write_marker "$completed"
jq -e '.terminal_marker_atomic_write_verified==true and .state=="abc_completed"' "$completed" >/dev/null
check_cache_free; cleanup_lock; trap - EXIT
printf 'E00_R6_ABC_DIAGNOSTIC_COMPLETE %s\n' "$final_audit"
