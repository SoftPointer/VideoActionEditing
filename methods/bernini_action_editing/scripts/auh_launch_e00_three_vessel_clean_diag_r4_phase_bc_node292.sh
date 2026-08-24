#!/usr/bin/env bash
set -euo pipefail

# R4 BC launcher: B, current-artifact gate, then capability-revalidated C.
if [ "$#" -ne 0 ]; then echo "usage: $0" >&2; exit 2; fi
case "$(hostname -s)" in auh-1b-cpu-login-002|auh7-ib|auh) ;; *) echo "run on AUH login host" >&2; exit 2 ;; esac
: "${E00_R4_PACKAGE_ROOT:?missing E00_R4_PACKAGE_ROOT}"
: "${E00_R4_PHASE_BC_AUTHORIZATION:?missing E00_R4_PHASE_BC_AUTHORIZATION}"
: "${E00_R4_PHASE_BC_TOKEN:?missing E00_R4_PHASE_BC_TOKEN}"
: "${E00_R4_OUTPUT_ROOT:?missing E00_R4_OUTPUT_ROOT}"
if [ "${ALLOW_E00_R4_PHASE_BC:-}" != I_REVIEWED_R4_A_AND_AUTHORIZE_B_GATE_C ]; then
  echo "R4 phase BC remains disabled" >&2; exit 3
fi

job=143808
node=auh7-1b-gpu-292
role_a=pure_noobserver_output_routeoff
role_b=observer_matched_output_routeoff
role_c=old_pureqk_temporal_routeon
package_root="$(cd -- "$E00_R4_PACKAGE_ROOT" && pwd -P)"
method_root="$package_root/methods/bernini_action_editing"
manifest="$package_root/e00-clean-diagnostic-r4-package.manifest.json"
review_marker="$package_root/R4_EXECUTION_REVIEW_REQUIRED.json"
protocol="$method_root/assets/e00_three_vessel_clean_diag_r4_protocol_20260821.json"
base_spec="$method_root/assets/e00_three_vessel_fresh_keyed_legacy_diagnostic_v1.json"
bridge="$method_root/scripts/auh_e00_three_vessel_clean_diag_r4_bridge.sh"
validator="$method_root/validate_e00_three_vessel_clean_diag_r4.py"
builder="$method_root/tools/build_e00_three_vessel_clean_diag_r4_package.py"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
builder_sha256=9f3e51a68b2360a6a6f5c4e0d1b6107742e0c8b5acdd6e278760fd73f6beb08a
phase_a_marker="$E00_R4_OUTPUT_ROOT/E00_R4_PHASE_A_STOPPED_REVIEW_REQUIRED.json"

for command_name in squeue srun sha256sum jq find; do command -v "$command_name" >/dev/null; done
for file in "$manifest" "$review_marker" "$protocol" "$base_spec" "$bridge" "$validator" "$builder" "$E00_R4_PHASE_BC_AUTHORIZATION" "$phase_a_marker"; do
  test -f "$file"; test ! -L "$file"
done
verify_sha() {
  local file="$1" expected="$2" actual
  actual="$(sha256sum -- "$file")"; actual="${actual%% *}"; test "$actual" = "$expected"
}
check_cache_free() {
  local cache_dir cache_file
  cache_dir="$(find "$package_root" -type d -name __pycache__ -print -quit)"
  cache_file="$(find "$package_root" -type f -name '*.pyc' -print -quit)"
  test -z "$cache_dir"; test -z "$cache_file"
}
verify_sha "$builder" "$builder_sha256"
check_cache_free
"$python_bin" -B "$builder" verify --package-root "$package_root" --manifest "$manifest" >/dev/null
"$python_bin" -B "$validator" protocol --protocol "$protocol" >/dev/null
token_sha="$(printf '%s' "$E00_R4_PHASE_BC_TOKEN" | sha256sum)"; token_sha="${token_sha%% *}"
"$python_bin" -B "$validator" bridge-capability --protocol "$protocol" \
  --phase B --arm-role "$role_b" --package-manifest "$manifest" \
  --authorization "$E00_R4_PHASE_BC_AUTHORIZATION" \
  --capability-token-sha256 "$token_sha" --phase-a-marker "$phase_a_marker" >/dev/null
check_cache_free

label_a="$(jq -er --arg role "$role_a" '.arms[]|select(.arm_role==$role)|.label' "$base_spec")"
label_b="$(jq -er --arg role "$role_b" '.arms[]|select(.arm_role==$role)|.label' "$base_spec")"
label_c="$(jq -er --arg role "$role_c" '.arms[]|select(.arm_role==$role)|.label' "$base_spec")"
video_a="$E00_R4_OUTPUT_ROOT/$label_a.mp4"; audit_a="$video_a.e00-r4-arm-audit.json"
video_b="$E00_R4_OUTPUT_ROOT/$label_b.mp4"; audit_b="$video_b.e00-r4-arm-audit.json"
video_c="$E00_R4_OUTPUT_ROOT/$label_c.mp4"; audit_c="$video_c.e00-r4-arm-audit.json"
for path in "$video_a" "$audit_a"; do test -f "$path"; test ! -L "$path"; done
for path in "$video_b" "$audit_b" "$video_c" "$audit_c"; do test ! -e "$path"; test ! -L "$path"; done

state="$(squeue -h -j "$job" -w "$node" -o '%T' | head -n 1)"; test "$state" = RUNNING
lock="$E00_R4_OUTPUT_ROOT/.r4-phase-bc-lock"; mkdir "$lock"
cleanup_lock() { rmdir "$lock" 2>/dev/null || true; }
manifest_sha="$(sha256sum -- "$manifest")"; manifest_sha="${manifest_sha%% *}"
auth_sha="$(sha256sum -- "$E00_R4_PHASE_BC_AUTHORIZATION")"; auth_sha="${auth_sha%% *}"
marker_sha="$(sha256sum -- "$phase_a_marker")"; marker_sha="${marker_sha%% *}"
started="$E00_R4_OUTPUT_ROOT/e00-r4-phase-bc.started.json"
failed="$E00_R4_OUTPUT_ROOT/e00-r4-phase-bc.failed.json"
gate="$E00_R4_OUTPUT_ROOT/e00-r4-ab-current-bit-exact.gate.json"
final_audit="$E00_R4_OUTPUT_ROOT/e00-r4-final-current-artifact.audit.json"
completed="$E00_R4_OUTPUT_ROOT/E00_R4_ABC_DIAGNOSTIC_COMPLETE.json"
for path in "$started" "$failed" "$gate" "$final_audit" "$completed"; do test ! -e "$path"; test ! -L "$path"; done
jq -n -S --arg job "$job" --arg node "$node" --arg manifest_sha "$manifest_sha" \
  --arg auth "$E00_R4_PHASE_BC_AUTHORIZATION" --arg auth_sha "$auth_sha" \
  --arg token_sha "$token_sha" --arg marker "$phase_a_marker" --arg marker_sha "$marker_sha" \
  --arg role_b "$role_b" --arg role_c "$role_c" '
  {
    schema_version:"bernini-e00-clean-diagnostic-r4-phase-bc-launch-v4",
    revision_tag:"E00_DFIX2_CLEAN_DIAG_R4_OVERLAY_CACHE_CLOSURE_20260821",
    state:"b_started_c_forbidden",complete:false,training_performed:false,optimization_steps:0,
    parent_job_id:$job,compute_node:$node,package_manifest_sha256:$manifest_sha,
    authorization:{path:$auth,sha256:$auth_sha},capability_token_sha256:$token_sha,
    phase_a_marker:{path:$marker,sha256:$marker_sha},authorized_arm_order:[$role_b,$role_c],
    package_cache_bytecode_scan_passed:true,
    c_requires_bridge_revalidated_current_ab_gate:true
  }
' > "$started"
finalize() {
  status="$?"; trap - EXIT; cleanup_lock
  if [ "$status" -ne 0 ]; then jq -S --argjson code "$status" '.state="bc_failed_c_not_admitted_or_incomplete"|.exit_code=$code' "$started" > "$failed"; fi
  exit "$status"
}
trap finalize EXIT

run_arm() {
  local phase="$1" role="$2"
  shift 2
  check_cache_free
  srun --jobid="$job" --exclusive --nodes=1 --ntasks=1 --cpus-per-task=64 \
    --gres=gpu:4 --mem=0 --nodelist="$node" \
    env E00_R4_ARM_ROLE="$role" E00_R4_PHASE="$phase" \
      E00_R4_OUTPUT_ROOT="$E00_R4_OUTPUT_ROOT" E00_R4_PACKAGE_MANIFEST="$manifest" \
      E00_R4_AUTHORIZATION="$E00_R4_PHASE_BC_AUTHORIZATION" \
      E00_R4_CAPABILITY_TOKEN="$E00_R4_PHASE_BC_TOKEN" \
      E00_R4_PHASE_A_MARKER="$phase_a_marker" "$@" bash "$bridge"
  check_cache_free
}

run_arm B "$role_b"

# Rebuild A and B arm audits from current native receipt, four per-rank RNG
# receipts, and MP4 bytes. C remains unreachable on any mismatch.
"$python_bin" -B "$validator" ab-gate --protocol "$protocol" \
  --a-audit "$audit_a" --b-audit "$audit_b" --gate-output "$gate"
"$python_bin" -B "$validator" verify-ab-gate --protocol "$protocol" --ab-gate "$gate" >/dev/null
check_cache_free

# The compute bridge independently repeats marker/auth/gate/current-artifact
# validation before it imports Torch for C.
run_arm C "$role_c" E00_R4_AB_GATE="$gate"

# Final closure again reloads every arm's native receipt and all four rank RNG
# receipts from the paths sealed in each arm audit.
"$python_bin" -B "$validator" abc-final --protocol "$protocol" \
  --a-audit "$audit_a" --b-audit "$audit_b" --c-audit "$audit_c" \
  --ab-gate "$gate" --audit-output "$final_audit"
check_cache_free
gate_sha="$(sha256sum -- "$gate")"; gate_sha="${gate_sha%% *}"
final_sha="$(sha256sum -- "$final_audit")"; final_sha="${final_sha%% *}"
jq -S --arg gate "$gate" --arg gate_sha "$gate_sha" --arg final "$final_audit" --arg final_sha "$final_sha" '
  .state="abc_completed"|.complete=true|.ab_gate={path:$gate,sha256:$gate_sha}|.final_audit={path:$final,sha256:$final_sha}
' "$started" > "$completed"
check_cache_free
cleanup_lock; trap - EXIT
printf 'E00_R4_ABC_DIAGNOSTIC_COMPLETE %s\n' "$final_audit"
