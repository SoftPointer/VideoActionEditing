#!/usr/bin/env bash
set -euo pipefail

# Phase BC launcher.  It requires a separately authorized, byte-bound A stop
# marker.  It runs B first, immediately recomputes the A/B bit-exact gate, and
# cannot reach the C srun unless that gate succeeds.
if [ "$#" -ne 0 ]; then echo "usage: $0" >&2; exit 2; fi
case "$(hostname -s)" in auh-1b-cpu-login-002|auh7-ib|auh) ;; *) echo "run on AUH login host" >&2; exit 2 ;; esac
: "${E00_R2_PACKAGE_ROOT:?missing E00_R2_PACKAGE_ROOT}"
: "${E00_R2_PHASE_BC_AUTHORIZATION:?missing E00_R2_PHASE_BC_AUTHORIZATION}"
: "${E00_R2_OUTPUT_ROOT:?missing E00_R2_OUTPUT_ROOT}"
if [ "${ALLOW_E00_R2_PHASE_BC:-}" != I_REVIEWED_PHASE_A_AND_AUTHORIZE_B_GATE_C ]; then
  echo "phase BC remains disabled pending review of the completed A bytes" >&2
  exit 3
fi

job=143808
node=auh7-1b-gpu-292
role_a=pure_noobserver_output_routeoff
role_b=observer_matched_output_routeoff
role_c=old_pureqk_temporal_routeon
package_root="$(cd -- "$E00_R2_PACKAGE_ROOT" && pwd -P)"
method_root="$package_root/methods/bernini_action_editing"
manifest="$package_root/e00-clean-diagnostic-r2-package.manifest.json"
review_marker="$package_root/EXECUTION_REVIEW_REQUIRED.json"
protocol="$method_root/assets/e00_three_vessel_fresh_keyed_two_phase_diagnostic_v2.json"
base_spec="$method_root/assets/e00_three_vessel_fresh_keyed_legacy_diagnostic_v1.json"
bridge="$method_root/scripts/auh_e00_three_vessel_clean_diag_r2_bridge.sh"
validator="$method_root/validate_e00_three_vessel_fresh_keyed_two_phase_diagnostic_v2.py"
builder="$method_root/tools/build_e00_three_vessel_clean_diag_r2_package_v2.py"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
phase_a_stopped="$E00_R2_OUTPUT_ROOT/E00_R2_PHASE_A_STOPPED_REVIEW_REQUIRED.json"

for command_name in squeue srun sha256sum jq; do command -v "$command_name" >/dev/null; done
for file in "$manifest" "$review_marker" "$protocol" "$base_spec" "$bridge" "$validator" "$builder" "$E00_R2_PHASE_BC_AUTHORIZATION" "$phase_a_stopped"; do
  test -f "$file"; test ! -L "$file"
done
"$python_bin" -B "$builder" verify --package-root "$package_root" --manifest "$manifest" >/dev/null
protocol_result="$("$python_bin" -B "$validator" protocol --protocol "$protocol")"
manifest_sha="$(sha256sum -- "$manifest")"; manifest_sha="${manifest_sha%% *}"
protocol_sha="$(sha256sum -- "$protocol")"; protocol_sha="${protocol_sha%% *}"
protocol_canonical="$(printf '%s' "$protocol_result" | jq -er '.canonical_sha256')"
phase_a_stopped_sha="$(sha256sum -- "$phase_a_stopped")"; phase_a_stopped_sha="${phase_a_stopped_sha%% *}"

label_a="$(jq -er --arg role "$role_a" '.arms[] | select(.arm_role == $role) | .label' "$base_spec")"
label_b="$(jq -er --arg role "$role_b" '.arms[] | select(.arm_role == $role) | .label' "$base_spec")"
label_c="$(jq -er --arg role "$role_c" '.arms[] | select(.arm_role == $role) | .label' "$base_spec")"
video_a="$E00_R2_OUTPUT_ROOT/$label_a.mp4"
audit_a="$video_a.e00-r2-arm-audit.json"
video_b="$E00_R2_OUTPUT_ROOT/$label_b.mp4"
audit_b="$video_b.e00-r2-arm-audit.json"
video_c="$E00_R2_OUTPUT_ROOT/$label_c.mp4"
audit_c="$video_c.e00-r2-arm-audit.json"
for path in "$video_a" "$audit_a"; do test -f "$path"; test ! -L "$path"; done
for path in "$video_b" "$audit_b" "$video_c" "$audit_c"; do test ! -e "$path"; test ! -L "$path"; done
video_a_sha="$(sha256sum -- "$video_a")"; video_a_sha="${video_a_sha%% *}"
audit_a_sha="$(sha256sum -- "$audit_a")"; audit_a_sha="${audit_a_sha%% *}"

jq -e \
  --arg manifest "$manifest_sha" --arg protocol_sha "$protocol_sha" \
  --arg protocol_canonical "$protocol_canonical" --arg marker_sha "$phase_a_stopped_sha" \
  --arg audit_a_sha "$audit_a_sha" --arg video_a_sha "$video_a_sha" \
  --arg job "$job" --arg node "$node" --arg role_b "$role_b" --arg role_c "$role_c" '
  (keys | sort) == ([
    "authorized_arm_order","authorized_by","authorized_phase","c_requires_fresh_ab_bit_exact_gate",
    "compute_node","execution_authorized","package_manifest_sha256","parent_job_id",
    "phase_a_arm_audit_sha256","phase_a_mp4_sha256","phase_a_stopped_marker_sha256",
    "protocol_canonical_sha256","protocol_file_sha256","schema_version",
    "sp4_observer_released_node292","stop_without_c_on_gate_failure"
  ] | sort) and
  .schema_version == "bernini-e00-clean-diagnostic-r2-phase-bc-execution-authorization-v2" and
  .execution_authorized == true and
  .authorized_phase == "B_THEN_AB_BIT_EXACT_GATE_THEN_C" and
  .package_manifest_sha256 == $manifest and
  .protocol_file_sha256 == $protocol_sha and
  .protocol_canonical_sha256 == $protocol_canonical and
  .phase_a_stopped_marker_sha256 == $marker_sha and
  .phase_a_arm_audit_sha256 == $audit_a_sha and .phase_a_mp4_sha256 == $video_a_sha and
  .parent_job_id == $job and .compute_node == $node and
  .authorized_arm_order == [$role_b,$role_c] and
  .c_requires_fresh_ab_bit_exact_gate == true and
  .stop_without_c_on_gate_failure == true and
  .sp4_observer_released_node292 == true and
  (.authorized_by | type == "string" and length > 0)
' "$E00_R2_PHASE_BC_AUTHORIZATION" >/dev/null

jq -e \
  --arg manifest "$manifest_sha" --arg protocol_sha "$protocol_sha" \
  --arg protocol_canonical "$protocol_canonical" --arg job "$job" --arg node "$node" \
  --arg role_a "$role_a" --arg video_a "$video_a" --arg video_a_sha "$video_a_sha" \
  --arg audit_a "$audit_a" --arg audit_a_sha "$audit_a_sha" '
  .schema_version == "bernini-e00-clean-diagnostic-r2-phase-a-stopped-marker-v2" and
  .state == "A_STOPPED_REVIEW_REQUIRED" and .complete == true and
  .training_performed == false and .optimization_steps == 0 and
  .parent_job_id == $job and .compute_node == $node and
  .package_manifest_sha256 == $manifest and
  .protocol_file_sha256 == $protocol_sha and
  .protocol_canonical_sha256 == $protocol_canonical and
  .observed_arm_order == [$role_a] and
  .phase_a_video == {path:$video_a,sha256:$video_a_sha} and
  .phase_a_arm_audit == {path:$audit_a,sha256:$audit_a_sha} and
  .must_stop_after_a == true and .phase_bc_execution_authorized == false and
  .next_required_state == "EXTERNAL_BC_AUTHORIZATION_BOUND_TO_THIS_MARKER_AND_A_BYTES"
' "$phase_a_stopped" >/dev/null

state="$(squeue -h -j "$job" -w "$node" -o '%T' | head -n 1)"
test "$state" = RUNNING
lock="$E00_R2_OUTPUT_ROOT/.phase-bc-exclusive-lock"
mkdir "$lock"
cleanup_lock() { rmdir "$lock" 2>/dev/null || true; }

authorization_sha="$(sha256sum -- "$E00_R2_PHASE_BC_AUTHORIZATION")"; authorization_sha="${authorization_sha%% *}"
started="$E00_R2_OUTPUT_ROOT/e00-r2-phase-bc.started.json"
failed="$E00_R2_OUTPUT_ROOT/e00-r2-phase-bc.failed.json"
completed="$E00_R2_OUTPUT_ROOT/E00_R2_ABC_DIAGNOSTIC_COMPLETE.json"
ab_gate="$E00_R2_OUTPUT_ROOT/e00-r2-ab-bit-exact.gate.json"
final_audit="$E00_R2_OUTPUT_ROOT/e00-r2-abc-final.audit.json"
for path in "$started" "$failed" "$completed" "$ab_gate" "$final_audit"; do test ! -e "$path"; test ! -L "$path"; done
jq -n -S \
  --arg job "$job" --arg node "$node" --arg package "$package_root" \
  --arg manifest "$manifest" --arg manifest_sha "$manifest_sha" \
  --arg auth "$E00_R2_PHASE_BC_AUTHORIZATION" --arg auth_sha "$authorization_sha" \
  --arg marker "$phase_a_stopped" --arg marker_sha "$phase_a_stopped_sha" \
  --arg role_b "$role_b" --arg role_c "$role_c" '
  {
    schema_version:"bernini-e00-clean-diagnostic-r2-phase-bc-launch-receipt-v2",
    state:"b_started_c_forbidden", complete:false,
    training_performed:false, optimization_steps:0,
    parent_job_id:$job, compute_node:$node, package_root:$package,
    package_manifest:{path:$manifest,sha256:$manifest_sha},
    authorization:{path:$auth,sha256:$auth_sha},
    phase_a_stopped_marker:{path:$marker,sha256:$marker_sha},
    authorized_arm_order:[$role_b,$role_c],
    c_requires_fresh_ab_bit_exact_gate:true
  }
' > "$started"

finalize_phase_bc() {
  status="$?"
  trap - EXIT
  cleanup_lock
  if [ "$status" -ne 0 ]; then
    jq -S --argjson exit_code "$status" \
      '.state="bc_failed_c_not_admitted_or_incomplete" | .complete=false | .exit_code=$exit_code' \
      "$started" > "$failed"
  fi
  exit "$status"
}
trap finalize_phase_bc EXIT

run_arm() {
  local role="$1"
  srun --jobid="$job" --exclusive --nodes=1 --ntasks=1 --cpus-per-task=64 \
    --gres=gpu:4 --mem=0 --nodelist="$node" \
    env E00_R2_ARM_ROLE="$role" E00_R2_OUTPUT_ROOT="$E00_R2_OUTPUT_ROOT" \
      E00_R2_PACKAGE_MANIFEST="$manifest" \
      EXPECTED_COMPUTE_NODE="$node" EXPECTED_PARENT_JOB="$job" \
      bash "$bridge"
}

run_arm "$role_b"

# This is the only admission edge to C.  It re-reads both arm audits and both
# MP4 files and recomputes latent, MP4, fixed-RNG, noise, schedule, and frozen
# model equality immediately after B.  A failure exits before the C srun.
"$python_bin" -B "$validator" ab-gate \
  --a-audit "$audit_a" --b-audit "$audit_b" \
  --a-video "$video_a" --b-video "$video_b" \
  --gate-output "$ab_gate"
jq -e '
  .schema_version == "bernini-e00-clean-diagnostic-r2-ab-bit-exact-gate-v2" and
  .complete == true and .c_execution_gate_passed == true and
  .only_admitted_next_arm == "old_pureqk_temporal_routeon" and
  .training_performed == false
' "$ab_gate" >/dev/null

run_arm "$role_c"

"$python_bin" -B "$validator" abc-final \
  --a-audit "$audit_a" --b-audit "$audit_b" --c-audit "$audit_c" \
  --ab-gate "$ab_gate" \
  --a-video "$video_a" --b-video "$video_b" --c-video "$video_c" \
  --audit-output "$final_audit"
final_audit_sha="$(sha256sum -- "$final_audit")"; final_audit_sha="${final_audit_sha%% *}"
ab_gate_sha="$(sha256sum -- "$ab_gate")"; ab_gate_sha="${ab_gate_sha%% *}"
jq -S --arg final "$final_audit" --arg final_sha "$final_audit_sha" \
  --arg gate "$ab_gate" --arg gate_sha "$ab_gate_sha" '
  .state="abc_completed" | .complete=true |
  .ab_bit_exact_gate={path:$gate,sha256:$gate_sha} |
  .final_audit={path:$final,sha256:$final_sha}
' "$started" > "$completed"
cleanup_lock
trap - EXIT
printf 'E00_R2_ABC_DIAGNOSTIC_COMPLETE %s\n' "$final_audit"
