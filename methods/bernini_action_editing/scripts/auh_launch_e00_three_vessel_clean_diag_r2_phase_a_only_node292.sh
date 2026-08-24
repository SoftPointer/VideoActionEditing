#!/usr/bin/env bash
set -euo pipefail

# Phase A launcher.  Its executable surface contains exactly one arm and must
# terminate in A_STOPPED_REVIEW_REQUIRED.  It cannot run B or C.
if [ "$#" -ne 0 ]; then echo "usage: $0" >&2; exit 2; fi
case "$(hostname -s)" in auh-1b-cpu-login-002|auh7-ib|auh) ;; *) echo "run on AUH login host" >&2; exit 2 ;; esac
: "${E00_R2_PACKAGE_ROOT:?missing E00_R2_PACKAGE_ROOT}"
: "${E00_R2_PHASE_A_AUTHORIZATION:?missing E00_R2_PHASE_A_AUTHORIZATION}"
: "${E00_R2_OUTPUT_ROOT:?missing E00_R2_OUTPUT_ROOT}"
if [ "${ALLOW_E00_R2_PHASE_A_ONLY:-}" != I_REVIEWED_R2_AND_AUTHORIZE_A_ONLY_THEN_STOP ]; then
  echo "phase A remains disabled pending its independent package authorization" >&2
  exit 3
fi

job=143808
node=auh7-1b-gpu-292
role_a=pure_noobserver_output_routeoff
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

for command_name in squeue srun sha256sum jq; do command -v "$command_name" >/dev/null; done
for file in "$manifest" "$review_marker" "$protocol" "$base_spec" "$bridge" "$validator" "$builder" "$E00_R2_PHASE_A_AUTHORIZATION"; do
  test -f "$file"; test ! -L "$file"
done
"$python_bin" -B "$builder" verify --package-root "$package_root" --manifest "$manifest" >/dev/null
protocol_result="$("$python_bin" -B "$validator" protocol --protocol "$protocol")"
manifest_sha="$(sha256sum -- "$manifest")"; manifest_sha="${manifest_sha%% *}"
protocol_sha="$(sha256sum -- "$protocol")"; protocol_sha="${protocol_sha%% *}"
protocol_canonical="$(printf '%s' "$protocol_result" | jq -er '.canonical_sha256')"
jq -e \
  --arg manifest "$manifest_sha" --arg protocol_sha "$protocol_sha" \
  --arg protocol_canonical "$protocol_canonical" --arg job "$job" --arg node "$node" \
  --arg role "$role_a" '
  (keys | sort) == ([
    "authorized_by","authorized_phase","bc_execution_authorized","compute_node",
    "execution_authorized","must_stop_after_a","only_authorized_arm",
    "package_manifest_sha256","parent_job_id","protocol_canonical_sha256",
    "protocol_file_sha256","schema_version","sp4_observer_released_node292"
  ] | sort) and
  .schema_version == "bernini-e00-clean-diagnostic-r2-phase-a-execution-authorization-v2" and
  .execution_authorized == true and .authorized_phase == "A_ONLY_THEN_STOP" and
  .package_manifest_sha256 == $manifest and
  .protocol_file_sha256 == $protocol_sha and
  .protocol_canonical_sha256 == $protocol_canonical and
  .parent_job_id == $job and .compute_node == $node and
  .only_authorized_arm == $role and .must_stop_after_a == true and
  .bc_execution_authorized == false and
  .sp4_observer_released_node292 == true and
  (.authorized_by | type == "string" and length > 0)
' "$E00_R2_PHASE_A_AUTHORIZATION" >/dev/null
state="$(squeue -h -j "$job" -w "$node" -o '%T' | head -n 1)"
test "$state" = RUNNING
test ! -e "$E00_R2_OUTPUT_ROOT"; test ! -L "$E00_R2_OUTPUT_ROOT"
mkdir -p "$E00_R2_OUTPUT_ROOT"
lock="$E00_R2_OUTPUT_ROOT/.phase-a-exclusive-lock"
mkdir "$lock"
cleanup_lock() { rmdir "$lock" 2>/dev/null || true; }

authorization_sha="$(sha256sum -- "$E00_R2_PHASE_A_AUTHORIZATION")"; authorization_sha="${authorization_sha%% *}"
started="$E00_R2_OUTPUT_ROOT/e00-r2-phase-a.started.json"
failed="$E00_R2_OUTPUT_ROOT/e00-r2-phase-a.failed.json"
stopped="$E00_R2_OUTPUT_ROOT/E00_R2_PHASE_A_STOPPED_REVIEW_REQUIRED.json"
jq -n -S \
  --arg job "$job" --arg node "$node" --arg package "$package_root" \
  --arg manifest "$manifest" --arg manifest_sha "$manifest_sha" \
  --arg auth "$E00_R2_PHASE_A_AUTHORIZATION" --arg auth_sha "$authorization_sha" \
  --arg output "$E00_R2_OUTPUT_ROOT" --arg role "$role_a" '
  {
    schema_version:"bernini-e00-clean-diagnostic-r2-phase-a-launch-receipt-v2",
    state:"a_started", complete:false, training_performed:false, optimization_steps:0,
    parent_job_id:$job, compute_node:$node, package_root:$package,
    package_manifest:{path:$manifest,sha256:$manifest_sha},
    authorization:{path:$auth,sha256:$auth_sha}, output_root:$output,
    authorized_arm_order:[$role], must_stop_after_a:true, bc_execution_authorized:false
  }
' > "$started"

finalize_phase_a() {
  status="$?"
  trap - EXIT
  cleanup_lock
  if [ "$status" -ne 0 ]; then
    jq -S --argjson exit_code "$status" \
      '.state="a_failed" | .complete=false | .exit_code=$exit_code' \
      "$started" > "$failed"
  fi
  exit "$status"
}
trap finalize_phase_a EXIT

# The only srun in this launcher is arm A.
srun --jobid="$job" --exclusive --nodes=1 --ntasks=1 --cpus-per-task=64 \
  --gres=gpu:4 --mem=0 --nodelist="$node" \
  env E00_R2_ARM_ROLE="$role_a" E00_R2_OUTPUT_ROOT="$E00_R2_OUTPUT_ROOT" \
    E00_R2_PACKAGE_MANIFEST="$manifest" \
    EXPECTED_COMPUTE_NODE="$node" EXPECTED_PARENT_JOB="$job" \
    bash "$bridge"

label_a="$(jq -er --arg role "$role_a" '.arms[] | select(.arm_role == $role) | .label' "$base_spec")"
video_a="$E00_R2_OUTPUT_ROOT/$label_a.mp4"
audit_a="$video_a.e00-r2-arm-audit.json"
for path in "$video_a" "$audit_a"; do test -f "$path"; test ! -L "$path"; done
video_a_sha="$(sha256sum -- "$video_a")"; video_a_sha="${video_a_sha%% *}"
audit_a_sha="$(sha256sum -- "$audit_a")"; audit_a_sha="${audit_a_sha%% *}"

# Emit the mandatory hard stop.  No B/C authorization is inferred from A.
jq -n -S \
  --arg job "$job" --arg node "$node" --arg manifest_sha "$manifest_sha" \
  --arg protocol_sha "$protocol_sha" --arg protocol_canonical "$protocol_canonical" \
  --arg auth_sha "$authorization_sha" --arg role "$role_a" \
  --arg video "$video_a" --arg video_sha "$video_a_sha" \
  --arg audit "$audit_a" --arg audit_sha "$audit_a_sha" '
  {
    schema_version:"bernini-e00-clean-diagnostic-r2-phase-a-stopped-marker-v2",
    state:"A_STOPPED_REVIEW_REQUIRED", complete:true,
    training_performed:false, optimization_steps:0,
    parent_job_id:$job, compute_node:$node,
    package_manifest_sha256:$manifest_sha,
    protocol_file_sha256:$protocol_sha,
    protocol_canonical_sha256:$protocol_canonical,
    phase_a_authorization_sha256:$auth_sha,
    observed_arm_order:[$role],
    phase_a_video:{path:$video,sha256:$video_sha},
    phase_a_arm_audit:{path:$audit,sha256:$audit_sha},
    must_stop_after_a:true,
    phase_bc_execution_authorized:false,
    next_required_state:"EXTERNAL_BC_AUTHORIZATION_BOUND_TO_THIS_MARKER_AND_A_BYTES"
  }
' > "$stopped"
cleanup_lock
trap - EXIT
printf 'E00_R2_PHASE_A_STOPPED_REVIEW_REQUIRED %s\n' "$stopped"
