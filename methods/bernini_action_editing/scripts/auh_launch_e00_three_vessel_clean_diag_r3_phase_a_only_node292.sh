#!/usr/bin/env bash
set -euo pipefail

# R3 phase A launcher: one arm, one capability, mandatory hard stop.
if [ "$#" -ne 0 ]; then echo "usage: $0" >&2; exit 2; fi
case "$(hostname -s)" in auh-1b-cpu-login-002|auh7-ib|auh) ;; *) echo "run on AUH login host" >&2; exit 2 ;; esac
: "${E00_R3_PACKAGE_ROOT:?missing E00_R3_PACKAGE_ROOT}"
: "${E00_R3_PHASE_A_AUTHORIZATION:?missing E00_R3_PHASE_A_AUTHORIZATION}"
: "${E00_R3_PHASE_A_TOKEN:?missing E00_R3_PHASE_A_TOKEN}"
: "${E00_R3_OUTPUT_ROOT:?missing E00_R3_OUTPUT_ROOT}"
if [ "${ALLOW_E00_R3_PHASE_A_ONLY:-}" != I_REVIEWED_R3_AND_AUTHORIZE_A_ONLY_THEN_STOP ]; then
  echo "R3 phase A remains disabled" >&2; exit 3
fi

job=143808
node=auh7-1b-gpu-292
role_a=pure_noobserver_output_routeoff
package_root="$(cd -- "$E00_R3_PACKAGE_ROOT" && pwd -P)"
method_root="$package_root/methods/bernini_action_editing"
manifest="$package_root/e00-clean-diagnostic-r3-package.manifest.json"
review_marker="$package_root/R3_EXECUTION_REVIEW_REQUIRED.json"
protocol="$method_root/assets/e00_three_vessel_clean_diag_r3_protocol_20260821.json"
base_spec="$method_root/assets/e00_three_vessel_fresh_keyed_legacy_diagnostic_v1.json"
bridge="$method_root/scripts/auh_e00_three_vessel_clean_diag_r3_bridge.sh"
validator="$method_root/validate_e00_three_vessel_clean_diag_r3.py"
builder="$method_root/tools/build_e00_three_vessel_clean_diag_r3_package.py"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

for command_name in squeue srun sha256sum jq; do command -v "$command_name" >/dev/null; done
for file in "$manifest" "$review_marker" "$protocol" "$base_spec" "$bridge" "$validator" "$builder" "$E00_R3_PHASE_A_AUTHORIZATION"; do
  test -f "$file"; test ! -L "$file"
done
"$python_bin" -B "$builder" verify --package-root "$package_root" --manifest "$manifest" >/dev/null
"$python_bin" -B "$validator" protocol --protocol "$protocol" >/dev/null
token_sha="$(printf '%s' "$E00_R3_PHASE_A_TOKEN" | sha256sum)"; token_sha="${token_sha%% *}"
"$python_bin" -B "$validator" bridge-capability --protocol "$protocol" \
  --phase A --arm-role "$role_a" --package-manifest "$manifest" \
  --authorization "$E00_R3_PHASE_A_AUTHORIZATION" \
  --capability-token-sha256 "$token_sha" >/dev/null
state="$(squeue -h -j "$job" -w "$node" -o '%T' | head -n 1)"
test "$state" = RUNNING
test ! -e "$E00_R3_OUTPUT_ROOT"; test ! -L "$E00_R3_OUTPUT_ROOT"
mkdir -p "$E00_R3_OUTPUT_ROOT"
lock="$E00_R3_OUTPUT_ROOT/.r3-phase-a-lock"
mkdir "$lock"
cleanup_lock() { rmdir "$lock" 2>/dev/null || true; }

manifest_sha="$(sha256sum -- "$manifest")"; manifest_sha="${manifest_sha%% *}"
protocol_sha="$(sha256sum -- "$protocol")"; protocol_sha="${protocol_sha%% *}"
protocol_canonical="$("$python_bin" -B "$validator" protocol --protocol "$protocol" | jq -er '.canonical_sha256')"
auth_sha="$(sha256sum -- "$E00_R3_PHASE_A_AUTHORIZATION")"; auth_sha="${auth_sha%% *}"
started="$E00_R3_OUTPUT_ROOT/e00-r3-phase-a.started.json"
failed="$E00_R3_OUTPUT_ROOT/e00-r3-phase-a.failed.json"
stopped="$E00_R3_OUTPUT_ROOT/E00_R3_PHASE_A_STOPPED_REVIEW_REQUIRED.json"
jq -n -S --arg job "$job" --arg node "$node" --arg manifest_sha "$manifest_sha" \
  --arg auth "$E00_R3_PHASE_A_AUTHORIZATION" --arg auth_sha "$auth_sha" \
  --arg token_sha "$token_sha" --arg role "$role_a" '
  {
    schema_version:"bernini-e00-clean-diagnostic-r3-phase-a-launch-v3",
    revision_tag:"E00_DFIX2_CLEAN_DIAG_R3_CAPABILITY_FINALCLOSURE_20260821",
    state:"a_started",complete:false,training_performed:false,optimization_steps:0,
    parent_job_id:$job,compute_node:$node,package_manifest_sha256:$manifest_sha,
    authorization:{path:$auth,sha256:$auth_sha},capability_token_sha256:$token_sha,
    authorized_arm_order:[$role],must_stop_after_a:true,bc_execution_authorized:false
  }
' > "$started"
finalize() {
  status="$?"; trap - EXIT; cleanup_lock
  if [ "$status" -ne 0 ]; then jq -S --argjson code "$status" '.state="a_failed"|.exit_code=$code' "$started" > "$failed"; fi
  exit "$status"
}
trap finalize EXIT

srun --jobid="$job" --exclusive --nodes=1 --ntasks=1 --cpus-per-task=64 \
  --gres=gpu:4 --mem=0 --nodelist="$node" \
  env E00_R3_ARM_ROLE="$role_a" E00_R3_PHASE=A \
    E00_R3_OUTPUT_ROOT="$E00_R3_OUTPUT_ROOT" E00_R3_PACKAGE_MANIFEST="$manifest" \
    E00_R3_AUTHORIZATION="$E00_R3_PHASE_A_AUTHORIZATION" \
    E00_R3_CAPABILITY_TOKEN="$E00_R3_PHASE_A_TOKEN" bash "$bridge"

label_a="$(jq -er --arg role "$role_a" '.arms[]|select(.arm_role==$role)|.label' "$base_spec")"
video_a="$E00_R3_OUTPUT_ROOT/$label_a.mp4"
audit_a="$video_a.e00-r3-arm-audit.json"
for path in "$video_a" "$audit_a"; do test -f "$path"; test ! -L "$path"; done
video_sha="$(sha256sum -- "$video_a")"; video_sha="${video_sha%% *}"
audit_sha="$(sha256sum -- "$audit_a")"; audit_sha="${audit_sha%% *}"
jq -n -S --arg job "$job" --arg node "$node" --arg manifest_sha "$manifest_sha" \
  --arg protocol_sha "$protocol_sha" --arg protocol_canonical "$protocol_canonical" \
  --arg auth "$E00_R3_PHASE_A_AUTHORIZATION" --arg auth_sha "$auth_sha" \
  --arg token_sha "$token_sha" --arg role "$role_a" \
  --arg video "$video_a" --arg video_sha "$video_sha" \
  --arg audit "$audit_a" --arg audit_sha "$audit_sha" '
  {
    schema_version:"bernini-e00-clean-diagnostic-r3-phase-a-stopped-marker-v3",
    revision_tag:"E00_DFIX2_CLEAN_DIAG_R3_CAPABILITY_FINALCLOSURE_20260821",
    state:"A_STOPPED_REVIEW_REQUIRED",complete:true,training_performed:false,optimization_steps:0,
    parent_job_id:$job,compute_node:$node,package_manifest_sha256:$manifest_sha,
    protocol_file_sha256:$protocol_sha,protocol_canonical_sha256:$protocol_canonical,
    phase_a_authorization:{path:$auth,sha256:$auth_sha},
    phase_a_capability_token_sha256:$token_sha,observed_arm_order:[$role],
    phase_a_video:{path:$video,sha256:$video_sha},phase_a_arm_audit:{path:$audit,sha256:$audit_sha},
    must_stop_after_a:true,phase_bc_execution_authorized:false,
    next_required_state:"EXTERNAL_BC_AUTHORIZATION_AND_TOKEN_BOUND_TO_CURRENT_A_BYTES"
  }
' > "$stopped"
cleanup_lock; trap - EXIT
printf 'E00_R3_PHASE_A_STOPPED_REVIEW_REQUIRED %s\n' "$stopped"
