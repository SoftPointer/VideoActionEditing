#!/usr/bin/env bash
set -euo pipefail

# Frozen-base, zero-update max-strength clean-noised route probe.  This launcher is
# independent of every training watcher and never accepts a checkpoint path.
case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh) ;;
  *) echo "v15a launcher must run on the AUH login host" >&2; exit 2 ;;
esac
if [ "$#" -ne 0 ]; then echo "usage: $0" >&2; exit 2; fi
for command_name in squeue srun sha256sum jq ffprobe; do
  command -v "$command_name" >/dev/null
done

job=143808
node=auh7-1b-gpu-292
state="$(squeue -h -j "$job" -w "$node" -o '%T' | head -n 1)"
test "$state" = RUNNING

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
source_tree="$stage/source-online-anchor-targetowned-qk-decode-v14r3-gradgeom-dfix2"
dev="$source_tree/methods/bernini_action_editing"
bridge="$dev/scripts/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh"
deployment_validator="$dev/validate_v14r2_deployment_marker.py"
marker="$release/DEPLOYMENT_TESTS_PASS_decode_targetcoord_v14r3_gradgeom_dfix2.json"
training_marker="$release/DEPLOYMENT_TESTS_PASS_targetcoord_v14r3_gradgeom.json"
archive="$release/online-anchor-targetowned-qk-decode-v14r3-gradgeom-dfix2.tar"
revision="$release/online-anchor-targetowned-qk-decode-v14r3-gradgeom-dfix2.revision"
content="$release/online-anchor-targetowned-qk-decode-v14r3-gradgeom-dfix2.content.json"
authoring_manifest="$stage/interaction_complex8_multianchor_authoring_v2.json"
source_authoring_manifest="$dev/assets/interaction_complex8_multianchor_authoring_v2.json"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
script_dir="$(cd -- "$(dirname -- "$0")" && pwd -P)"
validator_name=validate_v15a_zero_update_upper_bound_sidecar_v1.py
validator_candidates=("$script_dir/../$validator_name" "$script_dir/$validator_name")
sidecar_validator=
validator_matches=0
for candidate in "${validator_candidates[@]}"; do
  if [ -f "$candidate" ] && [ ! -L "$candidate" ]; then
    sidecar_validator="$(cd -- "$(dirname -- "$candidate")" && pwd -P)/$(basename -- "$candidate")"
    validator_matches=$((validator_matches + 1))
  fi
done
test "$validator_matches" = 1

test -d "$source_tree"
test ! -L "$source_tree"
for file in "$bridge" "$deployment_validator" "$marker" "$training_marker" \
  "$archive" "$revision" "$content" "$authoring_manifest" \
  "$source_authoring_manifest" "$sidecar_validator"; do
  test -f "$file"
  test ! -L "$file"
done
test -x "$python_bin"

verify_sha() {
  local file="$1" expected="$2" actual
  actual="$(sha256sum -- "$file")"; actual="${actual%% *}"
  test "$actual" = "$expected"
}
verify_video_geometry() {
  local video="$1" probe
  probe="$(ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=nb_read_frames,avg_frame_rate -of json -- "$video")"
  jq -e '
    (.streams | length) == 1 and
    .streams[0].nb_read_frames == "81" and
    .streams[0].avg_frame_rate == "25/1"
  ' <<<"$probe" >/dev/null
}
wait_sha_file_visible() {
  local file="$1" expected_sha="$2" label="$3" attempt actual
  test "${#expected_sha}" = 64
  case "$expected_sha" in *[!0-9a-f]*) return 2 ;; esac
  for ((attempt = 1; attempt <= 60; attempt++)); do
    if [ -f "$file" ] && [ ! -L "$file" ]; then
      actual="$(sha256sum -- "$file" 2>/dev/null || true)"
      actual="${actual%% *}"
      if [ "$actual" = "$expected_sha" ]; then return 0; fi
    fi
    sleep 1
  done
  echo "Lustre SHA visibility timeout for $label: $file" >&2
  return 1
}
wait_complete_json_visible() {
  local file="$1" label="$2" attempt
  for ((attempt = 1; attempt <= 60; attempt++)); do
    if [ -f "$file" ] && [ ! -L "$file" ] \
      && jq -e '.complete == true' "$file" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Lustre complete-JSON visibility timeout for $label: $file" >&2
  return 1
}
wait_sha_complete_json_visible() {
  local file="$1" expected_sha="$2" label="$3" attempt actual
  test "${#expected_sha}" = 64
  case "$expected_sha" in *[!0-9a-f]*) return 2 ;; esac
  for ((attempt = 1; attempt <= 60; attempt++)); do
    if [ -f "$file" ] && [ ! -L "$file" ]; then
      actual="$(sha256sum -- "$file" 2>/dev/null || true)"
      actual="${actual%% *}"
      if [ "$actual" = "$expected_sha" ] \
        && jq -e '.complete == true' "$file" >/dev/null 2>&1; then
        return 0
      fi
    fi
    sleep 1
  done
  echo "Lustre SHA/complete visibility timeout for $label: $file" >&2
  return 1
}
verify_sha "$archive" 88c47356a83368ccdb0649718c8c06fa8c7baf368e7c97df8a48d9b93ab55bd9
verify_sha "$revision" be15562d0e75fa953a9464447d187265ef5a8e36b5541a1f73dc13b0f41a98a3
verify_sha "$content" ebb8d86545c9b9f476d3ee3b1855ccd310cad321c8fd2f93a8540e807e13a1af
verify_sha "$marker" 03bad1ae839a1135a1cca08990cd2a68788f4b44485abf90271453cbf3f9b969
verify_sha "$dev/anchor_sga_anc_controller.py" 1427a4908e0a4239e95a353d3406c41cb77fdb7f0be81727126a2cfd23f1f3ad
verify_sha "$dev/anchor_qk_transport.py" 37941e30853b16fa242a7c91940620069f87a1a975d2ecf610f3cde800557a99
verify_sha "$dev/infer_anchor_sga_anc_event_v1.py" dd3558a4c38c5541ba6b7ad455ac599f43eb48b1b56f207a07776c9e1819145f
verify_sha "$dev/infer_anchor_sga_anc_trained_editor_decode_v1.py" 4ed2f22df876613ecfc720a662a48f8e028eb89fe9778e491bc962a4f8f68ab1
verify_sha "$bridge" 0365aacb88d976fcdc1f9bf169384f5d336bd4abe2f3c899ab6bdd502a580034
verify_sha "$deployment_validator" 443e8a4966485edfa5abd375e921d3bc1e314bb16d2299659a537b271f7530ba
verify_sha "$authoring_manifest" 767aa8e0502f247c3ab576db4c40a132344295e54aef4170e728bc3ff71cafc5
verify_sha "$source_authoring_manifest" 767aa8e0502f247c3ab576db4c40a132344295e54aef4170e728bc3ff71cafc5
verify_sha "$sidecar_validator" e562959b1737d65bf954038119db24651940d69acdf151a519c4dad9e5bc2736

"$python_bin" -B "$deployment_validator" \
  --marker "$marker" --role decode --source-tree "$source_tree" \
  --archive "$archive" --revision "$revision" --content-manifest "$content" \
  --min-test-count 144 --training-marker "$training_marker" \
  --shared-core methods/bernini_action_editing/anchor_qk_transport.py \
  --shared-core methods/bernini_action_editing/anchor_sga_anc_controller.py \
  --shared-core methods/bernini_action_editing/infer_anchor_sga_anc_event_v1.py \
  --required-file methods/bernini_action_editing/anchor_qk_transport.py \
  --required-file methods/bernini_action_editing/anchor_sga_anc_controller.py \
  --required-file methods/bernini_action_editing/infer_anchor_sga_anc_event_v1.py \
  --required-file methods/bernini_action_editing/infer_anchor_sga_anc_trained_editor_decode_v1.py \
  --required-file methods/bernini_action_editing/assets/interaction_complex8_multianchor_authoring_v2.json \
  --required-file methods/bernini_action_editing/scripts/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh

source_video=/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/2f183dbf9e7a4d2e/source.mp4
anchor_dir="$stage/interaction_complex8_multianchor_v2_r1/e00_pour-liquid-into-cup"
anchor_video="$anchor_dir/v0/t2v.mp4"
anchor_gaussian="$anchor_dir/v0/t2v.official-initial-gaussian.safetensors"
verify_sha "$source_video" 888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de
verify_sha "$anchor_video" e485b2963d5ba191d8fc98158b7ab100bd1bd7584735221077c014ff0c3a19aa
verify_sha "$anchor_gaussian" e4828f4594caf65b45a937a271847dcf621a75e4f6862269a28216759ff087b7
verify_video_geometry "$source_video"
verify_video_geometry "$anchor_video"

tag=v15a_zero_update_dynamic_static_e00_maxstrength_routeprobe_r3_20260820
output_root="$release/dynaedit_maxstrength_routeprobe_v15a_r3/$tag"
test ! -e "$output_root"
test ! -L "$output_root"

source_caption="An East Asian woman in a black floral blouse pours amber tea from a white bowl into a clear glass pitcher on a wooden tea table while a small white cup sits nearby."
target_caption="The same woman in the same black floral blouse at the same table holds the same original glass pitcher and pours a continuous amber stream from its spout into the same original white cup, fills that cup, then returns the same pitcher upright. The white bowl remains unchanged and is not used as the pouring vessel."
noop_caption="The woman keeps the pitcher and cup in their initial state and does not lift or tilt the pitcher, pour a stream into the cup, fill the cup, or return a completed pour."
anchor_caption="An exactly 81-frame realistic continuous locked-camera video at 25 fps shows a young East Asian woman in a dark green cardigan at a warm wooden tea table, with a clear glass pitcher of amber tea and a small white ceramic cup. Starting with one hand on the pitcher handle and the other near the cup, the person lifts the pitcher, aligns its spout over the cup, tilts it smoothly, and pours a clearly visible continuous stream into the cup. The liquid level in the cup visibly rises. The person stops the stream, returns the pitcher upright, and holds the filled cup and upright pitcher in a stable terminal state. Keep exactly one person, one pitcher and one cup. Preserve their identity, material, colors, table, lighting, framing and camera throughout. No cuts, teleportation, morphing, replacement, duplication or disappearing contact."
action="$(jq -r '.events[0].action' "$authoring_manifest")"
constraints="$(jq -r '.events[0].constraints' "$authoring_manifest")"
setup="$(jq -r '.events[0].variants[0].setup' "$authoring_manifest")"
instruction="Use the source video as the sole authority for identity, appearance, clothing, object instances, background, lighting, framing, camera and initial state. Frame 0 must retain the original source state; do not pre-apply the requested endpoint. Perform only this temporal edit: $action $constraints The edit must be one continuous 81-frame video at 25 fps and must not introduce appearance changes as a substitute for the requested action."
test "$anchor_caption" = "$setup $action $constraints"

mkdir -p "$output_root"
lock="$output_root/.launch.lock"
mkdir "$lock"
cleanup_lock() { rmdir "$lock" 2>/dev/null || true; }
trap cleanup_lock EXIT

routeoff_label=E00_V15AR3_FROZEN_ZEROUPDATE_DYNSTATIC_MATCHED_PLAIN_FROZEN_ROUTEOFF_K0_A100
temporal_label=E00_V15AR3_FROZEN_ZEROUPDATE_DYNSTATIC_TARGETOWNED_TEMPORAL_ROUTEON_K40_A100
activity_label=E00_V15AR3_FROZEN_ZEROUPDATE_DYNSTATIC_TARGETOWNED_ACTIVITY25_ROUTEON_K40_A100
launcher_path="$script_dir/$(basename -- "$0")"
test -f "$launcher_path"; test ! -L "$launcher_path"
launcher_sha="$(sha256sum -- "$launcher_path")"; launcher_sha="${launcher_sha%% *}"
validator_sha="$(sha256sum -- "$sidecar_validator")"; validator_sha="${validator_sha%% *}"
bridge_sha="$(sha256sum -- "$bridge")"; bridge_sha="${bridge_sha%% *}"
controller_sha="$(sha256sum -- "$dev/anchor_sga_anc_controller.py")"; controller_sha="${controller_sha%% *}"
qk_sha="$(sha256sum -- "$dev/anchor_qk_transport.py")"; qk_sha="${qk_sha%% *}"
infer_sha="$(sha256sum -- "$dev/infer_anchor_sga_anc_event_v1.py")"; infer_sha="${infer_sha%% *}"
decode_sha="$(sha256sum -- "$dev/infer_anchor_sga_anc_trained_editor_decode_v1.py")"; decode_sha="${decode_sha%% *}"
deployment_validator_sha="$(sha256sum -- "$deployment_validator")"; deployment_validator_sha="${deployment_validator_sha%% *}"
authoring_sha="$(sha256sum -- "$authoring_manifest")"; authoring_sha="${authoring_sha%% *}"
marker_sha="$(sha256sum -- "$marker")"; marker_sha="${marker_sha%% *}"
archive_sha="$(sha256sum -- "$archive")"; archive_sha="${archive_sha%% *}"
revision_sha="$(sha256sum -- "$revision")"; revision_sha="${revision_sha%% *}"
content_sha="$(sha256sum -- "$content")"; content_sha="${content_sha%% *}"
launch_receipt="$output_root/v15a_launch_receipt.json"
test ! -e "$launch_receipt"; test ! -L "$launch_receipt"
launch_receipt_compute_sha="$(srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --mem=1G --nodelist="$node" \
  env EXPECTED_COMPUTE_NODE="$node" EXPECTED_PARENT_JOB="$job" \
    EXPERIMENT_TAG="$tag" FIXED_OUTPUT_ROOT="$output_root" \
    LAUNCH_RECEIPT="$launch_receipt" LAUNCHER_PATH="$launcher_path" LAUNCHER_SHA="$launcher_sha" \
    VALIDATOR_PATH="$sidecar_validator" VALIDATOR_SHA="$validator_sha" \
    BRIDGE_PATH="$bridge" BRIDGE_SHA="$bridge_sha" CONTROLLER_SHA="$controller_sha" \
    QK_SHA="$qk_sha" INFER_SHA="$infer_sha" DECODE_SHA="$decode_sha" \
    DEPLOYMENT_VALIDATOR_PATH="$deployment_validator" DEPLOYMENT_VALIDATOR_SHA="$deployment_validator_sha" \
    AUTHORING_MANIFEST="$authoring_manifest" AUTHORING_SHA="$authoring_sha" \
    MARKER_PATH="$marker" MARKER_SHA="$marker_sha" ARCHIVE_PATH="$archive" ARCHIVE_SHA="$archive_sha" \
    REVISION_PATH="$revision" REVISION_SHA="$revision_sha" CONTENT_PATH="$content" CONTENT_SHA="$content_sha" \
    ROUTEOFF_LABEL="$routeoff_label" TEMPORAL_LABEL="$temporal_label" ACTIVITY_LABEL="$activity_label" \
    bash -c '
      set -euo pipefail
      test "$(hostname -s)" = "$EXPECTED_COMPUTE_NODE"
      test "$SLURM_JOB_ID" = "$EXPECTED_PARENT_JOB"
      receipt_parent="${LAUNCH_RECEIPT%/*}"
      for ((attempt = 1; attempt <= 60; attempt++)); do
        if [ -d "$receipt_parent" ] && [ ! -L "$receipt_parent" ]; then break; fi
        sleep 1
      done
      test -d "$receipt_parent"; test ! -L "$receipt_parent"
      test ! -e "$LAUNCH_RECEIPT"
      receipt_tmp="${LAUNCH_RECEIPT}.compute-step-${SLURM_STEP_ID:-unknown}.tmp"
      test ! -e "$receipt_tmp"
      jq -n -S \
        --arg job "$SLURM_JOB_ID" --arg node "$(hostname -s)" \
        --arg tag "$EXPERIMENT_TAG" --arg root "$FIXED_OUTPUT_ROOT" \
        --arg lp "$LAUNCHER_PATH" --arg ls "$LAUNCHER_SHA" \
        --arg vp "$VALIDATOR_PATH" --arg vs "$VALIDATOR_SHA" \
        --arg bp "$BRIDGE_PATH" --arg bs "$BRIDGE_SHA" \
        --arg cs "$CONTROLLER_SHA" --arg qs "$QK_SHA" \
        --arg is "$INFER_SHA" --arg ds "$DECODE_SHA" \
        --arg dvp "$DEPLOYMENT_VALIDATOR_PATH" --arg dvs "$DEPLOYMENT_VALIDATOR_SHA" \
        --arg ap "$AUTHORING_MANIFEST" --arg as "$AUTHORING_SHA" \
        --arg mp "$MARKER_PATH" --arg ms "$MARKER_SHA" \
        --arg xp "$ARCHIVE_PATH" --arg xs "$ARCHIVE_SHA" \
        --arg rp "$REVISION_PATH" --arg rs "$REVISION_SHA" \
        --arg cp "$CONTENT_PATH" --arg cs2 "$CONTENT_SHA" \
        --arg rl "$ROUTEOFF_LABEL" --arg tl "$TEMPORAL_LABEL" --arg al "$ACTIVITY_LABEL" '\''
        {
          schema_version:"bernini-v15a-r3-max-strength-clean-noised-route-probe-launch-v1",
          complete:true,parent_job_id:$job,compute_node:$node,
          experiment_tag:$tag,fixed_output_root:$root,
          controls:{
            launcher:{path:$lp,sha256:$ls},validator:{path:$vp,sha256:$vs},
            bridge:{path:$bp,sha256:$bs},controller_sha256:$cs,qk_transport_sha256:$qs,
            infer_sha256:$is,decode_sha256:$ds,
            deployment_validator:{path:$dvp,sha256:$dvs},
            authoring_manifest:{path:$ap,sha256:$as},
            deployment_marker:{path:$mp,sha256:$ms},archive:{path:$xp,sha256:$xs},
            revision:{path:$rp,sha256:$rs},content_manifest:{path:$cp,sha256:$cs2}
          },
          zero_update_frozen_base:{training_performed:false,optimization_steps:0,checkpoint:null,adapter:null},
          matched_contract:{
            event:0,outer_seed:2027,frames:81,fps:25,
            source_sha256:"888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de",
            anchor_sha256:"e485b2963d5ba191d8fc98158b7ab100bd1bd7584735221077c014ff0c3a19aa",
            anchor_gaussian_file_sha256:"e4828f4594caf65b45a937a271847dcf621a75e4f6862269a28216759ff087b7",
            anchor_gaussian_raw_sha256:"38438c769db8a539ed84902e8493d66ab257144a97be9f4f57dfccb80880f832",
            initial_noise_proposal_mode:"anchor_candidate0",anchor_state_mode:"clean_noised",
            anchor_contrast_mode:"dynamic_static_same_caption",anchor_cfg_scope:"shared",
            source_cfg_scale:4.5,target_cfg_scale:4.5,initial_phase_clamp:true,
            sga_score_mode:"global_source_cosine",anchor_action_reward_used_for_sga:false,
            sga_weights_forced_to_anchor_candidate0:false,
            candidate_counts:([5,5,5] + [range(37) | 1]),
            prompts:{
              action_mv2v_sha256:"c535bd8ebf9b3ff2de08d15f1fbf0327f8610a42f4c6ff07754d5ccf4d747de2",
              source_noop_mv2v_sha256:"67a74d3aafbbca1a42598a431bef3b66f9b24789c066da08d4097a7eafdc223b",
              anchor_t2v_sha256:"8d33aa4b3bf0459cf7dc850f6a65c1d583e83614e51e9783d2754939a9d2a7f6",
              anchor_noop_t2v_sha256:"033b947d7d83223ae61c84adb08b2d9f81001834e451a79dee79e9f4c49de981",
              source_t2v_sha256:"c424fda6ec36b5c78a856f8abcba1db217c407f2920317832cc0e26095274451",
              target_t2v_sha256:"e7e4e5dd2b0cae49950f6cf42829a49d0c3995cfc69f1b80ca949d74d9191388",
              negative_sha256:"ce96e0324e4b54ce4b6e867f669ca520952e1a34cc116543516b1897f0d3c47e"
            }
          },
          arms:[
            {execution_phase:"S0_routeoff_preflight",role:"route_off_plain_frozen",label:$rl,transport:"self_target_owned_temporal_kernel_attn_output_v14r2",transport_steps:0},
            {execution_phase:"S1_routeon",role:"route_on_temporal",label:$tl,transport:"self_target_owned_temporal_kernel_attn_output_v14r2",transport_steps:40},
            {execution_phase:"S1_routeon",role:"route_on_activity25",label:$al,transport:"self_target_owned_activity_kernel25_attn_output_v14r2",transport_steps:40}
          ]
        }'\'' > "$receipt_tmp"
      test -f "$receipt_tmp"; test ! -L "$receipt_tmp"
      jq -e '\''.complete == true'\'' "$receipt_tmp" >/dev/null
      mv -- "$receipt_tmp" "$LAUNCH_RECEIPT"
      receipt_sha="$(sha256sum -- "$LAUNCH_RECEIPT")"; receipt_sha="${receipt_sha%% *}"
      printf "%s\n" "$receipt_sha"
    ')"
test "${#launch_receipt_compute_sha}" = 64
case "$launch_receipt_compute_sha" in *[!0-9a-f]*) exit 5 ;; esac
wait_sha_complete_json_visible \
  "$launch_receipt" "$launch_receipt_compute_sha" "v15a-r3 launch receipt"
jq -e --arg job "$job" --arg node "$node" --arg tag "$tag" --arg root "$output_root" '
  .complete == true and .parent_job_id == $job and .compute_node == $node and
  .experiment_tag == $tag and .fixed_output_root == $root and
  (.arms | map(.role)) == ["route_off_plain_frozen","route_on_temporal","route_on_activity25"] and
  (.arms | map(.transport_steps)) == [0,40,40] and
  .matched_contract.outer_seed == 2027 and
  .matched_contract.candidate_counts == ([5,5,5] + [range(37) | 1]) and
  (.matched_contract.prompts | length) == 7
' "$launch_receipt" >/dev/null

run_arm() {
  local label="$1" transport="$2" transport_steps="$3" arm_role="$4"
  local video sidecar audit sidecar_video_path sidecar_video_sha
  video="$output_root/$label.mp4"
  sidecar="$video.receipt.json"
  audit="$video.v15a-audit.json"
  test ! -e "$video"; test ! -L "$video"
  test ! -e "$sidecar"; test ! -L "$sidecar"
  test ! -e "$audit"; test ! -L "$audit"

  srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --cpus-per-task=64 \
    --gres=gpu:4 --mem=0 --nodelist="$node" \
    env DEV_OVERRIDE="$dev" LABEL_OVERRIDE="$label" OUTPUT_ROOT_OVERRIDE="$output_root" \
      TRANSPORT_OVERRIDE="$transport" STRENGTH_OVERRIDE=1.0 TRANSPORT_STEPS_OVERRIDE="$transport_steps" \
      BLOCKS_OVERRIDE=1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26,27,29 \
      EVENT_ORDINAL_OVERRIDE=0 SOURCE_VIDEO_OVERRIDE="$source_video" \
      EXPECTED_SOURCE_SHA256_OVERRIDE=888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de \
      ANCHOR_DIR_OVERRIDE="$anchor_dir" \
      EXPECTED_ANCHOR_SHA256_OVERRIDE=e485b2963d5ba191d8fc98158b7ab100bd1bd7584735221077c014ff0c3a19aa \
      ANCHOR_INITIAL_GAUSSIAN_OVERRIDE="$anchor_gaussian" \
      EXPECTED_ANCHOR_INITIAL_GAUSSIAN_SHA256_OVERRIDE=e4828f4594caf65b45a937a271847dcf621a75e4f6862269a28216759ff087b7 \
      EXPECTED_ANCHOR_INITIAL_GAUSSIAN_RAW_SHA256_OVERRIDE=38438c769db8a539ed84902e8493d66ab257144a97be9f4f57dfccb80880f832 \
      SOURCE_CAPTION_OVERRIDE="$source_caption" TARGET_CAPTION_OVERRIDE="$target_caption" \
      ANCHOR_CAPTION_OVERRIDE="$anchor_caption" ANCHOR_NOOP_CAPTION_OVERRIDE="$noop_caption" \
      ANCHOR_STATE_MODE=clean_noised \
      INITIAL_NOISE_PROPOSAL_MODE=anchor_candidate0 INITIAL_PHASE_CLAMP=1 \
      ANCHOR_CFG_SCOPE_OVERRIDE=shared SOURCE_CFG_SCALE_OVERRIDE=4.5 TARGET_CFG_SCALE_OVERRIDE=4.5 \
      ANCHOR_CONTRAST_MODE=dynamic_static_same_caption ANCHOR_SIGMA_CAP=1.0 \
      PRESERVATION_MODE=none PRESERVATION_RESIDUAL_FRACTION=0.0 \
      PRESERVATION_OBJECT_IDENTITY_STRENGTH=0.0 SGA_SCORE_MODE=global_source_cosine \
      ANCHOR_CANDIDATE_MODE=single_shared ANCHOR_SPATIAL_ALIGNMENT=none \
      EVENT01_FORCED_ROLE_PROPOSAL_INDEX=-1 EARLY_CANDIDATE_COUNT=5 ARM_OVERRIDE=AQK_SGA5 \
      TRAINED_ATTENTION_CHECKPOINT= TRAINED_ATTENTION_EXPECTED_STEP= \
      TRAINED_ATTENTION_EXPECTED_OBJECTIVE= TRAINED_ATTENTION_EXPECTED_ROUTE_OPERATOR= \
      TRAINED_ATTENTION_EXPECTED_ADAPTER_SHA256= TRAINED_ATTENTION_EXPECTED_ADAPTER_CONFIG_SHA256= \
      TRAINED_ATTENTION_EXPECTED_RECEIPT_SHA256= ALLOW_TRAINED_ROUTE_OFF_CONTROL=0 \
      EXPECTED_COMPUTE_NODE="$node" EXPECTED_PARENT_JOB="$job" BRIDGE_RUNNER="$bridge" \
      bash -c 'test "$(hostname -s)" = "$EXPECTED_COMPUTE_NODE"; test "$SLURM_JOB_ID" = "$EXPECTED_PARENT_JOB"; exec bash "$BRIDGE_RUNNER" 0'

  wait_complete_json_visible "$sidecar" "$arm_role native sidecar"
  sidecar_video_path="$(jq -r '.output.path' "$sidecar")"
  sidecar_video_sha="$(jq -r '.output.sha256' "$sidecar")"
  test "$sidecar_video_path" = "$video"
  wait_sha_file_visible "$video" "$sidecar_video_sha" "$arm_role MP4"
  "$python_bin" -B "$sidecar_validator" \
    --sidecar "$sidecar" --video "$video" --transport "$transport" \
    --arm-role "$arm_role" \
    --audit-output "$audit" --source-tree "$source_tree" \
    --deployment-marker "$marker" --archive "$archive" --revision "$revision" \
    --content-manifest "$content" --controller "$dev/anchor_sga_anc_controller.py" \
    --qk-transport "$dev/anchor_qk_transport.py" \
    --infer "$dev/infer_anchor_sga_anc_event_v1.py" \
    --decode "$dev/infer_anchor_sga_anc_trained_editor_decode_v1.py" \
    --bridge "$bridge" --deployment-validator "$deployment_validator" \
    --authoring-manifest "$authoring_manifest"
  echo "V15A_COMPLETE $arm_role $video $audit"
}

# Fail closed on the real plain-frozen route-off ABI before spending two
# full route-on decodes.  Only a validated S0 result permits S1 to start.
run_arm "$routeoff_label" self_target_owned_temporal_kernel_attn_output_v14r2 0 route_off_plain_frozen
run_arm "$temporal_label" self_target_owned_temporal_kernel_attn_output_v14r2 40 route_on_temporal
run_arm "$activity_label" self_target_owned_activity_kernel25_attn_output_v14r2 40 route_on_activity25

temporal_audit="$output_root/$temporal_label.mp4.v15a-audit.json"
activity_audit="$output_root/$activity_label.mp4.v15a-audit.json"
routeoff_audit="$output_root/$routeoff_label.mp4.v15a-audit.json"
manifest="$output_root/v15a_matched_triplet_manifest.json"
test ! -e "$manifest"; test ! -L "$manifest"
for audit in "$temporal_audit" "$activity_audit" "$routeoff_audit"; do
  test -f "$audit"; test ! -L "$audit"
done
temporal_sha="$(sha256sum -- "$temporal_audit")"; temporal_sha="${temporal_sha%% *}"
activity_sha="$(sha256sum -- "$activity_audit")"; activity_sha="${activity_sha%% *}"
routeoff_sha="$(sha256sum -- "$routeoff_audit")"; routeoff_sha="${routeoff_sha%% *}"
launch_receipt_sha="$(sha256sum -- "$launch_receipt")"; launch_receipt_sha="${launch_receipt_sha%% *}"
jq -n -S \
  --arg group e00-v15a-r3-plain-frozen-max-strength-clean-noised-route-probe-v1 \
  --arg tag "$tag" --arg root "$output_root" \
  --arg rl "$routeoff_label" --arg tl "$temporal_label" --arg al "$activity_label" \
  --arg ta "$temporal_audit" --arg ts "$temporal_sha" \
  --arg aa "$activity_audit" --arg as "$activity_sha" \
  --arg ra "$routeoff_audit" --arg rs "$routeoff_sha" \
  --arg lr "$launch_receipt" --arg lrs "$launch_receipt_sha" '
  {
    schema_version:"bernini-v15a-r3-zero-update-matched-triplet-manifest-v1",
    complete:true,
    comparison_group_id:$group,
    experiment_tag:$tag,
    fixed_output_root:$root,
    probe_kind:"max-strength clean-noised route probe",
    launch_receipt:{path:$lr,sha256:$lrs},
    matched_contract:{
      event:0, source_sha256:"888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de",
      anchor_sha256:"e485b2963d5ba191d8fc98158b7ab100bd1bd7584735221077c014ff0c3a19aa",
      anchor_gaussian_file_sha256:"e4828f4594caf65b45a937a271847dcf621a75e4f6862269a28216759ff087b7",
      authoring_manifest_sha256:"767aa8e0502f247c3ab576db4c40a132344295e54aef4170e728bc3ff71cafc5",
      outer_seed:2027, frames:81, fps:25,
      initial_noise_proposal_mode:"anchor_candidate0", arm:"AQK_SGA5",
      anchor_state_mode:"clean_noised",anchor_contrast_mode:"dynamic_static_same_caption",
      source_cfg_scale:4.5, target_cfg_scale:4.5, initial_phase_clamp:true,
      sga_score_mode:"global_source_cosine",anchor_action_reward_used_for_sga:false,
      sga_weights_forced_to_anchor_candidate0:false,
      candidate_counts:([5,5,5] + [range(37) | 1]),
      prompt_sha256:{
        action_mv2v_sha256:"c535bd8ebf9b3ff2de08d15f1fbf0327f8610a42f4c6ff07754d5ccf4d747de2",
        source_noop_mv2v_sha256:"67a74d3aafbbca1a42598a431bef3b66f9b24789c066da08d4097a7eafdc223b",
        anchor_t2v_sha256:"8d33aa4b3bf0459cf7dc850f6a65c1d583e83614e51e9783d2754939a9d2a7f6",
        anchor_noop_t2v_sha256:"033b947d7d83223ae61c84adb08b2d9f81001834e451a79dee79e9f4c49de981",
        source_t2v_sha256:"c424fda6ec36b5c78a856f8abcba1db217c407f2920317832cc0e26095274451",
        target_t2v_sha256:"e7e4e5dd2b0cae49950f6cf42829a49d0c3995cfc69f1b80ca949d74d9191388",
        negative_sha256:"ce96e0324e4b54ce4b6e867f669ca520952e1a34cc116543516b1897f0d3c47e"
      },
      frozen_base:true,training_performed:false,optimization_steps:0,
      trained_checkpoint:null,adapter:null
    },
    arms:[
      {execution_phase:"S0_routeoff_preflight",role:"route_off_plain_frozen",label:$rl,transport_steps:0,audit_path:$ra,audit_sha256:$rs},
      {execution_phase:"S1_routeon",role:"route_on_temporal",label:$tl,transport_steps:40,audit_path:$ta,audit_sha256:$ts},
      {execution_phase:"S1_routeon",role:"route_on_activity25",label:$al,transport_steps:40,audit_path:$aa,audit_sha256:$as}
    ]
  }' > "$manifest"
jq -e --arg group e00-v15a-r3-plain-frozen-max-strength-clean-noised-route-probe-v1 \
  --arg tag "$tag" --arg root "$output_root" '
  .complete == true and .comparison_group_id == $group and
  .experiment_tag == $tag and .fixed_output_root == $root and
  (.arms | map(.role)) == ["route_off_plain_frozen","route_on_temporal","route_on_activity25"] and
  (.arms | map(.transport_steps)) == [0,40,40] and
  .matched_contract.outer_seed == 2027 and
  .matched_contract.candidate_counts == ([5,5,5] + [range(37) | 1]) and
  (.matched_contract.prompt_sha256 | length) == 7 and
  all(.arms[]; (.audit_sha256 | test("^[0-9a-f]{64}$")))
' "$manifest" >/dev/null
jq -e -s --arg group e00-v15a-r3-plain-frozen-max-strength-clean-noised-route-probe-v1 '
  length == 3 and all(.[]; .complete == true and .comparison.group_id == $group) and
  (map(.comparison.arm_role)) == ["route_off_plain_frozen","route_on_temporal","route_on_activity25"] and
  (.[0].qk_route_proof | .route_injection_enabled == false and .active_steps == 0 and .capture_count == 0 and .replay_count == 0 and .cached_fields == [])
' "$routeoff_audit" "$temporal_audit" "$activity_audit" >/dev/null
touch "$output_root/V15A_MATCHED_TRIPLET_COMPLETE"
