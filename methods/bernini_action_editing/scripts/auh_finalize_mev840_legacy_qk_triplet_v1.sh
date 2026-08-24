#!/usr/bin/env bash
set -euo pipefail

# Login-node postflight for the completed MEV840 legacy-QK triplet.  The
# compute launcher intentionally remains immutable; its only failure was that
# ffprobe is not installed on the three compute-node images.

[ "$#" -eq 0 ] || { echo "usage: $0" >&2; exit 2; }

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
output_root="$stage/mev840_legacy_dynamic_static_qk_diag_v1_20260821"
control_root="$stage/mev840_legacy_dynamic_static_qk_diag_v1_20260821_control"
launcher="$control_root/auh_launch_mev840_legacy_qk_triplet_v1.sh"
launcher_sha=48636ffa162ed49106403f5e29bc12f40f51ff0830c1cd1fdb348d3e68a20f76
source_sha=a6e42a447d7fef26b073878c551a631afd6371987910a332e51e4a9e4dfb4646
self_anchor_sha=412399051be25c19ec9ab7d1406b1e6982e31e75cfa1b3920259a6c22f66113b
oracle_anchor_sha=355535f4f5ff83581c2286dfb70a64c7f5131f5ae81d76fbb6351b2aa972baf0

command -v ffprobe >/dev/null
command -v jq >/dev/null
command -v sha256sum >/dev/null
actual_launcher_sha="$(sha256sum -- "$launcher")"; actual_launcher_sha="${actual_launcher_sha%% *}"
[ "$actual_launcher_sha" = "$launcher_sha" ]

finalize_arm() {
  local arm="$1" label="$2" expected_job="$3" expected_node="$4" transport_steps="$5" anchor_sha="$6"
  local output native_receipt audit complete video_sha receipt_sha probe width height fps frames
  local audit_tmp complete_tmp audit_sha
  output="$output_root/$label.mp4"
  native_receipt="$output.receipt.json"
  audit="$output.legacy-mev840-audit.json"
  complete="$output.complete.json"
  [ -f "$output" ] && [ ! -L "$output" ]
  [ -f "$native_receipt" ] && [ ! -L "$native_receipt" ]
  [ ! -e "$audit" ] && [ ! -L "$audit" ]
  [ ! -e "$complete" ] && [ ! -L "$complete" ]

  video_sha="$(sha256sum -- "$output")"; video_sha="${video_sha%% *}"
  receipt_sha="$(sha256sum -- "$native_receipt")"; receipt_sha="${receipt_sha%% *}"
  [ "$(jq -r '.output.sha256' "$native_receipt")" = "$video_sha" ]
  jq -e \
    --arg source_sha "$source_sha" --arg anchor_sha "$anchor_sha" \
    --argjson transport_steps "$transport_steps" '
      .complete == true and
      .training_performed == false and
      .optimization_steps == 0 and
      .loaded_trained_attention_checkpoint == false and
      .trained_attention_checkpoint == null and
      .source.sha256 == $source_sha and
      .pure_t2v_anchor.sha256 == $anchor_sha and
      .mechanism.arm == "AQK_SGA5" and
      .mechanism.decode_audit_contract.transport_steps == $transport_steps and
      .mechanism.initial_noise_proposal_mode == "keyed_only" and
      .mechanism.anchor_state_mode == "clean_noised" and
      .mechanism.anchor_cfg_scope == "shared" and
      .mechanism.anchor_contrast_mode == "dynamic_static_same_caption" and
      .mechanism.preservation_mode == "none" and
      .freeze_before == .freeze_after and
      .freeze_before.base_frozen == true and
      .freeze_before.adapter_modules_absent == true and
      .freeze_before.lora_module_count == 0 and
      .freeze_before.trainable_parameter_tensors == 0 and
      .freeze_before.trainable_parameter_elements == 0 and
      .mechanism.trace.attention_cache.pending_entries == 0 and
      .mechanism.trace.attention_cache.qk_only_forbidden_cached_fields ==
        ["value","hidden_state","attention_output","rgb","latent","absolute_spatial_coordinate"]
    ' "$native_receipt" >/dev/null

  if [ "$transport_steps" -eq 0 ]; then
    jq -e '
      .mechanism.trace.anchor_model_forwards == 0 and
      .mechanism.trace.anchor_candidate_cells == 0 and
      .mechanism.trace.attention_cache.qk_only_capture_count == 0 and
      .mechanism.trace.attention_cache.qk_only_replay_count == 0 and
      .mechanism.pure_t2v_anchor_online_block_transport_enabled == false
    ' "$native_receipt" >/dev/null
  else
    jq -e '
      (.mechanism.trace.anchor_active_schedule | length) == 40 and
      .mechanism.trace.anchor_candidate_cells == 52 and
      .mechanism.trace.anchor_model_forwards == 104 and
      .mechanism.trace.attention_cache.qk_only_capture_count == 2288 and
      .mechanism.trace.attention_cache.qk_only_replay_count == 4576 and
      .mechanism.trace.anchor_target_activity_gated_hard_kernel == true and
      .mechanism.trace.anchor_temporal_attention_kernel_contrast == true and
      .mechanism.trace.anchor_temporal_kernel_applied_to_target_value_only == true and
      .mechanism.trace.anchor_value_stream_copied == false
    ' "$native_receipt" >/dev/null
  fi

  probe="$(ffprobe -v error -select_streams v:0 -count_frames \
    -show_entries stream=width,height,r_frame_rate,nb_read_frames -of json "$output")"
  width="$(jq -er '.streams[0].width' <<<"$probe")"
  height="$(jq -er '.streams[0].height' <<<"$probe")"
  fps="$(jq -er '.streams[0].r_frame_rate' <<<"$probe")"
  frames="$(jq -er '.streams[0].nb_read_frames' <<<"$probe")"
  [ "$width" = 656 ] && [ "$height" = 368 ] && [ "$fps" = 25/1 ] && [ "$frames" = 81 ]

  audit_tmp="$audit.tmp.$$"
  jq -n \
    --arg arm "$arm" --arg label "$label" --arg job "$expected_job" --arg node "$expected_node" \
    --arg source_sha "$source_sha" --arg anchor_sha "$anchor_sha" \
    --arg output "$output" --arg video_sha "$video_sha" \
    --arg receipt "$native_receipt" --arg receipt_sha "$receipt_sha" \
    --arg launcher "$launcher" --arg launcher_sha "$launcher_sha" \
    --argjson width "$width" --argjson height "$height" --arg fps "$fps" --argjson frames "$frames" \
    --argjson transport_steps "$transport_steps" \
    '{schema:"mev840-legacy-dynamic-static-qk-arm-audit-v1",complete:true,
      postflight_recovery:{required:true,reason:"compute node lacked ffprobe",inference_was_not_rerun:true,original_step_exit_code:127},
      claim_boundary:"legacy dynamic/static QK diagnostic; not v15b role graph; not source-property seam; oracle arm reads retimed real target",
      zero_update:true,training_performed:false,optimization_steps:0,
      arm:$arm,label:$label,slurm:{job_id:$job,node:$node},
      mechanism:{transport:"self_target_owned_activity_kernel25_attn_output_v14r2",transport_steps:$transport_steps,strength:1,
        initial_noise_proposal_mode:"keyed_only",anchor_gaussian_supplied:false,anchor_state_mode:"clean_noised",
        anchor_contrast_mode:"dynamic_static_same_caption",anchor_cfg_scope:"shared",preservation_mode:"none"},
      authority:{original_launcher:{path:$launcher,sha256:$launcher_sha},source_sha256:$source_sha,anchor_sha256:$anchor_sha},
      output:{path:$output,sha256:$video_sha,width:$width,height:$height,fps:$fps,frames:$frames},
      native_receipt:{path:$receipt,sha256:$receipt_sha}}' >"$audit_tmp"
  jq -e '.complete and .zero_update and (.postflight_recovery.inference_was_not_rerun == true)' "$audit_tmp" >/dev/null
  mv "$audit_tmp" "$audit"
  audit_sha="$(sha256sum -- "$audit")"; audit_sha="${audit_sha%% *}"
  complete_tmp="$complete.tmp.$$"
  jq -n --arg audit "$audit" --arg audit_sha "$audit_sha" --arg output "$output" --arg output_sha "$video_sha" \
    '{schema:"mev840-legacy-dynamic-static-qk-complete-v1",complete:true,audit:{path:$audit,sha256:$audit_sha},output:{path:$output,sha256:$output_sha}}' >"$complete_tmp"
  jq -e '.complete == true' "$complete_tmp" >/dev/null
  mv "$complete_tmp" "$complete"
  printf 'MEV840_POSTFLIGHT_COMPLETE %s %s %s\n' "$arm" "$video_sha" "$audit_sha"
}

finalize_arm routeoff MEV840_LEGACY_QK_MATCHED_ROUTEOFF_K0 147881 auh7-1b-gpu-213 0 "$self_anchor_sha"
finalize_arm self MEV840_LEGACY_QK_SELF_ANCHOR_ACTIVITY25_K40 147873 auh7-1b-gpu-284 40 "$self_anchor_sha"
finalize_arm oracle MEV840_LEGACY_QK_RETIMED_REAL_TARGET_ORACLE_ACTIVITY25_K40 147871 auh7-1b-gpu-232 40 "$oracle_anchor_sha"

echo "MEV840_LEGACY_QK_TRIPLET_POSTFLIGHT_COMPLETE $output_root"
