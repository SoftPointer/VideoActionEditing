#!/usr/bin/env bash
set -euo pipefail

# Login-node-only postflight for the fresh six-arm legacy oracle calibration.
[ "$#" -eq 0 ] || { echo "usage: $0" >&2; exit 2; }

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
output_root="$stage/mev840_legacy_oracle_activity25_lowstrength_sweep_v1_20260821"
control_root="$stage/mev840_legacy_oracle_activity25_lowstrength_sweep_v1_20260821_control"
launcher="$control_root/auh_launch_mev840_legacy_oracle_lowstrength_sweep_v1.sh"
launcher_sha=20a132a82c502d1b4f557ce37f2e435da579594aa970290e00726de8cf4252e3
source_sha=a6e42a447d7fef26b073878c551a631afd6371987910a332e51e4a9e4dfb4646
oracle_anchor_sha=355535f4f5ff83581c2286dfb70a64c7f5131f5ae81d76fbb6351b2aa972baf0
outer_schedule_digest=43cd53329945280dccea5c1a1aa3b5da05337a7f10cfec0ab5a727592ea77d25

command -v ffprobe >/dev/null
command -v jq >/dev/null
command -v sha256sum >/dev/null
actual_launcher_sha="$(sha256sum -- "$launcher")"; actual_launcher_sha="${actual_launcher_sha%% *}"
[ "$actual_launcher_sha" = "$launcher_sha" ]
[ -d "$output_root" ] && [ ! -L "$output_root" ]

set_arm() {
  case "$1" in
    s005_k05)
      expected_job=147881; expected_node=auh7-1b-gpu-213
      strength=0.05; steps=5; expected_cells=17
      label=MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S005_K05 ;;
    s010_k05)
      expected_job=147873; expected_node=auh7-1b-gpu-284
      strength=0.10; steps=5; expected_cells=17
      label=MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S010_K05 ;;
    s025_k05)
      expected_job=147871; expected_node=auh7-1b-gpu-232
      strength=0.25; steps=5; expected_cells=17
      label=MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S025_K05 ;;
    s005_k10)
      expected_job=143808; expected_node=auh7-1b-gpu-268
      strength=0.05; steps=10; expected_cells=22
      label=MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S005_K10 ;;
    s010_k10)
      expected_job=143808; expected_node=auh7-1b-gpu-315
      strength=0.10; steps=10; expected_cells=22
      label=MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S010_K10 ;;
    s025_k10)
      expected_job=143808; expected_node=auh7-1b-gpu-233
      strength=0.25; steps=10; expected_cells=22
      label=MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S025_K10 ;;
    *) echo "unknown arm: $1" >&2; exit 2 ;;
  esac
  expected_forwards=$((expected_cells * 2))
  expected_captures=$((expected_forwards * 22))
  expected_replays=$((expected_captures * 2))
}

arms=(s005_k05 s010_k05 s025_k05 s005_k10 s010_k10 s025_k10)

# Fail before writing any audit if one inference or output-chain target differs.
for arm in "${arms[@]}"; do
  set_arm "$arm"
  output="$output_root/$label.mp4"
  native_receipt="$output.receipt.json"
  worker_receipt="$output.worker.json"
  audit="$output.legacy-mev840-lowstrength-audit.json"
  complete="$output.complete.json"
  [ -f "$output" ] && [ ! -L "$output" ]
  [ -f "$native_receipt" ] && [ ! -L "$native_receipt" ]
  [ -f "$worker_receipt" ] && [ ! -L "$worker_receipt" ]
  [ ! -e "$audit" ] && [ ! -L "$audit" ]
  [ ! -e "$complete" ] && [ ! -L "$complete" ]
done

finalize_arm() {
  local arm="$1" output native_receipt worker_receipt audit complete
  local video_sha receipt_sha worker_sha probe width height fps frames
  local audit_tmp complete_tmp audit_sha step_id
  set_arm "$arm"
  output="$output_root/$label.mp4"
  native_receipt="$output.receipt.json"
  worker_receipt="$output.worker.json"
  audit="$output.legacy-mev840-lowstrength-audit.json"
  complete="$output.complete.json"

  video_sha="$(sha256sum -- "$output")"; video_sha="${video_sha%% *}"
  receipt_sha="$(sha256sum -- "$native_receipt")"; receipt_sha="${receipt_sha%% *}"
  worker_sha="$(sha256sum -- "$worker_receipt")"; worker_sha="${worker_sha%% *}"
  [ "$(jq -r '.output.sha256' "$native_receipt")" = "$video_sha" ]
  jq -e \
    --arg source_sha "$source_sha" --arg anchor_sha "$oracle_anchor_sha" \
    --arg schedule "$outer_schedule_digest" \
    --argjson strength "$strength" --argjson steps "$steps" \
    --argjson cells "$expected_cells" --argjson forwards "$expected_forwards" \
    --argjson captures "$expected_captures" --argjson replays "$expected_replays" '
      .complete == true and
      .training_performed == false and .optimization_steps == 0 and
      .loaded_trained_attention_checkpoint == false and .trained_attention_checkpoint == null and
      .source.sha256 == $source_sha and .pure_t2v_anchor.sha256 == $anchor_sha and
      .anchor_generation_initial_gaussian == null and
      .mechanism.arm == "AQK_SGA5" and
      .mechanism.transport == "self_target_owned_activity_kernel25_attn_output_v14r2" and
      .mechanism.transport_strength == $strength and
      .mechanism.decode_audit_contract.transport_steps == $steps and
      .mechanism.initial_noise_proposal_mode == "keyed_only" and
      .mechanism.anchor_state_mode == "clean_noised" and
      .mechanism.anchor_cfg_scope == "shared" and
      .mechanism.anchor_contrast_mode == "dynamic_static_same_caption" and
      .mechanism.field_guidance == "raw_cfg" and
      .mechanism.field_model == "first_phase_caption_i2v" and
      .mechanism.source_cfg_scale == 4.5 and .mechanism.target_cfg_scale == 4.5 and
      .mechanism.anchor_sigma_cap == 1 and .mechanism.preservation_mode == "none" and
      .mechanism.early_candidate_count == 5 and
      .mechanism.sga_temperature == 0.01 and
      .mechanism.sga_score_mode == "global_source_cosine" and
      .mechanism.anchor_candidate_mode == "single_shared" and
      .mechanism.anchor_spatial_alignment == "none" and
      .mechanism.trace.outer_schedule_digest == $schedule and
      (.mechanism.trace.anchor_active_schedule | length) == $steps and
      [.mechanism.trace.anchor_active_schedule[].step_index] == [range(0;$steps)] and
      .mechanism.trace.anchor_candidate_cells == $cells and
      .mechanism.trace.anchor_model_forwards == $forwards and
      .mechanism.trace.attention_cache.qk_only_capture_count == $captures and
      .mechanism.trace.attention_cache.qk_only_replay_count == $replays and
      .mechanism.trace.attention_cache.pending_entries == 0 and
      .mechanism.trace.attention_cache.qk_only_forbidden_cached_fields ==
        ["value","hidden_state","attention_output","rgb","latent","absolute_spatial_coordinate"] and
      .mechanism.trace.anchor_target_activity_gated_hard_kernel == true and
      .mechanism.trace.anchor_temporal_attention_kernel_contrast == true and
      .mechanism.trace.anchor_temporal_kernel_applied_to_target_value_only == true and
      .mechanism.trace.anchor_value_stream_copied == false and
      .freeze_before == .freeze_after and
      .freeze_before.base_frozen == true and
      .freeze_before.adapter_modules_absent == true and
      .freeze_before.lora_module_count == 0 and
      .freeze_before.trainable_parameter_tensors == 0 and
      .freeze_before.trainable_parameter_elements == 0 and
      (.rank_closure | length) == 4 and
      ([.rank_closure[].latent.content_sha256] | unique | length) == 1 and
      ([.rank_closure[].trace_digest] | unique | length) == 1
    ' "$native_receipt" >/dev/null

  jq -e \
    --arg arm "$arm" --arg job "$expected_job" --arg node "$expected_node" \
    --arg output_sha "$video_sha" --arg receipt_sha "$receipt_sha" \
    --argjson strength "$strength" --argjson steps "$steps" '
      .complete == true and .zero_update == true and
      .training_performed == false and .optimization_steps == 0 and
      .arm == $arm and .slurm.job_id == $job and .slurm.node == $node and
      (.slurm.step_id | length) > 0 and
      .mechanism.strength == $strength and .mechanism.transport_steps == $steps and
      .output.sha256 == $output_sha and .native_receipt.sha256 == $receipt_sha
    ' "$worker_receipt" >/dev/null
  step_id="$(jq -er '.slurm.step_id' "$worker_receipt")"

  probe="$(ffprobe -v error -select_streams v:0 -count_frames \
    -show_entries stream=width,height,r_frame_rate,nb_read_frames -of json "$output")"
  width="$(jq -er '.streams[0].width' <<<"$probe")"
  height="$(jq -er '.streams[0].height' <<<"$probe")"
  fps="$(jq -er '.streams[0].r_frame_rate' <<<"$probe")"
  frames="$(jq -er '.streams[0].nb_read_frames' <<<"$probe")"
  [ "$width" = 656 ] && [ "$height" = 368 ] && [ "$fps" = 25/1 ] && [ "$frames" = 81 ]

  audit_tmp="$audit.tmp.$$"
  jq -n \
    --arg arm "$arm" --arg label "$label" --arg job "$expected_job" --arg step "$step_id" --arg node "$expected_node" \
    --arg source_sha "$source_sha" --arg anchor_sha "$oracle_anchor_sha" \
    --arg output "$output" --arg video_sha "$video_sha" \
    --arg receipt "$native_receipt" --arg receipt_sha "$receipt_sha" \
    --arg worker "$worker_receipt" --arg worker_sha "$worker_sha" \
    --arg launcher "$launcher" --arg launcher_sha "$launcher_sha" \
    --argjson width "$width" --argjson height "$height" --arg fps "$fps" --argjson frames "$frames" \
    --argjson strength "$strength" --argjson transport_steps "$steps" \
    '{schema:"mev840-legacy-oracle-lowstrength-arm-audit-v1",complete:true,
      claim_boundary:"zero-update legacy activity25 Q/K operator calibration with normalized-real-target oracle; not v15 action graph; not training evidence",
      postflight:{location:"login_node",compute_ffprobe_invoked:false,inference_rerun:false},
      zero_update:true,training_performed:false,optimization_steps:0,oracle_reads_real_target:true,
      arm:$arm,label:$label,slurm:{job_id:$job,step_id:$step,node:$node},
      mechanism:{transport:"self_target_owned_activity_kernel25_attn_output_v14r2",strength:$strength,
        transport_steps:$transport_steps,initial_noise_proposal_mode:"keyed_only",anchor_gaussian_supplied:false,
        anchor_state_mode:"clean_noised",anchor_contrast_mode:"dynamic_static_same_caption",
        anchor_cfg_scope:"shared",activity_keep_fraction:0.25,preservation_mode:"none"},
      authority:{launcher:{path:$launcher,sha256:$launcher_sha},source_sha256:$source_sha,oracle_anchor_sha256:$anchor_sha},
      output:{path:$output,sha256:$video_sha,width:$width,height:$height,fps:$fps,frames:$frames},
      native_receipt:{path:$receipt,sha256:$receipt_sha},worker_receipt:{path:$worker,sha256:$worker_sha}}' >"$audit_tmp"
  jq -e '.complete and .zero_update and (.postflight.compute_ffprobe_invoked == false)' "$audit_tmp" >/dev/null
  mv "$audit_tmp" "$audit"
  audit_sha="$(sha256sum -- "$audit")"; audit_sha="${audit_sha%% *}"
  complete_tmp="$complete.tmp.$$"
  jq -n --arg audit "$audit" --arg audit_sha "$audit_sha" --arg output "$output" --arg output_sha "$video_sha" \
    '{schema:"mev840-legacy-oracle-lowstrength-complete-v1",complete:true,
      audit:{path:$audit,sha256:$audit_sha},output:{path:$output,sha256:$output_sha}}' >"$complete_tmp"
  jq -e '.complete == true' "$complete_tmp" >/dev/null
  mv "$complete_tmp" "$complete"
  printf 'MEV840_LEGACY_ORACLE_LOWSTRENGTH_POSTFLIGHT_COMPLETE %s %s %s %s\n' "$arm" "$step_id" "$video_sha" "$audit_sha"
}

for arm in "${arms[@]}"; do
  finalize_arm "$arm"
done

echo "MEV840_LEGACY_ORACLE_LOWSTRENGTH_SWEEP_POSTFLIGHT_COMPLETE $output_root"
