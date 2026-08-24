#!/usr/bin/env bash
set -euo pipefail

# Extract motion from the actual self-generated video, not only from an
# action-vs-noop caption change on the same dynamic latent.  The reference is
# the anchor's first latent phase repeated through time under the same action
# caption and the same candidate noise.
if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$1" in 0) devices=0,1,2,3 ;; 1) devices=4,5,6,7 ;; *) exit 2 ;; esac
case "$(hostname -s):$1" in
  auh7-1b-gpu-246:0)
    label=DYNSTATIC_VEL_A010_EARLY8
    transport=temporal_contrast_velocity
    strength=0.10
    blocks=8-13
    ;;
  auh7-1b-gpu-246:1)
    label=DYNSTATIC_VEL_A025_EARLY8
    transport=temporal_contrast_velocity
    strength=0.25
    blocks=8-13
    ;;
  auh7-1b-gpu-247:0)
    label=DYNSTATIC_SELFATT_A003_EARLY8_B8_13
    transport=temporal_contrast_attn_output
    strength=0.03
    blocks=8-13
    ;;
  auh7-1b-gpu-247:1)
    label=DYNSTATIC_CROSSATT_A003_EARLY8_B4_9
    transport=temporal_contrast_cross_attn_output
    strength=0.03
    blocks=4-9
    ;;
  auh7-1b-gpu-226:0|auh7-1b-gpu-226:1|\
  auh7-1b-gpu-233:0|auh7-1b-gpu-233:1|\
  auh7-1b-gpu-268:0|auh7-1b-gpu-268:1|\
  auh7-1b-gpu-292:0|auh7-1b-gpu-292:1|\
  auh7-1b-gpu-293:0|auh7-1b-gpu-293:1|\
  auh7-1b-gpu-306:0|auh7-1b-gpu-306:1|\
  auh7-1b-gpu-315:0|auh7-1b-gpu-315:1)
    label=ONLINE_ANCHOR_TRAINED_LORA_BRIDGE
    transport=temporal_contrast_cross_attn_output
    strength=1.0
    blocks=4-9
    ;;
  *) echo "this canary uses Job 140846 nodes 246-247" >&2; exit 3 ;;
esac
label="${LABEL_OVERRIDE:-$label}"
strength="${STRENGTH_OVERRIDE:-$strength}"
transport="${TRANSPORT_OVERRIDE:-$transport}"
transport_steps="${TRANSPORT_STEPS_OVERRIDE:-8}"
blocks="${BLOCKS_OVERRIDE:-$blocks}"
anchor_contrast_mode="${ANCHOR_CONTRAST_MODE:-dynamic_static_same_caption}"
anchor_sigma_cap="${ANCHOR_SIGMA_CAP:-1.0}"
preservation_mode="${PRESERVATION_MODE:-none}"
preservation_keep_fraction="${PRESERVATION_KEEP_FRACTION:-0.20}"
preservation_outside_scale="${PRESERVATION_OUTSIDE_SCALE:-0.0}"
preservation_dilation="${PRESERVATION_DILATION:-1}"
preservation_residual_fraction="${PRESERVATION_RESIDUAL_FRACTION:-0.0}"
preservation_object_identity_strength="${PRESERVATION_OBJECT_IDENTITY_STRENGTH:-0.0}"
preservation_start_step="${PRESERVATION_START_STEP:-0}"
preservation_ramp_steps="${PRESERVATION_RAMP_STEPS:-1}"
sga_score_mode="${SGA_SCORE_MODE:-global_source_cosine}"
early_candidate_count="${EARLY_CANDIDATE_COUNT:-5}"
initial_noise_proposal_mode="${INITIAL_NOISE_PROPOSAL_MODE:-keyed_only}"
anchor_state_mode="${ANCHOR_STATE_MODE:-clean_noised}"
anchor_candidate_mode="${ANCHOR_CANDIDATE_MODE:-single_shared}"
trained_attention_checkpoint="${TRAINED_ATTENTION_CHECKPOINT:-}"
trained_attention_expected_step="${TRAINED_ATTENTION_EXPECTED_STEP:-}"
trained_attention_expected_objective="${TRAINED_ATTENTION_EXPECTED_OBJECTIVE:-}"
trained_attention_expected_route_operator="${TRAINED_ATTENTION_EXPECTED_ROUTE_OPERATOR:-}"
trained_attention_expected_adapter_sha256="${TRAINED_ATTENTION_EXPECTED_ADAPTER_SHA256:-}"
trained_attention_expected_adapter_config_sha256="${TRAINED_ATTENTION_EXPECTED_ADAPTER_CONFIG_SHA256:-}"
trained_attention_expected_receipt_sha256="${TRAINED_ATTENTION_EXPECTED_RECEIPT_SHA256:-}"
allow_trained_route_off_control="${ALLOW_TRAINED_ROUTE_OFF_CONTROL:-0}"
anchor_cfg_scope="${ANCHOR_CFG_SCOPE_OVERRIDE:-target_conditional_only}"
source_cfg_scale="${SOURCE_CFG_SCALE_OVERRIDE:-4.5}"
target_cfg_scale="${TARGET_CFG_SCALE_OVERRIDE:-8.5}"
arm="${ARM_OVERRIDE:-AQK_SGA5}"
anchor_spatial_alignment="${ANCHOR_SPATIAL_ALIGNMENT:-none}"
event01_forced_role_proposal_index="${EVENT01_FORCED_ROLE_PROPOSAL_INDEX:--1}"
initial_phase_clamp="${INITIAL_PHASE_CLAMP:-0}"
initial_phase_args=(--no-initial-phase-clamp)
trained_route_off_args=()
case "$allow_trained_route_off_control" in
  0) ;;
  1) trained_route_off_args=(--allow-trained-route-off-control) ;;
  *) echo "ALLOW_TRAINED_ROUTE_OFF_CONTROL must be 0 or 1" >&2; exit 2 ;;
esac
if [ "$allow_trained_route_off_control" = 1 ] && { [ -z "$trained_attention_checkpoint" ] || [ "$transport_steps" != 0 ]; }; then
  echo "trained route-off control requires a checkpoint and zero transport steps" >&2
  exit 2
fi
if [ -n "$trained_attention_checkpoint" ] && [ "$transport_steps" = 0 ] && [ "$allow_trained_route_off_control" != 1 ]; then
  echo "zero-step trained decode requires explicit route-off causal-control opt-in" >&2
  exit 2
fi
if [ "$initial_phase_clamp" = 1 ]; then
  initial_phase_args=()
elif [ "$initial_phase_clamp" != 0 ]; then
  echo "INITIAL_PHASE_CLAMP must be 0 or 1" >&2
  exit 2
fi

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
dev="${DEV_OVERRIDE:-$stage/anchor_qk_dev_v1}"
runtime="$stage/source-be31323/methods/bernini_action_editing"
manifest="$stage/interaction_complex8_multianchor_authoring_v2.json"
source="${SOURCE_VIDEO_OVERRIDE:-/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/186d36f998a14e14/source.mp4}"
anchor_dir="${ANCHOR_DIR_OVERRIDE:-$stage/interaction_complex8_multianchor_v2_r1/e01_reach-grasp-lift-stone}"
anchor="${ANCHOR_VIDEO_OVERRIDE:-$anchor_dir/v0/t2v.mp4}"
anchor_initial_gaussian="${ANCHOR_INITIAL_GAUSSIAN_OVERRIDE:-$anchor_dir/v0/t2v.official-initial-gaussian.safetensors}"
anchor_v1="${ANCHOR_V1_OVERRIDE:-$anchor_dir/v1/t2v.mp4}"
anchor_v2="${ANCHOR_V2_OVERRIDE:-$anchor_dir/v2/t2v.mp4}"
anchor_v3="${ANCHOR_V3_OVERRIDE:-$anchor_dir/v3/t2v.mp4}"
expected_source_sha256="${EXPECTED_SOURCE_SHA256_OVERRIDE:-60618e5a988f3d8b4f48d4ae46bc7739032663a7f8805ee26d47b7d3c193af48}"
expected_anchor_sha256="${EXPECTED_ANCHOR_SHA256_OVERRIDE:-e0cdbab524bca22d2300ebc7c75723dc1a19f5eeabe619ddbddaa77f1960188e}"
expected_anchor_initial_gaussian_sha256="${EXPECTED_ANCHOR_INITIAL_GAUSSIAN_SHA256_OVERRIDE:-4f5c162add4fddc71fc68fd73d50a5187a1753c9d5533194529f46016a7bef8f}"
expected_anchor_initial_gaussian_raw_sha256="${EXPECTED_ANCHOR_INITIAL_GAUSSIAN_RAW_SHA256_OVERRIDE:-7f40fc9a2b7d85fbfcbaeefdf16a86c2eb6a71e588f2fad28f9eae32d0972e45}"
output_root="${OUTPUT_ROOT_OVERRIDE:-$stage/dynaedit_dynamic_static_anchor_event01_v27}"
output="$output_root/$label.mp4"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256

test -f "$dev/anchor_cross_attention_transport.py"
test -f "$dev/infer_anchor_sga_anc_event_v1.py"
test -f "$dev/infer_anchor_sga_anc_trained_editor_decode_v1.py"
test ! -L "$dev/infer_anchor_sga_anc_trained_editor_decode_v1.py"
test -f "$manifest"
test -f "$source"
test -f "$anchor"
extra_anchor_args=()
anchor_initial_noise_args=()
trained_attention_args=()
if [ "$anchor_candidate_mode" = bank_per_candidate ]; then
  test -f "$anchor_v1"
  test -f "$anchor_v2"
  test -f "$anchor_v3"
  extra_anchor_args=(
    --extra-anchor-video "$anchor_v1"
    --expected-extra-anchor-sha256 f1ca64bc10cb11944587e97c3055e9e9dbb5d8dbadc7ab3bf4a408ee847081a6
    --extra-anchor-video "$anchor_v2"
    --expected-extra-anchor-sha256 d3e76c7c0ffb01f8f0c2b27cc9dd0f202ce1b3117cb0d8e68f7417bb31151d1d
    --extra-anchor-video "$anchor_v3"
    --expected-extra-anchor-sha256 870dfecb1ddc11961fc99df6e8ee83b359bc481ad3b7bd9c956495c2976a8981
  )
fi
if [ -n "$trained_attention_checkpoint" ]; then
  test -f "$trained_attention_checkpoint/receipt.json"
  test -f "$trained_attention_checkpoint/adapter/adapter_config.json"
  test -f "$trained_attention_checkpoint/adapter/adapter_model.safetensors"
  case "$trained_attention_expected_step" in *[!0-9]*|'') exit 4 ;; esac
  if (( trained_attention_expected_step <= 0 )); then exit 4; fi
  case "$trained_attention_expected_objective" in target_fm|paired_delta_fm|real_source_teacher_delta|real_source_routed_teacher_delta|real_source_target_owned_routed_teacher_delta_v14r2) ;; *) exit 4 ;; esac
  case "$trained_attention_expected_route_operator" in cross_sparse|self_temporal_kernel|self_target_gated_kernel25|self_correspondence_kernel25|self_target_owned_temporal_kernel_v14r2|self_target_owned_activity_kernel10_v14r2|self_target_owned_activity_kernel25_v14r2) ;; *) exit 4 ;; esac
  if [ "${#trained_attention_expected_adapter_sha256}" -ne 64 ] || [[ "$trained_attention_expected_adapter_sha256" == *[!0-9a-f]* ]]; then exit 4; fi
  if [ "${#trained_attention_expected_adapter_config_sha256}" -ne 64 ] || [[ "$trained_attention_expected_adapter_config_sha256" == *[!0-9a-f]* ]]; then exit 4; fi
  if [ "${#trained_attention_expected_receipt_sha256}" -ne 64 ] || [[ "$trained_attention_expected_receipt_sha256" == *[!0-9a-f]* ]]; then exit 4; fi
  trained_attention_args=(
    --trained-attention-checkpoint "$trained_attention_checkpoint"
    --expected-trained-attention-step "$trained_attention_expected_step"
    --expected-trained-attention-objective "$trained_attention_expected_objective"
    --expected-trained-attention-route-operator "$trained_attention_expected_route_operator"
    --expected-trained-attention-adapter-sha256 "$trained_attention_expected_adapter_sha256"
    --expected-trained-attention-adapter-config-sha256 "$trained_attention_expected_adapter_config_sha256"
    --expected-trained-attention-receipt-sha256 "$trained_attention_expected_receipt_sha256"
  )
elif [ -n "$trained_attention_expected_step$trained_attention_expected_objective$trained_attention_expected_route_operator$trained_attention_expected_adapter_sha256$trained_attention_expected_adapter_config_sha256$trained_attention_expected_receipt_sha256" ]; then
  echo "trained attention expectations require a checkpoint" >&2
  exit 4
fi
if [ "$initial_noise_proposal_mode" != keyed_only ] || [ "$anchor_state_mode" = native_t2v_trajectory ]; then
  test -f "$anchor_initial_gaussian"
  anchor_initial_noise_args=(
    --anchor-initial-gaussian "$anchor_initial_gaussian"
    --expected-anchor-initial-gaussian-sha256 "$expected_anchor_initial_gaussian_sha256"
    --expected-anchor-initial-gaussian-raw-sha256 "$expected_anchor_initial_gaussian_raw_sha256"
  )
fi
test ! -e "$output"
test ! -e "$output.receipt.json"
mkdir -p "$output_root"

event_ordinal="${EVENT_ORDINAL_OVERRIDE:-1}"
case "$event_ordinal" in 0|1|2|3|4|5|6|7) ;; *) echo "EVENT_ORDINAL_OVERRIDE must be 0..7" >&2; exit 2 ;; esac
action="$(jq -r ".events[$event_ordinal].action" "$manifest")"
constraints="$(jq -r ".events[$event_ordinal].constraints" "$manifest")"
setup="$(jq -r ".events[$event_ordinal].variants[0].setup" "$manifest")"
instruction="Use the source video as the sole authority for identity, appearance, clothing, object instances, background, lighting, framing, camera and initial state. Frame 0 must retain the original source state; do not pre-apply the requested endpoint. Perform only this temporal edit: $action $constraints The edit must be one continuous 81-frame video at 25 fps and must not introduce appearance changes as a substitute for the requested action."
source_caption="${SOURCE_CAPTION_OVERRIDE:-A child in a pale blue outfit walks away from the camera along a stone path in a landscaped garden.}"
target_caption="${TARGET_CAPTION_OVERRIDE:-The same child in the same pale blue outfit and the same garden stops walking, crouches down beside a stone, reaches to grasp the stone, lifts it, stands back up, and holds the stone.}"
if [ "${ANCHOR_GENERIC_PROMPT:-0}" = 1 ]; then
  anchor_caption="A child stops, crouches, reaches to one stone, grasps it with visible contact, lifts the same stone, stands up, and visibly holds it."
  anchor_noop_caption="A child remains standing beside stones and does not crouch, reach, grasp, lift, or hold any stone."
else
  anchor_caption="${ANCHOR_CAPTION_OVERRIDE:-$setup $action $constraints}"
  anchor_noop_caption="${ANCHOR_NOOP_CAPTION_OVERRIDE:-$setup The child remains standing upright beside the stones and does not crouch, reach, pick up, lift, or hold any stone.}"
fi

scratch="/tmp/dynaedit-dynamic-static-$label-140846"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES="$devices"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
if [ "$trained_attention_expected_objective" = real_source_target_owned_routed_teacher_delta_v14r2 ]; then
  v14r2_modules=(
    action_preservation_decoded_eval_model_authority_v2
    anchor_cross_attention_transport anchor_qk_transport anchor_sga_anc_controller
    differential_sampler guided_source_aligned_controller
    exact_local_video_materializer_v1
    infer_anchor_sga_anc_event_v1 infer_anchor_sga_anc_trained_editor_decode_v1
    infer_lora
    infer_native_identity_generation_canary infer_source_aligned_controller_oracle
    infer_source_kv_carrier_oracle infer_source_value_residual_oracle
    self_generated_action_preservation_v2 source_aligned_controller
    source_kv_replay source_kv_route_batches source_value_residual
    train_lora tri_branch_unipc
  )
  for module in "${v14r2_modules[@]}"; do
    test -f "$dev/$module.py"
    test ! -L "$dev/$module.py"
  done
  test -f "$dev/tools/build_renderer_dataset.py"
  test ! -L "$dev/tools/build_renderer_dataset.py"
  test -f "$dev/tools/materialize_vae.py"
  test ! -L "$dev/tools/materialize_vae.py"
  export PYTHONPATH="$dev"
  export V14R2_METHOD_ROOT="$dev"
  V14R2_IMPORT_MODULES="$(IFS=:; echo "${v14r2_modules[*]}")"
  export V14R2_IMPORT_MODULES
  "$python_bin" -B -c 'import importlib, os, pathlib; root=pathlib.Path(os.environ["V14R2_METHOD_ROOT"]).resolve(); names=os.environ["V14R2_IMPORT_MODULES"].split(":"); modules=[importlib.import_module(name) for name in names]; bad=[str(pathlib.Path(module.__file__).resolve()) for module in modules if root not in pathlib.Path(module.__file__).resolve().parents]; (_ for _ in ()).throw(RuntimeError("v14r2 import escaped source tree: "+repr(bad))) if bad else None'
  "$python_bin" -B -c 'import os,pathlib,sys,types; root=pathlib.Path(os.environ["V14R2_METHOD_ROOT"]).resolve(); poison=types.ModuleType("tools"); poison.materialize_vae=object(); sys.modules["tools"]=poison; from exact_local_video_materializer_v1 import install_exact_local_video_materializer; materializer=install_exact_local_video_materializer(); builder=sys.modules["tools.build_renderer_dataset"]; expected={root/"tools/materialize_vae.py",root/"tools/build_renderer_dataset.py"}; actual={pathlib.Path(materializer.__file__).resolve(),pathlib.Path(builder.__file__).resolve()}; (_ for _ in ()).throw(RuntimeError("v14r3 exact video-tool closure differs")) if actual != expected else None'
else
  export PYTHONPATH="$dev:$runtime"
fi

exec "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$dev/infer_anchor_sga_anc_trained_editor_decode_v1.py" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$checkpoint" \
  --checkpoint-content-manifest "$checkpoint_manifest" \
  "${trained_attention_args[@]}" \
  "${trained_route_off_args[@]}" \
  --source-video "$source" \
  --expected-source-sha256 "$expected_source_sha256" \
  --anchor-video "$anchor" \
  --expected-anchor-sha256 "$expected_anchor_sha256" \
  "${extra_anchor_args[@]}" \
  "${anchor_initial_noise_args[@]}" \
  --instruction "$instruction" \
  --source-caption "$source_caption" \
  --target-caption "$target_caption" \
  --anchor-caption "$anchor_caption" \
  --anchor-noop-caption "$anchor_noop_caption" \
  --arm "$arm" \
  --transport "$transport" \
  --transport-strength "$strength" \
  --transport-steps "$transport_steps" \
  --blocks "$blocks" \
  --field-guidance raw_cfg \
  --field-model first_phase_caption_i2v \
  --source-cfg-scale "$source_cfg_scale" \
  --target-cfg-scale "$target_cfg_scale" \
  --sga-temperature 0.01 \
  --early-candidate-count "$early_candidate_count" \
  --initial-noise-proposal-mode "$initial_noise_proposal_mode" \
  --anchor-state-mode "$anchor_state_mode" \
  --anchor-cfg-scope "$anchor_cfg_scope" \
  --anchor-contrast-mode "$anchor_contrast_mode" \
  --anchor-sigma-cap "$anchor_sigma_cap" \
  --preservation-mode "$preservation_mode" \
  --preservation-keep-fraction "$preservation_keep_fraction" \
  --preservation-outside-scale "$preservation_outside_scale" \
  --preservation-dilation "$preservation_dilation" \
  --preservation-residual-fraction "$preservation_residual_fraction" \
  --preservation-object-identity-strength "$preservation_object_identity_strength" \
  --preservation-start-step "$preservation_start_step" \
  --preservation-ramp-steps "$preservation_ramp_steps" \
  --sga-score-mode "$sga_score_mode" \
  --anchor-candidate-mode "$anchor_candidate_mode" \
  --anchor-spatial-alignment "$anchor_spatial_alignment" \
  --event01-forced-role-proposal-index "$event01_forced_role_proposal_index" \
  "${initial_phase_args[@]}" \
  --output "$output"
