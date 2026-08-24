#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 EVENT(0|2|4|7) ARM(frozen|no_anchor|action_noop|dynamic_static|hybrid) SLOT(0|1)" >&2
  exit 2
fi
event="$1"
arm="$2"
slot="$3"
case "$(hostname -s)" in
  auh7-1b-gpu-226|auh7-1b-gpu-233|auh7-1b-gpu-268|auh7-1b-gpu-292|auh7-1b-gpu-293|auh7-1b-gpu-306|auh7-1b-gpu-315) ;;
  *) echo "decode is restricted to authorized existing allocations" >&2; exit 3 ;;
esac
case "$slot" in 0) gpu_group=0,1,2,3 ;; 1) gpu_group=4,5,6,7 ;; *) exit 4 ;; esac
case "$arm" in frozen|no_anchor|action_noop|dynamic_static|hybrid) ;; *) exit 5 ;; esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
bridge="${ONLINE_ANCHOR_DECODE_DEV_OVERRIDE:-$stage/online_anchor_trained_lora_bridge_v1}"
bridge_runner="${ONLINE_ANCHOR_DECODE_BRIDGE_RUNNER:-$bridge/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh}"
train_root="$stage/online_anchor_attention_training_v1"
source_root=/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos
anchor_root="$stage/interaction_complex8_multianchor_v2_r1"

case "$event" in
  0)
    slug=pour-liquid-into-cup; iid=2f183dbf9e7a4d2e
    source_sha=888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de
    anchor_dir="$anchor_root/e00_pour-liquid-into-cup"
    anchor_sha=e485b2963d5ba191d8fc98158b7ab100bd1bd7584735221077c014ff0c3a19aa
    gaussian_sha=e4828f4594caf65b45a937a271847dcf621a75e4f6862269a28216759ff087b7
    gaussian_raw=38438c769db8a539ed84902e8493d66ab257144a97be9f4f57dfccb80880f832
    source_caption="An East Asian woman in a black floral blouse pours amber tea from a white bowl into a clear glass pitcher on a wooden tea table while a small white cup sits nearby."
    target_caption="The same woman in the same black floral blouse at the same table holds the same original glass pitcher and pours a continuous amber stream from its spout into the same original white cup, fills that cup, then returns the same pitcher upright. The white bowl remains unchanged and is not used as the pouring vessel."
    noop="The woman keeps the pitcher and cup in their initial state and does not lift or tilt the pitcher, pour a stream into the cup, fill the cup, or return a completed pour."
    ;;
  2)
    slug=twist-pull-mushroom; iid=10ed90644f81461d
    source_sha=63dd620627e18d0f6836058bf725ffba7ab1f9b0b455e784c6a71dc68a8de46c
    anchor_dir="$anchor_root/e02_twist-pull-mushroom"
    anchor_sha=a1076cbe83c9dae4a4fddc25f73077288f6a3240324fd2d5e1854aa842b07b63
    gaussian_sha=f94e34960f738e80558f4531dacf171756a21bb1ed30c5a96f1a6fe08a178bc9
    gaussian_raw=9e0ace1b100fd24c52d23afca3820fafe17d5ab77a02c26caa2b56fb33de7195
    source_caption="Two light-skinned hands touch one red-capped mushroom that remains rooted in dark forest soil among pine needles and leaves."
    target_caption="The same hands grasp the lower stem of the same red mushroom, twist and pull it free with a small dirt clump, lift it above the newly empty hole, and hold it there."
    noop="The hand stays near the rooted mushroom without grasping its stem, twisting, pulling, detaching, lifting, or leaving an empty hole."
    ;;
  4)
    slug=close-door-then-drawer; iid=12eba2f9c15f4d3f
    source_sha=d699a8d5e35a57f09ae4ba5fc5124e733be9ed18a2bddb2ee90a1ba0232c53f5
    anchor_dir="$anchor_root/e04_close-door-then-drawer"
    anchor_sha=c6d6a4e2835972609fcde8a8fbc2357eb36396f4a54aef7366adf809d6593f5e
    gaussian_sha=f7259d6af87616392b7de251cdc08a396bf53aae0f9207fe53042a5c06033201
    gaussian_raw=b2624ed648c39f73543c9e9ef8c287c54c6b1971934b0cb82b8f0715dd39d895
    source_caption="A light-skinned hand is at a honey-colored wooden cabinet inside a blue camper van; the lower hinged door is open and the separate upper drawer is halfway extended while the fixed camera faces the cabinet."
    target_caption="The same camper-van cabinet is operated by one hand: the same lower hinged door closes flush first, then the separate upper drawer is pushed fully inward and both closed states remain stable."
    noop="The same light-skinned hand remains present at the same camper-van cabinet but does not move either component; the lower hinged door remains open and the separate upper drawer remains halfway extended throughout."
    ;;
  7)
    slug=players-contact-then-separate; iid=a023388fb2374e44
    source_sha=1164531fd34d3d1273d56930aed139eb1a5d8db708ac3cdc4f7434abc0080799
    anchor_dir="$anchor_root/e07_players-contact-then-separate"
    anchor_sha=7b128ed47a7f6122d40a12711cf31535e39bcd5b92ce97d031cc2ff49424f4fc
    gaussian_sha=e8732371b147e5a63733a7d62c5ee4cfb97511e022164cc19ee2f52700d1c8b9
    gaussian_raw=b8cd0160471c484a427742a225d5e3713cc62bdbd26ed653a6605f43c49a55b9
    source_caption="Two male soccer players, one in blue-and-black and one in white, run shoulder to shoulder in direct contact toward a ball on a stadium field."
    target_caption="The same two players make one clear forearm push-off, separate from contact into a visible gap, and continue forward on distinct paths while the same ball stays ahead."
    noop="The two players continue running side by side in direct contact and do not retract the contacting arm, push off, separate into a gap, or take distinct paths."
    ;;
  *) exit 6 ;;
esac

checkpoint=""
decode_experiment="${ONLINE_ANCHOR_DECODE_EXPERIMENT:-}"
decode_step="${ONLINE_ANCHOR_DECODE_STEP:-8}"
output_experiment="${ONLINE_ANCHOR_OUTPUT_EXPERIMENT:-$decode_experiment}"
transport_steps="${ONLINE_ANCHOR_DECODE_TRANSPORT_STEPS:-40}"
cfg_scope="${ONLINE_ANCHOR_DECODE_CFG_SCOPE:-shared}"
anchor_state_mode="${ONLINE_ANCHOR_DECODE_STATE_MODE:-clean_noised}"
preservation_mode="${ONLINE_ANCHOR_DECODE_PRESERVATION_MODE:-none}"
sga_score_mode="${ONLINE_ANCHOR_DECODE_SGA_SCORE_MODE:-global_source_cosine}"
allow_trained_route_off_control="${ONLINE_ANCHOR_ALLOW_TRAINED_ROUTE_OFF_CONTROL:-0}"
expected_training_objective="${ONLINE_ANCHOR_EXPECTED_TRAINING_OBJECTIVE:-}"
expected_route_operator="${ONLINE_ANCHOR_EXPECTED_TRAINING_ROUTE_OPERATOR:-}"
decode_transport="${ONLINE_ANCHOR_DECODE_TRANSPORT:-temporal_contrast_cross_attn_output}"
case "$allow_trained_route_off_control" in 0|1) ;; *) exit 14 ;; esac
case "$transport_steps" in *[!0-9]*|'') exit 9 ;; esac
if (( transport_steps < 0 || transport_steps > 40 )); then exit 9; fi
if (( transport_steps == 0 )) && [ "$allow_trained_route_off_control" != 1 ]; then exit 9; fi
if (( transport_steps > 0 )) && [ "$allow_trained_route_off_control" != 0 ]; then exit 9; fi
case "$cfg_scope" in shared|target_conditional_only) ;; *) exit 10 ;; esac
case "$anchor_state_mode" in clean_noised|native_t2v_trajectory) ;; *) exit 11 ;; esac
case "$preservation_mode" in none|source_motion_support|source_motion_support_snapshot_residual) ;; *) exit 12 ;; esac
case "$sga_score_mode" in global_source_cosine|background_source_cosine|background_plus_anchor_action_002|background_trust_anchor_action_003) ;; *) exit 13 ;; esac
if [ -n "$decode_experiment" ]; then
  case "$decode_experiment" in *[!A-Za-z0-9_.-]*|'') exit 7 ;; esac
  case "$output_experiment" in *[!A-Za-z0-9_.-]*|'') exit 7 ;; esac
  case "$decode_step" in *[!0-9]*|'') exit 8 ;; esac
  checkpoint="$train_root/train_${decode_experiment}/checkpoint-$(printf '%08d' "$decode_step")"
  test -f "$checkpoint/receipt.json"
elif [ "$arm" != frozen ]; then
  checkpoint="$train_root/train_${arm}_crossattn_r256_micro2_s8_v1/checkpoint-00000008"
  test -f "$checkpoint/receipt.json"
  expected_training_objective="${expected_training_objective:-target_fm}"
  expected_route_operator="${expected_route_operator:-cross_sparse}"
fi
if [ "$allow_trained_route_off_control" = 1 ] && [ -z "$checkpoint" ]; then
  echo "trained route-off control requires a trained checkpoint" >&2
  exit 14
fi
expected_step=""
expected_adapter_sha256=""
expected_config_sha256=""
expected_receipt_sha256=""
if [ -n "$checkpoint" ]; then
  if [ -z "$expected_training_objective" ] || [ -z "$expected_route_operator" ]; then
    echo "trained decode requires explicit expected training objective and route" >&2
    exit 14
  fi
  case "$expected_training_objective" in target_fm|paired_delta_fm|real_source_teacher_delta|real_source_routed_teacher_delta|real_source_target_owned_routed_teacher_delta_v14r2) ;; *) exit 15 ;; esac
  case "$expected_route_operator" in cross_sparse|self_temporal_kernel|self_target_gated_kernel25|self_correspondence_kernel25|self_target_owned_temporal_kernel_v14r2|self_target_owned_activity_kernel10_v14r2|self_target_owned_activity_kernel25_v14r2) ;; *) exit 16 ;; esac
  if [ "$expected_training_objective" = real_source_target_owned_routed_teacher_delta_v14r2 ]; then
    case "$expected_route_operator" in
      self_target_owned_temporal_kernel_v14r2) required_transport=self_target_owned_temporal_kernel_attn_output_v14r2 ;;
      self_target_owned_activity_kernel10_v14r2) required_transport=self_target_owned_activity_kernel10_attn_output_v14r2 ;;
      self_target_owned_activity_kernel25_v14r2) required_transport=self_target_owned_activity_kernel25_attn_output_v14r2 ;;
      *) exit 16 ;;
    esac
    test "$decode_transport" = "$required_transport"
  fi
  expected_step="$decode_step"
  adapter_sha_line="$(sha256sum -- "$checkpoint/adapter/adapter_model.safetensors")"
  config_sha_line="$(sha256sum -- "$checkpoint/adapter/adapter_config.json")"
  receipt_sha_line="$(sha256sum -- "$checkpoint/receipt.json")"
  actual_adapter_sha256="${adapter_sha_line%% *}"
  actual_config_sha256="${config_sha_line%% *}"
  actual_receipt_sha256="${receipt_sha_line%% *}"
  if [ "$expected_training_objective" = real_source_target_owned_routed_teacher_delta_v14r2 ]; then
    : "${ONLINE_ANCHOR_EXPECTED_ADAPTER_SHA256:?v14r2 requires an external adapter SHA-256 binding}"
    : "${ONLINE_ANCHOR_EXPECTED_ADAPTER_CONFIG_SHA256:?v14r2 requires an external adapter-config SHA-256 binding}"
    : "${ONLINE_ANCHOR_EXPECTED_RECEIPT_SHA256:?v14r2 requires an external receipt SHA-256 binding}"
  fi
  expected_adapter_sha256="${ONLINE_ANCHOR_EXPECTED_ADAPTER_SHA256:-$actual_adapter_sha256}"
  expected_config_sha256="${ONLINE_ANCHOR_EXPECTED_ADAPTER_CONFIG_SHA256:-$actual_config_sha256}"
  expected_receipt_sha256="${ONLINE_ANCHOR_EXPECTED_RECEIPT_SHA256:-$actual_receipt_sha256}"
  if [ "${#expected_adapter_sha256}" -ne 64 ] || [[ "$expected_adapter_sha256" == *[!0-9a-f]* ]]; then exit 17; fi
  if [ "${#expected_config_sha256}" -ne 64 ] || [[ "$expected_config_sha256" == *[!0-9a-f]* ]]; then exit 17; fi
  if [ "${#expected_receipt_sha256}" -ne 64 ] || [[ "$expected_receipt_sha256" == *[!0-9a-f]* ]]; then exit 18; fi
fi
if [ -n "$decode_experiment" ]; then
  output_root="$train_root/dynaedit_fullgrid_v2/$output_experiment/step_$(printf '%08d' "$decode_step")/e$(printf '%02d' "$event")"
  label="E$(printf '%02d' "$event")_${slug}_${output_experiment}_S${decode_step}_ONLINE_ANCHOR_REAL_SGA_ANC"
else
  output_root="$train_root/dynaedit_decode_v1/e$(printf '%02d' "$event")"
  label="E$(printf '%02d' "$event")_${slug}_${arm}_S8_ONLINE_ANCHOR_REAL_SGA_ANC"
fi

export ROCR_VISIBLE_DEVICES="$gpu_group" HIP_VISIBLE_DEVICES="$gpu_group" CUDA_VISIBLE_DEVICES="$gpu_group"
export DEV_OVERRIDE="$bridge"
export LABEL_OVERRIDE="$label"
export OUTPUT_ROOT_OVERRIDE="$output_root"
export TRAINED_ATTENTION_CHECKPOINT="$checkpoint"
export TRAINED_ATTENTION_EXPECTED_STEP="$expected_step"
export TRAINED_ATTENTION_EXPECTED_OBJECTIVE="$expected_training_objective"
export TRAINED_ATTENTION_EXPECTED_ROUTE_OPERATOR="$expected_route_operator"
export TRAINED_ATTENTION_EXPECTED_ADAPTER_SHA256="$expected_adapter_sha256"
export TRAINED_ATTENTION_EXPECTED_ADAPTER_CONFIG_SHA256="$expected_config_sha256"
export TRAINED_ATTENTION_EXPECTED_RECEIPT_SHA256="$expected_receipt_sha256"
export ALLOW_TRAINED_ROUTE_OFF_CONTROL="$allow_trained_route_off_control"
export TRANSPORT_OVERRIDE="$decode_transport"
export STRENGTH_OVERRIDE="${ONLINE_ANCHOR_DECODE_STRENGTH:-.25}"
export TRANSPORT_STEPS_OVERRIDE="$transport_steps"
export BLOCKS_OVERRIDE=1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26,27,29
export EVENT_ORDINAL_OVERRIDE="$event"
export SOURCE_VIDEO_OVERRIDE="$source_root/$iid/source.mp4"
export EXPECTED_SOURCE_SHA256_OVERRIDE="$source_sha"
export ANCHOR_DIR_OVERRIDE="$anchor_dir"
export EXPECTED_ANCHOR_SHA256_OVERRIDE="$anchor_sha"
export EXPECTED_ANCHOR_INITIAL_GAUSSIAN_SHA256_OVERRIDE="$gaussian_sha"
export EXPECTED_ANCHOR_INITIAL_GAUSSIAN_RAW_SHA256_OVERRIDE="$gaussian_raw"
export SOURCE_CAPTION_OVERRIDE="$source_caption"
export TARGET_CAPTION_OVERRIDE="$target_caption"
export ANCHOR_NOOP_CAPTION_OVERRIDE="$noop"
export ANCHOR_STATE_MODE="$anchor_state_mode"
export INITIAL_NOISE_PROPOSAL_MODE=keyed_only
export INITIAL_PHASE_CLAMP=1
export ANCHOR_CFG_SCOPE_OVERRIDE="$cfg_scope"
export SOURCE_CFG_SCALE_OVERRIDE="${ONLINE_ANCHOR_DECODE_SOURCE_CFG_SCALE:-4.5}"
export TARGET_CFG_SCALE_OVERRIDE="${ONLINE_ANCHOR_DECODE_TARGET_CFG_SCALE:-4.5}"
export ANCHOR_CONTRAST_MODE=caption_noop_same_video
export ANCHOR_SIGMA_CAP=1.0
export PRESERVATION_MODE="$preservation_mode"
export PRESERVATION_KEEP_FRACTION="${ONLINE_ANCHOR_DECODE_PRESERVATION_KEEP_FRACTION:-0.20}"
export PRESERVATION_OUTSIDE_SCALE="${ONLINE_ANCHOR_DECODE_PRESERVATION_OUTSIDE_SCALE:-0.0}"
export PRESERVATION_RESIDUAL_FRACTION="${ONLINE_ANCHOR_DECODE_PRESERVATION_RESIDUAL_FRACTION:-0.0}"
export PRESERVATION_OBJECT_IDENTITY_STRENGTH=0.0
export SGA_SCORE_MODE="$sga_score_mode"
export EARLY_CANDIDATE_COUNT=5
export ARM_OVERRIDE=AQK_SGA5

case "$bridge" in /*) ;; *) echo "decode dev override must be absolute" >&2; exit 19 ;; esac
case "$bridge_runner" in /*) ;; *) echo "decode bridge runner must be absolute" >&2; exit 20 ;; esac
test -d "$bridge"
test ! -L "$bridge"
test -f "$bridge_runner"
test ! -L "$bridge_runner"

exec bash "$bridge_runner" "$slot"
