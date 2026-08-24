#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 0|2|4|7" >&2
  exit 2
fi

event=$1
host=$(hostname -s)
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
source_root=/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos
anchor_root="$stage/interaction_complex8_multianchor_v2_r1"

case "$host:$event" in
  auh7-1b-gpu-246:0)
    device_slot=0
    gpu_group=0,1,2,3
    slug=pour-liquid-into-cup
    iid=2f183dbf9e7a4d2e
    source_sha=888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de
    anchor_dir="$anchor_root/e00_pour-liquid-into-cup"
    anchor_sha=e485b2963d5ba191d8fc98158b7ab100bd1bd7584735221077c014ff0c3a19aa
    gaussian_sha=e4828f4594caf65b45a937a271847dcf621a75e4f6862269a28216759ff087b7
    gaussian_raw=38438c769db8a539ed84902e8493d66ab257144a97be9f4f57dfccb80880f832
    source_caption="An East Asian woman in a black floral blouse pours amber tea from a white bowl into a clear glass pitcher on a wooden tea table while a small white cup sits nearby."
    target_caption="The same woman in the same black floral blouse at the same table holds the same glass pitcher and pours a continuous amber stream from its spout into the same white cup, fills the cup, then returns the pitcher upright."
    noop="The woman keeps the pitcher and cup in their initial state and does not lift or tilt the pitcher, pour a stream into the cup, fill the cup, or return a completed pour."
    ;;
  auh7-1b-gpu-246:4)
    device_slot=1
    gpu_group=4,5,6,7
    slug=close-door-then-drawer
    iid=12eba2f9c15f4d3f
    source_sha=d699a8d5e35a57f09ae4ba5fc5124e733be9ed18a2bddb2ee90a1ba0232c53f5
    anchor_dir="$anchor_root/e04_close-door-then-drawer"
    anchor_sha=c6d6a4e2835972609fcde8a8fbc2357eb36396f4a54aef7366adf809d6593f5e
    gaussian_sha=f7259d6af87616392b7de251cdc08a396bf53aae0f9207fe53042a5c06033201
    gaussian_raw=b2624ed648c39f73543c9e9ef8c287c54c6b1971934b0cb82b8f0715dd39d895
    source_caption="A honey-colored wooden cabinet inside a blue camper van has its lower hinged door open beneath a drawer while the fixed camera faces the cabinet."
    target_caption="The same camper-van cabinet is operated by one hand: the same lower hinged door closes flush first, then the separate upper drawer is pushed fully inward and both closed states remain stable."
    noop="The cabinet remains in its initial state; no hand enters, the lower hinged door does not close, and the separate drawer does not move inward."
    ;;
  auh7-1b-gpu-247:2)
    device_slot=0
    gpu_group=0,1,2,3
    slug=twist-pull-mushroom
    iid=10ed90644f81461d
    source_sha=63dd620627e18d0f6836058bf725ffba7ab1f9b0b455e784c6a71dc68a8de46c
    anchor_dir="$anchor_root/e02_twist-pull-mushroom"
    anchor_sha=a1076cbe83c9dae4a4fddc25f73077288f6a3240324fd2d5e1854aa842b07b63
    gaussian_sha=f94e34960f738e80558f4531dacf171756a21bb1ed30c5a96f1a6fe08a178bc9
    gaussian_raw=9e0ace1b100fd24c52d23afca3820fafe17d5ab77a02c26caa2b56fb33de7195
    source_caption="Two light-skinned hands touch one red-capped mushroom that remains rooted in dark forest soil among pine needles and leaves."
    target_caption="The same hands grasp the lower stem of the same red mushroom, twist and pull it free with a small dirt clump, lift it above the newly empty hole, and hold it there."
    noop="The hand stays near the rooted mushroom without grasping its stem, twisting, pulling, detaching, lifting, or leaving an empty hole."
    ;;
  auh7-1b-gpu-247:7)
    device_slot=1
    gpu_group=4,5,6,7
    slug=players-contact-then-separate
    iid=a023388fb2374e44
    source_sha=1164531fd34d3d1273d56930aed139eb1a5d8db708ac3cdc4f7434abc0080799
    anchor_dir="$anchor_root/e07_players-contact-then-separate"
    anchor_sha=7b128ed47a7f6122d40a12711cf31535e39bcd5b92ce97d031cc2ff49424f4fc
    gaussian_sha=e8732371b147e5a63733a7d62c5ee4cfb97511e022164cc19ee2f52700d1c8b9
    gaussian_raw=b8cd0160471c484a427742a225d5e3713cc62bdbd26ed653a6605f43c49a55b9
    source_caption="Two male soccer players, one in blue-and-black and one in white, run shoulder to shoulder in direct contact toward a ball on a stadium field."
    target_caption="The same two players make one clear forearm push-off, separate from contact into a visible gap, and continue forward on distinct paths while the same ball stays ahead."
    noop="The two players continue running side by side in direct contact and do not retract the contacting arm, push off, separate into a gap, or take distinct paths."
    ;;
  *)
    echo "Round 89 assigns events 0/4 to node246 and 2/7 to node247" >&2
    exit 3
    ;;
esac

# The AUH MI210 Slurm plugin accounts separate four-GPU GRES slices but does
# not expose distinct physical IDs to ROCm for overlapping job steps. Bind the
# two simultaneous events on each node explicitly and let torch map this list
# back to four logical ranks.
export ROCR_VISIBLE_DEVICES="$gpu_group"
export HIP_VISIBLE_DEVICES="$gpu_group"
export CUDA_VISIBLE_DEVICES="$gpu_group"

source_tree="$stage/anchor_qk_symmetry_v1"
export DEV_OVERRIDE="$source_tree"
export LABEL_OVERRIDE="COMPLEX4_E${event}_${slug}_OBSERVER_P39_R8"
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_complex4_observer_p39_event_v89"
export TRANSPORT_OVERRIDE=action_noop_observer_attn_output
export STRENGTH_OVERRIDE=1.0
export TRANSPORT_STEPS_OVERRIDE=3
export BLOCKS_OVERRIDE=4-9
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
export ANCHOR_STATE_MODE=native_t2v_trajectory
export INITIAL_NOISE_PROPOSAL_MODE=keyed_only
export INITIAL_PHASE_CLAMP=1
export ANCHOR_CFG_SCOPE_OVERRIDE=shared
export ANCHOR_CONTRAST_MODE=caption_noop_same_video
export ANCHOR_SIGMA_CAP=1.0
export EVENT01_FORCED_ROLE_PROPOSAL_INDEX=-1
export PRESERVATION_MODE=source_motion_support
export PRESERVATION_KEEP_FRACTION=0.20
export PRESERVATION_OUTSIDE_SCALE=0.05
export PRESERVATION_DILATION=1
export PRESERVATION_RESIDUAL_FRACTION=0.015
export PRESERVATION_OBJECT_IDENTITY_STRENGTH=0.0
export PRESERVATION_START_STEP=39
export PRESERVATION_RAMP_STEPS=8
export SGA_SCORE_MODE=background_trust_anchor_envelope_003
export ARM_OVERRIDE=AQK_SGA5

exec "$source_tree/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" "$device_slot"
