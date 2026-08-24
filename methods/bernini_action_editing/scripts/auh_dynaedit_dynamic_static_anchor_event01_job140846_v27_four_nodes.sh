#!/usr/bin/env bash
set -euo pipefail

# Round-27 four-node payload.  It contrasts the real dynamic anchor with its
# phase-0-static counterpart under identical caption/noise.  One SP4 model is
# run per node because Job 140846 grants only 64 GiB host RAM per node.
if [ "$#" -ne 0 ]; then
  echo "usage: $0" >&2
  exit 2
fi

devices=0,1,2,3
case "$(hostname -s)" in
  auh7-1b-gpu-246)
    label=DYNSTATIC_VEL_A010_EARLY8
    transport=temporal_contrast_velocity
    strength=0.10
    blocks=8-13
    ;;
  auh7-1b-gpu-248)
    label=DYNSTATIC_VEL_A025_EARLY8
    transport=temporal_contrast_velocity
    strength=0.25
    blocks=8-13
    ;;
  auh7-1b-gpu-279)
    label=DYNSTATIC_SELFATT_A003_EARLY8_B8_13
    transport=temporal_contrast_attn_output
    strength=0.03
    blocks=8-13
    ;;
  auh7-1b-gpu-247)
    label=DYNSTATIC_CROSSATT_A003_EARLY8_B4_9
    transport=temporal_contrast_cross_attn_output
    strength=0.03
    blocks=4-9
    ;;
  *)
    echo "this payload is restricted to Job 140846 nodes 246-248 and 279" >&2
    exit 3
    ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
dev="$stage/anchor_qk_dev_v1"
runtime="$stage/source-be31323/methods/bernini_action_editing"
manifest="$stage/interaction_complex8_multianchor_authoring_v2.json"
source=/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/186d36f998a14e14/source.mp4
anchor="$stage/interaction_complex8_multianchor_v2_r1/e01_reach-grasp-lift-stone/v0/t2v.mp4"
output_root="$stage/dynaedit_dynamic_static_anchor_event01_v27"
output="$output_root/$label.mp4"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256

require_sha256() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(sha256sum "$path" | awk '{print $1}')"
  if [ "$observed" != "$expected" ]; then
    echo "sha256 mismatch for $path: $observed" >&2
    exit 4
  fi
}

require_sha256 "$dev/anchor_cross_attention_transport.py" cbbbc490711f4f1131a701d042309b024fd5b3922f2d88e39e35bff7d39396f6
require_sha256 "$dev/anchor_sga_anc_controller.py" 9bc48c32865b49d6ca182eb7a192066313382f758be26085eeb803c8afa61a03
require_sha256 "$dev/infer_anchor_sga_anc_event_v1.py" e391b8cad4febd316cb8cdf6e9e5983def398345dafe11df745c6cdb75ee0b9d
require_sha256 "$manifest" 767aa8e0502f247c3ab576db4c40a132344295e54aef4170e728bc3ff71cafc5
test -f "$source"
test -f "$anchor"
test ! -e "$output"
test ! -e "$output.receipt.json"
mkdir -p "$output_root"

action="$(jq -r '.events[1].action' "$manifest")"
constraints="$(jq -r '.events[1].constraints' "$manifest")"
setup="$(jq -r '.events[1].variants[0].setup' "$manifest")"
instruction="Use the source video as the sole authority for identity, appearance, clothing, object instances, background, lighting, framing, camera and initial state. Frame 0 must retain the original source state; do not pre-apply the requested endpoint. Perform only this temporal edit: $action $constraints The edit must be one continuous 81-frame video at 25 fps and must not introduce appearance changes as a substitute for the requested action."
source_caption="A child in a pale blue outfit walks away from the camera along a stone path in a landscaped garden."
target_caption="The same child in the same pale blue outfit and the same garden stops walking, crouches down beside a stone, reaches to grasp the stone, lifts it, stands back up, and holds the stone."
anchor_caption="$setup $action $constraints"
anchor_noop_caption="$setup The child remains standing upright beside the stones and does not crouch, reach, pick up, lift, or hold any stone."

scratch="/tmp/dynaedit-dynamic-static-$label-140846-four-nodes"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES="$devices"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
export PYTHONPATH="$dev:$runtime"

exec "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$dev/infer_anchor_sga_anc_event_v1.py" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$checkpoint" \
  --checkpoint-content-manifest "$checkpoint_manifest" \
  --source-video "$source" \
  --expected-source-sha256 60618e5a988f3d8b4f48d4ae46bc7739032663a7f8805ee26d47b7d3c193af48 \
  --anchor-video "$anchor" \
  --expected-anchor-sha256 e0cdbab524bca22d2300ebc7c75723dc1a19f5eeabe619ddbddaa77f1960188e \
  --instruction "$instruction" \
  --source-caption "$source_caption" \
  --target-caption "$target_caption" \
  --anchor-caption "$anchor_caption" \
  --anchor-noop-caption "$anchor_noop_caption" \
  --arm AQK_SGA5 \
  --transport "$transport" \
  --transport-strength "$strength" \
  --transport-steps 8 \
  --blocks "$blocks" \
  --field-guidance raw_cfg \
  --field-model first_phase_caption_i2v \
  --source-cfg-scale 4.5 \
  --target-cfg-scale 8.5 \
  --sga-temperature 0.01 \
  --anchor-cfg-scope target_conditional_only \
  --anchor-contrast-mode dynamic_static_same_caption \
  --no-initial-phase-clamp \
  --output "$output"
