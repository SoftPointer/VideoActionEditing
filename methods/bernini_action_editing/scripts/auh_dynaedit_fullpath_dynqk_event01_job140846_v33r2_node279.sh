#!/usr/bin/env bash
set -euo pipefail

# Round-33 R2 single-variable canary. Its method parameters match the
# Round-32 dynamic-minus-static Q/K arm except transport_steps=40;
# the same 1.5% residual preservation basin is retained. Only rank zero
# loads/encodes UMT5. This is inference-only.
if [ "$#" -ne 0 ]; then
  echo "usage: $0" >&2
  exit 2
fi

devices=0,1,2,3
case "$(hostname -s)" in
  auh7-1b-gpu-279)
    label=FULLPATH_DYNQK_A010_B4_9
    transport=temporal_contrast_qk
    strength=0.10
    blocks=4-9
    anchor_sigma_cap=0.8
    ;;
  *)
    echo "this payload is restricted to Job 140846 node 279" >&2
    exit 3
    ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
dev="$stage/anchor_sigma_cap_v33r3_4686c2f1_3f3390e2_3e7f8e44"
runtime="$stage/source-be31323/methods/bernini_action_editing"
manifest="$stage/interaction_complex8_multianchor_authoring_v2.json"
source=/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/186d36f998a14e14/source.mp4
anchor="$stage/interaction_complex8_multianchor_v2_r1/e01_reach-grasp-lift-stone/v0/t2v.mp4"
output_root="$stage/dynaedit_fullpath_dynqk_event01_v33r3_4686c2f1_3f3390e2_3e7f8e44"
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

require_sha256 "$dev/infer_anchor_sga_anc_event_v1.py" 4686c2f144f89d01693e67e6b46fe52c6a80799547f774ffd91c153218752935
require_sha256 "$dev/anchor_sga_anc_controller.py" 3f3390e28310e0e18194b1d4380c7e2383c4eae1784cf3b18fa2af92d54d36ed
require_sha256 "$dev/anchor_cross_attention_transport.py" cbbbc490711f4f1131a701d042309b024fd5b3922f2d88e39e35bff7d39396f6
require_sha256 "$dev/anchor_qk_transport.py" 3140281b04c99a7ed7a965c19f453fefeb8daf2401a1c63345c1a0ec37f87c45
require_sha256 "$dev/guided_source_aligned_controller.py" 3e7f8e449447c8cc0f2678da82b9e298d84d0b5b9f729281a5b19369cba7ddc6
require_sha256 "$runtime/differential_sampler.py" 16738e7bfa48d6b44dfc35fc395d55068e3794212baabaefa2b2876c8774916f
require_sha256 "$runtime/source_aligned_controller.py" e8601c82d1fcf7e4df11daa658b9f237e01eabc489f77a88610fcab6ad3cf4a8
require_sha256 "$runtime/source_kv_replay.py" 45b43426dc7825dbd61280154fc35161c60476ec5cb9e53bc0225f3809c759f3
require_sha256 "$runtime/infer_lora.py" c2e55a4ea41a21d0761e660ab630002b1bc569705e8c0bcafa1bc8c6c38ccc06
require_sha256 "$runtime/infer_native_identity_generation_canary.py" a60c37591c40206c6130185f1a2d2a7a8e473f5af4425205e268ae4a8b58f334
require_sha256 "$runtime/infer_source_aligned_controller_oracle.py" 9ae3a41e52f520f66ebcddba331b26837a5c8291426d13379eaa4c8a01a80e02
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

scratch="/tmp/dynaedit-fullpath-dynqk-v33r3-$label-140846-node279"
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
  --transport-steps 40 \
  --blocks "$blocks" \
  --field-guidance raw_cfg \
  --field-model first_phase_caption_i2v \
  --source-cfg-scale 4.5 \
  --target-cfg-scale 8.5 \
  --sga-temperature 0.01 \
  --anchor-cfg-scope target_conditional_only \
  --anchor-contrast-mode dynamic_static_same_caption \
  --anchor-sigma-cap "$anchor_sigma_cap" \
  --preservation-mode source_motion_support \
  --preservation-keep-fraction 0.10 \
  --preservation-outside-scale 0.0 \
  --preservation-dilation 1 \
  --preservation-residual-fraction 0.015 \
  --no-initial-phase-clamp \
  --output "$output"
