#!/usr/bin/env bash
set -euo pipefail

# Test whether action information missing from Q/K-only transport lives in the
# temporal value stream.  Only frame-0-relative sparse residuals are moved;
# absolute anchor V/content is never copied.
if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$1" in 0) devices=0,1,2,3 ;; 1) devices=4,5,6,7 ;; *) exit 2 ;; esac
case "$(hostname -s):$1" in
  auh7-1b-gpu-246:1)
    label=RAW_TRQKV_A025_EARLY8; transport=temporal_residual_qkv; strength=0.25 ;;
  auh7-1b-gpu-247:1)
    label=RAW_TRV_A050_EARLY8; transport=temporal_residual_v; strength=0.50 ;;
  *) echo "this canary uses GPU group 1 on Job 140846 nodes 246-247" >&2; exit 3 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
dev="$stage/anchor_qk_dev_v1"
runtime="$stage/source-be31323/methods/bernini_action_editing"
manifest="$stage/interaction_complex8_multianchor_authoring_v2.json"
source=/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/186d36f998a14e14/source.mp4
anchor="$stage/interaction_complex8_multianchor_v2_r1/e01_reach-grasp-lift-stone/v0/t2v.mp4"
output_root="$stage/anchor_value_residual_event01_v10"
output="$output_root/$label.mp4"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256

test -f "$dev/infer_anchor_sga_anc_event_v1.py"
test -f "$manifest"
test -f "$source"
test -f "$anchor"
test ! -e "$output"
test ! -e "$output.receipt.json"
mkdir -p "$output_root"

action="$(jq -r '.events[1].action' "$manifest")"
constraints="$(jq -r '.events[1].constraints' "$manifest")"
setup="$(jq -r '.events[1].variants[0].setup' "$manifest")"
instruction="Use the source video as the sole authority for identity, appearance, clothing, object instances, background, lighting, framing, camera and initial state. Frame 0 must retain the original source state; do not pre-apply the requested endpoint. Perform only this temporal edit: $action $constraints The edit must be one continuous 81-frame video at 25 fps and must not introduce appearance changes as a substitute for the requested action."
anchor_caption="$setup $action $constraints"

scratch="/tmp/anchor-value-residual-$label-140846"
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
  --anchor-caption "$anchor_caption" \
  --arm AQK_ANC1 \
  --transport "$transport" \
  --transport-strength "$strength" \
  --transport-steps 8 \
  --blocks 8-21 \
  --field-guidance raw_conditional \
  --no-initial-phase-clamp \
  --output "$output"
