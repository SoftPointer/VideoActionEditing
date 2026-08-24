#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 STEP synthetic|real" >&2
  exit 2
fi
step="$1"
case "$step" in 1|5|10|20|40|64) ;; *) echo "unsupported checkpoint step" >&2; exit 2 ;; esac
case "$2" in synthetic|real) case_id="$2" ;; *) echo "case must be synthetic or real" >&2; exit 2 ;; esac
case "$(hostname -s)" in
  auh7-1b-gpu-246) mode=matched ;;
  auh7-1b-gpu-247) mode=cross ;;
  *) echo "inference is restricted to Job 140846 nodes 246/247" >&2; exit 3 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
source_tree="$stage/source-crossappearance-motion-v1"
old_source_tree="$stage/source-816e892-queryflow"
runtime_tree="$stage/source-be31323"
release="$stage/complex8_crossappearance_motion_v1"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
checkpoint="$release/train_joint_${mode}_q12x20_all30_r256_micro1_s64_v3/checkpoint-$(printf '%08d' "$step")"
event_root="$stage/interaction_complex8_multianchor_v2_r1/e01_reach-grasp-lift-stone"
instruction="The child notices one marked stone, stops, bends both knees into a crouch, reaches one hand to that exact stone, closes the fingers around it with visible contact, lifts it from the ground, rises back to standing, and holds the same stone visibly in the hand. The vacated spot remains empty. Keep exactly one child and the original set of stones. Preserve identity, clothing, all other stones, path, plants, lighting, framing and camera. No cuts, teleportation, object replacement, duplication or extra limbs."
if [ "$case_id" = synthetic ]; then
  source="$event_root/v0/noop.mp4"
  flow="$release/flows_${mode}_exact/e01-v0.safetensors"
else
  source=/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/186d36f998a14e14/source.mp4
  flow="$release/flows_real_event01/v0.safetensors"
fi
output_dir="$release/infer/${mode}/s$(printf '%03d' "$step")/$case_id"
output="$output_dir/output.mp4"
test -f "$checkpoint/adapter_model.safetensors"
test -f "$checkpoint/adapter/adapter_model.safetensors"
test -f "$flow"
test -f "$source"
test ! -e "$output_dir"
mkdir -p "$output_dir"

scratch="/tmp/complex8-joint-infer-${mode}-s${step}-${case_id}-140846"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES=0,1,2,3
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
export PYTHONPATH="$source_tree/methods/bernini_action_editing:$old_source_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"
revision=$(tr -d '\n' < "$release/method-source.revision")
archive_sha=$(sha256sum "$release/method-source.tar" | awk '{print $1}')

exec "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$source_tree/methods/bernini_action_editing/infer_same_video_dense_flow_adapter_v1.py" \
  --dense-flow-checkpoint "$checkpoint" \
  --dense-flow-scale 1.0 \
  --flow-bundle "$flow" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$base_checkpoint" \
  --source-video "$source" \
  --instruction "$instruction" \
  --output "$output" \
  --num-inference-steps 40 \
  --seed 2026081819 \
  --source-onset-policy hard1_every_step \
  --method-source-revision "$revision" \
  --method-source-archive-sha256 "$archive_sha"
