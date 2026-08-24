#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh7-1b-gpu-246) lr_tag=lr3e4; sr_tag=sr0 ;;
  auh7-1b-gpu-247) lr_tag=lr1p5e4; sr_tag=sr0 ;;
  auh7-1b-gpu-248) lr_tag=lr3e4; sr_tag=sr25 ;;
  auh7-1b-gpu-279) lr_tag=lr1p5e4; sr_tag=sr25 ;;
  *) echo "forbidden node: replay decode is restricted to Job 140846 nodes" >&2; exit 3 ;;
esac
if [ "$#" -ne 1 ]; then
  echo "usage: $0 GPU_GROUP" >&2
  exit 2
fi
case "$1" in
  0) devices=0,1,2,3; repeat_tag=r1x2 ;;
  1) devices=4,5,6,7; repeat_tag=r1x3 ;;
  *) echo "GPU_GROUP must be 0 or 1" >&2; exit 2 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
source_tree="$root/stage1/source-a2805a9-overlay"
runtime_tree="$root/stage1/source-be31323"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
pair_root="$root/stage1/same_video_pairs_r2"
arm="train_hardreplay_${repeat_tag}_${lr_tag}_${sr_tag}_s80_r1"
test -f "$root/stage1/$arm/TRAINING_COMPLETE"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES="$devices"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONPATH="$source_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"

decode_one() {
  local step="$1"
  local iid="$2"
  local instruction="$3"
  local checkpoint="$root/stage1/$arm/checkpoint-$(printf '%08d' "$step")"
  local output_dir="$root/stage1/infer_hardreplay_v1/${arm}_s$(printf '%03d' "$step")/${iid}_source_noop"
  local output="$output_dir/output.mp4"
  local scratch="/tmp/infer-${arm}-s${step}-${iid}-g$4-140846"
  test -f "$checkpoint/adapter_model.safetensors"
  if [ -f "$output" ]; then
    echo "already complete: $output"
    return
  fi
  test ! -e "$output_dir"
  mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton" "$output_dir"
  export MIOPEN_USER_DB_PATH="$scratch/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
  export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions"
  export TRITON_CACHE_DIR="$scratch/triton"
  "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$source_tree/methods/bernini_action_editing/infer_same_video_dense_flow_adapter_v1.py" \
    --dense-flow-checkpoint "$checkpoint" \
    --dense-flow-scale 1.0 \
    --flow-bundle "$root/flows/$iid.safetensors" \
    --bernini-root "$bernini_root" \
    --veomni-root "$veomni_root" \
    --checkpoint "$base_checkpoint" \
    --source-video "$pair_root/$iid/source_noop.mp4" \
    --instruction "$instruction" \
    --output "$output" \
    --num-inference-steps 40 \
    --seed 2026081701 \
    --source-onset-policy hard1_every_step \
    --method-source-revision a2805a9022e6b1c799635fba3478a7647e22d696 \
    --method-source-archive-sha256 e6ad2321f5ea1e0ea4cc040bc0b7ff28731d19de21a19873d4cc1ec1f8cf9508
}

dog_instruction='A fixed camera shows a single black-and-tan German shepherd standing on a leash in a grassy field, looking upward toward the viewer, with clumps of shed fur on the ground and subtle ear and breeze movement. The main German shepherd bends its hind legs, lowers its hips to the ground, settles into a stable sit facing the camera, and holds that seated pose. The shot stays continuous, the illumination remains stable, and the final frame is temporally coherent.'
woman_instruction='A static camera shows a single young woman with long brown hair kneeling on one knee and looking at the camera against a solid warm yellow background, wearing a black jacket, pink top, cream trousers, and breathing gently. The main person shifts weight onto both feet, rises smoothly from kneeling, straightens both legs and the torso, and holds a stable upright standing pose. The shot stays continuous, the illumination remains stable, and the final frame is temporally coherent.'

for step in 20 40 80; do
  decode_one "$step" 841b5e0080a1441d "$dog_instruction" "$1"
  decode_one "$step" a35b590961d24694 "$woman_instruction" "$1"
done
