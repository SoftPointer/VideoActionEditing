#!/usr/bin/env bash
set -euo pipefail

# One SP4 process owns one trained arm and serially decodes checkpoint exposure
# on one action-hard source plus three real original-source edits.
if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$1" in 0) devices=0,1,2,3 ;; 1) devices=4,5,6,7 ;; *) echo "GPU_GROUP must be 0 or 1" >&2; exit 2 ;; esac
case "$(hostname -s):$1" in
  auh7-1b-gpu-246:0) arm=train_bank9_full_lr3e4_s90_r2 ;;
  auh7-1b-gpu-246:1) arm=train_bank9_full_lr2e4_s90_r2 ;;
  auh7-1b-gpu-247:0) arm=train_bank9_full_lr1p5e4_s90_r2 ;;
  auh7-1b-gpu-247:1) arm=train_bank9_full_lr1e4_s90_r2 ;;
  auh7-1b-gpu-248:0) arm=train_bank9_mid250_750_lr3e4_s90_r2 ;;
  auh7-1b-gpu-248:1) arm=train_bank9_mid250_750_lr2e4_s90_r2 ;;
  auh7-1b-gpu-279:0) arm=train_bank9_noisy500_1000_lr2e4_s90_r2 ;;
  auh7-1b-gpu-279:1) arm=train_bank9_clean0_500_lr2e4_s90_r2 ;;
  *) echo "forbidden node/group: primary decode is restricted to Job 140846" >&2; exit 3 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
source_tree="$root/stage1/source-a2805a9-overlay"
runtime_tree="$root/stage1/source-be31323"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
media_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_endpoint_consensus_release_11f3ed1/source/md/action_editing/20260815_reward/model_gain_135096/review/media
pair_root="$root/stage1/same_video_pairs_r2"
test -f "$root/stage1/$arm/TRAINING_COMPLETE"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES="$devices"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONPATH="$source_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"

decode_one() {
  local step="$1" case_id="$2" iid="$3" source_video="$4" instruction="$5"
  local checkpoint="$root/stage1/$arm/checkpoint-$(printf '%08d' "$step")"
  local output_dir="$root/stage1/infer_bank9_v2_primary/$arm/s$(printf '%03d' "$step")/$case_id"
  local output="$output_dir/output.mp4"
  local scratch="/tmp/bank9-primary-${arm}-s${step}-${case_id}-g$1-140846"
  test -f "$checkpoint/adapter_model.safetensors"
  test -f "$source_video"
  test -f "$root/flows/$iid.safetensors"
  test ! -e "$output_dir"
  mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton" "$output_dir"
  export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
  export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
  "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$source_tree/methods/bernini_action_editing/infer_same_video_dense_flow_adapter_v1.py" \
    --dense-flow-checkpoint "$checkpoint" \
    --dense-flow-scale 1.0 \
    --flow-bundle "$root/flows/$iid.safetensors" \
    --bernini-root "$bernini_root" \
    --veomni-root "$veomni_root" \
    --checkpoint "$base_checkpoint" \
    --source-video "$source_video" \
    --instruction "$instruction" \
    --output "$output" \
    --num-inference-steps 40 \
    --seed 2026081801 \
    --source-onset-policy hard1_every_step \
    --method-source-revision a2805a9022e6b1c799635fba3478a7647e22d696 \
    --method-source-archive-sha256 e6ad2321f5ea1e0ea4cc040bc0b7ff28731d19de21a19873d4cc1ec1f8cf9508
}

shepherd_instruction='A fixed camera shows a single black-and-tan German shepherd standing on a leash in a grassy field, looking upward toward the viewer, with clumps of shed fur on the ground and subtle ear and breeze movement. The main German shepherd bends its hind legs, lowers its hips to the ground, settles into a stable sit facing the camera, and holds that seated pose. The shot stays continuous, the illumination remains stable, and the final frame is temporally coherent.'
pink_instruction='A static camera shows a single young woman with long brown hair kneeling on one knee and looking at the camera against a solid warm yellow background, wearing a black jacket, pink top, cream trousers, and breathing gently. The main person shifts weight onto both feet, rises smoothly from kneeling, straightens both legs and the torso, and holds a stable upright standing pose. The shot stays continuous, the illumination remains stable, and the final frame is temporally coherent.'
plaid_instruction='A static camera shows a single young woman in a low crouch on a city sidewalk beside a modern dark-glass building with horizontal metal bars, looking back over her shoulder, with tied dark hair, a black sports bra, grey trousers, and a blue-green plaid shirt in harsh sunlight and deep shadow. The main person presses through both feet, rises smoothly from the low crouch, straightens the legs and torso, faces the viewer, and holds upright. The shot stays continuous, the illumination remains stable, and the final frame is temporally coherent.'

for step in 20 40 80 90; do
  decode_one "$step" shepherd_hard 841b5e0080a1441d "$pair_root/841b5e0080a1441d/source_incomplete.mp4" "$shepherd_instruction"
  decode_one "$step" shepherd_real 841b5e0080a1441d "$media_root/841b5e0080a1441d/source.mp4" "$shepherd_instruction"
  decode_one "$step" pink_real a35b590961d24694 "$media_root/a35b590961d24694/source.mp4" "$pink_instruction"
  decode_one "$step" plaid_real a66e6818e4144928 "$media_root/a66e6818e4144928/source.mp4" "$plaid_instruction"
done
