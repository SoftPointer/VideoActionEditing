#!/usr/bin/env bash
set -euo pipefail

# Eight matched SP4 arms inside the existing four-node Job 140846 allocation.
# Node selects the model checkpoint; GPU group selects noop vs incomplete input.
case "$(hostname -s)" in
  auh7-1b-gpu-246)
    model_tag=frozen
    dense_flow_checkpoint=
    ;;
  auh7-1b-gpu-247)
    model_tag=row0_motion_s160
    dense_flow_checkpoint=train_row0_noop_varying_out3e4_cond3e4_s160_r1/checkpoint-00000160
    ;;
  auh7-1b-gpu-248)
    model_tag=all4_mixed_s080
    dense_flow_checkpoint=train_all4_mixed_varying_out3e4_cond3e4_s320_r1/checkpoint-00000080
    ;;
  auh7-1b-gpu-279)
    model_tag=all4_noop_s080
    dense_flow_checkpoint=train_all4_noop_varying_out3e4_cond3e4_s320_r1/checkpoint-00000080
    ;;
  *) echo "forbidden node: hard-input audit is restricted to Job 140846 nodes" >&2; exit 3 ;;
esac

if [ "$#" -ne 1 ]; then
  echo "usage: $0 GPU_GROUP" >&2
  exit 2
fi
case "$1" in
  0) devices=0,1,2,3; source_variant=source_noop ;;
  1) devices=4,5,6,7; source_variant=source_incomplete ;;
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
if [ "$model_tag" != frozen ]; then
  dense_flow_checkpoint="$root/stage1/$dense_flow_checkpoint"
  test -f "$dense_flow_checkpoint/adapter_model.safetensors"
fi

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES="$devices"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONPATH="$source_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"

decode_one() {
  local iid="$1"
  local instruction="$2"
  local output_dir="$root/stage1/infer_same_video_hard_v1/${iid}/${source_variant}/${model_tag}"
  local output="$output_dir/output.mp4"
  local scratch="/tmp/hard-v1-${iid}-${source_variant}-${model_tag}-g$3-140846"
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
  if [ "$model_tag" = frozen ]; then
    "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
      "$source_tree/methods/bernini_action_editing/infer_lora.py" \
      --bernini-root "$bernini_root" \
      --veomni-root "$veomni_root" \
      --checkpoint "$base_checkpoint" \
      --base-only \
      --source-video "$pair_root/$iid/${source_variant}.mp4" \
      --instruction "$instruction" \
      --output "$output" \
      --num-inference-steps 40 \
      --seed 2026081701 \
      --source-onset-policy hard1_every_step \
      --expected-bernini-commit 2d2b4591ac053ec25c6371b01a5a6746679e5793 \
      --expected-veomni-commit f90b3dc6fbb0ce693745223cc7a94064123dbf4d \
      --expected-checkpoint-tree-sha256 6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca \
      --method-source-revision a2805a9022e6b1c799635fba3478a7647e22d696 \
      --method-source-archive-sha256 e6ad2321f5ea1e0ea4cc040bc0b7ff28731d19de21a19873d4cc1ec1f8cf9508
  else
    "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
      "$source_tree/methods/bernini_action_editing/infer_same_video_dense_flow_adapter_v1.py" \
      --dense-flow-checkpoint "$dense_flow_checkpoint" \
      --dense-flow-scale 1.0 \
      --flow-bundle "$root/flows/$iid.safetensors" \
      --bernini-root "$bernini_root" \
      --veomni-root "$veomni_root" \
      --checkpoint "$base_checkpoint" \
      --source-video "$pair_root/$iid/${source_variant}.mp4" \
      --instruction "$instruction" \
      --output "$output" \
      --num-inference-steps 40 \
      --seed 2026081701 \
      --source-onset-policy hard1_every_step \
      --method-source-revision a2805a9022e6b1c799635fba3478a7647e22d696 \
      --method-source-archive-sha256 e6ad2321f5ea1e0ea4cc040bc0b7ff28731d19de21a19873d4cc1ec1f8cf9508
  fi
}

decode_one 7b88a1ca1f804f41 'A static camera shows a happy single grey French bulldog in a black harness, standing in a large pile of vibrant yellow autumn leaves in a sunlit park, facing the camera with its tongue out while its chest and the nearby leaves move subtly. The main French bulldog bends its hind legs, lowers its hips to the ground, settles into a stable sit facing the camera, and holds that seated pose. The shot stays continuous, the illumination remains stable, and the final frame is temporally coherent.' "$1"
decode_one 841b5e0080a1441d 'A fixed camera shows a single black-and-tan German shepherd standing on a leash in a grassy field, looking upward toward the viewer, with clumps of shed fur on the ground and subtle ear and breeze movement. The main German shepherd bends its hind legs, lowers its hips to the ground, settles into a stable sit facing the camera, and holds that seated pose. The shot stays continuous, the illumination remains stable, and the final frame is temporally coherent.' "$1"
decode_one a35b590961d24694 'A static camera shows a single young woman with long brown hair kneeling on one knee and looking at the camera against a solid warm yellow background, wearing a black jacket, pink top, cream trousers, and breathing gently. The main person shifts weight onto both feet, rises smoothly from kneeling, straightens both legs and the torso, and holds a stable upright standing pose. The shot stays continuous, the illumination remains stable, and the final frame is temporally coherent.' "$1"
decode_one a66e6818e4144928 'A static camera shows a single young woman in a low crouch on a city sidewalk beside a modern dark-glass building with horizontal metal bars, looking back over her shoulder, with tied dark hair, a black sports bra, grey trousers, and a blue-green plaid shirt in harsh sunlight and deep shadow. The main person presses through both feet, rises smoothly from the low crouch, straightens the legs and torso, faces the viewer, and holds upright. The shot stays continuous, the illumination remains stable, and the final frame is temporally coherent.' "$1"
