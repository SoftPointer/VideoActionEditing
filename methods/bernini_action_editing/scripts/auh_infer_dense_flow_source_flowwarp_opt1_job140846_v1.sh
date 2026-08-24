#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh7-1b-gpu-246) mode=phase0_flowwarp_raw; flow_tag=raw; scope=full ;;
  auh7-1b-gpu-247) mode=phase0_flowwarp_camera_residual; flow_tag=camera; scope=full ;;
  auh7-1b-gpu-248) mode=phase0_flowwarp_raw; flow_tag=raw; scope=midlate ;;
  auh7-1b-gpu-279) mode=phase0_flowwarp_camera_residual; flow_tag=camera; scope=midlate ;;
  *) echo "forbidden node: opt1 decode is restricted to 246/247/248/279" >&2; exit 3 ;;
esac
if [ "$#" -ne 2 ]; then
  echo "usage: $0 1e-4|5e-5 GPU_GROUP" >&2
  exit 2
fi
learning_rate="$1"
gpu_group="$2"
case "$learning_rate" in
  1e-4) lr_tag=lr1e4 ;;
  5e-5) lr_tag=lr5e5 ;;
  *) echo "learning rate must be 1e-4 or 5e-5" >&2; exit 2 ;;
esac
case "$gpu_group" in
  0) devices=0,1,2,3 ;;
  1) devices=4,5,6,7 ;;
  *) echo "GPU_GROUP must be 0 or 1" >&2; exit 2 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
source_tree="$root/stage1/source-1591f66-overlay"
runtime_tree="$root/stage1/source-be31323"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
media_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_endpoint_consensus_release_11f3ed1/source/md/action_editing/20260815_reward/model_gain_135096/review/media
motion_checkpoint="$root/stage1/train_row0_noop_varying_out3e4_cond3e4_s160_r1/checkpoint-00000160"
arm="opt1_${flow_tag}_${scope}_${lr_tag}_sr50_s80_r1"
source_copy_root="$root/stage1/$arm"
test -f "$source_copy_root/TRAINING_COMPLETE"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES="$devices"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONPATH="$source_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"

decode_one() {
  local step="$1"
  local row="$2"
  local iid instruction
  if [ "$row" = row0 ]; then
    iid=7b88a1ca1f804f41
    instruction='A static camera shows a happy single grey French bulldog in a black harness, standing in a large pile of vibrant yellow autumn leaves in a sunlit park, facing the camera with its tongue out while its chest and the nearby leaves move subtly. The main French bulldog bends its hind legs, lowers its hips to the ground, settles into a stable sit facing the camera, and holds that seated pose. The shot stays continuous, the illumination remains stable, and the final frame is temporally coherent.'
  elif [ "$row" = row3 ]; then
    iid=a66e6818e4144928
    instruction='A static camera shows a single young woman in a low crouch on a city sidewalk beside a modern dark-glass building with horizontal metal bars, looking back over her shoulder, with tied dark hair, a black sports bra, grey trousers, and a blue-green plaid shirt in harsh sunlight and deep shadow. The main person presses through both feet, rises smoothly from the low crouch, straightens the legs and torso, faces the viewer, and holds upright. The shot stays continuous, the illumination remains stable, and the final frame is temporally coherent.'
  else
    echo "row must be row0 or row3" >&2
    exit 2
  fi
  local checkpoint="$source_copy_root/checkpoint-00000${step}"
  local output_dir="$root/stage1/infer_opt1/${arm}_s${step}_${row}_r1"
  local scratch="/tmp/${arm}-s${step}-${row}-g${gpu_group}-140846"
  test -f "$checkpoint/adapter_model.safetensors"
  test ! -e "$output_dir"
  mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton" "$output_dir"
  export MIOPEN_USER_DB_PATH="$scratch/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
  export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions"
  export TRITON_CACHE_DIR="$scratch/triton"
  "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$source_tree/methods/bernini_action_editing/infer_same_video_dense_flow_adapter_v1.py" \
    --dense-flow-checkpoint "$motion_checkpoint" \
    --dense-flow-scale 1.0 \
    --dense-flow-source-copy-checkpoint "$checkpoint" \
    --dense-flow-source-copy-scale 1.0 \
    --dense-flow-source-copy-mode "$mode" \
    --flow-bundle "$root/flows/$iid.safetensors" \
    --bernini-root "$bernini_root" \
    --veomni-root "$veomni_root" \
    --checkpoint "$base_checkpoint" \
    --source-video "$media_root/$iid/source.mp4" \
    --instruction "$instruction" \
    --output "$output_dir/output.mp4" \
    --num-inference-steps 40 \
    --seed 2026081601 \
    --source-onset-policy hard1_every_step \
    --method-source-revision 1591f66a7b6b957950ef524c53db107ebd72aaa7 \
    --method-source-archive-sha256 c96c307799ff97cab7510b472ea2a36daf534dad56059989edd07bbc1c6d882d
}

for step in 020 040 080; do
  decode_one "$step" row0
  decode_one "$step" row3
done
