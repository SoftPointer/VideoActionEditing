#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh7-1b-gpu-246)
    mode=phase0_flowwarp_raw; flow_tag=raw; max_timestep=833; schedule_start=20
    ;;
  auh7-1b-gpu-247)
    mode=phase0_flowwarp_raw; flow_tag=raw; max_timestep=625; schedule_start=30
    ;;
  auh7-1b-gpu-248)
    mode=phase0_flowwarp_camera_residual; flow_tag=camera; max_timestep=833; schedule_start=20
    ;;
  auh7-1b-gpu-279)
    mode=phase0_flowwarp_camera_residual; flow_tag=camera; max_timestep=625; schedule_start=30
    ;;
  *) echo "forbidden node: opt2 decode is restricted to Job 140846 nodes" >&2; exit 3 ;;
esac
if [ "$#" -ne 2 ]; then
  echo "usage: $0 2|4 GPU_GROUP" >&2
  exit 2
fi
source_reconstruction_every="$1"
gpu_group="$2"
case "$source_reconstruction_every" in
  2) preservation_tag=pres50 ;;
  4) preservation_tag=pres25 ;;
  *) echo "source reconstruction cadence must be 2 or 4" >&2; exit 2 ;;
esac
case "$gpu_group" in
  0) devices=0,1,2,3 ;;
  1) devices=4,5,6,7 ;;
  *) echo "GPU_GROUP must be 0 or 1" >&2; exit 2 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
source_tree="$root/stage1/source-a2805a9-overlay"
runtime_tree="$root/stage1/source-be31323"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
media_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_endpoint_consensus_release_11f3ed1/source/md/action_editing/20260815_reward/model_gain_135096/review/media
motion_checkpoint="$root/stage1/train_row0_noop_varying_out3e4_cond3e4_s160_r1/checkpoint-00000160"
arm="opt2_${flow_tag}_midlate_t${max_timestep}_${preservation_tag}_lr1e4_sr50_s80_r2"
source_copy_checkpoint="$root/stage1/$arm/checkpoint-00000080"
test -f "$root/stage1/$arm/TRAINING_COMPLETE"
test -f "$source_copy_checkpoint/adapter_model.safetensors"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES="$devices"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONPATH="$source_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"

decode_one() {
  local row="$1"
  local iid instruction
  if [ "$row" = row1 ]; then
    iid=841b5e0080a1441d
    instruction='A fixed camera shows a single black-and-tan German shepherd standing on a leash in a grassy field, looking upward toward the viewer, with clumps of shed fur on the ground and subtle ear and breeze movement. The main German shepherd bends its hind legs, lowers its hips to the ground, settles into a stable sit facing the camera, and holds that seated pose. The shot stays continuous, the illumination remains stable, and the final frame is temporally coherent.'
  elif [ "$row" = row2 ]; then
    iid=a35b590961d24694
    instruction='A static camera shows a single young woman with long brown hair kneeling on one knee and looking at the camera against a solid warm yellow background, wearing a black jacket, pink top, cream trousers, and breathing gently. The main person shifts weight onto both feet, rises smoothly from kneeling, straightens both legs and the torso, and holds a stable upright standing pose. The shot stays continuous, the illumination remains stable, and the final frame is temporally coherent.'
  else
    echo "row must be row1 or row2" >&2
    exit 2
  fi
  local output_dir="$root/stage1/infer_opt2/${arm}_s080_start${schedule_start}_${row}_r1"
  local scratch="/tmp/${arm}-s080-start${schedule_start}-${row}-g${gpu_group}-140846"
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
    --dense-flow-source-copy-checkpoint "$source_copy_checkpoint" \
    --dense-flow-source-copy-scale 1.0 \
    --dense-flow-source-copy-schedule-start "$schedule_start" \
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
    --method-source-revision a2805a9022e6b1c799635fba3478a7647e22d696 \
    --method-source-archive-sha256 e6ad2321f5ea1e0ea4cc040bc0b7ff28731d19de21a19873d4cc1ec1f8cf9508
}

decode_one row1
decode_one row2
