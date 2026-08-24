#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh7-1b-gpu-246)
    mode=phase0_flowwarp_raw; flow_tag=raw; scope=full
    blocks=0,4,8,12,16,20,24,28
    ;;
  auh7-1b-gpu-247)
    mode=phase0_flowwarp_camera_residual; flow_tag=camera; scope=full
    blocks=0,4,8,12,16,20,24,28
    ;;
  auh7-1b-gpu-248)
    mode=phase0_flowwarp_raw; flow_tag=raw; scope=midlate
    blocks=8,12,16,20,24,28
    ;;
  auh7-1b-gpu-279)
    mode=phase0_flowwarp_camera_residual; flow_tag=camera; scope=midlate
    blocks=8,12,16,20,24,28
    ;;
  *) echo "forbidden node: opt1 is restricted to 246/247/248/279" >&2; exit 3 ;;
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
motion_checkpoint="$root/stage1/train_row0_noop_varying_out3e4_cond3e4_s160_r1/checkpoint-00000160"
output="$root/stage1/opt1_${flow_tag}_${scope}_${lr_tag}_sr50_s80_r1"
scratch="/tmp/flowwarp-opt1-${flow_tag}-${scope}-${lr_tag}-g${gpu_group}-140846-${SLURM_STEP_ID:-ssh}"

test ! -e "$output"
test -f "$motion_checkpoint/adapter_model.safetensors"
test -f "$motion_checkpoint/receipt.json"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES="$devices"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export MIOPEN_USER_DB_PATH="$scratch/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions"
export TRITON_CACHE_DIR="$scratch/triton"
export PYTHONPATH="$source_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"

"$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$source_tree/methods/bernini_action_editing/train_same_video_dense_flow_adapter_v1.py" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$base_checkpoint" \
  --pair-manifest "$root/stage1/same_video_pairs_r2/manifest.json" \
  --output "$output" \
  --max-steps 80 \
  --micro-records 1 \
  --source-variant mixed \
  --source-reconstruction-every 2 \
  --training-noise-policy varying \
  --learning-rate "$learning_rate" \
  --conditioner-learning-rate "$learning_rate" \
  --frozen-motion-checkpoint "$motion_checkpoint" \
  --source-copy-mode "$mode" \
  --source-copy-block-indices "$blocks" \
  --seed 2026081703 \
  --max-grad-norm 10 \
  --method-source-revision 1591f66a7b6b957950ef524c53db107ebd72aaa7 \
  --method-source-archive-sha256 c96c307799ff97cab7510b472ea2a36daf534dad56059989edd07bbc1c6d882d

test -f "$output/TRAINING_COMPLETE"
