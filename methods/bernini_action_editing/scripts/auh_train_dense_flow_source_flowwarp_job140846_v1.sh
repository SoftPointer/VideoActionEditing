#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh7-1b-gpu-246) mode=phase0_flowwarp_raw; tag=flowwarp_raw ;;
  auh7-1b-gpu-247) mode=phase0_flowwarp_camera_residual; tag=flowwarp_camera ;;
  *) echo "forbidden node: source flow-warp is restricted to 246/247" >&2; exit 3 ;;
esac
if [ "$#" -ne 2 ]; then
  echo "usage: $0 SOURCE_RECONSTRUCTION_EVERY GPU_GROUP" >&2
  exit 2
fi
source_reconstruction_every="$1"
gpu_group="$2"
case "$source_reconstruction_every" in
  2) ratio=sr50 ;;
  4) ratio=sr25 ;;
  *) echo "source reconstruction cadence must be 2 or 4" >&2; exit 2 ;;
esac
case "$gpu_group" in
  0) devices=0,1,2,3 ;;
  1) devices=4,5,6,7 ;;
  *) echo "GPU_GROUP must be 0 or 1" >&2; exit 2 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
source_tree="$root/stage1/source-78d109b-overlay"
runtime_tree="$root/stage1/source-be31323"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
motion_checkpoint="$root/stage1/train_row0_noop_varying_out3e4_cond3e4_s160_r1/checkpoint-00000160"
output="$root/stage1/train_source_${tag}_${ratio}_out3e4_cond3e4_s80_r1"

test ! -e "$output"
test -f "$motion_checkpoint/adapter_model.safetensors"
test -f "$motion_checkpoint/receipt.json"
scratch="/tmp/dense-flow-source-${tag}-${ratio}-g${gpu_group}-140846-${SLURM_STEP_ID:-ssh}"
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
  --source-reconstruction-every "$source_reconstruction_every" \
  --training-noise-policy varying \
  --learning-rate 3e-4 \
  --conditioner-learning-rate 3e-4 \
  --frozen-motion-checkpoint "$motion_checkpoint" \
  --source-copy-mode "$mode" \
  --seed 2026081702 \
  --max-grad-norm 10 \
  --method-source-revision 78d109b660b3135d0b8c2d00aadaba06b711d93f \
  --method-source-archive-sha256 4f4a8955016dab7925ca420152c05a80a6c773670325279f8e621fccdaf6e5de

test -f "$output/TRAINING_COMPLETE"
