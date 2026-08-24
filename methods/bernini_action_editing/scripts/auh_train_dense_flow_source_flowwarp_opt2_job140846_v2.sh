#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh7-1b-gpu-246)
    mode=phase0_flowwarp_raw; flow_tag=raw; max_timestep=833
    ;;
  auh7-1b-gpu-247)
    mode=phase0_flowwarp_raw; flow_tag=raw; max_timestep=625
    ;;
  auh7-1b-gpu-248)
    mode=phase0_flowwarp_camera_residual; flow_tag=camera; max_timestep=833
    ;;
  auh7-1b-gpu-279)
    mode=phase0_flowwarp_camera_residual; flow_tag=camera; max_timestep=625
    ;;
  *) echo "forbidden node: opt2 is restricted to Job 140846 nodes" >&2; exit 3 ;;
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
source_tree="$root/stage1/source-deb9fb0-overlay"
runtime_tree="$root/stage1/source-be31323"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
motion_checkpoint="$root/stage1/train_row0_noop_varying_out3e4_cond3e4_s160_r1/checkpoint-00000160"
output="$root/stage1/opt2_${flow_tag}_midlate_t${max_timestep}_${preservation_tag}_lr1e4_sr50_s80_r2"
scratch="/tmp/flowwarp-opt2-r2-${flow_tag}-t${max_timestep}-${preservation_tag}-g${gpu_group}-140846-${SLURM_STEP_ID:-ssh}"

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
  --source-reconstruction-every "$source_reconstruction_every" \
  --training-noise-policy varying \
  --training-max-timestep "$max_timestep" \
  --learning-rate 1e-4 \
  --conditioner-learning-rate 1e-4 \
  --frozen-motion-checkpoint "$motion_checkpoint" \
  --source-copy-mode "$mode" \
  --source-copy-block-indices 8,12,16,20,24,28 \
  --seed 2026081705 \
  --max-grad-norm 10 \
  --method-source-revision deb9fb0d31e38917b4a2bf471c1bbee7a9d6e762 \
  --method-source-archive-sha256 8b22e9281f0e67216d0441286b9d7d957977cc5c8fa798a98ba49264af74b297

test -f "$output/TRAINING_COMPLETE"
