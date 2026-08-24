#!/usr/bin/env bash
set -euo pipefail

# Eight standard-FM action-adapter arms.  The only sweep axes are hard-row
# replay frequency, LR, and matched original-source reconstruction replay.
case "$(hostname -s)" in
  auh7-1b-gpu-246) learning_rate=3e-4; sr_every=0 ;;
  auh7-1b-gpu-247) learning_rate=1.5e-4; sr_every=0 ;;
  auh7-1b-gpu-248) learning_rate=3e-4; sr_every=4 ;;
  auh7-1b-gpu-279) learning_rate=1.5e-4; sr_every=4 ;;
  *) echo "forbidden node: hard-row replay is restricted to Job 140846 nodes" >&2; exit 3 ;;
esac

if [ "$#" -ne 1 ]; then
  echo "usage: $0 GPU_GROUP" >&2
  exit 2
fi
case "$1" in
  0) devices=0,1,2,3; row_repeat=1,2,1,1; repeat_tag=r1x2 ;;
  1) devices=4,5,6,7; row_repeat=1,3,1,1; repeat_tag=r1x3 ;;
  *) echo "GPU_GROUP must be 0 or 1" >&2; exit 2 ;;
esac

case "$learning_rate" in
  3e-4) lr_tag=lr3e4 ;;
  1.5e-4) lr_tag=lr1p5e4 ;;
  *) echo "unexpected learning rate" >&2; exit 4 ;;
esac
case "$sr_every" in
  0) sr_tag=sr0 ;;
  4) sr_tag=sr25 ;;
  *) echo "unexpected source reconstruction cadence" >&2; exit 4 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
source_tree="$root/stage1/source-3e99529-overlay"
runtime_tree="$root/stage1/source-be31323"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
arm="train_hardreplay_${repeat_tag}_${lr_tag}_${sr_tag}_s80_r1"
output="$root/stage1/$arm"
test ! -e "$output"

scratch="/tmp/${arm}-g$1-140846"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
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

extra_args=()
if [ "$sr_every" -ne 0 ]; then
  extra_args+=(--source-reconstruction-every "$sr_every")
fi

"$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$source_tree/methods/bernini_action_editing/train_same_video_dense_flow_adapter_v1.py" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$base_checkpoint" \
  --pair-manifest "$root/stage1/same_video_pairs_r2/manifest.json" \
  --output "$output" \
  --max-steps 80 \
  --micro-records 1 \
  --row-repeat "$row_repeat" \
  --source-variant mixed \
  --training-noise-policy varying \
  --learning-rate "$learning_rate" \
  --conditioner-learning-rate "$learning_rate" \
  --seed 2026081703 \
  --max-grad-norm 10 \
  --method-source-revision 3e995296c650aa7c477356959bfff45d3ec34a90 \
  --method-source-archive-sha256 ad3526e2dc9525c101421f7e9ab09b8ad860eaad9ee0b5eb170d29d7243ee26f \
  "${extra_args[@]}"

test -f "$output/TRAINING_COMPLETE"
