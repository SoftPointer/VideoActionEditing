#!/usr/bin/env bash
set -euo pipefail

# Eight SP4 action-only arms over nine audited self-generated targets.  The
# matrix separates LR from denoise-cell coverage; every target is visited once
# per deterministic nine-row cycle, with mixed noop/incomplete sources.
if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$1" in 0) devices=0,1,2,3 ;; 1) devices=4,5,6,7 ;; *) echo "GPU_GROUP must be 0 or 1" >&2; exit 2 ;; esac
case "$(hostname -s):$1" in
  auh7-1b-gpu-246:0) band=full; learning_rate=3e-4 ;;
  auh7-1b-gpu-246:1) band=full; learning_rate=2e-4 ;;
  auh7-1b-gpu-247:0) band=full; learning_rate=1.5e-4 ;;
  auh7-1b-gpu-247:1) band=full; learning_rate=1e-4 ;;
  auh7-1b-gpu-248:0) band=mid250_750; learning_rate=3e-4 ;;
  auh7-1b-gpu-248:1) band=mid250_750; learning_rate=2e-4 ;;
  auh7-1b-gpu-279:0) band=noisy500_1000; learning_rate=2e-4 ;;
  auh7-1b-gpu-279:1) band=clean0_500; learning_rate=2e-4 ;;
  *) echo "forbidden node/group: expanded-bank training is restricted to Job 140846" >&2; exit 3 ;;
esac
case "$learning_rate" in 3e-4) lr_tag=lr3e4 ;; 2e-4) lr_tag=lr2e4 ;; 1.5e-4) lr_tag=lr1p5e4 ;; 1e-4) lr_tag=lr1e4 ;; *) exit 4 ;; esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
source_tree="$root/stage1/source-43ed61a-overlay"
prior_tree="$root/stage1/source-3e99529-overlay"
runtime_tree="$root/stage1/source-be31323"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
arm="train_bank9_${band}_${lr_tag}_s90_r2"
output="$root/stage1/$arm"
pair_manifest="$root/stage1/expanded_bank_v1/manifest_flowfix_v2.json"
test -f "$pair_manifest"
test ! -e "$output"

band_args=()
case "$band" in
  full) ;;
  mid250_750) band_args+=(--training-min-timestep 250 --training-max-timestep 750) ;;
  noisy500_1000) band_args+=(--training-min-timestep 500 --training-max-timestep 1000) ;;
  clean0_500) band_args+=(--training-min-timestep 0 --training-max-timestep 500) ;;
  *) exit 4 ;;
esac
scratch="/tmp/${arm}-g$1-140846"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES="$devices"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
export PYTHONPATH="$source_tree/methods/bernini_action_editing:$prior_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"

"$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$source_tree/methods/bernini_action_editing/train_same_video_dense_flow_adapter_v1.py" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$base_checkpoint" \
  --pair-manifest "$pair_manifest" \
  --output "$output" \
  --max-steps 90 \
  --micro-records 1 \
  --row-repeat auto \
  --source-variant mixed \
  --training-noise-policy varying \
  --learning-rate "$learning_rate" \
  --conditioner-learning-rate "$learning_rate" \
  --seed 2026081705 \
  --max-grad-norm 10 \
  --method-source-revision 43ed61a93b03f1fb45f7758fc6da4220f210fd70 \
  --method-source-archive-sha256 405fd1352fa08204078ec0907bd7eb06ed264ff59fe381a86df022ff9f08cb24 \
  "${band_args[@]}"

test -f "$output/TRAINING_COMPLETE"
