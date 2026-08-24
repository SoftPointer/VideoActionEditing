#!/usr/bin/env bash
set -euo pipefail

# Eight SP4 fresh-from-zero arms.  This round separates representation
# alignment (local coordinate injection vs per-phase query retrieval) from
# capacity (the original eight blocks vs all 30 transformer blocks).
if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$1" in 0) devices=0,1,2,3 ;; 1) devices=4,5,6,7 ;; *) echo "GPU_GROUP must be 0 or 1" >&2; exit 2 ;; esac

case "$(hostname -s):$1" in
  auh7-1b-gpu-246:0) mode=phase_attention_8x12; scope=block8; learning_rate=3e-4 ;;
  auh7-1b-gpu-246:1) mode=phase_attention_8x12; scope=all30; learning_rate=1e-4 ;;
  auh7-1b-gpu-247:0) mode=phase_attention_8x12; scope=all30; learning_rate=2e-4 ;;
  auh7-1b-gpu-247:1) mode=phase_attention_12x20; scope=block8; learning_rate=3e-4 ;;
  auh7-1b-gpu-248:0) mode=phase_attention_12x20; scope=all30; learning_rate=1e-4 ;;
  auh7-1b-gpu-248:1) mode=phase_attention_12x20; scope=all30; learning_rate=2e-4 ;;
  auh7-1b-gpu-279:0) mode=local_mlp; scope=all30; learning_rate=1e-4 ;;
  auh7-1b-gpu-279:1) mode=local_mlp; scope=all30; learning_rate=2e-4 ;;
  *) echo "forbidden node/group: query-flow training is restricted to Job 140846" >&2; exit 3 ;;
esac

case "$mode" in
  phase_attention_8x12) mode_tag=q8x12 ;;
  phase_attention_12x20) mode_tag=q12x20 ;;
  local_mlp) mode_tag=local ;;
  *) exit 4 ;;
esac
case "$learning_rate" in 3e-4) lr_tag=lr3e4 ;; 2e-4) lr_tag=lr2e4 ;; 1e-4) lr_tag=lr1e4 ;; *) exit 4 ;; esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
source_tree="$root/stage1/source-d7f8987-queryflowfix-min"
old_source_tree="$root/stage1/source-816e892-queryflow"
runtime_tree="$root/stage1/source-be31323"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
pair_manifest="$root/stage1/expanded_bank_v1/manifest_flowfix_v2.json"
arm="train_bank9_${mode_tag}_${scope}_${lr_tag}_s90_r2"
output="$root/stage1/$arm"
test -f "$pair_manifest"
test ! -e "$output"

block_args=()
case "$scope" in
  block8) ;;
  all30) block_args+=(--adapter-block-indices "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29") ;;
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
export PYTHONPATH="$source_tree/methods/bernini_action_editing:$old_source_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"

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
  --dense-flow-mode "$mode" \
  --learning-rate "$learning_rate" \
  --conditioner-learning-rate "$learning_rate" \
  --seed 2026081706 \
  --max-grad-norm 10 \
  --method-source-revision d7f898700277eb3c1f848378bf2866a491a96b71 \
  --method-source-archive-sha256 24b4f33b92faa213c2b2c31a6b248007e6a62b02e267af3b8296273c8d1ada12 \
  "${block_args[@]}"

test -f "$output/TRAINING_COMPLETE"
exit=0
