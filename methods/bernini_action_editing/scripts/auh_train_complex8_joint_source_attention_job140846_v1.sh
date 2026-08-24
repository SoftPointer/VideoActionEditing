#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh7-1b-gpu-246)
    tag=all30
    source_blocks=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
    ;;
  auh7-1b-gpu-247)
    tag=late15
    source_blocks=15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
    ;;
  *) echo "joint source-attention training is restricted to Job 140846 nodes 246/247" >&2; exit 3 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
source_tree="$stage/source-crossappearance-sourceattn-v3"
old_source_tree="$stage/source-816e892-queryflow"
runtime_tree="$stage/source-be31323"
release="$stage/complex8_crossappearance_motion_v1"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
manifest="$release/pairs_exact/manifest_cross.json"
motion_checkpoint="$release/train_joint_cross_q12x20_all30_r256_micro1_s64_v3/checkpoint-00000040"
output="$release/train_sourceattn_cross_s40_${tag}_q12x20_micro1_s64_v1"
archive="$release/method-source-sourceattn-v3.tar"
revision_file="$release/method-source-sourceattn-v3.revision"

test -f "$manifest"
test -f "$motion_checkpoint/adapter_model.safetensors"
test -f "$motion_checkpoint/adapter/adapter_model.safetensors"
test -f "$archive"
test -f "$revision_file"
test ! -e "$output"
revision=$(tr -d '\n' < "$revision_file")
archive_sha=$(sha256sum "$archive" | awk '{print $1}')

scratch="/tmp/complex8-sourceattn-${tag}-140846"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES=0,1,2,3
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
export PYTHONPATH="$source_tree/methods/bernini_action_editing:$old_source_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"

exec "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$source_tree/methods/bernini_action_editing/train_same_video_dense_flow_adapter_v1.py" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$base_checkpoint" \
  --pair-manifest "$manifest" \
  --output "$output" \
  --max-steps 64 \
  --micro-records 1 \
  --row-repeat auto \
  --source-variant mixed \
  --training-noise-policy varying \
  --dense-flow-mode phase_attention_12x20 \
  --frozen-motion-checkpoint "$motion_checkpoint" \
  --source-copy-mode phase0_attention_12x20 \
  --source-copy-block-indices "$source_blocks" \
  --learning-rate 1e-4 \
  --conditioner-learning-rate 1e-4 \
  --seed 2026081819 \
  --max-grad-norm 10 \
  --method-source-revision "$revision" \
  --method-source-archive-sha256 "$archive_sha"

test -f "$output/TRAINING_COMPLETE"
