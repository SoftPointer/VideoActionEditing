#!/usr/bin/env bash
set -euo pipefail

node="$(hostname -s)"
case "$node" in
  auh7-1b-gpu-246) arm=dpo_only_s4; steps=4; preserve=0.00; lr=2.0e-5 ;;
  auh7-1b-gpu-247) arm=dpo_identity005_s4; steps=4; preserve=0.05; lr=2.0e-5 ;;
  auh7-1b-gpu-248) arm=dpo_identity015_s4; steps=4; preserve=0.15; lr=2.0e-5 ;;
  auh7-1b-gpu-279) arm=dpo_identity010_s8; steps=8; preserve=0.10; lr=2.0e-5 ;;
  *) echo "forbidden node outside Job 140846" >&2; exit 3 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
runtime_tree="$stage/source-be31323"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
manifest="$stage/interaction_complex8_preference_v1.json"
trainer="$stage/train_interaction_complex8_large_lora_dpo_v1.py"
output="$stage/interaction_complex8_large_lora_dpo_v1/$arm"
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4

test -f "$manifest"
test -f "$trainer"
test ! -e "$output"
manifest_sha="$(sha256sum "$manifest" | awk '{print $1}')"
scratch="/tmp/interaction-complex8-large-dpo-${arm}-140846"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
export PYTHONPATH="$stage:$runtime_tree/methods/bernini_action_editing"

"$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$trainer" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$checkpoint" \
  --preference-manifest "$manifest" \
  --expected-preference-manifest-sha256 "$manifest_sha" \
  --output "$output" \
  --max-steps "$steps" \
  --learning-rate "$lr" \
  --beta 100 \
  --max-grad-norm 1.0 \
  --preservation-weight "$preserve" \
  --minimum-peak-memory-ratio 0.50 \
  --seed 20260817
test -f "$output/COMPLETE"
