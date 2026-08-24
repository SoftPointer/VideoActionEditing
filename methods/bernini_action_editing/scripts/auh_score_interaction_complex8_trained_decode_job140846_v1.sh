#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then echo "usage: $0 GPU_GROUP EVENT_ORDINAL" >&2; exit 2; fi
case "$1" in 0) devices=0,1,2,3 ;; 1) devices=4,5,6,7 ;; *) exit 2 ;; esac
event="$2"
case "$event" in 0|1|2|3|4|5|6|7) ;; *) exit 2 ;; esac
case "$(hostname -s)" in
  auh7-1b-gpu-246|auh7-1b-gpu-247|auh7-1b-gpu-248|auh7-1b-gpu-279) ;;
  *) echo "forbidden node outside Job 140846" >&2; exit 3 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
runtime_tree="$stage/source-be31323"
script="$stage/score_interaction_complex8_trained_decode_v1.py"
authoring="$stage/interaction_complex8_multianchor_authoring_v2.json"
rollout_root="$stage/interaction_complex8_rv2v_candidates_v1"
decode_root="$stage/interaction_complex8_large_lora_decode_v1"
output="$stage/interaction_complex8_large_lora_action_score_v1/event_$(printf '%02d' "$event")"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256

for arm in dpo_only_s4 dpo_identity005_s4 dpo_identity015_s4 dpo_identity010_s8; do
  while [ ! -f "$decode_root/$arm/event_$(printf '%02d' "$event")/COMPLETE" ]; do sleep 15; done
done
test ! -e "$output"
mkdir -p "$(dirname "$output")"
scratch="/tmp/interaction-complex8-trained-score-e${event}-g$1-140846"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ROCR_VISIBLE_DEVICES="$devices"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
export PYTHONPATH="$stage:$runtime_tree/methods/bernini_action_editing"

"$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$script" \
  --event "$event" \
  --authoring-manifest "$authoring" \
  --rollout-root "$rollout_root" \
  --decode-root "$decode_root" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$checkpoint" \
  --checkpoint-content-manifest "$checkpoint_manifest" \
  --output-dir "$output"
test -f "$output/COMPLETE"
