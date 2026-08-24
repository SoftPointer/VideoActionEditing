#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 GPU_GROUP [EVENT_ORDINAL]" >&2
  exit 2
fi
case "$1" in 0) devices=0,1,2,3; default_event=0 ;; 1) devices=4,5,6,7; default_event=4 ;; *) exit 2 ;; esac
event="${2:-$default_event}"
case "$event" in 0|1|2|3|4|5|6|7) ;; *) echo "event must be in [0,7]" >&2; exit 2 ;; esac
node="$(hostname -s)"
case "$node" in
  auh7-1b-gpu-246) arm=dpo_only_s4; final_step=4 ;;
  auh7-1b-gpu-247) arm=dpo_identity005_s4; final_step=4 ;;
  auh7-1b-gpu-248) arm=dpo_identity015_s4; final_step=4 ;;
  auh7-1b-gpu-279) arm=dpo_identity010_s8; final_step=8 ;;
  *) echo "forbidden node outside Job 140846" >&2; exit 3 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
runtime_tree="$stage/source-be31323"
train_root="$stage/interaction_complex8_large_lora_dpo_v1/$arm"
rollout_root="$stage/interaction_complex8_rv2v_candidates_v1"
output_root="$stage/interaction_complex8_large_lora_decode_v1/$arm/event_$(printf '%02d' "$event")"
wrapper="$stage/infer_interaction_complex8_large_lora_checkpoint_v1.py"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256

while [ ! -f "$train_root/COMPLETE" ]; do sleep 15; done
candidate="complex8-e$(printf '%02d' "$event")-rv2v-s0"
candidate_receipt="$rollout_root/$candidate/pair-v5-rollout-receipt.json"
test -f "$candidate_receipt"
source_video="$(jq -r .candidate.source_video "$candidate_receipt")"
source_sha="$(jq -r .candidate.source_video_sha256 "$candidate_receipt")"
caption="$(jq -r .candidate.complete_caption "$candidate_receipt")"
caption_sha="$(jq -r .candidate.complete_caption_sha256 "$candidate_receipt")"
seed="$(jq -r .candidate.seed "$candidate_receipt")"
mkdir -p "$output_root"

scratch="/tmp/interaction-complex8-decode-${arm}-g$1-140846"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
load_lock="$scratch/model-load.lock"
: >"$load_lock"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ROCR_VISIBLE_DEVICES="$devices"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
export NATIVE_SERIALIZED_HOST_LOAD_REQUIRED=1 NATIVE_V_AXIS_LOAD_LOCK="$load_lock"
export PYTHONPATH="$stage:$runtime_tree/methods/bernini_action_editing"

for step in 1 "$final_step"; do
  adapter="$train_root/checkpoints/step_$(printf '%04d' "$step").safetensors"
  output="$output_root/step_$(printf '%04d' "$step")"
  test -f "$adapter"
  if [ -f "$output/adapter-inference-receipt.json" ]; then continue; fi
  test ! -e "$output"
  adapter_sha="$(sha256sum "$adapter" | awk '{print $1}')"
  "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$wrapper" \
    --adapter "$adapter" \
    --expected-adapter-sha256 "$adapter_sha" \
    --adapter-label "${arm}-step-${step}" \
    --bernini-root "$bernini_root" \
    --veomni-root "$veomni_root" \
    --checkpoint "$checkpoint" \
    --checkpoint-content-manifest "$checkpoint_manifest" \
    --source-video "$source_video" \
    --expected-source-sha256 "$source_sha" \
    --action-prompt "$caption" \
    --expected-action-prompt-sha256 "$caption_sha" \
    --output-dir "$output" \
    --num-inference-steps 40 \
    --seed "$seed" \
    --method-source-revision be3132312b77125313901c928b7aedcfc2c72c12 \
    --method-source-archive-sha256 958b9350e32b5459053a4aa62dff6334fb0b251f41e8e60dcd16643cef0f9d3e
  test -f "$output/rv2v.mp4"
  test -f "$output/adapter-inference-receipt.json"
done
touch "$output_root/COMPLETE"
