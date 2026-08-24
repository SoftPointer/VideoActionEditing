#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh7-1b-gpu-246) tag=all30 ;;
  auh7-1b-gpu-247) tag=late15 ;;
  *) echo "joint source-attention inference is restricted to Job 140846 nodes 246/247" >&2; exit 3 ;;
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
motion_checkpoint="$release/train_joint_cross_q12x20_all30_r256_micro1_s64_v3/checkpoint-00000040"
source_copy_root="$release/train_sourceattn_cross_s40_${tag}_q12x20_micro1_s64_v1"
source=/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/186d36f998a14e14/source.mp4
flow="$release/flows_real_event01/v0.safetensors"
instruction="The child notices one marked stone, stops, bends both knees into a crouch, reaches one hand to that exact stone, closes the fingers around it with visible contact, lifts it from the ground, rises back to standing, and holds the same stone visibly in the hand. The vacated spot remains empty. Keep exactly one child and the original set of stones. Preserve identity, clothing, all other stones, path, plants, lighting, framing and camera. No cuts, teleportation, object replacement, duplication or extra limbs."
revision=$(tr -d '\n' < "$release/method-source-sourceattn-v3.revision")
archive_sha=$(sha256sum "$release/method-source-sourceattn-v3.tar" | awk '{print $1}')

test -f "$motion_checkpoint/adapter_model.safetensors"
test -f "$motion_checkpoint/adapter/adapter_model.safetensors"
test -f "$source_copy_root/TRAINING_COMPLETE"
test -f "$flow"
test -f "$source"

run_one() {
  local step="$1"
  local scale="$2"
  local label="s$(printf '%03d' "$step")_k$(printf '%03d' "$(awk -v value="$scale" 'BEGIN { print int(value * 100 + 0.5) }')")_t00"
  local source_copy_checkpoint="$source_copy_root/checkpoint-$(printf '%08d' "$step")"
  local output_dir="$release/infer_sourceattn/$tag/$label"
  local output="$output_dir/output.mp4"
  if [ -s "$output" ]; then
    echo "SKIP $tag $label"
    return
  fi
  test -f "$source_copy_checkpoint/adapter_model.safetensors"
  test ! -e "$output_dir"
  mkdir -p "$output_dir"
  local scratch="/tmp/complex8-sourceattn-infer-${tag}-${label}-140846"
  mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export ROCR_VISIBLE_DEVICES="${ROCR_DEVICE_SET:-0,1,2,3}"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
  export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
  export PYTHONPATH="$source_tree/methods/bernini_action_editing:$old_source_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"
  echo "START $tag $label source_copy_step=$step scale=$scale"
  "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$source_tree/methods/bernini_action_editing/infer_same_video_dense_flow_adapter_v1.py" \
    --dense-flow-checkpoint "$motion_checkpoint" \
    --dense-flow-scale 1.0 \
    --flow-bundle "$flow" \
    --dense-flow-source-copy-checkpoint "$source_copy_checkpoint" \
    --dense-flow-source-copy-mode phase0_attention_12x20 \
    --dense-flow-source-copy-scale "$scale" \
    --dense-flow-source-copy-schedule-start 0 \
    --bernini-root "$bernini_root" \
    --veomni-root "$veomni_root" \
    --checkpoint "$base_checkpoint" \
    --source-video "$source" \
    --instruction "$instruction" \
    --output "$output" \
    --num-inference-steps 40 \
    --seed 2026081819 \
    --source-onset-policy hard1_every_step \
    --method-source-revision "$revision" \
    --method-source-archive-sha256 "$archive_sha"
  test -s "$output"
  test -s "$output.receipt.json"
  test -s "$output.dense-flow.json"
  echo "DONE $tag $label"
}

case "${SOURCEATTN_WORKER:-all}" in
  all)
    run_one 10 1.0
    run_one 40 1.0
    run_one 64 1.0
    run_one 40 0.5
    printf 'complete\n' > "$release/infer_sourceattn/$tag/SWEEP_COMPLETE"
    ;;
  tail)
    run_one 64 1.0
    run_one 40 0.5
    printf 'complete\n' > "$release/infer_sourceattn/$tag/TAIL_WORKER_COMPLETE"
    ;;
  quarter)
    run_one 40 0.25
    printf 'complete\n' > "$release/infer_sourceattn/$tag/QUARTER_WORKER_COMPLETE"
    ;;
  *)
    echo "SOURCEATTN_WORKER must be all, tail or quarter" >&2
    exit 4
    ;;
esac
