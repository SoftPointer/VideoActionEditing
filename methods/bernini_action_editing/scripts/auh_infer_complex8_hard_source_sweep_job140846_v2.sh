#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh7-1b-gpu-246) mode=matched; step=10 ;;
  auh7-1b-gpu-247) mode=cross; step=40 ;;
  *) echo "hard-source sweep is restricted to Job 140846 nodes 246/247" >&2; exit 3 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
source_tree="$stage/source-crossappearance-hard-v2"
old_source_tree="$stage/source-816e892-queryflow"
runtime_tree="$stage/source-be31323"
release="$stage/complex8_crossappearance_motion_v1"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
checkpoint="$release/train_joint_${mode}_q12x20_all30_r256_micro1_s64_v3/checkpoint-$(printf '%08d' "$step")"
source=/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/186d36f998a14e14/source.mp4
flow="$release/flows_real_event01/v0.safetensors"
instruction="The child notices one marked stone, stops, bends both knees into a crouch, reaches one hand to that exact stone, closes the fingers around it with visible contact, lifts it from the ground, rises back to standing, and holds the same stone visibly in the hand. The vacated spot remains empty. Keep exactly one child and the original set of stones. Preserve identity, clothing, all other stones, path, plants, lighting, framing and camera. No cuts, teleportation, object replacement, duplication or extra limbs."
revision=$(tr -d '\n' < "$release/method-source-hard-v2.revision")
archive_sha=$(sha256sum "$release/method-source-hard-v2.tar" | awk '{print $1}')

test -f "$checkpoint/adapter_model.safetensors"
test -f "$checkpoint/adapter/adapter_model.safetensors"
test -f "$flow"
test -f "$source"

run_one() {
  local label="$1"
  local transport_mode="$2"
  local scale="$3"
  local schedule_start="$4"
  local blocks="$5"
  local output_dir="$release/infer_hard_source/$mode/$label"
  local output="$output_dir/output.mp4"
  if [ -s "$output" ]; then
    echo "SKIP $mode $label"
    return
  fi
  test ! -e "$output_dir"
  mkdir -p "$output_dir"
  local scratch="/tmp/complex8-hard-${mode}-${label}-140846"
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
  echo "START $mode $label step=$step mode=$transport_mode scale=$scale schedule=$schedule_start blocks=$blocks"
  "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$source_tree/methods/bernini_action_editing/infer_same_video_dense_flow_adapter_v1.py" \
    --dense-flow-checkpoint "$checkpoint" \
    --dense-flow-scale 1.0 \
    --flow-bundle "$flow" \
    --dense-flow-hard-source-mode "$transport_mode" \
    --dense-flow-hard-source-scale "$scale" \
    --dense-flow-hard-source-schedule-start "$schedule_start" \
    --dense-flow-hard-source-block-indices "$blocks" \
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
  echo "DONE $mode $label"
}

# Two axes are isolated: preservation strength/onset and whether anchor-flow
# correspondence is used.  The no-hard-transport baseline is the already
# decoded matched-s10 or cross-s40 video with exactly the same seed.
run_one cam_s05_t24_b18-22-26-29 phase0_flowwarp_camera_residual 0.05 24 18,22,26,29
run_one cam_s10_t28_b18-22-26-29 phase0_flowwarp_camera_residual 0.10 28 18,22,26,29
run_one cam_s15_t32_b18-22-26-29 phase0_flowwarp_camera_residual 0.15 32 18,22,26,29
run_one raw_s10_t28_b18-22-26-29 phase0_flowwarp_raw 0.10 28 18,22,26,29

printf 'complete\n' > "$release/infer_hard_source/$mode/SWEEP_COMPLETE"
