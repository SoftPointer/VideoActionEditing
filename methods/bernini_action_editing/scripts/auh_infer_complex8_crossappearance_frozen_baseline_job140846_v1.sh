#!/usr/bin/env bash
set -euo pipefail

[ "$(hostname -s)" = auh7-1b-gpu-247 ] || {
  echo "frozen baseline is restricted to node 247" >&2
  exit 3
}
root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
source_tree="$stage/source-crossappearance-motion-v1"
runtime_tree="$stage/source-be31323"
release="$stage/complex8_crossappearance_motion_v1"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
event_root="$stage/interaction_complex8_multianchor_v2_r1/e01_reach-grasp-lift-stone"
instruction="The child notices one marked stone, stops, bends both knees into a crouch, reaches one hand to that exact stone, closes the fingers around it with visible contact, lifts it from the ground, rises back to standing, and holds the same stone visibly in the hand. The vacated spot remains empty. Keep exactly one child and the original set of stones. Preserve identity, clothing, all other stones, path, plants, lighting, framing and camera. No cuts, teleportation, object replacement, duplication or extra limbs."
revision=$(tr -d '\n' < "$release/method-source.revision")
archive_sha=$(sha256sum "$release/method-source.tar" | awk '{print $1}')

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ROCR_VISIBLE_DEVICES=0,1,2,3
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONPATH="$source_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"

decode_one() {
  local case_id="$1" source="$2"
  local output_dir="$release/infer/frozen/$case_id"
  local scratch="/tmp/complex8-frozen-${case_id}-140846"
  test ! -e "$output_dir"
  mkdir -p "$output_dir" "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
  export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
  export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
  "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$runtime_tree/methods/bernini_action_editing/infer_lora.py" \
    --bernini-root "$bernini_root" \
    --veomni-root "$veomni_root" \
    --checkpoint "$base_checkpoint" \
    --base-only \
    --source-video "$source" \
    --instruction "$instruction" \
    --output "$output_dir/output.mp4" \
    --num-inference-steps 40 \
    --seed 2026081819 \
    --source-onset-policy hard1_every_step \
    --method-source-revision "$revision" \
    --method-source-archive-sha256 "$archive_sha"
}

decode_one synthetic "$event_root/v0/noop.mp4"
decode_one real /vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/186d36f998a14e14/source.mp4
