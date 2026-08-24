#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh7-1b-gpu-246) mode=phase0_attention_8x12; tag=attn8x12 ;;
  auh7-1b-gpu-247) mode=phase0_attention_12x20; tag=attn12x20 ;;
  *) echo "forbidden node: source attention is restricted to 246/247" >&2; exit 3 ;;
esac
if [ "$#" -ne 1 ] || { [ "$1" != row0 ] && [ "$1" != row3 ]; }; then
  echo "usage: $0 row0|row3" >&2
  exit 2
fi
row="$1"

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
source_tree="$root/stage1/source-593f7cf-overlay"
runtime_tree="$root/stage1/source-be31323"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
media_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_endpoint_consensus_release_11f3ed1/source/md/action_editing/20260815_reward/model_gain_135096/review/media
motion_checkpoint="$root/stage1/train_row0_noop_varying_out3e4_cond3e4_s160_r1/checkpoint-00000160"
source_copy_root="$root/stage1/train_source_${tag}_sr50_out3e4_cond3e4_s80_r1"

if [ "$row" = row0 ]; then
  iid=7b88a1ca1f804f41
  instruction='A static camera shows a happy single grey French bulldog in a black harness, standing in a large pile of vibrant yellow autumn leaves in a sunlit park, facing the camera with its tongue out while its chest and the nearby leaves move subtly. The main French bulldog bends its hind legs, lowers its hips to the ground, settles into a stable sit facing the camera, and holds that seated pose. The shot stays continuous, the illumination remains stable, and the final frame is temporally coherent.'
else
  iid=a66e6818e4144928
  instruction='A static camera shows a single young woman in a low crouch on a city sidewalk beside a modern dark-glass building with horizontal metal bars, looking back over her shoulder, with tied dark hair, a black sports bra, grey trousers, and a blue-green plaid shirt in harsh sunlight and deep shadow. The main person presses through both feet, rises smoothly from the low crouch, straightens the legs and torso, faces the viewer, and holds upright. The shot stays continuous, the illumination remains stable, and the final frame is temporally coherent.'
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$source_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"

decode_one() {
  local step="$1"
  local source_copy_checkpoint="$source_copy_root/checkpoint-00000${step}"
  local output_dir="$root/stage1/infer/source_${tag}_sr50_s${step}_k100_${row}_r1"
  local scratch="/tmp/dense-flow-source-${tag}-${step}-${row}-140846-${SLURM_STEP_ID:-ssh}"
  test ! -e "$output_dir"
  mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton" "$output_dir"
  export MIOPEN_USER_DB_PATH="$scratch/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
  export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions"
  export TRITON_CACHE_DIR="$scratch/triton"
  "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$source_tree/methods/bernini_action_editing/infer_same_video_dense_flow_adapter_v1.py" \
    --dense-flow-checkpoint "$motion_checkpoint" \
    --dense-flow-scale 1.0 \
    --dense-flow-source-copy-checkpoint "$source_copy_checkpoint" \
    --dense-flow-source-copy-scale 1.0 \
    --dense-flow-source-copy-mode "$mode" \
    --flow-bundle "$root/flows/$iid.safetensors" \
    --bernini-root "$bernini_root" \
    --veomni-root "$veomni_root" \
    --checkpoint "$base_checkpoint" \
    --source-video "$media_root/$iid/source.mp4" \
    --instruction "$instruction" \
    --output "$output_dir/output.mp4" \
    --num-inference-steps 40 \
    --seed 2026081601 \
    --source-onset-policy hard1_every_step \
    --method-source-revision 593f7cf470ccdcfe11a643a21978a57cc78e2b8b \
    --method-source-archive-sha256 a02121cd57c66f35e063bc5bf8cbd4fa4efca428083c2bf1245b149fd6908d41
}

decode_one 020
decode_one 080
