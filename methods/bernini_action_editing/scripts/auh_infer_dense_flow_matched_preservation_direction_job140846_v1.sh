#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh7-1b-gpu-246|auh7-1b-gpu-247) ;;
  *) echo "forbidden node: this experiment is restricted to 246/247" >&2; exit 3 ;;
esac
if [ "$#" -ne 1 ] || { [ "$1" != row0 ] && [ "$1" != row3 ]; }; then
  echo "usage: $0 row0|row3" >&2
  exit 2
fi
row="$1"

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
source_tree="$root/stage1/source-d373436-infer-overlay"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
media_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_endpoint_consensus_release_11f3ed1/source/md/action_editing/20260815_reward/model_gain_135096/review/media
main_checkpoint="$root/stage1/train_all4_mixed_sr4_varying_out3e4_cond3e4_s160_r1/checkpoint-00000080"
reference_checkpoint="$root/stage1/train_all4_mixed_varying_out3e4_cond3e4_s320_r1/checkpoint-00000080"

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
export PYTHONPATH="$source_tree/methods/bernini_action_editing:$root/stage1/source-be31323/methods/bernini_action_editing"

decode_one() {
  local beta="$1"
  local tag="$2"
  local output_dir="$root/stage1/infer/sr25_direction_b${tag}_${row}_step80_r1"
  local scratch="/tmp/dense-flow-sr25-direction-b${tag}-${row}-140846-${SLURM_STEP_ID}"
  test ! -e "$output_dir"
  mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton" "$output_dir"
  export MIOPEN_USER_DB_PATH="$scratch/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
  export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions"
  export TRITON_CACHE_DIR="$scratch/triton"
  "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$source_tree/methods/bernini_action_editing/infer_same_video_dense_flow_adapter_v1.py" \
    --dense-flow-checkpoint "$main_checkpoint" \
    --dense-flow-reference-checkpoint "$reference_checkpoint" \
    --dense-flow-reference-mix "$beta" \
    --dense-flow-scale 1.0 \
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
    --method-source-revision d373436cdc15c3c5d404e28c330996f8c5c4afdb \
    --method-source-archive-sha256 c7c67d8a2dc17bcdf55804cc2aff0419961f42f47c2216fc4f06e40d8aea46fb
}

decode_one 1.5 150
decode_one 2.0 200
