#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh7-1b-gpu-246|auh7-1b-gpu-247) ;;
  *) echo "forbidden node: frozen strict-state control is restricted to 246/247" >&2; exit 3 ;;
esac
if [ "$#" -ne 1 ]; then
  echo "usage: $0 GPU_GROUP" >&2
  exit 2
fi
case "$1" in
  0) devices=0,1,2,3 ;;
  1) devices=4,5,6,7 ;;
  *) echo "GPU_GROUP must be 0 or 1" >&2; exit 2 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
source_tree="$root/stage1/source-78d109b-overlay"
runtime_tree="$root/stage1/source-be31323"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
media_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_endpoint_consensus_release_11f3ed1/source/md/action_editing/20260815_reward/model_gain_135096/review/media
iid=a66e6818e4144928
instruction='A static camera shows a single young woman in a low crouch on a city sidewalk beside a modern dark-glass building with horizontal metal bars, looking back over her shoulder, with tied dark hair, a black sports bra, grey trousers, and a blue-green plaid shirt securely tied around her waist. The blue-green plaid shirt remains tied around her waist for the entire shot and the black sports bra remains fully visible and uncovered. The same woman presses through both feet, rises smoothly from the low crouch, straightens her legs and torso, turns to face the viewer, and holds upright facing the viewer. The framing, background, illumination, clothing state, and identity remain stable in one continuous temporally coherent shot.'
output_dir="$root/stage1/infer/frozen_u000_row3_strictstate_r3"
scratch="/tmp/frozen-row3-strictstate-g${1}-140846-v1"

test ! -e "$output_dir"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton" "$output_dir"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES="$devices"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export MIOPEN_USER_DB_PATH="$scratch/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions"
export TRITON_CACHE_DIR="$scratch/triton"
export PYTHONPATH="$source_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"

"$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$source_tree/methods/bernini_action_editing/infer_lora.py" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$base_checkpoint" \
  --base-only \
  --source-video "$media_root/$iid/source.mp4" \
  --instruction "$instruction" \
  --output "$output_dir/output.mp4" \
  --num-inference-steps 40 \
  --seed 2026081601 \
  --source-onset-policy hard1_every_step \
  --expected-bernini-commit 2d2b4591ac053ec25c6371b01a5a6746679e5793 \
  --expected-veomni-commit f90b3dc6fbb0ce693745223cc7a94064123dbf4d \
  --expected-checkpoint-tree-sha256 6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca \
  --method-source-revision 78d109b660b3135d0b8c2d00aadaba06b711d93f \
  --method-source-archive-sha256 4f4a8955016dab7925ca420152c05a80a6c773670325279f8e621fccdaf6e5de
