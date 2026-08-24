#!/usr/bin/env bash
set -euo pipefail

[ "$(hostname -s)" = auh7-1b-gpu-246 ] || {
  echo "pair materialization is restricted to node 246" >&2
  exit 3
}
root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
source_tree="$stage/source-crossappearance-motion-v1"
old_source_tree="$stage/source-816e892-queryflow"
runtime_tree="$stage/source-be31323"
release="$stage/complex8_crossappearance_motion_v1"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
authoring="$stage/interaction_complex8_multianchor_authoring_v2.json"
anchor_root="$stage/interaction_complex8_multianchor_v2_r1"
output="$release/pairs_exact"
test -f "$release/EXTRACT_EXACT_auh7-1b-gpu-246_COMPLETE"
test -f "$release/EXTRACT_EXACT_auh7-1b-gpu-247_COMPLETE"
test ! -e "$output"

scratch=/tmp/complex8-crossappearance-pairs-140846
mkdir -p "$scratch/cache" "$scratch/miopen-user" "$scratch/miopen-custom"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export ROCR_VISIBLE_DEVICES=0 TORCH_HOME=/vast/users/guangyi.chen/.cache/torch
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export XDG_CACHE_HOME="$scratch/cache" MIOPEN_USER_DB_PATH="$scratch/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export PYTHONPATH="$source_tree/methods/bernini_action_editing:$old_source_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"

exec "$python_bin" -B "$source_tree/methods/bernini_action_editing/materialize_cross_appearance_motion_pairs_v1.py" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$base_checkpoint" \
  --authoring "$authoring" \
  --anchor-root "$anchor_root" \
  --matched-flow-root "$release/flows_matched_exact" \
  --cross-flow-root "$release/flows_cross_exact" \
  --output "$output"
