#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh7-1b-gpu-226|auh7-1b-gpu-233|auh7-1b-gpu-268|auh7-1b-gpu-292|auh7-1b-gpu-293|auh7-1b-gpu-306|auh7-1b-gpu-315) ;;
  *) echo "generated-counterfactual materialization is restricted to authorized existing allocations" >&2; exit 3 ;;
esac

device="${GENERATED_PAIR_VISIBLE_DEVICE:-7}"
case "$device" in 0|1|2|3|4|5|6|7) ;; *) echo "invalid physical GCD" >&2; exit 4 ;; esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
source_tree="$stage/source-online-anchor-attention-training-v1"
old_source_tree="$stage/source-816e892-queryflow"
runtime_tree="$stage/source-be31323"
release="$stage/complex8_crossappearance_motion_v1"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
authoring="$stage/interaction_complex8_multianchor_authoring_v2.json"
anchor_root="$stage/interaction_complex8_multianchor_v2_r1"
output="$release/pairs_generated_counterfactual_v2"

test -f "$source_tree/methods/bernini_action_editing/materialize_cross_appearance_motion_pairs_v1.py"
test -f "$authoring"
test -d "$anchor_root"
test -d "$release/flows_matched_exact"
test -d "$release/flows_cross_exact"
test ! -e "$output"

scratch="/tmp/complex8-generated-counterfactual-pairs-${SLURM_JOB_ID:-existing}"
mkdir -p "$scratch/cache" "$scratch/miopen-user" "$scratch/miopen-custom"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export ROCR_VISIBLE_DEVICES="$device" TORCH_HOME=/vast/users/guangyi.chen/.cache/torch
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
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
  --source-authoring-mode generated_counterfactual \
  --output "$output"
