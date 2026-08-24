#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh7-1b-gpu-226|auh7-1b-gpu-233|auh7-1b-gpu-268|auh7-1b-gpu-292|auh7-1b-gpu-293|auh7-1b-gpu-306|auh7-1b-gpu-315) ;;
  *) echo "online-anchor training is restricted to authorized existing allocations" >&2; exit 3 ;;
esac

profile="${ONLINE_ANCHOR_PROFILE:?action_noop, dynamic_static, hybrid, or no_anchor}"
case "$profile" in
  action_noop|dynamic_static|hybrid|no_anchor) ;;
  *) echo "invalid ONLINE_ANCHOR_PROFILE" >&2; exit 4 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
source_tree="$stage/source-online-anchor-attention-training-v1"
fallback_tree="$stage/source-sga-anc-training-v1"
runtime_tree="$stage/source-be31323"
release="$stage/online_anchor_attention_training_v1"
pair_release="$stage/complex8_crossappearance_motion_v1"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
pair_manifest="$pair_release/pairs_exact/manifest_cross.json"
output="$release/train_${profile}_crossattn_r256_micro2_s8_v1"
archive="$release/method-source.tar"
revision_file="$release/method-source.revision"

test -f "$source_tree/methods/bernini_action_editing/train_online_anchor_attention_v1.py"
test -f "$source_tree/methods/bernini_action_editing/anchor_cross_attention_transport.py"
test -f "$pair_manifest"
test -f "$archive"
test -f "$revision_file"
test ! -e "$output"
revision="$(tr -d '\n' < "$revision_file")"
archive_sha="$(sha256sum "$archive" | awk '{print $1}')"

scratch="/tmp/online-anchor-train-${profile}-${SLURM_JOB_ID:-existing}"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES="${ONLINE_ANCHOR_VISIBLE_DEVICES:?four physical GCDs required}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
export PYTHONPATH="$source_tree/methods/bernini_action_editing:$fallback_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"

exec "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$source_tree/methods/bernini_action_editing/train_online_anchor_attention_v1.py" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$base_checkpoint" \
  --pair-manifest "$pair_manifest" \
  --output "$output" \
  --profile "$profile" \
  --max-steps 8 \
  --micro-records 2 \
  --source-variant mixed \
  --route-strength .25 \
  --source-reconstruction-weight .25 \
  --learning-rate 1e-5 \
  --seed 2026081921 \
  --max-grad-norm 10 \
  --method-source-revision "$revision" \
  --method-source-archive-sha256 "$archive_sha"

test -f "$output/TRAINING_COMPLETE"
