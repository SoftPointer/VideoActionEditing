#!/usr/bin/env bash
set -euo pipefail

expected_job=149363
expected_node=auh7-1b-gpu-312
if [ "$(hostname -s)" != "$expected_node" ]; then
  echo "v15r2 worker is restricted to $expected_node" >&2
  exit 3
fi
if [ "${SLURM_JOB_ID:-}" != "$expected_job" ]; then
  echo "v15r2 worker is restricted to allocation $expected_job" >&2
  exit 4
fi

experiment="${ONLINE_ANCHOR_EXPERIMENT:?unique v15r2 experiment label required}"
case "$experiment" in *[!A-Za-z0-9_.-]*|'') echo "invalid experiment label" >&2; exit 5 ;; esac
max_steps="${ONLINE_ANCHOR_MAX_STEPS:?8 or 32 required}"
case "$max_steps" in 8|32) ;; *) echo "v15r2 permits 8 or 32 steps" >&2; exit 6 ;; esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
source_tree="${ONLINE_ANCHOR_SOURCE_TREE:-$stage/source-online-anchor-targetowned-qk-routed-teacher-v15r2-collinear-fallback-20260823}"
release="${ONLINE_ANCHOR_RELEASE:-$stage/online_anchor_dynamic_static_v15r2_20260823}"
pair_manifest="${ONLINE_ANCHOR_PAIR_MANIFEST:-$stage/complex8_crossappearance_motion_v1/pairs_exact/manifest_cross.json}"
authoring="${ONLINE_ANCHOR_AUTHORING:-$source_tree/methods/bernini_action_editing/assets/interaction_complex8_multianchor_authoring_v2.json}"
real_source_manifest="${ONLINE_ANCHOR_REAL_SOURCE_MANIFEST:-$stage/online_anchor_attention_training_v1/complex8_real_source_latents_v13/manifest.json}"
expected_real_source_sha=8b0a9d7fd8ccc9d8b555a66f8efe6d2e5d91880f81f5e7b892e123d88235bc63
method_archive="${ONLINE_ANCHOR_METHOD_ARCHIVE:-$release/online-anchor-dynamic-static-v15r2.method-source.tar}"
method_revision_file="${ONLINE_ANCHOR_METHOD_REVISION_FILE:-$release/online-anchor-dynamic-static-v15r2.method-source.revision}"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
trainer="$source_tree/methods/bernini_action_editing/train_online_anchor_attention_dynamic_static_v15r2.py"
output="$release/train_${experiment}"

test -f "$trainer"
test ! -L "$trainer"
test -f "$source_tree/methods/bernini_action_editing/train_online_anchor_attention_dynamic_static_v15.py"
test -f "$source_tree/methods/bernini_action_editing/train_online_anchor_attention_v1.py"
test -f "$source_tree/methods/bernini_action_editing/anchor_qk_transport.py"
test -f "$pair_manifest"
test -f "$authoring"
test -f "$real_source_manifest"
test -f "$method_archive"
test -f "$method_revision_file"
test -d "$base_checkpoint"
test ! -e "$output"
real_source_sha="$(sha256sum "$real_source_manifest" | awk '{print $1}')"
test "$real_source_sha" = "$expected_real_source_sha"
method_revision="$(tr -d '\n' < "$method_revision_file")"
method_archive_sha="$(sha256sum "$method_archive" | awk '{print $1}')"
[[ "$method_revision" =~ ^[0-9a-f]{64}$ ]]
[[ "$method_archive_sha" =~ ^[0-9a-f]{64}$ ]]

scratch="/tmp/online-anchor-dynamic-static-v15r2-${experiment}-${SLURM_JOB_ID}"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES=0,1,2,3
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
export PYTHONPATH="$source_tree/methods/bernini_action_editing"

exec "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$trainer" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$base_checkpoint" \
  --pair-manifest "$pair_manifest" \
  --authoring "$authoring" \
  --output "$output" \
  --profile dynamic_static \
  --route-operator self_target_owned_activity_kernel25_v14r2 \
  --max-steps "$max_steps" \
  --micro-records 2 \
  --source-variant not_applicable \
  --route-strength 0.25 \
  --teacher-route-strength 0.50 \
  --training-objective real_source_target_owned_routed_teacher_delta_v14r2 \
  --training-interface first_phase_caption_i2v \
  --paired-target-fm-weight 0 \
  --real-source-manifest "$real_source_manifest" \
  --real-source-manifest-sha256 "$real_source_sha" \
  --teacher-delta-mode raw \
  --routed-teacher-mode same_action_route_only \
  --source-reconstruction-weight 0.025 \
  --replay-combine-mode action_priority_pcgrad_010 \
  --source-reconstruction-prompt action \
  --learning-rate 1e-5 \
  --seed 2026082301 \
  --max-grad-norm 10 \
  --method-source-revision "$method_revision" \
  --method-source-archive-sha256 "$method_archive_sha"
