#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh7-1b-gpu-226|auh7-1b-gpu-233|auh7-1b-gpu-268|auh7-1b-gpu-292|auh7-1b-gpu-293|auh7-1b-gpu-306|auh7-1b-gpu-315) ;;
  *) echo "online-anchor training is restricted to authorized existing allocations" >&2; exit 3 ;;
esac

profile="${ONLINE_ANCHOR_PROFILE:?action_noop, dynamic_static, hybrid, or no_anchor}"
case "$profile" in action_noop|dynamic_static|hybrid|no_anchor) ;; *) exit 4 ;; esac
visible="${ONLINE_ANCHOR_VISIBLE_DEVICES:?four physical GCDs required}"
experiment="${ONLINE_ANCHOR_EXPERIMENT:?unique output label required}"
case "$experiment" in *[!A-Za-z0-9_.-]*|'') echo "invalid experiment label" >&2; exit 5 ;; esac
max_steps="${ONLINE_ANCHOR_MAX_STEPS:-64}"
route_strength="${ONLINE_ANCHOR_ROUTE_STRENGTH:-0.25}"
teacher_route_strength="${ONLINE_ANCHOR_TEACHER_ROUTE_STRENGTH:-1.0}"
training_objective="${ONLINE_ANCHOR_TRAINING_OBJECTIVE:-target_fm}"
case "$training_objective" in target_fm|paired_delta_fm|real_source_teacher_delta|real_source_routed_teacher_delta|real_source_target_owned_routed_teacher_delta_v14r2) ;; *) exit 9 ;; esac
if [ -n "${ONLINE_ANCHOR_TRAINING_INTERFACE:-}" ]; then
  training_interface="$ONLINE_ANCHOR_TRAINING_INTERFACE"
elif [ "$training_objective" = real_source_teacher_delta ] || [ "$training_objective" = real_source_routed_teacher_delta ] || [ "$training_objective" = real_source_target_owned_routed_teacher_delta_v14r2 ]; then
  training_interface=first_phase_caption_i2v
else
  training_interface=mv2v_full_source
fi
case "$training_interface" in mv2v_full_source|first_phase_caption_i2v) ;; *) exit 10 ;; esac
if [ -n "${ONLINE_ANCHOR_PAIRED_TARGET_FM_WEIGHT:-}" ]; then
  paired_target_fm_weight="$ONLINE_ANCHOR_PAIRED_TARGET_FM_WEIGHT"
elif [ "$training_objective" = real_source_teacher_delta ] || [ "$training_objective" = real_source_routed_teacher_delta ] || [ "$training_objective" = real_source_target_owned_routed_teacher_delta_v14r2 ]; then
  paired_target_fm_weight=0
else
  paired_target_fm_weight=0.25
fi
real_source_manifest="${ONLINE_ANCHOR_REAL_SOURCE_MANIFEST:-}"
real_source_manifest_sha256="${ONLINE_ANCHOR_REAL_SOURCE_MANIFEST_SHA256:-}"
if [ -n "${ONLINE_ANCHOR_TEACHER_DELTA_MODE:-}" ]; then
  teacher_delta_mode="$ONLINE_ANCHOR_TEACHER_DELTA_MODE"
elif [ "$training_objective" = real_source_routed_teacher_delta ] || [ "$training_objective" = real_source_target_owned_routed_teacher_delta_v14r2 ]; then
  teacher_delta_mode=raw
else
  teacher_delta_mode=phase0_relative
fi
case "$teacher_delta_mode" in raw|phase0_relative) ;; *) exit 11 ;; esac
if [ -n "${ONLINE_ANCHOR_REPLAY_WEIGHT:-}" ]; then
  replay_weight="$ONLINE_ANCHOR_REPLAY_WEIGHT"
elif [ "$training_objective" = real_source_routed_teacher_delta ] || [ "$training_objective" = real_source_target_owned_routed_teacher_delta_v14r2 ]; then
  replay_weight=0.025
else
  replay_weight=0.25
fi
replay_prompt="${ONLINE_ANCHOR_REPLAY_PROMPT:-action}"
case "$replay_prompt" in action|noop|identity) ;; *) exit 7 ;; esac
routed_teacher_mode="${ONLINE_ANCHOR_ROUTED_TEACHER_MODE:-cross_caption_two_sided}"
case "$routed_teacher_mode" in same_action_route_only|cross_caption_two_sided) ;; *) exit 13 ;; esac
replay_combine_mode="${ONLINE_ANCHOR_REPLAY_COMBINE_MODE:-fixed_0025}"
case "$replay_combine_mode" in
  fixed_0025|first_order_safe|action_only|norm_balanced_005|norm_balanced_025|source_safe_cap025|source_halfspace_001|action_priority_pcgrad_010) ;;
  *) exit 14 ;;
esac
gradient_diagnostic_only="${ONLINE_ANCHOR_GRADIENT_DIAGNOSTIC_ONLY:-0}"
case "$gradient_diagnostic_only" in 0|1) ;; *) exit 15 ;; esac
gradient_diagnostic_args=()
if [ "$gradient_diagnostic_only" = 1 ]; then
  gradient_diagnostic_args+=(--gradient-diagnostic-only)
fi
if [ -n "${ONLINE_ANCHOR_SOURCE_VARIANT:-}" ]; then
  source_variant="$ONLINE_ANCHOR_SOURCE_VARIANT"
elif [ "$training_objective" = real_source_target_owned_routed_teacher_delta_v14r2 ]; then
  source_variant=not_applicable
else
  source_variant=mixed
fi
case "$source_variant" in noop|mixed|counterfactual4|not_applicable) ;; *) exit 8 ;; esac
learning_rate="${ONLINE_ANCHOR_LEARNING_RATE:-1e-5}"
if [ -n "${ONLINE_ANCHOR_ROUTE_OPERATOR:-}" ]; then
  route_operator="$ONLINE_ANCHOR_ROUTE_OPERATOR"
elif [ "$training_objective" = real_source_routed_teacher_delta ]; then
  route_operator=self_correspondence_kernel25
elif [ "$training_objective" = real_source_target_owned_routed_teacher_delta_v14r2 ]; then
  route_operator=self_target_owned_temporal_kernel_v14r2
else
  route_operator=cross_sparse
fi
case "$route_operator" in cross_sparse|self_temporal_kernel|self_target_gated_kernel25|self_correspondence_kernel25|self_target_owned_temporal_kernel_v14r2|self_target_owned_activity_kernel10_v14r2|self_target_owned_activity_kernel25_v14r2) ;; *) exit 6 ;; esac
seed="${ONLINE_ANCHOR_SEED:-2026081921}"

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
source_tree="${ONLINE_ANCHOR_SOURCE_TREE:-$stage/source-online-anchor-attention-training-v1}"
fallback_tree="$stage/source-sga-anc-training-v1"
runtime_tree="$stage/source-be31323"
release="$stage/online_anchor_attention_training_v1"
pair_manifest="${ONLINE_ANCHOR_PAIR_MANIFEST:-$stage/complex8_crossappearance_motion_v1/pairs_exact/manifest_cross.json}"
authoring="${ONLINE_ANCHOR_AUTHORING:-}"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
archive="${ONLINE_ANCHOR_METHOD_ARCHIVE:-$release/method-source.tar}"
revision_file="${ONLINE_ANCHOR_METHOD_REVISION_FILE:-$release/method-source.revision}"
output="$release/train_${experiment}"

test -f "$source_tree/methods/bernini_action_editing/train_online_anchor_attention_v1.py"
test -f "$source_tree/methods/bernini_action_editing/anchor_cross_attention_transport.py"
test -f "$source_tree/methods/bernini_action_editing/anchor_qk_transport.py"
test -f "$pair_manifest"
if [ "$training_interface" = first_phase_caption_i2v ]; then
  test -n "$authoring"
  test -f "$authoring"
fi
if [ "$training_objective" = real_source_teacher_delta ] || [ "$training_objective" = real_source_routed_teacher_delta ] || [ "$training_objective" = real_source_target_owned_routed_teacher_delta_v14r2 ]; then
  test -n "$real_source_manifest"
  test -f "$real_source_manifest"
  [[ "$real_source_manifest_sha256" =~ ^[0-9a-f]{64}$ ]] || exit 12
fi
test -f "$archive"
test -f "$revision_file"
test ! -e "$output"
revision="$(tr -d '\n' < "$revision_file")"
archive_sha="$(sha256sum "$archive" | awk '{print $1}')"

scratch="/tmp/online-anchor-fullgrid-${experiment}-${SLURM_JOB_ID:-existing}"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES="$visible"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
if [ "$training_objective" = real_source_target_owned_routed_teacher_delta_v14r2 ]; then
  # v14r2 is an immutable, self-contained closure. Missing modules must fail
  # instead of silently falling through to an older experimental source tree.
  export PYTHONPATH="$source_tree/methods/bernini_action_editing"
else
  export PYTHONPATH="$source_tree/methods/bernini_action_editing:$fallback_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"
fi

exec "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$source_tree/methods/bernini_action_editing/train_online_anchor_attention_v1.py" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$base_checkpoint" \
  --pair-manifest "$pair_manifest" \
  --authoring "$authoring" \
  --output "$output" \
  --profile "$profile" \
  --route-operator "$route_operator" \
  --max-steps "$max_steps" \
  --micro-records 2 \
  --source-variant "$source_variant" \
  --route-strength "$route_strength" \
  --teacher-route-strength "$teacher_route_strength" \
  --training-objective "$training_objective" \
  --training-interface "$training_interface" \
  --paired-target-fm-weight "$paired_target_fm_weight" \
  --real-source-manifest "$real_source_manifest" \
  --real-source-manifest-sha256 "$real_source_manifest_sha256" \
  --teacher-delta-mode "$teacher_delta_mode" \
  --routed-teacher-mode "$routed_teacher_mode" \
  --source-reconstruction-weight "$replay_weight" \
  --replay-combine-mode "$replay_combine_mode" \
  "${gradient_diagnostic_args[@]}" \
  --source-reconstruction-prompt "$replay_prompt" \
  --learning-rate "$learning_rate" \
  --seed "$seed" \
  --max-grad-norm 10 \
  --method-source-revision "$revision" \
  --method-source-archive-sha256 "$archive_sha"
