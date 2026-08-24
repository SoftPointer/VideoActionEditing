#!/usr/bin/env bash
set -euo pipefail

expected_job="${ONLINE_ANCHOR_JOB_ID:?Slurm allocation ID is required}"
expected_node="${ONLINE_ANCHOR_NODE:?short AUH node name is required}"
gpu_devices="${ONLINE_ANCHOR_GPU_DEVICES:-0,1,2,3}"
backward_scale="${V16R2_DIAGNOSTIC_BACKWARD_SCALE:?diagnostic backward scale is required}"
case "$expected_job" in *[!0-9]*|'') echo "invalid Slurm allocation ID" >&2; exit 2 ;; esac
case "$expected_node" in *[!A-Za-z0-9_.-]*|'') echo "invalid AUH node name" >&2; exit 2 ;; esac
case "$gpu_devices" in
  0,1,2,3|4,5,6,7) ;;
  *) echo "diagnostic requires one disjoint four-GPU group" >&2; exit 2 ;;
esac
case "$backward_scale" in
  1|0.0009765625) ;;
  *) echo "diagnostic scale must be 1 or 2^-10" >&2; exit 2 ;;
esac
if [ "$(hostname -s)" != "$expected_node" ]; then
  echo "diagnostic worker is restricted to $expected_node" >&2
  exit 3
fi
if [ "${SLURM_JOB_ID:-}" != "$expected_job" ]; then
  echo "diagnostic worker is restricted to allocation $expected_job" >&2
  exit 4
fi

source_tree="${ONLINE_ANCHOR_SOURCE_TREE:?frozen v16r2 source tree is required}"
diagnostic="${V16R2_S279_DIAGNOSTIC_SCRIPT:?diagnostic Python wrapper is required}"
output="${ONLINE_ANCHOR_OUTPUT:?unique diagnostic output sentinel is required}"
log_label="${ONLINE_ANCHOR_EXPERIMENT:?unique diagnostic label is required}"
case "$log_label" in *[!A-Za-z0-9_.-]*|'') echo "invalid diagnostic label" >&2; exit 5 ;; esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_dynamic_static_v16r2_full644_20260823"
full644_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_self_generated_anchor_v1_20260818/data/full644_action_anchor_manifest_v1.json
expected_manifest_sha=61da995eb680b9fba7ab3b7d3b6041c7b51c7e95253c74e607ddab6fdd6a61aa
method_prefix="$release/online-anchor-full644-dynamic-static-v16r2.method-source"
method_archive="$method_prefix.tar"
method_revision_file="$method_prefix.revision"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
s256_adapter="$release/train_full644_dynamic_static_activity25_pcgrad_actualdescent_v16r2_s644_job147881/checkpoint-00000256/adapter/adapter_model.safetensors"

test -d "$source_tree"
test -f "$diagnostic"
test -f "$full644_manifest"
test -f "$s256_adapter"
test -f "$method_archive"
test -f "$method_revision_file"
test -d "$base_checkpoint"
test ! -e "$output"
manifest_sha="$(sha256sum "$full644_manifest" | awk '{print $1}')"
test "$manifest_sha" = "$expected_manifest_sha"
method_revision="$(tr -d '\n' < "$method_revision_file")"
method_archive_sha="$(sha256sum "$method_archive" | awk '{print $1}')"
[[ "$method_revision" =~ ^[0-9a-f]{64}$ ]]
[[ "$method_archive_sha" =~ ^[0-9a-f]{64}$ ]]

scratch="/tmp/online-anchor-v16r2-s279-diagnostic-${log_label}-${SLURM_JOB_ID}"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES="$gpu_devices"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
export PYTHONPATH="$source_tree/methods/bernini_action_editing"
export V16R2_S256_ADAPTER="$s256_adapter"
export V16R2_DIAGNOSTIC_BACKWARD_SCALE="$backward_scale"

exec "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$diagnostic" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$base_checkpoint" \
  --pair-manifest "$full644_manifest" \
  --authoring "$full644_manifest" \
  --output "$output" \
  --profile dynamic_static \
  --route-operator self_target_owned_activity_kernel25_v14r2 \
  --max-steps 644 \
  --micro-records 2 \
  --source-variant not_applicable \
  --route-strength 0.25 \
  --teacher-route-strength 0.50 \
  --training-objective real_source_target_owned_routed_teacher_delta_v14r2 \
  --training-interface first_phase_caption_i2v \
  --paired-target-fm-weight 0 \
  --real-source-manifest "$full644_manifest" \
  --real-source-manifest-sha256 "$manifest_sha" \
  --full644-manifest-sha256 "$manifest_sha" \
  --teacher-delta-mode raw \
  --routed-teacher-mode same_action_route_only \
  --source-reconstruction-weight 0.025 \
  --replay-combine-mode action_priority_pcgrad_010 \
  --source-reconstruction-prompt action \
  --learning-rate 1e-5 \
  --seed 2026082302 \
  --max-grad-norm 10 \
  --method-source-revision "$method_revision" \
  --method-source-archive-sha256 "$method_archive_sha"
