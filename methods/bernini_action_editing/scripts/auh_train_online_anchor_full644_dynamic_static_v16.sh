#!/usr/bin/env bash
set -euo pipefail

expected_job="${ONLINE_ANCHOR_JOB_ID:?Slurm allocation ID is required}"
expected_node="${ONLINE_ANCHOR_NODE:?short AUH node name is required}"
gpu_devices="${ONLINE_ANCHOR_GPU_DEVICES:-0,1,2,3}"
case "$expected_job" in *[!0-9]*|'') echo "invalid Slurm allocation ID" >&2; exit 2 ;; esac
case "$expected_node" in *[!A-Za-z0-9_.-]*|'') echo "invalid AUH node name" >&2; exit 2 ;; esac
IFS=, read -r gpu0 gpu1 gpu2 gpu3 gpu_extra <<<"$gpu_devices"
for gpu in "$gpu0" "$gpu1" "$gpu2" "$gpu3"; do
  case "$gpu" in 0|1|2|3|4|5|6|7) ;; *) echo "invalid v16 GPU device list" >&2; exit 2 ;; esac
done
if [ -n "${gpu_extra:-}" ] || [ "$gpu0" = "$gpu1" ] || [ "$gpu0" = "$gpu2" ] || \
   [ "$gpu0" = "$gpu3" ] || [ "$gpu1" = "$gpu2" ] || [ "$gpu1" = "$gpu3" ] || \
   [ "$gpu2" = "$gpu3" ]; then
  echo "v16 requires four distinct GPU device ordinals" >&2
  exit 2
fi
if [ "$(hostname -s)" != "$expected_node" ]; then
  echo "v16 worker is restricted to $expected_node" >&2
  exit 3
fi
if [ "${SLURM_JOB_ID:-}" != "$expected_job" ]; then
  echo "v16 worker is restricted to allocation $expected_job" >&2
  exit 4
fi

source_tree="${ONLINE_ANCHOR_SOURCE_TREE:?fresh frozen v16 source tree is required}"
release="${ONLINE_ANCHOR_RELEASE:?fresh v16 release root is required}"
experiment="${ONLINE_ANCHOR_EXPERIMENT:?unique v16 experiment label required}"
case "$experiment" in *[!A-Za-z0-9_.-]*|'') echo "invalid experiment label" >&2; exit 5 ;; esac
max_steps="${ONLINE_ANCHOR_MAX_STEPS:?exact644 required}"
case "$max_steps" in 644) ;; *) echo "v16 requires one continuous exact644 run" >&2; exit 6 ;; esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
full644_manifest="${ONLINE_ANCHOR_FULL644_MANIFEST:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_self_generated_anchor_v1_20260818/data/full644_action_anchor_manifest_v1.json}"
expected_manifest_sha=61da995eb680b9fba7ab3b7d3b6041c7b51c7e95253c74e607ddab6fdd6a61aa
method_archive="${ONLINE_ANCHOR_METHOD_ARCHIVE:-$release/online-anchor-full644-dynamic-static-v16.method-source.tar}"
method_revision_file="${ONLINE_ANCHOR_METHOD_REVISION_FILE:-$release/online-anchor-full644-dynamic-static-v16.method-source.revision}"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
trainer="$source_tree/methods/bernini_action_editing/train_online_anchor_attention_full644_dynamic_static_v16.py"
output="$release/train_${experiment}"

test -d "$source_tree"
test ! -L "$source_tree"
test -d "$release"
test ! -L "$release"
test -f "$trainer"
test ! -L "$trainer"
test -f "$source_tree/methods/bernini_action_editing/train_online_anchor_attention_dynamic_static_v15r2.py"
test -f "$source_tree/methods/bernini_action_editing/train_online_anchor_attention_dynamic_static_v15.py"
test -f "$source_tree/methods/bernini_action_editing/train_online_anchor_attention_v1.py"
test -f "$source_tree/methods/bernini_action_editing/anchor_qk_transport.py"
test -f "$full644_manifest"
test ! -L "$full644_manifest"
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

scratch="/tmp/online-anchor-full644-dynamic-static-v16-${experiment}-${SLURM_JOB_ID}"
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

exec "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$trainer" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$base_checkpoint" \
  --pair-manifest "$full644_manifest" \
  --authoring "$full644_manifest" \
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
