#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 frozen|sgaanc|hard_sga|uniform_anc|no_gain STEP EVENT" >&2
  exit 2
fi
profile="$1"
step="$2"
event="$3"
case "$profile" in
  frozen) arm="" ;;
  sgaanc) arm=sgaanc_tau02_uniform25_gain10 ;;
  hard_sga) arm=sga_tau01_hard_gain10 ;;
  uniform_anc) arm=anc_uniform100_gain10 ;;
  no_gain) arm=sgaanc_tau02_uniform25_gain00 ;;
  *) echo "unsupported profile: $profile" >&2; exit 2 ;;
esac
case "$step" in 0|1|5|10|20|32) ;; *) echo "unsupported checkpoint step" >&2; exit 2 ;; esac
if [ "$profile" = frozen ] && [ "$step" -ne 0 ]; then
  echo "frozen profile requires step 0" >&2
  exit 2
fi
if [ "$profile" != frozen ] && [ "$step" -eq 0 ]; then
  echo "trained profile requires a positive checkpoint step" >&2
  exit 2
fi
case "$event" in 0|1|2|3|4|5|6|7) ;; *) echo "event must be in 0..7" >&2; exit 2 ;; esac
case "$(hostname -s)" in
  auh7-1b-gpu-226|auh7-1b-gpu-233|auh7-1b-gpu-246|auh7-1b-gpu-247|auh7-1b-gpu-268|auh7-1b-gpu-292|auh7-1b-gpu-293|auh7-1b-gpu-306|auh7-1b-gpu-315) ;;
  *) echo "checkpoint inference is restricted to explicitly authorized AUH nodes" >&2; exit 3 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
source_tree="$stage/source-sga-anc-training-v1"
runtime_tree="$stage/source-be31323"
release="$stage/complex8_sga_anc_training_v1"
pair_release="$stage/complex8_crossappearance_motion_v1"
pair_manifest="$pair_release/pairs_exact/manifest_cross.json"
candidate_root="$stage/interaction_complex8_rv2v_candidates_v1"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
event_pad="$(printf '%02d' "$event")"
candidate_receipt="$candidate_root/complex8-e${event_pad}-rv2v-s0/pair-v5-rollout-receipt.json"
iid="e${event_pad}-v0"
source="$(jq -er '.candidate.source_video' "$candidate_receipt")"
instruction="$(jq -er '.candidate.complete_caption' "$candidate_receipt")"
seed="$(jq -er '.candidate.seed' "$candidate_receipt")"
flow="$(jq -er --arg iid "$iid" '.rows[] | select(.iid == $iid) | .flow_bundle' "$pair_manifest")"
test -f "$source"
test -f "$flow"

if [ "$profile" = frozen ]; then
  output_dir="$release/decode_v1/frozen/event_${event_pad}/step_0000"
else
  train_root="$release/train_${arm}_all30_r256_micro2_s32_v2"
  checkpoint="$train_root/checkpoint-$(printf '%08d' "$step")"
  test -f "$checkpoint/adapter_model.safetensors"
  test -f "$checkpoint/adapter/adapter_model.safetensors"
  jq -e --argjson step "$step" \
    '.global_step == $step and .trainable_parameter_count == 212551680' \
    "$checkpoint/receipt.json" >/dev/null
  output_dir="$release/decode_v1/$profile/event_${event_pad}/step_$(printf '%04d' "$step")"
fi
test ! -e "$output_dir"
mkdir -p "$output_dir"
output="$output_dir/output.mp4"

scratch="/tmp/complex8-sgaanc-infer-${profile}-s${step}-e${event_pad}-${SLURM_JOB_ID:-none}"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES="${SGA_ANC_VISIBLE_DEVICES:?four physical GCDs required}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
export PYTHONPATH="$source_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"
revision="$(tr -d '\n' < "$release/method-source.revision")"
archive_sha="$(sha256sum "$release/method-source.tar" | awk '{print $1}')"

common=(
  --bernini-root "$bernini_root"
  --veomni-root "$veomni_root"
  --checkpoint "$base_checkpoint"
  --source-video "$source"
  --instruction "$instruction"
  --output "$output"
  --num-inference-steps 40
  --seed "$seed"
  --source-onset-policy hard1_every_step
  --method-source-revision "$revision"
  --method-source-archive-sha256 "$archive_sha"
)

if [ "$profile" = frozen ]; then
  exec "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$runtime_tree/methods/bernini_action_editing/infer_lora.py" --base-only "${common[@]}"
fi

exec "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$source_tree/methods/bernini_action_editing/infer_same_video_dense_flow_adapter_v1.py" \
  --dense-flow-checkpoint "$checkpoint" \
  --dense-flow-scale 1.0 \
  --flow-bundle "$flow" \
  "${common[@]}"
