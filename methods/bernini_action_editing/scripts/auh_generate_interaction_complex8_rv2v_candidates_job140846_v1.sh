#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$1" in 0) devices=0,1,2,3; group=sp4-a ;; 1) devices=4,5,6,7; group=sp4-b ;; *) exit 2 ;; esac
case "$(hostname -s):$1" in
  auh7-1b-gpu-246:0) shard=0 ;; auh7-1b-gpu-246:1) shard=1 ;;
  auh7-1b-gpu-247:0) shard=2 ;; auh7-1b-gpu-247:1) shard=3 ;;
  auh7-1b-gpu-248:0) shard=4 ;; auh7-1b-gpu-248:1) shard=5 ;;
  auh7-1b-gpu-279:0) shard=6 ;; auh7-1b-gpu-279:1) shard=7 ;;
  *) echo "forbidden node/group outside Job 140846" >&2; exit 3 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
runtime_tree="$stage/source-be31323"
spec="$stage/interaction_complex8_rv2v_specs_v1/$(hostname -s).json"
output_root="$stage/interaction_complex8_rv2v_candidates_v1"
anchor_root="$stage/interaction_complex8_multianchor_v2_r1"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256
contract="$runtime_tree/methods/bernini_action_editing/pair_v5_native_rollout_spec.py"
runner="$runtime_tree/methods/bernini_action_editing/infer_pair_v5_native_rv2v4_rollout.py"

test -f "$spec"
test -f "$contract"
test -f "$runner"
spec_sha="$(sha256sum "$spec" | awk '{print $1}')"
mkdir -p "$output_root"

# Never overlap the preceding pure-T2V generation on this SP4 group.
while [ ! -f "$anchor_root/SHARD_${shard}_COMPLETE" ]; do sleep 15; done

scratch="/tmp/interaction-complex8-rv2v-${shard}-140846"
test ! -e "$scratch"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
"$python_bin" -B "$contract" --spec "$spec" --expected-sha256 "$spec_sha" --output-dir "$scratch/plan" >/dev/null

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ROCR_VISIBLE_DEVICES="$devices"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
export PYTHONPATH="$runtime_tree/methods/bernini_action_editing"

for envelope in "$scratch/plan/$group"/*.json; do
  candidate_id="$(jq -r .candidate.candidate_id "$envelope")"
  output="$output_root/$candidate_id"
  if [ -f "$output/pair-v5-rollout-receipt.json" ]; then
    echo "already complete: $candidate_id"
    continue
  fi
  test ! -e "$output"
  "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$runner" \
    --candidate-spec "$envelope" \
    --expected-root-spec-sha256 "$spec_sha" \
    --output-dir "$output" \
    --bernini-root "$bernini_root" \
    --veomni-root "$veomni_root" \
    --checkpoint "$checkpoint" \
    --checkpoint-content-manifest "$checkpoint_manifest" \
    --method-source-revision be3132312b77125313901c928b7aedcfc2c72c12 \
    --method-source-archive-sha256 958b9350e32b5459053a4aa62dff6334fb0b251f41e8e60dcd16643cef0f9d3e
  test -f "$output/rv2v.mp4"
  test -f "$output/rv2v.normalized-clean-latent.safetensors"
  test -f "$output/pair-v5-rollout-receipt.json"
done
touch "$output_root/SHARD_${shard}_COMPLETE"
