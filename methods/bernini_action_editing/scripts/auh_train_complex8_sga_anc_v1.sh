#!/usr/bin/env bash
set -euo pipefail

node="$(hostname -s)"
case "$node" in
  auh7-1b-gpu-226|auh7-1b-gpu-233|auh7-1b-gpu-246|auh7-1b-gpu-247|auh7-1b-gpu-268|auh7-1b-gpu-292|auh7-1b-gpu-293|auh7-1b-gpu-306|auh7-1b-gpu-315) ;;
  *)
    echo "SGA/ANC training is restricted to explicitly authorized AUH nodes" >&2
    exit 3
    ;;
esac
profile="${SGA_ANC_PROFILE:-}"
case "$profile" in
  sgaanc)
    arm=sgaanc_tau02_uniform25_gain10
    sga_temperature=.02
    anc_uniform_mass=.25
    anchor_gain_weight=.10
    ;;
  hard_sga)
    arm=sga_tau01_hard_gain10
    sga_temperature=.01
    anc_uniform_mass=0
    anchor_gain_weight=.10
    ;;
  uniform_anc)
    arm=anc_uniform100_gain10
    sga_temperature=.02
    anc_uniform_mass=1
    anchor_gain_weight=.10
    ;;
  no_gain)
    arm=sgaanc_tau02_uniform25_gain00
    sga_temperature=.02
    anc_uniform_mass=.25
    anchor_gain_weight=0
    ;;
  *) echo "SGA_ANC_PROFILE must be sgaanc, hard_sga, uniform_anc, or no_gain" >&2; exit 4 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
source_tree="$stage/source-sga-anc-training-v1"
runtime_tree="$stage/source-be31323"
release="$stage/complex8_sga_anc_training_v1"
pair_release="$stage/complex8_crossappearance_motion_v1"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
pair_manifest="$pair_release/pairs_exact/manifest_cross.json"
bank_manifest="$release/sga_anc_motion_bank.json"
output="$release/train_${arm}_all30_r256_micro2_s32_v2"
archive="$release/method-source.tar"
revision_file="$release/method-source.revision"
test -f "$pair_manifest"
test -f "$bank_manifest"
test -f "$archive"
test -f "$revision_file"
test ! -e "$output"
revision="$(tr -d '\n' < "$revision_file")"
archive_sha="$(sha256sum "$archive" | awk '{print $1}')"

scratch="/tmp/complex8-sgaanc-${arm}-${SLURM_JOB_ID:-none}"
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

"$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$source_tree/methods/bernini_action_editing/train_same_video_dense_flow_adapter_v1.py" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$base_checkpoint" \
  --pair-manifest "$pair_manifest" \
  --sga-anc-bank-manifest "$bank_manifest" \
  --output "$output" \
  --max-steps 32 \
  --micro-records 2 \
  --row-repeat auto \
  --source-variant mixed \
  --training-noise-policy varying \
  --dense-flow-mode phase_attention_12x20 \
  --adapter-block-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 \
  --full-attention-lora \
  --learning-rate 1e-4 \
  --conditioner-learning-rate 1e-4 \
  --lora-learning-rate 1e-5 \
  --sga-temperature "$sga_temperature" \
  --anc-uniform-mass "$anc_uniform_mass" \
  --anchor-gain-weight "$anchor_gain_weight" \
  --anchor-gain-temperature .02 \
  --seed 2026081919 \
  --max-grad-norm 10 \
  --method-source-revision "$revision" \
  --method-source-archive-sha256 "$archive_sha"
test -f "$output/TRAINING_COMPLETE"
