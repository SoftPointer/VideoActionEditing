#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_INDEX" >&2; exit 2; fi
gpu="$1"
case "$(hostname -s)" in
  auh7-1b-gpu-246|auh7-1b-gpu-247|auh7-1b-gpu-248|auh7-1b-gpu-279) ;;
  *) echo "forbidden node: materialization is restricted to Job 140846 nodes" >&2; exit 3 ;;
esac
case "$gpu" in 0|1|2|3|4|5|6|7) ;; *) echo "GPU_INDEX must lie in [0,7]" >&2; exit 2 ;; esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
source_tree="$root/stage1/source-43ed61a-overlay"
prior_tree="$root/stage1/source-3e99529-overlay"
runtime_tree="$root/stage1/source-be31323"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
flow_root="$root/stage1/expanded_bank_v1/flows"
for iid in 311c82f83eca4a7f 31c34509415745ca 6d346c38cf504493 6ea45d35943742bb 99cde432839f4240; do
  test -f "$flow_root/$iid.safetensors"
  test -f "$flow_root/$iid.json"
done
output="$root/stage1/expanded_bank_v1/pairs_9row"
test ! -e "$output"

scratch="/tmp/expanded-bank-materialize-g${gpu}-140846"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES="$gpu"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
export PYTHONPATH="$source_tree/methods/bernini_action_editing:$prior_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"

exec "$python_bin" -B "$source_tree/methods/bernini_action_editing/materialize_expanded_self_generated_action_bank_v1.py" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$base_checkpoint" \
  --input-manifest "$source_tree/methods/bernini_action_editing/assets/expanded_self_generated_action_bank_v1.json" \
  --video-root "$root/stage1/expanded_bank_v1/source_videos" \
  --flow-root "$flow_root" \
  --base-pair-manifest "$root/stage1/same_video_pairs_r2/manifest.json" \
  --output "$output"
