#!/usr/bin/env bash
#SBATCH --job-name=mev-a2b-smoke
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=02:00:00

set -Eeuo pipefail
umask 077

repo_root="${ACTION_GAP_REPO_ROOT:?set ACTION_GAP_REPO_ROOT}"
manifest="${ACTION_GAP_MANIFEST:?set ACTION_GAP_MANIFEST}"
python_bin="${ACTION_GAP_GENERATION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12}"
bernini_root="${ACTION_GAP_BERNINI_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/Bernini-2d2b4591}"
veomni_root="${ACTION_GAP_VEOMNI_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/VeOmni-f90b3dc6}"
checkpoint="${ACTION_GAP_CHECKPOINT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4}"
checkpoint_manifest="${ACTION_GAP_CHECKPOINT_MANIFEST:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_counterfactual_identity_orbit_v5_20260808_c099c6f/runtime/source_ea900d5/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256}"
inference_root="${ACTION_GAP_INFERENCE_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_preservation_only_20260814/staging/1871b029a06d4d1b}"
ffmpeg_bin="${ACTION_GAP_FFMPEG:-/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=offline WANDB_DISABLED=true PYTHONHASHSEED=0
export PYTHONPATH="${repo_root}"
export ROCR_VISIBLE_DEVICES=0,1,2,3
runtime_root="/tmp/action-gap-smoke-${SLURM_JOB_ID}"
mkdir -p "${runtime_root}/home" "${runtime_root}/tmp" "${runtime_root}/xdg" \
  "${runtime_root}/torch" "${runtime_root}/torch-kernels" \
  "${runtime_root}/miopen-user" "${runtime_root}/miopen-custom"
export HOME="${runtime_root}/home"
export TMPDIR="${runtime_root}/tmp"
export XDG_CACHE_HOME="${runtime_root}/xdg"
export TORCH_HOME="${runtime_root}/torch"
export PYTORCH_KERNEL_CACHE_PATH="${runtime_root}/torch-kernels"
export MIOPEN_USER_DB_PATH="${runtime_root}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${runtime_root}/miopen-custom"

"${python_bin}" -B -m methods.action_anchor_target_gap_audit.generation_controller \
  --manifest "${manifest}" --repo-root "${repo_root}" --inference-root "${inference_root}" --python "${python_bin}" \
  --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
  --ffmpeg "${ffmpeg_bin}" \
  --worker-index 0 --num-workers 16 --master-port-base 29710
