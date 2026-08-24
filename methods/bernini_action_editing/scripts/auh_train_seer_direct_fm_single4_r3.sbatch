#!/bin/bash
#SBATCH --job-name=seer-direct-r3
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=12:00:00
#SBATCH --exclude=auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 077

EXP=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_seer_event_erasure_v1_20260813
METHOD_ROOT="${EXP}/staging/seer-event-erasure-b0-r1/methods/bernini_action_editing"
VAE="${EXP}/data/event-erasure-vae-r1"
FINAL="${EXP}/data/event-erasure-final-r1"
OUTPUT="${EXP}/runs/seer-direct-fm-4step-single4-r3"
PYTHON=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python
BERNINI=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
VEOMNI=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
CHECKPOINT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4

test "${SLURM_NNODES:?}" = 1
test "${SLURM_NTASKS:?}" = 1
test ! -e "${OUTPUT}"
test -f "${METHOD_ROOT}/train_seer_event_erasure_smoke.py"
test -d "${VAE}/shards"
test -f "${FINAL}/vae.summary.json"

cache="${SLURM_TMPDIR:-/tmp}/seer-direct-r3-${SLURM_JOB_ID}"
test ! -e "${cache}"
mkdir -m 0700 "${cache}"
cleanup() {
  find "${cache}" -xdev -mindepth 1 -delete 2>/dev/null || true
  rmdir "${cache}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export TMPDIR="${cache}"
export XDG_CACHE_HOME="${cache}/xdg"
export TORCH_EXTENSIONS_DIR="${cache}/torch-extensions"
export TRITON_CACHE_DIR="${cache}/triton"
export MIOPEN_USER_DB_PATH="${cache}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${cache}/miopen-custom"
mkdir -m 0700 "${XDG_CACHE_HOME}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

"${PYTHON}" -B -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  "${METHOD_ROOT}/train_seer_event_erasure_smoke.py" \
  --bernini-root "${BERNINI}" \
  --veomni-root "${VEOMNI}" \
  --checkpoint "${CHECKPOINT}" \
  --preprocessed-parquet-dir "${VAE}/shards" \
  --dataset-summary "${FINAL}/vae.summary.json" \
  --output "${OUTPUT}" \
  --max-steps 4 \
  --save-every 4 \
  --learning-rate 1e-6 \
  --seer-owner-spec "${METHOD_ROOT}/assets/seer_owner_core2_v1.json" \
  --expected-seer-owner-spec-sha256 42e0421404db59c198c6881ea4866344061e3b1ed03edb3c79f61efca68bb1f1 \
  --method-source-revision 6bae2f9a70fb851fd1bf87f5a01a1064a8fe8a6a \
  --method-source-archive-sha256 ec4064949b1cc8f4dba5d3c15fab375c9392a1407f9610ad01cee24c173bf822

test -f "${OUTPUT}/checkpoint-00000004/adapter/adapter_model.safetensors"
test -f "${OUTPUT}/checkpoint-00000004/receipt.json"
echo "SEER_DIRECT_SINGLE4_PASS output=${OUTPUT}"
