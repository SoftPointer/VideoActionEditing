#!/usr/bin/env bash
# Run one frozen-base or trained-LoRA evaluation video on exactly four visible GPUs.

set -Eeuo pipefail

mode="${REWARD_EVAL_MODE:?set REWARD_EVAL_MODE}"
iid="${REWARD_EVAL_IID:?set REWARD_EVAL_IID}"
source_video="${REWARD_EVAL_SOURCE_VIDEO:?set REWARD_EVAL_SOURCE_VIDEO}"
instruction="${REWARD_EVAL_INSTRUCTION:?set REWARD_EVAL_INSTRUCTION}"
output_video="${REWARD_EVAL_OUTPUT_VIDEO:?set REWARD_EVAL_OUTPUT_VIDEO}"
adapter_checkpoint="${REWARD_EVAL_ADAPTER_CHECKPOINT:-}"
seed="${REWARD_EVAL_SEED:-2026081601}"

[[ "${mode}" == frozen_base || "${mode}" == trained_adapter ]] || exit 2
[[ "${iid}" =~ ^[0-9a-f]{16}$ ]]
[[ "${seed}" =~ ^[0-9]{1,18}$ ]]
[[ -f "${source_video}" && ! -L "${source_video}" ]]
[[ ! -e "${output_video}" && ! -L "${output_video}" ]]
if [[ "${mode}" == trained_adapter ]]; then
  [[ -d "${adapter_checkpoint}" && ! -L "${adapter_checkpoint}" ]]
else
  [[ -z "${adapter_checkpoint}" ]]
fi

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_training_pairv5_20260815_v1
source_archive="${root}/release/source-a289922-infer.tar"
source_sha=aa9b911ba46a2cf9cf98e72714a2072ccf7f75ea056f047b7fa0fdd7ac2ca89b
source_revision=a289922b46a2755e9e5f35ed1ee0f3858b1c5cc3
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_sha}" ]]
scratch="${SLURM_TMPDIR:-/tmp}/reward-eval-${SLURM_JOB_ID}-${SLURM_STEP_ID:-none}-${mode}-${iid}-$$"
mkdir -p "${scratch}/source" "${scratch}/miopen-user" "${scratch}/miopen-custom" \
  "${scratch}/torch-extensions" "${scratch}/triton" "${output_video%/*}"
tar -xf "${source_archive}" -C "${scratch}/source"
method_root="${scratch}/source/methods/bernini_action_editing"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MIOPEN_USER_DB_PATH="${scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${scratch}/torch-extensions"
export TRITON_CACHE_DIR="${scratch}/triton"

command=(
  "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=4
  "${method_root}/infer_lora.py"
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --source-video "${source_video}"
  --instruction "${instruction}"
  --output "${output_video}"
  --num-inference-steps 40
  --seed "${seed}"
  --expected-bernini-commit 2d2b4591ac053ec25c6371b01a5a6746679e5793
  --expected-veomni-commit f90b3dc6fbb0ce693745223cc7a94064123dbf4d
  --expected-checkpoint-tree-sha256 6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
  --method-source-revision "${source_revision}"
  --method-source-archive-sha256 "${source_sha}"
)
if [[ "${mode}" == trained_adapter ]]; then
  command+=(--adapter-checkpoint "${adapter_checkpoint}")
else
  command+=(--base-only)
fi
"${command[@]}"

[[ -f "${output_video}" && -f "${output_video}.receipt.json" ]]
echo "PASS mode=${mode} iid=${iid} seed=${seed} output=${output_video}"
