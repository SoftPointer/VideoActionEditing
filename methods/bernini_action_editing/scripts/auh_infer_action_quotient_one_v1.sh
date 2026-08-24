#!/usr/bin/env bash
# One source-only Bernini base/adapter rollout on one SP4 GPU island.

set -Eeuo pipefail

mode="${ACTION_QUOTIENT_EVAL_MODE:?set ACTION_QUOTIENT_EVAL_MODE}"
source_video="${ACTION_QUOTIENT_EVAL_SOURCE_VIDEO:?set ACTION_QUOTIENT_EVAL_SOURCE_VIDEO}"
instruction="${ACTION_QUOTIENT_EVAL_INSTRUCTION:?set ACTION_QUOTIENT_EVAL_INSTRUCTION}"
output="${ACTION_QUOTIENT_EVAL_OUTPUT:?set ACTION_QUOTIENT_EVAL_OUTPUT}"
adapter="${ACTION_QUOTIENT_EVAL_ADAPTER:-}"
source_onset_policy="${ACTION_QUOTIENT_EVAL_SOURCE_ONSET_POLICY:-none}"
archive="${ACTION_QUOTIENT_INFER_ARCHIVE:?set ACTION_QUOTIENT_INFER_ARCHIVE}"
archive_sha="${ACTION_QUOTIENT_INFER_ARCHIVE_SHA256:?set ACTION_QUOTIENT_INFER_ARCHIVE_SHA256}"
revision="${ACTION_QUOTIENT_INFER_REVISION:?set ACTION_QUOTIENT_INFER_REVISION}"

[[ "${mode}" == base || "${mode}" == adapter ]]
case "${source_onset_policy}" in
  none|hard1|ramp3|hard1_every_step) ;;
  *) false ;;
esac
[[ -f "${source_video}" && -f "${archive}" && ! -e "${output}" ]]
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]]
if [[ "${mode}" == adapter ]]; then [[ -d "${adapter}" ]]; else [[ -z "${adapter}" ]]; fi

bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
scratch="${SLURM_TMPDIR:-/tmp}/action-quotient-infer-${SLURM_JOB_ID}-${SLURM_STEP_ID:-none}-$$"
mkdir -p "${scratch}/source" "${scratch}/miopen-user" "${scratch}/miopen-custom" \
  "${scratch}/torch-extensions" "${scratch}/triton" "${output%/*}"
tar -xf "${archive}" -C "${scratch}/source"
runner="${scratch}/source/methods/bernini_action_editing/infer_lora.py"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MIOPEN_USER_DB_PATH="${scratch}/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="${scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${scratch}/torch-extensions" TRITON_CACHE_DIR="${scratch}/triton"

command=(
  "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=4 "${runner}"
  --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" --checkpoint "${checkpoint}"
  --source-video "${source_video}" --instruction "${instruction}" --output "${output}"
  --num-inference-steps 40 --seed 2026081601 --source-onset-policy "${source_onset_policy}"
  --expected-bernini-commit 2d2b4591ac053ec25c6371b01a5a6746679e5793
  --expected-veomni-commit f90b3dc6fbb0ce693745223cc7a94064123dbf4d
  --expected-checkpoint-tree-sha256 6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
  --method-source-revision "${revision}" --method-source-archive-sha256 "${archive_sha}"
)
if [[ "${mode}" == adapter ]]; then command+=(--adapter-checkpoint "${adapter}"); else command+=(--base-only); fi
exec "${command[@]}"
