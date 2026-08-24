#!/usr/bin/env bash
# One SP4 cache or residual-margin-v2 training process in Job 140846.

set -Eeuo pipefail

mode="${ACTION_RESIDUAL_MODE:?set ACTION_RESIDUAL_MODE}"
archive="${ACTION_RESIDUAL_SOURCE_ARCHIVE:?set ACTION_RESIDUAL_SOURCE_ARCHIVE}"
archive_sha="${ACTION_RESIDUAL_SOURCE_ARCHIVE_SHA256:?set ACTION_RESIDUAL_SOURCE_ARCHIVE_SHA256}"
revision="${ACTION_RESIDUAL_SOURCE_REVISION:?set ACTION_RESIDUAL_SOURCE_REVISION}"
manifest="${ACTION_RESIDUAL_SOURCE_MANIFEST:?set ACTION_RESIDUAL_SOURCE_MANIFEST}"
manifest_sha="${ACTION_RESIDUAL_SOURCE_MANIFEST_SHA256:?set ACTION_RESIDUAL_SOURCE_MANIFEST_SHA256}"
cache="${ACTION_RESIDUAL_CACHE:?set ACTION_RESIDUAL_CACHE}"
output="${ACTION_RESIDUAL_OUTPUT:?set ACTION_RESIDUAL_OUTPUT}"
seed="${ACTION_RESIDUAL_SEED:?set ACTION_RESIDUAL_SEED}"
arm="${ACTION_RESIDUAL_ARM:-margin_010}"
slots="${ACTION_RESIDUAL_SLOTS:-4}"
limit_cells="${ACTION_RESIDUAL_LIMIT_CELLS:-0}"
max_steps="${ACTION_RESIDUAL_MAX_STEPS:-160}"

[[ "${mode}" == cache || "${mode}" == train ]]
[[ "${revision}" =~ ^[0-9a-f]{40}$ ]]
[[ "${archive_sha}" =~ ^[0-9a-f]{64}$ ]]
[[ "${manifest_sha}" =~ ^[0-9a-f]{64}$ ]]
[[ "${seed}" == 20260817 ]]
[[ -f "${archive}" && -f "${manifest}" ]]
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]]
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha}" ]]

bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
scratch="${SLURM_TMPDIR:-/tmp}/action-residual-${SLURM_JOB_ID}-${SLURM_STEP_ID:-none}-${arm}-$$"
mkdir -p "${scratch}/source" "${scratch}/miopen-user" "${scratch}/miopen-custom" \
  "${scratch}/torch-extensions" "${scratch}/triton"
tar -xf "${archive}" -C "${scratch}/source"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MIOPEN_USER_DB_PATH="${scratch}/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="${scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${scratch}/torch-extensions" TRITON_CACHE_DIR="${scratch}/triton"

common=(
  --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" --checkpoint "${checkpoint}"
  --source-manifest "${manifest}"
  --cache "${cache}" --output "${output}" --slots "${slots}" --seed "${seed}"
  --method-source-revision "${revision}" --method-source-archive-sha256 "${archive_sha}"
)
if [[ "${mode}" == cache ]]; then
  runner="${scratch}/source/methods/bernini_action_editing/train_self_generated_action_quotient_v1.py"
  exec "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=4 \
    "${runner}" --mode cache --arm action_only --limit-cells "${limit_cells}" \
    --max-steps 1 "${common[@]}"
fi

expected_cache_sha="${ACTION_RESIDUAL_EXPECTED_CACHE_SHA256:?set expected cache SHA-256}"
[[ "${expected_cache_sha}" =~ ^[0-9a-f]{64}$ && -f "${cache}" ]]
[[ "$(sha256sum "${cache}" | awk '{print $1}')" == "${expected_cache_sha}" ]]
runner="${scratch}/source/methods/bernini_action_editing/train_self_generated_action_residual_margin_v2.py"
exec "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${runner}" --arm "${arm}" --expected-cache-sha256 "${expected_cache_sha}" \
  --source-manifest-sha256 "${manifest_sha}" --max-steps "${max_steps}" "${common[@]}"
