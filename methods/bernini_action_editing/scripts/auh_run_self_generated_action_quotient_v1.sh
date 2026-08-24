#!/usr/bin/env bash
# One SP4 cache or training process inside the retained Job 140846 allocation.

set -Eeuo pipefail

mode="${ACTION_QUOTIENT_MODE:?set ACTION_QUOTIENT_MODE}"
archive="${ACTION_QUOTIENT_SOURCE_ARCHIVE:?set ACTION_QUOTIENT_SOURCE_ARCHIVE}"
archive_sha="${ACTION_QUOTIENT_SOURCE_ARCHIVE_SHA256:?set ACTION_QUOTIENT_SOURCE_ARCHIVE_SHA256}"
revision="${ACTION_QUOTIENT_SOURCE_REVISION:?set ACTION_QUOTIENT_SOURCE_REVISION}"
manifest="${ACTION_QUOTIENT_SOURCE_MANIFEST:?set ACTION_QUOTIENT_SOURCE_MANIFEST}"
manifest_sha="${ACTION_QUOTIENT_SOURCE_MANIFEST_SHA256:?set ACTION_QUOTIENT_SOURCE_MANIFEST_SHA256}"
cache="${ACTION_QUOTIENT_CACHE:?set ACTION_QUOTIENT_CACHE}"
output="${ACTION_QUOTIENT_OUTPUT:?set ACTION_QUOTIENT_OUTPUT}"
seed="${ACTION_QUOTIENT_SEED:?set ACTION_QUOTIENT_SEED}"
arm="${ACTION_QUOTIENT_ARM:-action_only}"
slots="${ACTION_QUOTIENT_SLOTS:-4}"
limit_cells="${ACTION_QUOTIENT_LIMIT_CELLS:-0}"
max_steps="${ACTION_QUOTIENT_MAX_STEPS:-160}"

[[ "${mode}" == cache || "${mode}" == train ]]
[[ "${revision}" =~ ^[0-9a-f]{40}$ ]]
[[ "${archive_sha}" =~ ^[0-9a-f]{64}$ ]]
[[ "${manifest_sha}" =~ ^[0-9a-f]{64}$ ]]
[[ "${seed}" =~ ^[0-9]+$ && "${seed}" == 20260817 ]]
[[ -f "${archive}" && -f "${manifest}" ]]
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]]
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha}" ]]

cache_contract=()
if [[ "${mode}" == train ]]; then
  expected_cache_sha="${ACTION_QUOTIENT_EXPECTED_CACHE_SHA256:?set expected teacher cache SHA-256}"
  [[ "${expected_cache_sha}" =~ ^[0-9a-f]{64}$ && -f "${cache}" ]]
  [[ "$(sha256sum "${cache}" | awk '{print $1}')" == "${expected_cache_sha}" ]]
  cache_contract=(--expected-cache-sha256 "${expected_cache_sha}")
fi

bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

scratch="${SLURM_TMPDIR:-/tmp}/action-quotient-${SLURM_JOB_ID}-${SLURM_STEP_ID:-none}-${arm}-$$"
mkdir -p "${scratch}/source" "${scratch}/miopen-user" "${scratch}/miopen-custom" \
  "${scratch}/torch-extensions" "${scratch}/triton"
tar -xf "${archive}" -C "${scratch}/source"
runner="${scratch}/source/methods/bernini_action_editing/train_self_generated_action_quotient_v1.py"
[[ -f "${runner}" ]]

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MIOPEN_USER_DB_PATH="${scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${scratch}/torch-extensions"
export TRITON_CACHE_DIR="${scratch}/triton"

exec "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${runner}" --mode "${mode}" --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" --checkpoint "${checkpoint}" \
  --source-manifest "${manifest}" --source-manifest-sha256 "${manifest_sha}" \
  --cache "${cache}" "${cache_contract[@]}" --output "${output}" \
  --arm "${arm}" --slots "${slots}" --limit-cells "${limit_cells}" \
  --max-steps "${max_steps}" --seed "${seed}" \
  --method-source-revision "${revision}" --method-source-archive-sha256 "${archive_sha}"
