#!/usr/bin/env bash
# One SP4 endpoint-consensus-v3 training process inside parent Job 140846.

set -Eeuo pipefail

archive="${ACTION_ENDPOINT_SOURCE_ARCHIVE:?set ACTION_ENDPOINT_SOURCE_ARCHIVE}"
archive_sha="${ACTION_ENDPOINT_SOURCE_ARCHIVE_SHA256:?set archive SHA-256}"
revision="${ACTION_ENDPOINT_SOURCE_REVISION:?set source revision}"
manifest="${ACTION_ENDPOINT_SOURCE_MANIFEST:?set source-only manifest}"
manifest_sha="${ACTION_ENDPOINT_SOURCE_MANIFEST_SHA256:?set manifest SHA-256}"
cache="${ACTION_ENDPOINT_CACHE:?set sealed teacher cache}"
cache_sha="${ACTION_ENDPOINT_CACHE_SHA256:?set teacher cache SHA-256}"
output="${ACTION_ENDPOINT_OUTPUT:?set create-only output}"
arm="${ACTION_ENDPOINT_ARM:?set endpoint arm}"
max_steps="${ACTION_ENDPOINT_MAX_STEPS:-80}"
seed="${ACTION_ENDPOINT_SEED:-20260817}"

[[ "${revision}" =~ ^[0-9a-f]{40}$ ]]
[[ "${archive_sha}" =~ ^[0-9a-f]{64}$ ]]
[[ "${manifest_sha}" =~ ^[0-9a-f]{64}$ ]]
[[ "${cache_sha}" =~ ^[0-9a-f]{64}$ ]]
[[ "${seed}" == 20260817 && "${max_steps}" =~ ^[1-9][0-9]*$ ]]
[[ "${arm}" == endpoint_cell_band || "${arm}" == endpoint_consensus_band || \
   "${arm}" == endpoint_consensus_trust_001 || \
   "${arm}" == endpoint_consensus_trust_010 ]]
[[ -f "${archive}" && -f "${manifest}" && -f "${cache}" && ! -e "${output}" ]]
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]]
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha}" ]]
[[ "$(sha256sum "${cache}" | awk '{print $1}')" == "${cache_sha}" ]]

bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
scratch="${SLURM_TMPDIR:-/tmp}/action-endpoint-${SLURM_JOB_ID}-${SLURM_STEP_ID:-none}-${arm}-$$"
mkdir -p "${scratch}/source" "${scratch}/miopen-user" "${scratch}/miopen-custom" \
  "${scratch}/torch-extensions" "${scratch}/triton"
tar -xf "${archive}" -C "${scratch}/source"
for required in \
  full30_action_learning_v1.py \
  self_generated_action_endpoint_consensus_v3.py \
  train_lora.py \
  train_self_generated_action_endpoint_consensus_v3.py \
  train_self_generated_action_quotient_v1.py \
  train_self_generated_action_residual_margin_v2.py; do
  [[ -f "${scratch}/source/methods/bernini_action_editing/${required}" ]]
done

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MIOPEN_USER_DB_PATH="${scratch}/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="${scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${scratch}/torch-extensions" TRITON_CACHE_DIR="${scratch}/triton"

runner="${scratch}/source/methods/bernini_action_editing/train_self_generated_action_endpoint_consensus_v3.py"
exec "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${runner}" \
  --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" --source-manifest "${manifest}" \
  --source-manifest-sha256 "${manifest_sha}" --cache "${cache}" \
  --expected-cache-sha256 "${cache_sha}" --output "${output}" --arm "${arm}" \
  --slots 4 --max-steps "${max_steps}" --seed "${seed}" \
  --method-source-revision "${revision}" \
  --method-source-archive-sha256 "${archive_sha}"
