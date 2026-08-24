#!/usr/bin/env bash
# Run one sequential two-arm SP4 shard inside Job 140846.

set -Eeuo pipefail

archive="${FLOW_STAGE0_SOURCE_ARCHIVE:?set source archive}"
archive_sha="${FLOW_STAGE0_SOURCE_ARCHIVE_SHA256:?set source archive SHA-256}"
revision="${FLOW_STAGE0_SOURCE_REVISION:?set source revision}"
manifest="${FLOW_STAGE0_MANIFEST:?set manifest}"
flow_root="${FLOW_STAGE0_FLOW_ROOT:?set flow root}"
output_root="${FLOW_STAGE0_OUTPUT_ROOT:?set fresh shard output root}"
arms="${FLOW_STAGE0_ARMS:?set space-separated arms}"
port_base="${FLOW_STAGE0_MASTER_PORT_BASE:?set master port base}"

[[ -f "${archive}" && -f "${manifest}" && -d "${flow_root}" && ! -e "${output_root}" ]]
[[ "${archive_sha}" =~ ^[0-9a-f]{64}$ && "${revision}" =~ ^[0-9a-f]{40}$ ]]
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]]
[[ "${port_base}" =~ ^[1-9][0-9]{3,4}$ ]]

python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
scratch="${SLURM_TMPDIR:-/tmp}/flow-stage0-infer-${SLURM_JOB_ID}-${SLURM_STEP_ID:-none}-$$"
mkdir -p "${scratch}/source" "${scratch}/miopen-user" "${scratch}/miopen-custom" \
  "${scratch}/torch-extensions" "${scratch}/triton" "${output_root}"
tar -xf "${archive}" -C "${scratch}/source"
method_root="${scratch}/source/methods/bernini_action_editing"
controller="${method_root}/run_flow_noise_action_stage0_v1.py"
[[ -f "${controller}" ]]

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MIOPEN_USER_DB_PATH="${scratch}/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="${scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${scratch}/torch-extensions" TRITON_CACHE_DIR="${scratch}/triton"

read -r -a arm_array <<<"${arms}"
exec "${python_bin}" -B "${controller}" \
  --method-root "${method_root}" --manifest "${manifest}" \
  --flow-root "${flow_root}" --output-root "${output_root}" \
  --arms "${arm_array[@]}" --seed 2026081601 \
  --method-source-revision "${revision}" \
  --method-source-archive-sha256 "${archive_sha}" \
  --master-port-base "${port_base}"
