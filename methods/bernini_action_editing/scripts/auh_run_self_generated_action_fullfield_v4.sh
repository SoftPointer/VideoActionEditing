#!/usr/bin/env bash
# One SP4 rank256/full30 V4 training process inside parent Job 140846.

set -Eeuo pipefail

archive="${ACTION_FULLFIELD_SOURCE_ARCHIVE:?set source archive}"
archive_sha="${ACTION_FULLFIELD_SOURCE_ARCHIVE_SHA256:?set source archive SHA-256}"
revision="${ACTION_FULLFIELD_SOURCE_REVISION:?set source revision}"
manifest="${ACTION_FULLFIELD_SOURCE_MANIFEST:?set source manifest}"
manifest_sha="${ACTION_FULLFIELD_SOURCE_MANIFEST_SHA256:?set source manifest SHA-256}"
output="${ACTION_FULLFIELD_OUTPUT:?set fresh output}"
arm="${ACTION_FULLFIELD_ARM:?set V4 arm}"
max_steps="${ACTION_FULLFIELD_MAX_STEPS:-40}"
micro_records="${ACTION_FULLFIELD_MICRO_RECORDS:-1}"
overfit_row="${ACTION_FULLFIELD_OVERFIT_ROW:-}"

[[ "${archive_sha}" =~ ^[0-9a-f]{64}$ && "${revision}" =~ ^[0-9a-f]{40}$ ]]
[[ "${manifest_sha}" =~ ^[0-9a-f]{64}$ ]]
[[ -f "${archive}" && -f "${manifest}" && ! -e "${output}" ]]
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]]
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha}" ]]
case "${arm}" in
  direct_anchor_sft|fullfield_action_noop|fullfield_action_noop_pcgrad_preserve|source_carrier_sft) ;;
  *) false ;;
esac
[[ "${max_steps}" =~ ^[1-9][0-9]*$ && "${micro_records}" =~ ^[1-8]$ ]]
[[ -z "${overfit_row}" || "${overfit_row}" =~ ^[0-3]$ ]]

bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
scratch="${SLURM_TMPDIR:-/tmp}/action-fullfield-${SLURM_JOB_ID}-${SLURM_STEP_ID:-none}-${arm}-$$"
mkdir -p "${scratch}/source" "${scratch}/miopen-user" "${scratch}/miopen-custom" \
  "${scratch}/torch-extensions" "${scratch}/triton"
tar -xf "${archive}" -C "${scratch}/source"
method_root="${scratch}/source/methods/bernini_action_editing"
for required in \
  full30_action_learning_v1.py \
  self_generated_action_fullfield_v4.py \
  self_generated_action_quotient_v1.py \
  train_lora.py \
  train_self_generated_action_fullfield_v4.py \
  train_self_generated_action_quotient_v1.py; do
  [[ -f "${method_root}/${required}" ]]
done

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MIOPEN_USER_DB_PATH="${scratch}/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="${scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${scratch}/torch-extensions" TRITON_CACHE_DIR="${scratch}/triton"

command=(
  "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=4
  "${method_root}/train_self_generated_action_fullfield_v4.py"
  --bernini-root "${bernini_root}" --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}" --source-manifest "${manifest}"
  --source-manifest-sha256 "${manifest_sha}" --output "${output}"
  --arm "${arm}" --max-steps "${max_steps}" --micro-records "${micro_records}"
  --seed 20260820 --max-grad-norm 100
  --method-source-revision "${revision}"
  --method-source-archive-sha256 "${archive_sha}"
)
if [[ -n "${overfit_row}" ]]; then command+=(--overfit-row "${overfit_row}"); fi
exec "${command[@]}"
