#!/usr/bin/env bash
# One matched source-only rollout on an SP4 island with per-rank cache isolation.

set -Eeuo pipefail

mode="${FULL644_EVAL_MODE:?}"
source_video="${FULL644_EVAL_SOURCE_VIDEO:?}"
instruction="${FULL644_EVAL_INSTRUCTION:?}"
output="${FULL644_EVAL_OUTPUT:?}"
adapter="${FULL644_EVAL_ADAPTER:-}"
archive="${FULL644_EVAL_INFER_ARCHIVE:?}"
archive_sha="${FULL644_EVAL_INFER_ARCHIVE_SHA256:?}"
revision="${FULL644_EVAL_INFER_REVISION:?}"

[[ "${mode}" == base || "${mode}" == adapter ]]
[[ -f "${source_video}" && -f "${archive}" && ! -e "${output}" ]]
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]]
if [[ "${mode}" == adapter ]]; then [[ -d "${adapter}" ]]; else [[ -z "${adapter}" ]]; fi

bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
scratch="${SLURM_TMPDIR:-/tmp}/full644-exact28-infer-${SLURM_JOB_ID}-${SLURM_STEP_ID:-none}-$$"
mkdir -p "${scratch}/source" "${output%/*}"
chmod 0700 "${scratch}"
tar -xf "${archive}" -C "${scratch}/source"
runner="${scratch}/source/methods/bernini_action_editing/infer_lora.py"
rank_wrapper="${scratch}/rank-wrapper.sh"
cat >"${rank_wrapper}" <<'RANK'
#!/usr/bin/env bash
set -Eeuo pipefail
: "${LOCAL_RANK:?}"
root="${FULL644_RANK_CACHE_ROOT:?}/rank-${LOCAL_RANK}"
[[ ! -e "${root}" ]]
mkdir -m 0700 "${root}"
for d in miopen-user miopen-custom xdg tmp triton inductor extensions pycache; do
  mkdir -m 0700 "${root}/${d}"
done
export MIOPEN_USER_DB_PATH="${root}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${root}/miopen-custom"
export XDG_CACHE_HOME="${root}/xdg" TMPDIR="${root}/tmp" TMP="${root}/tmp" TEMP="${root}/tmp"
export TRITON_CACHE_DIR="${root}/triton" TORCHINDUCTOR_CACHE_DIR="${root}/inductor"
export TORCH_EXTENSIONS_DIR="${root}/extensions" PYTHONPYCACHEPREFIX="${root}/pycache"
exec "$@"
RANK
chmod 0500 "${rank_wrapper}"

export FULL644_RANK_CACHE_ROOT="${scratch}/rank-caches"
mkdir -m 0700 "${FULL644_RANK_CACHE_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1

command=(
  "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=4 --no-python
  "${rank_wrapper}" "${python_bin}" "${runner}"
  --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" --checkpoint "${checkpoint}"
  --source-video "${source_video}" --instruction "${instruction}" --output "${output}"
  --num-inference-steps 40 --seed 2026081801 --source-onset-policy hard1_every_step
  --expected-bernini-commit 2d2b4591ac053ec25c6371b01a5a6746679e5793
  --expected-veomni-commit f90b3dc6fbb0ce693745223cc7a94064123dbf4d
  --expected-checkpoint-tree-sha256 6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
  --method-source-revision "${revision}" --method-source-archive-sha256 "${archive_sha}"
)
if [[ "${mode}" == adapter ]]; then
  command+=(--adapter-checkpoint "${adapter}")
else
  command+=(--base-only)
fi
exec "${command[@]}"
