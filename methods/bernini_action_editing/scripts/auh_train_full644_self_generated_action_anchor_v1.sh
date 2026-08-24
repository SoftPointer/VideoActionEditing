#!/usr/bin/env bash
# Exact644 engineering training: action from self-generated anchor, identity from source.

set -Eeuo pipefail

if [[ "${1:-}" == "__rank_worker__" ]]; then
  shift
  rank="${LOCAL_RANK:?torchrun must set LOCAL_RANK}"
  cache_root="${SLURM_TMPDIR:-/tmp}/full644-anchor-${SLURM_JOB_ID}-${SLURM_STEP_ID:-none}-r${rank}"
  [[ ! -e "${cache_root}" ]]
  mkdir -p "${cache_root}"/{miopen-user,miopen-custom,tmp,xdg,triton,inductor,extensions,pycache}
  chmod 0700 "${cache_root}" "${cache_root}"/*
  export MIOPEN_USER_DB_PATH="${cache_root}/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${cache_root}/miopen-custom"
  export TMPDIR="${cache_root}/tmp" TMP="${cache_root}/tmp" TEMP="${cache_root}/tmp"
  export XDG_CACHE_HOME="${cache_root}/xdg"
  export TRITON_CACHE_DIR="${cache_root}/triton"
  export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor"
  export TORCH_EXTENSIONS_DIR="${cache_root}/extensions"
  export PYTHONPYCACHEPREFIX="${cache_root}/pycache"
  exec /vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12 "$@"
fi

archive=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_self_generated_anchor_v1_20260818/release_2ad469e5_9ea76025/full644_self_generated_anchor_train_v2.tar
archive_sha=c5c6ad1cea7d547de6f7ca267e951596ecc5e8d8acb0ef8829fad6b4adebcce9
revision=0da23167e4158daf0875095243fe3816c4e16960
manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_self_generated_anchor_v1_20260818/data/full644_action_anchor_manifest_v1.json
manifest_sha=61da995eb680b9fba7ab3b7d3b6041c7b51c7e95253c74e607ddab6fdd6a61aa
output="${FULL644_ACTION_OUTPUT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_self_generated_anchor_v1_20260818/runs/exact644_pcgrad_seed20260820_v1}"
training_seed="${FULL644_ACTION_SEED:-20260820}"
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

[[ ! -e "${output}" && ! -L "${output}" ]]
[[ "${training_seed}" =~ ^[1-9][0-9]*$ ]]
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]]
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha}" ]]

source_root="${SLURM_TMPDIR:-/tmp}/full644-anchor-source-${SLURM_JOB_ID}-${SLURM_STEP_ID:-none}"
[[ ! -e "${source_root}" ]]
mkdir -p "${source_root}"
tar -xf "${archive}" -C "${source_root}"
method_root="${source_root}/methods/bernini_action_editing"
trainer="${method_root}/train_self_generated_action_fullfield_v4.py"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1

exec "${python_bin}" -m torch.distributed.run --standalone --nproc-per-node=4 --no-python \
  "$0" __rank_worker__ "${trainer}" \
  --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" --source-manifest "${manifest}" \
  --source-manifest-sha256 "${manifest_sha}" --output "${output}" \
  --arm fullfield_action_noop_pcgrad_preserve \
  --max-steps 644 --micro-records 1 --seed "${training_seed}" --max-grad-norm 100 \
  --method-source-revision "${revision}" \
  --method-source-archive-sha256 "${archive_sha}"
