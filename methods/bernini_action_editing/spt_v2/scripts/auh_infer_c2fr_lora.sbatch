#!/usr/bin/env bash
#SBATCH --job-name=bern-c2fr-lora81
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=12:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_archive="${BERNINI_C2FR_LORA_SOURCE_ARCHIVE:?set BERNINI_C2FR_LORA_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_C2FR_LORA_SOURCE_ARCHIVE_SHA256:?set BERNINI_C2FR_LORA_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_C2FR_LORA_SOURCE_REVISION:?set BERNINI_C2FR_LORA_SOURCE_REVISION}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
bernini_archive="${BERNINI_OFFICIAL_ARCHIVE:?set BERNINI_OFFICIAL_ARCHIVE}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
adapter_checkpoint="${BERNINI_C2FR_LORA_ADAPTER:?set BERNINI_C2FR_LORA_ADAPTER}"
source_video="${BERNINI_C2FR_LORA_SOURCE_VIDEO:?set BERNINI_C2FR_LORA_SOURCE_VIDEO}"
instruction="${BERNINI_C2FR_LORA_INSTRUCTION:?set BERNINI_C2FR_LORA_INSTRUCTION}"
output_video="${BERNINI_C2FR_LORA_OUTPUT:?set BERNINI_C2FR_LORA_OUTPUT}"
seed="${BERNINI_C2FR_LORA_SEED:-42}"
alpha="${BERNINI_C2FR_LORA_ALPHA:-1.0}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
bernini_archive_sha256="08a331958560544efb5e842666c236d819bfdc36d34b6eb9c1cdcee1546ae670"
wan_diffusion_sha256="59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512"
num_steps=40

fail() { echo "[bernini-c2fr-lora81] ERROR: $*" >&2; exit 2; }

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid method source archive"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini source root"
[[ -f "${bernini_archive}" && ! -L "${bernini_archive}" ]] || fail "invalid Bernini source archive"
[[ -f "${bernini_root}/bernini/models/wan_diffusion.py" && ! -L "${bernini_root}/bernini/models/wan_diffusion.py" ]] || fail "invalid wan_diffusion.py"
[[ -d "${veomni_root}/.git" ]] || fail "VeOmni root must be a Git checkout"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid base checkpoint"
[[ -d "${adapter_checkpoint}" && ! -L "${adapter_checkpoint}" ]] || fail "invalid adapter checkpoint"
[[ -d "${adapter_checkpoint}/adapter" && ! -L "${adapter_checkpoint}/adapter" ]] || fail "adapter/ directory missing"
[[ -f "${adapter_checkpoint}/adapter/adapter_config.json" && ! -L "${adapter_checkpoint}/adapter/adapter_config.json" ]] || fail "adapter_config.json missing"
[[ -f "${adapter_checkpoint}/adapter/adapter_model.safetensors" && ! -L "${adapter_checkpoint}/adapter/adapter_model.safetensors" ]] || fail "adapter_model.safetensors missing"
[[ -f "${adapter_checkpoint}/receipt.json" && ! -L "${adapter_checkpoint}/receipt.json" ]] || fail "training receipt.json missing"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "invalid source video"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ -n "${instruction//[[:space:]]/}" ]] || fail "instruction is empty"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid method archive hash"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid method source revision"
[[ "${seed}" =~ ^[0-9]+$ ]] || fail "seed must be non-negative"
[[ -n "${alpha}" ]] || fail "alpha is empty"
[[ "${output_video}" == /*.mp4 ]] || fail "output must be an absolute .mp4 path"
[[ ! -e "${output_video}" && ! -L "${output_video}" ]] || fail "refusing to overwrite output"
[[ ! -e "${output_video}.receipt.json" && ! -L "${output_video}.receipt.json" ]] || fail "refusing to overwrite receipt"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive hash mismatch"
[[ "$(sha256sum "${bernini_archive}" | awk '{print $1}')" == "${bernini_archive_sha256}" ]] || fail "Bernini archive hash mismatch"
[[ "$(sha256sum "${bernini_root}/bernini/models/wan_diffusion.py" | awk '{print $1}')" == "${wan_diffusion_sha256}" ]] || fail "wan_diffusion.py hash mismatch"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_commit}" ]] || fail "VeOmni commit mismatch"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source is dirty"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

task_scratch="${SLURM_TMPDIR:-/tmp}/bernini-c2fr-lora81-${SLURM_JOB_ID}"
mkdir -p \
  "${task_scratch}/source" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton" \
  "${task_scratch}/rank-cache" \
  "${output_video%/*}"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
export BERNINI_C2FR_RANK_CACHE_ROOT="${task_scratch}/rank-cache"

tar -xf "${source_archive}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
[[ -f "${method_root}/spt_v2/infer_c2fr_lora.py" ]] || fail "archive lacks C2FR LoRA inference entry"
[[ -f "${method_root}/spt_v2/infer_c2fr.py" ]] || fail "archive lacks frozen C2FR baseline"
[[ -f "${method_root}/train_delta_lora.py" ]] || fail "archive lacks v4 training contract"
[[ -f "${method_root}/inference_sigma_strata.py" ]] || fail "archive lacks exact sigma strata"
[[ -f "${method_root}/tri_branch_unipc.py" ]] || fail "archive lacks tri-branch hook"
[[ -f "${method_root}/spt_v2/counterfactual_clean_field.py" ]] || fail "archive lacks counterfactual executor"
[[ -f "${method_root}/spt_v2/generator_native_sparse_router.py" ]] || fail "archive lacks sparse router"
[[ -f "${method_root}/tests/test_infer_c2fr_lora_contract.py" ]] || fail "archive lacks C2FR LoRA contract tests"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-c2fr-lora81] source_revision=${source_revision}"
echo "[bernini-c2fr-lora81] source_archive_sha256=${source_archive_sha256}"
echo "[bernini-c2fr-lora81] source_video_sha256=$(sha256sum "${source_video}" | awk '{print $1}')"
echo "[bernini-c2fr-lora81] adapter_model_sha256=$(sha256sum "${adapter_checkpoint}/adapter/adapter_model.safetensors" | awk '{print $1}')"
echo "[bernini-c2fr-lora81] adapter_config_sha256=$(sha256sum "${adapter_checkpoint}/adapter/adapter_config.json" | awk '{print $1}')"
echo "[bernini-c2fr-lora81] training_receipt_sha256=$(sha256sum "${adapter_checkpoint}/receipt.json" | awk '{print $1}')"
echo "[bernini-c2fr-lora81] output=${output_video}"
echo "[bernini-c2fr-lora81] frames=81 fps=25 steps=40 branches=negative+action+noop ulysses=4 adapter_scale=1 unmerged=true alpha=${alpha} dense_causal_boundary=true temporal_low_pass=false binary_support=false"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

"${python_bin}" -m py_compile \
  "${method_root}/spt_v2/infer_c2fr_lora.py" \
  "${method_root}/spt_v2/infer_c2fr.py" \
  "${method_root}/spt_v2/generator_native_sparse_router.py" \
  "${method_root}/tri_branch_unipc.py"

for pattern in \
  test_train_delta_lora_contract.py \
  test_tri_branch_unipc.py \
  test_spt_counterfactual_clean_field.py \
  test_spt_generator_native_sparse_router.py \
  test_infer_c2fr_contract.py \
  test_infer_c2fr_lora_contract.py; do
  PYTHONPATH="${method_root}" "${python_bin}" -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${method_root}/spt_v2/infer_c2fr_lora.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --adapter-checkpoint "${adapter_checkpoint}" \
  --source-video "${source_video}" \
  --instruction "${instruction}" \
  --output "${output_video}" \
  --num-inference-steps "${num_steps}" \
  --alpha "${alpha}" \
  --seed "${seed}" \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}"

[[ -f "${output_video}" && ! -L "${output_video}" ]] || fail "inference output missing"
[[ -f "${output_video}.receipt.json" && ! -L "${output_video}.receipt.json" ]] || fail "inference receipt missing"
rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-c2fr-lora81] PASS"
