#!/usr/bin/env bash
#SBATCH --job-name=bern-pgt-v5-81
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
umask 077

source_archive="${BERNINI_PGT_SOURCE_ARCHIVE:?set BERNINI_PGT_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_PGT_SOURCE_ARCHIVE_SHA256:?set BERNINI_PGT_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_PGT_SOURCE_REVISION:?set BERNINI_PGT_SOURCE_REVISION}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
bernini_archive="${BERNINI_OFFICIAL_ARCHIVE:?set BERNINI_OFFICIAL_ARCHIVE}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
adapter_checkpoint="${BERNINI_PGT_ADAPTER:-}"
source_video="${BERNINI_PGT_SOURCE_VIDEO:?set BERNINI_PGT_SOURCE_VIDEO}"
instruction="${BERNINI_PGT_INSTRUCTION:?set BERNINI_PGT_INSTRUCTION}"
output_video="${BERNINI_PGT_OUTPUT:?set BERNINI_PGT_OUTPUT}"
execution_arm="${BERNINI_PGT_EXECUTION_ARM:-main}"
seed="${BERNINI_PGT_SEED:-2027}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
bernini_archive_sha256="08a331958560544efb5e842666c236d819bfdc36d34b6eb9c1cdcee1546ae670"
wan_diffusion_sha256="59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512"

fail() { echo "[bernini-pgt-v5-81] ERROR: $*" >&2; exit 2; }

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid method source archive"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini source root"
[[ -f "${bernini_archive}" && ! -L "${bernini_archive}" ]] || fail "invalid Bernini archive"
[[ -f "${bernini_root}/bernini/models/wan_diffusion.py" && ! -L "${bernini_root}/bernini/models/wan_diffusion.py" ]] || fail "invalid wan_diffusion.py"
[[ -d "${veomni_root}/.git" ]] || fail "VeOmni root must be a Git checkout"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid base checkpoint"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "invalid source video"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ -n "${instruction//[[:space:]]/}" ]] || fail "instruction is empty"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid source archive hash"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ "${seed}" =~ ^[0-9]+$ ]] || fail "seed must be non-negative"
[[ "${execution_arm}" =~ ^(main|frozen_prior|causal_frozen_prior|parallel_only)$ ]] || fail "invalid execution arm"
if [[ "${execution_arm}" =~ ^(main|parallel_only)$ && -z "${adapter_checkpoint}" ]]; then
  fail "${execution_arm} requires BERNINI_PGT_ADAPTER"
fi
if [[ -n "${adapter_checkpoint}" ]]; then
  [[ -d "${adapter_checkpoint}" && ! -L "${adapter_checkpoint}" ]] || fail "invalid adapter checkpoint"
  [[ -d "${adapter_checkpoint}/adapter" && ! -L "${adapter_checkpoint}/adapter" ]] || fail "adapter directory missing"
  [[ -f "${adapter_checkpoint}/adapter/adapter_config.json" && ! -L "${adapter_checkpoint}/adapter/adapter_config.json" ]] || fail "adapter config missing"
  [[ -f "${adapter_checkpoint}/adapter/adapter_model.safetensors" && ! -L "${adapter_checkpoint}/adapter/adapter_model.safetensors" ]] || fail "adapter tensors missing"
  [[ -f "${adapter_checkpoint}/receipt.json" && ! -L "${adapter_checkpoint}/receipt.json" ]] || fail "training receipt missing"
fi
[[ "${output_video}" == /*.mp4 ]] || fail "output must be an absolute MP4"
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

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent must be an absolute writable plain directory"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-pgt-v5-81-${SLURM_JOB_ID}.XXXXXX")"
[[ "${task_scratch}" == "${scratch_parent%/}"/* && -d "${task_scratch}" && ! -L "${task_scratch}" ]] || fail "private scratch creation failed"
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

staged_archive="${task_scratch}/method-source.tar"
cp -- "${source_archive}" "${staged_archive}"
[[ -f "${staged_archive}" && ! -L "${staged_archive}" ]] || fail "staged source archive is invalid"
[[ "$(sha256sum "${staged_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged method archive hash mismatch"
"${python_bin}" - "${staged_archive}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

archive = sys.argv[1]
with tarfile.open(archive, mode="r:*") as handle:
    members = handle.getmembers()
    if not members:
        raise SystemExit("empty method source archive")
    for member in members:
        path = PurePosixPath(member.name)
        if not member.name or path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe archive path: {member.name!r}")
        if member.issym() or member.islnk():
            raise SystemExit(f"archive links are forbidden: {member.name!r}")
        if member.isfifo() or member.isdev():
            raise SystemExit(f"archive special files are forbidden: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported archive member: {member.name!r}")
PY
tar --no-same-owner --no-same-permissions -xf "${staged_archive}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
for required in \
  infer_prior_tangent_lora.py \
  train_prior_tangent_lora.py \
  prior_guided_tangent.py \
  tri_branch_unipc.py \
  inference_sigma_strata.py \
  infer_delta_lora.py \
  tests/test_infer_prior_tangent_lora_contract.py; do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive lacks plain ${required}"
done

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-pgt-v5-81] source_revision=${source_revision}"
echo "[bernini-pgt-v5-81] source_archive_sha256=${source_archive_sha256}"
echo "[bernini-pgt-v5-81] source_video_sha256=$(sha256sum "${source_video}" | awk '{print $1}')"
if [[ -n "${adapter_checkpoint}" ]]; then
  echo "[bernini-pgt-v5-81] adapter_model_sha256=$(sha256sum "${adapter_checkpoint}/adapter/adapter_model.safetensors" | awk '{print $1}')"
  echo "[bernini-pgt-v5-81] training_receipt_sha256=$(sha256sum "${adapter_checkpoint}/receipt.json" | awk '{print $1}')"
else
  echo "[bernini-pgt-v5-81] adapter_loaded=false transformer_forwards=120"
fi
echo "[bernini-pgt-v5-81] output=${output_video} arm=${execution_arm} seed=${seed}"
if [[ -n "${adapter_checkpoint}" ]]; then
  echo "[bernini-pgt-v5-81] frames=81 latent_phases=21 steps=40 branches=frozen_negative+frozen_noop+frozen_action+adapted_action transformer_forwards=160 ulysses=4 mask_flow_pose=false"
else
  echo "[bernini-pgt-v5-81] frames=81 latent_phases=21 steps=40 branches=base_negative+base_noop+base_action transformer_forwards=120 ulysses=4 mask_flow_pose=false"
fi
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

"${python_bin}" -m py_compile \
  "${method_root}/infer_prior_tangent_lora.py" \
  "${method_root}/train_prior_tangent_lora.py" \
  "${method_root}/prior_guided_tangent.py" \
  "${method_root}/tri_branch_unipc.py"

for pattern in \
  test_prior_guided_tangent.py \
  test_train_prior_tangent_lora_contract.py \
  test_tri_branch_unipc.py \
  test_infer_prior_tangent_lora_contract.py; do
  PYTHONPATH="${method_root}" "${python_bin}" -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

adapter_args=()
if [[ -n "${adapter_checkpoint}" ]]; then
  adapter_args=(--adapter-checkpoint "${adapter_checkpoint}")
fi

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${method_root}/infer_prior_tangent_lora.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  "${adapter_args[@]}" \
  --source-video "${source_video}" \
  --instruction "${instruction}" \
  --output "${output_video}" \
  --execution-arm "${execution_arm}" \
  --num-inference-steps 40 \
  --alpha 1.0 \
  --seed "${seed}" \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}"

[[ -f "${output_video}" && ! -L "${output_video}" ]] || fail "inference output missing"
[[ -f "${output_video}.receipt.json" && ! -L "${output_video}.receipt.json" ]] || fail "inference receipt missing"
rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-pgt-v5-81] PASS arm=${execution_arm}"
