#!/usr/bin/env bash
#SBATCH --job-name=bern-rmc-v7-infer
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

source_archive="${BERNINI_V7_SOURCE_ARCHIVE:?set BERNINI_V7_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_V7_SOURCE_ARCHIVE_SHA256:?set BERNINI_V7_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_V7_SOURCE_REVISION:?set BERNINI_V7_SOURCE_REVISION}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
adapter_checkpoint="${BERNINI_V7_ADAPTER_CHECKPOINT:?set BERNINI_V7_ADAPTER_CHECKPOINT}"
source_video="${BERNINI_V7_SOURCE_VIDEO:?set BERNINI_V7_SOURCE_VIDEO}"
instruction="${BERNINI_V7_INSTRUCTION:?set BERNINI_V7_INSTRUCTION}"
output_video="${BERNINI_V7_INFERENCE_OUTPUT:?set BERNINI_V7_INFERENCE_OUTPUT}"
kappa="${BERNINI_V7_KAPPA:-0.25}"
seed="${BERNINI_V7_INFERENCE_SEED:-2027}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"

fail() { echo "[bernini-rmc-v7-infer] ERROR: $*" >&2; exit 2; }

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid source archive"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini root"
[[ -d "${veomni_root}/.git" ]] || fail "invalid VeOmni root"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid checkpoint"
[[ -d "${adapter_checkpoint}" && ! -L "${adapter_checkpoint}" ]] || fail "invalid adapter checkpoint"
[[ -d "${adapter_checkpoint}/adapter" && ! -L "${adapter_checkpoint}/adapter" ]] || fail "adapter directory missing"
[[ -f "${adapter_checkpoint}/adapter/adapter_config.json" && ! -L "${adapter_checkpoint}/adapter/adapter_config.json" ]] || fail "adapter config missing"
[[ -f "${adapter_checkpoint}/adapter/adapter_model.safetensors" && ! -L "${adapter_checkpoint}/adapter/adapter_model.safetensors" ]] || fail "adapter tensors missing"
[[ -f "${adapter_checkpoint}/receipt.json" && ! -L "${adapter_checkpoint}/receipt.json" ]] || fail "training receipt missing"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "invalid source video"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ -n "${instruction//[[:space:]]/}" ]] || fail "instruction is empty"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid archive SHA-256"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ "${seed}" =~ ^[0-9]+$ ]] || fail "seed must be non-negative"
[[ "${kappa}" == "0.25" || "${kappa}" == "0.5" || "${kappa}" == "1.0" ]] || fail "kappa must be 0.25, 0.5, or 1.0"
[[ "${output_video}" == /*.mp4 ]] || fail "output must be an absolute MP4 path"
[[ ! -e "${output_video}" && ! -L "${output_video}" ]] || fail "refusing to overwrite output"
[[ ! -e "${output_video}.receipt.json" && ! -L "${output_video}.receipt.json" ]] || fail "refusing to overwrite receipt"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash mismatch"
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

task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-rmc-v7-infer-${SLURM_JOB_ID}.XXXXXX")"
archive_copy="${task_scratch}/method.tar"
source_root="${task_scratch}/source"
mkdir -p \
  "${source_root}" \
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

cp -- "${source_archive}" "${archive_copy}"
[[ -f "${archive_copy}" && ! -L "${archive_copy}" ]] || fail "private archive copy is invalid"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "private archive hash mismatch"
"${python_bin}" - "${archive_copy}" <<'PY'
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
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${source_root}"
method_root="${source_root}/methods/bernini_action_editing"
for required in \
  run_relational_motion_commutator_inference.py \
  infer_relational_motion_commutator.py \
  motion_commutator.py \
  relational_commutator_objective.py \
  train_relational_motion_commutator_auh.py \
  infer_prior_tangent_lora.py \
  train_prior_tangent_lora.py \
  train_cross_mode_cmsg_auh.py \
  train_cross_mode_cmsg_lora.py \
  train_delta_lora.py \
  train_lora.py \
  motion_residual.py \
  cross_mode_motion_spectrum.py \
  tri_branch_unipc.py \
  inference_sigma_strata.py \
  infer_delta_lora.py \
  tests/test_infer_relational_motion_commutator.py \
  tests/test_run_relational_motion_commutator_inference_contract.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive lacks plain ${required}"
done

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-rmc-v7-infer] source_revision=${source_revision} archive_sha256=${source_archive_sha256}"
echo "[bernini-rmc-v7-infer] source_sha256=$(sha256sum "${source_video}" | awk '{print $1}')"
echo "[bernini-rmc-v7-infer] adapter_sha256=$(sha256sum "${adapter_checkpoint}/adapter/adapter_model.safetensors" | awk '{print $1}')"
echo "[bernini-rmc-v7-infer] receipt_sha256=$(sha256sum "${adapter_checkpoint}/receipt.json" | awk '{print $1}')"
echo "[bernini-rmc-v7-infer] frames=81 phases=21 steps=40 editor_forwards_per_step=5 generator_forwards=0 ulysses=4"
if [[ "${kappa}" == "0.25" ]]; then
  echo "[bernini-rmc-v7-infer] kappa=0.25 arm=training-matched-main"
else
  echo "[bernini-rmc-v7-infer] kappa=${kappa} arm=inference-only-ablation"
fi
echo "[bernini-rmc-v7-infer] output=${output_video} seed=${seed}"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

"${python_bin}" -m py_compile \
  "${method_root}/run_relational_motion_commutator_inference.py" \
  "${method_root}/infer_relational_motion_commutator.py" \
  "${method_root}/motion_commutator.py"
for pattern in \
  'test_motion_commutator.py' \
  'test_infer_relational_motion_commutator.py' \
  'test_run_relational_motion_commutator_inference_contract.py'
do
  PYTHONPATH="${method_root}" "${python_bin}" -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${method_root}/run_relational_motion_commutator_inference.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --adapter-checkpoint "${adapter_checkpoint}" \
  --source-video "${source_video}" \
  --instruction "${instruction}" \
  --output "${output_video}" \
  --num-inference-steps 40 \
  --kappa "${kappa}" \
  --seed "${seed}" \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}"

[[ -f "${output_video}" && ! -L "${output_video}" ]] || fail "inference output missing"
[[ -f "${output_video}.receipt.json" && ! -L "${output_video}.receipt.json" ]] || fail "inference receipt missing"
rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-rmc-v7-infer] PASS kappa=${kappa}"
