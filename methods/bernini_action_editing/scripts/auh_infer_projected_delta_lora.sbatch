#!/usr/bin/env bash
#SBATCH --job-name=bern-pdf-infer
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
source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:?set source archive}"
source_archive_sha256="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:?set source archive SHA}"
source_revision="${BERNINI_ACTION_SOURCE_REVISION:?set source revision}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set checkpoint}"
adapter="${BERNINI_ACTION_ADAPTER_CHECKPOINT:?set adapter checkpoint}"
source_video="${BERNINI_ACTION_SOURCE_VIDEO:?set source video}"
instruction="${BERNINI_ACTION_INSTRUCTION:?set instruction}"
output_video="${BERNINI_ACTION_INFERENCE_OUTPUT:?set output}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
steps="${BERNINI_ACTION_INFERENCE_STEPS:-40}"
substeps="${BERNINI_PDF_SOLVER_SUBSTEPS:-2}"
strength="${BERNINI_CDF_MOTION_STRENGTH:-1.0}"
seed="${BERNINI_ACTION_INFERENCE_SEED:-2027}"
bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_sha="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"

fail() { echo "[bernini-pdf-infer] ERROR: $*" >&2; exit 2; }
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid source archive"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "archive hash mismatch"
[[ -d "${adapter}" && -f "${adapter}/receipt.json" ]] || fail "invalid adapter"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "invalid source video"
[[ "${output_video}" == /*.mp4 && ! -e "${output_video}" ]] || fail "output must be a new absolute MP4"
[[ "${substeps}" =~ ^([2-9]|1[0-6])$ ]] || fail "substeps must be in [2,16]"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
task_scratch="${SLURM_TMPDIR:-/tmp}/bernini-pdf-infer-${SLURM_JOB_ID}"
mkdir -p "${task_scratch}/source" "${task_scratch}/miopen-user" "${task_scratch}/miopen-custom" "${task_scratch}/torch-extensions" "${task_scratch}/triton" "${output_video%/*}"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
tar -xf "${source_archive}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
[[ -f "${method_root}/infer_projected_delta_lora.py" ]] || fail "archive lacks PDF inference"
PYTHONPATH="${method_root}" "${python_bin}" -m unittest discover -s "${method_root}/tests" -p 'test_projected_differential_flow.py' -v

"${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${method_root}/infer_projected_delta_lora.py" \
  --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" --adapter-checkpoint "${adapter}" \
  --source-video "${source_video}" --instruction "${instruction}" --output "${output_video}" \
  --sampling-mode differential --motion-strength "${strength}" \
  --num-inference-steps "${steps}" --solver-substeps "${substeps}" --seed "${seed}" \
  --expected-bernini-commit "${bernini_commit}" --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_sha}" \
  --method-source-revision "${source_revision}" --method-source-archive-sha256 "${source_archive_sha256}"

[[ -f "${output_video}" && -f "${output_video}.receipt.json" ]] || fail "output or receipt missing"
echo "[bernini-pdf-infer] PASS"
