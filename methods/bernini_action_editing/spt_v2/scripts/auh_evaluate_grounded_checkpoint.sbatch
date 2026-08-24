#!/usr/bin/env bash
#SBATCH --job-name=bern-spt-v3-eval
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=08:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_root="${BERNINI_SPT_SOURCE_ROOT:?set BERNINI_SPT_SOURCE_ROOT}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
planner_checkpoint="${BERNINI_SPT_EVAL_CHECKPOINT:?set BERNINI_SPT_EVAL_CHECKPOINT}"
data_dir="${BERNINI_ACTION_PARQUET_DIR:?set BERNINI_ACTION_PARQUET_DIR}"
dataset_summary="${BERNINI_ACTION_DATASET_SUMMARY:?set BERNINI_ACTION_DATASET_SUMMARY}"
selected_membership="${BERNINI_SPT_SELECTED_MEMBERSHIP:?set BERNINI_SPT_SELECTED_MEMBERSHIP}"
output="${BERNINI_SPT_EVAL_OUTPUT:?set BERNINI_SPT_EVAL_OUTPUT}"
method_revision="${BERNINI_SPT_METHOD_SOURCE_REVISION:?set BERNINI_SPT_METHOD_SOURCE_REVISION}"
method_archive_sha256="${BERNINI_SPT_METHOD_SOURCE_ARCHIVE_SHA256:?set BERNINI_SPT_METHOD_SOURCE_ARCHIVE_SHA256}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
allow_incomplete="${BERNINI_ACTION_ALLOW_INCOMPLETE_DATASET:-0}"

fail() { echo "[bernini-spt-v3-eval] ERROR: $*" >&2; exit 2; }
[[ -f "${source_root}/spt_v2/evaluate_grounded_checkpoint.py" ]] || fail "source root lacks evaluator"
[[ -f "${source_root}/spt_v2/grounded_phase_planner.py" ]] || fail "source root lacks grounded planner"
[[ -f "${source_root}/train_lora.py" ]] || fail "source root lacks pinned Bernini helpers"
[[ -d "${bernini_root}" ]] || fail "missing Bernini root"
[[ -d "${veomni_root}/.git" ]] || fail "VeOmni must be a Git checkout"
[[ -d "${checkpoint}" ]] || fail "missing base checkpoint"
[[ -d "${planner_checkpoint}" && ! -L "${planner_checkpoint}" ]] || fail "invalid planner checkpoint"
for artifact in planner.safetensors planner_config.json receipt.json optimizer.pt; do
  [[ -f "${planner_checkpoint}/${artifact}" ]] || fail "planner checkpoint lacks ${artifact}"
done
[[ -d "${data_dir}" ]] || fail "missing parquet directory"
[[ -f "${dataset_summary}" && ! -L "${dataset_summary}" ]] || fail "invalid dataset summary"
[[ -f "${selected_membership}" && ! -L "${selected_membership}" ]] || fail "invalid selected membership"
[[ "${output}" = /* ]] || fail "output must be absolute"
[[ ! -e "${output}" && ! -L "${output}" ]] || fail "refusing to overwrite output"
[[ "${method_revision}" =~ ^[0-9a-fA-F]{40}$ ]] || fail "invalid method source revision"
[[ "${method_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid method archive SHA-256"
[[ "${allow_incomplete}" =~ ^[01]$ ]] || fail "invalid incomplete flag"

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

task_scratch="${SLURM_TMPDIR:-/tmp}/bernini-spt-v3-eval-${SLURM_JOB_ID}"
mkdir -p \
  "${task_scratch}/method" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton" \
  "${task_scratch}/rank-cache"
cp -a "${source_root}/." "${task_scratch}/method/"
method_root="${task_scratch}/method"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
export BERNINI_SPT_RANK_CACHE_ROOT="${task_scratch}/rank-cache"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

PYTHONPATH="${method_root}:${method_root}/spt_v2" "${python_bin}" -m unittest discover \
  -s "${method_root}/tests" -p 'test_spt_v3_checkpoint_evaluation.py' -v

command=(
  "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=4
  "${method_root}/spt_v2/evaluate_grounded_checkpoint.py"
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --planner-checkpoint "${planner_checkpoint}"
  --preprocessed-parquet-dir "${data_dir}"
  --dataset-summary "${dataset_summary}"
  --selected-membership "${selected_membership}"
  --output "${output}"
  --method-source-revision "${method_revision}"
  --method-source-archive-sha256 "${method_archive_sha256}"
)
if [[ "${allow_incomplete}" == 1 ]]; then command+=(--allow-incomplete-dataset); fi
"${command[@]}"

echo "[bernini-spt-v3-eval] PASS"
