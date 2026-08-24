#!/usr/bin/env bash
#SBATCH --job-name=bern-pdfv2
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=1-00:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:?set BERNINI_ACTION_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:?set BERNINI_ACTION_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_ACTION_SOURCE_REVISION:?set BERNINI_ACTION_SOURCE_REVISION}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
data_dir="${BERNINI_ACTION_PARQUET_DIR:?set BERNINI_ACTION_PARQUET_DIR}"
dataset_summary="${BERNINI_ACTION_DATASET_SUMMARY:?set BERNINI_ACTION_DATASET_SUMMARY}"
output_dir="${BERNINI_ACTION_TRAIN_OUTPUT:?set BERNINI_ACTION_TRAIN_OUTPUT}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

max_steps="${BERNINI_ACTION_MAX_STEPS:-644}"
save_every="${BERNINI_ACTION_SAVE_EVERY:-64}"
learning_rate="${BERNINI_ACTION_LR:-1e-4}"
lora_scope="${BERNINI_CDF_LORA_SCOPE:-q_out}"
copy_weight="${BERNINI_CDF_COPY_WEIGHT:-0.25}"
quotient_weight="${BERNINI_CDF_QUOTIENT_WEIGHT:-0.5}"
dc_weight="${BERNINI_PDF_DC_WEIGHT:-0.25}"
interval_power="${BERNINI_PDF_INTERVAL_POWER:-1.0}"
integration_steps="${BERNINI_ACTION_INFERENCE_STEPS:-40}"
flow_shift="${BERNINI_PDF_FLOW_SHIFT:-5.0}"
init_adapter="${BERNINI_CDF_INIT_ADAPTER:-}"
allow_incomplete="${BERNINI_ACTION_ALLOW_INCOMPLETE_DATASET:-0}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"

fail() { echo "[bernini-pdfv2] ERROR: $*" >&2; exit 2; }
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid source archive"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini root"
[[ -d "${veomni_root}/.git" ]] || fail "invalid VeOmni root"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid checkpoint"
[[ -d "${data_dir}" && ! -L "${data_dir}" ]] || fail "invalid parquet directory"
[[ -f "${dataset_summary}" && ! -L "${dataset_summary}" ]] || fail "invalid summary"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid archive SHA"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ "${max_steps}" =~ ^[1-9][0-9]*$ ]] || fail "invalid max steps"
[[ "${save_every}" =~ ^[0-9]+$ ]] || fail "invalid save interval"
[[ "${allow_incomplete}" =~ ^[01]$ ]] || fail "invalid incomplete flag"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "archive hash mismatch"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1

task_scratch="${SLURM_TMPDIR:-/tmp}/bernini-pdfv2-${SLURM_JOB_ID}"
mkdir -p "${task_scratch}/source" "${task_scratch}/miopen-user" "${task_scratch}/miopen-custom" "${task_scratch}/torch-extensions" "${task_scratch}/triton"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
tar -xf "${source_archive}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
[[ -f "${method_root}/train_projected_delta_lora.py" ]] || fail "archive lacks PDF-v2 trainer"

PYTHONPATH="${method_root}" "${python_bin}" -m unittest discover -s "${method_root}/tests" -p 'test_projected_differential_flow.py' -v

command=(
  "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=4
  "${method_root}/train_projected_delta_lora.py"
  --bernini-root "${bernini_root}" --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}" --preprocessed-parquet-dir "${data_dir}"
  --dataset-summary "${dataset_summary}" --output "${output_dir}"
  --num-frames 81 --max-steps "${max_steps}" --save-every "${save_every}"
  --learning-rate "${learning_rate}" --lora-scope "${lora_scope}"
  --motion-loss-weight 1.0 --copy-loss-weight "${copy_weight}"
  --anchor-loss-weight 0.0 --motion-objective quotient_multilag
  --quotient-weight "${quotient_weight}" --dc-loss-weight "${dc_weight}"
  --integration-steps "${integration_steps}" --integration-flow-shift "${flow_shift}"
  --interval-weight-power "${interval_power}"
  --expected-bernini-commit "${bernini_commit}" --expected-veomni-commit "${veomni_commit}"
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}"
  --method-source-revision "${source_revision}" --method-source-archive-sha256 "${source_archive_sha256}"
)
[[ "${allow_incomplete}" == 1 ]] && command+=(--allow-incomplete-dataset)
[[ -z "${init_adapter}" ]] || command+=(--init-adapter-checkpoint "${init_adapter}")
"${command[@]}"
echo "[bernini-pdfv2] PASS"
