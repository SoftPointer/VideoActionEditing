#!/usr/bin/env bash
#SBATCH --job-name=bern-spt-joint
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
planner_checkpoint="${BERNINI_SPT_JOINT_PLANNER_CHECKPOINT:?set BERNINI_SPT_JOINT_PLANNER_CHECKPOINT}"
output_dir="${BERNINI_SPT_JOINT_OUTPUT:?set BERNINI_SPT_JOINT_OUTPUT}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

max_steps="${BERNINI_SPT_JOINT_MAX_STEPS:-256}"
save_every="${BERNINI_SPT_JOINT_SAVE_EVERY:-32}"
train_prefix_rows="${BERNINI_SPT_JOINT_TRAIN_PREFIX_ROWS:-}"
selected_membership="${BERNINI_SPT_SELECTED_MEMBERSHIP:-}"
learning_rate="${BERNINI_SPT_JOINT_LR:-1e-4}"
generate_budget="${BERNINI_SPT_JOINT_MAX_ORACLE_GENERATE_FRACTION:-0.12}"
resume="${BERNINI_SPT_JOINT_RESUME:-}"
allow_incomplete="${BERNINI_ACTION_ALLOW_INCOMPLETE_DATASET:-0}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"

fail() { echo "[bernini-spt-joint] ERROR: $*" >&2; exit 2; }
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid source archive"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid source archive SHA"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash mismatch"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini root"
[[ -d "${veomni_root}/.git" ]] || fail "invalid VeOmni root"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid checkpoint"
[[ -d "${data_dir}" && ! -L "${data_dir}" ]] || fail "invalid parquet directory"
[[ -f "${dataset_summary}" && ! -L "${dataset_summary}" ]] || fail "invalid dataset summary"
[[ -d "${planner_checkpoint}" && ! -L "${planner_checkpoint}" ]] || fail "phase_query_v2 planner checkpoint is not ready"
for planner_file in planner.safetensors planner_config.json receipt.json; do
  [[ -f "${planner_checkpoint}/${planner_file}" && ! -L "${planner_checkpoint}/${planner_file}" ]] || fail "planner checkpoint lacks ${planner_file}"
done
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "${max_steps}" =~ ^[1-9][0-9]*$ ]] || fail "invalid max steps"
[[ "${save_every}" =~ ^[0-9]+$ ]] || fail "invalid save interval"
if [[ -z "${train_prefix_rows}" && -z "${selected_membership}" ]]; then train_prefix_rows=8; fi
[[ -z "${train_prefix_rows}" || "${train_prefix_rows}" =~ ^[1-9][0-9]*$ ]] || fail "invalid train-prefix rows"
[[ -z "${selected_membership}" || ( -f "${selected_membership}" && ! -L "${selected_membership}" ) ]] || fail "invalid selected membership"
[[ -z "${selected_membership}" || -z "${train_prefix_rows}" ]] || fail "selected membership and train prefix are mutually exclusive"
[[ "${allow_incomplete}" =~ ^[01]$ ]] || fail "invalid incomplete flag"
[[ -z "${resume}" || -d "${resume}" ]] || fail "invalid resume checkpoint"
[[ "${generate_budget}" =~ ^0\.[0-9]+$ ]] || fail "oracle generate budget must be a decimal in (0,0.12]"
"${python_bin}" -c "x=float('${generate_budget}'); assert 0.0 < x <= 0.12" || fail "oracle generate budget must lie in (0,0.12]"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1

task_scratch="${SLURM_TMPDIR:-/tmp}/bernini-spt-joint-${SLURM_JOB_ID}"
mkdir -p "${task_scratch}/source" "${task_scratch}/miopen-user" "${task_scratch}/miopen-custom" "${task_scratch}/torch-extensions" "${task_scratch}/triton"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
tar -xf "${source_archive}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
[[ -f "${method_root}/spt_v2/train_joint_lora.py" ]] || fail "archive lacks joint trainer"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

for pattern in test_spt_v2_contract.py test_spt_v2_phase_query_planner.py test_spt_v2_joint_lora_contract.py; do
  PYTHONPATH="${method_root}:${method_root}/spt_v2" "${python_bin}" -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

command=(
  "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=4
  "${method_root}/spt_v2/train_joint_lora.py"
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --preprocessed-parquet-dir "${data_dir}"
  --dataset-summary "${dataset_summary}"
  --planner-checkpoint "${planner_checkpoint}"
  --output "${output_dir}"
  --num-frames 81
  --max-steps "${max_steps}"
  --save-every "${save_every}"
  --learning-rate "${learning_rate}"
  --lora-scope cross_q_out
  --executor-loss-weight 1.0
  --generate-loss-weight 0.25
  --complement-loss-weight 0.25
  --base-noop-replay-loss-weight 0.25
  --max-oracle-generate-fraction "${generate_budget}"
  --integration-steps 40
  --integration-flow-shift 5
  --expected-bernini-commit "${bernini_commit}"
  --expected-veomni-commit "${veomni_commit}"
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}"
  --method-source-revision "${source_revision}"
  --method-source-archive-sha256 "${source_archive_sha256}"
)
[[ -z "${train_prefix_rows}" ]] || command+=(--train-prefix-rows "${train_prefix_rows}")
[[ -z "${selected_membership}" ]] || command+=(--selected-membership "${selected_membership}")
[[ "${allow_incomplete}" == 1 ]] && command+=(--allow-incomplete-dataset)
[[ -z "${resume}" ]] || command+=(--resume "${resume}")
"${command[@]}"

echo "[bernini-spt-joint] PASS"
