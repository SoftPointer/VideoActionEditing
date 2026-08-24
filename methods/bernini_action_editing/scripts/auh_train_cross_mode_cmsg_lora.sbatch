#!/usr/bin/env bash
#SBATCH --job-name=bern-cmmd-v6-81
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=04:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 077

source_archive="${BERNINI_V6_SOURCE_ARCHIVE:?set BERNINI_V6_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_V6_SOURCE_ARCHIVE_SHA256:?set BERNINI_V6_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_V6_SOURCE_REVISION:?set BERNINI_V6_SOURCE_REVISION}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
data_dir="${BERNINI_ACTION_PARQUET_DIR:?set BERNINI_ACTION_PARQUET_DIR}"
dataset_summary="${BERNINI_ACTION_DATASET_SUMMARY:?set BERNINI_ACTION_DATASET_SUMMARY}"
routing_jsonl="${BERNINI_V6_ROUTING_JSONL:?set BERNINI_V6_ROUTING_JSONL}"
routing_sha256="${BERNINI_V6_ROUTING_JSONL_SHA256:?set BERNINI_V6_ROUTING_JSONL_SHA256}"
output_dir="${BERNINI_V6_TRAIN_OUTPUT:?set BERNINI_V6_TRAIN_OUTPUT}"
max_steps="${BERNINI_V6_MAX_STEPS:-40}"
save_every="${BERNINI_V6_SAVE_EVERY:-40}"
max_gate_attempts="${BERNINI_V6_MAX_GATE_ATTEMPTS:-16}"
disable_gate="${BERNINI_V6_DISABLE_GATE:-0}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"

fail() { echo "[bernini-cmmd-v6-81] ERROR: $*" >&2; exit 2; }

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid source archive"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini root"
[[ -d "${veomni_root}/.git" ]] || fail "invalid VeOmni root"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid checkpoint"
[[ -d "${data_dir}" && ! -L "${data_dir}" ]] || fail "invalid parquet directory"
[[ -f "${dataset_summary}" && ! -L "${dataset_summary}" ]] || fail "invalid dataset summary"
[[ -f "${routing_jsonl}" && ! -L "${routing_jsonl}" ]] || fail "invalid routing JSONL"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid archive SHA-256"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ "${routing_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid routing SHA-256"
[[ "${max_steps}" =~ ^[1-9][0-9]*$ ]] || fail "max steps must be positive"
[[ "${save_every}" =~ ^[0-9]+$ ]] || fail "save interval must be non-negative"
[[ "${max_gate_attempts}" =~ ^[1-9][0-9]*$ ]] || fail "gate attempts must be positive"
[[ "${disable_gate}" =~ ^[01]$ ]] || fail "disable gate must be 0 or 1"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash mismatch"
[[ "$(sha256sum "${routing_jsonl}" | awk '{print $1}')" == "${routing_sha256}" ]] || fail "routing hash mismatch"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_commit}" ]] || fail "VeOmni commit mismatch"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source is dirty"
if [[ "${disable_gate}" == 1 && "${max_steps}" != 1 ]]; then
  fail "gate-disabled mode is restricted to a one-step diagnostic canary"
fi

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

task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-cmmd-v6-81-${SLURM_JOB_ID}.XXXXXX")"
archive_copy="${task_scratch}/method.tar"
source_root="${task_scratch}/source"
mkdir -p \
  "${source_root}" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton" \
  "${output_dir%/*}"
cp -- "${source_archive}" "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "private archive hash mismatch"
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${source_root}"
[[ -z "$(find "${source_root}" -type l -print -quit)" ]] || fail "source archive contains a symlink"

method_root="${source_root}/methods/bernini_action_editing"
for required in \
  cross_mode_branches.py \
  cross_mode_motion_spectrum.py \
  train_cross_mode_cmsg_lora.py \
  train_cross_mode_cmsg_auh.py \
  infer_cross_mode_cmsg_lora.py \
  train_prior_tangent_lora.py \
  train_delta_lora.py \
  train_lora.py \
  motion_residual.py \
  inference_sigma_strata.py \
  tri_branch_unipc.py \
  tests/test_cross_mode_branches.py \
  tests/test_cross_mode_motion_spectrum.py \
  tests/test_train_cross_mode_cmsg_lora.py \
  tests/test_train_cross_mode_cmsg_auh_contract.py \
  tests/test_infer_cross_mode_cmsg_lora_contract.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive lacks plain ${required}"
done

export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-cmmd-v6-81] source_revision=${source_revision} archive_sha256=${source_archive_sha256}"
echo "[bernini-cmmd-v6-81] routing_sha256=${routing_sha256} output=${output_dir}"
echo "[bernini-cmmd-v6-81] frames=81 phases=21 beta=0 forwards=6 scope=cross_q_plus_mid_self_q rank=8 lr=2e-5"
echo "[bernini-cmmd-v6-81] max_steps=${max_steps} save_every=${save_every} max_gate_attempts=${max_gate_attempts} disable_gate=${disable_gate}"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

"${python_bin}" -m py_compile \
  "${method_root}/cross_mode_branches.py" \
  "${method_root}/cross_mode_motion_spectrum.py" \
  "${method_root}/train_cross_mode_cmsg_lora.py" \
  "${method_root}/train_cross_mode_cmsg_auh.py" \
  "${method_root}/infer_cross_mode_cmsg_lora.py"
for pattern in \
  'test_cross_mode_branches.py' \
  'test_cross_mode_motion_spectrum.py' \
  'test_train_cross_mode_cmsg_lora.py' \
  'test_train_cross_mode_cmsg_auh_contract.py' \
  'test_infer_cross_mode_cmsg_lora_contract.py'
do
  PYTHONPATH="${method_root}" "${python_bin}" -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

command=(
  "${python_bin}" -m torch.distributed.run
  --standalone
  --nproc_per_node=4
  "${method_root}/train_cross_mode_cmsg_auh.py"
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --preprocessed-parquet-dir "${data_dir}"
  --dataset-summary "${dataset_summary}"
  --routing-jsonl "${routing_jsonl}"
  --expected-routing-jsonl-sha256 "${routing_sha256}"
  --output "${output_dir}"
  --num-frames 81
  --max-steps "${max_steps}"
  --save-every "${save_every}"
  --learning-rate 2e-5
  --max-gate-attempts "${max_gate_attempts}"
  --expected-bernini-commit "${bernini_commit}"
  --expected-veomni-commit "${veomni_commit}"
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}"
  --method-source-revision "${source_revision}"
  --method-source-archive-sha256 "${source_archive_sha256}"
)
if [[ "${disable_gate}" == 1 ]]; then
  command+=(--disable-frozen-prior-gate)
fi
"${command[@]}"

rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-cmmd-v6-81] PASS"
