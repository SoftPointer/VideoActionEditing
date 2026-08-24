#!/usr/bin/env bash
#SBATCH --job-name=bern-c2fr81
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

max_steps="${BERNINI_ACTION_MAX_STEPS:-1}"
save_every="${BERNINI_ACTION_SAVE_EVERY:-1}"
learning_rate="${BERNINI_ACTION_LR:-3e-5}"
lora_scope="${BERNINI_CDF_LORA_SCOPE:-cross_q}"
branch_state_mode="${BERNINI_C2FR_BRANCH_STATE_MODE:-source_target_bridge_clean_field}"
minimum_training_sigma="${BERNINI_C2FR_MINIMUM_SIGMA:-0.1}"
inverse_sigma_weight_floor="${BERNINI_C2FR_INVERSE_SIGMA_WEIGHT_FLOOR:-0.11765105277299881}"
motion_weight="${BERNINI_CDF_MOTION_WEIGHT:-1.0}"
copy_weight="${BERNINI_CDF_COPY_WEIGHT:-0.0}"
boundary_gauge_weight="${BERNINI_C2FR_BOUNDARY_GAUGE_WEIGHT:-0.0}"
motion_objective="${BERNINI_CDF_MOTION_OBJECTIVE:-causal_boundary_charbonnier}"
bridge_consistency_weight="${BERNINI_C2FR_BRIDGE_CONSISTENCY_WEIGHT:-0.1}"
charbonnier_scale="${BERNINI_C2FR_CHARBONNIER_SCALE:-0.1}"
quotient_weight="${BERNINI_CDF_QUOTIENT_WEIGHT:-0.5}"
high_noise_floor="${BERNINI_C2FR_HIGH_NOISE_FLOOR:-1.0}"
routing_jsonl="${BERNINI_CDF_ROUTING_JSONL:?set BERNINI_CDF_ROUTING_JSONL to the strict-359 route}"
routing_sha256="${BERNINI_CDF_ROUTING_JSONL_SHA256:?set BERNINI_CDF_ROUTING_JSONL_SHA256}"
unreviewed_tier="${BERNINI_CDF_UNREVIEWED_TIER:-reject}"
resume="${BERNINI_ACTION_RESUME:-}"
init_adapter="${BERNINI_CDF_INIT_ADAPTER:-}"
allow_incomplete="${BERNINI_ACTION_ALLOW_INCOMPLETE_DATASET:-0}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"

fail() {
  echo "[bernini-c2fr81] ERROR: $*" >&2
  exit 2
}

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid source archive"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini source root"
[[ -d "${veomni_root}/.git" ]] || fail "VeOmni root must be a Git checkout"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid checkpoint"
[[ -d "${data_dir}" && ! -L "${data_dir}" ]] || fail "invalid parquet directory"
[[ -f "${dataset_summary}" && ! -L "${dataset_summary}" ]] || fail "invalid dataset summary"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid source archive hash"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ "${max_steps}" =~ ^[1-9][0-9]*$ ]] || fail "max steps must be positive"
[[ "${save_every}" =~ ^[0-9]+$ ]] || fail "save interval must be non-negative"
[[ "${allow_incomplete}" =~ ^[01]$ ]] || fail "allow-incomplete flag must be 0 or 1"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash mismatch"
[[ -f "${routing_jsonl}" && ! -L "${routing_jsonl}" ]] || fail "invalid routing JSONL"
[[ "${routing_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid routing JSONL hash"
[[ "$(sha256sum "${routing_jsonl}" | awk '{print $1}')" == "${routing_sha256}" ]] || fail "routing JSONL hash mismatch"
[[ "${unreviewed_tier}" == reject ]] || fail "v4 formal training requires reject as the unreviewed tier"
if [[ -n "${resume}" ]]; then
  [[ -d "${resume}" && ! -L "${resume}" ]] || fail "invalid resume checkpoint"
fi
if [[ -n "${init_adapter}" ]]; then
  [[ -d "${init_adapter}" && ! -L "${init_adapter}" ]] || fail "invalid initialization adapter"
fi
[[ -z "${resume}" || -z "${init_adapter}" ]] || fail "resume and initialization are mutually exclusive"

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

task_scratch="${SLURM_TMPDIR:-/tmp}/bernini-c2fr81-${SLURM_JOB_ID}"
mkdir -p \
  "${task_scratch}/source" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"

tar -xf "${source_archive}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
[[ -f "${method_root}/train_delta_lora.py" ]] || fail "archive lacks C2FR trainer"

mkdir -p "${output_dir%/*}"
hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-c2fr81] source_revision=${source_revision}"
echo "[bernini-c2fr81] source_archive_sha256=${source_archive_sha256}"
echo "[bernini-c2fr81] scope=${lora_scope} branch_state=${branch_state_mode} minimum_sigma=${minimum_training_sigma} inverse_sigma_weight_floor=${inverse_sigma_weight_floor} action_noop=2x2_bridge objective=${motion_objective} representation=robust_idempotent_Q0 motion=${motion_weight} copy=${copy_weight} boundary_gauge=${boundary_gauge_weight} bridge_consistency=${bridge_consistency_weight} temporal_ema=disabled charbonnier=${charbonnier_scale} high_noise_floor=${high_noise_floor}"
echo "[bernini-c2fr81] data=${data_dir} max_steps=${max_steps}"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

export BERNINI_OFFICIAL_ROOT="${bernini_root}"
export BERNINI_VEOMNI_ROOT="${veomni_root}"
for pattern in \
  test_tri_branch_unipc.py \
  test_motion_residual.py \
  test_inference_sigma_strata.py \
  test_build_strict_routing.py \
  test_train_delta_lora_contract.py \
  test_infer_c2fr_lora_contract.py \
  test_differential_sampler_contract.py \
  test_infer_delta_lora_contract.py
do
  PYTHONPATH="${method_root}" "${python_bin}" -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

command=(
  "${python_bin}" -m torch.distributed.run
  --standalone
  --nproc_per_node=4
  "${method_root}/train_delta_lora.py"
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --preprocessed-parquet-dir "${data_dir}"
  --dataset-summary "${dataset_summary}"
  --output "${output_dir}"
  --num-frames 81
  --max-steps "${max_steps}"
  --save-every "${save_every}"
  --learning-rate "${learning_rate}"
  --lora-scope "${lora_scope}"
  --branch-state-mode "${branch_state_mode}"
  --minimum-training-sigma "${minimum_training_sigma}"
  --inverse-sigma-weight-floor "${inverse_sigma_weight_floor}"
  --motion-loss-weight "${motion_weight}"
  --copy-loss-weight "${copy_weight}"
  --boundary-gauge-loss-weight "${boundary_gauge_weight}"
  --motion-objective "${motion_objective}"
  --bridge-consistency-weight "${bridge_consistency_weight}"
  --charbonnier-scale "${charbonnier_scale}"
  --quotient-weight "${quotient_weight}"
  --high-noise-floor "${high_noise_floor}"
  --expected-bernini-commit "${bernini_commit}"
  --expected-veomni-commit "${veomni_commit}"
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}"
  --method-source-revision "${source_revision}"
  --method-source-archive-sha256 "${source_archive_sha256}"
  --routing-jsonl "${routing_jsonl}"
  --expected-routing-jsonl-sha256 "${routing_sha256}"
  --unreviewed-tier "${unreviewed_tier}"
)
if [[ "${allow_incomplete}" == 1 ]]; then
  command+=(--allow-incomplete-dataset)
fi
if [[ -n "${resume}" ]]; then
  command+=(--resume "${resume}")
fi
if [[ -n "${init_adapter}" ]]; then
  command+=(--init-adapter-checkpoint "${init_adapter}")
fi
"${command[@]}"

rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-c2fr81] PASS"
