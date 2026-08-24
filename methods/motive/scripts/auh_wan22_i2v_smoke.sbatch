#!/usr/bin/env bash
#SBATCH --job-name=wan22-i2v-smoke
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1024G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=02:00:00
#SBATCH --requeue

# One Slurm task supervises torchrun.  torchrun creates exactly eight ranks;
# all ranks cooperate on one sample through T5/DiT FSDP and Ulysses SP.

set -Eeuo pipefail
umask 077

source_snapshot="${MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT:?set MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT}"
generation_manifest="${MOTIVE_GOKU_ACTION_GENERATION_MANIFEST:?set MOTIVE_GOKU_ACTION_GENERATION_MANIFEST}"
signed_release="${MOTIVE_WAN22_SIGNED_RELEASE:?set MOTIVE_WAN22_SIGNED_RELEASE}"
wan_code_root="${MOTIVE_WAN22_CODE_ROOT:?set MOTIVE_WAN22_CODE_ROOT}"
checkpoint_dir="${MOTIVE_WAN22_CKPT_DIR:?set MOTIVE_WAN22_CKPT_DIR}"
python_bin="${MOTIVE_WAN22_PYTHON_BIN:?set MOTIVE_WAN22_PYTHON_BIN}"
ffprobe_bin="${MOTIVE_WAN22_FFPROBE_BIN:?set MOTIVE_WAN22_FFPROBE_BIN}"
output_root="${MOTIVE_WAN22_OUTPUT_ROOT:?set MOTIVE_WAN22_OUTPUT_ROOT}"
allow_pending_review="${MOTIVE_WAN22_ALLOW_PENDING_REVIEW:?set explicitly to 0}"
data_root="${MOTIVE_WAN22_DATA_ROOT:-}"
max_new_samples="${MOTIVE_WAN22_MAX_NEW_SAMPLES:-1}"
frame_num="${MOTIVE_WAN22_FRAME_NUM:-81}"
sample_steps="${MOTIVE_WAN22_SAMPLE_STEPS:-4}"
sample_shift="${MOTIVE_WAN22_SAMPLE_SHIFT:-3.0}"
size="${MOTIVE_WAN22_SIZE:-480*832}"
base_seed="${MOTIVE_WAN22_BASE_SEED:-260730}"
code_root="${source_snapshot}/methods/motive"

fail() {
  echo "[wan22-i2v-smoke] $*" >&2
  exit 2
}

require_absolute() {
  local label="$1"
  local value="$2"
  if [[ "${value}" != /* || "${value}" == "/" ]]; then
    fail "${label} must be a non-root absolute path: ${value}"
  fi
}

require_positive_integer() {
  local label="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    fail "${label} must be a positive integer: ${value}"
  fi
}

for binding in \
  "source_snapshot:${source_snapshot}" \
  "generation_manifest:${generation_manifest}" \
  "signed_release:${signed_release}" \
  "wan_code_root:${wan_code_root}" \
  "checkpoint_dir:${checkpoint_dir}" \
  "python_bin:${python_bin}" \
  "ffprobe_bin:${ffprobe_bin}" \
  "output_root:${output_root}"; do
  require_absolute "${binding%%:*}" "${binding#*:}"
done
if [[ -n "${data_root}" ]]; then
  require_absolute "data_root" "${data_root}"
fi
if [[ "${allow_pending_review}" != "0" ]]; then
  fail "MOTIVE_WAN22_ALLOW_PENDING_REVIEW must be 0; only the signed exact-eight-row release authorizes generation"
fi
require_positive_integer "MOTIVE_WAN22_MAX_NEW_SAMPLES" "${max_new_samples}"
require_positive_integer "MOTIVE_WAN22_FRAME_NUM" "${frame_num}"
require_positive_integer "MOTIVE_WAN22_SAMPLE_STEPS" "${sample_steps}"
require_positive_integer "MOTIVE_WAN22_BASE_SEED" "${base_seed}"
if (( (frame_num - 1) % 4 != 0 )); then
  fail "frame_num must be 4n+1: ${frame_num}"
fi

for required in \
  "${code_root}/motive/wan22_i2v_batch.py" \
  "${code_root}/motive/wan22_signed_release.py" \
  "${generation_manifest}" \
  "${signed_release}" \
  "${wan_code_root}/.git" \
  "${wan_code_root}/generate.py" \
  "${wan_code_root}/wan/image2video.py" \
  "${checkpoint_dir}/.cache/huggingface/download" \
  "${checkpoint_dir}/Wan2.1_VAE.pth" \
  "${checkpoint_dir}/models_t5_umt5-xxl-enc-bf16.pth" \
  "${checkpoint_dir}/high_noise_model/config.json" \
  "${checkpoint_dir}/high_noise_model/diffusion_pytorch_model.safetensors.index.json" \
  "${checkpoint_dir}/low_noise_model/config.json" \
  "${checkpoint_dir}/low_noise_model/diffusion_pytorch_model.safetensors.index.json" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    fail "missing required runtime path: ${required}"
  fi
done
if [[ ! -x "${python_bin}" ]]; then
  fail "Python is not executable: ${python_bin}"
fi
if [[ -L "${ffprobe_bin}" || ! -f "${ffprobe_bin}" || ! -x "${ffprobe_bin}" ]]; then
  fail "ffprobe must be an executable regular non-symlink file: ${ffprobe_bin}"
fi
actual_commit="$(git -C "${wan_code_root}" rev-parse HEAD)"
if [[ "${actual_commit}" != "42bf4cfaa384bc21833865abc2f9e6c0e67233dc" ]]; then
  fail "official Wan checkout commit mismatch: ${actual_commit}"
fi

runner_args=(
  --manifest "${generation_manifest}"
  --signed-release "${signed_release}"
  --output-root "${output_root}"
  --wan-code-root "${wan_code_root}"
  --ckpt-dir "${checkpoint_dir}"
  --official-commit "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
  --model-hf-revision "206a9ee1b7bfaaf8f7e4d81335650533490646a3"
  --size "${size}"
  --frame-num "${frame_num}"
  --sample-steps "${sample_steps}"
  --sample-shift "${sample_shift}"
  --sample-solver unipc
  --sample-guide-scale-low 3.5
  --sample-guide-scale-high 3.5
  --base-seed "${base_seed}"
  --max-new-samples "${max_new_samples}"
  --expected-world-size 8
  --require-rocm
  --expected-gpu-name-substring MI210
  --ffprobe "${ffprobe_bin}"
  --video-codec libx264
  --video-quality 8
)
if [[ -n "${data_root}" ]]; then
  runner_args+=(--data-root "${data_root}")
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${code_root}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="$(( ${SLURM_CPUS_PER_TASK:-128} / 8 ))"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export PYTHONUNBUFFERED=1

cache_parent="${SLURM_TMPDIR:-/tmp}"
cache_root="${cache_parent}/motive-wan22-i2v-smoke-${SLURM_JOB_ID:-manual}"
mkdir -p \
  "${cache_root}/tmp" \
  "${cache_root}/xdg" \
  "${cache_root}/huggingface" \
  "${cache_root}/torch" \
  "${cache_root}/torch-kernels" \
  "${cache_root}/inductor" \
  "${cache_root}/triton" \
  "${cache_root}/miopen" \
  "${cache_root}/miopen-custom"
export TMPDIR="${cache_root}/tmp"
export XDG_CACHE_HOME="${cache_root}/xdg"
export HF_HOME="${cache_root}/huggingface"
export TORCH_HOME="${cache_root}/torch"
export PYTORCH_KERNEL_CACHE_PATH="${cache_root}/torch-kernels"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor"
export TRITON_CACHE_DIR="${cache_root}/triton"
export MIOPEN_USER_DB_PATH="${cache_root}/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="${cache_root}/miopen-custom"
cd "${code_root}"

echo "[wan22-i2v-smoke] host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[wan22-i2v-smoke] eight ranks cooperate on one sample"
echo "[wan22-i2v-smoke] manifest=${generation_manifest}"
echo "[wan22-i2v-smoke] output=${output_root} temporal_grid=source_bound"

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  -m motive.wan22_i2v_batch \
  "${runner_args[@]}"

test -s "${output_root}/run_complete.json"
test -s "${output_root}/generated_manifest.jsonl"
sha256sum "${output_root}/run_complete.json" "${output_root}/generated_manifest.jsonl"
echo "[wan22-i2v-smoke] complete"
