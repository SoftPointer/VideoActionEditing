#!/usr/bin/env bash
#SBATCH --job-name=bern-t2v-motion-oracle
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=01:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

# Frozen Bernini-R 1.3B text-to-video diagnostic.  This does not claim to be
# an editor: it answers the narrower question whether the renderer's
# generation mode contains the requested action before that field is bridged
# back into the full-source V2V path.

set -Eeuo pipefail
umask 077

bernini_root="${BERNINI_T2V_ORACLE_ROOT:?set BERNINI_T2V_ORACLE_ROOT}"
bernini_archive="${BERNINI_T2V_ORACLE_ARCHIVE:?set BERNINI_T2V_ORACLE_ARCHIVE}"
veomni_root="${BERNINI_T2V_ORACLE_VEOMNI_ROOT:?set BERNINI_T2V_ORACLE_VEOMNI_ROOT}"
checkpoint="${BERNINI_T2V_ORACLE_CHECKPOINT:?set BERNINI_T2V_ORACLE_CHECKPOINT}"
prompt="${BERNINI_T2V_ORACLE_PROMPT:?set BERNINI_T2V_ORACLE_PROMPT}"
output="${BERNINI_T2V_ORACLE_OUTPUT:?set BERNINI_T2V_ORACLE_OUTPUT}"
seed="${BERNINI_T2V_ORACLE_SEED:-2027}"
height="${BERNINI_T2V_ORACLE_HEIGHT:-496}"
width="${BERNINI_T2V_ORACLE_WIDTH:-480}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
bernini_archive_sha256="08a331958560544efb5e842666c236d819bfdc36d34b6eb9c1cdcee1546ae670"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
wan_diffusion_sha256="59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512"

fail() { echo "[bernini-t2v-motion-oracle] ERROR: $*" >&2; exit 2; }

[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini root"
[[ -f "${bernini_archive}" && ! -L "${bernini_archive}" ]] || fail "invalid Bernini archive"
[[ -f "${bernini_root}/bernini/models/wan_diffusion.py" && ! -L "${bernini_root}/bernini/models/wan_diffusion.py" ]] || fail "invalid wan_diffusion.py"
[[ -d "${veomni_root}/.git" ]] || fail "VeOmni root is not a Git checkout"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid checkpoint"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ -n "${prompt//[[:space:]]/}" ]] || fail "prompt is empty"
[[ "${output}" == /*.mp4 ]] || fail "output must be one absolute MP4 path"
[[ ! -e "${output}" && ! -L "${output}" ]] || fail "refusing to overwrite output"
[[ "${seed}" =~ ^[0-9]+$ ]] || fail "seed must be non-negative"
[[ "${height}" =~ ^[1-9][0-9]*$ && "${width}" =~ ^[1-9][0-9]*$ ]] || fail "invalid geometry"
[[ "$(sha256sum "${bernini_archive}" | awk '{print $1}')" == "${bernini_archive_sha256}" ]] || fail "Bernini archive differs"
[[ "$(sha256sum "${bernini_root}/bernini/models/wan_diffusion.py" | awk '{print $1}')" == "${wan_diffusion_sha256}" ]] || fail "wan_diffusion.py differs"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_commit}" ]] || fail "VeOmni commit differs"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source is dirty"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf
# The pinned Bernini checkout imports VeOmni as a source dependency; neither
# repository is installed into the shared vace environment.
export PYTHONPATH="${veomni_root}:${bernini_root}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-t2v-oracle-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton" \
  "${output%/*}"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"

echo "[bernini-t2v-motion-oracle] bernini_commit=${bernini_commit}"
echo "[bernini-t2v-motion-oracle] veomni_commit=${veomni_commit}"
echo "[bernini-t2v-motion-oracle] checkpoint_tree_sha256=${checkpoint_tree_sha256}"
echo "[bernini-t2v-motion-oracle] seed=${seed} frames=81 fps=25 height=${height} width=${width}"
echo "[bernini-t2v-motion-oracle] guidance=t2v_apg steps=40 flow_shift=5 omega_txt=4"
echo "[bernini-t2v-motion-oracle] editor_claim=false source_video_used=false"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

cd "${bernini_root}"
"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  infer_multi_gpu.py \
  --config "${checkpoint}" \
  --ulysses 4 \
  --task_type t2v \
  --prompt "${prompt}" \
  --output "${output}" \
  --guidance_mode t2v_apg \
  --num_frames 81 \
  --height "${height}" \
  --width "${width}" \
  --fps 25 \
  --num_inference_steps 40 \
  --flow_shift 5.0 \
  --omega_txt 4.0 \
  --omega_scale 0.8 \
  --eta 0.5 \
  --momentum 0 \
  --seed "${seed}"

[[ -f "${output}" && ! -L "${output}" ]] || fail "oracle output missing"
echo "[bernini-t2v-motion-oracle] output_sha256=$(sha256sum "${output}" | awk '{print $1}')"
echo "[bernini-t2v-motion-oracle] PASS"
