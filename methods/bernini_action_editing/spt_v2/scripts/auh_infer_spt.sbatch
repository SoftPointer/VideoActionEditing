#!/usr/bin/env bash
#SBATCH --job-name=bern-spt-infer81
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

source_archive="${BERNINI_SPT_SOURCE_ARCHIVE:?set BERNINI_SPT_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_SPT_SOURCE_ARCHIVE_SHA256:?set BERNINI_SPT_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_SPT_SOURCE_REVISION:?set BERNINI_SPT_SOURCE_REVISION}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
bernini_archive="${BERNINI_OFFICIAL_ARCHIVE:?set BERNINI_OFFICIAL_ARCHIVE}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
planner_checkpoint="${BERNINI_SPT_PLANNER_CHECKPOINT:?set BERNINI_SPT_PLANNER_CHECKPOINT}"
source_video="${BERNINI_SPT_SOURCE_VIDEO:?set BERNINI_SPT_SOURCE_VIDEO}"
instruction="${BERNINI_SPT_INSTRUCTION:?set BERNINI_SPT_INSTRUCTION}"
output_video="${BERNINI_SPT_INFERENCE_OUTPUT:?set BERNINI_SPT_INFERENCE_OUTPUT}"
seed="${BERNINI_SPT_INFERENCE_SEED:-42}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
bernini_archive_sha256="08a331958560544efb5e842666c236d819bfdc36d34b6eb9c1cdcee1546ae670"
num_steps=40
max_generate_fraction=0.12

fail() { echo "[bernini-spt-infer81] ERROR: $*" >&2; exit 2; }

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid method source archive"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini source root"
[[ -f "${bernini_archive}" && ! -L "${bernini_archive}" ]] || fail "invalid Bernini source archive"
[[ -d "${veomni_root}/.git" ]] || fail "VeOmni root must be a Git checkout"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid base checkpoint"
[[ -d "${planner_checkpoint}" && ! -L "${planner_checkpoint}" ]] || fail "invalid phase-query checkpoint"
for name in planner.safetensors planner_config.json receipt.json optimizer.pt; do
  [[ -f "${planner_checkpoint}/${name}" && ! -L "${planner_checkpoint}/${name}" ]] || fail "planner checkpoint lacks ${name}"
done
[[ ! -e "${planner_checkpoint}/adapter" && ! -L "${planner_checkpoint}/adapter" ]] || fail "planner checkpoint is contaminated by an adapter"
[[ ! -e "${planner_checkpoint}/adapter_config.json" && ! -L "${planner_checkpoint}/adapter_config.json" ]] || fail "planner checkpoint contains PEFT config"
[[ ! -e "${planner_checkpoint}/adapter_model.safetensors" && ! -L "${planner_checkpoint}/adapter_model.safetensors" ]] || fail "planner checkpoint contains PEFT weights"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "invalid source video"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ -n "${instruction//[[:space:]]/}" ]] || fail "instruction is empty"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid method archive hash"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid method source revision"
[[ "${seed}" =~ ^[0-9]+$ ]] || fail "seed must be non-negative"
[[ "${output_video}" == /*.mp4 ]] || fail "output must be an absolute .mp4 path"
[[ ! -e "${output_video}" && ! -L "${output_video}" ]] || fail "refusing to overwrite output"
[[ ! -e "${output_video}.receipt.json" && ! -L "${output_video}.receipt.json" ]] || fail "refusing to overwrite receipt"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive hash mismatch"
[[ "$(sha256sum "${bernini_archive}" | awk '{print $1}')" == "${bernini_archive_sha256}" ]] || fail "Bernini archive hash mismatch"
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

task_scratch="${SLURM_TMPDIR:-/tmp}/bernini-spt-infer81-${SLURM_JOB_ID}"
mkdir -p \
  "${task_scratch}/source" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton" \
  "${output_video%/*}"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"

tar -xf "${source_archive}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
[[ -f "${method_root}/spt_v2/infer_spt.py" ]] || fail "archive lacks SPT inference entry"
[[ -f "${method_root}/spt_v2/unipc_projection.py" ]] || fail "archive lacks UniPC projection boundary"
[[ -f "${method_root}/spt_v2/phase_query_planner.py" ]] || fail "archive lacks phase_query_v2 planner"
[[ -f "${method_root}/tests/test_infer_spt_contract.py" ]] || fail "archive lacks SPT inference tests"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-spt-infer81] source_revision=${source_revision}"
echo "[bernini-spt-infer81] source_archive_sha256=${source_archive_sha256}"
echo "[bernini-spt-infer81] planner_checkpoint=${planner_checkpoint}"
echo "[bernini-spt-infer81] planner_sha256=$(sha256sum "${planner_checkpoint}/planner.safetensors" | awk '{print $1}')"
echo "[bernini-spt-infer81] source_video_sha256=$(sha256sum "${source_video}" | awk '{print $1}')"
echo "[bernini-spt-infer81] output=${output_video}"
echo "[bernini-spt-infer81] base_lora=none frames=81 fps=25 steps=40 guidance=v2v_apg shift=5 ulysses=4 generate_budget_per_phase=0.12"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

for pattern in test_spt_v2_unipc_projection.py test_infer_spt_contract.py; do
  PYTHONPATH="${method_root}" "${python_bin}" -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${method_root}/spt_v2/infer_spt.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --planner-checkpoint "${planner_checkpoint}" \
  --source-video "${source_video}" \
  --instruction "${instruction}" \
  --output "${output_video}" \
  --num-inference-steps "${num_steps}" \
  --max-generate-fraction "${max_generate_fraction}" \
  --seed "${seed}" \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}"

[[ -f "${output_video}" && ! -L "${output_video}" ]] || fail "inference output missing"
[[ -f "${output_video}.receipt.json" && ! -L "${output_video}.receipt.json" ]] || fail "inference receipt missing"
rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-spt-infer81] PASS"
