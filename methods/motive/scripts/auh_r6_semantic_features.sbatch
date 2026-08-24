#!/usr/bin/env bash
# One non-array job: UMT5-XXL is unloaded before CLIP is loaded.
# Pass run-unique MOTIVE_R6_INPUT and MOTIVE_R6_OUTPUT with sbatch --export.
#SBATCH --job-name=motive-r6-semantic
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=12:00:00

set -Eeuo pipefail

repo_root="${MOTIVE_REPO_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit}"
source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?MOTIVE_SOURCE_SNAPSHOT is required}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?MOTIVE_SOURCE_TREE_SHA256 is required}"
source_manifest_sha256="${MOTIVE_SOURCE_MANIFEST_SHA256:?MOTIVE_SOURCE_MANIFEST_SHA256 is required}"
code_root="${source_snapshot}/methods/motive"
input_manifest="${MOTIVE_R6_INPUT:-}"
output_root="${MOTIVE_R6_OUTPUT:-}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

umt5_root="${MOTIVE_R6_UMT5_ROOT:-${repo_root}/checkpoints/LucyEdit}"
umt5_revision="${MOTIVE_R6_UMT5_REVISION:-f12c3ab18266dcc1eb97f26b9102af42dfd327c5}"
clip_root="${MOTIVE_R6_CLIP_ROOT:-/vast/users/guangyi.chen/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41}"
clip_revision="${MOTIVE_R6_CLIP_REVISION:-32bd64288804d66eefd0ccbe215aa642df71cc41}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "[r6-semantic] refusing an array submission; use exactly one MI210 job" >&2
  exit 2
fi
if [[ -z "${input_manifest}" ]]; then
  echo "[r6-semantic] MOTIVE_R6_INPUT must name the committed R5 final manifest" >&2
  exit 2
fi
if [[ -z "${output_root}" ]]; then
  echo "[r6-semantic] MOTIVE_R6_OUTPUT must be a fresh run-unique directory" >&2
  exit 2
fi
for required in \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  "${code_root}/motive/r6_semantic_features.py" \
  "${input_manifest}" \
  "${umt5_root}/text_encoder/config.json" \
  "${umt5_root}/tokenizer/tokenizer.json" \
  "${clip_root}/config.json" \
  "${clip_root}/model.safetensors" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[r6-semantic] missing required path: ${required}" >&2
    exit 2
  fi
done

export PYTHONPATH="${code_root}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Prevent the mutable checkout from shadowing the immutable snapshot package.
cd "${code_root}"

# Bind the executable bytes before touching model/data artifacts.
"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

# This is a real PyTorch allocation and GEMM, not an environment-variable or
# rocm-smi proxy.  It makes a bad/empty GPU allocation fail before model hashes
# and UMT5 loading consume substantial time.
"${python_bin}" -c '
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("R6 GPU preflight: expected exactly one visible device")
x = torch.arange(4096, dtype=torch.float32, device="cuda:0").reshape(64, 64)
y = x @ x.T
if not bool(torch.isfinite(y).all().item()):
    raise SystemExit("R6 GPU preflight: non-finite GEMM")
torch.cuda.synchronize()
print(
    "[r6-semantic-gpu] "
    f"device={torch.cuda.get_device_name(0)} "
    f"torch={torch.__version__} hip={getattr(torch.version, '\''hip'\'', None)}"
)
'

"${python_bin}" -m motive.r6_semantic_features extract \
  --input "${input_manifest}" \
  --output-dir "${output_root}" \
  --source-snapshot "${source_snapshot}" \
  --source-tree-sha256 "${source_tree_sha256}" \
  --source-manifest-sha256 "${source_manifest_sha256}" \
  --umt5-root "${umt5_root}" \
  --umt5-revision "${umt5_revision}" \
  --umt5-max-length "${MOTIVE_R6_UMT5_MAX_LENGTH:-256}" \
  --umt5-batch-size "${MOTIVE_R6_UMT5_BATCH_SIZE:-2}" \
  --clip-root "${clip_root}" \
  --clip-revision "${clip_revision}" \
  --clip-max-length "${MOTIVE_R6_CLIP_MAX_LENGTH:-77}" \
  --clip-batch-size "${MOTIVE_R6_CLIP_BATCH_SIZE:-32}" \
  --resume

"${python_bin}" -m motive.r6_semantic_features validate \
  --output-dir "${output_root}"
