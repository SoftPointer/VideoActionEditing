#!/usr/bin/env bash

# Run one prompt-matched frozen-base Bernini control inside an existing AUH
# allocation.  The enclosing srun must assign exactly four otherwise-unused
# MI210 devices; this payload never selects physical GPU ordinals itself.

set -Eeuo pipefail

source_archive="${BERNINI_BASECTRL_SOURCE_ARCHIVE:?set source archive}"
source_archive_sha256="${BERNINI_BASECTRL_SOURCE_ARCHIVE_SHA256:?set source archive SHA-256}"
source_revision="${BERNINI_BASECTRL_SOURCE_REVISION:?set source revision}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set Bernini source root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set VeOmni source root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini checkpoint}"
source_video="${BERNINI_BASECTRL_SOURCE_VIDEO:?set exact-81 source video}"
instruction="${BERNINI_BASECTRL_INSTRUCTION:?set edit instruction}"
output_video="${BERNINI_BASECTRL_OUTPUT:?set fresh output .mp4 path}"
seed="${BERNINI_BASECTRL_SEED:?set matched seed}"
num_steps="${BERNINI_BASECTRL_STEPS:-40}"
python_bin="${BERNINI_BASECTRL_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12}"

readonly bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca

fail() { echo "[bernini-base-control] ERROR: $*" >&2; exit 2; }
require_plain_file() { [[ -f "$1" && ! -L "$1" ]] || fail "$2 differs"; }
require_directory() { [[ -d "$1" && ! -L "$1" ]] || fail "$2 differs"; }

[[ "${SLURM_JOB_ID:?existing Slurm allocation required}" =~ ^[1-9][0-9]*$ ]] || fail "Slurm job ID differs"
[[ "${source_archive}" == /* && "${source_video}" == /* && "${output_video}" == /*.mp4 ]] || fail "all artifact paths must be absolute"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ && "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source identity differs"
[[ "${seed}" =~ ^[0-9]+$ && "${num_steps}" =~ ^[1-9][0-9]*$ ]] || fail "sampling scalar differs"
# POSIX environment strings cannot carry NUL; the Python contract still checks
# it independently for direct invocations.
[[ -n "${instruction//[[:space:]]/}" ]] || fail "instruction differs"
require_plain_file "${source_archive}" "source archive"
require_plain_file "${source_video}" "source video"
require_directory "${bernini_root}" "Bernini source root"
require_directory "${veomni_root}" "VeOmni source root"
require_directory "${checkpoint}" "checkpoint"
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ ! -e "${output_video}" && ! -L "${output_video}" ]] || fail "refusing output reuse"
[[ ! -e "${output_video}.receipt.json" && ! -L "${output_video}.receipt.json" ]] || fail "refusing receipt reuse"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

# Fail closed if the enclosing srun exposes anything other than one genuinely
# idle SP4 group.  `srun --overlap` can map a new step onto physical devices
# that another step already uses even when a different physical quartet looks
# idle in a node-wide probe, so device count alone is not a sufficient guard.
visible_probe="$(${python_bin} -I -B - <<'PY'
import json
import torch

count = torch.cuda.device_count()
rows = []
for index in range(count):
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    rows.append(
        {
            "index": index,
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "used_bytes": int(total_bytes - free_bytes),
        }
    )
print(json.dumps({"count": count, "devices": rows}, sort_keys=True, separators=(",", ":")))
if count != 4:
    raise SystemExit(2)
# A fresh ROCm context may consume a small amount of memory.  Anything above
# 1 GiB is treated as an occupied device and aborts before model construction.
if any(row["used_bytes"] > 1024**3 for row in rows):
    raise SystemExit(3)
PY
)" || fail "srun did not expose four unoccupied GPUs"
echo "[bernini-base-control] visible_gpu_preflight=${visible_probe}"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-base-control-${SLURM_JOB_ID}.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/bernini-base-control-${SLURM_JOB_ID}."*) chmod -R u+w -- "${task_scratch}"; rm -rf -- "${task_scratch}" ;;
    *) status=2 ;;
  esac
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p -- "${task_scratch}/source" "${task_scratch}/tmp" \
  "${task_scratch}/cache/miopen-user" "${task_scratch}/cache/miopen-custom" \
  "${task_scratch}/cache/torch-extensions" "${task_scratch}/cache/triton" \
  "$(dirname -- "${output_video}")"
export TMPDIR="${task_scratch}/tmp"
export MIOPEN_USER_DB_PATH="${task_scratch}/cache/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/cache/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/cache/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/cache/triton"

tar --no-same-owner --no-same-permissions -xf "${source_archive}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
require_plain_file "${method_root}/infer_lora.py" "base/LoRA inference harness"
require_plain_file "${method_root}/tests/test_infer_lora_contract.py" "inference contract test"

echo "[bernini-base-control] host=$(hostname) job=${SLURM_JOB_ID} step=${SLURM_STEP_ID:-unknown}"
echo "[bernini-base-control] source_revision=${source_revision} archive_sha256=${source_archive_sha256}"
echo "[bernini-base-control] source_sha256=$(sha256sum "${source_video}" | awk '{print $1}') output=${output_video} seed=${seed} steps=${num_steps}"
rocm-smi --showuse --showmemuse 2>/dev/null || true

PYTHONPATH="${method_root}" "${python_bin}" -B -m unittest discover \
  -s "${method_root}/tests" -p 'test_infer_lora_contract.py' -v

PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  "${method_root}/infer_lora.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --base-only \
  --source-video "${source_video}" \
  --instruction "${instruction}" \
  --output "${output_video}" \
  --num-inference-steps "${num_steps}" \
  --seed "${seed}" \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}"

require_plain_file "${output_video}" "output"
require_plain_file "${output_video}.receipt.json" "receipt"
"${python_bin}" -I -B - "${output_video}.receipt.json" <<'PY'
import hashlib, json, sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
unsigned = dict(value)
claimed = unsigned.pop("receipt_digest", None)
canonical = json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
if claimed != hashlib.sha256(canonical).hexdigest():
    raise SystemExit("receipt digest differs")
adapter = value.get("adapter")
if not isinstance(adapter, dict) or adapter.get("enabled") is not False or adapter.get("mode") != "frozen_base_no_adapter" or adapter.get("tensor_count") != 0:
    raise SystemExit("base-only receipt differs")
if any(key in adapter for key in ("checkpoint_root", "adapter_model_path", "adapter_model_sha256")):
    raise SystemExit("base-only receipt claims adapter identity")
if value.get("scientific_claim_authorized") is not False:
    raise SystemExit("diagnostic receipt gained scientific authority")
PY

echo "BERNINI_FROZEN_BASE_CONTROL_OK output=${output_video}"
