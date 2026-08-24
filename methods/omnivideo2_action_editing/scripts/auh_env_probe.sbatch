#!/usr/bin/env bash
#SBATCH --job-name=pact-omni-env
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=00:10:00

set -Eeuo pipefail

python_bin="${PACT_PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  echo "PACT_PYTHON_BIN is not executable: ${python_bin}" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

task_scratch="${SLURM_TMPDIR:-/tmp}/pact-omni-env-${SLURM_JOB_ID}"
mkdir -p "${task_scratch}/miopen-user" "${task_scratch}/miopen-custom"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
printf 'python=%s\n' "${python_bin}"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

"${python_bin}" - <<'PY'
import importlib
import importlib.util
import json
import platform
import sys

import torch

result = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "packages": {},
}
for name in ("diffusers", "transformers", "einops", "easydict", "flash_attn"):
    spec = importlib.util.find_spec(name)
    entry = {"available": spec is not None, "origin": None if spec is None else spec.origin}
    if spec is not None:
        try:
            module = importlib.import_module(name)
            entry["version"] = getattr(module, "__version__", None)
        except Exception as exc:
            entry["import_error"] = f"{type(exc).__name__}: {exc}"
    result["packages"][name] = entry

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError(f"expected exactly one visible GPU, got {result}")
device = torch.device("cuda:0")
result["device_name"] = torch.cuda.get_device_name(device)
result["device_properties"] = str(torch.cuda.get_device_properties(device))

x = torch.randn(256, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
y = (x @ x.transpose(0, 1)).float().square().mean()
y.backward()
if x.grad is None or not bool(torch.isfinite(x.grad).all()):
    raise RuntimeError("MI210 bfloat16 matmul backward produced invalid gradients")
result["bf16_matmul_backward"] = "ok"
result["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
print(json.dumps(result, indent=2, sort_keys=True))
PY
