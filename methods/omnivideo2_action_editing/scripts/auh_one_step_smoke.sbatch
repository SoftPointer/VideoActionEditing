#!/usr/bin/env bash
#SBATCH --job-name=pact-omni-1step
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=01:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_archive="${PACT_SOURCE_ARCHIVE:?set PACT_SOURCE_ARCHIVE}"
source_archive_sha256="${PACT_SOURCE_ARCHIVE_SHA256:?set PACT_SOURCE_ARCHIVE_SHA256}"
source_commit="${PACT_SOURCE_COMMIT:?set PACT_SOURCE_COMMIT}"
omnivideo_root="${PACT_OMNIVIDEO_ROOT:?set PACT_OMNIVIDEO_ROOT}"
omnivideo_commit="${PACT_OMNIVIDEO_COMMIT:-adcee0a4a5b439ad3615f825298221b21177d4e3}"
checkpoint_dir="${PACT_CHECKPOINT_DIR:?set PACT_CHECKPOINT_DIR}"
run_dir="${PACT_RUN_DIR:?set PACT_RUN_DIR}"
python_bin="${PACT_PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

expected_transformer_sha256="${PACT_TRANSFORMER_SHA256:-f269fe8c6b35993bbb4ea340c535ee9893928ea215fb8c4be3d5e9f122d844d6}"
expected_special_tokens_sha256="${PACT_SPECIAL_TOKENS_SHA256:-72129ce9ade25aa0fbf738c005d3dc090c1b6c45918580e1c683b6ecef726ad4}"

fail() {
  echo "[pact-smoke] ERROR: $*" >&2
  exit 2
}

is_sha256() {
  [[ "$1" =~ ^[0-9a-f]{64}$ ]]
}

is_git_sha() {
  [[ "$1" =~ ^[0-9a-f]{40}$ ]]
}

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive must be a regular non-symlink file"
[[ -x "${python_bin}" ]] || fail "Python is not executable: ${python_bin}"
[[ -d "${omnivideo_root}/.git" ]] || fail "Omni-Video root is not a Git checkout"
[[ -d "${run_dir%/*}" ]] || fail "run parent does not exist: ${run_dir%/*}"
[[ ! -e "${run_dir}" && ! -L "${run_dir}" ]] || fail "run directory already exists: ${run_dir}"
is_sha256 "${source_archive_sha256}" || fail "invalid PACT_SOURCE_ARCHIVE_SHA256"
is_sha256 "${expected_transformer_sha256}" || fail "invalid transformer SHA-256"
is_sha256 "${expected_special_tokens_sha256}" || fail "invalid special-token SHA-256"
is_git_sha "${source_commit}" || fail "invalid PACT_SOURCE_COMMIT"
is_git_sha "${omnivideo_commit}" || fail "invalid PACT_OMNIVIDEO_COMMIT"

transformer="${checkpoint_dir}/transformer/pytorch_model.pt"
special_tokens="${checkpoint_dir}/special_tokens.pkl"
for file in "${transformer}" "${special_tokens}"; do
  [[ -f "${file}" && ! -L "${file}" ]] || fail "checkpoint input must be a regular non-symlink file: ${file}"
done

actual_source_archive_sha256="$(sha256sum "${source_archive}" | awk '{print $1}')"
[[ "${actual_source_archive_sha256}" == "${source_archive_sha256}" ]] || fail "source archive digest mismatch"
actual_transformer_sha256="$(sha256sum "${transformer}" | awk '{print $1}')"
[[ "${actual_transformer_sha256}" == "${expected_transformer_sha256}" ]] || fail "transformer digest mismatch"
actual_special_tokens_sha256="$(sha256sum "${special_tokens}" | awk '{print $1}')"
[[ "${actual_special_tokens_sha256}" == "${expected_special_tokens_sha256}" ]] || fail "special-token digest mismatch"

actual_omnivideo_commit="$(git -C "${omnivideo_root}" rev-parse HEAD)"
[[ "${actual_omnivideo_commit}" == "${omnivideo_commit}" ]] || fail "Omni-Video commit mismatch"
[[ -z "$(git -C "${omnivideo_root}" status --porcelain --untracked-files=no)" ]] || fail "Omni-Video tracked files are dirty"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

task_scratch="${SLURM_TMPDIR:-/tmp}/pact-omni-smoke-${SLURM_JOB_ID}"
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
project_root="${task_scratch}/source"
pact_root="${project_root}/methods/omnivideo2_action_editing"
[[ -f "${pact_root}/train_omnivideo2_pact.py" ]] || fail "source archive lacks the PACT trainer"
[[ -f "${pact_root}/SOURCE_COMMIT" ]] || fail "source archive lacks SOURCE_COMMIT"
[[ "$(tr -d '\n' < "${pact_root}/SOURCE_COMMIT")" == "${source_commit}" ]] || fail "source commit marker mismatch"

mkdir "${run_dir}"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
printf '[pact-smoke] source_commit=%s\n' "${source_commit}"
printf '[pact-smoke] source_archive_sha256=%s\n' "${source_archive_sha256}"
printf '[pact-smoke] omnivideo_commit=%s\n' "${omnivideo_commit}"
printf '[pact-smoke] transformer_sha256=%s\n' "${actual_transformer_sha256}"
printf '[pact-smoke] special_tokens_sha256=%s\n' "${actual_special_tokens_sha256}"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

"${python_bin}" - "${run_dir}/environment.json" <<'PY'
import importlib
import json
import os
import platform
import sys

import torch
from torch.nn import functional as F

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("the smoke job requires exactly one visible GPU")

packages = {}
for name in ("diffusers", "transformers", "einops", "easydict", "flash_attn"):
    module = importlib.import_module(name)
    packages[name] = {
        "version": getattr(module, "__version__", None),
        "file": getattr(module, "__file__", None),
    }

device = torch.device("cuda:0")
probe = torch.randn(256, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
loss = (probe @ probe.transpose(0, 1)).float().square().mean()
loss.backward()
if probe.grad is None or not bool(torch.isfinite(probe.grad).all()):
    raise RuntimeError("bfloat16 device probe returned non-finite gradients")

mask = torch.rand(1, 1, 3, 8, 8, device=device, dtype=torch.float32)
pooled = F.avg_pool3d(
    F.max_pool3d(mask, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
    kernel_size=(1, 3, 3),
    stride=1,
    padding=(0, 1, 1),
)
if not bool(torch.isfinite(pooled).all()):
    raise RuntimeError("float32 mask morphology probe returned non-finite values")

flash_attn = importlib.import_module("flash_attn")
q = torch.randn(8, 12, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
cu = torch.tensor([0, 8], device=device, dtype=torch.int32)
flash_output = flash_attn.flash_attn_varlen_func(
    q=q,
    k=k,
    v=v,
    cu_seqlens_q=cu,
    cu_seqlens_k=cu,
    max_seqlen_q=8,
    max_seqlen_k=8,
    dropout_p=0.0,
    causal=False,
)
flash_output.float().square().mean().backward()
for name, tensor in (("q", q), ("k", k), ("v", v)):
    if tensor.grad is None or not bool(torch.isfinite(tensor.grad).all()):
        raise RuntimeError(f"flash-attention backward returned invalid {name} gradients")

base = torch.nn.Linear(32, 32, device=device, dtype=torch.bfloat16).requires_grad_(False)
lora_a = torch.nn.Linear(32, 8, bias=False, device=device, dtype=torch.float32)
lora_b = torch.nn.Linear(8, 32, bias=False, device=device, dtype=torch.float32)
lora_optimizer = torch.optim.AdamW(
    list(lora_a.parameters()) + list(lora_b.parameters()), lr=1e-4
)
linear_input = torch.randn(4, 32, device=device, dtype=torch.bfloat16)
before = [parameter.detach().clone() for parameter in lora_a.parameters()]
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    base_output = base(linear_input)
    update = lora_b(lora_a(linear_input.float())).to(base_output.dtype)
    linear_output = base_output + update
linear_output.float().square().mean().backward()
for name, module in (("lora_a", lora_a), ("lora_b", lora_b)):
    for parameter in module.parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"BF16-base/FP32-master probe returned invalid {name} gradients")
lora_optimizer.step()
if not any(
    not torch.equal(old, new.detach())
    for old, new in zip(before, lora_a.parameters())
):
    raise RuntimeError("FP32 LoRA master weights did not update")
for state in lora_optimizer.state.values():
    for key in ("exp_avg", "exp_avg_sq"):
        if state[key].dtype != torch.float32 or not bool(torch.isfinite(state[key]).all()):
            raise RuntimeError(f"AdamW {key} is not a finite FP32 optimizer state")

payload = {
    "format": "pact-auh-environment-v2",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "hostname": platform.node(),
    "python": sys.version,
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "device_name": torch.cuda.get_device_name(device),
    "device_properties": str(torch.cuda.get_device_properties(device)),
    "bf16_matmul_backward": "ok",
    "float32_mask_morphology": "ok",
    "flash_attention_bf16_backward": "ok",
    "bf16_base_fp32_lora_adamw": "ok",
    "packages": packages,
}
with open(sys.argv[1], "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

(cd "${pact_root}" && "${python_bin}" -m unittest discover -s tests -v) \
  2>&1 | tee "${run_dir}/unit_tests.log"

fixture="${run_dir}/fixture"
training_output="${run_dir}/training"
"${python_bin}" "${pact_root}/tools/build_synthetic_smoke_fixture.py" \
  --output-dir "${fixture}"

"${python_bin}" "${pact_root}/train_omnivideo2_pact.py" \
  --config "${fixture}/configs/pact_1_3b_one_step_smoke.json" \
  --manifest "${fixture}/training/training_manifest.jsonl" \
  --payload-root "${fixture}/payloads" \
  --dry-run-contract \
  --dry-run-samples 1 | tee "${run_dir}/contract.json"

/usr/bin/time -v "${python_bin}" "${pact_root}/train_omnivideo2_pact.py" \
  --config "${fixture}/configs/pact_1_3b_one_step_smoke.json" \
  --manifest "${fixture}/training/training_manifest.jsonl" \
  --payload-root "${fixture}/payloads" \
  --omnivideo-root "${omnivideo_root}" \
  --checkpoint-dir "${checkpoint_dir}" \
  --output-dir "${training_output}" \
  --device 0

"${python_bin}" "${pact_root}/tools/verify_gpu_smoke.py" \
  --run-output-dir "${training_output}" \
  --omnivideo-root "${omnivideo_root}" \
  --checkpoint-dir "${checkpoint_dir}" | tee "${run_dir}/verification.json"

"${python_bin}" - "${run_dir}" "${source_commit}" "${source_archive_sha256}" "${omnivideo_commit}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
required_receipts = {
    "environment_sha256": root / "environment.json",
    "unit_tests_sha256": root / "unit_tests.log",
    "contract_sha256": root / "contract.json",
    "verification_sha256": root / "verification.json",
}
digests = {}
for field, path in required_receipts.items():
    if not path.is_file():
        raise RuntimeError(f"required receipt is missing: {path}")
    digests[field] = hashlib.sha256(path.read_bytes()).hexdigest()
receipt = {
    "format": "pact-auh-one-step-smoke-done-v2",
    "complete": True,
    "source_commit": sys.argv[2],
    "source_archive_sha256": sys.argv[3],
    "omnivideo_commit": sys.argv[4],
    **digests,
}
with (root / "done.json").open("x", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[pact-smoke] PASS"
