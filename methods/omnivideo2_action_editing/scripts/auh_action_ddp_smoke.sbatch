#!/usr/bin/env bash
#SBATCH --job-name=marp-omni-ddp4
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=01:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_archive="${MARP_SOURCE_ARCHIVE:?set MARP_SOURCE_ARCHIVE}"
source_archive_sha256="${MARP_SOURCE_ARCHIVE_SHA256:?set MARP_SOURCE_ARCHIVE_SHA256}"
source_revision="${MARP_SOURCE_REVISION:?set MARP_SOURCE_REVISION}"
omnivideo_root="${MARP_OMNIVIDEO_ROOT:?set MARP_OMNIVIDEO_ROOT}"
omnivideo_commit="${MARP_OMNIVIDEO_COMMIT:-adcee0a4a5b439ad3615f825298221b21177d4e3}"
checkpoint_dir="${MARP_CHECKPOINT_DIR:?set MARP_CHECKPOINT_DIR}"
run_dir="${MARP_RUN_DIR:?set MARP_RUN_DIR}"
python_bin="${MARP_PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

expected_transformer_sha256="${MARP_TRANSFORMER_SHA256:-f269fe8c6b35993bbb4ea340c535ee9893928ea215fb8c4be3d5e9f122d844d6}"
expected_special_tokens_sha256="${MARP_SPECIAL_TOKENS_SHA256:-72129ce9ade25aa0fbf738c005d3dc090c1b6c45918580e1c683b6ecef726ad4}"

fail() {
  echo "[marp-ddp4] ERROR: $*" >&2
  exit 2
}

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid source archive"
[[ -x "${python_bin}" ]] || fail "Python is not executable: ${python_bin}"
[[ -d "${omnivideo_root}/.git" ]] || fail "Omni root is not a Git checkout"
[[ -d "${run_dir%/*}" ]] || fail "run parent does not exist"
[[ ! -e "${run_dir}" && ! -L "${run_dir}" ]] || fail "run directory already exists"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid source archive digest"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be a Git SHA"
[[ "${omnivideo_commit}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid Omni commit"

transformer="${checkpoint_dir}/transformer/pytorch_model.pt"
special_tokens="${checkpoint_dir}/special_tokens.pkl"
for file in "${transformer}" "${special_tokens}"; do
  [[ -f "${file}" && ! -L "${file}" ]] || fail "invalid checkpoint input: ${file}"
done

[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive digest mismatch"
[[ "$(sha256sum "${transformer}" | awk '{print $1}')" == "${expected_transformer_sha256}" ]] || fail "transformer digest mismatch"
[[ "$(sha256sum "${special_tokens}" | awk '{print $1}')" == "${expected_special_tokens_sha256}" ]] || fail "special-token digest mismatch"
[[ "$(git -C "${omnivideo_root}" rev-parse HEAD)" == "${omnivideo_commit}" ]] || fail "Omni commit mismatch"
[[ -z "$(git -C "${omnivideo_root}" status --porcelain --untracked-files=no)" ]] || fail "Omni tracked files are dirty"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

task_scratch="${SLURM_TMPDIR:-/tmp}/marp-ddp4-${SLURM_JOB_ID}"
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
method_root="${project_root}/methods/omnivideo2_action_editing"
[[ -f "${method_root}/train_omnivideo2_action.py" ]] || fail "archive lacks action trainer"

mkdir "${run_dir}"
hostname
date -u +%Y-%m-%dT%H:%M:%SZ
printf '[marp-ddp4] source_revision=%s\n' "${source_revision}"
printf '[marp-ddp4] source_archive_sha256=%s\n' "${source_archive_sha256}"
printf '[marp-ddp4] omnivideo_commit=%s\n' "${omnivideo_commit}"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

"${python_bin}" - "${run_dir}/environment.json" <<'PY'
import json
import os
import platform
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise RuntimeError(f"expected four visible GPUs, got {torch.cuda.device_count()}")
probe = torch.randn(256, 256, device="cuda:0", dtype=torch.bfloat16, requires_grad=True)
probe.square().mean().backward()
if probe.grad is None or not bool(torch.isfinite(probe.grad).all()):
    raise RuntimeError("BF16 accelerator probe failed")
value = {
    "format": "marp-auh-ddp4-environment-v1",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "hostname": platform.node(),
    "python": sys.version,
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "device_count": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(i) for i in range(4)],
    "bf16_backward": "ok",
}
with open(sys.argv[1], "x", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(value, indent=2, sort_keys=True))
PY

(cd "${method_root}" && "${python_bin}" -m unittest \
  tests.test_action_config \
  tests.test_action_context \
  tests.test_action_dataset \
  tests.test_action_flow \
  tests.test_action_omni \
  tests.test_action_planner \
  tests.test_action_task_gate \
  tests.test_action_synthetic_fixture \
  tests.test_build_action_preview_manifest -v) \
  2>&1 | tee "${run_dir}/unit_tests.log"

fixture="${run_dir}/fixture"
"${python_bin}" "${method_root}/tools/build_action_synthetic_fixture.py" \
  --output-dir "${fixture}"

"${python_bin}" "${method_root}/train_omnivideo2_action.py" \
  --config "${fixture}/configs/marp_one_step.json" \
  --manifest "${fixture}/manifest.jsonl" \
  --payload-root "${fixture}/payloads" \
  --allow-preview-exploration \
  --dry-run-contract \
  --dry-run-samples 0 | tee "${run_dir}/contract.json"

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${method_root}/train_omnivideo2_action.py" \
  --config "${fixture}/configs/marp_one_step.json" \
  --manifest "${fixture}/manifest.jsonl" \
  --payload-root "${fixture}/payloads" \
  --omnivideo-root "${omnivideo_root}" \
  --checkpoint-dir "${checkpoint_dir}" \
  --output-dir "${run_dir}/training" \
  --allow-preview-exploration \
  --source-revision "${source_revision}" \
  --source-archive-sha256 "${source_archive_sha256}"

"${python_bin}" "${method_root}/tools/verify_action_ddp_smoke.py" \
  --run-output-dir "${run_dir}/training" \
  --expected-world-size 4 \
  --expected-source-revision "${source_revision}" | tee "${run_dir}/verification.json"

"${python_bin}" - "${run_dir}" "${source_revision}" "${source_archive_sha256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = {
    "environment_sha256": root / "environment.json",
    "unit_tests_sha256": root / "unit_tests.log",
    "contract_sha256": root / "contract.json",
    "verification_sha256": root / "verification.json",
    "training_done_sha256": root / "training" / "done.json",
}
digests = {}
for field, path in paths.items():
    if not path.is_file():
        raise RuntimeError(f"missing smoke artifact: {path}")
    digests[field] = hashlib.sha256(path.read_bytes()).hexdigest()
receipt = {
    "format": "marp-auh-ddp4-smoke-done-v1",
    "complete": True,
    "synthetic_integration_only": True,
    "motion_editing_quality_not_tested": True,
    "source_revision": sys.argv[2],
    "source_archive_sha256": sys.argv[3],
    **digests,
}
with (root / "smoke_done.json").open("x", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[marp-ddp4] PASS"
