#!/usr/bin/env bash
#SBATCH --job-name=marp-81f-ddp4
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=02:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_archive="${MARP_SOURCE_ARCHIVE:?set MARP_SOURCE_ARCHIVE}"
source_archive_sha256="${MARP_SOURCE_ARCHIVE_SHA256:?set MARP_SOURCE_ARCHIVE_SHA256}"
source_revision="${MARP_SOURCE_REVISION:?set MARP_SOURCE_REVISION}"
materialized_root="${MARP_MATERIALIZED_ROOT:?set MARP_MATERIALIZED_ROOT}"
sample_id="${MARP_SAMPLE_ID:?set MARP_SAMPLE_ID}"
omnivideo_root="${MARP_OMNIVIDEO_ROOT:?set MARP_OMNIVIDEO_ROOT}"
omnivideo_commit="${MARP_OMNIVIDEO_COMMIT:-adcee0a4a5b439ad3615f825298221b21177d4e3}"
checkpoint_dir="${MARP_CHECKPOINT_DIR:?set MARP_CHECKPOINT_DIR}"
run_dir="${MARP_RUN_DIR:?set MARP_RUN_DIR}"
python_bin="${MARP_PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
config_name="${MARP_ACTION_CONFIG_NAME:-marp_81f_fullres_real_one_step.json}"

fail() {
  echo "[marp-real-ddp4] ERROR: $*" >&2
  exit 2
}

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid source archive"
[[ -d "${materialized_root}" && ! -L "${materialized_root}" ]] || fail "invalid materialized root"
for path in manifest.jsonl materialization.json verification.json; do
  [[ -f "${materialized_root}/${path}" && ! -L "${materialized_root}/${path}" ]] || fail "missing materialized ${path}"
done
[[ -d "${omnivideo_root}/.git" ]] || fail "Omni root is not a Git checkout"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ -d "${run_dir%/*}" ]] || fail "run parent does not exist"
[[ ! -e "${run_dir}" && ! -L "${run_dir}" ]] || fail "run directory already exists"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid archive digest"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ "${omnivideo_commit}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid Omni commit"
[[ "${sample_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "unsafe sample ID"
[[ "${config_name}" =~ ^marp_[A-Za-z0-9._-]+\.json$ ]] || fail "unsafe config name"
case "${config_name}" in
  marp_81f_fullres_real_one_step.json|marp_81f_motion384_real_one_step.json) ;;
  *) fail "real DDP4 verifier accepts only an 81-frame config" ;;
esac
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "archive digest mismatch"
[[ "$(git -C "${omnivideo_root}" rev-parse HEAD)" == "${omnivideo_commit}" ]] || fail "Omni commit mismatch"
[[ -z "$(git -C "${omnivideo_root}" status --porcelain --untracked-files=no)" ]] || fail "Omni tracked files are dirty"

transformer="${checkpoint_dir}/transformer/pytorch_model.pt"
special_tokens="${checkpoint_dir}/special_tokens.pkl"
[[ "$(sha256sum "${transformer}" | awk '{print $1}')" == "f269fe8c6b35993bbb4ea340c535ee9893928ea215fb8c4be3d5e9f122d844d6" ]] || fail "transformer digest mismatch"
[[ "$(sha256sum "${special_tokens}" | awk '{print $1}')" == "72129ce9ade25aa0fbf738c005d3dc090c1b6c45918580e1c683b6ecef726ad4" ]] || fail "special-token digest mismatch"

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

task_scratch="${SLURM_TMPDIR:-/tmp}/marp-real-ddp4-${SLURM_JOB_ID}"
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
method_root="${task_scratch}/source/methods/omnivideo2_action_editing"
config="${method_root}/configs/${config_name}"
[[ -f "${config}" ]] || fail "archive lacks real-smoke config"

mkdir "${run_dir}"
hostname
date -u +%Y-%m-%dT%H:%M:%SZ
printf '[marp-real-ddp4] source_archive_sha256=%s\n' "${source_archive_sha256}"
printf '[marp-real-ddp4] materialized_root=%s\n' "${materialized_root}"
printf '[marp-real-ddp4] config=%s\n' "${config_name}"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

(cd "${method_root}" && "${python_bin}" -m unittest \
  tests.test_action_dataset \
  tests.test_action_config \
  tests.test_action_context \
  tests.test_action_omni \
  tests.test_action_runtime \
  tests.test_action_task_gate \
  tests.test_materialize_action_payloads \
  tests.test_verify_action_real_ddp_smoke -v) \
  2>&1 | tee "${run_dir}/unit_tests.log"

"${python_bin}" "${method_root}/train_omnivideo2_action.py" \
  --config "${config}" \
  --manifest "${materialized_root}/manifest.jsonl" \
  --payload-root "${materialized_root}/payloads" \
  --allow-preview-exploration \
  --dry-run-contract \
  --dry-run-samples 1 | tee "${run_dir}/contract.json"

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${method_root}/train_omnivideo2_action.py" \
  --config "${config}" \
  --manifest "${materialized_root}/manifest.jsonl" \
  --payload-root "${materialized_root}/payloads" \
  --omnivideo-root "${omnivideo_root}" \
  --checkpoint-dir "${checkpoint_dir}" \
  --output-dir "${run_dir}/training" \
  --allow-preview-exploration \
  --source-revision "${source_revision}" \
  --source-archive-sha256 "${source_archive_sha256}"

"${python_bin}" "${method_root}/tools/verify_action_real_ddp_smoke.py" \
  --materialized-root "${materialized_root}" \
  --run-output-dir "${run_dir}/training" \
  --omnivideo-root "${omnivideo_root}" \
  --checkpoint-dir "${checkpoint_dir}" \
  --expected-sample-id "${sample_id}" \
  --expected-world-size 4 \
  --expected-source-revision "${source_revision}" | tee "${run_dir}/verification.json"

"${python_bin}" - "${run_dir}" "${source_revision}" "${source_archive_sha256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = {
    "unit_tests_sha256": root / "unit_tests.log",
    "contract_sha256": root / "contract.json",
    "verification_sha256": root / "verification.json",
    "training_done_sha256": root / "training" / "done.json",
}
receipt = {
    "format": "marp-action-real-preview-ddp4-smoke-done-v1",
    "complete": True,
    "preview_only": True,
    "distributed_sample_replication": True,
    "motion_editing_quality_not_tested": True,
    "source_revision": sys.argv[2],
    "source_archive_sha256": sys.argv[3],
}
for field, path in paths.items():
    receipt[field] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "smoke_done.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[marp-real-ddp4] PASS"
