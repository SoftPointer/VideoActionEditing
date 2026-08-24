#!/usr/bin/env bash
#SBATCH --job-name=fitq-owner-xq-2x4
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=08:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# Engineering micro only.  The two registered seed cells are assigned whole
# to two isolated WORLD4/Ulysses4 groups.  Each group persistently loads one
# frozen Bernini 1.3B model, natively replays four fixed T2V/APG owner
# trajectories, and runs the full owner x scheduler-state x prompt observer
# Cartesian product.  Cross-query fields are discarded; only the untouched
# owner field reaches APG and the original UniPC scheduler.  No bank
# intermediate state is expected or reconstructed: the replay is accepted
# only if both the observed native Gaussian and final clean FP32 latent match
# the finalized factor-bank artifacts exactly.

source_archive="${FITQ_CROSS_SOURCE_ARCHIVE:?set FITQ_CROSS_SOURCE_ARCHIVE}"
source_archive_sha256="${FITQ_CROSS_SOURCE_ARCHIVE_SHA256:?set FITQ_CROSS_SOURCE_ARCHIVE_SHA256}"
source_revision="${FITQ_CROSS_SOURCE_REVISION:?set FITQ_CROSS_SOURCE_REVISION}"
manifest="${FITQ_CROSS_FACTOR_MANIFEST:?set FITQ_CROSS_FACTOR_MANIFEST}"
manifest_sha256="${FITQ_CROSS_FACTOR_MANIFEST_SHA256:?set FITQ_CROSS_FACTOR_MANIFEST_SHA256}"
bank_receipt="${FITQ_CROSS_BANK_RECEIPT:?set FITQ_CROSS_BANK_RECEIPT}"
bank_receipt_sha256="${FITQ_CROSS_BANK_RECEIPT_SHA256:?set FITQ_CROSS_BANK_RECEIPT_SHA256}"
bank_output_root="${FITQ_CROSS_BANK_OUTPUT_ROOT:?set FITQ_CROSS_BANK_OUTPUT_ROOT}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
output_dir="${FITQ_CROSS_OUTPUT_DIR:?set FITQ_CROSS_OUTPUT_DIR}"
python_bin="${FITQ_CROSS_PYTHON_BIN:?set FITQ_CROSS_PYTHON_BIN}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
sp4_a_visible_gpus="0,1,2,3"
sp4_b_visible_gpus="4,5,6,7"

fail() {
  echo "[fitq-owner-xq] ERROR: $*" >&2
  exit 2
}

require_absolute() {
  local value="$1"
  local label="$2"
  [[ "${value}" == /* ]] || fail "${label} must be absolute"
}

for value_name in \
  source_archive manifest bank_receipt bank_output_root bernini_root \
  veomni_root checkpoint checkpoint_manifest output_dir python_bin
do
  require_absolute "${!value_name}" "${value_name}"
done

[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source archive SHA-256 must be lowercase"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be a full lowercase SHA-1"
[[ "${manifest_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "factor manifest SHA-256 must be lowercase"
[[ "${bank_receipt_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "bank receipt SHA-256 must be lowercase"

source_archive="$(realpath -e -- "${source_archive}")"
manifest="$(realpath -e -- "${manifest}")"
bank_receipt="$(realpath -e -- "${bank_receipt}")"
bank_output_root="$(realpath -e -- "${bank_output_root}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
launcher_path="$(realpath -e -- "$0")"

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "method source archive is not plain"
[[ -f "${manifest}" && ! -L "${manifest}" ]] || fail "factor manifest is not plain"
[[ -f "${bank_receipt}" && ! -L "${bank_receipt}" ]] || fail "factor bank receipt is not plain"
[[ -d "${bank_output_root}" && ! -L "${bank_output_root}" ]] || fail "factor bank root is invalid"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini source root is invalid"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni source root is invalid"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint root is invalid"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest is not plain"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "runtime Python is invalid"
[[ -f "${launcher_path}" && ! -L "${launcher_path}" ]] || fail "launcher is not plain"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method source archive hash differs"
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha256}" ]] || fail "factor manifest hash differs"
[[ "$(sha256sum "${bank_receipt}" | awk '{print $1}')" == "${bank_receipt_sha256}" ]] || fail "factor bank receipt hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "method archive embedded revision differs"

[[ "${output_dir}" != / ]] || fail "output directory cannot be filesystem root"
output_basename="${output_dir##*/}"
[[ "${output_basename}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename is unsafe"
[[ "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output directory must be canonical"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing to reuse output directory"
output_parent="$(dirname -- "${output_dir}")"
mkdir -p -- "${output_parent}"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent is invalid"
[[ "$(realpath -e -- "${output_parent}")" == "${output_parent}" ]] || fail "output parent must be canonical"

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

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "invalid Slurm job ID"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent is invalid"
task_scratch="$(mktemp -d "${scratch_parent%/}/fitq-owner-xq-${SLURM_JOB_ID}.XXXXXX")"
sp4_a_pid=""
sp4_b_pid=""

cleanup() {
  local exit_status=$?
  local cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/fitq-owner-xq-${SLURM_JOB_ID}."*) ;;
    *) echo "[fitq-owner-xq] refusing unsafe scratch cleanup" >&2; cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  if [[ "${cleanup_failed}" != 0 && "${exit_status}" == 0 ]]; then
    exit_status=2
  fi
  if [[ "${exit_status}" == 0 ]]; then
    echo "[fitq-owner-xq] PASS topology=two-isolated-WORLD4 output=${output_dir}"
  else
    echo "[fitq-owner-xq] EXIT status=${exit_status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${exit_status}"
}

terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${sp4_a_pid}" "${sp4_b_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then
      kill "${child}" 2>/dev/null || true
    fi
  done
  for child in "${sp4_a_pid}" "${sp4_b_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then
      wait "${child}" 2>/dev/null || true
    fi
  done
  exit "${status}"
}

trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -p -- "${task_scratch}/source-tree" "${task_scratch}/groups"
archive_copy="${task_scratch}/method-source.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod 0444 -- "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged archive hash differs"
[[ "$(git get-tar-commit-id <"${archive_copy}")" == "${source_revision}" ]] || fail "staged archive revision differs"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

required = {
    "methods/bernini_action_editing/dmiq_t2v_factorial_bank.py",
    "methods/bernini_action_editing/infer_fitq_owner_prompt_cross_query_micro.py",
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
    "methods/bernini_action_editing/inference_sigma_strata.py",
    "methods/bernini_action_editing/internal_temporal_quotient_observer.py",
    "methods/bernini_action_editing/tri_branch_unipc.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
    "methods/bernini_action_editing/source_kv_replay.py",
    "methods/bernini_action_editing/source_kv_route_batches.py",
    "methods/bernini_action_editing/source_value_residual.py",
    "methods/bernini_action_editing/scripts/auh_infer_fitq_owner_prompt_cross_query_micro_dual4.sbatch",
}
seen = set()
with tarfile.open(sys.argv[1], "r:*") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit("archive member escaped repository")
        normalized = path.as_posix().lstrip("./")
        if member.issym() or member.islnk() or member.isfifo() or member.isdev():
            raise SystemExit("archive contains a link or device")
        if normalized.startswith("methods/bernini_action_editing/"):
            seen.add(normalized)
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"archive lacks owner cross-query closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
runtime_path="${method_root}/infer_fitq_owner_prompt_cross_query_micro.py"
staged_launcher="${method_root}/scripts/auh_infer_fitq_owner_prompt_cross_query_micro_dual4.sbatch"
[[ -f "${runtime_path}" && ! -L "${runtime_path}" ]] || fail "staged runtime is absent"
[[ -f "${staged_launcher}" && ! -L "${staged_launcher}" ]] || fail "staged launcher is absent"
find "${method_root}" -type f -exec chmod a-w -- {} +
launcher_source_sha256="$(sha256sum "${staged_launcher}" | awk '{print $1}')"
[[ "$(sha256sum "${launcher_path}" | awk '{print $1}')" == "${launcher_source_sha256}" ]] || fail "executed launcher differs from source archive"

mkdir -- "${output_dir}"
mkdir -- "${output_dir}/logs"
cp -- "${manifest}" "${output_dir}/factor-manifest.json"
cp -- "${bank_receipt}" "${output_dir}/factor-bank.receipt.json"
chmod 0444 -- "${output_dir}/factor-manifest.json" "${output_dir}/factor-bank.receipt.json"
[[ "$(sha256sum "${output_dir}/factor-manifest.json" | awk '{print $1}')" == "${manifest_sha256}" ]] || fail "persistent manifest hash differs"
[[ "$(sha256sum "${output_dir}/factor-bank.receipt.json" | awk '{print $1}')" == "${bank_receipt_sha256}" ]] || fail "persistent bank receipt hash differs"

job_mod=$((SLURM_JOB_ID % 10000))
sp4_a_master_port=$((30000 + 2 * job_mod))
sp4_b_master_port=$((sp4_a_master_port + 1))
[[ "${sp4_a_master_port}" -ne "${sp4_b_master_port}" ]] || fail "rendezvous ports collided"

launch_group() (
  set -Eeuo pipefail
  local group="$1"
  local visible_gpus="$2"
  local master_port="$3"
  local group_root="${task_scratch}/groups/${group}"
  local group_output="${output_dir}/${group}"
  mkdir -p -- \
    "${group_root}/cache/miopen-user" \
    "${group_root}/cache/miopen-custom" \
    "${group_root}/cache/torch-extensions" \
    "${group_root}/cache/triton" \
    "${group_root}/cache/xdg" \
    "${group_root}/cache/pycache" \
    "${group_root}/tmp"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible_gpus}"
  export MIOPEN_USER_DB_PATH="${group_root}/cache/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${group_root}/cache/miopen-custom"
  export TORCH_EXTENSIONS_DIR="${group_root}/cache/torch-extensions"
  export TRITON_CACHE_DIR="${group_root}/cache/triton"
  export XDG_CACHE_HOME="${group_root}/cache/xdg"
  export PYTHONPYCACHEPREFIX="${group_root}/cache/pycache"
  export TMPDIR="${group_root}/tmp"
  export TORCHELASTIC_ERROR_FILE="${group_root}/torch-elastic.json"
  export BERNINI_OUTPUT_TRANSACTION_ID="${SLURM_JOB_ID}-${group}-fitq-owner-xq"

  echo "[fitq-owner-xq] group=${group} owners=4 states=4 prompts=10 native_replay=true"
  "${python_bin}" -B -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port="${master_port}" \
    "${runtime_path}" \
    --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --factor-manifest "${output_dir}/factor-manifest.json" \
    --expected-factor-manifest-file-sha256 "${manifest_sha256}" \
    --factor-bank-receipt "${output_dir}/factor-bank.receipt.json" \
    --expected-factor-bank-receipt-file-sha256 "${bank_receipt_sha256}" \
    --bank-output-root "${bank_output_root}" \
    --execution-group "${group}" \
    --output-dir "${group_output}" \
    --runtime-source-revision "${source_revision}" \
    --runtime-source-archive-sha256 "${source_archive_sha256}" \
    --launcher-source-sha256 "${launcher_source_sha256}" \
    --expected-bernini-commit "${bernini_commit}" \
    --expected-veomni-commit "${veomni_commit}" \
    --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}"
)

sp4_a_log="${output_dir}/logs/sp4-a.log"
sp4_b_log="${output_dir}/logs/sp4-b.log"
echo "[fitq-owner-xq] semantic_inputs=fixed_factor_bank_prompts source=false target=false mask=false flow=false pose=false track=false trajectory=false reference=false first_frame=false external_noise=false"
launch_group sp4-a "${sp4_a_visible_gpus}" "${sp4_a_master_port}" >"${sp4_a_log}" 2>&1 &
sp4_a_pid=$!
launch_group sp4-b "${sp4_b_visible_gpus}" "${sp4_b_master_port}" >"${sp4_b_log}" 2>&1 &
sp4_b_pid=$!

sp4_a_status=0
sp4_b_status=0
wait "${sp4_a_pid}" || sp4_a_status=$?
wait "${sp4_b_pid}" || sp4_b_status=$?
if [[ "${sp4_a_status}" -ne 0 || "${sp4_b_status}" -ne 0 ]]; then
  tail -n 200 -- "${sp4_a_log}" >&2 || true
  tail -n 200 -- "${sp4_b_log}" >&2 || true
  fail "one or more owner cross-query WORLD4 groups failed"
fi

for group in sp4-a sp4-b; do
  [[ -f "${output_dir}/${group}/receipt.json" && ! -L "${output_dir}/${group}/receipt.json" ]] || fail "${group} receipt is absent"
done
echo "FITQ_OWNER_PROMPT_CROSS_QUERY_MICRO_COMPLETE_ANALYSIS_AND_ALL_SCIENTIFIC_GATES_PENDING"
