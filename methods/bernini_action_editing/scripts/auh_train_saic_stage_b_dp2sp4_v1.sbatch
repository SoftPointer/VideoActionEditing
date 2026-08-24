#!/usr/bin/env bash
#SBATCH --job-name=saic-b-preflight
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=04:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 027

# This launcher is deliberately a WORLD8 runtime preflight in the present
# revision.  SAIC_STAGE_B_MODE=train reaches the executable's fail-closed
# blocker and cannot create an optimizer or output until the native sampler,
# critic, codec/seven-axis, serial-VJP, and fresh rollback bridges are closed.

fail() { echo "[saic-stage-b] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
require_sha256() {
  local path="$1" expected="$2" label="$3"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || fail "${label} hash differs"
}

source_archive="${SAIC_STAGE_B_SOURCE_ARCHIVE:?set source archive}"
source_archive_sha256="${SAIC_STAGE_B_SOURCE_ARCHIVE_SHA256:?set source archive SHA-256}"
source_revision="${SAIC_STAGE_B_SOURCE_REVISION:?set source revision}"
source_manifest="${SAIC_STAGE_B_SOURCE_MANIFEST:?set immutable eight-source manifest}"
source_manifest_sha256="${SAIC_STAGE_B_SOURCE_MANIFEST_SHA256:?set source manifest SHA-256}"
stage_a_adapter="${SAIC_STAGE_A_ADAPTER:?set passed Stage-A adapter.safetensors}"
stage_a_adapter_sha256="${SAIC_STAGE_A_ADAPTER_SHA256:?set Stage-A adapter SHA-256}"
stage_a_receipt="${SAIC_STAGE_A_RECEIPT:?set passed Stage-A receipt}"
stage_a_receipt_sha256="${SAIC_STAGE_A_RECEIPT_SHA256:?set Stage-A receipt SHA-256}"
critic_checkpoint="${SAIC_EVENT_CRITIC_CHECKPOINT:?set qualified critic checkpoint}"
critic_checkpoint_sha256="${SAIC_EVENT_CRITIC_CHECKPOINT_SHA256:?set critic checkpoint SHA-256}"
critic_qualification="${SAIC_EVENT_CRITIC_QUALIFICATION:?set critic qualification receipt}"
critic_qualification_sha256="${SAIC_EVENT_CRITIC_QUALIFICATION_SHA256:?set qualification file SHA-256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini-R checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint content manifest}"
checkpoint_manifest_sha256="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set checkpoint manifest SHA-256}"
python_bin="${SAIC_STAGE_B_PYTHON_BIN:?set vace Python executable}"
output="${SAIC_STAGE_B_OUTPUT:?set fresh prospective output path}"
mode="${SAIC_STAGE_B_MODE:-runtime-preflight}"

for name in source_archive source_manifest stage_a_adapter stage_a_receipt critic_checkpoint critic_qualification bernini_root veomni_root checkpoint checkpoint_manifest python_bin output; do
  require_absolute "${!name}" "${name}"
done
for digest in "${source_archive_sha256}" "${source_manifest_sha256}" "${stage_a_adapter_sha256}" "${stage_a_receipt_sha256}" "${critic_checkpoint_sha256}" "${critic_qualification_sha256}" "${checkpoint_manifest_sha256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || fail "SHA-256 differs"
done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
case "${mode}" in
  artifact-preflight|runtime-preflight|train) ;;
  *) fail "SAIC_STAGE_B_MODE must be artifact-preflight, runtime-preflight, or train" ;;
esac

source_archive="$(realpath -e -- "${source_archive}")"
source_manifest="$(realpath -e -- "${source_manifest}")"
stage_a_adapter="$(realpath -e -- "${stage_a_adapter}")"
stage_a_receipt="$(realpath -e -- "${stage_a_receipt}")"
critic_checkpoint="$(realpath -e -- "${critic_checkpoint}")"
critic_qualification="$(realpath -e -- "${critic_qualification}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"

for file in source_archive source_manifest stage_a_adapter stage_a_receipt critic_checkpoint critic_qualification checkpoint_manifest; do
  [[ -f "${!file}" && ! -L "${!file}" ]] || fail "${file} must be a plain file"
done
for directory in bernini_root veomni_root checkpoint; do
  [[ -d "${!directory}" && ! -L "${!directory}" ]] || fail "${directory} must be a plain directory"
done
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "Python executable differs"
require_sha256 "${source_archive}" "${source_archive_sha256}" "source archive"
require_sha256 "${source_manifest}" "${source_manifest_sha256}" "source manifest"
require_sha256 "${stage_a_adapter}" "${stage_a_adapter_sha256}" "Stage-A adapter"
require_sha256 "${stage_a_receipt}" "${stage_a_receipt_sha256}" "Stage-A receipt"
require_sha256 "${critic_checkpoint}" "${critic_checkpoint_sha256}" "critic checkpoint"
require_sha256 "${critic_qualification}" "${critic_qualification_sha256}" "critic qualification"
require_sha256 "${checkpoint_manifest}" "${checkpoint_manifest_sha256}" "checkpoint manifest"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"

output_parent="$(dirname -- "${output}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent must already exist and be plain"
output_parent="$(realpath -e -- "${output_parent}")"
[[ "${output}" == "${output_parent}/${output##*/}" && "${output##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output must be canonical and safe"
[[ ! -e "${output}" && ! -L "${output}" ]] || fail "output is create-only"

(
  cd "${checkpoint}"
  sha256sum --strict --status -c "${checkpoint_manifest}"
) || fail "checkpoint content verification failed"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/saic-stage-b-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/saic-stage-b-${SLURM_JOB_ID}."*) ;;
    *) cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p -- "${task_scratch}/source-tree" "${task_scratch}/cache" "${task_scratch}/tmp"
archive_copy="${task_scratch}/source.tar"
cp -- "${source_archive}" "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged archive differs"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile

required = {
    "methods/bernini_action_editing/train_saic_stage_b_v1.py",
    "methods/bernini_action_editing/infer_saic_stage_b_v1.py",
    "methods/bernini_action_editing/saic_online_motion_field_v1.py",
    "methods/bernini_action_editing/saic_temporal_action_operator_v2.py",
    "methods/bernini_action_editing/saic_event_reward_v1.py",
    "methods/bernini_action_editing/saic_rollout_preference_set_v1.py",
    "methods/bernini_action_editing/saic_rf_preference_objective_v1.py",
    "methods/bernini_action_editing/saic_inverse_recoverability_v1.py",
    "methods/bernini_action_editing/saic_source_anchor_adapter_v1.py",
    "methods/bernini_action_editing/train_saic_source_anchor_v1.py",
    "methods/bernini_action_editing/build_saic_reversible_source_set_v1.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/inference_sigma_strata.py",
    "methods/bernini_action_editing/source_self_native_ref_contrastive_v3.py",
    "methods/bernini_action_editing/source_self_native_rv2v_guidance.py",
    "methods/bernini_action_editing/source_self_runtime.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/tests/test_train_saic_stage_b_v1.py",
    "methods/bernini_action_editing/tests/test_infer_saic_stage_b_v1.py",
}
seen = set()
with tarfile.open(sys.argv[1], "r:*") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise SystemExit("unsafe source archive member")
        seen.add(path.as_posix().lstrip("./"))
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"source archive lacks SAIC Stage-B closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
find "${method_root}" -type f -exec chmod a-w -- {} +

export MIOPEN_USER_DB_PATH="${task_scratch}/cache/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/cache/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/cache/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/cache/triton"
export XDG_CACHE_HOME="${task_scratch}/cache/xdg"
export PYTHONPYCACHEPREFIX="${task_scratch}/cache/pycache"
export TMPDIR="${task_scratch}/tmp"
export TORCHELASTIC_ERROR_FILE="${task_scratch}/torch-elastic-error.json"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
mkdir -p -- "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${PYTHONPYCACHEPREFIX}" "${TMPDIR}"

for test_file in test_train_saic_stage_b_v1.py test_infer_saic_stage_b_v1.py; do
  PYTHONPATH="${method_root}" "${python_bin}" -B "${method_root}/tests/${test_file}"
done

mode_args=()
case "${mode}" in
  artifact-preflight) mode_args+=(--artifact-preflight-only) ;;
  runtime-preflight) mode_args+=(--runtime-preflight-only) ;;
  train) ;;
esac
master_port=$((30000 + SLURM_JOB_ID % 10000))
echo "[saic-stage-b] allocation=8xMI210 topology=WORLD8/DP2xSP4 exact81=true K=4 mode=${mode}"
echo "[saic-stage-b] expected_status=runtime-incomplete optimizer=false output=false"

PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --nproc_per_node=8 --master_addr=127.0.0.1 --master_port="${master_port}" \
  "${method_root}/train_saic_stage_b_v1.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --checkpoint-content-manifest "${checkpoint_manifest}" \
  --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}" \
  --source-manifest "${source_manifest}" \
  --expected-source-manifest-sha256 "${source_manifest_sha256}" \
  --stage-a-adapter "${stage_a_adapter}" \
  --expected-stage-a-adapter-sha256 "${stage_a_adapter_sha256}" \
  --stage-a-receipt "${stage_a_receipt}" \
  --expected-stage-a-receipt-sha256 "${stage_a_receipt_sha256}" \
  --critic-checkpoint "${critic_checkpoint}" \
  --expected-critic-checkpoint-sha256 "${critic_checkpoint_sha256}" \
  --critic-qualification "${critic_qualification}" \
  --expected-critic-qualification-sha256 "${critic_qualification_sha256}" \
  --output "${output}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  --ack-incomplete-runtime-preflight \
  "${mode_args[@]}"

[[ ! -e "${output}" && ! -L "${output}" ]] || fail "incomplete preflight improperly created output"
if [[ "${mode}" == train ]]; then
  fail "train mode unexpectedly returned success while runtime blockers remain"
fi
echo "SAIC_STAGE_B_WORLD8_PREFLIGHT_ONLY_OK optimizer_updates=0 training_started=false"
