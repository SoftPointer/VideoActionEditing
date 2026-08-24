#!/usr/bin/env bash
#SBATCH --job-name=pair5-t2v-guidance
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=24:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_archive="${PAIR_V5_SOURCE_ARCHIVE:?set PAIR_V5_SOURCE_ARCHIVE}"
source_archive_sha256="${PAIR_V5_SOURCE_ARCHIVE_SHA256:?set PAIR_V5_SOURCE_ARCHIVE_SHA256}"
source_revision="${PAIR_V5_SOURCE_REVISION:?set PAIR_V5_SOURCE_REVISION}"
event_manifest="${PAIR_V5_T2V_GUIDANCE_MANIFEST:?set PAIR_V5_T2V_GUIDANCE_MANIFEST}"
event_manifest_sha256="${PAIR_V5_T2V_GUIDANCE_MANIFEST_SHA256:?set PAIR_V5_T2V_GUIDANCE_MANIFEST_SHA256}"
cagd_evidence="${PAIR_V5_CAGD_VALIDATOR_EVIDENCE:?set PAIR_V5_CAGD_VALIDATOR_EVIDENCE}"
cagd_evidence_sha256="${PAIR_V5_CAGD_VALIDATOR_EVIDENCE_SHA256:?set PAIR_V5_CAGD_VALIDATOR_EVIDENCE_SHA256}"
scorer_group_a="${PAIR_V5_T2V_SCORER_GROUP_A_RECEIPT:?set PAIR_V5_T2V_SCORER_GROUP_A_RECEIPT}"
scorer_group_a_sha256="${PAIR_V5_T2V_SCORER_GROUP_A_RECEIPT_SHA256:?set PAIR_V5_T2V_SCORER_GROUP_A_RECEIPT_SHA256}"
scorer_group_b="${PAIR_V5_T2V_SCORER_GROUP_B_RECEIPT:?set PAIR_V5_T2V_SCORER_GROUP_B_RECEIPT}"
scorer_group_b_sha256="${PAIR_V5_T2V_SCORER_GROUP_B_RECEIPT_SHA256:?set PAIR_V5_T2V_SCORER_GROUP_B_RECEIPT_SHA256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
checkpoint_manifest_sha256="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256}"
output_dir="${PAIR_V5_T2V_GUIDANCE_OUTPUT_DIR:?set PAIR_V5_T2V_GUIDANCE_OUTPUT_DIR}"
python_bin="${PAIR_V5_PYTHON_BIN:?set PAIR_V5_PYTHON_BIN}"
schedule_steps="${PAIR_V5_T2V_GUIDANCE_SCHEDULE_STEPS:-40}"
learning_rate="${PAIR_V5_T2V_GUIDANCE_LR:-1e-6}"

fail() { echo "[pair-v5-t2v-guidance] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }

for name in source_archive event_manifest cagd_evidence scorer_group_a scorer_group_b bernini_root veomni_root checkpoint checkpoint_manifest output_dir python_bin; do
  require_absolute "${!name}" "${name}"
done
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source archive SHA-256 differs"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${event_manifest_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "event manifest SHA-256 differs"
[[ "${cagd_evidence_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "CAGD evidence SHA-256 differs"
[[ "${scorer_group_a_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "scorer group A SHA-256 differs"
[[ "${scorer_group_b_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "scorer group B SHA-256 differs"
[[ "${checkpoint_manifest_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "checkpoint manifest SHA-256 differs"
[[ "${schedule_steps}" =~ ^[0-9]+$ ]] || fail "schedule steps must be decimal"
(( schedule_steps >= 40 && schedule_steps % 40 == 0 )) || fail "schedule steps must be a positive exact40 multiple"

source_archive="$(realpath -e -- "${source_archive}")"
event_manifest="$(realpath -e -- "${event_manifest}")"
cagd_evidence="$(realpath -e -- "${cagd_evidence}")"
scorer_group_a="$(realpath -e -- "${scorer_group_a}")"
scorer_group_b="$(realpath -e -- "${scorer_group_b}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive is not a plain file"
[[ -f "${event_manifest}" && ! -L "${event_manifest}" ]] || fail "event manifest is not a plain file"
[[ -f "${cagd_evidence}" && ! -L "${cagd_evidence}" ]] || fail "CAGD evidence is not a plain file"
[[ -f "${scorer_group_a}" && ! -L "${scorer_group_a}" ]] || fail "scorer group A is not a plain file"
[[ -f "${scorer_group_b}" && ! -L "${scorer_group_b}" ]] || fail "scorer group B is not a plain file"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini root differs"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni root differs"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint root differs"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest differs"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "Python executable differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "$(sha256sum "${event_manifest}" | awk '{print $1}')" == "${event_manifest_sha256}" ]] || fail "event manifest hash differs"
[[ "$(sha256sum "${cagd_evidence}" | awk '{print $1}')" == "${cagd_evidence_sha256}" ]] || fail "CAGD evidence hash differs"
[[ "$(sha256sum "${scorer_group_a}" | awk '{print $1}')" == "${scorer_group_a_sha256}" ]] || fail "scorer group A hash differs"
[[ "$(sha256sum "${scorer_group_b}" | awk '{print $1}')" == "${scorer_group_b_sha256}" ]] || fail "scorer group B hash differs"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest hash differs"

[[ "${output_dir}" != / ]] || fail "output cannot be root"
output_basename="${output_dir##*/}"
[[ "${output_basename}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$ ]] || fail "output basename is unsafe"
[[ "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output must be canonical"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing output reuse"
output_parent="$(dirname -- "${output_dir}")"
mkdir -p -- "${output_parent}"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/pair-v5-t2v-guidance-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in "${scratch_parent%/}/pair-v5-t2v-guidance-${SLURM_JOB_ID}."*) ;; *) cleanup_failed=1 ;; esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[pair-v5-t2v-guidance] PASS topology=DP2xSP4 GPUs=8 exact81=true exact40=true output=${output_dir}"
  else
    echo "[pair-v5-t2v-guidance] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p -- "${task_scratch}/source-tree" "${task_scratch}/cache/miopen-user" \
  "${task_scratch}/cache/miopen-custom" "${task_scratch}/cache/torch-extensions" \
  "${task_scratch}/cache/triton" "${task_scratch}/cache/xdg" \
  "${task_scratch}/cache/pycache" "${task_scratch}/tmp"
archive_copy="${task_scratch}/method-source.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod 0444 -- "${archive_copy}"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile
required = {
    "methods/bernini_action_editing/pair_v5_t2v_guidance_distill.py",
    "methods/bernini_action_editing/train_pair_v5_t2v_guidance_distill.py",
    "methods/bernini_action_editing/validate_pair_v5_cagd_evidence_v3.py",
    "methods/bernini_action_editing/finalize_pair_v5_t2v_cagd_v3.py",
    "methods/bernini_action_editing/pair_v5_t2v_energy_calibration_v3.py",
    "methods/bernini_action_editing/score_pair_v5_t2v_energy_bank_v3.py",
    "methods/bernini_action_editing/pair_v5_t2v_calibration_bank_spec.py",
    "methods/bernini_action_editing/infer_pair_v5_t2v_calibration_bank.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/pair_v5_action_adapter.py",
    "methods/bernini_action_editing/pair_v5_native_bridge.py",
    "methods/bernini_action_editing/mace_candidate_action_energy.py",
    "methods/bernini_action_editing/dclr_runtime_contract.py",
    "methods/bernini_action_editing/source_self_native_ref_contrastive_v3.py",
    "methods/bernini_action_editing/source_self_runtime.py",
    "methods/bernini_action_editing/train_lora.py",
}
seen = set()
with tarfile.open(sys.argv[1], "r:*") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise SystemExit("source archive member escaped or is not a plain tree entry")
        seen.add(path.as_posix().lstrip("./"))
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"source archive lacks guidance trainer closure: {missing}")
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

master_port=$((30000 + SLURM_JOB_ID % 10000))
echo "[pair-v5-t2v-guidance] allocation=8xMI210 topology=DP2xUlysses-SP4 frames=81 schedule_steps=${schedule_steps}"
echo "[pair-v5-t2v-guidance] pure_t2v_role=same_coordinate_field_query_only rv2v_target=false rv2v_input=false rv2v_noise=false donor=false mask=false flow=false pose=false track=false"

"${python_bin}" -B -m torch.distributed.run \
  --nproc_per_node=8 --master_addr=127.0.0.1 --master_port="${master_port}" \
  "${method_root}/train_pair_v5_t2v_guidance_distill.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --checkpoint-content-manifest "${checkpoint_manifest}" \
  --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}" \
  --event-manifest "${event_manifest}" \
  --expected-event-manifest-sha256 "${event_manifest_sha256}" \
  --cagd-validator-evidence "${cagd_evidence}" \
  --expected-cagd-validator-evidence-sha256 "${cagd_evidence_sha256}" \
  --scorer-group-receipt "${scorer_group_a}" \
  --expected-scorer-group-receipt-sha256 "${scorer_group_a_sha256}" \
  --scorer-group-receipt "${scorer_group_b}" \
  --expected-scorer-group-receipt-sha256 "${scorer_group_b_sha256}" \
  --output "${output_dir}" \
  --max-schedule-steps "${schedule_steps}" \
  --learning-rate "${learning_rate}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  --ack-experimental-no-action-success-claim

[[ -f "${output_dir}/receipt.json" && -f "${output_dir}/adapter.safetensors" ]] || fail "completed run artifacts are absent"
echo "PAIR_V5_SAME_STATE_T2V_GUIDANCE_DP2SP4_STRONG_AUDIT_OK"
