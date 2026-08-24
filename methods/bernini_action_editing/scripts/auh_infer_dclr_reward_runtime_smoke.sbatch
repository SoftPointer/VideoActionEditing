#!/usr/bin/env bash
#SBATCH --job-name=bern-dclr-reward-sp4
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=02:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# One calibration cell is exactly one WORLD4/Ulysses4 process group.  This is
# a frozen raw-reward engineering smoke, not training and not a WORLD8 job.
# On an eight-MI210 host the other four devices are intentionally outside this
# four-GPU allocation; no DP replica or fictitious second reward group exists.

source_archive="${DCLR_REWARD_SOURCE_ARCHIVE:?set DCLR_REWARD_SOURCE_ARCHIVE}"
source_archive_sha256="${DCLR_REWARD_SOURCE_ARCHIVE_SHA256:?set DCLR_REWARD_SOURCE_ARCHIVE_SHA256}"
source_revision="${DCLR_REWARD_SOURCE_REVISION:?set DCLR_REWARD_SOURCE_REVISION}"
bernini_root="${DCLR_REWARD_BERNINI_ROOT:?set DCLR_REWARD_BERNINI_ROOT}"
veomni_root="${DCLR_REWARD_VEOMNI_ROOT:?set DCLR_REWARD_VEOMNI_ROOT}"
checkpoint="${DCLR_REWARD_CHECKPOINT:?set DCLR_REWARD_CHECKPOINT}"
checkpoint_manifest="${DCLR_REWARD_CHECKPOINT_CONTENT_MANIFEST:?set DCLR_REWARD_CHECKPOINT_CONTENT_MANIFEST}"
preprocessed_parquet_dir="${DCLR_REWARD_PREPROCESSED_PARQUET_DIR:?set DCLR_REWARD_PREPROCESSED_PARQUET_DIR}"
dataset_summary="${DCLR_REWARD_DATASET_SUMMARY:?set DCLR_REWARD_DATASET_SUMMARY}"
dataset_summary_sha256="${DCLR_REWARD_DATASET_SUMMARY_SHA256:?set DCLR_REWARD_DATASET_SUMMARY_SHA256}"
candidate_row_index="${DCLR_REWARD_CANDIDATE_ROW_INDEX:?set DCLR_REWARD_CANDIDATE_ROW_INDEX}"
candidate_iid="${DCLR_REWARD_EXPECTED_CANDIDATE_IID:?set DCLR_REWARD_EXPECTED_CANDIDATE_IID}"
proposal_source_iid="${DCLR_REWARD_PROPOSAL_SOURCE_IID:?set DCLR_REWARD_PROPOSAL_SOURCE_IID}"
wrong_source_row_index="${DCLR_REWARD_WRONG_SOURCE_ROW_INDEX:?set DCLR_REWARD_WRONG_SOURCE_ROW_INDEX}"
wrong_source_iid="${DCLR_REWARD_EXPECTED_WRONG_SOURCE_IID:?set DCLR_REWARD_EXPECTED_WRONG_SOURCE_IID}"
wrong_source_clean_latent="${DCLR_REWARD_WRONG_SOURCE_CLEAN_LATENT:?set DCLR_REWARD_WRONG_SOURCE_CLEAN_LATENT}"
wrong_source_clean_latent_sha256="${DCLR_REWARD_WRONG_SOURCE_CLEAN_LATENT_SHA256:?set DCLR_REWARD_WRONG_SOURCE_CLEAN_LATENT_SHA256}"
wrong_source_provenance_receipt="${DCLR_REWARD_WRONG_SOURCE_PROVENANCE_RECEIPT:?set DCLR_REWARD_WRONG_SOURCE_PROVENANCE_RECEIPT}"
wrong_source_provenance_receipt_sha256="${DCLR_REWARD_WRONG_SOURCE_PROVENANCE_RECEIPT_SHA256:?set DCLR_REWARD_WRONG_SOURCE_PROVENANCE_RECEIPT_SHA256}"
wrong_source_video_sha256="${DCLR_REWARD_WRONG_SOURCE_VIDEO_SHA256:?set DCLR_REWARD_WRONG_SOURCE_VIDEO_SHA256}"
wrong_source_match="${DCLR_REWARD_WRONG_SOURCE_MATCH_JSON:?set DCLR_REWARD_WRONG_SOURCE_MATCH_JSON}"
wrong_source_match_sha256="${DCLR_REWARD_WRONG_SOURCE_MATCH_SHA256:?set DCLR_REWARD_WRONG_SOURCE_MATCH_SHA256}"
action_instruction="${DCLR_REWARD_ACTION_INSTRUCTION:?set DCLR_REWARD_ACTION_INSTRUCTION}"
action_instruction_sha256="${DCLR_REWARD_ACTION_INSTRUCTION_SHA256:?set DCLR_REWARD_ACTION_INSTRUCTION_SHA256}"
hard_negative_instruction="${DCLR_REWARD_HARD_NEGATIVE_INSTRUCTION:?set DCLR_REWARD_HARD_NEGATIVE_INSTRUCTION}"
hard_negative_instruction_sha256="${DCLR_REWARD_HARD_NEGATIVE_INSTRUCTION_SHA256:?set DCLR_REWARD_HARD_NEGATIVE_INSTRUCTION_SHA256}"
output_dir="${DCLR_REWARD_OUTPUT_DIR:?set DCLR_REWARD_OUTPUT_DIR}"
python_bin="${DCLR_REWARD_PYTHON_BIN:?set DCLR_REWARD_PYTHON_BIN}"

pinned_bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
pinned_veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
pinned_checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
pinned_checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
checkpoint_tree_sha256="${DCLR_REWARD_CHECKPOINT_TREE_SHA256:-${pinned_checkpoint_tree_sha256}}"
checkpoint_manifest_sha256="${DCLR_REWARD_CHECKPOINT_CONTENT_MANIFEST_SHA256:-${pinned_checkpoint_manifest_sha256}}"
positive_control="${DCLR_REWARD_POSITIVE_CONTROL_PAIRED_TARGET:-0}"
allow_incomplete_dataset="${DCLR_REWARD_ALLOW_INCOMPLETE_DATASET:-0}"
noise_seed="${DCLR_REWARD_NOISE_SEED:-20260808}"
sigmas_text="${DCLR_REWARD_SIGMAS:-0.80 0.60 0.35 0.15}"
sigma_weights_text="${DCLR_REWARD_SIGMA_WEIGHTS:-1 1 1 1}"

fail() {
  echo "[bernini-dclr-reward-sp4] ERROR: $*" >&2
  exit 2
}

require_absolute() {
  local value="$1"
  local label="$2"
  [[ "${value}" == /* ]] || fail "${label} must be absolute"
}

require_plain_file() {
  local value="$1"
  local label="$2"
  [[ -f "${value}" && ! -L "${value}" ]] || fail "${label} must be a plain file"
}

require_sha256() {
  local value="$1"
  local label="$2"
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || fail "${label} must be a lowercase SHA-256"
}

for value_name in \
  source_archive bernini_root veomni_root checkpoint checkpoint_manifest \
  preprocessed_parquet_dir dataset_summary wrong_source_clean_latent \
  wrong_source_provenance_receipt wrong_source_match output_dir python_bin
do
  require_absolute "${!value_name}" "${value_name}"
done

require_sha256 "${source_archive_sha256}" "source archive SHA-256"
require_sha256 "${checkpoint_tree_sha256}" "checkpoint tree SHA-256"
require_sha256 "${checkpoint_manifest_sha256}" "checkpoint manifest SHA-256"
require_sha256 "${dataset_summary_sha256}" "dataset summary SHA-256"
require_sha256 "${wrong_source_clean_latent_sha256}" "wrong-source clean latent SHA-256"
require_sha256 "${wrong_source_provenance_receipt_sha256}" "wrong-source provenance receipt SHA-256"
require_sha256 "${wrong_source_video_sha256}" "wrong-source video SHA-256"
require_sha256 "${wrong_source_match_sha256}" "wrong-source match SHA-256"
require_sha256 "${action_instruction_sha256}" "action instruction SHA-256"
require_sha256 "${hard_negative_instruction_sha256}" "hard-negative instruction SHA-256"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be a full lowercase SHA-1"
[[ "${checkpoint_tree_sha256}" == "${pinned_checkpoint_tree_sha256}" ]] || fail "checkpoint tree differs from pinned Bernini-R release"
[[ "${checkpoint_manifest_sha256}" == "${pinned_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest identity differs from pinned release"
[[ "${candidate_row_index}" =~ ^[0-9]+$ ]] || fail "candidate row index must be nonnegative"
[[ "${wrong_source_row_index}" =~ ^[0-9]+$ ]] || fail "wrong-source row index must be nonnegative"
[[ "${candidate_row_index}" != "${wrong_source_row_index}" ]] || fail "candidate and wrong-source rows must differ"
[[ -n "${candidate_iid//[[:space:]]/}" && -n "${proposal_source_iid//[[:space:]]/}" && -n "${wrong_source_iid//[[:space:]]/}" ]] || fail "template/proposal/wrong-source IIDs must be nonempty"
[[ "${proposal_source_iid}" != "${wrong_source_iid}" ]] || fail "proposal and wrong-source IIDs must differ"
[[ "${positive_control}" == 0 || "${positive_control}" == 1 ]] || fail "positive-control opt-in must be exactly 0 or 1"
[[ "${allow_incomplete_dataset}" == 0 || "${allow_incomplete_dataset}" == 1 ]] || fail "incomplete-dataset opt-in must be exactly 0 or 1"
[[ "${noise_seed}" =~ ^[0-9]+$ && "${noise_seed}" -lt 2147483648 ]] || fail "noise seed must lie in [0,2^31)"

source_archive="$(realpath -e -- "${source_archive}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
preprocessed_parquet_dir="$(realpath -e -- "${preprocessed_parquet_dir}")"
dataset_summary="$(realpath -e -- "${dataset_summary}")"
wrong_source_clean_latent="$(realpath -e -- "${wrong_source_clean_latent}")"
wrong_source_provenance_receipt="$(realpath -e -- "${wrong_source_provenance_receipt}")"
wrong_source_match="$(realpath -e -- "${wrong_source_match}")"
python_bin="$(realpath -e -- "${python_bin}")"

require_plain_file "${source_archive}" "method source archive"
require_plain_file "${checkpoint_manifest}" "checkpoint content manifest"
require_plain_file "${dataset_summary}" "dataset summary"
require_plain_file "${wrong_source_clean_latent}" "wrong-source clean latent"
require_plain_file "${wrong_source_provenance_receipt}" "wrong-source provenance receipt"
require_plain_file "${wrong_source_match}" "wrong-source match manifest"
[[ "${wrong_source_clean_latent}" == *.safetensors ]] || fail "wrong-source clean latent must be safetensors"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini source root is invalid"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni source root is invalid"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint root is invalid"
[[ -d "${preprocessed_parquet_dir}" && ! -L "${preprocessed_parquet_dir}" ]] || fail "preprocessed parquet root is invalid"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "runtime Python is invalid"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method source archive hash differs"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint content manifest hash differs"
[[ "$(sha256sum "${dataset_summary}" | awk '{print $1}')" == "${dataset_summary_sha256}" ]] || fail "dataset summary hash differs"
[[ "$(sha256sum "${wrong_source_clean_latent}" | awk '{print $1}')" == "${wrong_source_clean_latent_sha256}" ]] || fail "wrong-source clean latent hash differs"
[[ "$(sha256sum "${wrong_source_provenance_receipt}" | awk '{print $1}')" == "${wrong_source_provenance_receipt_sha256}" ]] || fail "wrong-source provenance receipt hash differs"
[[ "$(sha256sum "${wrong_source_match}" | awk '{print $1}')" == "${wrong_source_match_sha256}" ]] || fail "wrong-source match hash differs"

actual_archive_revision="$(git get-tar-commit-id <"${source_archive}")" || fail "method source must be a git tar archive"
[[ "${actual_archive_revision}" == "${source_revision}" ]] || fail "method archive embedded revision differs"

actual_action_sha256="$("${python_bin}" -B - "${action_instruction}" <<'PY'
import hashlib
import sys

value = sys.argv[1].strip()
if not value or "\x00" in value:
    raise SystemExit("action instruction is empty")
print(hashlib.sha256(value.encode("utf-8")).hexdigest())
PY
)"
[[ "${actual_action_sha256}" == "${action_instruction_sha256}" ]] || fail "action instruction hash differs"
actual_hard_sha256="$("${python_bin}" -B - "${hard_negative_instruction}" <<'PY'
import hashlib
import sys

value = sys.argv[1].strip()
if not value or "\x00" in value:
    raise SystemExit("hard-negative instruction is empty")
print(hashlib.sha256(value.encode("utf-8")).hexdigest())
PY
)"
[[ "${actual_hard_sha256}" == "${hard_negative_instruction_sha256}" ]] || fail "hard-negative instruction hash differs"
[[ "${action_instruction_sha256}" != "${hard_negative_instruction_sha256}" ]] || fail "action and hard-negative hashes must differ"

candidate_clean_latent=""
candidate_clean_latent_sha256=""
correct_source_clean_latent=""
correct_source_clean_latent_sha256=""
candidate_arm=""
candidate_provenance_receipt=""
candidate_provenance_receipt_sha256=""
source_provenance_receipt=""
source_provenance_receipt_sha256=""
proposal_source_video_sha256=""
proposal_origin="native_rollout_predecode_latent"
if [[ "${positive_control}" == 0 ]]; then
  candidate_clean_latent="${DCLR_REWARD_CANDIDATE_CLEAN_LATENT:?native mode requires DCLR_REWARD_CANDIDATE_CLEAN_LATENT}"
  candidate_clean_latent_sha256="${DCLR_REWARD_CANDIDATE_CLEAN_LATENT_SHA256:?native mode requires DCLR_REWARD_CANDIDATE_CLEAN_LATENT_SHA256}"
  correct_source_clean_latent="${DCLR_REWARD_CORRECT_SOURCE_CLEAN_LATENT:?native mode requires DCLR_REWARD_CORRECT_SOURCE_CLEAN_LATENT}"
  correct_source_clean_latent_sha256="${DCLR_REWARD_CORRECT_SOURCE_CLEAN_LATENT_SHA256:?native mode requires DCLR_REWARD_CORRECT_SOURCE_CLEAN_LATENT_SHA256}"
  candidate_arm="${DCLR_REWARD_CANDIDATE_ARM:?native mode requires DCLR_REWARD_CANDIDATE_ARM}"
  candidate_provenance_receipt="${DCLR_REWARD_CANDIDATE_PROVENANCE_RECEIPT:?native mode requires DCLR_REWARD_CANDIDATE_PROVENANCE_RECEIPT}"
  candidate_provenance_receipt_sha256="${DCLR_REWARD_CANDIDATE_PROVENANCE_RECEIPT_SHA256:?native mode requires DCLR_REWARD_CANDIDATE_PROVENANCE_RECEIPT_SHA256}"
  source_provenance_receipt="${DCLR_REWARD_SOURCE_PROVENANCE_RECEIPT:?native mode requires DCLR_REWARD_SOURCE_PROVENANCE_RECEIPT}"
  source_provenance_receipt_sha256="${DCLR_REWARD_SOURCE_PROVENANCE_RECEIPT_SHA256:?native mode requires DCLR_REWARD_SOURCE_PROVENANCE_RECEIPT_SHA256}"
  proposal_source_video_sha256="${DCLR_REWARD_PROPOSAL_SOURCE_VIDEO_SHA256:?native mode requires DCLR_REWARD_PROPOSAL_SOURCE_VIDEO_SHA256}"
  require_absolute "${candidate_clean_latent}" "candidate clean latent"
  require_absolute "${correct_source_clean_latent}" "correct-source clean latent"
  require_absolute "${candidate_provenance_receipt}" "candidate provenance receipt"
  require_absolute "${source_provenance_receipt}" "source provenance receipt"
  require_sha256 "${candidate_clean_latent_sha256}" "candidate clean latent SHA-256"
  require_sha256 "${correct_source_clean_latent_sha256}" "correct-source clean latent SHA-256"
  require_sha256 "${candidate_provenance_receipt_sha256}" "candidate provenance receipt SHA-256"
  require_sha256 "${source_provenance_receipt_sha256}" "source provenance receipt SHA-256"
  require_sha256 "${proposal_source_video_sha256}" "proposal source video SHA-256"
  [[ "${candidate_arm}" == t2v || "${candidate_arm}" == r2v || "${candidate_arm}" == rv2v ]] || fail "candidate arm must be t2v, r2v, or rv2v"
  candidate_clean_latent="$(realpath -e -- "${candidate_clean_latent}")"
  correct_source_clean_latent="$(realpath -e -- "${correct_source_clean_latent}")"
  candidate_provenance_receipt="$(realpath -e -- "${candidate_provenance_receipt}")"
  source_provenance_receipt="$(realpath -e -- "${source_provenance_receipt}")"
  require_plain_file "${candidate_clean_latent}" "candidate clean latent"
  require_plain_file "${correct_source_clean_latent}" "correct-source clean latent"
  require_plain_file "${candidate_provenance_receipt}" "candidate provenance receipt"
  require_plain_file "${source_provenance_receipt}" "source provenance receipt"
  [[ "${candidate_clean_latent}" == *.safetensors ]] || fail "candidate clean latent must be safetensors"
  [[ "${correct_source_clean_latent}" == *.safetensors ]] || fail "correct-source clean latent must be safetensors"
  [[ "$(sha256sum "${candidate_clean_latent}" | awk '{print $1}')" == "${candidate_clean_latent_sha256}" ]] || fail "candidate clean latent hash differs"
  [[ "$(sha256sum "${correct_source_clean_latent}" | awk '{print $1}')" == "${correct_source_clean_latent_sha256}" ]] || fail "correct-source clean latent hash differs"
  [[ "$(sha256sum "${candidate_provenance_receipt}" | awk '{print $1}')" == "${candidate_provenance_receipt_sha256}" ]] || fail "candidate provenance receipt hash differs"
  [[ "$(sha256sum "${source_provenance_receipt}" | awk '{print $1}')" == "${source_provenance_receipt_sha256}" ]] || fail "source provenance receipt hash differs"
else
  proposal_origin="paired_target_positive_control"
  [[ "${proposal_source_iid}" == "${candidate_iid}" ]] || fail "positive-control proposal source IID must equal candidate row IID"
  for forbidden_name in \
    DCLR_REWARD_CANDIDATE_CLEAN_LATENT DCLR_REWARD_CANDIDATE_CLEAN_LATENT_SHA256 \
    DCLR_REWARD_CORRECT_SOURCE_CLEAN_LATENT DCLR_REWARD_CORRECT_SOURCE_CLEAN_LATENT_SHA256 \
    DCLR_REWARD_CANDIDATE_ARM DCLR_REWARD_CANDIDATE_PROVENANCE_RECEIPT \
    DCLR_REWARD_CANDIDATE_PROVENANCE_RECEIPT_SHA256 DCLR_REWARD_SOURCE_PROVENANCE_RECEIPT \
    DCLR_REWARD_SOURCE_PROVENANCE_RECEIPT_SHA256 DCLR_REWARD_PROPOSAL_SOURCE_VIDEO_SHA256
  do
    [[ -z "${!forbidden_name:-}" ]] || fail "positive control rejects ${forbidden_name}"
  done
fi

read -r -a sigma_values <<<"${sigmas_text}"
read -r -a sigma_weight_values <<<"${sigma_weights_text}"
"${python_bin}" -B - "${#sigma_values[@]}" "${sigma_values[@]}" "${#sigma_weight_values[@]}" "${sigma_weight_values[@]}" <<'PY'
import math
import struct
import sys

count = int(sys.argv[1])
sigmas = [float(value) for value in sys.argv[2 : 2 + count]]
weight_count_index = 2 + count
weight_count = int(sys.argv[weight_count_index])
weights = [float(value) for value in sys.argv[weight_count_index + 1 :]]
if len(sigmas) < 2 or len(weights) != weight_count or weight_count != len(sigmas):
    raise SystemExit("sigma/weight vectors require the same length >=2")
if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in sigmas):
    raise SystemExit("sigmas must be finite and lie in (0,1)")
bits = [struct.pack("!f", value) for value in sigmas]
if len(set(bits)) != len(bits):
    raise SystemExit("sigmas must remain distinct in FP32")
if any(not math.isfinite(value) or value < 0.0 for value in weights) or sum(weights) <= 0.0:
    raise SystemExit("sigma weights must be finite, nonnegative, and have positive mass")
PY

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
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "invalid Slurm job ID"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent is invalid"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-dclr-reward-sp4-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  local exit_status=$?
  local cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/bernini-dclr-reward-sp4-${SLURM_JOB_ID}."*) ;;
    *) echo "[bernini-dclr-reward-sp4] refusing unsafe scratch cleanup" >&2; cleanup_failed=1 ;;
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
    echo "[bernini-dclr-reward-sp4] PASS topology=WORLD4/Ulysses4 calibration_group=1 training=false proposal_origin=${proposal_origin} receipt=${output_dir}/receipt.json"
  else
    echo "[bernini-dclr-reward-sp4] EXIT status=${exit_status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${exit_status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p -- \
  "${task_scratch}/source-tree" \
  "${task_scratch}/cache/miopen-user" \
  "${task_scratch}/cache/miopen-custom" \
  "${task_scratch}/cache/torch-extensions" \
  "${task_scratch}/cache/triton" \
  "${task_scratch}/cache/xdg" \
  "${task_scratch}/cache/pycache" \
  "${task_scratch}/tmp"
archive_copy="${task_scratch}/method-source.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod 0444 -- "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged method archive hash differs"
[[ "$(git get-tar-commit-id <"${archive_copy}")" == "${source_revision}" ]] || fail "staged method archive revision differs"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

required = {
    "methods/bernini_action_editing/infer_dclr_reward_runtime_smoke.py",
    "methods/bernini_action_editing/dual_conditional_ratio_core.py",
    "methods/bernini_action_editing/dclr_runtime_contract.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/motion_residual.py",
    "methods/bernini_action_editing/source_kv_replay.py",
    "methods/bernini_action_editing/source_kv_route_batches.py",
}
seen = set()
with tarfile.open(sys.argv[1], "r:") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit("archive member escaped repository-relative closure")
        normalized = path.as_posix().lstrip("./")
        if not normalized or normalized in seen:
            raise SystemExit("archive contains an empty or duplicate member")
        seen.add(normalized)
        if member.issym() or member.islnk() or member.isfifo() or member.isdev():
            raise SystemExit("archive contains a link or device")
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"archive lacks raw DCLR reward closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
[[ -d "${method_root}" && ! -L "${method_root}" ]] || fail "extracted method root is invalid"
find "${method_root}" -type f -exec chmod a-w -- {} +
for required in \
  infer_dclr_reward_runtime_smoke.py dual_conditional_ratio_core.py \
  dclr_runtime_contract.py infer_source_kv_carrier_oracle.py infer_lora.py \
  train_lora.py motion_residual.py source_kv_replay.py source_kv_route_batches.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "missing staged ${required}"
done

allocated_rocr_devices="${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
[[ "${allocated_rocr_devices}" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || fail "WORLD4 calibration requires exactly four numeric ROCr devices"
"${python_bin}" -B - "${allocated_rocr_devices}" <<'PY'
import sys

devices = sys.argv[1].split(",")
if len(devices) != 4 or len(set(devices)) != 4:
    raise SystemExit("ROCr device allocation is not four distinct devices")
PY
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export ROCR_VISIBLE_DEVICES="${allocated_rocr_devices}"
export MIOPEN_USER_DB_PATH="${task_scratch}/cache/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/cache/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/cache/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/cache/triton"
export XDG_CACHE_HOME="${task_scratch}/cache/xdg"
export PYTHONPYCACHEPREFIX="${task_scratch}/cache/pycache"
export TMPDIR="${task_scratch}/tmp"
export TORCHELASTIC_ERROR_FILE="${task_scratch}/torch-elastic-error.json"

mkdir -- "${output_dir}"
receipt_path="${output_dir}/receipt.json"
torchrun_log="${output_dir}/torchrun.log"
master_port=$((31000 + SLURM_JOB_ID % 10000))

runner_args=(
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --checkpoint-content-manifest "${checkpoint_manifest}"
  --preprocessed-parquet-dir "${preprocessed_parquet_dir}"
  --dataset-summary "${dataset_summary}"
  --candidate-row-index "${candidate_row_index}"
  --wrong-source-row-index "${wrong_source_row_index}"
  --expected-candidate-iid "${candidate_iid}"
  --proposal-source-iid "${proposal_source_iid}"
  --expected-wrong-source-iid "${wrong_source_iid}"
  --wrong-source-clean-latent "${wrong_source_clean_latent}"
  --expected-wrong-source-clean-latent-sha256 "${wrong_source_clean_latent_sha256}"
  --wrong-source-provenance-receipt "${wrong_source_provenance_receipt}"
  --expected-wrong-source-provenance-receipt-sha256 "${wrong_source_provenance_receipt_sha256}"
  --expected-wrong-source-video-sha256 "${wrong_source_video_sha256}"
  --wrong-source-match-json "${wrong_source_match}"
  --expected-wrong-source-match-sha256 "${wrong_source_match_sha256}"
  --action-instruction "${action_instruction}"
  --expected-action-instruction-sha256 "${action_instruction_sha256}"
  --hard-negative-instruction "${hard_negative_instruction}"
  --expected-hard-negative-instruction-sha256 "${hard_negative_instruction_sha256}"
  --sigmas "${sigma_values[@]}"
  --sigma-weights "${sigma_weight_values[@]}"
  --noise-seed "${noise_seed}"
  --num-frames 81
  --expected-bernini-commit "${pinned_bernini_commit}"
  --expected-veomni-commit "${pinned_veomni_commit}"
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}"
  --method-source-revision "${source_revision}"
  --method-source-archive-sha256 "${source_archive_sha256}"
  --output-receipt "${receipt_path}"
)
if [[ "${positive_control}" == 0 ]]; then
  runner_args+=(
    --candidate-clean-latent "${candidate_clean_latent}"
    --expected-candidate-clean-latent-sha256 "${candidate_clean_latent_sha256}"
    --correct-source-clean-latent "${correct_source_clean_latent}"
    --expected-correct-source-clean-latent-sha256 "${correct_source_clean_latent_sha256}"
    --candidate-arm "${candidate_arm}"
    --candidate-provenance-receipt "${candidate_provenance_receipt}"
    --expected-candidate-provenance-receipt-sha256 "${candidate_provenance_receipt_sha256}"
    --source-provenance-receipt "${source_provenance_receipt}"
    --expected-source-provenance-receipt-sha256 "${source_provenance_receipt_sha256}"
    --expected-proposal-source-video-sha256 "${proposal_source_video_sha256}"
  )
else
  # Privileged paired target access is impossible unless this explicit env
  # opt-in selected the positive-control-only branch above.
  runner_args+=(--positive-control-paired-target)
fi
if [[ "${allow_incomplete_dataset}" == 1 ]]; then
  runner_args+=(--allow-incomplete-dataset)
fi

echo "[bernini-dclr-reward-sp4] topology=WORLD4/Ulysses4 calibration_group=1 allocated_gpus=4 training=false optimizer=false proposal_origin=${proposal_origin}"
"${python_bin}" -B -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port="${master_port}" \
  "${method_root}/infer_dclr_reward_runtime_smoke.py" \
  "${runner_args[@]}" >"${torchrun_log}" 2>&1 || {
    tail -n 200 -- "${torchrun_log}" >&2 || true
    fail "raw DCLR reward runtime failed"
  }

[[ -f "${receipt_path}" && ! -L "${receipt_path}" ]] || fail "raw reward receipt is missing"
printf -v sigmas_csv '%s,' "${sigma_values[@]}"
sigmas_csv="${sigmas_csv%,}"
printf -v sigma_weights_csv '%s,' "${sigma_weight_values[@]}"
sigma_weights_csv="${sigma_weights_csv%,}"

audit_receipt() {
  "${python_bin}" -B - \
    "${receipt_path}" \
    "${source_revision}" "${source_archive_sha256}" \
    "${pinned_bernini_commit}" "${pinned_veomni_commit}" \
    "${checkpoint_tree_sha256}" "${checkpoint_manifest}" "${checkpoint_manifest_sha256}" \
    "${dataset_summary}" "${dataset_summary_sha256}" \
    "${candidate_row_index}" "${candidate_iid}" "${proposal_source_iid}" \
    "${wrong_source_row_index}" "${wrong_source_iid}" \
    "${wrong_source_clean_latent}" "${wrong_source_clean_latent_sha256}" \
    "${wrong_source_provenance_receipt}" "${wrong_source_provenance_receipt_sha256}" \
    "${wrong_source_video_sha256}" \
    "${wrong_source_match}" "${wrong_source_match_sha256}" \
    "${action_instruction_sha256}" "${hard_negative_instruction_sha256}" \
    "${proposal_origin}" \
    "${candidate_clean_latent:--}" "${candidate_clean_latent_sha256:--}" \
    "${correct_source_clean_latent:--}" "${correct_source_clean_latent_sha256:--}" \
    "${candidate_arm:--}" \
    "${candidate_provenance_receipt:--}" "${candidate_provenance_receipt_sha256:--}" \
    "${source_provenance_receipt:--}" "${source_provenance_receipt_sha256:--}" \
    "${proposal_source_video_sha256:--}" \
    "${sigmas_csv}" "${sigma_weights_csv}" "${noise_seed}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

(
    receipt_value,
    source_revision,
    source_archive_sha256,
    bernini_commit,
    veomni_commit,
    checkpoint_tree_sha256,
    checkpoint_manifest_value,
    checkpoint_manifest_sha256,
    dataset_summary_value,
    dataset_summary_sha256,
    candidate_row_index,
    candidate_iid,
    proposal_source_iid,
    wrong_row_index,
    wrong_iid,
    wrong_latent_value,
    wrong_latent_sha256,
    wrong_provenance_value,
    wrong_provenance_sha256,
    wrong_source_video_sha256,
    wrong_match_value,
    wrong_match_sha256,
    action_sha256,
    hard_sha256,
    proposal_origin,
    candidate_latent_value,
    candidate_latent_sha256,
    source_latent_value,
    source_latent_sha256,
    candidate_arm,
    candidate_provenance_value,
    candidate_provenance_sha256,
    source_provenance_value,
    source_provenance_sha256,
    proposal_source_video_sha256,
    sigmas_value,
    weights_value,
    noise_seed,
) = sys.argv[1:]

sha_re = re.compile(r"[0-9a-f]{64}")

def require(condition, message):
    if not condition:
        raise SystemExit(message)

def plain_file(value):
    path = Path(value).resolve(strict=True)
    require(path.is_file() and not path.is_symlink(), f"not a plain file: {path}")
    return path

def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def canonical_sha(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()

def f32(value):
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]

def finite_number(value):
    return type(value) in (int, float) and not isinstance(value, bool) and math.isfinite(value)

def tensor_identity(value, *, ndim=None):
    require(isinstance(value, dict), "tensor identity is absent")
    require(value.get("finite") is True, "tensor identity is not finite")
    shape = value.get("shape")
    require(isinstance(shape, list) and shape and all(type(x) is int and x > 0 for x in shape), "tensor shape differs")
    if ndim is not None:
        require(len(shape) == ndim, "tensor rank differs")
    require(type(value.get("numel")) is int and math.prod(shape) == value["numel"], "tensor numel differs")
    require(type(value.get("byte_count")) is int and value["byte_count"] > 0, "tensor byte count differs")
    require(sha_re.fullmatch(str(value.get("content_sha256"))) is not None, "tensor content digest differs")
    require(sha_re.fullmatch(str(value.get("raw_storage_sha256"))) is not None, "tensor storage digest differs")
    return value

def tensor_payload_identity(value):
    """Compare one tensor independent of the receipt-local diagnostic label.

    ``content_sha256`` intentionally includes ``label``.  The artifact loader
    names a source tensor by its artifact role, while the four-branch runtime
    names the same bytes by branch role (for example ``correct_source``).
    Binding those two views therefore uses exact storage bytes plus all
    physical tensor metadata, and audits each label/content digest separately.
    """
    checked = tensor_identity(value, ndim=5)
    return {
        key: checked.get(key)
        for key in (
            "shape",
            "dtype",
            "numel",
            "byte_count",
            "raw_storage_sha256",
            "finite",
        )
    }

receipt_path = plain_file(receipt_value)
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
require(isinstance(receipt, dict), "receipt is not an object")
unsigned = dict(receipt)
declared_receipt_digest = unsigned.pop("receipt_digest", None)
require(declared_receipt_digest == canonical_sha(unsigned), "receipt self digest differs")
require(
    receipt.get("schema_version") == "bernini-frozen-dclr-runtime-smoke-receipt-v1"
    and receipt.get("method") == "bernini-frozen-dclr-runtime-smoke-v1",
    "raw reward receipt schema differs",
)
require(
    receipt.get("engineering_smoke_only") is True
    and receipt.get("training_pair_authorized") is False
    and receipt.get("scientific_claim_authorized") is False
    and receipt.get("production_claim_forbidden") is True
    and receipt.get("denoising_error_proxy_not_likelihood") is True,
    "engineering-only authorization state differs",
)
paired = proposal_origin == "paired_target_positive_control"
require(
    receipt.get("proposal_origin") == proposal_origin
    and receipt.get("paired_target_positive_control") is paired
    and receipt.get("wrong_source_paired_target_accessed") is False,
    "top-level proposal-origin state differs",
)

local = receipt.get("local_evidence")
require(isinstance(local, dict), "local evidence is absent")
local_digest = canonical_sha(local)
distributed = receipt.get("distributed")
require(
    isinstance(distributed, dict)
    and distributed.get("world_size") == 4
    and distributed.get("ulysses_size") == 4
    and distributed.get("upstream_shared_step_returns_full_sequence_per_rank") is True
    and distributed.get("reward_reduction") == "none"
    and distributed.get("all_gather_evidence_only") is True,
    "WORLD4/Ulysses4 evidence contract differs",
)
rank_records = distributed.get("rank_records")
require(isinstance(rank_records, list) and len(rank_records) == 4, "rank evidence count differs")
require([row.get("rank") for row in rank_records] == [0, 1, 2, 3], "rank ordering differs")
for row in rank_records:
    require(
        row.get("world_size") == 4
        and row.get("ulysses_size") == 4
        and row.get("local_evidence_digest") == local_digest,
        "SP4 ranks did not report byte-identical canonical evidence",
    )

require(
    local.get("method_source_revision") == source_revision
    and local.get("method_source_archive_sha256") == source_archive_sha256
    and local.get("bernini_commit") == bernini_commit
    and local.get("veomni_commit") == veomni_commit
    and local.get("checkpoint_tree_sha256") == checkpoint_tree_sha256,
    "source/model provenance differs",
)
checkpoint_identity = local.get("checkpoint_content_identity")
require(
    isinstance(checkpoint_identity, dict)
    and checkpoint_identity.get("manifest_path") == str(Path(checkpoint_manifest_value).resolve())
    and checkpoint_identity.get("manifest_sha256_computed") == checkpoint_manifest_sha256
    and checkpoint_identity.get("manifest_sha256_expected") == checkpoint_manifest_sha256
    and checkpoint_identity.get("verified_file_count") == 23
    and checkpoint_identity.get("every_file_sha256_verified") is True
    and sha_re.fullmatch(str(checkpoint_identity.get("verified_entries_digest"))) is not None,
    "checkpoint content-manifest audit differs",
)
require(
    local.get("dataset_summary_sha256") == dataset_summary_sha256
    and sha_re.fullmatch(str(local.get("dataset_signature"))) is not None,
    "dataset binding differs",
)

candidate = local.get("candidate")
require(
    isinstance(candidate, dict)
    and candidate.get("row_index") == int(candidate_row_index)
    and candidate.get("message_template_iid") == candidate_iid
    and candidate.get("message_template_columns_loaded") == [
        "iid", "inputs", "source_video_sha256"
    ]
    and candidate.get("proposal_source_iid") == proposal_source_iid
    and candidate.get("action_instruction_sha256") == action_sha256
    and candidate.get("proposal_origin") == proposal_origin
    and candidate.get("paired_target_accessed") is paired
    and candidate.get("positive_control_only") is paired,
    "candidate receipt binding differs",
)
wrong = local.get("wrong_source")
match = wrong.get("match_manifest") if isinstance(wrong, dict) else None
wrong_artifact = wrong.get("source_artifact") if isinstance(wrong, dict) else None
wrong_provenance = wrong.get("source_provenance") if isinstance(wrong, dict) else None
criteria = match.get("criteria") if isinstance(match, dict) else None
expected_criteria = {
    "distinct_identity",
    "same_actor_class",
    "same_actor_count",
    "same_spatial_bucket",
    "same_camera_class",
    "same_composition_class",
    "same_length",
    "manual_reviewed",
}
require(
    isinstance(wrong, dict)
    and wrong.get("row_index") == int(wrong_row_index)
    and wrong.get("iid") == wrong_iid
    and wrong.get("source_video_sha256") == wrong_source_video_sha256
    and wrong.get("message_template_columns_loaded") == [
        "iid", "inputs", "source_video_sha256"
    ]
    and wrong.get("paired_target_accessed") is False
    and isinstance(wrong_artifact, dict)
    and isinstance(wrong_provenance, dict)
    and isinstance(match, dict)
    and match.get("candidate_iid") == proposal_source_iid
    and match.get("wrong_source_iid") == wrong_iid
    and sha_re.fullmatch(str(match.get("candidate_source_video_sha256"))) is not None
    and match.get("wrong_source_video_sha256") == wrong_source_video_sha256
    and match.get("candidate_source_video_sha256") != match.get("wrong_source_video_sha256")
    and match.get("path") == str(Path(wrong_match_value).resolve())
    and match.get("file_sha256") == wrong_match_sha256
    and isinstance(criteria, dict)
    and set(criteria) == expected_criteria
    and all(type(criteria.get(key)) is bool for key in expected_criteria)
    and all(criteria.get(key) is True for key in (
        "distinct_identity",
        "same_actor_class",
        "same_actor_count",
        "same_spatial_bucket",
        "same_length",
        "manual_reviewed",
    ))
    and match.get("declared_use") in ("runtime_plumbing_only", "reward_calibration")
    and match.get("scientific_eligibility") is all(criteria.values())
    and match.get("source_reward_calibration_authorized") is (
        match.get("declared_use") == "reward_calibration" and all(criteria.values())
    )
    and isinstance(match.get("reviewer"), str)
    and bool(match["reviewer"].strip()),
    "wrong-source manual match evidence differs",
)
if not paired:
    require(
        match.get("candidate_source_video_sha256") == proposal_source_video_sha256,
        "native proposal source video differs from wrong-source manifest",
    )
text = local.get("text_conditions")
require(
    isinstance(text, dict)
    and text.get("hard_negative_instruction_sha256") == hard_sha256
    and all(sha_re.fullmatch(str(text.get(key))) is not None for key in (
        "t2v_action_prompt_sha256",
        "t2v_hard_negative_prompt_sha256",
        "mv2v_action_prompt_sha256",
    ))
    and text.get("t2v_action_prompt_sha256") != text.get("t2v_hard_negative_prompt_sha256"),
    "action/hard-negative text evidence differs",
)

student_identity = tensor_identity(local.get("shared_student_clean"), ndim=5)
source_identity = tensor_identity(local.get("correct_source"), ndim=5)
wrong_identity = tensor_identity(local.get("wrong_source_tensor"), ndim=5)
epsilon_identity = tensor_identity(local.get("shared_epsilon"), ndim=5)
for identity in (student_identity, source_identity, wrong_identity, epsilon_identity):
    require(identity["shape"][:3] == [1, 16, 21] and identity.get("dtype") == "torch.float32", "clean/noise latent contract differs")
require(
    student_identity["shape"] == source_identity["shape"] == wrong_identity["shape"] == epsilon_identity["shape"],
    "proposal/source/wrong/noise geometry differs",
)
require(source_identity["raw_storage_sha256"] != wrong_identity["raw_storage_sha256"], "wrong source is tensor-identical to correct source")

def audit_artifact(artifact, expected_path, expected_sha, role, identity):
    require(isinstance(artifact, dict), f"{role} artifact is absent")
    expected_source = (
        "native_sampler_before_vae_decode"
        if role == "native_sampler_proposal"
        else "source_video_vae_encode_before_any_decode"
    )
    artifact_identity = tensor_identity(artifact.get("tensor_identity"), ndim=5)
    branch_identity = tensor_identity(identity, ndim=5)
    require(
        artifact.get("path") == str(Path(expected_path).resolve())
        and artifact.get("file_sha256") == expected_sha
        and artifact.get("tensor_key") == "normalized_clean_latent"
        and artifact.get("mp4_decode_reencode_used") is False
        and artifact.get("metadata") == {
            "coordinate": "bernini_normalized_clean_vae_latent",
            "frame_contract": "exact81_latent21",
            "artifact_role": role,
            "source": expected_source,
        }
        and artifact_identity.get("label") == role
        and sha_re.fullmatch(str(artifact_identity.get("content_sha256"))) is not None
        and sha_re.fullmatch(str(branch_identity.get("content_sha256"))) is not None
        and tensor_payload_identity(artifact_identity) == tensor_payload_identity(branch_identity),
        f"{role} artifact binding differs",
    )

audit_artifact(
    wrong_artifact,
    wrong_latent_value,
    wrong_latent_sha256,
    "source_video_condition",
    wrong_identity,
)
wrong_source_receipt = (
    wrong_provenance.get("source_receipt")
    if isinstance(wrong_provenance, dict)
    else None
)
require(
    isinstance(wrong_provenance, dict)
    and wrong_provenance.get("source_iid") == wrong_iid
    and wrong_provenance.get("source_video_sha256") == wrong_source_video_sha256
    and wrong_provenance.get("checkpoint_tree_sha256") == checkpoint_tree_sha256
    and wrong_provenance.get("checkpoint_content_identity") == checkpoint_identity
    and wrong_provenance.get("source_condition_latent_sha256") == wrong_latent_sha256
    and wrong_provenance.get("source_columns_accessed") == [
        "iid", "source_video", "source_video_sha256"
    ]
    and wrong_provenance.get("target_columns_accessed") == []
    and wrong_provenance.get("target_media_accessed") is False
    and wrong_provenance.get("paired_target_accessed") is False
    and wrong_provenance.get("mp4_decode_reencode_used") is False
    and isinstance(wrong_source_receipt, dict)
    and wrong_source_receipt.get("path") == str(Path(wrong_provenance_value).resolve())
    and wrong_source_receipt.get("file_sha256") == wrong_provenance_sha256
    and wrong_source_receipt.get("schema_version") == "bernini-source-only-vae-materialization-v1"
    and sha_re.fullmatch(str(wrong_source_receipt.get("receipt_digest"))) is not None,
    "source-only wrong-source VAE provenance closure differs",
)

if paired:
    require(candidate.get("proposal_artifact") is None and candidate.get("native_provenance") is None and local.get("correct_source_artifact") is None, "positive control claimed native artifacts")
    require(candidate_latent_value == source_latent_value == candidate_arm == candidate_provenance_value == source_provenance_value == proposal_source_video_sha256 == "-", "positive-control launcher unexpectedly supplied native provenance")
else:
    audit_artifact(candidate.get("proposal_artifact"), candidate_latent_value, candidate_latent_sha256, "native_sampler_proposal", student_identity)
    audit_artifact(local.get("correct_source_artifact"), source_latent_value, source_latent_sha256, "source_video_condition", source_identity)
    provenance = candidate.get("native_provenance")
    candidate_receipt = provenance.get("candidate_receipt") if isinstance(provenance, dict) else None
    source_receipt = provenance.get("source_receipt") if isinstance(provenance, dict) else None
    require(
        isinstance(provenance, dict)
        and provenance.get("candidate_arm") == candidate_arm
        and provenance.get("source_video_sha256") == proposal_source_video_sha256
        and provenance.get("action_prompt_sha256") == action_sha256
        and provenance.get("checkpoint_tree_sha256") == checkpoint_tree_sha256
        and provenance.get("checkpoint_content_identity") == checkpoint_identity
        and provenance.get("proposal_latent_sha256") == candidate_latent_sha256
        and provenance.get("source_condition_latent_sha256") == source_latent_sha256
        and provenance.get("paired_target_accessed") is False
        and provenance.get("mp4_decode_reencode_used") is False
        and isinstance(candidate_receipt, dict)
        and candidate_receipt.get("path") == str(Path(candidate_provenance_value).resolve())
        and candidate_receipt.get("file_sha256") == candidate_provenance_sha256
        and sha_re.fullmatch(str(candidate_receipt.get("receipt_digest"))) is not None
        and isinstance(source_receipt, dict)
        and source_receipt.get("path") == str(Path(source_provenance_value).resolve())
        and source_receipt.get("file_sha256") == source_provenance_sha256
        and sha_re.fullmatch(str(source_receipt.get("receipt_digest"))) is not None,
        "native sampler/source provenance closure differs",
    )

require(
    local.get("num_frames") == 81
    and local.get("latent_phases") == 21
    and local.get("patch_size") == [1, 2, 2]
    and local.get("noise_seed") == int(noise_seed)
    and local.get("forward_implementation") == "renderer.get_t5_text_embeddings+renderer.diff_dec.shared_step"
    and local.get("adapter_state") == "absent_frozen_base"
    and local.get("model_mode") == "eval_inference_mode"
    and local.get("raw_positive_conditional_only") is True
    and local.get("cfg") is False
    and local.get("apg") is False
    and local.get("renderer_forward_scalar_loss") is False
    and local.get("reward_reduction") == "none",
    "frozen raw-positive reward contract differs",
)
geometry = local.get("geometry")
target_tokens = geometry.get("target_tokens") if isinstance(geometry, dict) else None
require(
    isinstance(geometry, dict)
    and geometry.get("verified") is True
    and type(target_tokens) is int
    and target_tokens > 0
    and target_tokens % 21 == 0
    and geometry.get("t2v_total_tokens") == target_tokens
    and geometry.get("mv2v_total_tokens") == 2 * target_tokens
    and geometry.get("t2v_is_correct_mv2v_target_tail_view") is True
    and geometry.get("wrong_source_changes_prefix_only") is True
    and geometry.get("correct_wrong_full_rotary_exact_equal") is True
    and geometry.get("target_source_id") == 0
    and geometry.get("source_source_id") == 1
    and geometry.get("rotary_dtype") == "torch.complex128",
    "shared-state query geometry differs",
)

sigmas = [float(value) for value in sigmas_value.split(",")]
weights = [float(value) for value in weights_value.split(",")]
require(local.get("sigma_weights") == weights, "sigma weights differ")
records = local.get("sigma_records")
branch_order = [
    "t2v_target_action",
    "t2v_hard_negative",
    "mv2v_correct_source",
    "mv2v_matched_wrong_source",
]
require(local.get("branch_order") == branch_order, "branch order differs")
require(isinstance(records, list) and len(records) == len(sigmas), "sigma record count differs")
require(local.get("forwards_per_rank") == 4 * len(sigmas), "forward count differs")
for requested_sigma, record in zip(sigmas, records):
    point = record.get("flow_query") if isinstance(record, dict) else None
    sigma32 = f32(requested_sigma)
    timestep32 = f32(sigma32 * 1000.0)
    require(
        isinstance(point, dict)
        and point.get("sigma") == sigma32
        and point.get("timestep") == timestep32
        and point.get("sigma_float32_bits_hex") == struct.pack("!f", sigma32).hex()
        and point.get("timestep_float32_bits_hex") == struct.pack("!f", timestep32).hex()
        and record.get("mode_shared_sigma_and_timestep") is True
        and record.get("t2v_shift_sampling_density_only") == 3.0
        and record.get("mv2v_shift_sampling_density_only") == 5.0
        and record.get("raw_positive_conditional_only") is True
        and record.get("cfg") is False
        and record.get("apg") is False,
        "explicit shared sigma/timestep record differs",
    )
    errors = record.get("errors")
    require(
        isinstance(errors, dict)
        and list(errors) == branch_order
        and all(finite_number(errors[key]) and errors[key] >= 0.0 for key in branch_order),
        "raw branch energy record differs",
    )
    tensor_identity(record.get("target_query"), ndim=5)

normalized_weights = [value / sum(weights) for value in weights]
def audit_diagnostics(value, preferred_key, contrast_key):
    require(isinstance(value, dict), "multi-sigma diagnostics are absent")
    preferred = [record["errors"][preferred_key] for record in records]
    contrast = [record["errors"][contrast_key] for record in records]
    require(value.get("preferred_error_by_sigma") == [[item] for item in preferred], "preferred raw errors differ")
    require(value.get("contrast_error_by_sigma") == [[item] for item in contrast], "contrast raw errors differ")
    returned_weights = value.get("normalized_sigma_weights")
    require(isinstance(returned_weights, list) and len(returned_weights) == len(weights), "normalized sigma weights differ")
    for got, expected in zip(returned_weights, normalized_weights):
        require(isinstance(got, (int, float)) and math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-12), "normalized sigma weight differs")
    preferred_aggregate = sum(weight * item for weight, item in zip(normalized_weights, preferred))
    contrast_aggregate = sum(weight * item for weight, item in zip(normalized_weights, contrast))
    proxy = math.log((contrast_aggregate + 1.0e-8) / (preferred_aggregate + 1.0e-8))
    require(len(value.get("preferred_error", [])) == len(value.get("contrast_error", [])) == len(value.get("proxy", [])) == 1, "aggregate diagnostic shape differs")
    require(math.isclose(value["preferred_error"][0], preferred_aggregate, rel_tol=1e-10, abs_tol=1e-10), "preferred aggregate differs")
    require(math.isclose(value["contrast_error"][0], contrast_aggregate, rel_tol=1e-10, abs_tol=1e-10), "contrast aggregate differs")
    require(math.isclose(value["proxy"][0], proxy, rel_tol=1e-10, abs_tol=1e-10), "log-ratio proxy differs")
    require(value.get("sign_convention") == "positive means preferred condition has lower denoising error", "proxy sign convention differs")

audit_diagnostics(local.get("action_target_vs_hard_negative"), branch_order[0], branch_order[1])
audit_diagnostics(local.get("source_correct_vs_matched_wrong"), branch_order[2], branch_order[3])

expected_source_reward_authorized = bool(
    match["source_reward_calibration_authorized"] and not paired
)
require(
    receipt.get("source_reward_calibration_authorized") is expected_source_reward_authorized,
    "top-level source-reward calibration authorization differs",
)
if match["declared_use"] == "runtime_plumbing_only":
    require(receipt.get("source_reward_calibration_authorized") is False, "runtime-only donor was promoted to calibration evidence")
if paired:
    require(receipt.get("source_reward_calibration_authorized") is False, "positive-control paired target authorized source calibration")

require(file_sha(plain_file(checkpoint_manifest_value)) == checkpoint_manifest_sha256, "checkpoint manifest changed after runtime")
require(file_sha(plain_file(dataset_summary_value)) == dataset_summary_sha256, "dataset summary changed after runtime")
require(file_sha(plain_file(wrong_latent_value)) == wrong_latent_sha256, "wrong-source latent changed after runtime")
require(file_sha(plain_file(wrong_provenance_value)) == wrong_provenance_sha256, "wrong-source provenance receipt changed after runtime")
require(file_sha(plain_file(wrong_match_value)) == wrong_match_sha256, "wrong-source match changed after runtime")
if not paired:
    require(file_sha(plain_file(candidate_latent_value)) == candidate_latent_sha256, "candidate latent changed after runtime")
    require(file_sha(plain_file(source_latent_value)) == source_latent_sha256, "source latent changed after runtime")
    require(file_sha(plain_file(candidate_provenance_value)) == candidate_provenance_sha256, "candidate provenance receipt changed after runtime")
    require(file_sha(plain_file(source_provenance_value)) == source_provenance_sha256, "source provenance receipt changed after runtime")
print("BERNINI_DCLR_RAW_REWARD_SP4_STRONG_AUDIT_OK")
PY
}

receipt_sha_round1="$(sha256sum "${receipt_path}" | awk '{print $1}')"
audit_receipt || fail "receipt audit round 1 failed"
[[ "$(sha256sum "${receipt_path}" | awk '{print $1}')" == "${receipt_sha_round1}" ]] || fail "receipt changed during audit round 1"
receipt_sha_round2="$(sha256sum "${receipt_path}" | awk '{print $1}')"
[[ "${receipt_sha_round2}" == "${receipt_sha_round1}" ]] || fail "receipt hash closure differs"
audit_receipt || fail "receipt audit round 2 failed"
[[ "$(sha256sum "${receipt_path}" | awk '{print $1}')" == "${receipt_sha_round2}" ]] || fail "receipt changed during audit round 2"

[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "durable method archive changed"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged method archive changed"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest changed"
[[ "$(sha256sum "${dataset_summary}" | awk '{print $1}')" == "${dataset_summary_sha256}" ]] || fail "dataset summary changed"
[[ "$(sha256sum "${wrong_source_clean_latent}" | awk '{print $1}')" == "${wrong_source_clean_latent_sha256}" ]] || fail "wrong-source latent changed"
[[ "$(sha256sum "${wrong_source_provenance_receipt}" | awk '{print $1}')" == "${wrong_source_provenance_receipt_sha256}" ]] || fail "wrong-source provenance receipt changed"
[[ "$(sha256sum "${wrong_source_match}" | awk '{print $1}')" == "${wrong_source_match_sha256}" ]] || fail "wrong-source match changed"
if [[ "${positive_control}" == 0 ]]; then
  [[ "$(sha256sum "${candidate_clean_latent}" | awk '{print $1}')" == "${candidate_clean_latent_sha256}" ]] || fail "candidate latent changed"
  [[ "$(sha256sum "${correct_source_clean_latent}" | awk '{print $1}')" == "${correct_source_clean_latent_sha256}" ]] || fail "source latent changed"
  [[ "$(sha256sum "${candidate_provenance_receipt}" | awk '{print $1}')" == "${candidate_provenance_receipt_sha256}" ]] || fail "candidate provenance receipt changed"
  [[ "$(sha256sum "${source_provenance_receipt}" | awk '{print $1}')" == "${source_provenance_receipt_sha256}" ]] || fail "source provenance receipt changed"
fi
chmod 0444 -- "${receipt_path}" "${torchrun_log}"
echo "[bernini-dclr-reward-sp4] COMPLETE receipt_sha256=${receipt_sha_round2} audit=BERNINI_DCLR_RAW_REWARD_SP4_STRONG_AUDIT_OK"
