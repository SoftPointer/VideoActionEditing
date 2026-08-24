#!/usr/bin/env bash
#SBATCH --job-name=bern-iar-dp2sp4
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

# One allocation runs two isolated WORLD4/Ulysses4 process groups (DP2 x SP4),
# one per noise seed. Bernini-R 1.3B has twelve attention heads, so Ulysses-8
# is invalid. This launcher is an exact-81 engineering smoke only: S=4, K=2
# hard negatives, M=1 source-only wrong donor and L=3 bridge fractions, with
# no optimizer, backward, adapter, or checkpoint.

source_archive="${IAR_RUNTIME_SOURCE_ARCHIVE:?set IAR_RUNTIME_SOURCE_ARCHIVE}"
source_archive_sha256="${IAR_RUNTIME_SOURCE_ARCHIVE_SHA256:?set IAR_RUNTIME_SOURCE_ARCHIVE_SHA256}"
source_revision="${IAR_RUNTIME_SOURCE_REVISION:?set IAR_RUNTIME_SOURCE_REVISION}"
bernini_root="${IAR_RUNTIME_BERNINI_ROOT:?set IAR_RUNTIME_BERNINI_ROOT}"
veomni_root="${IAR_RUNTIME_VEOMNI_ROOT:?set IAR_RUNTIME_VEOMNI_ROOT}"
checkpoint="${IAR_RUNTIME_CHECKPOINT:?set IAR_RUNTIME_CHECKPOINT}"
checkpoint_manifest="${IAR_RUNTIME_CHECKPOINT_CONTENT_MANIFEST:?set IAR_RUNTIME_CHECKPOINT_CONTENT_MANIFEST}"
preprocessed_parquet_dir="${IAR_RUNTIME_PREPROCESSED_PARQUET_DIR:?set IAR_RUNTIME_PREPROCESSED_PARQUET_DIR}"
dataset_summary="${IAR_RUNTIME_DATASET_SUMMARY:?set IAR_RUNTIME_DATASET_SUMMARY}"
dataset_summary_sha256="${IAR_RUNTIME_DATASET_SUMMARY_SHA256:?set IAR_RUNTIME_DATASET_SUMMARY_SHA256}"
proposal_source_iid="${IAR_RUNTIME_PROPOSAL_SOURCE_IID:?set IAR_RUNTIME_PROPOSAL_SOURCE_IID}"
candidate_clean_latent="${IAR_RUNTIME_CANDIDATE_CLEAN_LATENT:?set IAR_RUNTIME_CANDIDATE_CLEAN_LATENT}"
candidate_clean_latent_sha256="${IAR_RUNTIME_CANDIDATE_CLEAN_LATENT_SHA256:?set IAR_RUNTIME_CANDIDATE_CLEAN_LATENT_SHA256}"
correct_source_clean_latent="${IAR_RUNTIME_CORRECT_SOURCE_CLEAN_LATENT:?set IAR_RUNTIME_CORRECT_SOURCE_CLEAN_LATENT}"
correct_source_clean_latent_sha256="${IAR_RUNTIME_CORRECT_SOURCE_CLEAN_LATENT_SHA256:?set IAR_RUNTIME_CORRECT_SOURCE_CLEAN_LATENT_SHA256}"
candidate_arm="${IAR_RUNTIME_CANDIDATE_ARM:?set IAR_RUNTIME_CANDIDATE_ARM}"
candidate_provenance_receipt="${IAR_RUNTIME_CANDIDATE_PROVENANCE_RECEIPT:?set IAR_RUNTIME_CANDIDATE_PROVENANCE_RECEIPT}"
candidate_provenance_receipt_sha256="${IAR_RUNTIME_CANDIDATE_PROVENANCE_RECEIPT_SHA256:?set IAR_RUNTIME_CANDIDATE_PROVENANCE_RECEIPT_SHA256}"
source_provenance_receipt="${IAR_RUNTIME_SOURCE_PROVENANCE_RECEIPT:?set IAR_RUNTIME_SOURCE_PROVENANCE_RECEIPT}"
source_provenance_receipt_sha256="${IAR_RUNTIME_SOURCE_PROVENANCE_RECEIPT_SHA256:?set IAR_RUNTIME_SOURCE_PROVENANCE_RECEIPT_SHA256}"
proposal_source_video_sha256="${IAR_RUNTIME_PROPOSAL_SOURCE_VIDEO_SHA256:?set IAR_RUNTIME_PROPOSAL_SOURCE_VIDEO_SHA256}"
wrong_source_row_index="${IAR_RUNTIME_WRONG_SOURCE_ROW_INDEX:?set IAR_RUNTIME_WRONG_SOURCE_ROW_INDEX}"
wrong_source_iid="${IAR_RUNTIME_EXPECTED_WRONG_SOURCE_IID:?set IAR_RUNTIME_EXPECTED_WRONG_SOURCE_IID}"
wrong_source_clean_latent="${IAR_RUNTIME_WRONG_SOURCE_CLEAN_LATENT:?set IAR_RUNTIME_WRONG_SOURCE_CLEAN_LATENT}"
wrong_source_clean_latent_sha256="${IAR_RUNTIME_WRONG_SOURCE_CLEAN_LATENT_SHA256:?set IAR_RUNTIME_WRONG_SOURCE_CLEAN_LATENT_SHA256}"
wrong_source_provenance_receipt="${IAR_RUNTIME_WRONG_SOURCE_PROVENANCE_RECEIPT:?set IAR_RUNTIME_WRONG_SOURCE_PROVENANCE_RECEIPT}"
wrong_source_provenance_receipt_sha256="${IAR_RUNTIME_WRONG_SOURCE_PROVENANCE_RECEIPT_SHA256:?set IAR_RUNTIME_WRONG_SOURCE_PROVENANCE_RECEIPT_SHA256}"
wrong_source_video_sha256="${IAR_RUNTIME_WRONG_SOURCE_VIDEO_SHA256:?set IAR_RUNTIME_WRONG_SOURCE_VIDEO_SHA256}"
wrong_source_match="${IAR_RUNTIME_WRONG_SOURCE_MATCH_JSON:?set IAR_RUNTIME_WRONG_SOURCE_MATCH_JSON}"
wrong_source_match_sha256="${IAR_RUNTIME_WRONG_SOURCE_MATCH_SHA256:?set IAR_RUNTIME_WRONG_SOURCE_MATCH_SHA256}"
action_instruction="${IAR_RUNTIME_ACTION_INSTRUCTION:?set IAR_RUNTIME_ACTION_INSTRUCTION}"
action_instruction_sha256="${IAR_RUNTIME_ACTION_INSTRUCTION_SHA256:?set IAR_RUNTIME_ACTION_INSTRUCTION_SHA256}"
noop_instruction="${IAR_RUNTIME_NOOP_INSTRUCTION:?set IAR_RUNTIME_NOOP_INSTRUCTION}"
noop_instruction_sha256="${IAR_RUNTIME_NOOP_INSTRUCTION_SHA256:?set IAR_RUNTIME_NOOP_INSTRUCTION_SHA256}"
hard_negative_manifest="${IAR_RUNTIME_HARD_NEGATIVE_MANIFEST:?set IAR_RUNTIME_HARD_NEGATIVE_MANIFEST}"
hard_negative_manifest_sha256="${IAR_RUNTIME_HARD_NEGATIVE_MANIFEST_SHA256:?set IAR_RUNTIME_HARD_NEGATIVE_MANIFEST_SHA256}"
output_dir="${IAR_RUNTIME_OUTPUT_DIR:?set IAR_RUNTIME_OUTPUT_DIR}"
python_bin="${IAR_RUNTIME_PYTHON_BIN:?set IAR_RUNTIME_PYTHON_BIN}"

pinned_bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
pinned_veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
pinned_checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
pinned_checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
checkpoint_tree_sha256="${IAR_RUNTIME_CHECKPOINT_TREE_SHA256:-${pinned_checkpoint_tree_sha256}}"
checkpoint_manifest_sha256="${IAR_RUNTIME_CHECKPOINT_CONTENT_MANIFEST_SHA256:-${pinned_checkpoint_manifest_sha256}}"
allow_incomplete_dataset="${IAR_RUNTIME_ALLOW_INCOMPLETE_DATASET:-0}"
noise_seed_a="${IAR_RUNTIME_NOISE_SEED_A:-20260808}"
noise_seed_b="${IAR_RUNTIME_NOISE_SEED_B:-20260809}"
sigmas_text="${IAR_RUNTIME_SIGMAS:-0.80 0.60 0.35 0.15}"
bridge_fractions_text="${IAR_RUNTIME_BRIDGE_FRACTIONS:-1 0.5 0}"

fail() {
  echo "[bernini-iar-field-sp4] ERROR: $*" >&2
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
  preprocessed_parquet_dir dataset_summary candidate_clean_latent \
  correct_source_clean_latent candidate_provenance_receipt \
  source_provenance_receipt wrong_source_clean_latent \
  wrong_source_provenance_receipt wrong_source_match hard_negative_manifest \
  output_dir python_bin
do
  require_absolute "${!value_name}" "${value_name}"
done

for digest_name in \
  source_archive_sha256 checkpoint_tree_sha256 checkpoint_manifest_sha256 \
  dataset_summary_sha256 candidate_clean_latent_sha256 \
  correct_source_clean_latent_sha256 candidate_provenance_receipt_sha256 \
  source_provenance_receipt_sha256 proposal_source_video_sha256 \
  wrong_source_clean_latent_sha256 wrong_source_provenance_receipt_sha256 \
  wrong_source_video_sha256 wrong_source_match_sha256 \
  action_instruction_sha256 noop_instruction_sha256 \
  hard_negative_manifest_sha256
do
  require_sha256 "${!digest_name}" "${digest_name}"
done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be a full lowercase SHA-1"
[[ "${checkpoint_tree_sha256}" == "${pinned_checkpoint_tree_sha256}" ]] || fail "checkpoint tree differs from pinned Bernini-R release"
[[ "${checkpoint_manifest_sha256}" == "${pinned_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest differs from pinned release"
[[ "${wrong_source_row_index}" =~ ^[0-9]+$ ]] || fail "wrong-source row index must be nonnegative"
[[ -n "${proposal_source_iid//[[:space:]]/}" && -n "${wrong_source_iid//[[:space:]]/}" ]] || fail "proposal/wrong-source IIDs must be nonempty"
[[ "${proposal_source_iid}" != "${wrong_source_iid}" ]] || fail "proposal and wrong-source IIDs must differ"
[[ "${candidate_arm}" == t2v || "${candidate_arm}" == r2v || "${candidate_arm}" == rv2v ]] || fail "candidate arm must be t2v, r2v, or rv2v"
[[ "${allow_incomplete_dataset}" == 0 || "${allow_incomplete_dataset}" == 1 ]] || fail "incomplete-dataset opt-in must be exactly 0 or 1"
[[ "${noise_seed_a}" =~ ^[0-9]+$ && "${noise_seed_a}" -lt 2147483648 ]] || fail "noise seed A must lie in [0,2^31)"
[[ "${noise_seed_b}" =~ ^[0-9]+$ && "${noise_seed_b}" -lt 2147483648 ]] || fail "noise seed B must lie in [0,2^31)"
[[ "${noise_seed_a}" != "${noise_seed_b}" ]] || fail "dual WORLD4 noise seeds must differ"
[[ -n "${action_instruction//[[:space:]]/}" ]] || fail "action instruction is invalid"
[[ -n "${noop_instruction//[[:space:]]/}" ]] || fail "no-op instruction is invalid"

source_archive="$(realpath -e -- "${source_archive}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
preprocessed_parquet_dir="$(realpath -e -- "${preprocessed_parquet_dir}")"
dataset_summary="$(realpath -e -- "${dataset_summary}")"
candidate_clean_latent="$(realpath -e -- "${candidate_clean_latent}")"
correct_source_clean_latent="$(realpath -e -- "${correct_source_clean_latent}")"
candidate_provenance_receipt="$(realpath -e -- "${candidate_provenance_receipt}")"
source_provenance_receipt="$(realpath -e -- "${source_provenance_receipt}")"
wrong_source_clean_latent="$(realpath -e -- "${wrong_source_clean_latent}")"
wrong_source_provenance_receipt="$(realpath -e -- "${wrong_source_provenance_receipt}")"
wrong_source_match="$(realpath -e -- "${wrong_source_match}")"
hard_negative_manifest="$(realpath -e -- "${hard_negative_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
launcher_source="$(realpath -e -- "${BASH_SOURCE[0]}")"
require_plain_file "${launcher_source}" "executing launcher"
launcher_source_sha256="$(sha256sum "${launcher_source}" | awk '{print $1}')"
require_sha256 "${launcher_source_sha256}" "executing launcher SHA-256"

for pair in \
  "${source_archive}|method source archive" \
  "${checkpoint_manifest}|checkpoint content manifest" \
  "${dataset_summary}|dataset summary" \
  "${candidate_clean_latent}|candidate clean latent" \
  "${correct_source_clean_latent}|correct-source clean latent" \
  "${candidate_provenance_receipt}|candidate provenance receipt" \
  "${source_provenance_receipt}|source provenance receipt" \
  "${wrong_source_clean_latent}|wrong-source clean latent" \
  "${wrong_source_provenance_receipt}|wrong-source provenance receipt" \
  "${wrong_source_match}|wrong-source match manifest" \
  "${hard_negative_manifest}|hard-negative manifest"
do
  require_plain_file "${pair%%|*}" "${pair#*|}"
done
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini source root is invalid"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni source root is invalid"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint root is invalid"
[[ -d "${preprocessed_parquet_dir}" && ! -L "${preprocessed_parquet_dir}" ]] || fail "preprocessed parquet root is invalid"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "runtime Python is invalid"
[[ "${candidate_clean_latent}" == *.safetensors && "${correct_source_clean_latent}" == *.safetensors && "${wrong_source_clean_latent}" == *.safetensors ]] || fail "all latent artifacts must be safetensors"

declare -a immutable_files=(
  "${launcher_source}|${launcher_source_sha256}|executing launcher"
  "${source_archive}|${source_archive_sha256}|method source archive"
  "${checkpoint_manifest}|${checkpoint_manifest_sha256}|checkpoint content manifest"
  "${dataset_summary}|${dataset_summary_sha256}|dataset summary"
  "${candidate_clean_latent}|${candidate_clean_latent_sha256}|candidate latent"
  "${correct_source_clean_latent}|${correct_source_clean_latent_sha256}|correct-source latent"
  "${candidate_provenance_receipt}|${candidate_provenance_receipt_sha256}|candidate provenance receipt"
  "${source_provenance_receipt}|${source_provenance_receipt_sha256}|source provenance receipt"
  "${wrong_source_clean_latent}|${wrong_source_clean_latent_sha256}|wrong-source latent"
  "${wrong_source_provenance_receipt}|${wrong_source_provenance_receipt_sha256}|wrong-source provenance receipt"
  "${wrong_source_match}|${wrong_source_match_sha256}|wrong-source match manifest"
  "${hard_negative_manifest}|${hard_negative_manifest_sha256}|hard-negative manifest"
)
for entry in "${immutable_files[@]}"; do
  path="${entry%%|*}"
  remainder="${entry#*|}"
  expected="${remainder%%|*}"
  label="${remainder#*|}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || fail "${label} hash differs"
done

actual_archive_revision="$(git get-tar-commit-id <"${source_archive}")" || fail "method source must be a git tar archive"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method source archive hash differs"
[[ "${actual_archive_revision}" == "${source_revision}" ]] || fail "method archive embedded revision differs"
actual_action_sha256="$("${python_bin}" -B -c 'import hashlib,sys; value=sys.argv[1]; assert value and value == value.strip() and "\0" not in value; print(hashlib.sha256(value.encode("utf-8")).hexdigest())' "${action_instruction}")"
[[ "${actual_action_sha256}" == "${action_instruction_sha256}" ]] || fail "action instruction hash differs"
actual_noop_sha256="$("${python_bin}" -B -c 'import hashlib,sys; value=sys.argv[1]; assert value and value == value.strip() and "\0" not in value; print(hashlib.sha256(value.encode("utf-8")).hexdigest())' "${noop_instruction}")"
[[ "${actual_noop_sha256}" == "${noop_instruction_sha256}" ]] || fail "no-op instruction hash differs"
[[ "${action_instruction_sha256}" != "${noop_instruction_sha256}" ]] || fail "action and no-op instructions must differ"

read -r -a sigma_values <<<"${sigmas_text}"
"${python_bin}" -B - "${#sigma_values[@]}" "${sigma_values[@]}" <<'PY'
import math
import struct
import sys

count = int(sys.argv[1])
values = [float(item) for item in sys.argv[2:]]
expected = [0.80, 0.60, 0.35, 0.15]
if count != 4 or len(values) != 4:
    raise SystemExit("IAR frozen field smoke requires exactly four sigmas")
if any(not math.isfinite(value) for value in values):
    raise SystemExit("sigma contains NaN or infinity")
if [struct.pack("!f", value) for value in values] != [
    struct.pack("!f", value) for value in expected
]:
    raise SystemExit("sigmas must be exact descending 0.80,0.60,0.35,0.15 in FP32")
PY

read -r -a bridge_fraction_values <<<"${bridge_fractions_text}"
"${python_bin}" -B - "${#bridge_fraction_values[@]}" "${bridge_fraction_values[@]}" <<'PY'
import math
import struct
import sys

count = int(sys.argv[1])
values = [float(item) for item in sys.argv[2:]]
expected = [1.0, 0.5, 0.0]
if count != 3 or len(values) != 3:
    raise SystemExit("IAR bridge smoke requires exactly L=3 fractions")
if any(not math.isfinite(value) for value in values):
    raise SystemExit("bridge fraction contains NaN or infinity")
if [struct.pack("!f", value) for value in values] != [
    struct.pack("!f", value) for value in expected
]:
    raise SystemExit("bridge fractions must be exact 1,0.5,0 in FP32")
PY

"${python_bin}" -B - \
  "${hard_negative_manifest}" "${action_instruction_sha256}" \
  "${noop_instruction_sha256}" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

path = Path(sys.argv[1]).resolve(strict=True)
action_sha = sys.argv[2]
noop_sha = sys.argv[3]
value = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(value, dict):
    raise SystemExit("hard-negative manifest must be an object")
if set(value) != {
    "schema_version", "action_instruction_sha256", "hard_negatives",
    "energy_semantics", "declared_use",
}:
    raise SystemExit("hard-negative manifest fields differ")
if value.get("schema_version") != "bernini-iar-hard-negative-manifest-v1":
    raise SystemExit("hard-negative manifest schema differs")
if value.get("action_instruction_sha256") != action_sha:
    raise SystemExit("hard-negative manifest action binding differs")
if value.get("declared_use") != "engineering_plumbing_only":
    raise SystemExit("hard-negative manifest must remain engineering_plumbing_only")
if value.get("energy_semantics") != "runtime_fp32_mse_to_epsilon_minus_bridge_clean_lower_is_harder":
    raise SystemExit("hard-negative energy semantics differ")
rows = value.get("hard_negatives")
if not isinstance(rows, list) or len(rows) != 2:
    raise SystemExit("hard-negative manifest requires exactly K=2")
sha_re = re.compile(r"[0-9a-f]{64}")
seen_ids = set()
seen_instructions = set()
seen_hashes = set()
for row in rows:
    if not isinstance(row, dict) or set(row) != {
        "condition_id", "instruction", "instruction_sha256"
    }:
        raise SystemExit("hard-negative row fields differ")
    condition_id = row["condition_id"]
    instruction = row["instruction"]
    digest = row["instruction_sha256"]
    if not isinstance(condition_id, str) or not condition_id.strip() or condition_id != condition_id.strip() or "\0" in condition_id:
        raise SystemExit("hard-negative condition id is invalid")
    if not isinstance(instruction, str) or not instruction.strip() or instruction != instruction.strip() or "\0" in instruction:
        raise SystemExit("hard-negative instruction is invalid")
    if sha_re.fullmatch(str(digest)) is None or digest != hashlib.sha256(instruction.strip().encode("utf-8")).hexdigest():
        raise SystemExit("hard-negative instruction hash differs")
    if digest == action_sha:
        raise SystemExit("hard-negative instruction equals target action")
    seen_ids.add(condition_id)
    seen_instructions.add(instruction.strip())
    seen_hashes.add(digest)
if len(seen_ids) != 2 or len(seen_instructions) != 2 or len(seen_hashes) != 2:
    raise SystemExit("hard-negative rows must be pairwise content-distinct")
if noop_sha not in seen_hashes:
    raise SystemExit("no-op instruction must occur exactly once among K=2 negatives")
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
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "invalid Slurm job ID"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent is invalid"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-iar-dp2sp4-${SLURM_JOB_ID}.XXXXXX")"
group_a_pid=""
group_b_pid=""

cleanup() {
  local exit_status=$?
  local cleanup_failed=0
  trap - EXIT TERM INT
  for child_pid in "${group_a_pid:-}" "${group_b_pid:-}"; do
    if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
      kill -TERM "${child_pid}" 2>/dev/null || true
    fi
  done
  case "${task_scratch:-}" in
    "${scratch_parent%/}/bernini-iar-dp2sp4-${SLURM_JOB_ID}."*) ;;
    *) echo "[bernini-iar-dp2sp4] refusing unsafe scratch cleanup" >&2; cleanup_failed=1 ;;
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
    echo "[bernini-iar-dp2sp4] PASS topology=DP2xWORLD4/Ulysses4 exact81=true seeds=${noise_seed_a},${noise_seed_b} S=4 K=2 M=1 L=3 branches=7 forwards_per_rank_per_group=84 training=false calibration=false cross_seed_numeric_stability_authorized=false receipts=${output_dir}/seed-a/receipt.json,${output_dir}/seed-b/receipt.json"
  else
    echo "[bernini-iar-dp2sp4] EXIT status=${exit_status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${exit_status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p -- \
  "${task_scratch}/source-tree" \
  "${task_scratch}/group-a/cache/miopen-user" \
  "${task_scratch}/group-a/cache/miopen-custom" \
  "${task_scratch}/group-a/cache/torch-extensions" \
  "${task_scratch}/group-a/cache/triton" \
  "${task_scratch}/group-a/cache/xdg" \
  "${task_scratch}/group-a/cache/pycache" \
  "${task_scratch}/group-a/tmp" \
  "${task_scratch}/group-b/cache/miopen-user" \
  "${task_scratch}/group-b/cache/miopen-custom" \
  "${task_scratch}/group-b/cache/torch-extensions" \
  "${task_scratch}/group-b/cache/triton" \
  "${task_scratch}/group-b/cache/xdg" \
  "${task_scratch}/group-b/cache/pycache" \
  "${task_scratch}/group-b/tmp"
archive_copy="${task_scratch}/method-source.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod 0444 -- "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged method archive hash differs"
[[ "$(git get-tar-commit-id <"${archive_copy}")" == "${source_revision}" ]] || fail "staged archive revision differs"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

required = {
    "methods/bernini_action_editing/infer_iar_official_runtime_smoke.py",
    "methods/bernini_action_editing/identity_anchored_action_residual.py",
    "methods/bernini_action_editing/infer_dclr_reward_runtime_smoke.py",
    "methods/bernini_action_editing/dual_conditional_ratio_core.py",
    "methods/bernini_action_editing/dclr_runtime_contract.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/motion_residual.py",
    "methods/bernini_action_editing/source_kv_replay.py",
    "methods/bernini_action_editing/source_kv_route_batches.py",
    "methods/bernini_action_editing/scripts/auh_infer_iar_official_runtime_smoke.sbatch",
}
forbidden = {
    "methods/bernini_action_editing/infer_iar_field_feasibility.py",
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
            raise SystemExit("archive contains a link, FIFO, or device")
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"archive lacks IAR official runtime closure: {missing}")
unexpected = sorted(forbidden & seen)
if unexpected:
    raise SystemExit(f"archive retains forbidden replaceable callback scaffold: {unexpected}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
[[ -d "${method_root}" && ! -L "${method_root}" ]] || fail "extracted method root is invalid"
find "${method_root}" -type f -exec chmod a-w -- {} +
for required in \
  infer_iar_official_runtime_smoke.py identity_anchored_action_residual.py \
  infer_dclr_reward_runtime_smoke.py \
  dual_conditional_ratio_core.py dclr_runtime_contract.py \
  infer_source_kv_carrier_oracle.py infer_lora.py train_lora.py \
  motion_residual.py source_kv_replay.py source_kv_route_batches.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "missing staged ${required}"
done
archived_launcher="${method_root}/scripts/auh_infer_iar_official_runtime_smoke.sbatch"
[[ -f "${archived_launcher}" && ! -L "${archived_launcher}" ]] || fail "archived launcher is missing"
archived_launcher_sha256="$(sha256sum "${archived_launcher}" | awk '{print $1}')"
[[ "${archived_launcher_sha256}" == "${launcher_source_sha256}" ]] || fail "executing launcher differs from git-archive launcher"

allocated_rocr_devices="${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
[[ "${allocated_rocr_devices}" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || fail "DP2 x WORLD4 IAR smoke requires exactly eight numeric ROCr devices"
"${python_bin}" -B - "${allocated_rocr_devices}" <<'PY'
import sys

devices = sys.argv[1].split(",")
if len(devices) != 8 or len(set(devices)) != 8:
    raise SystemExit("ROCr allocation is not eight distinct devices")
PY
IFS=',' read -r -a allocated_devices <<<"${allocated_rocr_devices}"
printf -v group_a_devices '%s,' "${allocated_devices[@]:0:4}"
group_a_devices="${group_a_devices%,}"
printf -v group_b_devices '%s,' "${allocated_devices[@]:4:4}"
group_b_devices="${group_b_devices%,}"
[[ "${group_a_devices}" != "${group_b_devices}" ]] || fail "WORLD4 device groups overlap completely"

mkdir -- "${output_dir}"
mkdir -- "${output_dir}/seed-a" "${output_dir}/seed-b"
receipt_path_a="${output_dir}/seed-a/receipt.json"
receipt_path_b="${output_dir}/seed-b/receipt.json"
torchrun_log_a="${output_dir}/seed-a/torchrun.log"
torchrun_log_b="${output_dir}/seed-b/torchrun.log"
master_port_a=$((22000 + SLURM_JOB_ID % 10000))
master_port_b=$((32000 + SLURM_JOB_ID % 10000))

runner_args=(
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --checkpoint-content-manifest "${checkpoint_manifest}"
  --preprocessed-parquet-dir "${preprocessed_parquet_dir}"
  --dataset-summary "${dataset_summary}"
  --wrong-source-row-index "${wrong_source_row_index}"
  --proposal-source-iid "${proposal_source_iid}"
  --expected-wrong-source-iid "${wrong_source_iid}"
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
  --wrong-source-clean-latent "${wrong_source_clean_latent}"
  --expected-wrong-source-clean-latent-sha256 "${wrong_source_clean_latent_sha256}"
  --wrong-source-provenance-receipt "${wrong_source_provenance_receipt}"
  --expected-wrong-source-provenance-receipt-sha256 "${wrong_source_provenance_receipt_sha256}"
  --expected-wrong-source-video-sha256 "${wrong_source_video_sha256}"
  --wrong-source-match-json "${wrong_source_match}"
  --expected-wrong-source-match-sha256 "${wrong_source_match_sha256}"
  --action-instruction "${action_instruction}"
  --expected-action-instruction-sha256 "${action_instruction_sha256}"
  --noop-instruction "${noop_instruction}"
  --expected-noop-instruction-sha256 "${noop_instruction_sha256}"
  --hard-negative-manifest "${hard_negative_manifest}"
  --expected-hard-negative-manifest-sha256 "${hard_negative_manifest_sha256}"
  --sigmas "${sigma_values[@]}"
  --bridge-fractions "${bridge_fraction_values[@]}"
  --num-frames 81
  --expected-bernini-commit "${pinned_bernini_commit}"
  --expected-veomni-commit "${pinned_veomni_commit}"
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}"
  --method-source-revision "${source_revision}"
  --method-source-archive-sha256 "${source_archive_sha256}"
  --launcher-source-sha256 "${launcher_source_sha256}"
)
if [[ "${allow_incomplete_dataset}" == 1 ]]; then
  runner_args+=(--allow-incomplete-dataset)
fi

launch_group() {
  local group_label="$1"
  local visible_devices="$2"
  local master_port="$3"
  local group_seed="$4"
  local receipt_path="$5"
  local torchrun_log="$6"
  local scratch_root="$7"
  (
    # ROCr selects physical devices, then remaps each group to local 0..3.
    # A second HIP/CUDA physical-index filter would hide seed-B's devices.
    unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
    export ROCR_VISIBLE_DEVICES="${visible_devices}"
    export MIOPEN_USER_DB_PATH="${scratch_root}/cache/miopen-user"
    export MIOPEN_CUSTOM_CACHE_DIR="${scratch_root}/cache/miopen-custom"
    export TORCH_EXTENSIONS_DIR="${scratch_root}/cache/torch-extensions"
    export TRITON_CACHE_DIR="${scratch_root}/cache/triton"
    export XDG_CACHE_HOME="${scratch_root}/cache/xdg"
    export PYTHONPYCACHEPREFIX="${scratch_root}/cache/pycache"
    export TMPDIR="${scratch_root}/tmp"
    export TORCHELASTIC_ERROR_FILE="${scratch_root}/torch-elastic-error.json"
    echo "[bernini-iar-dp2sp4] group=${group_label} topology=WORLD4/Ulysses4 allocated_gpus=4 seed=${group_seed} exact81=true S=4 K=2 M=1 L=3 branches=7 forwards_per_rank=84 training=false optimizer=false backward=false checkpoint_save=false donor_plumbing_only=true source_calibration=false action_invariance_calibration=false"
    "${python_bin}" -B -m torch.distributed.run \
      --nproc_per_node=4 \
      --master_addr=127.0.0.1 \
      --master_port="${master_port}" \
      "${method_root}/infer_iar_official_runtime_smoke.py" \
      "${runner_args[@]}" \
      --noise-seed "${group_seed}" \
      --output-receipt "${receipt_path}" >"${torchrun_log}" 2>&1
  )
}

echo "[bernini-iar-dp2sp4] topology=DP2xWORLD4/Ulysses4 allocated_gpus=8 groups=2 seeds=${noise_seed_a},${noise_seed_b} exact81=true S=4 K=2 M=1 L=3 branches=7 forwards_per_rank_per_group=84 training=false optimizer=false backward=false checkpoint_save=false cross_seed_numeric_stability_authorized=false"
launch_group seed-a "${group_a_devices}" "${master_port_a}" "${noise_seed_a}" "${receipt_path_a}" "${torchrun_log_a}" "${task_scratch}/group-a" &
group_a_pid=$!
launch_group seed-b "${group_b_devices}" "${master_port_b}" "${noise_seed_b}" "${receipt_path_b}" "${torchrun_log_b}" "${task_scratch}/group-b" &
group_b_pid=$!

if wait "${group_a_pid}"; then
  group_a_status=0
else
  group_a_status=$?
  kill -TERM "${group_b_pid}" 2>/dev/null || true
fi
group_a_pid=""
if wait "${group_b_pid}"; then
  group_b_status=0
else
  group_b_status=$?
fi
group_b_pid=""
if [[ "${group_a_status}" != 0 || "${group_b_status}" != 0 ]]; then
  echo "[bernini-iar-dp2sp4] group failure status_a=${group_a_status} status_b=${group_b_status}" >&2
  tail -n 240 -- "${torchrun_log_a}" >&2 || true
  tail -n 240 -- "${torchrun_log_b}" >&2 || true
  fail "one or both IAR official frozen-field WORLD4 runtimes failed"
fi

[[ -f "${receipt_path_a}" && ! -L "${receipt_path_a}" ]] || fail "seed-A IAR runtime receipt is missing"
[[ -f "${receipt_path_b}" && ! -L "${receipt_path_b}" ]] || fail "seed-B IAR runtime receipt is missing"
printf -v sigmas_csv '%s,' "${sigma_values[@]}"
sigmas_csv="${sigmas_csv%,}"
printf -v bridge_fractions_csv '%s,' "${bridge_fraction_values[@]}"
bridge_fractions_csv="${bridge_fractions_csv%,}"

audit_receipt() {
  local receipt_value="$1"
  local expected_seed="$2"
  "${python_bin}" -B - \
    "${receipt_value}" \
    "${source_revision}" "${source_archive_sha256}" "${launcher_source_sha256}" \
    "${pinned_bernini_commit}" "${pinned_veomni_commit}" \
    "${checkpoint}" "${checkpoint_tree_sha256}" "${checkpoint_manifest}" "${checkpoint_manifest_sha256}" \
    "${dataset_summary}" "${dataset_summary_sha256}" \
    "${proposal_source_iid}" \
    "${candidate_clean_latent}" "${candidate_clean_latent_sha256}" \
    "${correct_source_clean_latent}" "${correct_source_clean_latent_sha256}" \
    "${candidate_arm}" \
    "${candidate_provenance_receipt}" "${candidate_provenance_receipt_sha256}" \
    "${source_provenance_receipt}" "${source_provenance_receipt_sha256}" \
    "${proposal_source_video_sha256}" \
    "${wrong_source_row_index}" "${wrong_source_iid}" \
    "${wrong_source_clean_latent}" "${wrong_source_clean_latent_sha256}" \
    "${wrong_source_provenance_receipt}" "${wrong_source_provenance_receipt_sha256}" \
    "${wrong_source_video_sha256}" \
    "${wrong_source_match}" "${wrong_source_match_sha256}" \
    "${action_instruction}" "${action_instruction_sha256}" \
    "${noop_instruction}" "${noop_instruction_sha256}" \
    "${hard_negative_manifest}" "${hard_negative_manifest_sha256}" \
    "${sigmas_csv}" "${bridge_fractions_csv}" "${expected_seed}" <<'PY'
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
    launcher_source_sha256,
    bernini_commit,
    veomni_commit,
    checkpoint_value,
    checkpoint_tree_sha256,
    checkpoint_manifest_value,
    checkpoint_manifest_sha256,
    dataset_summary_value,
    dataset_summary_sha256,
    proposal_source_iid,
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
    wrong_row_index,
    wrong_iid,
    wrong_latent_value,
    wrong_latent_sha256,
    wrong_provenance_value,
    wrong_provenance_sha256,
    wrong_source_video_sha256,
    wrong_match_value,
    wrong_match_sha256,
    action_instruction,
    action_sha256,
    noop_instruction,
    noop_sha256,
    negative_manifest_value,
    negative_manifest_sha256,
    sigmas_value,
    bridge_fractions_value,
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


def canonical_sample_identity(instruction, instruction_sha256):
    require(
        isinstance(instruction, str)
        and instruction
        and instruction == instruction.strip()
        and hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        == instruction_sha256,
        "canonical sample instruction binding differs",
    )
    messages = [
        {"type": "video", "has_loss": 0},
        {"type": "text", "text": instruction, "has_loss": 0},
        {"type": "video_gen", "has_loss": 1},
    ]
    inputs = json.dumps(
        messages,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return {
        "instruction_sha256": instruction_sha256,
        "canonical_inputs_sha256": hashlib.sha256(inputs).hexdigest(),
    }


def finite_number(value):
    return (
        type(value) in (int, float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def finite_tree(value):
    if finite_number(value):
        return True
    if isinstance(value, list):
        return bool(value) and all(finite_tree(item) for item in value)
    return False


def f32(value):
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def tensor_identity(value, *, ndim=None):
    require(isinstance(value, dict), "tensor identity is absent")
    require(value.get("finite") is True, "tensor identity is not finite")
    shape = value.get("shape")
    require(
        isinstance(shape, list)
        and shape
        and all(type(item) is int and item > 0 for item in shape),
        "tensor shape differs",
    )
    if ndim is not None:
        require(len(shape) == ndim, "tensor rank differs")
    require(
        type(value.get("numel")) is int
        and math.prod(shape) == value["numel"],
        "tensor numel differs",
    )
    require(
        type(value.get("byte_count")) is int and value["byte_count"] > 0,
        "tensor byte count differs",
    )
    require(
        sha_re.fullmatch(str(value.get("content_sha256"))) is not None
        and sha_re.fullmatch(str(value.get("raw_storage_sha256"))) is not None,
        "tensor digest differs",
    )
    return value


def tensor_payload_identity(value):
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
require(
    set(receipt)
    == {
        "schema_version",
        "method",
        "local_evidence",
        "distributed",
        "engineering_smoke_only",
        "donor_plumbing_only",
        "source_reward_calibration_authorized",
        "source_action_invariance_calibration_authorized",
        "training_authorized",
        "training_pair_authorized",
        "scientific_claim_authorized",
        "production_claim_forbidden",
        "paired_target_accessed",
        "forward_callback_present",
        "custom_core_present",
        "receipt_digest",
    },
    "receipt field set differs",
)
unsigned = dict(receipt)
claimed_digest = unsigned.pop("receipt_digest", None)
require(claimed_digest == canonical_sha(unsigned), "receipt self digest differs")
require(
    receipt.get("schema_version")
    == "bernini-iar-official-runtime-smoke-receipt-v1"
    and receipt.get("method") == "bernini-iar-official-runtime-smoke-v1",
    "IAR official runtime receipt schema differs",
)
require(
    receipt.get("engineering_smoke_only") is True
    and receipt.get("scientific_claim_authorized") is False
    and receipt.get("production_claim_forbidden") is True
    and receipt.get("donor_plumbing_only") is True
    and receipt.get("source_reward_calibration_authorized") is False
    and receipt.get("training_authorized") is False
    and receipt.get("training_pair_authorized") is False
    and receipt.get("source_action_invariance_calibration_authorized") is False
    and receipt.get("paired_target_accessed") is False
    and receipt.get("forward_callback_present") is False
    and receipt.get("custom_core_present") is False,
    "engineering-only authorization state differs",
)

local = receipt.get("local_evidence")
require(isinstance(local, dict), "local evidence is absent")
require(
    set(local)
    == {
        "method",
        "method_source_revision",
        "method_source_archive_sha256",
        "launcher_source_sha256",
        "bernini_commit",
        "veomni_commit",
        "checkpoint_path",
        "checkpoint_tree_sha256",
        "checkpoint_content_identity",
        "dataset_signature",
        "dataset_summary_sha256",
        "candidate",
        "wrong_sources",
        "text_conditions",
        "hard_negative_manifest",
        "homotopy",
        "bridge_fractions",
        "sigmas",
        "num_frames",
        "latent_phases",
        "patch_size",
        "geometry",
        "hard_negative_count",
        "wrong_source_count",
        "branch_order",
        "cell_records",
        "continuity",
        "iar_core",
        "forwards_per_rank",
        "forward_implementation",
        "active_model_id",
        "adapter_state",
        "model_mode",
        "raw_positive_conditional_only",
        "cfg",
        "apg",
        "backend",
        "donor_plumbing_only",
        "source_reward_calibration_authorized",
        "source_action_invariance_calibration_authorized",
        "training_authorized",
        "training_pair_authorized",
        "scientific_claim_authorized",
        "production_claim_forbidden",
        "paired_target_accessed",
        "forward_callback_present",
        "custom_core_present",
        "training",
    },
    "local evidence field set differs",
)
local_digest = canonical_sha(local)
distributed = receipt.get("distributed")
require(
    isinstance(distributed, dict)
    and set(distributed)
    == {
        "world_size",
        "ulysses_size",
        "upstream_shared_step_returns_full_sequence_per_rank",
        "all_gather_full_evidence_digest_only",
        "rank_records",
    }
    and distributed.get("world_size") == 4
    and distributed.get("ulysses_size") == 4
    and distributed.get("upstream_shared_step_returns_full_sequence_per_rank")
    is True
    and distributed.get("all_gather_full_evidence_digest_only") is True,
    "WORLD4/Ulysses4 evidence contract differs",
)
rank_records = distributed.get("rank_records")
require(
    isinstance(rank_records, list) and len(rank_records) == 4,
    "rank evidence count differs",
)
require(
    [row.get("rank") for row in rank_records] == [0, 1, 2, 3],
    "rank ordering differs",
)
for row in rank_records:
    require(
        row.get("world_size") == 4
        and row.get("ulysses_size") == 4
        and row.get("local_evidence_digest") == local_digest,
        "SP4 ranks did not report byte-identical canonical evidence",
    )

require(
    local.get("method") == "bernini-iar-official-runtime-smoke-v1"
    and local.get("method_source_revision") == source_revision
    and local.get("method_source_archive_sha256") == source_archive_sha256
    and local.get("launcher_source_sha256") == launcher_source_sha256
    and local.get("bernini_commit") == bernini_commit
    and local.get("veomni_commit") == veomni_commit
    and local.get("checkpoint_path") == str(Path(checkpoint_value).resolve())
    and local.get("checkpoint_tree_sha256") == checkpoint_tree_sha256,
    "source/model provenance differs",
)
checkpoint_identity = local.get("checkpoint_content_identity")
require(
    isinstance(checkpoint_identity, dict)
    and checkpoint_identity.get("manifest_path")
    == str(Path(checkpoint_manifest_value).resolve())
    and checkpoint_identity.get("manifest_sha256_computed")
    == checkpoint_manifest_sha256
    and checkpoint_identity.get("manifest_sha256_expected")
    == checkpoint_manifest_sha256
    and checkpoint_identity.get("verified_file_count") == 23
    and checkpoint_identity.get("every_file_sha256_verified") is True
    and sha_re.fullmatch(str(checkpoint_identity.get("verified_entries_digest")))
    is not None,
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
    and set(candidate)
    == {
        "proposal_source_iid",
        "proposal_origin",
        "proposal_artifact",
        "correct_source_artifact",
        "native_provenance",
        "paired_target_accessed",
        "canonical_message_schema",
    }
    and candidate.get("proposal_source_iid") == proposal_source_iid
    and candidate.get("proposal_origin") == "native_rollout_predecode_latent"
    and candidate.get("paired_target_accessed") is False,
    "candidate binding differs",
)
message_schema = candidate.get("canonical_message_schema")
require(isinstance(message_schema, dict), "canonical message schema is absent")
unsigned_message_schema = dict(message_schema)
message_schema_digest = unsigned_message_schema.pop("schema_evidence_digest", None)
raw_negative_manifest = json.loads(
    plain_file(negative_manifest_value).read_text(encoding="utf-8")
)
raw_negative_rows = raw_negative_manifest.get("hard_negatives")
require(
    isinstance(raw_negative_rows, list) and len(raw_negative_rows) == 2,
    "canonical message negative inputs differ",
)
require(
    set(message_schema)
    == {
        "schema_version",
        "sample_mapping_keys",
        "ordered_messages",
        "canonical_json_ascii_sort_keys",
        "constructed_inside_runtime_per_instruction",
        "candidate_dataset_row_accessed",
        "candidate_dataset_index_consumed",
        "candidate_dataset_iid_consumed",
        "encode_renderer_messages_exact_input_guard",
        "official_encoder_calls_per_task_condition",
        "action_sample",
        "hard_negative_samples",
        "noop_sample",
        "schema_evidence_digest",
    }
    and message_schema.get("schema_version")
    == "bernini-iar-canonical-message-v1"
    and message_schema.get("sample_mapping_keys") == ["inputs"]
    and message_schema.get("ordered_messages")
    == [
        {
            "index": 0,
            "type": "video",
            "has_loss": 0,
            "keys": ["has_loss", "type"],
        },
        {
            "index": 1,
            "type": "text",
            "has_loss": 0,
            "keys": ["has_loss", "text", "type"],
            "text_source": "hash_bound_instruction",
        },
        {
            "index": 2,
            "type": "video_gen",
            "has_loss": 1,
            "keys": ["has_loss", "type"],
        },
    ]
    and message_schema.get("canonical_json_ascii_sort_keys") is True
    and message_schema.get("constructed_inside_runtime_per_instruction") is True
    and message_schema.get("candidate_dataset_row_accessed") is False
    and message_schema.get("candidate_dataset_index_consumed") is False
    and message_schema.get("candidate_dataset_iid_consumed") is False
    and message_schema.get("encode_renderer_messages_exact_input_guard") is True
    and message_schema.get("official_encoder_calls_per_task_condition") == 1
    and message_schema.get("action_sample")
    == canonical_sample_identity(action_instruction, action_sha256)
    and message_schema.get("hard_negative_samples")
    == [
        canonical_sample_identity(row["instruction"], row["instruction_sha256"])
        for row in raw_negative_rows
    ]
    and message_schema.get("noop_sample")
    == canonical_sample_identity(noop_instruction, noop_sha256)
    and sha_re.fullmatch(str(message_schema_digest)) is not None
    and canonical_sha(unsigned_message_schema) == message_schema_digest,
    "canonical candidate-free message schema differs",
)
text_conditions = local.get("text_conditions")
require(
    isinstance(text_conditions, dict)
    and set(text_conditions)
    == {
        "action_instruction_sha256",
        "noop_instruction_sha256",
        "t2v_action_prompt_sha256",
        "t2v_hard_negative_prompt_sha256s",
        "mv2v_noop_prompt_sha256",
        "mv2v_action_prompt_sha256",
        "official_tokenization",
    }
    and text_conditions.get("action_instruction_sha256") == action_sha256
    and text_conditions.get("noop_instruction_sha256") == noop_sha256
    and text_conditions.get("official_tokenization") is True
    and sha_re.fullmatch(str(text_conditions.get("t2v_action_prompt_sha256")))
    is not None
    and isinstance(text_conditions.get("t2v_hard_negative_prompt_sha256s"), list)
    and len(text_conditions["t2v_hard_negative_prompt_sha256s"]) == 2
    and all(
        sha_re.fullmatch(str(item)) is not None
        for item in text_conditions["t2v_hard_negative_prompt_sha256s"]
    )
    and sha_re.fullmatch(str(text_conditions.get("mv2v_noop_prompt_sha256")))
    is not None
    and sha_re.fullmatch(str(text_conditions.get("mv2v_action_prompt_sha256")))
    is not None,
    "official text-condition binding differs",
)
wrong_sources = local.get("wrong_sources")
require(
    isinstance(wrong_sources, list) and len(wrong_sources) == 1,
    "exact M=1 wrong-source evidence differs",
)
wrong = wrong_sources[0]
match = wrong.get("match_manifest") if isinstance(wrong, dict) else None
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
    and set(wrong)
    == {
        "row_index",
        "iid",
        "source_video_sha256",
        "message_template_columns_loaded",
        "artifact",
        "provenance",
        "match_manifest",
        "paired_target_accessed",
    }
    and wrong.get("row_index") == int(wrong_row_index)
    and wrong.get("iid") == wrong_iid
    and wrong.get("source_video_sha256") == wrong_source_video_sha256
    and wrong.get("message_template_columns_loaded")
    == ["iid", "inputs", "source_video_sha256"]
    and wrong.get("paired_target_accessed") is False
    and isinstance(match, dict)
    and match.get("candidate_iid") == proposal_source_iid
    and match.get("candidate_source_video_sha256")
    == proposal_source_video_sha256
    and match.get("wrong_source_iid") == wrong_iid
    and match.get("wrong_source_video_sha256") == wrong_source_video_sha256
    and match.get("path") == str(Path(wrong_match_value).resolve())
    and match.get("file_sha256") == wrong_match_sha256
    and isinstance(criteria, dict)
    and set(criteria) == expected_criteria
    and all(type(criteria.get(key)) is bool for key in expected_criteria)
    and all(
        criteria.get(key) is True
        for key in (
            "distinct_identity",
            "same_actor_class",
            "same_actor_count",
            "same_spatial_bucket",
            "same_length",
            "manual_reviewed",
        )
    )
    and match.get("declared_use") == "runtime_plumbing_only"
    and match.get("source_reward_calibration_authorized") is False
    and match.get("scientific_eligibility") is False,
    "wrong-source plumbing evidence differs",
)
require(
    criteria.get("same_camera_class") is False
    or criteria.get("same_composition_class") is False,
    "launcher is pinned to the current camera/composition-confounded donor",
)

manifest = local.get("hard_negative_manifest")
manifest_rows = manifest.get("hard_negatives") if isinstance(manifest, dict) else None
require(
    isinstance(manifest, dict)
    and set(manifest)
    == {
        "schema_version",
        "action_instruction_sha256",
        "hard_negatives",
        "hard_negative_count",
        "noop_instruction_sha256",
        "noop_present",
        "energy_semantics",
        "energies_supplied_externally",
        "declared_use",
        "manifest_digest",
        "path",
        "file_sha256",
    }
    and manifest.get("schema_version")
    == "bernini-iar-hard-negative-manifest-v1"
    and manifest.get("path") == str(Path(negative_manifest_value).resolve())
    and manifest.get("file_sha256") == negative_manifest_sha256
    and manifest.get("action_instruction_sha256") == action_sha256
    and manifest.get("energy_semantics")
    == "runtime_fp32_mse_to_epsilon_minus_bridge_clean_lower_is_harder"
    and manifest.get("declared_use") == "engineering_plumbing_only"
    and manifest.get("hard_negative_count") == 2
    and manifest.get("noop_instruction_sha256") == noop_sha256
    and manifest.get("noop_present") is True
    and manifest.get("energies_supplied_externally") is False
    and isinstance(manifest_rows, list)
    and len(manifest_rows) == 2,
    "hard-negative manifest receipt binding differs",
)
require(
    manifest.get("manifest_digest") == canonical_sha(raw_negative_manifest),
    "hard-negative manifest canonical digest differs",
)
negative_hashes = [row.get("instruction_sha256") for row in manifest_rows]
require(
    len(set(negative_hashes)) == 2
    and action_sha256 not in negative_hashes
    and negative_hashes.count(noop_sha256) == 1,
    "K=2 action/no-op digest relations differ",
)

proposal_artifact = candidate.get("proposal_artifact")
source_artifact = candidate.get("correct_source_artifact")
wrong_artifact = wrong.get("artifact")
require(
    isinstance(proposal_artifact, dict)
    and isinstance(source_artifact, dict)
    and isinstance(wrong_artifact, dict),
    "proposal/correct/wrong latent artifacts are absent",
)
proposal_identity = tensor_identity(proposal_artifact.get("tensor_identity"), ndim=5)
correct_identity = tensor_identity(source_artifact.get("tensor_identity"), ndim=5)
wrong_identity = tensor_identity(wrong_artifact.get("tensor_identity"), ndim=5)
homotopy = local.get("homotopy")
require(isinstance(homotopy, dict), "HAT homotopy evidence is absent")
epsilon_identity = tensor_identity(homotopy.get("epsilon"), ndim=5)
for identity in (
    proposal_identity,
    correct_identity,
    wrong_identity,
    epsilon_identity,
):
    require(
        identity["shape"][:3] == [1, 16, 21]
        and identity.get("dtype") == "torch.float32",
        "proposal/source/noise latent contract differs",
    )
require(
    proposal_identity["shape"]
    == correct_identity["shape"]
    == wrong_identity["shape"]
    == epsilon_identity["shape"],
    "proposal/correct/wrong/noise geometry differs",
)
require(
    correct_identity["raw_storage_sha256"]
    != wrong_identity["raw_storage_sha256"],
    "wrong source is tensor-identical to correct source",
)


def audit_artifact(artifact, expected_path, expected_sha, role, identity):
    require(isinstance(artifact, dict), f"{role} artifact is absent")
    expected_source = (
        "native_sampler_before_vae_decode"
        if role == "native_sampler_proposal"
        else "source_video_vae_encode_before_any_decode"
    )
    artifact_identity = tensor_identity(artifact.get("tensor_identity"), ndim=5)
    require(
        artifact.get("path") == str(Path(expected_path).resolve())
        and artifact.get("file_sha256") == expected_sha
        and artifact.get("tensor_key") == "normalized_clean_latent"
        and artifact.get("mp4_decode_reencode_used") is False
        and artifact.get("metadata")
        == {
            "coordinate": "bernini_normalized_clean_vae_latent",
            "frame_contract": "exact81_latent21",
            "artifact_role": role,
            "source": expected_source,
        }
        and artifact_identity.get("label") == role
        and tensor_payload_identity(artifact_identity)
        == tensor_payload_identity(identity),
        f"{role} artifact binding differs",
    )


audit_artifact(
    proposal_artifact,
    candidate_latent_value,
    candidate_latent_sha256,
    "native_sampler_proposal",
    proposal_identity,
)
audit_artifact(
    source_artifact,
    source_latent_value,
    source_latent_sha256,
    "source_video_condition",
    correct_identity,
)
audit_artifact(
    wrong_artifact,
    wrong_latent_value,
    wrong_latent_sha256,
    "source_video_condition",
    wrong_identity,
)

native_provenance = candidate.get("native_provenance")
candidate_receipt = (
    native_provenance.get("candidate_receipt")
    if isinstance(native_provenance, dict)
    else None
)
source_receipt = (
    native_provenance.get("source_receipt")
    if isinstance(native_provenance, dict)
    else None
)
require(
    isinstance(native_provenance, dict)
    and native_provenance.get("candidate_arm") == candidate_arm
    and native_provenance.get("source_video_sha256")
    == proposal_source_video_sha256
    and native_provenance.get("action_prompt_sha256") == action_sha256
    and native_provenance.get("checkpoint_tree_sha256")
    == checkpoint_tree_sha256
    and native_provenance.get("checkpoint_content_identity")
    == checkpoint_identity
    and native_provenance.get("proposal_latent_sha256")
    == candidate_latent_sha256
    and native_provenance.get("source_condition_latent_sha256")
    == source_latent_sha256
    and native_provenance.get("paired_target_accessed") is False
    and native_provenance.get("mp4_decode_reencode_used") is False
    and isinstance(candidate_receipt, dict)
    and candidate_receipt.get("path")
    == str(Path(candidate_provenance_value).resolve())
    and candidate_receipt.get("file_sha256") == candidate_provenance_sha256
    and sha_re.fullmatch(str(candidate_receipt.get("receipt_digest"))) is not None
    and isinstance(source_receipt, dict)
    and source_receipt.get("path")
    == str(Path(source_provenance_value).resolve())
    and source_receipt.get("file_sha256") == source_provenance_sha256
    and sha_re.fullmatch(str(source_receipt.get("receipt_digest"))) is not None,
    "native proposal/source provenance closure differs",
)

wrong_provenance = wrong.get("provenance")
wrong_source_receipt = (
    wrong_provenance.get("source_receipt")
    if isinstance(wrong_provenance, dict)
    else None
)
require(
    isinstance(wrong_provenance, dict)
    and wrong_provenance.get("source_iid") == wrong_iid
    and wrong_provenance.get("source_video_sha256")
    == wrong_source_video_sha256
    and wrong_provenance.get("checkpoint_tree_sha256")
    == checkpoint_tree_sha256
    and wrong_provenance.get("checkpoint_content_identity")
    == checkpoint_identity
    and wrong_provenance.get("source_condition_latent_sha256")
    == wrong_latent_sha256
    and wrong_provenance.get("source_columns_accessed")
    == ["iid", "source_video", "source_video_sha256"]
    and wrong_provenance.get("target_columns_accessed") == []
    and wrong_provenance.get("target_media_accessed") is False
    and wrong_provenance.get("paired_target_accessed") is False
    and wrong_provenance.get("mp4_decode_reencode_used") is False
    and isinstance(wrong_source_receipt, dict)
    and wrong_source_receipt.get("path")
    == str(Path(wrong_provenance_value).resolve())
    and wrong_source_receipt.get("file_sha256") == wrong_provenance_sha256
    and wrong_source_receipt.get("schema_version")
    == "bernini-source-only-vae-materialization-v1"
    and sha_re.fullmatch(str(wrong_source_receipt.get("receipt_digest")))
    is not None,
    "source-only wrong donor provenance closure differs",
)

require(
    local.get("num_frames") == 81
    and local.get("latent_phases") == 21
    and local.get("patch_size") == [1, 2, 2]
    and homotopy.get("noise_seed") == int(noise_seed)
    and local.get("forward_implementation")
    == "renderer.get_t5_text_embeddings+renderer.diff_dec.shared_step"
    and local.get("active_model_id") in ("transformer_1", "transformer_2")
    and isinstance(local.get("backend"), str)
    and bool(local["backend"])
    and local.get("adapter_state") == "absent_frozen_base"
    and local.get("model_mode") == "eval_inference_mode"
    and local.get("raw_positive_conditional_only") is True
    and local.get("cfg") is False
    and local.get("apg") is False
    and local.get("donor_plumbing_only") is True
    and local.get("source_reward_calibration_authorized") is False
    and local.get("source_action_invariance_calibration_authorized") is False
    and local.get("training_authorized") is False
    and local.get("training_pair_authorized") is False
    and local.get("scientific_claim_authorized") is False
    and local.get("production_claim_forbidden") is True
    and local.get("paired_target_accessed") is False
    and local.get("forward_callback_present") is False
    and local.get("custom_core_present") is False,
    "frozen official-field contract differs",
)
training = local.get("training")
require(
    training
    == {
        "forward_only": True,
        "backward_performed": False,
        "optimizer_present": False,
        "checkpoint_saved": False,
        "adapter_present": False,
    },
    "runtime training non-authorization evidence differs",
)

sigmas = [float(item) for item in sigmas_value.split(",")]
bridges = [float(item) for item in bridge_fractions_value.split(",")]
require(sigmas == [0.8, 0.6, 0.35, 0.15], "S=4 sigma vector differs")
require(bridges == [1.0, 0.5, 0.0], "L=3 bridge input vector differs")
canonical_bridges = bridges
require(local.get("bridge_fractions") == bridges, "receipt bridge order differs")
sigma_records = local.get("sigmas")
require(
    isinstance(sigma_records, list) and len(sigma_records) == 4,
    "receipt sigma records differ",
)
for record, sigma in zip(sigma_records, sigmas):
    sigma_f32 = f32(sigma)
    timestep_f32 = f32(sigma_f32 * 1000.0)
    require(
        record
        == {
            "sigma": sigma_f32,
            "timestep": timestep_f32,
            "sigma_float32_bits_hex": struct.pack("!f", sigma_f32).hex(),
            "timestep_float32_bits_hex": struct.pack("!f", timestep_f32).hex(),
        },
        "receipt sigma RF mapping differs",
    )
require(
    homotopy.get("formula")
    == "q=(1-sigma)*((1-lambda)*S+lambda*P)+sigma*epsilon"
    and homotopy.get("constructed_inside_runtime") is True
    and homotopy.get("caller_provided_sigma_states") is False
    and homotopy.get("epsilon_generated_once_on_cpu") is True
    and homotopy.get("one_proposal_P_for_all_cells") is True
    and homotopy.get("one_correct_source_S_for_all_cells") is True
    and homotopy.get("one_epsilon_for_all_cells") is True
    and homotopy.get("paired_target_used_as_clean_coordinate") is False
    and homotopy.get("linear_bridge_is_field_query_not_reconstruction_target") is True,
    "HAT homotopy construction differs",
)
require(
    tensor_payload_identity(homotopy.get("proposal_P"))
    == tensor_payload_identity(proposal_identity)
    and tensor_payload_identity(homotopy.get("correct_source_S"))
    == tensor_payload_identity(correct_identity),
    "HAT shared P/S tensor identities differ",
)
bridge_clean_coordinates = homotopy.get("bridge_clean_coordinates")
require(
    isinstance(bridge_clean_coordinates, list)
    and len(bridge_clean_coordinates) == 3,
    "HAT bridge-clean coordinate count differs",
)
for bridge_record, bridge_value in zip(bridge_clean_coordinates, bridges):
    bridge_identity = tensor_identity(bridge_record.get("clean_coordinate"), ndim=5)
    require(
        bridge_record.get("bridge_fraction") == bridge_value
        and bridge_record.get("bridge_fraction_float32_bits_hex")
        == struct.pack("!f", bridge_value).hex()
        and bridge_record.get("source_endpoint_exact") is True
        and bridge_record.get("proposal_endpoint_exact") is True
        and bridge_record.get("constructed_from_shared_P_and_S") is True
        and bridge_identity["shape"] == proposal_identity["shape"],
        "HAT bridge-clean coordinate differs",
    )
branch_order = [
    "frozen_t2v_action",
    "frozen_t2v_hard_negative[0]",
    "frozen_t2v_hard_negative[1]",
    "frozen_identity_noop_correct",
    "frozen_identity_noop_wrong_source[0]",
    "frozen_identity_action_correct",
    "frozen_identity_action_wrong_source[0]",
]
require(local.get("branch_order") == branch_order, "seven-branch order differs")
require(local.get("hard_negative_count") == 2, "K count differs")
require(local.get("wrong_source_count") == 1, "M count differs")
require(local.get("forwards_per_rank") == 84, "L*S*7 forward count differs")

cells = local.get("cell_records")
require(isinstance(cells, list) and len(cells) == 12, "L*S cell grid differs")
seen_coordinates = set()
for cell in cells:
    require(isinstance(cell, dict), "cell record is not an object")
    unsigned_cell = dict(cell)
    cell_digest = unsigned_cell.pop("cell_digest", None)
    require(
        sha_re.fullmatch(str(cell_digest)) is not None
        and canonical_sha(unsigned_cell) == cell_digest,
        "cell canonical digest differs",
    )
    bridge = cell.get("bridge_fraction")
    point = cell.get("flow_query")
    require(
        bridge in canonical_bridges and isinstance(point, dict),
        "cell coordinate differs",
    )
    sigma = point.get("sigma")
    require(sigma in [f32(item) for item in sigmas], "cell sigma differs")
    coordinate = (bridge, sigma)
    require(coordinate not in seen_coordinates, "duplicate bridge/sigma cell")
    seen_coordinates.add(coordinate)
    require(
        point.get("timestep") == f32(sigma * 1000.0)
        and cell.get("bridge_fraction_float32_bits_hex")
        == struct.pack("!f", bridge).hex()
        and point.get("sigma_float32_bits_hex")
        == struct.pack("!f", sigma).hex()
        and point.get("timestep_float32_bits_hex")
        == struct.pack("!f", f32(sigma * 1000.0)).hex()
        and cell.get("branch_order") == branch_order
        and cell.get("forward_count") == 7,
        "cell shared coordinate/branch execution differs",
    )
    require(
        cell.get("single_variable_intervention")
        == {
            "t2v_is_direct_correct_mv2v_target_tail": True,
            "wrong_source_changes_prefix_only": True,
            "same_target_tail_and_rope": True,
            "same_sigma_and_timestep": True,
        }
        and cell.get("energies_computed_inside_runtime_from_actual_fields") is True
        and cell.get("hard_negative_energy_semantics")
        == "runtime_fp32_mse_to_epsilon_minus_bridge_clean_lower_is_harder"
        and cell.get("no_forward_callback") is True
        and cell.get("direct_official_shared_step") is True,
        "cell intervention/runtime provenance differs",
    )
    geometry = cell.get("geometry")
    require(
        isinstance(geometry, dict)
        and geometry.get("verified") is True
        and type(geometry.get("target_tokens")) is int
        and geometry["target_tokens"] > 0
        and geometry["target_tokens"] % 21 == 0
        and geometry.get("t2v_total_tokens") == geometry["target_tokens"]
        and geometry.get("mv2v_total_tokens") == 2 * geometry["target_tokens"]
        and geometry.get("t2v_is_correct_mv2v_target_tail_view") is True
        and geometry.get("wrong_source_changes_prefix_only") is True
        and geometry.get("correct_wrong_full_rotary_exact_equal") is True
        and geometry.get("target_source_id") == 0
        and geometry.get("source_source_id") == 1
        and geometry.get("rotary_dtype") == "torch.complex128"
        and geometry.get("target_tail_mask")
        == "T2V all N; MV2V final N of 2N",
        "cell exact81 T2V/MV2V geometry differs",
    )
    require(geometry == local.get("geometry"), "cell geometry differs from grid geometry")
    target_query = tensor_identity(cell.get("target_query"), ndim=5)
    true_velocity = tensor_identity(cell.get("true_velocity"), ndim=3)
    teacher_residual = tensor_identity(cell.get("teacher_action_residual"), ndim=3)
    require(
        target_query["shape"] == proposal_identity["shape"]
        and true_velocity["shape"] == teacher_residual["shape"],
        "cell query/field tensor geometry differs",
    )
    fields = cell.get("field_identities")
    require(
        isinstance(fields, dict)
        and set(fields) == set(branch_order)
        and cell.get("field_set_digest") == canonical_sha(fields),
        "cell frozen field-set closure differs",
    )
    field_shapes = []
    for name in branch_order:
        field_identity = tensor_identity(fields.get(name), ndim=3)
        require(field_identity.get("label") == name, "frozen field label differs")
        field_shapes.append(field_identity["shape"])
    require(
        len({tuple(item) for item in field_shapes}) == 1
        and field_shapes[0] == teacher_residual["shape"],
        "seven frozen fields do not share one target-tail geometry",
    )
    energies = cell.get("hard_negative_energies")
    require(
        isinstance(energies, list)
        and len(energies) == 2
        and all(finite_number(item) and item >= 0.0 for item in energies),
        "runtime FP32 negative energies differ",
    )
    action_energy = cell.get("action_energy_EA")
    margins = cell.get("ordering_margins_Ek_minus_EA")
    require(
        isinstance(action_energy, list)
        and len(action_energy) == 1
        and finite_number(action_energy[0])
        and action_energy[0] >= 0.0
        and isinstance(margins, list)
        and len(margins) == 1
        and isinstance(margins[0], list)
        and len(margins[0]) == 2
        and all(finite_number(item) for item in margins[0])
        and all(
            math.isclose(
                margin,
                negative_energy - action_energy[0],
                rel_tol=1.0e-6,
                abs_tol=1.0e-7,
            )
            for margin, negative_energy in zip(margins[0], energies)
        )
        and cell.get("rf_energy_reduction")
        == "mean_over_all_nonbatch_target_token_and_patch_feature_dimensions"
        and cell.get("rf_squared_error_proxy_not_likelihood_or_free_energy") is True
        and cell.get("ordering_is_diagnostic_not_training_authorization") is True,
        "actual-action RF energy/order diagnostic differs",
    )
    metrics = cell.get("metrics")
    require(
        isinstance(metrics, dict)
        and metrics.get("source_action_invariance_calibration_authorized") is False
        and metrics.get("M_equals_one_plumbing_only") is True
        and isinstance(metrics.get("tangent_rank"), list)
        and len(metrics["tangent_rank"]) == 1
        and all(type(item) is int and item >= 0 for item in metrics["tangent_rank"]),
        "cell diagnostic authorization/rank differs",
    )
    for metric_name in (
        "projection_retention",
        "raw_action_rms",
        "projected_action_rms",
        "capped_action_rms",
        "max_abs_postprojection_tangent_cosine",
        "source_action_invariance_cosine",
        "source_action_invariance_symmetric_norm_ratio",
        "projected_action_alignment_correct",
        "projected_action_alignment_wrong_sources",
    ):
        require(finite_tree(metrics.get(metric_name)), f"non-finite {metric_name}")
    independent = cell.get("independent_recompute")
    independent_identities = (
        independent.get("tensor_identities") if isinstance(independent, dict) else None
    )
    require(
        isinstance(independent, dict)
        and independent.get("verified") is True
        and independent.get("implementation_independent_of_core_receipt") is True
        and independent.get("projection_uses_noop_source_swaps_only") is True
        and independent.get("action_source_swaps_diagnostic_only") is True
        and independent.get("gauge_invariant_cap_recomputed") is True
        and independent.get("sigma_schedule_recomputed_without_core_helper") is True
        and independent.get("capped_teacher_recomputed") is True
        and independent.get("low_sigma_exact_zero_verified") is True
        and independent.get("action_energy_EA") == action_energy
        and independent.get("negative_energies") == [energies]
        and independent.get("ordering_margins_Ek_minus_EA") == margins
        and independent.get("rf_energy_reduction")
        == "mean_over_all_nonbatch_target_token_and_patch_feature_dimensions"
        and independent.get("rf_squared_error_proxy_not_likelihood_or_free_energy")
        is True
        and independent.get("ordering_is_diagnostic_not_training_authorization")
        is True
        and independent.get("source_action_invariance_calibration_authorized") is False
        and isinstance(independent_identities, dict)
        and independent.get("digest") == canonical_sha(independent_identities)
        and sha_re.fullmatch(str(cell.get("core_receipt_digest"))) is not None,
        "cell independent IAR recomputation differs",
    )
    for identity in independent_identities.values():
        tensor_identity(identity)
require(
    seen_coordinates
    == {(bridge, f32(sigma)) for bridge in canonical_bridges for sigma in sigmas},
    "bridge/sigma grid is incomplete",
)

continuity = local.get("continuity")
require(isinstance(continuity, dict), "bridge continuity diagnostics are absent")
comparisons = continuity.get("adjacent_comparisons")
require(
    continuity.get("grid_complete") is True
    and continuity.get("bridge_order_preserved") == bridges
    and isinstance(comparisons, list)
    and len(comparisons) == 8
    and continuity.get("comparison_count") == 8
    and continuity.get("metric_only_no_training_gate") is True
    and continuity.get("source_action_invariance_calibration_authorized") is False,
    "bridge continuity contract differs",
)
expected_comparisons = [
    (struct.pack("!f", f32(sigma)).hex(), left, right)
    for sigma in sigmas
    for left, right in zip(bridges, bridges[1:])
]
for row, (sigma_bits, left, right) in zip(comparisons, expected_comparisons):
    require(
        isinstance(row, dict)
        and row.get("sigma_float32_bits_hex") == sigma_bits
        and row.get("left_bridge_fraction") == left
        and row.get("right_bridge_fraction") == right
        and finite_tree(row.get("projection_retention_left"))
        and finite_tree(row.get("projection_retention_right")),
        "adjacent-lambda continuity coordinate differs",
    )
    for diagnostic_name in (
        "teacher_residual",
        "identity_noop_anchor",
        "raw_action_residual",
    ):
        diagnostic = row.get(diagnostic_name)
        require(
            isinstance(diagnostic, dict)
            and set(diagnostic)
            == {
                "cosine_defined",
                "cosine",
                "left_rms",
                "right_rms",
                "relative_rms_jump",
            }
            and isinstance(diagnostic.get("cosine_defined"), list)
            and len(diagnostic["cosine_defined"]) == 1
            and all(type(item) is bool for item in diagnostic["cosine_defined"])
            and finite_tree(diagnostic.get("cosine"))
            and finite_tree(diagnostic.get("left_rms"))
            and finite_tree(diagnostic.get("right_rms"))
            and finite_tree(diagnostic.get("relative_rms_jump")),
            f"non-finite {diagnostic_name} continuity diagnostic",
        )
iar_core = local.get("iar_core")
require(isinstance(iar_core, dict), "IAR core evidence is absent")
core_closure = iar_core.get("full_cell_grid_closure")
require(
    iar_core.get("method") == "bernini-identity-anchored-action-residual-v2"
    and iar_core.get("receipt_schema")
    == "bernini-identity-anchored-action-residual-receipt-v2"
    and iar_core.get("direct_corrected_core_call") is True
    and iar_core.get("replaceable_core") is False
    and iar_core.get("projection_nuisance")
    == "mv2v_noop_correct_minus_mv2v_noop_wrong"
    and iar_core.get("action_source_swaps_diagnostic_only") is True
    and iar_core.get("M_equals_one_plumbing_only_uncalibrated") is True
    and iar_core.get("source_action_invariance_calibration_authorized") is False
    and iar_core.get("training_authorized_by_diagnostic") is False
    and iar_core.get("independent_recompute_every_cell") is True
    and isinstance(core_closure, dict)
    and core_closure.get("cell_digests")
    == [cell["cell_digest"] for cell in cells]
    and core_closure.get("field_set_digests")
    == [cell["field_set_digest"] for cell in cells]
    and core_closure.get("independent_recompute_digests")
    == [cell["independent_recompute"]["digest"] for cell in cells]
    and core_closure.get("core_receipt_digests")
    == [cell["core_receipt_digest"] for cell in cells]
    and iar_core.get("full_cell_grid_digest") == canonical_sha(core_closure),
    "IAR core frozen-field evidence differs",
)

for path_value, expected_sha in (
    (checkpoint_manifest_value, checkpoint_manifest_sha256),
    (dataset_summary_value, dataset_summary_sha256),
    (candidate_latent_value, candidate_latent_sha256),
    (source_latent_value, source_latent_sha256),
    (candidate_provenance_value, candidate_provenance_sha256),
    (source_provenance_value, source_provenance_sha256),
    (wrong_latent_value, wrong_latent_sha256),
    (wrong_provenance_value, wrong_provenance_sha256),
    (wrong_match_value, wrong_match_sha256),
    (negative_manifest_value, negative_manifest_sha256),
):
    require(file_sha(plain_file(path_value)) == expected_sha, "bound input changed")

print("BERNINI_IAR_OFFICIAL_RUNTIME_SP4_STRONG_AUDIT_OK")
PY
}

receipt_a_sha_round1="$(sha256sum "${receipt_path_a}" | awk '{print $1}')"
audit_receipt "${receipt_path_a}" "${noise_seed_a}" || fail "seed-A receipt audit round 1 failed"
[[ "$(sha256sum "${receipt_path_a}" | awk '{print $1}')" == "${receipt_a_sha_round1}" ]] || fail "seed-A receipt changed during audit round 1"
receipt_a_sha_round2="$(sha256sum "${receipt_path_a}" | awk '{print $1}')"
[[ "${receipt_a_sha_round2}" == "${receipt_a_sha_round1}" ]] || fail "seed-A receipt hash closure differs"
audit_receipt "${receipt_path_a}" "${noise_seed_a}" || fail "seed-A receipt audit round 2 failed"
[[ "$(sha256sum "${receipt_path_a}" | awk '{print $1}')" == "${receipt_a_sha_round2}" ]] || fail "seed-A receipt changed during audit round 2"

receipt_b_sha_round1="$(sha256sum "${receipt_path_b}" | awk '{print $1}')"
audit_receipt "${receipt_path_b}" "${noise_seed_b}" || fail "seed-B receipt audit round 1 failed"
[[ "$(sha256sum "${receipt_path_b}" | awk '{print $1}')" == "${receipt_b_sha_round1}" ]] || fail "seed-B receipt changed during audit round 1"
receipt_b_sha_round2="$(sha256sum "${receipt_path_b}" | awk '{print $1}')"
[[ "${receipt_b_sha_round2}" == "${receipt_b_sha_round1}" ]] || fail "seed-B receipt hash closure differs"
audit_receipt "${receipt_path_b}" "${noise_seed_b}" || fail "seed-B receipt audit round 2 failed"
[[ "$(sha256sum "${receipt_path_b}" | awk '{print $1}')" == "${receipt_b_sha_round2}" ]] || fail "seed-B receipt changed during audit round 2"

"${python_bin}" -B - \
  "${receipt_path_a}" "${receipt_a_sha_round2}" "${noise_seed_a}" \
  "${receipt_path_b}" "${receipt_b_sha_round2}" "${noise_seed_b}" \
  "${checkpoint}" "${checkpoint_manifest}" "${checkpoint_manifest_sha256}" <<'PY'
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import sys


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_receipt(path_value, expected_sha):
    path = Path(path_value).resolve(strict=True)
    require(path.is_file() and not path.is_symlink(), "dual audit receipt is not plain")
    require(file_sha(path) == expected_sha, "dual audit receipt hash differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "dual audit receipt is not an object")
    return value


(
    receipt_a_value,
    receipt_a_sha,
    seed_a_value,
    receipt_b_value,
    receipt_b_sha,
    seed_b_value,
    checkpoint_value,
    checkpoint_manifest_value,
    checkpoint_manifest_sha,
) = sys.argv[1:]
seed_a = int(seed_a_value)
seed_b = int(seed_b_value)
require(seed_a != seed_b, "dual WORLD4 seeds are not distinct")
receipt_a = load_receipt(receipt_a_value, receipt_a_sha)
receipt_b = load_receipt(receipt_b_value, receipt_b_sha)

for label, receipt, seed in (
    ("A", receipt_a, seed_a),
    ("B", receipt_b, seed_b),
):
    require(
        receipt.get("schema_version")
        == "bernini-iar-official-runtime-smoke-receipt-v1"
        and receipt.get("method") == "bernini-iar-official-runtime-smoke-v1",
        f"seed-{label} schema/method differs",
    )
    require(
        receipt.get("engineering_smoke_only") is True
        and receipt.get("donor_plumbing_only") is True
        and receipt.get("source_reward_calibration_authorized") is False
        and receipt.get("source_action_invariance_calibration_authorized") is False
        and receipt.get("training_authorized") is False
        and receipt.get("training_pair_authorized") is False
        and receipt.get("scientific_claim_authorized") is False
        and receipt.get("production_claim_forbidden") is True
        and receipt.get("paired_target_accessed") is False
        and receipt.get("forward_callback_present") is False
        and receipt.get("custom_core_present") is False,
        f"seed-{label} authorization state differs",
    )
    distributed = receipt.get("distributed")
    local = receipt.get("local_evidence")
    homotopy = local.get("homotopy") if isinstance(local, dict) else None
    require(
        isinstance(distributed, dict)
        and distributed.get("world_size") == 4
        and distributed.get("ulysses_size") == 4
        and isinstance(local, dict)
        and local.get("forwards_per_rank") == 84
        and local.get("num_frames") == 81
        and local.get("latent_phases") == 21
        and local.get("source_reward_calibration_authorized") is False
        and local.get("source_action_invariance_calibration_authorized") is False
        and local.get("training_authorized") is False
        and local.get("training_pair_authorized") is False
        and local.get("training")
        == {
            "forward_only": True,
            "backward_performed": False,
            "optimizer_present": False,
            "checkpoint_saved": False,
            "adapter_present": False,
        }
        and local.get("branch_order")
        == [
            "frozen_t2v_action",
            "frozen_t2v_hard_negative[0]",
            "frozen_t2v_hard_negative[1]",
            "frozen_identity_noop_correct",
            "frozen_identity_noop_wrong_source[0]",
            "frozen_identity_action_correct",
            "frozen_identity_action_wrong_source[0]",
        ]
        and isinstance(homotopy, dict)
        and homotopy.get("noise_seed") == seed,
        f"seed-{label} local WORLD4/exact81 contract differs",
    )

local_a = receipt_a["local_evidence"]
local_b = receipt_b["local_evidence"]
seed_dependent_local = {"homotopy", "cell_records", "continuity", "iar_core"}
stable_a = {
    key: value for key, value in local_a.items() if key not in seed_dependent_local
}
stable_b = {
    key: value for key, value in local_b.items() if key not in seed_dependent_local
}
require(stable_a == stable_b, "dual WORLD4 inputs/provenance differ outside seed")

homotopy_a = dict(local_a["homotopy"])
homotopy_b = dict(local_b["homotopy"])
epsilon_a = homotopy_a.pop("epsilon", None)
epsilon_b = homotopy_b.pop("epsilon", None)
homotopy_a.pop("noise_seed", None)
homotopy_b.pop("noise_seed", None)
require(
    homotopy_a == homotopy_b,
    "dual WORLD4 HAT construction differs outside epsilon/seed",
)
sha_re = re.compile(r"[0-9a-f]{64}")
require(
    isinstance(epsilon_a, dict)
    and isinstance(epsilon_b, dict)
    and sha_re.fullmatch(str(epsilon_a.get("raw_storage_sha256"))) is not None
    and sha_re.fullmatch(str(epsilon_b.get("raw_storage_sha256"))) is not None
    and epsilon_a.get("shape") == epsilon_b.get("shape")
    and epsilon_a.get("dtype") == epsilon_b.get("dtype") == "torch.float32"
    and epsilon_a.get("raw_storage_sha256") != epsilon_b.get("raw_storage_sha256"),
    "dual seeds did not produce distinct same-geometry FP32 epsilon",
)

# Rehash the entire active checkpoint after both runtimes.  The two runtime
# receipts already prove the same identity at load time; this closes mutation
# between model loading and the final launcher audit.
checkpoint_root = Path(checkpoint_value).resolve(strict=True)
manifest_path = Path(checkpoint_manifest_value).resolve(strict=True)
require(
    checkpoint_root.is_dir()
    and not checkpoint_root.is_symlink()
    and manifest_path.is_file()
    and not manifest_path.is_symlink(),
    "post-runtime checkpoint/manifest path is invalid",
)
require(
    file_sha(manifest_path) == checkpoint_manifest_sha,
    "post-runtime checkpoint manifest hash differs",
)
lines = manifest_path.read_text(encoding="utf-8").splitlines()
require(len(lines) == 23, "post-runtime checkpoint manifest count differs")
manifest_line = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)")
expected_files = {}
for line in lines:
    match = manifest_line.fullmatch(line)
    require(match is not None, "checkpoint manifest line is non-canonical")
    expected_sha, raw_path = match.groups()
    relative = PurePosixPath(raw_path)
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        "checkpoint manifest path is unsafe",
    )
    normalized = PurePosixPath(
        *(part for part in relative.parts if part not in ("", "."))
    ).as_posix()
    require(
        bool(normalized) and normalized not in expected_files,
        "checkpoint manifest path is empty or duplicated",
    )
    expected_files[normalized] = expected_sha

actual_files = set()
for path in checkpoint_root.rglob("*"):
    relative = path.relative_to(checkpoint_root)
    if ".cache" in relative.parts:
        continue
    mode = path.lstat().st_mode
    require(not stat.S_ISLNK(mode), "checkpoint contains a non-cache symlink")
    if stat.S_ISREG(mode):
        actual_files.add(relative.as_posix())
    else:
        require(stat.S_ISDIR(mode), "checkpoint contains a non-regular entry")
require(
    actual_files == set(expected_files),
    "post-runtime checkpoint file set differs from manifest",
)
verified_entries = []
for relative in sorted(expected_files):
    path = checkpoint_root / relative
    require(path.is_file() and not path.is_symlink(), "checkpoint file is not plain")
    actual_sha = file_sha(path)
    require(
        actual_sha == expected_files[relative],
        f"post-runtime checkpoint content differs: {relative}",
    )
    verified_entries.append({"path": relative, "sha256": actual_sha})
post_checkpoint_identity = {
    "manifest_path": str(manifest_path),
    "manifest_sha256_computed": checkpoint_manifest_sha,
    "manifest_sha256_expected": checkpoint_manifest_sha,
    "verified_file_count": len(verified_entries),
    "every_file_sha256_verified": True,
    "verified_entries_digest": object_sha(verified_entries),
}
require(
    post_checkpoint_identity == local_a.get("checkpoint_content_identity")
    == local_b.get("checkpoint_content_identity"),
    "post-runtime checkpoint identity differs from both WORLD4 receipts",
)
print(
    "BERNINI_IAR_DP2_SP4_CROSS_SEED_INPUT_AUDIT_OK "
    "cross_seed_numeric_stability_authorized=false"
)
PY

[[ "$(sha256sum "${receipt_path_a}" | awk '{print $1}')" == "${receipt_a_sha_round2}" ]] || fail "seed-A receipt changed during dual audit"
[[ "$(sha256sum "${receipt_path_b}" | awk '{print $1}')" == "${receipt_b_sha_round2}" ]] || fail "seed-B receipt changed during dual audit"

for entry in "${immutable_files[@]}"; do
  path="${entry%%|*}"
  remainder="${entry#*|}"
  expected="${remainder%%|*}"
  label="${remainder#*|}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || fail "immutable input changed after runtime: ${label}"
done
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged source archive changed after runtime"
[[ "$(sha256sum "${archived_launcher}" | awk '{print $1}')" == "${launcher_source_sha256}" ]] || fail "archived launcher changed after runtime"
chmod 0444 -- "${receipt_path_a}" "${torchrun_log_a}" "${receipt_path_b}" "${torchrun_log_b}"
echo "[bernini-iar-dp2sp4] COMPLETE receipt_a_sha256=${receipt_a_sha_round2} receipt_b_sha256=${receipt_b_sha_round2} local_audit=BERNINI_IAR_OFFICIAL_RUNTIME_SP4_STRONG_AUDIT_OK cross_seed_input_audit=BERNINI_IAR_DP2_SP4_CROSS_SEED_INPUT_AUDIT_OK cross_seed_numeric_stability_authorized=false"
