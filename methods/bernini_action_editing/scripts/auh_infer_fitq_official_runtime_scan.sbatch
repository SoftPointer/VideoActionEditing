#!/usr/bin/env bash
#SBATCH --job-name=bern-fitq-dp2sp4
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=05:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# One eight-MI210 allocation is split into two independent WORLD4/Ulysses4
# groups. Bernini-R has 12 heads, so Ulysses8 is illegal. This launcher runs
# only the frozen exact81 FITQ observation gate; it never invokes a trainer.

source_archive="${FITQ_RUNTIME_SOURCE_ARCHIVE:?set FITQ_RUNTIME_SOURCE_ARCHIVE}"
source_archive_sha256="${FITQ_RUNTIME_SOURCE_ARCHIVE_SHA256:?set FITQ_RUNTIME_SOURCE_ARCHIVE_SHA256}"
source_revision="${FITQ_RUNTIME_SOURCE_REVISION:?set FITQ_RUNTIME_SOURCE_REVISION}"
bernini_root="${FITQ_RUNTIME_BERNINI_ROOT:?set FITQ_RUNTIME_BERNINI_ROOT}"
veomni_root="${FITQ_RUNTIME_VEOMNI_ROOT:?set FITQ_RUNTIME_VEOMNI_ROOT}"
checkpoint="${FITQ_RUNTIME_CHECKPOINT:?set FITQ_RUNTIME_CHECKPOINT}"
checkpoint_manifest="${FITQ_RUNTIME_CHECKPOINT_CONTENT_MANIFEST:?set FITQ_RUNTIME_CHECKPOINT_CONTENT_MANIFEST}"
preprocessed_parquet_dir="${FITQ_RUNTIME_PREPROCESSED_PARQUET_DIR:?set FITQ_RUNTIME_PREPROCESSED_PARQUET_DIR}"
dataset_summary="${FITQ_RUNTIME_DATASET_SUMMARY:?set FITQ_RUNTIME_DATASET_SUMMARY}"
dataset_summary_sha256="${FITQ_RUNTIME_DATASET_SUMMARY_SHA256:?set FITQ_RUNTIME_DATASET_SUMMARY_SHA256}"
proposal_source_iid="${FITQ_RUNTIME_PROPOSAL_SOURCE_IID:?set FITQ_RUNTIME_PROPOSAL_SOURCE_IID}"
candidate_clean_latent="${FITQ_RUNTIME_CANDIDATE_CLEAN_LATENT:?set FITQ_RUNTIME_CANDIDATE_CLEAN_LATENT}"
candidate_clean_latent_sha256="${FITQ_RUNTIME_CANDIDATE_CLEAN_LATENT_SHA256:?set FITQ_RUNTIME_CANDIDATE_CLEAN_LATENT_SHA256}"
correct_source_clean_latent="${FITQ_RUNTIME_CORRECT_SOURCE_CLEAN_LATENT:?set FITQ_RUNTIME_CORRECT_SOURCE_CLEAN_LATENT}"
correct_source_clean_latent_sha256="${FITQ_RUNTIME_CORRECT_SOURCE_CLEAN_LATENT_SHA256:?set FITQ_RUNTIME_CORRECT_SOURCE_CLEAN_LATENT_SHA256}"
candidate_arm="${FITQ_RUNTIME_CANDIDATE_ARM:?set FITQ_RUNTIME_CANDIDATE_ARM}"
candidate_provenance_receipt="${FITQ_RUNTIME_CANDIDATE_PROVENANCE_RECEIPT:?set FITQ_RUNTIME_CANDIDATE_PROVENANCE_RECEIPT}"
candidate_provenance_receipt_sha256="${FITQ_RUNTIME_CANDIDATE_PROVENANCE_RECEIPT_SHA256:?set FITQ_RUNTIME_CANDIDATE_PROVENANCE_RECEIPT_SHA256}"
source_provenance_receipt="${FITQ_RUNTIME_SOURCE_PROVENANCE_RECEIPT:?set FITQ_RUNTIME_SOURCE_PROVENANCE_RECEIPT}"
source_provenance_receipt_sha256="${FITQ_RUNTIME_SOURCE_PROVENANCE_RECEIPT_SHA256:?set FITQ_RUNTIME_SOURCE_PROVENANCE_RECEIPT_SHA256}"
proposal_source_video_sha256="${FITQ_RUNTIME_PROPOSAL_SOURCE_VIDEO_SHA256:?set FITQ_RUNTIME_PROPOSAL_SOURCE_VIDEO_SHA256}"
wrong_source_row_index="${FITQ_RUNTIME_WRONG_SOURCE_ROW_INDEX:?set FITQ_RUNTIME_WRONG_SOURCE_ROW_INDEX}"
wrong_source_iid="${FITQ_RUNTIME_EXPECTED_WRONG_SOURCE_IID:?set FITQ_RUNTIME_EXPECTED_WRONG_SOURCE_IID}"
wrong_source_clean_latent="${FITQ_RUNTIME_WRONG_SOURCE_CLEAN_LATENT:?set FITQ_RUNTIME_WRONG_SOURCE_CLEAN_LATENT}"
wrong_source_clean_latent_sha256="${FITQ_RUNTIME_WRONG_SOURCE_CLEAN_LATENT_SHA256:?set FITQ_RUNTIME_WRONG_SOURCE_CLEAN_LATENT_SHA256}"
wrong_source_provenance_receipt="${FITQ_RUNTIME_WRONG_SOURCE_PROVENANCE_RECEIPT:?set FITQ_RUNTIME_WRONG_SOURCE_PROVENANCE_RECEIPT}"
wrong_source_provenance_receipt_sha256="${FITQ_RUNTIME_WRONG_SOURCE_PROVENANCE_RECEIPT_SHA256:?set FITQ_RUNTIME_WRONG_SOURCE_PROVENANCE_RECEIPT_SHA256}"
wrong_source_video_sha256="${FITQ_RUNTIME_WRONG_SOURCE_VIDEO_SHA256:?set FITQ_RUNTIME_WRONG_SOURCE_VIDEO_SHA256}"
wrong_source_match="${FITQ_RUNTIME_WRONG_SOURCE_MATCH_JSON:?set FITQ_RUNTIME_WRONG_SOURCE_MATCH_JSON}"
wrong_source_match_sha256="${FITQ_RUNTIME_WRONG_SOURCE_MATCH_SHA256:?set FITQ_RUNTIME_WRONG_SOURCE_MATCH_SHA256}"
action_instruction="${FITQ_RUNTIME_ACTION_INSTRUCTION:?set FITQ_RUNTIME_ACTION_INSTRUCTION}"
action_instruction_sha256="${FITQ_RUNTIME_ACTION_INSTRUCTION_SHA256:?set FITQ_RUNTIME_ACTION_INSTRUCTION_SHA256}"
noop_instruction="${FITQ_RUNTIME_NOOP_INSTRUCTION:?set FITQ_RUNTIME_NOOP_INSTRUCTION}"
noop_instruction_sha256="${FITQ_RUNTIME_NOOP_INSTRUCTION_SHA256:?set FITQ_RUNTIME_NOOP_INSTRUCTION_SHA256}"
hard_negative_manifest="${FITQ_RUNTIME_HARD_NEGATIVE_MANIFEST:?set FITQ_RUNTIME_HARD_NEGATIVE_MANIFEST}"
hard_negative_manifest_sha256="${FITQ_RUNTIME_HARD_NEGATIVE_MANIFEST_SHA256:?set FITQ_RUNTIME_HARD_NEGATIVE_MANIFEST_SHA256}"
output_dir="${FITQ_RUNTIME_OUTPUT_DIR:?set FITQ_RUNTIME_OUTPUT_DIR}"
python_bin="${FITQ_RUNTIME_PYTHON_BIN:?set FITQ_RUNTIME_PYTHON_BIN}"

pinned_bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
pinned_veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
pinned_checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
pinned_checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
checkpoint_tree_sha256="${FITQ_RUNTIME_CHECKPOINT_TREE_SHA256:-${pinned_checkpoint_tree_sha256}}"
checkpoint_manifest_sha256="${FITQ_RUNTIME_CHECKPOINT_CONTENT_MANIFEST_SHA256:-${pinned_checkpoint_manifest_sha256}}"
allow_incomplete_dataset="${FITQ_RUNTIME_ALLOW_INCOMPLETE_DATASET:-0}"
noise_seed_a="${FITQ_RUNTIME_NOISE_SEED_A:-20260808}"
noise_seed_b="${FITQ_RUNTIME_NOISE_SEED_B:-20260809}"
sigmas_text="${FITQ_RUNTIME_SIGMAS:-0.80 0.60 0.35 0.15}"
bridge_fractions_text="${FITQ_RUNTIME_BRIDGE_FRACTIONS:-1 0.5 0}"

fail() {
  echo "[bernini-fitq-dp2sp4] ERROR: $*" >&2
  exit 2
}

require_absolute() {
  local value="$1"
  local label="$2"
  [[ "${value}" == /* ]] || fail "${label} must be absolute"
}

require_sha256() {
  local value="$1"
  local label="$2"
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || fail "${label} must be lowercase SHA-256"
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
  source_archive_sha256 dataset_summary_sha256 checkpoint_tree_sha256 \
  checkpoint_manifest_sha256 candidate_clean_latent_sha256 \
  correct_source_clean_latent_sha256 candidate_provenance_receipt_sha256 \
  source_provenance_receipt_sha256 proposal_source_video_sha256 \
  wrong_source_clean_latent_sha256 wrong_source_provenance_receipt_sha256 \
  wrong_source_video_sha256 wrong_source_match_sha256 \
  action_instruction_sha256 noop_instruction_sha256 hard_negative_manifest_sha256
do
  require_sha256 "${!digest_name}" "${digest_name}"
done

[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be full lowercase SHA-1"
[[ "${checkpoint_tree_sha256}" == "${pinned_checkpoint_tree_sha256}" ]] || fail "checkpoint tree is not the pinned release"
[[ "${checkpoint_manifest_sha256}" == "${pinned_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest is not pinned"
[[ "${wrong_source_row_index}" =~ ^[0-9]+$ ]] || fail "wrong-source row index must be nonnegative"
[[ "${candidate_arm}" == t2v || "${candidate_arm}" == r2v || "${candidate_arm}" == rv2v ]] || fail "candidate arm is invalid"
[[ "${allow_incomplete_dataset}" == 0 || "${allow_incomplete_dataset}" == 1 ]] || fail "allow-incomplete flag must be 0 or 1"
[[ "${noise_seed_a}" =~ ^[0-9]+$ && "${noise_seed_a}" -lt 2147483648 ]] || fail "seed A is invalid"
[[ "${noise_seed_b}" =~ ^[0-9]+$ && "${noise_seed_b}" -lt 2147483648 ]] || fail "seed B is invalid"
[[ "${noise_seed_a}" != "${noise_seed_b}" ]] || fail "independent groups require different seeds"
[[ -n "${action_instruction//[[:space:]]/}" && -n "${noop_instruction//[[:space:]]/}" ]] || fail "instructions must be nonempty"
[[ "${action_instruction}" != "${noop_instruction}" ]] || fail "action and no-op must differ"

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
launcher_source_sha256="$(sha256sum "${launcher_source}" | awk '{print $1}')"
require_sha256 "${launcher_source_sha256}" "launcher SHA-256"

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive must be a plain file"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini root is invalid"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni root is invalid"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint root is invalid"
[[ -d "${preprocessed_parquet_dir}" && ! -L "${preprocessed_parquet_dir}" ]] || fail "dataset root is invalid"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "Python executable is invalid"

declare -a immutable_files=(
  "${launcher_source}|${launcher_source_sha256}|launcher"
  "${source_archive}|${source_archive_sha256}|source archive"
  "${checkpoint_manifest}|${checkpoint_manifest_sha256}|checkpoint manifest"
  "${dataset_summary}|${dataset_summary_sha256}|dataset summary"
  "${candidate_clean_latent}|${candidate_clean_latent_sha256}|candidate latent"
  "${correct_source_clean_latent}|${correct_source_clean_latent_sha256}|source latent"
  "${candidate_provenance_receipt}|${candidate_provenance_receipt_sha256}|candidate receipt"
  "${source_provenance_receipt}|${source_provenance_receipt_sha256}|source receipt"
  "${wrong_source_clean_latent}|${wrong_source_clean_latent_sha256}|wrong-source latent"
  "${wrong_source_provenance_receipt}|${wrong_source_provenance_receipt_sha256}|wrong-source receipt"
  "${wrong_source_match}|${wrong_source_match_sha256}|wrong-source match"
  "${hard_negative_manifest}|${hard_negative_manifest_sha256}|hard-negative manifest"
)
for entry in "${immutable_files[@]}"; do
  path="${entry%%|*}"
  remainder="${entry#*|}"
  expected="${remainder%%|*}"
  label="${remainder#*|}"
  [[ -f "${path}" && ! -L "${path}" ]] || fail "${label} is not a plain file"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || fail "${label} hash differs"
done

actual_archive_revision="$(git get-tar-commit-id <"${source_archive}")" || fail "source must be a git archive"
[[ "${actual_archive_revision}" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive changed"

actual_action_sha256="$("${python_bin}" -B -c 'import hashlib,sys; x=sys.argv[1]; assert x and x==x.strip() and "\0" not in x; print(hashlib.sha256(x.encode()).hexdigest())' "${action_instruction}")"
actual_noop_sha256="$("${python_bin}" -B -c 'import hashlib,sys; x=sys.argv[1]; assert x and x==x.strip() and "\0" not in x; print(hashlib.sha256(x.encode()).hexdigest())' "${noop_instruction}")"
[[ "${actual_action_sha256}" == "${action_instruction_sha256}" ]] || fail "action hash differs"
[[ "${actual_noop_sha256}" == "${noop_instruction_sha256}" ]] || fail "no-op hash differs"

"${python_bin}" -B - "${hard_negative_manifest}" "${action_instruction_sha256}" "${noop_instruction_sha256}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1]).resolve(strict=True)
action_sha, noop_sha = sys.argv[2:]
value = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(value, dict) or value.get("schema_version") != "bernini-iar-hard-negative-manifest-v1":
    raise SystemExit("FITQ hard-negative manifest schema differs")
if value.get("action_instruction_sha256") != action_sha:
    raise SystemExit("FITQ hard-negative manifest action binding differs")
if value.get("declared_use") != "engineering_plumbing_only":
    raise SystemExit("FITQ hard negatives must remain engineering-only")
rows = value.get("hard_negatives")
if not isinstance(rows, list) or len(rows) != 2:
    raise SystemExit("FITQ A0 requires exact K=2 hard negatives")
ids = [row.get("condition_id") for row in rows if isinstance(row, dict)]
if ids != ["semantic_noop", "incomplete_approach_without_pickup"]:
    raise SystemExit("FITQ A0 requires ordered no-op and incomplete controls")
digests = []
for row in rows:
    instruction = row.get("instruction")
    digest = row.get("instruction_sha256")
    if not isinstance(instruction, str) or not instruction or instruction != instruction.strip():
        raise SystemExit("FITQ hard-negative instruction is invalid")
    if digest != hashlib.sha256(instruction.encode("utf-8")).hexdigest():
        raise SystemExit("FITQ hard-negative instruction hash differs")
    digests.append(digest)
if digests[0] != noop_sha or action_sha in digests or len(set(digests)) != 2:
    raise SystemExit("FITQ no-op/incomplete semantic binding differs")
PY

read -r -a sigma_values <<<"${sigmas_text}"
read -r -a bridge_fraction_values <<<"${bridge_fractions_text}"
"${python_bin}" -B - "${#sigma_values[@]}" "${sigma_values[@]}" -- "${#bridge_fraction_values[@]}" "${bridge_fraction_values[@]}" <<'PY'
import struct
import sys

separator = sys.argv.index("--")
sigma_count = int(sys.argv[1])
sigmas = [float(item) for item in sys.argv[2:separator]]
lambda_count = int(sys.argv[separator + 1])
lambdas = [float(item) for item in sys.argv[separator + 2:]]
f32 = lambda x: struct.pack("!f", x)
if sigma_count != 4 or len(sigmas) != 4 or [f32(x) for x in sigmas] != [f32(x) for x in (0.80, 0.60, 0.35, 0.15)]:
    raise SystemExit("FITQ requires exact S=4 sigmas 0.80,0.60,0.35,0.15")
if lambda_count != 3 or len(lambdas) != 3 or [f32(x) for x in lambdas] != [f32(x) for x in (1.0, 0.5, 0.0)]:
    raise SystemExit("FITQ requires exact L=3 lambdas 1,0.5,0")
PY

[[ "${output_dir}" != / ]] || fail "output cannot be filesystem root"
[[ "${output_dir##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename is unsafe"
[[ "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output path must be canonical"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing to reuse output"
output_parent="$(dirname -- "${output_dir}")"
mkdir -p -- "${output_parent}"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent is invalid"

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

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID is invalid"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent is invalid"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-fitq-dp2sp4-${SLURM_JOB_ID}.XXXXXX")"
group_a_pid=""
group_b_pid=""

cleanup() {
  local exit_status=$?
  trap - EXIT TERM INT
  for child_pid in "${group_a_pid:-}" "${group_b_pid:-}"; do
    if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
      kill -TERM "${child_pid}" 2>/dev/null || true
    fi
  done
  case "${task_scratch:-}" in
    "${scratch_parent%/}/bernini-fitq-dp2sp4-${SLURM_JOB_ID}."*) rm -rf -- "${task_scratch}" ;;
    *) echo "[bernini-fitq-dp2sp4] refusing unsafe scratch cleanup" >&2; exit 97 ;;
  esac
  exit "${exit_status}"
}
trap cleanup EXIT TERM INT

mkdir -- "${task_scratch}/source-tree" "${task_scratch}/group-a" "${task_scratch}/group-b"
mkdir -p -- \
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

archive_copy="${task_scratch}/source.tar"
cp -- "${source_archive}" "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged source hash differs"
"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

with tarfile.open(sys.argv[1], "r:*") as archive:
    names = []
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isfifo() or member.isdev():
            raise SystemExit("source archive contains an unsafe member")
        names.append(member.name)
required = {
    "methods/bernini_action_editing/infer_fitq_official_runtime_scan.py",
    "methods/bernini_action_editing/internal_temporal_quotient_observer.py",
    "methods/bernini_action_editing/infer_iar_official_runtime_smoke.py",
    "methods/bernini_action_editing/scripts/auh_infer_fitq_official_runtime_scan.sbatch",
}
if not required.issubset(names):
    raise SystemExit("source archive lacks FITQ official runtime closure")
PY
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
for required in \
  infer_fitq_official_runtime_scan.py internal_temporal_quotient_observer.py \
  infer_iar_official_runtime_smoke.py identity_anchored_action_residual.py \
  infer_dclr_reward_runtime_smoke.py dual_conditional_ratio_core.py \
  dclr_runtime_contract.py infer_source_kv_carrier_oracle.py infer_lora.py \
  train_lora.py motion_residual.py source_kv_replay.py source_kv_route_batches.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "missing staged ${required}"
done
archived_launcher="${method_root}/scripts/auh_infer_fitq_official_runtime_scan.sbatch"
[[ "$(sha256sum "${archived_launcher}" | awk '{print $1}')" == "${launcher_source_sha256}" ]] || fail "executing launcher differs from archive"
find "${method_root}" -type f -exec chmod a-w -- {} +

allocated_rocr_devices="${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
[[ "${allocated_rocr_devices}" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || fail "DP2 x WORLD4 FITQ requires exactly eight numeric ROCr devices"
IFS=',' read -r -a allocated_devices <<<"${allocated_rocr_devices}"
[[ "${#allocated_devices[@]}" == 8 ]] || fail "device count differs"
[[ "$(printf '%s\n' "${allocated_devices[@]}" | sort -u | wc -l)" == 8 ]] || fail "ROCr devices are not distinct"
printf -v group_a_devices '%s,' "${allocated_devices[@]:0:4}"
group_a_devices="${group_a_devices%,}"
printf -v group_b_devices '%s,' "${allocated_devices[@]:4:4}"
group_b_devices="${group_b_devices%,}"

mkdir -- "${output_dir}"
mkdir -- "${output_dir}/seed-a" "${output_dir}/seed-b"
receipt_path_a="${output_dir}/seed-a/receipt.json"
receipt_path_b="${output_dir}/seed-b/receipt.json"
statistics_dir_a="${output_dir}/seed-a/statistics"
statistics_dir_b="${output_dir}/seed-b/statistics"
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
  local statistics_dir="$6"
  local torchrun_log="$7"
  local scratch_root="$8"
  (
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
    echo "[bernini-fitq-dp2sp4] group=${group_label} topology=WORLD4/Ulysses4 gpus=4 seed=${group_seed} exact81=true grid_forwards=84 duplicate_forwards=1 hook_off_references=1 actual_forwards_per_rank=86 forward_hooks_present=true training=false optimizer=false backward=false checkpoint_save=false"
    "${python_bin}" -B -m torch.distributed.run \
      --nproc_per_node=4 \
      --master_addr=127.0.0.1 \
      --master_port="${master_port}" \
      "${method_root}/infer_fitq_official_runtime_scan.py" \
      "${runner_args[@]}" \
      --noise-seed "${group_seed}" \
      --output-receipt "${receipt_path}" \
      --output-statistics-dir "${statistics_dir}" >"${torchrun_log}" 2>&1
  )
}

echo "[bernini-fitq-dp2sp4] topology=DP2xWORLD4/Ulysses4 allocated_gpus=8 groups=2 exact81=true actual_forwards_per_rank_per_group=86 fitq_go_authorized=false optimizer_update=null"
launch_group seed-a "${group_a_devices}" "${master_port_a}" "${noise_seed_a}" "${receipt_path_a}" "${statistics_dir_a}" "${torchrun_log_a}" "${task_scratch}/group-a" &
group_a_pid=$!
launch_group seed-b "${group_b_devices}" "${master_port_b}" "${noise_seed_b}" "${receipt_path_b}" "${statistics_dir_b}" "${torchrun_log_b}" "${task_scratch}/group-b" &
group_b_pid=$!

if wait "${group_a_pid}"; then group_a_status=0; else group_a_status=$?; kill -TERM "${group_b_pid}" 2>/dev/null || true; fi
group_a_pid=""
if wait "${group_b_pid}"; then group_b_status=0; else group_b_status=$?; fi
group_b_pid=""
if [[ "${group_a_status}" != 0 || "${group_b_status}" != 0 ]]; then
  tail -n 240 -- "${torchrun_log_a}" >&2 || true
  tail -n 240 -- "${torchrun_log_b}" >&2 || true
  fail "one or both FITQ WORLD4 groups failed"
fi

audit_receipt() {
  local receipt_path="$1"
  local statistics_dir="$2"
  local expected_seed="$3"
  "${python_bin}" -B - "${receipt_path}" "${statistics_dir}" "${expected_seed}" "${launcher_source_sha256}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

receipt_path = Path(sys.argv[1]).resolve(strict=True)
statistics_dir = Path(sys.argv[2]).resolve(strict=True)
expected_seed = int(sys.argv[3])
launcher_sha = sys.argv[4]

def require(condition, message):
    if not condition:
        raise SystemExit(message)

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()

def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

require(receipt_path.is_file() and not receipt_path.is_symlink(), "FITQ receipt is not plain")
require(statistics_dir.is_dir() and not statistics_dir.is_symlink(), "FITQ statistics directory is invalid")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
declared = receipt.pop("receipt_digest", None)
require(canonical_sha(receipt) == declared, "FITQ receipt digest differs")
receipt["receipt_digest"] = declared
require(receipt.get("schema_version") == "bernini-fitq-official-runtime-scan-receipt-v1", "FITQ schema differs")
require(receipt.get("method") == "bernini-fitq-official-runtime-scan-v1", "FITQ method differs")
require(receipt.get("exact81") is True, "FITQ runtime is not exact81")
require(receipt.get("read_only_forward_hooks_present") is True, "FITQ hook provenance is absent")
require(receipt.get("forward_callback_present") is True, "FITQ callback provenance is false")
require(receipt.get("custom_forward_core_present") is False, "custom forward core is forbidden")
require(receipt.get("custom_analysis_core_present") is True, "analysis core is absent")
require(receipt.get("official_field_output_modified") is False, "observer modified a field")
require(receipt.get("training_authorized") is False and receipt.get("fitq_stage1_authorized") is False, "FITQ incorrectly authorizes training")
require(receipt.get("optimizer_update") == "null", "optimizer update is not null")
training = receipt.get("training")
require(isinstance(training, dict) and training == {
    "forward_only": True,
    "backward_performed": False,
    "optimizer_present": False,
    "checkpoint_saved": False,
    "adapter_present": False,
    "model_weights_written": False,
    "analysis_statistics_written": True,
}, "training non-authorization differs")
distributed = receipt.get("distributed")
require(isinstance(distributed, dict) and distributed.get("world_size") == 4 and distributed.get("ulysses_size") == 4, "not WORLD4/Ulysses4")
rank_rows = distributed.get("rank_fitq_records")
require(isinstance(rank_rows, list) and [row.get("rank") for row in rank_rows] == [0, 1, 2, 3], "FITQ rank closure differs")
actual_runtime_digest = distributed.get("actual_runtime_evidence_digest")
for row in rank_rows:
    cells = row.get("same_state_block0_parity")
    require(isinstance(cells, list) and len(cells) == 12, "rank parity cell count differs")
    require(all(cell.get("all_pairs_byte_exact") is True for cell in cells), "same-state action/no-op/incomplete parity failed")
    require(row.get("actual_runtime_evidence_digest") == actual_runtime_digest, "rank truthful-runtime digest differs")

shadow = receipt.get("underlying_iar_schema_validation_shadow")
require(isinstance(shadow, dict) and shadow.get("hypothetical_no_hook_schema_check_only") is True, "IAR schema-validation shadow is mislabeled")
require(shadow.get("is_runtime_provenance") is False and shadow.get("emitted_as_receipt") is False, "IAR shadow was presented as runtime provenance")

field = receipt.get("official_field_evidence")
require(isinstance(field, dict) and field.get("method") == "bernini-fitq-official-runtime-scan-v1", "field evidence differs")
require(field.get("launcher_source_sha256") == launcher_sha, "launcher binding differs")
require(field.get("num_frames") == 81 and field.get("latent_phases") == 21, "field geometry differs")
require(field.get("field_grid_forwards_per_rank") == 84, "grid forward count differs")
require(field.get("hooked_action_duplicate_forwards_per_rank") == 1, "duplicate count differs")
require(field.get("hook_off_reference_forwards_per_rank") == 1, "reference count differs")
require(field.get("forwards_per_rank") == 86, "actual forward count differs")
require(field.get("forward_callback_present") is True and field.get("read_only_forward_hooks_present") is True, "field hook provenance differs")
require(field.get("custom_forward_core_present") is False and field.get("custom_analysis_core_present") is True, "field core provenance differs")
require(field.get("underlying_iar_shadow_is_hypothetical_not_runtime_provenance") is True, "field shadow label differs")
homotopy = field.get("homotopy")
require(isinstance(homotopy, dict) and homotopy.get("noise_seed") == expected_seed, "noise seed differs")

observation = receipt.get("fitq_observation")
require(isinstance(observation, dict), "FITQ observation is absent")
require(canonical_sha(field) == distributed.get("truthful_field_evidence_digest"), "truthful field evidence digest differs")
require(canonical_sha({"official_field_evidence": field, "fitq_observation": observation}) == actual_runtime_digest, "actual hooked-runtime evidence digest differs")
require(observation.get("analysis_statistics") == "phase_head_mean_second_moment", "analysis statistic differs")
require(observation.get("field_grid_forwards_per_rank") == 84, "observation grid count differs")
require(observation.get("hooked_action_duplicate_forwards_per_rank") == 1, "observation duplicate count differs")
require(observation.get("observed_hooked_forwards_per_rank") == 85, "observed forward count differs")
require(observation.get("hook_off_reference_forwards_per_rank") == 1, "hook-off count differs")
require(observation.get("total_official_forwards_per_rank") == 86, "total forward count differs")
require(observation.get("hook_on_off_field_parity", {}).get("byte_exact_equal") is True, "hook on/off field parity failed")
duplicate = observation.get("action_duplicate_numerical_floor")
require(isinstance(duplicate, dict) and duplicate.get("block0_same_state_exact", {}).get("all") is True, "action duplicate parity failed")
require(observation.get("proposal_bank_size") == 1 and observation.get("proposal_bank_status") == "insufficient_bank", "single-proposal status differs")
require(observation.get("decision_scope") == "engineering_N0_like_diagnostic_only", "decision scope differs")
require(observation.get("scientific_fitq_outcome") == "not_evaluated_single_proposal", "single-proposal scientific outcome differs")
require(observation.get("cross_mode_transport_gate_evaluated") is False and observation.get("causal_intervention_gate_evaluated") is False, "A0 overclaims later FITQ gates")
require(observation.get("fitq_go_authorized") is False, "FITQ GO is forbidden")
require(observation.get("tokenwise_localization_available") is False, "tokenwise claim is invalid")

artifacts = observation.get("statistics_artifacts")
contexts = observation.get("contexts")
require(isinstance(artifacts, list) and len(artifacts) == 85, "artifact count differs")
require(isinstance(contexts, list) and len(contexts) == 85, "context count differs")
require(sum(context.get("branch") == "frozen_t2v_action_duplicate" for context in contexts) == 1, "duplicate context differs")
actual_files = {path.resolve() for path in statistics_dir.glob("*.fitq-stats.pt") if path.is_file() and not path.is_symlink()}
declared_files = set()
for ordinal, artifact in enumerate(artifacts):
    require(artifact.get("ordinal") == ordinal, "artifact ordinal differs")
    path = Path(artifact.get("path", "")).resolve(strict=True)
    require(path.parent == statistics_dir and path.is_file() and not path.is_symlink(), "artifact escaped statistics directory")
    require(file_sha(path) == artifact.get("sha256"), "artifact SHA-256 differs")
    require(path.stat().st_size == artifact.get("size_bytes"), "artifact size differs")
    require(artifact.get("schema_version") == "bernini-fitq-phase-head-mean-second-moment-v1", "artifact schema differs")
    require(artifact.get("mean_shape") == [121, 1, 21, 1, 1536], "mean shape differs")
    require(artifact.get("second_moment_shape") == [121, 1, 21, 1, 1536], "second moment shape differs")
    require(artifact.get("count_shape") == [21, 12] and artifact.get("dtype") == "torch.float32", "artifact dtype/count differs")
    require(artifact.get("contains_model_weights") is False and artifact.get("is_checkpoint") is False, "statistics mislabeled as weights")
    declared_files.add(path)
require(actual_files == declared_files, "statistics directory file closure differs")
print("BERNINI_FITQ_WORLD4_READ_ONLY_AUDIT_OK training_authorized=false fitq_go_authorized=false optimizer_update=null")
PY
}

audit_receipt "${receipt_path_a}" "${statistics_dir_a}" "${noise_seed_a}"
audit_receipt "${receipt_path_b}" "${statistics_dir_b}" "${noise_seed_b}"

# Close the claim that the read-only scan did not change any checkpoint byte.
"${python_bin}" -B - "${checkpoint}" "${checkpoint_manifest}" "${checkpoint_manifest_sha256}" "${receipt_path_a}" "${receipt_path_b}" <<'PY'
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys

checkpoint_value, manifest_value, manifest_sha, receipt_a_value, receipt_b_value = sys.argv[1:]

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
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()

checkpoint = Path(checkpoint_value).resolve(strict=True)
manifest = Path(manifest_value).resolve(strict=True)
require(checkpoint.is_dir() and not checkpoint.is_symlink(), "post-FITQ checkpoint root is invalid")
require(manifest.is_file() and not manifest.is_symlink(), "post-FITQ checkpoint manifest is invalid")
require(file_sha(manifest) == manifest_sha, "post-FITQ checkpoint manifest changed")
lines = manifest.read_text(encoding="utf-8").splitlines()
require(len(lines) == 23, "post-FITQ checkpoint manifest count differs")
line_pattern = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)")
expected = {}
for line in lines:
    match = line_pattern.fullmatch(line)
    require(match is not None, "checkpoint manifest line is non-canonical")
    digest, raw_path = match.groups()
    relative = PurePosixPath(raw_path)
    require(not relative.is_absolute() and ".." not in relative.parts, "checkpoint manifest path is unsafe")
    normalized = PurePosixPath(*(part for part in relative.parts if part not in ("", "."))).as_posix()
    require(normalized and normalized not in expected, "checkpoint manifest path is empty or duplicate")
    expected[normalized] = digest
actual = set()
for path in checkpoint.rglob("*"):
    relative = path.relative_to(checkpoint)
    if ".cache" in relative.parts:
        continue
    mode = path.lstat().st_mode
    require(not stat.S_ISLNK(mode), "checkpoint contains a non-cache symlink")
    if stat.S_ISREG(mode):
        actual.add(relative.as_posix())
    else:
        require(stat.S_ISDIR(mode), "checkpoint contains a non-regular entry")
require(actual == set(expected), "post-FITQ checkpoint file set differs")
verified = []
for relative in sorted(expected):
    path = checkpoint / relative
    require(path.is_file() and not path.is_symlink(), "checkpoint file is not plain")
    digest = file_sha(path)
    require(digest == expected[relative], "post-FITQ checkpoint content differs: {}".format(relative))
    verified.append({"path": relative, "sha256": digest})
post_checkpoint_identity = {
    "manifest_path": str(manifest),
    "manifest_sha256_computed": manifest_sha,
    "manifest_sha256_expected": manifest_sha,
    "verified_file_count": len(verified),
    "every_file_sha256_verified": True,
    "verified_entries_digest": object_sha(verified),
}
for receipt_value in (receipt_a_value, receipt_b_value):
    receipt = json.loads(Path(receipt_value).read_text(encoding="utf-8"))
    field = receipt.get("official_field_evidence")
    require(isinstance(field, dict), "FITQ receipt field evidence is absent")
    require(field.get("checkpoint_content_identity") == post_checkpoint_identity, "post-FITQ checkpoint identity differs from load-time receipt")
print("BERNINI_FITQ_POST_CHECKPOINT_BYTE_IDENTITY_OK model_weights_written=false")
PY

for entry in "${immutable_files[@]}"; do
  path="${entry%%|*}"
  remainder="${entry#*|}"
  expected="${remainder%%|*}"
  label="${remainder#*|}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || fail "${label} changed after runtime"
done

summary_path="${output_dir}/dual-group-summary.json"
"${python_bin}" -B - "${receipt_path_a}" "${receipt_path_b}" "${summary_path}" "${noise_seed_a}" "${noise_seed_b}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

receipt_a, receipt_b, output, seed_a, seed_b = sys.argv[1:]
def file_sha(value):
    return hashlib.sha256(Path(value).read_bytes()).hexdigest()
payload = {
    "schema_version": "bernini-fitq-dp2sp4-dual-group-summary-v1",
    "topology": "DP2xWORLD4/Ulysses4",
    "allocated_gpus": 8,
    "groups": [
        {"label": "seed-a", "noise_seed": int(seed_a), "receipt": receipt_a, "receipt_sha256": file_sha(receipt_a)},
        {"label": "seed-b", "noise_seed": int(seed_b), "receipt": receipt_b, "receipt_sha256": file_sha(receipt_b)},
    ],
    "exact81": True,
    "actual_forwards_per_rank_per_group": 86,
    "proposal_bank_status": "insufficient_bank",
    "scientific_fitq_outcome": "not_evaluated_single_proposal_per_group",
    "training_authorized": False,
    "fitq_stage1_authorized": False,
    "optimizer_update": "null",
    "scientific_claim_authorized": False,
}
raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
temporary = Path(output).with_name(".dual-group-summary.tmp-{}".format(os.getpid()))
temporary.write_bytes(raw + b"\n")
os.replace(str(temporary), output)
PY

chmod 0444 -- "${receipt_path_a}" "${receipt_path_b}" "${torchrun_log_a}" "${torchrun_log_b}" "${summary_path}"
find "${statistics_dir_a}" "${statistics_dir_b}" -type f -name '*.fitq-stats.pt' -exec chmod 0444 -- {} +
echo "[bernini-fitq-dp2sp4] complete output=${output_dir} exact81=true topology=DP2xWORLD4/Ulysses4 training_authorized=false fitq_go_authorized=false optimizer_update=null"
