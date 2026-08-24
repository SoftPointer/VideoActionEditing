#!/usr/bin/env bash
#SBATCH --job-name=starc-vjp-2xsp4
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=06:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# Two independent, frozen, current-candidate input-VJP probes use all eight
# allocated MI210s concurrently.  Each WORLD4 owns one candidate and writes one
# authenticated composite receipt.  This launcher never trains or updates an
# editor and never authorizes a scientific/action-editing claim.

source_archive="${STARC_LIVE_VJP_SOURCE_ARCHIVE:?set STARC_LIVE_VJP_SOURCE_ARCHIVE}"
source_archive_sha256="${STARC_LIVE_VJP_SOURCE_ARCHIVE_SHA256:?set STARC_LIVE_VJP_SOURCE_ARCHIVE_SHA256}"
source_revision="${STARC_LIVE_VJP_SOURCE_REVISION:?set STARC_LIVE_VJP_SOURCE_REVISION}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
materializer_master="${STARC_LIVE_VJP_MATERIALIZER_MASTER:?set STARC_LIVE_VJP_MATERIALIZER_MASTER}"
materializer_master_sha256="${STARC_LIVE_VJP_MATERIALIZER_MASTER_SHA256:?set STARC_LIVE_VJP_MATERIALIZER_MASTER_SHA256}"
critic_checkpoint="${STARC_LIVE_VJP_CRITIC_CHECKPOINT:?set STARC_LIVE_VJP_CRITIC_CHECKPOINT}"
critic_checkpoint_sha256="${STARC_LIVE_VJP_CRITIC_CHECKPOINT_SHA256:?set STARC_LIVE_VJP_CRITIC_CHECKPOINT_SHA256}"
critic_checkpoint_receipt="${STARC_LIVE_VJP_CRITIC_CHECKPOINT_RECEIPT:?set STARC_LIVE_VJP_CRITIC_CHECKPOINT_RECEIPT}"
critic_checkpoint_receipt_sha256="${STARC_LIVE_VJP_CRITIC_CHECKPOINT_RECEIPT_SHA256:?set STARC_LIVE_VJP_CRITIC_CHECKPOINT_RECEIPT_SHA256}"
critic_config_receipt="${STARC_LIVE_VJP_CRITIC_CONFIG_RECEIPT:?set STARC_LIVE_VJP_CRITIC_CONFIG_RECEIPT}"
critic_config_receipt_sha256="${STARC_LIVE_VJP_CRITIC_CONFIG_RECEIPT_SHA256:?set STARC_LIVE_VJP_CRITIC_CONFIG_RECEIPT_SHA256}"
output_root="${STARC_LIVE_VJP_OUTPUT_ROOT:?set STARC_LIVE_VJP_OUTPUT_ROOT}"
python_bin="${STARC_LIVE_VJP_PYTHON_BIN:?set STARC_LIVE_VJP_PYTHON_BIN}"

candidate_a_manifest="${STARC_LIVE_VJP_A_CANDIDATE_MANIFEST:?set STARC_LIVE_VJP_A_CANDIDATE_MANIFEST}"
candidate_a_manifest_sha256="${STARC_LIVE_VJP_A_CANDIDATE_MANIFEST_SHA256:?set STARC_LIVE_VJP_A_CANDIDATE_MANIFEST_SHA256}"
candidate_a_source="${STARC_LIVE_VJP_A_SOURCE_VIDEO:?set STARC_LIVE_VJP_A_SOURCE_VIDEO}"
candidate_a_source_sha256="${STARC_LIVE_VJP_A_SOURCE_VIDEO_SHA256:?set STARC_LIVE_VJP_A_SOURCE_VIDEO_SHA256}"
candidate_a_instruction="${STARC_LIVE_VJP_A_INSTRUCTION_FILE:?set STARC_LIVE_VJP_A_INSTRUCTION_FILE}"
candidate_a_instruction_sha256="${STARC_LIVE_VJP_A_INSTRUCTION_FILE_SHA256:?set STARC_LIVE_VJP_A_INSTRUCTION_FILE_SHA256}"
candidate_a_noop="${STARC_LIVE_VJP_A_NOOP_CAPTION_FILE:?set STARC_LIVE_VJP_A_NOOP_CAPTION_FILE}"
candidate_a_noop_sha256="${STARC_LIVE_VJP_A_NOOP_CAPTION_FILE_SHA256:?set STARC_LIVE_VJP_A_NOOP_CAPTION_FILE_SHA256}"
candidate_a_clean="${STARC_LIVE_VJP_A_CLEAN_LATENT:?set STARC_LIVE_VJP_A_CLEAN_LATENT}"
candidate_a_clean_sha256="${STARC_LIVE_VJP_A_CLEAN_LATENT_SHA256:?set STARC_LIVE_VJP_A_CLEAN_LATENT_SHA256}"
candidate_a_noise="${STARC_LIVE_VJP_A_NATIVE_NOISE:?set STARC_LIVE_VJP_A_NATIVE_NOISE}"
candidate_a_noise_sha256="${STARC_LIVE_VJP_A_NATIVE_NOISE_SHA256:?set STARC_LIVE_VJP_A_NATIVE_NOISE_SHA256}"

candidate_b_manifest="${STARC_LIVE_VJP_B_CANDIDATE_MANIFEST:?set STARC_LIVE_VJP_B_CANDIDATE_MANIFEST}"
candidate_b_manifest_sha256="${STARC_LIVE_VJP_B_CANDIDATE_MANIFEST_SHA256:?set STARC_LIVE_VJP_B_CANDIDATE_MANIFEST_SHA256}"
candidate_b_source="${STARC_LIVE_VJP_B_SOURCE_VIDEO:?set STARC_LIVE_VJP_B_SOURCE_VIDEO}"
candidate_b_source_sha256="${STARC_LIVE_VJP_B_SOURCE_VIDEO_SHA256:?set STARC_LIVE_VJP_B_SOURCE_VIDEO_SHA256}"
candidate_b_instruction="${STARC_LIVE_VJP_B_INSTRUCTION_FILE:?set STARC_LIVE_VJP_B_INSTRUCTION_FILE}"
candidate_b_instruction_sha256="${STARC_LIVE_VJP_B_INSTRUCTION_FILE_SHA256:?set STARC_LIVE_VJP_B_INSTRUCTION_FILE_SHA256}"
candidate_b_noop="${STARC_LIVE_VJP_B_NOOP_CAPTION_FILE:?set STARC_LIVE_VJP_B_NOOP_CAPTION_FILE}"
candidate_b_noop_sha256="${STARC_LIVE_VJP_B_NOOP_CAPTION_FILE_SHA256:?set STARC_LIVE_VJP_B_NOOP_CAPTION_FILE_SHA256}"
candidate_b_clean="${STARC_LIVE_VJP_B_CLEAN_LATENT:?set STARC_LIVE_VJP_B_CLEAN_LATENT}"
candidate_b_clean_sha256="${STARC_LIVE_VJP_B_CLEAN_LATENT_SHA256:?set STARC_LIVE_VJP_B_CLEAN_LATENT_SHA256}"
candidate_b_noise="${STARC_LIVE_VJP_B_NATIVE_NOISE:?set STARC_LIVE_VJP_B_NATIVE_NOISE}"
candidate_b_noise_sha256="${STARC_LIVE_VJP_B_NATIVE_NOISE_SHA256:?set STARC_LIVE_VJP_B_NATIVE_NOISE_SHA256}"

readonly pinned_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly pinned_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly pinned_checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly pinned_checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831

fail() { echo "[starc-live-vjp] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }

for name in source_archive bernini_root veomni_root checkpoint checkpoint_manifest materializer_master critic_checkpoint critic_checkpoint_receipt critic_config_receipt output_root python_bin candidate_a_manifest candidate_a_source candidate_a_instruction candidate_a_noop candidate_a_clean candidate_a_noise candidate_b_manifest candidate_b_source candidate_b_instruction candidate_b_noop candidate_b_clean candidate_b_noise; do
  require_absolute "${!name}" "${name}"
done
for name in source_archive_sha256 materializer_master_sha256 critic_checkpoint_sha256 critic_checkpoint_receipt_sha256 critic_config_receipt_sha256 candidate_a_manifest_sha256 candidate_a_source_sha256 candidate_a_instruction_sha256 candidate_a_noop_sha256 candidate_a_clean_sha256 candidate_a_noise_sha256 candidate_b_manifest_sha256 candidate_b_source_sha256 candidate_b_instruction_sha256 candidate_b_noop_sha256 candidate_b_clean_sha256 candidate_b_noise_sha256; do
  require_sha256 "${!name}" "${name}"
done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be lowercase 40-hex"

source_archive="$(realpath -e -- "${source_archive}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
materializer_master="$(realpath -e -- "${materializer_master}")"
critic_checkpoint="$(realpath -e -- "${critic_checkpoint}")"
critic_checkpoint_receipt="$(realpath -e -- "${critic_checkpoint_receipt}")"
critic_config_receipt="$(realpath -e -- "${critic_config_receipt}")"
python_bin="$(realpath -e -- "${python_bin}")"
for name in candidate_a_manifest candidate_a_source candidate_a_instruction candidate_a_noop candidate_a_clean candidate_a_noise candidate_b_manifest candidate_b_source candidate_b_instruction candidate_b_noop candidate_b_clean candidate_b_noise; do
  declare "${name}=$(realpath -e -- "${!name}")"
done

for path in "${source_archive}" "${checkpoint_manifest}" "${materializer_master}" "${critic_checkpoint}" "${critic_checkpoint_receipt}" "${critic_config_receipt}" "${candidate_a_manifest}" "${candidate_a_source}" "${candidate_a_instruction}" "${candidate_a_noop}" "${candidate_a_clean}" "${candidate_a_noise}" "${candidate_b_manifest}" "${candidate_b_source}" "${candidate_b_instruction}" "${candidate_b_noop}" "${candidate_b_clean}" "${candidate_b_noise}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "plain input file differs: ${path}"
done
for path in "${bernini_root}" "${veomni_root}" "${checkpoint}"; do
  [[ -d "${path}" && ! -L "${path}" ]] || fail "plain input directory differs: ${path}"
done
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python executable differs"

check_hash() { [[ "$(sha256sum "$1" | awk '{print $1}')" == "$2" ]] || fail "$3 SHA-256 differs"; }
check_hash "${source_archive}" "${source_archive_sha256}" source_archive
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
check_hash "${checkpoint_manifest}" "${pinned_checkpoint_manifest_sha256}" checkpoint_manifest
check_hash "${materializer_master}" "${materializer_master_sha256}" materializer_master
check_hash "${critic_checkpoint}" "${critic_checkpoint_sha256}" critic_checkpoint
check_hash "${critic_checkpoint_receipt}" "${critic_checkpoint_receipt_sha256}" critic_checkpoint_receipt
check_hash "${critic_config_receipt}" "${critic_config_receipt_sha256}" critic_config_receipt
for prefix in candidate_a candidate_b; do
  for suffix in manifest source instruction noop clean noise; do
    value_name="${prefix}_${suffix}"
    digest_name="${value_name}_sha256"
    check_hash "${!value_name}" "${!digest_name}" "${value_name}"
  done
done

[[ "${output_root}" != / && "$(realpath -m -- "${output_root}")" == "${output_root}" ]] || fail "output root must be canonical non-root"
[[ "${output_root##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename differs"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"
mkdir -p -- "$(dirname -- "${output_root}")"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/starc-live-vjp-${SLURM_JOB_ID}.XXXXXX")"
sp4_a_pid=""
sp4_b_pid=""

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/starc-live-vjp-${SLURM_JOB_ID}."*) ;;
    *) cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[starc-live-vjp] PASS all8=true topology=2xSP4 candidates=2 optimizer=false authority=false output=${output_root}"
  else
    echo "[starc-live-vjp] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${status}"
}
terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${sp4_a_pid}" "${sp4_b_pid}"; do [[ "${child}" =~ ^[0-9]+$ ]] && kill "${child}" 2>/dev/null || true; done
  for child in "${sp4_a_pid}" "${sp4_b_pid}"; do [[ "${child}" =~ ^[0-9]+$ ]] && wait "${child}" 2>/dev/null || true; done
  exit "${status}"
}
trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -p -- "${task_scratch}/source-tree" "${task_scratch}/groups"
archive_copy="${task_scratch}/method-source.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod 0444 -- "${archive_copy}"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

required = {
    "methods/bernini_action_editing/run_starc_live_vjp_sp4_v1.py",
    "methods/bernini_action_editing/starch_live_vjp_bridge_v1.py",
    "methods/bernini_action_editing/run_starc_core4_critic_pilot_v1.py",
    "methods/bernini_action_editing/materialize_starc_core4_hidden_v1.py",
    "methods/bernini_action_editing/scripts/auh_run_starc_live_vjp_dual4_v1.sbatch",
}
seen = set()
with tarfile.open(sys.argv[1], "r:*") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        normalized = path.as_posix().lstrip("./")
        if not normalized.startswith("methods/bernini_action_editing/"):
            continue
        if (
            path.is_absolute()
            or ".." in path.parts
            or member.issym()
            or member.islnk()
            or member.isdev()
            or member.isfifo()
        ):
            raise SystemExit("source archive contains an unsafe method member")
        seen.add(normalized)
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"source archive lacks STARC live-VJP closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree" methods/bernini_action_editing

source_tree="${task_scratch}/source-tree"
method_root="${source_tree}/methods/bernini_action_editing"
loader="${method_root}/run_starc_live_vjp_sp4_v1.py"
bridge="${method_root}/starch_live_vjp_bridge_v1.py"
runner="${method_root}/run_starc_core4_critic_pilot_v1.py"
materializer="${method_root}/materialize_starc_core4_hidden_v1.py"
launcher="${method_root}/scripts/auh_run_starc_live_vjp_dual4_v1.sbatch"
for path in "${loader}" "${bridge}" "${runner}" "${materializer}" "${launcher}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "source archive runtime closure differs: ${path}"
done
[[ "$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')" == "$(sha256sum "${launcher}" | awk '{print $1}')" ]] || fail "running launcher differs from source archive"
loader_sha256="$(sha256sum "${loader}" | awk '{print $1}')"
find "${source_tree}" -type f -exec chmod a-w -- {} +

mkdir -- "${output_root}"
mkdir -- "${output_root}/logs"
echo "[starc-live-vjp] allocation=8xMI210 topology=sp4-a:0,1,2,3+sp4-b:4,5,6,7 exact81=true same_x_sigma=true optimizer=false authority=false"

common_args=(
  run
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --checkpoint-content-manifest "${checkpoint_manifest}"
  --expected-checkpoint-tree-sha256 "${pinned_checkpoint_tree_sha256}"
  --expected-checkpoint-content-manifest-sha256 "${pinned_checkpoint_manifest_sha256}"
  --expected-bernini-commit "${pinned_bernini_commit}"
  --expected-veomni-commit "${pinned_veomni_commit}"
  --critic-checkpoint "${critic_checkpoint}"
  --expected-critic-checkpoint-sha256 "${critic_checkpoint_sha256}"
  --critic-checkpoint-receipt "${critic_checkpoint_receipt}"
  --expected-critic-checkpoint-receipt-sha256 "${critic_checkpoint_receipt_sha256}"
  --critic-config-receipt "${critic_config_receipt}"
  --expected-critic-config-receipt-sha256 "${critic_config_receipt_sha256}"
  --materializer-master "${materializer_master}"
  --expected-materializer-master-sha256 "${materializer_master_sha256}"
  --source-archive "${source_archive}"
  --expected-source-archive-sha256 "${source_archive_sha256}"
  --source-git-revision "${source_revision}"
  --expected-loader-source-sha256 "${loader_sha256}"
  --ack-mechanism-probe-only
  --ack-no-editor-parameter-or-update
  --ack-no-scientific-or-action-editing-claim
)

job_mod=$((SLURM_JOB_ID % 10000))
sp4_a_port=$((30000 + 2 * job_mod))
sp4_b_port=$((sp4_a_port + 1))

run_candidate() (
  set -Eeuo pipefail
  local group_id="$1" visible_gpus="$2" master_port="$3" prefix="$4"
  local group_scratch="${task_scratch}/groups/${group_id}"
  mkdir -p -- "${group_scratch}/cache/miopen-user" "${group_scratch}/cache/miopen-custom" \
    "${group_scratch}/cache/torch-extensions" "${group_scratch}/cache/triton" \
    "${group_scratch}/cache/xdg" "${group_scratch}/cache/pycache" "${group_scratch}/tmp"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible_gpus}"
  export MIOPEN_USER_DB_PATH="${group_scratch}/cache/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${group_scratch}/cache/miopen-custom"
  export TORCH_EXTENSIONS_DIR="${group_scratch}/cache/torch-extensions"
  export TRITON_CACHE_DIR="${group_scratch}/cache/triton"
  export XDG_CACHE_HOME="${group_scratch}/cache/xdg"
  export PYTHONPYCACHEPREFIX="${group_scratch}/cache/pycache"
  export TMPDIR="${group_scratch}/tmp"
  export TORCHELASTIC_ERROR_FILE="${group_scratch}/torch-elastic-error.json"
  local manifest_name="${prefix}_manifest" manifest_sha_name="${prefix}_manifest_sha256"
  local source_name="${prefix}_source" source_sha_name="${prefix}_source_sha256"
  local instruction_name="${prefix}_instruction" instruction_sha_name="${prefix}_instruction_sha256"
  local noop_name="${prefix}_noop" noop_sha_name="${prefix}_noop_sha256"
  local clean_name="${prefix}_clean" clean_sha_name="${prefix}_clean_sha256"
  local noise_name="${prefix}_noise" noise_sha_name="${prefix}_noise_sha256"
  echo "[starc-live-vjp:${group_id}] START visible=${visible_gpus} world=4"
  PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
    --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="${master_port}" \
    "${loader}" "${common_args[@]}" \
    --candidate-manifest "${!manifest_name}" \
    --expected-candidate-manifest-sha256 "${!manifest_sha_name}" \
    --source-video "${!source_name}" \
    --expected-source-video-sha256 "${!source_sha_name}" \
    --instruction-file "${!instruction_name}" \
    --expected-instruction-file-sha256 "${!instruction_sha_name}" \
    --noop-caption-file "${!noop_name}" \
    --expected-noop-caption-file-sha256 "${!noop_sha_name}" \
    --current-clean-latent "${!clean_name}" \
    --expected-current-clean-latent-sha256 "${!clean_sha_name}" \
    --native-noise "${!noise_name}" \
    --expected-native-noise-sha256 "${!noise_sha_name}" \
    --output "${output_root}/${group_id}-current-rv2v-live-vjp-composite-v2.json"
)

run_candidate sp4-a "0,1,2,3" "${sp4_a_port}" candidate_a >"${output_root}/logs/sp4-a.log" 2>&1 & sp4_a_pid=$!
run_candidate sp4-b "4,5,6,7" "${sp4_b_port}" candidate_b >"${output_root}/logs/sp4-b.log" 2>&1 & sp4_b_pid=$!
sp4_a_status=0
sp4_b_status=0
wait "${sp4_a_pid}" || sp4_a_status=$?
wait "${sp4_b_pid}" || sp4_b_status=$?
if [[ "${sp4_a_status}" != 0 || "${sp4_b_status}" != 0 ]]; then
  tail -n 240 -- "${output_root}/logs/sp4-a.log" >&2 || true
  tail -n 240 -- "${output_root}/logs/sp4-b.log" >&2 || true
  fail "one or more live VJP SP4 groups failed"
fi

for group_id in sp4-a sp4-b; do
  receipt="${output_root}/${group_id}-current-rv2v-live-vjp-composite-v2.json"
  [[ -f "${receipt}" && ! -L "${receipt}" ]] || fail "${group_id} composite receipt missing"
  [[ "$(stat -c '%a' "${receipt}")" == 400 ]] || fail "${group_id} composite receipt mode differs"
  "${python_bin}" -B - "${receipt}" "${source_archive}" "${source_archive_sha256}" "${source_revision}" <<'PY'
import hashlib, json
from pathlib import Path
import sys

receipt_path, archive_path, archive_sha, revision = sys.argv[1:]
with open(receipt_path, "r", encoding="ascii") as handle:
    row = json.load(handle)
if row.get("schema_version") != "bernini-starc-current-rv2v-live-vjp-composite-binding-v2":
    raise SystemExit("live VJP composite schema differs")
bridge = row.get("live_bridge_binding", {})
if (
    bridge.get("source_archive_path") != archive_path
    or bridge.get("source_archive_file_sha256") != archive_sha
    or bridge.get("source_git_revision") != revision
    or bridge.get("adapter_enabled") is not False
    or bridge.get("frozen_bernini_and_critic") is not True
):
    raise SystemExit("live VJP source/runtime binding differs")
if hashlib.sha256(Path(archive_path).read_bytes()).hexdigest() != archive_sha:
    raise SystemExit("live VJP durable source archive changed")
if (
    row.get("generated_t2v_target_consumed") is not False
    or row.get("editor_parameter_or_optimizer_present") is not False
    or row.get("editor_optimizer_authorized") is not False
    or row.get("scientific_critic_claim_authorized") is not False
):
    raise SystemExit("live VJP composite exceeded mechanism-only authority")
gradient = row.get("gradient_audit", {})
if gradient.get("finite") is not True or gradient.get("nonzero") is not True:
    raise SystemExit("live VJP gradient audit failed")
print("STARC_LIVE_VJP_COMPOSITE_OK authority=false")
PY
done
