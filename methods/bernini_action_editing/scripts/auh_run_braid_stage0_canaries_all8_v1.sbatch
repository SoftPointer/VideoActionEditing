#!/usr/bin/env bash
#SBATCH --job-name=braid-stage0-all8
#SBATCH --partition=faculty
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=10:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 027

stage0_scope="${BRAID_STAGE0_SCOPE:-full-six-arm}"
case "${stage0_scope}" in
  full-six-arm|reference-4f-a-only) ;;
  *) echo "[braid-stage0-all8] ERROR: unsupported BRAID_STAGE0_SCOPE=${stage0_scope}" >&2; exit 2 ;;
esac

# BRAID Stage-0 forward-only launcher skeleton.
#
# One allocation hosts two concurrent but independent WORLD4/SP4 groups:
# dog owns GPUs 0..3 and human owns GPUs 4..7.  Every cell/arm starts through
# a distinct torchrun invocation, so its model, scheduler, APG buffers, and
# process group are fresh.  Arms are sequential; dog/human within one arm are
# concurrent.  The launcher never runs decode, backward, an optimizer, an
# update, or a checkpoint write.  Its final manifest cannot authorize Stage A.
#
# The production WORLD4 adapter currently implements only the first reset-off
# c0/c0 reference-4f arm.  This full six-arm launcher verifies the adapter's
# implemented-arm closure before creating output or loading a model and fails
# closed until every preregistered arm has real mechanism-specific evidence.

fail() { echo "[braid-stage0-all8] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* && "$1" != / ]] || fail "$2 must be absolute non-root"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }
require_sha1() { [[ "$1" =~ ^[0-9a-f]{40}$ ]] || fail "$2 must be full lowercase Git identity"; }
hash_file() { sha256sum -- "$1" | awk '{print $1}'; }
check_hash() { [[ "$(hash_file "$1")" == "$2" ]] || fail "$3 SHA-256 differs"; }

[[ "${BRAID_STAGE0_ACK_FORWARD_ENGINEERING_ONLY:-NO}" == YES ]] || \
  fail "set BRAID_STAGE0_ACK_FORWARD_ENGINEERING_ONLY=YES"
[[ "${BRAID_STAGE0_ACK_NO_DECODE_BACKWARD_OPTIMIZER_UPDATE:-NO}" == YES ]] || \
  fail "set BRAID_STAGE0_ACK_NO_DECODE_BACKWARD_OPTIMIZER_UPDATE=YES"
[[ "${BRAID_STAGE0_ACK_FIXED_DOG_HUMAN_ARMS:-NO}" == YES ]] || \
  fail "set BRAID_STAGE0_ACK_FIXED_DOG_HUMAN_ARMS=YES"
if [[ "${stage0_scope}" == reference-4f-a-only ]]; then
  [[ "${BRAID_STAGE0_ACK_PARTIAL_STAGE0_NO_STAGE_A_AUTHORITY:-NO}" == YES ]] || \
    fail "set BRAID_STAGE0_ACK_PARTIAL_STAGE0_NO_STAGE_A_AUTHORITY=YES"
fi

source_archive="${BRAID_STAGE0_SOURCE_ARCHIVE:?set immutable source archive}"
source_archive_sha256="${BRAID_STAGE0_SOURCE_ARCHIVE_SHA256:?set source archive SHA-256}"
source_revision="${BRAID_STAGE0_SOURCE_REVISION:?set source revision}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set pinned Bernini-R 1.3B checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
python_bin="${BRAID_STAGE0_PYTHON_BIN:?set vace Python executable}"
output_root="${BRAID_STAGE0_OUTPUT_ROOT:?set fresh final output root}"
editor_public_key="${BRAID_STAGE0_EDITOR_PUBLIC_KEY:?set editor packet public key}"
editor_public_key_sha256="${BRAID_STAGE0_EDITOR_PUBLIC_KEY_SHA256:?set public key SHA-256}"

dog_editor_receipt="${BRAID_STAGE0_DOG_EDITOR_RECEIPT:?set dog signed editor receipt}"
dog_editor_receipt_sha256="${BRAID_STAGE0_DOG_EDITOR_RECEIPT_SHA256:?set dog editor receipt SHA-256}"
dog_editor_root="${BRAID_STAGE0_DOG_EDITOR_ROOT:?set dog editor artifact root}"
human_editor_receipt="${BRAID_STAGE0_HUMAN_EDITOR_RECEIPT:?set human signed editor receipt}"
human_editor_receipt_sha256="${BRAID_STAGE0_HUMAN_EDITOR_RECEIPT_SHA256:?set human editor receipt SHA-256}"
human_editor_root="${BRAID_STAGE0_HUMAN_EDITOR_ROOT:?set human editor artifact root}"

readonly owner_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_self_imagined_owner_c3e6f4b_20260809/runs/core2_owner_v1
readonly owner_master="${owner_root}/owner-core2-master-receipt.json"
readonly owner_master_sha256=c0d8f3e4a7f3b95269b5196c0d8844327d9e7296dda1828493683a9ae7d707de
readonly owner_sidecar=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_self_imagined_owner_c3e6f4b_20260809/audits/full81-v1/qm-owner-audit-131524-sidecar.json
readonly owner_sidecar_sha256=24746c91e88e4051c49fe18b06e0e58bb2c4b119b3d946586d9dd6092308030b
readonly owner_evidence=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_self_imagined_owner_c3e6f4b_20260809/audits/full81-v1/bernini_self_imagined_owner_full81_audit_131524.md
readonly owner_public_key=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_self_imagined_owner_c3e6f4b_20260809/audits/full81-v1/qm-owner-audit-131524-ed25519-public.pem
readonly owner_public_key_sha256=d1bba83ca1d162128bda71e21c419c476b9328c7892bd1998adcd24c09c577ec
readonly quotient_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_qm_owner_quotient_680324e_20260809/runs/core2_quotients_v1
readonly dog_quotient_receipt_sha256=5630c0f511360a6ae0386855f4c00e78e226fea32f71d340773db83ab5c49bd2
readonly human_quotient_receipt_sha256=fb6a37464e98841fe340e5a1411dffe8135640410fd0cef5c1f89b86fe81184e

readonly checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly query_registry_sha256=01fe53b02fa42da8eb5c187a81e6737f323604e7dc26b3eee4f941ad4de82d96
readonly braid_arm_registry_sha256=8a771b459f3a70a1ee618b154527e67cecfaf88aae7575db8ce47a4b7ebd84c3
readonly pinned_editor_public_key_sha256=b1357fcf5d3b30e51d686a2f1170bc139a7d8c5ea3ef99dc7cc9b2b008d3052d
readonly pinned_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly pinned_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly dog_seed=2026081502
readonly human_seed=2026081505

for name in source_archive bernini_root veomni_root checkpoint checkpoint_manifest python_bin output_root editor_public_key dog_editor_receipt dog_editor_root human_editor_receipt human_editor_root owner_root owner_master owner_sidecar owner_evidence owner_public_key quotient_root; do
  require_absolute "${!name}" "${name}"
done
for name in source_archive_sha256 editor_public_key_sha256 dog_editor_receipt_sha256 human_editor_receipt_sha256; do
  require_sha256 "${!name}" "${name}"
done
require_sha1 "${source_revision}" source_revision
[[ "${editor_public_key_sha256}" == "${pinned_editor_public_key_sha256}" ]] || \
  fail "editor public key is not the pinned Stage-0 authority"
for name in source_archive checkpoint_manifest editor_public_key dog_editor_receipt human_editor_receipt owner_master owner_sidecar owner_evidence owner_public_key; do
  [[ -f "${!name}" && ! -L "${!name}" ]] || fail "${name} must be a plain file"
done
for name in bernini_root veomni_root checkpoint dog_editor_root human_editor_root owner_root quotient_root; do
  [[ -d "${!name}" && ! -L "${!name}" ]] || fail "${name} must be a plain directory"
done
python_bin="$(realpath -e -- "${python_bin}")"
[[ -f "${python_bin}" && ! -L "${python_bin}" && -x "${python_bin}" ]] || fail "Python executable differs"
check_hash "${source_archive}" "${source_archive_sha256}" source_archive
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
check_hash "${checkpoint_manifest}" "${checkpoint_manifest_sha256}" checkpoint_manifest
check_hash "${editor_public_key}" "${editor_public_key_sha256}" editor_public_key
check_hash "${dog_editor_receipt}" "${dog_editor_receipt_sha256}" dog_editor_receipt
check_hash "${human_editor_receipt}" "${human_editor_receipt_sha256}" human_editor_receipt
check_hash "${owner_master}" "${owner_master_sha256}" owner_master
check_hash "${owner_sidecar}" "${owner_sidecar_sha256}" owner_sidecar
check_hash "${owner_public_key}" "${owner_public_key_sha256}" owner_public_key

output_parent="$(dirname -- "${output_root}")"
output_name="${output_root##*/}"
[[ "$(realpath -m -- "${output_root}")" == "${output_root}" ]] || fail "output root must be canonical"
[[ "${output_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename differs"
[[ -d "${output_parent}" && ! -L "${output_parent}" && -w "${output_parent}" ]] || fail "output parent differs"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"
[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
scratch_parent="$(realpath -e -- "${scratch_parent}")"
task_scratch="$(mktemp -d "${scratch_parent%/}/braid-stage0-all8-${SLURM_JOB_ID}.XXXXXX")"
published=false
dog_pid=""
human_pid=""

terminate_children() {
  local status="$1" child
  trap - TERM INT
  for child in "${dog_pid}" "${human_pid}"; do
    [[ "${child}" =~ ^[0-9]+$ ]] && kill "${child}" 2>/dev/null || true
  done
  for child in "${dog_pid}" "${human_pid}"; do
    [[ "${child}" =~ ^[0-9]+$ ]] && wait "${child}" 2>/dev/null || true
  done
  exit "${status}"
}
cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  if [[ "${published}" != true && -d "${output_root}" && ! -L "${output_root}" ]]; then
    chmod -R u+w -- "${output_root}" || cleanup_failed=1
    [[ "${output_root}" == "${output_parent}/${output_name}" ]] || cleanup_failed=1
    if [[ "${cleanup_failed}" == 0 ]]; then rm -rf -- "${output_root}" || cleanup_failed=1; fi
  fi
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    case "${task_scratch}" in
      "${scratch_parent%/}/braid-stage0-all8-${SLURM_JOB_ID}."*)
        chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
        rm -rf -- "${task_scratch}" || cleanup_failed=1
        ;;
      *) cleanup_failed=1 ;;
    esac
  fi
  if [[ "${status}" == 0 && "${cleanup_failed}" == 0 && "${published}" == true ]]; then
    echo "[braid-stage0-all8] PASS scope=${stage0_scope} authority=engineering-forward-only topology=dog:0-3+human:4-7 output=${output_root}"
  else
    [[ "${status}" != 0 ]] || status=2
    echo "[braid-stage0-all8] EXIT status=${status} published=${published}" >&2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -- "${task_scratch}/source-tree" "${task_scratch}/caches" "${task_scratch}/logs" "${task_scratch}/execution-key"
execution_private_key="${task_scratch}/execution-key/stage0-execution-ed25519-private.pem"
execution_public_key_staged="${task_scratch}/execution-key/stage0-execution-ed25519-public.pem"
"${python_bin}" -B - "${execution_private_key}" "${execution_public_key_staged}" <<'PY'
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import os
from pathlib import Path
import sys

private_path = Path(sys.argv[1])
public_path = Path(sys.argv[2])
key = Ed25519PrivateKey.generate()
private_bytes = key.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption(),
)
public_bytes = key.public_key().public_bytes(
    serialization.Encoding.PEM,
    serialization.PublicFormat.SubjectPublicKeyInfo,
)
for path, payload, mode in (
    (private_path, private_bytes, 0o600),
    (public_path, public_bytes, 0o444),
):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
PY
execution_public_key_sha256="$(hash_file "${execution_public_key_staged}")"
archive_copy="${task_scratch}/source-archive.tar"
cp -- "${source_archive}" "${archive_copy}"
check_hash "${archive_copy}" "${source_archive_sha256}" staged_source_archive
"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

required = {
    "methods/bernini_action_editing/braid_dual_native_apg_runtime_v1.py",
    "methods/bernini_action_editing/braid_cumulative_action_program_v1.py",
    "methods/bernini_action_editing/braid_stage_a_source_reconstruction_v1.py",
    "methods/bernini_action_editing/braid_stage0_all8_orchestrator_v1.py",
    "methods/bernini_action_editing/run_braid_stage0_world4_v1.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
    "methods/bernini_action_editing/materialize_qmosaic_editor_runtime_v1.py",
    "methods/bernini_action_editing/self_guided_action_field_v1.py",
    "methods/bernini_action_editing/self_imagined_native_rv2v_hidden_vjp_v1.py",
    "methods/bernini_action_editing/source_self_native_ref_contrastive_v3.py",
    "methods/bernini_action_editing/tri_branch_unipc.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/assets/bernini_braid_exact81_arms_v1.json",
    "methods/bernini_action_editing/assets/self_imagined_motion_cotangent_core2_v1.json",
    "methods/bernini_action_editing/scripts/auh_run_braid_stage0_canaries_all8_v1.sbatch",
    "methods/bernini_action_editing/scripts/auh_run_braid_stage0_reference_4f_a_only_all8_v1.sbatch",
    "methods/bernini_action_editing/tests/test_braid_dual_native_apg_runtime_v1.py",
    "methods/bernini_action_editing/tests/test_braid_stage0_all8_orchestrator_v1.py",
    "methods/bernini_action_editing/tests/test_run_braid_stage0_world4_v1.py",
    "methods/bernini_action_editing/tests/test_auh_run_braid_stage0_canaries_all8_v1.py",
    "methods/bernini_action_editing/tests/test_auh_run_braid_stage0_reference_4f_a_only_all8_v1.py",
}
seen = set()
with tarfile.open(sys.argv[1], "r:") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        name = path.as_posix().rstrip("/")
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or name in seen
            or member.issym()
            or member.islnk()
            or member.isdev()
            or member.isfifo()
            or (not member.isfile() and not member.isdir())
        ):
            raise SystemExit("source archive contains an unsafe or duplicate member")
        seen.add(name)
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"source archive lacks BRAID Stage-0 closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" \
  -C "${task_scratch}/source-tree" methods/bernini_action_editing
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
runtime="${method_root}/braid_dual_native_apg_runtime_v1.py"
orchestrator="${method_root}/braid_stage0_all8_orchestrator_v1.py"
runner="${method_root}/run_braid_stage0_world4_v1.py"
query_registry="${method_root}/assets/self_imagined_motion_cotangent_core2_v1.json"
braid_arm_registry="${method_root}/assets/bernini_braid_exact81_arms_v1.json"
launcher="${method_root}/scripts/auh_run_braid_stage0_canaries_all8_v1.sbatch"
partial_launcher="${method_root}/scripts/auh_run_braid_stage0_reference_4f_a_only_all8_v1.sbatch"
for path in "${runtime}" "${orchestrator}" "${runner}" "${query_registry}" "${braid_arm_registry}" "${launcher}" "${partial_launcher}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "authenticated Stage-0 runtime closure differs"
done
check_hash "${query_registry}" "${query_registry_sha256}" query_registry
check_hash "${braid_arm_registry}" "${braid_arm_registry_sha256}" braid_arm_registry
[[ "$(hash_file "${BASH_SOURCE[0]}")" == "$(hash_file "${launcher}")" ]] || fail "running launcher differs from authenticated source archive"
if [[ "${stage0_scope}" == reference-4f-a-only ]]; then
  scope_launcher="${BRAID_STAGE0_SCOPE_LAUNCHER:?partial scope requires authenticated wrapper path}"
  require_absolute "${scope_launcher}" scope_launcher
  [[ -f "${scope_launcher}" && ! -L "${scope_launcher}" ]] || fail "partial scope launcher must be a plain file"
  [[ "$(hash_file "${scope_launcher}")" == "$(hash_file "${partial_launcher}")" ]] || \
    fail "partial scope launcher differs from authenticated source archive"
fi
runtime_sha256="$(hash_file "${runtime}")"
runner_sha256="$(hash_file "${runner}")"
find "${task_scratch}/source-tree" -type f -exec chmod a-w -- {} +
find "${task_scratch}/source-tree" -type d -exec chmod a-w -- {} +

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 GIT_OPTIONAL_LOCKS=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
export PYTHONPATH="${method_root}:${task_scratch}/source-tree"

# Pinned source-tree authentication happens before any GPU/model work.  The
# Bernini snapshot need not contain .git; train_lora validates its fixed file
# content manifest while VeOmni is pinned to its clean revision.
"${python_bin}" -B - "${method_root}/train_lora.py" "${bernini_root}" "${veomni_root}" "${pinned_bernini_commit}" "${pinned_veomni_commit}" <<'PY'
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
spec = spec_from_file_location("braid_stage0_source_auth", sys.argv[1])
if spec is None or spec.loader is None:
    raise SystemExit("cannot load authenticated source validator")
module = module_from_spec(spec)
spec.loader.exec_module(module)
bernini, veomni, bernini_revision, veomni_revision = module.validate_source_trees(
    Path(sys.argv[2]), Path(sys.argv[3]),
    expected_bernini_commit=sys.argv[4], expected_veomni_commit=sys.argv[5],
)
if bernini_revision != sys.argv[4] or veomni_revision != sys.argv[5]:
    raise SystemExit("authenticated source revisions differ")
PY

# CPU contracts and implemented-arm closure run before every model command.
for test_name in \
  test_braid_dual_native_apg_runtime_v1.py \
  test_braid_stage0_all8_orchestrator_v1.py \
  test_run_braid_stage0_world4_v1.py \
  test_auh_run_braid_stage0_canaries_all8_v1.py \
  test_auh_run_braid_stage0_reference_4f_a_only_all8_v1.py; do
  "${python_bin}" -B -m unittest discover -v -s "${method_root}/tests" -p "${test_name}"
done

# Do not spend two exact40 WORLD4 runs and then discover that a later arm is
# still a stub.  The first-arm runner is independently usable, but this full
# launcher publishes nothing until all six real adapters exist.
"${python_bin}" -B - "${runner}" "${stage0_scope}" <<'PY'
from importlib.util import module_from_spec, spec_from_file_location
import sys

spec = spec_from_file_location("braid_stage0_world4_scope", sys.argv[1])
if spec is None or spec.loader is None:
    raise SystemExit("cannot load WORLD4 implementation closure")
module = module_from_spec(spec)
spec.loader.exec_module(module)
expected = {
    "parity-reset-off-reference-4f-a",
    "parity-reset-off-reference-4f-b",
    "parity-reset-off-shared-negative-3f",
    "reset-on-reference-4f",
    "capacity-source-bias-off-reference-4f",
    "capacity-source-bias-on-reference-4f",
}
implemented = set(module.stage0.IMPLEMENTED_WORLD4_ARM_IDS)
scope = sys.argv[2]
if scope == "full-six-arm" and implemented != expected:
    raise SystemExit(
        "full all8 launcher is intentionally blocked before GPU model load; "
        f"implemented={sorted(implemented)} missing={sorted(expected - implemented)}"
    )
if scope == "reference-4f-a-only" and module.SUPPORTED_ARM_ID not in implemented:
    raise SystemExit("reference-4f-a-only runner is not implemented")
PY

mkdir -- "${output_root}"
chmod 0700 -- "${output_root}"
execution_public_key="${output_root}/stage0-execution-ed25519-public.pem"
cp -- "${execution_public_key_staged}" "${execution_public_key}"
chmod 0444 -- "${execution_public_key}"
check_hash "${execution_public_key}" "${execution_public_key_sha256}" execution_public_key
mkdir -- "${output_root}/evidence" "${output_root}/evidence/dog" "${output_root}/evidence/human"
plan="${output_root}/stage0.plan.json"
"${python_bin}" -B "${orchestrator}" write-plan \
  --output "${plan}" --slurm-job-id "${SLURM_JOB_ID}" --output-root "${output_root}" \
  --method-source-revision "${source_revision}" --source-archive-sha256 "${source_archive_sha256}" \
  --runtime-source-sha256 "${runtime_sha256}" --runner-source-sha256 "${runner_sha256}" \
  --dog-editor-receipt-file-sha256 "${dog_editor_receipt_sha256}" \
  --human-editor-receipt-file-sha256 "${human_editor_receipt_sha256}" \
  --execution-public-key-file-sha256 "${execution_public_key_sha256}"
plan_file_sha256="$(hash_file "${plan}")"

readonly full_arms=(
  parity-reset-off-reference-4f-a
  parity-reset-off-reference-4f-b
  parity-reset-off-shared-negative-3f
  reset-on-reference-4f
  capacity-source-bias-off-reference-4f
  capacity-source-bias-on-reference-4f
)
readonly partial_arms=(parity-reset-off-reference-4f-a)
if [[ "${stage0_scope}" == full-six-arm ]]; then
  arms=("${full_arms[@]}")
else
  arms=("${partial_arms[@]}")
fi
readonly arms

launch_cell() {
  local cell="$1" seed="$2" devices="$3" port="$4" arm="$5"
  local cache="${task_scratch}/caches/${cell}-${arm}"
  local log="${task_scratch}/logs/${cell}-${arm}.log"
  local final_root="${output_root}/evidence/${cell}/${arm}"
  local editor_receipt editor_receipt_sha editor_root
  local quotient_cell="${quotient_root}/${cell}"
  local quotient_receipt="${quotient_cell}/owner-motion-specificity-receipt.json"
  local quotient_sha
  if [[ "${cell}" == dog ]]; then
    editor_receipt="${dog_editor_receipt}"; editor_receipt_sha="${dog_editor_receipt_sha256}"; editor_root="${dog_editor_root}"
    quotient_sha="${dog_quotient_receipt_sha256}"
  else
    editor_receipt="${human_editor_receipt}"; editor_receipt_sha="${human_editor_receipt_sha256}"; editor_root="${human_editor_root}"
    quotient_sha="${human_quotient_receipt_sha256}"
  fi
  mkdir -p -- "${cache}/torch" "${cache}/miopen" "${cache}/triton" "${cache}/xdg" "${cache}/tmp"
  env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    ROCR_VISIBLE_DEVICES="${devices}" TORCH_EXTENSIONS_DIR="${cache}/torch" \
    MIOPEN_USER_DB_PATH="${cache}/miopen" TRITON_CACHE_DIR="${cache}/triton" \
    XDG_CACHE_HOME="${cache}/xdg" TMPDIR="${cache}/tmp" \
    TORCHELASTIC_ERROR_FILE="${cache}/torch-elastic-error.json" \
    "${python_bin}" -B -m torch.distributed.run --nproc_per_node=4 \
      --master_addr=127.0.0.1 --master_port="${port}" \
      "${runner}" run-world4 \
      --plan "${plan}" --expected-plan-file-sha256 "${plan_file_sha256}" \
      --cell-id "${cell}" --query-seed "${seed}" --arm-id "${arm}" \
      --query-registry "${query_registry}" --expected-query-registry-sha256 "${query_registry_sha256}" \
      --braid-arm-registry "${braid_arm_registry}" --expected-braid-arm-registry-sha256 "${braid_arm_registry_sha256}" \
      --dual-runtime-source "${runtime}" --expected-dual-runtime-source-sha256 "${runtime_sha256}" \
      --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
      --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}" \
      --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
      --expected-bernini-commit "${pinned_bernini_commit}" --expected-veomni-commit "${pinned_veomni_commit}" \
      --owner-root "${owner_root}" --owner-master-receipt "${owner_master}" \
      --expected-owner-master-receipt-sha256 "${owner_master_sha256}" \
      --owner-audit-sidecar "${owner_sidecar}" --expected-owner-audit-sidecar-sha256 "${owner_sidecar_sha256}" \
      --owner-audit-evidence "${owner_evidence}" --owner-audit-public-key "${owner_public_key}" \
      --expected-owner-audit-public-key-sha256 "${owner_public_key_sha256}" \
      --owner-cell-root "${quotient_cell}" --owner-cell-receipt "${quotient_receipt}" \
      --expected-owner-cell-receipt-sha256 "${quotient_sha}" \
      --editor-receipt "${editor_receipt}" --expected-editor-receipt-sha256 "${editor_receipt_sha}" \
      --editor-public-key "${editor_public_key}" --expected-editor-public-key-sha256 "${editor_public_key_sha256}" \
      --editor-artifact-root "${editor_root}" \
      --execution-private-key "${execution_private_key}" \
      --execution-public-key "${execution_public_key}" \
      --expected-execution-public-key-sha256 "${execution_public_key_sha256}" \
      --output-dir "${final_root}" \
      --ack-forward-only-no-decode-backward-optimizer-update >"${log}" 2>&1
}

run_pair() {
  local arm="$1" port_base="$2"
  echo "[braid-stage0-all8] arm=${arm} topology=dog:0-3+human:4-7 fresh_process=true"
  launch_cell dog "${dog_seed}" 0,1,2,3 "${port_base}" "${arm}" & dog_pid=$!
  launch_cell human "${human_seed}" 4,5,6,7 "$((port_base + 1))" "${arm}" & human_pid=$!
  set +e
  wait "${dog_pid}"; local dog_status=$?
  wait "${human_pid}"; local human_status=$?
  set -e
  dog_pid=""; human_pid=""
  if [[ "${dog_status}" != 0 || "${human_status}" != 0 ]]; then
    tail -n 180 -- "${task_scratch}/logs/dog-${arm}.log" >&2 || true
    tail -n 180 -- "${task_scratch}/logs/human-${arm}.log" >&2 || true
    fail "${arm} WORLD4 pair failed dog=${dog_status} human=${human_status}"
  fi
  for cell in dog human; do
    local receipt="${output_root}/evidence/${cell}/${arm}/world4.receipt.json"
    [[ -f "${receipt}" && ! -L "${receipt}" ]] || fail "${cell}/${arm} WORLD4 receipt absent"
    "${python_bin}" -B "${orchestrator}" validate-world4 \
      --plan "${plan}" --receipt "${receipt}" --cell-id "${cell}" --arm-id "${arm}"
  done
}

base_port=$((25000 + (SLURM_JOB_ID % 1000) * 16))
for index in "${!arms[@]}"; do
  run_pair "${arms[$index]}" "$((base_port + index * 2))"
done

# Re-seal all immutable inputs after all 12 fresh WORLD4 arms.  The aggregate
# reopens every receipt, enforces the exact output closure, and is the only
# complete publication operation.
check_hash "${source_archive}" "${source_archive_sha256}" source_archive_terminal
check_hash "${checkpoint_manifest}" "${checkpoint_manifest_sha256}" checkpoint_manifest_terminal
check_hash "${editor_public_key}" "${editor_public_key_sha256}" editor_public_key_terminal
check_hash "${dog_editor_receipt}" "${dog_editor_receipt_sha256}" dog_editor_receipt_terminal
check_hash "${human_editor_receipt}" "${human_editor_receipt_sha256}" human_editor_receipt_terminal
check_hash "${owner_master}" "${owner_master_sha256}" owner_master_terminal
check_hash "${owner_sidecar}" "${owner_sidecar_sha256}" owner_sidecar_terminal
check_hash "${owner_public_key}" "${owner_public_key_sha256}" owner_public_key_terminal
check_hash "${runtime}" "${runtime_sha256}" runtime_source_terminal
check_hash "${runner}" "${runner_sha256}" runner_source_terminal
check_hash "${execution_public_key}" "${execution_public_key_sha256}" execution_public_key_terminal

if [[ "${stage0_scope}" == full-six-arm ]]; then
  publication="${output_root}/all8.manifest.json"
  "${python_bin}" -B "${orchestrator}" aggregate-all8 \
    --plan "${plan}" --evidence-root "${output_root}/evidence" \
    --output "${publication}"
else
  publication="${output_root}/reference-4f-a-only-all8.receipt.json"
  "${python_bin}" -B "${orchestrator}" aggregate-reference4f-a-only-all8 \
    --plan "${plan}" --evidence-root "${output_root}/evidence" \
    --output "${publication}"
fi
chmod a-w -- "${publication}" "${plan}" "${execution_public_key}"
find "${output_root}/evidence" -type f -exec chmod a-w -- {} +
find "${output_root}" -type d -exec chmod a-w -- {} +
published=true
