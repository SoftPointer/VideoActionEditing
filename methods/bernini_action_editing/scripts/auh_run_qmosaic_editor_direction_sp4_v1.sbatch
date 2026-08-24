#!/usr/bin/env bash
#SBATCH --job-name=qmosaic-dir-sp4
#SBATCH --partition=faculty
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=12:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 027

# QoS is deliberately a submission-time choice (for example, sbatch with its
# qos option).  This immutable launcher never selects or attempts to bypass a
# cluster QoS.  One invocation owns exactly one preregistered cell/query seed.
#
# The signed materializer and direction receipts contain absolute artifact
# paths.  Moving a completed outer tree would therefore invalidate both.  The
# final root is create-only; each child runtime publishes at its final absolute
# path, and COMPLETE.json is the only publication boundary.  A root without
# COMPLETE.json is not a published result and is removed on launcher failure.

fail() { echo "[qmosaic-direction-sp4] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
require_sha256() {
  [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"
}
require_sha1() {
  [[ "$1" =~ ^[0-9a-f]{40}$ ]] || fail "$2 must be a full lowercase Git identity"
}
sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }

[[ "${QMOSAIC_ACK_ENGINEERING_SMOKE_ONLY:-NO}" == YES ]] || \
  fail "set QMOSAIC_ACK_ENGINEERING_SMOKE_ONLY=YES"
[[ "${QMOSAIC_ACK_NO_LORA_VJP_OR_UPDATE:-NO}" == YES ]] || \
  fail "set QMOSAIC_ACK_NO_LORA_VJP_OR_UPDATE=YES"
[[ "${QMOSAIC_ACK_ONE_FIXED_QUERY_SEED:-NO}" == YES ]] || \
  fail "set QMOSAIC_ACK_ONE_FIXED_QUERY_SEED=YES"

source_archive="${QMOSAIC_SOURCE_ARCHIVE:?set immutable git archive}"
source_archive_sha256="${QMOSAIC_SOURCE_ARCHIVE_SHA256:?set git archive SHA-256}"
source_revision="${QMOSAIC_SOURCE_REVISION:?set git archive revision}"
registry="${QMOSAIC_REGISTRY:?set sealed core2 registry}"
registry_sha256="${QMOSAIC_REGISTRY_SHA256:?set registry SHA-256}"
cell_id="${QMOSAIC_CELL_ID:?set exactly one cell: dog or human}"
query_seed="${QMOSAIC_QUERY_SEED:?set exactly one preregistered query seed}"

owner_root="${QMOSAIC_OWNER_ROOT:?set authenticated owner root}"
owner_master="${QMOSAIC_OWNER_MASTER_RECEIPT:?set owner master receipt}"
owner_master_sha256="${QMOSAIC_OWNER_MASTER_RECEIPT_SHA256:?set owner master SHA-256}"
owner_audit_sidecar="${QMOSAIC_OWNER_AUDIT_SIDECAR:?set signed full81 audit sidecar}"
owner_audit_sidecar_sha256="${QMOSAIC_OWNER_AUDIT_SIDECAR_SHA256:?set sidecar SHA-256}"
owner_audit_evidence="${QMOSAIC_OWNER_AUDIT_EVIDENCE:?set detached full81 audit evidence}"
owner_audit_evidence_sha256="${QMOSAIC_OWNER_AUDIT_EVIDENCE_SHA256:?set evidence SHA-256}"
owner_audit_public_key="${QMOSAIC_OWNER_AUDIT_PUBLIC_KEY:?set owner audit Ed25519 public key}"
owner_audit_public_key_sha256="${QMOSAIC_OWNER_AUDIT_PUBLIC_KEY_SHA256:?set owner public-key SHA-256}"
owner_cell_root="${QMOSAIC_OWNER_CELL_ROOT:?set fixed owner cell root}"
owner_cell_receipt="${QMOSAIC_OWNER_CELL_RECEIPT:?set fixed owner cell receipt}"
owner_cell_receipt_sha256="${QMOSAIC_OWNER_CELL_RECEIPT_SHA256:?set cell receipt SHA-256}"

bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini source root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni source root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set pinned Bernini-R 1.3B checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint content manifest}"
checkpoint_manifest_sha256="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set checkpoint manifest SHA-256}"
checkpoint_tree_sha256="${BERNINI_CHECKPOINT_TREE_SHA256:?set checkpoint tree SHA-256}"

authority_private_key="${QMOSAIC_EDITOR_AUTHORITY_PRIVATE_KEY:?set editor Ed25519 private key}"
authority_private_key_sha256="${QMOSAIC_EDITOR_AUTHORITY_PRIVATE_KEY_SHA256:?set private-key SHA-256}"
authority_public_key="${QMOSAIC_EDITOR_AUTHORITY_PUBLIC_KEY:?set editor Ed25519 public key}"
authority_public_key_sha256="${QMOSAIC_EDITOR_AUTHORITY_PUBLIC_KEY_SHA256:?set public-key SHA-256}"
output_root="${QMOSAIC_OUTPUT_ROOT:?set fresh final output root}"
python_bin="${QMOSAIC_PYTHON_BIN:?set vace Python executable}"

readonly expected_registry_sha256="01fe53b02fa42da8eb5c187a81e6737f323604e7dc26b3eee4f941ad4de82d96"
readonly expected_checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
readonly expected_checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
readonly expected_bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
readonly expected_veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"

for name in \
  source_archive registry owner_root owner_master owner_audit_sidecar \
  owner_audit_evidence owner_audit_public_key owner_cell_root \
  owner_cell_receipt bernini_root veomni_root checkpoint checkpoint_manifest \
  authority_private_key authority_public_key output_root python_bin; do
  require_absolute "${!name}" "${name}"
done
for name in \
  source_archive_sha256 registry_sha256 owner_master_sha256 \
  owner_audit_sidecar_sha256 owner_audit_evidence_sha256 \
  owner_audit_public_key_sha256 owner_cell_receipt_sha256 \
  checkpoint_manifest_sha256 checkpoint_tree_sha256 \
  authority_private_key_sha256 authority_public_key_sha256; do
  require_sha256 "${!name}" "${name}"
done
require_sha1 "${source_revision}" source_revision
[[ "${query_seed}" =~ ^[0-9]+$ ]] || fail "query seed must be decimal"
case "${cell_id}:${query_seed}" in
  dog:2026081502|dog:2026081503|human:2026081505|human:2026081506) ;;
  *) fail "cell/query seed is outside the sealed preregistration" ;;
esac
readonly fixed_cell_id="${cell_id}"
readonly fixed_query_seed="${query_seed}"
readonly fixed_editor_noise_seed="$((fixed_query_seed + 1000))"
[[ "${fixed_editor_noise_seed}" != "${fixed_query_seed}" ]] || fail "editor noise seed must be independent"

[[ "${registry_sha256}" == "${expected_registry_sha256}" ]] || fail "registry pin differs"
[[ "${checkpoint_manifest_sha256}" == "${expected_checkpoint_manifest_sha256}" ]] || \
  fail "checkpoint manifest pin differs"
[[ "${checkpoint_tree_sha256}" == "${expected_checkpoint_tree_sha256}" ]] || \
  fail "checkpoint tree pin differs"

canonical_plain_file() {
  local value="$1" label="$2" resolved
  [[ -f "${value}" && ! -L "${value}" ]] || fail "${label} must be a plain file"
  resolved="$(realpath -e -- "${value}")"
  [[ "${resolved}" == "${value}" ]] || fail "${label} must be canonical"
  printf '%s\n' "${resolved}"
}
canonical_plain_dir() {
  local value="$1" label="$2" resolved
  [[ -d "${value}" && ! -L "${value}" ]] || fail "${label} must be a plain directory"
  resolved="$(realpath -e -- "${value}")"
  [[ "${resolved}" == "${value}" ]] || fail "${label} must be canonical"
  printf '%s\n' "${resolved}"
}

source_archive="$(canonical_plain_file "${source_archive}" source_archive)"
registry="$(canonical_plain_file "${registry}" registry)"
owner_root="$(canonical_plain_dir "${owner_root}" owner_root)"
owner_master="$(canonical_plain_file "${owner_master}" owner_master)"
owner_audit_sidecar="$(canonical_plain_file "${owner_audit_sidecar}" owner_audit_sidecar)"
owner_audit_evidence="$(canonical_plain_file "${owner_audit_evidence}" owner_audit_evidence)"
owner_audit_public_key="$(canonical_plain_file "${owner_audit_public_key}" owner_audit_public_key)"
owner_cell_root="$(canonical_plain_dir "${owner_cell_root}" owner_cell_root)"
owner_cell_receipt="$(canonical_plain_file "${owner_cell_receipt}" owner_cell_receipt)"
bernini_root="$(canonical_plain_dir "${bernini_root}" bernini_root)"
veomni_root="$(canonical_plain_dir "${veomni_root}" veomni_root)"
checkpoint="$(canonical_plain_dir "${checkpoint}" checkpoint)"
checkpoint_manifest="$(canonical_plain_file "${checkpoint_manifest}" checkpoint_manifest)"
authority_private_key="$(canonical_plain_file "${authority_private_key}" authority_private_key)"
authority_public_key="$(canonical_plain_file "${authority_public_key}" authority_public_key)"
python_bin="$(realpath -e -- "${python_bin}")"
[[ -f "${python_bin}" && ! -L "${python_bin}" && -x "${python_bin}" ]] || \
  fail "resolved Python executable must be a plain executable"
[[ "$(realpath -e -- "${python_bin}")" == "${python_bin}" ]] || fail "resolved Python must be canonical"

[[ "${output_root}" != / && "$(realpath -m -- "${output_root}")" == "${output_root}" ]] || \
  fail "output root must be canonical non-root"
[[ "${output_root##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || \
  fail "unsafe output basename"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"
output_parent="$(dirname -- "${output_root}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" && -w "${output_parent}" ]] || \
  fail "output parent must be a writable plain directory"
output_parent="$(realpath -e -- "${output_parent}")"
[[ "${output_root}" == "${output_parent}/${output_root##*/}" ]] || fail "output path differs"

verify_file_and_checkpoint_seals() {
  [[ "$(sha256_file "${source_archive}")" == "${source_archive_sha256}" ]] || fail "source archive changed"
  [[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
  [[ "$(sha256_file "${registry}")" == "${registry_sha256}" ]] || fail "registry changed"
  [[ "$(sha256_file "${owner_master}")" == "${owner_master_sha256}" ]] || fail "owner master changed"
  [[ "$(sha256_file "${owner_audit_sidecar}")" == "${owner_audit_sidecar_sha256}" ]] || fail "owner audit sidecar changed"
  [[ "$(sha256_file "${owner_audit_evidence}")" == "${owner_audit_evidence_sha256}" ]] || fail "owner audit evidence changed"
  [[ "$(sha256_file "${owner_audit_public_key}")" == "${owner_audit_public_key_sha256}" ]] || fail "owner audit public key changed"
  [[ "$(sha256_file "${owner_cell_receipt}")" == "${owner_cell_receipt_sha256}" ]] || fail "owner cell receipt changed"
  [[ "$(sha256_file "${checkpoint_manifest}")" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest changed"
  [[ "$(sha256_file "${authority_private_key}")" == "${authority_private_key_sha256}" ]] || fail "editor private key changed"
  [[ "$(sha256_file "${authority_public_key}")" == "${authority_public_key_sha256}" ]] || fail "editor public key changed"
  (
    cd "${checkpoint}"
    sha256sum --strict --status -c "${checkpoint_manifest}"
  ) || fail "checkpoint content verification failed"
}
verify_file_and_checkpoint_seals

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export GIT_OPTIONAL_LOCKS=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || \
  fail "scratch parent differs"
scratch_parent="$(realpath -e -- "${scratch_parent}")"
task_scratch="$(mktemp -d "${scratch_parent%/}/qmosaic-direction-sp4-${SLURM_JOB_ID}.XXXXXX")"
published=false

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/qmosaic-direction-sp4-${SLURM_JOB_ID}."*) ;;
    *) cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  if [[ "${published}" != true && "${cleanup_failed}" == 0 && -d "${output_root}" && ! -L "${output_root}" ]]; then
    [[ ! -e "${output_root}/COMPLETE.json" && "${output_root}" == "${output_parent}/${output_root##*/}" ]] || cleanup_failed=1
    if [[ "${cleanup_failed}" == 0 ]]; then
      chmod -R u+w -- "${output_root}" || cleanup_failed=1
      rm -rf -- "${output_root}" || cleanup_failed=1
    fi
  fi
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[qmosaic-direction-sp4] PASS WORLD4/SP4 exact81@25 fixed_cell=${fixed_cell_id} fixed_query_seed=${fixed_query_seed} seed_selection=false lora_vjp=false update=false output=${output_root}"
  fi
  exit "${status}"
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
chmod 0440 -- "${archive_copy}"
[[ "$(sha256_file "${archive_copy}")" == "${source_archive_sha256}" ]] || fail "staged source archive differs"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

required = {
    "methods/bernini_action_editing/materialize_qmosaic_editor_runtime_v1.py",
    "methods/bernini_action_editing/run_qmosaic_editor_direction_sp4_v1.py",
    "methods/bernini_action_editing/postflight_qmosaic_editor_direction_v1.py",
    "methods/bernini_action_editing/self_imagined_native_rv2v_hidden_vjp_v1.py",
    "methods/bernini_action_editing/self_imagined_motion_cotangent_v1.py",
    "methods/bernini_action_editing/materialize_self_imagined_owner_core2_v1.py",
    "methods/bernini_action_editing/source_self_native_ref_contrastive_v3.py",
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
    "methods/bernini_action_editing/pair_v5_action_adapter.py",
    "methods/bernini_action_editing/pair_v5_native_bridge.py",
    "methods/bernini_action_editing/tri_branch_unipc.py",
    "methods/bernini_action_editing/starch_live_vjp_bridge_v1.py",
    "methods/bernini_action_editing/dclr_runtime_contract.py",
    "methods/bernini_action_editing/audit_pair_v7_phase_a_geometry.py",
    "methods/bernini_action_editing/scripts/auh_run_qmosaic_editor_direction_sp4_v1.sbatch",
    "methods/bernini_action_editing/tests/test_materialize_qmosaic_editor_runtime_v1.py",
    "methods/bernini_action_editing/tests/test_run_qmosaic_editor_direction_sp4_v1.py",
    "methods/bernini_action_editing/tests/test_postflight_qmosaic_editor_direction_v1.py",
    "methods/bernini_action_editing/tests/test_auh_run_qmosaic_editor_direction_sp4_v1.py",
}
seen = set()
with tarfile.open(sys.argv[1], "r:") as handle:
    for member in handle.getmembers():
        pure = PurePosixPath(member.name)
        name = pure.as_posix().rstrip("/")
        if (
            not name
            or pure.is_absolute()
            or ".." in pure.parts
            or member.issym()
            or member.islnk()
            or member.isdev()
            or member.isfifo()
            or (not member.isfile() and not member.isdir())
            or name in seen
        ):
            raise SystemExit("source archive contains an unsafe or duplicate member")
        seen.add(name)
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"source archive lacks Q-MOSAIC runtime closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" \
  -C "${task_scratch}/source-tree" methods/bernini_action_editing
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
materializer="${method_root}/materialize_qmosaic_editor_runtime_v1.py"
runner="${method_root}/run_qmosaic_editor_direction_sp4_v1.py"
postflight="${method_root}/postflight_qmosaic_editor_direction_v1.py"
launcher="${method_root}/scripts/auh_run_qmosaic_editor_direction_sp4_v1.sbatch"
for pinned_file in "${materializer}" "${runner}" "${postflight}" "${launcher}"; do
  [[ -f "${pinned_file}" && ! -L "${pinned_file}" ]] || fail "extracted source closure differs"
done
[[ "$(sha256_file "${BASH_SOURCE[0]}")" == "$(sha256_file "${launcher}")" ]] || \
  fail "running launcher differs from authenticated source archive"
find "${task_scratch}/source-tree" -type f -exec chmod a-w -- {} +
find "${task_scratch}/source-tree" -type d -exec chmod a-w -- {} +

method_tree_sha256() {
  "${python_bin}" -B - "$1" <<'PY'
from pathlib import Path
import hashlib
import sys

root = Path(sys.argv[1]).resolve(strict=True)
digest = hashlib.sha256()
for path in sorted(root.rglob("*"), key=lambda row: row.relative_to(root).as_posix()):
    if path.is_symlink():
        raise SystemExit("method source tree contains a symlink")
    if not path.is_file():
        continue
    relative = path.relative_to(root).as_posix().encode("utf-8")
    payload = path.read_bytes()
    digest.update(len(relative).to_bytes(8, "big"))
    digest.update(relative)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(hashlib.sha256(payload).digest())
print(digest.hexdigest())
PY
}
method_tree_before="$(method_tree_sha256 "${method_root}")"
require_sha256 "${method_tree_before}" method_tree_before
materializer_sha256="$(sha256_file "${materializer}")"

# The official Bernini AUH vendor tree is an immutable release extraction and
# intentionally has no .git directory.  Source identity must therefore use
# train_lora's pinned file-hash fallback, not an unconditional git rev-parse.
# Load that verifier only from the authenticated, extracted, read-only method
# archive; no Python from either external source tree is imported here.
verify_source_tree_identities() {
  PYTHONPATH="${method_root}" "${python_bin}" -B - \
    "${method_root}/train_lora.py" \
    "${bernini_root}" "${veomni_root}" \
    "${expected_bernini_commit}" "${expected_veomni_commit}" <<'PY'
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

module_path = Path(sys.argv[1]).resolve(strict=True)
method_root = module_path.parent.resolve(strict=True)
if module_path != method_root / "train_lora.py" or module_path.is_symlink():
    raise SystemExit("authenticated train_lora verifier path differs")
source_before = hashlib.sha256(module_path.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location(
    "qmosaic_authenticated_train_lora_identity", module_path
)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load authenticated train_lora verifier")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
if Path(module.__file__).resolve(strict=True) != module_path:
    raise SystemExit("source-tree verifier escaped authenticated method root")
expected_bernini = sys.argv[4]
expected_veomni = sys.argv[5]
if (
    module.BERNINI_OFFICIAL_COMMIT != expected_bernini
    or module.VEOMNI_TESTED_COMMIT != expected_veomni
):
    raise SystemExit("authenticated source-tree verifier pins differ")
bernini, veomni, bernini_revision, veomni_revision = module.validate_source_trees(
    sys.argv[2],
    sys.argv[3],
    expected_bernini_commit=expected_bernini,
    expected_veomni_commit=expected_veomni,
)
if (
    bernini != Path(sys.argv[2])
    or veomni != Path(sys.argv[3])
    or bernini_revision != expected_bernini
    or veomni_revision != expected_veomni
    or hashlib.sha256(module_path.read_bytes()).hexdigest() != source_before
):
    raise SystemExit("authenticated source-tree identity result differs")
print(
    json.dumps(
        {
            "schema_version": "qmosaic-source-tree-identity-preflight-v1",
            "authenticated_verifier_path": str(module_path),
            "bernini_root": str(bernini),
            "bernini_revision": bernini_revision,
            "bernini_identity_mode": (
                "git_clean_commit" if (bernini / ".git").is_dir()
                else "pinned_release_file_hashes_without_git"
            ),
            "veomni_root": str(veomni),
            "veomni_revision": veomni_revision,
            "gpu_or_model_load_performed": False,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY
}

# First source-tree identity seal: after authenticated extraction/read-only
# hashing and before contract tests or either WORLD4 model-loading phase.
verify_source_tree_identities

export MIOPEN_USER_DB_PATH="${task_scratch}/cache/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/cache/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/cache/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/cache/triton"
export XDG_CACHE_HOME="${task_scratch}/cache/xdg"
export PYTHONPYCACHEPREFIX="${task_scratch}/cache/pycache"
export TMPDIR="${task_scratch}/tmp"
export TORCHELASTIC_ERROR_FILE="${task_scratch}/torch-elastic-error.json"

# CPU contract tests precede model loading and both WORLD4 launches.
for test_name in \
  test_materialize_qmosaic_editor_runtime_v1.py \
  test_run_qmosaic_editor_direction_sp4_v1.py \
  test_postflight_qmosaic_editor_direction_v1.py \
  test_auh_run_qmosaic_editor_direction_sp4_v1.py; do
  PYTHONPATH="${method_root}" "${python_bin}" -B -m unittest discover -v \
    -s "${method_root}/tests" -p "${test_name}"
done

mkdir -- "${output_root}"
chmod 0700 -- "${output_root}"
editor_root="${output_root}/editor-runtime"
direction_root="${output_root}/direction"

owner_args=(
  --registry "${registry}"
  --expected-registry-sha256 "${registry_sha256}"
  --cell-id "${fixed_cell_id}"
  --owner-root "${owner_root}"
  --owner-master-receipt "${owner_master}"
  --expected-owner-master-receipt-sha256 "${owner_master_sha256}"
  --owner-audit-sidecar "${owner_audit_sidecar}"
  --expected-owner-audit-sidecar-sha256 "${owner_audit_sidecar_sha256}"
  --owner-audit-evidence "${owner_audit_evidence}"
  --owner-audit-public-key "${owner_audit_public_key}"
  --expected-owner-audit-public-key-sha256 "${owner_audit_public_key_sha256}"
  --owner-cell-root "${owner_cell_root}"
  --owner-cell-receipt "${owner_cell_receipt}"
  --expected-owner-cell-receipt-sha256 "${owner_cell_receipt_sha256}"
)
model_args=(
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --checkpoint-content-manifest "${checkpoint_manifest}"
  --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}"
)

echo "[qmosaic-direction-sp4] phase=materialize WORLD4/SP4 rank0_signed_publish=true fixed=${fixed_cell_id}:${fixed_query_seed}"
PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 \
  "${materializer}" \
  "${owner_args[@]}" \
  "${model_args[@]}" \
  --owner-query-seed "${fixed_query_seed}" \
  --editor-noise-seed "${fixed_editor_noise_seed}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --expected-bernini-commit "${expected_bernini_commit}" \
  --expected-veomni-commit "${expected_veomni_commit}" \
  --authority-private-key "${authority_private_key}" \
  --authority-public-key "${authority_public_key}" \
  --expected-authority-public-key-sha256 "${authority_public_key_sha256}" \
  --output-dir "${editor_root}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive "${archive_copy}" \
  --expected-method-source-archive-sha256 "${source_archive_sha256}" \
  --expected-materializer-source-sha256 "${materializer_sha256}"

editor_receipt="${editor_root}/editor-runtime-input.json"
[[ -f "${editor_receipt}" && ! -L "${editor_receipt}" ]] || fail "signed editor receipt missing"
editor_receipt_sha256="$(sha256_file "${editor_receipt}")"
require_sha256 "${editor_receipt_sha256}" editor_receipt_sha256

echo "[qmosaic-direction-sp4] phase=direction WORLD4/SP4 arms=base,plus,minus exact81@25 lora_vjp=false"
PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 \
  "${runner}" \
  "${owner_args[@]}" \
  "${model_args[@]}" \
  --query-seed "${fixed_query_seed}" \
  --editor-receipt "${editor_receipt}" \
  --expected-editor-receipt-sha256 "${editor_receipt_sha256}" \
  --editor-public-key "${authority_public_key}" \
  --expected-editor-public-key-sha256 "${authority_public_key_sha256}" \
  --editor-artifact-root "${editor_root}" \
  --output-dir "${direction_root}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  --expected-bernini-commit "${expected_bernini_commit}" \
  --expected-veomni-commit "${expected_veomni_commit}" \
  --no-lora-vjp

run_receipt="${direction_root}/run.receipt.json"
[[ -f "${run_receipt}" && ! -L "${run_receipt}" ]] || fail "direction run receipt missing"
run_receipt_sha256="$(sha256_file "${run_receipt}")"
require_sha256 "${run_receipt_sha256}" run_receipt_sha256
postflight_receipt="${direction_root}/postflight.receipt.json"

echo "[qmosaic-direction-sp4] phase=postflight live_exact81_25fps_reopen=true semantic=UNASSESSED"
PYTHONPATH="${method_root}" "${python_bin}" -B "${postflight}" \
  --run-receipt "${run_receipt}" \
  --expected-run-receipt-sha256 "${run_receipt_sha256}" \
  --artifact-root "${direction_root}" \
  --output "${postflight_receipt}"

verify_file_and_checkpoint_seals
method_tree_after="$(method_tree_sha256 "${method_root}")"
[[ "${method_tree_after}" == "${method_tree_before}" ]] || fail "authenticated method source tree changed"
verify_source_tree_identities

# Check the complete pre-publication closure.  No source packet or direction
# output is moved: their absolute signed paths are already final.
"${python_bin}" -B - "${output_root}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
expected_files = {
    "editor-runtime/source-video.mp4",
    "editor-runtime/runtime-inputs.safetensors",
    "editor-runtime/materialization-receipt.json",
    "editor-runtime/editor-runtime-input.json",
    "editor-runtime/method-source-archive.tar",
    "editor-runtime/runtime-code/00-qmosaic_materializer.py",
    "editor-runtime/runtime-code/01-native_rv2v_loader_encoder_entrypoint.py",
    "editor-runtime/runtime-code/02-native_rv2v_legacy_contract.py",
    "editor-runtime/runtime-code/03-source_snapshot_and_freeze_audit.py",
    "editor-runtime/runtime-code/04-owner_registry_loader.py",
    "editor-runtime/runtime-code/05-editor_runtime_core_loader.py",
    "editor-runtime/runtime-code/06-native_sampler_contract.py",
    "editor-runtime/runtime-code/07-native_exact40_schedule_contract.py",
    "direction/base.mp4",
    "direction/plus.mp4",
    "direction/minus.mp4",
    "direction/run.receipt.json",
    "direction/postflight.receipt.json",
}
expected_dirs = {"editor-runtime", "editor-runtime/runtime-code", "direction"}
files = set()
dirs = set()
for path in root.rglob("*"):
    if path.is_symlink():
        raise SystemExit("pre-publication output contains a symlink")
    relative = path.relative_to(root).as_posix()
    if path.is_file():
        files.add(relative)
    elif path.is_dir():
        dirs.add(relative)
    else:
        raise SystemExit("pre-publication output contains a non-file")
if files != expected_files or dirs != expected_dirs:
    raise SystemExit(f"pre-publication closure differs: files={sorted(files)} dirs={sorted(dirs)}")
PY

find "${editor_root}" "${direction_root}" -type f -exec chmod a-w -- {} +
find "${editor_root}" "${direction_root}" -type d -exec chmod a-w -- {} +

complete_tmp="${output_root}/.COMPLETE.${SLURM_JOB_ID}.tmp"
complete_final="${output_root}/COMPLETE.json"
[[ ! -e "${complete_tmp}" && ! -L "${complete_tmp}" && ! -e "${complete_final}" && ! -L "${complete_final}" ]] || \
  fail "publication marker already exists"
"${python_bin}" -B - \
  "${output_root}" "${complete_tmp}" "${fixed_cell_id}" "${fixed_query_seed}" \
  "${source_revision}" "${source_archive_sha256}" "${method_tree_before}" <<'PY'
from pathlib import Path
import hashlib
import json
import os
import sys

root = Path(sys.argv[1]).resolve(strict=True)
target = Path(sys.argv[2])
relative_files = (
    "editor-runtime/source-video.mp4",
    "editor-runtime/runtime-inputs.safetensors",
    "editor-runtime/materialization-receipt.json",
    "editor-runtime/editor-runtime-input.json",
    "editor-runtime/method-source-archive.tar",
    "editor-runtime/runtime-code/00-qmosaic_materializer.py",
    "editor-runtime/runtime-code/01-native_rv2v_loader_encoder_entrypoint.py",
    "editor-runtime/runtime-code/02-native_rv2v_legacy_contract.py",
    "editor-runtime/runtime-code/03-source_snapshot_and_freeze_audit.py",
    "editor-runtime/runtime-code/04-owner_registry_loader.py",
    "editor-runtime/runtime-code/05-editor_runtime_core_loader.py",
    "editor-runtime/runtime-code/06-native_sampler_contract.py",
    "editor-runtime/runtime-code/07-native_exact40_schedule_contract.py",
    "direction/base.mp4",
    "direction/plus.mp4",
    "direction/minus.mp4",
    "direction/run.receipt.json",
    "direction/postflight.receipt.json",
)
artifacts = []
for relative in relative_files:
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"publication artifact differs: {relative}")
    artifacts.append(
        {
            "relative_path": relative,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "byte_count": path.stat().st_size,
        }
    )
unsigned = {
    "schema_version": "bernini-qmosaic-editor-direction-sp4-publication-v1",
    "output_root": str(root),
    "fixed_cell_id": sys.argv[3],
    "fixed_query_seed": int(sys.argv[4]),
    "fixed_editor_noise_seed": int(sys.argv[4]) + 1000,
    "owner_editor_noise_seed_shared": False,
    "fixed_single_query_seed": True,
    "seed_selection": False,
    "seed_averaging": False,
    "topology": "WORLD4/SP4_on_4xMI210",
    "decoded_contract": {"frame_count": 81, "fps": 25, "arms": ["base", "plus", "minus"]},
    "execution_authority": {
        "engineering_smoke_only": True,
        "semantic_assessment": "UNASSESSED_NO_METHOD_OWNED_EVALUATOR",
        "lora_vjp": False,
        "optimizer_created": False,
        "parameter_update": False,
    },
    "method_source": {
        "git_revision": sys.argv[5],
        "git_archive_sha256": sys.argv[6],
        "extracted_method_tree_sha256": sys.argv[7],
    },
    "artifacts": artifacts,
    "artifact_count": len(artifacts),
    "absolute_signed_artifact_paths_preserved": True,
    "complete_marker_is_only_publication_boundary": True,
}
payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
receipt = {**unsigned, "receipt_digest": hashlib.sha256(payload).hexdigest()}
wire = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
with os.fdopen(fd, "wb") as handle:
    handle.write(wire)
    handle.flush()
    os.fsync(handle.fileno())
PY

# Same-filesystem rename makes COMPLETE.json visible in one atomic operation.
# Until this succeeds, the create-only root is explicitly unpublished.
chmod a-w -- "${complete_tmp}"
mv -T -- "${complete_tmp}" "${complete_final}"
published=true
