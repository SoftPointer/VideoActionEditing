#!/usr/bin/env bash
#SBATCH --job-name=braid-packets-all8
#SBATCH --partition=faculty
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=05:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 027

# Persistent input materialization for the BRAID Stage-0 partial runner.
# This is exactly the packet phase of the tracked Q-MOSAIC ABBA all8 launcher:
# dog and human run concurrently as two independent WORLD4/SP4 groups.  The
# existing materializer owns all VAE/T5/native-RV2V mathematics and its own
# per-cell same-parent staging+rename publication.
#
# Signed packet receipts contain absolute artifact paths.  Consequently the
# outer output root cannot be renamed after materialization.  A fresh root is
# treated as unpublished until the create-only all8 master manifest exists;
# every ordinary failure/signal path removes that exact root.  Job-local source
# extraction, caches, and logs remain in scratch and never publish.
#
# This launcher performs no decode, backward, optimizer construction, parameter
# update, checkpoint write, selection, or semantic action-editing assessment.

fail() { echo "[braid-packets-all8] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* && "$1" != / ]] || fail "$2 must be absolute non-root"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }
require_sha1() { [[ "$1" =~ ^[0-9a-f]{40}$ ]] || fail "$2 must be full lowercase Git identity"; }
hash_file() { sha256sum -- "$1" | awk '{print $1}'; }
check_hash() { [[ "$(hash_file "$1")" == "$2" ]] || fail "$3 SHA-256 differs"; }

[[ "${BRAID_STAGE0_PACKET_ACK_MATERIALIZATION_ONLY:-NO}" == YES ]] || \
  fail "set BRAID_STAGE0_PACKET_ACK_MATERIALIZATION_ONLY=YES"
[[ "${BRAID_STAGE0_PACKET_ACK_NO_DECODE_BACKWARD_OPTIMIZER_UPDATE:-NO}" == YES ]] || \
  fail "set BRAID_STAGE0_PACKET_ACK_NO_DECODE_BACKWARD_OPTIMIZER_UPDATE=YES"
[[ "${BRAID_STAGE0_PACKET_ACK_FIXED_DOG_HUMAN_SEEDS:-NO}" == YES ]] || \
  fail "set BRAID_STAGE0_PACKET_ACK_FIXED_DOG_HUMAN_SEEDS=YES"

source_archive="${BRAID_STAGE0_PACKET_SOURCE_ARCHIVE:?set immutable source archive}"
source_archive_sha256="${BRAID_STAGE0_PACKET_SOURCE_ARCHIVE_SHA256:?set source archive SHA-256}"
source_revision="${BRAID_STAGE0_PACKET_SOURCE_REVISION:?set source revision}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set pinned Bernini-R 1.3B checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint content manifest}"
python_bin="${BRAID_STAGE0_PACKET_PYTHON_BIN:?set vace Python executable}"
output_root="${BRAID_STAGE0_EDITOR_PACKET_OUTPUT_ROOT:?set fresh persistent packet root}"

readonly registry_sha256=01fe53b02fa42da8eb5c187a81e6737f323604e7dc26b3eee4f941ad4de82d96
readonly checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly pinned_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly pinned_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly pinned_materializer_source_sha256=9c7121f5f2de4ac048b1a5ea1c560ef58dde4c8c77a670380acad9357d451a52
readonly pinned_rank_guard_source_sha256=e36c91b0479e3ed2eae7cd170e0342d1885c9a725e1b8d6f502e56846a0f9275
readonly dog_seed=2026081502
readonly human_seed=2026081505
readonly dog_editor_seed=2026082502
readonly human_editor_seed=2026082505

readonly owner_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_self_imagined_owner_c3e6f4b_20260809/runs/core2_owner_v1
readonly owner_master="${owner_root}/owner-core2-master-receipt.json"
readonly owner_master_sha256=c0d8f3e4a7f3b95269b5196c0d8844327d9e7296dda1828493683a9ae7d707de
readonly owner_sidecar=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_self_imagined_owner_c3e6f4b_20260809/audits/full81-v1/qm-owner-audit-131524-sidecar.json
readonly owner_sidecar_sha256=24746c91e88e4051c49fe18b06e0e58bb2c4b119b3d946586d9dd6092308030b
readonly owner_evidence=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_self_imagined_owner_c3e6f4b_20260809/audits/full81-v1/bernini_self_imagined_owner_full81_audit_131524.md
readonly owner_evidence_sha256=3e2335d4d335a9ee8262aa319fc2790dbac3e59b20e554f54b4dc1273f259dc3
readonly owner_public_key=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_self_imagined_owner_c3e6f4b_20260809/audits/full81-v1/qm-owner-audit-131524-ed25519-public.pem
readonly owner_public_key_sha256=d1bba83ca1d162128bda71e21c419c476b9328c7892bd1998adcd24c09c577ec
readonly quotient_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_qm_owner_quotient_680324e_20260809/runs/core2_quotients_v1
readonly dog_quotient_root="${quotient_root}/dog"
readonly human_quotient_root="${quotient_root}/human"
readonly dog_quotient_receipt="${dog_quotient_root}/owner-motion-specificity-receipt.json"
readonly human_quotient_receipt="${human_quotient_root}/owner-motion-specificity-receipt.json"
readonly dog_quotient_receipt_sha256=5630c0f511360a6ae0386855f4c00e78e226fea32f71d340773db83ab5c49bd2
readonly human_quotient_receipt_sha256=fb6a37464e98841fe340e5a1411dffe8135640410fd0cef5c1f89b86fe81184e
readonly editor_authority_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_qmosaic_editor_authority_20260809
readonly editor_private_key="${editor_authority_root}/editor-runtime-ed25519-private.pem"
readonly editor_public_key="${editor_authority_root}/editor-runtime-ed25519-public.pem"
readonly editor_public_key_sha256=b1357fcf5d3b30e51d686a2f1170bc139a7d8c5ea3ef99dc7cc9b2b008d3052d

for name in source_archive bernini_root veomni_root checkpoint checkpoint_manifest python_bin output_root owner_root owner_master owner_sidecar owner_evidence owner_public_key quotient_root dog_quotient_root human_quotient_root dog_quotient_receipt human_quotient_receipt editor_private_key editor_public_key; do
  require_absolute "${!name}" "${name}"
done
require_sha256 "${source_archive_sha256}" source_archive_sha256
require_sha1 "${source_revision}" source_revision
for name in source_archive checkpoint_manifest owner_master owner_sidecar owner_evidence owner_public_key dog_quotient_receipt human_quotient_receipt editor_private_key editor_public_key; do
  [[ -f "${!name}" && ! -L "${!name}" ]] || fail "${name} must be a plain file"
done
for name in bernini_root veomni_root checkpoint owner_root quotient_root dog_quotient_root human_quotient_root; do
  [[ -d "${!name}" && ! -L "${!name}" ]] || fail "${name} must be a plain directory"
done
python_bin="$(realpath -e -- "${python_bin}")"
[[ -f "${python_bin}" && ! -L "${python_bin}" && -x "${python_bin}" ]] || \
  fail "Python executable differs"

check_hash "${source_archive}" "${source_archive_sha256}" source_archive
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || \
  fail "source archive revision differs"
check_hash "${checkpoint_manifest}" "${checkpoint_manifest_sha256}" checkpoint_manifest
check_hash "${owner_master}" "${owner_master_sha256}" owner_master
check_hash "${owner_sidecar}" "${owner_sidecar_sha256}" owner_sidecar
check_hash "${owner_evidence}" "${owner_evidence_sha256}" owner_evidence
check_hash "${owner_public_key}" "${owner_public_key_sha256}" owner_public_key
check_hash "${dog_quotient_receipt}" "${dog_quotient_receipt_sha256}" dog_quotient_receipt
check_hash "${human_quotient_receipt}" "${human_quotient_receipt_sha256}" human_quotient_receipt
check_hash "${editor_public_key}" "${editor_public_key_sha256}" editor_public_key
[[ "$(stat -c %a -- "${editor_private_key}")" == 600 ]] || \
  fail "editor private key mode differs"
[[ "$(stat -c %a -- "${editor_public_key}")" == 444 ]] || \
  fail "editor public key mode differs"
"${python_bin}" -B - "${editor_private_key}" "${editor_public_key}" <<'PY'
from pathlib import Path
import sys
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

private = serialization.load_pem_private_key(Path(sys.argv[1]).read_bytes(), password=None)
public = serialization.load_pem_public_key(Path(sys.argv[2]).read_bytes())
if not isinstance(private, Ed25519PrivateKey) or not isinstance(public, Ed25519PublicKey):
    raise SystemExit("editor authority is not an Ed25519 keypair")
encoding = serialization.Encoding.Raw
form = serialization.PublicFormat.Raw
if private.public_key().public_bytes(encoding, form) != public.public_bytes(encoding, form):
    raise SystemExit("editor authority private/public keys do not match")
PY

output_parent="$(dirname -- "${output_root}")"
output_name="${output_root##*/}"
[[ "$(realpath -m -- "${output_root}")" == "${output_root}" ]] || \
  fail "output root must be canonical"
[[ "${output_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || \
  fail "output basename differs"
[[ -d "${output_parent}" && ! -L "${output_parent}" && -w "${output_parent}" ]] || \
  fail "output parent differs"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || \
  fail "persistent packet root must be fresh"
[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || \
  fail "Slurm job ID differs"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || \
  fail "scratch parent differs"
scratch_parent="$(realpath -e -- "${scratch_parent}")"
case "${output_root}" in
  "${scratch_parent}"|"${scratch_parent%/}/"*)
    fail "persistent packet root must not be inside job-local scratch"
    ;;
esac
task_scratch="$(mktemp -d "${scratch_parent%/}/braid-packets-all8-${SLURM_JOB_ID}.XXXXXX")"
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
      "${scratch_parent%/}/braid-packets-all8-${SLURM_JOB_ID}."*)
        chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
        rm -rf -- "${task_scratch}" || cleanup_failed=1
        ;;
      *) cleanup_failed=1 ;;
    esac
  fi
  if [[ "${status}" == 0 && "${cleanup_failed}" == 0 && "${published}" == true ]]; then
    echo "[braid-packets-all8] PASS topology=dog:0-3+human:4-7 persistent=true materialization_only=true output=${output_root}"
  else
    [[ "${status}" != 0 ]] || status=2
    echo "[braid-packets-all8] EXIT status=${status} published=${published}" >&2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -- "${task_scratch}/source-tree" "${task_scratch}/caches" "${task_scratch}/logs"
archive_copy="${task_scratch}/source-archive.tar"
cp -- "${source_archive}" "${archive_copy}"
check_hash "${archive_copy}" "${source_archive_sha256}" staged_source_archive
"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

required = {
    "methods/bernini_action_editing/materialize_qmosaic_editor_runtime_v1.py",
    "methods/bernini_action_editing/braid_stage0_editor_packet_rank_guard_v1.py",
    "methods/bernini_action_editing/self_imagined_native_rv2v_hidden_vjp_v1.py",
    "methods/bernini_action_editing/pair_v5_action_adapter.py",
    "methods/bernini_action_editing/pair_v5_native_bridge.py",
    "methods/bernini_action_editing/assets/self_imagined_motion_cotangent_core2_v1.json",
    "methods/bernini_action_editing/scripts/auh_materialize_braid_stage0_editor_packets_all8_v1.sbatch",
    "methods/bernini_action_editing/tests/test_materialize_qmosaic_editor_runtime_v1.py",
    "methods/bernini_action_editing/tests/test_auh_materialize_braid_stage0_editor_packets_all8_v1.py",
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
    raise SystemExit(f"source archive lacks BRAID packet materialization closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" \
  -C "${task_scratch}/source-tree" methods/bernini_action_editing
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
materializer="${method_root}/materialize_qmosaic_editor_runtime_v1.py"
rank_guard="${method_root}/braid_stage0_editor_packet_rank_guard_v1.py"
registry="${method_root}/assets/self_imagined_motion_cotangent_core2_v1.json"
launcher="${method_root}/scripts/auh_materialize_braid_stage0_editor_packets_all8_v1.sbatch"
launcher_test="${method_root}/tests/test_auh_materialize_braid_stage0_editor_packets_all8_v1.py"
materializer_test="${method_root}/tests/test_materialize_qmosaic_editor_runtime_v1.py"
for path in "${materializer}" "${rank_guard}" "${registry}" "${launcher}" "${launcher_test}" "${materializer_test}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "authenticated packet runtime closure differs"
done
check_hash "${registry}" "${registry_sha256}" registry
[[ "$(hash_file "${BASH_SOURCE[0]}")" == "$(hash_file "${launcher}")" ]] || \
  fail "running packet launcher differs from authenticated source archive"
materializer_sha256="$(hash_file "${materializer}")"
rank_guard_sha256="$(hash_file "${rank_guard}")"
[[ "${materializer_sha256}" == "${pinned_materializer_source_sha256}" ]] || \
  fail "packet materializer source is not the preregistered tracked implementation"
[[ "${rank_guard_sha256}" == "${pinned_rank_guard_source_sha256}" ]] || \
  fail "packet live-device guard source differs from preregistration"
find "${task_scratch}/source-tree" -type f -exec chmod a-w -- {} +
find "${task_scratch}/source-tree" -type d -exec chmod a-w -- {} +

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 GIT_OPTIONAL_LOCKS=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
export PYTHONPATH="${method_root}:${task_scratch}/source-tree"

# These CPU contracts execute before output creation or any torchrun rank.
for test_name in \
  test_materialize_qmosaic_editor_runtime_v1.py \
  test_auh_materialize_braid_stage0_editor_packets_all8_v1.py; do
  "${python_bin}" -B -m unittest discover -v -s "${method_root}/tests" -p "${test_name}"
done

# The root name is now final because signed child receipts embed it.  It is a
# quarantined, removable root until all8-editor-packets.manifest.json is the
# final create-only operation.  Each materializer atomically publishes its own
# dog/human packet directory from a same-parent private staging directory.
mkdir -- "${output_root}"
chmod 0700 -- "${output_root}"
mkdir -- "${output_root}/device-environment" \
  "${output_root}/device-environment/dog" \
  "${output_root}/device-environment/human"
published_editor_public_key="${output_root}/editor-runtime-ed25519-public.pem"
cp -- "${editor_public_key}" "${published_editor_public_key}"
chmod 0444 -- "${published_editor_public_key}"
check_hash "${published_editor_public_key}" "${editor_public_key_sha256}" published_editor_public_key

launch_cell() {
  local cell="$1" seed="$2" editor_seed="$3" devices="$4" port="$5"
  local cache="${task_scratch}/caches/${cell}"
  local log="${task_scratch}/logs/${cell}.log"
  local packet_root="${output_root}/${cell}"
  local evidence_dir="${output_root}/device-environment/${cell}"
  local quotient_cell quotient_receipt quotient_sha
  if [[ "${cell}" == dog ]]; then
    quotient_cell="${dog_quotient_root}"
    quotient_receipt="${dog_quotient_receipt}"
    quotient_sha="${dog_quotient_receipt_sha256}"
  else
    quotient_cell="${human_quotient_root}"
    quotient_receipt="${human_quotient_receipt}"
    quotient_sha="${human_quotient_receipt_sha256}"
  fi
  [[ "${editor_seed}" == "$((seed + 1000))" ]] || fail "${cell} editor seed domain differs"
  mkdir -p -- "${cache}/torch" "${cache}/miopen" "${cache}/triton" "${cache}/xdg" "${cache}/tmp"
  env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    ROCR_VISIBLE_DEVICES="${devices}" TORCH_EXTENSIONS_DIR="${cache}/torch" \
    MIOPEN_USER_DB_PATH="${cache}/miopen" TRITON_CACHE_DIR="${cache}/triton" \
    XDG_CACHE_HOME="${cache}/xdg" TMPDIR="${cache}/tmp" \
    TORCHELASTIC_ERROR_FILE="${cache}/torch-elastic-error.json" \
    "${python_bin}" -B -m torch.distributed.run --nproc_per_node=4 \
      --master_addr=127.0.0.1 --master_port="${port}" \
      "${rank_guard}" \
      --cell-id "${cell}" --expected-rocr-visible-devices "${devices}" \
      --evidence-dir "${evidence_dir}" --python-bin "${python_bin}" \
      --materializer "${materializer}" \
      --expected-guard-source-sha256 "${rank_guard_sha256}" \
      --expected-materializer-source-sha256 "${materializer_sha256}" -- \
      --registry "${registry}" --expected-registry-sha256 "${registry_sha256}" \
      --cell-id "${cell}" --owner-query-seed "${seed}" --editor-noise-seed "${editor_seed}" \
      --owner-root "${owner_root}" --owner-master-receipt "${owner_master}" \
      --expected-owner-master-receipt-sha256 "${owner_master_sha256}" \
      --owner-audit-sidecar "${owner_sidecar}" --expected-owner-audit-sidecar-sha256 "${owner_sidecar_sha256}" \
      --owner-audit-evidence "${owner_evidence}" --owner-audit-public-key "${owner_public_key}" \
      --expected-owner-audit-public-key-sha256 "${owner_public_key_sha256}" \
      --owner-cell-root "${quotient_cell}" --owner-cell-receipt "${quotient_receipt}" \
      --expected-owner-cell-receipt-sha256 "${quotient_sha}" \
      --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
      --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}" \
      --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
      --expected-bernini-commit "${pinned_bernini_commit}" \
      --expected-veomni-commit "${pinned_veomni_commit}" \
      --authority-private-key "${editor_private_key}" \
      --authority-public-key "${editor_public_key}" \
      --expected-authority-public-key-sha256 "${editor_public_key_sha256}" \
      --output-dir "${packet_root}" --method-source-revision "${source_revision}" \
      --method-source-archive "${archive_copy}" \
      --expected-method-source-archive-sha256 "${source_archive_sha256}" \
      --expected-materializer-source-sha256 "${materializer_sha256}" >"${log}" 2>&1
}

run_pair() {
  local port_base="$1"
  echo "[braid-packets-all8] phase=packet topology=dog:0-3+human:4-7 seeds=${dog_seed},${human_seed}"
  launch_cell dog "${dog_seed}" "${dog_editor_seed}" 0,1,2,3 "${port_base}" & dog_pid=$!
  launch_cell human "${human_seed}" "${human_editor_seed}" 4,5,6,7 "$((port_base + 1))" & human_pid=$!
  set +e
  wait "${dog_pid}"; local dog_status=$?
  wait "${human_pid}"; local human_status=$?
  set -e
  dog_pid=""; human_pid=""
  if [[ "${dog_status}" != 0 || "${human_status}" != 0 ]]; then
    tail -n 180 -- "${task_scratch}/logs/dog.log" >&2 || true
    tail -n 180 -- "${task_scratch}/logs/human.log" >&2 || true
    fail "packet WORLD4 pair failed dog=${dog_status} human=${human_status}"
  fi
}

base_port=$((23000 + (SLURM_JOB_ID % 1000) * 4))
run_pair "${base_port}"

for cell in dog human; do
  [[ -d "${output_root}/${cell}" && ! -L "${output_root}/${cell}" ]] || \
    fail "${cell} persistent editor packet is absent"
  [[ -f "${output_root}/${cell}/editor-runtime-input.json" && ! -L "${output_root}/${cell}/editor-runtime-input.json" ]] || \
    fail "${cell} signed editor receipt is absent"
  for rank in 0 1 2 3; do
    [[ -f "${output_root}/device-environment/${cell}/rank-${rank}.environment.json" ]] || \
      fail "${cell} rank ${rank} live-device evidence is absent"
  done
done

# Rehash every fixed external authority after both WORLD4 groups.  The master
# builder then reopens both signed packets, all eight live-device receipts and
# every persistent file before its create-only manifest write.
check_hash "${source_archive}" "${source_archive_sha256}" source_archive_terminal
check_hash "${archive_copy}" "${source_archive_sha256}" staged_source_archive_terminal
check_hash "${checkpoint_manifest}" "${checkpoint_manifest_sha256}" checkpoint_manifest_terminal
check_hash "${owner_master}" "${owner_master_sha256}" owner_master_terminal
check_hash "${owner_sidecar}" "${owner_sidecar_sha256}" owner_sidecar_terminal
check_hash "${owner_evidence}" "${owner_evidence_sha256}" owner_evidence_terminal
check_hash "${owner_public_key}" "${owner_public_key_sha256}" owner_public_key_terminal
check_hash "${dog_quotient_receipt}" "${dog_quotient_receipt_sha256}" dog_quotient_terminal
check_hash "${human_quotient_receipt}" "${human_quotient_receipt_sha256}" human_quotient_terminal
check_hash "${editor_public_key}" "${editor_public_key_sha256}" editor_public_key_terminal
check_hash "${published_editor_public_key}" "${editor_public_key_sha256}" published_editor_public_key_terminal
check_hash "${materializer}" "${materializer_sha256}" materializer_source_terminal
check_hash "${rank_guard}" "${rank_guard_sha256}" rank_guard_source_terminal

master_manifest="${output_root}/all8-editor-packets.manifest.json"
"${python_bin}" -B - \
  "${method_root}" "${output_root}" "${SLURM_JOB_ID}" \
  "${source_revision}" "${source_archive_sha256}" \
  "${materializer_sha256}" "${rank_guard_sha256}" \
  "${registry_sha256}" "${editor_public_key_sha256}" \
  "${owner_master_sha256}" "${owner_sidecar_sha256}" "${owner_evidence_sha256}" "${owner_public_key_sha256}" \
  "${dog_quotient_receipt_sha256}" "${human_quotient_receipt_sha256}" \
  "${checkpoint_manifest_sha256}" "${checkpoint_tree_sha256}" \
  "${pinned_bernini_commit}" "${pinned_veomni_commit}" <<'PY'
from pathlib import Path
import hashlib
import json
import os
import sys

method_root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(method_root))
import braid_stage0_editor_packet_rank_guard_v1 as guard
import materialize_qmosaic_editor_runtime_v1 as materializer

root = Path(sys.argv[2]).resolve(strict=True)
job_id = int(sys.argv[3])
(
    source_revision,
    source_archive_sha256,
    materializer_sha256,
    rank_guard_sha256,
    registry_sha256,
    editor_public_key_sha256,
    owner_master_sha256,
    owner_sidecar_sha256,
    owner_evidence_sha256,
    owner_public_key_sha256,
    dog_quotient_sha256,
    human_quotient_sha256,
    checkpoint_manifest_sha256,
    checkpoint_tree_sha256,
    bernini_revision,
    veomni_revision,
) = sys.argv[4:]

def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")

def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def strict_json(path):
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"manifest input is not a plain JSON file: {path}")
    try:
        value = json.loads(path.read_bytes().decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"manifest JSON cannot be reopened: {path}: {error}")
    if type(value) is not dict:
        raise SystemExit(f"manifest JSON is not one object: {path}")
    return value

public_key = root / "editor-runtime-ed25519-public.pem"
if sha_file(public_key) != editor_public_key_sha256:
    raise SystemExit("published editor public key differs")

expected_files = {"editor-runtime-ed25519-public.pem"}
expected_dirs = {"dog", "human", "device-environment", "device-environment/dog", "device-environment/human"}
cell_rows = []
cell_specs = (
    ("dog", 2026081502, 2026082502, [0, 1, 2, 3], dog_quotient_sha256),
    ("human", 2026081505, 2026082505, [4, 5, 6, 7], human_quotient_sha256),
)
for cell, owner_seed, editor_seed, devices, quotient_sha in cell_specs:
    packet_root = root / cell
    runtime_path = packet_root / "editor-runtime-input.json"
    materialization_path = packet_root / "materialization-receipt.json"
    runtime_sha = sha_file(runtime_path)
    materialization_sha = sha_file(materialization_path)
    postflight = dict(materializer.validate_published_files(
        output_root=packet_root,
        expected_materialization_receipt_file_sha256=materialization_sha,
        expected_runtime_receipt_file_sha256=runtime_sha,
        public_key_path=public_key,
        expected_public_key_file_sha256=editor_public_key_sha256,
    ))
    runtime = strict_json(runtime_path)
    materialization = strict_json(materialization_path)
    if (
        runtime.get("cell_id") != cell
        or runtime.get("owner_query_seed") != owner_seed
        or runtime.get("editor_noise_seed") != editor_seed
        or materialization.get("cell_id") != cell
        or materialization.get("owner_query_seed") != owner_seed
        or materialization.get("editor_noise_seed") != editor_seed
        or runtime.get("authority_public_key_sha256") != editor_public_key_sha256
    ):
        raise SystemExit(f"{cell} packet fixed coordinate differs")

    packet_files = []
    for path in sorted(packet_root.rglob("*")):
        if path.is_symlink():
            raise SystemExit(f"{cell} packet contains a symlink")
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            expected_dirs.add(relative)
        elif path.is_file():
            expected_files.add(relative)
            packet_files.append(relative)
        else:
            raise SystemExit(f"{cell} packet contains a non-file")

    environment_receipts = []
    evidence_root = root / "device-environment" / cell
    expected_environment_names = {
        f"rank-{rank}.environment.json" for rank in range(4)
    }
    if {path.name for path in evidence_root.iterdir()} != expected_environment_names:
        raise SystemExit(f"{cell} live-device file closure differs")
    for rank in range(4):
        evidence_path = evidence_root / f"rank-{rank}.environment.json"
        evidence = guard.validate_live_environment_receipt(
            strict_json(evidence_path), cell_id=cell, rank=rank,
            expected_guard_source_sha256=rank_guard_sha256,
            expected_materializer_source_sha256=materializer_sha256,
        )
        relative = evidence_path.relative_to(root).as_posix()
        expected_files.add(relative)
        environment_receipts.append({
            "rank": rank,
            "path": str(evidence_path),
            "file_sha256": sha_file(evidence_path),
            "receipt_digest": evidence["receipt_digest"],
        })

    cell_rows.append({
        "cell_id": cell,
        "owner_query_seed": owner_seed,
        "editor_noise_seed": editor_seed,
        "visible_devices": devices,
        "packet_root": str(packet_root),
        "editor_receipt_path": str(runtime_path),
        "editor_receipt_file_sha256": runtime_sha,
        "editor_receipt_digest": runtime["receipt_digest"],
        "materialization_receipt_path": str(materialization_path),
        "materialization_receipt_file_sha256": materialization_sha,
        "materialization_receipt_digest": materialization["receipt_digest"],
        "quotient_receipt_file_sha256": quotient_sha,
        "packet_file_count": len(packet_files),
        "packet_files": packet_files,
        "postflight": postflight,
        "live_device_environment_receipts": environment_receipts,
    })

observed_files = set()
observed_dirs = set()
for path in root.rglob("*"):
    if path.is_symlink():
        raise SystemExit("persistent packet tree contains a symlink")
    relative = path.relative_to(root).as_posix()
    if path.is_file():
        observed_files.add(relative)
    elif path.is_dir():
        observed_dirs.add(relative)
    else:
        raise SystemExit("persistent packet tree contains a non-file")
if observed_files != expected_files or observed_dirs != expected_dirs:
    raise SystemExit(
        f"pre-manifest persistent packet closure differs: "
        f"files={sorted(observed_files ^ expected_files)} "
        f"dirs={sorted(observed_dirs ^ expected_dirs)}"
    )

file_rows = []
for relative in sorted(expected_files):
    path = root / relative
    file_rows.append({
        "relative_path": relative,
        "absolute_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "file_sha256": sha_file(path),
    })

consumer = {
    "DOG_EDITOR_PACKET_ROOT": str(root / "dog"),
    "HUMAN_EDITOR_PACKET_ROOT": str(root / "human"),
    "BRAID_STAGE0_EDITOR_PUBLIC_KEY": str(public_key),
    "BRAID_STAGE0_EDITOR_PUBLIC_KEY_SHA256": editor_public_key_sha256,
    "BRAID_STAGE0_DOG_EDITOR_RECEIPT": cell_rows[0]["editor_receipt_path"],
    "BRAID_STAGE0_DOG_EDITOR_RECEIPT_SHA256": cell_rows[0]["editor_receipt_file_sha256"],
    "BRAID_STAGE0_DOG_EDITOR_ROOT": cell_rows[0]["packet_root"],
    "BRAID_STAGE0_HUMAN_EDITOR_RECEIPT": cell_rows[1]["editor_receipt_path"],
    "BRAID_STAGE0_HUMAN_EDITOR_RECEIPT_SHA256": cell_rows[1]["editor_receipt_file_sha256"],
    "BRAID_STAGE0_HUMAN_EDITOR_ROOT": cell_rows[1]["packet_root"],
}
unsigned = {
    "schema_version": "bernini-braid-stage0-editor-packets-all8-manifest-v1",
    "method": "qmosaic-method-owned-native-rv2v-editor-runtime-v1",
    "classification": "PERSISTENT_STAGE0_INPUT_PACKETS_ONLY",
    "slurm_job_id": job_id,
    "output_root": str(root),
    "source_revision": source_revision,
    "source_archive_sha256": source_archive_sha256,
    "materializer_source_sha256": materializer_sha256,
    "rank_guard_source_sha256": rank_guard_sha256,
    "registry_file_sha256": registry_sha256,
    "fixed_authorities": {
        "owner_master_receipt_file_sha256": owner_master_sha256,
        "owner_audit_sidecar_file_sha256": owner_sidecar_sha256,
        "owner_audit_evidence_file_sha256": owner_evidence_sha256,
        "owner_audit_public_key_file_sha256": owner_public_key_sha256,
        "dog_quotient_receipt_file_sha256": dog_quotient_sha256,
        "human_quotient_receipt_file_sha256": human_quotient_sha256,
        "editor_public_key_file_sha256": editor_public_key_sha256,
        "checkpoint_content_manifest_file_sha256": checkpoint_manifest_sha256,
        "checkpoint_tree_sha256": checkpoint_tree_sha256,
        "bernini_revision": bernini_revision,
        "veomni_revision": veomni_revision,
    },
    "topology": {
        "node_count": 1,
        "physical_gpu_count": 8,
        "independent_world4_group_count": 2,
        "world_size_per_group": 4,
        "sequence_parallel_size": 4,
        "dog_visible_devices": [0, 1, 2, 3],
        "human_visible_devices": [4, 5, 6, 7],
        "groups_launched_concurrently": True,
    },
    "cells": cell_rows,
    "published_editor_public_key": {
        "path": str(public_key),
        "file_sha256": editor_public_key_sha256,
        "private_key_published": False,
    },
    "consumer_environment": consumer,
    "pre_manifest_file_count": len(file_rows),
    "files": file_rows,
    "publication_contract": {
        "per_cell_packet_atomic_rename": True,
        "master_manifest_is_final_completion_boundary": True,
        "absolute_packet_paths_require_final_root_name_before_materialization": True,
        "failed_job_root_is_removed": True,
    },
    "execution_authority": {
        "materialization_only": True,
        "decode_executed": False,
        "backward_executed": False,
        "optimizer_created": False,
        "parameter_update_performed": False,
        "checkpoint_written": False,
        "stage0_training_authority": False,
        "stage_a_authority": False,
        "semantic_action_editing_success_claim": False,
    },
}
receipt = {**unsigned, "receipt_digest": hashlib.sha256(canonical(unsigned)).hexdigest()}
target = root / "all8-editor-packets.manifest.json"
staged = root / f".all8-editor-packets.manifest.json.stage-{job_id}"
payload = canonical(receipt) + b"\n"
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
if target.exists() or target.is_symlink() or staged.exists() or staged.is_symlink():
    raise SystemExit("master manifest publication targets are not fresh")
descriptor = os.open(staged, flags, 0o400)
with os.fdopen(descriptor, "wb", closefd=True) as handle:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
if staged.read_bytes() != payload:
    raise SystemExit("staged master manifest bytes differ")
os.replace(staged, target)
directory_descriptor = os.open(root, os.O_RDONLY)
try:
    os.fsync(directory_descriptor)
finally:
    os.close(directory_descriptor)
reopened = strict_json(target)
reopened_unsigned = dict(reopened)
reopened_digest = reopened_unsigned.pop("receipt_digest", None)
if reopened_digest != hashlib.sha256(canonical(reopened_unsigned)).hexdigest():
    raise SystemExit("reopened master manifest seal differs")
for artifact in reopened["files"]:
    path = root / artifact["relative_path"]
    if (
        str(path) != artifact["absolute_path"]
        or path.stat().st_size != artifact["file_size_bytes"]
        or sha_file(path) != artifact["file_sha256"]
    ):
        raise SystemExit("reopened master manifest file binding differs")
print(payload.decode("ascii").strip(), flush=True)
PY

[[ -f "${master_manifest}" && ! -L "${master_manifest}" ]] || fail "master manifest is absent"
master_manifest_sha256="$(hash_file "${master_manifest}")"
dog_editor_receipt="${output_root}/dog/editor-runtime-input.json"
human_editor_receipt="${output_root}/human/editor-runtime-input.json"
dog_editor_receipt_sha256="$(hash_file "${dog_editor_receipt}")"
human_editor_receipt_sha256="$(hash_file "${human_editor_receipt}")"

find "${output_root}/dog" "${output_root}/human" "${output_root}/device-environment" -type f -exec chmod a-w -- {} +
chmod a-w -- "${published_editor_public_key}" "${master_manifest}"
find "${output_root}" -type d -exec chmod a-w -- {} +

echo "[braid-packets-all8] MASTER_MANIFEST=${master_manifest}"
echo "[braid-packets-all8] MASTER_MANIFEST_SHA256=${master_manifest_sha256}"
echo "export BRAID_STAGE0_EDITOR_PUBLIC_KEY=${published_editor_public_key}"
echo "export BRAID_STAGE0_EDITOR_PUBLIC_KEY_SHA256=${editor_public_key_sha256}"
echo "export BRAID_STAGE0_DOG_EDITOR_RECEIPT=${dog_editor_receipt}"
echo "export BRAID_STAGE0_DOG_EDITOR_RECEIPT_SHA256=${dog_editor_receipt_sha256}"
echo "export BRAID_STAGE0_DOG_EDITOR_ROOT=${output_root}/dog"
echo "export BRAID_STAGE0_HUMAN_EDITOR_RECEIPT=${human_editor_receipt}"
echo "export BRAID_STAGE0_HUMAN_EDITOR_RECEIPT_SHA256=${human_editor_receipt_sha256}"
echo "export BRAID_STAGE0_HUMAN_EDITOR_ROOT=${output_root}/human"
published=true
