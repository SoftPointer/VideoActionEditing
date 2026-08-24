#!/usr/bin/env bash
#SBATCH --job-name=qmosaic-abba-all8
#SBATCH --partition=faculty
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=08:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 027

# One diagnostic allocation, two concurrent WORLD4/SP4 groups: dog owns GPUs
# 0..3 and human owns GPUs 4..7.  Each group first materializes its signed
# runtime input in job-local scratch, then runs only M/O0/N0/N1/O1/P and writes
# create-only JSON receipts.  The final all8 manifest is the publication
# boundary.  There is no decode, VJP, optimizer, update, checkpoint write, or
# semantic action-editing success claim in this launcher.

fail() { echo "[qmosaic-abba-all8] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* && "$1" != / ]] || fail "$2 must be absolute non-root"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }
require_sha1() { [[ "$1" =~ ^[0-9a-f]{40}$ ]] || fail "$2 must be full lowercase Git identity"; }
hash_file() { sha256sum -- "$1" | awk '{print $1}'; }
check_hash() { [[ "$(hash_file "$1")" == "$2" ]] || fail "$3 SHA-256 differs"; }

[[ "${QMOSAIC_ABBA_ACK_DIAGNOSTIC_ONLY:-NO}" == YES ]] || \
  fail "set QMOSAIC_ABBA_ACK_DIAGNOSTIC_ONLY=YES"
[[ "${QMOSAIC_ABBA_ACK_NO_DECODE_VJP_UPDATE:-NO}" == YES ]] || \
  fail "set QMOSAIC_ABBA_ACK_NO_DECODE_VJP_UPDATE=YES"
[[ "${QMOSAIC_ABBA_ACK_FIXED_DOG_HUMAN_SEEDS:-NO}" == YES ]] || \
  fail "set QMOSAIC_ABBA_ACK_FIXED_DOG_HUMAN_SEEDS=YES"

source_archive="${QMOSAIC_ABBA_SOURCE_ARCHIVE:?set immutable source archive}"
source_archive_sha256="${QMOSAIC_ABBA_SOURCE_ARCHIVE_SHA256:?set source archive SHA-256}"
source_revision="${QMOSAIC_ABBA_SOURCE_REVISION:?set source revision}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set pinned Bernini-R 1.3B checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
python_bin="${QMOSAIC_ABBA_PYTHON_BIN:?set vace Python executable}"
output_root="${QMOSAIC_ABBA_OUTPUT_ROOT:?set fresh final output root}"

readonly registry_sha256=01fe53b02fa42da8eb5c187a81e6737f323604e7dc26b3eee4f941ad4de82d96
readonly checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly pinned_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly pinned_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly dog_seed=2026081502
readonly human_seed=2026081505

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
readonly editor_authority_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_qmosaic_editor_authority_20260809
readonly editor_private_key="${editor_authority_root}/editor-runtime-ed25519-private.pem"
readonly editor_public_key="${editor_authority_root}/editor-runtime-ed25519-public.pem"
readonly editor_public_key_sha256=b1357fcf5d3b30e51d686a2f1170bc139a7d8c5ea3ef99dc7cc9b2b008d3052d

for name in source_archive bernini_root veomni_root checkpoint checkpoint_manifest python_bin output_root; do
  require_absolute "${!name}" "${name}"
done
require_sha256 "${source_archive_sha256}" source_archive_sha256
require_sha1 "${source_revision}" source_revision
for name in source_archive checkpoint_manifest owner_master owner_sidecar owner_evidence owner_public_key editor_private_key editor_public_key; do
  [[ -f "${!name}" && ! -L "${!name}" ]] || fail "${name} must be a plain file"
done
for name in bernini_root veomni_root checkpoint owner_root quotient_root; do
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
check_hash "${owner_public_key}" "${owner_public_key_sha256}" owner_public_key
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
[[ "$(realpath -m -- "${output_root}")" == "${output_root}" ]] || fail "output root must be canonical"
[[ "${output_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename differs"
[[ -d "${output_parent}" && ! -L "${output_parent}" && -w "${output_parent}" ]] || fail "output parent differs"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"
[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || \
  fail "scratch parent differs"
scratch_parent="$(realpath -e -- "${scratch_parent}")"
task_scratch="$(mktemp -d "${scratch_parent%/}/qmosaic-abba-all8-${SLURM_JOB_ID}.XXXXXX")"
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
      "${scratch_parent%/}/qmosaic-abba-all8-${SLURM_JOB_ID}."*)
        chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
        rm -rf -- "${task_scratch}" || cleanup_failed=1
        ;;
      *) cleanup_failed=1 ;;
    esac
  fi
  if [[ "${status}" == 0 && "${cleanup_failed}" == 0 && "${published}" == true ]]; then
    echo "[qmosaic-abba-all8] PASS diagnostic_only=true topology=dog:0-3+human:4-7 output=${output_root}"
  else
    [[ "${status}" != 0 ]] || status=2
    echo "[qmosaic-abba-all8] EXIT status=${status} published=${published}" >&2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -- "${task_scratch}/source-tree" "${task_scratch}/caches" "${task_scratch}/packets" "${task_scratch}/logs"
archive_copy="${task_scratch}/source-archive.tar"
cp -- "${source_archive}" "${archive_copy}"
check_hash "${archive_copy}" "${source_archive_sha256}" staged_source_archive
"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

required = {
    "methods/bernini_action_editing/materialize_qmosaic_editor_runtime_v1.py",
    "methods/bernini_action_editing/run_qmosaic_editor_direction_sp4_v1.py",
    "methods/bernini_action_editing/run_qmosaic_abba_parity_diagnostic_sp4_v1.py",
    "methods/bernini_action_editing/self_imagined_native_rv2v_hidden_vjp_v1.py",
    "methods/bernini_action_editing/pair_v5_action_adapter.py",
    "methods/bernini_action_editing/pair_v5_native_bridge.py",
    "methods/bernini_action_editing/assets/self_imagined_motion_cotangent_core2_v1.json",
    "methods/bernini_action_editing/scripts/auh_run_qmosaic_abba_parity_diagnostic_all8.sbatch",
    "methods/bernini_action_editing/tests/test_run_qmosaic_abba_parity_diagnostic_sp4_v1.py",
    "methods/bernini_action_editing/tests/test_auh_run_qmosaic_abba_parity_diagnostic_all8_launcher.py",
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
    raise SystemExit(f"source archive lacks ABBA diagnostic closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" \
  -C "${task_scratch}/source-tree" methods/bernini_action_editing
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
materializer="${method_root}/materialize_qmosaic_editor_runtime_v1.py"
diagnostic="${method_root}/run_qmosaic_abba_parity_diagnostic_sp4_v1.py"
registry="${method_root}/assets/self_imagined_motion_cotangent_core2_v1.json"
launcher="${method_root}/scripts/auh_run_qmosaic_abba_parity_diagnostic_all8.sbatch"
for path in "${materializer}" "${diagnostic}" "${registry}" "${launcher}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "authenticated runtime closure differs"
done
check_hash "${registry}" "${registry_sha256}" registry
[[ "$(hash_file "${BASH_SOURCE[0]}")" == "$(hash_file "${launcher}")" ]] || \
  fail "running launcher differs from authenticated source archive"
find "${task_scratch}/source-tree" -type f -exec chmod a-w -- {} +
find "${task_scratch}/source-tree" -type d -exec chmod a-w -- {} +

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 GIT_OPTIONAL_LOCKS=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
export PYTHONPATH="${method_root}:${task_scratch}/source-tree"

# CPU contracts precede packet materialization and every GPU diagnostic call.
for test_name in \
  test_run_qmosaic_abba_parity_diagnostic_sp4_v1.py \
  test_auh_run_qmosaic_abba_parity_diagnostic_all8_launcher.py; do
  "${python_bin}" -B -m unittest discover -v -s "${method_root}/tests" -p "${test_name}"
done

mkdir -- "${output_root}"
chmod 0700 -- "${output_root}"

owner_args_for_cell() {
  local cell="$1"
  local quotient_cell="${quotient_root}/${cell}"
  local quotient_receipt="${quotient_cell}/owner-motion-specificity-receipt.json"
  local quotient_sha
  if [[ "${cell}" == dog ]]; then quotient_sha="${dog_quotient_receipt_sha256}"; else quotient_sha="${human_quotient_receipt_sha256}"; fi
  printf '%s\0' \
    --registry "${registry}" --expected-registry-sha256 "${registry_sha256}" \
    --cell-id "${cell}" \
    --owner-root "${owner_root}" --owner-master-receipt "${owner_master}" \
    --expected-owner-master-receipt-sha256 "${owner_master_sha256}" \
    --owner-audit-sidecar "${owner_sidecar}" --expected-owner-audit-sidecar-sha256 "${owner_sidecar_sha256}" \
    --owner-audit-evidence "${owner_evidence}" --owner-audit-public-key "${owner_public_key}" \
    --expected-owner-audit-public-key-sha256 "${owner_public_key_sha256}" \
    --owner-cell-root "${quotient_cell}" --owner-cell-receipt "${quotient_receipt}" \
    --expected-owner-cell-receipt-sha256 "${quotient_sha}"
}

launch_cell() {
  local phase="$1" cell="$2" seed="$3" devices="$4" port="$5"
  local cache="${task_scratch}/caches/${phase}-${cell}"
  local packet_root="${task_scratch}/packets/${cell}"
  local final_root="${output_root}/${cell}"
  local log="${task_scratch}/logs/${phase}-${cell}.log"
  mkdir -p -- "${cache}/torch" "${cache}/miopen" "${cache}/triton" "${cache}/xdg" "${cache}/tmp"
  local owner_args=()
  while IFS= read -r -d '' item; do owner_args+=("${item}"); done < <(owner_args_for_cell "${cell}")
  local common_model=(
    --bernini-root "${bernini_root}" --veomni-root "${veomni_root}"
    --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}"
    --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}"
  )
  if [[ "${phase}" == packet ]]; then
    env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
      ROCR_VISIBLE_DEVICES="${devices}" TORCH_EXTENSIONS_DIR="${cache}/torch" \
      MIOPEN_USER_DB_PATH="${cache}/miopen" TRITON_CACHE_DIR="${cache}/triton" \
      XDG_CACHE_HOME="${cache}/xdg" TMPDIR="${cache}/tmp" \
      TORCHELASTIC_ERROR_FILE="${cache}/torch-elastic-error.json" \
      "${python_bin}" -B -m torch.distributed.run --nproc_per_node=4 \
        --master_addr=127.0.0.1 --master_port="${port}" \
        "${materializer}" "${owner_args[@]}" "${common_model[@]}" \
        --owner-query-seed "${seed}" --editor-noise-seed "$((seed + 1000))" \
        --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
        --expected-bernini-commit "${pinned_bernini_commit}" \
        --expected-veomni-commit "${pinned_veomni_commit}" \
        --authority-private-key "${editor_private_key}" --authority-public-key "${editor_public_key}" \
        --expected-authority-public-key-sha256 "${editor_public_key_sha256}" \
        --output-dir "${packet_root}" --method-source-revision "${source_revision}" \
        --method-source-archive "${archive_copy}" \
        --expected-method-source-archive-sha256 "${source_archive_sha256}" \
        --expected-materializer-source-sha256 "$(hash_file "${materializer}")" >"${log}" 2>&1
  else
    local editor_receipt="${packet_root}/editor-runtime-input.json"
    [[ -f "${editor_receipt}" && ! -L "${editor_receipt}" ]] || fail "${cell} editor packet missing"
    env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
      ROCR_VISIBLE_DEVICES="${devices}" TORCH_EXTENSIONS_DIR="${cache}/torch" \
      MIOPEN_USER_DB_PATH="${cache}/miopen" TRITON_CACHE_DIR="${cache}/triton" \
      XDG_CACHE_HOME="${cache}/xdg" TMPDIR="${cache}/tmp" \
      TORCHELASTIC_ERROR_FILE="${cache}/torch-elastic-error.json" \
      "${python_bin}" -B -m torch.distributed.run --nproc_per_node=4 \
        --master_addr=127.0.0.1 --master_port="${port}" \
        "${diagnostic}" run-world4 "${owner_args[@]}" "${common_model[@]}" \
        --query-seed "${seed}" --editor-receipt "${editor_receipt}" \
        --expected-editor-receipt-sha256 "$(hash_file "${editor_receipt}")" \
        --editor-public-key "${editor_public_key}" \
        --expected-editor-public-key-sha256 "${editor_public_key_sha256}" \
        --editor-artifact-root "${packet_root}" --output-dir "${final_root}" \
        --method-source-revision "${source_revision}" \
        --method-source-archive-sha256 "${source_archive_sha256}" \
        --expected-bernini-commit "${pinned_bernini_commit}" \
        --expected-veomni-commit "${pinned_veomni_commit}" --diagnostic-only >"${log}" 2>&1
  fi
}

run_pair() {
  local phase="$1" port_base="$2"
  echo "[qmosaic-abba-all8] phase=${phase} topology=dog:0-3+human:4-7"
  launch_cell "${phase}" dog "${dog_seed}" 0,1,2,3 "${port_base}" & dog_pid=$!
  launch_cell "${phase}" human "${human_seed}" 4,5,6,7 "$((port_base + 1))" & human_pid=$!
  set +e
  wait "${dog_pid}"; local dog_status=$?
  wait "${human_pid}"; local human_status=$?
  set -e
  dog_pid=""; human_pid=""
  if [[ "${dog_status}" != 0 || "${human_status}" != 0 ]]; then
    tail -n 160 -- "${task_scratch}/logs/${phase}-dog.log" >&2 || true
    tail -n 160 -- "${task_scratch}/logs/${phase}-human.log" >&2 || true
    fail "${phase} WORLD4 pair failed dog=${dog_status} human=${human_status}"
  fi
}

base_port=$((24000 + (SLURM_JOB_ID % 1000) * 4))
run_pair packet "${base_port}"
run_pair diagnostic "$((base_port + 2))"

dog_world4="${output_root}/dog/world4.receipt.json"
human_world4="${output_root}/human/world4.receipt.json"
[[ -f "${dog_world4}" && -f "${human_world4}" ]] || fail "WORLD4 receipts are absent"

# Before the publication manifest exists, the exact output closure is eight
# rank receipts and two WORLD4 receipts.  Scratch packets/logs never publish.
"${python_bin}" -B - "${output_root}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve(strict=True)
expected_files = set()
for cell in ("dog", "human"):
    expected_files.add(f"{cell}/world4.receipt.json")
    for rank in range(4):
        expected_files.add(f"{cell}/rank-{rank}.receipt.json")
files = set()
dirs = set()
for path in root.rglob("*"):
    if path.is_symlink():
        raise SystemExit("diagnostic output contains a symlink")
    relative = path.relative_to(root).as_posix()
    if path.is_file(): files.add(relative)
    elif path.is_dir(): dirs.add(relative)
    else: raise SystemExit("diagnostic output contains a non-file")
if files != expected_files or dirs != {"dog", "human"}:
    raise SystemExit(f"pre-manifest diagnostic closure differs: files={sorted(files)} dirs={sorted(dirs)}")
PY

check_hash "${source_archive}" "${source_archive_sha256}" source_archive_terminal
check_hash "${checkpoint_manifest}" "${checkpoint_manifest_sha256}" checkpoint_manifest_terminal
check_hash "${owner_master}" "${owner_master_sha256}" owner_master_terminal
check_hash "${owner_sidecar}" "${owner_sidecar_sha256}" owner_sidecar_terminal
check_hash "${owner_public_key}" "${owner_public_key_sha256}" owner_public_key_terminal
# The all8 manifest is the final publication operation.  It reopens and hashes
# both WORLD4 receipts and all eight rank receipts before its create-only write.
"${python_bin}" -B "${diagnostic}" aggregate-all8 \
  --dog-world4-receipt "${dog_world4}" \
  --human-world4-receipt "${human_world4}" \
  --output "${output_root}/all8.manifest.json"
chmod a-w -- "${output_root}/all8.manifest.json"
find "${output_root}/dog" "${output_root}/human" -type f -exec chmod a-w -- {} +
find "${output_root}" -type d -exec chmod a-w -- {} +
published=true
