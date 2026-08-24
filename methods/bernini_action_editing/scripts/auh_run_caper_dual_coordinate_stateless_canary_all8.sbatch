#!/usr/bin/env bash
#SBATCH --job-name=caper-dir-2wave
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=08:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# Direction-materialization only.  Each wave runs two simultaneous WORLD4/SP4
# groups: dog on GPUs 0..3 and human on GPUs 4..7.  Wave 1 uses the first
# preregistered seed for each cell; wave 2 uses the second.  Every group first
# observes a real native-RV2V sampler call and writes a signed editor packet,
# then the existing clean-direction runner publishes base/+q/-q exact81 MP4s.
# No LoRA VJP, optimizer, adapter update, or scientific success claim exists.

source_archive="${CAPER_SOURCE_ARCHIVE:?set CAPER_SOURCE_ARCHIVE}"
source_archive_sha256="${CAPER_SOURCE_ARCHIVE_SHA256:?set CAPER_SOURCE_ARCHIVE_SHA256}"
source_revision="${CAPER_SOURCE_REVISION:?set CAPER_SOURCE_REVISION}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
output_root="${CAPER_OUTPUT_ROOT:?set CAPER_OUTPUT_ROOT}"
python_bin="${CAPER_PYTHON_BIN:?set CAPER_PYTHON_BIN}"

readonly asset_sha256=b8fe179905cf77951fda2fdc6cf18622b11510263c56df4b306457a3ce717f57
readonly checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly owner_master_sha256=c0d8f3e4a7f3b95269b5196c0d8844327d9e7296dda1828493683a9ae7d707de
readonly owner_sidecar_sha256=24746c91e88e4051c49fe18b06e0e58bb2c4b119b3d946586d9dd6092308030b
readonly owner_public_key_sha256=d1bba83ca1d162128bda71e21c419c476b9328c7892bd1998adcd24c09c577ec
readonly dog_quotient_receipt_sha256=5630c0f511360a6ae0386855f4c00e78e226fea32f71d340773db83ab5c49bd2
readonly human_quotient_receipt_sha256=fb6a37464e98841fe340e5a1411dffe8135640410fd0cef5c1f89b86fe81184e
readonly pinned_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly pinned_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d

readonly owner_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_self_imagined_owner_c3e6f4b_20260809/runs/core2_owner_v1
readonly owner_master="${owner_root}/owner-core2-master-receipt.json"
readonly owner_sidecar=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_self_imagined_owner_c3e6f4b_20260809/audits/full81-v1/qm-owner-audit-131524-sidecar.json
readonly owner_evidence=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_self_imagined_owner_c3e6f4b_20260809/audits/full81-v1/bernini_self_imagined_owner_full81_audit_131524.md
readonly owner_public_key=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_self_imagined_owner_c3e6f4b_20260809/audits/full81-v1/qm-owner-audit-131524-ed25519-public.pem
readonly quotient_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_qm_owner_quotient_680324e_20260809/runs/core2_quotients_v1
readonly editor_authority_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_qmosaic_editor_authority_20260809
readonly editor_authority_private_key="${editor_authority_root}/editor-runtime-ed25519-private.pem"
readonly editor_authority_public_key="${editor_authority_root}/editor-runtime-ed25519-public.pem"
readonly editor_authority_public_key_sha256=b1357fcf5d3b30e51d686a2f1170bc139a7d8c5ea3ef99dc7cc9b2b008d3052d

fail() { echo "[caper-direction] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* && "$1" != / ]] || fail "$2 must be absolute non-root"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }
hash_file() { sha256sum -- "$1" | awk '{print $1}'; }
check_hash() { [[ "$(hash_file "$1")" == "$2" ]] || fail "$3 SHA-256 differs"; }

for name in source_archive bernini_root veomni_root checkpoint checkpoint_manifest output_root python_bin; do
  require_absolute "${!name}" "$name"
done
require_sha256 "${source_archive_sha256}" source_archive_sha256
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be full SHA-1"
for name in source_archive checkpoint_manifest; do
  [[ -f "${!name}" && ! -L "${!name}" ]] || fail "$name must be a plain file"
done
for name in bernini_root veomni_root checkpoint; do
  [[ -d "${!name}" && ! -L "${!name}" ]] || fail "$name must be a plain directory"
done
[[ -f "${python_bin}" && -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python executable differs"
check_hash "${source_archive}" "${source_archive_sha256}" source_archive
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
check_hash "${checkpoint_manifest}" "${checkpoint_manifest_sha256}" checkpoint_manifest
check_hash "${owner_master}" "${owner_master_sha256}" owner_master
check_hash "${owner_sidecar}" "${owner_sidecar_sha256}" owner_sidecar
check_hash "${owner_public_key}" "${owner_public_key_sha256}" owner_public_key
[[ -f "${editor_authority_private_key}" && ! -L "${editor_authority_private_key}" ]] || fail "editor authority private key differs"
[[ -f "${editor_authority_public_key}" && ! -L "${editor_authority_public_key}" ]] || fail "editor authority public key differs"
[[ "$(stat -c %a -- "${editor_authority_private_key}")" == 600 ]] || fail "editor authority private key mode differs"
[[ "$(stat -c %a -- "${editor_authority_public_key}")" == 444 ]] || fail "editor authority public key mode differs"
check_hash "${editor_authority_public_key}" "${editor_authority_public_key_sha256}" editor_authority_public_key
"${python_bin}" -B - "${editor_authority_private_key}" "${editor_authority_public_key}" <<'PY'
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
output_name="$(basename -- "${output_root}")"
[[ "$(realpath -m -- "${output_root}")" == "${output_root}" ]] || fail "output root must be canonical"
[[ "${output_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename differs"
[[ -d "${output_parent}" && ! -L "${output_parent}" && -w "${output_parent}" ]] || fail "output parent differs"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"
[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/caper-dir-${SLURM_JOB_ID}.XXXXXX")"
# Signed child receipts contain absolute artifact paths, so the outer root
# must never be renamed after publication.  Each packet/direction child uses
# its own create-only atomic rename; an unsuccessful job removes this exact,
# prevalidated fresh root.
staging_root="${output_root}"
mkdir -- "${staging_root}"
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
  if [[ "${published}" != true && -d "${staging_root}" && ! -L "${staging_root}" ]]; then
    case "${staging_root}" in
      "${output_root}") rm -rf -- "${staging_root}" || cleanup_failed=1 ;;
      *) cleanup_failed=1 ;;
    esac
  fi
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    case "${task_scratch}" in
      "${scratch_parent%/}/caper-dir-${SLURM_JOB_ID}."*) chmod -R u+w -- "${task_scratch}" || cleanup_failed=1; rm -rf -- "${task_scratch}" || cleanup_failed=1 ;;
      *) cleanup_failed=1 ;;
    esac
  fi
  if [[ "${status}" == 0 && "${cleanup_failed}" == 0 && "${published}" == true ]]; then
    echo "[caper-direction] PASS all8=true waves=2 groups_per_wave=2 sp4=true semantics=UNASSESSED_NO_LORA output=${output_root}"
  else
    [[ "${status}" != 0 ]] || status=2
    echo "[caper-direction] EXIT status=${status} published=${published}" >&2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -- "${task_scratch}/source-tree" "${task_scratch}/caches"
private_source_archive="${task_scratch}/source-archive.tar"
cp -- "${source_archive}" "${private_source_archive}"
check_hash "${private_source_archive}" "${source_archive_sha256}" private_source_archive
"${python_bin}" -B - "${private_source_archive}" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile
archive = sys.argv[1]
required = {
    "methods/bernini_action_editing/run_caper_dual_coordinate_stateless_canary_v1.py",
    "methods/bernini_action_editing/materialize_qmosaic_editor_runtime_v1.py",
    "methods/bernini_action_editing/run_qmosaic_editor_direction_sp4_v1.py",
    "methods/bernini_action_editing/postflight_qmosaic_editor_direction_v1.py",
    "methods/bernini_action_editing/assets/caper_dual_coordinate_core4_canary_v1.json",
    "methods/bernini_action_editing/assets/self_imagined_motion_cotangent_core2_v1.json",
    "methods/bernini_action_editing/scripts/auh_run_caper_dual_coordinate_stateless_canary_all8.sbatch",
}
seen = set()
with tarfile.open(archive, "r:*") as handle:
    for member in handle.getmembers():
        name = member.name
        path = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or path.is_absolute()
            or ".." in path.parts
            or name in seen
            or not (member.isdir() or member.isfile())
        ):
            raise SystemExit(f"unsafe/duplicate source archive member: {name!r}")
        seen.add(name)
if not required.issubset(seen):
    raise SystemExit(f"source archive runtime closure missing: {sorted(required - seen)}")
PY
check_hash "${private_source_archive}" "${source_archive_sha256}" private_source_archive_pre_extract
tar --no-same-owner --no-same-permissions -xf "${private_source_archive}" -C "${task_scratch}/source-tree"
check_hash "${private_source_archive}" "${source_archive_sha256}" private_source_archive_post_extract
source_tree="${task_scratch}/source-tree"
method_root="${source_tree}/methods/bernini_action_editing"
planner="${method_root}/run_caper_dual_coordinate_stateless_canary_v1.py"
materializer="${method_root}/materialize_qmosaic_editor_runtime_v1.py"
direction_runner="${method_root}/run_qmosaic_editor_direction_sp4_v1.py"
postflight="${method_root}/postflight_qmosaic_editor_direction_v1.py"
asset="${method_root}/assets/caper_dual_coordinate_core4_canary_v1.json"
registry="${method_root}/assets/self_imagined_motion_cotangent_core2_v1.json"
launcher="${method_root}/scripts/auh_run_caper_dual_coordinate_stateless_canary_all8.sbatch"
for path in "${planner}" "${materializer}" "${direction_runner}" "${postflight}" "${asset}" "${registry}" "${launcher}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "source archive runtime closure differs"
done
check_hash "${asset}" "${asset_sha256}" asset
check_hash "${registry}" 01fe53b02fa42da8eb5c187a81e6737f323604e7dc26b3eee4f941ad4de82d96 registry
running_launcher="$(realpath -e -- "${BASH_SOURCE[0]}")"
[[ "$(hash_file "${running_launcher}")" == "$(hash_file "${launcher}")" ]] || fail "running launcher differs from source archive"
find "${source_tree}" -type f -exec chmod a-w -- {} +

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
export PYTHONPATH="${method_root}:${source_tree}"

"${python_bin}" -B "${planner}" preflight \
  --asset "${asset}" --expected-asset-sha256 "${asset_sha256}" \
  >"${staging_root}/preflight.receipt.json"

launch_world4() {
  local phase="$1" cell="$2" seed="$3" devices="$4" port="$5" wave_root="$6" log="$7"
  local cache="${task_scratch}/caches/${phase}-${cell}-${seed}"
  mkdir -- "${cache}" "${cache}/torch" "${cache}/miopen" "${cache}/triton" "${cache}/xdg" "${cache}/tmp"
  local packet_root="${wave_root}/${cell}-${seed}/editor-packet"
  local direction_root="${wave_root}/${cell}-${seed}/direction"
  local quotient_cell="${quotient_root}/${cell}"
  local quotient_receipt="${quotient_cell}/owner-motion-specificity-receipt.json"
  local quotient_sha
  if [[ "${cell}" == dog ]]; then quotient_sha="${dog_quotient_receipt_sha256}"; else quotient_sha="${human_quotient_receipt_sha256}"; fi
  if [[ "${phase}" == materialize ]]; then
    env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
      ROCR_VISIBLE_DEVICES="${devices}" \
      TORCH_EXTENSIONS_DIR="${cache}/torch" MIOPEN_USER_DB_PATH="${cache}/miopen" \
      TRITON_CACHE_DIR="${cache}/triton" XDG_CACHE_HOME="${cache}/xdg" TMPDIR="${cache}/tmp" \
      TORCHELASTIC_ERROR_FILE="${cache}/torch-elastic-error.json" \
      "${python_bin}" -B -m torch.distributed.run --nproc_per_node=4 \
        --master_addr=127.0.0.1 --master_port="${port}" \
        "${materializer}" \
        --registry "${registry}" --expected-registry-sha256 01fe53b02fa42da8eb5c187a81e6737f323604e7dc26b3eee4f941ad4de82d96 \
        --cell-id "${cell}" --owner-query-seed "${seed}" --editor-noise-seed "$((seed + 1000))" \
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
        --expected-bernini-commit "${pinned_bernini_commit}" --expected-veomni-commit "${pinned_veomni_commit}" \
        --authority-private-key "${editor_authority_private_key}" \
        --authority-public-key "${editor_authority_public_key}" \
        --expected-authority-public-key-sha256 "${editor_authority_public_key_sha256}" \
        --output-dir "${packet_root}" --method-source-revision "${source_revision}" \
        --method-source-archive "${private_source_archive}" \
        --expected-method-source-archive-sha256 "${source_archive_sha256}" \
        --expected-materializer-source-sha256 "$(hash_file "${materializer}")" >"${log}" 2>&1
  else
    local editor_receipt="${packet_root}/editor-runtime-input.json"
    [[ -f "${editor_receipt}" ]] || fail "${cell}-${seed} editor packet is absent"
    env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
      ROCR_VISIBLE_DEVICES="${devices}" \
      TORCH_EXTENSIONS_DIR="${cache}/torch" MIOPEN_USER_DB_PATH="${cache}/miopen" \
      TRITON_CACHE_DIR="${cache}/triton" XDG_CACHE_HOME="${cache}/xdg" TMPDIR="${cache}/tmp" \
      TORCHELASTIC_ERROR_FILE="${cache}/torch-elastic-error.json" \
      "${python_bin}" -B -m torch.distributed.run --nproc_per_node=4 \
        --master_addr=127.0.0.1 --master_port="${port}" \
        "${direction_runner}" --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
        --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
        --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}" \
        --registry "${registry}" --expected-registry-sha256 01fe53b02fa42da8eb5c187a81e6737f323604e7dc26b3eee4f941ad4de82d96 \
        --cell-id "${cell}" --query-seed "${seed}" \
        --owner-root "${owner_root}" --owner-master-receipt "${owner_master}" \
        --expected-owner-master-receipt-sha256 "${owner_master_sha256}" \
        --owner-audit-sidecar "${owner_sidecar}" --expected-owner-audit-sidecar-sha256 "${owner_sidecar_sha256}" \
        --owner-audit-evidence "${owner_evidence}" --owner-audit-public-key "${owner_public_key}" \
        --expected-owner-audit-public-key-sha256 "${owner_public_key_sha256}" \
        --owner-cell-root "${quotient_cell}" --owner-cell-receipt "${quotient_receipt}" \
        --expected-owner-cell-receipt-sha256 "${quotient_sha}" \
        --editor-receipt "${editor_receipt}" --expected-editor-receipt-sha256 "$(hash_file "${editor_receipt}")" \
        --editor-public-key "${editor_authority_public_key}" --expected-editor-public-key-sha256 "${editor_authority_public_key_sha256}" \
        --editor-artifact-root "${packet_root}" --output-dir "${direction_root}" \
        --method-source-revision "${source_revision}" --method-source-archive-sha256 "${source_archive_sha256}" \
        --expected-bernini-commit "${pinned_bernini_commit}" --expected-veomni-commit "${pinned_veomni_commit}" \
        --no-lora-vjp >"${log}" 2>&1
  fi
}

run_pair() {
  local phase="$1" wave="$2" dog_seed="$3" human_seed="$4"
  local wave_root="${staging_root}/wave${wave}"
  mkdir -p -- "${wave_root}/dog-${dog_seed}" "${wave_root}/human-${human_seed}"
  local base_port=$((21000 + (SLURM_JOB_ID % 10000) * 4 + (wave - 1) * 2))
  launch_world4 "${phase}" dog "${dog_seed}" 0,1,2,3 "${base_port}" "${wave_root}" "${wave_root}/dog-${dog_seed}/${phase}.log" & dog_pid=$!
  launch_world4 "${phase}" human "${human_seed}" 4,5,6,7 "$((base_port + 1))" "${wave_root}" "${wave_root}/human-${human_seed}/${phase}.log" & human_pid=$!
  set +e
  wait "${dog_pid}"; local dog_status=$?
  wait "${human_pid}"; local human_status=$?
  set -e
  dog_pid=""; human_pid=""
  if [[ "${dog_status}" != 0 || "${human_status}" != 0 ]]; then
    tail -n 120 -- "${wave_root}/dog-${dog_seed}/${phase}.log" >&2 || true
    tail -n 120 -- "${wave_root}/human-${human_seed}/${phase}.log" >&2 || true
    fail "wave ${wave} ${phase} failed dog=${dog_status} human=${human_status}"
  fi
}

for wave in 1 2; do
  if [[ "${wave}" == 1 ]]; then dog_seed=2026081502; human_seed=2026081505; else dog_seed=2026081503; human_seed=2026081506; fi
  echo "[caper-direction] wave=${wave}/2 phase=packet topology=dog:0-3+human:4-7"
  run_pair materialize "${wave}" "${dog_seed}" "${human_seed}"
  echo "[caper-direction] wave=${wave}/2 phase=direction topology=dog:0-3+human:4-7 no_lora_vjp=true"
  run_pair direction "${wave}" "${dog_seed}" "${human_seed}"
  for cell_seed in "dog-${dog_seed}" "human-${human_seed}"; do
    direction_root="${staging_root}/wave${wave}/${cell_seed}/direction"
    run_receipt="${direction_root}/run.receipt.json"
    [[ -f "${run_receipt}" ]] || fail "${cell_seed} direction receipt is absent"
    "${python_bin}" -B "${postflight}" --run-receipt "${run_receipt}" \
      --expected-run-receipt-sha256 "$(hash_file "${run_receipt}")" \
      --artifact-root "${direction_root}" --output "${direction_root}/postflight.receipt.json"
  done
done

# Recheck immutable inputs after all eight-GPU work, then publish one directory.
check_hash "${source_archive}" "${source_archive_sha256}" source_archive_postflight
check_hash "${checkpoint_manifest}" "${checkpoint_manifest_sha256}" checkpoint_manifest_postflight
check_hash "${owner_master}" "${owner_master_sha256}" owner_master_postflight
check_hash "${owner_sidecar}" "${owner_sidecar_sha256}" owner_sidecar_postflight
check_hash "${owner_public_key}" "${owner_public_key_sha256}" owner_public_key_postflight

"${python_bin}" -B - "${staging_root}" "${source_revision}" "${source_archive_sha256}" <<'PY'
from pathlib import Path
import hashlib,json,os,sys
root=Path(sys.argv[1]).resolve(strict=True)
rows=[]
for path in sorted(root.rglob('*')):
    if path.is_symlink(): raise SystemExit('staging tree contains symlink')
    if path.is_file():
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({'path':path.relative_to(root).as_posix(),'file_sha256':digest})
unsigned={
 'schema_version':'bernini-caper-direction-all8-two-wave-master-v1',
 'source_revision':sys.argv[2], 'source_archive_sha256':sys.argv[3],
 'topology':'two_waves_x_two_concurrent_WORLD4_SP4',
 'query_cells':[['dog',2026081502],['human',2026081505],['dog',2026081503],['human',2026081506]],
 'actual_native_sampler_editor_packets_materialized':True,
 'base_plus_minus_exact81_direction_artifacts_materialized':True,
 'decoded_semantics_status':'UNASSESSED_NO_LORA',
 'lora_vjp_authorized':False, 'parameter_update_authorized':False,
 'scientific_action_editing_success_claim':False, 'artifacts':rows,
}
payload=json.dumps(unsigned,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('ascii')
receipt={**unsigned,'receipt_digest':hashlib.sha256(payload).hexdigest()}
out=json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('ascii')+b'\n'
path=root/'master.receipt.json'
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,'wb') as handle: handle.write(out); handle.flush(); os.fsync(handle.fileno())
PY

published=true
