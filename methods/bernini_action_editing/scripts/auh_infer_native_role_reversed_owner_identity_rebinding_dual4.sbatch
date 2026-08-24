#!/usr/bin/env bash
#SBATCH --job-name=bernini-role-reverse
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=08:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# Frozen exact81 diagnostic only.  GPUs 0-3 run dog and GPUs 4-7 run human as
# two isolated WORLD4/Ulysses4 groups.  Each group renders the three matched
# owner-only / owner+source-refs / source+source-refs arms at exact40.

repo_root="${ROLE_REVERSED_REPO_ROOT:?set clean repository root}"
source_archive="${ROLE_REVERSED_SOURCE_ARCHIVE:?set git archive}"
source_archive_sha256="${ROLE_REVERSED_SOURCE_ARCHIVE_SHA256:?set git archive SHA-256}"
registry="${ROLE_REVERSED_REGISTRY:?set sealed core2 registry}"
registry_sha256="${ROLE_REVERSED_REGISTRY_SHA256:?set registry SHA-256}"
owner_root="${ROLE_REVERSED_OWNER_ROOT:?set job131524 owner root}"
owner_master="${ROLE_REVERSED_OWNER_MASTER:?set job131524 master receipt}"
owner_master_sha256="${ROLE_REVERSED_OWNER_MASTER_SHA256:?set master receipt SHA-256}"
audit_sidecar="${ROLE_REVERSED_AUDIT_SIDECAR:?set signed full81 audit sidecar}"
audit_sidecar_sha256="${ROLE_REVERSED_AUDIT_SIDECAR_SHA256:?set audit sidecar SHA-256}"
audit_evidence="${ROLE_REVERSED_AUDIT_EVIDENCE:?set detached full81 audit evidence}"
audit_public_key="${ROLE_REVERSED_AUDIT_PUBLIC_KEY:?set Ed25519 public key}"
audit_public_key_sha256="${ROLE_REVERSED_AUDIT_PUBLIC_KEY_SHA256:?set public key SHA-256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
output_root="${ROLE_REVERSED_OUTPUT_ROOT:?set fresh output root}"
python_bin="${ROLE_REVERSED_PYTHON_BIN:?set vace Python executable}"

fail() {
  echo "[role-reversed-owner] ERROR: $*" >&2
  exit 2
}

for value in \
  "${repo_root}" "${source_archive}" "${registry}" "${owner_root}" \
  "${owner_master}" "${audit_sidecar}" "${audit_evidence}" "${audit_public_key}" \
  "${bernini_root}" "${veomni_root}" "${checkpoint}" \
  "${checkpoint_manifest}" "${output_root}" "${python_bin}"
do
  [[ "${value}" == /* ]] || fail "all paths must be absolute"
done
for value in \
  "${source_archive_sha256}" "${registry_sha256}" "${owner_master_sha256}" \
  "${audit_sidecar_sha256}" "${audit_public_key_sha256}"
do
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || fail "SHA-256 input differs"
done

repo_root="$(realpath -e -- "${repo_root}")"
source_archive="$(realpath -e -- "${source_archive}")"
registry="$(realpath -e -- "${registry}")"
owner_root="$(realpath -e -- "${owner_root}")"
owner_master="$(realpath -e -- "${owner_master}")"
audit_sidecar="$(realpath -e -- "${audit_sidecar}")"
audit_evidence="$(realpath -e -- "${audit_evidence}")"
audit_public_key="$(realpath -e -- "${audit_public_key}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"

[[ -d "${repo_root}" && ! -L "${repo_root}" ]] || fail "repository root differs"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive differs"
[[ -f "${registry}" && ! -L "${registry}" ]] || fail "registry differs"
[[ -d "${owner_root}" && ! -L "${owner_root}" ]] || fail "owner root differs"
[[ -f "${owner_master}" && ! -L "${owner_master}" ]] || fail "owner master differs"
[[ -f "${audit_sidecar}" && ! -L "${audit_sidecar}" ]] || fail "audit sidecar differs"
[[ -f "${audit_evidence}" && ! -L "${audit_evidence}" ]] || fail "audit evidence differs"
[[ -f "${audit_public_key}" && ! -L "${audit_public_key}" ]] || fail "audit public key differs"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini root differs"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni root differs"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint differs"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest differs"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "Python runtime differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive changed"
[[ "$(sha256sum "${registry}" | awk '{print $1}')" == "${registry_sha256}" ]] || fail "registry changed"
[[ "$(sha256sum "${owner_master}" | awk '{print $1}')" == "${owner_master_sha256}" ]] || fail "owner master changed"
[[ "$(sha256sum "${audit_sidecar}" | awk '{print $1}')" == "${audit_sidecar_sha256}" ]] || fail "audit sidecar changed"
[[ "$(sha256sum "${audit_public_key}" | awk '{print $1}')" == "${audit_public_key_sha256}" ]] || fail "audit public key changed"

method_root="${repo_root}/methods/bernini_action_editing"
runtime="${method_root}/infer_native_role_reversed_owner_identity_rebinding_canary.py"
repo_launcher="${method_root}/scripts/auh_infer_native_role_reversed_owner_identity_rebinding_dual4.sbatch"
running_launcher="$(realpath -e -- "$0")"
repo_closure=(
  "${runtime}"
  "${method_root}/materialize_self_imagined_owner_core2_v1.py"
  "${method_root}/generate_self_imagined_owner_core2_v1.py"
  "${method_root}/self_imagined_motion_cotangent_v1.py"
  "${method_root}/infer_native_action_scaffold_identity_rebinding_canary.py"
  "${method_root}/infer_native_multivideo_motion_donor_oracle.py"
  "${method_root}/infer_native_identity_generation_canary.py"
  "${method_root}/tri_branch_unipc.py"
  "${method_root}/infer_lora.py"
  "${method_root}/train_lora.py"
  "${method_root}/infer_source_kv_carrier_oracle.py"
  "${method_root}/infer_source_value_residual_oracle.py"
  "${method_root}/source_kv_replay.py"
  "${method_root}/source_kv_route_batches.py"
  "${method_root}/source_value_residual.py"
  "${method_root}/tests/test_infer_native_role_reversed_owner_identity_rebinding_canary.py"
  "${method_root}/tests/test_auh_native_role_reversed_owner_identity_rebinding_launcher.py"
  "${method_root}/tests/test_materialize_self_imagined_owner_core2_v1.py"
  "${repo_launcher}"
)
closure=("${repo_closure[@]}" "${registry}")
for path in "${closure[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "source closure file missing: ${path}"
done
[[ "$(sha256sum "${running_launcher}" | awk '{print $1}')" == "$(sha256sum "${repo_launcher}" | awk '{print $1}')" ]] || fail "running launcher differs from repository launcher"

source_revision="$(git -C "${repo_root}" rev-parse HEAD)"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive/repository revision differs"
for path in "${repo_closure[@]}"; do
  relative="${path#"${repo_root}/"}"
  [[ "${relative}" != "${path}" ]] || fail "repository closure path escaped"
  git -C "${repo_root}" ls-files --error-unmatch -- "${relative}" >/dev/null || fail "repository closure path is untracked: ${relative}"
done
git -C "${repo_root}" diff --quiet -- "${repo_closure[@]}" || fail "runtime closure has unstaged edits"
git -C "${repo_root}" diff --cached --quiet -- "${repo_closure[@]}" || fail "runtime closure has staged edits"
launcher_sha256="$(sha256sum "${repo_launcher}" | awk '{print $1}')"
source_closure_sha256="$(sha256sum "${closure[@]}" | sha256sum | awk '{print $1}')"

[[ "${output_root}" != / && ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"
output_parent="$(dirname -- "${output_root}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"
output_parent="$(realpath -e -- "${output_parent}")"
[[ "${output_root}" == "${output_parent}/${output_root##*/}" ]] || fail "output path is not canonical"
[[ "${output_root##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "unsafe output basename"

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
export NCCL_DEBUG=WARN

for test_file in \
  test_infer_native_role_reversed_owner_identity_rebinding_canary.py \
  test_auh_native_role_reversed_owner_identity_rebinding_launcher.py \
  test_materialize_self_imagined_owner_core2_v1.py
do
  PYTHONPATH="${method_root}" "${python_bin}" -B "${method_root}/tests/${test_file}"
done

[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "Slurm ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/role-reversed-owner-${SLURM_JOB_ID}.XXXXXX")"
dog_pid=""
human_pid=""

cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/role-reversed-owner-${SLURM_JOB_ID}."*) ;;
    *) exit 2 ;;
  esac
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}"
    rm -rf -- "${task_scratch}"
  fi
  [[ ! -e "${task_scratch}" ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[role-reversed-owner] PASS all8=true exact81=true exact40=true training=false output=${output_root}"
  fi
  exit "${status}"
}

terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${dog_pid}" "${human_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then kill "${child}" 2>/dev/null || true; fi
  done
  for child in "${dog_pid}" "${human_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then wait "${child}" 2>/dev/null || true; fi
  done
  exit "${status}"
}

trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -- "${output_root}"
mkdir -- "${output_root}/logs"
cp -- "${registry}" "${output_root}/sealed-self-imagined-core2.json"
chmod 0444 -- "${output_root}/sealed-self-imagined-core2.json"

job_mod=$((SLURM_JOB_ID % 10000))
dog_port=$((40000 + 2 * job_mod))
human_port=$((dog_port + 1))

launch_group() (
  set -Eeuo pipefail
  local label="$1"
  local visible="$2"
  local port="$3"
  local group_scratch="${task_scratch}/${label}"
  mkdir -p -- "${group_scratch}/miopen-user" "${group_scratch}/miopen-custom" "${group_scratch}/torch-extensions" "${group_scratch}/triton" "${group_scratch}/xdg" "${group_scratch}/pycache" "${group_scratch}/tmp"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible}"
  export MIOPEN_USER_DB_PATH="${group_scratch}/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${group_scratch}/miopen-custom"
  export TORCH_EXTENSIONS_DIR="${group_scratch}/torch-extensions"
  export TRITON_CACHE_DIR="${group_scratch}/triton"
  export XDG_CACHE_HOME="${group_scratch}/xdg"
  export PYTHONPYCACHEPREFIX="${group_scratch}/pycache"
  export TMPDIR="${group_scratch}/tmp"
  export BERNINI_OUTPUT_TRANSACTION_ID="role-reversed-${SLURM_JOB_ID}-${label}"
  exec "${python_bin}" -B -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port="${port}" \
    "${runtime}" \
    --registry "${output_root}/sealed-self-imagined-core2.json" \
    --expected-registry-sha256 "${registry_sha256}" \
    --owner-root "${owner_root}" \
    --owner-master-receipt "${owner_master}" \
    --expected-owner-master-receipt-sha256 "${owner_master_sha256}" \
    --audit-sidecar "${audit_sidecar}" \
    --expected-audit-sidecar-sha256 "${audit_sidecar_sha256}" \
    --audit-evidence "${audit_evidence}" \
    --audit-public-key "${audit_public_key}" \
    --expected-audit-public-key-sha256 "${audit_public_key_sha256}" \
    --cell-id "${label}" \
    --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --output-dir "${output_root}/${label}" \
    --num-inference-steps 40 \
    --runtime-source-revision "${source_revision}" \
    --runtime-source-archive-sha256 "${source_archive_sha256}" \
    --launcher-source-sha256 "${launcher_sha256}" \
    --experimental-owner-primal-ack frozen_role_reversed_owner_primal_diagnostic_only
)

launch_group dog 0,1,2,3 "${dog_port}" >"${output_root}/logs/dog.log" 2>&1 &
dog_pid=$!
launch_group human 4,5,6,7 "${human_port}" >"${output_root}/logs/human.log" 2>&1 &
human_pid=$!

dog_status=0
human_status=0
wait "${dog_pid}" || dog_status=$?
wait "${human_pid}" || human_status=$?
[[ "${dog_status}" == 0 ]] || fail "dog SP4 failed with ${dog_status}"
[[ "${human_status}" == 0 ]] || fail "human SP4 failed with ${human_status}"

"${python_bin}" -B - "${output_root}" "${registry_sha256}" "${source_revision}" "${source_archive_sha256}" "${source_closure_sha256}" <<'PY'
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

root = Path(sys.argv[1]).resolve(strict=True)
registry_sha, revision, archive_sha, closure_sha = sys.argv[2:]

def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def object_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()

children = []
for cell in ("dog", "human"):
    path = root / cell / "receipt.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(row)
    declared = unsigned.pop("receipt_digest")
    if object_sha(unsigned) != declared:
        raise SystemExit(f"{cell} receipt seal differs")
    if row.get("cell_id") != cell or row.get("arms_in_order") != ["owner-only", "owner-source-refs", "source-source-refs"]:
        raise SystemExit(f"{cell} receipt contract differs")
    matched = row.get("matched_target", {})
    if matched.get("frame_count") != 81 or matched.get("num_inference_steps") != 40:
        raise SystemExit(f"{cell} exact81/exact40 differs")
    interpretation = row.get("interpretation", {})
    if interpretation.get("training_performed") is not False or interpretation.get("parameter_update") is not False:
        raise SystemExit(f"{cell} frozen boundary differs")
    children.append({
        "cell_id": cell,
        "receipt_path": str(path),
        "receipt_file_sha256": file_sha(path),
        "receipt_digest": declared,
        "output_mp4_sha256": {arm: row["outputs"][arm]["sha256"] for arm in row["arms_in_order"]},
    })

receipt = {
    "schema_version": "bernini-role-reversed-owner-identity-rebinding-all8-receipt-v1",
    "method": "frozen-bernini-role-reversed-owner-identity-rebinding-canary",
    "topology": "two_concurrent_world4_sp4_groups_on_one_8gpu_node",
    "all8_used": True,
    "cell_order": ["dog", "human"],
    "registry_file_sha256": registry_sha,
    "runtime_source_revision": revision,
    "runtime_source_archive_sha256": archive_sha,
    "runtime_source_closure_sha256": closure_sha,
    "exact81": True,
    "exact40": True,
    "training_performed": False,
    "parameter_update": False,
    "pseudo_target_distillation_performed": False,
    "scientific_or_action_editing_claim_authorized": False,
    "children": children,
}
receipt["receipt_digest"] = object_sha(receipt)
target = root / "all8-receipt.json"
with tempfile.NamedTemporaryFile(dir=root, prefix=".all8-receipt.", suffix=".tmp", delete=False) as handle:
    temporary = Path(handle.name)
    handle.write(json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
PY
