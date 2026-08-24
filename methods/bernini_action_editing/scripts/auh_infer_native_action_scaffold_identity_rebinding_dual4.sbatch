#!/usr/bin/env bash
#SBATCH --job-name=bernini-rebind-2x4
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=12:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# One frozen role-rebinding stage.  GPUs 0-3 render the fixed first four arms;
# GPUs 4-7 render the fixed last four.  Both isolated WORLD4/Ulysses4 groups
# bind the same sealed sp4-a proposal cell.  The login submitter schedules a
# one-step engineering stage and exact40 with afterok:<one-step>.

source_archive="${ROLE_REBINDING_SOURCE_ARCHIVE:?set ROLE_REBINDING_SOURCE_ARCHIVE}"
source_archive_sha256="${ROLE_REBINDING_SOURCE_ARCHIVE_SHA256:?set ROLE_REBINDING_SOURCE_ARCHIVE_SHA256}"
source_revision="${ROLE_REBINDING_SOURCE_REVISION:?set ROLE_REBINDING_SOURCE_REVISION}"
manifest="${ROLE_REBINDING_FACTOR_MANIFEST:?set ROLE_REBINDING_FACTOR_MANIFEST}"
manifest_sha256="${ROLE_REBINDING_FACTOR_MANIFEST_SHA256:?set ROLE_REBINDING_FACTOR_MANIFEST_SHA256}"
bank_receipt="${ROLE_REBINDING_BANK_RECEIPT:?set ROLE_REBINDING_BANK_RECEIPT}"
bank_receipt_sha256="${ROLE_REBINDING_BANK_RECEIPT_SHA256:?set ROLE_REBINDING_BANK_RECEIPT_SHA256}"
bank_output_root="${ROLE_REBINDING_BANK_OUTPUT_ROOT:?set ROLE_REBINDING_BANK_OUTPUT_ROOT}"
source_video="${ROLE_REBINDING_SOURCE_VIDEO:?set ROLE_REBINDING_SOURCE_VIDEO}"
wrong_source_video="${ROLE_REBINDING_WRONG_SOURCE_VIDEO:?set ROLE_REBINDING_WRONG_SOURCE_VIDEO}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
output_dir="${ROLE_REBINDING_OUTPUT_DIR:?set ROLE_REBINDING_OUTPUT_DIR}"
python_bin="${ROLE_REBINDING_PYTHON_BIN:?set ROLE_REBINDING_PYTHON_BIN}"
num_inference_steps="${ROLE_REBINDING_NUM_INFERENCE_STEPS:?set ROLE_REBINDING_NUM_INFERENCE_STEPS to 1 or 40}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
cdf_dog_source_sha256="5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
cdf_dog_wrong_source_sha256="da7e3efa6f4fabac1f1c57b9376667366ca2ad43d4710adea5892eb313cc5e7a"
group_a_visible_gpus="0,1,2,3"
group_b_visible_gpus="4,5,6,7"

fail() {
  echo "[native-role-rebinding] ERROR: $*" >&2
  exit 2
}

require_absolute() {
  local value="$1"
  local label="$2"
  [[ "${value}" == /* ]] || fail "${label} must be absolute"
}

for value_name in \
  source_archive manifest bank_receipt bank_output_root source_video wrong_source_video \
  bernini_root veomni_root checkpoint checkpoint_manifest output_dir python_bin
do
  require_absolute "${!value_name}" "${value_name}"
done
[[ "${num_inference_steps}" == "1" || "${num_inference_steps}" == "40" ]] || fail "steps must be exactly 1 or 40"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "archive SHA-256 must be lowercase"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be full lowercase SHA-1"
[[ "${manifest_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "manifest SHA-256 must be lowercase"
[[ "${bank_receipt_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "bank receipt SHA-256 must be lowercase"
[[ ! -L "${source_video}" ]] || fail "source video input must not be a symlink"
[[ ! -L "${wrong_source_video}" ]] || fail "wrong-source video input must not be a symlink"

source_archive="$(realpath -e -- "${source_archive}")"
manifest="$(realpath -e -- "${manifest}")"
bank_receipt="$(realpath -e -- "${bank_receipt}")"
bank_output_root="$(realpath -e -- "${bank_output_root}")"
source_video="$(realpath -e -- "${source_video}")"
wrong_source_video="$(realpath -e -- "${wrong_source_video}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
launcher_path="$(realpath -e -- "$0")"

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "method archive is not plain"
[[ -f "${manifest}" && ! -L "${manifest}" ]] || fail "factor manifest is not plain"
[[ -f "${bank_receipt}" && ! -L "${bank_receipt}" ]] || fail "factor bank receipt is not plain"
[[ -d "${bank_output_root}" && ! -L "${bank_output_root}" ]] || fail "bank output root is invalid"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "source video is not plain"
[[ -f "${wrong_source_video}" && ! -L "${wrong_source_video}" ]] || fail "wrong-source video is not plain"
[[ "${wrong_source_video}" != "${source_video}" ]] || fail "correct and wrong source videos alias"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini root is invalid"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni root is invalid"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint is invalid"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest is invalid"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "Python runtime is invalid"
[[ -f "${launcher_path}" && ! -L "${launcher_path}" ]] || fail "launcher is not plain"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "archive revision differs"
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha256}" ]] || fail "manifest hash differs"
[[ "$(sha256sum "${bank_receipt}" | awk '{print $1}')" == "${bank_receipt_sha256}" ]] || fail "bank receipt hash differs"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${cdf_dog_source_sha256}" ]] || fail "source is not pinned CDF dog"
[[ "$(sha256sum "${wrong_source_video}" | awk '{print $1}')" == "${cdf_dog_wrong_source_sha256}" ]] || fail "wrong source is not pinned exact81 identity control"

[[ "${output_dir}" != / ]] || fail "output cannot be filesystem root"
output_basename="${output_dir##*/}"
[[ "${output_basename}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename is unsafe"
[[ "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output must be canonical"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing reused output"
output_parent="$(dirname -- "${output_dir}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent must exist"
[[ "$(realpath -e -- "${output_parent}")" == "${output_parent}" ]] || fail "output parent is noncanonical"

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
task_scratch="$(mktemp -d "${scratch_parent%/}/native-role-rebinding-${SLURM_JOB_ID}.XXXXXX")"
group_a_pid=""
group_b_pid=""

cleanup() {
  local exit_status=$?
  local cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/native-role-rebinding-${SLURM_JOB_ID}."*) ;;
    *) echo "[native-role-rebinding] refusing unsafe scratch cleanup" >&2; cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  if [[ "${cleanup_failed}" != 0 && "${exit_status}" == 0 ]]; then exit_status=2; fi
  if [[ "${exit_status}" == 0 ]]; then
    echo "[native-role-rebinding] PASS steps=${num_inference_steps} topology=two-isolated-WORLD4 output=${output_dir}"
  else
    echo "[native-role-rebinding] EXIT status=${exit_status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${exit_status}"
}

terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${group_a_pid}" "${group_b_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then kill "${child}" 2>/dev/null || true; fi
  done
  for child in "${group_a_pid}" "${group_b_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then wait "${child}" 2>/dev/null || true; fi
  done
  exit "${status}"
}

trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -p -- "${task_scratch}/source-tree" "${task_scratch}/groups"
archive_copy="${task_scratch}/method-source.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod 0444 -- "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged archive hash differs"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

required = {
    "methods/bernini_action_editing/infer_native_action_scaffold_identity_rebinding_canary.py",
    "methods/bernini_action_editing/infer_native_multivideo_motion_donor_oracle.py",
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
    "methods/bernini_action_editing/dmiq_t2v_factorial_bank.py",
    "methods/bernini_action_editing/infer_fitq_owner_prompt_cross_query_micro.py",
    "methods/bernini_action_editing/tri_branch_unipc.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
    "methods/bernini_action_editing/source_kv_replay.py",
    "methods/bernini_action_editing/source_kv_route_batches.py",
    "methods/bernini_action_editing/source_value_residual.py",
    "methods/bernini_action_editing/scripts/auh_infer_native_action_scaffold_identity_rebinding_dual4.sbatch",
    "methods/bernini_action_editing/scripts/auh_submit_native_action_scaffold_identity_rebinding_chain.sh",
}
seen = set()
with tarfile.open(sys.argv[1], "r:*") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit("archive member escaped repository")
        normalized = path.as_posix().lstrip("./")
        if normalized.startswith("methods/bernini_action_editing/"):
            if member.issym() or member.islnk() or member.isfifo() or member.isdev():
                raise SystemExit("method closure contains a link or device")
            seen.add(normalized)
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"archive lacks role-rebinding closure: {missing}")
PY

tar --no-same-owner --no-same-permissions \
  -xf "${archive_copy}" -C "${task_scratch}/source-tree" \
  methods/bernini_action_editing
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
runtime_path="${method_root}/infer_native_action_scaffold_identity_rebinding_canary.py"
staged_launcher="${method_root}/scripts/auh_infer_native_action_scaffold_identity_rebinding_dual4.sbatch"
[[ -f "${runtime_path}" && ! -L "${runtime_path}" ]] || fail "staged runtime is absent"
[[ -f "${staged_launcher}" && ! -L "${staged_launcher}" ]] || fail "staged launcher is absent"
find "${method_root}" -type f -exec chmod a-w -- {} +
launcher_source_sha256="$(sha256sum "${staged_launcher}" | awk '{print $1}')"
[[ "$(sha256sum "${launcher_path}" | awk '{print $1}')" == "${launcher_source_sha256}" ]] || fail "executed launcher differs from archive"

mkdir -- "${output_dir}"
mkdir -- "${output_dir}/logs"
cp -- "${manifest}" "${output_dir}/factor-manifest.json"
cp -- "${bank_receipt}" "${output_dir}/factor-bank.receipt.json"
chmod 0444 -- "${output_dir}/factor-manifest.json" "${output_dir}/factor-bank.receipt.json"

job_mod=$((SLURM_JOB_ID % 10000))
group_a_master_port=$((30000 + 2 * job_mod))
group_b_master_port=$((group_a_master_port + 1))

launch_group() (
  set -Eeuo pipefail
  local group="$1"
  local visible_gpus="$2"
  local master_port="$3"
  shift 3
  local group_root="${task_scratch}/groups/${group}"
  local group_output="${output_dir}/${group}"
  mkdir -p -- \
    "${group_root}/cache/miopen-user" "${group_root}/cache/miopen-custom" \
    "${group_root}/cache/torch-extensions" "${group_root}/cache/triton" \
    "${group_root}/cache/xdg" "${group_root}/cache/pycache" "${group_root}/tmp"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible_gpus}"
  export MIOPEN_USER_DB_PATH="${group_root}/cache/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${group_root}/cache/miopen-custom"
  export TORCH_EXTENSIONS_DIR="${group_root}/cache/torch-extensions"
  export TRITON_CACHE_DIR="${group_root}/cache/triton"
  export XDG_CACHE_HOME="${group_root}/cache/xdg"
  export PYTHONPYCACHEPREFIX="${group_root}/cache/pycache"
  export TMPDIR="${group_root}/tmp"
  export TORCHELASTIC_ERROR_FILE="${group_root}/torch-elastic.json"
  export BERNINI_OUTPUT_TRANSACTION_ID="${SLURM_JOB_ID}-${group}-native-role-rebinding-s${num_inference_steps}"

  echo "[native-role-rebinding] group=${group} steps=${num_inference_steps} arms=$* factor_cell=sp4-a"
  "${python_bin}" -B -m torch.distributed.run \
    --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="${master_port}" \
    "${runtime_path}" \
    --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --source-video "${source_video}" \
    --wrong-source-video "${wrong_source_video}" \
    --factor-manifest "${output_dir}/factor-manifest.json" \
    --expected-factor-manifest-file-sha256 "${manifest_sha256}" \
    --factor-bank-receipt "${output_dir}/factor-bank.receipt.json" \
    --expected-factor-bank-receipt-file-sha256 "${bank_receipt_sha256}" \
    --bank-output-root "${bank_output_root}" \
    --factor-execution-group sp4-a \
    --arm-group "${group}" \
    --arms "$@" \
    --num-inference-steps "${num_inference_steps}" \
    --output-dir "${group_output}" \
    --runtime-source-revision "${source_revision}" \
    --runtime-source-archive-sha256 "${source_archive_sha256}" \
    --launcher-source-sha256 "${launcher_source_sha256}" \
    --expected-bernini-commit "${bernini_commit}" \
    --expected-veomni-commit "${veomni_commit}" \
    --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}"
)

group_a_log="${output_dir}/logs/group-a.log"
group_b_log="${output_dir}/logs/group-b.log"
echo "[native-role-rebinding] semantic_inputs=source_exact81+correct_and_wrong_source_rgb_refs+registered_predecode_proposal+source_specific_prompt wrong_target=false wrong_parquet=false wrong_latent=false target=false proposal_mp4=false mask=false flow=false pose=false track=false trajectory=false optimization=false"
launch_group group-a "${group_a_visible_gpus}" "${group_a_master_port}" \
  source-video-source-refs action-donor-source-refs \
  noop-donor-source-refs reverse-donor-source-refs >"${group_a_log}" 2>&1 &
group_a_pid=$!
launch_group group-b "${group_b_visible_gpus}" "${group_b_master_port}" \
  action-donor-only source-action-source-refs \
  action-source-source-refs action-donor-wrong-refs >"${group_b_log}" 2>&1 &
group_b_pid=$!

group_a_status=0
group_b_status=0
wait "${group_a_pid}" || group_a_status=$?
wait "${group_b_pid}" || group_b_status=$?
if [[ "${group_a_status}" -ne 0 || "${group_b_status}" -ne 0 ]]; then
  tail -n 220 -- "${group_a_log}" >&2 || true
  tail -n 220 -- "${group_b_log}" >&2 || true
  fail "one or more role-rebinding WORLD4 groups failed"
fi

"${python_bin}" -B - "${output_dir}" "${num_inference_steps}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

root = Path(sys.argv[1]).resolve(strict=True)
steps = int(sys.argv[2])
expected_groups = {
    "group-a": [
        "source-video-source-refs", "action-donor-source-refs",
        "noop-donor-source-refs", "reverse-donor-source-refs",
    ],
    "group-b": [
        "action-donor-only", "source-action-source-refs",
        "action-source-source-refs", "action-donor-wrong-refs",
    ],
}
global_arms = expected_groups["group-a"] + expected_groups["group-b"]

def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def checked_file(value, group_root, label):
    if not isinstance(value, dict):
        raise SystemExit(f"{label} receipt is absent")
    path = Path(value.get("path", ""))
    declared = value.get("sha256")
    if (
        not path.is_absolute() or path.is_symlink()
        or path.resolve(strict=True) != path or not path.is_file()
        or path.parent != group_root or not isinstance(declared, str)
        or len(declared) != 64 or file_sha256(path) != declared
    ):
        raise SystemExit(f"{label} artifact binding differs")
    return path

receipts = {}
group_rows = {}
for group, expected_arms in expected_groups.items():
    group_root = (root / group).resolve(strict=True)
    if group_root.parent != root or group_root.is_symlink() or not group_root.is_dir():
        raise SystemExit(f"{group} root is invalid")
    receipt_path = group_root / "receipt.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise SystemExit(f"{group} receipt is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    if hashlib.sha256(canonical(unsigned)).hexdigest() != declared:
        raise SystemExit(f"{group} receipt digest differs")
    if (
        receipt.get("schema_version")
        != "bernini-native-action-scaffold-identity-rebinding-receipt-v3"
        or receipt.get("arm_group") != group
        or receipt.get("arms_in_execution_order") != expected_arms
        or receipt.get("fixed_global_arm_order") != global_arms
        or receipt.get("matched_target", {}).get("num_inference_steps") != steps
        or receipt.get("matched_target", {}).get("same_target_gaussian_all_arms_in_group") is not True
        or receipt.get("source", {}).get("reference_indices") != [0, 27, 53, 80]
        or receipt.get("source", {}).get("video_sha256")
        != "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
        or receipt.get("source", {}).get("references_independently_vae_encoded_from_rgb") is not True
        or receipt.get("source", {}).get("references_sliced_from_full_video_latent") is not False
        or receipt.get("factor_bank", {}).get("fixed_execution_group") != "sp4-a"
        or receipt.get("factor_bank", {}).get("all_proposals_same_registered_cell") is not True
        or receipt.get("factor_bank", {}).get("proposal_mp4_consumed") is not False
        or receipt.get("wrong_source", {}).get("reference_indices") != [0, 27, 53, 80]
        or receipt.get("wrong_source", {}).get("video_sha256")
        != "da7e3efa6f4fabac1f1c57b9376667366ca2ad43d4710adea5892eb313cc5e7a"
        or receipt.get("wrong_source", {}).get("references_independently_vae_encoded_from_rgb") is not True
        or receipt.get("wrong_source", {}).get("full_video_vae_encode_performed") is not False
        or receipt.get("wrong_source", {}).get("references_sliced_from_full_video_latent") is not False
        or receipt.get("wrong_source", {}).get("source_video_only") is not True
        or receipt.get("wrong_source", {}).get("paired_target_accessed") is not False
        or receipt.get("wrong_source", {}).get("paired_parquet_accessed") is not False
        or receipt.get("wrong_source", {}).get("precomputed_latent_accessed") is not False
        or receipt.get("prompt", {}).get("same_prompt_all_eight_arms") is not True
        or receipt.get("checkpoint", {}).get("unchanged") is not True
        or receipt.get("interpretation", {}).get("training_performed") is not False
        or receipt.get("interpretation", {}).get("quality_claim") is not False
        or receipt.get("interpretation", {}).get("scientific_claim_authorized") is not False
        or receipt.get("interpretation", {}).get("initial_gaussian_observer_changed_return_object") is not False
    ):
        raise SystemExit(f"{group} semantic receipt contract differs")
    if set(receipt.get("arms", {})) != set(expected_arms):
        raise SystemExit(f"{group} arm receipt set differs")
    noise_hash = receipt.get("matched_target", {}).get("target_gaussian_raw_storage_sha256")
    if not isinstance(noise_hash, str) or len(noise_hash) != 64:
        raise SystemExit(f"{group} target Gaussian hash is invalid")
    checked_file(
        receipt["source"]["normalized_clean_latent_artifact"],
        group_root, f"{group} source clean latent",
    )
    reference_artifacts = receipt["source"].get("reference_artifacts", {})
    if set(reference_artifacts) != {"0", "27", "53", "80"}:
        raise SystemExit(f"{group} source reference artifact set differs")
    for frame_index, artifact in reference_artifacts.items():
        checked_file(
            artifact, group_root,
            f"{group} source reference {frame_index}",
        )
        identity = receipt["source"]["reference_identities"][frame_index]
        if (
            artifact.get("frame_index") != int(frame_index)
            or artifact.get("raw_storage_sha256")
            != identity.get("identity", {}).get("raw_storage_sha256")
            or artifact.get("reference_video_role") != "source_video"
            or artifact.get("independent_rgb_frame_vae_encode") is not True
            or artifact.get("temporal_video_latent_slice") is not False
        ):
            raise SystemExit(f"{group} source reference {frame_index} identity differs")
    wrong_reference_artifacts = receipt["wrong_source"].get("reference_artifacts", {})
    if set(wrong_reference_artifacts) != {"0", "27", "53", "80"}:
        raise SystemExit(f"{group} wrong-source reference artifact set differs")
    for frame_index, artifact in wrong_reference_artifacts.items():
        checked_file(
            artifact, group_root,
            f"{group} wrong-source reference {frame_index}",
        )
        identity = receipt["wrong_source"]["reference_identities"][frame_index]
        if (
            artifact.get("frame_index") != int(frame_index)
            or artifact.get("raw_storage_sha256")
            != identity.get("identity", {}).get("raw_storage_sha256")
            or artifact.get("reference_video_role") != "wrong_source_video"
            or artifact.get("independent_rgb_frame_vae_encode") is not True
            or artifact.get("temporal_video_latent_slice") is not False
        ):
            raise SystemExit(f"{group} wrong-source reference {frame_index} identity differs")
    for arm in expected_arms:
        arm_receipt = receipt["arms"][arm]
        if (
            arm_receipt.get("guidance_mode") != "rv2v"
            or arm_receipt.get("factor_label_independently_verified") is not False
            or arm_receipt.get("audit", {}).get("observer_modified_numerics") is not False
        ):
            raise SystemExit(f"{group}/{arm} native audit differs")
        source_ids = arm_receipt.get("conditioning", {}).get("source_ids", {})
        if (
            source_ids.get("native_interpolate_src_id") is not True
            or source_ids.get("native_max_trained_source_id") != 5
            or source_ids.get("all_patch_source_ids_within_trained_interval_0_through_5") is not True
            or source_ids.get("conditioning_source_id_extrapolation_used") is not False
            or arm_receipt.get("audit", {}).get("source_id_order_per_step")
            != source_ids.get("patch_source_id_order_per_step")
        ):
            raise SystemExit(f"{group}/{arm} native source-ID audit differs")
        if arm == "action-donor-wrong-refs" and (
            arm_receipt.get("reference_video_role") != "wrong_source_video"
            or source_ids.get("vi_video_source_ids") != [1.0]
            or source_ids.get("vi_reference_source_ids") != [2.0, 3.0, 4.0, 5.0]
            or source_ids.get("image_only_reference_source_ids") != [1.0, 2.0, 3.0, 4.0]
            or source_ids.get("native_source_id_interpolation_used") is not False
        ):
            raise SystemExit(f"{group}/{arm} wrong-source/source-ID contract differs")
        if arm in {"source-action-source-refs", "action-source-source-refs"} and (
            arm_receipt.get("reference_video_role") != "source_video"
            or source_ids.get("vi_video_source_ids")
            != [1.0, 1.7999999523162842]
            or source_ids.get("vi_reference_source_ids")
            != [2.5999999046325684, 3.4000000953674316, 4.199999809265137, 5.0]
            or source_ids.get("image_only_reference_source_ids") != [1.0, 2.0, 3.0, 4.0]
            or source_ids.get("native_source_id_interpolation_used") is not True
        ):
            raise SystemExit(f"{group}/{arm} native source-ID interpolation receipt differs")
        noise = receipt["initial_noise_artifacts"][arm]
        checked_file(noise, group_root, f"{group}/{arm} noise")
        if (
            noise.get("raw_value_sha256") != noise_hash
            or noise.get("observer_changed_return_value") is not False
            or noise.get("original_return_tensor_forwarded_by_identity") is not True
        ):
            raise SystemExit(f"{group}/{arm} Gaussian observer differs")
        output = receipt["outputs"][arm]
        clean = output.get("normalized_clean_latent")
        checked_file(clean, group_root, f"{group}/{arm} clean latent")
        if steps == 40:
            checked_file(
                {"path": output.get("path"), "sha256": output.get("sha256")},
                group_root, f"{group}/{arm} mp4",
            )
            if output.get("frame_count") != 81 or output.get("fps") != 25:
                raise SystemExit(f"{group}/{arm} exact81 output differs")
        elif output.get("mp4") is not None or output.get("decode_performed") is not False:
            raise SystemExit(f"{group}/{arm} one-step decode contract differs")
    receipts[group] = receipt
    group_rows[group] = {
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": file_sha256(receipt_path),
        "receipt_digest": declared,
        "arms": expected_arms,
        "target_gaussian_raw_storage_sha256": noise_hash,
    }

left = receipts["group-a"]
right = receipts["group-b"]
for label, getter in {
    "runtime archive": lambda r: r["runtime_source"]["archive_sha256"],
    "checkpoint tree": lambda r: r["checkpoint"]["tree_sha256"],
    "factor manifest": lambda r: r["factor_bank"]["manifest_digest"],
    "factor receipt": lambda r: r["factor_bank"]["bank_receipt_digest"],
    "proposal cell": lambda r: r["factor_bank"]["proposal_cell"],
    "registered proposal tensors": lambda r: r["proposal_condition_all_rank_identities"],
    "source video": lambda r: r["source"]["video_sha256"],
    "source video latent": lambda r: r["source"]["full_video_identity"],
    "correct source references": lambda r: r["source"]["reference_identities"],
    "wrong source video": lambda r: r["wrong_source"]["video_sha256"],
    "wrong source references": lambda r: r["wrong_source"]["reference_identities"],
    "prompt": lambda r: r["prompt"]["full_prompt_utf8_sha256"],
    "target Gaussian": lambda r: r["matched_target"]["target_gaussian_raw_storage_sha256"],
}.items():
    if getter(left) != getter(right):
        raise SystemExit(f"cross-group {label} differs")

correct_action = receipts["group-a"]["arms"]["action-donor-source-refs"]
wrong_action = receipts["group-b"]["arms"]["action-donor-wrong-refs"]
if (
    correct_action["conditioning"]["video_latent_raw_storage_sha256_in_order"]
    != wrong_action["conditioning"]["video_latent_raw_storage_sha256_in_order"]
    or correct_action["conditioning"]["source_reference_raw_storage_sha256_in_order"]
    == wrong_action["conditioning"]["source_reference_raw_storage_sha256_in_order"]
):
    raise SystemExit("correct/wrong reference causal pair binding differs")

stage = {
    "schema_version": "bernini-native-action-scaffold-identity-rebinding-stage-v3",
    "complete": True,
    "num_inference_steps": steps,
    "frame_count": 81,
    "fps": 25,
    "topology": "two-isolated-WORLD4-Ulysses4",
    "gpu_slots": {"group-a": [0, 1, 2, 3], "group-b": [4, 5, 6, 7]},
    "arms_in_global_order": global_arms,
    "groups": group_rows,
    "same_sp4_a_proposal_cell_both_groups": True,
    "same_prompt_both_groups": True,
    "same_official_target_gaussian_all_eight_arms": True,
    "same_correct_source_wrong_source_proposals_and_prompt_both_groups": True,
    "wrong_reference_arm_vi_source_ids": [1.0, 2.0, 3.0, 4.0, 5.0],
    "two_video_correct_reference_native_vi_source_id_interpolation": [
        1.0, 1.7999999523162842, 2.5999999046325684,
        3.4000000953674316, 4.199999809265137, 5.0,
    ],
    "conditioning_source_id_extrapolation_used": False,
    "wrong_source_target_parquet_or_precomputed_latent_accessed": False,
    "target_video": False,
    "proposal_mp4_consumed": False,
    "mask_flow_pose_track_trajectory": False,
    "training_or_optimization": False,
    "quality_claim": False,
}
stage["receipt_digest"] = hashlib.sha256(canonical(stage)).hexdigest()
final = root / "stage.receipt.json"
with tempfile.NamedTemporaryFile(
    dir=root, prefix=".stage.receipt.", suffix=".tmp", delete=False
) as handle:
    temporary = Path(handle.name)
    handle.write(canonical(stage) + b"\n")
    handle.flush()
    os.fsync(handle.fileno())
try:
    os.replace(temporary, final)
finally:
    if temporary.exists():
        temporary.unlink()
descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
print(json.dumps(stage, sort_keys=True))
PY

[[ -f "${output_dir}/stage.receipt.json" && ! -L "${output_dir}/stage.receipt.json" ]] || fail "sealed stage receipt is absent"
echo "[native-role-rebinding] outputs=${output_dir}/group-a,${output_dir}/group-b receipt=${output_dir}/stage.receipt.json"
