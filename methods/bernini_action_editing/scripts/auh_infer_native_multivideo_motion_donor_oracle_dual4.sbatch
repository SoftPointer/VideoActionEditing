#!/usr/bin/env bash
#SBATCH --job-name=bernini-native-donor-2x4
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=16:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# One stage of the frozen native multi-video donor oracle.  The login-side
# submitter first schedules steps=1 and submits steps=40 with afterok:<canary>.
# Within either allocation the two complete registered donor cells run as
# isolated WORLD4/Ulysses4 groups on GPU slots 0-3 and 4-7.  There is no
# training, donor MP4 input, target, mask, flow, pose, track or trajectory.

source_archive="${MOTION_DONOR_SOURCE_ARCHIVE:?set MOTION_DONOR_SOURCE_ARCHIVE}"
source_archive_sha256="${MOTION_DONOR_SOURCE_ARCHIVE_SHA256:?set MOTION_DONOR_SOURCE_ARCHIVE_SHA256}"
source_revision="${MOTION_DONOR_SOURCE_REVISION:?set MOTION_DONOR_SOURCE_REVISION}"
manifest="${MOTION_DONOR_FACTOR_MANIFEST:?set MOTION_DONOR_FACTOR_MANIFEST}"
manifest_sha256="${MOTION_DONOR_FACTOR_MANIFEST_SHA256:?set MOTION_DONOR_FACTOR_MANIFEST_SHA256}"
bank_receipt="${MOTION_DONOR_BANK_RECEIPT:?set MOTION_DONOR_BANK_RECEIPT}"
bank_receipt_sha256="${MOTION_DONOR_BANK_RECEIPT_SHA256:?set MOTION_DONOR_BANK_RECEIPT_SHA256}"
bank_output_root="${MOTION_DONOR_BANK_OUTPUT_ROOT:?set MOTION_DONOR_BANK_OUTPUT_ROOT}"
source_video="${MOTION_DONOR_SOURCE_VIDEO:?set MOTION_DONOR_SOURCE_VIDEO}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
output_dir="${MOTION_DONOR_OUTPUT_DIR:?set MOTION_DONOR_OUTPUT_DIR}"
python_bin="${MOTION_DONOR_PYTHON_BIN:?set MOTION_DONOR_PYTHON_BIN}"
num_inference_steps="${MOTION_DONOR_NUM_INFERENCE_STEPS:?set MOTION_DONOR_NUM_INFERENCE_STEPS to 1 or 40}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
cdf_dog_source_sha256="5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
sp4_a_visible_gpus="0,1,2,3"
sp4_b_visible_gpus="4,5,6,7"

fail() {
  echo "[native-motion-donor] ERROR: $*" >&2
  exit 2
}

require_absolute() {
  local value="$1"
  local label="$2"
  [[ "${value}" == /* ]] || fail "${label} must be absolute"
}

for value_name in \
  source_archive manifest bank_receipt bank_output_root source_video \
  bernini_root veomni_root checkpoint checkpoint_manifest output_dir python_bin
do
  require_absolute "${!value_name}" "${value_name}"
done
[[ "${num_inference_steps}" == "1" || "${num_inference_steps}" == "40" ]] || fail "stage steps must be exactly 1 or 40"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source archive SHA-256 must be lowercase"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be full lowercase SHA-1"
[[ "${manifest_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "factor manifest SHA-256 must be lowercase"
[[ "${bank_receipt_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "bank receipt SHA-256 must be lowercase"

source_archive="$(realpath -e -- "${source_archive}")"
manifest="$(realpath -e -- "${manifest}")"
bank_receipt="$(realpath -e -- "${bank_receipt}")"
bank_output_root="$(realpath -e -- "${bank_output_root}")"
source_video="$(realpath -e -- "${source_video}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
launcher_path="$(realpath -e -- "$0")"

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "method archive is not plain"
[[ -f "${manifest}" && ! -L "${manifest}" ]] || fail "factor manifest is not plain"
[[ -f "${bank_receipt}" && ! -L "${bank_receipt}" ]] || fail "bank receipt is not plain"
[[ -d "${bank_output_root}" && ! -L "${bank_output_root}" ]] || fail "bank output root is invalid"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "source video is not plain"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini source root is invalid"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni source root is invalid"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint root is invalid"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest is not plain"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "runtime Python is invalid"
[[ -f "${launcher_path}" && ! -L "${launcher_path}" ]] || fail "launcher is not plain"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive hash differs"
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha256}" ]] || fail "factor manifest hash differs"
[[ "$(sha256sum "${bank_receipt}" | awk '{print $1}')" == "${bank_receipt_sha256}" ]] || fail "factor bank receipt hash differs"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${cdf_dog_source_sha256}" ]] || fail "source is not the pinned CDF dog video"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "method archive embedded revision differs"

[[ "${output_dir}" != / ]] || fail "output directory cannot be filesystem root"
output_basename="${output_dir##*/}"
[[ "${output_basename}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename is unsafe"
[[ "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output directory must be canonical"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing to reuse output directory"
output_parent="$(dirname -- "${output_dir}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "stage output parent must already exist"
[[ "$(realpath -e -- "${output_parent}")" == "${output_parent}" ]] || fail "stage output parent must be canonical"

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
task_scratch="$(mktemp -d "${scratch_parent%/}/native-motion-donor-${SLURM_JOB_ID}.XXXXXX")"
sp4_a_pid=""
sp4_b_pid=""

cleanup() {
  local exit_status=$?
  local cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/native-motion-donor-${SLURM_JOB_ID}."*) ;;
    *) echo "[native-motion-donor] refusing unsafe scratch cleanup" >&2; cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  if [[ "${cleanup_failed}" != 0 && "${exit_status}" == 0 ]]; then exit_status=2; fi
  if [[ "${exit_status}" == 0 ]]; then
    echo "[native-motion-donor] PASS steps=${num_inference_steps} topology=two-isolated-WORLD4 output=${output_dir}"
  else
    echo "[native-motion-donor] EXIT status=${exit_status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${exit_status}"
}

terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${sp4_a_pid}" "${sp4_b_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then kill "${child}" 2>/dev/null || true; fi
  done
  for child in "${sp4_a_pid}" "${sp4_b_pid}"; do
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
[[ "$(git get-tar-commit-id <"${archive_copy}")" == "${source_revision}" ]] || fail "staged archive revision differs"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

required = {
    "methods/bernini_action_editing/dmiq_t2v_factorial_bank.py",
    "methods/bernini_action_editing/infer_fitq_owner_prompt_cross_query_micro.py",
    "methods/bernini_action_editing/infer_native_multivideo_motion_donor_oracle.py",
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
    "methods/bernini_action_editing/inference_sigma_strata.py",
    "methods/bernini_action_editing/internal_temporal_quotient_observer.py",
    "methods/bernini_action_editing/tri_branch_unipc.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
    "methods/bernini_action_editing/source_kv_replay.py",
    "methods/bernini_action_editing/source_kv_route_batches.py",
    "methods/bernini_action_editing/source_value_residual.py",
    "methods/bernini_action_editing/scripts/auh_infer_native_multivideo_motion_donor_oracle_dual4.sbatch",
    "methods/bernini_action_editing/scripts/auh_submit_native_multivideo_motion_donor_oracle_chain.sh",
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
                raise SystemExit("Bernini method closure contains a link or device")
            seen.add(normalized)
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"archive lacks native motion-donor closure: {missing}")
PY

tar --no-same-owner --no-same-permissions \
  -xf "${archive_copy}" \
  -C "${task_scratch}/source-tree" \
  methods/bernini_action_editing
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
runtime_path="${method_root}/infer_native_multivideo_motion_donor_oracle.py"
staged_launcher="${method_root}/scripts/auh_infer_native_multivideo_motion_donor_oracle_dual4.sbatch"
[[ -f "${runtime_path}" && ! -L "${runtime_path}" ]] || fail "staged runtime is absent"
[[ -f "${staged_launcher}" && ! -L "${staged_launcher}" ]] || fail "staged launcher is absent"
find "${method_root}" -type f -exec chmod a-w -- {} +
launcher_source_sha256="$(sha256sum "${staged_launcher}" | awk '{print $1}')"
[[ "$(sha256sum "${launcher_path}" | awk '{print $1}')" == "${launcher_source_sha256}" ]] || fail "executed launcher differs from source archive"

mkdir -- "${output_dir}"
mkdir -- "${output_dir}/logs"
cp -- "${manifest}" "${output_dir}/factor-manifest.json"
cp -- "${bank_receipt}" "${output_dir}/factor-bank.receipt.json"
chmod 0444 -- "${output_dir}/factor-manifest.json" "${output_dir}/factor-bank.receipt.json"
[[ "$(sha256sum "${output_dir}/factor-manifest.json" | awk '{print $1}')" == "${manifest_sha256}" ]] || fail "persistent manifest hash differs"
[[ "$(sha256sum "${output_dir}/factor-bank.receipt.json" | awk '{print $1}')" == "${bank_receipt_sha256}" ]] || fail "persistent bank receipt hash differs"

job_mod=$((SLURM_JOB_ID % 10000))
sp4_a_master_port=$((30000 + 2 * job_mod))
sp4_b_master_port=$((sp4_a_master_port + 1))
[[ "${sp4_a_master_port}" -ne "${sp4_b_master_port}" ]] || fail "rendezvous ports collided"

launch_group() (
  set -Eeuo pipefail
  local group="$1"
  local visible_gpus="$2"
  local master_port="$3"
  local group_root="${task_scratch}/groups/${group}"
  local group_output="${output_dir}/${group}"
  mkdir -p -- \
    "${group_root}/cache/miopen-user" \
    "${group_root}/cache/miopen-custom" \
    "${group_root}/cache/torch-extensions" \
    "${group_root}/cache/triton" \
    "${group_root}/cache/xdg" \
    "${group_root}/cache/pycache" \
    "${group_root}/tmp"
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
  export BERNINI_OUTPUT_TRANSACTION_ID="${SLURM_JOB_ID}-${group}-native-motion-donor-s${num_inference_steps}"

  echo "[native-motion-donor] group=${group} steps=${num_inference_steps} arms=9 rv2v_donor_replicate=true"
  "${python_bin}" -B -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port="${master_port}" \
    "${runtime_path}" \
    --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --source-video "${source_video}" \
    --factor-manifest "${output_dir}/factor-manifest.json" \
    --expected-factor-manifest-file-sha256 "${manifest_sha256}" \
    --factor-bank-receipt "${output_dir}/factor-bank.receipt.json" \
    --expected-factor-bank-receipt-file-sha256 "${bank_receipt_sha256}" \
    --bank-output-root "${bank_output_root}" \
    --execution-group "${group}" \
    --num-inference-steps "${num_inference_steps}" \
    --output-dir "${group_output}" \
    --runtime-source-revision "${source_revision}" \
    --runtime-source-archive-sha256 "${source_archive_sha256}" \
    --launcher-source-sha256 "${launcher_source_sha256}" \
    --expected-bernini-commit "${bernini_commit}" \
    --expected-veomni-commit "${veomni_commit}" \
    --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}"
)

sp4_a_log="${output_dir}/logs/sp4-a.log"
sp4_b_log="${output_dir}/logs/sp4-b.log"
echo "[native-motion-donor] semantic_inputs=source_video+registered_predecode_clean_donors+action_prompt target=false image_condition=false donor_mp4=false mask=false flow=false pose=false track=false trajectory=false external_noise=false"
launch_group sp4-a "${sp4_a_visible_gpus}" "${sp4_a_master_port}" >"${sp4_a_log}" 2>&1 &
sp4_a_pid=$!
launch_group sp4-b "${sp4_b_visible_gpus}" "${sp4_b_master_port}" >"${sp4_b_log}" 2>&1 &
sp4_b_pid=$!

sp4_a_status=0
sp4_b_status=0
wait "${sp4_a_pid}" || sp4_a_status=$?
wait "${sp4_b_pid}" || sp4_b_status=$?
if [[ "${sp4_a_status}" -ne 0 || "${sp4_b_status}" -ne 0 ]]; then
  tail -n 200 -- "${sp4_a_log}" >&2 || true
  tail -n 200 -- "${sp4_b_log}" >&2 || true
  fail "one or more native donor WORLD4 groups failed"
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
expected_arms = [
    "O0", "Z0", "D-action", "D-noop", "D-reverse", "D-wrong-actor",
    "D-duplicate-source", "D-action-order-swap", "A-source-v2v-apg",
]

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

def bound_artifact(value, group_root, expected_suffix, label):
    if not isinstance(value, dict):
        raise SystemExit(f"{label} identity is absent")
    raw_path = value.get("path")
    expected_sha = value.get("sha256")
    if (
        not isinstance(raw_path, str)
        or not isinstance(expected_sha, str)
        or len(expected_sha) != 64
    ):
        raise SystemExit(f"{label} path/SHA contract differs")
    path = Path(raw_path)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or path.resolve(strict=True) != path
        or not path.is_file()
        or path.parent != group_root
        or path.suffix != expected_suffix
    ):
        raise SystemExit(f"{label} is not one canonical group-local plain file")
    if file_sha256(path) != expected_sha:
        raise SystemExit(f"{label} file SHA differs")
    return path

groups = []
for group in ("sp4-a", "sp4-b"):
    group_root = (root / group).resolve(strict=True)
    if group_root.parent != root or group_root.is_symlink() or not group_root.is_dir():
        raise SystemExit(f"{group} output root is invalid")
    path = group_root / "receipt.json"
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"{group} receipt is absent")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    if hashlib.sha256(canonical(unsigned)).hexdigest() != declared:
        raise SystemExit(f"{group} receipt digest differs")
    if (
        receipt.get("execution_group") != group
        or receipt.get("arms_in_execution_order") != expected_arms
        or receipt.get("matched_target", {}).get("num_inference_steps") != steps
        or receipt.get("matched_target", {}).get("same_target_gaussian_all_arms") is not True
        or receipt.get("z0_wrapper_parity", {}).get("byte_exact_fp32") is not True
        or receipt.get("factor_bank", {}).get("donor_mp4_consumed") is not False
        or receipt.get("interpretation", {}).get("training_performed") is not False
        or receipt.get("interpretation", {}).get("scientific_claim_authorized") is not False
    ):
        raise SystemExit(f"{group} final runtime closure differs")
    if steps == 1 and receipt.get("stage") != "engineering_oom_callpath_canary":
        raise SystemExit(f"{group} one-step stage differs")
    if steps == 40 and receipt.get("stage") != "matched_exact40_qualitative_causal_pilot":
        raise SystemExit(f"{group} exact40 stage differs")

    arms = receipt.get("arms")
    noise = receipt.get("initial_noise_artifacts")
    outputs = receipt.get("outputs")
    identities = receipt.get("generated_identities")
    expected_keyset = set(expected_arms)
    if any(
        not isinstance(value, dict) or set(value) != expected_keyset
        for value in (arms, noise, outputs, identities)
    ):
        raise SystemExit(f"{group} arm/artifact keyset differs")

    expected_files = {"receipt.json"}
    source = bound_artifact(
        receipt.get("source", {}).get("normalized_clean_latent_artifact"),
        group_root,
        ".safetensors",
        f"{group} source latent",
    )
    expected_files.add(source.name)
    artifact_rows = []
    for arm in expected_arms:
        noise_path = bound_artifact(
            noise[arm], group_root, ".safetensors", f"{group}/{arm} noise"
        )
        clean_path = bound_artifact(
            outputs[arm].get("normalized_clean_latent"),
            group_root,
            ".safetensors",
            f"{group}/{arm} clean latent",
        )
        expected_files.update((noise_path.name, clean_path.name))
        row = {
            "arm": arm,
            "noise_path": str(noise_path),
            "noise_sha256": file_sha256(noise_path),
            "clean_latent_path": str(clean_path),
            "clean_latent_sha256": file_sha256(clean_path),
        }
        if steps == 40:
            mp4_path = bound_artifact(
                outputs[arm], group_root, ".mp4", f"{group}/{arm} MP4"
            )
            expected_files.add(mp4_path.name)
            row.update({
                "mp4_path": str(mp4_path),
                "mp4_sha256": file_sha256(mp4_path),
            })
        elif outputs[arm].get("mp4") is not None or outputs[arm].get("decode_performed") is not False:
            raise SystemExit(f"{group}/{arm} one-step decode contract differs")
        artifact_rows.append(row)

    actual_files = set()
    for item in group_root.iterdir():
        if item.is_symlink() or not item.is_file():
            raise SystemExit(f"{group} contains a non-plain artifact")
        actual_files.add(item.name)
    if actual_files != expected_files:
        raise SystemExit(f"{group} artifact file closure differs")
    groups.append({
        "group": group,
        "receipt_path": str(path),
        "receipt_file_sha256": file_sha256(path),
        "receipt_digest": declared,
        "source_latent_path": str(source),
        "source_latent_sha256": file_sha256(source),
        "artifacts": artifact_rows,
    })

stage = {
    "schema_version": "bernini-native-multivideo-motion-donor-stage-receipt-v1",
    "complete": True,
    "steps": steps,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "groups": groups,
    "group_count": 2,
    "arms_per_group": len(expected_arms),
    "all_artifact_paths_canonical_group_local": True,
    "all_artifact_file_sha256_recomputed": True,
    "training_performed": False,
    "scientific_claim_authorized": False,
}
stage["stage_receipt_digest"] = hashlib.sha256(canonical(stage)).hexdigest()
stage_path = root / "stage.receipt.json"
if stage_path.exists() or stage_path.is_symlink():
    raise SystemExit("stage receipt already exists")
payload = canonical(stage) + b"\n"
with tempfile.NamedTemporaryFile(
    dir=root, prefix=".stage.receipt.", suffix=".tmp", delete=False
) as handle:
    temporary = Path(handle.name)
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
try:
    os.replace(temporary, stage_path)
finally:
    if temporary.exists():
        temporary.unlink()
descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
print(canonical(stage).decode("ascii"))
PY

echo "NATIVE_MULTIVIDEO_MOTION_DONOR_STAGE_COMPLETE steps=${num_inference_steps} donor_is_not_target=true claims=false"
