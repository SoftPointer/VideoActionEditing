#!/usr/bin/env bash
#SBATCH --job-name=bernini-phase-noise-2x4
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

# Run one phase-noise stage as two isolated native WORLD4/Ulysses4 processes.
# Both groups bind the same receipt-sealed proposal cell and baseline seed;
# only the registered arm subset differs.  No proposal MP4, edit target, mask,
# flow, pose, track, trajectory, training, or external initial-noise file is read.

source_archive="${PHASE_NOISE_SOURCE_ARCHIVE:?set PHASE_NOISE_SOURCE_ARCHIVE}"
source_archive_sha256="${PHASE_NOISE_SOURCE_ARCHIVE_SHA256:?set PHASE_NOISE_SOURCE_ARCHIVE_SHA256}"
source_revision="${PHASE_NOISE_SOURCE_REVISION:?set PHASE_NOISE_SOURCE_REVISION}"
manifest="${PHASE_NOISE_FACTOR_MANIFEST:?set PHASE_NOISE_FACTOR_MANIFEST}"
manifest_sha256="${PHASE_NOISE_FACTOR_MANIFEST_SHA256:?set PHASE_NOISE_FACTOR_MANIFEST_SHA256}"
bank_receipt="${PHASE_NOISE_BANK_RECEIPT:?set PHASE_NOISE_BANK_RECEIPT}"
bank_receipt_sha256="${PHASE_NOISE_BANK_RECEIPT_SHA256:?set PHASE_NOISE_BANK_RECEIPT_SHA256}"
bank_output_root="${PHASE_NOISE_BANK_OUTPUT_ROOT:?set PHASE_NOISE_BANK_OUTPUT_ROOT}"
execution_group="${PHASE_NOISE_EXECUTION_GROUP:?set PHASE_NOISE_EXECUTION_GROUP to sp4-a or sp4-b}"
condition_mode="${PHASE_NOISE_CONDITION_MODE:?set PHASE_NOISE_CONDITION_MODE to r2v5 or rv2v4}"
source_video="${PHASE_NOISE_SOURCE_VIDEO:?set PHASE_NOISE_SOURCE_VIDEO}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
output_dir="${PHASE_NOISE_OUTPUT_DIR:?set PHASE_NOISE_OUTPUT_DIR}"
python_bin="${PHASE_NOISE_PYTHON_BIN:?set PHASE_NOISE_PYTHON_BIN}"
num_inference_steps="${PHASE_NOISE_NUM_INFERENCE_STEPS:?set PHASE_NOISE_NUM_INFERENCE_STEPS to 1 or 40}"
seed="${PHASE_NOISE_SEED:-20260810}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
cdf_dog_source_sha256="5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
sp4_a_visible_gpus="0,1,2,3"
sp4_b_visible_gpus="4,5,6,7"

fail() {
  echo "[self-imagined-phase-noise] ERROR: $*" >&2
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
[[ "${num_inference_steps}" == "1" || "${num_inference_steps}" == "40" ]] || fail "stage steps must be 1 or 40"
[[ "${execution_group}" == "sp4-a" || "${execution_group}" == "sp4-b" ]] || fail "execution group must be one registered cell"
[[ "${condition_mode}" == "r2v5" || "${condition_mode}" == "rv2v4" ]] || fail "condition mode must be native r2v5 or rv2v4"
[[ "${seed}" =~ ^[0-9]+$ ]] || fail "seed must be a nonnegative integer"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "archive SHA-256 must be lowercase"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be full lowercase SHA-1"
[[ "${manifest_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "manifest SHA-256 must be lowercase"
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
[[ -f "${bank_receipt}" && ! -L "${bank_receipt}" ]] || fail "factor bank receipt is not plain"
[[ -d "${bank_output_root}" && ! -L "${bank_output_root}" ]] || fail "bank root is invalid"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "source video is not plain"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini root is invalid"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni root is invalid"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint is invalid"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest is not plain"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "runtime Python is invalid"
[[ -f "${launcher_path}" && ! -L "${launcher_path}" ]] || fail "launcher is not plain"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "archive hash differs"
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha256}" ]] || fail "manifest hash differs"
[[ "$(sha256sum "${bank_receipt}" | awk '{print $1}')" == "${bank_receipt_sha256}" ]] || fail "bank receipt hash differs"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${cdf_dog_source_sha256}" ]] || fail "source is not pinned CDF dog"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "archive revision differs"

[[ "${output_dir}" != / ]] || fail "output cannot be root"
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
task_scratch="$(mktemp -d "${scratch_parent%/}/self-imagined-phase-noise-${SLURM_JOB_ID}.XXXXXX")"
sp4_a_pid=""
sp4_b_pid=""

cleanup() {
  local exit_status=$?
  local cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/self-imagined-phase-noise-${SLURM_JOB_ID}."*) ;;
    *) echo "[self-imagined-phase-noise] refusing unsafe scratch cleanup" >&2; cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  if [[ "${cleanup_failed}" != 0 && "${exit_status}" == 0 ]]; then exit_status=2; fi
  if [[ "${exit_status}" == 0 ]]; then
    echo "[self-imagined-phase-noise] PASS steps=${num_inference_steps} dual-WORLD4 output=${output_dir}"
  else
    echo "[self-imagined-phase-noise] EXIT status=${exit_status}" >&2
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

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

required = {
    "methods/bernini_action_editing/infer_native_self_imagined_phase_noise_canary.py",
    "methods/bernini_action_editing/self_imagined_phase_noise.py",
    "methods/bernini_action_editing/source_spectral_bridge.py",
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
    "methods/bernini_action_editing/scripts/auh_infer_native_self_imagined_phase_noise_dual4.sbatch",
    "methods/bernini_action_editing/scripts/auh_submit_native_self_imagined_phase_noise_chain.sh",
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
    raise SystemExit(f"archive lacks self-imagined phase-noise closure: {missing}")
PY

tar --no-same-owner --no-same-permissions \
  -xf "${archive_copy}" -C "${task_scratch}/source-tree" \
  methods/bernini_action_editing
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
runtime_path="${method_root}/infer_native_self_imagined_phase_noise_canary.py"
staged_launcher="${method_root}/scripts/auh_infer_native_self_imagined_phase_noise_dual4.sbatch"
[[ -f "${runtime_path}" && ! -L "${runtime_path}" ]] || fail "staged runtime absent"
[[ -f "${staged_launcher}" && ! -L "${staged_launcher}" ]] || fail "staged launcher absent"
find "${method_root}" -type f -exec chmod a-w -- {} +
launcher_source_sha256="$(sha256sum "${staged_launcher}" | awk '{print $1}')"
[[ "$(sha256sum "${launcher_path}" | awk '{print $1}')" == "${launcher_source_sha256}" ]] || fail "executed launcher differs from archive"

mkdir -- "${output_dir}"
mkdir -- "${output_dir}/logs"
cp -- "${manifest}" "${output_dir}/factor-manifest.json"
cp -- "${bank_receipt}" "${output_dir}/factor-bank.receipt.json"
chmod 0444 -- "${output_dir}/factor-manifest.json" "${output_dir}/factor-bank.receipt.json"

job_mod=$((SLURM_JOB_ID % 10000))
sp4_a_master_port=$((32000 + 2 * job_mod))
sp4_b_master_port=$((sp4_a_master_port + 1))

launch_group() (
  set -Eeuo pipefail
  local group="$1"
  local visible_gpus="$2"
  local master_port="$3"
  shift 3
  local group_root="${task_scratch}/groups/${group}"
  local group_output="${output_dir}/${group}"
  local -a arms=("$@")
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
  export BERNINI_OUTPUT_TRANSACTION_ID="${SLURM_JOB_ID}-${group}-phase-noise-s${num_inference_steps}"

  echo "[self-imagined-phase-noise] group=${group} WORLD4 arms=${arms[*]} injection=true observer=false proposal_mp4=false"
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
    --execution-group "${execution_group}" \
    --condition-mode "${condition_mode}" \
    --arms "${arms[@]}" \
    --num-inference-steps "${num_inference_steps}" \
    --seed "${seed}" \
    --output-dir "${group_output}" \
    --runtime-source-revision "${source_revision}" \
    --runtime-source-archive-sha256 "${source_archive_sha256}" \
    --launcher-source-sha256 "${launcher_source_sha256}" \
    --expected-bernini-commit "${bernini_commit}" \
    --expected-veomni-commit "${veomni_commit}" \
    --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}"
)

launch_group arms-a "${sp4_a_visible_gpus}" "${sp4_a_master_port}" \
  action-phi noop-phi reverse-phi \
  >"${output_dir}/logs/arms-a.log" 2>&1 &
sp4_a_pid=$!
launch_group arms-b "${sp4_b_visible_gpus}" "${sp4_b_master_port}" \
  matched-gaussian action-phi-source-dc-rho02 \
  >"${output_dir}/logs/arms-b.log" 2>&1 &
sp4_b_pid=$!

set +e
wait "${sp4_a_pid}"
sp4_a_status=$?
wait "${sp4_b_pid}"
sp4_b_status=$?
set -e
[[ "${sp4_a_status}" == 0 && "${sp4_b_status}" == 0 ]] || fail "WORLD4 group failure arms-a=${sp4_a_status} arms-b=${sp4_b_status}"

"${python_bin}" -B - "${output_dir}" "${num_inference_steps}" "${condition_mode}" \
  "${execution_group}" "${source_revision}" "${source_archive_sha256}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

root = Path(sys.argv[1]).resolve(strict=True)
steps = int(sys.argv[2])
condition_mode = sys.argv[3]
execution_group = sys.argv[4]
source_revision = sys.argv[5]
source_archive_sha256 = sys.argv[6]
expected_arms = {
    "arms-a": {"action-phi", "noop-phi", "reverse-phi"},
    "arms-b": {"matched-gaussian", "action-phi-source-dc-rho02"},
}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

receipts = {}
artifact_inventory = {}
baseline_hashes = set()
proposal_hash_sets = []
source_hashes = set()
for group, arm_set in expected_arms.items():
    group_root = root / group
    receipt_path = group_root / "receipt.json"
    if not group_root.is_dir() or group_root.is_symlink() or not receipt_path.is_file() or receipt_path.is_symlink():
        raise SystemExit(f"invalid transactional group output: {group}")
    if list(root.glob(f".{group}.partial-*")):
        raise SystemExit(f"partial group output remains: {group}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    declared = receipt.pop("receipt_digest", None)
    if declared != hashlib.sha256(canonical(receipt)).hexdigest():
        raise SystemExit(f"sealed receipt digest differs: {group}")
    receipt["receipt_digest"] = declared
    expected_stage = "engineering_oom_callpath_canary" if steps == 1 else "matched_exact40_qualitative_phase_noise_pilot"
    if receipt.get("stage") != expected_stage:
        raise SystemExit(f"stage differs: {group}")
    if set(receipt.get("arms_in_execution_order", [])) != arm_set:
        raise SystemExit(f"arm registry differs: {group}")
    if receipt.get("execution_group") != execution_group:
        raise SystemExit(f"proposal cell differs: {group}")
    if receipt.get("source", {}).get("native_condition", {}).get("native_arm") != ("r2v" if condition_mode == "r2v5" else "rv2v"):
        raise SystemExit(f"native condition differs: {group}")
    factor = receipt.get("factor_bank", {})
    if factor.get("proposal_mp4_consumed") is not False or factor.get("only_predecode_fp32_normalized_clean_latents_consumed") is not True:
        raise SystemExit(f"proposal input closure differs: {group}")
    if receipt.get("interpretation", {}).get("randn_wrapper_is_injection_not_observation") is not True:
        raise SystemExit(f"injection interpretation differs: {group}")
    if receipt.get("interpretation", {}).get("scientific_claim_authorized") is not False:
        raise SystemExit(f"scientific claim gate differs: {group}")
    forbidden = receipt.get("forbidden_inputs", {})
    if any(value is not False for value in forbidden.values()):
        raise SystemExit(f"forbidden input consumed: {group}")
    baseline_hashes.add(receipt["matched_design"]["baseline_raw_storage_sha256"])
    proposal_hash_sets.append({
        key: value["clean_latent_raw_storage_sha256"]
        for key, value in factor["proposal_provenance"].items()
    })
    source_hashes.add(receipt["source"]["phase_carrier_cpu_identity"]["identity"]["raw_storage_sha256"])
    for arm, arm_receipt in receipt.get("arms", {}).items():
        if arm == "matched-gaussian":
            if arm_receipt.get("injection_performed") is not False or arm_receipt.get("original_return_object_forwarded") is not True:
                raise SystemExit("matched Gaussian is not exact pass-through")
        elif arm_receipt.get("injection_performed") is not True or arm_receipt.get("wrapper_is_injector_not_observer") is not True:
            raise SystemExit(f"active arm is not injection: {arm}")
    receipts[group] = {
        "path": str(receipt_path),
        "file_sha256": file_sha256(receipt_path),
        "receipt_digest": declared,
        "arms": sorted(arm_set),
    }

if len(baseline_hashes) != 1 or len(source_hashes) != 1 or proposal_hash_sets[0] != proposal_hash_sets[1]:
    raise SystemExit("dual-WORLD4 matched baseline/source/proposal closure differs")

for path in sorted(root.rglob("*")):
    if path.is_symlink():
        raise SystemExit(f"output contains symlink: {path}")
    if path.is_file() and path.name != "stage.receipt.json":
        relative = path.relative_to(root).as_posix()
        artifact_inventory[relative] = {
            "sha256": file_sha256(path),
            "byte_count": path.stat().st_size,
        }

stage = {
    "schema_version": "bernini-self-imagined-phase-noise-stage-receipt-v1",
    "method": "frozen-bernini-self-imagined-phase-noise-canary",
    "num_inference_steps": steps,
    "condition_mode": condition_mode,
    "execution_group": execution_group,
    "runtime_source_revision": source_revision,
    "runtime_source_archive_sha256": source_archive_sha256,
    "dual_world4_on_one_eight_gpu_node": True,
    "world4_groups": receipts,
    "complete_arm_set": sorted(set().union(*expected_arms.values())),
    "same_baseline_gaussian_across_groups": True,
    "baseline_raw_storage_sha256": next(iter(baseline_hashes)),
    "same_source_phase_carrier_across_groups": True,
    "same_registered_proposal_latents_across_groups": True,
    "proposal_mp4_consumed": False,
    "injection_not_observation": True,
    "all_artifact_file_sha256_recomputed": True,
    "artifact_inventory": artifact_inventory,
    "scientific_claim_authorized": False,
}
stage["stage_receipt_digest"] = hashlib.sha256(canonical(stage)).hexdigest()
stage_path = root / "stage.receipt.json"
with tempfile.NamedTemporaryFile(dir=root, prefix=".stage.receipt.", suffix=".tmp", delete=False) as handle:
    temporary = Path(handle.name)
    handle.write(canonical(stage) + b"\n")
    handle.flush()
    os.fsync(handle.fileno())
try:
    os.replace(temporary, stage_path)
    descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
finally:
    if temporary.exists():
        temporary.unlink()
print(json.dumps(stage, sort_keys=True))
PY

chmod -R a-w -- "${output_dir}"
