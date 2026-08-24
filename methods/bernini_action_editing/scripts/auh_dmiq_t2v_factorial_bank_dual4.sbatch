#!/usr/bin/env bash
#SBATCH --job-name=dmiq-t2v-bank2x4
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=48:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# This launcher does not submit itself.  Within one allocated 8xMI210 node it
# executes the pre-registered pure-T2V factorial manifest as two isolated SP4
# groups.  Bernini-R has 12 attention heads, so one Ulysses-8 group is invalid.
# Each matched cell keeps full_action plus all nine negatives on one SP4 group
# and calls the existing frozen native T2V renderer ten times.  Source pixels
# are used only for exact81 bucket selection/hash verification.  The finalizer
# rejects source/reference latents, non-[1,16,21,62,60] geometry, missing
# observed-native Gaussian artifacts, or unequal Gaussian values within a cell.
# Rendering never self-authorizes FITQ: a separate blind split-isolated event
# audit must verify >=8 discovery and >=4 confirmation positive/contrast cells.
# This launcher is deliberately restricted to engineering_micro timing runs:
# it starts one torchrun/model load per entry.  The 320-render scientific bank
# is NOT scale-ready until one persistent frozen-model worker exists per SP4.
# It also refuses rung>0 because exact prior-artifact reuse/lineage is not yet
# implemented; blindly rerendering rung0 would not retain the audited failures.

source_archive="${DMIQ_T2V_SOURCE_ARCHIVE:?set DMIQ_T2V_SOURCE_ARCHIVE}"
source_archive_sha256="${DMIQ_T2V_SOURCE_ARCHIVE_SHA256:?set DMIQ_T2V_SOURCE_ARCHIVE_SHA256}"
source_revision="${DMIQ_T2V_SOURCE_REVISION:?set DMIQ_T2V_SOURCE_REVISION}"
manifest="${DMIQ_T2V_BANK_MANIFEST:?set DMIQ_T2V_BANK_MANIFEST}"
manifest_file_sha256="${DMIQ_T2V_BANK_MANIFEST_SHA256:?set DMIQ_T2V_BANK_MANIFEST_SHA256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
source_video="${DMIQ_T2V_SOURCE_VIDEO:?set DMIQ_T2V_SOURCE_VIDEO}"
output_dir="${DMIQ_T2V_BANK_OUTPUT_DIR:?set DMIQ_T2V_BANK_OUTPUT_DIR}"
python_bin="${DMIQ_T2V_PYTHON_BIN:?set DMIQ_T2V_PYTHON_BIN}"

sp4_a_visible_gpus="0,1,2,3"
sp4_b_visible_gpus="4,5,6,7"

fail() {
  echo "[dmiq-t2v-bank] ERROR: $*" >&2
  exit 2
}

require_absolute() {
  local value="$1"
  local label="$2"
  [[ "${value}" == /* ]] || fail "${label} must be absolute"
}

for value_name in \
  source_archive manifest bernini_root veomni_root checkpoint \
  checkpoint_manifest source_video output_dir python_bin
do
  require_absolute "${!value_name}" "${value_name}"
done

[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source archive SHA-256 must be lowercase"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be a full lowercase SHA-1"
[[ "${manifest_file_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "manifest file SHA-256 must be lowercase"

source_archive="$(realpath -e -- "${source_archive}")"
manifest="$(realpath -e -- "${manifest}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
source_video="$(realpath -e -- "${source_video}")"
python_bin="$(realpath -e -- "${python_bin}")"

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "method source archive is not plain"
[[ -f "${manifest}" && ! -L "${manifest}" ]] || fail "factorial manifest is not plain"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini source root is invalid"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni source root is invalid"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint root is invalid"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest is not plain"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "source geometry video is not plain"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "runtime Python is invalid"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method source archive hash differs"
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_file_sha256}" ]] || fail "factorial manifest file hash differs"
actual_archive_revision="$(git get-tar-commit-id <"${source_archive}")" || fail "method source must be a git tar archive"
[[ "${actual_archive_revision}" == "${source_revision}" ]] || fail "method archive embedded revision differs"

manifest_profile_and_rung="$("${python_bin}" -B - "${manifest}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    value = json.load(handle)
    print(f"{value['profile']}:{value['attempt_rung']}")
PY
)"
[[ "${manifest_profile_and_rung}" == "engineering_micro:0" ]] || fail "only engineering_micro rung0 timing is authorized; scientific persistence and cumulative topup reuse are not implemented"

[[ "${output_dir}" != / ]] || fail "output directory cannot be filesystem root"
output_basename="${output_dir##*/}"
[[ "${output_basename}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename is unsafe"
[[ "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output directory must be canonical"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing to reuse output directory"
output_parent="$(dirname -- "${output_dir}")"
mkdir -p -- "${output_parent}"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent is invalid"
[[ "$(realpath -e -- "${output_parent}")" == "${output_parent}" ]] || fail "output parent must be canonical"

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
task_scratch="$(mktemp -d "${scratch_parent%/}/dmiq-t2v-bank-${SLURM_JOB_ID}.XXXXXX")"
sp4_a_pid=""
sp4_b_pid=""

cleanup() {
  local exit_status=$?
  local cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/dmiq-t2v-bank-${SLURM_JOB_ID}."*) ;;
    *) echo "[dmiq-t2v-bank] refusing unsafe scratch cleanup" >&2; cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  if [[ "${cleanup_failed}" != 0 && "${exit_status}" == 0 ]]; then
    exit_status=2
  fi
  if [[ "${exit_status}" == 0 ]]; then
    echo "[dmiq-t2v-bank] PASS topology=two-isolated-SP4 exact81 pure_t2v=true output=${output_dir}"
  else
    echo "[dmiq-t2v-bank] EXIT status=${exit_status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${exit_status}"
}

terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${sp4_a_pid}" "${sp4_b_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then
      kill "${child}" 2>/dev/null || true
    fi
  done
  for child in "${sp4_a_pid}" "${sp4_b_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then
      wait "${child}" 2>/dev/null || true
    fi
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
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged method archive hash differs"
[[ "$(git get-tar-commit-id <"${archive_copy}")" == "${source_revision}" ]] || fail "staged method archive revision differs"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

archive = sys.argv[1]
required = {
    "methods/bernini_action_editing/dmiq_t2v_factorial_bank.py",
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
    "methods/bernini_action_editing/source_kv_replay.py",
    "methods/bernini_action_editing/source_kv_route_batches.py",
    "methods/bernini_action_editing/source_value_residual.py",
    "methods/bernini_action_editing/tools/materialize_vae.py",
}
seen = set()
with tarfile.open(archive, "r:*") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit("archive member escaped method subtree")
        normalized = path.as_posix().lstrip("./")
        if member.issym() or member.islnk() or member.isfifo() or member.isdev():
            raise SystemExit("archive contains a link or device")
        if normalized.startswith("methods/bernini_action_editing/"):
            seen.add(normalized)
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"archive lacks DMIQ T2V bank closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
[[ -d "${method_root}" && ! -L "${method_root}" ]] || fail "extracted method root is invalid"
find "${method_root}" -type f -exec chmod a-w -- {} +

for required in \
  dmiq_t2v_factorial_bank.py \
  infer_native_identity_generation_canary.py \
  infer_lora.py \
  train_lora.py \
  infer_source_kv_carrier_oracle.py \
  infer_source_value_residual_oracle.py \
  source_kv_replay.py \
  source_kv_route_batches.py \
  source_value_residual.py \
  tools/materialize_vae.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "missing staged ${required}"
done

mkdir -- "${output_dir}"
mkdir -- "${output_dir}/entries" "${output_dir}/logs"
persistent_manifest="${output_dir}/manifest.json"
cp -- "${manifest}" "${persistent_manifest}"
chmod 0444 -- "${persistent_manifest}"
[[ "$(sha256sum "${persistent_manifest}" | awk '{print $1}')" == "${manifest_file_sha256}" ]] || fail "persistent manifest hash differs"

"${python_bin}" -B "${method_root}/dmiq_t2v_factorial_bank.py" validate \
  --manifest "${persistent_manifest}" \
  --expected-file-sha256 "${manifest_file_sha256}" \
  >"${output_dir}/manifest-validation.json"

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

  mapfile -t group_entries < <(
    "${python_bin}" -B "${method_root}/dmiq_t2v_factorial_bank.py" list-entry-ids \
      --manifest "${persistent_manifest}" \
      --expected-file-sha256 "${manifest_file_sha256}" \
      --group "${group}"
  )
  [[ "${#group_entries[@]}" -gt 0 ]] || fail "${group} has no manifest entries"

  for entry_id in "${group_entries[@]}"; do
    [[ "${entry_id}" =~ ^[a-z0-9][a-z0-9._-]{0,95}$ ]] || fail "unsafe entry id from validated manifest"
    export TORCHELASTIC_ERROR_FILE="${group_root}/torch-elastic-${entry_id}.json"
    export BERNINI_OUTPUT_TRANSACTION_ID="${SLURM_JOB_ID}-${group}-${entry_id}"
    echo "[dmiq-t2v-bank] group=${group} entry=${entry_id} pure_t2v=true source_latent=false"
    "${python_bin}" -B -m torch.distributed.run \
      --nproc_per_node=4 \
      --master_addr=127.0.0.1 \
      --master_port="${master_port}" \
      "${method_root}/dmiq_t2v_factorial_bank.py" render-entry \
      --manifest "${persistent_manifest}" \
      --expected-file-sha256 "${manifest_file_sha256}" \
      --entry-id "${entry_id}" \
      --output-root "${output_dir}" \
      --bernini-root "${bernini_root}" \
      --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" \
      --checkpoint-content-manifest "${checkpoint_manifest}" \
      --source-video "${source_video}" \
      --method-source-revision "${source_revision}" \
      --method-source-archive-sha256 "${source_archive_sha256}"
  done
)

sp4_a_log="${output_dir}/logs/sp4-a.log"
sp4_b_log="${output_dir}/logs/sp4-b.log"
echo "[dmiq-t2v-bank] semantic_inputs=prompt,source_geometry_bucket target=false source_latent=false reference=false mask=false flow=false pose=false track=false trajectory=false first_frame=false"
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
  fail "one or more pure-T2V SP4 groups failed"
fi

"${python_bin}" -B "${method_root}/dmiq_t2v_factorial_bank.py" finalize \
  --manifest "${persistent_manifest}" \
  --expected-file-sha256 "${manifest_file_sha256}" \
  --output-root "${output_dir}" \
  --output-receipt "${output_dir}/bank.receipt.json" \
  >"${output_dir}/bank-finalize.log"

[[ -f "${output_dir}/bank.receipt.json" && ! -L "${output_dir}/bank.receipt.json" ]] || fail "bank receipt was not finalized"
echo "DMIQ_T2V_FACTORIAL_BANK_PROVENANCE_OK_BLIND_SPLIT_EVENT_AUDIT_PENDING"
