#!/usr/bin/env bash
#SBATCH --job-name=bern-pass-a-2x4
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

# One 8xMI210 allocation becomes two independent WORLD4/Ulysses-4 groups.
# sp4-a renders all four branches at registered seed 20260808 while sp4-b
# renders all four at seed 20260809.  These are the already registered CDF-dog
# DMIQ seeds, not a new visual seed search.  Both groups run concurrently. Each
# branch is a fresh process around the existing native T2V canary, avoiding
# distributed-global reuse while leaving the official sampler untouched.
#
# The source MP4 is used only for byte hash verification and the fixed 496x480
# bucket.  The finalizer rejects any source/reference latent or privileged
# conditioning.  An exact40 render is still only a candidate proposal bank:
# independent manual event qualification must accept the complete 2x4 bank,
# otherwise Pass A is rejected without choosing a lucky seed or branch.

source_archive="${PASS_A_SOURCE_ARCHIVE:?set PASS_A_SOURCE_ARCHIVE}"
source_archive_sha256="${PASS_A_SOURCE_ARCHIVE_SHA256:?set PASS_A_SOURCE_ARCHIVE_SHA256}"
source_revision="${PASS_A_SOURCE_REVISION:?set PASS_A_SOURCE_REVISION}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
source_video="${PASS_A_SOURCE_VIDEO:?set PASS_A_SOURCE_VIDEO}"
source_sha256="${PASS_A_SOURCE_SHA256:?set PASS_A_SOURCE_SHA256}"
output_dir="${PASS_A_OUTPUT_DIR:?set PASS_A_OUTPUT_DIR}"
python_bin="${PASS_A_PYTHON_BIN:?set PASS_A_PYTHON_BIN}"
num_steps="${PASS_A_NUM_INFERENCE_STEPS:?set PASS_A_NUM_INFERENCE_STEPS=1 or 40}"

sp4_a_visible_gpus="0,1,2,3"
sp4_b_visible_gpus="4,5,6,7"

fail() {
  echo "[bernini-pass-a] ERROR: $*" >&2
  exit 2
}

require_absolute() {
  local value="$1"
  local label="$2"
  [[ "${value}" == /* ]] || fail "${label} must be absolute"
}

for value_name in \
  source_archive bernini_root veomni_root checkpoint checkpoint_manifest \
  source_video output_dir python_bin
do
  require_absolute "${!value_name}" "${value_name}"
done

[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source archive SHA-256 must be lowercase"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be full lowercase SHA-1"
[[ "${source_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source video SHA-256 must be lowercase"
[[ "${num_steps}" == 1 || "${num_steps}" == 40 ]] || fail "steps must be exactly 1 or 40"

source_archive="$(realpath -e -- "${source_archive}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
source_video="$(realpath -e -- "${source_video}")"
python_bin="$(realpath -e -- "${python_bin}")"
launcher_path="$(realpath -e -- "$0")"

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "method source archive is not plain"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini source root is invalid"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni source root is invalid"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint root is invalid"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest is not plain"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "CDF-dog source video is not plain"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "runtime Python is invalid"
[[ -f "${launcher_path}" && ! -L "${launcher_path}" ]] || fail "compute launcher is not plain"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive hash differs"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${source_sha256}" ]] || fail "CDF-dog source hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "method archive embedded revision differs"

[[ "${output_dir}" != / ]] || fail "output directory cannot be root"
output_basename="${output_dir##*/}"
[[ "${output_basename}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename is unsafe"
[[ "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output directory must be canonical"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing to reuse output directory"
output_parent="$(dirname -- "${output_dir}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent must exist and be plain"
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
task_scratch="$(mktemp -d "${scratch_parent%/}/bern-pass-a-${SLURM_JOB_ID}.XXXXXX")"
sp4_a_pid=""
sp4_b_pid=""

cleanup() {
  local exit_status=$?
  local cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/bern-pass-a-${SLURM_JOB_ID}."*) ;;
    *) echo "[bernini-pass-a] refusing unsafe scratch cleanup" >&2; cleanup_failed=1 ;;
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
    echo "[bernini-pass-a] PASS topology=dual-WORLD4 exact81=true steps=${num_steps} entries=8 event_qualification=pending output=${output_dir}"
  else
    echo "[bernini-pass-a] EXIT status=${exit_status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
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
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged archive hash differs"
[[ "$(git get-tar-commit-id <"${archive_copy}")" == "${source_revision}" ]] || fail "staged archive revision differs"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

required = {
    "methods/bernini_action_editing/source_caption_t2v_pass_a.py",
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
    "methods/bernini_action_editing/source_kv_replay.py",
    "methods/bernini_action_editing/source_kv_route_batches.py",
    "methods/bernini_action_editing/source_value_residual.py",
    "methods/bernini_action_editing/tools/build_renderer_dataset.py",
    "methods/bernini_action_editing/tools/materialize_vae.py",
    "methods/bernini_action_editing/scripts/auh_source_caption_t2v_pass_a_dual4.sbatch",
    "methods/bernini_action_editing/scripts/auh_submit_source_caption_t2v_pass_a_chain.sh",
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
    raise SystemExit(f"archive lacks Pass A runtime closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" \
  -C "${task_scratch}/source-tree" methods/bernini_action_editing
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
[[ -d "${method_root}" && ! -L "${method_root}" ]] || fail "extracted method root is invalid"
find "${method_root}" -type f -exec chmod a-w -- {} +
staged_launcher="${method_root}/scripts/auh_source_caption_t2v_pass_a_dual4.sbatch"
[[ "$(sha256sum "${launcher_path}" | awk '{print $1}')" == "$(sha256sum "${staged_launcher}" | awk '{print $1}')" ]] || fail "executed compute launcher differs from source archive"

for required in \
  source_caption_t2v_pass_a.py \
  infer_native_identity_generation_canary.py \
  infer_lora.py \
  train_lora.py \
  infer_source_kv_carrier_oracle.py \
  infer_source_value_residual_oracle.py \
  source_kv_replay.py \
  source_kv_route_batches.py \
  source_value_residual.py \
  tools/build_renderer_dataset.py \
  tools/materialize_vae.py \
  scripts/auh_source_caption_t2v_pass_a_dual4.sbatch \
  scripts/auh_submit_source_caption_t2v_pass_a_chain.sh
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "missing staged ${required}"
done

mkdir -- "${output_dir}"
mkdir -- "${output_dir}/entries" "${output_dir}/logs"
manifest="${output_dir}/manifest.json"
manifest_sha256="$(
  "${python_bin}" -B "${method_root}/source_caption_t2v_pass_a.py" build-manifest \
    --source-video "${source_video}" \
    --expected-source-sha256 "${source_sha256}" \
    --num-inference-steps "${num_steps}" \
    --method-source-revision "${source_revision}" \
    --method-source-archive-sha256 "${source_archive_sha256}" \
    --output "${manifest}"
)"
[[ "${manifest_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "manifest builder returned invalid SHA-256"
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha256}" ]] || fail "persistent manifest hash differs"
chmod 0444 -- "${manifest}"

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
    "${python_bin}" -B "${method_root}/source_caption_t2v_pass_a.py" list-entry-ids \
      --manifest "${manifest}" \
      --expected-file-sha256 "${manifest_sha256}" \
      --group "${group}"
  )
  [[ "${#group_entries[@]}" -eq 4 ]] || fail "${group} must contain exactly four branches"

  for entry_id in "${group_entries[@]}"; do
    [[ "${entry_id}" =~ ^[a-z0-9][a-z0-9._-]{0,95}$ ]] || fail "unsafe entry id"
    export TORCHELASTIC_ERROR_FILE="${group_root}/torch-elastic-${entry_id}.json"
    export BERNINI_OUTPUT_TRANSACTION_ID="${SLURM_JOB_ID}-${group}-${entry_id}"
    echo "[bernini-pass-a] group=${group} entry=${entry_id} pure_t2v=true source_condition=false"
    "${python_bin}" -B -m torch.distributed.run \
      --nproc_per_node=4 \
      --master_addr=127.0.0.1 \
      --master_port="${master_port}" \
      "${method_root}/source_caption_t2v_pass_a.py" render-entry \
      --manifest "${manifest}" \
      --expected-file-sha256 "${manifest_sha256}" \
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

sp4_a_log="${output_dir}/logs/sp4-a-seed20260808.log"
sp4_b_log="${output_dir}/logs/sp4-b-seed20260809.log"
echo "[bernini-pass-a] conditions=text_only source_role=hash_and_bucket multi_video=false multi_image=false image=false target=false"
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
  fail "one or more Pass A WORLD4 groups failed"
fi

"${python_bin}" -B "${method_root}/source_caption_t2v_pass_a.py" finalize \
  --manifest "${manifest}" \
  --expected-file-sha256 "${manifest_sha256}" \
  --output-root "${output_dir}" \
  --output-receipt "${output_dir}/pass-a.receipt.json" \
  >"${output_dir}/finalize.log"

[[ -f "${output_dir}/pass-a.receipt.json" && ! -L "${output_dir}/pass-a.receipt.json" ]] || fail "Pass A receipt is absent"
echo "BERNINI_CDF_DOG_PASS_A_RENDER_COMPLETE_LABELS_UNVERIFIED_MANUAL_QUALIFICATION_REQUIRED"
