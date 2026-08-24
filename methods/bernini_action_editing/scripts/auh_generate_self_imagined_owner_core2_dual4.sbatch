#!/usr/bin/env bash
#SBATCH --job-name=bernini-qm-owner
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

repo_root="${SIM_OWNER_REPO_ROOT:?set clean repository root}"
registry="${SIM_OWNER_REGISTRY:?set sealed core2 registry}"
registry_sha256="${SIM_OWNER_REGISTRY_SHA256:?set registry SHA-256}"
source_archive="${SIM_OWNER_SOURCE_ARCHIVE:?set git source archive}"
source_archive_sha256="${SIM_OWNER_SOURCE_ARCHIVE_SHA256:?set git archive SHA-256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
output_root="${SIM_OWNER_OUTPUT_ROOT:?set fresh output root}"
python_bin="${SIM_OWNER_PYTHON_BIN:?set vace Python executable}"

fail() {
  echo "[self-imagined-owner] ERROR: $*" >&2
  exit 2
}

for value in "${repo_root}" "${registry}" "${source_archive}" "${bernini_root}" "${veomni_root}" "${checkpoint}" "${checkpoint_manifest}" "${output_root}" "${python_bin}"; do
  [[ "${value}" == /* ]] || fail "all paths must be absolute"
done
[[ "${registry_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "registry SHA-256 differs"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source archive SHA-256 differs"

repo_root="$(realpath -e -- "${repo_root}")"
registry="$(realpath -e -- "${registry}")"
source_archive="$(realpath -e -- "${source_archive}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
[[ "$(sha256sum "${registry}" | awk '{print $1}')" == "${registry_sha256}" ]] || fail "registry changed"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive must be a plain file"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive changed"

method_root="${repo_root}/methods/bernini_action_editing"
runtime="${method_root}/generate_self_imagined_owner_core2_v1.py"
repo_launcher="${method_root}/scripts/auh_generate_self_imagined_owner_core2_dual4.sbatch"
running_launcher="$(realpath -e -- "$0")"
closure=(
  "${runtime}"
  "${method_root}/self_imagined_motion_cotangent_v1.py"
  "${method_root}/infer_native_identity_generation_canary.py"
  "${method_root}/infer_pair_v5_t2v_calibration_bank.py"
  "${method_root}/pair_v5_t2v_calibration_bank_spec.py"
  "${method_root}/infer_lora.py"
  "${method_root}/train_lora.py"
  "${method_root}/infer_source_kv_carrier_oracle.py"
  "${method_root}/infer_source_value_residual_oracle.py"
  "${registry}"
  "${repo_launcher}"
)
for path in "${closure[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "source closure file missing: ${path}"
done
[[ "$(sha256sum "${running_launcher}" | awk '{print $1}')" == "$(sha256sum "${repo_launcher}" | awk '{print $1}')" ]] || fail "running launcher differs from repository launcher"

source_revision="$(git -C "${repo_root}" rev-parse HEAD)"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
source_closure_sha256="$(sha256sum "${closure[@]}" | sha256sum | awk '{print $1}')"

[[ "${output_root}" != / && ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"
output_parent="$(dirname -- "${output_root}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"
output_parent="$(realpath -e -- "${output_parent}")"
[[ "${output_root}" == "${output_parent}/${output_root##*/}" ]] || fail "output root is not canonical"
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

for test_file in test_self_imagined_motion_cotangent_v1.py test_generate_self_imagined_owner_core2_v1.py test_auh_generate_self_imagined_owner_core2_launcher.py; do
  [[ -f "${method_root}/tests/${test_file}" && ! -L "${method_root}/tests/${test_file}" ]] || fail "test missing: ${test_file}"
  PYTHONPATH="${method_root}" "${python_bin}" -B "${method_root}/tests/${test_file}"
done

[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "Slurm ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/self-imagined-owner-${SLURM_JOB_ID}.XXXXXX")"
dog_pid=""
human_pid=""

cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/self-imagined-owner-${SLURM_JOB_ID}."*) ;;
    *) exit 2 ;;
  esac
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}"
    rm -rf -- "${task_scratch}"
  fi
  [[ ! -e "${task_scratch}" ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[self-imagined-owner] PASS all8=true exact81=true owner_count=2 semantic_audit=pending output=${output_root}"
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
dog_port=$((38000 + 2 * job_mod))
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
  exec "${python_bin}" -B -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port="${port}" \
    "${runtime}" generate-cell \
    --registry "${output_root}/sealed-self-imagined-core2.json" \
    --expected-registry-sha256 "${registry_sha256}" \
    --cell-id "${label}" \
    --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --output-dir "${output_root}/${label}" \
    --method-source-revision "${source_revision}" \
    --method-source-archive-sha256 "${source_archive_sha256}"
)

echo "[self-imagined-owner] GPUs0-3=dog GPUs4-7=human pure-T2V exact81 owners=2"
launch_group dog 0,1,2,3 "${dog_port}" >"${output_root}/logs/dog.out" 2>"${output_root}/logs/dog.err" &
dog_pid=$!
launch_group human 4,5,6,7 "${human_port}" >"${output_root}/logs/human.out" 2>"${output_root}/logs/human.err" &
human_pid=$!

dog_status=0
human_status=0
wait "${dog_pid}" || dog_status=$?
wait "${human_pid}" || human_status=$?
[[ "${dog_status}" == 0 && "${human_status}" == 0 ]] || fail "WORLD4 owner group failed dog=${dog_status} human=${human_status}"

"${python_bin}" -B "${runtime}" audit-master \
  --registry "${output_root}/sealed-self-imagined-core2.json" \
  --expected-registry-sha256 "${registry_sha256}" \
  --output-root "${output_root}"

[[ -f "${output_root}/owner-core2-master-receipt.json" ]] || fail "master receipt missing"
