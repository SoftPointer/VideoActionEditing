#!/usr/bin/env bash
# Run one heldout source as concurrent frozen/adapted SP4 arms inside an idle
# eight-MI210 allocation.  Launch this controller once per source/allocation.

set -Eeuo pipefail
umask 077

existing_job_id="${BERNINI_HELDOUT_EXISTING_SLURM_JOB_ID:?set existing allocation job ID}"
existing_node="${BERNINI_HELDOUT_EXISTING_SLURM_NODE:?set existing allocation node}"
iid="${BERNINI_HELDOUT_IID:?set one sealed heldout IID}"
source_archive="${BERNINI_HELDOUT_METHOD_SOURCE_ARCHIVE:?set method source archive}"
source_archive_sha256="${BERNINI_HELDOUT_METHOD_SOURCE_ARCHIVE_SHA256:?set method archive SHA-256}"
source_revision="${BERNINI_HELDOUT_METHOD_SOURCE_REVISION:?set method source revision}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set Bernini source root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set VeOmni source root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set base checkpoint}"
adapter_checkpoint="${BERNINI_ACTION_ADAPTER_CHECKPOINT:?set trained adapter checkpoint}"
output_root="${BERNINI_HELDOUT_OUTPUT_ROOT:?set fresh output root}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
worker_cpus="${BERNINI_HELDOUT_WORKER_CPUS:-32}"
worker_mem="${BERNINI_HELDOUT_WORKER_MEM:-56G}"
readonly trained_infer_runner="infer_seer_scoped_lora.py"

readonly expected_spec_sha256="82fbe0f042d86f8d54aa254ce72a384e70aa5bdc3c1ac66d5422037cd4b4051c"

fail() { echo "[heldout-pair-existing-job] ERROR: $*" >&2; exit 2; }

[[ "${existing_job_id}" =~ ^[1-9][0-9]*$ ]] || fail "existing job ID differs"
[[ "${existing_node}" =~ ^auh[0-9A-Za-z-]+$ ]] || fail "existing node differs"
case "${iid}" in
  99cde432839f4240|6ea45d35943742bb|311c82f83eca4a7f|6d346c38cf504493) ;;
  *) fail "IID is not one immutable core4 confirmation source" ;;
esac
[[ -z "${BERNINI_HELDOUT_TRAINED_INFER_RUNNER:-}" ]] || \
  fail "trained inference runner override is forbidden for fixed B0 evaluation"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source archive SHA-256 differs"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${worker_cpus}" =~ ^[1-9][0-9]*$ ]] || fail "worker CPU count differs"
[[ "${worker_mem}" =~ ^[1-9][0-9]*G$ ]] || fail "worker memory differs"
for path in source_archive bernini_root veomni_root checkpoint adapter_checkpoint output_root python_bin; do
  [[ "${!path}" == /* && "${!path}" != / ]] || fail "${path} must be absolute and non-root"
done
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive bytes differ"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini root differs"
[[ -d "${veomni_root}" ]] || fail "VeOmni root differs"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint differs"
[[ -d "${adapter_checkpoint}" && ! -L "${adapter_checkpoint}" ]] || fail "adapter checkpoint differs"
[[ -x "${python_bin}" ]] || fail "Python executable differs"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"
[[ -d "${output_root%/*}" && -w "${output_root%/*}" ]] || fail "output parent differs"

job_record="$(scontrol show job -o "${existing_job_id}")"
for required in "JobState=RUNNING" "NodeList=${existing_node}" "gres/gpu:mi210=8"; do
  [[ "${job_record}" == *"${required}"* ]] || fail "allocation binding differs: ${required}"
done
if squeue -s -j "${existing_job_id}" -h -o '%i' | grep -Eq '[.][0-9]+$'; then
  fail "existing allocation already has a numbered Slurm step"
fi

mkdir -m 0700 "${output_root}"
source_root="${output_root}/source"
results_root="${output_root}/results"
cache_root="${output_root}/cache"
mkdir -m 0700 "${source_root}" "${results_root}" "${cache_root}"
tar --delay-directory-restore --no-same-owner --no-same-permissions \
  -xf "${source_archive}" -C "${source_root}"
method_root="${source_root}/methods/bernini_action_editing"
runner="${method_root}/run_self_generated_action_lora_heldout_core4_v1.py"
spec="${method_root}/assets/self_generated_action_lora_heldout_core4_v1.json"
[[ -f "${runner}" && ! -L "${runner}" ]] || fail "archive lacks heldout runner"
[[ -f "${spec}" && ! -L "${spec}" ]] || fail "archive lacks heldout spec"
[[ "$(sha256sum "${spec}" | awk '{print $1}')" == "${expected_spec_sha256}" ]] || fail "heldout spec bytes differ"

"${python_bin}" -B "${runner}" \
  --spec "${spec}" --expected-spec-sha256 "${expected_spec_sha256}" \
  inspect --verify-files

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 SLURM_EXPORT_ENV=ALL

run_arm() {
  local arm="$1" port="$2" cache
  cache="${cache_root}/${arm}"
  mkdir -m 0700 -p \
    "${cache}/tmp" "${cache}/xdg" "${cache}/torch-extensions" \
    "${cache}/triton" "${cache}/miopen-user" "${cache}/miopen-custom"
  local adaptation=()
  if [[ "${arm}" == trained_adapter ]]; then
    adaptation=(--adapter-checkpoint "${adapter_checkpoint}")
  fi
  srun \
    --jobid="${existing_job_id}" --nodelist="${existing_node}" \
    --nodes=1 --exclusive --exact --kill-on-bad-exit=1 --ntasks=1 \
    --cpus-per-task="${worker_cpus}" --mem="${worker_mem}" \
    --gres=gpu:mi210:4 \
    env \
      TMPDIR="${cache}/tmp" XDG_CACHE_HOME="${cache}/xdg" \
      TORCH_EXTENSIONS_DIR="${cache}/torch-extensions" \
      TRITON_CACHE_DIR="${cache}/triton" \
      MIOPEN_USER_DB_PATH="${cache}/miopen-user" \
      MIOPEN_CUSTOM_CACHE_DIR="${cache}/miopen-custom" \
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      "${python_bin}" -B "${runner}" \
        --spec "${spec}" --expected-spec-sha256 "${expected_spec_sha256}" \
        run-arm --iid "${iid}" --arm "${arm}" \
        --method-root "${method_root}" --python-bin "${python_bin}" \
        --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
        --checkpoint "${checkpoint}" "${adaptation[@]}" \
        --trained-infer-runner "${trained_infer_runner}" \
        --output-root "${results_root}" --master-port "${port}" \
        --method-source-revision "${source_revision}" \
        --method-source-archive-sha256 "${source_archive_sha256}"
}

echo "[heldout-pair-existing-job] START job=${existing_job_id} node=${existing_node} iid=${iid} arms=base+trained"
base_status=0
trained_status=0
run_arm frozen_base 29431 >"${output_root}/frozen-base.log" 2>&1 || base_status=$?
if (( base_status == 0 )); then
  run_arm trained_adapter 29432 >"${output_root}/trained-adapter.log" 2>&1 || trained_status=$?
fi
if (( base_status != 0 || trained_status != 0 )); then
  tail -n 160 "${output_root}/frozen-base.log" >&2 || true
  tail -n 160 "${output_root}/trained-adapter.log" >&2 || true
  fail "paired inference arm failed: base=${base_status} trained=${trained_status}"
fi

verify_args=(
  --spec "${spec}" --expected-spec-sha256 "${expected_spec_sha256}"
  verify-pair --iid "${iid}" --adapter-checkpoint "${adapter_checkpoint}"
  --output-root "${results_root}"
)
if command -v ffmpeg >/dev/null 2>&1; then
  verify_args+=(--ffmpeg "$(command -v ffmpeg)")
fi
"${python_bin}" -B "${runner}" "${verify_args[@]}" \
  >"${output_root}/paired-verification.json"

echo "[heldout-pair-existing-job] PASS_GENERATION_NOT_METHOD_SUCCESS job=${existing_job_id} iid=${iid} result=${results_root}/${iid}"
