#!/usr/bin/env bash
#SBATCH --job-name=cage-mf-obs-2x4
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

# Frozen observation only: 2 families x 2 identities x 2 generation seeds,
# six same-clip views, 30 zero-output LoRA probes, dual SP4.  The built-in
# flow backend is a negative control and the resulting receipt says so.

source_archive="${CAGE_MF_SOURCE_ARCHIVE:?set CAGE_MF_SOURCE_ARCHIVE}"
source_archive_sha256="${CAGE_MF_SOURCE_ARCHIVE_SHA256:?set CAGE_MF_SOURCE_ARCHIVE_SHA256}"
base_spec="${CAGE_MF_BASE_SPEC:?set CAGE_MF_BASE_SPEC}"
base_spec_sha256="${CAGE_MF_BASE_SPEC_SHA256:-a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95}"
base_bank_root="${CAGE_MF_BASE_BANK_ROOT:?set CAGE_MF_BASE_BANK_ROOT}"
base_bank_receipt="${CAGE_MF_BASE_BANK_RECEIPT:?set CAGE_MF_BASE_BANK_RECEIPT}"
base_bank_receipt_sha256="${CAGE_MF_BASE_BANK_RECEIPT_SHA256:?set CAGE_MF_BASE_BANK_RECEIPT_SHA256}"
topup_spec="${CAGE_MF_TOPUP_SPEC:?set CAGE_MF_TOPUP_SPEC}"
topup_spec_sha256="${CAGE_MF_TOPUP_SPEC_SHA256:-be6134679fe2d791fd32e8680eb20370ee13e75f764118bd51d9c0f65b615916}"
topup_output="${CAGE_MF_TOPUP_OUTPUT_DIR:?set CAGE_MF_TOPUP_OUTPUT_DIR}"
event_index="${CAGE_MF_EVENT_INDEX:?set CAGE_MF_EVENT_INDEX}"
event_index_sha256="${CAGE_MF_EVENT_INDEX_SHA256:?set CAGE_MF_EVENT_INDEX_SHA256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
output_root="${CAGE_MF_OBSERVATION_OUTPUT_DIR:?set CAGE_MF_OBSERVATION_OUTPUT_DIR}"
python_bin="${CAGE_MF_PYTHON_BIN:?set CAGE_MF_PYTHON_BIN}"
scalar_backend="${CAGE_MF_SCALAR_BACKEND:?set CAGE_MF_SCALAR_BACKEND}"
external_backend_implementation="${CAGE_MF_EXTERNAL_BACKEND_IMPLEMENTATION:-}"
external_backend_sha256="${CAGE_MF_EXTERNAL_BACKEND_SHA256:-}"

builtin_backend="frozen_flow_log_ratio_to_same_state_noop_negative_control_v1"
external_backend="frozen_text_conditioned_temporal_event_critic_raw_score_vjp_v1"
fail() { echo "[cage-mf-observation] ERROR: $*" >&2; exit 2; }
[[ "${scalar_backend}" == "${builtin_backend}" || "${scalar_backend}" == "${external_backend}" ]] || fail "scalar backend differs"
if [[ "${scalar_backend}" == "${external_backend}" ]]; then
  [[ "${external_backend_implementation}" == /* && "${external_backend_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "external critic must be hash-pinned"
fi
for value in "${source_archive}" "${base_spec}" "${base_bank_root}" "${base_bank_receipt}" "${topup_spec}" "${topup_output}" "${event_index}" "${bernini_root}" "${veomni_root}" "${checkpoint}" "${checkpoint_manifest}" "${output_root}" "${python_bin}"; do
  [[ "${value}" == /* ]] || fail "all paths must be absolute"
done
for digest in "${source_archive_sha256}" "${base_spec_sha256}" "${base_bank_receipt_sha256}" "${topup_spec_sha256}" "${event_index_sha256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || fail "SHA-256 differs"
done

source_archive="$(realpath -e -- "${source_archive}")"
base_spec="$(realpath -e -- "${base_spec}")"
base_bank_root="$(realpath -e -- "${base_bank_root}")"
base_bank_receipt="$(realpath -e -- "${base_bank_receipt}")"
topup_spec="$(realpath -e -- "${topup_spec}")"
topup_output="$(realpath -e -- "${topup_output}")"
event_index="$(realpath -e -- "${event_index}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(sha256sum "${base_spec}" | awk '{print $1}')" == "${base_spec_sha256}" ]] || fail "base spec hash differs"
[[ "$(sha256sum "${base_bank_receipt}" | awk '{print $1}')" == "${base_bank_receipt_sha256}" ]] || fail "base bank receipt hash differs"
[[ "$(sha256sum "${topup_spec}" | awk '{print $1}')" == "${topup_spec_sha256}" ]] || fail "top-up spec hash differs"
[[ "$(sha256sum "${event_index}" | awk '{print $1}')" == "${event_index_sha256}" ]] || fail "event index hash differs"
[[ -d "${base_bank_root}" && -d "${topup_output}" && -d "${checkpoint}" ]] || fail "input directory differs"
[[ -x "${python_bin}" ]] || fail "Python executable differs"
[[ "${output_root}" != / && "$(realpath -m -- "${output_root}")" == "${output_root}" ]] || fail "output path differs"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "refusing output reuse"
mkdir -p -- "$(dirname -- "${output_root}")"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

scratch_parent="${SLURM_TMPDIR:-/tmp}"
task_scratch="$(mktemp -d "${scratch_parent%/}/cage-mf-observation-${SLURM_JOB_ID:?}.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch}" in "${scratch_parent%/}/cage-mf-observation-${SLURM_JOB_ID}."*) chmod -R u+w -- "${task_scratch}"; rm -rf -- "${task_scratch}" ;; *) status=2 ;; esac
  if [[ "${status}" == 0 ]]; then
    echo "CAGE_MOTION_FISHER_OBSERVATION_DUAL4_OK clips=8 views=6 blocks=30 exact81=true optimizer=false backend=${scalar_backend} output=${output_root}"
  fi
  exit "${status}"
}
trap cleanup EXIT

mkdir -p -- "${task_scratch}/source"
tar --no-same-owner --no-same-permissions -xf "${source_archive}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
probe="${method_root}/cage_t2v_motion_fisher_observation_probe.py"
registration="${method_root}/assets/cage_t2v_motion_fisher_observation_probe_v1.json"
[[ -f "${probe}" && -f "${registration}" ]] || fail "source archive lacks observation closure"
registration_sha256="2f21c3036f1320ce87067ae969be3df8aa4d1fd03abc8201903bc20a1b7b8fa7"
[[ "$(sha256sum "${registration}" | awk '{print $1}')" == "${registration_sha256}" ]] || fail "registration hash differs"

common_args=(
  --registration "${registration}" --expected-registration-sha256 "${registration_sha256}"
  --root-spec "${base_spec}" --expected-root-spec-sha256 "${base_spec_sha256}"
  --bank-output-dir "${base_bank_root}" --bank-receipt "${base_bank_receipt}"
  --expected-bank-receipt-sha256 "${base_bank_receipt_sha256}"
  --topup-spec "${topup_spec}" --expected-topup-spec-sha256 "${topup_spec_sha256}"
  --topup-output-dir "${topup_output}"
  --event-index "${event_index}" --expected-event-index-sha256 "${event_index_sha256}"
)
"${python_bin}" -B "${probe}" preflight "${common_args[@]}" --group-id sp4-a >"${task_scratch}/preflight-a.json"
"${python_bin}" -B "${probe}" preflight "${common_args[@]}" --group-id sp4-b >"${task_scratch}/preflight-b.json"
mkdir -- "${output_root}"

backend_args=(--scalar-backend "${scalar_backend}" --ack-observation-only)
if [[ "${scalar_backend}" == "${builtin_backend}" ]]; then
  backend_args+=(--ack-negative-control)
else
  external_backend_implementation="$(realpath -e -- "${external_backend_implementation}")"
  [[ "$(sha256sum "${external_backend_implementation}" | awk '{print $1}')" == "${external_backend_sha256}" ]] || fail "external critic hash differs"
  backend_args+=(--external-backend-implementation "${external_backend_implementation}" --expected-external-backend-sha256 "${external_backend_sha256}")
fi

job_mod=$((SLURM_JOB_ID % 10000))
run_probe() (
  set -Eeuo pipefail
  local group_id="$1" visible="$2" port="$3" group_output="$4"
  local group_scratch="${task_scratch}/${group_id}"
  mkdir -p -- "${group_scratch}/tmp" "${group_scratch}/cache"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible}"
  export TMPDIR="${group_scratch}/tmp" XDG_CACHE_HOME="${group_scratch}/cache"
  "${python_bin}" -B -m torch.distributed.run --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="${port}" \
    "${probe}" probe "${common_args[@]}" --group-id "${group_id}" \
    --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
    --output-dir "${group_output}" "${backend_args[@]}"
)

run_probe sp4-a "0,1,2,3" "$((34000 + 2 * job_mod))" "${output_root}/sp4-a" >"${output_root}/sp4-a.log" 2>&1 & pid_a=$!
run_probe sp4-b "4,5,6,7" "$((34001 + 2 * job_mod))" "${output_root}/sp4-b" >"${output_root}/sp4-b.log" 2>&1 & pid_b=$!
status_a=0; status_b=0
wait "${pid_a}" || status_a=$?
wait "${pid_b}" || status_b=$?
[[ "${status_a}" == 0 && "${status_b}" == 0 ]] || fail "one SP4 observation arm failed"

"${python_bin}" -B "${probe}" aggregate --sp4-a-dir "${output_root}/sp4-a" \
  --sp4-b-dir "${output_root}/sp4-b" --output-dir "${output_root}/master" \
  >"${output_root}/aggregate.json"
[[ -f "${output_root}/master/motion-fisher-observation-master.json" ]] || fail "master receipt absent"

