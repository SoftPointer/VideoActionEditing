#!/usr/bin/env bash
#SBATCH --job-name=cage-mf-topup-2x4
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

# Generates exactly four *action-only* second-seed T2V clips.  It deliberately
# does not regenerate the nine MACE negatives.  The output is still not
# eligible for the Motion-Fisher probe until four detached event-positive
# receipts have been authored and sealed into an eight-action event index.

source_archive="${CAGE_MF_SOURCE_ARCHIVE:?set CAGE_MF_SOURCE_ARCHIVE}"
source_archive_sha256="${CAGE_MF_SOURCE_ARCHIVE_SHA256:?set CAGE_MF_SOURCE_ARCHIVE_SHA256}"
source_revision="${CAGE_MF_SOURCE_REVISION:?set CAGE_MF_SOURCE_REVISION}"
base_spec="${CAGE_MF_BASE_SPEC:?set CAGE_MF_BASE_SPEC}"
base_spec_sha256="${CAGE_MF_BASE_SPEC_SHA256:-a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95}"
topup_spec="${CAGE_MF_TOPUP_SPEC:?set CAGE_MF_TOPUP_SPEC}"
topup_spec_sha256="${CAGE_MF_TOPUP_SPEC_SHA256:-be6134679fe2d791fd32e8680eb20370ee13e75f764118bd51d9c0f65b615916}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
output_dir="${CAGE_MF_TOPUP_OUTPUT_DIR:?set CAGE_MF_TOPUP_OUTPUT_DIR}"
python_bin="${CAGE_MF_PYTHON_BIN:?set CAGE_MF_PYTHON_BIN}"

fail() { echo "[cage-mf-topup] ERROR: $*" >&2; exit 2; }
for value in "${source_archive}" "${base_spec}" "${topup_spec}" "${bernini_root}" "${veomni_root}" "${checkpoint}" "${checkpoint_manifest}" "${output_dir}" "${python_bin}"; do
  [[ "${value}" == /* ]] || fail "all paths must be absolute"
done
for digest in "${source_archive_sha256}" "${base_spec_sha256}" "${topup_spec_sha256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || fail "SHA-256 differs"
done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"

source_archive="$(realpath -e -- "${source_archive}")"
base_spec="$(realpath -e -- "${base_spec}")"
topup_spec="$(realpath -e -- "${topup_spec}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "$(sha256sum "${base_spec}" | awk '{print $1}')" == "${base_spec_sha256}" ]] || fail "base spec hash differs"
[[ "$(sha256sum "${topup_spec}" | awk '{print $1}')" == "${topup_spec_sha256}" ]] || fail "top-up spec hash differs"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini root differs"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni root differs"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint differs"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest differs"
[[ -x "${python_bin}" ]] || fail "Python executable differs"
[[ "${output_dir}" != / && "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output path differs"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing output reuse"
mkdir -p -- "$(dirname -- "${output_dir}")"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

scratch_parent="${SLURM_TMPDIR:-/tmp}"
task_scratch="$(mktemp -d "${scratch_parent%/}/cage-mf-topup-${SLURM_JOB_ID:?}.XXXXXX")"
sp4_a_pid=""; sp4_b_pid=""
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch}" in "${scratch_parent%/}/cage-mf-topup-${SLURM_JOB_ID}."*) chmod -R u+w -- "${task_scratch}"; rm -rf -- "${task_scratch}" ;; *) status=2 ;; esac
  if [[ "${status}" == 0 ]]; then
    echo "CAGE_MOTION_FISHER_ACTION_TOPUP_DUAL4_OK candidates=4 exact81=true steps=40 event_audit_pending=true output=${output_dir}"
  fi
  exit "${status}"
}
trap cleanup EXIT

mkdir -p -- "${task_scratch}/source"
tar --no-same-owner --no-same-permissions -xf "${source_archive}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
for required in cage_t2v_motion_fisher_observation_probe.py infer_pair_v5_t2v_calibration_bank.py infer_native_identity_generation_canary.py; do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive lacks ${required}"
done
plan_dir="${task_scratch}/topup-plan"
"${python_bin}" -B "${method_root}/cage_t2v_motion_fisher_observation_probe.py" materialize-topup-plan \
  --topup-spec "${topup_spec}" --expected-topup-spec-sha256 "${topup_spec_sha256}" \
  --root-spec "${base_spec}" --expected-root-spec-sha256 "${base_spec_sha256}" \
  --output-dir "${plan_dir}" >"${task_scratch}/topup-plan.log"
mkdir -- "${output_dir}"
cp -- "${topup_spec}" "${output_dir}/sealed-action-topup-spec.json"
chmod 0444 -- "${output_dir}/sealed-action-topup-spec.json"

job_mod=$((SLURM_JOB_ID % 10000))
run_group() (
  set -Eeuo pipefail
  local group_id="$1" visible="$2" port="$3" envelope candidate_id candidate_output
  local group_scratch="${task_scratch}/${group_id}"
  mkdir -p -- "${group_scratch}/tmp" "${group_scratch}/cache"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible}"
  export TMPDIR="${group_scratch}/tmp" XDG_CACHE_HOME="${group_scratch}/cache"
  while IFS= read -r envelope; do
    candidate_id="$("${python_bin}" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate"]["candidate_id"])' "${envelope}")"
    [[ "${candidate_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$ ]] || fail "candidate ID differs"
    candidate_output="${output_dir}/${candidate_id}"
    "${python_bin}" -B -m torch.distributed.run --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="${port}" \
      "${method_root}/infer_pair_v5_t2v_calibration_bank.py" \
      --candidate-spec "${envelope}" --expected-root-spec-sha256 "${topup_spec_sha256}" \
      --output-dir "${candidate_output}" --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
      --method-source-revision "${source_revision}" --method-source-archive-sha256 "${source_archive_sha256}"
  done < <(find "${plan_dir}/${group_id}" -type f -name '*.json' -print | sort)
)

run_group sp4-a "0,1,2,3" "$((32000 + 2 * job_mod))" >"${output_dir}/sp4-a.log" 2>&1 & sp4_a_pid=$!
run_group sp4-b "4,5,6,7" "$((32001 + 2 * job_mod))" >"${output_dir}/sp4-b.log" 2>&1 & sp4_b_pid=$!
status_a=0; status_b=0
wait "${sp4_a_pid}" || status_a=$?
wait "${sp4_b_pid}" || status_b=$?
[[ "${status_a}" == 0 && "${status_b}" == 0 ]] || fail "one SP4 top-up arm failed"

"${python_bin}" -B "${method_root}/cage_t2v_motion_fisher_observation_probe.py" audit-topup \
  --topup-spec "${topup_spec}" --expected-topup-spec-sha256 "${topup_spec_sha256}" \
  --root-spec "${base_spec}" --expected-root-spec-sha256 "${base_spec_sha256}" \
  --topup-output-dir "${output_dir}" >"${output_dir}/topup-strong-audit.json"

