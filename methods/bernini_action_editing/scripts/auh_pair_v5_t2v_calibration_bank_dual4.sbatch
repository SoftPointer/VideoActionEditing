#!/usr/bin/env bash
#SBATCH --job-name=pair5-t2v-bank-2x4
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=24:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# Auditable first-bank path: the two SP4 groups run concurrently and candidates
# are serial inside a group.  This pilot reloads frozen Bernini per branch; a
# later persistent worker may remove load overhead without changing receipts.
# The preregistered first job is the 40-branch ``core4`` selection asset; the
# 80-branch first8 registry is an authoring registry, not the default GPU job.

source_archive="${PAIR_V5_T2V_SOURCE_ARCHIVE:?set PAIR_V5_T2V_SOURCE_ARCHIVE}"
source_archive_sha256="${PAIR_V5_T2V_SOURCE_ARCHIVE_SHA256:?set PAIR_V5_T2V_SOURCE_ARCHIVE_SHA256}"
source_revision="${PAIR_V5_T2V_SOURCE_REVISION:?set PAIR_V5_T2V_SOURCE_REVISION}"
bank_spec="${PAIR_V5_T2V_BANK_SPEC:?set PAIR_V5_T2V_BANK_SPEC}"
bank_spec_sha256="${PAIR_V5_T2V_BANK_SPEC_SHA256:?set PAIR_V5_T2V_BANK_SPEC_SHA256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
output_dir="${PAIR_V5_T2V_BANK_OUTPUT_DIR:?set PAIR_V5_T2V_BANK_OUTPUT_DIR}"
python_bin="${PAIR_V5_T2V_PYTHON_BIN:?set PAIR_V5_T2V_PYTHON_BIN}"

fail() { echo "[pair-v5-t2v-bank] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }

for name in source_archive bank_spec bernini_root veomni_root checkpoint checkpoint_manifest output_dir python_bin; do
  require_absolute "${!name}" "${name}"
done
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source archive SHA-256 differs"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${bank_spec_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "bank spec SHA-256 differs"

source_archive="$(realpath -e -- "${source_archive}")"
bank_spec="$(realpath -e -- "${bank_spec}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive is not plain"
[[ -f "${bank_spec}" && ! -L "${bank_spec}" ]] || fail "bank spec is not plain"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini root differs"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni root differs"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint differs"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest differs"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "Python executable differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "$(sha256sum "${bank_spec}" | awk '{print $1}')" == "${bank_spec_sha256}" ]] || fail "sealed bank spec hash differs"

[[ "${output_dir}" != / ]] || fail "output directory cannot be root"
output_basename="${output_dir##*/}"
[[ "${output_basename}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename is unsafe"
[[ "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output directory must be canonical"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing output reuse"
output_parent="$(dirname -- "${output_dir}")"
mkdir -p -- "${output_parent}"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/pair-v5-t2v-${SLURM_JOB_ID}.XXXXXX")"
sp4_a_pid=""; sp4_b_pid=""

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in "${scratch_parent%/}/pair-v5-t2v-${SLURM_JOB_ID}."*) ;; *) cleanup_failed=1 ;; esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[pair-v5-t2v-bank] PASS topology=two-concurrent-SP4 candidates=serial exact81=true steps=40 calibration_only=true output=${output_dir}"
  else
    echo "[pair-v5-t2v-bank] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${status}"
}
terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${sp4_a_pid}" "${sp4_b_pid}"; do [[ "${child}" =~ ^[0-9]+$ ]] && kill "${child}" 2>/dev/null || true; done
  for child in "${sp4_a_pid}" "${sp4_b_pid}"; do [[ "${child}" =~ ^[0-9]+$ ]] && wait "${child}" 2>/dev/null || true; done
  exit "${status}"
}
trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -p -- "${task_scratch}/source-tree" "${task_scratch}/groups"
archive_copy="${task_scratch}/method-source.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod 0444 -- "${archive_copy}"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile
required = {
 "methods/bernini_action_editing/pair_v5_t2v_calibration_bank_spec.py",
 "methods/bernini_action_editing/infer_pair_v5_t2v_calibration_bank.py",
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
seen=set()
with tarfile.open(sys.argv[1], "r:*") as handle:
 for member in handle.getmembers():
  path=PurePosixPath(member.name)
  if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
   raise SystemExit("archive member escaped or is not a plain tree entry")
  seen.add(path.as_posix().lstrip("./"))
missing=sorted(required-seen)
if missing: raise SystemExit(f"archive lacks PAIR-v5 T2V bank closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
find "${method_root}" -type f -exec chmod a-w -- {} +
plan_root="${task_scratch}/plan"
"${python_bin}" -B "${method_root}/pair_v5_t2v_calibration_bank_spec.py" \
  --spec "${bank_spec}" --expected-sha256 "${bank_spec_sha256}" --output-dir "${plan_root}" \
  >"${task_scratch}/plan-materialization.json"

mkdir -- "${output_dir}"
cp -- "${bank_spec}" "${output_dir}/sealed-bank-spec.json"
chmod 0444 -- "${output_dir}/sealed-bank-spec.json"
echo "[pair-v5-t2v-bank] semantic_input=full_t2v_caption geometry_source=bucket_shape_only source_condition=false target=false donor=false pseudo_target=false student_input=false custom_noise=false mask=false flow=false pose=false track=false trajectory=false"
echo "[pair-v5-t2v-bank] allocation=8xMI210 topology=sp4-a:0,1,2,3+sp4-b:4,5,6,7 candidates_serial_within_group=true same_cell_gaussian=required"

job_mod=$((SLURM_JOB_ID % 10000))
sp4_a_port=$((30000 + 2 * job_mod))
sp4_b_port=$((sp4_a_port + 1))

run_group() (
  set -Eeuo pipefail
  local group_id="$1" visible_gpus="$2" master_port="$3"
  local group_scratch="${task_scratch}/groups/${group_id}"
  mkdir -p -- "${group_scratch}/cache/miopen-user" "${group_scratch}/cache/miopen-custom" \
    "${group_scratch}/cache/torch-extensions" "${group_scratch}/cache/triton" \
    "${group_scratch}/cache/xdg" "${group_scratch}/cache/pycache" "${group_scratch}/tmp"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible_gpus}"
  export MIOPEN_USER_DB_PATH="${group_scratch}/cache/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${group_scratch}/cache/miopen-custom"
  export TORCH_EXTENSIONS_DIR="${group_scratch}/cache/torch-extensions"
  export TRITON_CACHE_DIR="${group_scratch}/cache/triton"
  export XDG_CACHE_HOME="${group_scratch}/cache/xdg"
  export PYTHONPYCACHEPREFIX="${group_scratch}/cache/pycache"
  export TMPDIR="${group_scratch}/tmp"
  export TORCHELASTIC_ERROR_FILE="${group_scratch}/torch-elastic-error.json"
  local candidate_spec candidate_id candidate_output
  while IFS= read -r candidate_spec; do
    [[ -f "${candidate_spec}" && ! -L "${candidate_spec}" ]] || fail "candidate envelope differs"
    candidate_id="$("${python_bin}" -B - "${candidate_spec}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["candidate"]["candidate_id"])
PY
)"
    [[ "${candidate_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || fail "candidate ID differs"
    candidate_output="${output_dir}/${candidate_id}"
    [[ ! -e "${candidate_output}" && ! -L "${candidate_output}" ]] || fail "refusing candidate output reuse"
    echo "[pair-v5-t2v-bank:${group_id}] START candidate=${candidate_id} visible=${visible_gpus}"
    "${python_bin}" -B -m torch.distributed.run --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="${master_port}" \
      "${method_root}/infer_pair_v5_t2v_calibration_bank.py" \
      --candidate-spec "${candidate_spec}" --expected-root-spec-sha256 "${bank_spec_sha256}" \
      --output-dir "${candidate_output}" --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
      --method-source-revision "${source_revision}" --method-source-archive-sha256 "${source_archive_sha256}"
  done < <(find "${plan_root}/${group_id}" -mindepth 1 -maxdepth 1 -type f -name '*.json' -print | sort)
)

run_group sp4-a "0,1,2,3" "${sp4_a_port}" >"${output_dir}/sp4-a.log" 2>&1 & sp4_a_pid=$!
run_group sp4-b "4,5,6,7" "${sp4_b_port}" >"${output_dir}/sp4-b.log" 2>&1 & sp4_b_pid=$!
sp4_a_status=0; sp4_b_status=0
wait "${sp4_a_pid}" || sp4_a_status=$?
wait "${sp4_b_pid}" || sp4_b_status=$?
if [[ "${sp4_a_status}" != 0 || "${sp4_b_status}" != 0 ]]; then
  tail -n 160 -- "${output_dir}/sp4-a.log" >&2 || true
  tail -n 160 -- "${output_dir}/sp4-b.log" >&2 || true
  fail "one or more SP4 T2V groups failed"
fi

"${python_bin}" -B "${method_root}/infer_pair_v5_t2v_calibration_bank.py" \
  --audit-bank --root-spec "${bank_spec}" \
  --expected-root-spec-sha256 "${bank_spec_sha256}" --output-dir "${output_dir}" \
  >"${output_dir}/strong-audit.log"
[[ -f "${output_dir}/pair-v5-t2v-calibration-bank-receipt.json" ]] || fail "bank receipt is absent"
echo "PAIR_V5_T2V_CALIBRATION_BANK_DUAL4_STRONG_AUDIT_OK"
