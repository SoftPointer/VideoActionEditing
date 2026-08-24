#!/usr/bin/env bash
#SBATCH --job-name=starc-family-swap-2x4
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

# Generate exactly two prospective confirmation cells.  Each SP4 group owns
# one ten-branch cell and runs its branches serially, so all eight GPUs remain
# occupied while every branch keeps Bernini's native SP4 sampler unchanged.

source_archive="${WRONG_FAMILY_SOURCE_ARCHIVE:?set WRONG_FAMILY_SOURCE_ARCHIVE}"
source_archive_sha256="${WRONG_FAMILY_SOURCE_ARCHIVE_SHA256:?set WRONG_FAMILY_SOURCE_ARCHIVE_SHA256}"
source_revision="${WRONG_FAMILY_SOURCE_REVISION:?set WRONG_FAMILY_SOURCE_REVISION}"
registry="${WRONG_FAMILY_REGISTRY:?set WRONG_FAMILY_REGISTRY}"
source_bank="${WRONG_FAMILY_SOURCE_BANK:?set WRONG_FAMILY_SOURCE_BANK}"
seed_inventory_root="${WRONG_FAMILY_SEED_INVENTORY_ROOT:?set WRONG_FAMILY_SEED_INVENTORY_ROOT}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
output_dir="${WRONG_FAMILY_OUTPUT_DIR:?set WRONG_FAMILY_OUTPUT_DIR}"
python_bin="${WRONG_FAMILY_PYTHON_BIN:?set WRONG_FAMILY_PYTHON_BIN}"

pinned_registry_sha256="f8978311ee7db7f524e827b49747f74ec1a5e0d568e2bbada3fd212225f20cff"
pinned_source_bank_sha256="a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95"

fail() { echo "[starc-family-swap] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }

for name in source_archive registry source_bank seed_inventory_root bernini_root veomni_root checkpoint checkpoint_manifest output_dir python_bin; do
  require_absolute "${!name}" "${name}"
done
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source archive SHA-256 differs"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"

source_archive="$(realpath -e -- "${source_archive}")"
registry="$(realpath -e -- "${registry}")"
source_bank="$(realpath -e -- "${source_bank}")"
seed_inventory_root="$(realpath -e -- "${seed_inventory_root}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive is not plain"
[[ -f "${registry}" && ! -L "${registry}" ]] || fail "registry is not plain"
[[ -f "${source_bank}" && ! -L "${source_bank}" ]] || fail "source bank is not plain"
[[ -d "${seed_inventory_root}" && ! -L "${seed_inventory_root}" ]] || fail "full seed inventory root differs"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini root differs"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni root differs"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint differs"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest differs"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "Python executable differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "$(sha256sum "${registry}" | awk '{print $1}')" == "${pinned_registry_sha256}" ]] || fail "prospective registry hash differs"
[[ "$(sha256sum "${source_bank}" | awk '{print $1}')" == "${pinned_source_bank_sha256}" ]] || fail "source bank hash differs"

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
task_scratch="$(mktemp -d "${scratch_parent%/}/starc-family-swap-${SLURM_JOB_ID}.XXXXXX")"
sp4_a_pid=""; sp4_b_pid=""

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in "${scratch_parent%/}/starc-family-swap-${SLURM_JOB_ID}."*) ;; *) cleanup_failed=1 ;; esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[starc-family-swap] PASS candidates=20 cells=2 exact81=true steps=40 topology=dual-SP4 output=${output_dir}"
  else
    echo "[starc-family-swap] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
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
 "methods/bernini_action_editing/assets/pair_v5_t2v_calibration_core4_bank_v2.json",
 "methods/bernini_action_editing/assets/wrong_family_prompt_swap_pilot_registry_v1.json",
 "methods/bernini_action_editing/wrong_family_prompt_swap_pilot_v1.py",
 "methods/bernini_action_editing/infer_wrong_family_prompt_swap_pilot_v1.py",
 "methods/bernini_action_editing/scripts/auh_generate_wrong_family_prompt_swap_pilot_v1_dual4.sbatch",
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
if missing: raise SystemExit(f"archive lacks wrong-family runtime closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
archived_launcher="${method_root}/scripts/auh_generate_wrong_family_prompt_swap_pilot_v1_dual4.sbatch"
[[ -f "${archived_launcher}" && ! -L "${archived_launcher}" ]] || fail "archived launcher differs"
[[ "$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')" == "$(sha256sum "${archived_launcher}" | awk '{print $1}')" ]] || fail "running launcher differs from source archive"
find "${method_root}" -type f -exec chmod a-w -- {} +

# The caller-declared inventory is scanned before output creation.  The exact
# registry file is added separately so it is the sole allowed fresh-seed hit.
generation_plan_scratch="${task_scratch}/wrong-family-generation-plan.json"
"${python_bin}" -B "${method_root}/wrong_family_prompt_swap_pilot_v1.py" build-plan \
  --registry "${registry}" --expected-registry-sha256 "${pinned_registry_sha256}" \
  --source-bank "${source_bank}" --seed-scan-root "${registry}" \
  --seed-scan-root "${seed_inventory_root}" --output "${generation_plan_scratch}" \
  >"${task_scratch}/generation-plan-build.log"
generation_plan_sha256="$(sha256sum "${generation_plan_scratch}" | awk '{print $1}')"

mkdir -- "${output_dir}"
cp -- "${generation_plan_scratch}" "${output_dir}/sealed-generation-plan.json"
cp -- "${registry}" "${output_dir}/sealed-registry.json"
cp -- "${source_bank}" "${output_dir}/sealed-source-bank.json"
chmod 0444 -- "${output_dir}/sealed-generation-plan.json" "${output_dir}/sealed-registry.json" "${output_dir}/sealed-source-bank.json"
generation_plan="${output_dir}/sealed-generation-plan.json"
runtime_root="${output_dir}/runtime-plan"
"${python_bin}" -B "${method_root}/infer_wrong_family_prompt_swap_pilot_v1.py" materialize-runtime \
  --generation-plan "${generation_plan}" \
  --expected-generation-plan-sha256 "${generation_plan_sha256}" \
  --output-dir "${runtime_root}" >"${output_dir}/runtime-plan-build.log"
runtime_plan="${runtime_root}/wrong-family-runtime-plan-manifest.json"
runtime_plan_sha256="$(sha256sum "${runtime_plan}" | awk '{print $1}')"

echo "[starc-family-swap] seed_preflight=collision-free caller_inventory=${seed_inventory_root} registry_only_allowed=true"
echo "[starc-family-swap] generation=query_captions_false source_condition=false target=false donor=false noise=false mask=false flow=false pose=false track=false"
echo "[starc-family-swap] allocation=8xMI210 topology=sp4-a:0,1,2,3+sp4-b:4,5,6,7 each_cell_ten_serial=true"

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
  local candidate_envelope candidate_id candidate_output
  while IFS= read -r candidate_envelope; do
    [[ -f "${candidate_envelope}" && ! -L "${candidate_envelope}" ]] || fail "candidate envelope differs"
    candidate_id="$("${python_bin}" -B - "${candidate_envelope}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["candidate"]["candidate_id"])
PY
)"
    [[ "${candidate_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "candidate ID differs"
    candidate_output="${output_dir}/${candidate_id}"
    [[ ! -e "${candidate_output}" && ! -L "${candidate_output}" ]] || fail "refusing candidate output reuse"
    echo "[starc-family-swap:${group_id}] START candidate=${candidate_id} visible=${visible_gpus}"
    "${python_bin}" -B -m torch.distributed.run --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="${master_port}" \
      "${method_root}/infer_wrong_family_prompt_swap_pilot_v1.py" render-candidate \
      --generation-plan "${generation_plan}" --expected-generation-plan-sha256 "${generation_plan_sha256}" \
      --runtime-plan "${runtime_plan}" --expected-runtime-plan-sha256 "${runtime_plan_sha256}" \
      --candidate-envelope "${candidate_envelope}" --output-dir "${candidate_output}" \
      --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
      --method-source-revision "${source_revision}" --method-source-archive-sha256 "${source_archive_sha256}"
  done < <(find "${runtime_root}/${group_id}" -mindepth 1 -maxdepth 1 -type f -name '*.json' -print | sort)
)

run_group sp4-a "0,1,2,3" "${sp4_a_port}" >"${output_dir}/sp4-a.log" 2>&1 & sp4_a_pid=$!
run_group sp4-b "4,5,6,7" "${sp4_b_port}" >"${output_dir}/sp4-b.log" 2>&1 & sp4_b_pid=$!
sp4_a_status=0; sp4_b_status=0
wait "${sp4_a_pid}" || sp4_a_status=$?
wait "${sp4_b_pid}" || sp4_b_status=$?
if [[ "${sp4_a_status}" != 0 || "${sp4_b_status}" != 0 ]]; then
  tail -n 160 -- "${output_dir}/sp4-a.log" >&2 || true
  tail -n 160 -- "${output_dir}/sp4-b.log" >&2 || true
  fail "one or more prospective SP4 groups failed"
fi

"${python_bin}" -B "${method_root}/infer_wrong_family_prompt_swap_pilot_v1.py" audit-bank \
  --generation-plan "${generation_plan}" --expected-generation-plan-sha256 "${generation_plan_sha256}" \
  --runtime-plan "${runtime_plan}" --expected-runtime-plan-sha256 "${runtime_plan_sha256}" \
  --output-dir "${output_dir}" >"${output_dir}/strong-audit.log"
[[ -f "${output_dir}/wrong-family-generation-master-receipt.json" ]] || fail "master receipt is absent"
[[ -f "${output_dir}/wrong-family-gaussian-binding.json" ]] || fail "Gaussian binding is absent"
[[ -f "${output_dir}/wrong-family-private-adjudication-plan.json" ]] || fail "private adjudication plan is absent"
[[ -f "${output_dir}/wrong-family-blinded-review-packet.json" ]] || fail "blinded reviewer packet is absent"
echo "STARC_WRONG_FAMILY_PROMPT_SWAP_DUAL4_STRONG_AUDIT_OK"
