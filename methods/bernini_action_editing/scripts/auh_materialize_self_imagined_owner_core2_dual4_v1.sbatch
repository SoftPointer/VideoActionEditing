#!/usr/bin/env bash
#SBATCH --job-name=qm-own-q-v1
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=06:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# job131524 owner materialization only.  A signed detached full81 audit is
# authenticated before either torchrun is started.  Dog owns GPUs0-3 and human
# owns GPUs4-7 concurrently.  The final tree appears through one same-filesystem
# rename and contains only four detached normalized quotients and specificity
# receipts; owner RGB/clean/noise/text/hidden/velocity never enters that tree.

source_archive="${SIM_OWNER_MAT_SOURCE_ARCHIVE:?set source git archive}"
source_archive_sha256="${SIM_OWNER_MAT_SOURCE_ARCHIVE_SHA256:?set archive SHA-256}"
source_revision="${SIM_OWNER_MAT_SOURCE_REVISION:?set archive git revision}"
registry="${SIM_OWNER_MAT_REGISTRY:?set sealed core2 registry}"
registry_sha256="${SIM_OWNER_MAT_REGISTRY_SHA256:?set registry SHA-256}"
owner_root="${SIM_OWNER_JOB131524_ROOT:?set job131524 owner root}"
owner_master="${SIM_OWNER_JOB131524_MASTER:?set job131524 master receipt}"
owner_master_sha256="${SIM_OWNER_JOB131524_MASTER_SHA256:?set master receipt SHA-256}"
audit_sidecar="${SIM_OWNER_FULL81_AUDIT_SIDECAR:?set signed full81 audit sidecar}"
audit_sidecar_sha256="${SIM_OWNER_FULL81_AUDIT_SIDECAR_SHA256:?set sidecar SHA-256}"
audit_evidence="${SIM_OWNER_FULL81_AUDIT_EVIDENCE:?set detached audit evidence}"
audit_public_key="${SIM_OWNER_FULL81_AUDIT_PUBLIC_KEY:?set Ed25519 public key}"
audit_public_key_sha256="${SIM_OWNER_FULL81_AUDIT_PUBLIC_KEY_SHA256:?set public-key SHA-256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
output_root="${SIM_OWNER_MAT_OUTPUT_ROOT:?set fresh final output root}"
python_bin="${SIM_OWNER_MAT_PYTHON_BIN:?set vace Python executable}"

readonly required_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly required_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d

fail() { echo "[owner-quotient-v1] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }
require_sha1() { [[ "$1" =~ ^[0-9a-f]{40}$ ]] || fail "$2 must be lowercase SHA-1"; }

for name in source_archive registry owner_root owner_master audit_sidecar audit_evidence audit_public_key bernini_root veomni_root checkpoint checkpoint_manifest output_root python_bin; do
  require_absolute "${!name}" "${name}"
done
for name in source_archive_sha256 registry_sha256 owner_master_sha256 audit_sidecar_sha256 audit_public_key_sha256; do
  require_sha256 "${!name}" "${name}"
done
require_sha1 "${source_revision}" source_revision

source_archive="$(realpath -e -- "${source_archive}")"
registry="$(realpath -e -- "${registry}")"
owner_root="$(realpath -e -- "${owner_root}")"
owner_master="$(realpath -e -- "${owner_master}")"
audit_sidecar="$(realpath -e -- "${audit_sidecar}")"
audit_evidence="$(realpath -e -- "${audit_evidence}")"
audit_public_key="$(realpath -e -- "${audit_public_key}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"

for plain_file in "${source_archive}" "${registry}" "${owner_master}" "${audit_sidecar}" "${audit_evidence}" "${audit_public_key}" "${checkpoint_manifest}"; do
  [[ -f "${plain_file}" && ! -L "${plain_file}" ]] || fail "plain input file differs: ${plain_file}"
done
for plain_dir in "${owner_root}" "${bernini_root}" "${veomni_root}" "${checkpoint}"; do
  [[ -d "${plain_dir}" && ! -L "${plain_dir}" ]] || fail "plain input directory differs: ${plain_dir}"
done
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python executable differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive changed"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "$(sha256sum "${registry}" | awk '{print $1}')" == "${registry_sha256}" ]] || fail "registry changed"
[[ "$(sha256sum "${owner_master}" | awk '{print $1}')" == "${owner_master_sha256}" ]] || fail "job131524 master changed"
[[ "$(sha256sum "${audit_sidecar}" | awk '{print $1}')" == "${audit_sidecar_sha256}" ]] || fail "signed audit sidecar changed"
[[ "$(sha256sum "${audit_public_key}" | awk '{print $1}')" == "${audit_public_key_sha256}" ]] || fail "audit public key changed"

[[ "${output_root}" != / && "$(realpath -m -- "${output_root}")" == "${output_root}" ]] || fail "output root must be canonical non-root"
[[ "${output_root##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename differs"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"
output_parent="$(dirname -- "${output_root}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" && -w "${output_parent}" ]] || fail "output parent differs"
output_parent="$(realpath -e -- "${output_parent}")"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/owner-quotient-v1-${SLURM_JOB_ID}.XXXXXX")"
staging_root="${output_parent}/.${output_root##*/}.staging-${SLURM_JOB_ID}"
[[ ! -e "${staging_root}" && ! -L "${staging_root}" ]] || fail "staging root already exists"
dog_pid=""
human_pid=""
published=false

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/owner-quotient-v1-${SLURM_JOB_ID}."*) ;;
    *) cleanup_failed=1 ;;
  esac
  case "${staging_root:-}" in
    "${output_parent}/.${output_root##*/}.staging-${SLURM_JOB_ID}") ;;
    *) cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  if [[ "${published}" != true && "${cleanup_failed}" == 0 && -d "${staging_root}" && ! -L "${staging_root}" ]]; then
    chmod -R u+w -- "${staging_root}" || cleanup_failed=1
    rm -rf -- "${staging_root}" || cleanup_failed=1
  fi
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[owner-quotient-v1] PASS all8=true topology=dog_WORLD4+human_WORLD4 query_seeds=4 seed_selection=false seed_averaging=false output=${output_root}"
  fi
  exit "${status}"
}
terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${dog_pid}" "${human_pid}"; do [[ "${child}" =~ ^[0-9]+$ ]] && kill "${child}" 2>/dev/null || true; done
  for child in "${dog_pid}" "${human_pid}"; do [[ "${child}" =~ ^[0-9]+$ ]] && wait "${child}" 2>/dev/null || true; done
  exit "${status}"
}
trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -p -- "${task_scratch}/source-tree" "${task_scratch}/groups" "${task_scratch}/logs"
archive_copy="${task_scratch}/method-source.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod 0444 -- "${archive_copy}"

# Inspect every selected archive member before extraction.  Links, devices,
# FIFOs, absolute paths and parent traversal are forbidden.
"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile

required = {
    "methods/bernini_action_editing/materialize_self_imagined_owner_core2_v1.py",
    "methods/bernini_action_editing/materialize_starc_core4_hidden_v1.py",
    "methods/bernini_action_editing/self_imagined_motion_cotangent_v1.py",
    "methods/bernini_action_editing/generate_self_imagined_owner_core2_v1.py",
    "methods/bernini_action_editing/score_pair_v5_t2v_energy_bank_frozen_d541801.py",
    "methods/bernini_action_editing/scripts/auh_materialize_self_imagined_owner_core2_dual4_v1.sbatch",
    "methods/bernini_action_editing/tests/test_materialize_self_imagined_owner_core2_v1.py",
    "methods/bernini_action_editing/tests/test_auh_materialize_self_imagined_owner_core2_launcher.py",
}
seen = set()
with tarfile.open(sys.argv[1], "r:*") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        normalized = path.as_posix().lstrip("./")
        if not normalized.startswith("methods/bernini_action_editing/"):
            continue
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise SystemExit("source archive contains an unsafe selected member")
        seen.add(normalized)
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"source archive lacks owner materializer closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree" methods/bernini_action_editing
source_tree="${task_scratch}/source-tree"
method_root="${source_tree}/methods/bernini_action_editing"
runtime="${method_root}/materialize_self_imagined_owner_core2_v1.py"
launcher="${method_root}/scripts/auh_materialize_self_imagined_owner_core2_dual4_v1.sbatch"
for pinned_file in "${runtime}" "${method_root}/materialize_starc_core4_hidden_v1.py" "${method_root}/self_imagined_motion_cotangent_v1.py" "${method_root}/generate_self_imagined_owner_core2_v1.py" "${method_root}/score_pair_v5_t2v_energy_bank_frozen_d541801.py" "${launcher}"; do
  [[ -f "${pinned_file}" && ! -L "${pinned_file}" ]] || fail "extracted source closure differs: ${pinned_file}"
done
[[ "$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')" == "$(sha256sum "${launcher}" | awk '{print $1}')" ]] || fail "running launcher differs from source archive"
find "${source_tree}" -type f -exec chmod a-w -- {} +
runtime_sha256="$(sha256sum "${runtime}" | awk '{print $1}')"

authority_args=(
  --registry "${registry}"
  --expected-registry-sha256 "${registry_sha256}"
  --owner-root "${owner_root}"
  --owner-master-receipt "${owner_master}"
  --expected-owner-master-receipt-sha256 "${owner_master_sha256}"
  --audit-sidecar "${audit_sidecar}"
  --expected-audit-sidecar-sha256 "${audit_sidecar_sha256}"
  --audit-evidence "${audit_evidence}"
  --audit-public-key "${audit_public_key}"
  --expected-audit-public-key-sha256 "${audit_public_key_sha256}"
)

# Mandatory CPU-only trust gate.  If either full81 owner action audit is not a
# signed PASS, no output staging directory and no torchrun process is created.
PYTHONPATH="${method_root}" "${python_bin}" -B "${runtime}" preflight "${authority_args[@]}"

mkdir -- "${staging_root}"
materialize_args=(
  "${authority_args[@]}"
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --checkpoint-content-manifest "${checkpoint_manifest}"
  --expected-bernini-commit "${required_bernini_commit}"
  --expected-veomni-commit "${required_veomni_commit}"
  --method-source-revision "${source_revision}"
  --method-source-archive-sha256 "${source_archive_sha256}"
  --expected-materializer-source-sha256 "${runtime_sha256}"
)

job_mod=$((SLURM_JOB_ID % 10000))
dog_port=$((34000 + 2 * job_mod))
human_port=$((dog_port + 1))

run_group() (
  set -Eeuo pipefail
  local cell_id="$1" visible_gpus="$2" master_port="$3"
  local group_scratch="${task_scratch}/groups/${cell_id}"
  mkdir -p -- "${group_scratch}/miopen-user" "${group_scratch}/miopen-custom" \
    "${group_scratch}/torch-extensions" "${group_scratch}/triton" \
    "${group_scratch}/xdg" "${group_scratch}/pycache" "${group_scratch}/tmp"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible_gpus}"
  export MIOPEN_USER_DB_PATH="${group_scratch}/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${group_scratch}/miopen-custom"
  export TORCH_EXTENSIONS_DIR="${group_scratch}/torch-extensions"
  export TRITON_CACHE_DIR="${group_scratch}/triton"
  export XDG_CACHE_HOME="${group_scratch}/xdg"
  export PYTHONPYCACHEPREFIX="${group_scratch}/pycache"
  export TMPDIR="${group_scratch}/tmp"
  export TORCHELASTIC_ERROR_FILE="${group_scratch}/torch-elastic-error.json"
  echo "[owner-quotient-v1:${cell_id}] START visible=${visible_gpus} WORLD4/SP4 query_seeds=2"
  PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
    --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="${master_port}" \
    "${runtime}" materialize-cell "${materialize_args[@]}" \
    --cell-id "${cell_id}" --output-dir "${staging_root}/${cell_id}"
)

echo "[owner-quotient-v1] all8 dog=0,1,2,3 human=4,5,6,7 exact81=true block15=true schedule33=true"
run_group dog "0,1,2,3" "${dog_port}" >"${task_scratch}/logs/dog.log" 2>&1 & dog_pid=$!
run_group human "4,5,6,7" "${human_port}" >"${task_scratch}/logs/human.log" 2>&1 & human_pid=$!
dog_status=0
human_status=0
wait "${dog_pid}" || dog_status=$?
wait "${human_pid}" || human_status=$?
if [[ "${dog_status}" != 0 || "${human_status}" != 0 ]]; then
  tail -n 240 -- "${task_scratch}/logs/dog.log" >&2 || true
  tail -n 240 -- "${task_scratch}/logs/human.log" >&2 || true
  fail "WORLD4 owner materialization failed dog=${dog_status} human=${human_status}"
fi

PYTHONPATH="${method_root}" "${python_bin}" -B "${runtime}" aggregate-master \
  "${authority_args[@]}" --output-root "${staging_root}"

[[ "$(find "${staging_root}" -type f -name owner-motion-quotients.safetensors | wc -l)" == 2 ]] || fail "quotient artifact count differs"
[[ "$(find "${staging_root}" -type f -name owner-motion-specificity-receipt.json | wc -l)" == 2 ]] || fail "cell specificity receipt count differs"
[[ -f "${staging_root}/owner-core2-specificity-receipt.json" && ! -L "${staging_root}/owner-core2-specificity-receipt.json" ]] || fail "master specificity receipt missing"
[[ "$(find "${staging_root}" -type f | wc -l)" == 5 ]] || fail "persistent output channel contains extra files"
chmod -R a-w -- "${staging_root}"
mv -T -- "${staging_root}" "${output_root}"
published=true
