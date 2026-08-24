#!/usr/bin/env bash
#SBATCH --job-name=pair5-sb-salvage
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=00:10:00

set -Eeuo pipefail
umask 077

fail() { echo "[pair-v5-source-bound-salvage] ERROR: $*" >&2; exit 2; }
absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }

experiment_root="${PAIR_V5_SOURCE_BOUND_EXPERIMENT_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808}"
source_archive="${PAIR_V5_SOURCE_BOUND_SALVAGE_SOURCE_ARCHIVE:?set immutable salvage source git archive}"
source_archive_sha256="${PAIR_V5_SOURCE_BOUND_SALVAGE_SOURCE_ARCHIVE_SHA256:?set salvage source archive SHA-256}"
source_revision="${PAIR_V5_SOURCE_BOUND_SALVAGE_SOURCE_REVISION:?set salvage source revision}"
rollout_spec="${PAIR_V5_SOURCE_BOUND_ROLLOUT_SPEC:-${experiment_root}/inputs/pair_v5_native_rv2v4_core4_action_population_v1.json}"
evaluator_spec="${PAIR_V5_SOURCE_BOUND_EVALUATOR_SPEC:-${experiment_root}/inputs/pair_v5_source_bound_preservation_evaluator_7c4c837_v1.json}"
legacy_archive="${PAIR_V5_SOURCE_BOUND_LEGACY_ARCHIVE:-${experiment_root}/inputs/pair_v5_source_bound_preservation_7c4c837_minimal.tar}"
legacy_run="${PAIR_V5_SOURCE_BOUND_LEGACY_RUN:-${experiment_root}/runs/core4_source_bound_v1_7c4c837}"
output="${PAIR_V5_SOURCE_BOUND_SALVAGE_OUTPUT:?set fresh absolute salvage receipt path}"
python_bin="${PAIR_V5_SOURCE_BOUND_SALVAGE_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12}"

for name in experiment_root source_archive rollout_spec evaluator_spec legacy_archive legacy_run output python_bin; do
  absolute "${!name}" "${name}"
done
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source archive SHA-256 differs"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"

source_archive="$(realpath -e -- "${source_archive}")"
rollout_spec="$(realpath -e -- "${rollout_spec}")"
evaluator_spec="$(realpath -e -- "${evaluator_spec}")"
legacy_archive="$(realpath -e -- "${legacy_archive}")"
legacy_run="$(realpath -e -- "${legacy_run}")"
python_bin_target="$(realpath -e -- "${python_bin}")"
for name in source_archive rollout_spec evaluator_spec legacy_archive; do
  [[ -f "${!name}" && ! -L "${!name}" ]] || fail "${name} is not a plain file"
done
[[ -d "${legacy_run}" && ! -L "${legacy_run}" ]] || fail "legacy run is not a plain directory"
[[ -f "${python_bin_target}" && -x "${python_bin}" ]] || fail "Python executable differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"

[[ "${output}" != / && "${output##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || fail "unsafe output path"
[[ "$(realpath -m -- "${output}")" == "${output}" ]] || fail "output path must be canonical"
[[ ! -e "${output}" && ! -L "${output}" ]] || fail "output receipt must be fresh"
case "${output}/" in "${legacy_run}/"*) fail "output cannot enter the immutable legacy run" ;; esac
output_parent="$(dirname -- "${output}")"
mkdir -p -- "${output_parent}"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/pair-v5-sb-salvage-${SLURM_JOB_ID}.XXXXXX")"
cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in "${scratch_parent%/}/pair-v5-sb-salvage-${SLURM_JOB_ID}."*) ;; *) cleanup_failed=1 ;; esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[pair-v5-source-bound-salvage] PASS receipt_only=true candidates=8 groups=2 roots=1 dino_recomputed=false absolute_preservation_pass=false optimizer_go=false output=${output}"
  else
    echo "[pair-v5-source-bound-salvage] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -- "${task_scratch}/source"
cp -- "${source_archive}" "${task_scratch}/source.tar"
[[ "$(sha256sum "${task_scratch}/source.tar" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged source archive differs"
member_hashes="$("${python_bin}" -B - "${task_scratch}/source.tar" "${source_revision}" <<'PY'
from pathlib import PurePosixPath
import hashlib,sys,tarfile
archive,revision=sys.argv[1:]
required={
 "methods/bernini_action_editing/salvage_pair_v5_source_bound_failed_launch_v1.py",
 "methods/bernini_action_editing/pair_v5_source_bound_preservation_evaluator_v1.py",
}
seen=set(); payload={}
with tarfile.open(archive,"r:*") as handle:
 if handle.pax_headers.get("comment") != revision:
  raise SystemExit("salvage archive revision differs")
 for member in handle.getmembers():
  path=PurePosixPath(member.name)
  normalized=path.as_posix().rstrip("/")
  if (not member.name or path.is_absolute() or ".." in path.parts or
      member.issym() or member.islnk() or member.isdev() or member.isfifo()):
   raise SystemExit("unsafe salvage archive member")
  if normalized in seen:
   raise SystemExit("duplicate salvage archive member")
  seen.add(normalized)
  if normalized in required:
   if not member.isfile(): raise SystemExit("required salvage member is not plain")
   stream=handle.extractfile(member)
   if stream is None: raise SystemExit("required salvage member is unreadable")
   payload[normalized]=stream.read()
if set(payload) != required:
 raise SystemExit("salvage archive source closure differs")
tool="methods/bernini_action_editing/salvage_pair_v5_source_bound_failed_launch_v1.py"
validator="methods/bernini_action_editing/pair_v5_source_bound_preservation_evaluator_v1.py"
tool_sha=hashlib.sha256(payload[tool]).hexdigest()
validator_sha=hashlib.sha256(payload[validator]).hexdigest()
if validator_sha != "c390d17b680b3d57048d04257fd5f89f98a7b421b79720f68eb837d992f392d4":
 raise SystemExit("salvage archive validator is not the pinned map-order patch")
print(tool_sha,validator_sha)
PY
)"
read -r tool_member_sha256 validator_member_sha256 <<<"${member_hashes}"
[[ "${tool_member_sha256}" =~ ^[0-9a-f]{64}$ && "${validator_member_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "salvage member hash preflight differs"
tar --no-same-owner --no-same-permissions -xf "${task_scratch}/source.tar" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
tool="${method_root}/salvage_pair_v5_source_bound_failed_launch_v1.py"
validator="${method_root}/pair_v5_source_bound_preservation_evaluator_v1.py"
[[ -f "${tool}" && -f "${validator}" && ! -L "${tool}" && ! -L "${validator}" ]] || fail "salvage source closure differs"
[[ "$(sha256sum "${tool}" | awk '{print $1}')" == "${tool_member_sha256}" ]] || fail "executed tool/archive member differs"
[[ "$(sha256sum "${validator}" | awk '{print $1}')" == "${validator_member_sha256}" ]] || fail "imported validator/archive member differs"
find "${task_scratch}/source" -type f -exec chmod a-w -- {} +

# gtqos currently requires a minimal GPU GRES request.  This receipt-only
# process deliberately cannot see or initialize the allocated accelerator.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[pair-v5-source-bound-salvage] allocation=1xMI210-hidden compute=cpu-json-only legacy_job=131222"

PYTHONPATH="${method_root}" "${python_bin}" -B "${tool}" \
  --rollout-spec "${rollout_spec}" \
  --evaluator-spec "${evaluator_spec}" \
  --legacy-method-archive "${legacy_archive}" \
  --legacy-run-dir "${legacy_run}" \
  --salvage-source-archive "${task_scratch}/source.tar" \
  --salvage-source-revision "${source_revision}" \
  --salvage-source-archive-sha256 "${source_archive_sha256}" \
  --patched-validator-source "${validator}" \
  --output "${output}"

PYTHONPATH="${method_root}" "${python_bin}" -B "${tool}" \
  --rollout-spec "${rollout_spec}" \
  --evaluator-spec "${evaluator_spec}" \
  --legacy-method-archive "${legacy_archive}" \
  --legacy-run-dir "${legacy_run}" \
  --salvage-source-archive "${task_scratch}/source.tar" \
  --salvage-source-revision "${source_revision}" \
  --salvage-source-archive-sha256 "${source_archive_sha256}" \
  --patched-validator-source "${validator}" \
  --output "${output}" \
  --verify-existing-output
