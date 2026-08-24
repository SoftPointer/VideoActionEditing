#!/usr/bin/env bash
#SBATCH --job-name=fact7-smoke
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --account=test-acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=06:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 077

source_archive="${FACTORIAL_BRANCH_SOURCE_ARCHIVE:?set source archive}"
source_archive_sha256="${FACTORIAL_BRANCH_SOURCE_ARCHIVE_SHA256:?set archive SHA-256}"
source_revision="${FACTORIAL_BRANCH_SOURCE_REVISION:?set source revision}"
manifest_sha256="${FACTORIAL_BRANCH_MANIFEST_SHA256:?set manifest SHA-256}"
released_cells="${FACTORIAL_BRANCH_RELEASED_CELLS:?set sorted comma-separated source:seed cells}"
output_root="${FACTORIAL_BRANCH_OUTPUT_ROOT:?set fresh output root}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set checkpoint}"
python_bin="${FACTORIAL_BRANCH_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12}"

fail() { echo "[factorial-branch-smoke] ERROR: $*" >&2; exit 2; }

for name in source_archive output_root bernini_root veomni_root checkpoint python_bin; do
  [[ "${!name}" == /* && "${!name}" != / ]] || fail "${name} must be an absolute non-root path"
done
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "archive SHA differs"
[[ "${manifest_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "manifest SHA differs"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive differs"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"
[[ -d "${bernini_root}" && -d "${veomni_root}" && -d "${checkpoint}" ]] || fail "model roots differ"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "archive revision differs"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"
mkdir -p "${output_root%/*}"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/factorial-branch-${SLURM_JOB_ID}.XXXXXX")"
lane0_pid=""
lane1_pid=""

cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  for pid in "${lane0_pid}" "${lane1_pid}"; do
    [[ "${pid}" =~ ^[0-9]+$ ]] && kill "${pid}" 2>/dev/null || true
  done
  case "${task_scratch}" in
    "${scratch_parent%/}/factorial-branch-${SLURM_JOB_ID}."*) chmod -R u+w "${task_scratch}" 2>/dev/null || true; rm -rf "${task_scratch}" ;;
    *) echo "[factorial-branch-smoke] refusing unsafe scratch cleanup" >&2; rc=2 ;;
  esac
  exit "${rc}"
}
trap cleanup EXIT INT TERM HUP

mkdir -p "${task_scratch}/source" "${task_scratch}/lane-0" "${task_scratch}/lane-1"
tar --no-same-owner --no-same-permissions -xf "${source_archive}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
runner="${method_root}/run_prospective_factorial_branch_shard_v1.py"
manifest="${method_root}/assets/prospective_factorial_branch_manifest_v1.json"
[[ -f "${runner}" && ! -L "${runner}" && -f "${manifest}" && ! -L "${manifest}" ]] || fail "staged runtime closure differs"
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha256}" ]] || fail "staged manifest hash differs"

mkdir "${output_root}" "${output_root}/entries" "${output_root}/logs"
cp "${manifest}" "${output_root}/manifest.json"
chmod 0444 "${output_root}/manifest.json"

IFS=',' read -r -a cell_values <<<"${released_cells}"
[[ "${#cell_values[@]}" -eq 2 ]] || fail "smoke must release exactly two cells"
cell_args=()
for cell in "${cell_values[@]}"; do cell_args+=(--cell "${cell}"); done

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

job_mod=$((SLURM_JOB_ID % 10000))
port0=$((30000 + 2 * job_mod))
port1=$((port0 + 1))

run_lane() (
  set -Eeuo pipefail
  local lane="$1" visible="$2" port="$3" lane_root="${task_scratch}/lane-${1}"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible}"
  export MIOPEN_USER_DB_PATH="${lane_root}/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${lane_root}/miopen-custom"
  export TORCH_EXTENSIONS_DIR="${lane_root}/torch-extensions"
  export TRITON_CACHE_DIR="${lane_root}/triton"
  export XDG_CACHE_HOME="${lane_root}/xdg"
  export PYTHONPYCACHEPREFIX="${lane_root}/pycache"
  export TMPDIR="${lane_root}"
  mkdir -p "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${PYTHONPYCACHEPREFIX}"
  "${python_bin}" -B "${runner}" \
    --manifest "${manifest}" --expected-manifest-sha256 "${manifest_sha256}" \
    --split fit "${cell_args[@]}" run-lane \
    --lane-index "${lane}" --lane-count 2 --master-port "${port}" \
    --output-root "${output_root}" --method-root "${method_root}" \
    --python-bin "${python_bin}" --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" --checkpoint "${checkpoint}" \
    --method-source-revision "${source_revision}" \
    --method-source-archive-sha256 "${source_archive_sha256}"
)

run_lane 0 "0,1,2,3" "${port0}" >"${output_root}/logs/lane-0.log" 2>&1 & lane0_pid=$!
run_lane 1 "4,5,6,7" "${port1}" >"${output_root}/logs/lane-1.log" 2>&1 & lane1_pid=$!
lane0_rc=0
lane1_rc=0
wait "${lane0_pid}" || lane0_rc=$?
wait "${lane1_pid}" || lane1_rc=$?
lane0_pid=""
lane1_pid=""
if [[ "${lane0_rc}" -ne 0 || "${lane1_rc}" -ne 0 ]]; then
  tail -n 160 "${output_root}/logs/lane-0.log" >&2 || true
  tail -n 160 "${output_root}/logs/lane-1.log" >&2 || true
  fail "one or more SP4 lanes failed"
fi

"${python_bin}" -B "${runner}" \
  --manifest "${manifest}" --expected-manifest-sha256 "${manifest_sha256}" \
  --split fit "${cell_args[@]}" verify-release \
  --output-root "${output_root}" --output-receipt "${output_root}/release.receipt.json" \
  >"${output_root}/release-verification.log"
echo "FACTORIAL_BRANCH_SMOKE_COMPLETE_REVIEW_PENDING_NOT_TRAINING_TARGET"
