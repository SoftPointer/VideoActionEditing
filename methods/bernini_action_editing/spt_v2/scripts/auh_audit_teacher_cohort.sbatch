#!/usr/bin/env bash
#SBATCH --job-name=bern-spt-trust64
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=08:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:?set BERNINI_ACTION_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:?set BERNINI_ACTION_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_ACTION_SOURCE_REVISION:?set BERNINI_ACTION_SOURCE_REVISION}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
data_dir="${BERNINI_ACTION_PARQUET_DIR:?set BERNINI_ACTION_PARQUET_DIR}"
dataset_summary="${BERNINI_ACTION_DATASET_SUMMARY:?set BERNINI_ACTION_DATASET_SUMMARY}"
output_dir="${BERNINI_SPT_TRUST_COHORT_OUTPUT:?set BERNINI_SPT_TRUST_COHORT_OUTPUT}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

prefix_rows="${BERNINI_SPT_TRUST_PREFIX_ROWS:-64}"
range_start="${BERNINI_SPT_TRUST_RANGE_START:-}"
range_stop="${BERNINI_SPT_TRUST_RANGE_STOP:-}"
minimum_selected="${BERNINI_SPT_TRUST_MINIMUM_SELECTED:-8}"
max_prebudget_generate="${BERNINI_SPT_TRUST_MAX_PREBUDGET_GENERATE:-0.25}"
min_transport="${BERNINI_SPT_TRUST_MIN_TRANSPORT:-0.03}"
min_proxy_improvement="${BERNINI_SPT_TRUST_MIN_PROXY_IMPROVEMENT:-0.40}"
max_postbudget_per_phase_generate="${BERNINI_SPT_TRUST_MAX_POSTBUDGET_PER_PHASE_GENERATE:-0.12}"
allow_incomplete="${BERNINI_ACTION_ALLOW_INCOMPLETE_DATASET:-0}"

fail() { echo "[bernini-spt-trust] ERROR: $*" >&2; exit 2; }
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid source archive"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid source archive SHA"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash mismatch"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid checkpoint"
[[ -d "${data_dir}" && ! -L "${data_dir}" ]] || fail "invalid parquet directory"
[[ -f "${dataset_summary}" && ! -L "${dataset_summary}" ]] || fail "invalid dataset summary"
[[ "${output_dir}" == /* && ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "output must be a new absolute path"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "${prefix_rows}" =~ ^[1-9][0-9]*$ ]] || fail "invalid prefix row count"
[[ "${minimum_selected}" =~ ^[1-9][0-9]*$ ]] || fail "invalid minimum selected count"
[[ "${allow_incomplete}" =~ ^[01]$ ]] || fail "invalid incomplete flag"
if [[ -n "${range_start}" || -n "${range_stop}" ]]; then
  [[ "${range_start}" =~ ^[0-9]+$ && "${range_stop}" =~ ^[1-9][0-9]*$ ]] || fail "range requires integer START and STOP"
fi

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1

task_scratch="${SLURM_TMPDIR:-/tmp}/bernini-spt-trust-${SLURM_JOB_ID}"
mkdir -p "${task_scratch}/source" "${task_scratch}/miopen-user" "${task_scratch}/miopen-custom" "${task_scratch}/torch-extensions" "${task_scratch}/triton"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
tar -xf "${source_archive}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
[[ -f "${method_root}/spt_v2/audit_teacher_cohort.py" ]] || fail "archive lacks trust cohort audit"
[[ -f "${method_root}/tests/test_spt_v2_teacher_cohort_audit.py" ]] || fail "archive lacks trust cohort tests"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

for pattern in test_spt_v2_contract.py test_spt_v2_teacher_cohort_audit.py; do
  PYTHONPATH="${method_root}:${method_root}/spt_v2" "${python_bin}" -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

scan_args=(--prefix-rows "${prefix_rows}")
if [[ -n "${range_start}" ]]; then
  scan_args=(--row-range "${range_start}" "${range_stop}")
fi

command=(
  "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=4
  "${method_root}/spt_v2/audit_teacher_cohort.py"
  --checkpoint "${checkpoint}"
  --preprocessed-parquet-dir "${data_dir}"
  --dataset-summary "${dataset_summary}"
  --output-dir "${output_dir}"
  "${scan_args[@]}"
  --max-prebudget-generate-fraction "${max_prebudget_generate}"
  --min-postbudget-transport-fraction "${min_transport}"
  --min-proxy-relative-improvement-over-copy "${min_proxy_improvement}"
  --max-postbudget-generate-fraction-per-phase "${max_postbudget_per_phase_generate}"
  --minimum-selected "${minimum_selected}"
  --method-source-revision "${source_revision}"
  --method-source-archive-sha256 "${source_archive_sha256}"
)
[[ "${allow_incomplete}" == 1 ]] && command+=(--allow-incomplete-dataset)
"${command[@]}"

[[ -f "${output_dir}/audit.json" && -f "${output_dir}/selected_membership.json" ]] || fail "audit output is incomplete"
"${python_bin}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["sufficient"] and p["selected_count"] >= int(sys.argv[2]); print(json.dumps({"selected":p["selected_count"],"rows":p["ordered_selected_row_indices"],"membership_digest":p["membership_digest"]},sort_keys=True))' "${output_dir}/selected_membership.json" "${minimum_selected}"

echo "[bernini-spt-trust] PASS output=${output_dir}"
