#!/usr/bin/env bash
#SBATCH --job-name=motive-rules-full
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# AUH faculty QOS enforces MinTRES=gres/gpu:1 even for metadata jobs.
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=04:00:00

set -Eeuo pipefail

repo_root="${MOTIVE_REPO_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit}"
data_root="${MOTIVE_DATA_ROOT:-${repo_root}/data/goku/subject_movement/extracted}"
output_root="${MOTIVE_OUTPUT_ROOT:-${repo_root}/methods/motive/outputs/goku_subject_movement_full}"
code_root="${MOTIVE_CODE_ROOT:-${repo_root}/methods/motive}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
rule_output="${output_root}/rules"

for required in \
  "${code_root}/motive/cascade.py" \
  "${data_root}/jsons/combine_json" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[motive-rules] missing required path: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${rule_output}"
cd "${repo_root}"

arguments=(
  -m motive.cascade rules
  --dataset-root "${data_root}"
  --output-dir "${rule_output}"
  --seed "${MOTIVE_SEED:-260108828}"
  --min-score "${MOTIVE_RULE_MIN_SCORE:-0.40}"
  --include-tiers high possible
  --reject-audit-fraction "${MOTIVE_REJECT_AUDIT_FRACTION:-0.01}"
  --continue-on-error
)
if [[ "${MOTIVE_RULE_OVERWRITE:-0}" == "1" ]]; then
  arguments+=(--overwrite)
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="${code_root}:${PYTHONPATH:-}"

echo "[motive-rules] dataset=${data_root}"
echo "[motive-rules] output=${rule_output}"
exec "${python_bin}" "${arguments[@]}"
