#!/usr/bin/env bash
#SBATCH --job-name=motive-review-pack
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# AUH faculty QOS enforces MinTRES=gres/gpu:1 even for this metadata-only job.
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=01:00:00

set -Eeuo pipefail

repo_root="${MOTIVE_REPO_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit}"
output_root="${MOTIVE_OUTPUT_ROOT:-${repo_root}/methods/motive/outputs/goku_subject_movement_full}"
code_root="${MOTIVE_CODE_ROOT:-${repo_root}/methods/motive}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
input_manifest="${MOTIVE_HUMAN_REVIEW_INPUT:-${output_root}/fused/all.jsonl}"
template_output="${MOTIVE_HUMAN_REVIEW_TEMPLATE:-${output_root}/fused/review_labels.template.jsonl}"
working_output="${MOTIVE_HUMAN_REVIEW_OUTPUT:-${output_root}/fused/review_labels.working.jsonl}"
summary_output="${template_output}.summary.json"

for required in \
  "${code_root}/motive/human_review.py" \
  "${input_manifest}" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[motive-review-pack] missing required path: ${required}" >&2
    exit 2
  fi
done
if [[ -e "${template_output}" || -e "${summary_output}" || -e "${working_output}" ]]; then
  echo "[motive-review-pack] output already exists; refusing to overwrite" >&2
  echo "  template=${template_output}" >&2
  echo "  summary=${summary_output}" >&2
  echo "  working=${working_output}" >&2
  exit 2
fi

cd "${repo_root}"
export PYTHONPATH="${code_root}:${PYTHONPATH:-}"
"${python_bin}" -m motive.human_review prepare \
  --input "${input_manifest}" \
  --output "${template_output}"
cp "${template_output}" "${working_output}"
chmod a-w "${template_output}" "${summary_output}"

if [[ ! -s "${template_output}" || ! -s "${summary_output}" || ! -s "${working_output}" ]]; then
  echo "[motive-review-pack] expected outputs were not committed" >&2
  exit 2
fi
echo "[motive-review-pack] template=${template_output}"
echo "[motive-review-pack] summary=${summary_output}"
echo "[motive-review-pack] working=${working_output}"
