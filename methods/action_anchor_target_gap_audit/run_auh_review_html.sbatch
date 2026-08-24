#!/usr/bin/env bash
#SBATCH --job-name=mev-gap16-html
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -Eeuo pipefail
umask 077

repo_root="${ACTION_GAP_REPO_ROOT:?set ACTION_GAP_REPO_ROOT}"
experiment_root="${ACTION_GAP_EXPERIMENT_ROOT:?set ACTION_GAP_EXPERIMENT_ROOT}"
output_dir="${ACTION_GAP_REVIEW_OUTPUT:?set ACTION_GAP_REVIEW_OUTPUT}"
python_bin="${ACTION_GAP_REVIEW_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/qwen/bin/python3.12}"
ffmpeg_bin="${ACTION_GAP_FFMPEG:-/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}"

[[ "${output_dir}" == /* && "${output_dir}" != "/" ]] || {
  echo "unsafe output directory: ${output_dir}" >&2; exit 2;
}
[[ "${output_dir}" != /vast/users/guangyi.chen/dataset/MEV/MEV* ]] || {
  echo "refusing protected MEV output directory: ${output_dir}" >&2; exit 2;
}
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || {
  echo "review output already exists: ${output_dir}" >&2; exit 2;
}

export PYTHONPATH="${repo_root}" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
"${python_bin}" -B -m methods.action_anchor_target_gap_audit.build_review_html \
  --manifest "${experiment_root}/manifest.json" \
  --qwen-summary "${experiment_root}/qwen_summary.json" \
  --semantic-moments-summary "${experiment_root}/semantic_moments_summary.json" \
  --output-dir "${output_dir}" --ffmpeg "${ffmpeg_bin}" \
  --workers 4 --ffmpeg-threads 4

(
  cd "${output_dir}"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
)
echo "[action-gap-review] COMPLETE job=${SLURM_JOB_ID} output=${output_dir}"
