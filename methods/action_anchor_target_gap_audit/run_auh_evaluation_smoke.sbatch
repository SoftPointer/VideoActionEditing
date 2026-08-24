#!/usr/bin/env bash
#SBATCH --job-name=mev-a2b-evalsmk
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=02:00:00

set -Eeuo pipefail
umask 077

repo_root="${ACTION_GAP_REPO_ROOT:?set ACTION_GAP_REPO_ROOT}"
manifest="${ACTION_GAP_MANIFEST:?set ACTION_GAP_MANIFEST}"
experiment_root="${ACTION_GAP_EXPERIMENT_ROOT:?set ACTION_GAP_EXPERIMENT_ROOT}"
python_bin="${ACTION_GAP_QWEN_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/qwen/bin/python3.12}"
qwen_model="${ACTION_GAP_QWEN_MODEL:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VLM/MEV-Annotation/checkpoints/Qwen3-VL-32B-Instruct}"
semantic_root="${ACTION_GAP_SEMANTIC_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/semantic_moments_reward_20260815/vendor/semantic-moments}"
dinov2_root="${ACTION_GAP_DINOV2_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/vendor/dinov2-base-f9e44c8}"
prefix="${ACTION_GAP_SMOKE_PAIR_PREFIX:-7fa338474c75}"
output_root="${ACTION_GAP_EVAL_SMOKE_OUTPUT_ROOT:-${experiment_root}/evaluation_smoke_${prefix}}"
module_path="methods.action_anchor_target_gap_audit.audit"

[[ "${output_root}" == /* && "${output_root}" != "/" ]] || {
  echo "unsafe smoke output root: ${output_root}" >&2; exit 2;
}
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || {
  echo "smoke output already exists: ${output_root}" >&2; exit 2;
}
mkdir -p "${output_root}/qwen" "${output_root}/semantic_moments/features"
runtime_root="/tmp/action-gap-eval-smoke-${SLURM_JOB_ID}"
for worker in qwen sm; do
  mkdir -p "${runtime_root}/${worker}/home" "${runtime_root}/${worker}/tmp" \
    "${runtime_root}/${worker}/xdg" "${runtime_root}/${worker}/torch" \
    "${runtime_root}/${worker}/torch-kernels" "${runtime_root}/${worker}/miopen-user" \
    "${runtime_root}/${worker}/miopen-custom"
done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=offline WANDB_DISABLED=true PYTHONHASHSEED=0 PYTHONPATH="${repo_root}"

env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0,1,2,3 \
  HOME="${runtime_root}/qwen/home" TMPDIR="${runtime_root}/qwen/tmp" \
  XDG_CACHE_HOME="${runtime_root}/qwen/xdg" TORCH_HOME="${runtime_root}/qwen/torch" \
  PYTORCH_KERNEL_CACHE_PATH="${runtime_root}/qwen/torch-kernels" \
  MIOPEN_USER_DB_PATH="${runtime_root}/qwen/miopen-user" \
  MIOPEN_CUSTOM_CACHE_DIR="${runtime_root}/qwen/miopen-custom" \
  "${python_bin}" -B -m "${module_path}" qwen-evaluate \
    --manifest "${manifest}" --model "${qwen_model}" --pair-prefix "${prefix}" \
    --shard-index 0 --num-shards 1 \
    --output "${output_root}/qwen/qwen-shard-0.jsonl"

env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0 \
  HOME="${runtime_root}/sm/home" TMPDIR="${runtime_root}/sm/tmp" \
  XDG_CACHE_HOME="${runtime_root}/sm/xdg" TORCH_HOME="${runtime_root}/sm/torch" \
  PYTORCH_KERNEL_CACHE_PATH="${runtime_root}/sm/torch-kernels" \
  MIOPEN_USER_DB_PATH="${runtime_root}/sm/miopen-user" \
  MIOPEN_CUSTOM_CACHE_DIR="${runtime_root}/sm/miopen-custom" \
  "${python_bin}" -B -m "${module_path}" semantic-moments-extract \
    --manifest "${manifest}" --semantic-moments-root "${semantic_root}" \
    --model-root "${dinov2_root}" --pair-prefix "${prefix}" \
    --shard-index 0 --num-shards 1 \
    --output "${output_root}/semantic_moments/features/features-shard-0.pt"

"${python_bin}" -B -m "${module_path}" qwen-summarize \
  --records-dir "${output_root}/qwen" --output "${output_root}/qwen_summary.json"
"${python_bin}" -B -m "${module_path}" semantic-moments-summarize \
  --features-dir "${output_root}/semantic_moments/features" \
  --output "${output_root}/semantic_moments_summary.json"
"${python_bin}" -B -m "${module_path}" final-report \
  --qwen-summary "${output_root}/qwen_summary.json" \
  --semantic-moments-summary "${output_root}/semantic_moments_summary.json" \
  --output "${output_root}/final_report.json"
echo "[action-gap-eval-smoke] COMPLETE report=${output_root}/final_report.json"
