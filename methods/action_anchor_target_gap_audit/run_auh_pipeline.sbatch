#!/usr/bin/env bash
#SBATCH --job-name=mev-a2b-gap16
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=24:00:00

set -Eeuo pipefail
umask 077

repo_root="${ACTION_GAP_REPO_ROOT:?set ACTION_GAP_REPO_ROOT}"
manifest="${ACTION_GAP_MANIFEST:?set ACTION_GAP_MANIFEST}"
experiment_root="${ACTION_GAP_EXPERIMENT_ROOT:?set ACTION_GAP_EXPERIMENT_ROOT}"
python_gen="${ACTION_GAP_GENERATION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12}"
python_qwen="${ACTION_GAP_QWEN_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/qwen/bin/python3.12}"
bernini_root="${ACTION_GAP_BERNINI_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/Bernini-2d2b4591}"
veomni_root="${ACTION_GAP_VEOMNI_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/VeOmni-f90b3dc6}"
checkpoint="${ACTION_GAP_CHECKPOINT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4}"
checkpoint_manifest="${ACTION_GAP_CHECKPOINT_MANIFEST:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_counterfactual_identity_orbit_v5_20260808_c099c6f/runtime/source_ea900d5/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256}"
inference_root="${ACTION_GAP_INFERENCE_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_preservation_only_20260814/staging/1871b029a06d4d1b}"
qwen_model="${ACTION_GAP_QWEN_MODEL:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VLM/MEV-Annotation/checkpoints/Qwen3-VL-32B-Instruct}"
semantic_root="${ACTION_GAP_SEMANTIC_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/semantic_moments_reward_20260815/vendor/semantic-moments}"
dinov2_root="${ACTION_GAP_DINOV2_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/vendor/dinov2-base-f9e44c8}"
ffmpeg_bin="${ACTION_GAP_FFMPEG:-/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}"
module_path="methods.action_anchor_target_gap_audit.audit"
controller_module="methods.action_anchor_target_gap_audit.generation_controller"

fail() { echo "[action-gap] ERROR: $*" >&2; exit 2; }
for file in "${manifest}" "${checkpoint_manifest}"; do
  [[ -f "${file}" && ! -L "${file}" ]] || fail "missing plain input: ${file}"
done
for directory in "${repo_root}" "${inference_root}" "${bernini_root}" "${veomni_root}" "${checkpoint}" "${qwen_model}" "${semantic_root}" "${dinov2_root}"; do
  [[ -d "${directory}" && ! -L "${directory}" ]] || fail "missing plain directory: ${directory}"
done
for executable in "${python_gen}" "${python_qwen}"; do
  [[ -x "${executable}" ]] || fail "missing executable: ${executable}"
done
for executable in "${ffmpeg_bin}"; do
  [[ -x "${executable}" ]] || fail "missing media executable: ${executable}"
done
[[ "${experiment_root}" == /* && "${experiment_root}" != "/" ]] || fail "unsafe experiment root"
mkdir -p "${experiment_root}/logs" "${experiment_root}/qwen" "${experiment_root}/semantic_moments/features"
runtime_root="/tmp/action-gap-${SLURM_JOB_ID}"
for worker in gen0 gen1 qwen0 qwen1 sm0 sm1 sm2 sm3 sm4 sm5 sm6 sm7; do
  mkdir -p "${runtime_root}/${worker}/home" "${runtime_root}/${worker}/tmp" \
    "${runtime_root}/${worker}/xdg" "${runtime_root}/${worker}/torch" \
    "${runtime_root}/${worker}/torch-kernels" "${runtime_root}/${worker}/miopen-user" \
    "${runtime_root}/${worker}/miopen-custom"
done

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=offline WANDB_DISABLED=true PYTHONHASHSEED=0
export PYTHONPATH="${repo_root}"

run_generation_worker() {
  local index="$1" devices="$2" port="$3"
  local cache="${runtime_root}/gen${index}"
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
    ROCR_VISIBLE_DEVICES="${devices}" \
    HOME="${cache}/home" TMPDIR="${cache}/tmp" XDG_CACHE_HOME="${cache}/xdg" \
    TORCH_HOME="${cache}/torch" PYTORCH_KERNEL_CACHE_PATH="${cache}/torch-kernels" \
    MIOPEN_USER_DB_PATH="${cache}/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="${cache}/miopen-custom" \
    "${python_gen}" -B -m "${controller_module}" \
      --manifest "${manifest}" --repo-root "${repo_root}" --inference-root "${inference_root}" --python "${python_gen}" \
      --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
      --ffmpeg "${ffmpeg_bin}" \
      --worker-index "${index}" --num-workers 2 --master-port-base "${port}"
}

run_generation_worker 0 0,1,2,3 29810 >"${experiment_root}/logs/generation-worker-0.log" 2>&1 &
pid_a=$!
run_generation_worker 1 4,5,6,7 29910 >"${experiment_root}/logs/generation-worker-1.log" 2>&1 &
pid_b=$!
status=0
wait "${pid_a}" || status=$?
wait "${pid_b}" || status=$?
(( status == 0 )) || fail "generation failed; see generation-worker logs"

run_qwen_worker() {
  local index="$1" devices="$2"
  local cache="${runtime_root}/qwen${index}"
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
    ROCR_VISIBLE_DEVICES="${devices}" \
    HOME="${cache}/home" TMPDIR="${cache}/tmp" XDG_CACHE_HOME="${cache}/xdg" \
    TORCH_HOME="${cache}/torch" PYTORCH_KERNEL_CACHE_PATH="${cache}/torch-kernels" \
    MIOPEN_USER_DB_PATH="${cache}/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="${cache}/miopen-custom" \
    "${python_qwen}" -B -m "${module_path}" qwen-evaluate \
      --manifest "${manifest}" --model "${qwen_model}" \
      --shard-index "${index}" --num-shards 2 \
      --output "${experiment_root}/qwen/qwen-shard-${index}.jsonl"
}

run_qwen_worker 0 0,1,2,3 >"${experiment_root}/logs/qwen-worker-0.log" 2>&1 &
pid_a=$!
run_qwen_worker 1 4,5,6,7 >"${experiment_root}/logs/qwen-worker-1.log" 2>&1 &
pid_b=$!
status=0
wait "${pid_a}" || status=$?
wait "${pid_b}" || status=$?
(( status == 0 )) || fail "Qwen evaluation failed; see qwen-worker logs"

pids=()
for index in 0 1 2 3 4 5 6 7; do
  cache="${runtime_root}/sm${index}"
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
    ROCR_VISIBLE_DEVICES="${index}" \
    HOME="${cache}/home" TMPDIR="${cache}/tmp" XDG_CACHE_HOME="${cache}/xdg" \
    TORCH_HOME="${cache}/torch" PYTORCH_KERNEL_CACHE_PATH="${cache}/torch-kernels" \
    MIOPEN_USER_DB_PATH="${cache}/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="${cache}/miopen-custom" \
    "${python_qwen}" -B -m "${module_path}" semantic-moments-extract \
      --manifest "${manifest}" --semantic-moments-root "${semantic_root}" \
      --model-root "${dinov2_root}" --shard-index "${index}" --num-shards 8 \
      --output "${experiment_root}/semantic_moments/features/features-shard-${index}.pt" \
      >"${experiment_root}/logs/semantic-moments-${index}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
(( status == 0 )) || fail "SemanticMoments extraction failed"

"${python_qwen}" -B -m "${module_path}" qwen-summarize \
  --records-dir "${experiment_root}/qwen" --output "${experiment_root}/qwen_summary.json"
"${python_qwen}" -B -m "${module_path}" semantic-moments-summarize \
  --features-dir "${experiment_root}/semantic_moments/features" \
  --output "${experiment_root}/semantic_moments_summary.json"
"${python_qwen}" -B -m "${module_path}" final-report \
  --qwen-summary "${experiment_root}/qwen_summary.json" \
  --semantic-moments-summary "${experiment_root}/semantic_moments_summary.json" \
  --output "${experiment_root}/final_report.json"
sha256sum "${manifest}" "${experiment_root}/qwen_summary.json" \
  "${experiment_root}/semantic_moments_summary.json" "${experiment_root}/final_report.json" \
  >"${experiment_root}/SHA256SUMS"
echo "[action-gap] COMPLETE job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} report=${experiment_root}/final_report.json"
