#!/usr/bin/env bash
# Resume the scale-512 prefilter/Qwen audit after the borrowed idle allocation
# ends. The controller itself validates and resumes committed Qwen shard rows.
#SBATCH --job-name=qwen512-resume
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=24:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318
#SBATCH --output=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/scale512_qwen3_v1_20260731T210828Z/resume-%j.out
#SBATCH --error=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/scale512_qwen3_v1_20260731T210828Z/resume-%j.err

set -Eeuo pipefail
umask 077

run_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/scale512_qwen3_v1_20260731T210828Z
controller="${run_root}/controller_resume.sh"
source_snapshot=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/software/goku-action-wan22-finalsel-v1-source-20260731T160500Z
input_fused=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_r7/goku_r7_20260727T143500Z/expansion/fused_v2/fused.jsonl
video_root=/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted
qwen_model=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VLM/MEV-Annotation/checkpoints/Qwen3-VL-32B-Instruct
python_bin=/vast/users/guangyi.chen/anaconda3/envs/qwen/bin/python3.12

fail() {
  echo "[qwen512-resume] $*" >&2
  exit 2
}

[[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] || fail "missing Slurm job ID"
[[ "${SLURM_NTASKS:-}" == 1 ]] || fail "allocation must expose one batch task"
node="${SLURMD_NODENAME:-}"
[[ "${node}" =~ ^auh[0-9A-Za-z-]+$ ]] || fail "invalid Slurm node name"
for path in \
  "${controller}" \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${input_fused}" \
  "${qwen_model}/config.json" \
  "${python_bin}"; do
  [[ ! -L "${path}" && -f "${path}" ]] || fail "missing regular file: ${path}"
done
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ ! -L "${video_root}" && -d "${video_root}" ]] || fail "video root differs"

export MOTIVE_EXISTING_SLURM_JOB_ID="${SLURM_JOB_ID}"
export MOTIVE_EXISTING_SLURM_NODE="${node}"
export MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT="${source_snapshot}"
export MOTIVE_GOKU_ACTION_INPUT_FUSED="${input_fused}"
export MOTIVE_GOKU_ACTION_VIDEO_ROOT="${video_root}"
export MOTIVE_GOKU_ACTION_SCALE512_ROOT="${run_root}"
export MOTIVE_GOKU_ACTION_QWEN_MODEL="${qwen_model}"
export MOTIVE_GOKU_ACTION_PYTHON_BIN="${python_bin}"

exec bash "${controller}"
