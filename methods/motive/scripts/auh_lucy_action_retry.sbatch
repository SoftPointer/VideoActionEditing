#!/usr/bin/env bash
#SBATCH --job-name=motive-action-retry
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=12:00:00
#SBATCH --array=0-5%2

set -Eeuo pipefail

phase="${MOTIVE_RETRY_PHASE:?MOTIVE_RETRY_PHASE is required}"
run_root="${MOTIVE_ACTION_RUN_ROOT:?MOTIVE_ACTION_RUN_ROOT is required}"
prep_root="${MOTIVE_ACTION_PREP_ROOT:?MOTIVE_ACTION_PREP_ROOT is required}"
runtime_repo="${MOTIVE_RUNTIME_REPO:?MOTIVE_RUNTIME_REPO is required}"
source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?MOTIVE_SOURCE_SNAPSHOT is required}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?MOTIVE_SOURCE_TREE_SHA256 is required}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
task_index="${SLURM_ARRAY_TASK_ID:-}"

for required in \
  "${runtime_repo}/lucy/scripts/motive/run_goku_action_screen.sh" \
  "${prep_root}/lucy_train_manifest.jsonl" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[motive-retry] missing required path: ${required}" >&2
    exit 2
  fi
done

export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"
"${python_bin}" -c \
  'import torch; n=torch.cuda.device_count(); print(f"[motive-retry] torch={torch.__version__} hip={torch.version.hip} visible_gpus={n}", flush=True); assert n == 8, f"expected 8 visible GPUs, got {n}"; assert torch.version.hip, "expected a ROCm build"'

base_arm=""
run_label=""
train_router=1
freeze_random_trunk=0
residual_strength=0.05
entropy_weight=0.005
balance_weight=0.03
orthogonal_weight=0.003
max_total_grad=100
max_transformer_grad=100
max_router_grad=100

case "${phase}:${task_index}" in
  smoke:0)
    base_arm="e2_random_router"
    run_label="s0_identity_fixed"
    train_router=0
    residual_strength=0
    entropy_weight=0
    balance_weight=0
    orthogonal_weight=0
    max_total_grad=1
    max_transformer_grad=1
    max_router_grad=1
    ;;
  smoke:1)
    base_arm="e2_random_router"
    run_label="s1_identity_random_linear_noreg"
    freeze_random_trunk=1
    residual_strength=0
    entropy_weight=0
    balance_weight=0
    orthogonal_weight=0
    max_total_grad=1
    max_transformer_grad=1
    max_router_grad=1
    ;;
  smoke:2)
    base_arm="e2_random_router"
    run_label="s2_identity_random_linear_reg"
    freeze_random_trunk=1
    residual_strength=0
    max_total_grad=1
    max_transformer_grad=1
    max_router_grad=1
    ;;
  smoke:3)
    base_arm="e2_random_router"
    run_label="s3_fixed_random"
    train_router=0
    residual_strength=0.05
    entropy_weight=0
    balance_weight=0
    orthogonal_weight=0
    ;;
  smoke:4)
    base_arm="e2_random_router"
    run_label="s4_random_linear"
    freeze_random_trunk=1
    ;;
  smoke:5)
    base_arm="e3_motive_frozen"
    run_label="s5_motive_linear"
    ;;
  full:0)
    base_arm="e2_random_router"
    run_label="e2_fixed_random"
    train_router=0
    entropy_weight=0
    balance_weight=0
    orthogonal_weight=0
    ;;
  full:1)
    base_arm="e2_random_router"
    run_label="e2_random_router"
    freeze_random_trunk=1
    ;;
  full:2)
    base_arm="e3_motive_frozen"
    run_label="e3_motive_frozen"
    ;;
  *)
    echo "[motive-retry] unsupported phase/task: ${phase}:${task_index}" >&2
    exit 2
    ;;
esac

if [[ "${phase}" == "smoke" ]]; then
  output_root="${run_root}/smoke/lucy"
  status_root="${run_root}/smoke/status"
  max_steps=15
  save_every=15
  log_every=1
elif [[ "${phase}" == "full" ]]; then
  output_root="${run_root}/lucy"
  status_root="${run_root}/status"
  max_steps=100
  save_every=25
  log_every=5
else
  echo "[motive-retry] MOTIVE_RETRY_PHASE must be smoke or full" >&2
  exit 2
fi

export MOTIVE_REPO_ROOT="${runtime_repo}"
export MOTIVE_ACTION_EXPERIMENT_ROOT="${prep_root}"
export MOTIVE_LUCY_OUTPUT_ROOT="${output_root}"
export MOTIVE_ACTION_STATUS_ROOT="${status_root}"
export MOTIVE_ACTION_RUN_LABEL="${run_label}"
export MOTIVE_SOURCE_TREE_SHA256="${source_tree_sha256}"
export MOTIVE_TRAIN_SEED="${MOTIVE_TRAIN_SEED:-2026}"
export MOTIVE_LUCY_TRAIN_SAMPLES="${MOTIVE_LUCY_TRAIN_SAMPLES:-64}"
export MOTIVE_LUCY_NUM_PROCESSES=8
export MOTIVE_LUCY_BATCH_SIZE=1
export MOTIVE_LUCY_GRAD_ACCUM=1
export MOTIVE_LUCY_MAX_STEPS="${max_steps}"
export MOTIVE_LUCY_SAVE_EVERY="${save_every}"
export MOTIVE_LUCY_LOG_EVERY="${log_every}"
export MOTIVE_V10_LORA_TRAIN_ROUTER="${train_router}"
export MOTIVE_V10_RANDOM_ROUTER_FREEZE_TRUNK="${freeze_random_trunk}"
export MOTIVE_V10_LORA_RESIDUAL_GATE_SCALE="${residual_strength}"
export MOTIVE_V10_LORA_GATE_ENTROPY_WEIGHT="${entropy_weight}"
export MOTIVE_V10_LORA_GATE_BALANCE_WEIGHT="${balance_weight}"
export MOTIVE_V10_LORA_GATE_ORTHOGONAL_WEIGHT="${orthogonal_weight}"
export MOTIVE_ACTION_MAX_TOTAL_GRAD="${max_total_grad}"
export MOTIVE_ACTION_MAX_TRANSFORMER_GRAD="${max_transformer_grad}"
export MOTIVE_ACTION_MAX_ROUTER_GRAD="${max_router_grad}"
export MOTIVE_ACTION_VALIDATE_TRAINING=1
export PYTHON_BIN="${python_bin}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${runtime_repo}"
exec bash lucy/scripts/motive/run_goku_action_screen.sh "${base_arm}"
