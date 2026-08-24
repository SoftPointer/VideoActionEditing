#!/usr/bin/env bash
#SBATCH --job-name=bern-spt-student
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=1-00:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_root="${BERNINI_SPT_SOURCE_ROOT:?set BERNINI_SPT_SOURCE_ROOT}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
data_dir="${BERNINI_ACTION_PARQUET_DIR:?set BERNINI_ACTION_PARQUET_DIR}"
dataset_summary="${BERNINI_ACTION_DATASET_SUMMARY:?set BERNINI_ACTION_DATASET_SUMMARY}"
output_dir="${BERNINI_SPT_TRAIN_OUTPUT:?set BERNINI_SPT_TRAIN_OUTPUT}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
max_steps="${BERNINI_SPT_MAX_STEPS:-644}"
save_every="${BERNINI_SPT_SAVE_EVERY:-64}"
resume="${BERNINI_SPT_RESUME:-}"
allow_incomplete="${BERNINI_ACTION_ALLOW_INCOMPLETE_DATASET:-0}"
planner_architecture="${BERNINI_SPT_PLANNER_ARCHITECTURE:-phase_query_v2}"
hidden_channels="${BERNINI_SPT_HIDDEN_CHANNELS:-}"
attention_heads="${BERNINI_SPT_ATTENTION_HEADS:-8}"
match_channels="${BERNINI_SPT_MATCH_CHANNELS:-32}"
edit_slots="${BERNINI_SPT_EDIT_SLOTS:-8}"
dense_query_chunk_size="${BERNINI_SPT_DENSE_QUERY_CHUNK_SIZE:-4096}"
learning_rate="${BERNINI_SPT_LEARNING_RATE:-0.0002}"
gate_loss_weight="${BERNINI_SPT_GATE_LOSS_WEIGHT:-1.0}"
conditional_gate_loss_weight="${BERNINI_SPT_CONDITIONAL_GATE_LOSS_WEIGHT:-1.0}"
gate_mass_loss_weight="${BERNINI_SPT_GATE_MASS_LOSS_WEIGHT:-0.05}"
offset_loss_weight="${BERNINI_SPT_OFFSET_LOSS_WEIGHT:-0.25}"
smooth_loss_weight="${BERNINI_SPT_SMOOTH_LOSS_WEIGHT:-0.01}"
noop_loss_weight="${BERNINI_SPT_NOOP_LOSS_WEIGHT:-0.25}"
change_tversky_weight="${BERNINI_SPT_CHANGE_TVERSKY_WEIGHT:-0.5}"
phase_change_mass_weight="${BERNINI_SPT_PHASE_CHANGE_MASS_WEIGHT:-0.1}"
phase_generate_mass_weight="${BERNINI_SPT_PHASE_GENERATE_MASS_WEIGHT:-0.1}"
mid_change_loss_weight="${BERNINI_SPT_MID_CHANGE_LOSS_WEIGHT:-0.25}"
coarse_change_loss_weight="${BERNINI_SPT_COARSE_CHANGE_LOSS_WEIGHT:-0.125}"
expected_offset_loss_weight="${BERNINI_SPT_EXPECTED_OFFSET_LOSS_WEIGHT:-0.1}"
noop_generate_weight="${BERNINI_SPT_NOOP_GENERATE_WEIGHT:-0.2}"
noop_offset_weight="${BERNINI_SPT_NOOP_OFFSET_WEIGHT:-0.25}"
counterfactual_change_loss_weight="${BERNINI_SPT_COUNTERFACTUAL_CHANGE_LOSS_WEIGHT:-0.0}"
change_polarization_loss_weight="${BERNINI_SPT_CHANGE_POLARIZATION_LOSS_WEIGHT:-0.0}"
train_prefix_rows="${BERNINI_SPT_TRAIN_PREFIX_ROWS:-}"
selected_membership="${BERNINI_SPT_SELECTED_MEMBERSHIP:-}"
nproc="${BERNINI_SPT_NPROC:-4}"

fail() { echo "[bernini-spt-student] ERROR: $*" >&2; exit 2; }
[[ -f "${source_root}/spt_v2/train_student.py" ]] || fail "source root lacks SPT trainer"
[[ -f "${source_root}/train_lora.py" ]] || fail "source root lacks pinned Bernini helpers"
[[ -d "${bernini_root}" ]] || fail "missing Bernini root"
[[ -d "${veomni_root}/.git" ]] || fail "VeOmni must be a Git checkout"
[[ -d "${checkpoint}" ]] || fail "missing checkpoint"
[[ -d "${data_dir}" ]] || fail "missing parquet directory"
[[ -f "${dataset_summary}" ]] || fail "missing dataset summary"
[[ "${max_steps}" =~ ^[1-9][0-9]*$ ]] || fail "invalid max steps"
[[ "${save_every}" =~ ^[0-9]+$ ]] || fail "invalid save interval"
[[ "${allow_incomplete}" =~ ^[01]$ ]] || fail "invalid incomplete flag"
[[ "${planner_architecture}" =~ ^(phase_query_v2|grounded_cell_phase_transport_v3)$ ]] || fail "unsupported planner architecture"
if [[ -z "${hidden_channels}" ]]; then
  if [[ "${planner_architecture}" == grounded_cell_phase_transport_v3 ]]; then
    hidden_channels=192
  else
    hidden_channels=128
  fi
fi
for integer_value in "${hidden_channels}" "${attention_heads}" "${match_channels}" "${edit_slots}" "${dense_query_chunk_size}"; do
  [[ "${integer_value}" =~ ^[1-9][0-9]*$ ]] || fail "planner integer settings must be positive"
done
[[ "${nproc}" =~ ^(1|4)$ ]] || fail "nproc must be 1 (debug) or 4 (AUH RCCL)"
if [[ -n "${train_prefix_rows}" ]]; then
  [[ "${train_prefix_rows}" =~ ^[1-9][0-9]*$ ]] || fail "invalid train prefix row count"
fi
[[ -z "${selected_membership}" || ( -f "${selected_membership}" && ! -L "${selected_membership}" ) ]] || fail "invalid selected membership"
[[ -z "${selected_membership}" || -z "${train_prefix_rows}" ]] || fail "selected membership and train prefix are mutually exclusive"
if [[ -n "${resume}" ]]; then [[ -d "${resume}" ]] || fail "missing resume checkpoint"; fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

task_scratch="${SLURM_TMPDIR:-/tmp}/bernini-spt-student-${SLURM_JOB_ID}"
mkdir -p \
  "${task_scratch}/method" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton" \
  "${task_scratch}/rank-cache"
cp -a "${source_root}/." "${task_scratch}/method/"
method_root="${task_scratch}/method"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
export BERNINI_SPT_RANK_CACHE_ROOT="${task_scratch}/rank-cache"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

for pattern in test_spt_v2_contract.py test_spt_v2_phase_query_planner.py test_spt_v2_train_student_contract.py test_spt_v3_grounded_planner.py test_spt_v3_train_student_contract.py; do
  PYTHONPATH="${method_root}:${method_root}/spt_v2" "${python_bin}" -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

command=(
  "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node="${nproc}"
  "${method_root}/spt_v2/train_student.py"
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --preprocessed-parquet-dir "${data_dir}"
  --dataset-summary "${dataset_summary}"
  --output "${output_dir}"
  --num-frames 81
  --planner-architecture "${planner_architecture}"
  --hidden-channels "${hidden_channels}"
  --attention-heads "${attention_heads}"
  --match-channels "${match_channels}"
  --edit-slots "${edit_slots}"
  --dense-query-chunk-size "${dense_query_chunk_size}"
  --learning-rate "${learning_rate}"
  --gate-loss-weight "${gate_loss_weight}"
  --conditional-gate-loss-weight "${conditional_gate_loss_weight}"
  --gate-mass-loss-weight "${gate_mass_loss_weight}"
  --offset-loss-weight "${offset_loss_weight}"
  --smooth-loss-weight "${smooth_loss_weight}"
  --noop-loss-weight "${noop_loss_weight}"
  --change-tversky-weight "${change_tversky_weight}"
  --phase-change-mass-weight "${phase_change_mass_weight}"
  --phase-generate-mass-weight "${phase_generate_mass_weight}"
  --mid-change-loss-weight "${mid_change_loss_weight}"
  --coarse-change-loss-weight "${coarse_change_loss_weight}"
  --expected-offset-loss-weight "${expected_offset_loss_weight}"
  --noop-generate-weight "${noop_generate_weight}"
  --noop-offset-weight "${noop_offset_weight}"
  --counterfactual-change-loss-weight "${counterfactual_change_loss_weight}"
  --change-polarization-loss-weight "${change_polarization_loss_weight}"
  --max-steps "${max_steps}"
  --save-every "${save_every}"
)
if [[ "${allow_incomplete}" == 1 ]]; then command+=(--allow-incomplete-dataset); fi
if [[ -n "${train_prefix_rows}" ]]; then command+=(--train-prefix-rows "${train_prefix_rows}"); fi
if [[ -n "${selected_membership}" ]]; then command+=(--selected-membership "${selected_membership}"); fi
if [[ -n "${resume}" ]]; then command+=(--resume "${resume}"); fi
"${command[@]}"

echo "[bernini-spt-student] PASS"
