#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 ABS_RUNNER ABS_MANIFEST ABS_PREREG ABS_OUTPUT" >&2
  exit 64
fi

readonly runner="$1"
readonly manifest="$2"
readonly prereg="$3"
readonly output="$4"
readonly python_bin="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
readonly model_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev_action_anchor_target_gap16_20260819_v3/vendor/vjepa2-vitl-fpc64-256-b3c1679"

for path in "$runner" "$manifest" "$prereg" "$python_bin"; do
  [[ "$path" = /* && -f "$path" && ! -L "$path" ]] || {
    echo "required absolute regular non-symlink file differs: $path" >&2
    exit 65
  }
done
[[ "$model_root" = /* && -d "$model_root" && ! -L "$model_root" ]] || exit 66
[[ "$output" = /* && ! -e "$output" && ! -L "$output" ]] || exit 67
[[ "${SLURM_NTASKS:-}" == "1" && "${SLURM_NNODES:-}" == "1" ]] || {
  echo "R1b requires one exact Slurm task/node" >&2
  exit 68
}
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || {
  echo "R1b requires one logical GPU" >&2
  exit 69
}

readonly source_root="$(cd "$(dirname "$runner")/../.." && pwd -P)"
for relative in \
  "methods/bernini_action_editing/target_middle_object_graph_teacher_pilot_v5.py" \
  "methods/bernini_action_editing/extract_vjepa2_ordered_contextual_features_v4c.py" \
  "methods/action_anchor_target_gap_audit/representation_eval.py" \
  "methods/action_anchor_target_gap_audit/audit.py" \
  "methods/action_anchor_target_gap_audit/corrected_eval.py"; do
  [[ -f "$source_root/$relative" && ! -L "$source_root/$relative" ]] || exit 70
done

export PYTHONPATH="$source_root${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

exec "$python_bin" -B "$runner" \
  --manifest "$manifest" \
  --prereg "$prereg" \
  --wrapper "$0" \
  --model-root "$model_root" \
  --device cuda:0 \
  --output "$output"
