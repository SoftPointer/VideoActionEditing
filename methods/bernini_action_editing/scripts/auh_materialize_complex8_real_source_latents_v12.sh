#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this controller must run on the AUH login host" >&2; exit 2 ;;
esac

job="${REAL_SOURCE_JOB:-143808}"
node="${REAL_SOURCE_NODE:-auh7-1b-gpu-268}"
visible="${REAL_SOURCE_VISIBLE_DEVICE:-7}"
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
source_tree="$stage/source-online-anchor-real-source-teacher-delta-v12"
source_fields="$source_tree/methods/bernini_action_editing/assets/interaction_complex8_real_source_fields_v1.json"
output="$release/complex8_real_source_latents_v12"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
scratch="$release/cache/real-source-latents-v12-${job}-${node}"

test -f "$source_fields"
test ! -e "$output"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions"
srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
  env \
    ROCR_VISIBLE_DEVICES="$visible" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    MIOPEN_USER_DB_PATH="$scratch/miopen-user" \
    MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom" \
    TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" \
    PYTHONPATH="$source_tree/methods/bernini_action_editing:$stage/source-sga-anc-training-v1/methods/bernini_action_editing:$stage/source-be31323/methods/bernini_action_editing" \
    "$python_bin" -B \
      "$source_tree/methods/bernini_action_editing/materialize_complex8_real_source_latents_v1.py" \
      --bernini-root "$bernini_root" \
      --veomni-root "$veomni_root" \
      --checkpoint "$checkpoint" \
      --source-fields "$source_fields" \
      --output "$output"
