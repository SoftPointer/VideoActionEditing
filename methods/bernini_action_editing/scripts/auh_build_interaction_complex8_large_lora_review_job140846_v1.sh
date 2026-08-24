#!/usr/bin/env bash
set -euo pipefail

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
score_root="$stage/interaction_complex8_large_lora_action_score_v1"
builder="$stage/build_interaction_complex8_large_lora_review_v1.py"
output="$stage/interaction_complex8_large_lora_review_v1"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
while [ "$(find "$score_root" -name COMPLETE -type f 2>/dev/null | wc -l)" -ne 8 ]; do sleep 15; done
test ! -e "$output"
"$python_bin" -B "$builder" --stage "$stage" --output "$output"
test -f "$output/index.html"
test -f "$output/manifest.json"
test "$(find "$output/media" -name '*.mp4' -type f | wc -l)" -eq 104
touch "$output/COMPLETE"
