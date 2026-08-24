#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SHARD_INDEX NUM_SHARDS" >&2
  exit 2
fi

shard_index=$1
num_shards=$2
repo=${ACTION_GAP_REPO:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit}
v2_root=${ACTION_GAP_V2_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev_action_anchor_target_gap16_20260819_v2}
v3_root=${ACTION_GAP_V3_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev_action_anchor_target_gap16_20260819_v3}
python_bin=${ACTION_GAP_QWEN_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/qwen/bin/python3.12}
model=${ACTION_GAP_QWEN_MODEL:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VLM/MEV-Annotation/checkpoints/Qwen3-VL-32B-Instruct}

cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" -m methods.action_anchor_target_gap_audit.source_residue_eval qwen-rescore \
  --manifest "$v2_root/manifest.json" \
  --target-contracts "$repo/methods/action_anchor_target_gap_audit/manual_action_contracts_v2.json" \
  --source-contracts "$repo/methods/action_anchor_target_gap_audit/manual_source_residue_contracts_v3.json" \
  --records "$v2_root/qwen_v2/full/qwen-v2-shard-0.jsonl" \
  --records "$v2_root/qwen_v2/full/qwen-v2-shard-1.jsonl" \
  --records "$v2_root/qwen_v2/full/qwen-v2-shard-2.jsonl" \
  --records "$v2_root/qwen_v2/full/qwen-v2-shard-3.jsonl" \
  --model "$model" \
  --shard-index "$shard_index" \
  --num-shards "$num_shards" \
  --max-new-tokens 768 \
  --resume \
  --output "$v3_root/qwen_v3/full/qwen-v3-shard-$shard_index.jsonl"
