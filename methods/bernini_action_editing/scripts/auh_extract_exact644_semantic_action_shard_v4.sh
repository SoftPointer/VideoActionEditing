#!/usr/bin/env bash
set -euo pipefail
# AUH exact8 task-local GPU binding, revision 4.

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RELEASE_DIR MANIFEST OUTPUT_DIR" >&2
  exit 2
fi
if [[ ${SLURM_NTASKS:-} != 8 ]]; then
  echo "this wrapper requires exact8 Slurm tasks" >&2
  exit 3
fi
rank=${SLURM_PROCID:?SLURM_PROCID is required}
if (( rank < 0 || rank > 7 )); then
  echo "invalid shard rank: $rank" >&2
  exit 4
fi

RELEASE_DIR=$1
MANIFEST=$2
OUTPUT_DIR=$3
PY=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python
EXTRACTOR=$RELEASE_DIR/run_audit.py
SEMANTIC_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/semantic_moments_reward_20260815/vendor/semantic-moments
MODEL_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/vendor/dinov2-base-f9e44c8
EXPECTED_EXTRACTOR_SHA=898951991905eada734a4440858538af15a805881c084017047ef8bbf75639d8

[[ -x $PY && -f $EXTRACTOR && -f $MANIFEST && -d $OUTPUT_DIR/features && -d $OUTPUT_DIR/logs ]]
[[ $(sha256sum "$EXTRACTOR" | awk '{print $1}') == "$EXPECTED_EXTRACTOR_SHA" ]]
[[ ! -e $OUTPUT_DIR/features/features-shard-$rank.pt ]]

CACHE_ROOT=/tmp/semantic-action-exact1288-j${SLURM_JOB_ID:?}-s${SLURM_STEP_ID:?}-r$rank
[[ ! -e $CACHE_ROOT ]]
mkdir -p \
  "$CACHE_ROOT/miopen-user" \
  "$CACHE_ROOT/miopen-custom" \
  "$CACHE_ROOT/xdg" \
  "$CACHE_ROOT/tmp" \
  "$CACHE_ROOT/triton" \
  "$CACHE_ROOT/inductor" \
  "$CACHE_ROOT/extensions" \
  "$CACHE_ROOT/pycache"
chmod 0700 "$CACHE_ROOT" "$CACHE_ROOT"/*
export MIOPEN_USER_DB_PATH=$CACHE_ROOT/miopen-user
export MIOPEN_CUSTOM_CACHE_DIR=$CACHE_ROOT/miopen-custom
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export TMPDIR=$CACHE_ROOT/tmp
export TMP=$CACHE_ROOT/tmp
export TEMP=$CACHE_ROOT/tmp
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHE_ROOT/inductor
export TORCH_EXTENSIONS_DIR=$CACHE_ROOT/extensions
export PYTHONPYCACHEPREFIX=$CACHE_ROOT/pycache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
# AUH exposes the allocation-wide device list to every Slurm task even with
# --gpus-per-task.  Bind each process to its task-local physical ordinal before
# importing torch; the one-task/eight-background-process layout is not valid on
# this cluster because that single task receives only one cgroup device.
export ROCR_VISIBLE_DEVICES=${SLURM_LOCALID}
# Do not apply a second logical-device filter after ROCR.  Setting all three
# variables to a physical ordinal leaves ranks 1..7 with an empty singleton.
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

"$PY" - <<'PY'
import json, os, torch
value = {
    "rank": int(os.environ["SLURM_PROCID"]),
    "local_rank": int(os.environ["SLURM_LOCALID"]),
    "visible_count": torch.cuda.device_count(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
}
print(json.dumps(value, sort_keys=True), flush=True)
if value["visible_count"] != 1 or value["device"] != "AMD Instinct MI210":
    raise SystemExit("Slurm did not assign exactly one MI210 to this task")
PY

"$PY" "$EXTRACTOR" extract-shard \
  --manifest "$MANIFEST" \
  --semantic-moments-root "$SEMANTIC_ROOT" \
  --model-root "$MODEL_ROOT" \
  --shard-index "$rank" \
  --num-shards 8 \
  --num-frames 32 \
  --frame-batch-size 16 \
  --device cuda:0 \
  --output "$OUTPUT_DIR/features/features-shard-$rank.pt"
