#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RELEASE_DIR SOURCE_MANIFEST OUTPUT_DIR" >&2
  exit 2
fi

RELEASE_DIR=$1
SOURCE_MANIFEST=$2
OUTPUT_DIR=$3
PY=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python
BUILDER=$RELEASE_DIR/build_exact644_semantic_action_feature_manifest_v1.py
EXTRACTOR=$RELEASE_DIR/run_audit.py
SEMANTIC_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/semantic_moments_reward_20260815/vendor/semantic-moments
MODEL_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/vendor/dinov2-base-f9e44c8
EXPECTED_BUILDER_SHA=59074171fac868fa7fcc1153ca9b43e9844fdd616de8d573d0765fb137f1f00b
EXPECTED_EXTRACTOR_SHA=898951991905eada734a4440858538af15a805881c084017047ef8bbf75639d8
EXPECTED_MODEL_SHA=d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841

[[ -x $PY && -f $BUILDER && -f $EXTRACTOR && -f $SOURCE_MANIFEST ]]
[[ $(sha256sum "$BUILDER" | awk '{print $1}') == "$EXPECTED_BUILDER_SHA" ]]
[[ $(sha256sum "$EXTRACTOR" | awk '{print $1}') == "$EXPECTED_EXTRACTOR_SHA" ]]
[[ $(sha256sum "$MODEL_ROOT/model.safetensors" | awk '{print $1}') == "$EXPECTED_MODEL_SHA" ]]
[[ ! -e $OUTPUT_DIR ]]
mkdir -p "$OUTPUT_DIR/features" "$OUTPUT_DIR/logs"
chmod 0700 "$OUTPUT_DIR" "$OUTPUT_DIR/features" "$OUTPUT_DIR/logs"

MANIFEST=$OUTPUT_DIR/exact644-source-anchor-feature-manifest.json
"$PY" "$BUILDER" \
  --source-manifest "$SOURCE_MANIFEST" \
  --output "$MANIFEST" \
  >"$OUTPUT_DIR/logs/build-manifest.stdout" \
  2>"$OUTPUT_DIR/logs/build-manifest.stderr"

if [[ -z ${SLURM_JOB_ID:-} || -z ${SLURM_STEP_ID:-} ]]; then
  echo "launcher must execute inside a Slurm job step" >&2
  exit 3
fi

CACHE_BASE=/tmp/semantic-action-exact1288-j${SLURM_JOB_ID}-s${SLURM_STEP_ID}
[[ ! -e $CACHE_BASE ]]
mkdir -p "$CACHE_BASE"
chmod 0700 "$CACHE_BASE"

pids=()
for rank in 0 1 2 3 4 5 6 7; do
  (
    set -euo pipefail
    CACHE_ROOT=$CACHE_BASE/rank-$rank
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
    export ROCR_VISIBLE_DEVICES=$rank
    export HIP_VISIBLE_DEVICES=$rank
    export CUDA_VISIBLE_DEVICES=$rank
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
  ) >"$OUTPUT_DIR/logs/shard-$rank.stdout" \
    2>"$OUTPUT_DIR/logs/shard-$rank.stderr" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ $failed -ne 0 ]]; then
  echo "at least one feature shard failed" >&2
  exit 4
fi

"$PY" - "$OUTPUT_DIR" "$MANIFEST" "$EXPECTED_MODEL_SHA" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import sys
import torch

root = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
expected_model_sha = sys.argv[3]

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

records = []
shards = []
manifest_digest = None
for index in range(8):
    path = root / "features" / f"features-shard-{index}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "semantic-moments-action-reward-features-v1":
        raise RuntimeError("feature schema differs")
    if payload.get("shard_index") != index or payload.get("num_shards") != 8:
        raise RuntimeError("feature shard placement differs")
    if manifest_digest is None:
        manifest_digest = payload.get("manifest_digest")
    elif manifest_digest != payload.get("manifest_digest"):
        raise RuntimeError("feature shards disagree on manifest")
    model_files = {
        row["relative_path"]: row["sha256"]
        for row in payload["runtime"]["model_files"]
    }
    if model_files.get("model.safetensors") != expected_model_sha:
        raise RuntimeError("frozen DINO weights differ")
    records.extend(payload["records"])
    shards.append({
        "index": index,
        "path": str(path),
        "sha256": sha(path),
        "size_bytes": path.stat().st_size,
        "record_count": payload["record_count"],
    })

if len(records) != 1288:
    raise RuntimeError(f"expected exact1288 records, got {len(records)}")
ids = [row["item_id"] for row in records]
if len(set(ids)) != 1288:
    raise RuntimeError("feature item IDs are not unique")
groups = {}
for row in records:
    if tuple(row["frame_sequence"].shape) != (32, 768):
        raise RuntimeError("ordered feature geometry differs")
    if tuple(row["components"].shape) != (3, 768):
        raise RuntimeError("moment component geometry differs")
    if not bool(torch.isfinite(row["frame_sequence"]).all()):
        raise RuntimeError("ordered features contain non-finite values")
    if not bool(torch.isfinite(row["components"]).all()):
        raise RuntimeError("moment components contain non-finite values")
    groups[row["group"]] = groups.get(row["group"], 0) + 1
if groups != {"exact644_action_anchor": 644, "exact644_source": 644}:
    raise RuntimeError(f"feature group coverage differs: {groups}")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest_digest != manifest["manifest_digest"]:
    raise RuntimeError("feature payload does not bind the published manifest")
receipt = {
    "schema_version": "semantic-action-exact644-feature-extraction-receipt-v1",
    "status": "FEATURES_EXTRACTED_NOT_REPRESENTATION_QUALIFIED",
    "authority": "feature_mechanics_diagnostic_only",
    "formal_training_authorized": False,
    "paired_ground_truth_claimed": False,
    "holder_job_id": str(__import__("os").environ["SLURM_JOB_ID"]),
    "step_id": str(__import__("os").environ["SLURM_STEP_ID"]),
    "hostname": socket.gethostname(),
    "manifest": {
        "path": str(manifest_path),
        "sha256": sha(manifest_path),
        "manifest_digest": manifest_digest,
    },
    "population": {
        "unique_base_clips": 644,
        "action_anchor_records": 644,
        "source_records_for_nuisance_probe": 644,
        "total_feature_records": 1288,
        "counterfactual_rows": 0,
    },
    "feature_geometry": {"frames": 32, "dimension": 768, "moments": 3},
    "frozen_teacher": {
        "kind": "DINOv2-base ordered per-frame descriptors",
        "weights_sha256": expected_model_sha,
        "semantic_moments_role": "unordered auxiliary only",
    },
    "shards": shards,
}
unsigned = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
receipt["receipt_digest"] = hashlib.sha256(unsigned).hexdigest()
raw = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False).encode("ascii") + b"\n"
destination = root / "feature_extraction_receipt.json"
with destination.open("xb") as handle:
    handle.write(raw)
    handle.flush()
print(json.dumps({"receipt": str(destination), "sha256": hashlib.sha256(raw).hexdigest(), "counts": groups}, sort_keys=True))
PY

chmod 0444 "$MANIFEST" "$OUTPUT_DIR/features"/*.pt "$OUTPUT_DIR/feature_extraction_receipt.json"
chmod 0444 "$OUTPUT_DIR/logs"/*
chmod 0555 "$OUTPUT_DIR/features" "$OUTPUT_DIR/logs" "$OUTPUT_DIR"
