#!/bin/bash
set -euo pipefail

SOURCE_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_object_grounded_case01_0821_source_r1
PROGRAM="$SOURCE_ROOT/materialize_case01_sam2_object_masklets_v1.py"
SPEC="$SOURCE_ROOT/case01_288545b9c031491a_sam2_boxes_v1.json"
OUTPUT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_object_grounded_case01_0821_sam2_masklets_r2
CACHE=/tmp/bernini-case01-sam2-job143808-node292-r2
PYTHON=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

test ! -e "$OUTPUT"
test ! -L "$OUTPUT"
test "$(/usr/bin/sha256sum "$PROGRAM" | /usr/bin/cut -d' ' -f1)" = 87d2b4b33379ed7c444ea4b8b57dc2a5eaf24188e97fcd144f6d377491f67404
test "$(/usr/bin/sha256sum "$SPEC" | /usr/bin/cut -d' ' -f1)" = 8d8284f588434e2b3553cf568489856adc7c15265a4b1339624f609c04c09207
test -x "$PYTHON"

exec /usr/bin/srun \
  --jobid=143808 \
  --nodes=1 \
  --nodelist=auh7-1b-gpu-292 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=32G \
  --gres=gpu:1 \
  --exclusive \
  --exact \
  --time=00:20:00 \
  --job-name=case01-sam2-masklets-r2 \
  /usr/bin/env -i \
    PATH=/usr/bin:/bin \
    LC_ALL=C \
    LANG=C \
    HOME=/vast/users/guangyi.chen \
    PROGRAM="$PROGRAM" \
    SPEC="$SPEC" \
    OUTPUT="$OUTPUT" \
    CACHE="$CACHE" \
    PYTHON="$PYTHON" \
    /bin/bash -c '
      set -euo pipefail
      test ! -e "$CACHE"
      test ! -L "$CACHE"
      test ! -e "$OUTPUT"
      test ! -L "$OUTPUT"
      /usr/bin/mkdir -p \
        "$CACHE/home" \
        "$CACHE/tmp" \
        "$CACHE/xdg" \
        "$CACHE/pycache" \
        "$CACHE/torch" \
        "$CACHE/miopen-user" \
        "$CACHE/miopen-custom" \
        "$CACHE/triton" \
        "$CACHE/torch-extensions" \
        "$CACHE/inductor"
      /usr/bin/chmod 0700 "$CACHE" "$CACHE"/*
      export HOME="$CACHE/home"
      export TMPDIR="$CACHE/tmp"
      export XDG_CACHE_HOME="$CACHE/xdg"
      export PYTHONPYCACHEPREFIX="$CACHE/pycache"
      export TORCH_HOME="$CACHE/torch"
      export MIOPEN_USER_DB_PATH="$CACHE/miopen-user"
      export MIOPEN_CUSTOM_CACHE_DIR="$CACHE/miopen-custom"
      export TRITON_CACHE_DIR="$CACHE/triton"
      export TORCH_EXTENSIONS_DIR="$CACHE/torch-extensions"
      export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
      exec "$PYTHON" -I -B "$PROGRAM" --spec "$SPEC" --output-dir "$OUTPUT"
    '
