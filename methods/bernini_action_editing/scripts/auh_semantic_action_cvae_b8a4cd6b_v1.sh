#!/usr/bin/env bash
set -euo pipefail

readonly PY=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python
readonly BASE=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/semantic_action_cvae_r0_20260819
readonly RELEASE="$BASE/release_canary_b8a4cd6b_v1"
readonly CODE="$RELEASE/methods/bernini_action_editing/semantic_action_cvae_canary_v1.py"
readonly TEST="$RELEASE/methods/bernini_action_editing/tests/test_semantic_action_cvae_canary_v1.py"
readonly FEATURE_ROOT="$BASE/runs/extract_exact1288_v4"
readonly FEATURE_RECEIPT="$FEATURE_ROOT/feature_extraction_receipt.json"
readonly CODE_SHA=b8a4cd6bac5f05a992f668578154bd74268d27526942ec446a2012aa2b2c1a97
readonly TEST_SHA=b7c67fe06f566231e252830d081caeb22ad01f9260f791c6cecc1f5237e2654e
readonly FEATURE_RECEIPT_SHA=8ff8f5fd5be36cb67ce40d5558a4406bdf70cbe9b72b0c43c71fa3abe8f6ad9c

if [[ $# -ne 1 ]]; then
  echo "usage: $0 smoke|full" >&2
  exit 64
fi
case "$1" in
  smoke)
    readonly STEPS=2
    readonly OUTPUT="$BASE/runs/canary_b8a4cd6b_smoke_v1"
    ;;
  full)
    readonly STEPS=1200
    readonly OUTPUT="$BASE/runs/canary_b8a4cd6b_exact644_v1"
    ;;
  *)
    echo "mode must be exactly smoke or full" >&2
    exit 64
    ;;
esac

[[ "${SLURM_JOB_ID:-}" == 141620 ]]
[[ -n "${SLURM_STEP_ID:-}" ]]
[[ "$(hostname -s)" == auh7-1b-gpu-226 ]]
[[ ! -e "$OUTPUT" ]]
[[ -d "$BASE/runs" ]]
[[ "$(sha256sum "$CODE" | cut -d' ' -f1)" == "$CODE_SHA" ]]
[[ "$(sha256sum "$TEST" | cut -d' ' -f1)" == "$TEST_SHA" ]]
[[ "$(sha256sum "$FEATURE_RECEIPT" | cut -d' ' -f1)" == "$FEATURE_RECEIPT_SHA" ]]
[[ "$(stat -c '%a:%h' "$CODE")" == 444:1 ]]
[[ "$(stat -c '%a:%h' "$TEST")" == 444:1 ]]
[[ "$(stat -c '%a:%h' "$FEATURE_RECEIPT")" == 444:1 ]]

umask 077
readonly CACHE_ROOT="/tmp/semantic-action-cvae-j${SLURM_JOB_ID}-s${SLURM_STEP_ID}-r0"
mkdir "$CACHE_ROOT"
mkdir "$CACHE_ROOT/tmp" "$CACHE_ROOT/xdg" "$CACHE_ROOT/miopen-user" \
  "$CACHE_ROOT/miopen-custom" "$CACHE_ROOT/triton" \
  "$CACHE_ROOT/torchinductor" "$CACHE_ROOT/torch-extensions" \
  "$CACHE_ROOT/pycache"

export TMPDIR="$CACHE_ROOT/tmp"
export TMP="$CACHE_ROOT/tmp"
export TEMP="$CACHE_ROOT/tmp"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export MIOPEN_USER_DB_PATH="$CACHE_ROOT/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="$CACHE_ROOT/miopen-custom"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch-extensions"
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RELEASE"

exec "$PY" "$CODE" \
  --feature-root "$FEATURE_ROOT" \
  --expected-feature-receipt-sha256 "$FEATURE_RECEIPT_SHA" \
  --output "$OUTPUT" \
  --device cuda:0 \
  --seed 20260819 \
  --pca-dim 64 \
  --latent-dim 32 \
  --hidden-dim 256 \
  --steps "$STEPS" \
  --batch-size 128 \
  --learning-rate 0.002 \
  --beta-kl 0.02 \
  --prior-samples 8
