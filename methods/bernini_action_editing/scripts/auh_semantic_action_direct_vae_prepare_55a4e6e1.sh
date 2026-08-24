#!/usr/bin/env bash
set -euo pipefail

readonly PY=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly BASE=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/semantic_action_direct_vae_v1_20260819
readonly RELEASE="$BASE/release_55a4e6e1_candidate1"
readonly CODE="$RELEASE/methods/bernini_action_editing/semantic_action_direct_vae_canary_v1.py"
readonly COMMON="$RELEASE/methods/bernini_action_editing/semantic_action_cvae_canary_v1.py"
readonly TEST="$RELEASE/methods/bernini_action_editing/tests/test_semantic_action_direct_vae_canary_v1.py"
readonly FEATURE_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/semantic_action_cvae_r0_20260819/runs/extract_exact1288_v4
readonly FEATURE_RECEIPT="$FEATURE_ROOT/feature_extraction_receipt.json"
readonly CODE_SHA=55a4e6e10d4dfa77075e508bd921aef3916be0537fe9e8049c288ca22a7faefb
readonly COMMON_SHA=74fe161f08eb92746ce22f00b50cca65f85ef05e4a4a6a5d0c7fb85867e94233
readonly TEST_SHA=633ccd6ae2e8ef2d072ff88710878e58e0e0b3c8f02d7d727b12f2488e3e2e70
readonly FEATURE_RECEIPT_SHA=8ff8f5fd5be36cb67ce40d5558a4406bdf70cbe9b72b0c43c71fa3abe8f6ad9c

if [[ $# -ne 1 ]]; then
  echo "usage: $0 smoke|full" >&2
  exit 64
fi
case "$1" in
  smoke)
    readonly STEPS=2
    readonly OUTPUT="$BASE/runs/smoke_prepare_55a4e6e1_v1"
    ;;
  full)
    readonly STEPS=1200
    readonly OUTPUT="$BASE/runs/full_prepare_55a4e6e1_v1"
    ;;
  *)
    echo "mode must be smoke or full" >&2
    exit 64
    ;;
esac

[[ "${SLURM_JOB_ID:-}" == 143812 ]]
[[ -n "${SLURM_STEP_ID:-}" ]]
[[ "$(hostname -s)" == auh7-1b-gpu-293 ]]
[[ ! -e "$OUTPUT" ]]
[[ "$(sha256sum "$CODE" | cut -d' ' -f1)" == "$CODE_SHA" ]]
[[ "$(sha256sum "$COMMON" | cut -d' ' -f1)" == "$COMMON_SHA" ]]
[[ "$(sha256sum "$TEST" | cut -d' ' -f1)" == "$TEST_SHA" ]]
[[ "$(sha256sum "$FEATURE_RECEIPT" | cut -d' ' -f1)" == "$FEATURE_RECEIPT_SHA" ]]
[[ "$(stat -c '%a:%h' "$CODE")" == 444:1 ]]
[[ "$(stat -c '%a:%h' "$COMMON")" == 444:1 ]]
[[ "$(stat -c '%a:%h' "$TEST")" == 444:1 ]]
[[ "$(stat -c '%a:%h' "$FEATURE_RECEIPT")" == 444:1 ]]

umask 077
readonly CACHE_ROOT="/tmp/semantic-action-direct-prepare-j${SLURM_JOB_ID}-s${SLURM_STEP_ID}-$1"
mkdir "$CACHE_ROOT"
mkdir "$CACHE_ROOT/tmp" "$CACHE_ROOT/xdg" "$CACHE_ROOT/pycache"
export TMPDIR="$CACHE_ROOT/tmp"
export TMP="$CACHE_ROOT/tmp"
export TEMP="$CACHE_ROOT/tmp"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=8
export PYTHONPATH="$RELEASE"

exec "$PY" "$CODE" prepare \
  --feature-root "$FEATURE_ROOT" \
  --expected-feature-receipt-sha256 "$FEATURE_RECEIPT_SHA" \
  --output "$OUTPUT" \
  --seed 20260819 \
  --pca-dim 64 \
  --latent-dim 32 \
  --hidden-dim 256 \
  --steps "$STEPS" \
  --batch-size 128 \
  --learning-rate 0.002 \
  --beta-kl 0.02 \
  --prior-samples 8
