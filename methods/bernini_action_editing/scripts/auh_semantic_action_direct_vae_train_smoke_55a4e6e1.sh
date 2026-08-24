#!/usr/bin/env bash
set -euo pipefail

readonly PY=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly BASE=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/semantic_action_direct_vae_v1_20260819
readonly RELEASE="$BASE/release_55a4e6e1_candidate1"
readonly CODE="$RELEASE/methods/bernini_action_editing/semantic_action_direct_vae_canary_v1.py"
readonly COMMON="$RELEASE/methods/bernini_action_editing/semantic_action_cvae_canary_v1.py"
readonly TEST="$RELEASE/methods/bernini_action_editing/tests/test_semantic_action_direct_vae_canary_v1.py"
readonly PREPARE_ROOT="$BASE/runs/smoke_prepare_55a4e6e1_v1"
readonly PREPARE_RECEIPT="$PREPARE_ROOT/prepare_receipt.json"
readonly TRAIN_BUNDLE="$PREPARE_ROOT/train_bundle.pt"
readonly CODE_SHA=55a4e6e10d4dfa77075e508bd921aef3916be0537fe9e8049c288ca22a7faefb
readonly COMMON_SHA=74fe161f08eb92746ce22f00b50cca65f85ef05e4a4a6a5d0c7fb85867e94233
readonly TEST_SHA=633ccd6ae2e8ef2d072ff88710878e58e0e0b3c8f02d7d727b12f2488e3e2e70
readonly PREPARE_RECEIPT_SHA=5212cec05d0c2bd928dc92f3cfe402a94a6bc8507569532277cf8c9a5ccc791e
readonly TRAIN_BUNDLE_SHA=b498edc5c9ab672a7133e45ee54ec370898e293e96c6e2c9831cba56e5ffdf54

if [[ $# -ne 1 ]]; then
  echo "usage: $0 deterministic_ae|direct_beta_vae" >&2
  exit 64
fi
readonly ARM="$1"
case "$ARM" in
  direct_beta_vae)
    readonly EXPECTED_JOB=143812
    readonly EXPECTED_HOST=auh7-1b-gpu-293
    readonly OUTPUT="$BASE/runs/smoke_direct_beta_vae_55a4e6e1_v1"
    ;;
  deterministic_ae)
    readonly EXPECTED_JOB=143811
    readonly EXPECTED_HOST=auh7-1b-gpu-306
    readonly OUTPUT="$BASE/runs/smoke_deterministic_ae_55a4e6e1_v1"
    ;;
  *)
    echo "arm differs" >&2
    exit 64
    ;;
esac

[[ "${SLURM_JOB_ID:-}" == "$EXPECTED_JOB" ]]
[[ -n "${SLURM_STEP_ID:-}" ]]
[[ "$(hostname -s)" == "$EXPECTED_HOST" ]]
[[ ! -e "$OUTPUT" ]]
[[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]
[[ "$(sha256sum "$CODE" | cut -d' ' -f1)" == "$CODE_SHA" ]]
[[ "$(sha256sum "$COMMON" | cut -d' ' -f1)" == "$COMMON_SHA" ]]
[[ "$(sha256sum "$TEST" | cut -d' ' -f1)" == "$TEST_SHA" ]]
[[ "$(sha256sum "$PREPARE_RECEIPT" | cut -d' ' -f1)" == "$PREPARE_RECEIPT_SHA" ]]
[[ "$(sha256sum "$TRAIN_BUNDLE" | cut -d' ' -f1)" == "$TRAIN_BUNDLE_SHA" ]]
for artifact in "$CODE" "$COMMON" "$TEST" "$PREPARE_RECEIPT" "$TRAIN_BUNDLE"; do
  [[ "$(stat -c '%a:%h' "$artifact")" == 444:1 ]]
done

umask 077
readonly CACHE_ROOT="/tmp/semantic-action-direct-$ARM-j${SLURM_JOB_ID}-s${SLURM_STEP_ID}"
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

"$PY" -m unittest \
  methods.bernini_action_editing.tests.test_semantic_action_direct_vae_canary_v1
"$PY" -O -m unittest \
  methods.bernini_action_editing.tests.test_semantic_action_direct_vae_canary_v1
"$PY" -c 'import torch; assert torch.cuda.device_count() == 1; assert torch.cuda.get_device_name(0) == "AMD Instinct MI210"'

exec "$PY" "$CODE" train \
  --arm "$ARM" \
  --prepare-receipt "$PREPARE_RECEIPT" \
  --expected-prepare-receipt-sha256 "$PREPARE_RECEIPT_SHA" \
  --train-bundle "$TRAIN_BUNDLE" \
  --expected-train-bundle-sha256 "$TRAIN_BUNDLE_SHA" \
  --output "$OUTPUT" \
  --device cuda:0
