#!/usr/bin/env bash
set -euo pipefail

readonly PY=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly BASE=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/semantic_action_direct_vae_v1_20260819
readonly RELEASE="$BASE/release_55a4e6e1_candidate1"
readonly CODE="$RELEASE/methods/bernini_action_editing/semantic_action_direct_vae_canary_v1.py"
readonly COMMON="$RELEASE/methods/bernini_action_editing/semantic_action_cvae_canary_v1.py"
readonly TEST="$RELEASE/methods/bernini_action_editing/tests/test_semantic_action_direct_vae_canary_v1.py"
readonly PREPARE_ROOT="$BASE/runs/full_prepare_55a4e6e1_v1"
readonly PREPARE_RECEIPT="$PREPARE_ROOT/prepare_receipt.json"
readonly HELD_BUNDLE="$PREPARE_ROOT/held_eval_bundle.pt"
readonly AE_RECEIPT="$BASE/runs/full_deterministic_ae_55a4e6e1_v1/receipt.json"
readonly VAE_RECEIPT="$BASE/runs/full_direct_beta_vae_55a4e6e1_v1/receipt.json"
readonly OUTPUT="$BASE/runs/full_final_comparison_55a4e6e1_v1"
readonly CODE_SHA=55a4e6e10d4dfa77075e508bd921aef3916be0537fe9e8049c288ca22a7faefb
readonly COMMON_SHA=74fe161f08eb92746ce22f00b50cca65f85ef05e4a4a6a5d0c7fb85867e94233
readonly TEST_SHA=633ccd6ae2e8ef2d072ff88710878e58e0e0b3c8f02d7d727b12f2488e3e2e70
readonly PREPARE_RECEIPT_SHA=dfeb07ecb579aadd3e2aae54df1dc68bb07572285e5e8bdcd4249b2a83747eb0
readonly HELD_BUNDLE_SHA=32dd75586f08c12fb9ce1b2961876f016844af0ab4542e333346a4c9768d4487
readonly AE_RECEIPT_SHA=8a31ae13bb1295f5a650ad8382c89901274513586fd8e7dd1b35bd1f618f7866
readonly VAE_RECEIPT_SHA=035d38240325efe4d1593e7b0927a9db50f6b99c683e7d18350965b8910615c3

[[ $# -eq 0 ]]
[[ "${SLURM_JOB_ID:-}" == 143812 ]]
[[ -n "${SLURM_STEP_ID:-}" ]]
[[ "$(hostname -s)" == auh7-1b-gpu-293 ]]
[[ ! -e "$OUTPUT" ]]
[[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]
[[ "$(sha256sum "$CODE" | cut -d' ' -f1)" == "$CODE_SHA" ]]
[[ "$(sha256sum "$COMMON" | cut -d' ' -f1)" == "$COMMON_SHA" ]]
[[ "$(sha256sum "$TEST" | cut -d' ' -f1)" == "$TEST_SHA" ]]
[[ "$(sha256sum "$PREPARE_RECEIPT" | cut -d' ' -f1)" == "$PREPARE_RECEIPT_SHA" ]]
[[ "$(sha256sum "$HELD_BUNDLE" | cut -d' ' -f1)" == "$HELD_BUNDLE_SHA" ]]
[[ "$(sha256sum "$AE_RECEIPT" | cut -d' ' -f1)" == "$AE_RECEIPT_SHA" ]]
[[ "$(sha256sum "$VAE_RECEIPT" | cut -d' ' -f1)" == "$VAE_RECEIPT_SHA" ]]
for artifact in "$CODE" "$COMMON" "$TEST" "$PREPARE_RECEIPT" \
  "$HELD_BUNDLE" "$AE_RECEIPT" "$VAE_RECEIPT"; do
  [[ "$(stat -c '%a:%h' "$artifact")" == 444:1 ]]
done

umask 077
readonly CACHE_ROOT="/tmp/semantic-action-direct-final-j${SLURM_JOB_ID}-s${SLURM_STEP_ID}"
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

exec "$PY" "$CODE" finalize \
  --prepare-receipt "$PREPARE_RECEIPT" \
  --expected-prepare-receipt-sha256 "$PREPARE_RECEIPT_SHA" \
  --held-eval-bundle "$HELD_BUNDLE" \
  --expected-held-eval-bundle-sha256 "$HELD_BUNDLE_SHA" \
  --deterministic-arm-receipt "$AE_RECEIPT" \
  --expected-deterministic-arm-receipt-sha256 "$AE_RECEIPT_SHA" \
  --direct-beta-vae-arm-receipt "$VAE_RECEIPT" \
  --expected-direct-beta-vae-arm-receipt-sha256 "$VAE_RECEIPT_SHA" \
  --output "$OUTPUT" \
  --device cuda:0
