#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 RELEASE PYTHON FEATURE_ROOT FEATURE_RECEIPT_SHA V4A_RECEIPT OUTPUT" >&2
  exit 2
fi

v4b_release=$1
v4b_python=$2
v4b_feature_root=$3
v4b_feature_receipt_sha=$4
v4b_v4a_receipt=$5
v4b_output=$6
v4b_runtime=$v4b_release/methods/bernini_action_editing/semantic_anchor_temporal_convae_v4b_fast.py
v4b_test=$v4b_release/methods/bernini_action_editing/tests/test_semantic_anchor_temporal_convae_v4b_fast.py
v4a_runtime=$v4b_release/methods/bernini_action_editing/semantic_anchor_linear_frontier_v4_fast.py
v2_runtime=$v4b_release/methods/bernini_action_editing/semantic_anchor_action_sequence_vae_v2.py
feature_runtime=$v4b_release/methods/bernini_action_editing/semantic_action_cvae_canary_v1.py

check_file() {
  local path=$1
  local expected=$2
  [[ -f $path ]]
  [[ $(stat -c '%a:%h' "$path") == 444:1 ]]
  [[ $(sha256sum "$path" | awk '{print $1}') == "$expected" ]]
}

check_release() {
  check_file "$v4b_runtime" 6afa9fc39f993cedcb7ef672ca1297412ab95f5fdacbaf33a431fb49ef586ac4
  check_file "$v4b_test" 494c980fbdf8a86ab4c6e57a3c57371d3daf82fb15073478254ea2bf9e7e382d
  check_file "$v4a_runtime" e7e755a430b79c34fdc86f5fceaba8a9f69c66dd1e66b47c8f4115eac5265973
  check_file "$v2_runtime" 46927772a1861354ad5edeb2072ae9b1b505d235de7c2615fb11a6648f2bddca
  check_file "$feature_runtime" 74fe161f08eb92746ce22f00b50cca65f85ef05e4a4a6a5d0c7fb85867e94233
}

check_release
[[ $v4b_feature_receipt_sha == 8ff8f5fd5be36cb67ce40d5558a4406bdf70cbe9b72b0c43c71fa3abe8f6ad9c ]]
[[ -d $v4b_feature_root ]]
[[ -f $v4b_v4a_receipt ]]
[[ $(sha256sum "$v4b_v4a_receipt" | awk '{print $1}') == 568ef85d9812bcc2a771952e1806392c80f8248f5597dd32e4c95e7e1f5a3fa2 ]]
[[ $v4b_output = /* ]]
[[ $v4b_output == *.json ]]
[[ -d $(dirname "$v4b_output") ]]
[[ ! -e $v4b_output ]]
for v4b_fold in 0 1 2 3 4; do
  [[ ! -e ${v4b_output%.json}.selected_fold${v4b_fold}.pt ]]
done

v4b_cache=/tmp/anchor-v4b-exact5-${SLURM_JOB_ID:?}-${SLURM_STEP_ID:?}
[[ ! -e $v4b_cache ]]
mkdir -m 0700 "$v4b_cache"
export TMPDIR=$v4b_cache/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export XDG_CACHE_HOME=$v4b_cache/xdg
export PYTHONPYCACHEPREFIX=$v4b_cache/pycache
export MIOPEN_USER_DB_PATH=$v4b_cache/miopen-db
export MIOPEN_CUSTOM_CACHE_DIR=$v4b_cache/miopen-custom
export TRITON_CACHE_DIR=$v4b_cache/triton
export TORCHINDUCTOR_CACHE_DIR=$v4b_cache/inductor
export TORCH_EXTENSIONS_DIR=$v4b_cache/extensions
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$PYTHONPYCACHEPREFIX" \
  "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
export PYTHONPATH=$v4b_release
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONSAFEPATH=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=8

v4b_layout=
if [[ ${SLURM_NTASKS:-} == 1 \
      && ${SLURM_NNODES:-} == 1 \
      && ${SLURM_PROCID:-} == 0 \
      && ${SLURM_LOCALID:-} == 0 \
      && ${SLURM_GPUS_ON_NODE:-} == 1 ]]; then
  v4b_layout=exact1
elif [[ ${SLURM_NTASKS:-} == 5 \
        && ${SLURM_NNODES:-} == 1 \
        && ${SLURM_PROCID:-} == 4 \
        && ${SLURM_LOCALID:-} == 4 \
        && ${SLURM_GPUS_ON_NODE:-} == 5 ]]; then
  v4b_layout=placeholder5_rank4_scientific_exact1
else
  echo "unexpected Slurm scientific-process layout" >&2
  exit 1
fi
[[ ${SLURM_STEP_GPUS:-} =~ ^[0-9]+$ ]]
[[ ${ROCR_VISIBLE_DEVICES:-} == 0 ]]
echo "V4B_SLURM_LAYOUT=$v4b_layout" >&2
cd "$v4b_release"
"$v4b_python" -P -B - <<'PY'
import json
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("expected exactly one visible GPU")
name = torch.cuda.get_device_name(0)
if name != "AMD Instinct MI210" or not torch.version.hip:
    raise SystemExit(f"unexpected GPU/runtime: {name!r}, hip={torch.version.hip!r}")
props = torch.cuda.get_device_properties(0)
print(json.dumps({
    "gpu_gate": "PASS",
    "name": name,
    "logical_device_count": torch.cuda.device_count(),
    "uuid": str(props.uuid),
    "hip": torch.version.hip,
}, sort_keys=True), flush=True)
PY

"$v4b_python" -P -B - <<'PY'
import torch
from methods.bernini_action_editing import semantic_anchor_linear_frontier_v4_fast as a
from methods.bernini_action_editing import semantic_anchor_temporal_convae_v4b_fast as b

device = torch.device("cuda:0")
b._seed_everything(20260820, device)
generator = torch.Generator().manual_seed(20260820)
temporal = torch.linalg.qr(torch.randn(32, 4, generator=generator)).Q
content = torch.linalg.qr(torch.randn(768, 96, generator=generator)).Q
fit = a.FrontierFit(
    frame_mean=torch.randn(1, 768, generator=generator) * 0.01,
    frame_basis=content[:, :32],
    clip_mean=torch.empty(0),
    clip_basis=torch.empty(0),
    temporal_basis=temporal,
    content_basis=content,
    fit_iid_digest="gpu-smoke",
    fit_input_sha256="gpu-smoke",
    diagnostics={},
)
model = b.TuckerInitializedTemporalConvAE(fit, torch.ones(1)).to(device)
value = torch.randn(4, 32, 768, generator=generator).to(device)
value = value - value.mean(dim=1, keepdim=True)
reference = b._analytic_tucker_decode(value, fit)
actual = model(value)
if not torch.equal(actual, reference):
    raise SystemExit(f"step0 mismatch: {float((actual-reference).abs().max())}")
optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
warped = b._training_warp_batch(value)
loss, _ = b._fixed_training_loss(model(value), value, model(warped), warped)
optimizer.zero_grad(set_to_none=True)
loss.backward()
if not torch.isfinite(loss) or any(
    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    for parameter in model.parameters()
):
    raise SystemExit("non-finite GPU smoke")
optimizer.step()
print(f"V4B_GPU_SMOKE_PASS loss={float(loss.detach().cpu()):.12g}", flush=True)
PY

check_release
"$v4b_python" -P -B "$v4b_runtime" run-exact5 \
  --feature-root "$v4b_feature_root" \
  --expected-feature-receipt-sha256 "$v4b_feature_receipt_sha" \
  --v4a-receipt "$v4b_v4a_receipt" \
  --device cuda:0 \
  --output "$v4b_output"
check_release
[[ $(stat -c '%a:%h' "$v4b_output") == 444:1 ]]
for v4b_fold in 0 1 2 3 4; do
  v4b_checkpoint=${v4b_output%.json}.selected_fold${v4b_fold}.pt
  [[ $(stat -c '%a:%h' "$v4b_checkpoint") == 444:1 ]]
done
echo V4B_GPU_EXACT5_PASS
