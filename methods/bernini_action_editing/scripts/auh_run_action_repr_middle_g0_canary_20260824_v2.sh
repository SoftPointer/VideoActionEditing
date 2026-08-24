#!/usr/bin/env bash
set -euo pipefail

# Representation-only G0 canary.  This script intentionally has no trainer,
# optimizer, backward pass, checkpoint writer, or generator action route.
experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_repr_target_selfgen_middle_g1_20260824_v2
source_root="$experiment_root/source"
method_root="$source_root/methods/bernini_action_editing"
runtime_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/source-online-anchor-targetowned-qk-decode-v14r3-gradgeom-dfix2/methods/bernini_action_editing
output_root="$experiment_root/middle_repr_canary/0be6494dfac3/real_forward"
status_path="$experiment_root/MIDDLE_G0_CANARY_STATUS.json"

python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
video=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev_action_anchor_target_gap16_20260819_v2/controls/0be6494dfac3/target-forward.mp4

prereg="$source_root/stage1_v2_preregistration.json"
extractor="$method_root/materialize_decoded_middle_action_repr_v1.py"
expected_prereg_sha=294168e596212bd61e8d555e72702ceeeb993fb18c7fa7536a43d0b00ad592b3
expected_extractor_sha=e2dbf3de577a3f59732d58f2b80183a34bccd1a2e9314645d6819fa301a43831
expected_video_sha=f732dd4ffcfb272a0d9ab8b6c035df102f8e25f636b9120d0dc3f26bab3d51fc
expected_full30_sha=67275ae09e7cb7b1e7e8fc43ce2928031b3fe8aabe213e8626000f37abad4ead
expected_quotient_sha=a9bfec2816ec1b6ccb2a336ea25600f15f22557aea76b1ea0605bbeb737b501c
expected_preservation_sha=11bc0792174a60c2e449eb61ff8f81da97808e02ee2707b5c4f20ee2118f4b5c
expected_build_renderer_sha=afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5
expected_materialize_vae_sha=a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_sha() {
  local path="$1" expected="$2" actual
  test -f "$path"
  actual="$(sha256_file "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch: $path expected=$expected actual=$actual" >&2
    return 1
  fi
}

test "$(hostname -s)" = auh7-1b-gpu-213
test "${SLURM_JOB_ID:-}" = 147881
test -x "$python_bin"
test -d "$bernini_root"
test -d "$veomni_root"
test -d "$checkpoint"
test -d "$runtime_root"
test -f "$runtime_root/full30_action_learning_v1.py"
test ! -e "$output_root"
test ! -e "$status_path"
require_sha "$prereg" "$expected_prereg_sha"
require_sha "$extractor" "$expected_extractor_sha"
require_sha "$video" "$expected_video_sha"
require_sha "$runtime_root/full30_action_learning_v1.py" "$expected_full30_sha"
require_sha "$runtime_root/self_generated_action_quotient_v1.py" "$expected_quotient_sha"
require_sha "$runtime_root/self_generated_action_preservation_v2.py" "$expected_preservation_sha"
require_sha "$method_root/tools/build_renderer_dataset.py" "$expected_build_renderer_sha"
require_sha "$method_root/tools/materialize_vae.py" "$expected_materialize_vae_sha"

export PYTHONPATH="$method_root:$runtime_root"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf
export TORCH_HOME=/vast/users/guangyi.chen/.cache/torch
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=4
export MIOPEN_FIND_MODE=NORMAL
export MIOPEN_USER_DB_PATH="/tmp/action-repr-middle-v2-${SLURM_JOB_ID}-${SLURM_STEP_ID}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/action-repr-middle-v2-${SLURM_JOB_ID}-${SLURM_STEP_ID}/miopen-custom"
export TORCH_EXTENSIONS_DIR="/tmp/action-repr-middle-v2-${SLURM_JOB_ID}-${SLURM_STEP_ID}/torch-extensions"
export TRITON_CACHE_DIR="/tmp/action-repr-middle-v2-${SLURM_JOB_ID}-${SLURM_STEP_ID}/triton"
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"

"$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$extractor" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$checkpoint" \
  --video "$video" \
  --video-sha256 "$expected_video_sha" \
  --input-role real_forward \
  --case-id 0be6494dfac3 \
  --instruction "Edit the action so that the cockatoo turns its head towards frame left, raising its curved yellow crest." \
  --seed 2026081908 \
  --output "$output_root"

test -s "$output_root/middle_repr.safetensors"
test -s "$output_root/receipt.json"
"$python_bin" -B - "$output_root/receipt.json" "$status_path" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

receipt_path = Path(sys.argv[1])
status_path = Path(sys.argv[2])
receipt = json.loads(receipt_path.read_text())
training = receipt.get("training_authority", {})
if training.get("optimization_steps") != 0:
    raise SystemExit("extractor receipt contains parameter updates")
if training.get("optimizer_created") is not False:
    raise SystemExit("extractor receipt does not close optimizer absence")
if training.get("generator_parameters_updated") is not False:
    raise SystemExit("extractor receipt contains generator parameter updates")
status = {
    "schema_version": "bernini-action-repr-middle-g0-canary-status-v1",
    "complete": True,
    "experiment_id": "action_repr_target_selfgen_middle_g1_20260824_v2",
    "case_id": "0be6494dfac3",
    "anchor_kind": "target",
    "input_role": "real_forward",
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_step_id": os.environ["SLURM_STEP_ID"],
    "optimization_steps": 0,
    "optimizer_created": False,
    "parameter_updates": False,
    "representation_receipt": str(receipt_path),
    "representation_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    "claim_boundary": "real_WORLD4_middle_representation_canary_not_G1_not_trained_ours"
}
status["status_digest"] = hashlib.sha256(
    json.dumps(status, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
).hexdigest()
status_path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=status_path.name + ".", suffix=".tmp", dir=status_path.parent)
try:
    with os.fdopen(fd, "w", encoding="ascii") as handle:
        json.dump(status, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, status_path)
finally:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
print(json.dumps(status, sort_keys=True))
PY
