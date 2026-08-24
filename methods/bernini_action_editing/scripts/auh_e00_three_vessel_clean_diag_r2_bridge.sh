#!/usr/bin/env bash
set -euo pipefail

# Compute-node payload for exactly one R2 diagnostic arm.  The login-host
# phase launchers own the A -> stop -> B -> bit-exact gate -> C state machine.
# This bridge performs zero-update frozen inference only.
if [ "$#" -ne 0 ]; then echo "usage: $0" >&2; exit 2; fi
: "${E00_R2_ARM_ROLE:?missing E00_R2_ARM_ROLE}"
: "${E00_R2_OUTPUT_ROOT:?missing E00_R2_OUTPUT_ROOT}"
: "${E00_R2_PACKAGE_MANIFEST:?missing E00_R2_PACKAGE_MANIFEST}"
: "${EXPECTED_COMPUTE_NODE:?missing EXPECTED_COMPUTE_NODE}"
: "${EXPECTED_PARENT_JOB:?missing EXPECTED_PARENT_JOB}"
test "$(hostname -s)" = "$EXPECTED_COMPUTE_NODE"
test "${SLURM_JOB_ID:-}" = "$EXPECTED_PARENT_JOB"
test "$EXPECTED_COMPUTE_NODE" = auh7-1b-gpu-292
case "$E00_R2_ARM_ROLE" in
  pure_noobserver_output_routeoff|observer_matched_output_routeoff|old_pureqk_temporal_routeon) ;;
  *) echo "unknown E00 R2 arm" >&2; exit 2 ;;
esac

script_dir="$(cd -- "$(dirname -- "$0")" && pwd -P)"
method_root="$(cd -- "$script_dir/.." && pwd -P)"
package_root="$(cd -- "$method_root/../.." && pwd -P)"
protocol="$method_root/assets/e00_three_vessel_fresh_keyed_two_phase_diagnostic_v2.json"
wrapper="$method_root/e00_legacy_infer_fixed_rng_wrapper_v2.py"
validator="$method_root/validate_e00_three_vessel_fresh_keyed_two_phase_diagnostic_v2.py"
package_builder="$method_root/tools/build_e00_three_vessel_clean_diag_r2_package_v2.py"
infer="$method_root/infer_anchor_sga_anc_event_v1.py"
controller="$method_root/anchor_sga_anc_controller.py"
qk_transport="$method_root/anchor_qk_transport.py"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256

for command_name in sha256sum jq ffprobe; do command -v "$command_name" >/dev/null; done
for file in "$protocol" "$wrapper" "$validator" "$package_builder" "$infer" "$controller" "$qk_transport" "$E00_R2_PACKAGE_MANIFEST"; do
  test -f "$file"; test ! -L "$file"
done
test -x "$python_bin"

verify_sha() {
  local file="$1" expected="$2" actual
  actual="$(sha256sum -- "$file")"; actual="${actual%% *}"
  test "$actual" = "$expected"
}
verify_sha "$controller" 1427a4908e0a4239e95a353d3406c41cb77fdb7f0be81727126a2cfd23f1f3ad
verify_sha "$qk_transport" 37941e30853b16fa242a7c91940620069f87a1a975d2ecf610f3cde800557a99
verify_sha "$infer" dd3558a4c38c5541ba6b7ad455ac599f43eb48b1b56f207a07776c9e1819145f
"$python_bin" -B "$package_builder" verify --package-root "$package_root" --manifest "$E00_R2_PACKAGE_MANIFEST" >/dev/null
"$python_bin" -B "$validator" protocol --protocol "$protocol" >/dev/null

base_spec_relative="$(jq -er '.base_diagnostic_spec.package_relative_path' "$protocol")"
case "$base_spec_relative" in /*|*../*) exit 2 ;; esac
spec="$package_root/$base_spec_relative"
test -f "$spec"; test ! -L "$spec"
source_video="$(jq -er '.data_and_prompt_contract.source_video' "$spec")"
source_sha="$(jq -er '.data_and_prompt_contract.source_video_sha256' "$spec")"
action_anchor_video="$(jq -er '.data_and_prompt_contract.anchor_video' "$spec")"
action_anchor_sha="$(jq -er '.data_and_prompt_contract.anchor_video_sha256' "$spec")"
placeholder_relative="$(jq -er '.data_and_prompt_contract.pure_noobserver_placeholder.package_relative_path' "$spec")"
placeholder_sha="$(jq -er '.data_and_prompt_contract.pure_noobserver_placeholder.sha256' "$spec")"
instruction="$(jq -er '.data_and_prompt_contract.editing_instruction' "$spec")"
source_caption="$(jq -er '.data_and_prompt_contract.source_caption' "$spec")"
target_caption="$(jq -er '.data_and_prompt_contract.target_caption' "$spec")"
anchor_caption="$(jq -er '.data_and_prompt_contract.anchor_caption' "$spec")"
anchor_noop_caption="$(jq -er '.data_and_prompt_contract.anchor_noop_caption' "$spec")"
label="$(jq -er --arg role "$E00_R2_ARM_ROLE" '.arms[] | select(.arm_role == $role) | .label' "$spec")"
transport="$(jq -er --arg role "$E00_R2_ARM_ROLE" '.arms[] | select(.arm_role == $role) | .legacy_transport' "$spec")"
transport_steps="$(jq -er --arg role "$E00_R2_ARM_ROLE" '.arms[] | select(.arm_role == $role) | .legacy_transport_steps' "$spec")"
if [ "$E00_R2_ARM_ROLE" = pure_noobserver_output_routeoff ]; then
  anchor_video="$package_root/$placeholder_relative"
  anchor_sha="$placeholder_sha"
else
  anchor_video="$action_anchor_video"
  anchor_sha="$action_anchor_sha"
fi
test -n "$label"; test -n "$transport"; case "$transport_steps" in 0|40) ;; *) exit 2 ;; esac
verify_sha "$source_video" "$source_sha"
verify_sha "$anchor_video" "$anchor_sha"

output="$E00_R2_OUTPUT_ROOT/$label.mp4"
native_receipt="$output.receipt.json"
rng_prefix="$output.fixed-rng-r2"
audit="$output.e00-r2-arm-audit.json"
for path in "$output" "$native_receipt" "$audit" "$rng_prefix.rank0.json" "$rng_prefix.rank1.json" "$rng_prefix.rank2.json" "$rng_prefix.rank3.json"; do
  test ! -e "$path"; test ! -L "$path"
done
mkdir -p "$E00_R2_OUTPUT_ROOT"
test ! -L "$E00_R2_OUTPUT_ROOT"

scratch="/tmp/e00-clean-diag-r2-${E00_R2_ARM_ROLE}-${SLURM_STEP_ID:-nostep}"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Deliberately preserve the exact Slurm visible-device namespace.  Do not set,
# unset, reorder, or reinterpret CUDA_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES.
export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
export PYTHONPATH="$method_root"

"$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$wrapper" \
  --wrapper-arm-role "$E00_R2_ARM_ROLE" \
  --wrapper-audit-prefix "$rng_prefix" \
  --wrapper-protocol "$protocol" \
  --legacy-entrypoint infer_anchor_sga_anc_event_v1 \
  -- \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$checkpoint" \
  --checkpoint-content-manifest "$checkpoint_manifest" \
  --source-video "$source_video" \
  --expected-source-sha256 "$source_sha" \
  --anchor-video "$anchor_video" \
  --expected-anchor-sha256 "$anchor_sha" \
  --instruction "$instruction" \
  --source-caption "$source_caption" \
  --target-caption "$target_caption" \
  --anchor-caption "$anchor_caption" \
  --anchor-noop-caption "$anchor_noop_caption" \
  --arm AQK_IID1 \
  --transport "$transport" \
  --transport-strength 1.0 \
  --transport-steps "$transport_steps" \
  --blocks 1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26,27,29 \
  --field-guidance raw_cfg \
  --field-model first_phase_caption_i2v \
  --source-cfg-scale 4.5 \
  --target-cfg-scale 4.5 \
  --sga-temperature 0.01 \
  --early-candidate-count 5 \
  --initial-noise-proposal-mode keyed_only \
  --anchor-state-mode clean_noised \
  --anchor-cfg-scope shared \
  --anchor-contrast-mode dynamic_static_same_caption \
  --anchor-sigma-cap 1.0 \
  --preservation-mode none \
  --preservation-keep-fraction 0.20 \
  --preservation-outside-scale 0.0 \
  --preservation-dilation 1 \
  --preservation-residual-fraction 0.0 \
  --preservation-object-identity-strength 0.0 \
  --preservation-start-step 0 \
  --preservation-ramp-steps 1 \
  --sga-score-mode global_source_cosine \
  --anchor-candidate-mode single_shared \
  --anchor-spatial-alignment none \
  --event01-forced-role-proposal-index -1 \
  --output "$output"

for path in "$output" "$native_receipt" "$rng_prefix.rank0.json" "$rng_prefix.rank1.json" "$rng_prefix.rank2.json" "$rng_prefix.rank3.json"; do
  test -f "$path"; test ! -L "$path"
done
"$python_bin" -B "$validator" arm \
  --protocol "$protocol" --arm-role "$E00_R2_ARM_ROLE" \
  --native-receipt "$native_receipt" --video "$output" \
  --rng-receipt "$rng_prefix.rank0.json" \
  --rng-receipt "$rng_prefix.rank1.json" \
  --rng-receipt "$rng_prefix.rank2.json" \
  --rng-receipt "$rng_prefix.rank3.json" \
  --audit-output "$audit"
printf 'E00_CLEAN_DIAG_R2_ARM_COMPLETE %s %s %s\n' "$E00_R2_ARM_ROLE" "$output" "$audit"
