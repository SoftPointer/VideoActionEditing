#!/usr/bin/env bash
set -euo pipefail

# R6 bridge: old-QK white-leakage diagnostic only, never a preservation fix.
if [ "$#" -ne 0 ]; then echo "usage: $0" >&2; exit 2; fi
: "${E00_R6_BOOTSTRAP_VERIFIED:?missing external bootstrap proof}"
: "${E00_R6_PACKAGE_ROOT_VERIFIED:?missing verified package root}"
: "${E00_R6_ROOT_PATH_VERIFIED:?missing verified root path}"
: "${E00_R6_EXPECTED_ROOT_SHA256:?missing verified root SHA}"
: "${E00_R6_ARM_ROLE:?missing E00_R6_ARM_ROLE}"
: "${E00_R6_PHASE:?missing E00_R6_PHASE}"
: "${E00_R6_OUTPUT_ROOT:?missing E00_R6_OUTPUT_ROOT}"
: "${E00_R6_PACKAGE_MANIFEST:?missing E00_R6_PACKAGE_MANIFEST}"
: "${E00_R6_AUTHORIZATION:?missing E00_R6_AUTHORIZATION}"
: "${E00_R6_CAPABILITY_TOKEN:?missing E00_R6_CAPABILITY_TOKEN}"
test "$E00_R6_BOOTSTRAP_VERIFIED" = 1
fixed_parent_job=143808
fixed_compute_node=auh7-1b-gpu-292
test "$(hostname -s)" = "$fixed_compute_node"
test "${SLURM_JOB_ID:-}" = "$fixed_parent_job"
# The external bootstrap executes this bridge from its already verified open
# descriptor, so $0 is /proc/self/fd/<n>, not a package path.  Resolve every
# dependency from the package root that the bootstrap verified instead.
package_root="$(cd -- "$E00_R6_PACKAGE_ROOT_VERIFIED" && pwd -P)"
test "$package_root" = "$E00_R6_PACKAGE_ROOT_VERIFIED"
method_root="$package_root/methods/bernini_action_editing"
test -d "$method_root"; test ! -L "$method_root"
root="$method_root/assets/e00_three_vessel_clean_diag_r6_EXTERNAL_ROOT.json"
test "$root" = "$E00_R6_ROOT_PATH_VERIFIED"
manifest="$package_root/e00-clean-diagnostic-r6-package.manifest.json"
test "$manifest" = "$E00_R6_PACKAGE_MANIFEST"
protocol="$method_root/assets/e00_three_vessel_clean_diag_r6_protocol_20260821.json"
wrapper="$method_root/e00_legacy_infer_fixed_rng_wrapper_r6.py"
validator="$method_root/validate_e00_three_vessel_clean_diag_r6.py"
builder="$method_root/tools/build_e00_three_vessel_clean_diag_r6_package.py"
infer="$method_root/infer_anchor_sga_anc_event_v1.py"
controller="$method_root/anchor_sga_anc_controller.py"
qk_transport="$method_root/anchor_qk_transport.py"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256

for command_name in sha256sum jq ffprobe find; do command -v "$command_name" >/dev/null; done
for file in "$root" "$manifest" "$protocol" "$wrapper" "$validator" "$builder" "$infer" "$controller" "$qk_transport" "$E00_R6_AUTHORIZATION"; do test -f "$file"; test ! -L "$file"; done
test -x "$python_bin"
check_cache_free() {
  local cache_dir cache_file
  cache_dir="$(find "$package_root" -type d -name __pycache__ -print -quit)"
  cache_file="$(find "$package_root" -type f -name '*.pyc' -print -quit)"
  test -z "$cache_dir"; test -z "$cache_file"
}
check_cache_free
"$python_bin" -B "$builder" verify --package-root "$package_root" --manifest "$manifest" \
  --expected-root-sha256 "$E00_R6_EXPECTED_ROOT_SHA256" >/dev/null
"$python_bin" -B "$validator" protocol --protocol "$protocol" >/dev/null

token_sha="$(printf '%s' "$E00_R6_CAPABILITY_TOKEN" | sha256sum)"; token_sha="${token_sha%% *}"
capability_args=(
  bridge-capability --protocol "$protocol" --phase "$E00_R6_PHASE"
  --arm-role "$E00_R6_ARM_ROLE" --package-manifest "$manifest"
  --authorization "$E00_R6_AUTHORIZATION" --capability-token-sha256 "$token_sha"
)
case "$E00_R6_PHASE" in
  A)
    test "$E00_R6_ARM_ROLE" = pure_noobserver_output_routeoff
    test -z "${E00_R6_PHASE_A_MARKER:-}"; test -z "${E00_R6_AB_GATE:-}"
    ;;
  B)
    test "$E00_R6_ARM_ROLE" = observer_matched_output_routeoff
    : "${E00_R6_PHASE_A_MARKER:?phase B requires E00_R6_PHASE_A_MARKER}"
    test -z "${E00_R6_AB_GATE:-}"
    capability_args+=(--phase-a-marker "$E00_R6_PHASE_A_MARKER")
    ;;
  C)
    test "$E00_R6_ARM_ROLE" = old_pureqk_temporal_routeon
    : "${E00_R6_PHASE_A_MARKER:?phase C requires E00_R6_PHASE_A_MARKER}"
    : "${E00_R6_AB_GATE:?phase C requires E00_R6_AB_GATE}"
    capability_args+=(--phase-a-marker "$E00_R6_PHASE_A_MARKER" --ab-gate "$E00_R6_AB_GATE")
    ;;
  *) echo "unknown R6 phase" >&2; exit 2 ;;
esac
"$python_bin" -B "$validator" "${capability_args[@]}" >/dev/null
check_cache_free

base_spec_relative="$(jq -er '.base_diagnostic_spec.package_relative_path' "$protocol")"
case "$base_spec_relative" in /*|*../*) exit 2 ;; esac
spec="$package_root/$base_spec_relative"
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
label="$(jq -er --arg role "$E00_R6_ARM_ROLE" '.arms[]|select(.arm_role==$role)|.label' "$spec")"
transport="$(jq -er --arg role "$E00_R6_ARM_ROLE" '.arms[]|select(.arm_role==$role)|.legacy_transport' "$spec")"
transport_steps="$(jq -er --arg role "$E00_R6_ARM_ROLE" '.arms[]|select(.arm_role==$role)|.legacy_transport_steps' "$spec")"
if [ "$E00_R6_PHASE" = A ]; then anchor_video="$package_root/$placeholder_relative"; anchor_sha="$placeholder_sha"; else anchor_video="$action_anchor_video"; anchor_sha="$action_anchor_sha"; fi

verify_sha() { local file="$1" expected="$2" actual; actual="$(sha256sum -- "$file")"; actual="${actual%% *}"; test "$actual" = "$expected"; }
verify_sha "$source_video" "$source_sha"; verify_sha "$anchor_video" "$anchor_sha"
output="$E00_R6_OUTPUT_ROOT/$label.mp4"
native_receipt="$output.receipt.json"; rng_prefix="$output.fixed-rng-r6"; audit="$output.e00-r6-arm-audit.json"
for path in "$output" "$native_receipt" "$audit" "$rng_prefix.rank0.json" "$rng_prefix.rank1.json" "$rng_prefix.rank2.json" "$rng_prefix.rank3.json"; do test ! -e "$path"; test ! -L "$path"; done
mkdir -p "$E00_R6_OUTPUT_ROOT"; test ! -L "$E00_R6_OUTPUT_ROOT"
scratch="/tmp/e00-clean-diag-r6-${E00_R6_PHASE}-${SLURM_STEP_ID:-nostep}"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom" TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
export PYTHONPATH="$method_root"

"$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$wrapper" --wrapper-arm-role "$E00_R6_ARM_ROLE" --wrapper-audit-prefix "$rng_prefix" --wrapper-protocol "$protocol" --legacy-entrypoint infer_anchor_sga_anc_event_v1 -- \
  --bernini-root "$bernini_root" --veomni-root "$veomni_root" --checkpoint "$checkpoint" --checkpoint-content-manifest "$checkpoint_manifest" \
  --source-video "$source_video" --expected-source-sha256 "$source_sha" --anchor-video "$anchor_video" --expected-anchor-sha256 "$anchor_sha" \
  --instruction "$instruction" --source-caption "$source_caption" --target-caption "$target_caption" --anchor-caption "$anchor_caption" --anchor-noop-caption "$anchor_noop_caption" \
  --arm AQK_IID1 --transport "$transport" --transport-strength 1.0 --transport-steps "$transport_steps" \
  --blocks 1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26,27,29 --field-guidance raw_cfg --field-model first_phase_caption_i2v \
  --source-cfg-scale 4.5 --target-cfg-scale 4.5 --sga-temperature 0.01 --early-candidate-count 5 --initial-noise-proposal-mode keyed_only \
  --anchor-state-mode clean_noised --anchor-cfg-scope shared --anchor-contrast-mode dynamic_static_same_caption --anchor-sigma-cap 1.0 \
  --preservation-mode none --preservation-keep-fraction 0.20 --preservation-outside-scale 0.0 --preservation-dilation 1 --preservation-residual-fraction 0.0 \
  --preservation-object-identity-strength 0.0 --preservation-start-step 0 --preservation-ramp-steps 1 --sga-score-mode global_source_cosine \
  --anchor-candidate-mode single_shared --anchor-spatial-alignment none --event01-forced-role-proposal-index -1 --output "$output"

for path in "$output" "$native_receipt" "$rng_prefix.rank0.json" "$rng_prefix.rank1.json" "$rng_prefix.rank2.json" "$rng_prefix.rank3.json"; do test -f "$path"; test ! -L "$path"; done
"$python_bin" -B "$validator" arm --protocol "$protocol" --arm-role "$E00_R6_ARM_ROLE" --native-receipt "$native_receipt" --video "$output" \
  --rng-receipt "$rng_prefix.rank0.json" --rng-receipt "$rng_prefix.rank1.json" --rng-receipt "$rng_prefix.rank2.json" --rng-receipt "$rng_prefix.rank3.json" --audit-output "$audit"
check_cache_free
printf 'E00_CLEAN_DIAG_R6_ARM_COMPLETE %s %s %s\n' "$E00_R6_ARM_ROLE" "$output" "$audit"
