#!/usr/bin/env bash
set -euo pipefail

experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_repr_target_selfgen_20260824_v1
source_root="$experiment_root/source"
manifest="$source_root/experiment_manifest.json"
flow_root="$experiment_root/flows"
infer_root="${INFER_ROOT_OVERRIDE:-$experiment_root/inference}"
log_root="$experiment_root/logs"

history_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
sealed_source="$history_root/stage1/source-sga-anc-training-v1"
runtime_source="$history_root/stage1/source-be31323"
carrier_checkpoint="$history_root/stage1/complex8_sga_anc_training_v1/train_sgaanc_tau02_uniform25_gain10_all30_r256_micro2_s32_v2/checkpoint-00000032"

python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4

extractor="$sealed_source/methods/bernini_action_editing/extract_anchor_raft_flow_v1.py"
inference="$sealed_source/methods/bernini_action_editing/infer_same_video_dense_flow_adapter_v1.py"

expected_gap16_manifest_sha=d4ca564af7730754e4838222b11ad8c1452d8b08fe49bc605069b9e7969b0f78
expected_controls_manifest_sha=bb7d8214a7338bd60a3c9023ad4fb501a8682a241c98e77eba56ce245c1d93d9
expected_extractor_sha=4624f675d08e8cadd0ac03d46a39f34785561728354f6b39c29f8991cb1378df
expected_inference_sha=6b21f2844b065c9da5e2c5663f6c592f57764d15c002de2dcc9025696209e14d
expected_carrier_receipt_sha=1c957964802355fbd7e3634db172b52a3bcc2b6f530b6aefe4fd42cf8c28d003
expected_carrier_flow_sha=7cd933c8cb4cbbf02559ad87cfa1f52a9db643f3352050cc324537d831fd6b1a
expected_carrier_lora_sha=54b419c585e6d24b236ab68e9fc01a31f34d005103bc38cd4e800abaccee696f

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_sha() {
  local path="$1" expected="$2" actual
  test -f "$path"
  actual="$(sha256_file "$path")"
  if [ "$actual" != "$expected" ]; then
    echo "sha256 mismatch: $path expected=$expected actual=$actual" >&2
    return 1
  fi
}

manifest_value() {
  local iid="$1" expression="$2"
  jq -er --arg iid "$iid" ".cases[] | select(.case_id == \$iid) | $expression" "$manifest"
}

common_preflight() {
  test -x "$python_bin"
  test -f "$manifest"
  test -f "$extractor"
  test -f "$inference"
  test -d "$bernini_root"
  test -d "$veomni_root"
  test -d "$base_checkpoint"
  test -d "$carrier_checkpoint"
  require_sha "$extractor" "$expected_extractor_sha"
  require_sha "$inference" "$expected_inference_sha"
  require_sha "$carrier_checkpoint/receipt.json" "$expected_carrier_receipt_sha"
  require_sha "$carrier_checkpoint/adapter_model.safetensors" "$expected_carrier_flow_sha"
  require_sha "$carrier_checkpoint/adapter/adapter_model.safetensors" "$expected_carrier_lora_sha"
  require_sha "$(jq -er '.source_manifests.gap16.path' "$manifest")" "$expected_gap16_manifest_sha"
  require_sha "$(jq -er '.source_manifests.temporal_controls.path' "$manifest")" "$expected_controls_manifest_sha"
  test "$(jq -r '.cases | length' "$manifest")" = 8
  test "$(jq -r '.flow_roles | length' "$manifest")" = 4
  test "$(jq -r '.current_experiment_optimization_steps' "$manifest")" = 0
  test "$(jq -r '.historical_carrier_global_step' "$manifest")" = 32
}

media_preflight() {
  local iid role path expected
  while IFS= read -r iid; do
    for role in source real_forward temporal_shuffle reverse self_generated frozen; do
      path="$(manifest_value "$iid" ".$role.path")"
      expected="$(manifest_value "$iid" ".$role.sha256")"
      require_sha "$path" "$expected"
    done
  done < <(jq -r '.cases[].case_id' "$manifest")
}

extract_case() (
  set -euo pipefail
  local gpu="$1" iid="$2" role source anchor output scratch
  source="$(manifest_value "$iid" '.source.path')"
  export ROCR_VISIBLE_DEVICES="$gpu"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  export TORCH_HOME=/vast/users/guangyi.chen/.cache/torch
  export XDG_CACHE_HOME="/tmp/action-repr-flow-${iid}-g${gpu}-${SLURM_JOB_ID}-${SLURM_STEP_ID}/cache"
  export MIOPEN_USER_DB_PATH="/tmp/action-repr-flow-${iid}-g${gpu}-${SLURM_JOB_ID}-${SLURM_STEP_ID}/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="/tmp/action-repr-flow-${iid}-g${gpu}-${SLURM_JOB_ID}-${SLURM_STEP_ID}/miopen-custom"
  mkdir -p "$XDG_CACHE_HOME" "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR" "$flow_root/$iid" "$log_root/flow/$iid"
  for role in real_forward temporal_shuffle reverse self_generated; do
    anchor="$(manifest_value "$iid" ".$role.path")"
    output="$flow_root/$iid/${role}.safetensors"
    test ! -e "$output"
    test ! -e "${output%.safetensors}.json"
    "$python_bin" -B "$extractor" \
      --source "$source" \
      --anchor "$anchor" \
      --output "$output" \
      --latent-height 46 \
      --latent-width 82 \
      >"$log_root/flow/$iid/${role}.log" 2>&1
  done
)

run_extract() {
  test "$(hostname -s)" = auh7-1b-gpu-213
  test "${SLURM_JOB_ID:-}" = 147881
  common_preflight
  media_preflight
  mkdir -p "$flow_root" "$log_root/flow"
  test ! -e "$experiment_root/FLOW_EXTRACTION_COMPLETE"
  test ! -e "$experiment_root/FLOW_EXTRACTION_FAILED"
  mapfile -t cases < <(jq -r '.cases[].case_id' "$manifest")
  pids=()
  for gpu in 0 1 2 3 4 5 6 7; do
    extract_case "$gpu" "${cases[$gpu]}" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [ "$status" -ne 0 ]; then
    touch "$experiment_root/FLOW_EXTRACTION_FAILED"
    return 1
  fi
  test "$(find "$flow_root" -type f -name '*.safetensors' | wc -l | tr -d ' ')" = 32
  test "$(find "$flow_root" -type f -name '*.json' | wc -l | tr -d ' ')" = 32
  touch "$experiment_root/FLOW_EXTRACTION_COMPLETE"
}

wait_for_flows() {
  local attempts=0
  while [ ! -f "$experiment_root/FLOW_EXTRACTION_COMPLETE" ]; do
    if [ -e "$experiment_root/FLOW_EXTRACTION_FAILED" ]; then
      echo "flow extraction failed" >&2
      return 1
    fi
    attempts=$((attempts + 1))
    if [ "$attempts" -gt 360 ]; then
      echo "timed out waiting for flow extraction" >&2
      return 1
    fi
    sleep 10
  done
}

write_calibration_receipt() {
  local iid="$1" role="$2" output="$3" flow="$4" receipt="$5"
  local target_accessed=false anchor_kind="$role"
  case "$role" in
    real_forward|temporal_shuffle|reverse) target_accessed=true ;;
  esac
  jq -n \
    --arg schema_version bernini-mev-fixed-carrier-calibration-receipt-v1 \
    --arg experiment_id action_repr_target_selfgen_20260824_v1 \
    --arg case_id "$iid" \
    --arg route_kind "$role" \
    --arg anchor_kind "$anchor_kind" \
    --argjson target_video_accessed_by_extractor "$target_accessed" \
    --arg carrier_checkpoint "$carrier_checkpoint" \
    --arg carrier_receipt_sha256 "$expected_carrier_receipt_sha" \
    --arg flow_bundle "$flow" \
    --arg flow_bundle_sha256 "$(sha256_file "$flow")" \
    --arg output "$output" \
    --arg output_sha256 "$(sha256_file "$output")" \
    '{
      schema_version: $schema_version,
      experiment_id: $experiment_id,
      case_id: $case_id,
      route_kind: $route_kind,
      anchor_kind: $anchor_kind,
      historical_carrier_global_step: 32,
      current_experiment_optimization_steps: 0,
      parameter_updates_in_current_experiment: false,
      carrier_checkpoint: $carrier_checkpoint,
      carrier_receipt_sha256: $carrier_receipt_sha256,
      flow_bundle: $flow_bundle,
      flow_bundle_sha256: $flow_bundle_sha256,
      output: $output,
      output_sha256: $output_sha256,
      information_firewall: {
        target_video_accessed_by_extractor: $target_video_accessed_by_extractor,
        target_video_accessed_by_trainer: false,
        target_video_accessed_by_renderer: false,
        target_rgb_or_vae_target_used: false,
        anchor_role: "detached_dense_flow_representation_only"
      },
      claim_boundary: "fixed_historically_trained_carrier_zero_new_update_admission_probe_not_new_ours_training"
    }' >"$receipt"
}

decode_case() {
  local iid="$1" role source instruction seed flow output_dir output scratch
  source="$(manifest_value "$iid" '.source.path')"
  instruction="$(manifest_value "$iid" '.instruction')"
  seed="$(manifest_value "$iid" '.seed')"
  for role in real_forward temporal_shuffle reverse self_generated; do
    flow="$flow_root/$iid/${role}.safetensors"
    test -f "$flow"
    output_dir="$infer_root/$iid/$role"
    output="$output_dir/output.mp4"
    test ! -e "$output_dir"
    mkdir -p "$output_dir"
    scratch="/tmp/action-repr-infer-${iid}-${role}-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
    mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
    export MIOPEN_USER_DB_PATH="$scratch/miopen-user"
    export MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
    export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions"
    export TRITON_CACHE_DIR="$scratch/triton"
    "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
      "$inference" \
      --dense-flow-checkpoint "$carrier_checkpoint" \
      --dense-flow-scale 1.0 \
      --flow-bundle "$flow" \
      --bernini-root "$bernini_root" \
      --veomni-root "$veomni_root" \
      --checkpoint "$base_checkpoint" \
      --source-video "$source" \
      --instruction "$instruction" \
      --output "$output" \
      --num-inference-steps 40 \
      --seed "$seed" \
      --source-onset-policy hard1_every_step \
      --method-source-revision e1e67ca9974d28b6691ac77e00de19a485b99ee9 \
      --method-source-archive-sha256 5127d99310cf5b0cce7128f2cb14e02cf58e561ab3529d0e66b1e047e1d29d7c \
      >"$output_dir/run.log" 2>&1
    test -s "$output"
    test -s "$output.receipt.json"
    test -s "$output.dense-flow.json"
    write_calibration_receipt "$iid" "$role" "$output" "$flow" "$output_dir/calibration_receipt.json"
  done
}

run_lane() (
  set -euo pipefail
  local lane="$1"
  shift
  case "$lane" in
    0) export ROCR_VISIBLE_DEVICES=0,1,2,3 ;;
    1) export ROCR_VISIBLE_DEVICES=4,5,6,7 ;;
    *) echo "invalid lane $lane" >&2; return 2 ;;
  esac
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  for iid in "$@"; do
    decode_case "$iid"
  done
)

run_infer_split() {
  local split="$1" expected_host expected_job
  case "$split" in
    fit) expected_host=auh7-1b-gpu-284; expected_job=147873 ;;
    heldout) expected_host=auh7-1b-gpu-232; expected_job=147871 ;;
    *) echo "split must be fit or heldout" >&2; return 2 ;;
  esac
  test "$(hostname -s)" = "$expected_host"
  test "${SLURM_JOB_ID:-}" = "$expected_job"
  common_preflight
  media_preflight
  wait_for_flows
  mkdir -p "$infer_root" "$log_root/inference"
  test ! -e "$experiment_root/INFERENCE_${split}_COMPLETE"
  test ! -e "$experiment_root/INFERENCE_${split}_FAILED"
  mapfile -t cases < <(jq -r --arg split "$split" '.cases[] | select(.split == $split) | .case_id' "$manifest")
  test "${#cases[@]}" = 4
  run_lane 0 "${cases[0]}" "${cases[2]}" & lane0=$!
  run_lane 1 "${cases[1]}" "${cases[3]}" & lane1=$!
  status=0
  if ! wait "$lane0"; then status=1; fi
  if ! wait "$lane1"; then status=1; fi
  if [ "$status" -ne 0 ]; then
    touch "$experiment_root/INFERENCE_${split}_FAILED"
    return 1
  fi
  test "$(find "$infer_root" -type f -name calibration_receipt.json | wc -l | tr -d ' ')" -ge 16
  touch "$experiment_root/INFERENCE_${split}_COMPLETE"
}

run_infer_split_serial() {
  local split="$1" expected_host expected_job revision batch_tag marker_prefix failed_marker expected_root iid role
  case "$split" in
    fit) expected_host=auh7-1b-gpu-213; expected_job=147881 ;;
    heldout) expected_host=auh7-1b-gpu-232; expected_job=147871 ;;
    *) echo "split must be fit or heldout" >&2; return 2 ;;
  esac
  revision="${SERIAL_REVISION:-r2}"
  batch_tag="${SERIAL_BATCH_TAG:-$split}"
  case "$revision" in r2|r3) ;; *) echo "serial revision must be r2 or r3" >&2; return 2 ;; esac
  case "$batch_tag" in *[!A-Za-z0-9_-]*|'') echo "invalid serial batch tag" >&2; return 2 ;; esac
  marker_prefix="INFERENCE_SERIAL_${revision}_${batch_tag}"
  failed_marker="$experiment_root/${marker_prefix}_FAILED"
  expected_root="$experiment_root/inference_serial_${revision}"
  test "$infer_root" = "$expected_root"
  test "$(hostname -s)" = "$expected_host"
  test "${SLURM_JOB_ID:-}" = "$expected_job"
  common_preflight
  media_preflight
  wait_for_flows
  mkdir -p "$infer_root" "$log_root/inference_serial_${revision}"
  test ! -e "$experiment_root/${marker_prefix}_COMPLETE"
  test ! -e "$experiment_root/${marker_prefix}_FAILED"
  if [ -n "${SERIAL_CASE_SET:-}" ]; then
    IFS=',' read -r -a cases <<<"$SERIAL_CASE_SET"
    test "${#cases[@]}" -ge 1
    for iid in "${cases[@]}"; do
      test "$(manifest_value "$iid" '.split')" = "$split"
    done
  else
    mapfile -t cases < <(jq -r --arg split "$split" '.cases[] | select(.split == $split) | .case_id' "$manifest")
    test "${#cases[@]}" = 4
  fi
  trap 'touch "$failed_marker"' ERR
  for iid in "${cases[@]}"; do
    decode_case "$iid"
  done
  for iid in "${cases[@]}"; do
    for role in real_forward temporal_shuffle reverse self_generated; do
      test -s "$infer_root/$iid/$role/calibration_receipt.json"
    done
  done
  trap - ERR
  touch "$experiment_root/${marker_prefix}_COMPLETE"
}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$sealed_source/methods/bernini_action_editing:$runtime_source/methods/bernini_action_editing"

if [ "$#" -ne 1 ]; then
  echo "usage: $0 preflight|extract|infer-fit|infer-heldout|infer-fit-serial|infer-heldout-serial" >&2
  exit 2
fi

case "$1" in
  preflight)
    common_preflight
    media_preflight
    ;;
  extract)
    run_extract
    ;;
  infer-fit)
    run_infer_split fit
    ;;
  infer-heldout)
    run_infer_split heldout
    ;;
  infer-fit-serial|infer-fit-serial-r2)
    run_infer_split_serial fit
    ;;
  infer-heldout-serial|infer-heldout-serial-r2)
    run_infer_split_serial heldout
    ;;
  *)
    echo "unknown mode: $1" >&2
    exit 2
    ;;
esac
