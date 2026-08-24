#!/usr/bin/env bash
set -euo pipefail

experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_repr_target_selfgen_middle_g1_20260824_v2
source_root="$experiment_root/source_stage_b_t0_retry8"
method_root="$source_root/methods/bernini_action_editing"
runner="$method_root/train_action_repr_target_t0_canary_retry8_v1.py"
authority="$source_root/stage_b_t0_single_update_retry8_authority_addendum.json"
manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_repr_target_selfgen_20260824_v1/source/experiment_manifest.json
g1_receipt="$experiment_root/stage1_v2_7/g1_admission/target/receipt.json"
g2a_receipt="$experiment_root/stage1_v2_7/g2a/production_world4/0be6494dfac3/receipt.json"
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
stage_root="$experiment_root/stage_b_t0_retry8/target_t0/0be6494dfac3"
output_root="$stage_root/single_update"
attempt_claim="$stage_root/.single_update.retry8.attempt_claim.json"
log_root="$experiment_root/logs/stage_b_t0_retry8"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

fail() {
  printf '[stage-b-t0-retry8] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  printf 'usage: %s {preflight|launch JOB_ID|worker JOB_ID|status}\n' "$0" >&2
  exit 2
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

create_empty_once() {
  local path="$1"
  (set -o noclobber; umask 077; : >"$path") 2>/dev/null \
    || fail "create-only log already exists: $path"
}

require_fresh_attempt() {
  if [ -L "$attempt_claim" ] || [ -e "$attempt_claim" ]; then
    if [ ! -e "$output_root" ] && [ ! -L "$output_root" ]; then
      fail "CLAIMED_INCOMPLETE retry8 is permanently consumed; use retry9: $attempt_claim"
    fi
    fail "retry8 attempt already exists; use retry9"
  fi
  [ ! -e "$output_root" ] && [ ! -L "$output_root" ] \
    || fail "retry8 output exists without its permanent claim"
}

require_authority() {
  local external_sha observed relative expected path pins_json pins_digest
  external_sha="${STAGE_B_T0_AUTHORITY_SHA256-}"
  [[ "$external_sha" =~ ^[0-9a-f]{64}$ ]] \
    || fail "STAGE_B_T0_AUTHORITY_SHA256 is absent or malformed"
  [ -f "$authority" ] && [ ! -L "$authority" ] \
    || fail "retry8 authority is unavailable"
  observed="$(sha256_file "$authority")"
  [ "$observed" = "$external_sha" ] \
    || fail "external retry8 authority seal differs"
  [ "$(jq -er '.schema_version' "$authority")" = bernini-action-repr-stage-b-t0-single-update-retry8-authority-addendum-v1 ] \
    || fail "retry8 authority schema differs"
  [ "$(jq -er '.activation.state' "$authority")" = ACTIVE_CREATE_ONCE_AUTHORITY ] \
    || fail "retry8 authority is not active"
  pins_json="$(jq -cS '.source_hash_pins' "$authority")"
  pins_digest="$(printf '%s' "$pins_json" | sha256sum | awk '{print $1}')"
  [ "$(jq -er '.source_hash_pins_digest' "$authority")" = "$pins_digest" ] \
    || fail "retry8 source-pin digest differs"
  while IFS=$'\t' read -r relative expected; do
    path="$source_root/$relative"
    [ -f "$path" ] && [ ! -L "$path" ] \
      || fail "pinned retry8 source is unavailable: $relative"
    [ "$(sha256_file "$path")" = "$expected" ] \
      || fail "pinned retry8 source differs: $relative"
  done < <(jq -r '.source_hash_pins | to_entries[] | [.key,.value] | @tsv' "$authority")
}

require_upstream() {
  local path expected
  while IFS=$'\t' read -r path expected; do
    [ -f "$path" ] && [ ! -L "$path" ] || fail "upstream evidence missing: $path"
    [ "$(sha256_file "$path")" = "$expected" ] \
      || fail "upstream evidence hash differs: $path"
  done < <(jq -r '
    .upstream_gate_evidence
    | [
        [.manifest.path,.manifest.sha256],
        [.g1_target.path,.g1_target.receipt_sha256],
        [.production_g2a.path,.production_g2a.receipt_sha256]
      ][] | @tsv
  ' "$authority")
  [ "$(jq -er '.g1_target_status' "$g1_receipt")" = passed ] \
    || fail "target G1 is not passed"
  [ "$(jq -er '.g1_selfgen_status' "$g1_receipt")" = not_evaluated ] \
    || fail "target G1/selfgen boundary differs"
  [ "$(jq -er '.passed' "$g2a_receipt")" = true ] \
    || fail "production G2a is not passed"
  [ "$(jq -er '.training_authority.optimizer_created' "$g2a_receipt")" = false ] \
    || fail "production G2a unexpectedly created an optimizer"
  path="$(jq -er '.representation_contract.batch_replay_diagnostic.path' "$authority")"
  expected="$(jq -er '.representation_contract.batch_replay_diagnostic.file_sha256' "$authority")"
  [ -f "$path" ] && [ ! -L "$path" ] && [ "$(sha256_file "$path")" = "$expected" ] \
    || fail "batch replay diagnostic evidence differs"
  jq -e '
    .passed == true
    and .diagnostic_only == true
    and .renderer_model_loaded == false
    and .optimizer_created == false
    and .optimization_steps == 0
    and .tensor_payload_persisted == false
    and .rematerialized.source_posterior_matches_historical == false
    and .rematerialized.matched_native_batch_matches_historical == false
  ' "$path" >/dev/null || fail "batch replay diagnostic contract differs"
}

common_preflight() {
  [ -x "$python_bin" ] || fail "vace Python is unavailable"
  command -v jq >/dev/null || fail "jq is unavailable"
  command -v sha256sum >/dev/null || fail "sha256sum is unavailable"
  [ -d "$source_root" ] || fail "retry8 source root is unavailable"
  [ -d "$bernini_root" ] && [ -d "$veomni_root" ] && [ -d "$checkpoint" ] \
    || fail "pinned Bernini runtime is unavailable"
  require_authority
  require_upstream
  require_fresh_attempt
}

job_row() {
  scontrol show job "$1" -o
}

job_node() {
  local row node
  row="$(job_row "$1")"
  node="$(sed -n 's/.* NodeList=\([^ ]*\).*/\1/p' <<<"$row")"
  [[ "$node" =~ ^auh7-1b-gpu-[0-9]+$ ]] || fail "job node differs: $node"
  printf '%s\n' "$node"
}

validate_job() {
  local job="$1" row node alloc
  [[ "$job" =~ ^[0-9]+$ ]] || fail "job id must be numeric"
  row="$(job_row "$job")"
  node="$(job_node "$job")"
  [[ " $row " == *" UserId=guangyi.chen(2012) "* ]] || fail "job owner differs"
  [[ " $row " == *" Account=faculty-acc "* ]] || fail "job account differs"
  [[ " $row " == *" QOS=bgqos "* ]] || fail "job QOS differs"
  [[ " $row " == *" Partition=faculty "* ]] || fail "job partition differs"
  [[ " $row " == *" JobState=RUNNING "* ]] || fail "job is not RUNNING"
  [[ " $row " == *" NodeList=$node "* ]] || fail "job/node binding differs"
  [[ " $row " == *" NumNodes=1 "* ]] || fail "job must have one node"
  alloc="$(sed -n 's/.* AllocTRES=\([^ ]*\).*/\1/p' <<<"$row")"
  [[ ",$alloc," == *",gres/gpu:mi210=4,"* || ",$alloc," == *",gres/gpu:mi210=8,"* ]] \
    || fail "job must allocate 4 or 8 MI210 GPUs"
  if [[ " $row " != *" MinMemoryNode=64G "* && " $row " != *" MinMemoryNode=65536M "* ]]; then
    fail "job must advertise 64G host memory"
  fi
  printf '%s\n' "$node"
}

validate_worker() {
  local expected_job="$1" raw count token seen=,
  local -a devices
  validate_job "$expected_job" >/dev/null
  [ "${SLURM_JOB_ID-}" = "$expected_job" ] || fail "worker job id differs"
  [ "$(hostname -s)" = "$(job_node "$expected_job")" ] || fail "worker node differs"
  [ -z "${HIP_VISIBLE_DEVICES-}" ] && [ -z "${CUDA_VISIBLE_DEVICES-}" ] \
    || fail "HIP/CUDA visibility overrides are forbidden"
  raw="${ROCR_VISIBLE_DEVICES-}"
  IFS=',' read -r -a devices <<<"$raw"
  [ "${#devices[@]}" -eq 4 ] || fail "worker requires exactly four visible GPUs"
  for token in "${devices[@]}"; do
    [[ "$token" =~ ^[0-7]$ ]] || fail "ROCR device token differs"
    [[ "$seen" != *",$token,"* ]] || fail "ROCR device token is duplicated"
    seen+="$token,"
  done
  count="$($python_bin -c 'import torch; print(torch.cuda.device_count())')"
  [ "$count" = 4 ] || fail "PyTorch does not see exactly four MI210 GPUs"
  export ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES="$raw"
  export ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES_SHA256
  ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES_SHA256="$(printf '%s' "$raw" | sha256sum | awk '{print $1}')"
  export ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES_COUNT=4
  export ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES_PRESERVED=true
}

validate_output() {
  [ -f "$output_root/receipt.json" ] || fail "retry8 receipt is absent"
  PYTHONPATH="$source_root:$method_root" "$python_bin" -B -c \
    'import sys; from train_action_repr_target_t0_canary_retry8_v1 import validate_published_t0_output; validate_published_t0_output(sys.argv[1])' \
    "$output_root"
  jq -e '
    .schema_version == "bernini-action-repr-target-t0-one-step-retry8-receipt-v1"
    and .optimizer_created == true
    and .optimization_steps == 1
    and .parameter_updates > 0
    and .training.same_runtime_g2a_gate.pre_adapter_native_baseline_executed == true
    and .training.same_runtime_g2a_gate.same_batch_used_by_optimizer_canary == true
    and .training.same_runtime_g2a_gate.route_off_and_six_zero_init_routes_exact_native_bits == true
    and .training.cross_run_historical_match_required == false
    and .training.renderer_base_identity_versions_bytes_unchanged == true
    and .decoded_video_generated == false
    and .ours_model_claimed == false
    and .quality_success_claimed == false
  ' "$output_root/receipt.json" >/dev/null || fail "retry8 receipt contract differs"
}

run_worker() {
  local job="$1" step_token scratch log
  common_preflight
  validate_worker "$job"
  mkdir -p "$stage_root" "$log_root"
  log="$log_root/target-t0-0be6494dfac3-job${job}-world4.log"
  create_empty_once "$log"
  step_token="${SLURM_STEP_ID//[^A-Za-z0-9_.-]/_}"
  scratch="/tmp/action-repr-stage-b-t0-retry8-0be6494dfac3-${job}-${step_token}"
  [ ! -e "$scratch" ] || fail "retry8 scratch path already exists"
  mkdir -p "$scratch/xdg" "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
  export XDG_CACHE_HOME="$scratch/xdg"
  export MIOPEN_USER_DB_PATH="$scratch/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
  export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions"
  export TRITON_CACHE_DIR="$scratch/triton"
  /usr/bin/timeout --signal=TERM --kill-after=60s 60m \
    "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$runner" \
    --authorization-addendum "$authority" \
    --manifest "$manifest" \
    --g1-admission-receipt "$g1_receipt" \
    --g2a-receipt "$g2a_receipt" \
    --bernini-root "$bernini_root" \
    --veomni-root "$veomni_root" \
    --checkpoint "$checkpoint" \
    --output "$output_root" >>"$log" 2>&1 \
    || fail "WORLD4 retry8 runner failed"
  validate_output
  printf '[stage-b-t0-retry8] SINGLE_UPDATE_PASS case=0be6494dfac3 WORLD4=true SP4=true steps=1 output=%s\n' "$output_root"
}

launch() {
  local job="$1" node outer
  common_preflight
  node="$(validate_job "$job")"
  mkdir -p "$log_root"
  outer="$log_root/target-t0-0be6494dfac3-job${job}-outer.log"
  create_empty_once "$outer"
  srun --jobid="$job" --exclusive --exact --nodelist="$node" --nodes=1 --ntasks=1 \
    --gres=gpu:mi210:4 --cpus-per-task=16 --mem=0 \
    --export="ALL,STAGE_B_T0_AUTHORITY_SHA256=${STAGE_B_T0_AUTHORITY_SHA256}" \
    "$0" worker "$job" >>"$outer" 2>&1 \
    || fail "retry8 srun failed; claim, logs, and output must be preserved"
  validate_output
  printf '[stage-b-t0-retry8] srun complete job=%s output=%s\n' "$job" "$output_root"
}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export ACTION_REPR_T0_TORCH_NUM_THREADS=1 ACTION_REPR_T0_TORCH_NUM_INTEROP_THREADS=1
export ACTION_REPR_T0_STRICT_LEGACY_REPLAY=true
export MALLOC_ARENA_MAX=2 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_HOME=/vast/users/guangyi.chen/.cache/torch
export PYTHONPATH="$source_root:$method_root"

[ "$#" -ge 1 ] || usage
command="$1"
shift
case "$command" in
  preflight)
    [ "$#" -eq 0 ] || usage
    common_preflight
    echo '[stage-b-t0-retry8] PREFLIGHT_PASS authority=active same_runtime_G2a=true WORLD4_SP4=true steps=1'
    ;;
  launch)
    [ "$#" -eq 1 ] || usage
    launch "$1"
    ;;
  worker)
    [ "$#" -eq 1 ] || usage
    run_worker "$1"
    ;;
  status)
    [ "$#" -eq 0 ] || usage
    require_authority
    require_upstream
    if [ -e "$output_root/receipt.json" ]; then
      validate_output
      echo '[stage-b-t0-retry8] COMPLETED_VALID'
    elif [ -e "$attempt_claim" ] || [ -L "$attempt_claim" ]; then
      fail "CLAIMED_INCOMPLETE retry8 is permanently consumed; use retry9: $attempt_claim"
    else
      echo '[stage-b-t0-retry8] FRESH_UNCLAIMED'
    fi
    ;;
  *) usage ;;
esac
