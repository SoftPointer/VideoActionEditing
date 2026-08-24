#!/usr/bin/env bash
# Create-once Stage-B target-T0 retry-5 one-update canary launcher.
#
# Unlike the frozen Stage-1 launcher, this launcher may create exactly one
# optimizer and execute exactly one update.  It is inert until a separately
# sealed ACTIVE authority addendum is supplied by its exact SHA-256 through
# STAGE_B_T0_AUTHORITY_SHA256.  The checked-in .template.json is deliberately
# rejected and never authorizes a launch.
set -euo pipefail
umask 077

experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_repr_target_selfgen_middle_g1_20260824_v2
source_root="$experiment_root/source_stage_b_t0_retry5"
method_root="$source_root/methods/bernini_action_editing"
authority_addendum="$source_root/stage_b_t0_single_update_retry5_authority_addendum.json"
stage_root="$experiment_root/stage_b_t0_retry5"
log_root="$experiment_root/logs/stage_b_t0_retry5"

runner="$method_root/train_action_repr_target_t0_canary_v1.py"
manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_repr_target_selfgen_20260824_v1/source/experiment_manifest.json
g1_receipt="$experiment_root/stage1_v2_7/g1_admission/target/receipt.json"
g2a_receipt="$experiment_root/stage1_v2_7/g2a/production_world4/0be6494dfac3/receipt.json"
output_root="$stage_root/target_t0/0be6494dfac3/single_update"

python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4

expected_manifest_sha=c78e42f0661e5905407505037ce322d32d67ffec0b70b1cab466f895dc8d0632
expected_g1_receipt_sha=974471a39e8c904cc5234b244d2e59fbf6540d5e951cc2c285454486c8f8066e
expected_g2a_receipt_sha=7ea0ab20709d942ca51a3062f2306407be8f9d0f4445926dca57af9b83fc3f09
expected_g2a_receipt_digest=a39b1be65887532a24378fda25f517bb7b8edb60831e91aaf55468b70f2802b7
expected_projection_sha=6291f3e65908fc8500b7529873f1165011b8bd61916b19d82d696f2485a01dbe

fail() {
  echo "[stage-b-t0] ERROR: $*" >&2
  exit 1
}

normalize_step_token() {
  local raw="${SLURM_STEP_ID:-batch}"
  case "$raw" in
    *"/"*|*"\\"*) fail "Slurm step token contains a path separator" ;;
  esac
  [[ "$raw" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] \
    || fail "Slurm step token is outside the strict whitelist"
  printf '%s\n' "$raw"
}

usage() {
  cat >&2 <<'EOF'
usage:
  STAGE_B_T0_AUTHORITY_SHA256=<64-hex> auh_stage_b_t0_single_update_20260824_retry5.sh preflight
  STAGE_B_T0_AUTHORITY_SHA256=<64-hex> auh_stage_b_t0_single_update_20260824_retry5.sh launch <parent-job-id>
  STAGE_B_T0_AUTHORITY_SHA256=<64-hex> auh_stage_b_t0_single_update_20260824_retry5.sh status

worker is an internal srun entrypoint.  This launcher exposes only the fixed
target-T0 fit case, WORLD4/SP4, and one optimizer update.  It has no TP,
source-copy, self-generated, graph, resume, second-step, decode, or automatic
expansion mode.
EOF
  exit 2
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_sha() {
  local path="$1" expected="$2" actual
  test -f "$path" || fail "missing regular file: $path"
  test ! -L "$path" || fail "symlink is forbidden: $path"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || fail "invalid expected SHA-256 for $path"
  actual="$(sha256_file "$path")"
  test "$actual" = "$expected" \
    || fail "SHA-256 mismatch: $path expected=$expected actual=$actual"
}

authority_value() {
  jq -er "$1" "$authority_addendum"
}

require_authority_hash() {
  local expected="${STAGE_B_T0_AUTHORITY_SHA256:-}"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] \
    || fail "STAGE_B_T0_AUTHORITY_SHA256 must be the final active addendum SHA-256"
  require_sha "$authority_addendum" "$expected"
}

require_source_pin() {
  local relative="$1" expected
  expected="$(jq -er --arg relative "$relative" '.source_hash_pins[$relative]' "$authority_addendum")"
  require_sha "$source_root/$relative" "$expected"
}

require_active_authority_contract() {
  require_authority_hash
  ! grep -Eq '__FINAL_[A-Z0-9_]+_SHA256__' "$authority_addendum" \
    || fail "active authority contains an unresolved source placeholder"
  test "$(authority_value '.schema_version')" = bernini-action-repr-stage-b-t0-single-update-authority-addendum-v1 \
    || fail "Stage-B authority schema differs"
  test "$(authority_value '.experiment_id')" = action_repr_target_selfgen_middle_g1_20260824_v2 \
    || fail "Stage-B experiment id differs"
  test "$(authority_value '.activation.state')" = ACTIVE_CREATE_ONCE_AUTHORITY \
    || fail "draft/revoked Stage-B authority cannot create an optimizer"
  test "$(authority_value '.activation.create_once')" = true \
    || fail "Stage-B authority is not create-once"
  test "$(authority_value '.activation.all_placeholders_replaced')" = true \
    || fail "Stage-B authority source pins are incomplete"
  test "$(authority_value '.canonical_preregistration.sha256')" = 294168e596212bd61e8d555e72702ceeeb993fb18c7fa7536a43d0b00ad592b3 \
    || fail "canonical preregistration binding differs"
  test "$(authority_value '.prior_quantized_energy_match_addendum.sha256')" = 39a2879c35bdc0fc87c67f05adc11e5766f7dae61792c75f44653450b7ee04da \
    || fail "v2.7 quantized-energy authority binding differs"

  test "$(authority_value '.upstream_gate_evidence.manifest.path')" = "$manifest" \
    || fail "authority manifest path differs"
  test "$(authority_value '.upstream_gate_evidence.manifest.sha256')" = "$expected_manifest_sha" \
    || fail "authority manifest SHA-256 differs"
  test "$(authority_value '.upstream_gate_evidence.g1_target.path')" = "$g1_receipt" \
    || fail "authority G1 receipt path differs"
  test "$(authority_value '.upstream_gate_evidence.g1_target.receipt_sha256')" = "$expected_g1_receipt_sha" \
    || fail "authority G1 receipt SHA-256 differs"
  test "$(authority_value '.upstream_gate_evidence.g1_target.required_status')" = passed \
    || fail "authority G1 status differs"
  test "$(authority_value '.upstream_gate_evidence.g1_target.selfgen_status_required')" = not_evaluated \
    || fail "authority misrepresents self-generated admission"
  test "$(authority_value '.upstream_gate_evidence.g1_target.optimizer_creation_authorized_by_receipt')" = false \
    || fail "authority misrepresents G1 as optimizer authority"
  test "$(authority_value '.upstream_gate_evidence.production_g2a.path')" = "$g2a_receipt" \
    || fail "authority production G2a receipt path differs"
  test "$(authority_value '.upstream_gate_evidence.production_g2a.receipt_sha256')" = "$expected_g2a_receipt_sha" \
    || fail "authority production G2a receipt SHA-256 differs"
  test "$(authority_value '.upstream_gate_evidence.production_g2a.receipt_digest')" = "$expected_g2a_receipt_digest" \
    || fail "authority production G2a receipt digest differs"
  test "$(authority_value '.upstream_gate_evidence.production_g2a.required_status')" = PASSED \
    || fail "authority production G2a status differs"

  test "$(authority_value '.canary_scope.arm')" = T0 || fail "only target T0 is authorized"
  test "$(authority_value '.canary_scope.case_id')" = 0be6494dfac3 || fail "T0 case differs"
  test "$(authority_value '.canary_scope.case_split')" = fit || fail "T0 canary is not fit-only"
  test "$(authority_value '.canary_scope.world_size')" = 4 || fail "T0 world size differs"
  test "$(authority_value '.canary_scope.sequence_parallel_size')" = 4 || fail "T0 SP size differs"
  test "$(authority_value '.canary_scope.optimization_steps')" = 1 || fail "T0 step count differs"
  test "$(authority_value '.canary_scope.parameter_updates_required')" = true || fail "real update is not required"
  test "$(authority_value '.canary_scope.target_representation')" = true || fail "target representation is disabled"
  test "$(authority_value '.canary_scope.TP')" = false || fail "TP is not authorized"
  test "$(authority_value '.canary_scope.sourcecopy')" = false || fail "source-copy is not authorized"
  test "$(authority_value '.canary_scope.selfgen')" = false || fail "selfgen is not authorized"
  test "$(authority_value '.canary_scope.graph')" = false || fail "graph is not authorized"
  test "$(authority_value '.canary_scope.automatic_expansion')" = false || fail "automatic expansion is forbidden"
  test "$(authority_value '.canary_scope.longer_training_authorized')" = false || fail "longer training is forbidden"
  test "$(authority_value '.canary_scope.decode_authorized_by_this_addendum')" = false || fail "decode is outside this authority"

  test "$(authority_value '.representation_contract.fixed_jl.kind')" = case_independent_fixed_rademacher_jl \
    || fail "fixed projection kind differs"
  test "$(authority_value '.representation_contract.fixed_jl.seed')" = 2026082401 \
    || fail "fixed projection seed differs"
  test "$(authority_value '.representation_contract.fixed_jl.input_width')" = 1536 \
    || fail "student projection input width differs"
  test "$(authority_value '.representation_contract.fixed_jl.output_width')" = 256 \
    || fail "teacher projection width differs"
  test "$(authority_value '.representation_contract.fixed_jl.tensor_sha256')" = "$expected_projection_sha" \
    || fail "fixed projection tensor SHA-256 differs"
  test "$(authority_value '.representation_contract.phase_activity.phase0_active')" = false \
    || fail "real phase-0 inactivity contract differs"
  test "$(authority_value '.representation_contract.phase_activity.onset_active')" = true \
    || fail "onset must remain active"
  test "$(authority_value '.representation_contract.phase_activity.terminal_active')" = true \
    || fail "terminal must remain active"

  test "$(authority_value '.counterfactual_gradient_contract.required_controls_in_order | join(",")')" = zero,temporal_shuffle,reverse,incomplete,wrong_action \
    || fail "counterfactual control order differs"
  test "$(authority_value '.counterfactual_gradient_contract.no_grad_hinge_prepass')" = true \
    || fail "counterfactual prepass is not required"
  test "$(authority_value '.counterfactual_gradient_contract.correct_side_gradient_passes')" = 1 \
    || fail "correct-side gradient pass count differs"
  test "$(authority_value '.counterfactual_gradient_contract.separate_control_gradient_passes')" = 5 \
    || fail "five control-side gradient passes are not required"
  test "$(authority_value '.counterfactual_gradient_contract.detached_control_scores_without_control_side_gradient_are_sufficient')" = false \
    || fail "detached counterfactual scores cannot substitute for control gradients"

  test "$(authority_value '.optimizer_contract.kind')" = AdamW || fail "optimizer kind differs"
  jq -e '.optimizer_contract.learning_rate == 0.0001' "$authority_addendum" >/dev/null \
    || fail "optimizer learning rate differs"
  jq -e '.optimizer_contract.weight_decay == 0.0' "$authority_addendum" >/dev/null \
    || fail "optimizer weight decay differs"
  test "$(authority_value '.optimizer_contract.steps_exact')" = 1 || fail "optimizer exact step count differs"
  test "$(authority_value '.optimizer_contract.second_step_forbidden')" = true || fail "second step is not forbidden"
  test "$(authority_value '.optimizer_contract.resume_forbidden')" = true || fail "resume is not forbidden"
  test "$(authority_value '.parameter_firewall.trainable_roles_exact | join(",")')" = motion_adapter,middle_projector \
    || fail "trainable role allowlist differs"
  test "$(authority_value '.parameter_firewall.base_generator_frozen')" = true || fail "base must be frozen"
  test "$(authority_value '.parameter_firewall.vae_frozen')" = true || fail "VAE must be frozen"
  test "$(authority_value '.parameter_firewall.text_encoder_frozen')" = true || fail "text encoder must be frozen"
  test "$(authority_value '.parameter_firewall.lora_enabled')" = false || fail "LoRA is forbidden"
  test "$(authority_value '.distributed_contract.world_size')" = 4 || fail "distributed world size differs"
  test "$(authority_value '.distributed_contract.sequence_parallel_size')" = 4 || fail "distributed SP size differs"
  test "$(authority_value '.distributed_contract.runtime_backend')" = nccl/rccl || fail "distributed ROCm backend differs"
  test "$(authority_value '.distributed_contract.rank0_initial_parameter_broadcast')" = true || fail "initial broadcast is required"
  test "$(authority_value '.distributed_contract.explicit_gradient_all_reduce')" = true || fail "gradient all-reduce is required"
  test "$(authority_value '.distributed_contract.post_update_parameter_digest_consensus')" = true || fail "post-update consensus is required"
  test "$(authority_value '.distributed_contract.slurm_rocr_visible_devices_mapping_preserved')" = true \
    || fail "Slurm ROCR device mapping must be preserved"
  test "$(authority_value '.distributed_contract.slurm_rocr_visible_devices_mapping_receipted')" = true \
    || fail "Slurm ROCR device mapping must be receipted"
  test "$(authority_value '.distributed_contract.slurm_rocr_visible_devices_exact_device_count')" = 4 \
    || fail "Slurm ROCR device mapping must contain exactly four devices"
  test "$(authority_value '.distributed_contract.slurm_rocr_visible_devices_physical_range_inclusive | join(",")')" = 0,7 \
    || fail "Slurm ROCR physical device range differs"
  test "$(authority_value '.distributed_contract.hip_visible_devices_must_be_empty')" = true \
    || fail "HIP_VISIBLE_DEVICES must remain empty"
  test "$(authority_value '.distributed_contract.cuda_visible_devices_must_be_empty')" = true \
    || fail "CUDA_VISIBLE_DEVICES must remain empty"
  test "$(authority_value '.output_contract.fresh_create_only')" = true || fail "output must be fresh/create-only"
  test "$(authority_value '.output_contract.step0_and_step1_adapter_states_required')" = true || fail "step0/step1 states are required"
  test "$(authority_value '.claim_boundary')" = optimizer_integration_canary_only_not_ours_not_quality_not_decoded_video \
    || fail "claim boundary differs"
  test "$(authority_value '.runtime_paths.fresh_source_root_name')" = source_stage_b_t0_retry5 \
    || fail "fresh source revision differs"
  test "$(authority_value '.runtime_paths.fresh_stage_root_name')" = stage_b_t0_retry5 \
    || fail "fresh stage revision differs"
  test "$(authority_value '.runtime_paths.bernini_root')" = "$bernini_root" || fail "Bernini root differs"
  test "$(authority_value '.runtime_paths.veomni_root')" = "$veomni_root" || fail "VeOmni root differs"
  test "$(authority_value '.runtime_paths.checkpoint')" = "$base_checkpoint" || fail "base checkpoint differs"

  require_source_pin methods/bernini_action_editing/train_action_repr_target_t0_canary_v1.py
  require_source_pin methods/bernini_action_editing/action_representation_joint_objective_v1.py
  require_source_pin methods/bernini_action_editing/action_repr_g2a_adapter_v1.py
  require_source_pin methods/bernini_action_editing/audit_action_repr_g2a_world4_v1.py
  require_source_pin methods/bernini_action_editing/materialize_decoded_middle_action_repr_v1.py
  require_source_pin methods/bernini_action_editing/dense_flow_token_adapter_v1.py
  require_source_pin methods/bernini_action_editing/exact_local_video_materializer_v1.py
  require_source_pin methods/bernini_action_editing/train_lora.py
  require_source_pin methods/bernini_action_editing/train_self_generated_action_quotient_v1.py
  require_source_pin methods/bernini_action_editing/scripts/auh_stage_b_t0_single_update_20260824_retry5.sh
  require_source_pin methods/bernini_action_editing/tests/test_train_action_repr_target_t0_canary_v1.py
  require_source_pin tests/test_auh_stage_b_t0_single_update_20260824_v1.py
}

require_upstream_gate_receipts() {
  require_sha "$manifest" "$expected_manifest_sha"
  require_sha "$g1_receipt" "$expected_g1_receipt_sha"
  require_sha "$g2a_receipt" "$expected_g2a_receipt_sha"
  test "$(jq -er '.g1_target_status' "$g1_receipt")" = passed || fail "G1_target did not pass"
  test "$(jq -er '.g1_selfgen_status' "$g1_receipt")" = not_evaluated \
    || fail "G1_target receipt misrepresents selfgen status"
  test "$(jq -r '.optimizer_creation_authorized_by_this_receipt' "$g1_receipt")" = false \
    || fail "G1 receipt unexpectedly authorizes an optimizer"
  test "$(jq -er '.passed' "$g2a_receipt")" = true || fail "production G2a did not pass"
  test "$(jq -er '.case_id' "$g2a_receipt")" = 0be6494dfac3 || fail "production G2a case differs"
  test "$(jq -er '.receipt_digest' "$g2a_receipt")" = "$expected_g2a_receipt_digest" \
    || fail "production G2a receipt digest differs"
  test "$(jq -er '.runtime.world_size' "$g2a_receipt")" = 4 || fail "production G2a was not WORLD4"
  test "$(jq -er '.runtime.ulysses_size' "$g2a_receipt")" = 4 || fail "production G2a was not SP4"
  test "$(jq -er '.representation_routes.step0_required_routes | join(",")' "$g2a_receipt")" = correct,zero,temporal_shuffle,reverse,incomplete,wrong_action \
    || fail "production G2a six-route order differs"
  test "$(jq -r '.training_authority.optimizer_created' "$g2a_receipt")" = false \
    || fail "production G2a unexpectedly created an optimizer"
  test "$(jq -er '.training_authority.optimization_steps' "$g2a_receipt")" = 0 \
    || fail "production G2a step count differs"
}

common_preflight() {
  test -x "$python_bin" || fail "vace Python is unavailable"
  test -x /usr/bin/timeout || fail "GNU timeout is unavailable"
  test -x "$(command -v jq)" || fail "jq is unavailable"
  test -x "$(command -v sha256sum)" || fail "sha256sum is unavailable"
  test -d "$source_root" || fail "fresh Stage-B source revision is unavailable"
  test -d "$bernini_root" || fail "pinned Bernini tree is unavailable"
  test -d "$veomni_root" || fail "pinned VeOmni tree is unavailable"
  test -d "$base_checkpoint" || fail "pinned Bernini checkpoint is unavailable"
  require_active_authority_contract
  require_upstream_gate_receipts
}

parent_job_row() {
  scontrol show job "$1" -o
}

validate_parent_gpu_tres() {
  local alloc_tres="$1" generic_count="" mi210_count="" part
  local -a fields
  IFS=',' read -r -a fields <<<"$alloc_tres"
  for part in "${fields[@]}"; do
    case "$part" in
      gres/gpu=*)
        test -z "$generic_count" || fail "parent allocation has duplicate generic GPU TRES"
        generic_count="${part#gres/gpu=}"
        ;;
      gres/gpu:mi210=*)
        test -z "$mi210_count" || fail "parent allocation has duplicate MI210 GPU TRES"
        mi210_count="${part#gres/gpu:mi210=}"
        ;;
    esac
  done
  [[ "$generic_count" =~ ^(4|8)$ ]] \
    || fail "parent allocation must contain exactly 4 or exactly 8 generic GPUs"
  test "$mi210_count" = "$generic_count" \
    || fail "parent generic and MI210 GPU TRES counts differ"
  printf '%s\n' "$generic_count"
}

expected_node_for_job() {
  local row node
  row="$(parent_job_row "$1")"
  node="$(sed -n 's/.* NodeList=\([^ ]*\).*/\1/p' <<<"$row")"
  [[ "$node" =~ ^auh7-1b-gpu-[0-9]+$ ]] \
    || fail "parent job is not bound to exactly one AUH MI210 node: $1/$node"
  printf '%s\n' "$node"
}

validate_parent_job() {
  local job="$1" node row alloc_tres
  [[ "$job" =~ ^[0-9]+$ ]] || fail "parent job id must be numeric"
  row="$(parent_job_row "$job")"
  node="$(expected_node_for_job "$job")"
  [[ " $row " == *" UserId=guangyi.chen(2012) "* ]] || fail "parent job owner differs: $job"
  [[ " $row " == *" Account=faculty-acc "* ]] || fail "parent job account differs: $job"
  [[ " $row " == *" QOS=bgqos "* ]] || fail "parent job QOS differs: $job"
  [[ " $row " == *" Partition=faculty "* ]] || fail "parent job partition differs: $job"
  [[ " $row " == *" JobState=RUNNING "* ]] || fail "parent job is not RUNNING: $job"
  [[ " $row " == *" NodeList=$node "* ]] || fail "parent job/node binding differs: $job/$node"
  [[ " $row " == *" NumNodes=1 "* ]] || fail "parent job must allocate exactly one node: $job"
  alloc_tres="$(sed -n 's/.* AllocTRES=\([^ ]*\).*/\1/p' <<<"$row")"
  test -n "$alloc_tres" || fail "parent job AllocTRES is absent: $job"
  validate_parent_gpu_tres "$alloc_tres" >/dev/null
  if [[ " $row " != *" MinMemoryNode=64G "* && " $row " != *" MinMemoryNode=65536M "* ]]; then
    fail "parent job does not advertise the required 64G host memory: $job"
  fi
}

validate_slurm_rocr_mapping() {
  local raw="$1" token seen=","
  local -a devices
  test -n "$raw" || fail "Slurm did not provide ROCR_VISIBLE_DEVICES"
  case "$raw" in
    *"/"*|*"\\"*) fail "Slurm ROCR device mapping contains a path separator" ;;
  esac
  [[ "$raw" != *[[:space:]]* ]] \
    || fail "Slurm ROCR device mapping contains whitespace"
  [[ "$raw" != ,* && "$raw" != *, && "$raw" != *,,* ]] \
    || fail "Slurm ROCR device mapping contains an empty token"
  IFS=',' read -r -a devices <<<"$raw"
  test "${#devices[@]}" -eq 4 \
    || fail "Slurm ROCR device mapping must contain exactly four tokens"
  for token in "${devices[@]}"; do
    [[ "$token" =~ ^[0-7]$ ]] \
      || fail "Slurm ROCR device token is not one canonical AUH physical device 0..7"
    [[ "$seen" != *",${token},"* ]] \
      || fail "Slurm ROCR device mapping contains a duplicate token"
    seen+="${token},"
  done
  printf '%s\n' "$raw"
}

validate_worker_allocation() {
  local expected_job="$1" expected_node slurm_rocr mapping_sha visible_count
  expected_node="$(expected_node_for_job "$expected_job")"
  test "${SLURM_JOB_ID:-}" = "$expected_job" || fail "worker SLURM job differs"
  test "$(hostname -s)" = "$expected_node" || fail "worker host differs"
  validate_parent_job "$expected_job"
  test -z "${HIP_VISIBLE_DEVICES-}" \
    || fail "nonempty HIP_VISIBLE_DEVICES would override the Slurm ROCR mapping"
  test -z "${CUDA_VISIBLE_DEVICES-}" \
    || fail "nonempty CUDA_VISIBLE_DEVICES would override the Slurm ROCR mapping"
  slurm_rocr="${ROCR_VISIBLE_DEVICES-}"
  validate_slurm_rocr_mapping "$slurm_rocr" >/dev/null
  mapping_sha="$(printf '%s' "$slurm_rocr" | sha256sum | awk '{print $1}')"
  export ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES="$slurm_rocr"
  export ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES_SHA256="$mapping_sha"
  export ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES_COUNT=4
  export ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES_PRESERVED=true
  test "${ROCR_VISIBLE_DEVICES-}" = "$slurm_rocr" \
    || fail "Slurm ROCR device mapping changed during worker validation"
  visible_count="$($python_bin -c 'import torch; print(torch.cuda.device_count())')"
  test "$visible_count" = 4 \
    || fail "WORLD4 worker does not see exactly four MI210 devices"
  test "${ROCR_VISIBLE_DEVICES-}" = "$slurm_rocr" \
    || fail "Slurm ROCR device mapping changed after torch initialization"
}

cgroup_memory_files() {
  local unified relative base current maximum
  unified="$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)"
  if [ -n "$unified" ]; then
    base="/sys/fs/cgroup${unified}"
    current="$base/memory.current"
    maximum="$base/memory.max"
    if [ -r "$current" ] && [ -r "$maximum" ]; then
      printf '%s\t%s\n' "$current" "$maximum"
      return 0
    fi
  fi
  relative="$(awk -F: '$2 ~ /(^|,)memory(,|$)/ {print $3}' /proc/self/cgroup)"
  base="/sys/fs/cgroup/memory${relative}"
  current="$base/memory.usage_in_bytes"
  maximum="$base/memory.limit_in_bytes"
  [ -r "$current" ] && [ -r "$maximum" ] || return 1
  printf '%s\t%s\n' "$current" "$maximum"
}

memory_guard_preflight() {
  local files current_file max_file current maximum_raw maximum available gib proof_source
  files="$(cgroup_memory_files)" || fail "cannot prove the Slurm memory cgroup"
  IFS=$'\t' read -r current_file max_file <<<"$files"
  current="$(<"$current_file")"
  maximum_raw="$(<"$max_file")"
  [[ "$current" =~ ^[0-9]+$ ]] || fail "Slurm memory current value is nonnumeric"
  gib=$((1024 * 1024 * 1024))
  if [[ "$maximum_raw" =~ ^[0-9]+$ ]]; then
    maximum="$maximum_raw"
    proof_source=cgroup_memory_max
  elif [ "$maximum_raw" = max ] && [ "${SLURM_MEM_PER_NODE:-}" = 65536 ]; then
    maximum=$((SLURM_MEM_PER_NODE * 1024 * 1024))
    proof_source=slurm_mem_per_node_65536_cgroup_max_unbounded
  else
    fail "Slurm memory limit is neither numeric nor the sealed 65536 MiB allocation"
  fi
  [ "$maximum" -ge $((60 * gib)) ] && [ "$maximum" -le $((70 * gib)) ] \
    || fail "proved cgroup is not the required 64G class: $maximum bytes"
  available=$((maximum - current))
  [ "$available" -ge $((48 * gib)) ] \
    || fail "less than 48 GiB remains before WORLD4 T0: $available bytes"
  printf '%s\t%s\t%s\n' "$current_file" "$maximum" "$proof_source"
}

memory_watchdog() {
  local child="$1" current_file="$2" maximum="$3" marker="$4" threshold current gib
  gib=$((1024 * 1024 * 1024))
  threshold=$((maximum - 2 * gib))
  while kill -0 "$child" 2>/dev/null; do
    current="$(<"$current_file")"
    if [[ "$current" =~ ^[0-9]+$ ]] && [ "$current" -ge "$threshold" ]; then
      printf 'memory_watchdog_limit_bytes=%s\nobserved_bytes=%s\n' "$threshold" "$current" >"$marker"
      kill -TERM "$child" 2>/dev/null || true
      return 0
    fi
    sleep 2
  done
}

run_world4() {
  local log="$1" current_file="$2" maximum="$3"
  shift 3
  local marker child watchdog status
  marker="${log}.memory-guard"
  test ! -e "$marker" || fail "memory guard marker already exists: $marker"
  "$@" >"$log" 2>&1 &
  child=$!
  memory_watchdog "$child" "$current_file" "$maximum" "$marker" &
  watchdog=$!
  status=0
  wait "$child" || status=$?
  kill -TERM "$watchdog" 2>/dev/null || true
  wait "$watchdog" 2>/dev/null || true
  test ! -e "$marker" || fail "64G memory watchdog stopped WORLD4 T0: $marker"
  [ "$status" -eq 0 ] || fail "WORLD4 T0 runner failed with status $status"
}

validate_completed_output() {
  local receipt="$output_root/receipt.json" receipt_rocr receipt_rocr_sha computed_rocr_sha
  test -f "$receipt" || fail "T0 runner returned without receipt.json"
  test ! -L "$receipt" || fail "T0 receipt may not be a symlink"
  "$python_bin" -B -c \
    'import sys; from train_action_repr_target_t0_canary_v1 import validate_published_t0_output; validate_published_t0_output(sys.argv[1])' \
    "$output_root"
  test "$(jq -er '.schema_version' "$receipt")" = bernini-action-repr-target-t0-one-step-receipt-v1 \
    || fail "T0 receipt schema differs"
  test "$(jq -er '.case_id' "$receipt")" = 0be6494dfac3 || fail "T0 receipt case differs"
  test "$(jq -er '.optimizer_created' "$receipt")" = true || fail "T0 optimizer was not created"
  test "$(jq -er '.optimization_steps' "$receipt")" = 1 || fail "T0 did not execute exactly one step"
  jq -e '.parameter_updates > 0' "$receipt" >/dev/null || fail "T0 produced no parameter update"
  test "$(jq -er '.training.optimizer_kind' "$receipt")" = AdamW || fail "T0 optimizer differs"
  jq -e '.training.learning_rate == 0.0001 and .training.weight_decay == 0.0' "$receipt" >/dev/null \
    || fail "T0 optimizer hyperparameters differ"
  test "$(jq -er '.training.source_copy_adapter_enabled' "$receipt")" = false || fail "source-copy unexpectedly enabled"
  test "$(jq -er '.training.all_five_control_gradient_passes_executed' "$receipt")" = true \
    || fail "five control gradient passes did not execute"
  test "$(jq -er '.training.renderer_base_identity_versions_bytes_unchanged' "$receipt")" = true \
    || fail "frozen renderer base changed"
  test "$(jq -er '.upstream_authority.authorization_addendum_sha256' "$receipt")" = "${STAGE_B_T0_AUTHORITY_SHA256}" \
    || fail "T0 receipt is not bound to the active addendum bytes"
  test "$(jq -er '.runtime.slurm_rocr_visible_devices.mapping_preserved' "$receipt")" = true \
    || fail "T0 receipt does not preserve the Slurm ROCR mapping"
  test "$(jq -er '.runtime.slurm_rocr_visible_devices.device_count' "$receipt")" = 4 \
    || fail "T0 receipt does not bind exactly four Slurm ROCR devices"
  receipt_rocr="$(jq -er '.runtime.slurm_rocr_visible_devices.raw' "$receipt")"
  validate_slurm_rocr_mapping "$receipt_rocr" >/dev/null
  receipt_rocr_sha="$(jq -er '.runtime.slurm_rocr_visible_devices.sha256' "$receipt")"
  computed_rocr_sha="$(printf '%s' "$receipt_rocr" | sha256sum | awk '{print $1}')"
  test "$receipt_rocr_sha" = "$computed_rocr_sha" \
    || fail "T0 receipt Slurm ROCR mapping SHA-256 differs"
  if [ -n "${ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES-}" ]; then
    test "$receipt_rocr" = "$ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES" \
      || fail "T0 receipt Slurm ROCR mapping differs from the worker audit"
    test "$receipt_rocr_sha" = "${ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES_SHA256-}" \
      || fail "T0 receipt Slurm ROCR mapping audit SHA-256 differs"
  fi
  test "$(jq -er '.claim_scope' "$receipt")" = one_step_optimizer_execution_canary_only_not_ours_or_quality_success \
    || fail "T0 receipt claim boundary differs"
  test -f "$output_root/step0000/adapter_model.safetensors" || fail "step0 adapter state is absent"
  test -f "$output_root/step0001/adapter_model.safetensors" || fail "step1 adapter state is absent"
  test -f "$output_root/step0000/receipt.json" || fail "step0 state receipt is absent"
  test -f "$output_root/step0001/receipt.json" || fail "step1 state receipt is absent"
}

run_worker() {
  local expected_job="$1" scratch files current_file maximum proof_source log step_token
  common_preflight
  validate_worker_allocation "$expected_job"
  test ! -e "$output_root" || fail "T0 output must be a fresh create-only path: $output_root"
  mkdir -p "$(dirname "$output_root")" "$log_root"
  log="$log_root/target-t0-0be6494dfac3-job${expected_job}-world4.log"
  test ! -e "$log" || fail "T0 WORLD4 log must be create-only: $log"
  step_token="$(normalize_step_token)"
  scratch="/tmp/action-repr-stage-b-t0-0be6494dfac3-${SLURM_JOB_ID}-${step_token}"
  test ! -e "$scratch" || fail "T0 scratch path already exists: $scratch"
  mkdir -p "$scratch/xdg" "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
  export XDG_CACHE_HOME="$scratch/xdg"
  export MIOPEN_USER_DB_PATH="$scratch/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
  export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions"
  export TRITON_CACHE_DIR="$scratch/triton"
  files="$(memory_guard_preflight)"
  IFS=$'\t' read -r current_file maximum proof_source <<<"$files"
  run_world4 "$log" "$current_file" "$maximum" \
    /usr/bin/timeout --signal=TERM --kill-after=60s 45m \
      "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
      "$runner" \
      --authorization-addendum "$authority_addendum" \
      --manifest "$manifest" \
      --g1-admission-receipt "$g1_receipt" \
      --g2a-receipt "$g2a_receipt" \
      --bernini-root "$bernini_root" \
      --veomni-root "$veomni_root" \
      --checkpoint "$base_checkpoint" \
      --output "$output_root"
  validate_completed_output
  echo "[stage-b-t0] SINGLE_UPDATE_PASS case=0be6494dfac3 arm=T0 WORLD4=true SP4=true steps=1 claim=optimizer_integration_only output=$output_root memory_proof=$proof_source"
}

launch_canary() {
  local job="$1" node outer_log
  common_preflight
  validate_parent_job "$job"
  test ! -e "$output_root" || fail "T0 output is not fresh: $output_root"
  node="$(expected_node_for_job "$job")"
  mkdir -p "$log_root"
  outer_log="$log_root/target-t0-0be6494dfac3-job${job}-outer.log"
  test ! -e "$outer_log" || fail "T0 launch log is not fresh: $outer_log"
  srun --jobid="$job" --exclusive --exact --nodelist="$node" --nodes=1 --ntasks=1 \
    --gres=gpu:mi210:4 --cpus-per-task=16 --mem=0 \
    --export="ALL,STAGE_B_T0_AUTHORITY_SHA256=${STAGE_B_T0_AUTHORITY_SHA256}" \
    "$method_root/scripts/auh_stage_b_t0_single_update_20260824_retry5.sh" \
      worker "$job" >"$outer_log" 2>&1
  validate_completed_output
  echo "[stage-b-t0] srun complete case=0be6494dfac3 job=$job steps=1 output=$output_root log=$outer_log"
}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
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
    test ! -e "$output_root" || fail "T0 output is not fresh"
    echo "[stage-b-t0] PREFLIGHT_PASS authority=active G1_target=passed production_G2a=passed WORLD4_SP4=true arm=T0 steps=1 TP=false sourcecopy=false selfgen=false graph=false"
    ;;
  launch)
    [ "$#" -eq 1 ] || usage
    launch_canary "$1"
    ;;
  worker)
    [ "$#" -eq 1 ] || usage
    run_worker "$1"
    ;;
  status)
    [ "$#" -eq 0 ] || usage
    common_preflight
    if [ ! -e "$output_root" ]; then
      echo "[stage-b-t0] PENDING no output exists"
      exit 0
    fi
    validate_completed_output
    echo "[stage-b-t0] COMPLETED_VALID steps=1 claim=optimizer_integration_only output=$output_root"
    ;;
  *)
    usage
    ;;
esac


