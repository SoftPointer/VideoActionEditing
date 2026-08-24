#!/usr/bin/env bash
#SBATCH --job-name=bern-csvart-v9-inf
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=12:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 077

source_archive="${BERNINI_V9_SOURCE_ARCHIVE:?set BERNINI_V9_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_V9_SOURCE_ARCHIVE_SHA256:?set BERNINI_V9_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_V9_SOURCE_REVISION:?set BERNINI_V9_SOURCE_REVISION}"
source_repository="${BERNINI_V9_SOURCE_REPOSITORY:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
adapter_checkpoint="${BERNINI_V9_ADAPTER_CHECKPOINT:?set BERNINI_V9_ADAPTER_CHECKPOINT}"
source_video="${BERNINI_V9_SOURCE_VIDEO:?set BERNINI_V9_SOURCE_VIDEO}"
instruction="${BERNINI_V9_INSTRUCTION:?set BERNINI_V9_INSTRUCTION}"
output_video="${BERNINI_V9_INFERENCE_OUTPUT:?set BERNINI_V9_INFERENCE_OUTPUT}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
checkpoint_content_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
checkpoint_content_file_count=23
noop_instruction_sha256="fb5f23b5b9de175696cff019f035e81eb1ee6a1123db7e3b63afb604b88daf3a"
target_modules_sha256="16e5dc87ca134419841e2e9af6d26091141aa473aa4cc11ae53d2e4e28e0e4b5"

fail() { echo "[bernini-csvart-v9-infer] ERROR: $*" >&2; exit 2; }

validate_checkpoint_content() {
  local manifest="$1"
  local actual_count
  [[ -f "${manifest}" && ! -L "${manifest}" ]] || fail "checkpoint manifest is not plain"
  [[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${checkpoint_content_manifest_sha256}" ]] || fail "checkpoint manifest hash differs"
  [[ "$(wc -l < "${manifest}")" -eq "${checkpoint_content_file_count}" ]] || fail "checkpoint manifest count differs"
  [[ -z "$(find "${checkpoint}" -path "${checkpoint}/.cache" -prune -o -type l -print -quit)" ]] || fail "checkpoint contains symlink"
  actual_count="$(find "${checkpoint}" -path "${checkpoint}/.cache" -prune -o -type f -print | wc -l)"
  [[ "${actual_count}" -eq "${checkpoint_content_file_count}" ]] || fail "checkpoint file count differs"
  (cd "${checkpoint}" && sha256sum --strict --status -c "${manifest}") || fail "checkpoint content differs"
}

[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid method archive SHA"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid method revision"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid method archive"
[[ -d "${source_repository}/.git" ]] || fail "invalid method repository"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini root"
[[ -d "${veomni_root}/.git" ]] || fail "invalid VeOmni root"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid base checkpoint"
[[ -d "${adapter_checkpoint}" && ! -L "${adapter_checkpoint}" ]] || fail "invalid V9 adapter checkpoint"
for adapter_file in adapter/adapter_config.json adapter/adapter_model.safetensors receipt.json optimizer.pt; do
  [[ -f "${adapter_checkpoint}/${adapter_file}" && ! -L "${adapter_checkpoint}/${adapter_file}" ]] || fail "missing plain adapter ${adapter_file}"
done
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "invalid source video"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ -n "${instruction//[[:space:]]/}" ]] || fail "instruction is empty"
[[ "${output_video}" == /*.mp4 ]] || fail "output must be an absolute MP4"
[[ ! -e "${output_video}" && ! -L "${output_video}" ]] || fail "refusing to overwrite output"
[[ ! -e "${output_video}.receipt.json" && ! -L "${output_video}.receipt.json" ]] || fail "refusing to overwrite receipt"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive hash differs"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_commit}" ]] || fail "VeOmni revision differs"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked tree is dirty"

jq -e \
  --arg adapter_sha "$(sha256sum "${adapter_checkpoint}/adapter/adapter_model.safetensors" | awk '{print $1}')" \
  --arg revision "${source_revision}" \
  --arg archive "${source_archive_sha256}" '
  .schema_version == "bernini-source-kv-route-auh-training-receipt-v9" and
  .method == "bernini-frozen-source-kv-csv-art-v9-auh" and
  .global_step == 40 and
  .formal_exact40_complete == true and
  .exact40_audit.validated == true and
  .exact40_audit.all30_main == true and
  .exact40_audit.selected_block_count == 30 and
  .adapter.target_module_count == 92 and
  .adapter.adapter_tensor_count == 184 and
  .adapter.trainable_parameter_count == 2260992 and
  .immutable_contract.value.run_role == "v9_main" and
  .immutable_contract.value.method_source_revision == $revision and
  .immutable_contract.value.method_source_archive_sha256 == $archive and
  .immutable_contract.value.carrier.selection == "all" and
  .immutable_contract.value.carrier.selected_blocks == [range(0;30)] and
  .immutable_contract.value.fresh_initialization.v8_warm_start == false and
  .paired_target_model_forward_access == false and
  .artifact_validation.verified == true and
  .artifact_validation.status == "post_save_adapter_tensor_file_roundtrip_and_optimizer_load_state_dict_complete" and
  .artifact_validation.adapter_tensor_file_roundtrip_verified == true and
  .artifact_validation.adapter_tensor_file_runtime_equality == true and
  .artifact_validation.torch_deserialize_verified == true and
  .artifact_validation.fresh_optimizer_load_state_dict_verified == true and
  .artifact_validation.optimizer_state_logical_equality_verified == true and
  .artifact_validation.runtime_adapter_loader_verified == false and
  .artifact_validation.fresh_base_peft_from_pretrained_verified == false and
  .artifact_validation.deployment_loader_claim_forbidden == true and
  .artifact_validation.adapter_model_sha256 == $adapter_sha
' "${adapter_checkpoint}/receipt.json" >/dev/null || fail "adapter is not completed exact40 V9-main"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export BERNINI_V9_PINNED_ROOT="${bernini_root}"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-csvart-v9-infer-${SLURM_JOB_ID}.XXXXXX")"
cleanup_task_scratch() {
  if [[ -n "${task_scratch:-}" && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    rm -rf -- "${task_scratch}"
  fi
}
trap cleanup_task_scratch EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
archive_copy="${task_scratch}/method.tar"
archive_from_revision="${task_scratch}/method-from-revision.tar"
source_root="${task_scratch}/source"
mkdir -p \
  "${source_root}" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton" \
  "${output_video%/*}"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
export TMPDIR="${task_scratch}"

cp -- "${source_archive}" "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "private archive copy differs"
[[ "$(git -C "${source_repository}" cat-file -t "${source_revision}^{commit}")" == commit ]] || fail "revision is not a commit"
git -C "${source_repository}" archive --format=tar \
  -o "${archive_from_revision}" \
  "${source_revision}" methods/bernini_action_editing
[[ "$(sha256sum "${archive_from_revision}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "archive bytes do not bind revision"
"${python_bin}" - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile
with tarfile.open(sys.argv[1], mode="r:*") as handle:
    members = handle.getmembers()
    if not members:
        raise SystemExit("empty method archive")
    for member in members:
        path = PurePosixPath(member.name)
        if not member.name or path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe archive path: {member.name!r}")
        if member.issym() or member.islnk() or member.isfifo() or member.isdev():
            raise SystemExit(f"unsafe archive member: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported archive member: {member.name!r}")
PY
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${source_root}"
[[ -z "$(find "${source_root}" -type l -print -quit)" ]] || fail "extracted method contains symlink"
method_root="${source_root}/methods/bernini_action_editing"
for required in \
  infer_source_kv_route_v9.py \
  infer_lora.py \
  source_kv_replay.py \
  source_kv_route_batches.py \
  source_kv_route_objective.py \
  source_kv_route_scope.py \
  train_source_kv_route_auh.py \
  tri_branch_unipc.py \
  inference_sigma_strata.py \
  tests/test_infer_source_kv_route_v9.py \
  tests/test_source_kv_replay.py \
  tests/test_source_kv_route_scope.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive lacks plain ${required}"
done
checkpoint_manifest="${method_root}/audits/bernini_r13_ff4c5d4_checkpoint.sha256"
validate_checkpoint_content "${checkpoint_manifest}"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-csvart-v9-infer] source_revision=${source_revision} archive_sha256=${source_archive_sha256}"
echo "[bernini-csvart-v9-infer] source_sha256=$(sha256sum "${source_video}" | awk '{print $1}')"
echo "[bernini-csvart-v9-infer] adapter_sha256=$(sha256sum "${adapter_checkpoint}/adapter/adapter_model.safetensors" | awk '{print $1}')"
echo "[bernini-csvart-v9-infer] exact92/184 all30 81f/21phase 40step seed2027 Ulysses4"
echo "[bernini-csvart-v9-infer] E_k=S+Q0(A_theta-N_theta); v=(x_k-E_k)/sigma; rho/radius/clipping/mix=none"
echo "[bernini-csvart-v9-infer] expected per-layer capture40 replay200 noop_sha=${noop_instruction_sha256} target_sha=${target_modules_sha256}"
echo "[bernini-csvart-v9-infer] output=${output_video}"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

"${python_bin}" -m py_compile \
  "${method_root}/infer_source_kv_route_v9.py" \
  "${method_root}/source_kv_replay.py" \
  "${method_root}/source_kv_route_scope.py"
for pattern in \
  'test_infer_source_kv_route_v9.py' \
  'test_source_kv_replay.py' \
  'test_source_kv_route_scope.py'
do
  PYTHONPATH="${method_root}" "${python_bin}" -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${method_root}/infer_source_kv_route_v9.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --adapter-checkpoint "${adapter_checkpoint}" \
  --source-video "${source_video}" \
  --instruction "${instruction}" \
  --output "${output_video}" \
  --num-inference-steps 40 \
  --seed 2027 \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}"

receipt_path="${output_video}.receipt.json"
[[ -f "${output_video}" && ! -L "${output_video}" ]] || fail "output video missing"
[[ -f "${receipt_path}" && ! -L "${receipt_path}" ]] || fail "output receipt missing"
jq -e \
  --arg source_sha "$(sha256sum "${source_video}" | awk '{print $1}')" \
  --arg adapter_config_sha "$(sha256sum "${adapter_checkpoint}/adapter/adapter_config.json" | awk '{print $1}')" \
  --arg adapter_model_sha "$(sha256sum "${adapter_checkpoint}/adapter/adapter_model.safetensors" | awk '{print $1}')" \
  --arg training_receipt_sha "$(sha256sum "${adapter_checkpoint}/receipt.json" | awk '{print $1}')" \
  --arg optimizer_sha "$(sha256sum "${adapter_checkpoint}/optimizer.pt" | awk '{print $1}')" \
  --arg output_sha "$(sha256sum "${output_video}" | awk '{print $1}')" '
  .schema_version == "bernini-source-kv-route-inference-receipt-v9" and
  .method == "bernini-frozen-source-kv-csv-art-v9-deployment" and
  .input.accepted_external_conditions == ["source_video", "action_instruction"] and
  .input.source_video_sha256 == $source_sha and
  .input.target_video_argument == false and
  .input.target_accessed_by_inference == false and
  .input.mask_or_swept_tube == false and
  .input.track_pose_trajectory_or_optical_flow == false and
  .input.first_frame_anchor == false and
  .adapter.loaded == true and
  .adapter.merged == false and
  .adapter.target_module_count == 92 and
  .adapter.adapter_tensor_count == 184 and
  .adapter.trainable_parameter_count == 2260992 and
  .adapter.adapter_config_sha256 == $adapter_config_sha and
  .adapter.adapter_model_sha256 == $adapter_model_sha and
  .adapter.training_receipt_file_sha256 == $training_receipt_sha and
  .adapter.optimizer_checkpoint_sha256 == $optimizer_sha and
  .adapter.target_modules_sha256 == "16e5dc87ca134419841e2e9af6d26091141aa473aa4cc11ae53d2e4e28e0e4b5" and
  .adapter.four_plain_files_staged_with_no_follow_fd == true and
  .adapter.config_receipt_and_peft_load_read_private_snapshot_only == true and
  .adapter.shared_and_private_snapshot_hashes_reverified_after_sampling == true and
  .checkpoint_runtime_content.validated == true and
  .checkpoint_runtime_content.verified_file_count == 23 and
  .checkpoint_runtime_content.every_non_cache_file_verified == true and
  .checkpoint_runtime_content.verified_before_and_after_sampling == true and
  .sampling.num_frames == 81 and
  .sampling.num_inference_steps == 40 and
  .sampling.seed == 2027 and
  .sampling.ulysses_size == 4 and
  .runtime_versions.peft == "0.19.1" and
  .deployment_operator.clean_field == "E_k=S+Q0(A_theta,k-N_theta,k)" and
  .deployment_operator.scheduler_velocity == "v_deploy,k=(x_k-E_k)/sigma_k" and
  .deployment_operator.rho == null and
  .deployment_operator.radius == null and
  .deployment_operator.clipping == false and
  .deployment_operator.field_mix == false and
  .deployment_operator.target_energy_retention == 1 and
  .deployment_operator.target_clipped_fraction == 0 and
  .carrier_runtime.four_rank_certificate.all_four_ranks_exact == true and
  .carrier_runtime.four_rank_certificate.all_four_ranks_input_model_prompt_exact == true and
  .carrier_runtime.four_rank_certificate.all_four_ranks_trace_core_exact == true and
  .carrier_runtime.four_rank_certificate.all_four_ranks_generated_latent_exact == true and
  .carrier_runtime.forward_identity_gather.validated_before_forward == true and
  .carrier_runtime.forward_identity_gather.rank_count == 4 and
  .carrier_runtime.rank0_trace.sample_calls == 1 and
  .carrier_runtime.rank0_trace.step_count == 40 and
  (.carrier_runtime.rank0_trace.steps | length) == 40 and
  .output.sha256 == $output_sha and
  .carrier_runtime.four_rank_certificate.per_layer_capture_calls == 40 and
  .carrier_runtime.four_rank_certificate.per_layer_replay_calls == 200 and
  (.carrier_runtime.four_rank_certificate.per_rank | length) == 4 and
  (all(.carrier_runtime.four_rank_certificate.per_rank[];
    .rank_local_capture_calls == 1200 and
    .rank_local_replay_lookups == 6000 and
    .unique_retired_step_identities == 40 and
    .source_prefix_verified == true and
    .processor_restore == true and
    .sampler_scheduler_restore == true
  )) and
  .production_claim_forbidden == true and
  .scientific_claim_authorized == false
' "${receipt_path}" >/dev/null || fail "V9 inference receipt audit failed"

"${python_bin}" - "${receipt_path}" <<'PY'
import hashlib, json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    receipt = json.load(handle)
stored = receipt.pop("receipt_digest", None)
payload = json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
if stored != hashlib.sha256(payload).hexdigest():
    raise SystemExit("receipt digest differs")
carrier = receipt["carrier_runtime"]
trace = carrier["rank0_trace"]
trace_payload = json.dumps(trace, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
trace_digest = hashlib.sha256(trace_payload).hexdigest()
if trace_digest != carrier["rank0_trace_digest"]:
    raise SystemExit("rank0 trace digest differs")
certificate = carrier["four_rank_certificate"]
if trace_digest != certificate["trace_digest"]:
    raise SystemExit("rank0/four-rank trace binding differs")
if len(trace.get("steps", [])) != 40 or [row.get("step_index") for row in trace["steps"]] != list(range(40)):
    raise SystemExit("rank0 trace does not contain exact ordered 40-step records")
digests = (
    "forward_identity_digest",
    "trace_digest",
    "core_runtime_digest",
    "generated_latent_digest",
)
per_rank = certificate["per_rank"]
if len(per_rank) != 4 or any(any(row.get(key) != certificate.get(key) for key in digests) for row in per_rank):
    raise SystemExit("four-rank identity/trace/core/generated digests differ")
PY
final_output_sha="$(sha256sum "${output_video}" | awk '{print $1}')"
receipt_output_sha="$(jq -er '.output.sha256' "${receipt_path}")"
[[ "${final_output_sha}" == "${receipt_output_sha}" ]] || fail "final output/receipt SHA binding differs"
validate_checkpoint_content "${checkpoint_manifest}"
rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-csvart-v9-infer] PASS exact_deployment_operator=true efficacy_claim=false"
