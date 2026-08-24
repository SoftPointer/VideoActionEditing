#!/usr/bin/env bash
#SBATCH --job-name=bern-kv-carrier
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

source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:?set BERNINI_ACTION_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:?set BERNINI_ACTION_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_ACTION_SOURCE_REVISION:?set BERNINI_ACTION_SOURCE_REVISION}"
source_repository="${BERNINI_ACTION_SOURCE_REPOSITORY:?set BERNINI_ACTION_SOURCE_REPOSITORY}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
bernini_archive="${BERNINI_OFFICIAL_ARCHIVE:?set BERNINI_OFFICIAL_ARCHIVE}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
source_video="${BERNINI_ACTION_SOURCE_VIDEO:?set BERNINI_ACTION_SOURCE_VIDEO}"
instruction="${BERNINI_ACTION_INSTRUCTION:?set BERNINI_ACTION_INSTRUCTION}"
output_video="${BERNINI_ACTION_INFERENCE_OUTPUT:?set BERNINI_ACTION_INFERENCE_OUTPUT}"
replay_mode="${BERNINI_SOURCE_KV_REPLAY:-on}"
block_selection="${BERNINI_SOURCE_KV_BLOCK_SELECTION:-all}"
expected_source_tokens="${BERNINI_SOURCE_KV_EXPECTED_TOKENS:-19530}"
num_steps="${BERNINI_ACTION_INFERENCE_STEPS:-40}"
seed="${BERNINI_ACTION_INFERENCE_SEED:-2027}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
checkpoint_content_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
checkpoint_content_file_count=23
bernini_archive_sha256="08a331958560544efb5e842666c236d819bfdc36d34b6eb9c1cdcee1546ae670"
noop_instruction_sha256="fb5f23b5b9de175696cff019f035e81eb1ee6a1123db7e3b63afb604b88daf3a"

fail() {
  echo "[bernini-source-kv-carrier] ERROR: $*" >&2
  exit 2
}

validate_checkpoint_content() {
  local manifest="$1"
  local actual_count
  [[ -f "${manifest}" && ! -L "${manifest}" ]] || fail "checkpoint manifest is not a plain file"
  [[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${checkpoint_content_manifest_sha256}" ]] || fail "checkpoint manifest digest mismatch"
  [[ "$(wc -l < "${manifest}")" -eq "${checkpoint_content_file_count}" ]] || fail "checkpoint manifest file count differs"
  [[ -z "$(find "${checkpoint}" -path "${checkpoint}/.cache" -prune -o -type l -print -quit)" ]] || fail "checkpoint contains a non-cache symlink"
  actual_count="$(find "${checkpoint}" -path "${checkpoint}/.cache" -prune -o -type f -print | wc -l)"
  [[ "${actual_count}" -eq "${checkpoint_content_file_count}" ]] || fail "checkpoint non-cache file set differs"
  (cd "${checkpoint}" && sha256sum --strict --status -c "${manifest}") || fail "checkpoint content hash differs"
}

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid method source archive"
[[ -d "${source_repository}/.git" ]] || fail "invalid method source repository"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini source root"
[[ -f "${bernini_archive}" && ! -L "${bernini_archive}" ]] || fail "invalid Bernini source archive"
[[ -d "${veomni_root}/.git" ]] || fail "VeOmni root must be a Git checkout"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid base checkpoint"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "invalid source video"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ -n "${instruction//[[:space:]]/}" ]] || fail "instruction is empty"
[[ "${replay_mode}" == on || "${replay_mode}" == off ]] || fail "replay must be on or off"
[[ "${block_selection}" == all || "${block_selection}" == mid || "${block_selection}" == late ]] || fail "block selection must be all, mid, or late"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid method archive hash"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid method source revision"
[[ "${num_steps}" == 40 ]] || fail "oracle is fixed to 40 inference steps"
[[ "${expected_source_tokens}" == 19530 ]] || fail "dog oracle source geometry must be N=19530"
[[ "${seed}" =~ ^[0-9]+$ ]] || fail "seed must be non-negative"
[[ "${output_video}" == /*.mp4 ]] || fail "output must be an absolute .mp4 path"
[[ ! -e "${output_video}" && ! -L "${output_video}" ]] || fail "refusing to overwrite output"
[[ ! -e "${output_video}.receipt.json" && ! -L "${output_video}.receipt.json" ]] || fail "refusing to overwrite receipt"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive hash mismatch"
[[ "$(sha256sum "${bernini_archive}" | awk '{print $1}')" == "${bernini_archive_sha256}" ]] || fail "Bernini archive hash mismatch"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_commit}" ]] || fail "VeOmni commit mismatch"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source is dirty"

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

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-source-kv-${SLURM_JOB_ID}.XXXXXX")"
archive_copy="${task_scratch}/method.tar"
archive_from_revision="${task_scratch}/method-from-revision.tar"
mkdir -p \
  "${task_scratch}/source" \
  "${task_scratch}/input" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton" \
  "${output_video%/*}"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"

original_source_sha256="$(sha256sum "${source_video}" | awk '{print $1}')"
staged_source_video="${task_scratch}/input/source.mp4"
cp -- "${source_video}" "${staged_source_video}"
[[ -f "${staged_source_video}" && ! -L "${staged_source_video}" ]] || fail "staged source is not a plain file"
staged_source_sha256="$(sha256sum "${staged_source_video}" | awk '{print $1}')"
[[ "${staged_source_sha256}" == "${original_source_sha256}" ]] || fail "staged source digest differs from original"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${original_source_sha256}" ]] || fail "original source changed during staging"

cp -- "${source_archive}" "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "private method archive hash mismatch"
[[ "$(git -C "${source_repository}" cat-file -t "${source_revision}^{commit}")" == commit ]] || fail "method source revision is not a commit"
git -C "${source_repository}" archive --format=tar \
  -o "${archive_from_revision}" \
  "${source_revision}" methods/bernini_action_editing
[[ "$(sha256sum "${archive_from_revision}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive bytes are not the declared Git revision"
"${python_bin}" - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

with tarfile.open(sys.argv[1], mode="r:*") as handle:
    members = handle.getmembers()
    if not members:
        raise SystemExit("empty method source archive")
    for member in members:
        path = PurePosixPath(member.name)
        if not member.name or path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe archive path: {member.name!r}")
        if member.issym() or member.islnk() or member.isfifo() or member.isdev():
            raise SystemExit(f"unsafe archive member: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported archive member: {member.name!r}")
PY
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
for required in \
  infer_lora.py \
  infer_source_kv_carrier_oracle.py \
  source_kv_replay.py \
  source_kv_route_batches.py \
  tests/test_infer_source_kv_carrier_oracle.py \
  tests/test_source_kv_replay.py
do
  [[ -f "${method_root}/${required}" ]] || fail "archive lacks ${required}"
done
[[ -f "${method_root}/audits/bernini_r13_ff4c5d4_checkpoint.sha256" ]] || fail "archive lacks checkpoint content manifest"
checkpoint_manifest="${method_root}/audits/bernini_r13_ff4c5d4_checkpoint.sha256"
validate_checkpoint_content "${checkpoint_manifest}"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-source-kv-carrier] source_revision=${source_revision}"
echo "[bernini-source-kv-carrier] method_archive_sha256=${source_archive_sha256}"
echo "[bernini-source-kv-carrier] replay=${replay_mode} blocks=${block_selection} frames=81 steps=40 seed=${seed}"
echo "[bernini-source-kv-carrier] expected_source_tokens=19530 expected_pair_tokens=39060"
echo "[bernini-source-kv-carrier] noop_instruction_sha256=${noop_instruction_sha256}"
echo "[bernini-source-kv-carrier] adapter_argument=false adapter_load=false adapter_merge=false training=0"
echo "[bernini-source-kv-carrier] original_source=${source_video} staged_source=${staged_source_video} source_sha256=${staged_source_sha256}"
echo "[bernini-source-kv-carrier] checkpoint_manifest_sha256=${checkpoint_content_manifest_sha256} files=${checkpoint_content_file_count}"
echo "[bernini-source-kv-carrier] output=${output_video}"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

for pattern in \
  test_source_kv_replay.py \
  test_infer_source_kv_carrier_oracle.py \
  test_infer_lora_contract.py
do
  PYTHONPATH="${method_root}" "${python_bin}" -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${method_root}/infer_source_kv_carrier_oracle.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --checkpoint-content-manifest "${checkpoint_manifest}" \
  --source-video "${staged_source_video}" \
  --original-source-path "${source_video}" \
  --expected-source-sha256 "${staged_source_sha256}" \
  --instruction "${instruction}" \
  --output "${output_video}" \
  --replay "${replay_mode}" \
  --block-selection "${block_selection}" \
  --expected-source-tokens "${expected_source_tokens}" \
  --num-inference-steps "${num_steps}" \
  --seed "${seed}" \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}"

receipt_path="${output_video}.receipt.json"
[[ -f "${output_video}" && ! -L "${output_video}" ]] || fail "output video missing"
[[ -f "${receipt_path}" && ! -L "${receipt_path}" ]] || fail "output receipt missing"
actual_output_sha256="$(sha256sum "${output_video}" | awk '{print $1}')"
"${python_bin}" - "${receipt_path}" "${replay_mode}" "${block_selection}" "${source_revision}" "${source_archive_sha256}" "${source_video}" "${staged_source_video}" "${staged_source_sha256}" "${checkpoint_content_manifest_sha256}" "${actual_output_sha256}" <<'PY'
import hashlib
import json
import sys

(
    path,
    replay,
    selection,
    revision,
    archive_sha,
    original_source,
    staged_source,
    source_sha,
    checkpoint_manifest_sha,
    actual_output_sha,
) = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    receipt = json.load(handle)

indices = {
    "all": list(range(30)),
    "mid": list(range(7, 23)),
    "late": list(range(20, 30)),
}[selection]
actual_indices = indices if replay == "on" else []
layers = len(indices) if replay == "on" else 0
expected_capture = layers * 40
expected_replay = layers * 80
weights = receipt.get("weights", {})
optimization = receipt.get("optimization", {})
inputs = receipt.get("input", {})
prompt = receipt.get("prompt_contract", {})
checkpoint_content = receipt.get("checkpoint_content_identity", {})
control = receipt.get("causal_control", {})
oracle = receipt.get("oracle", {})
runtime = oracle.get("runtime_execution_certificate", {})
per_rank = runtime.get("per_rank", [])
checks = (
    receipt.get("schema_version") == "bernini-r-1p3b-frozen-source-kv-carrier-oracle-v1",
    receipt.get("method_source_revision") == revision,
    receipt.get("method_source_archive_sha256") == archive_sha,
    checkpoint_content.get("manifest_sha256_computed") == checkpoint_manifest_sha,
    checkpoint_content.get("manifest_sha256_expected") == checkpoint_manifest_sha,
    checkpoint_content.get("verified_file_count") == 23,
    checkpoint_content.get("every_file_sha256_verified") is True,
    "adapter" not in receipt,
    weights.get("base_checkpoint_loaded") is True,
    weights.get("base_frozen") is True,
    weights.get("base_checkpoint_content_verified") is True,
    weights.get("base_checkpoint_verified_file_count") == 23,
    weights.get("base_checkpoint_content_manifest_sha256") == checkpoint_manifest_sha,
    weights.get("adapter_argument_supported") is False,
    weights.get("legacy_full644_artifact_used") is False,
    weights.get("adapter_weights_loaded") is False,
    weights.get("adapter_weights_merged") is False,
    weights.get("peft_model_constructed") is False,
    weights.get("lora_module_count") == 0,
    optimization.get("zero_training") is True,
    optimization.get("training_steps") == 0,
    optimization.get("optimizer_constructed") is False,
    optimization.get("backward_calls") == 0,
    optimization.get("trainable_parameter_tensors") == 0,
    optimization.get("trainable_parameter_elements") == 0,
    inputs.get("accepted_model_conditions") == ["source_video", "edit_instruction"],
    inputs.get("source_video_path") == original_source,
    inputs.get("staged_source_video_path") == staged_source,
    inputs.get("source_video_sha256") == source_sha,
    inputs.get("expected_source_video_sha256") == source_sha,
    inputs.get("staged_source_hash_verified") is True,
    inputs.get("target_video_argument") is False,
    inputs.get("external_mask_or_swept_tube") is False,
    inputs.get("external_tracking_pose_or_trajectory") is False,
    inputs.get("optical_flow") is False,
    inputs.get("reference_image_or_video") is False,
    inputs.get("first_frame_anchor") is False,
    prompt.get("semantic_noop_instruction_sha256") == "fb5f23b5b9de175696cff019f035e81eb1ee6a1123db7e3b63afb604b88daf3a",
    receipt.get("sampling", {}).get("num_frames") == 81,
    receipt.get("sampling", {}).get("num_inference_steps") == 40,
    receipt.get("sampling", {}).get("ulysses_size") == 4,
    control.get("replay") == replay,
    control.get("block_selection") == selection,
    isinstance(control.get("causal_pairing_digest"), str) and len(control["causal_pairing_digest"]) == 64,
    oracle.get("actual_block_indices") == actual_indices,
    runtime.get("validated") is True,
    runtime.get("all_four_ranks_exact") is True,
    runtime.get("actual_block_indices") == actual_indices,
    runtime.get("per_rank_capture_calls") == expected_capture,
    runtime.get("per_rank_replay_lookups") == expected_replay,
    receipt.get("output", {}).get("sha256") == actual_output_sha,
    len(per_rank) == 4,
    all(row.get("rank_local_bank_capture_calls") == expected_capture for row in per_rank),
    all(row.get("rank_local_bank_replay_lookups") == expected_replay for row in per_rank),
    all(row.get("actual_installed_block_indices") == actual_indices for row in per_rank),
)
if replay == "on":
    checks += (
        all(row.get("per_layer_capture_calls") == 40 for row in per_rank),
        all(row.get("per_layer_replay_calls") == 80 for row in per_rank),
        all(row.get("unique_step_identities") == 40 for row in per_rank),
        all(row.get("source_tokens_runtime") == 19530 for row in per_rank),
        all(row.get("pair_tokens_runtime") == 39060 for row in per_rank),
        all(row.get("processor_restore") is True for row in per_rank),
        all(row.get("sampler_hook_restore") is True for row in per_rank),
    )
else:
    checks += (
        oracle.get("rank0_source_kv_core_receipt") is None,
        all(row.get("processor_patch_installed") is False for row in per_rank),
        all(row.get("carrier_sampler_hook_installed") is False for row in per_rank),
        all(row.get("read_only_observer_installed") is True for row in per_rank),
        all(row.get("observer_hook_restore") is True for row in per_rank),
        all(row.get("observed_shared_step_calls") == 80 for row in per_rank),
        all(row.get("official_negative_forwards") == 40 for row in per_rank),
        all(row.get("official_action_forwards") == 40 for row in per_rank),
        all(row.get("unique_step_identities") == 40 for row in per_rank),
        all(row.get("source_tokens_runtime") == 19530 for row in per_rank),
        all(row.get("pair_tokens_runtime") == 39060 for row in per_rank),
    )

stored = receipt.pop("receipt_digest", None)
payload = json.dumps(
    receipt,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
checks += (stored == hashlib.sha256(payload).hexdigest(),)
raise SystemExit(0 if all(checks) else 1)
PY
[[ $? -eq 0 ]] || fail "oracle receipt audit failed"
[[ "$(sha256sum "${output_video}" | awk '{print $1}')" == "${actual_output_sha256}" ]] || fail "output changed during receipt audit"

rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-source-kv-carrier] PASS replay=${replay_mode} blocks=${block_selection}"
