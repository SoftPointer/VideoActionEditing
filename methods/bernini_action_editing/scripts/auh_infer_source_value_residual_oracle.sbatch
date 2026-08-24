#!/usr/bin/env bash
#SBATCH --job-name=bern-value-oracle
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=18:00:00
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
output_video="${BERNINI_ACTION_INFERENCE_OUTPUT:?set BERNINI_ACTION_INFERENCE_OUTPUT}"
arm="${BERNINI_SOURCE_VALUE_ARM:?set BERNINI_SOURCE_VALUE_ARM to one registered arm}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
receipt_path="${output_video}.receipt.json"
artifact_paths_validated=0
artifact_committed=0

source_video="/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/288545b9c031491a/source.mp4"
instruction="Make the dog pick up the bone and hold it in its mouth."
expected_source_sha256="5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
expected_source_tokens=19530
expected_pair_tokens=39060
num_steps=40
num_frames=81
seed=2027

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
checkpoint_content_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
checkpoint_content_file_count=23
bernini_archive_sha256="08a331958560544efb5e842666c236d819bfdc36d34b6eb9c1cdcee1546ae670"
noop_instruction_sha256="fb5f23b5b9de175696cff019f035e81eb1ee6a1123db7e3b63afb604b88daf3a"
historical_o0_mp4_sha256="980b0daa85a15feac427f6f2611985d3e33896a0ac45e1ee57938e146ac987c9"
arm_table_sha256="8664401d6372632e41ecaf8623dc38f4245b60e7f3350ad2f6adb5ed3cdbd202"

fail() {
  echo "[bernini-source-value-oracle] ERROR: $*" >&2
  exit 2
}

case "${arm}" in
  Z0)
    operator="full_k_value"; selection="late"; gate="0.0"; layers=10; residual_per_layer=0; lookups_per_layer=0
    ;;
  F25)
    operator="full_k_value"; selection="late"; gate="0.25"; layers=10; residual_per_layer=80; lookups_per_layer=80
    ;;
  F50)
    operator="full_k_value"; selection="late"; gate="0.5"; layers=10; residual_per_layer=80; lookups_per_layer=80
    ;;
  F100)
    operator="full_k_value"; selection="late"; gate="1.0"; layers=10; residual_per_layer=80; lookups_per_layer=80
    ;;
  FA25)
    operator="full_k_value"; selection="all"; gate="0.25"; layers=30; residual_per_layer=80; lookups_per_layer=80
    ;;
  SN10)
    operator="source_normalized_value"; selection="late"; gate="0.1"; layers=10; residual_per_layer=80; lookups_per_layer=80
    ;;
  CK10)
    operator="centered_cached_kv"; selection="late"; gate="0.1"; layers=10; residual_per_layer=160; lookups_per_layer=80
    ;;
  *) fail "arm must be Z0,F25,F50,F100,FA25,SN10,or CK10" ;;
esac

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

method_tree_digest() {
  "${python_bin}" - "$1" <<'PY'
import hashlib
import json
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1]).resolve(strict=True)
rows = []
for path in sorted(root.rglob("*")):
    mode = path.lstat().st_mode
    relative = path.relative_to(root).as_posix()
    if stat.S_ISLNK(mode):
        raise SystemExit(f"method runtime tree contains symlink: {relative}")
    if stat.S_ISDIR(mode):
        continue
    if not stat.S_ISREG(mode):
        raise SystemExit(f"method runtime tree contains non-regular entry: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    rows.append({"path": relative, "sha256": digest.hexdigest()})
payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(payload).hexdigest())
PY
}

pinned_source_identity() {
  PYTHONPATH="${method_root}" "${python_bin}" - \
    "${bernini_root}" "${veomni_root}" "${bernini_commit}" "${veomni_commit}" <<'PY'
import sys
import infer_lora as legacy

bernini, veomni, expected_bernini, expected_veomni = sys.argv[1:]
try:
    bernini_root, _, bernini_revision, veomni_revision = (
        legacy.trainer.validate_source_trees(
            bernini,
            veomni,
            expected_bernini_commit=expected_bernini,
            expected_veomni_commit=expected_veomni,
        )
    )
    inference = legacy.validate_inference_source_files(bernini_root)
except (legacy.trainer.TrainingContractError, legacy.InferenceContractError) as error:
    raise SystemExit(str(error))
value = {
    "bernini_revision": bernini_revision,
    "veomni_revision": veomni_revision,
    "bernini_training_files": dict(legacy.trainer.BERNINI_PINNED_FILE_HASHES),
    "bernini_inference_files": inference,
}
print(legacy.object_sha256(value))
PY
}

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid method source archive"
[[ -d "${source_repository}/.git" ]] || fail "invalid method source repository"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini source root"
[[ -f "${bernini_archive}" && ! -L "${bernini_archive}" ]] || fail "invalid Bernini source archive"
[[ -d "${veomni_root}/.git" ]] || fail "VeOmni root must be a Git checkout"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid base checkpoint"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "canonical dog source is invalid"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid method archive hash"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid method source revision"
[[ "${output_video}" == /*.mp4 ]] || fail "output must be an absolute .mp4 path"
output_name="${output_video##*/}"
[[ "${output_name}" =~ ^[A-Za-z0-9._-]+\.mp4$ ]] || fail "output basename contains unsupported characters"
output_dir="${output_video%/*}"
receipt_name="${receipt_path##*/}"
transaction_id="slurm-${SLURM_JOB_ID}"
output_temporary="${output_dir}/.${output_name%.mp4}.tmp-${transaction_id}.mp4"
receipt_temporary="${output_dir}/.${receipt_name}.tmp-${transaction_id}"
[[ ! -e "${output_video}" && ! -L "${output_video}" ]] || fail "refusing to overwrite output"
[[ ! -e "${receipt_path}" && ! -L "${receipt_path}" ]] || fail "refusing to overwrite receipt"
[[ ! -e "${output_temporary}" && ! -L "${output_temporary}" ]] || fail "stale job-specific video temporary exists"
[[ ! -e "${receipt_temporary}" && ! -L "${receipt_temporary}" ]] || fail "stale job-specific receipt temporary exists"
artifact_paths_validated=1
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive hash mismatch"
[[ "$(sha256sum "${bernini_archive}" | awk '{print $1}')" == "${bernini_archive_sha256}" ]] || fail "Bernini archive hash mismatch"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "canonical dog source hash mismatch"
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
export BERNINI_OUTPUT_TRANSACTION_ID="${transaction_id}"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-source-value-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  local exit_status=$?
  local prefix="${scratch_parent%/}/bernini-source-value-${SLURM_JOB_ID}."
  local cleanup_failed=0
  set +e
  if [[ "${artifact_paths_validated:-0}" == 1 ]]; then
    rm -f -- "${output_temporary}" "${receipt_temporary}" || cleanup_failed=1
    if [[ "${artifact_committed:-0}" != 1 ]]; then
      # Both final paths were required absent before the job.  On failure,
      # remove only this job's incomplete MP4/receipt transaction.
      rm -f -- "${output_video}" "${receipt_path}" || cleanup_failed=1
    fi
  fi
  if [[ -n "${task_scratch:-}" && "${task_scratch}" == "${prefix}"* && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  if [[ -n "${task_scratch:-}" && -e "${task_scratch}" ]]; then
    cleanup_failed=1
  fi
  if [[ "${artifact_paths_validated:-0}" == 1 ]] && {
    [[ -e "${output_temporary}" ]] || [[ -L "${output_temporary}" ]] ||
    [[ -e "${receipt_temporary}" ]] || [[ -L "${receipt_temporary}" ]];
  }; then
    cleanup_failed=1
  fi
  if [[ "${cleanup_failed}" != 0 && "${exit_status}" == 0 ]]; then
    exit_status=2
  fi
  if [[ "${exit_status}" == 0 && "${artifact_committed:-0}" == 1 ]]; then
    echo "[bernini-source-value-oracle] PASS cleanup_verified=true output_sha256=${output_sha_round2:-unknown} receipt_file_sha256=${receipt_sha_round2:-unknown}"
  else
    echo "[bernini-source-value-oracle] EXIT status=${exit_status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  trap - EXIT TERM INT
  exit "${exit_status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

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
export TMPDIR="${task_scratch}"

staged_source_video="${task_scratch}/input/source.mp4"
cp -- "${source_video}" "${staged_source_video}"
[[ -f "${staged_source_video}" && ! -L "${staged_source_video}" ]] || fail "staged source is not a plain file"
[[ "$(sha256sum "${staged_source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "staged dog source digest differs"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "original dog source changed during staging"

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
  infer_source_value_residual_oracle.py \
  source_kv_replay.py \
  source_kv_route_batches.py \
  source_value_residual.py \
  tests/test_infer_source_kv_carrier_oracle.py \
  tests/test_infer_source_value_residual_oracle.py \
  tests/test_source_kv_replay.py \
  tests/test_source_value_residual.py
do
  [[ -f "${method_root}/${required}" ]] || fail "archive lacks ${required}"
done
[[ -f "${method_root}/audits/bernini_r13_ff4c5d4_checkpoint.sha256" ]] || fail "archive lacks checkpoint content manifest"
checkpoint_manifest="${method_root}/audits/bernini_r13_ff4c5d4_checkpoint.sha256"
validate_checkpoint_content "${checkpoint_manifest}"
method_tree_digest_pre="$(method_tree_digest "${task_scratch}/source")"
pinned_source_identity_pre="$(pinned_source_identity)"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-source-value-oracle] source_revision=${source_revision} archive_sha256=${source_archive_sha256}"
echo "[bernini-source-value-oracle] arm=${arm} operator=${operator} selection=${selection} gate=${gate}"
echo "[bernini-source-value-oracle] source=${source_video} sha256=${expected_source_sha256}"
echo "[bernini-source-value-oracle] frames=${num_frames} steps=${num_steps} seed=${seed} Ulysses=4"
echo "[bernini-source-value-oracle] source_tokens=${expected_source_tokens} pair_tokens=${expected_pair_tokens}"
echo "[bernini-source-value-oracle] noop_sha256=${noop_instruction_sha256} historical_o0_mp4_sha256=${historical_o0_mp4_sha256}"
echo "[bernini-source-value-oracle] method_tree_digest_pre=${method_tree_digest_pre} pinned_source_identity_pre=${pinned_source_identity_pre}"
echo "[bernini-source-value-oracle] target=false mask=false flow=false pose=false adapter=false training=0"
echo "[bernini-source-value-oracle] output=${output_video}"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

for pattern in \
  test_source_kv_replay.py \
  test_source_value_residual.py \
  test_infer_source_kv_carrier_oracle.py \
  test_infer_source_value_residual_oracle.py \
  test_infer_lora_contract.py
do
  PYTHONPATH="${method_root}" "${python_bin}" -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${method_root}/infer_source_value_residual_oracle.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --checkpoint-content-manifest "${checkpoint_manifest}" \
  --source-video "${staged_source_video}" \
  --original-source-path "${source_video}" \
  --expected-source-sha256 "${expected_source_sha256}" \
  --instruction "${instruction}" \
  --output "${output_video}" \
  --arm "${arm}" \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}"

receipt_path="${output_video}.receipt.json"
[[ -f "${output_video}" && ! -L "${output_video}" ]] || fail "output video missing"
[[ -f "${receipt_path}" && ! -L "${receipt_path}" ]] || fail "output receipt missing"
output_sha_round1="$(sha256sum "${output_video}" | awk '{print $1}')"
receipt_sha_round1="$(sha256sum "${receipt_path}" | awk '{print $1}')"

audit_receipt() {
  "${python_bin}" - \
    "${receipt_path}" \
    "${arm}" \
    "${operator}" \
    "${selection}" \
    "${gate}" \
    "${layers}" \
    "${residual_per_layer}" \
    "${lookups_per_layer}" \
    "${source_revision}" \
    "${source_archive_sha256}" \
    "${source_video}" \
    "${staged_source_video}" \
    "${expected_source_sha256}" \
    "${checkpoint_content_manifest_sha256}" \
    "${output_sha_round1}" \
    "${historical_o0_mp4_sha256}" \
    "${arm_table_sha256}" <<'PY'
import hashlib
import json
import math
import sys

(
    path,
    arm,
    operator,
    selection,
    gate_text,
    layers_text,
    residual_per_layer_text,
    lookups_per_layer_text,
    revision,
    archive_sha,
    original_source,
    staged_source,
    source_sha,
    checkpoint_manifest_sha,
    actual_output_sha,
    historical_o0_sha,
    arm_table_sha,
) = sys.argv[1:]
gate = float(gate_text)
layers = int(layers_text)
residual_per_layer = int(residual_per_layer_text)
lookups_per_layer = int(lookups_per_layer_text)
indices = list(range(30)) if selection == "all" else list(range(20, 30))

with open(path, encoding="utf-8") as handle:
    receipt = json.load(handle)

weights = receipt.get("weights", {})
optimization = receipt.get("optimization", {})
inputs = receipt.get("input", {})
preprocessing = receipt.get("preprocessing", {})
prompt = receipt.get("prompt_contract", {})
sampling = receipt.get("sampling", {})
checkpoint_content = receipt.get("checkpoint_content_identity", {})
registry = receipt.get("arm_registry", {})
control = receipt.get("causal_control", {})
oracle = receipt.get("oracle", {})
runtime = oracle.get("runtime_execution_certificate", {})
core = oracle.get("rank0_source_value_core_receipt", {})
per_rank = runtime.get("per_rank", [])
z0 = receipt.get("z0_control")
pairing = dict(control.get("pairing_contract", {}))
pairing_digest = pairing.pop("causal_pairing_digest", None)
computed_pairing_digest = hashlib.sha256(
    json.dumps(
        pairing,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()

expected_capture = layers * 40
expected_lookup = layers * lookups_per_layer
expected_residual = layers * residual_per_layer
checks = (
    receipt.get("schema_version") == "bernini-r-1p3b-frozen-source-value-residual-oracle-v10-v1",
    receipt.get("method_source_revision") == revision,
    receipt.get("method_source_archive_sha256") == archive_sha,
    checkpoint_content.get("manifest_sha256_computed") == checkpoint_manifest_sha,
    checkpoint_content.get("manifest_sha256_expected") == checkpoint_manifest_sha,
    checkpoint_content.get("verified_file_count") == 23,
    checkpoint_content.get("every_file_sha256_verified") is True,
    "adapter" not in receipt,
    weights.get("base_checkpoint_loaded") is True,
    weights.get("base_frozen") is True,
    weights.get("adapter_argument_supported") is False,
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
    inputs.get("target_accessed_by_inference") is False,
    inputs.get("paired_target_video_available_to_runner") is False,
    inputs.get("external_mask_or_swept_tube") is False,
    inputs.get("external_tracking_pose_or_trajectory") is False,
    inputs.get("optical_flow") is False,
    inputs.get("reference_image_or_video") is False,
    inputs.get("first_frame_anchor") is False,
    preprocessing.get("source_tokens_from_input_geometry") == 19530,
    preprocessing.get("pair_tokens_from_input_geometry") == 39060,
    prompt.get("semantic_noop_instruction_sha256") == "fb5f23b5b9de175696cff019f035e81eb1ee6a1123db7e3b63afb604b88daf3a",
    sampling.get("num_frames") == 81,
    sampling.get("num_inference_steps") == 40,
    sampling.get("seed") == 2027,
    sampling.get("ulysses_size") == 4,
    registry.get("arm_table_sha256") == arm_table_sha,
    registry.get("allowed_arms") == ["Z0", "F25", "F50", "F100", "FA25", "SN10", "CK10"],
    registry.get("selected", {}).get("arm") == arm,
    registry.get("method_cli_controls") == ["arm"],
    registry.get("operator_scope_gate_override_supported") is False,
    control.get("arm") == arm,
    control.get("operator") == operator,
    control.get("block_selection") == selection,
    math.isclose(float(control.get("fixed_gate", -1)), gate, rel_tol=0, abs_tol=1e-12),
    control.get("arm_excluded_from_pairing_digest") is True,
    pairing_digest == computed_pairing_digest,
    control.get("causal_pairing_digest") == computed_pairing_digest,
    all(name not in pairing for name in ("arm", "operator", "fixed_gate", "block_selection", "output_path")),
    oracle.get("arm") == arm,
    oracle.get("operator") == operator,
    math.isclose(float(oracle.get("fixed_gate", -1)), gate, rel_tol=0, abs_tol=1e-12),
    oracle.get("actual_block_indices") == indices,
    runtime.get("validated") is True,
    runtime.get("all_four_ranks_exact") is True,
    runtime.get("all_rank_generated_latent_exact") is True,
    runtime.get("actual_block_indices") == indices,
    runtime.get("per_rank_capture_calls") == expected_capture,
    runtime.get("per_rank_replay_lookups") == expected_lookup,
    runtime.get("per_rank_residual_varlen_calls") == expected_residual,
    isinstance(runtime.get("all_rank_certificate_digest"), str) and len(runtime["all_rank_certificate_digest"]) == 64,
    isinstance(runtime.get("full_generated_latent_sha256"), str) and len(runtime["full_generated_latent_sha256"]) == 64,
    core.get("operator") == operator,
    math.isclose(float(core.get("fixed_gate", -1)), gate, rel_tol=0, abs_tol=1e-12),
    core.get("block_indices") == indices,
    core.get("trained_parameters") == 0,
    receipt.get("output", {}).get("sha256") == actual_output_sha,
    receipt.get("output", {}).get("frame_count") == 81,
    receipt.get("output", {}).get("height") == 496,
    receipt.get("output", {}).get("width") == 480,
    receipt.get("output", {}).get("generated_latent_sha256") == runtime.get("full_generated_latent_sha256"),
    len(per_rank) == 4,
    sorted(row.get("rank") for row in per_rank) == [0, 1, 2, 3],
    all(row.get("arm") == arm for row in per_rank),
    all(row.get("operator") == operator for row in per_rank),
    all(row.get("actual_installed_block_indices") == indices for row in per_rank),
    all(row.get("rank_local_bank_capture_calls") == expected_capture for row in per_rank),
    all(row.get("rank_local_bank_replay_lookups") == expected_lookup for row in per_rank),
    all(row.get("per_layer_residual_varlen_calls") == residual_per_layer for row in per_rank),
    all(row.get("source_tokens_runtime") == 19530 for row in per_rank),
    all(row.get("pair_tokens_runtime") == 39060 for row in per_rank),
    all(row.get("processor_restore") is True for row in per_rank),
    all(row.get("sampler_hook_restore") is True for row in per_rank),
)
if arm == "Z0":
    checks += (
        z0 is not None,
        z0.get("same_job_official_o0_executed_first") is True,
        z0.get("same_job_generated_latent_byte_exact") is True,
        z0.get("historical_o0_mp4_sha256_binding") == historical_o0_sha,
        runtime.get("z0_same_job_official_byte_exact") is True,
        all(row.get("z0_control", {}).get("byte_exact") is True for row in per_rank),
        all(row.get("z0_control", {}).get("official_o0_latent") == row.get("generated_latent") for row in per_rank),
    )
else:
    checks += (
        z0 is None,
        runtime.get("z0_same_job_official_byte_exact") is False,
        all(row.get("z0_control") is None for row in per_rank),
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
}

audit_receipt || fail "oracle receipt audit round 1 failed"
[[ "$(sha256sum "${output_video}" | awk '{print $1}')" == "${output_sha_round1}" ]] || fail "output changed after audit round 1"
[[ "$(sha256sum "${receipt_path}" | awk '{print $1}')" == "${receipt_sha_round1}" ]] || fail "receipt changed after audit round 1"

output_sha_round2="$(sha256sum "${output_video}" | awk '{print $1}')"
receipt_sha_round2="$(sha256sum "${receipt_path}" | awk '{print $1}')"
[[ "${output_sha_round2}" == "${output_sha_round1}" ]] || fail "output SHA closure differs between rounds"
[[ "${receipt_sha_round2}" == "${receipt_sha_round1}" ]] || fail "receipt SHA closure differs between rounds"
audit_receipt || fail "oracle receipt audit round 2 failed"
[[ "$(sha256sum "${output_video}" | awk '{print $1}')" == "${output_sha_round2}" ]] || fail "output changed after audit round 2"
[[ "$(sha256sum "${receipt_path}" | awk '{print $1}')" == "${receipt_sha_round2}" ]] || fail "receipt changed after audit round 2"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "canonical dog source changed during oracle"
validate_checkpoint_content "${checkpoint_manifest}"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_commit}" ]] || fail "VeOmni commit changed during oracle"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source changed during oracle"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "original method archive changed during oracle"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "private method archive changed during oracle"
[[ "$(sha256sum "${archive_from_revision}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "pre-run revision archive changed during oracle"
archive_after_revision="${task_scratch}/method-after-revision.tar"
git -C "${source_repository}" archive --format=tar \
  -o "${archive_after_revision}" \
  "${source_revision}" methods/bernini_action_editing
[[ "$(sha256sum "${archive_after_revision}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method revision/archive binding changed during oracle"
[[ "$(sha256sum "${bernini_archive}" | awk '{print $1}')" == "${bernini_archive_sha256}" ]] || fail "Bernini source archive changed during oracle"
method_tree_digest_post="$(method_tree_digest "${task_scratch}/source")"
[[ "${method_tree_digest_post}" == "${method_tree_digest_pre}" ]] || fail "executed method source tree changed during oracle"
pinned_source_identity_post="$(pinned_source_identity)"
[[ "${pinned_source_identity_post}" == "${pinned_source_identity_pre}" ]] || fail "Bernini/VeOmni pinned source identity changed during oracle"
[[ ! -e "${output_temporary}" && ! -L "${output_temporary}" ]] || fail "video temporary remains after runner"
[[ ! -e "${receipt_temporary}" && ! -L "${receipt_temporary}" ]] || fail "receipt temporary remains after runner"

echo "[bernini-source-value-oracle] method_tree_digest_post=${method_tree_digest_post} pinned_source_identity_post=${pinned_source_identity_post}"

rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
artifact_committed=1
