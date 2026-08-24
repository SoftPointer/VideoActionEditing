#!/usr/bin/env bash
#SBATCH --job-name=bern-sac-oracle
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
arm="${BERNINI_SOURCE_ALIGNED_ARM:?set BERNINI_SOURCE_ALIGNED_ARM to C0, K1, or SGA5}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

source_video="/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/288545b9c031491a/source.mp4"
instruction="Make the dog pick up the bone and hold it in its mouth."
expected_source_sha256="5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
checkpoint_content_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
checkpoint_content_file_count=23
bernini_archive_sha256="08a331958560544efb5e842666c236d819bfdc36d34b6eb9c1cdcee1546ae670"
arm_table_sha256="f1d7f6f8c9c33e8e8f2eacc9f9d99e03deb7cc0bf2ad749edaabf79c723d5851"
receipt_path="${output_video}.receipt.json"
artifact_paths_validated=0
artifact_committed=0

fail() {
  echo "[bernini-sac-oracle] ERROR: $*" >&2
  exit 2
}

case "${arm}" in
  C0)
    motion_scale="0.0"; sga_steps=0; expected_calls=0; expected_draws=0; candidate_mode="identity"
    ;;
  K1)
    motion_scale="1.0"; sga_steps=0; expected_calls=80; expected_draws=40; candidate_mode="k1"
    ;;
  SGA5)
    motion_scale="1.0"; sga_steps=3; expected_calls=104; expected_draws=52; candidate_mode="sga5"
    ;;
  *) fail "arm must be exactly C0, K1, or SGA5" ;;
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

root = Path(sys.argv[1])
if root.is_symlink():
    raise SystemExit("method root is a symlink")
root = root.resolve(strict=True)
rows = []
for path in sorted(root.rglob("*")):
    relative = path.relative_to(root).as_posix()
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        raise SystemExit(f"method source contains symlink: {relative}")
    if stat.S_ISDIR(mode):
        if path.name == "__pycache__":
            raise SystemExit("method source contains __pycache__")
        continue
    if not stat.S_ISREG(mode):
        raise SystemExit(f"method source contains non-regular entry: {relative}")
    if path.suffix in (".pyc", ".pyo"):
        raise SystemExit(f"method source contains bytecode: {relative}")
    if mode & 0o222:
        raise SystemExit(f"method source file is writable: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    rows.append({"path": relative, "sha256": digest.hexdigest()})
payload = json.dumps(
    rows,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
print(hashlib.sha256(payload).hexdigest())
PY
}

pinned_source_identity() {
  PYTHONPATH="${method_root}" "${python_bin}" -B - \
    "${bernini_root}" "${veomni_root}" "${bernini_commit}" "${veomni_commit}" <<'PY'
import sys
import infer_lora as legacy

bernini, veomni, expected_bernini, expected_veomni = sys.argv[1:]
try:
    root, _, bernini_revision, veomni_revision = legacy.trainer.validate_source_trees(
        bernini,
        veomni,
        expected_bernini_commit=expected_bernini,
        expected_veomni_commit=expected_veomni,
    )
    inference = legacy.validate_inference_source_files(root)
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
[[ -d "${source_repository}/.git" && ! -L "${source_repository}" ]] || fail "invalid method source repository"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini source root"
[[ -f "${bernini_archive}" && ! -L "${bernini_archive}" ]] || fail "invalid Bernini source archive"
[[ -d "${veomni_root}/.git" && ! -L "${veomni_root}" ]] || fail "VeOmni must be a Git checkout"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid base checkpoint"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "canonical dog source is invalid"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid method archive hash"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid method revision"
[[ "${output_video}" == /*.mp4 ]] || fail "output must be an absolute .mp4 path"
output_name="${output_video##*/}"
[[ "${output_name}" =~ ^[A-Za-z0-9._-]+\.mp4$ ]] || fail "output basename contains unsupported characters"
output_dir="${output_video%/*}"
transaction_id="slurm-${SLURM_JOB_ID}"
output_temporary="${output_dir}/.${output_name%.mp4}.tmp-${transaction_id}.mp4"
receipt_name="${receipt_path##*/}"
receipt_temporary="${output_dir}/.${receipt_name}.tmp-${transaction_id}"
[[ ! -e "${output_video}" && ! -L "${output_video}" ]] || fail "refusing to overwrite output"
[[ ! -e "${receipt_path}" && ! -L "${receipt_path}" ]] || fail "refusing to overwrite receipt"
[[ ! -e "${output_temporary}" && ! -L "${output_temporary}" ]] || fail "stale video temporary exists"
[[ ! -e "${receipt_temporary}" && ! -L "${receipt_temporary}" ]] || fail "stale receipt temporary exists"
artifact_paths_validated=1
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive hash mismatch"
[[ "$(sha256sum "${bernini_archive}" | awk '{print $1}')" == "${bernini_archive_sha256}" ]] || fail "Bernini archive hash mismatch"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "canonical source hash mismatch"
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
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-sac-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  local exit_status=$?
  local prefix="${scratch_parent%/}/bernini-sac-${SLURM_JOB_ID}."
  local cleanup_failed=0
  set +e
  if [[ "${artifact_paths_validated:-0}" == 1 ]]; then
    rm -f -- "${output_temporary}" "${receipt_temporary}" || cleanup_failed=1
    if [[ "${artifact_committed:-0}" != 1 ]]; then
      rm -f -- "${output_video}" "${receipt_path}" || cleanup_failed=1
    fi
  fi
  if [[ -n "${task_scratch:-}" && "${task_scratch}" == "${prefix}"* && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" 2>/dev/null || true
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  if [[ -n "${task_scratch:-}" && -e "${task_scratch}" ]]; then
    cleanup_failed=1
  fi
  if [[ "${cleanup_failed}" != 0 && "${exit_status}" == 0 ]]; then
    exit_status=2
  fi
  if [[ "${exit_status}" == 0 && "${artifact_committed:-0}" == 1 ]]; then
    echo "[bernini-sac-oracle] PASS cleanup_verified=true output_sha256=${output_sha_round2:-unknown} receipt_file_sha256=${receipt_sha_round2:-unknown}"
  else
    echo "[bernini-sac-oracle] EXIT status=${exit_status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  trap - EXIT TERM INT
  exit "${exit_status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p \
  "${task_scratch}/source" \
  "${task_scratch}/input" \
  "${task_scratch}/pycache-prefix" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton" \
  "${output_dir}"
export PYTHONPYCACHEPREFIX="${task_scratch}/pycache-prefix"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
export TMPDIR="${task_scratch}"

staged_source_video="${task_scratch}/input/source.mp4"
cp -- "${source_video}" "${staged_source_video}"
chmod a-w -- "${staged_source_video}"
[[ -f "${staged_source_video}" && ! -L "${staged_source_video}" ]] || fail "staged source is not plain"
[[ "$(sha256sum "${staged_source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "staged source digest differs"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "original source changed during staging"

archive_copy="${task_scratch}/method.tar"
archive_from_revision="${task_scratch}/method-from-revision.tar"
cp -- "${source_archive}" "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "private method archive differs"
[[ "$(git -C "${source_repository}" cat-file -t "${source_revision}^{commit}")" == commit ]] || fail "method revision is not a commit"
git -C "${source_repository}" archive --format=tar -o "${archive_from_revision}" "${source_revision}" methods/bernini_action_editing
[[ "$(sha256sum "${archive_from_revision}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive is not the declared Git revision"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

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
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
for required in \
  infer_source_aligned_controller_oracle.py \
  source_aligned_controller.py \
  differential_sampler.py \
  infer_lora.py \
  train_lora.py \
  tools/materialize_vae.py \
  tests/test_source_aligned_controller.py \
  tests/test_infer_source_aligned_controller_oracle.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive lacks ${required}"
done
checkpoint_manifest="${method_root}/audits/bernini_r13_ff4c5d4_checkpoint.sha256"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "archive lacks checkpoint manifest"
find "${method_root}" -type f -exec chmod a-w -- {} +
chmod a-w -- "${archive_copy}" "${archive_from_revision}"
[[ -z "$(find "${method_root}" -type d -name __pycache__ -print -quit)" ]] || fail "private method tree contains pycache"
validate_checkpoint_content "${checkpoint_manifest}"
method_tree_digest_pre="$(method_tree_digest "${method_root}")"
pinned_source_identity_pre="$(pinned_source_identity)"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-sac-oracle] revision=${source_revision} archive_sha256=${source_archive_sha256}"
echo "[bernini-sac-oracle] arm=${arm} motion_scale=${motion_scale} sga_steps=${sga_steps} configured_candidates=5 candidate_mode=${candidate_mode}"
echo "[bernini-sac-oracle] expected_calls=${expected_calls} expected_draws=${expected_draws}"
echo "[bernini-sac-oracle] frames=81 fps=25 steps=40 seed=2027 shift=5 Ulysses=4"
echo "[bernini-sac-oracle] source=${source_video} sha256=${expected_source_sha256}"
echo "[bernini-sac-oracle] target=false mask=false flow=false pose=false track=false trajectory=false first_frame_anchor=false adapter=false"
echo "[bernini-sac-oracle] method_tree_digest_pre=${method_tree_digest_pre} pinned_source_identity_pre=${pinned_source_identity_pre}"
echo "[bernini-sac-oracle] output=${output_video}"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

for pattern in \
  test_differential_sampler_contract.py \
  test_source_aligned_controller.py \
  test_infer_lora_contract.py \
  test_infer_source_aligned_controller_oracle.py
do
  PYTHONPATH="${method_root}" "${python_bin}" -B -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

"${python_bin}" -B -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${method_root}/infer_source_aligned_controller_oracle.py" \
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
  --method-source-archive "${archive_copy}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  --method-source-tree-sha256 "${method_tree_digest_pre}"

[[ -f "${output_video}" && ! -L "${output_video}" ]] || fail "output video missing"
[[ -f "${receipt_path}" && ! -L "${receipt_path}" ]] || fail "output receipt missing"
output_sha_round1="$(sha256sum "${output_video}" | awk '{print $1}')"
receipt_sha_round1="$(sha256sum "${receipt_path}" | awk '{print $1}')"

audit_receipt() {
  "${python_bin}" -B - \
    "${receipt_path}" "${arm}" "${motion_scale}" "${sga_steps}" \
    "${expected_calls}" "${expected_draws}" "${candidate_mode}" \
    "${source_revision}" "${source_archive_sha256}" "${method_tree_digest_pre}" \
    "${source_video}" "${staged_source_video}" "${expected_source_sha256}" \
    "${output_video}" "${output_sha_round1}" "${arm_table_sha256}" <<'PY'
import hashlib
import json
import math
import sys

(
    receipt_path,
    arm,
    motion_text,
    sga_steps_text,
    calls_text,
    draws_text,
    candidate_mode,
    revision,
    archive_sha,
    method_tree_sha,
    original_source,
    staged_source,
    source_sha,
    output_path,
    output_sha,
    arm_table_sha,
) = sys.argv[1:]
motion = float(motion_text)
sga_steps = int(sga_steps_text)
calls = int(calls_text)
draws = int(draws_text)
with open(receipt_path, encoding="utf-8") as handle:
    receipt = json.load(handle)

method = receipt.get("method_provenance", {})
pre = method.get("pre", {})
post = method.get("post", {})
model = receipt.get("model_provenance", {})
checkpoint = model.get("checkpoint_content", {})
registry = receipt.get("arm_registry", {})
selected = registry.get("selected", {})
weights = receipt.get("weights", {})
optimization = receipt.get("optimization", {})
inputs = receipt.get("input", {})
preprocessing = receipt.get("preprocessing", {})
prompt = receipt.get("prompt_contract", {})
sampling = receipt.get("sampling", {})
config = sampling.get("controller_config", {})
contract = sampling.get("controller_contract", {})
runtime = sampling.get("runtime_execution_certificate", {})
trace_validation = runtime.get("trace_validation", {})
per_rank = runtime.get("per_rank", [])
c0 = receipt.get("c0_control")
expected_counts = [] if candidate_mode == "identity" else (
    [5, 5, 5] + [1] * 37 if candidate_mode == "sga5" else [1] * 40
)

checks = (
    receipt.get("schema_version") == "bernini-r-1p3b-source-aligned-controller-oracle-v1",
    receipt.get("official_dynaedit_reproduction_claimed") is False,
    "not_official_reproduction" in receipt.get("method_status", ""),
    method.get("pre_post_exact") is True,
    pre == post,
    pre.get("revision") == revision,
    pre.get("archive_sha256") == archive_sha,
    pre.get("runtime_tree", {}).get("tree_sha256") == method_tree_sha,
    pre.get("runtime_tree", {}).get("all_plain_read_only") is True,
    pre.get("runtime_tree", {}).get("bytecode_absent") is True,
    pre.get("bytecode_policy", {}).get("dont_write_bytecode") is True,
    model.get("bernini_commit") == "2d2b4591ac053ec25c6371b01a5a6746679e5793",
    model.get("veomni_commit") == "f90b3dc6fbb0ce693745223cc7a94064123dbf4d",
    model.get("checkpoint_tree_sha256") == "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca",
    checkpoint.get("manifest_sha256_computed") == "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
    checkpoint.get("manifest_sha256_expected") == "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
    checkpoint.get("verified_file_count") == 23,
    checkpoint.get("every_file_sha256_verified") is True,
    registry.get("allowed_arms") == ["C0", "K1", "SGA5"],
    registry.get("arm_table_sha256") == arm_table_sha,
    registry.get("method_cli_controls") == ["arm"],
    registry.get("arbitrary_hyperparameter_cli_supported") is False,
    selected.get("arm") == arm,
    math.isclose(float(selected.get("motion_scale", -1)), motion, rel_tol=0, abs_tol=1e-12),
    selected.get("sga_steps") == sga_steps,
    selected.get("configured_sga_candidates") == 5,
    selected.get("expected_shared_step_calls") == calls,
    selected.get("expected_fresh_noise_draws") == draws,
    "adapter" not in receipt,
    weights.get("base_checkpoint_loaded") is True,
    weights.get("base_frozen") is True,
    weights.get("adapter_argument_supported") is False,
    weights.get("adapter_loaded") is False,
    weights.get("lora_module_count") == 0,
    optimization == {"zero_training": True, "training_steps": 0, "optimizer_constructed": False, "backward_calls": 0},
    inputs.get("accepted_external_conditions") == ["source_video", "edit_instruction"],
    inputs.get("source_video_path") == original_source,
    inputs.get("staged_source_video_path") == staged_source,
    inputs.get("source_video_sha256") == source_sha,
    inputs.get("target_video_argument") is False,
    inputs.get("target_video_accessed") is False,
    all(inputs.get(name) is False for name in ("mask", "optical_flow", "pose", "track", "trajectory", "swept_tube", "first_frame_anchor")),
    preprocessing.get("frame_count") == 81,
    math.isclose(float(preprocessing.get("fps", -1)), 25.0, rel_tol=0, abs_tol=1e-9),
    preprocessing.get("source_derived_bucket_hw") == [496, 480],
    prompt.get("action_prompt_uses_official_mv2v_training_path") is True,
    prompt.get("noop_is_internal_fixed_semantic_instruction") is True,
    prompt.get("noop_instruction_sha256") == "fb5f23b5b9de175696cff019f035e81eb1ee6a1123db7e3b63afb604b88daf3a",
    isinstance(prompt.get("action_full_prompt_utf8_sha256"), str) and len(prompt["action_full_prompt_utf8_sha256"]) == 64,
    isinstance(prompt.get("noop_full_prompt_utf8_sha256"), str) and len(prompt["noop_full_prompt_utf8_sha256"]) == 64,
    sampling.get("num_frames") == 81,
    math.isclose(float(sampling.get("fps", -1)), 25.0, rel_tol=0, abs_tol=1e-9),
    sampling.get("num_inference_steps") == 40,
    sampling.get("seed") == 2027,
    math.isclose(float(sampling.get("flow_shift", -1)), 5.0, rel_tol=0, abs_tol=1e-12),
    sampling.get("ulysses_size") == 4,
    config.get("num_inference_steps") == 40,
    config.get("seed") == 2027,
    math.isclose(float(config.get("motion_scale", -1)), motion, rel_tol=0, abs_tol=1e-12),
    config.get("sga_steps") == sga_steps,
    config.get("sga_candidates") == 5,
    contract.get("user_inputs") == ["source_video", "edit_instruction"],
    "not_official_reproduction" in contract.get("status", ""),
    runtime.get("validated") is True,
    runtime.get("all_four_ranks_exact") is True,
    runtime.get("all_rank_source_latent_exact") is True,
    runtime.get("all_rank_prompt_embeddings_exact") is True,
    runtime.get("all_rank_generated_latent_exact") is True,
    runtime.get("all_rank_controller_trace_exact") is True,
    trace_validation.get("validated") is True,
    trace_validation.get("shared_step_calls") == calls,
    trace_validation.get("fresh_noise_draws") == draws,
    trace_validation.get("effective_candidate_counts") == expected_counts,
    runtime.get("c0_source_latent_byte_exact") is (arm == "C0"),
    len(per_rank) == 4,
    sorted(row.get("rank") for row in per_rank) == [0, 1, 2, 3],
    all(row.get("arm") == arm for row in per_rank),
    all(row.get("shared_step_audit_restored") is True for row in per_rank),
    all(row.get("trace_validation", {}).get("effective_candidate_counts") == expected_counts for row in per_rank),
    all(row.get("identity_object_reused") is (arm == "C0") for row in per_rank),
    receipt.get("output", {}).get("path") == output_path,
    receipt.get("output", {}).get("sha256") == output_sha,
    receipt.get("output", {}).get("frame_count") == 81,
    receipt.get("output", {}).get("fps") == 25.0,
    receipt.get("output", {}).get("height") == 496,
    receipt.get("output", {}).get("width") == 480,
    receipt.get("output", {}).get("all_rank_generated_latent_exact") is True,
    receipt.get("production_claim_forbidden") is True,
    receipt.get("scientific_claim_authorized") is False,
)
if arm == "C0":
    checks += (
        isinstance(c0, dict),
        c0.get("source_latent_object_reused") is True,
        c0.get("source_latent_bytes_exact") is True,
        c0.get("operation") == "source_vae_encode_then_decode_only",
        c0.get("source_mp4_byte_copy_claimed") is False,
    )
else:
    checks += (c0 is None,)
candidate = dict(receipt)
stored = candidate.pop("receipt_digest", None)
computed = hashlib.sha256(
    json.dumps(candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
).hexdigest()
checks += (stored == computed,)
raise SystemExit(0 if all(checks) else 1)
PY
}

audit_receipt || fail "receipt audit round 1 failed"
[[ "$(sha256sum "${output_video}" | awk '{print $1}')" == "${output_sha_round1}" ]] || fail "output changed after audit round 1"
[[ "$(sha256sum "${receipt_path}" | awk '{print $1}')" == "${receipt_sha_round1}" ]] || fail "receipt changed after audit round 1"
output_sha_round2="$(sha256sum "${output_video}" | awk '{print $1}')"
receipt_sha_round2="$(sha256sum "${receipt_path}" | awk '{print $1}')"
[[ "${output_sha_round2}" == "${output_sha_round1}" ]] || fail "output hash closure differs"
[[ "${receipt_sha_round2}" == "${receipt_sha_round1}" ]] || fail "receipt hash closure differs"
audit_receipt || fail "receipt audit round 2 failed"
[[ "$(sha256sum "${output_video}" | awk '{print $1}')" == "${output_sha_round2}" ]] || fail "output changed after audit round 2"
[[ "$(sha256sum "${receipt_path}" | awk '{print $1}')" == "${receipt_sha_round2}" ]] || fail "receipt changed after audit round 2"

[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "canonical source changed"
[[ "$(sha256sum "${staged_source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "staged source changed"
validate_checkpoint_content "${checkpoint_manifest}"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_commit}" ]] || fail "VeOmni commit changed"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source changed"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "original method archive changed"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "private method archive changed"
[[ "$(sha256sum "${archive_from_revision}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "revision archive changed"
archive_after_revision="${task_scratch}/method-after-revision.tar"
git -C "${source_repository}" archive --format=tar -o "${archive_after_revision}" "${source_revision}" methods/bernini_action_editing
[[ "$(sha256sum "${archive_after_revision}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method revision/archive binding changed"
[[ "$(sha256sum "${bernini_archive}" | awk '{print $1}')" == "${bernini_archive_sha256}" ]] || fail "Bernini archive changed"
method_tree_digest_post="$(method_tree_digest "${method_root}")"
[[ "${method_tree_digest_post}" == "${method_tree_digest_pre}" ]] || fail "executed method tree changed"
pinned_source_identity_post="$(pinned_source_identity)"
[[ "${pinned_source_identity_post}" == "${pinned_source_identity_pre}" ]] || fail "Bernini/VeOmni source identity changed"
[[ -z "$(find "${task_scratch}/pycache-prefix" -mindepth 1 -print -quit)" ]] || fail "private pycache prefix is not empty"
[[ ! -e "${output_temporary}" && ! -L "${output_temporary}" ]] || fail "video temporary remains"
[[ ! -e "${receipt_temporary}" && ! -L "${receipt_temporary}" ]] || fail "receipt temporary remains"

echo "[bernini-sac-oracle] method_tree_digest_post=${method_tree_digest_post} pinned_source_identity_post=${pinned_source_identity_post}"
rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
artifact_committed=1
