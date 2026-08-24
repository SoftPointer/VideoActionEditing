#!/usr/bin/env bash
#SBATCH --job-name=bern-epmc-v12
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=24:00:00
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
source_video="${BERNINI_EPMC_HELDOUT_SOURCE_VIDEO:?set BERNINI_EPMC_HELDOUT_SOURCE_VIDEO}"
instruction="${BERNINI_EPMC_HELDOUT_INSTRUCTION:?set BERNINI_EPMC_HELDOUT_INSTRUCTION}"
prototype_state="${BERNINI_EPMC_PROTOTYPE_STATE:?set BERNINI_EPMC_PROTOTYPE_STATE}"
prototype_receipt="${BERNINI_EPMC_PROTOTYPE_RECEIPT:?set BERNINI_EPMC_PROTOTYPE_RECEIPT}"
prototype_state_sha256="${BERNINI_EPMC_PROTOTYPE_STATE_SHA256:?set BERNINI_EPMC_PROTOTYPE_STATE_SHA256}"
prototype_receipt_sha256="${BERNINI_EPMC_PROTOTYPE_RECEIPT_SHA256:?set BERNINI_EPMC_PROTOTYPE_RECEIPT_SHA256}"
allow_no_go_diagnostic="${BERNINI_EPMC_ALLOW_NO_GO_DIAGNOSTIC:-0}"
output_dir="${BERNINI_EPMC_INFERENCE_OUTPUT_DIR:?set BERNINI_EPMC_INFERENCE_OUTPUT_DIR}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

expected_source_sha256="4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed"
expected_instruction_sha256="105ee8052a0f65d700736a8a25fdf02eb56f1b60d581403c328a8db3d500558c"
bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
checkpoint_content_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
checkpoint_content_file_count=23
bernini_archive_sha256="08a331958560544efb5e842666c236d819bfdc36d34b6eb9c1cdcee1546ae670"
expected_frames=81
expected_fps=25
expected_steps=40
proposal_seed=2027
render_seed=2028

fail() {
  echo "[bernini-epmc-v12] ERROR: $*" >&2
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

method_tree_digest() {
  "${python_bin}" -B - "$1" <<'PY'
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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    rows.append({"path": relative, "sha256": digest.hexdigest()})
payload = json.dumps(
    rows,
    ensure_ascii=True,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
print(hashlib.sha256(payload).hexdigest())
PY
}

for value_name in source_archive source_repository bernini_root bernini_archive veomni_root checkpoint source_video prototype_state prototype_receipt output_dir; do
  value="${!value_name}"
  [[ "${value}" == /* ]] || fail "${value_name} must be an absolute path"
done
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid durable method source archive"
[[ -z "$(find "${source_archive}" -maxdepth 0 -perm /222 -print -quit)" ]] || fail "durable method archive must be read-only"
[[ -d "${source_repository}/.git" && ! -L "${source_repository}" ]] || fail "invalid method source repository"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini source root"
[[ -f "${bernini_archive}" && ! -L "${bernini_archive}" ]] || fail "invalid Bernini source archive"
[[ -d "${veomni_root}/.git" && ! -L "${veomni_root}" ]] || fail "VeOmni must be a Git checkout"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid base checkpoint"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "invalid held-out source video"
[[ -f "${prototype_state}" && ! -L "${prototype_state}" ]] || fail "invalid prototype state"
[[ -f "${prototype_receipt}" && ! -L "${prototype_receipt}" ]] || fail "invalid prototype receipt"
[[ "${prototype_state}" != "${prototype_receipt}" ]] || fail "prototype state and receipt paths must differ"
[[ "${prototype_state}" == *.safetensors ]] || fail "prototype state must use .safetensors"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ -n "${instruction//[[:space:]]/}" ]] || fail "held-out instruction is empty"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid method archive SHA256"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid method source revision"
[[ "${prototype_state_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid prototype state SHA256"
[[ "${prototype_receipt_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid prototype receipt SHA256"
[[ "${prototype_state_sha256}" != "${prototype_receipt_sha256}" ]] || fail "prototype state and receipt hashes must differ"
[[ "${allow_no_go_diagnostic}" == 0 || "${allow_no_go_diagnostic}" == 1 ]] || fail "BERNINI_EPMC_ALLOW_NO_GO_DIAGNOSTIC must be 0 or 1"
[[ "${output_dir}" != / ]] || fail "output directory cannot be filesystem root"
output_name="${output_dir##*/}"
[[ "${output_name}" =~ ^[A-Za-z0-9_-]+$ ]] || fail "output basename contains unsupported characters or a suffix"
canonical_output_dir="$(realpath -m -- "${output_dir}")"
[[ "${canonical_output_dir}" == "${output_dir}" ]] || fail "output directory must be canonical and contain no symlink traversal"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing to reuse output directory"
output_parent="$(dirname -- "${output_dir}")"
mkdir -p -- "${output_parent}"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent must be a non-symlink directory"

[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method source archive hash mismatch"
[[ "$(sha256sum "${bernini_archive}" | awk '{print $1}')" == "${bernini_archive_sha256}" ]] || fail "Bernini source archive hash mismatch"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "held-out source hash mismatch"
[[ "$(sha256sum "${prototype_state}" | awk '{print $1}')" == "${prototype_state_sha256}" ]] || fail "prototype state hash mismatch"
[[ "$(sha256sum "${prototype_receipt}" | awk '{print $1}')" == "${prototype_receipt_sha256}" ]] || fail "prototype receipt hash mismatch"
actual_instruction_sha256="$("${python_bin}" -B - "${instruction}" <<'PY'
import hashlib
import sys
print(hashlib.sha256(sys.argv[1].encode("utf-8")).hexdigest())
PY
)"
[[ "${actual_instruction_sha256}" == "${expected_instruction_sha256}" ]] || fail "held-out instruction hash mismatch"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_commit}" ]] || fail "VeOmni revision mismatch"
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

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "invalid SLURM_JOB_ID"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-epmc-v12-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  local exit_status=$?
  local cleanup_failed=0
  trap - EXIT TERM INT
  if [[ -n "${task_scratch:-}" ]]; then
    case "${task_scratch}" in
      "${scratch_parent%/}/bernini-epmc-v12-${SLURM_JOB_ID}."*) ;;
      *) echo "[bernini-epmc-v12] refusing unsafe scratch cleanup: ${task_scratch}" >&2; cleanup_failed=1 ;;
    esac
    if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
      chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
      rm -rf -- "${task_scratch}" || cleanup_failed=1
    fi
    [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  fi
  if [[ "${cleanup_failed}" != 0 && "${exit_status}" == 0 ]]; then
    exit_status=2
  fi
  if [[ "${exit_status}" == 0 ]]; then
    echo "[bernini-epmc-v12] PASS cleanup_verified=true output_dir=${output_dir}"
  else
    echo "[bernini-epmc-v12] EXIT status=${exit_status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
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
  "${task_scratch}/triton"
export PYTHONPYCACHEPREFIX="${task_scratch}/pycache-prefix"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
export TMPDIR="${task_scratch}"

staged_source_video="${task_scratch}/input/source.mp4"
state_name="${prototype_state##*/}"
receipt_name="${prototype_receipt##*/}"
[[ "${state_name}" =~ ^[A-Za-z0-9._-]+\.safetensors$ ]] || fail "prototype state basename is unsafe"
[[ "${receipt_name}" =~ ^[A-Za-z0-9._-]+\.json$ ]] || fail "prototype receipt basename is unsafe"
staged_prototype_state="${task_scratch}/input/${state_name}"
staged_prototype_receipt="${task_scratch}/input/${receipt_name}"
[[ "${staged_prototype_state}" != "${staged_prototype_receipt}" ]] || fail "staged prototype paths collide"
cp -- "${source_video}" "${staged_source_video}"
cp -- "${prototype_state}" "${staged_prototype_state}"
cp -- "${prototype_receipt}" "${staged_prototype_receipt}"
chmod a-w -- "${staged_source_video}" "${staged_prototype_state}" "${staged_prototype_receipt}"
[[ "$(sha256sum "${staged_source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "staged source hash differs"
[[ "$(sha256sum "${staged_prototype_state}" | awk '{print $1}')" == "${prototype_state_sha256}" ]] || fail "staged prototype state hash differs"
[[ "$(sha256sum "${staged_prototype_receipt}" | awk '{print $1}')" == "${prototype_receipt_sha256}" ]] || fail "staged prototype receipt hash differs"

archive_copy="${task_scratch}/method.tar"
archive_from_revision="${task_scratch}/method-from-revision.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod a-w -- "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "private method archive hash differs"
[[ "$(git -C "${source_repository}" cat-file -t "${source_revision}^{commit}")" == commit ]] || fail "method source revision is not a commit"
git -C "${source_repository}" archive --format=tar -o "${archive_from_revision}" "${source_revision}" methods/bernini_action_editing
[[ "$(sha256sum "${archive_from_revision}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive bytes are not the declared commit"

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
        allowed_parent = path.parts in (("methods",), ("methods", "bernini_action_editing"))
        allowed_child = path.parts[:2] == ("methods", "bernini_action_editing")
        if not (allowed_parent or allowed_child):
            raise SystemExit(f"archive member escaped method subtree: {member.name!r}")
        if member.issym() or member.islnk() or member.isfifo() or member.isdev():
            raise SystemExit(f"unsafe archive member: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported archive member: {member.name!r}")
PY
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
for required in \
  infer_fewshot_motion_code.py \
  counterfactual_proposal_motion_runtime.py \
  counterfactual_proposal_motion_branch.py \
  counterfactual_proposal_motion_rebinding.py \
  fewshot_episode_io.py \
  fewshot_motion_branch.py \
  fewshot_privileged_motion_code.py \
  fewshot_proposal_motion_carrier.py \
  infer_counterfactual_proposal_motion_oracle.py \
  infer_lora.py \
  infer_source_kv_carrier_oracle.py \
  infer_source_value_residual_oracle.py \
  source_kv_route_batches.py \
  source_kv_replay.py \
  source_value_residual.py \
  train_lora.py \
  tools/materialize_vae.py \
  scripts/auh_infer_fewshot_motion_code.sbatch \
  tests/test_infer_fewshot_motion_code_contract.py \
  tests/test_auh_infer_fewshot_motion_code_launcher.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive lacks plain ${required}"
done
[[ -z "$(find "${method_root}" -type l -print -quit)" ]] || fail "extracted method tree contains a symlink"
checkpoint_manifest="${method_root}/audits/bernini_r13_ff4c5d4_checkpoint.sha256"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "archive lacks checkpoint content manifest"
find "${method_root}" -type f -exec chmod a-w -- {} +
chmod a-w -- "${archive_from_revision}"
[[ -z "$(find "${method_root}" -type d -name __pycache__ -print -quit)" ]] || fail "extracted method tree contains pycache"
validate_checkpoint_content "${checkpoint_manifest}"
method_tree_digest_pre="$(method_tree_digest "${method_root}")"

diagnostic_args=()
case "${allow_no_go_diagnostic}" in
  0) ;;
  1) diagnostic_args+=(--allow-no-go-diagnostic) ;;
  *) fail "unreachable diagnostic flag state" ;;
esac

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-epmc-v12] revision=${source_revision} archive_sha256=${source_archive_sha256} method_tree_sha256=${method_tree_digest_pre}"
echo "[bernini-epmc-v12] frames=${expected_frames} fps=${expected_fps} steps=${expected_steps} Ulysses=4 GPUs=4xMI210 proposals=2 arms=5 videos=7"
echo "[bernini-epmc-v12] source_sha256=${expected_source_sha256} instruction_sha256=${expected_instruction_sha256}"
echo "[bernini-epmc-v12] prototype_state_sha256=${prototype_state_sha256} prototype_receipt_sha256=${prototype_receipt_sha256} allow_no_go_diagnostic=${allow_no_go_diagnostic}"
echo "[bernini-epmc-v12] target=false support=false mask=false flow=false pose=false track=false trajectory=false reference=false heldout_oracle=false"
echo "[bernini-epmc-v12] output_dir=${output_dir} output_reuse=false"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

for pattern in \
  test_infer_fewshot_motion_code_contract.py \
  test_auh_infer_fewshot_motion_code_launcher.py
do
  PYTHONPATH="${method_root}" "${python_bin}" -B -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${method_root}/infer_fewshot_motion_code.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --checkpoint-content-manifest "${checkpoint_manifest}" \
  --source-video "${staged_source_video}" \
  --instruction "${instruction}" \
  --prototype-state "${staged_prototype_state}" \
  --prototype-receipt "${staged_prototype_receipt}" \
  --expected-prototype-state-sha256 "${prototype_state_sha256}" \
  --expected-prototype-receipt-sha256 "${prototype_receipt_sha256}" \
  --output-dir "${output_dir}" \
  --expected-source-sha256 "${expected_source_sha256}" \
  --expected-instruction-sha256 "${expected_instruction_sha256}" \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  --num-inference-steps "${expected_steps}" \
  --proposal-seed "${proposal_seed}" \
  --render-seed "${render_seed}" \
  "${diagnostic_args[@]}"

[[ -d "${output_dir}" && ! -L "${output_dir}" ]] || fail "runner output directory missing"
for name in proposal_action proposal_noop B0 Z0 PROTO REVERSE SHUFFLE; do
  [[ -f "${output_dir}/${name}.mp4" && ! -L "${output_dir}/${name}.mp4" ]] || fail "missing plain ${name}.mp4"
done
[[ -f "${output_dir}/receipt.json" && ! -L "${output_dir}/receipt.json" ]] || fail "missing plain receipt.json"
[[ "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 8 ]] || fail "output directory must contain exactly seven videos and one receipt"
[[ -z "$(find "${output_dir}" -mindepth 1 -maxdepth 1 ! -type f -print -quit)" ]] || fail "output directory contains a non-file entry"

"${python_bin}" -B - \
  --receipt "${output_dir}/receipt.json" \
  --output-dir "${output_dir}" \
  --revision "${source_revision}" \
  --archive-sha256 "${source_archive_sha256}" \
  --source "${staged_source_video}" \
  --source-sha256 "${expected_source_sha256}" \
  --instruction-sha256 "${expected_instruction_sha256}" \
  --prototype-state-sha256 "${prototype_state_sha256}" \
  --prototype-receipt-sha256 "${prototype_receipt_sha256}" \
  --checkpoint-manifest-sha256 "${checkpoint_content_manifest_sha256}" \
  --allow-no-go-diagnostic "${allow_no_go_diagnostic}" <<'PY'
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--receipt", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--revision", required=True)
parser.add_argument("--archive-sha256", required=True)
parser.add_argument("--source", required=True)
parser.add_argument("--source-sha256", required=True)
parser.add_argument("--instruction-sha256", required=True)
parser.add_argument("--prototype-state-sha256", required=True)
parser.add_argument("--prototype-receipt-sha256", required=True)
parser.add_argument("--checkpoint-manifest-sha256", required=True)
parser.add_argument("--allow-no-go-diagnostic", choices=("0", "1"), required=True)
args = parser.parse_args()

receipt_path = Path(args.receipt)
output_dir = Path(args.output_dir)
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
expected_videos = (
    "proposal_action",
    "proposal_noop",
    "B0",
    "Z0",
    "PROTO",
    "REVERSE",
    "SHUFFLE",
)
expected_files = {*(f"{name}.mp4" for name in expected_videos), "receipt.json"}
actual_files = {path.name for path in output_dir.iterdir()}
if actual_files != expected_files:
    raise SystemExit("output file set differs")

prototype = receipt.get("prototype", {})
source = receipt.get("source", {})
schedule = receipt.get("schedule", {})
arms = receipt.get("arms", {})
claims = receipt.get("verified_claims", {})
outputs = receipt.get("outputs", {})
checkpoint = receipt.get("checkpoint", {})
runtime = receipt.get("runtime_traces", {})
expected_gate = "NO_GO" if args.allow_no_go_diagnostic == "1" else "GO"
checks = (
    receipt.get("schema_version") == "bernini-epmc-v12-source-instruction-inference-v1",
    receipt.get("method_revision") == args.revision,
    receipt.get("method_archive_sha256") == args.archive_sha256,
    receipt.get("scientific_claim") is False,
    receipt.get("video_quality_claim") is False,
    receipt.get("source_instruction_only_inference") is True,
    receipt.get("semantic_inputs") == ["source_video", "instruction"],
    receipt.get("heldout_oracle_arm_exists") is False,
    receipt.get("heldout_oracle_used") is False,
    receipt.get("representability_gate") == expected_gate,
    receipt.get("diagnostic_only") == (expected_gate == "NO_GO"),
    receipt.get("no_go_diagnostic_override") == (args.allow_no_go_diagnostic == "1"),
    source.get("path") == args.source,
    source.get("sha256") == args.source_sha256,
    receipt.get("instruction_sha256") == args.instruction_sha256,
    prototype.get("state_file_sha256") == args.prototype_state_sha256,
    prototype.get("receipt_file_sha256") == args.prototype_receipt_sha256,
    prototype.get("all_12_heads_byte_exact_tied_per_block") is True,
    prototype.get("source_instruction_only_deployment") is True,
    prototype.get("heldout_oracle_used") is False,
    prototype.get("representability_gate") == expected_gate,
    schedule.get("frames") == 81,
    schedule.get("fps") == 25,
    schedule.get("steps") == 40,
    schedule.get("proposal_action_noop_same_seed") is True,
    schedule.get("all_five_render_arms_same_seed") is True,
    receipt.get("seeds") == {"proposal": 2027, "render": 2028},
    arms.get("order") == ["B0", "Z0", "PROTO", "REVERSE", "SHUFFLE"],
    set(arms.get("codes", {})) == {"Z0", "PROTO", "REVERSE", "SHUFFLE"},
    set(outputs) == set(expected_videos),
    claims.get("z0_full_latent_byte_exact_b0") is True,
    claims.get("all_patched_arms_complete_40_step_binding") is True,
    claims.get("every_output_is_81_frames_25fps") is True,
    claims.get("heldout_oracle_not_used") is True,
    checkpoint.get("manifest_sha256_computed") == args.checkpoint_manifest_sha256,
    checkpoint.get("manifest_sha256_expected") == args.checkpoint_manifest_sha256,
    checkpoint.get("verified_file_count") == 23,
    checkpoint.get("every_file_sha256_verified") is True,
    set(runtime) == {"Z0", "PROTO", "REVERSE", "SHUFFLE"},
    all(item.get("completed_steps") == 40 for item in runtime.values()),
    all(item.get("all_bindings_complete") is True for item in runtime.values()),
    receipt.get("patch", {}).get("restored_after_context") is True,
)
if not all(checks):
    raise SystemExit("inference receipt contract differs")

for name in expected_videos:
    path = output_dir / f"{name}.mp4"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    row = outputs[name]
    if not (
        row.get("path") == str(path)
        and row.get("mp4_sha256") == digest
        and row.get("frames") == 81
        and row.get("fps") == 25
        and row.get("bucket_hw") == [480, 496]
    ):
        raise SystemExit(f"output receipt differs for {name}")

candidate = dict(receipt)
stored = candidate.pop("receipt_digest", None)
payload = json.dumps(
    candidate,
    ensure_ascii=True,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
if stored != hashlib.sha256(payload).hexdigest():
    raise SystemExit("inference receipt digest differs")
PY

method_tree_digest_post="$(method_tree_digest "${method_root}")"
[[ "${method_tree_digest_post}" == "${method_tree_digest_pre}" ]] || fail "extracted method tree changed during inference"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "private method archive changed"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "durable method archive changed"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "held-out source changed"
[[ "$(sha256sum "${staged_source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "staged held-out source changed"
[[ "$(sha256sum "${prototype_state}" | awk '{print $1}')" == "${prototype_state_sha256}" ]] || fail "prototype state changed"
[[ "$(sha256sum "${staged_prototype_state}" | awk '{print $1}')" == "${prototype_state_sha256}" ]] || fail "staged prototype state changed"
[[ "$(sha256sum "${prototype_receipt}" | awk '{print $1}')" == "${prototype_receipt_sha256}" ]] || fail "prototype receipt changed"
[[ "$(sha256sum "${staged_prototype_receipt}" | awk '{print $1}')" == "${prototype_receipt_sha256}" ]] || fail "staged prototype receipt changed"
validate_checkpoint_content "${checkpoint_manifest}"
[[ "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 8 ]] || fail "output file count changed after audit"

rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-epmc-v12] COMPLETE arms=B0,Z0,PROTO,REVERSE,SHUFFLE videos=7 receipt=${output_dir}/receipt.json"
