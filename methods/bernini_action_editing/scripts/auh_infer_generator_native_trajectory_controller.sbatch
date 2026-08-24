#!/usr/bin/env bash
#SBATCH --job-name=bern-egntc-eval
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=04:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_archive="${BERNINI_EGNTC_SOURCE_ARCHIVE:?set BERNINI_EGNTC_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_EGNTC_SOURCE_ARCHIVE_SHA256:?set BERNINI_EGNTC_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_EGNTC_SOURCE_REVISION:?set BERNINI_EGNTC_SOURCE_REVISION}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
bernini_archive="${BERNINI_OFFICIAL_ARCHIVE:?set BERNINI_OFFICIAL_ARCHIVE}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
source_video="${BERNINI_EGNTC_HELDOUT_SOURCE_VIDEO:?set BERNINI_EGNTC_HELDOUT_SOURCE_VIDEO}"
source_video_sha256="${BERNINI_EGNTC_HELDOUT_SOURCE_SHA256:?set BERNINI_EGNTC_HELDOUT_SOURCE_SHA256}"
instruction="${BERNINI_EGNTC_HELDOUT_INSTRUCTION:?set BERNINI_EGNTC_HELDOUT_INSTRUCTION}"
instruction_sha256="${BERNINI_EGNTC_HELDOUT_INSTRUCTION_SHA256:?set BERNINI_EGNTC_HELDOUT_INSTRUCTION_SHA256}"
controller_state="${BERNINI_EGNTC_CONTROLLER_STATE:?set BERNINI_EGNTC_CONTROLLER_STATE}"
controller_receipt="${BERNINI_EGNTC_CONTROLLER_RECEIPT:?set BERNINI_EGNTC_CONTROLLER_RECEIPT}"
controller_state_sha256="${BERNINI_EGNTC_CONTROLLER_STATE_SHA256:?set BERNINI_EGNTC_CONTROLLER_STATE_SHA256}"
controller_receipt_sha256="${BERNINI_EGNTC_CONTROLLER_RECEIPT_SHA256:?set BERNINI_EGNTC_CONTROLLER_RECEIPT_SHA256}"
output_dir="${BERNINI_EGNTC_INFERENCE_OUTPUT_DIR:?set BERNINI_EGNTC_INFERENCE_OUTPUT_DIR}"
python_bin="${BERNINI_EGNTC_PYTHON_BIN:?set BERNINI_EGNTC_PYTHON_BIN}"
inference_seed="${BERNINI_EGNTC_INFERENCE_SEED:-20260807}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
checkpoint_manifest_file_count=23
bernini_archive_sha256="08a331958560544efb5e842666c236d819bfdc36d34b6eb9c1cdcee1546ae670"
ffprobe_helper_sha256="c2739db02e1e3abedebd5b7d84dc253432884dc797d4e16102df8362f158c23f"
expected_pyav_version="13.1.0"
expected_frames=81
expected_fps=25
expected_steps=40

fail() {
  echo "[bernini-egntc-eval] ERROR: $*" >&2
  exit 2
}

require_absolute_canonical() {
  local value="$1"
  local label="$2"
  [[ "${value}" == /* ]] || fail "${label} must be absolute"
  [[ "$(realpath -e -- "${value}")" == "${value}" ]] || fail "${label} must be canonical"
}

validate_checkpoint_content() {
  local manifest="$1"
  local actual_count
  [[ -f "${manifest}" && ! -L "${manifest}" ]] || fail "checkpoint manifest is not plain"
  [[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest hash differs"
  [[ "$(wc -l < "${manifest}")" -eq "${checkpoint_manifest_file_count}" ]] || fail "checkpoint manifest count differs"
  [[ -z "$(find "${checkpoint}" -path "${checkpoint}/.cache" -prune -o -type l -print -quit)" ]] || fail "checkpoint contains a non-cache symlink"
  actual_count="$(find "${checkpoint}" -path "${checkpoint}/.cache" -prune -o -type f -print | wc -l)"
  [[ "${actual_count}" -eq "${checkpoint_manifest_file_count}" ]] || fail "checkpoint file set differs"
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
        for block in iter(lambda: handle.read(1 << 20), b""):
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

for value_name in \
  source_archive bernini_root bernini_archive veomni_root checkpoint \
  source_video controller_state controller_receipt output_dir python_bin
do
  value="${!value_name}"
  [[ "${value}" == /* ]] || fail "${value_name} must be absolute"
done
python_bin="$(realpath -e -- "${python_bin}")"
for value_name in \
  source_archive_sha256 source_video_sha256 instruction_sha256 \
  controller_state_sha256 controller_receipt_sha256
do
  value="${!value_name}"
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || fail "${value_name} must be a lowercase SHA-256"
done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be a full lowercase SHA-1"
[[ "${inference_seed}" =~ ^(0|[1-9][0-9]*)$ ]] || fail "inference seed must be a non-negative integer"
"${python_bin}" -B -c \
  'import sys; value = int(sys.argv[1]); raise SystemExit(0 if value < 2**63 else 2)' \
  "${inference_seed}" || fail "inference seed exceeds signed 63-bit range"
[[ -n "${instruction//[[:space:]]/}" ]] || fail "held-out instruction is empty"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "method source archive is not plain"
[[ "$(stat -c '%A' "${source_archive}")" != *w* ]] || fail "method source archive must be read-only"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini source root is invalid"
[[ -f "${bernini_archive}" && ! -L "${bernini_archive}" ]] || fail "Bernini source archive is invalid"
[[ -d "${veomni_root}/.git" && ! -L "${veomni_root}" ]] || fail "VeOmni source root is invalid"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "base checkpoint is invalid"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "held-out source video is not plain"
[[ -f "${controller_state}" && ! -L "${controller_state}" ]] || fail "controller state is not plain"
[[ -f "${controller_receipt}" && ! -L "${controller_receipt}" ]] || fail "controller receipt is not plain"
[[ "${controller_state}" == *.safetensors ]] || fail "controller state must use .safetensors"
[[ "${controller_state}" != "${controller_receipt}" ]] || fail "controller state and receipt paths collide"
[[ "${controller_state_sha256}" != "${controller_receipt_sha256}" ]] || fail "controller state and receipt hashes collide"
[[ -f "${python_bin}" && -x "${python_bin}" ]] || fail "runtime Python is invalid"

for value_name in source_archive bernini_root bernini_archive veomni_root checkpoint source_video controller_state controller_receipt python_bin; do
  require_absolute_canonical "${!value_name}" "${value_name}"
done

[[ "${output_dir}" != / ]] || fail "output directory cannot be filesystem root"
output_name="${output_dir##*/}"
[[ "${output_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "output directory basename is unsafe"
[[ "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output directory must be canonical"
# The ca1114f runner publishes with os.replace.  These checks provide strict
# preflight non-overwrite isolation, not an atomic create-only transaction.
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing to reuse inference output directory"
output_parent="$(dirname -- "${output_dir}")"
mkdir -p -- "${output_parent}"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent must be a non-symlink directory"
[[ "$(realpath -e -- "${output_parent}")" == "${output_parent}" ]] || fail "output parent must be canonical"
output_path="${output_dir}/prototype.mp4"
output_receipt="${output_path}.receipt.json"
[[ ! -e "${output_path}" && ! -L "${output_path}" ]] || fail "refusing to overwrite output video"
[[ ! -e "${output_receipt}" && ! -L "${output_receipt}" ]] || fail "refusing to overwrite output receipt"

[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "method source archive revision differs"
[[ "$(sha256sum "${bernini_archive}" | awk '{print $1}')" == "${bernini_archive_sha256}" ]] || fail "Bernini source archive hash differs"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${source_video_sha256}" ]] || fail "held-out source video hash differs"
[[ "$(sha256sum "${controller_state}" | awk '{print $1}')" == "${controller_state_sha256}" ]] || fail "controller state hash differs"
[[ "$(sha256sum "${controller_receipt}" | awk '{print $1}')" == "${controller_receipt_sha256}" ]] || fail "controller receipt hash differs"
actual_instruction_sha256="$("${python_bin}" -B - "${instruction}" <<'PY'
import hashlib
import sys

print(hashlib.sha256(sys.argv[1].encode("utf-8")).hexdigest())
PY
)"
[[ "${actual_instruction_sha256}" == "${instruction_sha256}" ]] || fail "held-out instruction hash differs"

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
export BERNINI_EPMC_PYTHON_BIN="${python_bin}"

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "invalid Slurm job ID"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent is invalid"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-egntc-eval-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  local exit_status=$?
  local cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/bernini-egntc-eval-${SLURM_JOB_ID}."*) ;;
    *) echo "[bernini-egntc-eval] refusing unsafe scratch cleanup: ${task_scratch:-unset}" >&2; cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  if [[ "${cleanup_failed}" != 0 && "${exit_status}" == 0 ]]; then
    exit_status=2
  fi
  if [[ "${exit_status}" == 0 ]]; then
    echo "[bernini-egntc-eval] PASS evaluation_only=true deployment=false scientific_claim=false cleanup_verified=true output=${output_path}"
  else
    echo "[bernini-egntc-eval] EXIT status=${exit_status} evaluation_only=true cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${exit_status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p \
  "${task_scratch}/source-tree" \
  "${task_scratch}/input/source" \
  "${task_scratch}/input/controller" \
  "${task_scratch}/bin" \
  "${task_scratch}/pycache" \
  "${task_scratch}/rank-caches"
export PYTHONPYCACHEPREFIX="${task_scratch}/pycache"
export BERNINI_EGNTC_RANK_CACHE_ROOT="${task_scratch}/rank-caches"
export TMPDIR="${task_scratch}"

staged_source_video="${task_scratch}/input/source/source.mp4"
state_name="${controller_state##*/}"
receipt_name="${controller_receipt##*/}"
[[ "${state_name}" =~ ^[A-Za-z0-9._-]+\.safetensors$ ]] || fail "controller state basename is unsafe"
[[ "${receipt_name}" =~ ^[A-Za-z0-9._-]+\.json$ ]] || fail "controller receipt basename is unsafe"
staged_controller_state="${task_scratch}/input/controller/${state_name}"
staged_controller_receipt="${task_scratch}/input/controller/${receipt_name}"
[[ "${staged_controller_state}" != "${staged_controller_receipt}" ]] || fail "staged controller paths collide"
cp -- "${source_video}" "${staged_source_video}"
cp -- "${controller_state}" "${staged_controller_state}"
cp -- "${controller_receipt}" "${staged_controller_receipt}"
chmod 0444 -- "${staged_source_video}" "${staged_controller_state}" "${staged_controller_receipt}"
[[ "$(sha256sum "${staged_source_video}" | awk '{print $1}')" == "${source_video_sha256}" ]] || fail "staged source hash differs"
[[ "$(sha256sum "${staged_controller_state}" | awk '{print $1}')" == "${controller_state_sha256}" ]] || fail "staged controller state hash differs"
[[ "$(sha256sum "${staged_controller_receipt}" | awk '{print $1}')" == "${controller_receipt_sha256}" ]] || fail "staged controller receipt hash differs"

archive_copy="${task_scratch}/method.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod 0444 -- "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged method archive hash differs"
[[ "$(git get-tar-commit-id <"${archive_copy}")" == "${source_revision}" ]] || fail "staged method archive revision differs"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

with tarfile.open(sys.argv[1], mode="r:*") as handle:
    members = handle.getmembers()
    if not members:
        raise SystemExit("method archive is empty")
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
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
for required in \
  generator_native_trajectory_controller.py \
  infer_generator_native_trajectory_controller.py \
  infer_lora.py \
  inference_sigma_strata.py \
  motion_residual.py \
  train_lora.py \
  tri_branch_unipc.py \
  tools/ffprobe_pyav_compat.py \
  tools/materialize_vae.py \
  audits/bernini_r13_ff4c5d4_checkpoint.sha256 \
  scripts/auh_infer_generator_native_trajectory_controller.sbatch \
  tests/test_generator_native_trajectory_controller.py \
  tests/test_tri_branch_unipc.py \
  tests/test_infer_generator_native_trajectory_controller_contract.py \
  tests/test_auh_infer_generator_native_trajectory_controller_launcher.py \
  tests/test_ffprobe_pyav_compat.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive lacks plain ${required}"
done
[[ -z "$(find "${method_root}" -type l -print -quit)" ]] || fail "extracted method tree contains a symlink"

ffprobe_helper="${method_root}/tools/ffprobe_pyav_compat.py"
[[ "$(sha256sum "${ffprobe_helper}" | awk '{print $1}')" == "${ffprobe_helper_sha256}" ]] || fail "source PyAV ffprobe shim hash differs"
ffprobe_bin="${task_scratch}/bin/ffprobe"
[[ ! -e "${ffprobe_bin}" && ! -L "${ffprobe_bin}" ]] || fail "task-private ffprobe destination exists"
cp -- "${ffprobe_helper}" "${ffprobe_bin}"
chmod 0555 -- "${ffprobe_bin}"
[[ "$(sha256sum "${ffprobe_bin}" | awk '{print $1}')" == "${ffprobe_helper_sha256}" ]] || fail "copied PyAV ffprobe shim hash differs"
export PATH="${task_scratch}/bin:${PATH}"
hash -r
[[ "$(command -v ffprobe)" == "${ffprobe_bin}" ]] || fail "task-private PyAV ffprobe shim is not active"
[[ "$(stat -c '%a' "${ffprobe_bin}")" == 555 ]] || fail "PyAV ffprobe shim mode differs"
[[ "$("${BERNINI_EPMC_PYTHON_BIN}" -B -c 'import av; print(av.__version__)')" == "${expected_pyav_version}" ]] || fail "runtime Python must provide pinned PyAV ${expected_pyav_version}"

find "${method_root}" -type f -exec chmod a-w -- {} +
[[ -z "$(find "${method_root}" -type d -name __pycache__ -print -quit)" ]] || fail "extracted method tree contains pycache"
method_tree_digest_pre="$(method_tree_digest "${method_root}")"
checkpoint_manifest="${method_root}/audits/bernini_r13_ff4c5d4_checkpoint.sha256"
validate_checkpoint_content "${checkpoint_manifest}"

PYTHONPATH="${method_root}" "${python_bin}" -B - \
  "${staged_controller_state}" "${staged_controller_receipt}" \
  "${controller_state_sha256}" "${controller_receipt_sha256}" <<'PY'
import sys

import infer_generator_native_trajectory_controller as inference

bundle = inference.load_controller_bundle(
    sys.argv[1],
    sys.argv[2],
    expected_state_sha256=sys.argv[3],
    expected_receipt_sha256=sys.argv[4],
    allow_diagnostic_no_go=True,
)
# In the current v1 schema, formal runs are exactly GO/NO_GO while the smoke
# artifact has the distinct NOT_EVALUATED_ENGINEERING_SMOKE label.  GO and
# NO_GO are both accepted here; the scalar diagnostic does not veto decoding.
if bundle.representability_gate not in ("GO", "NO_GO"):
    raise SystemExit("held-out evaluation requires a formal controller artifact")
if bundle.deployable is not False or bundle.diagnostic_override is not True:
    raise SystemExit("controller evaluation-only policy differs")
print(
    "[bernini-egntc-eval] CONTROLLER_OK "
    f"formal_envelope=true diagnostic={bundle.representability_gate} "
    "evaluation_only=true deployable=false"
)
PY

PYTHONPATH="${method_root}" "${python_bin}" -B - \
  "${bernini_root}" "${veomni_root}" "${bernini_commit}" "${veomni_commit}" <<'PY'
from pathlib import Path
import sys

import train_lora

bernini, veomni, bernini_revision, veomni_revision = train_lora.validate_source_trees(
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    expected_bernini_commit=sys.argv[3],
    expected_veomni_commit=sys.argv[4],
)
if bernini_revision != sys.argv[3] or veomni_revision != sys.argv[4]:
    raise SystemExit("upstream source revision differs")
print(
    "[bernini-egntc-eval] SOURCE_TREES_OK "
    f"bernini={bernini_revision} veomni={veomni_revision}"
)
PY

"${python_bin}" -B - "${ffprobe_bin}" "${staged_source_video}" <<'PY'
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

ffprobe = Path(sys.argv[1]).resolve(strict=True)
video = Path(sys.argv[2]).resolve(strict=True)
result = subprocess.run(
    [
        str(ffprobe),
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames,nb_frames,avg_frame_rate",
        "-of", "json",
        str(video),
    ],
    check=True,
    capture_output=True,
    text=True,
)
payload = json.loads(result.stdout)
streams = payload.get("streams")
if not isinstance(streams, list) or len(streams) != 1:
    raise SystemExit("held-out source must have exactly one selected video stream")
stream = streams[0]
if stream.get("nb_read_frames") != "81" or Fraction(stream.get("avg_frame_rate", "0/1")) != 25:
    raise SystemExit("held-out source is not exact 81 frames at 25 fps")
print("[bernini-egntc-eval] SOURCE_MEDIA_OK frames=81 fps=25 backend=pyav_ffprobe_shim")
PY

echo "[bernini-egntc-eval] host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[bernini-egntc-eval] revision=${source_revision} archive_sha256=${source_archive_sha256} method_tree_sha256=${method_tree_digest_pre}"
echo "[bernini-egntc-eval] world=4 Ulysses=4 GPUs=4xMI210 frames=${expected_frames} fps=${expected_fps} steps=${expected_steps} forwards=120"
echo "[bernini-egntc-eval] source_sha256=${source_video_sha256} instruction_sha256=${instruction_sha256} controller_state_sha256=${controller_state_sha256} controller_receipt_sha256=${controller_receipt_sha256}"
echo "[bernini-egntc-eval] semantic_inputs=source_video,edit_instruction,controller target=false support=false mask=false flow=false pose=false track=false trajectory=false reference=false"
echo "[bernini-egntc-eval] evaluation_only=true diagnostic_override=true deployment=false scientific_claim=false production_claim=false"
echo "[bernini-egntc-eval] output=${output_path} publication_policy=preflight_nonoverwrite_not_atomic_create_only output_reuse=false"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

for pattern in \
  test_generator_native_trajectory_controller.py \
  test_tri_branch_unipc.py \
  test_infer_generator_native_trajectory_controller_contract.py \
  test_ffprobe_pyav_compat.py \
  test_auh_infer_generator_native_trajectory_controller_launcher.py
do
  PYTHONPATH="${method_root}" "${python_bin}" -B -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

[[ "$(command -v ffprobe)" == "${ffprobe_bin}" ]] || fail "PyAV ffprobe shim changed before torchrun"
mkdir -- "${output_dir}"
[[ -d "${output_dir}" && ! -L "${output_dir}" ]] || fail "could not create plain inference output directory"

PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${method_root}/infer_generator_native_trajectory_controller.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --controller-state "${staged_controller_state}" \
  --controller-receipt "${staged_controller_receipt}" \
  --expected-controller-state-sha256 "${controller_state_sha256}" \
  --expected-controller-receipt-sha256 "${controller_receipt_sha256}" \
  --source-video "${staged_source_video}" \
  --instruction "${instruction}" \
  --output "${output_path}" \
  --allow-diagnostic-no-go \
  --num-inference-steps "${expected_steps}" \
  --seed "${inference_seed}" \
  --expected-source-sha256 "${source_video_sha256}" \
  --expected-instruction-sha256 "${instruction_sha256}" \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}"

[[ -f "${output_path}" && ! -L "${output_path}" ]] || fail "inference output mp4 is missing or not plain"
[[ -f "${output_receipt}" && ! -L "${output_receipt}" ]] || fail "inference receipt is missing or not plain"
[[ "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 2 ]] || fail "output directory must contain exactly mp4 plus receipt"
[[ -z "$(find "${output_dir}" -mindepth 1 -maxdepth 1 ! -type f -print -quit)" ]] || fail "output directory contains a non-file entry"

PYTHONPATH="${method_root}" "${python_bin}" -B - \
  --output "${output_path}" \
  --receipt "${output_receipt}" \
  --output-dir "${output_dir}" \
  --source "${staged_source_video}" \
  --source-sha256 "${source_video_sha256}" \
  --instruction-sha256 "${instruction_sha256}" \
  --instruction-bytes "$("${python_bin}" -B -c 'import sys; print(len(sys.argv[1].encode("utf-8")))' "${instruction}")" \
  --controller-state "${staged_controller_state}" \
  --controller-state-sha256 "${controller_state_sha256}" \
  --controller-receipt "${staged_controller_receipt}" \
  --controller-receipt-sha256 "${controller_receipt_sha256}" \
  --method-root "${method_root}" \
  --source-revision "${source_revision}" \
  --source-archive-sha256 "${source_archive_sha256}" \
  --bernini-root "${bernini_root}" \
  --bernini-commit "${bernini_commit}" \
  --veomni-commit "${veomni_commit}" \
  --checkpoint "${checkpoint}" \
  --checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --seed "${inference_seed}" \
  --ffprobe "${ffprobe_bin}" <<'PY'
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import stat
import subprocess


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_nonfinite(value):
    raise ValueError(f"non-finite JSON value: {value}")


def strict_json(path):
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(value, dict):
        raise SystemExit(f"JSON root is not an object: {path}")
    return value


def canonical_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def plain_file(path, label):
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), f"{label} is not plain")


def probe_video(ffprobe, path):
    result = subprocess.run(
        [
            str(ffprobe),
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames,nb_frames,avg_frame_rate",
            "-of", "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams")
    require(isinstance(streams, list) and len(streams) == 1, "media stream count differs")
    stream = streams[0]
    require(stream.get("nb_read_frames") == "81", "decoded output frame count differs")
    require(Fraction(stream.get("avg_frame_rate", "0/1")) == 25, "decoded output fps differs")
    require(stream.get("nb_frames") in ("81", "N/A"), "declared output frame count differs")
    return {"frame_count": 81, "fps": 25}


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--receipt", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--source", required=True)
parser.add_argument("--source-sha256", required=True)
parser.add_argument("--instruction-sha256", required=True)
parser.add_argument("--instruction-bytes", type=int, required=True)
parser.add_argument("--controller-state", required=True)
parser.add_argument("--controller-state-sha256", required=True)
parser.add_argument("--controller-receipt", required=True)
parser.add_argument("--controller-receipt-sha256", required=True)
parser.add_argument("--method-root", required=True)
parser.add_argument("--source-revision", required=True)
parser.add_argument("--source-archive-sha256", required=True)
parser.add_argument("--bernini-root", required=True)
parser.add_argument("--bernini-commit", required=True)
parser.add_argument("--veomni-commit", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--checkpoint-tree-sha256", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--ffprobe", required=True)
args = parser.parse_args()

output = Path(args.output).resolve(strict=True)
receipt_path = Path(args.receipt).resolve(strict=True)
output_dir = Path(args.output_dir).resolve(strict=True)
source = Path(args.source).resolve(strict=True)
controller_state = Path(args.controller_state).resolve(strict=True)
controller_receipt = Path(args.controller_receipt).resolve(strict=True)
method_root = Path(args.method_root).resolve(strict=True)
ffprobe = Path(args.ffprobe).resolve(strict=True)
for path, label in (
    (output, "output mp4"),
    (receipt_path, "inference receipt"),
    (source, "held-out source"),
    (controller_state, "controller state"),
    (controller_receipt, "controller receipt"),
    (ffprobe, "PyAV ffprobe shim"),
):
    plain_file(path, label)
require(
    {path.name for path in output_dir.iterdir()} == {"prototype.mp4", "prototype.mp4.receipt.json"},
    "output closure differs",
)
media = probe_video(ffprobe, output)
receipt = strict_json(receipt_path)

unsigned_receipt = dict(receipt)
stored_receipt_digest = unsigned_receipt.pop("receipt_digest", None)
require(
    stored_receipt_digest == hashlib.sha256(canonical_bytes(unsigned_receipt)).hexdigest(),
    "inference receipt digest differs",
)
expected_method_files = {
    name: file_sha256(method_root / name)
    for name in (
        "infer_generator_native_trajectory_controller.py",
        "generator_native_trajectory_controller.py",
        "tri_branch_unipc.py",
        "infer_lora.py",
        "motion_residual.py",
    )
}
controller = receipt.get("controller_checkpoint", {})
input_contract = receipt.get("input", {})
sampling = receipt.get("sampling", {})
execution = receipt.get("execution_trace", {})
certificate = execution.get("certificate", {})
tri_branch = execution.get("tri_branch", {})
controller_trace = execution.get("controller", {})
controller_core = execution.get("controller_core", {})
output_contract = receipt.get("output", {})
base_checkpoint = receipt.get("base_checkpoint", {})
wan_diffusion = receipt.get("wan_diffusion", {})

require(receipt.get("schema_version") == "bernini-egntc-source-instruction-inference-v1", "inference schema differs")
require(receipt.get("method") == "episodic-generator-native-trajectory-controller-egntc-v1", "method differs")
require(receipt.get("method_source_revision") == args.source_revision, "method revision differs")
require(receipt.get("method_source_archive_sha256") == args.source_archive_sha256, "method archive hash differs")
require(receipt.get("method_files_sha256") == expected_method_files, "executed method file hashes differ")
require(receipt.get("bernini_commit") == args.bernini_commit, "Bernini commit differs")
require(receipt.get("veomni_commit") == args.veomni_commit, "VeOmni commit differs")
require(base_checkpoint.get("path") == args.checkpoint, "base checkpoint path differs")
require(base_checkpoint.get("tree_sha256") == args.checkpoint_tree_sha256, "base checkpoint tree hash differs")
require(base_checkpoint.get("bernini_r_1p3b") is True and base_checkpoint.get("frozen") is True, "base checkpoint contract differs")
require(wan_diffusion.get("validated_at_hook_installation") is True, "Wan hook source was not validated")
require(wan_diffusion.get("sha256") == "59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512", "Wan diffusion hash differs")

require(controller.get("state_path") == str(controller_state), "controller state path differs")
require(controller.get("receipt_path") == str(controller_receipt), "controller receipt path differs")
require(controller.get("state_file_sha256") == args.controller_state_sha256, "controller state binding differs")
require(controller.get("receipt_file_sha256") == args.controller_receipt_sha256, "controller receipt binding differs")
require(controller.get("state_schema_version") == "bernini-egntc-controller-state-v1", "controller state schema differs")
require(controller.get("training_receipt_schema_version") == "bernini-egntc-controller-training-artifact-receipt-v1", "controller training receipt schema differs")
require(controller.get("tensor_key") == "controller_raw_36d" and controller.get("dimension") == 36, "controller tensor contract differs")
require(controller.get("representability_gate") in ("GO", "NO_GO"), "formal controller diagnostic label differs")
require(controller.get("deployable") is False, "controller unexpectedly claims deployability")
require(controller.get("diagnostic_override") is True, "explicit diagnostic override is absent")

require(input_contract.get("source_video_path") == str(source), "source path binding differs")
require(input_contract.get("source_video_sha256") == args.source_sha256, "source hash binding differs")
require(input_contract.get("instruction_utf8_sha256") == args.instruction_sha256, "instruction hash binding differs")
require(input_contract.get("instruction_utf8_bytes") == args.instruction_bytes, "instruction byte count differs")
require(input_contract.get("accepted_external_conditions") == ["source_video", "edit_instruction"], "semantic input closure differs")
for field in (
    "target_video_argument",
    "target_accessed_by_inference",
    "support_accessed_by_inference",
    "external_mask_flow_pose_track_trajectory",
    "first_frame_anchor",
):
    require(input_contract.get(field) is False, f"forbidden inference condition active: {field}")

require(sampling.get("num_frames") == 81, "sampling frame count differs")
require(sampling.get("num_inference_steps") == 40, "sampling step count differs")
require(sampling.get("seed") == args.seed, "sampling seed differs")
require(sampling.get("guidance_mode") == "v2v_apg" and sampling.get("flow_shift") == 5.0, "sampling guidance differs")
require(sampling.get("omega_txt") == 4.0 and sampling.get("eta") == 0.5, "sampling APG differs")
require(sampling.get("single_expert") == "transformer_1", "single-expert contract differs")
require(sampling.get("ulysses_size") == 4, "Ulysses size differs")
require(sampling.get("rank0_decode_and_save_only") is True, "rank-zero output contract differs")
require(sampling.get("custom_integrator") is False, "custom integrator unexpectedly active")

require(certificate.get("step_count") == 40, "execution certificate step count differs")
require(certificate.get("official_action_apg_exact_steps") == 40, "official APG parity count differs")
require(certificate.get("original_unipc_calls") == 40, "UniPC scheduler call count differs")
require(certificate.get("transformer_forwards") == 120, "transformer forward count differs")
require(certificate.get("controller_callback_calls") == 40, "controller callback count differs")
require(certificate.get("custom_integrator") is False, "execution certificate reports custom integrator")
tri_steps = tri_branch.get("steps")
require(tri_branch.get("sample_calls") == 1 and tri_branch.get("step_count") == 40, "tri-branch trace envelope differs")
require(isinstance(tri_steps, list) and len(tri_steps) == 40, "tri-branch steps differ")
require(
    sum(step.get("transformer_forwards", -1) for step in tri_steps) == 120
    and all(
        step.get("step_index") == index
        and step.get("transformer_forwards") == 3
        and step.get("shared_negative_forwards") == 1
        and step.get("action_forwards") == 1
        and step.get("noop_forwards") == 1
        and step.get("original_scheduler_calls") == 1
        and step.get("official_action_exact_parity") is True
        for index, step in enumerate(tri_steps)
    ),
    "per-step tri-branch execution differs",
)
require(controller_trace.get("step_count") == 40 and len(controller_trace.get("steps", [])) == 40, "controller trace differs")
require(all(step.get("parity_bypass_byte_exact") is True for step in controller_trace.get("steps", [])), "controller parity bypass differs")
core_state = controller_core.get("state", {})
require(core_state.get("completed") is True and core_state.get("step_count") == 40, "controller core completion differs")
require(len(controller_core.get("steps", [])) == 40, "controller core step trace differs")
unsigned_trace = dict(execution)
stored_trace_digest = unsigned_trace.pop("trace_digest", None)
require(stored_trace_digest == hashlib.sha256(canonical_bytes(unsigned_trace)).hexdigest(), "execution trace digest differs")

require(output_contract.get("path") == str(output), "output path binding differs")
require(output_contract.get("sha256") == file_sha256(output), "output hash binding differs")
require(output_contract.get("frame_count") == media["frame_count"], "receipt/output frame count differs")
require(output_contract.get("fps") == float(media["fps"]), "receipt/output fps differs")
require(type(output_contract.get("height")) is int and output_contract["height"] > 0, "output height differs")
require(type(output_contract.get("width")) is int and output_contract["width"] > 0, "output width differs")
require(output_contract.get("audio_preserved") is False, "audio contract differs")
require(receipt.get("diagnostic_only") is True, "evaluation artifact is not diagnostic-labelled")
require(receipt.get("deployable_output") is False, "evaluation output claims deployability")
require(receipt.get("scientific_claim_authorized") is False, "evaluation output claims scientific authorization")
require(receipt.get("production_claim_forbidden") is True, "production claim is not forbidden")

print(
    json.dumps(
        {
            "event": "EGNTC_HELDOUT_STRONG_AUDIT_OK",
            "evaluation_only": True,
            "deployment": False,
            "scientific_claim": False,
            "source_instruction_controller_only": True,
            "frames": 81,
            "fps": 25,
            "unipc_steps": 40,
            "transformer_forwards": 120,
            "output_sha256": file_sha256(output),
            "receipt_sha256": file_sha256(receipt_path),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY

method_tree_digest_post="$(method_tree_digest "${method_root}")"
[[ "${method_tree_digest_post}" == "${method_tree_digest_pre}" ]] || fail "extracted method tree changed during inference"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged method archive changed"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "durable method archive changed"
[[ "$(sha256sum "${bernini_archive}" | awk '{print $1}')" == "${bernini_archive_sha256}" ]] || fail "Bernini source archive changed"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${source_video_sha256}" ]] || fail "held-out source video changed"
[[ "$(sha256sum "${staged_source_video}" | awk '{print $1}')" == "${source_video_sha256}" ]] || fail "staged held-out source changed"
[[ "$(sha256sum "${controller_state}" | awk '{print $1}')" == "${controller_state_sha256}" ]] || fail "controller state changed"
[[ "$(sha256sum "${staged_controller_state}" | awk '{print $1}')" == "${controller_state_sha256}" ]] || fail "staged controller state changed"
[[ "$(sha256sum "${controller_receipt}" | awk '{print $1}')" == "${controller_receipt_sha256}" ]] || fail "controller receipt changed"
[[ "$(sha256sum "${staged_controller_receipt}" | awk '{print $1}')" == "${controller_receipt_sha256}" ]] || fail "staged controller receipt changed"
[[ "$(command -v ffprobe)" == "${ffprobe_bin}" ]] || fail "PyAV ffprobe shim changed after audit"
[[ "$(sha256sum "${ffprobe_bin}" | awk '{print $1}')" == "${ffprobe_helper_sha256}" ]] || fail "PyAV ffprobe shim bytes changed"
validate_checkpoint_content "${checkpoint_manifest}"
[[ "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 2 ]] || fail "audited output closure changed"

rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-egntc-eval] COMPLETE evaluation_only=true deployment=false scientific_claim=false video=${output_path} receipt=${output_receipt}"
