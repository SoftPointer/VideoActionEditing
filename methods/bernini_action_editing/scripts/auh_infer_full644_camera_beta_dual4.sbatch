#!/usr/bin/env bash
#SBATCH --job-name=bern-cam-beta2x4
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=08:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# Bernini-R 1.3B has 12 attention heads, so one sample cannot use Ulysses-8.
# This launcher uses the eight-GPU allocation for two independent legal
# Ulysses-4 fixed-beta probes instead.
source_archive="${BERNINI_CAMERA_DUAL4_SOURCE_ARCHIVE:?set BERNINI_CAMERA_DUAL4_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_CAMERA_DUAL4_SOURCE_ARCHIVE_SHA256:?set BERNINI_CAMERA_DUAL4_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_CAMERA_DUAL4_SOURCE_REVISION:?set BERNINI_CAMERA_DUAL4_SOURCE_REVISION}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
bernini_archive="${BERNINI_OFFICIAL_ARCHIVE:?set BERNINI_OFFICIAL_ARCHIVE}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
adapter_checkpoint="${BERNINI_FULL644_ADAPTER_CHECKPOINT:?set BERNINI_FULL644_ADAPTER_CHECKPOINT}"
source_video="${BERNINI_CAMERA_DUAL4_SOURCE_VIDEO:?set BERNINI_CAMERA_DUAL4_SOURCE_VIDEO}"
instruction="${BERNINI_CAMERA_DUAL4_INSTRUCTION:?set BERNINI_CAMERA_DUAL4_INSTRUCTION}"
output_dir="${BERNINI_CAMERA_DUAL4_OUTPUT_DIR:?set BERNINI_CAMERA_DUAL4_OUTPUT_DIR}"
python_bin="${BERNINI_CAMERA_DUAL4_PYTHON_BIN:?set BERNINI_CAMERA_DUAL4_PYTHON_BIN}"
beta050="${BERNINI_CAMERA_BETA050:-0.5}"
beta100="${BERNINI_CAMERA_BETA100:-1.0}"
camera_estimator="${BERNINI_CAMERA_ESTIMATOR:-global_svd}"

expected_source_sha256="4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed"
expected_instruction_sha256="105ee8052a0f65d700736a8a25fdf02eb56f1b60d581403c328a8db3d500558c"
expected_adapter_config_sha256="b91c3a236b0e0e893e7c992be043ec28cfa05c73b7792c0b93b4013db15aef39"
expected_adapter_model_sha256="9217ff653e47f915105fe8fa64856037d63811562cec1e9fd53ae9e4613a9774"
expected_adapter_receipt_sha256="5931f7544d1bd185adf3fc07edb046e6bf27811b0835de8446f91c8a25c782c6"
expected_training_receipt_digest="6b5f2a053be048881b1426d9b7c4c380dc8b82f6098bfbda9c80034b26df17d1"
bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
checkpoint_manifest_file_count=23
bernini_archive_sha256="08a331958560544efb5e842666c236d819bfdc36d34b6eb9c1cdcee1546ae670"
expected_frames=81
expected_fps=25
expected_steps=40
expected_seed=2027
beta050_visible_gpus="0,1,2,3"
beta100_visible_gpus="4,5,6,7"

fail() {
  echo "[bernini-camera-beta-dual4] ERROR: $*" >&2
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
  adapter_checkpoint source_video output_dir python_bin
do
  value="${!value_name}"
  [[ "${value}" == /* ]] || fail "${value_name} must be absolute"
done
python_bin="$(realpath -e -- "${python_bin}")"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "method archive SHA-256 must be lowercase"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "method source revision must be a full lowercase SHA-1"
[[ -n "${instruction//[[:space:]]/}" ]] || fail "instruction is empty"
case "${camera_estimator}" in
  global_svd|grid_consensus) ;;
  *) fail "BERNINI_CAMERA_ESTIMATOR must be exactly global_svd or grid_consensus" ;;
esac
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "method source archive is not plain"
[[ "$(stat -c '%A' "${source_archive}")" != *w* ]] || fail "method source archive must be read-only"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini source root is invalid"
[[ -f "${bernini_archive}" && ! -L "${bernini_archive}" ]] || fail "Bernini source archive is invalid"
[[ -d "${veomni_root}/.git" && ! -L "${veomni_root}" ]] || fail "VeOmni root must be a Git checkout"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "base checkpoint is invalid"
[[ -d "${adapter_checkpoint}" && ! -L "${adapter_checkpoint}" ]] || fail "full644 adapter checkpoint is invalid"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "7b88 source video is not plain"
[[ -f "${python_bin}" && -x "${python_bin}" ]] || fail "runtime Python is invalid"

adapter_config="${adapter_checkpoint}/adapter/adapter_config.json"
adapter_model="${adapter_checkpoint}/adapter/adapter_model.safetensors"
adapter_receipt="${adapter_checkpoint}/receipt.json"
for value_name in adapter_config adapter_model adapter_receipt; do
  value="${!value_name}"
  [[ -f "${value}" && ! -L "${value}" ]] || fail "${value_name} is not plain"
done
for value_name in \
  source_archive bernini_root bernini_archive veomni_root checkpoint \
  adapter_checkpoint source_video python_bin
do
  require_absolute_canonical "${!value_name}" "${value_name}"
done

beta_slug_pair="$("${python_bin}" -B - "${beta050}" "${beta100}" <<'PY'
import math
import sys

values = []
for label, raw in zip(("beta050", "beta100"), sys.argv[1:]):
    try:
        value = float(raw)
    except ValueError as error:
        raise SystemExit(f"{label} is not numeric") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise SystemExit(f"{label} must be finite and in [0,1]")
    values.append(value)
if values[0] == values[1]:
    raise SystemExit("fixed-beta arms must use distinct beta values")
print(*(f"beta{format(value, '.17g').replace('.', 'p')}" for value in values))
PY
)"
read -r beta050_slug beta100_slug <<<"${beta_slug_pair}"
[[ "${beta050_slug}" =~ ^beta[0-9]+(p[0-9]+)?$ ]] || fail "beta050 canonical slug is unsafe"
[[ "${beta100_slug}" =~ ^beta[0-9]+(p[0-9]+)?$ ]] || fail "beta100 canonical slug is unsafe"
[[ "${beta050_slug}" != "${beta100_slug}" ]] || fail "canonical beta slugs collide"

[[ "${output_dir}" != / ]] || fail "output directory cannot be filesystem root"
output_name="${output_dir##*/}"
[[ "${output_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "output directory basename is unsafe"
[[ "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output directory must be canonical"
# The runner publishes with os.replace. These checks are strict preflight
# non-overwrite isolation, not an atomic create-only transaction.
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing to reuse camera dual4 output directory"
output_parent="$(dirname -- "${output_dir}")"
mkdir -p -- "${output_parent}"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent is not a plain directory"
[[ "$(realpath -e -- "${output_parent}")" == "${output_parent}" ]] || fail "output parent must be canonical"

[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "method source archive revision differs"
[[ "$(sha256sum "${bernini_archive}" | awk '{print $1}')" == "${bernini_archive_sha256}" ]] || fail "Bernini source archive hash differs"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "7b88 source video hash differs"
[[ "$(sha256sum "${adapter_config}" | awk '{print $1}')" == "${expected_adapter_config_sha256}" ]] || fail "full644 adapter config hash differs"
[[ "$(sha256sum "${adapter_model}" | awk '{print $1}')" == "${expected_adapter_model_sha256}" ]] || fail "full644 adapter model hash differs"
[[ "$(sha256sum "${adapter_receipt}" | awk '{print $1}')" == "${expected_adapter_receipt_sha256}" ]] || fail "full644 adapter receipt hash differs"
actual_instruction_sha256="$("${python_bin}" -B - "${instruction}" <<'PY'
import hashlib
import sys

print(hashlib.sha256(sys.argv[1].encode("utf-8")).hexdigest())
PY
)"
[[ "${actual_instruction_sha256}" == "${expected_instruction_sha256}" ]] || fail "7b88 instruction hash differs"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_commit}" ]] || fail "VeOmni revision differs"
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

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "invalid Slurm job ID"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent is invalid"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-camera-beta-dual4-${SLURM_JOB_ID}.XXXXXX")"
beta050_pid=""
beta100_pid=""

cleanup() {
  local exit_status=$?
  local cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/bernini-camera-beta-dual4-${SLURM_JOB_ID}."*) ;;
    *) echo "[bernini-camera-beta-dual4] refusing unsafe scratch cleanup: ${task_scratch:-unset}" >&2; cleanup_failed=1 ;;
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
    echo "[bernini-camera-beta-dual4] PASS estimator=${camera_estimator} beta050=${beta050} beta100=${beta100} world_per_arm=4 frames=81 fps=25 steps=40 seed=2027 cleanup_verified=true output=${output_dir}"
  else
    echo "[bernini-camera-beta-dual4] EXIT status=${exit_status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${exit_status}"
}

terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${beta050_pid}" "${beta100_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then
      kill "${child}" 2>/dev/null || true
    fi
  done
  for child in "${beta050_pid}" "${beta100_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then
      wait "${child}" 2>/dev/null || true
    fi
  done
  exit "${status}"
}

trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -p \
  "${task_scratch}/source-tree" \
  "${task_scratch}/arms/beta050/input/adapter/adapter" \
  "${task_scratch}/arms/beta050/input/source" \
  "${task_scratch}/arms/beta100/input/adapter/adapter" \
  "${task_scratch}/arms/beta100/input/source"

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
  generator_native_camera_stabilizer.py \
  fixed_grid_camera_consensus.py \
  fixed_grid_camera_consensus_stabilizer.py \
  infer_full644_camera_stabilizer.py \
  tri_branch_unipc.py \
  infer_lora.py \
  motion_residual.py \
  train_lora.py \
  tools/materialize_vae.py \
  audits/bernini_r13_ff4c5d4_checkpoint.sha256 \
  scripts/auh_infer_full644_camera_beta_dual4.sbatch \
  tests/test_generator_native_camera_stabilizer.py \
  tests/test_fixed_grid_camera_consensus.py \
  tests/test_fixed_grid_camera_consensus_stabilizer.py \
  tests/test_infer_full644_camera_stabilizer_contract.py \
  tests/test_auh_infer_full644_camera_beta_dual4_launcher.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "method archive closure lacks ${required}"
done
find "${method_root}" -type f -exec chmod a-w -- {} +
[[ -z "$(find "${method_root}" -type l -print -quit)" ]] || fail "extracted method tree contains a symlink"
method_tree_digest_pre="$(method_tree_digest "${method_root}")"
checkpoint_manifest="${method_root}/audits/bernini_r13_ff4c5d4_checkpoint.sha256"
validate_checkpoint_content "${checkpoint_manifest}"

training_source_revision="$(jq -er '.method_source_revision' "${adapter_receipt}")" || fail "full644 training source revision is missing"
training_source_archive_sha256="$(jq -er '.method_source_archive_sha256' "${adapter_receipt}")" || fail "full644 training archive identity is missing"
training_receipt_digest="$(jq -er '.receipt_digest' "${adapter_receipt}")" || fail "full644 training receipt digest is missing"
training_global_step="$(jq -er '.global_step' "${adapter_receipt}")" || fail "full644 global step is missing"
[[ "${training_source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "full644 training source revision is invalid"
[[ "${training_source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "full644 training archive SHA-256 is invalid"
[[ "${training_receipt_digest}" == "${expected_training_receipt_digest}" ]] || fail "full644 training receipt digest differs"
[[ "${training_global_step}" == 644 ]] || fail "full644 adapter is not global step 644"

for arm in beta050 beta100; do
  arm_root="${task_scratch}/arms/${arm}"
  cp -- "${source_video}" "${arm_root}/input/source/source.mp4"
  cp -- "${adapter_config}" "${arm_root}/input/adapter/adapter/adapter_config.json"
  cp -- "${adapter_model}" "${arm_root}/input/adapter/adapter/adapter_model.safetensors"
  cp -- "${adapter_receipt}" "${arm_root}/input/adapter/receipt.json"
  chmod 0444 -- \
    "${arm_root}/input/source/source.mp4" \
    "${arm_root}/input/adapter/adapter/adapter_config.json" \
    "${arm_root}/input/adapter/adapter/adapter_model.safetensors" \
    "${arm_root}/input/adapter/receipt.json"
  mkdir -p \
    "${arm_root}/cache/miopen-user" \
    "${arm_root}/cache/miopen-custom" \
    "${arm_root}/cache/torch-extensions" \
    "${arm_root}/cache/triton" \
    "${arm_root}/cache/xdg" \
    "${arm_root}/cache/torch" \
    "${arm_root}/cache/pycache" \
    "${arm_root}/tmp"
  [[ "$(sha256sum "${arm_root}/input/source/source.mp4" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "${arm} staged source hash differs"
  [[ "$(sha256sum "${arm_root}/input/adapter/adapter/adapter_config.json" | awk '{print $1}')" == "${expected_adapter_config_sha256}" ]] || fail "${arm} staged adapter config hash differs"
  [[ "$(sha256sum "${arm_root}/input/adapter/adapter/adapter_model.safetensors" | awk '{print $1}')" == "${expected_adapter_model_sha256}" ]] || fail "${arm} staged adapter model hash differs"
  [[ "$(sha256sum "${arm_root}/input/adapter/receipt.json" | awk '{print $1}')" == "${expected_adapter_receipt_sha256}" ]] || fail "${arm} staged adapter receipt hash differs"
done

PYTHONPATH="${method_root}" "${python_bin}" -B - \
  "${bernini_root}" "${veomni_root}" "${bernini_commit}" "${veomni_commit}" <<'PY'
from pathlib import Path
import sys

import train_lora

_, _, bernini_revision, veomni_revision = train_lora.validate_source_trees(
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    expected_bernini_commit=sys.argv[3],
    expected_veomni_commit=sys.argv[4],
)
if bernini_revision != sys.argv[3] or veomni_revision != sys.argv[4]:
    raise SystemExit("upstream source revision differs")
print(
    "[bernini-camera-beta-dual4] SOURCE_TREES_OK "
    f"bernini={bernini_revision} veomni={veomni_revision}"
)
PY

PYTHONPATH="${method_root}" "${python_bin}" -B - \
  "${task_scratch}/arms/beta050/input/source/source.mp4" <<'PY'
from fractions import Fraction
from pathlib import Path
import sys

from tools import materialize_vae

frames, fps, _ = materialize_vae._decode_exact_video(Path(sys.argv[1]))
if int(frames.shape[0]) != 81 or Fraction(str(float(fps))) != 25:
    raise SystemExit("7b88 source is not exact 81 frames at 25 fps")
print("[bernini-camera-beta-dual4] SOURCE_MEDIA_OK frames=81 fps=25")
PY

echo "[bernini-camera-beta-dual4] host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[bernini-camera-beta-dual4] revision=${source_revision} archive_sha256=${source_archive_sha256} method_tree_sha256=${method_tree_digest_pre}"
echo "[bernini-camera-beta-dual4] allocation=8xMI210 topology=dual-4-Ulysses beta050_visible_gpus=${beta050_visible_gpus} beta100_visible_gpus=${beta100_visible_gpus}"
echo "[bernini-camera-beta-dual4] camera_estimator=${camera_estimator} estimator_override_env=BERNINI_CAMERA_ESTIMATOR"
echo "[bernini-camera-beta-dual4] beta050=${beta050} beta100=${beta100} beta_override_env=BERNINI_CAMERA_BETA050,BERNINI_CAMERA_BETA100"
echo "[bernini-camera-beta-dual4] beta050_slug=${beta050_slug} beta100_slug=${beta100_slug}"
echo "[bernini-camera-beta-dual4] source_iid=7b88a1ca1f804f41 source_sha256=${expected_source_sha256} instruction_sha256=${expected_instruction_sha256}"
echo "[bernini-camera-beta-dual4] adapter_config_sha256=${expected_adapter_config_sha256} adapter_model_sha256=${expected_adapter_model_sha256} adapter_receipt_sha256=${expected_adapter_receipt_sha256}"
echo "[bernini-camera-beta-dual4] frames=${expected_frames} fps=${expected_fps} steps=${expected_steps} seed=${expected_seed}"
echo "[bernini-camera-beta-dual4] semantic_inputs=source_video,edit_instruction target=false support=false mask=false flow=false pose=false track=false trajectory=false reference=false first_frame=false"
echo "[bernini-camera-beta-dual4] output=${output_dir} publication_policy=preflight_nonoverwrite_not_atomic_create_only output_reuse=false"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

for pattern in \
  test_generator_native_camera_stabilizer.py \
  test_fixed_grid_camera_consensus.py \
  test_fixed_grid_camera_consensus_stabilizer.py \
  test_infer_full644_camera_stabilizer_contract.py \
  test_infer_lora_contract.py \
  test_auh_infer_full644_camera_beta_dual4_launcher.py
do
  PYTHONPATH="${method_root}" "${python_bin}" -B -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

mkdir -- "${output_dir}"
[[ -d "${output_dir}" && ! -L "${output_dir}" ]] || fail "could not create plain output directory"
beta050_output="${output_dir}/${beta050_slug}.mp4"
beta100_output="${output_dir}/${beta100_slug}.mp4"
beta050_receipt="${beta050_output}.receipt.json"
beta100_receipt="${beta100_output}.receipt.json"
beta050_log="${output_dir}/${beta050_slug}.torchrun.log"
beta100_log="${output_dir}/${beta100_slug}.torchrun.log"
for candidate in "${beta050_output}" "${beta050_receipt}" "${beta100_output}" "${beta100_receipt}" "${beta050_log}" "${beta100_log}"; do
  [[ ! -e "${candidate}" && ! -L "${candidate}" ]] || fail "refusing to overwrite output artifact ${candidate}"
done

job_mod=$((10#${SLURM_JOB_ID} % 10000))
beta050_master_port=$((40000 + 2 * job_mod))
beta100_master_port=$((beta050_master_port + 1))
[[ "${beta050_master_port}" -ne "${beta100_master_port}" ]] || fail "arm rendezvous ports collide"

launch_arm() (
  local arm="$1"
  local beta="$2"
  local visible_gpus="$3"
  local master_port="$4"
  local arm_output="$5"
  local arm_root="${task_scratch}/arms/${arm}"

  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK MASTER_ADDR MASTER_PORT
  export ROCR_VISIBLE_DEVICES="${visible_gpus}"
  export MIOPEN_USER_DB_PATH="${arm_root}/cache/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${arm_root}/cache/miopen-custom"
  export TORCH_EXTENSIONS_DIR="${arm_root}/cache/torch-extensions"
  export TRITON_CACHE_DIR="${arm_root}/cache/triton"
  export XDG_CACHE_HOME="${arm_root}/cache/xdg"
  export TORCH_HOME="${arm_root}/cache/torch"
  export PYTHONPYCACHEPREFIX="${arm_root}/cache/pycache"
  export TMPDIR="${arm_root}/tmp"
  export TORCHELASTIC_ERROR_FILE="${arm_root}/torch-elastic-error.json"

  echo "[bernini-camera-beta-dual4:${arm}] START beta=${beta} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES} master_port=${master_port} world=4 Ulysses=4"
  exec env PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port="${master_port}" \
    "${method_root}/infer_full644_camera_stabilizer.py" \
    --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" \
    --adapter-checkpoint "${arm_root}/input/adapter" \
    --source-video "${arm_root}/input/source/source.mp4" \
    --instruction "${instruction}" \
    --output "${arm_output}" \
    --beta "${beta}" \
    --camera-estimator "${camera_estimator}" \
    --num-inference-steps "${expected_steps}" \
    --seed "${expected_seed}" \
    --expected-source-sha256 "${expected_source_sha256}" \
    --expected-instruction-sha256 "${expected_instruction_sha256}" \
    --expected-bernini-commit "${bernini_commit}" \
    --expected-veomni-commit "${veomni_commit}" \
    --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
    --method-source-revision "${source_revision}" \
    --method-source-archive-sha256 "${source_archive_sha256}"
)

launch_arm beta050 "${beta050}" "${beta050_visible_gpus}" "${beta050_master_port}" "${beta050_output}" >"${beta050_log}" 2>&1 &
beta050_pid=$!
launch_arm beta100 "${beta100}" "${beta100_visible_gpus}" "${beta100_master_port}" "${beta100_output}" >"${beta100_log}" 2>&1 &
beta100_pid=$!
echo "[bernini-camera-beta-dual4] launched beta050_pid=${beta050_pid} beta100_pid=${beta100_pid} beta050_port=${beta050_master_port} beta100_port=${beta100_master_port}"

beta050_status=0
beta100_status=0
wait "${beta050_pid}" || beta050_status=$?
wait "${beta100_pid}" || beta100_status=$?
beta050_pid=""
beta100_pid=""
if [[ "${beta050_status}" -ne 0 || "${beta100_status}" -ne 0 ]]; then
  echo "[bernini-camera-beta-dual4] arm failure beta050_status=${beta050_status} beta100_status=${beta100_status}" >&2
  tail -n 80 -- "${beta050_log}" >&2 || true
  tail -n 80 -- "${beta100_log}" >&2 || true
  fail "one or more camera-stabilizer arms failed"
fi

for candidate in "${beta050_output}" "${beta050_receipt}" "${beta100_output}" "${beta100_receipt}" "${beta050_log}" "${beta100_log}"; do
  [[ -f "${candidate}" && ! -L "${candidate}" ]] || fail "expected output artifact is missing or not plain: ${candidate}"
done
[[ "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 6 ]] || fail "camera dual4 output directory must contain two mp4, two receipts, and two logs"
[[ -z "$(find "${output_dir}" -mindepth 1 -maxdepth 1 ! -type f -print -quit)" ]] || fail "camera dual4 output directory contains a non-file entry"

PYTHONPATH="${method_root}" "${python_bin}" -B - \
  --beta050-output "${beta050_output}" \
  --beta050-receipt "${beta050_receipt}" \
  --beta050-source "${task_scratch}/arms/beta050/input/source/source.mp4" \
  --beta050-adapter "${task_scratch}/arms/beta050/input/adapter" \
  --beta050-value "${beta050}" \
  --beta100-output "${beta100_output}" \
  --beta100-receipt "${beta100_receipt}" \
  --beta100-source "${task_scratch}/arms/beta100/input/source/source.mp4" \
  --beta100-adapter "${task_scratch}/arms/beta100/input/adapter" \
  --beta100-value "${beta100}" \
  --camera-estimator "${camera_estimator}" \
  --method-root "${method_root}" \
  --source-sha256 "${expected_source_sha256}" \
  --instruction-sha256 "${expected_instruction_sha256}" \
  --adapter-config-sha256 "${expected_adapter_config_sha256}" \
  --adapter-model-sha256 "${expected_adapter_model_sha256}" \
  --adapter-receipt-sha256 "${expected_adapter_receipt_sha256}" \
  --training-receipt-digest "${expected_training_receipt_digest}" \
  --training-source-revision "${training_source_revision}" \
  --training-source-archive-sha256 "${training_source_archive_sha256}" \
  --source-revision "${source_revision}" \
  --source-archive-sha256 "${source_archive_sha256}" \
  --bernini-commit "${bernini_commit}" \
  --veomni-commit "${veomni_commit}" \
  --checkpoint-tree-sha256 "${checkpoint_tree_sha256}" <<'PY'
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import stat

from tools import materialize_vae


def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


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


def require_plain(path, label):
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), f"{label} is not plain")


def audit_arm(args, *, label, beta, output, receipt_path, source, adapter):
    for path, description in (
        (output, f"{label} output"),
        (receipt_path, f"{label} receipt"),
        (source, f"{label} source"),
        (adapter / "adapter" / "adapter_config.json", f"{label} adapter config"),
        (adapter / "adapter" / "adapter_model.safetensors", f"{label} adapter model"),
        (adapter / "receipt.json", f"{label} adapter receipt"),
    ):
        require_plain(path, description)
    require(file_sha256(source) == args.source_sha256, f"{label} source hash differs")
    require(file_sha256(adapter / "adapter" / "adapter_config.json") == args.adapter_config_sha256, f"{label} adapter config hash differs")
    require(file_sha256(adapter / "adapter" / "adapter_model.safetensors") == args.adapter_model_sha256, f"{label} adapter model hash differs")
    require(file_sha256(adapter / "receipt.json") == args.adapter_receipt_sha256, f"{label} adapter receipt hash differs")

    frames, fps, _ = materialize_vae._decode_exact_video(output)
    require(int(frames.shape[0]) == 81, f"{label} output frame count differs")
    require(Fraction(str(float(fps))) == 25, f"{label} output fps differs")
    receipt = strict_json(receipt_path)
    unsigned = dict(receipt)
    stored_digest = unsigned.pop("receipt_digest", None)
    require(stored_digest == hashlib.sha256(canonical_bytes(unsigned)).hexdigest(), f"{label} receipt digest differs")

    method_files = receipt.get("method_files_sha256", {})
    expected_method_files = {
        name: file_sha256(args.method_root / name)
        for name in (
            "infer_full644_camera_stabilizer.py",
            "generator_native_camera_stabilizer.py",
            "fixed_grid_camera_consensus.py",
            "fixed_grid_camera_consensus_stabilizer.py",
            "tri_branch_unipc.py",
            "infer_lora.py",
            "motion_residual.py",
        )
    }
    input_contract = receipt.get("input", {})
    preprocessing = receipt.get("preprocessing", {})
    sampling = receipt.get("sampling", {})
    output_contract = receipt.get("output", {})
    adapter_identity = receipt.get("full644_adapter", {})
    camera = receipt.get("camera_stabilizer", {})
    execution = camera.get("execution", {})
    certificate = execution.get("certificate", {})
    tri_branch = execution.get("tri_branch", {})
    camera_trace = execution.get("camera", {})

    require(receipt.get("schema_version") == "bernini-full644-camera-tangent-inference-v1", f"{label} schema differs")
    require(receipt.get("method") == "full644-generator-native-camera-tangent-stabilizer-v1", f"{label} method differs")
    require(receipt.get("method_files_sha256") == expected_method_files, f"{label} method hashes differ")
    require(receipt.get("method_source_revision") == args.source_revision, f"{label} method revision differs")
    require(receipt.get("method_source_archive_sha256") == args.source_archive_sha256, f"{label} method archive differs")
    require(receipt.get("bernini_commit") == args.bernini_commit, f"{label} Bernini revision differs")
    require(receipt.get("veomni_commit") == args.veomni_commit, f"{label} VeOmni revision differs")
    require(receipt.get("checkpoint_tree_sha256") == args.checkpoint_tree_sha256, f"{label} checkpoint hash differs")
    require(receipt.get("production_claim_forbidden") is True, f"{label} production policy differs")
    require(receipt.get("scientific_claim_authorized") is False, f"{label} scientific policy differs")

    require(adapter_identity.get("checkpoint_root") == str(adapter), f"{label} adapter path differs")
    require(adapter_identity.get("adapter_config_sha256") == args.adapter_config_sha256, f"{label} receipt adapter config differs")
    require(adapter_identity.get("adapter_model_sha256") == args.adapter_model_sha256, f"{label} receipt adapter model differs")
    require(adapter_identity.get("training_receipt_file_sha256") == args.adapter_receipt_sha256, f"{label} receipt adapter receipt differs")
    require(adapter_identity.get("training_receipt_digest") == args.training_receipt_digest, f"{label} training receipt digest differs")
    require(adapter_identity.get("training_method_source_revision") == args.training_source_revision, f"{label} training method revision differs")
    require(adapter_identity.get("training_method_source_archive_sha256") == args.training_source_archive_sha256, f"{label} training archive differs")
    require(adapter_identity.get("training_global_step") == 644, f"{label} training step differs")
    require(adapter_identity.get("target_module_count") == 240, f"{label} target module count differs")
    require(adapter_identity.get("adapter_tensor_count") == 480, f"{label} adapter tensor count differs")
    for field in (
        "strict_240_module_scope",
        "strict_480_tensor_reload",
        "safe_merge_before_camera_hook",
        "frozen_during_inference",
    ):
        require(adapter_identity.get(field) is True, f"{label} adapter evidence differs: {field}")

    require(input_contract.get("source_video_path") == str(source), f"{label} source path differs")
    require(input_contract.get("source_video_sha256") == args.source_sha256, f"{label} receipt source hash differs")
    require(input_contract.get("instruction_utf8_sha256") == args.instruction_sha256, f"{label} instruction hash differs")
    require(input_contract.get("accepted_external_conditions") == ["source_video", "edit_instruction"], f"{label} external input closure differs")
    for field in (
        "target_video_argument",
        "target_accessed_by_inference",
        "external_mask_or_swept_tube",
        "external_tracking_pose_or_trajectory",
        "reference_image_or_video",
        "external_shared_i0",
        "support_accessed_by_inference",
        "external_mask_flow_pose_track_trajectory",
        "edited_first_frame_or_anchor",
    ):
        require(input_contract.get(field) is False, f"{label} forbidden condition active: {field}")

    require(preprocessing.get("frame_count") == 81 and preprocessing.get("fps") == 25.0, f"{label} preprocessing contract differs")
    require(sampling.get("num_frames") == 81, f"{label} sampling frames differ")
    require(sampling.get("num_inference_steps") == 40, f"{label} inference steps differ")
    require(sampling.get("seed") == 2027, f"{label} seed differs")
    require(sampling.get("ulysses_size") == 4, f"{label} Ulysses size differs")
    require(sampling.get("rank0_decode_and_save_only") is True, f"{label} rank-zero output contract differs")
    require(sampling.get("per_step_transformer_forwards") == 3, f"{label} per-step forwards differ")
    require(sampling.get("total_transformer_forwards") == 120, f"{label} total forwards differ")
    require(sampling.get("custom_integrator") is False, f"{label} custom integrator active")

    require(math.isclose(float(camera.get("beta")), beta, rel_tol=0.0, abs_tol=1e-12), f"{label} beta differs")
    require(camera.get("estimator") == args.camera_estimator, f"{label} receipt camera estimator differs")
    require(camera.get("enabled") is True, f"{label} stabilizer disabled")
    require(camera.get("source_and_instruction_only") is True, f"{label} camera semantic closure differs")
    if args.camera_estimator == "global_svd":
        require(camera.get("camera_edit_requested") is False, f"{label} camera-edit mode active")
        require(camera.get("estimator_semantics") == "global_source_tangent_thin_svd", f"{label} global estimator semantics differ")
    else:
        require("camera_edit_requested" not in camera, f"{label} grid receipt inherited a global-only camera-edit field")
        require(camera.get("estimator_semantics") == "fixed_grid_physical_coefficients_median_MAD_trimmed_consensus_per_phase", f"{label} grid estimator semantics differ")
        require(camera.get("nonconsensus_phase") == "exact_action_value_passthrough", f"{label} grid non-consensus semantics differ")
    unsigned_execution = dict(execution)
    stored_trace_digest = unsigned_execution.pop("trace_digest", None)
    require(stored_trace_digest == hashlib.sha256(canonical_bytes(unsigned_execution)).hexdigest(), f"{label} execution trace digest differs")
    require(certificate.get("frame_count") == 81, f"{label} certificate frames differ")
    require(certificate.get("step_count") == 40, f"{label} certificate steps differ")
    require(certificate.get("transformer_forwards") == 120, f"{label} certificate forwards differ")
    require(certificate.get("official_action_apg_exact_steps") == 40, f"{label} APG parity differs")
    require(certificate.get("camera_callback_calls") == 40, f"{label} camera callback count differs")
    require(certificate.get("original_unipc_calls") == 40, f"{label} UniPC count differs")
    require(certificate.get("camera_estimator") == args.camera_estimator, f"{label} certificate camera estimator differs")
    require(certificate.get("beta_zero_exact_full644_passthrough") is (beta == 0.0), f"{label} beta-zero certificate differs")
    active = beta != 0.0
    expected_basis_builds = int(active and args.camera_estimator == "global_svd")
    expected_basis_reuses = 39 if active and args.camera_estimator == "global_svd" else 0
    expected_geometry_builds = int(active and args.camera_estimator == "grid_consensus")
    expected_geometry_reuses = 39 if active and args.camera_estimator == "grid_consensus" else 0
    require(certificate.get("camera_basis_build_count") == expected_basis_builds, f"{label} camera basis build count differs")
    require(certificate.get("camera_basis_reuse_count") == expected_basis_reuses, f"{label} camera basis reuse count differs")
    require(certificate.get("camera_geometry_build_count") == expected_geometry_builds, f"{label} camera geometry build count differs")
    require(certificate.get("camera_geometry_reuse_count") == expected_geometry_reuses, f"{label} camera geometry reuse count differs")
    require(certificate.get("camera_estimator_state_build_count") == expected_basis_builds + expected_geometry_builds, f"{label} camera estimator state build count differs")
    require(certificate.get("camera_estimator_state_reuse_count") == expected_basis_reuses + expected_geometry_reuses, f"{label} camera estimator state reuse count differs")
    require(certificate.get("projection_boundary") == "after_exact_official_action_apg_before_original_unipc_step", f"{label} projection boundary differs")
    require(certificate.get("custom_integrator") is False, f"{label} certificate reports custom integrator")
    rank_authority = certificate.get("grid_rank_authority")
    if args.camera_estimator == "global_svd":
        require(rank_authority is None, f"{label} global estimator claimed grid rank authority")
    else:
        require(isinstance(rank_authority, dict), f"{label} grid rank authority certificate is absent")
        require(rank_authority.get("schema_version") == "bernini-grid-rank0-authority-v1", f"{label} grid rank authority schema differs")
        require(rank_authority.get("world_size") == 4, f"{label} grid authority world size differs")
        require(rank_authority.get("receipt_group_rank") == 0, f"{label} receipt was not aggregated on process-group rank 0")
        require(rank_authority.get("process_group_rank0") == 0, f"{label} grid authority source rank differs")
        require(rank_authority.get("rank0_receipt_aggregated_steps") == 40, f"{label} rank-0 receipt did not aggregate 40 steps")
        require(rank_authority.get("large_object_all_gather") is False, f"{label} grid authority used a large object gather")
        if beta == 0.0:
            require(rank_authority.get("zero_beta_no_collective_steps") == 40, f"{label} beta-zero no-collective count differs")
            require(rank_authority.get("rank0_authoritative_broadcast_steps") == 0, f"{label} beta-zero issued a correction broadcast")
            require(rank_authority.get("source_clean_cross_rank_exact") is None, f"{label} beta-zero claimed an unexecuted source collective")
            require(rank_authority.get("post_broadcast_exact_steps") == 0, f"{label} beta-zero claimed post-broadcast evidence")
        else:
            disagreements = rank_authority.get("pre_broadcast_max_abs_disagreement_by_step")
            require(rank_authority.get("source_clean_cross_rank_exact") is True, f"{label} source clean field was not exact across ranks")
            require(rank_authority.get("source_clean_rank0_reference_exact") is True, f"{label} source rank-0 reference proof differs")
            require(rank_authority.get("source_clean_exact_certification_steps") == 1, f"{label} source exactness was not certified exactly once")
            require(rank_authority.get("action_clean_cross_rank_exact_steps") == 40, f"{label} action clean fields were not exact for 40 steps")
            require(rank_authority.get("all_rank_candidate_success_steps") == 40, f"{label} local grid candidates did not succeed on all ranks")
            require(rank_authority.get("rank0_authoritative_broadcast_steps") == 40, f"{label} rank-0 authoritative broadcasts did not cover 40 steps")
            require(rank_authority.get("post_broadcast_exact_steps") == 40, f"{label} post-broadcast exact evidence did not cover 40 steps")
            require(rank_authority.get("all_rank_participant_count_per_step") == 4, f"{label} all-rank participant count differs")
            require(rank_authority.get("collective_order_identical_all_ranks") is True, f"{label} collective order is not certified")
            require(isinstance(disagreements, list) and len(disagreements) == 40, f"{label} pre-broadcast disagreement trace differs")
            require(all(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0.0 for value in disagreements), f"{label} pre-broadcast disagreement values are invalid")
    require(tri_branch.get("step_count") == 40 and len(tri_branch.get("steps", [])) == 40, f"{label} tri-branch trace differs")
    require(camera_trace.get("step_count") == 40 and len(camera_trace.get("steps", [])) == 40, f"{label} camera trace differs")
    require(camera_trace.get("estimator") == args.camera_estimator, f"{label} trace camera estimator differs")
    require(camera_trace.get("basis_build_count") == expected_basis_builds, f"{label} trace basis build count differs")
    require(camera_trace.get("basis_reuse_count") == expected_basis_reuses, f"{label} trace basis reuse count differs")
    require(camera_trace.get("geometry_build_count") == expected_geometry_builds, f"{label} trace geometry build count differs")
    require(camera_trace.get("geometry_reuse_count") == expected_geometry_reuses, f"{label} trace geometry reuse count differs")
    require(math.isclose(float(camera_trace.get("beta")), beta, rel_tol=0.0, abs_tol=1e-12), f"{label} camera trace beta differs")
    require(all(math.isclose(float(row.get("beta")), beta, rel_tol=0.0, abs_tol=1e-12) for row in camera_trace["steps"]), f"{label} per-step beta differs")
    require(all(row.get("estimator") == args.camera_estimator for row in camera_trace["steps"]), f"{label} per-step camera estimator differs")
    if args.camera_estimator == "global_svd":
        require(all(row.get("grid_rank_authority") is None for row in camera_trace["steps"]), f"{label} global steps claimed grid rank authority")
    elif beta == 0.0:
        require(all(row.get("grid_rank_authority", {}).get("mode") == "zero_beta_exact_action_no_collective" and row.get("grid_rank_authority", {}).get("collective_sequence") == [] for row in camera_trace["steps"]), f"{label} beta-zero per-step collective evidence differs")
    else:
        require(all(row.get("grid_rank_authority", {}).get("world_size") == 4 and row.get("grid_rank_authority", {}).get("rank0_authoritative_broadcast") is True and row.get("grid_rank_authority", {}).get("post_broadcast_exact") is True for row in camera_trace["steps"]), f"{label} active grid per-step rank authority differs")
    if beta == 0.0:
        require(all(row.get("basis_built_this_step") is False and row.get("basis_reused_from_prior_step") is False and row.get("geometry_built_this_step") is False and row.get("geometry_reused_from_prior_step") is False for row in camera_trace["steps"]), f"{label} beta-zero estimator-state trace differs")
    elif args.camera_estimator == "global_svd":
        require(all(row.get("geometry_built_this_step") is False and row.get("geometry_reused_from_prior_step") is False for row in camera_trace["steps"]), f"{label} global trace unexpectedly built grid geometry")
        require(camera_trace["steps"][0].get("basis_built_this_step") is True, f"{label} first-step basis build is absent")
        require(camera_trace["steps"][0].get("basis_reused_from_prior_step") is False, f"{label} first-step basis reuse is invalid")
        require(all(row.get("basis_built_this_step") is False and row.get("basis_reused_from_prior_step") is True for row in camera_trace["steps"][1:]), f"{label} cached basis reuse trace differs")
    else:
        require(all(row.get("basis_built_this_step") is False and row.get("basis_reused_from_prior_step") is False for row in camera_trace["steps"]), f"{label} grid trace unexpectedly built global basis")
        require(camera_trace["steps"][0].get("geometry_built_this_step") is True, f"{label} first-step grid geometry build is absent")
        require(camera_trace["steps"][0].get("geometry_reused_from_prior_step") is False, f"{label} first-step grid geometry reuse is invalid")
        require(all(row.get("geometry_built_this_step") is False and row.get("geometry_reused_from_prior_step") is True for row in camera_trace["steps"][1:]), f"{label} cached grid geometry reuse trace differs")

    require(output_contract.get("path") == str(output), f"{label} output path differs")
    require(output_contract.get("sha256") == file_sha256(output), f"{label} output hash differs")
    require(output_contract.get("frame_count") == 81 and output_contract.get("fps") == 25.0, f"{label} output media receipt differs")
    return {
        "arm": label,
        "beta": beta,
        "output_sha256": file_sha256(output),
        "receipt_sha256": file_sha256(receipt_path),
    }


parser = argparse.ArgumentParser()
for name in (
    "beta050-output", "beta050-receipt", "beta050-source", "beta050-adapter", "beta050-value",
    "beta100-output", "beta100-receipt", "beta100-source", "beta100-adapter", "beta100-value",
    "camera-estimator", "method-root", "source-sha256", "instruction-sha256", "adapter-config-sha256",
    "adapter-model-sha256", "adapter-receipt-sha256", "training-receipt-digest",
    "training-source-revision", "training-source-archive-sha256", "source-revision",
    "source-archive-sha256", "bernini-commit", "veomni-commit", "checkpoint-tree-sha256",
):
    parser.add_argument(f"--{name}", required=True)
args = parser.parse_args()
args.method_root = Path(args.method_root).resolve(strict=True)
require(args.camera_estimator in ("global_svd", "grid_consensus"), "audit camera estimator is unsupported")

results = []
for label in ("beta050", "beta100"):
    results.append(
        audit_arm(
            args,
            label=label,
            beta=float(getattr(args, f"{label}_value")),
            output=Path(getattr(args, f"{label}_output")).resolve(strict=True),
            receipt_path=Path(getattr(args, f"{label}_receipt")).resolve(strict=True),
            source=Path(getattr(args, f"{label}_source")).resolve(strict=True),
            adapter=Path(getattr(args, f"{label}_adapter")).resolve(strict=True),
        )
    )
print(
    json.dumps(
        {
            "event": "FULL644_CAMERA_BETA_DUAL4_STRONG_AUDIT_OK",
            "source_instruction_only": True,
            "dual_four_rank_ulysses": True,
            "camera_estimator": args.camera_estimator,
            "frames": 81,
            "fps": 25,
            "steps": 40,
            "seed": 2027,
            "arms": results,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY

[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive changed during inference"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "method archive revision changed during inference"
[[ "$(sha256sum "${bernini_archive}" | awk '{print $1}')" == "${bernini_archive_sha256}" ]] || fail "Bernini archive changed during inference"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "7b88 source changed during inference"
[[ "$(sha256sum "${adapter_config}" | awk '{print $1}')" == "${expected_adapter_config_sha256}" ]] || fail "full644 adapter config changed during inference"
[[ "$(sha256sum "${adapter_model}" | awk '{print $1}')" == "${expected_adapter_model_sha256}" ]] || fail "full644 adapter model changed during inference"
[[ "$(sha256sum "${adapter_receipt}" | awk '{print $1}')" == "${expected_adapter_receipt_sha256}" ]] || fail "full644 adapter receipt changed during inference"
validate_checkpoint_content "${checkpoint_manifest}"
PYTHONPATH="${method_root}" "${python_bin}" -B - \
  "${bernini_root}" "${veomni_root}" "${bernini_commit}" "${veomni_commit}" <<'PY'
from pathlib import Path
import sys

import train_lora

_, _, bernini_revision, veomni_revision = train_lora.validate_source_trees(
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    expected_bernini_commit=sys.argv[3],
    expected_veomni_commit=sys.argv[4],
)
if bernini_revision != sys.argv[3] or veomni_revision != sys.argv[4]:
    raise SystemExit("upstream source changed during inference")
PY
method_tree_digest_post="$(method_tree_digest "${method_root}")"
[[ "${method_tree_digest_post}" == "${method_tree_digest_pre}" ]] || fail "extracted method tree changed during inference"
rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
