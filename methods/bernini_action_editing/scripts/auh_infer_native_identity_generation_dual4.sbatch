#!/usr/bin/env bash
#SBATCH --job-name=bern-native-id2x4
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=04:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# Draft production launcher for the frozen native identity-generation canary.
# Bernini-R 1.3B has 12 attention heads, so Ulysses-8 is invalid.  One 8-GPU
# allocation is split into two isolated Ulysses-4 processes:
#   GPUs 0-3: t2v + r2v (2 + 3 transformer forwards per solver step)
#   GPUs 4-7: rv2v       (4 transformer forwards per solver step)
# Both groups use the same source, action prompt, target shape, seed and the
# official Gaussian-start sampler.

source_archive="${BERNINI_NATIVE_SOURCE_ARCHIVE:?set BERNINI_NATIVE_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_NATIVE_SOURCE_ARCHIVE_SHA256:?set BERNINI_NATIVE_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_NATIVE_SOURCE_REVISION:?set BERNINI_NATIVE_SOURCE_REVISION}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
source_video="${BERNINI_NATIVE_SOURCE_VIDEO:?set BERNINI_NATIVE_SOURCE_VIDEO}"
expected_source_sha256="${BERNINI_NATIVE_SOURCE_SHA256:?set BERNINI_NATIVE_SOURCE_SHA256}"
action_prompt="${BERNINI_NATIVE_ACTION_PROMPT:?set BERNINI_NATIVE_ACTION_PROMPT}"
expected_action_prompt_sha256="${BERNINI_NATIVE_ACTION_PROMPT_SHA256:?set BERNINI_NATIVE_ACTION_PROMPT_SHA256}"
output_dir="${BERNINI_NATIVE_OUTPUT_DIR:?set BERNINI_NATIVE_OUTPUT_DIR}"
python_bin="${BERNINI_NATIVE_PYTHON_BIN:?set BERNINI_NATIVE_PYTHON_BIN}"

expected_frames=81
expected_fps=25
expected_steps=40
expected_seed=2027
t2v_r2v_visible_gpus="0,1,2,3"
rv2v_visible_gpus="4,5,6,7"

fail() {
  echo "[bernini-native-dual4] ERROR: $*" >&2
  exit 2
}

require_absolute() {
  local value="$1"
  local label="$2"
  [[ "${value}" == /* ]] || fail "${label} must be absolute"
}

for value_name in \
  source_archive bernini_root veomni_root checkpoint checkpoint_manifest \
  source_video output_dir python_bin
do
  require_absolute "${!value_name}" "${value_name}"
done

[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source archive SHA-256 must be lowercase"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be a full lowercase SHA-1"
[[ "${expected_source_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source video SHA-256 must be lowercase"
[[ "${expected_action_prompt_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "action prompt SHA-256 must be lowercase"
[[ -n "${action_prompt//[[:space:]]/}" ]] || fail "action prompt is empty"

source_archive="$(realpath -e -- "${source_archive}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
source_video="$(realpath -e -- "${source_video}")"
python_bin="$(realpath -e -- "${python_bin}")"

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "method source archive is not plain"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini source root is invalid"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni source root is invalid"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint root is invalid"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest is not plain"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "source video is not plain"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "runtime Python is invalid"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method source archive hash differs"
actual_archive_revision="$(git get-tar-commit-id <"${source_archive}")" || fail "method source must be a git tar archive"
[[ "${actual_archive_revision}" == "${source_revision}" ]] || fail "method archive embedded revision differs"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "source video hash differs"

actual_action_prompt_sha256="$("${python_bin}" -B - "${action_prompt}" <<'PY'
import hashlib
import sys

print(hashlib.sha256(sys.argv[1].encode("utf-8")).hexdigest())
PY
)"
[[ "${actual_action_prompt_sha256}" == "${expected_action_prompt_sha256}" ]] || fail "action prompt hash differs"

[[ "${output_dir}" != / ]] || fail "output directory cannot be filesystem root"
output_basename="${output_dir##*/}"
[[ "${output_basename}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename is unsafe"
[[ "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output directory must be canonical"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing to reuse output directory"
output_parent="$(dirname -- "${output_dir}")"
mkdir -p -- "${output_parent}"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent is invalid"
[[ "$(realpath -e -- "${output_parent}")" == "${output_parent}" ]] || fail "output parent must be canonical"

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
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-native-dual4-${SLURM_JOB_ID}.XXXXXX")"
t2v_r2v_pid=""
rv2v_pid=""

cleanup() {
  local exit_status=$?
  local cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/bernini-native-dual4-${SLURM_JOB_ID}."*) ;;
    *) echo "[bernini-native-dual4] refusing unsafe scratch cleanup" >&2; cleanup_failed=1 ;;
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
    echo "[bernini-native-dual4] PASS topology=dual-4-Ulysses arms=t2v,r2v,rv2v frames=81 fps=25 steps=40 seed=2027 output=${output_dir}"
  else
    echo "[bernini-native-dual4] EXIT status=${exit_status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${exit_status}"
}

terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${t2v_r2v_pid}" "${rv2v_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then
      kill "${child}" 2>/dev/null || true
    fi
  done
  for child in "${t2v_r2v_pid}" "${rv2v_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then
      wait "${child}" 2>/dev/null || true
    fi
  done
  exit "${status}"
}

trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -p -- "${task_scratch}/source-tree" "${task_scratch}/groups"
archive_copy="${task_scratch}/method-source.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod 0444 -- "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged method archive hash differs"
[[ "$(git get-tar-commit-id <"${archive_copy}")" == "${source_revision}" ]] || fail "staged method archive revision differs"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

archive = sys.argv[1]
required = {
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
    "methods/bernini_action_editing/source_kv_replay.py",
    "methods/bernini_action_editing/source_kv_route_batches.py",
    "methods/bernini_action_editing/source_value_residual.py",
    "methods/bernini_action_editing/tools/materialize_vae.py",
}
seen = set()
with tarfile.open(archive, "r:*") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit("archive member escaped method subtree")
        normalized = path.as_posix().lstrip("./")
        if member.issym() or member.islnk() or member.isfifo() or member.isdev():
            raise SystemExit("archive contains a link or device")
        if normalized.startswith("methods/bernini_action_editing/"):
            seen.add(normalized)
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"archive lacks native canary closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
[[ -d "${method_root}" && ! -L "${method_root}" ]] || fail "extracted method root is invalid"
find "${method_root}" -type f -exec chmod a-w -- {} +

for required in \
  infer_native_identity_generation_canary.py \
  infer_lora.py \
  train_lora.py \
  infer_source_kv_carrier_oracle.py \
  infer_source_value_residual_oracle.py \
  source_kv_replay.py \
  source_kv_route_batches.py \
  source_value_residual.py \
  tools/materialize_vae.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "missing staged ${required}"
done

mkdir -- "${output_dir}"
t2v_r2v_log="${output_dir}/t2v-r2v.torchrun.log"
rv2v_log="${output_dir}/rv2v.torchrun.log"
t2v_r2v_output="${output_dir}/t2v-r2v"
rv2v_output="${output_dir}/rv2v"

job_mod=$((SLURM_JOB_ID % 10000))
t2v_r2v_master_port=$((30000 + 2 * job_mod))
rv2v_master_port=$((t2v_r2v_master_port + 1))
[[ "${t2v_r2v_master_port}" -ne "${rv2v_master_port}" ]] || fail "rendezvous ports collided"

launch_group() (
  set -Eeuo pipefail
  local group="$1"
  local visible_gpus="$2"
  local master_port="$3"
  local group_output="$4"
  shift 4
  local arms=("$@")
  local group_root="${task_scratch}/groups/${group}"
  mkdir -p -- \
    "${group_root}/cache/miopen-user" \
    "${group_root}/cache/miopen-custom" \
    "${group_root}/cache/torch-extensions" \
    "${group_root}/cache/triton" \
    "${group_root}/cache/xdg" \
    "${group_root}/cache/pycache" \
    "${group_root}/tmp"
  # ROCr performs the physical-device selection.  Do not set the HIP/CUDA
  # aliases to the same physical indices: after ROCr remaps a group to local
  # devices 0..3, a second 4,5,6,7 filter can hide the whole second group.
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible_gpus}"
  export MIOPEN_USER_DB_PATH="${group_root}/cache/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${group_root}/cache/miopen-custom"
  export TORCH_EXTENSIONS_DIR="${group_root}/cache/torch-extensions"
  export TRITON_CACHE_DIR="${group_root}/cache/triton"
  export XDG_CACHE_HOME="${group_root}/cache/xdg"
  export PYTHONPYCACHEPREFIX="${group_root}/cache/pycache"
  export TMPDIR="${group_root}/tmp"
  export TORCHELASTIC_ERROR_FILE="${group_root}/torch-elastic-error.json"
  export BERNINI_OUTPUT_TRANSACTION_ID="${SLURM_JOB_ID}-${group}"

  "${python_bin}" -B -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port="${master_port}" \
    "${method_root}/infer_native_identity_generation_canary.py" \
    --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --source-video "${source_video}" \
    --expected-source-sha256 "${expected_source_sha256}" \
    --action-prompt "${action_prompt}" \
    --expected-action-prompt-sha256 "${expected_action_prompt_sha256}" \
    --output-dir "${group_output}" \
    --arms "${arms[@]}" \
    --num-inference-steps "${expected_steps}" \
    --seed "${expected_seed}" \
    --method-source-revision "${source_revision}" \
    --method-source-archive-sha256 "${source_archive_sha256}"
)

echo "[bernini-native-dual4] semantic_inputs=source_video,action_prompt target=false external_reference=false mask=false flow=false pose=false track=false trajectory=false first_frame=false"
launch_group t2v-r2v "${t2v_r2v_visible_gpus}" "${t2v_r2v_master_port}" "${t2v_r2v_output}" t2v r2v >"${t2v_r2v_log}" 2>&1 &
t2v_r2v_pid=$!
launch_group rv2v "${rv2v_visible_gpus}" "${rv2v_master_port}" "${rv2v_output}" rv2v >"${rv2v_log}" 2>&1 &
rv2v_pid=$!

t2v_r2v_status=0
rv2v_status=0
wait "${t2v_r2v_pid}" || t2v_r2v_status=$?
wait "${rv2v_pid}" || rv2v_status=$?
if [[ "${t2v_r2v_status}" -ne 0 || "${rv2v_status}" -ne 0 ]]; then
  tail -n 160 -- "${t2v_r2v_log}" >&2 || true
  tail -n 160 -- "${rv2v_log}" >&2 || true
  fail "one or more native canary groups failed"
fi

strong_audit_log="${output_dir}/strong-audit.log"
"${python_bin}" -B - \
  "${t2v_r2v_output}/receipt.json" \
  "${rv2v_output}/receipt.json" \
  "${expected_source_sha256}" \
  "${expected_action_prompt_sha256}" >"${strong_audit_log}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def load(path):
    receipt_path = Path(path)
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    declared = value.pop("receipt_digest", None)
    if hashlib.sha256(canonical(value)).hexdigest() != declared:
        raise SystemExit(f"receipt digest differs: {receipt_path}")
    value["receipt_digest"] = declared
    return value


left = load(sys.argv[1])
right = load(sys.argv[2])
expected_source = sys.argv[3]
expected_prompt = sys.argv[4]
if left.get("schema_version") != "bernini-native-identity-generation-canary-v1":
    raise SystemExit("left receipt schema differs")
if right.get("schema_version") != "bernini-native-identity-generation-canary-v1":
    raise SystemExit("right receipt schema differs")
if left.get("arms") != ["t2v", "r2v"] or right.get("arms") != ["rv2v"]:
    raise SystemExit("dual4 arm partition differs")
for receipt in (left, right):
    inputs = receipt.get("input", {})
    if inputs.get("source_video_sha256") != expected_source:
        raise SystemExit("source identity differs")
    if inputs.get("action_prompt_utf8_sha256") != expected_prompt:
        raise SystemExit("action prompt identity differs")
    if inputs.get("accepted_external_conditions") != ["source_video", "action_prompt"]:
        raise SystemExit("external condition closure differs")
    if inputs.get("target_video") is not False or inputs.get("external_reference_image_or_video") is not False:
        raise SystemExit("privileged external input appeared")
    if receipt.get("freeze_certificate", {}).get("base_frozen") is not True:
        raise SystemExit("base freeze certificate differs")
    source_clean = receipt.get("source_condition_artifact")
    if "rv2v" in receipt.get("arms", []):
        if not isinstance(source_clean, dict):
            raise SystemExit("rv2v source-condition latent receipt is absent")
        source_clean_path = Path(str(source_clean.get("path", "")))
        if not source_clean_path.is_file() or source_clean_path.is_symlink():
            raise SystemExit("rv2v source-condition latent is absent")
        if hashlib.sha256(source_clean_path.read_bytes()).hexdigest() != source_clean.get("sha256"):
            raise SystemExit("rv2v source-condition latent digest differs")
        if source_clean.get("shape")[:3] != [1, 16, 21] or source_clean.get("stored_dtype") != "torch.float32":
            raise SystemExit("rv2v source-condition latent geometry differs")
        if source_clean.get("artifact_role") != "source_video_condition":
            raise SystemExit("rv2v source-condition role differs")
        if source_clean.get("source_video_vae_encode_before_any_decode") is not True:
            raise SystemExit("rv2v source-condition latent origin differs")
        if source_clean.get("mp4_decode_reencode_used") is not False:
            raise SystemExit("rv2v source-condition latent used an MP4 round trip")
    elif source_clean is not None:
        raise SystemExit("non-rv2v receipt unexpectedly published a full source latent")

expected = {
    "t2v": {"mode": "t2v_apg", "refs": [], "videos": 0},
    "r2v": {"mode": "r2v_apg", "refs": [0, 20, 40, 60, 80], "videos": 0},
    "rv2v": {"mode": "rv2v", "refs": [0, 27, 53, 80], "videos": 1},
}
outputs = {}
for receipt in (left, right):
    for arm in receipt["arms"]:
        sampling = receipt["sampling"][arm]
        condition = receipt["conditioning"][arm]
        wanted = expected[arm]
        if sampling.get("num_frames") != 81 or sampling.get("num_inference_steps") != 40:
            raise SystemExit(f"{arm} exact81/40 contract differs")
        if sampling.get("seed") != 2027 or sampling.get("guidance_mode") != wanted["mode"]:
            raise SystemExit(f"{arm} seed/guidance differs")
        if sampling.get("target_initialization") != "official_gen_wanx22_fresh_gaussian":
            raise SystemExit(f"{arm} target is not native Gaussian")
        if sampling.get("target_mixed_with_source_latent") is not False:
            raise SystemExit(f"{arm} target was source mixed")
        if condition.get("source_frame_indices") != wanted["refs"]:
            raise SystemExit(f"{arm} reference indices differ")
        if condition.get("full_source_video_count") != wanted["videos"]:
            raise SystemExit(f"{arm} source-video count differs")
        if condition.get("reference_from_temporal_video_latent_slice") is not False:
            raise SystemExit(f"{arm} reference encoding differs")
        output = receipt["outputs"][arm]
        path = Path(output["path"])
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"{arm} output is absent")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != output.get("sha256"):
            raise SystemExit(f"{arm} output digest differs")
        if output.get("frame_count") != 81 or output.get("fps") != 25:
            raise SystemExit(f"{arm} encoded metadata differs")
        clean = output.get("normalized_clean_latent")
        if not isinstance(clean, dict):
            raise SystemExit(f"{arm} pre-decode clean latent receipt is absent")
        clean_path = Path(str(clean.get("path", "")))
        if not clean_path.is_file() or clean_path.is_symlink():
            raise SystemExit(f"{arm} pre-decode clean latent is absent")
        if hashlib.sha256(clean_path.read_bytes()).hexdigest() != clean.get("sha256"):
            raise SystemExit(f"{arm} pre-decode clean latent digest differs")
        clean_shape = clean.get("shape")
        if not isinstance(clean_shape, list) or clean_shape[:3] != [1, 16, 21] or clean.get("stored_dtype") != "torch.float32":
            raise SystemExit(f"{arm} pre-decode clean latent geometry differs")
        if clean.get("native_sampler_before_vae_decode") is not True:
            raise SystemExit(f"{arm} clean latent was not captured before decode")
        if clean.get("mp4_decode_reencode_used") is not False:
            raise SystemExit(f"{arm} clean latent used a lossy MP4 round trip")
        outputs[arm] = str(path)
if set(outputs) != {"t2v", "r2v", "rv2v"} or len(set(outputs.values())) != 3:
    raise SystemExit("three distinct native outputs were not published")
print("BERNINI_NATIVE_IDENTITY_DUAL4_STRONG_AUDIT_OK")
PY

grep -q "BERNINI_NATIVE_IDENTITY_DUAL4_STRONG_AUDIT_OK" "${strong_audit_log}" || fail "strong audit marker failed"
cat -- "${strong_audit_log}"

echo "[bernini-native-dual4] outputs=${t2v_r2v_output}/t2v.mp4,${t2v_r2v_output}/r2v.mp4,${rv2v_output}/rv2v.mp4"
