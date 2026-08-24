#!/usr/bin/env bash
#SBATCH --job-name=bern-dclr-src-vae
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=01:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# Single-GPU, source-only VAE materialization.  The launcher has no target,
# edited-video, parquet, posterior-blob, or distributed input surface.

source_archive="${DCLR_SOURCE_ONLY_SOURCE_ARCHIVE:?set DCLR_SOURCE_ONLY_SOURCE_ARCHIVE}"
source_archive_sha256="${DCLR_SOURCE_ONLY_SOURCE_ARCHIVE_SHA256:?set DCLR_SOURCE_ONLY_SOURCE_ARCHIVE_SHA256}"
source_revision="${DCLR_SOURCE_ONLY_SOURCE_REVISION:?set DCLR_SOURCE_ONLY_SOURCE_REVISION}"
checkpoint="${DCLR_SOURCE_ONLY_CHECKPOINT:?set DCLR_SOURCE_ONLY_CHECKPOINT}"
checkpoint_manifest="${DCLR_SOURCE_ONLY_CHECKPOINT_CONTENT_MANIFEST:?set DCLR_SOURCE_ONLY_CHECKPOINT_CONTENT_MANIFEST}"
iid="${DCLR_SOURCE_ONLY_IID:?set DCLR_SOURCE_ONLY_IID}"
source_video="${DCLR_SOURCE_ONLY_SOURCE_VIDEO:?set DCLR_SOURCE_ONLY_SOURCE_VIDEO}"
source_video_sha256="${DCLR_SOURCE_ONLY_SOURCE_VIDEO_SHA256:?set DCLR_SOURCE_ONLY_SOURCE_VIDEO_SHA256}"
bucket_height="${DCLR_SOURCE_ONLY_BUCKET_HEIGHT:?set DCLR_SOURCE_ONLY_BUCKET_HEIGHT}"
bucket_width="${DCLR_SOURCE_ONLY_BUCKET_WIDTH:?set DCLR_SOURCE_ONLY_BUCKET_WIDTH}"
output_dir="${DCLR_SOURCE_ONLY_OUTPUT_DIR:?set DCLR_SOURCE_ONLY_OUTPUT_DIR}"
python_bin="${DCLR_SOURCE_ONLY_PYTHON_BIN:?set DCLR_SOURCE_ONLY_PYTHON_BIN}"

pinned_checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
pinned_checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
pinned_vae_config_sha256="f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527"
checkpoint_tree_sha256="${DCLR_SOURCE_ONLY_CHECKPOINT_TREE_SHA256:-${pinned_checkpoint_tree_sha256}}"
checkpoint_manifest_sha256="${DCLR_SOURCE_ONLY_CHECKPOINT_CONTENT_MANIFEST_SHA256:-${pinned_checkpoint_manifest_sha256}}"
vae_config_sha256="${DCLR_SOURCE_ONLY_VAE_CONFIG_SHA256:-${pinned_vae_config_sha256}}"

fail() {
  echo "[bernini-dclr-source-only] ERROR: $*" >&2
  exit 2
}

require_absolute() {
  local value="$1"
  local label="$2"
  [[ "${value}" == /* ]] || fail "${label} must be absolute"
}

require_plain_file() {
  local value="$1"
  local label="$2"
  [[ -f "${value}" && ! -L "${value}" ]] || fail "${label} must be a plain file"
}

require_sha256() {
  local value="$1"
  local label="$2"
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || fail "${label} must be a lowercase SHA-256"
}

for value_name in source_archive checkpoint checkpoint_manifest source_video output_dir python_bin; do
  require_absolute "${!value_name}" "${value_name}"
done
for hash_name in \
  source_archive_sha256 source_video_sha256 checkpoint_tree_sha256 \
  checkpoint_manifest_sha256 vae_config_sha256
do
  require_sha256 "${!hash_name}" "${hash_name}"
done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be a full lowercase SHA-1"
[[ "${iid}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "IID is unsafe"
[[ "${bucket_height}" =~ ^[0-9]+$ && "${bucket_width}" =~ ^[0-9]+$ ]] || fail "bucket dimensions must be positive integers"
(( bucket_height > 0 && bucket_width > 0 )) || fail "bucket dimensions must be positive"
(( bucket_height % 16 == 0 && bucket_width % 16 == 0 )) || fail "bucket dimensions must be divisible by 16"
(( bucket_height * bucket_width <= 245760 )) || fail "bucket exceeds pinned pixel budget"
[[ "${checkpoint_tree_sha256}" == "${pinned_checkpoint_tree_sha256}" ]] || fail "checkpoint tree differs from pinned release"
[[ "${checkpoint_manifest_sha256}" == "${pinned_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest differs from pinned release"
[[ "${vae_config_sha256}" == "${pinned_vae_config_sha256}" ]] || fail "Wan VAE config differs from pinned release"

source_archive="$(realpath -e -- "${source_archive}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
source_video="$(realpath -e -- "${source_video}")"
python_bin="$(realpath -e -- "${python_bin}")"
require_plain_file "${source_archive}" "method source archive"
require_plain_file "${checkpoint_manifest}" "checkpoint content manifest"
require_plain_file "${source_video}" "raw source MP4"
[[ "${source_video}" == *.mp4 ]] || fail "source input must be MP4"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint root is invalid"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "runtime Python is invalid"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive embedded revision differs"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest hash differs"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${source_video_sha256}" ]] || fail "raw source MP4 hash differs"
[[ "$(sha256sum "${checkpoint}/vae/config.json" | awk '{print $1}')" == "${vae_config_sha256}" ]] || fail "checkpoint VAE config hash differs"

[[ "${output_dir}" != / ]] || fail "output directory cannot be filesystem root"
output_basename="${output_dir##*/}"
[[ "${output_basename}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename is unsafe"
[[ "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output directory must be canonical"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "output directory must be fresh"
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
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "invalid Slurm job ID"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent is invalid"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-dclr-source-only-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  local exit_status=$?
  local cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/bernini-dclr-source-only-${SLURM_JOB_ID}."*) ;;
    *) echo "[bernini-dclr-source-only] refusing unsafe scratch cleanup" >&2; cleanup_failed=1 ;;
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
    echo "[bernini-dclr-source-only] PASS iid=${iid} source_only=true target_access=false artifact=${output_dir}/source.normalized-clean-latent.safetensors"
  else
    echo "[bernini-dclr-source-only] EXIT status=${exit_status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${exit_status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p -- \
  "${task_scratch}/source-tree" \
  "${task_scratch}/cache/miopen-user" \
  "${task_scratch}/cache/miopen-custom" \
  "${task_scratch}/cache/torch-extensions" \
  "${task_scratch}/cache/triton" \
  "${task_scratch}/cache/xdg" \
  "${task_scratch}/cache/pycache" \
  "${task_scratch}/tmp"
archive_copy="${task_scratch}/method-source.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod 0444 -- "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged archive hash differs"
[[ "$(git get-tar-commit-id <"${archive_copy}")" == "${source_revision}" ]] || fail "staged archive revision differs"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

required = {
    "methods/bernini_action_editing/materialize_dclr_source_condition.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/source_kv_replay.py",
    "methods/bernini_action_editing/source_kv_route_batches.py",
    "methods/bernini_action_editing/tools/materialize_vae.py",
}
seen = set()
with tarfile.open(sys.argv[1], "r:") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit("archive member escaped repository-relative closure")
        normalized = path.as_posix().lstrip("./")
        if not normalized or normalized in seen:
            raise SystemExit("archive contains an empty or duplicate member")
        seen.add(normalized)
        if member.issym() or member.islnk() or member.isfifo() or member.isdev():
            raise SystemExit("archive contains a link or device")
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"archive lacks source-only materializer closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
[[ -d "${method_root}" && ! -L "${method_root}" ]] || fail "extracted method root is invalid"
find "${method_root}" -type f -exec chmod a-w -- {} +
for required in \
  materialize_dclr_source_condition.py infer_source_kv_carrier_oracle.py \
  infer_lora.py train_lora.py source_kv_replay.py source_kv_route_batches.py \
  tools/materialize_vae.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "missing staged ${required}"
done

allocated_rocr_device="${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
[[ "${allocated_rocr_device}" =~ ^[0-9]+$ ]] || fail "source materializer requires exactly one numeric ROCr device"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export ROCR_VISIBLE_DEVICES="${allocated_rocr_device}"
export MIOPEN_USER_DB_PATH="${task_scratch}/cache/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/cache/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/cache/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/cache/triton"
export XDG_CACHE_HOME="${task_scratch}/cache/xdg"
export PYTHONPYCACHEPREFIX="${task_scratch}/cache/pycache"
export TMPDIR="${task_scratch}/tmp"

runtime_log="${task_scratch}/runtime.log"
echo "[bernini-dclr-source-only] iid=${iid} source_only=true target_path=false target_columns=false target_posterior=false target_media=false gpu_count=1"
"${python_bin}" -B "${method_root}/materialize_dclr_source_condition.py" \
  --iid "${iid}" \
  --source-video "${source_video}" \
  --expected-source-sha256 "${source_video_sha256}" \
  --expected-bucket-hw "${bucket_height}" "${bucket_width}" \
  --checkpoint "${checkpoint}" \
  --checkpoint-content-manifest "${checkpoint_manifest}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}" \
  --expected-vae-config-sha256 "${vae_config_sha256}" \
  --output-dir "${output_dir}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  >"${runtime_log}" 2>&1 || {
    tail -n 200 -- "${runtime_log}" >&2 || true
    fail "source-only Wan VAE materialization failed"
  }

receipt_path="${output_dir}/receipt.json"
artifact_path="${output_dir}/source.normalized-clean-latent.safetensors"
[[ -f "${receipt_path}" && ! -L "${receipt_path}" ]] || fail "receipt is missing"
[[ -f "${artifact_path}" && ! -L "${artifact_path}" ]] || fail "source artifact is missing"

audit_outputs() {
  "${python_bin}" -B - \
    "${receipt_path}" "${artifact_path}" \
    "${iid}" "${source_video}" "${source_video_sha256}" \
    "${bucket_height}" "${bucket_width}" \
    "${checkpoint}" "${checkpoint_manifest}" "${checkpoint_tree_sha256}" \
    "${checkpoint_manifest_sha256}" "${vae_config_sha256}" \
    "${source_revision}" "${source_archive_sha256}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from safetensors import safe_open

(
    receipt_value,
    artifact_value,
    iid,
    source_value,
    source_sha256,
    bucket_height,
    bucket_width,
    checkpoint_value,
    manifest_value,
    checkpoint_tree_sha256,
    manifest_sha256,
    vae_config_sha256,
    source_revision,
    source_archive_sha256,
) = sys.argv[1:]

sha_re = re.compile(r"[0-9a-f]{64}")

def require(condition, message):
    if not condition:
        raise SystemExit(message)

def plain_file(value):
    requested = Path(value)
    require(requested.is_absolute() and requested.is_file() and not requested.is_symlink(), f"not a plain file: {requested}")
    return requested.resolve(strict=True)

def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def canonical_sha(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

receipt_path = plain_file(receipt_value)
artifact_path = plain_file(artifact_value)
source_path = plain_file(source_value)
manifest_path = plain_file(manifest_value)
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
unsigned = dict(receipt)
declared_digest = unsigned.pop("receipt_digest", None)
require(declared_digest == canonical_sha(unsigned), "receipt self digest differs")
require(
    receipt.get("schema_version") == "bernini-source-only-vae-materialization-v1"
    and receipt.get("method") == "bernini-dclr-source-only-condition-materializer-v1"
    and receipt.get("source_only") is True
    and receipt.get("method_source_revision") == source_revision
    and receipt.get("method_source_archive_sha256") == source_archive_sha256,
    "materializer provenance differs",
)
source = receipt.get("input")
media = source.get("media") if isinstance(source, dict) else None
require(
    isinstance(source, dict)
    and source.get("source_iid") == iid
    and source.get("source_video_path") == str(source_path)
    and source.get("source_video_sha256") == source_sha256
    and source.get("input_kind") == "raw_source_mp4"
    and isinstance(media, dict)
    and media.get("frame_count") == 81
    and float(media.get("fps")) == 25.0
    and media.get("source_derived_bucket_hw") == [int(bucket_height), int(bucket_width)]
    and media.get("temporal_policy") == "all_integer_frames_0_through_80_no_subsampling"
    and media.get("decoded_from_private_byte_snapshot") is True
    and media.get("snapshot_sha256") == source_sha256
    and media.get("original_pre_snapshot_sha256") == source_sha256
    and media.get("original_post_snapshot_sha256") == source_sha256
    and media.get("original_stat_identity_stable") is True,
    "source byte/media binding differs",
)
closed = receipt.get("access_audit")
require(
    closed == {
        "source_columns_accessed": ["iid", "source_video", "source_video_sha256"],
        "target_columns_accessed": [],
        "target_media_accessed": False,
        "paired_target_accessed": False,
        "source_video_argument_exposed": True,
        "source_path_accessed": str(source_path),
        "source_media_decoded": True,
        "source_vae_posterior_mode_accessed": True,
        "source_directory_enumerated": False,
        "parquet_argument_exposed": False,
        "parquet_path_accessed": None,
        "parquet_columns_accessed": [],
        "target_path_argument_exposed": False,
        "target_path_accessed": None,
        "target_vae_posterior_accessed": False,
        "edited_sibling_path_inferred_or_accessed": False,
    },
    "closed source-only input surface differs",
)
checkpoint = receipt.get("checkpoint")
content = checkpoint.get("content") if isinstance(checkpoint, dict) else None
vae_config = checkpoint.get("vae_config") if isinstance(checkpoint, dict) else None
require(
    isinstance(checkpoint, dict)
    and checkpoint.get("path") == str(Path(checkpoint_value).resolve())
    and checkpoint.get("tree_sha256") == checkpoint_tree_sha256
    and checkpoint.get("vae_config_sha256") == vae_config_sha256
    and isinstance(content, dict)
    and content.get("manifest_path") == str(manifest_path)
    and content.get("manifest_sha256_computed") == manifest_sha256
    and content.get("manifest_sha256_expected") == manifest_sha256
    and content.get("verified_file_count") == 23
    and content.get("every_file_sha256_verified") is True
    and sha_re.fullmatch(str(content.get("verified_entries_digest"))) is not None
    and isinstance(vae_config, dict)
    and vae_config.get("class_name") == "AutoencoderKLWan"
    and vae_config.get("z_dim") == 16
    and vae_config.get("temperal_downsample") == [False, True, True]
    and len(vae_config.get("latents_mean", [])) == 16
    and len(vae_config.get("latents_std", [])) == 16,
    "pinned checkpoint/VAE binding differs",
)
expected_shape = [1, 16, 21, int(bucket_height) // 8, int(bucket_width) // 8]
encoding = receipt.get("encoding")
require(
    isinstance(encoding, dict)
    and encoding.get("vae_class") == "AutoencoderKLWan"
    and encoding.get("posterior_statistic") == "mode"
    and encoding.get("normalization") == "(posterior_mode-latents_mean)/latents_std"
    and encoding.get("latent_coordinate") == "bernini_normalized_clean_vae_latent"
    and encoding.get("latent_shape") == expected_shape
    and encoding.get("latent_dtype") == "torch.float32"
    and encoding.get("before_any_vae_decode") is True
    and encoding.get("target_posterior_used") is False,
    "Wan source encoding contract differs",
)
artifact = receipt.get("source_condition_artifact")
identity = artifact.get("tensor_identity") if isinstance(artifact, dict) else None
require(
    isinstance(artifact, dict)
    and artifact.get("path") == str(artifact_path)
    and artifact.get("sha256") == file_sha(artifact_path)
    and artifact.get("tensor_key") == "normalized_clean_latent"
    and artifact.get("shape") == expected_shape
    and artifact.get("artifact_role") == "source_video_condition"
    and artifact.get("coordinate") == "bernini_normalized_clean_vae_latent"
    and artifact.get("frame_contract") == "exact81_latent21"
    and artifact.get("metadata") == {
        "coordinate": "bernini_normalized_clean_vae_latent",
        "frame_contract": "exact81_latent21",
        "artifact_role": "source_video_condition",
        "source": "source_video_vae_encode_before_any_decode",
    }
    and artifact.get("stored_dtype") == "torch.float32"
    and artifact.get("source_video_vae_encode_before_any_decode") is True
    and artifact.get("mp4_decode_reencode_used") is False
    and artifact.get("roundtrip_tensor_exact") is True
    and isinstance(identity, dict)
    and identity.get("shape") == expected_shape
    and identity.get("dtype") == "torch.float32"
    and identity.get("finite") is True
    and identity.get("numel") == math.prod(expected_shape)
    and identity.get("byte_count") == 4 * math.prod(expected_shape)
    and sha_re.fullmatch(str(identity.get("content_sha256"))) is not None
    and sha_re.fullmatch(str(identity.get("raw_storage_sha256"))) is not None,
    "source latent artifact receipt differs",
)
with safe_open(str(artifact_path), framework="pt", device="cpu") as opened:
    require(list(opened.keys()) == ["normalized_clean_latent"], "safetensors key differs")
    require(dict(opened.metadata() or {}) == artifact["metadata"], "safetensors metadata differs")
    tensor = opened.get_tensor("normalized_clean_latent")
require(str(tensor.dtype) == "torch.float32" and list(tensor.shape) == expected_shape, "stored tensor dtype/shape differs")
require(bool(tensor.isfinite().all().item()), "stored source tensor is non-finite")
require(
    receipt.get("single_gpu") is True
    and receipt.get("distributed") is False
    and receipt.get("training_performed") is False
    and receipt.get("optimizer_constructed") is False
    and receipt.get("engineering_materialization_only") is True
    and receipt.get("training_pair_authorized") is False
    and receipt.get("source_reward_calibration_authorized") is False
    and receipt.get("scientific_claim_authorized") is False
    and receipt.get("production_claim_forbidden") is True
    and receipt.get("runtime", {}).get("device_count") == 1
    and receipt.get("runtime", {}).get("diffusers") == "0.38.0",
    "claim/device state differs",
)
require(file_sha(source_path) == source_sha256, "source MP4 changed after materialization")
require(file_sha(manifest_path) == manifest_sha256, "checkpoint manifest changed after materialization")
require(file_sha(Path(checkpoint_value).resolve() / "vae/config.json") == vae_config_sha256, "VAE config changed after materialization")
print("BERNINI_DCLR_SOURCE_ONLY_MATERIALIZATION_AUDIT_OK")
PY
}

receipt_sha_round1="$(sha256sum "${receipt_path}" | awk '{print $1}')"
artifact_sha_round1="$(sha256sum "${artifact_path}" | awk '{print $1}')"
audit_outputs || fail "output audit round 1 failed"
[[ "$(sha256sum "${receipt_path}" | awk '{print $1}')" == "${receipt_sha_round1}" ]] || fail "receipt changed during audit"
[[ "$(sha256sum "${artifact_path}" | awk '{print $1}')" == "${artifact_sha_round1}" ]] || fail "artifact changed during audit"
audit_outputs || fail "output audit round 2 failed"
[[ "$(sha256sum "${receipt_path}" | awk '{print $1}')" == "${receipt_sha_round1}" ]] || fail "receipt hash closure differs"
[[ "$(sha256sum "${artifact_path}" | awk '{print $1}')" == "${artifact_sha_round1}" ]] || fail "artifact hash closure differs"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${source_video_sha256}" ]] || fail "source MP4 changed"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest changed"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "durable archive changed"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged archive changed"
cp -- "${runtime_log}" "${output_dir}/runtime.log"
chmod 0444 -- "${receipt_path}" "${artifact_path}" "${output_dir}/runtime.log"
echo "[bernini-dclr-source-only] COMPLETE receipt_sha256=${receipt_sha_round1} artifact_sha256=${artifact_sha_round1} audit=BERNINI_DCLR_SOURCE_ONLY_MATERIALIZATION_AUDIT_OK"
