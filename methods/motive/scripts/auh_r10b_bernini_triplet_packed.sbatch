#!/usr/bin/env bash
#SBATCH --job-name=motive-r10b-bernini-triplet
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --gres=gpu:mi210:3
#SBATCH --time=08:00:00

set -Eeuo pipefail

source_snapshot="${MOTIVE_R10B_SOURCE_SNAPSHOT:?set MOTIVE_R10B_SOURCE_SNAPSHOT}"
source_tree_sha256="${MOTIVE_R10B_SOURCE_TREE_SHA256:?set MOTIVE_R10B_SOURCE_TREE_SHA256}"
qwen_packed_root="${MOTIVE_R10B_QWEN_PACKED_ROOT:?set MOTIVE_R10B_QWEN_PACKED_ROOT}"
qwen_packed_commit_digest="${MOTIVE_R10B_QWEN_PACKED_COMMIT_DIGEST:?set MOTIVE_R10B_QWEN_PACKED_COMMIT_DIGEST}"
track_cache="${MOTIVE_R10B_TRACK_CACHE:?set MOTIVE_R10B_TRACK_CACHE}"
track_cache_sha256="${MOTIVE_R10B_TRACK_CACHE_SHA256:?set MOTIVE_R10B_TRACK_CACHE_SHA256}"
model_path="${MOTIVE_R10B_MODEL_PATH:?set MOTIVE_R10B_MODEL_PATH}"
model_revision="${MOTIVE_R10B_MODEL_REVISION:?set MOTIVE_R10B_MODEL_REVISION}"
model_tree_sha256="${MOTIVE_R10B_MODEL_TREE_SHA256:?set MOTIVE_R10B_MODEL_TREE_SHA256}"
bernini_repo="${MOTIVE_R10B_BERNINI_REPO:?set MOTIVE_R10B_BERNINI_REPO}"
bernini_source_commit="${MOTIVE_R10B_BERNINI_SOURCE_COMMIT:?set MOTIVE_R10B_BERNINI_SOURCE_COMMIT}"
bernini_source_bundle_sha256="${MOTIVE_R10B_BERNINI_SOURCE_BUNDLE_SHA256:?set MOTIVE_R10B_BERNINI_SOURCE_BUNDLE_SHA256}"
output_root="${MOTIVE_R10B_OUTPUT_ROOT:?set MOTIVE_R10B_OUTPUT_ROOT}"
python_bin="${MOTIVE_R10B_PYTHON_BIN:?set MOTIVE_R10B_PYTHON_BIN}"

scheduler_index="${MOTIVE_R10B_SCHEDULER_INDEX:-25}"
noise_mode="${MOTIVE_R10B_NOISE_MODE:-iid_spatiotemporal}"
width="${MOTIVE_R10B_WIDTH:-256}"
height="${MOTIVE_R10B_HEIGHT:-256}"
projection_dim="${MOTIVE_R10B_PROJECTION_DIM:-2048}"

artifact_kind="controlled_retrieval_pilot"
resize_mode="aspect_preserving_center_crop"
num_frames="17"
scheduler_steps="50"
projection_seed_1="260108851"
projection_seed_2="260108852"
diffusion_noise_seed="260108853"
torch_seed="260108854"
task_count="3"

require_absolute_path() {
  local label="$1"
  local path="$2"
  if [[ "${path}" != /* || "${path}" == "/" ]]; then
    echo "${label} must be a non-root absolute path: ${path}" >&2
    exit 2
  fi
}

require_sha256() {
  local label="$1"
  local digest="$2"
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "${label} must be one lowercase SHA-256" >&2
    exit 2
  fi
}

require_git_commit() {
  local label="$1"
  local digest="$2"
  if [[ ! "${digest}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "${label} must be one lowercase Git commit" >&2
    exit 2
  fi
}

require_positive_integer() {
  local label="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${label} must be one positive integer: ${value}" >&2
    exit 2
  fi
}

for binding in \
  "source_snapshot:${source_snapshot}" \
  "qwen_packed_root:${qwen_packed_root}" \
  "track_cache:${track_cache}" \
  "model_path:${model_path}" \
  "bernini_repo:${bernini_repo}" \
  "output_root:${output_root}" \
  "python_bin:${python_bin}"; do
  require_absolute_path "${binding%%:*}" "${binding#*:}"
done
for binding in \
  "source_tree_sha256:${source_tree_sha256}" \
  "qwen_packed_commit_digest:${qwen_packed_commit_digest}" \
  "track_cache_sha256:${track_cache_sha256}" \
  "model_tree_sha256:${model_tree_sha256}" \
  "bernini_source_bundle_sha256:${bernini_source_bundle_sha256}"; do
  require_sha256 "${binding%%:*}" "${binding#*:}"
done
require_git_commit "model_revision" "${model_revision}"
require_git_commit "bernini_source_commit" "${bernini_source_commit}"
require_positive_integer "width" "${width}"
require_positive_integer "height" "${height}"
require_positive_integer "projection_dim" "${projection_dim}"

if [[ ! "${scheduler_index}" =~ ^[0-9]+$ ]] \
  || (( scheduler_index >= scheduler_steps )); then
  echo "scheduler index must be in [0, ${scheduler_steps}): ${scheduler_index}" >&2
  exit 2
fi
if [[ "${noise_mode}" != "iid_spatiotemporal" \
  && "${noise_mode}" != "temporal_broadcast" ]]; then
  echo "unsupported tangent noise mode: ${noise_mode}" >&2
  exit 2
fi
if [[ "${SLURM_NTASKS:-}" != "${task_count}" ]]; then
  echo "Bernini triplet job requires exactly 3 real Slurm tasks" >&2
  exit 2
fi
if [[ -L "${source_snapshot}" || ! -d "${source_snapshot}" ]]; then
  echo "source snapshot must be one non-symlink directory" >&2
  exit 2
fi
if [[ -L "${qwen_packed_root}" || ! -d "${qwen_packed_root}" ]]; then
  echo "Qwen packed root must be one non-symlink directory" >&2
  exit 2
fi
if [[ -L "${track_cache}" || ! -f "${track_cache}" ]]; then
  echo "track cache must be one regular non-symlink file" >&2
  exit 2
fi
if [[ -L "${model_path}" || ! -d "${model_path}" ]]; then
  echo "model path must be one non-symlink directory" >&2
  exit 2
fi
if [[ -L "${bernini_repo}" || ! -d "${bernini_repo}" ]]; then
  echo "Bernini source must be one non-symlink directory" >&2
  exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable is unavailable: ${python_bin}" >&2
  exit 2
fi
if [[ -e "${output_root}" || -L "${output_root}" ]]; then
  echo "refusing to reuse Bernini triplet output: ${output_root}" >&2
  exit 2
fi
output_parent="${output_root%/*}"
if [[ ! -d "${output_parent}" || ! -w "${output_parent}" ]]; then
  echo "output parent must be an existing writable directory" >&2
  exit 2
fi

code_root="${source_snapshot}/methods/motive"
pilot_dir="${qwen_packed_root}/pilot_manifest"
variant_dir="${qwen_packed_root}/prompt_variants"
canonical_manifest="${pilot_dir}/manifest.jsonl"
original_manifest="${variant_dir}/original.jsonl"
cross_family_manifest="${variant_dir}/cross_family_shuffle.jsonl"
for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  "${code_root}/motive/attribution.py" \
  "${code_root}/motive/r10b_bernini_pilot_manifest.py" \
  "${code_root}/motive/r10b_bernini_prompt_variants.py" \
  "${code_root}/motive/r10b_bernini_retrieval_audit.py" \
  "${code_root}/motive/r10b_bernini_tangent_extract.py" \
  "${code_root}/motive/r10b_lucy_tangent_extract.py" \
  "${code_root}/motive/r10b_tangent_core.py" \
  "${qwen_packed_root}/pipeline_summary.json" \
  "${qwen_packed_root}/pipeline_done.json" \
  "${canonical_manifest}" \
  "${original_manifest}" \
  "${cross_family_manifest}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required Bernini triplet input: ${required}" >&2
    exit 2
  fi
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHONPATH="${source_snapshot}:${code_root}"

job_scratch="${SLURM_TMPDIR:-/tmp}/motive-r10b-bernini-triplet-${SLURM_JOB_ID:-manual}"
mkdir -p "${job_scratch}"

record_binding() {
  printf 'motive_r10b_bernini_triplet.%s=%s\n' "$1" "$2"
}

record_binding "source_snapshot" "${source_snapshot}"
record_binding "source_tree_sha256" "${source_tree_sha256}"
record_binding "qwen_packed_root" "${qwen_packed_root}"
record_binding "qwen_packed_commit_digest" "${qwen_packed_commit_digest}"
record_binding "track_cache" "${track_cache}"
record_binding "track_cache_sha256" "${track_cache_sha256}"
record_binding "model_path" "${model_path}"
record_binding "model_revision" "${model_revision}"
record_binding "model_tree_sha256" "${model_tree_sha256}"
record_binding "bernini_repo" "${bernini_repo}"
record_binding "bernini_source_commit" "${bernini_source_commit}"
record_binding "bernini_source_bundle_sha256" "${bernini_source_bundle_sha256}"
record_binding "output_root" "${output_root}"
record_binding "artifact_kind" "${artifact_kind}"
record_binding "resize_mode" "${resize_mode}"
record_binding "scheduler_steps" "${scheduler_steps}"
record_binding "scheduler_index" "${scheduler_index}"
record_binding "noise_mode" "${noise_mode}"
record_binding "width" "${width}"
record_binding "height" "${height}"
record_binding "num_frames" "${num_frames}"
record_binding "projection_dim" "${projection_dim}"
record_binding "projection_seeds" "${projection_seed_1},${projection_seed_2}"
record_binding "diffusion_noise_seed" "${diffusion_noise_seed}"
record_binding "torch_seed" "${torch_seed}"
record_binding "videos_copied" "0"
record_binding "videos_rendered" "0"
record_binding "renderer_calls" "0"
record_binding "optimizer_steps" "0"
record_binding "representation_gate_passed" "false"
record_binding "renderer_probe_authorized" "false"
record_binding "generation_authorized" "false"
record_binding "training_authorized" "false"
record_binding "downstream_submitted" "false"

verify_bound_inputs() {
  local stage="$1"
  record_binding "input_verification_stage" "${stage}"
  "${python_bin}" \
    "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
    verify \
    --snapshot "${source_snapshot}" \
    --expected-tree-sha256 "${source_tree_sha256}"

  # This first verifier intentionally has no project imports.  It validates
  # the exact balanced packed commit before any output directory or GPU task
  # can be created.
  "${python_bin}" - \
    "${qwen_packed_root}" \
    "${qwen_packed_commit_digest}" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

root = Path(sys.argv[1]).resolve(strict=True)
expected_commit_digest = sys.argv[2]

def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")

def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def strict_json(path):
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"commit JSON is not one regular file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"commit JSON is invalid: {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"commit JSON is not one object: {path}")
    expected = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if raw != expected:
        raise SystemExit(f"commit JSON is not canonical pretty JSON: {path}")
    return value, raw

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def canonical_digest(value):
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def tree_record(directory, *, excluded=()):
    excluded_set = set(excluded)
    rows = []
    total_bytes = 0
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise SystemExit(f"packed commit contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise SystemExit(f"packed commit contains a non-file: {path}")
        relative = path.relative_to(directory).as_posix()
        if relative in excluded_set:
            continue
        size = path.stat().st_size
        rows.append(
            {
                "relative_path": relative,
                "bytes": size,
                "sha256": sha256(path),
            }
        )
        total_bytes += size
    encoded = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    return {
        "path": str(directory),
        "files": len(rows),
        "bytes": total_bytes,
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
    }

expected_components = {
    "gpu_probes",
    "shard_queues",
    "shard_audits",
    "merged_audit",
    "pilot_manifest",
    "prompt_variants",
}
expected_root_names = expected_components | {
    "pipeline_summary.json",
    "pipeline_done.json",
}
observed_root_names = {path.name for path in root.iterdir()}
if observed_root_names != expected_root_names:
    raise SystemExit(
        "balanced Qwen packed closure differs: "
        f"expected={expected_root_names}, observed={observed_root_names}"
    )
for name in expected_components:
    path = root / name
    if path.is_symlink() or not path.is_dir():
        raise SystemExit(f"Qwen packed component is not one directory: {path}")

summary, summary_raw = strict_json(root / "pipeline_summary.json")
done, _done_raw = strict_json(root / "pipeline_done.json")
if (
    summary.get("schema_version")
    != "motive-r10b-qwen-v2-packed-pipeline-v1"
    or summary.get("status") != "ready_for_commit"
    or done.get("schema_version")
    != "motive-r10b-qwen-v2-packed-pipeline-done-v1"
    or done.get("status") != "complete"
):
    raise SystemExit("Qwen packed pipeline schema/status differs")
qwen_source = summary.get("inputs", {}).get("source_snapshot")
if (
    not isinstance(qwen_source, dict)
    or set(qwen_source) != {"path", "tree_sha256"}
    or not isinstance(qwen_source.get("path"), str)
    or not qwen_source["path"].startswith("/")
    or not isinstance(qwen_source.get("tree_sha256"), str)
    or len(qwen_source["tree_sha256"]) != 64
    or any(
        character not in "0123456789abcdef"
        for character in qwen_source["tree_sha256"]
    )
):
    raise SystemExit("Qwen packed recorded source snapshot/tree binding differs")
if set(summary.get("outputs", {})) != expected_components:
    raise SystemExit("Qwen packed summary output closure differs")
for name in expected_components:
    if summary["outputs"][name] != tree_record(root / name):
        raise SystemExit(f"Qwen packed output tree changed: {name}")

authorization = {
    "formal_evidence": False,
    "representation_promoted": False,
    "renderer_probe_authorized": False,
    "generation_authorized": False,
    "training_authorized": False,
}
for field in (
    "formal_evidence",
    "representation_gate_passed",
    "renderer_probe_authorized",
    "generation_authorized",
    "training_authorized",
    "downstream_submitted",
):
    if summary.get(field) is not False:
        raise SystemExit(f"Qwen packed summary gate is open: {field}")
if (
    summary.get("authorization") != authorization
    or done.get("authorization") != authorization
    or summary.get("balanced_pilot_ready") is not True
    or summary.get("prompt_variants_materialized") is not True
    or done.get("balanced_pilot_ready") is not True
    or summary.get("videos_copied") != 0
    or summary.get("videos_rendered") != 0
    or summary.get("optimizer_steps") != 0
    or done.get("real_gpu_tasks_verified") != 8
    or done.get("production_backend_cuda_only") is not True
    or done.get("generation_error_rows") != 0
):
    raise SystemExit("Qwen packed balanced/runtime/closed-scope gate differs")
execution = summary.get("execution", {})
backend = execution.get("backend_execution", {})
if (
    execution.get("real_gpu_tasks") != 8
    or execution.get("mi210_gpus") != 8
    or execution.get("generation_error_rows") != 0
    or backend.get("mode") != "production_local_qwen"
    or backend.get("cuda_only") is not True
):
    raise SystemExit("Qwen packed production execution evidence differs")

controlled_digest = summary.get("controlled_pilot_commit_digest")
if (
    not isinstance(controlled_digest, str)
    or len(controlled_digest) != 64
    or any(character not in "0123456789abcdef" for character in controlled_digest)
    or done.get("controlled_pilot_commit_digest") != controlled_digest
):
    raise SystemExit("Qwen packed controlled-pilot digest differs")
summary_record = {
    "bytes": len(summary_raw),
    "sha256": hashlib.sha256(summary_raw).hexdigest(),
}
output_tree = tree_record(root, excluded={"pipeline_done.json"})
binding = {
    "pipeline_summary": summary_record,
    "output_tree_without_done": output_tree,
    "controlled_pilot_commit_digest": controlled_digest,
}
if (
    done.get("pipeline_summary") != summary_record
    or done.get("output_tree_without_done") != output_tree
    or done.get("commit_digest") != canonical_digest(binding)
    or done.get("commit_digest") != expected_commit_digest
):
    raise SystemExit("Qwen packed pipeline_done closure differs")
print(f"motive_r10b_bernini_triplet.qwen_commit_digest={done['commit_digest']}")
print(
    "motive_r10b_bernini_triplet.controlled_pilot_commit_digest="
    + controlled_digest
)
PY

  # Project validators bind the controlled pilot, its two prompt transforms,
  # the track cache, the model bytes/revision, and the vendored Bernini source.
  "${python_bin}" - \
    "${pilot_dir}" \
    "${variant_dir}" \
    "${track_cache}" \
    "${track_cache_sha256}" \
    "${model_path}" \
    "${model_revision}" \
    "${model_tree_sha256}" \
    "${bernini_repo}" \
    "${bernini_source_commit}" \
    "${bernini_source_bundle_sha256}" \
    "${canonical_manifest}" \
    "${original_manifest}" \
    "${cross_family_manifest}" \
    "${qwen_packed_root}" \
    "${qwen_packed_commit_digest}" <<'PY'
from pathlib import Path
import hashlib
import json
import re
import sys

from motive.r10b_bernini_prompt_variants import validate_prompt_variants
from motive.r10b_bernini_retrieval_audit import (
    validate_controlled_pilot_commit,
)
from motive.r10b_bernini_tangent_extract import (
    EXPECTED_REPO_REVISION,
    EXPECTED_SOURCE_BUNDLE_SHA256,
    EXPECTED_SOURCE_COMMIT,
    _bernini_source_manifest,
    _checkpoint_manifest,
    _checkpoint_revision_metadata,
)

(
    pilot_raw,
    variants_raw,
    track_raw,
    expected_track_sha256,
    model_raw,
    model_revision,
    expected_model_tree,
    bernini_raw,
    source_commit,
    expected_source_bundle,
    canonical_raw,
    original_raw,
    cross_raw,
    qwen_raw,
    expected_qwen_commit_digest,
) = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

pilot = Path(pilot_raw).resolve(strict=True)
variants = Path(variants_raw).resolve(strict=True)
track = Path(track_raw).resolve(strict=True)
model = Path(model_raw).resolve(strict=True)
bernini = Path(bernini_raw).resolve(strict=True)
qwen = Path(qwen_raw).resolve(strict=True)
if any(path.is_symlink() for path in (pilot, variants, track, model, bernini)):
    raise SystemExit("a bound Bernini triplet input is a symlink")
if sha256(track) != expected_track_sha256:
    raise SystemExit("track-cache SHA-256 differs")
pilot_result = validate_controlled_pilot_commit(pilot)
if (
    pilot_result.get("status") != "VALID"
    or pilot_result.get("balanced_pilot_ready") is not True
    or pilot_result.get("representation_gate_passed") is not False
    or pilot_result.get("renderer_probe_authorized") is not False
    or pilot_result.get("training_authorized") is not False
):
    raise SystemExit("controlled-pilot balanced/closed gate differs")
qwen_summary = json.loads(
    (qwen / "pipeline_summary.json").read_text(encoding="utf-8")
)
qwen_done = json.loads(
    (qwen / "pipeline_done.json").read_text(encoding="utf-8")
)
if (
    pilot_result["commit_digest"]
    != qwen_summary.get("controlled_pilot_commit_digest")
    or pilot_result["commit_digest"]
    != qwen_done.get("controlled_pilot_commit_digest")
    or qwen_done.get("commit_digest") != expected_qwen_commit_digest
):
    raise SystemExit("controlled pilot does not match the pinned Qwen commit")
variant_result = validate_prompt_variants(
    variant_dir=variants,
    pilot_dir=pilot,
)
if (
    variant_result.get("status") != "VALID"
    or variant_result.get("balanced_pilot_ready") is not True
    or variant_result.get("video_bytes_copied") is not False
    or variant_result.get("rendering_performed") is not False
    or variant_result.get("generation_performed") is not False
    or variant_result.get("training_performed") is not False
):
    raise SystemExit("prompt-variant commit gate differs")

expected_manifests = {
    "canonical": pilot / "manifest.jsonl",
    "original": variants / "original.jsonl",
    "cross_family": variants / "cross_family_shuffle.jsonl",
}
supplied_manifests = {
    "canonical": Path(canonical_raw).resolve(strict=True),
    "original": Path(original_raw).resolve(strict=True),
    "cross_family": Path(cross_raw).resolve(strict=True),
}
if supplied_manifests != expected_manifests:
    raise SystemExit("triplet manifest routing differs")
for tag, path in supplied_manifests.items():
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{tag} manifest is not one regular file")

if model_revision != EXPECTED_REPO_REVISION:
    raise SystemExit("Bernini model revision differs from frozen code")
if source_commit != EXPECTED_SOURCE_COMMIT:
    raise SystemExit("Bernini source commit differs from frozen code")
if expected_source_bundle != EXPECTED_SOURCE_BUNDLE_SHA256:
    raise SystemExit("Bernini source bundle differs from frozen code")
checkpoint = _checkpoint_manifest(model)
revision = _checkpoint_revision_metadata(model)
vendor = _bernini_source_manifest(bernini)
if (
    checkpoint.get("tree_sha256") != expected_model_tree
    or revision.get("revision") != model_revision
    or vendor.get("bundle_sha256") != expected_source_bundle
):
    raise SystemExit("Bernini model/vendor immutable binding differs")
if re.fullmatch(r"[0-9a-f]{64}", pilot_result["commit_digest"]) is None:
    raise SystemExit("controlled-pilot commit digest is invalid")
print(
    "motive_r10b_bernini_triplet.prompt_variants_commit_digest="
    + variant_result["commit_digest"]
)
print(
    "motive_r10b_bernini_triplet.verified_model_tree_sha256="
    + checkpoint["tree_sha256"]
)
print(
    "motive_r10b_bernini_triplet.verified_source_bundle_sha256="
    + vendor["bundle_sha256"]
)
PY
}

probe_bound_gpu() {
  local probe_path="$1"
  local tag="$2"
  "${MOTIVE_TRIPLET_PYTHON}" - \
    "${probe_path}" \
    "${tag}" \
    "${SLURM_PROCID:?SLURM_PROCID is required}" \
    "${SLURM_LOCALID:?SLURM_LOCALID is required}" \
    "${SLURM_NTASKS:?SLURM_NTASKS is required}" <<'PY'
from pathlib import Path
import json
import os
import sys
import tempfile

import torch

probe_path = Path(sys.argv[1])
tag = sys.argv[2]
rank = int(sys.argv[3])
local_rank = int(sys.argv[4])
slurm_ntasks = int(sys.argv[5])
tags = ("canonical", "original", "cross_family")
if (
    rank not in range(3)
    or local_rank != rank
    or slurm_ntasks != 3
    or tag != tags[rank]
):
    raise SystemExit("Bernini triplet GPU rank/tag topology differs")
if os.path.lexists(probe_path) or not probe_path.parent.is_dir():
    raise SystemExit(f"refusing invalid/reused GPU probe: {probe_path}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Bernini triplet rank must see exactly one GPU")
device_name = str(torch.cuda.get_device_name(0))
if "MI210" not in device_name.upper():
    raise SystemExit(f"Bernini triplet rank GPU is not MI210: {device_name}")
properties = torch.cuda.get_device_properties(0)
device_uuid = getattr(properties, "uuid", None)
if device_uuid is None and hasattr(torch.cuda, "get_device_uuid"):
    try:
        device_uuid = torch.cuda.get_device_uuid(0)
    except Exception:
        device_uuid = None
if device_uuid is not None:
    device_uuid = str(device_uuid)
tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
tensor_value = float((tensor * tensor).sum().item())
torch.cuda.synchronize(0)
if str(tensor.device) != "cuda:0" or tensor_value != 14.0:
    raise SystemExit("Bernini triplet cuda:0 tensor probe differs")
visibility_names = (
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NTASKS",
    "SLURM_JOB_ID",
    "SLURMD_NODENAME",
    "HOSTNAME",
)
payload = {
    "schema_version": "motive-r10b-bernini-triplet-packed-gpu-probe-v1",
    "status": "passed",
    "tag": tag,
    "rank": rank,
    "local_rank": local_rank,
    "slurm_ntasks": slurm_ntasks,
    "device_count": 1,
    "device_index": 0,
    "device_name": device_name,
    "device_uuid": device_uuid,
    "tensor_probe": {
        "device": str(tensor.device),
        "sum_of_squares": tensor_value,
        "passed": True,
    },
    "visibility": {name: os.environ.get(name) for name in visibility_names},
}
raw = (
    json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
).encode("utf-8")
descriptor, temporary_raw = tempfile.mkstemp(
    prefix=f".{probe_path.name}.tmp.",
    dir=str(probe_path.parent),
)
temporary = Path(temporary_raw)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, probe_path)
    directory_descriptor = os.open(probe_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
finally:
    if temporary.exists():
        temporary.unlink()
PY
}

require_closed_artifact() {
  local tag="$1"
  local artifact_dir="$2"
  local manifest="$3"
  "${MOTIVE_TRIPLET_PYTHON}" - \
    "${tag}" \
    "${artifact_dir}" \
    "${manifest}" \
    "${MOTIVE_TRIPLET_TRACK_CACHE}" \
    "${MOTIVE_TRIPLET_TRACK_CACHE_SHA256}" \
    "${MOTIVE_TRIPLET_MODEL_PATH}" \
    "${MOTIVE_TRIPLET_MODEL_REVISION}" \
    "${MOTIVE_TRIPLET_MODEL_TREE_SHA256}" \
    "${MOTIVE_TRIPLET_BERNINI_SOURCE_COMMIT}" \
    "${MOTIVE_TRIPLET_BERNINI_SOURCE_BUNDLE_SHA256}" \
    "${MOTIVE_TRIPLET_SOURCE_TREE_SHA256}" \
    "${MOTIVE_TRIPLET_SCHEDULER_INDEX}" \
    "${MOTIVE_TRIPLET_NOISE_MODE}" \
    "${MOTIVE_TRIPLET_WIDTH}" \
    "${MOTIVE_TRIPLET_HEIGHT}" \
    "${MOTIVE_TRIPLET_PROJECTION_DIM}" \
    "${MOTIVE_TRIPLET_PROJECTION_SEED_1}" \
    "${MOTIVE_TRIPLET_PROJECTION_SEED_2}" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

(
    tag,
    artifact_raw,
    manifest_raw,
    track_raw,
    expected_track_sha256,
    model_raw,
    model_revision,
    expected_model_tree,
    source_commit,
    source_bundle,
    source_tree,
    scheduler_index,
    noise_mode,
    width,
    height,
    projection_dim,
    projection_seed_1,
    projection_seed_2,
) = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

artifact = Path(artifact_raw).resolve(strict=True)
manifest = Path(manifest_raw).resolve(strict=True)
track = Path(track_raw).resolve(strict=True)
model = Path(model_raw).resolve(strict=True)
if {path.name for path in artifact.iterdir()} != {
    "features.npz",
    "rows.jsonl",
    "summary.json",
    "done.json",
}:
    raise SystemExit(f"{tag} tangent artifact closure differs")
if any(path.is_symlink() for path in artifact.rglob("*")):
    raise SystemExit(f"{tag} tangent artifact contains a symlink")
summary = json.loads((artifact / "summary.json").read_text(encoding="utf-8"))
done = json.loads((artifact / "done.json").read_text(encoding="utf-8"))
measurement = summary.get("measurement", {})
contract = measurement.get("run_contract", {})
scheduler = contract.get("scheduler", {})
noise = contract.get("noise", {})
resize = contract.get("resize", {})
runtime = summary.get("runtime", {})
data = summary.get("data", {})
safety = summary.get("safety", {})
if (
    summary.get("status") != "complete"
    or summary.get("artifact_kind") != "controlled_retrieval_pilot"
    or done.get("status") != "complete"
    or done.get("artifact_kind") != "controlled_retrieval_pilot"
    or contract.get("artifact_kind") != "controlled_retrieval_pilot"
    or scheduler.get("steps") != 50
    or scheduler.get("index") != int(scheduler_index)
    or noise.get("mode") != noise_mode
    or noise.get("seed") != 260108853
    or resize.get("mode") != "aspect_preserving_center_crop"
    or resize.get("width") != int(width)
    or resize.get("height") != int(height)
    or resize.get("num_frames") != 17
    or runtime.get("width") != int(width)
    or runtime.get("height") != int(height)
    or runtime.get("num_frames") != 17
    or measurement.get("projection_dimension_per_role") != int(projection_dim)
    or measurement.get("projection_seeds")
    != [int(projection_seed_1), int(projection_seed_2)]
):
    raise SystemExit(f"{tag} frozen tangent contract differs")
if (
    data.get("manifest") != str(manifest)
    or data.get("manifest_sha256") != sha256(manifest)
    or data.get("track_cache") != str(track)
    or data.get("track_cache_sha256") != expected_track_sha256
    or sha256(track) != expected_track_sha256
    or summary.get("model", {}).get("model_path") != str(model)
    or summary.get("model", {}).get("huggingface_revision") != model_revision
    or summary.get("model", {})
    .get("checkpoint_manifest", {})
    .get("tree_sha256") != expected_model_tree
    or summary.get("official_bernini_source", {}).get("commit") != source_commit
    or summary.get("official_bernini_source", {}).get("bundle_sha256")
    != source_bundle
    or summary.get("source_tree_sha256") != source_tree
):
    raise SystemExit(f"{tag} tangent provenance binding differs")
closed = {
    "optimizer_created": False,
    "optimizer_steps": 0,
    "scheduler_steps_executed": 0,
    "renderer_calls": 0,
    "videos_rendered": 0,
    "videos_copied": 0,
    "checkpoint_mutated": False,
    "representation_gate_passed": False,
    "renderer_probe_authorized": False,
    "editor_training_authorized": False,
}
for field, expected in closed.items():
    if safety.get(field) != expected:
        raise SystemExit(f"{tag} tangent safety field differs: {field}")
for field in (
    "representation_gate_passed",
    "renderer_probe_authorized",
    "editor_training_authorized",
):
    if done.get(field) is not False:
        raise SystemExit(f"{tag} tangent done gate is open: {field}")
artifact_digest = done.get("artifact_digest")
if (
    not isinstance(artifact_digest, str)
    or len(artifact_digest) != 64
    or any(character not in "0123456789abcdef" for character in artifact_digest)
):
    raise SystemExit(f"{tag} tangent artifact digest differs")
print(f"motive_r10b_bernini_triplet.{tag}.artifact_digest={artifact_digest}")
PY
}

export -f probe_bound_gpu
export -f require_closed_artifact

verify_bound_inputs "before"

mkdir "${output_root}"
artifacts_root="${output_root}/artifacts"
gpu_probes="${output_root}/gpu_probes"
retrieval_audit="${output_root}/retrieval_audit"
mkdir "${artifacts_root}" "${gpu_probes}"

export MOTIVE_TRIPLET_PYTHON="${python_bin}"
export MOTIVE_TRIPLET_TRACK_CACHE="${track_cache}"
export MOTIVE_TRIPLET_TRACK_CACHE_SHA256="${track_cache_sha256}"
export MOTIVE_TRIPLET_MODEL_PATH="${model_path}"
export MOTIVE_TRIPLET_MODEL_REVISION="${model_revision}"
export MOTIVE_TRIPLET_MODEL_TREE_SHA256="${model_tree_sha256}"
export MOTIVE_TRIPLET_BERNINI_REPO="${bernini_repo}"
export MOTIVE_TRIPLET_BERNINI_SOURCE_COMMIT="${bernini_source_commit}"
export MOTIVE_TRIPLET_BERNINI_SOURCE_BUNDLE_SHA256="${bernini_source_bundle_sha256}"
export MOTIVE_TRIPLET_SOURCE_TREE_SHA256="${source_tree_sha256}"
export MOTIVE_TRIPLET_ARTIFACTS_ROOT="${artifacts_root}"
export MOTIVE_TRIPLET_GPU_PROBES="${gpu_probes}"
export MOTIVE_TRIPLET_SCRATCH="${job_scratch}"
export MOTIVE_TRIPLET_CANONICAL_MANIFEST="${canonical_manifest}"
export MOTIVE_TRIPLET_ORIGINAL_MANIFEST="${original_manifest}"
export MOTIVE_TRIPLET_CROSS_FAMILY_MANIFEST="${cross_family_manifest}"
export MOTIVE_TRIPLET_SCHEDULER_INDEX="${scheduler_index}"
export MOTIVE_TRIPLET_NOISE_MODE="${noise_mode}"
export MOTIVE_TRIPLET_WIDTH="${width}"
export MOTIVE_TRIPLET_HEIGHT="${height}"
export MOTIVE_TRIPLET_PROJECTION_DIM="${projection_dim}"
export MOTIVE_TRIPLET_PROJECTION_SEED_1="${projection_seed_1}"
export MOTIVE_TRIPLET_PROJECTION_SEED_2="${projection_seed_2}"

cd "${code_root}"

# One prompt intervention per real rank/GPU.  All three invocations use the
# same frozen scheduler/noise/resize/projection contract and differ only in
# their already-committed prompt manifest.
srun \
  --kill-on-bad-exit=1 \
  --ntasks=3 \
  --ntasks-per-node=3 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  bash -c '
    set -Eeuo pipefail
    if [[ ! "${SLURM_PROCID:-}" =~ ^[0-2]$ ]] \
      || [[ "${SLURM_LOCALID:-}" != "${SLURM_PROCID}" ]] \
      || [[ "${SLURM_NTASKS:-}" != "3" ]]; then
      echo "invalid Bernini triplet Slurm rank topology" >&2
      exit 3
    fi
    case "${SLURM_PROCID}" in
      0)
        tag="canonical"
        manifest="${MOTIVE_TRIPLET_CANONICAL_MANIFEST}"
        ;;
      1)
        tag="original"
        manifest="${MOTIVE_TRIPLET_ORIGINAL_MANIFEST}"
        ;;
      2)
        tag="cross_family"
        manifest="${MOTIVE_TRIPLET_CROSS_FAMILY_MANIFEST}"
        ;;
      *)
        echo "unreachable Bernini triplet rank" >&2
        exit 3
        ;;
    esac
    artifact_dir="${MOTIVE_TRIPLET_ARTIFACTS_ROOT}/${tag}"
    if [[ -e "${artifact_dir}" || -L "${artifact_dir}" ]]; then
      echo "refusing reused tangent artifact: ${artifact_dir}" >&2
      exit 3
    fi
    rank_scratch="${MOTIVE_TRIPLET_SCRATCH}/${tag}"
    mkdir -p \
      "${rank_scratch}/hf" \
      "${rank_scratch}/miopen-user" \
      "${rank_scratch}/miopen-custom"
    export HF_HOME="${rank_scratch}/hf"
    export MIOPEN_USER_DB_PATH="${rank_scratch}/miopen-user"
    export MIOPEN_CUSTOM_CACHE_DIR="${rank_scratch}/miopen-custom"
    probe_bound_gpu "${MOTIVE_TRIPLET_GPU_PROBES}/${tag}.json" "${tag}"
    "${MOTIVE_TRIPLET_PYTHON}" -m motive.r10b_bernini_tangent_extract \
      --model-path "${MOTIVE_TRIPLET_MODEL_PATH}" \
      --model-revision "${MOTIVE_TRIPLET_MODEL_REVISION}" \
      --bernini-repo "${MOTIVE_TRIPLET_BERNINI_REPO}" \
      --bernini-source-commit "${MOTIVE_TRIPLET_BERNINI_SOURCE_COMMIT}" \
      --manifest "${manifest}" \
      --track-cache "${MOTIVE_TRIPLET_TRACK_CACHE}" \
      --output-dir "${artifact_dir}" \
      --artifact-kind "controlled_retrieval_pilot" \
      --resize-mode "aspect_preserving_center_crop" \
      --width "${MOTIVE_TRIPLET_WIDTH}" \
      --height "${MOTIVE_TRIPLET_HEIGHT}" \
      --num-frames 17 \
      --projection-dim "${MOTIVE_TRIPLET_PROJECTION_DIM}" \
      --projection-seeds \
        "${MOTIVE_TRIPLET_PROJECTION_SEED_1}" \
        "${MOTIVE_TRIPLET_PROJECTION_SEED_2}" \
      --diffusion-noise-seed 260108853 \
      --noise-mode "${MOTIVE_TRIPLET_NOISE_MODE}" \
      --torch-seed 260108854 \
      --scheduler-steps 50 \
      --scheduler-index "${MOTIVE_TRIPLET_SCHEDULER_INDEX}" \
      --source-tree-sha256 "${MOTIVE_TRIPLET_SOURCE_TREE_SHA256}"
    "${MOTIVE_TRIPLET_PYTHON}" -m motive.r10b_bernini_tangent_extract \
      --output-dir "${artifact_dir}" \
      --validate-only
    require_closed_artifact "${tag}" "${artifact_dir}" "${manifest}"
  '

for tag in canonical original cross_family; do
  if [[ -L "${artifacts_root}/${tag}" \
    || ! -d "${artifacts_root}/${tag}" \
    || -L "${gpu_probes}/${tag}.json" \
    || ! -f "${gpu_probes}/${tag}.json" ]]; then
    echo "missing completed Bernini triplet rank output: ${tag}" >&2
    exit 4
  fi
done

verify_bound_inputs "after_tangent"

"${python_bin}" -m motive.r10b_bernini_retrieval_audit \
  --pilot-dir "${pilot_dir}" \
  --artifact "canonical=${artifacts_root}/canonical" \
  --artifact "original=${artifacts_root}/original" \
  --artifact "cross_family=${artifacts_root}/cross_family" \
  --output-dir "${retrieval_audit}"
"${python_bin}" -m motive.r10b_bernini_retrieval_audit \
  --output-dir "${retrieval_audit}" \
  --validate-only

verify_bound_inputs "after_retrieval"

"${python_bin}" - \
  "${output_root}" \
  "${source_snapshot}" \
  "${source_tree_sha256}" \
  "${qwen_packed_root}" \
  "${qwen_packed_commit_digest}" \
  "${track_cache}" \
  "${track_cache_sha256}" \
  "${model_path}" \
  "${model_revision}" \
  "${model_tree_sha256}" \
  "${bernini_repo}" \
  "${bernini_source_commit}" \
  "${bernini_source_bundle_sha256}" \
  "${scheduler_index}" \
  "${noise_mode}" \
  "${width}" \
  "${height}" \
  "${projection_dim}" \
  "${projection_seed_1}" \
  "${projection_seed_2}" \
  "${canonical_manifest}" \
  "${original_manifest}" \
  "${cross_family_manifest}" \
  "${python_bin}" <<'PY'
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile

(
    output_raw,
    snapshot_raw,
    source_tree_sha256,
    qwen_raw,
    expected_qwen_commit_digest,
    track_raw,
    track_sha256,
    model_raw,
    model_revision,
    model_tree_sha256,
    bernini_raw,
    source_commit,
    source_bundle_sha256,
    scheduler_index,
    noise_mode,
    width,
    height,
    projection_dim,
    projection_seed_1,
    projection_seed_2,
    canonical_raw,
    original_raw,
    cross_raw,
    python_raw,
) = sys.argv[1:]
output = Path(output_raw).resolve(strict=True)
snapshot = Path(snapshot_raw).resolve(strict=True)
qwen = Path(qwen_raw).resolve(strict=True)
track = Path(track_raw).resolve(strict=True)
model = Path(model_raw).resolve(strict=True)
bernini = Path(bernini_raw).resolve(strict=True)
python_bin = Path(python_raw).resolve(strict=True)
manifests = {
    "canonical": Path(canonical_raw).resolve(strict=True),
    "original": Path(original_raw).resolve(strict=True),
    "cross_family": Path(cross_raw).resolve(strict=True),
}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def canonical_digest(value):
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def tree_record(root, *, excluded=()):
    excluded_set = set(excluded)
    rows = []
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise SystemExit(f"published triplet contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise SystemExit(f"published triplet contains a non-file: {path}")
        relative = path.relative_to(root).as_posix()
        if relative in excluded_set:
            continue
        size = path.stat().st_size
        rows.append(
            {
                "relative_path": relative,
                "bytes": size,
                "sha256": sha256(path),
            }
        )
        total_bytes += size
    encoded = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    return {
        "path": str(root),
        "files": len(rows),
        "bytes": total_bytes,
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
    }

def strict_json(path):
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"expected one regular JSON file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"JSON payload is not one object: {path}")
    return value, raw

def atomic_write_new(path, payload):
    if os.path.lexists(path):
        raise SystemExit(f"refusing reused triplet commit file: {path}")
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.",
        dir=str(path.parent),
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()

if {path.name for path in output.iterdir()} != {
    "artifacts",
    "gpu_probes",
    "retrieval_audit",
}:
    raise SystemExit("Bernini triplet pre-commit topology differs")
artifacts_root = output / "artifacts"
probe_root = output / "gpu_probes"
retrieval_root = output / "retrieval_audit"
if {path.name for path in artifacts_root.iterdir()} != set(manifests):
    raise SystemExit("Bernini triplet artifact tag closure differs")
if {path.name for path in probe_root.iterdir()} != {
    f"{tag}.json" for tag in manifests
}:
    raise SystemExit("Bernini triplet GPU-probe closure differs")

visibility_names = {
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NTASKS",
    "SLURM_JOB_ID",
    "SLURMD_NODENAME",
    "HOSTNAME",
}
tags = ("canonical", "original", "cross_family")
gpu_evidence = []
for rank, tag in enumerate(tags):
    path = probe_root / f"{tag}.json"
    probe, _raw = strict_json(path)
    if (
        set(probe)
        != {
            "schema_version",
            "status",
            "tag",
            "rank",
            "local_rank",
            "slurm_ntasks",
            "device_count",
            "device_index",
            "device_name",
            "device_uuid",
            "tensor_probe",
            "visibility",
        }
        or probe.get("schema_version")
        != "motive-r10b-bernini-triplet-packed-gpu-probe-v1"
        or probe.get("status") != "passed"
        or probe.get("tag") != tag
        or probe.get("rank") != rank
        or probe.get("local_rank") != rank
        or probe.get("slurm_ntasks") != 3
        or probe.get("device_count") != 1
        or probe.get("device_index") != 0
        or "MI210" not in str(probe.get("device_name", "")).upper()
        or probe.get("tensor_probe")
        != {"device": "cuda:0", "sum_of_squares": 14.0, "passed": True}
        or set(probe.get("visibility", {})) != visibility_names
        or probe["visibility"].get("SLURM_PROCID") != str(rank)
        or probe["visibility"].get("SLURM_LOCALID") != str(rank)
        or probe["visibility"].get("SLURM_NTASKS") != "3"
    ):
        raise SystemExit(f"{tag} final GPU probe differs")
    gpu_evidence.append(
        {
            "tag": tag,
            "rank": rank,
            "relative_path": path.relative_to(output).as_posix(),
            "sha256": sha256(path),
            "device_name": probe["device_name"],
            "device_uuid": probe["device_uuid"],
            "visibility": probe["visibility"],
        }
    )

manifest_records = {}
artifact_records = {}
artifact_digests = {}
reference_contract = None
reference_parameter_manifest = None
measurement_video_decodes = 0
for tag in tags:
    manifest = manifests[tag]
    manifest_raw = manifest.read_bytes()
    if manifest.is_symlink() or not manifest.is_file() or not manifest_raw:
        raise SystemExit(f"{tag} manifest is not one non-empty regular file")
    if not manifest_raw.endswith(b"\n"):
        raise SystemExit(f"{tag} manifest lacks a final newline")
    manifest_records[tag] = {
        "path": str(manifest),
        "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "rows": len(manifest_raw.splitlines()),
    }
    artifact = artifacts_root / tag
    if artifact.is_symlink() or not artifact.is_dir():
        raise SystemExit(f"{tag} artifact is not one directory")
    if {path.name for path in artifact.iterdir()} != {
        "features.npz",
        "rows.jsonl",
        "summary.json",
        "done.json",
    }:
        raise SystemExit(f"{tag} artifact closure differs at final commit")
    summary, _summary_raw = strict_json(artifact / "summary.json")
    done, _done_raw = strict_json(artifact / "done.json")
    measurement = summary.get("measurement", {})
    contract = measurement.get("run_contract")
    data = summary.get("data", {})
    safety = summary.get("safety", {})
    if (
        summary.get("status") != "complete"
        or summary.get("artifact_kind") != "controlled_retrieval_pilot"
        or done.get("status") != "complete"
        or done.get("artifact_kind") != "controlled_retrieval_pilot"
        or data.get("manifest") != str(manifest)
        or data.get("manifest_sha256") != manifest_records[tag]["sha256"]
        or data.get("track_cache") != str(track)
        or data.get("track_cache_sha256") != track_sha256
        or summary.get("source_tree_sha256") != source_tree_sha256
        or summary.get("model", {}).get("model_path") != str(model)
        or summary.get("model", {}).get("huggingface_revision") != model_revision
        or summary.get("model", {})
        .get("checkpoint_manifest", {})
        .get("tree_sha256") != model_tree_sha256
        or summary.get("official_bernini_source", {}).get("commit")
        != source_commit
        or summary.get("official_bernini_source", {}).get("bundle_sha256")
        != source_bundle_sha256
        or measurement.get("projection_dimension_per_role")
        != int(projection_dim)
        or measurement.get("projection_seeds")
        != [int(projection_seed_1), int(projection_seed_2)]
        or data.get("rows") != 20
        or data.get("videos_read") != 40
        or safety.get("videos_decoded_for_measurement") != 40
    ):
        raise SystemExit(f"{tag} final tangent provenance differs")
    measurement_video_decodes += safety["videos_decoded_for_measurement"]
    scheduler = contract.get("scheduler", {}) if isinstance(contract, dict) else {}
    noise = contract.get("noise", {}) if isinstance(contract, dict) else {}
    resize = contract.get("resize", {}) if isinstance(contract, dict) else {}
    if (
        not isinstance(contract, dict)
        or contract.get("artifact_kind") != "controlled_retrieval_pilot"
        or scheduler.get("steps") != 50
        or scheduler.get("index") != int(scheduler_index)
        or noise.get("mode") != noise_mode
        or noise.get("seed") != 260108853
        or resize.get("mode") != "aspect_preserving_center_crop"
        or resize.get("width") != int(width)
        or resize.get("height") != int(height)
        or resize.get("num_frames") != 17
    ):
        raise SystemExit(f"{tag} final tangent contract differs")
    if reference_contract is None:
        reference_contract = contract
        reference_parameter_manifest = summary.get("parameter_manifest_sha256")
    elif (
        contract != reference_contract
        or summary.get("parameter_manifest_sha256")
        != reference_parameter_manifest
    ):
        raise SystemExit("triplet tangent contracts/parameter spaces differ")
    expected_safety = {
        "optimizer_created": False,
        "optimizer_steps": 0,
        "scheduler_steps_executed": 0,
        "renderer_calls": 0,
        "videos_rendered": 0,
        "videos_copied": 0,
        "checkpoint_mutated": False,
        "representation_gate_passed": False,
        "renderer_probe_authorized": False,
        "editor_training_authorized": False,
    }
    for field, expected in expected_safety.items():
        if safety.get(field) != expected:
            raise SystemExit(f"{tag} final tangent safety differs: {field}")
    for field in (
        "representation_gate_passed",
        "renderer_probe_authorized",
        "editor_training_authorized",
    ):
        if done.get(field) is not False:
            raise SystemExit(f"{tag} final tangent done gate is open: {field}")
    artifact_digest = done.get("artifact_digest")
    if (
        not isinstance(artifact_digest, str)
        or len(artifact_digest) != 64
        or any(character not in "0123456789abcdef" for character in artifact_digest)
    ):
        raise SystemExit(f"{tag} artifact digest is invalid")
    artifact_digests[tag] = artifact_digest
    artifact_records[tag] = tree_record(artifact)

if sha256(track) != track_sha256:
    raise SystemExit("track cache changed before triplet commit")

if {path.name for path in retrieval_root.iterdir()} != {
    "retrieval_audit.json",
    "done.json",
}:
    raise SystemExit("retrieval-audit closure differs")
retrieval, _retrieval_raw = strict_json(
    retrieval_root / "retrieval_audit.json"
)
retrieval_done, _retrieval_done_raw = strict_json(retrieval_root / "done.json")
retrieval_authorization = {
    "human_label": False,
    "formal_evidence": False,
    "representation_promoted": False,
    "renderer_probe_authorized": False,
    "generation_authorized": False,
    "training_authorized": False,
}
for field in (
    "formal_evidence",
    "representation_gate_passed",
    "renderer_probe_authorized",
    "generation_authorized",
    "editor_training_authorized",
    "training_authorized",
):
    if retrieval.get(field) is not False:
        raise SystemExit(f"retrieval audit gate is open: {field}")
if (
    retrieval.get("status") != "complete"
    or retrieval.get("authorization") != retrieval_authorization
    or retrieval.get("media_io")
    != {
        "video_files_read": 0,
        "video_files_copied": 0,
        "video_files_rendered": 0,
        "feature_arrays_read_only": True,
    }
):
    raise SystemExit("retrieval audit closed-scope evidence differs")
for field in (
    "representation_gate_passed",
    "renderer_probe_authorized",
    "editor_training_authorized",
):
    if retrieval_done.get(field) is not False:
        raise SystemExit(f"retrieval done gate is open: {field}")
development = retrieval.get("decision", {}).get(
    "development_signal_requires_sigma_noise_dimension_holdout"
)
if not isinstance(development, bool):
    raise SystemExit("retrieval development signal is not boolean")
appearance_control = retrieval.get("leakage_readouts", {}).get("appearance")
if (
    not isinstance(appearance_control, dict)
    or appearance_control.get("available") is not False
    or appearance_control.get("sufficient_for_gate") is not False
    or appearance_control.get("passed") is not False
):
    raise SystemExit("independent appearance control is not explicitly missing")
retrieval_artifact_digest = retrieval_done.get("artifact_digest")
if (
    not isinstance(retrieval_artifact_digest, str)
    or len(retrieval_artifact_digest) != 64
):
    raise SystemExit("retrieval artifact digest is invalid")

qwen_summary, qwen_summary_raw = strict_json(
    qwen / "pipeline_summary.json"
)
qwen_done, _qwen_done_raw = strict_json(qwen / "pipeline_done.json")
qwen_summary_record = {
    "bytes": len(qwen_summary_raw),
    "sha256": hashlib.sha256(qwen_summary_raw).hexdigest(),
}
qwen_tree = tree_record(qwen, excluded={"pipeline_done.json"})
qwen_binding = {
    "pipeline_summary": qwen_summary_record,
    "output_tree_without_done": qwen_tree,
    "controlled_pilot_commit_digest": qwen_summary.get(
        "controlled_pilot_commit_digest"
    ),
}
if (
    qwen_summary.get("balanced_pilot_ready") is not True
    or qwen_summary.get("prompt_variants_materialized") is not True
    or qwen_done.get("balanced_pilot_ready") is not True
    or qwen_done.get("pipeline_summary") != qwen_summary_record
    or qwen_done.get("output_tree_without_done") != qwen_tree
    or qwen_done.get("commit_digest") != canonical_digest(qwen_binding)
    or qwen_done.get("commit_digest") != expected_qwen_commit_digest
):
    raise SystemExit("Qwen packed commit changed before triplet commit")
controlled_pilot_commit_digest = qwen_summary.get(
    "controlled_pilot_commit_digest"
)
if retrieval.get("pilot", {}).get(
    "commit_digest"
) != controlled_pilot_commit_digest:
    raise SystemExit("retrieval audit pilot binding differs")
feature_bindings = retrieval.get("feature_artifacts")
if (
    not isinstance(feature_bindings, list)
    or [item.get("tag") for item in feature_bindings] != list(tags)
):
    raise SystemExit("retrieval audit feature-tag closure differs")
for item in feature_bindings:
    tag = item["tag"]
    if (
        item.get("path") != str((artifacts_root / tag).resolve())
        or item.get("artifact_digest") != artifact_digests[tag]
    ):
        raise SystemExit(f"retrieval audit feature binding differs: {tag}")
variant_files = {
    name: sha256(qwen / "prompt_variants" / name)
    for name in (
        "original.jsonl",
        "cross_family_shuffle.jsonl",
        "summary.json",
        "done.json",
    )
}
prompt_variants_commit_digest = canonical_digest(variant_files)

outputs = {
    "artifacts": tree_record(artifacts_root),
    "gpu_probes": tree_record(probe_root),
    "retrieval_audit": tree_record(retrieval_root),
}
authorization = {
    "formal_evidence": False,
    "representation_promoted": False,
    "renderer_probe_authorized": False,
    "generation_authorized": False,
    "training_authorized": False,
}
summary = {
    "schema_version": "motive-r10b-bernini-triplet-packed-pipeline-v1",
    "status": "ready_for_commit",
    "scope": (
        "three frozen prompt-intervention tangents plus read-only retrieval "
        "audit; no representation promotion, rendering, generation, or training"
    ),
    "inputs": {
        "source_snapshot": {
            "path": str(snapshot),
            "tree_sha256": source_tree_sha256,
        },
        "qwen_packed": {
            "path": str(qwen),
            "pipeline_done_sha256": sha256(qwen / "pipeline_done.json"),
            "commit_digest": qwen_done["commit_digest"],
            "controlled_pilot_commit_digest": qwen_summary[
                "controlled_pilot_commit_digest"
            ],
            "source_snapshot": qwen_summary.get("inputs", {}).get(
                "source_snapshot"
            ),
            "balanced_pilot_ready": True,
            "prompt_variants_materialized": True,
            "prompt_variants_commit_digest": prompt_variants_commit_digest,
        },
        "manifests": manifest_records,
        "track_cache": {
            "path": str(track),
            "sha256": track_sha256,
        },
        "model": {
            "path": str(model),
            "revision": model_revision,
            "tree_sha256": model_tree_sha256,
        },
        "bernini_source": {
            "path": str(bernini),
            "commit": source_commit,
            "bundle_sha256": source_bundle_sha256,
        },
        "python": str(python_bin),
    },
    "execution": {
        "nodes": 1,
        "real_gpu_tasks": 3,
        "mi210_gpus": 3,
        "rank_to_prompt_variant": {
            "0": "canonical",
            "1": "original",
            "2": "cross_family",
        },
        "gpu_probe_evidence": gpu_evidence,
        "frozen_run_contract": reference_contract,
        "projection_dimension_per_role": int(projection_dim),
        "projection_seeds": [
            int(projection_seed_1),
            int(projection_seed_2),
        ],
    },
    "outputs": outputs,
    "tangent_artifacts": {
        tag: {
            "artifact_digest": artifact_digests[tag],
            "tree": artifact_records[tag],
        }
        for tag in tags
    },
    "retrieval": {
        "artifact_digest": retrieval_artifact_digest,
        "development_signal_requires_sigma_noise_dimension_holdout": development,
        "appearance_control_available": False,
        "appearance_control_sufficient_for_gate": False,
        "appearance_control_status": (
            "missing_independent_dino_appearance_control_artifact"
        ),
        "development_signal_promotion_eligible": False,
        "representation_promoted": False,
    },
    "videos_decoded_for_measurement": measurement_video_decodes,
    "videos_copied": 0,
    "videos_rendered": 0,
    "renderer_calls": 0,
    "optimizer_steps": 0,
    "formal_evidence": False,
    "representation_gate_passed": False,
    "renderer_probe_authorized": False,
    "generation_authorized": False,
    "editor_training_authorized": False,
    "training_authorized": False,
    "downstream_submitted": False,
    "authorization": authorization,
}
summary_payload = (
    json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
).encode("utf-8")
summary_path = output / "pipeline_summary.json"
atomic_write_new(summary_path, summary_payload)
summary_record = {
    "bytes": len(summary_payload),
    "sha256": hashlib.sha256(summary_payload).hexdigest(),
}
if {path.name for path in output.iterdir()} != {
    "artifacts",
    "gpu_probes",
    "retrieval_audit",
    "pipeline_summary.json",
}:
    raise SystemExit("Bernini triplet summary topology differs")
output_tree = tree_record(output)
commit_binding = {
    "pipeline_summary": summary_record,
    "output_tree_without_done": output_tree,
    "qwen_packed_commit_digest": qwen_done["commit_digest"],
    "tangent_artifact_digests": artifact_digests,
    "retrieval_artifact_digest": retrieval_artifact_digest,
}
done = {
    "schema_version": "motive-r10b-bernini-triplet-packed-pipeline-done-v1",
    "status": "complete",
    "pipeline_summary": summary_record,
    "output_tree_without_done": output_tree,
    "commit_digest": canonical_digest(commit_binding),
    "qwen_packed_commit_digest": qwen_done["commit_digest"],
    "tangent_artifact_digests": artifact_digests,
    "retrieval_artifact_digest": retrieval_artifact_digest,
    "real_gpu_tasks_verified": 3,
    "balanced_pilot_ready": True,
    "development_signal_requires_sigma_noise_dimension_holdout": development,
    "appearance_control_available": False,
    "development_signal_promotion_eligible": False,
    "formal_evidence": False,
    "representation_gate_passed": False,
    "renderer_probe_authorized": False,
    "generation_authorized": False,
    "editor_training_authorized": False,
    "training_authorized": False,
    "downstream_submitted": False,
    "authorization": authorization,
}
done_payload = (
    json.dumps(done, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
).encode("utf-8")
done_path = output / "pipeline_done.json"
atomic_write_new(done_path, done_payload)
if {path.name for path in output.iterdir()} != {
    "artifacts",
    "gpu_probes",
    "retrieval_audit",
    "pipeline_summary.json",
    "pipeline_done.json",
}:
    raise SystemExit("Bernini triplet final topology differs")
if (
    sha256(summary_path) != done["pipeline_summary"]["sha256"]
    or tree_record(output, excluded={"pipeline_done.json"})
    != done["output_tree_without_done"]
    or done["commit_digest"] != canonical_digest(commit_binding)
):
    raise SystemExit("Bernini triplet final commit closure differs")
print(
    "motive_r10b_bernini_triplet.pipeline_summary_sha256="
    + summary_record["sha256"]
)
print(
    "motive_r10b_bernini_triplet.pipeline_done_sha256="
    + hashlib.sha256(done_payload).hexdigest()
)
print(
    "motive_r10b_bernini_triplet.status="
    + (
        "development_signal_requires_holdouts"
        if development
        else "no_development_signal"
    )
)
PY

record_binding "status" "complete_closed_measurement_only"
