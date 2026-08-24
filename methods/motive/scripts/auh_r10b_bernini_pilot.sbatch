#!/usr/bin/env bash
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=03:00:00
#SBATCH --job-name=motive-r10b-bernini-pilot

set -euo pipefail

source_snapshot="${MOTIVE_R10B_SOURCE_SNAPSHOT:?set MOTIVE_R10B_SOURCE_SNAPSHOT}"
source_tree_sha256="${MOTIVE_R10B_SOURCE_TREE_SHA256:?set MOTIVE_R10B_SOURCE_TREE_SHA256}"
manifest="${MOTIVE_R10B_MANIFEST:?set MOTIVE_R10B_MANIFEST}"
manifest_sha256="${MOTIVE_R10B_MANIFEST_SHA256:?set MOTIVE_R10B_MANIFEST_SHA256}"
track_cache="${MOTIVE_R10B_TRACK_CACHE:?set MOTIVE_R10B_TRACK_CACHE}"
track_cache_sha256="${MOTIVE_R10B_TRACK_CACHE_SHA256:?set MOTIVE_R10B_TRACK_CACHE_SHA256}"
model_path="${MOTIVE_R10B_MODEL_PATH:?set MOTIVE_R10B_MODEL_PATH}"
model_revision="${MOTIVE_R10B_MODEL_REVISION:?set MOTIVE_R10B_MODEL_REVISION}"
model_tree_sha256="${MOTIVE_R10B_MODEL_TREE_SHA256:?set MOTIVE_R10B_MODEL_TREE_SHA256}"
bernini_repo="${MOTIVE_R10B_BERNINI_REPO:?set MOTIVE_R10B_BERNINI_REPO}"
bernini_source_commit="${MOTIVE_R10B_BERNINI_SOURCE_COMMIT:?set MOTIVE_R10B_BERNINI_SOURCE_COMMIT}"
bernini_source_bundle_sha256="${MOTIVE_R10B_BERNINI_SOURCE_BUNDLE_SHA256:?set MOTIVE_R10B_BERNINI_SOURCE_BUNDLE_SHA256}"
output_dir="${MOTIVE_R10B_OUTPUT_DIR:?set MOTIVE_R10B_OUTPUT_DIR}"
python_bin="${MOTIVE_R10B_PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

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

require_absolute_path() {
  local label="$1"
  local path="$2"
  if [[ "${path}" != /* ]]; then
    echo "${label} must be an absolute path: ${path}" >&2
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
    echo "${label} must be a positive integer: ${value}" >&2
    exit 2
  fi
}

for binding in \
  "source_snapshot:${source_snapshot}" \
  "manifest:${manifest}" \
  "track_cache:${track_cache}" \
  "model_path:${model_path}" \
  "bernini_repo:${bernini_repo}" \
  "output_dir:${output_dir}" \
  "python_bin:${python_bin}"; do
  require_absolute_path "${binding%%:*}" "${binding#*:}"
done

if [[ ! -d "${source_snapshot}" ]]; then
  echo "source snapshot is not a directory: ${source_snapshot}" >&2
  exit 2
fi
if [[ -L "${manifest}" || ! -f "${manifest}" ]]; then
  echo "manifest must be one regular non-symlink file: ${manifest}" >&2
  exit 2
fi
if [[ -L "${track_cache}" || ! -f "${track_cache}" ]]; then
  echo "track cache must be one regular non-symlink file: ${track_cache}" >&2
  exit 2
fi
if [[ ! -d "${model_path}" ]]; then
  echo "model path is not a directory: ${model_path}" >&2
  exit 2
fi
if [[ ! -d "${bernini_repo}" ]]; then
  echo "Bernini source path is not a directory: ${bernini_repo}" >&2
  exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable is unavailable: ${python_bin}" >&2
  exit 2
fi
if [[ -e "${output_dir}" || -L "${output_dir}" ]]; then
  echo "refusing to reuse R10B pilot output: ${output_dir}" >&2
  exit 2
fi

for binding in \
  "source_tree_sha256:${source_tree_sha256}" \
  "manifest_sha256:${manifest_sha256}" \
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
  echo "unsupported pilot noise mode: ${noise_mode}" >&2
  exit 2
fi

max_samples_args=()
max_samples_record="<unset>"
if [[ "${MOTIVE_R10B_MAX_SAMPLES+x}" == "x" ]]; then
  require_positive_integer "MOTIVE_R10B_MAX_SAMPLES" "${MOTIVE_R10B_MAX_SAMPLES}"
  max_samples_args=(--max-samples "${MOTIVE_R10B_MAX_SAMPLES}")
  max_samples_record="${MOTIVE_R10B_MAX_SAMPLES}"
fi

verify_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "${path}" | cut -d ' ' -f 1)"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA-256 mismatch: ${path}" >&2
    exit 3
  fi
}

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHONPATH="${source_snapshot}:${source_snapshot}/methods/motive"

task_scratch="${SLURM_TMPDIR:-/tmp}/motive-r10b-bernini-pilot-${SLURM_JOB_ID:-manual}"
mkdir -p "${task_scratch}/miopen-user" "${task_scratch}/miopen-custom"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"

record_binding() {
  printf 'motive_r10b_pilot.%s=%s\n' "$1" "$2"
}

record_binding "source_snapshot" "${source_snapshot}"
record_binding "source_tree_sha256" "${source_tree_sha256}"
record_binding "manifest" "${manifest}"
record_binding "manifest_sha256" "${manifest_sha256}"
record_binding "track_cache" "${track_cache}"
record_binding "track_cache_sha256" "${track_cache_sha256}"
record_binding "model_path" "${model_path}"
record_binding "model_revision" "${model_revision}"
record_binding "model_tree_sha256" "${model_tree_sha256}"
record_binding "bernini_repo" "${bernini_repo}"
record_binding "bernini_source_commit" "${bernini_source_commit}"
record_binding "bernini_source_bundle_sha256" "${bernini_source_bundle_sha256}"
record_binding "output_dir" "${output_dir}"
record_binding "python_bin" "${python_bin}"
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
record_binding "max_samples" "${max_samples_record}"

verify_bound_inputs() {
  local stage="$1"
  record_binding "input_verification_stage" "${stage}"
  "${python_bin}" \
    "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
    verify \
    --snapshot "${source_snapshot}" \
    --expected-tree-sha256 "${source_tree_sha256}"
  verify_sha256 "${manifest}" "${manifest_sha256}"
  verify_sha256 "${track_cache}" "${track_cache_sha256}"
  "${python_bin}" - \
    "${model_path}" \
    "${model_revision}" \
    "${model_tree_sha256}" \
    "${bernini_repo}" \
    "${bernini_source_commit}" \
    "${bernini_source_bundle_sha256}" <<'PY'
from pathlib import Path
import sys

from motive.r10b_bernini_tangent_extract import (
    EXPECTED_REPO_REVISION,
    EXPECTED_SOURCE_COMMIT,
    _bernini_source_manifest,
    _checkpoint_manifest,
    _checkpoint_revision_metadata,
)

(
    model_path_raw,
    model_revision,
    expected_model_tree,
    bernini_repo_raw,
    source_commit,
    expected_source_bundle,
) = sys.argv[1:]
if model_revision != EXPECTED_REPO_REVISION:
    raise SystemExit("model revision differs from the frozen backend")
if source_commit != EXPECTED_SOURCE_COMMIT:
    raise SystemExit("Bernini source commit differs from the frozen backend")

model_path = Path(model_path_raw)
checkpoint = _checkpoint_manifest(model_path)
revision = _checkpoint_revision_metadata(model_path)
if checkpoint["tree_sha256"] != expected_model_tree:
    raise SystemExit("model tree SHA-256 differs")
if revision["revision"] != model_revision:
    raise SystemExit("model metadata revision differs")

vendor = _bernini_source_manifest(Path(bernini_repo_raw))
if vendor["bundle_sha256"] != expected_source_bundle:
    raise SystemExit("Bernini source bundle SHA-256 differs")

print(f"motive_r10b_pilot.verified_model_tree_sha256={checkpoint['tree_sha256']}")
print(f"motive_r10b_pilot.verified_model_revision={revision['revision']}")
print(
    "motive_r10b_pilot.verified_bernini_source_bundle_sha256="
    f"{vendor['bundle_sha256']}"
)
PY
}

verify_bound_inputs "before"

extract_args=(
  --model-path "${model_path}"
  --model-revision "${model_revision}"
  --bernini-repo "${bernini_repo}"
  --bernini-source-commit "${bernini_source_commit}"
  --manifest "${manifest}"
  --track-cache "${track_cache}"
  --output-dir "${output_dir}"
  --artifact-kind "${artifact_kind}"
  --resize-mode "${resize_mode}"
  --width "${width}"
  --height "${height}"
  --num-frames "${num_frames}"
  --projection-dim "${projection_dim}"
  --projection-seeds "${projection_seed_1}" "${projection_seed_2}"
  --diffusion-noise-seed "${diffusion_noise_seed}"
  --noise-mode "${noise_mode}"
  --torch-seed "${torch_seed}"
  --scheduler-steps "${scheduler_steps}"
  --scheduler-index "${scheduler_index}"
  --source-tree-sha256 "${source_tree_sha256}"
)
extract_args+=("${max_samples_args[@]}")

"${python_bin}" -m motive.r10b_bernini_tangent_extract "${extract_args[@]}"

if [[ ! -d "${output_dir}" || -L "${output_dir}" ]]; then
  echo "extractor did not publish one regular output directory" >&2
  exit 4
fi

"${python_bin}" -m motive.r10b_bernini_tangent_extract \
  --output-dir "${output_dir}" \
  --validate-only

verify_bound_inputs "after"

"${python_bin}" - \
  "${output_dir}" \
  "${scheduler_index}" \
  "${noise_mode}" \
  "${width}" \
  "${height}" \
  "${projection_dim}" \
  "${projection_seed_1}" \
  "${projection_seed_2}" <<'PY'
from pathlib import Path
import json
import re
import sys

(
    output_raw,
    scheduler_index,
    noise_mode,
    width,
    height,
    projection_dim,
    projection_seed_1,
    projection_seed_2,
) = sys.argv[1:]
output = Path(output_raw)
summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
done = json.loads((output / "done.json").read_text(encoding="utf-8"))

measurement = summary.get("measurement", {})
contract = measurement.get("run_contract", {})
scheduler = contract.get("scheduler", {})
noise = contract.get("noise", {})
resize = contract.get("resize", {})
runtime = summary.get("runtime", {})
safety = summary.get("safety", {})

expected = {
    "artifact_kind": "controlled_retrieval_pilot",
    "scheduler_index": int(scheduler_index),
    "noise_mode": noise_mode,
    "resize_mode": "aspect_preserving_center_crop",
    "width": int(width),
    "height": int(height),
    "num_frames": 17,
    "projection_dim": int(projection_dim),
    "projection_seeds": [int(projection_seed_1), int(projection_seed_2)],
}
observed = {
    "artifact_kind": contract.get("artifact_kind"),
    "scheduler_index": scheduler.get("index"),
    "noise_mode": noise.get("mode"),
    "resize_mode": resize.get("mode"),
    "width": runtime.get("width"),
    "height": runtime.get("height"),
    "num_frames": runtime.get("num_frames"),
    "projection_dim": measurement.get("projection_dimension_per_role"),
    "projection_seeds": measurement.get("projection_seeds"),
}
if observed != expected:
    raise SystemExit(
        f"published pilot contract differs: expected={expected}, observed={observed}"
    )

closed = {
    "videos_rendered": 0,
    "videos_copied": 0,
    "optimizer_created": False,
    "optimizer_steps": 0,
    "renderer_calls": 0,
    "representation_gate_passed": False,
    "renderer_probe_authorized": False,
    "editor_training_authorized": False,
}
for key, expected_value in closed.items():
    if safety.get(key) != expected_value:
        raise SystemExit(
            f"published pilot safety gate is open: {key}={safety.get(key)!r}"
        )
for key in (
    "representation_gate_passed",
    "renderer_probe_authorized",
    "editor_training_authorized",
):
    if done.get(key) is not False:
        raise SystemExit(f"published done gate is open: {key}")

artifact_digest = done.get("artifact_digest")
if not isinstance(artifact_digest, str) or re.fullmatch(
    r"[0-9a-f]{64}", artifact_digest
) is None:
    raise SystemExit("published artifact digest is invalid")
print(f"motive_r10b_pilot.output_artifact_digest={artifact_digest}")
print("motive_r10b_pilot.status=validated_closed_measurement_only")
PY
