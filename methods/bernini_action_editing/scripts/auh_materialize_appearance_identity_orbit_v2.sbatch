#!/usr/bin/env bash
#SBATCH --job-name=bern-cio-vae
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=04:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 027

fail() {
  echo "[bernini-cio-vae] ERROR: $*" >&2
  exit 2
}

spec="${BERNINI_CIO_SPEC:?set BERNINI_CIO_SPEC}"
spec_sha256="${BERNINI_CIO_SPEC_SHA256:?set BERNINI_CIO_SPEC_SHA256}"
output="${BERNINI_CIO_DATASET_OUTPUT:?set a fresh BERNINI_CIO_DATASET_OUTPUT}"
source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:?set BERNINI_ACTION_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:?set BERNINI_ACTION_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_ACTION_SOURCE_REVISION:?set BERNINI_ACTION_SOURCE_REVISION}"
source_repository="${BERNINI_ACTION_SOURCE_REPOSITORY:?set BERNINI_ACTION_SOURCE_REPOSITORY}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
python_bin="${BERNINI_CIO_PYTHON_BIN:?set BERNINI_CIO_PYTHON_BIN}"

expected_checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"

for name in spec output source_archive source_repository checkpoint checkpoint_manifest python_bin; do
  value="${!name}"
  [[ "${value}" == /* ]] || fail "${name} must be absolute"
done
for digest in spec_sha256 source_archive_sha256; do
  [[ "${!digest}" =~ ^[0-9a-f]{64}$ ]] || fail "${digest} must be SHA-256"
done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be SHA-1"
[[ -f "${spec}" && ! -L "${spec}" ]] || fail "spec must be a plain file"
[[ "$(sha256sum "${spec}" | awk '{print $1}')" == "${spec_sha256}" ]] || fail "spec hash differs"
[[ "${output}" != / && "${output##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$ ]] || fail "unsafe output"
[[ "$(realpath -m -- "${output}")" == "${output}" ]] || fail "output must be canonical"
[[ ! -e "${output}" && ! -L "${output}" ]] || fail "output is create-only"
mkdir -p -- "$(dirname -- "${output}")"
[[ ! -L "$(dirname -- "${output}")" ]] || fail "output parent is a symlink"

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive is invalid"
[[ -z "$(find "${source_archive}" -maxdepth 0 -perm /222 -print -quit)" ]] || fail "source archive must be read-only"
[[ -d "${source_repository}/.git" && ! -L "${source_repository}" ]] || fail "source repository is invalid"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint is invalid"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest is invalid"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python is invalid"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${expected_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest hash differs"
[[ "$(git -C "${source_repository}" cat-file -t "${source_revision}")" == commit ]] || fail "source revision is unavailable"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "invalid Slurm job id"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-cio-vae-${SLURM_JOB_ID}.XXXXXX")"
mkdir -- \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton" \
  "${task_scratch}/tmp"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
export TMPDIR="${task_scratch}/tmp"

cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/bernini-cio-vae-${SLURM_JOB_ID}."*) ;;
    *) echo "[bernini-cio-vae] refusing unsafe scratch cleanup" >&2; exit 2 ;;
  esac
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || status=2
    rm -rf -- "${task_scratch}" || status=2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

archive_copy="${task_scratch}/method.tar"
git_archive="${task_scratch}/git-method.tar"
cp -- "${source_archive}" "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged archive differs"
git -C "${source_repository}" archive --format=tar --output="${git_archive}" "${source_revision}" methods/bernini_action_editing
[[ "$(sha256sum "${git_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "archive differs from commit subtree"
mkdir -- "${task_scratch}/source"
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
for required in \
  appearance_counterfactual_identity_orbit.py \
  source_self_runtime.py \
  scripts/auh_materialize_appearance_identity_orbit_v2.sbatch \
  tools/materialize_appearance_counterfactual_identity_orbit.py \
  tools/materialize_ramp_motion_analogy_vae.py \
  tools/materialize_vae.py; do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive is missing ${required}"
done
find "${method_root}" -type f -exec chmod a-w {} +

PYTHONPATH="${method_root}" "${python_bin}" -B \
  "${method_root}/tools/materialize_appearance_counterfactual_identity_orbit.py" \
  --spec "${spec}" \
  --expected-spec-sha256 "${spec_sha256}" \
  --checkpoint "${checkpoint}" \
  --checkpoint-content-manifest "${checkpoint_manifest}" \
  --output "${output}" \
  --device cuda:0

[[ -d "${output}" && ! -L "${output}" ]] || fail "dataset output was not published"
files="$(find "${output}" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)"
[[ "${files}" == $'dataset.parquet\nreceipt.json' ]] || fail "dataset artifact closure differs"
chmod a-w -- "${output}/dataset.parquet" "${output}/receipt.json"
"${python_bin}" -B - "${output}" "${spec_sha256}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import pyarrow.parquet as pq

root = Path(sys.argv[1]).resolve(strict=True)
spec_sha = sys.argv[2]
receipt_path = root / "receipt.json"
receipt = json.loads(receipt_path.read_text(encoding="ascii"))
declared = receipt.pop("receipt_digest", None)
payload = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
if hashlib.sha256(payload).hexdigest() != declared:
    raise SystemExit("dataset receipt digest differs")
if (
    receipt.get("schema_version") != "bernini-appearance-counterfactual-identity-orbit-dataset-receipt-v3"
    or receipt.get("complete") is not True
    or receipt.get("scientific_use_authorized") is not True
    or receipt.get("direct_action_edit_claim_authorized") is not False
    or receipt.get("spec", {}).get("file_sha256") != spec_sha
):
    raise SystemExit("dataset receipt contract differs")
encoding = receipt.get("encoding_contract", {})
if (
    encoding.get("reference_count") != 4
    or encoding.get("reference_rgb_indices") != [0, 27, 53, 80]
    or encoding.get("full_video_encode_calls_per_row") != 3
    or encoding.get("independent_rgb_reference_encode_calls_per_row") != 12
    or encoding.get("independent_vae_encode_calls_per_row") != 15
    or encoding.get("references_from_full_video_posterior_slice") is not False
    or encoding.get("native_deployment_visual_conditioning")
    != "one_video_plus_four_rgb_refs"
    or receipt.get("spec", {}).get("reference_encoding_contract_digest")
    != encoding.get("reference_encoding_contract_digest")
):
    raise SystemExit("dataset RV2V-4 encoding contract differs")
parquet = pq.ParquetFile(root / "dataset.parquet")
if parquet.metadata.num_rows != 2:
    raise SystemExit("portrait cohort must contain exactly two orbit rows")
required_posterior_columns = {
    f"{member}_full_posterior_blob"
    for member in ("source", "variant_a", "variant_b")
} | {
    f"{member}_ref{index}_posterior_blob"
    for member in ("source", "variant_a", "variant_b")
    for index in (0, 27, 53, 80)
}
actual_posterior_columns = {
    name for name in parquet.schema_arrow.names if name.endswith("_posterior_blob")
}
if actual_posterior_columns != required_posterior_columns:
    raise SystemExit("dataset RV2V-4 posterior column closure differs")
row_contracts = pq.read_table(
    root / "dataset.parquet",
    columns=[
        "reference_count",
        "reference_indices_json",
        "reference_encoding_contract_digest",
        "independent_vae_encode_calls",
        "three_full_videos_independently_encoded",
        "twelve_rgb_references_independently_encoded",
        "references_from_full_video_posterior_slice",
        "native_deployment_visual_conditioning",
    ],
).to_pylist()
if any(
    row.get("reference_count") != 4
    or row.get("reference_indices_json") != "[0,27,53,80]"
    or row.get("reference_encoding_contract_digest")
    != encoding.get("reference_encoding_contract_digest")
    or row.get("independent_vae_encode_calls") != 15
    or row.get("three_full_videos_independently_encoded") is not True
    or row.get("twelve_rgb_references_independently_encoded") is not True
    or row.get("references_from_full_video_posterior_slice") is not False
    or row.get("native_deployment_visual_conditioning")
    != "one_video_plus_four_rgb_refs"
    for row in row_contracts
):
    raise SystemExit("dataset rows are not strict RV2V-4/15-call rows")
PY

echo "[bernini-cio-vae] PASS output=${output} receipt_sha256=$(sha256sum "${output}/receipt.json" | awk '{print $1}')"
