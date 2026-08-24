#!/usr/bin/env bash
set -Eeuo pipefail

source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?set MOTIVE_SOURCE_SNAPSHOT}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?set MOTIVE_SOURCE_TREE_SHA256}"
experiment_root="${MOTIVE_R10A_EXPERIMENT_ROOT:?set MOTIVE_R10A_EXPERIMENT_ROOT}"
run_root="${MOTIVE_R10A_RUN_ROOT:?set MOTIVE_R10A_RUN_ROOT}"
parent_run="${MOTIVE_R7_PARENT_RUN_ROOT:?set MOTIVE_R7_PARENT_RUN_ROOT}"
model_workspace="${MOTIVE_MODEL_WORKSPACE:?set MOTIVE_MODEL_WORKSPACE}"
python_bin="${PYTHON_BIN:?set PYTHON_BIN}"
seed="${MOTIVE_R10A_SEED:?set MOTIVE_R10A_SEED}"
attempt="${MOTIVE_R10A_ATTEMPT:?set MOTIVE_R10A_ATTEMPT}"
job_name="${MOTIVE_R10A_JOB_NAME:?set MOTIVE_R10A_JOB_NAME}"
attempt_receipt="${MOTIVE_R10A_ATTEMPT_RECEIPT:?set MOTIVE_R10A_ATTEMPT_RECEIPT}"
slurm_job_id="${SLURM_JOB_ID:?set SLURM_JOB_ID}"
slurm_job_name="${SLURM_JOB_NAME:?set SLURM_JOB_NAME}"

candidate_done_sha256="${MOTIVE_R7_CANDIDATE_TEMPORAL_DONE_SHA256:?set MOTIVE_R7_CANDIDATE_TEMPORAL_DONE_SHA256}"
track_cache_done_sha256="${MOTIVE_R7_CANDIDATE_TRACK_CACHE_DONE_SHA256:?set MOTIVE_R7_CANDIDATE_TRACK_CACHE_DONE_SHA256}"
visual_features_done_sha256="${MOTIVE_R7_VISUAL_FEATURES_DONE_SHA256:?set MOTIVE_R7_VISUAL_FEATURES_DONE_SHA256}"
visual_candidates_sha256="${MOTIVE_R7_VISUAL_CANDIDATES_SHA256:?set MOTIVE_R7_VISUAL_CANDIDATES_SHA256}"
screen_inputs_receipt_sha256="${MOTIVE_R7_SCREEN_INPUTS_RECEIPT_SHA256:?set MOTIVE_R7_SCREEN_INPUTS_RECEIPT_SHA256}"

for digest in \
  "${source_tree_sha256}" \
  "${candidate_done_sha256}" \
  "${track_cache_done_sha256}" \
  "${visual_features_done_sha256}" \
  "${visual_candidates_sha256}" \
  "${screen_inputs_receipt_sha256}"; do
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[r10a-controller] invalid required SHA-256: ${digest}" >&2
    exit 2
  fi
done
if [[ ! "${seed}" =~ ^[0-9]+$ ]] || (( seed >= 4294967296 )); then
  echo "[r10a-controller] seed must be in [0,2**32)" >&2
  exit 2
fi
if [[ ! "${attempt}" =~ ^[0-9]+$ ]] || (( attempt < 1 )) \
  || [[ ! "${slurm_job_id}" =~ ^[0-9]+$ ]] \
  || [[ "${slurm_job_name}" != "${job_name}" ]]; then
  echo "[r10a-controller] Slurm attempt identity differs" >&2
  exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
  echo "[r10a-controller] Python is not executable: ${python_bin}" >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "[r10a-controller] flock is required" >&2
  exit 2
fi
snapshot_verifier="${source_snapshot}/methods/motive/scripts/action_source_snapshot.py"
path_contract_source="${source_snapshot}/methods/motive/motive/r10_path_contract.py"
"${python_bin}" - \
  "${experiment_root}" \
  "${source_snapshot}" \
  "${snapshot_verifier}" \
  "${path_contract_source}" <<'PY'
import os
from pathlib import Path
import stat
import sys

raw_root, raw_snapshot, raw_verifier, raw_contract = sys.argv[1:]
if (
    not os.path.isabs(raw_root)
    or raw_root.startswith("//")
    or raw_root != os.path.normpath(raw_root)
):
    raise SystemExit("experiment root is not canonical")
root = Path(raw_root)
if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
    raise SystemExit("experiment root is redirected")
expected = {
    Path(raw_snapshot): (root / "source_snapshot", "dir"),
    Path(raw_verifier): (
        root
        / "source_snapshot"
        / "methods"
        / "motive"
        / "scripts"
        / "action_source_snapshot.py",
        "file",
    ),
    Path(raw_contract): (
        root
        / "source_snapshot"
        / "methods"
        / "motive"
        / "motive"
        / "r10_path_contract.py",
        "file",
    ),
}
for observed, (target, leaf_kind) in expected.items():
    if observed != target:
        raise SystemExit("source bootstrap path differs")
    current = root
    for index, part in enumerate(target.relative_to(root).parts):
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise SystemExit(f"source bootstrap ancestry is symlinked: {current}")
        is_leaf = index == len(target.relative_to(root).parts) - 1
        if not is_leaf and not stat.S_ISDIR(metadata.st_mode):
            raise SystemExit("source bootstrap ancestor is not a directory")
        if is_leaf and leaf_kind == "dir" and not stat.S_ISDIR(metadata.st_mode):
            raise SystemExit("source snapshot is not a directory")
        if is_leaf and leaf_kind == "file" and not stat.S_ISREG(metadata.st_mode):
            raise SystemExit("source bootstrap leaf is not a regular file")
PY
"${python_bin}" "${snapshot_verifier}" verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"
export PYTHONPATH="${source_snapshot}/methods/motive"
export PYTHONDONTWRITEBYTECODE=1

candidate_manifest_dir="${parent_run}/expansion/candidate_temporal_screen_v1"
track_cache_final="${parent_run}/expansion/candidate_track_cache_v1/final"
screen_input_bundle="${parent_run}/expansion/screen_inputs_v14"
visual_features_final="${screen_input_bundle}/visual_features_v1/final"
visual_candidates_manifest="${screen_input_bundle}/visual_candidates_v1/candidates.jsonl"
screen_inputs_receipt="${parent_run}/provenance/screen_inputs_v14.receipt.txt"
registry_path="${source_snapshot}/methods/motive/configs/instruction_video_editor_registry_v1.json"
availability_output="${run_root}/renderer_registry/availability_before_representation_gate.json"
search_output="${run_root}/representation/search_seed_${seed}"

for required_dir in \
  "${source_snapshot}" \
  "${candidate_manifest_dir}" \
  "${track_cache_final}" \
  "${screen_input_bundle}" \
  "${visual_features_final}" \
  "$(dirname "${visual_candidates_manifest}")" \
  "${model_workspace}"; do
  if [[ ! -d "${required_dir}" ]] || [[ -L "${required_dir}" ]]; then
    echo "[r10a-controller] missing/symlinked directory: ${required_dir}" >&2
    exit 2
  fi
done
for required_file in \
  "${candidate_manifest_dir}/done.json" \
  "${track_cache_final}/done.json" \
  "${visual_features_final}/done.json" \
  "${visual_candidates_manifest}" \
  "${screen_inputs_receipt}" \
  "${registry_path}" \
  "${source_snapshot}/methods/motive/motive/r10_dynamic_dino_representation_search.py"; do
  if [[ ! -s "${required_file}" ]] || [[ -L "${required_file}" ]]; then
    echo "[r10a-controller] missing/empty/symlinked file: ${required_file}" >&2
    exit 2
  fi
done

"${python_bin}" - \
  "${experiment_root}" \
  "${run_root}" \
  "${attempt_receipt}" \
  "${seed}" \
  "${attempt}" <<'PY'
from pathlib import Path
import sys

from motive.r10_path_contract import (
    canonical_experiment_root,
    ensure_experiment_directory,
    require_attempt_receipt_path,
    require_experiment_path,
)

experiment_root, raw_run_root, raw_receipt, raw_seed, raw_attempt = sys.argv[1:]
root = canonical_experiment_root(experiment_root)
seed = int(raw_seed)
attempt = int(raw_attempt)
run_root = require_experiment_path(
    raw_run_root,
    root,
    f"seed_{seed}",
    kind="dir",
)
require_attempt_receipt_path(raw_receipt, root, seed, attempt)
for name in ("representation", "renderer_registry", "logs", "provenance"):
    ensure_experiment_directory(
        run_root / name,
        root,
        f"seed_{seed}/{name}",
    )
require_experiment_path(
    run_root / ".controller.lock",
    root,
    f"seed_{seed}/.controller.lock",
    kind="file",
    allow_missing=True,
)
require_experiment_path(
    run_root
    / "renderer_registry"
    / "availability_before_representation_gate.json",
    root,
    (
        f"seed_{seed}/renderer_registry/"
        "availability_before_representation_gate.json"
    ),
    kind="file",
    allow_missing=True,
)
require_experiment_path(
    run_root / "representation" / f"search_seed_{seed}",
    root,
    f"seed_{seed}/representation/search_seed_{seed}",
    kind="dir",
    allow_missing=True,
)
require_experiment_path(
    run_root / "provenance" / f"search_seed_{seed}.producer.json",
    root,
    f"seed_{seed}/provenance/search_seed_{seed}.producer.json",
    kind="file",
    allow_missing=True,
)
PY
exec 9>"${run_root}/.controller.lock"
if ! flock -n 9; then
  echo "[r10a-controller] another controller owns this run" >&2
  exit 4
fi

"${python_bin}" - \
  "${attempt_receipt}" \
  "${experiment_root}" \
  "${source_tree_sha256}" \
  "${seed}" \
  "${attempt}" \
  "${slurm_job_id}" \
  "${job_name}" \
  "${run_root}" <<'PY'
import json
from pathlib import Path
import stat
import sys

from motive.r10_path_contract import (
    canonical_experiment_root,
    require_attempt_receipt_path,
    require_experiment_path,
)

(
    raw_receipt,
    experiment_root,
    source_tree_sha256,
    raw_seed,
    raw_attempt,
    raw_job_id,
    job_name,
    run_root,
) = sys.argv[1:]
root = canonical_experiment_root(experiment_root)
seed = int(raw_seed)
attempt = int(raw_attempt)
require_experiment_path(
    run_root,
    root,
    f"seed_{seed}",
    kind="dir",
)
receipt = require_attempt_receipt_path(
    raw_receipt,
    root,
    seed,
    attempt,
)
if receipt.is_symlink() or not receipt.is_file() or receipt.stat().st_size < 1:
    raise SystemExit("attempt receipt is not one regular nonempty file")
metadata = receipt.stat()
if metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o444:
    raise SystemExit("attempt receipt permission contract differs")
with receipt.open(encoding="utf-8") as handle:
    row = json.load(handle)
expected = {
    "schema_version": "motive-r10a-job-attempt-receipt-v1",
    "submitted_at_utc": row.get("submitted_at_utc"),
    "experiment_root": experiment_root,
    "source_tree_sha256": source_tree_sha256,
    "seed": seed,
    "attempt": attempt,
    "job_id": int(raw_job_id),
    "job_name": job_name,
    "run_root": run_root,
}
if (
    set(row) != set(expected)
    or not isinstance(row.get("submitted_at_utc"), str)
    or not row["submitted_at_utc"]
    or row != expected
):
    raise SystemExit("attempt receipt identity differs")
PY

for anchored in \
  "${candidate_manifest_dir}/done.json|${candidate_done_sha256}" \
  "${track_cache_final}/done.json|${track_cache_done_sha256}" \
  "${visual_features_final}/done.json|${visual_features_done_sha256}" \
  "${visual_candidates_manifest}|${visual_candidates_sha256}" \
  "${screen_inputs_receipt}|${screen_inputs_receipt_sha256}"; do
  anchored_path="${anchored%%|*}"
  anchored_sha256="${anchored#*|}"
  if [[ "$(sha256sum "${anchored_path}" | awk '{print $1}')" \
    != "${anchored_sha256}" ]]; then
    echo "[r10a-controller] upstream SHA differs: ${anchored_path}" >&2
    exit 2
  fi
done
if [[ "$(stat -c '%h' "${screen_inputs_receipt}")" != "1" ]] \
  || [[ "$(stat -c '%a' "${screen_inputs_receipt}")" != "444" ]]; then
  echo "[r10a-controller] sealed input receipt mode differs" >&2
  exit 2
fi

actual_module="$(
  "${python_bin}" -c '
from pathlib import Path
import motive.r10_dynamic_dino_representation_search as module
print(Path(module.__file__).resolve(strict=True))
'
)"
expected_module="${source_snapshot}/methods/motive/motive/r10_dynamic_dino_representation_search.py"
if [[ "${actual_module}" != "${expected_module}" ]]; then
  echo "[r10a-controller] imported R10A module is not frozen" >&2
  exit 2
fi

"${python_bin}" -c '
from pathlib import Path
from motive.r7_artifact_permissions import assert_sealed_tree
import sys
for raw in sys.argv[1:]:
    assert_sealed_tree(Path(raw))
' \
  "${screen_input_bundle}" \
  "${visual_features_final}" \
  "$(dirname "${visual_candidates_manifest}")"

if [[ ! -e "${availability_output}" ]]; then
  "${python_bin}" -m motive.instruction_model_registry \
    --registry "${registry_path}" \
    --workspace "${model_workspace}" \
    --output "${availability_output}"
fi

common_arguments=(
  --candidate-manifest-dir "${candidate_manifest_dir}"
  --expected-candidate-manifest-done-sha256 "${candidate_done_sha256}"
  --track-cache-final "${track_cache_final}"
  --expected-track-cache-done-sha256 "${track_cache_done_sha256}"
  --visual-features-final "${visual_features_final}"
  --expected-visual-features-done-sha256 "${visual_features_done_sha256}"
  --visual-candidates-manifest "${visual_candidates_manifest}"
  --expected-visual-candidates-sha256 "${visual_candidates_sha256}"
  --source-tree-sha256 "${source_tree_sha256}"
  --source-tree-verified-by-controller
)

artifact_created=false
if [[ ! -e "${search_output}" ]] && [[ ! -L "${search_output}" ]]; then
  "${python_bin}" -m motive.r10_dynamic_dino_representation_search \
    "${common_arguments[@]}" \
    --output-dir "${search_output}" \
    --seed "${seed}" \
    --repeats 2 \
    --folds 3
  artifact_created=true
fi
"${python_bin}" -c '
from pathlib import Path
from motive.r10_dynamic_dino_representation_search import validate_published_search
import json
import sys
result = validate_published_search(Path(sys.argv[1]))
decision = result["summary"]["decision"]
assert decision["representation_gate_passed"] is False
assert decision["renderer_probe_authorized"] is False
assert decision["editor_training_authorized"] is False
print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
' "${search_output}"

producer_receipt="${run_root}/provenance/search_seed_${seed}.producer.json"
if [[ "${artifact_created}" == true ]]; then
  producer_mode=publish
else
  producer_mode=validate
fi
"${python_bin}" - \
  "${producer_receipt}" \
  "${search_output}" \
  "${source_tree_sha256}" \
  "${seed}" \
  "${slurm_job_id}" \
  "${attempt}" \
  "${job_name}" \
  "${attempt_receipt}" \
  "${run_root}" \
  "${experiment_root}" \
  "${producer_mode}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from datetime import datetime, timezone

from motive import r10_dynamic_dino_representation_search as r10
from motive.r10_path_contract import (
    canonical_experiment_root,
    require_attempt_receipt_path,
    require_experiment_path,
)

(
    raw_producer_receipt,
    raw_artifact,
    source_tree_sha256,
    raw_seed,
    raw_job_id,
    raw_attempt,
    job_name,
    raw_attempt_receipt,
    raw_run_root,
    raw_experiment_root,
    mode,
) = sys.argv[1:]
seed = int(raw_seed)
job_id = int(raw_job_id)
attempt = int(raw_attempt)
experiment_root = canonical_experiment_root(raw_experiment_root)
run_root = require_experiment_path(
    raw_run_root,
    experiment_root,
    f"seed_{seed}",
    kind="dir",
)
artifact = require_experiment_path(
    raw_artifact,
    experiment_root,
    f"seed_{seed}/representation/search_seed_{seed}",
    kind="dir",
)
attempt_receipt = require_attempt_receipt_path(
    raw_attempt_receipt,
    experiment_root,
    seed,
    attempt,
)
producer_receipt = require_experiment_path(
    raw_producer_receipt,
    experiment_root,
    f"seed_{seed}/provenance/search_seed_{seed}.producer.json",
    kind="file",
    allow_missing=mode == "publish",
)


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed_identity(path, context):
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
        raise SystemExit(f"{context} is not one regular nonempty file")
    metadata = path.stat()
    if metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o444:
        raise SystemExit(f"{context} permission contract differs")
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": digest_file(path),
        "bytes": metadata.st_size,
    }


attempt_identity = sealed_identity(attempt_receipt, "attempt receipt")
with attempt_receipt.open(encoding="utf-8") as handle:
    attempt_row = json.load(handle)
expected_attempt = {
    "schema_version": "motive-r10a-job-attempt-receipt-v1",
    "submitted_at_utc": attempt_row.get("submitted_at_utc"),
    "experiment_root": str(run_root.parent),
    "source_tree_sha256": source_tree_sha256,
    "seed": seed,
    "attempt": attempt,
    "job_id": job_id,
    "job_name": job_name,
    "run_root": str(run_root),
}
if (
    set(attempt_row) != set(expected_attempt)
    or not isinstance(attempt_row.get("submitted_at_utc"), str)
    or not attempt_row["submitted_at_utc"]
    or attempt_row != expected_attempt
):
    raise SystemExit("producer attempt receipt identity differs")

validated = r10.validate_published_search(artifact)
summary = validated["summary"]
source = summary.get("source_snapshot")
if (
    summary.get("seed") != seed
    or not isinstance(source, dict)
    or source.get("tree_sha256") != source_tree_sha256
    or source.get("exact_tree_verified_by_controller_before_search") is not True
):
    raise SystemExit("producer artifact provenance differs")
artifact_identity = {
    "root": str(artifact),
    "artifact_digest": validated["done"]["artifact_digest"],
    "done_sha256": digest_file(artifact / r10.DONE_NAME),
    "summary_sha256": digest_file(artifact / r10.SUMMARY_NAME),
}
payload = {
    "schema_version": "motive-r10a-artifact-producer-v1",
    "produced_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_tree_sha256": source_tree_sha256,
    "seed": seed,
    "artifact": artifact_identity,
    "producer_job": {
        "job_id": job_id,
        "attempt": attempt,
        "job_name": job_name,
        "attempt_receipt": attempt_identity,
    },
    "representation_gate_passed": False,
    "renderer_probe_authorized": False,
    "editor_training_authorized": False,
}


def validate_observed():
    identity = sealed_identity(producer_receipt, "producer receipt")
    with producer_receipt.open(encoding="utf-8") as handle:
        observed = json.load(handle)
    producer = observed.get("producer_job")
    if not isinstance(producer, dict):
        raise SystemExit("producer job identity is absent")
    producer_attempt_identity = producer.get("attempt_receipt")
    if not isinstance(producer_attempt_identity, dict):
        raise SystemExit("producer attempt receipt identity is absent")
    producer_attempt_path = Path(
        producer_attempt_identity.get("path", "")
    )
    producer_attempt_path = require_attempt_receipt_path(
        producer_attempt_path,
        experiment_root,
        seed,
        producer.get("attempt"),
    )
    actual_attempt_identity = sealed_identity(
        producer_attempt_path,
        "producer attempt receipt",
    )
    if actual_attempt_identity != producer_attempt_identity:
        raise SystemExit("producer attempt receipt digest differs")
    with producer_attempt_path.open(encoding="utf-8") as handle:
        producer_attempt = json.load(handle)
    expected_producer_attempt = {
        "schema_version": "motive-r10a-job-attempt-receipt-v1",
        "submitted_at_utc": producer_attempt.get("submitted_at_utc"),
        "experiment_root": str(run_root.parent),
        "source_tree_sha256": source_tree_sha256,
        "seed": seed,
        "attempt": producer.get("attempt"),
        "job_id": producer.get("job_id"),
        "job_name": producer.get("job_name"),
        "run_root": str(run_root),
    }
    expected = {
        "schema_version": "motive-r10a-artifact-producer-v1",
        "produced_at_utc": observed.get("produced_at_utc"),
        "source_tree_sha256": source_tree_sha256,
        "seed": seed,
        "artifact": artifact_identity,
        "producer_job": {
            "job_id": producer.get("job_id"),
            "attempt": producer.get("attempt"),
            "job_name": producer.get("job_name"),
            "attempt_receipt": actual_attempt_identity,
        },
        "representation_gate_passed": False,
        "renderer_probe_authorized": False,
        "editor_training_authorized": False,
    }
    if (
        set(observed) != set(expected)
        or not isinstance(observed.get("produced_at_utc"), str)
        or not observed["produced_at_utc"]
        or isinstance(producer.get("job_id"), bool)
        or not isinstance(producer.get("job_id"), int)
        or producer["job_id"] < 1
        or isinstance(producer.get("attempt"), bool)
        or not isinstance(producer.get("attempt"), int)
        or producer["attempt"] < 1
        or not isinstance(producer.get("job_name"), str)
        or not producer["job_name"]
        or set(producer_attempt) != set(expected_producer_attempt)
        or not isinstance(producer_attempt.get("submitted_at_utc"), str)
        or not producer_attempt["submitted_at_utc"]
        or producer_attempt != expected_producer_attempt
        or observed != expected
    ):
        raise SystemExit("producer receipt contract differs")
    return identity


if mode == "publish":
    if producer_receipt.exists() or producer_receipt.is_symlink():
        raise SystemExit("producer receipt already exists during first publish")
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = producer_receipt.with_name(
        f".{producer_receipt.name}.{os.getpid()}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(temporary, 0o444)
    try:
        os.link(temporary, producer_receipt, follow_symlinks=False)
    finally:
        temporary.unlink()
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    directory = os.open(producer_receipt.parent, directory_flags)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
elif mode != "validate":
    raise SystemExit("unsupported producer receipt mode")
validate_observed()
PY

echo \
  "[r10a-controller] completed search=${search_output} " \
  "validation_job=${slurm_job_id} producer_receipt=${producer_receipt}"
