#!/usr/bin/env bash
set -Eeuo pipefail

ssh_host="${MOTIVE_R10A_SSH_HOST:-auh}"
experiment_root="${MOTIVE_R10A_EXPERIMENT_ROOT:?set experiment root}"
job_ids_csv="${MOTIVE_R10A_JOB_IDS:?set two comma-separated job IDs}"
python_bin="${MOTIVE_R10A_PYTHON_BIN:?set remote Python}"
source_tree_sha256="${MOTIVE_R10A_SOURCE_TREE_SHA256:?set source digest}"
parent_run="${MOTIVE_R10A_PARENT_RUN:?set parent R7 run}"
model_workspace="${MOTIVE_R10A_MODEL_WORKSPACE:?set model workspace}"
remote_submit_script="${MOTIVE_R10A_REMOTE_SUBMIT_SCRIPT:?set local remote-submit script}"
poll_seconds="${MOTIVE_R10A_WATCH_POLL_SECONDS:-30}"
max_polls="${MOTIVE_R10A_WATCH_MAX_POLLS:-1200}"

allowed_prefix=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto
case "${experiment_root}" in
  "${allowed_prefix}"/goku_repr_auto_r10a_*) ;;
  *)
    echo "[r10a-watch] unsafe experiment root: ${experiment_root}" >&2
    exit 2
    ;;
esac
if [[ ! "${source_tree_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || [[ ! "${poll_seconds}" =~ ^[0-9]+$ ]] \
  || [[ ! "${max_polls}" =~ ^[0-9]+$ ]] \
  || ((poll_seconds < 5 || max_polls < 1)) \
  || [[ ! -f "${remote_submit_script}" ]] \
  || [[ ! -s "${remote_submit_script}" ]] \
  || [[ -L "${remote_submit_script}" ]]; then
  echo "[r10a-watch] invalid watcher inputs" >&2
  exit 2
fi

set_job_ids() {
  local candidate="${1:?job ID CSV}"
  if [[ ! "${candidate}" =~ ^[0-9]+,[0-9]+$ ]] \
    || [[ "${candidate%%,*}" == "${candidate#*,}" ]]; then
    echo "[r10a-watch] invalid coordinated job IDs: ${candidate}" >&2
    return 3
  fi
  job_ids_csv="${candidate}"
  IFS=, read -r -a job_ids <<< "${job_ids_csv}"
}

probe_auh() {
  ssh -o ConnectTimeout=10 -o BatchMode=yes "${ssh_host}" true
}

invoke_remote_submit() {
  ssh "${ssh_host}" bash -s -- \
    "${experiment_root}" \
    "${source_tree_sha256}" \
    "${parent_run}" \
    "${model_workspace}" \
    "${python_bin}" \
    < "${remote_submit_script}"
}

read_remote_job_ids() {
  ssh "${ssh_host}" bash -s -- \
    "${experiment_root}" \
    "${python_bin}" \
    "${source_tree_sha256}" <<'REMOTE'
set -Eeuo pipefail
root="$1"
python_bin="$2"
source_tree_sha256="$3"
receipt="${root}/submission.json"
snapshot="${root}/source_snapshot"
snapshot_verifier="${snapshot}/methods/motive/scripts/action_source_snapshot.py"
path_contract_source="${snapshot}/methods/motive/motive/r10_path_contract.py"
"${python_bin}" - \
  "${root}" \
  "${snapshot}" \
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
    parts = target.relative_to(root).parts
    current = root
    for index, part in enumerate(parts):
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise SystemExit(f"source bootstrap ancestry is symlinked: {current}")
        is_leaf = index == len(parts) - 1
        if not is_leaf and not stat.S_ISDIR(metadata.st_mode):
            raise SystemExit("source bootstrap ancestor is not a directory")
        if is_leaf and leaf_kind == "dir" and not stat.S_ISDIR(metadata.st_mode):
            raise SystemExit("source snapshot is not a directory")
        if is_leaf and leaf_kind == "file" and not stat.S_ISREG(metadata.st_mode):
            raise SystemExit("source bootstrap leaf is not a regular file")
PY
"${python_bin}" "${snapshot_verifier}" verify \
  --snapshot "${snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}" >/dev/null
if [[ ! -f "${receipt}" ]] || [[ -L "${receipt}" ]] \
  || [[ ! -s "${receipt}" ]]; then
  echo "[r10a-watch] remote submission receipt is invalid" >&2
  exit 3
fi
export PYTHONPATH="${root}/source_snapshot/methods/motive"
export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" - "${receipt}" "${root}" "${source_tree_sha256}" <<'PY'
import hashlib
import json
from pathlib import Path
import stat
import sys

from motive.r10_path_contract import (
    canonical_experiment_root,
    require_attempt_receipt_path,
    require_experiment_path,
)

raw_receipt, raw_root, source_tree_sha256 = sys.argv[1:]
root_path = canonical_experiment_root(raw_root)
root = str(root_path)
receipt = require_experiment_path(
    raw_receipt,
    root_path,
    "submission.json",
    kind="file",
)
with open(receipt, encoding="utf-8") as handle:
    row = json.load(handle)
jobs = row.get("jobs")
if (
    row.get("schema_version")
    != "motive-r10a-two-seed-submission-state-v4"
    or row.get("experiment_root") != root
    or row.get("source_tree_sha256") != source_tree_sha256
    or not isinstance(jobs, list)
    or len(jobs) != 2
    or row.get("renderer_probe_submitted") is not False
    or row.get("editor_training_submitted") is not False
):
    raise SystemExit("remote submission receipt contract differs")
experiment_token = hashlib.sha256(root.encode("utf-8")).hexdigest()


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


by_seed = {}
for job in jobs:
    if not isinstance(job, dict):
        raise SystemExit("remote submission job row is invalid")
    seed = job.get("seed")
    job_id = job.get("job_id")
    attempt = job.get("attempt")
    job_name = job.get("job_name")
    attempt_identity = job.get("attempt_receipt")
    run_root_path = require_experiment_path(
        f"{root}/seed_{seed}",
        root_path,
        f"seed_{seed}",
        kind="dir",
    )
    run_root = str(run_root_path)
    if (
        seed not in (260108837, 260108838)
        or seed in by_seed
        or isinstance(job_id, bool)
        or not isinstance(job_id, int)
        or job_id < 1
        or isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt < 1
        or job_name
        != f"m10a-{experiment_token}-s{seed}-a{attempt}"
        or not isinstance(job.get("state"), str)
        or not job["state"]
        or not isinstance(job.get("action"), str)
        or not job["action"]
        or not isinstance(job.get("seed_artifact_validated"), bool)
        or job.get("run_root") != run_root
        or not isinstance(attempt_identity, dict)
    ):
        raise SystemExit("remote submission job identity differs")
    attempt_path = require_attempt_receipt_path(
        attempt_identity.get("path", ""),
        root_path,
        seed,
        attempt,
    )
    if (
        attempt_path.is_symlink()
        or not attempt_path.is_file()
        or attempt_path.stat().st_size < 1
    ):
        raise SystemExit("remote attempt receipt is invalid")
    metadata = attempt_path.stat()
    actual_identity = {
        "path": str(attempt_path.resolve(strict=True)),
        "sha256": digest_file(attempt_path),
        "bytes": metadata.st_size,
    }
    if (
        metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or attempt_identity != actual_identity
    ):
        raise SystemExit("remote attempt receipt seal differs")
    with attempt_path.open(encoding="utf-8") as handle:
        attempt_row = json.load(handle)
    expected_attempt = {
        "schema_version": "motive-r10a-job-attempt-receipt-v1",
        "submitted_at_utc": attempt_row.get("submitted_at_utc"),
        "experiment_root": root,
        "source_tree_sha256": source_tree_sha256,
        "seed": seed,
        "attempt": attempt,
        "job_id": job_id,
        "job_name": job_name,
        "run_root": run_root,
    }
    if (
        set(attempt_row) != set(expected_attempt)
        or not isinstance(attempt_row.get("submitted_at_utc"), str)
        or not attempt_row["submitted_at_utc"]
        or attempt_row != expected_attempt
    ):
        raise SystemExit("remote attempt receipt identity differs")
    by_seed[seed] = job_id
if set(by_seed) != {260108837, 260108838}:
    raise SystemExit("remote submission seed closure differs")
if len(set(by_seed.values())) != 2:
    raise SystemExit("remote submission job IDs are not distinct")
print(f"{by_seed[260108837]},{by_seed[260108838]}")
PY
REMOTE
}

remote_finalization() {
  local mode="${1:?validate or finalize}"
  ssh "${ssh_host}" bash -s -- \
    "${experiment_root}" \
    "${python_bin}" \
    "${source_tree_sha256}" \
    "${mode}" <<'REMOTE'
set -Eeuo pipefail
root="$1"
python_bin="$2"
source_tree_sha256="$3"
mode="$4"
case "${mode}" in
  validate|finalize) ;;
  *) exit 2 ;;
esac
final_receipt="${root}/final_validation.json"
seed_receipt="${root}/seed_validation.json"
aggregate="${root}/cross_seed/final"
snapshot_verifier="${root}/source_snapshot/methods/motive/scripts/action_source_snapshot.py"
if [[ ! -f "${snapshot_verifier}" ]] || [[ -L "${snapshot_verifier}" ]] \
  || [[ ! -s "${snapshot_verifier}" ]]; then
  echo "[r10a-watch] source snapshot verifier is invalid" >&2
  exit 3
fi
"${python_bin}" - "${root}" "${snapshot_verifier}" <<'PY'
import os
from pathlib import Path
import stat
import sys

raw_root, raw_verifier = sys.argv[1:]
if (
    not os.path.isabs(raw_root)
    or raw_root.startswith("//")
    or raw_root != os.path.normpath(raw_root)
):
    raise SystemExit("experiment root is not canonical")
root = Path(raw_root)
if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
    raise SystemExit("experiment root is redirected")
verifier = Path(raw_verifier)
expected = (
    root
    / "source_snapshot"
    / "methods"
    / "motive"
    / "scripts"
    / "action_source_snapshot.py"
)
if verifier != expected:
    raise SystemExit("snapshot verifier path differs")
current = root
for part in expected.relative_to(root).parts:
    current = current / part
    metadata = current.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise SystemExit(f"snapshot verifier ancestry is symlinked: {current}")
lock_path = root / ".finalization.lock"
if lock_path.is_symlink():
    raise SystemExit("finalization lock must not be a symlink")
PY
if [[ "${mode}" == validate ]] \
  && [[ ! -e "${final_receipt}" ]] \
  && [[ ! -L "${final_receipt}" ]]; then
  exit 44
fi
"${python_bin}" "${snapshot_verifier}" verify \
  --snapshot "${root}/source_snapshot" \
  --expected-tree-sha256 "${source_tree_sha256}"
export PYTHONPATH="${root}/source_snapshot/methods/motive"
export PYTHONDONTWRITEBYTECODE=1
exec 9>"${root}/.finalization.lock"
flock -w 30 9
"${python_bin}" - "${root}" "${mode}" <<'PY'
import sys

from motive.r10_path_contract import (
    canonical_experiment_root,
    ensure_experiment_directory,
    require_experiment_path,
)

raw_root, mode = sys.argv[1:]
root = canonical_experiment_root(raw_root)
require_experiment_path(
    root / "source_snapshot",
    root,
    "source_snapshot",
    kind="dir",
)
if mode == "finalize":
    ensure_experiment_directory(root / "cross_seed", root, "cross_seed")
    require_experiment_path(
        root / "cross_seed" / "final",
        root,
        "cross_seed/final",
        kind="dir",
        allow_missing=True,
    )
    require_experiment_path(
        root / "seed_validation.json",
        root,
        "seed_validation.json",
        kind="file",
    )
    for seed in (260108837, 260108838):
        require_experiment_path(
            root
            / f"seed_{seed}"
            / "representation"
            / f"search_seed_{seed}",
            root,
            f"seed_{seed}/representation/search_seed_{seed}",
            kind="dir",
        )
else:
    require_experiment_path(
        root / "cross_seed",
        root,
        "cross_seed",
        kind="dir",
    )
    require_experiment_path(
        root / "final_validation.json",
        root,
        "final_validation.json",
        kind="file",
    )
PY
if [[ "${mode}" == finalize ]]; then
  if [[ ! -f "${seed_receipt}" ]] || [[ -L "${seed_receipt}" ]] \
    || [[ ! -s "${seed_receipt}" ]]; then
    echo "[r10a-watch] seed validation receipt is absent" >&2
    exit 3
  fi
  first="${root}/seed_260108837/representation/search_seed_260108837"
  second="${root}/seed_260108838/representation/search_seed_260108838"
  "${python_bin}" -m motive.r10_cross_seed_aggregate build \
    --seed-artifact-dir "${first}" \
    --seed-artifact-dir "${second}" \
    --expected-source-tree-sha256 "${source_tree_sha256}" \
    --output-dir "${aggregate}"
  "${python_bin}" -m motive.r10_cross_seed_aggregate validate \
    --output-dir "${aggregate}"
fi
"${python_bin}" - \
  "${root}" \
  "${source_tree_sha256}" \
  "${seed_receipt}" \
  "${aggregate}" \
  "${final_receipt}" \
  "${mode}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from datetime import datetime, timezone

from motive import r10_dynamic_dino_representation_search as r10
from motive.r10_cross_seed_aggregate import validate_published_aggregate
from motive.r10_path_contract import (
    canonical_experiment_root,
    require_attempt_receipt_path,
    require_experiment_path,
)

root = canonical_experiment_root(sys.argv[1])
source_tree_sha256 = sys.argv[2]
mode = sys.argv[6]
seed_receipt = require_experiment_path(
    sys.argv[3],
    root,
    "seed_validation.json",
    kind="file",
)
aggregate = require_experiment_path(
    sys.argv[4],
    root,
    "cross_seed/final",
    kind="dir",
)
final_receipt = require_experiment_path(
    sys.argv[5],
    root,
    "final_validation.json",
    kind="file",
    allow_missing=mode == "finalize",
)
required_seeds = (260108837, 260108838)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_file(path):
    return digest_bytes(path.read_bytes())


def object_digest(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return digest_bytes(encoded)


def require_sealed_file(path, context):
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


def validate_attempt_identity(identity, *, seed, job_id, attempt, job_name):
    if not isinstance(identity, dict):
        raise SystemExit("attempt receipt identity is absent")
    path = require_attempt_receipt_path(
        identity.get("path", ""),
        root,
        seed,
        attempt,
    )
    actual = require_sealed_file(path, "attempt receipt")
    if actual != identity:
        raise SystemExit("attempt receipt digest differs")
    with path.open(encoding="utf-8") as handle:
        row = json.load(handle)
    expected = {
        "schema_version": "motive-r10a-job-attempt-receipt-v1",
        "submitted_at_utc": row.get("submitted_at_utc"),
        "experiment_root": str(root),
        "source_tree_sha256": source_tree_sha256,
        "seed": seed,
        "attempt": attempt,
        "job_id": job_id,
        "job_name": job_name,
        "run_root": f"{root}/seed_{seed}",
    }
    if (
        set(row) != set(expected)
        or not isinstance(row.get("submitted_at_utc"), str)
        or not row["submitted_at_utc"]
        or row != expected
    ):
        raise SystemExit("attempt receipt identity differs")
    return actual


def load_seed_validation():
    require_sealed_file(seed_receipt, "seed validation receipt")
    with seed_receipt.open(encoding="utf-8") as handle:
        row = json.load(handle)
    expected_keys = {
        "schema_version",
        "validated_at_utc",
        "experiment_root",
        "source_tree_sha256",
        "jobs",
        "both_seed_artifacts_validated",
        "representation_gate_passed",
        "renderer_probe_authorized",
        "editor_training_authorized",
    }
    jobs = row.get("jobs")
    if (
        set(row) != expected_keys
        or row.get("schema_version")
        != "motive-r10a-two-seed-validation-v2"
        or row.get("experiment_root") != str(root)
        or row.get("source_tree_sha256") != source_tree_sha256
        or row.get("both_seed_artifacts_validated") is not True
        or row.get("representation_gate_passed") is not False
        or row.get("renderer_probe_authorized") is not False
        or row.get("editor_training_authorized") is not False
        or not isinstance(row.get("validated_at_utc"), str)
        or not row["validated_at_utc"]
        or not isinstance(jobs, list)
        or len(jobs) != 2
        or [job.get("seed") for job in jobs] != list(required_seeds)
    ):
        raise SystemExit("seed validation receipt contract differs")
    experiment_token = hashlib.sha256(
        str(root).encode("utf-8")
    ).hexdigest()
    validation_ids = set()
    producer_ids = set()
    for seed, job in zip(required_seeds, jobs, strict=True):
        if (
            not isinstance(job, dict)
            or set(job)
            != {
                "seed",
                "validation_job",
                "artifact",
                "artifact_producer",
            }
            or job.get("seed") != seed
        ):
            raise SystemExit("seed validation job closure differs")
        validation = job.get("validation_job")
        producer = job.get("artifact_producer")
        artifact_identity = job.get("artifact")
        if not isinstance(validation, dict) or not isinstance(producer, dict):
            raise SystemExit("seed job identities are absent")
        validation_job_id = validation.get("job_id")
        validation_attempt = validation.get("attempt")
        validation_name = validation.get("job_name")
        expected_validation_name = (
            f"m10a-{experiment_token}-s{seed}-a{validation_attempt}"
        )
        if (
            set(validation)
            != {
                "job_id",
                "attempt",
                "job_name",
                "exit_code",
                "attempt_receipt",
            }
            or isinstance(validation_job_id, bool)
            or not isinstance(validation_job_id, int)
            or validation_job_id < 1
            or isinstance(validation_attempt, bool)
            or not isinstance(validation_attempt, int)
            or validation_attempt < 1
            or validation_name != expected_validation_name
            or validation.get("exit_code") != "0:0"
        ):
            raise SystemExit("validation job contract differs")
        validate_attempt_identity(
            validation["attempt_receipt"],
            seed=seed,
            job_id=validation_job_id,
            attempt=validation_attempt,
            job_name=validation_name,
        )
        validation_ids.add(validation_job_id)

        producer_job_id = producer.get("job_id")
        producer_attempt = producer.get("attempt")
        producer_name = producer.get("job_name")
        expected_producer_name = (
            f"m10a-{experiment_token}-s{seed}-a{producer_attempt}"
        )
        if (
            set(producer)
            != {
                "job_id",
                "attempt",
                "job_name",
                "attempt_receipt",
                "producer_receipt",
            }
            or isinstance(producer_job_id, bool)
            or not isinstance(producer_job_id, int)
            or producer_job_id < 1
            or isinstance(producer_attempt, bool)
            or not isinstance(producer_attempt, int)
            or producer_attempt < 1
            or producer_name != expected_producer_name
        ):
            raise SystemExit("artifact producer job contract differs")
        producer_attempt_identity = validate_attempt_identity(
            producer["attempt_receipt"],
            seed=seed,
            job_id=producer_job_id,
            attempt=producer_attempt,
            job_name=producer_name,
        )
        producer_receipt_identity = producer.get("producer_receipt")
        if not isinstance(producer_receipt_identity, dict):
            raise SystemExit("producer receipt identity is absent")
        producer_receipt_path = Path(
            producer_receipt_identity.get("path", "")
        )
        producer_receipt_path = require_experiment_path(
            producer_receipt_path,
            root,
            f"seed_{seed}/provenance/search_seed_{seed}.producer.json",
            kind="file",
        )
        actual_producer_receipt = require_sealed_file(
            producer_receipt_path,
            "producer receipt",
        )
        if actual_producer_receipt != producer_receipt_identity:
            raise SystemExit("producer receipt digest differs")
        with producer_receipt_path.open(encoding="utf-8") as handle:
            producer_row = json.load(handle)
        expected_producer_row = {
            "schema_version": "motive-r10a-artifact-producer-v1",
            "produced_at_utc": producer_row.get("produced_at_utc"),
            "source_tree_sha256": source_tree_sha256,
            "seed": seed,
            "artifact": artifact_identity,
            "producer_job": {
                "job_id": producer_job_id,
                "attempt": producer_attempt,
                "job_name": producer_name,
                "attempt_receipt": producer_attempt_identity,
            },
            "representation_gate_passed": False,
            "renderer_probe_authorized": False,
            "editor_training_authorized": False,
        }
        if (
            set(producer_row) != set(expected_producer_row)
            or not isinstance(producer_row.get("produced_at_utc"), str)
            or not producer_row["produced_at_utc"]
            or producer_row != expected_producer_row
        ):
            raise SystemExit("producer receipt contract differs")
        producer_ids.add(producer_job_id)
    if len(validation_ids) != 2 or len(producer_ids) != 2:
        raise SystemExit("seed job identities are not distinct")
    return row


def current_seed_identities(seed_validation):
    identities = {}
    for seed, seed_job in zip(
        required_seeds,
        seed_validation["jobs"],
        strict=True,
    ):
        artifact = (
            root
            / f"seed_{seed}"
            / "representation"
            / f"search_seed_{seed}"
        )
        artifact = require_experiment_path(
            artifact,
            root,
            f"seed_{seed}/representation/search_seed_{seed}",
            kind="dir",
        )
        validated = r10.validate_published_search(artifact)
        summary = validated["summary"]
        source = summary.get("source_snapshot")
        if (
            summary.get("seed") != seed
            or not isinstance(source, dict)
            or source.get("tree_sha256") != source_tree_sha256
            or source.get(
                "exact_tree_verified_by_controller_before_search"
            )
            is not True
        ):
            raise SystemExit(f"seed={seed} artifact provenance differs")
        done = validated["done"]
        payload_files = done["payload_files"]
        current_artifact = {
            "root": str(artifact.resolve(strict=True)),
            "artifact_digest": done["artifact_digest"],
            "done_sha256": digest_file(artifact / r10.DONE_NAME),
            "summary_sha256": digest_file(artifact / r10.SUMMARY_NAME),
        }
        if seed_job.get("artifact") != current_artifact:
            raise SystemExit(f"seed={seed} receipt artifact identity differs")
        identities[str(seed)] = {
            "seed": seed,
            "artifact_digest": done["artifact_digest"],
            "done_sha256": current_artifact["done_sha256"],
            "summary_sha256": current_artifact["summary_sha256"],
            "folds_sha256": payload_files[r10.FOLDS_NAME]["sha256"],
            "development_fold_assignment_sha256":
                summary["fold_protocol"][
                    "development_fold_assignment_sha256"
                ],
            "frozen_transform_sha256":
                payload_files[r10.TRANSFORM_NAME]["sha256"],
            "frozen_transform_array_records_sha256": object_digest(
                summary["frozen_transform"]["array_records"]
            ),
        }
    return identities


def aggregate_contract(seed_validation):
    validated = validate_published_aggregate(aggregate)
    summary = validated["summary"]
    done = validated["done"]
    comparability = summary["input_comparability"]
    decision = summary["decision"]
    if (
        comparability.get("source_tree_sha256") != source_tree_sha256
        or comparability.get("external_source_tree_anchor_sha256")
        != source_tree_sha256
        or comparability.get("external_source_tree_anchor_verified")
        is not True
        or any(
            decision.get(field) is not False
            for field in (
                "representation_gate_passed",
                "renderer_probe_authorized",
                "editor_training_authorized",
            )
        )
    ):
        raise SystemExit("aggregate provenance or gate contract differs")
    seed_identities = current_seed_identities(seed_validation)
    if summary.get("inputs") != seed_identities:
        raise SystemExit("aggregate seed identities differ")
    identity = {
        "root": validated["root"],
        "artifact_digest": done["artifact_digest"],
        "done_sha256": digest_file(aggregate / "done.json"),
        "summary_sha256": digest_file(aggregate / "summary.json"),
        "decision_status": decision["status"],
    }
    return validated, identity, seed_identities


def expected_final_payload():
    seed_validation = load_seed_validation()
    _validated, aggregate_identity, seed_identities = aggregate_contract(
        seed_validation
    )
    return {
        "schema_version": "motive-r10a-final-validation-v3",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_tree_sha256": source_tree_sha256,
        "seed_validation": {
            "sha256": digest_file(seed_receipt),
            "bytes": seed_receipt.stat().st_size,
        },
        "aggregate": aggregate_identity,
        "seed_jobs": seed_validation["jobs"],
        "seed_artifacts": seed_identities,
        "both_seed_artifacts_validated": True,
        "cross_seed_aggregate_validated": True,
        "representation_gate_passed": False,
        "renderer_probe_authorized": False,
        "editor_training_authorized": False,
    }


def validate_final():
    require_sealed_file(final_receipt, "final validation receipt")
    with final_receipt.open(encoding="utf-8") as handle:
        observed = json.load(handle)
    expected = expected_final_payload()
    expected["validated_at_utc"] = observed.get("validated_at_utc")
    if (
        not isinstance(observed.get("validated_at_utc"), str)
        or not observed["validated_at_utc"]
        or observed != expected
    ):
        raise SystemExit("final validation receipt contract differs")


if final_receipt.exists() or final_receipt.is_symlink():
    validate_final()
    print("[r10a-watch] existing final validation strictly revalidated")
    raise SystemExit(0)
if mode != "finalize":
    raise SystemExit(44)

payload = expected_final_payload()
payload_bytes = (
    json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    + "\n"
).encode("utf-8")
temporary = final_receipt.with_name(
    f".{final_receipt.name}.{os.getpid()}.tmp"
)
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
descriptor = os.open(temporary, flags, 0o600)
try:
    with os.fdopen(descriptor, "wb", closefd=False) as handle:
        handle.write(payload_bytes)
        handle.flush()
    os.fsync(descriptor)
finally:
    os.close(descriptor)
os.chmod(temporary, 0o444)
try:
    os.link(temporary, final_receipt, follow_symlinks=False)
except FileExistsError:
    pass
finally:
    temporary.unlink()
directory_flags = os.O_RDONLY
if hasattr(os, "O_DIRECTORY"):
    directory_flags |= os.O_DIRECTORY
directory = os.open(root, directory_flags)
try:
    os.fsync(directory)
finally:
    os.close(directory)
validate_final()
print("[r10a-watch] aggregate-bound final validation published")
PY
REMOTE
}

reconcile_and_aggregate() {
  invoke_remote_submit || return $?
  remote_finalization finalize
}

set_job_ids "${job_ids_csv}"
python_observed=0

for ((poll = 1; poll <= max_polls; poll++)); do
  echo \
    "[r10a-watch] poll=${poll}/${max_polls} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if ! probe_auh; then
    echo "[r10a-watch] AUH unavailable; retaining watcher state" >&2
    sleep "${poll_seconds}"
    continue
  fi

  set +e
  remote_finalization validate
  final_status=$?
  set -e
  case "${final_status}" in
    0)
      echo "[r10a-watch] lifecycle already complete and strictly validated"
      exit 0
      ;;
    44) ;;
    255)
      echo "[r10a-watch] final validation probe lost connectivity" >&2
      sleep "${poll_seconds}"
      continue
      ;;
    *)
      echo \
        "[r10a-watch] existing final validation is invalid " \
        "(status=${final_status})" >&2
      exit 8
      ;;
  esac

  set +e
  accounting="$(
    ssh "${ssh_host}" \
      "sacct -j '${job_ids_csv}' -X -n -P -o JobIDRaw,State,ExitCode"
  )"
  accounting_status=$?
  set -e
  if ((accounting_status == 255)); then
    echo \
      "[r10a-watch] accounting read lost connectivity; retaining state" \
      >&2
    sleep "${poll_seconds}"
    continue
  fi
  if ((accounting_status != 0)); then
    echo \
      "[r10a-watch] accounting read failed deterministically " \
      "(status=${accounting_status})" >&2
    exit 8
  fi
  printf '%s\n' "${accounting}"
  all_terminal=1
  fatal_failure=0
  infrastructure_failure=0
  observed=0
  first_observed=0
  second_observed=0
  running_job=
  while IFS='|' read -r job_id state exit_code; do
    [[ -n "${job_id}" ]] || continue
    if [[ "${job_id}" == "${job_ids[0]}" ]]; then
      if ((first_observed == 1)); then
        continue
      fi
      first_observed=1
    elif [[ "${job_id}" == "${job_ids[1]}" ]]; then
      if ((second_observed == 1)); then
        continue
      fi
      second_observed=1
    else
      continue
    fi
    state="${state%%+*}"
    state="${state%% *}"
    observed=$((observed + 1))
    case "${state}" in
      COMPLETED)
        if [[ "${exit_code}" != "0:0" ]]; then
          fatal_failure=1
        fi
        ;;
      NODE_FAIL|PREEMPTED|BOOT_FAIL|REVOKED)
        infrastructure_failure=1
        ;;
      FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|DEADLINE)
        fatal_failure=1
        ;;
      PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|RESIZING)
        all_terminal=0
        if [[ "${state}" == RUNNING || "${state}" == COMPLETING ]]; then
          [[ -n "${running_job}" ]] || running_job="${job_id}"
        fi
        ;;
      *)
        all_terminal=0
        ;;
    esac
  done < <(printf '%s\n' "${accounting}")
  if ((observed != 2)); then
    all_terminal=0
  fi

  if ((python_observed == 0)) && [[ -n "${running_job}" ]]; then
    process_output="$(
      ssh "${ssh_host}" \
        "timeout 30s srun --overlap --jobid='${running_job}' -N1 -n1 --cpus-per-task=1 bash -lc 'ps -eo pid,ppid,etime,args | grep -E \"[p]ython.*(r10_dynamic_dino_representation_search|instruction_model_registry|action_source_snapshot)\"'" \
        2>&1 || true
    )"
    printf '%s\n' "${process_output}"
    if [[ "${process_output}" == *"python"* ]]; then
      python_observed=1
      echo "[r10a-watch] live Python process verified"
    fi
  fi

  if ((fatal_failure == 1)); then
    echo \
      "[r10a-watch] fatal terminal failure; no resubmission performed" \
      >&2
    exit 6
  fi
  if ((infrastructure_failure == 1)); then
    previous_job_ids="${job_ids_csv}"
    set +e
    invoke_remote_submit
    submit_status=$?
    set -e
    if ((submit_status == 255 || submit_status == 4)); then
      echo \
        "[r10a-watch] infrastructure submit connectivity/lock race; " \
        "retaining watcher state" >&2
      sleep "${poll_seconds}"
      continue
    fi
    if ((submit_status != 0)); then
      echo \
        "[r10a-watch] infrastructure replacement failed " \
        "(status=${submit_status})" >&2
      exit "${submit_status}"
    fi
    set +e
    replacement_job_ids="$(read_remote_job_ids)"
    receipt_status=$?
    set -e
    if ((receipt_status == 255)); then
      echo \
        "[r10a-watch] replacement receipt read lost connectivity; " \
        "retaining watcher state" >&2
      sleep "${poll_seconds}"
      continue
    fi
    if ((receipt_status != 0)); then
      echo \
        "[r10a-watch] replacement receipt is invalid " \
        "(status=${receipt_status})" >&2
      exit 7
    fi
    set_job_ids "${replacement_job_ids}"
    if [[ "${job_ids_csv}" == "${previous_job_ids}" ]]; then
      echo \
        "[r10a-watch] infrastructure reconciliation retained active IDs; " \
        "continuing" >&2
      sleep "${poll_seconds}"
      continue
    fi
    python_observed=0
    echo \
      "[r10a-watch] infrastructure replacement coordinated " \
      "old=${previous_job_ids} new=${job_ids_csv}"
    continue
  fi
  if ((all_terminal == 1)); then
    set +e
    reconcile_and_aggregate
    reconcile_status=$?
    set -e
    if ((reconcile_status == 255 || reconcile_status == 4)); then
      echo \
        "[r10a-watch] terminal reconciliation connectivity/lock race; " \
        "retaining watcher state" >&2
      sleep "${poll_seconds}"
      continue
    fi
    if ((reconcile_status != 0)); then
      echo \
        "[r10a-watch] terminal reconciliation failed " \
        "(status=${reconcile_status})" >&2
      exit "${reconcile_status}"
    fi
    echo \
      "[r10a-watch] jobs, seed validation, immutable aggregate, and final " \
      "receipt verified"
    exit 0
  fi
  sleep "${poll_seconds}"
done

echo "[r10a-watch] lifecycle deadline exhausted before terminal state" >&2
exit 75
