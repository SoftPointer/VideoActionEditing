#!/usr/bin/env bash
set -Eeuo pipefail

experiment_root="${1:?experiment root}"
source_tree_sha256="${2:?source tree SHA-256}"
parent_run="${3:?parent R7 run}"
model_workspace="${4:?model workspace}"
python_bin="${5:?Python executable}"

allowed_prefix=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto
case "${experiment_root}" in
  "${allowed_prefix}"/goku_repr_auto_r10a_*) ;;
  *)
    echo "[r10a-submit] unsafe experiment root: ${experiment_root}" >&2
    exit 2
    ;;
esac
if [[ ! "${source_tree_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "[r10a-submit] invalid source tree SHA-256" >&2
  exit 2
fi

source_snapshot="${experiment_root}/source_snapshot"
controller="${source_snapshot}/methods/motive/scripts/auh_r10a_representation_controller.sh"
registry="${source_snapshot}/methods/motive/configs/instruction_video_editor_registry_v1.json"
snapshot_verifier="${source_snapshot}/methods/motive/scripts/action_source_snapshot.py"
path_contract_source="${source_snapshot}/methods/motive/motive/r10_path_contract.py"
receipt="${experiment_root}/submission.json"
if [[ ! -x "${python_bin}" ]]; then
  echo "[r10a-submit] Python is not executable: ${python_bin}" >&2
  exit 2
fi
if [[ ! -d "${source_snapshot}" ]] || [[ -L "${source_snapshot}" ]]; then
  echo "[r10a-submit] source snapshot root is invalid" >&2
  exit 2
fi
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
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"
export PYTHONPATH="${source_snapshot}/methods/motive"
export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" - "${experiment_root}" "${source_snapshot}" <<'PY'
import sys

from motive.r10_path_contract import (
    canonical_experiment_root,
    ensure_experiment_directory,
    require_experiment_path,
)

raw_root, raw_snapshot = sys.argv[1:]
root = canonical_experiment_root(raw_root)
require_experiment_path(
    raw_snapshot,
    root,
    "source_snapshot",
    kind="dir",
)
for relative in (
    "logs",
    "provenance",
    "provenance/job_attempts",
    "provenance/job_attempts/seed_260108837",
    "provenance/job_attempts/seed_260108838",
    "seed_260108837",
    "seed_260108837/logs",
    "seed_260108838",
    "seed_260108838/logs",
):
    ensure_experiment_directory(root / relative, root, relative)
require_experiment_path(
    root / ".submission.lock",
    root,
    ".submission.lock",
    kind="file",
    allow_missing=True,
)
PY
exec 9>"${experiment_root}/.submission.lock"
if ! flock -n 9; then
  echo "[r10a-submit] another submitter owns this experiment" >&2
  exit 4
fi
# submission.json is a current-state summary, not a success sentinel.  Every
# invocation reconciles Slurm state and validates completed artifacts.
for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${source_snapshot}/SOURCE_PROVENANCE.json" \
  "${controller}" \
  "${registry}"; do
  if [[ ! -s "${required}" ]] || [[ -L "${required}" ]]; then
    echo "[r10a-submit] source snapshot is incomplete: ${required}" >&2
    exit 2
  fi
done
candidate_done_sha256=200bc59872547fb7029aa53ea422a86986f9df8fa614e487b81ae7dce1644c07
track_cache_done_sha256=3d64b89bd17b6880239d58b78b5718c50ddfa0b67ed8111b73cf147a0399ede2
visual_features_done_sha256=f92082f58e25a40d636633abb1669c950044b37dd3b8abca207ee07356125b71
visual_candidates_sha256=2026b87cc01107c77b6a1cc993bfb6eb64e0381a9b52f8a87ac59ac661db628d
screen_inputs_receipt_sha256=fad2f1d5c6ea30b6f0a33f150a256c22fbd57938d44ae88a5014f9ddf926c6fb

candidate_manifest_dir="${parent_run}/expansion/candidate_temporal_screen_v1"
track_cache_final="${parent_run}/expansion/candidate_track_cache_v1/final"
screen_input_bundle="${parent_run}/expansion/screen_inputs_v14"
visual_features_final="${screen_input_bundle}/visual_features_v1/final"
visual_candidates_manifest="${screen_input_bundle}/visual_candidates_v1/candidates.jsonl"
screen_inputs_receipt="${parent_run}/provenance/screen_inputs_v14.receipt.txt"
for required_dir in \
  "${candidate_manifest_dir}" \
  "${track_cache_final}" \
  "${screen_input_bundle}" \
  "${visual_features_final}" \
  "$(dirname "${visual_candidates_manifest}")" \
  "${model_workspace}"; do
  if [[ ! -d "${required_dir}" ]] || [[ -L "${required_dir}" ]]; then
    echo "[r10a-submit] preflight directory differs: ${required_dir}" >&2
    exit 2
  fi
done
for anchored in \
  "${candidate_manifest_dir}/done.json|${candidate_done_sha256}" \
  "${track_cache_final}/done.json|${track_cache_done_sha256}" \
  "${visual_features_final}/done.json|${visual_features_done_sha256}" \
  "${visual_candidates_manifest}|${visual_candidates_sha256}" \
  "${screen_inputs_receipt}|${screen_inputs_receipt_sha256}"; do
  anchored_path="${anchored%%|*}"
  anchored_sha256="${anchored#*|}"
  if [[ ! -f "${anchored_path}" ]] \
    || [[ -L "${anchored_path}" ]] \
    || [[ "$(sha256sum "${anchored_path}" | awk '{print $1}')" \
      != "${anchored_sha256}" ]]; then
    echo "[r10a-submit] preflight input differs: ${anchored_path}" >&2
    exit 2
  fi
done
if [[ "$(stat -c '%h' "${screen_inputs_receipt}")" != "1" ]] \
  || [[ "$(stat -c '%a' "${screen_inputs_receipt}")" != "444" ]]; then
  echo "[r10a-submit] sealed input receipt mode differs" >&2
  exit 2
fi
PYTHONPATH="${source_snapshot}/methods/motive" \
PYTHONDONTWRITEBYTECODE=1 \
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

for test_file in \
  test_r10_dynamic_dino_representation_search.py \
  test_r10_cross_seed_aggregate.py \
  test_r10_representation_orchestration.py \
  test_r10_submission_scripts.py \
  test_instruction_model_registry.py; do
  PYTHONPATH="${source_snapshot}/methods/motive" \
  PYTHONDONTWRITEBYTECODE=1 \
  "${python_bin}" \
    "${source_snapshot}/methods/motive/tests/${test_file}"
done

experiment_token="$(
  "${python_bin}" -c '
import hashlib
import sys
print(hashlib.sha256(sys.argv[1].encode("utf-8")).hexdigest())
' "${experiment_root}"
)"
attempt_receipt_root="${experiment_root}/provenance/job_attempts"

job_name_for() {
  local name_seed="${1:?seed}"
  local name_attempt="${2:?attempt}"
  printf "m10a-%s-s%s-a%s\n" \
    "${experiment_token}" "${name_seed}" "${name_attempt}"
}

attempt_receipt_for() {
  local receipt_seed="${1:?seed}"
  local receipt_attempt="${2:?attempt}"
  printf "%s/seed_%s/attempt_%s.json\n" \
    "${attempt_receipt_root}" "${receipt_seed}" "${receipt_attempt}"
}

append_attempt_event() {
  local log_path="${1:?attempt log}"
  local event_seed="${2:?seed}"
  local event_attempt="${3:?attempt}"
  local event="${4:?event}"
  local event_job_id="${5:-}"
  local event_job_name="${6:?job name}"
  local event_run_root="${7:?run root}"
  local state="${8:-}"
  local exit_code="${9:-}"
  "${python_bin}" -c '
import json
import os
import sys
from datetime import datetime, timezone

from motive.r10_path_contract import require_experiment_path

path = require_experiment_path(
    sys.argv[1],
    sys.argv[2],
    f"provenance/seed_{int(sys.argv[3])}.attempts.jsonl",
    kind="file",
    allow_missing=True,
)
payload = {
    "schema_version": "motive-r10a-job-attempt-event-v1",
    "utc": datetime.now(timezone.utc).isoformat(),
    "seed": int(sys.argv[3]),
    "attempt": int(sys.argv[4]),
    "event": sys.argv[5],
    "job_id": int(sys.argv[6]) if sys.argv[6] else None,
    "job_name": sys.argv[7],
    "run_root": sys.argv[8],
    "state": sys.argv[9] or None,
    "exit_code": sys.argv[10] or None,
    "source_tree_sha256": sys.argv[11],
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
' \
    "${log_path}" \
    "${experiment_root}" \
    "${event_seed}" \
    "${event_attempt}" \
    "${event}" \
    "${event_job_id}" \
    "${event_job_name}" \
    "${event_run_root}" \
    "${state}" \
    "${exit_code}" \
    "${source_tree_sha256}"
}

attempt_log_state() {
  local log_path="${1:?attempt log}"
  local log_seed="${2:?seed}"
  "${python_bin}" - "${log_path}" "${experiment_root}" "${log_seed}" \
    "${source_tree_sha256}" <<'PY'
import json
import os
import sys

from motive.r10_path_contract import require_experiment_path

raw_path, experiment_root, raw_seed, source_tree_sha256 = sys.argv[1:]
seed = int(raw_seed)
path = require_experiment_path(
    raw_path,
    experiment_root,
    f"provenance/seed_{seed}.attempts.jsonl",
    kind="file",
    allow_missing=True,
)
maximum = 0
events = []
if os.path.exists(path):
    if path.is_symlink() or not path.is_file():
        raise SystemExit("attempt log must be one regular file")
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (
                row.get("schema_version")
                != "motive-r10a-job-attempt-event-v1"
                or row.get("seed") != seed
                or row.get("source_tree_sha256") != source_tree_sha256
                or isinstance(row.get("attempt"), bool)
                or not isinstance(row.get("attempt"), int)
                or row["attempt"] < 1
                or not isinstance(row.get("event"), str)
                or not row["event"]
            ):
                raise SystemExit("attempt log contract differs")
            maximum = max(maximum, row["attempt"])
            events.append(row)
maximum_events = [
    row["event"] for row in events if row["attempt"] == maximum
]
print(
    f"{maximum}|"
    f"{str('INTENT' in maximum_events).lower()}|"
    f"{str(any(event in ('SUBMITTED', 'RECOVERED') for event in maximum_events)).lower()}"
)
PY
}

validate_attempt_receipt() {
  local attempt_receipt="${1:?attempt receipt}"
  local receipt_seed="${2:?seed}"
  local receipt_attempt="${3:?attempt}"
  local expected_name="${4:?job name}"
  local expected_run_root="${5:?run root}"
  "${python_bin}" - "${attempt_receipt}" "${experiment_root}" \
    "${source_tree_sha256}" "${receipt_seed}" "${receipt_attempt}" \
    "${expected_name}" "${expected_run_root}" <<'PY'
import json
from pathlib import Path
import stat
import sys

from motive.r10_path_contract import require_attempt_receipt_path

(
    raw_receipt,
    experiment_root,
    source_tree_sha256,
    raw_seed,
    raw_attempt,
    expected_name,
    run_root,
) = sys.argv[1:]
receipt = require_attempt_receipt_path(
    raw_receipt,
    experiment_root,
    int(raw_seed),
    int(raw_attempt),
)
if receipt.is_symlink() or not receipt.is_file() or receipt.stat().st_size < 1:
    raise SystemExit("attempt receipt is invalid")
metadata = receipt.stat()
if metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o444:
    raise SystemExit("attempt receipt permission contract differs")
with receipt.open(encoding="utf-8") as handle:
    row = json.load(handle)
job_id = row.get("job_id")
expected = {
    "schema_version": "motive-r10a-job-attempt-receipt-v1",
    "submitted_at_utc": row.get("submitted_at_utc"),
    "experiment_root": experiment_root,
    "source_tree_sha256": source_tree_sha256,
    "seed": int(raw_seed),
    "attempt": int(raw_attempt),
    "job_id": job_id,
    "job_name": expected_name,
    "run_root": run_root,
}
if (
    set(row) != set(expected)
    or not isinstance(row.get("submitted_at_utc"), str)
    or not row["submitted_at_utc"]
    or isinstance(job_id, bool)
    or not isinstance(job_id, int)
    or job_id < 1
    or row != expected
):
    raise SystemExit("attempt receipt identity differs")
print(job_id)
PY
}

publish_attempt_receipt() {
  local attempt_receipt="${1:?attempt receipt}"
  local receipt_seed="${2:?seed}"
  local receipt_attempt="${3:?attempt}"
  local receipt_job_id="${4:?job id}"
  local receipt_job_name="${5:?job name}"
  local receipt_run_root="${6:?run root}"
  "${python_bin}" - "${attempt_receipt}" "${experiment_root}" \
    "${source_tree_sha256}" "${receipt_seed}" "${receipt_attempt}" \
    "${receipt_job_id}" "${receipt_job_name}" "${receipt_run_root}" <<'PY'
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

from motive.r10_path_contract import require_attempt_receipt_path

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
receipt = require_attempt_receipt_path(
    raw_receipt,
    experiment_root,
    int(raw_seed),
    int(raw_attempt),
    allow_missing=True,
)
if receipt.exists() or receipt.is_symlink():
    raise SystemExit("immutable attempt receipt already exists")
payload = {
    "schema_version": "motive-r10a-job-attempt-receipt-v1",
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "experiment_root": experiment_root,
    "source_tree_sha256": source_tree_sha256,
    "seed": int(raw_seed),
    "attempt": int(raw_attempt),
    "job_id": int(raw_job_id),
    "job_name": job_name,
    "run_root": run_root,
}
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
temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
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
    os.link(temporary, receipt, follow_symlinks=False)
finally:
    temporary.unlink()
directory_flags = os.O_RDONLY
if hasattr(os, "O_DIRECTORY"):
    directory_flags |= os.O_DIRECTORY
directory = os.open(receipt.parent, directory_flags)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
}

recover_exact_job_id() {
  local recovery_name="${1:?job name}"
  local queue_rows
  local accounting_rows
  queue_rows="$(squeue -h -n "${recovery_name}" -o "%i|%.128j")" \
    || return 8
  accounting_rows="$(
    sacct -X -n -P --name="${recovery_name}" \
      -o JobIDRaw,JobName%128
  )" || return 8
  RECOVERY_NAME="${recovery_name}" \
  QUEUE_ROWS="${queue_rows}" \
  ACCOUNTING_ROWS="${accounting_rows}" \
    "${python_bin}" -c '
import os
name = os.environ["RECOVERY_NAME"]
job_ids = set()
for block in (os.environ["QUEUE_ROWS"], os.environ["ACCOUNTING_ROWS"]):
    for line in block.splitlines():
        fields = line.split("|")
        job_id = fields[0].strip() if fields else ""
        observed_name = fields[1].strip() if len(fields) >= 2 else ""
        if job_id.isdigit() and observed_name == name:
            job_ids.add(int(job_id))
if len(job_ids) != 1:
    raise SystemExit(
        f"exact-name recovery expected one job, observed {len(job_ids)}"
    )
print(next(iter(job_ids)))
'
}

job_observation() {
  local observed_job_id="${1:?job id}"
  local active
  active="$(
    squeue -h -j "${observed_job_id}" -o "%i|%T|%.128j|%r" \
      | awk -F'|' -v expected="${observed_job_id}" \
        '{
          job = $1
          state = $2
          name = $3
          reason = $4
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", job)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", state)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", reason)
          if (job == expected) {
            print state "||" name "|" reason
            exit
          }
        }'
  )"
  if [[ -n "${active}" ]]; then
    printf "%s\n" "${active}"
    return 0
  fi
  local accounting
  accounting="$(
    sacct -j "${observed_job_id}" -X -n -P \
      -o JobIDRaw,State,ExitCode,JobName%128 \
      | awk -F'|' -v expected="${observed_job_id}" \
        '{
          job = $1
          state = $2
          code = $3
          name = $4
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", job)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", state)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", code)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
          if (job == expected) {
            print state "|" code "|" name "|"
            exit
          }
        }'
  )"
  if [[ -n "${accounting}" ]]; then
    printf "%s\n" "${accounting}"
  else
    printf "UNKNOWN|||\n"
  fi
}

seed_artifact_identity() {
  local identity_seed="${1:?seed}"
  local identity_root="${2:?run root}"
  local search_output
  local producer_receipt
  search_output="${identity_root}/representation/search_seed_${identity_seed}"
  producer_receipt="${identity_root}/provenance/search_seed_${identity_seed}.producer.json"
  PYTHONPATH="${source_snapshot}/methods/motive" \
  PYTHONDONTWRITEBYTECODE=1 \
  "${python_bin}" - "${search_output}" "${producer_receipt}" \
    "${experiment_root}" "${source_tree_sha256}" "${identity_seed}" \
    "${identity_root}" "${experiment_token}" <<'PY'
import hashlib
import json
from pathlib import Path
import stat
import sys

from motive import r10_dynamic_dino_representation_search as r10
from motive.r10_path_contract import (
    canonical_experiment_root,
    require_attempt_receipt_path,
    require_experiment_path,
)

(
    raw_artifact,
    raw_producer_receipt,
    experiment_root,
    source_tree_sha256,
    raw_seed,
    raw_run_root,
    experiment_token,
) = sys.argv[1:]
seed = int(raw_seed)
root = canonical_experiment_root(experiment_root)
run_root = require_experiment_path(
    raw_run_root,
    root,
    f"seed_{seed}",
    kind="dir",
)
artifact = require_experiment_path(
    raw_artifact,
    root,
    f"seed_{seed}/representation/search_seed_{seed}",
    kind="dir",
)
producer_receipt = require_experiment_path(
    raw_producer_receipt,
    root,
    f"seed_{seed}/provenance/search_seed_{seed}.producer.json",
    kind="file",
)


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed_identity(path, context):
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
        raise SystemExit(f"{context} is invalid")
    metadata = path.stat()
    if metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o444:
        raise SystemExit(f"{context} permission contract differs")
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": digest_file(path),
        "bytes": metadata.st_size,
    }


validated = r10.validate_published_search(artifact)
summary = validated["summary"]
source = summary.get("source_snapshot")
if (
    summary.get("seed") != seed
    or not isinstance(source, dict)
    or source.get("tree_sha256") != source_tree_sha256
    or source.get("exact_tree_verified_by_controller_before_search") is not True
):
    raise SystemExit("seed artifact provenance differs")
artifact_identity = {
    "root": str(artifact),
    "artifact_digest": validated["done"]["artifact_digest"],
    "done_sha256": digest_file(artifact / r10.DONE_NAME),
    "summary_sha256": digest_file(artifact / r10.SUMMARY_NAME),
}
producer_receipt_identity = sealed_identity(
    producer_receipt,
    "producer receipt",
)
with producer_receipt.open(encoding="utf-8") as handle:
    producer_row = json.load(handle)
producer = producer_row.get("producer_job")
if not isinstance(producer, dict):
    raise SystemExit("producer job identity is absent")
producer_attempt_identity = producer.get("attempt_receipt")
if not isinstance(producer_attempt_identity, dict):
    raise SystemExit("producer attempt receipt identity is absent")
producer_attempt_path = Path(producer_attempt_identity.get("path", ""))
producer_attempt_path = require_attempt_receipt_path(
    producer_attempt_path,
    root,
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
    attempt_row = json.load(handle)
producer_job_id = producer.get("job_id")
producer_attempt = producer.get("attempt")
producer_job_name = producer.get("job_name")
expected_job_name = (
    f"m10a-{experiment_token}-s{seed}-a{producer_attempt}"
)
expected_attempt = {
    "schema_version": "motive-r10a-job-attempt-receipt-v1",
    "submitted_at_utc": attempt_row.get("submitted_at_utc"),
    "experiment_root": experiment_root,
    "source_tree_sha256": source_tree_sha256,
    "seed": seed,
    "attempt": producer_attempt,
    "job_id": producer_job_id,
    "job_name": producer_job_name,
    "run_root": str(run_root),
}
expected_producer = {
    "schema_version": "motive-r10a-artifact-producer-v1",
    "produced_at_utc": producer_row.get("produced_at_utc"),
    "source_tree_sha256": source_tree_sha256,
    "seed": seed,
    "artifact": artifact_identity,
    "producer_job": {
        "job_id": producer_job_id,
        "attempt": producer_attempt,
        "job_name": producer_job_name,
        "attempt_receipt": actual_attempt_identity,
    },
    "representation_gate_passed": False,
    "renderer_probe_authorized": False,
    "editor_training_authorized": False,
}
if (
    set(producer_row) != set(expected_producer)
    or not isinstance(producer_row.get("produced_at_utc"), str)
    or not producer_row["produced_at_utc"]
    or isinstance(producer_job_id, bool)
    or not isinstance(producer_job_id, int)
    or producer_job_id < 1
    or isinstance(producer_attempt, bool)
    or not isinstance(producer_attempt, int)
    or producer_attempt < 1
    or producer_job_name != expected_job_name
    or set(attempt_row) != set(expected_attempt)
    or not isinstance(attempt_row.get("submitted_at_utc"), str)
    or not attempt_row["submitted_at_utc"]
    or attempt_row != expected_attempt
    or producer_row != expected_producer
):
    raise SystemExit("artifact producer identity differs")
print(
    json.dumps(
        {
            "artifact": artifact_identity,
            "artifact_producer": {
                "job_id": producer_job_id,
                "attempt": producer_attempt,
                "job_name": producer_job_name,
                "attempt_receipt": actual_attempt_identity,
                "producer_receipt": producer_receipt_identity,
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY
}

validate_seed_receipt() {
  local validation_receipt="${1:?seed receipt}"
  local first_identity="${2:?first seed identity}"
  local second_identity="${3:?second seed identity}"
  "${python_bin}" - "${validation_receipt}" "${experiment_root}" \
    "${source_tree_sha256}" "${experiment_token}" \
    "${first_identity}" "${second_identity}" <<'PY'
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

(
    raw_receipt,
    experiment_root,
    source_tree_sha256,
    experiment_token,
    first_identity,
    second_identity,
) = sys.argv[1:]
root = canonical_experiment_root(experiment_root)
receipt = require_experiment_path(
    raw_receipt,
    root,
    "seed_validation.json",
    kind="file",
)


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed_identity(path, context):
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
        raise SystemExit(f"{context} is invalid")
    metadata = path.stat()
    if metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o444:
        raise SystemExit(f"{context} permission contract differs")
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": digest_file(path),
        "bytes": metadata.st_size,
    }


sealed_identity(receipt, "seed validation receipt")
with receipt.open(encoding="utf-8") as handle:
    row = json.load(handle)
identities = {
    260108837: json.loads(first_identity),
    260108838: json.loads(second_identity),
}
jobs = row.get("jobs")
if not isinstance(jobs, list) or len(jobs) != 2:
    raise SystemExit("seed validation jobs differ")
expected_jobs = []
for seed, job in zip((260108837, 260108838), jobs, strict=True):
    if not isinstance(job, dict) or job.get("seed") != seed:
        raise SystemExit("seed validation ordering differs")
    validation = job.get("validation_job")
    if not isinstance(validation, dict):
        raise SystemExit("validation job identity is absent")
    attempt_identity = validation.get("attempt_receipt")
    if not isinstance(attempt_identity, dict):
        raise SystemExit("validation attempt receipt identity is absent")
    job_id = validation.get("job_id")
    attempt = validation.get("attempt")
    job_name = validation.get("job_name")
    attempt_path = Path(attempt_identity.get("path", ""))
    attempt_path = require_attempt_receipt_path(
        attempt_path,
        root,
        seed,
        attempt,
    )
    actual_attempt_identity = sealed_identity(
        attempt_path,
        "validation attempt receipt",
    )
    if actual_attempt_identity != attempt_identity:
        raise SystemExit("validation attempt receipt digest differs")
    with attempt_path.open(encoding="utf-8") as handle:
        attempt_row = json.load(handle)
    expected_name = f"m10a-{experiment_token}-s{seed}-a{attempt}"
    expected_attempt = {
        "schema_version": "motive-r10a-job-attempt-receipt-v1",
        "submitted_at_utc": attempt_row.get("submitted_at_utc"),
        "experiment_root": experiment_root,
        "source_tree_sha256": source_tree_sha256,
        "seed": seed,
        "attempt": attempt,
        "job_id": job_id,
        "job_name": job_name,
        "run_root": f"{experiment_root}/seed_{seed}",
    }
    if (
        isinstance(job_id, bool)
        or not isinstance(job_id, int)
        or job_id < 1
        or isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt < 1
        or job_name != expected_name
        or validation.get("exit_code") != "0:0"
        or set(attempt_row) != set(expected_attempt)
        or not isinstance(attempt_row.get("submitted_at_utc"), str)
        or not attempt_row["submitted_at_utc"]
        or attempt_row != expected_attempt
    ):
        raise SystemExit("validation job identity differs")
    identity = identities[seed]
    expected_jobs.append(
        {
            "seed": seed,
            "validation_job": {
                "job_id": job_id,
                "attempt": attempt,
                "job_name": job_name,
                "exit_code": "0:0",
                "attempt_receipt": actual_attempt_identity,
            },
            "artifact": identity["artifact"],
            "artifact_producer": identity["artifact_producer"],
        }
    )
expected = {
    "schema_version": "motive-r10a-two-seed-validation-v2",
    "validated_at_utc": row.get("validated_at_utc"),
    "experiment_root": experiment_root,
    "source_tree_sha256": source_tree_sha256,
    "jobs": expected_jobs,
    "both_seed_artifacts_validated": True,
    "representation_gate_passed": False,
    "renderer_probe_authorized": False,
    "editor_training_authorized": False,
}
if (
    set(row) != set(expected)
    or not isinstance(row.get("validated_at_utc"), str)
    or not row["validated_at_utc"]
    or row != expected
):
    raise SystemExit("seed validation receipt contract differs")
PY
}

seed_receipt="${experiment_root}/seed_validation.json"
if [[ -e "${seed_receipt}" ]] || [[ -L "${seed_receipt}" ]]; then
  first_identity="$(
    seed_artifact_identity 260108837 \
      "${experiment_root}/seed_260108837"
  )"
  second_identity="$(
    seed_artifact_identity 260108838 \
      "${experiment_root}/seed_260108838"
  )"
  validate_seed_receipt \
    "${seed_receipt}" "${first_identity}" "${second_identity}"
  echo "[r10a-submit] existing seed receipt and both artifacts revalidated"
  exit 0
fi

job_ids=()
job_names=()
job_states=()
job_exit_codes=()
job_actions=()
job_seed_validated=()
job_attempt_receipts=()
seed_identities=()
run_roots=()
attempt_numbers=()
seeds=(260108837 260108838)
for seed in "${seeds[@]}"; do
  run_root="${experiment_root}/seed_${seed}"
  attempts_log="${experiment_root}/provenance/seed_${seed}.attempts.jsonl"
  legacy_job_receipt="${experiment_root}/provenance/seed_${seed}.job_id"
  if [[ -e "${legacy_job_receipt}" ]] || [[ -L "${legacy_job_receipt}" ]]; then
    echo \
      "[r10a-submit] legacy mutable job receipt requires diagnosis: " \
      "${legacy_job_receipt}" >&2
    exit 3
  fi
  state_row="$(attempt_log_state "${attempts_log}" "${seed}")"
  IFS='|' read -r attempt has_intent has_submission <<< "${state_row}"
  job_id=""
  state=""
  exit_code=""
  action=""
  seed_validated=false
  seed_identity=""
  job_name=""
  attempt_receipt=""
  if (( attempt > 0 )); then
    job_name="$(job_name_for "${seed}" "${attempt}")"
    attempt_receipt="$(attempt_receipt_for "${seed}" "${attempt}")"
    if [[ -e "${attempt_receipt}" ]] || [[ -L "${attempt_receipt}" ]]; then
      job_id="$(
        validate_attempt_receipt \
          "${attempt_receipt}" "${seed}" "${attempt}" \
          "${job_name}" "${run_root}"
      )"
    else
      if [[ "${has_intent}" != true ]] \
        || [[ "${has_submission}" == true ]]; then
        echo \
          "[r10a-submit] attempt log/receipt closure differs for " \
          "seed=${seed} attempt=${attempt}" >&2
        exit 7
      fi
      set +e
      job_id="$(recover_exact_job_id "${job_name}")"
      recovery_status=$?
      set -e
      if (( recovery_status != 0 )); then
        if (( recovery_status == 8 )); then
          echo \
            "[r10a-submit] exact-name recovery query unavailable; " \
            "retaining uncertain state" >&2
          exit 75
        fi
        echo \
          "[r10a-submit] uncertain submit has zero/multiple exact-name " \
          "matches; no resubmission seed=${seed} attempt=${attempt}" >&2
        exit 7
      fi
      publish_attempt_receipt \
        "${attempt_receipt}" "${seed}" "${attempt}" \
        "${job_id}" "${job_name}" "${run_root}"
      append_attempt_event \
        "${attempts_log}" "${seed}" "${attempt}" "RECOVERED" \
        "${job_id}" "${job_name}" "${run_root}" "PENDING" ""
      scontrol release "${job_id}"
      append_attempt_event \
        "${attempts_log}" "${seed}" "${attempt}" "RELEASED" \
        "${job_id}" "${job_name}" "${run_root}" "PENDING" ""
      state="PENDING"
      action="RECOVERED_UNCERTAIN_SUBMIT"
    fi
  fi
  if [[ -n "${job_id}" ]] && [[ -z "${state}" ]]; then
    active_identity="$(
      squeue -h -j "${job_id}" -o "%i|%.128j|%r" \
        | awk -F'|' -v expected="${job_id}" \
          '{
            job = $1
            name = $2
            reason = $3
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", job)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", reason)
            if (job == expected) {
              print name "|" reason
              exit
            }
          }'
    )"
    if [[ -n "${active_identity}" ]]; then
      active_name="${active_identity%%|*}"
      active_reason="${active_identity#*|}"
      if [[ "${active_name}" != "${job_name}" ]]; then
        echo "[r10a-submit] active Slurm job name differs" >&2
        exit 7
      fi
      if [[ "${active_reason}" == JobHeldUser* ]]; then
        scontrol release "${job_id}"
        append_attempt_event \
          "${attempts_log}" "${seed}" "${attempt}" "RELEASED" \
          "${job_id}" "${job_name}" "${run_root}" "PENDING" ""
      fi
    fi
    observation="$(job_observation "${job_id}")"
    IFS='|' read -r state exit_code observed_name _reason \
      <<< "${observation}"
    state="${state%%+*}"
    state="${state%% *}"
    if [[ "${state}" != UNKNOWN ]] \
      && [[ "${observed_name}" != "${job_name}" ]]; then
      echo "[r10a-submit] persisted job/name identity differs" >&2
      exit 7
    fi
    append_attempt_event \
      "${attempts_log}" "${seed}" "${attempt}" "OBSERVED" \
      "${job_id}" "${job_name}" "${run_root}" "${state}" "${exit_code}"
    case "${state}" in
      PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|RESIZING)
        action="KEEP_ACTIVE"
        ;;
      COMPLETED)
        if [[ "${exit_code}" == "0:0" ]]; then
          set +e
          seed_identity="$(seed_artifact_identity "${seed}" "${run_root}")"
          identity_status=$?
          set -e
        else
          identity_status=1
        fi
        if (( identity_status == 0 )); then
          action="SEED_VALIDATED"
          seed_validated=true
          append_attempt_event \
            "${attempts_log}" "${seed}" "${attempt}" "VALIDATED" \
            "${job_id}" "${job_name}" "${run_root}" \
            "${state}" "${exit_code}"
        else
          append_attempt_event \
            "${attempts_log}" "${seed}" "${attempt}" \
            "TERMINAL_ARTIFACT_INVALID" "${job_id}" "${job_name}" \
            "${run_root}" "${state}" "${exit_code}"
          echo \
            "[r10a-submit] completed job lacks a producer-bound artifact: " \
            "${job_id}" >&2
          exit 6
        fi
        ;;
      NODE_FAIL|PREEMPTED|BOOT_FAIL|REVOKED)
        append_attempt_event \
          "${attempts_log}" "${seed}" "${attempt}" \
          "TERMINAL_INFRASTRUCTURE" "${job_id}" "${job_name}" \
          "${run_root}" "${state}" "${exit_code}"
        job_id=""
        ;;
      CANCELLED|FAILED|OUT_OF_MEMORY|TIMEOUT|DEADLINE)
        append_attempt_event \
          "${attempts_log}" "${seed}" "${attempt}" "TERMINAL_FAILED" \
          "${job_id}" "${job_name}" "${run_root}" "${state}" "${exit_code}"
        echo \
          "[r10a-submit] terminal job requires diagnosis, not blind retry: " \
          "${job_id} ${state} ${exit_code}" >&2
        exit 6
        ;;
      UNKNOWN)
        echo \
          "[r10a-submit] job is absent from squeue and sacct; not resubmitting: " \
          "${job_id}" >&2
        exit 7
        ;;
      *)
        echo "[r10a-submit] unsupported Slurm state: ${state}" >&2
        exit 7
        ;;
    esac
  fi
  if [[ -z "${job_id}" ]]; then
    attempt=$((attempt + 1))
    job_name="$(job_name_for "${seed}" "${attempt}")"
    attempt_receipt="$(attempt_receipt_for "${seed}" "${attempt}")"
    append_attempt_event \
      "${attempts_log}" "${seed}" "${attempt}" "INTENT" \
      "" "${job_name}" "${run_root}" "" ""
    raw_job_id="$(
      sbatch \
        --parsable \
        --hold \
        --job-name="${job_name}" \
        --partition=faculty \
        --account=test-acc \
        --qos=stqos \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --mem=128G \
        --gres=gpu:mi210:1 \
        --time=08:00:00 \
        --output="${experiment_root}/logs/r10a_seed_${seed}_%j.out" \
        --error="${experiment_root}/logs/r10a_seed_${seed}_%j.err" \
        --export="ALL,MOTIVE_SOURCE_SNAPSHOT=${source_snapshot},MOTIVE_SOURCE_TREE_SHA256=${source_tree_sha256},MOTIVE_R10A_EXPERIMENT_ROOT=${experiment_root},MOTIVE_R10A_RUN_ROOT=${run_root},MOTIVE_R7_PARENT_RUN_ROOT=${parent_run},MOTIVE_MODEL_WORKSPACE=${model_workspace},MOTIVE_R10A_SEED=${seed},MOTIVE_R10A_ATTEMPT=${attempt},MOTIVE_R10A_JOB_NAME=${job_name},MOTIVE_R10A_ATTEMPT_RECEIPT=${attempt_receipt},MOTIVE_R7_CANDIDATE_TEMPORAL_DONE_SHA256=${candidate_done_sha256},MOTIVE_R7_CANDIDATE_TRACK_CACHE_DONE_SHA256=${track_cache_done_sha256},MOTIVE_R7_VISUAL_FEATURES_DONE_SHA256=${visual_features_done_sha256},MOTIVE_R7_VISUAL_CANDIDATES_SHA256=${visual_candidates_sha256},MOTIVE_R7_SCREEN_INPUTS_RECEIPT_SHA256=${screen_inputs_receipt_sha256},PYTHON_BIN=${python_bin}" \
        "${controller}"
    )"
    job_id="${raw_job_id%%;*}"
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
      echo \
        "[r10a-submit] uncertain sbatch response; exact-name recovery " \
        "required: ${raw_job_id}" >&2
      exit 75
    fi
    publish_attempt_receipt \
      "${attempt_receipt}" "${seed}" "${attempt}" \
      "${job_id}" "${job_name}" "${run_root}"
    append_attempt_event \
      "${attempts_log}" "${seed}" "${attempt}" "SUBMITTED" \
      "${job_id}" "${job_name}" "${run_root}" "PENDING" ""
    scontrol release "${job_id}"
    append_attempt_event \
      "${attempts_log}" "${seed}" "${attempt}" "RELEASED" \
      "${job_id}" "${job_name}" "${run_root}" "PENDING" ""
    state="PENDING"
    exit_code=""
    action="SUBMITTED"
  fi
  job_ids+=("${job_id}")
  job_names+=("${job_name}")
  job_states+=("${state}")
  job_exit_codes+=("${exit_code}")
  job_actions+=("${action}")
  job_seed_validated+=("${seed_validated}")
  job_attempt_receipts+=("${attempt_receipt}")
  seed_identities+=("${seed_identity}")
  run_roots+=("${run_root}")
  attempt_numbers+=("${attempt}")
done

temporary="${receipt}.tmp.$$"
"${python_bin}" - "${experiment_root}" "${source_tree_sha256}" \
  "${job_ids[0]}" "${attempt_numbers[0]}" "${job_names[0]}" \
  "${job_states[0]}" "${job_exit_codes[0]}" "${job_actions[0]}" \
  "${job_seed_validated[0]}" "${run_roots[0]}" \
  "${job_attempt_receipts[0]}" \
  "${job_ids[1]}" "${attempt_numbers[1]}" "${job_names[1]}" \
  "${job_states[1]}" "${job_exit_codes[1]}" "${job_actions[1]}" \
  "${job_seed_validated[1]}" "${run_roots[1]}" \
  "${job_attempt_receipts[1]}" "${temporary}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from datetime import datetime, timezone

from motive.r10_path_contract import (
    canonical_experiment_root,
    require_attempt_receipt_path,
    require_experiment_path,
)

experiment_root = sys.argv[1]
root = canonical_experiment_root(experiment_root)
source_tree_sha256 = sys.argv[2]
raw = sys.argv[3:-1]
output = Path(sys.argv[-1])
if (
    output.parent != root
    or not output.name.startswith("submission.json.tmp.")
):
    raise SystemExit("submission temporary path differs")
output = require_experiment_path(
    output,
    root,
    output.name,
    kind="file",
    allow_missing=True,
)


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attempt_identity(path, seed, attempt, job_id, job_name, run_root):
    path = require_attempt_receipt_path(
        path,
        root,
        seed,
        attempt,
    )
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
        raise SystemExit("submission attempt receipt is invalid")
    metadata = path.stat()
    if metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o444:
        raise SystemExit("submission attempt receipt permissions differ")
    with path.open(encoding="utf-8") as handle:
        row = json.load(handle)
    expected = {
        "schema_version": "motive-r10a-job-attempt-receipt-v1",
        "submitted_at_utc": row.get("submitted_at_utc"),
        "experiment_root": experiment_root,
        "source_tree_sha256": source_tree_sha256,
        "seed": seed,
        "attempt": attempt,
        "job_id": job_id,
        "job_name": job_name,
        "run_root": run_root,
    }
    if (
        set(row) != set(expected)
        or not isinstance(row.get("submitted_at_utc"), str)
        or not row["submitted_at_utc"]
        or row != expected
    ):
        raise SystemExit("submission attempt receipt identity differs")
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": digest_file(path),
        "bytes": metadata.st_size,
    }


jobs = []
offset = 0
for seed in (260108837, 260108838):
    (
        raw_job_id,
        raw_attempt,
        job_name,
        state,
        exit_code,
        action,
        raw_validated,
        run_root,
        attempt_receipt,
    ) = raw[offset:offset + 9]
    offset += 9
    job_id = int(raw_job_id)
    attempt = int(raw_attempt)
    expected_run_root = require_experiment_path(
        run_root,
        root,
        f"seed_{seed}",
        kind="dir",
    )
    jobs.append(
        {
            "seed": seed,
            "job_id": job_id,
            "attempt": attempt,
            "job_name": job_name,
            "state": state,
            "exit_code": exit_code or None,
            "action": action,
            "seed_artifact_validated": raw_validated == "true",
            "run_root": str(expected_run_root),
            "attempt_receipt": attempt_identity(
                attempt_receipt,
                seed,
                attempt,
                job_id,
                job_name,
                str(expected_run_root),
            ),
        }
    )
payload = {
    "schema_version": "motive-r10a-two-seed-submission-state-v4",
    "observed_at_utc": datetime.now(timezone.utc).isoformat(),
    "experiment_root": experiment_root,
    "source_tree_sha256": source_tree_sha256,
    "jobs": jobs,
    "seed_changes_appearance_group_fold_assignment": True,
    "maximum_concurrent_nodes": 2,
    "cpus_per_job": 16,
    "memory_gib_per_job": 128,
    "gpus_per_job": 1,
    "gpu_compute_expected": False,
    "gpu_allocation_reason": (
        "AUH faculty/stqos enforces MinTRES=gres/gpu=1. R10A is a NumPy "
        "representation diagnostic and requests only that minimum."
    ),
    "videos_copied_to_local_machine": False,
    "renderer_probe_submitted": False,
    "editor_training_submitted": False,
}
with output.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
"${python_bin}" - "${temporary}" "${receipt}" "${experiment_root}" <<'PY'
import os
from pathlib import Path
import sys

from motive.r10_path_contract import (
    canonical_experiment_root,
    require_experiment_path,
)

raw_temporary, raw_destination, raw_root = sys.argv[1:]
root = canonical_experiment_root(raw_root)
temporary = Path(raw_temporary)
if (
    temporary.parent != root
    or not temporary.name.startswith("submission.json.tmp.")
):
    raise SystemExit("submission temporary path differs")
temporary = require_experiment_path(
    temporary,
    root,
    temporary.name,
    kind="file",
)
destination = require_experiment_path(
    raw_destination,
    root,
    "submission.json",
    kind="file",
    allow_missing=True,
)
os.replace(temporary, destination)
directory_flags = os.O_RDONLY
if hasattr(os, "O_DIRECTORY"):
    directory_flags |= os.O_DIRECTORY
directory = os.open(root, directory_flags)
try:
    os.fsync(directory)
finally:
    os.close(directory)
require_experiment_path(
    destination,
    root,
    "submission.json",
    kind="file",
)
PY

if [[ "${job_seed_validated[0]}" == true ]] \
  && [[ "${job_seed_validated[1]}" == true ]]; then
  seed_temporary="${seed_receipt}.tmp.$$"
  "${python_bin}" - "${experiment_root}" "${source_tree_sha256}" \
    "${job_attempt_receipts[0]}" "${seed_identities[0]}" \
    "${job_attempt_receipts[1]}" "${seed_identities[1]}" \
    "${seed_temporary}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from datetime import datetime, timezone

from motive.r10_path_contract import (
    canonical_experiment_root,
    require_attempt_receipt_path,
    require_experiment_path,
)

(
    experiment_root,
    source_tree_sha256,
    first_attempt,
    first_identity,
    second_attempt,
    second_identity,
    raw_output,
) = sys.argv[1:]
root = canonical_experiment_root(experiment_root)


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validation_job(raw_path, seed):
    raw = Path(raw_path)
    prefix = f"provenance/job_attempts/seed_{seed}/attempt_"
    try:
        relative = raw.relative_to(root).as_posix()
    except ValueError as error:
        raise SystemExit("validation attempt receipt escapes root") from error
    if not relative.startswith(prefix) or not relative.endswith(".json"):
        raise SystemExit("validation attempt receipt path differs")
    raw_attempt = relative[len(prefix):-len(".json")]
    if not raw_attempt.isdigit() or int(raw_attempt) < 1:
        raise SystemExit("validation attempt receipt number differs")
    path = require_attempt_receipt_path(
        raw,
        root,
        seed,
        int(raw_attempt),
    )
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
        raise SystemExit("validation attempt receipt is invalid")
    metadata = path.stat()
    if metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o444:
        raise SystemExit("validation attempt receipt permissions differ")
    with path.open(encoding="utf-8") as handle:
        row = json.load(handle)
    if (
        row.get("schema_version")
        != "motive-r10a-job-attempt-receipt-v1"
        or row.get("experiment_root") != experiment_root
        or row.get("source_tree_sha256") != source_tree_sha256
        or row.get("seed") != seed
        or row.get("attempt") != int(raw_attempt)
        or row.get("run_root") != f"{experiment_root}/seed_{seed}"
    ):
        raise SystemExit("validation attempt receipt identity differs")
    return {
        "job_id": row["job_id"],
        "attempt": row["attempt"],
        "job_name": row["job_name"],
        "exit_code": "0:0",
        "attempt_receipt": {
            "path": str(path.resolve(strict=True)),
            "sha256": digest_file(path),
            "bytes": metadata.st_size,
        },
    }


jobs = []
for seed, raw_attempt, raw_identity in (
    (260108837, first_attempt, first_identity),
    (260108838, second_attempt, second_identity),
):
    identity = json.loads(raw_identity)
    jobs.append(
        {
            "seed": seed,
            "validation_job": validation_job(raw_attempt, seed),
            "artifact": identity["artifact"],
            "artifact_producer": identity["artifact_producer"],
        }
    )
payload = {
    "schema_version": "motive-r10a-two-seed-validation-v2",
    "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    "experiment_root": experiment_root,
    "source_tree_sha256": source_tree_sha256,
    "jobs": jobs,
    "both_seed_artifacts_validated": True,
    "representation_gate_passed": False,
    "renderer_probe_authorized": False,
    "editor_training_authorized": False,
}
output = Path(raw_output)
if (
    output.parent != root
    or not output.name.startswith("seed_validation.json.tmp.")
):
    raise SystemExit("seed validation temporary path differs")
output = require_experiment_path(
    output,
    root,
    output.name,
    kind="file",
    allow_missing=True,
)
with output.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
  chmod 0444 "${seed_temporary}"
  "${python_bin}" - "${seed_temporary}" "${seed_receipt}" <<'PY'
import os
from pathlib import Path
import sys

from motive.r10_path_contract import (
    canonical_experiment_root,
    require_experiment_path,
)

temporary, destination = map(Path, sys.argv[1:])
root = canonical_experiment_root(destination.parent)
if (
    temporary.parent != root
    or not temporary.name.startswith("seed_validation.json.tmp.")
):
    raise SystemExit("seed validation temporary path differs")
temporary = require_experiment_path(
    temporary,
    root,
    temporary.name,
    kind="file",
)
destination = require_experiment_path(
    destination,
    root,
    "seed_validation.json",
    kind="file",
    allow_missing=True,
)
try:
    os.link(temporary, destination, follow_symlinks=False)
finally:
    temporary.unlink()
directory_flags = os.O_RDONLY
if hasattr(os, "O_DIRECTORY"):
    directory_flags |= os.O_DIRECTORY
directory = os.open(destination.parent, directory_flags)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
  validate_seed_receipt \
    "${seed_receipt}" "${seed_identities[0]}" "${seed_identities[1]}"
  echo "[r10a-submit] both seed artifacts validated; aggregate pending"
else
  echo \
    "[r10a-submit] coordinated jobs=${job_ids[*]} states=${job_states[*]}"
fi
echo "[r10a-submit] state_receipt=${receipt}"
