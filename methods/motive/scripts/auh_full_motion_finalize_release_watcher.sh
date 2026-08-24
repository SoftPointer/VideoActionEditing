#!/usr/bin/env bash
# Close the full-Qwen output into a deterministic generation pool, materialize
# its contiguous shard manifests, and publish an offline signing request.
#
# This watcher is deliberately started before the distributed Qwen controller.
# It publishes a narrow readiness receipt after preflight, then waits for the
# exact current Qwen controller receipt.  It never signs a release and never
# starts Wan generation.

set -Eeuo pipefail
umask 077

job_id="${MOTIVE_EXISTING_SLURM_JOB_ID:?set existing Slurm job ID}"
nodes_csv="${MOTIVE_FULL_MOTION_PIPELINE_NODES:?set eight ordered allocation nodes}"
qwen_nodes_csv="${MOTIVE_FULL_MOTION_QWEN_NODES:?set four ordered Qwen nodes}"
finalize_node="${MOTIVE_FULL_MOTION_FINALIZE_NODE:?set finalizer node}"

snapshot="${MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT:?set frozen source snapshot}"
source_tree_sha="${MOTIVE_FULL_MOTION_SOURCE_TREE_SHA256:?set source tree SHA-256}"
snapshot_tool="${MOTIVE_FULL_MOTION_SNAPSHOT_TOOL:?set snapshot verifier}"
code_root="${snapshot}/methods/motive"

full_input="${MOTIVE_FULL_MOTION_FULL_INPUT:?set full-Qwen input}"
full_input_sha="${MOTIVE_FULL_MOTION_FULL_INPUT_SHA256:?set full-Qwen input SHA-256}"
qwen_root="${MOTIVE_FULL_MOTION_FULL_QWEN_ROOT:?set full-Qwen output root}"
qwen_done="${MOTIVE_FULL_MOTION_FULL_QWEN_DONE:?set full-Qwen controller receipt}"
final_pool="${MOTIVE_FULL_MOTION_FINAL_POOL:?set final-pool output root}"
production_root="${MOTIVE_FULL_MOTION_PRODUCTION_ROOT:?set production root}"
shard_root="${MOTIVE_FULL_MOTION_SHARD_MANIFEST_DIR:?set shard-manifest output root}"
release_dir="${MOTIVE_FULL_MOTION_RELEASE_DIR:?set release directory}"
release_request="${MOTIVE_FULL_MOTION_RELEASE_REQUEST:?set release request output}"
release_id="${MOTIVE_FULL_MOTION_RELEASE_ID:?set release ID}"
release_challenge="${MOTIVE_FULL_MOTION_RELEASE_CHALLENGE:?set release challenge SHA-256}"

python_bin="${MOTIVE_FULL_MOTION_QWEN_PYTHON:?set frozen Qwen Python}"
wait_seconds="${MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WAIT_SECONDS:?set Qwen wait timeout}"
poll_seconds="${MOTIVE_FULL_MOTION_FINALIZE_RELEASE_POLL_SECONDS:?set Qwen poll interval}"
finalize_cpus="${MOTIVE_FULL_MOTION_FINALIZE_CPUS:?set finalizer CPUs}"
ready_receipt="${MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WATCHER_READY:?set watcher readiness receipt}"
terminal_receipt="${MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WATCHER_RECEIPT:?set watcher terminal receipt}"

fail() {
  printf '[full-motion-finalize-release-watcher] %s\n' "$*" >&2
  exit 2
}

require_absolute() {
  local label="$1"
  local path="$2"
  [[ "${path}" == /* && "${path}" != "/" ]] \
    || fail "${label} must be a non-root absolute path: ${path}"
}

require_plain_file() {
  local label="$1"
  local path="$2"
  [[ ! -L "${path}" && -f "${path}" ]] \
    || fail "${label} must be a regular non-symlink file: ${path}"
}

require_plain_directory() {
  local label="$1"
  local path="$2"
  [[ ! -L "${path}" && -d "${path}" ]] \
    || fail "${label} must be a non-symlink directory: ${path}"
}

require_new() {
  local label="$1"
  local path="$2"
  [[ ! -e "${path}" && ! -L "${path}" ]] \
    || fail "create-only ${label} already exists: ${path}"
}

[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || fail "invalid Slurm job ID"
for value in "${wait_seconds}" "${poll_seconds}" "${finalize_cpus}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "invalid positive integer setting"
done
for digest in "${source_tree_sha}" "${full_input_sha}" "${release_challenge}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid lowercase SHA-256 setting"
done
[[ -n "${release_id}" && "${release_id}" == "${release_id# }" \
  && "${release_id}" == "${release_id% }" \
  && "${release_id}" != *$'\n'* && "${release_id}" != *$'\r'* ]] \
  || fail "release ID must be one canonical non-empty line"

IFS=, read -r -a nodes <<<"${nodes_csv}"
IFS=, read -r -a qwen_nodes <<<"${qwen_nodes_csv}"
(( ${#nodes[@]} == 8 )) || fail "exactly eight allocation nodes are required"
(( ${#qwen_nodes[@]} == 4 )) || fail "exactly four Qwen nodes are required"
declare -A seen_nodes=()
for node in "${nodes[@]}"; do
  [[ "${node}" =~ ^auh[0-9A-Za-z-]+$ ]] || fail "invalid allocation node: ${node}"
  [[ -z "${seen_nodes[${node}]:-}" ]] || fail "duplicate allocation node: ${node}"
  seen_nodes["${node}"]=1
done
for index in 0 1 2 3; do
  [[ "${qwen_nodes[${index}]}" == "${nodes[${index}]}" ]] \
    || fail "Qwen nodes must equal the first four ordered allocation nodes"
done
[[ "${finalize_node}" =~ ^auh[0-9A-Za-z-]+$ \
  && -n "${seen_nodes[${finalize_node}]:-}" ]] \
  || fail "finalizer node is outside the declared allocation"

for binding in \
  "snapshot:${snapshot}" "snapshot verifier:${snapshot_tool}" \
  "full input:${full_input}" "Qwen root:${qwen_root}" \
  "Qwen receipt:${qwen_done}" "final pool:${final_pool}" \
  "production root:${production_root}" "shard root:${shard_root}" \
  "release directory:${release_dir}" "release request:${release_request}" \
  "Qwen Python:${python_bin}" "readiness receipt:${ready_receipt}" \
  "terminal receipt:${terminal_receipt}"; do
  require_absolute "${binding%%:*}" "${binding#*:}"
done

require_plain_directory "source snapshot" "${snapshot}"
require_plain_file "source snapshot closure" "${snapshot}/SOURCE_FILES.jsonl"
require_plain_file "snapshot verifier" "${snapshot_tool}"
require_plain_directory "frozen Motive code root" "${code_root}"
for module in \
  goku_full_motion_finalize.py \
  goku_full_motion_shard_manifest.py \
  wan22_full_motion_signed_release.py; do
  require_plain_file "frozen implementation" "${code_root}/motive/${module}"
done
require_plain_file "full-Qwen input" "${full_input}"
require_plain_file "Qwen Python" "${python_bin}"
[[ -x "${python_bin}" ]] || fail "Qwen Python is not executable"
[[ "$(wc -l <"${full_input}" | tr -d '[:space:]')" == 768 ]] \
  || fail "full-Qwen input must contain exactly 768 rows"
require_plain_directory "production root" "${production_root}"
require_plain_directory "release directory" "${release_dir}"
[[ "${shard_root}" == "${production_root}/generation_shards" ]] \
  || fail "shard root differs from the fixed production layout"
[[ "${release_dir}" == "${production_root}/release" \
  && "${release_request}" == "${release_dir}/release_request.json" ]] \
  || fail "release request differs from the fixed production layout"

for parent in \
  "${qwen_root%/*}" "${qwen_done%/*}" "${final_pool%/*}" \
  "${shard_root%/*}" "${release_request%/*}" \
  "${ready_receipt%/*}" "${terminal_receipt%/*}"; do
  require_plain_directory "output parent" "${parent}"
  [[ -w "${parent}" ]] || fail "output parent is not writable: ${parent}"
done
require_new "full-Qwen controller receipt" "${qwen_done}"
if [[ -e "${qwen_root}" || -L "${qwen_root}" ]]; then
  require_plain_directory "reserved full-Qwen root" "${qwen_root}"
  [[ -z "$(find "${qwen_root}" -mindepth 1 -print -quit)" ]] \
    || fail "reserved full-Qwen root is not empty"
fi
require_new "final pool" "${final_pool}"
require_new "shard-manifest root" "${shard_root}"
require_new "release request" "${release_request}"
require_new "watcher readiness receipt" "${ready_receipt}"
require_new "watcher terminal receipt" "${terminal_receipt}"
[[ -z "$(find "${release_dir}" -mindepth 1 -print -quit)" ]] \
  || fail "release directory is not empty at watcher preflight"

"${python_bin}" - \
  "${snapshot}" "${snapshot_tool}" \
  "${full_input}" "${qwen_root}" "${qwen_done}" \
  "${final_pool}" "${production_root}" "${shard_root}" "${release_dir}" \
  "${release_request}" "${ready_receipt}" "${terminal_receipt}" <<'PY'
import os
import sys


def normalized(value: str) -> str:
    return os.path.normpath(os.path.abspath(value))


def overlaps(left: str, right: str) -> bool:
    common = os.path.commonpath((left, right))
    return common == left or common == right


(
    snapshot,
    snapshot_tool,
    full_input,
    qwen_root,
    qwen_done,
    final_pool,
    production_root,
    shard_root,
    release_dir,
    release_request,
    ready_receipt,
    terminal_receipt,
) = map(normalized, sys.argv[1:])

resolved_snapshot = os.path.realpath(snapshot)
resolved_snapshot_tool = os.path.realpath(snapshot_tool)
if os.path.commonpath((resolved_snapshot, resolved_snapshot_tool)) \
        != resolved_snapshot or resolved_snapshot_tool == resolved_snapshot:
    raise SystemExit("snapshot verifier must resolve inside the frozen snapshot")

qwen_paths = (full_input, qwen_root, qwen_done)
if len(set(qwen_paths)) != len(qwen_paths) or any(
    overlaps(left, right)
    for index, left in enumerate(qwen_paths)
    for right in qwen_paths[index + 1:]
):
    raise SystemExit("full input, Qwen root, and Qwen receipt must not overlap")

if overlaps(final_pool, production_root) or overlaps(final_pool, shard_root) \
        or overlaps(final_pool, release_dir):
    raise SystemExit("final pool must not overlap production outputs")

receipts = (ready_receipt, terminal_receipt)
if ready_receipt == terminal_receipt:
    raise SystemExit("watcher readiness and terminal receipts must be distinct")
for receipt in receipts:
    for protected in (final_pool, shard_root, release_dir):
        if overlaps(receipt, protected):
            raise SystemExit("watcher receipts must be outside stage output roots")
for output in (release_request, ready_receipt, terminal_receipt):
    if output in {full_input, qwen_done} or overlaps(output, qwen_root):
        raise SystemExit("watcher publication path overlaps a Qwen input/output")
PY

for command in scontrol squeue srun mktemp /bin/ln; do
  command -v "${command}" >/dev/null || fail "required command is unavailable: ${command}"
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
unset PYTHONOPTIMIZE || true
export PYTHONPATH="${code_root}:${snapshot}"

active_pid=""
release_stage_dir=""
receipt_stage_dir=""
release_stage_request=""
receipt_stage_file=""

cleanup_staging() {
  if [[ -n "${release_stage_request}" ]]; then
    rm -f -- "${release_stage_request}" 2>/dev/null || true
  fi
  if [[ -n "${release_stage_dir}" ]]; then
    rmdir "${release_stage_dir}" 2>/dev/null || true
  fi
  if [[ -n "${receipt_stage_file}" ]]; then
    rm -f -- "${receipt_stage_file}" 2>/dev/null || true
  fi
  if [[ -n "${receipt_stage_dir}" ]]; then
    rmdir "${receipt_stage_dir}" 2>/dev/null || true
  fi
}

abort_signal() {
  local status="$1"
  trap - TERM INT
  if [[ -n "${active_pid}" ]] && kill -0 "${active_pid}" 2>/dev/null; then
    kill -TERM "${active_pid}" 2>/dev/null || true
    wait "${active_pid}" 2>/dev/null || true
  fi
  active_pid=""
  cleanup_staging
  exit "${status}"
}

trap cleanup_staging EXIT
trap 'abort_signal 143' TERM
trap 'abort_signal 130' INT

verify_snapshot() {
  "${python_bin}" "${snapshot_tool}" verify \
    --snapshot "${snapshot}" --expected-tree-sha256 "${source_tree_sha}" \
    >/dev/null || fail "frozen source snapshot verification failed"
}

verify_allocation() {
  local job_record allocation_expression
  local -a allocated_nodes
  job_record="$(scontrol show job -o "${job_id}")" \
    || fail "cannot inspect existing allocation"
  for required in \
    "JobId=${job_id}" "UserId=$(id -un)(" "JobState=RUNNING" \
    "NumNodes=8" "gres/gpu:mi210=64"; do
    [[ "${job_record}" == *"${required}"* ]] \
      || fail "allocation identity differs: ${required}"
  done
  allocation_expression="$(squeue -j "${job_id}" -h -o '%N')" \
    || fail "cannot read allocation node expression"
  [[ -n "${allocation_expression}" && "${allocation_expression}" != *$'\n'* ]] \
    || fail "allocation node expression is not one line"
  mapfile -t allocated_nodes < <(scontrol show hostnames "${allocation_expression}")
  (( ${#allocated_nodes[@]} == 8 )) \
    || fail "allocation does not expand to exactly eight nodes"
  for index in 0 1 2 3 4 5 6 7; do
    [[ "${allocated_nodes[${index}]}" == "${nodes[${index}]}" ]] \
      || fail "ordered allocation nodes differ at index ${index}"
  done
}

file_sha256() {
  "${python_bin}" - "$1" <<'PY'
import hashlib
import os
import stat
import sys

path = os.path.abspath(sys.argv[1])
before = os.lstat(path)
if not stat.S_ISREG(before.st_mode):
    raise SystemExit("SHA-256 input is not a plain file")
digest = hashlib.sha256()
with open(path, "rb") as handle:
    while block := handle.read(4 * 1024 * 1024):
        digest.update(block)
after = os.lstat(path)
if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
):
    raise SystemExit("SHA-256 input changed while being read")
print(digest.hexdigest())
PY
}

validate_closure() {
  "${python_bin}" - "$@" <<'PY'
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys


def fail(message: str) -> None:
    raise SystemExit(message)


def reject_constant(value: str) -> None:
    fail(f"non-finite JSON constant: {value}")


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def object_sha(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def plain_file(path_value: str, context: str) -> Path:
    path = Path(os.path.abspath(path_value))
    try:
        before = path.lstat()
    except FileNotFoundError:
        fail(f"{context} is missing: {path}")
    if not stat.S_ISREG(before.st_mode):
        fail(f"{context} is not a regular non-symlink file: {path}")
    return path


def read(path_value: str | Path, context: str) -> bytes:
    path = plain_file(str(path_value), context)
    before = path.lstat()
    raw = path.read_bytes()
    after = path.lstat()
    identity_before = (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    )
    identity_after = (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    )
    if identity_before != identity_after or len(raw) != after.st_size:
        fail(f"{context} changed while it was read")
    return raw


def load(path_value: str | Path, context: str):
    raw = read(path_value, context)
    try:
        value = json.loads(
            raw.decode("utf-8"), parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{context} is not strict UTF-8 JSON: {error}")
    if not isinstance(value, dict):
        fail(f"{context} is not one JSON object")
    return value, raw


def check_digest(value, field: str, context: str) -> None:
    stored = value.get(field)
    if not isinstance(stored, str) or re.fullmatch(r"[0-9a-f]{64}", stored) is None:
        fail(f"{context} digest is malformed")
    payload = dict(value)
    del payload[field]
    if stored != object_sha(payload):
        fail(f"{context} digest differs")


def require_plain_dir(path_value: str, context: str) -> Path:
    path = Path(os.path.abspath(path_value))
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        fail(f"{context} is missing")
    if not stat.S_ISDIR(mode):
        fail(f"{context} is not a plain directory")
    return path


def line_count(raw: bytes, context: str) -> int:
    if not raw or not raw.endswith(b"\n") or b"\n\n" in raw:
        fail(f"{context} is not non-empty newline-terminated JSONL")
    return len(raw.splitlines())


mode = sys.argv[1]
if mode == "qwen":
    done_path, input_path, expected_input_sha, qwen_root = sys.argv[2:6]
    job_id, expected_nodes = sys.argv[6:8]
    raw = read(done_path, "Qwen controller receipt")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        fail("Qwen controller receipt is not UTF-8")
    if not raw.endswith(b"\n") or "\r" in text or "\x00" in text:
        fail("Qwen controller receipt is not canonical line text")
    lines = text.splitlines()
    expected_keys = [
        "schema", "status", "input", "input_sha256", "output_root",
        "slurm_job_id", "nodes", "completed_at_utc",
    ]
    if len(lines) != 8:
        fail("Qwen controller receipt must contain exactly eight lines")
    values = {}
    for index, (line, expected_key) in enumerate(zip(lines, expected_keys), 1):
        if "=" not in line:
            fail(f"Qwen controller receipt line {index} lacks '='")
        key, value = line.split("=", 1)
        if key != expected_key or key in values or not value:
            fail(f"Qwen controller receipt line {index} differs")
        values[key] = value
    expected = {
        "schema": "motive-goku-full-motion-qwen-controller-v1",
        "status": "complete",
        "input": input_path,
        "input_sha256": expected_input_sha,
        "output_root": qwen_root,
        "slurm_job_id": job_id,
        "nodes": expected_nodes,
    }
    for key, value in expected.items():
        if values.get(key) != value:
            fail(f"Qwen controller receipt {key} binding differs")
    try:
        datetime.strptime(values["completed_at_utc"], "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        fail("Qwen controller receipt timestamp is not strict UTC")
    input_raw = read(input_path, "full-Qwen input")
    if hashlib.sha256(input_raw).hexdigest() != expected_input_sha:
        fail("full-Qwen input SHA-256 differs")
    root = require_plain_dir(qwen_root, "Qwen output root")
    expected_entries = {
        f"qwen_shard_{index:03d}{suffix}"
        for index in range(8)
        for suffix in (".jsonl", ".receipt.json")
    }
    if set(os.listdir(root)) != expected_entries:
        fail("Qwen output root entries differ from the exact 8-shard closure")
    for name in expected_entries:
        read(root / name, f"Qwen artifact {name}")
    print(hashlib.sha256(raw).hexdigest())

elif mode == "outputs":
    final_pool, shard_root, request_path = sys.argv[2:5]
    input_path, expected_input_sha, qwen_root = sys.argv[5:8]
    release_id, challenge = sys.argv[8:10]
    final = require_plain_dir(final_pool, "final pool")
    final_entries = {
        "primary_256.jsonl", "reserve_64.jsonl", "review_candidates.jsonl",
        "summary.json", "done.json",
    }
    if set(os.listdir(final)) != final_entries:
        fail("final-pool entries differ")
    final_done, _ = load(final / "done.json", "finalizer done")
    if (
        final_done.get("schema_version")
        != "motive-goku-full-motion-finalize-done-v1"
        or final_done.get("status") != "complete"
    ):
        fail("finalizer done identity differs")
    check_digest(final_done, "done_digest", "finalizer done")
    artifacts = final_done.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != final_entries - {"done.json"}:
        fail("finalizer artifact closure differs")
    expected_rows = {"primary_256.jsonl": 256, "reserve_64.jsonl": 64}
    for name, metadata in artifacts.items():
        if not isinstance(metadata, dict) or set(metadata) != {"sha256", "bytes", "rows"}:
            fail(f"finalizer metadata differs for {name}")
        raw = read(final / name, f"finalizer artifact {name}")
        if name.endswith(".json"):
            rows = 1
        elif not raw and name == "review_candidates.jsonl":
            rows = 0
        else:
            rows = line_count(raw, name)
        if metadata != {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "rows": rows,
        }:
            fail(f"finalizer artifact binding differs for {name}")
        if name in expected_rows and rows != expected_rows[name]:
            fail(f"finalizer row count differs for {name}")
    inputs = final_done.get("inputs")
    candidate = inputs.get("candidate_manifest") if isinstance(inputs, dict) else None
    if not isinstance(candidate, dict):
        fail("finalizer candidate binding is absent")
    if candidate.get("path") != input_path or candidate.get("sha256") != expected_input_sha:
        fail("finalizer candidate binding differs")
    qwen_shards = inputs.get("qwen_shards")
    if not isinstance(qwen_shards, list) or len(qwen_shards) != 8:
        fail("finalizer Qwen shard closure differs")
    resolved_qwen = Path(qwen_root).resolve(strict=True)
    for shard in qwen_shards:
        if not isinstance(shard, dict):
            fail("finalizer Qwen shard binding is malformed")
        output = Path(str(shard.get("output_path", ""))).resolve(strict=True)
        receipt = Path(str(shard.get("receipt_path", ""))).resolve(strict=True)
        if output.parent != resolved_qwen or receipt.parent != resolved_qwen:
            fail("finalizer Qwen shard path escapes the Qwen root")

    primary_raw = read(final / "primary_256.jsonl", "primary manifest")
    shards = require_plain_dir(shard_root, "shard-manifest root")
    if set(os.listdir(shards)) != {"shards", "jobs.tsv", "summary.json", "done.json"}:
        fail("shard-manifest root entries differ")
    leaf = require_plain_dir(str(shards / "shards"), "shard-manifest leaf")
    shard_names = {f"shard_{index:03d}.jsonl" for index in range(32)}
    if set(os.listdir(leaf)) != shard_names:
        fail("shard-manifest leaf entries differ")
    shard_done, _ = load(shards / "done.json", "shard-manifest done")
    if (
        shard_done.get("schema_version")
        != "motive-goku-full-motion-shard-manifest-done-v1"
        or shard_done.get("status") != "complete"
    ):
        fail("shard-manifest done identity differs")
    check_digest(shard_done, "done_digest", "shard-manifest done")
    shard_artifacts = shard_done.get("artifacts")
    expected_artifacts = {f"shards/{name}" for name in shard_names} | {
        "jobs.tsv", "summary.json"
    }
    if not isinstance(shard_artifacts, dict) or set(shard_artifacts) != expected_artifacts:
        fail("shard-manifest artifact closure differs")
    ordered = []
    shard_job_records = []
    for index in range(32):
        name = f"shard_{index:03d}.jsonl"
        relative = f"shards/{name}"
        raw = read(leaf / name, f"generation shard {index}")
        if line_count(raw, relative) != 8:
            fail(f"generation shard {index} does not contain eight rows")
        metadata = shard_artifacts.get(relative)
        if metadata != {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "rows": 8,
        }:
            fail(f"generation shard {index} binding differs")
        try:
            rows = [json.loads(line.decode("utf-8")) for line in raw.splitlines()]
            iids = [row["iid"] for row in rows]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
            fail(f"generation shard {index} rows are invalid: {error}")
        if any(not isinstance(iid, str) or not iid for iid in iids):
            fail(f"generation shard {index} IID list is invalid")
        row_digests = [object_sha(row) for row in rows]
        shard_job_records.append(
            {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "iids": iids,
                "ordered_iids_sha256": hashlib.sha256(
                    b"".join(iid.encode("utf-8") + b"\n" for iid in iids)
                ).hexdigest(),
                "ordered_row_sha256": hashlib.sha256(
                    b"".join(
                        digest.encode("ascii") + b"\n" for digest in row_digests
                    )
                ).hexdigest(),
            }
        )
        ordered.append(raw)
    if b"".join(ordered) != primary_raw:
        fail("generation shards do not reconstruct the primary manifest")
    for name in ("jobs.tsv", "summary.json"):
        raw = read(shards / name, f"shard artifact {name}")
        metadata = shard_artifacts.get(name)
        expected_row_count = 32 if name == "jobs.tsv" else 1
        if metadata != {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "rows": expected_row_count,
        }:
            fail(f"shard artifact binding differs for {name}")
    jobs_raw = read(shards / "jobs.tsv", "shard jobs TSV")
    try:
        jobs_text = jobs_raw.decode("utf-8")
    except UnicodeDecodeError:
        fail("shard jobs TSV is not UTF-8")
    if not jobs_raw.endswith(b"\n") or "\r" in jobs_text or "\x00" in jobs_text:
        fail("shard jobs TSV is not canonical line text")
    job_lines = jobs_text.splitlines()
    expected_header = [
        "shard_index", "shard_id", "manifest_relpath",
        "root_row_start_zero_based", "root_row_end_exclusive", "row_count",
        "manifest_sha256", "manifest_bytes", "ordered_iids_sha256",
        "ordered_row_sha256", "ordered_iids_json",
    ]
    if len(job_lines) != 33 or job_lines[0].split("\t") != expected_header:
        fail("shard jobs TSV must contain the exact header and 32 data rows")
    for index, line in enumerate(job_lines[1:]):
        record = shard_job_records[index]
        expected_values = [
            str(index),
            f"shard_{index:03d}",
            f"shards/shard_{index:03d}.jsonl",
            str(index * 8),
            str((index + 1) * 8),
            "8",
            record["sha256"],
            str(record["bytes"]),
            record["ordered_iids_sha256"],
            record["ordered_row_sha256"],
            json.dumps(record["iids"], ensure_ascii=False, separators=(",", ":")),
        ]
        if line.split("\t") != expected_values:
            fail(f"shard jobs TSV binding differs at data row {index + 1}")

    request, request_raw = load(request_path, "release request")
    if set(request) != {
        "schema_version", "challenge_sha256", "builder", "signed",
        "request_digest",
    }:
        fail("release request keys differ")
    if (
        request.get("schema_version")
        != "motive-wan22-full-motion-release-request-v3"
        or request.get("challenge_sha256") != challenge
    ):
        fail("release request identity differs")
    check_digest(request, "request_digest", "release request")
    signed = request.get("signed")
    if not isinstance(signed, dict) or (
        signed.get("schema_version")
        != "motive-wan22-full-motion-root-release-payload-v3"
        or signed.get("release_id") != release_id
    ):
        fail("release request payload binding differs")
    try:
        datetime.fromisoformat(str(signed.get("issued_at_utc", "")).replace("Z", "+00:00"))
    except ValueError:
        fail("release request timestamp is invalid")
    scope = signed.get("root_manifest")
    if not isinstance(scope, dict) or (
        scope.get("sha256") != hashlib.sha256(primary_raw).hexdigest()
        or scope.get("bytes") != len(primary_raw)
        or scope.get("rows") != 256
        or scope.get("contiguous_shard_rows") != 8
    ):
        fail("release request root-manifest binding differs")
    print(hashlib.sha256(request_raw).hexdigest())

elif mode in {"ready_receipt", "terminal_receipt"}:
    output = Path(sys.argv[2])
    job_id, nodes_csv, qwen_nodes_csv, finalize_node = sys.argv[3:7]
    if mode == "ready_receipt":
        value = {
            "schema_version": "motive-goku-full-motion-finalize-release-watcher-ready-v1",
            "status": "ready",
            "slurm_job_id": job_id,
            "nodes": nodes_csv.split(","),
            "qwen_nodes": qwen_nodes_csv.split(","),
            "finalize_node": finalize_node,
        }
    else:
        snapshot, input_path, input_sha, qwen_done = sys.argv[7:11]
        qwen_done_sha, final_pool, shard_root = sys.argv[11:14]
        request_path, request_sha, completed = sys.argv[14:17]
        value = {
            "schema_version": "motive-goku-full-motion-finalize-release-watcher-v1",
            "status": "complete",
            "slurm_job_id": job_id,
            "nodes": nodes_csv.split(","),
            "finalize_node": finalize_node,
            "source_snapshot": snapshot,
            "full_input": {"path": input_path, "sha256": input_sha},
            "qwen_done": {"path": qwen_done, "sha256": qwen_done_sha},
            "final_pool": final_pool,
            "shard_manifest_root": shard_root,
            "release_request": {"path": request_path, "sha256": request_sha},
            "completed_at_utc": completed,
        }
    value["receipt_digest"] = object_sha(value)
    raw = canonical(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(output, flags, 0o400)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)
else:
    fail(f"unknown closure mode: {mode}")
PY
}

publish_hardlink() {
  local staging="$1"
  local output="$2"
  /bin/ln "${staging}" "${output}" \
    || fail "create-only hard-link publication failed: ${output}"
  "${python_bin}" - "${staging}" "${output}" <<'PY'
import os
import stat
import sys

source, output = map(os.path.abspath, sys.argv[1:])
left = os.lstat(source)
right = os.lstat(output)
if not stat.S_ISREG(left.st_mode) or not stat.S_ISREG(right.st_mode):
    raise SystemExit("hard-link publication is not two plain files")
if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
    raise SystemExit("hard-link publication inode differs")
if stat.S_IMODE(right.st_mode) != 0o400:
    raise SystemExit("hard-link publication mode differs")
PY
}

run_step() {
  local label="$1"
  shift
  verify_allocation
  verify_snapshot
  srun --overlap \
    --jobid="${job_id}" --nodelist="${finalize_node}" --nodes=1 \
    --exact --ntasks=1 --cpus-per-task="${finalize_cpus}" --mem=0 \
    env \
      PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
      PYTHONPATH="${PYTHONPATH}" \
      "$@" &
  active_pid="$!"
  local status=0
  wait "${active_pid}" || status="$?"
  active_pid=""
  (( status == 0 )) || fail "${label} Slurm step failed with status ${status}"
}

verify_allocation
verify_snapshot
actual_input_sha="$(file_sha256 "${full_input}")" \
  || fail "cannot hash the full-Qwen input"
[[ "${actual_input_sha}" == "${full_input_sha}" ]] \
  || fail "full-Qwen input SHA-256 differs"

# Readiness is intentionally narrower than completion.  Its existence lets an
# upstream controller start the expensive Qwen stage without a timing race.
receipt_stage_dir="$(mktemp -d "${ready_receipt%/*}/.watcher-ready.XXXXXX")"
receipt_stage_file="${receipt_stage_dir}/ready.json"
validate_closure ready_receipt "${receipt_stage_file}" \
  "${job_id}" "${nodes_csv}" "${qwen_nodes_csv}" "${finalize_node}"
publish_hardlink "${receipt_stage_file}" "${ready_receipt}"
rm -f -- "${receipt_stage_file}"
receipt_stage_file=""
rmdir "${receipt_stage_dir}"
receipt_stage_dir=""
printf '[full-motion-finalize-release-watcher] ready: %s\n' "${ready_receipt}"

deadline=$(( $(date +%s) + wait_seconds ))
while [[ ! -s "${qwen_done}" ]]; do
  [[ ! -L "${qwen_done}" ]] \
    || fail "full-Qwen controller receipt became a symlink"
  if [[ -e "${qwen_done}" ]]; then
    [[ -f "${qwen_done}" ]] \
      || fail "full-Qwen controller receipt is not a regular file"
  fi
  (( $(date +%s) < deadline )) \
    || fail "timed out waiting for the full-Qwen controller receipt"
  sleep "${poll_seconds}" &
  active_pid="$!"
  sleep_status=0
  wait "${active_pid}" || sleep_status="$?"
  active_pid=""
  (( sleep_status == 0 )) || fail "Qwen wait interval was interrupted"
done

verify_allocation
verify_snapshot
qwen_done_sha="$(validate_closure qwen \
  "${qwen_done}" "${full_input}" "${full_input_sha}" "${qwen_root}" \
  "${job_id}" "${qwen_nodes_csv}")" \
  || fail "full-Qwen controller closure is invalid"

run_step "full-motion finalizer" \
  "${python_bin}" -m motive.goku_full_motion_finalize \
  --candidate-manifest "${full_input}" \
  --qwen-dir "${qwen_root}" \
  --output-dir "${final_pool}" \
  --primary-size 256 --reserve-size 64 \
  --min-primary-multi-dynamic 64 \
  --target-signature-cap 32 --family-cap 32 \
  --require-iid 1dbe39537c984690

run_step "full-motion shard materialization" \
  "${python_bin}" -m motive.goku_full_motion_shard_manifest \
  --finalizer-dir "${final_pool}" \
  --output-dir "${shard_root}"

release_stage_dir="$(mktemp -d "${release_dir}/.release-request-stage.XXXXXX")"
release_stage_request="${release_stage_dir}/release_request.json"
issued_at_utc="$(date -u +%FT%TZ)"
run_step "full-motion release-request preparation" \
  "${python_bin}" -m motive.wan22_full_motion_signed_release prepare \
  --root-manifest "${final_pool}/primary_256.jsonl" \
  --request "${release_stage_request}" \
  --release-id "${release_id}" \
  --issued-at-utc "${issued_at_utc}" \
  --challenge "${release_challenge}"
require_plain_file "staged release request" "${release_stage_request}"
publish_hardlink "${release_stage_request}" "${release_request}"
rm -f -- "${release_stage_request}"
release_stage_request=""
rmdir "${release_stage_dir}"
release_stage_dir=""

verify_allocation
verify_snapshot
release_request_sha="$(validate_closure outputs \
  "${final_pool}" "${shard_root}" "${release_request}" \
  "${full_input}" "${full_input_sha}" "${qwen_root}" \
  "${release_id}" "${release_challenge}")" \
  || fail "finalizer, shard-manifest, or release-request closure is invalid"

completed_at_utc="$(date -u +%FT%TZ)"
receipt_stage_dir="$(mktemp -d "${terminal_receipt%/*}/.watcher-terminal.XXXXXX")"
receipt_stage_file="${receipt_stage_dir}/terminal.json"
validate_closure terminal_receipt "${receipt_stage_file}" \
  "${job_id}" "${nodes_csv}" "${qwen_nodes_csv}" "${finalize_node}" \
  "${snapshot}" "${full_input}" "${full_input_sha}" "${qwen_done}" \
  "${qwen_done_sha}" "${final_pool}" "${shard_root}" \
  "${release_request}" "${release_request_sha}" "${completed_at_utc}"
publish_hardlink "${receipt_stage_file}" "${terminal_receipt}"
rm -f -- "${receipt_stage_file}"
receipt_stage_file=""
rmdir "${receipt_stage_dir}"
receipt_stage_dir=""

printf '[full-motion-finalize-release-watcher] complete: %s\n' \
  "${terminal_receipt}"
