#!/usr/bin/env bash
set -Eeuo pipefail

archive="${MOTIVE_R10A_LOCAL_ARCHIVE:?set MOTIVE_R10A_LOCAL_ARCHIVE}"
archive_sha256="${MOTIVE_R10A_LOCAL_ARCHIVE_SHA256:?set MOTIVE_R10A_LOCAL_ARCHIVE_SHA256}"
experiment_root="${MOTIVE_R10A_REMOTE_EXPERIMENT_ROOT:?set MOTIVE_R10A_REMOTE_EXPERIMENT_ROOT}"
source_tree_sha256="${MOTIVE_R10A_SOURCE_TREE_SHA256:?set MOTIVE_R10A_SOURCE_TREE_SHA256}"
parent_run="${MOTIVE_R10A_PARENT_RUN:?set MOTIVE_R10A_PARENT_RUN}"
model_workspace="${MOTIVE_R10A_MODEL_WORKSPACE:?set MOTIVE_R10A_MODEL_WORKSPACE}"
python_bin="${MOTIVE_R10A_PYTHON_BIN:?set MOTIVE_R10A_PYTHON_BIN}"
remote_submit_script="${MOTIVE_R10A_REMOTE_SUBMIT_SCRIPT:?set MOTIVE_R10A_REMOTE_SUBMIT_SCRIPT}"
attempts="${MOTIVE_R10A_CONNECT_ATTEMPTS:-120}"
delay_seconds="${MOTIVE_R10A_CONNECT_DELAY_SECONDS:-30}"
ssh_host="${MOTIVE_R10A_SSH_HOST:-auh}"
script_dir="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
  pwd -P
)"
watcher_script="${MOTIVE_R10A_WATCHER_SCRIPT:-${script_dir}/watch_auh_r10a_jobs.sh}"
allowed_prefix=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto

case "${experiment_root}" in
  "${allowed_prefix}"/goku_repr_auto_r10a_*) ;;
  *)
    echo "[r10a-retry] unsafe experiment root: ${experiment_root}" >&2
    exit 2
    ;;
esac

if [[ ! -s "${archive}" ]] || [[ -L "${archive}" ]]; then
  echo "[r10a-retry] archive is missing/empty/symlinked: ${archive}" >&2
  exit 2
fi
for digest in "${archive_sha256}" "${source_tree_sha256}"; do
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[r10a-retry] invalid SHA-256: ${digest}" >&2
    exit 2
  fi
done
if [[ ! "${attempts}" =~ ^[0-9]+$ ]] || (( attempts < 1 )); then
  echo "[r10a-retry] invalid attempt count" >&2
  exit 2
fi
if [[ ! "${delay_seconds}" =~ ^[0-9]+$ ]] || (( delay_seconds < 1 )); then
  echo "[r10a-retry] invalid retry delay" >&2
  exit 2
fi
if [[ ! -f "${remote_submit_script}" ]] \
  || [[ ! -s "${remote_submit_script}" ]] \
  || [[ -L "${remote_submit_script}" ]]; then
  echo "[r10a-retry] remote submit script is missing" >&2
  exit 2
fi
if [[ ! -f "${watcher_script}" ]] \
  || [[ ! -s "${watcher_script}" ]] \
  || [[ -L "${watcher_script}" ]] \
  || [[ ! -x "${watcher_script}" ]]; then
  echo "[r10a-retry] lifecycle watcher is invalid: ${watcher_script}" >&2
  exit 2
fi

probe_auh() {
  ssh -o ConnectTimeout=10 -o BatchMode=yes "${ssh_host}" true
}

submit_once() {
  ssh "${ssh_host}" bash -s -- \
    "${experiment_root}" \
    "${allowed_prefix}" \
    "${python_bin}" \
    ensure-root <<'REMOTE' || return 1
set -Eeuo pipefail
root="$1"
allowed_prefix="$2"
python_bin="$3"
operation="$4"
[[ "${operation}" == ensure-root ]]
"${python_bin}" - "${root}" "${allowed_prefix}" <<'PY'
import os
from pathlib import Path
import stat
import sys

raw_root, raw_prefix = sys.argv[1:]
for raw, context in (
    (raw_root, "experiment root"),
    (raw_prefix, "allowed prefix"),
):
    if (
        not os.path.isabs(raw)
        or raw.startswith("//")
        or raw != os.path.normpath(raw)
    ):
        raise SystemExit(f"{context} is not canonical")
root = Path(raw_root)
prefix = Path(raw_prefix)
if (
    prefix.is_symlink()
    or not prefix.is_dir()
    or prefix.resolve(strict=True) != prefix
):
    raise SystemExit("allowed prefix is redirected")
if (
    root.parent != prefix
    or not root.name.startswith("goku_repr_auto_r10a_")
):
    raise SystemExit("experiment root is outside the allowed prefix")
try:
    metadata = root.lstat()
except FileNotFoundError:
    os.mkdir(root, 0o755)
    metadata = root.lstat()
if (
    stat.S_ISLNK(metadata.st_mode)
    or not stat.S_ISDIR(metadata.st_mode)
    or root.resolve(strict=True) != root
):
    raise SystemExit("experiment root is redirected or non-directory")
PY
REMOTE
  local remote_archive
  remote_archive="${experiment_root}/source_snapshot.${archive_sha256}.tar.gz"
  if ssh "${ssh_host}" "test -L '${remote_archive}'"; then
    echo "[r10a-retry] remote archive path is symlinked" >&2
    return 1
  fi
  if ssh "${ssh_host}" "test -s '${remote_archive}'"; then
    local remote_archive_sha256
    remote_archive_sha256="$(
      ssh "${ssh_host}" \
        "sha256sum '${remote_archive}' | awk '{print \$1}'"
    )" || return 1
    if [[ "${remote_archive_sha256}" != "${archive_sha256}" ]]; then
      echo \
        "[r10a-retry] existing archive has the wrong content digest" \
        >&2
      return 1
    fi
  else
    local remote_temporary
    remote_temporary="${remote_archive}.tmp.$$"
    if ssh "${ssh_host}" \
      "test ! -e '${remote_temporary}' && test ! -L '${remote_temporary}'"; then
      :
    else
      echo "[r10a-retry] remote archive temporary path is occupied" >&2
      return 1
    fi
    scp -q "${archive}" "${ssh_host}:${remote_temporary}" || return 1
    ssh "${ssh_host}" "
      set -eu
      actual=\$(sha256sum '${remote_temporary}' | awk '{print \$1}')
      test \"\${actual}\" = '${archive_sha256}'
      test ! -e '${remote_archive}'
      mv '${remote_temporary}' '${remote_archive}'
    " || return 1
  fi

  if ssh "${ssh_host}" \
    "test -L '${experiment_root}/source_snapshot'"; then
    echo "[r10a-retry] remote source snapshot path is symlinked" >&2
    return 1
  fi
  if ! ssh "${ssh_host}" \
    "test -d '${experiment_root}/source_snapshot'"; then
    ssh "${ssh_host}" "
      set -eu
      stage=\$(mktemp -d \
        '${experiment_root}/.source_snapshot.extract.XXXXXXXX')
      tar --delay-directory-restore -xzf '${remote_archive}' \
        -C \"\${stage}\"
      test -d \"\${stage}/source_snapshot\"
      find \"\${stage}/source_snapshot\" -depth -type d \
        -exec chmod 0555 {} +
      '${python_bin}' \
        \"\${stage}/source_snapshot/methods/motive/scripts/action_source_snapshot.py\" \
        verify \
        --snapshot \"\${stage}/source_snapshot\" \
        --expected-tree-sha256 '${source_tree_sha256}'
      test ! -e '${experiment_root}/source_snapshot'
      mv \"\${stage}/source_snapshot\" \
        '${experiment_root}/source_snapshot'
      rmdir \"\${stage}\"
    " || return 1
  fi
  ssh "${ssh_host}" "
    '${python_bin}' \
      '${experiment_root}/source_snapshot/methods/motive/scripts/action_source_snapshot.py' \
      verify \
      --snapshot '${experiment_root}/source_snapshot' \
      --expected-tree-sha256 '${source_tree_sha256}'
  " || return 1
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
  echo "[r10a-retry] remote submission receipt is invalid" >&2
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

for ((attempt = 1; attempt <= attempts; attempt++)); do
  echo \
    "[r10a-retry] attempt=${attempt}/${attempts} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if probe_auh; then
    set +e
    submit_once
    submit_status=$?
    set -e
    if (( submit_status == 0 )); then
      set +e
      job_ids_csv="$(read_remote_job_ids)"
      receipt_status=$?
      set -e
      if (( receipt_status == 255 )); then
        echo \
          "[r10a-retry] receipt read lost connectivity; retaining state" \
          >&2
        if (( attempt < attempts )); then
          sleep "${delay_seconds}"
        fi
        continue
      fi
      if (( receipt_status != 0 )); then
        echo \
          "[r10a-retry] submitted jobs lack a strict remote receipt" \
          >&2
        exit 3
      fi
      if [[ ! "${job_ids_csv}" =~ ^[0-9]+,[0-9]+$ ]] \
        || [[ "${job_ids_csv%%,*}" == "${job_ids_csv#*,}" ]]; then
        echo "[r10a-retry] invalid coordinated job IDs" >&2
        exit 3
      fi
      echo \
        "[r10a-retry] submission state coordinated; exec lifecycle watcher " \
        "jobs=${job_ids_csv}"
      export MOTIVE_R10A_SSH_HOST="${ssh_host}"
      export MOTIVE_R10A_EXPERIMENT_ROOT="${experiment_root}"
      export MOTIVE_R10A_JOB_IDS="${job_ids_csv}"
      export MOTIVE_R10A_PYTHON_BIN="${python_bin}"
      export MOTIVE_R10A_SOURCE_TREE_SHA256="${source_tree_sha256}"
      export MOTIVE_R10A_PARENT_RUN="${parent_run}"
      export MOTIVE_R10A_MODEL_WORKSPACE="${model_workspace}"
      export MOTIVE_R10A_REMOTE_SUBMIT_SCRIPT="${remote_submit_script}"
      exec "${watcher_script}"
    fi
    if (( submit_status == 2 || submit_status == 3 \
      || submit_status == 6 \
      || submit_status == 7 )); then
      echo \
        "[r10a-retry] deterministic remote coordination failure; " \
        "not retrying blindly (status=${submit_status})" >&2
      exit "${submit_status}"
    fi
  fi
  echo "[r10a-retry] attempt=${attempt} did not complete" >&2
  if (( attempt < attempts )); then
    sleep "${delay_seconds}"
  fi
done

echo "[r10a-retry] AUH remained unreachable for all attempts" >&2
exit 75
