#!/usr/bin/env bash
set -Eeuo pipefail

archive="${MOTIVE_R9_LOCAL_ARCHIVE:?set MOTIVE_R9_LOCAL_ARCHIVE}"
archive_sha256="${MOTIVE_R9_LOCAL_ARCHIVE_SHA256:?set MOTIVE_R9_LOCAL_ARCHIVE_SHA256}"
experiment_root="${MOTIVE_R9_REMOTE_EXPERIMENT_ROOT:?set MOTIVE_R9_REMOTE_EXPERIMENT_ROOT}"
source_tree_sha256="${MOTIVE_R9_SOURCE_TREE_SHA256:?set MOTIVE_R9_SOURCE_TREE_SHA256}"
parent_run="${MOTIVE_R9_PARENT_RUN:?set MOTIVE_R9_PARENT_RUN}"
model_workspace="${MOTIVE_R9_MODEL_WORKSPACE:?set MOTIVE_R9_MODEL_WORKSPACE}"
python_bin="${MOTIVE_R9_PYTHON_BIN:?set MOTIVE_R9_PYTHON_BIN}"
remote_submit_script="${MOTIVE_R9_REMOTE_SUBMIT_SCRIPT:?set MOTIVE_R9_REMOTE_SUBMIT_SCRIPT}"
attempts="${MOTIVE_R9_CONNECT_ATTEMPTS:-120}"
delay_seconds="${MOTIVE_R9_CONNECT_DELAY_SECONDS:-30}"

if [[ ! -s "${archive}" ]] || [[ -L "${archive}" ]]; then
  echo "[r9-retry] archive is missing/empty/symlinked: ${archive}" >&2
  exit 2
fi
for digest in "${archive_sha256}" "${source_tree_sha256}"; do
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[r9-retry] invalid SHA-256: ${digest}" >&2
    exit 2
  fi
done
if [[ ! "${attempts}" =~ ^[0-9]+$ ]] || (( attempts < 1 )); then
  echo "[r9-retry] invalid attempt count" >&2
  exit 2
fi
if [[ ! "${delay_seconds}" =~ ^[0-9]+$ ]] || (( delay_seconds < 1 )); then
  echo "[r9-retry] invalid retry delay" >&2
  exit 2
fi
if [[ ! -s "${remote_submit_script}" ]]; then
  echo "[r9-retry] remote submit script is missing" >&2
  exit 2
fi

probe_auh() {
  ssh -o ConnectTimeout=10 -o BatchMode=yes auh true &
  local probe_pid=$!
  local tick
  for ((tick = 0; tick < 20; tick++)); do
    if ! kill -0 "${probe_pid}" 2>/dev/null; then
      if wait "${probe_pid}"; then
        return 0
      fi
      return 1
    fi
    sleep 1
  done
  kill "${probe_pid}" 2>/dev/null || true
  wait "${probe_pid}" 2>/dev/null || true
  return 1
}

submit_once() {
  ssh auh "mkdir -p '${experiment_root}'" || return 1
  local remote_archive
  remote_archive="${experiment_root}/source_snapshot.${archive_sha256}.tar.gz"
  if ssh auh "test -s '${remote_archive}'"; then
    local remote_archive_sha256
    remote_archive_sha256="$(
      ssh auh "sha256sum '${remote_archive}' | awk '{print \$1}'"
    )" || return 1
    if [[ "${remote_archive_sha256}" != "${archive_sha256}" ]]; then
      echo \
        "[r9-retry] existing content-addressed archive has the wrong SHA" \
        >&2
      return 1
    fi
  else
    local remote_temporary
    remote_temporary="${remote_archive}.tmp.$$"
    scp -q "${archive}" "auh:${remote_temporary}" || return 1
    ssh auh "
      set -eu
      actual=\$(sha256sum '${remote_temporary}' | awk '{print \$1}')
      test \"\${actual}\" = '${archive_sha256}'
      test ! -e '${remote_archive}'
      mv '${remote_temporary}' '${remote_archive}'
    " || return 1
  fi

  if ! ssh auh "test -d '${experiment_root}/source_snapshot'"; then
    ssh auh "
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
  ssh auh "
    '${python_bin}' \
      '${experiment_root}/source_snapshot/methods/motive/scripts/action_source_snapshot.py' \
      verify \
      --snapshot '${experiment_root}/source_snapshot' \
      --expected-tree-sha256 '${source_tree_sha256}'
  " || return 1
  ssh auh bash -s -- \
    "${experiment_root}" \
    "${source_tree_sha256}" \
    "${parent_run}" \
    "${model_workspace}" \
    "${python_bin}" \
    < "${remote_submit_script}" || return 1
}

for ((attempt = 1; attempt <= attempts; attempt++)); do
  echo "[r9-retry] attempt=${attempt}/${attempts} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if probe_auh && submit_once; then
    echo "[r9-retry] submission completed"
    exit 0
  fi
  echo "[r9-retry] attempt=${attempt} did not reach a verified submission" >&2
  if (( attempt < attempts )); then
    sleep "${delay_seconds}"
  fi
done

echo "[r9-retry] AUH remained unreachable for all attempts" >&2
exit 75
