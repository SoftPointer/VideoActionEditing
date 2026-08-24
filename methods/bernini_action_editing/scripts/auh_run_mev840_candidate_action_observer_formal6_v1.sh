#!/usr/bin/env bash
set -euo pipefail
umask 077

readonly CONTROL_ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev840_candidate_action_observer_v1_20260822_control"
readonly RUNNER="${CONTROL_ROOT}/run_mev840_candidate_action_observer_batch_v1.py"
readonly EXTRACTOR="${CONTROL_ROOT}/extract_mev840_coordinate_free_action_oracle_v1.py"
readonly ORACLE="${CONTROL_ROOT}/mev840_coordinate_free_action_oracle_v1.py"
readonly AUDITOR="${CONTROL_ROOT}/audit_mev840_candidate_action_observer_formal6_v1.py"
readonly MANIFEST="${CONTROL_ROOT}/mev840_candidate_action_observer_formal6_v1.json"
readonly OUTPUT_ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev840_native_target_action_matrix_observer_formal6_v1_20260822"
readonly PYTHON="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
readonly JOB_ID="143808"
readonly NODE="auh7-1b-gpu-292"
readonly RUNNER_SHA256="e967c755b889f1fb9221bde9cdc0642276f96dea660c241ff84c678d6dbf50b5"
readonly EXTRACTOR_SHA256="f40649ff9b86f82f1dd4bfaa41a72e2a4c98ce86bf31b8f11dccbe41c35c2da2"
readonly ORACLE_SHA256="ed00d7d0c0050b2697e8052f085eb6dc9cb6d3872fb2948db2fd26d8b1ddcaae"
readonly AUDITOR_SHA256="d6d79810bffef62f5821ed861d69f13a7416efbfa938b919e01579070afe3de3"
readonly MANIFEST_SHA256="57fe3594a1a23184e862d29afcf68417f0571bb9039eea8eab8740051a73226d"
readonly PYTHON_SHA256="8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"

fail() {
  echo "[mev840-candidate-action-formal6] ERROR: $*" >&2
  exit 1
}

verify_control() {
  local pair path expected observed ignored
  for pair in \
    "${RUNNER}:${RUNNER_SHA256}" \
    "${EXTRACTOR}:${EXTRACTOR_SHA256}" \
    "${ORACLE}:${ORACLE_SHA256}" \
    "${AUDITOR}:${AUDITOR_SHA256}" \
    "${MANIFEST}:${MANIFEST_SHA256}" \
    "${PYTHON}:${PYTHON_SHA256}"
  do
    path="${pair%:*}"
    expected="${pair##*:}"
    [[ -f "${path}" && ! -L "${path}" ]] || fail "authority differs: ${path}"
    read -r observed ignored < <(/usr/bin/sha256sum "${path}")
    [[ "${observed}" == "${expected}" ]] || fail "authority hash differs: ${path}"
  done
}

[[ "$#" -eq 0 ]] || fail "launcher accepts no arguments"
[[ ! -e "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]] || fail "fresh output required"
verify_control
"${PYTHON}" -I -B "${AUDITOR}" \
  --manifest "${MANIFEST}" \
  --runner "${RUNNER}" \
  --expected-runner-sha256 "${RUNNER_SHA256}"

readonly CACHE="/tmp/mev840-candidate-action-formal6-v1-${JOB_ID}"
exec /usr/bin/srun \
  --jobid="${JOB_ID}" \
  --nodes=1 \
  --nodelist="${NODE}" \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=32G \
  --gres=gpu:1 \
  --exclusive \
  --exact \
  --time=00:15:00 \
  --job-name=mev840-formal6-observer \
  /usr/bin/env -i \
    PATH=/usr/bin:/bin \
    LC_ALL=C \
    LANG=C \
    HOME=/vast/users/guangyi.chen \
    CACHE="${CACHE}" \
    PYTHON="${PYTHON}" \
    RUNNER="${RUNNER}" \
    EXTRACTOR="${EXTRACTOR}" \
    ORACLE="${ORACLE}" \
    AUDITOR="${AUDITOR}" \
    MANIFEST="${MANIFEST}" \
    RUNNER_SHA256="${RUNNER_SHA256}" \
    EXTRACTOR_SHA256="${EXTRACTOR_SHA256}" \
    ORACLE_SHA256="${ORACLE_SHA256}" \
    AUDITOR_SHA256="${AUDITOR_SHA256}" \
    MANIFEST_SHA256="${MANIFEST_SHA256}" \
    PYTHON_SHA256="${PYTHON_SHA256}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    /bin/bash -c '
set -euo pipefail
[[ ! -e "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]]
[[ ! -e "${CACHE}" && ! -L "${CACHE}" ]]
for pair in \
  "${RUNNER}:${RUNNER_SHA256}" \
  "${EXTRACTOR}:${EXTRACTOR_SHA256}" \
  "${ORACLE}:${ORACLE_SHA256}" \
  "${AUDITOR}:${AUDITOR_SHA256}" \
  "${MANIFEST}:${MANIFEST_SHA256}" \
  "${PYTHON}:${PYTHON_SHA256}"
do
  path="${pair%:*}"
  expected="${pair##*:}"
  [[ -f "${path}" && ! -L "${path}" ]]
  read -r observed ignored < <(/usr/bin/sha256sum "${path}")
  [[ "${observed}" == "${expected}" ]]
done
/usr/bin/mkdir -p \
  "${CACHE}/home" \
  "${CACHE}/tmp" \
  "${CACHE}/xdg" \
  "${CACHE}/pycache" \
  "${CACHE}/torch" \
  "${CACHE}/miopen-user" \
  "${CACHE}/miopen-custom" \
  "${CACHE}/triton" \
  "${CACHE}/torch-extensions" \
  "${CACHE}/inductor"
/usr/bin/chmod 0700 "${CACHE}" "${CACHE}"/*
export HOME="${CACHE}/home"
export TMPDIR="${CACHE}/tmp"
export XDG_CACHE_HOME="${CACHE}/xdg"
export PYTHONPYCACHEPREFIX="${CACHE}/pycache"
export TORCH_HOME="${CACHE}/torch"
export MIOPEN_USER_DB_PATH="${CACHE}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${CACHE}/miopen-custom"
export TRITON_CACHE_DIR="${CACHE}/triton"
export TORCH_EXTENSIONS_DIR="${CACHE}/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="${CACHE}/inductor"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
"${PYTHON}" -I -B "${AUDITOR}" \
  --manifest "${MANIFEST}" \
  --runner "${RUNNER}" \
  --expected-runner-sha256 "${RUNNER_SHA256}"
exec "${PYTHON}" -I -B -u "${RUNNER}" --manifest "${MANIFEST}"
'
