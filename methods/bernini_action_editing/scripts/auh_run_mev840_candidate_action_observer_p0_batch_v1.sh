#!/usr/bin/env bash
set -euo pipefail
umask 077

readonly CONTROL_ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev840_candidate_action_observer_v1_20260822_control"
readonly RUNNER="${CONTROL_ROOT}/run_mev840_candidate_action_observer_batch_v1.py"
readonly EXTRACTOR="${CONTROL_ROOT}/extract_mev840_coordinate_free_action_oracle_v1.py"
readonly ORACLE="${CONTROL_ROOT}/mev840_coordinate_free_action_oracle_v1.py"
readonly MANIFEST="${CONTROL_ROOT}/mev840_candidate_action_observer_p0_batch_r2.json"
readonly OUTPUT_ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev840_candidate_action_observer_p0_v1_20260822_r3"
readonly PYTHON="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
readonly JOB_ID="143808"
readonly NODE="auh7-1b-gpu-233"
readonly RUNNER_SHA256="e967c755b889f1fb9221bde9cdc0642276f96dea660c241ff84c678d6dbf50b5"
readonly EXTRACTOR_SHA256="f40649ff9b86f82f1dd4bfaa41a72e2a4c98ce86bf31b8f11dccbe41c35c2da2"
readonly ORACLE_SHA256="ed00d7d0c0050b2697e8052f085eb6dc9cb6d3872fb2948db2fd26d8b1ddcaae"
readonly MANIFEST_SHA256="4d80fb321cb2e6a24861277416633ca7a550c29e77d654c2d35dfc7d9fdc3d0c"
readonly PYTHON_SHA256="8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"

fail() {
  echo "[mev840-candidate-action-p0] ERROR: $*" >&2
  exit 1
}

[[ "$#" -eq 0 ]] || fail "launcher accepts no arguments"
[[ ! -e "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]] || fail "fresh output required"
for pair in \
  "${RUNNER}:${RUNNER_SHA256}" \
  "${EXTRACTOR}:${EXTRACTOR_SHA256}" \
  "${ORACLE}:${ORACLE_SHA256}" \
  "${MANIFEST}:${MANIFEST_SHA256}" \
  "${PYTHON}:${PYTHON_SHA256}"
do
  path="${pair%:*}"
  expected="${pair##*:}"
  [[ -f "${path}" && ! -L "${path}" ]] || fail "authority differs: ${path}"
  read -r observed ignored < <(/usr/bin/sha256sum "${path}")
  [[ "${observed}" == "${expected}" ]] || fail "authority hash differs: ${path}"
done

readonly CACHE="/tmp/mev840-candidate-action-p0-r3-${JOB_ID}"
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
  --time=00:10:00 \
  --job-name=mev840-cand-action-p0-v1 \
  /usr/bin/env -i \
    PATH=/usr/bin:/bin \
    LC_ALL=C \
    LANG=C \
    HOME=/vast/users/guangyi.chen \
    CACHE="${CACHE}" \
    PYTHON="${PYTHON}" \
    RUNNER="${RUNNER}" \
    MANIFEST="${MANIFEST}" \
    /bin/bash -c '
set -euo pipefail
[[ ! -e "${CACHE}" && ! -L "${CACHE}" ]]
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
exec "${PYTHON}" -I -B "${RUNNER}" --manifest "${MANIFEST}"
'
