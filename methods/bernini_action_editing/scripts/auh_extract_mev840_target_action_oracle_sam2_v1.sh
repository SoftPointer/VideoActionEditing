#!/usr/bin/env bash
set -euo pipefail
umask 077

readonly CONTROL_ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev840_target_action_oracle_sam2_v1_20260822_r2_control"
readonly OUTPUT_ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev840_target_action_oracle_sam2_v1_20260822_r2"
readonly PROGRAM="${CONTROL_ROOT}/extract_mev840_coordinate_free_action_oracle_v1.py"
readonly ORACLE="${CONTROL_ROOT}/mev840_coordinate_free_action_oracle_v1.py"
readonly SPEC="${CONTROL_ROOT}/mev840_target_frozen_sam2_action_observer_spec_v1.json"
readonly PYTHON="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
readonly JOB_ID="143808"
readonly NODE="auh7-1b-gpu-233"
readonly PROGRAM_SHA256="5629025b6c6d2369fc2d3cd3c28a1be790ed26ad88f1f6f73ffdf8ac4398e582"
readonly ORACLE_SHA256="9c1ed616af59a52e66c8414478b143943f9b808231cec8529a5a10345a2c5654"
readonly SPEC_SHA256="6834200d5bf4b8098cf4958a1939f3255ea461c15122d9ebd7e6b369f39ae3ad"
readonly PYTHON_SHA256="8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"

fail() {
  echo "[mev840-target-action-sam2] ERROR: $*" >&2
  exit 1
}

[[ "$#" -eq 0 ]] || fail "launcher accepts no arguments"
[[ ! -e "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]] || fail "fresh output required"
for pair in \
  "${PROGRAM}:${PROGRAM_SHA256}" \
  "${ORACLE}:${ORACLE_SHA256}" \
  "${SPEC}:${SPEC_SHA256}" \
  "${PYTHON}:${PYTHON_SHA256}"
do
  path="${pair%:*}"
  expected="${pair##*:}"
  [[ -f "${path}" && ! -L "${path}" ]] || fail "authority is not a regular file: ${path}"
  read -r observed ignored < <(/usr/bin/sha256sum "${path}")
  [[ "${observed}" == "${expected}" ]] || fail "authority hash differs: ${path}"
done

readonly CACHE="/tmp/mev840-target-action-sam2-r2-${JOB_ID}"
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
  --time=00:30:00 \
  --job-name=mev840-target-action-sam2-v1 \
  /usr/bin/env -i \
    PATH=/usr/bin:/bin \
    LC_ALL=C \
    LANG=C \
    HOME=/vast/users/guangyi.chen \
    PROGRAM="${PROGRAM}" \
    SPEC="${SPEC}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    CACHE="${CACHE}" \
    PYTHON="${PYTHON}" \
    /bin/bash -c '
set -euo pipefail
[[ ! -e "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]]
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
exec "${PYTHON}" -I -B "${PROGRAM}" --spec "${SPEC}" --output-dir "${OUTPUT_ROOT}"
'
