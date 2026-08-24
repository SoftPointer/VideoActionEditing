#!/usr/bin/env bash
set -euo pipefail
umask 077

readonly CONTROL_ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/crosscase_target_graph_teacher_sam2_v1_20260822_r1_control"
readonly OUTPUT_ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/crosscase_target_graph_teacher_sam2_v1_20260822_r1"
readonly PROGRAM="${CONTROL_ROOT}/extract_crosscase_target_graph_teacher_sam2_v1.py"
readonly PYTHON="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
readonly PROGRAM_SHA256="f20345fdbbd94bd1eb11e929687840a754cf16d51d2d7c85168c7a98ae3ca3ea"
readonly PYTHON_SHA256="8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"

fail() {
  echo "[crosscase-target-graph-teacher] ERROR: $*" >&2
  exit 1
}

verify_file() {
  local path="$1"
  local expected="$2"
  local observed ignored
  [[ -f "${path}" && ! -L "${path}" ]] || fail "authority is not a regular file: ${path}"
  read -r observed ignored < <(/usr/bin/sha256sum "${path}")
  [[ "${observed}" == "${expected}" ]] || fail "authority hash differs: ${path}"
}

launch_case() {
  local case_id="$1"
  local job_id="$2"
  local node="$3"
  local spec_sha256="$4"
  local pid_variable="$5"
  local spec="${CONTROL_ROOT}/${case_id}_target_graph_teacher_sam2_spec_v1.json"
  local output="${OUTPUT_ROOT}/${case_id}"
  local cache="/tmp/crosscase-target-graph-teacher-r1-${case_id}-${job_id}"
  local log="${CONTROL_ROOT}/${case_id}.srun.log"
  verify_file "${spec}" "${spec_sha256}"
  [[ ! -e "${output}" && ! -L "${output}" ]] || fail "fresh case output required: ${output}"
  [[ ! -e "${cache}" && ! -L "${cache}" ]] || fail "fresh cache required: ${cache}"
  /usr/bin/srun \
    --jobid="${job_id}" \
    --nodes=1 \
    --nodelist="${node}" \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=32G \
    --gres=gpu:1 \
    --exclusive \
    --exact \
    --time=01:30:00 \
    --job-name="tg-${case_id}" \
    /usr/bin/env -i \
      PATH=/usr/bin:/bin \
      LC_ALL=C \
      LANG=C \
      HOME=/vast/users/guangyi.chen \
      PROGRAM="${PROGRAM}" \
      SPEC="${spec}" \
      OUTPUT="${output}" \
      CACHE="${cache}" \
      PYTHON="${PYTHON}" \
      /bin/bash -c '
set -euo pipefail
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
exec "${PYTHON}" -I -B "${PROGRAM}" --spec "${SPEC}" --output-dir "${OUTPUT}"
' >"${log}" 2>&1 &
  printf -v "${pid_variable}" '%s' "$!"
}

[[ "$#" -eq 0 ]] || fail "launcher accepts no arguments"
verify_file "${PROGRAM}" "${PROGRAM_SHA256}"
verify_file "${PYTHON}" "${PYTHON_SHA256}"
[[ ! -e "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]] || fail "fresh output root required"
/usr/bin/mkdir -p "${OUTPUT_ROOT}"

pid_8b05=""
pid_407=""
pid_5e83=""
launch_case 8b05aaf463db 147881 auh7-1b-gpu-213 a4cc4fe777c47c83161cb2a79fd4a5bc74f73d8912a187ea4b3659ad39ddc853 pid_8b05
launch_case 40712e1341dc 147873 auh7-1b-gpu-284 47a5c8a14aa48d8b54cd1e98e655dd0691f97474c3a71c77f1742c024ec269c5 pid_407
launch_case 5e83a9279951 147871 auh7-1b-gpu-232 309f4d98fb4b44293ed58eed74252a11d60e87e34bf535873845e6300eb0c2d4 pid_5e83

status=0
wait "${pid_8b05}" || status=1
wait "${pid_407}" || status=1
wait "${pid_5e83}" || status=1
exit "${status}"
