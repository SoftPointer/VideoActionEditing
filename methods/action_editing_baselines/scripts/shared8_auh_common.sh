#!/usr/bin/env bash

# Shared fail-closed setup for the AUH action-editing shared-8 launchers.
# This file is sourced by the sbatch entry points; it is not a job itself.

shared8_fail() {
  printf '[shared8-launch] ERROR: %s\n' "$*" >&2
  exit 2
}
shared8_require_sha256() {
  local value="$1"
  local label="$2"
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || shared8_fail "${label} must be one lowercase SHA-256 digest"
}

shared8_require_revision() {
  local value="$1"
  local label="$2"
  [[ "${value}" =~ ^[0-9a-f]{40}$ ]] || shared8_fail "${label} must be one lowercase 40-hex revision"
}

shared8_require_plain_file() {
  local path="$1"
  local label="$2"
  [[ -f "${path}" && ! -L "${path}" ]] || shared8_fail "${label} is not a plain file: ${path}"
}

shared8_require_directory() {
  local path="$1"
  local label="$2"
  [[ -d "${path}" ]] || shared8_fail "${label} is not a directory: ${path}"
}

shared8_configure_worker_cache() {
  local cache_root="$1"
  [[ "${cache_root}" == "${SHARED8_SCRATCH_ROOT}/"* ]] || shared8_fail "worker cache escaped job scratch"
  [[ ! -e "${cache_root}" && ! -L "${cache_root}" ]] || shared8_fail "worker cache already exists: ${cache_root}"

  mkdir -p \
    "${cache_root}/tmp" \
    "${cache_root}/xdg" \
    "${cache_root}/hf/hub" \
    "${cache_root}/transformers" \
    "${cache_root}/torch" \
    "${cache_root}/torch-extensions" \
    "${cache_root}/torch-inductor" \
    "${cache_root}/triton" \
    "${cache_root}/miopen-user" \
    "${cache_root}/miopen-custom"

  export TMPDIR="${cache_root}/tmp"
  export XDG_CACHE_HOME="${cache_root}/xdg"
  export HF_HOME="${cache_root}/hf"
  export HUGGINGFACE_HUB_CACHE="${cache_root}/hf/hub"
  export TRANSFORMERS_CACHE="${cache_root}/transformers"
  export TORCH_HOME="${cache_root}/torch"
  export TORCH_EXTENSIONS_DIR="${cache_root}/torch-extensions"
  export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torch-inductor"
  export TRITON_CACHE_DIR="${cache_root}/triton"
  export MIOPEN_USER_DB_PATH="${cache_root}/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${cache_root}/miopen-custom"
}

shared8_init_common() {
  local launcher_label="$1"
  [[ "${SHARED8_COMMON_INITIALIZED:-0}" == 0 ]] || shared8_fail "common setup was initialized twice"
  [[ -n "${SLURM_JOB_ID:-}" ]] || shared8_fail "this launcher must run inside a Slurm job"

  local source_archive="${ACTION_BASELINE_SOURCE_ARCHIVE:?set ACTION_BASELINE_SOURCE_ARCHIVE}"
  local source_archive_sha256="${ACTION_BASELINE_SOURCE_ARCHIVE_SHA256:?set ACTION_BASELINE_SOURCE_ARCHIVE_SHA256}"
  local source_revision="${ACTION_BASELINE_SOURCE_REVISION:?set ACTION_BASELINE_SOURCE_REVISION}"
  local manifest="${ACTION_BASELINE_MANIFEST:?set ACTION_BASELINE_MANIFEST}"
  local manifest_sha256="${ACTION_BASELINE_MANIFEST_SHA256:?set ACTION_BASELINE_MANIFEST_SHA256}"
  local output_root="${ACTION_BASELINE_OUTPUT_ROOT:?set ACTION_BASELINE_OUTPUT_ROOT}"
  local python_bin="${ACTION_BASELINE_PYTHON_BIN:?set ACTION_BASELINE_PYTHON_BIN}"
  local requested_ffprobe="${ACTION_BASELINE_FFPROBE:-}"

  shared8_require_plain_file "${source_archive}" "method source archive"
  shared8_require_plain_file "${manifest}" "source-only input manifest"
  shared8_require_sha256 "${source_archive_sha256}" "method source archive hash"
  shared8_require_sha256 "${manifest_sha256}" "input manifest hash"
  shared8_require_revision "${source_revision}" "method source revision"
  [[ -x "${python_bin}" ]] || shared8_fail "Python is not executable: ${python_bin}"
  [[ "${output_root}" == /* ]] || shared8_fail "output root must be absolute"
  [[ ! -L "${output_root}" ]] || shared8_fail "output root may not be a symbolic link"

  local observed_archive_sha256
  local observed_manifest_sha256
  observed_archive_sha256="$(sha256sum "${source_archive}" | awk '{print $1}')"
  observed_manifest_sha256="$(sha256sum "${manifest}" | awk '{print $1}')"
  [[ "${observed_archive_sha256}" == "${source_archive_sha256}" ]] || shared8_fail "method source archive hash mismatch"
  [[ "${observed_manifest_sha256}" == "${manifest_sha256}" ]] || shared8_fail "input manifest hash mismatch"

  umask 077
  local scratch_parent="${SLURM_TMPDIR:-/tmp}"
  local array_component="${SLURM_ARRAY_TASK_ID:-batch}"
  SHARED8_SCRATCH_ROOT="${scratch_parent}/action-shared8-${launcher_label}-${SLURM_JOB_ID}-${array_component}"
  [[ ! -e "${SHARED8_SCRATCH_ROOT}" && ! -L "${SHARED8_SCRATCH_ROOT}" ]] || shared8_fail "job scratch already exists: ${SHARED8_SCRATCH_ROOT}"
  mkdir -p "${SHARED8_SCRATCH_ROOT}/source"

  local archive_member
  while IFS= read -r archive_member; do
    [[ "${archive_member}" != /* ]] || shared8_fail "source archive contains an absolute path"
    [[ "/${archive_member}/" != *'/../'* ]] || shared8_fail "source archive contains a parent traversal"
  done < <(tar -tf "${source_archive}")
  tar -xf "${source_archive}" -C "${SHARED8_SCRATCH_ROOT}/source"

  SHARED8_SOURCE_ROOT="${SHARED8_SCRATCH_ROOT}/source"
  SHARED8_RUNNER="${SHARED8_SOURCE_ROOT}/methods/action_editing_baselines/run_shared8.py"
  local pyav_ffprobe="${SHARED8_SOURCE_ROOT}/methods/action_editing_baselines/ffprobe_pyav_compat.py"
  [[ -f "${SHARED8_RUNNER}" && ! -L "${SHARED8_RUNNER}" ]] || shared8_fail "source archive lacks the shared-8 runner"
  [[ -f "${SHARED8_SOURCE_ROOT}/methods/action_editing_baselines/shared8_contract.py" ]] || shared8_fail "source archive lacks the shared-8 contract"

  local ffprobe_bin
  local probe_backend
  if [[ -n "${requested_ffprobe}" ]]; then
    ffprobe_bin="$(command -v -- "${requested_ffprobe}" 2>/dev/null || true)"
    [[ -n "${ffprobe_bin}" ]] || shared8_fail "requested ffprobe is not executable: ${requested_ffprobe}"
    probe_backend="explicit-ffprobe"
  else
    [[ -f "${pyav_ffprobe}" && ! -L "${pyav_ffprobe}" && -x "${pyav_ffprobe}" ]] || \
      shared8_fail "source archive lacks an executable PyAV ffprobe backend"
    ffprobe_bin="${pyav_ffprobe}"
    probe_backend="frozen-pyav"
  fi

  mkdir -p "${output_root}" "${output_root}/launcher_logs"
  SHARED8_LAUNCHER_LOG_ROOT="${output_root}/launcher_logs"
  SHARED8_PYTHON_BIN="${python_bin}"
  SHARED8_MANIFEST="${manifest}"
  SHARED8_MANIFEST_SHA256="${manifest_sha256}"
  SHARED8_OUTPUT_ROOT="${output_root}"
  SHARED8_FFPROBE_BIN="${ffprobe_bin}"
  SHARED8_SOURCE_REVISION="${source_revision}"
  SHARED8_SOURCE_ARCHIVE_SHA256="${source_archive_sha256}"
  SHARED8_COMMON_INITIALIZED=1

  readonly \
    SHARED8_SCRATCH_ROOT \
    SHARED8_SOURCE_ROOT \
    SHARED8_RUNNER \
    SHARED8_LAUNCHER_LOG_ROOT \
    SHARED8_PYTHON_BIN \
    SHARED8_MANIFEST \
    SHARED8_MANIFEST_SHA256 \
    SHARED8_OUTPUT_ROOT \
    SHARED8_FFPROBE_BIN \
    SHARED8_SOURCE_REVISION \
    SHARED8_SOURCE_ARCHIVE_SHA256 \
    SHARED8_COMMON_INITIALIZED

  export PYTHONDONTWRITEBYTECODE=1
  export ACTION_BASELINE_PYTHON_BIN="${python_bin}"
  export PYTHONNOUSERSITE=1
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export HF_HUB_DISABLE_TELEMETRY=1
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  export OPENBLAS_NUM_THREADS=4
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

  hostname
  date -u +%Y-%m-%dT%H:%M:%SZ
  printf '[shared8-launch] launcher=%s source_revision=%s source_archive_sha256=%s manifest_sha256=%s output_root=%s\n' \
    "${launcher_label}" \
    "${SHARED8_SOURCE_REVISION}" \
    "${SHARED8_SOURCE_ARCHIVE_SHA256}" \
    "${SHARED8_MANIFEST_SHA256}" \
    "${SHARED8_OUTPUT_ROOT}"
  printf '[shared8-launch] video_probe_backend=%s executable=%s\n' \
    "${probe_backend}" "${SHARED8_FFPROBE_BIN}"
  if [[ "${probe_backend}" == "frozen-pyav" ]]; then
    local pyav_runtime
    pyav_runtime="$("${python_bin}" -c 'import av,json; print(json.dumps({"pyav":av.__version__,"libraries":av.library_versions},sort_keys=True))')" || \
      shared8_fail "could not identify the pinned PyAV/FFmpeg runtime"
    printf '[shared8-launch] video_probe_runtime=%s\n' "${pyav_runtime}"
  fi
}
