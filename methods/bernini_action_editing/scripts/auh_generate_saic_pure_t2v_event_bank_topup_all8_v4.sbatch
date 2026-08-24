#!/usr/bin/env bash
#SBATCH --job-name=saic-t2v-topup-r6-v2
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=24:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 077

readonly formal_release_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/releases/saic-t2v-topup-r6-formal-v2-guard1a38-20260812-r1
readonly expected_base_launcher_sha=12c1b2baaecfd479f65f9b5dbf0dbae17cd87196767e93c254fe2cffc895f29d
readonly expected_materializer_sha=5585abb927206d0813caca4cec8dc10b846fe0ace704538bb12e0ad5cffe8b97
readonly expected_effective_launcher_sha=4227b4a00b7b2dea786457baad56b4fdcb4b476929e9619cb533a353b369f9f0
readonly expected_gate_sha=99631ff6537e60d465a571b653fa44b91bfad94d0f3df9454e65c10905358296
readonly expected_guard_sha=1a38b2ac18f46e818b1596db884b883dff8e0612fcd5b0cf1ed78aca377ac965
readonly expected_archive_sha=3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b
readonly expected_source_revision=20c2193954e780e9654347754b1485f3492fbea5
readonly expected_source_manifest_sha=899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9
readonly expected_event_spec_sha=d693d0784530f007888e2825d15db3db808fdf4f1d111b5d080d968c894ff145
readonly expected_checkpoint_manifest_sha=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly expected_python_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly expected_ffmpeg_sha=e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99
readonly expected_ffmpeg_version_sha=389368da4bcd4e22d7bf9134f3a8c24dd36027de7d963015230969a87c9e3339
readonly expected_ffmpeg_first_line='ffmpeg version 7.0.2-static https://johnvansickle.com/ffmpeg/  Copyright (c) 2000-2024 the FFmpeg developers'
readonly expected_python=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly expected_ffmpeg=/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
readonly expected_bernini=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/Bernini-2d2b4591
readonly expected_veomni=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/VeOmni-f90b3dc6
readonly expected_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly expected_output=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/runs/t2v-events-topup-r6-formal-v2-guard1a38-20260812-r1
readonly expected_slurm_logs=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/slurm/saic-t2v-topup-r6-formal-v2-guard1a38-20260812-r1

# This wrapper owns only the operational trust transition.  After it admits
# the immutable v2 canary and its own exactly-once submission receipt, it
# execs the mechanically derived launcher whose scientific archive/runtime,
# specs, checkpoint, topology, and candidate order are unchanged.

base_launcher="${SAIC_T2V_FV2_BASE_LAUNCHER:?}"
base_launcher_sha="${SAIC_T2V_FV2_BASE_LAUNCHER_SHA256:?}"
materializer="${SAIC_T2V_FV2_MATERIALIZER:?}"
materializer_sha="${SAIC_T2V_FV2_MATERIALIZER_SHA256:?}"
effective_launcher="${SAIC_T2V_FV2_EFFECTIVE_LAUNCHER:?}"
effective_launcher_sha="${SAIC_T2V_FV2_EFFECTIVE_LAUNCHER_SHA256:?}"
wrapper="${SAIC_T2V_FV2_WRAPPER:?}"
wrapper_sha="${SAIC_T2V_FV2_WRAPPER_SHA256:?}"
gate="${SAIC_T2V_FV2_GATE:?}"
gate_sha="${SAIC_T2V_FV2_GATE_SHA256:?}"
guard="${SAIC_T2V_V4_EXTERNAL_RENDEZVOUS_GUARD:?}"
guard_sha="${SAIC_T2V_V3_RENDEZVOUS_GUARD_SHA256:?}"
canary_receipt="${SAIC_T2V_FV2_CANARY_RECEIPT:?}"
canary_submission="${SAIC_T2V_FV2_CANARY_SUBMISSION_RECEIPT:?}"
own_submission="${SAIC_T2V_FV2_OWN_SUBMISSION_RECEIPT:?}"
source_archive="${SAIC_T2V_V3_SOURCE_ARCHIVE:?}"
source_archive_sha="${SAIC_T2V_V3_SOURCE_ARCHIVE_SHA256:?}"
source_revision="${SAIC_T2V_V3_SOURCE_REVISION:?}"
source_manifest="${SAIC_T2V_V3_SOURCE_MANIFEST:?}"
source_manifest_sha="${SAIC_T2V_V3_SOURCE_MANIFEST_SHA256:?}"
event_spec="${SAIC_T2V_V3_EVENT_SPEC:?}"
event_spec_sha="${SAIC_T2V_V3_EVENT_SPEC_SHA256:?}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?}"
veomni_root="${BERNINI_VEOMNI_ROOT:?}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?}"
output_root="${SAIC_T2V_V3_OUTPUT_ROOT:?}"
python_bin="${SAIC_T2V_V3_PYTHON_BIN:?}"
python_sha="${SAIC_T2V_FV2_PYTHON_SHA256:?}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?}"
checkpoint_manifest_sha="${SAIC_T2V_FV2_CHECKPOINT_MANIFEST_SHA256:?}"
ffmpeg_bin="${SAIC_T2V_V3_STATIC_FFMPEG:?}"
ffmpeg_sha="${SAIC_T2V_FV2_STATIC_FFMPEG_SHA256:?}"
ffmpeg_version_sha="${SAIC_T2V_FV2_STATIC_FFMPEG_VERSION_STDOUT_SHA256:?}"
ffmpeg_first_line="${SAIC_T2V_FV2_STATIC_FFMPEG_VERSION_FIRST_LINE:?}"
slurm_log_dir="${SAIC_T2V_FV2_SLURM_LOG_DIR:?}"

fail() {
  echo "[saic-t2v-topup-formal-v2] ERROR: $*" >&2
  exit 2
}

[[ "${base_launcher}" == "${formal_release_root}/inputs/auh_generate_saic_pure_t2v_event_bank_topup_all8_v3.base.sbatch" && \
   "${materializer}" == "${formal_release_root}/inputs/materialize_saic_t2v_topup_formal_v2_launcher.py" && \
   "${effective_launcher}" == "${formal_release_root}/inputs/auh_generate_saic_pure_t2v_event_bank_topup_all8_formal_v2_effective.sbatch" && \
   "${wrapper}" == "${formal_release_root}/inputs/auh_generate_saic_pure_t2v_event_bank_topup_all8_v4.sbatch" && \
   "${gate}" == "${formal_release_root}/inputs/saic_t2v_formal_v2_gate.py" && \
   "${guard}" == "${formal_release_root}/inputs/saic_t2v_rendezvous_guard_v2.py" && \
   "${source_archive}" == "${formal_release_root}/inputs/videoedit-saic-20c2193-methods.tar" && \
   "${source_manifest}" == "${formal_release_root}/inputs/saic_reversible_source_set_v1.json" && \
   "${event_spec}" == "${formal_release_root}/inputs/saic_pure_t2v_event_bank_topup_v2.json" && \
   "${checkpoint_manifest}" == "${formal_release_root}/inputs/bernini-r13-ff4c5d4-checkpoint.sha256" && \
   "${python_bin}" == "${expected_python}" && "${ffmpeg_bin}" == "${expected_ffmpeg}" && \
   "${bernini_root}" == "${expected_bernini}" && "${veomni_root}" == "${expected_veomni}" && \
   "${checkpoint}" == "${expected_checkpoint}" && "${output_root}" == "${expected_output}" && \
   "${own_submission}" == "${expected_output}.submission.receipt.json" && \
   "${slurm_log_dir}" == "${expected_slurm_logs}" ]] || fail "formal hard-pinned paths differ"
[[ "${base_launcher_sha}" == "${expected_base_launcher_sha}" && \
   "${materializer_sha}" == "${expected_materializer_sha}" && \
   "${effective_launcher_sha}" == "${expected_effective_launcher_sha}" && \
   "${gate_sha}" == "${expected_gate_sha}" && "${guard_sha}" == "${expected_guard_sha}" && \
   "${source_archive_sha}" == "${expected_archive_sha}" && \
   "${source_revision}" == "${expected_source_revision}" && \
   "${source_manifest_sha}" == "${expected_source_manifest_sha}" && \
   "${event_spec_sha}" == "${expected_event_spec_sha}" && \
   "${checkpoint_manifest_sha}" == "${expected_checkpoint_manifest_sha}" && \
   "${python_sha}" == "${expected_python_sha}" && "${ffmpeg_sha}" == "${expected_ffmpeg_sha}" && \
   "${ffmpeg_version_sha}" == "${expected_ffmpeg_version_sha}" && \
   "${ffmpeg_first_line}" == "${expected_ffmpeg_first_line}" ]] || fail "formal hard-pinned SHA/version values differ"

for name in base_launcher materializer effective_launcher wrapper gate guard canary_receipt \
  canary_submission own_submission output_root python_bin checkpoint_manifest \
  ffmpeg_bin; do
  [[ "${!name}" == /* ]] || fail "${name} must be absolute"
done
for value in "${base_launcher_sha}" "${effective_launcher_sha}" \
  "${materializer_sha}" "${wrapper_sha}" "${gate_sha}" "${guard_sha}" "${python_sha}" \
  "${checkpoint_manifest_sha}" "${ffmpeg_sha}" "${ffmpeg_version_sha}"; do
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || fail "formal input SHA differs"
done
[[ "${SLURM_JOB_ID:?}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"

base_launcher="$(realpath -e -- "${base_launcher}")"
materializer="$(realpath -e -- "${materializer}")"
effective_launcher="$(realpath -e -- "${effective_launcher}")"
wrapper="$(realpath -e -- "${wrapper}")"
gate="$(realpath -e -- "${gate}")"
guard="$(realpath -e -- "${guard}")"
python_bin="$(realpath -e -- "${python_bin}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
ffmpeg_bin="$(realpath -e -- "${ffmpeg_bin}")"
executed_wrapper="$(realpath -e -- "$0")"

check_plain_sha() {
  local path="$1" expected="$2" label="$3"
  [[ -r "${path}" && -f "${path}" && ! -L "${path}" && \
     "$(stat -c '%h' -- "${path}")" == 1 && \
     "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || \
    fail "${label} differs"
}

check_executable_sha() {
  local path="$1" expected="$2" label="$3"
  [[ -x "${path}" && -f "${path}" && ! -L "${path}" && \
     "$(stat -c '%h' -- "${path}")" == 1 && \
     "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || \
    fail "${label} differs"
}

check_ffmpeg_contract() {
  check_executable_sha "${ffmpeg_bin}" "${ffmpeg_sha}" "static ffmpeg"
  local observed_first_line observed_version_sha
  observed_first_line="$("${ffmpeg_bin}" -version | sed -n '1p')"
  observed_version_sha="$("${ffmpeg_bin}" -version | sha256sum | awk '{print $1}')"
  [[ "${observed_first_line}" == "${ffmpeg_first_line}" && \
     "${observed_version_sha}" == "${ffmpeg_version_sha}" && \
     "${observed_first_line}" == *static* ]] || \
    fail "static ffmpeg version contract differs"
}

check_retained_fd() {
  local logical="$1" fd_path="$2" expected="$3" label="$4"
  [[ "${fd_path}" =~ ^/proc/[0-9]+/fd/[0-9]+$ && \
     -r "${logical}" && -f "${logical}" && ! -L "${logical}" && \
     -r "${fd_path}" && -f "${fd_path}" && \
     "$(stat -Lc '%d:%i' -- "${logical}")" == "$(stat -Lc '%d:%i' -- "${fd_path}")" && \
     "$(stat -Lc '%h' -- "${fd_path}")" == 1 && \
     "$(sha256sum "${fd_path}" | awk '{print $1}')" == "${expected}" ]] || \
    fail "${label} retained fd differs"
}

check_plain_sha "${base_launcher}" "${base_launcher_sha}" "base launcher"
check_plain_sha "${materializer}" "${materializer_sha}" "launcher materializer"
check_plain_sha "${effective_launcher}" "${effective_launcher_sha}" "effective launcher"
check_plain_sha "${wrapper}" "${wrapper_sha}" "logical wrapper"
check_plain_sha "${executed_wrapper}" "${wrapper_sha}" "executed Slurm wrapper"
check_plain_sha "${gate}" "${gate_sha}" "formal gate"
check_plain_sha "${guard}" "${guard_sha}" "external guard v2"
check_plain_sha "${checkpoint_manifest}" "${checkpoint_manifest_sha}" \
  "checkpoint manifest"
check_executable_sha "${python_bin}" "${python_sha}" "Python"
check_ffmpeg_contract
[[ "${own_submission}" == "${output_root}.submission.receipt.json" ]] || \
  fail "own submission receipt path differs"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || \
  fail "formal output is not fresh before admission"

# The spooled wrapper is the trust root.  It opens each new admission child
# once and the long-lived launcher PID retains those exact inodes across the
# gate, exec handoff, two WORLD4 subshells, and all 60 guard invocations.
shopt -u varredir_close
exec {gate_fd}<"${gate}"
exec {effective_fd}<"${effective_launcher}"
exec {guard_fd}<"${guard}"
gate_fd_path="/proc/$$/fd/${gate_fd}"
effective_fd_path="/proc/$$/fd/${effective_fd}"
guard_fd_path="/proc/$$/fd/${guard_fd}"
check_retained_fd "${gate}" "${gate_fd_path}" "${gate_sha}" "formal gate"
check_retained_fd "${effective_launcher}" "${effective_fd_path}" \
  "${effective_launcher_sha}" "effective launcher"
check_retained_fd "${guard}" "${guard_fd_path}" "${guard_sha}" "guard v2"
export SAIC_T2V_FV2_GATE_FD_PATH="${gate_fd_path}"
export SAIC_T2V_FV2_RENDEZVOUS_GUARD_FD_PATH="${guard_fd_path}"

"${python_bin}" -B "${gate_fd_path}" \
  --guard "${guard_fd_path}" --guard-sha256 "${guard_sha}" \
  --canary-receipt "${canary_receipt}" \
  --canary-submission-receipt "${canary_submission}" \
  --own-submission-receipt "${own_submission}" \
  --slurm-job-id "${SLURM_JOB_ID}"

# Nothing has been created by this wrapper.  Re-pin the executable handoff
# immediately before exec so a successful submission receipt can never admit
# different effective or guard bytes.
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || \
  fail "formal output appeared after admission"
check_plain_sha "${effective_launcher}" "${effective_launcher_sha}" \
  "effective launcher at handoff"
check_plain_sha "${guard}" "${guard_sha}" "guard v2 at handoff"
check_plain_sha "${checkpoint_manifest}" "${checkpoint_manifest_sha}" \
  "checkpoint manifest at handoff"
check_executable_sha "${python_bin}" "${python_sha}" "Python at handoff"
check_ffmpeg_contract
check_retained_fd "${gate}" "${gate_fd_path}" "${gate_sha}" \
  "formal gate at handoff"
check_retained_fd "${effective_launcher}" "${effective_fd_path}" \
  "${effective_launcher_sha}" "effective launcher at handoff"
check_retained_fd "${guard}" "${guard_fd_path}" "${guard_sha}" \
  "guard v2 at handoff"
[[ -x /usr/bin/bash && -f /usr/bin/bash && ! -L /usr/bin/bash ]] || \
  fail "absolute bash differs"
exec /usr/bin/bash "${effective_fd_path}"
