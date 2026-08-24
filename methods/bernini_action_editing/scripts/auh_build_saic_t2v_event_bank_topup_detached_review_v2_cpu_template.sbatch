#!/usr/bin/bash
#SBATCH --job-name=saic-t2v-topup-review-v2
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=08:00:00

# Fresh versioned template.  It is intentionally unlaunchable until all
# __REVIEW_*__ values are replaced from one successful formal full60 terminal
# result and the admitted AUH compute-Bash identity.  No hold or dependency is
# used: the submitter queues this CPU job only after independently observing the
# formal job as COMPLETED/0:0 and sealing the formal master-receipt pins.

set -Eeuo pipefail

readonly expected_source_archive_sha=3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b
readonly expected_source_revision=20c2193954e780e9654347754b1485f3492fbea5
readonly expected_adapter_sha=__REVIEW_ADAPTER_SHA256__
readonly expected_v1_builder_sha=eac827c2122d8eeb176f4dc2fcf9b95e1fe1c9cbe709c2542a4103424ff8d02b
readonly expected_ffprobe_wrapper_sha=06a5698bdaa8069e541e804e2f0e945665b2d16409633aa92d37f76ed7506710
readonly expected_diagnostics_sha=3658056640b0adc3411c04c029ce99efd5a4d9388be638f659bd8eb472399e0a
readonly expected_generation_runtime_sha=3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36
readonly expected_topup_contract_sha=508dde8d995dcc8deeccb47b35be71b9915a86964626383660d8eed952ef5278
readonly expected_python_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly expected_ffmpeg_sha=e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99
readonly expected_ffmpeg_version_sha=389368da4bcd4e22d7bf9134f3a8c24dd36027de7d963015230969a87c9e3339
readonly expected_ffmpeg_first_line='ffmpeg version 7.0.2-static https://johnvansickle.com/ffmpeg/  Copyright (c) 2000-2024 the FFmpeg developers'
readonly expected_sacct_sha=fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e
readonly expected_formal_job_id=__REVIEW_FORMAL_JOB_ID__
readonly expected_formal_master_sha=__REVIEW_FORMAL_MASTER_SHA256__
readonly expected_formal_master_digest=__REVIEW_FORMAL_MASTER_DIGEST__
readonly expected_formal_submission_sha=__REVIEW_FORMAL_SUBMISSION_SHA256__
readonly expected_formal_submission_digest=__REVIEW_FORMAL_SUBMISSION_DIGEST__
readonly expected_compute_bash_sha=__REVIEW_COMPUTE_BASH_SHA256__
readonly expected_compute_bash_version_sha=__REVIEW_COMPUTE_BASH_VERSION_STDOUT_SHA256__
readonly expected_compute_bash_first_line='__REVIEW_COMPUTE_BASH_VERSION_FIRST_LINE__'

readonly formal_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809
readonly expected_release_root="${formal_root}/releases/saic-t2v-topup-r6-formal-v2-r2-retfd-20260812-r1-review-v2-r1"
readonly expected_formal_root="${formal_root}/runs/t2v-events-topup-r6-formal-v2-r2-retfd-20260812-r1"
readonly expected_output_root="${formal_root}/reviews/t2v-events-topup-r6-formal-v2-r2-retfd-20260812-r1-detached-review-v2-r1"
readonly expected_automation_receipt="${expected_output_root}.automation.receipt.json"
readonly expected_terminal_admission="${expected_output_root}.terminal.admission.receipt.json"
readonly expected_source_archive="${expected_release_root}/inputs/videoedit-saic-20c2193-methods.tar"
readonly expected_adapter="${expected_release_root}/inputs/build_saic_t2v_event_bank_topup_detached_review_v2.py"
readonly expected_formal_master="${expected_formal_root}/saic-pure-t2v-event-bank-topup-receipt.json"
readonly expected_formal_submission="${expected_formal_root}.submission.receipt.json"
readonly expected_python=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly expected_ffmpeg=/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
readonly expected_compute_bash=/usr/bin/bash
readonly packet_id=t2v-events-topup-r6-formal-v2-r2-retfd-20260812-r1-detached-review-v2-r1

fail() {
  echo "[saic-t2v-topup-review-v2] ERROR: $*" >&2
  exit 2
}

reject_placeholders() {
  local value
  for value in \
    "${expected_adapter_sha}" "${expected_formal_job_id}" \
    "${expected_formal_master_sha}" "${expected_formal_master_digest}" \
    "${expected_formal_submission_sha}" "${expected_formal_submission_digest}" \
    "${expected_compute_bash_sha}" "${expected_compute_bash_version_sha}" \
    "${expected_compute_bash_first_line}"; do
    [[ "${value}" != *'__REVIEW_'* ]] || fail "unresolved review release placeholder"
  done
}

is_sha() { [[ "$1" =~ ^[0-9a-f]{64}$ ]]; }

plain_file_sha() {
  local path=$1 expected=$2 label=$3
  [[ -f "${path}" && ! -L "${path}" ]] || fail "${label} is not a plain file"
  [[ "$(stat -c '%h %a' -- "${path}")" == '1 444' ]] || fail "${label} mode/link differs"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || fail "${label} SHA-256 differs"
}

reject_placeholders
for value in \
  "${expected_adapter_sha}" "${expected_formal_master_sha}" \
  "${expected_formal_master_digest}" "${expected_formal_submission_sha}" \
  "${expected_formal_submission_digest}" "${expected_compute_bash_sha}" \
  "${expected_compute_bash_version_sha}"; do
  is_sha "${value}" || fail "release SHA-256 pin differs"
done
[[ "${expected_formal_job_id}" =~ ^[0-9]+$ ]] || fail "formal JobID pin differs"

release_root="${SAIC_T2V_TOPUP_REVIEW_RELEASE_ROOT:?}"
source_archive="${SAIC_T2V_TOPUP_REVIEW_SOURCE_ARCHIVE:?}"
source_archive_sha="${SAIC_T2V_TOPUP_REVIEW_SOURCE_ARCHIVE_SHA256:?}"
source_revision="${SAIC_T2V_TOPUP_REVIEW_SOURCE_REVISION:?}"
adapter="${SAIC_T2V_TOPUP_REVIEW_ADAPTER:?}"
adapter_sha="${SAIC_T2V_TOPUP_REVIEW_ADAPTER_SHA256:?}"
input_root="${SAIC_T2V_TOPUP_REVIEW_INPUT_ROOT:?}"
formal_master="${SAIC_T2V_TOPUP_REVIEW_FORMAL_MASTER:?}"
formal_master_sha="${SAIC_T2V_TOPUP_REVIEW_FORMAL_MASTER_SHA256:?}"
formal_master_digest="${SAIC_T2V_TOPUP_REVIEW_FORMAL_MASTER_DIGEST:?}"
formal_submission="${SAIC_T2V_TOPUP_REVIEW_FORMAL_SUBMISSION:?}"
formal_submission_sha="${SAIC_T2V_TOPUP_REVIEW_FORMAL_SUBMISSION_SHA256:?}"
formal_submission_digest="${SAIC_T2V_TOPUP_REVIEW_FORMAL_SUBMISSION_DIGEST:?}"
formal_job_id="${SAIC_T2V_TOPUP_REVIEW_FORMAL_JOB_ID:?}"
output_root="${SAIC_T2V_TOPUP_REVIEW_OUTPUT_ROOT:?}"
automation_receipt="${SAIC_T2V_TOPUP_REVIEW_AUTOMATION_RECEIPT:?}"
python_bin="${SAIC_T2V_TOPUP_REVIEW_PYTHON_BIN:?}"
python_sha="${SAIC_T2V_TOPUP_REVIEW_PYTHON_SHA256:?}"
ffmpeg_bin="${SAIC_T2V_TOPUP_REVIEW_FFMPEG_BIN:?}"
ffmpeg_sha="${SAIC_T2V_TOPUP_REVIEW_FFMPEG_SHA256:?}"
ffprobe_wrapper_sha="${SAIC_T2V_TOPUP_REVIEW_FFPROBE_WRAPPER_SHA256:?}"
compute_bash="${SAIC_T2V_TOPUP_REVIEW_COMPUTE_BASH:?}"
compute_bash_sha="${SAIC_T2V_TOPUP_REVIEW_COMPUTE_BASH_SHA256:?}"
compute_bash_version_sha="${SAIC_T2V_TOPUP_REVIEW_COMPUTE_BASH_VERSION_STDOUT_SHA256:?}"
# The comma-bearing first line is an embedded launcher pin, not a Slurm export.
compute_bash_first_line="${expected_compute_bash_first_line}"
workers="${SAIC_T2V_TOPUP_REVIEW_WORKERS:?}"

[[ "${release_root}" == "${expected_release_root}" && \
   "${source_archive}" == "${expected_source_archive}" && \
   "${source_archive_sha}" == "${expected_source_archive_sha}" && \
   "${source_revision}" == "${expected_source_revision}" && \
   "${adapter}" == "${expected_adapter}" && "${adapter_sha}" == "${expected_adapter_sha}" && \
   "${input_root}" == "${expected_formal_root}" && \
   "${formal_master}" == "${expected_formal_master}" && \
   "${formal_master_sha}" == "${expected_formal_master_sha}" && \
   "${formal_master_digest}" == "${expected_formal_master_digest}" && \
   "${formal_submission}" == "${expected_formal_submission}" && \
   "${formal_submission_sha}" == "${expected_formal_submission_sha}" && \
   "${formal_submission_digest}" == "${expected_formal_submission_digest}" && \
   "${formal_job_id}" == "${expected_formal_job_id}" && \
   "${output_root}" == "${expected_output_root}" && \
   "${automation_receipt}" == "${expected_automation_receipt}" && \
   "${python_bin}" == "${expected_python}" && "${python_sha}" == "${expected_python_sha}" && \
   "${ffmpeg_bin}" == "${expected_ffmpeg}" && "${ffmpeg_sha}" == "${expected_ffmpeg_sha}" && \
   "${ffprobe_wrapper_sha}" == "${expected_ffprobe_wrapper_sha}" && \
   "${compute_bash}" == "${expected_compute_bash}" && \
   "${compute_bash_sha}" == "${expected_compute_bash_sha}" && \
   "${compute_bash_version_sha}" == "${expected_compute_bash_version_sha}" && \
   "${compute_bash_first_line}" == "${expected_compute_bash_first_line}" && \
   "${workers}" == 16 ]] || fail "runtime exports differ from release pins"

[[ "${BASH}" == "${expected_compute_bash}" ]] || fail "launcher interpreter path differs"
[[ "$(realpath -e -- "/proc/$$/exe")" == "${expected_compute_bash}" ]] || fail "launcher interpreter identity differs"
[[ -x "${compute_bash}" && -f "${compute_bash}" && ! -L "${compute_bash}" ]] || fail "compute Bash differs"
[[ "$(sha256sum "${compute_bash}" | awk '{print $1}')" == "${compute_bash_sha}" ]] || fail "compute Bash SHA-256 differs"
[[ "$("${compute_bash}" --version | sha256sum | awk '{print $1}')" == "${compute_bash_version_sha}" ]] || fail "compute Bash version stdout differs"
IFS= read -r observed_bash_first_line < <("${compute_bash}" --version)
[[ "${observed_bash_first_line}" == "${compute_bash_first_line}" ]] || fail "compute Bash version first line differs"

[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "review JobID differs"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "review packet root must be fresh"
[[ ! -e "${automation_receipt}" && ! -L "${automation_receipt}" ]] || fail "automation receipt must be fresh"
[[ ! -e "${expected_terminal_admission}" && ! -L "${expected_terminal_admission}" ]] || fail "terminal admission must be fresh"
[[ "${output_root##*/}" == "${packet_id}" ]] || fail "review packet basename differs"
for path in "${release_root}" "${input_root}"; do
  [[ -d "${path}" && ! -L "${path}" ]] || fail "required directory differs"
done
plain_file_sha "${source_archive}" "${source_archive_sha}" "source archive"
exec {source_archive_fd}<"${source_archive}"
source_archive_fd_path="/proc/$$/fd/${source_archive_fd}"
[[ "${source_archive_fd_path}" =~ ^/proc/[0-9]+/fd/[0-9]+$ && \
   -r "${source_archive_fd_path}" && -f "${source_archive_fd_path}" && \
   "$(stat -Lc '%d:%i:%h' -- "${source_archive_fd_path}")" == \
     "$(stat -Lc '%d:%i:%h' -- "${source_archive}")" && \
   "$(sha256sum "${source_archive_fd_path}" | awk '{print $1}')" == \
     "${source_archive_sha}" ]] || fail "source archive retained FD differs"
plain_file_sha "${adapter}" "${adapter_sha}" "v2 review adapter"
plain_file_sha "${formal_master}" "${formal_master_sha}" "formal master receipt"
plain_file_sha "${formal_submission}" "${formal_submission_sha}" "formal submission receipt"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "Python executable differs"
[[ "$(sha256sum "${python_bin}" | awk '{print $1}')" == "${python_sha}" ]] || fail "Python SHA-256 differs"
[[ -x "${ffmpeg_bin}" && -f "${ffmpeg_bin}" && ! -L "${ffmpeg_bin}" ]] || fail "ffmpeg executable differs"
[[ "$(sha256sum "${ffmpeg_bin}" | awk '{print $1}')" == "${ffmpeg_sha}" ]] || fail "ffmpeg SHA-256 differs"
[[ "$("${ffmpeg_bin}" -version | sha256sum | awk '{print $1}')" == "${expected_ffmpeg_version_sha}" ]] || fail "ffmpeg version stdout differs"
IFS= read -r observed_ffmpeg_first_line < <("${ffmpeg_bin}" -version)
[[ "${observed_ffmpeg_first_line}" == "${expected_ffmpeg_first_line}" ]] || fail "ffmpeg first line differs"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/saic-t2v-topup-review-v2-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/saic-t2v-topup-review-v2-${SLURM_JOB_ID}."*) ;;
    *) cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[saic-t2v-topup-review-v2] PASS candidates=60 diagnostics=60 html=true blind_html=true human_labels=blank semantic=UNASSESSED machine_authority=zero packet=${output_root} automation=${automation_receipt}"
  else
    echo "[saic-t2v-topup-review-v2] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false) packet=${output_root}" >&2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

sacct=/usr/bin/sacct
[[ -x "${sacct}" && -f "${sacct}" && ! -L "${sacct}" ]] || fail "sacct executable differs"
[[ "$(sha256sum "${sacct}" | awk '{print $1}')" == "${expected_sacct_sha}" ]] || fail "sacct SHA-256 differs"
sacct_stdout="${task_scratch}/formal.sacct.stdout"
sacct_stderr="${task_scratch}/formal.sacct.stderr"
set +e
"${sacct}" -j "${formal_job_id}" -X --noheader -n -P -o 'JobIDRaw,State,ExitCode,AllocTRES%512,NodeList,ElapsedRaw,Start,End' >"${sacct_stdout}" 2>"${sacct_stderr}"
sacct_status=$?
set -e
[[ "${sacct_status}" == 0 && ! -s "${sacct_stderr}" ]] || fail "formal terminal sacct query failed"
[[ "$(wc -l <"${sacct_stdout}" | tr -d ' ')" == 1 ]] || fail "formal sacct row count differs"
IFS='|' read -r sacct_job_id sacct_state sacct_exit sacct_alloc sacct_node sacct_elapsed sacct_start sacct_end <"${sacct_stdout}"
[[ "${sacct_job_id}" == "${formal_job_id}" && "${sacct_state}" == COMPLETED && "${sacct_exit}" == 0:0 ]] || fail "formal job is not terminal success"
declare -A formal_alloc_seen=()
IFS=',' read -r -a formal_alloc_tokens <<<"${sacct_alloc}"
for token in "${formal_alloc_tokens[@]}"; do
  [[ "${token}" == *=* ]] || fail "formal allocation token differs"
  alloc_key="${token%%=*}"
  alloc_value="${token#*=}"
  [[ -n "${alloc_key}" && -n "${alloc_value}" && -z "${formal_alloc_seen[${alloc_key}]+present}" ]] || \
    fail "formal allocation closure differs"
  formal_alloc_seen["${alloc_key}"]="${alloc_value}"
done
[[ "${#formal_alloc_seen[@]}" == 6 && \
   "${formal_alloc_seen[billing]-}" == 32 && \
   "${formal_alloc_seen[cpu]-}" == 32 && \
   "${formal_alloc_seen[gres/gpu:mi210]-}" == 8 && \
   "${formal_alloc_seen[gres/gpu]-}" == 8 && \
   "${formal_alloc_seen[mem]-}" == 256G && \
   "${formal_alloc_seen[node]-}" == 1 ]] || fail "formal allocation differs"
[[ -n "${sacct_node}" && "${sacct_node}" != 'None assigned' && "${sacct_elapsed}" =~ ^[0-9]+$ && -n "${sacct_start}" && -n "${sacct_end}" ]] || fail "formal terminal accounting is incomplete"

"${python_bin}" -B - "${formal_master}" "${formal_master_sha}" "${formal_master_digest}" "${formal_submission}" "${formal_submission_sha}" "${formal_submission_digest}" "${formal_job_id}" "${input_root}" <<'PY'
import hashlib, json, pathlib, re, sys

master_path, master_sha, master_digest, submission_path, submission_sha, submission_digest, job_id, input_root = sys.argv[1:]
sha_re = re.compile(r"[0-9a-f]{64}\Z")
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
def load(path, expected_sha, expected_digest, label):
    raw = pathlib.Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise SystemExit(f"{label} file SHA differs")
    value = json.loads(raw.decode("ascii"), object_pairs_hook=no_dupes)
    if raw != canonical(value) + b"\n":
        raise SystemExit(f"{label} is not canonical JSON")
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    if declared != expected_digest or not isinstance(declared, str) or not sha_re.fullmatch(declared) or hashlib.sha256(canonical(unsigned)).hexdigest() != declared:
        raise SystemExit(f"{label} object seal differs")
    return value
def no_dupes(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"duplicate JSON key: {key}")
        value[key] = item
    return value

master = load(master_path, master_sha, master_digest, "formal master")
submission = load(submission_path, submission_sha, submission_digest, "formal submission")
if (master.get("schema_version") != "bernini-saic-pure-t2v-event-bank-topup-receipt-v2" or master.get("top_up_only") is not True or master.get("attempt_count") != 60 or master.get("row_count") != 8 or master.get("seed_cell_count") != 20 or master.get("branch_order") != ["incomplete", "camera_only", "appearance_only"] or len(master.get("attempts", [])) != 60):
    raise SystemExit("formal master cardinality/contract differs")
for field in ("detached_full81_event_review_complete", "event_verified", "identity_preservation_verified", "seed_selection_authorized", "training_target_authorized", "optimizer_or_parameter_update_authorized"):
    if master.get(field) is not False:
        raise SystemExit(f"formal master unexpectedly authorizes {field}")
submitted = submission.get("submitted_job", {})
request = submission.get("request", {})
outputs = submission.get("outputs", {})
if (submission.get("submission_success") is not True or submitted.get("job_id") != job_id or request.get("hold") is not False or request.get("dependency") is not None or request.get("candidate_count") != 60 or outputs.get("output_root") != input_root):
    raise SystemExit("formal submission binding differs")
PY

mkdir -- "${task_scratch}/source-tree"
cp -- "${source_archive_fd_path}" "${task_scratch}/source.tar"
chmod 0444 -- "${task_scratch}/source.tar"
[[ "$(sha256sum "${task_scratch}/source.tar" | awk '{print $1}')" == \
   "${source_archive_sha}" ]] || fail "scratch source archive SHA differs"
[[ "$(git get-tar-commit-id <"${task_scratch}/source.tar")" == "${source_revision}" ]] || fail "source archive revision differs"
tar --no-same-owner --no-same-permissions -xf "${task_scratch}/source.tar" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
motive_root="${task_scratch}/source-tree/methods/motive"
runtime_adapter="${method_root}/tools/build_saic_t2v_event_bank_topup_detached_review_v2.py"
[[ ! -e "${runtime_adapter}" && ! -L "${runtime_adapter}" ]] || fail "adapter destination is not fresh"
cp -- "${adapter}" "${runtime_adapter}"
chmod 0444 -- "${runtime_adapter}"

required=(
  "${runtime_adapter}"
  "${method_root}/tools/build_saic_t2v_event_bank_detached_review_v1.py"
  "${method_root}/tools/ffprobe_pyav_exact81_diagnostic_v1.py"
  "${method_root}/saic_exact81_media_diagnostics_v1.py"
  "${method_root}/decoded_temporal_event_evaluator_v1.py"
  "${method_root}/generate_saic_pure_t2v_event_bank_topup_v2.py"
  "${method_root}/generate_saic_pure_t2v_event_bank_v1.py"
  "${method_root}/saic_pure_t2v_event_bank_topup_v2.py"
  "${method_root}/saic_pure_t2v_event_bank_v1.py"
  "${method_root}/build_saic_reversible_source_set_v1.py"
  "${method_root}/assets/saic_reversible_source_set_v1.json"
  "${method_root}/assets/saic_pure_t2v_event_bank_v1.json"
  "${method_root}/assets/saic_pure_t2v_event_bank_topup_v2.json"
  "${motive_root}/motive/geometry.py"
  "${motive_root}/motive/descriptor.py"
  "${motive_root}/motive/selection.py"
  "${motive_root}/motive/semantics.py"
)
for path in "${required[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "runtime source closure absent: ${path}"
done
[[ "$(sha256sum "${runtime_adapter}" | awk '{print $1}')" == "${adapter_sha}" ]] || fail "runtime adapter differs"
[[ "$(sha256sum "${method_root}/tools/build_saic_t2v_event_bank_detached_review_v1.py" | awk '{print $1}')" == "${expected_v1_builder_sha}" ]] || fail "v1 review builder differs"
[[ "$(sha256sum "${method_root}/saic_exact81_media_diagnostics_v1.py" | awk '{print $1}')" == "${expected_diagnostics_sha}" ]] || fail "exact81 diagnostics differ"
[[ "$(sha256sum "${method_root}/generate_saic_pure_t2v_event_bank_topup_v2.py" | awk '{print $1}')" == "${expected_generation_runtime_sha}" ]] || fail "top-up generation runtime differs"
[[ "$(sha256sum "${method_root}/saic_pure_t2v_event_bank_topup_v2.py" | awk '{print $1}')" == "${expected_topup_contract_sha}" ]] || fail "top-up contract runtime differs"
find "${task_scratch}/source-tree" -type f -exec chmod a-w -- {} +

tool_bin="${task_scratch}/tool-bin"
mkdir -- "${tool_bin}"
ln -s -- "${ffmpeg_bin}" "${tool_bin}/ffmpeg"
ffprobe_source="${method_root}/tools/ffprobe_pyav_exact81_diagnostic_v1.py"
[[ "$(sha256sum "${ffprobe_source}" | awk '{print $1}')" == "${ffprobe_wrapper_sha}" ]] || fail "ffprobe wrapper source differs"
cp -- "${ffprobe_source}" "${tool_bin}/ffprobe"
chmod 0555 -- "${tool_bin}/ffprobe"
ffprobe_bin="$(realpath -e -- "${tool_bin}/ffprobe")"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${method_root}:${motive_root}"
export SAIC_T2V_REVIEW_PYTHON_BIN="${python_bin}"
export PATH="${tool_bin}:/usr/bin:/bin"
export SAIC_T2V_REVIEW_EXPECTED_FFMPEG="${ffmpeg_bin}"
export SAIC_T2V_REVIEW_EXPECTED_FFPROBE="${ffprobe_bin}"
export SAIC_T2V_REVIEW_FFMPEG_SHA256="${ffmpeg_sha}"
export SAIC_T2V_REVIEW_FFPROBE_SHA256="${ffprobe_wrapper_sha}"
export SAIC_T2V_REVIEW_CLEAN_PATH="${PATH}"
[[ "$(realpath -e -- "$(command -v ffmpeg)")" == "${ffmpeg_bin}" ]] || fail "clean PATH ffmpeg differs"
[[ "$(command -v ffprobe)" == "${ffprobe_bin}" ]] || fail "clean PATH ffprobe differs"
ffmpeg -version >/dev/null
ffprobe -version >/dev/null

output_parent="$(dirname -- "${output_root}")"
mkdir -p -- "${output_parent}"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "review output parent differs"
output_parent="$(realpath -e -- "${output_parent}")"
[[ "${output_root}" == "${output_parent}/${output_root##*/}" ]] || fail "review output root is not canonical"

"${python_bin}" -B "${runtime_adapter}" build \
  --input-root "${input_root}" \
  --output-root "${output_root}" \
  --job-id "${SLURM_JOB_ID}" \
  --workers "${workers}"
"${python_bin}" -B "${runtime_adapter}" validate --output-root "${output_root}"

[[ -f "${output_root}/detached-review-receipt.json" && ! -L "${output_root}/detached-review-receipt.json" ]] || fail "detached review receipt absent"
[[ -f "${output_root}/review-manifest.json" && ! -L "${output_root}/review-manifest.json" ]] || fail "review manifest absent"
[[ -f "${output_root}/index.html" && ! -L "${output_root}/index.html" ]] || fail "technical HTML absent"
[[ -f "${output_root}/blind-review.html" && ! -L "${output_root}/blind-review.html" ]] || fail "blind HTML absent"
[[ "$(find "${output_root}/diagnostics" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' ')" == 60 ]] || fail "technical diagnostic count differs"
[[ "$(find "${output_root}/observer-templates" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' ')" == 2 ]] || fail "blank observer template count differs"

"${python_bin}" -B - "${automation_receipt}" "${output_root}" "${formal_job_id}" "${formal_master}" "${formal_master_sha}" "${formal_master_digest}" "${formal_submission}" "${formal_submission_sha}" "${formal_submission_digest}" "${SLURM_JOB_ID}" "${sacct}" "${expected_sacct_sha}" "${sacct_stdout}" "${sacct_stderr}" <<'PY'
import hashlib, json, os, pathlib, stat, sys

(receipt_name, output_name, formal_job_id, master_name, master_sha, master_digest,
 submission_name, submission_sha, submission_digest, review_job_id, sacct_name,
 sacct_sha, sacct_stdout_name, sacct_stderr_name) = sys.argv[1:]
receipt_path = pathlib.Path(receipt_name)
output = pathlib.Path(output_name)
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def load(path):
    raw = pathlib.Path(path).read_bytes()
    value = json.loads(raw.decode("ascii"))
    if raw != canonical(value) + b"\n":
        raise SystemExit(f"noncanonical JSON: {path}")
    return value

review_receipt_path = output / "detached-review-receipt.json"
manifest_path = output / "review-manifest.json"
review_receipt = load(review_receipt_path)
manifest = load(manifest_path)
false_authority = {
    "machine_diagnostics_have_semantic_authority": False,
    "event_verified": False,
    "identity_preservation_verified": False,
    "candidate_selection_allowed": False,
    "seed_selection_allowed": False,
    "training_target_allowed": False,
    "training_allowed": False,
    "optimizer_step_allowed": False,
    "parameter_update_allowed": False,
    "absolute_action_editing_success_claimed": False,
}
template_fields = {
    "schema_version", "packet_id", "observer_slot", "template_only",
    "semantic_status", "observer_id", "observer_kind",
    "observer_authority_artifact", "observer_protocol_artifact",
    "completed_at", "independent_observer_required",
    "same_person_must_not_fill_both_slots",
    "copy_outside_sealed_packet_before_completion",
    "blindness_or_independence_established_by_template",
    "review_item_set_digest", "responses", "authority", "template_digest",
}
response_fields = {
    "review_item_id", "registered_q0_state_correct_frames_0_15",
    "registered_q0_state_held_full81",
    "registered_target_terminal_absent_full81",
    "incomplete_directional_progress_visible",
    "incomplete_partial_state_held_frames_73_80",
    "registered_smooth_camera_motion_visible",
    "camera_cut_or_discontinuity_absent",
    "registered_appearance_only_change_visible", "camera_locked_full81",
    "nonregistered_subject_appearance_or_geometry_change_absent",
    "event_branch_pass", "identity_preserved_full81",
    "technical_quality_acceptable_full81", "observer_notes",
}
expected_review_ids = [f"review-{index:04d}" for index in range(1, 61)]
protocol_binding = manifest.get("observer_protocol")
templates = []
for slot in (1, 2):
    path = output / f"observer-templates/observer-{slot}-blank.json"
    value = load(path)
    responses = value.get("responses")
    if (
        type(value) is not dict
        or set(value) != template_fields
        or value.get("schema_version") != "bernini-saic-t2v-event-bank-topup-independent-observer-blank-template-v2"
        or value.get("packet_id") != manifest.get("packet_id")
        or value.get("observer_slot") != slot
        or value.get("template_only") is not True
        or value.get("semantic_status") != "UNASSESSED"
        or value.get("observer_id") is not None
        or value.get("observer_kind") is not None
        or value.get("observer_authority_artifact") is not None
        or value.get("observer_protocol_artifact") != protocol_binding
        or value.get("completed_at") is not None
        or value.get("independent_observer_required") is not True
        or value.get("same_person_must_not_fill_both_slots") is not True
        or value.get("copy_outside_sealed_packet_before_completion") is not True
        or value.get("blindness_or_independence_established_by_template") is not False
        or value.get("review_item_set_digest") != protocol_binding.get("review_item_set_digest")
        or value.get("authority") != false_authority
        or type(responses) is not list
        or len(responses) != 60
        or any(
            type(row) is not dict
            or set(row) != response_fields
            or row.get("review_item_id") != expected_id
            or any(row[field] is not None for field in response_fields - {"review_item_id"})
            for expected_id, row in zip(expected_review_ids, responses, strict=True)
        )
    ):
        raise SystemExit("observer template is not blank")
    templates.append({"observer_slot": slot, "path": str(path), "sha256": sha(path), "template_digest": value["template_digest"], "template_only": True})
if (review_receipt.get("semantic_status") != "UNASSESSED" or review_receipt.get("machine_diagnostics_zero_authority") is not True or review_receipt.get("observer_labels_present") is not False or manifest.get("candidate_count") != 60 or manifest.get("machine_diagnostic_count") != 60):
    raise SystemExit("review packet authority/cardinality differs")
sacct_stdout = pathlib.Path(sacct_stdout_name).read_bytes()
sacct_stderr = pathlib.Path(sacct_stderr_name).read_bytes()
body = {
    "schema_version": "saic-t2v-topup-detached-review-v2-automation-receipt-v1",
    "status": "technical_packet_materialized_runtime_postflight_pending",
    "formal_input": {
        "job_id": formal_job_id,
        "terminal_state": "COMPLETED",
        "exit_code": "0:0",
        "master_receipt": master_name,
        "master_receipt_sha256": master_sha,
        "master_receipt_digest": master_digest,
        "submission_receipt": submission_name,
        "submission_receipt_sha256": submission_sha,
        "submission_receipt_digest": submission_digest,
        "sacct_executable": sacct_name,
        "sacct_executable_sha256": sacct_sha,
        "sacct_stdout_sha256": hashlib.sha256(sacct_stdout).hexdigest(),
        "sacct_stderr_sha256": hashlib.sha256(sacct_stderr).hexdigest(),
    },
    "review_job_id": review_job_id,
    "packet": {
        "output_root": str(output),
        "packet_id": manifest["packet_id"],
        "review_receipt": str(review_receipt_path),
        "review_receipt_sha256": sha(review_receipt_path),
        "review_receipt_digest": review_receipt["receipt_digest"],
        "review_manifest": str(manifest_path),
        "review_manifest_sha256": sha(manifest_path),
        "review_manifest_digest": manifest["manifest_digest"],
        "candidate_count": 60,
        "technical_diagnostic_count": 60,
    },
    "automatic_backfill_interface": {
        "technical_diagnostics_glob": str(output / "diagnostics/*.json"),
        "technical_html": str(output / "index.html"),
        "technical_html_sha256": sha(output / "index.html"),
        "blind_human_review_html": str(output / "blind-review.html"),
        "blind_human_review_html_sha256": sha(output / "blind-review.html"),
        "observer_protocol": str(output / "observer-protocol.json"),
        "observer_protocol_sha256": sha(output / "observer-protocol.json"),
        "observer_templates": templates,
        "machine_artifacts_materialized": True,
        "blind_stage_public_surface_only": True,
        "technical_html_may_publish_before_two_external_human_seals": False,
        "assessor_private_mapping_may_be_copied_to_stage1": False,
        "machine_ingest_ready": False,
        "terminal_machine_ingest_scope": "assessor_private_only",
        "human_visible_machine_backfill_ready": False,
        "external_terminal_postflight_required": True,
        "human_label_ingest_ready": False,
        "human_template_copyout_required": True,
        "blind_review_is_only_prelabel_ui": True,
        "technical_html_visibility_requires_two_external_human_seals": True,
    },
    "authority": {
        "machine_diagnostics_have_semantic_authority": False,
        "machine_backfill_may_precede_human_label_seal": False,
        "assessor_private_machine_ingest_may_precede_human_seals": True,
        "review_job_terminal_success_claimed": False,
        "human_labels_present": False,
        "event_verified": False,
        "identity_preservation_verified": False,
        "seed_selection_allowed": False,
        "training_target_allowed": False,
        "training_allowed": False,
        "optimizer_step_allowed": False,
        "parameter_update_allowed": False,
    },
}
value = {**body, "receipt_digest": hashlib.sha256(canonical(body)).hexdigest()}
payload = canonical(value) + b"\n"
receipt_parent = receipt_path.parent
parent_info = receipt_parent.lstat()
if (
    not receipt_path.is_absolute()
    or receipt_parent.resolve(strict=True) != receipt_parent
    or not stat.S_ISDIR(parent_info.st_mode)
    or stat.S_ISLNK(parent_info.st_mode)
    or parent_info.st_uid != os.getuid()
    or stat.S_IMODE(parent_info.st_mode) & 0o022
):
    raise SystemExit("automation receipt parent differs")
parent_descriptor = os.open(
    receipt_parent,
    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    | getattr(os, "O_CLOEXEC", 0),
)
try:
    opened_parent = os.fstat(parent_descriptor)
    if (
        not stat.S_ISDIR(opened_parent.st_mode)
        or (opened_parent.st_dev, opened_parent.st_ino)
        != (parent_info.st_dev, parent_info.st_ino)
    ):
        raise SystemExit("opened automation receipt parent differs")
    try:
        os.stat(receipt_path.name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise SystemExit("automation receipt target appeared before create")
    flags = (
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(
        receipt_path.name, flags, 0o600, dir_fd=parent_descriptor
    )
    try:
        reserved = os.fstat(descriptor)
        public = os.stat(
            receipt_path.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        identity = (reserved.st_dev, reserved.st_ino)
        if (
            not stat.S_ISREG(reserved.st_mode)
            or reserved.st_nlink != 1
            or reserved.st_uid != os.getuid()
            or stat.S_IMODE(reserved.st_mode) != 0o600
            or reserved.st_size != 0
            or (public.st_dev, public.st_ino) != identity
        ):
            raise SystemExit("automation receipt reservation differs")
        offset = 0
        while offset < len(payload):
            wrote = os.write(descriptor, payload[offset:])
            if wrote <= 0:
                raise SystemExit("automation receipt write stalled")
            offset += wrote
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.read(descriptor, len(payload) + 1) != payload:
            raise SystemExit("automation receipt same-FD reread differs")
        written = os.fstat(descriptor)
        public = os.stat(
            receipt_path.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(written.st_mode)
            or written.st_nlink != 1
            or written.st_uid != os.getuid()
            or stat.S_IMODE(written.st_mode) != 0o600
            or written.st_size != len(payload)
            or (written.st_dev, written.st_ino) != identity
            or (public.st_dev, public.st_ino) != identity
            or stat.S_IMODE(public.st_mode) != 0o600
            or public.st_size != len(payload)
        ):
            raise SystemExit("automation receipt written identity differs")
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.read(descriptor, len(payload) + 1) != payload:
            raise SystemExit("sealed automation receipt same-FD reread differs")
        sealed = os.fstat(descriptor)
        public = os.stat(
            receipt_path.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        public_path = receipt_path.lstat()
        if (
            not stat.S_ISREG(sealed.st_mode)
            or sealed.st_nlink != 1
            or sealed.st_uid != os.getuid()
            or stat.S_IMODE(sealed.st_mode) != 0o444
            or sealed.st_size != len(payload)
            or (sealed.st_dev, sealed.st_ino) != identity
            or (public.st_dev, public.st_ino) != identity
            or (public_path.st_dev, public_path.st_ino) != identity
            or stat.S_IMODE(public.st_mode) != 0o444
            or stat.S_IMODE(public_path.st_mode) != 0o444
            or public.st_size != len(payload)
            or public_path.st_size != len(payload)
        ):
            raise SystemExit("sealed automation receipt identity differs")
    finally:
        os.close(descriptor)
    os.fsync(parent_descriptor)
finally:
    os.close(parent_descriptor)
PY

[[ -f "${automation_receipt}" && ! -L "${automation_receipt}" && "$(stat -c '%h %a' -- "${automation_receipt}")" == '1 444' ]] || fail "automation receipt publication differs"
