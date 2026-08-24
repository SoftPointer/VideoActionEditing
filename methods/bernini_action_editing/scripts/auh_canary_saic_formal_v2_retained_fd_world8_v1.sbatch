#!/usr/bin/bash
#SBATCH --job-name=saic-fv2-fd-w8-cny1
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=00:15:00

set -Eeuo pipefail
umask 077
export PATH=/usr/bin:/bin
export LANG=C
export LC_ALL=C

readonly expected_payload_sha=96335bf5f5a0896fdbf7e88dbe774bd5f4c03e0e44ecab96757834b9413d6750
readonly expected_guard_sha=1a38b2ac18f46e818b1596db884b883dff8e0612fcd5b0cf1ed78aca377ac965
readonly expected_runtime_sha=3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36
readonly expected_source_archive_sha=3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b
readonly expected_python=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly expected_python_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly expected_probe_validator_sha=3b5e78a0815fbfdee0404293ad800c640133deacc6b1bfafec12a02ca40ef85b
readonly expected_reexec_nonce=bc4c40f0f87f513706bdf31fb8dca1db4539e801acc4560de761716b414ed169

fail() { echo "[saic-fv2-fd-world8-wrapper] ERROR: $*" >&2; exit 2; }

strict_preamble() {
payload="${SAIC_FV2_FD_CANARY_PAYLOAD:?}"
payload_sha="${SAIC_FV2_FD_CANARY_PAYLOAD_SHA256:?}"
guard="${SAIC_FV2_FD_CANARY_GUARD:?}"
guard_sha="${SAIC_FV2_FD_CANARY_GUARD_SHA256:?}"
runtime="${SAIC_FV2_FD_CANARY_RUNTIME:?}"
runtime_sha="${SAIC_FV2_FD_CANARY_RUNTIME_SHA256:?}"
source_archive="${SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE:?}"
source_archive_sha="${SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE_SHA256:?}"
python_bin="${SAIC_FV2_FD_CANARY_PYTHON:?}"
python_sha="${SAIC_FV2_FD_CANARY_PYTHON_SHA256:?}"
output_parent="${SAIC_FV2_FD_CANARY_OUTPUT_PARENT:?}"
output_parent_device="${SAIC_FV2_FD_CANARY_OUTPUT_PARENT_DEVICE:?}"
output_parent_inode="${SAIC_FV2_FD_CANARY_OUTPUT_PARENT_INODE:?}"
submission_receipt="${SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT:?}"
submission_receipt_device="${SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT_DEVICE:?}"
submission_receipt_inode="${SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT_INODE:?}"
wrapper_path="${SAIC_FV2_FD_CANARY_WRAPPER:?}"
wrapper_sha="${SAIC_FV2_FD_CANARY_WRAPPER_SHA256:?}"
postflight="${SAIC_FV2_FD_CANARY_POSTFLIGHT:?}"
postflight_sha="${SAIC_FV2_FD_CANARY_POSTFLIGHT_SHA256:?}"
release_manifest="${SAIC_FV2_FD_CANARY_RELEASE_MANIFEST:?}"
release_manifest_sha="${SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_SHA256:?}"
release_manifest_digest="${SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_DIGEST:?}"
probe_admission="${SAIC_FV2_FD_CANARY_PROBE_ADMISSION:?}"
probe_admission_sha="${SAIC_FV2_FD_CANARY_PROBE_ADMISSION_SHA256:?}"
probe_admission_digest="${SAIC_FV2_FD_CANARY_PROBE_ADMISSION_DIGEST:?}"
probe_validator="${SAIC_FV2_FD_CANARY_PROBE_VALIDATOR:?}"
probe_validator_sha="${SAIC_FV2_FD_CANARY_PROBE_VALIDATOR_SHA256:?}"

for path in payload guard runtime source_archive python_bin output_parent submission_receipt \
  wrapper_path postflight release_manifest probe_admission probe_validator; do
  [[ "${!path}" == /* ]] || fail "${path} must be absolute"
done
[[ "${SLURM_JOB_ID:?}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
for identity_component in output_parent_device output_parent_inode \
  submission_receipt_device submission_receipt_inode; do
  [[ "${!identity_component}" =~ ^[0-9]+$ ]] || \
    fail "${identity_component} differs"
done
[[ "${payload_sha}" == "${expected_payload_sha}" && \
   "${guard_sha}" == "${expected_guard_sha}" && \
   "${runtime_sha}" == "${expected_runtime_sha}" && \
   "${source_archive_sha}" == "${expected_source_archive_sha}" && \
   "${python_bin}" == "${expected_python}" && \
   "${python_sha}" == "${expected_python_sha}" ]] || fail "pinned input differs"
for release_sha in wrapper_sha postflight_sha release_manifest_sha \
  release_manifest_digest probe_admission_sha probe_admission_digest \
  probe_validator_sha; do
  [[ "${!release_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "${release_sha} differs"
done

python_bin="$(realpath -e -- "${python_bin}")"
output_parent="$(realpath -e -- "${output_parent}")"
check_plain() {
  local path="$1" sha="$2" label="$3"
  [[ -f "${path}" && ! -L "${path}" && "$(stat -c '%h' -- "${path}")" == 1 && \
     "$(sha256sum "${path}" | awk '{print $1}')" == "${sha}" ]] || fail "${label} differs"
}
[[ "${probe_validator_sha}" == "${expected_probe_validator_sha}" ]] || \
  fail "probe validator pin differs"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" && \
   "$(stat -c '%h' -- "${python_bin}")" == 1 && \
   "$(sha256sum "${python_bin}" | awk '{print $1}')" == "${python_sha}" ]] || fail "Python differs"
[[ "${BASH}" == /usr/bin/bash && "$(realpath -e -- "/proc/$$/exe")" == /usr/bin/bash && \
   "$(sha256sum /proc/$$/exe | awk '{print $1}')" == \
   59474588a312b6b6e73e5a42a59bf71e62b55416b6c9d5e4a6e1c630c2a9ecd4 && \
   "$(/usr/bin/bash --version | sha256sum | awk '{print $1}')" == \
   51bd40ffa4710175920033d329a0e9e1667e6b7f56178e302432ff4610d554a7 && \
   "$(/usr/bin/bash --version | sed -n '1p')" == \
   'GNU bash, version 5.1.16(1)-release (x86_64-pc-linux-gnu)' ]] || \
  fail "current compute Bash differs from admitted probe"
[[ "${submission_receipt}" == "${output_parent}/submission-receipt.json" ]] || \
  fail "submission receipt path differs"
owner_uid="$(id -u)"
[[ -d "${output_parent}" && ! -L "${output_parent}" && \
   "$(stat -c '%u:%a' -- "${output_parent}")" == "${owner_uid}:700" ]] || \
  fail "output parent safety differs"
output_root="${output_parent}/job-${SLURM_JOB_ID}"
failure_receipt="${output_parent}/job-${SLURM_JOB_ID}.failure.json"
[[ ! -e "${output_root}" && ! -L "${output_root}" && \
   ! -e "${failure_receipt}" && ! -L "${failure_receipt}" ]] || \
  fail "fresh pre-supervisor namespace differs"
}

# Stage 0 opens every job-executed release source and the own receipt with
# O_NOFOLLOW, marks those exact descriptors inheritable, then re-execs this
# spooled wrapper under the admitted /usr/bin/bash.  Stage 1 never opens those
# logical source paths for execution.
if [[ "${SAIC_FV2_FD_CANARY_REEXEC_STAGE+x}" != x ]]; then
  strict_preamble
  executed_spool="$0"
  exec "${python_bin}" -B - "${guard}" "${guard_sha}" "${payload}" "${payload_sha}" \
    "${probe_validator}" "${probe_validator_sha}" "${submission_receipt}" \
    "${submission_receipt_inode}" "${output_parent}" "${output_parent_inode}" \
    "${failure_receipt}" "${SLURM_JOB_ID}" "${wrapper_sha}" "${runtime_sha}" \
    "${executed_spool}" "${source_archive}" "${source_archive_sha}" <<'PY'
import hashlib, json, os, stat, sys, types
from pathlib import Path

(guard_path, guard_sha, payload_path, payload_sha, validator_path,
 validator_sha, receipt_path, receipt_history_inode, output_parent,
 output_history_inode, failure_path, job, wrapper_sha, runtime_sha,
 executed_spool, source_archive_path, source_archive_sha) = sys.argv[1:]
guard_path = Path(guard_path); payload_path = Path(payload_path)
validator_path = Path(validator_path); receipt_path = Path(receipt_path)
failure_path = Path(failure_path); output_parent = Path(output_parent)
source_archive_path = Path(source_archive_path)
fds = []
guard_module = None
stage = "stage0_retaining_failure_guard"
output_fd = os.open(output_parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
output_before = os.fstat(output_fd); output_leaf = output_parent.lstat()
if (not stat.S_ISDIR(output_before.st_mode) or output_before.st_uid != os.getuid()
        or stat.S_IMODE(output_before.st_mode) != 0o700
        or not stat.S_ISDIR(output_leaf.st_mode) or stat.S_ISLNK(output_leaf.st_mode)
        or (output_before.st_dev, output_before.st_ino)
           != (output_leaf.st_dev, output_leaf.st_ino)
        or output_before.st_ino != int(output_history_inode)):
    raise SystemExit("stage0 output parent identity differs")
output_local_identity = (output_before.st_dev, output_before.st_ino)

def key(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            stat.S_IMODE(info.st_mode), info.st_nlink, info.st_uid)

def read_source(path, expected_sha, label):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    fds.append(fd)
    before = os.fstat(fd); leaf = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
            or not stat.S_ISREG(leaf.st_mode) or stat.S_ISLNK(leaf.st_mode)
            or leaf.st_nlink != 1 or stat.S_IMODE(leaf.st_mode) != 0o444
            or before.st_uid != os.getuid()
            or path.resolve(strict=True) != path
            or (before.st_dev, before.st_ino) != (leaf.st_dev, leaf.st_ino)):
        raise RuntimeError(label + " identity differs")
    chunks = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk: break
        chunks.append(chunk)
    raw = b"".join(chunks); after = os.fstat(fd); leaf_after = path.lstat()
    if (key(before) != key(after) or key(after) != key(leaf_after)
            or path.resolve(strict=True) != path
            or hashlib.sha256(raw).hexdigest() != expected_sha):
        raise RuntimeError(label + " retained bytes differ")
    os.lseek(fd, 0, os.SEEK_SET); os.set_inheritable(fd, True)
    return fd, raw

def publish_failure(message):
    if guard_module is None:
        return
    current = os.fstat(output_fd); leaf = output_parent.lstat()
    if (not stat.S_ISDIR(current.st_mode) or current.st_uid != os.getuid()
            or stat.S_IMODE(current.st_mode) != 0o700
            or not stat.S_ISDIR(leaf.st_mode) or stat.S_ISLNK(leaf.st_mode)
            or (current.st_dev, current.st_ino) != output_local_identity
            or (leaf.st_dev, leaf.st_ino) != output_local_identity
            or current.st_ino != int(output_history_inode)
            or failure_path.parent != output_parent):
        return
    core = {
        "schema_version": "saic-formal-v2-retained-fd-world8-canary-failure-v2",
        "status": "job_execution_failed_before_external_terminal_admission",
        "slurm_job_id": job, "exit_status": 2, "failure_stage": stage,
        "job_success": None, "slurm_terminal_verified": False,
        "formal_admission": False, "guard_sha256": guard_sha,
        "payload_sha256": payload_sha, "wrapper_sha256": wrapper_sha,
        "runtime_sha256": runtime_sha,
        "submission_receipt_path": str(receipt_path),
        "operational_evidence_path": str(output_parent / f"job-{job}" /
                                         "operational-evidence.json"),
        "operational_evidence_may_coexist_but_is_non_authorizing": True,
        "scientific_generation_entered": False,
        "authority": {"scientific": False, "generation": False,
                      "training": False, "publication": False,
                      "formal_job_authorized": False,
                      "operational_gate": False},
    }
    try:
        sealed = guard_module.seal(core)
        raw = json.dumps(
            sealed, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii") + b"\n"
        name = failure_path.name
        fd = os.open(
            name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600, dir_fd=output_fd,
        )
        try:
            observed = os.fstat(fd); public = os.stat(name, dir_fd=output_fd,
                                                      follow_symlinks=False)
            if (not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1
                    or stat.S_IMODE(observed.st_mode) != 0o600
                    or (observed.st_dev, observed.st_ino)
                       != (public.st_dev, public.st_ino)):
                raise RuntimeError("stage0 failure reservation differs")
            view = memoryview(raw)
            while view:
                wrote = os.write(fd, view)
                if wrote <= 0: raise RuntimeError("stage0 failure write stalled")
                view = view[wrote:]
            os.fsync(fd); os.lseek(fd, 0, os.SEEK_SET)
            if os.read(fd, len(raw) + 1) != raw:
                raise RuntimeError("stage0 failure reread differs")
            staged_info = os.fstat(fd); public = os.stat(
                name, dir_fd=output_fd, follow_symlinks=False
            )
            if (stat.S_IMODE(staged_info.st_mode) != 0o600
                    or staged_info.st_nlink != 1
                    or staged_info.st_size != len(raw)
                    or (staged_info.st_dev, staged_info.st_ino)
                       != (public.st_dev, public.st_ino)):
                raise RuntimeError("stage0 failure publication differs")
            os.fsync(output_fd)
            os.fchmod(fd, 0o444)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
    except BaseException:
        pass
    try:
        os.write(2, ("[saic-fv2-fd-world8-wrapper] ERROR: " + stage + ": "
                     + str(message) + "\n").encode("ascii", "backslashreplace"))
    except OSError:
        pass

try:
    guard_fd, guard_raw = read_source(guard_path, guard_sha, "failure guard")
    guard_module = types.ModuleType("fd_canary_stage0_failure_guard")
    exec(compile(guard_raw, "retained-guard-v2", "exec"), guard_module.__dict__)
    stage = "stage0_retaining_executed_spooled_wrapper"
    spool_path = Path(executed_spool)
    spool_fd = os.open(spool_path, os.O_RDONLY | os.O_NOFOLLOW)
    fds.append(spool_fd); spool_before = os.fstat(spool_fd)
    spool_leaf = spool_path.lstat(); spool_chunks = []
    while True:
        chunk = os.read(spool_fd, 1024 * 1024)
        if not chunk: break
        spool_chunks.append(chunk)
    spool_after = os.fstat(spool_fd); spool_leaf_after = spool_path.lstat()
    if (not stat.S_ISREG(spool_before.st_mode) or spool_before.st_nlink != 1
            or stat.S_IMODE(spool_before.st_mode) & 0o022
            or not stat.S_ISREG(spool_leaf.st_mode) or stat.S_ISLNK(spool_leaf.st_mode)
            or spool_leaf.st_nlink != 1
            or key(spool_before) != key(spool_after)
            or key(spool_after) != key(spool_leaf_after)
            or hashlib.sha256(b"".join(spool_chunks)).hexdigest() != wrapper_sha):
        raise RuntimeError("executed spooled wrapper differs")
    os.lseek(spool_fd, 0, os.SEEK_SET); os.set_inheritable(spool_fd, True)
    stage = "stage0_retaining_release_payload"
    payload_fd, _ = read_source(payload_path, payload_sha, "payload")
    stage = "stage0_retaining_external_probe_validator"
    validator_fd, _ = read_source(validator_path, validator_sha, "probe validator")
    stage = "stage0_retaining_exact_formal_source_archive"
    source_archive_fd, _ = read_source(
        source_archive_path, source_archive_sha, "formal source archive"
    )
    stage = "stage0_retaining_own_submission_receipt"
    receipt_fd = os.open(receipt_path, os.O_RDONLY | os.O_NOFOLLOW)
    fds.append(receipt_fd)
    receipt_info = os.fstat(receipt_fd); receipt_leaf = receipt_path.lstat()
    if (not stat.S_ISREG(receipt_info.st_mode) or receipt_info.st_nlink != 1
            or stat.S_IMODE(receipt_info.st_mode) not in {0o600, 0o444}
            or not stat.S_ISREG(receipt_leaf.st_mode)
            or stat.S_ISLNK(receipt_leaf.st_mode) or receipt_leaf.st_nlink != 1
            or receipt_info.st_uid != os.getuid()
            or (receipt_info.st_dev, receipt_info.st_ino)
               != (receipt_leaf.st_dev, receipt_leaf.st_ino)
            or receipt_info.st_ino != int(receipt_history_inode)):
        raise RuntimeError("own submission receipt identity differs")
    output_info = os.fstat(output_fd); output_leaf_after = output_parent.lstat()
    if (not stat.S_ISDIR(output_info.st_mode) or stat.S_ISLNK(output_info.st_mode)
            or output_info.st_uid != os.getuid()
            or stat.S_IMODE(output_info.st_mode) != 0o700
            or (output_info.st_dev, output_info.st_ino) != output_local_identity
            or (output_leaf_after.st_dev, output_leaf_after.st_ino)
               != output_local_identity
            or output_info.st_ino != int(output_history_inode)):
        raise RuntimeError("output parent identity differs")
    os.set_inheritable(receipt_fd, True)
    os.set_inheritable(output_fd, True)
    source_fds = (
        guard_fd, spool_fd, payload_fd, validator_fd, source_archive_fd, receipt_fd
    )
    inherited_fds = source_fds + (output_fd,)
    if (len(set(inherited_fds)) != 7
            or any(not os.get_inheritable(fd) for fd in inherited_fds)
            or any(not stat.S_ISREG(os.fstat(fd).st_mode) for fd in source_fds)
            or not stat.S_ISDIR(os.fstat(output_fd).st_mode)):
        raise RuntimeError("inherited source descriptor closure differs")
    env = dict(os.environ)
    env.update({
        "SAIC_FV2_FD_CANARY_REEXEC_STAGE": (
            "bc4c40f0f87f513706bdf31fb8dca1db4539e801acc4560de761716b414ed169"
        ),
        "SAIC_FV2_FD_CANARY_SOURCE_GUARD_FD": str(guard_fd),
        "SAIC_FV2_FD_CANARY_SOURCE_SPOOL_FD": str(spool_fd),
        "SAIC_FV2_FD_CANARY_SOURCE_PAYLOAD_FD": str(payload_fd),
        "SAIC_FV2_FD_CANARY_SOURCE_PROBE_VALIDATOR_FD": str(validator_fd),
        "SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE_FD": str(source_archive_fd),
        "SAIC_FV2_FD_CANARY_SOURCE_SUBMISSION_RECEIPT_FD": str(receipt_fd),
        "SAIC_FV2_FD_CANARY_STAGE0_OUTPUT_DEVICE": str(output_info.st_dev),
        "SAIC_FV2_FD_CANARY_STAGE0_OUTPUT_INODE": str(output_info.st_ino),
        "SAIC_FV2_FD_CANARY_STAGE0_OUTPUT_FD": str(output_fd),
    })
    os.execve(
        "/usr/bin/bash", ["/usr/bin/bash", f"/proc/self/fd/{spool_fd}"], env
    )
except BaseException as error:
    publish_failure(error)
    os._exit(2)
PY
  exit 2
fi

# The create-only failure supervisor depends only on the admitted Python and
# this retained guard.  Install it before touching payload or validator bytes.
source_guard_fd="${SAIC_FV2_FD_CANARY_SOURCE_GUARD_FD:-}"
stage0_output_fd="${SAIC_FV2_FD_CANARY_STAGE0_OUTPUT_FD:-}"
stage1_job_id="${SLURM_JOB_ID:-}"
failure_python_bin="${expected_python}"
output_parent="${SAIC_FV2_FD_CANARY_OUTPUT_PARENT:-}"
output_parent_inode="${SAIC_FV2_FD_CANARY_OUTPUT_PARENT_INODE:-}"
submission_receipt="${SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT:-}"
failure_guard_sha="${expected_guard_sha}"
payload_sha="${SAIC_FV2_FD_CANARY_PAYLOAD_SHA256:-}"
wrapper_sha="${SAIC_FV2_FD_CANARY_WRAPPER_SHA256:-}"
failure_runtime_sha="${expected_runtime_sha}"
failure_receipt=""
output_root=""
if [[ "${stage1_job_id}" =~ ^[0-9]+$ && "${output_parent}" == /* ]]; then
  failure_receipt="${output_parent}/job-${stage1_job_id}.failure.json"
  output_root="${output_parent}/job-${stage1_job_id}"
fi
source_guard_fd_path="/proc/$$/fd/${source_guard_fd}"
[[ "${source_guard_fd}" =~ ^[0-9]+$ && "${source_guard_fd}" -ge 3 && \
   -f "${source_guard_fd_path}" && "$(stat -Lc '%u:%h:%a' -- "${source_guard_fd_path}")" == \
   "$(id -u):1:444" && \
   "$(sha256sum "${source_guard_fd_path}" | awk '{print $1}')" == \
   "${expected_guard_sha}" ]] || \
  fail "retained supervisor guard differs"
[[ "${stage0_output_fd}" =~ ^[0-9]+$ && "${stage0_output_fd}" -ge 3 && \
   -d "/proc/$$/fd/${stage0_output_fd}" && \
   "$(stat -Lc '%u:%a' -- "/proc/$$/fd/${stage0_output_fd}")" == \
   "$(id -u):700" ]] || fail "retained output directory fd differs"

payload_pid=""
failure_stage="retaining_release_payload"
supervise() {
  local status=$?
  trap - EXIT TERM INT
  if [[ "${payload_pid:-}" =~ ^[0-9]+$ ]]; then
    kill "${payload_pid}" 2>/dev/null || true
    wait "${payload_pid}" 2>/dev/null || true
  fi
  if [[ "${status}" != 0 && -n "${failure_receipt}" && \
        "${stage1_job_id}" =~ ^[0-9]+$ && \
        ! -e "${failure_receipt}" && ! -L "${failure_receipt}" ]]; then
    "${failure_python_bin}" -B - "${source_guard_fd_path}" "${failure_receipt}" \
      "${stage1_job_id}" "${status}" "${failure_guard_sha}" "${payload_sha}" \
      "${wrapper_sha}" "${failure_runtime_sha}" "${submission_receipt}" \
      "${output_root}/operational-evidence.json" "${failure_stage}" \
      "${output_parent}" "${output_parent_inode}" "${stage0_output_fd}" <<'PY' || true
import hashlib, json, os, stat, sys, types
from pathlib import Path
guard_path, receipt = map(Path, sys.argv[1:3])
(job, status, guard_sha, payload_sha, wrapper_sha, runtime_sha, submission,
 evidence, stage, output_parent, output_history_inode, output_fd_number) = sys.argv[3:15]
output_parent = Path(output_parent)
raw = guard_path.read_bytes()
if hashlib.sha256(raw).hexdigest() != guard_sha:
    raise SystemExit("failure publisher retained guard differs")
module = types.ModuleType("fd_canary_supervisor_failure_guard")
exec(compile(raw, "retained-guard-v2", "exec"), module.__dict__)
core = {
    "schema_version": "saic-formal-v2-retained-fd-world8-canary-failure-v2",
    "status": "job_execution_failed_before_external_terminal_admission",
    "slurm_job_id": job, "exit_status": int(status), "failure_stage": stage,
    "job_success": None, "slurm_terminal_verified": False,
    "formal_admission": False,
    "guard_sha256": guard_sha, "payload_sha256": payload_sha,
    "wrapper_sha256": wrapper_sha, "runtime_sha256": runtime_sha,
    "submission_receipt_path": submission,
    "operational_evidence_path": evidence,
    "operational_evidence_may_coexist_but_is_non_authorizing": True,
    "scientific_generation_entered": False,
    "authority": {"scientific": False, "generation": False, "training": False,
                  "publication": False, "formal_job_authorized": False,
                  "operational_gate": False},
}
sealed = module.seal(core)
payload = json.dumps(
    sealed, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
).encode("ascii") + b"\n"
directory = os.dup(int(output_fd_number))
try:
    dinfo = os.fstat(directory); leaf = output_parent.lstat()
    if (not stat.S_ISDIR(dinfo.st_mode) or dinfo.st_uid != os.getuid()
            or stat.S_IMODE(dinfo.st_mode) != 0o700
            or not stat.S_ISDIR(leaf.st_mode) or stat.S_ISLNK(leaf.st_mode)
            or (dinfo.st_dev, dinfo.st_ino) != (leaf.st_dev, leaf.st_ino)
            or dinfo.st_ino != int(output_history_inode)):
        raise SystemExit("failure output parent differs")
    name = receipt.name
    fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600, dir_fd=directory)
    try:
        before = os.fstat(fd); public = os.stat(
            name, dir_fd=directory, follow_symlinks=False
        )
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != 0o600
                or (before.st_dev, before.st_ino) != (public.st_dev, public.st_ino)):
            raise SystemExit("failure reservation differs")
        view = memoryview(payload)
        while view:
            wrote = os.write(fd, view)
            if wrote <= 0: raise SystemExit("failure write stalled")
            view = view[wrote:]
        os.fsync(fd); os.lseek(fd, 0, os.SEEK_SET)
        if os.read(fd, len(payload) + 1) != payload:
            raise SystemExit("failure reread differs")
        staged = os.fstat(fd); public = os.stat(
            name, dir_fd=directory, follow_symlinks=False
        )
        if (stat.S_IMODE(staged.st_mode) != 0o600 or staged.st_nlink != 1
                or staged.st_size != len(payload)
                or (staged.st_dev, staged.st_ino) != (public.st_dev, public.st_ino)):
            raise SystemExit("failure publication differs")
        os.fsync(directory)
        os.fchmod(fd, 0o444)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
finally:
    try:
        os.close(directory)
    except OSError:
        pass
os._exit(0)
PY
  fi
  exit "${status}"
}
trap supervise EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# Only the retained guard is trusted before the trap.  Validate every other
# stage-control fact, local leaf relationship, and inherited FD under it.
failure_stage="validating_stage1_inherited_boundary"
strict_preamble
[[ "${SAIC_FV2_FD_CANARY_REEXEC_STAGE}" == "${expected_reexec_nonce}" ]] || \
  fail "reexec stage nonce differs"
source_spool_fd="${SAIC_FV2_FD_CANARY_SOURCE_SPOOL_FD:?}"
source_payload_fd="${SAIC_FV2_FD_CANARY_SOURCE_PAYLOAD_FD:?}"
source_probe_validator_fd="${SAIC_FV2_FD_CANARY_SOURCE_PROBE_VALIDATOR_FD:?}"
source_archive_fd="${SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE_FD:?}"
source_submission_receipt_fd="${SAIC_FV2_FD_CANARY_SOURCE_SUBMISSION_RECEIPT_FD:?}"
stage0_output_device="${SAIC_FV2_FD_CANARY_STAGE0_OUTPUT_DEVICE:?}"
stage0_output_inode="${SAIC_FV2_FD_CANARY_STAGE0_OUTPUT_INODE:?}"
for inherited_fd in source_guard_fd source_spool_fd source_payload_fd \
  source_probe_validator_fd source_archive_fd source_submission_receipt_fd \
  stage0_output_fd; do
  [[ "${!inherited_fd}" =~ ^[0-9]+$ && "${!inherited_fd}" -ge 3 ]] || \
    fail "${inherited_fd} differs"
done
[[ "$(printf '%s\n' "${source_guard_fd}" "${source_spool_fd}" \
     "${source_payload_fd}" "${source_probe_validator_fd}" \
     "${source_archive_fd}" "${source_submission_receipt_fd}" \
     "${stage0_output_fd}" | sort -u | wc -l | tr -d ' ')" == 7 ]] || \
  fail "stage1 inherited descriptors are not distinct"
source_spool_fd_path="/proc/$$/fd/${source_spool_fd}"
[[ "$(sha256sum "${source_spool_fd_path}" | awk '{print $1}')" == "${wrapper_sha}" ]] || \
  fail "retained executed spool differs"
[[ "${stage0_output_device}" =~ ^[0-9]+$ && \
   "${stage0_output_inode}" =~ ^[1-9][0-9]*$ && \
   "$(stat -c '%d:%i' -- "${output_parent}")" == \
   "${stage0_output_device}:${stage0_output_inode}" && \
   "$(stat -Lc '%d:%i' -- "/proc/$$/fd/${stage0_output_fd}")" == \
   "${stage0_output_device}:${stage0_output_inode}" && \
   "${stage0_output_inode}" == "${output_parent_inode}" ]] || \
  fail "stage0 output parent identity differs"
unset SAIC_FV2_FD_CANARY_REEXEC_STAGE SAIC_FV2_FD_CANARY_SOURCE_GUARD_FD \
  SAIC_FV2_FD_CANARY_SOURCE_SPOOL_FD SAIC_FV2_FD_CANARY_SOURCE_PAYLOAD_FD \
  SAIC_FV2_FD_CANARY_SOURCE_PROBE_VALIDATOR_FD \
  SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE_FD \
  SAIC_FV2_FD_CANARY_SOURCE_SUBMISSION_RECEIPT_FD \
  SAIC_FV2_FD_CANARY_STAGE0_OUTPUT_DEVICE SAIC_FV2_FD_CANARY_STAGE0_OUTPUT_INODE \
  SAIC_FV2_FD_CANARY_STAGE0_OUTPUT_FD

# All remaining release inputs are first touched only under the supervisor.
payload="$(realpath -e -- "${payload}")"
runtime="$(realpath -e -- "${runtime}")"
source_archive="$(realpath -e -- "${source_archive}")"
wrapper_path="$(realpath -e -- "${wrapper_path}")"
postflight="$(realpath -e -- "${postflight}")"
release_manifest="$(realpath -e -- "${release_manifest}")"
probe_admission="$(realpath -e -- "${probe_admission}")"
probe_validator="$(realpath -e -- "${probe_validator}")"
check_plain "${payload}" "${payload_sha}" "payload"
check_plain "${runtime}" "${runtime_sha}" "runtime"
source_archive_fd_path="/proc/$$/fd/${source_archive_fd}"
[[ ! -L "${source_archive}" && \
   "$(stat -Lc '%d:%i:%h' -- "${source_archive}")" == \
   "$(stat -Lc '%d:%i:%h' -- "${source_archive_fd_path}")" && \
   "$(sha256sum "${source_archive_fd_path}" | awk '{print $1}')" == \
   "${source_archive_sha}" ]] || fail "retained source archive differs"
check_plain "${wrapper_path}" "${wrapper_sha}" "logical wrapper"
[[ "$(sha256sum "${source_spool_fd_path}" | awk '{print $1}')" == "${wrapper_sha}" ]] || \
  fail "executed retained spooled wrapper differs"
check_plain "${postflight}" "${postflight_sha}" "postflight"
check_plain "${release_manifest}" "${release_manifest_sha}" "release manifest"
check_plain "${probe_admission}" "${probe_admission_sha}" "probe admission"
source_payload_fd_path="/proc/$$/fd/${source_payload_fd}"
[[ ! -L "${payload}" && \
   "$(stat -Lc '%d:%i:%h' -- "${payload}")" == \
   "$(stat -Lc '%d:%i:%h' -- "${source_payload_fd_path}")" && \
   "$(sha256sum "${source_payload_fd_path}" | awk '{print $1}')" == "${payload_sha}" ]] || \
  fail "retained release payload differs"

# The external compute-Bash admission is a pre-existing, sealed prerequisite.
# Retain and deeply validate it only after the failure supervisor is installed.
failure_stage="retaining_external_probe_validator"
source_probe_validator_fd_path="/proc/$$/fd/${source_probe_validator_fd}"
[[ ! -L "${probe_validator}" && \
   "$(stat -Lc '%d:%i:%h' -- "${probe_validator}")" == \
   "$(stat -Lc '%d:%i:%h' -- "${source_probe_validator_fd_path}")" && \
   "$(sha256sum "${source_probe_validator_fd_path}" | awk '{print $1}')" == \
   "${probe_validator_sha}" ]] || fail "retained probe validator differs"
failure_stage="validating_external_probe_admission"
probe_binding="$(${python_bin} -B "${source_probe_validator_fd_path}" \
  --path "${probe_admission}" --sha256 "${probe_admission_sha}" \
  --digest "${probe_admission_digest}")" || fail "probe validator rejected admission"
[[ "${probe_binding}" == \{*\} ]] || fail "probe admission binding serialization differs"
failure_stage="retaining_own_submission_receipt"
source_submission_receipt_fd_path="/proc/$$/fd/${source_submission_receipt_fd}"
[[ "${source_submission_receipt_fd}" =~ ^[0-9]+$ && \
   "${source_submission_receipt_fd}" -ge 3 && \
   "$(stat -Lc '%d:%i' -- "${submission_receipt}")" == \
   "$(stat -Lc '%d:%i' -- "${source_submission_receipt_fd_path}")" && \
   "$(stat -Lc '%i' -- "${source_submission_receipt_fd_path}")" == \
   "${submission_receipt_inode}" ]] || \
  fail "retained own submission receipt differs"
failure_stage="waiting_for_own_submission_receipt"

# The exported identities are historical login-namespace facts sealed into the
# receipt.  VAST preserves inode numbers across AUH mount namespaces but may
# remap st_dev, so live cross-namespace checks compare the inode only.  The
# retained reader below still compares full dev+ino within this namespace.
[[ -d "${output_parent}" && ! -L "${output_parent}" && \
   "$(stat -c '%i' -- "${output_parent}")" == "${output_parent_inode}" ]] || \
  fail "trusted output parent inode differs before receipt wait"
receipt_identity_and_mode="$(stat -c '%d:%i:%h:%a' -- "${submission_receipt}" 2>/dev/null || true)"
[[ -f "${submission_receipt}" && ! -L "${submission_receipt}" && \
   "${receipt_identity_and_mode}" =~ ^[0-9]+:${submission_receipt_inode}:1:(600|444)$ ]] || \
  fail "trusted submission receipt inode differs before wait"

# Do not create the output/fixture namespace until the exact own receipt is
# terminal.  If a fast-started job only sees the 0600 reservation and times
# out, the already-installed supervisor records the pre-root failure.
submission_binding="$("${python_bin}" -B - "${source_guard_fd_path}" "${submission_receipt}" "${SLURM_JOB_ID}" \
  "${wrapper_path}" "${wrapper_sha}" "${payload}" "${payload_sha}" \
  "${guard}" "${guard_sha}" "${runtime}" "${runtime_sha}" \
  "${source_archive}" "${source_archive_sha}" \
  "${python_bin}" "${python_sha}" "${output_parent}" \
  "${output_parent_device}" "${output_parent_inode}" \
  "${submission_receipt_device}" "${submission_receipt_inode}" \
  "${postflight}" "${postflight_sha}" "${release_manifest}" \
  "${release_manifest_sha}" "${release_manifest_digest}" \
  "${probe_admission}" "${probe_admission_sha}" "${probe_admission_digest}" \
  "${probe_validator}" "${probe_validator_sha}" "${probe_binding}" \
  "${source_submission_receipt_fd}" <<'PY'
import hashlib, json, os, stat, sys, time, types
from pathlib import Path
guard_path, receipt = map(Path, sys.argv[1:3])
(job, wrapper, wrapper_sha, payload, payload_sha, guard, guard_sha,
 runtime, runtime_sha, source_archive, source_archive_sha, python, python_sha,
 output_parent) = sys.argv[3:17]
output_parent_identity = (int(sys.argv[17]), int(sys.argv[18]))
receipt_identity = (int(sys.argv[19]), int(sys.argv[20]))
postflight = Path(sys.argv[21])
postflight_sha = sys.argv[22]
release_manifest = Path(sys.argv[23])
release_manifest_sha = sys.argv[24]
release_manifest_digest = sys.argv[25]
probe_admission = sys.argv[26]
probe_admission_sha = sys.argv[27]
probe_admission_digest = sys.argv[28]
probe_validator = sys.argv[29]
probe_validator_sha = sys.argv[30]
probe_binding = json.loads(sys.argv[31])
receipt_fd_number = int(sys.argv[32])
raw_guard = guard_path.read_bytes(); module = types.ModuleType("fd_canary_receipt_guard")
exec(compile(raw_guard, "guard-v2", "exec"), module.__dict__)
parent_info = Path(output_parent).lstat()
compute_output_parent_identity = (parent_info.st_dev, parent_info.st_ino)
if (not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode)
        or parent_info.st_ino != output_parent_identity[1]):
    raise SystemExit("own fd canary output parent identity differs")
if receipt_fd_number < 3:
    raise SystemExit("own fd canary retained receipt number differs")
descriptor = os.dup(receipt_fd_number)
try:
    first = os.fstat(descriptor)
    local_identity = (first.st_dev, first.st_ino)
    deadline = time.monotonic() + module.WAIT_SECONDS
    while True:
        current = os.fstat(descriptor); leaf = receipt.lstat()
        if (not stat.S_ISREG(current.st_mode) or current.st_nlink != 1
                or not stat.S_ISREG(leaf.st_mode) or stat.S_ISLNK(leaf.st_mode)
                or leaf.st_nlink != 1
                or (leaf.st_dev, leaf.st_ino) != local_identity
                or (current.st_dev, current.st_ino) != local_identity
                or current.st_ino != receipt_identity[1]):
            raise SystemExit("own fd canary retained receipt identity differs")
        mode = stat.S_IMODE(current.st_mode)
        if mode == 0o444 and current.st_size > 0:
            break
        if mode != 0o600 or time.monotonic() >= deadline:
            raise SystemExit("own fd canary retained receipt did not seal")
        time.sleep(0.02)
    os.lseek(descriptor, 0, os.SEEK_SET); chunks = []
    while True:
        chunk = os.read(descriptor, 65536)
        if not chunk: break
        chunks.append(chunk)
    raw = b"".join(chunks)
    after = os.fstat(descriptor); leaf_after = receipt.lstat()
    if ((after.st_dev, after.st_ino) != local_identity
            or (leaf_after.st_dev, leaf_after.st_ino) != local_identity
            or stat.S_IMODE(after.st_mode) != 0o444 or after.st_nlink != 1
            or after.st_size != len(raw)):
        raise SystemExit("own fd canary retained receipt changed")
finally:
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.close(descriptor)
value = module._decode_sealed(
    raw, schema_version="saic-formal-v2-retained-fd-world8-submission-v2",
    exact_fields={"schema_version","status","submission_success","job_success",
                  "submitted_job","request","submission_boundary","inputs",
                  "outputs","authority","receipt_digest"},
)
expected_exports = [
    "SAIC_FV2_FD_CANARY_PAYLOAD","SAIC_FV2_FD_CANARY_PAYLOAD_SHA256",
    "SAIC_FV2_FD_CANARY_GUARD","SAIC_FV2_FD_CANARY_GUARD_SHA256",
    "SAIC_FV2_FD_CANARY_RUNTIME","SAIC_FV2_FD_CANARY_RUNTIME_SHA256",
    "SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE",
    "SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE_SHA256",
    "SAIC_FV2_FD_CANARY_PYTHON","SAIC_FV2_FD_CANARY_PYTHON_SHA256",
    "SAIC_FV2_FD_CANARY_OUTPUT_PARENT","SAIC_FV2_FD_CANARY_OUTPUT_PARENT_DEVICE",
    "SAIC_FV2_FD_CANARY_OUTPUT_PARENT_INODE","SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT",
    "SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT_DEVICE",
    "SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT_INODE",
    "SAIC_FV2_FD_CANARY_WRAPPER","SAIC_FV2_FD_CANARY_WRAPPER_SHA256",
    "SAIC_FV2_FD_CANARY_POSTFLIGHT","SAIC_FV2_FD_CANARY_POSTFLIGHT_SHA256",
    "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST",
    "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_SHA256",
    "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_DIGEST",
    "SAIC_FV2_FD_CANARY_PROBE_VALIDATOR",
    "SAIC_FV2_FD_CANARY_PROBE_VALIDATOR_SHA256",
    "SAIC_FV2_FD_CANARY_PROBE_ADMISSION",
    "SAIC_FV2_FD_CANARY_PROBE_ADMISSION_SHA256",
    "SAIC_FV2_FD_CANARY_PROBE_ADMISSION_DIGEST",
]
manifest_raw = module.read_ready_bytes(
    release_manifest, label="own fd canary release manifest"
)
manifest = module._decode_sealed(
    manifest_raw,
    schema_version="saic-formal-v2-retained-fd-world8-release-manifest-v1",
    exact_fields={"schema_version","status","stem","release_root","output_parent",
                  "inputs","postflight","executables","probe_admission","authority","receipt_digest"},
)
release_root = Path(payload).parent.parent
expected_zero_authority = {
    "scientific":False,"generation":False,"training":False,
    "publication":False,"formal_job_authorized":False,
}
if (hashlib.sha256(manifest_raw).hexdigest() != release_manifest_sha
        or manifest.get("receipt_digest") != release_manifest_digest
        or release_manifest != release_root / "release-manifest.json"
        or manifest.get("status") != "sealed_before_canary_submission"
        or manifest.get("stem") != release_root.name
        or manifest.get("release_root") != str(release_root)
        or manifest.get("output_parent") != output_parent
        or manifest.get("inputs") != {
            "wrapper":wrapper,"wrapper_sha256":wrapper_sha,
            "payload":payload,"payload_sha256":payload_sha,
            "guard":guard,"guard_sha256":guard_sha,
            "runtime":runtime,"runtime_sha256":runtime_sha,
            "source_archive":source_archive,
            "source_archive_sha256":source_archive_sha,
            "probe_validator":probe_validator,
            "probe_validator_sha256":probe_validator_sha,
        }
        or manifest.get("postflight") != {
            "path":str(postflight),"sha256":postflight_sha,
            "sha256_pinned_outside_postflight_source":True,
        }
        or manifest.get("executables") != {
            "python":python,"python_sha256":python_sha,
            "sacct":"/usr/bin/sacct",
            "sacct_sha256":"fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e",
        }
        or manifest.get("probe_admission") != probe_binding
        or manifest.get("authority") != expected_zero_authority):
    raise SystemExit("own fd canary release manifest differs")
submitted = value.get("submitted_job")
if (value.get("status") != "submitted" or value.get("submission_success") is not True
        or value.get("job_success") is not None or not isinstance(submitted, dict)
        or set(submitted) != {"job_id","cluster","stdout_sha256","stderr_sha256"}
        or submitted.get("job_id") != job
        or (submitted.get("cluster") is not None
            and (not isinstance(submitted.get("cluster"), str)
                 or not submitted["cluster"] or "\n" in submitted["cluster"]
                 or ";" in submitted["cluster"]))
        or module.SHA256.fullmatch(str(submitted.get("stdout_sha256", ""))) is None
        or module.SHA256.fullmatch(str(submitted.get("stderr_sha256", ""))) is None
        or value.get("request") != {
            "job_name":"saic-fv2-fd-w8-cny1","partition":"faculty","qos":"bgqos",
            "nodes":1,"ntasks":1,"cpus_per_task":16,"memory":"32G",
            "walltime":"00:15:00","gpu_resource_requested":"gpu:mi210:8",
            "world_topology":"two_concurrent_world4","hold":False,"dependency":None,
            "scientific_generation":False,
        }
        or value.get("submission_boundary") != {
            "environment_replaced":True,"exact_job_export_names":expected_exports,
            "export_all":False,"reservation_created_before_sbatch":True,
            "same_inode_retained":True,"launcher_submitted_from_retained_fd":True,
            "reservation_device":receipt_identity[0],
            "reservation_inode":receipt_identity[1],
            "output_parent_device":output_parent_identity[0],
            "output_parent_inode":output_parent_identity[1],
            "success_mode":"0444",
        }
        or value.get("inputs") != {
            "wrapper":wrapper,"wrapper_sha256":wrapper_sha,
            "payload":payload,"payload_sha256":payload_sha,
            "guard":guard,"guard_sha256":guard_sha,
            "runtime":runtime,"runtime_sha256":runtime_sha,
            "source_archive":source_archive,
            "source_archive_sha256":source_archive_sha,
            "python":python,"python_sha256":python_sha,
            "postflight":str(postflight),"postflight_sha256":postflight_sha,
            "release_manifest":str(release_manifest),
            "release_manifest_file_sha256":release_manifest_sha,
            "release_manifest_digest":release_manifest_digest,
            "probe_validator":probe_validator,
            "probe_validator_sha256":probe_validator_sha,
            "probe_admission":probe_admission,
            "probe_admission_sha256":probe_admission_sha,
            "probe_admission_digest":probe_admission_digest,
            "probe_admission_binding":probe_binding,
        }
        or value.get("outputs") != {
            "output_parent":output_parent,
            "job_output_root":str(Path(output_parent)/f"job-{job}"),
            "submission_receipt":str(receipt),"fresh_before_submission":True,
        }
        or value.get("authority") != {
            "scientific":False,"generation":False,"training":False,
            "publication":False,"formal_job_authorized":False,
        }):
    raise SystemExit("own fd canary submission receipt differs")
print(":".join((hashlib.sha256(raw).hexdigest(), value["receipt_digest"],
                str(local_identity[0]), str(local_identity[1]),
                str(compute_output_parent_identity[0]),
                str(compute_output_parent_identity[1]))))
PY
 )"
[[ "${submission_binding}" =~ ^[0-9a-f]{64}:[0-9a-f]{64}:[0-9]+:[1-9][0-9]*:[0-9]+:[1-9][0-9]*$ ]] || \
  fail "own submission binding differs"
IFS=: read -r initial_submission_file_sha initial_submission_digest \
  compute_submission_receipt_device compute_submission_receipt_inode \
  compute_output_parent_device compute_output_parent_inode <<<"${submission_binding}"
failure_stage="own_submission_receipt_admitted"
[[ ! -e "${failure_receipt}" && ! -L "${failure_receipt}" && \
   ! -e "${output_root}" && ! -L "${output_root}" ]] || \
  fail "fresh fd canary terminal namespace differs"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" ]] || \
  fail "scratch parent differs"

# Build an intentionally mutable /vast fixture from immutable release inputs.
# Copy it from the already-retained release descriptors, open the copies once
# in the trusted spooled wrapper, rename those exact inodes away, and install
# wrong logical-path decoys before exec.  Only retained descriptors may satisfy
# the payload and all eight workers.
fixture_root="${output_parent}/fd-fixture-job-${SLURM_JOB_ID}"
[[ ! -e "${fixture_root}" && ! -L "${fixture_root}" ]] || fail "fd fixture is not fresh"
mkdir -m 0700 -- "${fixture_root}"
payload_logical="${fixture_root}/payload.logical"
guard_logical="${fixture_root}/guard.logical"
cp -- "${source_payload_fd_path}" "${payload_logical}"
cp -- "${source_guard_fd_path}" "${guard_logical}"
chmod 0444 -- "${payload_logical}" "${guard_logical}"
check_plain "${payload_logical}" "${payload_sha}" "fixture payload"
check_plain "${guard_logical}" "${guard_sha}" "fixture guard"
exec {payload_fd}<"${payload_logical}"
exec {guard_fd}<"${guard_logical}"
payload_fd_path="/proc/$$/fd/${payload_fd}"
guard_fd_path="/proc/$$/fd/${guard_fd}"
failure_guard="${guard_fd_path}"
[[ "$(stat -Lc '%d:%i' -- "${payload_logical}")" == "$(stat -Lc '%d:%i' -- "${payload_fd_path}")" && \
   "$(stat -Lc '%d:%i' -- "${guard_logical}")" == "$(stat -Lc '%d:%i' -- "${guard_fd_path}")" ]] || \
  fail "retained descriptor identity differs"
payload_retained="${fixture_root}/payload.original"
guard_retained="${fixture_root}/guard.original"
mv -- "${payload_logical}" "${payload_retained}"
mv -- "${guard_logical}" "${guard_retained}"
printf '%s\n' '#!/usr/bin/env bash' 'exit 97' >"${payload_logical}"
printf '%s\n' 'raise SystemExit(98)' >"${guard_logical}"
chmod 0444 -- "${payload_logical}" "${guard_logical}"
[[ "$(stat -Lc '%d:%i' -- "${payload_retained}")" == "$(stat -Lc '%d:%i' -- "${payload_fd_path}")" && \
   "$(stat -Lc '%d:%i' -- "${guard_retained}")" == "$(stat -Lc '%d:%i' -- "${guard_fd_path}")" ]] || \
  fail "renamed retained descriptor identity differs"

export SAIC_FV2_FD_CANARY_GUARD_FD_PATH="${guard_fd_path}"
export SAIC_FV2_FD_CANARY_PAYLOAD_FD_PATH="${payload_fd_path}"
export SAIC_FV2_FD_CANARY_GUARD_RETAINED_LEAF="${guard_retained}"
export SAIC_FV2_FD_CANARY_GUARD_DECOY_LEAF="${guard_logical}"
export SAIC_FV2_FD_CANARY_PAYLOAD_RETAINED_LEAF="${payload_retained}"
export SAIC_FV2_FD_CANARY_PAYLOAD_DECOY_LEAF="${payload_logical}"
export SAIC_FV2_FD_CANARY_SCRATCH_PARENT="${scratch_parent}"
export SAIC_FV2_FD_CANARY_COMPUTE_OUTPUT_PARENT_DEVICE="${compute_output_parent_device}"
export SAIC_FV2_FD_CANARY_COMPUTE_OUTPUT_PARENT_INODE="${compute_output_parent_inode}"
export SAIC_FV2_FD_CANARY_COMPUTE_SUBMISSION_RECEIPT_DEVICE="${compute_submission_receipt_device}"
export SAIC_FV2_FD_CANARY_COMPUTE_SUBMISSION_RECEIPT_INODE="${compute_submission_receipt_inode}"
export SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT_FD_NUMBER="${source_submission_receipt_fd}"
export SAIC_FV2_FD_CANARY_STAGE0_GUARD_FD_NUMBER="${source_guard_fd}"
export SAIC_FV2_FD_CANARY_STAGE0_SPOOL_FD_NUMBER="${source_spool_fd}"
export SAIC_FV2_FD_CANARY_STAGE0_PAYLOAD_FD_NUMBER="${source_payload_fd}"
export SAIC_FV2_FD_CANARY_STAGE0_PROBE_VALIDATOR_FD_NUMBER="${source_probe_validator_fd}"
export SAIC_FV2_FD_CANARY_STAGE0_SOURCE_ARCHIVE_FD_NUMBER="${source_archive_fd}"
export SAIC_FV2_FD_CANARY_STAGE0_RECEIPT_FD_NUMBER="${source_submission_receipt_fd}"
export SAIC_FV2_FD_CANARY_STAGE0_OUTPUT_FD_NUMBER="${stage0_output_fd}"
export TORCH_DISABLE_SHARE_RDZV_TCP_STORE=0
failure_stage="retained_fd_world8_payload_running"
SAIC_FV2_FD_CANARY_PAYLOAD_FD_NUMBER="${payload_fd}" \
SAIC_FV2_FD_CANARY_GUARD_FD_NUMBER="${guard_fd}" \
/usr/bin/bash -c '
  set -Eeuo pipefail
  payload_number="${SAIC_FV2_FD_CANARY_PAYLOAD_FD_NUMBER:?}"
  guard_number="${SAIC_FV2_FD_CANARY_GUARD_FD_NUMBER:?}"
  [[ "${payload_number}" =~ ^[0-9]+$ && "${guard_number}" =~ ^[0-9]+$ ]]
  export SAIC_FV2_FD_CANARY_PAYLOAD_FD_PATH="/proc/$$/fd/${payload_number}"
  export SAIC_FV2_FD_CANARY_GUARD_FD_PATH="/proc/$$/fd/${guard_number}"
  [[ -r "${SAIC_FV2_FD_CANARY_PAYLOAD_FD_PATH}" && \
     -r "${SAIC_FV2_FD_CANARY_GUARD_FD_PATH}" ]]
  exec /usr/bin/bash "${SAIC_FV2_FD_CANARY_PAYLOAD_FD_PATH}"
' & payload_pid=$!
payload_process_pid="${payload_pid}"
expected_child_payload_fd_path="/proc/${payload_process_pid}/fd/${payload_fd}"
expected_child_guard_fd_path="/proc/${payload_process_pid}/fd/${guard_fd}"
payload_status=0
wait "${payload_pid}" || payload_status=$?
payload_pid=""
[[ "${payload_status}" == 0 ]] || exit "${payload_status}"
failure_stage="operational_evidence_published_accounting_unassessed"
# The job-side evidence is intentionally non-authorizing: a later signal or
# Slurm cancellation may coexist with it.  External postflight combines this
# evidence with authoritative terminal accounting before any operational gate.
"${python_bin}" -B - "${guard_fd_path}" "${output_root}/operational-evidence.json" \
  "${SLURM_JOB_ID}" "${guard_sha}" "${payload_sha}" "${runtime_sha}" \
  "${wrapper_sha}" "${submission_receipt}" "${payload}" "${guard}" \
  "${runtime}" "${source_archive}" "${source_archive_sha}" \
  "${python_bin}" "${python_sha}" "${output_parent}" \
  "${expected_child_guard_fd_path}" "${expected_child_payload_fd_path}" \
  "${guard_fd_path}" "${payload_fd_path}" "${guard_retained}" \
  "${payload_retained}" "${initial_submission_file_sha}" \
  "${initial_submission_digest}" "${output_parent_device}" \
  "${output_parent_inode}" "${submission_receipt_device}" \
  "${submission_receipt_inode}" "${compute_output_parent_device}" \
  "${compute_output_parent_inode}" "${compute_submission_receipt_device}" \
  "${compute_submission_receipt_inode}" "${source_submission_receipt_fd}" \
  "${postflight}" "${postflight_sha}" \
  "${release_manifest}" "${release_manifest_sha}" \
  "${release_manifest_digest}" "${probe_validator}" \
  "${probe_validator_sha}" "${probe_admission}" "${probe_admission_sha}" \
  "${probe_admission_digest}" "${probe_binding}" <<'PY'
import hashlib, json, os, stat, sys, types
from pathlib import Path
guard_path, evidence_path = map(Path, sys.argv[1:3])
job, guard_sha, payload_sha, runtime_sha, wrapper_sha = sys.argv[3:8]
submission_receipt = Path(sys.argv[8])
(payload, guard, runtime, source_archive, source_archive_sha, python,
 python_sha, output_parent,
 expected_child_guard_fd, expected_child_payload_fd, parent_guard_fd,
 parent_payload_fd, expected_guard_retained, expected_payload_retained,
 initial_submission_sha, initial_submission_digest) = sys.argv[9:25]
output_parent_identity = (int(sys.argv[25]), int(sys.argv[26]))
receipt_identity = (int(sys.argv[27]), int(sys.argv[28]))
compute_output_parent_identity = (int(sys.argv[29]), int(sys.argv[30]))
compute_receipt_identity = (int(sys.argv[31]), int(sys.argv[32]))
receipt_fd_number = int(sys.argv[33])
postflight = sys.argv[34]
postflight_sha = sys.argv[35]
release_manifest = sys.argv[36]
release_manifest_sha = sys.argv[37]
release_manifest_digest = sys.argv[38]
probe_validator = sys.argv[39]
probe_validator_sha = sys.argv[40]
probe_admission = sys.argv[41]
probe_admission_sha = sys.argv[42]
probe_admission_digest = sys.argv[43]
probe_binding = json.loads(sys.argv[44])
raw_guard = guard_path.read_bytes()
if hashlib.sha256(raw_guard).hexdigest() != guard_sha:
    raise SystemExit("outer retained guard differs")
module = types.ModuleType("fd_canary_outer_evidence_guard")
exec(compile(raw_guard, "retained-guard-v2", "exec"), module.__dict__)
raw = module.wait_ready_bytes(evidence_path, label="operational evidence")
value = module._decode_sealed(
    raw,
    schema_version="saic-formal-v2-retained-fd-world8-operational-evidence-v1",
    exact_fields={
        "schema_version", "status", "slurm_job_id", "job_success",
        "slurm_terminal_verified", "formal_admission", "topology",
        "requested_gpu_count", "world_size_total", "group_count",
        "rank_packet_count", "unique_actual_master_port_count",
        "collision_receipt_count", "all_launch_rdzv_id_count",
        "all_launch_rdzv_ids_unique", "guard_sha256", "payload_sha256",
        "runtime_sha256", "source_archive_path", "source_archive_sha256",
        "source_archive_retained_fd_number",
        "source_archive_read_from_stage0_retained_fd",
        "archive_member_manifest_sha256", "extracted_tree_manifest_sha256",
        "extracted_tree_manifest_source", "extracted_tree_mode_policy",
        "extracted_tree_entry_set_exact",
        "archive_member_count", "archive_regular_file_count",
        "archive_directory_count", "archive_manifest_canonical_json_bytes",
        "archive_binding_receipt_digest", "runtime_origin_schema_version",
        "runtime_origin_project_module_count", "runtime_origin_project_module_rows",
        "runtime_origin_manifest_sha256", "runtime_origin_receipt_digest",
        "runtime_import_origins_all_from_extracted_archive",
        "runtime_import_checker_isolated_python",
        "runtime_imported_from_pinned_source_archive",
        "runtime_import_scratch_removed_before_evidence",
        "guard_fd_path", "payload_fd_path",
        "replacement_fixture", "guard_retained_leaf", "guard_decoy_leaf",
        "payload_retained_leaf", "payload_decoy_leaf",
        "spooled_wrapper_retained_fd_submission_required",
        "payload_exec_from_parent_retained_fd",
        "guard_read_by_two_background_groups_and_all_workers_from_parent_retained_fd",
        "logical_leaf_replacement_after_fd_open_observed",
        "scratch_cleaned_before_operational_evidence_publication",
        "external_terminal_postflight_admission_required",
        "submission_receipt_path", "submission_receipt_file_sha256",
        "submission_receipt_digest", "output_parent_identity",
        "submission_receipt_identity", "compute_output_parent_identity",
        "compute_submission_receipt_identity",
        "submission_receipt_retained_fd_number",
        "submission_receipt_read_from_retained_fd",
        "stage0_o_nofollow_source_open",
        "stage0_python_execve_bash_handoff",
        "stage0_inherited_source_fd_numbers",
        "stage0_source_fds_distinct", "stage0_output_directory_fd_number",
        "stage0_output_directory_identity",
        "stage0_output_directory_o_nofollow", "wrapper_sha256",
        "probe_admission_binding",
        "scientific_generation_entered", "scientific_output_created",
        "formal_full60_result_claimed", "candidate_rows", "collision_rows",
        "authority", "receipt_digest",
    },
)
if receipt_fd_number < 3:
    raise SystemExit("outer retained submission fd number differs")
submission_descriptor = os.dup(receipt_fd_number)
try:
    before = os.fstat(submission_descriptor); leaf = submission_receipt.lstat()
    if ((before.st_dev, before.st_ino) != compute_receipt_identity
            or (leaf.st_dev, leaf.st_ino) != compute_receipt_identity
            or before.st_ino != receipt_identity[1]
            or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444):
        raise SystemExit("outer retained submission identity differs")
    os.lseek(submission_descriptor, 0, os.SEEK_SET); chunks = []
    while True:
        chunk = os.read(submission_descriptor, 65536)
        if not chunk: break
        chunks.append(chunk)
    submission_raw = b"".join(chunks)
    after = os.fstat(submission_descriptor); leaf_after = submission_receipt.lstat()
    if ((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            or (leaf_after.st_dev, leaf_after.st_ino) != compute_receipt_identity
            or after.st_size != len(submission_raw)):
        raise SystemExit("outer retained submission changed")
finally:
    os.lseek(submission_descriptor, 0, os.SEEK_SET)
    os.close(submission_descriptor)
submission = module._decode_sealed(
    submission_raw,
    schema_version="saic-formal-v2-retained-fd-world8-submission-v2",
    exact_fields={"schema_version", "status", "submission_success", "job_success",
                  "submitted_job", "request", "submission_boundary", "inputs",
                  "outputs", "authority", "receipt_digest"},
)
expected_exports = [
    "SAIC_FV2_FD_CANARY_PAYLOAD", "SAIC_FV2_FD_CANARY_PAYLOAD_SHA256",
    "SAIC_FV2_FD_CANARY_GUARD", "SAIC_FV2_FD_CANARY_GUARD_SHA256",
    "SAIC_FV2_FD_CANARY_RUNTIME", "SAIC_FV2_FD_CANARY_RUNTIME_SHA256",
    "SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE",
    "SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE_SHA256",
    "SAIC_FV2_FD_CANARY_PYTHON", "SAIC_FV2_FD_CANARY_PYTHON_SHA256",
    "SAIC_FV2_FD_CANARY_OUTPUT_PARENT", "SAIC_FV2_FD_CANARY_OUTPUT_PARENT_DEVICE",
    "SAIC_FV2_FD_CANARY_OUTPUT_PARENT_INODE", "SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT",
    "SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT_DEVICE",
    "SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT_INODE",
    "SAIC_FV2_FD_CANARY_WRAPPER", "SAIC_FV2_FD_CANARY_WRAPPER_SHA256",
    "SAIC_FV2_FD_CANARY_POSTFLIGHT", "SAIC_FV2_FD_CANARY_POSTFLIGHT_SHA256",
    "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST",
    "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_SHA256",
    "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_DIGEST",
    "SAIC_FV2_FD_CANARY_PROBE_VALIDATOR",
    "SAIC_FV2_FD_CANARY_PROBE_VALIDATOR_SHA256",
    "SAIC_FV2_FD_CANARY_PROBE_ADMISSION",
    "SAIC_FV2_FD_CANARY_PROBE_ADMISSION_SHA256",
    "SAIC_FV2_FD_CANARY_PROBE_ADMISSION_DIGEST",
]
submitted = submission.get("submitted_job")
zero_authority = {"scientific": False, "generation": False, "training": False,
                  "publication": False, "formal_job_authorized": False}
root = module.exact_directory(evidence_path.parent, label="outer fd canary output root")
parent_info = root.parent.lstat()
if (not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode)
        or (parent_info.st_dev, parent_info.st_ino) != compute_output_parent_identity
        or parent_info.st_ino != output_parent_identity[1]):
    raise SystemExit("outer output parent identity differs")
if sorted(item.name for item in root.iterdir()) != [
    "forbidden-attempts", "logs", "operational-evidence.json", "rendezvous"
]:
    raise SystemExit("outer fd canary output closure differs")
forbidden = module.exact_directory(
    root / "forbidden-attempts", label="outer forbidden output root"
)
if any(forbidden.iterdir()):
    raise SystemExit("outer observed scientific output")
expected_rows = []
ports = set()
all_rdzv_ids = set()
referenced_claims = set()
collision_rows = []
rendezvous = module.exact_directory(root / "rendezvous", label="outer rendezvous root")
if sorted(item.name for item in rendezvous.iterdir()) != ["port-claims", "sp4-a", "sp4-b"]:
    raise SystemExit("outer rendezvous root closure differs")
claim_root = module.exact_directory(rendezvous / "port-claims", label="outer claim root")
completion_fields = {
    "schema_version", "status", "slurm_job_id", "group_id", "candidate_index",
    "candidate_id", "launch_ordinal", "rdzv_id", "actual_master_port",
    "claim_path", "claim_digest", "admission_digest", "rank_packet_digests",
    "guard_observed_via_parent_proc_fd", "scientific_generation_entered",
    "source_archive_sha256", "archive_member_manifest_sha256",
    "extracted_tree_manifest_sha256", "extracted_tree_manifest_source",
    "extracted_tree_mode_policy", "extracted_tree_entry_set_exact",
    "archive_member_count",
    "archive_regular_file_count", "archive_directory_count",
    "archive_binding_receipt_digest", "runtime_origin_manifest_sha256",
    "runtime_origin_project_module_count", "runtime_origin_receipt_digest",
    "runtime_import_origins_all_from_extracted_archive",
    "authority", "receipt_digest",
}
for group in ("sp4-a", "sp4-b"):
    candidate_id = f"fd-canary-{group}-candidate-00"
    candidate_digest = hashlib.sha256(candidate_id.encode("ascii")).hexdigest()
    group_root = module.exact_directory(rendezvous / group, label="outer group root")
    candidate_root = group_root / f"candidate-00-{candidate_digest[:16]}"
    if sorted(item.name for item in group_root.iterdir()) != [candidate_root.name]:
        raise SystemExit("outer candidate root closure differs")
    candidate_root = module.exact_directory(candidate_root, label="outer candidate root")
    launches = sorted(candidate_root.iterdir(), key=lambda item: item.name)
    if (not launches or len(launches) > 16
            or [item.name for item in launches]
            != [f"launch-{ordinal:02d}" for ordinal in range(1, len(launches) + 1)]):
        raise SystemExit("outer launch closure differs")
    for ordinal, life in enumerate(launches[:-1], start=1):
        life = module.exact_directory(life, label="outer collision lifecycle")
        if sorted(item.name for item in life.iterdir()) != ["collision.json", "torchrun.log"]:
            raise SystemExit("outer collision lifecycle closure differs")
        log_info = (life / "torchrun.log").lstat()
        if (not stat.S_ISREG(log_info.st_mode) or stat.S_ISLNK(log_info.st_mode)
                or log_info.st_nlink != 1 or stat.S_IMODE(log_info.st_mode) != 0o444):
            raise SystemExit("outer collision log differs")
        collision = module._validate_collision_for_job_audit(
            life / "collision.json", claim_root=claim_root, slurm_job_id=job
        )
        rdzv_id = f"saic-{job}-{group}-c00-{candidate_digest[:16]}-l{ordinal:02d}"
        if (collision.get("group_id") != group
                or collision.get("candidate_index") != 0
                or collision.get("candidate_id") != candidate_id
                or collision.get("launch_ordinal") != ordinal
                or collision.get("rdzv_id") != rdzv_id
                or rdzv_id in all_rdzv_ids):
            raise SystemExit("outer collision identity differs")
        all_rdzv_ids.add(rdzv_id)
        collision_rows.append({
            "group_id": group, "launch_ordinal": ordinal, "rdzv_id": rdzv_id,
            "collision_digest": collision["receipt_digest"],
        })
    ordinal = len(launches)
    life = module.exact_directory(launches[-1], label="outer successful lifecycle")
    if sorted(item.name for item in life.iterdir()) != [
        "admission.json", "fd-completion.json", "rank-0.json", "rank-1.json",
        "rank-2.json", "rank-3.json", "torchrun.log",
    ]:
        raise SystemExit("outer successful lifecycle closure differs")
    log_info = (life / "torchrun.log").lstat()
    if (not stat.S_ISREG(log_info.st_mode) or stat.S_ISLNK(log_info.st_mode)
            or log_info.st_nlink != 1 or stat.S_IMODE(log_info.st_mode) != 0o444):
        raise SystemExit("outer successful log differs")
    completion = module.wait_load_sealed(
        life / "fd-completion.json",
        schema_version="saic-formal-v2-retained-fd-world4-completion-v1",
        exact_fields=completion_fields, label="outer group completion",
    )
    decision = module.wait_load_sealed(
        life / "admission.json", schema_version=module.DECISION_SCHEMA_VERSION,
        exact_fields=module.DECISION_FIELDS, label="outer group admission",
    )
    rdzv_id = f"saic-{job}-{group}-c00-{candidate_digest[:16]}-l{ordinal:02d}"
    claim_path = claim_root / f"port-{decision['actual_master_port']}.json"
    claim = module.wait_load_sealed(
        claim_path, schema_version=module.CLAIM_SCHEMA_VERSION,
        exact_fields=module.CLAIM_FIELDS, label="outer group claim",
    )
    module._validate_generic_admission(claim, require_rank_packets=True)
    if (completion.get("status") != "retained_guard_fd_world4_runtime_help_completed"
            or completion.get("slurm_job_id") != job
            or completion.get("group_id") != group
            or completion.get("candidate_index") != 0
            or completion.get("candidate_id") != candidate_id
            or completion.get("launch_ordinal") != ordinal
            or completion.get("rdzv_id") != rdzv_id
            or completion.get("actual_master_port") != decision.get("actual_master_port")
            or completion.get("claim_path") != str(claim_path)
            or completion.get("claim_digest") != claim.get("receipt_digest")
            or completion.get("admission_digest") != decision.get("receipt_digest")
            or completion.get("rank_packet_digests") != decision.get("rank_packet_digests")
            or completion.get("source_archive_sha256") != source_archive_sha
            or completion.get("archive_member_manifest_sha256")
            != "1f3c8af23f5b4d416cea04476900c5d479ad3000338746e11f0e655b995b0fcc"
            or module.SHA256.fullmatch(str(
                completion.get("extracted_tree_manifest_sha256", ""))) is None
            or completion.get("extracted_tree_manifest_source")
            != "actual_lstat_after_extraction"
            or completion.get("extracted_tree_mode_policy")
            != "directories_0700_files_0400"
            or completion.get("extracted_tree_entry_set_exact") is not True
            or completion.get("archive_member_count") != 864
            or completion.get("archive_regular_file_count") != 853
            or completion.get("archive_directory_count") != 11
            or module.SHA256.fullmatch(str(
                completion.get("archive_binding_receipt_digest", ""))) is None
            or completion.get("runtime_origin_manifest_sha256")
            != "2e9360581b21b56e6998e1e5db8df98e4cc66acf95fbb7819baffd1161eb98ba"
            or completion.get("runtime_origin_project_module_count") != 14
            or module.SHA256.fullmatch(str(
                completion.get("runtime_origin_receipt_digest", ""))) is None
            or completion.get("runtime_import_origins_all_from_extracted_archive")
            is not True
            or completion.get("guard_observed_via_parent_proc_fd") is not True
            or completion.get("scientific_generation_entered") is not False
            or completion.get("authority") != zero_authority
            or any(decision.get(key) != expected for key, expected in {
                "slurm_job_id": job, "group_id": group, "candidate_index": 0,
                "candidate_id": candidate_id, "launch_ordinal": ordinal,
                "rdzv_id": rdzv_id,
            }.items()) or rdzv_id in all_rdzv_ids):
        raise SystemExit("outer group linkage differs")
    all_rdzv_ids.add(rdzv_id)
    ports.add(completion["actual_master_port"])
    referenced_claims.add(claim_path)
    expected_rows.append({
        "group_id": group, "candidate_id": candidate_id,
        "successful_launch_ordinal": ordinal, "rdzv_id": rdzv_id,
        "actual_master_port": completion["actual_master_port"],
        "claim_path": str(claim_path), "claim_digest": completion["claim_digest"],
        "admission_digest": completion["admission_digest"],
        "rank_packet_digests": completion["rank_packet_digests"],
        "archive_binding_receipt_digest": completion["archive_binding_receipt_digest"],
        "archive_member_manifest_sha256": completion["archive_member_manifest_sha256"],
        "extracted_tree_manifest_sha256": completion[
            "extracted_tree_manifest_sha256"],
        "runtime_origin_manifest_sha256": completion["runtime_origin_manifest_sha256"],
        "runtime_origin_receipt_digest": completion["runtime_origin_receipt_digest"],
        "completion_digest": completion["receipt_digest"],
    })
if (set(claim_root.iterdir()) != referenced_claims or len(referenced_claims) != 2
        or len(ports) != 2):
    raise SystemExit("outer WORLD8 claim/port closure differs")
logs = module.exact_directory(root / "logs", label="outer logs root")
if {item.name for item in logs.iterdir()} != {"sp4-a.group.log", "sp4-b.group.log"}:
    raise SystemExit("outer log closure differs")
for log in logs.iterdir():
    info = log.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o444):
        raise SystemExit("outer log identity differs")
fixture = Path(value.get("replacement_fixture", ""))
fixture = module.exact_directory(fixture, label="outer replacement fixture")
expected_fixture = Path(output_parent := str(root.parent)) / f"fd-fixture-job-{job}"
if fixture != expected_fixture or sorted(item.name for item in fixture.iterdir()) != [
    "guard.logical", "guard.original", "payload.logical", "payload.original"
]:
    raise SystemExit("outer replacement fixture closure differs")
guard_retained = fixture / "guard.original"
guard_decoy = fixture / "guard.logical"
payload_retained = fixture / "payload.original"
payload_decoy = fixture / "payload.logical"
for path in (guard_retained, guard_decoy, payload_retained, payload_decoy):
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o444):
        raise SystemExit("outer replacement fixture identity differs")
if (hashlib.sha256(guard_retained.read_bytes()).hexdigest() != guard_sha
        or hashlib.sha256(payload_retained.read_bytes()).hexdigest() != payload_sha
        or hashlib.sha256(guard_decoy.read_bytes()).hexdigest() == guard_sha
        or hashlib.sha256(payload_decoy.read_bytes()).hexdigest() == payload_sha):
    raise SystemExit("outer retained/decoy hashes differ")
source_archive_fd_number = int(os.environ[
    "SAIC_FV2_FD_CANARY_STAGE0_SOURCE_ARCHIVE_FD_NUMBER"
])
source_archive_fd_path = Path(f"/proc/{os.getpid()}/fd/{source_archive_fd_number}")
source_archive_info = source_archive_fd_path.stat()
source_archive_leaf = Path(source_archive).lstat()
if (not stat.S_ISREG(source_archive_info.st_mode) or source_archive_info.st_nlink != 1
        or stat.S_IMODE(source_archive_info.st_mode) != 0o444
        or not stat.S_ISREG(source_archive_leaf.st_mode)
        or stat.S_ISLNK(source_archive_leaf.st_mode)
        or (source_archive_info.st_dev, source_archive_info.st_ino)
        != (source_archive_leaf.st_dev, source_archive_leaf.st_ino)
        or hashlib.sha256(source_archive_fd_path.read_bytes()).hexdigest()
        != source_archive_sha):
    raise SystemExit("outer retained source archive differs")
if (value.get("status") != "retained_fd_operational_evidence_complete"
        or value.get("slurm_job_id") != job or value.get("job_success") is not None
        or value.get("slurm_terminal_verified") is not False
        or value.get("formal_admission") is not False
        or value.get("topology") != "two_concurrent_world4_on_one_requested_8mi210_node"
        or value.get("requested_gpu_count") != 8 or value.get("world_size_total") != 8
        or value.get("group_count") != 2 or value.get("rank_packet_count") != 8
        or value.get("unique_actual_master_port_count") != 2
        or value.get("collision_receipt_count") != len(collision_rows)
        or value.get("all_launch_rdzv_id_count") != len(all_rdzv_ids)
        or value.get("all_launch_rdzv_ids_unique") is not True
        or value.get("guard_sha256") != guard_sha
        or value.get("payload_sha256") != payload_sha
        or value.get("runtime_sha256") != runtime_sha
        or value.get("source_archive_path") != source_archive
        or value.get("source_archive_sha256") != source_archive_sha
        or value.get("source_archive_retained_fd_number") != source_archive_fd_number
        or value.get("source_archive_read_from_stage0_retained_fd") is not True
        or value.get("archive_member_manifest_sha256")
        != "1f3c8af23f5b4d416cea04476900c5d479ad3000338746e11f0e655b995b0fcc"
        or module.SHA256.fullmatch(str(
            value.get("extracted_tree_manifest_sha256", ""))) is None
        or value.get("extracted_tree_manifest_source")
        != "actual_lstat_after_extraction"
        or value.get("extracted_tree_mode_policy")
        != "directories_0700_files_0400"
        or value.get("extracted_tree_entry_set_exact") is not True
        or value.get("archive_member_count") != 864
        or value.get("archive_regular_file_count") != 853
        or value.get("archive_directory_count") != 11
        or value.get("archive_manifest_canonical_json_bytes") != 173053
        or module.SHA256.fullmatch(str(
            value.get("archive_binding_receipt_digest", ""))) is None
        or value.get("runtime_origin_schema_version")
        != "saic-formal-v2-runtime-import-origin-closure-v1"
        or value.get("runtime_origin_project_module_count") != 14
        or not isinstance(value.get("runtime_origin_project_module_rows"), list)
        or value.get("runtime_origin_manifest_sha256")
        != "2e9360581b21b56e6998e1e5db8df98e4cc66acf95fbb7819baffd1161eb98ba"
        or module.object_sha256(value["runtime_origin_project_module_rows"])
        != value.get("runtime_origin_manifest_sha256")
        or module.SHA256.fullmatch(str(
            value.get("runtime_origin_receipt_digest", ""))) is None
        or value.get("runtime_import_origins_all_from_extracted_archive") is not True
        or value.get("runtime_import_checker_isolated_python") is not True
        or value.get("runtime_imported_from_pinned_source_archive") is not True
        or value.get("runtime_import_scratch_removed_before_evidence") is not True
        or value.get("wrapper_sha256") != wrapper_sha
        or value.get("guard_fd_path") != expected_child_guard_fd
        or value.get("payload_fd_path") != expected_child_payload_fd
        or value.get("replacement_fixture") != str(fixture)
        or value.get("guard_retained_leaf") != str(guard_retained)
        or value.get("guard_decoy_leaf") != str(guard_decoy)
        or value.get("payload_retained_leaf") != str(payload_retained)
        or value.get("payload_decoy_leaf") != str(payload_decoy)
        or value.get("spooled_wrapper_retained_fd_submission_required") is not True
        or value.get("payload_exec_from_parent_retained_fd") is not True
        or value.get("guard_read_by_two_background_groups_and_all_workers_from_parent_retained_fd") is not True
        or value.get("logical_leaf_replacement_after_fd_open_observed") is not True
        or value.get("scratch_cleaned_before_operational_evidence_publication") is not True
        or value.get("external_terminal_postflight_admission_required") is not True
        or value.get("scientific_generation_entered") is not False
        or value.get("scientific_output_created") is not False
        or value.get("formal_full60_result_claimed") is not False
        or value.get("candidate_rows") != expected_rows
        or value.get("collision_rows") != collision_rows
        or value.get("submission_receipt_path") != str(submission_receipt)
        or value.get("submission_receipt_file_sha256")
        != hashlib.sha256(submission_raw).hexdigest()
        or value.get("submission_receipt_digest") != submission["receipt_digest"]
        or value.get("output_parent_identity")
        != f"{output_parent_identity[0]}:{output_parent_identity[1]}"
        or value.get("submission_receipt_identity")
        != f"{receipt_identity[0]}:{receipt_identity[1]}"
        or value.get("compute_output_parent_identity")
        != f"{compute_output_parent_identity[0]}:{compute_output_parent_identity[1]}"
        or value.get("compute_submission_receipt_identity")
        != f"{compute_receipt_identity[0]}:{compute_receipt_identity[1]}"
        or value.get("submission_receipt_retained_fd_number") != receipt_fd_number
        or value.get("submission_receipt_read_from_retained_fd") is not True
        or value.get("stage0_o_nofollow_source_open") is not True
        or value.get("stage0_python_execve_bash_handoff") is not True
        or value.get("stage0_source_fds_distinct") is not True
        or value.get("stage0_inherited_source_fd_numbers") != {
            "guard": int(os.environ["SAIC_FV2_FD_CANARY_STAGE0_GUARD_FD_NUMBER"]),
            "spooled_wrapper": int(os.environ[
                "SAIC_FV2_FD_CANARY_STAGE0_SPOOL_FD_NUMBER"
            ]),
            "payload": int(os.environ["SAIC_FV2_FD_CANARY_STAGE0_PAYLOAD_FD_NUMBER"]),
            "probe_validator": int(os.environ[
                "SAIC_FV2_FD_CANARY_STAGE0_PROBE_VALIDATOR_FD_NUMBER"
            ]),
            "source_archive": source_archive_fd_number,
            "submission_receipt": receipt_fd_number,
        }
        or value.get("stage0_output_directory_fd_number") != int(
            os.environ["SAIC_FV2_FD_CANARY_STAGE0_OUTPUT_FD_NUMBER"]
        )
        or value.get("stage0_output_directory_identity")
        != f"{compute_output_parent_identity[0]}:{compute_output_parent_identity[1]}"
        or value.get("stage0_output_directory_o_nofollow") is not True
        or value.get("probe_admission_binding") != probe_binding
        or hashlib.sha256(submission_raw).hexdigest() != initial_submission_sha
        or submission.get("receipt_digest") != initial_submission_digest
        or not isinstance(submitted, dict)
        or set(submitted) != {"job_id", "cluster", "stdout_sha256", "stderr_sha256"}
        or submitted.get("job_id") != job
        or (submitted.get("cluster") is not None
            and (not isinstance(submitted.get("cluster"), str)
                 or not submitted["cluster"] or "\n" in submitted["cluster"]
                 or ";" in submitted["cluster"]))
        or module.SHA256.fullmatch(str(submitted.get("stdout_sha256", ""))) is None
        or module.SHA256.fullmatch(str(submitted.get("stderr_sha256", ""))) is None
        or submission.get("status") != "submitted"
        or submission.get("submission_success") is not True
        or submission.get("job_success") is not None
        or submission.get("request") != {
            "job_name": "saic-fv2-fd-w8-cny1", "partition": "faculty",
            "qos": "bgqos", "nodes": 1, "ntasks": 1, "cpus_per_task": 16,
            "memory": "32G", "walltime": "00:15:00",
            "gpu_resource_requested": "gpu:mi210:8",
            "world_topology": "two_concurrent_world4", "hold": False,
            "dependency": None, "scientific_generation": False,
        }
        or submission.get("submission_boundary") != {
            "environment_replaced": True, "exact_job_export_names": expected_exports,
            "export_all": False, "reservation_created_before_sbatch": True,
            "same_inode_retained": True, "launcher_submitted_from_retained_fd": True,
            "reservation_device": receipt_identity[0],
            "reservation_inode": receipt_identity[1],
            "output_parent_device": output_parent_identity[0],
            "output_parent_inode": output_parent_identity[1],
            "success_mode": "0444",
        }
        or submission.get("inputs") != {
            "wrapper": str(Path(payload).parent / "auh_canary_saic_formal_v2_retained_fd_world8_v1.sbatch"),
            "wrapper_sha256": wrapper_sha, "payload": payload,
            "payload_sha256": payload_sha, "guard": guard,
            "guard_sha256": guard_sha, "runtime": runtime,
            "runtime_sha256": runtime_sha, "source_archive": source_archive,
            "source_archive_sha256": source_archive_sha, "python": python,
            "python_sha256": python_sha,
            "postflight": postflight, "postflight_sha256": postflight_sha,
            "release_manifest": release_manifest,
            "release_manifest_file_sha256": release_manifest_sha,
            "release_manifest_digest": release_manifest_digest,
            "probe_validator": probe_validator,
            "probe_validator_sha256": probe_validator_sha,
            "probe_admission": probe_admission,
            "probe_admission_sha256": probe_admission_sha,
            "probe_admission_digest": probe_admission_digest,
            "probe_admission_binding": probe_binding,
        }
        or submission.get("outputs") != {
            "output_parent": output_parent,
            "job_output_root": str(Path(output_parent) / f"job-{job}"),
            "submission_receipt": str(submission_receipt),
            "fresh_before_submission": True,
        }
        or submission.get("authority") != zero_authority
        or value.get("authority") != zero_authority):
    raise SystemExit("outer operational evidence linkage differs")
for retained, fd_path in (
    (Path(expected_guard_retained), Path(parent_guard_fd)),
    (Path(expected_payload_retained), Path(parent_payload_fd)),
):
    if ((retained.stat().st_dev, retained.stat().st_ino)
            != (fd_path.stat().st_dev, fd_path.stat().st_ino)):
        raise SystemExit("outer retained fixture descriptor identity differs")
PY
trap - EXIT TERM INT
exit 0
