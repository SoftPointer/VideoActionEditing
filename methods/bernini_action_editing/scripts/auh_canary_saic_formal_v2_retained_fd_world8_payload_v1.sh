#!/usr/bin/bash
# Zero-science operational payload for the formal-v2 retained-FD handoff.

set -Eeuo pipefail
umask 077

readonly expected_guard_sha=1a38b2ac18f46e818b1596db884b883dff8e0612fcd5b0cf1ed78aca377ac965
readonly expected_runtime_sha=3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36
readonly expected_source_archive_sha=3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b
readonly expected_archive_member_manifest_sha=1f3c8af23f5b4d416cea04476900c5d479ad3000338746e11f0e655b995b0fcc
readonly expected_runtime_origin_manifest_sha=2e9360581b21b56e6998e1e5db8df98e4cc66acf95fbb7819baffd1161eb98ba
readonly expected_python_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a

guard_fd_path="${SAIC_FV2_FD_CANARY_GUARD_FD_PATH:?}"
payload_fd_path="${SAIC_FV2_FD_CANARY_PAYLOAD_FD_PATH:?}"
guard_sha="${SAIC_FV2_FD_CANARY_GUARD_SHA256:?}"
payload_sha="${SAIC_FV2_FD_CANARY_PAYLOAD_SHA256:?}"
payload_release="${SAIC_FV2_FD_CANARY_PAYLOAD:?}"
guard_release="${SAIC_FV2_FD_CANARY_GUARD:?}"
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
compute_output_parent_device="${SAIC_FV2_FD_CANARY_COMPUTE_OUTPUT_PARENT_DEVICE:?}"
compute_output_parent_inode="${SAIC_FV2_FD_CANARY_COMPUTE_OUTPUT_PARENT_INODE:?}"
compute_submission_receipt_device="${SAIC_FV2_FD_CANARY_COMPUTE_SUBMISSION_RECEIPT_DEVICE:?}"
compute_submission_receipt_inode="${SAIC_FV2_FD_CANARY_COMPUTE_SUBMISSION_RECEIPT_INODE:?}"
submission_receipt_fd_number="${SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT_FD_NUMBER:?}"
stage0_guard_fd_number="${SAIC_FV2_FD_CANARY_STAGE0_GUARD_FD_NUMBER:?}"
stage0_spool_fd_number="${SAIC_FV2_FD_CANARY_STAGE0_SPOOL_FD_NUMBER:?}"
stage0_payload_fd_number="${SAIC_FV2_FD_CANARY_STAGE0_PAYLOAD_FD_NUMBER:?}"
stage0_probe_validator_fd_number="${SAIC_FV2_FD_CANARY_STAGE0_PROBE_VALIDATOR_FD_NUMBER:?}"
stage0_source_archive_fd_number="${SAIC_FV2_FD_CANARY_STAGE0_SOURCE_ARCHIVE_FD_NUMBER:?}"
stage0_receipt_fd_number="${SAIC_FV2_FD_CANARY_STAGE0_RECEIPT_FD_NUMBER:?}"
stage0_output_fd_number="${SAIC_FV2_FD_CANARY_STAGE0_OUTPUT_FD_NUMBER:?}"
wrapper_sha="${SAIC_FV2_FD_CANARY_WRAPPER_SHA256:?}"
wrapper_path="${SAIC_FV2_FD_CANARY_WRAPPER:?}"
postflight="${SAIC_FV2_FD_CANARY_POSTFLIGHT:?}"
postflight_sha="${SAIC_FV2_FD_CANARY_POSTFLIGHT_SHA256:?}"
release_manifest="${SAIC_FV2_FD_CANARY_RELEASE_MANIFEST:?}"
release_manifest_sha="${SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_SHA256:?}"
release_manifest_digest="${SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_DIGEST:?}"
probe_validator="${SAIC_FV2_FD_CANARY_PROBE_VALIDATOR:?}"
probe_validator_sha="${SAIC_FV2_FD_CANARY_PROBE_VALIDATOR_SHA256:?}"
probe_admission="${SAIC_FV2_FD_CANARY_PROBE_ADMISSION:?}"
probe_admission_sha="${SAIC_FV2_FD_CANARY_PROBE_ADMISSION_SHA256:?}"
probe_admission_digest="${SAIC_FV2_FD_CANARY_PROBE_ADMISSION_DIGEST:?}"
scratch_parent="${SAIC_FV2_FD_CANARY_SCRATCH_PARENT:?}"
guard_retained="${SAIC_FV2_FD_CANARY_GUARD_RETAINED_LEAF:?}"
guard_decoy="${SAIC_FV2_FD_CANARY_GUARD_DECOY_LEAF:?}"
payload_retained="${SAIC_FV2_FD_CANARY_PAYLOAD_RETAINED_LEAF:?}"
payload_decoy="${SAIC_FV2_FD_CANARY_PAYLOAD_DECOY_LEAF:?}"

fail() { echo "[saic-fv2-fd-world8-payload] ERROR: $*" >&2; exit 2; }
for path in guard_fd_path payload_fd_path runtime source_archive python_bin output_parent \
  submission_receipt payload_release guard_release wrapper_path postflight \
  release_manifest probe_validator probe_admission scratch_parent \
  guard_retained guard_decoy payload_retained payload_decoy; do
  [[ "${!path}" == /* ]] || fail "${path} must be absolute"
done
[[ "${SLURM_JOB_ID:?}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
for identity_component in output_parent_device output_parent_inode \
  submission_receipt_device submission_receipt_inode \
  compute_output_parent_device compute_output_parent_inode \
  compute_submission_receipt_device compute_submission_receipt_inode; do
  [[ "${!identity_component}" =~ ^[0-9]+$ ]] || \
    fail "${identity_component} differs"
done
[[ "${guard_fd_path}" =~ ^/proc/[0-9]+/fd/[0-9]+$ && \
   "${payload_fd_path}" =~ ^/proc/[0-9]+/fd/[0-9]+$ ]] || \
  fail "retained fd paths differ"
[[ "${submission_receipt_fd_number}" =~ ^[0-9]+$ && \
   "${submission_receipt_fd_number}" -ge 3 ]] || \
  fail "retained submission receipt fd number differs"
submission_receipt_fd_path="/proc/$$/fd/${submission_receipt_fd_number}"
for stage0_fd in stage0_guard_fd_number stage0_spool_fd_number \
  stage0_payload_fd_number stage0_probe_validator_fd_number \
  stage0_source_archive_fd_number stage0_receipt_fd_number \
  stage0_output_fd_number; do
  [[ "${!stage0_fd}" =~ ^[0-9]+$ && "${!stage0_fd}" -ge 3 ]] || \
    fail "${stage0_fd} differs"
done
[[ "${stage0_receipt_fd_number}" == "${submission_receipt_fd_number}" ]] || \
  fail "stage0 receipt fd binding differs"
[[ "$(printf '%s\n' "${stage0_guard_fd_number}" "${stage0_spool_fd_number}" \
     "${stage0_payload_fd_number}" "${stage0_probe_validator_fd_number}" \
     "${stage0_source_archive_fd_number}" "${stage0_receipt_fd_number}" \
     "${stage0_output_fd_number}" | sort -u | wc -l | tr -d ' ')" == 7 ]] || \
  fail "stage0 inherited fds are not distinct"
stage0_guard_fd_path="/proc/$$/fd/${stage0_guard_fd_number}"
stage0_payload_fd_path="/proc/$$/fd/${stage0_payload_fd_number}"
stage0_probe_validator_fd_path="/proc/$$/fd/${stage0_probe_validator_fd_number}"
stage0_source_archive_fd_path="/proc/$$/fd/${stage0_source_archive_fd_number}"
[[ "${guard_sha}" == "${expected_guard_sha}" && \
   "${runtime_sha}" == "${expected_runtime_sha}" && \
   "${source_archive_sha}" == "${expected_source_archive_sha}" && \
   "${python_sha}" == "${expected_python_sha}" ]] || fail "pinned SHA differs"
for release_sha in payload_sha postflight_sha release_manifest_sha \
  release_manifest_digest wrapper_sha probe_validator_sha probe_admission_sha \
  probe_admission_digest; do
  [[ "${!release_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "${release_sha} differs"
done

runtime="$(realpath -e -- "${runtime}")"
source_archive="$(realpath -e -- "${source_archive}")"
python_bin="$(realpath -e -- "${python_bin}")"
output_parent="$(realpath -e -- "${output_parent}")"
scratch_parent="$(realpath -e -- "${scratch_parent}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" && \
   "$(stat -c '%d:%i' -- "${output_parent}")" == \
   "${compute_output_parent_device}:${compute_output_parent_inode}" && \
   "${compute_output_parent_inode}" == "${output_parent_inode}" ]] || \
  fail "output parent retained identity differs"
[[ -f "${submission_receipt}" && ! -L "${submission_receipt}" && \
   "$(stat -c '%d:%i:%h:%a' -- "${submission_receipt}")" == \
   "${compute_submission_receipt_device}:${compute_submission_receipt_inode}:1:444" && \
   "$(stat -Lc '%d:%i:%h:%a' -- "${submission_receipt_fd_path}")" == \
   "${compute_submission_receipt_device}:${compute_submission_receipt_inode}:1:444" && \
   "${compute_submission_receipt_inode}" == "${submission_receipt_inode}" ]] || \
  fail "submission receipt retained identity differs"
[[ -d "${scratch_parent}" && ! -L "${scratch_parent}" ]] || fail "scratch parent differs"
[[ -r "${guard_fd_path}" && -f "${guard_fd_path}" && \
   "$(stat -Lc '%h' -- "${guard_fd_path}")" == 1 && \
   "$(sha256sum "${guard_fd_path}" | awk '{print $1}')" == "${guard_sha}" ]] || \
  fail "retained guard fd differs after exec"
[[ -r "${payload_fd_path}" && -f "${payload_fd_path}" && \
   "$(stat -Lc '%h' -- "${payload_fd_path}")" == 1 && \
   "$(sha256sum "${payload_fd_path}" | awk '{print $1}')" == "${payload_sha}" ]] || \
  fail "retained payload fd differs after exec"
[[ "$(sha256sum "${stage0_guard_fd_path}" | awk '{print $1}')" == "${guard_sha}" && \
   "$(sha256sum "${stage0_payload_fd_path}" | awk '{print $1}')" == "${payload_sha}" && \
   "$(sha256sum "${stage0_probe_validator_fd_path}" | awk '{print $1}')" == \
   "${probe_validator_sha}" && \
   "$(sha256sum "${stage0_source_archive_fd_path}" | awk '{print $1}')" == \
   "${source_archive_sha}" ]] || fail "stage0 inherited source bytes differ"
[[ "$(sha256sum "$0" | awk '{print $1}')" == "${payload_sha}" ]] || \
  fail "executed payload bytes differ"
[[ -f "${runtime}" && ! -L "${runtime}" && \
   "$(stat -c '%h' -- "${runtime}")" == 1 && \
   "$(sha256sum "${runtime}" | awk '{print $1}')" == "${runtime_sha}" ]] || \
  fail "runtime differs"
[[ -f "${source_archive}" && ! -L "${source_archive}" && \
   "$(stat -Lc '%d:%i:%h:%a' -- "${source_archive}")" == \
   "$(stat -Lc '%d:%i:%h:%a' -- "${stage0_source_archive_fd_path}")" && \
   "$(sha256sum "${stage0_source_archive_fd_path}" | awk '{print $1}')" == \
   "${source_archive_sha}" ]] || fail "retained source archive differs"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" && \
   "$(stat -c '%h' -- "${python_bin}")" == 1 && \
   "$(sha256sum "${python_bin}" | awk '{print $1}')" == "${python_sha}" ]] || \
  fail "Python differs"
[[ -f "${guard_retained}" && -f "${payload_retained}" && \
   "$(stat -Lc '%d:%i' -- "${guard_retained}")" == "$(stat -Lc '%d:%i' -- "${guard_fd_path}")" && \
   "$(stat -Lc '%d:%i' -- "${payload_retained}")" == "$(stat -Lc '%d:%i' -- "${payload_fd_path}")" && \
   "$(sha256sum "${guard_retained}" | awk '{print $1}')" == "${guard_sha}" && \
   "$(sha256sum "${payload_retained}" | awk '{print $1}')" == "${payload_sha}" ]] || \
  fail "retained renamed leaves differ"
[[ -f "${guard_decoy}" && -f "${payload_decoy}" && \
   "$(sha256sum "${guard_decoy}" | awk '{print $1}')" != "${guard_sha}" && \
   "$(sha256sum "${payload_decoy}" | awk '{print $1}')" != "${payload_sha}" ]] || \
  fail "logical leaf replacement proof differs"
[[ "$(stat -Lc '%d:%i' -- "${guard_decoy}")" != "$(stat -Lc '%d:%i' -- "${guard_fd_path}")" && \
   "$(stat -Lc '%d:%i' -- "${payload_decoy}")" != "$(stat -Lc '%d:%i' -- "${payload_fd_path}")" ]] || \
  fail "logical leaf replacement inode differs"

output_root="${output_parent}/job-${SLURM_JOB_ID}"
group_a_pid=""; group_b_pid=""; scratch=""
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  for child in "${group_a_pid:-}" "${group_b_pid:-}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then kill "${child}" 2>/dev/null || true; wait "${child}" 2>/dev/null || true; fi
  done
  if [[ -n "${scratch:-}" ]]; then
    case "${scratch}" in
      "${scratch_parent%/}/saic-fv2-fd-canary-${SLURM_JOB_ID}."*)
        rm -rf -- "${scratch}" || true
        ;;
    esac
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output is not fresh"
scratch="$(mktemp -d "${scratch_parent%/}/saic-fv2-fd-canary-${SLURM_JOB_ID}.XXXXXX")"
source_tree="${scratch}/source"
mkdir -m 0700 -- "${source_tree}"
archive_binding="${scratch}/archive-binding.json"
runtime_origin="${scratch}/runtime-origin.json"
"${python_bin}" -I -B - "${stage0_source_archive_fd_number}" \
  "${source_archive_sha}" "${source_tree}" "${runtime_sha}" \
  "${expected_archive_member_manifest_sha}" "${archive_binding}" <<'PY'
import hashlib, json, os, stat, sys, tarfile
from pathlib import Path, PurePosixPath

archive_fd_number = int(sys.argv[1])
expected_archive_sha = sys.argv[2]
destination = Path(sys.argv[3])
expected_runtime_sha = sys.argv[4]
expected_manifest_sha = sys.argv[5]
binding_path = Path(sys.argv[6])
if archive_fd_number < 3:
    raise SystemExit("source archive retained fd number differs")
descriptor = os.dup(archive_fd_number)
try:
    before = os.fstat(descriptor)
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444):
        raise SystemExit("source archive descriptor identity differs")
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    after = os.fstat(descriptor)
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or digest.hexdigest() != expected_archive_sha):
        raise SystemExit("source archive changed during retained read")
    os.lseek(descriptor, 0, os.SEEK_SET)
    with os.fdopen(os.dup(descriptor), "rb") as source:
        with tarfile.open(fileobj=source, mode="r:") as bundle:
            if bundle.pax_headers.get("comment") != "20c2193954e780e9654347754b1485f3492fbea5":
                raise SystemExit("source archive revision differs")
            members = bundle.getmembers()
            if len(members) != 864:
                raise SystemExit("source archive member count differs")
            normalized = set()
            member_types = {}
            rows = []
            for member in members:
                name = member.name
                if (not name or name == "." or "\\" in name or "\x00" in name):
                    raise SystemExit("source archive member name differs")
                relative = PurePosixPath(member.name)
                if (relative.is_absolute() or relative.as_posix() != name
                        or any(part in {"", ".", ".."} for part in relative.parts)
                        or not relative.parts or relative.parts[0] != "methods"
                        or member.issym() or member.islnk() or member.isdev()
                        or member.isfifo()):
                    raise SystemExit("source archive member escaped")
                if not (member.isdir() or member.isfile()):
                    raise SystemExit("source archive member type differs")
                if name in normalized:
                    raise SystemExit("source archive normalized member duplicated")
                normalized.add(name)
                kind = "directory" if member.isdir() else "file"
                member_types[name] = kind
                row = {"name": name, "type": kind, "mode": member.mode,
                       "size": member.size}
                if member.isfile():
                    extracted = bundle.extractfile(member)
                    if extracted is None:
                        raise SystemExit("source archive member is unreadable")
                    try:
                        member_raw = extracted.read()
                    finally:
                        extracted.close()
                    if len(member_raw) != member.size:
                        raise SystemExit("source archive member size differs")
                    row["sha256"] = hashlib.sha256(member_raw).hexdigest()
                rows.append(row)
            for name, kind in member_types.items():
                parts = PurePosixPath(name).parts
                for index in range(1, len(parts)):
                    ancestor = "/".join(parts[:index])
                    if member_types.get(ancestor) != "directory":
                        raise SystemExit("source archive ancestor type differs")
                if kind == "file" and any(
                    other.startswith(name + "/") for other in member_types
                ):
                    raise SystemExit("source archive file/directory conflict")
            canonical_rows = json.dumps(
                rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            manifest_sha = hashlib.sha256(canonical_rows).hexdigest()
            file_count = sum(row["type"] == "file" for row in rows)
            directory_count = sum(row["type"] == "directory" for row in rows)
            if (manifest_sha != expected_manifest_sha or file_count != 853
                    or directory_count != 11 or len(canonical_rows) != 173053):
                raise SystemExit("source archive member manifest differs")
            bundle.members = members
            for member, row in zip(members, rows):
                relative = PurePosixPath(member.name)
                target = destination.joinpath(*relative.parts)
                target.relative_to(destination)
                if member.isdir():
                    target.mkdir(mode=0o700, exist_ok=False)
                    continue
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                extracted = bundle.extractfile(member)
                if extracted is None:
                    raise SystemExit("source archive member is unreadable")
                output = os.open(
                    target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o400,
                )
                try:
                    with os.fdopen(output, "wb") as handle:
                        member_digest = hashlib.sha256()
                        size = 0
                        while True:
                            chunk = extracted.read(1024 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)
                            member_digest.update(chunk); size += len(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
                        if (size != row["size"]
                                or member_digest.hexdigest() != row["sha256"]):
                            raise SystemExit("extracted member bytes differ")
                finally:
                    extracted.close()
            observed = {}
            for current, directory_names, file_names in os.walk(
                destination, topdown=True, followlinks=False
            ):
                current_path = Path(current)
                for leaf_name, kind in [(name, "directory") for name in directory_names] + [
                    (name, "file") for name in file_names
                ]:
                    leaf = current_path / leaf_name
                    info = leaf.lstat()
                    relative_name = leaf.relative_to(destination).as_posix()
                    required_mode = 0o700 if kind == "directory" else 0o400
                    if ((kind == "directory" and not stat.S_ISDIR(info.st_mode))
                            or (kind == "file" and not stat.S_ISREG(info.st_mode))
                            or stat.S_ISLNK(info.st_mode)
                            or stat.S_IMODE(info.st_mode) != required_mode
                            or (kind == "file" and info.st_nlink != 1)):
                        raise SystemExit("extracted tree inode differs")
                    observed[relative_name] = kind
            if observed != member_types:
                raise SystemExit("extracted tree member set differs")
            extracted_rows = []
            for row in rows:
                target = destination.joinpath(*PurePosixPath(row["name"]).parts)
                info = target.lstat()
                reproduced = {
                    "name": row["name"], "type": row["type"],
                    "mode": stat.S_IMODE(info.st_mode), "size": info.st_size,
                }
                if row["type"] == "file":
                    reproduced["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
                    if (reproduced["size"] != row["size"]
                            or reproduced["sha256"] != row["sha256"]):
                        raise SystemExit("extracted file metadata differs")
                extracted_rows.append(reproduced)
            extracted_raw = json.dumps(
                extracted_rows, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False,
            ).encode("ascii")
            extracted_sha = hashlib.sha256(extracted_raw).hexdigest()
    os.lseek(descriptor, 0, os.SEEK_SET)
    final_archive_digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        final_archive_digest.update(chunk)
    final_archive = os.fstat(descriptor)
    if ((final_archive.st_dev, final_archive.st_ino, final_archive.st_size,
         final_archive.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            or final_archive_digest.hexdigest() != expected_archive_sha):
        raise SystemExit("source archive changed during extraction")
    runtime = (
        destination / "methods/bernini_action_editing/"
        "generate_saic_pure_t2v_event_bank_topup_v2.py"
    )
    if hashlib.sha256(runtime.read_bytes()).hexdigest() != expected_runtime_sha:
        raise SystemExit("extracted runtime bytes differ")
    core = {
        "schema_version": "saic-formal-v2-source-archive-extraction-binding-v1",
        "status": "exact_formal_source_archive_extracted_and_verified",
        "source_archive_sha256": expected_archive_sha,
        "archive_member_manifest_sha256": manifest_sha,
        "extracted_tree_manifest_sha256": extracted_sha,
        "extracted_tree_manifest_source": "actual_lstat_after_extraction",
        "extracted_tree_mode_policy": "directories_0700_files_0400",
        "extracted_tree_entry_set_exact": True,
        "archive_member_count": len(rows),
        "archive_regular_file_count": file_count,
        "archive_directory_count": directory_count,
        "archive_manifest_canonical_json_bytes": len(canonical_rows),
        "source_archive_read_from_stage0_retained_fd": True,
        "authority": {"scientific": False, "generation": False,
                      "training": False, "publication": False,
                      "formal_job_authorized": False},
    }
    core["receipt_digest"] = hashlib.sha256(json.dumps(
        core, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")).hexdigest()
    payload = json.dumps(core, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
    output = os.open(binding_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                     os.O_NOFOLLOW, 0o400)
    try:
        view = memoryview(payload)
        while view:
            wrote = os.write(output, view)
            if wrote <= 0: raise SystemExit("archive binding write stalled")
            view = view[wrote:]
        os.fsync(output); os.fchmod(output, 0o444)
    finally:
        os.close(output)
finally:
    os.close(descriptor)
PY
runtime_import="${source_tree}/methods/bernini_action_editing/generate_saic_pure_t2v_event_bank_topup_v2.py"
[[ -f "${runtime_import}" && ! -L "${runtime_import}" && \
   "$(sha256sum "${runtime_import}" | awk '{print $1}')" == "${runtime_sha}" ]] || \
  fail "extracted runtime import closure differs"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  "${python_bin}" -I -B - "${runtime_import}" \
  "${source_tree}/methods/bernini_action_editing" "${runtime_sha}" \
  "${source_archive_sha}" "${expected_archive_member_manifest_sha}" \
  "${expected_runtime_origin_manifest_sha}" "${runtime_origin}" <<'PY'
import hashlib, importlib.util, json, os, stat, sys
from pathlib import Path

runtime, method_root = map(Path, sys.argv[1:3])
runtime_sha, archive_sha, archive_manifest_sha, expected_origin_sha = sys.argv[3:7]
receipt_path = Path(sys.argv[7])
method_root = method_root.resolve(strict=True)
if runtime.resolve(strict=True).parent != method_root:
    raise SystemExit("runtime checker method root differs")
canonical_module_name = "generate_saic_pure_t2v_event_bank_topup_v2"
if runtime.name != canonical_module_name + ".py" or canonical_module_name in sys.modules:
    raise SystemExit("runtime checker canonical module identity differs")
spec = importlib.util.spec_from_file_location(canonical_module_name, runtime)
if spec is None or spec.loader is None:
    raise SystemExit("runtime checker import spec differs")
module = importlib.util.module_from_spec(spec)
sys.modules[canonical_module_name] = module
try:
    spec.loader.exec_module(module)
except BaseException:
    if sys.modules.get(canonical_module_name) is module:
        del sys.modules[canonical_module_name]
    raise
if sys.modules.get(canonical_module_name) is not module:
    raise SystemExit("runtime checker canonical module registration differs")
try:
    result = module.main(["--help"])
except SystemExit as error:
    result = error.code
if result not in (None, 0):
    raise SystemExit("runtime help checker failed")
expected_modules = {
    "build_saic_reversible_source_set_v1", "generate_saic_pure_t2v_event_bank_v1",
    "infer_lora", "infer_native_identity_generation_canary",
    "infer_pair_v5_t2v_calibration_bank", "infer_source_kv_carrier_oracle",
    "infer_source_value_residual_oracle", "pair_v5_t2v_calibration_bank_spec",
    "saic_pure_t2v_event_bank_topup_v2", "saic_pure_t2v_event_bank_v1",
    "source_kv_replay", "source_kv_route_batches", "source_value_residual",
    "train_lora",
}
rows = []
for module_name, imported in sorted(sys.modules.items()):
    origin = getattr(imported, "__file__", None)
    if not isinstance(origin, str):
        continue
    try:
        path = Path(origin).resolve(strict=True)
        relative = path.relative_to(method_root).as_posix()
    except (FileNotFoundError, OSError, ValueError):
        continue
    info = path.lstat()
    if (module_name not in expected_modules or relative != module_name + ".py"
            or not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o400):
        raise SystemExit("runtime project import origin differs")
    rows.append({"module": module_name, "relative_path": relative,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
raw_rows = json.dumps(rows, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
origin_sha = hashlib.sha256(raw_rows).hexdigest()
if ({row["module"] for row in rows} != expected_modules or len(rows) != 14
        or origin_sha != expected_origin_sha or len(raw_rows) != 2331
        or hashlib.sha256(runtime.read_bytes()).hexdigest() != runtime_sha):
    raise SystemExit("runtime recursive import closure differs")
core = {
    "schema_version": "saic-formal-v2-runtime-import-origin-closure-v1",
    "status": "isolated_runtime_help_import_closure_verified",
    "source_archive_sha256": archive_sha,
    "archive_member_manifest_sha256": archive_manifest_sha,
    "runtime_sha256": runtime_sha,
    "runtime_relative_path": "methods/bernini_action_editing/" + runtime.name,
    "project_module_count": len(rows), "project_module_rows": rows,
    "project_module_manifest_sha256": origin_sha,
    "all_project_module_origins_from_extracted_archive": True,
    "isolated_python_no_environment_path": True,
    "runtime_help_exit_status": 0,
    "authority": {"scientific": False, "generation": False,
                  "training": False, "publication": False,
                  "formal_job_authorized": False},
}
core["receipt_digest"] = hashlib.sha256(json.dumps(
    core, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
).encode("ascii")).hexdigest()
payload = json.dumps(core, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                     os.O_NOFOLLOW, 0o400)
try:
    view = memoryview(payload)
    while view:
        wrote = os.write(descriptor, view)
        if wrote <= 0: raise SystemExit("runtime origin receipt write stalled")
        view = view[wrote:]
    os.fsync(descriptor); os.fchmod(descriptor, 0o444)
finally:
    os.close(descriptor)
PY
mkdir -m 0700 -- "${output_root}" "${output_root}/rendezvous" \
  "${output_root}/rendezvous/port-claims" "${output_root}/logs" \
  "${output_root}/forbidden-attempts"
claim_root="${output_root}/rendezvous/port-claims"
claim_identity="$(stat -Lc '%d:%i' -- "${claim_root}")"
export TORCH_DISABLE_SHARE_RDZV_TCP_STORE=0

run_group() (
  set -Eeuo pipefail
  local group_id="$1" visible="$2"
  local candidate_id="fd-canary-${group_id}-candidate-00"
  local candidate_digest rdzv_id group_root candidate_root life life_identity
  local log status ordinal collision_path completed=false
  candidate_digest="$(printf '%s' "${candidate_id}" | sha256sum | awk '{print $1}')"
  group_root="${output_root}/rendezvous/${group_id}"
  candidate_root="${group_root}/candidate-00-${candidate_digest:0:16}"
  mkdir -m 0700 -p -- "${candidate_root}"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible}"
  for ((ordinal=1; ordinal<=16; ordinal++)); do
    life="${candidate_root}/launch-$(printf '%02d' "${ordinal}")"
    mkdir -m 0700 -- "${life}"
    life_identity="$(stat -Lc '%d:%i' -- "${life}")"
    rdzv_id="saic-${SLURM_JOB_ID}-${group_id}-c00-${candidate_digest:0:16}-l$(printf '%02d' "${ordinal}")"
    log="${life}/torchrun.log"
    if env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
      "${python_bin}" -I -B -m torch.distributed.run \
      --nproc-per-node=4 --max-restarts=0 \
      --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 \
      --rdzv-id="${rdzv_id}" --local-addr=127.0.0.1 \
      "${guard_fd_path}" worker --runtime "${runtime_import}" \
      --expected-runtime-sha256 "${runtime_sha}" --slurm-job-id "${SLURM_JOB_ID}" \
      --group-id "${group_id}" --candidate-index 0 --candidate-id "${candidate_id}" \
      --launch-ordinal "${ordinal}" --expected-rdzv-id "${rdzv_id}" \
      --claim-root "${claim_root}" --claim-root-identity "${claim_identity}" \
      --lifecycle-dir "${life}" --lifecycle-dir-identity "${life_identity}" \
      -- --help >"${log}" 2>&1; then status=0; else status=$?; fi
    chmod 0444 -- "${log}"
    if [[ "${status}" != 0 ]]; then
      [[ "${status}" == 86 ]] || \
        fail "${group_id} ordinary torchrun failure status=${status}; collision admission forbidden"
      collision_path="${life}/collision.json"
      [[ -f "${collision_path}" && ! -L "${collision_path}" && \
         "$(stat -c '%h:%a' -- "${collision_path}")" == "1:444" && \
         "$(stat -c '%s' -- "${collision_path}")" -gt 0 ]] || \
        fail "${group_id} status=86 lacks an initially sealed collision leaf"
      if "${python_bin}" -B "${guard_fd_path}" admit-collision \
        --expected-runtime-sha256 "${runtime_sha}" --slurm-job-id "${SLURM_JOB_ID}" \
        --group-id "${group_id}" --candidate-index 0 --candidate-id "${candidate_id}" \
        --launch-ordinal "${ordinal}" --expected-rdzv-id "${rdzv_id}" \
        --claim-root "${claim_root}" --claim-root-identity "${claim_identity}" \
        --lifecycle-dir "${life}" --lifecycle-dir-identity "${life_identity}" \
        --candidate-output "${output_root}/forbidden-attempts/${candidate_id}" >/dev/null; then
        [[ "${ordinal}" -lt 16 ]] || fail "${group_id} collision budget exhausted"
        continue
      fi
      fail "${group_id} torchrun failed without valid collision status=${status}"
    fi
    "${python_bin}" -B - "${guard_fd_path}" "${life}" "${claim_root}" \
      "${SLURM_JOB_ID}" "${group_id}" "${candidate_id}" "${ordinal}" \
      "${rdzv_id}" "${archive_binding}" "${runtime_origin}" <<'PY'
import os, sys, types
from pathlib import Path
guard_path, life, claim_root = map(Path, sys.argv[1:4])
job, group, candidate, ordinal, rdzv_id = sys.argv[4:9]
archive_binding_path, runtime_origin_path = map(Path, sys.argv[9:11])
raw = guard_path.read_bytes(); module = types.ModuleType("fd_canary_group_guard")
exec(compile(raw, "retained-guard-v2", "exec"), module.__dict__)
decision = module.wait_load_sealed(
    life / "admission.json", schema_version=module.DECISION_SCHEMA_VERSION,
    exact_fields=module.DECISION_FIELDS, label="fd canary admission",
)
claim_path = claim_root / f"port-{decision['actual_master_port']}.json"
claim = module.wait_load_sealed(
    claim_path, schema_version=module.CLAIM_SCHEMA_VERSION,
    exact_fields=module.CLAIM_FIELDS, label="fd canary claim",
)
module._validate_generic_admission(claim, require_rank_packets=True)
archive_binding = module.load_sealed(
    archive_binding_path,
    schema_version="saic-formal-v2-source-archive-extraction-binding-v1",
    exact_fields={"schema_version", "status", "source_archive_sha256",
                  "archive_member_manifest_sha256", "extracted_tree_manifest_sha256",
                  "extracted_tree_manifest_source", "extracted_tree_mode_policy",
                  "extracted_tree_entry_set_exact",
                  "archive_member_count", "archive_regular_file_count",
                  "archive_directory_count", "archive_manifest_canonical_json_bytes",
                  "source_archive_read_from_stage0_retained_fd", "authority",
                  "receipt_digest"},
)
runtime_origin = module.load_sealed(
    runtime_origin_path,
    schema_version="saic-formal-v2-runtime-import-origin-closure-v1",
    exact_fields={"schema_version", "status", "source_archive_sha256",
                  "archive_member_manifest_sha256", "runtime_sha256",
                  "runtime_relative_path", "project_module_count",
                  "project_module_rows", "project_module_manifest_sha256",
                  "all_project_module_origins_from_extracted_archive",
                  "isolated_python_no_environment_path", "runtime_help_exit_status",
                  "authority", "receipt_digest"},
)
if (not isinstance(archive_binding["extracted_tree_manifest_sha256"], str)
        or len(archive_binding["extracted_tree_manifest_sha256"]) != 64
        or archive_binding["extracted_tree_manifest_source"]
        != "actual_lstat_after_extraction"
        or archive_binding["extracted_tree_mode_policy"]
        != "directories_0700_files_0400"
        or archive_binding["extracted_tree_entry_set_exact"] is not True
        or runtime_origin["archive_member_manifest_sha256"]
        != archive_binding["archive_member_manifest_sha256"]
        or runtime_origin["source_archive_sha256"]
        != archive_binding["source_archive_sha256"]):
    raise SystemExit("fd canary runtime/source closure binding differs")
if any(decision.get(key) != value for key, value in {
    "slurm_job_id": job, "group_id": group, "candidate_index": 0,
    "candidate_id": candidate, "launch_ordinal": int(ordinal), "rdzv_id": rdzv_id,
}.items()):
    raise SystemExit("fd canary admission identity differs")
core = {
    "schema_version": "saic-formal-v2-retained-fd-world4-completion-v1",
    "status": "retained_guard_fd_world4_runtime_help_completed",
    "slurm_job_id": job, "group_id": group, "candidate_index": 0,
    "candidate_id": candidate, "launch_ordinal": int(ordinal), "rdzv_id": rdzv_id,
    "actual_master_port": decision["actual_master_port"],
    "claim_path": str(claim_path), "claim_digest": claim["receipt_digest"],
    "admission_digest": decision["receipt_digest"],
    "rank_packet_digests": decision["rank_packet_digests"],
    "source_archive_sha256": archive_binding["source_archive_sha256"],
    "archive_member_manifest_sha256": archive_binding[
        "archive_member_manifest_sha256"],
    "extracted_tree_manifest_sha256": archive_binding[
        "extracted_tree_manifest_sha256"],
    "extracted_tree_manifest_source": archive_binding[
        "extracted_tree_manifest_source"],
    "extracted_tree_mode_policy": archive_binding["extracted_tree_mode_policy"],
    "extracted_tree_entry_set_exact": archive_binding[
        "extracted_tree_entry_set_exact"],
    "archive_member_count": archive_binding["archive_member_count"],
    "archive_regular_file_count": archive_binding["archive_regular_file_count"],
    "archive_directory_count": archive_binding["archive_directory_count"],
    "archive_binding_receipt_digest": archive_binding["receipt_digest"],
    "runtime_origin_manifest_sha256": runtime_origin[
        "project_module_manifest_sha256"],
    "runtime_origin_project_module_count": runtime_origin["project_module_count"],
    "runtime_origin_receipt_digest": runtime_origin["receipt_digest"],
    "runtime_import_origins_all_from_extracted_archive": True,
    "guard_observed_via_parent_proc_fd": True,
    "scientific_generation_entered": False,
    "authority": {"scientific": False, "generation": False, "training": False,
                  "publication": False, "formal_job_authorized": False},
}
module.write_create_only(life / "fd-completion.json", module.seal(core))
PY
    completed=true
    break
  done
  [[ "${completed}" == true ]] || fail "${group_id} did not complete"
)

run_group sp4-a 0,1,2,3 >"${output_root}/logs/sp4-a.group.log" 2>&1 & group_a_pid=$!
run_group sp4-b 4,5,6,7 >"${output_root}/logs/sp4-b.group.log" 2>&1 & group_b_pid=$!
group_a_status=0; group_b_status=0
wait "${group_a_pid}" || group_a_status=$?
wait "${group_b_pid}" || group_b_status=$?
group_a_pid=""; group_b_pid=""
[[ "${group_a_status}" == 0 && "${group_b_status}" == 0 ]] || \
  fail "WORLD4 groups failed a=${group_a_status} b=${group_b_status}"
chmod 0444 -- "${output_root}/logs/sp4-a.group.log" "${output_root}/logs/sp4-b.group.log"

terminal_status=0
"${python_bin}" -B - "${guard_fd_path}" "${output_root}" "${scratch}" \
  "${SLURM_JOB_ID}" "${guard_sha}" "${payload_sha}" "${runtime_sha}" \
  "${guard_fd_path}" "${payload_fd_path}" "${guard_retained}" \
  "${guard_decoy}" "${payload_retained}" "${payload_decoy}" \
  "${submission_receipt}" "${wrapper_sha}" \
  "${output_parent_device}" "${output_parent_inode}" \
  "${submission_receipt_device}" "${submission_receipt_inode}" \
  "${compute_output_parent_device}" "${compute_output_parent_inode}" \
  "${compute_submission_receipt_device}" "${compute_submission_receipt_inode}" \
  "${submission_receipt_fd_number}" \
  "${postflight}" "${postflight_sha}" "${release_manifest}" \
  "${release_manifest_sha}" "${release_manifest_digest}" \
  "${payload_release}" "${guard_release}" "${wrapper_path}" \
  "${python_bin}" "${python_sha}" "${runtime}" "${source_archive}" \
  "${source_archive_sha}" "${probe_validator}" \
  "${probe_validator_sha}" "${probe_admission}" "${probe_admission_sha}" \
  "${probe_admission_digest}" "${archive_binding}" "${runtime_origin}" \
  <<'PY' || terminal_status=$?
import hashlib, os, shutil, stat, sys, types
from pathlib import Path
guard_path, root, scratch = map(Path, sys.argv[1:4])
job, guard_sha, payload_sha, runtime_sha, guard_fd, payload_fd = sys.argv[4:10]
guard_retained, guard_decoy, payload_retained, payload_decoy = map(
    Path, sys.argv[10:14]
)
submission_receipt = Path(sys.argv[14])
wrapper_sha = sys.argv[15]
output_parent_identity = (int(sys.argv[16]), int(sys.argv[17]))
submission_receipt_identity = (int(sys.argv[18]), int(sys.argv[19]))
compute_output_parent_identity = (int(sys.argv[20]), int(sys.argv[21]))
compute_submission_receipt_identity = (int(sys.argv[22]), int(sys.argv[23]))
submission_receipt_fd_number = int(sys.argv[24])
postflight = sys.argv[25]
postflight_sha = sys.argv[26]
release_manifest = sys.argv[27]
release_manifest_sha = sys.argv[28]
release_manifest_digest = sys.argv[29]
payload_release = sys.argv[30]
guard_release = sys.argv[31]
wrapper_path = sys.argv[32]
python_bin = sys.argv[33]
python_sha = sys.argv[34]
runtime_path = sys.argv[35]
source_archive = sys.argv[36]
source_archive_sha = sys.argv[37]
probe_validator = sys.argv[38]
probe_validator_sha = sys.argv[39]
probe_admission = sys.argv[40]
probe_admission_sha = sys.argv[41]
probe_admission_digest = sys.argv[42]
archive_binding_path = Path(sys.argv[43])
runtime_origin_path = Path(sys.argv[44])
raw = guard_path.read_bytes()
if hashlib.sha256(raw).hexdigest() != guard_sha:
    raise SystemExit("terminal retained guard differs")
module = types.ModuleType("fd_canary_terminal_guard")
exec(compile(raw, "retained-guard-v2", "exec"), module.__dict__)
root = module.exact_directory(root, label="fd canary output root")
authority = {"scientific": False, "generation": False, "training": False,
             "publication": False, "formal_job_authorized": False}
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
archive_binding = module.load_sealed(
    archive_binding_path,
    schema_version="saic-formal-v2-source-archive-extraction-binding-v1",
    exact_fields={"schema_version", "status", "source_archive_sha256",
                  "archive_member_manifest_sha256", "extracted_tree_manifest_sha256",
                  "extracted_tree_manifest_source", "extracted_tree_mode_policy",
                  "extracted_tree_entry_set_exact",
                  "archive_member_count", "archive_regular_file_count",
                  "archive_directory_count", "archive_manifest_canonical_json_bytes",
                  "source_archive_read_from_stage0_retained_fd", "authority",
                  "receipt_digest"},
)
runtime_origin = module.load_sealed(
    runtime_origin_path,
    schema_version="saic-formal-v2-runtime-import-origin-closure-v1",
    exact_fields={"schema_version", "status", "source_archive_sha256",
                  "archive_member_manifest_sha256", "runtime_sha256",
                  "runtime_relative_path", "project_module_count",
                  "project_module_rows", "project_module_manifest_sha256",
                  "all_project_module_origins_from_extracted_archive",
                  "isolated_python_no_environment_path", "runtime_help_exit_status",
                  "authority", "receipt_digest"},
)
if (archive_binding.get("status") != "exact_formal_source_archive_extracted_and_verified"
        or archive_binding.get("source_archive_sha256") != source_archive_sha
        or archive_binding.get("archive_member_manifest_sha256")
        != "1f3c8af23f5b4d416cea04476900c5d479ad3000338746e11f0e655b995b0fcc"
        or not isinstance(archive_binding.get("extracted_tree_manifest_sha256"), str)
        or len(archive_binding["extracted_tree_manifest_sha256"]) != 64
        or archive_binding.get("extracted_tree_manifest_source")
        != "actual_lstat_after_extraction"
        or archive_binding.get("extracted_tree_mode_policy")
        != "directories_0700_files_0400"
        or archive_binding.get("extracted_tree_entry_set_exact") is not True
        or archive_binding.get("archive_member_count") != 864
        or archive_binding.get("archive_regular_file_count") != 853
        or archive_binding.get("archive_directory_count") != 11
        or archive_binding.get("archive_manifest_canonical_json_bytes") != 173053
        or archive_binding.get("source_archive_read_from_stage0_retained_fd") is not True
        or runtime_origin.get("status") != "isolated_runtime_help_import_closure_verified"
        or runtime_origin.get("source_archive_sha256") != source_archive_sha
        or runtime_origin.get("archive_member_manifest_sha256")
        != archive_binding.get("archive_member_manifest_sha256")
        or runtime_origin.get("runtime_sha256") != runtime_sha
        or runtime_origin.get("project_module_count") != 14
        or runtime_origin.get("project_module_manifest_sha256")
        != "2e9360581b21b56e6998e1e5db8df98e4cc66acf95fbb7819baffd1161eb98ba"
        or runtime_origin.get("all_project_module_origins_from_extracted_archive") is not True
        or runtime_origin.get("isolated_python_no_environment_path") is not True
        or runtime_origin.get("runtime_help_exit_status") != 0):
    raise SystemExit("terminal source archive/runtime closure differs")
if sorted(item.name for item in root.iterdir()) != [
    "forbidden-attempts", "logs", "rendezvous"
]:
    raise SystemExit("fd canary output root closure differs")
forbidden = module.exact_directory(
    root / "forbidden-attempts", label="fd canary forbidden output root"
)
if any(forbidden.iterdir()):
    raise SystemExit("fd canary entered scientific generation")
fixture = guard_retained.parent
fixture = module.exact_directory(fixture, label="fd replacement fixture")
if (fixture != payload_retained.parent or fixture != guard_decoy.parent
        or fixture != payload_decoy.parent
        or sorted(item.name for item in fixture.iterdir()) != [
            "guard.logical", "guard.original", "payload.logical", "payload.original"
        ]):
    raise SystemExit("fd canary replacement fixture closure differs")
for path in (guard_retained, guard_decoy, payload_retained, payload_decoy):
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o444):
        raise SystemExit("fd canary replacement fixture file differs")
if (hashlib.sha256(guard_retained.read_bytes()).hexdigest() != guard_sha
        or hashlib.sha256(payload_retained.read_bytes()).hexdigest() != payload_sha
        or hashlib.sha256(guard_decoy.read_bytes()).hexdigest() == guard_sha
        or hashlib.sha256(payload_decoy.read_bytes()).hexdigest() == payload_sha
        or (guard_retained.stat().st_dev, guard_retained.stat().st_ino)
        != (guard_path.stat().st_dev, guard_path.stat().st_ino)
        or (payload_retained.stat().st_dev, payload_retained.stat().st_ino)
        != (Path(payload_fd).stat().st_dev, Path(payload_fd).stat().st_ino)):
    raise SystemExit("fd canary retained/decoy linkage differs")
rows = []
ports = set()
referenced_claims = set()
collision_rows = []
all_rdzv_ids = set()
expected_root_entries = ["port-claims", "sp4-a", "sp4-b"]
rendezvous_root = module.exact_directory(
    root / "rendezvous", label="fd canary rendezvous root"
)
if sorted(item.name for item in rendezvous_root.iterdir()) != expected_root_entries:
    raise SystemExit("fd canary rendezvous root closure differs")
for group in ("sp4-a", "sp4-b"):
    group_root = module.exact_directory(
        rendezvous_root / group, label="fd canary rendezvous group"
    )
    candidate_id = f"fd-canary-{group}-candidate-00"
    candidate_digest = hashlib.sha256(candidate_id.encode("ascii")).hexdigest()
    candidate_root = group_root / f"candidate-00-{candidate_digest[:16]}"
    if sorted(item.name for item in group_root.iterdir()) != [candidate_root.name]:
        raise SystemExit("fd canary candidate root closure differs")
    launches = sorted(candidate_root.iterdir(), key=lambda path: path.name)
    if (not launches or len(launches) > 16
            or [path.name for path in launches]
            != [f"launch-{ordinal:02d}" for ordinal in range(1, len(launches) + 1)]):
        raise SystemExit("fd canary lifecycle count differs")
    candidate_root = module.exact_directory(
        candidate_root, label="fd canary candidate root"
    )
    for prior_ordinal, prior_life in enumerate(launches[:-1], start=1):
        prior_life = module.exact_directory(
            prior_life, label="fd canary collision lifecycle"
        )
        if sorted(item.name for item in prior_life.iterdir()) != [
            "collision.json", "torchrun.log"
        ]:
            raise SystemExit("fd canary collision evidence closure differs")
        prior_log = (prior_life / "torchrun.log").lstat()
        if (not stat.S_ISREG(prior_log.st_mode) or stat.S_ISLNK(prior_log.st_mode)
                or prior_log.st_nlink != 1 or stat.S_IMODE(prior_log.st_mode) != 0o444):
            raise SystemExit("fd canary collision log differs")
        collision = module._validate_collision_for_job_audit(
            prior_life / "collision.json", claim_root=root / "rendezvous/port-claims",
            slurm_job_id=job,
        )
        expected_collision_rdzv = (
            f"saic-{job}-{group}-c00-{candidate_digest[:16]}-l{prior_ordinal:02d}"
        )
        if (collision.get("group_id") != group
                or collision.get("candidate_index") != 0
                or collision.get("candidate_id") != candidate_id
                or collision.get("launch_ordinal") != prior_ordinal
                or collision.get("rdzv_id") != expected_collision_rdzv):
            raise SystemExit("fd canary collision identity differs")
        if expected_collision_rdzv in all_rdzv_ids:
            raise SystemExit("fd canary launch rdzv id reused")
        all_rdzv_ids.add(expected_collision_rdzv)
        collision_rows.append({
            "group_id": group, "launch_ordinal": prior_ordinal,
            "rdzv_id": expected_collision_rdzv,
            "collision_digest": collision["receipt_digest"],
        })
    life = module.exact_directory(
        launches[-1], label="fd canary successful launch lifecycle"
    )
    if sorted(item.name for item in life.iterdir()) != [
        "admission.json", "fd-completion.json", "rank-0.json", "rank-1.json",
        "rank-2.json", "rank-3.json", "torchrun.log",
    ]:
        raise SystemExit("fd canary lifecycle evidence closure differs")
    success_log = (life / "torchrun.log").lstat()
    if (not stat.S_ISREG(success_log.st_mode) or stat.S_ISLNK(success_log.st_mode)
            or success_log.st_nlink != 1 or stat.S_IMODE(success_log.st_mode) != 0o444):
        raise SystemExit("fd canary successful torchrun log differs")
    completion = module.wait_load_sealed(
        life / "fd-completion.json",
        schema_version="saic-formal-v2-retained-fd-world4-completion-v1",
        exact_fields=completion_fields,
        label="fd canary group completion",
    )
    decision = module.wait_load_sealed(
        life / "admission.json", schema_version=module.DECISION_SCHEMA_VERSION,
        exact_fields=module.DECISION_FIELDS, label="terminal fd canary admission",
    )
    successful_ordinal = len(launches)
    expected_rdzv = (
        f"saic-{job}-{group}-c00-{candidate_digest[:16]}-l{successful_ordinal:02d}"
    )
    claim_path = root / "rendezvous" / "port-claims" / f"port-{decision['actual_master_port']}.json"
    claim = module.wait_load_sealed(
        claim_path, schema_version=module.CLAIM_SCHEMA_VERSION,
        exact_fields=module.CLAIM_FIELDS, label="terminal fd canary claim",
    )
    module._validate_generic_admission(claim, require_rank_packets=True)
    if (completion.get("group_id") != group or completion.get("slurm_job_id") != job
            or completion.get("status") != "retained_guard_fd_world4_runtime_help_completed"
            or completion.get("candidate_index") != 0
            or completion.get("candidate_id") != candidate_id
            or completion.get("launch_ordinal") != successful_ordinal
            or completion.get("rdzv_id") != expected_rdzv
            or completion.get("actual_master_port") != decision.get("actual_master_port")
            or completion.get("claim_path") != str(claim_path)
            or completion.get("claim_digest") != claim.get("receipt_digest")
            or completion.get("admission_digest") != decision.get("receipt_digest")
            or completion.get("rank_packet_digests") != decision.get("rank_packet_digests")
            or completion.get("source_archive_sha256") != source_archive_sha
            or completion.get("archive_member_manifest_sha256")
            != archive_binding["archive_member_manifest_sha256"]
            or completion.get("extracted_tree_manifest_sha256")
            != archive_binding["extracted_tree_manifest_sha256"]
            or completion.get("extracted_tree_manifest_source")
            != archive_binding["extracted_tree_manifest_source"]
            or completion.get("extracted_tree_mode_policy")
            != archive_binding["extracted_tree_mode_policy"]
            or completion.get("extracted_tree_entry_set_exact") is not True
            or completion.get("archive_member_count") != 864
            or completion.get("archive_regular_file_count") != 853
            or completion.get("archive_directory_count") != 11
            or completion.get("archive_binding_receipt_digest")
            != archive_binding["receipt_digest"]
            or completion.get("runtime_origin_manifest_sha256")
            != runtime_origin["project_module_manifest_sha256"]
            or completion.get("runtime_origin_project_module_count") != 14
            or completion.get("runtime_origin_receipt_digest")
            != runtime_origin["receipt_digest"]
            or completion.get("runtime_import_origins_all_from_extracted_archive")
            is not True
            or completion.get("guard_observed_via_parent_proc_fd") is not True
            or completion.get("scientific_generation_entered") is not False
            or completion.get("authority") != authority
            or any(decision.get(key) != value for key, value in {
                "slurm_job_id": job, "group_id": group, "candidate_index": 0,
                "candidate_id": candidate_id, "launch_ordinal": successful_ordinal,
                "rdzv_id": expected_rdzv,
            }.items())):
        raise SystemExit("fd canary group linkage differs")
    if expected_rdzv in all_rdzv_ids:
        raise SystemExit("fd canary successful rdzv id reused")
    all_rdzv_ids.add(expected_rdzv)
    ports.add(completion["actual_master_port"])
    referenced_claims.add(claim_path)
    rows.append({
        "group_id": group, "candidate_id": candidate_id,
        "successful_launch_ordinal": successful_ordinal,
        "rdzv_id": expected_rdzv,
        "actual_master_port": completion["actual_master_port"],
        "claim_path": str(claim_path),
        "claim_digest": completion["claim_digest"],
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
claim_root = module.exact_directory(
    rendezvous_root / "port-claims", label="fd canary claim root"
)
claim_paths = set(claim_root.iterdir())
if (len(rows) != 2 or len(ports) != 2 or len(claim_paths) != 2
        or claim_paths != referenced_claims):
    raise SystemExit("fd canary WORLD8 uniqueness differs")
expected_logs = {
    "sp4-a.group.log", "sp4-b.group.log",
}
logs = module.exact_directory(root / "logs", label="fd canary logs")
if {item.name for item in logs.iterdir()} != expected_logs:
    raise SystemExit("fd canary log set differs")
for log in logs.iterdir():
    info = log.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o444:
        raise SystemExit("fd canary log closure differs")
source_archive_fd_number = int(os.environ[
    "SAIC_FV2_FD_CANARY_STAGE0_SOURCE_ARCHIVE_FD_NUMBER"
])
if source_archive_fd_number < 3:
    raise SystemExit("terminal retained source archive fd number differs")
archive_descriptor = os.dup(source_archive_fd_number)
try:
    archive_before = os.fstat(archive_descriptor)
    archive_leaf = Path(source_archive).lstat()
    if (not stat.S_ISREG(archive_before.st_mode) or archive_before.st_nlink != 1
            or stat.S_IMODE(archive_before.st_mode) != 0o444
            or not stat.S_ISREG(archive_leaf.st_mode) or stat.S_ISLNK(archive_leaf.st_mode)
            or archive_leaf.st_nlink != 1
            or (archive_before.st_dev, archive_before.st_ino)
            != (archive_leaf.st_dev, archive_leaf.st_ino)):
        raise SystemExit("terminal retained source archive identity differs")
    os.lseek(archive_descriptor, 0, os.SEEK_SET); archive_digest = hashlib.sha256()
    while True:
        chunk = os.read(archive_descriptor, 1024 * 1024)
        if not chunk: break
        archive_digest.update(chunk)
    archive_after = os.fstat(archive_descriptor)
    if ((archive_after.st_dev, archive_after.st_ino, archive_after.st_size,
         archive_after.st_mtime_ns)
            != (archive_before.st_dev, archive_before.st_ino,
                archive_before.st_size, archive_before.st_mtime_ns)
            or archive_digest.hexdigest() != source_archive_sha):
        raise SystemExit("terminal retained source archive bytes differ")
finally:
    os.lseek(archive_descriptor, 0, os.SEEK_SET)
    os.close(archive_descriptor)
shutil.rmtree(scratch)
if scratch.exists() or scratch.is_symlink():
    raise SystemExit("fd canary scratch cleanup differs")
if submission_receipt_fd_number < 3:
    raise SystemExit("terminal retained submission fd number differs")
submission_descriptor = os.dup(submission_receipt_fd_number)
try:
    before = os.fstat(submission_descriptor)
    leaf = submission_receipt.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
            or not stat.S_ISREG(leaf.st_mode) or stat.S_ISLNK(leaf.st_mode)
            or leaf.st_nlink != 1
            or (before.st_dev, before.st_ino) != compute_submission_receipt_identity
            or (leaf.st_dev, leaf.st_ino) != compute_submission_receipt_identity
            or before.st_ino != submission_receipt_identity[1]):
        raise SystemExit("terminal retained submission identity differs")
    os.lseek(submission_descriptor, 0, os.SEEK_SET); chunks = []
    while True:
        chunk = os.read(submission_descriptor, 65536)
        if not chunk: break
        chunks.append(chunk)
    submission_raw = b"".join(chunks)
    after = os.fstat(submission_descriptor); leaf_after = submission_receipt.lstat()
    if ((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            or (leaf_after.st_dev, leaf_after.st_ino) != compute_submission_receipt_identity
            or after.st_size != len(submission_raw)):
        raise SystemExit("terminal retained submission changed")
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
if (submission.get("status") != "submitted"
        or submission.get("submission_success") is not True
        or submission.get("job_success") is not None
        or submission.get("submitted_job", {}).get("job_id") != job
        or submission.get("inputs", {}).get("wrapper_sha256") != wrapper_sha
        or submission.get("inputs", {}).get("payload_sha256") != payload_sha
        or submission.get("inputs", {}).get("guard_sha256") != guard_sha
        or submission.get("inputs", {}).get("runtime_sha256") != runtime_sha
        or submission.get("inputs") != {
            "wrapper": wrapper_path,
            "wrapper_sha256": wrapper_sha,
            "payload": payload_release,
            "payload_sha256": payload_sha,
            "guard": guard_release,
            "guard_sha256": guard_sha,
            "runtime": runtime_path,
            "runtime_sha256": runtime_sha,
            "source_archive": source_archive,
            "source_archive_sha256": source_archive_sha,
            "python": python_bin,
            "python_sha256": python_sha,
            "postflight": postflight,
            "postflight_sha256": postflight_sha,
            "release_manifest": release_manifest,
            "release_manifest_file_sha256": release_manifest_sha,
            "release_manifest_digest": release_manifest_digest,
            "probe_validator": probe_validator,
            "probe_validator_sha256": probe_validator_sha,
            "probe_admission": probe_admission,
            "probe_admission_sha256": probe_admission_sha,
            "probe_admission_digest": probe_admission_digest,
            "probe_admission_binding": submission.get("inputs", {}).get(
                "probe_admission_binding"
            ),
        }
        or submission.get("outputs", {}).get("submission_receipt")
        != str(submission_receipt)
        or submission.get("outputs", {}).get("job_output_root") != str(root)
        or submission.get("submission_boundary", {}).get("reservation_device")
        != submission_receipt_identity[0]
        or submission.get("submission_boundary", {}).get("reservation_inode")
        != submission_receipt_identity[1]
        or submission.get("submission_boundary", {}).get("output_parent_device")
        != output_parent_identity[0]
        or submission.get("submission_boundary", {}).get("output_parent_inode")
        != output_parent_identity[1]
        or submission.get("authority") != authority):
    raise SystemExit("terminal own submission linkage differs")
parent_info = root.parent.lstat()
if ((parent_info.st_dev, parent_info.st_ino) != compute_output_parent_identity
        or parent_info.st_ino != output_parent_identity[1]):
    raise SystemExit("terminal output parent identity differs")
core = {
    "schema_version": "saic-formal-v2-retained-fd-world8-operational-evidence-v1",
    "status": "retained_fd_operational_evidence_complete",
    "slurm_job_id": job,
    "job_success": None,
    "slurm_terminal_verified": False,
    "formal_admission": False,
    "topology": "two_concurrent_world4_on_one_requested_8mi210_node",
    "requested_gpu_count": 8, "world_size_total": 8,
    "group_count": 2, "rank_packet_count": 8,
    "unique_actual_master_port_count": 2,
    "collision_receipt_count": len(collision_rows),
    "all_launch_rdzv_id_count": len(all_rdzv_ids),
    "all_launch_rdzv_ids_unique": True,
    "guard_sha256": guard_sha, "payload_sha256": payload_sha,
    "runtime_sha256": runtime_sha,
    "source_archive_path": source_archive,
    "source_archive_sha256": source_archive_sha,
    "source_archive_retained_fd_number": source_archive_fd_number,
    "source_archive_read_from_stage0_retained_fd": True,
    "archive_member_manifest_sha256": archive_binding[
        "archive_member_manifest_sha256"],
    "extracted_tree_manifest_sha256": archive_binding[
        "extracted_tree_manifest_sha256"],
    "extracted_tree_manifest_source": archive_binding[
        "extracted_tree_manifest_source"],
    "extracted_tree_mode_policy": archive_binding["extracted_tree_mode_policy"],
    "extracted_tree_entry_set_exact": archive_binding[
        "extracted_tree_entry_set_exact"],
    "archive_member_count": archive_binding["archive_member_count"],
    "archive_regular_file_count": archive_binding["archive_regular_file_count"],
    "archive_directory_count": archive_binding["archive_directory_count"],
    "archive_manifest_canonical_json_bytes": archive_binding[
        "archive_manifest_canonical_json_bytes"],
    "archive_binding_receipt_digest": archive_binding["receipt_digest"],
    "runtime_origin_schema_version": runtime_origin["schema_version"],
    "runtime_origin_project_module_count": runtime_origin["project_module_count"],
    "runtime_origin_project_module_rows": runtime_origin["project_module_rows"],
    "runtime_origin_manifest_sha256": runtime_origin[
        "project_module_manifest_sha256"],
    "runtime_origin_receipt_digest": runtime_origin["receipt_digest"],
    "runtime_import_origins_all_from_extracted_archive": True,
    "runtime_import_checker_isolated_python": True,
    "runtime_imported_from_pinned_source_archive": True,
    "runtime_import_scratch_removed_before_evidence": True,
    "guard_fd_path": guard_fd, "payload_fd_path": payload_fd,
    "replacement_fixture": str(fixture),
    "guard_retained_leaf": str(guard_retained),
    "guard_decoy_leaf": str(guard_decoy),
    "payload_retained_leaf": str(payload_retained),
    "payload_decoy_leaf": str(payload_decoy),
    "spooled_wrapper_retained_fd_submission_required": True,
    "payload_exec_from_parent_retained_fd": True,
    "guard_read_by_two_background_groups_and_all_workers_from_parent_retained_fd": True,
    "logical_leaf_replacement_after_fd_open_observed": True,
    "scratch_cleaned_before_operational_evidence_publication": True,
    "external_terminal_postflight_admission_required": True,
    "submission_receipt_path": str(submission_receipt),
    "submission_receipt_file_sha256": hashlib.sha256(submission_raw).hexdigest(),
    "submission_receipt_digest": submission["receipt_digest"],
    "output_parent_identity": f"{output_parent_identity[0]}:{output_parent_identity[1]}",
    "submission_receipt_identity": (
        f"{submission_receipt_identity[0]}:{submission_receipt_identity[1]}"
    ),
    "compute_output_parent_identity": (
        f"{compute_output_parent_identity[0]}:{compute_output_parent_identity[1]}"
    ),
    "compute_submission_receipt_identity": (
        f"{compute_submission_receipt_identity[0]}:{compute_submission_receipt_identity[1]}"
    ),
    "submission_receipt_retained_fd_number": submission_receipt_fd_number,
    "submission_receipt_read_from_retained_fd": True,
    "stage0_o_nofollow_source_open": True,
    "stage0_python_execve_bash_handoff": True,
    "stage0_inherited_source_fd_numbers": {
        "guard": int(os.environ["SAIC_FV2_FD_CANARY_STAGE0_GUARD_FD_NUMBER"]),
        "spooled_wrapper": int(os.environ[
            "SAIC_FV2_FD_CANARY_STAGE0_SPOOL_FD_NUMBER"
        ]),
        "payload": int(os.environ["SAIC_FV2_FD_CANARY_STAGE0_PAYLOAD_FD_NUMBER"]),
        "probe_validator": int(os.environ[
            "SAIC_FV2_FD_CANARY_STAGE0_PROBE_VALIDATOR_FD_NUMBER"
        ]),
        "source_archive": source_archive_fd_number,
        "submission_receipt": int(os.environ[
            "SAIC_FV2_FD_CANARY_STAGE0_RECEIPT_FD_NUMBER"
        ]),
    },
    "stage0_source_fds_distinct": True,
    "stage0_output_directory_fd_number": int(os.environ[
        "SAIC_FV2_FD_CANARY_STAGE0_OUTPUT_FD_NUMBER"
    ]),
    "stage0_output_directory_identity": (
        f"{compute_output_parent_identity[0]}:{compute_output_parent_identity[1]}"
    ),
    "stage0_output_directory_o_nofollow": True,
    "wrapper_sha256": wrapper_sha,
    "probe_admission_binding": submission["inputs"]["probe_admission_binding"],
    "scientific_generation_entered": False,
    "scientific_output_created": False,
    "formal_full60_result_claimed": False,
    "candidate_rows": rows, "collision_rows": collision_rows,
    "authority": authority,
}
module.write_create_only(root / "operational-evidence.json", module.seal(core)); os._exit(0)
PY
exit "${terminal_status}"
