#!/bin/bash -p

# Startup-TCB gate: this block intentionally precedes every function and every
# external command.  The sole formal invocation is:
#   env -i PATH=/usr/bin:/bin LC_ALL=C HOME=/nonexistent \
#     /bin/bash -p CONTROLLER EXPECTED_CONTROLLER_SHA256
if [[ $- != hpB || ${PATH-} != /usr/bin:/bin || ${LC_ALL-} != C \
      || ${HOME-} != /nonexistent || ${SHLVL-} != 1 \
      || -n ${BASH_ENV+x} || -n ${ENV+x} || -n ${LD_PRELOAD+x} \
      || -n ${LD_LIBRARY_PATH+x} ]]; then
  builtin printf 'ERROR: controller startup TCB/environment differs\n' >&2
  builtin exit 2
fi
# Linux exposes the original execve argv without re-parsing.  On the normal
# path this rejects every extra Bash switch (including every -O shopt) before
# any function definition or external command.  The local-only draft audit is
# deliberately non-authoritative and may run on hosts without procfs.
startup_process_argv=()
if [[ ${1-} != --draft-audit ]]; then
  startup_cmdline=/proc/${BASHPID}/cmdline
  if [[ ! -r ${startup_cmdline} ]]; then
    builtin printf 'ERROR: controller startup process argv unavailable\n' >&2
    builtin exit 2
  fi
  while IFS= builtin read -r -d '' startup_arg; do
    startup_process_argv+=("${startup_arg}")
  done <"${startup_cmdline}"
  if [[ ${#startup_process_argv[@]} -ne 4 \
        || ${startup_process_argv[0]} != /bin/bash \
        || ${startup_process_argv[1]} != -p \
        || ${startup_process_argv[2]} != "$0" \
        || ${startup_process_argv[3]} != "${1-}" ]]; then
    builtin printf 'ERROR: controller startup process argv differs\n' >&2
    builtin exit 2
  fi
fi
if shopt -q extdebug || shopt -q expand_aliases || shopt -q extglob \
    || shopt -q failglob || shopt -q globstar 2>/dev/null \
    || shopt -q inherit_errexit 2>/dev/null \
    || shopt -q lastpipe 2>/dev/null \
    || shopt -q nocaseglob || shopt -q nocasematch \
    || shopt -q nullglob || shopt -q shift_verbose || shopt -q xpg_echo \
    || ! shopt -q sourcepath || ! shopt -q extquote; then
  builtin printf 'ERROR: controller startup shopt state differs\n' >&2
  builtin exit 2
fi
startup_export_count=0
startup_export_bad=false
while IFS= builtin read -r startup_name; do
  case ${startup_name} in
    HOME|LC_ALL|PATH|PWD|SHLVL) ;;
    BASH_FUNC_*|*) startup_export_bad=true ;;
  esac
  startup_export_count=$((startup_export_count + 1))
done < <(builtin compgen -e)
if [[ ${startup_export_bad} != false || ${startup_export_count} -ne 5 \
      || ${PWD-} != /* ]]; then
  builtin printf 'ERROR: controller exported environment is not exact clean5\n' >&2
  builtin exit 2
fi
unset startup_name startup_arg startup_cmdline
readonly startup_export_count startup_export_bad
readonly startup_pwd=${PWD} startup_shell_version=${BASH_VERSION}
readonly startup_bashopts=${BASHOPTS}
readonly -a startup_process_argv

set -Eeuo pipefail
shopt -s nullglob
umask 077
readonly PATH LC_ALL HOME

# Intentional NO-GO construction flags. A detached final audit may flip only
# these booleans. The controller SHA is supplied solely by its one caller
# argument, so this source contains no cyclic self-hash pin.
release_sealed=false
python_pin_sealed=true
controller_contract_complete=false

expected_runtime_sha256=2e62a889410168d45dbe8c62ddb054ad04bda659af325b92b571ed84b795f5de
expected_runtime_size=102463
expected_tests_sha256=cdf4d8f48a87ac4140200921c4bf489660edfb534e156c413e314326dfadb945
expected_tests_size=71231
expected_manifest_sha256=b32aec0f414bd0ebdad317c31d6197d47e44d21820eb77239272b4d0b4e7c4ca
expected_manifest_size=1067
expected_manifest_digest=b093de01157a927d7a63e94024b4ab4e1e96c699fba29a9bf9db2cfd03280932
expected_release_tree_sha256=13d2231eb7a5ecd6ba6346e00030d23c02bd39a097a5b4d2a0006ab8b4fd1b45
expected_test_count=25

base=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/semantic_anchor_vjepa2_role_directed_teacher_margin_v4g_20260821
release_root=${base}/recovery_release_v1
runtime_rel=methods/bernini_action_editing/recover_v4g_scientific_no_go_attestation_v1.py
tests_rel=methods/bernini_action_editing/tests/test_recover_v4g_scientific_no_go_attestation_v1.py
manifest_rel=release-manifest-v4g-recovery.json
controller_path=${base}/controllers/auh_v4g_scientific_no_go_sibling_recovery_v1.sh
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
srun_bin=/usr/bin/srun
original_root=${base}/runs/exact5_parallel_38b2cbec_v1
recovery_root=${base}/runs/exact5_parallel_38b2cbec_v1_recovery_v1
execution_root=${base}/runs/exact5_parallel_38b2cbec_v1_recovery_controller_execution_v1
execution_name=execution-receipt.json
job_id=143808
node=auh7-1b-gpu-268

expected_python_sha256=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
expected_python_size=31490256
# The formal launcher spelling remains /bin/bash, while the executable
# authority binding uses AUH's audited canonical merged-/usr path.
shell_bin=/usr/bin/bash
expected_shell_sha256=59474588a312b6b6e73e5a42a59bf71e62b55416b6c9d5e4a6e1c630c2a9ecd4
expected_srun_sha256=2b8f60b30edf7efed35bb00864651da1b0bec68e75f942ce58b5ff82bc43cd9e
expected_exact26_sha256=14bf42749c97b20934a2a088a560fa23ed2b1e37555262e9d4c7f2f368e74265
expected_parent_signature=0540ac21631bc948db012c77003c99d0de32cb1f769ffe38e8ab8b8e380cac76
expected_torch_version=2.7.1+rocm6.3
expected_torch_hip_version=6.3.42131-fa1d09cbd

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

draft_audit() {
  DRAFT_RELEASE_SEALED=${release_sealed} \
  DRAFT_PYTHON_PIN_SEALED=${python_pin_sealed} \
  DRAFT_CONTRACT_COMPLETE=${controller_contract_complete} \
  DRAFT_RUNTIME_SHA=${expected_runtime_sha256} \
  DRAFT_TESTS_SHA=${expected_tests_sha256} \
  DRAFT_MANIFEST_SHA=${expected_manifest_sha256} \
  DRAFT_MANIFEST_DIGEST=${expected_manifest_digest} \
  DRAFT_TREE_SHA=${expected_release_tree_sha256} \
  DRAFT_TEST_COUNT=${expected_test_count} \
  /usr/bin/python3 -I -S -B - <<'PY'
import json, os
value = {
    "schema_version": "v4g-recovery-controller-draft-audit-v2",
    "intentional_no_go": os.environ["DRAFT_RELEASE_SEALED"] != "true",
    "release_sealed": os.environ["DRAFT_RELEASE_SEALED"] == "true",
    "python_pin_sealed": os.environ["DRAFT_PYTHON_PIN_SEALED"] == "true",
    "controller_contract_complete": os.environ["DRAFT_CONTRACT_COMPLETE"] == "true",
    "runtime_sha256": os.environ["DRAFT_RUNTIME_SHA"],
    "tests_sha256": os.environ["DRAFT_TESTS_SHA"],
    "manifest_sha256": os.environ["DRAFT_MANIFEST_SHA"],
    "manifest_digest": os.environ["DRAFT_MANIFEST_DIGEST"],
    "release_tree_sha256": os.environ["DRAFT_TREE_SHA"],
    "normal_and_optimized_test_count": int(os.environ["DRAFT_TEST_COUNT"]),
    "launch_performed": False,
}
print(json.dumps(value, sort_keys=True, separators=(",", ":")))
PY
}

if [[ ${1:-} == --draft-audit ]]; then
  [[ $# -eq 1 ]] || fail "draft-audit argument count differs"
  draft_audit
  exit 0
fi

# This is deliberately the first normal-entry gate.
[[ ${release_sealed} == true && ${python_pin_sealed} == true && \
   ${controller_contract_complete} == true ]] || \
  fail "v4G recovery controller is intentional NO-GO"

[[ $# -eq 1 ]] || fail "usage: $0 EXPECTED_CONTROLLER_SHA256"
expected_controller_sha256=$1
[[ ${expected_controller_sha256} =~ ^[0-9a-f]{64}$ ]] || \
  fail "controller SHA argument differs"
for value in "${expected_runtime_sha256}" "${expected_tests_sha256}" \
             "${expected_manifest_sha256}" "${expected_manifest_digest}" \
             "${expected_release_tree_sha256}" "${expected_python_sha256}" \
             "${expected_shell_sha256}" "${expected_srun_sha256}" \
             "${expected_exact26_sha256}" \
             "${expected_parent_signature}"; do
  [[ ${value} =~ ^[0-9a-f]{64}$ ]] || fail "frozen SHA pin differs"
done

# Bind both executable images before any Python authority is trusted.  The
# root-owned hash/stat/readlink tools authenticate the already-open FDs; every
# parent Python process below is then executed from the retained Python FD.
controller_pid=$$
exec {controller_source_image_fd}<"/proc/${controller_pid}/fd/255" || \
  fail "controller executed-source FD open failed"
exec {controller_shell_image_fd}<"/proc/${controller_pid}/exe" || \
  fail "controller shell process-image FD open failed"
exec {parent_python_image_fd}<"${python_bin}" || \
  fail "parent Python image FD open failed"
controller_source_image_path=/proc/${controller_pid}/fd/${controller_source_image_fd}
controller_shell_image_path=/proc/${controller_pid}/fd/${controller_shell_image_fd}
parent_python_image_path=/proc/${controller_pid}/fd/${parent_python_image_fd}
controller_fd_sha_output="$(/usr/bin/sha256sum -- \
  "${controller_source_image_path}")" || fail "controller executed-source SHA failed"
shell_fd_sha_output="$(/usr/bin/sha256sum -- "${controller_shell_image_path}")" || \
  fail "controller shell process-image SHA read failed"
python_fd_sha_output="$(/usr/bin/sha256sum -- "${parent_python_image_path}")" || \
  fail "parent Python image SHA read failed"
[[ ${shell_fd_sha_output%% *} == "${expected_shell_sha256}" ]] || \
  fail "controller shell process-image SHA differs"
[[ ${controller_fd_sha_output%% *} == "${expected_controller_sha256}" ]] || \
  fail "controller executed-source SHA differs"
[[ ${python_fd_sha_output%% *} == "${expected_python_sha256}" ]] || \
  fail "parent Python held-image SHA differs"
shell_fd_stat="$(/usr/bin/stat -Lc '%a:%h:%s:%d:%i:%Y:%Z:%F' -- \
  "${controller_shell_image_path}")" || fail "controller shell FD stat failed"
[[ ${shell_fd_stat} == \
   '755:1:1396520:64768:56624315:1710415907:1754402470:regular file' ]] || \
  fail "controller shell held-image physical pin differs"
python_fd_stat="$(/usr/bin/stat -Lc '%a:%h:%s:%u:%g:%F' -- \
  "${parent_python_image_path}")" || fail "parent Python FD stat failed"
[[ ${python_fd_stat} == \
   '755:1:31490256:2012:2000:regular file' ]] || \
  fail "parent Python held-image physical pin differs"
controller_full_stat="$(/usr/bin/stat -Lc \
  '%a:%h:%s:%d:%i:%Y:%Z:%u:%g:%F' -- \
  "${controller_source_image_path}")" || fail "controller source FD stat failed"
[[ ${controller_full_stat%%:*} == 555 && \
   $(/usr/bin/stat -Lc '%h:%F' -- "${controller_source_image_path}") == \
   '1:regular file' ]] || fail "controller executed-source seal differs"
shell_full_stat="$(/usr/bin/stat -Lc '%a:%h:%s:%d:%i:%Y:%Z:%u:%g:%F' -- \
  "${controller_shell_image_path}")" || fail "controller shell full stat failed"
python_full_stat="$(/usr/bin/stat -Lc '%a:%h:%s:%d:%i:%Y:%Z:%u:%g:%F' -- \
  "${parent_python_image_path}")" || fail "parent Python full stat failed"
[[ ${shell_full_stat} == "$(/usr/bin/stat -Lc \
   '%a:%h:%s:%d:%i:%Y:%Z:%u:%g:%F' -- "${shell_bin}")" ]] || \
  fail "controller shell held/path identity differs"
[[ ${python_full_stat} == "$(/usr/bin/stat -Lc \
   '%a:%h:%s:%d:%i:%Y:%Z:%u:%g:%F' -- "${python_bin}")" ]] || \
  fail "parent Python held/path identity differs"
[[ ${controller_full_stat} == "$(/usr/bin/stat -Lc \
   '%a:%h:%s:%d:%i:%Y:%Z:%u:%g:%F' -- "${controller_path}")" ]] || \
  fail "controller executed-source/lexical identity differs"
[[ $(/usr/bin/readlink -f -- "/proc/${controller_pid}/exe") == \
   $(/usr/bin/readlink -f -- "${shell_bin}") ]] || \
  fail "controller shell process-image realpath differs"
[[ ! -L ${python_bin} && $(/usr/bin/readlink -f -- "${python_bin}") == \
   "${python_bin}" ]] || fail "parent Python designated path differs"
readonly controller_pid controller_source_image_fd controller_shell_image_fd
readonly parent_python_image_fd controller_source_image_path
readonly controller_shell_image_path parent_python_image_path
readonly controller_full_stat shell_fd_stat python_fd_stat shell_full_stat
readonly python_full_stat
unset controller_fd_sha_output shell_fd_sha_output python_fd_sha_output

run_pinned_python() {
  (
    exec -a "${python_bin}" "${parent_python_image_path}" "$@"
  )
}

canonical_controller="$(readlink -f -- "$0")" || fail "controller canonicalization failed"
[[ ${canonical_controller} == "${controller_path}" && ! -L ${controller_path} ]] || \
  fail "detached controller path differs"
controller_sha="$(sha256sum -- "${controller_path}" | awk '{print $1}')" || \
  fail "controller SHA read failed"
[[ ${controller_sha} == "${expected_controller_sha256}" ]] || \
  fail "controller SHA differs"
[[ $(stat -Lc '%a:%h:%F' -- "${controller_path}") == '555:1:regular file' ]] || \
  fail "controller seal differs"

tmp_root="$(/usr/bin/mktemp -d /tmp/v4g-recovery-controller.XXXXXX)" || \
  fail "temporary root creation failed"
exec {tmp_fd}<"${tmp_root}" || fail "temporary root held-FD open failed"
tmp_identity="$(run_pinned_python -I -S -B - \
  "${tmp_root}" "${controller_pid}" "${tmp_fd}" <<'PY'
from pathlib import Path
import os, stat, sys
root, parent_pid, descriptor = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
held = os.open(
    f"/proc/{parent_pid}/fd/{descriptor}",
    os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY,
)
try:
    current, opened = root.lstat(), os.fstat(held)
    if (root.is_symlink() or root != root.resolve(strict=True)
            or not stat.S_ISDIR(current.st_mode)
            or stat.S_IMODE(current.st_mode) != 0o700
            or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)):
        raise SystemExit("temporary root creation/held-FD binding differs")
    print(f"{opened.st_dev}:{opened.st_ino}")
finally:
    os.close(held)
PY
)" || fail "temporary root identity capture failed"
cleanup_tmp() {
  run_pinned_python -I -S -B - \
    "${tmp_root}" "${controller_pid}" "${tmp_fd}" "${tmp_identity}" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys, types
root, parent_pid, descriptor, expected_identity = (
    Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
)
held = os.open(
    f"/proc/{parent_pid}/fd/{descriptor}",
    os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY,
)
try:
    opened = os.fstat(held)
    if f"{opened.st_dev}:{opened.st_ino}" != expected_identity:
        raise SystemExit("temporary cleanup held-FD identity differs")
    try:
        current = root.lstat()
    except FileNotFoundError:
        raise SystemExit("temporary cleanup lexical root is absent")
    if (root.is_symlink() or root != root.resolve(strict=True)
            or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)):
        raise SystemExit("temporary cleanup lexical/held root differs")
    ledger_name = "intermediate-ledger.json"
    try:
        ledger_fd = os.open(
            ledger_name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=held,
        )
    except FileNotFoundError:
        # Failure before ledger finalization is retained, never recursively
        # traversed or permission-mutated by cleanup.
        raise SystemExit(0)
    digest = hashlib.sha256(); chunks = []
    try:
        ledger_info = os.fstat(ledger_fd)
        while True:
            chunk = os.read(ledger_fd, 1024 * 1024)
            if not chunk: break
            digest.update(chunk); chunks.append(chunk)
    finally: os.close(ledger_fd)
    raw = b"".join(chunks)
    ledger = json.loads(raw.decode("ascii"))
    canonical = json.dumps(
        ledger, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    if (raw != canonical or stat.S_IMODE(ledger_info.st_mode) != 0o444
            or ledger_info.st_nlink != 1
            or digest.hexdigest() != hashlib.sha256(raw).hexdigest()
            or set(ledger) != {"schema_version", "root_identity", "members"}
            or ledger.get("root_identity") != expected_identity
            or type(ledger.get("members")) is not list):
        raise SystemExit("temporary cleanup ledger differs")
    members = ledger["members"]
    expected_names = [row.get("name") for row in members]
    if (any(type(row) is not dict or set(row) != {
                "name", "sha256", "size_bytes", "mode_octal", "nlink",
                "device", "inode", "mtime_ns", "ctime_ns",
                "single_fd_pre_post_identity_and_sha_exact",
            } for row in members)
            or expected_names != sorted(set(expected_names))
            or sorted(os.listdir(held)) != sorted([*expected_names, ledger_name])):
        raise SystemExit("temporary cleanup exact member ledger differs")
    for row in members:
        fd = os.open(
            row["name"], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=held,
        )
        file_digest = hashlib.sha256()
        try:
            info = os.fstat(fd)
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk: break
                file_digest.update(chunk)
        finally: os.close(fd)
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444
                or info.st_nlink != 1 or row != {
                    "name": row["name"], "sha256": file_digest.hexdigest(),
                    "size_bytes": info.st_size, "mode_octal": "0444",
                    "nlink": info.st_nlink, "device": info.st_dev,
                    "inode": info.st_ino, "mtime_ns": info.st_mtime_ns,
                    "ctime_ns": info.st_ctime_ns,
                    "single_fd_pre_post_identity_and_sha_exact": True,
                }):
            raise SystemExit("temporary cleanup member binding differs")
    os.fchmod(held, 0o700)
    for row in members:
        os.unlink(row["name"], dir_fd=held)
    os.unlink(ledger_name, dir_fd=held)
    os.fsync(held)
    if os.listdir(held):
        raise SystemExit("temporary cleanup root is not empty")
finally:
    os.close(held)
current = root.lstat()
if f"{current.st_dev}:{current.st_ino}" != expected_identity:
    raise SystemExit("temporary cleanup final lexical identity differs")
os.rmdir(root)
PY
}
trap cleanup_tmp EXIT

authority_snapshot() {
  run_pinned_python -I -S -B - \
    "${controller_path}" "${expected_controller_sha256}" \
    "${controller_source_image_path}" \
    "${shell_bin}" "${expected_shell_sha256}" \
    "${startup_shell_version}" "${startup_pwd}" "${startup_bashopts}" \
    "${controller_shell_image_path}" "${parent_python_image_path}" \
    "${python_bin}" "${expected_python_sha256}" \
    "${srun_bin}" "${expected_srun_sha256}" \
    "${release_root}" "${runtime_rel}" "${tests_rel}" "${manifest_rel}" \
    "${expected_runtime_sha256}" "${expected_runtime_size}" \
    "${expected_tests_sha256}" "${expected_tests_size}" \
    "${expected_manifest_sha256}" "${expected_manifest_size}" \
    "${expected_manifest_digest}" "${expected_release_tree_sha256}" \
    "${original_root}" "${expected_exact26_sha256}" \
    "${expected_parent_signature}" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys

(controller, controller_sha, controller_source_fd_path,
 shell_path, shell_sha, shell_version, startup_pwd, startup_bashopts,
 shell_image_fd_path, parent_python_fd_path, python_path, python_sha,
 srun_path, srun_sha,
 release_root, runtime_rel, tests_rel, manifest_rel,
 runtime_sha, runtime_size, tests_sha, tests_size, manifest_sha, manifest_size,
 manifest_digest, release_tree_sha, original_root, exact26_sha,
 parent_signature) = sys.argv[1:]
controller, shell_path, python_path, srun_path = map(
    Path, (controller, shell_path, python_path, srun_path)
)
controller_source_fd_path, shell_image_fd_path, parent_python_fd_path = map(
    Path, (controller_source_fd_path, shell_image_fd_path, parent_python_fd_path)
)
release_root, original_root = map(Path, (release_root, original_root))
runtime_size, tests_size, manifest_size = map(
    int, (runtime_size, tests_size, manifest_size)
)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")

def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, stat.S_IMODE(info.st_mode),
            info.st_nlink, info.st_mtime_ns, info.st_ctime_ns)

def regular_binding(path, mode, expected_sha=None, expected_size=None,
                    capture=False):
    if (not path.is_absolute() or path.is_symlink()
            or path != path.resolve(strict=True)):
        raise SystemExit("regular authority path differs")
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != mode
            or before.st_nlink != 1 or not hasattr(os, "O_NOFOLLOW")):
        raise SystemExit("regular authority seal differs")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    chunks = [] if capture else None
    digest = hashlib.sha256()
    try:
        opened = os.fstat(fd)
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        closed = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    if len({identity(item) for item in (before, opened, closed, after)}) != 1:
        raise SystemExit("regular authority same-FD identity differs")
    actual_sha = digest.hexdigest()
    if expected_sha is not None and actual_sha != expected_sha:
        raise SystemExit("regular authority SHA differs")
    if expected_size is not None and before.st_size != expected_size:
        raise SystemExit("regular authority size differs")
    binding = {
        "path": str(path), "sha256": actual_sha,
        "size_bytes": before.st_size, "mode_octal": f"{mode:04o}",
        "nlink": before.st_nlink, "device": before.st_dev,
        "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "single_fd_pre_post_identity_and_sha_exact": True,
    }
    return binding, (b"".join(chunks) if chunks is not None else None)

def directory_binding(path, mode):
    if (not path.is_absolute() or path.is_symlink()
            or path != path.resolve(strict=True)
            or not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW")):
        raise SystemExit("directory authority path differs")
    before = path.lstat()
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        members = sorted(os.listdir(fd))
        closed = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    if (not stat.S_ISDIR(before.st_mode) or stat.S_IMODE(before.st_mode) != mode
            or len({identity(item) for item in (before, opened, closed, after)}) != 1):
        raise SystemExit("directory authority same-FD identity differs")
    return {
        "path": str(path), "mode_octal": f"{mode:04o}",
        "nlink": before.st_nlink, "device": before.st_dev,
        "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns, "members": members,
        "single_fd_pre_post_identity_and_membership_exact": True,
    }

def held_image_binding(fd_path, logical_path, expected):
    descriptor = os.open(fd_path, os.O_RDONLY | os.O_CLOEXEC)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    actual = {
        "path": str(logical_path), "sha256": digest.hexdigest(),
        "size_bytes": before.st_size,
        "mode_octal": f"{stat.S_IMODE(before.st_mode):04o}",
        "nlink": before.st_nlink, "device": before.st_dev,
        "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "single_fd_pre_post_identity_and_sha_exact": True,
    }
    if identity(before) != identity(after) or actual != expected:
        raise SystemExit("held executable process-image binding differs")
    return actual

def strict_json(raw):
    def hook(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise SystemExit("duplicate JSON key")
            value[key] = item
        return value
    value = json.loads(raw.decode("ascii"), object_pairs_hook=hook)
    if canonical(value) + b"\n" != raw:
        raise SystemExit("noncanonical JSON authority")
    return value

controller_binding, _ = regular_binding(controller, 0o555, controller_sha)
controller_executed_source = held_image_binding(
    controller_source_fd_path, controller, controller_binding,
)
shell_binding, _ = regular_binding(shell_path, 0o755, shell_sha, 1396520)
if (shell_binding["device"] != 64768 or shell_binding["inode"] != 56624315
        or shell_binding["mtime_ns"] // 1_000_000_000 != 1710415907
        or shell_binding["ctime_ns"] // 1_000_000_000 != 1754402470
        or not shell_version.startswith("5.1.16")):
    raise SystemExit("controller shell physical/version pin differs")
python_binding, _ = regular_binding(python_path, 0o755, python_sha)
shell_process_image = held_image_binding(
    shell_image_fd_path, shell_path, shell_binding,
)
parent_python_held_image = held_image_binding(
    parent_python_fd_path, python_path, python_binding,
)
if (Path(sys.executable).resolve(strict=True) != python_path
        or Path("/proc/self/exe").resolve(strict=True) != python_path):
    raise SystemExit("authority process Python path differs")
flags = sys.flags
if (tuple(sys.version_info[:3]) != (3, 12, 13)
        or flags.isolated != 1 or flags.no_site != 1
        or flags.ignore_environment != 1 or flags.safe_path is not True
        or flags.dont_write_bytecode != 1 or flags.optimize != 0):
    raise SystemExit("authority process Python version/flags differ")
authority_python_process_image = held_image_binding(
    Path("/proc/self/exe"), python_path, python_binding,
)
srun_binding, _ = regular_binding(srun_path, 0o755, srun_sha, 164720)
if (srun_binding["device"] != 64768 or srun_binding["inode"] != 56640240
        or srun_binding["mtime_ns"] // 1_000_000_000 != 1720214072
        or srun_binding["ctime_ns"] // 1_000_000_000 != 1754628182):
    raise SystemExit("srun physical pin differs")

release_root_binding = directory_binding(release_root, 0o555)
expected_release_files = {runtime_rel, tests_rel, manifest_rel}
expected_release_dirs = {
    "methods", "methods/bernini_action_editing",
    "methods/bernini_action_editing/tests",
}
found_files, found_dirs, stack = set(), set(), [release_root]
while stack:
    parent = stack.pop()
    with os.scandir(parent) as entries:
        for entry in entries:
            path = Path(entry.path); rel = path.relative_to(release_root).as_posix()
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise SystemExit("release symlink differs")
            if stat.S_ISDIR(info.st_mode):
                if stat.S_IMODE(info.st_mode) != 0o555:
                    raise SystemExit("release directory mode differs")
                found_dirs.add(rel); stack.append(path)
            elif stat.S_ISREG(info.st_mode):
                found_files.add(rel)
            else:
                raise SystemExit("release special member differs")
if found_files != expected_release_files or found_dirs != expected_release_dirs:
    raise SystemExit("release exact3 membership differs")
release_dirs = [directory_binding(release_root / rel, 0o555)
                for rel in sorted(found_dirs)]
release_files, tree_rows, manifest_raw = {}, [], None
expected_by_rel = {
    runtime_rel: (runtime_sha, runtime_size),
    tests_rel: (tests_sha, tests_size),
    manifest_rel: (manifest_sha, manifest_size),
}
for rel in sorted(found_files):
    expected_sha, expected_size = expected_by_rel[rel]
    binding, raw = regular_binding(
        release_root / rel, 0o444, expected_sha, expected_size,
        capture=rel == manifest_rel,
    )
    release_files[rel] = binding
    tree_rows.append({"path": rel, "sha256": binding["sha256"],
                      "size_bytes": binding["size_bytes"]})
    if rel == manifest_rel:
        manifest_raw = raw
if object_sha(tree_rows) != release_tree_sha or manifest_raw is None:
    raise SystemExit("release exact3 tree digest differs")
manifest = strict_json(manifest_raw)
unsigned_manifest = dict(manifest)
stored_manifest_digest = unsigned_manifest.pop("manifest_digest", None)
expected_manifest_keys = {
    "schema_version", "status", "payload", "payload_count",
    "manifest_digest", "manifest_target_relative_path",
    "release_tree_contract", "authority_graph",
}
expected_payload = [
    {"relative_path": runtime_rel, "role": "recovery_runtime",
     "sha256": runtime_sha},
    {"relative_path": tests_rel, "role": "recovery_runtime_tests",
     "sha256": tests_sha},
]
if (set(manifest) != expected_manifest_keys
        or manifest.get("schema_version")
           != "v4g-scientific-no-go-recovery-detached-release-manifest-v1"
        or manifest.get("status") != "SEALED"
        or manifest.get("payload") != expected_payload
        or manifest.get("payload_count") != 2
        or manifest.get("manifest_target_relative_path") != manifest_rel
        or manifest.get("release_tree_contract") != {
            "exact_file_count_including_manifest": 3,
            "exact_directory_count_below_root": 3,
            "all_files_mode_0444_nlink1": True,
            "all_directories_mode_0555": True,
        }
        or manifest.get("authority_graph") != {
            "sha256_graph_is_directed_and_acyclic": True,
            "manifest_pins_runtime_and_tests": True,
            "runtime_pins_controller_or_manifest": False,
            "detached_controller_is_outside_release_tree": True,
        }
        or stored_manifest_digest != manifest_digest
        or object_sha(unsigned_manifest) != manifest_digest):
    raise SystemExit("release manifest semantic closure differs")

expected_dirs = {"fold0", "fold1", "fold2", "fold3", "fold4", "logs"}
expected_files = {"launch-plan.json"}
expected_files |= {
    f"fold{fold}/{name}" for fold in range(5)
    for name in ("preselection.pt", "fixed1200.pt", "inner.json")
}
expected_files |= {
    f"logs/train-fold{fold}.{stream}" for fold in range(5)
    for stream in ("stdout", "stderr")
}
original_root_binding = directory_binding(original_root, 0o700)
found_files, found_dirs, stack = set(), set(), [original_root]
while stack:
    parent = stack.pop()
    with os.scandir(parent) as entries:
        for entry in entries:
            path = Path(entry.path); rel = path.relative_to(original_root).as_posix()
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise SystemExit("original symlink differs")
            if stat.S_ISDIR(info.st_mode):
                if stat.S_IMODE(info.st_mode) != 0o700:
                    raise SystemExit("original directory mode differs")
                found_dirs.add(rel); stack.append(path)
            elif stat.S_ISREG(info.st_mode):
                found_files.add(rel)
            else:
                raise SystemExit("original special member differs")
if found_dirs != expected_dirs or found_files != expected_files:
    raise SystemExit("original exact26/root+6dirs membership differs")
original_dirs = [directory_binding(original_root / rel, 0o700)
                 for rel in sorted(found_dirs)]
original_rows = []
original_files = {}
for rel in sorted(found_files):
    binding, _ = regular_binding(original_root / rel, 0o444)
    original_files[rel] = binding
    original_rows.append({
        "path": rel, "sha256": binding["sha256"],
        "size_bytes": binding["size_bytes"],
        "mode_octal": binding["mode_octal"], "nlink": binding["nlink"],
        "device": binding["device"], "inode": binding["inode"],
    })
stable = {
    "directories": [{"path": rel, "mode_octal": "0700"}
                    for rel in sorted(expected_dirs)],
    "files": [{key: row[key] for key in
               ("path", "sha256", "size_bytes", "mode_octal", "nlink")}
              for row in original_rows],
}
if (len(original_rows) != 26 or object_sha(original_rows) != exact26_sha
        or object_sha(stable) != parent_signature):
    raise SystemExit("original exact26 digest/signature differs")

snapshot = {
    "schema_version": "v4g-recovery-controller-authority-snapshot-v1",
    "controller": controller_binding,
    "controller_executed_source": controller_executed_source,
    "controller_shell": {
        "binding": shell_binding,
        "process_image_binding": shell_process_image,
        "bash_version": shell_version,
        "privileged_mode": True,
        "startup_shell_flags": "hpB",
        "startup_shopt_profile_exact": True,
        "startup_bashopts_observed_after_exact_launcher_gate":
            startup_bashopts,
        "declared_formal_launcher": (
            "env -i PATH=/usr/bin:/bin LC_ALL=C HOME=/nonexistent "
            "/bin/bash -p CONTROLLER EXPECTED_CONTROLLER_SHA256"
        ),
        "observed_process_image_environment_and_argv_contract_exact": True,
        "literal_parent_env_utility_invocation_observable": False,
        "startup_exported_names": ["HOME", "LC_ALL", "PATH", "PWD", "SHLVL"],
        "startup_exported_values": {
            "HOME": "/nonexistent", "LC_ALL": "C", "PATH": "/usr/bin:/bin",
            "PWD": startup_pwd, "SHLVL": "1",
        },
        "startup_environment_exact_clean5": True,
        "dangerous_startup_variables_absent": True,
    },
    "python": python_binding,
    "parent_python_held_image": parent_python_held_image,
    "authority_python_process_image": authority_python_process_image,
    "python_environment_trust_boundary": {
        "pinned_python_executable_and_process_image": True,
        "python_3_12_13_isolated_no_site_safe_path": True,
        "torch_2_7_1_rocm6_3_and_hip_6_3_canonical_origins_required": True,
        "python_environment_full_tree_cryptographically_pinned": False,
        "procedural_trusted_conda_environment_required": True,
    },
    "srun": srun_binding,
    "recovery_release": {
        "root_binding": release_root_binding,
        "directory_bindings": release_dirs,
        "file_bindings": release_files,
        "tree_rows": tree_rows,
        "release_tree_sha256": release_tree_sha,
        "manifest": manifest,
        "manifest_digest": stored_manifest_digest,
    },
    "original_run": {
        "root_binding": original_root_binding,
        "directory_bindings": original_dirs,
        "file_bindings": original_files,
        "files": original_rows,
        "exact26_manifest_sha256": exact26_sha,
        "parent_stable_signature_sha256": parent_signature,
        "exact_file_count": 26,
        "exact_directory_count_below_root": 6,
    },
}
print(canonical(snapshot).decode("ascii"))
PY
}

authority_snapshot >"${tmp_root}/authority.preflight.json" || \
  fail "authority preflight failed"
[[ $(wc -l <"${tmp_root}/authority.preflight.json") -eq 1 ]] || \
  fail "authority preflight output differs"

[[ ! -e ${recovery_root} && ! -L ${recovery_root} ]] || \
  fail "recovery root is not fresh"
[[ ! -e ${execution_root} && ! -L ${execution_root} ]] || \
  fail "controller execution root is not fresh"

run_tests() {
  local label=$1
  local optimize=$2
  local output=${tmp_root}/tests.${label}.json
  local errors=${tmp_root}/tests.${label}.stderr
  local -a opt=()
  if [[ ${optimize} == 1 ]]; then
    opt=(-O)
  fi
  if ! run_pinned_python -I -S -B "${opt[@]}" - \
      "${release_root}/${runtime_rel}" "${expected_runtime_sha256}" \
      "${release_root}/${tests_rel}" "${expected_tests_sha256}" \
      "${expected_test_count}" "${label}" \
      >"${output}" 2>"${errors}" <<'PY'
from pathlib import Path
import hashlib, importlib.machinery, io, json, os, stat, sys, types, unittest
runtime_path, runtime_sha = Path(sys.argv[1]), sys.argv[2]
tests_path, tests_sha = Path(sys.argv[3]), sys.argv[4]
expected_count, label = int(sys.argv[5]), sys.argv[6]
flags = sys.flags
if (tuple(sys.version_info[:3]) != (3, 12, 13)
        or flags.isolated != 1 or flags.no_site != 1
        or flags.ignore_environment != 1 or flags.safe_path is not True
        or flags.dont_write_bytecode != 1):
    raise SystemExit("test process Python version/flags differ")

def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, stat.S_IMODE(info.st_mode),
            info.st_nlink, info.st_mtime_ns, info.st_ctime_ns)

def capture(path, expected_sha):
    if (not path.is_absolute() or path.is_symlink()
            or path != path.resolve(strict=True)):
        raise SystemExit("test execution source path differs")
    before = path.lstat()
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    digest = hashlib.sha256(); chunks = []
    try:
        opened = os.fstat(fd)
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            digest.update(chunk); chunks.append(chunk)
        closed = os.fstat(fd)
    finally: os.close(fd)
    after = path.lstat()
    if (len({identity(x) for x in (before, opened, closed, after)}) != 1
            or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444 or before.st_nlink != 1
            or digest.hexdigest() != expected_sha):
        raise SystemExit("test execution captured binding differs")
    return {
        "path": str(path), "sha256": digest.hexdigest(),
        "size_bytes": before.st_size, "mode_octal": "0444",
        "nlink": before.st_nlink, "device": before.st_dev,
        "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "single_fd_pre_post_identity_and_sha_exact": True,
    }, b"".join(chunks)

runtime_binding, runtime_bytes = capture(runtime_path, runtime_sha)
tests_binding, tests_bytes = capture(tests_path, tests_sha)
methods = types.ModuleType("methods")
methods.__path__ = [str(runtime_path.parents[2])]
methods.__package__ = "methods"
action = types.ModuleType("methods.bernini_action_editing")
action.__path__ = [str(runtime_path.parent)]
action.__package__ = "methods.bernini_action_editing"
methods.bernini_action_editing = action
runtime_name = (
    "methods.bernini_action_editing."
    "recover_v4g_scientific_no_go_attestation_v1"
)
runtime = types.ModuleType(runtime_name)
runtime.__file__ = str(runtime_path); runtime.__package__ = runtime_name.rpartition(".")[0]
runtime.__loader__ = None
runtime.__spec__ = importlib.machinery.ModuleSpec(
    runtime_name, loader=None, origin=str(runtime_path),
)
sys.modules.update({
    "methods": methods,
    "methods.bernini_action_editing": action,
    runtime_name: runtime,
})
setattr(action, "recover_v4g_scientific_no_go_attestation_v1", runtime)
exec(compile(runtime_bytes, str(runtime_path), "exec", dont_inherit=True,
             optimize=sys.flags.optimize), runtime.__dict__)
tests_name = (
    "methods.bernini_action_editing.tests."
    "test_recover_v4g_scientific_no_go_attestation_v1"
)
tests_module = types.ModuleType(tests_name)
tests_module.__file__ = str(tests_path)
tests_module.__package__ = tests_name.rpartition(".")[0]
tests_module.__loader__ = None
tests_module.__spec__ = importlib.machinery.ModuleSpec(
    tests_name, loader=None, origin=str(tests_path),
)
sys.modules[tests_name] = tests_module
exec(compile(tests_bytes, str(tests_path), "exec", dont_inherit=True,
             optimize=sys.flags.optimize), tests_module.__dict__)
runtime_post, runtime_post_bytes = capture(runtime_path, runtime_sha)
tests_post, tests_post_bytes = capture(tests_path, tests_sha)
if (runtime_post != runtime_binding or runtime_post_bytes != runtime_bytes
        or tests_post != tests_binding or tests_post_bytes != tests_bytes):
    raise SystemExit("test executed-byte authority changed")
suite = unittest.defaultTestLoader.loadTestsFromModule(tests_module)
stream = io.StringIO()
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
transcript = stream.getvalue().encode("utf-8")
value = {
    "schema_version": "v4g-recovery-controller-structured-unittest-v1",
    "mode": label,
    "python_optimize": sys.flags.optimize,
    "tests_run": result.testsRun,
    "tests_skipped": len(result.skipped),
    "failures": len(result.failures),
    "errors": len(result.errors),
    "expected_failures": len(result.expectedFailures),
    "unexpected_successes": len(result.unexpectedSuccesses),
    "successful": result.wasSuccessful(),
    "transcript_sha256": hashlib.sha256(transcript).hexdigest(),
    "executed_runtime_binding": runtime_binding,
    "executed_tests_binding": tests_binding,
    "captured_bytes_compiled_and_executed": True,
    "path_import_execution_used": False,
}
print(json.dumps(value, sort_keys=True, separators=(",", ":")))
expected_optimize = 0 if label == "normal" else 1
if (result.testsRun != expected_count or len(result.skipped) != 0
        or len(result.failures) != 0 or len(result.errors) != 0
        or len(result.expectedFailures) != 0
        or len(result.unexpectedSuccesses) != 0
        or not result.wasSuccessful() or sys.flags.optimize != expected_optimize):
    raise SystemExit(1)
PY
  then
    [[ -s ${errors} ]] && sed -n '1,120p' "${errors}" >&2
    fail "${label} structured tests failed"
  fi
  [[ ! -s ${errors} && $(wc -l <"${output}") -eq 1 ]] || \
    fail "${label} structured test output differs"
}

run_tests normal 0
run_tests optimized 1

if ! run_pinned_python -I -S -B - \
    "${release_root}/${runtime_rel}" "${expected_runtime_sha256}" \
    "${release_root}/${tests_rel}" "${expected_tests_sha256}" \
    >"${tmp_root}/compile-ast.json" 2>"${tmp_root}/compile-ast.stderr" <<'PY'
from pathlib import Path
import ast, hashlib, json, os, stat, sys
flags = sys.flags
if (tuple(sys.version_info[:3]) != (3, 12, 13)
        or flags.isolated != 1 or flags.no_site != 1
        or flags.ignore_environment != 1 or flags.safe_path is not True
        or flags.dont_write_bytecode != 1 or flags.optimize != 0):
    raise SystemExit("compile process Python version/flags differ")
rows = []
for raw_path, expected_sha in ((sys.argv[1], sys.argv[2]),
                               (sys.argv[3], sys.argv[4])):
    path = Path(raw_path)
    if path.is_symlink() or path != path.resolve(strict=True):
        raise SystemExit("compile source path differs")
    before = path.lstat()
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd); chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            chunks.append(chunk)
        closed = os.fstat(fd)
    finally: os.close(fd)
    after = path.lstat()
    identity = lambda x: (x.st_dev, x.st_ino, x.st_size,
                          stat.S_IMODE(x.st_mode), x.st_nlink,
                          x.st_mtime_ns, x.st_ctime_ns)
    raw = b"".join(chunks)
    if (len({identity(x) for x in (before, opened, closed, after)}) != 1
            or stat.S_IMODE(opened.st_mode) != 0o444 or opened.st_nlink != 1
            or hashlib.sha256(raw).hexdigest() != expected_sha):
        raise SystemExit("compile source binding differs")
    source = raw.decode("utf-8")
    tree = ast.parse(source, filename=str(path))
    assert_count = sum(isinstance(node, ast.Assert) for node in ast.walk(tree))
    if assert_count != 0:
        raise SystemExit("compile source contains assert statement")
    compile(source, str(path), "exec", dont_inherit=True, optimize=0)
    compile(source, str(path), "exec", dont_inherit=True, optimize=2)
    rows.append({"path": str(path), "sha256": expected_sha,
                 "assert_node_count": 0, "compiled_optimize_levels": [0, 2]})
value = {"schema_version": "v4g-recovery-controller-compile-ast-v1",
         "source_count": 2, "sources": rows, "all_assert_nodes_absent": True}
print(json.dumps(value, sort_keys=True, separators=(",", ":")))
PY
then
  sed -n '1,120p' "${tmp_root}/compile-ast.stderr" >&2
  fail "compile/AST gate failed"
fi
[[ ! -s ${tmp_root}/compile-ast.stderr \
   && $(wc -l <"${tmp_root}/compile-ast.json") -eq 1 ]] || \
  fail "compile/AST output differs"

child_stdout=${tmp_root}/child.stdout
child_stderr=${tmp_root}/child.stderr
if ! "${srun_bin}" --jobid="${job_id}" --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --nodelist="${node}" --overlap --exact --gres=none \
    --kill-on-bad-exit=1 \
    --export=PATH=/usr/bin:/bin,LC_ALL=C,PYTHONDONTWRITEBYTECODE=1 \
    "${shell_bin}" -p -c '
      set -Eeuo pipefail
      [[ $- == *p* && -z ${BASH_ENV+x} && -z ${ENV+x} \
         && -z ${LD_PRELOAD+x} && -z ${LD_LIBRARY_PATH+x} ]] || exit 97
      exec {held_shell_fd}</proc/$$/exe
      exec {held_python_fd}<"$1"
      designated_python=$1
      expected_python_sha=$2
      expected_python_size=$3
      expected_shell_sha=$4
      expected_shell_size=$5
      shift 5
      python_fd_path=/proc/$$/fd/${held_python_fd}
      shell_fd_path=/proc/$$/fd/${held_shell_fd}
      python_before=$(/usr/bin/stat -Lc "%a:%h:%s:%d:%i:%Y:%Z:%F" \
        -- "${python_fd_path}")
      shell_before=$(/usr/bin/stat -Lc "%a:%h:%s:%d:%i:%Y:%Z:%F" \
        -- "${shell_fd_path}")
      python_sha_output=$(/usr/bin/sha256sum -- "${python_fd_path}")
      shell_sha_output=$(/usr/bin/sha256sum -- "${shell_fd_path}")
      python_after=$(/usr/bin/stat -Lc "%a:%h:%s:%d:%i:%Y:%Z:%F" \
        -- "${python_fd_path}")
      shell_after=$(/usr/bin/stat -Lc "%a:%h:%s:%d:%i:%Y:%Z:%F" \
        -- "${shell_fd_path}")
      [[ ${python_before} == "${python_after}" \
         && ${shell_before} == "${shell_after}" \
         && ${python_sha_output%% *} == "${expected_python_sha}" \
         && ${shell_sha_output%% *} == "${expected_shell_sha}" \
         && ${python_before} == 755:1:${expected_python_size}:* \
         && ${shell_before} == 755:1:${expected_shell_size}:* \
         && ! -L ${designated_python} \
         && $(/usr/bin/readlink -f -- "${designated_python}") \
            == "${designated_python}" \
         && $(/usr/bin/stat -Lc "%a:%h:%s:%d:%i:%Y:%Z:%F" \
              -- "${designated_python}") == "${python_before}" ]] || exit 98
      exec -a "${designated_python}" "/proc/$$/fd/${held_python_fd}" \
        -I -S -B - "$@" "${held_shell_fd}" "${held_python_fd}"
    ' v4g-recovery-child "${python_bin}" "${expected_python_sha256}" \
      "${expected_python_size}" "${expected_shell_sha256}" 1396520 \
      "${release_root}/${runtime_rel}" "${expected_runtime_sha256}" \
      "${original_root}" "${recovery_root}" \
      "${python_bin}" "${expected_python_sha256}" \
      "${shell_bin}" "${expected_shell_sha256}" \
      "${job_id}" "${node}" "${expected_torch_version}" \
      "${expected_torch_hip_version}" \
      >"${child_stdout}" 2>"${child_stderr}" <<'PY'
from pathlib import Path
import contextlib, hashlib, io, json, os, socket, stat, sys
(runtime_path, runtime_sha, original_root, recovery_root,
 python_path, python_sha, shell_path, shell_sha, expected_job, expected_node,
 expected_torch, expected_hip, shell_fd_raw, python_fd_raw) = sys.argv[1:]
runtime_path, python_path, shell_path = map(
    Path, (runtime_path, python_path, shell_path)
)
shell_fd, python_fd = int(shell_fd_raw), int(python_fd_raw)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)

def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, stat.S_IMODE(info.st_mode),
            info.st_nlink, info.st_mtime_ns, info.st_ctime_ns)

def binding_from_fd(descriptor, path, expected_sha, mode):
    before = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk: break
        digest.update(chunk)
    after = os.fstat(descriptor)
    if (identity(before) != identity(after) or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != mode or before.st_nlink != 1
            or digest.hexdigest() != expected_sha):
        raise SystemExit("child inherited image FD binding differs")
    return {
        "path": str(path), "sha256": digest.hexdigest(),
        "size_bytes": before.st_size, "mode_octal": f"{mode:04o}",
        "nlink": before.st_nlink, "device": before.st_dev,
        "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "single_fd_pre_post_identity_and_sha_exact": True,
    }

def capture_path(path, expected_sha, mode):
    if (not path.is_absolute() or path.is_symlink()
            or path != path.resolve(strict=True)):
        raise SystemExit("child captured path differs")
    before = path.lstat()
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    digest = hashlib.sha256(); chunks = []
    try:
        opened = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk: break
            digest.update(chunk); chunks.append(chunk)
        closed = os.fstat(descriptor)
    finally: os.close(descriptor)
    after = path.lstat()
    if (len({identity(x) for x in (before, opened, closed, after)}) != 1
            or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != mode or before.st_nlink != 1
            or digest.hexdigest() != expected_sha):
        raise SystemExit("child captured path binding differs")
    return {
        "path": str(path), "sha256": digest.hexdigest(),
        "size_bytes": before.st_size, "mode_octal": f"{mode:04o}",
        "nlink": before.st_nlink, "device": before.st_dev,
        "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "single_fd_pre_post_identity_and_sha_exact": True,
    }, b"".join(chunks)

executable = Path(sys.executable).resolve(strict=True)
actual_executable = Path("/proc/self/exe").resolve(strict=True)
if executable != python_path or actual_executable != python_path:
    raise SystemExit("child Python path differs")
flags = sys.flags
if (tuple(sys.version_info[:3]) != (3, 12, 13)
        or flags.isolated != 1 or flags.no_site != 1
        or flags.ignore_environment != 1 or flags.safe_path is not True
        or flags.dont_write_bytecode != 1):
    raise SystemExit("child Python flags/version differ")
designated_python, _ = capture_path(python_path, python_sha, 0o755)
inherited_python = binding_from_fd(python_fd, python_path, python_sha, 0o755)
actual_fd = os.open("/proc/self/exe", os.O_RDONLY | os.O_CLOEXEC)
try:
    actual_python = binding_from_fd(actual_fd, python_path, python_sha, 0o755)
finally: os.close(actual_fd)
compute_shell = binding_from_fd(shell_fd, shell_path, shell_sha, 0o755)
binding_values = lambda value: tuple(value[key] for key in (
    "sha256", "size_bytes", "mode_octal", "nlink", "device", "inode",
    "mtime_ns", "ctime_ns", "single_fd_pre_post_identity_and_sha_exact",
))
if len({binding_values(value) for value in
        (designated_python, inherited_python, actual_python)}) != 1:
    raise SystemExit("child designated/inherited/actual Python image differs")

affinity = sorted(os.sched_getaffinity(0))
gpu_keys = ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "SLURM_STEP_GPUS",
            "SLURM_JOB_GPUS")
gpu_env = {key: os.environ.get(key) for key in gpu_keys}
allowed_hidden = {None, "", "NoDevFiles"}
if (len(affinity) != 1 or os.environ.get("SLURM_JOB_ID") != expected_job
        or os.environ.get("SLURM_STEP_NUM_NODES") != "1"
        or os.environ.get("SLURM_STEP_NUM_TASKS") != "1"
        or os.environ.get("SLURM_NNODES") != "1"
        or os.environ.get("SLURM_NTASKS") != "1"
        or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
        or os.environ.get("SLURM_PROCID") != "0"
        or os.environ.get("SLURM_LOCALID") != "0"
        or os.environ.get("SLURM_NODEID") != "0"
        or os.environ.get("SLURMD_NODENAME") != expected_node
        or socket.gethostname().split(".", 1)[0] != expected_node
        or os.environ.get("SLURM_STEP_GPUS") not in (None, "")
        or any(gpu_env[key] not in allowed_hidden for key in
               ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"))):
    raise SystemExit("child exact1 CPU/no-GPU observation differs")

runtime_binding, runtime_bytes = capture_path(runtime_path, runtime_sha, 0o444)
runtime_stdout = io.StringIO()
saved_argv = sys.argv
try:
    sys.argv = [str(runtime_path), "--original-run-root", original_root,
                "--recovery-root", recovery_root]
    with contextlib.redirect_stdout(runtime_stdout):
        try:
            namespace = {
                "__name__": "__main__", "__file__": str(runtime_path),
                "__package__": None, "__loader__": None, "__spec__": None,
            }
            exec(compile(runtime_bytes, str(runtime_path), "exec",
                         dont_inherit=True, optimize=sys.flags.optimize), namespace)
        except SystemExit as error:
            if error.code not in (0, None):
                raise
finally:
    sys.argv = saved_argv
runtime_post_binding, runtime_post_bytes = capture_path(
    runtime_path, runtime_sha, 0o444,
)
if runtime_post_binding != runtime_binding or runtime_post_bytes != runtime_bytes:
    raise SystemExit("child captured runtime changed during execution")
lines = runtime_stdout.getvalue().splitlines()
if len(lines) != 1:
    raise SystemExit("runtime stdout is not exact1")
def duplicate_rejecting_hook(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit("runtime result duplicate JSON key")
        value[key] = item
    return value
runtime_result = json.loads(lines[0], object_pairs_hook=duplicate_rejecting_hook)
if canonical(runtime_result) != lines[0]:
    raise SystemExit("runtime result is not canonical exact1 JSON")
torch = sys.modules.get("torch")
torch_version = str(getattr(torch, "__version__", None))
torch_hip_version = str(getattr(getattr(torch, "version", None), "hip", None))
torch_visible_gpu_count = int(torch.cuda.device_count()) if torch is not None else -1
if (torch_version != expected_torch or torch_hip_version != expected_hip
        or torch_visible_gpu_count != 0):
    raise SystemExit("child Torch/HIP pin differs")

value = {
    "schema_version": "v4g-recovery-controller-child-execution-v1",
    "slurm": {
        "job_id": os.environ["SLURM_JOB_ID"],
        "step_id": os.environ.get("SLURM_STEP_ID"),
        "step_num_nodes": os.environ.get("SLURM_STEP_NUM_NODES"),
        "step_num_tasks": os.environ.get("SLURM_STEP_NUM_TASKS"),
        "nnodes": os.environ.get("SLURM_NNODES"),
        "ntasks": os.environ.get("SLURM_NTASKS"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "procid": os.environ.get("SLURM_PROCID"),
        "localid": os.environ.get("SLURM_LOCALID"),
        "nodeid": os.environ.get("SLURM_NODEID"),
        "slurmd_nodename": os.environ.get("SLURMD_NODENAME"),
        "step_nodelist": os.environ.get("SLURM_STEP_NODELIST"),
        "job_nodelist": os.environ.get("SLURM_NODELIST"),
        "hostname": socket.gethostname().split(".", 1)[0],
        "cpu_affinity": affinity, "gres_request": "none",
        "gpu_environment": gpu_env,
        "torch_visible_gpu_count": torch_visible_gpu_count,
        "exact_one_cpu_observed": True, "no_gpu_observed": True,
    },
    "python": {
        "designated_binding": designated_python,
        "inherited_exec_fd_binding": inherited_python,
        "proc_self_exe_binding": actual_python,
        "compute_shell_inherited_fd_binding": compute_shell,
        "designated_inherited_proc_image_exact": True,
        "version": ".".join(map(str, sys.version_info[:3])),
        "isolated": True, "no_site": True, "ignore_environment": True,
        "safe_path": True, "dont_write_bytecode": True,
        "torch_version": torch_version,
        "torch_hip_version": torch_hip_version,
    },
    "runtime_execution": {
        "binding": runtime_binding,
        "captured_bytes_compiled_and_executed": True,
        "path_import_or_runpy_used": False,
    },
    "runtime_result": runtime_result,
}
print(canonical(value))
PY
then
  [[ -s ${child_stderr} ]] && sed -n '1,160p' "${child_stderr}" >&2
  fail "recovery CPU exact1 child failed"
fi
[[ ! -s ${child_stderr} && $(wc -l <"${child_stdout}") -eq 1 ]] || \
  fail "recovery child output differs"

if ! run_pinned_python -I -S -B - "${recovery_root}" \
    >"${tmp_root}/recovery-visibility.json" \
    2>"${tmp_root}/recovery-visibility.stderr" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys, time
root = Path(sys.argv[1])

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, stat.S_IMODE(info.st_mode),
            info.st_nlink, info.st_mtime_ns, info.st_ctime_ns)
def directory_binding(path):
    if (path.is_symlink() or path != path.resolve(strict=True)
            or not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW")):
        raise RuntimeError("visible recovery root path differs")
    before = path.lstat()
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd); members = sorted(os.listdir(fd)); closed = os.fstat(fd)
    finally: os.close(fd)
    after = path.lstat()
    if (len({identity(x) for x in (before, opened, closed, after)}) != 1
            or not stat.S_ISDIR(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o555
            or members != ["recovery-attestation.json"]):
        raise RuntimeError("visible recovery root exact1 binding differs")
    return {
        "path": str(path), "mode_octal": "0555", "nlink": before.st_nlink,
        "device": before.st_dev, "inode": before.st_ino,
        "mtime_ns": before.st_mtime_ns, "ctime_ns": before.st_ctime_ns,
        "members": members,
        "single_fd_pre_post_identity_and_membership_exact": True,
    }
def file_binding(path):
    before = path.lstat()
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    try:
        opened = os.fstat(fd)
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            digest.update(chunk)
        closed = os.fstat(fd)
    finally: os.close(fd)
    after = path.lstat()
    if (len({identity(x) for x in (before, opened, closed, after)}) != 1
            or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444 or before.st_nlink != 1):
        raise RuntimeError("visible recovery file binding differs")
    return {
        "path": str(path), "sha256": digest.hexdigest(),
        "size_bytes": before.st_size, "mode_octal": "0444",
        "nlink": before.st_nlink, "device": before.st_dev,
        "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "single_fd_pre_post_identity_and_sha_exact": True,
    }

previous = None; consecutive = 0; stable = None
for attempt in range(1, 21):
    if not os.path.lexists(root):
        previous = None; consecutive = 0
    else:
        try:
            current = {
                "root_binding": directory_binding(root),
                "file_binding": file_binding(root / "recovery-attestation.json"),
            }
        except (FileNotFoundError, OSError, RuntimeError):
            previous = None; consecutive = 0
        else:
            consecutive = consecutive + 1 if current == previous else 1
            previous = current
            if consecutive >= 3:
                stable = current
                break
    if attempt < 20:
        time.sleep(1.0)
if stable is None:
    raise SystemExit("recovery exact1 did not reach three consecutive stable samples")
value = {
    "schema_version": "v4g-recovery-controller-bounded-visibility-v1",
    "attempt_budget": 20, "interval_seconds": 1,
    "attempts_used": attempt, "required_consecutive_stable_samples": 3,
    "observed_consecutive_stable_samples": consecutive,
    "recovery": stable,
}
print(canonical(value).decode("ascii"))
PY
then
  sed -n '1,120p' "${tmp_root}/recovery-visibility.stderr" >&2
  fail "recovery bounded visibility failed"
fi
[[ ! -s ${tmp_root}/recovery-visibility.stderr \
   && $(wc -l <"${tmp_root}/recovery-visibility.json") -eq 1 ]] || \
  fail "recovery visibility output differs"

if ! run_pinned_python -I -S -B - \
    "${child_stdout}" "${tmp_root}/recovery-visibility.json" \
    "${tmp_root}/authority.preflight.json" \
    "${release_root}/${runtime_rel}" "${expected_runtime_sha256}" \
    "${original_root}" "${recovery_root}" "${expected_exact26_sha256}" \
    "${expected_runtime_sha256}" "${expected_tests_sha256}" \
    "${expected_manifest_sha256}" "${expected_manifest_digest}" \
    "${expected_release_tree_sha256}" "${expected_controller_sha256}" \
    "${expected_torch_version}" "${expected_torch_hip_version}" \
    >"${tmp_root}/recovery-attestation.validation.json" \
    2>"${tmp_root}/recovery-attestation.validation.stderr" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys, types
(child_path, visibility_path, post_path, recovery_runtime_path,
 recovery_runtime_sha, original_raw, recovery_raw, exact26_sha,
 runtime_sha, tests_sha, manifest_sha, manifest_digest, tree_sha,
 controller_sha, torch_version, torch_hip_version) = sys.argv[1:]
child_path, visibility_path, post_path, recovery_runtime_path = map(
    Path, (child_path, visibility_path, post_path, recovery_runtime_path)
)
original, recovery = Path(original_raw), Path(recovery_raw)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
def object_sha(value): return hashlib.sha256(canonical(value)).hexdigest()
def strict(raw):
    def hook(pairs):
        out = {}
        for key, value in pairs:
            if key in out: raise SystemExit("duplicate JSON key")
            out[key] = value
        return out
    value = json.loads(raw.decode("ascii"), object_pairs_hook=hook)
    if canonical(value) + b"\n" != raw:
        raise SystemExit("canonical JSON bytes differ")
    return value
def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, stat.S_IMODE(info.st_mode),
            info.st_nlink, info.st_mtime_ns, info.st_ctime_ns)
def read_exact(path):
    if (not path.is_absolute() or path.is_symlink()
            or path != path.resolve(strict=True)):
        raise SystemExit("attestation input path differs")
    before = path.lstat()
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    digest = hashlib.sha256(); chunks = []
    try:
        opened = os.fstat(fd)
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            digest.update(chunk); chunks.append(chunk)
        closed = os.fstat(fd)
    finally: os.close(fd)
    after = path.lstat()
    if (len({identity(x) for x in (before, opened, closed, after)}) != 1
            or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1):
        raise SystemExit("attestation same-FD identity differs")
    return before, digest.hexdigest(), b"".join(chunks)

def regular_binding(info, path, digest):
    return {
        "path": str(path), "sha256": digest, "size_bytes": info.st_size,
        "mode_octal": f"{stat.S_IMODE(info.st_mode):04o}",
        "nlink": info.st_nlink, "device": info.st_dev, "inode": info.st_ino,
        "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
        "single_fd_pre_post_identity_and_sha_exact": True,
    }

def cross_node_stable_projection(binding):
    if (type(binding) is not dict
            or binding.get("single_fd_pre_post_identity_and_sha_exact") is not True):
        raise SystemExit("cross-node binding lacks local same-FD evidence")
    return {key: binding.get(key) for key in (
        "path", "sha256", "size_bytes", "mode_octal", "nlink",
    )}

def cross_node_directory_projection(binding):
    if (type(binding) is not dict
            or binding.get(
                "single_fd_pre_post_identity_and_membership_exact") is not True):
        raise SystemExit("cross-node directory lacks local same-FD evidence")
    return {key: binding.get(key) for key in ("path", "mode_octal", "members")}

def burned_row_stable_projection(row):
    if type(row) is not dict:
        raise SystemExit("burned row stable projection schema differs")
    return {key: row.get(key) for key in (
        "path", "sha256", "size_bytes", "mode_octal", "nlink",
    )}

def historical_release_stable_projection(value):
    if type(value) is not dict or set(value) != {
        "root", "tree_sha256", "manifest", "manifest_digest",
        "file_count", "directory_count",
    }:
        raise SystemExit("historical release projection schema differs")
    return {
        **{key: value[key] for key in (
            "root", "tree_sha256", "manifest_digest", "file_count",
            "directory_count",
        )},
        "manifest": cross_node_stable_projection(value["manifest"]),
    }

def recovery_release_stable_projection(value):
    if type(value) is not dict or set(value) != {
        "release_root", "release_root_binding", "release_tree_sha256",
        "tree_rows", "manifest", "manifest_digest", "runtime", "tests",
        "controller", "exact_file_count", "exact_directory_count_below_root",
        "one_way_sha256_dag_reverified",
        "controller_identity_recorded_not_runtime_reverse_pinned",
    }:
        raise SystemExit("recovery release projection schema differs")
    return {
        **{key: value[key] for key in (
            "release_root", "release_tree_sha256", "tree_rows",
            "manifest_digest", "exact_file_count",
            "exact_directory_count_below_root", "one_way_sha256_dag_reverified",
            "controller_identity_recorded_not_runtime_reverse_pinned",
        )},
        "release_root_binding": cross_node_directory_projection(
            value["release_root_binding"]),
        **{key: cross_node_stable_projection(value[key]) for key in (
            "manifest", "runtime", "tests", "controller",
        )},
    }

def input_snapshot_stable_projection(value):
    if type(value) is not dict or set(value) != {
        "sha256", "ordered_rows", "feature_shard_bindings",
        "exact_receipt_count", "exact_feature_shard_count",
        "all_ten_files_single_fd_reverified",
    }:
        raise SystemExit("input snapshot projection schema differs")
    return {
        **{key: value[key] for key in (
            "sha256", "ordered_rows", "exact_receipt_count",
            "exact_feature_shard_count", "all_ten_files_single_fd_reverified",
        )},
        "feature_shard_bindings": [
            cross_node_stable_projection(binding)
            for binding in value["feature_shard_bindings"]
        ],
    }

child_info, child_sha, child_raw = read_exact(child_path)
visibility_info, visibility_sha, visibility_raw = read_exact(visibility_path)
post_info, post_sha, post_raw = read_exact(post_path)
if any(stat.S_IMODE(info.st_mode) != 0o600
       for info in (child_info, visibility_info, post_info)):
    raise SystemExit("validator intermediate input seal differs")
child = strict(child_raw)
visibility = strict(visibility_raw)
post = strict(post_raw)
validated_intermediate_bindings = {
    "child.stdout": regular_binding(child_info, child_path, child_sha),
    "recovery-visibility.json": regular_binding(
        visibility_info, visibility_path, visibility_sha,
    ),
    "authority.preflight.json": regular_binding(post_info, post_path, post_sha),
}
runtime_info, runtime_file_sha, recovery_runtime_bytes = read_exact(
    recovery_runtime_path,
)
recovery_runtime_binding = regular_binding(
    runtime_info, recovery_runtime_path, runtime_file_sha,
)
if (runtime_file_sha != recovery_runtime_sha
        or stat.S_IMODE(runtime_info.st_mode) != 0o444
        or runtime_info.st_nlink != 1
        or recovery_runtime_binding
           != post.get("recovery_release", {}).get("file_bindings", {}).get(
               "methods/bernini_action_editing/"
               "recover_v4g_scientific_no_go_attestation_v1.py")):
    raise SystemExit("validator captured recovery runtime binding differs")
runtime_module = types.ModuleType("_v4g_recovery_captured_authority")
runtime_module.__file__ = str(recovery_runtime_path)
runtime_module.__package__ = None
runtime_module.__loader__ = None
runtime_module.__spec__ = None
exec(compile(recovery_runtime_bytes, str(recovery_runtime_path), "exec",
             dont_inherit=True, optimize=sys.flags.optimize),
     runtime_module.__dict__)
if runtime_module.RELEASE_SEALED is not True:
    raise SystemExit("captured recovery runtime is not sealed")

if set(child) != {"schema_version", "slurm", "python", "runtime_execution",
                  "runtime_result"}:
    raise SystemExit("child exact schema differs")
if child.get("schema_version") != "v4g-recovery-controller-child-execution-v1":
    raise SystemExit("child schema version differs")
python = child["python"]
if (set(python) != {
        "designated_binding", "inherited_exec_fd_binding",
        "proc_self_exe_binding", "compute_shell_inherited_fd_binding",
        "designated_inherited_proc_image_exact", "version", "isolated",
        "no_site", "ignore_environment", "safe_path", "dont_write_bytecode",
        "torch_version", "torch_hip_version"}
        or python.get("torch_version") != torch_version
        or python.get("torch_hip_version") != torch_hip_version
        or any(cross_node_stable_projection(python.get(key))
               != cross_node_stable_projection(post.get("python"))
               for key in ("designated_binding", "inherited_exec_fd_binding",
                           "proc_self_exe_binding"))
        or cross_node_stable_projection(
               python.get("compute_shell_inherited_fd_binding"))
           != cross_node_stable_projection(
               post.get("controller_shell", {}).get("binding"))
        or python.get("designated_inherited_proc_image_exact") is not True
        or python.get("version") != "3.12.13"
        or any(python.get(key) is not True for key in
               ("isolated", "no_site", "ignore_environment", "safe_path",
                "dont_write_bytecode"))):
    raise SystemExit("child Python/Torch schema differs")
slurm = child["slurm"]
if (set(slurm) != {"job_id", "step_id", "step_num_nodes",
                   "step_num_tasks", "nnodes", "ntasks", "cpus_per_task",
                   "procid", "localid", "nodeid", "slurmd_nodename",
                   "step_nodelist", "job_nodelist", "hostname",
                   "cpu_affinity", "gres_request", "gpu_environment",
                   "torch_visible_gpu_count",
                   "exact_one_cpu_observed", "no_gpu_observed"}
        or slurm.get("job_id") != "143808"
        or slurm.get("step_num_nodes") != "1"
        or slurm.get("step_num_tasks") != "1"
        or slurm.get("nnodes") != "1" or slurm.get("ntasks") != "1"
        or slurm.get("cpus_per_task") != "1"
        or slurm.get("procid") != "0" or slurm.get("localid") != "0"
        or slurm.get("nodeid") != "0"
        or slurm.get("slurmd_nodename") != "auh7-1b-gpu-268"
        or slurm.get("hostname") != "auh7-1b-gpu-268"
        or type(slurm.get("step_id")) is not str or not slurm["step_id"]
        or type(slurm.get("cpu_affinity")) is not list
        or len(slurm["cpu_affinity"]) != 1
        or slurm.get("gres_request") != "none"
        or set(slurm.get("gpu_environment", {})) != {
            "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "SLURM_STEP_GPUS",
            "SLURM_JOB_GPUS"}
        or slurm["gpu_environment"].get("SLURM_STEP_GPUS") not in (None, "")
        or slurm.get("torch_visible_gpu_count") != 0
        or slurm.get("exact_one_cpu_observed") is not True
        or slurm.get("no_gpu_observed") is not True):
    raise SystemExit("child Slurm exact1/no-GPU schema differs")
runtime_execution = child["runtime_execution"]
if (set(runtime_execution) != {
        "binding", "captured_bytes_compiled_and_executed",
        "path_import_or_runpy_used"}
        or cross_node_stable_projection(runtime_execution.get("binding"))
           != cross_node_stable_projection(recovery_runtime_binding)
        or runtime_execution.get("captured_bytes_compiled_and_executed") is not True
        or runtime_execution.get("path_import_or_runpy_used") is not False):
    raise SystemExit("child captured runtime execution binding differs")

result = child["runtime_result"]
result_keys = {
    "path", "file_sha256", "size_bytes", "receipt_digest", "mode_octal",
    "nlink", "root_mode_octal", "exact_file_count",
    "create_only_name_claim", "failure_tombstone_root_mode_octal",
    "original_run_and_source_authorities_reverified_after_name_claim",
    "root_and_file_same_fd_precommit_verified_and_parent_fsynced",
    "producer_root_precommit_binding", "producer_attestation_final_binding",
    "root_creation_to_precommit_device_inode_exact",
    "file_creation_to_final_device_inode_exact",
    "final_mode_commit", "final_mode_commit_order",
    "schema_version", "original_run_postverified_unchanged",
    "original_run_exact26_manifest_sha256", "scientific_result",
    "original_controller_complete",
}
if (type(result) is not dict or set(result) != result_keys
        or result.get("schema_version")
           != "v4g-scientific-no-go-sibling-recovery-result-v4"
        or result.get("path") != str(recovery / "recovery-attestation.json")
        or result.get("mode_octal") != "0444" or result.get("nlink") != 1
        or result.get("root_mode_octal") != "0555"
        or result.get("exact_file_count") != 1
        or result.get("create_only_name_claim") is not True
        or result.get("failure_tombstone_root_mode_octal") != "0700"
        or result.get("original_run_and_source_authorities_reverified_after_name_claim") is not True
        or result.get("root_and_file_same_fd_precommit_verified_and_parent_fsynced") is not True
        or type(result.get("producer_root_precommit_binding")) is not dict
        or set(result["producer_root_precommit_binding"]) != {
            "path", "mode_octal", "nlink", "device", "inode", "mtime_ns",
            "ctime_ns", "members",
            "single_fd_pre_post_identity_and_membership_exact"}
        or result["producer_root_precommit_binding"].get("path") != str(recovery)
        or result["producer_root_precommit_binding"].get("mode_octal") != "0700"
        or result["producer_root_precommit_binding"].get("members")
           != ["recovery-attestation.json"]
        or result["producer_root_precommit_binding"].get(
            "single_fd_pre_post_identity_and_membership_exact") is not True
        or type(result.get("producer_attestation_final_binding")) is not dict
        or set(result["producer_attestation_final_binding"]) != {
            "path", "sha256", "size_bytes", "mode_octal", "nlink", "device",
            "inode", "mtime_ns", "ctime_ns",
            "single_fd_pre_post_identity_and_sha_exact"}
        or result["producer_attestation_final_binding"].get("path")
           != str(recovery / "recovery-attestation.json")
        or result["producer_attestation_final_binding"].get("sha256")
           != result.get("file_sha256")
        or result["producer_attestation_final_binding"].get("size_bytes")
           != result.get("size_bytes")
        or result["producer_attestation_final_binding"].get("mode_octal")
           != "0444"
        or result["producer_attestation_final_binding"].get("nlink") != 1
        or result["producer_attestation_final_binding"].get(
            "single_fd_pre_post_identity_and_sha_exact") is not True
        or result.get("root_creation_to_precommit_device_inode_exact") is not True
        or result.get("file_creation_to_final_device_inode_exact") is not True
        or result.get("final_mode_commit") is not True
        or result.get("final_mode_commit_order") != ["file_0444", "root_0555"]
        or result.get("original_run_postverified_unchanged") is not True
        or result.get("original_run_exact26_manifest_sha256") != exact26_sha
        or result.get("scientific_result") != "ALL_FIVE_INNER_NO_GO_ALL_OOF_UNREAD"
        or result.get("original_controller_complete") is not False):
    raise SystemExit("runtime result exact23 differs")

if (recovery.is_symlink() or recovery != recovery.resolve(strict=True)
        or stat.S_IMODE(recovery.lstat().st_mode) != 0o555
        or sorted(item.name for item in recovery.iterdir())
           != ["recovery-attestation.json"]):
    raise SystemExit("recovery exact1 root differs")
attestation_path = recovery / "recovery-attestation.json"
info, file_sha, raw = read_exact(attestation_path)
attestation_file_binding = regular_binding(info, attestation_path, file_sha)
live_root_binding = runtime_module._read_directory_binding(recovery, mode=0o555)
if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444
        or info.st_nlink != 1 or info.st_size != result.get("size_bytes")
        or file_sha != result.get("file_sha256")
        or cross_node_stable_projection(
               result.get("producer_attestation_final_binding"))
           != cross_node_stable_projection(attestation_file_binding)
        or any(type(result["producer_root_precommit_binding"].get(key)) is not int
               for key in ("nlink", "device", "inode", "mtime_ns", "ctime_ns"))
        or any(type(result["producer_attestation_final_binding"].get(key)) is not int
               for key in ("size_bytes", "nlink", "device", "inode",
                           "mtime_ns", "ctime_ns"))
        or result["producer_root_precommit_binding"].get("path")
           != live_root_binding.get("path")
        or result["producer_root_precommit_binding"].get("members")
           != live_root_binding.get("members")
        or set(visibility) != {
            "schema_version", "attempt_budget", "interval_seconds",
            "attempts_used", "required_consecutive_stable_samples",
            "observed_consecutive_stable_samples", "recovery"}
        or visibility.get("attempt_budget") != 20
        or visibility.get("interval_seconds") != 1
        or visibility.get("required_consecutive_stable_samples") != 3
        or visibility.get("observed_consecutive_stable_samples", 0) < 3
        or visibility.get("attempts_used", 0) not in range(3, 21)
        or visibility.get("recovery") != {
            "root_binding": live_root_binding,
            "file_binding": attestation_file_binding,
        }):
    raise SystemExit("recovery attestation file binding differs")
attestation = strict(raw)
attestation_keys = {
    "schema_version", "authority", "original_run_root", "recovery_root",
    "original_run_mutated_by_recovery", "original_run_postverified_unchanged",
    "original_controller_complete", "original_controller_exit_nonzero",
    "scientifically_verified_all_inner_no_go", "global_inner_barrier_created",
    "evaluate_fold_executed", "aggregate_executed",
    "all_fold_oof_semantic_tensor_read_count",
    "all_fold_oof_semantic_tensor_materialized_count",
    "recovery_ledger_reconstructed_from_burned_exact26",
    "burned_exact26_file_count", "burned_exact26_manifest_sha256",
    "burned_exact26_manifest", "burned_parent_stable_signature_sha256",
    "original_run_root_binding", "old_controller_identity_schema_bug",
    "source_authority", "launch_plan", "original_controller_logs",
    "failed_seal_child_accounting", "folds", "all_qualification_claims_false",
    "qualification_scope", "receipt_digest",
}
if type(attestation) is not dict or set(attestation) != attestation_keys:
    raise SystemExit("attestation canonical exact keys differ")
stored = attestation.pop("receipt_digest")
if stored != object_sha(attestation) or stored != result.get("receipt_digest"):
    raise SystemExit("attestation self digest differs")
post_original = post.get("original_run")
rows = attestation.get("burned_exact26_manifest")
expected_scope = {
    "known_exposed_development_gate": None,
    "known_exposed_development_gate_evaluated": False,
    "unseen_hostile_transform_gate": False,
    "unseen_hostile_transform_gate_evaluated": False,
    "latent_metric_qualified": False,
    "action_representation_qualified": False,
    "identity_disentanglement_qualified": False,
    "identity_preservation_qualified": False,
    "prior_qualified": False,
    "prior_generation_qualified": False,
    "generation_qualified": False,
    "renderer_qualified": False,
    "video_editing_qualified": False,
    "inference_authorized": False,
    "web_evaluation_authorized": False,
    "full644_refit_authorized": False,
    "video_model_training_performed": False,
    "html_or_video_generated": False,
    "vae_necessary": None,
}
expected_bug = {
    "runtime_receipt_identity_keys": ["device", "inode", "size_bytes"],
    "phase_binding_additional_keys": [
        "mode_octal", "nlink", "mtime_ns", "ctime_ns",
    ],
    "all_ten_checkpoint_three_field_projections_exact": True,
    "all_ten_checkpoint_full_object_equal": False,
    "mode_and_nlink_verified_independently": True,
    "bug_affected_scientific_values": False,
    "bug_only_blocked_final_controller_seal": True,
}
if (attestation.get("schema_version")
        != "v4g-scientific-no-go-sibling-recovery-attestation-v1"
        or attestation.get("authority")
           != "burned_known_transform_development_scientific_no_go_only"
        or attestation.get("original_run_root") != str(original)
        or attestation.get("recovery_root") != str(recovery)
        or attestation.get("original_run_mutated_by_recovery") is not False
        or attestation.get("original_run_postverified_unchanged") is not True
        or attestation.get("original_controller_complete") is not False
        or attestation.get("original_controller_exit_nonzero") is not True
        or attestation.get("scientifically_verified_all_inner_no_go") is not True
        or attestation.get("global_inner_barrier_created") is not False
        or attestation.get("evaluate_fold_executed") is not False
        or attestation.get("aggregate_executed") is not False
        or attestation.get("all_fold_oof_semantic_tensor_read_count") != 0
        or attestation.get("all_fold_oof_semantic_tensor_materialized_count") != 0
        or attestation.get("recovery_ledger_reconstructed_from_burned_exact26") is not True
        or attestation.get("burned_exact26_file_count") != 26
        or attestation.get("burned_exact26_manifest_sha256") != exact26_sha
        or type(rows) is not list or len(rows) != 26 or object_sha(rows) != exact26_sha
        or [burned_row_stable_projection(row) for row in rows]
           != [burned_row_stable_projection(row)
               for row in post_original.get("files", [])]
        or cross_node_directory_projection(
               attestation.get("original_run_root_binding"))
           != cross_node_directory_projection(post_original.get("root_binding"))
        or attestation.get("all_qualification_claims_false") is not True
        or attestation.get("qualification_scope") != expected_scope
        or attestation.get("old_controller_identity_schema_bug") != expected_bug):
    raise SystemExit("attestation scientific/original join differs")
folds = attestation.get("folds")
fold_keys = {
    "fold_index", "inner_receipt_sha256", "inner_receipt_digest",
    "inner_status", "inner_pass", "oof_semantic_tensor_read_count",
    "oof_semantic_tensor_materialized_count", "model_fit_count",
    "inner_count", "oof_count", "preselection_checkpoint",
    "fixed1200_checkpoint",
    "three_field_runtime_physical_identity_projection_exact",
    "mode_and_nlink_verified_separately", "fidelity_gate",
    "all_three_negative_full_gates", "complete_gate",
}
if (type(folds) is not list or len(folds) != 5
        or any(type(row) is not dict or set(row) != fold_keys for row in folds)
        or [row.get("fold_index") for row in folds] != list(range(5))
        or any(row.get("inner_pass") is not False for row in folds)
        or any(row.get("inner_status")
               != "V4G_FIXED1200_INNER_NO_GO_ALL_OOF_UNREAD" for row in folds)
        or any(row.get("oof_semantic_tensor_read_count") != 0 for row in folds)
        or any(row.get("oof_semantic_tensor_materialized_count") != 0 for row in folds)
        or any(row.get("fidelity_gate") is not False for row in folds)
        or any(row.get("all_three_negative_full_gates") is not False for row in folds)
        or any(row.get("complete_gate") is not False for row in folds)
        or any(row.get("three_field_runtime_physical_identity_projection_exact")
               is not True for row in folds)
        or any(row.get("mode_and_nlink_verified_separately") is not True
               for row in folds)):
    raise SystemExit("attestation exact5 fold summary differs")
source = attestation.get("source_authority")
if (type(source) is not dict or set(source) != {
        "authority_snapshot", "release", "controller", "python",
        "process_python", "input_receipts", "input_snapshot",
        "runtime_sha256", "tests_sha256", "recovery",
        "recovery_parser_torch"}):
    raise SystemExit("attestation source authority exact keys differ")
recovery_source = source.get("recovery")
release_post = post.get("recovery_release")
if (source.get("runtime_sha256") != "38b2cbecaf022e203ccf09e6808661013f4f23dee0d02ffa1756e24d0c167cf9"
        or source.get("tests_sha256") != "7fe6b42208f77171f99d44d5a9fc9eae58c3bb2d4663ca016e9b154a4d3c4996"
        or type(recovery_source) is not dict
        or set(recovery_source) != {
            "release_root", "release_root_binding", "release_tree_sha256",
            "tree_rows", "manifest", "manifest_digest", "runtime", "tests",
            "controller", "exact_file_count", "exact_directory_count_below_root",
            "one_way_sha256_dag_reverified",
            "controller_identity_recorded_not_runtime_reverse_pinned"}
        or recovery_source.get("release_tree_sha256") != tree_sha
        or recovery_source.get("manifest_digest") != manifest_digest
        or recovery_source.get("runtime", {}).get("sha256") != runtime_sha
        or recovery_source.get("tests", {}).get("sha256") != tests_sha
        or recovery_source.get("manifest", {}).get("sha256") != manifest_sha
        or recovery_source.get("controller", {}).get("sha256") != controller_sha
        or cross_node_directory_projection(
               recovery_source.get("release_root_binding"))
           != cross_node_directory_projection(release_post.get("root_binding"))
        or recovery_source.get("tree_rows") != release_post.get("tree_rows")
        or cross_node_stable_projection(recovery_source.get("runtime"))
           != cross_node_stable_projection(
               release_post.get("file_bindings", {}).get(
                   "methods/bernini_action_editing/recover_v4g_scientific_no_go_attestation_v1.py"))
        or cross_node_stable_projection(recovery_source.get("tests"))
           != cross_node_stable_projection(
               release_post.get("file_bindings", {}).get(
                   "methods/bernini_action_editing/tests/test_recover_v4g_scientific_no_go_attestation_v1.py"))
        or cross_node_stable_projection(recovery_source.get("manifest"))
           != cross_node_stable_projection(
               release_post.get("file_bindings", {}).get(
                   "release-manifest-v4g-recovery.json"))
        or cross_node_stable_projection(recovery_source.get("controller"))
           != cross_node_stable_projection(post.get("controller"))
        or recovery_source.get("exact_file_count") != 3
        or recovery_source.get("exact_directory_count_below_root") != 3
        or recovery_source.get("one_way_sha256_dag_reverified") is not True
        or recovery_source.get("controller_identity_recorded_not_runtime_reverse_pinned") is not True):
    raise SystemExit("attestation recovery authority join differs")
parser_torch = source.get("recovery_parser_torch")
site_packages = "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages"
if (type(parser_torch) is not dict or set(parser_torch) != {
        "torch_version", "torch_hip_version",
        "torch_version_exact_2_7_1_rocm6_3", "torch_hip_release_6_3",
        "torch_package_root", "torch_module_origins",
        "v4g_torch_module_identities_exact",
        "torch_standard_source_loaders_and_package_path_exact"}
        or parser_torch.get("torch_version") != torch_version
        or parser_torch.get("torch_hip_version") != torch_hip_version
        or parser_torch.get("torch_version_exact_2_7_1_rocm6_3") is not True
        or parser_torch.get("torch_hip_release_6_3") is not True
        or parser_torch.get("torch_package_root") != site_packages + "/torch"
        or parser_torch.get("torch_module_origins") != {
            "torch": site_packages + "/torch/__init__.py",
            "torch.nn": site_packages + "/torch/nn/__init__.py",
            "torch.nn.functional": site_packages + "/torch/nn/functional.py",
        }
        or parser_torch.get("v4g_torch_module_identities_exact") is not True
        or parser_torch.get(
            "torch_standard_source_loaders_and_package_path_exact") is not True):
    raise SystemExit("attestation Torch/HIP join differs")

# Reuse the sealed recovery runtime's audited validators against live
# authorities.  This closes every nested attestation claim rather than merely
# trusting the child-produced summary.
runtime_module._validate_publish_value(recovery, attestation)
live_rows, live_bindings = runtime_module._scan_original(original)
live_original_root = runtime_module._read_directory_binding(original, mode=0o700)
live_original_dirs = [
    runtime_module._read_directory_binding(original / relative, mode=0o700)
    for relative in sorted(runtime_module.EXPECTED_DIRS)
]
if ([burned_row_stable_projection(row) for row in live_rows]
        != [burned_row_stable_projection(row) for row in rows]
        or cross_node_directory_projection(live_original_root)
           != cross_node_directory_projection(
               attestation["original_run_root_binding"])
        or live_rows != post_original.get("files")
        or live_original_root != post_original.get("root_binding")
        or live_original_dirs != post_original.get("directory_bindings")
        or live_bindings != post_original.get("file_bindings")):
    raise SystemExit("full live original exact26/root+6dirs join differs")

live_historical_release = runtime_module._verify_release(runtime_module.RELEASE_ROOT)
live_recovery_release = runtime_module._verify_recovery_release_and_controller()
if (historical_release_stable_projection(live_historical_release)
        != historical_release_stable_projection(source.get("release"))
        or recovery_release_stable_projection(live_recovery_release)
           != recovery_release_stable_projection(recovery_source)
        or cross_node_directory_projection(
               live_recovery_release.get("release_root_binding"))
           != cross_node_directory_projection(release_post.get("root_binding"))
        or live_recovery_release.get("tree_rows") != release_post.get("tree_rows")
        or live_recovery_release.get("release_tree_sha256")
           != release_post.get("release_tree_sha256")):
    raise SystemExit("live historical/recovery release join differs")

historical_controller, _ = runtime_module._read_regular(
    runtime_module.CONTROLLER_PATH, mode=0o555, nlink=1,
)
designated_python, _ = runtime_module._read_regular(
    runtime_module.PYTHON_PATH, mode=0o755, nlink=1,
)
if (cross_node_stable_projection(historical_controller)
        != cross_node_stable_projection(source.get("controller"))
        or cross_node_stable_projection(designated_python)
           != cross_node_stable_projection(source.get("python"))
        or cross_node_stable_projection(designated_python)
           != cross_node_stable_projection(source.get("process_python"))
        or designated_python != post.get("python")
        or cross_node_stable_projection(live_recovery_release.get("controller"))
           != cross_node_stable_projection(post.get("controller"))):
    raise SystemExit("live historical/recovery controller/Python join differs")

live_input_receipts = []
live_input_raw = []
for path, expected_sha in runtime_module.AUTHORITY_FILES:
    binding, captured = runtime_module._read_regular(
        path, mode=0o444, nlink=1, capture=True,
    )
    if binding.get("sha256") != expected_sha or captured is None:
        raise SystemExit("live input receipt pin differs")
    live_input_receipts.append(binding); live_input_raw.append((binding, captured))
live_input_snapshot = runtime_module._verify_input_snapshot(live_input_raw)
if ([cross_node_stable_projection(binding)
     for binding in live_input_receipts]
        != [cross_node_stable_projection(binding)
            for binding in source.get("input_receipts", [])]
        or input_snapshot_stable_projection(live_input_snapshot)
           != input_snapshot_stable_projection(source.get("input_snapshot"))):
    raise SystemExit("live exact4 receipts/exact6 feature input snapshot join differs")

live_outer_stdout, _ = runtime_module._read_regular(
    runtime_module.OUTER_STDOUT_PATH, mode=0o600, nlink=1,
)
live_outer_stderr, _ = runtime_module._read_regular(
    runtime_module.OUTER_STDERR_PATH, mode=0o600, nlink=1,
)
recorded_logs = attestation.get("original_controller_logs")
if (type(recorded_logs) is not dict or set(recorded_logs) != {"stdout", "stderr"}
        or any(cross_node_stable_projection(live)
               != cross_node_stable_projection(recorded_logs[label])
               for label, live in (
                   ("stdout", live_outer_stdout), ("stderr", live_outer_stderr)))):
    raise SystemExit("live original controller log join differs")
live_launch_plan = runtime_module._verify_launch_plan(original, live_bindings)
if (cross_node_stable_projection(
        attestation.get("launch_plan", {}).get("binding"))
        != cross_node_stable_projection(live_bindings["launch-plan.json"])
        or attestation.get("launch_plan", {}).get("schema_version")
           != live_launch_plan.get("schema_version")):
    raise SystemExit("live launch-plan join differs")
live_accounting = runtime_module._query_failed_step()
recorded_accounting = attestation.get("failed_seal_child_accounting")
if (type(recorded_accounting) is not dict
        or set(recorded_accounting) != {
            "record", "sacct_executable", "query_columns", "exact_row_replayed",
        }
        or any(live_accounting.get(key) != recorded_accounting.get(key) for key in
               ("record", "query_columns", "exact_row_replayed"))
        or cross_node_stable_projection(live_accounting.get("sacct_executable"))
           != cross_node_stable_projection(
               recorded_accounting.get("sacct_executable"))):
    raise SystemExit("live failed seal-child sacct exact9 join differs")

live_row_by_path = {row["path"]: row for row in live_rows}
recorded_row_by_path = {row["path"]: row for row in rows}
expected_counts = ((400, 113, 131), (402, 115, 127), (401, 115, 128),
                   (403, 112, 129), (403, 112, 129))
for fold, summary in enumerate(folds):
    inner_row = live_row_by_path[f"fold{fold}/inner.json"]
    recorded_inner_row = recorded_row_by_path[f"fold{fold}/inner.json"]
    if (burned_row_stable_projection(inner_row)
            != burned_row_stable_projection(recorded_inner_row)
            or summary["inner_receipt_sha256"] != inner_row["sha256"]
            or summary["inner_receipt_sha256"]
               != runtime_module.INNER_RECEIPT_SHA256[fold]
            or summary["inner_receipt_digest"]
               != runtime_module.INNER_RECEIPT_DIGEST[fold]
            or (summary["model_fit_count"], summary["inner_count"],
                summary["oof_count"]) != expected_counts[fold]):
        raise SystemExit("live fold inner SHA/digest/count join differs")
    for filename, field in (("preselection.pt", "preselection_checkpoint"),
                            ("fixed1200.pt", "fixed1200_checkpoint")):
        burned = live_row_by_path[f"fold{fold}/{filename}"]
        recorded_burned = recorded_row_by_path[f"fold{fold}/{filename}"]
        checkpoint = summary[field]
        if (burned_row_stable_projection(burned)
                != burned_row_stable_projection(recorded_burned)
                or any(checkpoint.get(key) != recorded_burned[key] for key in
                       ("sha256", "size_bytes", "device", "inode"))
                or burned.get("mode_octal") != "0444"
                or burned.get("nlink") != 1):
            raise SystemExit("live fold checkpoint/burned-row join differs")

attestation["receipt_digest"] = stored
value = {
    "schema_version": "v4g-recovery-controller-attestation-validation-v1",
    "runtime_result": result,
    "recovery_root_binding": live_root_binding,
    "attestation_file_binding": attestation_file_binding,
    "attestation_receipt_digest": stored,
    "attestation_exact_key_count": len(attestation),
    "canonical_exact_keys": True,
    "scientific_claims_exact": True,
    "live_original_exact26_root_and_six_dirs_join": True,
    "fold_inner_checkpoint_full_live_join": True,
    "launch_logs_sacct_full_live_join": True,
    "historical_release_controller_python_input_full_live_join": True,
    "source_authority_full_live_join": True,
    "bounded_visibility": visibility,
    "validated_intermediate_bindings": validated_intermediate_bindings,
}
print(canonical(value).decode("ascii"))
PY
then
  sed -n '1,160p' "${tmp_root}/recovery-attestation.validation.stderr" >&2
  fail "recovery attestation postflight failed"
fi
[[ ! -s ${tmp_root}/recovery-attestation.validation.stderr \
   && $(wc -l <"${tmp_root}/recovery-attestation.validation.json") -eq 1 ]] || \
  fail "recovery attestation postflight output differs"

authority_snapshot >"${tmp_root}/authority.final-prepublication.json" || \
  fail "final prepublication authority snapshot failed"
[[ $(wc -l <"${tmp_root}/authority.final-prepublication.json") -eq 1 ]] || \
  fail "final prepublication authority output differs"
cmp -s "${tmp_root}/authority.preflight.json" \
       "${tmp_root}/authority.final-prepublication.json" || \
  fail "initial/final-prepublication authority snapshot differs"

intermediate_ledger_sha="$(run_pinned_python -I -S -B - \
  "${tmp_root}" "${controller_pid}" "${tmp_fd}" "${tmp_identity}" \
  authority.final-prepublication.json authority.preflight.json child.stderr \
  child.stdout compile-ast.json compile-ast.stderr \
  recovery-attestation.validation.json \
  recovery-attestation.validation.stderr recovery-visibility.json \
  recovery-visibility.stderr tests.normal.json tests.normal.stderr \
  tests.optimized.json tests.optimized.stderr \
  <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys
root, parent_pid, descriptor, expected_identity = (
    Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
)
names = sys.argv[5:]
if names != sorted(set(names)):
    raise SystemExit("intermediate expected-name ledger is not sorted exact")
held = os.open(
    f"/proc/{parent_pid}/fd/{descriptor}",
    os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY,
)
def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, stat.S_IMODE(info.st_mode),
            info.st_nlink, info.st_mtime_ns, info.st_ctime_ns)
try:
    root_info, held_info = root.lstat(), os.fstat(held)
    if (root.is_symlink() or root != root.resolve(strict=True)
            or f"{held_info.st_dev}:{held_info.st_ino}" != expected_identity
            or (root_info.st_dev, root_info.st_ino)
               != (held_info.st_dev, held_info.st_ino)
            or stat.S_IMODE(held_info.st_mode) != 0o700
            or sorted(os.listdir(held)) != names):
        raise SystemExit("intermediate root held-FD/exact membership differs")
    rows = []
    for name in names:
        before = os.stat(name, dir_fd=held, follow_symlinks=False)
        fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=held)
        digest = hashlib.sha256()
        try:
            opened = os.fstat(fd)
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk: break
                digest.update(chunk)
            os.fchmod(fd, 0o444); os.fsync(fd)
            sealed = os.fstat(fd)
        finally: os.close(fd)
        after = os.stat(name, dir_fd=held, follow_symlinks=False)
        if (not stat.S_ISREG(before.st_mode)
                or stat.S_IMODE(before.st_mode) != 0o600 or before.st_nlink != 1
                or identity(sealed) != identity(after)
                or stat.S_IMODE(sealed.st_mode) != 0o444 or sealed.st_nlink != 1
                or (before.st_dev, before.st_ino, before.st_size)
                   != (sealed.st_dev, sealed.st_ino, sealed.st_size)):
            raise SystemExit("intermediate member capture/seal differs")
        rows.append({
            "name": name, "sha256": digest.hexdigest(),
            "size_bytes": sealed.st_size, "mode_octal": "0444",
            "nlink": sealed.st_nlink, "device": sealed.st_dev,
            "inode": sealed.st_ino, "mtime_ns": sealed.st_mtime_ns,
            "ctime_ns": sealed.st_ctime_ns,
            "single_fd_pre_post_identity_and_sha_exact": True,
        })
    ledger = {
        "schema_version": "v4g-recovery-controller-intermediate-ledger-v1",
        "root_identity": expected_identity, "members": rows,
    }
    raw = json.dumps(
        ledger, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    ledger_fd = os.open(
        "intermediate-ledger.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600, dir_fd=held,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(ledger_fd, view)
            if written <= 0: raise RuntimeError("intermediate ledger write stalled")
            view = view[written:]
        os.fsync(ledger_fd); os.fchmod(ledger_fd, 0o444); os.fsync(ledger_fd)
        ledger_info = os.fstat(ledger_fd)
    finally: os.close(ledger_fd)
    if (stat.S_IMODE(ledger_info.st_mode) != 0o444 or ledger_info.st_nlink != 1
            or sorted(os.listdir(held))
               != sorted([*names, "intermediate-ledger.json"])):
        raise SystemExit("intermediate ledger seal/exact membership differs")
    os.fchmod(held, 0o555); os.fsync(held)
    print(hashlib.sha256(raw).hexdigest())
finally:
    os.close(held)
PY
)" || fail "intermediate exact-ledger seal failed"
[[ ${intermediate_ledger_sha} =~ ^[0-9a-f]{64}$ ]] || \
  fail "intermediate ledger SHA output differs"

trap - EXIT
exec {tmp_fd}<&-
exec -a "${python_bin}" "${parent_python_image_path}" -I -S -B - \
    "${execution_root}" "${execution_name}" \
    "${tmp_root}" "${tmp_identity}" "${intermediate_ledger_sha}" \
    "${controller_path}" "${expected_controller_sha256}" \
    "${shell_bin}" "${expected_shell_sha256}" "${python_bin}" \
    "${expected_python_sha256}" \
    "${srun_bin}" "${expected_srun_sha256}" "${job_id}" "${node}" \
    "${release_root}" "${original_root}" "${recovery_root}" \
    "${expected_test_count}" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys, types

(execution_raw, receipt_name, tmp_raw, tmp_identity, ledger_sha,
 controller_raw, controller_sha, shell_raw, shell_sha, python_raw, python_sha,
 srun_raw, srun_sha, job_id, node, release_raw, original_raw, recovery_raw,
 test_count_raw) = sys.argv[1:]
execution = Path(execution_raw)
parent = execution.parent
tmp_root = Path(tmp_raw)
controller, shell, python, srun = map(
    Path, (controller_raw, shell_raw, python_raw, srun_raw),
)
release, original, recovery_root = map(
    Path, (release_raw, original_raw, recovery_raw),
)
test_count = int(test_count_raw)
trusted_anchor = Path("/vast/users")
expected_intermediates = [
    "authority.final-prepublication.json",
    "authority.preflight.json",
    "child.stderr",
    "child.stdout",
    "compile-ast.json",
    "compile-ast.stderr",
    "recovery-attestation.validation.json",
    "recovery-attestation.validation.stderr",
    "recovery-visibility.json",
    "recovery-visibility.stderr",
    "tests.normal.json",
    "tests.normal.stderr",
    "tests.optimized.json",
    "tests.optimized.stderr",
]

def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")

def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def strict_json(raw):
    def hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError("duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(raw.decode("ascii"), object_pairs_hook=hook)
    if canonical(value) + b"\n" != raw:
        raise RuntimeError("noncanonical intermediate/receipt JSON")
    return value

def identity(info):
    return (
        info.st_dev, info.st_ino, info.st_size, stat.S_IMODE(info.st_mode),
        info.st_nlink, info.st_mtime_ns, info.st_ctime_ns,
    )

def regular_binding(path, mode):
    if (
        not path.is_absolute() or path.is_symlink()
        or path != path.resolve(strict=True)
    ):
        raise RuntimeError("regular authority path differs")
    before = path.lstat()
    descriptor = os.open(
        path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        closed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (
        len({identity(value) for value in (before, opened, closed, after)}) != 1
        or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != mode or before.st_nlink != 1
    ):
        raise RuntimeError("regular authority same-FD binding differs")
    return {
        "path": str(path), "sha256": digest.hexdigest(),
        "size_bytes": before.st_size, "mode_octal": f"{mode:04o}",
        "nlink": before.st_nlink, "device": before.st_dev,
        "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "single_fd_pre_post_identity_and_sha_exact": True,
    }

def captured_regular_binding(path, mode):
    if (
        not path.is_absolute() or path.is_symlink()
        or path != path.resolve(strict=True)
    ):
        raise RuntimeError("captured regular authority path differs")
    before = path.lstat()
    descriptor = os.open(
        path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    digest = hashlib.sha256()
    chunks = []
    try:
        opened = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        closed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (
        len({identity(value) for value in (before, opened, closed, after)}) != 1
        or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != mode or before.st_nlink != 1
    ):
        raise RuntimeError("captured regular same-FD binding differs")
    return ({
        "path": str(path), "sha256": digest.hexdigest(),
        "size_bytes": before.st_size, "mode_octal": f"{mode:04o}",
        "nlink": before.st_nlink, "device": before.st_dev,
        "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "single_fd_pre_post_identity_and_sha_exact": True,
    }, b"".join(chunks))

def current_process_image_binding(logical_path, expected):
    if (Path(sys.executable).resolve(strict=True) != logical_path
            or Path("/proc/self/exe").resolve(strict=True) != logical_path):
        raise RuntimeError("publisher process Python path differs")
    flags = sys.flags
    if (tuple(sys.version_info[:3]) != (3, 12, 13)
            or flags.isolated != 1 or flags.no_site != 1
            or flags.ignore_environment != 1 or flags.safe_path is not True
            or flags.dont_write_bytecode != 1 or flags.optimize != 0):
        raise RuntimeError("publisher process Python version/flags differ")
    descriptor = os.open("/proc/self/exe", os.O_RDONLY | os.O_CLOEXEC)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    actual = {
        "path": str(logical_path), "sha256": digest.hexdigest(),
        "size_bytes": before.st_size,
        "mode_octal": f"{stat.S_IMODE(before.st_mode):04o}",
        "nlink": before.st_nlink, "device": before.st_dev,
        "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "single_fd_pre_post_identity_and_sha_exact": True,
    }
    if identity(before) != identity(after) or actual != expected:
        raise RuntimeError("publisher process Python image binding differs")
    return actual

def cross_node_stable_projection(binding):
    if (type(binding) is not dict
            or binding.get("single_fd_pre_post_identity_and_sha_exact") is not True):
        raise RuntimeError("cross-node binding lacks local same-FD evidence")
    return {key: binding.get(key) for key in (
        "path", "sha256", "size_bytes", "mode_octal", "nlink",
    )}

def cross_node_directory_projection(binding):
    if (type(binding) is not dict
            or binding.get(
                "single_fd_pre_post_identity_and_membership_exact") is not True):
        raise RuntimeError("cross-node directory lacks local same-FD evidence")
    return {key: binding.get(key) for key in ("path", "mode_octal", "members")}

def burned_row_stable_projection(row):
    if type(row) is not dict:
        raise RuntimeError("burned row stable projection schema differs")
    return {key: row.get(key) for key in (
        "path", "sha256", "size_bytes", "mode_octal", "nlink",
    )}

def historical_release_stable_projection(value):
    if type(value) is not dict or set(value) != {
        "root", "tree_sha256", "manifest", "manifest_digest",
        "file_count", "directory_count",
    }:
        raise RuntimeError("historical release projection schema differs")
    return {
        **{key: value[key] for key in (
            "root", "tree_sha256", "manifest_digest", "file_count",
            "directory_count",
        )},
        "manifest": cross_node_stable_projection(value["manifest"]),
    }

def recovery_release_stable_projection(value):
    if type(value) is not dict or set(value) != {
        "release_root", "release_root_binding", "release_tree_sha256",
        "tree_rows", "manifest", "manifest_digest", "runtime", "tests",
        "controller", "exact_file_count", "exact_directory_count_below_root",
        "one_way_sha256_dag_reverified",
        "controller_identity_recorded_not_runtime_reverse_pinned",
    }:
        raise RuntimeError("recovery release projection schema differs")
    return {
        **{key: value[key] for key in (
            "release_root", "release_tree_sha256", "tree_rows",
            "manifest_digest", "exact_file_count",
            "exact_directory_count_below_root", "one_way_sha256_dag_reverified",
            "controller_identity_recorded_not_runtime_reverse_pinned",
        )},
        "release_root_binding": cross_node_directory_projection(
            value["release_root_binding"]),
        **{key: cross_node_stable_projection(value[key]) for key in (
            "manifest", "runtime", "tests", "controller",
        )},
    }

def input_snapshot_stable_projection(value):
    if type(value) is not dict or set(value) != {
        "sha256", "ordered_rows", "feature_shard_bindings",
        "exact_receipt_count", "exact_feature_shard_count",
        "all_ten_files_single_fd_reverified",
    }:
        raise RuntimeError("input snapshot projection schema differs")
    return {
        **{key: value[key] for key in (
            "sha256", "ordered_rows", "exact_receipt_count",
            "exact_feature_shard_count", "all_ten_files_single_fd_reverified",
        )},
        "feature_shard_bindings": [
            cross_node_stable_projection(binding)
            for binding in value["feature_shard_bindings"]
        ],
    }

def directory_binding(path, mode):
    if (
        not path.is_absolute() or path.is_symlink()
        or path != path.resolve(strict=True)
    ):
        raise RuntimeError("directory authority path differs")
    before = path.lstat()
    descriptor = os.open(
        path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        opened = os.fstat(descriptor)
        members = sorted(os.listdir(descriptor))
        closed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (
        len({identity(value) for value in (before, opened, closed, after)}) != 1
        or not stat.S_ISDIR(before.st_mode)
        or stat.S_IMODE(before.st_mode) != mode
    ):
        raise RuntimeError("directory authority same-FD binding differs")
    return {
        "path": str(path), "mode_octal": f"{mode:04o}",
        "nlink": before.st_nlink, "device": before.st_dev,
        "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns, "members": members,
        "single_fd_pre_post_identity_and_membership_exact": True,
    }

def member_binding(info, name, digest):
    return {
        "name": name, "sha256": digest,
        "size_bytes": info.st_size, "mode_octal": "0444",
        "nlink": info.st_nlink, "device": info.st_dev,
        "inode": info.st_ino, "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "single_fd_pre_post_identity_and_sha_exact": True,
    }

def read_member(directory_fd, name, expected=None):
    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    descriptor = os.open(
        name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    digest = hashlib.sha256()
    chunks = []
    try:
        opened = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        closed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (
        len({identity(value) for value in (before, opened, closed, after)}) != 1
        or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444 or before.st_nlink != 1
    ):
        raise RuntimeError("intermediate same-FD seal differs")
    row = member_binding(before, name, digest.hexdigest())
    if expected is not None and row != expected:
        raise RuntimeError("intermediate ledger/member binding differs")
    return row, b"".join(chunks)

def capture_intermediates():
    if (
        not tmp_root.is_absolute() or tmp_root.is_symlink()
        or tmp_root != tmp_root.resolve(strict=True)
    ):
        raise RuntimeError("intermediate lexical root differs")
    before = tmp_root.lstat()
    root_fd = os.open(
        tmp_root,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    retain_root_fd = False
    try:
        opened = os.fstat(root_fd)
        if (
            identity(before) != identity(opened)
            or f"{opened.st_dev}:{opened.st_ino}" != tmp_identity
            or not stat.S_ISDIR(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o555
        ):
            raise RuntimeError("intermediate root held-FD identity differs")
        ledger_binding, ledger_raw = read_member(
            root_fd, "intermediate-ledger.json",
        )
        if ledger_binding["sha256"] != ledger_sha:
            raise RuntimeError("intermediate ledger SHA differs")
        ledger = strict_json(ledger_raw)
        if (
            set(ledger) != {"schema_version", "root_identity", "members"}
            or ledger.get("schema_version")
               != "v4g-recovery-controller-intermediate-ledger-v1"
            or ledger.get("root_identity") != tmp_identity
            or type(ledger.get("members")) is not list
            or [row.get("name") for row in ledger["members"]]
               != expected_intermediates
            or sorted(os.listdir(root_fd))
               != sorted([*expected_intermediates, "intermediate-ledger.json"])
        ):
            raise RuntimeError("intermediate exact ledger differs")
        values = {}
        for expected in ledger["members"]:
            if type(expected) is not dict:
                raise RuntimeError("intermediate ledger row differs")
            row, raw = read_member(root_fd, expected.get("name"), expected)
            name = row["name"]
            if name.endswith(".stderr"):
                if raw != b"":
                    raise RuntimeError("sealed intermediate stderr differs")
            else:
                values[name] = strict_json(raw)
        after = tmp_root.lstat()
        closed = os.fstat(root_fd)
        if (
            len({identity(value) for value in (before, opened, closed, after)})
            != 1
            or sorted(os.listdir(root_fd))
               != sorted([*expected_intermediates, "intermediate-ledger.json"])
        ):
            raise RuntimeError("intermediate root changed during capture")
        evidence = {
            "schema_version": ledger["schema_version"],
            "temporary_root_path": str(tmp_root),
            "temporary_root_identity": tmp_identity,
            "ledger_sha256": ledger_sha,
            "ledger_binding": {
                "path": str(tmp_root / "intermediate-ledger.json"),
                **{key: value for key, value in ledger_binding.items()
                   if key != "name"},
            },
            "members": ledger["members"],
            "same_held_directory_fd_capture_exact": True,
        }
        retain_root_fd = True
        return values, evidence, root_fd
    finally:
        if not retain_root_fd:
            os.close(root_fd)

def retain_captured_intermediates(root_fd, evidence):
    opened = os.fstat(root_fd)
    try:
        lexical = tmp_root.lstat()
    except FileNotFoundError as error:
        raise RuntimeError("owned intermediate lexical root is absent") from error
    if (
        tmp_root.is_symlink() or tmp_root != tmp_root.resolve(strict=True)
        or (lexical.st_dev, lexical.st_ino) != (opened.st_dev, opened.st_ino)
        or f"{opened.st_dev}:{opened.st_ino}" != tmp_identity
        or stat.S_IMODE(opened.st_mode) != 0o555
        or sorted(os.listdir(root_fd))
           != sorted([*expected_intermediates, "intermediate-ledger.json"])
    ):
        raise RuntimeError("owned intermediate root cleanup binding differs")
    ledger_expected = evidence["ledger_binding"]
    ledger_row, _ = read_member(root_fd, "intermediate-ledger.json")
    if {
        "path": str(tmp_root / "intermediate-ledger.json"),
        **{key: value for key, value in ledger_row.items() if key != "name"},
    } != ledger_expected or ledger_expected.get("sha256") != ledger_sha:
        raise RuntimeError("owned intermediate cleanup ledger binding differs")
    for expected in evidence["members"]:
        read_member(root_fd, expected["name"], expected)
    # Explicit retention avoids an unavoidable same-UID rename race between a
    # final path/FD comparison and rmdir(2) in the shared /tmp namespace.  The
    # exact ledger, every member, and the root stay read-only for forensics.
    evidence["temporary_root_retained_read_only_for_forensics"] = True
    evidence["temporary_cleanup_mutation_performed"] = False
    os.close(root_fd)

def rejoin_authorities(snapshot, recovery_summary):
    if set(snapshot) != {
        "schema_version", "controller", "controller_executed_source",
        "controller_shell", "python", "parent_python_held_image",
        "authority_python_process_image", "python_environment_trust_boundary",
        "srun", "recovery_release", "original_run",
    } or snapshot.get("schema_version") != (
        "v4g-recovery-controller-authority-snapshot-v1"
    ):
        raise RuntimeError("authority snapshot schema differs")
    shell_value = snapshot["controller_shell"]
    if (
        set(shell_value) != {
            "binding", "process_image_binding", "bash_version",
            "privileged_mode", "startup_shell_flags",
            "startup_shopt_profile_exact",
            "startup_bashopts_observed_after_exact_launcher_gate",
            "declared_formal_launcher",
            "observed_process_image_environment_and_argv_contract_exact",
            "literal_parent_env_utility_invocation_observable",
            "startup_exported_names", "startup_exported_values",
            "startup_environment_exact_clean5",
            "dangerous_startup_variables_absent",
        }
        or shell_value.get("privileged_mode") is not True
        or shell_value.get("startup_shell_flags") != "hpB"
        or shell_value.get("startup_shopt_profile_exact") is not True
        or not isinstance(shell_value.get(
            "startup_bashopts_observed_after_exact_launcher_gate"
        ), str)
        or not shell_value[
            "startup_bashopts_observed_after_exact_launcher_gate"
        ]
        or shell_value.get(
            "observed_process_image_environment_and_argv_contract_exact"
        ) is not True
        or shell_value.get(
            "literal_parent_env_utility_invocation_observable"
        ) is not False
        or shell_value.get("startup_environment_exact_clean5") is not True
        or shell_value.get("dangerous_startup_variables_absent") is not True
        or shell_value.get("declared_formal_launcher") != (
            "env -i PATH=/usr/bin:/bin LC_ALL=C HOME=/nonexistent "
            "/bin/bash -p CONTROLLER EXPECTED_CONTROLLER_SHA256"
        )
        or shell_value.get("startup_exported_names")
           != ["HOME", "LC_ALL", "PATH", "PWD", "SHLVL"]
        or set(shell_value.get("startup_exported_values", {}))
           != {"HOME", "LC_ALL", "PATH", "PWD", "SHLVL"}
        or shell_value["startup_exported_values"].get("HOME") != "/nonexistent"
        or shell_value["startup_exported_values"].get("LC_ALL") != "C"
        or shell_value["startup_exported_values"].get("PATH") != "/usr/bin:/bin"
        or shell_value["startup_exported_values"].get("SHLVL") != "1"
        or not shell_value["startup_exported_values"].get("PWD", "").startswith("/")
    ):
        raise RuntimeError("controller shell/clean environment evidence differs")
    if (
        snapshot["controller_executed_source"] != snapshot["controller"]
        or shell_value["process_image_binding"] != shell_value["binding"]
        or snapshot["parent_python_held_image"] != snapshot["python"]
        or snapshot["authority_python_process_image"] != snapshot["python"]
        or snapshot["python_environment_trust_boundary"] != {
            "pinned_python_executable_and_process_image": True,
            "python_3_12_13_isolated_no_site_safe_path": True,
            "torch_2_7_1_rocm6_3_and_hip_6_3_canonical_origins_required": True,
            "python_environment_full_tree_cryptographically_pinned": False,
            "procedural_trusted_conda_environment_required": True,
        }
    ):
        raise RuntimeError("executed image/source or Python trust boundary differs")
    for expected, mode in (
        (snapshot["controller"], 0o555),
        (shell_value["binding"], 0o755),
        (snapshot["python"], 0o755),
        (snapshot["srun"], 0o755),
    ):
        if regular_binding(Path(expected["path"]), mode) != expected:
            raise RuntimeError("executable authority rejoin differs")
    release_value = snapshot["recovery_release"]
    if (
        Path(release_value["root_binding"]["path"]) != release
        or len(release_value.get("directory_bindings", [])) != 3
        or len(release_value.get("file_bindings", {})) != 3
    ):
        raise RuntimeError("recovery release snapshot cardinality differs")
    if directory_binding(release, 0o555) != release_value["root_binding"]:
        raise RuntimeError("recovery release root rejoin differs")
    for expected in release_value["directory_bindings"]:
        if directory_binding(Path(expected["path"]), 0o555) != expected:
            raise RuntimeError("recovery release directory rejoin differs")
    for expected in release_value["file_bindings"].values():
        if regular_binding(Path(expected["path"]), 0o444) != expected:
            raise RuntimeError("recovery release file rejoin differs")
    original_value = snapshot["original_run"]
    if (
        Path(original_value["root_binding"]["path"]) != original
        or len(original_value.get("directory_bindings", [])) != 6
        or len(original_value.get("file_bindings", {})) != 26
        or len(original_value.get("files", [])) != 26
    ):
        raise RuntimeError("original authority snapshot cardinality differs")
    if directory_binding(original, 0o700) != original_value["root_binding"]:
        raise RuntimeError("original root rejoin differs")
    for expected in original_value["directory_bindings"]:
        if directory_binding(Path(expected["path"]), 0o700) != expected:
            raise RuntimeError("original directory rejoin differs")
    for expected in original_value["file_bindings"].values():
        if regular_binding(Path(expected["path"]), 0o444) != expected:
            raise RuntimeError("original file rejoin differs")
    root_expected = recovery_summary["recovery_root_binding"]
    file_expected = recovery_summary["attestation_file_binding"]
    if (
        Path(root_expected["path"]) != recovery_root
        or Path(file_expected["path"])
           != recovery_root / "recovery-attestation.json"
        or directory_binding(recovery_root, 0o555) != root_expected
        or regular_binding(
            recovery_root / "recovery-attestation.json", 0o444,
        ) != file_expected
        or recovery_summary["bounded_visibility"].get("recovery") != {
            "root_binding": root_expected, "file_binding": file_expected,
        }
    ):
        raise RuntimeError("live recovery root/file rejoin differs")
    rejoin_nested_live_authorities(snapshot, recovery_summary)
    return True

values, intermediate_evidence, intermediate_root_fd = capture_intermediates()
pre = values["authority.preflight.json"]
post = values["authority.final-prepublication.json"]
normal = values["tests.normal.json"]
optimized = values["tests.optimized.json"]
compile_ast = values["compile-ast.json"]
recovery_summary = values["recovery-attestation.validation.json"]
visibility = values["recovery-visibility.json"]
child = values["child.stdout"]
if pre != post:
    raise SystemExit("publisher initial/final authority snapshot differs")
if (
    post.get("controller", {}).get("path") != str(controller)
    or post.get("controller", {}).get("sha256") != controller_sha
    or post.get("controller_shell", {}).get("binding", {}).get("path")
       != str(shell)
    or post.get("controller_shell", {}).get("binding", {}).get("sha256")
       != shell_sha
    or post.get("python", {}).get("path") != str(python)
    or post.get("python", {}).get("sha256") != python_sha
    or post.get("srun", {}).get("path") != str(srun)
    or post.get("srun", {}).get("sha256") != srun_sha
):
    raise SystemExit("publisher pinned executable authority join differs")
publisher_process_image = current_process_image_binding(
    python, post["python"],
)
publisher_python_runtime = {
    "version": ".".join(map(str, sys.version_info[:3])),
    "isolated": sys.flags.isolated == 1,
    "no_site": sys.flags.no_site == 1,
    "ignore_environment": sys.flags.ignore_environment == 1,
    "safe_path": sys.flags.safe_path is True,
    "dont_write_bytecode": sys.flags.dont_write_bytecode == 1,
    "optimize": sys.flags.optimize,
}

test_keys = {
    "schema_version", "mode", "python_optimize", "tests_run",
    "tests_skipped", "failures", "errors", "expected_failures",
    "unexpected_successes", "successful", "transcript_sha256",
    "executed_runtime_binding", "executed_tests_binding",
    "captured_bytes_compiled_and_executed", "path_import_execution_used",
}
runtime_rel = (
    "methods/bernini_action_editing/"
    "recover_v4g_scientific_no_go_attestation_v1.py"
)
tests_rel = (
    "methods/bernini_action_editing/tests/"
    "test_recover_v4g_scientific_no_go_attestation_v1.py"
)
release_files = post["recovery_release"]["file_bindings"]
rejoin_runtime_path = Path(release_files[runtime_rel]["path"])
rejoin_runtime_binding, rejoin_runtime_bytes = captured_regular_binding(
    rejoin_runtime_path, 0o444,
)
if rejoin_runtime_binding != release_files[runtime_rel]:
    raise SystemExit("publisher captured recovery runtime binding differs")
rejoin_runtime_module = types.ModuleType(
    "_v4g_recovery_publisher_live_authority",
)
rejoin_runtime_module.__file__ = str(rejoin_runtime_path)
rejoin_runtime_module.__package__ = None
rejoin_runtime_module.__loader__ = None
rejoin_runtime_module.__spec__ = None
exec(compile(
    rejoin_runtime_bytes, str(rejoin_runtime_path), "exec",
    dont_inherit=True, optimize=sys.flags.optimize,
), rejoin_runtime_module.__dict__)
if (rejoin_runtime_module.RELEASE_SEALED is not True
        or rejoin_runtime_module.RESULT_SCHEMA
           != "v4g-scientific-no-go-sibling-recovery-result-v4"):
    raise SystemExit("publisher captured recovery runtime release gate differs")

def rejoin_nested_live_authorities(snapshot, summary):
    """Re-run every nested mutable authority at each publication boundary."""
    attestation_path = recovery_root / "recovery-attestation.json"
    attestation_binding, attestation_raw = captured_regular_binding(
        attestation_path, 0o444,
    )
    if attestation_binding != summary["attestation_file_binding"]:
        raise RuntimeError("nested live attestation binding differs")
    attestation = strict_json(attestation_raw)
    stored_digest = attestation.get("receipt_digest")
    unsigned_attestation = dict(attestation)
    unsigned_attestation.pop("receipt_digest", None)
    if (stored_digest != object_sha(unsigned_attestation)
            or stored_digest != summary["attestation_receipt_digest"]):
        raise RuntimeError("nested live attestation self-digest differs")
    rejoin_runtime_module._validate_publish_value(
        recovery_root, unsigned_attestation,
    )

    source = attestation["source_authority"]
    original_snapshot = snapshot["original_run"]
    live_rows, live_bindings = rejoin_runtime_module._scan_original(original)
    recorded_rows = attestation["burned_exact26_manifest"]
    live_original_root = rejoin_runtime_module._read_directory_binding(
        original, mode=0o700,
    )
    live_original_dirs = [
        rejoin_runtime_module._read_directory_binding(
            original / relative, mode=0o700,
        )
        for relative in sorted(rejoin_runtime_module.EXPECTED_DIRS)
    ]
    if (
        [burned_row_stable_projection(row) for row in live_rows]
        != [burned_row_stable_projection(row) for row in recorded_rows]
        or cross_node_directory_projection(live_original_root)
           != cross_node_directory_projection(
               attestation["original_run_root_binding"])
        or live_rows != original_snapshot["files"]
        or live_bindings != original_snapshot["file_bindings"]
        or live_original_root != original_snapshot["root_binding"]
        or live_original_dirs != original_snapshot["directory_bindings"]
    ):
        raise RuntimeError("nested live original exact26/root+6dirs differs")

    live_historical_release = rejoin_runtime_module._verify_release(
        rejoin_runtime_module.RELEASE_ROOT,
    )
    live_recovery_release = (
        rejoin_runtime_module._verify_recovery_release_and_controller()
    )
    if (
        historical_release_stable_projection(live_historical_release)
        != historical_release_stable_projection(source["release"])
        or recovery_release_stable_projection(live_recovery_release)
           != recovery_release_stable_projection(source["recovery"])
        or recovery_release_stable_projection(live_recovery_release)
           != recovery_release_stable_projection({
               "release_root": str(release),
               "release_root_binding": snapshot["recovery_release"][
                   "root_binding"],
               "release_tree_sha256": snapshot["recovery_release"][
                   "release_tree_sha256"],
               "tree_rows": snapshot["recovery_release"]["tree_rows"],
               "manifest": snapshot["recovery_release"]["file_bindings"][
                   "release-manifest-v4g-recovery.json"],
               "manifest_digest": snapshot["recovery_release"][
                   "manifest_digest"],
               "runtime": snapshot["recovery_release"]["file_bindings"][
                   runtime_rel],
               "tests": snapshot["recovery_release"]["file_bindings"][
                   tests_rel],
               "controller": snapshot["controller"],
               "exact_file_count": 3,
               "exact_directory_count_below_root": 3,
               "one_way_sha256_dag_reverified": True,
               "controller_identity_recorded_not_runtime_reverse_pinned": True,
           })
    ):
        raise RuntimeError("nested live historical/recovery release differs")

    historical_controller, _ = rejoin_runtime_module._read_regular(
        rejoin_runtime_module.CONTROLLER_PATH, mode=0o555, nlink=1,
    )
    designated_python, _ = rejoin_runtime_module._read_regular(
        rejoin_runtime_module.PYTHON_PATH, mode=0o755, nlink=1,
    )
    if (
        cross_node_stable_projection(historical_controller)
        != cross_node_stable_projection(source["controller"])
        or any(cross_node_stable_projection(designated_python)
               != cross_node_stable_projection(source[key])
               for key in ("python", "process_python"))
        or designated_python != snapshot["python"]
        or cross_node_stable_projection(live_recovery_release["controller"])
           != cross_node_stable_projection(snapshot["controller"])
    ):
        raise RuntimeError("nested live controller/Python authority differs")

    live_receipts = []
    live_receipt_raw = []
    for path, expected_sha in rejoin_runtime_module.AUTHORITY_FILES:
        binding, captured = rejoin_runtime_module._read_regular(
            path, mode=0o444, nlink=1, capture=True,
        )
        if binding["sha256"] != expected_sha or captured is None:
            raise RuntimeError("nested live input receipt pin differs")
        live_receipts.append(binding)
        live_receipt_raw.append((binding, captured))
    live_input_snapshot = rejoin_runtime_module._verify_input_snapshot(
        live_receipt_raw,
    )
    if (
        [cross_node_stable_projection(binding) for binding in live_receipts]
        != [cross_node_stable_projection(binding)
            for binding in source["input_receipts"]]
        or input_snapshot_stable_projection(live_input_snapshot)
           != input_snapshot_stable_projection(source["input_snapshot"])
    ):
        raise RuntimeError("nested live input authority differs")

    live_stdout, _ = rejoin_runtime_module._read_regular(
        rejoin_runtime_module.OUTER_STDOUT_PATH, mode=0o600, nlink=1,
    )
    live_stderr, _ = rejoin_runtime_module._read_regular(
        rejoin_runtime_module.OUTER_STDERR_PATH, mode=0o600, nlink=1,
    )
    recorded_logs = attestation["original_controller_logs"]
    if any(
        cross_node_stable_projection(live)
        != cross_node_stable_projection(recorded_logs[label])
        for label, live in (("stdout", live_stdout), ("stderr", live_stderr))
    ):
        raise RuntimeError("nested live original controller logs differ")
    live_launch = rejoin_runtime_module._verify_launch_plan(
        original, live_bindings,
    )
    if (
        cross_node_stable_projection(attestation["launch_plan"]["binding"])
        != cross_node_stable_projection(live_bindings["launch-plan.json"])
        or attestation["launch_plan"]["schema_version"]
           != live_launch["schema_version"]
    ):
        raise RuntimeError("nested live launch plan differs")

    live_accounting = rejoin_runtime_module._query_failed_step()
    recorded_accounting = attestation["failed_seal_child_accounting"]
    if (
        any(live_accounting[key] != recorded_accounting[key] for key in (
            "record", "query_columns", "exact_row_replayed",
        ))
        or cross_node_stable_projection(live_accounting["sacct_executable"])
           != cross_node_stable_projection(
               recorded_accounting["sacct_executable"])
    ):
        raise RuntimeError("nested live failed sacct exact9 differs")

    live_by_path = {row["path"]: row for row in live_rows}
    recorded_by_path = {row["path"]: row for row in recorded_rows}
    expected_counts = (
        (400, 113, 131), (402, 115, 127), (401, 115, 128),
        (403, 112, 129), (403, 112, 129),
    )
    for fold, fold_summary in enumerate(attestation["folds"]):
        live_inner = live_by_path[f"fold{fold}/inner.json"]
        recorded_inner = recorded_by_path[f"fold{fold}/inner.json"]
        if (
            burned_row_stable_projection(live_inner)
            != burned_row_stable_projection(recorded_inner)
            or fold_summary["inner_receipt_sha256"] != live_inner["sha256"]
            or fold_summary["inner_receipt_sha256"]
               != rejoin_runtime_module.INNER_RECEIPT_SHA256[fold]
            or fold_summary["inner_receipt_digest"]
               != rejoin_runtime_module.INNER_RECEIPT_DIGEST[fold]
            or (fold_summary["model_fit_count"], fold_summary["inner_count"],
                fold_summary["oof_count"]) != expected_counts[fold]
        ):
            raise RuntimeError("nested live fold inner join differs")
        for filename, field in (
            ("preselection.pt", "preselection_checkpoint"),
            ("fixed1200.pt", "fixed1200_checkpoint"),
        ):
            live_burned = live_by_path[f"fold{fold}/{filename}"]
            recorded_burned = recorded_by_path[f"fold{fold}/{filename}"]
            checkpoint = fold_summary[field]
            if (
                burned_row_stable_projection(live_burned)
                != burned_row_stable_projection(recorded_burned)
                or any(checkpoint[key] != recorded_burned[key] for key in (
                    "sha256", "size_bytes", "device", "inode",
                ))
                or live_burned["mode_octal"] != "0444"
                or live_burned["nlink"] != 1
            ):
                raise RuntimeError("nested live fold checkpoint join differs")
    return True

for value, label, optimize in (
    (normal, "normal", 0), (optimized, "optimized", 1),
):
    if (
        set(value) != test_keys
        or value.get("schema_version")
           != "v4g-recovery-controller-structured-unittest-v1"
        or value.get("mode") != label
        or value.get("python_optimize") != optimize
        or value.get("tests_run") != test_count
        or any(value.get(key) != 0 for key in (
            "tests_skipped", "failures", "errors", "expected_failures",
            "unexpected_successes",
        ))
        or value.get("successful") is not True
        or type(value.get("transcript_sha256")) is not str
        or len(value["transcript_sha256"]) != 64
        or value.get("executed_runtime_binding") != release_files[runtime_rel]
        or value.get("executed_tests_binding") != release_files[tests_rel]
        or value.get("captured_bytes_compiled_and_executed") is not True
        or value.get("path_import_execution_used") is not False
    ):
        raise SystemExit("publisher structured test result differs")

if (
    set(compile_ast) != {
        "schema_version", "source_count", "sources", "all_assert_nodes_absent",
    }
    or compile_ast.get("schema_version")
       != "v4g-recovery-controller-compile-ast-v1"
    or compile_ast.get("source_count") != 2
    or type(compile_ast.get("sources")) is not list
    or len(compile_ast["sources"]) != 2
    or any(
        set(row) != {
            "path", "sha256", "assert_node_count", "compiled_optimize_levels",
        }
        or row.get("assert_node_count") != 0
        or row.get("compiled_optimize_levels") != [0, 2]
        for row in compile_ast["sources"]
    )
    or compile_ast.get("all_assert_nodes_absent") is not True
    or {row["path"]: row["sha256"] for row in compile_ast["sources"]} != {
        release_files[runtime_rel]["path"]: release_files[runtime_rel]["sha256"],
        release_files[tests_rel]["path"]: release_files[tests_rel]["sha256"],
    }
):
    raise SystemExit("publisher compile/AST result differs")

recovery_keys = {
    "schema_version", "runtime_result", "recovery_root_binding",
    "attestation_file_binding", "attestation_receipt_digest",
    "attestation_exact_key_count", "canonical_exact_keys",
    "scientific_claims_exact",
    "live_original_exact26_root_and_six_dirs_join",
    "fold_inner_checkpoint_full_live_join",
    "launch_logs_sacct_full_live_join",
    "historical_release_controller_python_input_full_live_join",
    "source_authority_full_live_join", "bounded_visibility",
    "validated_intermediate_bindings",
}
result_keys = {
    "path", "file_sha256", "size_bytes", "receipt_digest", "mode_octal",
    "nlink", "root_mode_octal", "exact_file_count",
    "create_only_name_claim", "failure_tombstone_root_mode_octal",
    "original_run_and_source_authorities_reverified_after_name_claim",
    "root_and_file_same_fd_precommit_verified_and_parent_fsynced",
    "producer_root_precommit_binding", "producer_attestation_final_binding",
    "root_creation_to_precommit_device_inode_exact",
    "file_creation_to_final_device_inode_exact",
    "final_mode_commit", "final_mode_commit_order",
    "schema_version", "original_run_postverified_unchanged",
    "original_run_exact26_manifest_sha256", "scientific_result",
    "original_controller_complete",
}
if (
    set(recovery_summary) != recovery_keys
    or set(recovery_summary.get("runtime_result", {})) != result_keys
    or recovery_summary["runtime_result"].get("schema_version")
       != "v4g-scientific-no-go-sibling-recovery-result-v4"
    or recovery_summary["runtime_result"].get("create_only_name_claim") is not True
    or recovery_summary["runtime_result"].get(
        "failure_tombstone_root_mode_octal") != "0700"
    or recovery_summary["runtime_result"].get(
        "original_run_and_source_authorities_reverified_after_name_claim")
       is not True
    or recovery_summary["runtime_result"].get(
        "root_and_file_same_fd_precommit_verified_and_parent_fsynced")
       is not True
    or type(recovery_summary["runtime_result"].get(
        "producer_root_precommit_binding")) is not dict
    or set(recovery_summary["runtime_result"][
        "producer_root_precommit_binding"]) != {
            "path", "mode_octal", "nlink", "device", "inode", "mtime_ns",
            "ctime_ns", "members",
            "single_fd_pre_post_identity_and_membership_exact"}
    or recovery_summary["runtime_result"][
        "producer_root_precommit_binding"].get("path")
       != recovery_summary["recovery_root_binding"].get("path")
    or recovery_summary["runtime_result"][
        "producer_root_precommit_binding"].get("mode_octal") != "0700"
    or recovery_summary["runtime_result"][
        "producer_root_precommit_binding"].get("members")
       != ["recovery-attestation.json"]
    or recovery_summary["runtime_result"][
        "producer_root_precommit_binding"].get(
            "single_fd_pre_post_identity_and_membership_exact") is not True
    or type(recovery_summary["runtime_result"].get(
        "producer_attestation_final_binding")) is not dict
    or set(recovery_summary["runtime_result"][
        "producer_attestation_final_binding"]) != {
            "path", "sha256", "size_bytes", "mode_octal", "nlink", "device",
            "inode", "mtime_ns", "ctime_ns",
            "single_fd_pre_post_identity_and_sha_exact"}
    or cross_node_stable_projection(recovery_summary["runtime_result"][
        "producer_attestation_final_binding"])
       != cross_node_stable_projection(recovery_summary[
           "attestation_file_binding"])
    or recovery_summary["runtime_result"].get(
        "root_creation_to_precommit_device_inode_exact") is not True
    or recovery_summary["runtime_result"].get(
        "file_creation_to_final_device_inode_exact") is not True
    or any(type(recovery_summary["runtime_result"][
               "producer_root_precommit_binding"].get(key)) is not int
           for key in ("nlink", "device", "inode", "mtime_ns", "ctime_ns"))
    or any(type(recovery_summary["runtime_result"][
               "producer_attestation_final_binding"].get(key)) is not int
           for key in ("size_bytes", "nlink", "device", "inode",
                       "mtime_ns", "ctime_ns"))
    or recovery_summary["runtime_result"].get("final_mode_commit") is not True
    or recovery_summary["runtime_result"].get("final_mode_commit_order")
       != ["file_0444", "root_0555"]
    or recovery_summary["runtime_result"].get("mode_octal") != "0444"
    or recovery_summary["runtime_result"].get("root_mode_octal") != "0555"
    or recovery_summary["runtime_result"].get("nlink") != 1
    or recovery_summary["runtime_result"].get("exact_file_count") != 1
    or recovery_summary.get("attestation_exact_key_count") != 29
    or recovery_summary.get("bounded_visibility") != visibility
    or any(recovery_summary.get(key) is not True for key in (
        "canonical_exact_keys", "scientific_claims_exact",
        "live_original_exact26_root_and_six_dirs_join",
        "fold_inner_checkpoint_full_live_join",
        "launch_logs_sacct_full_live_join",
        "historical_release_controller_python_input_full_live_join",
        "source_authority_full_live_join",
    ))
):
    raise SystemExit("publisher full-live recovery validation differs")
validated_intermediates = recovery_summary["validated_intermediate_bindings"]
ledger_rows = {
    row["name"]: row for row in intermediate_evidence["members"]
}
if set(validated_intermediates) != {
    "child.stdout", "recovery-visibility.json", "authority.preflight.json",
}:
    raise SystemExit("validator intermediate binding schema differs")
for name, validated in validated_intermediates.items():
    sealed = ledger_rows[name]
    if (
        set(validated) != {
            "path", "sha256", "size_bytes", "mode_octal", "nlink",
            "device", "inode", "mtime_ns", "ctime_ns",
            "single_fd_pre_post_identity_and_sha_exact",
        }
        or validated.get("path") != str(tmp_root / name)
        or validated.get("mode_octal") != "0600" or validated.get("nlink") != 1
        or validated.get("single_fd_pre_post_identity_and_sha_exact") is not True
        or any(validated.get(key) != sealed.get(key) for key in (
            "sha256", "size_bytes", "nlink", "device", "inode", "mtime_ns",
        ))
    ):
        raise SystemExit("validator/ledger intermediate full join differs")

if set(child) != {
    "schema_version", "slurm", "python", "runtime_execution", "runtime_result",
}:
    raise SystemExit("publisher child schema differs")
slurm_observed = child["slurm"]
if (
    child.get("schema_version")
       != "v4g-recovery-controller-child-execution-v1"
    or set(slurm_observed) != {
        "job_id", "step_id", "step_num_nodes", "step_num_tasks", "nnodes",
        "ntasks", "cpus_per_task", "procid", "localid", "nodeid",
        "slurmd_nodename", "step_nodelist", "job_nodelist", "hostname",
        "cpu_affinity", "gres_request", "gpu_environment",
        "torch_visible_gpu_count", "exact_one_cpu_observed", "no_gpu_observed",
    }
    or child.get("runtime_result") != recovery_summary["runtime_result"]
    or set(child.get("runtime_execution", {})) != {
        "binding", "captured_bytes_compiled_and_executed",
        "path_import_or_runpy_used",
    }
    or cross_node_stable_projection(
           child.get("runtime_execution", {}).get("binding"))
       != cross_node_stable_projection(release_files[runtime_rel])
    or child.get("runtime_execution", {}).get(
        "captured_bytes_compiled_and_executed") is not True
    or child.get("runtime_execution", {}).get("path_import_or_runpy_used")
       is not False
    or slurm_observed.get("job_id") != job_id
    or slurm_observed.get("slurmd_nodename") != node
    or slurm_observed.get("hostname") != node
    or any(slurm_observed.get(key) != "1" for key in (
        "step_num_nodes", "step_num_tasks", "nnodes", "ntasks",
        "cpus_per_task",
    ))
    or any(slurm_observed.get(key) != "0" for key in (
        "procid", "localid", "nodeid",
    ))
    or type(slurm_observed.get("step_id")) is not str
    or not slurm_observed["step_id"]
    or slurm_observed.get("exact_one_cpu_observed") is not True
    or slurm_observed.get("no_gpu_observed") is not True
    or slurm_observed.get("torch_visible_gpu_count") != 0
    or slurm_observed.get("gres_request") != "none"
    or type(slurm_observed.get("cpu_affinity")) is not list
    or len(slurm_observed["cpu_affinity"]) != 1
    or set(slurm_observed.get("gpu_environment", {})) != {
        "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "SLURM_STEP_GPUS",
        "SLURM_JOB_GPUS",
    }
    or slurm_observed["gpu_environment"].get("SLURM_STEP_GPUS") not in (None, "")
    or any(slurm_observed["gpu_environment"].get(key) not in (None, "", "NoDevFiles")
           for key in (
               "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
               "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
           ))
):
    raise SystemExit("publisher Slurm/runtime child join differs")
python_observed = child["python"]
if (
    set(python_observed) != {
        "designated_binding", "inherited_exec_fd_binding",
        "proc_self_exe_binding", "compute_shell_inherited_fd_binding",
        "designated_inherited_proc_image_exact", "version", "isolated",
        "no_site", "ignore_environment", "safe_path", "dont_write_bytecode",
        "torch_version", "torch_hip_version",
    }
    or any(cross_node_stable_projection(python_observed.get(key))
           != cross_node_stable_projection(post["python"])
           for key in ("designated_binding", "inherited_exec_fd_binding",
                       "proc_self_exe_binding"))
    or cross_node_stable_projection(
           python_observed.get("compute_shell_inherited_fd_binding"))
       != cross_node_stable_projection(post["controller_shell"]["binding"])
    or python_observed.get("torch_version") != "2.7.1+rocm6.3"
    or python_observed.get("torch_hip_version")
       != "6.3.42131-fa1d09cbd"
    or python_observed.get("version") != "3.12.13"
    or python_observed.get("designated_inherited_proc_image_exact") is not True
    or any(python_observed.get(key) is not True for key in (
        "isolated", "no_site", "ignore_environment", "safe_path",
        "dont_write_bytecode",
    ))
):
    raise SystemExit("publisher Python image/Torch/HIP join differs")

rejoin_authorities(post, recovery_summary)
if (
    not execution.is_absolute() or execution.name in {"", ".", ".."}
    or execution.name == receipt_name or receipt_name != "execution-receipt.json"
    or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY")
    or os.path.lexists(execution)
):
    raise SystemExit("execution receipt target/capability differs")

# Finalize the evidence object before the receipt aliases it.  The temporary
# tree is intentionally retained read-only; no later code may mutate this
# evidence or any nested object serialized into the exact12 receipt.
retain_captured_intermediates(intermediate_root_fd, intermediate_evidence)
intermediate_root_fd = -1

receipt = {
    "schema_version": "v4g-scientific-no-go-recovery-controller-execution-v1",
    "status": "SEALED_SCIENTIFIC_NO_GO_ATTESTED",
    "controller_invocation": {
        "controller": post["controller"],
        "controller_shell": post["controller_shell"],
        "declared_formal_launcher":
            post["controller_shell"]["declared_formal_launcher"],
        "observed_process_image_environment_and_argv_contract_exact": True,
        "literal_parent_env_utility_invocation_observable": False,
        "clean_environment_observed_exact": True,
        "caller_argument_count": 1,
        "caller_supplied_controller_sha256": controller_sha,
        "controller_contains_self_sha_pin": False,
        "intermediate_execution_ledger": intermediate_evidence,
        "publication_protocol": {
            "nfs_safe_direct_official_root_create": True,
            "rename_or_link_commit_used": False,
            "official_root_nonsealed_mode_before_commit": "0700",
            "receipt_create_mode_before_commit": "0000",
            "receipt_precommit_replay_uses_retained_rdwr_fd": True,
            "receipt_named_path_reopen_before_final_mode_used": False,
            "receipt_final_mode": "0444",
            "root_final_commit_mode": "0555",
            "failure_retains_nonsealed_tombstone": True,
            "terminal_complete_write_is_sole_postcommit_attempt": True,
            "terminal_success_requires_full_write_then_immediate_exit": True,
            "root_final_mode_transition_is_last_execution_namespace_mutation":
                True,
        },
    },
    "slurm_step": {
        "request": {
            "srun": post["srun"], "srun_sha256": srun_sha,
            "job_id": job_id, "node": node, "nodes": 1, "tasks": 1,
            "cpus_per_task": 1, "gres": "none", "overlap": True,
            "exact": True,
        },
        "observed": slurm_observed,
    },
    "python_execution": {
        "image": python_observed,
        "publisher_process_image_binding": publisher_process_image,
        "publisher_python_runtime": publisher_python_runtime,
        "runtime_execution": child["runtime_execution"],
        "torch_version": python_observed["torch_version"],
        "torch_hip_version": python_observed["torch_hip_version"],
        "python_environment_trust_boundary":
            post["python_environment_trust_boundary"],
    },
    "test_runs": {"normal": normal, "optimized": optimized},
    "compile_ast": compile_ast,
    "authority_preflight": pre,
    "authority_postflight": post,
    "original_run_postflight": post["original_run"],
    "recovery_attestation": recovery_summary,
}
unsigned_keys = {
    "schema_version", "status", "controller_invocation", "slurm_step",
    "python_execution", "test_runs", "compile_ast", "authority_preflight",
    "authority_postflight", "original_run_postflight",
    "recovery_attestation",
}
if set(receipt) != unsigned_keys or len(receipt) != 11:
    raise SystemExit("unsigned execution receipt exact11 differs")
unsigned_raw = canonical(receipt)
unsigned_snapshot = strict_json(unsigned_raw + b"\n")
if unsigned_snapshot != receipt or canonical(unsigned_snapshot) != unsigned_raw:
    raise SystemExit("unsigned execution receipt deep snapshot differs")
receipt["receipt_digest"] = object_sha(receipt)
if set(receipt) != unsigned_keys | {"receipt_digest"} or len(receipt) != 12:
    raise SystemExit("execution receipt top-level exact12 differs")
raw = canonical(receipt) + b"\n"
file_sha = hashlib.sha256(raw).hexdigest()
receipt = strict_json(raw)
if (canonical(receipt) + b"\n" != raw
        or receipt.get("receipt_digest") != object_sha(unsigned_snapshot)):
    raise SystemExit("execution receipt frozen deep snapshot differs")

def require_directory_path_fd(path, descriptor, mode):
    try:
        resolved = path.resolve(strict=True)
        current = path.lstat()
        opened = os.fstat(descriptor)
    except (FileNotFoundError, RuntimeError) as error:
        raise RuntimeError("directory path/FD authority unavailable") from error
    if (
        str(path) != str(resolved) or not stat.S_ISDIR(current.st_mode)
        or identity(current) != identity(opened)
        or stat.S_IMODE(current.st_mode) != mode
    ):
        raise RuntimeError("directory path/current held-FD join differs")
    return opened

def require_child_directory_fd(parent_fd, name, descriptor, mode):
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(current.st_mode) or identity(current) != identity(opened)
        or stat.S_IMODE(current.st_mode) != mode
    ):
        raise RuntimeError("child directory/current held-FD join differs")
    return opened

def open_held_chain(anchor, target):
    try:
        relative = target.relative_to(anchor)
    except ValueError as error:
        raise RuntimeError("execution target outside trusted anchor") from error
    if (
        not anchor.is_absolute() or not target.is_absolute()
        or str(anchor) != str(anchor.resolve(strict=True))
    ):
        raise RuntimeError("trusted execution anchor differs")
    anchor_fd = os.open(
        anchor,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    descriptors = [anchor_fd]
    links = []
    try:
        require_directory_path_fd(
            anchor, anchor_fd, stat.S_IMODE(os.fstat(anchor_fd).st_mode),
        )
        current_fd = anchor_fd
        for part in relative.parts:
            if part in {"", ".", ".."}:
                raise RuntimeError("trusted execution chain component differs")
            child_fd = os.open(
                part,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current_fd,
            )
            descriptors.append(child_fd)
            links.append((current_fd, part, child_fd))
            require_child_directory_fd(
                current_fd, part, child_fd,
                stat.S_IMODE(os.fstat(child_fd).st_mode),
            )
            current_fd = child_fd
        return descriptors, links, current_fd
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise

def require_held_chain(anchor, descriptors, links):
    if not descriptors:
        raise RuntimeError("trusted execution descriptor chain differs")
    require_directory_path_fd(
        anchor, descriptors[0], stat.S_IMODE(os.fstat(descriptors[0]).st_mode),
    )
    for parent_fd, name, child_fd in links:
        require_child_directory_fd(
            parent_fd, name, child_fd,
            stat.S_IMODE(os.fstat(child_fd).st_mode),
        )

def replay_retained_receipt(directory_fd, descriptor, expected_mode):
    before = os.stat(receipt_name, dir_fd=directory_fd, follow_symlinks=False)
    digest = hashlib.sha256()
    chunks = []
    opened = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        chunks.append(chunk)
    closed = os.fstat(descriptor)
    after = os.stat(receipt_name, dir_fd=directory_fd, follow_symlinks=False)
    actual_raw = b"".join(chunks)
    actual = strict_json(actual_raw)
    unsigned = dict(actual)
    stored = unsigned.pop("receipt_digest", None)
    if (
        len({identity(value) for value in (before, opened, closed, after)}) != 1
        or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != expected_mode
        or before.st_nlink != 1
        or actual_raw != raw or digest.hexdigest() != file_sha
        or set(actual) != unsigned_keys | {"receipt_digest"}
        or len(actual) != 12 or stored != object_sha(unsigned)
        or stored != receipt["receipt_digest"]
    ):
        raise RuntimeError("execution receipt retained-FD/self-digest differs")
    return before

chain_descriptors = []
chain_links = []
parent_fd = None
root_fd = None
receipt_fd = None
root_created = False
receipt_created = False
try:
    chain_descriptors, chain_links, parent_fd = open_held_chain(
        trusted_anchor, parent,
    )
    require_held_chain(trusted_anchor, chain_descriptors, chain_links)
    parent_before = parent.lstat()
    require_directory_path_fd(
        parent, parent_fd, stat.S_IMODE(parent_before.st_mode),
    )
    if os.path.lexists(execution):
        raise RuntimeError("official execution root is not fresh")

    # NFS-safe ownership claim.  The official root deliberately remains 0700
    # and therefore nonsealed throughout construction; no rename/link commit
    # primitive is used or trusted.
    os.mkdir(execution.name, 0o700, dir_fd=parent_fd)
    root_created = True
    root_fd = os.open(
        execution.name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    require_child_directory_fd(parent_fd, execution.name, root_fd, 0o700)
    if os.listdir(root_fd):
        raise RuntimeError("direct-created execution root is not empty")

    receipt_fd = os.open(
        receipt_name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o000, dir_fd=root_fd,
    )
    receipt_created = True
    view = memoryview(raw)
    while view:
        count = os.write(receipt_fd, view)
        if count <= 0:
            raise RuntimeError("execution receipt write stalled")
        view = view[count:]
    os.fsync(receipt_fd)
    written = os.fstat(receipt_fd)
    if (not stat.S_ISREG(written.st_mode)
            or stat.S_IMODE(written.st_mode) != 0o000
            or written.st_nlink != 1 or written.st_size != len(raw)):
        raise RuntimeError("direct-created receipt initial seal differs")
    if os.listdir(root_fd) != [receipt_name]:
        raise RuntimeError("direct-created execution initial exact1 differs")
    replay_retained_receipt(root_fd, receipt_fd, 0o000)

    # The direct-create root is still explicitly nonsealed.  Rejoin every
    # source and current lexical/held object before committing file mode.
    rejoin_authorities(post, recovery_summary)
    require_held_chain(trusted_anchor, chain_descriptors, chain_links)
    require_directory_path_fd(
        parent, parent_fd, stat.S_IMODE(os.fstat(parent_fd).st_mode),
    )
    require_child_directory_fd(parent_fd, execution.name, root_fd, 0o700)
    if os.listdir(root_fd) != [receipt_name]:
        raise RuntimeError("direct-created execution preseal exact1 differs")
    replay_retained_receipt(root_fd, receipt_fd, 0o000)

    os.fchmod(receipt_fd, 0o444)
    os.fsync(receipt_fd)
    replay_retained_receipt(root_fd, receipt_fd, 0o444)
    os.fsync(root_fd)
    os.fsync(parent_fd)

    # Final authority boundary while root mode remains the nonsealed 0700.
    rejoin_authorities(post, recovery_summary)
    require_held_chain(trusted_anchor, chain_descriptors, chain_links)
    require_directory_path_fd(
        parent, parent_fd, stat.S_IMODE(os.fstat(parent_fd).st_mode),
    )
    require_child_directory_fd(parent_fd, execution.name, root_fd, 0o700)
    if os.listdir(root_fd) != [receipt_name]:
        raise RuntimeError("direct-created execution commit-boundary exact1 differs")
    replay_retained_receipt(root_fd, receipt_fd, 0o444)
    os.fsync(receipt_fd)
    os.fsync(root_fd)
    os.fsync(parent_fd)

    # Precompute the sole success output before the final mode transition.
    terminal_line = (
        "V4G_RECOVERY_CONTROLLER_COMPLETE=" + file_sha + ":"
        + receipt["receipt_digest"] + "\n"
    ).encode("ascii")
    if len(terminal_line) != 163:
        raise RuntimeError("terminal completion line length differs")

    # This is the terminal authority boundary.  Everything remains explicitly
    # nonsealed until it succeeds, and the retained receipt FD is the only byte
    # source; the mode-0000 basename has never been reopened.
    rejoin_authorities(post, recovery_summary)
    require_held_chain(trusted_anchor, chain_descriptors, chain_links)
    require_directory_path_fd(
        parent, parent_fd, stat.S_IMODE(os.fstat(parent_fd).st_mode),
    )
    require_child_directory_fd(parent_fd, execution.name, root_fd, 0o700)
    if os.listdir(root_fd) != [receipt_name]:
        raise RuntimeError("direct-created execution terminal exact1 differs")
    replay_retained_receipt(root_fd, receipt_fd, 0o444)

except BaseException as error:
    tombstone_error = None
    if root_created and root_fd is not None and parent_fd is not None:
        try:
            held_root = os.fstat(root_fd)
            current_matches_held = False
            try:
                current_root = os.stat(
                    execution.name, dir_fd=parent_fd, follow_symlinks=False,
                )
                current_matches_held = (
                    stat.S_ISDIR(current_root.st_mode)
                    and (current_root.st_dev, current_root.st_ino)
                        == (held_root.st_dev, held_root.st_ino)
                )
            except FileNotFoundError:
                current_matches_held = False
            receipt_failure_mode = None
            if receipt_created and receipt_fd is not None:
                receipt_failure_mode = (
                    f"{stat.S_IMODE(os.fstat(receipt_fd).st_mode):04o}"
                )
            tombstone = {
                "schema_version": (
                    "v4g-recovery-controller-execution-failure-tombstone-v1"
                ),
                "status": "FAILED_NOT_SEALED",
                "controller_complete_stdout_emitted": False,
                "execution_root_mode_octal": "0700",
                "execution_receipt_mode_octal": receipt_failure_mode,
                "failure_retained_for_forensics": True,
                "retry_in_same_official_root_forbidden": True,
            }
            tombstone_raw = canonical(tombstone) + b"\n"
            tombstone_fd = os.open(
                "FAILED-NOT-SEALED.json",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o400, dir_fd=root_fd,
            )
            try:
                view = memoryview(tombstone_raw)
                while view:
                    count = os.write(tombstone_fd, view)
                    if count <= 0:
                        raise RuntimeError("failure tombstone write stalled")
                    view = view[count:]
                os.fsync(tombstone_fd)
                tombstone_info = os.fstat(tombstone_fd)
            finally:
                os.close(tombstone_fd)
            if (stat.S_IMODE(tombstone_info.st_mode) != 0o400
                    or tombstone_info.st_nlink != 1):
                raise RuntimeError("failure tombstone seal differs")
            os.fsync(root_fd)
            os.fsync(parent_fd)
            if current_matches_held:
                require_child_directory_fd(
                    parent_fd, execution.name, root_fd, 0o700,
                )
                expected_members = ["FAILED-NOT-SEALED.json"]
                if receipt_created:
                    expected_members.append(receipt_name)
                if sorted(os.listdir(root_fd)) != sorted(expected_members):
                    raise RuntimeError("failure tombstone membership differs")
        except BaseException as caught:
            tombstone_error = caught
    if receipt_fd is not None:
        try:
            os.close(receipt_fd)
        except OSError:
            pass
    if root_fd is not None:
        try:
            os.close(root_fd)
        except OSError:
            pass
    for descriptor in reversed(chain_descriptors):
        try:
            os.close(descriptor)
        except OSError:
            pass
    if tombstone_error is not None:
        raise RuntimeError(
            "execution direct-create failure tombstone failed"
        ) from tombstone_error
    raise error

# No cleanup or tombstone handler encloses this terminal commit.  Its dedicated
# handler performs only immediate _exit(74), so an asynchronous exception can
# never chmod back, tombstone, clean, or otherwise touch a committed 0555 root.
# Once fchmod returns, the only attempted syscall is the bounded stdout write,
# followed by immediate process exit.
try:
    os.fchmod(root_fd, 0o555)
    terminal_written = os.write(1, terminal_line)
except BaseException:
    os._exit(74)
os._exit(0 if terminal_written == len(terminal_line) else 74)
PY
