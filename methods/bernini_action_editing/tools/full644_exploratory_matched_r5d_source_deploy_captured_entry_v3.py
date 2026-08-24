"""Inline-only captured entry for the AUH r5d source deployment controller.

Do not execute this file by name.  A trusted caller must capture and hash its
exact bytes, then provide those bytes to root-owned ``/usr/bin/python3 -c``.
"""

import hashlib
import os
import stat
import sys


LOCAL_ENV = {
    "PATH": "/usr/bin:/bin",
    "LC_ALL": "C",
    "LANG": "C",
    "HOME": "/Users/siriuschu",
    "TMPDIR": "/private/tmp",
    "__CF_USER_TEXT_ENCODING": "0x1F5:0x0:0x0",
}
PYTHON_PATH = (
    "/Library/Developer/CommandLineTools/Library/Frameworks/"
    "Python3.framework/Versions/3.9/bin/python3.9"
)
PYTHON_SHA256 = "d23458804881b5c23d3aacae44311b9c43f961c4eba3a23163572aaebf58f44f"
PYTHON_SIZE = 102352
PYTHON_NLINK = 1
CONTROLLER_PATH = (
    "/Users/siriuschu/ML/VideoEditing/VideoEdit/methods/"
    "bernini_action_editing/tools/"
    "deploy_full644_exploratory_matched_r5d_sources_auh_v3.py"
)


def fail(message):
    raise RuntimeError(message)


def ident(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_gid,
        info.st_mode,
        info.st_nlink,
        info.st_rdev,
        info.st_size,
        getattr(info, "st_blocks", 0),
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def pread(descriptor, size):
    blocks = []
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1048576, size - offset), offset)
        if not block:
            break
        blocks.append(block)
        offset += len(block)
    raw = b"".join(blocks)
    if len(raw) != size:
        fail("captured entry held read is incomplete")
    return raw


if (
    sys.platform != "darwin"
    or sys.flags.isolated != 1
    or sys.flags.no_site != 1
    or sys.flags.ignore_environment != 1
    or not sys.dont_write_bytecode
    or os.environ != LOCAL_ENV
    or os.geteuid() != 501
    or os.getegid() != 20
    or sys.executable != PYTHON_PATH
    or len(sys.argv) not in (5, 6)
    or sys.argv[0] != "-c"
):
    fail("captured entry process differs")

entry_sha256, controller_path, controller_sha256 = sys.argv[1:4]
inner_argv = sys.argv[4:]
captured_entry_raw = globals().get("__R5D_CAPTURED_ENTRY_RAW")
if (
    not isinstance(captured_entry_raw, bytes)
    or hashlib.sha256(captured_entry_raw).hexdigest() != entry_sha256
    or controller_path != CONTROLLER_PATH
    or len(entry_sha256) != 64
    or len(controller_sha256) != 64
    or any(
        character not in "0123456789abcdef"
        for character in entry_sha256 + controller_sha256
    )
    or inner_argv[0] not in ("--audit-local", "--execute")
    or (inner_argv[0] == "--audit-local" and len(inner_argv) != 1)
    or (inner_argv[0] == "--execute" and len(inner_argv) != 2)
):
    fail("captured entry argv differs")

python_fd = os.open(PYTHON_PATH, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    python_info = os.fstat(python_fd)
    python_first = pread(python_fd, python_info.st_size)
    python_second = pread(python_fd, python_info.st_size)
    python_named = os.lstat(PYTHON_PATH)
    if (
        ident(python_info) != ident(os.fstat(python_fd))
        or ident(python_info) != ident(python_named)
        or not stat.S_ISREG(python_info.st_mode)
        or python_info.st_uid != 0
        or python_info.st_gid != 0
        or stat.S_IMODE(python_info.st_mode) != 0o755
        or python_info.st_nlink != PYTHON_NLINK
        or python_info.st_size != PYTHON_SIZE
        or python_first != python_second
        or hashlib.sha256(python_first).hexdigest() != PYTHON_SHA256
    ):
        fail("captured entry Python identity differs")
finally:
    os.close(python_fd)

controller_fd = os.open(
    controller_path,
    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
)
controller_info = os.fstat(controller_fd)
controller_raw = pread(controller_fd, controller_info.st_size)
controller_again = pread(controller_fd, controller_info.st_size)
controller_named = os.lstat(controller_path)
if (
    ident(controller_info) != ident(os.fstat(controller_fd))
    or ident(controller_info) != ident(controller_named)
    or not stat.S_ISREG(controller_info.st_mode)
    or controller_info.st_uid != 501
    or controller_info.st_gid != 20
    or stat.S_IMODE(controller_info.st_mode) != 0o644
    or controller_info.st_nlink != 1
    or controller_raw != controller_again
    or hashlib.sha256(controller_raw).hexdigest() != controller_sha256
):
    fail("captured controller identity or bytes differ")

controller_source = controller_raw.decode("utf-8", "strict")
controller_code = compile(
    controller_source,
    controller_path,
    "exec",
    dont_inherit=True,
)
controller_globals = {
    "__name__": "__main__",
    "__file__": controller_path,
    "__R5D_CAPTURED_SOURCE_RAW": controller_raw,
    "__R5D_CAPTURED_SOURCE_SHA256": controller_sha256,
    "__R5D_CAPTURED_SOURCE_PATH": controller_path,
    "__R5D_CAPTURED_SOURCE_FD": controller_fd,
    "__R5D_CAPTURED_ENTRY_SHA256": entry_sha256,
}
sys.argv = [controller_path, *inner_argv]
exec(controller_code, controller_globals)

