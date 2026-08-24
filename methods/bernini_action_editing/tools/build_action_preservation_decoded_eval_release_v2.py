#!/usr/bin/env python3
"""Create/audit the pinned exact15 retained-FD decoded-eval release.

The deterministic USTAR implementation is captured from the already audited
exact14 builder by same-FD double read and executed from those captured bytes.
This successor supplies a disjoint generation, exact 15-member closure, and
independent literal SHA/size/mode pins.  It never changes or delegates to the
deployed exact14 release directory.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import stat
from types import ModuleType
from typing import Any, Mapping, Sequence


RELEASE_GENERATION = "preservation-v2-decoded-eval-exact15-r1"
LEGACY_ENGINE = "build_action_preservation_decoded_eval_release_v1.py"
LEGACY_ENGINE_SHA256 = (
    "9eb03a87597fc1d6b2736f8a67830bb3b35f669e9bc4ee893ef6a1ff03c2c6fc"
)
LEGACY_ENGINE_SIZE = 23540
RUNTIME = "action_preservation_decoded_eval_verified_release_v1.py"

EXPECTED_COMPONENTS: Mapping[str, tuple[str, int, int]] = {
    "infer_lora.py": (
        "dde5e3293e4fc833618c970eb51ba61fef4c66ef38dd1e67ab0e12b142f05e48",
        95828,
        0o444,
    ),
    "train_lora.py": (
        "eae8eaac25197112637f466e611ba7eae574266d4cd1b83e625195fb22b0476e",
        84216,
        0o444,
    ),
    "self_generated_action_preservation_v2.py": (
        "11bc0792174a60c2e449eb61ff8f81da97808e02ee2707b5c4f20ee2118f4b5c",
        11334,
        0o444,
    ),
    "action_preservation_gate_v1.py": (
        "2c5e6d2a2e64b59c620b581aab38f243e9d7d0a53e764787fe013f0eede4f844",
        78097,
        0o444,
    ),
    "action_preservation_decoded_eval_plan_v1.py": (
        "287efb71142c91bd0ad78354f6f72948a7aebc5b746c96fb32f701aa7158072b",
        49347,
        0o444,
    ),
    "action_preservation_decoded_eval_bridge_v1.py": (
        "19f3f3b49804197abcd14ac2e95dca5c353c39cacbbc8756616735950f37ea08",
        89918,
        0o444,
    ),
    "action_preservation_decoded_eval_decoder_adapter_v1.py": (
        "0b30ff6d2e4d17b20844abbeea5c26e51d376740cab092f905854279ad713fd1",
        38381,
        0o555,
    ),
    "action_preservation_decoded_eval_executor_v2.py": (
        "8915693b5816d7309e9f66f5a2b08975e579286c6df9e8ea410791e0ad3cce29",
        105577,
        0o444,
    ),
    "action_preservation_decoded_eval_launcher_v1.py": (
        "3646bd09a6f1054d4d5664f8f1ea818a8c1254873acfd97f689afef0aa0c2280",
        27965,
        0o444,
    ),
    "action_preservation_decoded_eval_aggregate_v2.py": (
        "88d909f188372c588a5eac7ddd9d2edae278ba264b550c14656fbe48fc40b963",
        109226,
        0o444,
    ),
    "action_preservation_loop_controller_v1.py": (
        "b070cd82c11251b9b638ff1f39a3c346e8347a0137b8b1e17f8aa2a67661db6c",
        49068,
        0o444,
    ),
    "tools/materialize_vae.py": (
        "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
        32195,
        0o444,
    ),
    "tools/build_renderer_dataset.py": (
        "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
        31012,
        0o444,
    ),
    "action_preservation_decoded_eval_model_authority_v2.py": (
        "6ba965cfc81e073025a918a060a5aebceee836bf9d748d180924c98391b68f16",
        78592,
        0o444,
    ),
    RUNTIME: (
        "d2577dfac8d5c2e6c5d51d4fe03570b85d7857f170fbcf33af5aa2f5f87d8138",
        125419,
        0o444,
    ),
}
MEMBER_ORDER = tuple(EXPECTED_COMPONENTS)
ALLOWED_ENTRYPOINTS = tuple(
    sorted(
        {
            "infer_lora.py",
            "action_preservation_gate_v1.py",
            "action_preservation_decoded_eval_plan_v1.py",
            "action_preservation_decoded_eval_bridge_v1.py",
            "action_preservation_decoded_eval_decoder_adapter_v1.py",
            "action_preservation_decoded_eval_executor_v2.py",
            "action_preservation_decoded_eval_launcher_v1.py",
            "action_preservation_decoded_eval_aggregate_v2.py",
            "action_preservation_loop_controller_v1.py",
            "tools/materialize_vae.py",
            "tools/build_renderer_dataset.py",
        }
    )
)


class Exact15ReleaseBuildError(RuntimeError):
    pass


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_gid,
        value.st_mode,
        value.st_nlink,
        value.st_rdev,
        value.st_size,
        getattr(value, "st_blocks", 0),
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_fd(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _captured_engine() -> ModuleType:
    path = Path(__file__).resolve(strict=True).with_name(LEGACY_ENGINE)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise Exact15ReleaseBuildError("safe builder capture is unavailable")
    descriptor = os.open(path, flags | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        first = _read_fd(descriptor)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or _identity(before) != _identity(middle)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named)
        or first != second
        or len(first) != LEGACY_ENGINE_SIZE
        or hashlib.sha256(first).hexdigest() != LEGACY_ENGINE_SHA256
    ):
        raise Exact15ReleaseBuildError(
            "captured deterministic release engine differs"
        )
    module = ModuleType("_apv2_exact15_release_engine")
    module.__file__ = str(path)
    module.__package__ = None
    exec(compile(first, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def _engine() -> ModuleType:
    engine = _captured_engine()
    engine.RELEASE_GENERATION = RELEASE_GENERATION
    engine.RUNTIME = RUNTIME
    engine.FILES_AND_MODES = {
        relative: row[2] for relative, row in EXPECTED_COMPONENTS.items()
    }
    engine.MEMBER_ORDER = MEMBER_ORDER
    engine.EXPECTED_SHA256 = {
        relative: row[0] for relative, row in EXPECTED_COMPONENTS.items()
    }
    engine.EXPECTED_SIZE = {
        relative: row[1] for relative, row in EXPECTED_COMPONENTS.items()
    }
    engine.ALLOWED_ENTRYPOINTS = ALLOWED_ENTRYPOINTS
    return engine


def _result(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    if row.get("file_count") != 15 or row.get("exact_member_closure") is not True:
        raise Exact15ReleaseBuildError("exact15 build result closure differs")
    row["release_generation"] = RELEASE_GENERATION
    row["deterministic_engine_sha256"] = LEGACY_ENGINE_SHA256
    row["successor_builder"] = "retained-fd-model-authority-v2"
    return row


def build(release_dir: Path) -> dict[str, Any]:
    engine = _engine()
    try:
        return _result(engine.build(Path(release_dir)))
    except engine.EvalReleaseBuildError as error:
        raise Exact15ReleaseBuildError(str(error)) from error


def audit(release_dir: Path, *, against_workspace: bool) -> dict[str, Any]:
    engine = _engine()
    try:
        return _result(
            engine.audit(
                Path(release_dir), against_workspace=against_workspace
            )
        )
    except engine.EvalReleaseBuildError as error:
        raise Exact15ReleaseBuildError(str(error)) from error


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--release-dir", required=True)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--release-dir", required=True)
    audit_parser.add_argument("--against-workspace", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    release_dir = Path(args.release_dir).resolve(strict=False)
    result = (
        build(release_dir)
        if args.command == "build"
        else audit(release_dir, against_workspace=args.against_workspace)
    )
    print(_engine().canonical(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_ENTRYPOINTS",
    "EXPECTED_COMPONENTS",
    "Exact15ReleaseBuildError",
    "MEMBER_ORDER",
    "RELEASE_GENERATION",
    "audit",
    "build",
]
