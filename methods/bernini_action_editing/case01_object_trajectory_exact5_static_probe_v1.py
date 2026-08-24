#!/usr/bin/env python3
"""Pure-stdlib static admission for the sealed trajectory exact-five HOLD."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import types
from typing import Any, Mapping, Sequence


SCHEMA = "case01-object-trajectory-exact5-static-admission-v1"


class StaticProbeError(RuntimeError):
    pass


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode,
        info.st_nlink, info.st_rdev, info.st_size, info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _load_launcher(path_value: str, expected_sha256: str) -> types.ModuleType:
    path = Path(path_value)
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
        raise StaticProbeError("launcher path is not canonical")
    try:
        named = os.lstat(path)
    except OSError as error:
        raise StaticProbeError("launcher path is missing") from error
    if (
        not stat.S_ISREG(named.st_mode) or named.st_nlink != 1
        or path.resolve(strict=True) != path
    ):
        raise StaticProbeError("launcher is not one regular canonical file")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_size <= 0
            or _identity(before) != _identity(named)
        ):
            raise StaticProbeError("opened launcher differs before read")
        raw = os.pread(descriptor, before.st_size, 0)
        replay = os.pread(descriptor, before.st_size, 0)
        eof = os.pread(descriptor, 1, before.st_size)
        after = os.fstat(descriptor)
        named_after = os.lstat(path)
    finally:
        os.close(descriptor)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise StaticProbeError("launcher SHA differs")
    if (
        replay != raw or len(raw) != before.st_size or eof != b""
        or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
        or _identity(before) != _identity(named)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named_after)
    ):
        raise StaticProbeError("launcher identity changed")
    spec = importlib.util.spec_from_loader(
        "_case01_object_trajectory_hold_launcher_static", loader=None,
        origin=str(path),
    )
    if spec is None:
        raise StaticProbeError("cannot create launcher spec")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    exec(compile(raw.decode("utf-8", "strict"), str(path), "exec"), module.__dict__)
    return module


def probe(
    launcher_path: str, launcher_sha256: str, input_path: str,
) -> dict[str, Any]:
    launcher = _load_launcher(launcher_path, launcher_sha256)
    raw, identity = launcher.stable_file(input_path)
    value = launcher._strict_json(raw, label="static launch input")
    validated = launcher.validate_input(value, reopen=True)
    blocked = launcher.blocked_roles()
    if blocked:
        raise StaticProbeError("final source pin closure is incomplete")
    plan_row = validated["identities"]["plan"]
    plan_raw, _ = launcher.stable_file(
        plan_row["path"], expected_sha256=plan_row["sha256"],
        expected_size=plan_row["size"],
    )
    plan = launcher._strict_json(plan_raw, label="static HOLD plan")
    tasks = plan["tasks"]
    if (
        [task["oracle_arm"] for task in tasks] != list(launcher.ARM_ORDER)
        or any(task["source_onset_policy"] != "hard1_every_step" for task in tasks)
        or any(
            bool(task.get("external_conditions"))
            != (task["oracle_arm"] not in {"null_before", "null_after"})
            for task in tasks
        )
    ):
        raise StaticProbeError("five-arm static routing differs")
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "ADMITTED_STATIC_HOLD_ONLY",
        "launch_allowed": False,
        "blocked_roles": list(blocked),
        "final_source_pins_complete": True,
        "exact_identity_count": len(launcher.IDENTITY_ROLES),
        "task_ids": list(launcher.TASK_IDS),
        "arm_order": list(launcher.ARM_ORDER),
        "all_tasks_hard1_every_step": True,
        "null_arms_have_no_external_conditions": True,
        "route_and_active_arms_have_external_conditions": True,
        "torch_imported": False,
        "renderer_imported": False,
        "publication_performed": False,
        "input_sha256": identity["sha256"],
        "launcher_sha256": launcher_sha256,
    }
    result["receipt_digest"] = launcher.object_sha256(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = probe(args.launcher, args.launcher_sha256, args.input)
        launcher = _load_launcher(args.launcher, args.launcher_sha256)
        launcher.create_file(
            Path(args.output), launcher.canonical_json_bytes(result) + b"\n", 0o400,
        )
    except (OSError, StaticProbeError, RuntimeError, ValueError) as error:
        print(f"static probe refused: {error}", file=os.sys.stderr)
        return 96
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
