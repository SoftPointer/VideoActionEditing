#!/usr/bin/env python3
"""Captured-root fake-runner admission for the trajectory exact-five HOLD.

Controller mode pins and loads the sealed launcher's ``ROOT_BOOTSTRAP`` and
executes it under the exact Python identity. Captured mode is reached only by
that bootstrap after it has replayed all exact25 named identities.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types
from typing import Any, Mapping, Sequence


SCHEMA = "case01-object-trajectory-exact5-root-fake-admission-v4"
SPEC_SCHEMA = "case01-object-trajectory-exact5-root-bootstrap-diagnostic-v3"
ENTRY_SCHEMA = "case01-object-trajectory-exact5-captured-root-entry-v3"
CAMPAIGN = "case01-object-trajectory-exact5-r64-engineering-oracle"
ARMS = (
    "null_before", "route_off", "trajectory_bone_only",
    "trajectory_dog_bone", "null_after",
)
TASKS = tuple(f"case01-object-trajectory-{arm}-full644" for arm in ARMS)
IDENTITY_ROLES = (
    "runner", "legacy_exact5_runner", "object_eval", "legacy_exact5_eval",
    "frozen_runner", "bridge", "adapter", "legacy_infer_alias",
    "trajectory_projection", "trajectory_scaffold_module", "base_adapter",
    "eval_v1", "eval_v2", "model_authority", "torchrun_source",
    "torchrun_handler_source", "torch_local_agent_source",
    "torch_dynamic_rendezvous_source", "torch_multiprocessing_api_source",
    "base_model_manifest", "r64_checkpoint_manifest", "python", "ffmpeg",
    "ffprobe", "plan",
)
SHA_RE = re.compile(r"[0-9a-f]{64}")
MARKER = "CASE01_OBJECT_TRAJECTORY_ROOT_FAKE_PASS "


class RootFakeError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode,
        info.st_nlink, info.st_rdev, info.st_size,
        getattr(info, "st_blocks", 0), info.st_mtime_ns, info.st_ctime_ns,
    )


def stable(
    path: Path, expected_sha256: str | None = None,
    expected_size: int | None = None, *, executable: bool = False,
) -> bytes:
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
        raise RootFakeError(f"noncanonical named authority: {path}")
    try:
        named = os.lstat(path)
    except OSError as error:
        raise RootFakeError(f"missing named authority: {path}") from error
    if (
        not stat.S_ISREG(named.st_mode) or named.st_nlink != 1
        or (executable and not named.st_mode & 0o111)
        or path.resolve(strict=True) != path
    ):
        raise RootFakeError(f"named authority is not one regular file: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or identity(before) != identity(named)
            or (expected_size is not None and before.st_size != expected_size)
            or (executable and not before.st_mode & 0o111)
        ):
            raise RootFakeError(f"opened authority differs before read: {path}")
        pieces: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            piece = os.pread(
                descriptor, min(1_048_576, before.st_size - offset), offset,
            )
            if not piece:
                break
            pieces.append(piece); offset += len(piece)
        raw = b"".join(pieces)
        middle = os.fstat(descriptor)
        replay = b"".join(
            os.pread(descriptor, min(1_048_576, before.st_size - at), at)
            for at in range(0, before.st_size, 1_048_576)
        )
        eof = os.pread(descriptor, 1, before.st_size)
        after = os.fstat(descriptor); named_after = os.lstat(path)
    finally:
        os.close(descriptor)
    if (
        len(raw) != before.st_size or replay != raw or eof != b""
        or identity(named) != identity(before)
        or identity(before) != identity(middle)
        or identity(before) != identity(after)
        or identity(before) != identity(named_after)
        or (expected_sha256 is not None
            and hashlib.sha256(raw).hexdigest() != expected_sha256)
        or (expected_size is not None and len(raw) != expected_size)
    ):
        raise RootFakeError(f"named authority replay differs: {path}")
    return raw


def strict_json_raw(raw: bytes, *, newline: bool, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RootFakeError(f"duplicate key in {label}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise RootFakeError(f"invalid JSON: {label}") from error
    suffix = b"\n" if newline else b""
    if type(value) is not dict or raw != canonical(value) + suffix:
        raise RootFakeError(f"noncanonical JSON: {label}")
    return value


def strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = stable(path)
    return strict_json_raw(raw, newline=True, label=str(path)), raw


def load_launcher(path: Path, expected_sha256: str) -> types.ModuleType:
    raw = stable(path, expected_sha256)
    spec = importlib.util.spec_from_loader(
        "_trajectory_root_fake_launcher", loader=None, origin=str(path),
    )
    if spec is None:
        raise RootFakeError("cannot construct launcher module")
    module = importlib.util.module_from_spec(spec); module.__file__ = str(path)
    exec(compile(raw.decode("utf-8", "strict"), str(path), "exec"), module.__dict__)
    return module


def validate_plan_identity_crosslinks(
    plan: Mapping[str, Any], identities: Mapping[str, Mapping[str, Any]],
) -> None:
    producer = plan.get("producer") if isinstance(plan, Mapping) else None
    checkpoint = (
        plan.get("checkpoint_manifest") if isinstance(plan, Mapping) else None
    )
    expected = {
        "legacy_infer_alias": (
            "infer_lora_path", "infer_lora_sha256", "infer_lora_size",
        ),
        "adapter": (
            "inference_wrapper_path", "inference_wrapper_sha256",
            "inference_wrapper_size",
        ),
        "trajectory_projection": (
            "trajectory_projection_module_path",
            "trajectory_projection_module_sha256",
            "trajectory_projection_module_size",
        ),
        "trajectory_scaffold_module": (
            "trajectory_scaffold_module_path",
            "trajectory_scaffold_module_sha256",
            "trajectory_scaffold_module_size",
        ),
        "ffprobe": ("ffprobe_path", "ffprobe_sha256", "ffprobe_size"),
    }
    if not isinstance(producer, Mapping) or not isinstance(checkpoint, Mapping):
        raise RootFakeError("plan producer/checkpoint closure differs")
    for role, (path_key, sha_key, size_key) in expected.items():
        row = identities.get(role)
        if not isinstance(row, Mapping) or row != {
            "path": producer.get(path_key),
            "sha256": producer.get(sha_key),
            "size": producer.get(size_key),
        }:
            raise RootFakeError(f"plan producer identity differs: {role}")
    checkpoint_row = identities.get("r64_checkpoint_manifest")
    if (
        not isinstance(checkpoint_row, Mapping)
        or checkpoint.get("path") != checkpoint_row.get("path")
        or checkpoint.get("sha256") != checkpoint_row.get("sha256")
        or any(
            not isinstance(task, Mapping)
            or not isinstance(task.get("adapter"), Mapping)
            or task["adapter"].get("checkpoint_manifest") != checkpoint
            for task in plan.get("tasks", [])
        )
    ):
        raise RootFakeError("plan checkpoint identity differs")


def create(path: Path, value: Mapping[str, Any]) -> None:
    if (
        not path.is_absolute() or os.path.normpath(str(path)) != str(path)
        or os.path.lexists(path)
    ):
        raise RootFakeError("captured result target is not fresh/canonical")
    parent = path.parent
    named_parent = os.lstat(parent)
    if not stat.S_ISDIR(named_parent.st_mode) or parent.resolve(strict=True) != parent:
        raise RootFakeError("captured result parent differs")
    raw = canonical(value) + b"\n"
    descriptor = os.open(
        path, os.O_RDWR | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), 0,
    )
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise RootFakeError("captured result write made no progress")
            offset += count
        os.fsync(descriptor); os.fchmod(descriptor, 0o400); os.fsync(descriptor)
        before = os.fstat(descriptor); named = os.lstat(path)
        replay = os.pread(descriptor, len(raw), 0)
    finally:
        os.close(descriptor)
    if (
        replay != raw or stat.S_IMODE(before.st_mode) != 0o400
        or before.st_nlink != 1 or identity(before) != identity(named)
    ):
        raise RootFakeError("captured result replay differs")


def captured_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--captured-result", required=True)
    args = parser.parse_args(argv)
    allowed = {"CASE01_OBJECT_TRAJECTORY_CAPTURED_ROOT_ENTRY"}
    if set(os.environ) != allowed:
        raise RootFakeError("captured environment closure differs")
    raw_entry = os.environ["CASE01_OBJECT_TRAJECTORY_CAPTURED_ROOT_ENTRY"].encode()
    entry = strict_json_raw(raw_entry, newline=False, label="captured entry")
    unsigned = dict(entry); claimed = unsigned.pop("authority_digest", None)
    captured_runner = entry.get("captured_runner")
    production_runner = entry.get("production_runner")
    source = Path(__file__)
    source_raw = stable(
        source,
        captured_runner.get("sha256") if type(captured_runner) is dict else None,
        captured_runner.get("size") if type(captured_runner) is dict else None,
    )
    if (
        set(entry) != {
            "schema_version", "release_digest", "identity_roles",
            "identity_set_digest", "launch_input_sha256",
            "production_runner", "captured_runner",
            "captured_runner_identity", "plan_sha256", "task_ids",
            "arm_order", "all_exact25_named_identities_replayed",
            "captured_runner_outside_exact25",
            "captured_runner_bytes_compiled", "publication_performed",
            "authority_digest",
        }
        or entry.get("schema_version") != ENTRY_SCHEMA
        or claimed != digest(unsigned)
        or entry.get("identity_roles") != list(IDENTITY_ROLES)
        or SHA_RE.fullmatch(str(entry.get("identity_set_digest"))) is None
        or SHA_RE.fullmatch(str(entry.get("launch_input_sha256"))) is None
        or type(production_runner) is not dict
        or set(production_runner) != {"path", "sha256", "size"}
        or Path(str(production_runner.get("path"))).name
        != "case01_object_trajectory_exact5_runner_v1.py"
        or SHA_RE.fullmatch(str(production_runner.get("sha256"))) is None
        or type(production_runner.get("size")) is not int
        or production_runner["size"] <= 0
        or captured_runner != {
            "path": str(source), "sha256": hashlib.sha256(source_raw).hexdigest(),
            "size": len(source_raw),
        }
        or entry.get("captured_runner_identity")
        != list(identity(os.lstat(source)))
        or production_runner == captured_runner
        or entry.get("task_ids") != list(TASKS)
        or entry.get("arm_order") != list(ARMS)
        or entry.get("all_exact25_named_identities_replayed") is not True
        or entry.get("captured_runner_outside_exact25") is not True
        or entry.get("captured_runner_bytes_compiled") is not True
        or entry.get("publication_performed") is not False
        or "torch" in sys.modules
    ):
        raise RootFakeError("captured root entry differs")
    result: dict[str, Any] = {
        "schema_version": SCHEMA, "status": "PASS_CAPTURED_ROOT_FAKE_HOLD",
        "campaign_mode": CAMPAIGN, "launch_allowed": False,
        "exact_identity_count": 25, "identity_roles": list(IDENTITY_ROLES),
        "task_ids": list(TASKS), "arm_order": list(ARMS),
        "release_digest": entry["release_digest"],
        "identity_set_digest": entry["identity_set_digest"],
        "launch_input_sha256": entry["launch_input_sha256"],
        "entry_authority_digest": entry["authority_digest"],
        "plan_sha256": entry["plan_sha256"],
        "production_runner_sha256": production_runner["sha256"],
        "captured_runner_sha256": captured_runner["sha256"],
        "all_exact25_named_identities_replayed": True,
        "captured_runner_outside_exact25": True,
        "captured_runner_bytes_compiled": True, "torch_imported": False,
        "renderer_imported": False, "publication_performed": False,
    }
    result["receipt_digest"] = digest(result)
    create(Path(args.captured_result), result)
    print(MARKER + result["receipt_digest"])
    return 0


def probe(
    spec_path: str, *, launcher_path: str, launcher_sha256: str,
    python_path: str, python_sha256: str, output_path: str,
    launch_input_path: str, launch_input_sha256: str,
) -> dict[str, Any]:
    output = Path(output_path)
    value, _raw = strict_json(Path(spec_path))
    if (
        set(value) != {
            "schema_version", "campaign_mode", "launch_allowed",
            "identities", "captured_runner", "launch_input", "result_path",
        }
        or value.get("schema_version") != SPEC_SCHEMA
        or value.get("campaign_mode") != CAMPAIGN
        or value.get("launch_allowed") is not False
        or value.get("result_path") != str(output)
        or type(value.get("identities")) is not dict
        or set(value["identities"]) != set(IDENTITY_ROLES)
        or len(value["identities"]) != 25
    ):
        raise RootFakeError("root diagnostic spec closure differs")
    captured_runner = value.get("captured_runner")
    self_path = Path(__file__).resolve(); self_raw = stable(self_path)
    if captured_runner != {
        "path": str(self_path), "sha256": hashlib.sha256(self_raw).hexdigest(),
        "size": len(self_raw),
    } or captured_runner["path"] in {
        row.get("path") for row in value["identities"].values()
        if type(row) is dict
    }:
        raise RootFakeError("captured runner authority overlaps production exact25")
    python_row = value["identities"].get("python")
    python_raw = stable(
        Path(python_path), python_sha256,
        python_row.get("size") if type(python_row) is dict else None,
        executable=True,
    )
    if python_row != {
        "path": python_path, "sha256": python_sha256, "size": len(python_raw),
    }:
        raise RootFakeError("diagnostic Python row differs")
    launcher = load_launcher(Path(launcher_path), launcher_sha256)
    if not isinstance(getattr(launcher, "ROOT_BOOTSTRAP", None), str):
        raise RootFakeError("launcher lacks captured root bootstrap")
    if (
        getattr(launcher, "EXPECTED_CAPTURED_ROOT_FAKE_SHA256", None)
        != captured_runner["sha256"]
        or getattr(launcher, "EXPECTED_CAPTURED_ROOT_FAKE_SIZE", None)
        != captured_runner["size"]
    ):
        raise RootFakeError("launcher does not pin this captured runner authority")
    if SHA_RE.fullmatch(launch_input_sha256) is None:
        raise RootFakeError("launch input pin is incomplete")
    launch_raw = stable(Path(launch_input_path), launch_input_sha256)
    launch_value = strict_json_raw(
        launch_raw, newline=True, label="validated HOLD launch input",
    )
    launch_row = value.get("launch_input")
    if launch_row != {
        "path": launch_input_path,
        "sha256": launch_input_sha256,
        "size": len(launch_raw),
    }:
        raise RootFakeError("root diagnostic launch input authority differs")
    try:
        validated_launch = launcher.validate_input(launch_value, reopen=True)
    except Exception as error:
        raise RootFakeError("launcher rejected exact25 production identities") from error
    if validated_launch.get("identities") != value["identities"]:
        raise RootFakeError("diagnostic identities differ from launch input")
    plan_row = value["identities"]["plan"]
    plan_raw = stable(
        Path(plan_row["path"]), plan_row["sha256"], plan_row["size"],
    )
    plan = strict_json_raw(plan_raw, newline=True, label="bound HOLD plan")
    validate_plan_identity_crosslinks(plan, value["identities"])
    if os.path.lexists(output):
        raise RootFakeError("root fake output is not fresh")
    completed = subprocess.run(
        [
            python_path, "-I", "-S", "-B", "-c", launcher.ROOT_BOOTSTRAP,
            canonical(value).decode("utf-8"), digest(value),
        ],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, env={}, timeout=30,
        start_new_session=True,
    )
    if completed.returncode != 0 or completed.stderr != "":
        raise RootFakeError(
            "captured root bootstrap failed: " + completed.stderr[-1000:]
        )
    result, result_raw = strict_json(output)
    unsigned = dict(result); claimed = unsigned.pop("receipt_digest", None)
    if (
        set(result) != {
            "schema_version", "status", "campaign_mode", "launch_allowed",
            "exact_identity_count", "identity_roles", "task_ids", "arm_order",
            "release_digest", "identity_set_digest", "launch_input_sha256",
            "entry_authority_digest", "plan_sha256",
            "production_runner_sha256", "captured_runner_sha256",
            "all_exact25_named_identities_replayed",
            "captured_runner_outside_exact25",
            "captured_runner_bytes_compiled", "torch_imported",
            "renderer_imported", "publication_performed", "receipt_digest",
        }
        or result.get("schema_version") != SCHEMA
        or result.get("status") != "PASS_CAPTURED_ROOT_FAKE_HOLD"
        or result.get("campaign_mode") != CAMPAIGN
        or result.get("launch_allowed") is not False
        or result.get("exact_identity_count") != 25
        or result.get("identity_roles") != list(IDENTITY_ROLES)
        or result.get("task_ids") != list(TASKS)
        or result.get("arm_order") != list(ARMS)
        or result.get("release_digest") != digest(value)
        or result.get("identity_set_digest") != digest(value["identities"])
        or result.get("launch_input_sha256") != launch_input_sha256
        or result.get("plan_sha256") != value["identities"]["plan"]["sha256"]
        or result.get("production_runner_sha256")
        != value["identities"]["runner"]["sha256"]
        or result.get("captured_runner_sha256") != captured_runner["sha256"]
        or result.get("all_exact25_named_identities_replayed") is not True
        or result.get("captured_runner_outside_exact25") is not True
        or result.get("captured_runner_bytes_compiled") is not True
        or result.get("torch_imported") is not False
        or result.get("renderer_imported") is not False
        or result.get("publication_performed") is not False
        or claimed != digest(unsigned)
        or completed.stdout != MARKER + claimed + "\n"
        or result_raw != canonical(result) + b"\n"
    ):
        raise RootFakeError("captured root receipt differs")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--launch-input", required=True)
    parser.add_argument("--launch-input-sha256", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--python-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if "CASE01_OBJECT_TRAJECTORY_CAPTURED_ROOT_ENTRY" in os.environ:
        return captured_main(argv or sys.argv[1:])
    args = build_parser().parse_args(argv)
    try:
        result = probe(
            args.spec, launcher_path=args.launcher,
            launcher_sha256=args.launcher_sha256, python_path=args.python,
            python_sha256=args.python_sha256, output_path=args.output,
            launch_input_path=args.launch_input,
            launch_input_sha256=args.launch_input_sha256,
        )
    except (OSError, ValueError, subprocess.SubprocessError, RootFakeError) as error:
        print(f"root fake refused: {error}", file=sys.stderr)
        return 96
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
