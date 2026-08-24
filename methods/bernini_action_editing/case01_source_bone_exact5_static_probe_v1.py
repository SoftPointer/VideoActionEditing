#!/usr/bin/env python3
"""Renderer-free compute-node admission probe for the case01 exact5 package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any, Mapping, Sequence


CAMPAIGN = "case01-source-bone-exact5-r64-canary"
JOB_ID = "143808"
NODE = "auh7-1b-gpu-292"
TASK_IDS = (
    "case01-exact_original-full644",
    "case01-codec_only_present-full644",
    "case01-bone_removed-full644",
    "case01-bone_translated_up150-full644",
    "case01-sham_control_up150-full644",
)
RELEASE_FILES = {
    "methods/action_editing_baselines/manifests/goku_legacy_shared8_exposure.json": "953933f1161b6d62826d388ba5ed42e42792fbf5f2bdeea199c1eb13cd251b4a",
    "methods/action_editing_baselines/manifests/goku_legacy_heldout8_inputs.jsonl": "c05c4e5b5bf85de882bde32c71a984d736247733e586ed91d40026b12aaaf701",
    "methods/bernini_action_editing/action_preservation_decoded_eval_model_authority_v2.py": "b9457e434b8000e5368056c925edd0227b4dd3d8a439090494af088817d51ecf",
    "methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256": "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
    "methods/bernini_action_editing/full644_exploratory_matched_eval_v1.py": "d6ef0939a67598e66ccf2652d22520ae3a87a068789f70f921522ba86046138d",
    "methods/bernini_action_editing/full644_exploratory_matched_eval_v2.py": "b675b84fd5f1b95a21f6454c9eb8c53b0965d7dbdd3fedf1ea92b6ad153ac982",
    "methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_v2.py": "53b75aea4897a0ec5ad70c8ea2b2dd314b93d1331cf5e41d65c3b51339f4d4ca",
    "methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_auh_r5f.py": "52dcde8797484504ab28a5c59c532c0877a145ea24762edaf3105a21b0719e19",
    "methods/bernini_action_editing/full644_exploratory_matched_runner_auh_r5.py": "847b91a267fe55cfbfa793027548f82beb5ec9630efab329878576ae6c5a9223",
    "methods/bernini_action_editing/full644_exploratory_matched_spooled_launcher_auh_r5.py": "cb201398940d59393fa58471dc2c3f9fdf001c7e881ec891ce892bb460cf01ba",
    "methods/bernini_action_editing/full644_exploratory_matched_torchrun_fd_bridge_v2.py": "c91de7eb821a05c61f66349c02f9232ede27c49e54659f351f72930fb071d136",
    "methods/bernini_action_editing/infer_lora.py": "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553",
    "methods/bernini_action_editing/self_generated_action_preservation_v2.py": "11bc0792174a60c2e449eb61ff8f81da97808e02ee2707b5c4f20ee2118f4b5c",
    "methods/bernini_action_editing/tools/build_renderer_dataset.py": "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
    "methods/bernini_action_editing/tools/materialize_vae.py": "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
    "methods/bernini_action_editing/train_lora.py": "ead547b8309e1b5ae5c831444e9f5d1d8e1785fed5fe39cf7b97f13f82a9ce85",
    "methods/bernini_action_editing/case01_source_bone_exact5_runner_v1.py": "4a46e870b61ff42345c523cca97b8170c065033bc2af9a0353be9df4373ec3ea",
    "methods/bernini_action_editing/case01_source_bone_exact5_eval_v1.py": "891551d46b0ca11362fe8d446f202036b9957baa943de0aec6da1f3ad45d7d58",
    "methods/bernini_action_editing/case01_source_bone_exact5_spooled_launcher_auh_v1.py": "00b68ca8221dd343cb9ca8393c9205cccf6a61d474c56e56c9b081570418d390",
}
FFPROBE = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/"
    "runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe"
)
FFPROBE_SHA256 = (
    "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5"
)
RANK_CACHE_ROOT = Path(
    "/tmp/bernini-case01-exact5-r64-job143808-node292-r1-rank-cache"
)


class StaticProbeError(RuntimeError):
    """The renderer-free exact5 admission contract differs."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def ident(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode,
            info.st_nlink, info.st_rdev, info.st_size,
            getattr(info, "st_blocks", 0), info.st_mtime_ns, info.st_ctime_ns)


def stable(path: Path, expected: str | None = None, mode: int | None = None) -> bytes:
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path) or path.is_symlink() or path.resolve(strict=True) != path:
        raise StaticProbeError(f"noncanonical file: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        chunks = []
        offset = 0
        while offset < before.st_size:
            block = os.pread(fd, min(1_048_576, before.st_size - offset), offset)
            if not block:
                break
            chunks.append(block); offset += len(block)
        after = os.fstat(fd); named = path.lstat()
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not raw
            or len(raw) != before.st_size or ident(before) != ident(after)
            or ident(before) != ident(named)
            or (expected is not None and hashlib.sha256(raw).hexdigest() != expected)
            or (mode is not None and stat.S_IMODE(before.st_mode) != mode)):
        raise StaticProbeError(f"stable file differs: {path}")
    return raw


def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise StaticProbeError("duplicate JSON key")
        value[key] = item
    return value


def strict_json(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (UnicodeError, ValueError, TypeError) as error:
        raise StaticProbeError("strict JSON decode differs") from error
    if not isinstance(value, dict) or raw != canonical(value) + b"\n":
        raise StaticProbeError("canonical JSON plus LF differs")
    return value


def load_module(name: str, path: Path, expected: str) -> types.ModuleType:
    raw = stable(path, expected, 0o444)
    module = types.ModuleType(name)
    module.__file__ = str(path); module.__package__ = None
    module.__loader__ = None; module.__spec__ = None; module.__cached__ = None
    module.__builtins__ = __builtins__; sys.modules[name] = module
    exec(compile(raw.decode("utf-8", "strict"), str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def create_receipt(path: Path, value: Mapping[str, Any]) -> None:
    if (
        not path.is_absolute() or os.path.normpath(str(path)) != str(path)
        or path.name in {"", ".", ".."}
    ):
        raise StaticProbeError("receipt target path differs")
    raw = canonical(value) + b"\n"
    parent = path.parent
    parent_fd = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | os.O_CLOEXEC,
    )
    try:
        parent_info = os.fstat(parent_fd)
        parent_named = os.lstat(parent)
        if (
            os.path.realpath(parent) != str(parent)
            or not stat.S_ISDIR(parent_info.st_mode)
            or ident(parent_info) != ident(parent_named)
            or parent_info.st_uid != 2012 or parent_info.st_gid != 2000
            or stat.S_IMODE(parent_info.st_mode) != 0o755
        ):
            raise StaticProbeError("receipt parent authority differs")
        fd = os.open(
            path.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            0, dir_fd=parent_fd,
        )
        try:
            offset = 0
            while offset < len(raw):
                count = os.write(fd, raw[offset:])
                if count <= 0:
                    raise StaticProbeError("receipt write made no progress")
                offset += count
            os.fsync(fd)
            before = os.fstat(fd)
            named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                stat.S_IMODE(before.st_mode) != 0 or before.st_nlink != 1
                or before.st_uid != 2012 or before.st_gid != 2000
                or ident(before) != ident(named)
                or os.pread(fd, len(raw), 0) != raw
            ):
                raise StaticProbeError("receipt staging replay differs")
            os.fchmod(fd, 0o400); os.fsync(fd)
            after = os.fstat(fd)
            named_after = os.stat(
                path.name, dir_fd=parent_fd, follow_symlinks=False,
            )
            if (
                stat.S_IMODE(after.st_mode) != 0o400
                or ident(after) != ident(named_after)
                or os.pread(fd, len(raw), 0) != raw
            ):
                raise StaticProbeError("receipt commit replay differs")
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def run(args: argparse.Namespace) -> dict[str, Any]:
    required_env = {
        "SLURM_JOB_ID": JOB_ID, "SLURM_GPUS_ON_NODE": "8",
        "SLURM_GPUS_PER_NODE": "8", "SLURM_STEP_GPUS": "0,1,2,3,4,5,6,7",
        "SLURM_NNODES": "1", "SLURM_STEP_NUM_NODES": "1",
        "SLURM_JOB_NODELIST": NODE, "SLURM_STEP_NODELIST": NODE,
    }
    if any(os.environ.get(key) != value for key, value in required_env.items()):
        raise StaticProbeError("Slurm allocation environment differs")
    step = os.environ.get("SLURM_STEP_ID", "")
    if not step.isascii() or not step.isdecimal() or int(step) <= 394 or str(int(step)) != step:
        raise StaticProbeError("fresh numeric Slurm step differs")
    if "SLURM_JOB_GPUS" in os.environ or "SLURM_JOB_NUM_NODES" in os.environ:
        raise StaticProbeError("forbidden synthesized Slurm names present")
    root = Path(args.root)
    expected_root = Path("/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_object_grounded_case01_0821_exact5_r64_canary_v1")
    if root != expected_root or root.is_symlink() or root.resolve(strict=True) != root:
        raise StaticProbeError("package root binding differs")
    release_root = root / "release"
    release_entries = list(release_root.rglob("*"))
    if any(
        path.is_symlink() or (not path.is_file() and not path.is_dir())
        for path in release_entries
    ):
        raise StaticProbeError("exact19 release contains a link or special entry")
    actual_release = {
        str(path.relative_to(release_root))
        for path in release_entries if path.is_file()
    }
    if actual_release != set(RELEASE_FILES):
        raise StaticProbeError("exact19 release closure differs")
    expected_release_directories = {"."}
    for relative in RELEASE_FILES:
        parent = Path(relative).parent
        while str(parent) != ".":
            expected_release_directories.add(str(parent))
            parent = parent.parent
    actual_release_directories = {"."} | {
        str(path.relative_to(release_root))
        for path in release_entries if path.is_dir()
    }
    if actual_release_directories != expected_release_directories:
        raise StaticProbeError("exact19 release directory closure differs")
    for relative in expected_release_directories:
        path = release_root if relative == "." else release_root / relative
        info = path.lstat()
        if (
            not stat.S_ISDIR(info.st_mode) or path.is_symlink()
            or info.st_uid != 2012 or info.st_gid != 2000
            or stat.S_IMODE(info.st_mode) != 0o555
        ):
            raise StaticProbeError("exact19 release directory authority differs")
    for relative, expected in RELEASE_FILES.items():
        stable(release_root / relative, expected, 0o444)
    method = release_root / "methods/bernini_action_editing"
    eval1 = load_module("full644_exploratory_matched_eval_v1", method / "full644_exploratory_matched_eval_v1.py", RELEASE_FILES["methods/bernini_action_editing/full644_exploratory_matched_eval_v1.py"])
    del eval1
    eval2 = load_module("full644_exploratory_matched_eval_v2", method / "full644_exploratory_matched_eval_v2.py", RELEASE_FILES["methods/bernini_action_editing/full644_exploratory_matched_eval_v2.py"])
    exact_eval = load_module("_exact5_static_eval", method / "case01_source_bone_exact5_eval_v1.py", RELEASE_FILES["methods/bernini_action_editing/case01_source_bone_exact5_eval_v1.py"])
    plan_path = root / "plan/case01_source_bone_exact5_r64_plan_v1.json"
    plan_raw = stable(plan_path, args.plan_sha256, 0o444)
    plan = exact_eval.load_plan(str(plan_path), args.plan_sha256)
    if plan.get("task_count") != 5 or [row.get("task_id") for row in plan.get("tasks", [])] != list(TASK_IDS):
        raise StaticProbeError("exact5 task closure differs")
    if eval2.validate_terminal_checkpoint_manifest(plan["checkpoint_manifest"]["path"], plan["checkpoint_manifest"]["sha256"]) != plan["checkpoint_manifest"]:
        raise StaticProbeError("terminal checkpoint replay differs")
    ffprobe_raw = stable(FFPROBE, FFPROBE_SHA256)
    ffprobe_mode = stat.S_IMODE(FFPROBE.lstat().st_mode)
    if not ffprobe_mode & 0o111 or ffprobe_mode & 0o022:
        raise StaticProbeError("ffprobe executable authority differs")
    if (
        plan.get("producer", {}).get("ffprobe_path") != str(FFPROBE)
        or plan.get("producer", {}).get("ffprobe_sha256") != FFPROBE_SHA256
        or os.path.lexists(str(RANK_CACHE_ROOT))
    ):
        raise StaticProbeError("ffprobe/rank-cache plan authority differs")
    launcher = load_module("_exact5_static_launcher", method / "case01_source_bone_exact5_spooled_launcher_auh_v1.py", RELEASE_FILES["methods/bernini_action_editing/case01_source_bone_exact5_spooled_launcher_auh_v1.py"])
    input_path = root / "launch/root_launch_input_exact5_v1.json"
    input_value, input_identity = launcher._load_input(str(input_path))
    rebuilt_release, rebuilt_payload = launcher.build_release(input_value)
    receipt_path = root / "launch/root_launch_receipt_exact5_v1.json"
    receipt_raw = stable(receipt_path, args.launch_receipt_sha256, 0o400)
    receipt = strict_json(receipt_raw)
    payload_path = root / "launch/root_launch_payload_exact5_v1.sh"
    payload_raw = stable(payload_path, receipt.get("payload_sha256"), 0o444)
    unsigned = dict(receipt); receipt_digest = unsigned.pop("receipt_digest", None)
    if (receipt_digest != digest(unsigned) or receipt.get("release") != rebuilt_release
            or receipt.get("release_digest") != digest(rebuilt_release)
            or receipt.get("launch_input") != input_identity
            or payload_raw != rebuilt_payload or receipt.get("payload_size") != len(payload_raw)):
        raise StaticProbeError("exact5 launch receipt/rebuild differs")
    for relative in ("outputs/media", "final", "runtime"):
        if list((root / relative).iterdir()):
            raise StaticProbeError(f"fresh result directory differs: {relative}")
    if "torch" in sys.modules:
        raise StaticProbeError("static probe imported torch")
    result: dict[str, Any] = {
        "schema_version": "case01-source-bone-exact5-static-probe-v1",
        "status": "PASS", "campaign_mode": CAMPAIGN,
        "holder_job_id": JOB_ID, "expected_node": NODE, "slurm_step_id": step,
        "task_count": 5, "selected_task_ids": list(TASK_IDS),
        "release_file_count": 19, "launch_identity_count": 18,
        "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "plan_digest": plan["plan_digest"],
        "independent_audit_sha256": plan["asset_authority"]["independent_audit_receipt_sha256"],
        "independent_audit_digest": plan["asset_authority"]["independent_audit_receipt_digest"],
        "checkpoint_manifest_sha256": plan["checkpoint_manifest"]["sha256"],
        "launch_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "launch_receipt_digest": receipt["receipt_digest"],
        "payload_sha256": hashlib.sha256(payload_raw).hexdigest(),
        "ffprobe_path": str(FFPROBE), "ffprobe_sha256": hashlib.sha256(ffprobe_raw).hexdigest(),
        "rank_cache_root": str(RANK_CACHE_ROOT),
        "production_outputs_fresh": True, "rank_cache_fresh": True,
        "pure_metadata_only": True, "torch_imported": False,
        "renderer_executed": False,
    }
    result["receipt_digest"] = digest(result)
    create_receipt(Path(args.receipt), result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--root", required=True); value.add_argument("--plan-sha256", required=True)
    value.add_argument("--launch-receipt-sha256", required=True); value.add_argument("--receipt", required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    if sys.platform != "linux" or not Path("/proc/self/fd").is_dir() or sys.flags.isolated != 1 or sys.flags.no_site != 1 or sys.flags.ignore_environment != 1 or not sys.dont_write_bytecode:
        raise StaticProbeError("isolated static-probe startup differs")
    result = run(parser().parse_args(argv))
    print("CASE01_EXACT5_STATIC_PASS " + result["receipt_digest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
