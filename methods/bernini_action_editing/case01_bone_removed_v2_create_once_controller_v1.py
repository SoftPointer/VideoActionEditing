#!/usr/bin/env python3
"""Create-once controller for the frozen case01 bone-removed-v2 chain.

This source is intentionally frozen in ``HOLD_PRE_IO``.  Every deployment-
specific authority and every output pathname is the literal string
``BLOCKED``.  The first branch in :func:`main` is a pure in-memory admission
check, so this version cannot stat, open, create, publish, spawn, or touch a
GPU.  There is no command-line escape hatch for those pins.

The unreachable execution body is complete enough to audit the future
transition: replay immutable authorities, reserve one immutable attempt,
invoke the frozen generator exactly once, construct a producer receipt and
    validate it with the frozen acceptance contract, then publish one same-parent
    bundle with Linux ``renameat2(RENAME_NOREPLACE)``.  Only after rename, parent
    fsync, and published-tree replay does it create the success publication
    receipt.  Failure never retries.  After generator invocation, a private
    staging tree is cleanup-eligible only when the caller holds the module's
    affirmative PGID-zero token; every unknown lifecycle escape is retained
    as quarantine.  Cleanup also requires the captured inode at the exact
    pinned staging name, and a published final tree is never removed.

This controller does not perform the later two-reviewer semantic acceptance.
It only runs the acceptance module's producer-receipt validator before
publication.  The produced candidate therefore remains pending independent
all-81-frame review and never authorizes a scientific claim.
"""

from __future__ import annotations

from argparse import Namespace
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import types
from typing import Any, Mapping, MutableMapping, Sequence


CASE_ID = "case01"
IID = "288545b9c031491a"
ROLE = "aux_bone_removed_source_v2"
FRAME_COUNT = 81
SEED = 2026082201

EXECUTION_STATE = "HOLD_PRE_IO"
ARMED_STATE = "ARMED_CREATE_ONCE"
BLOCKED = "BLOCKED"

GENERATOR_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit/methods/bernini_action_editing/"
    "generate_case01_bone_removed_v2_vace_v1.py"
)
GENERATOR_SHA256 = "f6dc4edb5ea3da03e14dd00399a800c3af545379bd0030aeab0fc8e2a205ce86"
GENERATOR_SIZE = 85_957
ACCEPTANCE_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit/methods/bernini_action_editing/tools/"
    "case01_bone_removed_v2_acceptance_v1.py"
)
ACCEPTANCE_SHA256 = "7a11c7bc2c0e37b8f00dfcb21da7755f57856a433166e8c978fd400bbde16c51"
ACCEPTANCE_SIZE = 148_113

SOURCE_SHA256 = "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
SOURCE_SIZE = 10_887_043
SAM2_RECEIPT_SHA256 = "c9dc8567d4f147f943171d2d7abb55e52aadc685fbfc2f75fff4e837e2ab0b50"
SAM2_RECEIPT_SIZE = 22_160
FFMPEG_PATH = (
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
    "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)
FFMPEG_SHA256 = "e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99"
FFMPEG_SIZE = 79_826_272
FFPROBE_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_graft_v1_20260810/runtime/"
    "ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe"
)
FFPROBE_SHA256 = "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5"
FFPROBE_SIZE = 216_841

TREE_MANIFEST_SCHEMA = "bernini-case01-bone-removed-v2-authority-tree-manifest-v1"
ATTEMPT_SCHEMA = "bernini-case01-bone-removed-v2-create-only-attempt-v1"
PUBLICATION_SCHEMA = "bernini-case01-bone-removed-v2-create-only-publication-v1"
PRODUCER_SCHEMA = "bernini-case01-bone-removed-v2-producer-receipt-v1"
CONTROLLER_EVIDENCE_SCHEMA = (
    "bernini-case01-bone-removed-v2-create-once-controller-evidence-v1"
)
MODEL_AUTHORITY_ROLES = (
    "python_runtime_tree",
    "vace_checkpoint_tree",
    "vace_source_tree",
)

# Every deployment value remains blocked in this source.  There is no CLI
# override.  A later reviewed source must replace the pins and change only the
# state line above before it can reach a filesystem operation.
DYNAMIC_PINS: Mapping[str, Any] = {
    # The controller cannot embed its own eventual SHA-256 without a circular
    # fixed-point problem.  Its exact deployment path is pinned; the running
    # program derives and retains the path/SHA/size row before the attempt and
    # replays the same bytes after generation.
    "controller_program_path": BLOCKED,
    "source": {"path": BLOCKED, "sha256": SOURCE_SHA256, "size": SOURCE_SIZE},
    "sam2_receipt": {
        "path": BLOCKED,
        "sha256": SAM2_RECEIPT_SHA256,
        "size": SAM2_RECEIPT_SIZE,
    },
    "support_review_receipt": {
        "path": BLOCKED,
        "sha256": BLOCKED,
        "size": BLOCKED,
        "review_digest": BLOCKED,
    },
    "python_runtime_manifest": {
        "path": BLOCKED,
        "sha256": BLOCKED,
        "size": BLOCKED,
        "tree_digest": BLOCKED,
        "tree_root": BLOCKED,
    },
    "vace_checkpoint_manifest": {
        "path": BLOCKED,
        "sha256": BLOCKED,
        "size": BLOCKED,
        "tree_digest": BLOCKED,
        "tree_root": BLOCKED,
    },
    "vace_source_manifest": {
        "path": BLOCKED,
        "sha256": BLOCKED,
        "size": BLOCKED,
        "tree_digest": BLOCKED,
        "tree_root": BLOCKED,
    },
    "python_bin": {"path": BLOCKED, "sha256": BLOCKED, "size": BLOCKED},
    "gpu_visible_device": BLOCKED,
    "child_environment": BLOCKED,
    "bundle_staging_root": BLOCKED,
    "bundle_final_root": BLOCKED,
    "asset_staging_root": BLOCKED,
    "evidence_staging_root": BLOCKED,
    "asset_final_root": BLOCKED,
    "evidence_final_root": BLOCKED,
    "attempt_receipt": BLOCKED,
    "publication_receipt": BLOCKED,
    "producer_receipt": BLOCKED,
}

STATIC_FILE_ROWS: Mapping[str, Mapping[str, Any]] = {
    "generator": {
        "path": GENERATOR_PATH,
        "sha256": GENERATOR_SHA256,
        "size": GENERATOR_SIZE,
    },
    "acceptance": {
        "path": ACCEPTANCE_PATH,
        "sha256": ACCEPTANCE_SHA256,
        "size": ACCEPTANCE_SIZE,
    },
    "ffmpeg": {
        "path": FFMPEG_PATH,
        "sha256": FFMPEG_SHA256,
        "size": FFMPEG_SIZE,
    },
    "ffprobe": {
        "path": FFPROBE_PATH,
        "sha256": FFPROBE_SHA256,
        "size": FFPROBE_SIZE,
    },
}

SHA256_HEX = frozenset("0123456789abcdef")
MAX_GENERATOR_STDOUT = 8 * 1024 * 1024
MAX_GENERATOR_STDERR = 2 * 1024 * 1024
GENERATOR_TIMEOUT_SECONDS = 6 * 60 * 60
PROCESS_POLL_SECONDS = 0.05
PROCESS_TERM_GRACE_SECONDS = 10.0
PROCESS_KILL_GRACE_SECONDS = 10.0
RENAME_NOREPLACE = 1
AT_FDCWD = -100
_PROCESS_GROUP_ZERO_PROVEN = object()


class ControllerHold(RuntimeError):
    """No mutation or attempt is authorized."""


class AttemptFailed(RuntimeError):
    """One reserved attempt failed and must never be retried."""


class ProcessGroupZeroUnproven(AttemptFailed):
    """The generator group may still be live; its staging tree is quarantined."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and set(value) <= SHA256_HEX
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ControllerHold(message)


def _exact_keys(value: Mapping[str, Any], keys: Sequence[str], label: str) -> None:
    _require(type(value) is dict and set(value) == set(keys), "%s keys differ" % label)


def blocked_pin_names(value: Any = DYNAMIC_PINS, prefix: str = "") -> tuple[str, ...]:
    """Purely enumerate literal blocked leaves without touching any authority."""

    rows: list[str] = []
    if value == BLOCKED:
        rows.append(prefix or "<root>")
    elif type(value) is dict:
        for key in sorted(value):
            child = "%s.%s" % (prefix, key) if prefix else key
            rows.extend(blocked_pin_names(value[key], child))
    elif type(value) in (list, tuple):
        for index, child_value in enumerate(value):
            rows.extend(blocked_pin_names(child_value, "%s[%d]" % (prefix, index)))
    return tuple(rows)


def admit_pre_io(
    *,
    state: str = EXECUTION_STATE,
    pins: Mapping[str, Any] = DYNAMIC_PINS,
) -> None:
    """The first runtime gate; intentionally pure and fail-closed."""

    blocked = blocked_pin_names(pins)
    reasons: list[str] = []
    if type(state) is not str or state != ARMED_STATE:
        reasons.append("state=%r" % (state,))
    if blocked:
        reasons.append("blocked_pins=" + ",".join(blocked))
    if reasons:
        raise ControllerHold("HOLD_BEFORE_IO: " + "; ".join(reasons))
    validate_dynamic_pins(pins)


def _canonical_absolute_text(value: Any, label: str) -> str:
    _require(type(value) is str and value and value != BLOCKED, "%s path differs" % label)
    _require(os.path.isabs(value), "%s is not absolute" % label)
    _require(os.path.normpath(value) == value, "%s is not canonical" % label)
    return value


def _validate_file_pin(value: Any, label: str) -> None:
    _exact_keys(value, ("path", "sha256", "size"), label)
    _canonical_absolute_text(value["path"], label)
    _require(_is_sha256(value["sha256"]), "%s SHA-256 differs" % label)
    _require(type(value["size"]) is int and value["size"] > 0, "%s size differs" % label)


def _validate_manifest_pin(value: Any, label: str) -> None:
    _exact_keys(value, ("path", "sha256", "size", "tree_digest", "tree_root"), label)
    _canonical_absolute_text(value["path"], label)
    _canonical_absolute_text(value["tree_root"], "%s tree root" % label)
    _require(_is_sha256(value["sha256"]), "%s SHA-256 differs" % label)
    _require(_is_sha256(value["tree_digest"]), "%s tree digest differs" % label)
    _require(type(value["size"]) is int and value["size"] > 0, "%s size differs" % label)


def validate_dynamic_pins(pins: Mapping[str, Any]) -> None:
    _exact_keys(
        pins,
        (
            "controller_program_path", "source", "sam2_receipt",
            "support_review_receipt", "python_runtime_manifest",
            "vace_checkpoint_manifest", "vace_source_manifest", "python_bin",
            "gpu_visible_device", "child_environment", "bundle_staging_root",
            "bundle_final_root", "asset_staging_root", "evidence_staging_root",
            "asset_final_root", "evidence_final_root", "attempt_receipt",
            "publication_receipt", "producer_receipt",
        ),
        "dynamic pins",
    )
    _canonical_absolute_text(pins["controller_program_path"], "controller program")
    for name in ("source", "sam2_receipt", "python_bin"):
        _validate_file_pin(pins[name], name)
    _require(
        pins["source"]["sha256"] == SOURCE_SHA256
        and pins["source"]["size"] == SOURCE_SIZE,
        "source fixed authority differs",
    )
    _require(
        pins["sam2_receipt"]["sha256"] == SAM2_RECEIPT_SHA256
        and pins["sam2_receipt"]["size"] == SAM2_RECEIPT_SIZE,
        "SAM2 fixed authority differs",
    )
    support = pins["support_review_receipt"]
    _exact_keys(support, ("path", "sha256", "size", "review_digest"), "support review")
    _canonical_absolute_text(support["path"], "support review")
    _require(_is_sha256(support["sha256"]), "support review SHA-256 differs")
    _require(_is_sha256(support["review_digest"]), "support review digest differs")
    _require(type(support["size"]) is int and support["size"] > 0, "support review size differs")
    for name in (
        "python_runtime_manifest", "vace_checkpoint_manifest", "vace_source_manifest",
    ):
        _validate_manifest_pin(pins[name], name)
    device = pins["gpu_visible_device"]
    _require(type(device) is str and device.isdecimal() and "," not in device, "GPU device pin differs")
    environment = pins["child_environment"]
    _require(type(environment) is dict and environment, "child environment pin differs")
    required_environment = {
        "PYTHONHASHSEED": "20260822",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "LC_ALL": "C",
        "LANG": "C",
        "TZ": "UTC",
    }
    _require(
        all(type(key) is str and type(value) is str for key, value in environment.items()),
        "child environment types differ",
    )
    _require(
        all(environment.get(key) == value for key, value in required_environment.items()),
        "child environment fixed values differ",
    )
    _require(
        set(environment) <= set(required_environment) | {"LD_LIBRARY_PATH", "PATH"},
        "child environment has an unapproved key",
    )

    path_names = (
        "bundle_staging_root", "bundle_final_root", "asset_staging_root",
        "evidence_staging_root", "asset_final_root", "evidence_final_root",
        "attempt_receipt", "publication_receipt", "producer_receipt",
    )
    paths = {name: Path(_canonical_absolute_text(pins[name], name)) for name in path_names}
    _require(len({str(path) for path in paths.values()}) == len(paths), "output paths repeat")
    _require(
        paths["bundle_staging_root"].parent == paths["bundle_final_root"].parent,
        "bundle roots are not same-parent",
    )
    _require(
        paths["asset_staging_root"] == paths["bundle_staging_root"] / "assets"
        and paths["evidence_staging_root"] == paths["bundle_staging_root"] / "evidence",
        "staging child roots differ",
    )
    _require(
        paths["asset_final_root"] == paths["bundle_final_root"] / "assets"
        and paths["evidence_final_root"] == paths["bundle_final_root"] / "evidence",
        "final child roots differ",
    )
    _require(
        paths["publication_receipt"]
        == paths["evidence_final_root"] / "create_only_publication.json"
        and paths["producer_receipt"]
        == paths["evidence_final_root"] / "producer_receipt.json",
        "final receipt paths differ",
    )
    for name in ("attempt_receipt",):
        path = paths[name]
        for root_name in ("bundle_staging_root", "bundle_final_root"):
            try:
                path.relative_to(paths[root_name])
            except ValueError:
                continue
            raise ControllerHold("%s is inside %s" % (name, root_name))


def _parse_canonical_object(payload: bytes, label: str) -> Mapping[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, "duplicate JSON key in %s" % label)
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ControllerHold("non-finite JSON constant in %s: %s" % (label, value))

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControllerHold("invalid JSON: %s" % label) from error
    _require(type(value) is dict, "%s JSON root differs" % label)
    _require(payload == canonical_json_bytes(value) + b"\n", "%s JSON is not canonical one-LF" % label)
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )


def _read_stable_file(path_value: str, *, nlink1: bool = False) -> tuple[bytes, Mapping[str, Any]]:
    path = Path(_canonical_absolute_text(path_value, "authority file"))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), "authority is not regular: %s" % path)
        _require(not nlink1 or before.st_nlink == 1, "authority is not nlink1: %s" % path)
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        after = os.fstat(descriptor)
        named = path.lstat()
        _require(
            _stat_identity(before) == _stat_identity(after) == _stat_identity(named),
            "authority changed while read: %s" % path,
        )
    finally:
        os.close(descriptor)
    payload = b"".join(blocks)
    return payload, {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def _verify_file_pin(value: Mapping[str, Any], label: str, *, nlink1: bool = False) -> bytes:
    payload, observed = _read_stable_file(value["path"], nlink1=nlink1)
    _require(observed == {key: value[key] for key in ("path", "sha256", "size")}, "%s bytes differ" % label)
    return payload


def _load_frozen_module(row: Mapping[str, Any], module_name: str) -> types.ModuleType:
    payload = _verify_file_pin(row, module_name)
    module = types.ModuleType(module_name)
    module.__file__ = row["path"]
    module.__package__ = ""
    code = compile(payload, row["path"], "exec", dont_inherit=True, optimize=0)
    exec(code, module.__dict__)
    return module


def _generator_namespace(pins: Mapping[str, Any]) -> Namespace:
    return Namespace(
        source_video=Path(pins["source"]["path"]),
        sam2_receipt=Path(pins["sam2_receipt"]["path"]),
        support_review_receipt=Path(pins["support_review_receipt"]["path"]),
        vace_source_manifest=Path(pins["vace_source_manifest"]["path"]),
        vace_checkpoint_manifest=Path(pins["vace_checkpoint_manifest"]["path"]),
        python_runtime_manifest=Path(pins["python_runtime_manifest"]["path"]),
        python_bin=Path(pins["python_bin"]["path"]),
        vace_root=Path(pins["vace_source_manifest"]["tree_root"]),
        vace_checkpoint_root=Path(pins["vace_checkpoint_manifest"]["tree_root"]),
        ffmpeg=Path(FFMPEG_PATH),
        ffprobe=Path(FFPROBE_PATH),
        acceptance_contract=Path(ACCEPTANCE_PATH),
        seed=SEED,
    )


def _verify_manifest_pin(module: Any, pin: Mapping[str, Any], role: str) -> None:
    payload = _verify_file_pin(pin, "%s manifest" % role, nlink1=True)
    manifest = _parse_canonical_object(payload, "%s manifest" % role)
    _require(manifest.get("schema_version") == TREE_MANIFEST_SCHEMA, "%s schema differs" % role)
    _require(manifest.get("authority_role") == role, "%s role differs" % role)
    _require(manifest.get("tree_root") == pin["tree_root"], "%s root differs" % role)
    _require(manifest.get("tree_digest") == pin["tree_digest"], "%s tree digest differs" % role)
    replay = module.replay_tree_manifest(pin["path"], role)
    _require(replay["tree_digest"] == pin["tree_digest"], "%s replay digest differs" % role)


def _verify_fresh_topology(pins: Mapping[str, Any]) -> None:
    bundle_stage = Path(pins["bundle_staging_root"])
    bundle_final = Path(pins["bundle_final_root"])
    parent = bundle_stage.parent
    _require(not parent.is_symlink(), "bundle parent is symlink")
    _require(parent.resolve(strict=True) == parent and parent.is_dir(), "bundle parent differs")
    receipt_parent = Path(pins["attempt_receipt"]).parent
    _require(not receipt_parent.is_symlink(), "attempt parent is symlink")
    _require(
        receipt_parent.resolve(strict=True) == receipt_parent and receipt_parent.is_dir(),
        "attempt parent differs",
    )
    targets = (
        "bundle_staging_root", "bundle_final_root", "asset_staging_root",
        "evidence_staging_root", "asset_final_root", "evidence_final_root",
        "attempt_receipt", "publication_receipt", "producer_receipt",
    )
    _require(
        all(not os.path.lexists(pins[name]) for name in targets),
        "one or more create-only targets already exist",
    )
    _require(bundle_stage.parent == bundle_final.parent, "bundle roots lost same-parent topology")


def _preflight(
    pins: Mapping[str, Any],
) -> tuple[Any, Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    generator = _load_frozen_module(STATIC_FILE_ROWS["generator"], "frozen_bone_removed_v2_generator")
    acceptance = _load_frozen_module(STATIC_FILE_ROWS["acceptance"], "frozen_bone_removed_v2_acceptance")
    _verify_file_pin(STATIC_FILE_ROWS["ffmpeg"], "ffmpeg")
    _verify_file_pin(STATIC_FILE_ROWS["ffprobe"], "ffprobe")
    controller_payload, controller_row = _read_stable_file(
        pins["controller_program_path"], nlink1=True
    )
    _require(controller_payload, "controller program is empty")
    _require(
        controller_row["path"] == str(Path(__file__).resolve(strict=True)),
        "running controller path differs from deployment pin",
    )
    _verify_file_pin(pins["source"], "source")
    sam2_payload = _verify_file_pin(pins["sam2_receipt"], "SAM2 receipt")
    _parse_canonical_object(sam2_payload, "SAM2 receipt")
    support_payload = _verify_file_pin(pins["support_review_receipt"], "support review", nlink1=True)
    support = _parse_canonical_object(support_payload, "support review")
    _require(
        support.get("review_digest") == pins["support_review_receipt"]["review_digest"],
        "support review digest pin differs",
    )
    _verify_file_pin(pins["python_bin"], "Python executable", nlink1=True)
    role_pins = {
        "python_runtime_tree": pins["python_runtime_manifest"],
        "vace_checkpoint_tree": pins["vace_checkpoint_manifest"],
        "vace_source_tree": pins["vace_source_manifest"],
    }
    for role in MODEL_AUTHORITY_ROLES:
        _verify_manifest_pin(generator, role_pins[role], role)
    result = generator.preflight(_generator_namespace(pins))
    _require(result["status"] == "PASS_AUTHORITY_PREFLIGHT_NO_OUTPUT_CREATED", "generator preflight differs")
    expected_models = [
        {
            "role": role,
            "path": role_pins[role]["path"],
            "sha256": role_pins[role]["sha256"],
            "size": role_pins[role]["size"],
        }
        for role in MODEL_AUTHORITY_ROLES
    ]
    _require(result["model_authorities"] == expected_models, "preflight model rows differ")
    _require(result["source"] == dict(pins["source"]), "preflight source differs")
    _require(result["sam2_receipt"] == dict(pins["sam2_receipt"]), "preflight SAM2 differs")
    _require(
        result["support_review_receipt"]
        == {key: pins["support_review_receipt"][key] for key in ("path", "sha256", "size")},
        "preflight support review differs",
    )
    _require(result["acceptance_contract"] == dict(STATIC_FILE_ROWS["acceptance"]), "acceptance pin differs")
    _require(result["generation_execution_lineage_verified"] is False, "preflight overclaims lineage")
    _verify_fresh_topology(pins)
    return generator, acceptance, result, support, controller_row


def _write_create_only(path_value: str | Path, payload: bytes, mode: int = 0o400) -> Mapping[str, Any]:
    path = Path(path_value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, mode)
    try:
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0:
                raise AttemptFailed("short create-only write: %s" % path)
            offset += count
        os.fsync(descriptor)
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise AttemptFailed("create-only output topology differs: %s" % path)
    finally:
        os.close(descriptor)
    named = path.lstat()
    if not stat.S_ISREG(named.st_mode) or named.st_nlink != 1:
        raise AttemptFailed("create-only named output differs: %s" % path)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _build_attempt(
    pins: Mapping[str, Any],
    preflight: Mapping[str, Any],
    controller_sha256: str,
) -> MutableMapping[str, Any]:
    token_payload = {
        "case_id": CASE_ID,
        "iid": IID,
        "controller_program_sha256": controller_sha256,
        "generator_program_sha256": GENERATOR_SHA256,
        "dynamic_pins_digest": object_sha256(pins),
        "authority_replay_digest": preflight["authority_replay_digest"],
    }
    value: MutableMapping[str, Any] = {
        "schema_version": ATTEMPT_SCHEMA,
        "status": "RESERVED_FRESH_BEFORE_GENERATION",
        "case_id": CASE_ID,
        "iid": IID,
        "attempt_token": object_sha256(token_payload),
        "controller_program_sha256": controller_sha256,
        "generator_program_sha256": GENERATOR_SHA256,
        "model_authorities_digest": object_sha256(preflight["model_authorities"]),
        "final_root": pins["asset_final_root"],
        "staging_root": pins["asset_staging_root"],
        "preflight": {
            "performed_before_generation": True,
            "final_root_absent": True,
            "staging_root_absent": True,
            "all_target_paths_absent": True,
            "reservation_create_only": True,
        },
    }
    value["attempt_digest"] = object_sha256(value)
    return value


def _generator_argv(pins: Mapping[str, Any]) -> list[str]:
    return [
        pins["python_bin"]["path"], GENERATOR_PATH, "run",
        "--source-video", pins["source"]["path"],
        "--sam2-receipt", pins["sam2_receipt"]["path"],
        "--support-review-receipt", pins["support_review_receipt"]["path"],
        "--vace-source-manifest", pins["vace_source_manifest"]["path"],
        "--vace-checkpoint-manifest", pins["vace_checkpoint_manifest"]["path"],
        "--python-runtime-manifest", pins["python_runtime_manifest"]["path"],
        "--python-bin", pins["python_bin"]["path"],
        "--vace-root", pins["vace_source_manifest"]["tree_root"],
        "--vace-checkpoint-root", pins["vace_checkpoint_manifest"]["tree_root"],
        "--ffmpeg", FFMPEG_PATH,
        "--ffprobe", FFPROBE_PATH,
        "--acceptance-contract", ACCEPTANCE_PATH,
        "--seed", str(SEED),
        "--asset-staging-root", pins["asset_staging_root"],
        "--evidence-staging-root", pins["evidence_staging_root"],
        "--asset-final-root", pins["asset_final_root"],
        "--evidence-final-root", pins["evidence_final_root"],
        "--gpu-visible-device", pins["gpu_visible_device"],
    ]


def _process_group_present(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        if error.errno == errno.ESRCH:
            return False
        if error.errno == errno.EPERM:
            return True
        raise AttemptFailed("generator process-group probe differs") from error


def _signal_process_group(process_group: int, signal_number: int) -> None:
    try:
        os.killpg(process_group, signal_number)
    except ProcessLookupError:
        return
    except PermissionError:
        return
    except OSError as error:
        if error.errno in (errno.ESRCH, errno.EPERM):
            return
        raise AttemptFailed("generator process-group signal differs") from error


def _poll_group_absent(
    process: subprocess.Popen[bytes], process_group: int, deadline: float,
) -> bool:
    while True:
        # Always reap the owned leader while probing the PGID saved at spawn.
        # A leader that already exited does not prove that its descendants did.
        process.poll()
        if not _process_group_present(process_group):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(PROCESS_POLL_SECONDS)


def _process_group_absent(process_group: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if not _process_group_present(process_group):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(PROCESS_POLL_SECONDS)


def _close_process_pipes(process: subprocess.Popen[bytes]) -> None:
    errors: list[BaseException] = []
    for pipe in (process.stdin, process.stdout, process.stderr):
        if pipe is not None and not pipe.closed:
            try:
                pipe.close()
            except BaseException as error:
                errors.append(error)
    if errors:
        raise AttemptFailed("generator terminal pipe close differs") from errors[0]


def _seal_process_group(
    process: subprocess.Popen[bytes], process_group: int,
) -> None:
    """Enforce pipe closure, TERM/KILL, direct reap, and saved-PGID ESRCH."""

    pipe_error: BaseException | None = None
    try:
        _close_process_pipes(process)
    except BaseException as error:
        pipe_error = error

    try:
        _signal_process_group(process_group, signal.SIGTERM)
        term_deadline = time.monotonic() + PROCESS_TERM_GRACE_SECONDS
        _poll_group_absent(process, process_group, term_deadline)
        if _process_group_present(process_group):
            _signal_process_group(process_group, signal.SIGKILL)

        try:
            process.wait(timeout=PROCESS_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=PROCESS_KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired as error:
                raise ProcessGroupZeroUnproven(
                    "generator direct child did not reap"
                ) from error

        kill_deadline = time.monotonic() + PROCESS_KILL_GRACE_SECONDS
        if not _poll_group_absent(process, process_group, kill_deadline):
            raise ProcessGroupZeroUnproven(
                "generator process group did not reach ESRCH"
            )
        if process.poll() is None:
            raise ProcessGroupZeroUnproven(
                "generator direct child remains unreaped"
            )
    except ProcessGroupZeroUnproven:
        raise
    except BaseException as error:
        raise ProcessGroupZeroUnproven(
            "generator process-group zero proof differs"
        ) from error
    if pipe_error is not None:
        raise AttemptFailed("generator terminal pipe seal differs") from pipe_error


def _run_generator_once(
    pins: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], object]:
    argv = _generator_argv(pins)
    environment = dict(pins["child_environment"])
    process: subprocess.Popen[bytes] | None = None
    process_group: int | None = None
    process_group_zero = False
    stdout = b""
    stderr = b""
    try:
        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                close_fds=True,
                start_new_session=True,
            )
            # start_new_session=True establishes PGID==PID before exec.  Save
            # it immediately.  The surrounding finally covers every bytecode
            # edge after a successful spawn until the saved PGID reaches
            # ESRCH, including asynchronous BaseException delivery.
            process_group = process.pid
            stdout, stderr = process.communicate(
                timeout=GENERATOR_TIMEOUT_SECONDS
            )
            # A terminal leader can precede short-lived helpers.  Give the
            # saved group a bounded, signal-free chance to disappear.
            naturally_absent = _process_group_absent(
                process_group,
                PROCESS_TERM_GRACE_SECONDS,
            )
            if not naturally_absent:
                raise AttemptFailed(
                    "terminal generator process group required cleanup "
                    "after passive grace"
                )
            # This is the sole transition that releases the lifecycle guard.
            # Later parsing or pipe-accounting failures cannot race a live
            # generator group because ESRCH has already been observed.
            process_group_zero = True
        finally:
            if process is not None and not process_group_zero:
                active_error = sys.exc_info()[1]
                if process_group is None:
                    try:
                        process_group = process.pid
                    except BaseException as identity_error:
                        raise ProcessGroupZeroUnproven(
                            "generator saved process group is unavailable"
                        ) from (
                            active_error
                            if active_error is not None
                            else identity_error
                        )
                try:
                    _seal_process_group(process, process_group)
                except ProcessGroupZeroUnproven as cleanup_error:
                    raise ProcessGroupZeroUnproven(
                        "generator process/pipe zero gate is unproven"
                    ) from (
                        active_error
                        if active_error is not None
                        else cleanup_error
                    )
                except AttemptFailed:
                    # _seal_process_group raises AttemptFailed only after its
                    # saved-PGID ESRCH proof, for terminal pipe-accounting
                    # failures.  Staging cleanup is therefore safe.
                    raise
                except BaseException as cleanup_error:
                    raise ProcessGroupZeroUnproven(
                        "generator process/pipe zero gate is unproven"
                    ) from (
                        active_error
                        if active_error is not None
                        else cleanup_error
                    )
                process_group_zero = True
    except subprocess.TimeoutExpired as error:
        raise AttemptFailed("single generator attempt timed out") from error

    # ``process_group_zero`` can become true only through passive ESRCH above
    # or a successful guarded seal.  The latter always accompanies an error,
    # so only the passive path reaches this success accounting.
    if process is None or process_group is None or not process_group_zero:
        raise ProcessGroupZeroUnproven("generator lifecycle guard differs")
    _close_process_pipes(process)
    if process.poll() is None or process.returncode is None:
        raise AttemptFailed("generator direct child remains unreaped")
    if any(
        pipe is not None and not pipe.closed
        for pipe in (process.stdin, process.stdout, process.stderr)
    ):
        raise AttemptFailed("generator terminal pipe closure differs")

    if len(stdout) > MAX_GENERATOR_STDOUT or len(stderr) > MAX_GENERATOR_STDERR:
        raise AttemptFailed("generator output exceeded the frozen bound")
    if process.returncode != 0:
        tail = stderr[-4096:].decode("utf-8", errors="replace")
        raise AttemptFailed("single generator attempt failed rc=%d: %s" % (process.returncode, tail))
    result = _parse_canonical_object(stdout, "generator result")
    evidence = {
        "exact_argv": argv,
        "exact_environment": environment,
        "return_code": process.returncode,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stdout_size": len(stdout),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stderr_size": len(stderr),
        "stdout_is_canonical_generator_result": True,
        "direct_generator_child_invocations": 1,
        "start_new_session": True,
        "saved_process_group_id": process_group,
        "normal_exit_passive_grace_performed": True,
        "normal_exit_signal_sent": False,
        "terminal_pipes_closed": True,
        "process_group_zero": True,
        "automatic_retry_performed": False,
    }
    return result, evidence, _PROCESS_GROUP_ZERO_PROVEN


def _replace_stage_row(row: Mapping[str, Any], stage_root: Path, final_root: Path) -> Mapping[str, Any]:
    _exact_keys(row, ("path", "sha256", "size"), "staged asset row")
    staged = Path(_canonical_absolute_text(row["path"], "staged asset"))
    try:
        relative = staged.relative_to(stage_root)
    except ValueError as error:
        raise ControllerHold("staged asset escapes asset root") from error
    _require(relative.parts and relative != Path("."), "staged asset relative path differs")
    _verify_file_pin(row, "staged asset", nlink1=True)
    return {"path": str(final_root / relative), "sha256": row["sha256"], "size": row["size"]}


def _build_publication(
    pins: Mapping[str, Any],
    attempt: Mapping[str, Any],
    controller_sha256: str,
    assets: Mapping[str, Mapping[str, Any]],
) -> MutableMapping[str, Any]:
    value: MutableMapping[str, Any] = {
        "schema_version": PUBLICATION_SCHEMA,
        "status": "PUBLISHED_FRESH_NO_REPLACE",
        "case_id": CASE_ID,
        "iid": IID,
        "attempt_token": attempt["attempt_token"],
        "controller_program_sha256": controller_sha256,
        "final_root": pins["asset_final_root"],
        "staging_root": pins["asset_staging_root"],
        "published_assets": dict(assets),
        "publication": {
            "atomic_rename_noreplace": True,
            "final_root_absent_before_publish": True,
            "overwrite_performed": False,
            "staging_removed_after_publish": True,
            "published_tree_regular_nonsymlink_nlink1": True,
            "directory_fsync_performed": True,
        },
    }
    value["publication_digest"] = object_sha256(value)
    return value


def _future_file_row(path: str, payload: bytes) -> Mapping[str, Any]:
    return {"path": path, "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}


def _build_producer(
    pins: Mapping[str, Any],
    result: Mapping[str, Any],
    controller_row: Mapping[str, Any],
    attempt_row: Mapping[str, Any],
    publication_row: Mapping[str, Any],
    assets: Mapping[str, Mapping[str, Any]],
) -> MutableMapping[str, Any]:
    value: MutableMapping[str, Any] = {
        "schema_version": PRODUCER_SCHEMA,
        "status": "COMPLETE_CANDIDATE_PENDING_INDEPENDENT_ACCEPTANCE",
        "case_id": CASE_ID,
        "iid": IID,
        "role": ROLE,
        "source": dict(pins["source"]),
        "mask_authority": {
            "receipt": dict(pins["sam2_receipt"]),
            "bone_mask_count": FRAME_COUNT,
            "dog_mask_count": FRAME_COUNT,
            "all_81_masks_hash_bound": True,
        },
        "media_tools": {
            "ffmpeg": dict(STATIC_FILE_ROWS["ffmpeg"]),
            "ffprobe": dict(STATIC_FILE_ROWS["ffprobe"]),
        },
        "acceptance_contract": dict(STATIC_FILE_ROWS["acceptance"]),
        "generator": result["generator_fragment"],
        "support": {
            "tube": dict(assets["support"]),
            "definition": "per_frame_bone_plus_cast_shadow_support_v2",
            "frame_count": FRAME_COUNT,
            "contains_bone_and_cast_shadow_all_frames": True,
            "all_81_frames_manually_reviewed": True,
            "old_dilate3_tube_reused": False,
            "review_receipt": result["support_review_receipt"],
            "frame_masks": result["support_frame_masks"],
        },
        "canonical_candidate": {
            "video": dict(assets["canonical_candidate"]),
            "codec": "ffv1",
            "lossless": True,
            "stored_pixel_format": "bgr0",
            "decoded_pixel_format": "rgb24",
            "frame_count": FRAME_COUNT,
        },
        "delivery_candidate": {
            "video": dict(assets["delivery_candidate"]),
            "codec": "h264",
            "pixel_format": "yuv420p",
            "frame_count": FRAME_COUNT,
            "derived_only_from_canonical": True,
            **result["delivery_contract"],
        },
        "construction_audit": result["construction_audit"],
        "create_only_authority": {
            "controller_program": dict(controller_row),
            "attempt_receipt": dict(attempt_row),
            "publication_receipt": dict(publication_row),
            "controller_distinct_from_generator": True,
            "fresh_root": True,
            "existing_path_reused": False,
            "overwrite_performed": False,
            "atomic_publish": True,
            "staging_removed_after_publish": True,
        },
        "claim_limits": result["claim_limits"],
    }
    value["receipt_digest"] = object_sha256(value)
    return value


def _raise_walk_error(error: OSError) -> None:
    """Turn every ``os.walk`` scandir failure into a closed attempt."""

    location = error.filename if error.filename is not None else "<unknown>"
    raise AttemptFailed("filesystem tree walk failed: %s" % location) from error


def _replay_create_only_receipts(
    acceptance: types.ModuleType,
    attempt: Mapping[str, Any],
    publication: Mapping[str, Any],
    producer: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Run the frozen cross-receipt replay and authenticate its exact result."""

    try:
        replay = acceptance.validate_create_only_receipts(
            attempt,
            publication,
            producer=producer,
        )
    except Exception as error:
        raise AttemptFailed("frozen create-only receipt replay failed") from error
    expected = {
        "attempt_token": attempt["attempt_token"],
        "final_root": attempt["final_root"],
        "staging_root": attempt["staging_root"],
    }
    if type(replay) is not dict or replay != expected:
        raise AttemptFailed("frozen create-only receipt replay result differs")
    return replay


def _scan_regular_tree(root: Path) -> tuple[str, ...]:
    observed: list[str] = []
    for directory, dirnames, filenames in os.walk(
        root,
        topdown=True,
        onerror=_raise_walk_error,
        followlinks=False,
    ):
        dirnames.sort()
        filenames.sort()
        directory_path = Path(directory)
        row = directory_path.lstat()
        _require(stat.S_ISDIR(row.st_mode) and not directory_path.is_symlink(), "tree directory differs")
        for dirname in dirnames:
            child = directory_path / dirname
            child_row = child.lstat()
            _require(stat.S_ISDIR(child_row.st_mode) and not child.is_symlink(), "tree contains symlink/special")
        for filename in filenames:
            child = directory_path / filename
            child_row = child.lstat()
            _require(
                stat.S_ISREG(child_row.st_mode) and child_row.st_nlink == 1 and not child.is_symlink(),
                "tree file topology differs: %s" % child,
            )
            observed.append(child.relative_to(root).as_posix())
    return tuple(observed)


def _fsync_tree(root: Path) -> None:
    directories: list[Path] = []
    for directory, dirnames, filenames in os.walk(
        root,
        topdown=False,
        onerror=_raise_walk_error,
        followlinks=False,
    ):
        dirnames.sort()
        filenames.sort()
        directory_path = Path(directory)
        directories.append(directory_path)
        for filename in filenames:
            path = directory_path / filename
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in directories:
        _fsync_directory(directory)


def _rename_noreplace(source: Path, destination: Path) -> None:
    _require(source.parent == destination.parent, "atomic publication is not same-parent")
    library = ctypes.CDLL(None, use_errno=True)
    function = getattr(library, "renameat2", None)
    if function is None:
        raise AttemptFailed("renameat2(RENAME_NOREPLACE) is unavailable")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    result = function(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise AttemptFailed("create-only final bundle already exists")
        raise AttemptFailed("renameat2(RENAME_NOREPLACE) failed: %s" % os.strerror(number))


def _publish_bundle(stage: Path, final: Path, expected_identity: tuple[int, int]) -> None:
    _require(stage.parent == final.parent, "publication roots are not same-parent")
    before = stage.lstat()
    _require((before.st_dev, before.st_ino) == expected_identity, "staging inode changed before publication")
    _require(not os.path.lexists(final), "final bundle appeared before publication")
    _scan_regular_tree(stage)
    _fsync_tree(stage)
    _rename_noreplace(stage, final)
    _fsync_directory(final.parent)
    after = final.lstat()
    _require((after.st_dev, after.st_ino) == expected_identity, "published bundle inode differs")
    _require(not os.path.lexists(stage), "staging bundle remains after publication")


def _cleanup_private_stage(stage: Path, expected_identity: tuple[int, int]) -> None:
    if not os.path.lexists(stage):
        return
    row = stage.lstat()
    if not stat.S_ISDIR(row.st_mode) or stage.is_symlink():
        raise AttemptFailed("refusing cleanup of changed staging topology")
    if (row.st_dev, row.st_ino) != expected_identity:
        raise AttemptFailed("refusing cleanup of changed staging inode")
    for directory, dirnames, filenames in os.walk(
        stage,
        topdown=False,
        onerror=_raise_walk_error,
        followlinks=False,
    ):
        directory_path = Path(directory)
        for filename in filenames:
            child = directory_path / filename
            child_row = child.lstat()
            if not stat.S_ISREG(child_row.st_mode) or child.is_symlink():
                raise AttemptFailed("refusing cleanup of special staging entry")
            child.unlink()
        for dirname in dirnames:
            child = directory_path / dirname
            child_row = child.lstat()
            if not stat.S_ISDIR(child_row.st_mode) or child.is_symlink():
                raise AttemptFailed("refusing cleanup of special staging directory")
            child.rmdir()
    stage.rmdir()
    _fsync_directory(stage.parent)


def execute_create_once(pins: Mapping[str, Any] = DYNAMIC_PINS) -> Mapping[str, Any]:
    """Execute one reserved attempt.  Current source cannot reach this body."""

    generator, acceptance, preflight, _support, controller_row = _preflight(pins)
    attempt = _build_attempt(pins, preflight, controller_row["sha256"])
    attempt_payload = canonical_json_bytes(attempt) + b"\n"
    attempt_row = _write_create_only(pins["attempt_receipt"], attempt_payload, 0o400)
    _fsync_directory(Path(pins["attempt_receipt"]).parent)

    bundle_stage = Path(pins["bundle_staging_root"])
    bundle_final = Path(pins["bundle_final_root"])
    os.mkdir(bundle_stage, 0o700)
    stage_stat = bundle_stage.lstat()
    stage_identity = (stage_stat.st_dev, stage_stat.st_ino)
    published = False
    generator_invocation_started = False
    process_group_zero_token: object | None = None
    try:
        os.mkdir(pins["asset_staging_root"], 0o700)
        os.mkdir(pins["evidence_staging_root"], 0o700)
        # From this assignment onward, staging is quarantined by default.  It
        # becomes cleanup-eligible only after the call returns the private,
        # affirmative PGID-zero capability.  This inversion covers async
        # exceptions at Popen CALL->STORE, inside lifecycle-finally cleanup,
        # and at this call's own CALL->UNPACK/STORE bytecode edges.
        generator_invocation_started = True
        result, child_evidence, returned_zero_token = _run_generator_once(pins)
        if returned_zero_token is not _PROCESS_GROUP_ZERO_PROVEN:
            raise ProcessGroupZeroUnproven(
                "generator returned no affirmative process-group-zero token"
            )
        process_group_zero_token = returned_zero_token
        required_result_keys = {
            "generator_fragment", "support_review_receipt", "support_frame_masks",
            "support_tube_stage", "canonical_stage", "delivery_stage",
            "delivery_contract", "construction_audit", "claim_limits",
            "execution_evidence",
        }
        _require(set(result) == required_result_keys, "generator result keys differ")
        _require(
            result["generator_fragment"]["model_authorities"]
            == preflight["model_authorities"],
            "generator result model authorities differ from reserved attempt",
        )
        _require(
            result["support_review_receipt"]
            == preflight["support_review_receipt"]
            and result["support_frame_masks"] == preflight["support_frame_masks"],
            "generator result support authority differs from preflight",
        )
        asset_stage = Path(pins["asset_staging_root"])
        asset_final = Path(pins["asset_final_root"])
        assets = {
            "support": _replace_stage_row(result["support_tube_stage"], asset_stage, asset_final),
            "canonical_candidate": _replace_stage_row(result["canonical_stage"], asset_stage, asset_final),
            "delivery_candidate": _replace_stage_row(result["delivery_stage"], asset_stage, asset_final),
        }
        _require(
            set(_scan_regular_tree(asset_stage))
            == {
                Path(row["path"]).relative_to(asset_final).as_posix()
                for row in assets.values()
            },
            "asset staging tree is not exact-three",
        )
        generation_payload = canonical_json_bytes(result["execution_evidence"]) + b"\n"
        _write_create_only(
            Path(pins["evidence_staging_root"]) / "generation_evidence.json",
            generation_payload,
            0o400,
        )
        controller_evidence: MutableMapping[str, Any] = {
            "schema_version": CONTROLLER_EVIDENCE_SCHEMA,
            "status": "COMPLETE_SINGLE_GENERATOR_CHILD_BOUND_TO_CREATE_ONLY_BUNDLE",
            "case_id": CASE_ID,
            "iid": IID,
            "attempt_token": attempt["attempt_token"],
            "controller_program": dict(controller_row),
            "generator_program": dict(STATIC_FILE_ROWS["generator"]),
            "acceptance_contract": dict(STATIC_FILE_ROWS["acceptance"]),
            "model_authorities_digest": attempt["model_authorities_digest"],
            "authority_replay_digest": preflight["authority_replay_digest"],
            "child_process": dict(child_evidence),
            "generator_result_sha256": hashlib.sha256(
                canonical_json_bytes(result) + b"\n"
            ).hexdigest(),
            "generator_result_size": len(canonical_json_bytes(result) + b"\n"),
            "prospective_published_assets": {key: dict(value) for key, value in assets.items()},
            "bundle_staging_root": pins["bundle_staging_root"],
            "bundle_final_root": pins["bundle_final_root"],
            "same_parent_rename_noreplace_required": True,
            "controller_attestation_is_external_trust_root": False,
            "generation_execution_lineage_verified": False,
            "semantic_acceptance_performed": False,
            "scientific_claim_authorized": False,
        }
        controller_evidence["evidence_digest"] = object_sha256(controller_evidence)
        controller_evidence_payload = canonical_json_bytes(controller_evidence) + b"\n"
        controller_evidence_stage_path = (
            Path(pins["evidence_staging_root"]) / "controller_execution_evidence.json"
        )
        controller_evidence_stage_row = _write_create_only(
            controller_evidence_stage_path,
            controller_evidence_payload,
            0o400,
        )
        controller_evidence_final_row = {
            "path": str(
                Path(pins["evidence_final_root"]) / "controller_execution_evidence.json"
            ),
            "sha256": controller_evidence_stage_row["sha256"],
            "size": controller_evidence_stage_row["size"],
        }

        publication = _build_publication(
            pins,
            attempt,
            controller_row["sha256"],
            assets,
        )
        publication_payload = canonical_json_bytes(publication) + b"\n"
        future_publication_row = _future_file_row(pins["publication_receipt"], publication_payload)
        producer = _build_producer(
            pins,
            result,
            controller_row,
            attempt_row,
            future_publication_row,
            assets,
        )
        # This is the frozen acceptance contract's structural producer gate,
        # not the later human-observation/media acceptance.
        acceptance.validate_producer_receipt(producer)

        # The producer may bind the exact future publication-receipt bytes, but
        # the success receipt itself must not be visible before rename, parent
        # fsync, and post-rename replay have actually completed.  A crash in
        # that window leaves an intentionally incomplete, fail-closed bundle.
        producer_payload = canonical_json_bytes(producer) + b"\n"
        producer_stage_path = (
            Path(pins["evidence_staging_root"])
            / Path(pins["producer_receipt"]).relative_to(
                Path(pins["evidence_final_root"])
            )
        )
        staged_producer_row = _write_create_only(
            producer_stage_path, producer_payload, 0o400
        )
        future_producer_row = _future_file_row(
            pins["producer_receipt"], producer_payload
        )
        _require(
            (staged_producer_row["sha256"], staged_producer_row["size"])
            == (future_producer_row["sha256"], future_producer_row["size"]),
            "staged producer receipt bytes differ",
        )

        _verify_file_pin(STATIC_FILE_ROWS["generator"], "generator post-run")
        _verify_file_pin(STATIC_FILE_ROWS["acceptance"], "acceptance post-run")
        _controller_payload_after, controller_row_after = _read_stable_file(
            pins["controller_program_path"], nlink1=True
        )
        _require(controller_row_after == controller_row, "controller changed during generation")
        _verify_file_pin(pins["source"], "source post-run")
        for role in MODEL_AUTHORITY_ROLES:
            role_pin = {
                "python_runtime_tree": pins["python_runtime_manifest"],
                "vace_checkpoint_tree": pins["vace_checkpoint_manifest"],
                "vace_source_tree": pins["vace_source_manifest"],
            }[role]
            _verify_manifest_pin(generator, role_pin, role)

        staged_tree = _scan_regular_tree(bundle_stage)
        _publish_bundle(bundle_stage, bundle_final, stage_identity)
        published = True
        _require(
            _scan_regular_tree(bundle_final) == staged_tree,
            "published bundle inventory differs from staged inventory",
        )
        for name, row in assets.items():
            _verify_file_pin(row, "published asset %s" % name, nlink1=True)
        _verify_file_pin(future_producer_row, "published producer receipt", nlink1=True)
        _verify_file_pin(
            controller_evidence_final_row,
            "published controller execution evidence",
            nlink1=True,
        )
        _require(not os.path.lexists(pins["asset_staging_root"]), "asset staging reappeared")
        _require(not os.path.lexists(pins["bundle_staging_root"]), "bundle staging reappeared")
        _require(
            not os.path.lexists(pins["publication_receipt"]),
            "publication success receipt appeared before postchecks",
        )
        published_publication_row = _write_create_only(
            pins["publication_receipt"], publication_payload, 0o400
        )
        _require(
            published_publication_row == future_publication_row,
            "published publication receipt row differs",
        )
        _fsync_directory(Path(pins["publication_receipt"]).parent)
        _verify_file_pin(
            future_publication_row,
            "published publication receipt",
            nlink1=True,
        )
        _replay_create_only_receipts(
            acceptance,
            attempt,
            publication,
            producer,
        )
        publication_relative = Path(pins["publication_receipt"]).relative_to(
            bundle_final
        ).as_posix()
        _require(
            set(_scan_regular_tree(bundle_final))
            == set(staged_tree) | {publication_relative},
            "post-receipt published bundle inventory differs",
        )
        return {
            "status": "COMPLETE_CANDIDATE_PENDING_INDEPENDENT_ACCEPTANCE",
            "attempt_receipt": attempt_row,
            "publication_receipt": future_publication_row,
            "producer_receipt": future_producer_row,
            "controller_execution_evidence": controller_evidence_final_row,
            "final_root": pins["asset_final_root"],
            "semantic_acceptance_performed": False,
            "scientific_claim_authorized": False,
            "automatic_retry_performed": False,
        }
    except ProcessGroupZeroUnproven:
        # A live or unknown process group may still own, mutate, or recreate
        # staging paths.  Never race it with deletion; retain the captured tree
        # as a quarantined failed attempt for operator inspection.
        raise
    except BaseException:
        if (
            not published
            and (
                not generator_invocation_started
                or process_group_zero_token is _PROCESS_GROUP_ZERO_PROVEN
            )
        ):
            _cleanup_private_stage(bundle_stage, stage_identity)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    # Deliberately before parsing arguments or invoking any filesystem helper.
    # This source has no CLI authority override.
    try:
        admit_pre_io()
    except ControllerHold as error:
        print(str(error), file=sys.stderr)
        return 96
    if argv not in (None, (), []):
        print("HOLD_BEFORE_IO: command-line overrides are forbidden", file=sys.stderr)
        return 96
    try:
        result = execute_create_once()
    except (ControllerHold, AttemptFailed, OSError, subprocess.SubprocessError) as error:
        print("ATTEMPT_FAILED_NO_RETRY: %s" % error, file=sys.stderr)
        return 97
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
