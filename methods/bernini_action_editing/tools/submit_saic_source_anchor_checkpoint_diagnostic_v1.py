#!/usr/bin/env python3
"""Exactly-once submit one release-pinned source-anchor diagnostic canary.

The only scientific trust anchor accepted from the command line is a manifest
whose SHA-256 is also frozen in this source after AUH release materialization.
All archive, training, DINO, evaluator, executable, and wrapper pins are derived
from that manifest.  A reservation is created before the sole ``sbatch`` call;
an ambiguous boundary is never retried automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tarfile
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "saic-source-anchor-diagnostic-submission-v2"
RELEASE_SCHEMA_VERSION = "saic-source-anchor-diagnostic-release-v2"
# Filled only after the final six leaves, source member/origin manifests, formal
# training bundle, and visual release are materialized on AUH.  The non-hash
# placeholder is intentionally fail-closed and is a reported dynamic blocker.
EXPECTED_RELEASE_MANIFEST_SHA256 = (
    "UNRESOLVED_AFTER_AUH_RELEASE_MATERIALIZATION"
)
RELEASE_PIN_PLACEHOLDER = "UNRESOLVED_AFTER_AUH_RELEASE_MATERIALIZATION"
ARCHIVED_SUBMITTER = (
    "methods/bernini_action_editing/tools/"
    "submit_saic_source_anchor_checkpoint_diagnostic_v1.py"
)
EXPECTED_RENDEZVOUS_GUARD_SHA256 = (
    "1a38b2ac18f46e818b1596db884b883dff8e0612fcd5b0cf1ed78aca377ac965"
)
SHA1 = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
SAFE_QOS = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
SBATCH = Path("/usr/bin/sbatch")

RELEASE_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "release_root",
        "source_release",
        "code",
        "executables",
        "base_model",
        "scientific_input",
        "stage_a_release",
        "visual_release",
        "run_policy",
        "authority",
        "receipt_digest",
    }
)
RELEASE_AUTHORITY = {
    "training": False,
    "optimizer_step": False,
    "checkpoint": False,
    "candidate_selection": False,
    "identity": False,
    "semantic_action": False,
    "publication": False,
    "production": False,
    "scientific_success": False,
}
AUTHORITY = {
    "diagnostic_canary_submission": True,
    **RELEASE_AUTHORITY,
}

EXPORT_NAMES = (
    "SAIC_ANCHOR_DIAG_RELEASE_MANIFEST",
    "SAIC_ANCHOR_DIAG_RELEASE_MANIFEST_SHA256",
    "SAIC_ANCHOR_DIAG_RELEASE_MANIFEST_DIGEST",
    "SAIC_ANCHOR_DIAG_SOURCE_ARCHIVE",
    "SAIC_ANCHOR_DIAG_SOURCE_ARCHIVE_SHA256",
    "SAIC_ANCHOR_DIAG_SOURCE_REVISION",
    "SAIC_ANCHOR_DIAG_MEMBER_MANIFEST",
    "SAIC_ANCHOR_DIAG_MEMBER_MANIFEST_SHA256",
    "SAIC_ANCHOR_DIAG_MEMBER_MANIFEST_DIGEST",
    "SAIC_ANCHOR_DIAG_MEMBER_COUNT",
    "SAIC_ANCHOR_DIAG_ORIGIN_MANIFEST",
    "SAIC_ANCHOR_DIAG_ORIGIN_MANIFEST_SHA256",
    "SAIC_ANCHOR_DIAG_ORIGIN_MANIFEST_DIGEST",
    "SAIC_ANCHOR_DIAG_ORIGIN_COUNT",
    "SAIC_ANCHOR_DIAG_LAUNCHER_SHA256",
    "SAIC_ANCHOR_DIAG_PYTHON",
    "SAIC_ANCHOR_DIAG_PYTHON_SHA256",
    "SAIC_ANCHOR_DIAG_RENDEZVOUS_GUARD",
    "SAIC_ANCHOR_DIAG_RENDEZVOUS_GUARD_SHA256",
    "SAIC_ANCHOR_DIAG_BERNINI_ROOT",
    "SAIC_ANCHOR_DIAG_VEOMNI_ROOT",
    "SAIC_ANCHOR_DIAG_CHECKPOINT",
    "SAIC_ANCHOR_DIAG_CHECKPOINT_MANIFEST",
    "SAIC_ANCHOR_DIAG_CHECKPOINT_MANIFEST_SHA256",
    "SAIC_ANCHOR_DIAG_SOURCE_MANIFEST",
    "SAIC_ANCHOR_DIAG_SOURCE_MANIFEST_SHA256",
    "SAIC_ANCHOR_DIAG_STAGE_A_ADAPTER",
    "SAIC_ANCHOR_DIAG_STAGE_A_ADAPTER_SHA256",
    "SAIC_ANCHOR_DIAG_STAGE_A_RECEIPT",
    "SAIC_ANCHOR_DIAG_STAGE_A_RECEIPT_SHA256",
    "SAIC_ANCHOR_DIAG_STAGE_A_POSTFLIGHT",
    "SAIC_ANCHOR_DIAG_STAGE_A_POSTFLIGHT_SHA256",
    "SAIC_ANCHOR_DIAG_STAGE_A_HISTORY",
    "SAIC_ANCHOR_DIAG_STAGE_A_HISTORY_SHA256",
    "SAIC_ANCHOR_DIAG_STAGE_A_CHECKPOINT_RELEASE",
    "SAIC_ANCHOR_DIAG_STAGE_A_CHECKPOINT_RELEASE_SHA256",
    "SAIC_ANCHOR_DIAG_HELDOUT_ROW_INDEX",
    "SAIC_ANCHOR_DIAG_ACTION_CAPTION",
    "SAIC_ANCHOR_DIAG_ACTION_CAPTION_SHA256",
    "SAIC_ANCHOR_DIAG_VISUAL_CHECKPOINT",
    "SAIC_ANCHOR_DIAG_VISUAL_MANIFEST",
    "SAIC_ANCHOR_DIAG_VISUAL_MANIFEST_SHA256",
    "SAIC_ANCHOR_DIAG_VISUAL_RELEASE",
    "SAIC_ANCHOR_DIAG_VISUAL_RELEASE_SHA256",
    "SAIC_ANCHOR_DIAG_VISUAL_EVALUATOR_SPEC",
    "SAIC_ANCHOR_DIAG_VISUAL_EVALUATOR_SPEC_SHA256",
    "SAIC_ANCHOR_DIAG_OUTPUT",
    "SAIC_ANCHOR_DIAG_OUTPUT_PARENT_DEVICE",
    "SAIC_ANCHOR_DIAG_OUTPUT_PARENT_INODE",
    "SAIC_ANCHOR_DIAG_SUBMISSION_RECEIPT",
    "SAIC_ANCHOR_DIAG_SUBMISSION_RECEIPT_DEVICE",
    "SAIC_ANCHOR_DIAG_SUBMISSION_RECEIPT_INODE",
    "SAIC_ANCHOR_DIAG_SLURM_LOG_DIR",
    "SAIC_ANCHOR_DIAG_SLURM_LOG_DIR_DEVICE",
    "SAIC_ANCHOR_DIAG_SLURM_LOG_DIR_INODE",
    "SAIC_ANCHOR_DIAG_POSTFLIGHT",
    "SAIC_ANCHOR_DIAG_POSTFLIGHT_SHA256",
    "SAIC_ANCHOR_DIAG_SEED",
)


def die(message: str) -> None:
    raise SystemExit(f"submit-saic-source-anchor-diagnostic-v2: {message}")


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        die(f"value is not canonical finite ASCII JSON: {error}")
    raise AssertionError("unreachable")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def sha_file(path: Path) -> str:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        return sha_descriptor(descriptor)
    finally:
        os.close(descriptor)


def require_sha(value: Any, *, bits: int = 256, label: str) -> str:
    pattern = SHA1 if bits == 160 else SHA256
    if type(value) is not str or pattern.fullmatch(value) is None:
        die(f"{label} hash differs")
    return value


def strict_json_bytes(raw: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        die(f"{label} contains {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                die(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        die(f"cannot decode {label}: {error}")
    if not isinstance(value, Mapping) or raw != canonical(value) + b"\n":
        die(f"{label} is not canonical one-object JSON")
    return value


def validate_seal(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(value)
    digest = require_sha(unsigned.pop("receipt_digest", None), label=f"{label} digest")
    if sha_bytes(canonical(unsigned)) != digest:
        die(f"{label} digest differs")
    return digest


def exact_directory_row(value: Any, *, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "device",
        "inode",
        "mode",
        "uid",
    }:
        die(f"{label} directory snapshot schema differs")
    path = Path(str(value.get("path", "")))
    if not path.is_absolute() or path == Path("/"):
        die(f"{label} path differs")
    try:
        info = path.lstat()
    except OSError as error:
        die(f"cannot stat {label}: {error}")
    expected = {
        "path": str(path),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "uid": int(info.st_uid),
    }
    if (
        value != expected
        or path.resolve(strict=True) != path
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o022
    ):
        die(f"{label} directory identity differs")
    return path


def exact_file_row(
    value: Any,
    *,
    label: str,
    expected_mode: int | None = 0o444,
    executable: bool = False,
    root_owned: bool = False,
) -> Path:
    fields = {
        "path",
        "sha256",
        "byte_size",
        "device",
        "inode",
        "mode",
        "uid",
        "nlink",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        die(f"{label} file snapshot schema differs")
    path = Path(str(value.get("path", "")))
    digest = require_sha(value.get("sha256"), label=label)
    if not path.is_absolute() or path == Path("/"):
        die(f"{label} path differs")
    try:
        info = path.lstat()
    except OSError as error:
        die(f"cannot stat {label}: {error}")
    expected = {
        "path": str(path),
        "sha256": digest,
        "byte_size": int(info.st_size),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "uid": int(info.st_uid),
        "nlink": int(info.st_nlink),
    }
    if (
        value != expected
        or path.resolve(strict=True) != path
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or (expected_mode is not None and stat.S_IMODE(info.st_mode) != expected_mode)
        or (expected_mode is None and stat.S_IMODE(info.st_mode) & 0o022 != 0)
        or (root_owned and info.st_uid != 0)
        or (executable and not os.access(path, os.X_OK))
        or sha_file(path) != digest
    ):
        die(f"{label} file identity/bytes differ")
    return path


def snapshot(path: Path, sha256: str) -> Mapping[str, Any]:
    info = path.lstat()
    return {
        "path": str(path),
        "sha256": sha256,
        "byte_size": int(info.st_size),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "uid": int(info.st_uid),
        "nlink": int(info.st_nlink),
    }


def _mapping(value: Any, *, label: str, fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        die(f"{label} schema differs")
    return value


def validate_release_manifest(
    path: Path, *, expected_sha256: str
) -> tuple[Mapping[str, Any], Mapping[str, Path], str, int]:
    if expected_sha256 != EXPECTED_RELEASE_MANIFEST_SHA256:
        die("release manifest does not match the source-frozen external anchor")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        raw_chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            raw_chunks.append(chunk)
        raw = b"".join(raw_chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    leaf = path.lstat()
    if (
        not path.is_absolute()
        or path.resolve(strict=True) != path
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o444
        or (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or (leaf.st_dev, leaf.st_ino) != (info.st_dev, info.st_ino)
        or sha_bytes(raw) != expected_sha256
    ):
        die("release manifest retained identity/bytes differ")
    value = strict_json_bytes(raw, label="release manifest")
    if (
        set(value) != RELEASE_FIELDS
        or value.get("schema_version") != RELEASE_SCHEMA_VERSION
        or value.get("status") != "sealed_before_first_diagnostic_canary"
        or value.get("authority") != RELEASE_AUTHORITY
    ):
        die("release manifest top-level contract differs")
    release_digest = validate_seal(value, label="release manifest")
    release_root = exact_directory_row(value.get("release_root"), label="release root")
    if (
        path.parent != release_root
        or value.get("release_root", {}).get("mode") != "0555"
    ):
        die("release manifest is not rooted in its exact release directory")

    source = _mapping(
        value.get("source_release"),
        label="source release",
        fields={
            "archive",
            "revision",
            "member_manifest",
            "member_manifest_digest",
            "member_count",
            "origin_manifest",
            "origin_manifest_digest",
            "origin_count",
        },
    )
    revision = require_sha(source.get("revision"), bits=160, label="source revision")
    archive = exact_file_row(source.get("archive"), label="source archive")
    member = exact_file_row(source.get("member_manifest"), label="member manifest")
    origin = exact_file_row(source.get("origin_manifest"), label="origin manifest")
    member_digest = require_sha(
        source.get("member_manifest_digest"), label="member manifest digest"
    )
    origin_digest = require_sha(
        source.get("origin_manifest_digest"), label="origin manifest digest"
    )
    if (
        type(source.get("member_count")) is not int
        or source["member_count"] <= 0
        or type(source.get("origin_count")) is not int
        or source["origin_count"] <= 0
    ):
        die("source closure counts differ")
    with archive.open("rb") as handle:
        completed = subprocess.run(
            ["/usr/bin/git", "get-tar-commit-id"],
            stdin=handle,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
    if completed.returncode != 0 or completed.stdout.decode("ascii").strip() != revision:
        die("source archive revision differs")

    code = _mapping(
        value.get("code"),
        label="code release",
        fields={"launcher", "postflight", "rendezvous_guard"},
    )
    launcher = exact_file_row(code.get("launcher"), label="launcher")
    postflight = exact_file_row(code.get("postflight"), label="postflight")
    guard = exact_file_row(code.get("rendezvous_guard"), label="rendezvous guard")
    if code["rendezvous_guard"]["sha256"] != EXPECTED_RENDEZVOUS_GUARD_SHA256:
        die("release does not contain the audited rendezvous guard v2")

    executables = _mapping(
        value.get("executables"),
        label="executable release",
        fields={"python", "sbatch", "sacct"},
    )
    python_bin = exact_file_row(
        executables.get("python"),
        label="Python",
        expected_mode=None,
        executable=True,
    )
    sbatch = exact_file_row(
        executables.get("sbatch"),
        label="sbatch",
        expected_mode=None,
        executable=True,
        root_owned=True,
    )
    sacct = exact_file_row(
        executables.get("sacct"),
        label="sacct",
        expected_mode=None,
        executable=True,
        root_owned=True,
    )
    if sbatch != SBATCH or sacct != Path("/usr/bin/sacct"):
        die("Slurm executable path differs")

    base = _mapping(
        value.get("base_model"),
        label="base model release",
        fields={"bernini_root", "veomni_root", "checkpoint", "checkpoint_manifest"},
    )
    bernini = exact_directory_row(base.get("bernini_root"), label="Bernini root")
    veomni = exact_directory_row(base.get("veomni_root"), label="VeOmni root")
    checkpoint = exact_directory_row(base.get("checkpoint"), label="checkpoint")
    checkpoint_manifest = exact_file_row(
        base.get("checkpoint_manifest"), label="checkpoint manifest"
    )

    scientific = _mapping(
        value.get("scientific_input"),
        label="scientific input",
        fields={"source_manifest", "action_caption", "heldout_row_index"},
    )
    source_manifest = exact_file_row(
        scientific.get("source_manifest"), label="source-anchor manifest"
    )
    action_caption = exact_file_row(
        scientific.get("action_caption"), label="action caption"
    )
    heldout = scientific.get("heldout_row_index")
    if type(heldout) is not int or not 0 <= heldout < 16:
        die("held-out row index differs")

    stage = _mapping(
        value.get("stage_a_release"),
        label="Stage-A release",
        fields={
            "adapter",
            "training_receipt",
            "formal_postflight",
            "training_history",
            "checkpoint_release",
            "formal_postflight_digest",
            "formal_schema_version",
        },
    )
    adapter = exact_file_row(stage.get("adapter"), label="Stage-A adapter")
    training_receipt = exact_file_row(
        stage.get("training_receipt"), label="Stage-A training receipt"
    )
    formal_postflight = exact_file_row(
        stage.get("formal_postflight"), label="Stage-A formal postflight"
    )
    history = exact_file_row(
        stage.get("training_history"), label="Stage-A training history"
    )
    checkpoint_release = exact_file_row(
        stage.get("checkpoint_release"), label="Stage-A checkpoint release"
    )
    require_sha(stage.get("formal_postflight_digest"), label="formal postflight digest")
    if (
        stage.get("formal_schema_version")
        != "saic-source-anchor-formal32-terminal-admission-v1"
    ):
        die("formal Stage-A postflight schema pin differs")

    visual = _mapping(
        value.get("visual_release"),
        label="visual release",
        fields={
            "checkpoint_root",
            "content_manifest",
            "release_manifest",
            "evaluator_spec",
            "visual_schema_version",
        },
    )
    visual_checkpoint = exact_directory_row(
        visual.get("checkpoint_root"), label="visual checkpoint"
    )
    visual_manifest = exact_file_row(
        visual.get("content_manifest"), label="visual content manifest"
    )
    visual_release = exact_file_row(
        visual.get("release_manifest"), label="visual release manifest"
    )
    visual_spec = exact_file_row(
        visual.get("evaluator_spec"), label="visual evaluator spec"
    )
    if visual.get("visual_schema_version") != "saic-source-anchor-diagnostic-visual-release-v1":
        die("visual release schema pin differs")

    policy = _mapping(
        value.get("run_policy"),
        label="run policy",
        fields={
            "seed",
            "first_run",
            "cell_count",
            "world_size",
            "ulysses_size",
            "exact81",
            "exact40",
            "automatic_retry_allowed",
        },
    )
    if (
        type(policy.get("seed")) is not int
        or not 0 <= policy["seed"] < 2**63
        or policy.get("first_run") != "single_world4_canary"
        or policy.get("cell_count") != 7
        or policy.get("world_size") != 4
        or policy.get("ulysses_size") != 4
        or policy.get("exact81") is not True
        or policy.get("exact40") is not True
        or policy.get("automatic_retry_allowed") is not False
    ):
        die("release run policy differs")

    release_files = {path, archive, member, origin, launcher, postflight, guard}
    observed_release_files = {
        candidate
        for candidate in release_root.rglob("*")
        if candidate.is_file() or candidate.is_symlink()
    }
    if observed_release_files != release_files:
        die("release directory recursive file closure differs")
    expected_release_directories = {release_root}
    for release_file in release_files:
        try:
            release_file.relative_to(release_root)
        except ValueError:
            die("release file escaped the frozen release root")
        parent = release_file.parent
        while parent != release_root:
            expected_release_directories.add(parent)
            parent = parent.parent
    observed_release_directories = {release_root} | {
        candidate
        for candidate in release_root.rglob("*")
        if candidate.is_dir() and not candidate.is_symlink()
    }
    if observed_release_directories != expected_release_directories:
        die("release directory recursive tree closure differs")
    resolved = {
        "archive": archive,
        "member_manifest": member,
        "origin_manifest": origin,
        "launcher": launcher,
        "postflight": postflight,
        "guard": guard,
        "python": python_bin,
        "sbatch": sbatch,
        "sacct": sacct,
        "bernini_root": bernini,
        "veomni_root": veomni,
        "checkpoint": checkpoint,
        "checkpoint_manifest": checkpoint_manifest,
        "source_manifest": source_manifest,
        "action_caption": action_caption,
        "adapter": adapter,
        "training_receipt": training_receipt,
        "formal_postflight": formal_postflight,
        "history": history,
        "checkpoint_release": checkpoint_release,
        "visual_checkpoint": visual_checkpoint,
        "visual_manifest": visual_manifest,
        "visual_release": visual_release,
        "visual_spec": visual_spec,
    }
    return value, resolved, release_digest, info.st_ino


def validate_materialized_submitter_self(
    archive: Path, *, expected_release_sha256: str
) -> Mapping[str, Any]:
    """Prove this executable is one exact pin substitution of the archived template."""

    self_path = Path(__file__).resolve(strict=True)
    self_info = self_path.lstat()
    self_raw = self_path.read_bytes()
    if (
        not stat.S_ISREG(self_info.st_mode)
        or stat.S_ISLNK(self_info.st_mode)
        or self_info.st_nlink != 1
        or stat.S_IMODE(self_info.st_mode) != 0o444
    ):
        die("materialized submitter identity/mode differs")
    try:
        with tarfile.open(archive, mode="r:*") as handle:
            member = handle.getmember(ARCHIVED_SUBMITTER)
            stream = handle.extractfile(member)
            template_raw = stream.read() if stream is not None else b""
    except (KeyError, OSError, tarfile.TarError) as error:
        die(f"cannot read archived submitter template: {error}")
    old = RELEASE_PIN_PLACEHOLDER.encode("ascii")
    new = expected_release_sha256.encode("ascii")
    if template_raw.count(old) != 2:
        die("archived submitter release-pin placeholder closure differs")
    # One occurrence is the constant value; the second names the immutable
    # transformation token.  Replace both, then preserve the latter constant by
    # restoring its declaration so future self-validation still knows the token.
    materialized = template_raw.replace(old, new)
    materialized = materialized.replace(
        b'RELEASE_PIN_PLACEHOLDER = "' + new + b'"',
        b'RELEASE_PIN_PLACEHOLDER = "' + old + b'"',
        1,
    )
    if materialized != self_raw:
        die("materialized submitter is not the exact archived-template transform")
    return {
        "path": str(self_path),
        "sha256": sha_bytes(self_raw),
        "byte_size": len(self_raw),
        "device": int(self_info.st_dev),
        "inode": int(self_info.st_ino),
        "mode": "0444",
        "uid": int(self_info.st_uid),
        "nlink": int(self_info.st_nlink),
        "archived_template_path": ARCHIVED_SUBMITTER,
        "archived_template_sha256": sha_bytes(template_raw),
        "exact_release_pin_substitution": True,
    }


def exact_private_directory(value: str, *, label: str, require_empty: bool) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path("/"):
        die(f"{label} path differs")
    info = path.lstat()
    if (
        path.resolve(strict=True) != path
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
        or (require_empty and any(path.iterdir()))
    ):
        die(f"{label} identity/mode/closure differs")
    return path


def fresh_output(value: str) -> tuple[Path, Path]:
    output = Path(value)
    if (
        not output.is_absolute()
        or output == Path("/")
        or SAFE_NAME.fullmatch(output.name) is None
        or output.exists()
        or output.is_symlink()
    ):
        die("diagnostic output must be one fresh canonical child")
    parent = exact_private_directory(
        str(output.parent), label="output parent", require_empty=False
    )
    if output != parent / output.name:
        die("diagnostic output path is not canonical")
    if (parent / f"{output.name}.rendezvous").exists() or (
        parent / f"{output.name}.rendezvous"
    ).is_symlink():
        die("rendezvous evidence sibling is not fresh")
    return output, parent


def fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        wrote = os.write(descriptor, payload[offset:])
        if wrote <= 0:
            die("submission receipt write stalled")
        offset += wrote


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", required=True)
    parser.add_argument("--release-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--submission-receipt", required=True)
    parser.add_argument("--slurm-log-dir", required=True)
    parser.add_argument("--qos", required=True)
    parser.add_argument("--ack-submit-one-diagnostic-canary", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ack_submit_one_diagnostic_canary is not True:
        die("explicit one-canary submission acknowledgement is required")
    if SAFE_QOS.fullmatch(args.qos) is None:
        die("QOS name differs")
    if SHA256.fullmatch(EXPECTED_RELEASE_MANIFEST_SHA256) is None:
        die(
            "UNRESOLVED_RELEASE_PIN: materialize the final AUH release, then freeze "
            "EXPECTED_RELEASE_MANIFEST_SHA256 before submission"
        )
    expected_release_sha = require_sha(
        args.release_manifest_sha256, label="release manifest"
    )
    release_path = Path(args.release_manifest)
    release, resolved, release_digest, _ = validate_release_manifest(
        release_path, expected_sha256=expected_release_sha
    )
    submitter_self = validate_materialized_submitter_self(
        resolved["archive"], expected_release_sha256=expected_release_sha
    )
    output, output_parent = fresh_output(args.output)
    log_dir = exact_private_directory(
        args.slurm_log_dir, label="Slurm log directory", require_empty=True
    )
    output_info = output_parent.lstat()
    log_info = log_dir.lstat()

    receipt = Path(args.submission_receipt)
    expected_receipt = output_parent / f"{output.name}.submission.json"
    if (
        receipt != expected_receipt
        or not receipt.is_absolute()
        or receipt.exists()
        or receipt.is_symlink()
    ):
        die("submission receipt is not the fresh derived path")

    wrapper = resolved["launcher"]
    wrapper_descriptor = os.open(
        wrapper, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    wrapper_info = os.fstat(wrapper_descriptor)
    launcher_sha = str(release["code"]["launcher"]["sha256"])
    if (
        not stat.S_ISREG(wrapper_info.st_mode)
        or wrapper_info.st_nlink != 1
        or stat.S_IMODE(wrapper_info.st_mode) != 0o444
        or sha_descriptor(wrapper_descriptor) != launcher_sha
    ):
        os.close(wrapper_descriptor)
        die("retained launcher differs")

    descriptor = os.open(
        receipt,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    reserved = os.fstat(descriptor)
    if (
        not stat.S_ISREG(reserved.st_mode)
        or reserved.st_nlink != 1
        or stat.S_IMODE(reserved.st_mode) != 0o600
    ):
        os.close(wrapper_descriptor)
        os.close(descriptor)
        die("submission reservation differs")
    provisional = {
        "schema_version": SCHEMA_VERSION,
        "status": "reserved_before_sbatch_ambiguous_never_retry",
        "submission_success": False,
        "job_success": None,
        "release_manifest_sha256": expected_release_sha,
        "release_manifest_digest": release_digest,
        "launcher_sha256": launcher_sha,
        "output": str(output),
        "authority": AUTHORITY,
    }
    write_all(descriptor, canonical(provisional) + b"\n")
    os.fsync(descriptor)
    fsync_directory(output_parent)

    source = release["source_release"]
    code = release["code"]
    executables = release["executables"]
    base = release["base_model"]
    scientific = release["scientific_input"]
    stage = release["stage_a_release"]
    visual = release["visual_release"]
    policy = release["run_policy"]
    exports = {
        "SAIC_ANCHOR_DIAG_RELEASE_MANIFEST": str(release_path),
        "SAIC_ANCHOR_DIAG_RELEASE_MANIFEST_SHA256": expected_release_sha,
        "SAIC_ANCHOR_DIAG_RELEASE_MANIFEST_DIGEST": release_digest,
        "SAIC_ANCHOR_DIAG_SOURCE_ARCHIVE": str(resolved["archive"]),
        "SAIC_ANCHOR_DIAG_SOURCE_ARCHIVE_SHA256": source["archive"]["sha256"],
        "SAIC_ANCHOR_DIAG_SOURCE_REVISION": source["revision"],
        "SAIC_ANCHOR_DIAG_MEMBER_MANIFEST": str(resolved["member_manifest"]),
        "SAIC_ANCHOR_DIAG_MEMBER_MANIFEST_SHA256": source["member_manifest"]["sha256"],
        "SAIC_ANCHOR_DIAG_MEMBER_MANIFEST_DIGEST": source["member_manifest_digest"],
        "SAIC_ANCHOR_DIAG_MEMBER_COUNT": str(source["member_count"]),
        "SAIC_ANCHOR_DIAG_ORIGIN_MANIFEST": str(resolved["origin_manifest"]),
        "SAIC_ANCHOR_DIAG_ORIGIN_MANIFEST_SHA256": source["origin_manifest"]["sha256"],
        "SAIC_ANCHOR_DIAG_ORIGIN_MANIFEST_DIGEST": source["origin_manifest_digest"],
        "SAIC_ANCHOR_DIAG_ORIGIN_COUNT": str(source["origin_count"]),
        "SAIC_ANCHOR_DIAG_LAUNCHER_SHA256": launcher_sha,
        "SAIC_ANCHOR_DIAG_PYTHON": str(resolved["python"]),
        "SAIC_ANCHOR_DIAG_PYTHON_SHA256": executables["python"]["sha256"],
        "SAIC_ANCHOR_DIAG_RENDEZVOUS_GUARD": str(resolved["guard"]),
        "SAIC_ANCHOR_DIAG_RENDEZVOUS_GUARD_SHA256": code["rendezvous_guard"]["sha256"],
        "SAIC_ANCHOR_DIAG_BERNINI_ROOT": str(resolved["bernini_root"]),
        "SAIC_ANCHOR_DIAG_VEOMNI_ROOT": str(resolved["veomni_root"]),
        "SAIC_ANCHOR_DIAG_CHECKPOINT": str(resolved["checkpoint"]),
        "SAIC_ANCHOR_DIAG_CHECKPOINT_MANIFEST": str(resolved["checkpoint_manifest"]),
        "SAIC_ANCHOR_DIAG_CHECKPOINT_MANIFEST_SHA256": base["checkpoint_manifest"]["sha256"],
        "SAIC_ANCHOR_DIAG_SOURCE_MANIFEST": str(resolved["source_manifest"]),
        "SAIC_ANCHOR_DIAG_SOURCE_MANIFEST_SHA256": scientific["source_manifest"]["sha256"],
        "SAIC_ANCHOR_DIAG_STAGE_A_ADAPTER": str(resolved["adapter"]),
        "SAIC_ANCHOR_DIAG_STAGE_A_ADAPTER_SHA256": stage["adapter"]["sha256"],
        "SAIC_ANCHOR_DIAG_STAGE_A_RECEIPT": str(resolved["training_receipt"]),
        "SAIC_ANCHOR_DIAG_STAGE_A_RECEIPT_SHA256": stage["training_receipt"]["sha256"],
        "SAIC_ANCHOR_DIAG_STAGE_A_POSTFLIGHT": str(resolved["formal_postflight"]),
        "SAIC_ANCHOR_DIAG_STAGE_A_POSTFLIGHT_SHA256": stage["formal_postflight"]["sha256"],
        "SAIC_ANCHOR_DIAG_STAGE_A_HISTORY": str(resolved["history"]),
        "SAIC_ANCHOR_DIAG_STAGE_A_HISTORY_SHA256": stage["training_history"]["sha256"],
        "SAIC_ANCHOR_DIAG_STAGE_A_CHECKPOINT_RELEASE": str(
            resolved["checkpoint_release"]
        ),
        "SAIC_ANCHOR_DIAG_STAGE_A_CHECKPOINT_RELEASE_SHA256": stage[
            "checkpoint_release"
        ]["sha256"],
        "SAIC_ANCHOR_DIAG_HELDOUT_ROW_INDEX": str(scientific["heldout_row_index"]),
        "SAIC_ANCHOR_DIAG_ACTION_CAPTION": str(resolved["action_caption"]),
        "SAIC_ANCHOR_DIAG_ACTION_CAPTION_SHA256": scientific["action_caption"]["sha256"],
        "SAIC_ANCHOR_DIAG_VISUAL_CHECKPOINT": str(resolved["visual_checkpoint"]),
        "SAIC_ANCHOR_DIAG_VISUAL_MANIFEST": str(resolved["visual_manifest"]),
        "SAIC_ANCHOR_DIAG_VISUAL_MANIFEST_SHA256": visual["content_manifest"]["sha256"],
        "SAIC_ANCHOR_DIAG_VISUAL_RELEASE": str(resolved["visual_release"]),
        "SAIC_ANCHOR_DIAG_VISUAL_RELEASE_SHA256": visual["release_manifest"]["sha256"],
        "SAIC_ANCHOR_DIAG_VISUAL_EVALUATOR_SPEC": str(resolved["visual_spec"]),
        "SAIC_ANCHOR_DIAG_VISUAL_EVALUATOR_SPEC_SHA256": visual["evaluator_spec"]["sha256"],
        "SAIC_ANCHOR_DIAG_OUTPUT": str(output),
        "SAIC_ANCHOR_DIAG_OUTPUT_PARENT_DEVICE": str(output_info.st_dev),
        "SAIC_ANCHOR_DIAG_OUTPUT_PARENT_INODE": str(output_info.st_ino),
        "SAIC_ANCHOR_DIAG_SUBMISSION_RECEIPT": str(receipt),
        "SAIC_ANCHOR_DIAG_SUBMISSION_RECEIPT_DEVICE": str(reserved.st_dev),
        "SAIC_ANCHOR_DIAG_SUBMISSION_RECEIPT_INODE": str(reserved.st_ino),
        "SAIC_ANCHOR_DIAG_SLURM_LOG_DIR": str(log_dir),
        "SAIC_ANCHOR_DIAG_SLURM_LOG_DIR_DEVICE": str(log_info.st_dev),
        "SAIC_ANCHOR_DIAG_SLURM_LOG_DIR_INODE": str(log_info.st_ino),
        "SAIC_ANCHOR_DIAG_POSTFLIGHT": str(resolved["postflight"]),
        "SAIC_ANCHOR_DIAG_POSTFLIGHT_SHA256": code["postflight"]["sha256"],
        "SAIC_ANCHOR_DIAG_SEED": str(policy["seed"]),
    }
    if (
        tuple(exports) != EXPORT_NAMES
        or len(exports) != len(set(exports))
        or any(
            any(
                character in name or character in value
                for character in (",", "\n", "\r", "\x00", " ", "\t", "|", "'", '"', "\\")
            )
            for name, value in exports.items()
        )
    ):
        os.close(wrapper_descriptor)
        os.close(descriptor)
        die("exact Slurm export closure differs")
    public = receipt.lstat()
    retained = os.fstat(wrapper_descriptor)
    if (
        output.exists()
        or output.is_symlink()
        or (output_parent.lstat().st_dev, output_parent.lstat().st_ino)
        != (output_info.st_dev, output_info.st_ino)
        or (log_dir.lstat().st_dev, log_dir.lstat().st_ino)
        != (log_info.st_dev, log_info.st_ino)
        or (public.st_dev, public.st_ino) != (reserved.st_dev, reserved.st_ino)
        or (retained.st_dev, retained.st_ino)
        != (wrapper_info.st_dev, wrapper_info.st_ino)
        or sha_descriptor(wrapper_descriptor) != launcher_sha
    ):
        os.close(wrapper_descriptor)
        os.close(descriptor)
        die("pre-sbatch retained namespace differs")

    stdout_pattern = log_dir / "saic-anchor-diag-v2-%j.out"
    stderr_pattern = log_dir / "saic-anchor-diag-v2-%j.err"
    command = [
        str(SBATCH),
        "--parsable",
        f"--qos={args.qos}",
        f"--output={stdout_pattern}",
        f"--error={stderr_pattern}",
        "--export=NONE," + ",".join(f"{key}={value}" for key, value in exports.items()),
        f"/proc/self/fd/{wrapper_descriptor}",
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
            pass_fds=(wrapper_descriptor,),
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
    finally:
        os.close(wrapper_descriptor)
    try:
        stdout = completed.stdout.decode("ascii")
    except UnicodeDecodeError:
        os.close(descriptor)
        die("sbatch stdout is not ASCII; reservation retained")
    match = re.fullmatch(r"([1-9][0-9]*)(?:;([^\n;]+))?\n?", stdout)
    if completed.returncode != 0 or match is None:
        os.close(descriptor)
        die(
            "sbatch failed or returned an ambiguous ID; 0600 reservation retained; "
            f"exit={completed.returncode} stderr_sha256={sha_bytes(completed.stderr)}"
        )

    job_id = match.group(1)
    stdout_path = log_dir / f"saic-anchor-diag-v2-{job_id}.out"
    stderr_path = log_dir / f"saic-anchor-diag-v2-{job_id}.err"
    sentinel = (
        "SAIC_SOURCE_ANCHOR_CHECKPOINT_DIAGNOSTIC_V2_CANARY_OK "
        f"job_id={job_id} release_digest={release_digest} output={output}"
    )
    core = {
        "schema_version": SCHEMA_VERSION,
        "status": "submitted_single_attempt_no_retry",
        "submission_success": True,
        "job_success": None,
        "submitted_job": {
            "job_id": job_id,
            "cluster": match.group(2),
            "stdout_sha256": sha_bytes(completed.stdout),
            "stderr_sha256": sha_bytes(completed.stderr),
        },
        "request": {
            "job_name": "saic-anchor-diag-v2",
            "partition": "faculty",
            "qos": args.qos,
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 32,
            "memory": "256G",
            "walltime": "24:00:00",
            "gpu_resource_requested": "gpu:mi210:4",
            "world_size": 4,
            "ulysses_size": 4,
            "cell_count": 7,
            "exact81": True,
            "exact40": True,
            "first_run": "single_world4_canary",
            "hold": False,
            "dependency": None,
        },
        "single_attempt_boundary": {
            "reservation_created_before_sbatch": True,
            "automatic_retry_allowed": False,
            "ambiguous_reservation_requires_external_reconciliation": True,
            "job_waits_for_immutable_terminal_submission_receipt": True,
            "job_fails_before_model_load_if_receipt_is_not_terminal": True,
            "launcher_submitted_from_retained_fd": True,
            "retained_wrapper_device": int(wrapper_info.st_dev),
            "retained_wrapper_inode": int(wrapper_info.st_ino),
            "environment_replaced": True,
            "export_all": False,
            "exact_export_names": list(EXPORT_NAMES),
            "reservation_device": int(reserved.st_dev),
            "reservation_inode": int(reserved.st_ino),
            "output_parent_device": int(output_info.st_dev),
            "output_parent_inode": int(output_info.st_ino),
            "slurm_log_dir_device": int(log_info.st_dev),
            "slurm_log_dir_inode": int(log_info.st_ino),
        },
        "inputs": {
            "materialized_submitter": submitter_self,
            "release_manifest": snapshot(release_path, expected_release_sha),
            "release_manifest_digest": release_digest,
            "resolved_release": release,
        },
        "exports": exports,
        "outputs": {
            "diagnostic_output": str(output),
            "rendezvous_evidence": str(output_parent / f"{output.name}.rendezvous"),
            "submission_receipt": str(receipt),
            "slurm_stdout": str(stdout_path),
            "slurm_stderr": str(stderr_path),
            "terminal_stdout_sentinel": sentinel,
        },
        "authority": AUTHORITY,
    }
    terminal = {**core, "receipt_digest": sha_bytes(canonical(core))}
    payload = canonical(terminal) + b"\n"
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    write_all(descriptor, payload)
    os.fsync(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.read(descriptor, len(payload) + 1) != payload:
        os.close(descriptor)
        die("terminal submission receipt retained-FD reread differs")
    public = receipt.lstat()
    retained_receipt = os.fstat(descriptor)
    if (
        output.exists()
        or output.is_symlink()
        or public.st_dev != reserved.st_dev
        or public.st_ino != reserved.st_ino
        or public.st_nlink != 1
        or stat.S_IMODE(public.st_mode) != 0o600
        or public.st_size != len(payload)
        or (retained_receipt.st_dev, retained_receipt.st_ino)
        != (reserved.st_dev, reserved.st_ino)
        or (output_parent.lstat().st_dev, output_parent.lstat().st_ino)
        != (output_info.st_dev, output_info.st_ino)
        or (log_dir.lstat().st_dev, log_dir.lstat().st_ino)
        != (log_info.st_dev, log_info.st_ino)
    ):
        os.close(descriptor)
        die("pre-publication submission namespace differs")
    fsync_directory(output_parent)
    os.fchmod(descriptor, 0o444)
    try:
        os.close(descriptor)
    except OSError:
        # 0444 is the irreversible success transition; close cannot revoke it.
        pass
    os._exit(0)


__all__ = [
    "AUTHORITY",
    "EXPECTED_RELEASE_MANIFEST_SHA256",
    "EXPORT_NAMES",
    "RELEASE_FIELDS",
    "RELEASE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_parser",
    "canonical",
    "main",
    "validate_release_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
