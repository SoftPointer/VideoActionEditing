#!/usr/bin/env python3
"""Frozen release checks for the full644 R64 exploratory one-pass run.

This module is deliberately stdlib-only.  It builds/audits the byte-exact
review archive, verifies the two training-output profiles, and turns the
capacity smoke's terminal Slurm accounting row into a fail-closed gate.  It
does not submit work, run a model, or grant a scientific/formal claim.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class ReleaseError(RuntimeError):
    """A frozen release invariant differs."""


REVIEW_SNAPSHOT_PATH = (
    "methods/bernini_action_editing/audits/"
    "full644_exploratory_r64_review_snapshot_20260819_v3.json"
)
REVIEW_SNAPSHOT_SHA256 = (
    "bb228326a05f85573a413217d7f7941f9ccc1b6beb4ccdbefff68d73693e863c"
)
SOURCE_AUTHORITY_PATH = (
    "md/action_editing/20260814_man/evidence/stage_r64_joint_136309_r2/"
    "run_receipt.json"
)
SOURCE_AUTHORITY_SHA256 = (
    "0bcf24ce8aafabb37cf38eafe9da6b13c70043bb0f4c3146f16dc0bafd35618f"
)
SOURCE_AUTHORITY_SIZE = 302_520
ARCHIVE_MANIFEST_PATH = "_release/full644_exploratory_r64_source_manifest_v1.json"
# Deterministic USTAR built from the exact table below.  The builder does not
# consume this value, so rebuilding independently must reproduce these bytes.
SOURCE_ARCHIVE_SHA256 = (
    "12a28ddec99704963af42f1a82b09dff31828e3af8e53e5d0bbd0d43db272828"
)

METHOD_SOURCE_REVISION = "ce4cffc1e8a144448c92252d9fb63087f03bbd8c"
HOLDER_JOB_ID = "141620"
HOLDER_NODE = "auh7-1b-gpu-226"
WORLD_SIZE = 4
ULYSSES_SIZE = 4
LORA_RANK = 64
LORA_ALPHA = 64
TRAINABLE_PARAMETER_COUNT = 47_185_920
TRAIN_RECEIPT_SCHEMA = "bernini-r-1p3b-action-lora-receipt-v2"
CHECKPOINT_MANIFEST_SCHEMA = "bernini-r-action-lora-checkpoint-manifest-v1"
FULL644_PROFILE = "full644-r64-reference-dpo-preservation-one-pass-v1"
DATASET_ROWS = 644
DATASET_SUMMARY_SHA256 = (
    "5dc45b4a6d700b3cd0108e941242ae364396458f20f41249744e74e00acc02dd"
)
DATASET_SUMMARY_DIGEST = (
    "29e2341f09d58289590ae48d17d02f2299bac3201df772584b6269bec0dbbe82"
)
DATASET_INDEX_SHA256 = (
    "d36fb5de3487ba5bf494589948430a60e214851d29776cc4f439e4e2d54ee52b"
)
CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
PEFT_VERSION = "0.19.1"
BERNINI_TRAINING_FILES_INDEX_SHA256 = (
    "faeaa381cb076febd07ac0eb90d17396b61ff400eac2e02a6c7b3c70ff062764"
)

# 64 GiB allocation: require at least eight GiB of observed headroom after the
# real WORLD4/R64 optimizer update.  This smoke primarily covers the known
# all-rank T5/renderer host-load peak.  It is explicitly not full644 evidence.
CAPACITY_ALLOCATED_BYTES = 64 * 1024**3
CAPACITY_MIN_HEADROOM_BYTES = 8 * 1024**3
CAPACITY_MAX_RSS_BYTES = CAPACITY_ALLOCATED_BYTES - CAPACITY_MIN_HEADROOM_BYTES

REVIEW_MEMBERS: Tuple[Tuple[str, str, int], ...] = (
    ("methods/bernini_action_editing/train_lora.py", "ead547b8309e1b5ae5c831444e9f5d1d8e1785fed5fe39cf7b97f13f82a9ce85", 157494),
    ("methods/bernini_action_editing/infer_lora.py", "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553", 177300),
    ("methods/bernini_action_editing/action_preservation_decoded_eval_model_authority_v2.py", "760ed9988147a44965fd47f68a08fd353ce1d900e661b55bb818088ec9ef848e", 115128),
    ("methods/bernini_action_editing/full644_exploratory_matched_eval_v1.py", "d6ef0939a67598e66ccf2652d22520ae3a87a068789f70f921522ba86046138d", 85344),
    ("methods/bernini_action_editing/tests/test_train_lora_contract.py", "e1a22366f243b8764944661b4abfa4aeaefeb97be092feda853c789b6d24baa5", 48100),
    ("methods/bernini_action_editing/tests/test_infer_lora_contract.py", "68356aea8beb835b9e72adcb581fa14914ace4eab60966045adf287a8a216091", 37154),
    ("methods/bernini_action_editing/tests/test_action_preservation_decoded_eval_model_authority_v2.py", "bd8e1bec5d2e1035a27042d2884530b6ba4b17142f9c25bc41567873a92343ae", 20110),
    ("methods/bernini_action_editing/tests/test_full644_exploratory_matched_eval_v1.py", "e8c5c8d63b9b91f333350d6f036f21afa784c7a042868889cb0284a1c020950e", 38241),
    ("md/action_editing/20260817_man/05_full644_exploratory_training_review.md", "369e6a5bd9f57c9793f48f324ced9d26500ce60fb628550d5218b536f09f7c78", 7607),
    ("md/action_editing/20260817_man/README.md", "28c820126c8812e5369859ef55c23d40871be934761200e848495ff1993ed9e7", 6902),
    ("md/action_editing/20260817_man/04_execution_ledger.md", "0ed9626cdd56c2b54b38b3931835469398eaa7c571cd2afd9d36f7c478b7c87e", 30054),
)


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


def _reject_duplicate_pairs(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseError("JSON contains a duplicate key: %s" % key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ReleaseError("JSON contains a non-finite constant: %s" % value)


def parse_json_bytes(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ReleaseError) as error:
        raise ReleaseError("%s is not strict JSON: %s" % (label, error)) from error


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_stable_file(
    path_value: str | Path,
    *,
    label: str,
    expected_sha256: Optional[str] = None,
    expected_size: Optional[int] = None,
) -> bytes:
    path = Path(path_value)
    if not path.is_absolute():
        raise ReleaseError("%s must be absolute" % label)
    try:
        before_lstat = path.lstat()
        if not stat.S_ISREG(before_lstat.st_mode) or stat.S_ISLNK(before_lstat.st_mode):
            raise ReleaseError("%s is not a plain regular file" % label)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
    except ReleaseError:
        raise
    except OSError as error:
        raise ReleaseError("cannot open %s: %s" % (label, error)) from error
    try:
        before = os.fstat(descriptor)
        chunks: List[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_mode,
    )
    if identity(before_lstat) != identity(before) or identity(before) != identity(after):
        raise ReleaseError("%s changed during its stable read" % label)
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise ReleaseError("%s byte count differs" % label)
    if expected_size is not None and len(raw) != expected_size:
        raise ReleaseError("%s size differs" % label)
    digest = sha256_bytes(raw)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ReleaseError("%s SHA-256 differs" % label)
    return raw


def _relative_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if (
        member.is_absolute()
        or not member.parts
        or any(part in ("", ".", "..") for part in member.parts)
    ):
        raise ReleaseError("unsafe archive member path: %r" % name)
    return member


def _review_snapshot_contract(raw: bytes) -> Mapping[str, Any]:
    snapshot = parse_json_bytes(raw, "review snapshot")
    if not isinstance(snapshot, Mapping):
        raise ReleaseError("review snapshot root differs")
    expected = [
        {"path": path, "sha256": digest, "size": size}
        for path, digest, size in REVIEW_MEMBERS
    ]
    if snapshot.get("members") != expected:
        raise ReleaseError("review snapshot member table differs")
    if snapshot.get("schema_version") != "full644-exploratory-r64-review-snapshot-v1":
        raise ReleaseError("review snapshot schema differs")
    return snapshot


def _archive_payloads(repo_root_value: str | Path) -> Mapping[str, bytes]:
    root = Path(repo_root_value)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ReleaseError("repo root must be one plain absolute directory")
    payloads: Dict[str, bytes] = {}
    snapshot_raw = read_stable_file(
        root / REVIEW_SNAPSHOT_PATH,
        label="review snapshot",
        expected_sha256=REVIEW_SNAPSHOT_SHA256,
    )
    _review_snapshot_contract(snapshot_raw)
    payloads[REVIEW_SNAPSHOT_PATH] = snapshot_raw
    for member, digest, size in REVIEW_MEMBERS:
        payloads[member] = read_stable_file(
            root / member,
            label="review member %s" % member,
            expected_sha256=digest,
            expected_size=size,
        )
    payloads[SOURCE_AUTHORITY_PATH] = read_stable_file(
        root / SOURCE_AUTHORITY_PATH,
        label="full644 source authority",
        expected_sha256=SOURCE_AUTHORITY_SHA256,
        expected_size=SOURCE_AUTHORITY_SIZE,
    )
    manifest = {
        "schema_version": "full644-exploratory-r64-source-manifest-v1",
        "claim": "EXPOSED_FULL644_EXPLORATORY_ABLATION_ONLY",
        "formal_or_scientific_authority": False,
        "method_source_revision": METHOD_SOURCE_REVISION,
        "review_snapshot": {
            "path": REVIEW_SNAPSHOT_PATH,
            "sha256": REVIEW_SNAPSHOT_SHA256,
            "size": len(snapshot_raw),
        },
        "source_authority": {
            "path": SOURCE_AUTHORITY_PATH,
            "sha256": SOURCE_AUTHORITY_SHA256,
            "size": SOURCE_AUTHORITY_SIZE,
        },
        "members": [
            {"path": name, "sha256": sha256_bytes(raw), "size": len(raw)}
            for name, raw in sorted(payloads.items())
        ],
    }
    payloads[ARCHIVE_MANIFEST_PATH] = canonical_json_bytes(manifest) + b"\n"
    return payloads


def _ustar_octal(value: int, width: int) -> bytes:
    if type(value) is not int or value < 0:
        raise ReleaseError("USTAR integer differs")
    encoded = ("%0*o" % (width - 1, value)).encode("ascii") + b"\0"
    if len(encoded) != width:
        raise ReleaseError("USTAR integer does not fit its frozen field")
    return encoded


def _canonical_ustar_header(name: str, size: int) -> bytes:
    name_raw = name.encode("ascii")
    if len(name_raw) > 99 or type(size) is not int or size < 0:
        raise ReleaseError("USTAR member name/size differs")
    header = bytearray(512)
    header[0 : len(name_raw)] = name_raw
    header[100:108] = _ustar_octal(0o444, 8)
    header[108:116] = _ustar_octal(0, 8)
    header[116:124] = _ustar_octal(0, 8)
    header[124:136] = _ustar_octal(size, 12)
    header[136:148] = _ustar_octal(0, 12)
    header[148:156] = b"        "
    header[156:157] = b"0"
    header[257:263] = b"ustar\0"
    header[263:265] = b"00"
    checksum = sum(header)
    checksum_raw = ("%06o\0 " % checksum).encode("ascii")
    if len(checksum_raw) != 8:
        raise ReleaseError("USTAR checksum does not fit")
    header[148:156] = checksum_raw
    return bytes(header)


def canonical_ustar_bytes(payloads: Mapping[str, bytes]) -> bytes:
    """Return portable canonical USTAR bytes without ``tarfile`` writing.

    Python 3.8/3.9/3.12 tarfile writers do not promise identical headers.
    These bytes define every header field explicitly and use no PAX/GNU data.
    """

    output = bytearray()
    for name, raw in sorted(payloads.items()):
        _relative_member(name)
        output.extend(_canonical_ustar_header(name, len(raw)))
        output.extend(raw)
        output.extend(b"\0" * ((-len(raw)) % 512))
    output.extend(b"\0" * 1024)
    return bytes(output)


def build_source_archive(repo_root: str | Path, output_value: str | Path) -> Mapping[str, Any]:
    output = Path(output_value)
    if not output.is_absolute() or not output.parent.is_dir() or output.parent.is_symlink():
        raise ReleaseError("archive output parent differs")
    if output.exists() or output.is_symlink():
        raise ReleaseError("archive output must be fresh")
    payloads = _archive_payloads(repo_root)
    archive_raw = canonical_ustar_bytes(payloads)
    handle = tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=".%s." % output.name, delete=False
    )
    temporary = Path(handle.name)
    try:
        handle.write(archive_raw)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        # link is the portable no-replace publication primitive used here.
        os.link(temporary, output)
    except Exception:
        # In particular, never unlink ``output`` here: if link(2) lost an
        # EEXIST race, that path belongs to the competing publisher.
        raise
    finally:
        if not handle.closed:
            handle.close()
        if temporary.exists():
            temporary.unlink()
    os.chmod(output, 0o444)
    raw = read_stable_file(output, label="built source archive")
    return {
        "path": str(output),
        "sha256": sha256_bytes(raw),
        "size": len(raw),
        "mode": "0444",
        "file_count": len(payloads),
    }


def _expected_archive_payloads() -> Mapping[str, Tuple[str, int]]:
    # The generated manifest is checked from its contents below; its byte hash
    # is intentionally derived rather than duplicated as another hand pin.
    expected: Dict[str, Tuple[str, int]] = {
        REVIEW_SNAPSHOT_PATH: (REVIEW_SNAPSHOT_SHA256, -1),
        SOURCE_AUTHORITY_PATH: (SOURCE_AUTHORITY_SHA256, SOURCE_AUTHORITY_SIZE),
    }
    expected.update({path: (digest, size) for path, digest, size in REVIEW_MEMBERS})
    expected[ARCHIVE_MANIFEST_PATH] = ("", -1)
    return expected


def audit_source_archive(
    archive_value: str | Path, *, require_frozen_archive_sha: bool = True
) -> Mapping[str, bytes]:
    archive_path = Path(archive_value)
    try:
        archive_info = archive_path.lstat()
    except OSError as error:
        raise ReleaseError("source archive metadata is unavailable: %s" % error) from error
    if (
        not stat.S_ISREG(archive_info.st_mode)
        or stat.S_ISLNK(archive_info.st_mode)
        or archive_info.st_nlink != 1
        or stat.S_IMODE(archive_info.st_mode) != 0o444
    ):
        raise ReleaseError("source archive publication metadata differs")
    expected_archive_sha = SOURCE_ARCHIVE_SHA256 if require_frozen_archive_sha else None
    raw_archive = read_stable_file(
        archive_path,
        label="source archive",
        expected_sha256=expected_archive_sha,
    )
    expected = _expected_archive_payloads()
    payloads: Dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(raw_archive), mode="r:") as archive:
        members = archive.getmembers()
        names = [entry.name for entry in members]
        if names != sorted(expected) or set(names) != set(expected) or len(names) != len(set(names)):
            raise ReleaseError("source archive member closure/order differs")
        for entry in members:
            _relative_member(entry.name)
            if (
                not entry.isreg()
                or entry.mode != 0o444
                or entry.uid != 0
                or entry.gid != 0
                or entry.uname != ""
                or entry.gname != ""
                or entry.mtime != 0
            ):
                raise ReleaseError("source archive metadata differs: %s" % entry.name)
            stream = archive.extractfile(entry)
            if stream is None:
                raise ReleaseError("source archive member is unreadable: %s" % entry.name)
            payload = stream.read()
            if len(payload) != entry.size:
                raise ReleaseError("source archive member size changed: %s" % entry.name)
            payloads[entry.name] = payload
    for name, (digest, size) in expected.items():
        payload = payloads[name]
        if digest and sha256_bytes(payload) != digest:
            raise ReleaseError("source archive member SHA differs: %s" % name)
        if size >= 0 and len(payload) != size:
            raise ReleaseError("source archive member size differs: %s" % name)
    _review_snapshot_contract(payloads[REVIEW_SNAPSHOT_PATH])
    manifest = parse_json_bytes(payloads[ARCHIVE_MANIFEST_PATH], "archive manifest")
    expected_manifest_members = [
        {"path": name, "sha256": sha256_bytes(payload), "size": len(payload)}
        for name, payload in sorted(payloads.items())
        if name != ARCHIVE_MANIFEST_PATH
    ]
    expected_snapshot = {
        "path": REVIEW_SNAPSHOT_PATH,
        "sha256": REVIEW_SNAPSHOT_SHA256,
        "size": len(payloads[REVIEW_SNAPSHOT_PATH]),
    }
    expected_authority = {
        "path": SOURCE_AUTHORITY_PATH,
        "sha256": SOURCE_AUTHORITY_SHA256,
        "size": SOURCE_AUTHORITY_SIZE,
    }
    if (
        not isinstance(manifest, Mapping)
        or set(manifest)
        != {
            "schema_version",
            "claim",
            "formal_or_scientific_authority",
            "method_source_revision",
            "review_snapshot",
            "source_authority",
            "members",
        }
        or manifest.get("schema_version")
        != "full644-exploratory-r64-source-manifest-v1"
        or manifest.get("claim") != "EXPOSED_FULL644_EXPLORATORY_ABLATION_ONLY"
        or manifest.get("formal_or_scientific_authority") is not False
        or manifest.get("method_source_revision") != METHOD_SOURCE_REVISION
        or manifest.get("review_snapshot") != expected_snapshot
        or manifest.get("source_authority") != expected_authority
        or manifest.get("members") != expected_manifest_members
    ):
        raise ReleaseError("source archive manifest closure differs")
    return payloads


def extract_source_archive(archive: str | Path, output_value: str | Path) -> Mapping[str, Any]:
    payloads = audit_source_archive(archive)
    output = Path(output_value)
    if not output.is_absolute() or not output.parent.is_dir() or output.parent.is_symlink():
        raise ReleaseError("extraction output parent differs")
    try:
        output.mkdir(mode=0o700, parents=False, exist_ok=False)
        for name, raw in sorted(payloads.items()):
            target = output.joinpath(*_relative_member(name).parts)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o400,
            )
            try:
                view = memoryview(raw)
                while view:
                    count = os.write(descriptor, view)
                    view = view[count:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.chmod(target, 0o444)
    except Exception:
        # Never remove a partly extracted tree automatically: retaining it is
        # safer evidence, and callers require a new fresh output on retry.
        raise
    actual_files: set[str] = set()
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ReleaseError("extraction contains a symlink: %s" % relative)
        if stat.S_ISDIR(info.st_mode):
            continue
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o444
        ):
            raise ReleaseError("extracted member metadata differs: %s" % relative)
        actual_files.add(relative)
        read_stable_file(
            path,
            label="extracted member %s" % relative,
            expected_sha256=sha256_bytes(payloads[relative]),
            expected_size=len(payloads[relative]),
        )
    if actual_files != set(payloads):
        raise ReleaseError("extracted member closure differs")
    return {"output": str(output), "file_count": len(payloads), "mode": "0700"}


def verify_runtime_inputs(
    dataset_summary_value: str | Path,
    dataset_index_value: str | Path,
    source_authority_value: str | Path,
) -> Mapping[str, Any]:
    summary_raw = read_stable_file(
        dataset_summary_value,
        label="dataset summary",
        expected_sha256=DATASET_SUMMARY_SHA256,
    )
    index_raw = read_stable_file(
        dataset_index_value,
        label="dataset index",
        expected_sha256=DATASET_INDEX_SHA256,
    )
    authority_raw = read_stable_file(
        source_authority_value,
        label="source authority",
        expected_sha256=SOURCE_AUTHORITY_SHA256,
        expected_size=SOURCE_AUTHORITY_SIZE,
    )
    summary = parse_json_bytes(summary_raw, "dataset summary")
    if not isinstance(summary, Mapping):
        raise ReleaseError("full644 dataset summary authority differs")
    unsigned_summary = dict(summary)
    declared_summary_digest = unsigned_summary.pop("summary_digest", None)
    if (
        declared_summary_digest != DATASET_SUMMARY_DIGEST
        or object_sha256(unsigned_summary) != declared_summary_digest
        or summary.get("index_sha256") != DATASET_INDEX_SHA256
        or summary.get("expected_sample_count") != DATASET_ROWS
        or summary.get("materialized_sample_count") != DATASET_ROWS
        or summary.get("missing_sample_count") != 0
        or summary.get("complete") is not True
        or summary.get("schema_version") != "bernini-r-action-vae-dataset-summary-v2"
        or summary.get("preview_only") is not True
        or summary.get("training_authorized") is not False
        or summary.get("training_use_forbidden") is not True
        or summary.get("scientific_claim_authorized") is not False
    ):
        raise ReleaseError("full644 dataset summary authority differs")
    try:
        declared_index = Path(str(summary.get("index_path"))).resolve(strict=True)
        supplied_index = Path(dataset_index_value).resolve(strict=True)
    except OSError as error:
        raise ReleaseError("dataset index path binding is unavailable: %s" % error) from error
    if declared_index != supplied_index:
        raise ReleaseError("dataset summary points to a different dataset index")
    authority = parse_json_bytes(authority_raw, "source authority")
    if not isinstance(authority, Mapping):
        raise ReleaseError("source authority JSON root differs")
    return {
        "dataset_summary_sha256": sha256_bytes(summary_raw),
        "dataset_index_sha256": sha256_bytes(index_raw),
        "source_authority_sha256": sha256_bytes(authority_raw),
        "rows": DATASET_ROWS,
    }


def _plain_absolute_directory(path_value: str | Path, label: str) -> Path:
    path = Path(path_value)
    if (
        not path.is_absolute()
        or not path.is_dir()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise ReleaseError("%s must be one canonical plain directory" % label)
    return path


def _json_file(path: Path, label: str) -> Tuple[Mapping[str, Any], bytes]:
    raw = read_stable_file(path, label=label)
    value = parse_json_bytes(raw, label)
    if not isinstance(value, Mapping):
        raise ReleaseError("%s JSON root differs" % label)
    return value, raw


def _validate_digest_field(value: Mapping[str, Any], field: str, label: str) -> None:
    claimed = value.get(field)
    if not isinstance(claimed, str) or re.fullmatch(r"[0-9a-f]{64}", claimed) is None:
        raise ReleaseError("%s %s differs" % (label, field))
    unsigned = dict(value)
    del unsigned[field]
    if object_sha256(unsigned) != claimed:
        raise ReleaseError("%s %s is invalid" % (label, field))


def _verify_checkpoint(checkpoint: Path, expected_step: int) -> Mapping[str, Any]:
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise ReleaseError("terminal checkpoint directory differs")
    manifest, manifest_raw = _json_file(
        checkpoint / "checkpoint_manifest.json", "checkpoint manifest"
    )
    if (
        set(manifest)
        != {
            "schema_version",
            "global_step",
            "receipt_digest",
            "file_count",
            "entries",
            "manifest_digest",
        }
        or manifest.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA
        or manifest.get("global_step") != expected_step
    ):
        raise ReleaseError("checkpoint manifest contract differs")
    _validate_digest_field(manifest, "manifest_digest", "checkpoint manifest")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or manifest.get("file_count") != len(entries):
        raise ReleaseError("checkpoint manifest entry closure differs")
    entry_paths = [entry.get("path") for entry in entries if isinstance(entry, Mapping)]
    if len(entry_paths) != len(entries) or entry_paths != sorted(entry_paths):
        raise ReleaseError("checkpoint manifest entry order differs")
    seen: set[str] = set()
    expected_directories: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256", "size"}:
            raise ReleaseError("checkpoint manifest entry schema differs")
        relative = entry.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise ReleaseError("checkpoint manifest path differs")
        seen.add(relative)
        member = _relative_member(relative)
        expected_directories.update(
            PurePosixPath(*member.parts[:depth]).as_posix()
            for depth in range(1, len(member.parts))
        )
        raw = read_stable_file(
            checkpoint.joinpath(*PurePosixPath(relative).parts),
            label="checkpoint member %s" % relative,
            expected_sha256=entry.get("sha256"),
            expected_size=entry.get("size"),
        )
        if not raw:
            raise ReleaseError("checkpoint member is empty: %s" % relative)
    required = {"adapter/adapter_config.json", "optimizer.pt", "receipt.json"}
    if not required.issubset(seen) or not any(
        name.startswith("adapter/adapter_model") and name.endswith(".safetensors")
        for name in seen
    ):
        raise ReleaseError("checkpoint payload is incomplete")
    actual: set[str] = set()
    actual_directories: set[str] = set()
    for path in sorted(checkpoint.rglob("*")):
        relative = path.relative_to(checkpoint).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ReleaseError("checkpoint contains a symlink: %s" % relative)
        if stat.S_ISDIR(info.st_mode):
            actual_directories.add(relative)
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ReleaseError("checkpoint contains a non-plain member: %s" % relative)
        if relative != "checkpoint_manifest.json":
            actual.add(relative)
    if actual != seen or actual_directories != expected_directories:
        raise ReleaseError("checkpoint manifest does not close actual membership")
    receipt, receipt_raw = _json_file(checkpoint / "receipt.json", "training receipt")
    _validate_digest_field(receipt, "receipt_digest", "training receipt")
    if manifest.get("receipt_digest") != receipt.get("receipt_digest"):
        raise ReleaseError("checkpoint receipt digest binding differs")
    return {
        "manifest": manifest,
        "manifest_raw": manifest_raw,
        "receipt": receipt,
        "receipt_raw": receipt_raw,
    }


def _verify_output_root_closure(
    output: Path, mode: str
) -> Mapping[int, Mapping[str, Any]]:
    steps = [1] if mode == "capacity-smoke" else list(range(64, 641, 64)) + [644]
    expected_names = {"latest.json"} | {"checkpoint-%08d" % step for step in steps}
    actual_names: set[str] = set()
    for path in output.iterdir():
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ReleaseError("training output root contains a symlink: %s" % path.name)
        if path.name == "latest.json":
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ReleaseError("latest checkpoint pointer metadata differs")
        elif path.name in expected_names:
            if not stat.S_ISDIR(info.st_mode):
                raise ReleaseError("checkpoint root member is not a directory")
        else:
            raise ReleaseError("training output contains an extra root member: %s" % path.name)
        actual_names.add(path.name)
    if actual_names != expected_names:
        raise ReleaseError("training output root closure differs")
    checked: Dict[int, Mapping[str, Any]] = {}
    for step in steps:
        item = _verify_checkpoint(output / ("checkpoint-%08d" % step), step)
        receipt = item["receipt"]
        if (
            receipt.get("schema_version") != TRAIN_RECEIPT_SCHEMA
            or receipt.get("global_step") != step
            or receipt.get("max_steps") != (1 if mode == "capacity-smoke" else 644)
            or receipt.get("method_source_revision") != METHOD_SOURCE_REVISION
            or receipt.get("method_source_archive_sha256") != SOURCE_ARCHIVE_SHA256
            or receipt.get("resumed_from") is not None
        ):
            raise ReleaseError("checkpoint receipt sequence binding differs")
        if mode == "full644":
            profile = receipt.get("exploratory_full644")
            terminal = step == 644
            if (
                not isinstance(profile, Mapping)
                or profile.get("profile") != FULL644_PROFILE
                or profile.get("optimizer_rows_consumed") != step
                or profile.get("next_row_index") != (None if terminal else step)
                or profile.get("row_sequence_prefix") != "0..%d" % (step - 1)
                or profile.get("row_sequence_sha256")
                != object_sha256(list(range(step)))
                or profile.get("no_replacement_within_pass") is not True
                or profile.get("complete_one_pass") is not terminal
                or profile.get("intermediate_checkpoints_archival_only") is not True
                or profile.get("interrupted_run_requires_fresh_step0_restart") is not True
                or profile.get(
                    "indexed_source_and_target_vae_shards_reverified_after_training"
                )
                is not terminal
            ):
                raise ReleaseError("full644 checkpoint sequence differs")
        checked[step] = item
    return checked


def verify_training_output(mode: str, output_value: str | Path) -> Mapping[str, Any]:
    if mode not in ("capacity-smoke", "full644"):
        raise ReleaseError("training output mode differs")
    output = _plain_absolute_directory(output_value, "training output")
    expected_step = 1 if mode == "capacity-smoke" else 644
    checkpoint = output / ("checkpoint-%08d" % expected_step)
    checkpoints = _verify_output_root_closure(output, mode)
    checked = checkpoints[expected_step]
    receipt = checked["receipt"]
    training = receipt.get("training_contract")
    distributed = receipt.get("distributed")
    dataset = receipt.get("dataset")
    optimizer = receipt.get("optimizer")
    if not all(isinstance(value, Mapping) for value in (training, distributed, dataset, optimizer)):
        raise ReleaseError("training receipt nested contract differs")
    summary = dataset.get("summary")
    if not isinstance(summary, Mapping):
        raise ReleaseError("training receipt dataset summary differs")
    expected_receipt_keys = {
        "schema_version",
        "global_step",
        "max_steps",
        "last_loss",
        "last_preclip_gradient_norm",
        "bernini_commit",
        "bernini_training_files_index_sha256",
        "veomni_commit",
        "method_source_revision",
        "method_source_archive_sha256",
        "checkpoint",
        "checkpoint_tree_sha256",
        "dataset",
        "training_contract",
        "optimizer",
        "distributed",
        "seed",
        "target_module_count",
        "target_modules_sha256",
        "trainable_parameter_count",
        "resumed_from",
        "experimental_training",
        "production_claim_forbidden",
        "scientific_claim_authorized",
        "receipt_digest",
    }
    if mode == "full644":
        expected_receipt_keys.add("exploratory_full644")
    expected_training_keys = {
        "model",
        "single_expert",
        "noise_tmin",
        "noise_tmax",
        "mv2v_flow_shift",
        "num_frames",
        "latent_frames",
        "task_source_name",
        "external_spatial_mask",
        "external_tracking_or_swept_tube",
        "conditioning",
        "supervision",
        "target_embedding_or_caption_conditioning",
        "lora_rank",
        "lora_alpha",
        "lora_scope",
        "tokenizer_fix_mistral_regex",
        "peft_version",
        "transformers_version",
        "gradient_checkpointing",
        "objective",
        "preference_weight",
        "preference_margin",
        "preference_temperature",
        "dpo_beta",
        "preservation_weight",
        "contrastive_negative_kinds",
        "contrastive_negative_schedule",
        "preservation_branch",
    }
    expected_distributed_keys = {
        "world_size",
        "ulysses_size",
        "backend",
        "same_sample_all_ranks",
        "same_seed_all_ranks",
        "lora_initialization_seeded_all_ranks",
        "lora_parameters_broadcast_from_rank",
        "lora_initialization_digest",
        "explicit_lora_gradient_all_reduce",
    }
    expected_summary_keys = {
        "path",
        "sha256",
        "summary_digest",
        "complete",
        "allow_incomplete",
        "expected_rows",
        "materialized_rows",
        "index_path",
        "index_sha256",
        "indexed_shards_sha256",
        "dataset_content_signature",
        "reward_selected_synthetic_targets",
        "arm",
    }
    fingerprint = receipt.get("checkpoint")
    if (
        set(receipt) != expected_receipt_keys
        or set(training) != expected_training_keys
        or set(distributed) != expected_distributed_keys
        or set(dataset) != {"path", "rows", "signature", "content_signature", "summary"}
        or set(summary) != expected_summary_keys
        or set(optimizer) != {"type", "learning_rate", "weight_decay", "max_gradient_norm"}
        or not isinstance(fingerprint, Mapping)
        or set(fingerprint) != {"path", "configs"}
        or not isinstance(fingerprint.get("configs"), Mapping)
    ):
        raise ReleaseError("training receipt exact schema closure differs")
    common_ok = (
        receipt.get("schema_version") == TRAIN_RECEIPT_SCHEMA
        and receipt.get("global_step") == expected_step
        and receipt.get("max_steps") == expected_step
        and receipt.get("method_source_revision") == METHOD_SOURCE_REVISION
        and receipt.get("method_source_archive_sha256") == SOURCE_ARCHIVE_SHA256
        and receipt.get("bernini_commit") == BERNINI_COMMIT
        and receipt.get("bernini_training_files_index_sha256")
        == BERNINI_TRAINING_FILES_INDEX_SHA256
        and receipt.get("veomni_commit") == VEOMNI_COMMIT
        and receipt.get("checkpoint_tree_sha256") == CHECKPOINT_TREE_SHA256
        and receipt.get("resumed_from") is None
        and receipt.get("target_module_count") == 240
        and isinstance(receipt.get("target_modules_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", receipt["target_modules_sha256"])
        is not None
        and receipt.get("trainable_parameter_count") == TRAINABLE_PARAMETER_COUNT
        and receipt.get("experimental_training") is True
        and receipt.get("production_claim_forbidden") is True
        and receipt.get("scientific_claim_authorized") is False
        and training.get("lora_rank") == LORA_RANK
        and training.get("lora_alpha") == LORA_ALPHA
        and training.get("peft_version") == PEFT_VERSION
        and training.get("model") == "Bernini-R-1.3B-Diffusers renderer-only"
        and training.get("single_expert") == "transformer_1"
        and training.get("noise_tmin") == 0.0
        and training.get("noise_tmax") == 1.0
        and training.get("mv2v_flow_shift") == 5.0
        and training.get("num_frames") == 81
        and training.get("latent_frames") == 21
        and training.get("task_source_name") == "mv2v$action_editing_81f"
        and training.get("external_spatial_mask") is False
        and training.get("external_tracking_or_swept_tube") is False
        and training.get("conditioning")
        == ["clean_source_video_vae", "edit_instruction"]
        and training.get("supervision")
        == ["noisy_target_video_vae", "target_velocity"]
        and training.get("target_embedding_or_caption_conditioning") is False
        and training.get("lora_scope")
        == "all Wan attn1/attn2 q,k,v,out projections"
        and training.get("tokenizer_fix_mistral_regex") is True
        and isinstance(training.get("transformers_version"), str)
        and bool(training.get("transformers_version"))
        and training.get("gradient_checkpointing") is True
        and training.get("preference_weight") == 1.0
        and training.get("preference_margin") == 0.05
        and training.get("preference_temperature") == 20.0
        and training.get("dpo_beta") == 10.0
        and training.get("preservation_weight") == 0.25
        and training.get("contrastive_negative_kinds")
        == ["noop", "reverse", "incomplete"]
        and training.get("contrastive_negative_schedule") == "rotate"
        and distributed.get("world_size") == WORLD_SIZE
        and distributed.get("ulysses_size") == ULYSSES_SIZE
        and distributed.get("backend") == "nccl/rccl"
        and distributed.get("explicit_lora_gradient_all_reduce") is True
        and distributed.get("same_sample_all_ranks") is True
        and distributed.get("same_seed_all_ranks") is True
        and distributed.get("lora_initialization_seeded_all_ranks") is True
        and distributed.get("lora_parameters_broadcast_from_rank") == 0
        and isinstance(distributed.get("lora_initialization_digest"), str)
        and re.fullmatch(r"[0-9a-f]{64}", distributed["lora_initialization_digest"])
        is not None
        and dataset.get("rows") == DATASET_ROWS
        and isinstance(dataset.get("path"), str)
        and str(dataset.get("path", "")).startswith("/")
        and isinstance(dataset.get("signature"), str)
        and bool(dataset.get("signature"))
        and isinstance(dataset.get("content_signature"), str)
        and bool(dataset.get("content_signature"))
        and summary.get("complete") is True
        and summary.get("allow_incomplete") is False
        and summary.get("expected_rows") == DATASET_ROWS
        and summary.get("materialized_rows") == DATASET_ROWS
        and summary.get("sha256") == DATASET_SUMMARY_SHA256
        and summary.get("summary_digest") == DATASET_SUMMARY_DIGEST
        and summary.get("index_sha256") == DATASET_INDEX_SHA256
        and isinstance(summary.get("path"), str)
        and str(summary.get("path", "")).startswith("/")
        and isinstance(summary.get("index_path"), str)
        and str(summary.get("index_path", "")).startswith("/")
        and isinstance(summary.get("indexed_shards_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", summary["indexed_shards_sha256"])
        is not None
        and summary.get("dataset_content_signature") == dataset.get("content_signature")
        and summary.get("reward_selected_synthetic_targets") is False
        and summary.get("arm") is None
        and optimizer.get("type") == "AdamW"
        and optimizer.get("learning_rate") == 1.0e-4
        and optimizer.get("weight_decay") == 0.0
        and optimizer.get("max_gradient_norm") == 1.0
        and receipt.get("seed") == 20260817
    )
    if not common_ok:
        raise ReleaseError("training receipt common frozen contract differs")
    loss = receipt.get("last_loss")
    grad = receipt.get("last_preclip_gradient_norm")
    if type(loss) not in (int, float) or not math.isfinite(loss):
        raise ReleaseError("terminal loss is not finite")
    if type(grad) not in (int, float) or not math.isfinite(grad) or grad <= 0:
        raise ReleaseError("terminal gradient norm does not prove a real optimizer update")
    full = receipt.get("exploratory_full644")
    if mode == "capacity-smoke":
        if (
            training.get("objective") != "sft"
            or training.get("preservation_branch") is not None
            or full is not None
        ):
            raise ReleaseError("capacity smoke was mislabeled as full644")
    else:
        if (
            training.get("objective") != "reference_dpo_preservation"
            or training.get("preservation_branch")
            != "source_as_target_conditional_identity"
            or not isinstance(full, Mapping)
            or full.get("profile") != FULL644_PROFILE
            or full.get("optimizer_rows_consumed") != 644
            or full.get("complete_one_pass") is not True
            or full.get("row_sequence_prefix") != "0..643"
            or full.get("no_replacement_within_pass") is not True
            or full.get("resume_policy") != "forbidden_for_this_profile"
            or full.get("dataset_quality_accepted_under_0817") is not False
            or full.get("formal_training_dataset_authorized") is not False
            or full.get("formal_heldout_contribution") != 0
            or full.get("dataset_summary_sha256") != DATASET_SUMMARY_SHA256
            or full.get("dataset_summary_digest") != DATASET_SUMMARY_DIGEST
            or full.get("dataset_index_sha256") != DATASET_INDEX_SHA256
            or full.get("indexed_source_and_target_vae_shards_reverified_after_training")
            is not True
            or not isinstance(full.get("source_authority"), Mapping)
            or full["source_authority"].get("sha256") != SOURCE_AUTHORITY_SHA256
        ):
            raise ReleaseError("terminal full644 one-pass contract differs")
    latest, latest_raw = _json_file(output / "latest.json", "latest checkpoint pointer")
    if (
        latest.get("checkpoint") != str(checkpoint)
        or latest.get("global_step") != expected_step
        or latest.get("checkpoint_manifest_path") != str(checkpoint / "checkpoint_manifest.json")
        or latest.get("checkpoint_manifest_sha256")
        != sha256_bytes(checked["manifest_raw"])
        or latest.get("checkpoint_receipt_sha256") != sha256_bytes(checked["receipt_raw"])
    ):
        raise ReleaseError("latest checkpoint pointer binding differs")
    return {
        "mode": mode,
        "output": str(output),
        "global_step": expected_step,
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": sha256_bytes(checked["manifest_raw"]),
        "training_receipt_sha256": sha256_bytes(checked["receipt_raw"]),
        "optimizer_update_proven": True,
        "scientific_claim_authorized": False,
    }


def _atomic_create_json(path_value: str | Path, value: Mapping[str, Any]) -> str:
    path = Path(path_value)
    if (
        not path.is_absolute()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.parent.resolve(strict=True) != path.parent
    ):
        raise ReleaseError("receipt output parent differs")
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        os.fchmod(descriptor, 0o400)
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            view = view[count:]
        os.fsync(descriptor)
        written = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        info = path.lstat()
    except OSError as error:
        raise ReleaseError("published receipt disappeared: %s" % error) from error
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o400
        or (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        != (written.st_dev, written.st_ino, written.st_size, written.st_mtime_ns)
    ):
        raise ReleaseError("published receipt identity/mode differs")
    reread = read_stable_file(
        path,
        label="published receipt",
        expected_sha256=sha256_bytes(payload),
        expected_size=len(payload),
    )
    if reread != payload:
        raise ReleaseError("published receipt bytes differ")
    return sha256_bytes(payload)


def _runner_completion_value(
    mode: str,
    output: str | Path,
    cache_receipt: str | Path,
    *,
    slurm_job_id: str,
    slurm_step_id: str,
    node: str,
) -> Dict[str, Any]:
    if (
        slurm_job_id != HOLDER_JOB_ID
        or node != HOLDER_NODE
        or re.fullmatch(r"[0-9]+", slurm_step_id) is None
    ):
        raise ReleaseError("runner is outside frozen job/node/numeric-step authority")
    verified = verify_training_output(mode, output)
    cache_verified = verify_rank_cache_receipt(
        cache_receipt,
        mode=mode,
        slurm_job_id=slurm_job_id,
        slurm_step_id=slurm_step_id,
        node=node,
    )
    receipt: Dict[str, Any] = {
        "schema_version": "full644-exploratory-r64-runner-completion-v1",
        "status": (
            "PRELAUNCH_CAPACITY_ONLY_COMPLETE"
            if mode == "capacity-smoke"
            else "EXPOSED_FULL644_EXPLORATORY_ABLATION_COMPLETE"
        ),
        "mode": mode,
        "slurm_job_id": slurm_job_id,
        "slurm_step_id": slurm_step_id,
        "node": node,
        "world_size": WORLD_SIZE,
        "ulysses_size": ULYSSES_SIZE,
        "method_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "review_snapshot_sha256": REVIEW_SNAPSHOT_SHA256,
        "training_output": dict(verified),
        "rank_cache_receipt": dict(cache_verified),
        "prelaunch_capacity_only": mode == "capacity-smoke",
        "this_receipt_authorizes_full644_training_result": False,
        "scientific_claim_authorized": False,
        "formal_claim_authorized": False,
        "formal_or_scientific_authority": False,
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    return receipt


def publish_runner_completion(
    mode: str,
    output: str | Path,
    receipt_output: str | Path,
    cache_receipt: str | Path,
    *,
    slurm_job_id: str,
    slurm_step_id: str,
    node: str,
) -> Mapping[str, Any]:
    receipt = _runner_completion_value(
        mode,
        output,
        cache_receipt,
        slurm_job_id=slurm_job_id,
        slurm_step_id=slurm_step_id,
        node=node,
    )
    file_sha = _atomic_create_json(receipt_output, receipt)
    return {"path": str(receipt_output), "sha256": file_sha, "receipt": receipt}


def audit_runner_completion(
    mode: str,
    output: str | Path,
    receipt_value: str | Path,
    cache_receipt: str | Path,
    *,
    slurm_job_id: str,
    slurm_step_id: str,
    node: str,
) -> Mapping[str, Any]:
    expected = _runner_completion_value(
        mode,
        output,
        cache_receipt,
        slurm_job_id=slurm_job_id,
        slurm_step_id=slurm_step_id,
        node=node,
    )
    receipt_path = Path(receipt_value)
    observed, raw = _json_file(receipt_path, "runner completion")
    _validate_digest_field(observed, "receipt_digest", "runner completion")
    info = receipt_path.lstat()
    if (
        observed != expected
        or raw != canonical_json_bytes(expected) + b"\n"
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o400
    ):
        raise ReleaseError("runner completion bytes/bindings differ on re-audit")
    return {
        "mode": mode,
        "status": expected["status"],
        "receipt": str(receipt_path),
        "sha256": sha256_bytes(raw),
        "slurm_job_step": "%s.%s" % (slurm_job_id, slurm_step_id),
        "formal_or_scientific_authority": False,
    }


def parse_max_rss(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:[.][0-9]+)?)([KMGTP]?)", value.strip())
    if match is None:
        raise ReleaseError("Slurm MaxRSS encoding differs")
    number = float(match.group(1))
    if not math.isfinite(number) or number <= 0:
        raise ReleaseError("Slurm MaxRSS is not positive")
    power = {"": 1, "K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[match.group(2)]
    # Slurm accounting units are binary; a bare number is KiB.
    return int(number * (1024**power))


def verify_rank_cache_receipt(
    path_value: str | Path,
    *,
    mode: str,
    slurm_job_id: str,
    slurm_step_id: str,
    node: str,
) -> Mapping[str, Any]:
    if mode not in ("capacity-smoke", "full644"):
        raise ReleaseError("rank-cache mode differs")
    expected_mode = (
        "PRELAUNCH_CAPACITY_ONLY" if mode == "capacity-smoke" else "FULL644_EXPLORATORY"
    )
    if (
        slurm_job_id != HOLDER_JOB_ID
        or node != HOLDER_NODE
        or re.fullmatch(r"[0-9]+", slurm_step_id) is None
    ):
        raise ReleaseError("rank-cache Slurm binding differs")
    path = Path(path_value)
    value, raw = _json_file(path, "rank-cache receipt")
    _validate_digest_field(value, "receipt_digest", "rank-cache receipt")
    expected_keys = {
        "schema_version",
        "mode",
        "job_id",
        "step_id",
        "node",
        "filesystem_type",
        "cache_root",
        "cache_root_device",
        "cache_root_inode",
        "cache_root_uid",
        "cache_root_mode",
        "rank_caches",
        "world_size",
        "rank_local",
        "scheduler_tmpdir_observed",
        "scheduler_tmpdir_normalized_to_unset",
        "receipt_digest",
    }
    filesystem_type = value.get("filesystem_type")
    cache_uid = value.get("cache_root_uid")
    expected_root = (
        "/tmp/cache/full644-r64-u%s-j%s-s%s-v1"
        % (cache_uid, HOLDER_JOB_ID, slurm_step_id)
    )
    bad_filesystems = ("nfs", "lustre", "gpfs", "cifs", "smb", "fuse", "autofs")
    if (
        set(value) != expected_keys
        or value.get("schema_version") != "full644-r64-rank-cache-receipt-v1"
        or value.get("mode") != expected_mode
        or value.get("job_id") != HOLDER_JOB_ID
        or value.get("step_id") != slurm_step_id
        or value.get("node") != HOLDER_NODE
        or type(filesystem_type) is not str
        or not filesystem_type
        or any(filesystem_type.lower().startswith(prefix) for prefix in bad_filesystems)
        or type(cache_uid) is not int
        or cache_uid < 0
        or value.get("cache_root") != expected_root
        or type(value.get("cache_root_device")) is not int
        or value.get("cache_root_device", -1) < 0
        or type(value.get("cache_root_inode")) is not int
        or value.get("cache_root_inode", 0) <= 0
        or value.get("cache_root_mode") != "0700"
        or value.get("world_size") != WORLD_SIZE
        or value.get("rank_local") is not True
        or value.get("scheduler_tmpdir_observed") not in ("absent", "/tmp")
        or value.get("scheduler_tmpdir_normalized_to_unset") is not True
    ):
        raise ReleaseError("rank-cache receipt contract differs")
    rows = value.get("rank_caches")
    if not isinstance(rows, list) or len(rows) != WORLD_SIZE:
        raise ReleaseError("rank-cache row closure differs")
    seen_inodes: set[int] = set()
    for rank, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"rank", "path", "device", "inode", "uid", "mode"}
            or row.get("rank") != rank
            or row.get("path") != "%s/rank-%d" % (expected_root, rank)
            or row.get("device") != value.get("cache_root_device")
            or row.get("uid") != cache_uid
            or row.get("mode") != "0700"
            or type(row.get("inode")) is not int
            or row.get("inode", 0) <= 0
            or row.get("inode") in seen_inodes
        ):
            raise ReleaseError("rank-cache row differs")
        seen_inodes.add(row["inode"])
    info = path.lstat()
    if (
        not path.is_absolute()
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o400
        or raw != canonical_json_bytes(value) + b"\n"
    ):
        raise ReleaseError("rank-cache receipt publication differs")
    return {
        "path": str(path),
        "sha256": sha256_bytes(raw),
        "receipt_digest": value["receipt_digest"],
        "schema_version": value["schema_version"],
        "mode": value["mode"],
        "world_size": WORLD_SIZE,
        "rank_local": True,
    }


def _load_runner_completion(path_value: str | Path) -> Mapping[str, Any]:
    path = Path(path_value)
    value, raw = _json_file(path, "runner completion")
    _validate_digest_field(value, "receipt_digest", "runner completion")
    if (
        set(value)
        != {
            "schema_version",
            "status",
            "mode",
            "slurm_job_id",
            "slurm_step_id",
            "node",
            "world_size",
            "ulysses_size",
            "method_source_archive_sha256",
            "review_snapshot_sha256",
            "training_output",
            "rank_cache_receipt",
            "prelaunch_capacity_only",
            "this_receipt_authorizes_full644_training_result",
            "scientific_claim_authorized",
            "formal_claim_authorized",
            "formal_or_scientific_authority",
            "receipt_digest",
        }
        or value.get("schema_version")
        != "full644-exploratory-r64-runner-completion-v1"
        or value.get("status") != "PRELAUNCH_CAPACITY_ONLY_COMPLETE"
        or value.get("mode") != "capacity-smoke"
        or value.get("slurm_job_id") != HOLDER_JOB_ID
        or value.get("node") != HOLDER_NODE
        or value.get("world_size") != WORLD_SIZE
        or value.get("ulysses_size") != ULYSSES_SIZE
        or value.get("method_source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or value.get("review_snapshot_sha256") != REVIEW_SNAPSHOT_SHA256
        or value.get("prelaunch_capacity_only") is not True
        or value.get("this_receipt_authorizes_full644_training_result") is not False
        or value.get("scientific_claim_authorized") is not False
        or value.get("formal_claim_authorized") is not False
        or value.get("formal_or_scientific_authority") is not False
    ):
        raise ReleaseError("capacity runner completion differs")
    cache_binding = value.get("rank_cache_receipt")
    if not isinstance(cache_binding, Mapping):
        raise ReleaseError("capacity runner rank-cache binding differs")
    observed_cache = verify_rank_cache_receipt(
        str(cache_binding.get("path", "")),
        mode="capacity-smoke",
        slurm_job_id=HOLDER_JOB_ID,
        slurm_step_id=str(value.get("slurm_step_id", "")),
        node=HOLDER_NODE,
    )
    if cache_binding != observed_cache:
        raise ReleaseError("capacity runner rank-cache binding changed")
    info = path.lstat()
    if (
        info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o400
        or raw != canonical_json_bytes(value) + b"\n"
    ):
        raise ReleaseError("capacity runner completion publication differs")
    return value


def _capacity_gate_value(
    training_output: str | Path,
    runner_completion_value: str | Path,
    sacct_row_file_value: str | Path,
    slurm_step_id: str,
) -> Dict[str, Any]:
    if re.fullmatch(r"[0-9]+", slurm_step_id) is None:
        raise ReleaseError("capacity Slurm step id differs")
    verified = verify_training_output("capacity-smoke", training_output)
    completion = _load_runner_completion(runner_completion_value)
    if (
        completion.get("slurm_step_id") != slurm_step_id
        or completion.get("training_output") != verified
    ):
        raise ReleaseError("capacity completion/output binding differs")
    sacct_raw = read_stable_file(sacct_row_file_value, label="capacity sacct row")
    try:
        text = sacct_raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ReleaseError("capacity sacct row is not ASCII") from error
    rows: List[List[str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        # ``sacct -P`` commonly emits one terminal delimiter.  Accept exactly
        # that empty fifth field, while still rejecting any real extra column.
        if len(fields) == 5 and fields[-1] == "":
            fields.pop()
        rows.append(fields)
    expected_job_step = "%s.%s" % (HOLDER_JOB_ID, slurm_step_id)
    selected = [row for row in rows if row and row[0] == expected_job_step]
    if len(selected) != 1 or len(selected[0]) != 4:
        raise ReleaseError("capacity sacct exact-row closure differs")
    job_step, state, exit_code, max_rss = selected[0]
    if state != "COMPLETED" or exit_code != "0:0":
        raise ReleaseError("capacity smoke did not terminate successfully")
    max_rss_bytes = parse_max_rss(max_rss)
    if max_rss_bytes > CAPACITY_MAX_RSS_BYTES:
        raise ReleaseError(
            "capacity MaxRSS exceeds frozen headroom gate: %d > %d"
            % (max_rss_bytes, CAPACITY_MAX_RSS_BYTES)
        )
    gate: Dict[str, Any] = {
        "schema_version": "full644-exploratory-r64-capacity-gate-v1",
        "status": "PASS_FULL644_MAY_START_FRESH",
        "scope": "PRELAUNCH_CAPACITY_ONLY",
        "slurm_job_step": job_step,
        "state": state,
        "exit_code": exit_code,
        "max_rss_raw": max_rss,
        "max_rss_bytes": max_rss_bytes,
        "allocated_bytes": CAPACITY_ALLOCATED_BYTES,
        "minimum_headroom_bytes": CAPACITY_MIN_HEADROOM_BYTES,
        "observed_headroom_bytes": CAPACITY_ALLOCATED_BYTES - max_rss_bytes,
        "capacity_training_output": verified,
        "runner_completion_receipt_digest": completion.get("receipt_digest"),
        "sacct_row_file_sha256": sha256_bytes(sacct_raw),
        "capacity_smoke_is_full644_training": False,
        "full644_training_result_exists": False,
        "fresh_full644_start_prerequisite_satisfied": True,
        "scientific_claim_authorized": False,
        "formal_claim_authorized": False,
        "formal_or_scientific_authority": False,
    }
    gate["receipt_digest"] = object_sha256(gate)
    return gate


def seal_capacity_gate(
    training_output: str | Path,
    runner_completion_value: str | Path,
    sacct_row_file_value: str | Path,
    slurm_step_id: str,
    output_value: str | Path,
) -> Mapping[str, Any]:
    gate = _capacity_gate_value(
        training_output,
        runner_completion_value,
        sacct_row_file_value,
        slurm_step_id,
    )
    file_sha = _atomic_create_json(output_value, gate)
    return {"path": str(output_value), "sha256": file_sha, "gate": gate}


def audit_capacity_gate(
    training_output: str | Path,
    runner_completion_value: str | Path,
    sacct_row_file_value: str | Path,
    slurm_step_id: str,
    gate_value: str | Path,
) -> Mapping[str, Any]:
    expected = _capacity_gate_value(
        training_output,
        runner_completion_value,
        sacct_row_file_value,
        slurm_step_id,
    )
    observed, raw = _json_file(Path(gate_value), "capacity gate")
    _validate_digest_field(observed, "receipt_digest", "capacity gate")
    if observed != expected or raw != canonical_json_bytes(expected) + b"\n":
        raise ReleaseError("capacity gate bytes/bindings differ on re-audit")
    info = Path(gate_value).lstat()
    if info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o400:
        raise ReleaseError("capacity gate publication metadata differs")
    return {
        "status": "PASS_FULL644_MAY_START_FRESH",
        "gate": str(gate_value),
        "sha256": sha256_bytes(raw),
        "max_rss_bytes": expected["max_rss_bytes"],
        "minimum_headroom_bytes": CAPACITY_MIN_HEADROOM_BYTES,
    }


def _print_json(value: Any) -> None:
    print(canonical_json_bytes(value).decode("ascii"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-source-archive")
    build.add_argument("--repo-root", required=True)
    build.add_argument("--output", required=True)
    audit = sub.add_parser("audit-source-archive")
    audit.add_argument("--archive", required=True)
    extract = sub.add_parser("extract-source-archive")
    extract.add_argument("--archive", required=True)
    extract.add_argument("--output", required=True)
    runtime = sub.add_parser("verify-runtime-inputs")
    runtime.add_argument("--dataset-summary", required=True)
    runtime.add_argument("--dataset-index", required=True)
    runtime.add_argument("--source-authority", required=True)
    verify = sub.add_parser("verify-training-output")
    verify.add_argument("--mode", choices=("capacity-smoke", "full644"), required=True)
    verify.add_argument("--output", required=True)
    publish = sub.add_parser("publish-runner-completion")
    publish.add_argument("--mode", choices=("capacity-smoke", "full644"), required=True)
    publish.add_argument("--output", required=True)
    publish.add_argument("--receipt-output", required=True)
    publish.add_argument("--cache-receipt", required=True)
    publish.add_argument("--slurm-job-id", required=True)
    publish.add_argument("--slurm-step-id", required=True)
    publish.add_argument("--node", required=True)
    completion_audit = sub.add_parser("audit-runner-completion")
    completion_audit.add_argument(
        "--mode", choices=("capacity-smoke", "full644"), required=True
    )
    completion_audit.add_argument("--output", required=True)
    completion_audit.add_argument("--receipt", required=True)
    completion_audit.add_argument("--cache-receipt", required=True)
    completion_audit.add_argument("--slurm-job-id", required=True)
    completion_audit.add_argument("--slurm-step-id", required=True)
    completion_audit.add_argument("--node", required=True)
    gate = sub.add_parser("seal-capacity-gate")
    gate.add_argument("--training-output", required=True)
    gate.add_argument("--runner-completion", required=True)
    gate.add_argument("--sacct-row-file", required=True)
    gate.add_argument("--slurm-step-id", required=True)
    gate.add_argument("--output", required=True)
    gate_audit = sub.add_parser("audit-capacity-gate")
    gate_audit.add_argument("--training-output", required=True)
    gate_audit.add_argument("--runner-completion", required=True)
    gate_audit.add_argument("--sacct-row-file", required=True)
    gate_audit.add_argument("--slurm-step-id", required=True)
    gate_audit.add_argument("--gate", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build-source-archive":
            result = build_source_archive(args.repo_root, args.output)
        elif args.command == "audit-source-archive":
            result = {
                "archive": args.archive,
                "sha256": SOURCE_ARCHIVE_SHA256,
                "file_count": len(audit_source_archive(args.archive)),
            }
        elif args.command == "extract-source-archive":
            result = extract_source_archive(args.archive, args.output)
        elif args.command == "verify-runtime-inputs":
            result = verify_runtime_inputs(
                args.dataset_summary, args.dataset_index, args.source_authority
            )
        elif args.command == "verify-training-output":
            result = verify_training_output(args.mode, args.output)
        elif args.command == "publish-runner-completion":
            result = publish_runner_completion(
                args.mode,
                args.output,
                args.receipt_output,
                args.cache_receipt,
                slurm_job_id=args.slurm_job_id,
                slurm_step_id=args.slurm_step_id,
                node=args.node,
            )
        elif args.command == "audit-runner-completion":
            result = audit_runner_completion(
                args.mode,
                args.output,
                args.receipt,
                args.cache_receipt,
                slurm_job_id=args.slurm_job_id,
                slurm_step_id=args.slurm_step_id,
                node=args.node,
            )
        elif args.command == "seal-capacity-gate":
            result = seal_capacity_gate(
                args.training_output,
                args.runner_completion,
                args.sacct_row_file,
                args.slurm_step_id,
                args.output,
            )
        elif args.command == "audit-capacity-gate":
            result = audit_capacity_gate(
                args.training_output,
                args.runner_completion,
                args.sacct_row_file,
                args.slurm_step_id,
                args.gate,
            )
        else:  # pragma: no cover - argparse closure
            raise ReleaseError("unknown command")
    except ReleaseError as error:
        parser.error(str(error))
    _print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
