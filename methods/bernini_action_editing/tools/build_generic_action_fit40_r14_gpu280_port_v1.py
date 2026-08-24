#!/usr/bin/env python3
"""Build the exact18 fit40 R14 port for holder 136309 / gpu280.

The input is the frozen R14 tar/manifest pair.  Four members receive the
smallest registered holder-port substitutions; fourteen members are copied
byte-for-byte.  The source pair, every pre/post image, replacement count,
member rename, and output pair are bound by a canonical port receipt.

This utility creates release artifacts only.  It does not deploy, launch,
authorize review, create Phi/P/O, create an optimizer, or train.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import tempfile
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence


SOURCE_ARCHIVE_SHA256 = "7de587a260cba28b18ce06a41c16350e040e105d39bf9083384dc325a8a8a180"
SOURCE_MANIFEST_SHA256 = "1e9dae7429e2ada55c27e3efdf00c8851141fedd37558c8c76df49063ffa4770"
SOURCE_MANIFEST_DIGEST = "0257b4a988de6b0c802f0b4224bf42255ba5f49594c9e9b4dc60b3be4a92e6b8"
SOURCE_CONTENT_CLOSURE_SHA1 = "f7f5adc4b93cf57380bbf532dcfdb87f998bd447"
SOURCE_FILE_COUNT = 18
MEMBER_ROOT = "methods/bernini_action_editing"
PORT_RECEIPT_SCHEMA = "bernini-generic-action-fit40-r14-gpu280-port-release-v1"
SOURCE_JOB = "136141"
PORT_JOB = "136309"
SOURCE_NODE = "auh7-1b-gpu-299"
PORT_NODE = "auh7-1b-gpu-280"
SOURCE_LAUNCHER = "scripts/auh_generic_action_data_prep_136141_world4_v1.sh"
PORT_LAUNCHER = "scripts/auh_generic_action_data_prep_136309_world4_v1.sh"
PORT_CONFIRMATION = "launch-approved-generic-action-fit40-generation-136309"
PORT_PLAN_ID = "generic-action-fit40-generation-136309-r14-gpu280-p1"


class PortReleaseError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise PortReleaseError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise PortReleaseError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _stable_plain_bytes(path: Path, label: str) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be absolute and non-symlink")
    resolved = path.resolve(strict=True)
    if resolved != path:
        fail(f"{label} canonical path differs")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            fail(f"{label} must be a plain file")
        chunks = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.extend(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise PortReleaseError(f"cannot read {label} stably") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(chunks) != before.st_size
        or not chunks
    ):
        fail(f"{label} changed while reading or is empty")
    return bytes(chunks)


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_source_manifest(raw: bytes) -> dict[str, Any]:
    if hashlib.sha256(raw).hexdigest() != SOURCE_MANIFEST_SHA256:
        fail("frozen R14 source manifest SHA-256 differs")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PortReleaseError(f"non-finite JSON constant is forbidden: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PortReleaseError("frozen R14 source manifest is invalid") from error
    if (
        type(value) is not dict
        or raw != canonical_json_bytes(value) + b"\n"
        or value.get("manifest_digest") != SOURCE_MANIFEST_DIGEST
        or value.get("content_closure_sha1") != SOURCE_CONTENT_CLOSURE_SHA1
        or value.get("file_count") != SOURCE_FILE_COUNT
        or value.get("member_root") != MEMBER_ROOT
        or value.get("exact_member_closure") is not True
        or value.get("release_generation") != "r14"
        or value.get("release_scope")
        != "reserve4-fit40-media-only-pending-external-blind-review"
    ):
        fail("frozen R14 source manifest identity/authority differs")
    unsigned = dict(value)
    declared = unsigned.pop("manifest_digest")
    if declared != object_sha256(unsigned):
        fail("frozen R14 source manifest digest does not replay")
    return value


def _load_source_archive(
    raw: bytes, manifest: Mapping[str, Any]
) -> tuple[dict[str, bytes], dict[str, int]]:
    if hashlib.sha256(raw).hexdigest() != SOURCE_ARCHIVE_SHA256:
        fail("frozen R14 source archive SHA-256 differs")
    rows = manifest.get("files")
    if not isinstance(rows, list) or len(rows) != SOURCE_FILE_COUNT:
        fail("frozen R14 source member rows differ")
    expected = [f"{MEMBER_ROOT}/{row['path']}" for row in rows]
    payloads: dict[str, bytes] = {}
    modes: dict[str, int] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            members = archive.getmembers()
            if [member.name for member in members] != expected:
                fail("frozen R14 archive exact member order/set differs")
            for member, row in zip(members, rows):
                handle = archive.extractfile(member)
                payload = b"" if handle is None else handle.read()
                relative = str(row["path"])
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != 0
                    or member.mode != row["mode"]
                    or member.size != row["size"]
                    or member.pax_headers
                    or hashlib.sha256(payload).hexdigest() != row["sha256"]
                ):
                    fail(f"frozen R14 source member differs: {member.name}")
                payloads[relative] = payload
                modes[relative] = int(row["mode"])
    except (OSError, tarfile.TarError) as error:
        raise PortReleaseError("cannot verify frozen R14 archive") from error
    return payloads, modes


# A SHA-pinned preimage makes every byte operation exact.  Counts below are
# checked immediately before each replacement and all old holder/node tokens
# are rejected after the complete release projection.
TRANSFORMS: Mapping[str, tuple[tuple[bytes, bytes, int, str], ...]] = {
    "generic_action_data_prep_controller_v1.py": (
        (
            b"generic-action-fit40-generation-136141-r14",
            b"generic-action-fit40-generation-136309-r14-gpu280-p1",
            1,
            "plan-id",
        ),
        (b"136141", b"136309", 6, "controller-job-schema-confirm-launcher"),
        (b"auh7-1b-gpu-299", b"auh7-1b-gpu-280", 1, "controller-node"),
    ),
    "tools/reserve4_fixed_generation_sp4_v1.py": (
        (b"136141", b"136309", 9, "generator-eight-bindings-plus-error"),
    ),
    "tools/build_generic_action_data_prep_release_v1.py": (
        (b"136141", b"136309", 4, "builder-launcher-member-entrypoint"),
    ),
    SOURCE_LAUNCHER: (
        (b"136141", b"136309", 4, "launcher-job-confirm-self"),
        (b"gpu299", b"gpu280", 1, "launcher-doc-node"),
        (b"gpu-299", b"gpu-280", 1, "launcher-node"),
    ),
}
EXPECTED_PREIMAGE_SHA256: Mapping[str, str] = {
    "generic_action_data_prep_controller_v1.py": "764346494d4e949a125b2e9d4991d451aac03f5e4c5ccf7fdd364360d933ac23",
    "tools/reserve4_fixed_generation_sp4_v1.py": "4e27afcb154ba002feb2f8eaaf80203a28813582a214c27fd3674dba127505f7",
    "tools/build_generic_action_data_prep_release_v1.py": "2e17019fa63d7387c389350613a0e3c8d99e96dc44a8e4b9ce64dbe9967c7955",
    SOURCE_LAUNCHER: "1d4bc30ff20d18b1d44d2cbe1a91305ba15d63980c6070ad59871da3e17ac994",
}
EXPECTED_POSTIMAGE_SHA256: Mapping[str, str] = {
    "generic_action_data_prep_controller_v1.py": "37b9fe18116d793f6814e2dd0ea3e7123b183a47c753c67d007173e920cb1ca1",
    "tools/reserve4_fixed_generation_sp4_v1.py": "8d5251b70bb97454afe21b5419a76c15c95d5581e8d11eab8a45f16ab8cb9f58",
    "tools/build_generic_action_data_prep_release_v1.py": "0f0ea55601f6fc53cbbb8995907bc1ec6b76ff170a181691f832e5c8d993d61c",
    PORT_LAUNCHER: "8ed4d7bde14bbc4d00712509318a35b76bc007f94b8d831171af15fa6c1fc7c2",
}


def _ast_holder_sites(raw: bytes, relative: str) -> list[dict[str, Any]]:
    if not relative.endswith(".py"):
        return []
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=relative)
    except (UnicodeError, SyntaxError) as error:
        raise PortReleaseError(f"Python preimage AST differs: {relative}") from error
    return [
        {"line": node.lineno, "column": node.col_offset, "value": node.value}
        for node in sorted(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and (SOURCE_JOB in node.value or SOURCE_NODE in node.value)
            ),
            key=lambda item: (item.lineno, item.col_offset),
        )
    ]


def _transform_payloads(
    payloads: Mapping[str, bytes], modes: Mapping[str, int]
) -> tuple[dict[str, bytes], dict[str, int], list[dict[str, Any]]]:
    if set(TRANSFORMS) - set(payloads):
        fail("one or more registered R14 transform members are absent")
    output = dict(payloads)
    output_modes = dict(modes)
    receipts: list[dict[str, Any]] = []
    for relative in sorted(TRANSFORMS):
        raw = output[relative]
        if hashlib.sha256(raw).hexdigest() != EXPECTED_PREIMAGE_SHA256[relative]:
            fail(f"registered transform preimage differs: {relative}")
        ast_sites = _ast_holder_sites(raw, relative)
        current = raw
        replacement_rows = []
        for old, new, count, label in TRANSFORMS[relative]:
            observed = current.count(old)
            if observed != count:
                fail(f"registered replacement count differs: {relative}/{label}")
            before_sha = hashlib.sha256(current).hexdigest()
            current = current.replace(old, new)
            replacement_rows.append(
                {
                    "label": label,
                    "old_ascii": old.decode("ascii"),
                    "new_ascii": new.decode("ascii"),
                    "count": count,
                    "input_sha256": before_sha,
                    "output_sha256": hashlib.sha256(current).hexdigest(),
                }
            )
        post_relative = PORT_LAUNCHER if relative == SOURCE_LAUNCHER else relative
        expected_post = EXPECTED_POSTIMAGE_SHA256[post_relative]
        if hashlib.sha256(current).hexdigest() != expected_post:
            fail(f"registered transform postimage differs: {post_relative}")
        if relative.endswith(".py"):
            try:
                ast.parse(current.decode("utf-8"), filename=post_relative)
            except (UnicodeError, SyntaxError) as error:
                raise PortReleaseError(
                    f"Python postimage AST differs: {post_relative}"
                ) from error
        del output[relative]
        del output_modes[relative]
        output[post_relative] = current
        output_modes[post_relative] = modes[relative]
        receipts.append(
            {
                "source_member": relative,
                "port_member": post_relative,
                "preimage_sha256": hashlib.sha256(raw).hexdigest(),
                "postimage_sha256": expected_post,
                "preimage_size": len(raw),
                "postimage_size": len(current),
                "ast_old_holder_sites": ast_sites,
                "replacements": replacement_rows,
            }
        )
    if len(output) != SOURCE_FILE_COUNT or set(output) != set(output_modes):
        fail("ported exact18 member closure differs")
    forbidden = {
        SOURCE_JOB.encode("ascii"): "old holder job",
        SOURCE_NODE.encode("ascii"): "old holder node",
        SOURCE_LAUNCHER.encode("ascii"): "old launcher path",
    }
    for token, label in forbidden.items():
        found = [relative for relative, raw in output.items() if token in raw]
        if found:
            fail(f"{label} remains in ported release: {found}")
    return output, output_modes, receipts


def _port_manifest(
    source: Mapping[str, Any], payloads: Mapping[str, bytes], modes: Mapping[str, int]
) -> dict[str, Any]:
    rows = [
        {
            "path": relative,
            "mode": modes[relative],
            "size": len(payloads[relative]),
            "sha256": hashlib.sha256(payloads[relative]).hexdigest(),
        }
        for relative in sorted(payloads)
    ]
    row_by_path = {row["path"]: row for row in rows}
    component_paths = {
        "controller_sha256": "generic_action_data_prep_controller_v1.py",
        "launcher_sha256": PORT_LAUNCHER,
        "generator_sha256": "tools/reserve4_fixed_generation_sp4_v1.py",
        "rank_cache_wrapper_sha256": "scripts/auh_generic_action_data_prep_rank_exec_v1.sh",
    }
    value = dict(source)
    value["files"] = rows
    value["file_count"] = len(rows)
    value["component_pins"] = {
        label: row_by_path[path]["sha256"] for label, path in component_paths.items()
    }
    value["allowed_entrypoints"] = [
        "generic_action_data_prep_controller_v1.py",
        PORT_LAUNCHER,
    ]
    closure = {"member_root": MEMBER_ROOT, "files": rows}
    value["content_closure_sha1"] = hashlib.sha1(
        canonical_json_bytes(closure)
    ).hexdigest()
    value.pop("manifest_digest", None)
    value["manifest_digest"] = object_sha256(value)
    return value


def _build_archive(
    manifest: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for row in manifest["files"]:
            relative = str(row["path"])
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
                fail("ported member path differs")
            raw = payloads[relative]
            member = tarfile.TarInfo(f"{MEMBER_ROOT}/{relative}")
            member.size = len(raw)
            member.mode = int(row["mode"])
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            member.mtime = 0
            member.type = tarfile.REGTYPE
            member.devmajor = 0
            member.devminor = 0
            archive.addfile(member, io.BytesIO(raw))
    # CPython 3.8 serialized zero devmajor/devminor fields for regular USTAR
    # members as octal zero, while 3.9+ serializes those inapplicable fields as
    # NULs even when TarInfo.devmajor/devminor are explicitly zero.  Normalize
    # the two fields and replay the header checksum so the envelope is stable
    # without changing any member path, mode, order, or payload byte.
    normalized = bytearray(output.getvalue())
    offset = 0
    for row in manifest["files"]:
        if offset + tarfile.BLOCKSIZE > len(normalized):
            fail("ported USTAR header closure is truncated")
        header = normalized[offset : offset + tarfile.BLOCKSIZE]
        if header == b"\0" * tarfile.BLOCKSIZE or header[156:157] != tarfile.REGTYPE:
            fail("ported USTAR regular header differs")
        header[329:345] = b"0000000\0" * 2
        header[148:156] = b" " * 8
        checksum = sum(header)
        if checksum > 0o777777:
            fail("ported USTAR checksum is out of range")
        header[148:156] = f"{checksum:06o}\0 ".encode("ascii")
        normalized[offset : offset + tarfile.BLOCKSIZE] = header
        blocks = (int(row["size"]) + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE
        offset += tarfile.BLOCKSIZE * (1 + blocks)
    if any(normalized[offset:]) or len(normalized) % tarfile.RECORDSIZE != 0:
        fail("ported USTAR zero trailer differs")
    return bytes(normalized)


def _write_create_only(path: Path, raw: bytes, mode: int) -> None:
    if not path.is_absolute() or path == Path("/") or path.exists() or path.is_symlink():
        fail("port output must be one fresh absolute file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build(
    source_archive: Path,
    source_manifest: Path,
    output_archive: Path,
    output_manifest: Path,
    output_receipt: Path,
) -> Mapping[str, Any]:
    archive_raw = _stable_plain_bytes(source_archive, "frozen R14 source archive")
    manifest_raw = _stable_plain_bytes(source_manifest, "frozen R14 source manifest")
    source = _load_source_manifest(manifest_raw)
    payloads, modes = _load_source_archive(archive_raw, source)
    port_payloads, port_modes, transforms = _transform_payloads(payloads, modes)
    manifest = _port_manifest(source, port_payloads, port_modes)
    port_archive_raw = _build_archive(manifest, port_payloads)
    if _build_archive(manifest, port_payloads) != port_archive_raw:
        fail("ported archive rebuild is not byte-identical")
    port_manifest_raw = canonical_json_bytes(manifest) + b"\n"
    port_archive_sha = hashlib.sha256(port_archive_raw).hexdigest()
    port_manifest_sha = hashlib.sha256(port_manifest_raw).hexdigest()
    unchanged = sorted(set(payloads) - set(TRANSFORMS))
    if any(payloads[path] != port_payloads[path] for path in unchanged):
        fail("one of fourteen unchanged R14 members drifted")
    unsigned_receipt = {
        "schema_version": PORT_RECEIPT_SCHEMA,
        "source": {
            "archive_sha256": SOURCE_ARCHIVE_SHA256,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "manifest_digest": SOURCE_MANIFEST_DIGEST,
            "content_closure_sha1": SOURCE_CONTENT_CLOSURE_SHA1,
            "file_count": SOURCE_FILE_COUNT,
        },
        "port": {
            "holder_job": 136309,
            "holder_node": PORT_NODE,
            "confirmation": PORT_CONFIRMATION,
            "plan_id": PORT_PLAN_ID,
            "launcher_member": PORT_LAUNCHER,
            "archive_sha256": port_archive_sha,
            "manifest_sha256": port_manifest_sha,
            "manifest_digest": manifest["manifest_digest"],
            "content_closure_sha1": manifest["content_closure_sha1"],
            "file_count": len(port_payloads),
        },
        "transforms": transforms,
        "unchanged_member_count": len(unchanged),
        "unchanged_members": [
            {
                "path": path,
                "sha256": hashlib.sha256(port_payloads[path]).hexdigest(),
            }
            for path in unchanged
        ],
        "old_holder_job_token_count": sum(
            raw.count(SOURCE_JOB.encode("ascii")) for raw in port_payloads.values()
        ),
        "old_holder_node_token_count": sum(
            raw.count(SOURCE_NODE.encode("ascii")) for raw in port_payloads.values()
        ),
        "old_launcher_path_count": sum(
            raw.count(SOURCE_LAUNCHER.encode("ascii")) for raw in port_payloads.values()
        ),
        "analysis_split": "fit",
        "candidate_count": 40,
        "seed_cell_count": 4,
        "world4_formal_invocation_count": 40,
        "world4_shard_runner_count": 4,
        "all_model_invocations_strictly_serial": True,
        "generated_media_role": "pending-external-review-authoring-media-only",
        "confirmation_generation_authorized": False,
        "independent_full81_blind_review_present": False,
        "phi_v1_extraction_authorized": False,
        "p_or_o_manifest_materialization_authorized": False,
        "optimizer_authorized": False,
        "training_authorized": False,
        "remote_deploy_or_launch_performed": False,
    }
    receipt = {
        **unsigned_receipt,
        "receipt_digest": object_sha256(unsigned_receipt),
    }
    receipt_raw = canonical_json_bytes(receipt) + b"\n"
    _write_create_only(output_archive, port_archive_raw, 0o444)
    _write_create_only(output_manifest, port_manifest_raw, 0o444)
    _write_create_only(output_receipt, receipt_raw, 0o444)
    return receipt


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-archive", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-receipt", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    receipt = build(
        Path(args.source_archive),
        Path(args.source_manifest),
        Path(args.output_archive),
        Path(args.output_manifest),
        Path(args.output_receipt),
    )
    print(canonical_json_bytes(receipt).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
