#!/usr/bin/env python3
"""Build the deterministic exact-member CSVC checkpoint-review release.

The release starts with the sealed 15-member Stage-B runtime, adds the exact
14 files introduced by the checkpoint-review change, and recursively closes
the local Python import graph.  Pinned Bernini/VeOmni trees remain external
authorities.  In particular, ``tools.materialize_vae`` is intentionally
provided by the pinned Bernini tree after ``activate_source_trees``; it is not
silently rebound to the similarly named local helper.
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


SCHEMA_VERSION = (
    "bernini-clean-source-visual-context-checkpoint-review-release-v1"
)
RELEASE_GENERATION = "r1"
ARCHIVE_FORMAT = "ustar-owner0-mtime0-mode0444-v1"
MEMBER_ROOT = "methods/bernini_action_editing"
IMPORT_SCANNER_VERSION = "local-ast-import-closure-v1"

# This tuple must remain byte-for-byte aligned with
# build_clean_source_visual_context_stage_b_release_v1.RELEASE_FILES.
STAGE_B_RELEASE_FILES = (
    "clean_source_visual_context_adapter_v1.py",
    "clean_source_visual_context_training_v1.py",
    "clean_source_visual_context_stage_b_contract_v1.py",
    "train_clean_source_visual_context_stage_b_v1.py",
    "clean_source_visual_context_pair_controller_v1.py",
    "source_self_runtime.py",
    "train_lora.py",
    "inference_sigma_strata.py",
    "scripts/auh_preservation_rank_cache_exec_v1.sh",
    "scripts/auh_train_clean_source_visual_context_stage_b_holder_v1.sh",
    "scripts/auh_train_clean_source_visual_context_main_holder_v1.sh",
    "scripts/auh_train_clean_source_visual_context_noised_holder_v1.sh",
    "scripts/auh_preflight_clean_source_visual_context_main_holder_v1.sh",
    "scripts/auh_preflight_clean_source_visual_context_noised_holder_v1.sh",
    "scripts/auh_materialize_clean_source_visual_context_source_only_v3_holder_v1.sh",
)

CHECKPOINT_REVIEW_FILES = (
    "clean_source_visual_context_checkpoint_decode_runtime_v1.py",
    "clean_source_visual_context_checkpoint_review_contract_v1.py",
    "infer_clean_source_visual_context_checkpoint_review_v1.py",
    "scripts/auh_decode_clean_source_visual_context_checkpoint_review_holder_v1.sh",
    "scripts/auh_train_then_review_clean_source_visual_context_holder_v1.sh",
    "tests/test_auh_clean_source_visual_context_checkpoint_review_chain_v1.py",
    "tests/test_build_clean_source_visual_context_checkpoint_review_html_v1.py",
    "tests/test_clean_source_visual_context_checkpoint_decode_runtime_v1.py",
    "tests/test_infer_clean_source_visual_context_checkpoint_review_v1.py",
    "tests/test_materialize_clean_source_visual_context_checkpoint_review_manifest_v1.py",
    "tests/test_materialize_clean_source_visual_context_review_authoring_v2.py",
    "tools/build_clean_source_visual_context_checkpoint_review_html_v1.py",
    "tools/materialize_clean_source_visual_context_checkpoint_review_manifest_v1.py",
    "tools/materialize_clean_source_visual_context_review_authoring_v2.py",
)

# These are not arbitrary support files: every member is reached recursively
# from the two seed tuples above by ``discover_import_closure``.
RECURSIVE_IMPORT_FILES = (
    "infer_lora.py",
    "infer_native_identity_generation_canary.py",
    "infer_native_v_axis_exact81_probe_v1.py",
    "infer_orderless_source_frame_set_noise_canary.py",
    "infer_source_kv_carrier_oracle.py",
    "infer_source_value_residual_oracle.py",
    "native_i_axis_guidance.py",
    "native_v_axis_guidance_v1.py",
    "orderless_source_frame_set_noise.py",
    "source_kv_replay.py",
    "source_kv_route_batches.py",
    "source_value_residual.py",
    "tri_branch_unipc.py",
)

RELEASE_FILES = (
    *STAGE_B_RELEASE_FILES,
    *CHECKPOINT_REVIEW_FILES,
    *RECURSIVE_IMPORT_FILES,
)

VENDOR_IMPORT_EXCEPTIONS = (
    "infer_lora.py",
    "infer_native_identity_generation_canary.py",
    "infer_orderless_source_frame_set_noise_canary.py",
    "infer_source_kv_carrier_oracle.py",
    "infer_source_value_residual_oracle.py",
)

SHELL_DEPENDENCY_EDGES = (
    (
        "scripts/auh_decode_clean_source_visual_context_checkpoint_review_holder_v1.sh",
        "infer_clean_source_visual_context_checkpoint_review_v1.py",
    ),
    (
        "scripts/auh_decode_clean_source_visual_context_checkpoint_review_holder_v1.sh",
        "clean_source_visual_context_checkpoint_decode_runtime_v1.py",
    ),
    (
        "scripts/auh_decode_clean_source_visual_context_checkpoint_review_holder_v1.sh",
        "clean_source_visual_context_checkpoint_review_contract_v1.py",
    ),
    (
        "scripts/auh_decode_clean_source_visual_context_checkpoint_review_holder_v1.sh",
        "tools/build_clean_source_visual_context_checkpoint_review_html_v1.py",
    ),
    (
        "scripts/auh_train_then_review_clean_source_visual_context_holder_v1.sh",
        "scripts/auh_train_clean_source_visual_context_stage_b_holder_v1.sh",
    ),
    (
        "scripts/auh_train_then_review_clean_source_visual_context_holder_v1.sh",
        "scripts/auh_decode_clean_source_visual_context_checkpoint_review_holder_v1.sh",
    ),
)


class CheckpointReviewReleaseError(RuntimeError):
    """Raised before an incomplete or mutable runtime release is published."""


def fail(message: str) -> NoReturn:
    raise CheckpointReviewReleaseError(message)


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
        raise CheckpointReviewReleaseError(
            "release value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, *, label: str, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        fail(f"{label} must be lowercase SHA-{length * 4}")
    return value


def _canonical_method_root(value: str | Path) -> Path:
    root = Path(value).expanduser()
    if not root.is_absolute() or root.is_symlink():
        fail("method root must be an absolute non-symlink directory")
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise CheckpointReviewReleaseError("method root is unavailable") from error
    if resolved != root or not root.is_dir() or root.is_symlink():
        fail("method root must be one canonical plain directory")
    return root


def _safe_relative(value: str) -> str:
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or ".." in pure.parts
        or pure.as_posix() != value
    ):
        fail("release member path differs")
    return value


def _stable_plain_bytes(path: Path, *, root: Path) -> bytes:
    if path.is_symlink():
        fail(f"release input is a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
        before = path.lstat()
    except OSError as error:
        raise CheckpointReviewReleaseError(
            f"release input is unavailable: {path}"
        ) from error
    if (
        resolved != path
        or root not in path.parents
        or not stat.S_ISREG(before.st_mode)
    ):
        fail(f"release input is not a canonical plain file: {path}")
    raw = path.read_bytes()
    after = path.lstat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(raw) != before.st_size
        or not raw
    ):
        fail(f"release input changed while reading: {path}")
    return raw


def _qualified_local_path(
    *, root: Path, module: str, aliases: Iterable[str]
) -> tuple[str, ...]:
    prefix = "methods.bernini_action_editing"
    if module == prefix:
        candidates = [root / f"{alias}.py" for alias in aliases]
    elif module.startswith(prefix + "."):
        tail = module[len(prefix) + 1 :]
        direct = root / (tail.replace(".", "/") + ".py")
        if direct.is_file() or direct.is_symlink():
            candidates = [direct]
        else:
            directory = root / tail.replace(".", "/")
            candidates = [directory / f"{alias}.py" for alias in aliases]
    else:
        return ()
    result = []
    for candidate in candidates:
        if candidate.is_file() or candidate.is_symlink():
            result.append(candidate.relative_to(root).as_posix())
    return tuple(result)


def _local_imports(
    *, root: Path, relative: str, raw: bytes
) -> tuple[set[str], bool]:
    try:
        text = raw.decode("utf-8")
        tree = ast.parse(text, filename=relative)
    except (UnicodeError, SyntaxError) as error:
        raise CheckpointReviewReleaseError(
            f"cannot parse Python release member {relative}"
        ) from error
    found: set[str] = set()
    vendor_materialize_vae = False

    def add_top_level(module: str) -> None:
        if not module or "." in module:
            return
        candidates = [
            candidate
            for candidate in (
                root / f"{module}.py",
                root / "tools" / f"{module}.py",
                root / "tests" / f"{module}.py",
            )
            if candidate.is_file() or candidate.is_symlink()
        ]
        if len(candidates) > 1:
            fail(f"ambiguous local import {module!r} in {relative}")
        if candidates:
            found.add(candidates[0].relative_to(root).as_posix())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                add_top_level(name)
                found.update(
                    _qualified_local_path(root=root, module=name, aliases=())
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            aliases = tuple(alias.name for alias in node.names)
            if module == "tools" and "materialize_vae" in aliases:
                vendor_materialize_vae = True
                continue
            add_top_level(module)
            found.update(
                _qualified_local_path(root=root, module=module, aliases=aliases)
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "__import__"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            add_top_level(node.args[0].value)
    return found, vendor_materialize_vae


def discover_import_closure(
    root: Path, payloads: Mapping[str, bytes]
) -> tuple[tuple[str, ...], tuple[Mapping[str, str], ...], tuple[str, ...]]:
    """Return recursively reached members, edges and pinned-vendor exceptions."""

    selected = set((*STAGE_B_RELEASE_FILES, *CHECKPOINT_REVIEW_FILES))
    queue = sorted(relative for relative in selected if relative.endswith(".py"))
    edges: set[tuple[str, str]] = set()
    vendor_importers: set[str] = set()
    while queue:
        importer = queue.pop(0)
        raw = payloads.get(importer)
        if raw is None:
            raw = _stable_plain_bytes(root / importer, root=root)
        imports, has_vendor_import = _local_imports(
            root=root, relative=importer, raw=raw
        )
        if has_vendor_import:
            vendor_importers.add(importer)
        for imported in sorted(imports):
            edges.add((importer, imported))
            if imported not in selected:
                selected.add(imported)
                queue.append(imported)
                queue.sort()
    discovered = tuple(
        [relative for relative in RELEASE_FILES if relative in selected]
        + sorted(selected - set(RELEASE_FILES))
    )
    edge_rows = tuple(
        {"importer": importer, "imported": imported}
        for importer, imported in sorted(edges)
    )
    return discovered, edge_rows, tuple(sorted(vendor_importers))


def _validate_shell_edges(payloads: Mapping[str, bytes]) -> None:
    for importer, imported in SHELL_DEPENDENCY_EDGES:
        try:
            source = payloads[importer].decode("utf-8")
        except (KeyError, UnicodeError) as error:
            raise CheckpointReviewReleaseError(
                f"cannot inspect shell dependency source: {importer}"
            ) from error
        if Path(imported).name not in source:
            fail(f"shell dependency edge disappeared: {importer} -> {imported}")


def build_manifest(method_root: Path) -> tuple[Mapping[str, Any], Mapping[str, bytes]]:
    root = _canonical_method_root(method_root)
    if len(RELEASE_FILES) != len(set(RELEASE_FILES)):
        fail("declared release member closure contains duplicates")
    payloads: dict[str, bytes] = {}
    for relative in RELEASE_FILES:
        _safe_relative(relative)
        payloads[relative] = _stable_plain_bytes(root / relative, root=root)
    discovered, import_edges, vendor_importers = discover_import_closure(
        root, payloads
    )
    if discovered != RELEASE_FILES:
        missing = sorted(set(discovered) - set(RELEASE_FILES))
        unreachable = sorted(set(RELEASE_FILES) - set(discovered))
        fail(
            "recursive local import closure differs; "
            f"missing={missing}, unreachable={unreachable}"
        )
    if vendor_importers != tuple(sorted(VENDOR_IMPORT_EXCEPTIONS)):
        fail("pinned Bernini tools.materialize_vae import boundary differs")
    _validate_shell_edges(payloads)
    rows = [
        {
            "path": relative,
            "sha256": hashlib.sha256(payloads[relative]).hexdigest(),
            "size": len(payloads[relative]),
            "mode": "0444",
        }
        for relative in RELEASE_FILES
    ]
    by_path = {row["path"]: row for row in rows}
    stage_b_rows = [by_path[path] for path in STAGE_B_RELEASE_FILES]
    review_rows = [by_path[path] for path in CHECKPOINT_REVIEW_FILES]
    recursive_rows = [by_path[path] for path in RECURSIVE_IMPORT_FILES]
    closure = {"member_root": MEMBER_ROOT, "files": rows}
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "release_generation": RELEASE_GENERATION,
        "archive_format": ARCHIVE_FORMAT,
        "member_root": MEMBER_ROOT,
        "file_count": len(rows),
        "files": rows,
        "stage_b_base": {
            "file_count": len(stage_b_rows),
            "members_digest": object_sha256(stage_b_rows),
        },
        "checkpoint_review_seed": {
            "file_count": len(review_rows),
            "members_digest": object_sha256(review_rows),
        },
        "recursive_import_closure": {
            "scanner_version": IMPORT_SCANNER_VERSION,
            "file_count": len(recursive_rows),
            "members_digest": object_sha256(recursive_rows),
            "edges": list(import_edges),
            "complete": True,
        },
        "external_runtime_imports": [
            {
                "module": "tools.materialize_vae",
                "provider": "pinned-bernini-root-prepended-by-activate_source_trees",
                "importers": list(vendor_importers),
                "packaged_as_local_member": False,
            }
        ],
        "shell_dependency_edges": [
            {"importer": importer, "imported": imported}
            for importer, imported in SHELL_DEPENDENCY_EDGES
        ],
        "revision_kind": "content-closure-sha1",
        "content_closure_sha1": hashlib.sha1(
            canonical_json_bytes(closure)
        ).hexdigest(),
        "git_commit_claimed": False,
        "exact_member_closure": True,
    }
    return {**unsigned, "manifest_digest": object_sha256(unsigned)}, payloads


def build_archive(manifest: Mapping[str, Any], payloads: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for row in manifest["files"]:
            relative = str(row["path"])
            raw = payloads[relative]
            member = tarfile.TarInfo(f"{MEMBER_ROOT}/{relative}")
            member.size = len(raw)
            member.mode = 0o444
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            member.mtime = 0
            member.type = tarfile.REGTYPE
            archive.addfile(member, io.BytesIO(raw))
    return output.getvalue()


def _validate_manifest(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "release_generation",
        "archive_format",
        "member_root",
        "file_count",
        "files",
        "stage_b_base",
        "checkpoint_review_seed",
        "recursive_import_closure",
        "external_runtime_imports",
        "shell_dependency_edges",
        "revision_kind",
        "content_closure_sha1",
        "git_commit_claimed",
        "exact_member_closure",
        "manifest_digest",
    }
    unsigned = dict(value)
    digest = unsigned.pop("manifest_digest", None)
    if (
        set(value) != expected
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("release_generation") != RELEASE_GENERATION
        or value.get("archive_format") != ARCHIVE_FORMAT
        or value.get("member_root") != MEMBER_ROOT
        or value.get("revision_kind") != "content-closure-sha1"
        or value.get("git_commit_claimed") is not False
        or value.get("exact_member_closure") is not True
        or digest != object_sha256(unsigned)
    ):
        fail("release manifest schema or embedded digest differs")
    rows = value.get("files")
    if (
        not isinstance(rows, list)
        or value.get("file_count") != len(RELEASE_FILES)
        or len(rows) != len(RELEASE_FILES)
        or tuple(row.get("path") for row in rows if isinstance(row, Mapping))
        != RELEASE_FILES
    ):
        fail("release manifest exact member closure differs")
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256", "size", "mode"}
            or _safe_relative(str(row.get("path"))) != row.get("path")
            or type(row.get("size")) is not int
            or row["size"] <= 0
            or row.get("mode") != "0444"
        ):
            fail("release manifest member metadata differs")
        _sha(row.get("sha256"), label=f"release member {row.get('path')}")
    by_path = {row["path"]: row for row in rows}
    expected_stage_b_rows = [by_path[path] for path in STAGE_B_RELEASE_FILES]
    expected_review_rows = [by_path[path] for path in CHECKPOINT_REVIEW_FILES]
    expected_recursive_rows = [by_path[path] for path in RECURSIVE_IMPORT_FILES]
    if value.get("stage_b_base") != {
        "file_count": len(STAGE_B_RELEASE_FILES),
        "members_digest": object_sha256(expected_stage_b_rows),
    }:
        fail("release Stage-B base closure differs")
    if value.get("checkpoint_review_seed") != {
        "file_count": len(CHECKPOINT_REVIEW_FILES),
        "members_digest": object_sha256(expected_review_rows),
    }:
        fail("release checkpoint-review seed closure differs")
    recursive = value.get("recursive_import_closure")
    if (
        not isinstance(recursive, Mapping)
        or set(recursive)
        != {
            "scanner_version",
            "file_count",
            "members_digest",
            "edges",
            "complete",
        }
        or recursive.get("scanner_version") != IMPORT_SCANNER_VERSION
        or recursive.get("file_count") != len(RECURSIVE_IMPORT_FILES)
        or recursive.get("members_digest")
        != object_sha256(expected_recursive_rows)
        or recursive.get("complete") is not True
        or not isinstance(recursive.get("edges"), list)
    ):
        fail("release recursive import closure metadata differs")
    edge_pairs: list[tuple[str, str]] = []
    for edge in recursive["edges"]:
        if (
            not isinstance(edge, Mapping)
            or set(edge) != {"importer", "imported"}
            or edge.get("importer") not in RELEASE_FILES
            or edge.get("imported") not in RELEASE_FILES
        ):
            fail("release recursive import edge differs")
        edge_pairs.append((edge["importer"], edge["imported"]))
    if edge_pairs != sorted(set(edge_pairs)):
        fail("release recursive import edges are not unique/sorted")
    reached = set((*STAGE_B_RELEASE_FILES, *CHECKPOINT_REVIEW_FILES))
    changed = True
    while changed:
        changed = False
        for importer, imported in edge_pairs:
            if importer in reached and imported not in reached:
                reached.add(imported)
                changed = True
    if reached != set(RELEASE_FILES):
        fail("release recursive import graph does not reach exact closure")
    expected_external = [
        {
            "module": "tools.materialize_vae",
            "provider": "pinned-bernini-root-prepended-by-activate_source_trees",
            "importers": list(sorted(VENDOR_IMPORT_EXCEPTIONS)),
            "packaged_as_local_member": False,
        }
    ]
    if value.get("external_runtime_imports") != expected_external:
        fail("release pinned external runtime import boundary differs")
    expected_shell_edges = [
        {"importer": importer, "imported": imported}
        for importer, imported in SHELL_DEPENDENCY_EDGES
    ]
    if value.get("shell_dependency_edges") != expected_shell_edges:
        fail("release shell dependency closure differs")
    closure = {"member_root": MEMBER_ROOT, "files": rows}
    if value.get("content_closure_sha1") != hashlib.sha1(
        canonical_json_bytes(closure)
    ).hexdigest():
        fail("release content closure revision differs")


def verify_archive(raw: bytes, manifest: Mapping[str, Any]) -> None:
    _validate_manifest(manifest)
    expected_names = [
        f"{MEMBER_ROOT}/{row['path']}" for row in manifest["files"]
    ]
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            members = archive.getmembers()
            if [member.name for member in members] != expected_names:
                fail("release archive exact member closure differs")
            for member, row in zip(members, manifest["files"]):
                payload = archive.extractfile(member)
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != 0
                    or stat.S_IMODE(member.mode) != 0o444
                    or member.size != row["size"]
                    or member.pax_headers
                    or payload is None
                    or hashlib.sha256(payload.read()).hexdigest()
                    != row["sha256"]
                ):
                    fail(f"release archive member differs: {member.name}")
    except (OSError, tarfile.TarError) as error:
        raise CheckpointReviewReleaseError(
            f"cannot verify release archive: {error}"
        ) from error


def _plain_release_file(
    value: str | Path, *, expected_sha256: str, label: str
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CheckpointReviewReleaseError(f"{label} is unavailable") from error
    if resolved != path or not path.is_file() or path.is_symlink():
        fail(f"{label} must be one canonical plain file")
    if file_sha256(path) != _sha(expected_sha256, label=f"{label} expected SHA"):
        fail(f"{label} SHA-256 differs")
    return path


def load_release_bundle(
    *,
    archive_path: str | Path,
    archive_sha256: str,
    manifest_path: str | Path,
    manifest_sha256: str,
) -> Mapping[str, Any]:
    archive = _plain_release_file(
        archive_path, expected_sha256=archive_sha256, label="review release archive"
    )
    manifest_file = _plain_release_file(
        manifest_path,
        expected_sha256=manifest_sha256,
        label="review release manifest",
    )
    raw_manifest = manifest_file.read_bytes()
    try:
        value = json.loads(raw_manifest.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointReviewReleaseError("cannot read release manifest") from error
    if (
        not isinstance(value, Mapping)
        or raw_manifest != canonical_json_bytes(value) + b"\n"
    ):
        fail("release manifest is not canonical ASCII JSON")
    verify_archive(archive.read_bytes(), value)
    return value


def verify_executed_root(
    *,
    method_root: str | Path,
    archive_path: str | Path,
    archive_sha256: str,
    manifest_path: str | Path,
    manifest_sha256: str,
    method_revision: str,
) -> Mapping[str, Any]:
    manifest = load_release_bundle(
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
    root = _canonical_method_root(method_root)
    if tuple(root.parts[-2:]) != ("methods", "bernini_action_editing"):
        fail("executed method root is not the canonical extracted member root")
    observed_files: set[str] = set()
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            fail("executed method root contains a symlink or special entry")
        if stat.S_ISREG(mode):
            observed_files.add(path.relative_to(root).as_posix())
    if observed_files != set(RELEASE_FILES):
        fail("executed method root exact file set differs")
    for row in manifest["files"]:
        path = root / row["path"]
        mode = path.lstat().st_mode
        if (
            not stat.S_ISREG(mode)
            or stat.S_IMODE(mode) != 0o444
            or path.stat().st_size != row["size"]
            or file_sha256(path) != row["sha256"]
        ):
            fail(f"executed method member differs: {row['path']}")
    rebuilt, _ = build_manifest(root)
    if rebuilt != manifest:
        fail("executed method root import/content closure differs")
    revision = _sha(method_revision, label="method revision", length=40)
    if revision != manifest["content_closure_sha1"]:
        fail("executed method revision differs")
    entries = [
        {"path": row["path"], "sha256": row["sha256"]}
        for row in manifest["files"]
    ]
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "method_root": str(root),
        "archive": str(Path(archive_path)),
        "archive_sha256": archive_sha256,
        "manifest": str(Path(manifest_path)),
        "manifest_sha256": manifest_sha256,
        "manifest_digest": manifest["manifest_digest"],
        "method_revision": revision,
        "exact_member_count": len(entries),
        "archive_members_verified": True,
        "executed_tree_exact_member_closure": True,
        "recursive_import_closure_verified": True,
        "executed_entries_digest": object_sha256(entries),
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


def _fresh_output(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.exists() or path.is_symlink():
        fail(f"{label} must be a fresh absolute file")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise CheckpointReviewReleaseError(f"{label} parent is unavailable") from error
    if path.parent != parent or not parent.is_dir() or parent.is_symlink():
        fail(f"{label} parent must be one canonical plain directory")
    return path


def _write_create_only(path: Path, raw: bytes) -> None:
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build(
    method_root: Path, archive_path: Path, manifest_path: Path
) -> Mapping[str, Any]:
    archive = _fresh_output(archive_path, label="review release archive")
    manifest_file = _fresh_output(manifest_path, label="review release manifest")
    if archive == manifest_file:
        fail("release archive and manifest paths must differ")
    manifest, payloads = build_manifest(method_root)
    archive_raw = build_archive(manifest, payloads)
    verify_archive(archive_raw, manifest)
    if build_archive(manifest, payloads) != archive_raw:
        fail("release archive rebuild is not byte-identical")
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    _write_create_only(archive, archive_raw)
    _write_create_only(manifest_file, manifest_raw)
    archive_sha = hashlib.sha256(archive_raw).hexdigest()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    load_release_bundle(
        archive_path=archive,
        archive_sha256=archive_sha,
        manifest_path=manifest_file,
        manifest_sha256=manifest_sha,
    )
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "archive": str(archive),
        "archive_sha256": archive_sha,
        "manifest": str(manifest_file),
        "manifest_sha256": manifest_sha,
        "manifest_digest": manifest["manifest_digest"],
        "content_closure_sha1": manifest["content_closure_sha1"],
        "file_count": len(RELEASE_FILES),
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    result = build(
        Path(args.method_root), Path(args.archive), Path(args.manifest)
    )
    print(canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
