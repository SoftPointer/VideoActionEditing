#!/usr/bin/env python3
"""Build and verify the deterministic v16r6 A/B prefix-32 source release.

The release extends the authenticated v16r5-r3 source archive.  It replaces
only the small v1 LoRA-target indirection needed by variant B and adds the two
single-variable wrappers, their shared non-exact644 debug mechanics, and the
preflight tests.  Archive members are sorted USTAR files owned by uid/gid 0
with mtime 0, so identical inputs produce byte-identical releases.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, Mapping, Optional, Sequence


BASE_ARCHIVE_SHA256 = (
    "b87b51027c4819e63292e2beadbee728cd908883017b05233e8a58a1448adfde"
)
BASE_MANIFEST_SHA256 = (
    "0a46c8cf7c2b7c06d7aa4ca2d53977c856947f1abd8f94aaaaa434b52e08b376"
)
BASE_CLOSURE_SHA256 = (
    "1aaea36f41cc6591ca38920f0d70c8be09744f0be7a2de14d499a538eda4d781"
)
BASE_SCHEMA = "bernini-v16r5-full644-source-release-v1"
SOURCE_SCHEMA = "bernini-v16r6-ab-debug32-source-release-v1"
RELEASE_SCHEMA = "bernini-v16r6-ab-debug32-release-v1"
ARCHIVE_FORMAT = "ustar-sorted-owner0-mtime0-exact-mode-v1"
BASE_FILE_COUNT = 34
SOURCE_FILE_COUNT = 39

ORIGINAL_V1_SHA256 = (
    "fd8c5b6d8d7fb94de9cb8d2811a953b116643216162b5cfe758f50ef0b55626c"
)
V16R5_TRAINER_SHA256 = (
    "dcd7998870bf1e3b9a049a2e27e500537a4b19b895952b76c8d558ea314abeb3"
)
HELDOUT8_SHA256 = (
    "c05c4e5b5bf85de882bde32c71a984d736247733e586ed91d40026b12aaaf701"
)
FULL644_MANIFEST_SHA256 = (
    "61da995eb680b9fba7ab3b7d3b6041c7b51c7e95253c74e607ddab6fdd6a61aa"
)

V1_MEMBER = "methods/bernini_action_editing/train_online_anchor_attention_v1.py"
V16R5_MEMBER = (
    "methods/bernini_action_editing/"
    "train_online_anchor_attention_full644_dynamic_static_v16r5.py"
)
HELDOUT8_MEMBER = (
    "methods/action_editing_baselines/manifests/"
    "goku_legacy_heldout8_inputs.jsonl"
)
SOURCE_MEMBERS = {
    V1_MEMBER: (
        "methods/bernini_action_editing/train_online_anchor_attention_v1.py",
        "5e152ebf4ccdb4211c96c7c66b1891b0b38750bbc30f62d61a9a93261a73c178",
    ),
    "methods/bernini_action_editing/train_online_anchor_attention_v16r6_debug_common.py": (
        "methods/bernini_action_editing/train_online_anchor_attention_v16r6_debug_common.py",
        "50da3b02bd138ccc08d4431636297ae029fd95f69455081b279046734d17e9ae",
    ),
    "methods/bernini_action_editing/train_online_anchor_attention_full644_dynamic_static_v16r6a_lr1e7_32.py": (
        "methods/bernini_action_editing/train_online_anchor_attention_full644_dynamic_static_v16r6a_lr1e7_32.py",
        "3a163f7d63c6731a1d2ce06e5373896636c4017268b8fe4b5625e4b8a2829b4c",
    ),
    "methods/bernini_action_editing/train_online_anchor_attention_full644_dynamic_static_v16r6b_route_qk32.py": (
        "methods/bernini_action_editing/train_online_anchor_attention_full644_dynamic_static_v16r6b_route_qk32.py",
        "2eb229fce8cde3810abec39ccfb5d5dcc2ed9b82752f1d424a1e79e7ffae9fba",
    ),
    "methods/bernini_action_editing/tests/test_train_online_anchor_attention_v16r6_ab.py": (
        "methods/bernini_action_editing/tests/test_train_online_anchor_attention_v16r6_ab.py",
        "220bfc9b2fb3f47907c75f4ed57ca4c9abb880be3664556bd48369d7625ef373",
    ),
    "methods/bernini_action_editing/tests/test_train_online_anchor_attention_full644_dynamic_static_v16r5.py": (
        "methods/bernini_action_editing/tests/test_train_online_anchor_attention_full644_dynamic_static_v16r5.py",
        "be2c2c5d3fee68d6c66b2de44da412ad0e12da43ac7c6c323f6226785a078ef8",
    ),
}
CONTROL_MEMBERS = (
    "methods/bernini_action_editing/scripts/auh_run_v16r6_ab_debug32.sh",
    "methods/bernini_action_editing/scripts/auh_launch_v16r6_ab_debug32.sh",
)


class ReleaseError(RuntimeError):
    pass


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha1(raw: bytes) -> str:
    return hashlib.sha1(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ReleaseError(f"{label} must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ReleaseError(f"{label} is not a regular file")
    return resolved


def parse_object(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseError(f"{label} cannot be parsed") from error
    if not isinstance(value, Mapping):
        raise ReleaseError(f"{label} is not an object")
    return value


def _verify_rows(
    archive_raw: bytes,
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, bytes]:
    paths = [str(row.get("path", "")) for row in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ReleaseError(f"{label} path closure differs")
    data: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:") as handle:
            members = handle.getmembers()
            if [member.name for member in members] != paths:
                raise ReleaseError(f"{label} archive member order differs")
            for member, row in zip(members, rows):
                pure = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or pure.is_absolute()
                    or ".." in pure.parts
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != 0
                    or member.mode != int(row.get("mode", -1))
                    or member.size != int(row.get("size", -1))
                ):
                    raise ReleaseError(f"{label} metadata differs: {member.name}")
                stream = handle.extractfile(member)
                if stream is None:
                    raise ReleaseError(f"{label} member is unreadable: {member.name}")
                raw = stream.read()
                if sha256(raw) != row.get("sha256"):
                    raise ReleaseError(f"{label} member SHA differs: {member.name}")
                data[member.name] = raw
    except (OSError, tarfile.TarError) as error:
        raise ReleaseError(f"{label} archive cannot be parsed") from error
    return data


def verify_base(
    archive: Path, manifest_path: Path
) -> tuple[Mapping[str, Any], dict[str, bytes]]:
    archive_raw = regular_file(archive, "base archive").read_bytes()
    manifest_raw = regular_file(manifest_path, "base manifest").read_bytes()
    if sha256(archive_raw) != BASE_ARCHIVE_SHA256:
        raise ReleaseError("authenticated v16r5-r3 archive SHA differs")
    if sha256(manifest_raw) != BASE_MANIFEST_SHA256:
        raise ReleaseError("authenticated v16r5-r3 manifest SHA differs")
    manifest = parse_object(manifest_raw, "base manifest")
    rows = manifest.get("files")
    if (
        manifest.get("schema_version") != BASE_SCHEMA
        or manifest.get("archive_format") != ARCHIVE_FORMAT
        or manifest.get("content_closure_sha256") != BASE_CLOSURE_SHA256
        or manifest.get("file_count") != BASE_FILE_COUNT
        or not isinstance(rows, list)
        or len(rows) != BASE_FILE_COUNT
    ):
        raise ReleaseError("authenticated v16r5-r3 manifest envelope differs")
    closure = {"schema_version": BASE_SCHEMA, "files": rows}
    if sha256(canonical(closure)) != BASE_CLOSURE_SHA256:
        raise ReleaseError("authenticated v16r5-r3 content closure differs")
    data = _verify_rows(archive_raw, rows, label="base")
    if sha256(data.get(V1_MEMBER, b"")) != ORIGINAL_V1_SHA256:
        raise ReleaseError("base v1 trainer SHA differs")
    if sha256(data.get(V16R5_MEMBER, b"")) != V16R5_TRAINER_SHA256:
        raise ReleaseError("base v16r5 trainer SHA differs")
    if sha256(data.get(HELDOUT8_MEMBER, b"")) != HELDOUT8_SHA256:
        raise ReleaseError("base Heldout8 manifest SHA differs")
    return manifest, data


def build_source(
    workspace_root: Path,
    base_archive: Path,
    base_manifest: Path,
) -> tuple[bytes, bytes, Mapping[str, Any]]:
    old, data = verify_base(base_archive, base_manifest)
    for member, (relative, expected_sha) in SOURCE_MEMBERS.items():
        raw = regular_file(workspace_root / relative, member).read_bytes()
        if sha256(raw) != expected_sha:
            raise ReleaseError(f"pinned source SHA differs: {member}")
        data[member] = raw

    if len(data) != SOURCE_FILE_COUNT:
        raise ReleaseError("v16r6 source file count differs")
    old_rows = {
        str(row["path"]): dict(row)
        for row in old["files"]
        if isinstance(row, Mapping)
    }
    for member, raw in data.items():
        old_rows[member] = {
            "mode": 0o444,
            "path": member,
            "sha256": sha256(raw),
            "size": len(raw),
        }
    rows = [old_rows[path] for path in sorted(data)]
    closure_raw = canonical({"schema_version": SOURCE_SCHEMA, "files": rows})
    document = {
        "archive_format": ARCHIVE_FORMAT,
        "base_content_closure_sha256": BASE_CLOSURE_SHA256,
        "base_source_archive_sha256": BASE_ARCHIVE_SHA256,
        "base_source_manifest_sha256": BASE_MANIFEST_SHA256,
        "content_closure_sha1": sha1(closure_raw),
        "content_closure_sha256": sha256(closure_raw),
        "file_count": SOURCE_FILE_COUNT,
        "files": rows,
        "member_root": ".",
        "schema_version": SOURCE_SCHEMA,
    }
    manifest_raw = canonical(document)
    sink = io.BytesIO()
    with tarfile.open(fileobj=sink, mode="w:", format=tarfile.USTAR_FORMAT) as handle:
        for row in rows:
            raw = data[str(row["path"])]
            info = tarfile.TarInfo(str(row["path"]))
            info.size = len(raw)
            info.mode = 0o444
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.type = tarfile.REGTYPE
            handle.addfile(info, io.BytesIO(raw))
    archive_raw = sink.getvalue()
    verify_source_bytes(
        archive_raw,
        manifest_raw,
        expected_archive_sha256=sha256(archive_raw),
        expected_manifest_sha256=sha256(manifest_raw),
    )
    return archive_raw, manifest_raw, document


def verify_source_bytes(
    archive_raw: bytes,
    manifest_raw: bytes,
    *,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    if sha256(archive_raw) != expected_archive_sha256:
        raise ReleaseError("v16r6 source archive SHA differs")
    if sha256(manifest_raw) != expected_manifest_sha256:
        raise ReleaseError("v16r6 source manifest SHA differs")
    manifest = parse_object(manifest_raw, "v16r6 source manifest")
    rows = manifest.get("files")
    if (
        manifest.get("schema_version") != SOURCE_SCHEMA
        or manifest.get("archive_format") != ARCHIVE_FORMAT
        or manifest.get("file_count") != SOURCE_FILE_COUNT
        or manifest.get("base_source_archive_sha256") != BASE_ARCHIVE_SHA256
        or manifest.get("base_source_manifest_sha256") != BASE_MANIFEST_SHA256
        or manifest.get("base_content_closure_sha256") != BASE_CLOSURE_SHA256
        or not isinstance(rows, list)
        or len(rows) != SOURCE_FILE_COUNT
    ):
        raise ReleaseError("v16r6 source manifest envelope differs")
    closure_raw = canonical({"schema_version": SOURCE_SCHEMA, "files": rows})
    if (
        manifest.get("content_closure_sha256") != sha256(closure_raw)
        or manifest.get("content_closure_sha1") != sha1(closure_raw)
    ):
        raise ReleaseError("v16r6 source content closure differs")
    data = _verify_rows(archive_raw, rows, label="v16r6")
    for member, (_, expected_sha) in SOURCE_MEMBERS.items():
        if sha256(data.get(member, b"")) != expected_sha:
            raise ReleaseError(f"v16r6 required member differs: {member}")
    return {
        "archive_sha256": expected_archive_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "content_closure_sha256": str(manifest["content_closure_sha256"]),
        "content_closure_sha1": str(manifest["content_closure_sha1"]),
        "file_count": SOURCE_FILE_COUNT,
    }


def verify_source(
    archive: Path,
    manifest: Path,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    return verify_source_bytes(
        regular_file(archive, "v16r6 source archive").read_bytes(),
        regular_file(manifest, "v16r6 source manifest").read_bytes(),
        expected_archive_sha256=expected_archive_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
    )


def write_new(path: Path, raw: bytes, mode: int) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    path.chmod(mode)


def build(
    workspace_root: Path,
    base_archive: Path,
    base_manifest: Path,
    output: Path,
) -> Mapping[str, Any]:
    workspace_root = workspace_root.resolve(strict=True)
    archive_raw, manifest_raw, source = build_source(
        workspace_root, base_archive, base_manifest
    )
    if output.exists() or output.is_symlink():
        raise ReleaseError("release output must be fresh")

    builder_raw = Path(__file__).resolve(strict=True).read_bytes()
    controls = {
        Path(relative).name: regular_file(workspace_root / relative, relative).read_bytes()
        for relative in CONTROL_MEMBERS
    }
    controls[Path(__file__).name] = builder_raw
    control_hashes = {name: sha256(raw) for name, raw in sorted(controls.items())}
    archive_sha = sha256(archive_raw)
    manifest_sha = sha256(manifest_raw)
    release = {
        "schema_version": RELEASE_SCHEMA,
        "created_date": "2026-08-24",
        "source_release": {
            "archive": "v16r6ab-source.tar",
            "manifest": "v16r6ab-source.manifest.json",
            "archive_sha256": archive_sha,
            "manifest_sha256": manifest_sha,
            "content_closure_sha256": source["content_closure_sha256"],
            "content_closure_sha1": source["content_closure_sha1"],
            "file_count": SOURCE_FILE_COUNT,
            "base_archive_sha256": BASE_ARCHIVE_SHA256,
            "base_manifest_sha256": BASE_MANIFEST_SHA256,
            "base_content_closure_sha256": BASE_CLOSURE_SHA256,
        },
        "control_file_sha256": control_hashes,
        "debug_contract": {
            "sealed_manifest_row_count": 644,
            "optimizer_step_budget": 32,
            "schedule": "sealed_full644_family_round_robin_prefix32_exact_once_debug",
            "exact644_training_complete": False,
            "terminal_full644_checkpoint": False,
            "scientific_claim_authorized": False,
            "training_complete_filename_is_only_process_completion": True,
            "resume_allowed": False,
            "seed": 2026082302,
        },
        "variants": {
            "a": {
                "trainer_member": "methods/bernini_action_editing/train_online_anchor_attention_full644_dynamic_static_v16r6a_lr1e7_32.py",
                "method": "bernini-online-anchor-v16r6a-full-lora-lr1e7-prefix32",
                "sole_changed_variable": "active_coordinate_rms_learning_rate_only",
                "learning_rate": 1e-7,
                "lora_scope": "all_30_blocks_attn1_attn2_qkvo",
                "target_module_count": 240,
                "trainable_tensor_count": 480,
                "trainable_parameter_count": 188_743_680,
                "target_modules_sha256": "d253ba3f11ec5ac26710a829d543a18b939c6f111c64be785264fcd852f3f35a",
            },
            "b": {
                "trainer_member": "methods/bernini_action_editing/train_online_anchor_attention_full644_dynamic_static_v16r6b_route_qk32.py",
                "method": "bernini-online-anchor-v16r6b-route-attn1-qk-prefix32",
                "sole_changed_variable": "lora_target_scope_only",
                "learning_rate": 1e-6,
                "lora_scope": "route_blocks_22_attn1_qk_only",
                "target_module_count": 44,
                "trainable_tensor_count": 88,
                "trainable_parameter_count": 34_603_008,
                "target_modules_sha256": "55d23681c5ee165e6f6b94f97730d7fe7e93031a0b83fd6ede20ce316f905cb4",
                "attention": "attn1",
                "projections": ["to_k", "to_q"],
                "nonroute_trainable": False,
                "attn2_trainable": False,
                "value_or_output_trainable": False,
            },
        },
        "training_inputs": {
            "full644_manifest_sha256": FULL644_MANIFEST_SHA256,
            "heldout8_manifest_sha256": HELDOUT8_SHA256,
            "all_data_objective_route_seed_optimizer_geometry_unchanged_from_v16r5_except_variant_field": True,
        },
        "preflight": {
            "new_contract_test_member": "methods/bernini_action_editing/tests/test_train_online_anchor_attention_v16r6_ab.py",
            "new_contract_expected_test_count": 11,
            "v16r5_regression_test_member": "methods/bernini_action_editing/tests/test_train_online_anchor_attention_full644_dynamic_static_v16r5.py",
            "v16r5_regression_expected_test_count": 8,
            "runs_before_any_gpu_step": True,
        },
    }
    release_raw = json.dumps(
        release, indent=2, sort_keys=True, ensure_ascii=True
    ).encode("ascii") + b"\n"

    output.mkdir(parents=True, mode=0o755)
    write_new(output / "v16r6ab-source.tar", archive_raw, 0o444)
    write_new(output / "v16r6ab-source.manifest.json", manifest_raw, 0o444)
    for name, raw in controls.items():
        write_new(output / name, raw, 0o555 if name.endswith(".sh") else 0o444)
    write_new(output / "v16r6ab-release.json", release_raw, 0o444)
    sums: list[str] = []
    for name in sorted(
        (
            "v16r6ab-source.tar",
            "v16r6ab-source.manifest.json",
            "v16r6ab-release.json",
            *controls.keys(),
        )
    ):
        sums.append(f"{sha256((output / name).read_bytes())}  {name}")
    write_new(
        output / "SHA256SUMS", ("\n".join(sums) + "\n").encode("ascii"), 0o444
    )
    verified = verify_source(
        output / "v16r6ab-source.tar",
        output / "v16r6ab-source.manifest.json",
        archive_sha,
        manifest_sha,
    )
    return {
        "output": str(output.resolve()),
        **verified,
        "release_manifest_sha256": sha256(release_raw),
        "control_file_sha256": control_hashes,
        "sha256sums_sha256": sha256((output / "SHA256SUMS").read_bytes()),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--workspace-root", type=Path, required=True)
    build_parser.add_argument("--base-source-archive", type=Path, required=True)
    build_parser.add_argument("--base-source-manifest", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--archive", type=Path, required=True)
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--expected-archive-sha256", required=True)
    verify_parser.add_argument("--expected-manifest-sha256", required=True)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "build":
        result = build(
            args.workspace_root,
            args.base_source_archive,
            args.base_source_manifest,
            args.output,
        )
    else:
        result = verify_source(
            args.archive,
            args.manifest,
            args.expected_archive_sha256,
            args.expected_manifest_sha256,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
