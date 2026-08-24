#!/usr/bin/env python3
"""Build/audit the deterministic preservation-v2 four-holder training release."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
from typing import Any, Mapping, Sequence


SCHEMA = "bernini-self-generated-action-preservation-v2-release-v1"
ENVELOPE_SCHEMA = "bernini-self-generated-action-preservation-v2-deployment-v1"
RELEASE_GENERATION = "preservation-v2-seed20260818-four-holder-r1"
MEMBER_ROOT = "methods/bernini_action_editing"
ARCHIVE_FORMAT = "fixed-ustar-ascii-zero-dev-sorted-owner0-mtime0-record10240-v1"
FIXED_USTAR_BLOCK_SIZE = 512
FIXED_USTAR_RECORD_SIZE = 10240
SEED = 20260818
SOURCE_DATA_MANIFEST_SHA256 = "62fee73b3d84015f2e72edcd4da14b51f7695980a4ba892420ca137aa50e9ad8"
SOURCE_DATA_MANIFEST_DIGEST = "2fb367ed6f06275705e0b71020dd87fd68e13a010e80ef0bd2a122c94070f503"
FROZEN_SITE_PACKAGES = (
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages"
)
TORCHRUN_PATH = FROZEN_SITE_PACKAGES + "/torch/distributed/run.py"
TORCHRUN_SHA256 = "1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c"
TORCHRUN_SIZE = 31587
DETACHED_CONTROLLER = "auh_launch_self_generated_action_preservation_v2_four_holder_v1.sh"
DETACHED_CONTROLLER_SHA256 = "d522fa711014a5ca5b671448ce24afab14e3dbf63fd9df45b0112745a01dd995"
ARMS = (
    "v2_onset_all", "v2_noop020_all", "v2_func010_all", "v2_func025_all",
    "v2_func050_all", "v2_onset_cross_qo", "v2_func010_cross_qo",
    "v2_func025_cross_qo",
)
FILES_AND_MODES: Mapping[str, int] = {
    "action_preservation_completion_publisher_v1.py": 0o444,
    "action_preservation_verified_release_v1.py": 0o444,
    "audit_self_generated_action_preservation_v2.py": 0o444,
    "full30_action_learning_v1.py": 0o444,
    "scripts/auh_run_self_generated_action_preservation_v2.sh": 0o555,
    "self_generated_action_preservation_v2.py": 0o444,
    "self_generated_action_quotient_v1.py": 0o444,
    "train_lora.py": 0o444,
    "train_self_generated_action_quotient_v1.py": 0o444,
}
EXPECTED_SHA256 = {
    "action_preservation_completion_publisher_v1.py": "cce0993a31f233cf57ddf867b2d0d8b0c98a0a1583deb789c5dab898d5507714",
    "action_preservation_verified_release_v1.py": "d3be1415df9d1108daeb5935349db730fddd9b64956e49ba2afbd32c1b9a309a",
    "audit_self_generated_action_preservation_v2.py": "60019b15cd520bacb643959494f4af41eadf34c9012df61084e2554871c1c868",
    "full30_action_learning_v1.py": "67275ae09e7cb7b1e7e8fc43ce2928031b3fe8aabe213e8626000f37abad4ead",
    "scripts/auh_run_self_generated_action_preservation_v2.sh": "f4ded1acbb519684b724e9d6876d5bb2e259c3bfd15b7ee03fa633a0597474ce",
    "self_generated_action_preservation_v2.py": "11bc0792174a60c2e449eb61ff8f81da97808e02ee2707b5c4f20ee2118f4b5c",
    "self_generated_action_quotient_v1.py": "a9bfec2816ec1b6ccb2a336ea25600f15f22557aea76b1ea0605bbeb737b501c",
    "train_lora.py": "eae8eaac25197112637f466e611ba7eae574266d4cd1b83e625195fb22b0476e",
    "train_self_generated_action_quotient_v1.py": "a2f2153d37e21e9f77567eab39152318105e924bf84e7ffbc2cfda12c06ed24b",
}
EXPECTED_SIZE = {
    "action_preservation_completion_publisher_v1.py": 38816,
    "action_preservation_verified_release_v1.py": 44137,
    "audit_self_generated_action_preservation_v2.py": 39221,
    "full30_action_learning_v1.py": 25591,
    "scripts/auh_run_self_generated_action_preservation_v2.sh": 17630,
    "self_generated_action_preservation_v2.py": 11334,
    "self_generated_action_quotient_v1.py": 5549,
    "train_lora.py": 84216,
    "train_self_generated_action_quotient_v1.py": 111170,
}
ALLOWED_ENTRYPOINTS = (
    "scripts/auh_run_self_generated_action_preservation_v2.sh",
    "audit_self_generated_action_preservation_v2.py",
    "action_preservation_verified_release_v1.py",
    "action_preservation_completion_publisher_v1.py",
    DETACHED_CONTROLLER,
)


class ReleaseError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _unique_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(token: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {token}")


def load_canonical_json(raw: bytes, label: str) -> Any:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, ReleaseError) as error:
        raise ReleaseError(f"{label} is not strict JSON") from error
    require(canonical(value) + b"\n" == raw, f"{label} bytes are not canonical")
    return value


def exact_json(value: Any, expected: Any) -> bool:
    """Compare JSON values without Python's bool/int equality aliasing."""

    try:
        return canonical(value) == canonical(expected)
    except (TypeError, ValueError):
        return False


def method_root() -> Path:
    return Path(__file__).resolve().parents[1]


def payload_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    require(
        set(FILES_AND_MODES) == set(EXPECTED_SHA256) == set(EXPECTED_SIZE),
        "builder pin closure differs",
    )
    for relative in sorted(FILES_AND_MODES):
        source = root / relative
        details = source.lstat()
        require(stat.S_ISREG(details.st_mode) and not source.is_symlink(), f"member source differs: {relative}")
        raw = source.read_bytes()
        digest = sha256(raw)
        require(digest == EXPECTED_SHA256[relative], f"member SHA differs: {relative}")
        require(len(raw) == EXPECTED_SIZE[relative], f"member size differs: {relative}")
        payloads[relative] = raw
        rows.append(
            {
                "path": relative,
                "mode": FILES_AND_MODES[relative],
                "size": len(raw),
                "sha256": digest,
            }
        )
    return rows, payloads


def content_revision(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha1(canonical(list(rows))).hexdigest()


def expected_file_rows() -> list[dict[str, Any]]:
    return [
        {
            "path": relative,
            "mode": FILES_AND_MODES[relative],
            "size": EXPECTED_SIZE[relative],
            "sha256": EXPECTED_SHA256[relative],
        }
        for relative in sorted(FILES_AND_MODES)
    ]


def authority_value() -> dict[str, Any]:
    return {
        "training_kind": "preservation-v2-20-update-mechanism-canary",
        "objective_family": "preservation_v2",
        "seed": SEED,
        "initialization_seed": SEED,
        "teacher_cache_seed": SEED,
        "teacher_cache_must_be_fresh": True,
        "teacher_cache_shape": "four-iid-by-five-sigma-bin",
        "source_data_manifest_sha256": SOURCE_DATA_MANIFEST_SHA256,
        "source_data_manifest_digest": SOURCE_DATA_MANIFEST_DIGEST,
        "max_steps_per_arm": 20,
        "checkpoint_steps": [0, 5, 10, 20],
        "arms": list(ARMS),
        "holders": {
            "136719": "auh7-1b-gpu-306",
            "136141": "auh7-1b-gpu-299",
            "136309": "auh7-1b-gpu-280",
            "136140": "auh7-1b-gpu-215",
        },
        "parent_cancel_release_requeue_forbidden": True,
        "automatic_retry": False,
        "training_loss_promotion_forbidden": True,
        "decoded_identity_background_camera_claim_authorized": False,
        "blind_full_video_review_required_for_promotion": True,
        "scientific_claim_authorized": False,
        "experimental_training": True,
        "isolated_frozen_runtime": {
            "python_flags": ["-I", "-S", "-B", "-c"],
            "site_packages": FROZEN_SITE_PACKAGES,
            "site_packages_added_only_after_full_release_capture": True,
            "automatic_site_initialization_disabled": True,
            "torchrun_launcher": {
                "path": TORCHRUN_PATH,
                "sha256": TORCHRUN_SHA256,
                "size": TORCHRUN_SIZE,
                "uid": 2012,
                "gid": 2000,
                "mode": "0644",
                "link_count": 1,
            },
            "torchrun_same_fd_double_read_full_identity_required": True,
            "torchrun_executed_from_captured_source": True,
            "rank_python_is_root_verified_bootstrap": True,
        },
    }


def component_sha256_value(controller_sha: str) -> dict[str, str]:
    return {
        "trainer": EXPECTED_SHA256["train_self_generated_action_quotient_v1.py"],
        "objective": EXPECTED_SHA256["self_generated_action_preservation_v2.py"],
        "auditor": EXPECTED_SHA256["audit_self_generated_action_preservation_v2.py"],
        "node_runner": EXPECTED_SHA256[
            "scripts/auh_run_self_generated_action_preservation_v2.sh"
        ],
        "verified_release": EXPECTED_SHA256[
            "action_preservation_verified_release_v1.py"
        ],
        "completion_publisher": EXPECTED_SHA256[
            "action_preservation_completion_publisher_v1.py"
        ],
        "detached_controller": controller_sha,
    }


def make_manifest(rows: list[dict[str, Any]], controller_sha: str) -> dict[str, Any]:
    require(exact_json(rows, expected_file_rows()), "manifest source rows differ")
    require(
        controller_sha == DETACHED_CONTROLLER_SHA256,
        "detached controller pin differs",
    )
    value: dict[str, Any] = {
        "schema_version": SCHEMA,
        "release_generation": RELEASE_GENERATION,
        "member_root": MEMBER_ROOT,
        "archive_format": ARCHIVE_FORMAT,
        "file_count": len(rows),
        "exact_member_closure": True,
        "files": rows,
        "content_revision": content_revision(rows),
        "allowed_entrypoints": list(ALLOWED_ENTRYPOINTS),
        "authority": authority_value(),
        "component_sha256": component_sha256_value(controller_sha),
    }
    value["manifest_digest"] = sha256(canonical(value))
    return value


def _ustar_text(value: str, width: int, label: str) -> bytes:
    require(type(value) is str and "\0" not in value, f"{label} differs")
    try:
        raw = value.encode("ascii", "strict")
    except UnicodeEncodeError as error:
        raise ReleaseError(f"{label} is not canonical USTAR ASCII") from error
    require(len(raw) <= width, f"{label} exceeds canonical USTAR width")
    return raw + b"\0" * (width - len(raw))


def _ustar_octal(value: int, width: int, label: str) -> bytes:
    require(type(value) is int and value >= 0, f"{label} differs")
    digits = width - 1
    require(value < 8**digits, f"{label} exceeds canonical USTAR octal width")
    raw = f"{value:0{digits}o}".encode("ascii") + b"\0"
    require(len(raw) == width, f"{label} canonical USTAR width differs")
    return raw


def _ustar_name_fields(value: str) -> tuple[bytes, bytes]:
    require(type(value) is str and value and "\0" not in value, "USTAR member name differs")
    try:
        encoded = value.encode("ascii", "strict")
    except UnicodeEncodeError as error:
        raise ReleaseError("USTAR member name is not ASCII") from error
    if len(encoded) <= 100:
        return _ustar_text(value, 100, "USTAR name"), b"\0" * 155
    for index in range(len(value) - 1, -1, -1):
        if value[index] != "/":
            continue
        prefix, basename = value[:index], value[index + 1 :]
        if not prefix or not basename:
            continue
        try:
            prefix_raw = prefix.encode("ascii", "strict")
            basename_raw = basename.encode("ascii", "strict")
        except UnicodeEncodeError:
            continue
        if len(prefix_raw) <= 155 and len(basename_raw) <= 100:
            return (
                _ustar_text(basename, 100, "USTAR name"),
                _ustar_text(prefix, 155, "USTAR prefix"),
            )
    raise ReleaseError("USTAR member name cannot be represented without extensions")


def fixed_ustar_header(name: str, *, size: int, mode: int) -> bytes:
    """Serialize one regular USTAR header without host ``tarfile`` behavior."""

    name_field, prefix_field = _ustar_name_fields(name)
    header = bytearray(FIXED_USTAR_BLOCK_SIZE)
    header[0:100] = name_field
    header[100:108] = _ustar_octal(mode, 8, "USTAR mode")
    header[108:116] = _ustar_octal(0, 8, "USTAR uid")
    header[116:124] = _ustar_octal(0, 8, "USTAR gid")
    header[124:136] = _ustar_octal(size, 12, "USTAR size")
    header[136:148] = _ustar_octal(0, 12, "USTAR mtime")
    header[148:156] = b" " * 8
    header[156:157] = b"0"
    header[157:257] = b"\0" * 100
    header[257:263] = b"ustar\0"
    header[263:265] = b"00"
    header[265:297] = b"\0" * 32
    header[297:329] = b"\0" * 32
    header[329:337] = _ustar_octal(0, 8, "USTAR devmajor")
    header[337:345] = _ustar_octal(0, 8, "USTAR devminor")
    header[345:500] = prefix_field
    header[500:512] = b"\0" * 12
    checksum = sum(header)
    require(checksum < 8**6, "USTAR checksum exceeds field width")
    header[148:156] = f"{checksum:06o}\0 ".encode("ascii")
    require(len(header) == FIXED_USTAR_BLOCK_SIZE, "USTAR header size differs")
    return bytes(header)


def make_archive(payloads: Mapping[str, bytes]) -> bytes:
    require(set(payloads) == set(FILES_AND_MODES), "archive payload closure differs")
    output = bytearray()
    for relative in sorted(payloads):
        raw = payloads[relative]
        require(type(raw) is bytes, f"archive payload type differs: {relative}")
        output.extend(
            fixed_ustar_header(
                f"{MEMBER_ROOT}/{relative}",
                size=len(raw),
                mode=FILES_AND_MODES[relative],
            )
        )
        output.extend(raw)
        output.extend(b"\0" * (-len(raw) % FIXED_USTAR_BLOCK_SIZE))
    output.extend(b"\0" * (2 * FIXED_USTAR_BLOCK_SIZE))
    output.extend(b"\0" * (-len(output) % FIXED_USTAR_RECORD_SIZE))
    require(
        len(output) % FIXED_USTAR_RECORD_SIZE == 0,
        "archive fixed USTAR record boundary differs",
    )
    return bytes(output)


def validate_manifest(value: Any) -> Mapping[str, Any]:
    top_fields = {
        "schema_version", "release_generation", "member_root", "archive_format",
        "file_count", "exact_member_closure", "files", "content_revision",
        "allowed_entrypoints", "authority", "component_sha256", "manifest_digest",
    }
    require(type(value) is dict and set(value) == top_fields, "manifest top-level field closure differs")
    unsigned = dict(value)
    declared = unsigned.pop("manifest_digest", None)
    require(
        type(declared) is str and declared == sha256(canonical(unsigned)),
        "manifest digest differs",
    )
    require(
        value["schema_version"] == SCHEMA
        and value["release_generation"] == RELEASE_GENERATION
        and value["member_root"] == MEMBER_ROOT
        and value["archive_format"] == ARCHIVE_FORMAT
        and type(value["file_count"]) is int
        and value["file_count"] == len(FILES_AND_MODES)
        and value["exact_member_closure"] is True,
        "manifest identity/value closure differs",
    )
    rows = value["files"]
    expected_rows = expected_file_rows()
    require(type(rows) is list and len(rows) == len(expected_rows), "manifest file count differs")
    for row, expected in zip(rows, expected_rows):
        require(
            type(row) is dict
            and set(row) == {"path", "mode", "size", "sha256"}
            and exact_json(row, expected),
            f"manifest file row field/value closure differs: {row!r}",
        )
    require(
        value["content_revision"] == content_revision(expected_rows),
        "manifest content revision differs",
    )
    require(
        exact_json(value["allowed_entrypoints"], list(ALLOWED_ENTRYPOINTS)),
        "manifest entrypoint closure differs",
    )
    authority = value["authority"]
    expected_authority = authority_value()
    require(
        type(authority) is dict
        and set(authority) == set(expected_authority)
        and exact_json(authority, expected_authority),
        "manifest authority field/value closure or semantic authority differs",
    )
    components = value["component_sha256"]
    expected_components = component_sha256_value(DETACHED_CONTROLLER_SHA256)
    require(
        type(components) is dict
        and set(components) == set(expected_components)
        and exact_json(components, expected_components),
        "manifest component field/value closure differs",
    )
    expected = make_manifest(expected_rows, DETACHED_CONTROLLER_SHA256)
    require(exact_json(value, expected), "manifest exact authority closure differs")
    return value


def make_envelope(
    archive_raw: bytes, manifest_raw: bytes, manifest: Mapping[str, Any],
    controller_raw: bytes,
) -> dict[str, Any]:
    validate_manifest(manifest)
    require(
        canonical(manifest) + b"\n" == manifest_raw,
        "envelope source manifest bytes differ",
    )
    require(
        sha256(controller_raw) == DETACHED_CONTROLLER_SHA256,
        "envelope detached controller pin differs",
    )
    value: dict[str, Any] = {
        "schema_version": ENVELOPE_SCHEMA,
        "release_generation": RELEASE_GENERATION,
        "seed": SEED,
        "remote_release_exact_entries": [
            DETACHED_CONTROLLER, "deployment-envelope.json", "source.manifest.json", "source.tar"
        ],
        "source_archive": {
            "basename": "source.tar", "sha256": sha256(archive_raw), "mode": 0o444
        },
        "source_manifest": {
            "basename": "source.manifest.json",
            "sha256": sha256(manifest_raw),
            "manifest_digest": manifest["manifest_digest"],
            "content_revision": manifest["content_revision"],
            "file_count": manifest["file_count"],
            "mode": 0o444,
        },
        "detached_controller": {
            "basename": DETACHED_CONTROLLER,
            "sha256": sha256(controller_raw),
            "mode": 0o555,
        },
        "create_only_deployment_required": True,
        "fresh_experiment_root_required": True,
        "launch_authorized_by_user_request": True,
        "evaluation_and_blind_review_still_required": True,
        "automatic_scientific_promotion_authorized": False,
    }
    value["envelope_digest"] = sha256(canonical(value))
    return value


def validate_envelope(
    value: Any,
    *,
    archive_raw: bytes,
    manifest_raw: bytes,
    manifest: Mapping[str, Any],
    controller_raw: bytes,
) -> Mapping[str, Any]:
    top_fields = {
        "schema_version", "release_generation", "seed",
        "remote_release_exact_entries", "source_archive", "source_manifest",
        "detached_controller", "create_only_deployment_required",
        "fresh_experiment_root_required", "launch_authorized_by_user_request",
        "evaluation_and_blind_review_still_required",
        "automatic_scientific_promotion_authorized", "envelope_digest",
    }
    require(type(value) is dict and set(value) == top_fields, "envelope top-level field closure differs")
    require(
        type(value.get("source_archive")) is dict
        and set(value["source_archive"]) == {"basename", "sha256", "mode"},
        "envelope source archive field closure differs",
    )
    require(
        type(value.get("source_manifest")) is dict
        and set(value["source_manifest"])
        == {
            "basename", "sha256", "manifest_digest", "content_revision",
            "file_count", "mode",
        },
        "envelope source manifest field closure differs",
    )
    require(
        type(value.get("detached_controller")) is dict
        and set(value["detached_controller"]) == {"basename", "sha256", "mode"},
        "envelope detached controller field closure differs",
    )
    unsigned = dict(value)
    declared = unsigned.pop("envelope_digest", None)
    require(
        type(declared) is str and declared == sha256(canonical(unsigned)),
        "envelope digest differs",
    )
    expected = make_envelope(
        archive_raw, manifest_raw, manifest, controller_raw
    )
    require(
        exact_json(value, expected),
        "envelope field/value closure or scientific authority differs",
    )
    return value


def write_create_only(path: Path, raw: bytes, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, f"short write: {path}")
            view = view[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build(release_dir: Path) -> dict[str, Any]:
    root = method_root()
    controller_source = root / "scripts" / DETACHED_CONTROLLER
    details = controller_source.lstat()
    require(stat.S_ISREG(details.st_mode) and not controller_source.is_symlink(), "controller source differs")
    controller_raw = controller_source.read_bytes()
    require(
        sha256(controller_raw) == DETACHED_CONTROLLER_SHA256,
        "detached controller SHA differs",
    )
    rows, payloads = payload_rows(root)
    manifest = make_manifest(rows, sha256(controller_raw))
    validate_manifest(manifest)
    manifest_raw = canonical(manifest) + b"\n"
    archive_raw = make_archive(payloads)
    envelope = make_envelope(archive_raw, manifest_raw, manifest, controller_raw)
    validate_envelope(
        envelope,
        archive_raw=archive_raw,
        manifest_raw=manifest_raw,
        manifest=manifest,
        controller_raw=controller_raw,
    )
    envelope_raw = canonical(envelope) + b"\n"
    require(not release_dir.exists() and not release_dir.is_symlink(), "release directory is not fresh")
    release_dir.mkdir(mode=0o700, parents=False)
    write_create_only(release_dir / "source.tar", archive_raw, 0o444)
    write_create_only(release_dir / "source.manifest.json", manifest_raw, 0o444)
    write_create_only(release_dir / DETACHED_CONTROLLER, controller_raw, 0o555)
    write_create_only(release_dir / "deployment-envelope.json", envelope_raw, 0o444)
    descriptor = os.open(release_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return audit(release_dir, against_workspace=True)


def audit(release_dir: Path, *, against_workspace: bool) -> dict[str, Any]:
    expected_names = {
        "source.tar", "source.manifest.json", DETACHED_CONTROLLER, "deployment-envelope.json"
    }
    require(release_dir.is_dir() and not release_dir.is_symlink(), "release directory differs")
    require({path.name for path in release_dir.iterdir()} == expected_names, "release entry closure differs")
    for name, mode in {
        "source.tar": 0o444,
        "source.manifest.json": 0o444,
        DETACHED_CONTROLLER: 0o555,
        "deployment-envelope.json": 0o444,
    }.items():
        path = release_dir / name
        details = path.lstat()
        require(stat.S_ISREG(details.st_mode) and not path.is_symlink() and details.st_nlink == 1, f"release topology differs: {name}")
        require(stat.S_IMODE(details.st_mode) == mode, f"release mode differs: {name}")
    manifest_raw = (release_dir / "source.manifest.json").read_bytes()
    manifest = load_canonical_json(manifest_raw, "release manifest")
    validate_manifest(manifest)
    rows = manifest["files"]
    archive_raw = (release_dir / "source.tar").read_bytes()
    payloads: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:") as archive:
            members = archive.getmembers()
            expected_members = [
                f"{MEMBER_ROOT}/{relative}" for relative in sorted(FILES_AND_MODES)
            ]
            require(
                [member.name for member in members] == expected_members,
                "archive member closure/order differs",
            )
            for member, row in zip(members, rows):
                require(
                    member.isfile()
                    and member.type == tarfile.REGTYPE
                    and not member.linkname,
                    f"archive member kind differs: {member.name}",
                )
                require(
                    member.uid == member.gid == member.mtime == 0
                    and member.uname == member.gname == ""
                    and not member.pax_headers,
                    f"archive metadata differs: {member.name}",
                )
                require(
                    member.mode == row["mode"] and member.size == row["size"],
                    f"archive row differs: {member.name}",
                )
                handle = archive.extractfile(member)
                require(handle is not None, "archive payload absent")
                raw = handle.read()
                require(
                    len(raw) == row["size"] and sha256(raw) == row["sha256"],
                    f"archive payload SHA/size differs: {member.name}",
                )
                payloads[row["path"]] = raw
    except (OSError, tarfile.TarError) as error:
        raise ReleaseError("release archive is not strict USTAR") from error
    require(set(payloads) == set(FILES_AND_MODES), "archive payload closure differs")
    require(
        archive_raw == make_archive(payloads),
        "archive canonical USTAR byte closure differs",
    )
    controller_raw = (release_dir / DETACHED_CONTROLLER).read_bytes()
    require(
        sha256(controller_raw) == DETACHED_CONTROLLER_SHA256,
        "detached controller SHA differs",
    )
    if against_workspace:
        workspace_rows, workspace_payloads = payload_rows(method_root())
        require(rows == workspace_rows and payloads == workspace_payloads, "workspace payload closure differs")
        require(
            controller_raw
            == (method_root() / "scripts" / DETACHED_CONTROLLER).read_bytes(),
            "detached controller differs",
        )
    envelope_raw = (release_dir / "deployment-envelope.json").read_bytes()
    envelope = load_canonical_json(envelope_raw, "deployment envelope")
    validate_envelope(
        envelope,
        archive_raw=archive_raw,
        manifest_raw=manifest_raw,
        manifest=manifest,
        controller_raw=controller_raw,
    )
    return {
        "static_audit_go": True,
        "release_dir": str(release_dir),
        "archive_sha256": sha256(archive_raw),
        "manifest_sha256": sha256(manifest_raw),
        "manifest_digest": manifest["manifest_digest"],
        "content_revision": manifest["content_revision"],
        "controller_sha256": envelope["detached_controller"]["sha256"],
        "envelope_sha256": sha256(envelope_raw),
        "envelope_digest": envelope["envelope_digest"],
        "file_count": len(rows),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    sub = value.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--release-dir", required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--release-dir", required=True)
    audit_parser.add_argument("--against-workspace", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    release_dir = Path(args.release_dir).resolve()
    result = (
        build(release_dir)
        if args.command == "build"
        else audit(release_dir, against_workspace=args.against_workspace)
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
