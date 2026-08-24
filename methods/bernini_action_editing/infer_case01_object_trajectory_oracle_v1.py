#!/usr/bin/env python3
"""Fail-closed case01 object-trajectory oracle wrapper for frozen Full644 R5.

This file is an experimental runner integration, not a learned method.  It
peels only its own CLI arguments and delegates all remaining arguments to an
exact, SHA-pinned legacy ``infer_lora.py`` snapshot.  The ``off`` arm is the
strong null: it loads no oracle asset and installs no patch.  ``route_off``
validates the complete oracle authority but does not encode or consume it.
The two active arms temporarily add one rank-zero VAE encode of a
bone-removed source and replace the legacy every-step phase-zero clamp with a
masked source-object trajectory projection.

The frozen legacy snapshot is deliberately not vendored by this source file.
Release materialization must create ``LEGACY_BASENAME`` from the sealed R5
snapshot.  A missing or byte-different snapshot fails before delegation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any, Iterator, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
LEGACY_BASENAME = "infer_lora_full644_r5_frozen_acc46.py"
LEGACY_INFER_LORA_SHA256 = (
    "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553"
)
LEGACY_INFER_LORA_SIZE = 177_300
PROJECTION_BASENAME = "object_trajectory_projection_v1.py"
PROJECTION_SHA256 = (
    "a851afd220d770e6c8082ff8e7f0a0c2b5a5be894bcbf9feeaf8bca4cc6c9e7e"
)
PROJECTION_SIZE = 47_588
SCAFFOLD_BASENAME = "case01_oracle_object_trajectory_v1.py"
# Filled only after the independently tested scaffold source lands.  Active
# and route_off arms fail closed while this remains None; arm=off never reads
# or checks the scaffold module.
SCAFFOLD_SOURCE_SHA256: str | None = (
    "a7d4e008e78d373370b832c0754e5d7420b504fa8b2022eef9a9bb899added8a"
)
SCAFFOLD_SOURCE_SIZE = 35_803
CASE01_SCAFFOLD_JSON_SHA256 = (
    "7b1bec6e9764a1297bb0029f8fea01ebe4b2deab0acc2c7f07fdee96bc0a098a"
)
CASE01_SCAFFOLD_JSON_SIZE = 54_801
CASE01_SCAFFOLD_ARTIFACT_DIGEST = (
    "5e6156909d8261a23c3add3134059bec20505b682ca0eb13dc88fa8512eeace1"
)
CASE01_BONE_REMOVED_SHA256 = (
    "8c525385832586fa7b7fd7ae6e5701c599694d26ee27b502dbf0bb582e55e1c9"
)
CASE01_BONE_REMOVED_SIZE = 5_424_975

WRAPPER_RECEIPT_SCHEMA = (
    "bernini-r-1p3b-case01-object-trajectory-oracle-inference-receipt-v3"
)
RUNTIME_TRACE_SCHEMA = "bernini-case01-object-trajectory-oracle-runtime-v3"
ORACLE_ARMS = (
    "off",
    "route_off",
    "trajectory_bone_only",
    "trajectory_dog_bone",
)
ACTIVE_ARMS = frozenset(("trajectory_bone_only", "trajectory_dog_bone"))
PACKED_CHANNELS = 64
EXPECTED_LATENT_PHASES = 21
EXPECTED_TOKEN_SHAPE = (21, 31, 30)
EXPECTED_SPATIAL_TOKENS = 31 * 30
EXPECTED_SEGMENT_TOKENS = 21 * 31 * 30
SHA256_HEX_LENGTH = 64


class ObjectOracleWrapperError(RuntimeError):
    """Raised before a claim when the wrapper cannot prove its authority."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ObjectOracleWrapperError(
            f"value is not canonical JSON: {error}"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ObjectOracleWrapperError(f"{label} must be one lowercase SHA-256")
    return value


def _canonical_plain_path(raw: str | Path, *, label: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise ObjectOracleWrapperError(f"{label} must be an absolute path")
    if "\x00" in str(path):
        raise ObjectOracleWrapperError(f"{label} contains NUL")
    canonical = Path(os.path.realpath(os.fspath(path)))
    if canonical != path:
        raise ObjectOracleWrapperError(
            f"{label} must already be canonical and contain no symlink component: {path}"
        )
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ObjectOracleWrapperError(f"cannot stat {label}: {path}: {error}") from error
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise ObjectOracleWrapperError(f"{label} must be one plain non-symlink file")
    return path


def _identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(info.st_mode),
        "nlink": int(info.st_nlink),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


@dataclass
class StableFileAuthority:
    """One open, no-follow file identity retained across active consumption."""

    path: Path
    label: str
    descriptor: int
    sha256: str
    identity: dict[str, int]

    @classmethod
    def open(
        cls,
        raw: str | Path,
        *,
        label: str,
        expected_sha256: str | None = None,
        expected_size: int | None = None,
    ) -> "StableFileAuthority":
        path = _canonical_plain_path(raw, label=label)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise ObjectOracleWrapperError(
                f"cannot open {label} without following links: {path}: {error}"
            ) from error
        try:
            os.set_inheritable(descriptor, False)
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) not in (0o444, 0o644)
            ):
                raise ObjectOracleWrapperError(
                    f"{label} must be one-link regular mode 0444/0644"
                )
            digest = hashlib.sha256()
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
            after = os.fstat(descriptor)
            if _identity(before) != _identity(after):
                raise ObjectOracleWrapperError(f"{label} changed while hashing")
            observed = digest.hexdigest()
            if expected_sha256 is not None and observed != _strict_sha256(
                expected_sha256, label=f"expected {label} SHA-256"
            ):
                raise ObjectOracleWrapperError(
                    f"{label} SHA-256 differs: {observed} != {expected_sha256}"
                )
            if expected_size is not None and int(after.st_size) != expected_size:
                raise ObjectOracleWrapperError(
                    f"{label} size differs: {after.st_size} != {expected_size}"
                )
            current = path.lstat()
            if _identity(current) != _identity(after):
                raise ObjectOracleWrapperError(
                    f"{label} pathname and retained descriptor identities differ"
                )
            os.lseek(descriptor, 0, os.SEEK_SET)
            return cls(path, label, descriptor, observed, _identity(after))
        except Exception:
            os.close(descriptor)
            raise

    def read_all(self) -> bytes:
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            block = os.read(self.descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        if _identity(os.fstat(self.descriptor)) != self.identity:
            raise ObjectOracleWrapperError(f"retained {self.label} identity changed")
        payload = b"".join(chunks)
        if hashlib.sha256(payload).hexdigest() != self.sha256:
            raise ObjectOracleWrapperError(f"retained {self.label} bytes changed")
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        return payload

    def replay(self) -> None:
        digest = hashlib.sha256()
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        while True:
            block = os.read(self.descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        if (
            digest.hexdigest() != self.sha256
            or _identity(os.fstat(self.descriptor)) != self.identity
            or _identity(self.path.lstat()) != self.identity
        ):
            raise ObjectOracleWrapperError(f"{self.label} changed after validation")

    def receipt(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "identity": dict(self.identity),
            "authority_digest": _object_sha256(
                {
                    "path": str(self.path),
                    "sha256": self.sha256,
                    "identity": self.identity,
                }
            ),
        }

    def close(self) -> None:
        descriptor, self.descriptor = self.descriptor, -1
        if descriptor >= 0:
            os.close(descriptor)


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ObjectOracleWrapperError(f"invalid {label} JSON: {error}") from error
    if type(value) is not dict:
        raise ObjectOracleWrapperError(f"{label} must contain one JSON object")
    return value


def _linux_retained_fd_consumer_path(descriptor: int) -> Path:
    """Return the production aux consumer path; tests may patch this seam only."""

    if sys.platform != "linux" or type(descriptor) is not int or descriptor < 0:
        raise ObjectOracleWrapperError(
            "active oracle requires a valid retained descriptor on Linux"
        )
    path = Path(f"/proc/self/fd/{descriptor}")
    if not path.exists():
        raise ObjectOracleWrapperError(
            "active oracle requires Linux /proc/self/fd retained aux consumption"
        )
    return path


def _load_module_from_authority(
    authority: StableFileAuthority, *, module_name: str
) -> types.ModuleType:
    raw = authority.read_all()
    try:
        code = compile(raw, str(authority.path), "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as error:
        raise ObjectOracleWrapperError(
            f"cannot compile pinned {authority.label}: {error}"
        ) from error
    module = types.ModuleType(module_name)
    module.__file__ = str(authority.path)
    module.__package__ = ""
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def _load_frozen_legacy() -> tuple[types.ModuleType, StableFileAuthority]:
    authority = StableFileAuthority.open(
        METHOD_ROOT / LEGACY_BASENAME,
        label="frozen Full644 R5 infer_lora source",
        expected_sha256=LEGACY_INFER_LORA_SHA256,
        expected_size=LEGACY_INFER_LORA_SIZE,
    )
    try:
        module = _load_module_from_authority(
            authority, module_name="_bernini_full644_r5_infer_lora_acc46"
        )
    except Exception:
        authority.close()
        raise
    return module, authority


def _load_pinned_support(
    basename: str,
    expected_sha256: str | None,
    *,
    expected_size: int,
    label: str,
    module_name: str,
) -> tuple[types.ModuleType, StableFileAuthority]:
    if expected_sha256 is None:
        raise ObjectOracleWrapperError(
            f"{label} source SHA-256 is not frozen; non-off arms are disabled"
        )
    authority = StableFileAuthority.open(
        METHOD_ROOT / basename,
        label=label,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    )
    try:
        return _load_module_from_authority(authority, module_name=module_name), authority
    except Exception:
        authority.close()
        raise


@dataclass(frozen=True)
class OracleCLI:
    arm: str
    scaffold: str | None
    scaffold_sha256: str | None
    scaffold_digest: str | None
    bone_removed_video: str | None
    bone_removed_video_sha256: str | None


def peel_object_oracle_cli(
    argv: Optional[Sequence[str]],
) -> tuple[OracleCLI, list[str]]:
    values = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--object-oracle-arm", choices=ORACLE_ARMS, default="off")
    parser.add_argument("--object-oracle-scaffold")
    parser.add_argument("--object-oracle-scaffold-sha256")
    parser.add_argument("--object-oracle-scaffold-digest")
    parser.add_argument("--object-oracle-bone-removed-video")
    parser.add_argument("--object-oracle-bone-removed-video-sha256")
    parsed, legacy = parser.parse_known_args(values)
    return (
        OracleCLI(
            arm=parsed.object_oracle_arm,
            scaffold=parsed.object_oracle_scaffold,
            scaffold_sha256=parsed.object_oracle_scaffold_sha256,
            scaffold_digest=parsed.object_oracle_scaffold_digest,
            bone_removed_video=parsed.object_oracle_bone_removed_video,
            bone_removed_video_sha256=parsed.object_oracle_bone_removed_video_sha256,
        ),
        legacy,
    )


def _legacy_option(argv: Sequence[str], name: str) -> str | None:
    matches: list[str] = []
    for index, item in enumerate(argv):
        if item == name:
            if index + 1 >= len(argv):
                raise ObjectOracleWrapperError(f"legacy option {name} lacks a value")
            matches.append(argv[index + 1])
        elif item.startswith(f"{name}="):
            matches.append(item.split("=", 1)[1])
    if len(matches) > 1:
        raise ObjectOracleWrapperError(f"legacy option {name} is duplicated")
    return matches[0] if matches else None


@dataclass
class OracleAssets:
    cli: OracleCLI
    scaffold: dict[str, Any]
    scaffold_file: StableFileAuthority
    aux_file: StableFileAuthority
    projection_module: types.ModuleType
    projection_source: StableFileAuthority
    scaffold_module: types.ModuleType
    scaffold_source: StableFileAuthority
    legacy_source: StableFileAuthority

    def close(self) -> None:
        for authority in (
            self.scaffold_file,
            self.aux_file,
            self.projection_source,
            self.scaffold_source,
        ):
            authority.close()

    def producer_hashes(self) -> dict[str, str]:
        self.legacy_source.replay()
        self.projection_source.replay()
        self.scaffold_source.replay()
        wrapper = StableFileAuthority.open(
            Path(__file__).resolve(), label="object oracle wrapper source"
        )
        try:
            wrapper_sha = wrapper.sha256
        finally:
            wrapper.close()
        return {
            "wrapper_source_sha256": wrapper_sha,
            "legacy_infer_lora_source_sha256": self.legacy_source.sha256,
            "projection_source_sha256": self.projection_source.sha256,
            "scaffold_source_sha256": self.scaffold_source.sha256,
        }

    def replay_all(self) -> None:
        for authority in (
            self.legacy_source,
            self.projection_source,
            self.scaffold_source,
            self.scaffold_file,
            self.aux_file,
        ):
            authority.replay()


def _prepare_oracle_assets(
    cli: OracleCLI,
    legacy_argv: Sequence[str],
    *,
    legacy_source: StableFileAuthority,
) -> OracleAssets:
    values = (
        cli.scaffold,
        cli.scaffold_sha256,
        cli.scaffold_digest,
        cli.bone_removed_video,
        cli.bone_removed_video_sha256,
    )
    if any(value is None for value in values):
        raise ObjectOracleWrapperError(
            "non-off object oracle arms require scaffold path/SHA/digest and "
            "bone-removed video path/SHA"
        )
    scaffold_digest = _strict_sha256(
        cli.scaffold_digest, label="object oracle scaffold digest"
    )
    if (
        cli.scaffold_sha256 != CASE01_SCAFFOLD_JSON_SHA256
        or scaffold_digest != CASE01_SCAFFOLD_ARTIFACT_DIGEST
        or cli.bone_removed_video_sha256 != CASE01_BONE_REMOVED_SHA256
    ):
        raise ObjectOracleWrapperError(
            "case01 oracle CLI authority differs from the frozen exact artifact"
        )
    projection_module, projection_source = _load_pinned_support(
        PROJECTION_BASENAME,
        PROJECTION_SHA256,
        expected_size=PROJECTION_SIZE,
        label="object trajectory projection source",
        module_name="_bernini_object_trajectory_projection_v1_pinned",
    )
    try:
        scaffold_module, scaffold_source = _load_pinned_support(
            SCAFFOLD_BASENAME,
            SCAFFOLD_SOURCE_SHA256,
            expected_size=SCAFFOLD_SOURCE_SIZE,
            label="case01 object trajectory scaffold source",
            module_name="_bernini_case01_object_trajectory_scaffold_v1_pinned",
        )
    except Exception:
        projection_source.close()
        raise
    scaffold_file: StableFileAuthority | None = None
    aux_file: StableFileAuthority | None = None
    try:
        scaffold_file = StableFileAuthority.open(
            cli.scaffold,
            label="object trajectory scaffold JSON",
            expected_sha256=cli.scaffold_sha256,
            expected_size=CASE01_SCAFFOLD_JSON_SIZE,
        )
        scaffold = _strict_json_object(
            scaffold_file.read_all(), label="object trajectory scaffold"
        )
        validator = getattr(scaffold_module, "_validate_artifact", None)
        object_digest = getattr(scaffold_module, "object_sha256", None)
        if not callable(validator) or not callable(object_digest):
            raise ObjectOracleWrapperError(
                "pinned scaffold module lacks _validate_artifact/object_sha256"
            )
        validator(scaffold)
        digest_payload = dict(scaffold)
        embedded_digest = digest_payload.pop("artifact_digest", None)
        observed_digest = object_digest(digest_payload)
        if (
            observed_digest != scaffold_digest
            or embedded_digest != scaffold_digest
        ):
            raise ObjectOracleWrapperError(
                "scaffold CLI, embedded, and recomputed artifact digests differ"
            )

        aux_file = StableFileAuthority.open(
            cli.bone_removed_video,
            label="auxiliary bone-removed source video",
            expected_sha256=cli.bone_removed_video_sha256,
            expected_size=CASE01_BONE_REMOVED_SIZE,
        )
        source_sha = _legacy_option(legacy_argv, "--source-video-sha256")
        if source_sha is None:
            raise ObjectOracleWrapperError(
                "non-off oracle arms require legacy --source-video-sha256"
            )
        _strict_sha256(source_sha, label="legacy source video SHA-256")
        authority = scaffold.get("authority")
        if not isinstance(authority, Mapping):
            raise ObjectOracleWrapperError("scaffold lacks authority")
        source_authority = authority.get("source_video")
        aux_authority = authority.get("bone_removed_auxiliary_video")
        if not isinstance(source_authority, Mapping) or not isinstance(
            aux_authority, Mapping
        ):
            raise ObjectOracleWrapperError(
                "scaffold lacks source/bone-removed authority"
            )
        if source_authority.get("sha256") != source_sha:
            raise ObjectOracleWrapperError(
                "scaffold exact-source SHA differs from legacy source authority"
            )
        if aux_authority.get("sha256") != aux_file.sha256:
            raise ObjectOracleWrapperError(
                "scaffold bone-removed SHA differs from retained aux authority"
            )
        if aux_authority.get("size") != aux_file.identity["size"]:
            raise ObjectOracleWrapperError(
                "scaffold bone-removed size differs from retained aux authority"
            )
        return OracleAssets(
            cli=cli,
            scaffold=scaffold,
            scaffold_file=scaffold_file,
            aux_file=aux_file,
            projection_module=projection_module,
            projection_source=projection_source,
            scaffold_module=scaffold_module,
            scaffold_source=scaffold_source,
            legacy_source=legacy_source,
        )
    except Exception:
        if scaffold_file is not None:
            scaffold_file.close()
        if aux_file is not None:
            aux_file.close()
        projection_source.close()
        scaffold_source.close()
        raise


def _strict_index_list(
    value: Any, *, upper: int, label: str
) -> tuple[int, ...]:
    if type(value) is not list or any(type(item) is not int for item in value):
        raise ObjectOracleWrapperError(f"{label} must be one integer list")
    result = tuple(value)
    if tuple(sorted(set(result))) != result:
        raise ObjectOracleWrapperError(f"{label} must be sorted and unique")
    if any(item < 0 or item >= upper for item in result):
        raise ObjectOracleWrapperError(f"{label} index is out of range")
    return result


def _validate_scaffold_geometry(scaffold: Mapping[str, Any]) -> None:
    layout = scaffold.get("latent_layout")
    if not isinstance(layout, Mapping):
        raise ObjectOracleWrapperError("scaffold lacks latent_layout")
    if (
        layout.get("latent_phases") != EXPECTED_TOKEN_SHAPE[0]
        or layout.get("patch_rows") != EXPECTED_TOKEN_SHAPE[1]
        or layout.get("patch_cols") != EXPECTED_TOKEN_SHAPE[2]
        or layout.get("tokens_per_phase") != EXPECTED_SPATIAL_TOKENS
        or layout.get("packed_token_count") != EXPECTED_SEGMENT_TOKENS
        or layout.get("attention_target_half_offset") != EXPECTED_SEGMENT_TOKENS
    ):
        raise ObjectOracleWrapperError("scaffold packed geometry differs")


def compile_scaffold_token_plan(scaffold: Mapping[str, Any]) -> dict[str, Any]:
    """Compile the validated JSON graph into tensor-free packed-token indices."""

    _validate_scaffold_geometry(scaffold)
    latent_rows = scaffold.get("latent_phases")
    if type(latent_rows) is not list or len(latent_rows) != EXPECTED_LATENT_PHASES:
        raise ObjectOracleWrapperError("scaffold must contain exactly 21 latent rows")
    phases: list[dict[str, Any]] = []
    all_pairs: list[list[int]] = []
    all_origin: set[int] = set()
    all_scaffold_origin: set[int] = set()
    all_target: set[int] = set()
    all_dog: set[int] = set()
    all_responsibility: set[int] = set()
    for expected_phase, row in enumerate(latent_rows):
        if not isinstance(row, Mapping) or row.get("phase_index") != expected_phase:
            raise ObjectOracleWrapperError("latent phase ordering differs")
        local_source = _strict_index_list(
            row.get("source_bone_tokens"),
            upper=EXPECTED_SPATIAL_TOKENS,
            label=f"latent[{expected_phase}].source_bone",
        )
        local_target = _strict_index_list(
            row.get("target_bone_tokens"),
            upper=EXPECTED_SPATIAL_TOKENS,
            label=f"latent[{expected_phase}].target_bone",
        )
        local_origin = _strict_index_list(
            row.get("origin_clear_tokens"),
            upper=EXPECTED_SPATIAL_TOKENS,
            label=f"latent[{expected_phase}].origin_clear",
        )
        local_dog = _strict_index_list(
            row.get("dog_identity_core_tokens"),
            upper=EXPECTED_SPATIAL_TOKENS,
            label=f"latent[{expected_phase}].dog_identity_core",
        )
        local_responsibility = _strict_index_list(
            row.get("target_responsibility_tokens"),
            upper=EXPECTED_SPATIAL_TOKENS,
            label=f"latent[{expected_phase}].target_responsibility",
        )
        if not set(local_target).issubset(local_responsibility):
            raise ObjectOracleWrapperError(
                "target bone support escapes the action responsibility tube"
            )
        offset = expected_phase * EXPECTED_SPATIAL_TOKENS
        source_global = {offset + index for index in local_source}
        target_global = {offset + index for index in local_target}
        origin_global = {offset + index for index in local_origin}
        dog_global = {offset + index for index in local_dog}
        responsibility_global = {
            offset + index for index in local_responsibility
        }
        raw_pairs = row.get("bone_token_correspondence")
        if type(raw_pairs) is not list:
            raise ObjectOracleWrapperError(
                f"latent[{expected_phase}] correspondence must be a list"
            )
        phase_pairs: list[list[int]] = []
        for pair in raw_pairs:
            if (
                type(pair) is not list
                or len(pair) != 2
                or any(type(item) is not int for item in pair)
            ):
                raise ObjectOracleWrapperError("bone correspondence row differs")
            source_local, target_local = pair
            if source_local not in local_source or target_local not in local_target:
                raise ObjectOracleWrapperError(
                    "bone correspondence escapes phase source/target support"
                )
            phase_pairs.append([offset + source_local, offset + target_local])
        if len(phase_pairs) != len({tuple(pair) for pair in phase_pairs}):
            raise ObjectOracleWrapperError("bone correspondence contains duplicates")
        if {pair[0] for pair in phase_pairs} != source_global:
            raise ObjectOracleWrapperError("bone correspondence source support differs")
        if {pair[1] for pair in phase_pairs} != target_global:
            raise ObjectOracleWrapperError("bone correspondence target support differs")
        if dog_global & (origin_global | target_global):
            raise ObjectOracleWrapperError(
                "dog identity core overlaps the patient projection"
            )
        effective_origin = origin_global - target_global
        phases.append(
            {
                "phase_index": expected_phase,
                "typed_stage": row.get("typed_stage"),
                "source": sorted(source_global),
                "target": sorted(target_global),
                "scaffold_origin": sorted(origin_global),
                "effective_origin": sorted(effective_origin),
                "dog_identity_core": sorted(dog_global),
                "responsibility": sorted(responsibility_global),
                "correspondence": phase_pairs,
            }
        )
        all_pairs.extend(phase_pairs)
        all_scaffold_origin.update(origin_global)
        all_origin.update(effective_origin)
        all_target.update(target_global)
        all_dog.update(dog_global)
        all_responsibility.update(responsibility_global)
    plan: dict[str, Any] = {
        "phases": phases,
        "bone_origin_clear_token_count": len(all_origin),
        "bone_scaffold_origin_support_token_count": len(all_scaffold_origin),
        "bone_target_tube_token_count": len(all_target),
        "bone_correspondence_count": len(all_pairs),
        "bone_correspondence_sha256": _object_sha256(all_pairs),
        "dog_core_token_count": len(all_dog),
        "responsibility_tube_token_count": len(all_responsibility),
        "overlapping_origin_target_policy": "target_source_bone_detail_wins",
    }
    plan["plan_digest"] = _object_sha256(plan)
    return plan


def _tensor_byte_authority(value: Any, *, label: str) -> dict[str, Any]:
    """Hash exact contiguous tensor bytes without dtype conversion."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - production has torch
        raise ObjectOracleWrapperError("tensor authority requires torch") from error
    if not isinstance(value, torch.Tensor) or value.requires_grad:
        raise ObjectOracleWrapperError(f"{label} must be one detached tensor")
    snapshot = value.detach().contiguous().cpu()
    if not bool(torch.isfinite(snapshot).all().item()):
        raise ObjectOracleWrapperError(f"{label} contains non-finite values")
    try:
        raw = snapshot.view(torch.uint8).numpy().tobytes(order="C")
    except Exception as error:
        raise ObjectOracleWrapperError(
            f"cannot expose exact bytes for {label}"
        ) from error
    expected_nbytes = int(snapshot.numel()) * int(snapshot.element_size())
    if len(raw) != expected_nbytes:
        raise ObjectOracleWrapperError(f"{label} byte length differs")
    return {
        "label": label,
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "device_type": value.device.type,
        "contiguous_before_snapshot": bool(value.is_contiguous()),
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _select_packed_tokens(value: Any, indices: Sequence[int], *, label: str) -> Any:
    import torch

    if not indices:
        raise ObjectOracleWrapperError(f"{label} token selection is empty")
    if tuple(sorted(set(indices))) != tuple(indices):
        raise ObjectOracleWrapperError(f"{label} token selection is not canonical")
    index = torch.tensor(indices, dtype=torch.int64, device=value.device)
    return value.index_select(1, index).detach().contiguous()


def _row_tensor_authority(
    *,
    plan: Mapping[str, Any],
    source_packed: Any,
    aux_packed: Any,
    legacy_phase0_clean: Any,
    bone_clean: Any,
    dog_clean: Any,
) -> dict[str, Any]:
    import torch

    phases = plan.get("phases")
    if not isinstance(phases, list):
        raise ObjectOracleWrapperError("token plan lacks phases")
    source_indices = sorted(
        {index for phase in phases for index in phase["source"]}
    )
    effective_origin = sorted(
        {index for phase in phases for index in phase["effective_origin"]}
    )
    target_indices = sorted(
        {index for phase in phases for index in phase["target"]}
    )
    dog_indices = sorted(
        {index for phase in phases for index in phase["dog_identity_core"]}
    )
    bone_selected = sorted(set(effective_origin) | set(target_indices))
    source_origin = _select_packed_tokens(
        source_packed, effective_origin, label="source effective origin"
    )
    aux_origin = _select_packed_tokens(
        aux_packed, effective_origin, label="aux effective origin"
    )
    differing = int(torch.count_nonzero(source_origin != aux_origin).item())
    if differing <= 0:
        raise ObjectOracleWrapperError(
            "source and bone-removed aux are identical on effective origin support"
        )
    tensors = {
        "source_packed_full": _tensor_byte_authority(
            source_packed, label="source_packed_full"
        ),
        "aux_packed_full": _tensor_byte_authority(
            aux_packed, label="aux_packed_full"
        ),
        "legacy_phase0_selected_clean": _tensor_byte_authority(
            _select_packed_tokens(
                legacy_phase0_clean,
                list(range(EXPECTED_SPATIAL_TOKENS)),
                label="legacy phase0 selected clean",
            ),
            label="legacy_phase0_selected_clean",
        ),
        "source_bone_correspondence_values": _tensor_byte_authority(
            _select_packed_tokens(
                source_packed, source_indices, label="source bone correspondence"
            ),
            label="source_bone_correspondence_values",
        ),
        "source_effective_origin_values": _tensor_byte_authority(
            source_origin, label="source_effective_origin_values"
        ),
        "aux_effective_origin_values": _tensor_byte_authority(
            aux_origin, label="aux_effective_origin_values"
        ),
        "constructed_bone_selected_clean": _tensor_byte_authority(
            _select_packed_tokens(
                bone_clean, bone_selected, label="constructed bone clean"
            ),
            label="constructed_bone_selected_clean",
        ),
        "constructed_dog_identity_clean": _tensor_byte_authority(
            _select_packed_tokens(
                dog_clean, dog_indices, label="constructed dog identity clean"
            ),
            label="constructed_dog_identity_clean",
        ),
    }
    contract = {
        "tensors": tensors,
        "effective_origin_element_count": int(source_origin.numel()),
        "source_aux_effective_origin_differing_element_count": differing,
        "source_aux_effective_origin_differ": True,
        "local_device": str(source_packed.device),
    }
    consensus_payload = dict(contract)
    consensus_payload.pop("local_device")
    contract["content_contract_digest"] = _object_sha256(consensus_payload)
    return contract


def _projection_row_specs(rows: Sequence[Any]) -> list[dict[str, Any]]:
    import torch

    specs: list[dict[str, Any]] = []
    for row in rows:
        weights = row.projection_weights
        if not isinstance(weights, torch.Tensor) or weights.ndim != 3:
            raise ObjectOracleWrapperError("projection row weights ABI differs")
        selected_tokens = int(
            weights.to(dtype=torch.bool).any(dim=2).count_nonzero().item()
        )
        gates = row.step_gates
        specs.append(
            {
                "name": row.name,
                "selected_token_count": selected_tokens,
                "weight_shape": [int(item) for item in weights.shape],
                "active_next_sigma_min": row.active_next_sigma_min,
                "active_next_sigma_max": row.active_next_sigma_max,
                "step_gates": list(gates) if gates is not None else None,
                "gate_policy": (
                    "all_steps_intersect_sigma_bounds"
                    if gates is None
                    else "explicit_steps_intersect_sigma_bounds"
                ),
            }
        )
    return specs


def _projection_contract(
    *,
    arm: str,
    expected_steps: int,
    row_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "arm": arm,
        "expected_steps": expected_steps,
        "row_names": row_evidence.get("row_names"),
        "row_specs": row_evidence.get("row_specs"),
        "token_plan_digest": row_evidence.get("plan_digest"),
        "tensor_content_contract_digest": row_evidence.get(
            "tensor_authority", {}
        ).get("content_contract_digest"),
    }
    for label, digest in (
        ("token plan digest", payload["token_plan_digest"]),
        ("tensor content contract digest", payload["tensor_content_contract_digest"]),
    ):
        _strict_sha256(digest, label=label)
    if (
        arm not in ACTIVE_ARMS
        or type(expected_steps) is not int
        or expected_steps <= 0
        or not isinstance(payload["row_names"], list)
        or not isinstance(payload["row_specs"], list)
        or [item.get("name") for item in payload["row_specs"]]
        != payload["row_names"]
    ):
        raise ObjectOracleWrapperError("projection contract fields differ")
    payload["projection_contract_digest"] = _object_sha256(payload)
    return payload


def _validate_four_rank_projection_digests(
    local_digest: str, gathered: Sequence[Any]
) -> None:
    _strict_sha256(local_digest, label="local projection contract digest")
    if list(gathered) != [local_digest] * 4:
        raise ObjectOracleWrapperError(
            "four-rank arm/row/gate/plan/tensor projection contracts differ"
        )


def _four_rank_projection_consensus(contract: Mapping[str, Any]) -> dict[str, Any]:
    import torch.distributed as dist

    world_size = int(dist.get_world_size())
    if world_size != 4:
        raise ObjectOracleWrapperError(
            f"object projection authority requires four ranks, got {world_size}"
        )
    digest = contract.get("projection_contract_digest")
    _strict_sha256(digest, label="projection contract digest")
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, digest)
    _validate_four_rank_projection_digests(digest, gathered)
    return {
        "world_size": world_size,
        "all_ranks_exact_projection_contract_equal": True,
        "ordered_projection_contract_digests": gathered,
    }


def _four_rank_stage_gate(
    *, stage: str, error: BaseException | None
) -> dict[str, Any]:
    import torch.distributed as dist

    if type(stage) is not str or not stage:
        raise ObjectOracleWrapperError("four-rank gate stage differs")
    world_size = int(dist.get_world_size())
    if world_size != 4:
        raise ObjectOracleWrapperError(
            f"object {stage} gate requires four ranks, got {world_size}"
        )
    status = {
        "stage": stage,
        "ok": error is None,
        "error_type": None if error is None else type(error).__name__,
        "error_text_sha256": (
            None
            if error is None
            else hashlib.sha256(str(error).encode("utf-8")).hexdigest()
        ),
    }
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, status)
    if any(not isinstance(item, Mapping) or item.get("ok") is not True for item in gathered):
        raise ObjectOracleWrapperError(
            f"one or more ranks failed {stage}: "
            f"status_digest={_object_sha256(gathered)}"
        )
    return {
        "stage": stage,
        "world_size": world_size,
        "all_ranks_reported_ok": True,
        "ordered_status_digest": _object_sha256(gathered),
    }


def build_projection_rows(
    *,
    arm: str,
    scaffold: Mapping[str, Any],
    source_packed: Any,
    aux_packed: Any,
    projection_module: types.ModuleType,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Build conservation rows from source detail and the oracle support graph."""

    if arm not in ACTIVE_ARMS:
        raise ObjectOracleWrapperError("projection rows require one active arm")
    try:
        import torch
    except ImportError as error:  # pragma: no cover - production has torch
        raise ObjectOracleWrapperError("projection rows require torch") from error
    if (
        not isinstance(source_packed, torch.Tensor)
        or not isinstance(aux_packed, torch.Tensor)
        or source_packed.ndim != 3
        or tuple(source_packed.shape) != tuple(aux_packed.shape)
        or tuple(source_packed.shape)[1:] != (
            EXPECTED_SEGMENT_TOKENS,
            PACKED_CHANNELS,
        )
        or source_packed.device != aux_packed.device
        or source_packed.dtype != aux_packed.dtype
        or source_packed.requires_grad
        or aux_packed.requires_grad
    ):
        raise ObjectOracleWrapperError(
            "oracle rows require matching detached packed [B,19530,64] latents"
        )
    plan = compile_scaffold_token_plan(scaffold)

    bone_clean = torch.zeros_like(source_packed).contiguous()
    bone_weights = torch.zeros(
        (int(source_packed.shape[0]), EXPECTED_SEGMENT_TOKENS, 1),
        dtype=torch.bool,
        device=source_packed.device,
    ).contiguous()
    dog_clean = torch.zeros_like(source_packed).contiguous()
    dog_weights = torch.zeros_like(bone_weights).contiguous()
    legacy_phase0_clean = source_packed.detach().clone().contiguous()
    legacy_phase0_weights = torch.zeros_like(bone_weights).contiguous()
    legacy_phase0_weights[:, :EXPECTED_SPATIAL_TOKENS, 0] = True
    for phase in plan["phases"]:
        for index in phase["effective_origin"]:
            bone_clean[:, index, :] = aux_packed[:, index, :]
            bone_weights[:, index, 0] = True
        for source_index, target_index in phase["correspondence"]:
            bone_clean[:, target_index, :] = source_packed[:, source_index, :]
            bone_weights[:, target_index, 0] = True
        for index in phase["dog_identity_core"]:
            dog_clean[:, index, :] = source_packed[:, index, :]
            dog_weights[:, index, 0] = True

    ProjectionRow = getattr(projection_module, "ProjectionRow", None)
    if ProjectionRow is None:
        raise ObjectOracleWrapperError("pinned projection module lacks ProjectionRow")
    rows: list[Any] = [
        ProjectionRow(
            name="legacy_phase0_hard1_every_step",
            clean_packed=legacy_phase0_clean,
            projection_weights=legacy_phase0_weights,
        ),
        ProjectionRow(
            name="bone_conservation_all_sigma",
            clean_packed=bone_clean,
            projection_weights=bone_weights,
        )
    ]
    if arm == "trajectory_dog_bone":
        rows.append(
            ProjectionRow(
                name="dog_core_low_mid",
                clean_packed=dog_clean,
                projection_weights=dog_weights,
                active_next_sigma_max=0.5,
            )
        )
    evidence = {
        "row_names": [row.name for row in rows],
        "row_specs": _projection_row_specs(rows),
        **{key: value for key, value in plan.items() if key != "phases"},
        "dog_row_consumed": arm == "trajectory_dog_bone",
        "origin_authority": "aux_bone_removed_source_packed",
        "target_bone_detail_authority": "same_source_bone_correspondence_scatter",
        "dog_detail_authority": "same_source_packed_dog_core",
        "single_instance_conservation_constructed": True,
        "matched_legacy_phase0_baseline": True,
        "legacy_phase0_selected_token_count": EXPECTED_SPATIAL_TOKENS,
        "legacy_phase0_sigma_gate": "all_steps_all_sigma",
    }
    evidence["tensor_authority"] = _row_tensor_authority(
        plan=plan,
        source_packed=source_packed,
        aux_packed=aux_packed,
        legacy_phase0_clean=legacy_phase0_clean,
        bone_clean=bone_clean,
        dog_clean=dog_clean,
    )
    return tuple(rows), evidence


@dataclass
class OracleExecutionState:
    legacy: types.ModuleType
    assets: OracleAssets
    aux_latent: Any = None
    pipeline_module: types.ModuleType | None = None
    original_vae_encode: Any = None
    vae_patch_installed: bool = False
    activate_calls: int = 0
    source_vae_encode_calls: int = 0
    aux_vae_encode_calls: int = 0
    aux_vae_encode_attempts: int = 0
    aux_encode_error: BaseException | None = None
    aux_broadcast_calls: int = 0
    aux_collective_gates: list[dict[str, Any]] = field(default_factory=list)
    projection_collective_gates: list[dict[str, Any]] = field(default_factory=list)
    projection_trace: dict[str, Any] | None = None
    row_evidence: dict[str, Any] | None = None

    def install_vae_patch(self) -> None:
        if self.vae_patch_installed:
            raise ObjectOracleWrapperError("VAE encode patch is already installed")
        pipeline = importlib.import_module("bernini.pipeline")
        original = getattr(pipeline, "_vae_encode", None)
        if not callable(original):
            raise ObjectOracleWrapperError("bernini.pipeline lacks callable _vae_encode")

        def wrapped(vae: Any, source_tensor: Any) -> Any:
            if self.source_vae_encode_calls != 0 or self.aux_vae_encode_calls != 0:
                raise ObjectOracleWrapperError(
                    "legacy rank-zero source VAE encode was not called exactly once"
                )
            source_latent = original(vae, source_tensor)
            self.source_vae_encode_calls += 1
            self.aux_vae_encode_attempts += 1
            try:
                proc_path = _linux_retained_fd_consumer_path(
                    self.assets.aux_file.descriptor
                )
                aux_tensor, aux_metadata = self.legacy.prepare_exact_source(proc_path)
                if tuple(aux_tensor.shape) != tuple(source_tensor.shape):
                    raise ObjectOracleWrapperError(
                        "bone-removed source preprocessing differs from exact source"
                    )
                if aux_metadata.get("source_derived_bucket_hw") != [
                    int(source_tensor.shape[-2]),
                    int(source_tensor.shape[-1]),
                ]:
                    raise ObjectOracleWrapperError("bone-removed source bucket differs")
                aux_latent = original(
                    vae,
                    aux_tensor.to(
                        device=source_tensor.device, dtype=source_tensor.dtype
                    ),
                )
                self.aux_vae_encode_calls += 1
                if (
                    tuple(aux_latent.shape) != tuple(source_latent.shape)
                    or aux_latent.device != source_latent.device
                    or aux_latent.dtype != source_latent.dtype
                ):
                    raise ObjectOracleWrapperError(
                        "bone-removed latent differs from source latent ABI"
                    )
                self.aux_latent = aux_latent
            except BaseException as error:
                # Do not strand ranks 1..3 in legacy's subsequent source-latent
                # broadcast.  Return the already valid source latent, then all
                # ranks synchronously reject the stored aux failure at clamp entry.
                self.aux_encode_error = error
                self.aux_latent = None
            return source_latent

        setattr(wrapped, "_bernini_case01_object_oracle_vae_encode_v1", True)
        pipeline._vae_encode = wrapped
        self.pipeline_module = pipeline
        self.original_vae_encode = original
        self.vae_patch_installed = True

    def restore_vae_patch(self) -> None:
        if not self.vae_patch_installed:
            return
        if self.pipeline_module is None:
            raise ObjectOracleWrapperError("installed VAE patch lost its pipeline")
        self.pipeline_module._vae_encode = self.original_vae_encode
        self.vae_patch_installed = False

    def distributed_aux(self, source_latent: Any) -> Any:
        import torch
        import torch.distributed as dist

        rank = int(dist.get_rank())
        local_error: BaseException | None = None
        aux: Any = None
        try:
            if rank == 0:
                if (
                    self.source_vae_encode_calls != 1
                    or self.aux_vae_encode_attempts != 1
                    or self.aux_vae_encode_calls != 1
                    or self.aux_encode_error is not None
                ):
                    raise self.aux_encode_error or ObjectOracleWrapperError(
                        "rank zero lacks exact one successful source plus aux VAE encode"
                    )
                aux = self.aux_latent
            else:
                if (
                    self.source_vae_encode_calls != 0
                    or self.aux_vae_encode_attempts != 0
                    or self.aux_vae_encode_calls != 0
                ):
                    raise ObjectOracleWrapperError(
                        "nonzero rank unexpectedly encoded VAE input"
                    )
                aux = torch.empty_like(source_latent)
            if (
                not isinstance(source_latent, torch.Tensor)
                or not isinstance(aux, torch.Tensor)
                or tuple(aux.shape) != tuple(source_latent.shape)
                or aux.dtype != source_latent.dtype
                or aux.device != source_latent.device
            ):
                raise ObjectOracleWrapperError(
                    "pre-broadcast aux latent ABI differs"
                )
        except BaseException as error:
            local_error = error
        self.aux_collective_gates.append(
            _four_rank_stage_gate(stage="aux_readiness", error=local_error)
        )
        if local_error is not None or not isinstance(aux, torch.Tensor):
            raise ObjectOracleWrapperError("local aux readiness failed")
        post_error: BaseException | None = None
        try:
            dist.broadcast(aux, src=0)
            self.aux_broadcast_calls += 1
            if (
                not isinstance(aux, torch.Tensor)
                or tuple(aux.shape) != tuple(source_latent.shape)
                or aux.dtype != source_latent.dtype
                or aux.device != source_latent.device
                or not bool(torch.isfinite(aux).all().item())
            ):
                raise ObjectOracleWrapperError(
                    "post-broadcast aux latent ABI differs"
                )
        except BaseException as error:
            post_error = error
        self.aux_collective_gates.append(
            _four_rank_stage_gate(stage="aux_post_broadcast", error=post_error)
        )
        if post_error is not None:
            raise ObjectOracleWrapperError("local post-broadcast aux ABI failed")
        return aux

    @contextmanager
    def clamp(
        self, diffusion: Any, source_latent: Any, *, expected_steps: int
    ) -> Iterator[Any]:
        local_runtime_error: BaseException | None = None
        scheduler: Any = None
        try:
            if getattr(diffusion, "use_unipc", None) is not True:
                raise ObjectOracleWrapperError(
                    "oracle projection requires use_unipc=True"
                )
            scheduler = getattr(diffusion, "scheduler", None)
            if scheduler is None:
                raise ObjectOracleWrapperError(
                    "oracle projection requires scheduler"
                )
        except BaseException as error:
            local_runtime_error = error
        runtime_gate = _four_rank_stage_gate(
            stage="projection_runtime_readiness", error=local_runtime_error
        )
        if local_runtime_error is not None or scheduler is None:
            raise ObjectOracleWrapperError("local projection runtime is unavailable")
        self.projection_collective_gates.append(runtime_gate)
        aux_latent = self.distributed_aux(source_latent)
        local_build_error: BaseException | None = None
        rows: tuple[Any, ...] | None = None
        row_evidence: dict[str, Any] | None = None
        try:
            source_packed = self.legacy._pack_wan_source_latent(source_latent)
            aux_packed = self.legacy._pack_wan_source_latent(aux_latent)
            rows, row_evidence = build_projection_rows(
                arm=self.assets.cli.arm,
                scaffold=self.assets.scaffold,
                source_packed=source_packed,
                aux_packed=aux_packed,
                projection_module=self.assets.projection_module,
            )
        except BaseException as error:
            local_build_error = error
        build_gate = _four_rank_stage_gate(
            stage="projection_row_build", error=local_build_error
        )
        if local_build_error is not None or rows is None or row_evidence is None:
            # The collective gate above raises on every rank.  This fallback
            # protects against a malformed collective implementation in tests.
            raise ObjectOracleWrapperError("local projection build failed")
        self.projection_collective_gates.append(build_gate)
        local_contract_error: BaseException | None = None
        contract: dict[str, Any] | None = None
        try:
            contract = _projection_contract(
                arm=self.assets.cli.arm,
                expected_steps=expected_steps,
                row_evidence=row_evidence,
            )
        except BaseException as error:
            local_contract_error = error
        contract_gate = _four_rank_stage_gate(
            stage="projection_contract_build", error=local_contract_error
        )
        if local_contract_error is not None or contract is None:
            raise ObjectOracleWrapperError("local projection contract build failed")
        self.projection_collective_gates.append(contract_gate)
        contract["four_rank_consensus"] = _four_rank_projection_consensus(contract)
        row_evidence["pre_projection_build_gate"] = build_gate
        row_evidence["pre_projection_contract_gate"] = contract_gate
        row_evidence["projection_contract"] = contract
        self.row_evidence = row_evidence
        local_lookup_error: BaseException | None = None
        projector: Any = None
        try:
            projector = getattr(
                self.assets.projection_module,
                "project_object_trajectory_unipc_steps",
                None,
            )
            if not callable(projector):
                raise ObjectOracleWrapperError(
                    "pinned projection context is unavailable"
                )
        except BaseException as error:
            local_lookup_error = error
        lookup_gate = _four_rank_stage_gate(
            stage="projection_projector_lookup", error=local_lookup_error
        )
        if local_lookup_error is not None or not callable(projector):
            raise ObjectOracleWrapperError("local projection projector lookup failed")
        self.projection_collective_gates.append(lookup_gate)
        row_evidence["projector_lookup_gate"] = lookup_gate
        outer = self

        class TraceFacade:
            def __init__(self) -> None:
                self.core_trace: Any = None

            def as_dict(self) -> dict[str, Any]:
                if self.core_trace is None:
                    raise ObjectOracleWrapperError(
                        "oracle projection never observed a native scheduler step"
                    )
                core = self.core_trace.as_dict()
                outer.assets.scaffold_file.replay()
                outer.assets.aux_file.replay()
                result = {
                    "schema_version": RUNTIME_TRACE_SCHEMA,
                    "arm": outer.assets.cli.arm,
                    "manual_oracle": True,
                    "zero_training": True,
                    "renderer_abi_integration": True,
                    "legacy_clamp_replaced": True,
                    "projection_installation": (
                        "lazy_at_first_native_step_after_runtime_schedule"
                    ),
                    "aux_latent_broadcast_from_rank0": True,
                    "aux_latent_broadcast_calls": outer.aux_broadcast_calls,
                    "vae_encode": {
                        "rank0_source_original_calls": outer.source_vae_encode_calls,
                        "rank0_aux_attempts": outer.aux_vae_encode_attempts,
                        "rank0_aux_original_calls": outer.aux_vae_encode_calls,
                    },
                    "aux_collective_gates": list(outer.aux_collective_gates),
                    "projection_collective_gates": list(
                        outer.projection_collective_gates
                    ),
                    "row_construction": dict(row_evidence),
                    "typed_action_program_scope": (
                        "patient_support_trajectory_and_dog_identity_exclusion_only"
                    ),
                    "approach_contact_dynamics_directly_enforced": False,
                    "new_action_signal_for_unprojected_dynamics": (
                        "legacy_edit_instruction_prompt"
                    ),
                    "authority": {
                        "scaffold": {
                            **outer.assets.scaffold_file.receipt(),
                            "artifact_digest": outer.assets.cli.scaffold_digest,
                        },
                        "aux_bone_removed_source": {
                            **outer.assets.aux_file.receipt(),
                            "consumed_via_retained_fd": True,
                        },
                        "embedded_authorities_digest": _object_sha256(
                            outer.assets.scaffold["authority"]
                        ),
                        "direct_runtime_authorities": [
                            "object_trajectory_scaffold",
                            "aux_bone_removed_source",
                        ],
                        "derived_scaffold_authorities": [
                            "stage0_object_masks",
                            "g0_mouth_track",
                        ],
                        "raw_stage0_or_g0_runtime_accessed": False,
                        "producer_hashes": outer.assets.producer_hashes(),
                    },
                    "tensor_core": core,
                    "target_video_accessed": False,
                    "learned_method_claim": False,
                }
                result["trace_digest"] = _object_sha256(result)
                outer.projection_trace = result
                return result
        facade = TraceFacade()
        had_instance_step = False
        old_instance_step: Any = None
        original_step: Any = None
        local_bootstrap_error: BaseException | None = None
        try:
            instance_dict = vars(scheduler)
            had_instance_step = "step" in instance_dict
            old_instance_step = instance_dict.get("step")
            original_step = getattr(scheduler, "step", None)
            if not callable(original_step):
                raise ObjectOracleWrapperError("scheduler.step must be callable")
            if getattr(
                original_step, "_bernini_case01_lazy_object_projection_v1", False
            ):
                raise ObjectOracleWrapperError(
                    "scheduler already has a lazy object projection"
                )
        except BaseException as error:
            local_bootstrap_error = error
        core_context: Any = None
        bootstrap_installed = False

        def restore_bootstrap() -> None:
            nonlocal bootstrap_installed
            if not bootstrap_installed:
                return
            if had_instance_step:
                setattr(scheduler, "step", old_instance_step)
            else:
                delattr(scheduler, "step")
            bootstrap_installed = False

        def first_step(*args: Any, **kwargs: Any) -> Any:
            nonlocal core_context
            if core_context is not None:
                raise ObjectOracleWrapperError(
                    "lazy projection bootstrap was called more than once"
                )
            # model.sample has now initialized scheduler.sigmas/timesteps.
            # Remove this bootstrap before the pinned core audits and wraps the
            # exact original scheduler.step instance.
            candidate: Any = None
            core_trace: Any = None
            local_install_error: BaseException | None = None
            try:
                restore_bootstrap()
                candidate = projector(
                    scheduler,
                    rows=rows,
                    initial_noise=None,
                    source_token_count=EXPECTED_SEGMENT_TOKENS,
                    target_token_count=EXPECTED_SEGMENT_TOKENS,
                    expected_steps=expected_steps,
                )
                core_trace = candidate.__enter__()
                if core_trace is None:
                    raise ObjectOracleWrapperError(
                        "projection context returned no trace"
                    )
            except BaseException as error:
                local_install_error = error
            try:
                install_gate = _four_rank_stage_gate(
                    stage="projection_projector_install",
                    error=local_install_error,
                )
            except BaseException:
                if local_install_error is None and candidate is not None:
                    peer_error = ObjectOracleWrapperError(
                        "peer rank rejected projection projector installation"
                    )
                    try:
                        candidate.__exit__(
                            type(peer_error), peer_error, peer_error.__traceback__
                        )
                    except BaseException:
                        pass
                raise
            if (
                local_install_error is not None
                or candidate is None
                or core_trace is None
            ):
                raise ObjectOracleWrapperError(
                    "local projection projector installation failed"
                )
            outer.projection_collective_gates.append(install_gate)
            row_evidence["projector_install_gate"] = install_gate
            core_context = candidate
            facade.core_trace = core_trace
            return getattr(scheduler, "step")(*args, **kwargs)

        if local_bootstrap_error is None:
            try:
                setattr(first_step, "_bernini_case01_lazy_object_projection_v1", True)
                setattr(scheduler, "step", first_step)
                bootstrap_installed = True
            except BaseException as error:
                local_bootstrap_error = error
        try:
            bootstrap_gate = _four_rank_stage_gate(
                stage="projection_lazy_bootstrap_install",
                error=local_bootstrap_error,
            )
        except BaseException:
            if local_bootstrap_error is None:
                try:
                    restore_bootstrap()
                except BaseException:
                    pass
            raise
        if local_bootstrap_error is not None or not bootstrap_installed:
            raise ObjectOracleWrapperError(
                "local projection lazy bootstrap installation failed"
            )
        self.projection_collective_gates.append(bootstrap_gate)
        row_evidence["lazy_bootstrap_install_gate"] = bootstrap_gate
        error_info: tuple[Any, Any, Any] = (None, None, None)
        completed_body = False
        try:
            yield facade
            completed_body = True
        except BaseException:
            error_info = sys.exc_info()
            raise
        finally:
            local_final_error: BaseException | None = None
            if core_context is None:
                restore_bootstrap()
                if completed_body:
                    local_final_error = ObjectOracleWrapperError(
                        "model.sample completed without a native scheduler step"
                    )
            else:
                try:
                    core_context.__exit__(*error_info)
                except BaseException as error:
                    local_final_error = error
            if completed_body:
                final_gate = _four_rank_stage_gate(
                    stage="projection_final_validation",
                    error=local_final_error,
                )
                if local_final_error is not None:
                    raise ObjectOracleWrapperError(
                        "local projection final validation failed"
                    )
                self.projection_collective_gates.append(final_gate)
                row_evidence["final_validation_gate"] = final_gate
            elif local_final_error is not None:
                raise local_final_error


def _customize_receipt(
    receipt: Mapping[str, Any], *, state: OracleExecutionState | None, assets: OracleAssets
) -> dict[str, Any]:
    value = dict(receipt)
    value.pop("receipt_digest", None)
    input_receipt = dict(value.get("input", {}))
    input_receipt.update(
        {
            "accepted_model_conditions": [
                "source_video",
                "edit_instruction",
                "stage0_object_masks",
                "g0_mouth_track",
                "object_trajectory_scaffold",
                "aux_bone_removed_source",
            ],
            "direct_runtime_conditions": [
                "source_video",
                "edit_instruction",
                "object_trajectory_scaffold",
                "aux_bone_removed_source",
            ],
            "derived_scaffold_authorities": [
                "stage0_object_masks",
                "g0_mouth_track",
            ],
            "raw_stage0_masks_accessed_at_runtime": False,
            "raw_g0_annotations_accessed_at_runtime": False,
            "target_video_argument": False,
            "target_accessed_by_inference": False,
            "external_mask_or_swept_tube": True,
            "external_tracking_pose_or_trajectory": True,
            "reference_image_or_video": True,
            "external_shared_i0": False,
        }
    )
    value["input"] = input_receipt
    value["schema_version"] = WRAPPER_RECEIPT_SCHEMA
    legacy_sampling = value.get("sampling", {})
    if not isinstance(legacy_sampling, Mapping):
        raise ObjectOracleWrapperError("legacy receipt sampling field differs")
    legacy_source_onset_policy = legacy_sampling.get("source_onset_policy")
    legacy_solver_trace_present = "source_onset_solver_trace" in legacy_sampling
    active = assets.cli.arm in ACTIVE_ARMS
    if active:
        if state is None or state.projection_trace is None:
            raise ObjectOracleWrapperError("active receipt lacks finalized projection trace")
        sampling_trace = value.get("sampling", {}).get("source_onset_solver_trace")
        if sampling_trace != state.projection_trace:
            raise ObjectOracleWrapperError(
                "legacy sampling receipt and active oracle trace differ"
            )
        sampling = dict(value.get("sampling", {}))
        sampling["legacy_dispatch_source_onset_policy"] = sampling.get(
            "source_onset_policy"
        )
        sampling["source_onset_policy"] = "case01_object_trajectory_oracle_v3"
        value["sampling"] = sampling
        status = "consumed_projection"
        runtime: dict[str, Any] = {
            "object_oracle_renderer_or_scheduler_patched": True,
            "receipt_builder_augmented": True,
            "aux_bytes_consumed_by_renderer": True,
            "legacy_dispatch_source_onset_policy": legacy_source_onset_policy,
            "legacy_source_onset_solver_trace_present": legacy_solver_trace_present,
            "vae_encode": {
                "rank0_source_original_calls": state.source_vae_encode_calls,
                "rank0_aux_attempts": state.aux_vae_encode_attempts,
                "rank0_aux_original_calls": state.aux_vae_encode_calls,
            },
            "aux_collective_gates": list(state.aux_collective_gates),
            "projection_collective_gates": list(
                state.projection_collective_gates
            ),
            "aux_latent_broadcast_calls": state.aux_broadcast_calls,
            "projection_trace": dict(state.projection_trace),
            "direct_runtime_conditions_consumed": [
                "source_video",
                "edit_instruction",
                "object_trajectory_scaffold",
                "aux_bone_removed_source",
            ],
            "oracle_runtime_conditions_consumed": [
                "object_trajectory_scaffold",
                "aux_bone_removed_source",
            ],
            "derived_scaffold_authorities_consumed_directly": [],
        }
    else:
        status = "validated_not_consumed"
        runtime = {
            "object_oracle_renderer_or_scheduler_patched": False,
            "receipt_builder_augmented": True,
            "aux_bytes_consumed_by_renderer": False,
            "legacy_dispatch_source_onset_policy": legacy_source_onset_policy,
            "legacy_source_onset_solver_trace_present": legacy_solver_trace_present,
            "vae_encode": {
                "rank0_source_original_calls": 0,
                "rank0_aux_attempts": 0,
                "rank0_aux_original_calls": 0,
            },
            "aux_collective_gates": [],
            "projection_collective_gates": [],
            "aux_latent_broadcast_calls": 0,
            "projection_trace": None,
            "direct_runtime_conditions_consumed": [
                "source_video",
                "edit_instruction",
            ],
            "oracle_runtime_conditions_consumed": [],
            "derived_scaffold_authorities_consumed_directly": [],
        }
    assets.scaffold_file.replay()
    assets.aux_file.replay()
    value["object_oracle"] = {
        "schema_version": RUNTIME_TRACE_SCHEMA,
        "arm": assets.cli.arm,
        "status": status,
        "manual_oracle": True,
        "zero_training": True,
        "production_method_claim": False,
        "assets": {
            "scaffold": {
                **assets.scaffold_file.receipt(),
                "artifact_digest": assets.cli.scaffold_digest,
            },
            "aux_bone_removed_source": {
                **assets.aux_file.receipt(),
                "consumed_via_retained_fd": active,
            },
            "embedded_authorities": assets.scaffold["authority"],
            "embedded_authorities_digest": _object_sha256(
                assets.scaffold["authority"]
            ),
            "direct_runtime_authorities": {
                "object_trajectory_scaffold": {
                    "sha256": assets.scaffold_file.sha256,
                    "artifact_digest": assets.cli.scaffold_digest,
                },
                "aux_bone_removed_source": {
                    "sha256": assets.aux_file.sha256,
                    "consumed_by_renderer": active,
                },
            },
            "derived_scaffold_authorities": {
                "stage0_object_masks": assets.scaffold["authority"][
                    "stage0_receipt"
                ],
                "g0_mouth_track": assets.scaffold["authority"][
                    "g0_sparse_annotations"
                ],
                "raw_files_opened_at_runtime": False,
            },
        },
        "producer_hashes": assets.producer_hashes(),
        "runtime": runtime,
    }
    value["receipt_digest"] = _object_sha256(value)
    return value


@contextmanager
def _patched_legacy(
    legacy: types.ModuleType, assets: OracleAssets
) -> Iterator[OracleExecutionState | None]:
    original_receipt = legacy.build_inference_receipt
    original_activate = None
    original_clamp = None
    state: OracleExecutionState | None = None
    active = assets.cli.arm in ACTIVE_ARMS
    if active:
        state = OracleExecutionState(legacy=legacy, assets=assets)
        original_activate = legacy.trainer.activate_source_trees
        original_clamp = legacy.hard_phase0_source_trajectory_clamp

        def activate(*args: Any, **kwargs: Any) -> Any:
            if state is None:  # pragma: no cover - guarded by active above
                raise ObjectOracleWrapperError("active execution state disappeared")
            state.activate_calls += 1
            if state.activate_calls != 1:
                raise ObjectOracleWrapperError(
                    "legacy activate_source_trees must run exactly once"
                )
            result = original_activate(*args, **kwargs)
            state.install_vae_patch()
            return result

        legacy.trainer.activate_source_trees = activate
        legacy.hard_phase0_source_trajectory_clamp = state.clamp

    def receipt_wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        receipt = original_receipt(*args, **kwargs)
        return _customize_receipt(receipt, state=state, assets=assets)

    legacy.build_inference_receipt = receipt_wrapper
    try:
        yield state
    finally:
        legacy.build_inference_receipt = original_receipt
        if active:
            if state is None:  # pragma: no cover - guarded by active above
                raise ObjectOracleWrapperError("active execution state disappeared")
            state.restore_vae_patch()
            legacy.trainer.activate_source_trees = original_activate
            legacy.hard_phase0_source_trajectory_clamp = original_clamp


def main(argv: Optional[Sequence[str]] = None) -> int:
    cli, legacy_argv = peel_object_oracle_cli(argv)
    legacy, legacy_source = _load_frozen_legacy()
    try:
        # Strong null: no oracle file/module is opened and no legacy object is
        # patched.  Passing unused oracle CLI strings does not weaken this
        # property because they are peeled but never resolved or inspected.
        if cli.arm == "off":
            result = legacy.main(legacy_argv)
            legacy_source.replay()
            return result
        assets = _prepare_oracle_assets(
            cli, legacy_argv, legacy_source=legacy_source
        )
        try:
            if cli.arm in ACTIVE_ARMS:
                policy = _legacy_option(legacy_argv, "--source-onset-policy")
                if policy != "hard1_every_step":
                    raise ObjectOracleWrapperError(
                        "active oracle arms require legacy "
                        "--source-onset-policy hard1_every_step"
                    )
            with _patched_legacy(legacy, assets):
                result = legacy.main(legacy_argv)
            assets.replay_all()
            return result
        finally:
            assets.close()
    finally:
        legacy_source.close()


if __name__ == "__main__":
    raise SystemExit(main())
