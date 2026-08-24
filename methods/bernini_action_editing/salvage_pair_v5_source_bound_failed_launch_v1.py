#!/usr/bin/env python3
"""Receipt-only salvage for the completed AUH 131222 source-bound run.

Job 131222 completed all eight frozen-DINO candidate evaluations and wrote two
group receipts plus the root receipt.  Its final durable reread failed because
the old validator expected rollout order for JSON object members even though
canonical JSON necessarily reloaded those members in lexical order.  This
tool reopens the existing JSON evidence with the narrowly patched validator.

It never decodes video, loads DINO, recomputes a metric, edits an old artifact,
or authorizes an optimizer step.  The only new artifact is a create-only JSON
receipt authenticating execution/raw-evidence closure under the map-order fix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
from typing import Any, Mapping

import pair_v5_source_bound_preservation_evaluator_v1 as contract


SALVAGE_SCHEMA = "bernini-pair-v5-source-bound-failed-launch-salvage-v1"
LEGACY_ROOT_NAME = "pair-v5-source-bound-preservation-root-v1.json"
LEGACY_GROUP_NAME = "pair-v5-source-bound-preservation-{group_id}-v1.json"
LEGACY_IMPLEMENTATION_MEMBER = (
    "methods/bernini_action_editing/score_pair_v5_source_bound_preservation_v1.py"
)
LEGACY_CONTRACT_MEMBER = (
    "methods/bernini_action_editing/pair_v5_source_bound_preservation_evaluator_v1.py"
)
SALVAGE_TOOL_MEMBER = (
    "methods/bernini_action_editing/salvage_pair_v5_source_bound_failed_launch_v1.py"
)
SALVAGE_VALIDATOR_MEMBER = LEGACY_CONTRACT_MEMBER

PATCHED_VALIDATOR_SOURCE_REVISION = "e836ac2c012b17299e25b6155db4fb7694f9304c"
PATCHED_VALIDATOR_SOURCE_SHA256 = (
    "c390d17b680b3d57048d04257fd5f89f98a7b421b79720f68eb837d992f392d4"
)

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class SourceBoundSalvageError(RuntimeError):
    """The sealed legacy evidence cannot be salvaged without recomputation."""


@dataclass(frozen=True)
class LegacySeal:
    slurm_job_id: int
    evaluator_spec_raw_sha256: str
    evaluator_spec_digest: str
    method_source_revision: str
    method_source_archive_sha256: str
    legacy_implementation_sha256: str
    legacy_contract_sha256: str
    legacy_root_file_sha256: str
    legacy_root_digest: str
    candidate_order: tuple[str, ...]
    group_receipt_digest_by_id: tuple[tuple[str, str], ...]
    group_receipt_file_sha256_by_id: tuple[tuple[str, str], ...]
    candidate_receipt_digest_by_id: tuple[tuple[str, str], ...]
    candidate_receipt_file_sha256_by_id: tuple[tuple[str, str], ...]
    patched_validator_source_revision: str
    patched_validator_source_sha256: str


@dataclass(frozen=True)
class SalvageSourceSeal:
    source_revision: str
    source_archive_sha256: str
    tool_member_sha256: str
    validator_member_sha256: str


REGISTERED_AUH_131222 = LegacySeal(
    slurm_job_id=131222,
    evaluator_spec_raw_sha256=(
        "6b18b9bc10589325ee2c09af339ef43a3eff507bcc754a2a6984cb70f0afd736"
    ),
    evaluator_spec_digest=(
        "94f097fcb1f012774a7fb63715a1d7c6dbd2ad8028d0276c9f838985d0560321"
    ),
    method_source_revision="7c4c837b946cc372537c7d98cffbbc6c3f38fa9f",
    method_source_archive_sha256=(
        "922cf05483e58faad8fbc7d244fb95ba94d0eed649f7129337e9d1bea70223b4"
    ),
    legacy_implementation_sha256=(
        "9e86ee8128841f624db92b99914235a37fee4d7b92aeda2e62104ab57e531b39"
    ),
    legacy_contract_sha256=(
        "183eaafaebef426f888aa3abe91632a884f827d39ae16db576d57da401a8533a"
    ),
    legacy_root_file_sha256=(
        "7ed6e4aace411415006dba555dd9aee2b77238efe229a15ce392e00030741de9"
    ),
    legacy_root_digest=(
        "a9730410e7a1a3c96d3caec92de39961310c378043502e8f82412d642387ef72"
    ),
    candidate_order=(
        "pair5-native-core4-v1-7b88a1ca1f804f41-action-s2026080901",
        "pair5-native-core4-v1-7b88a1ca1f804f41-action-s2026080902",
        "pair5-native-core4-v1-a66e6818e4144928-action-s2026080901",
        "pair5-native-core4-v1-a66e6818e4144928-action-s2026080902",
        "pair5-native-core4-v1-841b5e0080a1441d-action-s2026080901",
        "pair5-native-core4-v1-841b5e0080a1441d-action-s2026080902",
        "pair5-native-core4-v1-a35b590961d24694-action-s2026080901",
        "pair5-native-core4-v1-a35b590961d24694-action-s2026080902",
    ),
    group_receipt_digest_by_id=(
        ("sp4-a", "e40a42d0f5a88062a8176c61f7f516f8d95c030db7ef63c16b2e190aefa90ac6"),
        ("sp4-b", "472077c1cb34512f83f07b9eeb5a2b365733b49a3507d67a2bd9eb136a6b94fb"),
    ),
    group_receipt_file_sha256_by_id=(
        ("sp4-a", "b46707bea4efade4fc1625e5d2e8796d1c82bdd5970057fd9290207e02849628"),
        ("sp4-b", "84e6120fa1f847fac66c31566de0400efe2c3225c8e99812830318763b93e786"),
    ),
    candidate_receipt_digest_by_id=(
        ("pair5-native-core4-v1-7b88a1ca1f804f41-action-s2026080901", "838473593b24a858181cb6119ddcdc625d7b95bd0b891925c096843b3e78af42"),
        ("pair5-native-core4-v1-7b88a1ca1f804f41-action-s2026080902", "45a28e37a641ba8cbfce81bceb1dc3025adeeed1df67adbd9246c5c554b2a1ac"),
        ("pair5-native-core4-v1-841b5e0080a1441d-action-s2026080901", "41111478dfed6578d4e13e08fded0e2b4109cf2a3d321be3aad42ba18b1b142e"),
        ("pair5-native-core4-v1-841b5e0080a1441d-action-s2026080902", "0299360f9cc87f0dc82ef9900da6378ff1b0073515303ed988cda5dd28d4d054"),
        ("pair5-native-core4-v1-a35b590961d24694-action-s2026080901", "e0f026f3c4d18238baf6e07b2959453dfb2309eb30486151ddf2f0e3d909704f"),
        ("pair5-native-core4-v1-a35b590961d24694-action-s2026080902", "e7943432ea009a28d5ec5e83d7cd3f16da553452229c1628f566ecb5b5c1fd44"),
        ("pair5-native-core4-v1-a66e6818e4144928-action-s2026080901", "5c3c04e7b4f588b51f52cf4ab194c7520a56c4ab115b9096ed9e167931e33cf9"),
        ("pair5-native-core4-v1-a66e6818e4144928-action-s2026080902", "0246a18d2a9b36b0d2fbaf1a6078639cca0f13f12fc43cb225cf3edfd0f31985"),
    ),
    candidate_receipt_file_sha256_by_id=(
        ("pair5-native-core4-v1-7b88a1ca1f804f41-action-s2026080901", "ad90abbdbc276d998d54c41cf3e7f493df9a0eb4c9ecb770e9b8c86607efbb63"),
        ("pair5-native-core4-v1-7b88a1ca1f804f41-action-s2026080902", "5eeeaae8a5d861b3cc71718d67c4a140f314c158e1ddfb1ec833eac27656bcbe"),
        ("pair5-native-core4-v1-841b5e0080a1441d-action-s2026080901", "ef6a93049db268006274a25b8ecf449974758be9fc58b15c160e561fa7c851ae"),
        ("pair5-native-core4-v1-841b5e0080a1441d-action-s2026080902", "d0f137cb0064f8503aeb813eb27f6b58b4b79155b45833e8b88140f57e415188"),
        ("pair5-native-core4-v1-a35b590961d24694-action-s2026080901", "7e3cc36156417f5dda4a8ed4015d620585890486d76559d4888076c64a84820c"),
        ("pair5-native-core4-v1-a35b590961d24694-action-s2026080902", "33b18ffdeed9ebe2b40dc59044e914e5e5a412c74d076d66d70d3fea752d7845"),
        ("pair5-native-core4-v1-a66e6818e4144928-action-s2026080901", "87aded2346e6481b0fb3f1a60658c4248069691660459e3bc631f20f8f637863"),
        ("pair5-native-core4-v1-a66e6818e4144928-action-s2026080902", "4d21f4b81cbc9535eeeccb67d49b32150e6840e55a919879d4d5e5414ec24e19"),
    ),
    patched_validator_source_revision=PATCHED_VALIDATOR_SOURCE_REVISION,
    patched_validator_source_sha256=PATCHED_VALIDATOR_SOURCE_SHA256,
)


AUTHORITY_CLOSURE = {
    "execution_and_raw_evidence_closure_authenticated": True,
    "calibration_input_eligibility_revalidated": True,
    "dino_recomputed": False,
    "video_redecoded": False,
    "features_recomputed": False,
    "metrics_recomputed": False,
    "old_artifacts_modified": False,
    "new_model_or_media_artifacts": 0,
    "absolute_source_preservation_pass": False,
    "absolute_source_preservation_pass_claims": 0,
    "action_editing_success_authorized": False,
    "candidate_selection_authorized": False,
    "optimizer_go": False,
    "training_authorized": False,
    "training_performed": False,
}

_SALVAGE_FIELDS = frozenset(
    {
        "schema_version",
        "legacy_slurm_job_id",
        "salvage_scope",
        "legacy_evaluator",
        "patched_validator",
        "salvage_source",
        "legacy_root",
        "group_order",
        "group_receipt_digest_by_id",
        "group_receipt_file_sha256_by_id",
        "candidate_order",
        "candidate_receipt_digest_by_id",
        "candidate_receipt_file_sha256_by_id",
        "artifact_reopen_counts",
        "closure_checks",
        "authority_closure",
        "salvage_digest",
    }
)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_file(path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise SourceBoundSalvageError(f"{label} cannot be stat'ed") from error
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise SourceBoundSalvageError(f"{label} is not a plain file")
    return candidate.resolve(strict=True)


def _stable_read(path: str | Path, *, label: str) -> tuple[Path, bytes, str]:
    resolved = _plain_file(path, label=label)
    before = resolved.stat()
    raw = resolved.read_bytes()
    after = resolved.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(raw) != after.st_size:
        raise SourceBoundSalvageError(f"{label} changed while being read")
    return resolved, raw, _sha256_bytes(raw)


def _canonical_json(
    path: str | Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    _, raw, digest = _stable_read(path, label=label)
    if expected_sha256 is not None and digest != expected_sha256:
        raise SourceBoundSalvageError(f"{label} raw SHA-256 differs")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceBoundSalvageError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise SourceBoundSalvageError(f"{label} root is not an object")
    if raw != contract.canonical_json_bytes(value) + b"\n":
        raise SourceBoundSalvageError(f"{label} is not canonical JSON plus newline")
    return value, digest


def _archive_member_bytes(handle: tarfile.TarFile, name: str) -> bytes:
    matches = [member for member in handle.getmembers() if member.name == name]
    if len(matches) != 1 or not matches[0].isfile() or matches[0].issym() or matches[0].islnk():
        raise SourceBoundSalvageError(f"legacy archive member closure differs: {name}")
    extracted = handle.extractfile(matches[0])
    if extracted is None:
        raise SourceBoundSalvageError(f"legacy archive member cannot be read: {name}")
    return extracted.read()


def _audit_archive_members(handle: tarfile.TarFile, *, label: str) -> None:
    seen: set[str] = set()
    for member in handle.getmembers():
        pure = PurePosixPath(member.name)
        if (
            not member.name
            or pure.is_absolute()
            or ".." in pure.parts
            or member.issym()
            or member.islnk()
            or member.isdev()
            or member.isfifo()
        ):
            raise SourceBoundSalvageError(f"{label} contains an unsafe member")
        normalized = pure.as_posix().rstrip("/")
        if normalized in seen:
            raise SourceBoundSalvageError(f"{label} contains duplicate members")
        seen.add(normalized)


def _verify_legacy_archive(
    path: str | Path,
    *,
    seal: LegacySeal,
    spec: Mapping[str, Any],
) -> None:
    archive, _, archive_sha = _stable_read(path, label="legacy method archive")
    if archive_sha != seal.method_source_archive_sha256:
        raise SourceBoundSalvageError("legacy method archive raw SHA-256 differs")
    try:
        with tarfile.open(archive, "r:*") as handle:
            archive_revision = handle.pax_headers.get("comment")
            if archive_revision != seal.method_source_revision:
                raise SourceBoundSalvageError("legacy git archive revision differs")
            _audit_archive_members(handle, label="legacy method archive")
            implementation = _archive_member_bytes(handle, LEGACY_IMPLEMENTATION_MEMBER)
            legacy_contract = _archive_member_bytes(handle, LEGACY_CONTRACT_MEMBER)
    except (tarfile.TarError, OSError) as error:
        raise SourceBoundSalvageError("legacy method archive cannot be audited") from error
    if _sha256_bytes(implementation) != spec["implementation_sha256"]:
        raise SourceBoundSalvageError("legacy evaluator implementation/archive binding differs")
    if _sha256_bytes(legacy_contract) != spec["contract_sha256"]:
        raise SourceBoundSalvageError("legacy evaluator contract/archive binding differs")


def verify_salvage_source_archive(
    path: str | Path,
    *,
    expected_source_revision: str,
    expected_source_archive_sha256: str,
    executed_tool_path: str | Path,
    imported_validator_path: str | Path,
) -> SalvageSourceSeal:
    """Bind the running Python files to an immutable, safely audited git tar."""

    if _SHA1.fullmatch(expected_source_revision) is None:
        raise SourceBoundSalvageError("salvage source revision differs")
    if _SHA256.fullmatch(expected_source_archive_sha256) is None:
        raise SourceBoundSalvageError("salvage source archive SHA-256 differs")
    archive, _, archive_sha = _stable_read(path, label="salvage source archive")
    if archive_sha != expected_source_archive_sha256:
        raise SourceBoundSalvageError("salvage source archive raw SHA-256 differs")
    try:
        with tarfile.open(archive, "r:*") as handle:
            if handle.pax_headers.get("comment") != expected_source_revision:
                raise SourceBoundSalvageError("salvage git archive revision differs")
            _audit_archive_members(handle, label="salvage source archive")
            tool_member = _archive_member_bytes(handle, SALVAGE_TOOL_MEMBER)
            validator_member = _archive_member_bytes(handle, SALVAGE_VALIDATOR_MEMBER)
    except (tarfile.TarError, OSError) as error:
        raise SourceBoundSalvageError("salvage source archive cannot be audited") from error
    tool = _plain_file(executed_tool_path, label="executed salvage tool")
    validator = _plain_file(imported_validator_path, label="imported patched validator")
    if tool != Path(__file__).resolve(strict=True):
        raise SourceBoundSalvageError("verified salvage tool is not the executing module")
    if validator != Path(contract.__file__).resolve(strict=True):
        raise SourceBoundSalvageError("verified validator is not the imported module")
    tool_sha = _sha256_bytes(tool_member)
    validator_sha = _sha256_bytes(validator_member)
    if _sha256_file(tool) != tool_sha:
        raise SourceBoundSalvageError("archive salvage tool/executed bytes differ")
    if _sha256_file(validator) != validator_sha:
        raise SourceBoundSalvageError("archive validator/imported bytes differ")
    if validator_sha != PATCHED_VALIDATOR_SOURCE_SHA256:
        raise SourceBoundSalvageError("salvage archive does not contain the pinned validator")
    return SalvageSourceSeal(
        source_revision=expected_source_revision,
        source_archive_sha256=archive_sha,
        tool_member_sha256=tool_sha,
        validator_member_sha256=validator_sha,
    )


def _verify_patched_validator(path: str | Path, *, seal: LegacySeal) -> Path:
    validator, _, digest = _stable_read(path, label="patched validator source")
    imported = Path(contract.__file__).resolve(strict=True)
    if validator != imported:
        raise SourceBoundSalvageError("verified patched validator is not the imported module")
    if (
        seal.patched_validator_source_revision != PATCHED_VALIDATOR_SOURCE_REVISION
        or seal.patched_validator_source_sha256 != PATCHED_VALIDATOR_SOURCE_SHA256
        or digest != seal.patched_validator_source_sha256
    ):
        raise SourceBoundSalvageError("patched validator revision/source binding differs")
    return validator


def _closed(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SourceBoundSalvageError(f"{label} key closure differs")
    return value


def validate_salvage_receipt(
    value: Any,
    *,
    salvage_source_seal: SalvageSourceSeal,
    seal: LegacySeal = REGISTERED_AUH_131222,
) -> dict[str, Any]:
    row = dict(_closed(value, _SALVAGE_FIELDS, label="salvage receipt"))
    if row["schema_version"] != SALVAGE_SCHEMA or row["legacy_slurm_job_id"] != seal.slurm_job_id:
        raise SourceBoundSalvageError("salvage schema/job binding differs")
    expected_scope = {
        "kind": "receipt_only_revalidation_of_existing_raw_evidence",
        "legacy_failure_stage": "durable_root_reread_map_order_validator",
        "legacy_failure_was_model_or_metric_compute": False,
        "validator_patch_changes_evidence_values": False,
    }
    if row["salvage_scope"] != expected_scope:
        raise SourceBoundSalvageError("salvage scope differs")
    legacy = _closed(
        row["legacy_evaluator"],
        frozenset(
            {
                "evaluator_spec_raw_sha256",
                "evaluator_spec_digest",
                "method_source_revision",
                "method_source_archive_sha256",
                "implementation_sha256",
                "contract_sha256",
            }
        ),
        label="legacy evaluator binding",
    )
    for field, expected in (
        ("evaluator_spec_raw_sha256", seal.evaluator_spec_raw_sha256),
        ("evaluator_spec_digest", seal.evaluator_spec_digest),
        ("method_source_revision", seal.method_source_revision),
        ("method_source_archive_sha256", seal.method_source_archive_sha256),
        ("implementation_sha256", seal.legacy_implementation_sha256),
        ("contract_sha256", seal.legacy_contract_sha256),
    ):
        if legacy[field] != expected:
            raise SourceBoundSalvageError(f"legacy evaluator binding differs: {field}")
    patched = _closed(
        row["patched_validator"],
        frozenset({"source_revision", "source_sha256", "patch_scope"}),
        label="patched validator binding",
    )
    if patched != {
        "source_revision": seal.patched_validator_source_revision,
        "source_sha256": seal.patched_validator_source_sha256,
        "patch_scope": "canonical_json_object_map_order_only",
    }:
        raise SourceBoundSalvageError("patched validator receipt binding differs")
    salvage_source = _closed(
        row["salvage_source"],
        frozenset(
            {
                "source_revision",
                "source_archive_sha256",
                "tool_member",
                "tool_member_sha256",
                "validator_member",
                "validator_member_sha256",
                "archive_members_equal_executed_files",
            }
        ),
        label="salvage source binding",
    )
    if salvage_source != {
        "source_revision": salvage_source_seal.source_revision,
        "source_archive_sha256": salvage_source_seal.source_archive_sha256,
        "tool_member": SALVAGE_TOOL_MEMBER,
        "tool_member_sha256": salvage_source_seal.tool_member_sha256,
        "validator_member": SALVAGE_VALIDATOR_MEMBER,
        "validator_member_sha256": salvage_source_seal.validator_member_sha256,
        "archive_members_equal_executed_files": True,
    }:
        raise SourceBoundSalvageError("salvage source receipt binding differs")
    root = _closed(
        row["legacy_root"],
        frozenset({"schema_version", "file_sha256", "root_digest", "complete"}),
        label="legacy root binding",
    )
    if root != {
        "schema_version": contract.ROOT_SCHEMA,
        "file_sha256": seal.legacy_root_file_sha256,
        "root_digest": seal.legacy_root_digest,
        "complete": True,
    }:
        raise SourceBoundSalvageError("legacy root receipt binding differs")
    if row["group_order"] != list(contract.EXPECTED_GROUPS):
        raise SourceBoundSalvageError("salvage group order differs")
    candidate_order = row["candidate_order"]
    if candidate_order != list(seal.candidate_order):
        raise SourceBoundSalvageError("salvage candidate order differs")
    for field, expected_items in (
        ("group_receipt_digest_by_id", seal.group_receipt_digest_by_id),
        ("group_receipt_file_sha256_by_id", seal.group_receipt_file_sha256_by_id),
        ("candidate_receipt_digest_by_id", seal.candidate_receipt_digest_by_id),
        ("candidate_receipt_file_sha256_by_id", seal.candidate_receipt_file_sha256_by_id),
    ):
        mapping = row[field]
        expected = dict(expected_items)
        if (
            not isinstance(mapping, Mapping)
            or list(mapping) != [key for key, _ in expected_items]
            or dict(mapping) != expected
        ):
            raise SourceBoundSalvageError(f"salvage exact map closure differs: {field}")
    if row["artifact_reopen_counts"] != {
        "candidate_receipts": 8,
        "group_receipts": 2,
        "root_receipts": 1,
    }:
        raise SourceBoundSalvageError("artifact reopen counts differ")
    if row["closure_checks"] != {
        "canonical_json": True,
        "file_sha256": True,
        "nested_object_digest": True,
        "nested_file_binding": True,
        "key_and_population_closure": True,
    }:
        raise SourceBoundSalvageError("salvage closure-check declaration differs")
    if row["authority_closure"] != AUTHORITY_CLOSURE:
        raise SourceBoundSalvageError("salvage authority closure differs")
    if not isinstance(row["salvage_digest"], str) or _SHA256.fullmatch(row["salvage_digest"]) is None:
        raise SourceBoundSalvageError("salvage digest differs")
    unsigned = dict(row)
    declared = unsigned.pop("salvage_digest")
    if contract.object_sha256(unsigned) != declared:
        raise SourceBoundSalvageError("salvage receipt digest differs")
    return row


def audit_legacy_run(
    *,
    rollout_spec_path: str | Path,
    evaluator_spec_path: str | Path,
    legacy_method_archive_path: str | Path,
    legacy_run_dir: str | Path,
    salvage_source_archive_path: str | Path,
    executed_tool_path: str | Path,
    patched_validator_source_path: str | Path,
    salvage_source_seal: SalvageSourceSeal,
    seal: LegacySeal = REGISTERED_AUH_131222,
) -> dict[str, Any]:
    """Reopen every old receipt and return a path-free salvage receipt."""

    _verify_patched_validator(patched_validator_source_path, seal=seal)
    repeated_source_seal = verify_salvage_source_archive(
        salvage_source_archive_path,
        expected_source_revision=salvage_source_seal.source_revision,
        expected_source_archive_sha256=salvage_source_seal.source_archive_sha256,
        executed_tool_path=executed_tool_path,
        imported_validator_path=patched_validator_source_path,
    )
    if repeated_source_seal != salvage_source_seal:
        raise SourceBoundSalvageError("salvage source seal changed across full audit")

    rollout, rollout_sha = contract.load_current_family_rollout_spec(
        rollout_spec_path, contract.CURRENT_FAMILY_ROLLOUT_SPEC_RAW_SHA256
    )
    spec_value, spec_raw_sha = _canonical_json(
        evaluator_spec_path,
        label="legacy evaluator spec",
        expected_sha256=seal.evaluator_spec_raw_sha256,
    )
    spec = contract.validate_evaluator_spec(spec_value, normalized_rollout=rollout)
    if rollout_sha != spec["rollout_spec_raw_sha256"]:
        raise SourceBoundSalvageError("rollout/evaluator spec raw binding differs")
    if (
        spec["spec_digest"] != seal.evaluator_spec_digest
        or spec["method_source_revision"] != seal.method_source_revision
        or spec["method_source_archive_sha256"] != seal.method_source_archive_sha256
        or spec["implementation_sha256"] != seal.legacy_implementation_sha256
        or spec["contract_sha256"] != seal.legacy_contract_sha256
    ):
        raise SourceBoundSalvageError("registered legacy evaluator binding differs")
    _verify_legacy_archive(legacy_method_archive_path, seal=seal, spec=spec)

    run_root = Path(legacy_run_dir).resolve(strict=True)
    if not run_root.is_dir() or run_root.is_symlink():
        raise SourceBoundSalvageError("legacy run root is not a plain directory")
    root_path = run_root / LEGACY_ROOT_NAME
    root_value, root_file_sha = _canonical_json(
        root_path,
        label="legacy root receipt",
        expected_sha256=seal.legacy_root_file_sha256,
    )
    root = contract.validate_root_receipt(
        root_value,
        evaluator_spec=spec,
        evaluator_spec_raw_sha256=spec_raw_sha,
    )
    if root["root_digest"] != seal.legacy_root_digest:
        raise SourceBoundSalvageError("legacy root object digest differs")
    if (
        root["absolute_source_preservation_pass_claims"] != 0
        or not root["complete"]
        or not root["all_evidence_valid"]
    ):
        raise SourceBoundSalvageError("legacy root evidence/claim state differs")
    exact_root_bindings = {
        "group_receipt_digest_by_id": dict(seal.group_receipt_digest_by_id),
        "group_receipt_file_sha256_by_id": dict(seal.group_receipt_file_sha256_by_id),
        "candidate_receipt_digest_by_id": dict(seal.candidate_receipt_digest_by_id),
        "candidate_receipt_file_sha256_by_id": dict(
            seal.candidate_receipt_file_sha256_by_id
        ),
    }
    if root["candidate_order"] != list(seal.candidate_order):
        raise SourceBoundSalvageError("registered root candidate order differs")
    for field, expected in exact_root_bindings.items():
        if root[field] != expected:
            raise SourceBoundSalvageError(f"registered root exact map differs: {field}")

    groups: dict[str, dict[str, Any]] = {}
    group_file_hashes: dict[str, str] = {}
    candidates: dict[str, dict[str, Any]] = {}
    candidate_file_hashes: dict[str, str] = {}
    for group_id in contract.EXPECTED_GROUPS:
        group_dir = run_root / group_id
        if not group_dir.is_dir() or group_dir.is_symlink():
            raise SourceBoundSalvageError(f"legacy group directory differs: {group_id}")
        wanted = [
            candidate_id
            for candidate_id in spec["candidate_order"]
            if spec["candidate_group_by_id"][candidate_id] == group_id
        ]
        expected_json_names = {LEGACY_GROUP_NAME.format(group_id=group_id)} | {
            f"{candidate_id}.json" for candidate_id in wanted
        }
        actual_json_names = {path.name for path in group_dir.glob("*.json")}
        if actual_json_names != expected_json_names:
            raise SourceBoundSalvageError(f"legacy group JSON file closure differs: {group_id}")
        group_path = group_dir / LEGACY_GROUP_NAME.format(group_id=group_id)
        group_value, group_file_sha = _canonical_json(
            group_path, label=f"legacy group receipt {group_id}"
        )
        group = contract.validate_group_receipt(
            group_value,
            evaluator_spec=spec,
            evaluator_spec_raw_sha256=spec_raw_sha,
        )
        if (
            group_file_sha != root["group_receipt_file_sha256_by_id"][group_id]
            or group["group_digest"] != root["group_receipt_digest_by_id"][group_id]
        ):
            raise SourceBoundSalvageError(f"root/group nested binding differs: {group_id}")
        groups[group_id] = group
        group_file_hashes[group_id] = group_file_sha

        for candidate_id in wanted:
            candidate_path = group_dir / f"{candidate_id}.json"
            candidate_value, candidate_file_sha = _canonical_json(
                candidate_path, label=f"legacy candidate receipt {candidate_id}"
            )
            candidate = contract.validate_candidate_receipt(
                candidate_value,
                evaluator_spec=spec,
                evaluator_spec_raw_sha256=spec_raw_sha,
            )
            if candidate["candidate_id"] != candidate_id or candidate["group_id"] != group_id:
                raise SourceBoundSalvageError(f"candidate identity/group closure differs: {candidate_id}")
            if not candidate["evidence_valid"] or not candidate["eligible_for_downstream_calibration"]:
                raise SourceBoundSalvageError(f"candidate raw-evidence state differs: {candidate_id}")
            if candidate["absolute_source_preservation_pass_claim"]:
                raise SourceBoundSalvageError(f"candidate contains an absolute pass claim: {candidate_id}")
            if (
                candidate_file_sha != group["candidate_receipt_file_sha256_by_id"][candidate_id]
                or candidate_file_sha != root["candidate_receipt_file_sha256_by_id"][candidate_id]
                or candidate["receipt_digest"] != group["candidate_receipt_digest_by_id"][candidate_id]
                or candidate["receipt_digest"] != root["candidate_receipt_digest_by_id"][candidate_id]
            ):
                raise SourceBoundSalvageError(f"candidate nested binding differs: {candidate_id}")
            candidates[candidate_id] = candidate
            candidate_file_hashes[candidate_id] = candidate_file_sha

    if list(candidates) != spec["candidate_order"] or len(candidates) != 8:
        raise SourceBoundSalvageError("eight-candidate population/order closure differs")
    if root["candidate_order"] != spec["candidate_order"]:
        raise SourceBoundSalvageError("root/spec candidate order differs")
    if root["group_order"] != list(contract.EXPECTED_GROUPS):
        raise SourceBoundSalvageError("root group order differs")

    unsigned = {
        "schema_version": SALVAGE_SCHEMA,
        "legacy_slurm_job_id": seal.slurm_job_id,
        "salvage_scope": {
            "kind": "receipt_only_revalidation_of_existing_raw_evidence",
            "legacy_failure_stage": "durable_root_reread_map_order_validator",
            "legacy_failure_was_model_or_metric_compute": False,
            "validator_patch_changes_evidence_values": False,
        },
        "legacy_evaluator": {
            "evaluator_spec_raw_sha256": spec_raw_sha,
            "evaluator_spec_digest": spec["spec_digest"],
            "method_source_revision": spec["method_source_revision"],
            "method_source_archive_sha256": spec["method_source_archive_sha256"],
            "implementation_sha256": spec["implementation_sha256"],
            "contract_sha256": spec["contract_sha256"],
        },
        "patched_validator": {
            "source_revision": seal.patched_validator_source_revision,
            "source_sha256": seal.patched_validator_source_sha256,
            "patch_scope": "canonical_json_object_map_order_only",
        },
        "salvage_source": {
            "source_revision": salvage_source_seal.source_revision,
            "source_archive_sha256": salvage_source_seal.source_archive_sha256,
            "tool_member": SALVAGE_TOOL_MEMBER,
            "tool_member_sha256": salvage_source_seal.tool_member_sha256,
            "validator_member": SALVAGE_VALIDATOR_MEMBER,
            "validator_member_sha256": salvage_source_seal.validator_member_sha256,
            "archive_members_equal_executed_files": True,
        },
        "legacy_root": {
            "schema_version": root["schema_version"],
            "file_sha256": root_file_sha,
            "root_digest": root["root_digest"],
            "complete": root["complete"],
        },
        "group_order": list(contract.EXPECTED_GROUPS),
        "group_receipt_digest_by_id": {
            group_id: groups[group_id]["group_digest"]
            for group_id in contract.EXPECTED_GROUPS
        },
        "group_receipt_file_sha256_by_id": {
            group_id: group_file_hashes[group_id]
            for group_id in contract.EXPECTED_GROUPS
        },
        "candidate_order": list(spec["candidate_order"]),
        "candidate_receipt_digest_by_id": {
            candidate_id: candidates[candidate_id]["receipt_digest"]
            for candidate_id in sorted(candidates)
        },
        "candidate_receipt_file_sha256_by_id": {
            candidate_id: candidate_file_hashes[candidate_id]
            for candidate_id in sorted(candidates)
        },
        "artifact_reopen_counts": {
            "candidate_receipts": 8,
            "group_receipts": 2,
            "root_receipts": 1,
        },
        "closure_checks": {
            "canonical_json": True,
            "file_sha256": True,
            "nested_object_digest": True,
            "nested_file_binding": True,
            "key_and_population_closure": True,
        },
        "authority_closure": dict(AUTHORITY_CLOSURE),
    }
    return validate_salvage_receipt(
        {**unsigned, "salvage_digest": contract.object_sha256(unsigned)},
        salvage_source_seal=salvage_source_seal,
        seal=seal,
    )


def write_fresh_receipt(
    path: str | Path,
    receipt: Mapping[str, Any],
    *,
    legacy_run_dir: str | Path,
    salvage_source_seal: SalvageSourceSeal,
    seal: LegacySeal = REGISTERED_AUH_131222,
) -> str:
    checked = validate_salvage_receipt(
        receipt, salvage_source_seal=salvage_source_seal, seal=seal
    )
    legacy_root = Path(legacy_run_dir).resolve(strict=True)
    output = Path(path)
    if not output.is_absolute():
        raise SourceBoundSalvageError("salvage output path must be absolute")
    parent = output.parent.resolve(strict=True)
    if parent.is_symlink() or not parent.is_dir():
        raise SourceBoundSalvageError("salvage output parent must be a plain directory")
    resolved_output = parent / output.name
    try:
        resolved_output.relative_to(legacy_root)
    except ValueError:
        pass
    else:
        raise SourceBoundSalvageError("salvage output cannot modify the legacy run tree")
    raw = contract.canonical_json_bytes(checked) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(resolved_output, flags, 0o400)
    except OSError as error:
        raise SourceBoundSalvageError("salvage output must be fresh (O_EXCL)") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise
    directory_descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    reread, file_sha = _canonical_json(
        resolved_output, label="new salvage receipt"
    )
    validate_salvage_receipt(
        reread, salvage_source_seal=salvage_source_seal, seal=seal
    )
    if reread != checked:
        raise SourceBoundSalvageError("salvage receipt changed across durable reread")
    return file_sha


def verify_existing_receipt(
    path: str | Path,
    *,
    expected_receipt: Mapping[str, Any],
    salvage_source_seal: SalvageSourceSeal,
    seal: LegacySeal = REGISTERED_AUH_131222,
) -> str:
    """Require durable bytes to equal a newly recomputed full legacy audit."""

    checked = validate_salvage_receipt(
        expected_receipt, salvage_source_seal=salvage_source_seal, seal=seal
    )
    _, raw, file_sha = _stable_read(path, label="durable salvage receipt")
    expected_raw = contract.canonical_json_bytes(checked) + b"\n"
    if raw != expected_raw:
        raise SourceBoundSalvageError(
            "durable salvage receipt is not byte-equal to the repeated full audit"
        )
    try:
        durable = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceBoundSalvageError("durable salvage receipt is not JSON") from error
    validate_salvage_receipt(
        durable, salvage_source_seal=salvage_source_seal, seal=seal
    )
    return file_sha


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-spec", required=True)
    parser.add_argument("--evaluator-spec", required=True)
    parser.add_argument("--legacy-method-archive", required=True)
    parser.add_argument("--legacy-run-dir", required=True)
    parser.add_argument("--salvage-source-archive", required=True)
    parser.add_argument("--salvage-source-revision", required=True)
    parser.add_argument("--salvage-source-archive-sha256", required=True)
    parser.add_argument("--patched-validator-source", default=str(Path(contract.__file__)))
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-existing-output", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    implementation_path = Path(__file__).resolve(strict=True)
    salvage_source_seal = verify_salvage_source_archive(
        args.salvage_source_archive,
        expected_source_revision=args.salvage_source_revision,
        expected_source_archive_sha256=args.salvage_source_archive_sha256,
        executed_tool_path=implementation_path,
        imported_validator_path=args.patched_validator_source,
    )
    receipt = audit_legacy_run(
        rollout_spec_path=args.rollout_spec,
        evaluator_spec_path=args.evaluator_spec,
        legacy_method_archive_path=args.legacy_method_archive,
        legacy_run_dir=args.legacy_run_dir,
        salvage_source_archive_path=args.salvage_source_archive,
        executed_tool_path=implementation_path,
        patched_validator_source_path=args.patched_validator_source,
        salvage_source_seal=salvage_source_seal,
    )
    if args.verify_existing_output:
        output_sha = verify_existing_receipt(
            args.output,
            expected_receipt=receipt,
            salvage_source_seal=salvage_source_seal,
        )
        print(
            "PAIR_V5_SOURCE_BOUND_131222_DURABLE_SALVAGE_POSTFLIGHT_OK "
            f"repeated_full_audit=true byte_equal=true output_sha256={output_sha} "
            "absolute_preservation_pass=false optimizer_go=false"
        )
    else:
        output_sha = write_fresh_receipt(
            args.output,
            receipt,
            legacy_run_dir=args.legacy_run_dir,
            salvage_source_seal=salvage_source_seal,
        )
        print(
            "PAIR_V5_SOURCE_BOUND_131222_RECEIPT_SALVAGE_OK "
            f"candidates=8 groups=2 roots=1 output_sha256={output_sha} "
            "dino_recomputed=false absolute_preservation_pass=false optimizer_go=false"
        )


if __name__ == "__main__":
    main()
