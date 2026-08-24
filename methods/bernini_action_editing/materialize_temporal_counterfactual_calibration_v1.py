#!/usr/bin/env python3
"""Materialize exact40 temporal-counterfactual calibration evidence.

This is a CPU-only JSON receipt bridge.  It authenticates the frozen core4-v2
spec and bank receipt, both temporal score-group receipts and all 40 score
receipts, the separately reviewed d541801-v3 label manifest, and the exact 40
detached event audits named by the pinned d541801 bridge receipt.  It then
invokes ``temporal_counterfactual_calibration_v1`` and immediately validates
the result by replaying every score, audit, group, and preregistration input.

The program never opens media, latent/tensor artifacts, a checkpoint, or a
model.  It has no optimizer or training entry point.  A GO JSON object has no
standalone authority: validation of a GO materialization requires the exact
external replay population again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile
from typing import Any, Iterable, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = METHOD_ROOT / "tools"
for search_root in (METHOD_ROOT, TOOLS_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import author_pair_v5_core4_event_labels_d541801_v3 as label_author  # noqa: E402
import materialize_pair_v5_t2v_energy_calibration_bridge_d541801_v3 as audit_bridge  # noqa: E402
import pair_v5_t2v_energy_calibration_v3 as detached_events  # noqa: E402
import temporal_counterfactual_calibration_v1 as calibration  # noqa: E402
import temporal_counterfactual_contract_v1 as contract  # noqa: E402


MATERIALIZATION_SCHEMA = (
    "bernini-temporal-counterfactual-calibration-materialization-v1"
)
MATERIALIZATION_FILENAME = (
    "temporal-counterfactual-calibration-materialization-v1-receipt.json"
)
PREREGISTRATION_FILENAME = "temporal-counterfactual-preregistration-v1.json"
CALIBRATION_FILENAME = "temporal-counterfactual-calibration-v1.json"
GROUP_IDS = ("sp4-a", "sp4-b")
GROUP_SIZE = 20
CANDIDATE_COUNT = 40
SCORE_FILENAME = "temporal-counterfactual-action-score-v1.json"
GROUP_FILENAME = "temporal-counterfactual-action-score-{group_id}-v1.json"
SCORE_ROOT_LOGS = ("sp4-a.log", "sp4-b.log")

# Formal authorities frozen before job 131237 was submitted.
REQUIRED_AUDIT_BRIDGE_FILE_SHA256 = (
    "5401ca7075ef1a5818251d9fbdad150131eb770f0822ef93666e2d5677d7b45f"
)
REQUIRED_AUDIT_BRIDGE_RECEIPT_DIGEST = (
    "1d7a4d42942ff4e5199afa809ae83efda177739109f1a0a697e6fb5df5867aa6"
)
REQUIRED_LABEL_MANIFEST_FILE_SHA256 = (
    "9246504e97e1ee46c2cdcf7dfac0f41364dca40f26e5c26f28f0968d0443808d"
)
REQUIRED_PREREGISTRATION_FILE_SHA256 = (
    "d6df656bfe65aff6966d68bb002ecff071320b4af12e526564781771cb74bd25"
)
REQUIRED_PREREGISTRATION_DIGEST = (
    "1620be58a1525a163b4dccc8de8d97bf1b86a4a659677ab0d2b1f883bfea4a9e"
)
REQUIRED_CALIBRATOR_SOURCE_REVISION = (
    "f7ffa5921026b991b0b74592ad9bdce6e81bf76d"
)
REQUIRED_CALIBRATOR_SOURCE_ARCHIVE_SHA256 = (
    "55a7ae5f4555b290a17339e6bd09c2f6fff39962535ce012f0581b7f63ded8f2"
)
REQUIRED_CALIBRATOR_SOURCE_SHA256 = (
    "18fbcb381c894188832423d7af5099e4dc299c05bcf99215b9bf725979e918e1"
)

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MATERIALIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "materializer_source_sha256",
        "source_root_spec",
        "source_bank_receipt",
        "score_root_binding",
        "score_group_file_sha256_authority",
        "score_group_bindings",
        "audit_authority_bridge",
        "detached_event_label_manifest",
        "event_audit_root_binding",
        "preregistration_input_binding",
        "calibrator_source_binding",
        "candidate_count",
        "candidate_order",
        "calibrator_input_contract",
        "preregistration_output_binding",
        "calibration_output_binding",
        "reviewed_label_manifest_acknowledged",
        "optimizer_authority_requires_exact_external_replay",
        "optimizer_authorized",
        "failure_reasons",
        "confirmation_samples_consumed_by_optimizer",
        "training_performed",
        "scientific_action_editing_claim",
        "receipt_digest",
    }
)


class TemporalCounterfactualMaterializationError(RuntimeError):
    """A file, receipt, authority, join, or replay check failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise TemporalCounterfactualMaterializationError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _json_file_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_file_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise TemporalCounterfactualMaterializationError(
            f"{label} must be lowercase SHA-1"
        )
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TemporalCounterfactualMaterializationError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _no_symlink_components(path: Path, *, label: str) -> None:
    if not path.is_absolute():
        raise TemporalCounterfactualMaterializationError(
            f"{label} must be absolute"
        )
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise TemporalCounterfactualMaterializationError(
                f"{label} contains a symlink component"
            )


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    _no_symlink_components(path, label=label)
    if not path.is_file() or path.resolve(strict=True) != path:
        raise TemporalCounterfactualMaterializationError(
            f"{label} must be a normalized absolute plain file"
        )
    return path


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    _no_symlink_components(path, label=label)
    if not path.is_dir() or path.resolve(strict=True) != path:
        raise TemporalCounterfactualMaterializationError(
            f"{label} must be a normalized absolute plain directory"
        )
    return path


def _fresh_output_directory(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path("/") or path.exists() or path.is_symlink():
        raise TemporalCounterfactualMaterializationError(
            "output must be a fresh normalized absolute directory"
        )
    parent = _plain_directory(path.parent, label="output parent")
    if parent / path.name != path:
        raise TemporalCounterfactualMaterializationError(
            "output path must be normalized under its plain parent"
        )
    return path


def _reject_constant(token: str) -> None:
    raise TemporalCounterfactualMaterializationError(
        f"non-finite JSON is forbidden: {token}"
    )


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TemporalCounterfactualMaterializationError(
                f"duplicate JSON key: {key!r}"
            )
        result[key] = value
    return result


def _read_strict_json(
    value: str | Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], Path, str]:
    path = _plain_file(value, label=label)
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and observed != _sha256(
        expected_sha256, label=f"{label} expected SHA-256"
    ):
        raise TemporalCounterfactualMaterializationError(f"{label} SHA-256 differs")
    try:
        decoded = json.loads(
            raw.decode("ascii"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TemporalCounterfactualMaterializationError(
            f"{label} is invalid ASCII JSON"
        ) from error
    if not isinstance(decoded, dict):
        raise TemporalCounterfactualMaterializationError(
            f"{label} root must be an object"
        )
    return decoded, path, observed


def _require_exact_authorities(
    *,
    expected_root_spec_sha256: str,
    expected_bank_receipt_sha256: str,
    expected_audit_bridge_sha256: str,
    expected_detached_label_manifest_sha256: str,
    expected_preregistration_sha256: str,
    calibrator_source_archive: str | Path,
    calibrator_source_revision: str,
    calibrator_source_archive_sha256: str,
    expected_calibrator_source_sha256: str,
    expected_materializer_source_sha256: str,
) -> Path:
    expected = {
        "root spec SHA-256": (
            expected_root_spec_sha256,
            contract.REQUIRED_CORE4_V2_SPEC_SHA256,
        ),
        "bank receipt SHA-256": (
            expected_bank_receipt_sha256,
            contract.REQUIRED_CORE4_V2_BANK_RECEIPT_FILE_SHA256,
        ),
        "audit bridge SHA-256": (
            expected_audit_bridge_sha256,
            REQUIRED_AUDIT_BRIDGE_FILE_SHA256,
        ),
        "detached label manifest SHA-256": (
            expected_detached_label_manifest_sha256,
            REQUIRED_LABEL_MANIFEST_FILE_SHA256,
        ),
        "preregistration SHA-256": (
            expected_preregistration_sha256,
            REQUIRED_PREREGISTRATION_FILE_SHA256,
        ),
        "calibrator archive SHA-256": (
            calibrator_source_archive_sha256,
            REQUIRED_CALIBRATOR_SOURCE_ARCHIVE_SHA256,
        ),
        "calibrator source SHA-256": (
            expected_calibrator_source_sha256,
            REQUIRED_CALIBRATOR_SOURCE_SHA256,
        ),
    }
    for label, (observed, required) in expected.items():
        if _sha256(observed, label=label) != required:
            raise TemporalCounterfactualMaterializationError(
                f"formal {label} authority differs"
            )
    if (
        _sha1(calibrator_source_revision, label="calibrator source revision")
        != REQUIRED_CALIBRATOR_SOURCE_REVISION
    ):
        raise TemporalCounterfactualMaterializationError(
            "formal calibrator source revision differs"
        )
    materializer_sha = _sha256(
        expected_materializer_source_sha256,
        label="materializer source SHA-256",
    )
    if materializer_sha != file_sha256(Path(__file__).resolve()):
        raise TemporalCounterfactualMaterializationError(
            "loaded materializer source SHA-256 differs"
        )
    if REQUIRED_CALIBRATOR_SOURCE_SHA256 != file_sha256(
        Path(calibration.__file__).resolve()
    ):
        raise TemporalCounterfactualMaterializationError(
            "loaded calibrator source SHA-256 differs"
        )
    archive_path = _plain_file(
        calibrator_source_archive, label="calibrator source archive"
    )
    if file_sha256(archive_path) != REQUIRED_CALIBRATOR_SOURCE_ARCHIVE_SHA256:
        raise TemporalCounterfactualMaterializationError(
            "calibrator source archive file SHA-256 differs"
        )
    try:
        with tarfile.open(archive_path, "r:*") as handle:
            archive_revision = handle.pax_headers.get("comment")
    except (tarfile.TarError, OSError) as error:
        raise TemporalCounterfactualMaterializationError(
            "calibrator source archive is not a readable git archive"
        ) from error
    if archive_revision != REQUIRED_CALIBRATOR_SOURCE_REVISION:
        raise TemporalCounterfactualMaterializationError(
            "calibrator source archive git revision differs"
        )
    return archive_path


def _validate_score_root_closure(root: Path) -> list[dict[str, str]]:
    expected = set(GROUP_IDS) | set(SCORE_ROOT_LOGS)
    observed = {entry.name for entry in root.iterdir()}
    if observed != expected:
        raise TemporalCounterfactualMaterializationError(
            "score root closure differs: "
            f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )
    bindings = []
    for name in SCORE_ROOT_LOGS:
        path = _plain_file(root / name, label=f"{name} scorer log")
        bindings.append({"name": name, "file_sha256": file_sha256(path)})
    return bindings


def load_ordered_score_population(
    *,
    score_root: str | Path,
    expected_group_file_sha256_by_id: Mapping[str, str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, str]],
    Path,
]:
    root = _plain_directory(score_root, label="temporal score root")
    log_bindings = _validate_score_root_closure(root)
    if set(expected_group_file_sha256_by_id) != set(GROUP_IDS):
        raise TemporalCounterfactualMaterializationError(
            "score-group SHA registry differs"
        )
    groups: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    group_bindings: list[dict[str, Any]] = []
    for group_id in GROUP_IDS:
        group_root = _plain_directory(
            root / group_id, label=f"{group_id} temporal score root"
        )
        group_name = GROUP_FILENAME.format(group_id=group_id)
        raw_group, group_path, group_file_sha = _read_strict_json(
            group_root / group_name,
            label=f"{group_id} temporal group receipt",
            expected_sha256=expected_group_file_sha256_by_id[group_id],
        )
        try:
            group = contract.validate_group_receipt(raw_group)
        except contract.TemporalCounterfactualContractError as error:
            raise TemporalCounterfactualMaterializationError(str(error)) from error
        candidate_ids = group["candidate_order"]
        expected_entries = set(candidate_ids) | {group_name}
        observed_entries = {entry.name for entry in group_root.iterdir()}
        if (
            group["group_id"] != group_id
            or group["candidate_count"] != GROUP_SIZE
            or len(candidate_ids) != GROUP_SIZE
            or len(set(candidate_ids)) != GROUP_SIZE
            or observed_entries != expected_entries
        ):
            raise TemporalCounterfactualMaterializationError(
                f"{group_id} temporal score filesystem/order closure differs"
            )
        group_scores: list[dict[str, Any]] = []
        score_bindings: list[dict[str, Any]] = []
        for ordinal, candidate_id in enumerate(candidate_ids):
            candidate_root = _plain_directory(
                group_root / candidate_id,
                label=f"{candidate_id} score directory",
            )
            if {entry.name for entry in candidate_root.iterdir()} != {SCORE_FILENAME}:
                raise TemporalCounterfactualMaterializationError(
                    f"{candidate_id} score directory closure differs"
                )
            raw_score, score_path, score_file_sha = _read_strict_json(
                candidate_root / SCORE_FILENAME,
                label=f"{candidate_id} temporal score receipt",
            )
            try:
                score = contract.validate_candidate_score_receipt(raw_score)
            except contract.TemporalCounterfactualContractError as error:
                raise TemporalCounterfactualMaterializationError(str(error)) from error
            if (
                score["group_id"] != group_id
                or score["candidate_identity"]["candidate_id"] != candidate_id
            ):
                raise TemporalCounterfactualMaterializationError(
                    f"{candidate_id} score/group identity differs"
                )
            group_scores.append(score)
            score_bindings.append(
                {
                    "ordinal": ordinal,
                    "candidate_id": candidate_id,
                    "path": str(score_path),
                    "file_sha256": score_file_sha,
                    "receipt_digest": score["receipt_digest"],
                }
            )
        try:
            group = contract.validate_group_receipt(
                group, candidate_receipts=group_scores
            )
        except contract.TemporalCounterfactualContractError as error:
            raise TemporalCounterfactualMaterializationError(str(error)) from error
        groups.append(group)
        scores.extend(group_scores)
        group_bindings.append(
            {
                "group_id": group_id,
                "path": str(group_path),
                "file_sha256": group_file_sha,
                "receipt_digest": group["receipt_digest"],
                "candidate_count": GROUP_SIZE,
                "candidate_order": list(candidate_ids),
                "candidate_receipt_digests": [
                    row["receipt_digest"] for row in group_scores
                ],
                "score_receipt_files": score_bindings,
            }
        )
    candidate_order = [
        score["candidate_identity"]["candidate_id"] for score in scores
    ]
    if (
        len(scores) != CANDIDATE_COUNT
        or len(set(candidate_order)) != CANDIDATE_COUNT
        or contract.object_sha256(candidate_order)
        != contract.REQUIRED_CORE4_V2_CANDIDATE_ORDER_DIGEST
        or contract.object_sha256(
            [score["candidate_identity"] for score in scores]
        )
        != contract.REQUIRED_CORE4_V2_CANDIDATE_IDENTITY_DIGEST
    ):
        raise TemporalCounterfactualMaterializationError(
            "global temporal score population differs from formal exact40"
        )
    return groups, scores, group_bindings, log_bindings, root


def _validate_audit_authority(
    *,
    audit_bridge_receipt: str | Path,
    expected_audit_bridge_sha256: str,
    event_audit_root: str | Path,
    detached_label_manifest: str | Path,
    expected_detached_label_manifest_sha256: str,
    root_spec_path: Path,
    bank_receipt_path: Path,
    scores: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, Any],
    Path,
    str,
    dict[str, Any],
    Path,
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
    Path,
]:
    raw_bridge, bridge_path, bridge_file_sha = _read_strict_json(
        audit_bridge_receipt,
        label="formal d541801 audit bridge receipt",
        expected_sha256=expected_audit_bridge_sha256,
    )
    try:
        bridge = audit_bridge.validate_bridge_receipt(raw_bridge)
    except audit_bridge.PairV5Core4CalibrationBridgeError as error:
        raise TemporalCounterfactualMaterializationError(str(error)) from error
    if (
        bridge["receipt_digest"] != REQUIRED_AUDIT_BRIDGE_RECEIPT_DIGEST
        or bridge["source_root_spec"]
        != {
            "path": str(root_spec_path),
            "file_sha256": contract.REQUIRED_CORE4_V2_SPEC_SHA256,
            "schema_version": "pair-v5-frozen-bernini-t2v-calibration-bank-spec-v2",
        }
        or bridge["source_bank_receipt"]["path"] != str(bank_receipt_path)
        or bridge["source_bank_receipt"]["file_sha256"]
        != contract.REQUIRED_CORE4_V2_BANK_RECEIPT_FILE_SHA256
        or bridge["source_bank_receipt"]["receipt_digest"]
        != contract.REQUIRED_CORE4_V2_BANK_RECEIPT_DIGEST
    ):
        raise TemporalCounterfactualMaterializationError(
            "formal audit bridge root/bank authority differs"
        )

    raw_labels, label_path, label_file_sha = _read_strict_json(
        detached_label_manifest,
        label="formal detached label manifest",
        expected_sha256=expected_detached_label_manifest_sha256,
    )
    label_binding = bridge["detached_event_label_manifest"]
    expected_acknowledgements = {
        name: True for name in label_author.ACKNOWLEDGEMENT_FIELDS
    }
    if (
        label_binding.get("path") != str(label_path)
        or label_binding.get("file_sha256") != label_file_sha
        or label_binding.get("candidate_count") != CANDIDATE_COUNT
        or label_binding.get("author_acknowledgements")
        != expected_acknowledgements
        or raw_labels.get("manifest_digest") != label_binding.get("manifest_digest")
    ):
        raise TemporalCounterfactualMaterializationError(
            "audit bridge/label-manifest binding differs"
        )

    candidate_order = [
        score["candidate_identity"]["candidate_id"] for score in scores
    ]
    bound_rows = [
        {
            "group_id": score["group_id"],
            "candidate": dict(score["candidate_identity"]),
            "generation_receipt_digest": score["generation_binding"][
                "generation_receipt_digest"
            ],
        }
        for score in scores
    ]
    try:
        labels = label_author.validate_label_manifest(
            raw_labels,
            root_spec_raw_sha256=contract.REQUIRED_CORE4_V2_SPEC_SHA256,
            bank_receipt_digest=contract.REQUIRED_CORE4_V2_BANK_RECEIPT_DIGEST,
            bound_rows=bound_rows,
            verify_external_artifacts=False,
        )
    except label_author.PairV5Core4LabelAuthoringError as error:
        raise TemporalCounterfactualMaterializationError(str(error)) from error
    if bridge["candidate_order"] != candidate_order:
        raise TemporalCounterfactualMaterializationError(
            "audit bridge/temporal score candidate order differs"
        )

    audit_root = _plain_directory(event_audit_root, label="event-audit root")
    bindings = bridge["event_audit_receipt_bindings"]
    expected_names = {Path(binding["path"]).name for binding in bindings}
    if (
        len(expected_names) != CANDIDATE_COUNT
        or {entry.name for entry in audit_root.iterdir()} != expected_names
        or any(Path(binding["path"]).parent != audit_root for binding in bindings)
    ):
        raise TemporalCounterfactualMaterializationError(
            "event-audit filesystem/bridge path closure differs"
        )
    audits: list[dict[str, Any]] = []
    audit_bindings: list[dict[str, Any]] = []
    label_rows = labels["rows"]
    generation_rows = bridge["generation_bindings"]
    for ordinal, (binding, label, generation, score) in enumerate(
        zip(bindings, label_rows, generation_rows, scores)
    ):
        candidate_id = candidate_order[ordinal]
        raw_audit, audit_path, audit_file_sha = _read_strict_json(
            binding["path"],
            label=f"{candidate_id} detached event audit",
            expected_sha256=binding["file_sha256"],
        )
        try:
            audit = detached_events.validate_event_audit_receipt(raw_audit)
        except detached_events.PairV5EnergyCalibrationV3Error as error:
            raise TemporalCounterfactualMaterializationError(str(error)) from error
        identity = score["candidate_identity"]
        identity_fields = (
            "candidate_id",
            "analysis_split",
            "action_family_id",
            "calibration_group_id",
            "actor_group_id",
            "scene_group_id",
            "action_group_id",
            "semantic_branch",
        )
        label_fields = (
            "audit_source_kind",
            "external_audit_artifact_sha256",
            *label_author.LABEL_BOOLEAN_FIELDS,
        )
        if (
            binding.get("ordinal") != ordinal
            or binding.get("candidate_id") != candidate_id
            or binding.get("path") != str(audit_path)
            or binding.get("receipt_digest") != audit["receipt_digest"]
            or label.get("ordinal") != ordinal
            or label.get("group_id") != score["group_id"]
            or any(audit[name] != identity[name] for name in identity_fields)
            or any(label[name] != audit[name] for name in (*identity_fields, *label_fields))
            or generation.get("ordinal") != ordinal
            or generation.get("group_id") != score["group_id"]
            or generation.get("candidate_id") != candidate_id
            or generation.get("generation_receipt_digest")
            != score["generation_binding"]["generation_receipt_digest"]
            or audit["generation_receipt_digest"]
            != score["generation_binding"]["generation_receipt_digest"]
        ):
            raise TemporalCounterfactualMaterializationError(
                f"{candidate_id} score/label/bridge/audit join differs"
            )
        audits.append(audit)
        audit_bindings.append(
            {
                "ordinal": ordinal,
                "candidate_id": candidate_id,
                "path": str(audit_path),
                "file_sha256": audit_file_sha,
                "receipt_digest": audit["receipt_digest"],
            }
        )
    return (
        bridge,
        bridge_path,
        bridge_file_sha,
        labels,
        label_path,
        label_file_sha,
        audits,
        audit_bindings,
        audit_root,
    )


def _output_binding(
    path: Path, value: Mapping[str, Any], *, digest_field: str
) -> dict[str, Any]:
    return {
        "path": str(path),
        "file_sha256": _json_file_sha256(value),
        digest_field: value[digest_field],
    }


def make_materialization_receipt(
    *,
    materializer_source_sha256: str,
    root_spec_path: Path,
    bank_receipt_path: Path,
    score_root: Path,
    score_log_bindings: Sequence[Mapping[str, Any]],
    score_group_file_sha256_authority: str,
    score_group_bindings: Sequence[Mapping[str, Any]],
    bridge: Mapping[str, Any],
    bridge_path: Path,
    bridge_file_sha256: str,
    labels: Mapping[str, Any],
    label_path: Path,
    label_file_sha256: str,
    audit_root: Path,
    audit_bindings: Sequence[Mapping[str, Any]],
    preregistration_path: Path,
    preregistration_file_sha256: str,
    preregistration: Mapping[str, Any],
    calibration_receipt: Mapping[str, Any],
    calibrator_source_revision: str,
    calibrator_source_archive_path: Path,
    calibrator_source_archive_sha256: str,
    expected_calibrator_source_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    candidate_order = [row["candidate_id"] for row in audit_bindings]
    unsigned = {
        "schema_version": MATERIALIZATION_SCHEMA,
        "materializer_source_sha256": materializer_source_sha256,
        "source_root_spec": {
            "path": str(root_spec_path),
            "file_sha256": contract.REQUIRED_CORE4_V2_SPEC_SHA256,
        },
        "source_bank_receipt": {
            "path": str(bank_receipt_path),
            "file_sha256": contract.REQUIRED_CORE4_V2_BANK_RECEIPT_FILE_SHA256,
            "receipt_digest": contract.REQUIRED_CORE4_V2_BANK_RECEIPT_DIGEST,
        },
        "score_root_binding": {
            "path": str(score_root),
            "root_entries": [*GROUP_IDS, *SCORE_ROOT_LOGS],
            "nonsemantic_log_files": [dict(row) for row in score_log_bindings],
        },
        "score_group_file_sha256_authority": score_group_file_sha256_authority,
        "score_group_bindings": [dict(row) for row in score_group_bindings],
        "audit_authority_bridge": {
            "path": str(bridge_path),
            "file_sha256": bridge_file_sha256,
            "receipt_digest": bridge["receipt_digest"],
        },
        "detached_event_label_manifest": {
            "path": str(label_path),
            "file_sha256": label_file_sha256,
            "manifest_digest": labels["manifest_digest"],
            "author_acknowledgements": labels["author_acknowledgements"],
        },
        "event_audit_root_binding": {
            "path": str(audit_root),
            "candidate_count": CANDIDATE_COUNT,
            "receipt_files": [dict(row) for row in audit_bindings],
        },
        "preregistration_input_binding": {
            "path": str(preregistration_path),
            "file_sha256": preregistration_file_sha256,
            "preregistration_digest": preregistration[
                "preregistration_digest"
            ],
        },
        "calibrator_source_binding": {
            "method_source_revision": calibrator_source_revision,
            "method_source_archive_path": str(calibrator_source_archive_path),
            "method_source_archive_sha256": calibrator_source_archive_sha256,
            "calibrator_source_sha256": expected_calibrator_source_sha256,
        },
        "candidate_count": CANDIDATE_COUNT,
        "candidate_order": candidate_order,
        "calibrator_input_contract": {
            "input_object_kinds": [
                "temporal_counterfactual_score_receipts",
                "detached_boolean_event_audit_receipts",
                "temporal_score_group_receipts",
                "preregistration",
            ],
            "calibrator_input_files_read": ["ascii_json_receipts_only"],
            "source_archive_verified_as_provenance_outside_calibrator": True,
            "media_opened_or_decoded": False,
            "latent_or_tensor_artifact_opened": False,
            "checkpoint_or_model_loaded": False,
            "external_audit_artifact_opened": False,
            "optimizer_or_training_entry_point_present": False,
            "t2v_receipts_may_enter_rv2v_condition_target_donor_or_noise": False,
        },
        "preregistration_output_binding": _output_binding(
            output_dir / PREREGISTRATION_FILENAME,
            preregistration,
            digest_field="preregistration_digest",
        ),
        "calibration_output_binding": {
            **_output_binding(
                output_dir / CALIBRATION_FILENAME,
                calibration_receipt,
                digest_field="receipt_digest",
            ),
            "optimizer_authorized": calibration_receipt[
                "optimizer_authorized"
            ],
            "failure_reasons": calibration_receipt["failure_reasons"],
        },
        "reviewed_label_manifest_acknowledged": True,
        "optimizer_authority_requires_exact_external_replay": True,
        "optimizer_authorized": calibration_receipt["optimizer_authorized"],
        "failure_reasons": calibration_receipt["failure_reasons"],
        "confirmation_samples_consumed_by_optimizer": False,
        "training_performed": False,
        "scientific_action_editing_claim": False,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def validate_materialization_receipt(
    value: Any,
    *,
    calibration_receipt: Mapping[str, Any] | None = None,
    score_receipts: Sequence[Mapping[str, Any]] | None = None,
    event_audit_receipts: Sequence[Mapping[str, Any]] | None = None,
    preregistration: Mapping[str, Any] | None = None,
    group_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_MATERIALIZATION_FIELDS):
        raise TemporalCounterfactualMaterializationError(
            "materialization receipt field closure differs"
        )
    row = dict(value)
    unsigned = dict(row)
    digest = _sha256(
        unsigned.pop("receipt_digest"), label="materialization receipt digest"
    )
    if object_sha256(unsigned) != digest:
        raise TemporalCounterfactualMaterializationError(
            "materialization receipt digest differs"
        )
    failures = row["failure_reasons"]
    if (
        type(row["optimizer_authorized"]) is not bool
        or type(failures) is not list
        or any(type(reason) is not str or not reason for reason in failures)
        or failures != sorted(set(failures))
    ):
        raise TemporalCounterfactualMaterializationError(
            "materialization optimizer/failure fields differ"
        )
    groups = row["score_group_bindings"]
    audits = row["event_audit_root_binding"].get("receipt_files", [])
    candidate_order = row["candidate_order"]
    if (
        row["schema_version"] != MATERIALIZATION_SCHEMA
        or row["candidate_count"] != CANDIDATE_COUNT
        or not isinstance(candidate_order, list)
        or len(candidate_order) != CANDIDATE_COUNT
        or len(set(candidate_order)) != CANDIDATE_COUNT
        or contract.object_sha256(candidate_order)
        != contract.REQUIRED_CORE4_V2_CANDIDATE_ORDER_DIGEST
        or not isinstance(groups, list)
        or [group.get("group_id") for group in groups] != list(GROUP_IDS)
        or any(group.get("candidate_count") != GROUP_SIZE for group in groups)
        or [candidate_id for group in groups for candidate_id in group["candidate_order"]]
        != candidate_order
        or not isinstance(audits, list)
        or [audit.get("ordinal") for audit in audits]
        != list(range(CANDIDATE_COUNT))
        or [audit.get("candidate_id") for audit in audits] != candidate_order
        or row["source_root_spec"].get("file_sha256")
        != contract.REQUIRED_CORE4_V2_SPEC_SHA256
        or row["source_bank_receipt"].get("file_sha256")
        != contract.REQUIRED_CORE4_V2_BANK_RECEIPT_FILE_SHA256
        or row["source_bank_receipt"].get("receipt_digest")
        != contract.REQUIRED_CORE4_V2_BANK_RECEIPT_DIGEST
        or row["score_group_file_sha256_authority"]
        not in {"presealed_external", "observed_afterok_runtime"}
        or row["audit_authority_bridge"].get("file_sha256")
        != REQUIRED_AUDIT_BRIDGE_FILE_SHA256
        or row["audit_authority_bridge"].get("receipt_digest")
        != REQUIRED_AUDIT_BRIDGE_RECEIPT_DIGEST
        or row["detached_event_label_manifest"].get("file_sha256")
        != REQUIRED_LABEL_MANIFEST_FILE_SHA256
        or row["preregistration_input_binding"].get("file_sha256")
        != REQUIRED_PREREGISTRATION_FILE_SHA256
        or row["preregistration_input_binding"].get("preregistration_digest")
        != REQUIRED_PREREGISTRATION_DIGEST
        or not isinstance(row["calibrator_source_binding"], Mapping)
        or set(row["calibrator_source_binding"])
        != {
            "method_source_revision",
            "method_source_archive_path",
            "method_source_archive_sha256",
            "calibrator_source_sha256",
        }
        or row["calibrator_source_binding"].get("method_source_revision")
        != REQUIRED_CALIBRATOR_SOURCE_REVISION
        or row["calibrator_source_binding"].get("method_source_archive_sha256")
        != REQUIRED_CALIBRATOR_SOURCE_ARCHIVE_SHA256
        or row["calibrator_source_binding"].get("calibrator_source_sha256")
        != REQUIRED_CALIBRATOR_SOURCE_SHA256
        or not isinstance(
            row["calibrator_source_binding"].get("method_source_archive_path"),
            str,
        )
        or row["calibrator_input_contract"]
        != {
            "input_object_kinds": [
                "temporal_counterfactual_score_receipts",
                "detached_boolean_event_audit_receipts",
                "temporal_score_group_receipts",
                "preregistration",
            ],
            "calibrator_input_files_read": ["ascii_json_receipts_only"],
            "source_archive_verified_as_provenance_outside_calibrator": True,
            "media_opened_or_decoded": False,
            "latent_or_tensor_artifact_opened": False,
            "checkpoint_or_model_loaded": False,
            "external_audit_artifact_opened": False,
            "optimizer_or_training_entry_point_present": False,
            "t2v_receipts_may_enter_rv2v_condition_target_donor_or_noise": False,
        }
        or row["reviewed_label_manifest_acknowledged"] is not True
        or row["optimizer_authority_requires_exact_external_replay"] is not True
        or row["confirmation_samples_consumed_by_optimizer"] is not False
        or row["training_performed"] is not False
        or row["scientific_action_editing_claim"] is not False
        or row["optimizer_authorized"] == bool(failures)
        or row["calibration_output_binding"].get("optimizer_authorized")
        is not row["optimizer_authorized"]
        or row["calibration_output_binding"].get("failure_reasons") != failures
    ):
        raise TemporalCounterfactualMaterializationError(
            "materialization receipt semantics differ"
        )
    replay_values = (
        calibration_receipt,
        score_receipts,
        event_audit_receipts,
        preregistration,
        group_receipts,
    )
    supplied = [item is not None for item in replay_values]
    if any(supplied) and not all(supplied):
        raise TemporalCounterfactualMaterializationError(
            "materialization replay requires calibration, scores, audits, preregistration, and groups together"
        )
    if row["optimizer_authorized"] is True and not all(supplied):
        raise TemporalCounterfactualMaterializationError(
            "GO materialization requires exact external replay"
        )
    if all(supplied):
        try:
            checked = calibration.validate_calibration_receipt(
                calibration_receipt,
                score_receipts=score_receipts,
                event_audit_receipts=event_audit_receipts,
                preregistration=preregistration,
                group_receipts=group_receipts,
            )
        except calibration.TemporalCounterfactualCalibrationError as error:
            raise TemporalCounterfactualMaterializationError(str(error)) from error
        binding = row["calibration_output_binding"]
        if (
            checked["receipt_digest"] != binding.get("receipt_digest")
            or _json_file_sha256(checked) != binding.get("file_sha256")
            or checked["optimizer_authorized"] is not row["optimizer_authorized"]
            or checked["failure_reasons"] != failures
        ):
            raise TemporalCounterfactualMaterializationError(
                "materialization/calibration replay binding differs"
            )
    return row


def _write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise TemporalCounterfactualMaterializationError(
            f"refusing to overwrite {path}"
        )
    raw = _json_file_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o400)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise TemporalCounterfactualMaterializationError(
            f"refusing to overwrite {path}"
        ) from error
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if path.exists() and not path.is_symlink():
            path.unlink()
        raise
    return hashlib.sha256(raw).hexdigest()


def materialize(
    *,
    root_spec: str | Path,
    expected_root_spec_sha256: str,
    bank_receipt: str | Path,
    expected_bank_receipt_sha256: str,
    score_root: str | Path,
    expected_sp4_a_score_group_sha256: str,
    expected_sp4_b_score_group_sha256: str,
    score_group_file_sha256_authority: str,
    audit_bridge_receipt: str | Path,
    expected_audit_bridge_sha256: str,
    event_audit_root: str | Path,
    detached_label_manifest: str | Path,
    expected_detached_label_manifest_sha256: str,
    preregistration: str | Path,
    expected_preregistration_sha256: str,
    calibrator_source_archive: str | Path,
    calibrator_source_revision: str,
    calibrator_source_archive_sha256: str,
    expected_calibrator_source_sha256: str,
    expected_materializer_source_sha256: str,
    output_dir: str | Path,
    acknowledge_reviewed_label_manifest: bool,
) -> dict[str, Any]:
    if acknowledge_reviewed_label_manifest is not True:
        raise TemporalCounterfactualMaterializationError(
            "--acknowledge-reviewed-label-manifest is required"
        )
    if score_group_file_sha256_authority not in {
        "presealed_external",
        "observed_afterok_runtime",
    }:
        raise TemporalCounterfactualMaterializationError(
            "score-group file hash authority differs"
        )
    calibrator_archive_path = _require_exact_authorities(
        expected_root_spec_sha256=expected_root_spec_sha256,
        expected_bank_receipt_sha256=expected_bank_receipt_sha256,
        expected_audit_bridge_sha256=expected_audit_bridge_sha256,
        expected_detached_label_manifest_sha256=(
            expected_detached_label_manifest_sha256
        ),
        expected_preregistration_sha256=expected_preregistration_sha256,
        calibrator_source_archive=calibrator_source_archive,
        calibrator_source_revision=calibrator_source_revision,
        calibrator_source_archive_sha256=calibrator_source_archive_sha256,
        expected_calibrator_source_sha256=expected_calibrator_source_sha256,
        expected_materializer_source_sha256=expected_materializer_source_sha256,
    )
    output = _fresh_output_directory(output_dir)
    root_spec_value, root_spec_path, _root_spec_file_sha = _read_strict_json(
        root_spec,
        label="formal root spec",
        expected_sha256=expected_root_spec_sha256,
    )
    if (
        root_spec_value.get("schema_version")
        != "pair-v5-frozen-bernini-t2v-calibration-bank-spec-v2"
    ):
        raise TemporalCounterfactualMaterializationError(
            "formal root spec schema differs"
        )
    bank_value, bank_receipt_path, _bank_file_sha = _read_strict_json(
        bank_receipt,
        label="formal bank receipt",
        expected_sha256=expected_bank_receipt_sha256,
    )
    if bank_value.get("receipt_digest") != contract.REQUIRED_CORE4_V2_BANK_RECEIPT_DIGEST:
        raise TemporalCounterfactualMaterializationError(
            "formal bank receipt digest differs"
        )

    groups, scores, group_bindings, log_bindings, resolved_score_root = (
        load_ordered_score_population(
            score_root=score_root,
            expected_group_file_sha256_by_id={
                "sp4-a": expected_sp4_a_score_group_sha256,
                "sp4-b": expected_sp4_b_score_group_sha256,
            },
        )
    )
    (
        bridge,
        bridge_path,
        bridge_file_sha,
        labels,
        label_path,
        label_file_sha,
        audits,
        audit_bindings,
        resolved_audit_root,
    ) = _validate_audit_authority(
        audit_bridge_receipt=audit_bridge_receipt,
        expected_audit_bridge_sha256=expected_audit_bridge_sha256,
        event_audit_root=event_audit_root,
        detached_label_manifest=detached_label_manifest,
        expected_detached_label_manifest_sha256=(
            expected_detached_label_manifest_sha256
        ),
        root_spec_path=root_spec_path,
        bank_receipt_path=bank_receipt_path,
        scores=scores,
    )
    prereg, prereg_path, prereg_file_sha = _read_strict_json(
        preregistration,
        label="frozen temporal preregistration",
        expected_sha256=expected_preregistration_sha256,
    )
    try:
        prereg = calibration.validate_preregistration(prereg)
    except calibration.TemporalCounterfactualCalibrationError as error:
        raise TemporalCounterfactualMaterializationError(str(error)) from error
    if prereg["preregistration_digest"] != REQUIRED_PREREGISTRATION_DIGEST:
        raise TemporalCounterfactualMaterializationError(
            "formal preregistration digest differs"
        )

    try:
        calibration_receipt = calibration.calibrate_temporal_counterfactual_scores(
            scores,
            audits,
            prereg,
            groups,
            source_bank_spec_sha256=expected_root_spec_sha256,
            source_bank_receipt_digest=contract.REQUIRED_CORE4_V2_BANK_RECEIPT_DIGEST,
            calibrator_source_revision=calibrator_source_revision,
            calibrator_source_archive_sha256=calibrator_source_archive_sha256,
            expected_calibrator_source_sha256=expected_calibrator_source_sha256,
        )
        calibration.validate_calibration_receipt(
            calibration_receipt,
            score_receipts=scores,
            event_audit_receipts=audits,
            preregistration=prereg,
            group_receipts=groups,
        )
    except calibration.TemporalCounterfactualCalibrationError as error:
        raise TemporalCounterfactualMaterializationError(str(error)) from error

    receipt = make_materialization_receipt(
        materializer_source_sha256=expected_materializer_source_sha256,
        root_spec_path=root_spec_path,
        bank_receipt_path=bank_receipt_path,
        score_root=resolved_score_root,
        score_log_bindings=log_bindings,
        score_group_file_sha256_authority=score_group_file_sha256_authority,
        score_group_bindings=group_bindings,
        bridge=bridge,
        bridge_path=bridge_path,
        bridge_file_sha256=bridge_file_sha,
        labels=labels,
        label_path=label_path,
        label_file_sha256=label_file_sha,
        audit_root=resolved_audit_root,
        audit_bindings=audit_bindings,
        preregistration_path=prereg_path,
        preregistration_file_sha256=prereg_file_sha,
        preregistration=prereg,
        calibration_receipt=calibration_receipt,
        calibrator_source_revision=calibrator_source_revision,
        calibrator_source_archive_path=calibrator_archive_path,
        calibrator_source_archive_sha256=calibrator_source_archive_sha256,
        expected_calibrator_source_sha256=expected_calibrator_source_sha256,
        output_dir=output,
    )
    validate_materialization_receipt(
        receipt,
        calibration_receipt=calibration_receipt,
        score_receipts=scores,
        event_audit_receipts=audits,
        preregistration=prereg,
        group_receipts=groups,
    )

    # No output exists until every input, calibration, and replay check passes.
    output.mkdir()
    prereg_output = output / PREREGISTRATION_FILENAME
    calibration_output = output / CALIBRATION_FILENAME
    receipt_output = output / MATERIALIZATION_FILENAME
    if _write_create_only(prereg_output, prereg) != receipt[
        "preregistration_output_binding"
    ]["file_sha256"]:
        raise TemporalCounterfactualMaterializationError(
            "written preregistration SHA-256 differs"
        )
    if _write_create_only(calibration_output, calibration_receipt) != receipt[
        "calibration_output_binding"
    ]["file_sha256"]:
        raise TemporalCounterfactualMaterializationError(
            "written calibration SHA-256 differs"
        )
    _write_create_only(receipt_output, receipt)

    if {entry.name for entry in output.iterdir()} != {
        PREREGISTRATION_FILENAME,
        CALIBRATION_FILENAME,
        MATERIALIZATION_FILENAME,
    }:
        raise TemporalCounterfactualMaterializationError(
            "written output filesystem closure differs"
        )
    written_prereg, _, _ = _read_strict_json(
        prereg_output,
        label="written preregistration",
        expected_sha256=receipt["preregistration_output_binding"]["file_sha256"],
    )
    written_calibration, _, _ = _read_strict_json(
        calibration_output,
        label="written calibration",
        expected_sha256=receipt["calibration_output_binding"]["file_sha256"],
    )
    written_receipt, _, _ = _read_strict_json(
        receipt_output,
        label="written materialization receipt",
    )
    if written_prereg != prereg or written_calibration != calibration_receipt:
        raise TemporalCounterfactualMaterializationError(
            "written preregistration/calibration semantics differ"
        )
    validate_materialization_receipt(
        written_receipt,
        calibration_receipt=written_calibration,
        score_receipts=scores,
        event_audit_receipts=audits,
        preregistration=written_prereg,
        group_receipts=groups,
    )
    os.chmod(output, 0o500)
    return receipt


def replay_materialized_output(
    *,
    root_spec: str | Path,
    expected_root_spec_sha256: str,
    bank_receipt: str | Path,
    expected_bank_receipt_sha256: str,
    score_root: str | Path,
    expected_sp4_a_score_group_sha256: str,
    expected_sp4_b_score_group_sha256: str,
    score_group_file_sha256_authority: str,
    audit_bridge_receipt: str | Path,
    expected_audit_bridge_sha256: str,
    event_audit_root: str | Path,
    detached_label_manifest: str | Path,
    expected_detached_label_manifest_sha256: str,
    preregistration: str | Path,
    expected_preregistration_sha256: str,
    calibrator_source_archive: str | Path,
    calibrator_source_revision: str,
    calibrator_source_archive_sha256: str,
    expected_calibrator_source_sha256: str,
    expected_materializer_source_sha256: str,
    output_dir: str | Path,
    acknowledge_reviewed_label_manifest: bool,
) -> dict[str, Any]:
    """Re-read every external input and reproduce an existing output exactly."""

    if acknowledge_reviewed_label_manifest is not True:
        raise TemporalCounterfactualMaterializationError(
            "--acknowledge-reviewed-label-manifest is required"
        )
    if score_group_file_sha256_authority not in {
        "presealed_external",
        "observed_afterok_runtime",
    }:
        raise TemporalCounterfactualMaterializationError(
            "score-group file hash authority differs"
        )
    calibrator_archive_path = _require_exact_authorities(
        expected_root_spec_sha256=expected_root_spec_sha256,
        expected_bank_receipt_sha256=expected_bank_receipt_sha256,
        expected_audit_bridge_sha256=expected_audit_bridge_sha256,
        expected_detached_label_manifest_sha256=(
            expected_detached_label_manifest_sha256
        ),
        expected_preregistration_sha256=expected_preregistration_sha256,
        calibrator_source_archive=calibrator_source_archive,
        calibrator_source_revision=calibrator_source_revision,
        calibrator_source_archive_sha256=calibrator_source_archive_sha256,
        expected_calibrator_source_sha256=expected_calibrator_source_sha256,
        expected_materializer_source_sha256=expected_materializer_source_sha256,
    )
    root_spec_value, root_spec_path, _ = _read_strict_json(
        root_spec,
        label="formal root spec",
        expected_sha256=expected_root_spec_sha256,
    )
    if (
        root_spec_value.get("schema_version")
        != "pair-v5-frozen-bernini-t2v-calibration-bank-spec-v2"
    ):
        raise TemporalCounterfactualMaterializationError(
            "formal root spec schema differs"
        )
    bank_value, bank_receipt_path, _ = _read_strict_json(
        bank_receipt,
        label="formal bank receipt",
        expected_sha256=expected_bank_receipt_sha256,
    )
    if bank_value.get("receipt_digest") != contract.REQUIRED_CORE4_V2_BANK_RECEIPT_DIGEST:
        raise TemporalCounterfactualMaterializationError(
            "formal bank receipt digest differs"
        )
    groups, scores, group_bindings, log_bindings, resolved_score_root = (
        load_ordered_score_population(
            score_root=score_root,
            expected_group_file_sha256_by_id={
                "sp4-a": expected_sp4_a_score_group_sha256,
                "sp4-b": expected_sp4_b_score_group_sha256,
            },
        )
    )
    (
        bridge,
        bridge_path,
        bridge_file_sha,
        labels,
        label_path,
        label_file_sha,
        audits,
        audit_bindings,
        resolved_audit_root,
    ) = _validate_audit_authority(
        audit_bridge_receipt=audit_bridge_receipt,
        expected_audit_bridge_sha256=expected_audit_bridge_sha256,
        event_audit_root=event_audit_root,
        detached_label_manifest=detached_label_manifest,
        expected_detached_label_manifest_sha256=(
            expected_detached_label_manifest_sha256
        ),
        root_spec_path=root_spec_path,
        bank_receipt_path=bank_receipt_path,
        scores=scores,
    )
    prereg, prereg_path, prereg_file_sha = _read_strict_json(
        preregistration,
        label="frozen temporal preregistration",
        expected_sha256=expected_preregistration_sha256,
    )
    try:
        prereg = calibration.validate_preregistration(prereg)
    except calibration.TemporalCounterfactualCalibrationError as error:
        raise TemporalCounterfactualMaterializationError(str(error)) from error
    if prereg["preregistration_digest"] != REQUIRED_PREREGISTRATION_DIGEST:
        raise TemporalCounterfactualMaterializationError(
            "formal preregistration digest differs"
        )

    output = _plain_directory(output_dir, label="materialized output directory")
    expected_names = {
        PREREGISTRATION_FILENAME,
        CALIBRATION_FILENAME,
        MATERIALIZATION_FILENAME,
    }
    if {entry.name for entry in output.iterdir()} != expected_names:
        raise TemporalCounterfactualMaterializationError(
            "materialized output filesystem closure differs"
        )
    written_prereg, _, _ = _read_strict_json(
        output / PREREGISTRATION_FILENAME,
        label="materialized preregistration",
    )
    written_calibration, _, _ = _read_strict_json(
        output / CALIBRATION_FILENAME,
        label="materialized calibration",
    )
    written_receipt, _, _ = _read_strict_json(
        output / MATERIALIZATION_FILENAME,
        label="materialization receipt",
    )
    if written_prereg != prereg:
        raise TemporalCounterfactualMaterializationError(
            "materialized preregistration differs from frozen input"
        )
    try:
        reproduced_calibration = calibration.calibrate_temporal_counterfactual_scores(
            scores,
            audits,
            prereg,
            groups,
            source_bank_spec_sha256=expected_root_spec_sha256,
            source_bank_receipt_digest=contract.REQUIRED_CORE4_V2_BANK_RECEIPT_DIGEST,
            calibrator_source_revision=calibrator_source_revision,
            calibrator_source_archive_sha256=calibrator_source_archive_sha256,
            expected_calibrator_source_sha256=expected_calibrator_source_sha256,
        )
        calibration.validate_calibration_receipt(
            written_calibration,
            score_receipts=scores,
            event_audit_receipts=audits,
            preregistration=prereg,
            group_receipts=groups,
        )
    except calibration.TemporalCounterfactualCalibrationError as error:
        raise TemporalCounterfactualMaterializationError(str(error)) from error
    if written_calibration != reproduced_calibration:
        raise TemporalCounterfactualMaterializationError(
            "materialized calibration does not exactly reproduce"
        )
    expected_receipt = make_materialization_receipt(
        materializer_source_sha256=expected_materializer_source_sha256,
        root_spec_path=root_spec_path,
        bank_receipt_path=bank_receipt_path,
        score_root=resolved_score_root,
        score_log_bindings=log_bindings,
        score_group_file_sha256_authority=score_group_file_sha256_authority,
        score_group_bindings=group_bindings,
        bridge=bridge,
        bridge_path=bridge_path,
        bridge_file_sha256=bridge_file_sha,
        labels=labels,
        label_path=label_path,
        label_file_sha256=label_file_sha,
        audit_root=resolved_audit_root,
        audit_bindings=audit_bindings,
        preregistration_path=prereg_path,
        preregistration_file_sha256=prereg_file_sha,
        preregistration=prereg,
        calibration_receipt=written_calibration,
        calibrator_source_revision=calibrator_source_revision,
        calibrator_source_archive_path=calibrator_archive_path,
        calibrator_source_archive_sha256=calibrator_source_archive_sha256,
        expected_calibrator_source_sha256=expected_calibrator_source_sha256,
        output_dir=output,
    )
    if written_receipt != expected_receipt:
        raise TemporalCounterfactualMaterializationError(
            "materialization receipt does not exactly reproduce"
        )
    return validate_materialization_receipt(
        written_receipt,
        calibration_receipt=written_calibration,
        score_receipts=scores,
        event_audit_receipts=audits,
        preregistration=prereg,
        group_receipts=groups,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-spec", required=True)
    parser.add_argument("--expected-root-spec-sha256", required=True)
    parser.add_argument("--bank-receipt", required=True)
    parser.add_argument("--expected-bank-receipt-sha256", required=True)
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--expected-sp4-a-score-group-sha256", required=True)
    parser.add_argument("--expected-sp4-b-score-group-sha256", required=True)
    parser.add_argument(
        "--score-group-file-sha256-authority",
        choices=("presealed_external", "observed_afterok_runtime"),
        required=True,
    )
    parser.add_argument("--audit-bridge-receipt", required=True)
    parser.add_argument("--expected-audit-bridge-sha256", required=True)
    parser.add_argument("--event-audit-root", required=True)
    parser.add_argument("--detached-label-manifest", required=True)
    parser.add_argument("--expected-detached-label-manifest-sha256", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--calibrator-source-archive", required=True)
    parser.add_argument("--calibrator-source-revision", required=True)
    parser.add_argument("--calibrator-source-archive-sha256", required=True)
    parser.add_argument("--expected-calibrator-source-sha256", required=True)
    parser.add_argument("--expected-materializer-source-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--acknowledge-reviewed-label-manifest", action="store_true")
    parser.add_argument("--verify-existing-output", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    operation = (
        replay_materialized_output
        if args.verify_existing_output
        else materialize
    )
    receipt = operation(
        root_spec=args.root_spec,
        expected_root_spec_sha256=args.expected_root_spec_sha256,
        bank_receipt=args.bank_receipt,
        expected_bank_receipt_sha256=args.expected_bank_receipt_sha256,
        score_root=args.score_root,
        expected_sp4_a_score_group_sha256=(
            args.expected_sp4_a_score_group_sha256
        ),
        expected_sp4_b_score_group_sha256=(
            args.expected_sp4_b_score_group_sha256
        ),
        score_group_file_sha256_authority=(
            args.score_group_file_sha256_authority
        ),
        audit_bridge_receipt=args.audit_bridge_receipt,
        expected_audit_bridge_sha256=args.expected_audit_bridge_sha256,
        event_audit_root=args.event_audit_root,
        detached_label_manifest=args.detached_label_manifest,
        expected_detached_label_manifest_sha256=(
            args.expected_detached_label_manifest_sha256
        ),
        preregistration=args.preregistration,
        expected_preregistration_sha256=args.expected_preregistration_sha256,
        calibrator_source_archive=args.calibrator_source_archive,
        calibrator_source_revision=args.calibrator_source_revision,
        calibrator_source_archive_sha256=(
            args.calibrator_source_archive_sha256
        ),
        expected_calibrator_source_sha256=(
            args.expected_calibrator_source_sha256
        ),
        expected_materializer_source_sha256=(
            args.expected_materializer_source_sha256
        ),
        output_dir=args.output_dir,
        acknowledge_reviewed_label_manifest=(
            args.acknowledge_reviewed_label_manifest
        ),
    )
    print(
        json.dumps(
            {
                "candidate_count": receipt["candidate_count"],
                "optimizer_authorized": receipt["optimizer_authorized"],
                "failure_reasons": receipt["failure_reasons"],
                "materialization_receipt_digest": receipt["receipt_digest"],
                "training_performed": False,
                "operation": (
                    "verify_existing_output"
                    if args.verify_existing_output
                    else "materialize"
                ),
                "output_dir": str(Path(args.output_dir)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CALIBRATION_FILENAME",
    "MATERIALIZATION_FILENAME",
    "MATERIALIZATION_SCHEMA",
    "PREREGISTRATION_FILENAME",
    "TemporalCounterfactualMaterializationError",
    "build_parser",
    "load_ordered_score_population",
    "make_materialization_receipt",
    "materialize",
    "replay_materialized_output",
    "validate_materialization_receipt",
]
