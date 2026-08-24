#!/usr/bin/env python3
"""Author a detached event-label manifest for the sealed core4-v2 T2V bank.

The ``template`` command copies identity and provenance from the authenticated
40-candidate bank, but deliberately leaves all event judgments as JSON null.
The ``seal`` command accepts only a fully completed template and requires four
explicit acknowledgements.  It never derives a label from ``semantic_branch``.

The external audit artifact may be a manual annotation sidecar or the sealed
output of a VLM audit.  Its bytes are hash-bound, but its contents are not
parsed and may never enter the scorer, calibrator, or model condition.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_t2v_calibration_bank_spec as bank_contract  # noqa: E402
import pair_v5_t2v_energy_calibration_v3 as calibration  # noqa: E402


TEMPLATE_SCHEMA = "bernini-pair-v5-core4-v2-event-label-authoring-template-v3"
LABEL_MANIFEST_SCHEMA = "bernini-pair-v5-core4-v2-detached-event-label-manifest-v3"
CORE4_PREFIX = "pair5-t2v-core4-v2-"
CORE4_GROUPS = ("sp4-a", "sp4-b")
CORE4_GROUP_SIZE = 20
CORE4_CANDIDATE_COUNT = 40
CORE4_CELL_COUNT = 4

LABEL_BOOLEAN_FIELDS = (
    "complete_target_transition_observed",
    "terminal_hold_observed",
    "full_target_action_observed",
    "full_target_action_false_confirmed",
)
ACKNOWLEDGEMENT_FIELDS = (
    "all_40_rows_individually_reviewed",
    "labels_not_inferred_from_semantic_branch",
    "ambiguous_rows_left_ambiguous_not_guessed",
    "audit_artifacts_are_detached_and_never_model_conditioning",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TEMPLATE_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "root_spec_raw_sha256",
        "bank_receipt_digest",
        "candidate_count",
        "candidate_order",
        "rows",
        "author_acknowledgements",
        "template_not_a_sealed_label_manifest",
    }
)
_TEMPLATE_ROW_FIELDS = frozenset(
    {
        "ordinal",
        "group_id",
        "candidate_id",
        "analysis_split",
        "action_family_id",
        "calibration_group_id",
        "actor_group_id",
        "scene_group_id",
        "action_group_id",
        "semantic_branch",
        "full_t2v_caption",
        "generation_receipt_digest",
        "review_media_path",
        "review_media_sha256",
        "audit_source_kind",
        "external_audit_artifact_path",
        "external_audit_artifact_sha256",
        *LABEL_BOOLEAN_FIELDS,
        "annotation_complete",
    }
)
_LABEL_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "root_spec_raw_sha256",
        "bank_receipt_digest",
        "candidate_count",
        "candidate_order",
        "rows",
        "author_acknowledgements",
        "labels_are_external_and_detached",
        "labels_may_enter_model_condition",
        "ambiguity_fails_calibration_closed",
        "manifest_digest",
    }
)
_LABEL_ROW_FIELDS = frozenset(
    {
        "ordinal",
        "group_id",
        "candidate_id",
        "analysis_split",
        "action_family_id",
        "calibration_group_id",
        "actor_group_id",
        "scene_group_id",
        "action_group_id",
        "semantic_branch",
        "generation_receipt_digest",
        "audit_source_kind",
        "external_audit_artifact_path",
        "external_audit_artifact_sha256",
        *LABEL_BOOLEAN_FIELDS,
    }
)


class PairV5Core4LabelAuthoringError(RuntimeError):
    """The bank, authoring draft, or detached label seal failed closed."""


def _scorer_runtime() -> Any:
    """Load the Torch-bearing verifier only for real bank authentication."""

    try:
        return importlib.import_module("score_pair_v5_t2v_energy_bank_v3")
    except (ImportError, ModuleNotFoundError) as error:
        raise PairV5Core4LabelAuthoringError(
            "real bank authentication requires the pinned vace/Torch runtime"
        ) from error


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
        raise PairV5Core4LabelAuthoringError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairV5Core4LabelAuthoringError(f"{label} must be lowercase SHA-256")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PairV5Core4LabelAuthoringError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _fresh_output_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or path == Path("/")
        or path.exists()
        or path.is_symlink()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
    ):
        raise PairV5Core4LabelAuthoringError(
            f"{label} must be a fresh absolute file under an existing plain directory"
        )
    return path


def _reject_constant(token: str) -> None:
    raise PairV5Core4LabelAuthoringError(f"non-finite JSON constant is forbidden: {token}")


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PairV5Core4LabelAuthoringError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json_file(
    value: str | Path, *, expected_sha256: str, label: str
) -> tuple[dict[str, Any], Path, str]:
    path = _plain_file(value, label=label)
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != _sha256(expected_sha256, label=f"{label} expected SHA-256"):
        raise PairV5Core4LabelAuthoringError(f"{label} SHA-256 differs")
    try:
        decoded = json.loads(
            raw.decode("ascii"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PairV5Core4LabelAuthoringError(f"{label} is invalid ASCII JSON") from error
    if not isinstance(decoded, dict):
        raise PairV5Core4LabelAuthoringError(f"{label} root must be an object")
    return decoded, path, observed


def _write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise PairV5Core4LabelAuthoringError(f"refusing to overwrite {path}")
    raw = canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    os.chmod(path, 0o400)
    return hashlib.sha256(raw).hexdigest()


def _candidate_identity(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: candidate[key]
        for key in (
            "candidate_id",
            "analysis_split",
            "action_family_id",
            "calibration_group_id",
            "actor_group_id",
            "scene_group_id",
            "action_group_id",
            "semantic_branch",
        )
    }


def validate_core4_bound_rows(
    spec: Mapping[str, Any],
    bank: Mapping[str, Any],
    rows_by_group: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_root_spec_sha256: str,
) -> list[dict[str, Any]]:
    """Enforce the exact core4-v2 topology and return sealed spec order."""

    expected_spec_sha = _sha256(
        expected_root_spec_sha256, label="root spec SHA-256"
    )
    if (
        spec.get("schema_version") != bank_contract.SCHEMA_VERSION_V2
        or list(rows_by_group) != list(CORE4_GROUPS)
        or [group.get("group_id") for group in spec.get("groups", [])]
        != list(CORE4_GROUPS)
        or [len(group.get("candidates", [])) for group in spec.get("groups", [])]
        != [CORE4_GROUP_SIZE, CORE4_GROUP_SIZE]
        or bank.get("root_spec_raw_sha256") != expected_spec_sha
        or bank.get("candidate_count") != CORE4_CANDIDATE_COUNT
        or bank.get("cell_count") != CORE4_CELL_COUNT
    ):
        raise PairV5Core4LabelAuthoringError("input is not the sealed core4-v2 bank")

    flattened: list[dict[str, Any]] = []
    cell_ids: set[str] = set()
    for group in spec["groups"]:
        group_id = group["group_id"]
        bound_rows = list(rows_by_group[group_id])
        if len(bound_rows) != CORE4_GROUP_SIZE:
            raise PairV5Core4LabelAuthoringError(f"{group_id} row count differs")
        for expected_candidate, bound in zip(group["candidates"], bound_rows):
            candidate = bound.get("candidate")
            if (
                not isinstance(candidate, Mapping)
                or dict(candidate) != dict(expected_candidate)
                or not candidate["candidate_id"].startswith(CORE4_PREFIX)
            ):
                raise PairV5Core4LabelAuthoringError(
                    f"{group_id} candidate identity/order differs"
                )
            cell_ids.add(candidate["calibration_group_id"])
            flattened.append({"group_id": group_id, **dict(bound)})
    if (
        len(flattened) != CORE4_CANDIDATE_COUNT
        or len({row["candidate"]["candidate_id"] for row in flattened})
        != CORE4_CANDIDATE_COUNT
        or len(cell_ids) != CORE4_CELL_COUNT
    ):
        raise PairV5Core4LabelAuthoringError("core4-v2 candidate/cell closure differs")
    return flattened


def load_core4_bound_bank(
    *,
    root_spec: str | Path,
    root_spec_sha256: str,
    bank_output_dir: str | Path,
    bank_receipt: str | Path,
    bank_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Authenticate all generation receipts/artifacts in the two SP4 groups."""

    rows_by_group: dict[str, list[dict[str, Any]]] = {}
    common_spec: dict[str, Any] | None = None
    common_bank: dict[str, Any] | None = None
    scorer = _scorer_runtime()
    for group_id in CORE4_GROUPS:
        try:
            spec, bank, rows = scorer.load_group_bank(
                root_spec=root_spec,
                root_spec_sha256=root_spec_sha256,
                bank_output_dir=bank_output_dir,
                bank_receipt=bank_receipt,
                bank_receipt_sha256=bank_receipt_sha256,
                group_id=group_id,
            )
        except scorer.PairV5T2VEnergyScoringError as error:
            raise PairV5Core4LabelAuthoringError(str(error)) from error
        if common_spec is not None and spec != common_spec:
            raise PairV5Core4LabelAuthoringError("two group root specs differ")
        if common_bank is not None and (
            bank["receipt_digest"] != common_bank["receipt_digest"]
            or bank["file_sha256"] != common_bank["file_sha256"]
        ):
            raise PairV5Core4LabelAuthoringError("two group bank bindings differ")
        common_spec = spec
        common_bank = bank
        rows_by_group[group_id] = rows
    assert common_spec is not None and common_bank is not None
    flattened = validate_core4_bound_rows(
        common_spec,
        common_bank,
        rows_by_group,
        expected_root_spec_sha256=root_spec_sha256,
    )
    return common_spec, common_bank, flattened


def make_authoring_template(
    *,
    root_spec_raw_sha256: str,
    bank_receipt_digest: str,
    bound_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(bound_rows) != CORE4_CANDIDATE_COUNT:
        raise PairV5Core4LabelAuthoringError("template requires exactly 40 candidates")
    rows: list[dict[str, Any]] = []
    for ordinal, bound in enumerate(bound_rows):
        candidate = bound["candidate"]
        artifact = bound["artifacts"]["mp4"]
        rows.append(
            {
                "ordinal": ordinal,
                "group_id": bound["group_id"],
                **_candidate_identity(candidate),
                "full_t2v_caption": candidate["full_t2v_caption"],
                "generation_receipt_digest": bound["generation_receipt_digest"],
                "review_media_path": artifact["path"],
                "review_media_sha256": artifact["sha256"],
                "audit_source_kind": None,
                "external_audit_artifact_path": None,
                "external_audit_artifact_sha256": None,
                **{name: None for name in LABEL_BOOLEAN_FIELDS},
                "annotation_complete": False,
            }
        )
    return {
        "schema_version": TEMPLATE_SCHEMA,
        "root_spec_raw_sha256": _sha256(
            root_spec_raw_sha256, label="root spec SHA-256"
        ),
        "bank_receipt_digest": _sha256(
            bank_receipt_digest, label="bank receipt digest"
        ),
        "candidate_count": CORE4_CANDIDATE_COUNT,
        "candidate_order": [row["candidate_id"] for row in rows],
        "rows": rows,
        "author_acknowledgements": {
            name: False for name in ACKNOWLEDGEMENT_FIELDS
        },
        "template_not_a_sealed_label_manifest": True,
    }


def validate_authoring_template(
    value: Any,
    *,
    root_spec_raw_sha256: str,
    bank_receipt_digest: str,
    bound_rows: Sequence[Mapping[str, Any]],
    require_complete: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_TEMPLATE_ROOT_FIELDS):
        raise PairV5Core4LabelAuthoringError("authoring template root fields differ")
    root = dict(value)
    expected_order = [row["candidate"]["candidate_id"] for row in bound_rows]
    if (
        root["schema_version"] != TEMPLATE_SCHEMA
        or root["root_spec_raw_sha256"] != root_spec_raw_sha256
        or root["bank_receipt_digest"] != bank_receipt_digest
        or root["candidate_count"] != CORE4_CANDIDATE_COUNT
        or root["candidate_order"] != expected_order
        or root["template_not_a_sealed_label_manifest"] is not True
        or not isinstance(root["rows"], list)
        or len(root["rows"]) != CORE4_CANDIDATE_COUNT
        or root["author_acknowledgements"]
        != {name: False for name in ACKNOWLEDGEMENT_FIELDS}
    ):
        raise PairV5Core4LabelAuthoringError("authoring template binding differs")

    for ordinal, (raw_row, bound) in enumerate(zip(root["rows"], bound_rows)):
        if not isinstance(raw_row, Mapping) or set(raw_row) != set(_TEMPLATE_ROW_FIELDS):
            raise PairV5Core4LabelAuthoringError(f"template row {ordinal} fields differ")
        row = dict(raw_row)
        candidate = bound["candidate"]
        expected_identity = {
            "ordinal": ordinal,
            "group_id": bound["group_id"],
            **_candidate_identity(candidate),
            "full_t2v_caption": candidate["full_t2v_caption"],
            "generation_receipt_digest": bound["generation_receipt_digest"],
            "review_media_path": bound["artifacts"]["mp4"]["path"],
            "review_media_sha256": bound["artifacts"]["mp4"]["sha256"],
        }
        if any(row[name] != expected for name, expected in expected_identity.items()):
            raise PairV5Core4LabelAuthoringError(
                f"template row {ordinal} identity/provenance differs"
            )
        if require_complete:
            if row["annotation_complete"] is not True:
                raise PairV5Core4LabelAuthoringError(
                    f"template row {ordinal} annotation is incomplete"
                )
            if row["audit_source_kind"] not in calibration.AUDIT_SOURCE_KINDS:
                raise PairV5Core4LabelAuthoringError(
                    f"template row {ordinal} audit source differs"
                )
            for name in LABEL_BOOLEAN_FIELDS:
                if type(row[name]) is not bool:
                    raise PairV5Core4LabelAuthoringError(
                        f"template row {ordinal} {name} is not an explicit boolean"
                    )
            if (
                row["full_target_action_observed"]
                and row["full_target_action_false_confirmed"]
            ):
                raise PairV5Core4LabelAuthoringError(
                    f"template row {ordinal} action is both observed and false"
                )
            artifact = _plain_file(
                row["external_audit_artifact_path"],
                label=f"template row {ordinal} external audit artifact",
            )
            if file_sha256(artifact) != _sha256(
                row["external_audit_artifact_sha256"],
                label=f"template row {ordinal} audit artifact SHA-256",
            ):
                raise PairV5Core4LabelAuthoringError(
                    f"template row {ordinal} external audit artifact changed"
                )
        else:
            for name in (
                "audit_source_kind",
                "external_audit_artifact_path",
                "external_audit_artifact_sha256",
                *LABEL_BOOLEAN_FIELDS,
            ):
                if row[name] is not None:
                    raise PairV5Core4LabelAuthoringError(
                        f"fresh template row {ordinal} silently contains {name}"
                    )
            if row["annotation_complete"] is not False:
                raise PairV5Core4LabelAuthoringError(
                    f"fresh template row {ordinal} claims completion"
                )
    return root


def seal_label_manifest(
    *,
    completed_template: Mapping[str, Any],
    root_spec_raw_sha256: str,
    bank_receipt_digest: str,
    bound_rows: Sequence[Mapping[str, Any]],
    acknowledgements: Mapping[str, bool],
) -> dict[str, Any]:
    expected_acknowledgements = {name: True for name in ACKNOWLEDGEMENT_FIELDS}
    if dict(acknowledgements) != expected_acknowledgements:
        raise PairV5Core4LabelAuthoringError(
            "all detached-label author acknowledgements are required"
        )
    template = validate_authoring_template(
        completed_template,
        root_spec_raw_sha256=root_spec_raw_sha256,
        bank_receipt_digest=bank_receipt_digest,
        bound_rows=bound_rows,
        require_complete=True,
    )
    rows = [
        {
            name: row[name]
            for name in _LABEL_ROW_FIELDS
        }
        for row in template["rows"]
    ]
    unsigned = {
        "schema_version": LABEL_MANIFEST_SCHEMA,
        "root_spec_raw_sha256": root_spec_raw_sha256,
        "bank_receipt_digest": bank_receipt_digest,
        "candidate_count": CORE4_CANDIDATE_COUNT,
        "candidate_order": list(template["candidate_order"]),
        "rows": rows,
        "author_acknowledgements": expected_acknowledgements,
        "labels_are_external_and_detached": True,
        "labels_may_enter_model_condition": False,
        "ambiguity_fails_calibration_closed": True,
    }
    return {**unsigned, "manifest_digest": object_sha256(unsigned)}


def validate_label_manifest(
    value: Any,
    *,
    root_spec_raw_sha256: str,
    bank_receipt_digest: str,
    bound_rows: Sequence[Mapping[str, Any]],
    verify_external_artifacts: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_LABEL_ROOT_FIELDS):
        raise PairV5Core4LabelAuthoringError("detached label manifest fields differ")
    root = dict(value)
    unsigned = dict(root)
    declared = _sha256(unsigned.pop("manifest_digest"), label="manifest digest")
    expected_order = [row["candidate"]["candidate_id"] for row in bound_rows]
    if (
        object_sha256(unsigned) != declared
        or root["schema_version"] != LABEL_MANIFEST_SCHEMA
        or root["root_spec_raw_sha256"] != root_spec_raw_sha256
        or root["bank_receipt_digest"] != bank_receipt_digest
        or root["candidate_count"] != CORE4_CANDIDATE_COUNT
        or root["candidate_order"] != expected_order
        or root["author_acknowledgements"]
        != {name: True for name in ACKNOWLEDGEMENT_FIELDS}
        or root["labels_are_external_and_detached"] is not True
        or root["labels_may_enter_model_condition"] is not False
        or root["ambiguity_fails_calibration_closed"] is not True
        or not isinstance(root["rows"], list)
        or len(root["rows"]) != CORE4_CANDIDATE_COUNT
    ):
        raise PairV5Core4LabelAuthoringError("detached label manifest binding differs")
    for ordinal, (raw_row, bound) in enumerate(zip(root["rows"], bound_rows)):
        if not isinstance(raw_row, Mapping) or set(raw_row) != set(_LABEL_ROW_FIELDS):
            raise PairV5Core4LabelAuthoringError(f"label row {ordinal} fields differ")
        row = dict(raw_row)
        expected_identity = {
            "ordinal": ordinal,
            "group_id": bound["group_id"],
            **_candidate_identity(bound["candidate"]),
            "generation_receipt_digest": bound["generation_receipt_digest"],
        }
        if any(row[name] != expected for name, expected in expected_identity.items()):
            raise PairV5Core4LabelAuthoringError(
                f"label row {ordinal} identity/order differs"
            )
        if row["audit_source_kind"] not in calibration.AUDIT_SOURCE_KINDS:
            raise PairV5Core4LabelAuthoringError(f"label row {ordinal} audit source differs")
        for name in LABEL_BOOLEAN_FIELDS:
            if type(row[name]) is not bool:
                raise PairV5Core4LabelAuthoringError(
                    f"label row {ordinal} {name} must be boolean"
                )
        if row["full_target_action_observed"] and row[
            "full_target_action_false_confirmed"
        ]:
            raise PairV5Core4LabelAuthoringError(
                f"label row {ordinal} action is both observed and false"
            )
        artifact_sha = _sha256(
            row["external_audit_artifact_sha256"],
            label=f"label row {ordinal} external artifact SHA-256",
        )
        if verify_external_artifacts:
            artifact = _plain_file(
                row["external_audit_artifact_path"],
                label=f"label row {ordinal} external audit artifact",
            )
            if file_sha256(artifact) != artifact_sha:
                raise PairV5Core4LabelAuthoringError(
                    f"label row {ordinal} external audit artifact changed"
                )
    return root


def load_label_manifest(
    value: str | Path,
    *,
    expected_sha256: str,
    root_spec_raw_sha256: str,
    bank_receipt_digest: str,
    bound_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], Path, str]:
    root, path, observed = _strict_json_file(
        value,
        expected_sha256=expected_sha256,
        label="detached event-label manifest",
    )
    checked = validate_label_manifest(
        root,
        root_spec_raw_sha256=root_spec_raw_sha256,
        bank_receipt_digest=bank_receipt_digest,
        bound_rows=bound_rows,
        verify_external_artifacts=True,
    )
    return checked, path, observed


def _common_bank_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root-spec", required=True)
    parser.add_argument("--expected-root-spec-sha256", required=True)
    parser.add_argument("--bank-output-dir", required=True)
    parser.add_argument("--bank-receipt", required=True)
    parser.add_argument("--expected-bank-receipt-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    template = subparsers.add_parser(
        "template", description="Create an unlabelled exact-40 authoring template."
    )
    _common_bank_arguments(template)
    template.add_argument("--output", required=True)

    seal = subparsers.add_parser(
        "seal", description="Seal a completely and explicitly labelled template."
    )
    _common_bank_arguments(seal)
    seal.add_argument("--completed-template", required=True)
    seal.add_argument("--expected-completed-template-sha256", required=True)
    seal.add_argument("--output", required=True)
    seal.add_argument("--ack-all-40-rows-individually-reviewed", action="store_true")
    seal.add_argument("--ack-no-semantic-branch-defaults", action="store_true")
    seal.add_argument("--ack-ambiguity-left-unresolved", action="store_true")
    seal.add_argument("--ack-detached-artifacts-never-model-conditioning", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _spec, bank, rows = load_core4_bound_bank(
        root_spec=args.root_spec,
        root_spec_sha256=args.expected_root_spec_sha256,
        bank_output_dir=args.bank_output_dir,
        bank_receipt=args.bank_receipt,
        bank_receipt_sha256=args.expected_bank_receipt_sha256,
    )
    output = _fresh_output_file(args.output, label="author output")
    if args.command == "template":
        value = make_authoring_template(
            root_spec_raw_sha256=args.expected_root_spec_sha256,
            bank_receipt_digest=bank["receipt_digest"],
            bound_rows=rows,
        )
        # Prove the program did not pre-fill any judgment.
        validate_authoring_template(
            value,
            root_spec_raw_sha256=args.expected_root_spec_sha256,
            bank_receipt_digest=bank["receipt_digest"],
            bound_rows=rows,
            require_complete=False,
        )
    else:
        template, _path, _sha = _strict_json_file(
            args.completed_template,
            expected_sha256=args.expected_completed_template_sha256,
            label="completed label template",
        )
        acknowledgements = {
            "all_40_rows_individually_reviewed": args.ack_all_40_rows_individually_reviewed,
            "labels_not_inferred_from_semantic_branch": args.ack_no_semantic_branch_defaults,
            "ambiguous_rows_left_ambiguous_not_guessed": args.ack_ambiguity_left_unresolved,
            "audit_artifacts_are_detached_and_never_model_conditioning": (
                args.ack_detached_artifacts_never_model_conditioning
            ),
        }
        value = seal_label_manifest(
            completed_template=template,
            root_spec_raw_sha256=args.expected_root_spec_sha256,
            bank_receipt_digest=bank["receipt_digest"],
            bound_rows=rows,
            acknowledgements=acknowledgements,
        )
        validate_label_manifest(
            value,
            root_spec_raw_sha256=args.expected_root_spec_sha256,
            bank_receipt_digest=bank["receipt_digest"],
            bound_rows=rows,
            verify_external_artifacts=True,
        )
    observed = _write_create_only(output, value)
    print(
        json.dumps(
            {
                "command": args.command,
                "output": str(output),
                "file_sha256": observed,
                "candidate_count": CORE4_CANDIDATE_COUNT,
                "labels_invented_by_author_tool": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACKNOWLEDGEMENT_FIELDS",
    "CORE4_CANDIDATE_COUNT",
    "CORE4_GROUPS",
    "LABEL_BOOLEAN_FIELDS",
    "LABEL_MANIFEST_SCHEMA",
    "PairV5Core4LabelAuthoringError",
    "TEMPLATE_SCHEMA",
    "load_core4_bound_bank",
    "load_label_manifest",
    "make_authoring_template",
    "seal_label_manifest",
    "validate_authoring_template",
    "validate_core4_bound_rows",
    "validate_label_manifest",
]
