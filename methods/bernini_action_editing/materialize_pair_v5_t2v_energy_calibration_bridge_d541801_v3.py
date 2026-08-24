#!/usr/bin/env python3
"""Bridge the completed d541801 core4-v2 T2V v3 scores to calibration.

This is deliberately a receipt/scalar bridge, not a model program.  Before
calibration it authenticates the sealed root and rendered bank, both ordered
SP4 score-group receipts, all 40 generation/score joins, and one separately
sealed manual/VLM label row per candidate.  It then passes only 40 scalar
score rows, 40 detached-boolean event receipts, and one preregistration to
``pair_v5_t2v_energy_calibration_v3``.  This uniquely named implementation is
the immutable migration boundary for formal job 131177.  It accepts only the
v3 score/group schemas and only group receipts whose method source revision is
``d541801a162796aacde34c2bfc2b1f0472d954d2``; later v4 experiments cannot
silently change the meaning of the already completed evidence.

No video, path, tensor, latent, Gaussian, model, prompt, or label artifact
content enters the calibrator.  Ambiguity is preserved and produces a NO-GO
through the existing event/fit/confirmation gates.  This bridge never adds a
new optimizer authorization rule and never upgrades calibration evidence to
an action-editing claim.
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


METHOD_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = METHOD_ROOT / "tools"
for search_root in (METHOD_ROOT, TOOLS_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import author_pair_v5_core4_event_labels_d541801_v3 as label_author  # noqa: E402
import pair_v5_t2v_energy_calibration_v3 as calibration  # noqa: E402


BRIDGE_SCHEMA = (
    "bernini-pair-v5-core4-v2-energy-calibration-bridge-d541801-v3-receipt-v1"
)
CALIBRATOR_ID = "pair5-t2v-core4-v2-d541801-global-energy-v3"
REQUIRED_SCORER_SOURCE_REVISION = "d541801a162796aacde34c2bfc2b1f0472d954d2"
REQUIRED_SCORE_RECEIPT_SCHEMA = "bernini-pair-v5-frozen-t2v-global-energy-score-v3"
REQUIRED_GROUP_RECEIPT_SCHEMA = "bernini-pair-v5-frozen-t2v-global-energy-group-v3"
SCORE_FILENAME = "pair-v5-t2v-global-energy-score-v3.json"
GROUP_FILENAME = "pair-v5-t2v-global-energy-{group_id}-v3.json"
BRIDGE_RECEIPT_FILENAME = (
    "pair-v5-t2v-energy-calibration-bridge-d541801-v3-receipt.json"
)
GROUP_IDS = label_author.CORE4_GROUPS
GROUP_SIZE = label_author.CORE4_GROUP_SIZE
CANDIDATE_COUNT = label_author.CORE4_CANDIDATE_COUNT

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GROUP_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "group_id",
        "root_spec_raw_sha256",
        "bank_receipt_digest",
        "frozen_checkpoint_receipt_digest",
        "checkpoint_content_binding",
        "schedule_coordinate",
        "candidate_count",
        "candidate_receipt_digests",
        "primary_score_field",
        "phase_conjunctive_role",
        "input_closure",
        "training_performed",
        "optimizer_authorized",
        "scientific_action_editing_claim",
        "method_source_revision",
        "method_source_archive_sha256",
        "bernini_revision",
        "veomni_revision",
        "receipt_digest",
    }
)
_BRIDGE_FIELDS = frozenset(
    {
        "schema_version",
        "required_frozen_scorer_source_revision",
        "source_root_spec",
        "source_bank_receipt",
        "detached_event_label_manifest",
        "score_group_receipts",
        "candidate_count",
        "candidate_order",
        "generation_bindings",
        "score_receipt_bindings",
        "event_audit_receipt_bindings",
        "score_row_bindings",
        "preregistration_binding",
        "calibration_binding",
        "calibrator_input_contract",
        "ambiguous_candidate_ids",
        "optimizer_authorized",
        "optimizer_authorization_source",
        "confirmation_samples_consumed_by_optimizer",
        "training_performed",
        "scientific_action_editing_claim",
        "receipt_digest",
    }
)


class PairV5Core4CalibrationBridgeError(RuntimeError):
    """A sealed generation, score, label, or output binding failed closed."""


def _scorer_runtime() -> Any:
    """Load the Torch-bearing native scorer verifier only when it is needed."""

    try:
        runtime = importlib.import_module("score_pair_v5_t2v_energy_bank_v3")
    except (ImportError, ModuleNotFoundError) as error:
        raise PairV5Core4CalibrationBridgeError(
            "real score authentication requires the pinned vace/Torch runtime"
        ) from error
    if (
        getattr(runtime, "SCORE_RECEIPT_SCHEMA", None)
        != REQUIRED_SCORE_RECEIPT_SCHEMA
        or getattr(runtime, "GROUP_RECEIPT_SCHEMA", None)
        != REQUIRED_GROUP_RECEIPT_SCHEMA
    ):
        raise PairV5Core4CalibrationBridgeError(
            "this bridge requires the sealed d541801 v3 scorer runtime"
        )
    return runtime


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
        raise PairV5Core4CalibrationBridgeError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _json_file_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_file_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairV5Core4CalibrationBridgeError(f"{label} must be lowercase SHA-256")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PairV5Core4CalibrationBridgeError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise PairV5Core4CalibrationBridgeError(
            f"{label} must be an absolute plain directory"
        )
    return path.resolve(strict=True)


def _fresh_output_directory(value: str | Path) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or path == Path("/")
        or path.exists()
        or path.is_symlink()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
    ):
        raise PairV5Core4CalibrationBridgeError(
            "output must be a fresh absolute directory under an existing plain directory"
        )
    return path


def _reject_constant(token: str) -> None:
    raise PairV5Core4CalibrationBridgeError(f"non-finite JSON is forbidden: {token}")


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PairV5Core4CalibrationBridgeError(f"duplicate JSON key: {key!r}")
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
        raise PairV5Core4CalibrationBridgeError(f"{label} SHA-256 differs")
    try:
        decoded = json.loads(
            raw.decode("ascii"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PairV5Core4CalibrationBridgeError(f"{label} is invalid ASCII JSON") from error
    if not isinstance(decoded, dict):
        raise PairV5Core4CalibrationBridgeError(f"{label} root must be an object")
    return decoded, path, observed


def validate_score_group_receipt(
    value: Any,
    *,
    group_id: str,
    root_spec_raw_sha256: str,
    bank_receipt_digest: str,
) -> dict[str, Any]:
    scorer = _scorer_runtime()
    if not isinstance(value, Mapping) or set(value) != set(_GROUP_RECEIPT_FIELDS):
        raise PairV5Core4CalibrationBridgeError(
            f"{group_id} score-group receipt fields differ"
        )
    row = dict(value)
    unsigned = dict(row)
    declared = _sha256(
        unsigned.pop("receipt_digest"), label=f"{group_id} group receipt digest"
    )
    digests = row["candidate_receipt_digests"]
    if (
        object_sha256(unsigned) != declared
        or row["schema_version"] != REQUIRED_GROUP_RECEIPT_SCHEMA
        or row["group_id"] != group_id
        or row["root_spec_raw_sha256"] != root_spec_raw_sha256
        or row["bank_receipt_digest"] != bank_receipt_digest
        or row["candidate_count"] != GROUP_SIZE
        or not isinstance(digests, list)
        or len(digests) != GROUP_SIZE
        or len(set(digests)) != GROUP_SIZE
        or row["primary_score_field"] != "raw_global_action_energy_score"
        or row["phase_conjunctive_role"]
        != "diagnostic_only_never_calibration_gate"
        or row["input_closure"] != scorer.SCORE_INPUT_CLOSURE
        or row["training_performed"] is not False
        or row["optimizer_authorized"] is not False
        or row["scientific_action_editing_claim"] is not False
        or row["schedule_coordinate"] != scorer.schedule_coordinate_receipt()
        or row["method_source_revision"] != REQUIRED_SCORER_SOURCE_REVISION
    ):
        raise PairV5Core4CalibrationBridgeError(
            f"{group_id} score-group receipt semantics differ"
        )
    for ordinal, digest in enumerate(digests):
        _sha256(digest, label=f"{group_id} candidate digest {ordinal}")
    for name in (
        "frozen_checkpoint_receipt_digest",
        "method_source_archive_sha256",
    ):
        _sha256(row[name], label=f"{group_id} {name}")
    for name in ("method_source_revision", "bernini_revision", "veomni_revision"):
        if not isinstance(row[name], str) or _SHA1_RE.fullmatch(row[name]) is None:
            raise PairV5Core4CalibrationBridgeError(f"{group_id} {name} differs")
    if not isinstance(row["checkpoint_content_binding"], Mapping):
        raise PairV5Core4CalibrationBridgeError(
            f"{group_id} checkpoint binding differs"
        )
    return row


def _score_path(score_root: Path, group_id: str, candidate_id: str) -> Path:
    return score_root / group_id / candidate_id / SCORE_FILENAME


def _validate_group_filesystem_closure(
    group_root: Path,
    *,
    group_id: str,
    candidate_ids: Sequence[str],
) -> None:
    expected = set(candidate_ids) | {GROUP_FILENAME.format(group_id=group_id)}
    observed = {path.name for path in group_root.iterdir()}
    if observed != expected:
        raise PairV5Core4CalibrationBridgeError(
            f"{group_id} score filesystem closure differs: "
            f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )
    for candidate_id in candidate_ids:
        candidate_dir = group_root / candidate_id
        if (
            not candidate_dir.is_dir()
            or candidate_dir.is_symlink()
            or {path.name for path in candidate_dir.iterdir()} != {SCORE_FILENAME}
        ):
            raise PairV5Core4CalibrationBridgeError(
                f"{candidate_id} score directory closure differs"
            )


def _cell_registries(
    bound_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    runtime_by_cell: dict[str, dict[str, Any]] = {}
    captions_by_cell: dict[str, dict[str, str]] = {}
    for bound in bound_rows:
        candidate = bound["candidate"]
        cell_id = candidate["calibration_group_id"]
        runtime_by_cell.setdefault(cell_id, {})[candidate["semantic_branch"]] = bound[
            "generation_runtime_binding"
        ]
        captions_by_cell.setdefault(cell_id, {})[candidate["semantic_branch"]] = candidate[
            "full_t2v_caption"
        ]
    expected_branches = list(calibration.BRANCH_ORDER)
    for cell_id in runtime_by_cell:
        if (
            list(runtime_by_cell[cell_id]) != expected_branches
            or list(captions_by_cell[cell_id]) != expected_branches
        ):
            raise PairV5Core4CalibrationBridgeError(
                f"{cell_id} generation registry branch order differs"
            )
    return runtime_by_cell, captions_by_cell


def validate_score_generation_join(
    score: Mapping[str, Any],
    bound: Mapping[str, Any],
    *,
    group_receipt: Mapping[str, Any],
    root_spec_raw_sha256: str,
    bank_receipt_digest: str,
    expected_generation_registry: Mapping[str, Any],
    expected_caption_registry: Mapping[str, str],
) -> None:
    candidate = bound["candidate"]
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
    artifacts = bound["artifacts"]
    gaussian = artifacts["official_initial_gaussian"]
    expected = {
        "candidate_envelope_sha256": bound["candidate_envelope_sha256"],
        "root_spec_raw_sha256": root_spec_raw_sha256,
        "bank_receipt_digest": bank_receipt_digest,
        "generation_receipt_digest": bound["generation_receipt_digest"],
        "generation_receipt_file_sha256": bound["generation_receipt_file_sha256"],
        "native_rollout_receipt_digest": bound["native_rollout_receipt_digest"],
        "native_rollout_receipt_file_sha256": bound[
            "native_rollout_receipt_file_sha256"
        ],
        "generated_mp4_sha256": artifacts["mp4"]["sha256"],
        "clean_latent_artifact_sha256": artifacts["predecode_clean_latent"]["sha256"],
        "geometry_source_video_sha256": candidate["geometry_source_video_sha256"],
        "full_t2v_caption_utf8_sha256": candidate[
            "full_t2v_caption_utf8_sha256"
        ],
        "official_gaussian_artifact_sha256": gaussian["sha256"],
        "official_gaussian_raw_value_sha256": gaussian["raw_value_sha256"],
        "official_gaussian_content_sha256": gaussian["content_sha256"],
        "frozen_checkpoint_receipt_digest": group_receipt[
            "frozen_checkpoint_receipt_digest"
        ],
        "checkpoint_content_binding": group_receipt["checkpoint_content_binding"],
        "schedule_coordinate": group_receipt["schedule_coordinate"],
        "generation_runtime_binding_by_branch": dict(expected_generation_registry),
        "full_t2v_caption_by_branch": dict(expected_caption_registry),
    }
    if score.get("schema_version") != REQUIRED_SCORE_RECEIPT_SCHEMA:
        raise PairV5Core4CalibrationBridgeError(
            f"{candidate['candidate_id']} is not a d541801 v3 score receipt"
        )
    if any(score.get(name) != candidate[name] for name in identity_fields):
        raise PairV5Core4CalibrationBridgeError(
            f"{candidate['candidate_id']} score/candidate identity differs"
        )
    mismatches = [name for name, value in expected.items() if score.get(name) != value]
    if mismatches:
        raise PairV5Core4CalibrationBridgeError(
            f"{candidate['candidate_id']} score/generation join differs: {mismatches}"
        )


def load_ordered_score_inputs(
    *,
    score_root: str | Path,
    expected_group_file_sha256_by_id: Mapping[str, str],
    root_spec_raw_sha256: str,
    bank_receipt_digest: str,
    bound_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scorer = _scorer_runtime()
    root = _plain_directory(score_root, label="score root")
    if set(expected_group_file_sha256_by_id) != set(GROUP_IDS):
        raise PairV5Core4CalibrationBridgeError("score-group SHA registry differs")
    runtime_by_cell, captions_by_cell = _cell_registries(bound_rows)
    by_group = {
        group_id: [row for row in bound_rows if row["group_id"] == group_id]
        for group_id in GROUP_IDS
    }
    group_bindings: list[dict[str, Any]] = []
    joined: list[dict[str, Any]] = []
    common_authority: dict[str, Any] | None = None
    for group_id in GROUP_IDS:
        group_root = _plain_directory(root / group_id, label=f"{group_id} score root")
        group_rows = by_group[group_id]
        candidate_ids = [row["candidate"]["candidate_id"] for row in group_rows]
        _validate_group_filesystem_closure(
            group_root, group_id=group_id, candidate_ids=candidate_ids
        )
        group_path = group_root / GROUP_FILENAME.format(group_id=group_id)
        raw_group, resolved_group_path, group_file_sha = _read_strict_json(
            group_path,
            label=f"{group_id} score-group receipt",
            expected_sha256=expected_group_file_sha256_by_id[group_id],
        )
        group_receipt = validate_score_group_receipt(
            raw_group,
            group_id=group_id,
            root_spec_raw_sha256=root_spec_raw_sha256,
            bank_receipt_digest=bank_receipt_digest,
        )
        authority = {
            name: group_receipt[name]
            for name in (
                "frozen_checkpoint_receipt_digest",
                "checkpoint_content_binding",
                "schedule_coordinate",
                "method_source_revision",
                "method_source_archive_sha256",
                "bernini_revision",
                "veomni_revision",
            )
        }
        if common_authority is not None and authority != common_authority:
            raise PairV5Core4CalibrationBridgeError(
                "two score groups do not share one frozen scorer authority"
            )
        common_authority = authority
        observed_digests: list[str] = []
        score_file_bindings: list[dict[str, Any]] = []
        for group_ordinal, bound in enumerate(group_rows):
            candidate = bound["candidate"]
            candidate_id = candidate["candidate_id"]
            raw_score, score_path, score_file_sha = _read_strict_json(
                _score_path(root, group_id, candidate_id),
                label=f"{candidate_id} frozen score receipt",
            )
            try:
                score = scorer.validate_score_receipt(raw_score)
            except scorer.PairV5T2VEnergyScoringError as error:
                raise PairV5Core4CalibrationBridgeError(str(error)) from error
            validate_score_generation_join(
                score,
                bound,
                group_receipt=group_receipt,
                root_spec_raw_sha256=root_spec_raw_sha256,
                bank_receipt_digest=bank_receipt_digest,
                expected_generation_registry=runtime_by_cell[
                    candidate["calibration_group_id"]
                ],
                expected_caption_registry=captions_by_cell[
                    candidate["calibration_group_id"]
                ],
            )
            observed_digests.append(score["receipt_digest"])
            binding = {
                "group_ordinal": group_ordinal,
                "candidate_id": candidate_id,
                "path": str(score_path),
                "file_sha256": score_file_sha,
                "receipt_digest": score["receipt_digest"],
            }
            score_file_bindings.append(binding)
            joined.append(
                {
                    **dict(bound),
                    "score": score,
                    "score_path": str(score_path),
                    "score_file_sha256": score_file_sha,
                }
            )
        if observed_digests != group_receipt["candidate_receipt_digests"]:
            raise PairV5Core4CalibrationBridgeError(
                f"{group_id} score candidate digest/order differs"
            )
        group_bindings.append(
            {
                "group_id": group_id,
                "path": str(resolved_group_path),
                "file_sha256": group_file_sha,
                "receipt_digest": group_receipt["receipt_digest"],
                "candidate_count": GROUP_SIZE,
                "candidate_order": candidate_ids,
                "candidate_receipt_digests": observed_digests,
                "score_receipt_files": score_file_bindings,
                "frozen_scorer_authority": authority,
            }
        )
    expected_order = [row["candidate"]["candidate_id"] for row in bound_rows]
    if [row["candidate"]["candidate_id"] for row in joined] != expected_order:
        raise PairV5Core4CalibrationBridgeError("global score candidate order differs")
    return group_bindings, joined


def build_calibration_payloads(
    *,
    joined_rows: Sequence[Mapping[str, Any]],
    label_manifest: Mapping[str, Any],
    root_spec_raw_sha256: str,
    bank_receipt_digest: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Build only detached booleans/scalars and invoke the existing calibrator."""

    if len(joined_rows) != CANDIDATE_COUNT:
        raise PairV5Core4CalibrationBridgeError("calibration bridge requires 40 scores")
    expected_order = [row["candidate"]["candidate_id"] for row in joined_rows]
    if (
        label_manifest.get("candidate_order") != expected_order
        or not isinstance(label_manifest.get("rows"), list)
        or [row.get("candidate_id") for row in label_manifest["rows"]]
        != expected_order
    ):
        raise PairV5Core4CalibrationBridgeError("label/score order differs")

    audits: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for ordinal, (joined, label) in enumerate(zip(joined_rows, label_manifest["rows"])):
        candidate = joined["candidate"]
        score = joined["score"]
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
        if (
            label.get("ordinal") != ordinal
            or label.get("group_id") != joined["group_id"]
            or any(label.get(name) != candidate[name] for name in identity_fields)
            or label.get("generation_receipt_digest")
            != joined["generation_receipt_digest"]
        ):
            raise PairV5Core4CalibrationBridgeError(
                f"label identity differs for {candidate['candidate_id']}"
            )
        try:
            audit = calibration.seal_event_audit_receipt(
                **{name: candidate[name] for name in identity_fields},
                generation_receipt_digest=joined["generation_receipt_digest"],
                audit_source_kind=label["audit_source_kind"],
                external_audit_artifact_sha256=label[
                    "external_audit_artifact_sha256"
                ],
                **{name: label[name] for name in label_author.LABEL_BOOLEAN_FIELDS},
            )
            score_row = calibration.make_score_row(
                row_id=f"core4-v2-{ordinal:02d}",
                **{name: candidate[name] for name in identity_fields},
                raw_global_action_energy_score=score[
                    "raw_global_action_energy_score"
                ],
                generation_receipt_digest=joined["generation_receipt_digest"],
                frozen_scorer_receipt_digest=score[
                    "frozen_scorer_receipt_digest"
                ],
                event_audit_receipt_digest=audit["receipt_digest"],
            )
        except calibration.PairV5EnergyCalibrationV3Error as error:
            raise PairV5Core4CalibrationBridgeError(str(error)) from error
        audits.append(calibration.validate_event_audit_receipt(audit))
        score_rows.append(calibration.validate_score_row(score_row))

    families = list(
        dict.fromkeys(row["candidate"]["action_family_id"] for row in joined_rows)
    )
    try:
        preregistration = calibration.make_preregistration(CALIBRATOR_ID, families)
        calibration_receipt = calibration.calibrate_global_action_energy(
            score_rows,
            audits,
            preregistration,
            source_bank_spec_sha256=root_spec_raw_sha256,
            source_bank_receipt_digest=bank_receipt_digest,
        )
    except calibration.PairV5EnergyCalibrationV3Error as error:
        raise PairV5Core4CalibrationBridgeError(str(error)) from error
    return audits, score_rows, preregistration, calibration_receipt


def _output_binding(path: Path, value: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "file_sha256": _json_file_sha256(value),
        digest_field: value[digest_field],
    }


def make_bridge_receipt(
    *,
    output_dir: Path,
    root_spec_path: Path,
    root_spec_raw_sha256: str,
    bank_receipt_path: Path,
    bank_receipt_file_sha256: str,
    bank_receipt_digest: str,
    label_manifest_path: Path,
    label_manifest_file_sha256: str,
    label_manifest: Mapping[str, Any],
    score_group_bindings: Sequence[Mapping[str, Any]],
    joined_rows: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
    calibration_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_order = [row["candidate"]["candidate_id"] for row in joined_rows]
    audit_bindings: list[dict[str, Any]] = []
    row_bindings: list[dict[str, Any]] = []
    generation_bindings: list[dict[str, Any]] = []
    score_bindings: list[dict[str, Any]] = []
    for ordinal, (joined, audit, score_row) in enumerate(
        zip(joined_rows, audits, score_rows)
    ):
        candidate_id = candidate_order[ordinal]
        audit_path = output_dir / "event_audits" / f"{ordinal:02d}-{candidate_id}.json"
        row_path = output_dir / "score_rows" / f"{ordinal:02d}-{candidate_id}.json"
        audit_bindings.append(
            {
                "ordinal": ordinal,
                "candidate_id": candidate_id,
                **_output_binding(audit_path, audit, "receipt_digest"),
            }
        )
        row_bindings.append(
            {
                "ordinal": ordinal,
                "candidate_id": candidate_id,
                **_output_binding(row_path, score_row, "row_digest"),
            }
        )
        score = joined["score"]
        generation_bindings.append(
            {
                "ordinal": ordinal,
                "group_id": joined["group_id"],
                "candidate_id": candidate_id,
                "candidate_envelope_sha256": joined[
                    "candidate_envelope_sha256"
                ],
                "generation_receipt_digest": joined["generation_receipt_digest"],
                "generation_receipt_file_sha256": joined[
                    "generation_receipt_file_sha256"
                ],
                "native_rollout_receipt_digest": joined[
                    "native_rollout_receipt_digest"
                ],
                "native_rollout_receipt_file_sha256": joined[
                    "native_rollout_receipt_file_sha256"
                ],
            }
        )
        score_bindings.append(
            {
                "ordinal": ordinal,
                "group_id": joined["group_id"],
                "candidate_id": candidate_id,
                "path": joined["score_path"],
                "file_sha256": joined["score_file_sha256"],
                "receipt_digest": score["receipt_digest"],
                "frozen_scorer_receipt_digest": score[
                    "frozen_scorer_receipt_digest"
                ],
                "raw_global_action_energy_score": score[
                    "raw_global_action_energy_score"
                ],
                "score_row_digest": score_row["row_digest"],
            }
        )
    prereg_path = output_dir / "pair-v5-t2v-energy-preregistration-v3.json"
    calibration_path = output_dir / "pair-v5-t2v-energy-calibration-receipt-v3.json"
    ambiguous = [
        row["candidate_id"]
        for row in label_manifest["rows"]
        if row["full_target_action_observed"] is False
        and row["full_target_action_false_confirmed"] is False
    ]
    unsigned = {
        "schema_version": BRIDGE_SCHEMA,
        "required_frozen_scorer_source_revision": REQUIRED_SCORER_SOURCE_REVISION,
        "source_root_spec": {
            "path": str(root_spec_path),
            "file_sha256": root_spec_raw_sha256,
            "schema_version": "pair-v5-frozen-bernini-t2v-calibration-bank-spec-v2",
        },
        "source_bank_receipt": {
            "path": str(bank_receipt_path),
            "file_sha256": bank_receipt_file_sha256,
            "receipt_digest": bank_receipt_digest,
        },
        "detached_event_label_manifest": {
            "path": str(label_manifest_path),
            "file_sha256": label_manifest_file_sha256,
            "manifest_digest": label_manifest["manifest_digest"],
            "candidate_count": CANDIDATE_COUNT,
            "author_acknowledgements": label_manifest[
                "author_acknowledgements"
            ],
        },
        "score_group_receipts": [dict(row) for row in score_group_bindings],
        "candidate_count": CANDIDATE_COUNT,
        "candidate_order": candidate_order,
        "generation_bindings": generation_bindings,
        "score_receipt_bindings": score_bindings,
        "event_audit_receipt_bindings": audit_bindings,
        "score_row_bindings": row_bindings,
        "preregistration_binding": _output_binding(
            prereg_path, preregistration, "preregistration_digest"
        ),
        "calibration_binding": {
            **_output_binding(
                calibration_path, calibration_receipt, "receipt_digest"
            ),
            "optimizer_authorized": calibration_receipt["optimizer_authorized"],
            "failure_reasons": calibration_receipt["failure_reasons"],
            "gates": calibration_receipt["gates"],
        },
        "calibrator_input_contract": {
            "input_object_kinds": [
                "scalar_score_rows",
                "detached_boolean_event_audit_receipts",
                "preregistration",
            ],
            "score_scalar_field": "raw_global_action_energy_score",
            "media_or_path_fields_present": False,
            "tensor_or_latent_fields_present": False,
            "model_or_prompt_fields_present": False,
            "external_audit_artifact_contents_enter_calibrator": False,
            "generation_artifact_contents_enter_calibrator": False,
            "provenance_files_verified_before_calibrator": True,
        },
        "ambiguous_candidate_ids": ambiguous,
        "optimizer_authorized": calibration_receipt["optimizer_authorized"],
        "optimizer_authorization_source": (
            "pair_v5_t2v_energy_calibration_v3_existing_fit_and_confirmation_gates_only"
        ),
        "confirmation_samples_consumed_by_optimizer": False,
        "training_performed": False,
        "scientific_action_editing_claim": False,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def validate_bridge_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_BRIDGE_FIELDS):
        raise PairV5Core4CalibrationBridgeError("bridge receipt fields differ")
    root = dict(value)
    unsigned = dict(root)
    declared = _sha256(unsigned.pop("receipt_digest"), label="bridge receipt digest")
    order = root["candidate_order"]
    list_fields = (
        "generation_bindings",
        "score_receipt_bindings",
        "event_audit_receipt_bindings",
        "score_row_bindings",
    )
    score_groups = root["score_group_receipts"]
    if (
        object_sha256(unsigned) != declared
        or root["schema_version"] != BRIDGE_SCHEMA
        or root["required_frozen_scorer_source_revision"]
        != REQUIRED_SCORER_SOURCE_REVISION
        or root["candidate_count"] != CANDIDATE_COUNT
        or not isinstance(order, list)
        or len(order) != CANDIDATE_COUNT
        or len(set(order)) != CANDIDATE_COUNT
        or any(not item.startswith(label_author.CORE4_PREFIX) for item in order)
        or root["calibrator_input_contract"]
        != {
            "input_object_kinds": [
                "scalar_score_rows",
                "detached_boolean_event_audit_receipts",
                "preregistration",
            ],
            "score_scalar_field": "raw_global_action_energy_score",
            "media_or_path_fields_present": False,
            "tensor_or_latent_fields_present": False,
            "model_or_prompt_fields_present": False,
            "external_audit_artifact_contents_enter_calibrator": False,
            "generation_artifact_contents_enter_calibrator": False,
            "provenance_files_verified_before_calibrator": True,
        }
        or root["optimizer_authorization_source"]
        != "pair_v5_t2v_energy_calibration_v3_existing_fit_and_confirmation_gates_only"
        or root["confirmation_samples_consumed_by_optimizer"] is not False
        or root["training_performed"] is not False
        or root["scientific_action_editing_claim"] is not False
        or root["optimizer_authorized"]
        is not root["calibration_binding"].get("optimizer_authorized")
        or not isinstance(score_groups, list)
        or len(score_groups) != len(GROUP_IDS)
        or [group.get("group_id") for group in score_groups] != list(GROUP_IDS)
        or any(group.get("candidate_count") != GROUP_SIZE for group in score_groups)
        or any(
            not isinstance(group.get("frozen_scorer_authority"), Mapping)
            or group["frozen_scorer_authority"].get("method_source_revision")
            != REQUIRED_SCORER_SOURCE_REVISION
            for group in score_groups
        )
    ):
        raise PairV5Core4CalibrationBridgeError("bridge receipt semantics differ")
    for field in list_fields:
        rows = root[field]
        if (
            not isinstance(rows, list)
            or len(rows) != CANDIDATE_COUNT
            or [row.get("ordinal") for row in rows] != list(range(CANDIDATE_COUNT))
            or [row.get("candidate_id") for row in rows] != order
        ):
            raise PairV5Core4CalibrationBridgeError(
                f"bridge {field} identity/order differs"
            )
    if (
        [candidate_id for group in score_groups for candidate_id in group["candidate_order"]]
        != order
        or
        [row["receipt_digest"] for row in root["score_receipt_bindings"]]
        != [
            digest
            for group in score_groups
            for digest in group["candidate_receipt_digests"]
        ]
        or [row["score_row_digest"] for row in root["score_receipt_bindings"]]
        != [row["row_digest"] for row in root["score_row_bindings"]]
        or set(root["ambiguous_candidate_ids"]) - set(order)
    ):
        raise PairV5Core4CalibrationBridgeError("bridge digest/order closure differs")
    return root


def _write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise PairV5Core4CalibrationBridgeError(f"refusing to overwrite {path}")
    raw = _json_file_bytes(value)
    path.write_bytes(raw)
    os.chmod(path, 0o400)
    return hashlib.sha256(raw).hexdigest()


def materialize(
    *,
    root_spec: str | Path,
    expected_root_spec_sha256: str,
    bank_output_dir: str | Path,
    bank_receipt: str | Path,
    expected_bank_receipt_sha256: str,
    score_root: str | Path,
    expected_sp4_a_score_group_sha256: str,
    expected_sp4_b_score_group_sha256: str,
    detached_label_manifest: str | Path,
    expected_detached_label_manifest_sha256: str,
    output_dir: str | Path,
    acknowledge_reviewed_label_manifest: bool,
) -> dict[str, Any]:
    if acknowledge_reviewed_label_manifest is not True:
        raise PairV5Core4CalibrationBridgeError(
            "--acknowledge-reviewed-label-manifest is required"
        )
    output = _fresh_output_directory(output_dir)
    root_spec_path = _plain_file(root_spec, label="root spec")
    bank_receipt_path = _plain_file(bank_receipt, label="bank receipt")
    _spec, bank, bound_rows = label_author.load_core4_bound_bank(
        root_spec=root_spec_path,
        root_spec_sha256=expected_root_spec_sha256,
        bank_output_dir=bank_output_dir,
        bank_receipt=bank_receipt_path,
        bank_receipt_sha256=expected_bank_receipt_sha256,
    )
    if bank["file_sha256"] != expected_bank_receipt_sha256:
        raise PairV5Core4CalibrationBridgeError("bank file SHA-256 differs")
    score_groups, joined_rows = load_ordered_score_inputs(
        score_root=score_root,
        expected_group_file_sha256_by_id={
            "sp4-a": expected_sp4_a_score_group_sha256,
            "sp4-b": expected_sp4_b_score_group_sha256,
        },
        root_spec_raw_sha256=expected_root_spec_sha256,
        bank_receipt_digest=bank["receipt_digest"],
        bound_rows=bound_rows,
    )
    try:
        labels, label_path, label_file_sha = label_author.load_label_manifest(
            detached_label_manifest,
            expected_sha256=expected_detached_label_manifest_sha256,
            root_spec_raw_sha256=expected_root_spec_sha256,
            bank_receipt_digest=bank["receipt_digest"],
            bound_rows=bound_rows,
        )
    except label_author.PairV5Core4LabelAuthoringError as error:
        raise PairV5Core4CalibrationBridgeError(str(error)) from error
    audits, score_rows, preregistration, calibration_receipt = (
        build_calibration_payloads(
            joined_rows=joined_rows,
            label_manifest=labels,
            root_spec_raw_sha256=expected_root_spec_sha256,
            bank_receipt_digest=bank["receipt_digest"],
        )
    )
    bridge_receipt = make_bridge_receipt(
        output_dir=output,
        root_spec_path=root_spec_path,
        root_spec_raw_sha256=expected_root_spec_sha256,
        bank_receipt_path=bank_receipt_path,
        bank_receipt_file_sha256=expected_bank_receipt_sha256,
        bank_receipt_digest=bank["receipt_digest"],
        label_manifest_path=label_path,
        label_manifest_file_sha256=label_file_sha,
        label_manifest=labels,
        score_group_bindings=score_groups,
        joined_rows=joined_rows,
        audits=audits,
        score_rows=score_rows,
        preregistration=preregistration,
        calibration_receipt=calibration_receipt,
    )
    validate_bridge_receipt(bridge_receipt)

    # All input validation and calibration complete before the fresh output is
    # created.  The following writes only canonical JSON receipts.
    output.mkdir()
    audit_root = output / "event_audits"
    row_root = output / "score_rows"
    audit_root.mkdir()
    row_root.mkdir()
    for binding, audit in zip(
        bridge_receipt["event_audit_receipt_bindings"], audits
    ):
        observed = _write_create_only(Path(binding["path"]), audit)
        if observed != binding["file_sha256"]:
            raise PairV5Core4CalibrationBridgeError("written event-audit hash differs")
    for binding, row in zip(bridge_receipt["score_row_bindings"], score_rows):
        observed = _write_create_only(Path(binding["path"]), row)
        if observed != binding["file_sha256"]:
            raise PairV5Core4CalibrationBridgeError("written score-row hash differs")
    for binding_name, artifact in (
        ("preregistration_binding", preregistration),
        ("calibration_binding", calibration_receipt),
    ):
        binding = bridge_receipt[binding_name]
        observed = _write_create_only(Path(binding["path"]), artifact)
        if observed != binding["file_sha256"]:
            raise PairV5Core4CalibrationBridgeError(f"written {binding_name} hash differs")
    _write_create_only(
        output / BRIDGE_RECEIPT_FILENAME,
        bridge_receipt,
    )
    os.chmod(audit_root, 0o500)
    os.chmod(row_root, 0o500)
    os.chmod(output, 0o500)
    return bridge_receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-spec", required=True)
    parser.add_argument("--expected-root-spec-sha256", required=True)
    parser.add_argument("--bank-output-dir", required=True)
    parser.add_argument("--bank-receipt", required=True)
    parser.add_argument("--expected-bank-receipt-sha256", required=True)
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--expected-sp4-a-score-group-sha256", required=True)
    parser.add_argument("--expected-sp4-b-score-group-sha256", required=True)
    parser.add_argument("--detached-label-manifest", required=True)
    parser.add_argument("--expected-detached-label-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--acknowledge-reviewed-label-manifest", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = materialize(
        root_spec=args.root_spec,
        expected_root_spec_sha256=args.expected_root_spec_sha256,
        bank_output_dir=args.bank_output_dir,
        bank_receipt=args.bank_receipt,
        expected_bank_receipt_sha256=args.expected_bank_receipt_sha256,
        score_root=args.score_root,
        expected_sp4_a_score_group_sha256=args.expected_sp4_a_score_group_sha256,
        expected_sp4_b_score_group_sha256=args.expected_sp4_b_score_group_sha256,
        detached_label_manifest=args.detached_label_manifest,
        expected_detached_label_manifest_sha256=(
            args.expected_detached_label_manifest_sha256
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
                "ambiguous_candidate_count": len(receipt["ambiguous_candidate_ids"]),
                "bridge_receipt_digest": receipt["receipt_digest"],
                "output_dir": str(Path(args.output_dir)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BRIDGE_SCHEMA",
    "BRIDGE_RECEIPT_FILENAME",
    "CALIBRATOR_ID",
    "PairV5Core4CalibrationBridgeError",
    "REQUIRED_GROUP_RECEIPT_SCHEMA",
    "REQUIRED_SCORER_SOURCE_REVISION",
    "REQUIRED_SCORE_RECEIPT_SCHEMA",
    "build_calibration_payloads",
    "load_ordered_score_inputs",
    "make_bridge_receipt",
    "materialize",
    "validate_bridge_receipt",
    "validate_score_generation_join",
    "validate_score_group_receipt",
]
