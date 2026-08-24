#!/usr/bin/env python3
"""Independent mainline validator for PAIR-v5 v4-score calibration.

This validator deliberately has no initialization-ablation, distilled-teacher,
adapter, or optimizer-manifest dependency.  It reconstructs the formal scalar
calibration directly from:

* the sealed frozen-T2V bank spec and scalar rendered-bank manifest;
* all forty frozen global-MACE score receipts;
* all detached event-audit receipts and scalar score rows;
* the preregistration; and
* the stored v3 calibration receipt.

It exports only the authenticated per-family clipped-affine maps, threshold,
prompt-cell text/spec metadata, and scalar provenance digests.  T2V videos,
latents, Gaussians, or proposals are never exported as a native-RV2V model
input, target, teacher, donor, adapter input, or noise.  This validator never
opens a generated T2V MP4, latent, or Gaussian artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import pair_v5_t2v_energy_calibration_v3 as calibration  # noqa: E402
import pair_v5_t2v_calibration_bank_spec as bank_contract  # noqa: E402
import score_pair_v5_t2v_energy_bank_v3 as scorer  # noqa: E402


AUTHORIZATION_SCHEMA = "bernini-pair-v5-t2v-calibration-mainline-authorization-v4"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_BANK_FIELDS = frozenset(
    {
        "schema_version",
        "root_spec_raw_sha256",
        "candidate_count",
        "cell_count",
        "mace_branch_order",
        "sampling_contract",
        "semantic_input_closure",
        "artifact_use_contract",
        "split_contract",
        "split_group_membership",
        "fit_confirmation_all_registered_axes_disjoint",
        "same_cell_gaussian_proofs",
        "candidate_receipts",
        "interpretation",
        "receipt_digest",
    }
)
_BANK_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "analysis_split",
        "action_family_id",
        "calibration_group_id",
        "semantic_branch",
        "receipt_path",
        "receipt_sha256",
        "receipt_digest",
        "mp4_sha256",
        "predecode_clean_latent_sha256",
        "official_initial_gaussian_sha256",
    }
)
_CELL_PROOF_FIELDS = frozenset(
    {
        "analysis_split",
        "action_family_id",
        "calibration_group_id",
        "semantic_branch_count",
        "semantic_branch_order",
        "all_ten_official_gaussian_tensor_values_byte_equal",
        "all_container_files_individually_sha256_verified",
        "official_gaussian_file_sha256_by_branch",
        "official_gaussian_raw_value_sha256",
        "official_gaussian_content_sha256",
        "seed",
    }
)


class PairV5MainlineCalibrationError(RuntimeError):
    """The formal scalar calibration graph failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return calibration.canonical_json_bytes(value)


def object_sha256(value: Any) -> str:
    return calibration.object_sha256(value)


def file_sha256(path: Path) -> str:
    try:
        return scorer.file_sha256(path)
    except scorer.PairV5T2VEnergyScoringError as error:
        raise PairV5MainlineCalibrationError(str(error)) from error


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairV5MainlineCalibrationError(f"{label} must be lowercase SHA-256")
    return value


def _plain_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise PairV5MainlineCalibrationError(f"{label} path differs")
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PairV5MainlineCalibrationError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _plain_directory(value: Any, *, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise PairV5MainlineCalibrationError(f"{label} path differs")
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise PairV5MainlineCalibrationError(
            f"{label} must be an absolute plain directory"
        )
    return path.resolve(strict=True)


def _reject_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PairV5MainlineCalibrationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise PairV5MainlineCalibrationError(f"{label} contains {token}")

    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=_reject_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PairV5MainlineCalibrationError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise PairV5MainlineCalibrationError(f"{label} root must be an object")
    return value


def _read_bound_json(
    value: Any, expected_sha256: Any, *, label: str
) -> tuple[dict[str, Any], Path]:
    path = _plain_file(value, label=label)
    if file_sha256(path) != _sha256(expected_sha256, label=f"{label} SHA-256"):
        raise PairV5MainlineCalibrationError(f"{label} SHA-256 differs")
    return _read_json(path, label=label), path


def _score_path(score_root: Path, group_id: str, candidate_id: str) -> Path:
    return (
        score_root
        / group_id
        / candidate_id
        / scorer.SCORE_RECEIPT_FILENAME
    )


def _generation_checkpoint_trees(score: Mapping[str, Any]) -> set[str]:
    registry = score.get("generation_runtime_binding_by_branch")
    if not isinstance(registry, Mapping) or set(registry) != set(
        calibration.BRANCH_ORDER
    ):
        raise PairV5MainlineCalibrationError(
            "formal score generation-runtime registry differs"
        )
    values = {
        registry[branch].get("checkpoint_tree_sha256")
        for branch in calibration.BRANCH_ORDER
    }
    if any(not isinstance(value, str) for value in values):
        raise PairV5MainlineCalibrationError(
            "formal score generation checkpoint binding differs"
        )
    return values


def _load_formal_bank_manifest(
    *,
    root_spec: str | Path,
    root_spec_sha256: str,
    bank_receipt: str | Path,
    bank_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], list[tuple[str, dict[str, Any], dict[str, Any]]]]:
    """Authenticate scalar bank provenance without opening generated media."""

    try:
        spec, observed_spec_sha = bank_contract.load_sealed_spec(
            root_spec, root_spec_sha256
        )
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise PairV5MainlineCalibrationError(str(error)) from error
    if observed_spec_sha != root_spec_sha256:
        raise PairV5MainlineCalibrationError("T2V bank spec digest differs")
    bank, bank_path = _read_bound_json(
        bank_receipt,
        bank_receipt_sha256,
        label="rendered T2V bank manifest",
    )
    if set(bank) != set(_BANK_FIELDS):
        raise PairV5MainlineCalibrationError(
            "rendered T2V bank manifest field closure differs"
        )
    unsigned = dict(bank)
    bank_digest = _sha256(
        unsigned.pop("receipt_digest", None), label="T2V bank manifest digest"
    )
    if (
        bank_contract.sha256_bytes(bank_contract.canonical_json_bytes(unsigned))
        != bank_digest
    ):
        raise PairV5MainlineCalibrationError(
            "rendered T2V bank manifest embedded digest differs"
        )

    flattened = [
        (group["group_id"], candidate)
        for group in spec["groups"]
        for candidate in group["candidates"]
    ]
    expected_membership = {
        split: {
            axis: sorted(
                {
                    candidate[axis]
                    for _, candidate in flattened
                    if candidate["analysis_split"] == split
                }
            )
            for axis in bank_contract.SPLIT_GROUP_AXES
        }
        for split in bank_contract.ANALYSIS_SPLITS
    }
    expected_interpretation = {
        "calibration_evidence_only": True,
        "event_qualification_performed": False,
        "action_success_not_implied": True,
        "training_performed": False,
        "parameter_update_performed": False,
        "optimizer_authorized": False,
        "t2v_negative_media_are_rv2v_policy_candidates": False,
        "t2v_media_as_condition_target_donor_or_noise_forbidden": True,
    }
    if (
        bank["schema_version"] != bank_contract.BANK_RECEIPT_SCHEMA_VERSION
        or bank["root_spec_raw_sha256"] != observed_spec_sha
        or bank["candidate_count"] != len(flattened)
        or bank["candidate_count"] != 40
        or bank["cell_count"] != 4
        or bank["mace_branch_order"] != list(calibration.BRANCH_ORDER)
        or bank["sampling_contract"] != bank_contract.SAMPLING_CONTRACT
        or bank["semantic_input_closure"] != bank_contract.SEMANTIC_INPUT_CLOSURE
        or bank["artifact_use_contract"] != bank_contract.ARTIFACT_USE_CONTRACT
        or bank["split_contract"] != bank_contract.SPLIT_CONTRACT
        or bank["split_group_membership"] != expected_membership
        or bank["fit_confirmation_all_registered_axes_disjoint"] is not True
        or bank["interpretation"] != expected_interpretation
    ):
        raise PairV5MainlineCalibrationError(
            "rendered T2V bank manifest semantic closure differs"
        )

    indexed_rows = bank["candidate_receipts"]
    if (
        not isinstance(indexed_rows, list)
        or len(indexed_rows) != len(flattened)
        or any(
            not isinstance(row, Mapping)
            or set(row) != set(_BANK_CANDIDATE_FIELDS)
            for row in indexed_rows
        )
    ):
        raise PairV5MainlineCalibrationError(
            "rendered T2V candidate index closure differs"
        )
    bound: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for (group_id, candidate), raw_row in zip(flattened, indexed_rows):
        row = dict(raw_row)
        receipt_path = row["receipt_path"]
        if (
            any(
                row[field] != candidate[field]
                for field in (
                    "candidate_id",
                    "analysis_split",
                    "action_family_id",
                    "calibration_group_id",
                    "semantic_branch",
                )
            )
            or not isinstance(receipt_path, str)
            or not receipt_path.startswith("/")
            or "\x00" in receipt_path
        ):
            raise PairV5MainlineCalibrationError(
                "rendered T2V candidate index identity differs"
            )
        for field in (
            "receipt_sha256",
            "receipt_digest",
            "mp4_sha256",
            "predecode_clean_latent_sha256",
            "official_initial_gaussian_sha256",
        ):
            _sha256(row[field], label=f"{candidate['candidate_id']} {field}")
        bound.append((group_id, candidate, row))

    proofs = bank["same_cell_gaussian_proofs"]
    expected_cells = {
        candidate["calibration_group_id"]
        for _, candidate in flattened
    }
    if (
        not isinstance(proofs, list)
        or len(proofs) != 4
        or any(
            not isinstance(proof, Mapping)
            or set(proof) != set(_CELL_PROOF_FIELDS)
            for proof in proofs
        )
        or {proof["calibration_group_id"] for proof in proofs} != expected_cells
    ):
        raise PairV5MainlineCalibrationError(
            "rendered T2V same-cell proof closure differs"
        )
    by_cell = {
        cell_id: [
            candidate
            for _, candidate in flattened
            if candidate["calibration_group_id"] == cell_id
        ]
        for cell_id in expected_cells
    }
    for proof in proofs:
        rows = by_cell[proof["calibration_group_id"]]
        gaussian_files = proof["official_gaussian_file_sha256_by_branch"]
        if (
            proof["analysis_split"] != rows[0]["analysis_split"]
            or proof["action_family_id"] != rows[0]["action_family_id"]
            or proof["semantic_branch_count"] != 10
            or proof["semantic_branch_order"] != list(calibration.BRANCH_ORDER)
            or proof["all_ten_official_gaussian_tensor_values_byte_equal"]
            is not True
            or proof["all_container_files_individually_sha256_verified"] is not True
            or not isinstance(gaussian_files, Mapping)
            or set(gaussian_files) != set(calibration.BRANCH_ORDER)
            or type(proof["seed"]) is not int
        ):
            raise PairV5MainlineCalibrationError(
                "rendered T2V same-cell proof semantics differ"
            )
        for value in (
            *gaussian_files.values(),
            proof["official_gaussian_raw_value_sha256"],
            proof["official_gaussian_content_sha256"],
        ):
            _sha256(value, label="same-cell Gaussian proof SHA-256")
    return (
        spec,
        {
            **bank,
            "receipt_digest": bank_digest,
            "file_sha256": bank_receipt_sha256,
            "path": str(bank_path),
        },
        bound,
    )


def load_mainline_calibration_bundle(
    *,
    root_spec: str | Path,
    root_spec_sha256: str,
    bank_receipt: str | Path,
    bank_receipt_sha256: str,
    score_root: str | Path,
    calibration_root: str | Path,
    calibration_receipt_sha256: str,
    preregistration_sha256: str,
    checkpoint_tree_sha256: str,
) -> dict[str, Any]:
    """Recompute the formal scalar calibration from its mainline artifacts."""

    spec_sha = _sha256(root_spec_sha256, label="T2V bank spec SHA-256")
    bank_sha = _sha256(bank_receipt_sha256, label="T2V bank receipt SHA-256")
    calibration_sha = _sha256(
        calibration_receipt_sha256, label="calibration receipt SHA-256"
    )
    prereg_sha = _sha256(
        preregistration_sha256, label="preregistration SHA-256"
    )
    checkpoint_tree = _sha256(
        checkpoint_tree_sha256, label="checkpoint tree SHA-256"
    )
    spec_path = _plain_file(root_spec, label="T2V bank spec")
    bank_path = _plain_file(bank_receipt, label="rendered T2V bank receipt")
    score_root_path = _plain_directory(score_root, label="formal T2V score root")
    scalar_root = _plain_directory(
        calibration_root, label="formal scalar calibration root"
    )

    t2v_spec, bank_value, grouped_bound_rows = _load_formal_bank_manifest(
        root_spec=spec_path,
        root_spec_sha256=spec_sha,
        bank_receipt=bank_path,
        bank_receipt_sha256=bank_sha,
    )

    prereg_raw, prereg_path = _read_bound_json(
        scalar_root / "preregistration-v3.json",
        prereg_sha,
        label="formal v3 preregistration",
    )
    try:
        prereg = calibration.validate_preregistration(prereg_raw)
    except calibration.PairV5EnergyCalibrationV3Error as error:
        raise PairV5MainlineCalibrationError(str(error)) from error
    stored_raw, calibration_path = _read_bound_json(
        scalar_root / "calibration-receipt-v3.json",
        calibration_sha,
        label="formal v3 calibration receipt",
    )

    if len(grouped_bound_rows) != 40 or len(
        {candidate["candidate_id"] for _, candidate, _ in grouped_bound_rows}
    ) != 40:
        raise PairV5MainlineCalibrationError(
            "formal T2V bank must contain exactly forty unique candidates"
        )
    score_paths = [
        _plain_file(
            _score_path(score_root_path, group_id, candidate["candidate_id"]),
            label=f"{candidate['candidate_id']} formal frozen-T2V v4 score",
        )
        for group_id, candidate, _ in grouped_bound_rows
    ]
    validated_score_rows: list[dict[str, Any]] = []
    for score_path in score_paths:
        try:
            validated_score_rows.append(
                scorer.validate_score_receipt(
                    _read_json(score_path, label=f"{score_path.parent.name} v4 score")
                )
            )
        except scorer.PairV5T2VEnergyScoringError as error:
            raise PairV5MainlineCalibrationError(str(error)) from error
    if [score["candidate_id"] for score in validated_score_rows] != [
        candidate["candidate_id"] for _, candidate, _ in grouped_bound_rows
    ]:
        raise PairV5MainlineCalibrationError(
            "formal v4 score order differs from the sealed bank"
        )

    audits: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    score_bindings: list[dict[str, Any]] = []
    validated_scores: dict[str, dict[str, Any]] = {}
    for (group_id, candidate, bank_row), score_path, score in zip(
        grouped_bound_rows, score_paths, validated_score_rows
    ):
        candidate_id = candidate["candidate_id"]
        audit_path = _plain_file(
            scalar_root / "event-audits" / f"{candidate_id}.json",
            label=f"{candidate_id} detached event audit",
        )
        row_path = _plain_file(
            scalar_root / "score-rows" / f"{candidate_id}.json",
            label=f"{candidate_id} scalar score row",
        )
        try:
            audit = calibration.validate_event_audit_receipt(
                _read_json(audit_path, label=f"{candidate_id} event audit")
            )
            scalar = calibration.validate_score_row(
                _read_json(row_path, label=f"{candidate_id} scalar row")
            )
        except calibration.PairV5EnergyCalibrationV3Error as error:
            raise PairV5MainlineCalibrationError(str(error)) from error
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
            any(score[field] != candidate[field] for field in identity_fields)
            or any(scalar[field] != candidate[field] for field in identity_fields)
            or any(audit[field] != candidate[field] for field in identity_fields)
            or score["root_spec_raw_sha256"] != spec_sha
            or score["bank_receipt_digest"] != bank_value["receipt_digest"]
            or score["generation_receipt_digest"]
            != bank_row["receipt_digest"]
            or score["generation_receipt_file_sha256"]
            != bank_row["receipt_sha256"]
            or score["generated_mp4_sha256"] != bank_row["mp4_sha256"]
            or score["clean_latent_artifact_sha256"]
            != bank_row["predecode_clean_latent_sha256"]
            or score["official_gaussian_artifact_sha256"]
            != bank_row["official_initial_gaussian_sha256"]
            or score["geometry_source_video_sha256"]
            != candidate["geometry_source_video_sha256"]
            or score["full_t2v_caption_utf8_sha256"]
            != hashlib.sha256(candidate["full_t2v_caption"].encode("utf-8")).hexdigest()
            or audit["generation_receipt_digest"]
            != bank_row["receipt_digest"]
            or scalar["generation_receipt_digest"]
            != bank_row["receipt_digest"]
            or scalar["event_audit_receipt_digest"] != audit["receipt_digest"]
            or scalar["frozen_scorer_receipt_digest"]
            != score["frozen_scorer_receipt_digest"]
            or scalar["raw_global_action_energy_score"]
            != score["raw_global_action_energy_score"]
            or _generation_checkpoint_trees(score) != {checkpoint_tree}
        ):
            raise PairV5MainlineCalibrationError(
                f"{candidate_id} formal scalar provenance join differs"
            )
        audits.append(audit)
        scalar_rows.append(scalar)
        validated_scores[candidate_id] = score
        score_bindings.append(
            {
                "candidate_id": candidate_id,
                "score_file_sha256": file_sha256(score_path),
                "score_receipt_digest": score["receipt_digest"],
                "event_audit_file_sha256": file_sha256(audit_path),
                "event_audit_receipt_digest": audit["receipt_digest"],
                "score_row_file_sha256": file_sha256(row_path),
                "score_row_digest": scalar["row_digest"],
                "raw_global_action_energy_score": score[
                    "raw_global_action_energy_score"
                ],
            }
        )

    candidates_by_cell: dict[str, list[dict[str, Any]]] = {}
    for _, candidate, _ in grouped_bound_rows:
        candidates_by_cell.setdefault(candidate["calibration_group_id"], []).append(
            candidate
        )
    for cell_id, candidates in candidates_by_cell.items():
        if [candidate["semantic_branch"] for candidate in candidates] != list(
            calibration.BRANCH_ORDER
        ):
            raise PairV5MainlineCalibrationError(
                "formal T2V cell branch order differs"
            )
        registry = {
            candidate["semantic_branch"]: validated_scores[
                candidate["candidate_id"]
            ]["generation_runtime_binding_by_branch"][candidate["semantic_branch"]]
            for candidate in candidates
        }
        for candidate in candidates:
            if (
                validated_scores[candidate["candidate_id"]][
                    "generation_runtime_binding_by_branch"
                ]
                != registry
            ):
                raise PairV5MainlineCalibrationError(
                    f"{cell_id} formal generation-runtime registry differs"
                )

    try:
        recomputed = calibration.calibrate_global_action_energy(
            scalar_rows,
            audits,
            prereg,
            source_bank_spec_sha256=spec_sha,
            source_bank_receipt_digest=bank_value["receipt_digest"],
        )
    except calibration.PairV5EnergyCalibrationV3Error as error:
        raise PairV5MainlineCalibrationError(str(error)) from error
    if stored_raw != recomputed:
        raise PairV5MainlineCalibrationError(
            "stored v3 calibration does not equal scalar-provenance recomputation"
        )
    if (
        recomputed["optimizer_authorized"] is not True
        or recomputed["failure_reasons"] != []
        or recomputed["t2v_media_as_rv2v_target_donor_input_or_noise"] is not False
        or recomputed["confirmation_rows_consumed_by_optimizer"] is not False
        or recomputed["scientific_action_editing_claim"] is not False
    ):
        raise PairV5MainlineCalibrationError(
            "formal frozen-T2V scalar calibration is not GO"
        )
    score_bindings.sort(key=lambda item: item["candidate_id"])
    score_set_digest = object_sha256(score_bindings)
    map_set_digest = object_sha256(
        {
            family: recomputed["mapping_by_family"][family]["mapping_digest"]
            for family in recomputed["action_family_order"]
        }
    )
    authorization_unsigned = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "source_bank_spec_sha256": spec_sha,
        "source_bank_receipt_digest": bank_value["receipt_digest"],
        "formal_score_provenance_set_digest": score_set_digest,
        "formal_score_schema": scorer.SCORE_RECEIPT_SCHEMA,
        "formal_score_filename": scorer.SCORE_RECEIPT_FILENAME,
        "formal_score_scalar_definition": (
            scorer.CANONICAL_ACTION_ENERGY_ARITHMETIC_SCHEMA
        ),
        "formal_score_arithmetic_contract_digest": (
            scorer.canonical_action_energy_arithmetic_contract()[
                "contract_digest"
            ]
        ),
        "preregistration_digest": prereg["preregistration_digest"],
        "calibration_receipt_digest": recomputed["receipt_digest"],
        "family_mapping_set_digest": map_set_digest,
        "checkpoint_tree_sha256": checkpoint_tree,
        "score_count": len(score_bindings),
        "branch_order": list(calibration.BRANCH_ORDER),
        "action_family_order": list(recomputed["action_family_order"]),
        "all_formal_scalar_provenance_recomputed": True,
        "formal_receipts_validated_by_active_v4_canonical_code": True,
        "active_repository_score_schema_consumed": True,
        "legacy_v3_compatibility_score_consumed": False,
        "initialization_ablation_teacher_or_adapter_artifact_consumed": False,
        "t2v_media_latent_gaussian_or_proposal_exported_to_native_scorer": False,
        "only_family_maps_threshold_prompts_and_scalar_digests_exported": True,
        "calibration_maps_authorized": True,
        "native_rv2v_optimizer_authorized": False,
        "scientific_action_editing_claim": False,
    }
    authorization = {
        **authorization_unsigned,
        "authorization_digest": object_sha256(authorization_unsigned),
    }
    return {
        "authorization": authorization,
        "t2v_spec": t2v_spec,
        "t2v_spec_path": str(spec_path),
        "t2v_spec_sha256": spec_sha,
        "t2v_bank_receipt_path": str(bank_path),
        "t2v_bank_receipt_file_sha256": bank_sha,
        "t2v_bank_receipt_digest": bank_value["receipt_digest"],
        "t2v_score_root": str(score_root_path),
        "formal_score_provenance_set_digest": score_set_digest,
        "formal_score_bindings": score_bindings,
        "formal_score_arithmetic_contract": (
            scorer.canonical_action_energy_arithmetic_contract()
        ),
        "calibration_root": str(scalar_root),
        "preregistration": prereg,
        "preregistration_path": str(prereg_path),
        "preregistration_file_sha256": prereg_sha,
        "calibration": recomputed,
        "calibration_path": str(calibration_path),
        "calibration_file_sha256": calibration_sha,
        "family_mapping_set_digest": map_set_digest,
        "checkpoint_tree_sha256": checkpoint_tree,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-spec", required=True)
    parser.add_argument("--expected-root-spec-sha256", required=True)
    parser.add_argument("--bank-receipt", required=True)
    parser.add_argument("--expected-bank-receipt-sha256", required=True)
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--expected-calibration-receipt-sha256", required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--checkpoint-tree-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = load_mainline_calibration_bundle(
        root_spec=args.root_spec,
        root_spec_sha256=args.expected_root_spec_sha256,
        bank_receipt=args.bank_receipt,
        bank_receipt_sha256=args.expected_bank_receipt_sha256,
        score_root=args.score_root,
        calibration_root=args.calibration_root,
        calibration_receipt_sha256=args.expected_calibration_receipt_sha256,
        preregistration_sha256=args.expected_preregistration_sha256,
        checkpoint_tree_sha256=args.checkpoint_tree_sha256,
    )
    print(canonical_json_bytes(bundle["authorization"]).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "PairV5MainlineCalibrationError",
    "load_mainline_calibration_bundle",
]
