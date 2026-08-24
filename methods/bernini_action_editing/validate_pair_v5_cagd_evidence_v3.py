#!/usr/bin/env python3
"""Recompute PAIR-v5 CAGD authorization from immutable v3 evidence.

This validator is the authority boundary that the legacy CAGD trainer lacks.
It does not trust eligibility booleans or arbitrary 64-hex receipt strings.
Instead it hashes every referenced file, revalidates the source bank and each
generation/score/audit join, reconstructs every v3 scalar row, reruns fit and
held-out confirmation calibration, re-seals each fit eligibility from the
stored clean latent and official Gaussian, and validates the final trainer
manifest.  Only the resulting authorization receipt may be used to launch
CAGD; a standalone legacy eligibility JSON is explicitly insufficient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import finalize_pair_v5_t2v_cagd_v3 as finalizer  # noqa: E402
import pair_v5_t2v_energy_calibration_v3 as calibration  # noqa: E402
import pair_v5_t2v_guidance_distill as guidance  # noqa: E402
import score_pair_v5_t2v_energy_bank_v3 as scorer  # noqa: E402
import train_pair_v5_t2v_guidance_distill as guidance_trainer  # noqa: E402


AUTHORIZATION_SCHEMA = "bernini-pair-v5-cagd-recomputed-authorization-v3"
_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "validator_contract",
        "source_bank_spec",
        "source_bank_receipt",
        "checkpoint_tree_sha256",
        "external_label_manifest",
        "preregistration",
        "score_receipt_files",
        "event_audit_receipt_files",
        "score_row_files",
        "calibration_receipt",
        "eligibility_receipt_files",
        "guidance_manifest",
        "expected_recomputed_optimizer_authorized",
        "expected_fit_action_candidate_ids",
        "confirmation_samples_consumed_by_optimizer",
        "scientific_action_editing_claim",
        "evidence_digest",
    }
)


class PairV5CAGDEvidenceError(RuntimeError):
    """The evidence graph cannot independently authorize CAGD."""


def _plain_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, str):
        raise PairV5CAGDEvidenceError(f"{label} path must be text")
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PairV5CAGDEvidenceError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _read_json_bound(
    value: Any, expected_sha256: Any, *, label: str
) -> tuple[dict[str, Any], Path]:
    path = _plain_file(value, label=label)
    if scorer.file_sha256(path) != expected_sha256:
        raise PairV5CAGDEvidenceError(f"{label} file SHA-256 differs")
    try:
        decoded = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PairV5CAGDEvidenceError(f"{label} is invalid ASCII JSON") from error
    if not isinstance(decoded, dict):
        raise PairV5CAGDEvidenceError(f"{label} root must be an object")
    return decoded, path


def _verify_embedded(
    value: Mapping[str, Any], *, field: str, label: str
) -> str:
    unsigned = dict(value)
    declared = unsigned.pop(field, None)
    if not isinstance(declared, str) or calibration.object_sha256(unsigned) != declared:
        raise PairV5CAGDEvidenceError(f"{label} embedded digest differs")
    return declared


def _binding_index(
    values: Any, *, digest_field: str, label: str
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(values, list) or any(not isinstance(row, Mapping) for row in values):
        raise PairV5CAGDEvidenceError(f"{label} must be a list of bindings")
    result = {row.get("candidate_id"): row for row in values}
    if len(result) != len(values) or any(not isinstance(key, str) for key in result):
        raise PairV5CAGDEvidenceError(f"{label} candidate IDs repeat")
    for candidate_id, row in result.items():
        _read_json_bound(row.get("path"), row.get("file_sha256"), label=f"{label} {candidate_id}")
        if not isinstance(row.get(digest_field), str):
            raise PairV5CAGDEvidenceError(f"{label} {candidate_id} digest differs")
    return result


def validate_evidence(
    evidence_path: str | Path,
    *,
    expected_evidence_sha256: str,
    checkpoint_tree_sha256: str,
) -> dict[str, Any]:
    evidence, resolved = _read_json_bound(
        str(evidence_path), expected_evidence_sha256, label="CAGD evidence"
    )
    evidence_digest = _verify_embedded(
        evidence, field="evidence_digest", label="CAGD evidence"
    )
    if set(evidence) != set(_EVIDENCE_FIELDS):
        raise PairV5CAGDEvidenceError("CAGD evidence field closure differs")
    if (
        evidence.get("schema_version") != finalizer.VALIDATOR_EVIDENCE_SCHEMA
        or evidence.get("scientific_action_editing_claim") is not False
    ):
        raise PairV5CAGDEvidenceError("CAGD evidence schema differs")
    if evidence.get("checkpoint_tree_sha256") != checkpoint_tree_sha256:
        raise PairV5CAGDEvidenceError("CAGD evidence checkpoint tree differs")
    expected_contract = {
        "recompute_calibration_from_raw_files": True,
        "legacy_eligibility_self_declared_booleans_are_insufficient": True,
        "every_file_sha256_revalidated": True,
        "source_bank_generation_bindings_revalidated": True,
        "fit_confirmation_raw_scores_and_thresholds_revalidated": True,
        "event_audit_artifacts_are_external_detached_hash_bindings": True,
        "confirmation_samples_may_enter_optimizer": False,
    }
    if evidence.get("validator_contract") != expected_contract:
        raise PairV5CAGDEvidenceError("validator contract differs")
    if evidence.get("confirmation_samples_consumed_by_optimizer") is not False:
        raise PairV5CAGDEvidenceError("confirmation samples entered optimizer evidence")

    spec_binding = evidence.get("source_bank_spec")
    bank_binding = evidence.get("source_bank_receipt")
    label_binding = evidence.get("external_label_manifest")
    prereg_binding = evidence.get("preregistration")
    calibration_binding = evidence.get("calibration_receipt")
    for name, binding in (
        ("source bank spec", spec_binding),
        ("source bank receipt", bank_binding),
        ("external label manifest", label_binding),
        ("preregistration", prereg_binding),
        ("calibration receipt", calibration_binding),
    ):
        if not isinstance(binding, Mapping):
            raise PairV5CAGDEvidenceError(f"{name} binding differs")
    spec_path = _plain_file(spec_binding["path"], label="source bank spec")
    if scorer.file_sha256(spec_path) != spec_binding["file_sha256"]:
        raise PairV5CAGDEvidenceError("source bank spec SHA-256 differs")
    bank_path = _plain_file(bank_binding["path"], label="source bank receipt")
    if scorer.file_sha256(bank_path) != bank_binding["file_sha256"]:
        raise PairV5CAGDEvidenceError("source bank receipt SHA-256 differs")

    # Revalidate both complete generation groups, including every stored
    # latent/Gaussian artifact and every generation receipt/hash join.
    generation_by_id: dict[str, dict[str, Any]] = {}
    for group_id in ("sp4-a", "sp4-b"):
        try:
            _spec, bank, rows = scorer.load_group_bank(
                root_spec=spec_path,
                root_spec_sha256=spec_binding["file_sha256"],
                bank_output_dir=bank_path.parent,
                bank_receipt=bank_path,
                bank_receipt_sha256=bank_binding["file_sha256"],
                group_id=group_id,
            )
        except scorer.PairV5T2VEnergyScoringError as error:
            raise PairV5CAGDEvidenceError(str(error)) from error
        if bank["receipt_digest"] != bank_binding["receipt_digest"]:
            raise PairV5CAGDEvidenceError("bank embedded digest differs")
        for row in rows:
            candidate_id = row["candidate"]["candidate_id"]
            if candidate_id in generation_by_id:
                raise PairV5CAGDEvidenceError("generation candidate repeats")
            generation_by_id[candidate_id] = row

    label_manifest, _ = finalizer.load_external_label_manifest(
        label_binding["path"], expected_sha256=label_binding["file_sha256"]
    )
    if (
        label_manifest["manifest_digest"] != label_binding["manifest_digest"]
        or label_manifest["bank_receipt_digest"] != bank_binding["receipt_digest"]
    ):
        raise PairV5CAGDEvidenceError("external label manifest binding differs")
    label_by_id = {row["candidate_id"]: row for row in label_manifest["rows"]}
    if set(label_by_id) != set(generation_by_id):
        raise PairV5CAGDEvidenceError("external labels do not cover the generation bank")
    for candidate_id, label in label_by_id.items():
        artifact = _plain_file(
            label["external_audit_artifact_path"],
            label=f"{candidate_id} external audit artifact",
        )
        if scorer.file_sha256(artifact) != label["external_audit_artifact_sha256"]:
            raise PairV5CAGDEvidenceError(
                f"{candidate_id} external audit artifact changed"
            )
        if label["generation_receipt_digest"] != generation_by_id[candidate_id][
            "generation_receipt_digest"
        ]:
            raise PairV5CAGDEvidenceError(f"{candidate_id} label/generation join differs")

    score_bindings = _binding_index(
        evidence.get("score_receipt_files"),
        digest_field="receipt_digest",
        label="score receipt",
    )
    audit_bindings = _binding_index(
        evidence.get("event_audit_receipt_files"),
        digest_field="receipt_digest",
        label="event audit receipt",
    )
    row_bindings = _binding_index(
        evidence.get("score_row_files"),
        digest_field="row_digest",
        label="score row",
    )
    if not (
        set(score_bindings)
        == set(audit_bindings)
        == set(row_bindings)
        == set(generation_by_id)
    ):
        raise PairV5CAGDEvidenceError("score/audit/row/generation file closure differs")

    scores: dict[str, dict[str, Any]] = {}
    audits: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for candidate_id in sorted(generation_by_id):
        score_raw, _ = _read_json_bound(
            score_bindings[candidate_id]["path"],
            score_bindings[candidate_id]["file_sha256"],
            label=f"score receipt {candidate_id}",
        )
        try:
            score = scorer.validate_score_receipt(score_raw)
        except scorer.PairV5T2VEnergyScoringError as error:
            raise PairV5CAGDEvidenceError(str(error)) from error
        audit_raw, _ = _read_json_bound(
            audit_bindings[candidate_id]["path"],
            audit_bindings[candidate_id]["file_sha256"],
            label=f"event audit receipt {candidate_id}",
        )
        row_raw, _ = _read_json_bound(
            row_bindings[candidate_id]["path"],
            row_bindings[candidate_id]["file_sha256"],
            label=f"score row {candidate_id}",
        )
        try:
            audit = calibration.validate_event_audit_receipt(audit_raw)
            row = calibration.validate_score_row(row_raw)
        except calibration.PairV5EnergyCalibrationV3Error as error:
            raise PairV5CAGDEvidenceError(str(error)) from error
        generation = generation_by_id[candidate_id]
        candidate = generation["candidate"]
        if (
            score["receipt_digest"] != score_bindings[candidate_id]["receipt_digest"]
            or audit["receipt_digest"] != audit_bindings[candidate_id]["receipt_digest"]
            or row["row_digest"] != row_bindings[candidate_id]["row_digest"]
            or score_bindings[candidate_id]["raw_global_action_energy_score"]
            != score["raw_global_action_energy_score"]
            or score["raw_global_action_energy_score"]
            != row["raw_global_action_energy_score"]
            or score["generation_receipt_digest"]
            != generation["generation_receipt_digest"]
            or audit["generation_receipt_digest"]
            != generation["generation_receipt_digest"]
            or row["generation_receipt_digest"]
            != generation["generation_receipt_digest"]
            or row["event_audit_receipt_digest"] != audit["receipt_digest"]
            or row["frozen_scorer_receipt_digest"]
            != score["frozen_scorer_receipt_digest"]
            or any(
                row[field] != candidate[field]
                for field in (
                    "candidate_id",
                    "analysis_split",
                    "action_family_id",
                    "calibration_group_id",
                    "actor_group_id",
                    "scene_group_id",
                    "action_group_id",
                    "semantic_branch",
                )
            )
        ):
            raise PairV5CAGDEvidenceError(f"{candidate_id} evidence join differs")
        scores[candidate_id] = score
        audits.append(audit)
        rows.append(row)

    prereg_raw, _ = _read_json_bound(
        prereg_binding["path"],
        prereg_binding["file_sha256"],
        label="preregistration",
    )
    try:
        prereg = calibration.validate_preregistration(prereg_raw)
    except calibration.PairV5EnergyCalibrationV3Error as error:
        raise PairV5CAGDEvidenceError(str(error)) from error
    if prereg["preregistration_digest"] != prereg_binding["preregistration_digest"]:
        raise PairV5CAGDEvidenceError("preregistration digest binding differs")
    try:
        recomputed = calibration.calibrate_global_action_energy(
            rows,
            audits,
            prereg,
            source_bank_spec_sha256=spec_binding["file_sha256"],
            source_bank_receipt_digest=bank_binding["receipt_digest"],
        )
    except calibration.PairV5EnergyCalibrationV3Error as error:
        raise PairV5CAGDEvidenceError(str(error)) from error
    stored_calibration, _ = _read_json_bound(
        calibration_binding["path"],
        calibration_binding["file_sha256"],
        label="calibration receipt",
    )
    if (
        stored_calibration != recomputed
        or recomputed["receipt_digest"] != calibration_binding["receipt_digest"]
        or recomputed["raw_score_evidence_by_family"]
        != calibration_binding["raw_score_evidence_by_family"]
        or recomputed["decision_threshold"] != calibration_binding["decision_threshold"]
        or recomputed["confirmation_thresholds"]
        != calibration_binding["confirmation_thresholds"]
        or recomputed["optimizer_authorized"]
        != evidence["expected_recomputed_optimizer_authorized"]
        or recomputed["fit_event_qualified_action_candidate_ids"]
        != evidence["expected_fit_action_candidate_ids"]
    ):
        raise PairV5CAGDEvidenceError("recomputed calibration differs from evidence")

    eligibility_values: dict[str, Mapping[str, Any]] = {}
    eligibility_bindings = evidence.get("eligibility_receipt_files")
    if not isinstance(eligibility_bindings, list):
        raise PairV5CAGDEvidenceError("eligibility bindings differ")
    for binding in eligibility_bindings:
        if not isinstance(binding, Mapping):
            raise PairV5CAGDEvidenceError("eligibility binding is not an object")
        candidate_id = binding.get("candidate_id")
        stored, _ = _read_json_bound(
            binding.get("path"),
            binding.get("file_sha256"),
            label=f"eligibility {candidate_id}",
        )
        generation = generation_by_id.get(candidate_id)
        score = scores.get(candidate_id)
        audit = next((item for item in audits if item["candidate_id"] == candidate_id), None)
        if generation is None or score is None or audit is None:
            raise PairV5CAGDEvidenceError("eligibility candidate is outside evidence")
        clean = scorer._load_exact81_tensor(
            generation["artifacts"]["predecode_clean_latent"],
            key="normalized_clean_latent",
            label=f"{candidate_id} eligibility clean latent",
        )
        epsilon = scorer._load_exact81_tensor(
            generation["artifacts"]["official_initial_gaussian"],
            key="official_initial_gaussian",
            label=f"{candidate_id} eligibility Gaussian",
        )
        expected = guidance.seal_eligibility(
            sample_id=candidate_id,
            action_family=generation["candidate"]["action_family_id"],
            analysis_split="fit",
            event_latent=clean,
            official_epsilon=epsilon,
            official_gaussian_artifact_sha256=generation["artifacts"][
                "official_initial_gaussian"
            ]["sha256"],
            checkpoint_tree_sha256=checkpoint_tree_sha256,
            prompt_by_branch=score["prompt_by_branch"],
            event_qualified=True,
            calibration_confirmation_passed=recomputed["optimizer_authorized"],
            calibration_optimizer_authorized=recomputed["optimizer_authorized"],
            event_qualification_receipt_digest=audit["receipt_digest"],
            calibration_receipt_digest=recomputed["receipt_digest"],
        )
        expected_value = {**expected.payload(), "receipt_digest": expected.receipt_digest}
        if (
            stored != expected_value
            or binding.get("receipt_digest") != expected.receipt_digest
            or binding.get("optimizer_authorized") != expected.optimizer_authorized
        ):
            raise PairV5CAGDEvidenceError(
                f"{candidate_id} eligibility is not derivable from source evidence"
            )
        eligibility_values[candidate_id] = stored

    guidance_binding = evidence.get("guidance_manifest")
    if recomputed["optimizer_authorized"]:
        if not isinstance(guidance_binding, Mapping):
            raise PairV5CAGDEvidenceError("passing calibration lacks guidance manifest")
        manifest = guidance_trainer.load_manifest(
            guidance_binding["path"], guidance_binding["file_sha256"]
        )
        if (
            manifest.manifest_digest != guidance_binding["manifest_digest"]
            or {event.event_id for event in manifest.events} != set(eligibility_values)
            or any(event.analysis_split != "fit" for event in manifest.events)
        ):
            raise PairV5CAGDEvidenceError("guidance manifest evidence differs")
        # Tensor loading rechecks eligibility hashes and the latent/Gaussian
        # identities used by the trainer itself.
        guidance_trainer.load_event_tensors(manifest)
    elif guidance_binding is not None:
        raise PairV5CAGDEvidenceError("failed calibration emitted a guidance manifest")

    authorization = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "evidence_path": str(resolved),
        "evidence_file_sha256": expected_evidence_sha256,
        "evidence_digest": evidence_digest,
        "source_bank_spec_sha256": spec_binding["file_sha256"],
        "source_bank_receipt_digest": bank_binding["receipt_digest"],
        "recomputed_calibration_receipt_digest": recomputed["receipt_digest"],
        "guidance_manifest_file_sha256": (
            guidance_binding["file_sha256"] if isinstance(guidance_binding, Mapping) else None
        ),
        "fit_event_count": len(eligibility_values),
        "confirmation_event_count_for_optimizer": 0,
        "legacy_eligibility_self_declaration_trusted": False,
        "all_source_files_and_receipts_revalidated": True,
        "calibration_recomputed_from_raw_global_scores": True,
        "optimizer_authorized": recomputed["optimizer_authorized"],
        "scientific_action_editing_claim": False,
    }
    return {**authorization, "authorization_digest": calibration.object_sha256(authorization)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--expected-evidence-sha256", required=True)
    parser.add_argument("--checkpoint-tree-sha256", required=True)
    parser.add_argument("--authorization-output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = validate_evidence(
        args.evidence,
        expected_evidence_sha256=args.expected_evidence_sha256,
        checkpoint_tree_sha256=args.checkpoint_tree_sha256,
    )
    if args.authorization_output:
        path = Path(args.authorization_output)
        if not path.is_absolute() or path.exists() or path.is_symlink() or path == Path("/"):
            raise PairV5CAGDEvidenceError(
                "authorization output must be a fresh absolute file"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(calibration.canonical_json_bytes(receipt) + b"\n")
        os.chmod(path, 0o400)
    print(calibration.canonical_json_bytes(receipt).decode("ascii"), flush=True)
    # NO-GO is a scientifically valid validator completion, not a shell error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "PairV5CAGDEvidenceError",
    "validate_evidence",
]
