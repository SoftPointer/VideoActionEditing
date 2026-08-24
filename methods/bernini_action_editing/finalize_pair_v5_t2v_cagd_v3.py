#!/usr/bin/env python3
"""Finalize PAIR-v5 event audit, v3 calibration, and CAGD eligibility.

The input label manifest is an external scientific annotation artifact.  Each
row points to a hash-bound manual/VLM sidecar and carries only four detached
booleans.  This program verifies those hashes, binds one event-audit receipt
to every rendered branch, joins the receipts to frozen global-MACE scores,
and performs the preregistered fit/confirmation calibration.

Only event-qualified *fit action* generations may receive a CAGD
``GuidanceEligibility``.  In the core4 pilot those are exactly IID
``7b88a1ca1f804f41`` (dog) and ``a35b590961d24694`` (human).  Confirmation
samples are metric-only and are never written to the optimizer manifest.  A
failed gate still produces a complete NO-GO calibration receipt and, where
applicable, explicitly ineligible fit receipts; it never produces an
optimizer-authorized event manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import pair_v5_t2v_energy_calibration_v3 as calibration  # noqa: E402
import pair_v5_t2v_guidance_distill as guidance  # noqa: E402
import score_pair_v5_t2v_energy_bank_v3 as scorer_runtime  # noqa: E402
import train_pair_v5_t2v_guidance_distill as guidance_trainer  # noqa: E402


LABEL_MANIFEST_SCHEMA = "bernini-pair-v5-external-event-label-manifest-v3"
VALIDATOR_EVIDENCE_SCHEMA = "bernini-pair-v5-cagd-validator-evidence-v3"
FINALIZATION_RECEIPT_SCHEMA = "bernini-pair-v5-cagd-finalization-receipt-v3"
CORE4_FIT_ACTION_IIDS = ("7b88a1ca1f804f41", "a35b590961d24694")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_LABEL_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "bank_receipt_digest",
        "candidate_count",
        "rows",
        "labels_are_external_and_detached",
        "labels_may_enter_model_condition",
        "manifest_digest",
    }
)
_LABEL_ROW_FIELDS = frozenset(
    {
        "candidate_id",
        "generation_receipt_digest",
        "audit_source_kind",
        "external_audit_artifact_path",
        "external_audit_artifact_sha256",
        "complete_target_transition_observed",
        "terminal_hold_observed",
        "full_target_action_observed",
        "full_target_action_false_confirmed",
    }
)


class PairV5CAGDFinalizationError(RuntimeError):
    """A score, label, generation, or CAGD binding failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return calibration.canonical_json_bytes(value)


def object_sha256(value: Any) -> str:
    return calibration.object_sha256(value)


def file_sha256(path: Path) -> str:
    return scorer_runtime.file_sha256(path)


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairV5CAGDFinalizationError(f"{label} must be lowercase SHA-256")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PairV5CAGDFinalizationError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _strict_json(
    value: str | Path, *, expected_sha256: str, label: str
) -> tuple[dict[str, Any], Path]:
    path = _plain_file(value, label=label)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _sha256(
        expected_sha256, label=f"{label} SHA-256"
    ):
        raise PairV5CAGDFinalizationError(f"{label} SHA-256 differs")

    def reject_constant(token: str) -> None:
        raise PairV5CAGDFinalizationError(f"{label} contains {token}")

    def reject_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise PairV5CAGDFinalizationError(f"{label} duplicate key {key!r}")
            result[key] = item
        return result

    try:
        decoded = json.loads(
            raw.decode("ascii"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PairV5CAGDFinalizationError(f"{label} is invalid ASCII JSON") from error
    if not isinstance(decoded, dict):
        raise PairV5CAGDFinalizationError(f"{label} root must be an object")
    return decoded, path


def make_external_label_manifest(
    *, bank_receipt_digest: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Create a sealed authoring manifest; artifact files are checked later."""

    normalized: list[dict[str, Any]] = []
    for ordinal, value in enumerate(rows):
        if not isinstance(value, Mapping) or set(value) != set(_LABEL_ROW_FIELDS):
            raise PairV5CAGDFinalizationError(f"label row {ordinal} fields differ")
        row = dict(value)
        if row["audit_source_kind"] not in calibration.AUDIT_SOURCE_KINDS:
            raise PairV5CAGDFinalizationError(f"label row {ordinal} audit source differs")
        path = Path(row["external_audit_artifact_path"])
        if not path.is_absolute() or path == Path("/"):
            raise PairV5CAGDFinalizationError(
                f"label row {ordinal} audit artifact path differs"
            )
        for name in (
            "complete_target_transition_observed",
            "terminal_hold_observed",
            "full_target_action_observed",
            "full_target_action_false_confirmed",
        ):
            if type(row[name]) is not bool:
                raise PairV5CAGDFinalizationError(
                    f"label row {ordinal} {name} must be boolean"
                )
        _sha256(row["generation_receipt_digest"], label="generation receipt digest")
        _sha256(
            row["external_audit_artifact_sha256"], label="audit artifact SHA-256"
        )
        normalized.append(row)
    candidate_ids = [row["candidate_id"] for row in normalized]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise PairV5CAGDFinalizationError("label candidate IDs repeat")
    unsigned = {
        "schema_version": LABEL_MANIFEST_SCHEMA,
        "bank_receipt_digest": _sha256(
            bank_receipt_digest, label="bank receipt digest"
        ),
        "candidate_count": len(normalized),
        "rows": normalized,
        "labels_are_external_and_detached": True,
        "labels_may_enter_model_condition": False,
    }
    return {**unsigned, "manifest_digest": object_sha256(unsigned)}


def load_external_label_manifest(
    value: str | Path, *, expected_sha256: str
) -> tuple[dict[str, Any], Path]:
    root, path = _strict_json(
        value, expected_sha256=expected_sha256, label="external label manifest"
    )
    if set(root) != set(_LABEL_ROOT_FIELDS):
        raise PairV5CAGDFinalizationError("external label manifest fields differ")
    unsigned = dict(root)
    declared = _sha256(unsigned.pop("manifest_digest"), label="label manifest digest")
    if object_sha256(unsigned) != declared:
        raise PairV5CAGDFinalizationError("external label manifest digest differs")
    rebuilt = make_external_label_manifest(
        bank_receipt_digest=root["bank_receipt_digest"], rows=root["rows"]
    )
    if root != rebuilt:
        raise PairV5CAGDFinalizationError("external label manifest semantics differ")
    if (
        root["schema_version"] != LABEL_MANIFEST_SCHEMA
        or root["candidate_count"] != len(root["rows"])
        or root["labels_are_external_and_detached"] is not True
        or root["labels_may_enter_model_condition"] is not False
    ):
        raise PairV5CAGDFinalizationError("external label information flow differs")
    return root, path


def _score_path(score_root: Path, group_id: str, candidate_id: str) -> Path:
    return (
        score_root
        / group_id
        / candidate_id
        / scorer_runtime.SCORE_RECEIPT_FILENAME
    )


def _load_score(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PairV5CAGDFinalizationError(f"score receipt is invalid: {path}") from error
    try:
        return scorer_runtime.validate_score_receipt(value)
    except scorer_runtime.PairV5T2VEnergyScoringError as error:
        raise PairV5CAGDFinalizationError(str(error)) from error


def _load_tensor(artifact: Mapping[str, Any], *, key: str, label: str) -> Any:
    return scorer_runtime._load_exact81_tensor(artifact, key=key, label=label)


def _write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise PairV5CAGDFinalizationError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    os.chmod(path, 0o400)
    return file_sha256(path)


def _guidance_event(
    *,
    candidate: Mapping[str, Any],
    prompt_by_branch: Mapping[str, str],
    clean_artifact: Mapping[str, Any],
    gaussian_artifact: Mapping[str, Any],
    eligibility_path: Path,
    eligibility_file_sha256: str,
) -> dict[str, Any]:
    unsigned = {
        "schema_version": guidance_trainer.EVENT_SCHEMA,
        "event_id": candidate["candidate_id"],
        "action_family": candidate["action_family_id"],
        "analysis_split": "fit",
        "prompt_by_branch": dict(prompt_by_branch),
        "prompt_bank_sha256": guidance.prompt_bank_sha256(prompt_by_branch),
        "clean_latent_path": clean_artifact["path"],
        "clean_latent_file_sha256": clean_artifact["sha256"],
        "clean_latent_tensor_key": "normalized_clean_latent",
        "official_gaussian_path": gaussian_artifact["path"],
        "official_gaussian_file_sha256": gaussian_artifact["sha256"],
        "official_gaussian_tensor_key": "official_initial_gaussian",
        "eligibility_receipt_path": str(eligibility_path),
        "eligibility_receipt_file_sha256": eligibility_file_sha256,
    }
    return {**unsigned, "event_digest": guidance_trainer.object_sha256(unsigned)}


def finalize(
    *,
    root_spec: str | Path,
    root_spec_sha256: str,
    bank_output_dir: str | Path,
    bank_receipt: str | Path,
    bank_receipt_sha256: str,
    score_root: str | Path,
    external_label_manifest: str | Path,
    external_label_manifest_sha256: str,
    checkpoint_tree_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    _sha256(checkpoint_tree_sha256, label="checkpoint tree SHA-256")
    output = Path(output_dir)
    if not output.is_absolute() or output == Path("/") or output.exists() or output.is_symlink():
        raise PairV5CAGDFinalizationError("finalization output must be a fresh absolute directory")
    score_root_path = Path(score_root)
    if not score_root_path.is_absolute() or not score_root_path.is_dir() or score_root_path.is_symlink():
        raise PairV5CAGDFinalizationError("score root must be an absolute plain directory")

    groups: dict[str, list[dict[str, Any]]] = {}
    bank_value: Optional[dict[str, Any]] = None
    for group_id in ("sp4-a", "sp4-b"):
        try:
            _spec, loaded_bank, rows = scorer_runtime.load_group_bank(
                root_spec=root_spec,
                root_spec_sha256=root_spec_sha256,
                bank_output_dir=bank_output_dir,
                bank_receipt=bank_receipt,
                bank_receipt_sha256=bank_receipt_sha256,
                group_id=group_id,
            )
        except scorer_runtime.PairV5T2VEnergyScoringError as error:
            raise PairV5CAGDFinalizationError(str(error)) from error
        if bank_value is not None and (
            loaded_bank["receipt_digest"] != bank_value["receipt_digest"]
            or loaded_bank["file_sha256"] != bank_value["file_sha256"]
        ):
            raise PairV5CAGDFinalizationError("two group bank bindings differ")
        bank_value = loaded_bank
        groups[group_id] = rows
    assert bank_value is not None

    labels, _label_path = load_external_label_manifest(
        external_label_manifest, expected_sha256=external_label_manifest_sha256
    )
    if labels["bank_receipt_digest"] != bank_value["receipt_digest"]:
        raise PairV5CAGDFinalizationError("label manifest binds another bank")
    label_by_id = {row["candidate_id"]: row for row in labels["rows"]}
    all_rows = [row for rows in groups.values() for row in rows]
    if set(label_by_id) != {row["candidate"]["candidate_id"] for row in all_rows}:
        raise PairV5CAGDFinalizationError("label candidate coverage differs")

    output.mkdir(parents=True)
    audit_receipts: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    score_by_candidate: dict[str, dict[str, Any]] = {}
    bound_by_candidate: dict[str, dict[str, Any]] = {}
    audit_file_bindings: list[dict[str, Any]] = []
    score_file_bindings: list[dict[str, Any]] = []
    for group_id, rows in groups.items():
        for bound in rows:
            candidate = bound["candidate"]
            candidate_id = candidate["candidate_id"]
            label = label_by_id[candidate_id]
            artifact = _plain_file(
                label["external_audit_artifact_path"],
                label=f"{candidate_id} external audit artifact",
            )
            if file_sha256(artifact) != label["external_audit_artifact_sha256"]:
                raise PairV5CAGDFinalizationError(
                    f"{candidate_id} external audit artifact SHA-256 differs"
                )
            if label["generation_receipt_digest"] != bound["generation_receipt_digest"]:
                raise PairV5CAGDFinalizationError(
                    f"{candidate_id} label/generation binding differs"
                )
            audit = calibration.seal_event_audit_receipt(
                candidate_id=candidate_id,
                analysis_split=candidate["analysis_split"],
                action_family_id=candidate["action_family_id"],
                calibration_group_id=candidate["calibration_group_id"],
                actor_group_id=candidate["actor_group_id"],
                scene_group_id=candidate["scene_group_id"],
                action_group_id=candidate["action_group_id"],
                semantic_branch=candidate["semantic_branch"],
                generation_receipt_digest=bound["generation_receipt_digest"],
                audit_source_kind=label["audit_source_kind"],
                external_audit_artifact_sha256=label[
                    "external_audit_artifact_sha256"
                ],
                complete_target_transition_observed=label[
                    "complete_target_transition_observed"
                ],
                terminal_hold_observed=label["terminal_hold_observed"],
                full_target_action_observed=label["full_target_action_observed"],
                full_target_action_false_confirmed=label[
                    "full_target_action_false_confirmed"
                ],
            )
            audit_path = output / "event-audits" / f"{candidate_id}.json"
            audit_sha = _write_create_only(audit_path, audit)
            audit_file_bindings.append(
                {
                    "candidate_id": candidate_id,
                    "path": str(audit_path),
                    "file_sha256": audit_sha,
                    "receipt_digest": audit["receipt_digest"],
                }
            )
            score_path = _score_path(score_root_path, group_id, candidate_id)
            score = _load_score(_plain_file(score_path, label=f"{candidate_id} score"))
            score_file_bindings.append(
                {
                    "candidate_id": candidate_id,
                    "path": str(score_path),
                    "file_sha256": file_sha256(score_path),
                    "receipt_digest": score["receipt_digest"],
                    "raw_global_action_energy_score": score[
                        "raw_global_action_energy_score"
                    ],
                }
            )
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
                or score["root_spec_raw_sha256"] != root_spec_sha256
                or score["bank_receipt_digest"] != bank_value["receipt_digest"]
                or score["generation_receipt_digest"]
                != bound["generation_receipt_digest"]
                or score["official_gaussian_artifact_sha256"]
                != bound["artifacts"]["official_initial_gaussian"]["sha256"]
            ):
                raise PairV5CAGDFinalizationError(
                    f"{candidate_id} score/generation binding differs"
                )
            row = calibration.make_score_row(
                row_id=f"global-energy-{candidate_id}",
                candidate_id=candidate_id,
                analysis_split=candidate["analysis_split"],
                action_family_id=candidate["action_family_id"],
                calibration_group_id=candidate["calibration_group_id"],
                actor_group_id=candidate["actor_group_id"],
                scene_group_id=candidate["scene_group_id"],
                action_group_id=candidate["action_group_id"],
                semantic_branch=candidate["semantic_branch"],
                raw_global_action_energy_score=score[
                    "raw_global_action_energy_score"
                ],
                generation_receipt_digest=bound["generation_receipt_digest"],
                frozen_scorer_receipt_digest=score[
                    "frozen_scorer_receipt_digest"
                ],
                event_audit_receipt_digest=audit["receipt_digest"],
            )
            audit_receipts.append(audit)
            score_rows.append(row)
            score_by_candidate[candidate_id] = score
            bound_by_candidate[candidate_id] = bound

    families = sorted({row["action_family_id"] for row in score_rows})
    prereg = calibration.make_preregistration(
        "pair5-core4-global-energy-index33-v3", families
    )
    calibration_receipt = calibration.calibrate_global_action_energy(
        score_rows,
        audit_receipts,
        prereg,
        source_bank_spec_sha256=root_spec_sha256,
        source_bank_receipt_digest=bank_value["receipt_digest"],
    )
    prereg_path = output / "preregistration-v3.json"
    prereg_sha = _write_create_only(prereg_path, prereg)
    row_file_bindings: list[dict[str, Any]] = []
    for row in score_rows:
        row_path = output / "score-rows" / f"{row['candidate_id']}.json"
        row_sha = _write_create_only(row_path, row)
        row_file_bindings.append(
            {
                "candidate_id": row["candidate_id"],
                "path": str(row_path),
                "file_sha256": row_sha,
                "row_digest": row["row_digest"],
            }
        )
    calibration_path = output / "calibration-receipt-v3.json"
    calibration_sha = _write_create_only(calibration_path, calibration_receipt)

    fit_action_audits = [
        audit
        for audit in audit_receipts
        if audit["analysis_split"] == "fit"
        and audit["semantic_branch"] == calibration.ACTION_BRANCH
        and audit["event_qualified_action_positive"] is True
    ]
    # The pilot may never silently substitute another fit event.
    unexpected = [
        audit["candidate_id"]
        for audit in fit_action_audits
        if not any(iid in audit["candidate_id"] for iid in CORE4_FIT_ACTION_IIDS)
    ]
    if unexpected:
        raise PairV5CAGDFinalizationError(
            f"unexpected core4 optimizer fit candidates: {unexpected}"
        )
    fit_action_audits.sort(
        key=lambda audit: next(
            index
            for index, iid in enumerate(CORE4_FIT_ACTION_IIDS)
            if iid in audit["candidate_id"]
        )
    )
    event_rows: list[dict[str, Any]] = []
    eligibility_receipts: list[dict[str, Any]] = []
    eligibility_file_bindings: list[dict[str, Any]] = []
    for audit in fit_action_audits:
        candidate_id = audit["candidate_id"]
        bound = bound_by_candidate[candidate_id]
        candidate = bound["candidate"]
        score = score_by_candidate[candidate_id]
        clean_artifact = bound["artifacts"]["predecode_clean_latent"]
        gaussian_artifact = bound["artifacts"]["official_initial_gaussian"]
        clean = _load_tensor(
            clean_artifact,
            key="normalized_clean_latent",
            label=f"{candidate_id} eligibility clean latent",
        )
        epsilon = _load_tensor(
            gaussian_artifact,
            key="official_initial_gaussian",
            label=f"{candidate_id} eligibility Gaussian",
        )
        eligibility = guidance.seal_eligibility(
            sample_id=candidate_id,
            action_family=candidate["action_family_id"],
            analysis_split="fit",
            event_latent=clean,
            official_epsilon=epsilon,
            official_gaussian_artifact_sha256=gaussian_artifact["sha256"],
            checkpoint_tree_sha256=checkpoint_tree_sha256,
            prompt_by_branch=score["prompt_by_branch"],
            event_qualified=True,
            calibration_confirmation_passed=calibration_receipt[
                "optimizer_authorized"
            ],
            calibration_optimizer_authorized=calibration_receipt[
                "optimizer_authorized"
            ],
            event_qualification_receipt_digest=audit["receipt_digest"],
            calibration_receipt_digest=calibration_receipt["receipt_digest"],
        )
        eligibility_value = {
            **eligibility.payload(),
            "receipt_digest": eligibility.receipt_digest,
        }
        eligibility_path = output / "eligibility" / f"{candidate_id}.json"
        eligibility_sha = _write_create_only(eligibility_path, eligibility_value)
        eligibility_receipts.append(eligibility_value)
        eligibility_file_bindings.append(
            {
                "candidate_id": candidate_id,
                "path": str(eligibility_path),
                "file_sha256": eligibility_sha,
                "receipt_digest": eligibility.receipt_digest,
                "optimizer_authorized": eligibility.optimizer_authorized,
            }
        )
        if eligibility.optimizer_authorized:
            event_rows.append(
                _guidance_event(
                    candidate=candidate,
                    prompt_by_branch=score["prompt_by_branch"],
                    clean_artifact=clean_artifact,
                    gaussian_artifact=gaussian_artifact,
                    eligibility_path=eligibility_path,
                    eligibility_file_sha256=eligibility_sha,
                )
            )

    guidance_manifest_path: Optional[Path] = None
    guidance_manifest_sha256: Optional[str] = None
    guidance_manifest_digest: Optional[str] = None
    if calibration_receipt["optimizer_authorized"]:
        if len(event_rows) != 2 or not all(
            any(iid in row["event_id"] for row in event_rows)
            for iid in CORE4_FIT_ACTION_IIDS
        ):
            raise PairV5CAGDFinalizationError(
                "passing calibration lacks exactly dog+human fit action events"
            )
        root_unsigned = {
            "schema_version": guidance_trainer.MANIFEST_SCHEMA,
            "optimizer_authorized": True,
            "checkpoint_tree_sha256": checkpoint_tree_sha256,
            "event_count": len(event_rows),
            "events": event_rows,
            "input_closure": guidance_trainer._INPUT_CLOSURE,
        }
        manifest = {
            **root_unsigned,
            "manifest_digest": guidance_trainer.object_sha256(root_unsigned),
        }
        guidance_manifest_digest = manifest["manifest_digest"]
        guidance_manifest_path = output / "pair-v5-t2v-guidance-event-manifest-v1.json"
        guidance_manifest_sha256 = _write_create_only(guidance_manifest_path, manifest)

    evidence_unsigned = {
        "schema_version": VALIDATOR_EVIDENCE_SCHEMA,
        "validator_contract": {
            "recompute_calibration_from_raw_files": True,
            "legacy_eligibility_self_declared_booleans_are_insufficient": True,
            "every_file_sha256_revalidated": True,
            "source_bank_generation_bindings_revalidated": True,
            "fit_confirmation_raw_scores_and_thresholds_revalidated": True,
            "event_audit_artifacts_are_external_detached_hash_bindings": True,
            "confirmation_samples_may_enter_optimizer": False,
        },
        "source_bank_spec": {
            "path": str(Path(root_spec)),
            "file_sha256": root_spec_sha256,
        },
        "source_bank_receipt": {
            "path": str(Path(bank_receipt)),
            "file_sha256": bank_receipt_sha256,
            "receipt_digest": bank_value["receipt_digest"],
        },
        "checkpoint_tree_sha256": checkpoint_tree_sha256,
        "external_label_manifest": {
            "path": str(Path(external_label_manifest)),
            "file_sha256": external_label_manifest_sha256,
            "manifest_digest": labels["manifest_digest"],
        },
        "preregistration": {
            "path": str(prereg_path),
            "file_sha256": prereg_sha,
            "preregistration_digest": prereg["preregistration_digest"],
        },
        "score_receipt_files": sorted(
            score_file_bindings, key=lambda item: item["candidate_id"]
        ),
        "event_audit_receipt_files": sorted(
            audit_file_bindings, key=lambda item: item["candidate_id"]
        ),
        "score_row_files": sorted(
            row_file_bindings, key=lambda item: item["candidate_id"]
        ),
        "calibration_receipt": {
            "path": str(calibration_path),
            "file_sha256": calibration_sha,
            "receipt_digest": calibration_receipt["receipt_digest"],
            "score_field": "raw_global_action_energy_score",
            "raw_score_evidence_by_family": calibration_receipt[
                "raw_score_evidence_by_family"
            ],
            "decision_threshold": calibration_receipt["decision_threshold"],
            "confirmation_thresholds": calibration_receipt[
                "confirmation_thresholds"
            ],
        },
        "eligibility_receipt_files": eligibility_file_bindings,
        "guidance_manifest": (
            {
                "path": str(guidance_manifest_path),
                "file_sha256": guidance_manifest_sha256,
                "manifest_digest": guidance_manifest_digest,
            }
            if guidance_manifest_path is not None
            else None
        ),
        "expected_recomputed_optimizer_authorized": calibration_receipt[
            "optimizer_authorized"
        ],
        "expected_fit_action_candidate_ids": calibration_receipt[
            "fit_event_qualified_action_candidate_ids"
        ],
        "confirmation_samples_consumed_by_optimizer": False,
        "scientific_action_editing_claim": False,
    }
    evidence = {
        **evidence_unsigned,
        "evidence_digest": object_sha256(evidence_unsigned),
    }
    evidence_path = output / "cagd-validator-evidence-v3.json"
    evidence_sha = _write_create_only(evidence_path, evidence)

    final_unsigned = {
        "schema_version": FINALIZATION_RECEIPT_SCHEMA,
        "root_spec_raw_sha256": root_spec_sha256,
        "bank_receipt_digest": bank_value["receipt_digest"],
        "external_label_manifest_digest": labels["manifest_digest"],
        "candidate_count": len(score_rows),
        "event_audit_receipt_count": len(audit_receipts),
        "calibration_receipt_path": str(calibration_path),
        "calibration_receipt_file_sha256": calibration_sha,
        "calibration_receipt_digest": calibration_receipt["receipt_digest"],
        "calibration_score_field": "raw_global_action_energy_score",
        "phase_conjunctive_used_for_calibration": False,
        "fit_action_eligibility_count": len(eligibility_receipts),
        "confirmation_eligibility_count": 0,
        "guidance_manifest_path": (
            str(guidance_manifest_path) if guidance_manifest_path else None
        ),
        "guidance_manifest_file_sha256": guidance_manifest_sha256,
        "cagd_validator_evidence_path": str(evidence_path),
        "cagd_validator_evidence_file_sha256": evidence_sha,
        "cagd_validator_evidence_digest": evidence["evidence_digest"],
        "legacy_eligibility_without_v3_evidence_is_not_authoritative": True,
        "confirmation_samples_consumed_by_optimizer": False,
        "t2v_media_as_rv2v_target_donor_input_or_noise": False,
        "optimizer_authorized": calibration_receipt["optimizer_authorized"],
        "scientific_action_editing_claim": False,
        "result": (
            "CAGD_ELIGIBLE_ENGINEERING_INPUT_ONLY"
            if calibration_receipt["optimizer_authorized"]
            else "SCIENTIFIC_NO_GO_NULL_UPDATE"
        ),
    }
    final_receipt = {
        **final_unsigned,
        "receipt_digest": object_sha256(final_unsigned),
    }
    _write_create_only(output / "finalization-receipt-v3.json", final_receipt)
    return final_receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-spec", required=True)
    parser.add_argument("--expected-root-spec-sha256", required=True)
    parser.add_argument("--bank-output-dir", required=True)
    parser.add_argument("--bank-receipt", required=True)
    parser.add_argument("--expected-bank-receipt-sha256", required=True)
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--external-label-manifest", required=True)
    parser.add_argument("--expected-external-label-manifest-sha256", required=True)
    parser.add_argument("--checkpoint-tree-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = finalize(
        root_spec=args.root_spec,
        root_spec_sha256=args.expected_root_spec_sha256,
        bank_output_dir=args.bank_output_dir,
        bank_receipt=args.bank_receipt,
        bank_receipt_sha256=args.expected_bank_receipt_sha256,
        score_root=args.score_root,
        external_label_manifest=args.external_label_manifest,
        external_label_manifest_sha256=args.expected_external_label_manifest_sha256,
        checkpoint_tree_sha256=args.checkpoint_tree_sha256,
        output_dir=args.output_dir,
    )
    print(canonical_json_bytes(receipt).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CORE4_FIT_ACTION_IIDS",
    "FINALIZATION_RECEIPT_SCHEMA",
    "LABEL_MANIFEST_SCHEMA",
    "PairV5CAGDFinalizationError",
    "finalize",
    "load_external_label_manifest",
    "make_external_label_manifest",
]
