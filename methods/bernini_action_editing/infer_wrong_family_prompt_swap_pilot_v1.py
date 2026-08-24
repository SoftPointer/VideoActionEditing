#!/usr/bin/env python3
"""Render the prospective STARC wrong-family pilot with native Bernini T2V.

This is intentionally a thin orchestration layer.  Bernini's frozen native
``t2v`` arm remains the only sampler, and the existing PAIR-v5 native-receipt
verifier remains the authority for exact81/40, source-free conditioning,
predecode clean latents, and official-Gaussian provenance.

The wrapper adds only the prospective protocol that the generic bank runner
does not know about: generation-plan authentication, fixed dual-SP4 placement,
per-candidate prompt-query exclusion receipts, a 20-candidate master receipt,
and materialization of the Gaussian binding plus detached 24-judgment plan.
No query prompt is passed to the generator and no generated artifact receives
editor, scientific-critic, optimizer, target, donor, or noise authority.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_identity_generation_canary as native  # noqa: E402
import infer_pair_v5_t2v_calibration_bank as native_receipts  # noqa: E402
import pair_v5_t2v_calibration_bank_spec as bank_contract  # noqa: E402
import wrong_family_prompt_swap_pilot_v1 as pilot  # noqa: E402


RUNTIME_PLAN_SCHEMA = "bernini-starc-wrong-family-runtime-plan-v1"
CANDIDATE_ENVELOPE_SCHEMA = "bernini-starc-wrong-family-candidate-envelope-v1"
CANDIDATE_RECEIPT_SCHEMA = "bernini-starc-wrong-family-candidate-receipt-v1"
MASTER_RECEIPT_SCHEMA = "bernini-starc-wrong-family-generation-master-receipt-v1"
BLINDED_REVIEW_SCHEMA = "bernini-starc-wrong-family-blinded-review-packet-v1"

RUNTIME_MANIFEST_BASENAME = "wrong-family-runtime-plan-manifest.json"
CANDIDATE_RECEIPT_BASENAME = "wrong-family-candidate-receipt.json"
GAUSSIAN_BINDING_BASENAME = "wrong-family-gaussian-binding.json"
AUDIT_PLAN_BASENAME = "wrong-family-private-adjudication-plan.json"
BLINDED_REVIEW_PACKET_BASENAME = "wrong-family-blinded-review-packet.json"
BLINDED_MEDIA_DIR_BASENAME = "blinded-review-media"
MASTER_RECEIPT_BASENAME = "wrong-family-generation-master-receipt.json"

GROUP_LAYOUT = (
    ("sp4-a", [0, 1, 2, 3], pilot.PROSPECTIVE_IIDS[0]),
    ("sp4-b", [4, 5, 6, 7], pilot.PROSPECTIVE_IIDS[1]),
)

CANDIDATE_INTERPRETATION = {
    **pilot.INTERPRETATION_CONTRACT,
    "training_performed": False,
    "parameter_update_performed": False,
    "query_prompts_consumed_by_generator": False,
    "generation_caption_source": "sealed_core4_caption_copy_only",
    "geometry_source_role": "bucket_shape_only_never_conditioning",
    "native_t2v_sampler_reused_without_modification": True,
    "private_audit_plan_is_not_reviewer_facing": True,
    "reviewer_packet_uses_opaque_media_aliases": True,
}

_SHA_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

_ENVELOPE_FIELDS = {
    "schema_version",
    "pilot_id",
    "generation_plan_raw_sha256",
    "generation_plan_digest",
    "group_id",
    "visible_gpus",
    "ordinal",
    "sampling_contract",
    "interpretation_contract",
    "candidate",
    "candidate_envelope_digest",
}
_RUNTIME_FIELDS = {
    "schema_version",
    "pilot_id",
    "generation_plan_raw_sha256",
    "generation_plan_digest",
    "group_layout",
    "candidate_count",
    "candidate_records",
    "sampling_contract",
    "interpretation_contract",
    "runtime_plan_digest",
}
_RUNTIME_RECORD_FIELDS = {
    "group_id",
    "visible_gpus",
    "ordinal",
    "iid",
    "candidate_id",
    "candidate_envelope_path",
    "candidate_envelope_sha256",
    "candidate_envelope_digest",
}
_CANDIDATE_RECEIPT_FIELDS = {
    "schema_version",
    "pilot_id",
    "generation_plan_raw_sha256",
    "generation_plan_digest",
    "runtime_plan_raw_sha256",
    "runtime_plan_digest",
    "candidate_envelope_path",
    "candidate_envelope_sha256",
    "candidate_envelope_digest",
    "group_id",
    "visible_gpus",
    "runtime_topology",
    "ordinal",
    "candidate",
    "sampling_contract",
    "native_receipt_path",
    "native_receipt_sha256",
    "native_receipt_digest",
    "artifacts",
    "interpretation",
    "receipt_digest",
}
_MASTER_FIELDS = {
    "schema_version",
    "pilot_id",
    "generation_plan_raw_sha256",
    "generation_plan_digest",
    "runtime_plan_raw_sha256",
    "runtime_plan_digest",
    "candidate_count",
    "cell_count",
    "group_layout",
    "sampling_contract",
    "candidate_receipts",
    "same_cell_gaussian_proofs",
    "gaussian_binding",
    "private_adjudication_plan",
    "blinded_review_packet",
    "interpretation",
    "receipt_digest",
}
_BLINDED_PACKET_FIELDS = {
    "schema_version",
    "pilot_id",
    "private_audit_plan_digest",
    "blinding_contract",
    "review_item_count",
    "unique_media_count",
    "review_items",
    "review_packet_digest",
}
_BLINDED_ITEM_FIELDS = {
    "review_item_id",
    "opaque_media_id",
    "opaque_media_path",
    "mp4_sha256",
    "evaluated_family_id",
    "rubric",
    "rubric_sha256",
    "full_exact81_required",
    "generation_prompt_hidden",
    "semantic_branch_hidden",
    "required_outcome_hidden",
    "decision_vocabulary",
}
BLINDING_CONTRACT = {
    "reviewer_receives_generation_caption": False,
    "reviewer_receives_candidate_id": False,
    "reviewer_receives_semantic_branch": False,
    "reviewer_receives_required_outcome": False,
    "reviewer_receives_original_media_path": False,
    "reviewer_receives_evaluated_family_rubric": True,
    "opaque_media_aliases_reordered_by_private_digest": True,
    "private_adjudication_plan_must_not_be_given_to_reviewer": True,
}


class WrongFamilyRuntimeError(RuntimeError):
    """Raised before incomplete or authority-expanding output is accepted."""


def _closed(value: Any, fields: Set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise WrongFamilyRuntimeError(
            "%s keys differ: expected=%r actual=%r" % (label, sorted(fields), actual)
        )
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise WrongFamilyRuntimeError("%s must be lowercase SHA-256" % label)
    return value


def _safe(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_RE.fullmatch(value) is None:
        raise WrongFamilyRuntimeError("%s must be a path-safe identifier" % label)
    return value


def _plain_file(path_value: Any, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise WrongFamilyRuntimeError("%s must be an absolute plain file" % label)
    if not stat.S_ISREG(path.stat().st_mode):
        raise WrongFamilyRuntimeError("%s must be regular" % label)
    return path


def _plain_dir(path_value: Any, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise WrongFamilyRuntimeError("%s must be an absolute plain directory" % label)
    return path


def _fresh_output_dir(path_value: Any, label: str) -> Path:
    path = Path(path_value)
    if (
        not path.is_absolute()
        or path == Path("/")
        or path.exists()
        or path.is_symlink()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
    ):
        raise WrongFamilyRuntimeError("%s must be a fresh absolute directory" % label)
    return path


def _inside(path_value: Any, root: Path, label: str) -> Path:
    path = _plain_file(path_value, label).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise WrongFamilyRuntimeError("%s escaped its candidate directory" % label) from error
    return path


def _load_strict(path: Path, label: str) -> Mapping[str, Any]:
    value = pilot.loads_strict(path.read_bytes(), label=label)
    if not isinstance(value, Mapping):
        raise WrongFamilyRuntimeError("%s must be an object" % label)
    return value


def _write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if (
        not path.is_absolute()
        or path == Path("/")
        or path.exists()
        or path.is_symlink()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
    ):
        raise WrongFamilyRuntimeError("output must be a fresh absolute plain-file path")
    payload = pilot.canonical_json_bytes(value) + b"\n"
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    digest = pilot.sha256_bytes(payload)
    if pilot.file_sha256(path) != digest:
        raise WrongFamilyRuntimeError("published JSON failed byte replay")
    return digest


def load_generation_plan(
    path_value: Any, expected_raw_sha256: str
) -> Tuple[Dict[str, Any], str, Path]:
    path = _plain_file(path_value, "generation plan")
    expected = _sha(expected_raw_sha256, "generation plan expected SHA-256")
    actual = pilot.file_sha256(path)
    if actual != expected:
        raise WrongFamilyRuntimeError("generation plan raw SHA-256 differs")
    try:
        plan = pilot.validate_generation_plan(_load_strict(path, "generation plan"))
    except pilot.WrongFamilyPromptSwapError as error:
        raise WrongFamilyRuntimeError("generation plan failed the prospective seal") from error
    return plan, actual, path.resolve()


def _expected_assignments(plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cells = {cell["iid"]: cell for cell in plan["prospective_cells"]}
    rows: List[Dict[str, Any]] = []
    for group_id, visible_gpus, iid in GROUP_LAYOUT:
        candidates = cells[iid]["generation_candidates"]
        for ordinal, candidate in enumerate(candidates):
            rows.append(
                {
                    "group_id": group_id,
                    "visible_gpus": list(visible_gpus),
                    "ordinal": ordinal,
                    "iid": iid,
                    "candidate": candidate,
                }
            )
    if len(rows) != 20:
        raise WrongFamilyRuntimeError("runtime assignment is not exactly 20 candidates")
    return rows


def _build_envelope(
    assignment: Mapping[str, Any],
    *,
    generation_plan_raw_sha256: str,
    generation_plan_digest: str,
) -> Dict[str, Any]:
    envelope = {
        "schema_version": CANDIDATE_ENVELOPE_SCHEMA,
        "pilot_id": pilot.PILOT_ID,
        "generation_plan_raw_sha256": generation_plan_raw_sha256,
        "generation_plan_digest": generation_plan_digest,
        "group_id": assignment["group_id"],
        "visible_gpus": assignment["visible_gpus"],
        "ordinal": assignment["ordinal"],
        "sampling_contract": bank_contract.SAMPLING_CONTRACT,
        "interpretation_contract": pilot.INTERPRETATION_CONTRACT,
        "candidate": assignment["candidate"],
    }
    envelope["candidate_envelope_digest"] = pilot.sha256_bytes(
        pilot.canonical_json_bytes(envelope)
    )
    return envelope


def validate_candidate_envelope(
    value: Any,
    *,
    plan: Mapping[str, Any],
    generation_plan_raw_sha256: str,
) -> Dict[str, Any]:
    envelope = dict(_closed(value, _ENVELOPE_FIELDS, "candidate envelope"))
    declared = _sha(
        envelope.pop("candidate_envelope_digest"), "candidate envelope digest"
    )
    if pilot.sha256_bytes(pilot.canonical_json_bytes(envelope)) != declared:
        raise WrongFamilyRuntimeError("candidate envelope digest differs")
    envelope["candidate_envelope_digest"] = declared
    if (
        envelope["schema_version"] != CANDIDATE_ENVELOPE_SCHEMA
        or envelope["pilot_id"] != pilot.PILOT_ID
        or envelope["generation_plan_raw_sha256"] != generation_plan_raw_sha256
        or envelope["generation_plan_digest"] != plan["generation_plan_digest"]
        or envelope["sampling_contract"] != bank_contract.SAMPLING_CONTRACT
        or envelope["interpretation_contract"] != pilot.INTERPRETATION_CONTRACT
    ):
        raise WrongFamilyRuntimeError("candidate envelope authority differs")
    try:
        envelope["candidate"] = bank_contract.validate_candidate(envelope["candidate"])
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise WrongFamilyRuntimeError("candidate envelope candidate differs") from error
    expected = [
        assignment
        for assignment in _expected_assignments(plan)
        if assignment["candidate"]["candidate_id"]
        == envelope["candidate"]["candidate_id"]
    ]
    if len(expected) != 1:
        raise WrongFamilyRuntimeError("candidate is not uniquely planned")
    assignment = expected[0]
    if any(
        envelope[field] != assignment[field]
        for field in ("group_id", "visible_gpus", "ordinal", "candidate")
    ):
        raise WrongFamilyRuntimeError("candidate envelope placement or bytes differ")
    return envelope


def materialize_runtime_plan(
    *,
    generation_plan_path: Any,
    expected_generation_plan_sha256: str,
    output_dir: Any,
) -> Dict[str, Any]:
    plan, plan_raw_sha, _ = load_generation_plan(
        generation_plan_path, expected_generation_plan_sha256
    )
    root = _fresh_output_dir(output_dir, "runtime plan output")
    root.mkdir(mode=0o700)
    records: List[Dict[str, Any]] = []
    try:
        for group_id, visible_gpus, _iid in GROUP_LAYOUT:
            group_dir = root / group_id
            group_dir.mkdir(mode=0o700)
            for assignment in (
                row for row in _expected_assignments(plan) if row["group_id"] == group_id
            ):
                envelope = _build_envelope(
                    assignment,
                    generation_plan_raw_sha256=plan_raw_sha,
                    generation_plan_digest=plan["generation_plan_digest"],
                )
                checked = validate_candidate_envelope(
                    envelope, plan=plan, generation_plan_raw_sha256=plan_raw_sha
                )
                candidate_id = checked["candidate"]["candidate_id"]
                path = group_dir / ("%04d-%s.json" % (checked["ordinal"], candidate_id))
                raw_sha = _write_create_only(path.resolve(), checked)
                records.append(
                    {
                        "group_id": group_id,
                        "visible_gpus": list(visible_gpus),
                        "ordinal": checked["ordinal"],
                        "iid": assignment["iid"],
                        "candidate_id": candidate_id,
                        "candidate_envelope_path": str(path.resolve()),
                        "candidate_envelope_sha256": raw_sha,
                        "candidate_envelope_digest": checked[
                            "candidate_envelope_digest"
                        ],
                    }
                )
        manifest = {
            "schema_version": RUNTIME_PLAN_SCHEMA,
            "pilot_id": pilot.PILOT_ID,
            "generation_plan_raw_sha256": plan_raw_sha,
            "generation_plan_digest": plan["generation_plan_digest"],
            "group_layout": [
                {"group_id": group, "visible_gpus": gpus, "iid": iid}
                for group, gpus, iid in GROUP_LAYOUT
            ],
            "candidate_count": len(records),
            "candidate_records": records,
            "sampling_contract": bank_contract.SAMPLING_CONTRACT,
            "interpretation_contract": pilot.INTERPRETATION_CONTRACT,
        }
        manifest["runtime_plan_digest"] = pilot.sha256_bytes(
            pilot.canonical_json_bytes(manifest)
        )
        validate_runtime_plan(
            manifest, plan=plan, generation_plan_raw_sha256=plan_raw_sha
        )
        _write_create_only((root / RUNTIME_MANIFEST_BASENAME).resolve(), manifest)
        return manifest
    except Exception:
        # Keep partial immutable evidence for diagnosis.  The enclosing launcher
        # uses a fresh output root, so no partial plan can be resumed as valid.
        raise


def validate_runtime_plan(
    value: Any,
    *,
    plan: Mapping[str, Any],
    generation_plan_raw_sha256: str,
) -> Dict[str, Any]:
    manifest = dict(_closed(value, _RUNTIME_FIELDS, "runtime plan"))
    declared = _sha(manifest.pop("runtime_plan_digest"), "runtime plan digest")
    if pilot.sha256_bytes(pilot.canonical_json_bytes(manifest)) != declared:
        raise WrongFamilyRuntimeError("runtime plan digest differs")
    manifest["runtime_plan_digest"] = declared
    expected_layout = [
        {"group_id": group, "visible_gpus": gpus, "iid": iid}
        for group, gpus, iid in GROUP_LAYOUT
    ]
    if (
        manifest["schema_version"] != RUNTIME_PLAN_SCHEMA
        or manifest["pilot_id"] != pilot.PILOT_ID
        or manifest["generation_plan_raw_sha256"] != generation_plan_raw_sha256
        or manifest["generation_plan_digest"] != plan["generation_plan_digest"]
        or manifest["group_layout"] != expected_layout
        or manifest["candidate_count"] != 20
        or manifest["sampling_contract"] != bank_contract.SAMPLING_CONTRACT
        or manifest["interpretation_contract"] != pilot.INTERPRETATION_CONTRACT
    ):
        raise WrongFamilyRuntimeError("runtime plan closure differs")
    rows = manifest["candidate_records"]
    if not isinstance(rows, list) or len(rows) != 20:
        raise WrongFamilyRuntimeError("runtime plan must contain exactly 20 records")
    expected_assignments = _expected_assignments(plan)
    seen: Set[str] = set()
    for index, (raw_row, assignment) in enumerate(zip(rows, expected_assignments)):
        row = _closed(raw_row, _RUNTIME_RECORD_FIELDS, "runtime record[%d]" % index)
        candidate = assignment["candidate"]
        if (
            row["group_id"] != assignment["group_id"]
            or row["visible_gpus"] != assignment["visible_gpus"]
            or row["ordinal"] != assignment["ordinal"]
            or row["iid"] != assignment["iid"]
            or row["candidate_id"] != candidate["candidate_id"]
            or row["candidate_id"] in seen
        ):
            raise WrongFamilyRuntimeError("runtime candidate order or placement differs")
        seen.add(row["candidate_id"])
        path = _plain_file(row["candidate_envelope_path"], "candidate envelope")
        if pilot.file_sha256(path) != _sha(
            row["candidate_envelope_sha256"], "candidate envelope raw SHA-256"
        ):
            raise WrongFamilyRuntimeError("candidate envelope file SHA-256 differs")
        checked = validate_candidate_envelope(
            _load_strict(path, "candidate envelope"),
            plan=plan,
            generation_plan_raw_sha256=generation_plan_raw_sha256,
        )
        if (
            checked["candidate_envelope_digest"]
            != row["candidate_envelope_digest"]
            or checked["candidate"] != candidate
        ):
            raise WrongFamilyRuntimeError("candidate envelope replay differs")
    return manifest


def load_runtime_plan(
    path_value: Any,
    expected_raw_sha256: str,
    *,
    plan: Mapping[str, Any],
    generation_plan_raw_sha256: str,
) -> Tuple[Dict[str, Any], str, Path]:
    path = _plain_file(path_value, "runtime plan manifest")
    expected = _sha(expected_raw_sha256, "runtime plan expected SHA-256")
    actual = pilot.file_sha256(path)
    if actual != expected:
        raise WrongFamilyRuntimeError("runtime plan raw SHA-256 differs")
    checked = validate_runtime_plan(
        _load_strict(path, "runtime plan manifest"),
        plan=plan,
        generation_plan_raw_sha256=generation_plan_raw_sha256,
    )
    return checked, actual, path.resolve()


def load_planned_candidate_envelope(
    *,
    envelope_path: Any,
    plan: Mapping[str, Any],
    generation_plan_raw_sha256: str,
    runtime_plan: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str, Path]:
    path = _plain_file(envelope_path, "candidate envelope")
    raw_sha = pilot.file_sha256(path)
    matching = [
        row
        for row in runtime_plan["candidate_records"]
        if row["candidate_envelope_path"] == str(path.resolve())
    ]
    if len(matching) != 1 or matching[0]["candidate_envelope_sha256"] != raw_sha:
        raise WrongFamilyRuntimeError("candidate envelope is not in the runtime manifest")
    checked = validate_candidate_envelope(
        _load_strict(path, "candidate envelope"),
        plan=plan,
        generation_plan_raw_sha256=generation_plan_raw_sha256,
    )
    if checked["candidate_envelope_digest"] != matching[0]["candidate_envelope_digest"]:
        raise WrongFamilyRuntimeError("candidate envelope digest differs from runtime plan")
    return checked, raw_sha, path.resolve()


def _verify_artifact(value: Any, label: str, candidate_root: Path) -> Dict[str, Any]:
    try:
        artifact = native_receipts._verify_file_artifact(value, label)
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise WrongFamilyRuntimeError("%s failed native artifact verification" % label) from error
    _inside(artifact["path"], candidate_root, label)
    return artifact


def bind_candidate_receipt(
    *,
    output_dir: Any,
    envelope: Mapping[str, Any],
    envelope_path: Path,
    envelope_raw_sha256: str,
    generation_plan: Mapping[str, Any],
    generation_plan_raw_sha256: str,
    runtime_plan: Mapping[str, Any],
    runtime_plan_raw_sha256: str,
) -> Path:
    root = _plain_dir(output_dir, "candidate output")
    native_path = _inside(root / "receipt.json", root, "native receipt")
    try:
        native_receipt = native_receipts._load_json(native_path, "native receipt")
        verified = native_receipts._verify_native_receipt(
            native_receipt, envelope["candidate"]
        )
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise WrongFamilyRuntimeError("native Bernini receipt failed closed") from error
    mp4 = _verify_artifact(verified["mp4"], "T2V MP4", root)
    clean = _verify_artifact(
        verified["predecode_clean_latent"], "T2V predecode clean latent", root
    )
    gaussian = _verify_artifact(
        verified["official_initial_gaussian"], "official initial Gaussian", root
    )
    receipt = {
        "schema_version": CANDIDATE_RECEIPT_SCHEMA,
        "pilot_id": pilot.PILOT_ID,
        "generation_plan_raw_sha256": generation_plan_raw_sha256,
        "generation_plan_digest": generation_plan["generation_plan_digest"],
        "runtime_plan_raw_sha256": runtime_plan_raw_sha256,
        "runtime_plan_digest": runtime_plan["runtime_plan_digest"],
        "candidate_envelope_path": str(envelope_path),
        "candidate_envelope_sha256": envelope_raw_sha256,
        "candidate_envelope_digest": envelope["candidate_envelope_digest"],
        "group_id": envelope["group_id"],
        "visible_gpus": envelope["visible_gpus"],
        "runtime_topology": {
            "world_size": 4,
            "ulysses_size": 4,
            "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        },
        "ordinal": envelope["ordinal"],
        "candidate": envelope["candidate"],
        "sampling_contract": bank_contract.SAMPLING_CONTRACT,
        "native_receipt_path": str(native_path),
        "native_receipt_sha256": pilot.file_sha256(native_path),
        "native_receipt_digest": verified["native_receipt_digest"],
        "artifacts": {
            "mp4": mp4,
            "predecode_clean_latent": clean,
            "official_initial_gaussian": gaussian,
        },
        "interpretation": CANDIDATE_INTERPRETATION,
    }
    receipt["receipt_digest"] = pilot.sha256_bytes(
        pilot.canonical_json_bytes(receipt)
    )
    receipt_path = root / CANDIDATE_RECEIPT_BASENAME
    _write_create_only(receipt_path, receipt)
    return receipt_path


def _validate_receipt_digest(value: Mapping[str, Any], label: str) -> str:
    unsigned = dict(value)
    declared = _sha(unsigned.pop("receipt_digest", None), label + " digest")
    if pilot.sha256_bytes(pilot.canonical_json_bytes(unsigned)) != declared:
        raise WrongFamilyRuntimeError("%s digest differs" % label)
    return declared


def validate_candidate_receipt(
    value: Any,
    *,
    receipt_path: Path,
    candidate_root: Path,
    plan: Mapping[str, Any],
    generation_plan_raw_sha256: str,
    runtime_plan: Mapping[str, Any],
    runtime_plan_raw_sha256: str,
) -> Dict[str, Any]:
    receipt = dict(
        _closed(value, _CANDIDATE_RECEIPT_FIELDS, "candidate receipt")
    )
    declared = _validate_receipt_digest(receipt, "candidate receipt")
    candidate_id = receipt.get("candidate", {}).get("candidate_id")
    records = [
        row for row in runtime_plan["candidate_records"] if row["candidate_id"] == candidate_id
    ]
    if len(records) != 1:
        raise WrongFamilyRuntimeError("candidate receipt is not uniquely planned")
    record = records[0]
    envelope, envelope_raw_sha, envelope_path = load_planned_candidate_envelope(
        envelope_path=receipt["candidate_envelope_path"],
        plan=plan,
        generation_plan_raw_sha256=generation_plan_raw_sha256,
        runtime_plan=runtime_plan,
    )
    expected_topology = {
        "world_size": 4,
        "ulysses_size": 4,
        "rocr_visible_devices": ",".join(str(item) for item in record["visible_gpus"]),
    }
    if (
        receipt["schema_version"] != CANDIDATE_RECEIPT_SCHEMA
        or receipt["pilot_id"] != pilot.PILOT_ID
        or receipt["generation_plan_raw_sha256"] != generation_plan_raw_sha256
        or receipt["generation_plan_digest"] != plan["generation_plan_digest"]
        or receipt["runtime_plan_raw_sha256"] != runtime_plan_raw_sha256
        or receipt["runtime_plan_digest"] != runtime_plan["runtime_plan_digest"]
        or receipt["candidate_envelope_sha256"] != envelope_raw_sha
        or receipt["candidate_envelope_digest"]
        != envelope["candidate_envelope_digest"]
        or receipt["group_id"] != record["group_id"]
        or receipt["visible_gpus"] != record["visible_gpus"]
        or receipt["runtime_topology"] != expected_topology
        or receipt["ordinal"] != record["ordinal"]
        or receipt["candidate"] != envelope["candidate"]
        or receipt["sampling_contract"] != bank_contract.SAMPLING_CONTRACT
        or receipt["interpretation"] != CANDIDATE_INTERPRETATION
    ):
        raise WrongFamilyRuntimeError("candidate receipt authority or topology differs")
    if receipt_path.resolve() != (candidate_root / CANDIDATE_RECEIPT_BASENAME).resolve():
        raise WrongFamilyRuntimeError("candidate receipt path differs")
    native_path = _inside(receipt["native_receipt_path"], candidate_root, "native receipt")
    if pilot.file_sha256(native_path) != _sha(
        receipt["native_receipt_sha256"], "native receipt raw SHA-256"
    ):
        raise WrongFamilyRuntimeError("native receipt bytes differ")
    try:
        native_value = native_receipts._load_json(native_path, "native receipt")
        verified = native_receipts._verify_native_receipt(
            native_value, envelope["candidate"]
        )
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise WrongFamilyRuntimeError("bound native receipt failed replay") from error
    if verified["native_receipt_digest"] != receipt["native_receipt_digest"]:
        raise WrongFamilyRuntimeError("native receipt digest binding differs")
    expected_artifacts = {
        "mp4": verified["mp4"],
        "predecode_clean_latent": verified["predecode_clean_latent"],
        "official_initial_gaussian": verified["official_initial_gaussian"],
    }
    if receipt["artifacts"] != expected_artifacts:
        raise WrongFamilyRuntimeError("candidate artifacts differ from native receipt")
    for label, artifact in expected_artifacts.items():
        _verify_artifact(artifact, label, candidate_root)
    receipt["receipt_digest"] = declared
    return receipt


def _binding_row(receipt_path: Path, receipt: Mapping[str, Any]) -> Dict[str, Any]:
    gaussian = receipt["artifacts"]["official_initial_gaussian"]
    mp4 = receipt["artifacts"]["mp4"]
    return {
        "candidate_id": receipt["candidate"]["candidate_id"],
        "seed": receipt["candidate"]["seed"],
        "candidate_receipt_path": str(receipt_path.resolve()),
        "candidate_receipt_sha256": pilot.file_sha256(receipt_path),
        "mp4_path": mp4["path"],
        "mp4_sha256": mp4["sha256"],
        "official_gaussian_path": gaussian["path"],
        "official_gaussian_artifact_sha256": gaussian["sha256"],
        "raw_value_sha256": gaussian["raw_value_sha256"],
        "content_sha256": gaussian["content_sha256"],
        "tensor_key": gaussian["tensor_key"],
        "shape": gaussian["shape"],
        "dtype": gaussian["dtype"],
        "stored_dtype": gaussian["stored_dtype"],
        "generator_initial_seed": gaussian["generator_initial_seed"],
        "captured_from_native_sampler": gaussian[
            "captured_from_native_sampler"
        ],
        "external_initial_noise_injection": gaussian[
            "external_initial_noise_injection"
        ],
        "source_or_target_derived": gaussian["source_or_target_derived"],
    }


def _artifact_reference(path: Path, digest_key: str, value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": pilot.file_sha256(path),
        digest_key: value[digest_key],
    }


def _opaque_token(domain: str, private_digest: str, private_key: Any) -> str:
    return pilot.sha256_bytes(
        pilot.canonical_json_bytes(
            {
                "domain": domain,
                "private_digest": private_digest,
                "private_key": private_key,
            }
        )
    )[:24]


def build_blinded_review_packet(
    *,
    generation_plan: Mapping[str, Any],
    gaussian_binding: Mapping[str, Any],
    private_audit_plan: Mapping[str, Any],
    output_dir: Any,
) -> Dict[str, Any]:
    """Create the only packet that may be shown to a detached reviewer.

    The private audit plan intentionally contains preregistered labels and the
    original candidate IDs.  This projection replaces every original media
    path with a digest-derived opaque hard link, removes branch/label fields,
    and deterministically permutes the 24 review items.
    """

    try:
        plan = pilot.validate_generation_plan(generation_plan)
        binding = pilot.validate_gaussian_binding(gaussian_binding, plan)
        audit = pilot.validate_audit_plan(private_audit_plan)
    except pilot.WrongFamilyPromptSwapError as error:
        raise WrongFamilyRuntimeError("cannot blind an invalid private audit plan") from error
    if (
        audit["generation_plan_digest"] != plan["generation_plan_digest"]
        or audit["gaussian_binding_digest"] != binding["gaussian_binding_digest"]
    ):
        raise WrongFamilyRuntimeError("private audit plan authority differs before blinding")
    root = _plain_dir(output_dir, "review packet output")
    media_root = root / BLINDED_MEDIA_DIR_BASENAME
    if media_root.exists() or media_root.is_symlink():
        raise WrongFamilyRuntimeError("refusing blinded media directory reuse")
    media_root.mkdir(mode=0o700)
    binding_by_candidate = {
        row["candidate_id"]: row for row in binding["bindings"]
    }
    cell_by_candidate: Dict[str, Mapping[str, Any]] = {}
    for cell in plan["prospective_cells"]:
        for candidate in cell["generation_candidates"]:
            cell_by_candidate[candidate["candidate_id"]] = cell
    alias_by_candidate: Dict[str, Dict[str, str]] = {}
    try:
        for candidate_id, media in binding_by_candidate.items():
            token = _opaque_token(
                "wrong-family-review-media-v1",
                audit["audit_plan_digest"],
                candidate_id,
            )
            opaque_id = "wf-media-" + token
            target = media_root / (opaque_id + ".mp4")
            source = _plain_file(media["mp4_path"], "review source MP4")
            if pilot.file_sha256(source) != media["mp4_sha256"]:
                raise WrongFamilyRuntimeError("review source MP4 bytes differ")
            os.link(str(source), str(target), follow_symlinks=False)
            if (
                not target.is_file()
                or target.is_symlink()
                or pilot.file_sha256(target) != media["mp4_sha256"]
            ):
                raise WrongFamilyRuntimeError("opaque media hard-link replay differs")
            alias_by_candidate[candidate_id] = {
                "opaque_media_id": opaque_id,
                "opaque_media_path": str(target.resolve()),
            }
    except Exception:
        # Partial aliases remain diagnostic-only.  The enclosing experiment
        # root is create-once, so they cannot be mistaken for a valid packet.
        raise
    items: List[Dict[str, Any]] = []
    for private_row in audit["judgments"]:
        candidate_id = private_row["candidate_id"]
        cell = cell_by_candidate.get(candidate_id)
        if cell is None or candidate_id not in alias_by_candidate:
            raise WrongFamilyRuntimeError("private review row lacks planned opaque media")
        matching_rubrics = [
            rubric
            for rubric in cell["family_rubrics"].values()
            if rubric["evaluated_family_id"] == private_row["evaluated_family_id"]
        ]
        if len(matching_rubrics) != 1:
            raise WrongFamilyRuntimeError("review family rubric is not unique")
        rubric = matching_rubrics[0]
        if pilot.sha256_bytes(pilot.canonical_json_bytes(rubric)) != private_row[
            "rubric_sha256"
        ]:
            raise WrongFamilyRuntimeError("review rubric bytes differ from private plan")
        alias = alias_by_candidate[candidate_id]
        review_item_id = "wf-review-" + _opaque_token(
            "wrong-family-review-item-v1",
            audit["audit_plan_digest"],
            private_row["audit_key"],
        )
        items.append(
            {
                "review_item_id": review_item_id,
                "opaque_media_id": alias["opaque_media_id"],
                "opaque_media_path": alias["opaque_media_path"],
                "mp4_sha256": private_row["mp4_sha256"],
                "evaluated_family_id": private_row["evaluated_family_id"],
                "rubric": rubric,
                "rubric_sha256": private_row["rubric_sha256"],
                "full_exact81_required": True,
                "generation_prompt_hidden": True,
                "semantic_branch_hidden": True,
                "required_outcome_hidden": True,
                "decision_vocabulary": ["true", "false", "unknown", "ambiguous"],
            }
        )
    items.sort(
        key=lambda row: pilot.sha256_bytes(
            (audit["audit_plan_digest"] + row["review_item_id"]).encode("utf-8")
        )
    )
    os.chmod(media_root, 0o500)
    packet = {
        "schema_version": BLINDED_REVIEW_SCHEMA,
        "pilot_id": pilot.PILOT_ID,
        "private_audit_plan_digest": audit["audit_plan_digest"],
        "blinding_contract": BLINDING_CONTRACT,
        "review_item_count": len(items),
        "unique_media_count": len(alias_by_candidate),
        "review_items": items,
    }
    packet["review_packet_digest"] = pilot.sha256_bytes(
        pilot.canonical_json_bytes(packet)
    )
    return validate_blinded_review_packet(packet)


def validate_blinded_review_packet(value: Any) -> Dict[str, Any]:
    packet = dict(_closed(value, _BLINDED_PACKET_FIELDS, "blinded review packet"))
    declared = _sha(
        packet.pop("review_packet_digest"), "blinded review packet digest"
    )
    if pilot.sha256_bytes(pilot.canonical_json_bytes(packet)) != declared:
        raise WrongFamilyRuntimeError("blinded review packet digest differs")
    packet["review_packet_digest"] = declared
    if (
        packet["schema_version"] != BLINDED_REVIEW_SCHEMA
        or packet["pilot_id"] != pilot.PILOT_ID
        or packet["blinding_contract"] != BLINDING_CONTRACT
        or packet["review_item_count"] != 24
        or packet["unique_media_count"] != 20
    ):
        raise WrongFamilyRuntimeError("blinded review packet closure differs")
    _sha(packet["private_audit_plan_digest"], "private audit plan digest")
    rows = packet["review_items"]
    if not isinstance(rows, list) or len(rows) != 24:
        raise WrongFamilyRuntimeError("blinded packet must contain 24 review items")
    review_ids: Set[str] = set()
    media_ids: Set[str] = set()
    media_paths_by_id: Dict[str, str] = {}
    for index, raw_row in enumerate(rows):
        row = _closed(raw_row, _BLINDED_ITEM_FIELDS, "review item[%d]" % index)
        review_id = str(row["review_item_id"])
        media_id = str(row["opaque_media_id"])
        if (
            re.fullmatch(r"wf-review-[0-9a-f]{24}", review_id) is None
            or re.fullmatch(r"wf-media-[0-9a-f]{24}", media_id) is None
            or review_id in review_ids
        ):
            raise WrongFamilyRuntimeError("opaque review identity differs")
        review_ids.add(review_id)
        media_ids.add(media_id)
        path = _plain_file(row["opaque_media_path"], "opaque review MP4")
        if (
            path.parent.name != BLINDED_MEDIA_DIR_BASENAME
            or path.name != media_id + ".mp4"
            or pilot.file_sha256(path) != _sha(row["mp4_sha256"], "review MP4 SHA-256")
        ):
            raise WrongFamilyRuntimeError("opaque review MP4 binding differs")
        prior_path = media_paths_by_id.setdefault(media_id, str(path.resolve()))
        if prior_path != str(path.resolve()):
            raise WrongFamilyRuntimeError("opaque media ID aliases multiple paths")
        _safe(row["evaluated_family_id"], "review evaluated family")
        rubric = row["rubric"]
        if (
            not isinstance(rubric, Mapping)
            or pilot.sha256_bytes(pilot.canonical_json_bytes(rubric))
            != _sha(row["rubric_sha256"], "review rubric SHA-256")
            or rubric.get("evaluated_family_id") != row["evaluated_family_id"]
        ):
            raise WrongFamilyRuntimeError("blinded review rubric differs")
        if (
            row["full_exact81_required"] is not True
            or row["generation_prompt_hidden"] is not True
            or row["semantic_branch_hidden"] is not True
            or row["required_outcome_hidden"] is not True
            or row["decision_vocabulary"]
            != ["true", "false", "unknown", "ambiguous"]
        ):
            raise WrongFamilyRuntimeError("blinded review hard gate was relaxed")
    if len(media_ids) != 20:
        raise WrongFamilyRuntimeError("blinded review packet lost its 20 opaque media")
    return packet


def validate_master_receipt(value: Any) -> Dict[str, Any]:
    receipt = dict(_closed(value, _MASTER_FIELDS, "master receipt"))
    declared = _validate_receipt_digest(receipt, "master receipt")
    expected_layout = [
        {"group_id": group, "visible_gpus": gpus, "iid": iid}
        for group, gpus, iid in GROUP_LAYOUT
    ]
    if (
        receipt["schema_version"] != MASTER_RECEIPT_SCHEMA
        or receipt["pilot_id"] != pilot.PILOT_ID
        or receipt["candidate_count"] != 20
        or receipt["cell_count"] != 2
        or receipt["group_layout"] != expected_layout
        or receipt["sampling_contract"] != bank_contract.SAMPLING_CONTRACT
        or receipt["interpretation"] != CANDIDATE_INTERPRETATION
    ):
        raise WrongFamilyRuntimeError("master receipt closure differs")
    _sha(receipt["generation_plan_raw_sha256"], "master generation plan SHA-256")
    _sha(receipt["generation_plan_digest"], "master generation plan digest")
    _sha(receipt["runtime_plan_raw_sha256"], "master runtime plan SHA-256")
    _sha(receipt["runtime_plan_digest"], "master runtime plan digest")
    rows = receipt["candidate_receipts"]
    if (
        not isinstance(rows, list)
        or len(rows) != 20
        or len({row.get("candidate_id") for row in rows if isinstance(row, Mapping)}) != 20
    ):
        raise WrongFamilyRuntimeError("master candidate population differs")
    row_fields = {
        "candidate_id",
        "iid",
        "group_id",
        "ordinal",
        "path",
        "sha256",
        "receipt_digest",
        "mp4_sha256",
        "predecode_clean_latent_sha256",
        "official_initial_gaussian_sha256",
    }
    for index, row in enumerate(rows):
        checked = _closed(row, row_fields, "master candidate[%d]" % index)
        _safe(checked["candidate_id"], "master candidate ID")
        _safe(checked["iid"], "master candidate IID")
        for key in (
            "sha256",
            "receipt_digest",
            "mp4_sha256",
            "predecode_clean_latent_sha256",
            "official_initial_gaussian_sha256",
        ):
            _sha(checked[key], "master candidate " + key)
        path = _plain_file(checked["path"], "master candidate receipt")
        if pilot.file_sha256(path) != checked["sha256"]:
            raise WrongFamilyRuntimeError("master candidate receipt bytes differ")
    proofs = receipt["same_cell_gaussian_proofs"]
    if not isinstance(proofs, list) or [row.get("iid") for row in proofs] != list(
        pilot.PROSPECTIVE_IIDS
    ):
        raise WrongFamilyRuntimeError("master Gaussian cell proofs differ")
    for proof in proofs:
        if (
            set(proof)
            != {
                "iid",
                "seed",
                "candidate_count",
                "raw_value_sha256",
                "content_sha256",
                "all_ten_tensor_values_equal",
            }
            or proof["candidate_count"] != 10
            or proof["all_ten_tensor_values_equal"] is not True
        ):
            raise WrongFamilyRuntimeError("master same-cell Gaussian proof differs")
        _sha(proof["raw_value_sha256"], "master Gaussian raw digest")
        _sha(proof["content_sha256"], "master Gaussian content digest")
    reference_fields = {"path", "sha256", "gaussian_binding_digest"}
    if set(receipt["gaussian_binding"]) != reference_fields:
        raise WrongFamilyRuntimeError("master Gaussian manifest reference differs")
    reference_fields = {"path", "sha256", "audit_plan_digest"}
    if set(receipt["private_adjudication_plan"]) != reference_fields:
        raise WrongFamilyRuntimeError("master private audit plan reference differs")
    reference_fields = {"path", "sha256", "review_packet_digest"}
    if set(receipt["blinded_review_packet"]) != reference_fields:
        raise WrongFamilyRuntimeError("master blinded review packet reference differs")
    for reference in (
        receipt["gaussian_binding"],
        receipt["private_adjudication_plan"],
        receipt["blinded_review_packet"],
    ):
        path = _plain_file(reference["path"], "master referenced manifest")
        if pilot.file_sha256(path) != _sha(reference["sha256"], "manifest SHA-256"):
            raise WrongFamilyRuntimeError("master referenced manifest bytes differ")
    gaussian_value = dict(
        _load_strict(
            Path(receipt["gaussian_binding"]["path"]), "master Gaussian binding"
        )
    )
    gaussian_declared = _sha(
        gaussian_value.pop("gaussian_binding_digest", None),
        "master Gaussian binding digest",
    )
    if (
        gaussian_declared
        != receipt["gaussian_binding"]["gaussian_binding_digest"]
        or pilot.sha256_bytes(pilot.canonical_json_bytes(gaussian_value))
        != gaussian_declared
    ):
        raise WrongFamilyRuntimeError("master Gaussian binding digest replay differs")
    try:
        private_plan = pilot.validate_audit_plan(
            _load_strict(
                Path(receipt["private_adjudication_plan"]["path"]),
                "master private adjudication plan",
            )
        )
    except pilot.WrongFamilyPromptSwapError as error:
        raise WrongFamilyRuntimeError("master private audit plan failed replay") from error
    if (
        private_plan["audit_plan_digest"]
        != receipt["private_adjudication_plan"]["audit_plan_digest"]
    ):
        raise WrongFamilyRuntimeError("master private audit plan digest differs")
    blinded = validate_blinded_review_packet(
        _load_strict(
            Path(receipt["blinded_review_packet"]["path"]),
            "master blinded review packet",
        )
    )
    if (
        blinded["review_packet_digest"]
        != receipt["blinded_review_packet"]["review_packet_digest"]
        or blinded["private_audit_plan_digest"]
        != private_plan["audit_plan_digest"]
    ):
        raise WrongFamilyRuntimeError("master blinded review authority differs")
    receipt["receipt_digest"] = declared
    return receipt


def audit_rendered_pilot(
    *,
    generation_plan_path: Any,
    expected_generation_plan_sha256: str,
    runtime_plan_path: Any,
    expected_runtime_plan_sha256: str,
    output_dir: Any,
) -> Dict[str, Any]:
    plan, plan_raw_sha, _ = load_generation_plan(
        generation_plan_path, expected_generation_plan_sha256
    )
    runtime_plan, runtime_raw_sha, _ = load_runtime_plan(
        runtime_plan_path,
        expected_runtime_plan_sha256,
        plan=plan,
        generation_plan_raw_sha256=plan_raw_sha,
    )
    root = _plain_dir(output_dir, "rendered pilot output")
    checked_rows: List[Tuple[Mapping[str, Any], Path, Dict[str, Any]]] = []
    for record in runtime_plan["candidate_records"]:
        candidate_root = _plain_dir(
            root / record["candidate_id"], "candidate output directory"
        )
        receipt_path = _plain_file(
            candidate_root / CANDIDATE_RECEIPT_BASENAME, "candidate receipt"
        )
        receipt = validate_candidate_receipt(
            _load_strict(receipt_path, "candidate receipt"),
            receipt_path=receipt_path,
            candidate_root=candidate_root,
            plan=plan,
            generation_plan_raw_sha256=plan_raw_sha,
            runtime_plan=runtime_plan,
            runtime_plan_raw_sha256=runtime_raw_sha,
        )
        checked_rows.append((record, receipt_path, receipt))
    bindings = [_binding_row(path, receipt) for _record, path, receipt in checked_rows]
    try:
        gaussian_binding = pilot.build_gaussian_binding_manifest(plan, bindings)
        audit_plan = pilot.build_audit_plan(plan, gaussian_binding)
    except pilot.WrongFamilyPromptSwapError as error:
        raise WrongFamilyRuntimeError(
            "rendered tuple failed official-Gaussian or detached-audit closure"
        ) from error
    gaussian_path = root / GAUSSIAN_BINDING_BASENAME
    audit_path = root / AUDIT_PLAN_BASENAME
    _write_create_only(gaussian_path, gaussian_binding)
    _write_create_only(audit_path, audit_plan)
    blinded_packet = build_blinded_review_packet(
        generation_plan=plan,
        gaussian_binding=gaussian_binding,
        private_audit_plan=audit_plan,
        output_dir=root,
    )
    blinded_packet_path = root / BLINDED_REVIEW_PACKET_BASENAME
    _write_create_only(blinded_packet_path, blinded_packet)
    proofs = []
    for iid in pilot.PROSPECTIVE_IIDS:
        rows = [
            row
            for row in gaussian_binding["bindings"]
            if ("-%s-" % iid) in row["candidate_id"]
        ]
        raw_values = {row["raw_value_sha256"] for row in rows}
        content_values = {row["content_sha256"] for row in rows}
        seeds = {row["seed"] for row in rows}
        if len(rows) != 10 or len(raw_values) != 1 or len(content_values) != 1 or len(seeds) != 1:
            raise WrongFamilyRuntimeError("same-cell Gaussian proof replay differs")
        proofs.append(
            {
                "iid": iid,
                "seed": next(iter(seeds)),
                "candidate_count": 10,
                "raw_value_sha256": next(iter(raw_values)),
                "content_sha256": next(iter(content_values)),
                "all_ten_tensor_values_equal": True,
            }
        )
    candidate_receipts = []
    for record, path, receipt in checked_rows:
        artifacts = receipt["artifacts"]
        candidate_receipts.append(
            {
                "candidate_id": record["candidate_id"],
                "iid": record["iid"],
                "group_id": record["group_id"],
                "ordinal": record["ordinal"],
                "path": str(path.resolve()),
                "sha256": pilot.file_sha256(path),
                "receipt_digest": receipt["receipt_digest"],
                "mp4_sha256": artifacts["mp4"]["sha256"],
                "predecode_clean_latent_sha256": artifacts[
                    "predecode_clean_latent"
                ]["sha256"],
                "official_initial_gaussian_sha256": artifacts[
                    "official_initial_gaussian"
                ]["sha256"],
            }
        )
    master = {
        "schema_version": MASTER_RECEIPT_SCHEMA,
        "pilot_id": pilot.PILOT_ID,
        "generation_plan_raw_sha256": plan_raw_sha,
        "generation_plan_digest": plan["generation_plan_digest"],
        "runtime_plan_raw_sha256": runtime_raw_sha,
        "runtime_plan_digest": runtime_plan["runtime_plan_digest"],
        "candidate_count": 20,
        "cell_count": 2,
        "group_layout": [
            {"group_id": group, "visible_gpus": gpus, "iid": iid}
            for group, gpus, iid in GROUP_LAYOUT
        ],
        "sampling_contract": bank_contract.SAMPLING_CONTRACT,
        "candidate_receipts": candidate_receipts,
        "same_cell_gaussian_proofs": proofs,
        "gaussian_binding": _artifact_reference(
            gaussian_path, "gaussian_binding_digest", gaussian_binding
        ),
        "private_adjudication_plan": _artifact_reference(
            audit_path, "audit_plan_digest", audit_plan
        ),
        "blinded_review_packet": _artifact_reference(
            blinded_packet_path, "review_packet_digest", blinded_packet
        ),
        "interpretation": CANDIDATE_INTERPRETATION,
    }
    master["receipt_digest"] = pilot.sha256_bytes(pilot.canonical_json_bytes(master))
    validate_master_receipt(master)
    _write_create_only(root / MASTER_RECEIPT_BASENAME, master)
    return master


def render_candidate(args: argparse.Namespace) -> int:
    plan, plan_raw_sha, _ = load_generation_plan(
        args.generation_plan, args.expected_generation_plan_sha256
    )
    runtime_plan, runtime_raw_sha, _ = load_runtime_plan(
        args.runtime_plan,
        args.expected_runtime_plan_sha256,
        plan=plan,
        generation_plan_raw_sha256=plan_raw_sha,
    )
    envelope, envelope_raw_sha, envelope_path = load_planned_candidate_envelope(
        envelope_path=args.candidate_envelope,
        plan=plan,
        generation_plan_raw_sha256=plan_raw_sha,
        runtime_plan=runtime_plan,
    )
    expected_visible = ",".join(str(value) for value in envelope["visible_gpus"])
    if os.environ.get("ROCR_VISIBLE_DEVICES") != expected_visible:
        raise WrongFamilyRuntimeError("runtime GPU visibility differs from sealed SP4")
    if os.environ.get("WORLD_SIZE") not in (None, "4"):
        raise WrongFamilyRuntimeError("runtime world size is not four")
    candidate = envelope["candidate"]
    guidance = bank_contract.SAMPLING_CONTRACT["guidance"]
    native.OMEGA_TEXT = guidance["omega_txt"]
    native.OMEGA_VIDEO = guidance["omega_vid"]
    native.OMEGA_IMAGE = guidance["omega_img"]
    native_argv = [
        "--bernini-root",
        args.bernini_root,
        "--veomni-root",
        args.veomni_root,
        "--checkpoint",
        args.checkpoint,
        "--checkpoint-content-manifest",
        args.checkpoint_content_manifest,
        "--source-video",
        candidate["geometry_source_video"],
        "--expected-source-sha256",
        candidate["geometry_source_video_sha256"],
        "--action-prompt",
        candidate["full_t2v_caption"],
        "--expected-action-prompt-sha256",
        candidate["full_t2v_caption_utf8_sha256"],
        "--output-dir",
        args.output_dir,
        "--arms",
        "t2v",
        "--num-inference-steps",
        "40",
        "--seed",
        str(candidate["seed"]),
        "--method-source-revision",
        args.method_source_revision,
        "--method-source-archive-sha256",
        args.method_source_archive_sha256,
    ]
    status = native.main(native_argv)
    if status == 0 and int(os.environ.get("RANK", "0")) == 0:
        bind_candidate_receipt(
            output_dir=args.output_dir,
            envelope=envelope,
            envelope_path=envelope_path,
            envelope_raw_sha256=envelope_raw_sha,
            generation_plan=plan,
            generation_plan_raw_sha256=plan_raw_sha,
            runtime_plan=runtime_plan,
            runtime_plan_raw_sha256=runtime_raw_sha,
        )
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize-runtime")
    materialize.add_argument("--generation-plan", required=True)
    materialize.add_argument("--expected-generation-plan-sha256", required=True)
    materialize.add_argument("--output-dir", required=True)

    render = subparsers.add_parser("render-candidate")
    render.add_argument("--generation-plan", required=True)
    render.add_argument("--expected-generation-plan-sha256", required=True)
    render.add_argument("--runtime-plan", required=True)
    render.add_argument("--expected-runtime-plan-sha256", required=True)
    render.add_argument("--candidate-envelope", required=True)
    render.add_argument("--output-dir", required=True)
    render.add_argument("--bernini-root", required=True)
    render.add_argument("--veomni-root", required=True)
    render.add_argument("--checkpoint", required=True)
    render.add_argument("--checkpoint-content-manifest", required=True)
    render.add_argument("--method-source-revision", required=True)
    render.add_argument("--method-source-archive-sha256", required=True)

    audit = subparsers.add_parser("audit-bank")
    audit.add_argument("--generation-plan", required=True)
    audit.add_argument("--expected-generation-plan-sha256", required=True)
    audit.add_argument("--runtime-plan", required=True)
    audit.add_argument("--expected-runtime-plan-sha256", required=True)
    audit.add_argument("--output-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize-runtime":
        manifest = materialize_runtime_plan(
            generation_plan_path=args.generation_plan,
            expected_generation_plan_sha256=args.expected_generation_plan_sha256,
            output_dir=args.output_dir,
        )
        path = Path(args.output_dir) / RUNTIME_MANIFEST_BASENAME
        print(
            pilot.canonical_json_bytes(
                {
                    "path": str(path.resolve()),
                    "sha256": pilot.file_sha256(path),
                    "runtime_plan_digest": manifest["runtime_plan_digest"],
                }
            ).decode("utf-8"),
            flush=True,
        )
        return 0
    if args.command == "render-candidate":
        return render_candidate(args)
    master = audit_rendered_pilot(
        generation_plan_path=args.generation_plan,
        expected_generation_plan_sha256=args.expected_generation_plan_sha256,
        runtime_plan_path=args.runtime_plan,
        expected_runtime_plan_sha256=args.expected_runtime_plan_sha256,
        output_dir=args.output_dir,
    )
    print(pilot.canonical_json_bytes(master).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_PLAN_BASENAME",
    "BLINDED_MEDIA_DIR_BASENAME",
    "BLINDED_REVIEW_PACKET_BASENAME",
    "BLINDED_REVIEW_SCHEMA",
    "BLINDING_CONTRACT",
    "CANDIDATE_ENVELOPE_SCHEMA",
    "CANDIDATE_INTERPRETATION",
    "CANDIDATE_RECEIPT_BASENAME",
    "CANDIDATE_RECEIPT_SCHEMA",
    "GAUSSIAN_BINDING_BASENAME",
    "GROUP_LAYOUT",
    "MASTER_RECEIPT_BASENAME",
    "MASTER_RECEIPT_SCHEMA",
    "RUNTIME_MANIFEST_BASENAME",
    "RUNTIME_PLAN_SCHEMA",
    "WrongFamilyRuntimeError",
    "audit_rendered_pilot",
    "bind_candidate_receipt",
    "build_blinded_review_packet",
    "load_generation_plan",
    "load_planned_candidate_envelope",
    "load_runtime_plan",
    "materialize_runtime_plan",
    "render_candidate",
    "validate_candidate_envelope",
    "validate_candidate_receipt",
    "validate_blinded_review_packet",
    "validate_master_receipt",
    "validate_runtime_plan",
]
