"""Fail-closed V3 contract for Bernini self-imagined action editing.

V3 deliberately has no free-form, episode-owned Pass-A proposal bank and no
episode-owned renderer budget.  Its three trust roots are supplied out of band
as canonical JSON bytes with caller-pinned SHA-256 values:

* the *actual* ``source_caption_t2v_pass_a.BANK_RECEIPT_SCHEMA`` receipt;
* an independent, blind Pass-A qualification seal; and
* a Pass-B preregistration seal containing the immutable policy, thresholds,
  candidate budget, causal pairs, source-copy bindings, and 2x2 controls.

The episode contains only Pass-B observations.  It cannot add a candidate,
widen a threshold, rename a Pass-A branch, or choose a new causal pair after
rendering.  Artifact hashes are supplied independently from the episode and
must equal the hashes in the sealed receipts.  This module emits upstream
causal qualifications only; production DCLR consumption remains impossible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any

try:
    from . import source_caption_t2v_pass_a as pass_a_native
    from .dclr_counterfactual_bank import canonical_object_sha256
except ImportError:  # Direct import with METHOD_ROOT on sys.path.
    import source_caption_t2v_pass_a as pass_a_native  # type: ignore
    from dclr_counterfactual_bank import canonical_object_sha256  # type: ignore


EPISODE_SCHEMA = "bernini-self-imagined-pareto-episode-v3"
QUALIFICATION_SEAL_SCHEMA = "bernini-source-caption-pass-a-qualification-seal-v3"
PREREGISTRATION_SEAL_SCHEMA = "bernini-self-imagined-pass-b-preregistration-v3"
CANDIDATE_SCHEMA = "bernini-self-imagined-renderer-candidate-v3"
INVOCATION_SCHEMA = "bernini-self-imagined-native-invocation-v3"
RECEIPT_SCHEMA = "bernini-self-imagined-upstream-qualification-v3"

FRAME_COUNT = pass_a_native.FRAME_COUNT
PROPOSAL_BRANCHES = tuple(pass_a_native.BRANCH_ORDER)
EXPECTED_SEED_ROWS = tuple(dict(row) for row in pass_a_native.SEED_ROWS)
EVENT_AXES = ("actor", "direction", "contact", "order", "terminal")
RENDER_GATES = ("A", "I", "C", "Q")
PAIR_TYPES = ("action_donor_nearmiss", "identity_reference_nearmiss")
OFFICIAL_GAUSSIAN_SOURCE = "official_bernini_module_global_randn_tensor"

_EXPECTED_BRANCH_EVENTS: Mapping[str, Mapping[str, bool]] = MappingProxyType(
    {
        "full_action": MappingProxyType({axis: True for axis in EVENT_AXES}),
        "noop": MappingProxyType({axis: False for axis in EVENT_AXES}),
        "incomplete": MappingProxyType(
            {
                "actor": True,
                "direction": True,
                "contact": True,
                "order": True,
                "terminal": False,
            }
        ),
        "reverse": MappingProxyType(
            {
                "actor": True,
                "direction": False,
                "contact": True,
                "order": False,
                "terminal": False,
            }
        ),
    }
)

COUNTERFACTUAL_ARMS = MappingProxyType(
    {
        "neither": ("factorial_neither", "off", "off"),
        "identity_refs_only": ("factorial_refs_only", "off", "correct"),
        "donor_only": ("factorial_donor_only", "proposal", "off"),
        "donor_and_identity_refs": (
            "factorial_donor_refs",
            "proposal",
            "correct",
        ),
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")

_PASS_A_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "method",
        "stage",
        "manifest_path",
        "manifest_file_sha256",
        "manifest_digest",
        "method_source_revision",
        "method_source_archive_sha256",
        "entry_count",
        "seed_count",
        "branch_count_per_seed",
        "entries",
        "initial_gaussian_contract",
        "condition_closure",
        "qualification",
        "interpretation",
        "receipt_digest",
    }
)
_PASS_A_ENTRY_FIELDS = frozenset(
    {
        "entry_id",
        "seed_id",
        "seed",
        "execution_group",
        "semantic_branch",
        "native_receipt_path",
        "native_receipt_sha256",
        "native_receipt_digest",
        "video_path",
        "video_sha256",
        "clean_latent_path",
        "clean_latent_sha256",
        "initial_gaussian_path",
        "initial_gaussian_file_sha256",
        "initial_gaussian_value_sha256",
        "initial_gaussian_independently_parsed",
        "pure_t2v_condition_audit_pass",
        "semantic_event_verified",
    }
)
_PASS_A_ARTIFACT_FIELDS = frozenset(
    {
        "native_receipt_sha256",
        "native_receipt_digest",
        "video_sha256",
        "clean_latent_sha256",
        "initial_gaussian_file_sha256",
        "initial_gaussian_value_sha256",
    }
)
_PASS_A_GAUSSIAN_CONTRACT_FIELDS = frozenset(
    {
        "per_seed_value_sha256",
        "same_value_across_all_four_branches_within_seed",
        "different_values_across_the_two_seeds",
        "tensor_values_recomputed_from_safetensors",
        "posthoc_seed_selection",
    }
)
_PASS_A_CLOSURE_FIELDS = frozenset(
    {
        "renderer_arm",
        "guidance_mode",
        "source_video_role",
        "source_pixels_forwarded_to_sampler",
        "source_video_latent_consumed",
        "source_reference_latent_consumed",
        "target_video_consumed",
        "mask_flow_pose_track_trajectory_consumed",
        "all_native_entry_condition_audits_pass",
    }
)
_PASS_A_NATIVE_QUALIFICATION_FIELDS = frozenset(
    {
        "manifest_semantic_labels_are_not_event_acceptance",
        "semantic_events_verified",
        "exact40_manual_qualification_required",
        "qualification_unit",
        "reject_pass_a_if_either_seed_or_any_branch_fails",
        "single_seed_or_branch_selection_forbidden",
        "reward_or_training_use_authorized",
        "pass_a_status",
    }
)
_PASS_A_INTERPRETATION_FIELDS = frozenset(
    {
        "render_complete",
        "pure_t2v_action_proposal_bank",
        "editing_result",
        "quality_claim",
        "model_training_performed",
        "scientific_claim_authorized",
    }
)
_QUALIFICATION_SEAL_FIELDS = frozenset(
    {
        "schema_version",
        "seal_id",
        "pass_a_receipt_file_sha256",
        "pass_a_receipt_digest",
        "evaluator_sha256",
        "calibration_sha256",
        "absolute_uncertainty_threshold",
        "blinded_before_pass_b",
        "pass_b_artifacts_available",
        "qualification_unit",
        "entries",
        "seal_digest",
    }
)
_QUALIFICATION_ENTRY_FIELDS = frozenset(
    {
        "entry_id",
        "seed_id",
        "semantic_branch",
        "video_sha256",
        "clean_latent_sha256",
        "initial_gaussian_value_sha256",
        "calibrated",
        "absolute_uncertainty",
        "event_axis_pass",
        "branch_contract_pass",
    }
)


class SelfImaginedParetoContractError(ValueError):
    """An external seal, artifact binding, episode, or causal rule is invalid."""


class ProductionDCLRBridgeUnavailable(SelfImaginedParetoContractError):
    """V3 has no validated production DCLR adapter."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SelfImaginedParetoContractError("value is not canonical JSON") from error


def _duplicate_safe_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SelfImaginedParetoContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise SelfImaginedParetoContractError(f"non-finite JSON number: {value}")


def _sha(label: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SelfImaginedParetoContractError(f"{label} must be lowercase SHA-256")
    return value


def _sha1(label: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise SelfImaginedParetoContractError(f"{label} must be lowercase SHA-1")
    return value


def _slug(label: str, value: Any) -> str:
    if not isinstance(value, str) or _SLUG_RE.fullmatch(value) is None:
        raise SelfImaginedParetoContractError(f"{label} must be a canonical slug")
    return value


def _boolean(label: str, value: Any) -> bool:
    if type(value) is not bool:
        raise SelfImaginedParetoContractError(f"{label} must be boolean")
    return value


def _integer(label: str, value: Any, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SelfImaginedParetoContractError(f"{label} must be integer >= {minimum}")
    return value


def _number(
    label: str,
    value: Any,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelfImaginedParetoContractError(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SelfImaginedParetoContractError(f"{label} must be finite numeric")
    if minimum is not None:
        invalid = result <= minimum if strict_minimum else result < minimum
        if invalid:
            raise SelfImaginedParetoContractError(f"{label} is outside its domain")
    return result


def _text(label: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SelfImaginedParetoContractError(f"{label} must be nonempty trimmed text")
    return value


def _exact(label: str, value: Any, fields: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelfImaginedParetoContractError(f"{label} must be an object")
    observed = frozenset(value)
    if observed != fields:
        raise SelfImaginedParetoContractError(
            f"{label} fields are closed: missing={sorted(fields-observed)}, "
            f"extra={sorted(observed-fields)}"
        )
    return value


def _embedded_digest(label: str, value: Mapping[str, Any], field: str) -> str:
    declared = _sha(f"{label}.{field}", value[field])
    unsigned = dict(value)
    unsigned.pop(field)
    if canonical_object_sha256(unsigned) != declared:
        raise SelfImaginedParetoContractError(f"{label} embedded digest differs")
    return declared


def _load_external_json(
    label: str,
    payload: bytes,
    expected_sha256: str,
    *,
    digest_field: str,
) -> tuple[Mapping[str, Any], str]:
    if not isinstance(payload, bytes) or not payload:
        raise SelfImaginedParetoContractError(f"{label} must be nonempty bytes")
    expected = _sha(f"expected {label} SHA-256", expected_sha256)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise SelfImaginedParetoContractError(f"{label} bytes differ from external seal")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SelfImaginedParetoContractError(f"{label} is not canonical JSON") from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != payload:
        raise SelfImaginedParetoContractError(f"{label} bytes are not canonical")
    _embedded_digest(label, value, digest_field)
    return value, observed


def _sha_list(label: str, value: Any, *, length: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise SelfImaginedParetoContractError(f"{label} must contain {length} hashes")
    result = tuple(_sha(f"{label}[{index}]", item) for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise SelfImaginedParetoContractError(f"{label} repeats an artifact")
    return result


def _text_digest(label: str, text: Any, digest: Any) -> str:
    result = _text(label, text)
    if hashlib.sha256(result.encode("utf-8")).hexdigest() != _sha(
        f"{label}_sha256", digest
    ):
        raise SelfImaginedParetoContractError(f"{label} digest differs")
    return result


def _expected_pass_a_rows() -> list[tuple[str, int, str, str, str]]:
    result: list[tuple[str, int, str, str, str]] = []
    for seed_row in EXPECTED_SEED_ROWS:
        for branch in PROPOSAL_BRANCHES:
            result.append(
                (
                    str(seed_row["seed_id"]),
                    int(seed_row["seed"]),
                    str(seed_row["execution_group"]),
                    branch,
                    f"{seed_row['seed_id']}-{branch.replace('_', '-')}",
                )
            )
    return result


def _validate_pass_a_receipt(
    value: Mapping[str, Any],
    artifact_hashes: Any,
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    receipt = _exact("actual Pass-A bank receipt", value, _PASS_A_RECEIPT_FIELDS)
    if receipt["schema_version"] != pass_a_native.BANK_RECEIPT_SCHEMA:
        raise SelfImaginedParetoContractError("actual Pass-A BANK_RECEIPT_SCHEMA differs")
    if receipt["method"] != pass_a_native.METHOD:
        raise SelfImaginedParetoContractError("actual Pass-A method differs")
    _sha("Pass-A manifest file SHA-256", receipt["manifest_file_sha256"])
    _sha("Pass-A manifest digest", receipt["manifest_digest"])
    _sha1("Pass-A method source revision", receipt["method_source_revision"])
    _sha("Pass-A method source archive SHA-256", receipt["method_source_archive_sha256"])
    if (
        receipt["entry_count"] != 8
        or receipt["seed_count"] != 2
        or receipt["branch_count_per_seed"] != 4
    ):
        raise SelfImaginedParetoContractError("Pass-A must be exact two-seed by four-branch")

    rows = receipt["entries"]
    expected = _expected_pass_a_rows()
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise SelfImaginedParetoContractError("actual Pass-A receipt must contain eight entries")
    if not isinstance(artifact_hashes, Mapping) or frozenset(artifact_hashes) != {
        row[-1] for row in expected
    }:
        raise SelfImaginedParetoContractError(
            "independent Pass-A artifact registry must exactly follow receipt order"
        )

    indexed: dict[str, Mapping[str, Any]] = {}
    all_artifacts: set[str] = set()
    gaussian_by_seed: dict[str, str] = {}
    for index, (raw, expected_row) in enumerate(zip(rows, expected)):
        seed_id, seed, group, branch, entry_id = expected_row
        row = _exact(f"actual Pass-A entry {index}", raw, _PASS_A_ENTRY_FIELDS)
        if (
            row["entry_id"] != entry_id
            or row["seed_id"] != seed_id
            or row["seed"] != seed
            or row["execution_group"] != group
            or row["semantic_branch"] != branch
        ):
            raise SelfImaginedParetoContractError(
                "actual Pass-A entries differ from exact native seed/branch order"
            )
        registry = _exact(
            f"Pass-A artifact registry {entry_id}",
            artifact_hashes[entry_id],
            _PASS_A_ARTIFACT_FIELDS,
        )
        for path_field in (
            "native_receipt_path",
            "video_path",
            "clean_latent_path",
            "initial_gaussian_path",
        ):
            path = _text(f"Pass-A {entry_id} {path_field}", row[path_field])
            if not path.startswith("/"):
                raise SelfImaginedParetoContractError(
                    f"Pass-A {entry_id} {path_field} must be absolute"
                )
        for field in _PASS_A_ARTIFACT_FIELDS:
            sealed_hash = _sha(f"Pass-A {entry_id} {field}", row[field])
            if registry[field] != sealed_hash:
                raise SelfImaginedParetoContractError(
                    f"Pass-A {entry_id} artifact bytes/hash registry differs"
                )
        for field in ("video_sha256", "clean_latent_sha256"):
            artifact = str(row[field])
            if artifact in all_artifacts:
                raise SelfImaginedParetoContractError(
                    "Pass-A output video/clean latent hashes must be globally distinct"
                )
            all_artifacts.add(artifact)
        if (
            row["initial_gaussian_independently_parsed"] is not True
            or row["pure_t2v_condition_audit_pass"] is not True
            or row["semantic_event_verified"] is not False
        ):
            raise SelfImaginedParetoContractError(
                "actual Pass-A native condition/Gaussian receipt differs"
            )
        gaussian = str(row["initial_gaussian_value_sha256"])
        previous = gaussian_by_seed.setdefault(seed_id, gaussian)
        if previous != gaussian:
            raise SelfImaginedParetoContractError(
                "Pass-A branches within one seed do not share byte-identical Gaussian"
            )
        indexed[entry_id] = row
    if len(set(gaussian_by_seed.values())) != 2:
        raise SelfImaginedParetoContractError(
            "the two exact Pass-A seeds must have distinct Gaussian values"
        )

    gaussian_contract = _exact(
        "actual Pass-A Gaussian contract",
        receipt["initial_gaussian_contract"],
        _PASS_A_GAUSSIAN_CONTRACT_FIELDS,
    )
    if (
        gaussian_contract.get("per_seed_value_sha256") != gaussian_by_seed
        or gaussian_contract.get("same_value_across_all_four_branches_within_seed")
        is not True
        or gaussian_contract.get("different_values_across_the_two_seeds") is not True
        or gaussian_contract.get("tensor_values_recomputed_from_safetensors") is not True
        or gaussian_contract.get("posthoc_seed_selection") is not False
    ):
        raise SelfImaginedParetoContractError("actual Pass-A Gaussian contract differs")
    closure = _exact(
        "actual Pass-A condition closure",
        receipt["condition_closure"],
        _PASS_A_CLOSURE_FIELDS,
    )
    if (
        closure.get("renderer_arm") != "t2v"
        or closure.get("guidance_mode") != "t2v_apg"
        or closure.get("source_video_role")
        != "hash_verification_and_fixed_496x480_bucket_only"
        or closure.get("source_pixels_forwarded_to_sampler") is not False
        or closure.get("source_video_latent_consumed") is not False
        or closure.get("source_reference_latent_consumed") is not False
        or closure.get("target_video_consumed") is not False
        or closure.get("mask_flow_pose_track_trajectory_consumed") is not False
        or closure.get("all_native_entry_condition_audits_pass") is not True
    ):
        raise SelfImaginedParetoContractError("actual Pass-A condition closure differs")
    qualification = _exact(
        "actual Pass-A native qualification contract",
        receipt["qualification"],
        _PASS_A_NATIVE_QUALIFICATION_FIELDS,
    )
    exact40 = receipt["stage"] == "exact40-qualification-candidate"
    expected_status = (
        "pending_independent_manual_qualification"
        if exact40
        else "engineering_only_no_semantic_claim"
    )
    if (
        qualification.get("manifest_semantic_labels_are_not_event_acceptance")
        is not True
        or qualification.get("semantic_events_verified") is not False
        or qualification.get("exact40_manual_qualification_required") is not exact40
        or qualification.get("qualification_unit")
        != "complete_two_seed_by_four_branch_bank"
        or qualification.get("reject_pass_a_if_either_seed_or_any_branch_fails")
        is not True
        or qualification.get("single_seed_or_branch_selection_forbidden") is not True
        or qualification.get("reward_or_training_use_authorized") is not False
        or qualification.get("pass_a_status") != expected_status
    ):
        raise SelfImaginedParetoContractError("actual Pass-A qualification contract differs")
    interpretation = _exact(
        "actual Pass-A interpretation",
        receipt["interpretation"],
        _PASS_A_INTERPRETATION_FIELDS,
    )
    if interpretation != {
        "render_complete": True,
        "pure_t2v_action_proposal_bank": True,
        "editing_result": False,
        "quality_claim": False,
        "model_training_performed": False,
        "scientific_claim_authorized": False,
    }:
        raise SelfImaginedParetoContractError("actual Pass-A interpretation differs")
    reasons: list[str] = []
    if receipt["stage"] != "exact40-qualification-candidate":
        reasons.append("pass_a_receipt_is_not_exact40")
    return indexed, reasons


def _event_map(label: str, value: Any) -> dict[str, bool]:
    if not isinstance(value, Mapping) or frozenset(value) != frozenset(EVENT_AXES):
        raise SelfImaginedParetoContractError(f"{label} event axes differ")
    return {axis: _boolean(f"{label}.{axis}", value[axis]) for axis in EVENT_AXES}


def _validate_pass_a_qualification(
    value: Mapping[str, Any],
    *,
    seal_sha256: str,
    receipt_sha256: str,
    receipt_digest: str,
    entries: Mapping[str, Mapping[str, Any]],
    initial_reasons: Sequence[str],
) -> tuple[list[str], Mapping[str, Any]]:
    seal = _exact("Pass-A qualification seal", value, _QUALIFICATION_SEAL_FIELDS)
    if seal["schema_version"] != QUALIFICATION_SEAL_SCHEMA:
        raise SelfImaginedParetoContractError("Pass-A qualification seal schema differs")
    _slug("Pass-A qualification seal_id", seal["seal_id"])
    if (
        seal["pass_a_receipt_file_sha256"] != receipt_sha256
        or seal["pass_a_receipt_digest"] != receipt_digest
    ):
        raise SelfImaginedParetoContractError(
            "Pass-A qualification is not bound to actual receipt bytes"
        )
    _sha("Pass-A event evaluator", seal["evaluator_sha256"])
    _sha("Pass-A event calibration", seal["calibration_sha256"])
    threshold = _number(
        "Pass-A absolute uncertainty threshold",
        seal["absolute_uncertainty_threshold"],
        minimum=0.0,
        strict_minimum=True,
    )
    if (
        seal["blinded_before_pass_b"] is not True
        or seal["pass_b_artifacts_available"] != []
        or seal["qualification_unit"] != "complete_two_seed_by_four_branch_bank"
    ):
        raise SelfImaginedParetoContractError(
            "Pass-A qualification is not blind/all-or-nothing"
        )
    rows = seal["entries"]
    expected = _expected_pass_a_rows()
    if not isinstance(rows, list) or len(rows) != 8:
        raise SelfImaginedParetoContractError("Pass-A qualification must cover all eight entries")
    reasons = list(initial_reasons)
    for index, (raw, expected_row) in enumerate(zip(rows, expected)):
        seed_id, _, _, branch, entry_id = expected_row
        row = _exact(
            f"Pass-A qualification entry {index}", raw, _QUALIFICATION_ENTRY_FIELDS
        )
        artifact = entries[entry_id]
        if (
            row["entry_id"] != entry_id
            or row["seed_id"] != seed_id
            or row["semantic_branch"] != branch
            or row["video_sha256"] != artifact["video_sha256"]
            or row["clean_latent_sha256"] != artifact["clean_latent_sha256"]
            or row["initial_gaussian_value_sha256"]
            != artifact["initial_gaussian_value_sha256"]
        ):
            raise SelfImaginedParetoContractError(
                "Pass-A qualification entry is not byte-bound to actual branch artifact"
            )
        calibrated = _boolean("Pass-A calibrated", row["calibrated"])
        uncertainty = _number(
            "Pass-A absolute uncertainty", row["absolute_uncertainty"], minimum=0.0
        )
        events = _event_map(f"Pass-A {entry_id}", row["event_axis_pass"])
        expected_events = dict(_EXPECTED_BRANCH_EVENTS[branch])
        derived_branch_pass = events == expected_events
        if row["branch_contract_pass"] is not derived_branch_pass:
            raise SelfImaginedParetoContractError(
                "Pass-A branch_contract_pass is not derived from exact branch semantics"
            )
        if not calibrated or uncertainty > threshold:
            reasons.append(f"{entry_id}:uncalibrated_or_uncertain")
        if not derived_branch_pass:
            reasons.append(f"{entry_id}:branch_contract_failed")
        if branch == "full_action" and not all(events.values()):
            reasons.append(f"{seed_id}:full_action_failed")
    # The seal hash is used below by the Pass-B preregistration trust root.
    _sha("Pass-A qualification external SHA-256", seal_sha256)
    return sorted(set(reasons)), seal


@dataclass(frozen=True)
class ExternalContext:
    pass_a_receipt_sha256: str
    pass_a_receipt_digest: str
    pass_a_entries: Mapping[str, Mapping[str, Any]]
    pass_a_qualification_sha256: str
    pass_a_reasons: tuple[str, ...]
    preregistration_sha256: str
    preregistration: Mapping[str, Any]
    specs: Mapping[str, Mapping[str, Any]]


_SOURCE_FIELDS = frozenset(
    {
        "sample_id",
        "source_video_sha256",
        "source_video_latent_sha256",
        "source_video_latent_shape",
        "source_vae_receipt_sha256",
        "correct_reference_tensor_sha256",
        "reference_frame_indices",
        "reference_vae_receipt_sha256",
        "stable_content_caption",
        "stable_content_caption_sha256",
        "observed_source_action",
        "observed_source_action_sha256",
        "captioner_artifact_sha256",
        "caption_artifact_sha256",
        "edit_instruction",
        "edit_instruction_sha256",
        "frame_count",
    }
)
_POLICY_FIELDS = frozenset(
    {
        "policy_id",
        "policy_revision",
        "bernini_commit",
        "veomni_commit",
        "checkpoint_sha256",
        "method_source_revision",
        "method_source_archive_sha256",
        "renderer_prompt",
        "renderer_prompt_sha256",
        "sampler_config_sha256",
        "guidance_config_sha256",
        "guidance_kind",
        "gate_order",
        "evaluator_sha256_by_gate",
        "calibration_sha256_by_gate",
        "absolute_uncertainty_threshold_by_gate",
    }
)
_LAYOUT_FIELDS = frozenset(
    {
        "layout_id",
        "renderer_mode",
        "frame_count",
        "reference_frame_indices",
        "target_source_id",
        "vi_source_count",
        "image_only_source_count",
        "vi_video_source_ids",
        "vi_reference_source_ids",
        "image_only_reference_source_ids",
        "patch_source_id_order_per_step",
        "guidance_branch_order",
        "native_source_id_interpolation_used",
    }
)
_SPEC_FIELDS = frozenset(
    {
        "candidate_id",
        "candidate_role",
        "donor_mode",
        "donor_semantic_branch",
        "pass_a_entry_id",
        "donor_latent_sha256",
        "reference_mode",
        "reference_tensor_sha256",
        "renderer_seed",
        "official_gaussian_sha256",
        "condition_layout",
        "max_gpu_seconds",
    }
)
_PAIR_FIELDS = frozenset(
    {
        "pair_id",
        "pair_type",
        "winner_candidate_id",
        "loser_candidate_id",
        "pass_a_seed_id",
        "source_copy_candidate_id",
    }
)
_COUNTERFACTUAL_FIELDS = frozenset(
    {"layout_id", "arm_to_candidate_id", "gate_rule"}
)
_PREREGISTRATION_FIELDS = frozenset(
    {
        "schema_version",
        "seal_id",
        "sealed_before_pass_b",
        "topup_allowed",
        "pass_a_receipt_file_sha256",
        "pass_a_qualification_seal_file_sha256",
        "source",
        "renderer_policy",
        "off_donor_latent_sha256",
        "reference_tensor_sha256_by_mode",
        "candidate_specs",
        "causal_pairs",
        "counterfactual_2x2",
        "candidate_count",
        "total_gpu_seconds_budget",
        "seal_digest",
    }
)

_ALLOWED_ROLES = frozenset(
    {
        "action_positive",
        "action_nearmiss",
        "preservation_nearmiss",
        "factorial_neither",
        "factorial_refs_only",
        "factorial_donor_only",
        "factorial_donor_refs",
        "source_copy_control",
        "quality_control",
    }
)


def caption_artifact_sha256(
    stable_content_caption: str,
    observed_source_action: str,
    captioner_artifact_sha256: str,
) -> str:
    return canonical_object_sha256(
        {
            "stable_content_caption": _text(
                "stable_content_caption", stable_content_caption
            ),
            "observed_source_action": _text(
                "observed_source_action", observed_source_action
            ),
            "captioner_artifact_sha256": _sha(
                "captioner_artifact_sha256", captioner_artifact_sha256
            ),
        }
    )


def build_renderer_prompt(
    stable_content_caption: str,
    observed_source_action: str,
    edit_instruction: str,
) -> str:
    return (
        f"Preserve this stable content: {_text('stable content', stable_content_caption)}\n"
        f"Replace this observed action: {_text('observed action', observed_source_action)}\n"
        f"Requested new action: {_text('edit instruction', edit_instruction)}\n"
        "Use only the preregistered donor/reference conditioning artifacts."
    )


def _validate_source(value: Any) -> Mapping[str, Any]:
    source = _exact("preregistered source", value, _SOURCE_FIELDS)
    _slug("source.sample_id", source["sample_id"])
    if source["source_video_sha256"] != pass_a_native.CDF_DOG_SOURCE_SHA256:
        raise SelfImaginedParetoContractError(
            "Pass-B source bytes differ from source-caption Pass-A source"
        )
    _sha("source latent", source["source_video_latent_sha256"])
    if source["source_video_latent_shape"] != list(pass_a_native.LATENT_SHAPE):
        raise SelfImaginedParetoContractError("source exact81 latent shape differs")
    _sha("source VAE receipt", source["source_vae_receipt_sha256"])
    _sha_list(
        "correct source reference tensors",
        source["correct_reference_tensor_sha256"],
        length=4,
    )
    if source["reference_frame_indices"] != [0, 27, 53, 80]:
        raise SelfImaginedParetoContractError("source reference frame indices differ")
    _sha_list(
        "source reference VAE receipts",
        source["reference_vae_receipt_sha256"],
        length=4,
    )
    stable = _text_digest(
        "stable content caption",
        source["stable_content_caption"],
        source["stable_content_caption_sha256"],
    )
    action = _text_digest(
        "observed source action",
        source["observed_source_action"],
        source["observed_source_action_sha256"],
    )
    if stable == action:
        raise SelfImaginedParetoContractError(
            "stable content and observed action must remain separate"
        )
    captioner = _sha("captioner artifact", source["captioner_artifact_sha256"])
    if source["caption_artifact_sha256"] != caption_artifact_sha256(
        stable, action, captioner
    ):
        raise SelfImaginedParetoContractError("caption artifact binding differs")
    _text_digest(
        "edit instruction",
        source["edit_instruction"],
        source["edit_instruction_sha256"],
    )
    if type(source["frame_count"]) is not int or source["frame_count"] != FRAME_COUNT:
        raise SelfImaginedParetoContractError("source must be exact81")
    return source


def _ordered_gate_map(label: str, value: Any, validator: Any) -> dict[str, Any]:
    # External JSON is canonicalized with sort_keys=True, so semantic order is
    # reconstructed from RENDER_GATES rather than trusted from object key order.
    if not isinstance(value, Mapping) or frozenset(value) != frozenset(RENDER_GATES):
        raise SelfImaginedParetoContractError(f"{label} must contain exact A/I/C/Q")
    return {gate: validator(f"{label}.{gate}", value[gate]) for gate in RENDER_GATES}


def _validate_policy(value: Any, source: Mapping[str, Any]) -> Mapping[str, Any]:
    policy = _exact("externally preregistered renderer policy", value, _POLICY_FIELDS)
    _slug("renderer policy_id", policy["policy_id"])
    _integer("renderer policy_revision", policy["policy_revision"])
    if policy["bernini_commit"] != pass_a_native.BERNINI_COMMIT:
        raise SelfImaginedParetoContractError("renderer Bernini commit is not pinned")
    if policy["veomni_commit"] != pass_a_native.VEOMNI_COMMIT:
        raise SelfImaginedParetoContractError("renderer VeOmni commit is not pinned")
    _sha("renderer checkpoint", policy["checkpoint_sha256"])
    _sha1("renderer method source revision", policy["method_source_revision"])
    _sha("renderer method source archive", policy["method_source_archive_sha256"])
    expected_prompt = build_renderer_prompt(
        str(source["stable_content_caption"]),
        str(source["observed_source_action"]),
        str(source["edit_instruction"]),
    )
    if policy["renderer_prompt"] != expected_prompt:
        raise SelfImaginedParetoContractError("renderer prompt content differs")
    _text_digest(
        "renderer prompt", policy["renderer_prompt"], policy["renderer_prompt_sha256"]
    )
    _sha("sampler config", policy["sampler_config_sha256"])
    _sha("guidance config", policy["guidance_config_sha256"])
    if policy["guidance_kind"] != "text_cfg_only_no_spatial_guidance":
        raise SelfImaginedParetoContractError("spatial renderer guidance is forbidden")
    if policy["gate_order"] != list(RENDER_GATES):
        raise SelfImaginedParetoContractError("renderer gate order differs")
    _ordered_gate_map(
        "renderer evaluators", policy["evaluator_sha256_by_gate"], _sha
    )
    _ordered_gate_map(
        "renderer calibrations", policy["calibration_sha256_by_gate"], _sha
    )
    _ordered_gate_map(
        "renderer absolute uncertainty thresholds",
        policy["absolute_uncertainty_threshold_by_gate"],
        lambda label, item: _number(
            label, item, minimum=0.0, strict_minimum=True
        ),
    )
    return policy


def _validate_layout(value: Any) -> Mapping[str, Any]:
    layout = _exact("native condition layout", value, _LAYOUT_FIELDS)
    _slug("native layout_id", layout["layout_id"])
    expected = {
        "renderer_mode": "native_rv2v",
        "frame_count": FRAME_COUNT,
        "reference_frame_indices": [0, 27, 53, 80],
        "target_source_id": 0.0,
        "vi_source_count": 5,
        "image_only_source_count": 4,
        "vi_video_source_ids": [1.0],
        "vi_reference_source_ids": [2.0, 3.0, 4.0, 5.0],
        "image_only_reference_source_ids": [1.0, 2.0, 3.0, 4.0],
        "patch_source_id_order_per_step": [
            1.0,
            2.0,
            1.0,
            3.0,
            2.0,
            4.0,
            3.0,
            5.0,
            4.0,
            0.0,
        ],
        "guidance_branch_order": ["none", "V", "VI_uncond", "VI_text"],
        "native_source_id_interpolation_used": False,
    }
    if any(layout[key] != item for key, item in expected.items()):
        raise SelfImaginedParetoContractError(
            "native rv2v VI/I source-ID and guidance order differs"
        )
    return layout


def _validate_reference_banks(
    value: Any, source: Mapping[str, Any]
) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or frozenset(value) != {
        "correct",
        "wrong",
        "off",
    }:
        raise SelfImaginedParetoContractError(
            "reference banks must be ordered correct/wrong/off"
        )
    result = {
        mode: _sha_list(f"reference bank {mode}", value[mode], length=4)
        for mode in ("correct", "wrong", "off")
    }
    if result["correct"] != tuple(source["correct_reference_tensor_sha256"]):
        raise SelfImaginedParetoContractError("correct refs differ from sealed source")
    for left, right in (("correct", "wrong"), ("correct", "off"), ("wrong", "off")):
        if set(result[left]) & set(result[right]):
            raise SelfImaginedParetoContractError("reference banks alias artifacts")
    return result


def _pass_a_seed_id(entry_id: str, entries: Mapping[str, Mapping[str, Any]]) -> str:
    if entry_id not in entries:
        raise SelfImaginedParetoContractError("candidate references absent Pass-A entry")
    return str(entries[entry_id]["seed_id"])


def _validate_spec(
    value: Any,
    *,
    source: Mapping[str, Any],
    entries: Mapping[str, Mapping[str, Any]],
    off_donor: str,
    reference_banks: Mapping[str, tuple[str, ...]],
) -> Mapping[str, Any]:
    spec = _exact("externally preregistered candidate spec", value, _SPEC_FIELDS)
    _slug("candidate_id", spec["candidate_id"])
    if spec["candidate_role"] not in _ALLOWED_ROLES:
        raise SelfImaginedParetoContractError("candidate role differs")
    mode = spec["donor_mode"]
    if mode == "proposal":
        entry_id = spec["pass_a_entry_id"]
        if entry_id not in entries:
            raise SelfImaginedParetoContractError("proposal entry is absent")
        entry = entries[entry_id]
        if (
            spec["donor_semantic_branch"] != entry["semantic_branch"]
            or spec["donor_latent_sha256"] != entry["clean_latent_sha256"]
        ):
            raise SelfImaginedParetoContractError(
                "candidate donor differs from actual Pass-A latent bytes"
            )
    elif mode == "source_video":
        if (
            spec["pass_a_entry_id"] is not None
            or spec["donor_semantic_branch"] != "source_action"
            or spec["donor_latent_sha256"] != source["source_video_latent_sha256"]
        ):
            raise SelfImaginedParetoContractError("source-copy donor differs")
    elif mode == "off":
        if (
            spec["pass_a_entry_id"] is not None
            or spec["donor_semantic_branch"] != "off"
            or spec["donor_latent_sha256"] != off_donor
        ):
            raise SelfImaginedParetoContractError("off donor differs")
    else:
        raise SelfImaginedParetoContractError("candidate donor mode differs")
    reference_mode = spec["reference_mode"]
    if reference_mode not in reference_banks:
        raise SelfImaginedParetoContractError("candidate reference mode differs")
    refs = _sha_list(
        "candidate reference tensors", spec["reference_tensor_sha256"], length=4
    )
    if refs != reference_banks[reference_mode]:
        raise SelfImaginedParetoContractError("candidate refs differ from sealed bank")
    _integer("candidate renderer seed", spec["renderer_seed"])
    _sha("candidate official Gaussian", spec["official_gaussian_sha256"])
    _validate_layout(spec["condition_layout"])
    _number(
        "candidate max GPU seconds",
        spec["max_gpu_seconds"],
        minimum=0.0,
        strict_minimum=True,
    )
    return spec


def _spec_common_signature(spec: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        spec["renderer_seed"],
        spec["official_gaussian_sha256"],
        spec["condition_layout"],
        float(spec["max_gpu_seconds"]),
    )


def _validate_preregistered_pairs(
    rows: Any,
    *,
    specs: Mapping[str, Mapping[str, Any]],
    entries: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise SelfImaginedParetoContractError("causal pairs must be preregistered")
    result: list[Mapping[str, Any]] = []
    pair_ids: set[str] = set()
    pair_types: set[str] = set()
    for raw in rows:
        pair = _exact("preregistered causal pair", raw, _PAIR_FIELDS)
        pair_id = _slug("causal pair_id", pair["pair_id"])
        if pair_id in pair_ids or pair["pair_type"] not in PAIR_TYPES:
            raise SelfImaginedParetoContractError("causal pair id/type differs")
        pair_ids.add(pair_id)
        pair_types.add(str(pair["pair_type"]))
        winner_id = str(pair["winner_candidate_id"])
        loser_id = str(pair["loser_candidate_id"])
        source_copy_id = str(pair["source_copy_candidate_id"])
        if (
            winner_id not in specs
            or loser_id not in specs
            or source_copy_id not in specs
            or len({winner_id, loser_id, source_copy_id}) != 3
        ):
            raise SelfImaginedParetoContractError("causal pair candidate binding differs")
        winner = specs[winner_id]
        loser = specs[loser_id]
        source_copy = specs[source_copy_id]
        seed_id = _slug("causal pair Pass-A seed_id", pair["pass_a_seed_id"])
        if (
            winner["candidate_role"] != "action_positive"
            or winner["donor_mode"] != "proposal"
            or winner["donor_semantic_branch"] != "full_action"
            or winner["reference_mode"] != "correct"
            or source_copy["candidate_role"] != "source_copy_control"
            or source_copy["donor_mode"] != "source_video"
            or source_copy["reference_mode"] != "correct"
            or source_copy["reference_tensor_sha256"]
            != winner["reference_tensor_sha256"]
            or _spec_common_signature(source_copy) != _spec_common_signature(winner)
        ):
            raise SelfImaginedParetoContractError(
                "source-copy must bind the selected correct-ref action winner"
            )
        winner_entry = str(winner["pass_a_entry_id"])
        if _pass_a_seed_id(winner_entry, entries) != seed_id:
            raise SelfImaginedParetoContractError("winner Pass-A seed binding differs")
        if pair["pair_type"] == "action_donor_nearmiss":
            loser_entry = str(loser["pass_a_entry_id"])
            if (
                loser["candidate_role"] != "action_nearmiss"
                or loser["donor_mode"] != "proposal"
                or loser["donor_semantic_branch"]
                not in {"noop", "incomplete", "reverse"}
                or _pass_a_seed_id(loser_entry, entries) != seed_id
                or loser["reference_mode"] != "correct"
                or loser["reference_tensor_sha256"]
                != winner["reference_tensor_sha256"]
                or _spec_common_signature(loser) != _spec_common_signature(winner)
            ):
                raise SelfImaginedParetoContractError(
                    "action contrast is not same-Pass-A-seed and one-factor"
                )
        else:
            if (
                loser["candidate_role"] != "preservation_nearmiss"
                or loser["donor_mode"] != "proposal"
                or loser["pass_a_entry_id"] != winner_entry
                or loser["donor_latent_sha256"] != winner["donor_latent_sha256"]
                or loser["reference_mode"] not in {"wrong", "off"}
                or _spec_common_signature(loser) != _spec_common_signature(winner)
            ):
                raise SelfImaginedParetoContractError(
                    "identity contrast is not an exact reference-bank intervention"
                )
        result.append(pair)
    if pair_types != set(PAIR_TYPES):
        raise SelfImaginedParetoContractError(
            "both action and identity causal pair types must be preregistered"
        )
    return result


def _validate_preregistered_counterfactual(
    value: Any,
    *,
    specs: Mapping[str, Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    grid = _exact("preregistered counterfactual 2x2", value, _COUNTERFACTUAL_FIELDS)
    layout_id = _slug("counterfactual layout_id", grid["layout_id"])
    if grid["gate_rule"] != (
        "donor_artifact_controls_A_reference_artifact_controls_I_"
        "with_CQ_pass_and_all_other_invocation_fields_equal"
    ):
        raise SelfImaginedParetoContractError("counterfactual gate rule differs")
    mapping = grid["arm_to_candidate_id"]
    if not isinstance(mapping, Mapping) or frozenset(mapping) != frozenset(
        COUNTERFACTUAL_ARMS
    ):
        raise SelfImaginedParetoContractError("counterfactual arms differ")
    if len(set(mapping.values())) != 4:
        raise SelfImaginedParetoContractError("counterfactual candidates repeat")
    arm_specs: dict[str, Mapping[str, Any]] = {}
    for arm, (role, donor_mode, ref_mode) in COUNTERFACTUAL_ARMS.items():
        candidate_id = str(mapping[arm])
        if candidate_id not in specs:
            raise SelfImaginedParetoContractError("counterfactual candidate is absent")
        spec = specs[candidate_id]
        if (
            spec["candidate_role"] != role
            or spec["donor_mode"] != donor_mode
            or spec["reference_mode"] != ref_mode
            or spec["condition_layout"]["layout_id"] != layout_id
        ):
            raise SelfImaginedParetoContractError("counterfactual arm binding differs")
        arm_specs[arm] = spec
    donor = arm_specs["donor_and_identity_refs"]
    action_winner_donors = {
        specs[str(pair["winner_candidate_id"])]["donor_latent_sha256"]
        for pair in pairs
        if pair["pair_type"] == "action_donor_nearmiss"
    }
    if (
        donor["donor_semantic_branch"] != "full_action"
        or donor["donor_latent_sha256"] not in action_winner_donors
        or arm_specs["donor_only"]["donor_latent_sha256"]
        != donor["donor_latent_sha256"]
        or arm_specs["identity_refs_only"]["reference_tensor_sha256"]
        != donor["reference_tensor_sha256"]
    ):
        raise SelfImaginedParetoContractError(
            "2x2 is not bound to the selected action winner donor/refs"
        )
    common = _spec_common_signature(arm_specs["neither"])
    if any(_spec_common_signature(spec) != common for spec in arm_specs.values()):
        raise SelfImaginedParetoContractError(
            "2x2 prompt/Gaussian/layout/budget is not byte-identical"
        )
    return grid


def _validate_preregistration(
    value: Mapping[str, Any],
    *,
    pass_a_receipt_sha256: str,
    pass_a_qualification_sha256: str,
    entries: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    seal = _exact("external Pass-B preregistration", value, _PREREGISTRATION_FIELDS)
    if seal["schema_version"] != PREREGISTRATION_SEAL_SCHEMA:
        raise SelfImaginedParetoContractError("Pass-B preregistration schema differs")
    _slug("Pass-B seal_id", seal["seal_id"])
    if seal["sealed_before_pass_b"] is not True or seal["topup_allowed"] is not False:
        raise SelfImaginedParetoContractError(
            "external Pass-B seal must precede rendering and forbid top-up"
        )
    if (
        seal["pass_a_receipt_file_sha256"] != pass_a_receipt_sha256
        or seal["pass_a_qualification_seal_file_sha256"]
        != pass_a_qualification_sha256
    ):
        raise SelfImaginedParetoContractError(
            "Pass-B seal is not bound to actual Pass-A/qualification bytes"
        )
    source = _validate_source(seal["source"])
    _validate_policy(seal["renderer_policy"], source)
    off_donor = _sha("off donor latent", seal["off_donor_latent_sha256"])
    if off_donor == source["source_video_latent_sha256"] or off_donor in {
        row["clean_latent_sha256"] for row in entries.values()
    }:
        raise SelfImaginedParetoContractError("off donor aliases source/Pass-A")
    ref_banks = _validate_reference_banks(
        seal["reference_tensor_sha256_by_mode"], source
    )
    rows = seal["candidate_specs"]
    if not isinstance(rows, list) or not rows:
        raise SelfImaginedParetoContractError("candidate specs are absent")
    specs: dict[str, Mapping[str, Any]] = {}
    seed_to_gaussian: dict[int, str] = {}
    gaussian_to_seed: dict[str, int] = {}
    for raw in rows:
        spec = _validate_spec(
            raw,
            source=source,
            entries=entries,
            off_donor=off_donor,
            reference_banks=ref_banks,
        )
        candidate_id = str(spec["candidate_id"])
        if candidate_id in specs:
            raise SelfImaginedParetoContractError("candidate spec repeats")
        seed = int(spec["renderer_seed"])
        gaussian = str(spec["official_gaussian_sha256"])
        if seed in seed_to_gaussian and seed_to_gaussian[seed] != gaussian:
            raise SelfImaginedParetoContractError(
                "one renderer seed maps to multiple Gaussian tensors"
            )
        if gaussian in gaussian_to_seed and gaussian_to_seed[gaussian] != seed:
            raise SelfImaginedParetoContractError(
                "different renderer seeds reuse one Gaussian tensor"
            )
        seed_to_gaussian[seed] = gaussian
        gaussian_to_seed[gaussian] = seed
        specs[candidate_id] = spec
    if type(seal["candidate_count"]) is not int or seal["candidate_count"] != len(specs):
        raise SelfImaginedParetoContractError("candidate count differs from external seal")
    total_budget = _number(
        "total GPU seconds budget",
        seal["total_gpu_seconds_budget"],
        minimum=0.0,
        strict_minimum=True,
    )
    if total_budget < sum(float(spec["max_gpu_seconds"]) for spec in specs.values()):
        raise SelfImaginedParetoContractError("total budget is below candidate maxima")
    pairs = _validate_preregistered_pairs(
        seal["causal_pairs"], specs=specs, entries=entries
    )
    _validate_preregistered_counterfactual(
        seal["counterfactual_2x2"], specs=specs, pairs=pairs
    )
    return seal, specs


def build_external_context(
    *,
    pass_a_receipt_bytes: bytes,
    expected_pass_a_receipt_sha256: str,
    pass_a_artifact_hashes: Mapping[str, Mapping[str, str]],
    pass_a_qualification_seal_bytes: bytes,
    expected_pass_a_qualification_seal_sha256: str,
    preregistration_seal_bytes: bytes,
    expected_preregistration_seal_sha256: str,
) -> ExternalContext:
    """Build V3 trust context exclusively from caller-pinned external bytes."""

    receipt, receipt_sha = _load_external_json(
        "actual Pass-A bank receipt",
        pass_a_receipt_bytes,
        expected_pass_a_receipt_sha256,
        digest_field="receipt_digest",
    )
    entries, initial_reasons = _validate_pass_a_receipt(
        receipt, pass_a_artifact_hashes
    )
    qualification, qualification_sha = _load_external_json(
        "Pass-A qualification seal",
        pass_a_qualification_seal_bytes,
        expected_pass_a_qualification_seal_sha256,
        digest_field="seal_digest",
    )
    reasons, _ = _validate_pass_a_qualification(
        qualification,
        seal_sha256=qualification_sha,
        receipt_sha256=receipt_sha,
        receipt_digest=str(receipt["receipt_digest"]),
        entries=entries,
        initial_reasons=initial_reasons,
    )
    preregistration, prereg_sha = _load_external_json(
        "Pass-B preregistration seal",
        preregistration_seal_bytes,
        expected_preregistration_seal_sha256,
        digest_field="seal_digest",
    )
    preregistration, specs = _validate_preregistration(
        preregistration,
        pass_a_receipt_sha256=receipt_sha,
        pass_a_qualification_sha256=qualification_sha,
        entries=entries,
    )
    return ExternalContext(
        pass_a_receipt_sha256=receipt_sha,
        pass_a_receipt_digest=str(receipt["receipt_digest"]),
        pass_a_entries=MappingProxyType(dict(entries)),
        pass_a_qualification_sha256=qualification_sha,
        pass_a_reasons=tuple(reasons),
        preregistration_sha256=prereg_sha,
        preregistration=MappingProxyType(dict(preregistration)),
        specs=MappingProxyType(dict(specs)),
    )


_EPISODE_FIELDS = frozenset(
    {
        "schema_version",
        "episode_id",
        "external_seals",
        "candidates",
        "measured_gpu_seconds",
        "episode_digest",
    }
)
_EXTERNAL_SEAL_FIELDS = frozenset(
    {
        "pass_a_receipt_file_sha256",
        "pass_a_qualification_seal_file_sha256",
        "pass_b_preregistration_seal_file_sha256",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "native_invocation",
        "gates",
        "joint_pass",
        "disposition",
        "candidate_digest",
    }
)
_INVOCATION_FIELDS = frozenset(
    {
        "schema_version",
        "invocation_id",
        "candidate_id",
        "checkpoint_sha256",
        "donor",
        "references",
        "condition_layout",
        "renderer_prompt",
        "renderer_prompt_sha256",
        "sampler_config_sha256",
        "guidance_config_sha256",
        "guidance_kind",
        "official_gaussian",
        "external_inputs",
        "output_clean_latent_sha256",
        "output_video_sha256",
        "output_frame_count",
        "output_fps",
        "output_height",
        "output_width",
        "output_clean_latent_shape",
        "measured_gpu_seconds",
        "invocation_digest",
    }
)
_DONOR_FIELDS = frozenset(
    {"mode", "semantic_branch", "pass_a_entry_id", "latent_sha256"}
)
_REFERENCE_FIELDS = frozenset({"mode", "tensor_sha256"})
_GAUSSIAN_FIELDS = frozenset({"source", "seed", "tensor_sha256"})
_GATE_FIELDS = frozenset(
    {
        "evaluator_sha256",
        "calibration_sha256",
        "absolute_uncertainty_threshold",
        "calibrated",
        "margin",
        "absolute_uncertainty",
        "pass",
        "evaluation_input_digest",
    }
)
_DISPOSITION_FIELDS = frozenset({"status", "failed_gates"})
_PASS_B_ARTIFACT_FIELDS = frozenset(
    {
        "invocation_digest",
        "official_gaussian_sha256",
        "output_clean_latent_sha256",
        "output_video_sha256",
        "output_frame_count",
        "output_fps",
        "output_height",
        "output_width",
        "output_clean_latent_shape",
    }
)


def _validate_gate(
    value: Any,
    *,
    gate_name: str,
    policy: Mapping[str, Any],
    source: Mapping[str, Any],
    output_video_sha256: str,
) -> Mapping[str, Any]:
    gate = _exact(f"candidate gate {gate_name}", value, _GATE_FIELDS)
    evaluator = policy["evaluator_sha256_by_gate"][gate_name]
    calibration = policy["calibration_sha256_by_gate"][gate_name]
    threshold = policy["absolute_uncertainty_threshold_by_gate"][gate_name]
    if (
        gate["evaluator_sha256"] != evaluator
        or gate["calibration_sha256"] != calibration
        or gate["absolute_uncertainty_threshold"] != threshold
    ):
        raise SelfImaginedParetoContractError(
            "candidate evaluator/calibration/threshold differs from external seal"
        )
    calibrated = _boolean("candidate calibrated", gate["calibrated"])
    uncertainty = _number(
        "candidate absolute uncertainty", gate["absolute_uncertainty"], minimum=0.0
    )
    passed = _boolean("candidate gate pass", gate["pass"])
    if calibrated:
        margin = _number("candidate gate margin", gate["margin"])
        if margin == 0.0:
            raise SelfImaginedParetoContractError("candidate margin lies on threshold")
        expected_pass = margin > 0.0 and uncertainty <= float(threshold)
    else:
        if gate["margin"] is not None:
            raise SelfImaginedParetoContractError(
                "uncalibrated candidate cannot claim a margin"
            )
        expected_pass = False
    if passed is not expected_pass:
        raise SelfImaginedParetoContractError("candidate hard gate is not derived")
    expected_input = canonical_object_sha256(
        {
            "gate": gate_name,
            "output_video_sha256": output_video_sha256,
            "source_video_sha256": source["source_video_sha256"],
            "edit_instruction_sha256": source["edit_instruction_sha256"],
            "evaluator_sha256": evaluator,
            "calibration_sha256": calibration,
            "absolute_uncertainty_threshold": threshold,
        }
    )
    if gate["evaluation_input_digest"] != expected_input:
        raise SelfImaginedParetoContractError(
            "critic evidence is not bound to output/source/instruction/policy"
        )
    return gate


def _validate_invocation(
    value: Any,
    *,
    spec: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> Mapping[str, Any]:
    invocation = _exact("native Pass-B invocation", value, _INVOCATION_FIELDS)
    if invocation["schema_version"] != INVOCATION_SCHEMA:
        raise SelfImaginedParetoContractError("native invocation schema differs")
    _slug("native invocation_id", invocation["invocation_id"])
    if invocation["candidate_id"] != spec["candidate_id"]:
        raise SelfImaginedParetoContractError("native invocation candidate differs")
    for field in (
        "checkpoint_sha256",
        "sampler_config_sha256",
        "guidance_config_sha256",
        "guidance_kind",
        "renderer_prompt",
        "renderer_prompt_sha256",
    ):
        if invocation[field] != policy[field]:
            raise SelfImaginedParetoContractError(
                "native invocation policy/config/prompt differs from external seal"
            )
    donor = _exact("native donor", invocation["donor"], _DONOR_FIELDS)
    if donor != {
        "mode": spec["donor_mode"],
        "semantic_branch": spec["donor_semantic_branch"],
        "pass_a_entry_id": spec["pass_a_entry_id"],
        "latent_sha256": spec["donor_latent_sha256"],
    }:
        raise SelfImaginedParetoContractError("native donor differs from external seal")
    refs = _exact("native references", invocation["references"], _REFERENCE_FIELDS)
    if refs != {
        "mode": spec["reference_mode"],
        "tensor_sha256": spec["reference_tensor_sha256"],
    }:
        raise SelfImaginedParetoContractError("native refs differ from external seal")
    _validate_layout(invocation["condition_layout"])
    if invocation["condition_layout"] != spec["condition_layout"]:
        raise SelfImaginedParetoContractError("native source-ID layout differs")
    gaussian = _exact(
        "native official Gaussian", invocation["official_gaussian"], _GAUSSIAN_FIELDS
    )
    if gaussian != {
        "source": OFFICIAL_GAUSSIAN_SOURCE,
        "seed": spec["renderer_seed"],
        "tensor_sha256": spec["official_gaussian_sha256"],
    }:
        raise SelfImaginedParetoContractError(
            "native renderer Gaussian differs from external seal"
        )
    if invocation["external_inputs"] != [
        "checkpoint",
        "donor_latent",
        "reference_tensors",
        "renderer_prompt",
        "sampler_config",
        "guidance_config",
        "official_gaussian",
    ]:
        raise SelfImaginedParetoContractError("native external-input closure differs")
    latent = _sha("native output clean latent", invocation["output_clean_latent_sha256"])
    video = _sha("native output video", invocation["output_video_sha256"])
    if latent == video:
        raise SelfImaginedParetoContractError("native latent/video artifacts alias")
    if (
        type(invocation["output_frame_count"]) is not int
        or invocation["output_frame_count"] != FRAME_COUNT
        or type(invocation["output_fps"]) is not int
        or invocation["output_fps"] != pass_a_native.FPS
        or type(invocation["output_height"]) is not int
        or invocation["output_height"] != pass_a_native.VIDEO_HEIGHT
        or type(invocation["output_width"]) is not int
        or invocation["output_width"] != pass_a_native.VIDEO_WIDTH
        or invocation["output_clean_latent_shape"]
        != list(pass_a_native.LATENT_SHAPE)
    ):
        raise SelfImaginedParetoContractError(
            "native Pass-B output is not exact81/25fps/registered latent geometry"
        )
    measured = _number(
        "native measured GPU seconds",
        invocation["measured_gpu_seconds"],
        minimum=0.0,
        strict_minimum=True,
    )
    if measured > float(spec["max_gpu_seconds"]):
        raise SelfImaginedParetoContractError("native invocation exceeded sealed budget")
    _embedded_digest("native invocation", invocation, "invocation_digest")
    return invocation


def _validate_candidate(
    value: Any,
    *,
    spec: Mapping[str, Any],
    policy: Mapping[str, Any],
    source: Mapping[str, Any],
    artifact_registry: Mapping[str, Any],
) -> Mapping[str, Any]:
    candidate = _exact("Pass-B candidate", value, _CANDIDATE_FIELDS)
    if candidate["schema_version"] != CANDIDATE_SCHEMA:
        raise SelfImaginedParetoContractError("candidate schema differs")
    if candidate["candidate_id"] != spec["candidate_id"]:
        raise SelfImaginedParetoContractError("candidate order/id differs from external seal")
    invocation = _validate_invocation(
        candidate["native_invocation"], spec=spec, policy=policy
    )
    registry = _exact(
        f"Pass-B artifact registry {candidate['candidate_id']}",
        artifact_registry,
        _PASS_B_ARTIFACT_FIELDS,
    )
    expected_registry = {
        "invocation_digest": invocation["invocation_digest"],
        "official_gaussian_sha256": invocation["official_gaussian"]["tensor_sha256"],
        "output_clean_latent_sha256": invocation["output_clean_latent_sha256"],
        "output_video_sha256": invocation["output_video_sha256"],
        "output_frame_count": invocation["output_frame_count"],
        "output_fps": invocation["output_fps"],
        "output_height": invocation["output_height"],
        "output_width": invocation["output_width"],
        "output_clean_latent_shape": invocation["output_clean_latent_shape"],
    }
    if dict(registry) != expected_registry:
        raise SelfImaginedParetoContractError(
            "Pass-B independently observed artifact hashes differ"
        )
    gates = candidate["gates"]
    if not isinstance(gates, Mapping) or frozenset(gates) != frozenset(RENDER_GATES):
        raise SelfImaginedParetoContractError("candidate gates must be ordered A/I/C/Q")
    checked = {
        gate: _validate_gate(
            gates[gate],
            gate_name=gate,
            policy=policy,
            source=source,
            output_video_sha256=str(invocation["output_video_sha256"]),
        )
        for gate in RENDER_GATES
    }
    joint = all(bool(checked[gate]["pass"]) for gate in RENDER_GATES)
    if candidate["joint_pass"] is not joint:
        raise SelfImaginedParetoContractError(
            "candidate joint pass must be strict A AND I AND C AND Q"
        )
    failed = [gate for gate in RENDER_GATES if not checked[gate]["pass"]]
    disposition = _exact("candidate disposition", candidate["disposition"], _DISPOSITION_FIELDS)
    expected_disposition = {
        "status": "strict_joint_accepted" if not failed else "explicitly_rejected",
        "failed_gates": failed,
    }
    if dict(disposition) != expected_disposition:
        raise SelfImaginedParetoContractError("candidate disposition differs")
    _embedded_digest(
        f"candidate {candidate['candidate_id']}", candidate, "candidate_digest"
    )
    return candidate


def _trusted(candidate: Mapping[str, Any]) -> bool:
    return all(
        gate["calibrated"]
        and float(gate["absolute_uncertainty"])
        <= float(gate["absolute_uncertainty_threshold"])
        for gate in candidate["gates"].values()
    )


def _failed(candidate: Mapping[str, Any]) -> list[str]:
    return [gate for gate in RENDER_GATES if not candidate["gates"][gate]["pass"]]


def _candidate_nonworse(
    winner: Mapping[str, Any], loser: Mapping[str, Any], gates: Sequence[str]
) -> bool:
    return all(
        float(winner["gates"][gate]["margin"])
        >= float(loser["gates"][gate]["margin"])
        for gate in gates
    )


def _validate_episode_internal(
    value: Any,
    *,
    context: ExternalContext,
    pass_b_artifact_hashes: Mapping[str, Mapping[str, str]],
) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    episode = _exact("V3 episode", value, _EPISODE_FIELDS)
    if episode["schema_version"] != EPISODE_SCHEMA:
        raise SelfImaginedParetoContractError("episode schema differs")
    _slug("episode_id", episode["episode_id"])
    seals = _exact("episode external seals", episode["external_seals"], _EXTERNAL_SEAL_FIELDS)
    if seals != {
        "pass_a_receipt_file_sha256": context.pass_a_receipt_sha256,
        "pass_a_qualification_seal_file_sha256": context.pass_a_qualification_sha256,
        "pass_b_preregistration_seal_file_sha256": context.preregistration_sha256,
    }:
        raise SelfImaginedParetoContractError("episode external trust roots differ")
    rows = episode["candidates"]
    measured = _number(
        "episode measured GPU seconds", episode["measured_gpu_seconds"], minimum=0.0
    )
    if context.pass_a_reasons:
        if rows != [] or measured != 0.0 or dict(pass_b_artifact_hashes) != {}:
            raise SelfImaginedParetoContractError(
                "failed Pass-A bank must null before any Pass-B rendering"
            )
        _embedded_digest("episode", episode, "episode_digest")
        return episode, {}
    if not isinstance(rows, list) or len(rows) != len(context.specs):
        raise SelfImaginedParetoContractError(
            "Pass-B outputs do not exactly exhaust external candidate seal"
        )
    if not isinstance(pass_b_artifact_hashes, Mapping) or frozenset(
        pass_b_artifact_hashes
    ) != frozenset(context.specs):
        raise SelfImaginedParetoContractError(
            "Pass-B artifact registry does not exactly exhaust external seal"
        )
    policy = context.preregistration["renderer_policy"]
    source = context.preregistration["source"]
    candidates: dict[str, Mapping[str, Any]] = {}
    output_hashes: set[str] = set()
    for raw, (candidate_id, spec) in zip(rows, context.specs.items()):
        candidate = _validate_candidate(
            raw,
            spec=spec,
            policy=policy,
            source=source,
            artifact_registry=pass_b_artifact_hashes[candidate_id],
        )
        invocation = candidate["native_invocation"]
        for artifact in (
            invocation["output_clean_latent_sha256"],
            invocation["output_video_sha256"],
        ):
            if artifact in output_hashes:
                raise SelfImaginedParetoContractError(
                    "Pass-B output artifacts must be globally distinct"
                )
            output_hashes.add(str(artifact))
        candidates[candidate_id] = candidate
    observed = sum(
        float(candidate["native_invocation"]["measured_gpu_seconds"])
        for candidate in candidates.values()
    )
    if not math.isclose(measured, observed, rel_tol=0.0, abs_tol=1e-9):
        raise SelfImaginedParetoContractError("episode GPU sum differs")
    if measured > float(context.preregistration["total_gpu_seconds_budget"]):
        raise SelfImaginedParetoContractError("episode exceeded external total budget")
    _embedded_digest("episode", episode, "episode_digest")
    return episode, candidates


def validate_episode(
    value: Any,
    *,
    context: ExternalContext,
    pass_b_artifact_hashes: Mapping[str, Mapping[str, str]],
) -> Mapping[str, Any]:
    """Validate an episode against caller-pinned external seals and registries."""

    episode, _ = _validate_episode_internal(
        value, context=context, pass_b_artifact_hashes=pass_b_artifact_hashes
    )
    return episode


def _runtime_common_signature(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    invocation = candidate["native_invocation"]
    return (
        invocation["checkpoint_sha256"],
        invocation["condition_layout"],
        invocation["renderer_prompt"],
        invocation["renderer_prompt_sha256"],
        invocation["sampler_config_sha256"],
        invocation["guidance_config_sha256"],
        invocation["guidance_kind"],
        invocation["official_gaussian"],
        tuple(invocation["external_inputs"]),
    )


def _validate_runtime_controls(
    *,
    context: ExternalContext,
    candidates: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    grid = context.preregistration["counterfactual_2x2"]
    mapping = grid["arm_to_candidate_id"]
    rows = {arm: candidates[str(candidate_id)] for arm, candidate_id in mapping.items()}
    common = _runtime_common_signature(rows["neither"])
    output_pairs: set[tuple[str, str]] = set()
    for arm, candidate in rows.items():
        if _runtime_common_signature(candidate) != common:
            raise SelfImaginedParetoContractError(
                "runtime 2x2 changed prompt/Gaussian/layout/policy"
            )
        invocation = candidate["native_invocation"]
        output_pair = (
            str(invocation["output_clean_latent_sha256"]),
            str(invocation["output_video_sha256"]),
        )
        if output_pair in output_pairs:
            raise SelfImaginedParetoContractError("runtime 2x2 repeats output artifacts")
        output_pairs.add(output_pair)
        if not _trusted(candidate):
            reasons.append(f"{arm}:critic_uncalibrated_or_uncertain")
        donor_present = COUNTERFACTUAL_ARMS[arm][1] == "proposal"
        refs_present = COUNTERFACTUAL_ARMS[arm][2] == "correct"
        expected = {"A": donor_present, "I": refs_present, "C": True, "Q": True}
        for gate, passed in expected.items():
            if candidate["gates"][gate]["pass"] is not passed:
                reasons.append(f"{arm}:{gate}_causal_pattern_failed")
    return reasons


def _action_pair_eligible(
    winner: Mapping[str, Any], loser: Mapping[str, Any]
) -> bool:
    win = winner["native_invocation"]
    lose = loser["native_invocation"]
    return (
        winner["joint_pass"] is True
        and _failed(loser) == ["A"]
        and _trusted(winner)
        and _trusted(loser)
        and _runtime_common_signature(winner) == _runtime_common_signature(loser)
        and win["references"] == lose["references"]
        and win["references"]["mode"] == "correct"
        and win["donor"]["semantic_branch"] == "full_action"
        and lose["donor"]["semantic_branch"] in {"noop", "incomplete", "reverse"}
        and win["donor"]["pass_a_entry_id"] != lose["donor"]["pass_a_entry_id"]
        and win["donor"]["latent_sha256"] != lose["donor"]["latent_sha256"]
        and _candidate_nonworse(winner, loser, ("I", "C", "Q"))
    )


def _identity_pair_eligible(
    winner: Mapping[str, Any], loser: Mapping[str, Any]
) -> bool:
    win = winner["native_invocation"]
    lose = loser["native_invocation"]
    return (
        winner["joint_pass"] is True
        and _failed(loser) == ["I"]
        and _trusted(winner)
        and _trusted(loser)
        and _runtime_common_signature(winner) == _runtime_common_signature(loser)
        and win["donor"] == lose["donor"]
        and win["references"]["mode"] == "correct"
        and lose["references"]["mode"] in {"wrong", "off"}
        and win["references"]["tensor_sha256"]
        != lose["references"]["tensor_sha256"]
        and _candidate_nonworse(winner, loser, ("A", "C", "Q"))
    )


def _qualification(
    pair: Mapping[str, Any],
    winner: Mapping[str, Any],
    loser: Mapping[str, Any],
) -> dict[str, Any]:
    pair_type = str(pair["pair_type"])
    result: dict[str, Any] = {
        "qualification_schema": "bernini-self-imagined-causal-pair-qualification-v3",
        "preregistered_pair_id": pair["pair_id"],
        "causal_pair_type": pair_type,
        "pass_a_seed_id": pair["pass_a_seed_id"],
        "winner_candidate_digest": winner["candidate_digest"],
        "rejected_candidate_digest": loser["candidate_digest"],
        "source_copy_candidate_id": pair["source_copy_candidate_id"],
        "changed_condition_factor": (
            "same_seed_proposal_donor_semantic_branch"
            if pair_type == "action_donor_nearmiss"
            else "reference_tensor_bank"
        ),
        "unaffected_gates": (
            ["I", "C", "Q"]
            if pair_type == "action_donor_nearmiss"
            else ["A", "C", "Q"]
        ),
        "winner_margins_nonworse_on_unaffected_gates": True,
        "not_a_dclr_preference_pair": True,
        "production_dclr_consumption_authorized": False,
    }
    result["qualification_digest"] = canonical_object_sha256(result)
    return result


def _null_receipt(
    *,
    episode: Mapping[str, Any],
    context: ExternalContext,
    reasons: Sequence[str],
    renderer_count: int,
    measured: float,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "episode_id": episode["episode_id"],
        "episode_digest": episode["episode_digest"],
        "upstream_pair_qualified": False,
        "status": "null_update",
        "null_reasons": sorted(set(reasons)),
        "qualifications": [],
        "external_trust_roots": {
            "pass_a_receipt_file_sha256": context.pass_a_receipt_sha256,
            "pass_a_qualification_seal_file_sha256": context.pass_a_qualification_sha256,
            "pass_b_preregistration_seal_file_sha256": context.preregistration_sha256,
        },
        "search_accounting": {
            "pass_a_generation_count": 8,
            "pass_b_candidate_count": renderer_count,
            "pass_b_measured_gpu_seconds": measured,
            "total_model_generations": 8 + renderer_count,
            "topup_observed": False,
        },
        "production_dclr_consumption_authorized": False,
        "bridge_status": "absent_requires_separately_validated_adapter",
    }
    receipt["receipt_digest"] = canonical_object_sha256(receipt)
    return receipt


def select_upstream_qualification(
    value: Any,
    *,
    context: ExternalContext,
    pass_b_artifact_hashes: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Return all preregistered causal pairs or one fail-closed null update."""

    episode, candidates = _validate_episode_internal(
        value, context=context, pass_b_artifact_hashes=pass_b_artifact_hashes
    )
    measured = float(episode["measured_gpu_seconds"])
    if context.pass_a_reasons:
        return _null_receipt(
            episode=episode,
            context=context,
            reasons=context.pass_a_reasons,
            renderer_count=0,
            measured=0.0,
        )
    reasons = _validate_runtime_controls(context=context, candidates=candidates)
    if any(not _trusted(candidate) for candidate in candidates.values()):
        reasons.append("critic_uncalibrated_or_above_external_uncertainty_threshold")
    qualifications: list[dict[str, Any]] = []
    for pair in context.preregistration["causal_pairs"]:
        winner = candidates[str(pair["winner_candidate_id"])]
        loser = candidates[str(pair["loser_candidate_id"])]
        source_copy = candidates[str(pair["source_copy_candidate_id"])]
        if (
            _runtime_common_signature(source_copy)
            != _runtime_common_signature(winner)
            or source_copy["native_invocation"]["references"]
            != winner["native_invocation"]["references"]
            or source_copy["native_invocation"]["references"]["mode"] != "correct"
            or not _trusted(source_copy)
            or _failed(source_copy) != ["A"]
        ):
            reasons.append(f"{pair['pair_id']}:bound_source_copy_control_failed")
        eligible = (
            _action_pair_eligible(winner, loser)
            if pair["pair_type"] == "action_donor_nearmiss"
            else _identity_pair_eligible(winner, loser)
        )
        if not eligible:
            reasons.append(f"{pair['pair_id']}:preregistered_causal_pair_failed")
        else:
            qualifications.append(_qualification(pair, winner, loser))
    # No partial training: every externally sealed pair and every control passes,
    # or the entire episode is a null update.
    if reasons or len(qualifications) != len(context.preregistration["causal_pairs"]):
        if any(_failed(candidate) in (["C"], ["Q"]) for candidate in candidates.values()):
            reasons.append("C_or_Q_only_rejections_are_diagnostic_not_identity_routes")
        return _null_receipt(
            episode=episode,
            context=context,
            reasons=reasons or ["no_legal_preregistered_causal_pair"],
            renderer_count=len(candidates),
            measured=measured,
        )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "episode_id": episode["episode_id"],
        "episode_digest": episode["episode_digest"],
        "upstream_pair_qualified": True,
        "status": "upstream_qualification",
        "null_reasons": [],
        "qualifications": qualifications,
        "external_trust_roots": {
            "pass_a_receipt_file_sha256": context.pass_a_receipt_sha256,
            "pass_a_qualification_seal_file_sha256": context.pass_a_qualification_sha256,
            "pass_b_preregistration_seal_file_sha256": context.preregistration_sha256,
        },
        "search_accounting": {
            "pass_a_generation_count": 8,
            "pass_b_candidate_count": len(candidates),
            "pass_b_measured_gpu_seconds": measured,
            "total_model_generations": 8 + len(candidates),
            "topup_observed": False,
        },
        "production_dclr_consumption_authorized": False,
        "bridge_status": "absent_requires_separately_validated_adapter",
    }
    receipt["receipt_digest"] = canonical_object_sha256(receipt)
    return receipt


def validate_selection_receipt(
    receipt: Any,
    episode: Any,
    *,
    context: ExternalContext,
    pass_b_artifact_hashes: Mapping[str, Mapping[str, str]],
) -> Mapping[str, Any]:
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise SelfImaginedParetoContractError("upstream receipt schema differs")
    declared = _sha("upstream receipt digest", receipt.get("receipt_digest"))
    unsigned = dict(receipt)
    unsigned.pop("receipt_digest", None)
    if canonical_object_sha256(unsigned) != declared:
        raise SelfImaginedParetoContractError("upstream receipt digest differs")
    expected = select_upstream_qualification(
        episode,
        context=context,
        pass_b_artifact_hashes=pass_b_artifact_hashes,
    )
    if dict(receipt) != expected:
        raise SelfImaginedParetoContractError(
            "upstream receipt differs from deterministic recomputation"
        )
    return receipt


def to_production_dclr_preference_pair(*_: Any, **__: Any) -> None:
    raise ProductionDCLRBridgeUnavailable(
        "V3 emits externally sealed upstream causal qualifications, not "
        "bernini-dclr-preference-pair-v3; production consumption is forbidden"
    )


__all__ = [
    "CANDIDATE_SCHEMA",
    "COUNTERFACTUAL_ARMS",
    "EPISODE_SCHEMA",
    "EVENT_AXES",
    "ExternalContext",
    "FRAME_COUNT",
    "INVOCATION_SCHEMA",
    "OFFICIAL_GAUSSIAN_SOURCE",
    "PAIR_TYPES",
    "PROPOSAL_BRANCHES",
    "ProductionDCLRBridgeUnavailable",
    "PREREGISTRATION_SEAL_SCHEMA",
    "QUALIFICATION_SEAL_SCHEMA",
    "RECEIPT_SCHEMA",
    "RENDER_GATES",
    "SelfImaginedParetoContractError",
    "build_external_context",
    "build_renderer_prompt",
    "canonical_json_bytes",
    "caption_artifact_sha256",
    "select_upstream_qualification",
    "to_production_dclr_preference_pair",
    "validate_episode",
    "validate_selection_receipt",
]
