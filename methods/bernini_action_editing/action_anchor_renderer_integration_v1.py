#!/usr/bin/env python3
"""Local-only structural bridge from action plans toward the Bernini renderer.

This module is deliberately an integration boundary, not a trainer or a
launcher.  It performs one conditioner predictor forward from detached FP32
source/instruction tensors, retains that FP32 result for distillation, and
derives the renderer-dtype plan from the *same autograd graph*.  It also binds
all training-only receipts to a caller-pinned, closed sidecar envelope.

The sidecar's ``predictor_artifact_sha256`` is a caller-observed
implementation/load-authority identifier, matching the distillation producer
field.  This pure bridge neither reads a checkpoint nor hashes live learned
parameter bytes, so that field must never be reported as a live-weights hash.

V1 deliberately does *not* claim production renderer-flow evidence: it can
authenticate the conditioner target-hidden injection calls, but it is not
wired to Bernini's existing block/post-hook route and it rejects checkpointed
first forwards.  Its preservation connectivity audit is conditioner-only.
Those limitations keep every returned receipt training/optimizer-NO-GO.

Nothing here qualifies a teacher, authorizes training, reads media, launches
work, or changes model state.  It reads only the imported distillation and
action-plan Python sources to verify their hard-pinned artifact hashes.  In particular a
``candidate_unqualified`` materialization cannot authorize itself through the
sidecar: the current distillation contract still validates the externally
qualified per-item evidence and its independently pinned authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

if __package__:
    from . import action_anchor_distillation_v1 as distillation
    from . import action_plan_predictor_v1 as action_plan
else:  # Direct import from methods/bernini_action_editing.
    import action_anchor_distillation_v1 as distillation
    import action_plan_predictor_v1 as action_plan


SCHEMA_VERSION = "bernini-action-anchor-renderer-integration-v1"
SIDECAR_ENVELOPE_SCHEMA = "bernini-action-anchor-renderer-sidecar-v1"
PREPARED_ROUTE_SCHEMA = "bernini-action-anchor-prepared-renderer-route-v1"
COMBINED_LOSS_SCHEMA = (
    "bernini-action-anchor-combined-structural-probe-loss-v1"
)
RENDERER_FLOW_ARTIFACT_SCHEMA = (
    "bernini-action-anchor-structural-renderer-probe-artifact-v1"
)
RENDERER_FLOW_BACKWARD_RECEIPT_SCHEMA = (
    "bernini-action-anchor-structural-renderer-probe-backward-receipt-v1"
)
EVALUATION_ROUTE_SCHEMA = "bernini-action-anchor-evaluation-routes-v1"

LOCAL_ONLY = True
NO_TRAINING = True
NO_LAUNCH = True
IMPLEMENTS_TEACHER_QUALIFICATION = False
IMPLEMENTS_RENDERER = False
GRADIENT_CHECKPOINTING_SUPPORTED = False
CONDITIONER_REGULARIZER_ONLY = True
PRODUCTION_RENDERER_PRESERVATION_ARTIFACT_IMPLEMENTED = False
STRUCTURAL_ROUTE_EVIDENCE_ONLY = True
REAL_RENDERER_FLOW_AUTHORIZED = False
PRODUCTION_BLOCK_POST_HOOK_ROUTE_REQUIRED = True

# These pins deliberately make an update of either independently reviewed
# dependency boundary an explicit integration-version event.  Merely
# importing another module that exposes a self-consistent CONTRACT_SHA256 is
# not sufficient.
PINNED_DISTILLATION_CONTRACT_SHA256 = (
    "6e2159102c712c57b35037679eaf31768eebaa2554ef0efe0ac1553fecca8a5b"
)
PINNED_DISTILLATION_MODULE_SOURCE_SHA256 = (
    "6f0b0a762b53b13cc20176a29f6db2ad53efbab5cb9a9dd06778897aa758ce64"
)
PINNED_ACTION_PLAN_MODULE_SOURCE_SHA256 = (
    "464cd500f0ba1edb6cbe6d4f07287bfff346ae0ba7968c0d7c7f3cc7cb667308"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SIDECAR_FIELDS = {
    "schema_version",
    "local_only",
    "training_authorized",
    "launch_authorized",
    "teacher_authorization_origin",
    "candidate_teacher_may_self_authorize",
    "predictor_artifact_sha256",
    "predictor_artifact_hash_semantics",
    "conditioner_state_abi_sha256",
    "action_plan_module_source_sha256",
    "distillation_contract_sha256",
    "distillation_module_source_sha256",
    "teacher_authority_sha256",
    "classification_authority_sha256",
    "batch_size",
    "row_ids",
    "source_token_tensor_sha256",
    "instruction_token_tensor_sha256",
    "q_y_receipt_digest",
    "q_y_qualification_receipt_digests",
    "q_anchor_receipt_digests",
    "q_anchor_qualification_receipt_digests",
    "compatibility_decision_receipt_digests",
    "envelope_digest",
}
_TEACHER_AUTHORIZATION_ORIGIN = "external-qualified-receipts-only"
_PREDICTOR_ARTIFACT_HASH_SEMANTICS = (
    "caller-observed-implementation-or-load-authority-not-live-parameter-hash"
)
_EVALUATION_ARM_NAMES = ("correct", "zero", "shuffled", "reverse")


class ActionAnchorRendererIntegrationError(RuntimeError):
    """Raised before an unpinned or ambiguous route can affect a loss."""


def _make_factory_issuance_registry_v1() -> tuple[Any, ...]:
    """Create a closure-held, one-shot exact-identity issuance ledger.

    Construction nonces are deliberately *not* authority: they remain visible
    on a prepared recorder and only make accidental direct construction fail
    early.  The supported builders stage the exact object identity before it
    can be returned, and recorder registration can only consume that already
    staged identity.  Public recorder lifecycle calls therefore cannot mint an
    artifact or combined objective merely by reproducing all field values.
    """

    artifact_records: dict[Any, tuple[str, Any]] = {}
    combined_records: dict[Any, tuple[str, Any]] = {}

    def stage(
        records: dict[Any, tuple[str, Any]],
        recorder: Any,
        value: Any,
        *,
        label: str,
    ) -> None:
        if recorder in records:
            raise ActionAnchorRendererIntegrationError(
                f"{label} factory issuance is repeated"
            )
        records[recorder] = ("staged", value)

    def require_staged(
        records: dict[Any, tuple[str, Any]],
        recorder: Any,
        value: Any,
        *,
        label: str,
    ) -> None:
        record = records.get(recorder)
        if record is None or record[0] != "staged" or record[1] is not value:
            raise ActionAnchorRendererIntegrationError(
                f"{label} was not staged by its supported factory"
            )

    def consume(
        records: dict[Any, tuple[str, Any]],
        recorder: Any,
        value: Any,
        *,
        label: str,
    ) -> None:
        require_staged(records, recorder, value, label=label)
        records[recorder] = ("issued", value)

    def verify(
        records: dict[Any, tuple[str, Any]],
        recorder: Any,
        value: Any,
        *,
        label: str,
    ) -> None:
        record = records.get(recorder)
        if record is None or record[0] != "issued" or record[1] is not value:
            raise ActionAnchorRendererIntegrationError(
                f"{label} lacks exact factory issuance"
            )

    def stage_artifact(recorder: Any, value: Any) -> None:
        stage(
            artifact_records,
            recorder,
            value,
            label="structural flow artifact",
        )

    def require_staged_artifact(recorder: Any, value: Any) -> None:
        require_staged(
            artifact_records,
            recorder,
            value,
            label="structural flow artifact",
        )

    def consume_artifact(recorder: Any, value: Any) -> None:
        consume(
            artifact_records,
            recorder,
            value,
            label="structural flow artifact",
        )

    def verify_artifact(recorder: Any, value: Any) -> None:
        verify(
            artifact_records,
            recorder,
            value,
            label="structural flow artifact",
        )

    def stage_combined(recorder: Any, value: Any) -> None:
        artifact_record = artifact_records.get(recorder)
        if artifact_record is None or artifact_record[0] != "issued":
            raise ActionAnchorRendererIntegrationError(
                "combined structural loss cannot precede artifact issuance"
            )
        stage(
            combined_records,
            recorder,
            value,
            label="combined structural loss",
        )

    def require_staged_combined(recorder: Any, value: Any) -> None:
        require_staged(
            combined_records,
            recorder,
            value,
            label="combined structural loss",
        )

    def consume_combined(recorder: Any, value: Any) -> None:
        consume(
            combined_records,
            recorder,
            value,
            label="combined structural loss",
        )

    def verify_combined(recorder: Any, value: Any) -> None:
        verify(
            combined_records,
            recorder,
            value,
            label="combined structural loss",
        )

    def revoke(recorder: Any) -> None:
        artifact_records.pop(recorder, None)
        combined_records.pop(recorder, None)

    def retire(recorder: Any, artifact: Any, combined: Any) -> None:
        verify_artifact(recorder, artifact)
        verify_combined(recorder, combined)
        revoke(recorder)

    return (
        stage_artifact,
        require_staged_artifact,
        consume_artifact,
        verify_artifact,
        stage_combined,
        require_staged_combined,
        consume_combined,
        verify_combined,
        revoke,
        retire,
    )


(
    _stage_factory_artifact_issuance_v1,
    _require_staged_factory_artifact_v1,
    _consume_factory_artifact_issuance_v1,
    _verify_factory_artifact_issuance_v1,
    _stage_factory_combined_issuance_v1,
    _require_staged_factory_combined_v1,
    _consume_factory_combined_issuance_v1,
    _verify_factory_combined_issuance_v1,
    _revoke_factory_issuance_v1,
    _retire_factory_issuance_v1,
) = _make_factory_issuance_registry_v1()
del _make_factory_issuance_registry_v1


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ActionAnchorRendererIntegrationError(
            f"value is not closed finite canonical JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ActionAnchorRendererIntegrationError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _authority_sha256(value: Any, *, label: str) -> str:
    digest = _sha256(value, label=label)
    if digest == "0" * 64:
        raise ActionAnchorRendererIntegrationError(
            f"{label} must be a non-zero SHA-256 authority pin"
        )
    return digest


def _module_source_sha256(module: Any, *, label: str) -> str:
    module_path = getattr(module, "__file__", None)
    if type(module_path) is not str or not module_path.endswith(".py"):
        raise ActionAnchorRendererIntegrationError(
            f"{label} has no auditable Python source artifact"
        )
    try:
        with open(module_path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError as error:
        raise ActionAnchorRendererIntegrationError(
            f"cannot verify {label} source artifact: {error}"
        ) from error


def _verify_pinned_dependencies() -> None:
    """Verify semantic and exact-source pins for both imported boundaries."""

    if distillation.CONTRACT_SHA256 != PINNED_DISTILLATION_CONTRACT_SHA256:
        raise ActionAnchorRendererIntegrationError(
            "imported distillation contract differs from the V1 hard pin"
        )
    if _module_source_sha256(
        distillation, label="distillation module"
    ) != PINNED_DISTILLATION_MODULE_SOURCE_SHA256:
        raise ActionAnchorRendererIntegrationError(
            "distillation module source differs from the V1 hard pin"
        )
    if _module_source_sha256(
        action_plan, label="action-plan module"
    ) != PINNED_ACTION_PLAN_MODULE_SOURCE_SHA256:
        raise ActionAnchorRendererIntegrationError(
            "action-plan module source differs from the V1 hard pin"
        )


def _closed_dict(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ActionAnchorRendererIntegrationError(f"{label} must be an exact dict")
    if set(value) != fields:
        raise ActionAnchorRendererIntegrationError(
            f"{label} field closure differs: "
            f"missing={sorted(fields - set(value))} extra={sorted(set(value) - fields)}"
        )
    return value


def _exact_sha_list(
    value: Any, *, label: str, identity_leaves: bool = False
) -> list[str]:
    if type(value) is not list:
        raise ActionAnchorRendererIntegrationError(f"{label} must be an exact list")
    validator = _authority_sha256 if identity_leaves else _sha256
    return [validator(item, label=f"{label}[{index}]") for index, item in enumerate(value)]


def _exact_nested_sha_list(
    value: Any, *, label: str, identity_leaves: bool = False
) -> list[list[str]]:
    if type(value) is not list:
        raise ActionAnchorRendererIntegrationError(f"{label} must be an exact list")
    return [
        _exact_sha_list(
            item,
            label=f"{label}[{index}]",
            identity_leaves=identity_leaves,
        )
        for index, item in enumerate(value)
    ]


def build_sidecar_envelope_v1(
    *,
    predictor_artifact_sha256: str,
    conditioner_state_abi_sha256: str,
    source_token_tensor_sha256: str,
    instruction_token_tensor_sha256: str,
    teacher_authority_sha256: str,
    classification_authority_sha256: str,
    row_ids: Sequence[str],
    q_y_receipt_digest: str,
    q_y_qualification_receipt_digests: Sequence[str],
    q_anchor_receipt_digests: Sequence[str] = (),
    q_anchor_qualification_receipt_digests: Sequence[Sequence[str]] = (),
    compatibility_decision_receipt_digests: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a closed local-review sidecar; it does not authorize training."""

    _verify_pinned_dependencies()

    if type(row_ids) not in (list, tuple) or not row_ids:
        raise ActionAnchorRendererIntegrationError(
            "sidecar row IDs must be one non-empty exact list or tuple"
        )
    if type(q_y_qualification_receipt_digests) not in (list, tuple):
        raise ActionAnchorRendererIntegrationError(
            "q_y qualification pins must be an exact list or tuple"
        )
    if type(q_anchor_receipt_digests) not in (list, tuple) or type(
        q_anchor_qualification_receipt_digests
    ) not in (list, tuple) or type(compatibility_decision_receipt_digests) not in (
        list,
        tuple,
    ):
        raise ActionAnchorRendererIntegrationError(
            "anchor authority pins must be exact lists or tuples"
        )
    unsigned = {
        "schema_version": SIDECAR_ENVELOPE_SCHEMA,
        "local_only": True,
        "training_authorized": False,
        "launch_authorized": False,
        "teacher_authorization_origin": _TEACHER_AUTHORIZATION_ORIGIN,
        "candidate_teacher_may_self_authorize": False,
        "predictor_artifact_sha256": _authority_sha256(
            predictor_artifact_sha256, label="predictor artifact SHA-256"
        ),
        "predictor_artifact_hash_semantics": (
            _PREDICTOR_ARTIFACT_HASH_SEMANTICS
        ),
        "conditioner_state_abi_sha256": _authority_sha256(
            conditioner_state_abi_sha256, label="conditioner state ABI SHA-256"
        ),
        "action_plan_module_source_sha256": (
            PINNED_ACTION_PLAN_MODULE_SOURCE_SHA256
        ),
        "distillation_contract_sha256": PINNED_DISTILLATION_CONTRACT_SHA256,
        "distillation_module_source_sha256": (
            PINNED_DISTILLATION_MODULE_SOURCE_SHA256
        ),
        "teacher_authority_sha256": _authority_sha256(
            teacher_authority_sha256, label="teacher authority SHA-256"
        ),
        "classification_authority_sha256": _authority_sha256(
            classification_authority_sha256,
            label="classification authority SHA-256",
        ),
        "batch_size": len(row_ids),
        "row_ids": [
            _authority_sha256(value, label=f"sidecar row ID[{index}]")
            for index, value in enumerate(row_ids)
        ],
        "source_token_tensor_sha256": _authority_sha256(
            source_token_tensor_sha256,
            label="externally pinned source-token tensor SHA-256",
        ),
        "instruction_token_tensor_sha256": _authority_sha256(
            instruction_token_tensor_sha256,
            label="externally pinned instruction-token tensor SHA-256",
        ),
        "q_y_receipt_digest": _authority_sha256(
            q_y_receipt_digest, label="q_y receipt digest"
        ),
        "q_y_qualification_receipt_digests": [
            _authority_sha256(value, label=f"q_y qualification pin[{index}]")
            for index, value in enumerate(q_y_qualification_receipt_digests)
        ],
        "q_anchor_receipt_digests": [
            _authority_sha256(value, label=f"q_anchor receipt pin[{index}]")
            for index, value in enumerate(q_anchor_receipt_digests)
        ],
        "q_anchor_qualification_receipt_digests": [
            [
                _authority_sha256(
                    value,
                    label=f"q_anchor[{anchor_index}] qualification[{row_index}]",
                )
                for row_index, value in enumerate(per_anchor)
            ]
            for anchor_index, per_anchor in enumerate(
                q_anchor_qualification_receipt_digests
            )
        ],
        "compatibility_decision_receipt_digests": [
            _authority_sha256(
                value, label=f"compatibility decision pin[{index}]"
            )
            for index, value in enumerate(compatibility_decision_receipt_digests)
        ],
    }
    if len(unsigned["q_y_qualification_receipt_digests"]) != len(row_ids):
        raise ActionAnchorRendererIntegrationError(
            "q_y qualification pin coverage differs from sidecar batch"
        )
    if (
        unsigned["source_token_tensor_sha256"]
        == unsigned["instruction_token_tensor_sha256"]
    ):
        raise ActionAnchorRendererIntegrationError(
            "source/instruction tensor identity pins must be distinct"
        )
    if len(set(unsigned["row_ids"])) != len(row_ids) or len(
        set(unsigned["q_y_qualification_receipt_digests"])
    ) != len(row_ids):
        raise ActionAnchorRendererIntegrationError(
            "sidecar row IDs and q_y qualification leaves must be unique within batch"
        )
    anchor_count = len(unsigned["q_anchor_receipt_digests"])
    if (
        len(unsigned["q_anchor_qualification_receipt_digests"]) != anchor_count
        or len(unsigned["compatibility_decision_receipt_digests"]) != anchor_count
        or any(
            len(per_anchor) != len(row_ids)
            for per_anchor in unsigned[
                "q_anchor_qualification_receipt_digests"
            ]
        )
        or len(set(unsigned["q_anchor_receipt_digests"])) != anchor_count
        or len(set(unsigned["compatibility_decision_receipt_digests"]))
        != anchor_count
        or any(
            len(set(per_anchor)) != len(row_ids)
            for per_anchor in unsigned[
                "q_anchor_qualification_receipt_digests"
            ]
        )
    ):
        raise ActionAnchorRendererIntegrationError(
            "anchor receipt/qualification/decision pin coverage differs"
        )
    identity_leaves = (
        unsigned["row_ids"]
        + [unsigned["q_y_receipt_digest"]]
        + unsigned["q_y_qualification_receipt_digests"]
        + unsigned["q_anchor_receipt_digests"]
        + [
            leaf
            for per_anchor in unsigned[
                "q_anchor_qualification_receipt_digests"
            ]
            for leaf in per_anchor
        ]
        + unsigned["compatibility_decision_receipt_digests"]
    )
    if len(identity_leaves) != len(set(identity_leaves)):
        raise ActionAnchorRendererIntegrationError(
            "sidecar row/receipt/qualification/decision identity leaves overlap"
        )
    return {**unsigned, "envelope_digest": object_sha256(unsigned)}


def validate_sidecar_envelope_v1(
    value: Any,
    *,
    expected_envelope_digest: str,
    expected_teacher_authority_sha256: str,
    expected_classification_authority_sha256: str,
) -> dict[str, Any]:
    """Validate the sidecar against pins supplied outside the sidecar itself."""

    _verify_pinned_dependencies()

    envelope = _closed_dict(value, _SIDECAR_FIELDS, label="renderer sidecar")
    expected_digest = _authority_sha256(
        expected_envelope_digest, label="externally expected sidecar digest"
    )
    expected_teacher = _authority_sha256(
        expected_teacher_authority_sha256,
        label="externally expected teacher authority SHA-256",
    )
    expected_classification = _authority_sha256(
        expected_classification_authority_sha256,
        label="externally expected classification authority SHA-256",
    )
    if (
        envelope["schema_version"] != SIDECAR_ENVELOPE_SCHEMA
        or envelope["local_only"] is not True
        or envelope["training_authorized"] is not False
        or envelope["launch_authorized"] is not False
        or envelope["teacher_authorization_origin"]
        != _TEACHER_AUTHORIZATION_ORIGIN
        or envelope["candidate_teacher_may_self_authorize"] is not False
        or envelope["predictor_artifact_hash_semantics"]
        != _PREDICTOR_ARTIFACT_HASH_SEMANTICS
        or envelope["distillation_contract_sha256"]
        != PINNED_DISTILLATION_CONTRACT_SHA256
        or envelope["distillation_module_source_sha256"]
        != PINNED_DISTILLATION_MODULE_SOURCE_SHA256
        or envelope["action_plan_module_source_sha256"]
        != PINNED_ACTION_PLAN_MODULE_SOURCE_SHA256
    ):
        raise ActionAnchorRendererIntegrationError(
            "sidecar safety/contract semantics differ"
        )
    for name in (
        "predictor_artifact_sha256",
        "conditioner_state_abi_sha256",
        "source_token_tensor_sha256",
        "instruction_token_tensor_sha256",
        "q_y_receipt_digest",
        "envelope_digest",
    ):
        _authority_sha256(envelope[name], label=f"sidecar {name}")
    if (
        envelope["source_token_tensor_sha256"]
        == envelope["instruction_token_tensor_sha256"]
    ):
        raise ActionAnchorRendererIntegrationError(
            "sidecar source/instruction tensor identity pins must be distinct"
        )
    _authority_sha256(
        envelope["teacher_authority_sha256"],
        label="sidecar teacher authority SHA-256",
    )
    _authority_sha256(
        envelope["classification_authority_sha256"],
        label="sidecar classification authority SHA-256",
    )
    if (
        envelope["teacher_authority_sha256"] != expected_teacher
        or envelope["classification_authority_sha256"]
        != expected_classification
    ):
        raise ActionAnchorRendererIntegrationError(
            "sidecar teacher/classification authority is not externally pinned"
        )
    if type(envelope["batch_size"]) is not int or envelope["batch_size"] <= 0:
        raise ActionAnchorRendererIntegrationError(
            "sidecar batch size must be a positive exact integer"
        )
    row_ids = _exact_sha_list(
        envelope["row_ids"], label="sidecar row IDs", identity_leaves=True
    )
    q_y_pins = _exact_sha_list(
        envelope["q_y_qualification_receipt_digests"],
        label="sidecar q_y qualification pins",
        identity_leaves=True,
    )
    anchor_receipts = _exact_sha_list(
        envelope["q_anchor_receipt_digests"],
        label="sidecar q_anchor receipt pins",
        identity_leaves=True,
    )
    anchor_qualifications = _exact_nested_sha_list(
        envelope["q_anchor_qualification_receipt_digests"],
        label="sidecar q_anchor qualification pins",
        identity_leaves=True,
    )
    decisions = _exact_sha_list(
        envelope["compatibility_decision_receipt_digests"],
        label="sidecar compatibility decision pins",
        identity_leaves=True,
    )
    batch_size = envelope["batch_size"]
    if len(row_ids) != batch_size or len(q_y_pins) != batch_size:
        raise ActionAnchorRendererIntegrationError(
            "sidecar row/q_y qualification coverage differs from batch"
        )
    if len(set(row_ids)) != batch_size or len(set(q_y_pins)) != batch_size:
        raise ActionAnchorRendererIntegrationError(
            "sidecar row IDs and q_y qualification leaves must be unique within batch"
        )
    if (
        len(anchor_receipts) != len(anchor_qualifications)
        or len(anchor_receipts) != len(decisions)
        or any(len(per_anchor) != batch_size for per_anchor in anchor_qualifications)
        or len(set(anchor_receipts)) != len(anchor_receipts)
        or len(set(decisions)) != len(decisions)
        or any(
            len(set(per_anchor)) != batch_size
            for per_anchor in anchor_qualifications
        )
    ):
        raise ActionAnchorRendererIntegrationError(
            "sidecar anchor authority coverage differs"
        )
    identity_leaves = (
        row_ids
        + [_authority_sha256(
            envelope["q_y_receipt_digest"], label="sidecar q_y receipt digest"
        )]
        + q_y_pins
        + anchor_receipts
        + [leaf for per_anchor in anchor_qualifications for leaf in per_anchor]
        + decisions
    )
    if len(identity_leaves) != len(set(identity_leaves)):
        raise ActionAnchorRendererIntegrationError(
            "sidecar row/receipt/qualification/decision identity leaves overlap"
        )
    declared = envelope["envelope_digest"]
    unsigned = dict(envelope)
    del unsigned["envelope_digest"]
    if object_sha256(unsigned) != declared or declared != expected_digest:
        raise ActionAnchorRendererIntegrationError(
            "sidecar digest is not externally pinned"
        )
    # Never return caller-owned nested containers.
    return json.loads(canonical_json_bytes(envelope).decode("ascii"))


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - host dependent
        raise ActionAnchorRendererIntegrationError(
            "PyTorch is required for renderer integration operations"
        ) from error
    return torch


def _require_plan(
    value: Any,
    *,
    label: str,
    batch_size: int,
    dtype: Any,
    device: Any,
) -> action_plan.ActionPlanOutput:
    torch = _torch()
    if type(value) is not action_plan.ActionPlanOutput:
        raise ActionAnchorRendererIntegrationError(
            f"{label} must be the exact ActionPlanOutput type"
        )
    phase = value.phase_tokens
    global_token = value.global_token
    if (
        type(phase) is not torch.Tensor
        or type(global_token) is not torch.Tensor
        or phase.dtype != dtype
        or global_token.dtype != dtype
        or phase.device != device
        or global_token.device != device
        or not phase.is_contiguous()
        or not global_token.is_contiguous()
        or tuple(phase.shape)
        != (batch_size, action_plan.PHASE_COUNT, action_plan.ACTION_WIDTH)
        or tuple(global_token.shape)
        != (batch_size, action_plan.ACTION_WIDTH)
        or not bool(torch.isfinite(phase.detach()).all().item())
        or not bool(torch.isfinite(global_token.detach()).all().item())
    ):
        raise ActionAnchorRendererIntegrationError(
            f"{label} shape/dtype/device/finite ABI differs"
        )
    return value


def _cast_plan_with_graph(
    value: action_plan.ActionPlanOutput, *, dtype: Any
) -> action_plan.ActionPlanOutput:
    return action_plan.ActionPlanOutput(
        phase_tokens=value.phase_tokens.to(dtype=dtype).contiguous(),
        global_token=value.global_token.to(dtype=dtype).contiguous(),
    )


def tensor_sha256_v1(value: Any) -> str:
    """Hash an exact finite strided tensor, including shape and bit dtype."""

    torch = _torch()
    if (
        type(value) is not torch.Tensor
        or value.layout != torch.strided
        or not value.is_floating_point()
        or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or value.numel() <= 0
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise ActionAnchorRendererIntegrationError(
            "tensor digest input must be finite strided FP16/BF16/FP32"
        )
    cpu = value.detach().cpu().contiguous()
    dtype_name = str(cpu.dtype)
    if dtype_name.startswith("torch."):
        dtype_name = dtype_name[len("torch.") :]
    header = {
        "schema_version": "bernini-action-anchor-exact-tensor-sha256-v1",
        "dtype": dtype_name,
        "shape": [int(item) for item in cpu.shape],
    }
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(header))
    digest.update(b"\x00")
    # PyTorch versions used with Python 3.8 reject dtype-view on a 0-D tensor.
    # Flattening first preserves the exact bytes while keeping the original
    # scalar shape in the authenticated header above.
    payload = cpu.reshape(-1).view(torch.uint8).reshape(-1)
    for start in range(0, int(payload.numel()), 1 << 20):
        digest.update(bytes(payload[start : start + (1 << 20)].tolist()))
    return digest.hexdigest()


def _contiguous_storage_span(value: Any) -> tuple[str, int, int, int]:
    if not value.is_contiguous():
        raise ActionAnchorRendererIntegrationError(
            "renderer flow tensors must be contiguous for exact storage ownership"
        )
    element_size = int(value.element_size())
    start = int(value.storage_offset()) * element_size
    stop = start + int(value.numel()) * element_size
    return (
        str(value.device),
        int(value.untyped_storage().data_ptr()),
        start,
        stop,
    )


def _spans_overlap(
    left: tuple[str, int, int, int], right: tuple[str, int, int, int]
) -> bool:
    return (
        left[0] == right[0]
        and left[1] == right[1]
        and max(left[2], right[2]) < min(left[3], right[3])
    )


class _RouteUseRecorderV1:
    """One-shot, fail-closed witness over the exact 30 injection projections."""

    def __init__(
        self,
        *,
        conditioner: Any,
        renderer_plan: Any,
        route: Any,
        initial_target_hidden: Any,
    ):
        torch = _torch()
        projections = tuple(conditioner.injection.projections)
        if (
            len(projections) != action_plan.TRANSFORMER_BLOCK_COUNT
            or len({id(module) for module in projections}) != len(projections)
        ):
            raise ActionAnchorRendererIntegrationError(
                "route recorder requires 30 distinct injection projections"
            )
        global_by_phase = renderer_plan.global_token.unsqueeze(1).expand(
            -1, action_plan.PHASE_COUNT, -1
        )
        self.expected_condition = torch.cat(
            (renderer_plan.phase_tokens.float(), global_by_phase.float()), dim=-1
        )
        self.projections = projections
        self.injection = conditioner.injection
        self.route = route
        self.initial_target_hidden = initial_target_hidden
        self.allowed_trainables = tuple(
            parameter
            for parameter in conditioner.parameters()
            if parameter.requires_grad
        )
        self.predictor_trainables = tuple(
            parameter
            for parameter in conditioner.predictor.parameters()
            if parameter.requires_grad
        )
        self.projection_parameters = tuple(
            (projection.weight, projection.bias) for projection in projections
        )
        if not self.allowed_trainables:
            raise ActionAnchorRendererIntegrationError(
                "conditioner exposes no allowed trainables"
            )
        self.indices: list[int] = []
        self.injection_indices: list[int] = []
        self.inputs: list[Any] = []
        self.outputs: list[Any] = []
        self.target_hidden_inputs: list[Any] = []
        self.injection_outputs: list[Any] = []
        self._active_injection_index: int | None = None
        self.forward_handles: list[Any] = []
        self.gradient_handles: list[Any] = []
        self.output_gradient_counts = [0] * len(projections)
        self.output_gradient_finite = [False] * len(projections)
        self.output_gradient_nonzero = [False] * len(projections)
        self.injection_gradient_counts = [0] * len(projections)
        self.injection_gradient_finite = [False] * len(projections)
        self.injection_gradient_nonzero = [False] * len(projections)
        self.plan_gradient_counts = {"phase": 0, "global": 0}
        self.plan_gradient_finite = {"phase": False, "global": False}
        self._artifact_nonce = object()
        self._combined_nonce = object()
        self._registered_artifact: Any = None
        self._registered_combined: Any = None
        self._registered_flow_tensor: Any = None
        self._registered_artifact_tensors: Any = None
        self._registered_total_tensor: Any = None
        self._registered_distillation: Any = None
        self._registered_distillation_tensors: Any = None
        self._registered_q_y: Any = None
        self._registered_q_y_receipt: Any = None
        self._registered_anchors: Any = None
        self._registered_conditioner_regularizer: Any = None
        self._registered_config: Any = None
        self._artifact_snapshot: dict[str, Any] | None = None
        self._combined_snapshot: dict[str, Any] | None = None
        self._internal_vjp_return_state: str | None = None
        self._flow_dependency_audited = False
        self._combined_dependency_audited = False
        self.state = "recording"
        try:
            self.forward_handles.append(
                self.injection.register_forward_pre_hook(
                    self._injection_pre_hook, with_kwargs=True
                )
            )
            self.forward_handles.append(
                self.injection.register_forward_hook(
                    self._injection_forward_hook, with_kwargs=True
                )
            )
            for index, projection in enumerate(projections):
                self.forward_handles.append(
                    projection.register_forward_hook(self._make_forward_hook(index))
                )
        except Exception:
            self._remove_forward_hooks()
            self.state = "failed"
            raise

    def _remove_forward_hooks(self) -> None:
        for handle in self.forward_handles:
            handle.remove()
        self.forward_handles.clear()

    def _remove_gradient_hooks(self) -> None:
        for handle in self.gradient_handles:
            handle.remove()
        self.gradient_handles.clear()

    def abort(self) -> None:
        self._remove_forward_hooks()
        self._remove_gradient_hooks()
        _revoke_factory_issuance_v1(self)
        self.state = "failed"

    def register_artifact(self, value: Any) -> None:
        if (
            self.state != "artifact_armed"
            or self._registered_artifact is not None
            or getattr(value, "_construction_nonce", None)
            is not self._artifact_nonce
        ):
            self._fail("structural flow artifact registration is forged or repeated")
        try:
            _require_staged_factory_artifact_v1(self, value)
            _revalidate_flow_artifact_v1(
                value._prepared,
                value,
                allowed_states=("artifact_armed",),
                require_registered=False,
            )
            _consume_factory_artifact_issuance_v1(self, value)
            self._registered_artifact = value
            self._registered_flow_tensor = value.flow
            self._registered_artifact_tensors = (
                value.flow,
                value.prediction,
                value.target_clean,
                value.noise,
                value.target_velocity,
            )
            self._artifact_snapshot = _snapshot_structural_flow_artifact_v1(
                value
            )
        except Exception:
            self.abort()
            raise

    def register_combined(self, value: Any) -> None:
        if (
            self.state != "artifact_armed"
            or self._registered_combined is not None
            or getattr(value, "_construction_nonce", None)
            is not self._combined_nonce
        ):
            self._fail("combined structural loss registration is forged or repeated")
        try:
            _require_staged_factory_combined_v1(self, value)
            _revalidate_combined_loss_v1(
                value,
                allowed_states=("artifact_armed",),
                require_registered=False,
            )
            _consume_factory_combined_issuance_v1(self, value)
            self._registered_combined = value
            self._registered_total_tensor = value.total
            self._registered_distillation = value.distillation
            self._registered_distillation_tensors = tuple(
                getattr(value.distillation, name)
                for name in (
                    "total",
                    "smooth_l1",
                    "cosine",
                    "infonce",
                    "preservation",
                )
            )
            self._registered_q_y = value._q_y
            self._registered_q_y_receipt = value._q_y_receipt
            self._registered_anchors = value._anchors
            self._registered_conditioner_regularizer = (
                value._conditioner_regularizer_loss
            )
            self._registered_config = value._config
            self._combined_snapshot = _snapshot_combined_loss_v1(value)
        except Exception:
            self.abort()
            raise

    def _fail(self, message: str) -> None:
        self.abort()
        raise ActionAnchorRendererIntegrationError(message)

    def _make_forward_hook(self, index: int) -> Any:
        def record(_module: Any, inputs: Any, output: Any) -> None:
            torch = _torch()
            if self.state != "recording":
                self._fail("injection projection called outside its one-shot lease")
            expected_index = len(self.indices)
            if expected_index >= len(self.projections) or index != expected_index:
                self._fail(
                    "injection traversal must call each projection exactly once in 0..29 order"
                )
            if type(inputs) is not tuple or len(inputs) != 1:
                self._fail("injection projection input arity differs")
            condition = inputs[0]
            if (
                type(condition) is not torch.Tensor
                or type(output) is not torch.Tensor
                or condition.dtype != torch.float32
                or condition.device != self.expected_condition.device
                or tuple(condition.shape) != tuple(self.expected_condition.shape)
                or not bool(torch.isfinite(condition.detach()).all().item())
                or not bool(torch.equal(
                    condition.detach(), self.expected_condition.detach()
                ))
            ):
                self._fail(
                    "injection projection input differs from the prepared renderer plan"
                )
            if (
                not output.is_floating_point()
                or output.device != condition.device
                or tuple(output.shape[:-1]) != tuple(condition.shape[:-1])
                or not bool(torch.isfinite(output.detach()).all().item())
            ):
                self._fail("injection projection output ABI differs")
            self.indices.append(index)
            self.inputs.append(condition)
            self.outputs.append(output)

        return record

    def _injection_pre_hook(
        self, _module: Any, args: Any, kwargs: Any
    ) -> None:
        torch = _torch()
        if self.state != "recording":
            self._fail("target-hidden injection called outside its one-shot lease")
        expected_index = len(self.injection_indices)
        if (
            expected_index >= action_plan.TRANSFORMER_BLOCK_COUNT
            or type(args) is not tuple
            or len(args) != 2
            or type(kwargs) is not dict
            or set(kwargs) != {"block_index"}
            or type(kwargs["block_index"]) is not int
            or kwargs["block_index"] != expected_index
            or args[1] is not self.route
            or self._active_injection_index is not None
        ):
            self._fail(
                "target-hidden injection must use the prepared route exactly once in 0..29 order"
            )
        target_hidden = args[0]
        if (
            type(target_hidden) is not torch.Tensor
            or not target_hidden.is_floating_point()
            or tuple(target_hidden.shape)
            != tuple(self.route.ownership.target_shape)
            or target_hidden.dtype != self.route.plan.phase_tokens.dtype
            or target_hidden.device != self.route.plan.phase_tokens.device
            or not bool(torch.isfinite(target_hidden.detach()).all().item())
            or (expected_index == 0 and target_hidden is not self.initial_target_hidden)
        ):
            self._fail("target-hidden injection ownership differs from prepare")
        self.injection_indices.append(expected_index)
        self.target_hidden_inputs.append(target_hidden)
        self._active_injection_index = expected_index

    def _injection_forward_hook(
        self, _module: Any, _args: Any, _kwargs: Any, output: Any
    ) -> None:
        torch = _torch()
        index = self._active_injection_index
        if (
            self.state != "recording"
            or index is None
            or len(self.indices) != index + 1
            or type(output) is not torch.Tensor
            or tuple(output.shape)
            != tuple(self.route.ownership.target_shape)
            or output.dtype != self.route.plan.phase_tokens.dtype
            or output.device != self.route.plan.phase_tokens.device
            or not bool(torch.isfinite(output.detach()).all().item())
        ):
            self._fail("target-hidden injection output/traversal differs")
        self.injection_outputs.append(output)
        self._active_injection_index = None

    def seal_forward(self) -> None:
        torch = _torch()
        expected = list(range(action_plan.TRANSFORMER_BLOCK_COUNT))
        if (
            self.state != "recording"
            or self.indices != expected
            or self.injection_indices != expected
            or len(self.injection_outputs) != len(expected)
            or self._active_injection_index is not None
        ):
            self._fail(
                "renderer route did not traverse exactly 30 projections in 0..29 order"
            )
        if any(
            type(output) is not torch.Tensor or not output.requires_grad
            for output in self.outputs + self.injection_outputs
        ):
            self._fail(
                "checkpoint/no-grad projection output is unsupported by renderer integration V1"
            )
        # Remove module hooks before any dependency VJP.  A checkpoint
        # recomputation therefore cannot impersonate an additional forward.
        self._remove_forward_hooks()
        self.state = "forward_sealed"

    def arm_backward(self, *, renderer_plan: Any) -> None:
        if self.state != "artifact_ready":
            self._fail("renderer flow artifact is not ready for backward arming")

        def output_hook(index: int) -> Any:
            def observe(gradient: Any) -> Any:
                torch = _torch()
                if self.state == "internal_vjp":
                    return gradient
                if self.state != "backward_active":
                    self._fail(
                        "renderer flow graph must be consumed through CombinedActionAnchorLossV1.backward"
                    )
                self.output_gradient_counts[index] += 1
                if self.output_gradient_counts[index] != 1:
                    self._fail("an injection output received duplicate backward traversal")
                finite = type(gradient) is torch.Tensor and bool(
                    torch.isfinite(gradient.detach()).all().item()
                )
                nonzero = finite and bool(
                    torch.count_nonzero(gradient.detach()).item() > 0
                )
                self.output_gradient_finite[index] = finite
                self.output_gradient_nonzero[index] = nonzero
                return gradient

            return observe

        def injection_hook(index: int) -> Any:
            def observe(gradient: Any) -> Any:
                torch = _torch()
                if self.state == "internal_vjp":
                    return gradient
                if self.state != "backward_active":
                    self._fail(
                        "target-hidden injection graph must use the combined backward"
                    )
                self.injection_gradient_counts[index] += 1
                if self.injection_gradient_counts[index] != 1:
                    self._fail("an injection output received duplicate gradient")
                finite = type(gradient) is torch.Tensor and bool(
                    torch.isfinite(gradient.detach()).all().item()
                )
                self.injection_gradient_finite[index] = finite
                self.injection_gradient_nonzero[index] = finite and bool(
                    torch.count_nonzero(gradient.detach()).item() > 0
                )
                return gradient

            return observe

        def plan_hook(name: str) -> Any:
            def observe(gradient: Any) -> Any:
                torch = _torch()
                if self.state == "internal_vjp":
                    return gradient
                if self.state != "backward_active":
                    self._fail(
                        "renderer plan graph must be consumed through the combined backward"
                    )
                self.plan_gradient_counts[name] += 1
                if self.plan_gradient_counts[name] != 1:
                    self._fail("renderer plan received duplicate backward traversal")
                self.plan_gradient_finite[name] = (
                    type(gradient) is torch.Tensor
                    and bool(torch.isfinite(gradient.detach()).all().item())
                )
                return gradient

            return observe

        try:
            self.gradient_handles.extend(
                output.register_hook(output_hook(index))
                for index, output in enumerate(self.outputs)
            )
            self.gradient_handles.extend(
                output.register_hook(injection_hook(index))
                for index, output in enumerate(self.injection_outputs)
            )
            self.gradient_handles.append(
                renderer_plan.phase_tokens.register_hook(plan_hook("phase"))
            )
            self.gradient_handles.append(
                renderer_plan.global_token.register_hook(plan_hook("global"))
            )
        except Exception:
            self.abort()
            raise
        self.state = "artifact_armed"

    def begin_internal_vjp(self) -> None:
        if self.state not in ("artifact_armed", "combined_ready"):
            self._fail("internal VJP lease is out of order")
        self._internal_vjp_return_state = self.state
        self.state = "internal_vjp"

    def end_internal_vjp(self) -> None:
        if self.state != "internal_vjp":
            self._fail("internal VJP completion is out of order")
        # Internal audits never count as the one real combined backward.
        self.output_gradient_counts = [0] * len(self.projections)
        self.output_gradient_finite = [False] * len(self.projections)
        self.output_gradient_nonzero = [False] * len(self.projections)
        self.injection_gradient_counts = [0] * len(self.projections)
        self.injection_gradient_finite = [False] * len(self.projections)
        self.injection_gradient_nonzero = [False] * len(self.projections)
        self.plan_gradient_counts = {"phase": 0, "global": 0}
        self.plan_gradient_finite = {"phase": False, "global": False}
        if self._internal_vjp_return_state not in (
            "artifact_armed",
            "combined_ready",
        ):
            self._fail("internal VJP return state is invalid")
        self.state = self._internal_vjp_return_state
        self._internal_vjp_return_state = None

    def mark_combined_ready(self) -> None:
        if (
            self.state != "artifact_armed"
            or self._registered_combined is None
            or self._flow_dependency_audited is not True
            or self._combined_dependency_audited is not True
        ):
            self._fail("renderer artifact cannot be combined twice")
        if any(
            parameter.grad is not None for parameter in self.allowed_trainables
        ):
            self._fail(
                "structural V1 requires clear conditioner gradients before its one-shot backward"
            )
        self.state = "combined_ready"

    def begin_backward(self) -> None:
        if (
            self.state != "combined_ready"
            or self._flow_dependency_audited is not True
            or self._combined_dependency_audited is not True
        ):
            self._fail("combined renderer loss is one-shot and not backward-ready")
        self.state = "backward_active"

    def backward_returned(self) -> None:
        if self.state != "backward_active":
            self._fail("combined backward lifecycle differs")
        self.state = "awaiting_finalize"

    def finalize_backward(self) -> dict[str, Any]:
        torch = _torch()
        if (
            self.state != "awaiting_finalize"
            or self._flow_dependency_audited is not True
            or self._combined_dependency_audited is not True
        ):
            self._fail("renderer backward finalize is one-shot and out of order")
        expected_count = action_plan.TRANSFORMER_BLOCK_COUNT
        projection_gradient_tensor_count = 0
        projection_nonzero_head_count = 0
        for weight, bias in self.projection_parameters:
            gradients = (weight.grad, bias.grad)
            if any(
                type(gradient) is not torch.Tensor
                or not bool(torch.isfinite(gradient.detach()).all().item())
                for gradient in gradients
            ):
                self._fail(
                    "an injection projection parameter lacks a finite gradient"
                )
            projection_gradient_tensor_count += len(gradients)
            if any(
                bool(torch.count_nonzero(gradient.detach()).item() > 0)
                for gradient in gradients
            ):
                projection_nonzero_head_count += 1
        predictor_gradients = [
            parameter.grad for parameter in self.predictor_trainables
        ]
        if (
            not predictor_gradients
            or any(
                type(gradient) is not torch.Tensor
                or not bool(torch.isfinite(gradient.detach()).all().item())
                for gradient in predictor_gradients
            )
            or not any(
                bool(torch.count_nonzero(gradient.detach()).item() > 0)
                for gradient in predictor_gradients
            )
        ):
            self._fail("predictor trainables lack finite nonzero distillation gradients")
        if (
            self.output_gradient_counts != [1] * expected_count
            or not all(self.output_gradient_finite)
            or not all(self.output_gradient_nonzero)
            or self.injection_gradient_counts != [1] * expected_count
            or not all(self.injection_gradient_finite)
            or not all(self.injection_gradient_nonzero)
            or self.plan_gradient_counts != {"phase": 1, "global": 1}
            or not all(self.plan_gradient_finite.values())
            or projection_gradient_tensor_count != 2 * expected_count
            or projection_nonzero_head_count != expected_count
        ):
            self._fail(
                "renderer backward did not deliver one finite nonzero gradient to every injection output"
            )
        self._remove_gradient_hooks()
        self.state = "consumed"
        return {
            "schema_version": RENDERER_FLOW_BACKWARD_RECEIPT_SCHEMA,
            "exact_projection_count": expected_count,
            "projection_gradient_count": sum(self.output_gradient_counts),
            "all_projection_gradients_finite": True,
            "all_projection_gradients_nonzero": True,
            "target_hidden_injection_gradient_count": sum(
                self.injection_gradient_counts
            ),
            "all_target_hidden_injection_gradients_finite_nonzero": True,
            "projection_parameter_gradient_tensor_count": (
                projection_gradient_tensor_count
            ),
            "projection_nonzero_head_count": projection_nonzero_head_count,
            "predictor_gradient_tensor_count": len(predictor_gradients),
            "predictor_has_finite_nonzero_gradient": True,
            "phase_global_gradients_finite": True,
            "gradient_checkpointing_supported": False,
            "one_shot_consumed": True,
            "training_authorized": False,
            "optimizer_step_authorized": False,
            "structural_route_evidence_only": True,
            "real_renderer_flow_authorized": False,
        }


@dataclass(frozen=True)
class PreparedActionAnchorRendererV1:
    schema_version: str
    q_pred_fp32: action_plan.ActionPlanOutput
    q_pred_receipt: Mapping[str, Any]
    renderer_plan: action_plan.ActionPlanOutput
    ownership: action_plan.TargetOwnershipCertificate
    route: action_plan.ActionPlanInjectionRoute
    sidecar: Mapping[str, Any]
    externally_expected_sidecar_digest: str
    externally_expected_teacher_authority_sha256: str
    externally_expected_classification_authority_sha256: str
    source_token_tensor_sha256: str
    instruction_token_tensor_sha256: str
    q_pred_receipt_digest: str
    q_pred_phase_raw_sha256: str
    q_pred_global_raw_sha256: str
    renderer_phase_sha256: str
    renderer_global_sha256: str
    _recorder: _RouteUseRecorderV1


def prepare_action_anchor_renderer_v1(
    *,
    conditioner: Any,
    source_tokens: Any,
    instruction_tokens: Any,
    target_hidden: Any,
    source_prefix_tokens: int,
    packed_total_tokens: int,
    q_pred_bindings: Sequence[Mapping[str, Any]],
    predictor_artifact_sha256: str,
    sidecar_envelope: Mapping[str, Any],
    expected_sidecar_envelope_digest: str,
    expected_teacher_authority_sha256: str,
    expected_classification_authority_sha256: str,
) -> PreparedActionAnchorRendererV1:
    """Prepare one q_pred and renderer route from exactly one predictor call."""

    torch = _torch()
    if type(conditioner) is not action_plan.ActionPlanConditionerV1:
        raise ActionAnchorRendererIntegrationError(
            "conditioner must be the exact ActionPlanConditionerV1 type"
        )
    envelope = validate_sidecar_envelope_v1(
        sidecar_envelope,
        expected_envelope_digest=expected_sidecar_envelope_digest,
        expected_teacher_authority_sha256=expected_teacher_authority_sha256,
        expected_classification_authority_sha256=
        expected_classification_authority_sha256,
    )
    predictor_sha = _sha256(
        predictor_artifact_sha256, label="observed predictor artifact SHA-256"
    )
    if envelope["predictor_artifact_sha256"] != predictor_sha:
        raise ActionAnchorRendererIntegrationError(
            "predictor artifact differs from pinned sidecar"
        )
    observed_abi = action_plan.exact_state_dict_abi(conditioner)["abi_sha256"]
    if envelope["conditioner_state_abi_sha256"] != observed_abi:
        raise ActionAnchorRendererIntegrationError(
            "conditioner state ABI differs from pinned sidecar"
        )
    if (
        type(source_tokens) is not torch.Tensor
        or type(instruction_tokens) is not torch.Tensor
        or type(target_hidden) is not torch.Tensor
        or not source_tokens.is_floating_point()
        or not instruction_tokens.is_floating_point()
        or not target_hidden.is_floating_point()
        or source_tokens.dtype
        not in (torch.float16, torch.bfloat16, torch.float32)
        or instruction_tokens.dtype
        not in (torch.float16, torch.bfloat16, torch.float32)
        or not bool(torch.isfinite(source_tokens.detach()).all().item())
        or not bool(torch.isfinite(instruction_tokens.detach()).all().item())
        or not bool(torch.isfinite(target_hidden.detach()).all().item())
    ):
        raise ActionAnchorRendererIntegrationError(
            "source, instruction, and target must be finite supported floating tensors"
        )
    if target_hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ActionAnchorRendererIntegrationError(
            "target hidden dtype is outside the renderer compute policy"
        )
    source_token_sha = tensor_sha256_v1(source_tokens)
    instruction_token_sha = tensor_sha256_v1(instruction_tokens)
    if (
        source_token_sha != envelope["source_token_tensor_sha256"]
        or instruction_token_sha
        != envelope["instruction_token_tensor_sha256"]
    ):
        raise ActionAnchorRendererIntegrationError(
            "source/instruction token tensors differ from externally pinned sidecar identities"
        )
    # Detach the two semantic inputs before the sole predictor forward.  This
    # keeps gradients in predictor parameters while forbidding a shortcut
    # into an upstream source/text encoder.
    source_fp32 = source_tokens.detach().float().contiguous()
    instruction_fp32 = instruction_tokens.detach().float().contiguous()
    # Do not inherit a caller's mixed-precision region.  There is exactly one
    # predictor call: an FP32 ABI failure is fatal and must never trigger a
    # silent retry/second forward.
    with torch.autocast(device_type=source_fp32.device.type, enabled=False):
        q_pred_fp32 = conditioner.predictor(source_fp32, instruction_fp32)
    batch_size = int(source_fp32.shape[0])
    _require_plan(
        q_pred_fp32,
        label="q_pred FP32",
        batch_size=batch_size,
        dtype=torch.float32,
        device=source_fp32.device,
    )
    if not q_pred_fp32.phase_tokens.requires_grad or not q_pred_fp32.global_token.requires_grad:
        raise ActionAnchorRendererIntegrationError(
            "q_pred FP32 lost its predictor autograd path"
        )
    q_pred_receipt = distillation.build_q_receipt_v1(
        q_kind="q_pred",
        plan=q_pred_fp32,
        bindings=q_pred_bindings,
        producer_artifact_sha256=predictor_sha,
    )
    if (
        q_pred_receipt["layout"]["batch_size"] != envelope["batch_size"]
        or [item["row_id"] for item in q_pred_receipt["items"]]
        != envelope["row_ids"]
    ):
        raise ActionAnchorRendererIntegrationError(
            "q_pred receipt rows differ from pinned sidecar"
        )
    renderer_plan = _cast_plan_with_graph(q_pred_fp32, dtype=target_hidden.dtype)
    _require_plan(
        renderer_plan,
        label="renderer plan",
        batch_size=batch_size,
        dtype=target_hidden.dtype,
        device=target_hidden.device,
    )
    if renderer_plan.phase_tokens.grad_fn is None or renderer_plan.global_token.grad_fn is None:
        raise ActionAnchorRendererIntegrationError(
            "renderer-dtype cast detached the predictor graph"
        )
    ownership = action_plan.certify_closed_target_suffix_route(
        target_hidden,
        source_prefix_tokens=source_prefix_tokens,
        packed_total_tokens=packed_total_tokens,
        audit_finite=True,
    )
    route = conditioner.injection.bind_route(
        renderer_plan, ownership, audit_finite=True
    )
    recorder = _RouteUseRecorderV1(
        conditioner=conditioner,
        renderer_plan=renderer_plan,
        route=route,
        initial_target_hidden=target_hidden,
    )
    return PreparedActionAnchorRendererV1(
        schema_version=PREPARED_ROUTE_SCHEMA,
        q_pred_fp32=q_pred_fp32,
        q_pred_receipt=q_pred_receipt,
        renderer_plan=renderer_plan,
        ownership=ownership,
        route=route,
        sidecar=envelope,
        externally_expected_sidecar_digest=expected_sidecar_envelope_digest,
        externally_expected_teacher_authority_sha256=
        expected_teacher_authority_sha256,
        externally_expected_classification_authority_sha256=
        expected_classification_authority_sha256,
        source_token_tensor_sha256=source_token_sha,
        instruction_token_tensor_sha256=instruction_token_sha,
        q_pred_receipt_digest=_authority_sha256(
            q_pred_receipt["receipt_digest"], label="q_pred receipt digest"
        ),
        q_pred_phase_raw_sha256=distillation._raw_fp32_tensor_sha256(
            q_pred_fp32.phase_tokens
        ),
        q_pred_global_raw_sha256=distillation._raw_fp32_tensor_sha256(
            q_pred_fp32.global_token
        ),
        renderer_phase_sha256=tensor_sha256_v1(renderer_plan.phase_tokens),
        renderer_global_sha256=tensor_sha256_v1(renderer_plan.global_token),
        _recorder=recorder,
    )


def cancel_prepared_renderer_route_v1(
    prepared: PreparedActionAnchorRendererV1,
) -> None:
    """Release an unused forward recorder without producing any authority."""

    if (
        type(prepared) is not PreparedActionAnchorRendererV1
        or type(prepared._recorder) is not _RouteUseRecorderV1
        or prepared._recorder.state != "recording"
    ):
        raise ActionAnchorRendererIntegrationError(
            "only one unused prepared renderer route can be cancelled"
        )
    prepared._recorder._remove_forward_hooks()
    prepared._recorder.state = "cancelled"


def _revalidate_prepared_v1(
    prepared: Any,
) -> tuple[PreparedActionAnchorRendererV1, dict[str, Any]]:
    torch = _torch()
    if (
        type(prepared) is not PreparedActionAnchorRendererV1
        or prepared.schema_version != PREPARED_ROUTE_SCHEMA
        or type(prepared._recorder) is not _RouteUseRecorderV1
    ):
        raise ActionAnchorRendererIntegrationError(
            "prepared route must be exact PreparedActionAnchorRendererV1"
        )
    sidecar = validate_sidecar_envelope_v1(
        prepared.sidecar,
        expected_envelope_digest=prepared.externally_expected_sidecar_digest,
        expected_teacher_authority_sha256=
        prepared.externally_expected_teacher_authority_sha256,
        expected_classification_authority_sha256=
        prepared.externally_expected_classification_authority_sha256,
    )
    if (
        prepared.q_pred_receipt.get("receipt_digest")
        != prepared.q_pred_receipt_digest
    ):
        raise ActionAnchorRendererIntegrationError(
            "q_pred receipt changed after prepare"
        )
    try:
        validated_pred = distillation.validate_q_receipt_v1(
            prepared.q_pred_receipt, plan=prepared.q_pred_fp32
        )
    except Exception as error:
        raise ActionAnchorRendererIntegrationError(
            f"q_pred receipt/tensor binding changed after prepare: {error}"
        ) from error
    if (
        validated_pred["receipt_digest"] != prepared.q_pred_receipt_digest
        or [item["row_id"] for item in validated_pred["items"]]
        != sidecar["row_ids"]
        or prepared.source_token_tensor_sha256
        != sidecar["source_token_tensor_sha256"]
        or prepared.instruction_token_tensor_sha256
        != sidecar["instruction_token_tensor_sha256"]
        or distillation._raw_fp32_tensor_sha256(
            prepared.q_pred_fp32.phase_tokens
        )
        != prepared.q_pred_phase_raw_sha256
        or distillation._raw_fp32_tensor_sha256(
            prepared.q_pred_fp32.global_token
        )
        != prepared.q_pred_global_raw_sha256
        or tensor_sha256_v1(prepared.renderer_plan.phase_tokens)
        != prepared.renderer_phase_sha256
        or tensor_sha256_v1(prepared.renderer_plan.global_token)
        != prepared.renderer_global_sha256
        or prepared.route.plan is not prepared.renderer_plan
        or prepared.route.plan.phase_tokens is not prepared.renderer_plan.phase_tokens
        or prepared.route.plan.global_token is not prepared.renderer_plan.global_token
    ):
        raise ActionAnchorRendererIntegrationError(
            "prepared q_pred/renderer plan snapshot changed"
        )
    expected_renderer = _cast_plan_with_graph(
        prepared.q_pred_fp32, dtype=prepared.renderer_plan.phase_tokens.dtype
    )
    if (
        not bool(torch.equal(
            expected_renderer.phase_tokens.detach(),
            prepared.renderer_plan.phase_tokens.detach(),
        ))
        or not bool(torch.equal(
            expected_renderer.global_token.detach(),
            prepared.renderer_plan.global_token.detach(),
        ))
    ):
        raise ActionAnchorRendererIntegrationError(
            "renderer plan no longer equals the frozen q_pred cast"
        )
    return prepared, sidecar


def _tensor_version(value: Any) -> int:
    version = getattr(value, "_version", None)
    if type(version) is not int or version < 0:
        raise ActionAnchorRendererIntegrationError(
            "tensor has no exact mutation-version counter"
        )
    return version


class StructuralRendererFlowArtifactV1:
    """Recorder-issued immutable handle; never a real-renderer authority."""

    __slots__ = (
        "schema_version",
        "flow",
        "prediction",
        "target_clean",
        "noise",
        "target_velocity",
        "caller_observed_prediction_sha256",
        "expected_target_clean_sha256",
        "expected_noise_sha256",
        "gradient_accumulation",
        "structural_route_evidence_only",
        "real_renderer_flow_authorized",
        "_prepared",
        "_recorder",
        "_construction_nonce",
    )

    def __init__(
        self,
        *,
        schema_version: str,
        flow: Any,
        prediction: Any,
        target_clean: Any,
        noise: Any,
        target_velocity: Any,
        caller_observed_prediction_sha256: str,
        expected_target_clean_sha256: str,
        expected_noise_sha256: str,
        gradient_accumulation: int,
        structural_route_evidence_only: bool,
        real_renderer_flow_authorized: bool,
        _prepared: PreparedActionAnchorRendererV1,
        _recorder: _RouteUseRecorderV1,
        _construction_nonce: Any,
    ):
        if (
            type(_recorder) is not _RouteUseRecorderV1
            or _construction_nonce is not _recorder._artifact_nonce
            or _recorder._registered_artifact is not None
        ):
            raise ActionAnchorRendererIntegrationError(
                "StructuralRendererFlowArtifactV1 is factory-only"
            )
        for name, value in (
            ("schema_version", schema_version),
            ("flow", flow),
            ("prediction", prediction),
            ("target_clean", target_clean),
            ("noise", noise),
            ("target_velocity", target_velocity),
            (
                "caller_observed_prediction_sha256",
                caller_observed_prediction_sha256,
            ),
            ("expected_target_clean_sha256", expected_target_clean_sha256),
            ("expected_noise_sha256", expected_noise_sha256),
            ("gradient_accumulation", gradient_accumulation),
            (
                "structural_route_evidence_only",
                structural_route_evidence_only,
            ),
            ("real_renderer_flow_authorized", real_renderer_flow_authorized),
            ("_prepared", _prepared),
            ("_recorder", _recorder),
            ("_construction_nonce", _construction_nonce),
        ):
            object.__setattr__(self, name, value)

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise ActionAnchorRendererIntegrationError(
            "structural flow artifact fields are immutable"
        )


# Compatibility name only; the class and every receipt remain explicitly
# structural-only and real-renderer-NO-GO.
RendererFlowArtifactV1 = StructuralRendererFlowArtifactV1


def _canonical_structural_flow_detached_v1(
    value: StructuralRendererFlowArtifactV1,
) -> Any:
    torch = _torch()
    target_velocity = (
        value.noise.detach().float() - value.target_clean.detach().float()
    ).contiguous()
    return (
        torch.nn.functional.mse_loss(
            value.prediction.detach().float(),
            target_velocity,
            reduction="mean",
        )
        / float(value.gradient_accumulation)
    ).float().reshape(())


def _snapshot_structural_flow_artifact_v1(
    value: StructuralRendererFlowArtifactV1,
) -> dict[str, Any]:
    canonical = _canonical_structural_flow_detached_v1(value)
    if not bool(_torch().equal(value.flow.detach(), canonical)):
        raise ActionAnchorRendererIntegrationError(
            "structural flow differs from canonical MSE(noise-target)"
        )
    tensors = {
        "flow": value.flow,
        "prediction": value.prediction,
        "target_clean": value.target_clean,
        "noise": value.noise,
        "target_velocity": value.target_velocity,
    }
    return {
        "metadata": {
            "schema_version": value.schema_version,
            "gradient_accumulation": value.gradient_accumulation,
            "structural_route_evidence_only": (
                value.structural_route_evidence_only
            ),
            "real_renderer_flow_authorized": (
                value.real_renderer_flow_authorized
            ),
            "caller_observed_prediction_sha256": (
                value.caller_observed_prediction_sha256
            ),
            "expected_target_clean_sha256": (
                value.expected_target_clean_sha256
            ),
            "expected_noise_sha256": value.expected_noise_sha256,
            "prepared_id": id(value._prepared),
            "recorder_id": id(value._recorder),
            "construction_nonce_id": id(value._construction_nonce),
        },
        "object_ids": {name: id(tensor) for name, tensor in tensors.items()},
        "versions": {
            name: _tensor_version(tensor) for name, tensor in tensors.items()
        },
        "sha256": {
            name: tensor_sha256_v1(tensor) for name, tensor in tensors.items()
        },
        "flow_value": float(value.flow.detach().item()),
        "canonical_flow_sha256": tensor_sha256_v1(canonical),
    }


def _build_renderer_flow_artifact_impl_v1(
    *,
    prepared: PreparedActionAnchorRendererV1,
    prediction: Any,
    target_clean: Any,
    noise: Any,
    caller_observed_prediction_sha256: str,
    expected_target_clean_sha256: str,
    expected_noise_sha256: str,
    gradient_accumulation: int = 1,
    _factory_issue: Any,
) -> RendererFlowArtifactV1:
    """Seal exact-30 route use and construct rectified-flow MSE internally."""

    torch = _torch()
    recorder = getattr(prepared, "_recorder", None)
    try:
        checked, _sidecar = _revalidate_prepared_v1(prepared)
        if type(gradient_accumulation) is not int or gradient_accumulation <= 0:
            raise ActionAnchorRendererIntegrationError(
                "gradient accumulation must be a positive exact integer"
            )
        tensors = (prediction, target_clean, noise)
        if any(
            type(value) is not torch.Tensor
            or not value.is_floating_point()
            or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or value.ndim <= 0
            for value in tensors
        ):
            raise ActionAnchorRendererIntegrationError(
                "prediction/target/noise must be floating FP16/BF16/FP32 tensors"
            )
        if (
            tuple(prediction.shape) != tuple(target_clean.shape)
            or tuple(prediction.shape) != tuple(noise.shape)
            or prediction.dtype != target_clean.dtype
            or prediction.dtype != noise.dtype
            or prediction.device != target_clean.device
            or prediction.device != noise.device
            or prediction.device != checked.q_pred_fp32.phase_tokens.device
            or int(prediction.shape[0])
            != int(checked.q_pred_fp32.phase_tokens.shape[0])
            or not prediction.requires_grad
            or target_clean.requires_grad
            or noise.requires_grad
            or target_clean.grad_fn is not None
            or noise.grad_fn is not None
            or any(
                not bool(torch.isfinite(value.detach()).all().item())
                for value in tensors
            )
        ):
            raise ActionAnchorRendererIntegrationError(
                "prediction/target/noise geometry, dtype, ownership, or autograd role differs"
            )
        spans = [_contiguous_storage_span(value) for value in tensors]
        if any(
            _spans_overlap(spans[left], spans[right])
            for left in range(len(spans))
            for right in range(left + 1, len(spans))
        ):
            raise ActionAnchorRendererIntegrationError(
                "prediction/target/noise storage ownership overlaps"
            )
        observed_p = _authority_sha256(
            caller_observed_prediction_sha256,
            label="caller-observed renderer prediction tensor digest",
        )
        expected_t = _authority_sha256(
            expected_target_clean_sha256,
            label="externally expected clean target tensor digest",
        )
        expected_n = _authority_sha256(
            expected_noise_sha256,
            label="externally expected flow noise tensor digest",
        )
        if (
            tensor_sha256_v1(prediction) != observed_p
            or tensor_sha256_v1(target_clean) != expected_t
            or tensor_sha256_v1(noise) != expected_n
            or len({observed_p, expected_t, expected_n}) != 3
        ):
            raise ActionAnchorRendererIntegrationError(
                "prediction integrity snapshot or external target/noise pins differ"
            )
        recorder.seal_forward()
        target_velocity = (
            noise.detach().float() - target_clean.detach().float()
        ).contiguous()
        flow = torch.nn.functional.mse_loss(
            prediction.float(), target_velocity, reduction="mean"
        ) / float(gradient_accumulation)
        if not bool(torch.isfinite(flow.detach()).item()):
            raise ActionAnchorRendererIntegrationError(
                "internally constructed renderer flow loss is non-finite"
            )
        dependency_inputs = (
            tuple(recorder.outputs)
            + tuple(recorder.injection_outputs)
            + (
            checked.renderer_plan.phase_tokens,
            checked.renderer_plan.global_token,
            )
        )
        dependencies = torch.autograd.grad(
            flow,
            dependency_inputs,
            allow_unused=True,
            retain_graph=True,
        )
        if any(
            gradient is None
            or not bool(torch.isfinite(gradient.detach()).all().item())
            for gradient in dependencies
        ):
            raise ActionAnchorRendererIntegrationError(
                "renderer flow is disconnected from an injection output or renderer plan"
            )
        structural_output_count = 2 * action_plan.TRANSFORMER_BLOCK_COUNT
        if any(
            not bool(torch.count_nonzero(gradient.detach()).item() > 0)
            for gradient in dependencies[:structural_output_count]
        ):
            raise ActionAnchorRendererIntegrationError(
                "renderer flow has a zero dependency on a projection or target-hidden injection output"
            )
        recorder.state = "artifact_ready"
        recorder.arm_backward(renderer_plan=checked.renderer_plan)
        artifact = _factory_issue(
            recorder=recorder,
            schema_version=RENDERER_FLOW_ARTIFACT_SCHEMA,
            flow=flow.float().reshape(()),
            prediction=prediction,
            target_clean=target_clean,
            noise=noise,
            target_velocity=target_velocity,
            caller_observed_prediction_sha256=observed_p,
            expected_target_clean_sha256=expected_t,
            expected_noise_sha256=expected_n,
            gradient_accumulation=gradient_accumulation,
            structural_route_evidence_only=True,
            real_renderer_flow_authorized=False,
            _prepared=checked,
            _recorder=recorder,
            _construction_nonce=recorder._artifact_nonce,
        )
        return artifact
    except Exception as error:
        if type(recorder) is _RouteUseRecorderV1:
            recorder.abort()
        if isinstance(error, ActionAnchorRendererIntegrationError):
            raise
        raise ActionAnchorRendererIntegrationError(
            f"renderer flow artifact construction failed: {error}"
        ) from error


def _bind_renderer_flow_artifact_factory_v1(
    implementation: Any, factory_stage: Any
) -> Any:
    """Keep the issuance minter in the supported builder's closure."""

    def issue(*, recorder: Any, **values: Any) -> Any:
        artifact = StructuralRendererFlowArtifactV1(**values)
        factory_stage(recorder, artifact)
        recorder.register_artifact(artifact)
        return artifact

    def build_renderer_flow_artifact_v1(
        *,
        prepared: PreparedActionAnchorRendererV1,
        prediction: Any,
        target_clean: Any,
        noise: Any,
        caller_observed_prediction_sha256: str,
        expected_target_clean_sha256: str,
        expected_noise_sha256: str,
        gradient_accumulation: int = 1,
    ) -> RendererFlowArtifactV1:
        return implementation(
            prepared=prepared,
            prediction=prediction,
            target_clean=target_clean,
            noise=noise,
            caller_observed_prediction_sha256=(
                caller_observed_prediction_sha256
            ),
            expected_target_clean_sha256=expected_target_clean_sha256,
            expected_noise_sha256=expected_noise_sha256,
            gradient_accumulation=gradient_accumulation,
            _factory_issue=issue,
        )

    return build_renderer_flow_artifact_v1


build_renderer_flow_artifact_v1 = _bind_renderer_flow_artifact_factory_v1(
    _build_renderer_flow_artifact_impl_v1,
    _stage_factory_artifact_issuance_v1,
)
del _bind_renderer_flow_artifact_factory_v1
del _stage_factory_artifact_issuance_v1


def _revalidate_flow_artifact_v1(
    prepared: PreparedActionAnchorRendererV1,
    value: Any,
    *,
    allowed_states: Sequence[str] = ("artifact_armed",),
    require_registered: bool = True,
) -> RendererFlowArtifactV1:
    torch = _torch()
    prepared, _sidecar = _revalidate_prepared_v1(prepared)
    if type(allowed_states) not in (list, tuple) or not allowed_states:
        raise ActionAnchorRendererIntegrationError(
            "flow artifact state allowlist must be a nonempty exact list/tuple"
        )
    recorder = getattr(prepared, "_recorder", None)
    tensors = (
        getattr(value, "prediction", None),
        getattr(value, "target_clean", None),
        getattr(value, "noise", None),
    )
    artifact_tensors = (
        getattr(value, "flow", None),
        *tensors,
        getattr(value, "target_velocity", None),
    )
    if (
        type(value) is not RendererFlowArtifactV1
        or value.schema_version != RENDERER_FLOW_ARTIFACT_SCHEMA
        or value._prepared is not prepared
        or type(recorder) is not _RouteUseRecorderV1
        or value._recorder is not recorder
        or value._construction_nonce is not recorder._artifact_nonce
        or recorder.state not in tuple(allowed_states)
        or value.structural_route_evidence_only is not True
        or value.real_renderer_flow_authorized is not False
        or type(value.gradient_accumulation) is not int
        or value.gradient_accumulation <= 0
        or any(
            type(tensor) is not torch.Tensor
            or not tensor.is_floating_point()
            or tensor.dtype
            not in (torch.float16, torch.bfloat16, torch.float32)
            or tensor.ndim <= 0
            for tensor in tensors
        )
        or tuple(value.prediction.shape) != tuple(value.target_clean.shape)
        or tuple(value.prediction.shape) != tuple(value.noise.shape)
        or value.prediction.dtype != value.target_clean.dtype
        or value.prediction.dtype != value.noise.dtype
        or value.prediction.device != value.target_clean.device
        or value.prediction.device != value.noise.device
        or value.prediction.device
        != prepared.q_pred_fp32.phase_tokens.device
        or int(value.prediction.shape[0])
        != int(prepared.q_pred_fp32.phase_tokens.shape[0])
        or not value.prediction.requires_grad
        or value.target_clean.requires_grad
        or value.noise.requires_grad
        or value.target_clean.grad_fn is not None
        or value.noise.grad_fn is not None
        or any(
            not bool(torch.isfinite(tensor.detach()).all().item())
            for tensor in tensors
        )
        or type(value.flow) is not torch.Tensor
        or not value.flow.is_floating_point()
        or value.flow.dtype != torch.float32
        or value.flow.device != value.prediction.device
        or value.flow.numel() != 1
        or not value.flow.requires_grad
        or not bool(torch.isfinite(value.flow.detach()).item())
        or type(value.target_velocity) is not torch.Tensor
        or value.target_velocity.dtype != torch.float32
        or value.target_velocity.device != value.prediction.device
        or value.target_velocity.requires_grad
        or value.target_velocity.grad_fn is not None
        or not value.target_velocity.is_contiguous()
        or tensor_sha256_v1(value.prediction)
        != value.caller_observed_prediction_sha256
        or tensor_sha256_v1(value.target_clean)
        != value.expected_target_clean_sha256
        or tensor_sha256_v1(value.noise) != value.expected_noise_sha256
        or not bool(torch.equal(
            value.target_velocity,
            value.noise.detach().float() - value.target_clean.detach().float(),
        ))
        or (
            require_registered
            and (
                recorder._registered_artifact is not value
                or recorder._registered_flow_tensor is not value.flow
                or recorder._registered_artifact_tensors is None
                or len(recorder._registered_artifact_tensors) != 5
                or any(
                    registered is not current
                    for registered, current in zip(
                        recorder._registered_artifact_tensors,
                        artifact_tensors,
                    )
                )
                or recorder._artifact_snapshot is None
            )
        )
        or (
            not require_registered
            and (
                recorder._registered_artifact is not None
                or recorder._registered_artifact_tensors is not None
                or recorder._artifact_snapshot is not None
            )
        )
    ):
        raise ActionAnchorRendererIntegrationError(
            "renderer flow artifact changed, was reused, or belongs to another route"
        )
    spans = [_contiguous_storage_span(tensor) for tensor in tensors]
    if any(
        _spans_overlap(spans[left], spans[right])
        for left in range(len(spans))
        for right in range(left + 1, len(spans))
    ):
        raise ActionAnchorRendererIntegrationError(
            "renderer structural flow tensor storage ownership overlaps"
        )
    try:
        current_snapshot = _snapshot_structural_flow_artifact_v1(value)
    except Exception as error:
        raise ActionAnchorRendererIntegrationError(
            f"renderer structural flow formula/snapshot changed: {error}"
        ) from error
    if require_registered and current_snapshot != recorder._artifact_snapshot:
        raise ActionAnchorRendererIntegrationError(
            "renderer structural flow identity, version, raw hash, or formula changed"
        )
    if require_registered:
        _verify_factory_artifact_issuance_v1(recorder, value)
    if recorder.state in ("artifact_armed", "combined_ready"):
        dependency_inputs = (
            tuple(recorder.outputs)
            + tuple(recorder.injection_outputs)
            + (
                prepared.renderer_plan.phase_tokens,
                prepared.renderer_plan.global_token,
            )
        )
        recorder.begin_internal_vjp()
        try:
            dependencies = torch.autograd.grad(
                value.flow,
                dependency_inputs,
                allow_unused=True,
                retain_graph=True,
            )
        except Exception as error:
            raise ActionAnchorRendererIntegrationError(
                f"structural flow dependency replay failed: {error}"
            ) from error
        finally:
            if recorder.state == "internal_vjp":
                recorder.end_internal_vjp()
        structural_output_count = 2 * action_plan.TRANSFORMER_BLOCK_COUNT
        if (
            len(dependencies)
            != structural_output_count + 2
            or any(
                gradient is None
                or not bool(torch.isfinite(gradient.detach()).all().item())
                for gradient in dependencies
            )
            or any(
                not bool(torch.count_nonzero(gradient.detach()).item() > 0)
                for gradient in dependencies[:structural_output_count]
            )
        ):
            raise ActionAnchorRendererIntegrationError(
                "structural flow dependency replay differs from exact-30 route"
            )
        recorder._flow_dependency_audited = True
    return value


class CombinedStructuralActionAnchorLossV1:
    """Factory-only one-shot structural objective handle."""

    __slots__ = (
        "schema_version",
        "total",
        "flow",
        "distillation",
        "training_authorized",
        "optimizer_step_authorized",
        "gradient_checkpointing_supported",
        "conditioner_regularizer_only",
        "structural_route_evidence_only",
        "real_renderer_flow_authorized",
        "sidecar_envelope_digest",
        "predictor_artifact_sha256",
        "conditioner_state_abi_sha256",
        "teacher_authority_sha256",
        "classification_authority_sha256",
        "row_ids",
        "source_token_tensor_sha256",
        "instruction_token_tensor_sha256",
        "q_pred_receipt_digest",
        "q_y_receipt_digest",
        "q_anchor_receipt_digests",
        "q_y_qualification_receipt_digests",
        "q_anchor_qualification_receipt_digests",
        "compatibility_decision_receipt_digests",
        "caller_observed_prediction_sha256",
        "expected_target_clean_sha256",
        "expected_noise_sha256",
        "canonical_flow_sha256",
        "canonical_total_sha256",
        "distillation_config",
        "conditioner_regularizer_sha256",
        "_artifact",
        "_recorder",
        "_q_y",
        "_q_y_receipt",
        "_anchors",
        "_conditioner_regularizer_loss",
        "_config",
        "_construction_nonce",
    )

    def __init__(self, *, _construction_nonce: Any, **values: Any):
        recorder = values.get("_recorder")
        expected_fields = set(self.__slots__) - {"_construction_nonce"}
        if (
            type(recorder) is not _RouteUseRecorderV1
            or _construction_nonce is not recorder._combined_nonce
            or recorder._registered_combined is not None
            or set(values) != expected_fields
        ):
            raise ActionAnchorRendererIntegrationError(
                "CombinedStructuralActionAnchorLossV1 is factory-only"
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_construction_nonce", _construction_nonce)

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise ActionAnchorRendererIntegrationError(
            "combined structural loss fields are immutable"
        )

    def backward(self, *, retain_graph: bool = False) -> None:
        if type(retain_graph) is not bool or retain_graph is not False:
            raise ActionAnchorRendererIntegrationError(
                "combined backward is one-shot and requires retain_graph=False"
            )
        try:
            _revalidate_combined_loss_v1(self, allowed_states=("combined_ready",))
            self._recorder.begin_backward()
            self.total.backward(retain_graph=False)
            _revalidate_combined_loss_v1(self, allowed_states=("backward_active",))
            self._recorder.backward_returned()
        except Exception as error:
            self._recorder.abort()
            if isinstance(error, ActionAnchorRendererIntegrationError):
                raise
            raise ActionAnchorRendererIntegrationError(
                f"combined structural backward provenance failed: {error}"
            ) from error


CombinedActionAnchorLossV1 = CombinedStructuralActionAnchorLossV1


_DISTILLATION_CONFIG_FIELDS = (
    "smooth_l1_weight",
    "cosine_weight",
    "infonce_weight",
    "preservation_weight",
    "smooth_l1_beta",
    "temperature",
)


def _distillation_config_record_v1(value: Any) -> tuple[tuple[str, float], ...]:
    if type(value) is not distillation.DistillationLossConfigV1:
        raise ActionAnchorRendererIntegrationError(
            "distillation config must be exact DistillationLossConfigV1"
        )
    value.validate()
    return tuple(
        (name, float(getattr(value, name)))
        for name in _DISTILLATION_CONFIG_FIELDS
    )


def _canonical_combined_total_detached_v1(
    value: CombinedStructuralActionAnchorLossV1,
) -> Any:
    return (
        value.flow.detach().float()
        + value.distillation.total.detach().float()
    ).reshape(())


def _canonical_distillation_total_detached_v1(
    value: CombinedStructuralActionAnchorLossV1,
) -> Any:
    cfg = value._config
    distilled = value.distillation
    return (
        float(cfg.smooth_l1_weight) * distilled.smooth_l1.detach()
        + float(cfg.cosine_weight) * distilled.cosine.detach()
        + float(cfg.infonce_weight) * distilled.infonce.detach()
        + float(cfg.preservation_weight) * distilled.preservation.detach()
    ).float().reshape(())


def _snapshot_combined_loss_v1(
    value: CombinedStructuralActionAnchorLossV1,
) -> dict[str, Any]:
    torch = _torch()
    if (
        type(value.distillation) is not distillation.DistillationLossV1
        or value.distillation.schema_version != distillation.LOSS_SCHEMA
        or type(value._q_y) is not action_plan.ActionPlanOutput
        or type(value._q_y_receipt) is not dict
        or type(value._anchors) is not tuple
        or any(
            type(anchor) is not distillation.RoutedAnchorV1
            for anchor in value._anchors
        )
        or _distillation_config_record_v1(value._config)
        != value.distillation_config
    ):
        raise ActionAnchorRendererIntegrationError(
            "combined structural distillation provenance types differ"
        )
    distilled_tensors = {
        "distillation_total": value.distillation.total,
        "distillation_smooth_l1": value.distillation.smooth_l1,
        "distillation_cosine": value.distillation.cosine,
        "distillation_infonce": value.distillation.infonce,
        "distillation_preservation": value.distillation.preservation,
    }
    if any(
        type(tensor) is not torch.Tensor
        or tensor.dtype != torch.float32
        or tensor.numel() != 1
        or tensor.device != value.flow.device
        or not bool(torch.isfinite(tensor.detach()).item())
        for tensor in distilled_tensors.values()
    ):
        raise ActionAnchorRendererIntegrationError(
            "distillation components must be finite scalar FP32 tensors"
        )
    canonical_distillation = _canonical_distillation_total_detached_v1(
        value
    )
    if not bool(
        torch.equal(
            value.distillation.total.detach(), canonical_distillation
        )
    ):
        raise ActionAnchorRendererIntegrationError(
            "distillation total differs from its four configured components"
        )
    canonical = _canonical_combined_total_detached_v1(value)
    if not bool(torch.equal(value.total.detach(), canonical)):
        raise ActionAnchorRendererIntegrationError(
            "combined total differs from structural flow + distillation"
        )
    tensors = {
        "total": value.total,
        "flow": value.flow,
        **distilled_tensors,
    }
    if value._conditioner_regularizer_loss is not None:
        tensors["conditioner_regularizer"] = (
            value._conditioner_regularizer_loss
        )
    plans = (("q_y", value._q_y),) + tuple(
        (f"q_anchor[{index}]", anchor.plan)
        for index, anchor in enumerate(value._anchors)
    )
    plan_tensors = {
        f"{name}.phase": plan.phase_tokens
        for name, plan in plans
    }
    plan_tensors.update(
        {
            f"{name}.global": plan.global_token
            for name, plan in plans
        }
    )
    return {
        "metadata": {
            name: getattr(value, name)
            for name in (
                "schema_version",
                "training_authorized",
                "optimizer_step_authorized",
                "gradient_checkpointing_supported",
                "conditioner_regularizer_only",
                "structural_route_evidence_only",
                "real_renderer_flow_authorized",
                "sidecar_envelope_digest",
                "predictor_artifact_sha256",
                "conditioner_state_abi_sha256",
                "teacher_authority_sha256",
                "classification_authority_sha256",
                "row_ids",
                "source_token_tensor_sha256",
                "instruction_token_tensor_sha256",
                "q_pred_receipt_digest",
                "q_y_receipt_digest",
                "q_anchor_receipt_digests",
                "q_y_qualification_receipt_digests",
                "q_anchor_qualification_receipt_digests",
                "compatibility_decision_receipt_digests",
                "caller_observed_prediction_sha256",
                "expected_target_clean_sha256",
                "expected_noise_sha256",
                "canonical_flow_sha256",
                "canonical_total_sha256",
                "distillation_config",
                "conditioner_regularizer_sha256",
            )
        },
        "object_ids": {name: id(tensor) for name, tensor in tensors.items()},
        "versions": {
            name: _tensor_version(tensor) for name, tensor in tensors.items()
        },
        "sha256": {
            name: tensor_sha256_v1(tensor) for name, tensor in tensors.items()
        },
        "total_value": float(value.total.detach().item()),
        "canonical_total_sha256": tensor_sha256_v1(canonical),
        "canonical_distillation_sha256": tensor_sha256_v1(
            canonical_distillation
        ),
        "distillation_counts": tuple(
            getattr(value.distillation, name)
            for name in (
                "point_pair_count",
                "contrastive_positive_pair_count",
                "contrastive_negative_pair_count",
                "excluded_pair_count",
            )
        ),
        "plan_object_ids": {
            name: id(tensor) for name, tensor in plan_tensors.items()
        },
        "plan_versions": {
            name: _tensor_version(tensor)
            for name, tensor in plan_tensors.items()
        },
        "plan_sha256": {
            name: tensor_sha256_v1(tensor)
            for name, tensor in plan_tensors.items()
        },
        "q_y_receipt_sha256": object_sha256(value._q_y_receipt),
        "anchor_q_receipt_sha256": tuple(
            object_sha256(anchor.q_receipt) for anchor in value._anchors
        ),
        "anchor_compatibility_receipt_sha256": tuple(
            object_sha256(anchor.compatibility_receipt)
            for anchor in value._anchors
        ),
        "artifact_id": id(value._artifact),
        "recorder_id": id(value._recorder),
        "q_y_id": id(value._q_y),
        "q_y_receipt_id": id(value._q_y_receipt),
        "anchor_ids": tuple(id(anchor) for anchor in value._anchors),
        "config_id": id(value._config),
        "conditioner_regularizer_id": (
            None
            if value._conditioner_regularizer_loss is None
            else id(value._conditioner_regularizer_loss)
        ),
        "construction_nonce_id": id(value._construction_nonce),
    }


def _gradients_v1(loss: Any, inputs: Sequence[Any]) -> tuple[Any, ...]:
    if not inputs:
        return ()
    return tuple(
        _torch().autograd.grad(
            loss,
            tuple(inputs),
            allow_unused=True,
            retain_graph=True,
        )
    )


def _same_gradients_v1(left: Sequence[Any], right: Sequence[Any]) -> bool:
    torch = _torch()
    if len(left) != len(right):
        return False
    for left_gradient, right_gradient in zip(left, right):
        if left_gradient is None or right_gradient is None:
            if left_gradient is not None or right_gradient is not None:
                return False
            continue
        if (
            type(left_gradient) is not torch.Tensor
            or type(right_gradient) is not torch.Tensor
            or not bool(torch.isfinite(left_gradient.detach()).all().item())
            or not bool(torch.isfinite(right_gradient.detach()).all().item())
            or not bool(
                torch.equal(
                    left_gradient.detach(), right_gradient.detach()
                )
            )
        ):
            return False
    return True


def _replay_distillation_provenance_v1(
    value: CombinedStructuralActionAnchorLossV1,
    *,
    prepared: PreparedActionAnchorRendererV1,
    sidecar: Mapping[str, Any],
    audit_gradients: bool,
) -> None:
    torch = _torch()
    artifact = value._artifact
    regularizer = value._conditioner_regularizer_loss
    config_record = _distillation_config_record_v1(value._config)
    regularizer_sha = (
        None if regularizer is None else tensor_sha256_v1(regularizer)
    )
    expected_metadata = {
        "schema_version": COMBINED_LOSS_SCHEMA,
        "training_authorized": False,
        "optimizer_step_authorized": False,
        "gradient_checkpointing_supported": False,
        "conditioner_regularizer_only": True,
        "structural_route_evidence_only": True,
        "real_renderer_flow_authorized": False,
        "sidecar_envelope_digest": sidecar["envelope_digest"],
        "predictor_artifact_sha256": sidecar["predictor_artifact_sha256"],
        "conditioner_state_abi_sha256": sidecar[
            "conditioner_state_abi_sha256"
        ],
        "teacher_authority_sha256": sidecar["teacher_authority_sha256"],
        "classification_authority_sha256": sidecar[
            "classification_authority_sha256"
        ],
        "row_ids": tuple(sidecar["row_ids"]),
        "source_token_tensor_sha256": sidecar[
            "source_token_tensor_sha256"
        ],
        "instruction_token_tensor_sha256": sidecar[
            "instruction_token_tensor_sha256"
        ],
        "q_pred_receipt_digest": prepared.q_pred_receipt_digest,
        "q_y_receipt_digest": sidecar["q_y_receipt_digest"],
        "q_anchor_receipt_digests": tuple(
            sidecar["q_anchor_receipt_digests"]
        ),
        "q_y_qualification_receipt_digests": tuple(
            sidecar["q_y_qualification_receipt_digests"]
        ),
        "q_anchor_qualification_receipt_digests": tuple(
            tuple(per_anchor)
            for per_anchor in sidecar[
                "q_anchor_qualification_receipt_digests"
            ]
        ),
        "compatibility_decision_receipt_digests": tuple(
            sidecar["compatibility_decision_receipt_digests"]
        ),
        "caller_observed_prediction_sha256": (
            artifact.caller_observed_prediction_sha256
        ),
        "expected_target_clean_sha256": artifact.expected_target_clean_sha256,
        "expected_noise_sha256": artifact.expected_noise_sha256,
        "canonical_flow_sha256": tensor_sha256_v1(
            _canonical_structural_flow_detached_v1(artifact)
        ),
        "distillation_config": config_record,
        "conditioner_regularizer_sha256": regularizer_sha,
    }
    for name, expected in expected_metadata.items():
        if getattr(value, name, None) != expected:
            raise ActionAnchorRendererIntegrationError(
                f"combined {name} is not derived from prepared sidecar/artifact"
            )
    if (
        type(value._q_y_receipt) is not dict
        or value._q_y_receipt.get("receipt_digest")
        != sidecar["q_y_receipt_digest"]
        or len(value._anchors)
        != len(sidecar["q_anchor_receipt_digests"])
        or tuple(
            anchor.q_receipt.get("receipt_digest")
            for anchor in value._anchors
        )
        != tuple(sidecar["q_anchor_receipt_digests"])
        or tuple(
            anchor.compatibility_receipt.get("receipt_digest")
            for anchor in value._anchors
        )
        != tuple(sidecar["compatibility_decision_receipt_digests"])
    ):
        raise ActionAnchorRendererIntegrationError(
            "combined q_y/q_anchor receipt provenance differs from sidecar"
        )
    try:
        replayed = distillation.action_anchor_distillation_loss_v1(
            q_pred=prepared.q_pred_fp32,
            q_y=value._q_y,
            q_pred_receipt=prepared.q_pred_receipt,
            q_y_receipt=value._q_y_receipt,
            expected_teacher_authority_sha256=sidecar[
                "teacher_authority_sha256"
            ],
            expected_classification_authority_sha256=sidecar[
                "classification_authority_sha256"
            ],
            expected_q_y_qualification_receipt_digests=sidecar[
                "q_y_qualification_receipt_digests"
            ],
            expected_anchor_qualification_receipt_digests=sidecar[
                "q_anchor_qualification_receipt_digests"
            ],
            expected_compatibility_decision_receipt_digests=sidecar[
                "compatibility_decision_receipt_digests"
            ],
            anchors=value._anchors,
            preservation_loss=regularizer,
            config=value._config,
        )
    except Exception as error:
        raise ActionAnchorRendererIntegrationError(
            f"distillation provenance replay failed: {error}"
        ) from error
    distilled = value.distillation
    if (
        type(distilled) is not distillation.DistillationLossV1
        or distilled.schema_version != distillation.LOSS_SCHEMA
        or any(
            type(getattr(distilled, name)) is not int
            or getattr(distilled, name) < 0
            or getattr(distilled, name) != getattr(replayed, name)
            for name in (
                "point_pair_count",
                "contrastive_positive_pair_count",
                "contrastive_negative_pair_count",
                "excluded_pair_count",
            )
        )
    ):
        raise ActionAnchorRendererIntegrationError(
            "distillation type/schema/counts differ from receipt replay"
        )
    component_names = (
        "total",
        "smooth_l1",
        "cosine",
        "infonce",
        "preservation",
    )
    for name in component_names:
        actual = getattr(distilled, name)
        expected = getattr(replayed, name)
        if (
            type(actual) is not torch.Tensor
            or type(expected) is not torch.Tensor
            or actual.dtype != torch.float32
            or actual.numel() != 1
            or actual.device != artifact.flow.device
            or actual.requires_grad != expected.requires_grad
            or actual.is_leaf != expected.is_leaf
            or (actual.grad_fn is None) != (expected.grad_fn is None)
            or not bool(torch.isfinite(actual.detach()).item())
            or not bool(torch.equal(actual.detach(), expected.detach()))
            or tensor_sha256_v1(actual) != tensor_sha256_v1(expected)
        ):
            raise ActionAnchorRendererIntegrationError(
                f"distillation {name} differs from exact input/config replay"
            )
    canonical_distillation = _canonical_distillation_total_detached_v1(
        value
    )
    canonical_total = _canonical_combined_total_detached_v1(value)
    expected_total_sha = tensor_sha256_v1(canonical_total)
    if (
        not bool(torch.equal(distilled.total.detach(), canonical_distillation))
        or not bool(torch.equal(value.total.detach(), canonical_total))
        or value.canonical_total_sha256 != expected_total_sha
    ):
        raise ActionAnchorRendererIntegrationError(
            "combined/distillation configured formulas differ"
        )
    if not audit_gradients:
        return
    recorder = value._recorder
    recorder.begin_internal_vjp()
    try:
        if float(value._config.preservation_weight) > 0.0:
            if (
                type(regularizer) is not torch.Tensor
                or not regularizer.is_floating_point()
                or regularizer.numel() != 1
                or regularizer.device
                != prepared.q_pred_fp32.phase_tokens.device
                or not regularizer.requires_grad
                or not bool(torch.isfinite(regularizer.detach()).item())
            ):
                raise ActionAnchorRendererIntegrationError(
                    "positive conditioner regularizer provenance differs"
                )
            regularizer_gradients = _gradients_v1(
                regularizer, recorder.allowed_trainables
            )
            connected_regularizer_gradients = tuple(
                gradient
                for gradient in regularizer_gradients
                if gradient is not None
            )
            if (
                not connected_regularizer_gradients
                or any(
                    not bool(torch.isfinite(gradient.detach()).all().item())
                    for gradient in connected_regularizer_gradients
                )
                or not any(
                    bool(torch.count_nonzero(gradient.detach()).item() > 0)
                    for gradient in connected_regularizer_gradients
                )
            ):
                raise ActionAnchorRendererIntegrationError(
                    "conditioner regularizer is not nonconstant on allowed trainables"
                )
        total_edges = _gradients_v1(
            value.total, (value.flow, distilled.total)
        )
        if (
            len(total_edges) != 2
            or any(gradient is None for gradient in total_edges)
            or any(
                not bool(
                    torch.equal(
                        gradient.detach(),
                        torch.ones_like(gradient.detach()),
                    )
                )
                for gradient in total_edges
            )
        ):
            raise ActionAnchorRendererIntegrationError(
                "combined total lacks unit flow/distillation graph edges"
            )
        components = (
            distilled.smooth_l1,
            distilled.cosine,
            distilled.infonce,
            distilled.preservation,
        )
        component_edges = _gradients_v1(distilled.total, components)
        expected_weights = (
            float(value._config.smooth_l1_weight),
            float(value._config.cosine_weight),
            float(value._config.infonce_weight),
            float(value._config.preservation_weight),
        )
        if len(component_edges) != 4 or any(
            gradient is None
            or not bool(torch.isfinite(gradient.detach()).all().item())
            or not bool(
                torch.equal(
                    gradient.detach(),
                    torch.full_like(gradient.detach(), expected_weight),
                )
            )
            for gradient, expected_weight in zip(
                component_edges, expected_weights
            )
        ):
            raise ActionAnchorRendererIntegrationError(
                "distillation total lacks exact configured component graph edges"
            )
        q_pred_tensors = (
            prepared.q_pred_fp32.phase_tokens,
            prepared.q_pred_fp32.global_token,
        )
        for name in component_names:
            actual_gradients = _gradients_v1(
                getattr(distilled, name), q_pred_tensors
            )
            replayed_gradients = _gradients_v1(
                getattr(replayed, name), q_pred_tensors
            )
            if not _same_gradients_v1(
                actual_gradients, replayed_gradients
            ):
                raise ActionAnchorRendererIntegrationError(
                    f"distillation {name} q_pred gradient differs from replay"
                )
        trainable_gradients = _gradients_v1(
            distilled.total, recorder.allowed_trainables
        )
        replayed_trainable_gradients = _gradients_v1(
            replayed.total, recorder.allowed_trainables
        )
        if (
            not _same_gradients_v1(
                trainable_gradients, replayed_trainable_gradients
            )
            or not any(
                gradient is not None
                and bool(torch.count_nonzero(gradient.detach()).item() > 0)
                for gradient in trainable_gradients
            )
        ):
            raise ActionAnchorRendererIntegrationError(
                "distillation trainable gradients differ from replay"
            )
        predictor_gradients = _gradients_v1(
            distilled.total, recorder.predictor_trainables
        )
        if (
            not predictor_gradients
            or any(
                gradient is not None
                and not bool(torch.isfinite(gradient.detach()).all().item())
                for gradient in predictor_gradients
            )
            or not any(
                gradient is not None
                and bool(torch.count_nonzero(gradient.detach()).item() > 0)
                for gradient in predictor_gradients
            )
        ):
            raise ActionAnchorRendererIntegrationError(
                "distillation lacks a finite nonzero predictor gradient"
            )
        teacher_tensors = tuple(
            tensor
            for plan in (value._q_y,) + tuple(
                anchor.plan for anchor in value._anchors
            )
            for tensor in (plan.phase_tokens, plan.global_token)
            if tensor.requires_grad
        )
        if teacher_tensors and any(
            gradient is not None
            for gradient in _gradients_v1(distilled.total, teacher_tensors)
        ):
            raise ActionAnchorRendererIntegrationError(
                "distillation total carries a teacher gradient path"
            )
    finally:
        if recorder.state == "internal_vjp":
            recorder.end_internal_vjp()
    recorder._combined_dependency_audited = True


def _revalidate_combined_loss_v1(
    value: Any,
    *,
    allowed_states: Sequence[str],
    require_registered: bool = True,
) -> CombinedStructuralActionAnchorLossV1:
    if type(allowed_states) not in (list, tuple) or not allowed_states:
        raise ActionAnchorRendererIntegrationError(
            "combined state allowlist must be a nonempty exact list/tuple"
        )
    if type(value) is not CombinedStructuralActionAnchorLossV1:
        raise ActionAnchorRendererIntegrationError(
            "combined structural loss type differs"
        )
    if type(value.distillation) is not distillation.DistillationLossV1:
        raise ActionAnchorRendererIntegrationError(
            "combined distillation type differs"
        )
    current_distillation_tensors = tuple(
        getattr(value.distillation, name)
        for name in (
            "total",
            "smooth_l1",
            "cosine",
            "infonce",
            "preservation",
        )
    )
    recorder = value._recorder
    if (
        type(recorder) is not _RouteUseRecorderV1
        or value._construction_nonce is not recorder._combined_nonce
        or recorder.state not in tuple(allowed_states)
        or value._artifact is not recorder._registered_artifact
        or value.flow is not value._artifact.flow
        or (
            require_registered
            and (
                recorder._registered_combined is not value
                or recorder._registered_total_tensor is not value.total
                or recorder._registered_distillation is not value.distillation
                or recorder._registered_distillation_tensors is None
                or any(
                    registered is not current
                    for registered, current in zip(
                        recorder._registered_distillation_tensors,
                        current_distillation_tensors,
                    )
                )
                or len(recorder._registered_distillation_tensors) != 5
                or recorder._registered_q_y is not value._q_y
                or recorder._registered_q_y_receipt is not value._q_y_receipt
                or recorder._registered_anchors is not value._anchors
                or recorder._registered_conditioner_regularizer
                is not value._conditioner_regularizer_loss
                or recorder._registered_config is not value._config
                or recorder._combined_snapshot is None
            )
        )
        or (
            not require_registered
            and (
                recorder._registered_combined is not None
                or recorder._registered_distillation_tensors is not None
                or recorder._registered_q_y is not None
                or recorder._registered_q_y_receipt is not None
                or recorder._registered_anchors is not None
                or recorder._registered_conditioner_regularizer is not None
                or recorder._registered_config is not None
                or recorder._combined_snapshot is not None
            )
        )
    ):
        raise ActionAnchorRendererIntegrationError(
            "combined structural loss identity is forged, replaced, or out of order"
        )
    prepared, sidecar = _revalidate_prepared_v1(value._artifact._prepared)
    _revalidate_flow_artifact_v1(
        prepared,
        value._artifact,
        allowed_states=allowed_states,
    )
    _replay_distillation_provenance_v1(
        value,
        prepared=prepared,
        sidecar=sidecar,
        audit_gradients=recorder.state
        in ("artifact_armed", "combined_ready"),
    )
    try:
        current_snapshot = _snapshot_combined_loss_v1(value)
    except Exception as error:
        raise ActionAnchorRendererIntegrationError(
            f"combined total formula/snapshot changed: {error}"
        ) from error
    if require_registered and current_snapshot != recorder._combined_snapshot:
        raise ActionAnchorRendererIntegrationError(
            "combined total identity, version, raw hash, or formula changed"
        )
    if require_registered:
        _verify_factory_combined_issuance_v1(recorder, value)
    return value


def _combine_flow_and_action_anchor_loss_impl_v1(
    *,
    prepared: PreparedActionAnchorRendererV1,
    flow_artifact: RendererFlowArtifactV1,
    q_y: action_plan.ActionPlanOutput,
    q_y_receipt: Mapping[str, Any],
    anchors: Sequence[distillation.RoutedAnchorV1] = (),
    conditioner_regularizer_loss: Any | None,
    config: distillation.DistillationLossConfigV1 | None = None,
    _factory_issue: Any,
) -> CombinedActionAnchorLossV1:
    """Combine a sealed renderer-flow artifact with action distillation."""

    torch = _torch()
    recorder = getattr(prepared, "_recorder", None)
    try:
        _checked, sidecar = _revalidate_prepared_v1(prepared)
        artifact = _revalidate_flow_artifact_v1(prepared, flow_artifact)
        resolved_config = config or distillation.DistillationLossConfigV1()
        if type(resolved_config) is not distillation.DistillationLossConfigV1:
            raise ActionAnchorRendererIntegrationError(
                "distillation config type differs"
            )
        resolved_config.validate()
        if type(anchors) not in (list, tuple):
            raise ActionAnchorRendererIntegrationError(
                "anchors must be an exact list or tuple"
            )
        if q_y_receipt.get("receipt_digest") != sidecar["q_y_receipt_digest"]:
            raise ActionAnchorRendererIntegrationError(
                "q_y receipt differs from pinned sidecar"
            )
        if [anchor.q_receipt.get("receipt_digest") for anchor in anchors] != sidecar[
            "q_anchor_receipt_digests"
        ]:
            raise ActionAnchorRendererIntegrationError(
                "q_anchor receipts differ from pinned sidecar"
            )
        if [
            anchor.compatibility_receipt.get("receipt_digest") for anchor in anchors
        ] != sidecar["compatibility_decision_receipt_digests"]:
            raise ActionAnchorRendererIntegrationError(
                "compatibility decisions differ from pinned sidecar"
            )
        recorder.begin_internal_vjp()
        if float(resolved_config.preservation_weight) > 0.0:
            if (
                type(conditioner_regularizer_loss) is not torch.Tensor
                or not conditioner_regularizer_loss.is_floating_point()
                or conditioner_regularizer_loss.numel() != 1
                or conditioner_regularizer_loss.device
                != prepared.q_pred_fp32.phase_tokens.device
                or not conditioner_regularizer_loss.requires_grad
                or not bool(torch.isfinite(
                    conditioner_regularizer_loss.detach()
                ).item())
            ):
                raise ActionAnchorRendererIntegrationError(
                    "positive structural regularizer weight requires a finite differentiable conditioner_regularizer_loss"
                )
            preservation_gradients = torch.autograd.grad(
                conditioner_regularizer_loss,
                recorder.allowed_trainables,
                allow_unused=True,
                retain_graph=True,
            )
            connected = [
                gradient
                for gradient in preservation_gradients
                if gradient is not None
            ]
            if (
                not connected
                or any(
                    not bool(torch.isfinite(gradient.detach()).all().item())
                    for gradient in connected
                )
                or not any(
                    bool(torch.count_nonzero(gradient.detach()).item() > 0)
                    for gradient in connected
                )
            ):
                raise ActionAnchorRendererIntegrationError(
                    "conditioner regularizer is not nonconstant on allowed conditioner trainables"
                )
        elif conditioner_regularizer_loss is not None and (
            type(conditioner_regularizer_loss) is not torch.Tensor
            or not conditioner_regularizer_loss.is_floating_point()
            or conditioner_regularizer_loss.numel() != 1
        ):
            raise ActionAnchorRendererIntegrationError(
                "disabled conditioner regularizer, when present, must still be a scalar tensor"
            )
        distilled = distillation.action_anchor_distillation_loss_v1(
            q_pred=prepared.q_pred_fp32,
            q_y=q_y,
            q_pred_receipt=prepared.q_pred_receipt,
            q_y_receipt=q_y_receipt,
            expected_teacher_authority_sha256=sidecar["teacher_authority_sha256"],
            expected_classification_authority_sha256=
            sidecar["classification_authority_sha256"],
            expected_q_y_qualification_receipt_digests=
            sidecar["q_y_qualification_receipt_digests"],
            expected_anchor_qualification_receipt_digests=
            sidecar["q_anchor_qualification_receipt_digests"],
            expected_compatibility_decision_receipt_digests=
            sidecar["compatibility_decision_receipt_digests"],
            anchors=anchors,
            preservation_loss=conditioner_regularizer_loss,
            config=resolved_config,
        )
        recorder.end_internal_vjp()
        total = artifact.flow + distilled.total
        if not bool(torch.isfinite(total.detach()).item()):
            raise ActionAnchorRendererIntegrationError("combined loss is non-finite")
        canonical_flow_sha = tensor_sha256_v1(
            _canonical_structural_flow_detached_v1(artifact)
        )
        canonical_total_sha = tensor_sha256_v1(total.detach())
        combined = _factory_issue(
            recorder=recorder,
            schema_version=COMBINED_LOSS_SCHEMA,
            total=total,
            flow=artifact.flow,
            distillation=distilled,
            training_authorized=False,
            optimizer_step_authorized=False,
            gradient_checkpointing_supported=False,
            conditioner_regularizer_only=True,
            structural_route_evidence_only=True,
            real_renderer_flow_authorized=False,
            sidecar_envelope_digest=sidecar["envelope_digest"],
            predictor_artifact_sha256=sidecar["predictor_artifact_sha256"],
            conditioner_state_abi_sha256=sidecar[
                "conditioner_state_abi_sha256"
            ],
            teacher_authority_sha256=sidecar["teacher_authority_sha256"],
            classification_authority_sha256=sidecar[
                "classification_authority_sha256"
            ],
            row_ids=tuple(sidecar["row_ids"]),
            source_token_tensor_sha256=sidecar[
                "source_token_tensor_sha256"
            ],
            instruction_token_tensor_sha256=sidecar[
                "instruction_token_tensor_sha256"
            ],
            q_pred_receipt_digest=prepared.q_pred_receipt_digest,
            q_y_receipt_digest=sidecar["q_y_receipt_digest"],
            q_anchor_receipt_digests=tuple(
                sidecar["q_anchor_receipt_digests"]
            ),
            q_y_qualification_receipt_digests=tuple(
                sidecar["q_y_qualification_receipt_digests"]
            ),
            q_anchor_qualification_receipt_digests=tuple(
                tuple(per_anchor)
                for per_anchor in sidecar[
                    "q_anchor_qualification_receipt_digests"
                ]
            ),
            compatibility_decision_receipt_digests=tuple(
                sidecar["compatibility_decision_receipt_digests"]
            ),
            caller_observed_prediction_sha256=(
                artifact.caller_observed_prediction_sha256
            ),
            expected_target_clean_sha256=artifact.expected_target_clean_sha256,
            expected_noise_sha256=artifact.expected_noise_sha256,
            canonical_flow_sha256=canonical_flow_sha,
            canonical_total_sha256=canonical_total_sha,
            distillation_config=_distillation_config_record_v1(
                resolved_config
            ),
            conditioner_regularizer_sha256=(
                None
                if conditioner_regularizer_loss is None
                else tensor_sha256_v1(conditioner_regularizer_loss)
            ),
            _artifact=artifact,
            _recorder=recorder,
            _q_y=q_y,
            _q_y_receipt=q_y_receipt,
            _anchors=tuple(anchors),
            _conditioner_regularizer_loss=conditioner_regularizer_loss,
            _config=resolved_config,
            _construction_nonce=recorder._combined_nonce,
        )
        recorder.mark_combined_ready()
        return combined
    except Exception as error:
        if type(recorder) is _RouteUseRecorderV1:
            recorder.abort()
        if isinstance(error, ActionAnchorRendererIntegrationError):
            raise
        raise ActionAnchorRendererIntegrationError(
            f"combined action-anchor loss rejected: {error}"
        ) from error


def _bind_combined_structural_loss_factory_v1(
    implementation: Any, factory_stage: Any
) -> Any:
    """Keep the combined-object issuance minter in the factory closure."""

    def issue(*, recorder: Any, **values: Any) -> Any:
        combined = CombinedStructuralActionAnchorLossV1(**values)
        factory_stage(recorder, combined)
        recorder.register_combined(combined)
        return combined

    def combine_flow_and_action_anchor_loss_v1(
        *,
        prepared: PreparedActionAnchorRendererV1,
        flow_artifact: RendererFlowArtifactV1,
        q_y: action_plan.ActionPlanOutput,
        q_y_receipt: Mapping[str, Any],
        anchors: Sequence[distillation.RoutedAnchorV1] = (),
        conditioner_regularizer_loss: Any | None,
        config: distillation.DistillationLossConfigV1 | None = None,
    ) -> CombinedActionAnchorLossV1:
        return implementation(
            prepared=prepared,
            flow_artifact=flow_artifact,
            q_y=q_y,
            q_y_receipt=q_y_receipt,
            anchors=anchors,
            conditioner_regularizer_loss=conditioner_regularizer_loss,
            config=config,
            _factory_issue=issue,
        )

    return combine_flow_and_action_anchor_loss_v1


combine_flow_and_action_anchor_loss_v1 = (
    _bind_combined_structural_loss_factory_v1(
        _combine_flow_and_action_anchor_loss_impl_v1,
        _stage_factory_combined_issuance_v1,
    )
)
del _bind_combined_structural_loss_factory_v1
del _stage_factory_combined_issuance_v1


def finalize_renderer_flow_backward_v1(
    value: CombinedActionAnchorLossV1,
) -> dict[str, Any]:
    """Finalize the unique combined backward; never authorize an optimizer step."""

    recorder = getattr(value, "_recorder", None)
    try:
        checked = _revalidate_combined_loss_v1(
            value, allowed_states=("awaiting_finalize",)
        )
        if (
            checked.schema_version != COMBINED_LOSS_SCHEMA
            or checked.training_authorized is not False
            or checked.optimizer_step_authorized is not False
            or checked.gradient_checkpointing_supported is not False
            or checked.conditioner_regularizer_only is not True
            or checked.structural_route_evidence_only is not True
            or checked.real_renderer_flow_authorized is not False
        ):
            raise ActionAnchorRendererIntegrationError(
                "combined structural loss cannot issue a backward receipt"
            )
        gradient_receipt = recorder.finalize_backward()
        unsigned = {
            **gradient_receipt,
            "action_plan_module_source_sha256": (
                PINNED_ACTION_PLAN_MODULE_SOURCE_SHA256
            ),
            "distillation_contract_sha256": (
                PINNED_DISTILLATION_CONTRACT_SHA256
            ),
            "distillation_module_source_sha256": (
                PINNED_DISTILLATION_MODULE_SOURCE_SHA256
            ),
            "sidecar_envelope_digest": checked.sidecar_envelope_digest,
            "predictor_artifact_sha256": checked.predictor_artifact_sha256,
            "predictor_artifact_hash_semantics": (
                _PREDICTOR_ARTIFACT_HASH_SEMANTICS
            ),
            "conditioner_state_abi_sha256": (
                checked.conditioner_state_abi_sha256
            ),
            "teacher_authority_sha256": checked.teacher_authority_sha256,
            "classification_authority_sha256": (
                checked.classification_authority_sha256
            ),
            "row_ids": list(checked.row_ids),
            "source_token_tensor_sha256": (
                checked.source_token_tensor_sha256
            ),
            "instruction_token_tensor_sha256": (
                checked.instruction_token_tensor_sha256
            ),
            "q_pred_receipt_digest": checked.q_pred_receipt_digest,
            "q_y_receipt_digest": checked.q_y_receipt_digest,
            "q_anchor_receipt_digests": list(
                checked.q_anchor_receipt_digests
            ),
            "q_y_qualification_receipt_digests": list(
                checked.q_y_qualification_receipt_digests
            ),
            "q_anchor_qualification_receipt_digests": [
                list(per_anchor)
                for per_anchor in checked.q_anchor_qualification_receipt_digests
            ],
            "compatibility_decision_receipt_digests": list(
                checked.compatibility_decision_receipt_digests
            ),
            "caller_observed_prediction_sha256": (
                checked.caller_observed_prediction_sha256
            ),
            "prediction_hash_semantics": (
                "caller-observed-identity-only-not-independent-producer-authority"
            ),
            "expected_target_clean_sha256": (
                checked.expected_target_clean_sha256
            ),
            "expected_noise_sha256": checked.expected_noise_sha256,
            "target_noise_hash_semantics": (
                "externally-supplied-expected-identities-not-sidecar-derived"
            ),
            "canonical_structural_flow_sha256": (
                checked.canonical_flow_sha256
            ),
            "canonical_combined_total_sha256": (
                checked.canonical_total_sha256
            ),
            "canonical_distillation_total_sha256": tensor_sha256_v1(
                _canonical_distillation_total_detached_v1(checked)
            ),
            "distillation_component_sha256": {
                name: tensor_sha256_v1(
                    getattr(checked.distillation, name)
                )
                for name in (
                    "total",
                    "smooth_l1",
                    "cosine",
                    "infonce",
                    "preservation",
                )
            },
            "distillation_counts": {
                name: getattr(checked.distillation, name)
                for name in (
                    "point_pair_count",
                    "contrastive_positive_pair_count",
                    "contrastive_negative_pair_count",
                    "excluded_pair_count",
                )
            },
            "distillation_config": {
                name: numeric
                for name, numeric in checked.distillation_config
            },
            "conditioner_regularizer_sha256": (
                checked.conditioner_regularizer_sha256
            ),
            "conditioner_regularizer_only": True,
            "production_renderer_preservation_artifact_implemented": False,
            "production_block_post_hook_route_required": True,
            "artifact_factory_issuance_verified": True,
            "combined_factory_issuance_verified": True,
            "factory_issuance_semantics": (
                "closure-held-one-shot-exact-identity-registry-not-construction-nonce"
            ),
        }
        receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
        _retire_factory_issuance_v1(recorder, checked._artifact, checked)
        return receipt
    except Exception as error:
        if type(recorder) is _RouteUseRecorderV1:
            _revoke_factory_issuance_v1(recorder)
        if type(recorder) is _RouteUseRecorderV1 and recorder.state != "consumed":
            recorder.abort()
        if isinstance(error, ActionAnchorRendererIntegrationError):
            raise
        raise ActionAnchorRendererIntegrationError(
            f"structural renderer backward receipt failed: {error}"
        ) from error


build_structural_renderer_probe_artifact_v1 = build_renderer_flow_artifact_v1
finalize_structural_renderer_probe_backward_v1 = (
    finalize_renderer_flow_backward_v1
)


@dataclass(frozen=True)
class EvaluationInterventionArmV1:
    name: str
    source_tokens: Any
    initial_target_hidden: Any
    route: action_plan.ActionPlanInjectionRoute


@dataclass(frozen=True)
class EvaluationInterventionRoutesV1:
    schema_version: str
    arms: tuple[EvaluationInterventionArmV1, ...]
    structural_only_not_video_science_gate: bool
    source_tensor_sha256: str
    target_tensor_sha256: str
    q_y_receipt_digest: str
    reverse_q_receipt_digest: str
    reverse_decision_receipt_digest: str


def build_evaluation_intervention_routes_v1(
    *,
    conditioner: Any,
    prepared: PreparedActionAnchorRendererV1,
    source_tokens: Any,
    target_hidden: Any,
    q_y_receipt: Mapping[str, Any],
    reverse_anchor: distillation.RoutedAnchorV1,
    expected_source_tensor_sha256: str,
    expected_target_tensor_sha256: str,
) -> EvaluationInterventionRoutesV1:
    """Construct correct/zero/shuffled/reverse routes for evaluation only."""

    torch = _torch()
    if type(conditioner) is not action_plan.ActionPlanConditionerV1 or type(
        prepared
    ) is not PreparedActionAnchorRendererV1:
        raise ActionAnchorRendererIntegrationError(
            "evaluation requires exact conditioner and prepared route types"
        )
    if type(source_tokens) is not torch.Tensor or type(target_hidden) is not torch.Tensor:
        raise ActionAnchorRendererIntegrationError(
            "evaluation source/target must be exact tensors"
        )
    _checked, sidecar = _revalidate_prepared_v1(prepared)
    if prepared._recorder.state != "cancelled" or tuple(
        conditioner.injection.projections
    ) != prepared._recorder.projections:
        raise ActionAnchorRendererIntegrationError(
            "structural evaluation requires a cancelled lease on the same conditioner"
        )
    expected_source = _authority_sha256(
        expected_source_tensor_sha256,
        label="externally expected evaluation source tensor digest",
    )
    expected_target = _authority_sha256(
        expected_target_tensor_sha256,
        label="externally expected evaluation target tensor digest",
    )
    if (
        tensor_sha256_v1(source_tokens) != expected_source
        or tensor_sha256_v1(target_hidden) != expected_target
        or expected_source == expected_target
    ):
        raise ActionAnchorRendererIntegrationError(
            "evaluation source/target tensor receipts differ"
        )
    if type(reverse_anchor) is not distillation.RoutedAnchorV1:
        raise ActionAnchorRendererIntegrationError(
            "reverse evaluation requires one receipt-bound RoutedAnchorV1"
        )
    reverse_digest = reverse_anchor.q_receipt.get("receipt_digest")
    matching = [
        index
        for index, digest in enumerate(sidecar["q_anchor_receipt_digests"])
        if digest == reverse_digest
    ]
    if len(matching) != 1:
        raise ActionAnchorRendererIntegrationError(
            "reverse q receipt is not uniquely pinned by the sidecar"
        )
    reverse_index = matching[0]
    if q_y_receipt.get("receipt_digest") != sidecar["q_y_receipt_digest"]:
        raise ActionAnchorRendererIntegrationError(
            "evaluation q_y receipt differs from the sidecar"
        )
    try:
        validated_reverse = distillation.validate_q_receipt_v1(
            reverse_anchor.q_receipt,
            plan=reverse_anchor.plan,
            expected_teacher_authority_sha256=sidecar[
                "teacher_authority_sha256"
            ],
            expected_qualification_receipt_digests=sidecar[
                "q_anchor_qualification_receipt_digests"
            ][reverse_index],
        )
        compatibility = distillation.validate_compatibility_receipt_v1(
            reverse_anchor.compatibility_receipt,
            q_y_receipt=q_y_receipt,
            q_anchor_receipt=validated_reverse,
            expected_teacher_authority_sha256=sidecar[
                "teacher_authority_sha256"
            ],
            expected_classification_authority_sha256=sidecar[
                "classification_authority_sha256"
            ],
            expected_q_y_qualification_receipt_digests=sidecar[
                "q_y_qualification_receipt_digests"
            ],
            expected_q_anchor_qualification_receipt_digests=sidecar[
                "q_anchor_qualification_receipt_digests"
            ][reverse_index],
            expected_decision_receipt_digest=sidecar[
                "compatibility_decision_receipt_digests"
            ][reverse_index],
        )
    except Exception as error:
        raise ActionAnchorRendererIntegrationError(
            f"reverse evaluation receipt validation failed: {error}"
        ) from error
    if any(
        item["candidate_kind"] != "reverse"
        or item["training_use"] != "contrastive-only"
        or item["contrastive_role"] != "negative"
        for item in compatibility["items"]
    ):
        raise ActionAnchorRendererIntegrationError(
            "evaluation reverse receipt is not an accepted reverse relation"
        )
    batch_size = int(prepared.q_pred_fp32.phase_tokens.shape[0])
    if batch_size < 2:
        raise ActionAnchorRendererIntegrationError(
            "shuffled evaluation requires at least two records"
        )
    _require_plan(
        reverse_anchor.plan,
        label="reverse q plan",
        batch_size=batch_size,
        dtype=torch.float32,
        device=prepared.q_pred_fp32.phase_tokens.device,
    )
    correct_fp32 = action_plan.ActionPlanOutput(
        phase_tokens=prepared.q_pred_fp32.phase_tokens.detach().clone().contiguous(),
        global_token=prepared.q_pred_fp32.global_token.detach().clone().contiguous(),
    )
    zero_fp32 = action_plan.ActionPlanOutput(
        phase_tokens=torch.zeros_like(correct_fp32.phase_tokens).contiguous(),
        global_token=torch.zeros_like(correct_fp32.global_token).contiguous(),
    )
    shuffled_fp32 = action_plan.ActionPlanOutput(
        phase_tokens=correct_fp32.phase_tokens.roll(1, dims=0).contiguous(),
        global_token=correct_fp32.global_token.roll(1, dims=0).contiguous(),
    )
    reverse_fp32 = action_plan.ActionPlanOutput(
        phase_tokens=reverse_anchor.plan.phase_tokens.detach().clone().contiguous(),
        global_token=reverse_anchor.plan.global_token.detach().clone().contiguous(),
    )
    plans = (correct_fp32, zero_fp32, shuffled_fp32, reverse_fp32)
    for left in range(len(plans)):
        for right in range(left + 1, len(plans)):
            if torch.equal(
                plans[left].phase_tokens, plans[right].phase_tokens
            ) and torch.equal(
                plans[left].global_token, plans[right].global_token
            ):
                raise ActionAnchorRendererIntegrationError(
                    "evaluation intervention plans must be pairwise distinct"
                )
    arms: list[EvaluationInterventionArmV1] = []
    for name, plan_fp32 in zip(_EVALUATION_ARM_NAMES, plans):
        renderer_plan = _cast_plan_with_graph(
            plan_fp32, dtype=target_hidden.dtype
        )
        route = conditioner.injection.bind_route(
            renderer_plan, prepared.ownership, audit_finite=True
        )
        arms.append(
            EvaluationInterventionArmV1(
                name=name,
                source_tokens=source_tokens,
                initial_target_hidden=target_hidden,
                route=route,
            )
        )
    return EvaluationInterventionRoutesV1(
        schema_version=EVALUATION_ROUTE_SCHEMA,
        arms=tuple(arms),
        structural_only_not_video_science_gate=True,
        source_tensor_sha256=expected_source,
        target_tensor_sha256=expected_target,
        q_y_receipt_digest=sidecar["q_y_receipt_digest"],
        reverse_q_receipt_digest=validated_reverse["receipt_digest"],
        reverse_decision_receipt_digest=compatibility["receipt_digest"],
    )


def apply_evaluation_intervention_routes_v1(
    *,
    conditioner: Any,
    routes: EvaluationInterventionRoutesV1,
) -> tuple[Any, ...]:
    """Traverse exactly 30 heads without gradients or mutation."""

    torch = _torch()
    if type(conditioner) is not action_plan.ActionPlanConditionerV1 or type(
        routes
    ) is not EvaluationInterventionRoutesV1:
        raise ActionAnchorRendererIntegrationError(
            "evaluation apply types differ"
        )
    if (
        routes.schema_version != EVALUATION_ROUTE_SCHEMA
        or routes.structural_only_not_video_science_gate is not True
        or tuple(arm.name for arm in routes.arms) != _EVALUATION_ARM_NAMES
    ):
        raise ActionAnchorRendererIntegrationError(
            "evaluation route semantics differ"
        )
    source = routes.arms[0].source_tokens
    target = routes.arms[0].initial_target_hidden
    if any(
        arm.source_tokens is not source or arm.initial_target_hidden is not target
        for arm in routes.arms
    ):
        raise ActionAnchorRendererIntegrationError(
            "evaluation interventions changed source/initial-target ownership"
        )
    if (
        tensor_sha256_v1(source) != routes.source_tensor_sha256
        or tensor_sha256_v1(target) != routes.target_tensor_sha256
    ):
        raise ActionAnchorRendererIntegrationError(
            "evaluation source/target changed after receipt binding"
        )
    block_indices = conditioner.injection.validate_block_traversal(
        list(range(action_plan.TRANSFORMER_BLOCK_COUNT))
    )
    outputs = []
    with torch.no_grad():
        for arm in routes.arms:
            hidden = arm.initial_target_hidden
            for block_index in block_indices:
                hidden = conditioner(
                    hidden, arm.route, block_index=block_index
                ).target_hidden
            outputs.append(hidden)
    return tuple(outputs)


__all__ = [
    "COMBINED_LOSS_SCHEMA",
    "CONDITIONER_REGULARIZER_ONLY",
    "EVALUATION_ROUTE_SCHEMA",
    "GRADIENT_CHECKPOINTING_SUPPORTED",
    "IMPLEMENTS_RENDERER",
    "IMPLEMENTS_TEACHER_QUALIFICATION",
    "LOCAL_ONLY",
    "NO_LAUNCH",
    "NO_TRAINING",
    "PINNED_DISTILLATION_CONTRACT_SHA256",
    "PINNED_DISTILLATION_MODULE_SOURCE_SHA256",
    "PINNED_ACTION_PLAN_MODULE_SOURCE_SHA256",
    "PREPARED_ROUTE_SCHEMA",
    "PRODUCTION_BLOCK_POST_HOOK_ROUTE_REQUIRED",
    "PRODUCTION_RENDERER_PRESERVATION_ARTIFACT_IMPLEMENTED",
    "REAL_RENDERER_FLOW_AUTHORIZED",
    "RENDERER_FLOW_ARTIFACT_SCHEMA",
    "RENDERER_FLOW_BACKWARD_RECEIPT_SCHEMA",
    "SCHEMA_VERSION",
    "SIDECAR_ENVELOPE_SCHEMA",
    "ActionAnchorRendererIntegrationError",
    "CombinedActionAnchorLossV1",
    "CombinedStructuralActionAnchorLossV1",
    "EvaluationInterventionArmV1",
    "EvaluationInterventionRoutesV1",
    "PreparedActionAnchorRendererV1",
    "RendererFlowArtifactV1",
    "StructuralRendererFlowArtifactV1",
    "STRUCTURAL_ROUTE_EVIDENCE_ONLY",
    "apply_evaluation_intervention_routes_v1",
    "build_evaluation_intervention_routes_v1",
    "build_renderer_flow_artifact_v1",
    "build_structural_renderer_probe_artifact_v1",
    "build_sidecar_envelope_v1",
    "cancel_prepared_renderer_route_v1",
    "canonical_json_bytes",
    "combine_flow_and_action_anchor_loss_v1",
    "finalize_renderer_flow_backward_v1",
    "finalize_structural_renderer_probe_backward_v1",
    "object_sha256",
    "prepare_action_anchor_renderer_v1",
    "tensor_sha256_v1",
    "validate_sidecar_envelope_v1",
]
