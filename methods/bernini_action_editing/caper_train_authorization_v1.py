#!/usr/bin/env python3
"""Single fail-closed CAPER optimizer-construction authorization boundary.

This module is the only CAPER API allowed to construct the fixed AdamW and run
its one-step training dry-run.  It performs an AND, not a fallback or union.
The live WORLD/SP/rank coordinate is read from torch.distributed and Bernini;
callers cannot inject an optimizer or parallel coordinate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any, Optional

import torch

import caper_phase_action_quotient_probe as paq
import caper_sigma_gated_target_row_lora as lora
import caper_stage1_preference_admission_v1 as stage1


SCHEMA_VERSION = "bernini-caper-train-authorization-v1"
COMMON_LEDGER_SCHEMA_VERSION = "bernini-caper-common-authority-ledger-v1"
DRY_RUN_RECEIPT_SCHEMA_VERSION = "bernini-caper-one-step-dry-run-v1"
ADAMW_LR = 1.0e-4
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1.0e-8
ADAMW_WEIGHT_DECAY = 0.0
ABSENT_OPTIMIZER_STATE = {
    "schema_version": "bernini-caper-absent-optimizer-state-v1",
    "optimizer_created": False,
    "state": None,
    "param_groups": None,
}


class CAPERTrainAuthorizationError(RuntimeError):
    """The sealed inputs or optimizer factory violated the authorization API."""


@dataclass(frozen=True)
class CAPERTrainAuthorization:
    optimizer: Optional[Any]
    authorized: bool
    receipt: Mapping[str, Any]


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(stage1.canonical_json_bytes(value)).hexdigest()


def _tensor_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        owned = value.detach().cpu().contiguous()
        # AdamW stores ``step`` as a zero-dimensional tensor.  Viewing a scalar
        # directly as bytes is rejected when the element size changes; flatten
        # first so scalar, empty and ordinary optimizer tensors share one
        # deterministic raw-storage path.
        payload = owned.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        return {
            "kind": "tensor",
            "dtype": str(owned.dtype),
            "shape": list(owned.shape),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _tensor_state(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_tensor_state(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    return {"kind": "opaque", "type": f"{type(value).__module__}.{type(value).__qualname__}"}


def optimizer_state_sha256(optimizer: Optional[Any]) -> str:
    if optimizer is None:
        return _object_sha256(ABSENT_OPTIMIZER_STATE)
    state_dict = getattr(optimizer, "state_dict", None)
    if not callable(state_dict):
        raise CAPERTrainAuthorizationError("optimizer must expose state_dict()")
    return _object_sha256(
        {
            "schema_version": "bernini-caper-present-optimizer-state-v1",
            "optimizer_created": True,
            "optimizer_type": (
                f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
            ),
            "state_dict": _tensor_state(state_dict()),
        }
    )


def _seal_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["authorization_receipt_sha256"] = _object_sha256(result)
    return result


def verify_authorization_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CAPERTrainAuthorizationError("authorization receipt must be an object")
    row = dict(value)
    seal = row.pop("authorization_receipt_sha256", None)
    if seal != _object_sha256(row):
        raise CAPERTrainAuthorizationError("authorization receipt seal differs")
    if row.get("schema_version") != SCHEMA_VERSION:
        raise CAPERTrainAuthorizationError("authorization receipt schema differs")
    authorized = row.get("authorized")
    if type(authorized) is not bool:
        raise CAPERTrainAuthorizationError("authorization bit must be boolean")
    pair_ids = row.get("authorized_optimizer_pair_ids")
    if not isinstance(pair_ids, list) or len(pair_ids) != len(set(pair_ids)):
        raise CAPERTrainAuthorizationError("authorized pair IDs differ")
    if authorized:
        if row.get("optimizer_created") is not True or not pair_ids:
            raise CAPERTrainAuthorizationError("positive receipt lacks optimizer/pairs")
        common = row.get("common_authority_ledger")
        if not isinstance(common, Mapping):
            raise CAPERTrainAuthorizationError("positive receipt lacks common ledger")
        common_body = dict(common)
        common_seal = common_body.pop("ledger_sha256", None)
        if (
            common_body.get("schema_version") != COMMON_LEDGER_SCHEMA_VERSION
            or common_seal != _object_sha256(common_body)
            or common_body.get("authorized_optimizer_pair_ids") != pair_ids
        ):
            raise CAPERTrainAuthorizationError("common authority ledger seal differs")
        dry_run = row.get("dry_run_receipt")
        if not isinstance(dry_run, Mapping):
            raise CAPERTrainAuthorizationError("positive receipt lacks dry-run receipt")
        dry_body = dict(dry_run)
        dry_seal = dry_body.pop("dry_run_receipt_sha256", None)
        if (
            dry_body.get("schema_version") != DRY_RUN_RECEIPT_SCHEMA_VERSION
            or dry_seal != _object_sha256(dry_body)
            or dry_body.get("only_lora_B_changed") is not True
            or not isinstance(dry_body.get("changed_parameter_names"), list)
            or not dry_body["changed_parameter_names"]
            or any(
                type(name) is not str
                or not name.endswith(".caper_lora_B.weight")
                for name in dry_body["changed_parameter_names"]
            )
        ):
            raise CAPERTrainAuthorizationError("dry-run receipt seal differs")
    elif not (
        row.get("optimizer_created") is False
        and pair_ids == []
        and row.get("common_authority_ledger") is None
        and row.get("optimizer_state_before_sha256")
        == row.get("optimizer_state_after_sha256")
        and row.get("caper_adapter_before_sha256")
        == row.get("caper_adapter_after_sha256")
    ):
        raise CAPERTrainAuthorizationError("negative receipt changed optimizer/A-B state")
    return dict(value)


def _adapter_sha256(handle: Any) -> str:
    if type(handle) is not lora.CAPERHandle:
        raise CAPERTrainAuthorizationError("caper_handle must be an exact CAPERHandle")
    handle.assert_scope()
    digest = handle.trainable_parameter_values_sha256()
    if not isinstance(digest, str) or len(digest) != 64:
        raise CAPERTrainAuthorizationError("CAPER A/B value checksum is unavailable")
    return digest


def _verify_paq_decision(
    decision: Any, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Reconstruct the PAQ decision seal and return common authority fields."""

    if type(decision) is not paq.PAQDecision:
        raise CAPERTrainAuthorizationError("PAQ decision must be the exact sealed type")
    required_manifest = {
        "schema_version",
        "probe_id",
        "checkpoint_sha256",
        "policy_sha256",
        "source_revision_sha256",
        "source_exposure_registry_sha256",
        "intervention_scale",
        "intervention_scale_bits",
        "phase_order",
        "anchor_formula",
        "records",
        "manifest_sha256",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != required_manifest:
        raise CAPERTrainAuthorizationError("PAQ manifest does not match its closed schema")
    unsigned_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if paq.object_sha256(unsigned_manifest) != manifest["manifest_sha256"]:
        raise CAPERTrainAuthorizationError("PAQ manifest seal differs")
    payload = {
        "schema_version": paq.DECISION_SCHEMA_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "status": decision.status,
        "observational_candidate_passed": decision.observational_candidate_passed,
        "causal_intervention_passed": decision.causal_intervention_passed,
        "admitted_code": decision.admitted_code,
        "training_updates_authorized": decision.training_updates_authorized,
        "parameter_updates_executed": decision.parameter_updates_executed,
        "reasons": list(decision.reasons),
        "observational_metrics": dict(decision.observational_metrics),
        "candidate_code_sha256": decision.candidate_code_sha256,
        "intervention_receipt_sha256": decision.intervention_receipt_sha256,
        "linear_decodability_is_admission_evidence": False,
    }
    if paq.object_sha256(payload) != decision.decision_receipt_sha256:
        raise CAPERTrainAuthorizationError("PAQ decision receipt seal differs")
    if not (
        decision.status == paq.ADMISSION_STATUS
        and decision.observational_candidate_passed is True
        and decision.causal_intervention_passed is True
        and decision.admitted_code is True
        and type(decision.training_updates_authorized) is int
        and decision.training_updates_authorized == 1
        and type(decision.parameter_updates_executed) is int
        and decision.parameter_updates_executed == 0
        and decision.reasons == ()
        and decision.candidate_code_sha256 is not None
        and decision.intervention_receipt_sha256 is not None
    ):
        raise CAPERTrainAuthorizationError("PAQ decision does not authorize one update")
    records = manifest["records"]
    if not isinstance(records, list) or not records:
        raise CAPERTrainAuthorizationError("PAQ manifest has no observations")
    action_families = {row.get("action_family_id") for row in records}
    requested_actions = {row.get("requested_action_id") for row in records}
    if len(action_families) != 1 or len(requested_actions) != 1:
        raise CAPERTrainAuthorizationError("PAQ action coordinate is not global")
    return {
        "paq_manifest_sha256": manifest["manifest_sha256"],
        "paq_decision_receipt_sha256": decision.decision_receipt_sha256,
        "checkpoint_tree_sha256": manifest["checkpoint_sha256"],
        "inference_contract_sha256": manifest["policy_sha256"],
        "source_revision_sha256": manifest["source_revision_sha256"],
        "exposure_ledger_artifact_sha256": manifest[
            "source_exposure_registry_sha256"
        ],
        "requested_action_id": next(iter(requested_actions)),
        "action_family_id": next(iter(action_families)),
    }


def _verify_route(value: Any) -> dict[str, Any]:
    if type(value) is not lora.CAPERRoute:
        raise CAPERTrainAuthorizationError(
            "route must be the exact authority-constructed CAPERRoute"
        )
    row = dict(value.receipt())
    digest = row.pop("digest", None)
    if digest != lora.object_sha256(row):
        raise CAPERTrainAuthorizationError("route receipt seal differs")
    pack = row.get("preference_pack_receipt")
    parallel = row.get("parallel_state_receipt")
    if not isinstance(pack, Mapping) or not isinstance(parallel, Mapping):
        raise CAPERTrainAuthorizationError("route nested receipts are missing")
    pack_body = dict(pack)
    pack_digest = pack_body.pop("digest", None)
    parallel_body = dict(parallel)
    parallel_digest = parallel_body.pop("digest", None)
    if (
        pack_digest != lora.object_sha256(pack_body)
        or parallel_digest != lora.object_sha256(parallel_body)
    ):
        raise CAPERTrainAuthorizationError("route nested receipt seal differs")
    if not (
        row.get("schema_version") == lora.SCHEMA_VERSION
        and row.get("adapter_active") is True
        and pack_body.get("layout") == "[S,y+,S,y-]"
        and pack_body.get("target_intervals") == 2
        and pack_body.get("target_selector_sha256")
        == row.get("target_selector_sha256")
    ):
        raise CAPERTrainAuthorizationError(
            "training route must be active canonical [S,y+,S,y-]"
        )
    return {
        "route_receipt_sha256": digest,
        "target_selector_sha256": row["target_selector_sha256"],
        "preference_pack_receipt_sha256": pack_digest,
        "parallel_state_receipt_sha256": parallel_digest,
    }


def _common_ledger(
    stage: Mapping[str, Any], paq_binding: Mapping[str, Any]
) -> dict[str, Any]:
    bindings = stage["bindings"]
    exact = (
        "checkpoint_tree_sha256",
        "inference_contract_sha256",
        "source_revision_sha256",
        "exposure_ledger_artifact_sha256",
    )
    for key in exact:
        if bindings.get(key) != paq_binding.get(key):
            raise CAPERTrainAuthorizationError(f"common authority mismatch: {key}")
    fit_actions = bindings.get("authorized_fit_action_families")
    if fit_actions != [paq_binding["action_family_id"]]:
        raise CAPERTrainAuthorizationError(
            "one PAQ action coordinate must exactly equal fit action families"
        )
    required_stage_digests = (
        "action_taxonomy_sha256",
        "reward_audit_protocol_sha256",
        "exposure_ledger_payload_sha256",
    )
    if any(not isinstance(bindings.get(key), str) for key in required_stage_digests):
        raise CAPERTrainAuthorizationError("Stage-1 authority ledger is incomplete")
    body = {
        "schema_version": COMMON_LEDGER_SCHEMA_VERSION,
        "stage1_materialization_sha256": stage["materialization_sha256"],
        "paq_manifest_sha256": paq_binding["paq_manifest_sha256"],
        "paq_decision_receipt_sha256": paq_binding[
            "paq_decision_receipt_sha256"
        ],
        "checkpoint_tree_sha256": bindings["checkpoint_tree_sha256"],
        "inference_contract_sha256": bindings["inference_contract_sha256"],
        "source_revision_sha256": bindings["source_revision_sha256"],
        "action_taxonomy_sha256": bindings["action_taxonomy_sha256"],
        "exposure_ledger_artifact_sha256": bindings[
            "exposure_ledger_artifact_sha256"
        ],
        "exposure_ledger_payload_sha256": bindings[
            "exposure_ledger_payload_sha256"
        ],
        "reward_audit_protocol_sha256": bindings[
            "reward_audit_protocol_sha256"
        ],
        "requested_action_id": paq_binding["requested_action_id"],
        "action_family_id": paq_binding["action_family_id"],
        "authorized_optimizer_pair_ids": list(
            stage["authorized_optimizer_pair_ids"]
        ),
    }
    return {**body, "ledger_sha256": _object_sha256(body)}


def _optimizer_owns_exact_parameters(
    optimizer: Any, parameters: Sequence[torch.nn.Parameter]
) -> None:
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list):
        raise CAPERTrainAuthorizationError("optimizer has no param_groups list")
    observed = [parameter for group in groups for parameter in group.get("params", [])]
    if len(observed) != len(parameters) or {id(item) for item in observed} != {
        id(item) for item in parameters
    }:
        raise CAPERTrainAuthorizationError("optimizer parameter closure differs from CAPER A/B")


def _parameter_value_sha256(parameter: torch.nn.Parameter) -> str:
    value = parameter.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def _parameter_inventory(handle: lora.CAPERHandle) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    rows = handle.trainable_named_parameters()
    names = [name for name, _ in rows]
    if (
        not rows
        or len(names) != len(set(names))
        or len({id(parameter) for _, parameter in rows}) != len(rows)
        or any(
            not name.endswith((".caper_lora_A.weight", ".caper_lora_B.weight"))
            for name in names
        )
        or any(parameter.grad is not None for _, parameter in rows)
    ):
        raise CAPERTrainAuthorizationError(
            "CAPER A/B parameter closure or clean-gradient state differs"
        )
    if any(
        binding.parameter.grad is not None
        for binding in handle.base_parameter_bindings
    ):
        raise CAPERTrainAuthorizationError("frozen base has a preset gradient")
    if handle.trainable_parameter_values_sha256() != handle.initial_trainable_parameter_sha256:
        raise CAPERTrainAuthorizationError("CAPER A/B do not equal their sealed initial values")
    for name, parameter in rows:
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            raise CAPERTrainAuthorizationError(f"CAPER initial value is non-finite: {name}")
        if name.endswith(".caper_lora_B.weight") and bool(
            torch.count_nonzero(parameter.detach()).item()
        ):
            raise CAPERTrainAuthorizationError("CAPER LoRA B must be exactly zero initially")
    return rows


def _fixed_adamw(parameters: Sequence[torch.nn.Parameter]) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        tuple(parameters),
        lr=ADAMW_LR,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        weight_decay=ADAMW_WEIGHT_DECAY,
        amsgrad=False,
        foreach=False,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )


def _live_route_from_spec(route: Any) -> lora.CAPERRoute:
    _verify_route(route)
    return lora.CAPERRoute.from_live_runtime_sigma(
        global_target_selector=route.global_target_selector,
        pack_segments=route.pack_segments,
        sigma_schedule_index=route.sigma_schedule_index,
        sigma=lora.sigma_strata.PINNED_POSITIVE_SIGMAS[route.sigma_schedule_index],
        enabled=route.enabled,
    )


def _run_one_step_dry_run(
    *,
    handle: lora.CAPERHandle,
    route: lora.CAPERRoute,
    optimizer: torch.optim.AdamW,
    forward: Callable[[], torch.Tensor],
    base_before: str,
) -> Mapping[str, Any]:
    rows = _parameter_inventory(handle)
    before = {name: _parameter_value_sha256(parameter) for name, parameter in rows}
    try:
        with handle.route(route):
            loss = forward()
            if (
                not isinstance(loss, torch.Tensor)
                or loss.numel() != 1
                or not loss.requires_grad
                or not bool(torch.isfinite(loss.detach()).all().item())
            ):
                raise CAPERTrainAuthorizationError(
                    "dry-run forward must return one finite differentiable scalar"
                )
            loss.backward()
        handle.assert_scope()
        certificate = handle.freeze_checksum_certificate()
        if certificate["frozen_transformer_current_sha256"] != base_before:
            raise CAPERTrainAuthorizationError("frozen base changed during forward/backward")
        if any(
            binding.parameter.grad is not None
            for binding in handle.base_parameter_bindings
        ):
            raise CAPERTrainAuthorizationError("dry-run produced a frozen-base gradient")
        gradients: list[dict[str, Any]] = []
        participating: list[str] = []
        changed_b_candidates: list[str] = []
        for name, parameter in rows:
            gradient = parameter.grad
            if gradient is None:
                continue
            participating.append(name)
            finite = bool(torch.isfinite(gradient).all().item())
            nonzero = int(torch.count_nonzero(gradient).item())
            gradients.append(
                {
                    "name": name,
                    "sha256": _tensor_state(gradient)["sha256"],
                    "finite": finite,
                    "nonzero_elements": nonzero,
                }
            )
            if not finite:
                raise CAPERTrainAuthorizationError("dry-run gradient is non-finite")
            if name.endswith(".caper_lora_A.weight") and nonzero:
                raise CAPERTrainAuthorizationError(
                    "zero-B initialization produced an illegal LoRA A gradient"
                )
            if name.endswith(".caper_lora_B.weight") and nonzero:
                changed_b_candidates.append(name)
        if not participating or not changed_b_candidates:
            raise CAPERTrainAuthorizationError(
                "dry-run did not produce a legal nonzero LoRA B gradient"
            )
        before_step = {
            name: _parameter_value_sha256(parameter) for name, parameter in rows
        }
        if before_step != before:
            raise CAPERTrainAuthorizationError(
                "dry-run forward/backward changed A/B values before optimizer step"
            )
        if optimizer.state:
            raise CAPERTrainAuthorizationError(
                "AdamW acquired state before its authorized first step"
            )
        # All checks that can invalidate the update happen before this sole step.
        optimizer.step()
        handle.assert_scope()
        certificate_after = handle.freeze_checksum_certificate()
        if certificate_after["frozen_transformer_current_sha256"] != base_before:
            raise CAPERTrainAuthorizationError("frozen base changed after optimizer step")
        after = {name: _parameter_value_sha256(parameter) for name, parameter in rows}
        changed = [name for name in before if before[name] != after[name]]
        if (
            not changed
            or any(not name.endswith(".caper_lora_B.weight") for name in changed)
            or any(name not in changed_b_candidates for name in changed)
        ):
            raise CAPERTrainAuthorizationError(
                "one AdamW step changed parameters outside legal LoRA B gradients"
            )
        route_receipt = dict(route.receipt())
        body = {
            "schema_version": DRY_RUN_RECEIPT_SCHEMA_VERSION,
            "loss": float(loss.detach().cpu().item()),
            "participating_parameter_names": participating,
            "gradient_receipts": gradients,
            "sp4_shard_route": route_receipt,
            "changed_parameter_names": changed,
            "only_lora_B_changed": True,
            "frozen_base_before_sha256": base_before,
            "frozen_base_after_sha256": certificate_after[
                "frozen_transformer_current_sha256"
            ],
        }
        return {**body, "dry_run_receipt_sha256": _object_sha256(body)}
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise


def authorize_caper_training_and_create_optimizer(
    *,
    paq_decision: Any,
    paq_manifest: Mapping[str, Any],
    stage1_materialization: Mapping[str, Any],
    caper_handle: Any,
    caper_route: Optional[Any],
    optimizer_factory: Optional[Callable[[Sequence[torch.nn.Parameter]], Any]] = None,
    dry_run_forward: Optional[Callable[[], torch.Tensor]] = None,
) -> CAPERTrainAuthorization:
    """Authorize fixed AdamW and execute exactly one fail-closed dry-run step.

    ``optimizer_factory`` remains only as a compatibility trap: passing one is
    rejected.  The module owns the exact AdamW construction.
    """

    stage = stage1.verify_materialization_seal(stage1_materialization)
    before_ab = _adapter_sha256(caper_handle)
    absent_before = optimizer_state_sha256(None)
    pair_ids = list(stage["authorized_optimizer_pair_ids"])
    if not pair_ids:
        after_ab = _adapter_sha256(caper_handle)
        absent_after = optimizer_state_sha256(None)
        if before_ab != after_ab or absent_before != absent_after:
            raise CAPERTrainAuthorizationError("zero-pair state changed")
        receipt = _seal_receipt(
            {
                "schema_version": SCHEMA_VERSION,
                "authorized": False,
                "reason": "stage1_has_no_legal_fit_pair",
                "stage1_materialization_sha256": stage[
                    "materialization_sha256"
                ],
                "paq_decision_receipt_sha256": None,
                "common_authority_ledger": None,
                "route_receipt_sha256": None,
                "caper_adapter_before_sha256": before_ab,
                "caper_adapter_after_sha256": after_ab,
                "optimizer_created": False,
                "optimizer_state_before_sha256": absent_before,
                "optimizer_state_after_sha256": absent_after,
                "authorized_optimizer_pair_ids": [],
            }
        )
        return CAPERTrainAuthorization(None, False, receipt)

    paq_binding = _verify_paq_decision(paq_decision, paq_manifest)
    common = _common_ledger(stage, paq_binding)
    live_route = _live_route_from_spec(caper_route)
    route = _verify_route(live_route)
    parallel = live_route.parallel_state
    if parallel.sequence_parallel_size != 4 or parallel.test_only:
        raise CAPERTrainAuthorizationError("training requires live native SP4")
    certificate = caper_handle.freeze_checksum_certificate()
    certificate_digest = certificate.get("digest")
    if not isinstance(certificate_digest, str):
        raise CAPERTrainAuthorizationError("CAPER freeze certificate is unsealed")
    if certificate.get("trainable_parameter_current_values_sha256") != before_ab:
        raise CAPERTrainAuthorizationError("CAPER certificate/A-B checksum differs")
    base_before = certificate["frozen_transformer_current_sha256"]
    inventory = _parameter_inventory(caper_handle)
    if optimizer_factory is not None:
        raise CAPERTrainAuthorizationError("external optimizer_factory is forbidden")
    if not callable(dry_run_forward):
        raise CAPERTrainAuthorizationError("dry_run_forward must be callable")
    parameters = tuple(parameter for _, parameter in inventory)
    optimizer = _fixed_adamw(parameters)
    _optimizer_owns_exact_parameters(optimizer, parameters)
    after_ab = _adapter_sha256(caper_handle)
    if after_ab != before_ab:
        raise CAPERTrainAuthorizationError("optimizer construction changed CAPER A/B")
    if any(parameter.grad is not None for parameter in parameters):
        raise CAPERTrainAuthorizationError("optimizer construction installed gradients")
    certificate_after_optimizer = caper_handle.freeze_checksum_certificate()
    if (
        certificate_after_optimizer["frozen_transformer_current_sha256"] != base_before
        or certificate_after_optimizer["trainable_parameter_current_values_sha256"] != before_ab
    ):
        raise CAPERTrainAuthorizationError("optimizer construction changed frozen/A-B state")
    dry_run = _run_one_step_dry_run(
        handle=caper_handle,
        route=live_route,
        optimizer=optimizer,
        forward=dry_run_forward,
        base_before=base_before,
    )
    optimizer_after = optimizer_state_sha256(optimizer)
    receipt = _seal_receipt(
        {
            "schema_version": SCHEMA_VERSION,
            "authorized": True,
            "reason": "paq_and_stage1_and_common_ledger_and_route_passed",
            "stage1_materialization_sha256": stage["materialization_sha256"],
            "paq_decision_receipt_sha256": paq_binding[
                "paq_decision_receipt_sha256"
            ],
            "common_authority_ledger": common,
            "route_receipt_sha256": route["route_receipt_sha256"],
            "live_parallel_state_receipt": dict(parallel.receipt()),
            "caper_freeze_certificate_sha256": certificate_digest,
            "caper_adapter_before_sha256": before_ab,
            "caper_adapter_after_sha256": caper_handle.trainable_parameter_values_sha256(),
            "optimizer_created": True,
            "optimizer_state_before_sha256": absent_before,
            "optimizer_state_after_sha256": optimizer_after,
            "authorized_optimizer_pair_ids": pair_ids,
            "optimizer_contract": {
                "type": "torch.optim.AdamW",
                "lr": ADAMW_LR,
                "betas": list(ADAMW_BETAS),
                "eps": ADAMW_EPS,
                "weight_decay": ADAMW_WEIGHT_DECAY,
            },
            "dry_run_receipt": dry_run,
        }
    )
    return CAPERTrainAuthorization(optimizer, True, receipt)


__all__ = [
    "ABSENT_OPTIMIZER_STATE",
    "ADAMW_BETAS",
    "ADAMW_EPS",
    "ADAMW_LR",
    "ADAMW_WEIGHT_DECAY",
    "CAPERTrainAuthorization",
    "CAPERTrainAuthorizationError",
    "COMMON_LEDGER_SCHEMA_VERSION",
    "DRY_RUN_RECEIPT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "authorize_caper_training_and_create_optimizer",
    "optimizer_state_sha256",
    "verify_authorization_receipt",
]
