#!/usr/bin/env python3
"""Global common-direction solver for PAIR-v7 Phase-A2 replication.

The historical PAIR-v7 solver accepts one event or one DP2 pair.  A shared
LoRA, however, cannot be authorized by independently projecting four cells
and averaging their answers.  This module keeps every unprojected row and
performs two nested read-only audits:

* four local cells: ``{fit, confirmation} x {s16, s35}``;
* one global solve over eight action rows and sixty-four identity rows.

The global result is GO only when its one safe direction remains a descent
direction for all eight action conditions, every K4 source/sigma/family group
has rank at least three, every local cell has identity rank at least eight,
and the global identity rank is at least sixteen.  No optimizer, parameter
mutation, gradient artifact, adapter, or action-success claim is produced.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

import torch

import pair_v7_dual_coordinate_nullspace_transport as core


METHOD_NAME = "bernini-pair-v7-multicondition-common-nullspace-transport"
RECEIPT_SCHEMA = "bernini-pair-v7-multicondition-common-nullspace-v1"
CELL_RECEIPT_SCHEMA = "bernini-pair-v7-multicondition-local-cell-v1"
ROUTING_DIAGNOSTIC_SCHEMA = "bernini-pair-v7-typed-routing-diagnostic-v1"
SUBSPACE_DIAGNOSTIC_SCHEMA = (
    "bernini-pair-v7-fit-to-confirmation-event-subspace-diagnostic-v1"
)
PRIMARY_PAIR_IDS = ("fit", "confirmation")
PRIMARY_SCHEDULE_INDICES = (16, 35)
EXPECTED_ACTION_CONDITION_COUNT = 8
EXPECTED_IDENTITY_PROBE_COUNT = 64
WORLD_INPUT_SCHEMA = "bernini-pair-v7-phase-a2-world-gradient-bank-input-v1"
SKETCHES_PER_GROUP = 4
MINIMUM_GROUP_RANK = 3
MINIMUM_CELL_RANK = 8
MINIMUM_GLOBAL_RANK = 16
_SAFE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,255}")


class PairV7MulticonditionError(RuntimeError):
    """The prospective Phase-A2 row set or common solve is ambiguous."""


@dataclass(frozen=True)
class ActionConditionGradient:
    condition_id: str
    pair_id: str
    source_sample_id: str
    schedule_index: int
    candidate_id: str
    action_family: str
    event_digest: str
    gradient_computation_receipt_digest: str
    gradient_by_parameter: Mapping[str, torch.Tensor]
    checkpoint_content_receipt_digest: str
    parameter_state_sha256: str


@dataclass(frozen=True)
class IdentityConditionProbe:
    pair_id: str
    source_sample_id: str
    schedule_index: int
    sketch_index: int
    probe: core.IdentityGradientProbe


@dataclass(frozen=True)
class MulticonditionTransportResult:
    global_transport: core.TransportResult
    primary_replication_go: bool
    receipt: Mapping[str, Any]


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_digest" in unsigned:
        raise PairV7MulticonditionError("receipt is already sealed")
    value = dict(unsigned)
    for name in (
        "global_population_go",
        "optimizer_authorized",
        "parameter_update_authorized",
        "action_success_claimed",
    ):
        if name in value and value[name] is not False:
            raise PairV7MulticonditionError(f"{name} must remain false")
        value[name] = False
    return {**value, "receipt_digest": core.object_sha256(value)}


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PairV7MulticonditionError(f"{label} must be lowercase SHA-256")
    return value


def _safe(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE.fullmatch(value) is None:
        raise PairV7MulticonditionError(f"{label} is unsafe")
    return value


def _gradient_digest(
    layout: core.GradientLayout, mapping: Mapping[str, torch.Tensor], *, label: str
) -> str:
    return core._tensor_sha256(layout.flatten(mapping, label=label).float())


def _mean_gradient(
    layout: core.GradientLayout, rows: Sequence[ActionConditionGradient]
) -> Mapping[str, torch.Tensor]:
    flats = [
        layout.flatten(row.gradient_by_parameter, label=row.condition_id)
        for row in rows
    ]
    return layout.unflatten(
        torch.stack(flats, dim=0).mean(dim=0), label="multicondition mean"
    )


def _rank_rows(
    *,
    layout: core.GradientLayout,
    rows: Sequence[IdentityConditionProbe],
    config: core.TransportConfig,
) -> tuple[list[Mapping[str, Any]], list[str]]:
    grouped: dict[tuple[str, str, int, str], list[IdentityConditionProbe]] = {}
    for row in rows:
        grouped.setdefault(
            (
                row.pair_id,
                row.source_sample_id,
                row.schedule_index,
                row.probe.family,
            ),
            [],
        ).append(row)
    expected_keys = {
        (pair, source, schedule, family)
        for pair, source in {(row.pair_id, row.source_sample_id) for row in rows}
        for schedule in PRIMARY_SCHEDULE_INDICES
        for family in core.REQUIRED_IDENTITY_FAMILIES
    }
    if set(grouped) != expected_keys:
        raise PairV7MulticonditionError("identity source/sigma/family closure differs")
    receipts: list[Mapping[str, Any]] = []
    failures: list[str] = []
    for key, probes in sorted(grouped.items()):
        if (
            len(probes) != SKETCHES_PER_GROUP
            or {row.sketch_index for row in probes} != set(range(SKETCHES_PER_GROUP))
            or len({row.probe.feature_sketch_sha256 for row in probes})
            != SKETCHES_PER_GROUP
            or len(
                {row.probe.source_coordinate_receipt_digest for row in probes}
            )
            != 1
        ):
            raise PairV7MulticonditionError(f"identity K4 closure differs: {key}")
        unit: list[torch.Tensor] = []
        for row in probes:
            flat = layout.flatten(
                row.probe.gradient_by_parameter,
                label=f"identity rank {row.probe.probe_id}",
            ).double()
            norm = torch.linalg.vector_norm(flat)
            if float(norm.item()) > config.minimum_identity_probe_norm:
                unit.append(flat / norm)
        if unit:
            matrix = torch.stack(unit, dim=0)
            eigenvalues = torch.linalg.eigvalsh(
                matrix @ matrix.transpose(0, 1)
            )
            largest = max(float(eigenvalues[-1].item()), 0.0)
            threshold = max(
                config.eigenvalue_absolute_tolerance,
                config.singular_value_relative_tolerance**2 * largest,
            )
            rank = int((eigenvalues > threshold).sum().item())
        else:
            eigenvalues = torch.empty(0, dtype=torch.float64)
            threshold = config.eigenvalue_absolute_tolerance
            rank = 0
        pair, source, schedule, family = key
        passed = rank >= MINIMUM_GROUP_RANK
        if not passed:
            failures.append(
                f"IDENTITY_GROUP_RANK_BELOW_3:{source}:s{schedule}:{family}"
            )
        receipts.append(
            {
                "pair_id": pair,
                "source_sample_id": source,
                "schedule_index": schedule,
                "family": family,
                "probe_count": len(probes),
                "effective_rank": rank,
                "minimum_required_rank": MINIMUM_GROUP_RANK,
                "rank_threshold": threshold,
                "gram_eigenvalues": [float(x) for x in eigenvalues.tolist()],
                "passed": passed,
            }
        )
    return receipts, failures


def _validate_world_input_receipt(
    *,
    receipt: Mapping[str, Any],
    actions: Sequence[ActionConditionGradient],
    identities: Sequence[IdentityConditionProbe],
    authority_digest: str,
    checkpoint_digest: str,
    state_digest: str,
) -> str:
    """Bind the CPU solver to the WORLD8-consensus measurement bank."""

    if not isinstance(receipt, Mapping):
        raise PairV7MulticonditionError("sealed WORLD input receipt is required")
    expected_keys = {
        "schema_version",
        "manifest_digest",
        "checkpoint_content_receipt_digest",
        "parameter_state_sha256",
        "action_condition_count",
        "identity_probe_count",
        "action_rows",
        "identity_rows",
        "identity_cross_family_coordinate_cells",
        "raw_gradient_values_persisted",
        "input_digest",
    }
    if set(receipt) != expected_keys:
        raise PairV7MulticonditionError("WORLD input receipt field set differs")
    unsigned = dict(receipt)
    declared = _sha(unsigned.pop("input_digest", None), label="WORLD input digest")
    if (
        core.object_sha256(unsigned) != declared
        or receipt.get("schema_version") != WORLD_INPUT_SCHEMA
        or receipt.get("manifest_digest") != authority_digest
        or receipt.get("checkpoint_content_receipt_digest") != checkpoint_digest
        or receipt.get("parameter_state_sha256") != state_digest
        or receipt.get("action_condition_count") != EXPECTED_ACTION_CONDITION_COUNT
        or receipt.get("identity_probe_count") != EXPECTED_IDENTITY_PROBE_COUNT
        or receipt.get("raw_gradient_values_persisted") is not False
    ):
        raise PairV7MulticonditionError("WORLD input receipt seal/binding differs")

    action_rows = []
    for row in actions:
        layout = core.GradientLayout.from_named_gradients(row.gradient_by_parameter)
        action_rows.append(
            {
                "condition_id": row.condition_id,
                "pair_id": row.pair_id,
                "source_sample_id": row.source_sample_id,
                "schedule_index": row.schedule_index,
                "candidate_id": row.candidate_id,
                "action_family": row.action_family,
                "event_digest": row.event_digest,
                "gradient_computation_receipt_digest": (
                    row.gradient_computation_receipt_digest
                ),
                "gradient_sha256": _gradient_digest(
                    layout,
                    row.gradient_by_parameter,
                    label=f"WORLD action {row.condition_id}",
                ),
            }
        )
    identity_rows = []
    coordinate_groups: dict[tuple[str, str, int], list[IdentityConditionProbe]] = {}
    for row in identities:
        layout = core.GradientLayout.from_named_gradients(
            row.probe.gradient_by_parameter
        )
        identity_rows.append(
            {
                "probe_id": row.probe.probe_id,
                "pair_id": row.pair_id,
                "source_sample_id": row.source_sample_id,
                "schedule_index": row.schedule_index,
                "family": row.probe.family,
                "sketch_index": row.sketch_index,
                "feature_sketch_sha256": row.probe.feature_sketch_sha256,
                "source_coordinate_receipt_digest": (
                    row.probe.source_coordinate_receipt_digest
                ),
                "gradient_computation_receipt_digest": (
                    row.probe.gradient_computation_receipt_digest
                ),
                "gradient_sha256": _gradient_digest(
                    layout,
                    row.probe.gradient_by_parameter,
                    label=f"WORLD identity {row.probe.probe_id}",
                ),
            }
        )
        coordinate_groups.setdefault(
            (row.pair_id, row.source_sample_id, row.schedule_index), []
        ).append(row)
    coordinate_cells = []
    for (pair, source, schedule), rows in sorted(coordinate_groups.items()):
        coordinates = {
            row.probe.source_coordinate_receipt_digest for row in rows
        }
        observed = {
            (row.probe.family, row.sketch_index) for row in rows
        }
        expected = {
            (family, sketch)
            for family in core.REQUIRED_IDENTITY_FAMILIES
            for sketch in range(SKETCHES_PER_GROUP)
        }
        if len(rows) != 8 or observed != expected or len(coordinates) != 1:
            raise PairV7MulticonditionError(
                "identity families do not share one sealed source coordinate"
            )
        coordinate = _sha(
            next(iter(coordinates)), label="identity source coordinate"
        )
        coordinate_cells.append(
            {
                "pair_id": pair,
                "source_sample_id": source,
                "schedule_index": schedule,
                "identity_family_count": len(core.REQUIRED_IDENTITY_FAMILIES),
                "identity_probe_count": len(rows),
                "source_coordinate_receipt_digest": coordinate,
                "cross_family_coordinate_consensus": True,
            }
        )
    if (
        receipt.get("action_rows")
        != sorted(action_rows, key=lambda row: row["condition_id"])
        or receipt.get("identity_rows")
        != sorted(identity_rows, key=lambda row: row["probe_id"])
        or receipt.get("identity_cross_family_coordinate_cells")
        != coordinate_cells
    ):
        raise PairV7MulticonditionError(
            "live 8x64 gradients differ from sealed WORLD input rows"
        )
    return declared


def _descent_rows(
    *,
    layout: core.GradientLayout,
    safe_gradient: Mapping[str, torch.Tensor],
    rows: Sequence[ActionConditionGradient],
    thresholds: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[str]]:
    safe = layout.flatten(safe_gradient, label="common safe gradient")
    safe_norm = float(torch.linalg.vector_norm(safe).item())
    receipts: list[Mapping[str, Any]] = []
    failures: list[str] = []
    for row in rows:
        gradient = layout.flatten(
            row.gradient_by_parameter, label=f"action {row.condition_id}"
        )
        norm = float(torch.linalg.vector_norm(gradient).item())
        dot = float(torch.dot(gradient, safe).item())
        cosine = dot / (norm * safe_norm) if norm > 0.0 and safe_norm > 0.0 else None
        passed = (
            dot > float(thresholds["minimum_action_descent_gain"])
            and cosine is not None
            and cosine >= float(thresholds["minimum_action_descent_cosine"])
        )
        if not passed:
            failures.append(f"PER_CONDITION_ACTION_DESCENT_FAILED:{row.condition_id}")
        receipts.append(
            {
                "condition_id": row.condition_id,
                "pair_id": row.pair_id,
                "source_sample_id": row.source_sample_id,
                "schedule_index": row.schedule_index,
                "candidate_id": row.candidate_id,
                "action_family": row.action_family,
                "component_gradient_sha256": _gradient_digest(
                    layout,
                    row.gradient_by_parameter,
                    label=f"digest {row.condition_id}",
                ),
                "component_gradient_norm": norm,
                "dot_with_safe_gradient": dot,
                "descent_cosine": cosine,
                "passed": passed,
            }
        )
    return receipts, failures


def _aggregate_provenance(
    *,
    aggregate: Mapping[str, torch.Tensor],
    authority_digest: str,
    label: str,
) -> core.ActionGradientProvenance:
    layout = core.GradientLayout.from_named_gradients(aggregate)
    return core.ActionGradientProvenance(
        candidate_ids=(f"{label}-aggregate",),
        action_families=(f"{label}-common-action",),
        event_digests=(core.object_sha256({"label": label, "authority": authority_digest}),),
        component_gradient_sha256=(
            _gradient_digest(layout, aggregate, label=f"{label} aggregate"),
        ),
        gradient_computation_receipt_digests=(authority_digest,),
        fit_only_geometry_authority_digest=authority_digest,
        aggregation="single_fit_only_geometry_event",
    )


def _project_rows(
    *,
    actions: Sequence[ActionConditionGradient],
    identities: Sequence[IdentityConditionProbe],
    authority_digest: str,
    checkpoint_digest: str,
    state_digest: str,
    label: str,
    minimum_rank: int,
    config: core.TransportConfig,
) -> tuple[core.TransportResult, Mapping[str, Any]]:
    layout = core.GradientLayout.from_named_gradients(
        actions[0].gradient_by_parameter
    )
    aggregate = _mean_gradient(layout, actions)
    transport = core.project_action_gradient_to_identity_nullspace(
        action_gradient_by_parameter=aggregate,
        action_gradient_provenance=_aggregate_provenance(
            aggregate=aggregate, authority_digest=authority_digest, label=label
        ),
        identity_probes=tuple(row.probe for row in identities),
        checkpoint_content_receipt_digest=checkpoint_digest,
        parameter_state_sha256=state_digest,
        config=config,
    )
    descent, failures = _descent_rows(
        layout=transport.layout,
        safe_gradient=transport.safe_gradient_by_parameter,
        rows=actions,
        thresholds=transport.receipt["thresholds"],
    )
    rank = int(transport.receipt["identity_effective_rank"])
    if rank < minimum_rank:
        failures.append(f"IDENTITY_EFFECTIVE_RANK_BELOW_{minimum_rank}")
    if not transport.geometry_authorized:
        failures.extend(transport.receipt["failure_codes"])
        failures.append("NULLSPACE_TRANSPORT_NO_GO")
    failures = sorted(set(failures))
    receipt = _seal(
        {
            "schema_version": CELL_RECEIPT_SCHEMA,
            "label": label,
            "geometry_audit_passed": not failures,
            "failure_codes": failures,
            "action_condition_count": len(actions),
            "identity_probe_count": len(identities),
            "identity_effective_rank": rank,
            "identity_minimum_effective_rank": minimum_rank,
            "per_condition_action_descent": descent,
            "transport_receipt_digest": transport.receipt["receipt_digest"],
            "checkpoint_content_receipt_digest": checkpoint_digest,
            "parameter_state_sha256": state_digest,
            "parameter_mutation_performed": False,
            "scientific_action_editing_success_claim": False,
        }
    )
    return transport, receipt


def _typed_routing_diagnostic(
    *,
    actions: Sequence[ActionConditionGradient],
    identities: Sequence[IdentityConditionProbe],
    authority_digest: str,
    checkpoint_digest: str,
    state_digest: str,
    config: core.TransportConfig,
) -> Mapping[str, Any]:
    """Audit disjoint parameter-routing hypotheses without exporting directions.

    Each group below represents an exactly disjoint parameter partition.  Its
    action rows are averaged and projected only against identity VJPs that can
    activate that same partition.  This is deliberately a receipt-only model
    selection diagnostic: even a passing routing hypothesis cannot authorize
    an optimizer or Phase-B update.

    ``pair_x_sigma`` is retained only as an oracle upper bound because the
    fit/confirmation split is not observable at deployment.  The deployable
    hypotheses are ``sigma``, ``action_family``, and
    ``sigma_x_action_family``.
    """

    family_by_source: dict[tuple[str, str], str] = {}
    for row in actions:
        key = (row.pair_id, row.source_sample_id)
        previous = family_by_source.setdefault(key, row.action_family)
        if previous != row.action_family:
            raise PairV7MulticonditionError(
                "one source event maps to multiple action families"
            )
    if len(family_by_source) != 4:
        raise PairV7MulticonditionError(
            "typed routing requires four source-to-action-family bindings"
        )
    if {
        (row.pair_id, row.source_sample_id) for row in identities
    } != set(family_by_source):
        raise PairV7MulticonditionError(
            "typed routing identity/source family closure differs"
        )

    modes = (
        (
            "sigma",
            lambda row: f"s{row.schedule_index}",
            lambda row: f"s{row.schedule_index}",
            8,
            True,
        ),
        (
            "action_family",
            lambda row: row.action_family,
            lambda row: family_by_source[(row.pair_id, row.source_sample_id)],
            8,
            True,
        ),
        (
            "sigma_x_action_family",
            lambda row: f"s{row.schedule_index}:{row.action_family}",
            lambda row: (
                f"s{row.schedule_index}:"
                f"{family_by_source[(row.pair_id, row.source_sample_id)]}"
            ),
            8,
            True,
        ),
        (
            "pair_x_sigma",
            lambda row: f"{row.pair_id}:s{row.schedule_index}",
            lambda row: f"{row.pair_id}:s{row.schedule_index}",
            MINIMUM_CELL_RANK,
            False,
        ),
    )
    mode_receipts: list[Mapping[str, Any]] = []
    for mode, action_key, identity_key, minimum_rank, deployable in modes:
        actions_by_group: dict[str, list[ActionConditionGradient]] = {}
        identities_by_group: dict[str, list[IdentityConditionProbe]] = {}
        for row in actions:
            actions_by_group.setdefault(action_key(row), []).append(row)
        for row in identities:
            identities_by_group.setdefault(identity_key(row), []).append(row)
        if set(actions_by_group) != set(identities_by_group):
            raise PairV7MulticonditionError(
                f"typed routing group closure differs: {mode}"
            )
        group_receipts: list[Mapping[str, Any]] = []
        for group_id in sorted(actions_by_group):
            _transport, group_receipt = _project_rows(
                actions=tuple(actions_by_group[group_id]),
                identities=tuple(identities_by_group[group_id]),
                authority_digest=authority_digest,
                checkpoint_digest=checkpoint_digest,
                state_digest=state_digest,
                label=f"typed-{mode}-{group_id}",
                minimum_rank=minimum_rank,
                config=config,
            )
            group_receipts.append(group_receipt)
            del _transport
        failures = [
            row["label"]
            for row in group_receipts
            if row["geometry_audit_passed"] is not True
        ]
        mode_receipts.append(
            {
                "route_mode": mode,
                "deployment_observable_route": deployable,
                "parameter_partitions_are_disjoint": True,
                "group_count": len(group_receipts),
                "group_action_condition_counts": [
                    row["action_condition_count"] for row in group_receipts
                ],
                "group_identity_probe_counts": [
                    row["identity_probe_count"] for row in group_receipts
                ],
                "all_partition_geometry_passed": not failures,
                "failed_partition_labels": failures,
                "partition_receipts": group_receipts,
            }
        )
    return _seal(
        {
            "schema_version": ROUTING_DIAGNOSTIC_SCHEMA,
            "diagnostic_only": True,
            "shared_measurement_bank_reused_without_remeasurement": True,
            "raw_gradient_or_safe_direction_persisted": False,
            "routing_modes": mode_receipts,
            "parameter_mutation_performed": False,
            "scientific_action_editing_success_claim": False,
        }
    )


def _orthonormal_safe_basis(
    vectors: Sequence[torch.Tensor], *, config: core.TransportConfig
) -> tuple[torch.Tensor, ...]:
    """Return a deterministic basis without persisting any source vector."""

    if not vectors:
        return ()
    maximum_norm = max(float(torch.linalg.vector_norm(row).item()) for row in vectors)
    threshold = max(
        config.minimum_action_norm,
        config.singular_value_relative_tolerance * maximum_norm,
    )
    basis: list[torch.Tensor] = []
    for vector in vectors:
        residual = vector.clone()
        # Two passes are cheap for at most four fit rows and reduce loss of
        # orthogonality when two safe action directions are nearly collinear.
        for _pass in range(2):
            for unit in basis:
                residual.sub_(unit, alpha=float(torch.dot(unit, residual).item()))
        norm = float(torch.linalg.vector_norm(residual).item())
        if norm > threshold:
            basis.append(residual / norm)
    return tuple(basis)


def _event_subspace_diagnostic(
    *,
    actions: Sequence[ActionConditionGradient],
    identities: Sequence[IdentityConditionProbe],
    authority_digest: str,
    checkpoint_digest: str,
    state_digest: str,
    config: core.TransportConfig,
) -> Mapping[str, Any]:
    """Audit fit-learned event subspaces on held-out confirmation sources.

    A static common direction can fail even when a small action subspace is
    reusable.  For each deployment-observable route, fit action gradients are
    intersected with fit identity tangent rows, then orthonormalized.  At a
    held-out source, this fixed fit basis is projected through that source's
    identity nullspace.  A confirmation action passes only if its live
    gradient has sufficient support in the resulting source-safe subspace.
    The confirmation action chooses coefficients but cannot add a basis
    vector.  This models an on-policy source-state router, not a static
    prompt-only expert and not a deployable checkpoint.
    """

    family_by_source = {
        (row.pair_id, row.source_sample_id): row.action_family for row in actions
    }
    modes = (
        (
            "global",
            lambda row: "all",
            lambda row: "all",
            MINIMUM_GLOBAL_RANK,
        ),
        (
            "sigma",
            lambda row: f"s{row.schedule_index}",
            lambda row: f"s{row.schedule_index}",
            8,
        ),
        (
            "action_family",
            lambda row: row.action_family,
            lambda row: family_by_source[(row.pair_id, row.source_sample_id)],
            8,
        ),
        (
            "sigma_x_action_family",
            lambda row: f"s{row.schedule_index}:{row.action_family}",
            lambda row: (
                f"s{row.schedule_index}:"
                f"{family_by_source[(row.pair_id, row.source_sample_id)]}"
            ),
            2 * MINIMUM_GROUP_RANK,
        ),
    )
    mode_receipts: list[Mapping[str, Any]] = []
    for mode, action_key, identity_key, minimum_rank in modes:
        fit_by_group: dict[str, list[ActionConditionGradient]] = {}
        confirmation_by_group: dict[str, list[ActionConditionGradient]] = {}
        identities_by_group: dict[str, list[IdentityConditionProbe]] = {}
        for row in actions:
            destination = fit_by_group if row.pair_id == "fit" else confirmation_by_group
            destination.setdefault(action_key(row), []).append(row)
        for row in identities:
            identities_by_group.setdefault(identity_key(row), []).append(row)
        if not (
            set(fit_by_group)
            == set(confirmation_by_group)
            == set(identities_by_group)
        ):
            raise PairV7MulticonditionError(
                f"event subspace route closure differs: {mode}"
            )

        group_receipts: list[Mapping[str, Any]] = []
        for group_id in sorted(fit_by_group):
            fit_rows = tuple(
                sorted(fit_by_group[group_id], key=lambda row: row.condition_id)
            )
            confirmation_rows = tuple(
                sorted(
                    confirmation_by_group[group_id],
                    key=lambda row: row.condition_id,
                )
            )
            identity_rows = tuple(identities_by_group[group_id])
            fit_identity_rows = tuple(
                row for row in identity_rows if row.pair_id == "fit"
            )
            layout = core.GradientLayout.from_named_gradients(
                fit_rows[0].gradient_by_parameter
            )
            safe_fit_vectors: list[torch.Tensor] = []
            fit_projection_receipts: list[Mapping[str, Any]] = []
            fit_identity_rank: int | None = None
            for row in fit_rows:
                transport = core.project_action_gradient_to_identity_nullspace(
                    action_gradient_by_parameter=row.gradient_by_parameter,
                    action_gradient_provenance=_aggregate_provenance(
                        aggregate=row.gradient_by_parameter,
                        authority_digest=authority_digest,
                        label=f"subspace-{mode}-{group_id}-{row.condition_id}",
                    ),
                    identity_probes=tuple(
                        item.probe for item in fit_identity_rows
                    ),
                    checkpoint_content_receipt_digest=checkpoint_digest,
                    parameter_state_sha256=state_digest,
                    config=config,
                )
                safe = transport.layout.flatten(
                    transport.safe_gradient_by_parameter,
                    label=f"safe fit {row.condition_id}",
                )
                safe_fit_vectors.append(safe)
                observed_rank = int(
                    transport.receipt["identity_effective_rank"]
                )
                if fit_identity_rank is None:
                    fit_identity_rank = observed_rank
                elif fit_identity_rank != observed_rank:
                    raise PairV7MulticonditionError(
                        "fit event-subspace identity rank differs"
                    )
                fit_projection_receipts.append(
                    {
                        "condition_id": row.condition_id,
                        "safe_projection_geometry_passed": (
                            transport.geometry_authorized
                        ),
                        "safe_gradient_sha256": core._tensor_sha256(safe),
                        "transport_receipt_digest": transport.receipt[
                            "receipt_digest"
                        ],
                    }
                )
            basis = _orthonormal_safe_basis(safe_fit_vectors, config=config)
            assert fit_identity_rank is not None
            confirmation_receipts: list[Mapping[str, Any]] = []
            for row in confirmation_rows:
                source_identity_rows = tuple(
                    item
                    for item in identity_rows
                    if (
                        item.pair_id == "confirmation"
                        and item.source_sample_id == row.source_sample_id
                        and item.schedule_index == row.schedule_index
                    )
                )
                if len(source_identity_rows) != 2 * SKETCHES_PER_GROUP:
                    raise PairV7MulticonditionError(
                        "confirmation source identity cell differs"
                    )
                source_projected_vectors: list[torch.Tensor] = []
                source_projection_receipts: list[Mapping[str, Any]] = []
                source_identity_rank: int | None = None
                for basis_index, unit in enumerate(basis):
                    basis_mapping = layout.unflatten(
                        unit, label="fit-learned event basis"
                    )
                    transport = core.project_action_gradient_to_identity_nullspace(
                        action_gradient_by_parameter=basis_mapping,
                        action_gradient_provenance=_aggregate_provenance(
                            aggregate=basis_mapping,
                            authority_digest=authority_digest,
                            label=(
                                f"source-project-{mode}-{group_id}-"
                                f"{row.condition_id}-b{basis_index}"
                            ),
                        ),
                        identity_probes=tuple(
                            item.probe for item in source_identity_rows
                        ),
                        checkpoint_content_receipt_digest=checkpoint_digest,
                        parameter_state_sha256=state_digest,
                        config=config,
                    )
                    source_projected = transport.layout.flatten(
                        transport.safe_gradient_by_parameter,
                        label=(
                            f"source-projected basis {row.condition_id} "
                            f"{basis_index}"
                        ),
                    )
                    source_projected_vectors.append(source_projected)
                    observed_rank = int(
                        transport.receipt["identity_effective_rank"]
                    )
                    if source_identity_rank is None:
                        source_identity_rank = observed_rank
                    elif source_identity_rank != observed_rank:
                        raise PairV7MulticonditionError(
                            "confirmation identity rank differs across basis"
                        )
                    source_projection_receipts.append(
                        {
                            "fit_basis_index": basis_index,
                            "source_projection_geometry_passed": (
                                transport.geometry_authorized
                            ),
                            "source_projected_basis_sha256": (
                                core._tensor_sha256(source_projected)
                            ),
                            "transport_receipt_digest": transport.receipt[
                                "receipt_digest"
                            ],
                        }
                    )
                source_basis = _orthonormal_safe_basis(
                    source_projected_vectors, config=config
                )
                action = layout.flatten(
                    row.gradient_by_parameter,
                    label=f"confirmation action {row.condition_id}",
                )
                action_norm = float(torch.linalg.vector_norm(action).item())
                coefficients = [
                    float(torch.dot(unit, action).item())
                    for unit in source_basis
                ]
                projected = torch.zeros_like(action)
                for coefficient, unit in zip(coefficients, source_basis):
                    projected.add_(unit, alpha=coefficient)
                projected_norm = float(torch.linalg.vector_norm(projected).item())
                cosine = (
                    projected_norm / action_norm if action_norm > 0.0 else None
                )
                dot = projected_norm * projected_norm
                maximum_identity_cosine = 0.0
                for identity in source_identity_rows:
                    probe = layout.flatten(
                        identity.probe.gradient_by_parameter,
                        label=f"subspace identity {identity.probe.probe_id}",
                    )
                    probe_norm = float(torch.linalg.vector_norm(probe).item())
                    if projected_norm > 0.0 and probe_norm > 0.0:
                        maximum_identity_cosine = max(
                            maximum_identity_cosine,
                            abs(float(torch.dot(probe, projected).item()))
                            / (probe_norm * projected_norm),
                        )
                passed = (
                    source_identity_rank is not None
                    and source_identity_rank >= 2 * MINIMUM_GROUP_RANK
                    and bool(source_basis)
                    and all(
                        item["source_projection_geometry_passed"]
                        for item in source_projection_receipts
                    )
                    and cosine is not None
                    and cosine >= config.minimum_action_descent_cosine
                    and dot > config.minimum_action_descent_gain
                    and maximum_identity_cosine <= config.maximum_identity_cosine
                )
                confirmation_receipts.append(
                    {
                        "condition_id": row.condition_id,
                        "source_sample_id": row.source_sample_id,
                        "schedule_index": row.schedule_index,
                        "action_family": row.action_family,
                        "action_gradient_norm": action_norm,
                        "safe_subspace_projection_norm": projected_norm,
                        "safe_subspace_coverage_cosine": cosine,
                        "action_dot_with_optimal_subspace_direction": dot,
                        "maximum_identity_cosine": maximum_identity_cosine,
                        "source_identity_probe_count": len(
                            source_identity_rows
                        ),
                        "source_identity_effective_rank": source_identity_rank,
                        "source_identity_minimum_effective_rank": (
                            2 * MINIMUM_GROUP_RANK
                        ),
                        "fit_learned_subspace_rank": len(basis),
                        "source_projected_subspace_rank": len(source_basis),
                        "coefficient_count": len(coefficients),
                        "source_projection_receipts": (
                            source_projection_receipts
                        ),
                        "passed": passed,
                    }
                )
            failure_codes: list[str] = []
            if fit_identity_rank < minimum_rank:
                failure_codes.append(
                    f"IDENTITY_EFFECTIVE_RANK_BELOW_{minimum_rank}"
                )
            if not basis:
                failure_codes.append("FIT_SAFE_SUBSPACE_EMPTY")
            if not all(
                row["safe_projection_geometry_passed"]
                for row in fit_projection_receipts
            ):
                failure_codes.append("FIT_SAFE_PROJECTION_NO_GO")
            failure_codes.extend(
                f"CONFIRMATION_SUBSPACE_COVERAGE_FAILED:{row['condition_id']}"
                for row in confirmation_receipts
                if row["passed"] is not True
            )
            group_receipts.append(
                {
                    "group_id": group_id,
                    "fit_action_condition_count": len(fit_rows),
                    "confirmation_action_condition_count": len(
                        confirmation_rows
                    ),
                    "fit_identity_probe_count": len(fit_identity_rows),
                    "fit_identity_effective_rank": fit_identity_rank,
                    "fit_identity_minimum_effective_rank": minimum_rank,
                    "fit_learned_safe_subspace_rank": len(basis),
                    "fit_projection_receipts": fit_projection_receipts,
                    "confirmation_receipts": confirmation_receipts,
                    "geometry_audit_passed": not failure_codes,
                    "failure_codes": sorted(set(failure_codes)),
                }
            )
        failed_groups = [
            row["group_id"]
            for row in group_receipts
            if row["geometry_audit_passed"] is not True
        ]
        mode_receipts.append(
            {
                "route_mode": mode,
                "fit_learned_basis": True,
                "confirmation_action_added_basis_vector": False,
                "confirmation_live_action_selects_signed_coefficients": True,
                "fit_identity_rows_constrain_learned_basis": True,
                "confirmation_identity_rows_project_basis_per_source": True,
                "group_count": len(group_receipts),
                "all_group_geometry_passed": not failed_groups,
                "failed_group_ids": failed_groups,
                "group_receipts": group_receipts,
            }
        )
    return _seal(
        {
            "schema_version": SUBSPACE_DIAGNOSTIC_SCHEMA,
            "diagnostic_only": True,
            "representation_hypothesis": (
                "fit_learned_event_subspace_intersected_with_live_source_identity_nullspace"
            ),
            "shared_measurement_bank_reused_without_remeasurement": True,
            "raw_gradient_basis_or_direction_persisted": False,
            "routing_modes": mode_receipts,
            "parameter_mutation_performed": False,
            "scientific_action_editing_success_claim": False,
        }
    )


def solve_multicondition_common_direction(
    *,
    action_conditions: Sequence[ActionConditionGradient],
    identity_conditions: Sequence[IdentityConditionProbe],
    multicondition_authority_digest: str,
    validated_measurement_input_receipt: Mapping[str, Any],
    config: core.TransportConfig = core.TransportConfig(),
) -> MulticonditionTransportResult:
    """Solve and seal the fixed 4-cell Phase-A2 common direction."""

    config.validate()
    authority_digest = _sha(
        multicondition_authority_digest, label="multicondition authority"
    )
    actions = tuple(action_conditions)
    identities = tuple(identity_conditions)
    if (
        len(actions) != EXPECTED_ACTION_CONDITION_COUNT
        or len(identities) != EXPECTED_IDENTITY_PROBE_COUNT
        or any(not isinstance(row, ActionConditionGradient) for row in actions)
        or any(not isinstance(row, IdentityConditionProbe) for row in identities)
    ):
        raise PairV7MulticonditionError("Phase-A2 row count/type closure differs")
    condition_ids = []
    sources_by_pair: dict[str, set[str]] = {pair: set() for pair in PRIMARY_PAIR_IDS}
    checkpoints: set[str] = set()
    states: set[str] = set()
    first_layout: core.GradientLayout | None = None
    for row in actions:
        condition_ids.append(_safe(row.condition_id, label="condition ID"))
        pair = _safe(row.pair_id, label="pair ID")
        source = _safe(row.source_sample_id, label="source sample ID")
        _safe(row.candidate_id, label="candidate ID")
        _safe(row.action_family, label="action family")
        _sha(row.event_digest, label="event digest")
        _sha(
            row.gradient_computation_receipt_digest,
            label="action gradient receipt",
        )
        checkpoints.add(
            _sha(
                row.checkpoint_content_receipt_digest,
                label="checkpoint content receipt",
            )
        )
        states.add(_sha(row.parameter_state_sha256, label="parameter state"))
        if pair not in PRIMARY_PAIR_IDS or row.schedule_index not in PRIMARY_SCHEDULE_INDICES:
            raise PairV7MulticonditionError("action condition is outside preregistration")
        sources_by_pair[pair].add(source)
        layout = core.GradientLayout.from_named_gradients(row.gradient_by_parameter)
        if first_layout is None:
            first_layout = layout
        elif layout.manifest() != first_layout.manifest() or layout.device != first_layout.device:
            raise PairV7MulticonditionError("action gradient layout differs")
    if (
        len(set(condition_ids)) != len(condition_ids)
        or set(sources_by_pair) != set(PRIMARY_PAIR_IDS)
        or any(len(values) != 2 for values in sources_by_pair.values())
        or len(checkpoints) != 1
        or len(states) != 1
    ):
        raise PairV7MulticonditionError("action condition factorial closure differs")
    expected_action_keys = {
        (pair, source, schedule)
        for pair, sources in sources_by_pair.items()
        for source in sources
        for schedule in PRIMARY_SCHEDULE_INDICES
    }
    if {(r.pair_id, r.source_sample_id, r.schedule_index) for r in actions} != expected_action_keys:
        raise PairV7MulticonditionError("action 2-pair x 2-sigma factorial differs")

    checkpoint_digest = next(iter(checkpoints))
    state_digest = next(iter(states))
    expected_identity_keys = {
        (pair, source, schedule, family, sketch)
        for pair, sources in sources_by_pair.items()
        for source in sources
        for schedule in PRIMARY_SCHEDULE_INDICES
        for family in core.REQUIRED_IDENTITY_FAMILIES
        for sketch in range(SKETCHES_PER_GROUP)
    }
    observed_identity_keys = set()
    for row in identities:
        if row.pair_id not in PRIMARY_PAIR_IDS or row.schedule_index not in PRIMARY_SCHEDULE_INDICES:
            raise PairV7MulticonditionError("identity condition is outside preregistration")
        row.probe.validate_metadata()
        if (
            row.source_sample_id not in sources_by_pair[row.pair_id]
            or row.probe.checkpoint_content_receipt_digest != checkpoint_digest
            or row.probe.parameter_state_sha256 != state_digest
            or not 0 <= row.sketch_index < SKETCHES_PER_GROUP
        ):
            raise PairV7MulticonditionError("identity condition binding differs")
        observed_identity_keys.add(
            (
                row.pair_id,
                row.source_sample_id,
                row.schedule_index,
                row.probe.family,
                row.sketch_index,
            )
        )
    if observed_identity_keys != expected_identity_keys:
        raise PairV7MulticonditionError("identity 4-source x 2-sigma x 2-family x K4 closure differs")

    world_input_digest = _validate_world_input_receipt(
        receipt=validated_measurement_input_receipt,
        actions=actions,
        identities=identities,
        authority_digest=authority_digest,
        checkpoint_digest=checkpoint_digest,
        state_digest=state_digest,
    )

    assert first_layout is not None
    rank_rows, rank_failures = _rank_rows(
        layout=first_layout, rows=identities, config=config
    )
    cell_receipts: list[Mapping[str, Any]] = []
    cell_failures: list[str] = []
    for pair in PRIMARY_PAIR_IDS:
        for schedule in PRIMARY_SCHEDULE_INDICES:
            label = f"{pair}-s{schedule}"
            cell_actions = tuple(
                row
                for row in actions
                if row.pair_id == pair and row.schedule_index == schedule
            )
            cell_identities = tuple(
                row
                for row in identities
                if row.pair_id == pair and row.schedule_index == schedule
            )
            _transport, cell_receipt = _project_rows(
                actions=cell_actions,
                identities=cell_identities,
                authority_digest=authority_digest,
                checkpoint_digest=checkpoint_digest,
                state_digest=state_digest,
                label=label,
                minimum_rank=MINIMUM_CELL_RANK,
                config=config,
            )
            cell_receipts.append(cell_receipt)
            if cell_receipt["geometry_audit_passed"] is not True:
                cell_failures.append(f"LOCAL_CELL_NO_GO:{label}")

    global_transport, global_receipt = _project_rows(
        actions=actions,
        identities=identities,
        authority_digest=authority_digest,
        checkpoint_digest=checkpoint_digest,
        state_digest=state_digest,
        label="primary-global-8x64",
        minimum_rank=MINIMUM_GLOBAL_RANK,
        config=config,
    )
    typed_routing_diagnostic = _typed_routing_diagnostic(
        actions=actions,
        identities=identities,
        authority_digest=authority_digest,
        checkpoint_digest=checkpoint_digest,
        state_digest=state_digest,
        config=config,
    )
    event_subspace_diagnostic = _event_subspace_diagnostic(
        actions=actions,
        identities=identities,
        authority_digest=authority_digest,
        checkpoint_digest=checkpoint_digest,
        state_digest=state_digest,
        config=config,
    )
    failures = sorted(
        set(
            rank_failures
            + cell_failures
            + ([] if global_receipt["geometry_audit_passed"] else ["GLOBAL_COMMON_DIRECTION_NO_GO"])
        )
    )
    passed = not failures
    receipt = _seal(
        {
            "schema_version": RECEIPT_SCHEMA,
            "method_name": METHOD_NAME,
            "primary_replication_go": passed,
            "geometry_audit_passed": passed,
            "failure_codes": failures,
            "primary_pair_ids": list(PRIMARY_PAIR_IDS),
            "primary_schedule_indices": list(PRIMARY_SCHEDULE_INDICES),
            "pilot_schedule_index_33_included_in_primary_gate": False,
            "action_condition_count": len(actions),
            "identity_probe_count": len(identities),
            "identity_group_rank_gates": rank_rows,
            "local_cell_receipts": cell_receipts,
            "global_common_direction_receipt": global_receipt,
            "global_transport_receipt_digest": global_transport.receipt[
                "receipt_digest"
            ],
            "typed_routing_diagnostic": typed_routing_diagnostic,
            "event_subspace_diagnostic": event_subspace_diagnostic,
            "multicondition_authority_digest": authority_digest,
            "validated_world_input_digest": world_input_digest,
            "checkpoint_content_receipt_digest": checkpoint_digest,
            "parameter_state_sha256": state_digest,
            "parameter_mutation_performed": False,
            "gradient_or_adapter_artifact_written": False,
            "scientific_action_editing_success_claim": False,
        }
    )
    return MulticonditionTransportResult(
        global_transport=global_transport,
        primary_replication_go=passed,
        receipt=receipt,
    )


__all__ = [
    "ActionConditionGradient",
    "IdentityConditionProbe",
    "MulticonditionTransportResult",
    "PairV7MulticonditionError",
    "PRIMARY_PAIR_IDS",
    "PRIMARY_SCHEDULE_INDICES",
    "RECEIPT_SCHEMA",
    "WORLD_INPUT_SCHEMA",
    "solve_multicondition_common_direction",
    "ROUTING_DIAGNOSTIC_SCHEMA",
    "SUBSPACE_DIAGNOSTIC_SCHEMA",
]
