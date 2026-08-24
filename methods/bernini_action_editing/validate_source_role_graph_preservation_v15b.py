"""Fail-closed validator for the independent v15b CPU reference seam.

This validator certifies tensor/ABI properties that it can recompute. It
deliberately cannot authorize a renderer/controller route: no production
runner is connected to this reference module, and hashes in a JSON receipt do
not prove that external model, RNG, or output bytes were physically reopened.
The r8 position fixture, role-label transport, and immutable phase-0 raw source
H/K/V bytes are reopenable CPU materials.  Successful replay proves internal
extraction/carrier consistency only; caller-supplied source video/latent hashes
remain externally unauthenticated and never establish position removal or
native flow.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import io
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

try:
    from . import source_role_graph_preservation_v15b as core
except ImportError:  # pragma: no cover
    import source_role_graph_preservation_v15b as core


ARM_RECEIPT_SCHEMA = "bernini-source-role-preservation-arm-receipt-v15b-r8"
SUITE_VALIDATION_SCHEMA = "bernini-source-role-preservation-suite-validation-v15b-r8"
CELL_RECOMPUTE_SCHEMA = "bernini-source-role-cell-tensor-recompute-v15b-r8"
CELL_MATERIAL_FIXTURE_SCHEMA = "bernini-source-role-cell-material-fixture-v15b-r8"
CELL_MATERIAL_FIXTURE_MAGIC = b"BERNINI_V15B_R8_CELL_MATERIAL\x00"

ARM_RECEIPT_FIELDS = frozenset(
    {
        "schema_version", "complete", "arm_id", "contract_digest",
        "source_video_sha256", "instruction_sha256", "binding_digest",
        "mask_digest", "track_authority_digest", "source_graph_digest",
        "canonical_trace_digest", "trace_extractor_code_sha256",
        "trace_extractor_config_sha256", "self_reported_model_checkpoint_sha256",
        "self_reported_model_code_sha256", "core_code_sha256", "validator_code_sha256",
        "self_reported_noise_sha256", "self_reported_candidate_schedule_sha256",
        "initial_noise_mode",
        "graph_anchor_slot", "signed_edit_graph_digest", "anchor_cached_fields",
        "self_reported_anchor_forbidden_access_count", "temporal_phases", "height", "width",
        "batch_size", "heads", "head_dim", "hidden_width", "tensor_dtype",
        "tensor_device", "denoise_steps", "transformer_blocks", "cfg_branches",
        "route_strength", "memory_strength", "self_reported_model_block_call_count",
        "self_reported_pre_block_call_count", "self_reported_post_block_call_count",
        "self_reported_route_call_count", "self_reported_source_content_memory_call_count",
        "self_reported_background_carrier_call_count",
        "self_reported_phase0_full_source_restore_call_count",
        "self_reported_optimizer_update_count", "self_reported_parameter_before_sha256",
        "self_reported_parameter_after_sha256", "self_reported_buffer_before_sha256",
        "self_reported_buffer_after_sha256", "self_reported_model_state_before_sha256",
        "self_reported_model_state_after_sha256", "self_reported_rng_state_before_sha256",
        "self_reported_rng_state_after_sha256", "self_reported_rng_replay_before_sha256",
        "self_reported_rng_replay_after_sha256",
        "self_reported_frame0_source_tensor_sha256",
        "self_reported_frame0_output_tensor_sha256",
        "self_reported_output_tensor_sha256", "self_reported_output_video_sha256",
        "runner_integration_present", "route_authorized", "self_reported_block_audits",
        "receipt_digest",
    }
)


class V15BReceiptValidationError(RuntimeError):
    pass


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V15BReceiptValidationError(f"{label} must be a mapping")
    return value


def _sha(value: Any, *, label: str) -> str:
    try:
        return core._sha(value, label=label)
    except core.V15BContractError as error:
        raise V15BReceiptValidationError(str(error)) from error


def _count(value: Any, *, label: str) -> int:
    try:
        return core._exact_int(value, label=label)
    except core.V15BContractError as error:
        raise V15BReceiptValidationError(str(error)) from error


def _finite(value: Any, *, label: str) -> float:
    try:
        return core._finite(value, label=label)
    except core.V15BContractError as error:
        raise V15BReceiptValidationError(str(error)) from error


def _zero(value: Any, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != 0.0:
        raise V15BReceiptValidationError(f"{label} must be exactly zero")


def _scan_forbidden(value: Any, *, path: str = "receipt") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise V15BReceiptValidationError(f"{path} has a non-string key")
            if key.lower() in core.FORBIDDEN_ANCHOR_FIELDS:
                raise V15BReceiptValidationError(f"forbidden anchor field at {path}.{key}")
            _scan_forbidden(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _scan_forbidden(item, path=f"{path}[{index}]")


def _validate_core_material(value: Any, *, label: str) -> None:
    try:
        core._revalidate_material(value, label=label)
    except core.V15BContractError as error:
        raise V15BReceiptValidationError(str(error)) from error


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def current_code_pins_v15b() -> tuple[str, str]:
    return _file_sha256(Path(core.__file__).resolve()), _file_sha256(Path(__file__).resolve())


def receipt_digest_v15b(receipt: Mapping[str, Any]) -> str:
    row = dict(receipt)
    row.pop("receipt_digest", None)
    return core.object_sha256(row)


def _audit(value: Any) -> core.BlockAuditV15B:
    row = _mapping(value, label="block audit")
    if set(row) != {f.name for f in fields(core.BlockAuditV15B)}:
        raise V15BReceiptValidationError("block audit fields differ")
    normalized = dict(row)
    if not isinstance(normalized["routed_roles"], (list, tuple)):
        raise V15BReceiptValidationError("routed roles must be a list")
    normalized["routed_roles"] = tuple(normalized["routed_roles"])
    for key in (
        "target_role_assigned_token_count_by_phase",
        "target_role_unassigned_corridor_count_by_phase",
    ):
        if not isinstance(normalized[key], (list, tuple)):
            raise V15BReceiptValidationError(f"{key} must be a list")
        normalized[key] = tuple(normalized[key])
    try:
        return core.BlockAuditV15B(**normalized)
    except core.V15BContractError as error:
        raise V15BReceiptValidationError(str(error)) from error


def _expected_audit_keys() -> set[tuple[int, int, str, str]]:
    return {
        (step, block, branch, stage)
        for step in range(core.DENOISE_STEPS)
        for block in range(core.TRANSFORMER_BLOCKS)
        for branch in core.CFG_BRANCHES
        for stage in ("pre", "post")
    }


def _validate_static_receipt_authority(
    contract: core.FourArmContractV15B, row: Mapping[str, Any]
) -> None:
    for key, expected in (
        ("contract_digest", contract.digest),
        ("source_video_sha256", contract.source_video_sha256),
        ("instruction_sha256", contract.instruction_sha256),
        ("binding_digest", contract.binding_digest),
        ("mask_digest", contract.mask_digest),
        ("track_authority_digest", contract.track_authority_digest),
        ("source_graph_digest", contract.source_graph_digest),
        ("canonical_trace_digest", contract.canonical_trace_digest),
        ("trace_extractor_code_sha256", contract.trace_extractor_code_sha256),
        ("trace_extractor_config_sha256", contract.trace_extractor_config_sha256),
        ("self_reported_model_checkpoint_sha256",
         contract.self_reported_model_checkpoint_sha256),
        ("self_reported_model_code_sha256", contract.self_reported_model_code_sha256),
        ("core_code_sha256", contract.core_code_sha256),
        ("validator_code_sha256", contract.validator_code_sha256),
    ):
        if _sha(row[key], label=key) != expected:
            raise V15BReceiptValidationError(f"{key} differs from contract")
    geometry = (
        row["temporal_phases"], row["height"], row["width"], row["batch_size"],
        row["heads"], row["head_dim"], row["hidden_width"], row["tensor_dtype"],
        row["tensor_device"], row["denoise_steps"], row["transformer_blocks"],
        tuple(row["cfg_branches"]) if isinstance(row["cfg_branches"], (list, tuple)) else None,
    )
    expected_geometry = (
        contract.temporal_phases, contract.height, contract.width,
        contract.batch_size, contract.heads, contract.head_dim,
        contract.hidden_width, contract.tensor_dtype, contract.tensor_device,
        contract.denoise_steps, contract.transformer_blocks, contract.cfg_branches,
    )
    if geometry != expected_geometry:
        raise V15BReceiptValidationError("receipt tensor/execution geometry differs")


def validate_arm_receipt_v15b(
    contract: core.FourArmContractV15B,
    arm: core.ArmContractV15B,
    masks: core.SourceRoleMaskSetV15B,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_core_material(contract, label="arm receipt contract")
    _validate_core_material(arm, label="arm receipt arm")
    _validate_core_material(masks, label="arm receipt masks")
    row = _mapping(receipt, label="arm receipt")
    _scan_forbidden(row)
    if set(row) != ARM_RECEIPT_FIELDS:
        raise V15BReceiptValidationError("arm receipt fields differ")
    if row["schema_version"] != ARM_RECEIPT_SCHEMA or row["complete"] is not True:
        raise V15BReceiptValidationError("arm receipt is incomplete/non-v15b-r8")
    if row["arm_id"] != arm.arm_id:
        raise V15BReceiptValidationError("arm identity differs")
    if receipt_digest_v15b(row) != _sha(row["receipt_digest"], label="arm receipt digest"):
        raise V15BReceiptValidationError("arm receipt digest differs")
    _validate_static_receipt_authority(contract, row)
    _sha(row["self_reported_noise_sha256"], label="self-reported noise")
    _sha(
        row["self_reported_candidate_schedule_sha256"],
        label="self-reported schedule",
    )
    if row["initial_noise_mode"] != "keyed_only":
        raise V15BReceiptValidationError("receipt admits anchor-seeded target noise")
    if row["runner_integration_present"] is not False or row["route_authorized"] is not False:
        raise V15BReceiptValidationError("CPU reference receipt cannot claim renderer route authority")

    expected_slot = None if arm.graph_slot is None else (
        contract.graph_a_slot if arm.graph_slot == "A" else contract.graph_b_slot
    )
    expected_graph = None if arm.graph_slot is None else (
        contract.signed_graph_a_digest if arm.graph_slot == "A" else contract.signed_graph_b_digest
    )
    if row["graph_anchor_slot"] != expected_slot or row["signed_edit_graph_digest"] != expected_graph:
        raise V15BReceiptValidationError("signed graph/anchor-slot authority differs")
    expected_cached = ["canonical_role_relation_graph"] if arm.route_enabled else []
    if row["anchor_cached_fields"] != expected_cached:
        raise V15BReceiptValidationError("anchor cache is not graph-only")
    if row["self_reported_anchor_forbidden_access_count"] != 0:
        raise V15BReceiptValidationError("receipt accessed forbidden anchor content")
    if (_finite(row["route_strength"], label="route strength") != arm.route_strength or
            _finite(row["memory_strength"], label="memory strength") != arm.memory_strength):
        raise V15BReceiptValidationError("receipt route/memory strength differs")

    cells = core.EXPECTED_EXECUTION_CELLS
    exact_counts = {
        "self_reported_model_block_call_count": cells,
        "self_reported_pre_block_call_count": cells,
        "self_reported_post_block_call_count": cells,
        "self_reported_route_call_count": cells if arm.route_enabled else 0,
        "self_reported_source_content_memory_call_count": (
            cells if arm.source_content_memory else 0
        ),
        "self_reported_background_carrier_call_count": cells * (
            int(arm.restore_background_pre) + int(arm.restore_background_post)
        ),
        "self_reported_phase0_full_source_restore_call_count": 2 * cells,
    }
    for key, expected in exact_counts.items():
        if _count(row[key], label=key) != expected:
            raise V15BReceiptValidationError(f"{key} differs from exact 40x22x2 closure")

    if _count(
        row["self_reported_optimizer_update_count"], label="self-reported optimizer updates"
    ) != 0:
        raise V15BReceiptValidationError("reference route must be zero-update")
    for key in (
        "self_reported_parameter_before_sha256", "self_reported_parameter_after_sha256",
        "self_reported_buffer_before_sha256", "self_reported_buffer_after_sha256",
        "self_reported_model_state_before_sha256", "self_reported_model_state_after_sha256",
        "self_reported_rng_state_before_sha256", "self_reported_rng_state_after_sha256",
        "self_reported_rng_replay_before_sha256", "self_reported_rng_replay_after_sha256",
        "self_reported_frame0_source_tensor_sha256",
        "self_reported_frame0_output_tensor_sha256",
        "self_reported_output_tensor_sha256", "self_reported_output_video_sha256",
    ):
        _sha(row[key], label=key)
    for before, after, label in (
        ("self_reported_parameter_before_sha256",
         "self_reported_parameter_after_sha256", "parameters"),
        ("self_reported_buffer_before_sha256",
         "self_reported_buffer_after_sha256", "buffers"),
        ("self_reported_model_state_before_sha256",
         "self_reported_model_state_after_sha256", "model state"),
        ("self_reported_rng_state_before_sha256",
         "self_reported_rng_replay_before_sha256", "initial RNG replay"),
        ("self_reported_rng_state_after_sha256",
         "self_reported_rng_replay_after_sha256", "terminal RNG replay"),
        ("self_reported_frame0_source_tensor_sha256",
         "self_reported_frame0_output_tensor_sha256", "frame0 source/output"),
    ):
        if row[before] != row[after]:
            raise V15BReceiptValidationError(f"{label} byte authority differs")

    raw_audits = row["self_reported_block_audits"]
    if not isinstance(raw_audits, list):
        raise V15BReceiptValidationError("block_audits must be a list")
    if len(raw_audits) != 2 * cells:
        raise V15BReceiptValidationError("block audit count is not exact pre/post 40x22x2")
    audits = tuple(_audit(item) for item in raw_audits)
    spatial = contract.height * contract.width
    phase0_role_union = None
    for role_mask in masks.role_masks.values():
        current = role_mask[:, :spatial]
        phase0_role_union = current.clone() if phase0_role_union is None else (
            phase0_role_union | current
        )
    if phase0_role_union is None:
        raise V15BReceiptValidationError("source role masks are empty")
    phase0_role_count = int(phase0_role_union.sum())
    corridor_by_phase = tuple(
        int(masks.editable_corridor_mask[
            :, phase * spatial:(phase + 1) * spatial
        ].sum())
        for phase in range(core.LATENT_PHASES)
    )
    seen: set[tuple[int, int, str, str]] = set()
    for audit in audits:
        key = (audit.step_index, audit.block_index, audit.branch, audit.stage)
        if key in seen:
            raise V15BReceiptValidationError("duplicate execution-cell/stage audit")
        seen.add(key)
        if audit.signed_graph_digest != expected_graph or audit.mask_digest != contract.mask_digest:
            raise V15BReceiptValidationError("audit graph/mask authority differs")
        for label, value in (
            ("raw source material", audit.raw_source_material_digest),
            ("source latent", audit.source_latent_sha256),
            ("canonical extraction config", audit.canonical_extraction_config_sha256),
        ):
            _sha(value, label=f"self-reported audit {label}")
        if audit.raw_source_material_reopened is not True:
            raise V15BReceiptValidationError(
                "self-reported audit omitted immutable raw source reopen"
            )
        audit_geometry = (
            audit.tensor_batch_size, audit.tensor_temporal_phases,
            audit.tensor_height, audit.tensor_width, audit.tensor_heads,
            audit.tensor_head_dim, audit.tensor_hidden_width,
            audit.tensor_dtype, audit.tensor_device,
        )
        contract_geometry = (
            contract.batch_size, contract.temporal_phases,
            contract.height, contract.width, contract.heads,
            contract.head_dim, contract.hidden_width,
            contract.tensor_dtype, contract.tensor_device,
        )
        if audit_geometry != contract_geometry:
            raise V15BReceiptValidationError("audit tensor geometry differs from contract")
        for name in (
            "route_delta_outside_corridor_max_abs", "memory_residual_outside_corridor_max_abs",
            "phase0_route_max_abs", "phase0_memory_max_abs", "disallowed_add_edge_max_abs",
            "disallowed_remove_edge_max_abs", "memory_hidden_mutation_max_abs",
            "memory_key_mutation_max_abs", "memory_convex_violation_max_abs",
            "cross_role_memory_write_max_abs", "target_cross_role_rename_count",
            "target_corridor_escape_count", "target_dual_position_component_count",
            "transition_background_overlap_count", "source_coordinate_target_write_count",
            "phase0_hidden_source_max_abs", "phase0_key_source_max_abs",
            "phase0_value_source_max_abs",
            "same_coordinate_object_kv_copy_count", "object_hidden_hard_restore_count",
            "phase_indexed_source_kv_access_count", "post_rope_source_kv_access_count",
            "anchor_forbidden_access_count",
        ):
            _zero(getattr(audit, name), label=name)
        if (audit.phase0_full_source_restore_call_count != 1 or
                audit.phase0_full_source_restore_token_count !=
                contract.batch_size * contract.height * contract.width):
            raise V15BReceiptValidationError("audit phase0 identity exception differs")
        if audit.stage == "pre":
            if audit.route_strength != arm.route_strength or audit.memory_strength != arm.memory_strength:
                raise V15BReceiptValidationError("pre audit route/memory strength differs")
            expected_operator = (
                "position_scrubbed_target_key_persistent_role_pool_query_scatter"
                if arm.route_enabled else "none"
            )
            if audit.relation_operator != expected_operator:
                raise V15BReceiptValidationError("pre audit relation operator differs")
            if arm.route_enabled != (audit.target_key_sha256 is not None):
                raise V15BReceiptValidationError("pre audit target-K read authority differs")
            if arm.route_enabled and audit.routed_roles != tuple(sorted(core.SIGNED_ROLES)):
                raise V15BReceiptValidationError("pre audit routed roles differ")
            if not arm.route_enabled and audit.routed_roles:
                raise V15BReceiptValidationError("K0 pre audit claims routed roles")
            if arm.source_content_memory:
                if (audit.role_memory_read_count != 4 or
                        audit.memory_builder_receipt_digest is None or
                        audit.target_role_state_digest is None or
                        audit.target_transport_digest is None or
                        audit.position_projector_sha256 is None or
                        audit.scrubbed_target_key_sha256 is None or
                        audit.persistent_support_sha256 is None or
                        audit.slot_provenance_digest is None or
                        audit.slot_uuid_mask_provenance_verified is not True or
                        audit.target_write_ownership_sha256 is None or
                        audit.target_write_ownership_verified is not True or
                        audit.cross_role_zero_proof_sha256 is None or
                        audit.target_role_assigned_token_count_by_phase[0] <= 0 or
                        audit.target_role_unassigned_corridor_count_by_phase[0] != 0):
                    raise V15BReceiptValidationError(
                        "pre audit lacks target-role/unique-builder memory authority"
                    )
                if (audit.target_role_assigned_token_count_by_phase[0] !=
                        phase0_role_count):
                    raise V15BReceiptValidationError(
                        "pre audit phase0 target roles differ from source masks"
                    )
                for phase in range(1, core.LATENT_PHASES):
                    observed = (
                        audit.target_role_assigned_token_count_by_phase[phase]
                        + audit.target_role_unassigned_corridor_count_by_phase[phase]
                    )
                    if observed != corridor_by_phase[phase]:
                        raise V15BReceiptValidationError(
                            "pre audit target-role assignment does not partition corridor"
                        )
            elif (audit.role_memory_read_count != 0 or
                  audit.memory_builder_receipt_digest is not None or
                  audit.target_role_state_digest is not None or
                  audit.target_transport_digest is not None or
                  audit.position_projector_sha256 is not None or
                  audit.scrubbed_target_key_sha256 is not None or
                  audit.persistent_support_sha256 is not None or
                  audit.slot_provenance_digest is not None or
                  audit.slot_uuid_mask_provenance_verified is not False or
                  audit.target_write_ownership_sha256 is not None or
                  audit.target_write_ownership_verified is not False or
                  audit.cross_role_zero_proof_sha256 is not None or
                  any(audit.target_role_assigned_token_count_by_phase) or
                  any(audit.target_role_unassigned_corridor_count_by_phase)):
                raise V15BReceiptValidationError("inactive memory emitted builder/read authority")
            restore = arm.restore_background_pre
        else:
            if (audit.route_strength != 0.0 or audit.memory_strength != 0.0 or
                    audit.relation_operator != "none" or audit.target_key_sha256 is not None or
                    audit.role_memory_read_count != 0 or
                    audit.memory_builder_receipt_digest is not None or
                    audit.target_role_state_digest is not None or
                    audit.target_transport_digest is not None or
                    audit.position_projector_sha256 is not None or
                    audit.scrubbed_target_key_sha256 is not None or
                    audit.persistent_support_sha256 is not None or
                    audit.slot_provenance_digest is not None or
                    audit.slot_uuid_mask_provenance_verified is not False or
                    audit.target_write_ownership_sha256 is not None or
                    audit.target_write_ownership_verified is not False or
                    audit.cross_role_zero_proof_sha256 is not None or
                    any(audit.target_role_assigned_token_count_by_phase) or
                    any(audit.target_role_unassigned_corridor_count_by_phase) or
                    audit.routed_roles):
                raise V15BReceiptValidationError("post audit may not route/read source object memory")
            restore = arm.restore_background_post
        metrics = (
            audit.background_hidden_max_abs, audit.background_key_max_abs,
            audit.background_value_max_abs,
        )
        if restore:
            for metric in metrics:
                _zero(metric, label="background hard restore")
        elif any(metric is not None for metric in metrics):
            raise V15BReceiptValidationError("inactive background restore emitted equality metrics")
    if seen != _expected_audit_keys():
        raise V15BReceiptValidationError("execution audits do not cover exact step/block/CFG/stage product")
    return {
        "arm_id": arm.arm_id, **exact_counts,
        "block_audit_count": len(audits),
        "externally_authenticated": False,
        "position_removed_claimed": False,
        "native_flow_claimed": False,
        "route_authorized": False,
        "scientific_claim_authorized": False,
    }


def _recompute_graph_authority(
    *, contract: core.FourArmContractV15B,
    binding: core.SourceActionRoleBindingV15B,
    masks: core.SourceRoleMaskSetV15B,
    source_graph: core.SourceRelationGraphV15B,
    source_warp: core.MonotonicEventWarpV15B,
    banks: Sequence[core.AnchorGraphBankV15B],
    warps: Sequence[core.MonotonicEventWarpV15B],
    traces: Sequence[core.RoleContactTraceV15B],
    canonical_trace: core.RoleContactTraceV15B,
    signed_graph_a: core.SignedEditGraphV15B,
    signed_graph_b: core.SignedEditGraphV15B,
    aligned_swap_report: core.GraphPairDiagnosticV15B,
    four_anchor_consensus: core.FourAnchorConsensusReportV15B,
) -> tuple[core.GraphPairDiagnosticV15B, core.FourAnchorConsensusReportV15B]:
    for label, value in (
        ("contract", contract), ("binding", binding), ("masks", masks),
        ("source graph", source_graph), ("source warp", source_warp),
        ("canonical trace", canonical_trace), ("signed graph A", signed_graph_a),
        ("signed graph B", signed_graph_b), ("swap report", aligned_swap_report),
        ("four-anchor consensus", four_anchor_consensus),
    ):
        _validate_core_material(value, label=f"graph authority {label}")
    for index, value in enumerate(banks):
        _validate_core_material(value, label=f"graph authority bank {index}")
    for index, value in enumerate(warps):
        _validate_core_material(value, label=f"graph authority warp {index}")
    for index, value in enumerate(traces):
        _validate_core_material(value, label=f"graph authority trace {index}")
    if binding.digest != contract.binding_digest:
        raise V15BReceiptValidationError("binding differs from contract")
    if (masks.digest != contract.mask_digest or
            masks.track_authority.digest != contract.track_authority_digest or
            masks.source_video_sha256 != contract.source_video_sha256 or
            (masks.temporal_phases, masks.height, masks.width) !=
            (contract.temporal_phases, contract.height, contract.width)):
        raise V15BReceiptValidationError("source mask/track/T-H-W authority differs")
    if source_graph.digest != contract.source_graph_digest:
        raise V15BReceiptValidationError("source graph differs from contract")
    graph_heads = [int(source_graph.graph.shape[0])]
    graph_heads.extend(int(bank.relation_graph.graph.shape[0]) for bank in banks)
    graph_heads.extend(
        (int(signed_graph_a.graph.shape[0]), int(signed_graph_b.graph.shape[0]))
    )
    if any(heads != contract.heads for heads in graph_heads):
        raise V15BReceiptValidationError("mask/graph/audit/contract head geometry differs")
    try:
        recomputed_consensus = core.diagnose_four_anchor_consensus_v15b(
            banks, warps, traces, canonical_trace
        )
    except core.V15BContractError as error:
        raise V15BReceiptValidationError(str(error)) from error
    by_slot = {bank.anchor_slot: bank for bank in banks}
    warp_by_slot = {warp.source_slot: warp for warp in warps}
    if len(by_slot) != 4 or len(warp_by_slot) != 4:
        raise V15BReceiptValidationError("graph authority has duplicate v0-v3 inputs")
    recomputed_pair = core.compare_anchor_graphs_v15b(
        by_slot[core.DEFAULT_GRAPH_A_SLOT], by_slot[core.DEFAULT_GRAPH_B_SLOT],
        warp_by_slot[core.DEFAULT_GRAPH_A_SLOT], warp_by_slot[core.DEFAULT_GRAPH_B_SLOT],
    )
    if recomputed_pair.digest != aligned_swap_report.digest:
        raise V15BReceiptValidationError("supplied v0/v1 report is not a recomputation")
    if recomputed_consensus.digest != four_anchor_consensus.digest:
        raise V15BReceiptValidationError("supplied v0-v3 consensus is not a recomputation")
    if (recomputed_pair.digest != contract.aligned_swap_report_digest or
            recomputed_consensus.digest != contract.four_anchor_consensus_digest or
            recomputed_consensus.canonical_trace_digest != contract.canonical_trace_digest or
            recomputed_consensus.extractor_code_sha256 != contract.trace_extractor_code_sha256 or
            recomputed_consensus.extractor_config_sha256 != contract.trace_extractor_config_sha256 or
            recomputed_consensus.asset_sha256_by_slot != contract.anchor_asset_sha256_by_slot):
        raise V15BReceiptValidationError("recomputed trace/asset/extractor/graph authority differs")
    if not recomputed_pair.passed or not recomputed_consensus.robust_passed:
        raise V15BReceiptValidationError("appearance-invariance graph preflight failed")
    if (source_warp.source_trace_digest != source_graph.timing_trace_digest or
            source_warp.canonical_trace_digest != canonical_trace.digest):
        raise V15BReceiptValidationError("source warp trace authority differs")
    try:
        rebuilt_a = core.build_signed_edit_graph_v15b(
            source_graph=source_graph, anchor_bank=by_slot[core.DEFAULT_GRAPH_A_SLOT],
            source_warp=source_warp, anchor_warp=warp_by_slot[core.DEFAULT_GRAPH_A_SLOT],
        )
        rebuilt_b = core.build_signed_edit_graph_v15b(
            source_graph=source_graph, anchor_bank=by_slot[core.DEFAULT_GRAPH_B_SLOT],
            source_warp=source_warp, anchor_warp=warp_by_slot[core.DEFAULT_GRAPH_B_SLOT],
        )
    except core.V15BContractError as error:
        raise V15BReceiptValidationError(str(error)) from error
    if (rebuilt_a.digest != signed_graph_a.digest or rebuilt_b.digest != signed_graph_b.digest or
            rebuilt_a.digest != contract.signed_graph_a_digest or
            rebuilt_b.digest != contract.signed_graph_b_digest):
        raise V15BReceiptValidationError("signed required/allowed edge graph was not recomputed")
    return recomputed_pair, recomputed_consensus


def validate_four_arm_receipts_v15b(
    *, contract: core.FourArmContractV15B,
    receipts: Sequence[Mapping[str, Any]],
    binding: core.SourceActionRoleBindingV15B,
    masks: core.SourceRoleMaskSetV15B,
    source_graph: core.SourceRelationGraphV15B,
    source_warp: core.MonotonicEventWarpV15B,
    banks: Sequence[core.AnchorGraphBankV15B],
    warps: Sequence[core.MonotonicEventWarpV15B],
    traces: Sequence[core.RoleContactTraceV15B],
    canonical_trace: core.RoleContactTraceV15B,
    signed_graph_a: core.SignedEditGraphV15B,
    signed_graph_b: core.SignedEditGraphV15B,
    aligned_swap_report: core.GraphPairDiagnosticV15B,
    four_anchor_consensus: core.FourAnchorConsensusReportV15B,
    require_route_authorization: bool = False,
) -> dict[str, Any]:
    if not isinstance(contract, core.FourArmContractV15B):
        raise V15BReceiptValidationError("validator requires a v15b contract")
    current_core, current_validator = current_code_pins_v15b()
    if (contract.core_code_sha256 != current_core or
            contract.validator_code_sha256 != current_validator):
        raise V15BReceiptValidationError("current core/validator bytes differ from contract pins")
    recomputed_pair, recomputed_consensus = _recompute_graph_authority(
        contract=contract, binding=binding, masks=masks, source_graph=source_graph,
        source_warp=source_warp, banks=banks, warps=warps, traces=traces,
        canonical_trace=canonical_trace, signed_graph_a=signed_graph_a,
        signed_graph_b=signed_graph_b, aligned_swap_report=aligned_swap_report,
        four_anchor_consensus=four_anchor_consensus,
    )
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)):
        raise V15BReceiptValidationError("receipts must be a sequence")
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in receipts:
        row = _mapping(item, label="arm receipt")
        arm_id = row.get("arm_id")
        if arm_id in by_id:
            raise V15BReceiptValidationError("duplicate arm receipt")
        by_id[arm_id] = row
    if set(by_id) != set(core.ARM_IDS):
        raise V15BReceiptValidationError("four-arm receipt set differs")
    summaries = []; noise = set(); schedule = set()
    for arm in contract.arms:
        row = by_id[arm.arm_id]
        summaries.append(validate_arm_receipt_v15b(contract, arm, masks, row))
        noise.add(row["self_reported_noise_sha256"])
        schedule.add(row["self_reported_candidate_schedule_sha256"])
    if len(noise) != 1 or len(schedule) != 1:
        raise V15BReceiptValidationError("four arms are not noise/schedule matched")
    if require_route_authorization:
        raise V15BReceiptValidationError(
            "route authorization refused: production runner/model/RNG/output bytes are not reopenable here"
        )
    result = {
        "schema_version": SUITE_VALIDATION_SCHEMA,
        "reference_validation_complete": True,
        "externally_authenticated": False,
        "source_video_latent_externally_authenticated": False,
        "cpu_mechanical_scope": "internal_phase0_raw_extraction_consistency_only",
        "position_removed_claimed": False,
        "native_flow_claimed": False,
        "route_authorized": False,
        "scientific_claim_authorized": False,
        "decision": "NO_GO_RUNNER_INTEGRATION_UNPROVEN",
        "contract_digest": contract.digest,
        "self_reported_noise_sha256": next(iter(noise)),
        "self_reported_candidate_schedule_sha256": next(iter(schedule)),
        "aligned_swap_report_digest": recomputed_pair.digest,
        "four_anchor_consensus_digest": recomputed_consensus.digest,
        "canonical_trace_digest": canonical_trace.digest,
        "raw_swap_cosine": recomputed_pair.raw_cosine,
        "raw_swap_distance": recomputed_pair.raw_normalized_frobenius_distance,
        "aligned_swap_cosine": recomputed_pair.aligned_cosine,
        "aligned_swap_distance": recomputed_pair.aligned_normalized_frobenius_distance,
        "warp_a_gate": recomputed_pair.warp_a_gate,
        "warp_b_gate": recomputed_pair.warp_b_gate,
        "arms": summaries,
    }
    result["validation_digest"] = core.object_sha256(result)
    return result


def _encode_material(value: Any) -> Any:
    """Encode only tensors/primitives plus explicit v15b dataclass tags."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__v15b_dataclass__": type(value).__name__,
            "fields": {
                field.name: _encode_material(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, Mapping):
        return {key: _encode_material(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_encode_material(item) for item in value)
    if isinstance(value, list):
        return [_encode_material(item) for item in value]
    return value


def _decode_material(value: Any) -> Any:
    """Allowlisted constructor replay; no custom class is unpickled."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        if set(value) == {"__v15b_dataclass__", "fields"}:
            name = value["__v15b_dataclass__"]
            constructor_fields = value["fields"]
            cls = getattr(core, name, None) if isinstance(name, str) else None
            if (cls is None or not isinstance(constructor_fields, Mapping) or
                    not is_dataclass(cls) or not name.endswith("V15B")):
                raise V15BReceiptValidationError(
                    "cell material dataclass tag is not allowlisted"
                )
            expected = {field.name for field in fields(cls)}
            if set(constructor_fields) != expected:
                raise V15BReceiptValidationError(
                    "cell material dataclass fields differ"
                )
            return cls(**{
                key: _decode_material(item)
                for key, item in constructor_fields.items()
            })
        return {key: _decode_material(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_decode_material(item) for item in value)
    if isinstance(value, list):
        return [_decode_material(item) for item in value]
    return value


def _serialize_cell_material(row: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(_encode_material(dict(row)), buffer)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).digest()
    return CELL_MATERIAL_FIXTURE_MAGIC + digest + payload


def _read_cell_material(source: Any, *, stage: str) -> tuple[dict[str, Any], str]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        raw = bytes(source)
    elif isinstance(source, os.PathLike):
        raw = Path(source).read_bytes()
    else:
        raise V15BReceiptValidationError(
            "cell validation requires serialized bytes or a path-like fixture"
        )
    prefix = len(CELL_MATERIAL_FIXTURE_MAGIC)
    if (len(raw) <= prefix + 32 or
            raw[:prefix] != CELL_MATERIAL_FIXTURE_MAGIC):
        raise V15BReceiptValidationError("cell material fixture magic differs")
    claimed = raw[prefix:prefix + 32]
    payload = raw[prefix + 32:]
    observed = hashlib.sha256(payload).digest()
    if claimed != observed:
        raise V15BReceiptValidationError("cell material fixture byte digest differs")
    try:
        loaded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as error:
        raise V15BReceiptValidationError("cannot reopen cell material fixture") from error
    row = _mapping(loaded, label="cell material fixture")
    if row.get("schema_version") != CELL_MATERIAL_FIXTURE_SCHEMA:
        raise V15BReceiptValidationError("cell material fixture schema differs")
    if row.get("stage") != stage:
        raise V15BReceiptValidationError("cell material fixture stage differs")
    # Decode each top-level field separately, deliberately breaking every
    # tensor alias before allowlisted dataclass constructors replay invariants.
    try:
        fresh = {key: _decode_material(value) for key, value in row.items()}
    except (core.V15BContractError, TypeError, ValueError,
            V15BReceiptValidationError) as error:
        raise V15BReceiptValidationError(
            "cell material failed fresh constructor validation"
        ) from error
    return fresh, hashlib.sha256(raw).hexdigest()


def serialize_pre_block_material_fixture_v15b(
    *, observed: core.PreBlockStateV15B,
    target_hidden: torch.Tensor, target_query: torch.Tensor,
    target_key: torch.Tensor, target_value: torch.Tensor,
    carrier: core.SourceBackgroundCarrierV15B,
    binding: core.SourceActionRoleBindingV15B,
    signed_graph: core.SignedEditGraphV15B | None,
    content_memory: core.SourceRoleContentMemoryV15B | None,
    target_native_transport: core.TargetNativeTransportV15B | None,
    route_strength: float, memory_strength: float, restore_background: bool,
) -> bytes:
    return _serialize_cell_material({
        "schema_version": CELL_MATERIAL_FIXTURE_SCHEMA, "stage": "pre",
        "observed": observed, "target_hidden": target_hidden,
        "target_query": target_query, "target_key": target_key,
        "target_value": target_value, "carrier": carrier, "binding": binding,
        "signed_graph": signed_graph, "content_memory": content_memory,
        "target_native_transport": target_native_transport,
        "route_strength": route_strength, "memory_strength": memory_strength,
        "restore_background": restore_background,
    })


def serialize_post_block_material_fixture_v15b(
    *, observed: core.PostBlockStateV15B,
    target_hidden: torch.Tensor, target_key: torch.Tensor,
    target_value: torch.Tensor, carrier: core.SourceBackgroundCarrierV15B,
    binding: core.SourceActionRoleBindingV15B,
    signed_graph_digest: str | None, restore_background: bool,
) -> bytes:
    return _serialize_cell_material({
        "schema_version": CELL_MATERIAL_FIXTURE_SCHEMA, "stage": "post",
        "observed": observed, "target_hidden": target_hidden,
        "target_key": target_key, "target_value": target_value,
        "carrier": carrier, "binding": binding,
        "signed_graph_digest": signed_graph_digest,
        "restore_background": restore_background,
    })


def _validate_fresh_raw_source_reopen_v15b(
    *, carrier: core.SourceBackgroundCarrierV15B,
    binding: core.SourceActionRoleBindingV15B,
    memory: core.SourceRoleContentMemoryV15B | None,
) -> dict[str, Any]:
    """Independently reopen raw H/K/V/masks and replay all phase-0 gates."""
    _validate_core_material(carrier, label="cell raw carrier")
    raw = carrier.raw_source_material
    _validate_core_material(raw, label="cell immutable raw source material")
    try:
        core._validate_raw_source_material_against_masks_v15b(
            raw, carrier.masks, binding
        )
        first_hidden, first_key, first_value, first_masks = raw.reopen()
        second_hidden, second_key, second_value, second_masks = raw.reopen()
    except core.V15BContractError as error:
        raise V15BReceiptValidationError(str(error)) from error
    if (not torch.equal(first_hidden, second_hidden) or
            not torch.equal(first_key, second_key) or
            not torch.equal(first_value, second_value) or
            any(not torch.equal(first_masks[role], second_masks[role])
                for role in raw.role_ids)):
        raise V15BReceiptValidationError("fresh raw source reopen is not exact")
    if (first_hidden.data_ptr() == second_hidden.data_ptr() or
            first_key.data_ptr() == second_key.data_ptr() or
            first_value.data_ptr() == second_value.data_ptr() or
            any(first_masks[role].data_ptr() == second_masks[role].data_ptr()
                for role in raw.role_ids)):
        raise V15BReceiptValidationError("fresh raw source reopen shared tensor storage")
    spatial = raw.height * raw.width
    if (
        tuple(carrier.hidden[:, :spatial].shape) != tuple(first_hidden.shape)
        or tuple(carrier.key[:, :spatial].shape) != tuple(first_key.shape)
        or tuple(carrier.value[:, :spatial].shape) != tuple(first_value.shape)
        or not torch.equal(carrier.hidden[:, :spatial], first_hidden)
        or not torch.equal(carrier.key[:, :spatial], first_key)
        or not torch.equal(carrier.value[:, :spatial], first_value)
    ):
        raise V15BReceiptValidationError(
            "carrier phase0 H/K/V differs from independent raw reopen"
        )
    result = {
        "raw_source_material_digest": raw.digest,
        "source_video_sha256": raw.source_video_sha256,
        "source_latent_sha256": raw.source_latent_sha256,
        "canonical_extraction_config_sha256": (
            raw.canonical_extraction_config_sha256
        ),
        "source_video_latent_authority_kind": raw.authority_kind,
        "source_video_latent_externally_authenticated": False,
        "cpu_mechanical_scope": "internal_phase0_raw_extraction_consistency_only",
        "raw_source_material_fresh_reopen": True,
        "carrier_phase0_raw_hkv_exact": True,
        "source_memory_reextracted_from_raw": False,
        "per_role_key_value_exact": False,
        "slot_uuid_mask_provenance_verified": False,
    }
    if memory is not None:
        _validate_core_material(memory, label="cell raw-replay memory")
        if memory.raw_source_material.digest != raw.digest:
            raise V15BReceiptValidationError(
                "cell carrier/memory raw source material differs"
            )
        try:
            replay = core._extract_source_role_content_from_raw_v15b(
                raw_source_material=raw,
                position_reference=memory.position_reference,
            )
        except core.V15BContractError as error:
            raise V15BReceiptValidationError(str(error)) from error
        counts = dict(replay["role_token_counts"])
        for role_index, role in enumerate(memory.role_ids):
            count = counts[role]
            if (not torch.equal(
                    memory.key_content[role_index, :count],
                    replay["key_content"][role_index, :count]
                ) or not torch.equal(
                    memory.value_content[role_index, :count],
                    replay["value_content"][role_index, :count]
                )):
                raise V15BReceiptValidationError(
                    f"cell role {role} K/V differs from fresh raw extraction"
                )
        if (not torch.equal(memory.slot_valid_mask, replay["slot_valid_mask"]) or
                not torch.equal(memory.null_key_content, replay["null_key_content"]) or
                memory.slot_provenance_by_role !=
                replay["slot_provenance_by_role"]):
            raise V15BReceiptValidationError(
                "cell memory slot provenance differs from fresh raw extraction"
            )
        result.update({
            "source_memory_reextracted_from_raw": True,
            "per_role_key_value_exact": True,
            "slot_uuid_mask_provenance_verified": True,
        })
    return result


def validate_pre_block_tensor_recompute_v15b(
    *, material_fixture: bytes | os.PathLike,
) -> dict[str, Any]:
    """Reopen and freshly instantiate one serialized CPU cell before replay."""
    row, fixture_sha256 = _read_cell_material(material_fixture, stage="pre")
    expected_fields = {
        "schema_version", "stage", "observed", "target_hidden", "target_query",
        "target_key", "target_value", "carrier", "binding", "signed_graph",
        "content_memory", "target_native_transport", "route_strength",
        "memory_strength", "restore_background",
    }
    if set(row) != expected_fields:
        raise V15BReceiptValidationError("pre-cell material fields differ")
    observed = row["observed"]
    if not isinstance(observed, core.PreBlockStateV15B):
        raise V15BReceiptValidationError("pre-cell fixture lacks a material pre state")
    raw_validation = _validate_fresh_raw_source_reopen_v15b(
        carrier=row["carrier"], binding=row["binding"],
        memory=row["content_memory"],
    )
    try:
        recomputed = core.apply_pre_block_v15b(
            target_hidden=row["target_hidden"], target_query=row["target_query"],
            target_key=row["target_key"], target_value=row["target_value"],
            carrier=row["carrier"], binding=row["binding"],
            signed_graph=row["signed_graph"], content_memory=row["content_memory"],
            target_native_transport=row["target_native_transport"],
            route_strength=row["route_strength"],
            memory_strength=row["memory_strength"],
            restore_background=row["restore_background"],
        )
    except core.V15BContractError as error:
        raise V15BReceiptValidationError(str(error)) from error
    for name in (
        "hidden", "query", "key", "value", "route_delta", "appearance_residual",
    ):
        if not torch.equal(getattr(observed, name), getattr(recomputed, name)):
            raise V15BReceiptValidationError(f"pre-cell {name} differs from tensor recomputation")
    if observed.audit.as_dict() != recomputed.audit.as_dict():
        raise V15BReceiptValidationError("pre-cell audit differs from tensor recomputation")
    observed_state = observed.target_role_state
    recomputed_state = recomputed.target_role_state
    if (observed_state is None) != (recomputed_state is None) or (
        observed_state is not None and observed_state.digest != recomputed_state.digest
    ):
        raise V15BReceiptValidationError("pre-cell persistent role-state differs")
    result = {
        "schema_version": CELL_RECOMPUTE_SCHEMA, "stage": "pre",
        "cell_material_fixture_sha256": fixture_sha256,
        "cell_tensor_abi_digest": recomputed.audit.cell_tensor_abi_digest,
        "tensor_recompute_exact": True, "fresh_material_reinstantiated": True,
        "persistent_transport_digest": recomputed.audit.target_transport_digest,
        "cross_role_memory_write_max_abs": (
            recomputed.audit.cross_role_memory_write_max_abs
        ),
        **raw_validation,
        "target_write_ownership_verified": (
            recomputed.audit.target_write_ownership_verified
        ),
        "externally_authenticated": False,
        "position_removed_claimed": False,
        "native_flow_claimed": False,
        "route_authorized": False, "scientific_claim_authorized": False,
    }
    result["validation_digest"] = core.object_sha256(result)
    return result


def validate_post_block_tensor_recompute_v15b(
    *, material_fixture: bytes | os.PathLike,
) -> dict[str, Any]:
    row, fixture_sha256 = _read_cell_material(material_fixture, stage="post")
    expected_fields = {
        "schema_version", "stage", "observed", "target_hidden", "target_key",
        "target_value", "carrier", "binding", "signed_graph_digest",
        "restore_background",
    }
    if set(row) != expected_fields:
        raise V15BReceiptValidationError("post-cell material fields differ")
    observed = row["observed"]
    if not isinstance(observed, core.PostBlockStateV15B):
        raise V15BReceiptValidationError("post-cell fixture lacks a material post state")
    raw_validation = _validate_fresh_raw_source_reopen_v15b(
        carrier=row["carrier"], binding=row["binding"], memory=None,
    )
    try:
        recomputed = core.apply_post_block_v15b(
            target_hidden=row["target_hidden"], target_key=row["target_key"],
            target_value=row["target_value"], carrier=row["carrier"],
            binding=row["binding"],
            signed_graph_digest=row["signed_graph_digest"],
            restore_background=row["restore_background"],
        )
    except core.V15BContractError as error:
        raise V15BReceiptValidationError(str(error)) from error
    for name in ("hidden", "key", "value"):
        if not torch.equal(getattr(observed, name), getattr(recomputed, name)):
            raise V15BReceiptValidationError(f"post-cell {name} differs from tensor recomputation")
    if observed.audit.as_dict() != recomputed.audit.as_dict():
        raise V15BReceiptValidationError("post-cell audit differs from tensor recomputation")
    result = {
        "schema_version": CELL_RECOMPUTE_SCHEMA, "stage": "post",
        "cell_material_fixture_sha256": fixture_sha256,
        "cell_tensor_abi_digest": recomputed.audit.cell_tensor_abi_digest,
        "tensor_recompute_exact": True, "fresh_material_reinstantiated": True,
        **raw_validation,
        "externally_authenticated": False,
        "position_removed_claimed": False,
        "native_flow_claimed": False,
        "route_authorized": False, "scientific_claim_authorized": False,
    }
    result["validation_digest"] = core.object_sha256(result)
    return result


__all__ = [
    "ARM_RECEIPT_FIELDS", "ARM_RECEIPT_SCHEMA", "SUITE_VALIDATION_SCHEMA",
    "CELL_RECOMPUTE_SCHEMA", "CELL_MATERIAL_FIXTURE_SCHEMA",
    "V15BReceiptValidationError", "current_code_pins_v15b", "receipt_digest_v15b",
    "validate_arm_receipt_v15b", "validate_four_arm_receipts_v15b",
    "validate_pre_block_tensor_recompute_v15b",
    "validate_post_block_tensor_recompute_v15b",
    "serialize_pre_block_material_fixture_v15b",
    "serialize_post_block_material_fixture_v15b",
]
