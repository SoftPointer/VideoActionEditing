#!/usr/bin/env python3
"""Fail-closed hard-gate primitives for an oracle regeneration diagnostic.

This file is deliberately isolated from the learned regeneration/action-state
modules.  It makes no training, representation, or qualification claim.  Its
only admissible gate is an externally reviewed, manually authored, boolean
delete/create intervention in source latent coordinates.

Two execution seams are exposed:

* the private FlowEdit seam keeps the existing FlowEdit state
  constructor and changes only the target noise in ``G`` at solver step zero;
* :class:`LocalOracleNativeBranchRuntimePatchV1` reuses the authenticated
  five-forward native V2V/R2V-4 runtime and sends a scheduled R2V-4 velocity
  only through ``G``.  The exact official V2V tensor is retained outside
  ``G``.  A null gate passes the original official object to the scheduler.

The two seams belong to different outer samplers and must not be described as
one connected GPU runner until a separate integration proves that ABI.  No GPU
launch or optimizer is authorized by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence

import native_branch_homotopy_runtime_v1 as native_runtime
import native_branch_homotopy_v1 as homotopy
import self_guided_action_field_v1 as sgaf


SCHEMA_VERSION = "bernini-oracle-hard-regeneration-canary-v2"
GATE_SCHEMA_VERSION = "bernini-manual-hard-delete-create-gate-v2"
RECEIPT_SCHEMA_VERSION = "bernini-manual-hard-delete-create-review-receipt-v2"
LOCAL_NATIVE_SCHEMA_VERSION = "bernini-local-oracle-native-r2v4-runtime-v2"
FLOWEDIT_RECEIPT_SCHEMA_VERSION = "bernini-oracle-flowedit-execution-receipt-v1"
ANNOTATION_LEAF_SCHEMA_VERSION = "bernini-manual-gate-authority-leaf-v1"
ANNOTATION_TREE_SHAPE = "perfect_binary_power_of_two_v1"
NATIVE_BINDING_RECEIPT_SCHEMA_VERSION = (
    "bernini-oracle-native-execution-binding-receipt-v1"
)
PHASE_COUNT = 21
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_CASES = ("e02", "e03")
ALLOWED_GATE_VARIANTS = (
    "union",
    "time_shift_5",
    "spatial_shift",
    "create_only",
    "delete_only",
    "null",
)

# Intentionally empty in this safe-BLOCKED scaffold.  A future, separately
# reviewed version must hard-pin independently issued per-case roots here.
# Caller arguments, JSON fields, environment variables, and CLI hashes cannot
# populate this trust anchor at runtime.
COMPILED_ANNOTATION_AUTHORITY_ROOTS: Mapping[str, str] = MappingProxyType({})
COMPILED_NATIVE_BINDING_RECEIPT_SHA256: Mapping[str, str] = MappingProxyType({})
COMPILED_FLOWEDIT_BINDING_RECEIPT_SHA256: Mapping[str, str] = MappingProxyType({})

# These are mandatory design gaps for a *future* activation version.  Keeping
# the list beside the empty trust anchors prevents this safe-BLOCKED scaffold
# from being mistaken for a runner that can be enabled by filling JSON fields.
FUTURE_ACTIVATION_BLOCKERS = (
    "native source references require independently VAE-encoded provenance, "
    "encoder/source-frame digests, pairwise storage disjointness, and distinct "
    "content digests",
    "FlowEdit independent noise requires a pinned domain/seed/generator receipt "
    "and content inequality from correlated source noise",
    "preflight must verify the selected case's physical native/Flow binding "
    "receipt file and exact compiled digest",
    "release authorization requires a new compiled-public-key signature scheme "
    "that avoids the authority/spec/component exact-SHA cycle",
    "a future FlowEdit runner ABI must pin callable code and closure identity",
)


class OracleRegenerationCanaryError(RuntimeError):
    """Raised before execution when an oracle-canary invariant differs."""


_VALIDATED_MANIFEST_TOKEN = object()
_VALIDATED_FLOWEDIT_TOKEN = object()
_VALIDATED_NATIVE_BINDING_TOKEN = object()


def _reject_duplicate_json_pairs_v1(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OracleRegenerationCanaryError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_load_path_v1(path: Path, *, label: str) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_pairs_v1,
        )
    except OracleRegenerationCanaryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OracleRegenerationCanaryError(f"{label} JSON is unreadable") from error


def canonical_json_bytes_v1(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def file_sha256_v1(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise OracleRegenerationCanaryError(f"{label} must be lowercase SHA-256")
    return value


def _validate_geometry(value: Any) -> tuple[int, int, int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 5
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise OracleRegenerationCanaryError("oracle latent_geometry differs")
    geometry = tuple(value)
    if geometry[:3] != (1, 1, PHASE_COUNT):
        raise OracleRegenerationCanaryError(
            "oracle gate must be [1,1,21,H,W]"
        )
    return geometry  # type: ignore[return-value]


def _validate_rle(
    value: Any,
    *,
    label: str,
    height: int,
    width: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if not isinstance(value, list) or len(value) != PHASE_COUNT:
        raise OracleRegenerationCanaryError(f"{label} must contain exact21 phases")
    limit = height * width
    phases = []
    for phase_index, spans in enumerate(value):
        if not isinstance(spans, list):
            raise OracleRegenerationCanaryError(
                f"{label}[{phase_index}] must be a span list"
            )
        previous_end = 0
        normalized = []
        for span_index, span in enumerate(spans):
            if (
                not isinstance(span, list)
                or len(span) != 2
                or type(span[0]) is not int
                or type(span[1]) is not int
            ):
                raise OracleRegenerationCanaryError(
                    f"{label}[{phase_index}][{span_index}] differs"
                )
            start, length = span
            end = start + length
            if start < previous_end or length <= 0 or end > limit:
                raise OracleRegenerationCanaryError(
                    f"{label}[{phase_index}] spans overlap, are unsorted, or leave bounds"
                )
            normalized.append((start, length))
            previous_end = end
        phases.append(tuple(normalized))
    if phases[0]:
        raise OracleRegenerationCanaryError(f"{label} phase zero must be empty")
    return tuple(phases)


def _gate_payload_for_digest(
    *,
    geometry: tuple[int, int, int, int, int],
    delete_rle: Sequence[Sequence[Sequence[int]]],
    create_rle: Sequence[Sequence[Sequence[int]]],
) -> Mapping[str, Any]:
    return {
        "latent_geometry": list(geometry),
        "flattening": "per_phase_row_major_yx",
        "delete_rle": delete_rle,
        "create_rle": create_rle,
        "dtype": "bool",
    }


def _annotation_leaf_payload_v1(
    *,
    case_id: str,
    source_sha256: str,
    anchor_sha256: str,
    action_caption_sha256: str,
    structured_action_program_sha256: str,
    mask_sha256: str,
    annotator: str,
    reviewer: str,
) -> Mapping[str, Any]:
    return {
        "schema_version": ANNOTATION_LEAF_SCHEMA_VERSION,
        "case_id": case_id,
        "source_sha256": source_sha256,
        "anchor_sha256": anchor_sha256,
        "action_caption_sha256": action_caption_sha256,
        "structured_action_program_sha256": structured_action_program_sha256,
        "mask_sha256": mask_sha256,
        "annotator": annotator,
        "reviewer": reviewer,
    }


def annotation_authority_leaf_sha256_v1(payload: Mapping[str, Any]) -> str:
    """Hash one externally governed annotation-ledger leaf.

    The leading zero is a domain separator.  Internal nodes use leading one,
    so a leaf payload cannot also be interpreted as a Merkle subtree.
    """

    return hashlib.sha256(b"\x00" + canonical_json_bytes_v1(payload)).hexdigest()


def _verify_annotation_inclusion_v1(
    authority: Any,
    *,
    expected_root_sha256: str,
    expected_leaf_sha256: str,
) -> tuple[int, int]:
    expected_root_sha256 = _require_sha256(
        expected_root_sha256, label="external annotation authority root"
    )
    if not isinstance(authority, Mapping):
        raise OracleRegenerationCanaryError("external annotation authority is absent")
    leaf_index = authority.get("leaf_index")
    tree_size = authority.get("tree_size")
    proof = authority.get("inclusion_proof")
    if (
        authority.get("tree_shape") != ANNOTATION_TREE_SHAPE
        or authority.get("ledger_root_sha256") != expected_root_sha256
        or authority.get("leaf_sha256") != expected_leaf_sha256
        or type(leaf_index) is not int
        or type(tree_size) is not int
        or tree_size <= 0
        or tree_size & (tree_size - 1)
        or leaf_index < 0
        or leaf_index >= tree_size
        or not isinstance(proof, list)
        or len(proof) != int(math.log2(tree_size))
    ):
        raise OracleRegenerationCanaryError(
            "external annotation authority inclusion geometry differs"
        )
    current = bytes.fromhex(expected_leaf_sha256)
    cursor = leaf_index
    for level, entry in enumerate(proof):
        expected_side = "right" if cursor % 2 == 0 else "left"
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"side", "sha256"}
            or entry.get("side") != expected_side
        ):
            raise OracleRegenerationCanaryError(
                f"annotation inclusion proof level {level} differs"
            )
        sibling = bytes.fromhex(
            _require_sha256(entry.get("sha256"), label="annotation proof node")
        )
        left, right = (current, sibling) if expected_side == "right" else (sibling, current)
        current = hashlib.sha256(b"\x01" + left + right).digest()
        cursor //= 2
    if current.hex() != expected_root_sha256:
        raise OracleRegenerationCanaryError(
            "annotation leaf is not included in caller-pinned external root"
        )
    return leaf_index, tree_size


@dataclass(frozen=True)
class ValidatedOracleGateManifestV1:
    path: Path
    file_sha256: str
    case_id: str
    source_sha256: str
    anchor_sha256: str
    action_caption_sha256: str
    structured_action_program_sha256: str
    latent_geometry: tuple[int, int, int, int, int]
    delete_rle: tuple[tuple[tuple[int, int], ...], ...]
    create_rle: tuple[tuple[tuple[int, int], ...], ...]
    mask_sha256: str
    review_receipt_path: Path
    review_receipt_sha256: str
    annotator: str
    reviewer: str
    annotation_authority_root_sha256: str
    annotation_authority_leaf_sha256: str
    annotation_authority_leaf_index: int
    annotation_authority_tree_size: int
    _validation_token: Any = field(repr=False, compare=False, default=None)


def validate_oracle_gate_manifest_v1(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_review_receipt_sha256: str,
    expected_case_id: str,
    expected_source_sha256: str,
    expected_anchor_sha256: str,
    expected_action_caption_sha256: str,
    expected_structured_action_program_sha256: str,
    expected_annotation_authority_root_sha256: str,
    expected_latent_geometry: Optional[tuple[int, int, int, int, int]] = None,
) -> ValidatedOracleGateManifestV1:
    """Authenticate a manual hard D/C artifact and independent review receipt.

    Merely putting ``qualified`` in the gate JSON is insufficient: a distinct
    receipt file must hash-bind the exact manifest bytes and mask payload.
    """

    if expected_case_id not in ALLOWED_CASES:
        raise OracleRegenerationCanaryError("oracle canary is restricted to e02/e03")
    _require_sha256(expected_file_sha256, label="expected gate file")
    _require_sha256(
        expected_review_receipt_sha256, label="expected review receipt"
    )
    _require_sha256(expected_source_sha256, label="expected source")
    _require_sha256(expected_anchor_sha256, label="expected anchor")
    _require_sha256(expected_action_caption_sha256, label="expected action caption")
    _require_sha256(
        expected_structured_action_program_sha256,
        label="expected structured action program",
    )
    _require_sha256(
        expected_annotation_authority_root_sha256,
        label="expected external annotation authority root",
    )
    if not path.is_absolute() or not path.is_file():
        raise OracleRegenerationCanaryError(
            "oracle gate path must be an existing absolute regular file"
        )
    observed_file_sha256 = file_sha256_v1(path)
    if observed_file_sha256 != expected_file_sha256:
        raise OracleRegenerationCanaryError("oracle gate file SHA-256 differs")
    raw = strict_json_load_path_v1(path, label="oracle gate")
    if not isinstance(raw, Mapping) or raw.get("schema_version") != GATE_SCHEMA_VERSION:
        raise OracleRegenerationCanaryError("oracle gate schema differs")
    if raw.get("case_id") != expected_case_id:
        raise OracleRegenerationCanaryError("oracle gate case differs")
    if raw.get("source_sha256") != expected_source_sha256:
        raise OracleRegenerationCanaryError("oracle gate source binding differs")
    if raw.get("anchor_sha256") != expected_anchor_sha256:
        raise OracleRegenerationCanaryError("oracle gate anchor binding differs")
    if raw.get("action_caption_sha256") != expected_action_caption_sha256:
        raise OracleRegenerationCanaryError("oracle gate action-caption binding differs")
    if (
        raw.get("structured_action_program_sha256")
        != expected_structured_action_program_sha256
    ):
        raise OracleRegenerationCanaryError(
            "oracle gate structured-action-program binding differs"
        )
    if raw.get("flattening") != "per_phase_row_major_yx":
        raise OracleRegenerationCanaryError("oracle gate flattening convention differs")
    if raw.get("hard_support") is not True or raw.get("dtype") != "bool":
        raise OracleRegenerationCanaryError("oracle support must be exact boolean")
    if raw.get("phase_zero_empty") is not True:
        raise OracleRegenerationCanaryError("oracle phase-zero authority differs")
    geometry = _validate_geometry(raw.get("latent_geometry"))
    if expected_latent_geometry is not None and geometry != expected_latent_geometry:
        raise OracleRegenerationCanaryError("oracle/runtime latent geometry differs")
    delete_rle = _validate_rle(
        raw.get("delete_rle"), label="delete_rle", height=geometry[-2], width=geometry[-1]
    )
    create_rle = _validate_rle(
        raw.get("create_rle"), label="create_rle", height=geometry[-2], width=geometry[-1]
    )
    if not any(delete_rle[1:]) or not any(create_rle[1:]):
        raise OracleRegenerationCanaryError(
            "diagnostic oracle requires nonempty delete and create support"
        )
    authority = raw.get("authority")
    required_forbidden = {
        "failed_active_video_or_latent": True,
        "raw_anchor_source_pixel_or_latent_difference": True,
        "predicted_soft_gate": True,
    }
    if (
        not isinstance(authority, Mapping)
        or authority.get("role")
        != "manual_source_coordinate_diagnostic_intervention_only"
        or authority.get("training_target_authorized") is not False
        or authority.get("action_representation_claimed") is not False
        or authority.get("forbidden_inputs_absent") != required_forbidden
    ):
        raise OracleRegenerationCanaryError("manual oracle authority differs")
    qualification = raw.get("qualification")
    if not isinstance(qualification, Mapping):
        raise OracleRegenerationCanaryError("oracle review qualification is absent")
    annotator = qualification.get("annotator")
    reviewer = qualification.get("reviewer")
    if (
        qualification.get("status") != "qualified_manual_diagnostic_oracle"
        or not isinstance(annotator, str)
        or not annotator.strip()
        or not isinstance(reviewer, str)
        or not reviewer.strip()
        or reviewer == annotator
    ):
        raise OracleRegenerationCanaryError(
            "oracle requires distinct nonempty annotator/reviewer identities"
        )
    mask_sha256 = _require_sha256(raw.get("mask_sha256"), label="mask payload")
    observed_mask_sha256 = hashlib.sha256(
        canonical_json_bytes_v1(
            _gate_payload_for_digest(
                geometry=geometry,
                delete_rle=raw["delete_rle"],
                create_rle=raw["create_rle"],
            )
        )
    ).hexdigest()
    if mask_sha256 != observed_mask_sha256:
        raise OracleRegenerationCanaryError("oracle mask payload SHA-256 differs")
    leaf_payload = _annotation_leaf_payload_v1(
        case_id=expected_case_id,
        source_sha256=expected_source_sha256,
        anchor_sha256=expected_anchor_sha256,
        action_caption_sha256=expected_action_caption_sha256,
        structured_action_program_sha256=expected_structured_action_program_sha256,
        mask_sha256=mask_sha256,
        annotator=annotator,
        reviewer=reviewer,
    )
    leaf_sha256 = annotation_authority_leaf_sha256_v1(leaf_payload)
    leaf_index, tree_size = _verify_annotation_inclusion_v1(
        raw.get("annotation_authority"),
        expected_root_sha256=expected_annotation_authority_root_sha256,
        expected_leaf_sha256=leaf_sha256,
    )
    receipt_value = qualification.get("review_receipt_path")
    if not isinstance(receipt_value, str):
        raise OracleRegenerationCanaryError("review receipt path is absent")
    receipt_path = Path(receipt_value)
    if not receipt_path.is_absolute() or not receipt_path.is_file():
        raise OracleRegenerationCanaryError(
            "review receipt must be an existing absolute regular file"
        )
    receipt_sha256 = file_sha256_v1(receipt_path)
    if receipt_sha256 != expected_review_receipt_sha256:
        raise OracleRegenerationCanaryError("review receipt SHA-256 differs")
    receipt = strict_json_load_path_v1(receipt_path, label="review receipt")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("case_id") != expected_case_id
        or receipt.get("source_sha256") != expected_source_sha256
        or receipt.get("anchor_sha256") != expected_anchor_sha256
        or receipt.get("action_caption_sha256") != expected_action_caption_sha256
        or receipt.get("structured_action_program_sha256")
        != expected_structured_action_program_sha256
        or receipt.get("gate_manifest_sha256") != observed_file_sha256
        or receipt.get("mask_sha256") != mask_sha256
        or receipt.get("annotation_authority_root_sha256")
        != expected_annotation_authority_root_sha256
        or receipt.get("annotation_authority_leaf_sha256") != leaf_sha256
        or receipt.get("reviewer") != reviewer
        or receipt.get("accepted") is not True
        or receipt.get("phase_zero_source_authority_checked") is not True
        or receipt.get("delete_create_semantics_checked") is not True
        or receipt.get("failed_active_used_to_author_mask") is not False
        or receipt.get("anchor_difference_used_to_author_mask") is not False
        or receipt.get("predicted_soft_gate_used_to_author_mask") is not False
    ):
        raise OracleRegenerationCanaryError("independent oracle review receipt differs")
    return ValidatedOracleGateManifestV1(
        path=path,
        file_sha256=observed_file_sha256,
        case_id=expected_case_id,
        source_sha256=expected_source_sha256,
        anchor_sha256=expected_anchor_sha256,
        action_caption_sha256=expected_action_caption_sha256,
        structured_action_program_sha256=expected_structured_action_program_sha256,
        latent_geometry=geometry,
        delete_rle=delete_rle,
        create_rle=create_rle,
        mask_sha256=mask_sha256,
        review_receipt_path=receipt_path,
        review_receipt_sha256=receipt_sha256,
        annotator=annotator,
        reviewer=reviewer,
        annotation_authority_root_sha256=expected_annotation_authority_root_sha256,
        annotation_authority_leaf_sha256=leaf_sha256,
        annotation_authority_leaf_index=leaf_index,
        annotation_authority_tree_size=tree_size,
        _validation_token=_VALIDATED_MANIFEST_TOKEN,
    )


@dataclass(frozen=True)
class _OwnedHardStateChangeGateV1:
    delete: Any
    create: Any
    support: Any
    preserve: Any
    provenance: str
    source_mask_sha256: str
    realized_gate_sha256: str
    variant: str = "union"
    source_delete_count: int = 0
    source_create_count: int = 0
    realized_delete_count: int = 0
    realized_create_count: int = 0
    permutation_mass_preserved: Optional[bool] = None


def _tensor_raw_bytes_v1(value: Any) -> bytes:
    import torch

    if not isinstance(value, torch.Tensor):
        raise OracleRegenerationCanaryError("tensor byte serialization requires tensor")
    cpu = value.detach().to(device="cpu").contiguous()
    # Do not call ``Tensor.numpy()``: the isolated ``vd`` environment has a
    # valid Torch build but deliberately incompatible NumPy ABI.  Byte-view
    # chunks keep the digest independent of NumPy and work on Torch 1.12.
    flat = cpu.view(torch.uint8).reshape(-1)
    chunk_size = 1024 * 1024
    return b"".join(
        bytes(flat[start : start + chunk_size].tolist())
        for start in range(0, int(flat.numel()), chunk_size)
    )


def _tensor_bytes_equal_v1(left: Any, right: Any) -> bool:
    import torch

    return (
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
        and torch.equal(
            left.contiguous().view(torch.uint8),
            right.contiguous().view(torch.uint8),
        )
    )


def _storage_data_ptr_compat_v1(value: Any) -> int:
    """Return an exact storage pointer on both Torch 1.12 and newer Torch.

    Torch 1.12 has only ``Tensor.storage()``; newer versions prefer
    ``untyped_storage()``.  This helper changes no alias criterion: callers
    still compare tensor data pointers, storage pointers, offsets and strides.
    """

    untyped = getattr(value, "untyped_storage", None)
    storage_getter = untyped if callable(untyped) else getattr(value, "storage", None)
    if not callable(storage_getter):
        raise OracleRegenerationCanaryError("tensor storage pointer is unavailable")
    try:
        storage = storage_getter()
        pointer = storage.data_ptr()
    except Exception as error:
        raise OracleRegenerationCanaryError(
            "tensor storage pointer cannot be resolved"
        ) from error
    if isinstance(pointer, bool) or not isinstance(pointer, int) or pointer < 0:
        raise OracleRegenerationCanaryError("tensor storage pointer differs")
    return int(pointer)


def _certify_expanded_timestep_compat_v1(
    shared_timestep: Any, scheduler_timestep: Any
) -> None:
    """Torch-1.12-compatible copy of the frozen exact alias certificate."""

    import torch

    if (
        not isinstance(shared_timestep, torch.Tensor)
        or not isinstance(scheduler_timestep, torch.Tensor)
        or shared_timestep.shape != (1,)
        or scheduler_timestep.ndim != 0
        or shared_timestep.dtype != scheduler_timestep.dtype
        or shared_timestep.device != scheduler_timestep.device
        or shared_timestep.stride() != (0,)
        or int(shared_timestep.storage_offset())
        != int(scheduler_timestep.storage_offset())
        or int(shared_timestep.data_ptr()) != int(scheduler_timestep.data_ptr())
        or _storage_data_ptr_compat_v1(shared_timestep)
        != _storage_data_ptr_compat_v1(scheduler_timestep)
        or not torch.equal(shared_timestep.reshape(()), scheduler_timestep)
    ):
        raise OracleRegenerationCanaryError(
            "shared timestep is not the authenticated zero-stride expand(1) "
            "view of the scheduler scalar"
        )


def _tensor_storage_interval_v1(value: Any) -> tuple[int, int]:
    import torch

    if not isinstance(value, torch.Tensor) or value.numel() <= 0:
        raise OracleRegenerationCanaryError("storage interval requires nonempty tensor")
    if not value.is_contiguous():
        raise OracleRegenerationCanaryError("storage interval requires contiguous tensor")
    start = int(value.data_ptr())
    end = start + int(value.numel()) * int(value.element_size())
    storage_start = _storage_data_ptr_compat_v1(value)
    expected_start = storage_start + int(value.storage_offset()) * int(value.element_size())
    if start != expected_start or end <= start:
        raise OracleRegenerationCanaryError("tensor storage interval differs")
    return start, end


def _require_pairwise_storage_disjoint_v1(values: Sequence[Any]) -> None:
    intervals = [_tensor_storage_interval_v1(value) for value in values]
    for left_index, (left_start, left_end) in enumerate(intervals):
        for right_start, right_end in intervals[left_index + 1 :]:
            if max(left_start, right_start) < min(left_end, right_end):
                raise OracleRegenerationCanaryError(
                    "FlowEdit tensors overlap in storage range"
                )


def tensor_content_sha256_v1(value: Any) -> str:
    import torch

    if not isinstance(value, torch.Tensor):
        raise OracleRegenerationCanaryError("tensor digest requires tensor")
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes_v1(
            {
                "schema_version": "bernini-tensor-content-sha256-v1",
                "shape": [int(item) for item in value.shape],
                "dtype": str(value.dtype),
                "byte_order": "native_torch_cpu_contiguous",
            }
        )
    )
    digest.update(_tensor_raw_bytes_v1(value))
    return digest.hexdigest()


def realized_gate_sha256_v1(
    *,
    delete: Any,
    create: Any,
    support: Any,
    preserve: Any,
    source_mask_sha256: str,
    variant: str,
) -> str:
    import torch

    _require_sha256(source_mask_sha256, label="source manual mask")
    if variant not in ALLOWED_GATE_VARIANTS:
        raise OracleRegenerationCanaryError("realized hard-gate variant differs")
    tensors = (delete, create, support, preserve)
    if any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bool
        or value.ndim != 5
        or not value.is_contiguous()
        for value in tensors
    ):
        raise OracleRegenerationCanaryError(
            "realized hard-gate digest requires contiguous bool tensors"
        )
    geometry = tuple(int(item) for item in support.shape)
    if any(tuple(int(item) for item in value.shape) != geometry for value in tensors):
        raise OracleRegenerationCanaryError("realized hard-gate digest geometry differs")
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes_v1(
            {
                "schema_version": "bernini-realized-hard-gate-tensor-v1",
                "source_mask_sha256": source_mask_sha256,
                "variant": variant,
                "geometry": list(geometry),
                "dtype": "bool",
                "tensor_order": ["delete", "create", "support", "preserve"],
            }
        )
    )
    for label, value in zip(("delete", "create", "support", "preserve"), tensors):
        payload = _tensor_raw_bytes_v1(value)
        digest.update(label.encode("ascii") + b"\x00")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _validate_owned_hard_gate_v1(
    gate: _OwnedHardStateChangeGateV1,
    *,
    expected_geometry: Optional[tuple[int, int, int, int, int]] = None,
) -> None:
    import torch

    if not isinstance(gate, _OwnedHardStateChangeGateV1):
        raise OracleRegenerationCanaryError("hard gate type differs")
    for label in ("delete", "create", "support", "preserve"):
        value = getattr(gate, label)
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.bool
            or value.ndim != 5
            or tuple(value.shape[:3]) != (1, 1, PHASE_COUNT)
            or not value.is_contiguous()
            or value.requires_grad
            or value.grad_fn is not None
        ):
            raise OracleRegenerationCanaryError(
                f"{label} must be exact bool [1,1,21,H,W]"
            )
    geometry = tuple(gate.support.shape)
    if any(tuple(getattr(gate, label).shape) != geometry for label in ("delete", "create", "preserve")):
        raise OracleRegenerationCanaryError("hard gate geometries differ")
    if any(
        getattr(gate, label).device != gate.support.device
        for label in ("delete", "create", "preserve")
    ):
        raise OracleRegenerationCanaryError("hard gate tensor devices differ")
    if expected_geometry is not None and geometry != expected_geometry:
        raise OracleRegenerationCanaryError("hard gate/runtime geometry differs")
    if not torch.equal(gate.support, torch.logical_or(gate.delete, gate.create)):
        raise OracleRegenerationCanaryError("hard support must equal delete OR create")
    if not torch.equal(gate.preserve, torch.logical_not(gate.support)):
        raise OracleRegenerationCanaryError("hard preserve must equal NOT support")
    if bool(gate.support[:, :, 0].any().item()):
        raise OracleRegenerationCanaryError("hard gate phase zero must be empty")
    if gate.variant not in ALLOWED_GATE_VARIANTS:
        raise OracleRegenerationCanaryError("hard gate variant differs")
    _require_sha256(gate.source_mask_sha256, label="source manual mask")
    _require_sha256(gate.realized_gate_sha256, label="realized hard gate")
    observed_digest = realized_gate_sha256_v1(
        delete=gate.delete,
        create=gate.create,
        support=gate.support,
        preserve=gate.preserve,
        source_mask_sha256=gate.source_mask_sha256,
        variant=gate.variant,
    )
    if observed_digest != gate.realized_gate_sha256:
        raise OracleRegenerationCanaryError(
            "realized hard-gate tensor/variant SHA-256 differs"
        )
    observed_counts = (
        int(gate.delete.sum().item()),
        int(gate.create.sum().item()),
    )
    if (
        type(gate.source_delete_count) is not int
        or type(gate.source_create_count) is not int
        or type(gate.realized_delete_count) is not int
        or type(gate.realized_create_count) is not int
        or (gate.realized_delete_count, gate.realized_create_count) != observed_counts
        or gate.permutation_mass_preserved not in (True, False, None)
        or min(
            gate.source_delete_count,
            gate.source_create_count,
            gate.realized_delete_count,
            gate.realized_create_count,
        )
        < 0
    ):
        raise OracleRegenerationCanaryError("hard gate mass receipt differs")
    if gate.variant in ("union", "time_shift_5", "spatial_shift"):
        if (
            gate.permutation_mass_preserved is not True
            or observed_counts
            != (gate.source_delete_count, gate.source_create_count)
        ):
            raise OracleRegenerationCanaryError(
                "permutation control did not preserve delete/create mass"
            )
    elif gate.variant == "create_only" and observed_counts != (
        0,
        gate.source_create_count,
    ):
        raise OracleRegenerationCanaryError("create-only mass receipt differs")
    elif gate.variant == "delete_only" and observed_counts != (
        gate.source_delete_count,
        0,
    ):
        raise OracleRegenerationCanaryError("delete-only mass receipt differs")
    elif gate.variant == "null" and observed_counts != (0, 0):
        raise OracleRegenerationCanaryError("null hard-gate mass receipt differs")
    if not isinstance(gate.provenance, str) or not gate.provenance:
        raise OracleRegenerationCanaryError("hard gate provenance is empty")


def _decode_rle_tensor(
    phases: Sequence[Sequence[tuple[int, int]]],
    *,
    height: int,
    width: int,
) -> Any:
    import torch

    value = torch.zeros(1, 1, PHASE_COUNT, height * width, dtype=torch.bool)
    for phase_index, spans in enumerate(phases):
        for start, length in spans:
            value[0, 0, phase_index, start : start + length] = True
    return value.reshape(1, 1, PHASE_COUNT, height, width)


def _require_validated_manifest_v1(manifest: Any) -> ValidatedOracleGateManifestV1:
    if (
        not isinstance(manifest, ValidatedOracleGateManifestV1)
        or manifest._validation_token is not _VALIDATED_MANIFEST_TOKEN
    ):
        raise OracleRegenerationCanaryError(
            "hard gate must originate from authenticated manifest validation"
        )
    return manifest


def _materialize_owned_hard_gate_v1(
    manifest: ValidatedOracleGateManifestV1,
    *,
    variant: str = "union",
    device: Any = "cpu",
) -> _OwnedHardStateChangeGateV1:
    """Private materializer; its tensor result is never an execution capability."""

    import torch

    manifest = _require_validated_manifest_v1(manifest)
    if variant not in ALLOWED_GATE_VARIANTS:
        raise OracleRegenerationCanaryError("unknown hard gate control variant")
    _, _, _, height, width = manifest.latent_geometry
    delete = _decode_rle_tensor(manifest.delete_rle, height=height, width=width)
    create = _decode_rle_tensor(manifest.create_rle, height=height, width=width)
    source_delete_count = int(delete.sum().item())
    source_create_count = int(create.sum().item())
    permutation_mass_preserved: Optional[bool] = None
    if variant == "time_shift_5":
        delete = torch.cat((delete[:, :, :1], torch.roll(delete[:, :, 1:], 5, dims=2)), dim=2)
        create = torch.cat((create[:, :, :1], torch.roll(create[:, :, 1:], 5, dims=2)), dim=2)
        permutation_mass_preserved = True
    elif variant == "spatial_shift":
        shifts = (max(1, height // 4), max(1, width // 3))
        delete = torch.roll(delete, shifts=shifts, dims=(-2, -1))
        create = torch.roll(create, shifts=shifts, dims=(-2, -1))
        permutation_mass_preserved = True
    elif variant == "create_only":
        delete = torch.zeros_like(delete)
    elif variant == "delete_only":
        create = torch.zeros_like(create)
    elif variant == "null":
        delete = torch.zeros_like(delete)
        create = torch.zeros_like(create)
    else:
        permutation_mass_preserved = True
    delete = delete.to(device=device).clone().detach().contiguous()
    create = create.to(device=device).clone().detach().contiguous()
    support = torch.logical_or(delete, create).contiguous()
    preserve = torch.logical_not(support).contiguous()
    realized_digest = realized_gate_sha256_v1(
        delete=delete,
        create=create,
        support=support,
        preserve=preserve,
        source_mask_sha256=manifest.mask_sha256,
        variant=variant,
    )
    gate = _OwnedHardStateChangeGateV1(
        delete=delete,
        create=create,
        support=support,
        preserve=preserve,
        provenance=(
            f"external-root-reviewed:{manifest.case_id}:"
            f"{manifest.annotation_authority_root_sha256}:{variant}"
        ),
        source_mask_sha256=manifest.mask_sha256,
        realized_gate_sha256=realized_digest,
        variant=variant,
        source_delete_count=source_delete_count,
        source_create_count=source_create_count,
        realized_delete_count=int(delete.sum().item()),
        realized_create_count=int(create.sum().item()),
        permutation_mass_preserved=permutation_mass_preserved,
    )
    _validate_owned_hard_gate_v1(gate, expected_geometry=manifest.latent_geometry)
    return gate


def revalidate_oracle_gate_manifest_v1(
    manifest: ValidatedOracleGateManifestV1,
    *,
    expected_annotation_authority_root_sha256: str,
) -> ValidatedOracleGateManifestV1:
    """Replay disk hashes, instruction bindings, receipt, and external root."""

    manifest = _require_validated_manifest_v1(manifest)
    if (
        manifest.annotation_authority_root_sha256
        != expected_annotation_authority_root_sha256
    ):
        raise OracleRegenerationCanaryError(
            "runtime/external annotation authority root differs"
        )
    return validate_oracle_gate_manifest_v1(
        manifest.path,
        expected_file_sha256=manifest.file_sha256,
        expected_review_receipt_sha256=manifest.review_receipt_sha256,
        expected_case_id=manifest.case_id,
        expected_source_sha256=manifest.source_sha256,
        expected_anchor_sha256=manifest.anchor_sha256,
        expected_action_caption_sha256=manifest.action_caption_sha256,
        expected_structured_action_program_sha256=(
            manifest.structured_action_program_sha256
        ),
        expected_annotation_authority_root_sha256=(
            expected_annotation_authority_root_sha256
        ),
        expected_latent_geometry=manifest.latent_geometry,
    )


def derive_regeneration_seed_v1(
    *, master_seed: int, case_id: str, candidate_index: int
) -> int:
    for label, value in (("master_seed", master_seed), ("candidate_index", candidate_index)):
        if type(value) is not int or value < 0:
            raise OracleRegenerationCanaryError(f"{label} must be non-negative int")
    if case_id not in ALLOWED_CASES:
        raise OracleRegenerationCanaryError("regeneration seed case differs")
    payload = (
        f"bernini-oracle-regeneration-target-step0-v1\0{master_seed}\0"
        f"{case_id}\0{candidate_index}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def draw_independent_keyed_gaussian_like_v1(
    reference: Any,
    *,
    master_seed: int,
    case_id: str,
    candidate_index: int,
) -> tuple[Any, Mapping[str, Any]]:
    import torch

    if (
        not isinstance(reference, torch.Tensor)
        or reference.dtype != torch.float32
        or reference.numel() <= 0
        or not reference.is_contiguous()
        or reference.requires_grad
        or reference.grad_fn is not None
        or not bool(torch.isfinite(reference).all().item())
    ):
        raise OracleRegenerationCanaryError(
            "independent regeneration noise reference must be finite fp32"
        )
    derived = derive_regeneration_seed_v1(
        master_seed=master_seed,
        case_id=case_id,
        candidate_index=candidate_index,
    )
    generator = torch.Generator(device="cpu").manual_seed(derived)
    cpu = torch.randn(
        tuple(int(item) for item in reference.shape),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    digest = hashlib.sha256(_tensor_raw_bytes_v1(cpu)).hexdigest()
    value = cpu.to(device=reference.device).clone().detach().contiguous()
    _require_pairwise_storage_disjoint_v1((reference, value))
    return value, {
        "schema_version": SCHEMA_VERSION,
        "domain": "target_regeneration_step0_only",
        "derived_seed": derived,
        "cpu_fp32_sha256": digest,
        "case_id": case_id,
        "candidate_index": candidate_index,
    }


def _expand_support(gate: _OwnedHardStateChangeGateV1, reference: Any) -> Any:
    import torch

    _validate_owned_hard_gate_v1(gate)
    if (
        not isinstance(reference, torch.Tensor)
        or reference.ndim != 5
        or tuple(reference.shape[:1] + reference.shape[2:])
        != tuple(gate.support.shape[:1] + gate.support.shape[2:])
    ):
        raise OracleRegenerationCanaryError("hard support/reference geometry differs")
    return gate.support.to(device=reference.device).expand_as(reference)


@dataclass(frozen=True)
class ValidatedFlowEditExecutionV1:
    constructor: Callable[..., tuple[Any, Any]]
    constructor_module: str
    constructor_qualname: str
    constructor_file_path: Path
    constructor_file_sha256: str
    receipt_path: Path
    receipt_sha256: str
    source_correlated_noise_sha256: str
    independent_target_noise_sha256: str
    source_tensor_sha256: str
    edit_tensor_sha256: str
    case_id: str
    sample_id: str
    gate_manifest_sha256: str
    gate_review_receipt_sha256: str
    annotation_authority_root_sha256: str
    gate_variant: str
    sigma_float64_hex: str
    step_index: int
    realized_gate_sha256: str
    _validation_token: Any = field(repr=False, compare=False, default=None)


def _validate_flowedit_tensor_set_v1(values: Sequence[Any]) -> None:
    import torch

    if len(values) != 4 or not all(isinstance(value, torch.Tensor) for value in values):
        raise OracleRegenerationCanaryError("FlowEdit values must be four tensors")
    source = values[0]
    if source.ndim != 5 or source.numel() <= 0:
        raise OracleRegenerationCanaryError(
            "FlowEdit oracle seam requires nonempty spatial [B,C,21,H,W] tensors"
        )
    for value in values:
        if (
            tuple(value.shape) != tuple(source.shape)
            or value.dtype != torch.float32
            or value.device != source.device
            or not value.is_contiguous()
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
        ):
            raise OracleRegenerationCanaryError(
                "FlowEdit tensors must share finite contiguous fp32 no-grad geometry/device"
            )
    _require_pairwise_storage_disjoint_v1(values)


def validate_flowedit_execution_receipt_v1(
    receipt_path: Path,
    *,
    expected_receipt_sha256: str,
    flowedit_constructor: Callable[..., tuple[Any, Any]],
    expected_constructor_file_sha256: str,
    validated_gate_manifest: ValidatedOracleGateManifestV1,
    gate_variant: str,
    sample_id: str,
    source: Any,
    edit: Any,
    source_correlated_noise: Any,
    independent_target_noise: Any,
    sigma: float,
    step_index: int,
) -> ValidatedFlowEditExecutionV1:
    """Bind exact constructor bytes and both noise tensors to an external pin."""

    manifest = _require_validated_manifest_v1(validated_gate_manifest)
    compiled_root = _compiled_case_sha256_v1(
        COMPILED_ANNOTATION_AUTHORITY_ROOTS,
        case_id=manifest.case_id,
        label="compiled annotation authority root",
    )
    compiled_receipt_sha256 = _compiled_case_sha256_v1(
        COMPILED_FLOWEDIT_BINDING_RECEIPT_SHA256,
        case_id=manifest.case_id,
        label="compiled FlowEdit binding receipt",
    )
    if (
        expected_receipt_sha256 != compiled_receipt_sha256
        or manifest.annotation_authority_root_sha256 != compiled_root
    ):
        raise OracleRegenerationCanaryError(
            "FlowEdit binding differs from compiled trust anchors"
        )
    manifest = revalidate_oracle_gate_manifest_v1(
        manifest,
        expected_annotation_authority_root_sha256=compiled_root,
    )
    gate = _materialize_owned_hard_gate_v1(manifest, variant=gate_variant)
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise OracleRegenerationCanaryError("FlowEdit sample id is empty")
    if type(step_index) is not int or step_index != 0:
        raise OracleRegenerationCanaryError(
            "FlowEdit receipt authorizes solver step zero only"
        )
    if (
        isinstance(sigma, bool)
        or not isinstance(sigma, (int, float))
        or not math.isfinite(float(sigma))
        or not 0.0 <= float(sigma) <= 1.0
    ):
        raise OracleRegenerationCanaryError("FlowEdit receipt sigma differs")
    _require_sha256(expected_receipt_sha256, label="FlowEdit execution receipt")
    _require_sha256(expected_constructor_file_sha256, label="FlowEdit constructor file")
    if not callable(flowedit_constructor):
        raise OracleRegenerationCanaryError("FlowEdit constructor is not callable")
    if not receipt_path.is_absolute() or not receipt_path.is_file():
        raise OracleRegenerationCanaryError(
            "FlowEdit receipt must be an existing absolute regular file"
        )
    if file_sha256_v1(receipt_path) != expected_receipt_sha256:
        raise OracleRegenerationCanaryError("FlowEdit receipt SHA-256 differs")
    constructor_source = inspect.getsourcefile(flowedit_constructor)
    if constructor_source is None:
        raise OracleRegenerationCanaryError("FlowEdit constructor source is unavailable")
    constructor_path = Path(constructor_source).resolve()
    if not constructor_path.is_file():
        raise OracleRegenerationCanaryError("FlowEdit constructor source file is absent")
    if file_sha256_v1(constructor_path) != expected_constructor_file_sha256:
        raise OracleRegenerationCanaryError("FlowEdit constructor source SHA-256 differs")
    constructor_module = getattr(flowedit_constructor, "__module__", None)
    constructor_qualname = getattr(flowedit_constructor, "__qualname__", None)
    if (
        not isinstance(constructor_module, str)
        or not constructor_module
        or not isinstance(constructor_qualname, str)
        or not constructor_qualname
    ):
        raise OracleRegenerationCanaryError("FlowEdit constructor identity differs")
    import torch

    values = (source, edit, source_correlated_noise, independent_target_noise)
    _validate_flowedit_tensor_set_v1(values)
    if (
        source.shape[0] != manifest.latent_geometry[0]
        or source.shape[1] <= 0
        or tuple(source.shape[2:]) != tuple(manifest.latent_geometry[2:])
    ):
        raise OracleRegenerationCanaryError(
            "FlowEdit tensors do not share bound gate latent coordinates"
        )
    source_sha256 = tensor_content_sha256_v1(source)
    edit_sha256 = tensor_content_sha256_v1(edit)
    correlated_sha256 = tensor_content_sha256_v1(source_correlated_noise)
    independent_sha256 = tensor_content_sha256_v1(independent_target_noise)
    receipt = strict_json_load_path_v1(receipt_path, label="FlowEdit execution receipt")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version") != FLOWEDIT_RECEIPT_SCHEMA_VERSION
        or receipt.get("constructor_module") != constructor_module
        or receipt.get("constructor_qualname") != constructor_qualname
        or receipt.get("constructor_file_path") != str(constructor_path)
        or receipt.get("constructor_file_sha256") != expected_constructor_file_sha256
        or receipt.get("case_id") != manifest.case_id
        or receipt.get("sample_id") != sample_id
        or receipt.get("source_sha256") != manifest.source_sha256
        or receipt.get("anchor_sha256") != manifest.anchor_sha256
        or receipt.get("action_caption_sha256") != manifest.action_caption_sha256
        or receipt.get("structured_action_program_sha256")
        != manifest.structured_action_program_sha256
        or receipt.get("gate_manifest_sha256") != manifest.file_sha256
        or receipt.get("gate_review_receipt_sha256")
        != manifest.review_receipt_sha256
        or receipt.get("annotation_authority_root_sha256") != compiled_root
        or receipt.get("gate_variant") != gate_variant
        or receipt.get("source_tensor_sha256") != source_sha256
        or receipt.get("edit_tensor_sha256") != edit_sha256
        or receipt.get("source_correlated_noise_sha256") != correlated_sha256
        or receipt.get("independent_target_noise_sha256") != independent_sha256
        or receipt.get("sigma_float64_hex") != float(sigma).hex()
        or receipt.get("step_index") != step_index
        or receipt.get("realized_gate_sha256") != gate.realized_gate_sha256
        or receipt.get("source_tensor_role")
        != "bound_source_latent_not_target_derived"
        or receipt.get("edit_tensor_role")
        != "bound_current_edit_state_not_teacher_target"
        or receipt.get("target_video_or_latent_used") is not False
        or receipt.get("diagnostic_only") is not True
        or receipt.get("training_target_authorized") is not False
        or receipt.get("accepted") is not True
    ):
        raise OracleRegenerationCanaryError("FlowEdit execution receipt differs")
    return ValidatedFlowEditExecutionV1(
        constructor=flowedit_constructor,
        constructor_module=constructor_module,
        constructor_qualname=constructor_qualname,
        constructor_file_path=constructor_path,
        constructor_file_sha256=expected_constructor_file_sha256,
        receipt_path=receipt_path,
        receipt_sha256=expected_receipt_sha256,
        source_correlated_noise_sha256=correlated_sha256,
        independent_target_noise_sha256=independent_sha256,
        source_tensor_sha256=source_sha256,
        edit_tensor_sha256=edit_sha256,
        case_id=manifest.case_id,
        sample_id=sample_id,
        gate_manifest_sha256=manifest.file_sha256,
        gate_review_receipt_sha256=manifest.review_receipt_sha256,
        annotation_authority_root_sha256=compiled_root,
        gate_variant=gate_variant,
        sigma_float64_hex=float(sigma).hex(),
        step_index=step_index,
        realized_gate_sha256=gate.realized_gate_sha256,
        _validation_token=_VALIDATED_FLOWEDIT_TOKEN,
    )


def _flowedit_step0_target_noise_v1(
    *,
    source: Any,
    edit: Any,
    source_correlated_noise: Any,
    independent_target_noise: Any,
    sigma: float,
    step_index: int,
    gate: _OwnedHardStateChangeGateV1,
    execution: ValidatedFlowEditExecutionV1,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Private tested core; direct calls have no release/execution authority."""

    import torch

    if type(step_index) is not int or step_index < 0:
        raise OracleRegenerationCanaryError("FlowEdit step index differs")
    if (
        isinstance(sigma, bool)
        or not isinstance(sigma, (int, float))
        or not math.isfinite(float(sigma))
        or not 0.0 <= float(sigma) <= 1.0
    ):
        raise OracleRegenerationCanaryError("FlowEdit sigma differs")
    values = (source, edit, source_correlated_noise, independent_target_noise)
    _validate_flowedit_tensor_set_v1(values)
    _validate_owned_hard_gate_v1(gate)
    if (
        not isinstance(execution, ValidatedFlowEditExecutionV1)
        or execution._validation_token is not _VALIDATED_FLOWEDIT_TOKEN
        or execution.step_index != step_index
        or execution.sigma_float64_hex != float(sigma).hex()
        or execution.gate_variant != gate.variant
        or execution.realized_gate_sha256 != gate.realized_gate_sha256
        or COMPILED_ANNOTATION_AUTHORITY_ROOTS.get(execution.case_id)
        != execution.annotation_authority_root_sha256
        or COMPILED_FLOWEDIT_BINDING_RECEIPT_SHA256.get(execution.case_id)
        != execution.receipt_sha256
        or file_sha256_v1(execution.receipt_path) != execution.receipt_sha256
        or file_sha256_v1(execution.constructor_file_path)
        != execution.constructor_file_sha256
        or tensor_content_sha256_v1(source_correlated_noise)
        != execution.source_correlated_noise_sha256
        or tensor_content_sha256_v1(independent_target_noise)
        != execution.independent_target_noise_sha256
        or tensor_content_sha256_v1(source) != execution.source_tensor_sha256
        or tensor_content_sha256_v1(edit) != execution.edit_tensor_sha256
    ):
        raise OracleRegenerationCanaryError(
            "FlowEdit execution/noise receipt is absent or changed"
        )
    support = _expand_support(gate, source)
    source_state, matched_target = execution.constructor(
        source, edit, source_correlated_noise, sigma=float(sigma)
    )
    for label, value in (("source state", source_state), ("matched target", matched_target)):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != source.shape
            or value.dtype != torch.float32
            or value.device != source.device
            or not value.is_contiguous()
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
        ):
            raise OracleRegenerationCanaryError(f"FlowEdit {label} differs")
    if step_index != 0 or not bool(gate.support.any().item()):
        reason = "nonzero_step" if step_index != 0 else "null_hard_support"
        return source_state, matched_target, {
            "step_index": step_index,
            "target_independent_noise_used": False,
            "inactive_reason": reason,
            "returned_target_is_original_matched_target_object": True,
            "flowedit_constructor_calls": 1,
        }
    mixed_target_noise = torch.where(
        support,
        independent_target_noise,
        source_correlated_noise,
    )
    _, regenerated_target = execution.constructor(
        source, edit, mixed_target_noise, sigma=float(sigma)
    )
    if (
        not isinstance(regenerated_target, torch.Tensor)
        or regenerated_target.shape != source.shape
        or regenerated_target.dtype != torch.float32
        or regenerated_target.device != source.device
        or not regenerated_target.is_contiguous()
        or regenerated_target.requires_grad
        or regenerated_target.grad_fn is not None
        or not bool(torch.isfinite(regenerated_target).all().item())
    ):
        raise OracleRegenerationCanaryError("FlowEdit regenerated target differs")
    outside = torch.logical_not(support)
    if not torch.equal(
        regenerated_target[outside].contiguous().view(torch.uint8),
        matched_target[outside].contiguous().view(torch.uint8),
    ):
        raise OracleRegenerationCanaryError(
            "FlowEdit target changed outside hard regeneration support"
        )
    if not torch.equal(
        regenerated_target[:, :, 0].contiguous().view(torch.uint8),
        matched_target[:, :, 0].contiguous().view(torch.uint8),
    ):
        raise OracleRegenerationCanaryError("FlowEdit phase zero changed")
    return source_state, regenerated_target, {
        "step_index": 0,
        "target_independent_noise_used": True,
        "source_state_uses_original_correlated_noise": True,
        "flowedit_constructor_called_for_source_and_mixed_target": True,
        "flowedit_constructor_calls": 2,
        "returned_target_is_original_matched_target_object": False,
        "execution_receipt_sha256": execution.receipt_sha256,
        "outside_hard_support_byte_exact": True,
        "phase_zero_byte_exact": True,
    }


def _packed_hard_support_v1(
    gate: _OwnedHardStateChangeGateV1,
    *,
    target_latent_shape: tuple[int, int, int, int, int],
    device: Any,
) -> Any:
    _validate_owned_hard_gate_v1(
        gate,
        expected_geometry=(
            target_latent_shape[0],
            1,
            target_latent_shape[2],
            target_latent_shape[3],
            target_latent_shape[4],
        ),
    )
    spatial = gate.support.to(device=device).expand(target_latent_shape)
    packed = sgaf._spatial_to_packed(spatial, target_latent_shape)
    return packed


def _scheduled_local_velocity_v1(
    *,
    sample: Any,
    high_r2v4_velocity: Any,
    official_v2v_velocity: Any,
    sigma: Any,
    gate: _OwnedHardStateChangeGateV1,
    target_latent_shape: tuple[int, int, int, int, int],
) -> tuple[Any, Mapping[str, Any]]:
    """Private adapter primitive; direct calls have no execution authority."""

    import torch

    if not all(isinstance(value, torch.Tensor) for value in (sample, official_v2v_velocity)):
        raise OracleRegenerationCanaryError("local velocity route requires sample/official tensors")
    if (
        sample.ndim != 3
        or sample.numel() <= 0
        or tuple(sample.shape) != tuple(official_v2v_velocity.shape)
        or sample.dtype != official_v2v_velocity.dtype
        or sample.device != official_v2v_velocity.device
        or not sample.dtype.is_floating_point
        or sample.requires_grad
        or sample.grad_fn is not None
        or official_v2v_velocity.requires_grad
        or official_v2v_velocity.grad_fn is not None
        or not bool(torch.isfinite(sample).all().item())
        or not bool(torch.isfinite(official_v2v_velocity).all().item())
    ):
        raise OracleRegenerationCanaryError(
            "sample/official must share finite floating no-grad dtype/device/geometry"
        )
    if (
        not isinstance(sigma, torch.Tensor)
        or sigma.ndim != 0
        or sigma.device.type != "cpu"
        or sigma.dtype != torch.float32
        or not bool(torch.isfinite(sigma).item())
        or not bool((sigma > 0).item())
    ):
        raise OracleRegenerationCanaryError("local velocity sigma must be CPU fp32 positive")
    packed_support = _packed_hard_support_v1(
        gate,
        target_latent_shape=target_latent_shape,
        device=official_v2v_velocity.device,
    )
    if tuple(packed_support.shape) != tuple(official_v2v_velocity.shape):
        raise OracleRegenerationCanaryError("packed hard support geometry differs")
    support_nonzero = bool(packed_support.any().item())
    sigma_float = float(sigma.item())
    if sigma_float <= homotopy.SIGMA_LOW:
        endpoint = "low_official_v2v_apg"
        high_weight = 0.0
    elif sigma_float >= homotopy.SIGMA_HIGH:
        endpoint = "high_r2v4_apg"
        high_weight = 1.0
    else:
        weight = homotopy.smoothstep_high_branch_weight(sigma)
        endpoint = "transition"
        high_weight = float(weight.item())
    low_weight = 1.0 - high_weight
    if not support_nonzero:
        return official_v2v_velocity, {
            "schema_version": LOCAL_NATIVE_SCHEMA_VERSION,
            "sigma": sigma_float,
            "high_r2v4_weight": high_weight,
            "low_official_v2v_apg_weight": low_weight,
            "endpoint": endpoint,
            "scheduled_endpoint_prelocal": endpoint,
            "scheduled_expert_evaluated": False,
            "high_velocity_aggregated": False,
            "scheduled_endpoint_prelocal_direct_return_verified": None,
            "executed_local_where": False,
            "null_gate": True,
            "scheduler_received_original_official_object": True,
            "outside_hard_support_byte_exact": True,
            "hard_support_fraction": 0.0,
            "realized_gate_sha256": gate.realized_gate_sha256,
        }
    if (
        not isinstance(high_r2v4_velocity, torch.Tensor)
        or tuple(high_r2v4_velocity.shape) != tuple(sample.shape)
        or high_r2v4_velocity.dtype != sample.dtype
        or high_r2v4_velocity.device != sample.device
        or high_r2v4_velocity.requires_grad
        or high_r2v4_velocity.grad_fn is not None
        or not bool(torch.isfinite(high_r2v4_velocity).all().item())
    ):
        raise OracleRegenerationCanaryError(
            "active high velocity must share finite no-grad dtype/device/geometry"
        )
    try:
        scheduled = homotopy.native_branch_homotopy_step(
            sample,
            high_r2v4_velocity,
            official_v2v_velocity,
            sigma,
            high_r2v4_momentum=0.0,
            low_official_v2v_apg_momentum=0.0,
        )
    except Exception as error:
        raise OracleRegenerationCanaryError(str(error)) from error
    expert = scheduled.velocity
    direct_verified = (
        (scheduled.endpoint == "high_r2v4_apg" and expert is high_r2v4_velocity)
        or (scheduled.endpoint == "low_official_v2v_apg" and expert is official_v2v_velocity)
    )
    if scheduled.endpoint != "transition" and not direct_verified:
        raise OracleRegenerationCanaryError(
            "scheduled prelocal endpoint did not directly return its branch object"
        )
    if expert is official_v2v_velocity:
        return official_v2v_velocity, {
            **scheduled.trace_dict(),
            "scheduled_endpoint_prelocal": scheduled.endpoint,
            "scheduled_expert_evaluated": True,
            "high_velocity_aggregated": False,
            "scheduled_endpoint_prelocal_direct_return_verified": True,
            "executed_local_where": False,
            "null_gate": False,
            "scheduler_received_original_official_object": True,
            "outside_hard_support_byte_exact": True,
            "hard_support_fraction": float(packed_support.float().mean().item()),
            "realized_gate_sha256": gate.realized_gate_sha256,
        }
    executed = torch.where(packed_support, expert, official_v2v_velocity)
    outside = torch.logical_not(packed_support)
    if not torch.equal(
        executed[outside].contiguous().view(torch.uint8),
        official_v2v_velocity[outside].contiguous().view(torch.uint8),
    ):
        raise OracleRegenerationCanaryError(
            "local execution changed official bytes outside hard support"
        )
    return executed, {
        **scheduled.trace_dict(),
        "scheduled_endpoint_prelocal": scheduled.endpoint,
        "scheduled_expert_evaluated": True,
        "high_velocity_aggregated": True,
        "scheduled_endpoint_prelocal_direct_return_verified": (
            direct_verified if scheduled.endpoint != "transition" else False
        ),
        "executed_local_where": True,
        "null_gate": False,
        "scheduler_received_original_official_object": False,
        "outside_hard_support_byte_exact": True,
        "hard_support_fraction": float(packed_support.float().mean().item()),
        "realized_gate_sha256": gate.realized_gate_sha256,
    }


def _compiled_case_sha256_v1(
    pins: Mapping[str, str], *, case_id: str, label: str
) -> str:
    value = pins.get(case_id)
    if value is None:
        raise OracleRegenerationCanaryError(
            f"{label} is not compiled into this safe-BLOCKED scaffold"
        )
    return _require_sha256(value, label=label)


@dataclass(frozen=True)
class ValidatedNativeExecutionBindingV1:
    manifest: ValidatedOracleGateManifestV1
    gate_variant: str
    realized_gate_sha256: str
    sample_id: str
    source_latent_sha256: str
    source_reference_latent_sha256: tuple[str, str, str, str]
    source_reference_rgb_indices: tuple[int, int, int, int]
    r2v_action_prompt_sha256: str
    r2v_action_prompt_embeds: Any = field(repr=False, compare=False)
    receipt_path: Path
    receipt_sha256: str
    _validation_token: Any = field(repr=False, compare=False, default=None)


def validate_native_execution_binding_receipt_v1(
    receipt_path: Path,
    *,
    expected_receipt_sha256: str,
    validated_gate_manifest: ValidatedOracleGateManifestV1,
    gate_variant: str,
    sample_id: str,
    source_video_latent: Any,
    source_reference_latents: Sequence[Any],
    source_reference_rgb_indices: Sequence[int],
    r2v_action_prompt_embeds: Any,
) -> ValidatedNativeExecutionBindingV1:
    """Mint the only public native execution capability.

    Both the manual annotation root and this exact execution-binding receipt
    must be compiled into the reviewed module.  In the checked-in scaffold the
    compiled maps are empty, so production minting always fails closed.  Unit
    tests temporarily replace those immutable module bindings to exercise the
    downstream fake runtime; there is no launcher path that does so.
    """

    import torch

    manifest = _require_validated_manifest_v1(validated_gate_manifest)
    compiled_root = _compiled_case_sha256_v1(
        COMPILED_ANNOTATION_AUTHORITY_ROOTS,
        case_id=manifest.case_id,
        label="compiled annotation authority root",
    )
    compiled_receipt_sha256 = _compiled_case_sha256_v1(
        COMPILED_NATIVE_BINDING_RECEIPT_SHA256,
        case_id=manifest.case_id,
        label="compiled native binding receipt",
    )
    if (
        expected_receipt_sha256 != compiled_receipt_sha256
        or manifest.annotation_authority_root_sha256 != compiled_root
    ):
        raise OracleRegenerationCanaryError(
            "native execution binding differs from compiled trust anchors"
        )
    manifest = revalidate_oracle_gate_manifest_v1(
        manifest,
        expected_annotation_authority_root_sha256=compiled_root,
    )
    if gate_variant not in ALLOWED_GATE_VARIANTS:
        raise OracleRegenerationCanaryError("native binding gate variant differs")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise OracleRegenerationCanaryError("native binding sample id is empty")
    if (
        len(source_reference_latents) != 4
        or len(source_reference_rgb_indices) != 4
        or any(type(value) is not int or value < 0 for value in source_reference_rgb_indices)
        or tuple(source_reference_rgb_indices) != (0, 27, 53, 80)
    ):
        raise OracleRegenerationCanaryError("native four-reference binding differs")
    tensors = (source_video_latent, *source_reference_latents)
    if any(
        not isinstance(value, torch.Tensor)
        or value.ndim != 5
        or value.shape[0] != 1
        or value.shape[1] != 16
        or value.numel() <= 0
        or not value.dtype.is_floating_point
        or not value.is_contiguous()
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
        for value in tensors
    ):
        raise OracleRegenerationCanaryError(
            "native source/reference binding tensors differ"
        )
    expected_source_shape = (
        manifest.latent_geometry[0],
        16,
        manifest.latent_geometry[2],
        manifest.latent_geometry[3],
        manifest.latent_geometry[4],
    )
    expected_reference_shape = (
        manifest.latent_geometry[0],
        16,
        1,
        manifest.latent_geometry[3],
        manifest.latent_geometry[4],
    )
    if tuple(source_video_latent.shape) != expected_source_shape or any(
        tuple(value.shape) != expected_reference_shape
        or value.dtype != source_video_latent.dtype
        or value.device != source_video_latent.device
        for value in source_reference_latents
    ):
        raise OracleRegenerationCanaryError(
            "native source/reference binding geometry differs"
        )
    if (
        not isinstance(r2v_action_prompt_embeds, torch.Tensor)
        or r2v_action_prompt_embeds.ndim != 3
        or r2v_action_prompt_embeds.shape[0] != 1
        or r2v_action_prompt_embeds.numel() <= 0
        or not r2v_action_prompt_embeds.dtype.is_floating_point
        or not r2v_action_prompt_embeds.is_contiguous()
        or r2v_action_prompt_embeds.requires_grad
        or r2v_action_prompt_embeds.grad_fn is not None
        or not bool(torch.isfinite(r2v_action_prompt_embeds).all().item())
    ):
        raise OracleRegenerationCanaryError("native R2V action prompt binding differs")
    source_sha256 = tensor_content_sha256_v1(source_video_latent)
    reference_sha256 = tuple(
        tensor_content_sha256_v1(value) for value in source_reference_latents
    )
    prompt_sha256 = tensor_content_sha256_v1(r2v_action_prompt_embeds)
    gate = _materialize_owned_hard_gate_v1(manifest, variant=gate_variant)
    if not receipt_path.is_absolute() or not receipt_path.is_file():
        raise OracleRegenerationCanaryError(
            "native binding receipt must be an existing absolute regular file"
        )
    if file_sha256_v1(receipt_path) != compiled_receipt_sha256:
        raise OracleRegenerationCanaryError("native binding receipt bytes differ")
    receipt = strict_json_load_path_v1(receipt_path, label="native binding receipt")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version") != NATIVE_BINDING_RECEIPT_SCHEMA_VERSION
        or receipt.get("case_id") != manifest.case_id
        or receipt.get("sample_id") != sample_id
        or receipt.get("source_sha256") != manifest.source_sha256
        or receipt.get("anchor_sha256") != manifest.anchor_sha256
        or receipt.get("action_caption_sha256") != manifest.action_caption_sha256
        or receipt.get("structured_action_program_sha256")
        != manifest.structured_action_program_sha256
        or receipt.get("gate_manifest_sha256") != manifest.file_sha256
        or receipt.get("gate_review_receipt_sha256")
        != manifest.review_receipt_sha256
        or receipt.get("annotation_authority_root_sha256") != compiled_root
        or receipt.get("annotation_authority_leaf_sha256")
        != manifest.annotation_authority_leaf_sha256
        or receipt.get("gate_variant") != gate_variant
        or receipt.get("realized_gate_sha256") != gate.realized_gate_sha256
        or receipt.get("source_latent_sha256") != source_sha256
        or receipt.get("source_reference_latent_sha256") != list(reference_sha256)
        or receipt.get("source_reference_rgb_indices")
        != list(source_reference_rgb_indices)
        or receipt.get("r2v_action_prompt_sha256") != prompt_sha256
        or receipt.get("source_latent_role")
        != "official_vae_encode_of_bound_source_media"
        or receipt.get("r2v_action_prompt_role")
        != "bound_action_caption_not_target_derived"
        or receipt.get("target_video_or_latent_used") is not False
        or receipt.get("diagnostic_only") is not True
        or receipt.get("training_target_authorized") is not False
        or receipt.get("accepted") is not True
    ):
        raise OracleRegenerationCanaryError("native execution binding receipt differs")
    return ValidatedNativeExecutionBindingV1(
        manifest=manifest,
        gate_variant=gate_variant,
        realized_gate_sha256=gate.realized_gate_sha256,
        sample_id=sample_id,
        source_latent_sha256=source_sha256,
        source_reference_latent_sha256=reference_sha256,  # type: ignore[arg-type]
        source_reference_rgb_indices=tuple(source_reference_rgb_indices),  # type: ignore[arg-type]
        r2v_action_prompt_sha256=prompt_sha256,
        r2v_action_prompt_embeds=(
            r2v_action_prompt_embeds.clone().detach().contiguous()
        ),
        receipt_path=receipt_path,
        receipt_sha256=compiled_receipt_sha256,
        _validation_token=_VALIDATED_NATIVE_BINDING_TOKEN,
    )


class LocalOracleNativeBranchRuntimePatchV1(
    native_runtime.NativeBranchHomotopyRuntimePatch
):
    """Authenticated five-forward runtime with local hard P/G execution."""

    def __init__(
        self,
        diffusion: Any,
        *,
        config: native_runtime.NativeBranchHomotopyRuntimeConfig,
        native_execution_binding: ValidatedNativeExecutionBindingV1,
        expected_bernini_commit: str = native_runtime.PINNED_BERNINI_COMMIT,
        observed_wan_diffusion_sha256: str = native_runtime.PINNED_WAN_DIFFUSION_SHA256,
    ) -> None:
        if (
            not isinstance(
                native_execution_binding, ValidatedNativeExecutionBindingV1
            )
            or native_execution_binding._validation_token
            is not _VALIDATED_NATIVE_BINDING_TOKEN
        ):
            raise OracleRegenerationCanaryError(
                "native runtime requires validated execution-binding capability"
            )
        binding = native_execution_binding
        compiled_root = _compiled_case_sha256_v1(
            COMPILED_ANNOTATION_AUTHORITY_ROOTS,
            case_id=binding.manifest.case_id,
            label="compiled annotation authority root",
        )
        compiled_binding_sha = _compiled_case_sha256_v1(
            COMPILED_NATIVE_BINDING_RECEIPT_SHA256,
            case_id=binding.manifest.case_id,
            label="compiled native binding receipt",
        )
        if (
            binding.receipt_sha256 != compiled_binding_sha
            or file_sha256_v1(binding.receipt_path) != compiled_binding_sha
            or binding.manifest.annotation_authority_root_sha256 != compiled_root
        ):
            raise OracleRegenerationCanaryError(
                "native execution-binding capability changed or is not compiled"
            )
        manifest = revalidate_oracle_gate_manifest_v1(
            binding.manifest,
            expected_annotation_authority_root_sha256=compiled_root,
        )
        owned_gate = _materialize_owned_hard_gate_v1(
            manifest,
            variant=binding.gate_variant,
            device="cpu",
        )
        if owned_gate.realized_gate_sha256 != binding.realized_gate_sha256:
            raise OracleRegenerationCanaryError(
                "native binding/owned realized gate digest differs"
            )
        _validate_owned_hard_gate_v1(
            owned_gate,
            expected_geometry=(
                config.target_latent_shape[0],
                1,
                config.target_latent_shape[2],
                config.target_latent_shape[3],
                config.target_latent_shape[4],
            ),
        )
        self._validated_gate_manifest = manifest
        self._native_execution_binding = binding
        self._expected_annotation_authority_root_sha256 = compiled_root
        self._owned_hard_gate = owned_gate
        self._expected_realized_gate_sha256 = owned_gate.realized_gate_sha256
        self._live_native_binding_tensors: Optional[tuple[Any, ...]] = None
        super().__init__(
            diffusion,
            r2v_action_prompt_embeds=binding.r2v_action_prompt_embeds,
            config=config,
            expected_bernini_commit=expected_bernini_commit,
            observed_wan_diffusion_sha256=observed_wan_diffusion_sha256,
        )

    def _validate_sample_contract(self, values: Mapping[str, Any]) -> Any:
        state = super()._validate_sample_contract(values)
        if self._live_native_binding_tensors is not None:
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "live native binding tensors were already captured"
            )
        self._live_native_binding_tensors = (
            state.source_video,
            *state.references,
            state.high_action_prompt,
        )
        self._certify_live_native_binding_tensors()
        return state

    def _certify_live_native_binding_tensors(self) -> None:
        binding = self._native_execution_binding
        values = self._live_native_binding_tensors
        if values is None or len(values) != 6:
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "live native binding tensor snapshot is absent"
            )
        source_video, *tail = values
        references = tuple(tail[:4])
        high_action_prompt = tail[4]
        if (
            tensor_content_sha256_v1(source_video) != binding.source_latent_sha256
            or tuple(
                tensor_content_sha256_v1(value) for value in references
            )
            != binding.source_reference_latent_sha256
            or tensor_content_sha256_v1(high_action_prompt)
            != binding.r2v_action_prompt_sha256
        ):
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "live source/reference/action tensors differ from native capability"
            )

    def _certify_owned_gate_snapshot(self, *, revalidate_files: bool) -> None:
        self._certify_live_native_binding_tensors()
        _validate_owned_hard_gate_v1(
            self._owned_hard_gate,
            expected_geometry=(
                self.config.target_latent_shape[0],
                1,
                self.config.target_latent_shape[2],
                self.config.target_latent_shape[3],
                self.config.target_latent_shape[4],
            ),
        )
        if (
            self._owned_hard_gate.realized_gate_sha256
            != self._expected_realized_gate_sha256
        ):
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "owned hard-gate snapshot digest changed"
            )
        manifest = self._validated_gate_manifest
        binding = self._native_execution_binding
        if (
            file_sha256_v1(manifest.path) != manifest.file_sha256
            or file_sha256_v1(manifest.review_receipt_path)
            != manifest.review_receipt_sha256
            or file_sha256_v1(binding.receipt_path) != binding.receipt_sha256
            or COMPILED_ANNOTATION_AUTHORITY_ROOTS.get(manifest.case_id)
            != self._expected_annotation_authority_root_sha256
            or COMPILED_NATIVE_BINDING_RECEIPT_SHA256.get(manifest.case_id)
            != binding.receipt_sha256
        ):
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "compiled root, manual gate, or execution receipt changed"
            )
        if revalidate_files:
            try:
                revalidate_oracle_gate_manifest_v1(
                    manifest,
                    expected_annotation_authority_root_sha256=(
                        self._expected_annotation_authority_root_sha256
                    ),
                )
            except OracleRegenerationCanaryError as error:
                raise native_runtime.NativeBranchHomotopyRuntimeError(str(error)) from error

    def _wrapped_scheduler_step(self, *args: Any, **kwargs: Any) -> Any:
        """Copy the frozen authentication seam, replacing only global blending."""

        import torch

        state = self._active
        if state is None:
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "scheduler.step ran outside authenticated sample"
            )
        self._certify_owned_gate_snapshot(revalidate_files=False)
        if (
            len(state.patch_results) != 10
            or tuple(item.source_id for item in state.patch_results)
            != native_runtime.EXPECTED_PATCH_SOURCE_IDS
            or len(state.low_forwards) != 2
            or len(state.high_forwards) != 3
        ):
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "scheduler.step arrived before five-forward closure"
            )
        official = sgaf._extract_argument(args, kwargs, index=0, name="model_output")
        timestep = sgaf._extract_argument(args, kwargs, index=1, name="timestep")
        sample = sgaf._extract_argument(args, kwargs, index=2, name="sample")
        try:
            _certify_expanded_timestep_compat_v1(
                state.low_forwards[1].values["timesteps"], timestep
            )
        except Exception as error:
            raise native_runtime._raise_from_sgaf(error) from error
        expected_shape = (
            1,
            self.config.target_patch_tokens,
            self.config.target_latent_shape[1] * 4,
        )
        for label, value in (
            ("official model_output", official),
            ("scheduler sample", sample),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or native_runtime._shape(value, label=label) != expected_shape
                or not value.dtype.is_floating_point
                or value.requires_grad
                or value.grad_fn is not None
                or not bool(torch.isfinite(value).all().item())
            ):
                raise native_runtime.NativeBranchHomotopyRuntimeError(
                    f"{label} geometry differs"
                )
        if official.device != sample.device or official.dtype != sample.dtype:
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "official output/sample dtype or device differs"
            )
        expected_target_patch_input = sgaf._packed_to_spatial(
            sample, self.config.target_latent_shape
        ).to(dtype=self.transformer.dtype)
        observed_target_patch_input = state.patch_results[9].input_value
        if (
            not isinstance(observed_target_patch_input, torch.Tensor)
            or observed_target_patch_input.shape != expected_target_patch_input.shape
            or observed_target_patch_input.device != expected_target_patch_input.device
            or observed_target_patch_input.dtype != expected_target_patch_input.dtype
            or not _tensor_bytes_equal_v1(
                observed_target_patch_input, expected_target_patch_input
            )
        ):
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "captured target patch input differs from scheduler sample"
            )
        try:
            step_index, sigma, sigma_float = sgaf._resolve_sigma(
                self.scheduler, timestep
            )
        except Exception as error:
            raise native_runtime._raise_from_sgaf(error) from error
        if step_index != state.completed_steps:
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "scheduler step index differs from local runtime state"
            )
        if (
            not isinstance(sigma, torch.Tensor)
            or sigma.ndim != 0
            or sigma.device.type != "cpu"
            or sigma.dtype != torch.float32
        ):
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "active UniPC sigma must remain a CPU fp32 scalar"
            )
        low_parameters = sgaf._APGParameters(
            guidance_scale=self.config.omega_text,
            eta=self.config.eta,
            norm_threshold=self.config.image_norm_threshold,
            momentum=0.0,
        )
        rebuilt_low = sgaf._guided_velocity(
            sample,
            state.low_forwards[0].target_tail,
            state.low_forwards[1].target_tail,
            sigma,
            shape=self.config.target_latent_shape,
            parameters=low_parameters,
            momentum_buffer=state.low_momentum,
            output_like=official,
        )
        parity_delta = rebuilt_low.float() - official.float()
        parity_rms = native_runtime._tensor_rms(parity_delta)
        parity_max = float(parity_delta.abs().max().item())
        if not _tensor_bytes_equal_v1(rebuilt_low, official):
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "locally rebuilt low V2V APG bytes differ from official model_output: "
                f"max_abs={parity_max:.9g} rms={parity_rms:.9g}"
            )
        gate_nonzero = bool(self._owned_hard_gate.support.any().item())
        high = (
            self._high_r2v4_velocity(
                state, sample=sample, sigma=sigma, official=official
            )
            if gate_nonzero
            else None
        )
        try:
            executed, local_trace = _scheduled_local_velocity_v1(
                sample=sample,
                high_r2v4_velocity=high,
                official_v2v_velocity=official,
                sigma=sigma,
                gate=self._owned_hard_gate,
                target_latent_shape=self.config.target_latent_shape,
            )
        except OracleRegenerationCanaryError as error:
            raise native_runtime.NativeBranchHomotopyRuntimeError(str(error)) from error
        if executed is official:
            call_args, call_kwargs = tuple(args), dict(kwargs)
        else:
            call_args, call_kwargs = native_runtime._replace(
                self.original_scheduler_step,
                args,
                kwargs,
                name="model_output",
                value=executed,
            )
        result = self.original_scheduler_step(*call_args, **call_kwargs)
        self.original_scheduler_call_count += 1
        state.completed_steps += 1
        high_low_rms = (
            native_runtime._tensor_rms(high.float() - official.float())
            if high is not None
            else None
        )
        self.trace.append(
            {
                "step_index": step_index,
                "timestep": native_runtime._scalar(timestep, label="timestep"),
                "sigma": sigma_float,
                "forward_order": list(native_runtime.PER_STEP_FORWARD_ORDER),
                "transformer_forwards": 5,
                "low_vi_forwards": 2,
                "high_r2v4_forwards": 3,
                "high_forwards_executed": True,
                "original_scheduler_calls": 1,
                "patch_call_count": 10,
                "patch_source_ids": list(native_runtime.EXPECTED_PATCH_SOURCE_IDS),
                "target_patch_tokens_P": self.config.target_patch_tokens,
                "reference_patch_tokens_R": self.config.reference_patch_tokens,
                "low_vi_total_tokens": self.config.low_vi_tokens,
                "high_i_total_tokens": self.config.high_i_tokens,
                "low_official_apg_exact_parity": True,
                "low_official_apg_parity_rms": parity_rms,
                "low_official_apg_parity_max_abs": parity_max,
                "high_low_velocity_delta_rms": high_low_rms,
                "vendor_high_apg_function": (
                    f"{self.vendor_chain.__module__}.{self.vendor_chain.__name__}"
                ),
                **local_trace,
                "schema_version": LOCAL_NATIVE_SCHEMA_VERSION,
                "scheduler_received_original_model_output_object": executed is official,
                "hard_gate_variant": self._owned_hard_gate.variant,
                "source_manual_mask_sha256": self._owned_hard_gate.source_mask_sha256,
                "hard_gate_dtype": "bool",
                "soft_gate_used": False,
                "freeze_safe_no_grad_outputs": all(
                    not value.requires_grad and value.grad_fn is None
                    for value in (
                        (official, executed)
                        if high is None
                        else (official, high, executed)
                    )
                ),
            }
        )
        state.patch_results.clear()
        state.low_forwards.clear()
        state.high_forwards.clear()
        return result

    def finalize(self) -> Mapping[str, Any]:
        if not self.restored or self.finalized:
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "local oracle patch finalize differs"
            )
        self._certify_owned_gate_snapshot(revalidate_files=True)
        steps = self.config.expected_steps
        if (
            self.sample_call_count != 1
            or self.schedule_preflight is None
            or self.patch_call_count != 10 * steps
            or self.low_forward_count != 2 * steps
            or self.high_forward_count != 3 * steps
            or self.original_scheduler_call_count != steps
            or len(self.trace) != steps
        ):
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "local oracle runtime call-count certificate differs"
            )
        if any(
            row.get("low_official_apg_exact_parity") is not True
            or row.get("transformer_forwards") != 5
            or row.get("original_scheduler_calls") != 1
            or row.get("patch_source_ids")
            != list(native_runtime.EXPECTED_PATCH_SOURCE_IDS)
            or row.get("hard_gate_dtype") != "bool"
            or row.get("soft_gate_used") is not False
            or row.get("high_forwards_executed") is not True
            or row.get("outside_hard_support_byte_exact") is not True
            or row.get("realized_gate_sha256")
            != self._expected_realized_gate_sha256
            or row.get("freeze_safe_no_grad_outputs") is not True
            for row in self.trace
        ):
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "local hard-gate trace certificate differs"
            )
        expected_endpoints = (
            ["high_r2v4_apg"] * 15
            + ["transition"] * 16
            + ["low_official_v2v_apg"] * 9
        )
        observed_endpoints = [
            str(row.get("scheduled_endpoint_prelocal")) for row in self.trace
        ]
        if observed_endpoints != expected_endpoints:
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "exact40 local scheduled endpoint partition differs"
            )
        realized_null = not bool(self._owned_hard_gate.support.any().item())
        if realized_null:
            if any(
                row.get("scheduled_expert_evaluated") is not False
                or row.get("high_velocity_aggregated") is not False
                or row.get("scheduled_endpoint_prelocal_direct_return_verified")
                is not None
                or row.get("executed_local_where") is not False
                or row.get("scheduler_received_original_model_output_object")
                is not True
                for row in self.trace
            ):
                raise native_runtime.NativeBranchHomotopyRuntimeError(
                    "realized null-gate direct-official certificate differs"
                )
        else:
            if any(
                row.get("scheduled_expert_evaluated") is not True
                for row in self.trace
            ) or any(
                row.get("scheduled_endpoint_prelocal_direct_return_verified")
                is not True
                or row.get("high_velocity_aggregated") is not True
                or row.get("executed_local_where") is not True
                or row.get("scheduler_received_original_model_output_object")
                is not False
                for row in self.trace[:15]
            ) or any(
                row.get("scheduled_endpoint_prelocal_direct_return_verified")
                is not False
                or row.get("high_velocity_aggregated") is not True
                or row.get("executed_local_where") is not True
                or row.get("scheduler_received_original_model_output_object")
                is not False
                for row in self.trace[15:31]
            ) or any(
                row.get("scheduled_endpoint_prelocal_direct_return_verified")
                is not True
                or row.get("high_velocity_aggregated") is not False
                or row.get("executed_local_where") is not False
                or row.get("scheduler_received_original_model_output_object")
                is not True
                for row in self.trace[31:]
            ):
                raise native_runtime.NativeBranchHomotopyRuntimeError(
                    "local scheduled/executed route certificate differs"
                )
        self.finalized = True
        manifest = self._validated_gate_manifest
        gate = self._owned_hard_gate
        return {
            "schema_version": LOCAL_NATIVE_SCHEMA_VERSION,
            "execution": "scheduled_r2v4_inside_hard_G_official_v2v_outside",
            "sample_calls": 1,
            "steps": steps,
            "transformer_forwards": self.low_forward_count + self.high_forward_count,
            "low_vi_forwards": self.low_forward_count,
            "high_r2v4_forwards": self.high_forward_count,
            "patch_vae_latent_calls": self.patch_call_count,
            "original_scheduler_calls": self.original_scheduler_call_count,
            "per_step_forward_order": list(native_runtime.PER_STEP_FORWARD_ORDER),
            "per_step_patch_source_ids": list(
                native_runtime.EXPECTED_PATCH_SOURCE_IDS
            ),
            "schedule_preflight": dict(self.schedule_preflight),
            "target_patch_tokens_P": self.config.target_patch_tokens,
            "reference_patch_tokens_R": self.config.reference_patch_tokens,
            "low_vi_total_tokens": self.config.low_vi_tokens,
            "high_i_total_tokens": self.config.high_i_tokens,
            "low_official_apg_byte_exact_parity_all_steps": True,
            "smoothstep_sigma_low": homotopy.SIGMA_LOW,
            "smoothstep_sigma_high": homotopy.SIGMA_HIGH,
            "exact40_scheduled_endpoint_partition": {
                "high_r2v4_apg_indices": list(range(0, 15)),
                "transition_indices": list(range(15, 31)),
                "low_official_v2v_apg_indices": list(range(31, 40)),
            },
            "scheduler_mutation_surface": "model_output_argument_only",
            "hard_gate_variant": gate.variant,
            "source_manual_mask_sha256": gate.source_mask_sha256,
            "realized_gate_sha256": gate.realized_gate_sha256,
            "hard_gate_dtype": "bool",
            "soft_gate_used": False,
            "gate_mass_receipt": {
                "source_delete_count": gate.source_delete_count,
                "source_create_count": gate.source_create_count,
                "realized_delete_count": gate.realized_delete_count,
                "realized_create_count": gate.realized_create_count,
                "permutation_mass_preserved": gate.permutation_mass_preserved,
            },
            "manual_gate_authority": {
                "gate_manifest_sha256": manifest.file_sha256,
                "review_receipt_sha256": manifest.review_receipt_sha256,
                "external_ledger_root_sha256": (
                    manifest.annotation_authority_root_sha256
                ),
                "annotation_leaf_sha256": (
                    manifest.annotation_authority_leaf_sha256
                ),
                "annotation_leaf_index": manifest.annotation_authority_leaf_index,
                "annotation_tree_size": manifest.annotation_authority_tree_size,
            },
            "native_execution_binding": {
                "sample_id": self._native_execution_binding.sample_id,
                "receipt_sha256": self._native_execution_binding.receipt_sha256,
                "source_latent_sha256": (
                    self._native_execution_binding.source_latent_sha256
                ),
                "source_reference_latent_sha256": list(
                    self._native_execution_binding.source_reference_latent_sha256
                ),
                "source_reference_rgb_indices": list(
                    self._native_execution_binding.source_reference_rgb_indices
                ),
                "r2v_action_prompt_sha256": (
                    self._native_execution_binding.r2v_action_prompt_sha256
                ),
                "compiled_trust_anchor_verified": True,
            },
            "outside_G_official_bytes_exact_all_steps": True,
            "G_zero_direct_official_object_capability": True,
            "realized_gate_is_G_zero": realized_null,
            "realized_G_zero_direct_official_object_all_steps": (
                True if realized_null else None
            ),
            "all40_raw_high_mode_available": False,
            "selection_authority": None,
            "automatic_replacement_of_successful_base_authorized": False,
            "vendor_source_modified": False,
            "training_performed": False,
            "optimizer_created": False,
            "parameters_updated": False,
            "gpu_launch_authorized": False,
            "trace": list(self.trace),
        }


def contract_v1() -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "cases": list(ALLOWED_CASES),
        "gate": "externally_reviewed_manual_exact_bool_delete_or_create",
        "gate_flattening": "per_phase_row_major_yx",
        "gate_runtime_input": "validated_native_execution_capability_only",
        "gate_instruction_bindings": [
            "source_sha256",
            "anchor_sha256",
            "action_caption_sha256",
            "structured_action_program_sha256",
        ],
        "annotation_authority": "compiled_per_case_root_plus_merkle_leaf_inclusion",
        "compiled_annotation_roots_present": bool(
            COMPILED_ANNOTATION_AUTHORITY_ROOTS
        ),
        "compiled_native_binding_receipts_present": bool(
            COMPILED_NATIVE_BINDING_RECEIPT_SHA256
        ),
        "compiled_flowedit_binding_receipts_present": bool(
            COMPILED_FLOWEDIT_BINDING_RECEIPT_SHA256
        ),
        "caller_supplied_hash_can_authorize_execution": False,
        "gate_snapshot_rehashed_each_step_and_finalize": True,
        "native_source_reference_action_rehashed_each_step_and_finalize": True,
        "independent_review_forbidden_input_evidence": [
            "failed_active_used_to_author_mask=false",
            "anchor_difference_used_to_author_mask=false",
            "predicted_soft_gate_used_to_author_mask=false",
        ],
        "realized_variant_tensor_digest_required": True,
        "spatial_control": "same_mass_cyclic_permutation",
        "soft_gate_allowed": False,
        "phase_zero_regeneration_allowed": False,
        "failed_active_can_author_gate": False,
        "anchor_difference_can_author_gate": False,
        "training_target_authorized": False,
        "action_representation_claimed": False,
        "flowedit_step0_seam": "private_tested_core_not_public_execution_surface",
        "flowedit_constructor_and_noise_receipt_required": True,
        "flowedit_arbitrary_callable_or_noise_allowed": False,
        "native_local_expert": (
            "scheduled_homotopy_R2V4_inside_G_official_V2V_outside_G"
        ),
        "native_local_expert_uses_five_forward_runtime": True,
        "native_binding_includes": [
            "source_latent",
            "four_source_reference_latents_and_rgb_indices",
            "r2v_action_prompt",
            "manual_gate_manifest_and_external_root",
        ],
        "native_and_flowedit_outer_samplers_proven_connected": False,
        "G_zero": "direct_original_official_object",
        "G_zero_early_return_before_high_validation_or_homotopy": True,
        "all40_raw_high_default": False,
        "selection_authority": None,
        "base_and_regen_must_be_reviewed_side_by_side": True,
        "automatic_replacement_of_successful_base_authorized": False,
        "e03_base_success_requires_abstain_or_non_regression": True,
        "gpu_launch_authorized": False,
        "optimizer_authorized": False,
        "real_native_runner_status": (
            "blocked_compiled_roots_and_execution_receipts_are_intentionally_empty"
        ),
        "future_activation_blockers": list(FUTURE_ACTIVATION_BLOCKERS),
    }


__all__ = [
    "ALLOWED_CASES",
    "ALLOWED_GATE_VARIANTS",
    "ANNOTATION_LEAF_SCHEMA_VERSION",
    "ANNOTATION_TREE_SHAPE",
    "FLOWEDIT_RECEIPT_SCHEMA_VERSION",
    "GATE_SCHEMA_VERSION",
    "LOCAL_NATIVE_SCHEMA_VERSION",
    "NATIVE_BINDING_RECEIPT_SCHEMA_VERSION",
    "LocalOracleNativeBranchRuntimePatchV1",
    "OracleRegenerationCanaryError",
    "RECEIPT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "ValidatedOracleGateManifestV1",
    "ValidatedFlowEditExecutionV1",
    "ValidatedNativeExecutionBindingV1",
    "annotation_authority_leaf_sha256_v1",
    "canonical_json_bytes_v1",
    "contract_v1",
    "derive_regeneration_seed_v1",
    "draw_independent_keyed_gaussian_like_v1",
    "file_sha256_v1",
    "realized_gate_sha256_v1",
    "revalidate_oracle_gate_manifest_v1",
    "validate_flowedit_execution_receipt_v1",
    "validate_native_execution_binding_receipt_v1",
    "validate_oracle_gate_manifest_v1",
    "strict_json_load_path_v1",
    "tensor_content_sha256_v1",
]
