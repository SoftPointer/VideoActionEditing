"""Pure-tensor identity-anchored action-residual (IAR) objective.

IAR composes two frozen Bernini priors on one exact noisy query:

* a T2V action velocity minus a soft minimum over matched hard-negative
  velocities; and
* a source-conditioned no-op identity velocity, with matched no-op
  wrong-source velocities defining identity-sensitive nuisance directions.

The action residual is orthogonally projected away from a normalized
wrong-source tangent span with an FP32 Gram pseudoinverse, then bounded by a
per-sigma, common-offset-invariant scale.  A student is trained through three
FP32 losses: its action-minus-stop-gradient-no-op residual follows the detached bounded
teacher, its no-op field follows a distinct detached no-op correct-source
field, and both source-carrier views are anchored to that no-op field while an
asymmetric stop-gradient consistency term couples them.

This file performs no model forward, sampling, optimizer step, distributed
collective, masking, tracking, optical-flow computation, or CLI work.  It
never receives a paired target video, RGB target, or clean target latent.  The
caller remains responsible for producing each velocity with the claimed
Bernini mode.  ``SharedStateBinding`` verifies that the caller handed every
branch the same Python tensor object; this deliberately does not claim to
observe or authenticate the upstream model forwards.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from numbers import Real
import re
from typing import Any


METHOD_NAME = "bernini-identity-anchored-action-residual-v2"
RECEIPT_SCHEMA = "bernini-identity-anchored-action-residual-receipt-v2"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class IdentityAnchoredActionResidualError(RuntimeError):
    """Raised before an invalid IAR tensor cell can produce a loss."""


@dataclass(frozen=True)
class IARConfig:
    """Numerical controls for the pure-tensor IAR cell."""

    hard_negative_temperature: float = 0.25
    projection_rank_rtol: float = 1.0e-5
    postprojection_cosine_tolerance: float = 2.0e-4
    action_rms_cap_ratio: float = 0.50
    high_sigma_min: float = 0.55
    mid_sigma_min: float = 0.25
    mid_sigma_action_scale: float = 0.50
    action_loss_weight: float = 1.0
    identity_loss_weight: float = 1.0
    view_loss_weight: float = 0.25
    view_consistency_weight: float = 0.25

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise IdentityAnchoredActionResidualError(
                    f"{name} must be a finite real number"
                )
        for name in (
            "hard_negative_temperature",
            "action_rms_cap_ratio",
            "action_loss_weight",
            "identity_loss_weight",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise IdentityAnchoredActionResidualError(
                    f"{name} must be strictly positive"
                )
        if self.view_loss_weight < 0.0:
            raise IdentityAnchoredActionResidualError(
                "view_loss_weight must be nonnegative"
            )
        if self.view_consistency_weight < 0.0:
            raise IdentityAnchoredActionResidualError(
                "view_consistency_weight must be nonnegative"
            )
        if not 0.0 < self.projection_rank_rtol < 1.0:
            raise IdentityAnchoredActionResidualError(
                "projection_rank_rtol must lie in (0,1)"
            )
        if not 0.0 < self.postprojection_cosine_tolerance <= 1.0e-3:
            raise IdentityAnchoredActionResidualError(
                "postprojection_cosine_tolerance must lie in (0,1e-3]"
            )
        if not 0.0 < self.mid_sigma_min < self.high_sigma_min < 1.0:
            raise IdentityAnchoredActionResidualError(
                "sigma gates must satisfy 0 < mid_sigma_min < "
                "high_sigma_min < 1"
            )
        if not 0.0 < self.mid_sigma_action_scale <= 1.0:
            raise IdentityAnchoredActionResidualError(
                "mid_sigma_action_scale must lie in (0,1]"
            )


@dataclass(frozen=True)
class SharedStateBinding:
    """Object-alias witness for one noisy query used by every branch.

    ``branch_state_refs`` intentionally stores references rather than hashes.
    The core rejects a value-equal clone: every entry must be the exact same
    object as ``noised_state``.  This catches accidental cross-state vector
    arithmetic at the tensor API boundary, while making no claim about model
    callbacks that happened before this object was created.
    """

    noised_state: Any
    branch_names: tuple[str, ...]
    branch_state_refs: tuple[Any, ...]


@dataclass(frozen=True)
class BranchSemantic:
    """Caller-supplied immutable mode/text/source digest for one branch.

    Digests let the core verify equality and distinctness relationships.  They
    do not prove what upstream text, source bytes, or model call produced a
    tensor; receipts explicitly preserve that boundary.
    """

    branch: str
    mode: str
    text_sha256: str
    source_sha256: str | None


@dataclass(frozen=True)
class BranchSemanticBinding:
    """Ordered semantic claims corresponding one-to-one with branch names."""

    branches: tuple[BranchSemantic, ...]


@dataclass(frozen=True)
class IARFrozenFields:
    """Detached frozen branches required by the training-free IAR teacher.

    Only the matched no-op correct/wrong fields define the nuisance span.
    Matched action-conditioned correct/wrong fields are deliberately separate
    and are used only for source-invariance and T2V-alignment diagnostics.
    """

    shared_state: SharedStateBinding
    semantic_binding: BranchSemanticBinding
    sigma: Any
    frozen_t2v_action: Any
    frozen_t2v_hard_negatives: Any
    hard_negative_energies: Any
    frozen_identity_noop_correct: Any
    frozen_identity_noop_wrong_sources: Any
    frozen_identity_action_correct: Any
    frozen_identity_action_wrong_sources: Any


@dataclass(frozen=True)
class IARFields:
    """Frozen teacher fields and graph-carrying student fields.

    Base velocity tensors have shape ``[B,...]``.  Hard-negative and
    no-op/action wrong-source tensors have ``[B,K,...]`` and ``[B,M,...]``
    layouts, with ``K >= 2`` and ``M >= 1``.  Every frozen
    tensor is required to be detached.  All four student tensors must retain
    autograd.  The semantic binding verifies digest relationships, never the
    truth or upstream provenance of the caller's digest claims.
    """

    shared_state: SharedStateBinding
    semantic_binding: BranchSemanticBinding
    sigma: Any
    frozen_t2v_action: Any
    frozen_t2v_hard_negatives: Any
    hard_negative_energies: Any
    frozen_identity_noop_correct: Any
    frozen_identity_noop_wrong_sources: Any
    frozen_identity_action_correct: Any
    frozen_identity_action_wrong_sources: Any
    student_action: Any
    student_noop: Any
    student_identity_view_a: Any
    student_identity_view_b: Any


@dataclass(frozen=True)
class IARDiagnostics:
    softmin_weights: Any
    hard_negative_barycenter: Any
    raw_action_residual: Any
    identity_tangents: Any
    normalized_identity_tangents: Any
    tangent_norms: Any
    source_tangent_rms: Any
    robust_source_tangent_rms: Any
    source_conditioned_action_residual_correct: Any
    source_conditioned_action_residual_wrong_sources: Any
    source_conditioned_action_correct_rms: Any
    source_conditioned_action_wrong_rms: Any
    source_action_invariance_cosine: Any
    source_action_invariance_symmetric_norm_ratio: Any
    projected_action_alignment_correct: Any
    projected_action_alignment_wrong_sources: Any
    normalized_tangent_gram_fp32: Any
    normalized_tangent_gram_pinv_fp32: Any
    tangent_rank: Any
    projection_coefficients: Any
    projected_action_residual: Any
    capped_action_residual: Any
    projection_retention: Any
    raw_action_rms: Any
    projected_action_rms: Any
    capped_action_rms: Any
    cap_reference_rms: Any
    action_rms_cap: Any
    cap_scale: Any
    sigma_action_scale: Any
    max_abs_postprojection_tangent_cosine: Any
    student_action_residual: Any
    action_per_sample: Any
    identity_per_sample: Any
    view_anchor_a_per_sample: Any
    view_anchor_b_per_sample: Any
    view_consistency_per_sample: Any
    view_per_sample: Any


@dataclass(frozen=True)
class IARFrozenDiagnostics:
    softmin_weights: Any
    hard_negative_barycenter: Any
    raw_action_residual: Any
    identity_tangents: Any
    normalized_identity_tangents: Any
    tangent_norms: Any
    source_tangent_rms: Any
    robust_source_tangent_rms: Any
    source_conditioned_action_residual_correct: Any
    source_conditioned_action_residual_wrong_sources: Any
    source_conditioned_action_correct_rms: Any
    source_conditioned_action_wrong_rms: Any
    source_action_invariance_cosine: Any
    source_action_invariance_symmetric_norm_ratio: Any
    projected_action_alignment_correct: Any
    projected_action_alignment_wrong_sources: Any
    normalized_tangent_gram_fp32: Any
    normalized_tangent_gram_pinv_fp32: Any
    tangent_rank: Any
    projection_coefficients: Any
    projected_action_residual: Any
    capped_action_residual: Any
    projection_retention: Any
    raw_action_rms: Any
    projected_action_rms: Any
    capped_action_rms: Any
    cap_reference_rms: Any
    action_rms_cap: Any
    cap_scale: Any
    sigma_action_scale: Any
    max_abs_postprojection_tangent_cosine: Any


@dataclass(frozen=True)
class IARFrozenTeacherResult:
    teacher_action_residual: Any
    diagnostics: IARFrozenDiagnostics
    receipt: dict[str, Any]


@dataclass(frozen=True)
class IARLossResult:
    total: Any
    action: Any
    identity: Any
    view: Any
    teacher_action_residual: Any
    diagnostics: IARDiagnostics
    receipt: dict[str, Any]


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependent
        raise IdentityAnchoredActionResidualError(
            "IAR tensor operations require PyTorch"
        ) from error
    return torch


def _require_tensor(value: Any, *, label: str) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise IdentityAnchoredActionResidualError(
            f"{label} must be a torch.Tensor"
        )
    if value.device.type == "meta":
        raise IdentityAnchoredActionResidualError(
            f"{label} cannot be a meta tensor"
        )
    return value


def _require_finite_float(value: Any, *, label: str) -> Any:
    torch = _torch()
    tensor = _require_tensor(value, label=label)
    if not tensor.is_floating_point():
        raise IdentityAnchoredActionResidualError(
            f"{label} must be floating point"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise IdentityAnchoredActionResidualError(
            f"{label} contains NaN or infinity"
        )
    return tensor


def _require_detached_float(value: Any, *, label: str) -> Any:
    tensor = _require_finite_float(value, label=label)
    if tensor.requires_grad or tensor.grad_fn is not None:
        raise IdentityAnchoredActionResidualError(
            f"{label} must be detached from every trainable graph"
        )
    return tensor


def _require_graph_float(value: Any, *, label: str) -> Any:
    tensor = _require_finite_float(value, label=label)
    if not tensor.requires_grad and tensor.grad_fn is None:
        raise IdentityAnchoredActionResidualError(
            f"{label} must retain a student graph"
        )
    return tensor


def expected_branch_names(
    hard_negative_count: int,
    wrong_source_count: int,
) -> tuple[str, ...]:
    """Return the exact branch order covered by a shared-state witness."""

    for name, value, minimum in (
        ("hard_negative_count", hard_negative_count, 2),
        ("wrong_source_count", wrong_source_count, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise IdentityAnchoredActionResidualError(
                f"{name} must be an integer >= {minimum}"
            )
    return expected_frozen_branch_names(
        hard_negative_count, wrong_source_count
    ) + (
        "student_action",
        "student_noop",
        "student_identity_view_a",
        "student_identity_view_b",
    )


def expected_frozen_branch_names(
    hard_negative_count: int,
    wrong_source_count: int,
) -> tuple[str, ...]:
    """Return the exact frozen branch order for a training-free teacher."""

    for name, value, minimum in (
        ("hard_negative_count", hard_negative_count, 2),
        ("wrong_source_count", wrong_source_count, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise IdentityAnchoredActionResidualError(
                f"{name} must be an integer >= {minimum}"
            )
    return (
        "frozen_t2v_action",
        *(f"frozen_t2v_hard_negative[{index}]" for index in range(hard_negative_count)),
        "frozen_identity_noop_correct",
        *(
            f"frozen_identity_noop_wrong_source[{index}]"
            for index in range(wrong_source_count)
        ),
        "frozen_identity_action_correct",
        *(
            f"frozen_identity_action_wrong_source[{index}]"
            for index in range(wrong_source_count)
        ),
    )


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise IdentityAnchoredActionResidualError(
            f"{label} must be one lowercase SHA-256 digest"
        )
    return value


def bind_branch_semantics(
    branches: tuple[BranchSemantic, ...],
) -> BranchSemanticBinding:
    """Bind basic immutable branch claims; relational checks happen per cell."""

    if not isinstance(branches, tuple) or not branches:
        raise IdentityAnchoredActionResidualError(
            "semantic branches must be one nonempty tuple"
        )
    seen: set[str] = set()
    for index, branch in enumerate(branches):
        if not isinstance(branch, BranchSemantic):
            raise IdentityAnchoredActionResidualError(
                f"semantic branches[{index}] must be a BranchSemantic"
            )
        if (
            not isinstance(branch.branch, str)
            or not branch.branch
            or branch.branch != branch.branch.strip()
            or branch.branch in seen
        ):
            raise IdentityAnchoredActionResidualError(
                "semantic branch names must be unique nonempty trimmed text"
            )
        if branch.mode not in ("t2v", "mv2v"):
            raise IdentityAnchoredActionResidualError(
                f"semantic branch {branch.branch} mode must be t2v or mv2v"
            )
        _require_sha256(
            branch.text_sha256,
            label=f"semantic branch {branch.branch}.text_sha256",
        )
        if branch.source_sha256 is not None:
            _require_sha256(
                branch.source_sha256,
                label=f"semantic branch {branch.branch}.source_sha256",
            )
        seen.add(branch.branch)
    return BranchSemanticBinding(branches=branches)


def _validate_semantic_binding(
    binding: Any,
    *,
    hard_negative_count: int,
    wrong_source_count: int,
    include_students: bool,
) -> BranchSemanticBinding:
    if not isinstance(binding, BranchSemanticBinding):
        raise IdentityAnchoredActionResidualError(
            "semantic_binding must be a BranchSemanticBinding"
        )
    checked = bind_branch_semantics(binding.branches)
    expected = expected_frozen_branch_names(
        hard_negative_count, wrong_source_count
    )
    if include_students:
        expected = expected_branch_names(
            hard_negative_count, wrong_source_count
        )
    if tuple(item.branch for item in checked.branches) != expected:
        raise IdentityAnchoredActionResidualError(
            "semantic branch names/order differ from the IAR branch contract"
        )

    by_name = {item.branch: item for item in checked.branches}
    action = by_name["frozen_t2v_action"]
    negatives = tuple(
        by_name[f"frozen_t2v_hard_negative[{index}]"]
        for index in range(hard_negative_count)
    )
    noop_correct = by_name["frozen_identity_noop_correct"]
    noop_wrong = tuple(
        by_name[f"frozen_identity_noop_wrong_source[{index}]"]
        for index in range(wrong_source_count)
    )
    action_correct = by_name["frozen_identity_action_correct"]
    action_wrong = tuple(
        by_name[f"frozen_identity_action_wrong_source[{index}]"]
        for index in range(wrong_source_count)
    )
    if action.mode != "t2v" or action.source_sha256 is not None:
        raise IdentityAnchoredActionResidualError(
            "frozen T2V action semantic must have mode=t2v and no source digest"
        )
    if any(item.mode != "t2v" or item.source_sha256 is not None for item in negatives):
        raise IdentityAnchoredActionResidualError(
            "frozen T2V negative semantics must have mode=t2v and no source digest"
        )
    negative_texts = tuple(item.text_sha256 for item in negatives)
    if (
        action.text_sha256 in negative_texts
        or len(set(negative_texts)) != len(negative_texts)
    ):
        raise IdentityAnchoredActionResidualError(
            "action and hard-negative text digests must be pairwise distinct"
        )
    if noop_correct.text_sha256 not in negative_texts:
        raise IdentityAnchoredActionResidualError(
            "noop text digest must occur in the matched hard-negative set"
        )
    if noop_correct.text_sha256 == action.text_sha256:
        raise IdentityAnchoredActionResidualError(
            "noop and action text digests must differ"
        )
    mv2v = (noop_correct, *noop_wrong, action_correct, *action_wrong)
    if any(item.mode != "mv2v" or item.source_sha256 is None for item in mv2v):
        raise IdentityAnchoredActionResidualError(
            "identity semantics must have mode=mv2v and one source digest"
        )
    if any(
        item.text_sha256 != noop_correct.text_sha256
        for item in noop_wrong
    ):
        raise IdentityAnchoredActionResidualError(
            "all noop source-swap branches must share noop text"
        )
    if action_correct.text_sha256 != action.text_sha256 or any(
        item.text_sha256 != action.text_sha256 for item in action_wrong
    ):
        raise IdentityAnchoredActionResidualError(
            "all action-conditioned source-swap branches must share action text"
        )
    correct_source = noop_correct.source_sha256
    if action_correct.source_sha256 != correct_source:
        raise IdentityAnchoredActionResidualError(
            "noop and action correct branches must share the correct source digest"
        )
    wrong_sources = tuple(item.source_sha256 for item in noop_wrong)
    if (
        correct_source in wrong_sources
        or len(set(wrong_sources)) != len(wrong_sources)
    ):
        raise IdentityAnchoredActionResidualError(
            "wrong-source digests must be unique and differ from the correct source"
        )
    if tuple(item.source_sha256 for item in action_wrong) != wrong_sources:
        raise IdentityAnchoredActionResidualError(
            "noop/action wrong-source branches must be source-paired in order"
        )

    if include_students:
        student_action = by_name["student_action"]
        student_noop = by_name["student_noop"]
        views = (
            by_name["student_identity_view_a"],
            by_name["student_identity_view_b"],
        )
        if (
            student_action.mode != "mv2v"
            or student_action.text_sha256 != action.text_sha256
            or student_action.source_sha256 != correct_source
        ):
            raise IdentityAnchoredActionResidualError(
                "student_action must bind the action text and correct source"
            )
        if any(
            item.mode != "mv2v"
            or item.text_sha256 != noop_correct.text_sha256
            or item.source_sha256 != correct_source
            for item in (student_noop, *views)
        ):
            raise IdentityAnchoredActionResidualError(
                "student noop/views must bind the noop text and correct source"
            )
    return checked


def bind_shared_state(
    noised_state: Any,
    branch_states: Mapping[str, Any],
) -> SharedStateBinding:
    """Create an immutable witness after exact object-alias validation."""

    state = _require_detached_float(noised_state, label="noised_state")
    if state.ndim < 2 or int(state.shape[0]) <= 0 or state.numel() == 0:
        raise IdentityAnchoredActionResidualError(
            "noised_state must have nonempty batch-first [B,...] layout"
        )
    if not isinstance(branch_states, Mapping) or not branch_states:
        raise IdentityAnchoredActionResidualError(
            "branch_states must be a nonempty name-to-state mapping"
        )
    names: list[str] = []
    refs: list[Any] = []
    for name, branch_state in branch_states.items():
        if not isinstance(name, str) or not name:
            raise IdentityAnchoredActionResidualError(
                "shared-state branch names must be nonempty strings"
            )
        if branch_state is not state:
            raise IdentityAnchoredActionResidualError(
                f"{name} did not receive the exact shared noisy-state object"
            )
        names.append(name)
        refs.append(branch_state)
    return SharedStateBinding(
        noised_state=state,
        branch_names=tuple(names),
        branch_state_refs=tuple(refs),
    )


def _same_representation(
    reference: Any,
    tensor: Any,
    *,
    label: str,
) -> None:
    if (
        tensor.dtype != reference.dtype
        or tensor.device != reference.device
        or tensor.layout != reference.layout
    ):
        raise IdentityAnchoredActionResidualError(
            f"{label} dtype, device, or layout differs from its reference field"
        )


def _validate_frozen_fields(
    fields: IARFrozenFields,
) -> tuple[int, int, int, tuple[int, ...]]:
    torch = _torch()
    if not isinstance(fields, IARFrozenFields):
        raise IdentityAnchoredActionResidualError(
            "fields must be an IARFrozenFields instance"
        )
    frozen_action = _require_detached_float(
        fields.frozen_t2v_action, label="frozen_t2v_action"
    )
    if (
        frozen_action.ndim < 2
        or int(frozen_action.shape[0]) <= 0
        or frozen_action.numel() == 0
        or frozen_action.layout != torch.strided
    ):
        raise IdentityAnchoredActionResidualError(
            "frozen_t2v_action must have nonempty strided batch-first [B,...] layout"
        )
    batch = int(frozen_action.shape[0])
    feature_shape = tuple(int(value) for value in frozen_action.shape[1:])

    detached_identity_names = (
        "frozen_identity_noop_correct",
        "frozen_identity_action_correct",
    )
    for name in detached_identity_names:
        tensor = _require_detached_float(getattr(fields, name), label=name)
        _same_representation(frozen_action, tensor, label=name)
        if tuple(tensor.shape) != tuple(frozen_action.shape):
            raise IdentityAnchoredActionResidualError(
                f"{name} shape differs from frozen_t2v_action"
            )

    hard_negatives = _require_detached_float(
        fields.frozen_t2v_hard_negatives,
        label="frozen_t2v_hard_negatives",
    )
    _same_representation(
        frozen_action,
        hard_negatives,
        label="frozen_t2v_hard_negatives",
    )
    if (
        hard_negatives.ndim != frozen_action.ndim + 1
        or int(hard_negatives.shape[0]) != batch
        or tuple(hard_negatives.shape[2:]) != feature_shape
    ):
        raise IdentityAnchoredActionResidualError(
            "frozen_t2v_hard_negatives must be [B,K,...] matching frozen_t2v_action"
        )
    hard_negative_count = int(hard_negatives.shape[1])
    if hard_negative_count < 2:
        raise IdentityAnchoredActionResidualError(
            "at least two matched hard-negative velocities are required"
        )

    wrong_source_counts: list[int] = []
    for name in (
        "frozen_identity_noop_wrong_sources",
        "frozen_identity_action_wrong_sources",
    ):
        wrong_sources = _require_detached_float(
            getattr(fields, name), label=name
        )
        _same_representation(frozen_action, wrong_sources, label=name)
        if (
            wrong_sources.ndim != frozen_action.ndim + 1
            or int(wrong_sources.shape[0]) != batch
            or tuple(wrong_sources.shape[2:]) != feature_shape
        ):
            raise IdentityAnchoredActionResidualError(
                f"{name} must be [B,M,...] matching frozen_t2v_action"
            )
        wrong_source_counts.append(int(wrong_sources.shape[1]))
    wrong_source_count = wrong_source_counts[0]
    if wrong_source_count < 1:
        raise IdentityAnchoredActionResidualError(
            "at least one matched wrong-source velocity is required"
        )
    if wrong_source_counts[1] != wrong_source_count:
        raise IdentityAnchoredActionResidualError(
            "noop/action wrong-source counts must match exactly"
        )
    _validate_semantic_binding(
        fields.semantic_binding,
        hard_negative_count=hard_negative_count,
        wrong_source_count=wrong_source_count,
        include_students=False,
    )

    energies = _require_detached_float(
        fields.hard_negative_energies,
        label="hard_negative_energies",
    )
    if (
        energies.dtype != torch.float32
        or energies.device != frozen_action.device
        or energies.layout != torch.strided
        or tuple(energies.shape) != (batch, hard_negative_count)
    ):
        raise IdentityAnchoredActionResidualError(
            "hard_negative_energies must be detached FP32 [B,K] on the velocity device"
        )

    sigma = _require_detached_float(fields.sigma, label="sigma")
    if (
        sigma.dtype != torch.float32
        or sigma.device != frozen_action.device
        or sigma.layout != torch.strided
        or tuple(sigma.shape) != (batch,)
    ):
        raise IdentityAnchoredActionResidualError(
            "sigma must be detached FP32 [B] on the velocity device"
        )
    if bool(((sigma < 0.0) | (sigma > 1.0)).any().item()):
        raise IdentityAnchoredActionResidualError(
            "sigma values must lie in the closed interval [0,1]"
        )

    if not isinstance(fields.shared_state, SharedStateBinding):
        raise IdentityAnchoredActionResidualError(
            "shared_state must be a SharedStateBinding"
        )
    state = _require_detached_float(
        fields.shared_state.noised_state,
        label="shared_state.noised_state",
    )
    if int(state.shape[0]) != batch or state.device != frozen_action.device:
        raise IdentityAnchoredActionResidualError(
            "shared noisy state batch/device differs from frozen velocity fields"
        )
    expected = expected_frozen_branch_names(
        hard_negative_count, wrong_source_count
    )
    if fields.shared_state.branch_names != expected:
        raise IdentityAnchoredActionResidualError(
            "shared-state branch names/order differ from the frozen IAR contract"
        )
    if len(fields.shared_state.branch_state_refs) != len(expected):
        raise IdentityAnchoredActionResidualError(
            "shared-state witness branch count differs"
        )
    if any(ref is not state for ref in fields.shared_state.branch_state_refs):
        raise IdentityAnchoredActionResidualError(
            "a branch no longer aliases the exact shared noisy-state object"
        )
    return batch, hard_negative_count, wrong_source_count, feature_shape


def _validate_fields(fields: IARFields) -> tuple[int, int, int, tuple[int, ...]]:
    torch = _torch()
    if not isinstance(fields, IARFields):
        raise IdentityAnchoredActionResidualError(
            "fields must be an IARFields instance"
        )

    student_action = _require_graph_float(
        fields.student_action, label="student_action"
    )
    if (
        student_action.ndim < 2
        or int(student_action.shape[0]) <= 0
        or student_action.numel() == 0
        or student_action.layout != torch.strided
    ):
        raise IdentityAnchoredActionResidualError(
            "student_action must have nonempty strided batch-first [B,...] layout"
        )
    batch = int(student_action.shape[0])
    feature_shape = tuple(int(value) for value in student_action.shape[1:])

    graph_names = (
        "student_noop",
        "student_identity_view_a",
        "student_identity_view_b",
    )
    for name in graph_names:
        tensor = _require_graph_float(getattr(fields, name), label=name)
        _same_representation(student_action, tensor, label=name)
        if tuple(tensor.shape) != tuple(student_action.shape):
            raise IdentityAnchoredActionResidualError(
                f"{name} shape differs from student_action"
            )

    detached_base_names = (
        "frozen_t2v_action",
        "frozen_identity_noop_correct",
        "frozen_identity_action_correct",
    )
    for name in detached_base_names:
        tensor = _require_detached_float(getattr(fields, name), label=name)
        _same_representation(student_action, tensor, label=name)
        if tuple(tensor.shape) != tuple(student_action.shape):
            raise IdentityAnchoredActionResidualError(
                f"{name} shape differs from student_action"
            )

    hard_negatives = _require_detached_float(
        fields.frozen_t2v_hard_negatives,
        label="frozen_t2v_hard_negatives",
    )
    _same_representation(
        student_action,
        hard_negatives,
        label="frozen_t2v_hard_negatives",
    )
    if (
        hard_negatives.ndim != student_action.ndim + 1
        or int(hard_negatives.shape[0]) != batch
        or tuple(hard_negatives.shape[2:]) != feature_shape
    ):
        raise IdentityAnchoredActionResidualError(
            "frozen_t2v_hard_negatives must be [B,K,...] matching student_action"
        )
    hard_negative_count = int(hard_negatives.shape[1])
    if hard_negative_count < 2:
        raise IdentityAnchoredActionResidualError(
            "at least two matched hard-negative velocities are required"
        )

    wrong_source_counts = []
    for name in (
        "frozen_identity_noop_wrong_sources",
        "frozen_identity_action_wrong_sources",
    ):
        wrong_sources = _require_detached_float(
            getattr(fields, name), label=name
        )
        _same_representation(student_action, wrong_sources, label=name)
        if (
            wrong_sources.ndim != student_action.ndim + 1
            or int(wrong_sources.shape[0]) != batch
            or tuple(wrong_sources.shape[2:]) != feature_shape
        ):
            raise IdentityAnchoredActionResidualError(
                f"{name} must be [B,M,...] matching student_action"
            )
        wrong_source_counts.append(int(wrong_sources.shape[1]))
    wrong_source_count = wrong_source_counts[0]
    if wrong_source_count < 1:
        raise IdentityAnchoredActionResidualError(
            "at least one matched wrong-source velocity is required"
        )
    if wrong_source_counts[1] != wrong_source_count:
        raise IdentityAnchoredActionResidualError(
            "noop/action wrong-source counts must match exactly"
        )
    _validate_semantic_binding(
        fields.semantic_binding,
        hard_negative_count=hard_negative_count,
        wrong_source_count=wrong_source_count,
        include_students=True,
    )

    energies = _require_detached_float(
        fields.hard_negative_energies,
        label="hard_negative_energies",
    )
    if (
        energies.dtype != torch.float32
        or energies.device != student_action.device
        or energies.layout != torch.strided
        or tuple(energies.shape) != (batch, hard_negative_count)
    ):
        raise IdentityAnchoredActionResidualError(
            "hard_negative_energies must be detached FP32 [B,K] on the velocity device"
        )

    sigma = _require_detached_float(fields.sigma, label="sigma")
    if (
        sigma.dtype != torch.float32
        or sigma.device != student_action.device
        or sigma.layout != torch.strided
        or tuple(sigma.shape) != (batch,)
    ):
        raise IdentityAnchoredActionResidualError(
            "sigma must be detached FP32 [B] on the velocity device"
        )
    if bool(((sigma < 0.0) | (sigma > 1.0)).any().item()):
        raise IdentityAnchoredActionResidualError(
            "sigma values must lie in the closed interval [0,1]"
        )

    if not isinstance(fields.shared_state, SharedStateBinding):
        raise IdentityAnchoredActionResidualError(
            "shared_state must be a SharedStateBinding"
        )
    state = _require_detached_float(
        fields.shared_state.noised_state,
        label="shared_state.noised_state",
    )
    if int(state.shape[0]) != batch or state.device != student_action.device:
        raise IdentityAnchoredActionResidualError(
            "shared noisy state batch/device differs from velocity fields"
        )
    expected = expected_branch_names(hard_negative_count, wrong_source_count)
    if fields.shared_state.branch_names != expected:
        raise IdentityAnchoredActionResidualError(
            "shared-state branch names/order differ from the IAR branch contract"
        )
    if len(fields.shared_state.branch_state_refs) != len(expected):
        raise IdentityAnchoredActionResidualError(
            "shared-state witness branch count differs"
        )
    if any(ref is not state for ref in fields.shared_state.branch_state_refs):
        raise IdentityAnchoredActionResidualError(
            "a branch no longer aliases the exact shared noisy-state object"
        )
    return batch, hard_negative_count, wrong_source_count, feature_shape


def sigma_action_scale(sigma: Any, config: IARConfig = IARConfig()) -> Any:
    """Return the high/mid/low action schedule for each physical sigma."""

    torch = _torch()
    config.validate()
    value = _require_detached_float(sigma, label="sigma")
    if value.dtype != torch.float32 or value.ndim != 1:
        raise IdentityAnchoredActionResidualError(
            "sigma_action_scale requires detached FP32 [B] sigma"
        )
    if bool(((value < 0.0) | (value > 1.0)).any().item()):
        raise IdentityAnchoredActionResidualError(
            "sigma values must lie in the closed interval [0,1]"
        )
    return torch.where(
        value >= float(config.high_sigma_min),
        torch.ones_like(value),
        torch.where(
            value >= float(config.mid_sigma_min),
            torch.full_like(value, float(config.mid_sigma_action_scale)),
            torch.zeros_like(value),
        ),
    )


def _batch_rms(value: Any) -> Any:
    return value.square().mean(dim=tuple(range(1, value.ndim))).sqrt()


def _batch_mse(left: Any, right: Any) -> Any:
    return (left - right).square().mean(dim=tuple(range(1, left.ndim)))


def _view_batch(value: Any, reference: Any) -> Any:
    return value.reshape(int(value.shape[0]), *([1] * (reference.ndim - 1)))


def _float_list(value: Any) -> list[float]:
    return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _int_list(value: Any) -> list[int]:
    return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _compute_frozen_teacher_validated(
    fields: IARFrozenFields,
    config: IARConfig,
    *,
    batch: int,
    hard_count: int,
    wrong_count: int,
    feature_shape: tuple[int, ...],
) -> IARFrozenTeacherResult:
    torch = _torch()
    frozen_action = fields.frozen_t2v_action.detach().to(dtype=torch.float32)
    frozen_negatives = fields.frozen_t2v_hard_negatives.detach().to(
        dtype=torch.float32
    )
    negative_energies = fields.hard_negative_energies.detach()
    identity_noop_correct = fields.frozen_identity_noop_correct.detach().to(
        dtype=torch.float32
    )
    identity_noop_wrong = fields.frozen_identity_noop_wrong_sources.detach().to(
        dtype=torch.float32
    )
    identity_action_correct = fields.frozen_identity_action_correct.detach().to(
        dtype=torch.float32
    )
    identity_action_wrong = (
        fields.frozen_identity_action_wrong_sources.detach().to(
            dtype=torch.float32
        )
    )

    softmin_weights = torch.softmax(
        -negative_energies / float(config.hard_negative_temperature), dim=1
    )
    negative_barycenter = (
        frozen_negatives
        * softmin_weights.reshape(
            batch, hard_count, *([1] * len(feature_shape))
        )
    ).sum(dim=1)
    raw_action = frozen_action - negative_barycenter

    # HAT nuisance tangents are strictly matched no-op source swaps.  The
    # action-conditioned fields below are diagnostic only and never enter Q.
    identity_tangents = (
        identity_noop_correct.unsqueeze(1) - identity_noop_wrong
    )
    source_action_residual_correct = (
        identity_action_correct - identity_noop_correct
    )
    source_action_residual_wrong = (
        identity_action_wrong - identity_noop_wrong
    )
    tangent_flat = identity_tangents.reshape(batch, wrong_count, -1)
    raw_flat = raw_action.reshape(batch, -1, 1)
    tiny = torch.finfo(torch.float32).tiny
    tangent_max_abs = tangent_flat.abs().amax(dim=2)
    nonzero_tangent = tangent_max_abs > 0.0
    scaled_tangents = torch.where(
        nonzero_tangent[:, :, None],
        tangent_flat / tangent_max_abs.clamp_min(tiny)[:, :, None],
        torch.zeros_like(tangent_flat),
    )
    scaled_l2 = torch.linalg.vector_norm(scaled_tangents, dim=2)
    normalized_tangents = torch.where(
        nonzero_tangent[:, :, None],
        scaled_tangents / scaled_l2.clamp_min(tiny)[:, :, None],
        torch.zeros_like(scaled_tangents),
    )
    feature_count = int(math.prod(feature_shape))
    source_tangent_rms = tangent_max_abs * scaled_tangents.square().mean(
        dim=2
    ).sqrt()
    tangent_norms = source_tangent_rms * math.sqrt(float(feature_count))
    normalized_gram = torch.bmm(
        normalized_tangents, normalized_tangents.transpose(1, 2)
    )
    try:
        normalized_gram_pinv = torch.linalg.pinv(
            normalized_gram,
            rtol=float(config.projection_rank_rtol),
            hermitian=True,
        )
    except RuntimeError as error:
        raise IdentityAnchoredActionResidualError(
            "FP32 normalized tangent Gram pseudoinverse failed"
        ) from error
    rhs = torch.bmm(normalized_tangents, raw_flat)
    coefficients = torch.bmm(normalized_gram_pinv, rhs)
    removed = torch.bmm(
        normalized_tangents.transpose(1, 2), coefficients
    )
    projected_flat = raw_flat - removed
    projected = projected_flat.reshape_as(raw_action)
    gram_eigenvalues = torch.linalg.eigvalsh(normalized_gram).clamp_min(0.0)
    largest_eigenvalue = gram_eigenvalues.amax(dim=1, keepdim=True)
    tangent_rank = (
        gram_eigenvalues
        > largest_eigenvalue * float(config.projection_rank_rtol)
    ).sum(dim=1)

    raw_l2 = torch.linalg.vector_norm(raw_flat.squeeze(-1), dim=1)
    projected_l2 = torch.linalg.vector_norm(projected_flat.squeeze(-1), dim=1)
    projection_retention = torch.where(
        raw_l2 > 0.0,
        projected_l2 / raw_l2.clamp_min(tiny),
        torch.zeros_like(raw_l2),
    )
    projected_dot = torch.bmm(
        normalized_tangents, projected_flat
    ).squeeze(-1)
    postprojection_cosine = torch.where(
        nonzero_tangent & (projected_l2[:, None] > 0.0),
        projected_dot / projected_l2[:, None].clamp_min(tiny),
        torch.zeros_like(projected_dot),
    )
    max_abs_postprojection_cosine = postprojection_cosine.abs().amax(dim=1)

    source_action_correct_flat = source_action_residual_correct.reshape(
        batch, -1
    )
    source_action_wrong_flat = source_action_residual_wrong.reshape(
        batch, wrong_count, -1
    )
    source_action_correct_l2 = torch.linalg.vector_norm(
        source_action_correct_flat, dim=1
    )
    source_action_wrong_l2 = torch.linalg.vector_norm(
        source_action_wrong_flat, dim=2
    )
    source_action_correct_rms = _batch_rms(
        source_action_residual_correct
    )
    source_action_wrong_rms = source_action_residual_wrong.square().mean(
        dim=tuple(range(2, source_action_residual_wrong.ndim))
    ).sqrt()
    source_action_dot = torch.bmm(
        source_action_wrong_flat,
        source_action_correct_flat[:, :, None],
    ).squeeze(-1)
    source_action_denominator = (
        source_action_wrong_l2 * source_action_correct_l2[:, None]
    )
    source_action_invariance_cosine = torch.where(
        source_action_denominator > 0.0,
        source_action_dot / source_action_denominator.clamp_min(tiny),
        torch.zeros_like(source_action_dot),
    )
    source_action_invariance_symmetric_norm_ratio = torch.where(
        (source_action_wrong_l2 > 0.0)
        & (source_action_correct_l2[:, None] > 0.0),
        torch.minimum(
            source_action_wrong_l2
            / source_action_correct_l2[:, None].clamp_min(tiny),
            source_action_correct_l2[:, None]
            / source_action_wrong_l2.clamp_min(tiny),
        ),
        torch.zeros_like(source_action_wrong_l2),
    )
    projected_action_alignment_correct = torch.where(
        (projected_l2 > 0.0) & (source_action_correct_l2 > 0.0),
        (source_action_correct_flat * projected_flat.squeeze(-1)).sum(dim=1)
        / (source_action_correct_l2 * projected_l2).clamp_min(tiny),
        torch.zeros_like(projected_l2),
    )
    projected_action_wrong_dot = torch.bmm(
        source_action_wrong_flat, projected_flat
    ).squeeze(-1)
    projected_action_alignment_wrong = torch.where(
        (projected_l2[:, None] > 0.0) & (source_action_wrong_l2 > 0.0),
        projected_action_wrong_dot
        / (source_action_wrong_l2 * projected_l2[:, None]).clamp_min(tiny),
        torch.zeros_like(source_action_wrong_l2),
    )
    correct_wrong_comparison_meaningful = (
        (source_action_correct_l2[:, None] > 0.0)
        & (source_action_wrong_l2 > 0.0)
    )
    projected_alignment_correct_meaningful = (
        (projected_l2 > 0.0) & (source_action_correct_l2 > 0.0)
    )
    projected_alignment_wrong_meaningful = (
        (projected_l2[:, None] > 0.0) & (source_action_wrong_l2 > 0.0)
    )

    action_scale = sigma_action_scale(fields.sigma, config)
    raw_rms = _batch_rms(raw_action)
    projected_rms = _batch_rms(projected)
    robust_source_tangent_rms = source_tangent_rms.median(dim=1).values
    cap_reference_rms = torch.minimum(
        raw_rms, robust_source_tangent_rms
    )
    action_cap = (
        float(config.action_rms_cap_ratio)
        * action_scale
        * cap_reference_rms
    )
    cap_scale = torch.minimum(
        torch.ones_like(projected_rms),
        action_cap / projected_rms.clamp_min(tiny),
    )
    cap_scale = torch.where(
        projected_rms > 0.0, cap_scale, torch.ones_like(cap_scale)
    )
    cap_scale = torch.where(
        action_scale > 0.0, cap_scale, torch.zeros_like(cap_scale)
    )
    capped_teacher = projected * _view_batch(cap_scale, projected)
    capped_teacher = torch.where(
        _view_batch(action_scale > 0.0, capped_teacher),
        capped_teacher,
        torch.zeros_like(capped_teacher),
    )
    capped_rms = _batch_rms(capped_teacher)

    detached_outputs = (
        softmin_weights,
        negative_barycenter,
        raw_action,
        identity_tangents,
        identity_noop_correct,
        identity_noop_wrong,
        identity_action_correct,
        identity_action_wrong,
        source_action_residual_correct,
        source_action_residual_wrong,
        source_action_correct_rms,
        source_action_wrong_rms,
        source_action_invariance_cosine,
        source_action_invariance_symmetric_norm_ratio,
        projected_action_alignment_correct,
        projected_action_alignment_wrong,
        normalized_tangents,
        tangent_norms,
        source_tangent_rms,
        robust_source_tangent_rms,
        normalized_gram,
        normalized_gram_pinv,
        coefficients,
        projected,
        capped_teacher,
        projection_retention,
        raw_rms,
        projected_rms,
        capped_rms,
        cap_reference_rms,
        action_cap,
        cap_scale,
        action_scale,
        max_abs_postprojection_cosine,
    )
    if any(
        value.dtype != torch.float32
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
        for value in detached_outputs
    ):
        raise IdentityAnchoredActionResidualError(
            "IAR teacher/projection diagnostics must remain finite detached FP32"
        )
    if bool((capped_rms > action_cap + 2.0e-6).any().item()):
        raise IdentityAnchoredActionResidualError(
            "per-sigma action residual escaped its gauge-invariant RMS cap"
        )
    if bool((projection_retention > 1.0 + 2.0e-5).any().item()):
        raise IdentityAnchoredActionResidualError(
            "wrong-source tangent projection increased action L2"
        )
    if bool(
        (
            max_abs_postprojection_cosine
            > float(config.postprojection_cosine_tolerance)
        ).any().item()
    ):
        raise IdentityAnchoredActionResidualError(
            "orthogonal span projection failed its post-tangent cosine gate"
        )
    if tangent_rank.dtype not in (torch.int32, torch.int64):
        raise IdentityAnchoredActionResidualError(
            "tangent rank diagnostic must remain integral"
        )
    low_sigma_mask = action_scale == 0.0
    if bool(
        (
            capped_teacher
            * _view_batch(low_sigma_mask, capped_teacher).to(torch.float32)
        ).ne(0.0).any().item()
    ):
        raise IdentityAnchoredActionResidualError(
            "zero-scale sigma must produce an exactly zero action teacher"
        )

    diagnostics = IARFrozenDiagnostics(
        softmin_weights=softmin_weights,
        hard_negative_barycenter=negative_barycenter,
        raw_action_residual=raw_action,
        identity_tangents=identity_tangents,
        normalized_identity_tangents=normalized_tangents,
        tangent_norms=tangent_norms,
        source_tangent_rms=source_tangent_rms,
        robust_source_tangent_rms=robust_source_tangent_rms,
        source_conditioned_action_residual_correct=(
            source_action_residual_correct
        ),
        source_conditioned_action_residual_wrong_sources=(
            source_action_residual_wrong
        ),
        source_conditioned_action_correct_rms=source_action_correct_rms,
        source_conditioned_action_wrong_rms=source_action_wrong_rms,
        source_action_invariance_cosine=source_action_invariance_cosine,
        source_action_invariance_symmetric_norm_ratio=(
            source_action_invariance_symmetric_norm_ratio
        ),
        projected_action_alignment_correct=(
            projected_action_alignment_correct
        ),
        projected_action_alignment_wrong_sources=(
            projected_action_alignment_wrong
        ),
        normalized_tangent_gram_fp32=normalized_gram,
        normalized_tangent_gram_pinv_fp32=normalized_gram_pinv,
        tangent_rank=tangent_rank,
        projection_coefficients=coefficients.squeeze(-1),
        projected_action_residual=projected,
        capped_action_residual=capped_teacher,
        projection_retention=projection_retention,
        raw_action_rms=raw_rms,
        projected_action_rms=projected_rms,
        capped_action_rms=capped_rms,
        cap_reference_rms=cap_reference_rms,
        action_rms_cap=action_cap,
        cap_scale=cap_scale,
        sigma_action_scale=action_scale,
        max_abs_postprojection_tangent_cosine=max_abs_postprojection_cosine,
    )
    retention_values = _float_list(projection_retention)
    post_cosine_values = _float_list(max_abs_postprojection_cosine)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD_NAME,
        "frozen_teacher_only": True,
        "experimental_tensor_core": True,
        "scientific_claim_authorized": False,
        "trainer_integration_authorized": False,
        "execution_scope": {
            "model_forward_performed_by_this_function": False,
            "optimizer_step_performed_by_this_function": False,
            "upstream_model_forward_provenance_verified": False,
        },
        "direct_core_arguments": {
            "paired_target_video_argument_present": False,
            "target_rgb_argument_present": False,
            "target_clean_latent_argument_present": False,
            "spatial_mask_argument_present": False,
            "object_track_argument_present": False,
            "optical_flow_argument_present": False,
            "pose_or_trajectory_argument_present": False,
            "upstream_derivation_provenance_verified": False,
        },
        "shape_contract": {
            "verified": True,
            "batch_size": batch,
            "velocity_feature_shape": list(feature_shape),
            "hard_negative_count": hard_count,
            "wrong_source_count": wrong_count,
        },
        "shared_query_contract": {
            "single_noisy_state_object": True,
            "branch_object_alias_verified": True,
            "branch_names": list(fields.shared_state.branch_names),
            "branch_count": len(fields.shared_state.branch_names),
            "model_forward_provenance_verified": False,
            "upstream_callback_audit_required": True,
        },
        "semantic_binding_contract": {
            "caller_supplied_immutable_mode_text_source_digests": True,
            "branch_order_and_digest_relationships_verified": True,
            "upstream_text_source_content_provenance_verified": False,
            "action_and_noop_text_distinct": True,
            "noop_present_in_hard_negative_set": True,
            "correct_source_shared_by_noop_and_action": True,
            "wrong_sources_unique_and_distinct": True,
            "noop_and_action_wrong_sources_paired_in_order": True,
        },
        "teacher": {
            "construction": (
                "orthogonal_normalized_pinv_projection_then_gauge_invariant_"
                "per_sigma_cap(action_minus_softmin_hard_negatives,"
                "noop_correct_minus_noop_wrong_source_nuisance_tangents)"
            ),
            "softmin_temperature": float(config.hard_negative_temperature),
            "lower_hard_negative_energy_receives_more_weight": True,
            "softmin_weight_sum_per_sample": _float_list(
                softmin_weights.sum(dim=1)
            ),
            "projection_dtype": "torch.float32",
            "projection_solver": (
                "torch.linalg.pinv(normalized_tangent_gram,hermitian=True)"
            ),
            "normalized_tangent_gram_shape": list(normalized_gram.shape),
            "projection_rank_rtol": float(config.projection_rank_rtol),
            "tangent_rank_per_sample": _int_list(tangent_rank),
            "postprojection_tangent_cosine_max_per_sample": post_cosine_values,
            "postprojection_tangent_cosine_tolerance": float(
                config.postprojection_cosine_tolerance
            ),
            "postprojection_tangent_cosine_gate_passed": True,
            "action_rms_cap_ratio": float(config.action_rms_cap_ratio),
            "cap_reference": (
                "min(raw_action_rms,median(noop_source_swap_tangent_rms))"
            ),
            "cap_reference_rms_per_sample": _float_list(cap_reference_rms),
            "robust_source_tangent_rms_per_sample": _float_list(
                robust_source_tangent_rms
            ),
            "projection_retention_per_sample": retention_values,
            "projection_retention_min": min(retention_values),
            "projection_retention_mean": sum(retention_values) / len(retention_values),
            "projection_retention_max": max(retention_values),
            "raw_action_rms_per_sample": _float_list(raw_rms),
            "projected_action_rms_per_sample": _float_list(projected_rms),
            "capped_action_rms_per_sample": _float_list(capped_rms),
            "action_rms_cap_per_sample": _float_list(action_cap),
            "cap_scale_per_sample": _float_list(cap_scale),
            "sigma_action_scale_per_sample": _float_list(action_scale),
            "zero_scale_teacher_exactly_zero": True,
            "finite": True,
        },
        "action_source_invariance_diagnostic": {
            "diagnostic_only_not_used_for_projection_or_cap": True,
            "correct_residual_definition": (
                "frozen_identity_action_correct_minus_"
                "frozen_identity_noop_correct"
            ),
            "wrong_residual_definition": (
                "frozen_identity_action_wrong_source_minus_"
                "paired_frozen_identity_noop_wrong_source"
            ),
            "wrong_source_count": wrong_count,
            "minimum_donors_for_calibrated_use": 2,
            "plumbing_only_uncalibrated": wrong_count == 1,
            "donor_count_meets_documented_minimum": wrong_count >= 2,
            "matched_donor_quality_or_calibration_verified": False,
            "projection_executed_by_tensor_core": True,
            "projection_authorized_for_training": False,
            "training_authorized_by_this_diagnostic": False,
            "correct_wrong_comparison_meaningful_per_sample_and_source": [
                [bool(value) for value in row]
                for row in correct_wrong_comparison_meaningful
                .detach()
                .cpu()
                .tolist()
            ],
            "projected_alignment_correct_meaningful_per_sample": [
                bool(value)
                for value in projected_alignment_correct_meaningful
                .detach()
                .cpu()
                .tolist()
            ],
            "projected_alignment_wrong_meaningful_per_sample_and_source": [
                [bool(value) for value in row]
                for row in projected_alignment_wrong_meaningful
                .detach()
                .cpu()
                .tolist()
            ],
            "correct_rms_per_sample": _float_list(
                source_action_correct_rms
            ),
            "wrong_rms_per_sample_and_source": [
                [float(value) for value in row]
                for row in source_action_wrong_rms.detach().cpu().tolist()
            ],
            "correct_wrong_cosine_per_sample_and_source": [
                [float(value) for value in row]
                for row in source_action_invariance_cosine.detach().cpu().tolist()
            ],
            "correct_wrong_symmetric_norm_ratio_per_sample_and_source": [
                [float(value) for value in row]
                for row in source_action_invariance_symmetric_norm_ratio
                .detach()
                .cpu()
                .tolist()
            ],
            "projected_t2v_alignment_correct_per_sample": _float_list(
                projected_action_alignment_correct
            ),
            "projected_t2v_alignment_wrong_per_sample_and_source": [
                [float(value) for value in row]
                for row in projected_action_alignment_wrong
                .detach()
                .cpu()
                .tolist()
            ],
            "upstream_branch_provenance_verified": False,
        },
        "stop_gradient_contract": {
            "query_and_sigma_detached": True,
            "all_frozen_velocity_inputs_detached": True,
            "hard_negative_energies_detached": True,
            "softmin_weights_detached": True,
            "identity_tangents_detached": True,
            "projection_and_cap_detached": True,
            "teacher_action_residual_detached": True,
        },
    }
    return IARFrozenTeacherResult(
        teacher_action_residual=capped_teacher,
        diagnostics=diagnostics,
        receipt=receipt,
    )


def compute_frozen_identity_anchored_teacher(
    fields: IARFrozenFields,
    *,
    config: IARConfig = IARConfig(),
) -> IARFrozenTeacherResult:
    """Evaluate the detached IAR teacher without student tensors or backward."""

    config.validate()
    batch, hard_count, wrong_count, feature_shape = _validate_frozen_fields(fields)
    return _compute_frozen_teacher_validated(
        fields,
        config,
        batch=batch,
        hard_count=hard_count,
        wrong_count=wrong_count,
        feature_shape=feature_shape,
    )


def compute_identity_anchored_action_residual(
    fields: IARFields,
    *,
    config: IARConfig = IARConfig(),
) -> IARLossResult:
    """Build one detached IAR teacher and three graph-carrying FP32 losses."""

    torch = _torch()
    config.validate()
    batch, hard_count, wrong_count, feature_shape = _validate_fields(fields)

    frozen_branch_count = len(
        expected_frozen_branch_names(hard_count, wrong_count)
    )
    frozen_fields = IARFrozenFields(
        shared_state=SharedStateBinding(
            noised_state=fields.shared_state.noised_state,
            branch_names=fields.shared_state.branch_names[:frozen_branch_count],
            branch_state_refs=fields.shared_state.branch_state_refs[
                :frozen_branch_count
            ],
        ),
        semantic_binding=BranchSemanticBinding(
            branches=fields.semantic_binding.branches[:frozen_branch_count]
        ),
        sigma=fields.sigma,
        frozen_t2v_action=fields.frozen_t2v_action,
        frozen_t2v_hard_negatives=fields.frozen_t2v_hard_negatives,
        hard_negative_energies=fields.hard_negative_energies,
        frozen_identity_noop_correct=fields.frozen_identity_noop_correct,
        frozen_identity_noop_wrong_sources=(
            fields.frozen_identity_noop_wrong_sources
        ),
        frozen_identity_action_correct=fields.frozen_identity_action_correct,
        frozen_identity_action_wrong_sources=(
            fields.frozen_identity_action_wrong_sources
        ),
    )
    frozen_result = _compute_frozen_teacher_validated(
        frozen_fields,
        config,
        batch=batch,
        hard_count=hard_count,
        wrong_count=wrong_count,
        feature_shape=feature_shape,
    )
    frozen_diagnostics = frozen_result.diagnostics
    softmin_weights = frozen_diagnostics.softmin_weights
    negative_barycenter = frozen_diagnostics.hard_negative_barycenter
    raw_action = frozen_diagnostics.raw_action_residual
    identity_tangents = frozen_diagnostics.identity_tangents
    normalized_tangents = frozen_diagnostics.normalized_identity_tangents
    tangent_norms = frozen_diagnostics.tangent_norms
    source_tangent_rms = frozen_diagnostics.source_tangent_rms
    robust_source_tangent_rms = (
        frozen_diagnostics.robust_source_tangent_rms
    )
    source_action_residual_correct = (
        frozen_diagnostics.source_conditioned_action_residual_correct
    )
    source_action_residual_wrong = (
        frozen_diagnostics.source_conditioned_action_residual_wrong_sources
    )
    source_action_correct_rms = (
        frozen_diagnostics.source_conditioned_action_correct_rms
    )
    source_action_wrong_rms = (
        frozen_diagnostics.source_conditioned_action_wrong_rms
    )
    source_action_invariance_cosine = (
        frozen_diagnostics.source_action_invariance_cosine
    )
    source_action_invariance_symmetric_norm_ratio = (
        frozen_diagnostics.source_action_invariance_symmetric_norm_ratio
    )
    projected_action_alignment_correct = (
        frozen_diagnostics.projected_action_alignment_correct
    )
    projected_action_alignment_wrong = (
        frozen_diagnostics.projected_action_alignment_wrong_sources
    )
    normalized_gram = frozen_diagnostics.normalized_tangent_gram_fp32
    normalized_gram_pinv = (
        frozen_diagnostics.normalized_tangent_gram_pinv_fp32
    )
    tangent_rank = frozen_diagnostics.tangent_rank
    projected = frozen_diagnostics.projected_action_residual
    capped_teacher = frozen_result.teacher_action_residual
    projection_retention = frozen_diagnostics.projection_retention
    raw_rms = frozen_diagnostics.raw_action_rms
    projected_rms = frozen_diagnostics.projected_action_rms
    capped_rms = frozen_diagnostics.capped_action_rms
    cap_reference_rms = frozen_diagnostics.cap_reference_rms
    action_cap = frozen_diagnostics.action_rms_cap
    cap_scale = frozen_diagnostics.cap_scale
    action_scale = frozen_diagnostics.sigma_action_scale
    max_abs_postprojection_cosine = (
        frozen_diagnostics.max_abs_postprojection_tangent_cosine
    )
    identity_noop_correct = fields.frozen_identity_noop_correct.detach().to(
        dtype=torch.float32
    )

    student_action = fields.student_action.to(dtype=torch.float32)
    student_noop = fields.student_noop.to(dtype=torch.float32)
    student_view_a = fields.student_identity_view_a.to(dtype=torch.float32)
    student_view_b = fields.student_identity_view_b.to(dtype=torch.float32)
    # The action objective may move the action branch, never the no-op
    # baseline.  Identity/no-op behavior has its own independently weighted
    # anchor below.
    student_action_residual = student_action - student_noop.detach()

    action_per_sample = _batch_mse(student_action_residual, capped_teacher)
    identity_per_sample = _batch_mse(student_noop, identity_noop_correct)
    view_anchor_a_per_sample = _batch_mse(
        student_view_a, identity_noop_correct
    )
    view_anchor_b_per_sample = _batch_mse(
        student_view_b, identity_noop_correct
    )
    view_consistency_per_sample = _batch_mse(
        student_view_a, student_view_b.detach()
    )
    view_per_sample = (
        0.5 * (view_anchor_a_per_sample + view_anchor_b_per_sample)
        + float(config.view_consistency_weight)
        * view_consistency_per_sample
    )
    # Sigma changes the detached teacher cap.  The MSE remains active at low
    # sigma so a zero teacher actually pulls the student residual to zero.
    action_loss = action_per_sample.mean()
    identity_loss = identity_per_sample.mean()
    view_loss = view_per_sample.mean()
    total = (
        float(config.action_loss_weight) * action_loss
        + float(config.identity_loss_weight) * identity_loss
        + float(config.view_loss_weight) * view_loss
    )

    loss_values = (
        total,
        action_loss,
        identity_loss,
        view_loss,
        action_per_sample,
        identity_per_sample,
        view_anchor_a_per_sample,
        view_anchor_b_per_sample,
        view_consistency_per_sample,
        view_per_sample,
        student_action_residual,
    )
    if any(
        value.dtype != torch.float32
        or not value.requires_grad
        or value.grad_fn is None
        or not bool(torch.isfinite(value).all().item())
        for value in loss_values
    ):
        raise IdentityAnchoredActionResidualError(
            "IAR student losses/diagnostics must be finite graph-carrying FP32"
        )
    if any(value.ndim != 0 for value in (total, action_loss, identity_loss, view_loss)):
        raise IdentityAnchoredActionResidualError(
            "IAR losses must be scalar tensors"
        )
    diagnostics = IARDiagnostics(
        softmin_weights=softmin_weights,
        hard_negative_barycenter=negative_barycenter,
        raw_action_residual=raw_action,
        identity_tangents=identity_tangents,
        normalized_identity_tangents=normalized_tangents,
        tangent_norms=tangent_norms,
        source_tangent_rms=source_tangent_rms,
        robust_source_tangent_rms=robust_source_tangent_rms,
        source_conditioned_action_residual_correct=(
            source_action_residual_correct
        ),
        source_conditioned_action_residual_wrong_sources=(
            source_action_residual_wrong
        ),
        source_conditioned_action_correct_rms=source_action_correct_rms,
        source_conditioned_action_wrong_rms=source_action_wrong_rms,
        source_action_invariance_cosine=source_action_invariance_cosine,
        source_action_invariance_symmetric_norm_ratio=(
            source_action_invariance_symmetric_norm_ratio
        ),
        projected_action_alignment_correct=(
            projected_action_alignment_correct
        ),
        projected_action_alignment_wrong_sources=(
            projected_action_alignment_wrong
        ),
        normalized_tangent_gram_fp32=normalized_gram,
        normalized_tangent_gram_pinv_fp32=normalized_gram_pinv,
        tangent_rank=tangent_rank,
        projection_coefficients=frozen_diagnostics.projection_coefficients,
        projected_action_residual=projected,
        capped_action_residual=capped_teacher,
        projection_retention=projection_retention,
        raw_action_rms=raw_rms,
        projected_action_rms=projected_rms,
        capped_action_rms=capped_rms,
        cap_reference_rms=cap_reference_rms,
        action_rms_cap=action_cap,
        cap_scale=cap_scale,
        sigma_action_scale=action_scale,
        max_abs_postprojection_tangent_cosine=max_abs_postprojection_cosine,
        student_action_residual=student_action_residual,
        action_per_sample=action_per_sample,
        identity_per_sample=identity_per_sample,
        view_anchor_a_per_sample=view_anchor_a_per_sample,
        view_anchor_b_per_sample=view_anchor_b_per_sample,
        view_consistency_per_sample=view_consistency_per_sample,
        view_per_sample=view_per_sample,
    )

    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD_NAME,
        "experimental_tensor_core": True,
        "scientific_claim_authorized": False,
        "trainer_integration_authorized": False,
        "execution_scope": {
            "model_forward_performed_by_this_function": False,
            "optimizer_step_performed_by_this_function": False,
            "upstream_model_forward_provenance_verified": False,
        },
        "direct_core_arguments": {
            "paired_target_video_argument_present": False,
            "target_rgb_argument_present": False,
            "target_clean_latent_argument_present": False,
            "spatial_mask_argument_present": False,
            "object_track_argument_present": False,
            "optical_flow_argument_present": False,
            "pose_or_trajectory_argument_present": False,
            "upstream_derivation_provenance_verified": False,
        },
        "shape_contract": {
            "verified": True,
            "batch_size": batch,
            "velocity_feature_shape": list(feature_shape),
            "hard_negative_count": hard_count,
            "wrong_source_count": wrong_count,
        },
        "shared_query_contract": {
            "single_noisy_state_object": True,
            "branch_object_alias_verified": True,
            "branch_names": list(fields.shared_state.branch_names),
            "branch_count": len(fields.shared_state.branch_names),
            "model_forward_provenance_verified": False,
            "upstream_callback_audit_required": True,
        },
        "semantic_binding_contract": {
            "caller_supplied_immutable_mode_text_source_digests": True,
            "branch_order_and_digest_relationships_verified": True,
            "upstream_text_source_content_provenance_verified": False,
            "student_action_binds_action_text_and_correct_source": True,
            "student_noop_and_views_bind_noop_text_and_correct_source": True,
        },
        "teacher": dict(frozen_result.receipt["teacher"]),
        "action_source_invariance_diagnostic": dict(
            frozen_result.receipt["action_source_invariance_diagnostic"]
        ),
        "stop_gradient_contract": {
            "query_and_sigma_detached": True,
            "all_frozen_velocity_inputs_detached": True,
            "hard_negative_energies_detached": True,
            "softmin_weights_detached": True,
            "identity_tangents_detached": True,
            "action_source_invariance_diagnostics_detached": True,
            "projection_and_cap_detached": True,
            "teacher_action_residual_detached": True,
            "student_tensor_autograd_enabled": True,
            "student_model_parameter_connectivity_verified": False,
            "student_noop_detached_inside_action_residual": True,
            "action_loss_noop_branch_gradient_blocked": True,
            "view_consistency_b_target_detached": True,
            "both_views_anchored_to_detached_noop_correct": True,
        },
        "loss": {
            "dtype": "torch.float32",
            "finite": True,
            "action_mse_uniform_across_sigma": True,
            "sigma_schedule_applied_only_to_teacher_cap": True,
            "action_residual": "student_action-stopgrad(student_noop)",
            "identity_target": "frozen_identity_noop_correct",
            "view_objective": (
                "mean(two_noop_anchors)+view_consistency_weight*"
                "mse(view_a,stopgrad(view_b))"
            ),
            "action": float(action_loss.detach().cpu().item()),
            "identity": float(identity_loss.detach().cpu().item()),
            "view": float(view_loss.detach().cpu().item()),
            "view_anchor_a": float(
                view_anchor_a_per_sample.mean().detach().cpu().item()
            ),
            "view_anchor_b": float(
                view_anchor_b_per_sample.mean().detach().cpu().item()
            ),
            "view_consistency": float(
                view_consistency_per_sample.mean().detach().cpu().item()
            ),
            "total": float(total.detach().cpu().item()),
            "weights": {
                "action": float(config.action_loss_weight),
                "identity": float(config.identity_loss_weight),
                "view": float(config.view_loss_weight),
                "view_consistency_inside_view": float(
                    config.view_consistency_weight
                ),
            },
        },
    }
    return IARLossResult(
        total=total,
        action=action_loss,
        identity=identity_loss,
        view=view_loss,
        teacher_action_residual=capped_teacher,
        diagnostics=diagnostics,
        receipt=receipt,
    )


__all__ = [
    "BranchSemantic",
    "BranchSemanticBinding",
    "IARConfig",
    "IARDiagnostics",
    "IARFields",
    "IARFrozenDiagnostics",
    "IARFrozenFields",
    "IARFrozenTeacherResult",
    "IARLossResult",
    "IdentityAnchoredActionResidualError",
    "METHOD_NAME",
    "RECEIPT_SCHEMA",
    "SharedStateBinding",
    "bind_branch_semantics",
    "bind_shared_state",
    "compute_frozen_identity_anchored_teacher",
    "compute_identity_anchored_action_residual",
    "expected_branch_names",
    "expected_frozen_branch_names",
    "sigma_action_scale",
]
