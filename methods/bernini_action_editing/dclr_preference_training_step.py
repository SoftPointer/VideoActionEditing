"""Production bridge from a packed Bernini preference pair to one LoRA step.

The bridge owns the operations intentionally absent from the pure batch and
objective modules: two real ``shared_step`` calls, exact candidate splitting,
backward, a model-wide gradient audit, and one explicit optimizer step.  It
does not accept predictions, energies, losses, or caller-filled scalar rewards.

Both forwards are content-bound to one base-checkpoint/query/pair/sigma/
epsilon/RoPE identity.  The current path must retain a non-leaf model graph;
the collection-reference model must be a distinct, eval-mode, fully frozen
parameter surface.  Only LoRA A/B weights on the route-authorized attention
Q/O projections may be trainable or receive gradients.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import re
from typing import Any

import torch

try:  # Package import.
    from . import dclr_counterfactual_bank as counterfactual_bank
    from . import dclr_preference_batch as preference_batch
    from . import dclr_preference_objective as preference_objective
    from . import dclr_runtime_contract as runtime_contract
except ImportError:  # Direct import with METHOD_ROOT on sys.path.
    import dclr_counterfactual_bank as counterfactual_bank
    import dclr_preference_batch as preference_batch
    import dclr_preference_objective as preference_objective
    import dclr_runtime_contract as runtime_contract


SCHEMA_VERSION = "bernini-dclr-preference-training-step-v1"
CHECKPOINT_DIGEST_ATTRIBUTE = "dclr_checkpoint_digest"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_LORA_A_MARKER = ".lora_A."
_LORA_B_MARKER = ".lora_B."


class DCLRPreferenceTrainingStepError(RuntimeError):
    """A forward, binding, route, gradient, or optimizer invariant failed."""


@dataclass(frozen=True)
class SharedStepBinding:
    """Content identity that both current and reference forwards must match.

    ``checkpoint_digest`` identifies their common immutable Bernini base
    checkpoint.  Current LoRA state may differ from the frozen collection
    reference; it is deliberately not represented as the common checkpoint.
    """

    checkpoint_digest: str
    query_digest: str
    pair_digest: str
    sigma_digest: str
    epsilon_digest: str
    rope_digest: str


@dataclass(frozen=True)
class GradientAudit:
    """Model-wide proof recorded immediately before ``optimizer.step``."""

    route_attention: str
    trainable_parameter_names: tuple[str, ...]
    finite_nonzero_gradient_names: tuple[str, ...]
    gradient_l2_norms: tuple[float, ...]
    frozen_parameter_count: int
    frozen_gradients_are_none: bool
    reference_parameter_count: int
    reference_gradients_are_none: bool


@dataclass(frozen=True)
class PreferenceTrainingStepResult:
    """Detached audit/metric receipt for one completed optimizer step."""

    schema_version: str
    binding: SharedStepBinding
    route: preference_objective.OneSidedNearMissRoute
    loss: float
    delta: float
    current_margin: float
    reference_margin: float
    winner_current_energy: float
    winner_reference_energy: float
    loser_current_energy: float
    loser_reference_energy: float
    gradient_audit: GradientAudit
    current_shared_step_calls: int
    reference_shared_step_calls: int
    optimizer_step_performed: bool


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DCLRPreferenceTrainingStepError(
            f"{label} must be one lowercase SHA-256 digest"
        )
    return value


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DCLRPreferenceTrainingStepError(
            f"binding payload is not canonical JSON: {error}"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _require_tensor(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise DCLRPreferenceTrainingStepError(f"{label} must be a torch.Tensor")
    if value.device.type == "meta" or value.layout != torch.strided:
        raise DCLRPreferenceTrainingStepError(
            f"{label} must be a concrete strided tensor"
        )
    return value


def tensor_content_sha256(value: Any, *, label: str) -> str:
    """Hash tensor dtype, shape, and exact logical bytes, device-independently."""

    tensor = _require_tensor(value, label=label)
    detached = tensor.detach().contiguous().cpu()
    byte_tensor = detached.view(torch.uint8).contiguous().clone()
    if byte_tensor.storage_offset() != 0 or not byte_tensor.is_contiguous():
        raise DCLRPreferenceTrainingStepError(
            f"{label} byte clone is not zero-offset contiguous storage"
        )
    untyped_storage = getattr(byte_tensor, "untyped_storage", None)
    has_untyped_storage = callable(untyped_storage)
    storage = untyped_storage() if has_untyped_storage else byte_tensor.storage()
    nbytes_method = getattr(storage, "nbytes", None)
    storage_nbytes = (
        int(nbytes_method()) if callable(nbytes_method) else int(storage.size())
    )
    if storage_nbytes != byte_tensor.numel():
        raise DCLRPreferenceTrainingStepError(
            f"{label} byte storage contains padding or aliased bytes"
        )
    if has_untyped_storage:
        raw = bytes(storage)
    else:
        # PyTorch before untyped_storage() exposes only TypedStorage.  Calling
        # bytes(TypedStorage) walks it one Python scalar at a time, so read the
        # already validated CPU uint8 clone directly instead.  string_at makes
        # no NumPy/ABI round trip and preserves the exact logical byte stream.
        byte_count = int(byte_tensor.numel())
        data_pointer = int(byte_tensor.data_ptr())
        if byte_count > 0 and data_pointer == 0:
            raise DCLRPreferenceTrainingStepError(
                f"{label} byte storage has a null data pointer"
            )
        raw = ctypes.string_at(data_pointer, byte_count)
    if len(raw) != byte_tensor.numel():
        raise DCLRPreferenceTrainingStepError(
            f"{label} byte serialization length differs from tensor content"
        )
    header = json.dumps(
        {
            "dtype": str(detached.dtype),
            "shape": [int(item) for item in detached.shape],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _validate_text_query(
    cond_embeds: Any,
    batch_text_seqlen: Any,
    *,
    batch: preference_batch.PackedPreferenceBatch,
) -> tuple[torch.Tensor, tuple[int, int]]:
    cond = _require_tensor(cond_embeds, label="cond_embeds")
    if (
        not cond.is_floating_point()
        or cond.ndim != 3
        or int(cond.shape[0]) != 1
        or int(cond.shape[1]) <= 0
        or int(cond.shape[2]) <= 0
        or cond.device != batch.noisy_latents.device
        or cond.requires_grad
        or cond.grad_fn is not None
        or not bool(torch.isfinite(cond).all().item())
    ):
        raise DCLRPreferenceTrainingStepError(
            "cond_embeds must be finite detached [1,T,D] on the packed device"
        )
    if (
        not isinstance(batch_text_seqlen, Sequence)
        or isinstance(batch_text_seqlen, (str, bytes))
        or len(batch_text_seqlen) != 2
    ):
        raise DCLRPreferenceTrainingStepError(
            "batch_text_seqlen must contain exactly two logical lengths"
        )
    lengths: list[int] = []
    for index, value in enumerate(batch_text_seqlen):
        if type(value) is not int or value <= 0:
            raise DCLRPreferenceTrainingStepError(
                f"batch_text_seqlen[{index}] must be a positive plain integer"
            )
        lengths.append(value)
    if sum(lengths) != int(cond.shape[1]):
        raise DCLRPreferenceTrainingStepError(
            "batch_text_seqlen does not cover the packed text embeddings"
        )
    return cond, (lengths[0], lengths[1])


def _require_model_id(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DCLRPreferenceTrainingStepError(
            "model_id must be one nonempty whitespace-trimmed string"
        )
    return value


def build_shared_step_binding(
    *,
    checkpoint_digest: str,
    pair_digest: str,
    batch: preference_batch.PackedPreferenceBatch,
    cond_embeds: torch.Tensor,
    batch_text_seqlen: Sequence[int],
    model_id: str,
) -> SharedStepBinding:
    """Recompute the complete identity of one packed ``shared_step`` query."""

    checked = preference_batch.validate_packed_preference_batch(batch)
    cond, text_lengths = _validate_text_query(
        cond_embeds, batch_text_seqlen, batch=checked
    )
    model_name = _require_model_id(model_id)
    checkpoint = _require_sha256(
        checkpoint_digest, label="checkpoint_digest"
    )
    pair = _require_sha256(pair_digest, label="pair_digest")
    noisy_digest = tensor_content_sha256(
        checked.noisy_latents, label="noisy_latents"
    )
    timestep_digest = tensor_content_sha256(
        checked.timesteps, label="timesteps"
    )
    cond_digest = tensor_content_sha256(cond, label="cond_embeds")
    rope_digest = tensor_content_sha256(
        checked.rotary_embs, label="rotary_embs"
    )
    query_digest = _canonical_sha256(
        {
            "model_id": model_name,
            "noisy_latents_digest": noisy_digest,
            "timesteps_digest": timestep_digest,
            "cond_embeds_digest": cond_digest,
            "rotary_embs_digest": rope_digest,
            "batch_vae_seqlen": list(checked.batch_vae_seqlen),
            "batch_text_seqlen": list(text_lengths),
        }
    )
    return SharedStepBinding(
        checkpoint_digest=checkpoint,
        query_digest=query_digest,
        pair_digest=pair,
        sigma_digest=tensor_content_sha256(
            checked.flow_state.sigma, label="flow_state.sigma"
        ),
        epsilon_digest=tensor_content_sha256(
            checked.flow_state.epsilon, label="flow_state.epsilon"
        ),
        rope_digest=rope_digest,
    )


def _validate_binding(value: Any, *, label: str) -> SharedStepBinding:
    if not isinstance(value, SharedStepBinding):
        raise DCLRPreferenceTrainingStepError(
            f"{label} must be a SharedStepBinding"
        )
    for field in (
        "checkpoint_digest",
        "query_digest",
        "pair_digest",
        "sigma_digest",
        "epsilon_digest",
        "rope_digest",
    ):
        _require_sha256(getattr(value, field), label=f"{label}.{field}")
    return value


def _validated_preference_route(
    pair: Any,
    *,
    receipts_by_digest: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    counterfactual_bank_document: Mapping[str, Any],
    split_ledger: Mapping[str, Any],
    source_manifest_sha256: str,
    artifacts_by_digest: Mapping[str, Mapping[str, Any]],
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    preference_objective.OneSidedNearMissRoute,
]:
    """Revalidate the full upstream closure and derive, never accept, route gates."""

    try:
        validated_pair = counterfactual_bank.validate_preference_pair(
            pair,
            receipts_by_digest,
            sources,
            counterfactual_bank_document,
            split_ledger=split_ledger,
            source_manifest_sha256=source_manifest_sha256,
            artifacts_by_digest=artifacts_by_digest,
        )
        winner = receipts_by_digest[validated_pair["winner_receipt_digest"]]
        loser = receipts_by_digest[validated_pair["loser_receipt_digest"]]
        route = preference_objective.route_one_sided_nearmiss(
            str(validated_pair["pair_type"]),
            winner_action_axis_pass=winner["action_axis_pass"],
            winner_preservation_axis_pass=winner["preservation_axis_pass"],
            loser_action_axis_pass=loser["action_axis_pass"],
            loser_preservation_axis_pass=loser["preservation_axis_pass"],
        )
    except (
        counterfactual_bank.DCLRCounterfactualBankError,
        preference_objective.DCLRPreferenceObjectiveError,
        KeyError,
        TypeError,
    ) as error:
        raise DCLRPreferenceTrainingStepError(str(error)) from error
    return validated_pair, winner, loser, route


def _named_parameters(model: Any, *, label: str) -> tuple[tuple[str, torch.Tensor], ...]:
    method = getattr(model, "named_parameters", None)
    if not callable(method):
        raise DCLRPreferenceTrainingStepError(
            f"{label} must expose model-wide named_parameters()"
        )
    try:
        named = tuple(method())
    except Exception as error:
        raise DCLRPreferenceTrainingStepError(
            f"cannot enumerate {label} parameters: {error}"
        ) from error
    if not named:
        raise DCLRPreferenceTrainingStepError(f"{label} parameter surface is empty")
    names: set[str] = set()
    identities: set[int] = set()
    for index, item in enumerate(named):
        if not isinstance(item, tuple) or len(item) != 2:
            raise DCLRPreferenceTrainingStepError(
                f"{label} named parameter {index} is malformed"
            )
        name, parameter = item
        if not isinstance(name, str) or not name or name in names:
            raise DCLRPreferenceTrainingStepError(
                f"{label} parameter names must be unique nonempty strings"
            )
        tensor = _require_tensor(parameter, label=f"{label}.{name}")
        if not tensor.is_floating_point() or tensor.numel() <= 0:
            raise DCLRPreferenceTrainingStepError(
                f"{label}.{name} must be a nonempty floating parameter"
            )
        if id(tensor) in identities:
            raise DCLRPreferenceTrainingStepError(
                f"{label} repeats one parameter under multiple names"
            )
        names.add(name)
        identities.add(id(tensor))
    return named


def _parameter_owner(model: Any, *, label: str) -> Any:
    """Resolve the module owned by an official ``GEN_Wanx22`` core.

    The official diffusion core owns ``shared_step`` while its ``transformer``
    owns the parameters.  Tests and thin wrappers may expose both on one object.
    """

    if callable(getattr(model, "named_parameters", None)):
        return model
    transformer = getattr(model, "transformer", None)
    if transformer is None or not callable(
        getattr(transformer, "named_parameters", None)
    ):
        raise DCLRPreferenceTrainingStepError(
            f"{label} exposes neither named_parameters() nor a transformer owner"
        )
    return transformer


def _classify_allowed_lora(
    name: str, *, route_attention: str
) -> tuple[str, str] | None:
    framed = f".{name}."
    matrix = None
    if _LORA_A_MARKER in framed:
        matrix = "A"
    elif _LORA_B_MARKER in framed:
        matrix = "B"
    if matrix is None:
        return None
    query = f".{route_attention}.to_q."
    output = f".{route_attention}.to_out.0."
    if query in framed:
        return "to_q", matrix
    if output in framed:
        return "to_out.0", matrix
    return None


def _validate_parameter_surface(
    current_model: Any,
    reference_model: Any,
    *,
    route: preference_objective.OneSidedNearMissRoute,
) -> tuple[
    tuple[tuple[str, torch.Tensor], ...],
    tuple[tuple[str, torch.Tensor], ...],
    tuple[tuple[str, torch.Tensor], ...],
    str,
]:
    if current_model is reference_model:
        raise DCLRPreferenceTrainingStepError(
            "current and collection-reference models must be distinct objects"
        )
    current_owner = _parameter_owner(current_model, label="current model")
    reference_owner = _parameter_owner(
        reference_model, label="collection reference"
    )
    if current_owner is reference_owner:
        raise DCLRPreferenceTrainingStepError(
            "current and collection-reference parameter owners must be distinct"
        )
    if getattr(current_owner, "training", None) is not True:
        raise DCLRPreferenceTrainingStepError("current model must be in train mode")
    if getattr(reference_owner, "training", None) is not False:
        raise DCLRPreferenceTrainingStepError(
            "collection-reference model must be in eval mode"
        )
    current = _named_parameters(current_owner, label="current model")
    reference = _named_parameters(
        reference_owner, label="collection reference"
    )
    if any(parameter.requires_grad for _, parameter in reference):
        raise DCLRPreferenceTrainingStepError(
            "collection-reference model must be fully frozen"
        )
    if any(parameter.grad is not None for _, parameter in reference):
        raise DCLRPreferenceTrainingStepError(
            "collection-reference parameters must have grad=None"
        )

    route_attention = (
        "attn2"
        if route.active_adapter == preference_objective.ACTION_ADAPTER
        else "attn1"
    )
    active: list[tuple[str, torch.Tensor]] = []
    observed_projections: set[str] = set()
    observed_matrices: set[str] = set()
    for name, parameter in current:
        classification = _classify_allowed_lora(
            name, route_attention=route_attention
        )
        if parameter.requires_grad:
            if classification is None:
                raise DCLRPreferenceTrainingStepError(
                    "model-wide trainability leak outside route-authorized "
                    f"{route_attention} Q/O LoRA: {name}"
                )
            projection, matrix = classification
            observed_projections.add(projection)
            observed_matrices.add(matrix)
            active.append((name, parameter))
        elif parameter.grad is not None:
            raise DCLRPreferenceTrainingStepError(
                f"frozen current parameter has a stale gradient: {name}"
            )
    if not active:
        raise DCLRPreferenceTrainingStepError(
            "route-authorized current LoRA parameter set is empty"
        )
    if observed_projections != {"to_q", "to_out.0"}:
        raise DCLRPreferenceTrainingStepError(
            f"active {route_attention} LoRA must cover both to_q and to_out.0"
        )
    if observed_matrices != {"A", "B"}:
        raise DCLRPreferenceTrainingStepError(
            f"active {route_attention} LoRA must cover both lora_A and lora_B"
        )

    reference_storage = {
        (parameter.device.type, parameter.device.index, parameter.data_ptr())
        for _, parameter in reference
    }
    for name, parameter in active:
        key = (parameter.device.type, parameter.device.index, parameter.data_ptr())
        if key in reference_storage:
            raise DCLRPreferenceTrainingStepError(
                "trainable current LoRA aliases frozen reference storage: "
                f"{name}"
            )
    return current, reference, tuple(active), route_attention


def _model_checkpoint_digest(model: Any, *, label: str) -> str:
    value = getattr(model, CHECKPOINT_DIGEST_ATTRIBUTE, None)
    if value is None:
        owner = _parameter_owner(model, label=label)
        value = getattr(owner, CHECKPOINT_DIGEST_ATTRIBUTE, None)
    return _require_sha256(
        value,
        label=f"{label}.{CHECKPOINT_DIGEST_ATTRIBUTE}",
    )


def _validate_optimizer(optimizer: Any, active: Sequence[tuple[str, torch.Tensor]]) -> None:
    if not callable(getattr(optimizer, "zero_grad", None)) or not callable(
        getattr(optimizer, "step", None)
    ):
        raise DCLRPreferenceTrainingStepError(
            "optimizer must expose callable zero_grad and step"
        )
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or not groups:
        raise DCLRPreferenceTrainingStepError(
            "optimizer must expose nonempty param_groups"
        )
    parameters: list[torch.Tensor] = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping) or "params" not in group:
            raise DCLRPreferenceTrainingStepError(
                f"optimizer param_group {index} is malformed"
            )
        try:
            parameters.extend(list(group["params"]))
        except TypeError as error:
            raise DCLRPreferenceTrainingStepError(
                f"optimizer param_group {index} params are not iterable"
            ) from error
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise DCLRPreferenceTrainingStepError(
            "optimizer repeats a parameter across param_groups"
        )
    expected = {id(parameter) for _, parameter in active}
    observed = {id(parameter) for parameter in parameters}
    if observed != expected:
        raise DCLRPreferenceTrainingStepError(
            "optimizer parameters differ from the route-authorized LoRA set"
        )


def _shared_step_kwargs(
    batch: preference_batch.PackedPreferenceBatch,
    *,
    cond_embeds: torch.Tensor,
    batch_text_seqlen: tuple[int, int],
    model_id: str,
) -> dict[str, Any]:
    kwargs = preference_batch.shared_step_visual_kwargs(batch)
    kwargs.update(
        {
            "model_id": model_id,
            "cond_embeds": cond_embeds,
            "batch_text_seqlen": list(batch_text_seqlen),
        }
    )
    return kwargs


def _call_shared_step(
    model: Any,
    *,
    kwargs: Mapping[str, Any],
    batch: preference_batch.PackedPreferenceBatch,
    label: str,
    require_graph: bool,
) -> torch.Tensor:
    shared_step = getattr(model, "shared_step", None)
    if not callable(shared_step):
        raise DCLRPreferenceTrainingStepError(
            f"{label} lacks callable shared_step"
        )
    try:
        inspect.signature(shared_step).bind(**dict(kwargs))
    except (TypeError, ValueError) as error:
        raise DCLRPreferenceTrainingStepError(
            f"{label} shared_step signature differs from the pinned query"
        ) from error
    if require_graph:
        with torch.enable_grad():
            prediction = shared_step(**dict(kwargs))
    else:
        with torch.no_grad():
            prediction = shared_step(**dict(kwargs))
    output = _require_tensor(prediction, label=f"{label} prediction")
    expected = (
        1,
        batch.total_visual_tokens,
        runtime_contract.PINNED_PATCH_DIM,
    )
    if (
        not output.is_floating_point()
        or tuple(output.shape) != expected
        or output.device != batch.noisy_latents.device
        or not bool(torch.isfinite(output).all().item())
    ):
        raise DCLRPreferenceTrainingStepError(
            f"{label} prediction must be finite [1,{expected[1]},"
            f"{runtime_contract.PINNED_PATCH_DIM}]"
        )
    if require_graph:
        if not output.requires_grad or output.grad_fn is None or output.is_leaf:
            raise DCLRPreferenceTrainingStepError(
                "current prediction must be a non-leaf result of the model graph"
            )
    elif output.requires_grad or output.grad_fn is not None:
        raise DCLRPreferenceTrainingStepError(
            "collection-reference prediction must be detached"
        )
    return output


def _audit_gradients(
    current: Sequence[tuple[str, torch.Tensor]],
    reference: Sequence[tuple[str, torch.Tensor]],
    active: Sequence[tuple[str, torch.Tensor]],
    *,
    route_attention: str,
) -> GradientAudit:
    active_ids = {id(parameter) for _, parameter in active}
    nonzero_names: list[str] = []
    norms: list[float] = []
    frozen_count = 0
    for name, parameter in current:
        if id(parameter) not in active_ids:
            frozen_count += 1
            if parameter.grad is not None:
                raise DCLRPreferenceTrainingStepError(
                    f"forbidden gradient reached frozen current parameter: {name}"
                )
            continue
        gradient = parameter.grad
        if gradient is None:
            raise DCLRPreferenceTrainingStepError(
                f"route-authorized parameter has grad=None: {name}"
            )
        if (
            not isinstance(gradient, torch.Tensor)
            or tuple(gradient.shape) != tuple(parameter.shape)
            or gradient.device != parameter.device
            or not gradient.is_floating_point()
            or not bool(torch.isfinite(gradient).all().item())
        ):
            raise DCLRPreferenceTrainingStepError(
                f"route-authorized parameter has a non-finite/malformed gradient: {name}"
            )
        norm = float(torch.linalg.vector_norm(gradient.detach().float()).item())
        if not math.isfinite(norm) or norm <= 0.0:
            raise DCLRPreferenceTrainingStepError(
                f"route-authorized parameter has a zero gradient: {name}"
            )
        nonzero_names.append(name)
        norms.append(norm)
    if any(parameter.grad is not None for _, parameter in reference):
        raise DCLRPreferenceTrainingStepError(
            "collection-reference model received a gradient"
        )
    trainable_names = tuple(name for name, _ in active)
    if tuple(nonzero_names) != trainable_names:
        raise DCLRPreferenceTrainingStepError(
            "gradient audit did not cover the full trainable allowlist"
        )
    return GradientAudit(
        route_attention=route_attention,
        trainable_parameter_names=trainable_names,
        finite_nonzero_gradient_names=tuple(nonzero_names),
        gradient_l2_norms=tuple(norms),
        frozen_parameter_count=frozen_count,
        frozen_gradients_are_none=True,
        reference_parameter_count=len(reference),
        reference_gradients_are_none=True,
    )


def _float_scalar(value: torch.Tensor, *, label: str) -> float:
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise DCLRPreferenceTrainingStepError(f"{label} must be one tensor scalar")
    result = float(value.detach().float().item())
    if not math.isfinite(result):
        raise DCLRPreferenceTrainingStepError(f"{label} is not finite")
    return result


def run_preference_training_step(
    *,
    current_model: Any,
    collection_reference_model: Any,
    optimizer: Any,
    batch: preference_batch.PackedPreferenceBatch,
    cond_embeds: torch.Tensor,
    batch_text_seqlen: Sequence[int],
    model_id: str,
    pair: Mapping[str, Any],
    receipts_by_digest: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    counterfactual_bank_document: Mapping[str, Any],
    split_ledger: Mapping[str, Any],
    source_manifest_sha256: str,
    artifacts_by_digest: Mapping[str, Mapping[str, Any]],
    current_binding: SharedStepBinding,
    reference_binding: SharedStepBinding,
    beta: float,
) -> PreferenceTrainingStepResult:
    """Execute exactly one reference-corrected packed preference update."""

    validated_pair, winner_receipt, loser_receipt, route = (
        _validated_preference_route(
            pair,
            receipts_by_digest=receipts_by_digest,
            sources=sources,
            counterfactual_bank_document=counterfactual_bank_document,
            split_ledger=split_ledger,
            source_manifest_sha256=source_manifest_sha256,
            artifacts_by_digest=artifacts_by_digest,
        )
    )
    pair_digest = str(validated_pair["pair_digest"])
    checked_batch = preference_batch.validate_packed_preference_batch(batch)
    cond, text_lengths = _validate_text_query(
        cond_embeds, batch_text_seqlen, batch=checked_batch
    )
    model_name = _require_model_id(model_id)
    if (
        isinstance(beta, bool)
        or not isinstance(beta, (int, float))
        or not math.isfinite(float(beta))
        or float(beta) <= 0.0
    ):
        raise DCLRPreferenceTrainingStepError(
            "beta must be one finite positive scalar"
        )
    beta_value = float(beta)
    current_bound = _validate_binding(current_binding, label="current_binding")
    reference_bound = _validate_binding(
        reference_binding, label="reference_binding"
    )
    if current_bound != reference_bound:
        raise DCLRPreferenceTrainingStepError(
            "current/reference checkpoint/query/pair/sigma/epsilon/RoPE bindings differ"
        )
    if current_bound.pair_digest != pair_digest:
        raise DCLRPreferenceTrainingStepError(
            "forward binding pair digest differs from validated preference pair"
        )
    current_checkpoint = _model_checkpoint_digest(
        current_model, label="current model"
    )
    reference_checkpoint = _model_checkpoint_digest(
        collection_reference_model, label="collection reference"
    )
    if (
        current_checkpoint != reference_checkpoint
        or current_checkpoint != current_bound.checkpoint_digest
    ):
        raise DCLRPreferenceTrainingStepError(
            "current/reference common base checkpoint binding differs"
        )
    expected_binding = build_shared_step_binding(
        checkpoint_digest=current_checkpoint,
        pair_digest=pair_digest,
        batch=checked_batch,
        cond_embeds=cond,
        batch_text_seqlen=text_lengths,
        model_id=model_name,
    )
    if current_bound != expected_binding:
        raise DCLRPreferenceTrainingStepError(
            "shared-step binding cannot be recomputed from the actual query"
        )

    current_named, reference_named, active, route_attention = (
        _validate_parameter_surface(
            current_model,
            collection_reference_model,
            route=route,
        )
    )
    _validate_optimizer(optimizer, active)
    try:
        optimizer.zero_grad(set_to_none=True)
    except Exception as error:
        raise DCLRPreferenceTrainingStepError(
            f"optimizer.zero_grad(set_to_none=True) failed: {error}"
        ) from error
    if any(parameter.grad is not None for _, parameter in current_named):
        raise DCLRPreferenceTrainingStepError(
            "optimizer.zero_grad did not establish model-wide grad=None"
        )

    step_performed = False
    try:
        current_prediction = _call_shared_step(
            current_model,
            kwargs=_shared_step_kwargs(
                checked_batch,
                cond_embeds=cond,
                batch_text_seqlen=text_lengths,
                model_id=model_name,
            ),
            batch=checked_batch,
            label="current model",
            require_graph=True,
        )
        # Revalidation after each call catches in-place query mutation by a
        # model wrapper before the other policy can observe a different query.
        if build_shared_step_binding(
            checkpoint_digest=current_checkpoint,
            pair_digest=pair_digest,
            batch=checked_batch,
            cond_embeds=cond,
            batch_text_seqlen=text_lengths,
            model_id=model_name,
        ) != expected_binding:
            raise DCLRPreferenceTrainingStepError(
                "current shared_step mutated the bound query"
            )
        reference_prediction = _call_shared_step(
            collection_reference_model,
            kwargs=_shared_step_kwargs(
                checked_batch,
                cond_embeds=cond,
                batch_text_seqlen=text_lengths,
                model_id=model_name,
            ),
            batch=checked_batch,
            label="collection reference",
            require_graph=False,
        )
        if build_shared_step_binding(
            checkpoint_digest=reference_checkpoint,
            pair_digest=pair_digest,
            batch=checked_batch,
            cond_embeds=cond,
            batch_text_seqlen=text_lengths,
            model_id=model_name,
        ) != expected_binding:
            raise DCLRPreferenceTrainingStepError(
                "collection-reference shared_step mutated the bound query"
            )
        if (
            current_prediction.dtype != reference_prediction.dtype
            or current_prediction.layout != reference_prediction.layout
        ):
            raise DCLRPreferenceTrainingStepError(
                "current/reference prediction representations differ"
            )

        n = checked_batch.source_token_count
        logical_tokens = 2 * n
        current_winner = current_prediction[:, :logical_tokens, :]
        current_loser = current_prediction[:, logical_tokens:, :]
        reference_winner = reference_prediction[:, :logical_tokens, :]
        reference_loser = reference_prediction[:, logical_tokens:, :]
        winner_target = checked_batch.target_true_velocity[:, :n, :]
        loser_target = checked_batch.target_true_velocity[:, n:, :]
        selector = checked_batch.candidate_target_selector
        try:
            winner_energies = (
                preference_objective.candidate_current_reference_target_tail_mse(
                    current_winner,
                    reference_winner,
                    winner_target,
                    selector,
                )
            )
            loser_energies = (
                preference_objective.candidate_current_reference_target_tail_mse(
                    current_loser,
                    reference_loser,
                    loser_target,
                    selector,
                )
            )
            routed = preference_objective.compute_routed_reference_corrected_dpo(
                winner_energies,
                loser_energies,
                beta=beta_value,
                pair_type=str(validated_pair["pair_type"]),
                winner_action_axis_pass=winner_receipt["action_axis_pass"],
                winner_preservation_axis_pass=winner_receipt[
                    "preservation_axis_pass"
                ],
                loser_action_axis_pass=loser_receipt["action_axis_pass"],
                loser_preservation_axis_pass=loser_receipt[
                    "preservation_axis_pass"
                ],
            )
        except preference_objective.DCLRPreferenceObjectiveError as error:
            raise DCLRPreferenceTrainingStepError(str(error)) from error
        if routed.route != route:
            raise DCLRPreferenceTrainingStepError(
                "objective route differs from the pre-forward trainability route"
            )
        routed.objective.loss.backward()
        audit = _audit_gradients(
            current_named,
            reference_named,
            active,
            route_attention=route_attention,
        )

        metrics = {
            "loss": _float_scalar(routed.objective.loss, label="loss"),
            "delta": _float_scalar(routed.objective.delta, label="delta"),
            "current_margin": _float_scalar(
                routed.objective.current_margin, label="current_margin"
            ),
            "reference_margin": _float_scalar(
                routed.objective.reference_margin, label="reference_margin"
            ),
            "winner_current_energy": _float_scalar(
                winner_energies.current, label="winner_current_energy"
            ),
            "winner_reference_energy": _float_scalar(
                winner_energies.reference, label="winner_reference_energy"
            ),
            "loser_current_energy": _float_scalar(
                loser_energies.current, label="loser_current_energy"
            ),
            "loser_reference_energy": _float_scalar(
                loser_energies.reference, label="loser_reference_energy"
            ),
        }
        optimizer.step()
        step_performed = True
        for name, parameter in active:
            if not bool(torch.isfinite(parameter).all().item()):
                raise DCLRPreferenceTrainingStepError(
                    f"optimizer produced a non-finite parameter: {name}"
                )
        if any(parameter.grad is not None for _, parameter in reference_named):
            raise DCLRPreferenceTrainingStepError(
                "reference gradients appeared during optimizer.step"
            )
    except Exception:
        if not step_performed:
            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
        raise

    return PreferenceTrainingStepResult(
        schema_version=SCHEMA_VERSION,
        binding=expected_binding,
        route=route,
        loss=metrics["loss"],
        delta=metrics["delta"],
        current_margin=metrics["current_margin"],
        reference_margin=metrics["reference_margin"],
        winner_current_energy=metrics["winner_current_energy"],
        winner_reference_energy=metrics["winner_reference_energy"],
        loser_current_energy=metrics["loser_current_energy"],
        loser_reference_energy=metrics["loser_reference_energy"],
        gradient_audit=audit,
        current_shared_step_calls=1,
        reference_shared_step_calls=1,
        optimizer_step_performed=True,
    )


__all__ = [
    "CHECKPOINT_DIGEST_ATTRIBUTE",
    "DCLRPreferenceTrainingStepError",
    "GradientAudit",
    "PreferenceTrainingStepResult",
    "SCHEMA_VERSION",
    "SharedStepBinding",
    "build_shared_step_binding",
    "run_preference_training_step",
    "tensor_content_sha256",
]
