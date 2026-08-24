"""Safe batch geometry for editor-to-generator Bernini branches.

The editor action branch consumes Bernini's ordinary packed sequence
``[clean source, noisy target]``.  A generator-native branch must instead
consume only the noisy target sequence.  This module performs that conversion
by taking views of the existing target tail; it never calls the data transform
and never constructs another diffusion sample.

Torch is intentionally imported only inside tensor paths so the structural
contract remains importable in lightweight audit environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


TEXT_FIELDS = ("input_ids", "attention_mask", "t5_input_lens")
_GEOMETRY_FIELDS = frozenset(
    {
        "input_vae_latents",
        "input_vae_rope",
        "vae_latents_mask",
        "vae_seqlen",
    }
)
_TEXT_GEOMETRY_FIELDS = frozenset({"vlm_seqlen", "num_tokens"})
_REQUIRED_FIELDS = frozenset(
    {
        *TEXT_FIELDS,
        *_GEOMETRY_FIELDS,
        *_TEXT_GEOMETRY_FIELDS,
        "timesteps",
        "target_velocity",
        "target_lens",
    }
)


class CrossModeBranchError(RuntimeError):
    """Raised when editor and generator branches do not share one state."""


@dataclass(frozen=True)
class CrossModeBranches:
    """The three text/geometry cells required by the cross-mode objective."""

    editor_action: Mapping[str, Any]
    generator_action: Mapping[str, Any]
    generator_negative: Mapping[str, Any]


def _require_tensor(value: Any, *, label: str, torch: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        raise CrossModeBranchError(f"{label} must be a torch.Tensor")
    return value


def _tensor_exact(left: Any, right: Any, *, torch: Any) -> bool:
    """Return exact tensor equality, including representation metadata."""

    return bool(
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and tuple(left.shape) == tuple(right.shape)
        and left.dtype == right.dtype
        and left.device == right.device
        and left.layout == right.layout
        and torch.equal(left, right)
    )


def _value_exact(left: Any, right: Any, *, torch: Any) -> bool:
    """Fail-closed equality for fields outside the explicit geometry delta."""

    if left is right:
        return True
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return _tensor_exact(left, right, torch=torch)
    try:
        equal = left == right
    except Exception:
        return False
    return equal if isinstance(equal, bool) else False


def _single_integer(tensor: Any, *, label: str, torch: Any) -> int:
    tensor = _require_tensor(tensor, label=label, torch=torch)
    if tensor.numel() != 1:
        raise CrossModeBranchError(f"{label} must contain exactly one value")
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if tensor.dtype not in integer_dtypes:
        raise CrossModeBranchError(f"{label} must have an integer dtype")
    try:
        value = int(tensor.item())
    except (RuntimeError, TypeError, ValueError) as error:
        raise CrossModeBranchError(f"{label} must contain one integer") from error
    if value <= 0:
        raise CrossModeBranchError(f"{label} must be positive")
    return value


def _validate_text_fields(
    fields: Mapping[str, Any], *, label: str, torch: Any
) -> int:
    ids = _require_tensor(fields.get("input_ids"), label=f"{label}.input_ids", torch=torch)
    attention = _require_tensor(
        fields.get("attention_mask"),
        label=f"{label}.attention_mask",
        torch=torch,
    )
    lengths = _require_tensor(
        fields.get("t5_input_lens"),
        label=f"{label}.t5_input_lens",
        torch=torch,
    )
    if (
        ids.ndim != 2
        or int(ids.shape[0]) != 1
        or tuple(attention.shape) != tuple(ids.shape)
    ):
        raise CrossModeBranchError(
            f"{label} text ids and attention must have matching [1,L] geometry"
        )
    length = _single_integer(
        lengths, label=f"{label}.t5_input_lens", torch=torch
    )
    if length <= 0 or length > int(ids.shape[1]):
        raise CrossModeBranchError(
            f"{label}.t5_input_lens must lie inside its token sequence"
        )
    return length


def _storage_data_ptr(tensor: Any) -> int:
    """Return the base-storage pointer across supported PyTorch releases."""

    untyped_storage = getattr(tensor, "untyped_storage", None)
    storage = untyped_storage() if untyped_storage is not None else tensor.storage()
    return int(storage.data_ptr())


def _is_exact_tail_view(
    candidate: Any,
    parent: Any,
    *,
    target_tokens: int,
    torch: Any,
) -> bool:
    """Require the exact target slice, including storage, offset, and stride."""

    expected = parent[target_tokens:]
    if not _tensor_exact(candidate, expected, torch=torch):
        return False
    return bool(
        tuple(candidate.stride()) == tuple(expected.stride())
        and int(candidate.storage_offset()) == int(expected.storage_offset())
        and _storage_data_ptr(candidate) == _storage_data_ptr(parent)
    )


def _validate_editor_action(
    editor_action: Mapping[str, Any], *, torch: Any
) -> tuple[int, int]:
    if not isinstance(editor_action, Mapping):
        raise CrossModeBranchError("editor_action must be a mapping")
    missing = sorted(_REQUIRED_FIELDS - editor_action.keys())
    if missing:
        raise CrossModeBranchError(f"editor_action is missing fields: {missing}")

    mask = _require_tensor(
        editor_action["vae_latents_mask"],
        label="editor_action.vae_latents_mask",
        torch=torch,
    )
    if mask.ndim != 2 or int(mask.shape[0]) != 1 or mask.dtype != torch.bool:
        raise CrossModeBranchError(
            "editor_action.vae_latents_mask must be bool [1,2N]"
        )
    total = int(mask.shape[1])
    if total <= 0 or total % 2:
        raise CrossModeBranchError("editor_action must contain source N + target N")
    target_tokens = total // 2
    if bool(mask[:, :target_tokens].any()) or not bool(
        mask[:, target_tokens:].all()
    ):
        raise CrossModeBranchError(
            "editor_action mask must select one contiguous target tail after source N"
        )

    latents = _require_tensor(
        editor_action["input_vae_latents"],
        label="editor_action.input_vae_latents",
        torch=torch,
    )
    rope = _require_tensor(
        editor_action["input_vae_rope"],
        label="editor_action.input_vae_rope",
        torch=torch,
    )
    velocity = _require_tensor(
        editor_action["target_velocity"],
        label="editor_action.target_velocity",
        torch=torch,
    )
    _require_tensor(
        editor_action["timesteps"], label="editor_action.timesteps", torch=torch
    )
    if latents.ndim < 1 or rope.ndim < 1 or velocity.ndim < 1:
        raise CrossModeBranchError(
            "editor_action latent, rope, and velocity tensors need a token axis"
        )
    if int(latents.shape[0]) != total or int(rope.shape[0]) != total:
        raise CrossModeBranchError(
            "editor_action latent/rope length must equal source N + target N"
        )
    if int(velocity.shape[0]) != target_tokens:
        raise CrossModeBranchError(
            "editor_action.target_velocity must contain exactly target N tokens"
        )
    if (
        _single_integer(
            editor_action["vae_seqlen"],
            label="editor_action.vae_seqlen",
            torch=torch,
        )
        != total
    ):
        raise CrossModeBranchError("editor_action.vae_seqlen must equal 2N")
    if (
        _single_integer(
            editor_action["target_lens"],
            label="editor_action.target_lens",
            torch=torch,
        )
        != target_tokens
    ):
        raise CrossModeBranchError("editor_action.target_lens must equal N")
    action_text_length = _validate_text_fields(
        editor_action, label="editor_action", torch=torch
    )
    if (
        _single_integer(
            editor_action["vlm_seqlen"],
            label="editor_action.vlm_seqlen",
            torch=torch,
        )
        != action_text_length
    ):
        raise CrossModeBranchError(
            "editor_action.vlm_seqlen must equal its action text length"
        )
    if (
        _single_integer(
            editor_action["num_tokens"],
            label="editor_action.num_tokens",
            torch=torch,
        )
        != total + action_text_length
    ):
        raise CrossModeBranchError(
            "editor_action.num_tokens must equal vae_seqlen + vlm_seqlen"
        )
    return target_tokens, action_text_length


def _validate_generator_geometry(
    branch: Mapping[str, Any],
    *,
    label: str,
    target_tokens: int,
    text_length: int,
    torch: Any,
) -> None:
    mask = _require_tensor(
        branch.get("vae_latents_mask"),
        label=f"{label}.vae_latents_mask",
        torch=torch,
    )
    latents = _require_tensor(
        branch.get("input_vae_latents"),
        label=f"{label}.input_vae_latents",
        torch=torch,
    )
    rope = _require_tensor(
        branch.get("input_vae_rope"),
        label=f"{label}.input_vae_rope",
        torch=torch,
    )
    if (
        mask.dtype != torch.bool
        or tuple(mask.shape) != (1, target_tokens)
        or not bool(mask.all())
    ):
        raise CrossModeBranchError(f"{label} mask must be all-True [1,N]")
    if latents.ndim < 1 or rope.ndim < 1:
        raise CrossModeBranchError(f"{label} latent and rope need a token axis")
    if int(latents.shape[0]) != target_tokens or int(rope.shape[0]) != target_tokens:
        raise CrossModeBranchError(f"{label} latent/rope length must equal N")
    if (
        _single_integer(
            branch.get("vae_seqlen"), label=f"{label}.vae_seqlen", torch=torch
        )
        != target_tokens
    ):
        raise CrossModeBranchError(f"{label}.vae_seqlen must equal N")
    if (
        _single_integer(
            branch.get("vlm_seqlen"),
            label=f"{label}.vlm_seqlen",
            torch=torch,
        )
        != text_length
    ):
        raise CrossModeBranchError(
            f"{label}.vlm_seqlen must equal its text length"
        )
    if (
        _single_integer(
            branch.get("num_tokens"),
            label=f"{label}.num_tokens",
            torch=torch,
        )
        != target_tokens + text_length
    ):
        raise CrossModeBranchError(
            f"{label}.num_tokens must equal vae_seqlen + vlm_seqlen"
        )


def validate_cross_mode_branches(
    editor_action: Mapping[str, Any],
    generator_action: Mapping[str, Any],
    generator_negative: Mapping[str, Any],
    *,
    generator_action_text_fields: Mapping[str, Any],
    generator_negative_text_fields: Mapping[str, Any],
) -> None:
    """Validate exact shared state across editor and target-only branches.

    The editor and generator sequence lengths intentionally differ.  Equality
    is therefore checked against the editor's target tail rather than against
    its full packed sequence.  Every diffusion-state field is exact: no
    tolerance-based comparison is used.
    """

    import torch

    target_tokens, _ = _validate_editor_action(editor_action, torch=torch)
    for label, branch in (
        ("generator_action", generator_action),
        ("generator_negative", generator_negative),
    ):
        if not isinstance(branch, Mapping):
            raise CrossModeBranchError(f"{label} must be a mapping")
        if set(branch) != set(editor_action):
            raise CrossModeBranchError(
                f"{label} keys must exactly match editor_action keys"
            )
        text_length = _validate_text_fields(branch, label=label, torch=torch)
        _validate_generator_geometry(
            branch,
            label=label,
            target_tokens=target_tokens,
            text_length=text_length,
            torch=torch,
        )

        if not _is_exact_tail_view(
            branch["input_vae_latents"],
            editor_action["input_vae_latents"],
            target_tokens=target_tokens,
            torch=torch,
        ):
            raise CrossModeBranchError(
                f"{label} noisy state differs from editor target tail "
                "or is not its direct storage view"
            )
        if not _is_exact_tail_view(
            branch["input_vae_rope"],
            editor_action["input_vae_rope"],
            target_tokens=target_tokens,
            torch=torch,
        ):
            raise CrossModeBranchError(
                f"{label} rope differs from editor target tail "
                "or is not its direct storage view"
            )
        for field in ("timesteps", "target_velocity", "target_lens"):
            if not _tensor_exact(branch[field], editor_action[field], torch=torch):
                raise CrossModeBranchError(
                    f"{label}.{field} differs from editor target state"
                )

    for label, supplied, branch_label, branch in (
        (
            "generator_action_text_fields",
            generator_action_text_fields,
            "generator_action",
            generator_action,
        ),
        (
            "generator_negative_text_fields",
            generator_negative_text_fields,
            "generator_negative",
            generator_negative,
        ),
    ):
        if not isinstance(supplied, Mapping) or set(supplied) != set(TEXT_FIELDS):
            raise CrossModeBranchError(
                f"{label} must contain exactly the three T5 text fields"
            )
        _validate_text_fields(supplied, label=label, torch=torch)
        for field in TEXT_FIELDS:
            if not _tensor_exact(branch[field], supplied[field], torch=torch):
                raise CrossModeBranchError(
                    f"{branch_label}.{field} differs from supplied text"
                )

    if _tensor_exact(
        generator_action["input_ids"], editor_action["input_ids"], torch=torch
    ):
        raise CrossModeBranchError(
            "generator action must use distinct official T2V text, not editor MV2V text"
        )
    if _tensor_exact(
        generator_negative["input_ids"],
        generator_action["input_ids"],
        torch=torch,
    ):
        raise CrossModeBranchError(
            "generator negative and T2V action text must be distinct"
        )

    for field in editor_action:
        if (
            field in _GEOMETRY_FIELDS
            or field in _TEXT_GEOMETRY_FIELDS
            or field in TEXT_FIELDS
        ):
            continue
        if not _value_exact(
            generator_action[field], editor_action[field], torch=torch
        ):
            raise CrossModeBranchError(
                f"generator_action changed non-geometry field {field}"
            )

    for field in generator_action:
        if field in TEXT_FIELDS or field in _TEXT_GEOMETRY_FIELDS:
            continue
        if not _value_exact(
            generator_negative[field], generator_action[field], torch=torch
        ):
            raise CrossModeBranchError(
                f"generator_negative changed non-text field {field}"
            )

def build_generator_branches(
    editor_action: Mapping[str, Any],
    generator_action_text_fields: Mapping[str, Any],
    generator_negative_text_fields: Mapping[str, Any],
) -> CrossModeBranches:
    """Derive action/negative generator cells from one editor action cell.

    ``input_vae_latents`` and ``input_vae_rope`` are direct target-tail views.
    The timestep, velocity target, target length, and all other untouched
    fields are shared with the editor batch.  The negative branch is a shallow
    copy of the generator action branch with only the three T5 fields and their
    derived ``vlm_seqlen``/``num_tokens`` geometry replaced.  Generator action
    text is supplied separately because its official T2V system prompt differs
    from the editor branch's MV2V system prompt.
    """

    import torch

    target_tokens, _ = _validate_editor_action(
        editor_action, torch=torch
    )
    for label, supplied in (
        ("generator_action_text_fields", generator_action_text_fields),
        ("generator_negative_text_fields", generator_negative_text_fields),
    ):
        if not isinstance(supplied, Mapping) or set(supplied) != set(TEXT_FIELDS):
            raise CrossModeBranchError(
                f"{label} must contain exactly the three T5 text fields"
            )
    action_text_length = _validate_text_fields(
        generator_action_text_fields,
        label="generator_action_text_fields",
        torch=torch,
    )
    negative_text_length = _validate_text_fields(
        generator_negative_text_fields,
        label="generator_negative_text_fields",
        torch=torch,
    )
    if _tensor_exact(
        generator_action_text_fields["input_ids"],
        editor_action["input_ids"],
        torch=torch,
    ):
        raise CrossModeBranchError(
            "generator action must use distinct official T2V text, not editor MV2V text"
        )
    if _tensor_exact(
        generator_negative_text_fields["input_ids"],
        generator_action_text_fields["input_ids"],
        torch=torch,
    ):
        raise CrossModeBranchError(
            "generator negative and T2V action text must be distinct"
        )

    generator_action = dict(editor_action)
    generator_action["input_vae_latents"] = editor_action["input_vae_latents"][
        target_tokens:
    ]
    generator_action["input_vae_rope"] = editor_action["input_vae_rope"][
        target_tokens:
    ]
    generator_action["vae_latents_mask"] = torch.ones_like(
        editor_action["vae_latents_mask"][:, target_tokens:], dtype=torch.bool
    )
    generator_action["vae_seqlen"] = torch.full_like(
        editor_action["vae_seqlen"], target_tokens
    )
    generator_action["vlm_seqlen"] = torch.full_like(
        editor_action["vlm_seqlen"], action_text_length
    )
    generator_action["num_tokens"] = torch.full_like(
        editor_action["num_tokens"], target_tokens + action_text_length
    )
    for field in TEXT_FIELDS:
        generator_action[field] = generator_action_text_fields[field]

    generator_negative = dict(generator_action)
    for field in TEXT_FIELDS:
        generator_negative[field] = generator_negative_text_fields[field]
    generator_negative["vlm_seqlen"] = torch.full_like(
        generator_action["vlm_seqlen"], negative_text_length
    )
    generator_negative["num_tokens"] = torch.full_like(
        generator_action["num_tokens"], target_tokens + negative_text_length
    )

    result = CrossModeBranches(
        editor_action=editor_action,
        generator_action=generator_action,
        generator_negative=generator_negative,
    )
    validate_cross_mode_branches(
        result.editor_action,
        result.generator_action,
        result.generator_negative,
        generator_action_text_fields=generator_action_text_fields,
        generator_negative_text_fields=generator_negative_text_fields,
    )
    return result


__all__ = [
    "CrossModeBranchError",
    "CrossModeBranches",
    "TEXT_FIELDS",
    "build_generator_branches",
    "validate_cross_mode_branches",
]
