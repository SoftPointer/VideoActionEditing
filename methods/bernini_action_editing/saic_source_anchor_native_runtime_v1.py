#!/usr/bin/env python3
"""Deploy the SAIC late source anchor in stock Bernini ``sample``.

The Stage-A trainer evaluates hand-built :class:`NativeRV2VBranch` objects.
Stock ``GEN_Wanx22.sample`` instead exposes only the already-packed V/VI
tensor at its ``shared_step`` boundary.  This module closes that deployment
seam without replacing the sampler: it authenticates one exact81/exact40
``v2v_apg`` call, reconstructs an immutable V or VI branch descriptor from
the *actual* shared-step tensors, and activates the existing late anchor
around each official negative and action forward.

Target-only teacher queries are deliberately passed through with no anchor
route.  This matters when a later action-field wrapper is installed outside
this wrapper: teacher/no-op/action T2V queries have no source prefix, while
the two stock RV2V calls do.  The source video, four optional reference rows,
and Ulysses append padding remain base-model bytes because the adapter derives
its selector from the native suffix and the live SP state.

The module also provides a strict loader for the closed Stage-A safetensors
artifact.  It binds the whole file hash, the complete metadata map, the exact
adapter key set, FP32 tensor geometry, and the adapter contract before any
parameter is changed.  A failed copy rolls back to the pre-load state.

Installation order is intentional: install this wrapper directly on the
unmodified diffusion instance, then install any target-only action/teacher
wrapper outside it.  No mask, pose, flow, track, trajectory, target video, or
proposal media is accepted or created here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import torch

if __package__:
    from . import inference_sigma_strata as sigma_strata
    from . import saic_source_anchor_adapter_v1 as source_anchor
    from . import source_self_native_ref_contrastive_v3 as native_pack
else:
    import inference_sigma_strata as sigma_strata
    import saic_source_anchor_adapter_v1 as source_anchor
    import source_self_native_ref_contrastive_v3 as native_pack


SCHEMA_VERSION = "bernini-saic-source-anchor-native-runtime-v1"
SAFETENSORS_SCHEMA_VERSION = "bernini-saic-source-anchor-safetensors-v1"
EXPECTED_FRAMES = 81
EXPECTED_LATENT_PHASES = 21
EXPECTED_STEPS = 40
EXPECTED_FLOW_SHIFT = 5.0
EXPECTED_GUIDANCE_MODE = "v2v_apg"
EXPECTED_MODEL_ID = "transformer_1"
EXPECTED_HIDDEN_DIM = 1536
EXPECTED_TEXT_TOKENS = 512
EXPECTED_TEXT_DIM = 4096
EXPECTED_OUTPUT_CHANNELS = 64
PINNED_DIFFUSION_CLASS = ("bernini.models.wan_diffusion", "GEN_Wanx22")
PINNED_SCHEDULER_CLASS = (
    "diffusers.schedulers.scheduling_unipc_multistep",
    "UniPCMultistepScheduler",
)
FULL_FORWARD_ORDER = ("negative", "action")
FORMAL_OPTIMIZER_UPDATES = 32
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MARKER = "_bernini_saic_source_anchor_native_runtime_v1"
_SAFETENSORS_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "adapter_schema_version",
        "adapter_contract_digest",
        "state_tensor_sha256",
        "state_key_sha256",
        "optimizer_updates",
        "heldout_gate_digest",
        "source_anchor_only",
        "semantic_action_success",
    }
)
_FORBIDDEN_SAMPLE_TOKENS = (
    "mask",
    "pose",
    "track",
    "trajectory",
    "swept_tube",
    "optical_flow",
    "flow_field",
    "target_video",
    "proposal_video",
    "donor_video",
)


class SAICSourceAnchorNativeRuntimeError(RuntimeError):
    """Raised before an unauthenticated checkpoint or native call can run."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SAICSourceAnchorNativeRuntimeError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise SAICSourceAnchorNativeRuntimeError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _plain_canonical_file(path: os.PathLike[str] | str, *, label: str) -> Path:
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        raise SAICSourceAnchorNativeRuntimeError(f"{label} must be absolute")
    try:
        before = requested.lstat()
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise SAICSourceAnchorNativeRuntimeError(
            f"cannot stat {label}: {requested}"
        ) from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or resolved != requested
    ):
        raise SAICSourceAnchorNativeRuntimeError(
            f"{label} must be one canonical plain regular file"
        )
    return requested


def _stable_file_sha256(path: Path, *, label: str) -> str:
    before = path.lstat()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SAICSourceAnchorNativeRuntimeError(
            f"cannot open {label}: {path}"
        ) from error
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (before.st_dev, before.st_ino):
            raise SAICSourceAnchorNativeRuntimeError(
                f"{label} changed while opening"
            )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named = path.lstat()
    identity = lambda item: (  # noqa: E731
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
    )
    if (
        stat.S_ISLNK(named.st_mode)
        or identity(before) != identity(opened)
        or identity(opened) != identity(after)
        or identity(after) != identity(named)
    ):
        raise SAICSourceAnchorNativeRuntimeError(
            f"{label} changed while hashing"
        )
    return digest.hexdigest()


def _validate_handle(
    handle: source_anchor.SAICSourceAnchorHandle,
    *,
    transformer: Optional[Any] = None,
) -> Mapping[str, Any]:
    if type(handle) is not source_anchor.SAICSourceAnchorHandle:
        raise SAICSourceAnchorNativeRuntimeError(
            "source anchor handle must have the exact registered type"
        )
    if bool(getattr(handle, "restored", True)):
        raise SAICSourceAnchorNativeRuntimeError("source anchor handle is restored")
    if transformer is not None and handle.transformer is not transformer:
        raise SAICSourceAnchorNativeRuntimeError(
            "source anchor handle owns a different transformer"
        )
    try:
        receipt = handle.receipt()
    except Exception as error:
        raise SAICSourceAnchorNativeRuntimeError(
            f"cannot audit source anchor handle: {error}"
        ) from error
    required = {
        "schema_version": source_anchor.SCHEMA_VERSION,
        "blocks": list(source_anchor.SOURCE_ANCHOR_BLOCK_INDICES),
        "projections": ["attn1.to_q", "attn1.to_out.0"],
        "rank": source_anchor.SOURCE_ANCHOR_RANK,
        "full_source_native_branches": list(source_anchor.FULL_SOURCE_BRANCHES),
        "active_sigma_indices": list(source_anchor.ACTIVE_SIGMA_INDICES),
        "exact40_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        "source_reference_padding_rows_exact_base": True,
        "prompt_role_agnostic_action_and_noop": True,
        "route_accepts_caller_rank_size_index_or_mask": False,
        "route_binds_live_parallel_native_mask_and_actual_scheduler_sigma": True,
        "accepted_timestep_representations": [
            "official_device_local_int64",
            "manual_device_local_float32",
        ],
        "only_registered_self_attention_qo_replaced": True,
        "base_parameters_frozen": True,
    }
    if not isinstance(receipt, Mapping) or any(
        receipt.get(name) != expected for name, expected in required.items()
    ):
        raise SAICSourceAnchorNativeRuntimeError(
            "source anchor handle contract differs"
        )
    _require_sha256(receipt.get("digest"), label="adapter contract digest")
    return receipt


def _validate_safetensors_metadata(
    metadata: Mapping[str, str],
    *,
    adapter_contract_digest: str,
) -> None:
    if (
        not isinstance(metadata, Mapping)
        or set(metadata) != set(_SAFETENSORS_METADATA_FIELDS)
        or any(
            type(key) is not str or type(value) is not str
            for key, value in metadata.items()
        )
    ):
        raise SAICSourceAnchorNativeRuntimeError(
            "source-anchor safetensors metadata closure differs"
        )
    expected_literals = {
        "schema_version": SAFETENSORS_SCHEMA_VERSION,
        "adapter_schema_version": source_anchor.SCHEMA_VERSION,
        "adapter_contract_digest": adapter_contract_digest,
        "optimizer_updates": str(FORMAL_OPTIMIZER_UPDATES),
        "source_anchor_only": "true",
        "semantic_action_success": "false",
    }
    if any(metadata.get(name) != value for name, value in expected_literals.items()):
        raise SAICSourceAnchorNativeRuntimeError(
            "source-anchor safetensors metadata values differ"
        )
    for name in ("state_tensor_sha256", "state_key_sha256", "heldout_gate_digest"):
        _require_sha256(metadata.get(name), label=f"metadata {name}")


def load_saic_source_anchor_safetensors(
    handle: source_anchor.SAICSourceAnchorHandle,
    path: os.PathLike[str] | str,
    *,
    expected_file_sha256: str,
    expected_metadata: Mapping[str, str],
) -> Mapping[str, Any]:
    """Strictly and transactionally load one formally published Stage-A adapter."""

    receipt = _validate_handle(handle)
    source = _plain_canonical_file(path, label="source-anchor safetensors")
    expected_digest = _require_sha256(
        expected_file_sha256, label="expected safetensors file digest"
    )
    adapter_contract_digest = str(receipt["digest"])
    _validate_safetensors_metadata(
        expected_metadata, adapter_contract_digest=adapter_contract_digest
    )
    first_digest = _stable_file_sha256(source, label="source-anchor safetensors")
    if first_digest != expected_digest:
        raise SAICSourceAnchorNativeRuntimeError(
            "source-anchor safetensors file digest differs"
        )
    try:
        from safetensors import safe_open
    except Exception as error:
        raise SAICSourceAnchorNativeRuntimeError(
            "safetensors runtime is unavailable"
        ) from error

    expected_parameters = dict(handle.trainable_named_parameters())
    if not expected_parameters:
        raise SAICSourceAnchorNativeRuntimeError(
            "source-anchor trainable key set is empty"
        )
    loaded: dict[str, torch.Tensor] = {}
    try:
        with safe_open(str(source), framework="pt", device="cpu") as opened:
            observed_metadata = opened.metadata()
            observed_keys = tuple(opened.keys())
            if len(observed_keys) != len(set(observed_keys)):
                raise SAICSourceAnchorNativeRuntimeError(
                    "source-anchor safetensors contains duplicate keys"
                )
            if observed_metadata != dict(expected_metadata):
                raise SAICSourceAnchorNativeRuntimeError(
                    "source-anchor safetensors metadata differs from registration"
                )
            if set(observed_keys) != set(expected_parameters):
                raise SAICSourceAnchorNativeRuntimeError(
                    "source-anchor safetensors tensor-key closure differs"
                )
            for name in observed_keys:
                value = opened.get_tensor(name)
                parameter = expected_parameters[name]
                if (
                    type(value) is not torch.Tensor
                    or value.device.type != "cpu"
                    or value.dtype != torch.float32
                    or value.layout != torch.strided
                    or value.requires_grad
                    or value.grad_fn is not None
                    or not value.is_contiguous()
                    or tuple(value.shape) != tuple(parameter.shape)
                    or not bool(torch.isfinite(value).all().item())
                ):
                    raise SAICSourceAnchorNativeRuntimeError(
                        f"source-anchor tensor {name} differs"
                    )
                loaded[name] = value
    except SAICSourceAnchorNativeRuntimeError:
        raise
    except Exception as error:
        raise SAICSourceAnchorNativeRuntimeError(
            f"cannot read source-anchor safetensors: {error}"
        ) from error

    second_digest = _stable_file_sha256(source, label="source-anchor safetensors")
    if second_digest != first_digest:
        raise SAICSourceAnchorNativeRuntimeError(
            "source-anchor safetensors changed while loading"
        )
    state_digest = source_anchor.trainable_state_digest(loaded)
    if state_digest != expected_metadata["state_tensor_sha256"]:
        raise SAICSourceAnchorNativeRuntimeError(
            "source-anchor safetensors state digest differs"
        )
    key_digest = _object_sha256(sorted(loaded))
    if key_digest != expected_metadata["state_key_sha256"]:
        raise SAICSourceAnchorNativeRuntimeError(
            "source-anchor safetensors key digest differs"
        )

    previous = dict(handle.state_dict_for_save())
    try:
        load_receipt = handle.load_trainable_state_dict(loaded)
        observed = dict(handle.state_dict_for_save())
        if set(observed) != set(loaded) or any(
            not torch.equal(observed[name], loaded[name]) for name in loaded
        ):
            raise SAICSourceAnchorNativeRuntimeError(
                "source-anchor loaded state differs from safetensors bytes"
            )
    except Exception as error:
        try:
            handle.load_trainable_state_dict(previous)
        except Exception as rollback_error:  # pragma: no cover - catastrophic failure
            raise SAICSourceAnchorNativeRuntimeError(
                "source-anchor load failed and rollback failed"
            ) from rollback_error
        if isinstance(error, SAICSourceAnchorNativeRuntimeError):
            raise
        raise SAICSourceAnchorNativeRuntimeError(
            f"source-anchor state load failed: {error}"
        ) from error

    value = {
        "schema_version": SAFETENSORS_SCHEMA_VERSION,
        "path": str(source),
        "file_sha256": first_digest,
        "adapter_contract_digest": adapter_contract_digest,
        "state_key_count": len(loaded),
        "state_key_sha256": key_digest,
        "state_tensor_sha256": state_digest,
        "optimizer_updates": FORMAL_OPTIMIZER_UPDATES,
        "heldout_gate_digest": expected_metadata["heldout_gate_digest"],
        "metadata_exact_registration_match": True,
        "transactional_rollback_armed": True,
        "load_receipt_digest": load_receipt["digest"],
        "semantic_action_success_claim": False,
    }
    return {**value, "digest": _object_sha256(value)}


def _resolve_diffusion_core(renderer_or_diffusion: Any) -> Any:
    queue = [renderer_or_diffusion]
    seen: set[int] = set()
    matches: dict[int, Any] = {}
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if all(
            callable(getattr(candidate, name, None))
            for name in ("sample", "shared_step")
        ) and callable(getattr(getattr(candidate, "scheduler", None), "step", None)):
            matches[id(candidate)] = candidate
        nested = getattr(candidate, "diff_dec", None)
        if nested is not None:
            queue.append(nested)
        getter = getattr(candidate, "get_base_model", None)
        if callable(getter):
            try:
                queue.append(getter())
            except Exception:
                pass
        for name in ("base_model", "model", "module"):
            nested = getattr(candidate, name, None)
            if nested is not None:
                queue.append(nested)
    if len(matches) != 1:
        raise SAICSourceAnchorNativeRuntimeError(
            "expected exactly one GEN_Wanx22-compatible diffusion core"
        )
    return next(iter(matches.values()))


def _flatten_bound_arguments(
    callable_object: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_object)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
    except (TypeError, ValueError) as error:
        raise SAICSourceAnchorNativeRuntimeError(
            "call does not match the pinned Bernini signature"
        ) from error
    values = dict(bound.arguments)
    for parameter in signature.parameters.values():
        if parameter.kind is not parameter.VAR_KEYWORD:
            continue
        extras = values.pop(parameter.name, {})
        if not isinstance(extras, Mapping):
            raise SAICSourceAnchorNativeRuntimeError(
                "variadic keyword payload differs"
            )
        for name, value in extras.items():
            if name in values:
                raise SAICSourceAnchorNativeRuntimeError(
                    f"duplicate variadic keyword {name}"
                )
            values[name] = value
    return values


def _extract_argument(
    args: Sequence[Any], kwargs: Mapping[str, Any], *, index: int, name: str
) -> Any:
    if len(args) > index and name in kwargs:
        raise SAICSourceAnchorNativeRuntimeError(f"call received duplicate {name}")
    if len(args) > index:
        return args[index]
    if name in kwargs:
        return kwargs[name]
    raise SAICSourceAnchorNativeRuntimeError(f"call is missing {name}")


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.shape)
    except Exception as error:
        raise SAICSourceAnchorNativeRuntimeError(
            f"{label} must expose an integer shape"
        ) from error


def _detached_finite_tensor(value: Any, *, label: str) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.layout != torch.strided
        or value.requires_grad
        or value.grad_fn is not None
        or value.numel() <= 0
        or not bool(torch.isfinite(value).all().item())
    ):
        raise SAICSourceAnchorNativeRuntimeError(
            f"{label} must be one detached finite strided tensor"
        )
    return value


def _scalar(value: Any, *, label: str) -> float:
    try:
        candidate = value.detach() if hasattr(value, "detach") else value
        if hasattr(candidate, "numel") and int(candidate.numel()) != 1:
            raise SAICSourceAnchorNativeRuntimeError(f"{label} must be scalar")
        if hasattr(candidate, "cpu"):
            candidate = candidate.cpu()
        if hasattr(candidate, "item"):
            candidate = candidate.item()
        result = float(candidate)
    except SAICSourceAnchorNativeRuntimeError:
        raise
    except Exception as error:
        raise SAICSourceAnchorNativeRuntimeError(f"{label} must be scalar") from error
    if not math.isfinite(result):
        raise SAICSourceAnchorNativeRuntimeError(f"{label} must be finite")
    return result


def _metadata_tuple(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or any(
        type(item) is not int or item <= 0 for item in value
    ):
        raise SAICSourceAnchorNativeRuntimeError(
            f"{label} must be positive integer list/tuple metadata"
        )
    return tuple(value)


def _same_object(left: Any, right: Any, *, label: str) -> None:
    if left is not right:
        raise SAICSourceAnchorNativeRuntimeError(
            f"{label} must be the exact same sampler object"
        )


def _storage_pointer(value: torch.Tensor) -> int:
    try:
        return int(value.untyped_storage().data_ptr())
    except AttributeError:  # pragma: no cover - older torch compatibility
        return int(value.storage().data_ptr())


def _certify_exact_suffix_view(
    full: torch.Tensor,
    suffix: torch.Tensor,
    *,
    token_dim: int,
    suffix_tokens: int,
    label: str,
) -> None:
    slices = [slice(None)] * full.ndim
    slices[token_dim] = slice(int(full.shape[token_dim]) - suffix_tokens, None)
    expected = full[tuple(slices)]
    if not (
        type(suffix) is torch.Tensor
        and tuple(suffix.shape) == tuple(expected.shape)
        and suffix.dtype == full.dtype
        and suffix.device == full.device
        and suffix.layout == full.layout
        and _storage_pointer(suffix) == _storage_pointer(full)
        and int(suffix.storage_offset()) == int(expected.storage_offset())
        and tuple(suffix.stride()) == tuple(expected.stride())
        and int(suffix.data_ptr()) == int(expected.data_ptr())
    ):
        raise SAICSourceAnchorNativeRuntimeError(
            f"{label} is not the exact full-pack target suffix view"
        )


def _certify_expanded_timestep(
    shared: torch.Tensor, scheduler_scalar: torch.Tensor
) -> None:
    if (
        type(shared) is not torch.Tensor
        or type(scheduler_scalar) is not torch.Tensor
        or tuple(shared.shape) != (1,)
        or scheduler_scalar.ndim != 0
        or shared.dtype != scheduler_scalar.dtype
        or shared.device != scheduler_scalar.device
        or tuple(shared.stride()) != (0,)
        or _storage_pointer(shared) != _storage_pointer(scheduler_scalar)
        or int(shared.storage_offset()) != int(scheduler_scalar.storage_offset())
        or int(shared.data_ptr()) != int(scheduler_scalar.data_ptr())
        or not torch.equal(shared.reshape(()), scheduler_scalar)
    ):
        raise SAICSourceAnchorNativeRuntimeError(
            "model timestep is not scheduler scalar t.expand(1)"
        )


@dataclass(frozen=True)
class SourceAnchorNativeRuntimeConfig:
    """Pinned geometry for one stock full-source exact81 render."""

    target_latent_shape: tuple[int, int, int, int, int]
    branch_name: str
    expected_steps: int = EXPECTED_STEPS

    @property
    def target_tokens(self) -> int:
        _, _, phases, height, width = self.target_latent_shape
        return phases * (height // 2) * (width // 2)

    @property
    def reference_tokens(self) -> int:
        return self.target_tokens // EXPECTED_LATENT_PHASES

    @property
    def condition_tokens(self) -> int:
        if self.branch_name == "V":
            return self.target_tokens
        return self.target_tokens + native_pack.REFERENCE_COUNT * self.reference_tokens

    @property
    def total_tokens(self) -> int:
        return self.condition_tokens + self.target_tokens

    def validate(self) -> None:
        shape = tuple(self.target_latent_shape)
        if (
            len(shape) != 5
            or any(type(value) is not int or value <= 0 for value in shape)
            or shape[0] != 1
            or shape[1] != 16
            or shape[2] != EXPECTED_LATENT_PHASES
            or shape[3] % 2
            or shape[4] % 2
        ):
            raise SAICSourceAnchorNativeRuntimeError(
                "target latent must be exact81 [1,16,21,even,even]"
            )
        if self.branch_name not in source_anchor.FULL_SOURCE_BRANCHES:
            raise SAICSourceAnchorNativeRuntimeError(
                "late source anchor supports only full-source V or VI"
            )
        if self.expected_steps != EXPECTED_STEPS:
            raise SAICSourceAnchorNativeRuntimeError(
                "late source anchor deployment requires exact40"
            )


@dataclass(frozen=True)
class _ForwardObservation:
    role: str
    noisy_latents: torch.Tensor = field(repr=False)
    timesteps: torch.Tensor = field(repr=False)
    rotary_embs: torch.Tensor = field(repr=False)


@dataclass
class _ActiveSample:
    action_prompt: torch.Tensor = field(repr=False)
    negative_prompt: torch.Tensor = field(repr=False)
    completed_steps: int = 0
    pending_full: list[_ForwardObservation] = field(default_factory=list)
    pending_teacher_roles: list[str] = field(default_factory=list)


class SAICSourceAnchorNativeRuntimePatch:
    """Reversible official-sampler wrapper for the existing late anchor."""

    def __init__(
        self,
        renderer_or_diffusion: Any,
        *,
        handle: source_anchor.SAICSourceAnchorHandle,
        config: SourceAnchorNativeRuntimeConfig,
    ) -> None:
        config.validate()
        diffusion = _resolve_diffusion_core(renderer_or_diffusion)
        transformer = getattr(diffusion, "transformer", None)
        scheduler = getattr(diffusion, "scheduler", None)
        if (
            type(diffusion).__module__,
            type(diffusion).__name__,
        ) != PINNED_DIFFUSION_CLASS:
            raise SAICSourceAnchorNativeRuntimeError(
                "diffusion class is not pinned bernini.models.wan_diffusion.GEN_Wanx22"
            )
        if (
            type(scheduler).__module__,
            type(scheduler).__name__,
        ) != PINNED_SCHEDULER_CLASS:
            raise SAICSourceAnchorNativeRuntimeError(
                "scheduler class is not the pinned Diffusers UniPC"
            )
        if (
            getattr(diffusion, "use_unipc", None) is not True
            or getattr(diffusion, "transformer_2", None) is not None
            or _scalar(
                getattr(diffusion, "switch_dit_boundary", None),
                label="switch_dit_boundary",
            )
            != 0.0
        ):
            raise SAICSourceAnchorNativeRuntimeError(
                "runtime requires single-expert Bernini-R UniPC"
            )
        handle_receipt = _validate_handle(handle, transformer=transformer)
        transformer_config = getattr(transformer, "config", None)

        def config_value(name: str) -> Any:
            if isinstance(transformer_config, Mapping):
                return transformer_config.get(name)
            return getattr(transformer_config, name, None)

        heads = config_value("num_attention_heads")
        head_dim = config_value("attention_head_dim")
        patch_embedding = getattr(transformer, "patch_embedding", None)
        if (
            type(heads) is not int
            or type(head_dim) is not int
            or heads * head_dim != EXPECTED_HIDDEN_DIM
            or not isinstance(patch_embedding, torch.nn.Conv3d)
            or patch_embedding.in_channels != 16
            or patch_embedding.out_channels != EXPECTED_HIDDEN_DIM
            or tuple(patch_embedding.kernel_size) != (1, 2, 2)
            or config_value("in_channels") != 16
            or config_value("out_channels") != 16
            or tuple(config_value("patch_size") or ()) != (1, 2, 2)
            or config_value("text_dim") != EXPECTED_TEXT_DIM
        ):
            raise SAICSourceAnchorNativeRuntimeError(
                "transformer is not pinned Bernini-R 1.3B geometry"
            )
        originals = {
            "sample": getattr(diffusion, "sample", None),
            "shared_step": getattr(diffusion, "shared_step", None),
            "scheduler.step": getattr(scheduler, "step", None),
        }
        if any(not callable(value) for value in originals.values()):
            raise SAICSourceAnchorNativeRuntimeError(
                "official sampler call surface differs"
            )
        # This must be the inner/native wrapper so auxiliary target-only
        # action-field wrappers can be installed outside it safely.
        for owner, name in (
            (diffusion, "sample"),
            (diffusion, "shared_step"),
            (scheduler, "step"),
        ):
            try:
                if name in vars(owner):
                    raise SAICSourceAnchorNativeRuntimeError(
                        f"install source-anchor runtime before any {name} wrapper"
                    )
            except TypeError as error:
                raise SAICSourceAnchorNativeRuntimeError(
                    f"cannot inspect {name} owner"
                ) from error
        if source_anchor.active_route() is not None:
            raise SAICSourceAnchorNativeRuntimeError(
                "source anchor has a pre-existing active route"
            )

        self.diffusion = diffusion
        self.transformer = transformer
        self.scheduler = scheduler
        self.handle = handle
        self.handle_receipt = dict(handle_receipt)
        self.config = config
        self.original_sample = originals["sample"]
        self.original_shared_step = originals["shared_step"]
        self.original_scheduler_step = originals["scheduler.step"]
        self._patches: list[tuple[Any, str, bool, Any, Any]] = []
        self._active: Optional[_ActiveSample] = None
        self.installed = False
        self.restored = False
        self.finalized = False
        self.schedule_audit: Optional[Mapping[str, Any]] = None
        self.sample_invocations = 0
        self.successful_samples = 0
        self.full_source_forwards = 0
        self.target_only_teacher_forwards = 0
        self.scheduler_steps = 0
        self.trace: list[dict[str, Any]] = []

    def _set_patch(self, owner: Any, name: str, value: Any) -> None:
        instance = vars(owner)
        had_instance = name in instance
        previous = instance.get(name)
        resolved = getattr(owner, name)
        setattr(owner, name, value)
        self._patches.append((owner, name, had_instance, previous, resolved))

    def install(self) -> None:
        if self.installed or self.restored or self.finalized:
            raise SAICSourceAnchorNativeRuntimeError(
                "source-anchor runtime lifecycle differs"
            )

        def sample_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_sample(*args, **kwargs)

        def shared_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_shared_step(*args, **kwargs)

        def scheduler_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_scheduler_step(*args, **kwargs)

        for wrapper in (sample_wrapper, shared_wrapper, scheduler_wrapper):
            setattr(wrapper, _MARKER, self)
        try:
            self._set_patch(self.scheduler, "step", scheduler_wrapper)
            self._set_patch(self.diffusion, "shared_step", shared_wrapper)
            self._set_patch(self.diffusion, "sample", sample_wrapper)
        except Exception:
            self._restore_patches(require_identity=False)
            raise
        self.installed = True

    def _restore_patches(self, *, require_identity: bool) -> None:
        errors: list[Exception] = []
        while self._patches:
            owner, name, had_instance, previous, resolved = self._patches.pop()
            try:
                current = getattr(owner, name, None)
                if require_identity and getattr(current, _MARKER, None) is not self:
                    errors.append(
                        SAICSourceAnchorNativeRuntimeError(
                            f"{name} changed during source-anchor runtime"
                        )
                    )
                if had_instance:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
                if getattr(owner, name, None) != resolved:
                    errors.append(
                        SAICSourceAnchorNativeRuntimeError(
                            f"{name} restoration failed"
                        )
                    )
            except Exception as error:
                errors.append(error)
        self._active = None
        if errors:
            raise SAICSourceAnchorNativeRuntimeError(
                f"failed to restore {len(errors)} source-anchor wrapper(s)"
            ) from errors[0]

    def restore(self) -> None:
        if not self.installed or self.restored:
            raise SAICSourceAnchorNativeRuntimeError(
                "source-anchor runtime restore lifecycle differs"
            )
        try:
            self._restore_patches(require_identity=True)
        finally:
            self.installed = False
            self.restored = not self._patches

    def _validate_prompt(self, value: Any, *, label: str) -> torch.Tensor:
        tensor = _detached_finite_tensor(value, label=label)
        if _shape(tensor, label=label) != (
            1,
            EXPECTED_TEXT_TOKENS,
            EXPECTED_TEXT_DIM,
        ):
            raise SAICSourceAnchorNativeRuntimeError(
                f"{label} prompt geometry differs"
            )
        return tensor

    def _validate_sample(self, values: Mapping[str, Any]) -> _ActiveSample:
        if (
            values.get("guidance_mode") != EXPECTED_GUIDANCE_MODE
            or values.get("num_frames") != EXPECTED_FRAMES
            or values.get("num_inference_steps") != EXPECTED_STEPS
            or _scalar(values.get("flow_shift"), label="flow_shift")
            != EXPECTED_FLOW_SHIFT
            or _scalar(values.get("omega_vid"), label="omega_vid") != 1.25
            or _scalar(values.get("omega_txt"), label="omega_txt") != 4.0
            or _scalar(values.get("omega_scale"), label="omega_scale") != 0.8
            or _scalar(values.get("eta"), label="eta") != 0.5
            or _scalar(values.get("momentum"), label="momentum") != 0.0
            or values.get("prompt_embeds_t2") is not None
            or values.get("uncond_embeds_t2") is not None
            or values.get("image_vae_latents") is not None
        ):
            raise SAICSourceAnchorNativeRuntimeError(
                "official exact81/exact40 v2v_apg sample contract differs"
            )
        threshold = values.get("norm_threshold")
        thresholds = (
            threshold if isinstance(threshold, (list, tuple)) else (threshold,)
        )
        if not thresholds or any(
            _scalar(value, label="norm_threshold") != 50.0
            for value in thresholds
        ):
            raise SAICSourceAnchorNativeRuntimeError(
                "norm_threshold must contain only 50"
            )
        expected_omega_img = 0.0 if self.config.branch_name == "V" else 4.5
        if _scalar(values.get("omega_img"), label="omega_img") != expected_omega_img:
            raise SAICSourceAnchorNativeRuntimeError(
                "omega_img differs from the registered V/VI branch"
            )
        forbidden = sorted(
            str(name)
            for name, value in values.items()
            if value is not None
            and any(token in str(name).lower() for token in _FORBIDDEN_SAMPLE_TOKENS)
        )
        if forbidden:
            raise SAICSourceAnchorNativeRuntimeError(
                "forbidden privileged sample inputs: " + ",".join(forbidden)
            )
        videos = values.get("multi_video_vae_latents")
        if not isinstance(videos, (list, tuple)) or len(videos) != 1:
            raise SAICSourceAnchorNativeRuntimeError(
                "full-source sampling requires exactly one source video latent"
            )
        source = _detached_finite_tensor(videos[0], label="source video latent")
        if tuple(source.shape) != tuple(self.config.target_latent_shape):
            raise SAICSourceAnchorNativeRuntimeError(
                "source video latent is not registered exact81 geometry"
            )
        references_value = values.get("multi_image_vae_latents")
        references = () if references_value is None else tuple(references_value)
        expected_refs = (
            0
            if self.config.branch_name == "V"
            else native_pack.REFERENCE_COUNT
        )
        if len(references) != expected_refs:
            raise SAICSourceAnchorNativeRuntimeError(
                "image-reference count differs from the registered V/VI branch"
            )
        expected_ref_shape = (
            self.config.target_latent_shape[0],
            self.config.target_latent_shape[1],
            1,
            self.config.target_latent_shape[3],
            self.config.target_latent_shape[4],
        )
        for index, reference in enumerate(references):
            checked = _detached_finite_tensor(reference, label=f"reference {index}")
            if tuple(checked.shape) != expected_ref_shape:
                raise SAICSourceAnchorNativeRuntimeError(
                    f"reference {index} geometry differs"
                )
        _, _, _, latent_height, latent_width = self.config.target_latent_shape
        if (
            type(values.get("height")) is not int
            or values["height"] != latent_height * 8
            or type(values.get("width")) is not int
            or values["width"] != latent_width * 8
        ):
            raise SAICSourceAnchorNativeRuntimeError(
                "decoded dimensions differ from registered latent geometry"
            )
        action = self._validate_prompt(values.get("prompt_embeds"), label="action")
        negative = self._validate_prompt(
            values.get("uncond_prompt_embeds"), label="negative"
        )
        if action is negative:
            raise SAICSourceAnchorNativeRuntimeError(
                "action/no-op and negative prompt objects must be distinct"
            )
        return _ActiveSample(action_prompt=action, negative_prompt=negative)

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        if (
            self._active is not None
            or self.sample_invocations != 0
            or source_anchor.active_route() is not None
            or self.diffusion.scheduler is not self.scheduler
            or self.handle.transformer is not self.transformer
            or bool(self.handle.restored)
        ):
            raise SAICSourceAnchorNativeRuntimeError(
                "source-anchor runtime permits one non-nested owned sample"
            )
        state = self._validate_sample(
            _flatten_bound_arguments(self.original_sample, args, kwargs)
        )
        self._active = state
        self.sample_invocations += 1
        try:
            result = self.original_sample(*args, **kwargs)
            if (
                state.completed_steps != EXPECTED_STEPS
                or state.pending_full
                or state.pending_teacher_roles
                or self.full_source_forwards != 2 * EXPECTED_STEPS
                or self.scheduler_steps != EXPECTED_STEPS
                or len(self.trace) != EXPECTED_STEPS
            ):
                raise SAICSourceAnchorNativeRuntimeError(
                    "sample returned without exact40 native call closure"
                )
            checked = _detached_finite_tensor(result, label="official sample result")
            if (
                tuple(checked.shape) != tuple(self.config.target_latent_shape)
                or checked.dtype != torch.float32
            ):
                raise SAICSourceAnchorNativeRuntimeError(
                    "official sample result is not FP32 exact81 latent geometry"
                )
            self.successful_samples += 1
            return result
        finally:
            self._active = None

    def _audit_schedule(self) -> None:
        if self.schedule_audit is not None:
            return
        try:
            audit = sigma_strata.audit_runtime_unipc_schedule(
                self.scheduler, initialize=False
            )
        except Exception as error:
            raise SAICSourceAnchorNativeRuntimeError(
                f"live exact40 UniPC schedule differs: {error}"
            ) from error
        if audit.get("schedule_sha256") != sigma_strata.SCHEDULE_SHA256:
            raise SAICSourceAnchorNativeRuntimeError(
                "live exact40 schedule digest differs"
            )
        self.schedule_audit = dict(audit)

    def _schedule_index(self, timestep: torch.Tensor, *, expected: int) -> int:
        tensor = _detached_finite_tensor(timestep, label="model timestep")
        if tuple(tensor.shape) != (1,) or tensor.dtype not in (torch.int64, torch.float32):
            raise SAICSourceAnchorNativeRuntimeError(
                "model timestep must be exact INT64/FP32 singleton"
            )
        numeric = _scalar(tensor, label="model timestep")
        matches = [
            index
            for index, registered in enumerate(sigma_strata.PINNED_TIMESTEPS)
            if numeric == float(registered)
        ]
        if len(matches) != 1 or matches[0] != expected:
            raise SAICSourceAnchorNativeRuntimeError(
                f"exact40 call order differs at expected index {expected}"
            )
        return matches[0]

    def _validate_shared_common(
        self,
        values: Mapping[str, Any],
        *,
        schedule_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.get("model_id") != EXPECTED_MODEL_ID:
            raise SAICSourceAnchorNativeRuntimeError("shared_step model route differs")
        noisy = _detached_finite_tensor(values.get("noisy_latents"), label="packed noisy")
        timestep = _detached_finite_tensor(values.get("timesteps"), label="shared timestep")
        prompt = self._validate_prompt(values.get("cond_embeds"), label="shared")
        rotary = _detached_finite_tensor(values.get("rotary_embs"), label="packed rotary")
        self._schedule_index(timestep, expected=schedule_index)
        if timestep.device != noisy.device:
            raise SAICSourceAnchorNativeRuntimeError(
                "shared INT64/FP32 timestep must be on the noisy forward device"
            )
        return noisy, timestep, prompt, rotary

    def _validate_pack_metadata(
        self,
        values: Mapping[str, Any],
        *,
        tokens: int,
        label: str,
    ) -> None:
        if _metadata_tuple(
            values.get("batch_vae_seqlen"), label=f"{label} VAE length"
        ) != (tokens,):
            raise SAICSourceAnchorNativeRuntimeError(
                f"{label} batch_vae_seqlen differs"
            )
        if _metadata_tuple(values.get("batch_text_seqlen"), label=f"{label} text length") != (
            EXPECTED_TEXT_TOKENS,
        ):
            raise SAICSourceAnchorNativeRuntimeError(
                f"{label} batch_text_seqlen differs"
            )

    def _native_branch(
        self,
        noisy: torch.Tensor,
        rotary: torch.Tensor,
    ) -> native_pack.NativeRV2VBranch:
        mask = torch.zeros(
            self.config.total_tokens, dtype=torch.bool, device=noisy.device
        )
        mask[self.config.condition_tokens :] = True
        if self.config.branch_name == "V":
            source_ids = native_pack.VI_VIDEO_SOURCE_IDS + (0.0,)
        else:
            source_ids = (
                native_pack.VI_VIDEO_SOURCE_IDS
                + native_pack.VI_IMAGE_SOURCE_IDS
                + (0.0,)
            )
        return native_pack.NativeRV2VBranch(
            name=self.config.branch_name,
            latents=noisy,
            rotary=rotary,
            target_mask=mask,
            total_tokens=self.config.total_tokens,
            condition_tokens=self.config.condition_tokens,
            source_ids=source_ids,
            concat_order=native_pack.BRANCH_CONCAT_ORDER[self.config.branch_name],
        )

    def _validate_prediction(self, result: Any, *, tokens: int, label: str) -> torch.Tensor:
        checked = _detached_finite_tensor(result, label=label)
        if tuple(checked.shape) != (1, tokens, EXPECTED_OUTPUT_CHANNELS):
            raise SAICSourceAnchorNativeRuntimeError(f"{label} geometry differs")
        return checked

    def _target_teacher_role(self, prompt: torch.Tensor, state: _ActiveSample) -> str:
        if prompt is state.negative_prompt:
            return "teacher-negative"
        if prompt is state.action_prompt:
            return "teacher-action-or-noop"
        return "teacher-auxiliary"

    def _wrapped_shared_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None or source_anchor.active_route() is not None:
            raise SAICSourceAnchorNativeRuntimeError(
                "shared_step ran outside an unambiguous sample boundary"
            )
        self._audit_schedule()
        schedule_index = state.completed_steps
        values = _flatten_bound_arguments(self.original_shared_step, args, kwargs)
        noisy, timestep, prompt, rotary = self._validate_shared_common(
            values, schedule_index=schedule_index
        )
        tokens = int(noisy.shape[1]) if noisy.ndim == 3 else -1
        if tokens == self.config.total_tokens:
            if len(state.pending_full) >= 2:
                raise SAICSourceAnchorNativeRuntimeError(
                    "more than two full-source forwards occurred before scheduler.step"
                )
            role = FULL_FORWARD_ORDER[len(state.pending_full)]
            expected_prompt = (
                state.negative_prompt if role == "negative" else state.action_prompt
            )
            _same_object(prompt, expected_prompt, label=f"official {role} prompt")
            if tuple(noisy.shape) != (1, self.config.total_tokens, EXPECTED_HIDDEN_DIM):
                raise SAICSourceAnchorNativeRuntimeError(
                    "full-source packed latent geometry differs"
                )
            if (
                rotary.ndim != 4
                or tuple(rotary.shape[:3]) != (1, 1, self.config.total_tokens)
                or int(rotary.shape[3]) <= 0
            ):
                raise SAICSourceAnchorNativeRuntimeError(
                    "full-source rotary geometry differs"
                )
            self._validate_pack_metadata(
                values, tokens=self.config.total_tokens, label="full-source"
            )
            if state.pending_full:
                first = state.pending_full[0]
                for name, left, right in (
                    ("noisy_latents", first.noisy_latents, noisy),
                    ("timesteps", first.timesteps, timestep),
                    ("rotary_embs", first.rotary_embs, rotary),
                ):
                    _same_object(left, right, label=f"negative/action {name}")
            branch = self._native_branch(noisy, rotary)
            with self.handle.route(
                branch=branch,
                scheduler=self.scheduler,
                timestep=timestep,
            ) as route:
                if source_anchor.active_route() is not route:
                    raise SAICSourceAnchorNativeRuntimeError(
                        "source anchor handle did not activate its exact route"
                    )
                result = self.original_shared_step(*args, **kwargs)
            if source_anchor.active_route() is not None:
                raise SAICSourceAnchorNativeRuntimeError(
                    "source anchor route leaked after official forward"
                )
            self._validate_prediction(
                result, tokens=self.config.total_tokens, label=f"official {role} result"
            )
            state.pending_full.append(
                _ForwardObservation(role, noisy, timestep, rotary)
            )
            self.full_source_forwards += 1
            return result

        if tokens == self.config.target_tokens:
            if not state.pending_full or len(state.pending_full) > 2:
                raise SAICSourceAnchorNativeRuntimeError(
                    "target-only teacher appeared outside a native full-source cell"
                )
            first = state.pending_full[0]
            _same_object(first.timesteps, timestep, label="teacher/full timestep")
            if tuple(noisy.shape) != (1, self.config.target_tokens, EXPECTED_HIDDEN_DIM):
                raise SAICSourceAnchorNativeRuntimeError(
                    "target-only teacher latent geometry differs"
                )
            if (
                rotary.ndim != 4
                or tuple(rotary.shape[:3]) != (1, 1, self.config.target_tokens)
                or int(rotary.shape[3]) <= 0
            ):
                raise SAICSourceAnchorNativeRuntimeError(
                    "target-only teacher rotary geometry differs"
                )
            _certify_exact_suffix_view(
                first.noisy_latents,
                noisy,
                token_dim=1,
                suffix_tokens=self.config.target_tokens,
                label="teacher latent",
            )
            _certify_exact_suffix_view(
                first.rotary_embs,
                rotary,
                token_dim=2,
                suffix_tokens=self.config.target_tokens,
                label="teacher rotary",
            )
            self._validate_pack_metadata(
                values, tokens=self.config.target_tokens, label="target-only teacher"
            )
            # No route: source/ref metadata do not exist in this branch.
            result = self.original_shared_step(*args, **kwargs)
            self._validate_prediction(
                result,
                tokens=self.config.target_tokens,
                label="target-only teacher result",
            )
            state.pending_teacher_roles.append(
                self._target_teacher_role(prompt, state)
            )
            self.target_only_teacher_forwards += 1
            return result

        raise SAICSourceAnchorNativeRuntimeError(
            "shared_step is neither the registered full-source pack nor its target suffix"
        )

    def _wrapped_scheduler_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None or source_anchor.active_route() is not None:
            raise SAICSourceAnchorNativeRuntimeError(
                "scheduler.step ran outside an unambiguous sample boundary"
            )
        if tuple(row.role for row in state.pending_full) != FULL_FORWARD_ORDER:
            raise SAICSourceAnchorNativeRuntimeError(
                "scheduler.step arrived before the official negative/action pair"
            )
        index = state.completed_steps
        output = _extract_argument(args, kwargs, index=0, name="model_output")
        timestep = _extract_argument(args, kwargs, index=1, name="timestep")
        sample = _extract_argument(args, kwargs, index=2, name="sample")
        self._validate_prediction(
            output,
            tokens=self.config.target_tokens,
            label="scheduler model_output",
        )
        self._validate_prediction(
            sample, tokens=self.config.target_tokens, label="scheduler sample"
        )
        _detached_finite_tensor(timestep, label="scheduler timestep")
        self._schedule_index(timestep.reshape(1), expected=index)
        _certify_expanded_timestep(state.pending_full[0].timesteps, timestep)
        live_index = getattr(self.scheduler, "step_index", None)
        if live_index is None:
            live_index = getattr(self.scheduler, "_step_index", None)
        if live_index is not None and int(
            _scalar(live_index, label="scheduler step_index")
        ) != index:
            raise SAICSourceAnchorNativeRuntimeError(
                "scheduler live step index differs from exact40 order"
            )
        result = self.original_scheduler_step(*args, **kwargs)
        self.scheduler_steps += 1
        self.trace.append(
            {
                "schema_version": SCHEMA_VERSION,
                "schedule_index": index,
                "timestep": sigma_strata.PINNED_TIMESTEPS[index],
                "timestep_dtype": str(state.pending_full[0].timesteps.dtype),
                "sigma_float32_be_hex": sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[index],
                "branch_name": self.config.branch_name,
                "full_source_forward_order": list(FULL_FORWARD_ORDER),
                "full_source_forwards": 2,
                "target_only_teacher_roles": list(state.pending_teacher_roles),
                "target_only_teacher_forwards": len(state.pending_teacher_roles),
                "anchor_active": index in source_anchor.ACTIVE_SIGMA_INDICES,
                "condition_tokens": self.config.condition_tokens,
                "target_tokens": self.config.target_tokens,
                "total_tokens": self.config.total_tokens,
                "source_reference_padding_rows_exact_base": True,
                "official_arguments_mutated": False,
                "original_scheduler_calls": 1,
            }
        )
        state.pending_full.clear()
        state.pending_teacher_roles.clear()
        state.completed_steps += 1
        return result

    def finalize(self) -> Mapping[str, Any]:
        if not self.restored or self.finalized:
            raise SAICSourceAnchorNativeRuntimeError(
                "source-anchor runtime finalize lifecycle differs"
            )
        if (
            self.schedule_audit is None
            or self.sample_invocations != 1
            or self.successful_samples != 1
            or self.full_source_forwards != 2 * EXPECTED_STEPS
            or self.scheduler_steps != EXPECTED_STEPS
            or len(self.trace) != EXPECTED_STEPS
            or [row["schedule_index"] for row in self.trace] != list(range(EXPECTED_STEPS))
        ):
            raise SAICSourceAnchorNativeRuntimeError(
                "source-anchor exact40 call-count certificate differs"
            )
        value = {
            "schema_version": SCHEMA_VERSION,
            "official_sample_calls": 1,
            "exact81": True,
            "exact40": True,
            "branch_name": self.config.branch_name,
            "official_full_source_forwards": self.full_source_forwards,
            "full_source_forward_order": list(FULL_FORWARD_ORDER),
            "target_only_teacher_forwards_unrouted": self.target_only_teacher_forwards,
            "official_scheduler_steps": self.scheduler_steps,
            "condition_tokens": self.config.condition_tokens,
            "target_tokens": self.config.target_tokens,
            "total_tokens": self.config.total_tokens,
            "active_schedule_indices": list(source_anchor.ACTIVE_SIGMA_INDICES),
            "exact40_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
            "adapter_contract_digest": self.handle_receipt["digest"],
            "source_reference_padding_rows_exact_base": True,
            "action_and_noop_share_anchor_route": True,
            "target_only_teacher_has_anchor_route": False,
            "install_order": "source_anchor_inner_then_optional_action_teacher_outer",
            "mask_pose_flow_track_trajectory_consumed": False,
            "target_or_proposal_video_consumed": False,
            "optimizer_created": False,
            "semantic_action_editing_claim": False,
            "appearance_preservation_claim": False,
            "trace": list(self.trace),
        }
        self.finalized = True
        return {**value, "digest": _object_sha256(value)}


@contextmanager
def saic_source_anchor_native_runtime(
    renderer_or_diffusion: Any,
    *,
    handle: source_anchor.SAICSourceAnchorHandle,
    config: SourceAnchorNativeRuntimeConfig,
) -> Iterator[SAICSourceAnchorNativeRuntimePatch]:
    patch = SAICSourceAnchorNativeRuntimePatch(
        renderer_or_diffusion, handle=handle, config=config
    )
    patch.install()
    try:
        yield patch
    finally:
        if patch.installed and not patch.restored:
            patch.restore()


__all__ = [
    "FORMAL_OPTIMIZER_UPDATES",
    "SAFETENSORS_SCHEMA_VERSION",
    "SAICSourceAnchorNativeRuntimeError",
    "SAICSourceAnchorNativeRuntimePatch",
    "SCHEMA_VERSION",
    "SourceAnchorNativeRuntimeConfig",
    "load_saic_source_anchor_safetensors",
    "saic_source_anchor_native_runtime",
]
