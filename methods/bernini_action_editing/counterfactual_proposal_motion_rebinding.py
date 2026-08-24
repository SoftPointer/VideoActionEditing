"""Counterfactual Proposal Motion Rebinding (CPMR), V11 tensor core.

This module intentionally stops at the frozen proposal-carrier boundary.  It does
not know about Bernini processors, sequence parallelism, Ulysses, LoRA, or a
training runner.  Its job is narrower and auditable:

1. reshape exact Bernini proposal patch embeddings;
2. form a first-order temporal proposal difference in FP32;
3. pool the 31 x 30 proposal grid to the fixed 8 x 8 carrier grid;
4. normalize, clip, and add a fixed 3-D coordinate code; and
5. construct the REV, SHUF, NEG, and NN scientific controls.

The canonical contract is deliberately closed: 81 RGB frames correspond to 21
latent phases, the hidden width is 1536, and every carrier has 21 * 8 * 8 =
1344 tokens per sample.  Any deviation is rejected rather than silently adapted.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence


METHOD_NAME = "counterfactual-proposal-motion-rebinding-v11"
SCHEMA_VERSION = "bernini-cpmr-carrier-v1"

RGB_FRAMES = 81
LATENT_PHASES = 21
PATCH_HEIGHT = 31
PATCH_WIDTH = 30
HIDDEN_SIZE = 1536
POOL_HEIGHT = 8
POOL_WIDTH = 8
CARRIER_TOKENS = LATENT_PHASES * POOL_HEIGHT * POOL_WIDTH

EPSILON = 1.0e-6
TOKEN_RMS_CAP = 4.0
COORDINATE_SCALE = 0.02
COORDINATE_AXIS_WIDTH = 512
COORDINATE_FREQUENCIES = 256
COORDINATE_BASE = 10000.0

IDENTITY_SOURCE_PHASES = tuple(range(LATENT_PHASES))
REVERSE_SOURCE_PHASES = (0, *tuple(range(LATENT_PHASES - 1, 0, -1)))
SHUFFLE_SOURCE_PHASES_NONZERO = (
    17,
    18,
    1,
    6,
    16,
    4,
    12,
    11,
    7,
    13,
    19,
    2,
    15,
    8,
    3,
    9,
    20,
    5,
    10,
    14,
)
SHUFFLE_SOURCE_PHASES = (0, *SHUFFLE_SOURCE_PHASES_NONZERO)
SHUFFLE_CANONICAL_JSON = json.dumps(
    list(SHUFFLE_SOURCE_PHASES_NONZERO), separators=(",", ":")
)
SHUFFLE_CANONICAL_SHA256 = (
    "399dbccd424e30cbb4a129c7e5535bdb6475cf6573d024c757eb8491803eb02f"
)

CONTROL_NAMES = ("correct", "reverse", "shuffle", "negative", "nn")


class CPMRTensorContractError(ValueError):
    """Raised when a tensor or configuration violates the frozen V11 contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the single canonical JSON representation used by all CPMR digests."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


if hashlib.sha256(SHUFFLE_CANONICAL_JSON.encode("utf-8")).hexdigest() != (
    SHUFFLE_CANONICAL_SHA256
):
    raise RuntimeError("the frozen SHUF phase permutation digest is inconsistent")


@dataclass(frozen=True)
class CPMRConfig:
    """Exact, fail-closed V11 carrier configuration.

    The fields are exposed for receipt generation, not as tuning knobs.  A
    non-canonical value is rejected in ``__post_init__``.
    """

    rgb_frames: int = RGB_FRAMES
    latent_phases: int = LATENT_PHASES
    patch_height: int = PATCH_HEIGHT
    patch_width: int = PATCH_WIDTH
    hidden_size: int = HIDDEN_SIZE
    pool_height: int = POOL_HEIGHT
    pool_width: int = POOL_WIDTH
    epsilon: float = EPSILON
    token_rms_cap: float = TOKEN_RMS_CAP
    coordinate_scale: float = COORDINATE_SCALE
    coordinate_axis_width: int = COORDINATE_AXIS_WIDTH
    coordinate_frequencies: int = COORDINATE_FREQUENCIES
    coordinate_base: float = COORDINATE_BASE
    shuffle_source_phases: tuple[int, ...] = SHUFFLE_SOURCE_PHASES_NONZERO

    def __post_init__(self) -> None:
        self.validate()

    @property
    def patch_tokens(self) -> int:
        return self.latent_phases * self.patch_height * self.patch_width

    @property
    def carrier_tokens(self) -> int:
        return self.latent_phases * self.pool_height * self.pool_width

    @property
    def source_patch_shape(self) -> tuple[int, int, int, int]:
        return (
            self.latent_phases,
            self.patch_height,
            self.patch_width,
            self.hidden_size,
        )

    @property
    def carrier_shape(self) -> tuple[int, int, int, int]:
        return (
            self.latent_phases,
            self.pool_height,
            self.pool_width,
            self.hidden_size,
        )

    def validate(self) -> None:
        exact: Mapping[str, Any] = {
            "rgb_frames": RGB_FRAMES,
            "latent_phases": LATENT_PHASES,
            "patch_height": PATCH_HEIGHT,
            "patch_width": PATCH_WIDTH,
            "hidden_size": HIDDEN_SIZE,
            "pool_height": POOL_HEIGHT,
            "pool_width": POOL_WIDTH,
            "epsilon": EPSILON,
            "token_rms_cap": TOKEN_RMS_CAP,
            "coordinate_scale": COORDINATE_SCALE,
            "coordinate_axis_width": COORDINATE_AXIS_WIDTH,
            "coordinate_frequencies": COORDINATE_FREQUENCIES,
            "coordinate_base": COORDINATE_BASE,
            "shuffle_source_phases": SHUFFLE_SOURCE_PHASES_NONZERO,
        }
        for name, wanted in exact.items():
            got = getattr(self, name)
            if isinstance(wanted, int) and isinstance(got, bool):
                raise CPMRTensorContractError(f"{name} must equal {wanted!r}, got bool")
            if got != wanted:
                raise CPMRTensorContractError(
                    f"{name} is frozen to {wanted!r} by V11, got {got!r}"
                )
        if self.coordinate_axis_width * 3 != self.hidden_size:
            raise CPMRTensorContractError("three coordinate axes must fill width 1536")
        if self.coordinate_frequencies * 2 != self.coordinate_axis_width:
            raise CPMRTensorContractError("each axis must contain 256 sin/cos pairs")
        if self.carrier_tokens != CARRIER_TOKENS:
            raise CPMRTensorContractError("the carrier must contain exactly 1344 tokens")

    def contract_dict(self) -> dict[str, Any]:
        return {
            "method": METHOD_NAME,
            "schema_version": SCHEMA_VERSION,
            "rgb_frames": self.rgb_frames,
            "latent_phases": self.latent_phases,
            "proposal_grid": [self.patch_height, self.patch_width],
            "carrier_grid": [self.pool_height, self.pool_width],
            "hidden_size": self.hidden_size,
            "patch_tokens": self.patch_tokens,
            "carrier_tokens": self.carrier_tokens,
            "arithmetic_dtype": "float32",
            "pooling": "adaptive_avg_pool2d",
            "temporal_difference": "D[0]=0; D[t]=C[t]-C[t-1]",
            "epsilon": self.epsilon,
            "precoordinate_token_rms_cap": self.token_rms_cap,
            "coordinate_scale": self.coordinate_scale,
            "coordinate_axis_width": self.coordinate_axis_width,
            "coordinate_frequencies": self.coordinate_frequencies,
            "coordinate_base": self.coordinate_base,
            "shuffle_source_phases_nonzero": list(self.shuffle_source_phases),
            "shuffle_sha256": SHUFFLE_CANONICAL_SHA256,
        }

    def contract_sha256(self) -> str:
        return canonical_json_sha256(self.contract_dict())


CANONICAL_CONFIG = CPMRConfig()
CONFIG_CONTRACT_SHA256 = (
    "cca36a368209cba2d1218591a7a0facc9e15cacc47a8b89c2a05b829e886a5ca"
)
if CANONICAL_CONFIG.contract_sha256() != CONFIG_CONTRACT_SHA256:
    raise RuntimeError("the frozen CPMR configuration digest is inconsistent")

COORDINATE_SPEC = {
    "axis_order": ["time", "y", "x"],
    "axis_sizes": [LATENT_PHASES, POOL_HEIGHT, POOL_WIDTH],
    "axis_width": COORDINATE_AXIS_WIDTH,
    "component_order": "frequency_interleaved_sin_cos",
    "coordinate_formula": "2*pi*index/(size-1)",
    "dtype": "float32",
    "frequencies_per_axis": COORDINATE_FREQUENCIES,
    "frequency_formula": "10000**(-2*k/512), k=0..255",
    "hidden_size": HIDDEN_SIZE,
}
COORDINATE_SPEC_SHA256 = (
    "f3712bd5420a313f3b98f0e06081998f3c14f735ddcc4d34ff42f592f2ecfb03"
)
if canonical_json_sha256(COORDINATE_SPEC) != COORDINATE_SPEC_SHA256:
    raise RuntimeError("the frozen CPMR coordinate specification digest is inconsistent")

CANONICAL_COORDINATE_TENSOR_SHA256 = (
    "de31a70bc0c62d4764bc5025f508805d030bd25207c05986a2b4372d9eb52861"
)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on caller environment
        raise RuntimeError("CPMR tensor operations require PyTorch") from exc
    return torch


def _config_or_default(config: CPMRConfig | None) -> CPMRConfig:
    result = CANONICAL_CONFIG if config is None else config
    if not isinstance(result, CPMRConfig):
        raise CPMRTensorContractError("config must be a CPMRConfig")
    result.validate()
    return result


def _ensure_tensor(name: str, value: Any, torch: Any) -> None:
    if not isinstance(value, torch.Tensor):
        raise CPMRTensorContractError(f"{name} must be a torch.Tensor")
    if value.device.type == "meta":
        raise CPMRTensorContractError(f"{name} cannot be a meta tensor")
    if not torch.is_floating_point(value):
        raise CPMRTensorContractError(f"{name} must have a floating dtype")


def _ensure_finite(name: str, value: Any, torch: Any) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise CPMRTensorContractError(f"{name} contains NaN or infinity")


def tensor_sha256(tensor: Any) -> str:
    """Hash tensor dtype, shape, and canonical contiguous CPU bytes."""

    torch = _require_torch()
    if not isinstance(tensor, torch.Tensor):
        raise CPMRTensorContractError("tensor_sha256 expects a torch.Tensor")
    if tensor.device.type == "meta":
        raise CPMRTensorContractError("cannot hash a meta tensor")
    value = tensor.detach().contiguous().cpu()
    metadata = {
        "dtype": str(value.dtype),
        "shape": [int(item) for item in value.shape],
    }
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(metadata))
    digest.update(b"\0")
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tensor_bytes_equal(left: Any, right: Any, torch: Any) -> bool:
    """Compare tensor metadata and payload bits, including the sign bit of zero."""

    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return False
    if left.device != right.device:
        return False
    left_bytes = left.detach().contiguous().reshape(-1).view(torch.uint8)
    right_bytes = right.detach().contiguous().reshape(-1).view(torch.uint8)
    return bool(torch.equal(left_bytes, right_bytes))


def reshape_proposal_patch_tokens(
    patch_embedding_output: Any,
    *,
    config: CPMRConfig | None = None,
) -> Any:
    """Return exact channel-last proposal patches ``[B,21,31,30,1536]``.

    Supported exact Bernini boundary layouts are channel-first 5-D, temporal
    channel-first 5-D, packed token 3-D, and their unbatched equivalents.  An
    already channel-last exact field is accepted as an idempotent operation.
    Spatial or temporal interpolation is never performed.
    """

    cfg = _config_or_default(config)
    torch = _require_torch()
    value = patch_embedding_output
    _ensure_tensor("patch_embedding_output", value, torch)

    t, h, w, d = cfg.source_patch_shape
    packed = cfg.patch_tokens
    shape = tuple(int(item) for item in value.shape)

    if value.ndim == 5 and shape[1:] == (d, t, h, w):
        result = value.permute(0, 2, 3, 4, 1).contiguous()
    elif value.ndim == 5 and shape[1:] == (t, d, h, w):
        result = value.permute(0, 1, 3, 4, 2).contiguous()
    elif value.ndim == 5 and shape[1:] == (t, h, w, d):
        result = value.contiguous()
    elif value.ndim == 4 and shape == (d, t, h, w):
        result = value.permute(1, 2, 3, 0).unsqueeze(0).contiguous()
    elif value.ndim == 4 and shape == (t, h, w, d):
        result = value.unsqueeze(0).contiguous()
    elif value.ndim == 3 and shape[1:] == (packed, d):
        result = value.reshape(shape[0], t, h, w, d).contiguous()
    elif value.ndim == 2 and shape == (packed, d):
        result = value.reshape(1, t, h, w, d).contiguous()
    else:
        raise CPMRTensorContractError(
            "patch_embedding_output cannot be reshaped without changing the frozen "
            f"[21,31,30,1536] contract; got {shape}"
        )

    if int(result.shape[0]) < 1:
        raise CPMRTensorContractError("proposal batch dimension must be positive")
    _ensure_finite("patch_embedding_output", result, torch)
    return result


def _coerce_patch_field(name: str, value: Any, cfg: CPMRConfig, torch: Any) -> Any:
    _ensure_tensor(name, value, torch)
    if value.ndim == 4 and tuple(int(item) for item in value.shape) == (
        cfg.source_patch_shape
    ):
        value = value.unsqueeze(0)
    expected_tail = cfg.source_patch_shape
    if value.ndim != 5 or tuple(int(item) for item in value.shape[1:]) != expected_tail:
        raise CPMRTensorContractError(
            f"{name} must be [B,{','.join(str(x) for x in expected_tail)}], "
            f"got {tuple(int(item) for item in value.shape)}"
        )
    if int(value.shape[0]) < 1:
        raise CPMRTensorContractError(f"{name} batch dimension must be positive")
    _ensure_finite(name, value, torch)
    return value


@dataclass(frozen=True)
class CPMRNormalizationResult:
    """FP32 result of activity gating, phase normalization, and token clipping."""

    activity: Any
    phase_rms: Any
    normalized_content: Any
    token_rms_before_clip: Any
    clip_scale: Any
    clipped_content: Any
    clip_fraction: Any


def _assert_inactive_exact_zero(
    name: str,
    tensor: Any,
    activity: Any,
    torch: Any,
) -> None:
    for batch_index in range(int(activity.shape[0])):
        inactive = torch.nonzero(~activity[batch_index], as_tuple=False).flatten()
        for phase_index in inactive.tolist():
            value = tensor[batch_index, phase_index]
            if not _tensor_bytes_equal(value, torch.zeros_like(value), torch):
                raise CPMRTensorContractError(
                    f"{name} is not byte-exact positive zero at inactive phase "
                    f"{phase_index}"
                )


def normalize_and_clip_pooled_increments(
    pooled_increments: Any,
    *,
    config: CPMRConfig | None = None,
) -> CPMRNormalizationResult:
    """Normalize a canonical pooled ``D`` field and apply the RMS/token cap.

    ``pooled_increments`` must already be FP32 ``[B,21,8,8,1536]`` and phase 0
    must be bit-exact zero.  Keeping this stage public makes its scientific
    invariants testable without constructing a 31 x 30 proposal field.
    """

    cfg = _config_or_default(config)
    torch = _require_torch()
    value = pooled_increments
    _ensure_tensor("pooled_increments", value, torch)
    expected = cfg.carrier_shape
    if value.ndim != 5 or tuple(int(item) for item in value.shape[1:]) != expected:
        raise CPMRTensorContractError(
            f"pooled_increments must be [B,{','.join(str(x) for x in expected)}], "
            f"got {tuple(int(item) for item in value.shape)}"
        )
    if value.dtype != torch.float32:
        raise CPMRTensorContractError("all carrier arithmetic must enter as float32")
    if int(value.shape[0]) < 1:
        raise CPMRTensorContractError("pooled_increments batch must be positive")
    _ensure_finite("pooled_increments", value, torch)
    if int(torch.count_nonzero(value[:, 0]).item()) != 0:
        raise CPMRTensorContractError("D[0] and pooled P[0] must be exact zero")

    with torch.no_grad():
        max_abs = value.abs().amax(dim=(2, 3, 4))
        activity = max_abs > 0
        activity[:, 0] = False

        phase_rms = value.square().mean(dim=(2, 3, 4)).sqrt()
        denominator = phase_rms.clamp_min(cfg.epsilon)
        normalized_raw = value / denominator[:, :, None, None, None]
        activity_field = activity[:, :, None, None, None]
        normalized = torch.where(
            activity_field, normalized_raw, torch.zeros_like(normalized_raw)
        )

        token_rms = normalized.square().mean(dim=-1).sqrt()
        clip_scale = torch.clamp(token_rms / cfg.token_rms_cap, min=1.0)
        clipped = normalized / clip_scale[..., None]
        clipped = torch.where(activity_field, clipped, torch.zeros_like(clipped))
        clip_fraction = (clip_scale > 1.0).to(torch.float32).mean(dim=(2, 3))
        clip_fraction = torch.where(
            activity, clip_fraction, torch.zeros_like(clip_fraction)
        )

    for name, tensor in (
        ("phase_rms", phase_rms),
        ("normalized_content", normalized),
        ("token_rms_before_clip", token_rms),
        ("clip_scale", clip_scale),
        ("clipped_content", clipped),
        ("clip_fraction", clip_fraction),
    ):
        _ensure_finite(name, tensor, torch)
    _assert_inactive_exact_zero("normalized_content", normalized, activity, torch)
    _assert_inactive_exact_zero("clipped_content", clipped, activity, torch)

    clipped_rms = clipped.square().mean(dim=-1).sqrt()
    if bool((clipped_rms > cfg.token_rms_cap + 1.0e-5).any().item()):
        raise CPMRTensorContractError("pre-coordinate token RMS cap was violated")

    return CPMRNormalizationResult(
        activity=activity,
        phase_rms=phase_rms,
        normalized_content=normalized,
        token_rms_before_clip=token_rms,
        clip_scale=clip_scale,
        clipped_content=clipped,
        clip_fraction=clip_fraction,
    )


_COORDINATE_CPU_CACHE: Any | None = None


def _axis_coordinate_encoding(size: int, torch: Any) -> Any:
    rows: list[list[float]] = []
    for index in range(size):
        position = 2.0 * math.pi * float(index) / float(size - 1)
        row: list[float] = []
        for frequency_index in range(COORDINATE_FREQUENCIES):
            omega = COORDINATE_BASE ** (
                -2.0 * float(frequency_index) / float(COORDINATE_AXIS_WIDTH)
            )
            angle = position * omega
            row.extend((math.sin(angle), math.cos(angle)))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32, device="cpu")


def fixed_3d_coordinate_encoding(
    *,
    device: Any = None,
    config: CPMRConfig | None = None,
) -> Any:
    """Build the frozen unscaled ``E_coord`` tensor ``[1,21,8,8,1536]``.

    Scalar values are first generated on CPU from Python's IEEE-754 ``math``
    functions and cast once to FP32.  This prevents device-specific transcendental
    kernels from changing the receipt digest.  A fresh tensor is returned so a
    caller cannot mutate the canonical cache.
    """

    _config_or_default(config)
    torch = _require_torch()
    global _COORDINATE_CPU_CACHE
    if _COORDINATE_CPU_CACHE is None:
        time_axis = _axis_coordinate_encoding(LATENT_PHASES, torch)
        y_axis = _axis_coordinate_encoding(POOL_HEIGHT, torch)
        x_axis = _axis_coordinate_encoding(POOL_WIDTH, torch)
        time_field = time_axis[:, None, None, :].expand(
            LATENT_PHASES, POOL_HEIGHT, POOL_WIDTH, COORDINATE_AXIS_WIDTH
        )
        y_field = y_axis[None, :, None, :].expand(
            LATENT_PHASES, POOL_HEIGHT, POOL_WIDTH, COORDINATE_AXIS_WIDTH
        )
        x_field = x_axis[None, None, :, :].expand(
            LATENT_PHASES, POOL_HEIGHT, POOL_WIDTH, COORDINATE_AXIS_WIDTH
        )
        _COORDINATE_CPU_CACHE = (
            torch.cat((time_field, y_field, x_field), dim=-1)
            .unsqueeze(0)
            .contiguous()
        )
    output = _COORDINATE_CPU_CACHE.clone()
    if device is not None:
        output = output.to(device=device)
    return output


def coordinate_tensor_sha256() -> str:
    """Return the receipt digest of the canonical unscaled coordinate tensor."""

    return tensor_sha256(fixed_3d_coordinate_encoding(device="cpu"))


def _field_phase_rms_fp32(value: Any, torch: Any) -> Any:
    """Compute per-phase RMS over the fixed spatial and channel dimensions."""

    if value.dtype != torch.float32 or value.ndim != 5:
        raise CPMRTensorContractError("per-phase carrier RMS requires a 5-D FP32 field")
    with torch.no_grad():
        return value.square().mean(dim=(2, 3, 4)).sqrt()


@dataclass(frozen=True)
class CPMRCarrierResult:
    """Auditable FP32 carrier and the sufficient statistics used to construct it."""

    control: str
    config_sha256: str
    carrier_fp32: Any
    activity: Any
    clipped_content: Any
    phase_rms: Any
    token_rms_before_clip: Any
    clip_scale: Any
    clip_fraction: Any
    coordinate_encoding: Any
    source_phase_indices: tuple[int, ...]

    @property
    def carrier(self) -> Any:
        return self.carrier_fp32

    @property
    def batch_size(self) -> int:
        return int(self.carrier_fp32.shape[0])

    def flattened(self) -> Any:
        """Return ``[B,1344,1536]`` in fixed time-major/y-major/x-major order."""

        return self.carrier_fp32.reshape(self.batch_size, CARRIER_TOKENS, HIDDEN_SIZE)

    def bfloat16(self) -> Any:
        """Perform the only permitted post-audit carrier cast."""

        torch = _require_torch()
        result = self.carrier_fp32.to(dtype=torch.bfloat16)
        _assert_inactive_exact_zero("carrier_bfloat16", result, self.activity, torch)
        return result

    def digest_manifest(self) -> dict[str, str]:
        torch = _require_torch()
        clipped_phase_rms = _field_phase_rms_fp32(self.clipped_content, torch)
        final_phase_rms = _field_phase_rms_fp32(self.carrier_fp32, torch)
        pooled_phase_rms_digest = tensor_sha256(self.phase_rms)
        return {
            "activity_sha256": tensor_sha256(self.activity),
            "carrier_fp32_sha256": tensor_sha256(self.carrier_fp32),
            "carrier_bfloat16_sha256": tensor_sha256(self.bfloat16()),
            "clipped_content_fp32_sha256": tensor_sha256(self.clipped_content),
            # Keep the legacy phase-rms key while binding its precise meaning.
            "phase_rms_fp32_sha256": pooled_phase_rms_digest,
            "pooled_increment_phase_rms_fp32_sha256": pooled_phase_rms_digest,
            "clipped_content_phase_rms_fp32_sha256": tensor_sha256(
                clipped_phase_rms
            ),
            "final_carrier_phase_rms_fp32_sha256": tensor_sha256(final_phase_rms),
            "token_rms_before_clip_fp32_sha256": tensor_sha256(
                self.token_rms_before_clip
            ),
            "clip_scale_fp32_sha256": tensor_sha256(self.clip_scale),
            "clip_fraction_fp32_sha256": tensor_sha256(self.clip_fraction),
            "coordinate_encoding_fp32_sha256": tensor_sha256(
                self.coordinate_encoding
            ),
            "coordinate_spec_sha256": COORDINATE_SPEC_SHA256,
            "config_sha256": self.config_sha256,
            "phase_mapping_sha256": canonical_json_sha256(
                list(self.source_phase_indices[1:])
            ),
        }

    def audit_receipt(self) -> dict[str, Any]:
        torch = _require_torch()
        activity_cpu = self.activity.detach().to(device="cpu")
        pooled_phase_rms_cpu = self.phase_rms.detach().to(
            device="cpu", dtype=torch.float32
        )
        clipped_phase_rms_cpu = _field_phase_rms_fp32(
            self.clipped_content, torch
        ).to(device="cpu")
        final_phase_rms_cpu = _field_phase_rms_fp32(
            self.carrier_fp32, torch
        ).to(device="cpu")
        clip_fraction_cpu = self.clip_fraction.detach().to(
            device="cpu", dtype=torch.float32
        )
        bitsets = [
            "".join("1" if bool(item) else "0" for item in row.tolist())
            for row in activity_cpu
        ]
        return {
            "method": METHOD_NAME,
            "schema_version": SCHEMA_VERSION,
            "control": self.control,
            "shape": [int(item) for item in self.carrier_fp32.shape],
            "flattened_shape": [self.batch_size, CARRIER_TOKENS, HIDDEN_SIZE],
            "activity_bitset": bitsets,
            # Backward-compatible alias; its semantics are now explicit below.
            "phase_rms_fp32": pooled_phase_rms_cpu.tolist(),
            "phase_rms_fp32_semantics": "pooled_increment_pre_normalization",
            "pooled_increment_phase_rms_fp32": pooled_phase_rms_cpu.tolist(),
            "clipped_content_phase_rms_fp32": clipped_phase_rms_cpu.tolist(),
            "final_carrier_phase_rms_fp32": final_phase_rms_cpu.tolist(),
            "clip_fraction": clip_fraction_cpu.tolist(),
            "source_phase_indices": list(self.source_phase_indices),
            "digests": self.digest_manifest(),
        }


def _compose_result(
    *,
    control: str,
    activity: Any,
    clipped_content: Any,
    phase_rms: Any,
    token_rms_before_clip: Any,
    clip_scale: Any,
    clip_fraction: Any,
    source_phase_indices: Sequence[int],
    config: CPMRConfig,
) -> CPMRCarrierResult:
    torch = _require_torch()
    if control not in CONTROL_NAMES:
        raise CPMRTensorContractError(f"unknown CPMR control {control!r}")
    indices = tuple(int(item) for item in source_phase_indices)
    if len(indices) != config.latent_phases or indices[0] != 0:
        raise CPMRTensorContractError("a control phase mapping must have length 21 and fix phase 0")

    coordinate = fixed_3d_coordinate_encoding(
        device=clipped_content.device, config=config
    )
    activity_field = activity[:, :, None, None, None]
    with torch.no_grad():
        active_value = clipped_content + config.coordinate_scale * coordinate
        carrier = torch.where(
            activity_field, active_value, torch.zeros_like(active_value)
        ).contiguous()

    if carrier.dtype != torch.float32:
        raise CPMRTensorContractError("carrier composition escaped FP32")
    _ensure_finite("carrier_fp32", carrier, torch)
    _assert_inactive_exact_zero("carrier_fp32", carrier, activity, torch)
    _assert_inactive_exact_zero("clipped_content", clipped_content, activity, torch)
    if bool(activity[:, 0].any().item()):
        raise CPMRTensorContractError("phase-0 activity must be false")
    if int(torch.count_nonzero(carrier[:, 0]).item()) != 0:
        raise CPMRTensorContractError("phase-0 carrier must be exact zero")

    return CPMRCarrierResult(
        control=control,
        config_sha256=config.contract_sha256(),
        carrier_fp32=carrier,
        activity=activity,
        clipped_content=clipped_content,
        phase_rms=phase_rms,
        token_rms_before_clip=token_rms_before_clip,
        clip_scale=clip_scale,
        clip_fraction=clip_fraction,
        coordinate_encoding=coordinate,
        source_phase_indices=indices,
    )


def build_motion_carrier(
    action_patch_tokens: Any,
    noop_patch_tokens: Any,
    *,
    config: CPMRConfig | None = None,
) -> CPMRCarrierResult:
    """Build the canonical FP32 carrier from frozen action/no-op proposals.

    Both inputs are exact channel-last fields ``[B,21,31,30,1536]`` (an
    unbatched field is accepted and promoted).  The implementation detaches both
    proposal branches by design: CPMR is a frozen oracle, not a gradient path.
    """

    cfg = _config_or_default(config)
    torch = _require_torch()
    action = _coerce_patch_field("action_patch_tokens", action_patch_tokens, cfg, torch)
    noop = _coerce_patch_field("noop_patch_tokens", noop_patch_tokens, cfg, torch)
    if tuple(action.shape) != tuple(noop.shape):
        raise CPMRTensorContractError("action and no-op proposal shapes must match")
    if action.dtype != noop.dtype:
        raise CPMRTensorContractError("action and no-op proposal dtypes must match")
    if action.device != noop.device:
        raise CPMRTensorContractError("action and no-op proposals must share a device")

    with torch.no_grad():
        # C_t = U_a,t - U_0,t, explicitly in FP32.
        differences = action.detach().to(dtype=torch.float32).clone()
        differences.sub_(noop.detach().to(dtype=torch.float32))

        # Descending in-place traversal preserves every C[t-1] until it is read.
        # D_0 = 0 and D_t = C_t - C_{t-1} for t=1,...,20.
        for phase_index in range(cfg.latent_phases - 1, 0, -1):
            differences[:, phase_index].sub_(differences[:, phase_index - 1])
        differences[:, 0].zero_()

        batch_size = int(differences.shape[0])
        pool_input = differences.permute(0, 1, 4, 2, 3).reshape(
            batch_size * cfg.latent_phases,
            cfg.hidden_size,
            cfg.patch_height,
            cfg.patch_width,
        )
        pooled_cf = torch.nn.functional.adaptive_avg_pool2d(
            pool_input, (cfg.pool_height, cfg.pool_width)
        )
        pooled = (
            pooled_cf.reshape(
                batch_size,
                cfg.latent_phases,
                cfg.hidden_size,
                cfg.pool_height,
                cfg.pool_width,
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        # The exact proposal field is much larger than the 8 x 8 carrier.  Drop
        # its FP32 intermediates before normalization/control composition.
        del differences, pool_input, pooled_cf

    if pooled.dtype != torch.float32:
        raise CPMRTensorContractError("adaptive proposal pooling escaped FP32")
    normalized = normalize_and_clip_pooled_increments(pooled, config=cfg)
    return _compose_result(
        control="correct",
        activity=normalized.activity,
        clipped_content=normalized.clipped_content,
        phase_rms=normalized.phase_rms,
        token_rms_before_clip=normalized.token_rms_before_clip,
        clip_scale=normalized.clip_scale,
        clip_fraction=normalized.clip_fraction,
        source_phase_indices=IDENTITY_SOURCE_PHASES,
        config=cfg,
    )


def _validate_base_result(
    base: CPMRCarrierResult, config: CPMRConfig, torch: Any
) -> None:
    if not isinstance(base, CPMRCarrierResult):
        raise CPMRTensorContractError("base must be a CPMRCarrierResult")
    if base.control != "correct":
        raise CPMRTensorContractError("scientific controls must derive from the correct carrier")
    if base.config_sha256 != config.contract_sha256():
        raise CPMRTensorContractError("base carrier config digest does not match")
    if (
        not isinstance(base.source_phase_indices, tuple)
        or base.source_phase_indices != IDENTITY_SOURCE_PHASES
    ):
        raise CPMRTensorContractError(
            "a correct base carrier must retain the identity source-phase mapping"
        )

    if not isinstance(base.carrier_fp32, torch.Tensor) or base.carrier_fp32.ndim != 5:
        raise CPMRTensorContractError("base.carrier_fp32 must be a 5-D tensor")
    if base.carrier_fp32.device.type == "meta":
        raise CPMRTensorContractError("base carrier cannot be a meta tensor")
    batch_size = int(base.carrier_fp32.shape[0])
    if batch_size < 1:
        raise CPMRTensorContractError("base carrier batch must be positive")
    carrier_shape = (batch_size, *config.carrier_shape)
    phase_shape = (batch_size, config.latent_phases)
    token_shape = (
        batch_size,
        config.latent_phases,
        config.pool_height,
        config.pool_width,
    )
    coordinate_shape = (1, *config.carrier_shape)
    device = base.carrier_fp32.device
    specifications = (
        ("carrier_fp32", base.carrier_fp32, carrier_shape, torch.float32),
        ("activity", base.activity, phase_shape, torch.bool),
        ("clipped_content", base.clipped_content, carrier_shape, torch.float32),
        ("phase_rms", base.phase_rms, phase_shape, torch.float32),
        (
            "token_rms_before_clip",
            base.token_rms_before_clip,
            token_shape,
            torch.float32,
        ),
        ("clip_scale", base.clip_scale, token_shape, torch.float32),
        ("clip_fraction", base.clip_fraction, phase_shape, torch.float32),
        (
            "coordinate_encoding",
            base.coordinate_encoding,
            coordinate_shape,
            torch.float32,
        ),
    )
    for name, value, expected_shape, expected_dtype in specifications:
        if not isinstance(value, torch.Tensor):
            raise CPMRTensorContractError(f"base.{name} must be a torch.Tensor")
        if tuple(int(item) for item in value.shape) != expected_shape:
            raise CPMRTensorContractError(
                f"base.{name} shape must be {expected_shape}, got "
                f"{tuple(int(item) for item in value.shape)}"
            )
        if value.dtype != expected_dtype:
            raise CPMRTensorContractError(
                f"base.{name} dtype must be {expected_dtype}, got {value.dtype}"
            )
        if value.device != device:
            raise CPMRTensorContractError(
                f"base.{name} must share carrier device {device}, got {value.device}"
            )
        if expected_dtype != torch.bool:
            if bool(value.requires_grad):
                raise CPMRTensorContractError(
                    f"base.{name} must be detached from autograd"
                )
            _ensure_finite(f"base.{name}", value, torch)

    if bool(base.activity[:, 0].any().item()):
        raise CPMRTensorContractError("base phase-0 activity must be false")
    derived_activity = base.clipped_content.abs().amax(dim=(2, 3, 4)) > 0
    if not _tensor_bytes_equal(base.activity, derived_activity, torch):
        raise CPMRTensorContractError(
            "base activity does not exactly match nonzero clipped-content phases"
        )

    _assert_inactive_exact_zero(
        "base.carrier_fp32", base.carrier_fp32, base.activity, torch
    )
    _assert_inactive_exact_zero(
        "base.clipped_content", base.clipped_content, base.activity, torch
    )
    _assert_inactive_exact_zero("base.phase_rms", base.phase_rms, base.activity, torch)
    _assert_inactive_exact_zero(
        "base.token_rms_before_clip",
        base.token_rms_before_clip,
        base.activity,
        torch,
    )
    _assert_inactive_exact_zero(
        "base.clip_fraction", base.clip_fraction, base.activity, torch
    )

    for batch_index in range(batch_size):
        inactive = torch.nonzero(~base.activity[batch_index], as_tuple=False).flatten()
        for phase_index in inactive.tolist():
            scale = base.clip_scale[batch_index, phase_index]
            if not _tensor_bytes_equal(scale, torch.ones_like(scale), torch):
                raise CPMRTensorContractError(
                    "base.clip_scale must be byte-exact one at every inactive phase"
                )

    if bool((base.phase_rms < 0).any().item()):
        raise CPMRTensorContractError("base.phase_rms cannot be negative")
    if bool((base.token_rms_before_clip < 0).any().item()):
        raise CPMRTensorContractError("base.token_rms_before_clip cannot be negative")
    if bool((base.clip_scale < 1.0).any().item()):
        raise CPMRTensorContractError("base.clip_scale cannot be smaller than one")
    if bool(((base.clip_fraction < 0) | (base.clip_fraction > 1)).any().item()):
        raise CPMRTensorContractError("base.clip_fraction must lie in [0,1]")

    with torch.no_grad():
        expected_clip_scale = torch.clamp(
            base.token_rms_before_clip / config.token_rms_cap, min=1.0
        )
        expected_clip_fraction = (expected_clip_scale > 1.0).to(
            torch.float32
        ).mean(dim=(2, 3))
        expected_clip_fraction = torch.where(
            base.activity,
            expected_clip_fraction,
            torch.zeros_like(expected_clip_fraction),
        )
        clipped_token_rms = base.clipped_content.square().mean(dim=-1).sqrt()
        expected_clipped_token_rms = (
            base.token_rms_before_clip / expected_clip_scale
        )

    if not _tensor_bytes_equal(base.clip_scale, expected_clip_scale, torch):
        raise CPMRTensorContractError(
            "base.clip_scale does not match the frozen token-RMS clipping rule"
        )
    if not _tensor_bytes_equal(base.clip_fraction, expected_clip_fraction, torch):
        raise CPMRTensorContractError(
            "base.clip_fraction does not match the frozen clipping diagnostics"
        )
    if not bool(
        torch.allclose(
            clipped_token_rms,
            expected_clipped_token_rms,
            rtol=2.0e-5,
            atol=1.0e-7,
        )
    ):
        raise CPMRTensorContractError(
            "base.clipped_content is inconsistent with token RMS and clip scale"
        )
    if bool((clipped_token_rms > config.token_rms_cap + 1.0e-5).any().item()):
        raise CPMRTensorContractError(
            "base.clipped_content violates the pre-coordinate RMS cap"
        )

    canonical_coordinate = fixed_3d_coordinate_encoding(device=device, config=config)
    if not _tensor_bytes_equal(
        base.coordinate_encoding, canonical_coordinate, torch
    ):
        raise CPMRTensorContractError(
            "base.coordinate_encoding does not match the frozen coordinate bytes"
        )

    with torch.no_grad():
        active_value = (
            base.clipped_content + config.coordinate_scale * canonical_coordinate
        )
        reconstructed_carrier = torch.where(
            base.activity[:, :, None, None, None],
            active_value,
            torch.zeros_like(active_value),
        ).contiguous()
    if not _tensor_bytes_equal(
        base.carrier_fp32, reconstructed_carrier, torch
    ):
        raise CPMRTensorContractError(
            "base.carrier_fp32 is not byte-exact to "
            "where(activity, clipped_content + 0.02 * coordinate, 0)"
        )


def _phase_rebind(
    base: CPMRCarrierResult,
    source_phase_indices: tuple[int, ...],
    *,
    control: str,
    config: CPMRConfig,
) -> CPMRCarrierResult:
    torch = _require_torch()
    _validate_base_result(base, config, torch)
    index = torch.tensor(
        source_phase_indices, dtype=torch.long, device=base.carrier_fp32.device
    )
    # Activity and clipped content move together.  Coordinates are intentionally
    # absent here; _compose_result regenerates them at fixed destination slots.
    return _compose_result(
        control=control,
        activity=base.activity.index_select(1, index).clone(),
        clipped_content=base.clipped_content.index_select(1, index).clone(),
        phase_rms=base.phase_rms.index_select(1, index).clone(),
        token_rms_before_clip=base.token_rms_before_clip.index_select(1, index).clone(),
        clip_scale=base.clip_scale.index_select(1, index).clone(),
        clip_fraction=base.clip_fraction.index_select(1, index).clone(),
        source_phase_indices=source_phase_indices,
        config=config,
    )


def build_reverse_control(
    base: CPMRCarrierResult, *, config: CPMRConfig | None = None
) -> CPMRCarrierResult:
    """REV: destination phases 1..20 receive source phases 20..1."""

    cfg = _config_or_default(config)
    return _phase_rebind(
        base, REVERSE_SOURCE_PHASES, control="reverse", config=cfg
    )


def build_shuffle_control(
    base: CPMRCarrierResult, *, config: CPMRConfig | None = None
) -> CPMRCarrierResult:
    """SHUF: apply the frozen V11 20-phase permutation and fixed coordinates."""

    cfg = _config_or_default(config)
    return _phase_rebind(
        base, SHUFFLE_SOURCE_PHASES, control="shuffle", config=cfg
    )


def build_negative_control(
    base: CPMRCarrierResult, *, config: CPMRConfig | None = None
) -> CPMRCarrierResult:
    """NEG: preserve activity, negate only clipped content, keep coordinates."""

    cfg = _config_or_default(config)
    torch = _require_torch()
    _validate_base_result(base, cfg, torch)
    with torch.no_grad():
        negated = torch.where(
            base.activity[:, :, None, None, None],
            -base.clipped_content,
            torch.zeros_like(base.clipped_content),
        )
    return _compose_result(
        control="negative",
        activity=base.activity.clone(),
        clipped_content=negated,
        phase_rms=base.phase_rms.clone(),
        token_rms_before_clip=base.token_rms_before_clip.clone(),
        clip_scale=base.clip_scale.clone(),
        clip_fraction=base.clip_fraction.clone(),
        source_phase_indices=IDENTITY_SOURCE_PHASES,
        config=cfg,
    )


def build_nn_control(
    first_noop_patch_tokens: Any,
    repeated_noop_patch_tokens: Any | None = None,
    *,
    config: CPMRConfig | None = None,
) -> CPMRCarrierResult:
    """NN: identical repeated no-op proposals, with exact-zero activity/carrier."""

    cfg = _config_or_default(config)
    second = (
        first_noop_patch_tokens
        if repeated_noop_patch_tokens is None
        else repeated_noop_patch_tokens
    )
    result = build_motion_carrier(first_noop_patch_tokens, second, config=cfg)
    torch = _require_torch()
    if bool(result.activity.any().item()):
        raise CPMRTensorContractError("NN proposals produced nonzero activity")
    if int(torch.count_nonzero(result.carrier_fp32).item()) != 0:
        raise CPMRTensorContractError("NN carrier is not bit-exact zero")
    if int(torch.count_nonzero(result.clipped_content).item()) != 0:
        raise CPMRTensorContractError("NN clipped content is not bit-exact zero")
    return replace(result, control="nn")


def build_control(
    base: CPMRCarrierResult,
    control: str,
    *,
    config: CPMRConfig | None = None,
) -> CPMRCarrierResult:
    """Dispatch a non-NN scientific control from a correct carrier."""

    normalized = str(control).strip().lower()
    if normalized in ("correct", "identity"):
        cfg = _config_or_default(config)
        _validate_base_result(base, cfg, _require_torch())
        return base
    if normalized in ("reverse", "rev"):
        return build_reverse_control(base, config=config)
    if normalized in ("shuffle", "shuf"):
        return build_shuffle_control(base, config=config)
    if normalized in ("negative", "neg"):
        return build_negative_control(base, config=config)
    if normalized in ("nn", "noop-noop"):
        raise CPMRTensorContractError(
            "NN requires two proposal tensors; call build_nn_control"
        )
    raise CPMRTensorContractError(f"unknown CPMR control {control!r}")


__all__ = [
    "CANONICAL_CONFIG",
    "CANONICAL_COORDINATE_TENSOR_SHA256",
    "CARRIER_TOKENS",
    "CONFIG_CONTRACT_SHA256",
    "CONTROL_NAMES",
    "COORDINATE_SPEC",
    "COORDINATE_SPEC_SHA256",
    "CPMRCarrierResult",
    "CPMRConfig",
    "CPMRNormalizationResult",
    "CPMRTensorContractError",
    "IDENTITY_SOURCE_PHASES",
    "REVERSE_SOURCE_PHASES",
    "SHUFFLE_CANONICAL_JSON",
    "SHUFFLE_CANONICAL_SHA256",
    "SHUFFLE_SOURCE_PHASES",
    "SHUFFLE_SOURCE_PHASES_NONZERO",
    "build_control",
    "build_motion_carrier",
    "build_negative_control",
    "build_nn_control",
    "build_reverse_control",
    "build_shuffle_control",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "coordinate_tensor_sha256",
    "fixed_3d_coordinate_encoding",
    "normalize_and_clip_pooled_increments",
    "reshape_proposal_patch_tokens",
    "tensor_sha256",
]
