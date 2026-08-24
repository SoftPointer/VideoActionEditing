#!/usr/bin/env python3
"""Renderer-only action-plan ABI for the 2026-08-17 Bernini program.

``ActionPlanPredictorV1`` has exactly two semantic inputs: complete clean
``[B,T,H,W,C]`` source patch grids and complete contextual instruction tokens.  Targets,
generated anchors, tracks, poses, masks, and other training-only annotations
are deliberately absent from its forward signature.

The formal profile follows ``md/action_editing/20260817_man``: two learned
actor/object queries softly pool the source, twenty-one phase queries and one
global query pass through six 512-wide, eight-head cross-attention blocks with
2048-wide MLPs, and the outputs are 21x256 phase tokens plus one 256-wide
global token.  A separately persisted target-only injection concatenates each
phase token with the global token and applies one of thirty block-specific,
exactly-zero-initialized projections into the Bernini hidden width.
Consequently step zero is an identity, all projection heads receive bootstrap
gradients when all thirty renderer blocks are traversed, and gradients reach
the predictor core after those projections have taken one update.  Injection
requires a digest-bound certificate issued from a closed source-prefix/target-
suffix route; tensor shape alone is never treated as ownership evidence.

The explicit ``cpu-test-v1`` profile preserves the same output and state-key
semantics while permitting a smaller internal width/depth.  It is rejected by
``require_formal_0817`` and must never be promoted as a formal checkpoint.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping


ARCHITECTURE_NAME = "bernini-action-plan-predictor-v1"
PREDICTOR_ABI_SCHEMA = "bernini-action-plan-predictor-state-dict-abi-v1"
INJECTION_ABI_SCHEMA = "bernini-zero-init-target-only-action-injection-abi-v1"
CONDITIONER_ABI_SCHEMA = "bernini-action-plan-conditioner-state-dict-abi-v1"
TARGET_OWNERSHIP_SCHEMA = "bernini-closed-target-suffix-ownership-v1"
INJECTION_ROUTE_SCHEMA = "bernini-action-plan-injection-route-v1"
SOURCE_POSITION_ENCODING_SCHEMA = "deterministic-source-thw-position-v1"
PHASE_POSITION_ENCODING_SCHEMA = "deterministic-phase-sinusoidal-position-v1"
COMPUTE_DTYPE_POLICY = "internal-fp32-output-dtype-restoration-v1"
ZERO_INIT_GATE_SEMANTICS = "block-projection-is-zero-init-residual-gate-v1"
TARGET_SUFFIX_ROUTE_SEMANTICS = "closed-source-prefix-then-target-suffix-v1"
FORMAL_PROFILE = "0817-formal-v1"
CPU_TEST_PROFILE = "cpu-test-v1"

FORMAL_SOURCE_TOKEN_WIDTH = 1536
FORMAL_INSTRUCTION_TOKEN_WIDTH = 4096
FORMAL_MODEL_WIDTH = 512
FORMAL_ATTENTION_HEADS = 8
FORMAL_MLP_WIDTH = 2048
FORMAL_LAYER_COUNT = 6
PHASE_COUNT = 21
ACTION_WIDTH = 256
ACTOR_OBJECT_QUERY_COUNT = 2
FORMAL_RENDERER_HIDDEN_WIDTH = 1536
TRANSFORMER_BLOCK_COUNT = 30


class ActionPlanPredictorError(ValueError):
    """Raised when the persisted action-plan ABI is violated."""


@dataclass(frozen=True)
class ActionPlanPredictorConfig:
    """Shape-complete configuration persisted by the predictor state ABI."""

    profile: str = FORMAL_PROFILE
    source_token_width: int = FORMAL_SOURCE_TOKEN_WIDTH
    instruction_token_width: int = FORMAL_INSTRUCTION_TOKEN_WIDTH
    model_width: int = FORMAL_MODEL_WIDTH
    attention_heads: int = FORMAL_ATTENTION_HEADS
    mlp_width: int = FORMAL_MLP_WIDTH
    layer_count: int = FORMAL_LAYER_COUNT
    phase_count: int = PHASE_COUNT
    action_width: int = ACTION_WIDTH
    actor_object_query_count: int = ACTOR_OBJECT_QUERY_COUNT

    def validate(self) -> None:
        if self.profile not in (FORMAL_PROFILE, CPU_TEST_PROFILE):
            raise ActionPlanPredictorError("unknown action-plan predictor profile")
        for name in (
            "source_token_width",
            "instruction_token_width",
            "model_width",
            "attention_heads",
            "mlp_width",
            "layer_count",
            "phase_count",
            "action_width",
            "actor_object_query_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ActionPlanPredictorError(f"{name} must be a positive integer")
        if self.model_width % self.attention_heads:
            raise ActionPlanPredictorError(
                "model_width must be divisible by attention_heads"
            )
        if self.phase_count != PHASE_COUNT:
            raise ActionPlanPredictorError("V1 requires exactly 21 phases")
        if self.action_width != ACTION_WIDTH:
            raise ActionPlanPredictorError("V1 requires 256-wide action tokens")
        if self.actor_object_query_count != ACTOR_OBJECT_QUERY_COUNT:
            raise ActionPlanPredictorError(
                "V1 requires exactly one actor and one object pooling query"
            )
        if self.profile == FORMAL_PROFILE:
            observed = (
                self.source_token_width,
                self.instruction_token_width,
                self.model_width,
                self.attention_heads,
                self.mlp_width,
                self.layer_count,
            )
            expected = (
                FORMAL_SOURCE_TOKEN_WIDTH,
                FORMAL_INSTRUCTION_TOKEN_WIDTH,
                FORMAL_MODEL_WIDTH,
                FORMAL_ATTENTION_HEADS,
                FORMAL_MLP_WIDTH,
                FORMAL_LAYER_COUNT,
            )
            if observed != expected:
                raise ActionPlanPredictorError(
                    "0817-formal-v1 architecture dimensions differ"
                )

    def require_formal_0817(self) -> None:
        self.validate()
        if self.profile != FORMAL_PROFILE:
            raise ActionPlanPredictorError(
                "cpu-test-v1 configuration cannot be used as a formal 0817 model"
            )

    def to_abi_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ActionPlanOutput:
    """The only product action representation emitted by the predictor."""

    phase_tokens: Any
    global_token: Any


@dataclass(frozen=True)
class ConditionedTargetOutput:
    """A target-only hidden tensor paired with the plan that conditioned it."""

    target_hidden: Any
    plan: ActionPlanOutput


@dataclass(frozen=True)
class TargetOwnershipCertificate:
    """Digest-bound claim for one closed source-prefix/target-suffix layout."""

    schema_version: str
    route_semantics: str
    compute_dtype_policy: str
    target_only: bool
    finite_audited: bool
    source_prefix_start: int
    source_prefix_stop: int
    target_suffix_start: int
    target_suffix_stop: int
    packed_total_tokens: int
    target_shape: tuple[int, ...]
    target_dtype: str
    target_device_type: str
    digest: str


@dataclass(frozen=True)
class ActionPlanInjectionRoute:
    """One finite-audited plan bound to one target-ownership certificate."""

    schema_version: str
    plan: ActionPlanOutput
    ownership: TargetOwnershipCertificate
    finite_audited: bool
    metadata_digest: str


def expected_predictor_parameter_count(config: ActionPlanPredictorConfig) -> int:
    """Return the exact trainable parameter count without importing PyTorch."""

    config.validate()
    width = config.model_width
    mlp = config.mlp_width
    per_block = 4 * width * width + 2 * width * mlp + 11 * width + mlp
    outside_blocks = width * (
        config.source_token_width
        + config.instruction_token_width
        + 2 * config.action_width
        + config.phase_count
        + config.actor_object_query_count
        + 9
    ) + 2 * config.action_width
    return outside_blocks + config.layer_count * per_block


def expected_injection_parameter_count(
    *,
    action_width: int = ACTION_WIDTH,
    hidden_width: int,
    block_count: int = TRANSFORMER_BLOCK_COUNT,
) -> int:
    """Return the exact count of all block-specific output projections."""

    if type(action_width) is not int or action_width <= 0:
        raise ActionPlanPredictorError("action_width must be a positive integer")
    if type(hidden_width) is not int or hidden_width <= 0:
        raise ActionPlanPredictorError("hidden_width must be a positive integer")
    if block_count != TRANSFORMER_BLOCK_COUNT:
        raise ActionPlanPredictorError("V1 requires exactly 30 injection blocks")
    return block_count * ((2 * action_width) * hidden_width + hidden_width)


def expected_conditioner_parameter_count(
    config: ActionPlanPredictorConfig,
    *, renderer_hidden_width: int = FORMAL_RENDERER_HIDDEN_WIDTH,
) -> int:
    config.validate()
    if (
        config.profile == FORMAL_PROFILE
        and renderer_hidden_width != FORMAL_RENDERER_HIDDEN_WIDTH
    ):
        raise ActionPlanPredictorError(
            "formal conditioner inventory requires renderer hidden width 1536"
        )
    return expected_predictor_parameter_count(config) + expected_injection_parameter_count(
        action_width=config.action_width,
        hidden_width=renderer_hidden_width,
        block_count=TRANSFORMER_BLOCK_COUNT,
    )


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _profile_code(profile: str) -> int:
    if profile == FORMAL_PROFILE:
        return 1
    if profile == CPU_TEST_PROFILE:
        return 2
    raise ActionPlanPredictorError("unknown action-plan predictor profile")


def _dtype_name(value: Any) -> str:
    name = str(value)
    return name[6:] if name.startswith("torch.") else name


def _ownership_body(certificate: TargetOwnershipCertificate) -> dict[str, Any]:
    return {
        "schema_version": certificate.schema_version,
        "route_semantics": certificate.route_semantics,
        "compute_dtype_policy": certificate.compute_dtype_policy,
        "target_only": certificate.target_only,
        "finite_audited": certificate.finite_audited,
        "source_prefix_start": certificate.source_prefix_start,
        "source_prefix_stop": certificate.source_prefix_stop,
        "target_suffix_start": certificate.target_suffix_start,
        "target_suffix_stop": certificate.target_suffix_stop,
        "packed_total_tokens": certificate.packed_total_tokens,
        "target_shape": list(certificate.target_shape),
        "target_dtype": certificate.target_dtype,
        "target_device_type": certificate.target_device_type,
    }


def _route_metadata_body(
    plan: ActionPlanOutput, ownership: TargetOwnershipCertificate
) -> dict[str, Any]:
    return {
        "schema_version": INJECTION_ROUTE_SCHEMA,
        "ownership_digest": ownership.digest,
        "phase_shape": [int(value) for value in plan.phase_tokens.shape],
        "global_shape": [int(value) for value in plan.global_token.shape],
        "phase_dtype": _dtype_name(plan.phase_tokens.dtype),
        "global_dtype": _dtype_name(plan.global_token.dtype),
        "phase_device_type": plan.phase_tokens.device.type,
        "global_device_type": plan.global_token.device.type,
        "phase_object_id": id(plan.phase_tokens),
        "global_object_id": id(plan.global_token),
        "phase_mutation_version": int(plan.phase_tokens._version),
        "global_mutation_version": int(plan.global_token._version),
        "compute_dtype_policy": COMPUTE_DTYPE_POLICY,
    }


try:
    import torch
    from torch import nn

    _SUPPORTED_COMPUTE_INPUT_DTYPES = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    )


    def _tensor_value_sha256(value: Any) -> str:
        tensor = value.detach().cpu().contiguous()
        raw = bytes(tensor.view(torch.uint8).reshape(-1).tolist())
        header = {
            "dtype": _dtype_name(tensor.dtype),
            "shape": [int(item) for item in tensor.shape],
        }
        return hashlib.sha256(_canonical_json(header) + b"\0" + raw).hexdigest()


    def _preflight_state_dict_before_load(
        module: nn.Module, state_dict: Mapping[str, Any]
    ) -> None:
        """Reject dtype or semantic-buffer drift before ``copy_`` can cast."""

        expected_state = module.state_dict()
        parameter_names = {name for name, _ in module.named_parameters()}
        problems = []
        for name, expected in expected_state.items():
            incoming = state_dict.get(name)
            if incoming is None or not isinstance(incoming, torch.Tensor):
                continue
            if incoming.dtype != expected.dtype:
                problems.append(
                    f"{name} dtype {_dtype_name(incoming.dtype)} != "
                    f"{_dtype_name(expected.dtype)}"
                )
                continue
            if name not in parameter_names and not torch.equal(
                incoming.detach().cpu(), expected.detach().cpu()
            ):
                problems.append(f"{name} semantic buffer value differs")
        if problems:
            raise RuntimeError(
                "state_dict preflight rejected before load/cast: "
                + "; ".join(problems)
            )


    def _load_state_dict_after_preflight(
        module: nn.Module,
        state_dict: Mapping[str, Any],
        *,
        strict: bool,
        assign: bool,
    ) -> Any:
        _preflight_state_dict_before_load(module, state_dict)
        try:
            return super(type(module), module).load_state_dict(
                state_dict, strict=strict, assign=assign
            )
        except TypeError:
            if assign:
                raise
            return super(type(module), module).load_state_dict(
                state_dict, strict=strict
            )


    def _require_fp32_trainables(module: nn.Module, *, label: str) -> None:
        wrong = [
            name
            for name, parameter in module.named_parameters()
            if parameter.dtype != torch.float32
        ]
        if wrong:
            raise ActionPlanPredictorError(
                f"{label} trainables must remain FP32 under {COMPUTE_DTYPE_POLICY}: "
                + ",".join(wrong[:4])
            )


    def _disabled_ambient_autocast(device_type: str) -> Any:
        try:
            return torch.autocast(device_type=device_type, enabled=False)
        except (RuntimeError, TypeError):
            return nullcontext()


    def _normalized_axis(length: int, *, device: Any) -> Any:
        if type(length) is not int or length <= 0:
            raise ActionPlanPredictorError("source THW axes must be positive")
        if length == 1:
            return torch.zeros(1, dtype=torch.float32, device=device)
        return torch.linspace(
            -1.0, 1.0, length, dtype=torch.float32, device=device
        )


    def deterministic_source_position_encoding(
        temporal: int,
        height: int,
        width: int,
        channels: int,
        *,
        device: Any,
    ) -> Any:
        """Return deterministic, external-annotation-free ``[T,H,W,C]`` codes."""

        if type(channels) is not int or channels <= 0:
            raise ActionPlanPredictorError("position channels must be positive")
        t = _normalized_axis(temporal, device=device)
        y = _normalized_axis(height, device=device)
        x = _normalized_axis(width, device=device)
        tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
        coordinates = torch.stack((tt, yy, xx), dim=-1)
        channel = torch.arange(channels, dtype=torch.int64, device=device)
        axis = channel.remainder(3)
        use_cosine = channel.div(3, rounding_mode="floor").remainder(2).bool()
        band = channel.div(6, rounding_mode="floor").to(torch.float32)
        band_count = max((channels + 5) // 6, 1)
        denominator = max(band_count - 1, 1)
        frequency = torch.exp(-math.log(10000.0) * band / denominator)
        selected = coordinates[..., axis]
        angle = math.pi * selected * frequency
        return torch.where(use_cosine, torch.cos(angle), torch.sin(angle))


    def _validate_ownership_certificate(
        certificate: TargetOwnershipCertificate,
        *,
        target_hidden: Any | None = None,
    ) -> None:
        if not isinstance(certificate, TargetOwnershipCertificate):
            raise ActionPlanPredictorError(
                "target injection requires a TargetOwnershipCertificate"
            )
        if (
            certificate.schema_version != TARGET_OWNERSHIP_SCHEMA
            or certificate.route_semantics != TARGET_SUFFIX_ROUTE_SEMANTICS
            or certificate.compute_dtype_policy != COMPUTE_DTYPE_POLICY
            or certificate.target_only is not True
            or certificate.finite_audited is not True
        ):
            raise ActionPlanPredictorError(
                "target ownership certificate semantics differ"
            )
        integer_fields = (
            certificate.source_prefix_start,
            certificate.source_prefix_stop,
            certificate.target_suffix_start,
            certificate.target_suffix_stop,
            certificate.packed_total_tokens,
        )
        if any(type(value) is not int for value in integer_fields):
            raise ActionPlanPredictorError(
                "target ownership boundaries must be exact integers"
            )
        if (
            type(certificate.target_shape) is not tuple
            or len(certificate.target_shape) < 3
            or any(
                type(value) is not int or value <= 0
                for value in certificate.target_shape
            )
            or certificate.target_shape[1] != PHASE_COUNT
        ):
            raise ActionPlanPredictorError(
                "target ownership canonical shape differs"
            )
        if certificate.target_dtype not in {
            "float16",
            "bfloat16",
            "float32",
        } or (
            not isinstance(certificate.target_device_type, str)
            or not certificate.target_device_type
        ):
            raise ActionPlanPredictorError(
                "target ownership dtype/device semantics differ"
            )
        if not (
            certificate.source_prefix_start == 0
            and certificate.source_prefix_stop > 0
            and certificate.target_suffix_start
            == certificate.source_prefix_stop
            and certificate.target_suffix_stop
            == certificate.packed_total_tokens
            and certificate.target_suffix_stop
            > certificate.target_suffix_start
            and certificate.target_suffix_stop
            - certificate.target_suffix_start
            == math.prod(certificate.target_shape[1:-1])
        ):
            raise ActionPlanPredictorError(
                "target ownership is not one closed prefix/suffix partition"
            )
        expected_digest = hashlib.sha256(
            _canonical_json(_ownership_body(certificate))
        ).hexdigest()
        if certificate.digest != expected_digest:
            raise ActionPlanPredictorError(
                "target ownership certificate digest differs"
            )
        if target_hidden is not None:
            if not isinstance(target_hidden, torch.Tensor):
                raise ActionPlanPredictorError("certified target must be a tensor")
            if tuple(target_hidden.shape) != certificate.target_shape:
                raise ActionPlanPredictorError(
                    "target tensor shape differs from ownership certificate"
                )
            if _dtype_name(target_hidden.dtype) != certificate.target_dtype:
                raise ActionPlanPredictorError(
                    "target tensor dtype differs from ownership certificate"
                )
            if target_hidden.device.type != certificate.target_device_type:
                raise ActionPlanPredictorError(
                    "target tensor device differs from ownership certificate"
                )


    def certify_closed_target_suffix_route(
        target_hidden: Any,
        *,
        source_prefix_tokens: int,
        packed_total_tokens: int,
        audit_finite: bool = True,
    ) -> TargetOwnershipCertificate:
        """Issue a certificate after a runner closes prefix/suffix ownership."""

        _require_float_tensor(target_hidden, label="target_hidden", ndim=None)
        if target_hidden.ndim < 3:
            raise ActionPlanPredictorError(
                "target_hidden must be canonical [B,21,...,hidden]"
            )
        if int(target_hidden.shape[0]) <= 0 or int(target_hidden.shape[1]) != PHASE_COUNT:
            raise ActionPlanPredictorError(
                "target_hidden must have exactly 21 phases on axis 1"
            )
        if type(source_prefix_tokens) is not int or source_prefix_tokens <= 0:
            raise ActionPlanPredictorError(
                "source_prefix_tokens must be a positive exact integer"
            )
        target_suffix_tokens = math.prod(int(value) for value in target_hidden.shape[1:-1])
        if (
            type(packed_total_tokens) is not int
            or packed_total_tokens
            != source_prefix_tokens + target_suffix_tokens
        ):
            raise ActionPlanPredictorError(
                "packed token count does not close source-prefix/target-suffix route"
            )
        if audit_finite is not True:
            raise ActionPlanPredictorError(
                "ownership issuance requires one explicit finite smoke audit"
            )
        if not bool(torch.isfinite(target_hidden).all()):
            raise ActionPlanPredictorError(
                "target hidden failed ownership-bound finite audit"
            )
        partial = TargetOwnershipCertificate(
            schema_version=TARGET_OWNERSHIP_SCHEMA,
            route_semantics=TARGET_SUFFIX_ROUTE_SEMANTICS,
            compute_dtype_policy=COMPUTE_DTYPE_POLICY,
            target_only=True,
            finite_audited=True,
            source_prefix_start=0,
            source_prefix_stop=source_prefix_tokens,
            target_suffix_start=source_prefix_tokens,
            target_suffix_stop=packed_total_tokens,
            packed_total_tokens=packed_total_tokens,
            target_shape=tuple(int(value) for value in target_hidden.shape),
            target_dtype=_dtype_name(target_hidden.dtype),
            target_device_type=target_hidden.device.type,
            digest="",
        )
        digest = hashlib.sha256(_canonical_json(_ownership_body(partial))).hexdigest()
        certificate = TargetOwnershipCertificate(
            **{**asdict(partial), "digest": digest}
        )
        _validate_ownership_certificate(certificate, target_hidden=target_hidden)
        return certificate

    def _sinusoidal_phase_encoding(phases: int, width: int) -> Any:
        half = max(width // 2, 1)
        position = torch.arange(phases, dtype=torch.float32).unsqueeze(1)
        denominator = max(half - 1, 1)
        frequency = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32)
            / denominator
        ).unsqueeze(0)
        encoding = torch.cat(
            (torch.sin(position * frequency), torch.cos(position * frequency)),
            dim=1,
        )
        if int(encoding.shape[1]) < width:
            encoding = torch.nn.functional.pad(
                encoding, (0, width - int(encoding.shape[1]))
            )
        return encoding[:, :width].contiguous()


    def _require_float_tensor(
        value: Any, *, label: str, ndim: int | None
    ) -> None:
        if not isinstance(value, torch.Tensor) or (
            ndim is not None and value.ndim != ndim
        ):
            expected = "a floating torch tensor" if ndim is None else f"a rank-{ndim} torch tensor"
            raise ActionPlanPredictorError(
                f"{label} must be {expected}"
            )
        if not value.is_floating_point() or value.dtype not in _SUPPORTED_COMPUTE_INPUT_DTYPES:
            raise ActionPlanPredictorError(
                f"{label} dtype is outside {COMPUTE_DTYPE_POLICY}"
            )


    class _ActionPlanCrossAttentionBlock(nn.Module):
        """One pre-norm phase/global-query cross-attention and MLP block."""

        def __init__(self, *, width: int, heads: int, mlp_width: int):
            super().__init__()
            self.query_norm = nn.LayerNorm(width)
            self.memory_norm = nn.LayerNorm(width)
            self.cross_attention = nn.MultiheadAttention(
                width,
                heads,
                dropout=0.0,
                batch_first=True,
            )
            self.mlp_norm = nn.LayerNorm(width)
            self.mlp = nn.Sequential(
                nn.Linear(width, mlp_width),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_width, width),
            )

        def forward(self, query_states: Any, memory: Any) -> Any:
            normalized_memory = self.memory_norm(memory)
            attended, _ = self.cross_attention(
                self.query_norm(query_states),
                normalized_memory,
                normalized_memory,
                need_weights=False,
            )
            query_states = query_states + attended
            return query_states + self.mlp(self.mlp_norm(query_states))


    class ActionPlanPredictorV1(nn.Module):
        """Predict 21 phase tokens and one global token from source+instruction."""

        architecture = ARCHITECTURE_NAME

        def __init__(self, config: ActionPlanPredictorConfig | None = None):
            super().__init__()
            self.config = config or ActionPlanPredictorConfig()
            self.config.validate()
            width = self.config.model_width

            self.source_projection = nn.Linear(
                self.config.source_token_width, width
            )
            self.source_norm = nn.LayerNorm(width)
            self.instruction_projection = nn.Linear(
                self.config.instruction_token_width, width
            )
            self.instruction_norm = nn.LayerNorm(width)
            self.actor_object_queries = nn.Parameter(
                torch.empty(self.config.actor_object_query_count, width)
            )
            self.phase_queries = nn.Parameter(
                torch.empty(self.config.phase_count, width)
            )
            self.global_query = nn.Parameter(torch.empty(1, width))
            nn.init.normal_(self.actor_object_queries, std=width ** -0.5)
            nn.init.normal_(self.phase_queries, std=width ** -0.5)
            nn.init.normal_(self.global_query, std=width ** -0.5)

            self.register_buffer(
                "phase_position_encoding",
                _sinusoidal_phase_encoding(self.config.phase_count, width),
                # This table is a deterministic runtime derivative of the
                # integer ABI below.  Persisting and raw-byte hashing the
                # floating result makes an otherwise identical checkpoint
                # ABI depend on PyTorch/libm rounding details.  Registered
                # non-persistently so ``module.to(device)`` still moves it.
                persistent=False,
            )
            self.register_buffer(
                "phase_position_encoding_abi",
                torch.tensor(
                    (1, self.config.phase_count, width, 10000),
                    dtype=torch.int64,
                ),
                persistent=True,
            )
            self.register_buffer(
                "source_position_encoding_abi",
                torch.tensor((1, 5, 3, 6, 10000), dtype=torch.int64),
                persistent=True,
            )
            self.register_buffer(
                "_abi_config",
                torch.tensor(
                    (
                        3,
                        _profile_code(self.config.profile),
                        5,
                        1,
                        self.config.source_token_width,
                        self.config.instruction_token_width,
                        width,
                        self.config.attention_heads,
                        self.config.mlp_width,
                        self.config.layer_count,
                        self.config.phase_count,
                        self.config.action_width,
                        self.config.actor_object_query_count,
                    ),
                    dtype=torch.int64,
                ),
                persistent=True,
            )
            self.blocks = nn.ModuleList(
                _ActionPlanCrossAttentionBlock(
                    width=width,
                    heads=self.config.attention_heads,
                    mlp_width=self.config.mlp_width,
                )
                for _ in range(self.config.layer_count)
            )
            self.output_norm = nn.LayerNorm(width)
            self.phase_output = nn.Linear(width, self.config.action_width)
            self.global_output = nn.Linear(width, self.config.action_width)

            observed = sum(parameter.numel() for parameter in self.parameters())
            expected = expected_predictor_parameter_count(self.config)
            if observed != expected:
                raise ActionPlanPredictorError(
                    f"predictor parameter inventory differs: {observed} != {expected}"
                )

        def load_state_dict(
            self,
            state_dict: Mapping[str, Any],
            strict: bool = True,
            assign: bool = False,
        ) -> Any:
            return _load_state_dict_after_preflight(
                self, state_dict, strict=strict, assign=assign
            )

        def _load_from_state_dict(
            self,
            state_dict: Mapping[str, Any],
            prefix: str,
            local_metadata: Mapping[str, Any],
            strict: bool,
            missing_keys: list[str],
            unexpected_keys: list[str],
            error_msgs: list[str],
        ) -> None:
            abi_key = prefix + "_abi_config"
            incoming = state_dict.get(abi_key)
            if incoming is not None and not torch.equal(
                incoming.detach().cpu(), self._abi_config.detach().cpu()
            ):
                error_msgs.append(
                    f"{abi_key} differs from the constructed predictor ABI"
                )
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

        def _validate_inputs(
            self, source_tokens: Any, instruction_tokens: Any
        ) -> None:
            _require_float_tensor(source_tokens, label="source_tokens", ndim=5)
            _require_float_tensor(
                instruction_tokens, label="instruction_tokens", ndim=3
            )
            if int(source_tokens.shape[0]) <= 0 or any(
                int(value) <= 0 for value in source_tokens.shape[1:4]
            ):
                raise ActionPlanPredictorError("source token B/THW geometry is empty")
            if (
                self.config.profile == FORMAL_PROFILE
                and int(source_tokens.shape[1]) != PHASE_COUNT
            ):
                raise ActionPlanPredictorError(
                    "formal source token grid requires exactly 21 latent phases"
                )
            if int(instruction_tokens.shape[1]) <= 0:
                raise ActionPlanPredictorError("instruction token sequence is empty")
            if int(source_tokens.shape[0]) != int(instruction_tokens.shape[0]):
                raise ActionPlanPredictorError(
                    "source/instruction batch sizes differ"
                )
            if int(source_tokens.shape[-1]) != self.config.source_token_width:
                raise ActionPlanPredictorError("source token width differs")
            if (
                int(instruction_tokens.shape[2])
                != self.config.instruction_token_width
            ):
                raise ActionPlanPredictorError("instruction token width differs")
            if source_tokens.device != instruction_tokens.device:
                raise ActionPlanPredictorError(
                    "source/instruction tokens must share one device"
                )
            if source_tokens.dtype != instruction_tokens.dtype:
                raise ActionPlanPredictorError(
                    "source/instruction dtypes must match for output restoration"
                )

        def _soft_pool_actor_object(self, source_states: Any) -> Any:
            normalized = self.source_norm(source_states)
            queries = self.actor_object_queries.unsqueeze(0).expand(
                int(source_states.shape[0]), -1, -1
            )
            scores = torch.matmul(queries, normalized.transpose(1, 2))
            scores = scores * (self.config.model_width ** -0.5)
            weights = torch.softmax(scores, dim=-1)
            return torch.matmul(weights, source_states) + queries

        def forward(
            self, source_tokens: Any, instruction_tokens: Any
        ) -> ActionPlanOutput:
            self._validate_inputs(source_tokens, instruction_tokens)
            _require_fp32_trainables(self, label="action-plan predictor")
            if not bool(torch.isfinite(source_tokens).all()) or not bool(
                torch.isfinite(instruction_tokens).all()
            ):
                raise ActionPlanPredictorError(
                    "source/instruction failed route-boundary finite audit"
                )
            output_dtype = source_tokens.dtype
            batch, temporal, height, spatial_width, _ = map(
                int, source_tokens.shape
            )
            with _disabled_ambient_autocast(source_tokens.device.type):
                source_states = self.source_projection(source_tokens.float())
                source_position = deterministic_source_position_encoding(
                    temporal,
                    height,
                    spatial_width,
                    self.config.model_width,
                    device=source_tokens.device,
                )
                source_states = source_states + source_position.unsqueeze(0)
                source_states = source_states.reshape(
                    batch, temporal * height * spatial_width, self.config.model_width
                )
                actor_object_memory = self._soft_pool_actor_object(source_states)
                instruction_memory = self.instruction_norm(
                    self.instruction_projection(instruction_tokens.float())
                )
                memory = torch.cat(
                    (actor_object_memory, instruction_memory), dim=1
                )

                phase = self.phase_queries + self.phase_position_encoding
                queries = torch.cat((phase, self.global_query), dim=0)
                query_states = queries.unsqueeze(0).expand(batch, -1, -1)
                for block in self.blocks:
                    query_states = block(query_states, memory)
                query_states = self.output_norm(query_states)
                phase_tokens = self.phase_output(
                    query_states[:, : self.config.phase_count]
                ).to(dtype=output_dtype)
                global_token = self.global_output(
                    query_states[:, self.config.phase_count]
                ).to(dtype=output_dtype)
            output = ActionPlanOutput(
                phase_tokens=phase_tokens,
                global_token=global_token,
            )
            self.validate_output(output, batch=batch)
            return output

        def validate_output(self, output: ActionPlanOutput, *, batch: int) -> None:
            if not isinstance(output, ActionPlanOutput):
                raise ActionPlanPredictorError("predictor output type differs")
            expected_phase = (
                batch,
                self.config.phase_count,
                self.config.action_width,
            )
            expected_global = (batch, self.config.action_width)
            if tuple(output.phase_tokens.shape) != expected_phase:
                raise ActionPlanPredictorError("phase action-token shape differs")
            if tuple(output.global_token.shape) != expected_global:
                raise ActionPlanPredictorError("global action-token shape differs")
            # Finiteness is audited exactly once by ``bind_route``.  Keeping it
            # out of this shape-only check avoids duplicate device syncs.


    class ZeroInitTargetOnlyActionInjectionV1(nn.Module):
        """Inject through a certificate-bound closed target-suffix route.

        Shape alone does not establish target ownership.  Packed tensors and
        bare plans are rejected: the runner must first issue a closed
        source-prefix/target-suffix certificate and bind one finite-audited
        plan route.  Each zero-initialized block projection is itself the
        residual gate; there is no second zero scalar that would suppress its
        bootstrap gradient.
        """

        gate_semantics = ZERO_INIT_GATE_SEMANTICS

        def __init__(
            self,
            *,
            hidden_width: int = FORMAL_RENDERER_HIDDEN_WIDTH,
            action_width: int = ACTION_WIDTH,
            phase_count: int = PHASE_COUNT,
            block_count: int = TRANSFORMER_BLOCK_COUNT,
        ):
            super().__init__()
            for name, value in (
                ("hidden_width", hidden_width),
                ("action_width", action_width),
                ("phase_count", phase_count),
                ("block_count", block_count),
            ):
                if type(value) is not int or value <= 0:
                    raise ActionPlanPredictorError(
                        f"{name} must be a positive integer"
                    )
            if (
                action_width != ACTION_WIDTH
                or phase_count != PHASE_COUNT
                or block_count != TRANSFORMER_BLOCK_COUNT
            ):
                raise ActionPlanPredictorError(
                    "V1 injection requires 30 blocks, 21 phases, and "
                    "256-wide action tokens"
                )
            self.hidden_width = hidden_width
            self.action_width = action_width
            self.phase_count = phase_count
            self.block_count = block_count
            self.projections = nn.ModuleList(
                nn.Linear(2 * action_width, hidden_width)
                for _ in range(block_count)
            )
            for projection in self.projections:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
            self.register_buffer(
                "_abi_config",
                torch.tensor(
                    (2, block_count, phase_count, action_width, hidden_width, 1),
                    dtype=torch.int64,
                ),
                persistent=True,
            )
            observed = sum(parameter.numel() for parameter in self.parameters())
            expected = expected_injection_parameter_count(
                action_width=action_width,
                hidden_width=hidden_width,
                block_count=block_count,
            )
            if observed != expected:
                raise ActionPlanPredictorError(
                    f"injection parameter inventory differs: {observed} != {expected}"
                )

        def load_state_dict(
            self,
            state_dict: Mapping[str, Any],
            strict: bool = True,
            assign: bool = False,
        ) -> Any:
            return _load_state_dict_after_preflight(
                self, state_dict, strict=strict, assign=assign
            )

        def _load_from_state_dict(
            self,
            state_dict: Mapping[str, Any],
            prefix: str,
            local_metadata: Mapping[str, Any],
            strict: bool,
            missing_keys: list[str],
            unexpected_keys: list[str],
            error_msgs: list[str],
        ) -> None:
            abi_key = prefix + "_abi_config"
            incoming = state_dict.get(abi_key)
            if incoming is not None and not torch.equal(
                incoming.detach().cpu(), self._abi_config.detach().cpu()
            ):
                error_msgs.append(
                    f"{abi_key} differs from the constructed injection ABI"
                )
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

        def _validate_plan(
            self,
            plan: ActionPlanOutput,
            *,
            batch: int,
            audit_finite: bool,
        ) -> None:
            if not isinstance(plan, ActionPlanOutput):
                raise ActionPlanPredictorError(
                    "injection requires an ActionPlanOutput"
                )
            if tuple(plan.phase_tokens.shape) != (
                batch,
                self.phase_count,
                self.action_width,
            ):
                raise ActionPlanPredictorError("injection phase-token shape differs")
            if tuple(plan.global_token.shape) != (batch, self.action_width):
                raise ActionPlanPredictorError("injection global-token shape differs")
            if plan.phase_tokens.device != plan.global_token.device:
                raise ActionPlanPredictorError(
                    "phase/global action tokens must share one device"
                )
            if plan.phase_tokens.dtype != plan.global_token.dtype:
                raise ActionPlanPredictorError(
                    "phase/global action token dtypes differ"
                )
            if plan.phase_tokens.dtype not in _SUPPORTED_COMPUTE_INPUT_DTYPES:
                raise ActionPlanPredictorError(
                    "action plan dtype is outside compute policy"
                )
            if audit_finite and (
                not bool(torch.isfinite(plan.phase_tokens).all())
                or not bool(torch.isfinite(plan.global_token).all())
            ):
                raise ActionPlanPredictorError(
                    "action plan failed route-boundary finite audit"
                )

        def bind_route(
            self,
            plan: ActionPlanOutput,
            ownership: TargetOwnershipCertificate,
            *,
            audit_finite: bool = True,
        ) -> ActionPlanInjectionRoute:
            """Validate ownership/plan once before the thirty-block hot path."""

            _validate_ownership_certificate(ownership)
            batch = int(ownership.target_shape[0])
            self._validate_plan(
                plan, batch=batch, audit_finite=audit_finite is True
            )
            if audit_finite is not True:
                raise ActionPlanPredictorError(
                    "injection route requires one explicit finite plan audit"
                )
            if _dtype_name(plan.phase_tokens.dtype) != ownership.target_dtype:
                raise ActionPlanPredictorError(
                    "plan dtype differs from certified target dtype"
                )
            if plan.phase_tokens.device.type != ownership.target_device_type:
                raise ActionPlanPredictorError(
                    "plan device differs from certified target device"
                )
            if ownership.target_shape[-1] != self.hidden_width:
                raise ActionPlanPredictorError(
                    "certified target hidden width differs from injection"
                )
            body = _route_metadata_body(plan, ownership)
            route = ActionPlanInjectionRoute(
                schema_version=INJECTION_ROUTE_SCHEMA,
                plan=plan,
                ownership=ownership,
                finite_audited=True,
                metadata_digest=hashlib.sha256(_canonical_json(body)).hexdigest(),
            )
            self._validate_route(route)
            return route

        def _validate_route(self, route: ActionPlanInjectionRoute) -> None:
            if not isinstance(route, ActionPlanInjectionRoute):
                raise ActionPlanPredictorError(
                    "injection requires an ActionPlanInjectionRoute"
                )
            if (
                route.schema_version != INJECTION_ROUTE_SCHEMA
                or route.finite_audited is not True
            ):
                raise ActionPlanPredictorError("injection route semantics differ")
            _validate_ownership_certificate(route.ownership)
            self._validate_plan(
                route.plan,
                batch=int(route.ownership.target_shape[0]),
                audit_finite=False,
            )
            expected = hashlib.sha256(
                _canonical_json(_route_metadata_body(route.plan, route.ownership))
            ).hexdigest()
            if route.metadata_digest != expected:
                raise ActionPlanPredictorError(
                    "injection route metadata digest differs"
                )

        def _validate_block_index(self, block_index: int) -> None:
            if type(block_index) is not int or not 0 <= block_index < self.block_count:
                raise ActionPlanPredictorError(
                    "block_index must be an exact integer in [0,29]"
                )

        def validate_block_traversal(self, block_indices: Any) -> tuple[int, ...]:
            """Validate the exact hook installation/traversal order ``0..29``."""

            if not isinstance(block_indices, (tuple, list)):
                raise ActionPlanPredictorError(
                    "block traversal must be an explicit tuple/list"
                )
            indices = tuple(block_indices)
            for block_index in indices:
                self._validate_block_index(block_index)
            if len(set(indices)) != len(indices):
                raise ActionPlanPredictorError(
                    "block traversal contains a duplicate block index"
                )
            expected = tuple(range(self.block_count))
            if set(indices) != set(expected):
                raise ActionPlanPredictorError(
                    "block traversal has a missing or extra block index"
                )
            if indices != expected:
                raise ActionPlanPredictorError(
                    "block traversal order must be exactly 0..29"
                )
            return indices

        def residual(
            self, route: ActionPlanInjectionRoute, *, block_index: int
        ) -> Any:
            self._validate_block_index(block_index)
            self._validate_route(route)
            projection = self.projections[block_index]
            _require_fp32_trainables(
                projection, label=f"injection projection {block_index}"
            )
            plan = route.plan
            global_by_phase = plan.global_token.unsqueeze(1).expand(
                -1, self.phase_count, -1
            )
            with _disabled_ambient_autocast(plan.phase_tokens.device.type):
                condition = torch.cat(
                    (plan.phase_tokens.float(), global_by_phase.float()), dim=-1
                )
                residual = projection(condition)
            return residual.to(dtype=plan.phase_tokens.dtype)

        def forward(
            self,
            target_hidden: Any,
            route: ActionPlanInjectionRoute,
            *,
            block_index: int,
        ) -> Any:
            self._validate_route(route)
            _validate_ownership_certificate(
                route.ownership, target_hidden=target_hidden
            )
            batch = int(target_hidden.shape[0])
            if batch <= 0 or int(target_hidden.shape[1]) != self.phase_count:
                raise ActionPlanPredictorError(
                    "target_hidden must have exactly 21 phases on axis 1"
                )
            if int(target_hidden.shape[-1]) != self.hidden_width:
                raise ActionPlanPredictorError("target hidden width differs")
            if target_hidden.device != route.plan.phase_tokens.device:
                raise ActionPlanPredictorError(
                    "target hidden and action plan must share one device"
                )
            residual = self.residual(route, block_index=block_index)
            for _ in range(target_hidden.ndim - 3):
                residual = residual.unsqueeze(2)
            return target_hidden + residual.to(dtype=target_hidden.dtype)


    class ActionPlanConditionerV1(nn.Module):
        """Persist predictor and target-only injection in one checkpoint tree."""

        def __init__(
            self,
            config: ActionPlanPredictorConfig | None = None,
            *,
            renderer_hidden_width: int = FORMAL_RENDERER_HIDDEN_WIDTH,
        ):
            super().__init__()
            self.config = config or ActionPlanPredictorConfig()
            self.config.validate()
            if (
                self.config.profile == FORMAL_PROFILE
                and renderer_hidden_width != FORMAL_RENDERER_HIDDEN_WIDTH
            ):
                raise ActionPlanPredictorError(
                    "formal ActionPlanConditionerV1 requires renderer hidden width 1536"
                )
            self.renderer_hidden_width = renderer_hidden_width
            self.predictor = ActionPlanPredictorV1(self.config)
            self.injection = ZeroInitTargetOnlyActionInjectionV1(
                hidden_width=renderer_hidden_width,
                action_width=self.config.action_width,
                phase_count=self.config.phase_count,
                block_count=TRANSFORMER_BLOCK_COUNT,
            )
            observed = sum(parameter.numel() for parameter in self.parameters())
            expected = expected_conditioner_parameter_count(
                self.config, renderer_hidden_width=renderer_hidden_width
            )
            if observed != expected:
                raise ActionPlanPredictorError(
                    f"conditioner parameter inventory differs: {observed} != {expected}"
                )

        def load_state_dict(
            self,
            state_dict: Mapping[str, Any],
            strict: bool = True,
            assign: bool = False,
        ) -> Any:
            return _load_state_dict_after_preflight(
                self, state_dict, strict=strict, assign=assign
            )

        def prepare_route(
            self,
            source_tokens: Any,
            instruction_tokens: Any,
            ownership: TargetOwnershipCertificate,
        ) -> ActionPlanInjectionRoute:
            plan = self.predictor(source_tokens, instruction_tokens)
            return self.injection.bind_route(
                plan, ownership, audit_finite=True
            )

        def forward(
            self,
            target_hidden: Any,
            route: ActionPlanInjectionRoute,
            *,
            block_index: int,
        ) -> ConditionedTargetOutput:
            return ConditionedTargetOutput(
                target_hidden=self.injection(
                    target_hidden, route, block_index=block_index
                ),
                plan=route.plan,
            )


    def exact_parameter_inventory(module: nn.Module) -> dict[str, Any]:
        """Return a deterministic, per-parameter trainability inventory."""

        entries = []
        total = 0
        trainable = 0
        for name, parameter in module.named_parameters():
            count = int(parameter.numel())
            total += count
            if parameter.requires_grad:
                trainable += count
            entries.append(
                {
                    "name": name,
                    "shape": [int(value) for value in parameter.shape],
                    "dtype": _dtype_name(parameter.dtype),
                    "requires_grad": bool(parameter.requires_grad),
                    "numel": count,
                }
            )
        return {
            "schema_version": "bernini-exact-parameter-inventory-v1",
            "module_class": type(module).__name__,
            "parameter_count": total,
            "trainable_parameter_count": trainable,
            "parameters": entries,
        }


    def exact_state_dict_abi(
        module: nn.Module, *, schema_version: str | None = None
    ) -> dict[str, Any]:
        """Hash key metadata and semantic buffers, but not learned parameters."""

        parameter_names = {name for name, _ in module.named_parameters()}
        entries = []
        for name, value in module.state_dict().items():
            is_parameter = name in parameter_names
            entry = {
                "name": name,
                "kind": "parameter" if is_parameter else "buffer",
                "shape": [int(item) for item in value.shape],
                "dtype": _dtype_name(value.dtype),
            }
            if not is_parameter:
                entry["semantic_value_sha256"] = _tensor_value_sha256(value)
            entries.append(entry)
        if schema_version is None:
            if isinstance(module, ActionPlanPredictorV1):
                schema_version = PREDICTOR_ABI_SCHEMA
            elif isinstance(module, ZeroInitTargetOnlyActionInjectionV1):
                schema_version = INJECTION_ABI_SCHEMA
            elif isinstance(module, ActionPlanConditionerV1):
                schema_version = CONDITIONER_ABI_SCHEMA
            else:
                raise ActionPlanPredictorError(
                    "state-dict ABI schema must be explicit for unknown modules"
                )
        body = {
            "schema_version": schema_version,
            "module_class": type(module).__name__,
            "entries": entries,
        }
        return {
            **body,
            "abi_sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
        }


except ImportError:  # pragma: no cover - local contract-only workspaces

    class ActionPlanPredictorV1:  # type: ignore[no-redef]
        architecture = ARCHITECTURE_NAME

        def __init__(self, config: ActionPlanPredictorConfig | None = None):
            (config or ActionPlanPredictorConfig()).validate()
            raise ActionPlanPredictorError("ActionPlanPredictorV1 requires PyTorch")

        def forward(
            self, source_tokens: Any, instruction_tokens: Any
        ) -> ActionPlanOutput:
            raise ActionPlanPredictorError("ActionPlanPredictorV1 requires PyTorch")


    class ZeroInitTargetOnlyActionInjectionV1:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            raise ActionPlanPredictorError(
                "ZeroInitTargetOnlyActionInjectionV1 requires PyTorch"
            )

        def forward(
            self,
            target_hidden: Any,
            route: ActionPlanInjectionRoute,
            *,
            block_index: int,
        ) -> Any:
            raise ActionPlanPredictorError(
                "ZeroInitTargetOnlyActionInjectionV1 requires PyTorch"
            )


    class ActionPlanConditionerV1:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            raise ActionPlanPredictorError("ActionPlanConditionerV1 requires PyTorch")

        def forward(
            self,
            target_hidden: Any,
            route: ActionPlanInjectionRoute,
            *,
            block_index: int,
        ) -> ConditionedTargetOutput:
            raise ActionPlanPredictorError("ActionPlanConditionerV1 requires PyTorch")


    def exact_parameter_inventory(module: Any) -> dict[str, Any]:
        raise ActionPlanPredictorError("exact parameter inventory requires PyTorch")


    def exact_state_dict_abi(
        module: Any, *, schema_version: str | None = None
    ) -> dict[str, Any]:
        raise ActionPlanPredictorError("exact state-dict ABI requires PyTorch")


    def certify_closed_target_suffix_route(
        target_hidden: Any,
        *,
        source_prefix_tokens: int,
        packed_total_tokens: int,
        audit_finite: bool = True,
    ) -> TargetOwnershipCertificate:
        raise ActionPlanPredictorError(
            "target ownership certification requires PyTorch"
        )


    def deterministic_source_position_encoding(
        temporal: int,
        height: int,
        width: int,
        channels: int,
        *,
        device: Any,
    ) -> Any:
        raise ActionPlanPredictorError(
            "source position encoding requires PyTorch"
        )


__all__ = [
    "ACTION_WIDTH",
    "ACTOR_OBJECT_QUERY_COUNT",
    "ARCHITECTURE_NAME",
    "COMPUTE_DTYPE_POLICY",
    "CONDITIONER_ABI_SCHEMA",
    "CPU_TEST_PROFILE",
    "FORMAL_PROFILE",
    "FORMAL_RENDERER_HIDDEN_WIDTH",
    "INJECTION_ABI_SCHEMA",
    "INJECTION_ROUTE_SCHEMA",
    "PHASE_COUNT",
    "PREDICTOR_ABI_SCHEMA",
    "SOURCE_POSITION_ENCODING_SCHEMA",
    "TARGET_OWNERSHIP_SCHEMA",
    "TARGET_SUFFIX_ROUTE_SEMANTICS",
    "TRANSFORMER_BLOCK_COUNT",
    "ZERO_INIT_GATE_SEMANTICS",
    "ActionPlanInjectionRoute",
    "ActionPlanConditionerV1",
    "ActionPlanOutput",
    "ActionPlanPredictorConfig",
    "ActionPlanPredictorError",
    "ActionPlanPredictorV1",
    "ConditionedTargetOutput",
    "TargetOwnershipCertificate",
    "ZeroInitTargetOnlyActionInjectionV1",
    "certify_closed_target_suffix_route",
    "deterministic_source_position_encoding",
    "exact_parameter_inventory",
    "exact_state_dict_abi",
    "expected_conditioner_parameter_count",
    "expected_injection_parameter_count",
    "expected_predictor_parameter_count",
]
