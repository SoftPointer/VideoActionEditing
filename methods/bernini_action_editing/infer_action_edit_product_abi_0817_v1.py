#!/usr/bin/env python3
"""Source+instruction-only product bridge for the 0817 action conditioner.

The public product route consumes clean source patch tokens, contextual T5
instruction tokens, and ordinary inference noise/scheduler controls.  It has no
clean edited target, action anchor, teacher feature, track, pose, contact, tube,
or mask input.  The exact persisted ``ActionPlanConditionerV1`` predictor and
all 30 zero-init projection heads are reused without an inference-only copy.

This is an offline engineering ABI, not a quality result.  It intentionally
does not choose checkpoints and every receipt remains PRE_D0/nonpromotable.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Iterator, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
METHOD = "bernini-action-edit-product-abi-0817-v1"
AUTHORITY = "PRE_D0_ENGINEERING_ONLY"
PRODUCT_ABI_SCHEMA = "bernini-action-edit-source-instruction-product-abi-v1"
PRODUCT_RECEIPT_SCHEMA = "bernini-action-edit-offline-product-receipt-v1"
CHECKPOINT_CONSUMER_SCHEMA = "bernini-action-edit-fresh-consumer-receipt-v1"
TRAINING_REFERENCE_SCHEMA = (
    "bernini-action-edit-training-attached-fixed-forward-reference-v1"
)
TRAINING_REFERENCE_BINDING_SCHEMA = (
    "bernini-action-edit-training-reference-checkpoint-binding-v1"
)
FRESH_PARITY_SCHEMA = "bernini-action-edit-fresh-a-b-fixed-forward-parity-v1"
REFERENCE_TENSOR_SCHEMA = "bernini-action-edit-fixed-forward-tensors-v1"
FULL_RENDERER_REFERENCE_SCHEMA = (
    "bernini-action-edit-training-attached-full-renderer-fixed-forward-reference-v1"
)
FULL_RENDERER_REFERENCE_BINDING_SCHEMA = (
    "bernini-action-edit-full-renderer-reference-checkpoint-binding-v1"
)
FULL_RENDERER_TENSOR_SCHEMA = (
    "bernini-action-edit-full-renderer-fixed-forward-tensors-v1"
)
FULL_RENDERER_CALLBACK_CONTRACT_SCHEMA = (
    "bernini-action-edit-full-renderer-forward-callback-contract-v1"
)
FULL_RENDERER_CALLBACK_RESULT_SCHEMA = (
    "bernini-action-edit-full-renderer-forward-callback-result-v1"
)
INFERENCE_POLICY_SCHEMA = "bernini-action-edit-offline-inference-policy-v1"
PINNED_UNIPC_SCHEDULE_SHA256 = (
    "3e5ad4473d133318026cc9e8f32399782bf06313691b58870c89d9c4c87c3d03"
)
PINNED_UNIPC_TIMESTEPS = (
    999, 994, 989, 984, 978, 972, 965, 959, 952, 945,
    937, 929, 921, 912, 902, 893, 882, 871, 859, 847,
    833, 819, 803, 787, 769, 750, 729, 707, 682, 655,
    625, 593, 556, 516, 470, 418, 359, 291, 211, 117,
)
TRANSFORMER_BLOCKS = 30
PHASES = 21
FORMAL_HIDDEN_WIDTH = 1536
FORMAL_INSTRUCTION_WIDTH = 4096
WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
FORBIDDEN_PRODUCT_ARGUMENT_FRAGMENTS = (
    "anchor",
    "teacher",
    "target",
    "track",
    "pose",
    "contact",
    "tube",
    "mask",
    "annotation",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ProductABIError(RuntimeError):
    """Raised before an ambiguous or teacher-dependent product call runs."""


def fail(message: str) -> NoReturn:
    raise ProductABIError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _stable_file_sha256(path: Path, *, label: str) -> tuple[str, Mapping[str, int]]:
    """Hash one open file description and return its stable filesystem identity."""

    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            digest = hashlib.sha256()
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise ProductABIError(f"{label} is unavailable: {error}") from error
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
    )
    if identity(before) != identity(after):
        fail(f"{label} changed during hashing")
    return digest.hexdigest(), {
        "device": before.st_dev,
        "inode": before.st_ino,
        "mode": before.st_mode,
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
    }


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be one lowercase full SHA-256")
    return value


def _torch_load_authenticated(
    path: Path, *, expected_sha256: str, torch_module: Any, label: str
) -> Any:
    expected = _require_sha(expected_sha256, label=f"expected {label} SHA")
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        digest = hashlib.sha256()
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
        if digest.hexdigest() != expected:
            fail(f"{label} stable open-file SHA differs")
        handle.seek(0)
        try:
            value = torch_module.load(
                handle, map_location="cpu", weights_only=True
            )
        except TypeError as error:
            raise ProductABIError(
                f"PyTorch weights_only {label} loading is mandatory"
            ) from error
        after = os.fstat(handle.fileno())
        identity = lambda item: (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_size,
            item.st_mtime_ns,
        )
        if identity(before) != identity(after):
            fail(f"{label} changed during deserialize")
    return value


@dataclass(frozen=True)
class OfflineInferencePolicyV1:
    """Frozen declarations for a product-path engineering inference."""

    seed: int
    schema_version: str = INFERENCE_POLICY_SCHEMA
    sampler: str = "bernini_paired_clean_source_prefix_flow_v1"
    scheduler_class: str = "UniPCMultistepScheduler"
    num_inference_steps: int = 40
    flow_shift: float = 5.0
    schedule_sha256: str = PINNED_UNIPC_SCHEDULE_SHA256
    inference_noise: str = "counter_based_torch_Generator_cpu"
    source_condition: str = "clean_source_vae_patch_prefix_each_denoise_step"
    evolving_target_state: str = "inference_gaussian_flow_state_not_clean_target"
    training_rng_restored: bool = False
    training_sampler_cursor_consumed: bool = False
    training_scheduler_object_consumed: bool = False

    def validate(self) -> None:
        expected = {
            "schema_version": INFERENCE_POLICY_SCHEMA,
            "sampler": "bernini_paired_clean_source_prefix_flow_v1",
            "scheduler_class": "UniPCMultistepScheduler",
            "num_inference_steps": 40,
            "flow_shift": 5.0,
            "schedule_sha256": PINNED_UNIPC_SCHEDULE_SHA256,
            "inference_noise": "counter_based_torch_Generator_cpu",
            "source_condition": "clean_source_vae_patch_prefix_each_denoise_step",
            "evolving_target_state": "inference_gaussian_flow_state_not_clean_target",
            "training_rng_restored": False,
            "training_sampler_cursor_consumed": False,
            "training_scheduler_object_consumed": False,
        }
        observed = asdict(self)
        observed.pop("seed")
        if observed != expected or type(self.seed) is not int or not 0 <= self.seed < 2**63:
            fail("offline inference sampler/RNG/scheduler policy differs")

    def receipt(self) -> Mapping[str, Any]:
        self.validate()
        return asdict(self)


def audit_live_inference_scheduler(
    *,
    scheduler: Any,
    inference_policy: OfflineInferencePolicyV1,
    sigma_contract_module: Any,
    initialize: bool = True,
) -> Mapping[str, Any]:
    """Bind the live scheduler to the frozen release's exact float32 grid."""

    inference_policy.validate()
    if (
        getattr(sigma_contract_module, "SCHEDULE_SHA256", None)
        != PINNED_UNIPC_SCHEDULE_SHA256
        or getattr(sigma_contract_module, "SCHEDULER_CLASS", None)
        != inference_policy.scheduler_class
        or getattr(sigma_contract_module, "NUM_INFERENCE_STEPS", None)
        != inference_policy.num_inference_steps
        or getattr(sigma_contract_module, "FLOW_SHIFT", None)
        != inference_policy.flow_shift
        or not callable(
            getattr(sigma_contract_module, "audit_runtime_unipc_schedule", None)
        )
    ):
        fail("authenticated inference sigma/scheduler contract differs")
    try:
        schedule = sigma_contract_module.audit_runtime_unipc_schedule(
            scheduler, initialize=initialize
        )
    except Exception as error:
        raise ProductABIError(f"live UniPC scheduler rejected: {error}") from error
    if (
        not isinstance(schedule, Mapping)
        or schedule.get("schedule_sha256") != inference_policy.schedule_sha256
        or len(schedule.get("timesteps", ())) != inference_policy.num_inference_steps
        or len(schedule.get("positive_sigmas_float32_be_hex", ()))
        != inference_policy.num_inference_steps
        or schedule.get("terminal_sigma_float32_be_hex") != "00000000"
    ):
        fail("live UniPC exact40 schedule receipt differs")
    return {
        "scheduler_class": inference_policy.scheduler_class,
        "flow_shift": inference_policy.flow_shift,
        "num_inference_steps": inference_policy.num_inference_steps,
        "schedule_sha256": inference_policy.schedule_sha256,
        "initialize_called": initialize,
        "exact_runtime_schedule_verified": True,
        "schedule": dict(schedule),
    }


@dataclass(frozen=True)
class ProductActionInputsV1:
    """The only tensor inputs accepted by the product action-plan boundary."""

    clean_source_tokens: Any
    instruction_tokens: Any
    inference_noise_hidden: Any


@dataclass(frozen=True)
class ProductRequestV1:
    """External product boundary: one source video plus one instruction."""

    source_video_path: str
    expected_source_video_sha256: str
    instruction: str
    inference_policy: OfflineInferencePolicyV1

    def validate(self) -> Mapping[str, Any]:
        source_sha = _require_sha(
            self.expected_source_video_sha256, label="expected source video SHA"
        )
        requested = Path(self.source_video_path).expanduser()
        try:
            resolved = requested.resolve(strict=True)
        except OSError as error:
            raise ProductABIError("product source video is unavailable") from error
        if (
            not requested.is_absolute()
            or requested.is_symlink()
            or resolved != requested
            or not requested.is_file()
        ):
            fail("product source video canonical file type differs")
        observed_sha, source_identity = _stable_file_sha256(
            requested, label="product source video"
        )
        if (
            observed_sha != source_sha
        ):
            fail("product source video canonical bytes differ")
        if (
            not isinstance(self.instruction, str)
            or not self.instruction
            or self.instruction != self.instruction.strip()
            or len(self.instruction.encode("utf-8")) > 16 * 1024
        ):
            fail("product editing instruction UTF-8 contract differs")
        self.inference_policy.validate()
        return {
            "source_video_path": str(requested),
            "source_video_sha256": source_sha,
            "source_file_identity": source_identity,
            "instruction_utf8_sha256": hashlib.sha256(
                self.instruction.encode("utf-8")
            ).hexdigest(),
            "inference_policy": dict(self.inference_policy.receipt()),
            "clean_target_present": False,
            "anchor_present": False,
            "teacher_or_external_annotation_present": False,
        }


@dataclass(frozen=True)
class ProductActionOutputV1:
    route: Any
    initial_inference_hidden: Any
    conditioned_inference_hidden: Any
    block_indices: tuple[int, ...]


def validate_public_product_signatures() -> Mapping[str, Any]:
    """Prevent teacher/anchor arguments from creeping into public functions."""

    functions = (
        materialize_product_request,
        prepare_product_action_route,
        run_product_action_fixed_cell,
        prepare_product_route_from_packed_embeddings,
    )
    signatures = {}
    for function in functions:
        names = tuple(inspect.signature(function).parameters)
        lowered = tuple(name.lower() for name in names)
        bad = sorted(
            name
            for name in lowered
            if any(fragment in name for fragment in FORBIDDEN_PRODUCT_ARGUMENT_FRAGMENTS)
        )
        if bad:
            fail(f"product function exposes forbidden training-only arguments: {bad}")
        signatures[function.__name__] = list(names)
    if list(inspect.signature(ProductActionInputsV1).parameters) != [
        "clean_source_tokens",
        "instruction_tokens",
        "inference_noise_hidden",
    ]:
        fail("product tensor-input dataclass fields differ")
    return {
        "schema_version": PRODUCT_ABI_SCHEMA,
        "public_signatures": signatures,
        "source_and_instruction_required": True,
        "inference_noise_is_not_training_target": True,
        "target_anchor_teacher_external_annotations_accepted": False,
    }


def _validate_encoder_receipt(
    receipt: Mapping[str, Any], *, conditioner: Any
) -> Mapping[str, Any]:
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema_version",
        "vae_source_code_sha256",
        "vae_weights_sha256",
        "t5_tokenizer_code_sha256",
        "t5_encoder_code_sha256",
        "t5_weights_sha256",
        "noise_factory_code_sha256",
        "vae_frozen",
        "t5_frozen",
        "source_token_width",
        "instruction_token_width",
        "source_preprocessing",
        "instruction_preprocessing",
        "noise_factory_semantics",
    }:
        fail("frozen product encoder receipt field set differs")
    for key in (
        "vae_source_code_sha256",
        "vae_weights_sha256",
        "t5_tokenizer_code_sha256",
        "t5_encoder_code_sha256",
        "t5_weights_sha256",
        "noise_factory_code_sha256",
    ):
        _require_sha(receipt.get(key), label=key)
    if (
        receipt.get("schema_version")
        != "bernini-frozen-source-instruction-encoders-v1"
        or receipt.get("vae_frozen") is not True
        or receipt.get("t5_frozen") is not True
        or receipt.get("source_token_width")
        != int(conditioner.config.source_token_width)
        or receipt.get("instruction_token_width")
        != int(conditioner.config.instruction_token_width)
        or receipt.get("source_preprocessing")
        != "exact81_rgb_to_normalized_clean_vae_patch_tokens"
        or receipt.get("instruction_preprocessing")
        != "complete_unpadded_contextual_frozen_t5_tokens"
        or receipt.get("noise_factory_semantics")
        != "counter_based_torch_Generator_cpu_no_global_rng_mutation"
    ):
        fail("frozen product encoder semantic ABI differs")
    return dict(receipt)


def materialize_product_request(
    *,
    request: ProductRequestV1,
    conditioner: Any,
    source_video_to_clean_source_tokens: Any,
    instruction_to_contextual_tokens: Any,
    inference_noise_factory: Any,
    encoder_receipt: Mapping[str, Any],
    torch_module: Any,
) -> tuple[ProductActionInputsV1, Mapping[str, Any]]:
    """Encode source+instruction and create inference noise without a target."""

    request_receipt = request.validate()
    frozen_encoders = _validate_encoder_receipt(
        encoder_receipt, conditioner=conditioner
    )
    if any(
        not callable(value)
        for value in (
            source_video_to_clean_source_tokens,
            instruction_to_contextual_tokens,
            inference_noise_factory,
        )
    ):
        fail("product source/instruction/noise encoder callables differ")
    torch = torch_module
    python_rng_before = random.getstate()
    cpu_rng_before = torch.get_rng_state().detach().cpu().clone()
    cuda_rng_before = (
        [value.detach().cpu().clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    try:
        source = source_video_to_clean_source_tokens(
            Path(request.source_video_path)
        )
        instruction = instruction_to_contextual_tokens(request.instruction)
        noise = inference_noise_factory(source, request.inference_policy)
        replay_noise = inference_noise_factory(source, request.inference_policy)
    except Exception:
        random.setstate(python_rng_before)
        torch.set_rng_state(cpu_rng_before)
        if cuda_rng_before:
            torch.cuda.set_rng_state_all(cuda_rng_before)
        raise
    cuda_rng_after = (
        [value.detach().cpu().clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    if (
        not isinstance(noise, torch.Tensor)
        or not isinstance(replay_noise, torch.Tensor)
        or noise.dtype != replay_noise.dtype
        or tuple(noise.shape) != tuple(replay_noise.shape)
        or not bool(
            torch.equal(
                noise.detach().contiguous().reshape(-1).view(torch.uint8),
                replay_noise.detach().contiguous().reshape(-1).view(torch.uint8),
            )
        )
        or random.getstate() != python_rng_before
        or not bool(torch.equal(torch.get_rng_state().cpu(), cpu_rng_before))
        or len(cuda_rng_before) != len(cuda_rng_after)
        or any(
            not bool(torch.equal(before, after))
            for before, after in zip(cuda_rng_before, cuda_rng_after)
        )
    ):
        random.setstate(python_rng_before)
        torch.set_rng_state(cpu_rng_before)
        if cuda_rng_before:
            torch.cuda.set_rng_state_all(cuda_rng_before)
        fail("inference noise is not replayable counter-based RNG isolation")
    del replay_noise
    source_path = Path(request.source_video_path)
    after_sha, after_identity = _stable_file_sha256(
        source_path, label="product source video after encoding"
    )
    if (
        after_identity != request_receipt["source_file_identity"]
        or after_sha != request.expected_source_video_sha256
    ):
        fail("product source video changed during frozen VAE encoding")
    inputs = ProductActionInputsV1(
        clean_source_tokens=source,
        instruction_tokens=instruction,
        inference_noise_hidden=noise,
    )
    _, source_tokens, inference_tokens = _validate_product_tensors(
        inputs, conditioner=conditioner, torch_module=torch_module
    )
    receipt = {
        "schema_version": PRODUCT_ABI_SCHEMA,
        "request": request_receipt,
        "frozen_encoders": frozen_encoders,
        "clean_source_shape": [int(value) for value in source.shape],
        "instruction_shape": [int(value) for value in instruction.shape],
        "inference_noise_shape": [int(value) for value in noise.shape],
        "source_tokens": source_tokens,
        "inference_tokens": inference_tokens,
        "source_tensor_sha256": tensor_sha256(
            source.contiguous(), torch_module=torch_module
        ),
        "instruction_tensor_sha256": tensor_sha256(
            instruction.contiguous(), torch_module=torch_module
        ),
        "inference_noise_sha256": tensor_sha256(
            noise.contiguous(), torch_module=torch_module
        ),
        "source_video_plus_instruction_only": True,
        "clean_target_or_anchor_consumed": False,
    }
    return inputs, receipt


def _validate_product_tensors(
    inputs: ProductActionInputsV1,
    *,
    conditioner: Any,
    torch_module: Any,
) -> tuple[int, int, int]:
    torch = torch_module
    source = inputs.clean_source_tokens
    instruction = inputs.instruction_tokens
    noise = inputs.inference_noise_hidden
    if any(not isinstance(value, torch.Tensor) for value in (source, instruction, noise)):
        fail("product action inputs must be tensors")
    if (
        source.ndim != 5
        or instruction.ndim != 3
        or noise.ndim < 3
        or int(source.shape[0]) != 1
        or int(source.shape[0]) != int(instruction.shape[0])
        or int(source.shape[0]) != int(noise.shape[0])
        or int(source.shape[1]) != PHASES
        or int(source.shape[-1]) != int(conditioner.config.source_token_width)
        or int(instruction.shape[-1])
        != int(conditioner.config.instruction_token_width)
        or int(noise.shape[1]) != PHASES
        or int(noise.shape[-1]) != int(conditioner.renderer_hidden_width)
        or source.device != instruction.device
        or source.device != noise.device
        or source.dtype != instruction.dtype
        or source.dtype != noise.dtype
        or not source.is_floating_point()
        or not instruction.is_floating_point()
        or not noise.is_floating_point()
        or any(value.requires_grad for value in (source, instruction, noise))
        or any(not bool(torch.isfinite(value).all().item()) for value in (source, instruction, noise))
    ):
        fail("source/instruction/inference-noise product tensor ABI differs")
    source_tokens = math.prod(int(value) for value in source.shape[1:-1])
    inference_tokens = math.prod(int(value) for value in noise.shape[1:-1])
    if source_tokens <= 0 or source_tokens != inference_tokens or inference_tokens % PHASES:
        fail("product clean-source/inference-noise token partition differs")
    return int(source.shape[0]), source_tokens, inference_tokens


def prepare_product_action_route(
    *,
    conditioner: Any,
    inputs: ProductActionInputsV1,
    predictor_module: Any,
    torch_module: Any,
) -> Any:
    """Create q_pred from clean source+instruction and certify noise ownership."""

    _, source_tokens, inference_tokens = _validate_product_tensors(
        inputs, conditioner=conditioner, torch_module=torch_module
    )
    ownership = predictor_module.certify_closed_target_suffix_route(
        inputs.inference_noise_hidden,
        source_prefix_tokens=source_tokens,
        packed_total_tokens=source_tokens + inference_tokens,
        audit_finite=True,
    )
    route = conditioner.prepare_route(
        inputs.clean_source_tokens,
        inputs.instruction_tokens,
        ownership,
    )
    if (
        route.ownership.digest != ownership.digest
        or route.ownership.target_only is not True
        or route.ownership.target_suffix_start != source_tokens
    ):
        fail("product action route changed the certified inference suffix")
    return route


def run_product_action_fixed_cell(
    *,
    conditioner: Any,
    inputs: ProductActionInputsV1,
    predictor_module: Any,
    torch_module: Any,
) -> ProductActionOutputV1:
    """Exercise the same predictor and all 30 injection heads without a teacher."""

    route = prepare_product_action_route(
        conditioner=conditioner,
        inputs=inputs,
        predictor_module=predictor_module,
        torch_module=torch_module,
    )
    hidden = inputs.inference_noise_hidden
    initial = hidden.detach().clone(memory_format=torch_module.contiguous_format)
    indices = tuple(range(TRANSFORMER_BLOCKS))
    conditioner.injection.validate_block_traversal(indices)
    for block_index in indices:
        hidden = conditioner(hidden, route, block_index=block_index).target_hidden
    return ProductActionOutputV1(
        route=route,
        initial_inference_hidden=initial,
        conditioned_inference_hidden=hidden,
        block_indices=indices,
    )


def deterministic_fixed_inputs(
    *, conditioner: Any, torch_module: Any, device: Any
) -> ProductActionInputsV1:
    """RNG-free small fixed cell shared by training writer and fresh verifier."""

    torch = torch_module
    source_width = int(conditioner.config.source_token_width)
    instruction_width = int(conditioner.config.instruction_token_width)
    hidden_width = int(conditioner.renderer_hidden_width)

    def values(count: int, *, start: float, stop: float) -> Any:
        # Generate canonical bytes on CPU first.  CUDA and CPU ``linspace``
        # implementations need not round every interior point identically.
        return torch.linspace(
            start,
            stop,
            steps=count,
            dtype=torch.float32,
            device="cpu",
        ).to(device=device)

    source = values(PHASES * source_width, start=-0.75, stop=0.875).reshape(
        1, PHASES, 1, 1, source_width
    )
    instruction = values(
        3 * instruction_width, start=-0.625, stop=0.5
    ).reshape(1, 3, instruction_width)
    noise = values(PHASES * hidden_width, start=-0.375, stop=0.75).reshape(
        1, PHASES, 1, hidden_width
    )
    for value in (source, instruction, noise):
        value.requires_grad_(False)
    return ProductActionInputsV1(
        clean_source_tokens=source,
        instruction_tokens=instruction,
        inference_noise_hidden=noise,
    )


def tensor_sha256(value: Any, *, torch_module: Any) -> str:
    torch = torch_module
    if not isinstance(value, torch.Tensor):
        fail("tensor SHA input must be one torch.Tensor")
    if not value.is_contiguous():
        value = value.contiguous()
    metadata = canonical_json_bytes(
        {
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
        }
    )
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    raw = value.detach().reshape(-1).view(torch.uint8).cpu().tolist()
    digest.update(bytes(raw))
    return digest.hexdigest()


def fixed_forward_tensors(
    *, conditioner: Any, predictor_module: Any, torch_module: Any
) -> Mapping[str, Any]:
    first = next(conditioner.parameters(), None)
    if first is None or first.dtype != torch_module.float32:
        fail("fixed forward requires a materialized FP32 conditioner")
    inputs = deterministic_fixed_inputs(
        conditioner=conditioner, torch_module=torch_module, device=first.device
    )
    with torch_module.no_grad():
        output = run_product_action_fixed_cell(
            conditioner=conditioner,
            inputs=inputs,
            predictor_module=predictor_module,
            torch_module=torch_module,
        )
    tensors = {
        "clean_source_tokens": inputs.clean_source_tokens.detach().cpu().contiguous(),
        "instruction_tokens": inputs.instruction_tokens.detach().cpu().contiguous(),
        "inference_noise_hidden": inputs.inference_noise_hidden.detach().cpu().contiguous(),
        "phase_tokens": output.route.plan.phase_tokens.detach().cpu().contiguous(),
        "global_token": output.route.plan.global_token.detach().cpu().contiguous(),
        "conditioned_inference_hidden": (
            output.conditioned_inference_hidden.detach().cpu().contiguous()
        ),
    }
    if any(not bool(torch_module.isfinite(value).all().item()) for value in tensors.values()):
        fail("fixed forward produced a non-finite tensor")
    return tensors


def fixed_forward_fingerprint(
    tensors: Mapping[str, Any], *, torch_module: Any
) -> Mapping[str, Any]:
    expected = (
        "clean_source_tokens",
        "instruction_tokens",
        "inference_noise_hidden",
        "phase_tokens",
        "global_token",
        "conditioned_inference_hidden",
    )
    if tuple(tensors) != expected:
        fail("fixed-forward tensor field order/set differs")
    rows = []
    for name in expected:
        value = tensors[name]
        rows.append(
            {
                "name": name,
                "shape": [int(item) for item in value.shape],
                "dtype": str(value.dtype),
                "sha256": tensor_sha256(value, torch_module=torch_module),
            }
        )
    return {
        "schema_version": "bernini-action-edit-fixed-forward-fingerprint-v1",
        "forward_scope": (
            "ActionPlanConditionerV1_predictor_plus_exact30_injection_heads"
        ),
        "full_bernini_renderer_forward_covered": False,
        "fixed_input_generator": "rng_free_torch_linspace_v1",
        "product_inputs": [
            "clean_source_tokens",
            "instruction_tokens",
            "inference_noise_hidden",
        ],
        "clean_target_present": False,
        "anchor_present": False,
        "teacher_or_external_annotation_present": False,
        "exact30_injection_heads_executed": True,
        "tensors": rows,
        "tensor_set_sha256": object_sha256(rows),
    }


def _plain_existing_directory(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink directory")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise ProductABIError(f"{label} is unavailable: {error}") from error
    if resolved != requested or not requested.is_dir() or requested.is_symlink():
        fail(f"{label} canonical directory differs")
    return requested


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_training_attached_fixed_forward_reference(
    *,
    output_dir: str | Path,
    conditioner: Any,
    predictor_module: Any,
    torch_module: Any,
    checkpoint_step: int,
    checkpoint_parameter_sha256: str,
    execution_phase: str,
) -> Mapping[str, Any]:
    """Future runner hook: atomically write the pre-save training reference."""

    root = _plain_existing_directory(output_dir, label="training reference output")
    parameter_sha = _require_sha(
        checkpoint_parameter_sha256, label="checkpoint parameter SHA"
    )
    if (
        type(checkpoint_step) is not int
        or checkpoint_step < 0
        or execution_phase != "immediately_before_save_checkpoint"
    ):
        fail("training-attached reference writer execution coordinate differs")
    final_tensor = root / f"training-fixed-forward-{checkpoint_step:08d}.pt"
    final_metadata = root / f"training-fixed-forward-{checkpoint_step:08d}.json"
    if (
        final_tensor.exists()
        or final_metadata.exists()
        or final_tensor.is_symlink()
        or final_metadata.is_symlink()
    ):
        fail("training-attached reference is create-only")
    tensors = fixed_forward_tensors(
        conditioner=conditioner,
        predictor_module=predictor_module,
        torch_module=torch_module,
    )
    fingerprint = fixed_forward_fingerprint(tensors, torch_module=torch_module)
    temporary = Path(tempfile.mkdtemp(prefix=".training-fixed-forward.", dir=root))
    tensor_path = temporary / final_tensor.name
    metadata_path = temporary / final_metadata.name
    torch_module.save(
        {
            "schema_version": REFERENCE_TENSOR_SCHEMA,
            "origin": "training_process_pre_checkpoint_export",
            "checkpoint_step": checkpoint_step,
            "checkpoint_parameter_sha256": parameter_sha,
            "tensors": dict(tensors),
        },
        tensor_path,
    )
    payload = {
        "schema_version": TRAINING_REFERENCE_SCHEMA,
        "method": METHOD,
        "authority": AUTHORITY,
        "promotable": False,
        "origin": "training_process_pre_checkpoint_export",
        "execution_phase": execution_phase,
        "checkpoint_step": checkpoint_step,
        "checkpoint_parameter_sha256": parameter_sha,
        "tensor_file": final_tensor.name,
        "tensor_file_sha256": file_sha256(tensor_path),
        "fingerprint": fingerprint,
        "fresh_consumer_parity_pass": False,
        "writer_process_pid": os.getpid(),
    }
    metadata_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    with tensor_path.open("rb") as handle:
        os.fsync(handle.fileno())
    with metadata_path.open("rb") as handle:
        os.fsync(handle.fileno())
    _fsync_directory(temporary)
    tensor_published = False
    metadata_published = False
    try:
        # Hard-link publication is create-only: unlike rename, it cannot replace
        # an artifact that raced the initial existence check.
        os.link(tensor_path, final_tensor)
        tensor_published = True
        os.link(metadata_path, final_metadata)
        metadata_published = True
        _fsync_directory(root)
    except Exception:
        if metadata_published:
            final_metadata.unlink()
        if tensor_published:
            final_tensor.unlink()
        _fsync_directory(root)
        raise
    finally:
        tensor_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        temporary.rmdir()
    return {
        **payload,
        "metadata_file": final_metadata.name,
        "metadata_file_sha256": file_sha256(final_metadata),
    }


def training_reference_checkpoint_binding(
    writer_receipt: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Small exact object a future runner must embed in checkpoint metadata."""

    if (
        not isinstance(writer_receipt, Mapping)
        or writer_receipt.get("schema_version") != TRAINING_REFERENCE_SCHEMA
        or writer_receipt.get("origin")
        != "training_process_pre_checkpoint_export"
        or writer_receipt.get("execution_phase")
        != "immediately_before_save_checkpoint"
        or type(writer_receipt.get("checkpoint_step")) is not int
        or not isinstance(writer_receipt.get("metadata_file"), str)
        or not isinstance(writer_receipt.get("tensor_file"), str)
    ):
        fail("training reference writer receipt cannot bind a checkpoint")
    parameter_sha = _require_sha(
        writer_receipt.get("checkpoint_parameter_sha256"),
        label="training reference checkpoint parameter SHA",
    )
    metadata_sha = _require_sha(
        writer_receipt.get("metadata_file_sha256"),
        label="training reference metadata SHA",
    )
    tensor_sha = _require_sha(
        writer_receipt.get("tensor_file_sha256"),
        label="training reference tensor SHA",
    )
    return {
        "schema_version": TRAINING_REFERENCE_BINDING_SCHEMA,
        "origin": "training_process_pre_checkpoint_export",
        "checkpoint_step": writer_receipt["checkpoint_step"],
        "checkpoint_parameter_sha256": parameter_sha,
        "metadata_file": writer_receipt["metadata_file"],
        "metadata_file_sha256": metadata_sha,
        "tensor_file": writer_receipt["tensor_file"],
        "tensor_file_sha256": tensor_sha,
    }


def _validated_tolerances(
    absolute_tolerance: Any, relative_tolerance: Any
) -> tuple[float, float]:
    values = (absolute_tolerance, relative_tolerance)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in values
    ):
        fail("fixed-forward parity tolerances must be finite and nonnegative")
    return float(absolute_tolerance), float(relative_tolerance)


def _tensor_differences(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    torch_module: Any,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[bool, bool, float, float, list[str]]:
    torch = torch_module
    if tuple(expected) != tuple(actual):
        fail("fixed parity tensor field set differs")
    exact = True
    bounded = True
    maximum_absolute = 0.0
    maximum_relative = 0.0
    mismatched = []
    for name in expected:
        left = expected[name]
        right = actual[name]
        if (
            not isinstance(left, torch.Tensor)
            or not isinstance(right, torch.Tensor)
            or tuple(left.shape) != tuple(right.shape)
            or left.dtype != right.dtype
        ):
            fail(f"fixed parity tensor ABI differs: {name}")
        if not bool(torch.equal(left, right)):
            exact = False
            mismatched.append(name)
        difference = (left.float() - right.float()).abs()
        maximum_absolute = max(
            maximum_absolute,
            float(difference.max().item()) if difference.numel() else 0.0,
        )
        denominator = left.float().abs().clamp_min(1.0e-12)
        relative = difference / denominator
        maximum_relative = max(
            maximum_relative,
            float(relative.max().item()) if relative.numel() else 0.0,
        )
        allowed = absolute_tolerance + relative_tolerance * left.float().abs()
        if not bool((difference <= allowed).all().item()):
            bounded = False
    return exact, bounded, maximum_absolute, maximum_relative, mismatched


def _require_fixed_inputs_bit_exact(
    left: Mapping[str, Any], right: Mapping[str, Any], *, torch_module: Any
) -> None:
    torch = torch_module
    for name in (
        "clean_source_tokens",
        "instruction_tokens",
        "inference_noise_hidden",
    ):
        if name not in left or name not in right or not bool(
            torch.equal(left[name], right[name])
        ):
            fail("fixed-forward parity inputs are not bit-exact")


def compare_fresh_a_b(
    *,
    conditioner_a: Any,
    conditioner_b: Any,
    predictor_module: Any,
    torch_module: Any,
    checkpoint_parameter_sha256: str,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> Mapping[str, Any]:
    """Post-hoc two-fresh-instance parity; never training-attached evidence."""

    parameter_sha = _require_sha(
        checkpoint_parameter_sha256, label="checkpoint parameter SHA"
    )
    absolute_tolerance, relative_tolerance = _validated_tolerances(
        absolute_tolerance, relative_tolerance
    )
    if conditioner_a is conditioner_b:
        fail("fresh A/B parity requires two disjoint conditioner objects")
    parameters_a = tuple(conditioner_a.parameters())
    parameters_b = tuple(conditioner_b.parameters())
    storage_identity = lambda parameter: (
        parameter.device.type,
        parameter.device.index,
        parameter.data_ptr(),
    )
    if (
        not parameters_a
        or len(parameters_a) != len(parameters_b)
        or {storage_identity(parameter) for parameter in parameters_a}
        & {storage_identity(parameter) for parameter in parameters_b}
    ):
        fail("fresh A/B conditioner parameter storage is shared or incomplete")
    left = fixed_forward_tensors(
        conditioner=conditioner_a,
        predictor_module=predictor_module,
        torch_module=torch_module,
    )
    right = fixed_forward_tensors(
        conditioner=conditioner_b,
        predictor_module=predictor_module,
        torch_module=torch_module,
    )
    _require_fixed_inputs_bit_exact(left, right, torch_module=torch_module)
    exact, bounded, max_abs, max_rel, mismatched = _tensor_differences(
        left,
        right,
        torch_module=torch_module,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    passed = exact or bounded
    if not passed:
        fail("two fresh instances exceed fixed-forward parity tolerance")
    return {
        "schema_version": FRESH_PARITY_SCHEMA,
        "method": METHOD,
        "authority": AUTHORITY,
        "promotable": False,
        "origin": "posthoc_two_disjoint_fresh_object_instances",
        "checkpoint_parameter_sha256": parameter_sha,
        "exact_parity": exact,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "maximum_absolute_difference": max_abs,
        "maximum_relative_difference": max_rel,
        "mismatched_tensor_names": mismatched,
        "exact_or_bounded_parity_pass": True,
        "elementwise_atol_plus_rtol_bound_used": True,
        "disjoint_object_and_parameter_storage_verified": True,
        "os_process_independence_proven": False,
        "checkpoint_bytes_independently_authenticated": False,
        "training_attached_reference": False,
        "training_to_fresh_forward_parity_claimed": False,
        "full_bernini_renderer_forward_parity_claimed": False,
        "left_fingerprint": fixed_forward_fingerprint(left, torch_module=torch_module),
        "right_fingerprint": fixed_forward_fingerprint(right, torch_module=torch_module),
    }


def verify_training_attached_reference(
    *,
    reference_metadata_path: str | Path,
    checkpoint_binding: Mapping[str, Any],
    conditioner: Any,
    predictor_module: Any,
    torch_module: Any,
    checkpoint_parameter_sha256: str,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> Mapping[str, Any]:
    """Verify a true pre-save training reference in a later fresh process."""

    absolute_tolerance, relative_tolerance = _validated_tolerances(
        absolute_tolerance, relative_tolerance
    )
    if not isinstance(checkpoint_binding, Mapping) or set(checkpoint_binding) != {
        "schema_version",
        "origin",
        "checkpoint_step",
        "checkpoint_parameter_sha256",
        "metadata_file",
        "metadata_file_sha256",
        "tensor_file",
        "tensor_file_sha256",
    }:
        fail("training reference checkpoint binding field set differs")
    metadata_expected = _require_sha(
        checkpoint_binding.get("metadata_file_sha256"),
        label="training reference metadata SHA",
    )
    parameter_sha = _require_sha(
        checkpoint_parameter_sha256, label="checkpoint parameter SHA"
    )
    path = Path(reference_metadata_path).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ProductABIError(
            f"training reference metadata is unavailable: {error}"
        ) from error
    if not path.is_absolute() or path.is_symlink() or resolved != path:
        fail("training reference metadata must be one canonical absolute file")
    if (
        checkpoint_binding.get("schema_version")
        != TRAINING_REFERENCE_BINDING_SCHEMA
        or checkpoint_binding.get("origin")
        != "training_process_pre_checkpoint_export"
        or checkpoint_binding.get("checkpoint_parameter_sha256") != parameter_sha
        or checkpoint_binding.get("metadata_file") != path.name
    ):
        fail("training reference checkpoint binding semantics differ")
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        payload = handle.read()
        after = os.fstat(handle.fileno())
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
    )
    if (
        identity(before) != identity(after)
        or hashlib.sha256(payload).hexdigest() != metadata_expected
    ):
        fail("training reference metadata stable bytes/SHA differ")
    try:
        metadata = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductABIError("training reference metadata is not UTF-8 JSON") from error
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema_version") != TRAINING_REFERENCE_SCHEMA
        or metadata.get("method") != METHOD
        or metadata.get("authority") != AUTHORITY
        or metadata.get("promotable") is not False
        or metadata.get("origin") != "training_process_pre_checkpoint_export"
        or metadata.get("execution_phase") != "immediately_before_save_checkpoint"
        or metadata.get("checkpoint_parameter_sha256") != parameter_sha
        or metadata.get("checkpoint_step") != checkpoint_binding.get("checkpoint_step")
        or metadata.get("fresh_consumer_parity_pass") is not False
    ):
        fail("training-attached reference authority differs")
    tensor_path = path.parent / str(metadata.get("tensor_file"))
    if (
        tensor_path.parent != path.parent
        or tensor_path.is_symlink()
        or not tensor_path.is_file()
        or file_sha256(tensor_path) != metadata.get("tensor_file_sha256")
        or metadata.get("tensor_file") != checkpoint_binding.get("tensor_file")
        or metadata.get("tensor_file_sha256")
        != checkpoint_binding.get("tensor_file_sha256")
    ):
        fail("training-attached reference tensor bytes differ")
    envelope = _torch_load_authenticated(
        tensor_path,
        expected_sha256=checkpoint_binding["tensor_file_sha256"],
        torch_module=torch_module,
        label="training reference tensor",
    )
    if (
        not isinstance(envelope, Mapping)
        or set(envelope) != {
            "schema_version",
            "origin",
            "checkpoint_step",
            "checkpoint_parameter_sha256",
            "tensors",
        }
        or envelope.get("schema_version") != REFERENCE_TENSOR_SCHEMA
        or envelope.get("origin") != "training_process_pre_checkpoint_export"
        or envelope.get("checkpoint_step") != metadata.get("checkpoint_step")
        or envelope.get("checkpoint_parameter_sha256") != parameter_sha
        or not isinstance(envelope.get("tensors"), Mapping)
    ):
        fail("training-attached reference tensor envelope differs")
    expected = envelope["tensors"]
    if fixed_forward_fingerprint(expected, torch_module=torch_module) != metadata.get(
        "fingerprint"
    ):
        fail("training-attached reference fingerprint differs")
    deterministic = deterministic_fixed_inputs(
        conditioner=conditioner,
        torch_module=torch_module,
        device=next(conditioner.parameters()).device,
    )
    for name, value in (
        ("clean_source_tokens", deterministic.clean_source_tokens),
        ("instruction_tokens", deterministic.instruction_tokens),
        ("inference_noise_hidden", deterministic.inference_noise_hidden),
    ):
        if not bool(torch_module.equal(expected[name], value.detach().cpu())):
            fail("training-attached reference fixed input bytes differ")
    actual = fixed_forward_tensors(
        conditioner=conditioner,
        predictor_module=predictor_module,
        torch_module=torch_module,
    )
    _require_fixed_inputs_bit_exact(expected, actual, torch_module=torch_module)
    exact, bounded, max_abs, max_rel, mismatched = _tensor_differences(
        expected,
        actual,
        torch_module=torch_module,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    passed = exact or bounded
    if not passed:
        fail("training-attached -> fresh forward parity exceeds tolerance")
    return {
        **dict(metadata),
        "metadata_file": path.name,
        "metadata_file_sha256": metadata_expected,
        "fresh_consumer_parity_pass": True,
        "exact_parity": exact,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "maximum_absolute_difference": max_abs,
        "maximum_relative_difference": max_rel,
        "mismatched_tensor_names": mismatched,
        "elementwise_atol_plus_rtol_bound_used": True,
        "fresh_fingerprint": fixed_forward_fingerprint(
            actual, torch_module=torch_module
        ),
        "training_to_fresh_forward_parity_verified": True,
        "conditioner_cell_training_to_fresh_forward_parity_verified": True,
        "full_bernini_renderer_training_to_fresh_forward_parity_verified": False,
        "promotable": False,
    }


def _validate_full_renderer_callback_contract(
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected_fields = {
        "schema_version",
        "training_callback_source_sha256",
        "fresh_callback_source_sha256",
        "input_semantics",
        "model_surface",
        "forward_coordinate",
        "source_instruction_only",
        "clean_target_or_anchor_consumed",
        "exact30_injection_required",
    }
    if not isinstance(contract, Mapping) or set(contract) != expected_fields:
        fail("full-renderer callback contract field set differs")
    for key in (
        "training_callback_source_sha256",
        "fresh_callback_source_sha256",
    ):
        _require_sha(contract.get(key), label=key)
    if (
        contract.get("schema_version") != FULL_RENDERER_CALLBACK_CONTRACT_SCHEMA
        or contract.get("input_semantics")
        != "deterministic_source_instruction_inference_noise_embeddings_v1"
        or contract.get("model_surface")
        != "full_persisted_bernini_renderer_plus_ActionPlanConditionerV1"
        or contract.get("forward_coordinate")
        != "one_fixed_denoiser_step_pre_decode"
        or contract.get("source_instruction_only") is not True
        or contract.get("clean_target_or_anchor_consumed") is not False
        or contract.get("exact30_injection_required") is not True
    ):
        fail("full-renderer callback semantic contract differs")
    return dict(contract)


def _full_renderer_callback_tensors(
    *,
    callback: Any,
    callback_role: str,
    callback_contract: Mapping[str, Any],
    checkpoint_parameter_sha256: str,
    torch_module: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not callable(callback) or tuple(inspect.signature(callback).parameters):
        fail("full-renderer fixed-forward callback must be a zero-argument closure")
    if callback_role not in ("training_pre_save", "fresh_consumer"):
        fail("full-renderer callback role differs")
    source_key = (
        "training_callback_source_sha256"
        if callback_role == "training_pre_save"
        else "fresh_callback_source_sha256"
    )
    callback_source_sha = callback_contract[source_key]
    callback_code = getattr(callback, "__code__", None)
    callback_source_name = getattr(callback_code, "co_filename", None)
    if not isinstance(callback_source_name, str):
        fail("full-renderer callback must expose one auditable source file")
    callback_source_path = Path(callback_source_name).expanduser()
    try:
        callback_source_resolved = callback_source_path.resolve(strict=True)
    except OSError as error:
        raise ProductABIError(
            f"full-renderer callback source is unavailable: {error}"
        ) from error
    # ``co_filename`` follows the path spelling used to launch/compile the
    # owning script.  A normal ``python relative/test.py`` therefore records a
    # relative name even though it refers to the same canonical auditable
    # source.  Resolve that spelling first, then hash only the canonical file.
    if callback_source_path.is_symlink() or not callback_source_resolved.is_file():
        fail("full-renderer callback source canonical file differs")
    callback_source_path = callback_source_resolved
    observed_callback_sha, _ = _stable_file_sha256(
        callback_source_path, label="full-renderer callback source"
    )
    if observed_callback_sha != callback_source_sha:
        fail("full-renderer callback source SHA differs")
    torch = torch_module
    python_before = random.getstate()
    cpu_before = torch.get_rng_state().detach().cpu().clone()
    cuda_before = (
        [value.detach().cpu().clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    try:
        result = callback()
    except Exception:
        random.setstate(python_before)
        torch.set_rng_state(cpu_before)
        if cuda_before:
            torch.cuda.set_rng_state_all(cuda_before)
        raise
    cuda_after = (
        [value.detach().cpu().clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    if (
        random.getstate() != python_before
        or not bool(torch.equal(torch.get_rng_state().cpu(), cpu_before))
        or len(cuda_before) != len(cuda_after)
        or any(
            not bool(torch.equal(before, after))
            for before, after in zip(cuda_before, cuda_after)
        )
    ):
        random.setstate(python_before)
        torch.set_rng_state(cpu_before)
        if cuda_before:
            torch.cuda.set_rng_state_all(cuda_before)
        fail("full-renderer fixed forward mutated ambient RNG")
    if not isinstance(result, Mapping) or set(result) != {
        "schema_version",
        "tensors",
        "execution_receipt",
    }:
        fail("full-renderer callback result envelope differs")
    raw_tensors = result.get("tensors")
    execution = result.get("execution_receipt")
    if (
        result.get("schema_version") != FULL_RENDERER_CALLBACK_RESULT_SCHEMA
        or not isinstance(raw_tensors, Mapping)
        or not 1 <= len(raw_tensors) <= 32
        or tuple(raw_tensors) != tuple(sorted(raw_tensors))
        or any(
            re.fullmatch(r"[a-z][a-z0-9_.]{0,127}", str(name)) is None
            for name in raw_tensors
        )
    ):
        fail("full-renderer callback tensor map differs")
    expected_execution = {
        "schema_version": "bernini-action-edit-full-renderer-forward-execution-v1",
        "callback_role": callback_role,
        "callback_source_sha256": callback_source_sha,
        "checkpoint_parameter_sha256": checkpoint_parameter_sha256,
        "source_instruction_inference_noise_only": True,
        "clean_target_or_anchor_consumed": False,
        "same_persisted_conditioner_and_30_heads": True,
        "persisted_trainable_bytes_unchanged": True,
        "exact_block_indices": list(range(TRANSFORMER_BLOCKS)),
        "full_bernini_renderer_forward_executed": True,
        "forward_coordinate": "one_fixed_denoiser_step_pre_decode",
        "model_mode": "eval_fixed_forward_with_training_state_bytes",
    }
    if not isinstance(execution, Mapping) or dict(execution) != expected_execution:
        fail("full-renderer callback execution receipt differs")
    tensors = {}
    for name, value in raw_tensors.items():
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or value.requires_grad
            or not bool(torch.isfinite(value).all().item())
        ):
            fail(f"full-renderer callback tensor ABI differs: {name}")
        tensors[name] = value.detach().cpu().contiguous()
    return tensors, dict(execution)


def _arbitrary_tensor_map_fingerprint(
    tensors: Mapping[str, Any], *, torch_module: Any
) -> Mapping[str, Any]:
    rows = [
        {
            "name": name,
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
            "sha256": tensor_sha256(value, torch_module=torch_module),
        }
        for name, value in tensors.items()
    ]
    return {
        "schema_version": "bernini-action-edit-arbitrary-tensor-map-fingerprint-v1",
        "tensors": rows,
        "tensor_set_sha256": object_sha256(rows),
    }


def write_training_attached_full_renderer_reference(
    *,
    output_dir: str | Path,
    full_forward_callback: Any,
    callback_contract: Mapping[str, Any],
    torch_module: Any,
    checkpoint_step: int,
    checkpoint_parameter_sha256: str,
    execution_phase: str,
) -> Mapping[str, Any]:
    """Future-r3 hook: persist one authenticated full-renderer pre-save forward."""

    root = _plain_existing_directory(output_dir, label="full-renderer reference output")
    contract = _validate_full_renderer_callback_contract(callback_contract)
    parameter_sha = _require_sha(
        checkpoint_parameter_sha256, label="checkpoint parameter SHA"
    )
    if (
        type(checkpoint_step) is not int
        or checkpoint_step < 0
        or execution_phase != "immediately_before_save_checkpoint"
    ):
        fail("full-renderer reference writer execution coordinate differs")
    tensors, execution = _full_renderer_callback_tensors(
        callback=full_forward_callback,
        callback_role="training_pre_save",
        callback_contract=contract,
        checkpoint_parameter_sha256=parameter_sha,
        torch_module=torch_module,
    )
    fingerprint = _arbitrary_tensor_map_fingerprint(
        tensors, torch_module=torch_module
    )
    stem = f"training-full-renderer-forward-{checkpoint_step:08d}"
    final_tensor = root / f"{stem}.pt"
    final_metadata = root / f"{stem}.json"
    if final_tensor.exists() or final_metadata.exists():
        fail("full-renderer training reference is create-only")
    temporary = Path(tempfile.mkdtemp(prefix=f".{stem}.", dir=root))
    tensor_path = temporary / final_tensor.name
    metadata_path = temporary / final_metadata.name
    torch_module.save(
        {
            "schema_version": FULL_RENDERER_TENSOR_SCHEMA,
            "checkpoint_step": checkpoint_step,
            "checkpoint_parameter_sha256": parameter_sha,
            "tensors": dict(tensors),
        },
        tensor_path,
    )
    payload = {
        "schema_version": FULL_RENDERER_REFERENCE_SCHEMA,
        "method": METHOD,
        "authority": AUTHORITY,
        "promotable": False,
        "origin": "training_process_pre_checkpoint_export",
        "execution_phase": execution_phase,
        "checkpoint_step": checkpoint_step,
        "checkpoint_parameter_sha256": parameter_sha,
        "callback_contract": contract,
        "callback_contract_sha256": object_sha256(contract),
        "training_execution_receipt": execution,
        "tensor_file": final_tensor.name,
        "tensor_file_sha256": file_sha256(tensor_path),
        "fingerprint": fingerprint,
        "fresh_consumer_parity_pass": False,
        "callback_protocol_requires_future_runner_release_pin": True,
    }
    metadata_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    for path in (tensor_path, metadata_path):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    _fsync_directory(temporary)
    tensor_published = False
    metadata_published = False
    try:
        os.link(tensor_path, final_tensor)
        tensor_published = True
        os.link(metadata_path, final_metadata)
        metadata_published = True
        _fsync_directory(root)
    except Exception:
        if metadata_published:
            final_metadata.unlink()
        if tensor_published:
            final_tensor.unlink()
        _fsync_directory(root)
        raise
    finally:
        tensor_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        temporary.rmdir()
    return {
        **payload,
        "metadata_file": final_metadata.name,
        "metadata_file_sha256": file_sha256(final_metadata),
    }


def full_renderer_reference_checkpoint_binding(
    writer_receipt: Mapping[str, Any]
) -> Mapping[str, Any]:
    if (
        not isinstance(writer_receipt, Mapping)
        or writer_receipt.get("schema_version") != FULL_RENDERER_REFERENCE_SCHEMA
        or writer_receipt.get("origin")
        != "training_process_pre_checkpoint_export"
        or writer_receipt.get("execution_phase")
        != "immediately_before_save_checkpoint"
    ):
        fail("full-renderer writer receipt cannot bind a checkpoint")
    return {
        "schema_version": FULL_RENDERER_REFERENCE_BINDING_SCHEMA,
        "origin": "training_process_pre_checkpoint_export",
        "checkpoint_step": writer_receipt["checkpoint_step"],
        "checkpoint_parameter_sha256": _require_sha(
            writer_receipt.get("checkpoint_parameter_sha256"),
            label="full-renderer checkpoint parameter SHA",
        ),
        "callback_contract_sha256": _require_sha(
            writer_receipt.get("callback_contract_sha256"),
            label="full-renderer callback contract SHA",
        ),
        "metadata_file": writer_receipt["metadata_file"],
        "metadata_file_sha256": _require_sha(
            writer_receipt.get("metadata_file_sha256"),
            label="full-renderer metadata SHA",
        ),
        "tensor_file": writer_receipt["tensor_file"],
        "tensor_file_sha256": _require_sha(
            writer_receipt.get("tensor_file_sha256"),
            label="full-renderer tensor SHA",
        ),
    }


def verify_training_attached_full_renderer_reference(
    *,
    reference_metadata_path: str | Path,
    checkpoint_binding: Mapping[str, Any],
    fresh_full_forward_callback: Any,
    expected_callback_contract: Mapping[str, Any],
    torch_module: Any,
    checkpoint_parameter_sha256: str,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> Mapping[str, Any]:
    """Future-r3 consumer side of the full-renderer fixed-forward protocol."""

    absolute_tolerance, relative_tolerance = _validated_tolerances(
        absolute_tolerance, relative_tolerance
    )
    contract = _validate_full_renderer_callback_contract(
        expected_callback_contract
    )
    parameter_sha = _require_sha(
        checkpoint_parameter_sha256, label="checkpoint parameter SHA"
    )
    expected_binding_fields = {
        "schema_version",
        "origin",
        "checkpoint_step",
        "checkpoint_parameter_sha256",
        "callback_contract_sha256",
        "metadata_file",
        "metadata_file_sha256",
        "tensor_file",
        "tensor_file_sha256",
    }
    if (
        not isinstance(checkpoint_binding, Mapping)
        or set(checkpoint_binding) != expected_binding_fields
        or checkpoint_binding.get("schema_version")
        != FULL_RENDERER_REFERENCE_BINDING_SCHEMA
        or checkpoint_binding.get("origin")
        != "training_process_pre_checkpoint_export"
        or checkpoint_binding.get("checkpoint_parameter_sha256") != parameter_sha
        or checkpoint_binding.get("callback_contract_sha256")
        != object_sha256(contract)
    ):
        fail("full-renderer checkpoint reference binding differs")
    metadata_path = Path(reference_metadata_path).expanduser()
    try:
        resolved = metadata_path.resolve(strict=True)
    except OSError as error:
        raise ProductABIError(
            f"full-renderer reference metadata is unavailable: {error}"
        ) from error
    if (
        not metadata_path.is_absolute()
        or metadata_path.is_symlink()
        or resolved != metadata_path
        or metadata_path.name != checkpoint_binding.get("metadata_file")
    ):
        fail("full-renderer reference metadata path differs")
    metadata_expected_sha = _require_sha(
        checkpoint_binding.get("metadata_file_sha256"),
        label="full-renderer reference metadata SHA",
    )
    with metadata_path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        payload = handle.read()
        after = os.fstat(handle.fileno())
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
    )
    if (
        identity(before) != identity(after)
        or hashlib.sha256(payload).hexdigest() != metadata_expected_sha
    ):
        fail("full-renderer reference metadata stable bytes differ")
    try:
        metadata = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductABIError(
            "full-renderer reference metadata is not UTF-8 JSON"
        ) from error
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema_version") != FULL_RENDERER_REFERENCE_SCHEMA
        or metadata.get("method") != METHOD
        or metadata.get("authority") != AUTHORITY
        or metadata.get("promotable") is not False
        or metadata.get("origin") != "training_process_pre_checkpoint_export"
        or metadata.get("execution_phase")
        != "immediately_before_save_checkpoint"
        or metadata.get("checkpoint_step")
        != checkpoint_binding.get("checkpoint_step")
        or metadata.get("checkpoint_parameter_sha256") != parameter_sha
        or metadata.get("callback_contract") != contract
        or metadata.get("callback_contract_sha256") != object_sha256(contract)
        or metadata.get("fresh_consumer_parity_pass") is not False
        or metadata.get("callback_protocol_requires_future_runner_release_pin")
        is not True
    ):
        fail("full-renderer training reference authority differs")
    tensor_name = metadata.get("tensor_file")
    if (
        not isinstance(tensor_name, str)
        or Path(tensor_name).name != tensor_name
        or tensor_name != checkpoint_binding.get("tensor_file")
        or metadata.get("tensor_file_sha256")
        != checkpoint_binding.get("tensor_file_sha256")
    ):
        fail("full-renderer reference tensor binding differs")
    tensor_path = metadata_path.parent / tensor_name
    try:
        tensor_resolved = tensor_path.resolve(strict=True)
    except OSError as error:
        raise ProductABIError(
            f"full-renderer reference tensor is unavailable: {error}"
        ) from error
    if (
        tensor_path.is_symlink()
        or not tensor_path.is_file()
        or tensor_resolved != tensor_path
        or tensor_path.parent != metadata_path.parent
    ):
        fail("full-renderer reference tensor path differs")
    envelope = _torch_load_authenticated(
        tensor_path,
        expected_sha256=checkpoint_binding["tensor_file_sha256"],
        torch_module=torch_module,
        label="full-renderer reference tensor",
    )
    if (
        not isinstance(envelope, Mapping)
        or set(envelope) != {
            "schema_version",
            "checkpoint_step",
            "checkpoint_parameter_sha256",
            "tensors",
        }
        or envelope.get("schema_version") != FULL_RENDERER_TENSOR_SCHEMA
        or envelope.get("checkpoint_step") != metadata.get("checkpoint_step")
        or envelope.get("checkpoint_parameter_sha256") != parameter_sha
        or not isinstance(envelope.get("tensors"), Mapping)
    ):
        fail("full-renderer reference tensor envelope differs")
    expected = envelope["tensors"]
    if _arbitrary_tensor_map_fingerprint(
        expected, torch_module=torch_module
    ) != metadata.get("fingerprint"):
        fail("full-renderer reference tensor fingerprint differs")
    actual, fresh_execution = _full_renderer_callback_tensors(
        callback=fresh_full_forward_callback,
        callback_role="fresh_consumer",
        callback_contract=contract,
        checkpoint_parameter_sha256=parameter_sha,
        torch_module=torch_module,
    )
    exact, bounded, max_abs, max_rel, mismatched = _tensor_differences(
        expected,
        actual,
        torch_module=torch_module,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    if not (exact or bounded):
        fail("training-attached full-renderer forward parity exceeds tolerance")
    return {
        **dict(metadata),
        "metadata_file": metadata_path.name,
        "metadata_file_sha256": metadata_expected_sha,
        "fresh_execution_receipt": fresh_execution,
        "fresh_consumer_parity_pass": True,
        "exact_parity": exact,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "maximum_absolute_difference": max_abs,
        "maximum_relative_difference": max_rel,
        "mismatched_tensor_names": mismatched,
        "training_to_fresh_forward_parity_verified": True,
        # This protocol compares the selected full-renderer tensors.  A
        # separate fixed-cell writer is the authority for an isolated
        # conditioner-cell parity claim.
        "conditioner_cell_training_to_fresh_forward_parity_verified": False,
        "full_bernini_renderer_training_to_fresh_forward_parity_verified": True,
        "offline_full40_product_inference_completed": False,
        "promotable": False,
    }


@dataclass
class OfflineLocalActionRoute:
    plan: Any
    source_tokens: int
    inference_tokens: int
    sequence_parallel_rank: int
    sequence_parallel_size: int
    row_identity: str
    calls: list[int] = field(default_factory=list)
    active: bool = False
    closed: bool = False

    def validate_geometry(self) -> None:
        if (
            self.source_tokens <= 0
            or self.inference_tokens != self.source_tokens
            or self.inference_tokens % PHASES
            or self.sequence_parallel_size not in (1, SP_SIZE)
            or not 0 <= self.sequence_parallel_rank < self.sequence_parallel_size
            or not self.row_identity
        ):
            fail("offline exact30 local route geometry differs")

    @property
    def total_tokens(self) -> int:
        return self.source_tokens + self.inference_tokens

    @property
    def local_length(self) -> int:
        return math.ceil(self.total_tokens / self.sequence_parallel_size)

    @property
    def spatial_tokens_per_phase(self) -> int:
        return self.inference_tokens // PHASES

    def local_phase_indices_tuple(self) -> tuple[int, ...]:
        start = self.sequence_parallel_rank * self.local_length
        result = []
        for global_index in range(start, start + self.local_length):
            inference_index = global_index - self.source_tokens
            if 0 <= inference_index < self.inference_tokens:
                result.append(inference_index // self.spatial_tokens_per_phase)
            else:
                result.append(-1)
        return tuple(result)

    def record(self, block_index: int) -> None:
        if not self.active or self.closed or block_index != len(self.calls):
            fail("offline action injection did not traverse exact blocks 0..29")
        self.calls.append(block_index)

    def finish(self) -> Mapping[str, Any]:
        if self.active or not self.closed or tuple(self.calls) != tuple(range(TRANSFORMER_BLOCKS)):
            fail("offline exact30 action route did not close")
        return {
            "row_identity": self.row_identity,
            "source_tokens": self.source_tokens,
            "inference_tokens": self.inference_tokens,
            "sequence_parallel_rank": self.sequence_parallel_rank,
            "sequence_parallel_size": self.sequence_parallel_size,
            "local_phase_indices_sha256": object_sha256(
                list(self.local_phase_indices_tuple())
            ),
            "exact_block_indices": list(self.calls),
            "source_and_padding_bit_exact_under_injection": True,
            "clean_target_present": False,
            "anchor_present": False,
        }


_ACTIVE_OFFLINE_ROUTE: ContextVar[Optional[OfflineLocalActionRoute]] = ContextVar(
    "bernini_action_edit_offline_product_route_v1", default=None
)


@contextmanager
def activate_offline_route(route: OfflineLocalActionRoute) -> Iterator[None]:
    if _ACTIVE_OFFLINE_ROUTE.get() is not None or route.active or route.closed:
        fail("offline product routes cannot be nested or reused")
    route.validate_geometry()
    route.active = True
    token: Token[Optional[OfflineLocalActionRoute]] = _ACTIVE_OFFLINE_ROUTE.set(route)
    try:
        yield
    finally:
        _ACTIVE_OFFLINE_ROUTE.reset(token)
        route.active = False
        route.closed = True


def _output_tensor(output: Any, *, torch_module: Any) -> tuple[Any, Any]:
    torch = torch_module
    if isinstance(output, torch.Tensor):
        return output, lambda value: value
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0], lambda value: (value, *output[1:])
    fail("offline Bernini block output must be Tensor or tensor-first tuple")


def _bits_equal(left: Any, right: Any, *, torch_module: Any) -> bool:
    torch = torch_module
    return bool(
        torch.equal(
            left.detach().contiguous().reshape(-1).view(torch.uint8),
            right.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    )


@dataclass
class InstalledOfflineActionHooks:
    transformer: Any
    conditioner: Any
    handles: tuple[Any, ...]
    restored: bool = False

    def restore(self) -> None:
        if self.restored or _ACTIVE_OFFLINE_ROUTE.get() is not None:
            fail("cannot restore active/already-restored offline hooks")
        for handle in self.handles:
            handle.remove()
        if getattr(self.transformer, "action_plan_conditioner_v1", None) is not self.conditioner:
            fail("offline conditioner ownership changed")
        delattr(self.transformer, "action_plan_conditioner_v1")
        self.restored = True


def install_offline_action_plan_hooks(
    *, transformer: Any, conditioner: Any, torch_module: Any
) -> InstalledOfflineActionHooks:
    """Install inference-only routing around the same persisted injection module."""

    torch = torch_module
    if (
        not callable(getattr(conditioner, "parameters", None))
        or not callable(
            getattr(getattr(conditioner, "injection", None), "validate_block_traversal", None)
        )
    ):
        fail("offline ActionPlanConditionerV1 interface differs")
    conditioner.injection.validate_block_traversal(tuple(range(TRANSFORMER_BLOCKS)))
    blocks = tuple(getattr(transformer, "blocks", ()))
    if len(blocks) != TRANSFORMER_BLOCKS or hasattr(
        transformer, "action_plan_conditioner_v1"
    ):
        fail("offline Bernini exact30 transformer structure differs")
    if any(
        not callable(getattr(block, "register_forward_hook", None))
        or bool(getattr(block, "_forward_hooks", {}))
        for block in blocks
    ):
        fail("offline exact30 blocks require empty auditable hook registries")
    transformer.add_module("action_plan_conditioner_v1", conditioner)
    handles = []
    try:
        for block_index, block in enumerate(blocks):
            def callback(
                _module: Any,
                _args: tuple[Any, ...],
                output: Any,
                *,
                bound_index: int = block_index,
            ) -> Any:
                route = _ACTIVE_OFFLINE_ROUTE.get()
                if route is None:
                    fail("offline Bernini block executed without an authenticated route")
                native, rebuild = _output_tensor(output, torch_module=torch)
                if (
                    native.ndim != 3
                    or int(native.shape[0]) != 1
                    or int(native.shape[1]) != route.local_length
                    or int(native.shape[2]) != int(conditioner.renderer_hidden_width)
                    or not native.is_floating_point()
                    or not bool(torch.isfinite(native).all().item())
                ):
                    fail("offline Bernini local hidden geometry differs")
                residual = conditioner.injection.residual(
                    route.plan, block_index=bound_index
                )
                if (
                    tuple(residual.shape)
                    != (1, PHASES, int(conditioner.renderer_hidden_width))
                    or residual.device != native.device
                    or not bool(torch.isfinite(residual).all().item())
                ):
                    fail("offline block-indexed phase residual ABI differs")
                phases = torch.tensor(
                    route.local_phase_indices_tuple(),
                    dtype=torch.int64,
                    device=native.device,
                )
                selector = phases >= 0
                local_delta = residual.index_select(1, phases.clamp_min(0))
                target_adapted = native + local_delta.to(dtype=native.dtype)
                adapted = torch.where(selector.view(1, -1, 1), target_adapted, native)
                if bool((~selector).any().item()) and not _bits_equal(
                    adapted[:, ~selector, :], native[:, ~selector, :], torch_module=torch
                ):
                    fail("offline action injection changed source or padding bytes")
                route.record(bound_index)
                return rebuild(adapted)
            handles.append(block.register_forward_hook(callback))
    except Exception:
        for handle in handles:
            handle.remove()
        delattr(transformer, "action_plan_conditioner_v1")
        raise
    return InstalledOfflineActionHooks(
        transformer=transformer,
        conditioner=conditioner,
        handles=tuple(handles),
    )


def prepare_product_route_from_packed_embeddings(
    *,
    conditioner: Any,
    predictor_module: Any,
    packed_embeddings: Any,
    instruction_tokens: Any,
    source_token_count: int,
    patch_grid: Sequence[int],
    sequence_parallel_rank: int,
    sequence_parallel_size: int,
    row_identity: str,
    torch_module: Any,
) -> OfflineLocalActionRoute:
    """Bridge complete pre-SP clean source tokens to the official block hooks."""

    torch = torch_module
    if (
        not isinstance(packed_embeddings, torch.Tensor)
        or packed_embeddings.ndim != 3
        or int(packed_embeddings.shape[0]) != 1
        or type(source_token_count) is not int
        or source_token_count <= 0
        or int(packed_embeddings.shape[1]) != 2 * source_token_count
        or len(patch_grid) != 3
        or math.prod(int(value) for value in patch_grid) != source_token_count
    ):
        fail("offline complete pre-SP packed embedding geometry differs")
    phases, height, width = (int(value) for value in patch_grid)
    if phases != PHASES:
        fail("offline product path requires the exact21 latent phases")
    source = packed_embeddings[:, :source_token_count, :].reshape(
        1, phases, height, width, int(packed_embeddings.shape[-1])
    )
    evolving_noise = packed_embeddings[:, source_token_count:, :].reshape(
        1, phases, height, width, int(packed_embeddings.shape[-1])
    )
    if (
        not isinstance(instruction_tokens, torch.Tensor)
        or instruction_tokens.ndim != 3
        or int(instruction_tokens.shape[0]) != 1
        or instruction_tokens.device != source.device
        or not instruction_tokens.is_floating_point()
        or instruction_tokens.requires_grad
        or not bool(torch.isfinite(instruction_tokens).all().item())
    ):
        fail("offline contextual instruction tensor ABI differs")
    # This is the same explicit boundary cast used by the training runner.
    instruction_tokens = instruction_tokens.to(dtype=source.dtype).contiguous()
    inputs = ProductActionInputsV1(
        clean_source_tokens=source,
        instruction_tokens=instruction_tokens,
        inference_noise_hidden=evolving_noise,
    )
    plan_route = prepare_product_action_route(
        conditioner=conditioner,
        inputs=inputs,
        predictor_module=predictor_module,
        torch_module=torch,
    )
    return OfflineLocalActionRoute(
        plan=plan_route,
        source_tokens=source_token_count,
        inference_tokens=source_token_count,
        sequence_parallel_rank=sequence_parallel_rank,
        sequence_parallel_size=sequence_parallel_size,
        row_identity=row_identity,
    )


def execute_offline_denoiser_step(
    *,
    route: OfflineLocalActionRoute,
    denoiser_step: Any,
    denoiser_kwargs: Mapping[str, Any],
    inference_policy: OfflineInferencePolicyV1,
    scheduler: Any,
    sigma_contract_module: Any,
    inference_step_index: int,
    runtime_timestep: int,
) -> tuple[Any, Mapping[str, Any]]:
    """Execute one official denoiser call with exactly one 0..29 hook traversal."""

    inference_policy.validate()
    if (
        not callable(denoiser_step)
        or not isinstance(denoiser_kwargs, Mapping)
        or type(inference_step_index) is not int
        or not 0 <= inference_step_index < inference_policy.num_inference_steps
        or type(runtime_timestep) is not int
    ):
        fail("offline denoiser call ABI differs")
    scheduler_receipt = audit_live_inference_scheduler(
        scheduler=scheduler,
        inference_policy=inference_policy,
        sigma_contract_module=sigma_contract_module,
        initialize=False,
    )
    expected_timestep = scheduler_receipt["schedule"]["timesteps"][
        inference_step_index
    ]
    if runtime_timestep != expected_timestep:
        fail("offline denoiser runtime timestep differs from exact40 schedule")
    with activate_offline_route(route):
        output = denoiser_step(**dict(denoiser_kwargs))
    return output, {
        **dict(route.finish()),
        "inference_step_index": inference_step_index,
        "scheduler_schedule_sha256": scheduler_receipt["schedule_sha256"],
        "runtime_timestep": runtime_timestep,
        "scheduler_timestep_bound_to_step": True,
        "live_scheduler_verified_before_denoiser": True,
    }


def build_nonpromotable_product_receipt(
    *,
    checkpoint_consumer_receipt: Mapping[str, Any],
    inference_policy: OfflineInferencePolicyV1,
    route_receipts: Sequence[Mapping[str, Any]],
    checkpoint_world8_consensus: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    inference_policy.validate()
    if not all(isinstance(row, Mapping) for row in route_receipts):
        fail("offline product route receipt sequence differs")
    if (
        checkpoint_consumer_receipt.get("schema_version")
        != CHECKPOINT_CONSUMER_SCHEMA
        or checkpoint_consumer_receipt.get("authority") != AUTHORITY
        or checkpoint_consumer_receipt.get("promotable") is not False
        or checkpoint_consumer_receipt.get("promotion_authorized") is not False
        or checkpoint_consumer_receipt.get("world8_consumer_complete") is not True
        or checkpoint_consumer_receipt.get(
            "fresh_world8_process_forward_exact_consensus_verified"
        )
        is not True
        or checkpoint_consumer_receipt.get("fresh_world8_process_forward_scope")
        != "conditioner_predictor_plus_exact30_cell_only_not_bernini_renderer"
        or checkpoint_consumer_receipt.get("full_bernini_renderer_forward_executed")
        is not False
        or checkpoint_consumer_receipt.get("training_attached_reference_absent")
        is not True
        or checkpoint_consumer_receipt.get(
            "training_attached_full_renderer_reference_absent"
        )
        is not True
        or checkpoint_consumer_receipt.get(
            "training_to_fresh_forward_parity_verified"
        )
        is not False
        or checkpoint_consumer_receipt.get("loaded_parameter_sha256")
        != checkpoint_consumer_receipt.get("checkpoint_parameter_sha256")
        or len(route_receipts) != inference_policy.num_inference_steps
        or [row.get("inference_step_index") for row in route_receipts]
        != list(range(inference_policy.num_inference_steps))
        or any(
            row.get("exact_block_indices") != list(range(TRANSFORMER_BLOCKS))
            or row.get("clean_target_present") is not False
            or row.get("anchor_present") is not False
            or row.get("scheduler_schedule_sha256")
            != inference_policy.schedule_sha256
            or row.get("scheduler_timestep_bound_to_step") is not True
            or row.get("runtime_timestep")
            != PINNED_UNIPC_TIMESTEPS[row.get("inference_step_index")]
            or row.get("live_scheduler_verified_before_denoiser") is not True
            for row in route_receipts
        )
    ):
        fail("offline product receipt inputs differ")
    if (
        not isinstance(checkpoint_world8_consensus, Mapping)
        or checkpoint_world8_consensus.get("all8_exact_consensus") is not True
        or checkpoint_world8_consensus.get("world_size") != WORLD_SIZE
        or checkpoint_world8_consensus.get("eight_distinct_fresh_process_sessions")
        is not True
        or checkpoint_consumer_receipt.get("world8_consensus")
        != checkpoint_world8_consensus
    ):
        fail("offline product WORLD8 consensus differs")
    route_identity_fields = (
        "row_identity",
        "source_tokens",
        "inference_tokens",
        "sequence_parallel_rank",
        "sequence_parallel_size",
        "local_phase_indices_sha256",
    )
    first_route_identity = {
        key: route_receipts[0].get(key) for key in route_identity_fields
    }
    if (
        not isinstance(first_route_identity["row_identity"], str)
        or not first_route_identity["row_identity"]
        or type(first_route_identity["source_tokens"]) is not int
        or first_route_identity["source_tokens"] <= 0
        or first_route_identity["inference_tokens"]
        != first_route_identity["source_tokens"]
        or first_route_identity["inference_tokens"] % PHASES
        or first_route_identity["sequence_parallel_size"] not in (1, SP_SIZE)
        or type(first_route_identity["sequence_parallel_rank"]) is not int
        or not 0
        <= first_route_identity["sequence_parallel_rank"]
        < first_route_identity["sequence_parallel_size"]
        or _SHA256.fullmatch(str(first_route_identity["local_phase_indices_sha256"]))
        is None
        or any(
            {key: row.get(key) for key in route_identity_fields}
            != first_route_identity
            for row in route_receipts
        )
    ):
        fail("offline product route identity changed across exact40 steps")
    unsigned = {
        "schema_version": PRODUCT_RECEIPT_SCHEMA,
        "method": METHOD,
        "authority": AUTHORITY,
        "complete": True,
        "promotable": False,
        "formal_training_started": False,
        "counts_as_d0": False,
        "scientific_claim_authorized": False,
        "action_quality_claim_authorized": False,
        "product_abi": validate_public_product_signatures(),
        "checkpoint_parameter_sha256": checkpoint_consumer_receipt[
            "checkpoint_parameter_sha256"
        ],
        "checkpoint_consumer_receipt_sha256": object_sha256(
            dict(checkpoint_consumer_receipt)
        ),
        "inference_policy": dict(inference_policy.receipt()),
        "route_receipts": [dict(row) for row in route_receipts],
        "world8_checkpoint_load_consensus": dict(checkpoint_world8_consensus),
        "completion_scope": (
            "one_rank_exact40_callback_bridge_smoke_with_"
            "world8_checkpoint_load_consensus"
        ),
        "world8_product_execution_consensus_verified": False,
        "engineering_bridge_smoke_only": True,
        "full_bernini_renderer_denoise_verified": False,
        "offline_product_inference_completed": False,
        "mp4_emitted": False,
        "training_attached_reference_absent": checkpoint_consumer_receipt.get(
            "training_attached_reference_absent", True
        ),
        "training_attached_full_renderer_reference_absent": True,
        "training_to_fresh_forward_parity_verified": checkpoint_consumer_receipt.get(
            "training_to_fresh_forward_parity_verified", False
        ),
        "conditioner_cell_training_to_fresh_forward_parity_verified": (
            checkpoint_consumer_receipt.get(
                "conditioner_cell_training_to_fresh_forward_parity_verified",
                False,
            )
        ),
        "full_bernini_renderer_training_to_fresh_forward_parity_verified": False,
        "source_video_plus_instruction_product_semantics_declared_only": True,
        "materialized_product_request_bound": False,
        "clean_target_or_anchor_consumed": False,
        "promotion_authorized": False,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


__all__ = [
    "AUTHORITY",
    "CHECKPOINT_CONSUMER_SCHEMA",
    "FRESH_PARITY_SCHEMA",
    "FULL_RENDERER_CALLBACK_CONTRACT_SCHEMA",
    "FULL_RENDERER_CALLBACK_RESULT_SCHEMA",
    "FULL_RENDERER_REFERENCE_BINDING_SCHEMA",
    "FULL_RENDERER_REFERENCE_SCHEMA",
    "INFERENCE_POLICY_SCHEMA",
    "InstalledOfflineActionHooks",
    "METHOD",
    "OfflineInferencePolicyV1",
    "OfflineLocalActionRoute",
    "PINNED_UNIPC_SCHEDULE_SHA256",
    "PINNED_UNIPC_TIMESTEPS",
    "PRODUCT_ABI_SCHEMA",
    "ProductABIError",
    "ProductActionInputsV1",
    "ProductActionOutputV1",
    "ProductRequestV1",
    "TRAINING_REFERENCE_SCHEMA",
    "TRAINING_REFERENCE_BINDING_SCHEMA",
    "activate_offline_route",
    "audit_live_inference_scheduler",
    "build_nonpromotable_product_receipt",
    "compare_fresh_a_b",
    "deterministic_fixed_inputs",
    "execute_offline_denoiser_step",
    "fixed_forward_fingerprint",
    "fixed_forward_tensors",
    "full_renderer_reference_checkpoint_binding",
    "install_offline_action_plan_hooks",
    "materialize_product_request",
    "prepare_product_action_route",
    "prepare_product_route_from_packed_embeddings",
    "run_product_action_fixed_cell",
    "training_reference_checkpoint_binding",
    "validate_public_product_signatures",
    "verify_training_attached_reference",
    "verify_training_attached_full_renderer_reference",
    "write_training_attached_full_renderer_reference",
    "write_training_attached_fixed_forward_reference",
]
