#!/usr/bin/env python3
"""Official frozen-Bernini field smoke for HAT/IAR.

This executable is an engineering gate, not a trainer.  It binds one native
T2V proposal ``P``, one correct-source latent ``S``, one CPU-generated
Gaussian ``epsilon``, and a caller-declared homotopy.  Every query is built
inside this file as

``y_lambda = (1-lambda) * S + lambda * P`` and
``q = (1-sigma) * y_lambda + sigma * epsilon``.

For every ``(lambda, sigma)`` cell it directly calls Bernini's official
``renderer.diff_dec.shared_step`` through the already-audited exact81 DCLR
packing helpers.  There is no forward callback, response certificate,
caller-provided energy, replaceable tensor core, paired target, optimizer, or
checkpoint write.  The seven K=2/M=1 branches are T2V action, T2V hard
negatives, MV2V no-op correct/wrong, and MV2V action correct/wrong.  Only the
no-op source swap defines the nuisance projection; action source swaps are an
un-calibrated residual-invariance diagnostic.

The runtime emits an SP4 receipt only after all four ranks report the same
full local-evidence digest.  Passing this smoke never authorizes training or a
scientific/production claim.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import identity_anchored_action_residual as iar_core  # noqa: E402
import infer_dclr_reward_runtime_smoke as dclr  # noqa: E402


METHOD_NAME = "bernini-iar-official-runtime-smoke-v1"
RECEIPT_SCHEMA = "bernini-iar-official-runtime-smoke-receipt-v1"
HARD_NEGATIVE_MANIFEST_SCHEMA = "bernini-iar-hard-negative-manifest-v1"
CANONICAL_MESSAGE_SCHEMA_VERSION = "bernini-iar-canonical-message-v1"
ENERGY_SEMANTICS = (
    "runtime_fp32_mse_to_epsilon_minus_bridge_clean_lower_is_harder"
)
HARD_NEGATIVE_DECLARED_USE = "engineering_plumbing_only"
FORWARD_IMPLEMENTATION = dclr.FORWARD_IMPLEMENTATION
NUM_FRAMES = dclr.NUM_FRAMES
LATENT_PHASES = dclr.LATENT_PHASES
PATCH_SIZE = dclr.PATCH_SIZE
DEFAULT_SIGMAS = (0.80, 0.60, 0.35, 0.15)
DEFAULT_BRIDGE_FRACTIONS = (1.0, 0.5, 0.0)
MIN_HARD_NEGATIVES = 2
MIN_WRONG_SOURCES = 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class IAROfficialRuntimeSmokeError(RuntimeError):
    """Raised before an ambiguous official field receipt can be emitted."""


@dataclass(frozen=True)
class HardNegativeCondition:
    """One content-bound instruction from the manifest, with no energy."""

    condition_id: str
    instruction: str
    instruction_sha256: str


@dataclass(frozen=True)
class OfficialCellResult:
    """Detached official fields, core result, and JSON-ready cell evidence."""

    branch_names: tuple[str, ...]
    frozen_fields: Any
    teacher_result: Any
    record: Mapping[str, Any]


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise IAROfficialRuntimeSmokeError(
            f"value is not canonical ASCII JSON: {error}"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise IAROfficialRuntimeSmokeError(
            f"{label} must be one lowercase SHA-256"
        )
    return value


def _require_sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise IAROfficialRuntimeSmokeError(f"{label} must be one full SHA-1")
    return value.lower()


def _instruction_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_instruction(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
    ):
        raise IAROfficialRuntimeSmokeError(
            f"{label} must be nonempty trimmed text without NUL"
        )
    return value


def build_canonical_renderer_sample(
    instruction: str,
    *,
    expected_instruction_sha256: str,
) -> dict[str, str]:
    """Build the only message shape allowed to reach Bernini's tokenizer.

    Candidate parquet rows are deliberately not accepted here.  The mapping
    contains only canonical JSON ``inputs`` constructed from one independently
    hash-bound instruction.
    """

    text = _require_instruction(instruction, label="canonical instruction")
    expected_sha = _require_sha256(
        expected_instruction_sha256,
        label="canonical instruction SHA-256",
    )
    if _instruction_sha256(text) != expected_sha:
        raise IAROfficialRuntimeSmokeError(
            "canonical instruction SHA-256 differs"
        )
    messages = [
        {"type": "video", "has_loss": 0},
        {"type": "text", "text": text, "has_loss": 0},
        {"type": "video_gen", "has_loss": 1},
    ]
    return {"inputs": _canonical_json_bytes(messages).decode("ascii")}


def _canonical_sample_identity(
    sample: Mapping[str, Any],
    *,
    expected_instruction_sha256: str,
) -> dict[str, str]:
    """Validate one internally built sample and return content evidence."""

    if not isinstance(sample, Mapping) or set(sample) != {"inputs"}:
        raise IAROfficialRuntimeSmokeError(
            "canonical renderer sample must contain only inputs"
        )
    inputs = sample.get("inputs")
    if not isinstance(inputs, str):
        raise IAROfficialRuntimeSmokeError(
            "canonical renderer sample inputs must be text"
        )
    try:
        messages = json.loads(inputs)
    except json.JSONDecodeError as error:
        raise IAROfficialRuntimeSmokeError(
            f"canonical renderer sample JSON is invalid: {error}"
        ) from error
    if (
        not isinstance(messages, list)
        or len(messages) != 3
        or not isinstance(messages[1], Mapping)
    ):
        raise IAROfficialRuntimeSmokeError(
            "canonical renderer sample message geometry differs"
        )
    instruction = messages[1].get("text")
    expected = build_canonical_renderer_sample(
        instruction,
        expected_instruction_sha256=expected_instruction_sha256,
    )
    if dict(sample) != expected:
        raise IAROfficialRuntimeSmokeError(
            "renderer sample differs from canonical message encoding"
        )
    return {
        "instruction_sha256": expected_instruction_sha256,
        "canonical_inputs_sha256": hashlib.sha256(
            inputs.encode("ascii")
        ).hexdigest(),
    }


def _tokenize_canonical_condition(
    *,
    renderer: Any,
    tokenizer: Any,
    encode_renderer_messages: Any,
    sample: Mapping[str, Any],
    expected_instruction_sha256: str,
    task_name: str,
    device: Any,
) -> Any:
    """Use official tokenization behind a fail-closed exact-message guard."""

    identity = _canonical_sample_identity(
        sample,
        expected_instruction_sha256=expected_instruction_sha256,
    )
    expected_inputs = str(sample["inputs"])
    calls = 0

    def guarded_encode(messages: Any, tokenizer_arg: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls != 1:
            raise IAROfficialRuntimeSmokeError(
                "official encoder must consume one canonical sample exactly once"
            )
        try:
            actual_inputs = _canonical_json_bytes(messages).decode("ascii")
        except IAROfficialRuntimeSmokeError:
            raise
        if actual_inputs != expected_inputs:
            raise IAROfficialRuntimeSmokeError(
                "official encoder received non-canonical renderer messages"
            )
        return encode_renderer_messages(messages, tokenizer_arg, **kwargs)

    condition = dclr._tokenize_positive_condition(
        renderer=renderer,
        tokenizer=tokenizer,
        encode_renderer_messages=guarded_encode,
        sample=sample,
        task_name=task_name,
        device=device,
    )
    if calls != 1 or condition.instruction_sha256 != identity["instruction_sha256"]:
        raise IAROfficialRuntimeSmokeError(
            "official tokenization did not preserve the canonical instruction"
        )
    return condition


def canonical_message_schema_evidence(
    *,
    action_sample: Mapping[str, Any],
    action_instruction_sha256: str,
    negative_samples: Sequence[Mapping[str, Any]],
    hard_negative_instruction_sha256s: Sequence[str],
    noop_sample: Mapping[str, Any],
    noop_instruction_sha256: str,
) -> dict[str, Any]:
    """Describe and content-bind every canonical sample used by the encoder."""

    negative_shas = tuple(
        _require_sha256(item, label="hard-negative instruction SHA-256")
        for item in hard_negative_instruction_sha256s
    )
    if len(negative_samples) != len(negative_shas):
        raise IAROfficialRuntimeSmokeError(
            "canonical hard-negative sample/hash counts differ"
        )
    action_identity = _canonical_sample_identity(
        action_sample,
        expected_instruction_sha256=action_instruction_sha256,
    )
    negative_identities = [
        _canonical_sample_identity(sample, expected_instruction_sha256=digest)
        for sample, digest in zip(negative_samples, negative_shas)
    ]
    noop_identity = _canonical_sample_identity(
        noop_sample,
        expected_instruction_sha256=noop_instruction_sha256,
    )
    evidence: dict[str, Any] = {
        "schema_version": CANONICAL_MESSAGE_SCHEMA_VERSION,
        "sample_mapping_keys": ["inputs"],
        "ordered_messages": [
            {
                "index": 0,
                "type": "video",
                "has_loss": 0,
                "keys": ["has_loss", "type"],
            },
            {
                "index": 1,
                "type": "text",
                "has_loss": 0,
                "keys": ["has_loss", "text", "type"],
                "text_source": "hash_bound_instruction",
            },
            {
                "index": 2,
                "type": "video_gen",
                "has_loss": 1,
                "keys": ["has_loss", "type"],
            },
        ],
        "canonical_json_ascii_sort_keys": True,
        "constructed_inside_runtime_per_instruction": True,
        "candidate_dataset_row_accessed": False,
        "candidate_dataset_index_consumed": False,
        "candidate_dataset_iid_consumed": False,
        "encode_renderer_messages_exact_input_guard": True,
        "official_encoder_calls_per_task_condition": 1,
        "action_sample": action_identity,
        "hard_negative_samples": negative_identities,
        "noop_sample": noop_identity,
    }
    evidence["schema_evidence_digest"] = _object_sha256(evidence)
    return evidence


def validate_canonical_message_schema_evidence(
    value: Mapping[str, Any],
    *,
    action_instruction_sha256: str,
    hard_negative_instruction_sha256s: Sequence[str],
    noop_instruction_sha256: str,
) -> dict[str, Any]:
    """Fail closed on receipt claims about dataset-independent messages."""

    if not isinstance(value, Mapping):
        raise IAROfficialRuntimeSmokeError(
            "canonical message schema evidence must be an object"
        )
    expected_fields = {
        "schema_version",
        "sample_mapping_keys",
        "ordered_messages",
        "canonical_json_ascii_sort_keys",
        "constructed_inside_runtime_per_instruction",
        "candidate_dataset_row_accessed",
        "candidate_dataset_index_consumed",
        "candidate_dataset_iid_consumed",
        "encode_renderer_messages_exact_input_guard",
        "official_encoder_calls_per_task_condition",
        "action_sample",
        "hard_negative_samples",
        "noop_sample",
        "schema_evidence_digest",
    }
    if set(value) != expected_fields:
        raise IAROfficialRuntimeSmokeError(
            "canonical message schema evidence fields differ"
        )
    unsigned = dict(value)
    declared_digest = unsigned.pop("schema_evidence_digest")
    if _object_sha256(unsigned) != _require_sha256(
        declared_digest, label="canonical message schema evidence digest"
    ):
        raise IAROfficialRuntimeSmokeError(
            "canonical message schema evidence digest differs"
        )
    expected_layout = [
        {
            "index": 0,
            "type": "video",
            "has_loss": 0,
            "keys": ["has_loss", "type"],
        },
        {
            "index": 1,
            "type": "text",
            "has_loss": 0,
            "keys": ["has_loss", "text", "type"],
            "text_source": "hash_bound_instruction",
        },
        {
            "index": 2,
            "type": "video_gen",
            "has_loss": 1,
            "keys": ["has_loss", "type"],
        },
    ]
    if (
        value.get("schema_version") != CANONICAL_MESSAGE_SCHEMA_VERSION
        or value.get("sample_mapping_keys") != ["inputs"]
        or value.get("ordered_messages") != expected_layout
        or value.get("canonical_json_ascii_sort_keys") is not True
        or value.get("constructed_inside_runtime_per_instruction") is not True
        or value.get("candidate_dataset_row_accessed") is not False
        or value.get("candidate_dataset_index_consumed") is not False
        or value.get("candidate_dataset_iid_consumed") is not False
        or value.get("encode_renderer_messages_exact_input_guard") is not True
        or value.get("official_encoder_calls_per_task_condition") != 1
    ):
        raise IAROfficialRuntimeSmokeError(
            "canonical message construction/encoder evidence differs"
        )

    action_sha = _require_sha256(
        action_instruction_sha256, label="canonical action instruction SHA-256"
    )
    negative_shas = [
        _require_sha256(item, label="canonical hard-negative instruction SHA-256")
        for item in hard_negative_instruction_sha256s
    ]
    noop_sha = _require_sha256(
        noop_instruction_sha256, label="canonical no-op instruction SHA-256"
    )

    def validate_identity(identity: Any, expected_sha: str, *, label: str) -> None:
        if not isinstance(identity, Mapping) or set(identity) != {
            "instruction_sha256",
            "canonical_inputs_sha256",
        }:
            raise IAROfficialRuntimeSmokeError(
                f"{label} canonical sample identity differs"
            )
        if identity.get("instruction_sha256") != expected_sha:
            raise IAROfficialRuntimeSmokeError(
                f"{label} canonical instruction digest differs"
            )
        _require_sha256(
            identity.get("canonical_inputs_sha256"),
            label=f"{label} canonical inputs SHA-256",
        )

    validate_identity(value.get("action_sample"), action_sha, label="action")
    negative_samples = value.get("hard_negative_samples")
    if not isinstance(negative_samples, list) or len(negative_samples) != len(
        negative_shas
    ):
        raise IAROfficialRuntimeSmokeError(
            "canonical hard-negative sample count differs"
        )
    for index, (identity, expected_sha) in enumerate(
        zip(negative_samples, negative_shas)
    ):
        validate_identity(
            identity,
            expected_sha,
            label=f"hard-negative[{index}]",
        )
    validate_identity(value.get("noop_sample"), noop_sha, label="no-op")
    return dict(value)


def validate_bridge_fractions(values: Sequence[Any]) -> tuple[float, ...]:
    """Canonicalize FP32 lambda values while preserving traversal direction.

    HAT v2 traverses proposal to source by default (1 -> .5 -> 0).  Reverse
    traversal is accepted for an explicit ablation, but arbitrary order is
    rejected.  Both clean endpoints are mandatory and the receipt preserves
    the supplied order.
    """

    if isinstance(values, (str, bytes)) or len(values) < 2:
        raise IAROfficialRuntimeSmokeError(
            "bridge fractions require at least two values"
        )
    import torch

    canonical: list[float] = []
    bits: list[str] = []
    for raw in values:
        if isinstance(raw, bool):
            raise IAROfficialRuntimeSmokeError(
                "bridge fractions must be finite values in [0,1]"
            )
        try:
            tensor = torch.tensor([float(raw)], dtype=torch.float32)
        except (TypeError, ValueError, OverflowError) as error:
            raise IAROfficialRuntimeSmokeError(
                "bridge fractions must be finite values in [0,1]"
            ) from error
        value = float(tensor.item())
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise IAROfficialRuntimeSmokeError(
                "bridge fractions must be finite values in [0,1]"
            )
        canonical.append(value)
        bits.append(struct.pack("!f", value).hex())
    if len(set(bits)) != len(bits):
        raise IAROfficialRuntimeSmokeError("bridge fractions must be distinct")
    deltas = [right - left for left, right in zip(canonical, canonical[1:])]
    if not (all(delta > 0.0 for delta in deltas) or all(delta < 0.0 for delta in deltas)):
        raise IAROfficialRuntimeSmokeError(
            "bridge fractions must be strictly monotone"
        )
    if {canonical[0], canonical[-1]} != {0.0, 1.0}:
        raise IAROfficialRuntimeSmokeError(
            "bridge traversal must include source lambda=0 and proposal lambda=1 endpoints"
        )
    return tuple(canonical)


def validate_hard_negative_manifest(
    value: Mapping[str, Any],
    *,
    action_instruction_sha256: str,
    noop_instruction_sha256: str,
) -> dict[str, Any]:
    """Validate ordered hard-negative text only; energies are runtime-owned."""

    if not isinstance(value, Mapping):
        raise IAROfficialRuntimeSmokeError(
            "hard-negative manifest must be an object"
        )
    expected_fields = {
        "schema_version",
        "action_instruction_sha256",
        "hard_negatives",
        "energy_semantics",
        "declared_use",
    }
    if set(value) != expected_fields:
        raise IAROfficialRuntimeSmokeError(
            "hard-negative manifest fields differ; external energies are forbidden"
        )
    if value.get("schema_version") != HARD_NEGATIVE_MANIFEST_SCHEMA:
        raise IAROfficialRuntimeSmokeError("hard-negative manifest schema differs")
    action_sha = _require_sha256(
        action_instruction_sha256, label="action instruction SHA-256"
    )
    noop_sha = _require_sha256(
        noop_instruction_sha256, label="no-op instruction SHA-256"
    )
    if value.get("action_instruction_sha256") != action_sha:
        raise IAROfficialRuntimeSmokeError(
            "hard-negative manifest action instruction digest differs"
        )
    if value.get("energy_semantics") != ENERGY_SEMANTICS:
        raise IAROfficialRuntimeSmokeError(
            "hard-negative manifest energy semantics differ"
        )
    if value.get("declared_use") != HARD_NEGATIVE_DECLARED_USE:
        raise IAROfficialRuntimeSmokeError(
            "hard-negative manifest declared use differs"
        )
    raw_items = value.get("hard_negatives")
    if not isinstance(raw_items, list) or len(raw_items) < MIN_HARD_NEGATIVES:
        raise IAROfficialRuntimeSmokeError(
            "hard-negative manifest requires at least two ordered conditions"
        )
    conditions: list[HardNegativeCondition] = []
    ids: set[str] = set()
    texts: set[str] = set()
    digests: set[str] = set()
    for index, item in enumerate(raw_items):
        if not isinstance(item, Mapping) or set(item) != {
            "condition_id",
            "instruction",
            "instruction_sha256",
        }:
            raise IAROfficialRuntimeSmokeError(
                f"hard-negative item {index} fields differ; energies are forbidden"
            )
        condition_id = _require_instruction(
            item.get("condition_id"), label=f"hard-negative[{index}] condition_id"
        )
        instruction = _require_instruction(
            item.get("instruction"), label=f"hard-negative[{index}] instruction"
        )
        declared_sha = _require_sha256(
            item.get("instruction_sha256"),
            label=f"hard-negative[{index}] instruction SHA-256",
        )
        if _instruction_sha256(instruction) != declared_sha:
            raise IAROfficialRuntimeSmokeError(
                f"hard-negative[{index}] instruction SHA-256 differs"
            )
        if declared_sha == action_sha or instruction in texts:
            raise IAROfficialRuntimeSmokeError(
                "action and hard-negative instructions must be pairwise distinct"
            )
        if condition_id in ids or declared_sha in digests:
            raise IAROfficialRuntimeSmokeError(
                "hard-negative IDs and instruction digests must be unique"
            )
        ids.add(condition_id)
        texts.add(instruction)
        digests.add(declared_sha)
        conditions.append(
            HardNegativeCondition(condition_id, instruction, declared_sha)
        )
    if noop_sha not in digests:
        raise IAROfficialRuntimeSmokeError(
            "the semantic no-op must occur in the hard-negative manifest"
        )
    return {
        "schema_version": HARD_NEGATIVE_MANIFEST_SCHEMA,
        "action_instruction_sha256": action_sha,
        "hard_negatives": [asdict(item) for item in conditions],
        "hard_negative_count": len(conditions),
        "noop_instruction_sha256": noop_sha,
        "noop_present": True,
        "energy_semantics": ENERGY_SEMANTICS,
        "energies_supplied_externally": False,
        "declared_use": HARD_NEGATIVE_DECLARED_USE,
        "manifest_digest": _object_sha256(dict(value)),
    }


def load_hard_negative_manifest(
    path_value: str | Path,
    *,
    expected_sha256: str,
    action_instruction_sha256: str,
    noop_instruction_sha256: str,
) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise IAROfficialRuntimeSmokeError(
            "hard-negative manifest must be absolute"
        )
    try:
        path = path.resolve(strict=True)
    except OSError as error:
        raise IAROfficialRuntimeSmokeError(
            f"hard-negative manifest is unavailable: {error}"
        ) from error
    if not path.is_file() or path.is_symlink():
        raise IAROfficialRuntimeSmokeError(
            "hard-negative manifest must be one plain file"
        )
    actual_sha = _file_sha256(path)
    if actual_sha != _require_sha256(
        expected_sha256, label="hard-negative manifest SHA-256"
    ):
        raise IAROfficialRuntimeSmokeError(
            "hard-negative manifest file SHA-256 differs"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IAROfficialRuntimeSmokeError(
            f"cannot decode hard-negative manifest: {error}"
        ) from error
    result = validate_hard_negative_manifest(
        value,
        action_instruction_sha256=action_instruction_sha256,
        noop_instruction_sha256=noop_instruction_sha256,
    )
    result.update({"path": str(path), "file_sha256": actual_sha})
    return result


def construct_bridge_clean(
    source_clean: Any, proposal_clean: Any, bridge_fraction: Any
) -> Any:
    """Construct one FP32 clean homotopy coordinate from fixed P and S."""

    import torch

    for label, tensor in (
        ("source clean", source_clean),
        ("proposal clean", proposal_clean),
    ):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or tensor.ndim != 5
            or tuple(int(item) for item in tensor.shape[:3]) != (1, 16, LATENT_PHASES)
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise IAROfficialRuntimeSmokeError(
                f"{label} must be detached exact81 FP32 [1,16,21,H,W]"
            )
    if tuple(source_clean.shape) != tuple(proposal_clean.shape) or source_clean.device != proposal_clean.device:
        raise IAROfficialRuntimeSmokeError(
            "source and proposal clean latents must share shape/device"
        )
    if isinstance(bridge_fraction, bool):
        raise IAROfficialRuntimeSmokeError(
            "bridge fraction must be one finite value in [0,1]"
        )
    try:
        value = float(
            torch.tensor([float(bridge_fraction)], dtype=torch.float32).item()
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise IAROfficialRuntimeSmokeError(
            "bridge fraction must be one finite value in [0,1]"
        ) from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise IAROfficialRuntimeSmokeError(
            "bridge fraction must be one finite value in [0,1]"
        )
    if value == 0.0:
        return source_clean
    if value == 1.0:
        return proposal_clean
    result = ((1.0 - value) * source_clean + value * proposal_clean).contiguous()
    if result.dtype != torch.float32 or not bool(torch.isfinite(result).all().item()):
        raise IAROfficialRuntimeSmokeError("bridge clean latent is not finite FP32")
    return result


def hard_negative_energies(
    negative_velocities: Any, true_velocity: Any
) -> Any:
    """Compute real FP32 MSE energy [B,K] from official negative fields."""

    import torch

    if (
        not isinstance(negative_velocities, torch.Tensor)
        or not isinstance(true_velocity, torch.Tensor)
        or negative_velocities.ndim != true_velocity.ndim + 1
        or tuple(negative_velocities.shape[:1]) != tuple(true_velocity.shape[:1])
        or tuple(negative_velocities.shape[2:]) != tuple(true_velocity.shape[1:])
        or int(negative_velocities.shape[1]) < MIN_HARD_NEGATIVES
        or negative_velocities.device != true_velocity.device
        or not negative_velocities.is_floating_point()
        or not true_velocity.is_floating_point()
        or not bool(torch.isfinite(negative_velocities).all().item())
        or not bool(torch.isfinite(true_velocity).all().item())
    ):
        raise IAROfficialRuntimeSmokeError(
            "negative fields/true velocity must be finite matching [B,K,...]/[B,...]"
        )
    negative_fp32 = negative_velocities.detach().to(dtype=torch.float32)
    target_fp32 = true_velocity.detach().to(dtype=torch.float32).unsqueeze(1)
    dimensions = tuple(range(2, negative_fp32.ndim))
    energies = (negative_fp32 - target_fp32).square().mean(dim=dimensions)
    if (
        energies.dtype != torch.float32
        or energies.requires_grad
        or energies.grad_fn is not None
        or not bool(torch.isfinite(energies).all().item())
    ):
        raise IAROfficialRuntimeSmokeError(
            "runtime hard-negative energies must be detached finite FP32"
        )
    return energies


def _tensor_equal(left: Any, right: Any) -> bool:
    import torch

    return bool(
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and tuple(left.shape) == tuple(right.shape)
        and left.dtype == right.dtype
        and left.device == right.device
        and left.layout == right.layout
        and torch.equal(left, right)
    )


def _tensor_identity(value: Any, *, label: str) -> dict[str, Any]:
    """Use the DCLR identity format, with a torch<1.13 byte fallback."""

    import torch

    tensor = dclr._require_floating_tensor(value, label=label, ndim=value.ndim)
    cpu = tensor.detach().to(device="cpu").contiguous().clone()
    untyped = getattr(cpu, "untyped_storage", None)
    if callable(untyped):
        raw = bytes(untyped())
    else:  # tcg uses a pre-untyped-storage PyTorch; official ROCm is newer.
        byte_count = int(cpu.numel() * cpu.element_size())
        pointer = int(cpu.data_ptr())
        if byte_count > 0 and pointer == 0:
            raise IAROfficialRuntimeSmokeError(f"{label} has a null data pointer")
        raw = ctypes.string_at(pointer, byte_count)
    if len(raw) != int(cpu.numel() * cpu.element_size()):
        raise IAROfficialRuntimeSmokeError(f"{label} storage byte count differs")
    metadata = {
        "label": label,
        "shape": [int(item) for item in cpu.shape],
        "dtype": str(cpu.dtype),
        "numel": int(cpu.numel()),
        "byte_count": len(raw),
    }
    payload = _canonical_json_bytes(metadata) + b"\0" + raw
    metadata["content_sha256"] = hashlib.sha256(payload).hexdigest()
    metadata["raw_storage_sha256"] = hashlib.sha256(raw).hexdigest()
    metadata["finite"] = bool(torch.isfinite(cpu).all().item())
    if metadata["finite"] is not True:
        raise IAROfficialRuntimeSmokeError(f"{label} contains NaN or infinity")
    return metadata


def _assert_close(left: Any, right: Any, *, label: str) -> None:
    import torch

    if not _tensor_equal(left, right) and not torch.allclose(
        left, right, rtol=2.0e-5, atol=2.0e-6
    ):
        raise IAROfficialRuntimeSmokeError(
            f"independent recomputation differs for {label}"
        )


def _tensor_rms(value: Any) -> Any:
    return value.float().square().mean(dim=tuple(range(1, value.ndim))).sqrt()


def _cosine_and_relative_rms(left: Any, right: Any) -> dict[str, Any]:
    import torch

    left_flat = left.detach().float().reshape(int(left.shape[0]), -1)
    right_flat = right.detach().float().reshape(int(right.shape[0]), -1)
    left_norm = torch.linalg.vector_norm(left_flat, dim=1)
    right_norm = torch.linalg.vector_norm(right_flat, dim=1)
    defined = (left_norm > 0.0) & (right_norm > 0.0)
    cosine = torch.where(
        defined,
        (left_flat * right_flat).sum(dim=1)
        / (left_norm * right_norm).clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(left_norm),
    )
    left_rms = _tensor_rms(left)
    right_rms = _tensor_rms(right)
    relative_jump = (right_rms - left_rms).abs() / torch.maximum(
        torch.maximum(left_rms, right_rms),
        torch.full_like(left_rms, torch.finfo(torch.float32).tiny),
    )
    return {
        "cosine_defined": [bool(item) for item in defined.cpu().tolist()],
        "cosine": [float(item) for item in cosine.cpu().tolist()],
        "left_rms": [float(item) for item in left_rms.cpu().tolist()],
        "right_rms": [float(item) for item in right_rms.cpu().tolist()],
        "relative_rms_jump": [float(item) for item in relative_jump.cpu().tolist()],
    }


def _branch_names(hard_count: int, wrong_count: int) -> tuple[str, ...]:
    try:
        names = iar_core.expected_frozen_branch_names(hard_count, wrong_count)
    except Exception as error:
        raise IAROfficialRuntimeSmokeError(str(error)) from error
    expected = (
        "frozen_t2v_action",
        *(f"frozen_t2v_hard_negative[{index}]" for index in range(hard_count)),
        "frozen_identity_noop_correct",
        *(f"frozen_identity_noop_wrong_source[{index}]" for index in range(wrong_count)),
        "frozen_identity_action_correct",
        *(f"frozen_identity_action_wrong_source[{index}]" for index in range(wrong_count)),
    )
    if tuple(names) != expected:
        raise IAROfficialRuntimeSmokeError(
            "IAR core branch contract differs from HAT v2 no-op nuisance semantics"
        )
    return expected


def _semantic_binding(
    *,
    branch_names: tuple[str, ...],
    action_instruction_sha256: str,
    hard_negative_instruction_sha256s: Sequence[str],
    noop_instruction_sha256: str,
    correct_source_sha256: str,
    wrong_source_sha256s: Sequence[str],
) -> Any:
    action_sha = _require_sha256(
        action_instruction_sha256, label="action instruction SHA-256"
    )
    negative_shas = tuple(
        _require_sha256(item, label="hard-negative instruction SHA-256")
        for item in hard_negative_instruction_sha256s
    )
    noop_sha = _require_sha256(
        noop_instruction_sha256, label="no-op instruction SHA-256"
    )
    correct_sha = _require_sha256(
        correct_source_sha256, label="correct source SHA-256"
    )
    wrong_shas = tuple(
        _require_sha256(item, label="wrong source SHA-256")
        for item in wrong_source_sha256s
    )
    if len(negative_shas) < MIN_HARD_NEGATIVES or len(wrong_shas) < MIN_WRONG_SOURCES:
        raise IAROfficialRuntimeSmokeError("K>=2 and M>=1 are required")
    semantics: list[Any] = [
        iar_core.BranchSemantic(branch_names[0], "t2v", action_sha, None)
    ]
    cursor = 1
    for digest in negative_shas:
        semantics.append(
            iar_core.BranchSemantic(branch_names[cursor], "t2v", digest, None)
        )
        cursor += 1
    semantics.append(
        iar_core.BranchSemantic(
            branch_names[cursor], "mv2v", noop_sha, correct_sha
        )
    )
    cursor += 1
    for source_sha in wrong_shas:
        semantics.append(
            iar_core.BranchSemantic(
                branch_names[cursor], "mv2v", noop_sha, source_sha
            )
        )
        cursor += 1
    semantics.append(
        iar_core.BranchSemantic(
            branch_names[cursor], "mv2v", action_sha, correct_sha
        )
    )
    cursor += 1
    for source_sha in wrong_shas:
        semantics.append(
            iar_core.BranchSemantic(
                branch_names[cursor], "mv2v", action_sha, source_sha
            )
        )
        cursor += 1
    if cursor != len(branch_names):
        raise IAROfficialRuntimeSmokeError("semantic branch count differs")
    try:
        return iar_core.bind_branch_semantics(tuple(semantics))
    except Exception as error:
        raise IAROfficialRuntimeSmokeError(str(error)) from error


def _validate_bundle_family(bundles: Sequence[Any]) -> dict[str, Any]:
    if isinstance(bundles, (str, bytes)) or len(bundles) < MIN_WRONG_SOURCES:
        raise IAROfficialRuntimeSmokeError(
            "at least one wrong-source DCLR query bundle is required"
        )
    geometry: Optional[dict[str, Any]] = None
    first = bundles[0]
    wrong_digests: set[str] = set()
    for index, bundle in enumerate(bundles):
        try:
            current = dclr.validate_query_bundle(bundle)
        except Exception as error:
            raise IAROfficialRuntimeSmokeError(str(error)) from error
        if geometry is None:
            geometry = current
        elif geometry != current:
            raise IAROfficialRuntimeSmokeError(
                "DCLR query geometry changed across wrong-source interventions"
            )
        for label in (
            "true_velocity_packed",
            "noisy_target_spatial",
            "correct_source_spatial",
            "student_clean_spatial",
            "epsilon_spatial",
            "t2v_noisy_latents",
            "t2v_rotary_embs",
            "mv2v_correct_noisy_latents",
            "mv2v_correct_rotary_embs",
        ):
            if not _tensor_equal(getattr(first, label), getattr(bundle, label)):
                raise IAROfficialRuntimeSmokeError(
                    f"bundle {index} changed shared {label}"
                )
        if bundle.point != first.point or bundle.target_tokens != first.target_tokens:
            raise IAROfficialRuntimeSmokeError(
                "bundle family changed sigma/timestep or target token count"
            )
        digest = _tensor_identity(
            bundle.wrong_source_spatial,
            label=f"wrong_source_spatial_{index}",
        )["content_sha256"]
        if digest in wrong_digests or _tensor_equal(
            bundle.wrong_source_spatial, first.correct_source_spatial
        ):
            raise IAROfficialRuntimeSmokeError(
                "wrong-source latents must be unique and differ from correct source"
            )
        wrong_digests.add(digest)
    assert geometry is not None
    return geometry


def _condition_contract(
    *,
    action_t2v: Any,
    negatives_t2v: Sequence[Any],
    noop_mv2v: Any,
    action_mv2v: Any,
) -> None:
    if not isinstance(action_t2v, dclr.TextCondition) or action_t2v.task_name != "t2v":
        raise IAROfficialRuntimeSmokeError("action T2V condition is invalid")
    if (
        isinstance(negatives_t2v, (str, bytes))
        or len(negatives_t2v) < MIN_HARD_NEGATIVES
        or any(
            not isinstance(item, dclr.TextCondition) or item.task_name != "t2v"
            for item in negatives_t2v
        )
    ):
        raise IAROfficialRuntimeSmokeError(
            "at least two official T2V negative conditions are required"
        )
    if not isinstance(noop_mv2v, dclr.TextCondition) or noop_mv2v.task_name != "mv2v":
        raise IAROfficialRuntimeSmokeError("no-op MV2V condition is invalid")
    if not isinstance(action_mv2v, dclr.TextCondition) or action_mv2v.task_name != "mv2v":
        raise IAROfficialRuntimeSmokeError("action MV2V condition is invalid")
    negative_shas = tuple(item.instruction_sha256 for item in negatives_t2v)
    if (
        action_t2v.instruction_sha256 != action_mv2v.instruction_sha256
        or action_t2v.instruction_sha256 in negative_shas
        or len(set(negative_shas)) != len(negative_shas)
        or noop_mv2v.instruction_sha256 not in negative_shas
    ):
        raise IAROfficialRuntimeSmokeError(
            "official text conditions violate action/negative/no-op semantics"
        )


def _direct_target_prediction(
    renderer: Any,
    *,
    model_id: str,
    noisy_latents: Any,
    rotary_embs: Any,
    target_tokens: int,
    target_mask: Any,
    timestep: Any,
    condition: Any,
) -> Any:
    """One non-replaceable source-level call to the audited official helper."""

    try:
        return dclr.shared_step_target_prediction(
            renderer,
            model_id=model_id,
            noisy_latents=noisy_latents,
            rotary_embs=rotary_embs,
            target_tokens=target_tokens,
            target_mask=target_mask,
            timestep=timestep,
            condition=condition,
        )
    except Exception as error:
        raise IAROfficialRuntimeSmokeError(str(error)) from error


def _independent_core_recompute(
    fields: Any,
    teacher_result: Any,
    config: Any,
    *,
    true_velocity: Any,
) -> dict[str, Any]:
    """Recompute the core geometry from raw official fields, independently."""

    import torch

    action = fields.frozen_t2v_action.detach().float()
    negatives = fields.frozen_t2v_hard_negatives.detach().float()
    noop_correct = fields.frozen_identity_noop_correct.detach().float()
    noop_wrong = fields.frozen_identity_noop_wrong_sources.detach().float()
    action_correct = fields.frozen_identity_action_correct.detach().float()
    action_wrong = fields.frozen_identity_action_wrong_sources.detach().float()
    batch, hard_count = int(negatives.shape[0]), int(negatives.shape[1])
    wrong_count = int(noop_wrong.shape[1])
    feature_shape = tuple(int(item) for item in action.shape[1:])
    if (
        not isinstance(true_velocity, torch.Tensor)
        or tuple(true_velocity.shape) != tuple(action.shape)
        or true_velocity.device != action.device
        or not true_velocity.is_floating_point()
        or not bool(torch.isfinite(true_velocity).all().item())
    ):
        raise IAROfficialRuntimeSmokeError(
            "independent RF energy target must match official action field"
        )
    true_velocity_fp32 = true_velocity.detach().float()
    energy_dimensions = tuple(range(1, action.ndim))
    action_energy = (action - true_velocity_fp32).square().mean(
        dim=energy_dimensions
    )
    negative_energy = (
        negatives - true_velocity_fp32[:, None]
    ).square().mean(dim=tuple(range(2, negatives.ndim)))
    ordering_margins = negative_energy - action_energy[:, None]
    _assert_close(
        negative_energy,
        fields.hard_negative_energies,
        label="runtime_negative_RF_energy",
    )

    weights = torch.softmax(
        -fields.hard_negative_energies.detach().float()
        / float(config.hard_negative_temperature),
        dim=1,
    )
    barycenter = (
        negatives
        * weights.reshape(batch, hard_count, *([1] * len(feature_shape)))
    ).sum(dim=1)
    raw = action - barycenter
    tangents = noop_correct[:, None] - noop_wrong
    correct_action_residual = action_correct - noop_correct
    wrong_action_residual = action_wrong - noop_wrong

    tangent_flat = tangents.reshape(batch, wrong_count, -1)
    raw_flat = raw.reshape(batch, -1, 1)
    tiny = torch.finfo(torch.float32).tiny
    max_abs = tangent_flat.abs().amax(dim=2)
    nonzero = max_abs > 0.0
    scaled = torch.where(
        nonzero[:, :, None],
        tangent_flat / max_abs.clamp_min(tiny)[:, :, None],
        torch.zeros_like(tangent_flat),
    )
    normalized = torch.where(
        nonzero[:, :, None],
        scaled
        / torch.linalg.vector_norm(scaled, dim=2).clamp_min(tiny)[:, :, None],
        torch.zeros_like(scaled),
    )
    gram = torch.bmm(normalized, normalized.transpose(1, 2))
    pinv = torch.linalg.pinv(
        gram,
        rtol=float(config.projection_rank_rtol),
        hermitian=True,
    )
    coefficients = torch.bmm(pinv, torch.bmm(normalized, raw_flat))
    projected_flat = raw_flat - torch.bmm(
        normalized.transpose(1, 2), coefficients
    )
    projected = projected_flat.reshape_as(raw)

    feature_count = int(math.prod(feature_shape))
    source_tangent_rms = max_abs * scaled.square().mean(dim=2).sqrt()
    tangent_norms = source_tangent_rms * math.sqrt(float(feature_count))
    robust_source_tangent_rms = source_tangent_rms.median(dim=1).values
    raw_rms = _tensor_rms(raw)
    projected_rms = _tensor_rms(projected)
    sigma = fields.sigma.detach().float()
    # Deliberately reproduce the documented piecewise schedule here instead
    # of calling iar_core.sigma_action_scale: this is an independent audit of
    # both the gate and the gauge-invariant cap.
    action_scale = torch.where(
        sigma >= float(config.high_sigma_min),
        torch.ones_like(sigma),
        torch.where(
            sigma >= float(config.mid_sigma_min),
            torch.full_like(sigma, float(config.mid_sigma_action_scale)),
            torch.zeros_like(sigma),
        ),
    )
    cap_reference_rms = torch.minimum(raw_rms, robust_source_tangent_rms)
    action_rms_cap = (
        float(config.action_rms_cap_ratio)
        * action_scale
        * cap_reference_rms
    )
    cap_scale = torch.minimum(
        torch.ones_like(projected_rms),
        action_rms_cap / projected_rms.clamp_min(tiny),
    )
    cap_scale = torch.where(
        projected_rms > 0.0, cap_scale, torch.ones_like(cap_scale)
    )
    cap_scale = torch.where(
        action_scale > 0.0, cap_scale, torch.zeros_like(cap_scale)
    )
    capped = projected * cap_scale.reshape(
        batch, *([1] * (projected.ndim - 1))
    )
    capped = torch.where(
        (action_scale > 0.0).reshape(
            batch, *([1] * (projected.ndim - 1))
        ),
        capped,
        torch.zeros_like(capped),
    )
    capped_rms = _tensor_rms(capped)

    correct_flat = correct_action_residual.reshape(batch, -1)
    wrong_flat = wrong_action_residual.reshape(batch, wrong_count, -1)
    correct_l2 = torch.linalg.vector_norm(correct_flat, dim=1)
    wrong_l2 = torch.linalg.vector_norm(wrong_flat, dim=2)
    action_dot = torch.bmm(wrong_flat, correct_flat[:, :, None]).squeeze(-1)
    invariance_cosine = torch.where(
        (correct_l2[:, None] * wrong_l2) > 0.0,
        action_dot / (correct_l2[:, None] * wrong_l2).clamp_min(tiny),
        torch.zeros_like(action_dot),
    )
    symmetric_norm_ratio = torch.where(
        (correct_l2[:, None] > 0.0) & (wrong_l2 > 0.0),
        torch.minimum(
            wrong_l2 / correct_l2[:, None].clamp_min(tiny),
            correct_l2[:, None] / wrong_l2.clamp_min(tiny),
        ),
        torch.zeros_like(wrong_l2),
    )

    diagnostics = teacher_result.diagnostics
    comparisons = (
        (weights, diagnostics.softmin_weights, "softmin_weights"),
        (barycenter, diagnostics.hard_negative_barycenter, "hard_negative_barycenter"),
        (raw, diagnostics.raw_action_residual, "raw_action_residual"),
        (tangents, diagnostics.identity_tangents, "noop_source_tangents"),
        (normalized, diagnostics.normalized_identity_tangents, "normalized_noop_tangents"),
        (gram, diagnostics.normalized_tangent_gram_fp32, "normalized_tangent_gram"),
        (pinv, diagnostics.normalized_tangent_gram_pinv_fp32, "tangent_gram_pinv"),
        (projected, diagnostics.projected_action_residual, "projected_action_residual"),
        (tangent_norms, diagnostics.tangent_norms, "tangent_norms"),
        (source_tangent_rms, diagnostics.source_tangent_rms, "source_tangent_rms"),
        (
            robust_source_tangent_rms,
            diagnostics.robust_source_tangent_rms,
            "robust_source_tangent_rms",
        ),
        (raw_rms, diagnostics.raw_action_rms, "raw_action_rms"),
        (projected_rms, diagnostics.projected_action_rms, "projected_action_rms"),
        (
            cap_reference_rms,
            diagnostics.cap_reference_rms,
            "gauge_invariant_cap_reference_rms",
        ),
        (action_scale, diagnostics.sigma_action_scale, "sigma_action_scale"),
        (action_rms_cap, diagnostics.action_rms_cap, "action_rms_cap"),
        (cap_scale, diagnostics.cap_scale, "cap_scale"),
        (capped, diagnostics.capped_action_residual, "capped_action_residual"),
        (capped_rms, diagnostics.capped_action_rms, "capped_action_rms"),
        (capped, teacher_result.teacher_action_residual, "teacher_action_residual"),
        (
            correct_action_residual,
            diagnostics.source_conditioned_action_residual_correct,
            "source_action_residual_correct",
        ),
        (
            wrong_action_residual,
            diagnostics.source_conditioned_action_residual_wrong_sources,
            "source_action_residual_wrong",
        ),
        (
            invariance_cosine,
            diagnostics.source_action_invariance_cosine,
            "source_action_invariance_cosine",
        ),
        (
            symmetric_norm_ratio,
            diagnostics.source_action_invariance_symmetric_norm_ratio,
            "source_action_invariance_symmetric_norm_ratio",
        ),
    )
    for independent, core_value, label in comparisons:
        _assert_close(independent, core_value, label=label)

    low_sigma = action_scale == 0.0
    if bool(low_sigma.any().item()):
        low_teacher = capped[low_sigma]
        low_core_teacher = teacher_result.teacher_action_residual[low_sigma]
        if bool(low_teacher.ne(0.0).any().item()) or bool(
            low_core_teacher.ne(0.0).any().item()
        ):
            raise IAROfficialRuntimeSmokeError(
                "independent low-sigma teacher must be exactly zero"
            )
    if bool((capped_rms > action_rms_cap + 2.0e-6).any().item()):
        raise IAROfficialRuntimeSmokeError(
            "independent capped teacher escaped its gauge-invariant RMS cap"
        )

    raw_l2 = torch.linalg.vector_norm(raw.reshape(batch, -1), dim=1)
    projected_l2 = torch.linalg.vector_norm(projected.reshape(batch, -1), dim=1)
    retention = torch.where(
        raw_l2 > 0.0,
        projected_l2 / raw_l2.clamp_min(tiny),
        torch.zeros_like(raw_l2),
    )
    _assert_close(
        retention, diagnostics.projection_retention, label="projection_retention"
    )
    identities = {
        "weights": _tensor_identity(weights, label="independent_weights"),
        "raw_action": _tensor_identity(raw, label="independent_raw_action"),
        "noop_tangents": _tensor_identity(tangents, label="independent_noop_tangents"),
        "projected_action": _tensor_identity(projected, label="independent_projected_action"),
        "capped_teacher": _tensor_identity(capped, label="independent_capped_teacher"),
        "cap_reference_rms": _tensor_identity(
            cap_reference_rms, label="independent_cap_reference_rms"
        ),
        "cap_scale": _tensor_identity(cap_scale, label="independent_cap_scale"),
        "source_action_residual_correct": _tensor_identity(
            correct_action_residual, label="independent_source_action_residual_correct"
        ),
        "source_action_residual_wrong": _tensor_identity(
            wrong_action_residual, label="independent_source_action_residual_wrong"
        ),
    }
    return {
        "verified": True,
        "implementation_independent_of_core_receipt": True,
        "projection_uses_noop_source_swaps_only": True,
        "action_source_swaps_diagnostic_only": True,
        "gauge_invariant_cap_recomputed": True,
        "sigma_schedule_recomputed_without_core_helper": True,
        "capped_teacher_recomputed": True,
        "low_sigma_exact_zero_verified": True,
        "action_energy_EA": [float(item) for item in action_energy.cpu().tolist()],
        "negative_energies": negative_energy.cpu().tolist(),
        "ordering_margins_Ek_minus_EA": ordering_margins.cpu().tolist(),
        "rf_energy_reduction": "mean_over_all_nonbatch_target_token_and_patch_feature_dimensions",
        "rf_squared_error_proxy_not_likelihood_or_free_energy": True,
        "ordering_is_diagnostic_not_training_authorization": True,
        "source_action_invariance_calibration_authorized": False,
        "tensor_identities": identities,
        "digest": _object_sha256(identities),
    }


def _run_official_cell(
    *,
    renderer: Any,
    model_id: str,
    bundles: Sequence[Any],
    bridge_fraction: float,
    action_t2v_condition: Any,
    hard_negative_t2v_conditions: Sequence[Any],
    noop_mv2v_condition: Any,
    action_mv2v_condition: Any,
    correct_source_sha256: str,
    wrong_source_sha256s: Sequence[str],
    config: Any = None,
) -> OfficialCellResult:
    """Run one internally packed cell through direct official shared_step calls."""

    import torch

    if config is None:
        config = iar_core.IARConfig()
    geometry = _validate_bundle_family(bundles)
    _condition_contract(
        action_t2v=action_t2v_condition,
        negatives_t2v=hard_negative_t2v_conditions,
        noop_mv2v=noop_mv2v_condition,
        action_mv2v=action_mv2v_condition,
    )
    hard_count = len(hard_negative_t2v_conditions)
    wrong_count = len(bundles)
    if len(wrong_source_sha256s) != wrong_count:
        raise IAROfficialRuntimeSmokeError(
            "wrong-source digest count differs from official bundle count"
        )
    branch_names = _branch_names(hard_count, wrong_count)
    first = bundles[0]
    device = first.t2v_noisy_latents.device
    timestep = torch.tensor(
        [first.point.timestep], dtype=torch.float32, device=device
    )

    with torch.inference_mode():
        action = _direct_target_prediction(
            renderer,
            model_id=model_id,
            noisy_latents=first.t2v_noisy_latents,
            rotary_embs=first.t2v_rotary_embs,
            target_tokens=first.target_tokens,
            target_mask=first.t2v_target_mask,
            timestep=timestep,
            condition=action_t2v_condition,
        )
        negatives = tuple(
            _direct_target_prediction(
                renderer,
                model_id=model_id,
                noisy_latents=first.t2v_noisy_latents,
                rotary_embs=first.t2v_rotary_embs,
                target_tokens=first.target_tokens,
                target_mask=first.t2v_target_mask,
                timestep=timestep,
                condition=condition,
            )
            for condition in hard_negative_t2v_conditions
        )
        noop_correct = _direct_target_prediction(
            renderer,
            model_id=model_id,
            noisy_latents=first.mv2v_correct_noisy_latents,
            rotary_embs=first.mv2v_correct_rotary_embs,
            target_tokens=first.target_tokens,
            target_mask=first.mv2v_correct_target_mask,
            timestep=timestep,
            condition=noop_mv2v_condition,
        )
        noop_wrong = tuple(
            _direct_target_prediction(
                renderer,
                model_id=model_id,
                noisy_latents=bundle.mv2v_wrong_noisy_latents,
                rotary_embs=bundle.mv2v_wrong_rotary_embs,
                target_tokens=bundle.target_tokens,
                target_mask=bundle.mv2v_wrong_target_mask,
                timestep=timestep,
                condition=noop_mv2v_condition,
            )
            for bundle in bundles
        )
        action_correct = _direct_target_prediction(
            renderer,
            model_id=model_id,
            noisy_latents=first.mv2v_correct_noisy_latents,
            rotary_embs=first.mv2v_correct_rotary_embs,
            target_tokens=first.target_tokens,
            target_mask=first.mv2v_correct_target_mask,
            timestep=timestep,
            condition=action_mv2v_condition,
        )
        action_wrong = tuple(
            _direct_target_prediction(
                renderer,
                model_id=model_id,
                noisy_latents=bundle.mv2v_wrong_noisy_latents,
                rotary_embs=bundle.mv2v_wrong_rotary_embs,
                target_tokens=bundle.target_tokens,
                target_mask=bundle.mv2v_wrong_target_mask,
                timestep=timestep,
                condition=action_mv2v_condition,
            )
            for bundle in bundles
        )

    all_fields = (action, *negatives, noop_correct, *noop_wrong, action_correct, *action_wrong)
    if len(all_fields) != len(branch_names):
        raise IAROfficialRuntimeSmokeError("official forward count differs from branch contract")
    reference = action
    if any(
        not isinstance(item, torch.Tensor)
        or tuple(item.shape) != tuple(reference.shape)
        or item.dtype != reference.dtype
        or item.device != reference.device
        or item.requires_grad
        or item.grad_fn is not None
        or not bool(torch.isfinite(item).all().item())
        for item in all_fields
    ):
        raise IAROfficialRuntimeSmokeError(
            "official target-tail fields do not share finite detached shape/dtype/device"
        )

    negative_tensor = torch.stack(negatives, dim=1)
    noop_wrong_tensor = torch.stack(noop_wrong, dim=1)
    action_wrong_tensor = torch.stack(action_wrong, dim=1)
    energies = hard_negative_energies(
        negative_tensor, first.true_velocity_packed
    )
    semantics = _semantic_binding(
        branch_names=branch_names,
        action_instruction_sha256=action_t2v_condition.instruction_sha256,
        hard_negative_instruction_sha256s=tuple(
            item.instruction_sha256 for item in hard_negative_t2v_conditions
        ),
        noop_instruction_sha256=noop_mv2v_condition.instruction_sha256,
        correct_source_sha256=correct_source_sha256,
        wrong_source_sha256s=wrong_source_sha256s,
    )
    shared_noisy_state = first.noisy_target_spatial
    try:
        shared_binding = iar_core.bind_shared_state(
            shared_noisy_state,
            {name: shared_noisy_state for name in branch_names},
        )
        fields = iar_core.IARFrozenFields(
            shared_state=shared_binding,
            semantic_binding=semantics,
            sigma=torch.tensor(
                [first.point.sigma], dtype=torch.float32, device=device
            ),
            frozen_t2v_action=action,
            frozen_t2v_hard_negatives=negative_tensor,
            hard_negative_energies=energies,
            frozen_identity_noop_correct=noop_correct,
            frozen_identity_noop_wrong_sources=noop_wrong_tensor,
            frozen_identity_action_correct=action_correct,
            frozen_identity_action_wrong_sources=action_wrong_tensor,
        )
        teacher_result = iar_core.compute_frozen_identity_anchored_teacher(
            fields, config=config
        )
    except Exception as error:
        raise IAROfficialRuntimeSmokeError(str(error)) from error
    independent = _independent_core_recompute(
        fields,
        teacher_result,
        config,
        true_velocity=first.true_velocity_packed,
    )

    field_identities = {
        name: _tensor_identity(field, label=name)
        for name, field in zip(branch_names, all_fields)
    }
    diagnostics = teacher_result.diagnostics
    action_energy = (
        action.detach().float() - first.true_velocity_packed.detach().float()
    ).square().mean(dim=tuple(range(1, action.ndim)))
    ordering_margins = energies - action_energy[:, None]
    independent_action_energy = independent["action_energy_EA"]
    independent_margins = independent["ordering_margins_Ek_minus_EA"]
    if (
        independent_action_energy
        != [float(item) for item in action_energy.cpu().tolist()]
        or independent_margins != ordering_margins.cpu().tolist()
    ):
        raise IAROfficialRuntimeSmokeError(
            "independent action/negative energy ordering differs"
        )
    metrics = {
        "projection_retention": [
            float(item) for item in diagnostics.projection_retention.cpu().tolist()
        ],
        "raw_action_rms": [
            float(item) for item in diagnostics.raw_action_rms.cpu().tolist()
        ],
        "projected_action_rms": [
            float(item) for item in diagnostics.projected_action_rms.cpu().tolist()
        ],
        "capped_action_rms": [
            float(item) for item in diagnostics.capped_action_rms.cpu().tolist()
        ],
        "tangent_rank": [
            int(item) for item in diagnostics.tangent_rank.cpu().tolist()
        ],
        "max_abs_postprojection_tangent_cosine": [
            float(item)
            for item in diagnostics.max_abs_postprojection_tangent_cosine.cpu().tolist()
        ],
        "source_action_invariance_cosine": diagnostics.source_action_invariance_cosine.cpu().tolist(),
        "source_action_invariance_symmetric_norm_ratio": diagnostics.source_action_invariance_symmetric_norm_ratio.cpu().tolist(),
        "projected_action_alignment_correct": diagnostics.projected_action_alignment_correct.cpu().tolist(),
        "projected_action_alignment_wrong_sources": diagnostics.projected_action_alignment_wrong_sources.cpu().tolist(),
        "source_action_invariance_calibration_authorized": False,
        "M_equals_one_plumbing_only": wrong_count == 1,
    }
    record: dict[str, Any] = {
        "bridge_fraction": float(bridge_fraction),
        "bridge_fraction_float32_bits_hex": struct.pack("!f", float(bridge_fraction)).hex(),
        "flow_query": first.point.as_dict(),
        "branch_order": list(branch_names),
        "forward_count": len(branch_names),
        "target_query": _tensor_identity(
            shared_noisy_state,
            label=(
                f"q_lambda_{struct.pack('!f', float(bridge_fraction)).hex()}_"
                f"sigma_{first.point.sigma_float32_bits_hex}"
            ),
        ),
        "true_velocity": _tensor_identity(
            first.true_velocity_packed, label="epsilon_minus_bridge_clean"
        ),
        "geometry": geometry,
        "single_variable_intervention": {
            "t2v_is_direct_correct_mv2v_target_tail": True,
            "wrong_source_changes_prefix_only": True,
            "same_target_tail_and_rope": True,
            "same_sigma_and_timestep": True,
        },
        "hard_negative_energies": [
            float(item) for item in energies[0].cpu().tolist()
        ],
        "action_energy_EA": [
            float(item) for item in action_energy.cpu().tolist()
        ],
        "ordering_margins_Ek_minus_EA": ordering_margins.cpu().tolist(),
        "hard_negative_energy_semantics": ENERGY_SEMANTICS,
        "rf_energy_reduction": "mean_over_all_nonbatch_target_token_and_patch_feature_dimensions",
        "rf_squared_error_proxy_not_likelihood_or_free_energy": True,
        "ordering_is_diagnostic_not_training_authorization": True,
        "energies_computed_inside_runtime_from_actual_fields": True,
        "field_identities": field_identities,
        "field_set_digest": _object_sha256(field_identities),
        "metrics": metrics,
        "independent_recompute": independent,
        "core_receipt_digest": _object_sha256(teacher_result.receipt),
        "teacher_action_residual": _tensor_identity(
            teacher_result.teacher_action_residual,
            label="teacher_action_residual",
        ),
        "no_forward_callback": True,
        "direct_official_shared_step": True,
    }
    record["cell_digest"] = _object_sha256(record)
    return OfficialCellResult(
        branch_names=branch_names,
        frozen_fields=fields,
        teacher_result=teacher_result,
        record=record,
    )


def build_bridge_continuity(
    cell_results: Sequence[OfficialCellResult],
    *,
    bridge_fractions: Sequence[float],
    points: Sequence[Any],
) -> dict[str, Any]:
    """Compare adjacent lambdas at fixed sigma without inventing a pass gate."""

    expected = len(bridge_fractions) * len(points)
    if len(cell_results) != expected:
        raise IAROfficialRuntimeSmokeError(
            "continuity grid does not cover every lambda/sigma cell"
        )
    by_cell: dict[tuple[str, str], OfficialCellResult] = {}
    for result in cell_results:
        record = result.record
        key = (
            str(record["bridge_fraction_float32_bits_hex"]),
            str(record["flow_query"]["sigma_float32_bits_hex"]),
        )
        if key in by_cell:
            raise IAROfficialRuntimeSmokeError("duplicate continuity cell")
        by_cell[key] = result
    rows = []
    for point in points:
        for left_lambda, right_lambda in zip(bridge_fractions, bridge_fractions[1:]):
            left_bits = struct.pack("!f", float(left_lambda)).hex()
            right_bits = struct.pack("!f", float(right_lambda)).hex()
            left = by_cell[(left_bits, point.sigma_float32_bits_hex)]
            right = by_cell[(right_bits, point.sigma_float32_bits_hex)]
            rows.append(
                {
                    "sigma_float32_bits_hex": point.sigma_float32_bits_hex,
                    "left_bridge_fraction": float(left_lambda),
                    "right_bridge_fraction": float(right_lambda),
                    "teacher_residual": _cosine_and_relative_rms(
                        left.teacher_result.teacher_action_residual,
                        right.teacher_result.teacher_action_residual,
                    ),
                    "identity_noop_anchor": _cosine_and_relative_rms(
                        left.frozen_fields.frozen_identity_noop_correct,
                        right.frozen_fields.frozen_identity_noop_correct,
                    ),
                    "raw_action_residual": _cosine_and_relative_rms(
                        left.teacher_result.diagnostics.raw_action_residual,
                        right.teacher_result.diagnostics.raw_action_residual,
                    ),
                    "projection_retention_left": left.record["metrics"]["projection_retention"],
                    "projection_retention_right": right.record["metrics"]["projection_retention"],
                }
            )
    return {
        "grid_complete": True,
        "bridge_order_preserved": [float(item) for item in bridge_fractions],
        "adjacent_comparisons": rows,
        "comparison_count": len(rows),
        "metric_only_no_training_gate": True,
        "source_action_invariance_calibration_authorized": False,
    }


def assemble_sp4_receipt(
    local_evidence: Mapping[str, Any],
    rank_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless exact world4/Ulysses4 evidence is identical."""

    if not isinstance(local_evidence, Mapping):
        raise IAROfficialRuntimeSmokeError("local evidence must be an object")
    if local_evidence.get("method") != METHOD_NAME:
        raise IAROfficialRuntimeSmokeError("local evidence method differs")
    _require_sha256(
        local_evidence.get("launcher_source_sha256"),
        label="launcher source SHA-256",
    )
    if local_evidence.get("forward_implementation") != FORWARD_IMPLEMENTATION:
        raise IAROfficialRuntimeSmokeError(
            "local evidence did not use the direct official shared_step path"
        )
    if (
        local_evidence.get("num_frames") != NUM_FRAMES
        or local_evidence.get("latent_phases") != LATENT_PHASES
        or local_evidence.get("patch_size") != list(PATCH_SIZE)
    ):
        raise IAROfficialRuntimeSmokeError("local evidence is not exact81")
    hard_manifest = local_evidence.get("hard_negative_manifest")
    if not isinstance(hard_manifest, Mapping):
        raise IAROfficialRuntimeSmokeError("hard-negative manifest evidence is absent")
    hard_count = hard_manifest.get("hard_negative_count")
    if type(hard_count) is not int or hard_count < MIN_HARD_NEGATIVES:
        raise IAROfficialRuntimeSmokeError("hard-negative count is invalid")
    if (
        hard_manifest.get("energies_supplied_externally") is not False
        or hard_manifest.get("energy_semantics") != ENERGY_SEMANTICS
        or hard_manifest.get("noop_present") is not True
    ):
        raise IAROfficialRuntimeSmokeError(
            "hard-negative energy/no-op evidence differs"
        )
    hard_items = hard_manifest.get("hard_negatives")
    if not isinstance(hard_items, list) or len(hard_items) != hard_count:
        raise IAROfficialRuntimeSmokeError(
            "hard-negative instruction evidence is incomplete"
        )
    hard_instruction_shas: list[str] = []
    for index, item in enumerate(hard_items):
        if not isinstance(item, Mapping):
            raise IAROfficialRuntimeSmokeError(
                f"hard-negative instruction {index} evidence differs"
            )
        hard_instruction_shas.append(
            _require_sha256(
                item.get("instruction_sha256"),
                label=f"hard-negative[{index}] instruction SHA-256",
            )
        )
    action_instruction_sha = _require_sha256(
        hard_manifest.get("action_instruction_sha256"),
        label="manifest action instruction SHA-256",
    )
    noop_instruction_sha = _require_sha256(
        hard_manifest.get("noop_instruction_sha256"),
        label="manifest no-op instruction SHA-256",
    )
    candidate = local_evidence.get("candidate")
    expected_candidate_fields = {
        "canonical_message_schema",
        "proposal_source_iid",
        "proposal_origin",
        "proposal_artifact",
        "correct_source_artifact",
        "native_provenance",
        "paired_target_accessed",
    }
    if not isinstance(candidate, Mapping) or set(candidate) != expected_candidate_fields:
        raise IAROfficialRuntimeSmokeError(
            "candidate canonical-message/provenance evidence fields differ"
        )
    _require_instruction(
        candidate.get("proposal_source_iid"), label="proposal source IID"
    )
    if (
        candidate.get("proposal_origin") != "native_rollout_predecode_latent"
        or not isinstance(candidate.get("proposal_artifact"), Mapping)
        or not isinstance(candidate.get("correct_source_artifact"), Mapping)
        or not isinstance(candidate.get("native_provenance"), Mapping)
        or candidate.get("paired_target_accessed") is not False
    ):
        raise IAROfficialRuntimeSmokeError(
            "candidate native provenance evidence differs"
        )
    validate_canonical_message_schema_evidence(
        candidate.get("canonical_message_schema"),
        action_instruction_sha256=action_instruction_sha,
        hard_negative_instruction_sha256s=hard_instruction_shas,
        noop_instruction_sha256=noop_instruction_sha,
    )
    wrong_count = local_evidence.get("wrong_source_count")
    if type(wrong_count) is not int or wrong_count < MIN_WRONG_SOURCES:
        raise IAROfficialRuntimeSmokeError("wrong-source count is invalid")
    expected_branches = list(_branch_names(hard_count, wrong_count))
    if local_evidence.get("branch_order") != expected_branches:
        raise IAROfficialRuntimeSmokeError("local branch order differs")

    bridge_values = local_evidence.get("bridge_fractions")
    sigma_records = local_evidence.get("sigmas")
    cells = local_evidence.get("cell_records")
    if not isinstance(bridge_values, list):
        raise IAROfficialRuntimeSmokeError("bridge fractions are absent")
    bridge = validate_bridge_fractions(bridge_values)
    if not isinstance(sigma_records, list) or len(sigma_records) < 2:
        raise IAROfficialRuntimeSmokeError("at least two sigma records are required")
    sigma_bits = []
    for item in sigma_records:
        if not isinstance(item, Mapping):
            raise IAROfficialRuntimeSmokeError("sigma record must be an object")
        point = dclr.flow_query_point(item.get("sigma"))
        if dict(item) != point.as_dict():
            raise IAROfficialRuntimeSmokeError("sigma record differs from RF mapping")
        sigma_bits.append(point.sigma_float32_bits_hex)
    if len(set(sigma_bits)) != len(sigma_bits):
        raise IAROfficialRuntimeSmokeError("sigma records are not distinct")
    if not isinstance(cells, list) or len(cells) != len(bridge) * len(sigma_records):
        raise IAROfficialRuntimeSmokeError("lambda/sigma cell grid is incomplete")
    expected_pairs = {
        (struct.pack("!f", value).hex(), sigma_bit)
        for value in bridge
        for sigma_bit in sigma_bits
    }
    actual_pairs = set()
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise IAROfficialRuntimeSmokeError("cell record must be an object")
        unsigned = dict(cell)
        declared_digest = unsigned.pop("cell_digest", None)
        if _object_sha256(unsigned) != _require_sha256(
            declared_digest, label="cell digest"
        ):
            raise IAROfficialRuntimeSmokeError("cell digest differs")
        if cell.get("branch_order") != expected_branches or cell.get(
            "forward_count"
        ) != len(expected_branches):
            raise IAROfficialRuntimeSmokeError("cell branch order/count differs")
        if (
            cell.get("energies_computed_inside_runtime_from_actual_fields") is not True
            or cell.get("hard_negative_energy_semantics") != ENERGY_SEMANTICS
            or cell.get("rf_energy_reduction")
            != "mean_over_all_nonbatch_target_token_and_patch_feature_dimensions"
            or cell.get("rf_squared_error_proxy_not_likelihood_or_free_energy") is not True
            or cell.get("ordering_is_diagnostic_not_training_authorization") is not True
            or cell.get("no_forward_callback") is not True
            or cell.get("direct_official_shared_step") is not True
        ):
            raise IAROfficialRuntimeSmokeError("cell official-runtime claims differ")
        independent = cell.get("independent_recompute")
        geometry = cell.get("geometry")
        if (
            not isinstance(independent, Mapping)
            or independent.get("verified") is not True
            or independent.get("projection_uses_noop_source_swaps_only") is not True
            or not isinstance(geometry, Mapping)
            or geometry.get("verified") is not True
        ):
            raise IAROfficialRuntimeSmokeError(
                "cell independent/core geometry verification is absent"
            )
        action_energy = cell.get("action_energy_EA")
        negative_energies = cell.get("hard_negative_energies")
        margins = cell.get("ordering_margins_Ek_minus_EA")
        if (
            not isinstance(action_energy, list)
            or len(action_energy) != 1
            or not isinstance(negative_energies, list)
            or len(negative_energies) != hard_count
            or not isinstance(margins, list)
            or len(margins) != 1
            or not isinstance(margins[0], list)
            or len(margins[0]) != hard_count
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or float(item) < 0.0
                for item in (*action_energy, *negative_energies)
            )
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in margins[0]
            )
            or any(
                not math.isclose(
                    float(margin),
                    float(negative) - float(action_energy[0]),
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-7,
                )
                for margin, negative in zip(margins[0], negative_energies)
            )
            or independent.get("action_energy_EA") != action_energy
            or independent.get("negative_energies") != [negative_energies]
            or independent.get("ordering_margins_Ek_minus_EA") != margins
            or independent.get("rf_squared_error_proxy_not_likelihood_or_free_energy") is not True
            or independent.get("ordering_is_diagnostic_not_training_authorization") is not True
        ):
            raise IAROfficialRuntimeSmokeError(
                "cell action/negative RF energy ordering evidence differs"
            )
        flow = cell.get("flow_query")
        if not isinstance(flow, Mapping):
            raise IAROfficialRuntimeSmokeError("cell flow query is absent")
        actual_pairs.add(
            (
                str(cell.get("bridge_fraction_float32_bits_hex")),
                str(flow.get("sigma_float32_bits_hex")),
            )
        )
    if actual_pairs != expected_pairs:
        raise IAROfficialRuntimeSmokeError("cell grid coordinates differ")
    expected_forwards = len(cells) * len(expected_branches)
    if local_evidence.get("forwards_per_rank") != expected_forwards:
        raise IAROfficialRuntimeSmokeError("per-rank official forward count differs")

    homotopy = local_evidence.get("homotopy")
    if (
        not isinstance(homotopy, Mapping)
        or homotopy.get("constructed_inside_runtime") is not True
        or homotopy.get("formula")
        != "q=(1-sigma)*((1-lambda)*S+lambda*P)+sigma*epsilon"
        or homotopy.get("one_proposal_P_for_all_cells") is not True
        or homotopy.get("one_correct_source_S_for_all_cells") is not True
        or homotopy.get("one_epsilon_for_all_cells") is not True
        or homotopy.get("caller_provided_sigma_states") is not False
        or not isinstance(homotopy.get("proposal_P"), Mapping)
        or not isinstance(homotopy.get("correct_source_S"), Mapping)
        or not isinstance(homotopy.get("epsilon"), Mapping)
    ):
        raise IAROfficialRuntimeSmokeError("homotopy provenance is incomplete")
    continuity = local_evidence.get("continuity")
    if (
        not isinstance(continuity, Mapping)
        or continuity.get("grid_complete") is not True
        or continuity.get("comparison_count")
        != (len(bridge) - 1) * len(sigma_records)
    ):
        raise IAROfficialRuntimeSmokeError("bridge continuity evidence differs")
    core_evidence = local_evidence.get("iar_core")
    if (
        not isinstance(core_evidence, Mapping)
        or core_evidence.get("direct_corrected_core_call") is not True
        or core_evidence.get("replaceable_core") is not False
        or core_evidence.get("projection_nuisance")
        != "mv2v_noop_correct_minus_mv2v_noop_wrong"
        or core_evidence.get("action_source_swaps_diagnostic_only") is not True
    ):
        raise IAROfficialRuntimeSmokeError("IAR core evidence differs")
    for flag, expected in (
        ("donor_plumbing_only", True),
        ("source_reward_calibration_authorized", False),
        ("source_action_invariance_calibration_authorized", False),
        ("training_authorized", False),
        ("training_pair_authorized", False),
        ("scientific_claim_authorized", False),
        ("production_claim_forbidden", True),
        ("paired_target_accessed", False),
        ("forward_callback_present", False),
        ("custom_core_present", False),
    ):
        if local_evidence.get(flag) is not expected:
            raise IAROfficialRuntimeSmokeError(
                f"local evidence flag {flag} differs"
            )
    training = local_evidence.get("training")
    if (
        not isinstance(training, Mapping)
        or training.get("forward_only") is not True
        or training.get("backward_performed") is not False
        or training.get("optimizer_present") is not False
        or training.get("checkpoint_saved") is not False
        or training.get("adapter_present") is not False
    ):
        raise IAROfficialRuntimeSmokeError("training non-authorization evidence differs")

    if len(rank_records) != 4:
        raise IAROfficialRuntimeSmokeError(
            "SP4 receipt requires exactly four rank records"
        )
    ranks: list[int] = []
    digests: list[str] = []
    for record in rank_records:
        if not isinstance(record, Mapping):
            raise IAROfficialRuntimeSmokeError("rank record must be an object")
        if record.get("world_size") != 4 or record.get("ulysses_size") != 4:
            raise IAROfficialRuntimeSmokeError(
                "rank record is not exact world4/Ulysses4"
            )
        rank = record.get("rank")
        if type(rank) is not int:
            raise IAROfficialRuntimeSmokeError("rank must be an integer")
        ranks.append(rank)
        digests.append(
            _require_sha256(
                record.get("local_evidence_digest"), label="rank evidence digest"
            )
        )
    local_digest = _object_sha256(dict(local_evidence))
    if sorted(ranks) != [0, 1, 2, 3] or len(set(digests)) != 1:
        raise IAROfficialRuntimeSmokeError(
            "SP4 ranks did not report one identical full evidence digest"
        )
    if digests[0] != local_digest:
        raise IAROfficialRuntimeSmokeError(
            "rank evidence digest differs from local evidence payload"
        )

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD_NAME,
        "local_evidence": dict(local_evidence),
        "distributed": {
            "world_size": 4,
            "ulysses_size": 4,
            "upstream_shared_step_returns_full_sequence_per_rank": True,
            "all_gather_full_evidence_digest_only": True,
            "rank_records": [
                dict(item) for item in sorted(rank_records, key=lambda row: row["rank"])
            ],
        },
        "engineering_smoke_only": True,
        "donor_plumbing_only": True,
        "source_reward_calibration_authorized": False,
        "source_action_invariance_calibration_authorized": False,
        "training_authorized": False,
        "training_pair_authorized": False,
        "scientific_claim_authorized": False,
        "production_claim_forbidden": True,
        "paired_target_accessed": False,
        "forward_callback_present": False,
        "custom_core_present": False,
    }
    receipt["receipt_digest"] = _object_sha256(receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen exact81 Bernini HAT/IAR official field smoke on SP4"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--preprocessed-parquet-dir", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--wrong-source-row-index", required=True, type=int)
    parser.add_argument("--proposal-source-iid", required=True)
    parser.add_argument("--expected-wrong-source-iid", required=True)
    parser.add_argument("--candidate-clean-latent", required=True)
    parser.add_argument("--expected-candidate-clean-latent-sha256", required=True)
    parser.add_argument("--correct-source-clean-latent", required=True)
    parser.add_argument("--expected-correct-source-clean-latent-sha256", required=True)
    parser.add_argument("--candidate-arm", required=True, choices=("t2v", "r2v", "rv2v"))
    parser.add_argument("--candidate-provenance-receipt", required=True)
    parser.add_argument("--expected-candidate-provenance-receipt-sha256", required=True)
    parser.add_argument("--source-provenance-receipt", required=True)
    parser.add_argument("--expected-source-provenance-receipt-sha256", required=True)
    parser.add_argument("--expected-proposal-source-video-sha256", required=True)
    parser.add_argument("--wrong-source-clean-latent", required=True)
    parser.add_argument("--expected-wrong-source-clean-latent-sha256", required=True)
    parser.add_argument("--wrong-source-provenance-receipt", required=True)
    parser.add_argument("--expected-wrong-source-provenance-receipt-sha256", required=True)
    parser.add_argument("--expected-wrong-source-video-sha256", required=True)
    parser.add_argument("--wrong-source-match-json", required=True)
    parser.add_argument("--expected-wrong-source-match-sha256", required=True)
    parser.add_argument("--action-instruction", required=True)
    parser.add_argument("--expected-action-instruction-sha256", required=True)
    parser.add_argument("--noop-instruction", required=True)
    parser.add_argument("--expected-noop-instruction-sha256", required=True)
    parser.add_argument("--hard-negative-manifest", required=True)
    parser.add_argument("--expected-hard-negative-manifest-sha256", required=True)
    parser.add_argument("--sigmas", nargs="+", type=float, default=DEFAULT_SIGMAS)
    parser.add_argument(
        "--bridge-fractions",
        nargs="+",
        type=float,
        default=DEFAULT_BRIDGE_FRACTIONS,
    )
    parser.add_argument("--noise-seed", type=int, default=20260808)
    parser.add_argument("--output-receipt", required=True)
    parser.add_argument("--allow-incomplete-dataset", action="store_true")
    parser.add_argument("--num-frames", type=int, choices=(81,), default=81)
    parser.add_argument(
        "--expected-bernini-commit", default=dclr.legacy.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=dclr.legacy.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=dclr.legacy.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--launcher-source-sha256", required=True)
    return parser


def validate_cli(
    args: argparse.Namespace,
) -> tuple[tuple[Any, ...], tuple[float, ...], dict[str, Any]]:
    if args.num_frames != NUM_FRAMES:
        raise IAROfficialRuntimeSmokeError("official runtime requires exact81")
    try:
        points, _ = dclr.validate_sigma_request(args.sigmas, None)
    except Exception as error:
        raise IAROfficialRuntimeSmokeError(str(error)) from error
    bridge = validate_bridge_fractions(args.bridge_fractions)
    if type(args.noise_seed) is not int or not 0 <= args.noise_seed < 2**31:
        raise IAROfficialRuntimeSmokeError("noise seed must lie in [0,2^31)")
    if type(args.wrong_source_row_index) is not int or args.wrong_source_row_index < 0:
        raise IAROfficialRuntimeSmokeError(
            "wrong-source row index must be a nonnegative integer"
        )
    for name in (
        "proposal_source_iid",
        "expected_wrong_source_iid",
    ):
        _require_instruction(getattr(args, name), label=name)
    if args.proposal_source_iid == args.expected_wrong_source_iid:
        raise IAROfficialRuntimeSmokeError(
            "proposal source and wrong-source IID must differ"
        )
    action = _require_instruction(args.action_instruction, label="action instruction")
    noop = _require_instruction(args.noop_instruction, label="no-op instruction")
    if action == noop:
        raise IAROfficialRuntimeSmokeError("action and no-op instructions must differ")
    action_sha = _instruction_sha256(action)
    noop_sha = _instruction_sha256(noop)
    if action_sha != _require_sha256(
        args.expected_action_instruction_sha256,
        label="expected action instruction SHA-256",
    ):
        raise IAROfficialRuntimeSmokeError("action instruction SHA-256 differs")
    if noop_sha != _require_sha256(
        args.expected_noop_instruction_sha256,
        label="expected no-op instruction SHA-256",
    ):
        raise IAROfficialRuntimeSmokeError("no-op instruction SHA-256 differs")
    hard_manifest = load_hard_negative_manifest(
        args.hard_negative_manifest,
        expected_sha256=args.expected_hard_negative_manifest_sha256,
        action_instruction_sha256=action_sha,
        noop_instruction_sha256=noop_sha,
    )

    absolute_paths = (
        "bernini_root",
        "veomni_root",
        "checkpoint",
        "checkpoint_content_manifest",
        "preprocessed_parquet_dir",
        "dataset_summary",
        "candidate_clean_latent",
        "correct_source_clean_latent",
        "candidate_provenance_receipt",
        "source_provenance_receipt",
        "wrong_source_clean_latent",
        "wrong_source_provenance_receipt",
        "wrong_source_match_json",
        "hard_negative_manifest",
        "output_receipt",
    )
    for name in absolute_paths:
        value = getattr(args, name)
        if not isinstance(value, str) or not Path(value).expanduser().is_absolute():
            raise IAROfficialRuntimeSmokeError(f"{name} must be an absolute path")
    digest_names = (
        "expected_candidate_clean_latent_sha256",
        "expected_correct_source_clean_latent_sha256",
        "expected_candidate_provenance_receipt_sha256",
        "expected_source_provenance_receipt_sha256",
        "expected_proposal_source_video_sha256",
        "expected_wrong_source_clean_latent_sha256",
        "expected_wrong_source_provenance_receipt_sha256",
        "expected_wrong_source_video_sha256",
        "expected_wrong_source_match_sha256",
        "expected_hard_negative_manifest_sha256",
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
        "launcher_source_sha256",
    )
    for name in digest_names:
        _require_sha256(getattr(args, name), label=name)
    bernini_sha = _require_sha1(
        args.expected_bernini_commit, label="expected Bernini commit"
    )
    veomni_sha = _require_sha1(
        args.expected_veomni_commit, label="expected VeOmni commit"
    )
    _require_sha1(args.method_source_revision, label="method source revision")
    if bernini_sha != dclr.legacy.BERNINI_OFFICIAL_COMMIT:
        raise IAROfficialRuntimeSmokeError("Bernini revision differs from pinned release")
    if veomni_sha != dclr.legacy.VEOMNI_TESTED_COMMIT:
        raise IAROfficialRuntimeSmokeError("VeOmni revision differs from pinned release")
    if args.expected_checkpoint_tree_sha256 != dclr.legacy.CHECKPOINT_TREE_SHA256:
        raise IAROfficialRuntimeSmokeError("checkpoint tree digest differs")
    return points, bridge, hard_manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    points, bridge_fractions, hard_manifest = validate_cli(args)
    try:
        output_receipt = dclr._validate_output_path(args.output_receipt)
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            dclr.legacy.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = dclr.legacy.validate_checkpoint(
            args.checkpoint
        )
    except Exception as error:
        raise IAROfficialRuntimeSmokeError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % 4:
        raise IAROfficialRuntimeSmokeError(
            "1.3B attention heads must divide Ulysses=4"
        )
    dclr.legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer

    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.training.data import encode_renderer_messages

    distributed = dclr.legacy.distributed_contract()
    if distributed.world_size != 4 or distributed.ulysses_size != 4:
        raise IAROfficialRuntimeSmokeError(
            "one official runtime group requires exact world4/Ulysses4"
        )
    device, backend = dclr.legacy.initialise_distributed(distributed)
    from bernini.parallel import init_parallel_state

    init_parallel_state(ulysses_size=4)
    dclr.legacy.seed_same_sample(args.noise_seed)

    checkpoint_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_rows[0] = {
                "ok": True,
                "identity": dclr.source_audit.validate_checkpoint_content(
                    checkpoint,
                    Path(args.checkpoint_content_manifest).expanduser(),
                ),
            }
        except Exception as error:
            checkpoint_rows[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_rows, src=0)
    checkpoint_result = checkpoint_rows[0]
    if not isinstance(checkpoint_result, Mapping) or checkpoint_result.get("ok") is not True:
        raise IAROfficialRuntimeSmokeError(
            f"rank-zero checkpoint content validation failed: {checkpoint_result}"
        )
    checkpoint_identity = dict(checkpoint_result["identity"])

    dataset = dclr.legacy.ParquetRowStore(args.preprocessed_parquet_dir)
    try:
        dataset_summary = dclr.legacy.validate_preprocessed_dataset_summary(
            args.dataset_summary,
            dataset,
            allow_incomplete=bool(args.allow_incomplete_dataset),
        )
    except Exception as error:
        raise IAROfficialRuntimeSmokeError(str(error)) from error
    if args.wrong_source_row_index >= len(dataset):
        raise IAROfficialRuntimeSmokeError("wrong-source row index is out of range")
    try:
        wrong_row = dclr._load_message_only_row(
            dataset, args.wrong_source_row_index
        )
        wrong_iid = dclr._row_iid(wrong_row, label="wrong source")
    except Exception as error:
        raise IAROfficialRuntimeSmokeError(str(error)) from error
    if wrong_iid != args.expected_wrong_source_iid:
        raise IAROfficialRuntimeSmokeError("runtime row IID differs from CLI binding")
    wrong_source_video_sha256 = str(wrong_row["source_video_sha256"])
    if wrong_source_video_sha256 != args.expected_wrong_source_video_sha256:
        raise IAROfficialRuntimeSmokeError(
            "wrong-source message-only video SHA-256 differs"
        )
    try:
        wrong_match = dclr.load_wrong_source_match_manifest(
            args.wrong_source_match_json,
            expected_sha256=args.expected_wrong_source_match_sha256,
            candidate_iid=args.proposal_source_iid,
            candidate_source_video_sha256=args.expected_proposal_source_video_sha256,
            wrong_source_iid=wrong_iid,
            wrong_source_video_sha256=wrong_source_video_sha256,
        )
    except Exception as error:
        raise IAROfficialRuntimeSmokeError(str(error)) from error

    action_sample = build_canonical_renderer_sample(
        args.action_instruction,
        expected_instruction_sha256=args.expected_action_instruction_sha256,
    )
    noop_sample = build_canonical_renderer_sample(
        args.noop_instruction,
        expected_instruction_sha256=args.expected_noop_instruction_sha256,
    )
    negative_samples = tuple(
        build_canonical_renderer_sample(
            item["instruction"],
            expected_instruction_sha256=item["instruction_sha256"],
        )
        for item in hard_manifest["hard_negatives"]
    )
    message_schema = canonical_message_schema_evidence(
        action_sample=action_sample,
        action_instruction_sha256=args.expected_action_instruction_sha256,
        negative_samples=negative_samples,
        hard_negative_instruction_sha256s=tuple(
            item["instruction_sha256"] for item in hard_manifest["hard_negatives"]
        ),
        noop_sample=noop_sample,
        noop_instruction_sha256=args.expected_noop_instruction_sha256,
    )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **dclr.legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        dclr.legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    except Exception as error:
        raise IAROfficialRuntimeSmokeError(str(error)) from error
    renderer = BerniniRendererModel(config)
    renderer.requires_grad_(False)
    renderer.eval()
    renderer.to(device)
    if any(parameter.requires_grad for parameter in renderer.parameters()):
        raise IAROfficialRuntimeSmokeError(
            "frozen renderer retains trainable parameters"
        )
    if any("lora" in name.lower() for name, _ in renderer.named_modules()):
        raise IAROfficialRuntimeSmokeError(
            "official frozen runtime unexpectedly contains a LoRA module"
        )
    try:
        model_id, transformer = dclr._active_transformer(renderer)
    except Exception as error:
        raise IAROfficialRuntimeSmokeError(str(error)) from error

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=dclr.legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    try:
        proposal_cpu, proposal_artifact = dclr.load_normalized_clean_latent_artifact(
            args.candidate_clean_latent,
            expected_sha256=args.expected_candidate_clean_latent_sha256,
            expected_role="native_sampler_proposal",
        )
        source_cpu, source_artifact = dclr.load_normalized_clean_latent_artifact(
            args.correct_source_clean_latent,
            expected_sha256=args.expected_correct_source_clean_latent_sha256,
            expected_role="source_video_condition",
        )
        native_provenance = dclr.validate_native_rollout_provenance(
            candidate_receipt_path=args.candidate_provenance_receipt,
            expected_candidate_receipt_sha256=(
                args.expected_candidate_provenance_receipt_sha256
            ),
            source_receipt_path=args.source_provenance_receipt,
            expected_source_receipt_sha256=(
                args.expected_source_provenance_receipt_sha256
            ),
            candidate_arm=args.candidate_arm,
            candidate_artifact_path=args.candidate_clean_latent,
            candidate_artifact_sha256=args.expected_candidate_clean_latent_sha256,
            source_artifact_path=args.correct_source_clean_latent,
            source_artifact_sha256=(
                args.expected_correct_source_clean_latent_sha256
            ),
            expected_source_video_sha256=(
                args.expected_proposal_source_video_sha256
            ),
            expected_action_prompt_sha256=args.expected_action_instruction_sha256,
            expected_checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
        )
        wrong_cpu, wrong_artifact = dclr.load_normalized_clean_latent_artifact(
            args.wrong_source_clean_latent,
            expected_sha256=args.expected_wrong_source_clean_latent_sha256,
            expected_role="source_video_condition",
        )
        wrong_provenance = dclr.validate_source_condition_provenance(
            source_receipt_path=args.wrong_source_provenance_receipt,
            expected_source_receipt_sha256=(
                args.expected_wrong_source_provenance_receipt_sha256
            ),
            source_iid=wrong_iid,
            source_artifact_path=args.wrong_source_clean_latent,
            source_artifact_sha256=(
                args.expected_wrong_source_clean_latent_sha256
            ),
            expected_source_video_sha256=wrong_source_video_sha256,
            expected_checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
        )
    except Exception as error:
        raise IAROfficialRuntimeSmokeError(str(error)) from error
    if (
        native_provenance["checkpoint_content_identity"] != checkpoint_identity
        or wrong_provenance["checkpoint_content_identity"] != checkpoint_identity
    ):
        raise IAROfficialRuntimeSmokeError(
            "latent provenance and active checkpoint identities differ"
        )
    if not (
        tuple(proposal_cpu.shape)
        == tuple(source_cpu.shape)
        == tuple(wrong_cpu.shape)
    ):
        raise IAROfficialRuntimeSmokeError(
            "proposal/correct/wrong clean latent geometry differs"
        )
    if torch.equal(source_cpu, wrong_cpu):
        raise IAROfficialRuntimeSmokeError(
            "wrong-source latent is tensor-identical to correct source"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.noise_seed)
    epsilon_cpu = torch.randn(
        tuple(proposal_cpu.shape), generator=generator, dtype=torch.float32
    )
    proposal = proposal_cpu.to(device)
    source = source_cpu.to(device)
    wrong = wrong_cpu.to(device)
    epsilon = epsilon_cpu.to(device)

    with torch.inference_mode():
        try:
            action_t2v = _tokenize_canonical_condition(
                renderer=renderer,
                tokenizer=tokenizer,
                encode_renderer_messages=encode_renderer_messages,
                sample=action_sample,
                expected_instruction_sha256=(
                    args.expected_action_instruction_sha256
                ),
                task_name="t2v",
                device=device,
            )
            negatives_t2v = tuple(
                _tokenize_canonical_condition(
                    renderer=renderer,
                    tokenizer=tokenizer,
                    encode_renderer_messages=encode_renderer_messages,
                    sample=sample,
                    expected_instruction_sha256=item["instruction_sha256"],
                    task_name="t2v",
                    device=device,
                )
                for sample, item in zip(
                    negative_samples, hard_manifest["hard_negatives"]
                )
            )
            noop_mv2v = _tokenize_canonical_condition(
                renderer=renderer,
                tokenizer=tokenizer,
                encode_renderer_messages=encode_renderer_messages,
                sample=noop_sample,
                expected_instruction_sha256=args.expected_noop_instruction_sha256,
                task_name="mv2v",
                device=device,
            )
            action_mv2v = _tokenize_canonical_condition(
                renderer=renderer,
                tokenizer=tokenizer,
                encode_renderer_messages=encode_renderer_messages,
                sample=action_sample,
                expected_instruction_sha256=(
                    args.expected_action_instruction_sha256
                ),
                task_name="mv2v",
                device=device,
            )
            _condition_contract(
                action_t2v=action_t2v,
                negatives_t2v=negatives_t2v,
                noop_mv2v=noop_mv2v,
                action_mv2v=action_mv2v,
            )
        except Exception as error:
            if isinstance(error, IAROfficialRuntimeSmokeError):
                raise
            raise IAROfficialRuntimeSmokeError(str(error)) from error

        cell_results: list[OfficialCellResult] = []
        bridge_records: list[dict[str, Any]] = []
        geometry: Optional[dict[str, Any]] = None
        for bridge_fraction in bridge_fractions:
            bridge_clean = construct_bridge_clean(
                source, proposal, bridge_fraction
            )
            expected_bridge = (
                (1.0 - float(bridge_fraction)) * source.float()
                + float(bridge_fraction) * proposal.float()
            )
            if not _tensor_equal(bridge_clean, expected_bridge):
                raise IAROfficialRuntimeSmokeError(
                    "internal bridge clean coordinate differs from HAT formula"
                )
            bridge_record = {
                "bridge_fraction": float(bridge_fraction),
                "bridge_fraction_float32_bits_hex": struct.pack(
                    "!f", float(bridge_fraction)
                ).hex(),
                "clean_coordinate": _tensor_identity(
                    bridge_clean,
                    label=f"bridge_clean_{struct.pack('!f', float(bridge_fraction)).hex()}",
                ),
                "source_endpoint_exact": bool(
                    float(bridge_fraction) != 0.0 or bridge_clean is source
                ),
                "proposal_endpoint_exact": bool(
                    float(bridge_fraction) != 1.0 or bridge_clean is proposal
                ),
                "constructed_from_shared_P_and_S": True,
            }
            bridge_records.append(bridge_record)
            for point in points:
                try:
                    bundle = dclr.build_same_state_query_bundle(
                        transformer,
                        correct_source_spatial=source,
                        wrong_source_spatial=wrong,
                        student_clean_spatial=bridge_clean,
                        epsilon_spatial=epsilon,
                        point=point,
                    )
                except Exception as error:
                    raise IAROfficialRuntimeSmokeError(str(error)) from error
                if (
                    bundle.student_clean_spatial is not bridge_clean
                    or bundle.epsilon_spatial is not epsilon
                    or bundle.correct_source_spatial is not source
                    or bundle.wrong_source_spatial is not wrong
                ):
                    raise IAROfficialRuntimeSmokeError(
                        "DCLR bundle did not retain the exact P/S/epsilon-derived objects"
                    )
                result = _run_official_cell(
                    renderer=renderer,
                    model_id=model_id,
                    bundles=(bundle,),
                    bridge_fraction=float(bridge_fraction),
                    action_t2v_condition=action_t2v,
                    hard_negative_t2v_conditions=negatives_t2v,
                    noop_mv2v_condition=noop_mv2v,
                    action_mv2v_condition=action_mv2v,
                    correct_source_sha256=(
                        args.expected_proposal_source_video_sha256
                    ),
                    wrong_source_sha256s=(wrong_source_video_sha256,),
                )
                current_geometry = dict(result.record["geometry"])
                if geometry is None:
                    geometry = current_geometry
                elif geometry != current_geometry:
                    raise IAROfficialRuntimeSmokeError(
                        "exact81 query geometry changed across the homotopy grid"
                    )
                cell_results.append(result)
    if geometry is None:
        raise IAROfficialRuntimeSmokeError("official field grid is empty")
    continuity = build_bridge_continuity(
        cell_results,
        bridge_fractions=bridge_fractions,
        points=points,
    )
    branch_order = list(
        _branch_names(hard_manifest["hard_negative_count"], 1)
    )
    cell_records = [dict(item.record) for item in cell_results]
    core_grid_closure = {
        "cell_digests": [item["cell_digest"] for item in cell_records],
        "field_set_digests": [item["field_set_digest"] for item in cell_records],
        "independent_recompute_digests": [
            item["independent_recompute"]["digest"] for item in cell_records
        ],
        "core_receipt_digests": [
            item["core_receipt_digest"] for item in cell_records
        ],
    }
    homotopy = {
        "formula": "q=(1-sigma)*((1-lambda)*S+lambda*P)+sigma*epsilon",
        "constructed_inside_runtime": True,
        "caller_provided_sigma_states": False,
        "proposal_P": _tensor_identity(proposal_cpu, label="shared_proposal_P"),
        "correct_source_S": _tensor_identity(source_cpu, label="shared_correct_source_S"),
        "epsilon": _tensor_identity(epsilon_cpu, label="shared_epsilon"),
        "noise_seed": args.noise_seed,
        "epsilon_generated_once_on_cpu": True,
        "one_proposal_P_for_all_cells": True,
        "one_correct_source_S_for_all_cells": True,
        "one_epsilon_for_all_cells": True,
        "bridge_clean_coordinates": bridge_records,
        "paired_target_used_as_clean_coordinate": False,
        "linear_bridge_is_field_query_not_reconstruction_target": True,
    }
    local_evidence: dict[str, Any] = {
        "method": METHOD_NAME,
        "method_source_revision": args.method_source_revision.lower(),
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "launcher_source_sha256": args.launcher_source_sha256,
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "checkpoint_path": str(checkpoint),
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "checkpoint_content_identity": checkpoint_identity,
        "dataset_signature": dataset.signature,
        "dataset_summary_sha256": dataset_summary["sha256"],
        "candidate": {
            "canonical_message_schema": message_schema,
            "proposal_source_iid": args.proposal_source_iid,
            "proposal_origin": "native_rollout_predecode_latent",
            "proposal_artifact": proposal_artifact,
            "correct_source_artifact": source_artifact,
            "native_provenance": native_provenance,
            "paired_target_accessed": False,
        },
        "wrong_sources": [
            {
                "row_index": args.wrong_source_row_index,
                "iid": wrong_iid,
                "source_video_sha256": wrong_source_video_sha256,
                "message_template_columns_loaded": [
                    "iid",
                    "inputs",
                    "source_video_sha256",
                ],
                "artifact": wrong_artifact,
                "provenance": wrong_provenance,
                "match_manifest": wrong_match,
                "paired_target_accessed": False,
            }
        ],
        "text_conditions": {
            "action_instruction_sha256": action_t2v.instruction_sha256,
            "noop_instruction_sha256": noop_mv2v.instruction_sha256,
            "t2v_action_prompt_sha256": action_t2v.prompt_sha256,
            "t2v_hard_negative_prompt_sha256s": [
                item.prompt_sha256 for item in negatives_t2v
            ],
            "mv2v_noop_prompt_sha256": noop_mv2v.prompt_sha256,
            "mv2v_action_prompt_sha256": action_mv2v.prompt_sha256,
            "official_tokenization": True,
        },
        "hard_negative_manifest": hard_manifest,
        "homotopy": homotopy,
        "bridge_fractions": [float(item) for item in bridge_fractions],
        "sigmas": [point.as_dict() for point in points],
        "num_frames": NUM_FRAMES,
        "latent_phases": LATENT_PHASES,
        "patch_size": list(PATCH_SIZE),
        "geometry": geometry,
        "hard_negative_count": hard_manifest["hard_negative_count"],
        "wrong_source_count": 1,
        "branch_order": branch_order,
        "cell_records": cell_records,
        "continuity": continuity,
        "iar_core": {
            "method": iar_core.METHOD_NAME,
            "receipt_schema": iar_core.RECEIPT_SCHEMA,
            "direct_corrected_core_call": True,
            "replaceable_core": False,
            "projection_nuisance": "mv2v_noop_correct_minus_mv2v_noop_wrong",
            "action_source_swaps_diagnostic_only": True,
            "M_equals_one_plumbing_only_uncalibrated": True,
            "source_action_invariance_calibration_authorized": False,
            "training_authorized_by_diagnostic": False,
            "independent_recompute_every_cell": True,
            "full_cell_grid_closure": core_grid_closure,
            "full_cell_grid_digest": _object_sha256(core_grid_closure),
        },
        "forwards_per_rank": len(cell_records) * len(branch_order),
        "forward_implementation": FORWARD_IMPLEMENTATION,
        "active_model_id": model_id,
        "adapter_state": "absent_frozen_base",
        "model_mode": "eval_inference_mode",
        "raw_positive_conditional_only": True,
        "cfg": False,
        "apg": False,
        "backend": backend,
        "donor_plumbing_only": True,
        "source_reward_calibration_authorized": False,
        "source_action_invariance_calibration_authorized": False,
        "training_authorized": False,
        "training_pair_authorized": False,
        "scientific_claim_authorized": False,
        "production_claim_forbidden": True,
        "paired_target_accessed": False,
        "forward_callback_present": False,
        "custom_core_present": False,
        "training": {
            "forward_only": True,
            "backward_performed": False,
            "optimizer_present": False,
            "checkpoint_saved": False,
            "adapter_present": False,
        },
    }

    local_digest = _object_sha256(local_evidence)
    rank_record = {
        "rank": distributed.rank,
        "world_size": distributed.world_size,
        "ulysses_size": distributed.ulysses_size,
        "local_evidence_digest": local_digest,
    }
    rank_records: list[Optional[dict[str, Any]]] = [None] * 4
    dist.all_gather_object(rank_records, rank_record)
    if any(item is None for item in rank_records):
        raise IAROfficialRuntimeSmokeError(
            "SP4 evidence gather returned an empty rank"
        )
    receipt = assemble_sp4_receipt(
        local_evidence, rank_records  # type: ignore[arg-type]
    )
    dist.barrier()
    if distributed.rank == 0:
        try:
            dclr._write_receipt_atomically(output_receipt, receipt)
        except Exception as error:
            raise IAROfficialRuntimeSmokeError(str(error)) from error
        print(
            _canonical_json_bytes(
                {
                    "receipt": str(output_receipt),
                    "receipt_digest": receipt["receipt_digest"],
                    "forwards_per_rank": local_evidence["forwards_per_rank"],
                    "cell_count": len(cell_records),
                    "training_authorized": False,
                }
            ).decode("ascii")
        )
    dist.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
