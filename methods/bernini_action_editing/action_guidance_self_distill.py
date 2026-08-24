"""Frozen self-guidance distillation on one exact T2V noisy query.

This module distils a model's own conditional action prior; it never consumes a
teacher RGB video or clean latent target.  A real model callback is invoked at
one identical noised state/timestep/proposal/seed for three branches:

* frozen conditional action velocity;
* frozen unconditional or semantic-no-op velocity; and
* conditional action velocity with the student Action-LoRA enabled.

The two frozen velocities define a detached, norm-bounded APG/CFG-style action
residual.  The student can train only Action-LoRA parameters on ``attn2.to_q``
and ``attn2.to_out``.  Its correction is kept inside a base-velocity trust
region and all loss energies are formed in FP32.  Only high- and mid-sigma
queries are eligible.  The convenience runner performs backward and then
fails closed unless the allowed module scope has finite, aggregate-nonzero
gradients and every other parameter remains gradient-free.

There is deliberately no CLI, sampler, optimizer, or trainer integration here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
import struct
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_NAME = "bernini-action-guidance-self-distill-v1"
RECEIPT_SCHEMA = "bernini-action-guidance-self-distill-receipt-v1"
QUERY_MODE = "t2v"
CHECKPOINT_BINDING_ATTRIBUTE = "action_distill_checkpoint_sha256"
NATIVE_T2V_RECEIPT_SCHEMA = "bernini-native-identity-generation-canary-v1"
NATIVE_T2V_RECEIPT_METHOD = "frozen-bernini-native-identity-generation-canary"
NATIVE_T2V_ARTIFACT_ROLE = "native_sampler_proposal"
NATIVE_T2V_ARTIFACT_ORIGIN = "native_sampler_before_vae_decode"
TEACHER_ACTION_BRANCH = "frozen_conditional_action"
TEACHER_NOOP_BRANCH = "frozen_unconditional_or_noop"
STUDENT_ACTION_BRANCH = "student_action_lora"
FORWARD_BRANCHES = (
    TEACHER_ACTION_BRANCH,
    TEACHER_NOOP_BRANCH,
    STUDENT_ACTION_BRANCH,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_CONDITION_KEY = re.compile(
    r"(?:"
    r"(?:teacher|target)_?(?:rgb|pixels?|images?|frames?|videos?|clean_latents?)"
    r"|(?:rgb|pixels?|clean_latents?|target_video|pixel_target)"
    r")",
    re.IGNORECASE,
)


class ActionGuidanceSelfDistillError(RuntimeError):
    """Raised before unbound guidance or an invalid gradient can be used."""


@dataclass(frozen=True)
class NativeT2VProposalEvidence:
    """Loader-sealed identity of one native pre-decode T2V proposal.

    This object contains metadata only.  The loader hashes the safetensors file
    and inspects its header, but never materializes the clean latent tensor.
    """

    receipt_path: str
    receipt_file_sha256: str
    receipt_digest: str
    receipt_schema: str
    arm: str
    proposal_artifact_path: str
    proposal_artifact_sha256: str
    proposal_artifact_role: str
    proposal_predecode: bool
    rollout_seed: int
    action_prompt_sha256: str
    checkpoint_sha256: str
    target_video: bool
    paired_target: bool
    validation_digest: str

    def validate(self) -> None:
        values = asdict(self)
        declared = values.pop("validation_digest", None)
        expected = hashlib.sha256(_canonical_json(values)).hexdigest()
        if declared != expected:
            raise ActionGuidanceSelfDistillError(
                "native T2V proposal evidence seal differs"
            )
        for name in (
            "receipt_file_sha256",
            "receipt_digest",
            "proposal_artifact_sha256",
            "action_prompt_sha256",
            "checkpoint_sha256",
            "validation_digest",
        ):
            if _SHA256.fullmatch(str(getattr(self, name))) is None:
                raise ActionGuidanceSelfDistillError(
                    f"native T2V evidence {name} must be a lowercase SHA-256"
                )
        if (
            self.receipt_schema != NATIVE_T2V_RECEIPT_SCHEMA
            or self.arm != QUERY_MODE
            or self.proposal_artifact_role != NATIVE_T2V_ARTIFACT_ROLE
            or self.proposal_predecode is not True
            or self.target_video is not False
            or self.paired_target is not False
        ):
            raise ActionGuidanceSelfDistillError(
                "native T2V proposal evidence role/access closure differs"
            )
        for name in ("receipt_path", "proposal_artifact_path"):
            path = Path(str(getattr(self, name)))
            if not path.is_absolute():
                raise ActionGuidanceSelfDistillError(
                    f"native T2V evidence {name} must be absolute"
                )
        if (
            isinstance(self.rollout_seed, bool)
            or not isinstance(self.rollout_seed, int)
            or not 0 <= self.rollout_seed < 2**63
        ):
            raise ActionGuidanceSelfDistillError(
                "native T2V evidence seed must be in [0,2^63)"
            )


@dataclass(frozen=True)
class DistillConfig:
    """Fixed numerical controls for self-guidance and base preservation."""

    high_sigma_min: float = 0.55
    mid_sigma_min: float = 0.25
    mid_sigma_weight: float = 0.5
    guidance_scale: float = 4.0
    apg_parallel_eta: float = 0.5
    raw_residual_l2_cap: float = 50.0
    teacher_max_reference_rms_ratio: float = 2.0
    student_base_trust_ratio: float = 0.20
    trust_penalty_weight: float = 0.10
    epsilon: float = 1.0e-6

    def validate(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ActionGuidanceSelfDistillError(f"{name} must be finite")
            if not math.isfinite(float(value)):
                raise ActionGuidanceSelfDistillError(f"{name} must be finite")
        if not 0.0 < self.mid_sigma_min < self.high_sigma_min < 1.0:
            raise ActionGuidanceSelfDistillError(
                "sigma gates must satisfy 0 < mid < high < 1"
            )
        if not 0.0 < self.mid_sigma_weight <= 1.0:
            raise ActionGuidanceSelfDistillError(
                "mid_sigma_weight must lie in (0,1]"
            )
        for name in (
            "guidance_scale",
            "raw_residual_l2_cap",
            "teacher_max_reference_rms_ratio",
            "student_base_trust_ratio",
            "epsilon",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ActionGuidanceSelfDistillError(f"{name} must be positive")
        if not 0.0 <= self.apg_parallel_eta <= 1.0:
            raise ActionGuidanceSelfDistillError(
                "apg_parallel_eta must lie in [0,1]"
            )
        if self.trust_penalty_weight < 0.0:
            raise ActionGuidanceSelfDistillError(
                "trust_penalty_weight must be non-negative"
            )


@dataclass(frozen=True)
class DistillProvenance:
    """Content binding shared by every forward in one distillation cell."""

    teacher_checkpoint_sha256: str
    student_base_checkpoint_sha256: str
    proposal_sha256: str
    proposal_iid: str
    rollout_seed: int
    query_state_sha256: str
    action_condition_sha256: str
    noop_condition_sha256: str
    proposal_receipt_sha256: str
    proposal_receipt_digest: str
    proposal_prompt_sha256: str
    proposal_evidence: NativeT2VProposalEvidence
    proposal_origin: str = "model_generated_t2v_prior"

    def validate(self) -> None:
        for name in (
            "teacher_checkpoint_sha256",
            "student_base_checkpoint_sha256",
            "proposal_sha256",
            "query_state_sha256",
            "action_condition_sha256",
            "noop_condition_sha256",
            "proposal_receipt_sha256",
            "proposal_receipt_digest",
            "proposal_prompt_sha256",
        ):
            if _SHA256.fullmatch(str(getattr(self, name))) is None:
                raise ActionGuidanceSelfDistillError(
                    f"{name} must be a lowercase SHA-256"
                )
        if self.teacher_checkpoint_sha256 != self.student_base_checkpoint_sha256:
            raise ActionGuidanceSelfDistillError(
                "teacher and student base checkpoint provenance differs"
            )
        if self.action_condition_sha256 == self.noop_condition_sha256:
            raise ActionGuidanceSelfDistillError(
                "action and unconditional/no-op conditions must differ"
            )
        if (
            not isinstance(self.proposal_iid, str)
            or not self.proposal_iid.strip()
            or "\x00" in self.proposal_iid
        ):
            raise ActionGuidanceSelfDistillError("proposal_iid must be non-empty")
        if (
            isinstance(self.rollout_seed, bool)
            or not isinstance(self.rollout_seed, int)
            or not 0 <= self.rollout_seed < 2**63
        ):
            raise ActionGuidanceSelfDistillError(
                "rollout_seed must be an integer in [0,2^63)"
            )
        if self.proposal_origin != "model_generated_t2v_prior":
            raise ActionGuidanceSelfDistillError(
                "proposal must originate from the model-generated T2V prior"
            )
        if not isinstance(self.proposal_evidence, NativeT2VProposalEvidence):
            raise ActionGuidanceSelfDistillError(
                "provenance requires loader-validated native T2V proposal evidence"
            )
        self.proposal_evidence.validate()
        evidence = self.proposal_evidence
        if (
            self.teacher_checkpoint_sha256 != evidence.checkpoint_sha256
            or self.proposal_sha256 != evidence.proposal_artifact_sha256
            or self.rollout_seed != evidence.rollout_seed
            or self.proposal_receipt_sha256 != evidence.receipt_file_sha256
            or self.proposal_receipt_digest != evidence.receipt_digest
            or self.proposal_prompt_sha256 != evidence.action_prompt_sha256
        ):
            raise ActionGuidanceSelfDistillError(
                "distillation provenance differs from verified native T2V evidence"
            )


@dataclass(frozen=True)
class ModelForwardRequest:
    """Immutable branch request passed to the real model callback."""

    branch: str
    mode: str
    noised_state: Any
    timestep: Any
    condition: Any
    condition_sha256: str
    adapter_enabled: bool
    checkpoint_sha256: str
    proposal_sha256: str
    proposal_iid: str
    rollout_seed: int
    query_state_sha256: str
    timestep_sha256: str


@dataclass(frozen=True)
class ModelForwardResponse:
    """Velocity plus callback-side binding evidence for one real forward."""

    velocity: Any
    branch: str
    mode: str
    adapter_enabled_observed: bool
    checkpoint_sha256: str
    proposal_sha256: str
    rollout_seed: int
    query_state_sha256: str
    timestep_sha256: str
    model_forward_executed: bool = True


@dataclass(frozen=True)
class GradientAudit:
    allowed_parameter_names: tuple[str, ...]
    finite_gradient_names: tuple[str, ...]
    nonzero_gradient_names: tuple[str, ...]
    per_parameter_fp32_energy: Mapping[str, float]
    total_fp32_gradient_energy: float
    forbidden_parameter_gradients: tuple[str, ...]
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DistillForwardResult:
    loss: Any
    distill_energy: Any
    trust_penalty: Any
    teacher_residual: Any
    student_residual: Any
    raw_student_correction: Any
    trusted_student_correction: Any
    sigma: float
    sigma_stratum: str
    sigma_gate_weight: float
    model_object_id: int
    allowed_parameter_ids: Mapping[str, int]
    diagnostics: Mapping[str, Any]
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class DistillStepResult:
    forward: DistillForwardResult
    gradient_audit: GradientAudit
    receipt: Mapping[str, Any]


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise ActionGuidanceSelfDistillError(
            "action-guidance self-distillation requires torch"
        ) from error
    return torch


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def tensor_sha256(value: Any) -> str:
    """Hash tensor geometry and exact storage bytes without NumPy."""

    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise ActionGuidanceSelfDistillError("tensor hash input is not a tensor")
    if value.device.type == "meta" or value.layout != torch.strided:
        raise ActionGuidanceSelfDistillError(
            "tensor hash input must be a concrete strided tensor"
        )
    tensor = value.detach().to(device="cpu").contiguous()
    octets = tensor.reshape(-1).view(torch.uint8).contiguous().clone()
    if octets.storage_offset() != 0 or not octets.is_contiguous():
        raise ActionGuidanceSelfDistillError(
            "tensor hash byte clone is not zero-offset contiguous storage"
        )
    untyped_storage = getattr(octets, "untyped_storage", None)
    storage = untyped_storage() if callable(untyped_storage) else octets.storage()
    nbytes_method = getattr(storage, "nbytes", None)
    storage_nbytes = (
        int(nbytes_method()) if callable(nbytes_method) else int(storage.size())
    )
    if storage_nbytes != int(octets.numel()):
        raise ActionGuidanceSelfDistillError(
            "tensor hash byte storage contains padding or aliased bytes"
        )
    raw = bytes(storage)
    if len(raw) != int(octets.numel()):
        raise ActionGuidanceSelfDistillError(
            "tensor hash byte serialization length differs"
        )
    header = {
        "shape": [int(item) for item in tensor.shape],
        "dtype": str(tensor.dtype),
    }
    header_bytes = _canonical_json(header)
    digest = hashlib.sha256()
    digest.update(len(header_bytes).to_bytes(8, "big"))
    digest.update(header_bytes)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if _SHA256.fullmatch(str(value)) is None:
        raise ActionGuidanceSelfDistillError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _plain_absolute_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        raise ActionGuidanceSelfDistillError(f"{label} must be absolute")
    if requested.is_symlink():
        raise ActionGuidanceSelfDistillError(f"{label} must not be a symlink")
    try:
        path = requested.resolve(strict=True)
    except OSError as error:
        raise ActionGuidanceSelfDistillError(
            f"{label} is unavailable: {error}"
        ) from error
    if not path.is_file():
        raise ActionGuidanceSelfDistillError(f"{label} must be a plain file")
    return path


def _canonical_utf8_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ActionGuidanceSelfDistillError(
            f"native T2V receipt is not canonical JSON: {error}"
        ) from error


def _load_content_addressed_receipt(
    path_value: str | Path, *, expected_sha256: str
) -> tuple[dict[str, Any], Path, str, str]:
    path = _plain_absolute_file(path_value, label="native T2V proposal receipt")
    expected = _require_sha256(
        expected_sha256, label="native T2V proposal receipt SHA-256"
    )
    actual = _file_sha256(path)
    if actual != expected:
        raise ActionGuidanceSelfDistillError(
            "native T2V proposal receipt file SHA-256 differs"
        )
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ActionGuidanceSelfDistillError("native T2V proposal receipt is too large")

    def reject_duplicate_keys(pairs: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ActionGuidanceSelfDistillError(
                    f"native T2V receipt repeats JSON key: {key}"
                )
            result[key] = item
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except ActionGuidanceSelfDistillError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ActionGuidanceSelfDistillError(
            f"cannot decode native T2V proposal receipt: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ActionGuidanceSelfDistillError(
            "native T2V proposal receipt must be an object"
        )
    body = dict(value)
    declared = _require_sha256(
        body.pop("receipt_digest", None),
        label="native T2V proposal embedded receipt digest",
    )
    if hashlib.sha256(_canonical_utf8_json(body)).hexdigest() != declared:
        raise ActionGuidanceSelfDistillError(
            "native T2V proposal embedded receipt digest differs"
        )
    value["receipt_digest"] = declared
    return value, path, actual, declared


def _validate_safetensors_proposal_header(
    path: Path, *, receipt_artifact: Mapping[str, Any]
) -> None:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError("missing safetensors header length")
            header_size = int(struct.unpack("<Q", prefix)[0])
            if not 2 <= header_size <= 1024 * 1024:
                raise ValueError("safetensors header size is invalid")
            header_bytes = handle.read(header_size)
            if len(header_bytes) != header_size:
                raise ValueError("truncated safetensors header")
        header = json.loads(header_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ActionGuidanceSelfDistillError(
            f"cannot inspect native T2V proposal safetensors header: {error}"
        ) from error
    if not isinstance(header, dict) or set(header) != {
        "__metadata__",
        "normalized_clean_latent",
    }:
        raise ActionGuidanceSelfDistillError(
            "native T2V proposal safetensors keys differ"
        )
    metadata = header.get("__metadata__")
    expected_metadata = {
        "coordinate": "bernini_normalized_clean_vae_latent",
        "frame_contract": "exact81_latent21",
        "artifact_role": NATIVE_T2V_ARTIFACT_ROLE,
        "source": NATIVE_T2V_ARTIFACT_ORIGIN,
    }
    if metadata != expected_metadata:
        raise ActionGuidanceSelfDistillError(
            "native T2V proposal embedded artifact role/coordinate differs"
        )
    tensor = header.get("normalized_clean_latent")
    shape = tensor.get("shape") if isinstance(tensor, Mapping) else None
    offsets = tensor.get("data_offsets") if isinstance(tensor, Mapping) else None
    if (
        not isinstance(shape, list)
        or len(shape) != 5
        or shape[:3] != [1, 16, 21]
        or any(type(item) is not int or item <= 0 for item in shape)
        or shape[3] % 2
        or shape[4] % 2
        or tensor.get("dtype") != "F32"
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or offsets[0] != 0
    ):
        raise ActionGuidanceSelfDistillError(
            "native T2V proposal must be exact81 FP32 normalized latent storage"
        )
    element_count = math.prod(int(item) for item in shape)
    if offsets[1] != element_count * 4:
        raise ActionGuidanceSelfDistillError(
            "native T2V proposal safetensors byte extent differs"
        )
    if path.stat().st_size != 8 + header_size + int(offsets[1]):
        raise ActionGuidanceSelfDistillError(
            "native T2V proposal safetensors contains trailing/truncated bytes"
        )
    if (
        receipt_artifact.get("shape") != shape
        or receipt_artifact.get("stored_dtype") != "torch.float32"
        or receipt_artifact.get("tensor_key") != "normalized_clean_latent"
        or receipt_artifact.get("coordinate")
        != "bernini_normalized_clean_vae_latent"
    ):
        raise ActionGuidanceSelfDistillError(
            "native T2V proposal receipt and safetensors geometry differ"
        )


def _reject_paired_target_claims(value: Any, *, path: str = "receipt") -> None:
    forbidden_flags = {
        "target_video",
        "paired_target",
        "paired_target_accessed",
        "target_media_accessed",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in forbidden_flags and item is not False:
                raise ActionGuidanceSelfDistillError(
                    f"native T2V proposal has non-false target access flag: {path}.{key}"
                )
            _reject_paired_target_claims(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_paired_target_claims(item, path=f"{path}[{index}]")


def load_native_t2v_proposal_evidence(
    *,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
    proposal_artifact_path: str | Path,
    expected_proposal_artifact_sha256: str,
    rollout_seed: int,
    action_prompt: str,
    checkpoint_sha256: str,
) -> NativeT2VProposalEvidence:
    """Validate a content-addressed native T2V pre-decode proposal receipt."""

    if (
        not isinstance(action_prompt, str)
        or not action_prompt.strip()
        or "\x00" in action_prompt
    ):
        raise ActionGuidanceSelfDistillError(
            "native T2V action prompt must be non-empty UTF-8 text"
        )
    if (
        isinstance(rollout_seed, bool)
        or not isinstance(rollout_seed, int)
        or not 0 <= rollout_seed < 2**63
    ):
        raise ActionGuidanceSelfDistillError(
            "native T2V rollout seed must be in [0,2^63)"
        )
    checkpoint = _require_sha256(
        checkpoint_sha256, label="native T2V checkpoint SHA-256"
    )
    expected_artifact_sha = _require_sha256(
        expected_proposal_artifact_sha256,
        label="native T2V proposal artifact SHA-256",
    )
    receipt, receipt_file, receipt_sha, receipt_digest = (
        _load_content_addressed_receipt(
            receipt_path, expected_sha256=expected_receipt_sha256
        )
    )
    if (
        receipt.get("schema_version") != NATIVE_T2V_RECEIPT_SCHEMA
        or receipt.get("method") != NATIVE_T2V_RECEIPT_METHOD
        or receipt.get("arms") != [QUERY_MODE]
        or receipt.get("experimental_canary") is not True
        or receipt.get("production_claim_forbidden") is not True
        or receipt.get("scientific_claim_authorized") is not False
    ):
        raise ActionGuidanceSelfDistillError(
            "native proposal receipt is not the exact T2V canary schema/arm"
        )
    _reject_paired_target_claims(receipt)
    prompt_sha = hashlib.sha256(action_prompt.encode("utf-8")).hexdigest()
    inputs = receipt.get("input")
    checkpoint_receipt = receipt.get("checkpoint")
    checkpoint_content = (
        checkpoint_receipt.get("content")
        if isinstance(checkpoint_receipt, Mapping)
        else None
    )
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("action_prompt_utf8_sha256") != prompt_sha
        or inputs.get("target_video") is not False
        or inputs.get("paired_target", False) is not False
        or inputs.get("accepted_external_conditions")
        != ["source_video", "action_prompt"]
        or not isinstance(checkpoint_receipt, Mapping)
        or checkpoint_receipt.get("tree_sha256") != checkpoint
        or not isinstance(checkpoint_content, Mapping)
        or checkpoint_content.get("every_file_sha256_verified") is not True
        or _SHA256.fullmatch(
            str(checkpoint_content.get("verified_entries_digest"))
        )
        is None
    ):
        raise ActionGuidanceSelfDistillError(
            "native T2V proposal prompt/checkpoint/no-target binding differs"
        )
    sampling = receipt.get("sampling")
    t2v_sampling = sampling.get(QUERY_MODE) if isinstance(sampling, Mapping) else None
    conditioning = receipt.get("conditioning")
    t2v_conditioning = (
        conditioning.get(QUERY_MODE) if isinstance(conditioning, Mapping) else None
    )
    if (
        not isinstance(t2v_sampling, Mapping)
        or t2v_sampling.get("seed") != rollout_seed
        or t2v_sampling.get("guidance_mode") != "t2v_apg"
        or t2v_sampling.get("target_initialization")
        != "official_gen_wanx22_fresh_gaussian"
        or t2v_sampling.get("target_mixed_with_source_latent") is not False
        or t2v_sampling.get("custom_sampler_or_scheduler") is not False
        or not isinstance(t2v_conditioning, Mapping)
        or t2v_conditioning.get("full_source_video_count") != 0
        or t2v_conditioning.get("source_derived_reference_count") != 0
        or t2v_conditioning.get("source_frame_indices") != []
        or t2v_conditioning.get("reference_encoding") != "none"
    ):
        raise ActionGuidanceSelfDistillError(
            "native T2V proposal sampling/zero-source-conditioning differs"
        )
    outputs = receipt.get("outputs")
    output = outputs.get(QUERY_MODE) if isinstance(outputs, Mapping) else None
    artifact = (
        output.get("normalized_clean_latent")
        if isinstance(output, Mapping)
        else None
    )
    if not isinstance(artifact, Mapping):
        raise ActionGuidanceSelfDistillError(
            "native T2V receipt lacks its pre-decode proposal artifact"
        )
    artifact_path = _plain_absolute_file(
        proposal_artifact_path, label="native T2V proposal artifact"
    )
    try:
        receipt_artifact_path = _plain_absolute_file(
            str(artifact.get("path", "")),
            label="receipt-declared native T2V proposal artifact",
        )
    except (TypeError, ValueError) as error:
        raise ActionGuidanceSelfDistillError(
            f"native T2V proposal artifact path is invalid: {error}"
        ) from error
    actual_artifact_sha = _file_sha256(artifact_path)
    if (
        receipt_artifact_path != artifact_path
        or actual_artifact_sha != expected_artifact_sha
        or artifact.get("sha256") != expected_artifact_sha
        or artifact.get("artifact_role") != NATIVE_T2V_ARTIFACT_ROLE
        or artifact.get("origin") != NATIVE_T2V_ARTIFACT_ORIGIN
        or artifact.get("native_sampler_before_vae_decode") is not True
        or artifact.get("source_video_vae_encode_before_any_decode") is not False
        or artifact.get("mp4_decode_reencode_used") is not False
        or artifact.get("roundtrip_byte_exact_fp32") is not True
    ):
        raise ActionGuidanceSelfDistillError(
            "native T2V pre-decode proposal artifact provenance differs"
        )
    _validate_safetensors_proposal_header(
        artifact_path, receipt_artifact=artifact
    )
    values = {
        "receipt_path": str(receipt_file),
        "receipt_file_sha256": receipt_sha,
        "receipt_digest": receipt_digest,
        "receipt_schema": NATIVE_T2V_RECEIPT_SCHEMA,
        "arm": QUERY_MODE,
        "proposal_artifact_path": str(artifact_path),
        "proposal_artifact_sha256": actual_artifact_sha,
        "proposal_artifact_role": NATIVE_T2V_ARTIFACT_ROLE,
        "proposal_predecode": True,
        "rollout_seed": rollout_seed,
        "action_prompt_sha256": prompt_sha,
        "checkpoint_sha256": checkpoint,
        "target_video": False,
        "paired_target": False,
    }
    evidence = NativeT2VProposalEvidence(
        **values,
        validation_digest=hashlib.sha256(_canonical_json(values)).hexdigest(),
    )
    evidence.validate()
    return evidence


def _condition_receipt(value: Any, *, path: str = "condition") -> Any:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all().item()):
            raise ActionGuidanceSelfDistillError(f"{path} contains non-finite tensor")
        return {
            "kind": "tensor",
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
            "sha256": tensor_sha256(value),
        }
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ActionGuidanceSelfDistillError(f"{path} is non-finite")
        return value
    if isinstance(value, Mapping):
        receipt: dict[str, Any] = {}
        keys = list(value)
        if not all(
            isinstance(key, str) and bool(key) and "\x00" not in key
            for key in keys
        ):
            raise ActionGuidanceSelfDistillError(
                f"{path} mapping keys must be non-empty text"
            )
        for key in sorted(keys):
            if _FORBIDDEN_CONDITION_KEY.search(key):
                raise ActionGuidanceSelfDistillError(
                    f"{path} contains forbidden teacher pixel/clean target field: {key}"
                )
            receipt[key] = _condition_receipt(value[key], path=f"{path}.{key}")
        return receipt
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _condition_receipt(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ActionGuidanceSelfDistillError(
        f"{path} must be tensor/JSON-like conditioning, not {type(value).__name__}"
    )


def condition_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(_condition_receipt(value))).hexdigest()


def bind_provenance(
    *,
    noised_state: Any,
    action_condition: Any,
    noop_condition: Any,
    proposal_evidence: NativeT2VProposalEvidence,
    proposal_iid: str,
) -> DistillProvenance:
    """Create the only provenance object accepted by the forward bridge."""

    if not isinstance(proposal_evidence, NativeT2VProposalEvidence):
        raise ActionGuidanceSelfDistillError(
            "bind_provenance requires loader-validated native T2V proposal evidence"
        )
    proposal_evidence.validate()
    provenance = DistillProvenance(
        teacher_checkpoint_sha256=proposal_evidence.checkpoint_sha256,
        student_base_checkpoint_sha256=proposal_evidence.checkpoint_sha256,
        proposal_sha256=proposal_evidence.proposal_artifact_sha256,
        proposal_iid=proposal_iid,
        rollout_seed=proposal_evidence.rollout_seed,
        query_state_sha256=tensor_sha256(noised_state),
        action_condition_sha256=condition_sha256(action_condition),
        noop_condition_sha256=condition_sha256(noop_condition),
        proposal_receipt_sha256=proposal_evidence.receipt_file_sha256,
        proposal_receipt_digest=proposal_evidence.receipt_digest,
        proposal_prompt_sha256=proposal_evidence.action_prompt_sha256,
        proposal_evidence=proposal_evidence,
    )
    provenance.validate()
    return provenance


def bind_forward_response(
    request: ModelForwardRequest, velocity: Any
) -> ModelForwardResponse:
    """Bind a callback's actual model velocity to its immutable request."""

    return ModelForwardResponse(
        velocity=velocity,
        branch=request.branch,
        mode=request.mode,
        adapter_enabled_observed=request.adapter_enabled,
        checkpoint_sha256=request.checkpoint_sha256,
        proposal_sha256=request.proposal_sha256,
        rollout_seed=request.rollout_seed,
        query_state_sha256=request.query_state_sha256,
        timestep_sha256=request.timestep_sha256,
        model_forward_executed=True,
    )


def sigma_gate(
    sigma: Real, config: DistillConfig = DistillConfig()
) -> tuple[str, float]:
    config.validate()
    if isinstance(sigma, bool) or not isinstance(sigma, Real):
        raise ActionGuidanceSelfDistillError("sigma must be a real scalar")
    value = float(sigma)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ActionGuidanceSelfDistillError("sigma must lie in (0,1)")
    if value >= config.high_sigma_min:
        return "high", 1.0
    if value >= config.mid_sigma_min:
        return "mid", float(config.mid_sigma_weight)
    return "low_ineligible", 0.0


def _is_allowed_action_lora_name(name: str) -> bool:
    lowered = name.lower()
    action_marker = (
        "action_lora" in lowered
        or ".lora_a.action." in lowered
        or ".lora_b.action." in lowered
        or ".lora_a.action_" in lowered
        or ".lora_b.action_" in lowered
    )
    projection = (
        ".attn2.to_q." in lowered
        or ".attn2.to_out.0." in lowered
        or ".attn2.to_out." in lowered
    )
    return "lora" in lowered and action_marker and projection


def validate_action_lora_scope(model: Any) -> dict[str, Any]:
    """Require that every trainable parameter is attn2 Q/O Action-LoRA."""

    torch = _torch()
    if not isinstance(model, torch.nn.Module):
        raise ActionGuidanceSelfDistillError("model must be a torch.nn.Module")
    if model.training:
        raise ActionGuidanceSelfDistillError(
            "model must be in eval mode for same-state deterministic distillation"
        )
    entries = list(model.named_parameters())
    if not entries:
        raise ActionGuidanceSelfDistillError("model exposes no parameters")
    trainable = [(name, parameter) for name, parameter in entries if parameter.requires_grad]
    if not trainable:
        raise ActionGuidanceSelfDistillError("model has no trainable Action-LoRA")
    forbidden = [name for name, _ in trainable if not _is_allowed_action_lora_name(name)]
    if forbidden:
        raise ActionGuidanceSelfDistillError(
            f"trainable scope is not attn2 Q/O Action-LoRA: {forbidden[:4]}"
        )
    if any(not parameter.is_floating_point() for _, parameter in trainable):
        raise ActionGuidanceSelfDistillError("Action-LoRA parameters must be floating")
    return {
        "allowed_parameter_names": tuple(name for name, _ in trainable),
        "allowed_parameter_ids": {name: id(parameter) for name, parameter in trainable},
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "all_non_lora_parameters_frozen": True,
    }


def _validate_query(noised_state: Any, timestep: Any, sigma: Real) -> float:
    torch = _torch()
    if (
        not isinstance(noised_state, torch.Tensor)
        or not noised_state.is_floating_point()
        or noised_state.ndim < 2
        or int(noised_state.shape[0]) <= 0
        or noised_state.requires_grad
        or not bool(torch.isfinite(noised_state).all().item())
    ):
        raise ActionGuidanceSelfDistillError(
            "noised_state must be one finite graph-free floating batch tensor"
        )
    if (
        not isinstance(timestep, torch.Tensor)
        or timestep.dtype != torch.float32
        or timestep.device.type != "cpu"
        or timestep.numel() != 1
        or timestep.requires_grad
        or not bool(torch.isfinite(timestep).all().item())
    ):
        raise ActionGuidanceSelfDistillError(
            "timestep must be one finite graph-free CPU FP32 scalar"
        )
    stratum, gate = sigma_gate(sigma)
    expected = torch.tensor(float(sigma) * 1000.0, dtype=torch.float32).item()
    if float(timestep.item()) != expected:
        raise ActionGuidanceSelfDistillError(
            "timestep must be the exact FP32 1000*sigma query coordinate"
        )
    if gate == 0.0 or stratum == "low_ineligible":
        raise ActionGuidanceSelfDistillError(
            "low-sigma query is ineligible for action self-distillation"
        )
    return float(sigma)


def _rms(value: Any) -> Any:
    return value.square().mean(dim=tuple(range(1, value.ndim))).sqrt()


def _view_batch(value: Any, reference: Any) -> Any:
    return value.reshape((int(reference.shape[0]),) + (1,) * (reference.ndim - 1))


def _bounded_teacher_residual(
    action_velocity: Any,
    noop_velocity: Any,
    config: DistillConfig,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    action = action_velocity.detach().to(dtype=torch.float32)
    noop = noop_velocity.detach().to(dtype=torch.float32)
    raw = action - noop
    raw_l2 = raw.flatten(1).norm(dim=1)
    clip_scale = torch.minimum(
        torch.ones_like(raw_l2),
        torch.full_like(raw_l2, config.raw_residual_l2_cap)
        / raw_l2.clamp_min(config.epsilon),
    )
    clipped = raw * _view_batch(clip_scale, raw)
    action_flat = action.flatten(1)
    direction = action_flat / action_flat.norm(dim=1, keepdim=True).clamp_min(
        config.epsilon
    )
    clipped_flat = clipped.flatten(1)
    parallel_flat = (clipped_flat * direction).sum(dim=1, keepdim=True) * direction
    parallel = parallel_flat.reshape_as(clipped)
    perpendicular = clipped - parallel
    apg = config.guidance_scale * (
        perpendicular + config.apg_parallel_eta * parallel
    )
    reference_rms = torch.maximum(_rms(noop), _rms(raw)).clamp_min(config.epsilon)
    teacher_cap = config.teacher_max_reference_rms_ratio * reference_rms
    apg_rms = _rms(apg)
    bound_scale = torch.minimum(
        torch.ones_like(apg_rms), teacher_cap / apg_rms.clamp_min(config.epsilon)
    )
    teacher = (apg * _view_batch(bound_scale, apg)).detach()
    teacher_rms = _rms(teacher)
    if (
        teacher.dtype != torch.float32
        or teacher.requires_grad
        or teacher.grad_fn is not None
        or not bool(torch.isfinite(teacher).all().item())
        or bool((teacher_rms <= config.epsilon).any().item())
        or bool((teacher_rms > teacher_cap + 2.0e-6).any().item())
    ):
        raise ActionGuidanceSelfDistillError(
            "detached bounded teacher action residual is invalid"
        )
    return teacher, {
        "raw_residual_rms": [float(item) for item in _rms(raw).cpu().tolist()],
        "teacher_residual_rms": [float(item) for item in teacher_rms.cpu().tolist()],
        "teacher_rms_cap": [float(item) for item in teacher_cap.cpu().tolist()],
        "raw_l2_clip_active": bool((clip_scale < 1.0).any().item()),
        "teacher_reference_bound_active": bool((bound_scale < 1.0).any().item()),
    }


def _validate_response(
    response: Any,
    request: ModelForwardRequest,
    *,
    student: bool,
) -> Any:
    torch = _torch()
    if not isinstance(response, ModelForwardResponse):
        raise ActionGuidanceSelfDistillError(
            "forward callback must return a bound ModelForwardResponse, not a scalar/tensor"
        )
    if (
        response.branch != request.branch
        or response.mode != QUERY_MODE
        or request.mode != QUERY_MODE
        or response.adapter_enabled_observed is not request.adapter_enabled
        or response.checkpoint_sha256 != request.checkpoint_sha256
        or response.proposal_sha256 != request.proposal_sha256
        or response.rollout_seed != request.rollout_seed
        or response.query_state_sha256 != request.query_state_sha256
        or response.timestep_sha256 != request.timestep_sha256
        or response.model_forward_executed is not True
    ):
        raise ActionGuidanceSelfDistillError(
            "model forward response provenance differs from its request"
        )
    velocity = response.velocity
    if (
        not isinstance(velocity, torch.Tensor)
        or tuple(velocity.shape) != tuple(request.noised_state.shape)
        or not velocity.is_floating_point()
        or velocity.device != request.noised_state.device
        or not bool(torch.isfinite(velocity).all().item())
    ):
        raise ActionGuidanceSelfDistillError(
            "model callback must return one full finite velocity matching the query state"
        )
    if student:
        if not velocity.requires_grad or velocity.grad_fn is None or velocity.is_leaf:
            raise ActionGuidanceSelfDistillError(
                "student callback velocity must be a non-leaf model-forward tensor"
            )
    elif velocity.requires_grad or velocity.grad_fn is not None:
        raise ActionGuidanceSelfDistillError(
            "frozen teacher/base velocity must be graph-free"
        )
    return velocity


def _call_model_bridge(
    *,
    model: Any,
    callback: Callable[[Any, ModelForwardRequest], ModelForwardResponse],
    request: ModelForwardRequest,
    student: bool,
) -> Any:
    torch = _torch()
    if not callable(callback):
        raise ActionGuidanceSelfDistillError("forward_callback must be callable")
    before = tensor_sha256(request.noised_state)
    if before != request.query_state_sha256:
        raise ActionGuidanceSelfDistillError("query state changed before model forward")
    observed_calls: list[dict[str, bool]] = []

    def contains_identity(value: Any, expected: Any) -> bool:
        if value is expected:
            return True
        if isinstance(value, Mapping):
            return any(contains_identity(item, expected) for item in value.values())
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return any(contains_identity(item, expected) for item in value)
        return False

    def observe_forward(_module: Any, args: Any) -> None:
        observed_calls.append(
            {
                "state_alias": contains_identity(args, request.noised_state),
                "timestep_alias": contains_identity(args, request.timestep),
                "condition_alias": contains_identity(args, request.condition),
            }
        )

    try:
        # Positional pre-hooks work on the pinned AUH Torch as well as current
        # Torch.  Adapter state is independently bound in ModelForwardResponse.
        hook = model.register_forward_pre_hook(observe_forward)
    except (AttributeError, TypeError) as error:
        raise ActionGuidanceSelfDistillError(
            "model must support an instrumented forward callback bridge"
        ) from error
    try:
        if student:
            with torch.enable_grad():
                response = callback(model, request)
        else:
            with torch.no_grad():
                response = callback(model, request)
    finally:
        hook.remove()
    if len(observed_calls) != 1 or not all(observed_calls[0].values()):
        raise ActionGuidanceSelfDistillError(
            "callback must execute exactly one model forward with the bound "
            "state/timestep/condition"
        )
    after = tensor_sha256(request.noised_state)
    if after != before:
        raise ActionGuidanceSelfDistillError("model callback mutated the shared query state")
    return _validate_response(response, request, student=student)


def _request(
    *,
    branch: str,
    state: Any,
    timestep: Any,
    condition: Any,
    condition_digest: str,
    adapter_enabled: bool,
    provenance: DistillProvenance,
) -> ModelForwardRequest:
    return ModelForwardRequest(
        branch=branch,
        mode=QUERY_MODE,
        noised_state=state,
        timestep=timestep,
        condition=condition,
        condition_sha256=condition_digest,
        adapter_enabled=adapter_enabled,
        checkpoint_sha256=provenance.teacher_checkpoint_sha256,
        proposal_sha256=provenance.proposal_sha256,
        proposal_iid=provenance.proposal_iid,
        rollout_seed=provenance.rollout_seed,
        query_state_sha256=provenance.query_state_sha256,
        timestep_sha256=tensor_sha256(timestep),
    )


def build_distill_forward(
    *,
    model: Any,
    forward_callback: Callable[[Any, ModelForwardRequest], ModelForwardResponse],
    noised_state: Any,
    timestep: Any,
    sigma: Real,
    action_condition: Any,
    noop_condition: Any,
    provenance: DistillProvenance,
    config: DistillConfig = DistillConfig(),
) -> DistillForwardResult:
    """Run the three real model forwards and build one FP32 distillation loss."""

    torch = _torch()
    config.validate()
    sigma_value = _validate_query(noised_state, timestep, sigma)
    stratum, gate_weight = sigma_gate(sigma_value, config)
    provenance.validate()
    model_checkpoint_sha256 = getattr(model, CHECKPOINT_BINDING_ATTRIBUTE, None)
    if model_checkpoint_sha256 != provenance.student_base_checkpoint_sha256:
        raise ActionGuidanceSelfDistillError(
            "runtime model checkpoint binding differs from teacher/base provenance"
        )
    actual_state_sha = tensor_sha256(noised_state)
    action_sha = condition_sha256(action_condition)
    noop_sha = condition_sha256(noop_condition)
    if (
        actual_state_sha != provenance.query_state_sha256
        or action_sha != provenance.action_condition_sha256
        or noop_sha != provenance.noop_condition_sha256
    ):
        raise ActionGuidanceSelfDistillError(
            "query state or condition differs from bound provenance"
        )
    scope = validate_action_lora_scope(model)
    stale = [name for name, parameter in model.named_parameters() if parameter.grad is not None]
    if stale:
        raise ActionGuidanceSelfDistillError(
            f"model has stale gradients before distillation: {stale[:4]}"
        )
    requests = (
        _request(
            branch=TEACHER_ACTION_BRANCH,
            state=noised_state,
            timestep=timestep,
            condition=action_condition,
            condition_digest=action_sha,
            adapter_enabled=False,
            provenance=provenance,
        ),
        _request(
            branch=TEACHER_NOOP_BRANCH,
            state=noised_state,
            timestep=timestep,
            condition=noop_condition,
            condition_digest=noop_sha,
            adapter_enabled=False,
            provenance=provenance,
        ),
        _request(
            branch=STUDENT_ACTION_BRANCH,
            state=noised_state,
            timestep=timestep,
            condition=action_condition,
            condition_digest=action_sha,
            adapter_enabled=True,
            provenance=provenance,
        ),
    )
    frozen_action = _call_model_bridge(
        model=model, callback=forward_callback, request=requests[0], student=False
    )
    frozen_noop = _call_model_bridge(
        model=model, callback=forward_callback, request=requests[1], student=False
    )
    student_action = _call_model_bridge(
        model=model, callback=forward_callback, request=requests[2], student=True
    )
    teacher_residual, teacher_audit = _bounded_teacher_residual(
        frozen_action, frozen_noop, config
    )
    frozen_action_fp32 = frozen_action.detach().to(dtype=torch.float32)
    frozen_noop_fp32 = frozen_noop.detach().to(dtype=torch.float32)
    student_action_fp32 = student_action.to(dtype=torch.float32)
    raw_student_correction = student_action_fp32 - frozen_action_fp32
    base_reference_rms = torch.maximum(
        _rms(frozen_action_fp32),
        _rms(frozen_action_fp32 - frozen_noop_fp32),
    ).clamp_min(config.epsilon)
    trust_radius = config.student_base_trust_ratio * base_reference_rms
    # ``sqrt(mean(x^2))`` has an undefined derivative at an exactly zero LoRA
    # correction.  Clamp only this trainable trust-radius statistic so a
    # legitimate zero-initialized adapter produces finite zero gradients.
    correction_rms = raw_student_correction.square().mean(
        dim=tuple(range(1, raw_student_correction.ndim))
    ).clamp_min(config.epsilon * config.epsilon).sqrt()
    trust_scale = torch.minimum(
        torch.ones_like(correction_rms),
        trust_radius / correction_rms.clamp_min(config.epsilon),
    )
    trusted_correction = raw_student_correction * _view_batch(
        trust_scale, raw_student_correction
    )
    trusted_student_velocity = frozen_action_fp32 + trusted_correction
    student_residual = trusted_student_velocity - frozen_noop_fp32
    distill_energy = (student_residual - teacher_residual).square().mean()
    trust_excess = torch.relu(correction_rms - trust_radius)
    trust_penalty = trust_excess.square().mean()
    loss = gate_weight * distill_energy + config.trust_penalty_weight * trust_penalty
    if (
        loss.dtype != torch.float32
        or distill_energy.dtype != torch.float32
        or trust_penalty.dtype != torch.float32
        or not loss.requires_grad
        or loss.grad_fn is None
        or loss.is_leaf
        or not bool(torch.isfinite(loss).item())
    ):
        raise ActionGuidanceSelfDistillError(
            "distillation must produce one finite non-leaf FP32 energy"
        )
    trusted_rms = _rms(trusted_correction)
    if bool((trusted_rms > trust_radius + 2.0e-6).any().item()):
        raise ActionGuidanceSelfDistillError("student correction escaped base trust region")
    callback_name = getattr(forward_callback, "__qualname__", type(forward_callback).__name__)
    diagnostics = {
        "teacher": teacher_audit,
        "student_raw_correction_rms": [
            float(item) for item in correction_rms.detach().cpu().tolist()
        ],
        "student_trusted_correction_rms": [
            float(item) for item in trusted_rms.detach().cpu().tolist()
        ],
        "student_base_trust_radius": [
            float(item) for item in trust_radius.detach().cpu().tolist()
        ],
        "student_trust_clip_active": bool((trust_scale < 1.0).any().item()),
        "distill_energy_fp32": float(distill_energy.detach().cpu().item()),
        "trust_penalty_fp32": float(trust_penalty.detach().cpu().item()),
        "loss_fp32": float(loss.detach().cpu().item()),
    }
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD_NAME,
        "model_self_generated_prior": True,
        "native_t2v_proposal_evidence_verified": True,
        "proposal_receipt_content_addressed": True,
        "proposal_artifact_predecode_role_verified": True,
        "production_authorized": False,
        "trainer_integration_authorized": False,
        "pixel_target_supervision": False,
        "teacher_rgb_received": False,
        "teacher_clean_latent_received": False,
        "proposal_clean_latent_received": False,
        "teacher_residual_detached": True,
        "teacher": {
            "branches": [TEACHER_ACTION_BRANCH, TEACHER_NOOP_BRANCH],
            "frozen_adapter_disabled": True,
            "construction": "bounded_apg_cfg_velocity_residual",
            "guidance_scale": config.guidance_scale,
            "parallel_eta": config.apg_parallel_eta,
            "raw_l2_cap": config.raw_residual_l2_cap,
            "max_reference_rms_ratio": config.teacher_max_reference_rms_ratio,
        },
        "query_binding": {
            "mode": QUERY_MODE,
            "same_noised_state_object_all_forwards": True,
            "same_timestep_object_all_forwards": True,
            "same_rollout_seed_all_forwards": True,
            "cross_state_vector_forbidden": True,
            "cross_seed_vector_forbidden": True,
            "query_state_sha256": actual_state_sha,
            "timestep_sha256": tensor_sha256(timestep),
        },
        "provenance": asdict(provenance),
        "runtime_model_checkpoint_sha256": model_checkpoint_sha256,
        "forward_bridge": {
            "callback": callback_name,
            "real_model_forward_required": True,
            "bound_response_required": True,
            "exactly_one_instrumented_model_forward_per_branch": True,
            "state_timestep_condition_object_alias_verified": True,
            "call_count": 3,
            "branch_order": list(FORWARD_BRANCHES),
        },
        "student_scope": {
            "only_attn2_q_out_action_lora": True,
            "allowed_parameter_names": list(scope["allowed_parameter_names"]),
            "trainable_parameter_count": scope["trainable_parameter_count"],
            "base_frozen": True,
        },
        "sigma_gate": {
            "sigma": sigma_value,
            "stratum": stratum,
            "weight": gate_weight,
            "low_sigma_forbidden": True,
        },
        "base_trust_region": {
            "ratio": config.student_base_trust_ratio,
            "hard_rms_clip": True,
            "excess_penalty_weight": config.trust_penalty_weight,
        },
        "energy": {
            "dtype": "torch.float32",
            "distill": "MSE(trusted_student_action_minus_noop,detached_teacher_residual)",
            "finite": True,
        },
        "backward_gradient_audit_required": True,
        "diagnostics": diagnostics,
    }
    return DistillForwardResult(
        loss=loss,
        distill_energy=distill_energy,
        trust_penalty=trust_penalty,
        teacher_residual=teacher_residual,
        student_residual=student_residual,
        raw_student_correction=raw_student_correction,
        trusted_student_correction=trusted_correction,
        sigma=sigma_value,
        sigma_stratum=stratum,
        sigma_gate_weight=gate_weight,
        model_object_id=id(model),
        allowed_parameter_ids=dict(scope["allowed_parameter_ids"]),
        diagnostics=diagnostics,
        receipt=receipt,
    )


def backward_and_audit(
    model: Any, forward: DistillForwardResult
) -> GradientAudit:
    """Backpropagate the bridge loss and audit the exact Action-LoRA scope."""

    torch = _torch()
    if not isinstance(forward, DistillForwardResult):
        raise ActionGuidanceSelfDistillError(
            "backward audit requires a DistillForwardResult from the model bridge"
        )
    if id(model) != forward.model_object_id:
        raise ActionGuidanceSelfDistillError("backward model differs from forward model")
    scope = validate_action_lora_scope(model)
    if dict(scope["allowed_parameter_ids"]) != dict(forward.allowed_parameter_ids):
        raise ActionGuidanceSelfDistillError(
            "Action-LoRA parameter identity changed after forward"
        )
    if (
        not isinstance(forward.loss, torch.Tensor)
        or forward.loss.dtype != torch.float32
        or not forward.loss.requires_grad
        or forward.loss.grad_fn is None
        or forward.loss.is_leaf
    ):
        raise ActionGuidanceSelfDistillError(
            "backward refuses arbitrary scalar/leaf loss"
        )
    if forward.teacher_residual.requires_grad or forward.teacher_residual.grad_fn is not None:
        raise ActionGuidanceSelfDistillError("teacher residual is not detached")
    if any(parameter.grad is not None for _, parameter in model.named_parameters()):
        raise ActionGuidanceSelfDistillError("gradients appeared before bridge backward")
    forward.loss.backward()
    allowed_names = tuple(scope["allowed_parameter_names"])
    allowed_set = set(allowed_names)
    finite_names: list[str] = []
    nonzero_names: list[str] = []
    energies: dict[str, float] = {}
    forbidden: list[str] = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if name not in allowed_set:
            if gradient is not None:
                forbidden.append(name)
            continue
        if gradient is None:
            raise ActionGuidanceSelfDistillError(
                f"allowed Action-LoRA parameter has no gradient: {name}"
            )
        gradient_fp32 = gradient.detach().to(dtype=torch.float32)
        if not bool(torch.isfinite(gradient_fp32).all().item()):
            raise ActionGuidanceSelfDistillError(
                f"Action-LoRA gradient is non-finite: {name}"
            )
        finite_names.append(name)
        energy = float(gradient_fp32.square().sum().cpu().item())
        energies[name] = energy
        if energy > 0.0:
            nonzero_names.append(name)
    total_energy = float(sum(energies.values()))
    if forbidden:
        raise ActionGuidanceSelfDistillError(
            f"forbidden parameter received gradients: {forbidden[:4]}"
        )
    if not nonzero_names or not math.isfinite(total_energy) or total_energy <= 0.0:
        raise ActionGuidanceSelfDistillError(
            "allowed Action-LoRA aggregate gradient energy is zero/non-finite"
        )
    return GradientAudit(
        allowed_parameter_names=allowed_names,
        finite_gradient_names=tuple(finite_names),
        nonzero_gradient_names=tuple(nonzero_names),
        per_parameter_fp32_energy=energies,
        total_fp32_gradient_energy=total_energy,
        forbidden_parameter_gradients=tuple(forbidden),
        passed=True,
    )


def run_action_guidance_self_distill(**kwargs: Any) -> DistillStepResult:
    """Run real forwards, backward, and the mandatory allowed-gradient audit."""

    model = kwargs.get("model")
    forward = build_distill_forward(**kwargs)
    audit = backward_and_audit(model, forward)
    receipt = dict(forward.receipt)
    receipt["gradient_audit"] = audit.as_dict()
    receipt["backward_gradient_audit_passed"] = True
    return DistillStepResult(
        forward=forward,
        gradient_audit=audit,
        receipt=receipt,
    )


__all__ = [
    "ActionGuidanceSelfDistillError",
    "DistillConfig",
    "DistillForwardResult",
    "DistillProvenance",
    "DistillStepResult",
    "FORWARD_BRANCHES",
    "GradientAudit",
    "METHOD_NAME",
    "NATIVE_T2V_ARTIFACT_ROLE",
    "NATIVE_T2V_RECEIPT_SCHEMA",
    "NativeT2VProposalEvidence",
    "ModelForwardRequest",
    "ModelForwardResponse",
    "RECEIPT_SCHEMA",
    "QUERY_MODE",
    "CHECKPOINT_BINDING_ATTRIBUTE",
    "STUDENT_ACTION_BRANCH",
    "TEACHER_ACTION_BRANCH",
    "TEACHER_NOOP_BRANCH",
    "backward_and_audit",
    "bind_forward_response",
    "bind_provenance",
    "build_distill_forward",
    "condition_sha256",
    "load_native_t2v_proposal_evidence",
    "run_action_guidance_self_distill",
    "sigma_gate",
    "tensor_sha256",
    "validate_action_lora_scope",
]
