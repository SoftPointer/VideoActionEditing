"""Candidate-own full-source binding energy for PAIR-v5.

This module is a frozen evaluator, never a student conditioning path.  For one
clean exact-81 RV2V candidate it constructs a single rectified-flow state

``x_sigma = (1 - sigma) * candidate + sigma * epsilon``

and asks frozen Bernini to explain that state under four counterfactual cells:

``correct``
    Native RV2V-4 with the candidate's sealed source video, four sealed source
    references, and the requested action text.
``wrong_source``
    The same native query with a sealed, source-disjoint video/reference set.
``reference_off``
    The native video-only branch with action text.  No synthetic zero image,
    mask, or non-native source-id is introduced.
``noop``
    Correct source video/references with a registered no-op text condition.

All cells share the same ``x_sigma``, physical sigma, timestep, frozen model,
and target velocity ``epsilon - candidate``.  Lower target-velocity MSE under
``correct`` than under every counterfactual yields a positive source-binding
margin.  This does not isolate actor identity: identity, background, camera,
objects, and old motion remain entangled in the native source condition.
Camera, background, and quality scores are accepted only after the
model calls as detached scalar outputs of post-video evaluators; they are not
used in a model condition or in the source-binding energy.

The public API requires typed content-bound source bundles and a strictly
positive preregistered margin; exact source copies and ties are rejected.  It
deliberately has no T2V proposal, donor, paired target, mask,
flow, pose, track, or trajectory input.  The resulting score packet is for
safe-Pareto candidate selection only.  It is not RGB/latent supervision and
must never be fed to the student.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import re
import struct
from typing import Any, Mapping, Sequence

import torch
from torch import nn

import dclr_runtime_contract as runtime_contract
import pair_v5_native_bridge as native_bridge
import source_self_native_ref_contrastive_v3 as native
import source_self_native_rv2v_guidance as guidance


SCHEMA_VERSION = "bernini-pair-v5-source-binding-energy-v2"
SCORER_SCHEMA = "bernini-pair-v5-frozen-rv2v4-source-binding-scorer-v2"
RECEIPT_SCHEMA = "bernini-pair-v5-source-binding-energy-receipt-v2"
SOURCE_PROVENANCE_SCHEMA = "bernini-pair-v5-source-condition-provenance-v1"
SOURCE_POLICY_SCHEMA = "bernini-pair-v5-source-binding-preregistration-v1"
SAFE_PARETO_PACKET_SCHEMA = "bernini-pair-v5-source-binding-safe-pareto-packet-v1"
FRAME_COUNT = 81
LATENT_CHANNELS = 16
LATENT_PHASES = 21
REFERENCE_COUNT = 4
REFERENCE_FRAME_INDICES = (0, 27, 53, 80)

CELL_ORDER = ("correct", "wrong_source", "reference_off", "noop")
COUNTERFACTUAL_CELL_ORDER = CELL_ORDER[1:]
POSTVIDEO_SCORE_ORDER = ("camera", "background", "quality")
SOURCE_BINDING_SCORE_SEMANTICS = (
    "unit_interval_full_source_condition_binding_proxy_not_pure_actor_identity"
)

FORBIDDEN_PUBLIC_INPUT_NAMES = frozenset(
    {
        "t2v",
        "t2v_video",
        "t2v_latent",
        "proposal",
        "proposal_video",
        "proposal_latent",
        "proposal_noise",
        "donor",
        "donor_video",
        "donor_latent",
        "paired_target",
        "target",
        "target_video",
        "target_latent",
        "mask",
        "motion_mask",
        "flow",
        "optical_flow",
        "pose",
        "track",
        "tracks",
        "trajectory",
        "trajectories",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SOURCE_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")


class PairV5SourceIdentityEnergyError(ValueError):
    """An identity-energy query violates the sealed PAIR-v5 contract."""


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
        raise PairV5SourceIdentityEnergyError(
            "receipt value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairV5SourceIdentityEnergyError(f"{label} must be lowercase SHA-256")
    return value


def _source_key(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SOURCE_KEY_RE.fullmatch(value) is None:
        raise PairV5SourceIdentityEnergyError(
            f"{label} must be a canonical nonempty source key"
        )
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().to(device="cpu").contiguous().clone()
    metadata = {
        "shape": [int(item) for item in cpu.shape],
        "dtype": str(cpu.dtype),
        "numel": int(cpu.numel()),
    }
    raw = cpu.view(torch.uint8).reshape(-1).numpy().tobytes()
    expected = int(cpu.numel() * cpu.element_size())
    if len(raw) != expected:
        raise PairV5SourceIdentityEnergyError(
            "tensor receipt storage byte count differs"
        )
    digest = hashlib.sha256()
    digest.update(_canonical_json(metadata))
    digest.update(b"\x00")
    digest.update(raw)
    return digest.hexdigest()


def _storage_ptr(value: torch.Tensor) -> int:
    getter = getattr(value, "untyped_storage", None)
    storage = getter() if getter is not None else value.storage()
    return int(storage.data_ptr())


def _fp32_scalar(value: Any, *, label: str, unit_interval: bool) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.numel() != 1
        or value.device.type == "meta"
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise PairV5SourceIdentityEnergyError(
            f"{label} must be one detached finite FP32 scalar"
        )
    result = value.reshape(()).detach()
    if unit_interval and not 0.0 <= float(result.item()) <= 1.0:
        raise PairV5SourceIdentityEnergyError(f"{label} must lie in [0,1]")
    return result


def _fp32_bits(value: torch.Tensor, *, label: str) -> str:
    scalar = _fp32_scalar(value, label=label, unit_interval=False)
    return struct.pack("!f", float(scalar.item())).hex()


def _exact81(value: Any, *, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.ndim != 5
        or tuple(int(item) for item in value.shape[:3])
        != (1, LATENT_CHANNELS, LATENT_PHASES)
        or int(value.shape[3]) <= 0
        or int(value.shape[4]) <= 0
        or int(value.shape[3]) % 2
        or int(value.shape[4]) % 2
        or value.device.type == "meta"
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise PairV5SourceIdentityEnergyError(
            f"{label} must be detached FP32 exact81 [1,16,21,H,W] with even H/W"
        )
    return value


def _references(
    values: Any,
    *,
    source_video: torch.Tensor,
    label: str,
) -> tuple[torch.Tensor, ...]:
    if isinstance(values, (str, bytes)):
        raise PairV5SourceIdentityEnergyError(
            f"{label} must contain exactly four references"
        )
    try:
        refs = tuple(values)
    except TypeError as error:
        raise PairV5SourceIdentityEnergyError(
            f"{label} must contain exactly four references"
        ) from error
    if len(refs) != REFERENCE_COUNT:
        raise PairV5SourceIdentityEnergyError(
            f"{label} must contain exactly four references"
        )
    expected = (
        1,
        LATENT_CHANNELS,
        1,
        int(source_video.shape[3]),
        int(source_video.shape[4]),
    )
    for index, value in enumerate(refs):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or tuple(int(item) for item in value.shape) != expected
            or value.device != source_video.device
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
        ):
            raise PairV5SourceIdentityEnergyError(
                f"{label}[{index}] must be detached FP32 [1,16,1,H,W]"
            )
    return refs


@dataclass(frozen=True)
class SourceConditionProvenance:
    """Content-bound provenance for one deploy-available source condition.

    The full-video tensor and the four independently encoded reference tensors
    are authenticated separately.  ``source_media_artifact_sha256`` binds the
    original source media, while the encoding/extraction receipts bind the
    deterministic conversion into Bernini's latent condition space.
    """

    source_key: str
    source_media_artifact_sha256: str
    source_media_receipt_digest: str
    full_video_tensor_sha256: str
    full_video_encoding_receipt_digest: str
    reference_frame_indices: tuple[int, ...]
    reference_tensor_sha256: tuple[str, ...]
    reference_extraction_receipt_digest: str
    receipt_digest: str

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": SOURCE_PROVENANCE_SCHEMA,
            "source_key": self.source_key,
            "source_media_artifact_sha256": self.source_media_artifact_sha256,
            "source_media_receipt_digest": self.source_media_receipt_digest,
            "full_video_tensor_sha256": self.full_video_tensor_sha256,
            "full_video_encoding_receipt_digest": (
                self.full_video_encoding_receipt_digest
            ),
            "reference_frame_indices": list(self.reference_frame_indices),
            "reference_tensor_sha256": list(self.reference_tensor_sha256),
            "reference_extraction_receipt_digest": (
                self.reference_extraction_receipt_digest
            ),
            "frame_count": FRAME_COUNT,
            "latent_phases": LATENT_PHASES,
            "reference_count": REFERENCE_COUNT,
            "condition_role": "deploy_available_source_video_plus_fixed_references",
        }

    def validate(self) -> None:
        _source_key(self.source_key, label="source provenance source_key")
        for name in (
            "source_media_artifact_sha256",
            "source_media_receipt_digest",
            "full_video_tensor_sha256",
            "full_video_encoding_receipt_digest",
            "reference_extraction_receipt_digest",
            "receipt_digest",
        ):
            _require_sha256(getattr(self, name), label=name)
        if self.reference_frame_indices != REFERENCE_FRAME_INDICES:
            raise PairV5SourceIdentityEnergyError(
                "source provenance reference indices must be exactly [0,27,53,80]"
            )
        if (
            not isinstance(self.reference_tensor_sha256, tuple)
            or len(self.reference_tensor_sha256) != REFERENCE_COUNT
        ):
            raise PairV5SourceIdentityEnergyError(
                "source provenance must bind exactly four reference tensor hashes"
            )
        for index, digest in enumerate(self.reference_tensor_sha256):
            _require_sha256(digest, label=f"reference_tensor_sha256[{index}]")
        if self.receipt_digest != object_sha256(self.payload()):
            raise PairV5SourceIdentityEnergyError(
                "source condition provenance receipt digest differs"
            )


def seal_source_condition_provenance(
    *,
    source_key: str,
    source_media_artifact_sha256: str,
    source_media_receipt_digest: str,
    source_video: torch.Tensor,
    source_references: Sequence[torch.Tensor],
    full_video_encoding_receipt_digest: str,
    reference_extraction_receipt_digest: str,
) -> SourceConditionProvenance:
    """Seal the typed source receipt produced by the trusted media loader."""

    video = _exact81(source_video, label="source provenance full video")
    references = _references(
        source_references,
        source_video=video,
        label="source provenance references",
    )
    key = _source_key(source_key, label="source provenance source_key")
    for label, value in (
        ("source_media_artifact_sha256", source_media_artifact_sha256),
        ("source_media_receipt_digest", source_media_receipt_digest),
        (
            "full_video_encoding_receipt_digest",
            full_video_encoding_receipt_digest,
        ),
        (
            "reference_extraction_receipt_digest",
            reference_extraction_receipt_digest,
        ),
    ):
        _require_sha256(value, label=label)
    provisional = SourceConditionProvenance(
        source_key=key,
        source_media_artifact_sha256=source_media_artifact_sha256,
        source_media_receipt_digest=source_media_receipt_digest,
        full_video_tensor_sha256=_tensor_sha256(video),
        full_video_encoding_receipt_digest=full_video_encoding_receipt_digest,
        reference_frame_indices=REFERENCE_FRAME_INDICES,
        reference_tensor_sha256=tuple(_tensor_sha256(item) for item in references),
        reference_extraction_receipt_digest=reference_extraction_receipt_digest,
        receipt_digest="",
    )
    result = SourceConditionProvenance(
        **{
            **provisional.__dict__,
            "receipt_digest": object_sha256(provisional.payload()),
        }
    )
    result.validate()
    return result


@dataclass(frozen=True)
class SourceConditionBundle:
    """Typed tensors plus their content-bound source provenance."""

    video: torch.Tensor
    references: tuple[torch.Tensor, ...]
    provenance: SourceConditionProvenance

    def validate(self, *, label: str) -> None:
        if not isinstance(self.provenance, SourceConditionProvenance):
            raise PairV5SourceIdentityEnergyError(
                f"{label} requires typed SourceConditionProvenance"
            )
        self.provenance.validate()
        video = _exact81(self.video, label=f"{label} full video")
        references = _references(
            self.references,
            source_video=video,
            label=f"{label} references",
        )
        if _tensor_sha256(video) != self.provenance.full_video_tensor_sha256:
            raise PairV5SourceIdentityEnergyError(
                f"{label} full video differs from source provenance"
            )
        observed = tuple(_tensor_sha256(item) for item in references)
        if observed != self.provenance.reference_tensor_sha256:
            raise PairV5SourceIdentityEnergyError(
                f"{label} references differ from fixed-index source provenance"
            )


def make_source_condition_bundle(
    source_video: torch.Tensor,
    source_references: Sequence[torch.Tensor],
    provenance: SourceConditionProvenance,
) -> SourceConditionBundle:
    if not isinstance(provenance, SourceConditionProvenance):
        raise PairV5SourceIdentityEnergyError(
            "source bundle requires typed SourceConditionProvenance, not an opaque digest"
        )
    video = _exact81(source_video, label="source bundle full video")
    references = _references(
        source_references,
        source_video=video,
        label="source bundle references",
    )
    result = SourceConditionBundle(video, references, provenance)
    result.validate(label="source bundle")
    return result


def _validate_disjoint_source_bundles(
    correct: SourceConditionBundle,
    wrong: SourceConditionBundle,
) -> None:
    correct.validate(label="correct source bundle")
    wrong.validate(label="wrong source bundle")
    left = correct.provenance
    right = wrong.provenance
    distinct_fields = (
        "source_key",
        "source_media_artifact_sha256",
        "source_media_receipt_digest",
        "full_video_tensor_sha256",
        "full_video_encoding_receipt_digest",
        "reference_extraction_receipt_digest",
        "receipt_digest",
    )
    aliased = [name for name in distinct_fields if getattr(left, name) == getattr(right, name)]
    if aliased:
        raise PairV5SourceIdentityEnergyError(
            f"correct/wrong source provenance is not source-disjoint: {aliased}"
        )
    left_content = {
        left.full_video_tensor_sha256,
        *left.reference_tensor_sha256,
    }
    right_content = {
        right.full_video_tensor_sha256,
        *right.reference_tensor_sha256,
    }
    if left_content & right_content:
        raise PairV5SourceIdentityEnergyError(
            "correct/wrong source bundles share condition content hashes"
        )
    left_storage = {_storage_ptr(correct.video), *(_storage_ptr(item) for item in correct.references)}
    right_storage = {_storage_ptr(wrong.video), *(_storage_ptr(item) for item in wrong.references)}
    if left_storage & right_storage:
        raise PairV5SourceIdentityEnergyError(
            "correct/wrong source bundles share tensor storage"
        )


@dataclass(frozen=True)
class SourceBindingPreregistration:
    """Immutable strictly-positive source-binding selection policy."""

    policy_id: str
    minimum_source_binding_margin: float
    unit_score_saturation_margin: float
    receipt_digest: str

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": SOURCE_POLICY_SCHEMA,
            "policy_id": self.policy_id,
            "minimum_source_binding_margin": self.minimum_source_binding_margin,
            "unit_score_saturation_margin": self.unit_score_saturation_margin,
            "pass_comparison": "every_counterfactual_gap_strictly_greater_than_minimum",
            "unit_score_mapping": "clip((margin-minimum)/(saturation-minimum),0,1)",
            "reject_exact_null_copy": True,
            "reject_candidate_source_storage_alias": True,
            "score_semantics": SOURCE_BINDING_SCORE_SEMANTICS,
        }

    def validate(self) -> None:
        _source_key(self.policy_id, label="source-binding policy_id")
        canonical: dict[str, float] = {}
        for name in (
            "minimum_source_binding_margin",
            "unit_score_saturation_margin",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value):
                raise PairV5SourceIdentityEnergyError(
                    f"source-binding policy {name} must be a finite float"
                )
            fp32 = torch.tensor(value, dtype=torch.float32)
            if not bool(torch.isfinite(fp32).item()) or float(fp32.item()) != value:
                raise PairV5SourceIdentityEnergyError(
                    f"source-binding policy {name} must be canonical finite FP32"
                )
            canonical[name] = float(fp32.item())
        if canonical["minimum_source_binding_margin"] <= 0.0:
            raise PairV5SourceIdentityEnergyError(
                "minimum source-binding margin must be strictly positive"
            )
        if (
            canonical["unit_score_saturation_margin"]
            <= canonical["minimum_source_binding_margin"]
        ):
            raise PairV5SourceIdentityEnergyError(
                "unit score saturation margin must exceed the minimum margin"
            )
        _require_sha256(self.receipt_digest, label="source-binding policy receipt_digest")
        if self.receipt_digest != object_sha256(self.payload()):
            raise PairV5SourceIdentityEnergyError(
                "source-binding preregistration receipt digest differs"
            )


def seal_source_binding_preregistration(
    *,
    policy_id: str,
    minimum_source_binding_margin: float,
    unit_score_saturation_margin: float,
) -> SourceBindingPreregistration:
    # Normalize policy values to the exact FP32 scalars used by the evaluator.
    _source_key(policy_id, label="source-binding policy_id")
    normalized: dict[str, float] = {}
    for name, value in (
        ("minimum_source_binding_margin", minimum_source_binding_margin),
        ("unit_score_saturation_margin", unit_score_saturation_margin),
    ):
        if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value):
            raise PairV5SourceIdentityEnergyError(f"{name} must be a finite float")
        fp32 = torch.tensor(value, dtype=torch.float32)
        if not bool(torch.isfinite(fp32).item()):
            raise PairV5SourceIdentityEnergyError(f"{name} is not finite FP32")
        normalized[name] = float(fp32.item())
    if normalized["minimum_source_binding_margin"] <= 0.0:
        raise PairV5SourceIdentityEnergyError(
            "minimum source-binding margin must be strictly positive"
        )
    if (
        normalized["unit_score_saturation_margin"]
        <= normalized["minimum_source_binding_margin"]
    ):
        raise PairV5SourceIdentityEnergyError(
            "unit score saturation margin must exceed the minimum margin"
        )
    provisional = SourceBindingPreregistration(
        policy_id=policy_id,
        minimum_source_binding_margin=normalized[
            "minimum_source_binding_margin"
        ],
        unit_score_saturation_margin=normalized["unit_score_saturation_margin"],
        receipt_digest="",
    )
    result = SourceBindingPreregistration(
        **{
            **provisional.__dict__,
            "receipt_digest": object_sha256(provisional.payload()),
        }
    )
    result.validate()
    return result


def _text_condition(value: Any, *, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(int(item) for item in value.shape)
        != (
            1,
            runtime_contract.PINNED_TEXT_TOKENS,
            runtime_contract.PINNED_TEXT_DIM,
        )
        or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or value.device.type == "meta"
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise PairV5SourceIdentityEnergyError(
            f"{label} must be frozen [1,512,4096]"
        )
    return value


def _validate_coordinate(
    sigma: Any,
    timestep: Any,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    try:
        sigma_value, timestep_value, index = native_bridge._native_schedule_coordinate(
            sigma, timestep
        )
    except native_bridge.PairV5NativeBridgeError as error:
        raise PairV5SourceIdentityEnergyError(str(error)) from error
    if sigma_value.device != device:
        raise PairV5SourceIdentityEnergyError(
            "native coordinate and candidate use different devices"
        )
    return sigma_value, timestep_value, index


@dataclass(frozen=True)
class FrozenIdentityCellPredictions:
    """Four detached frozen-model velocity fields in candidate coordinates."""

    velocities: Mapping[str, torch.Tensor]
    receipt: Mapping[str, Any]


class FrozenBerniniRV2V4IdentityScorer(nn.Module):
    """Frozen native RV2V-4 scorer with sealed action/no-op text conditions."""

    def __init__(
        self,
        diffusion: nn.Module,
        transformer: nn.Module,
        action_condition: torch.Tensor,
        unconditional_condition: torch.Tensor,
        noop_condition: torch.Tensor,
        *,
        frozen_model_receipt_digest: str,
    ) -> None:
        super().__init__()
        if not isinstance(diffusion, nn.Module) or not isinstance(transformer, nn.Module):
            raise PairV5SourceIdentityEnergyError(
                "diffusion and transformer must be auditable torch modules"
            )
        if not callable(getattr(diffusion, "shared_step", None)) or not callable(
            getattr(transformer, "patch_vae_latent", None)
        ):
            raise PairV5SourceIdentityEnergyError(
                "frozen scorer requires shared_step and patch_vae_latent"
            )
        if diffusion.training or transformer.training:
            raise PairV5SourceIdentityEnergyError(
                "frozen identity scorer modules must be in eval mode"
            )
        if any(parameter.requires_grad for parameter in diffusion.parameters()) or any(
            parameter.requires_grad for parameter in transformer.parameters()
        ):
            raise PairV5SourceIdentityEnergyError(
                "frozen identity scorer contains trainable parameters"
            )
        embedded_transformer = getattr(diffusion, "transformer", transformer)
        if embedded_transformer is not transformer:
            raise PairV5SourceIdentityEnergyError(
                "diffusion is bound to a different transformer"
            )
        action = _text_condition(action_condition, label="action_condition")
        unconditional = _text_condition(
            unconditional_condition, label="unconditional_condition"
        )
        noop = _text_condition(noop_condition, label="noop_condition")
        if action.device != unconditional.device or action.device != noop.device:
            raise PairV5SourceIdentityEnergyError(
                "all frozen text conditions must share one device"
            )
        if torch.equal(action, unconditional) or torch.equal(action, noop) or torch.equal(
            unconditional, noop
        ):
            raise PairV5SourceIdentityEnergyError(
                "action, no-op, and unconditional text conditions must be distinct"
            )

        self.diffusion = diffusion
        self.transformer = transformer
        self.register_buffer("_action_condition", action, persistent=False)
        self.register_buffer("_unconditional_condition", unconditional, persistent=False)
        self.register_buffer("_noop_condition", noop, persistent=False)
        self._condition_registry_digest = self._current_condition_registry_digest()
        self._frozen_model_receipt_digest = _require_sha256(
            frozen_model_receipt_digest, label="frozen_model_receipt_digest"
        )
        self._last_receipt: Mapping[str, Any] | None = None
        # ``nn.Module`` instances start in training mode even when all of
        # their children were already frozen/eval.  Seal this wrapper too.
        self.eval()

    @property
    def condition_registry_digest(self) -> str:
        return self._condition_registry_digest

    @property
    def last_receipt(self) -> Mapping[str, Any] | None:
        return self._last_receipt

    def _current_condition_registry_digest(self) -> str:
        return object_sha256(
            {
                "action": _tensor_sha256(self._action_condition),
                "unconditional": _tensor_sha256(self._unconditional_condition),
                "noop": _tensor_sha256(self._noop_condition),
            }
        )

    def forward(
        self,
        x_sigma: torch.Tensor,
        sigma: torch.Tensor,
        timestep: torch.Tensor,
        correct_source: SourceConditionBundle,
        wrong_source: SourceConditionBundle,
    ) -> FrozenIdentityCellPredictions:
        state = _exact81(x_sigma, label="candidate-own x_sigma")
        if not isinstance(correct_source, SourceConditionBundle) or not isinstance(
            wrong_source, SourceConditionBundle
        ):
            raise PairV5SourceIdentityEnergyError(
                "frozen scorer requires typed correct/wrong source bundles"
            )
        _validate_disjoint_source_bundles(correct_source, wrong_source)
        correct_video = correct_source.video
        wrong_video = wrong_source.video
        if any(
            tuple(value.shape) != tuple(state.shape) or value.device != state.device
            for value in (correct_video, wrong_video)
        ):
            raise PairV5SourceIdentityEnergyError(
                "candidate and correct/wrong source geometry/device differ"
            )
        correct_refs = correct_source.references
        wrong_refs = wrong_source.references
        sigma_value, timestep_value, schedule_index = _validate_coordinate(
            sigma, timestep, device=state.device
        )
        if any(
            condition.device != state.device
            for condition in (
                self._action_condition,
                self._unconditional_condition,
                self._noop_condition,
            )
        ):
            raise PairV5SourceIdentityEnergyError(
                "frozen text registry and candidate use different devices"
            )
        if self.training or self.diffusion.training or self.transformer.training:
            raise PairV5SourceIdentityEnergyError(
                "frozen identity scorer changed out of eval mode"
            )
        if any(parameter.requires_grad for parameter in self.diffusion.parameters()) or any(
            parameter.requires_grad for parameter in self.transformer.parameters()
        ):
            raise PairV5SourceIdentityEnergyError(
                "frozen identity scorer became trainable"
            )
        if self._current_condition_registry_digest() != self.condition_registry_digest:
            raise PairV5SourceIdentityEnergyError(
                "frozen action/no-op text-condition registry differs from its seal"
            )

        with torch.no_grad():
            try:
                correct_pack = native.build_native_rv2v_pack(
                    self.transformer,
                    donor_video=correct_video,
                    image_references=correct_refs,
                    noisy_target=state,
                )
                wrong_pack = native.build_native_rv2v_pack(
                    self.transformer,
                    donor_video=wrong_video,
                    image_references=wrong_refs,
                    noisy_target=state,
                )
                correct_packed, correct_components = (
                    native_bridge._guided_packed_prediction(
                        self.diffusion,
                        self.transformer,
                        correct_pack,
                        timestep=timestep_value,
                        cond_embeds=self._action_condition,
                        uncond_embeds=self._unconditional_condition,
                        adapter=None,
                        sequence_parallel_rank=0,
                        sequence_parallel_size=1,
                        sigma_schedule_index=schedule_index,
                        adapter_enabled=False,
                        no_grad=True,
                    )
                )
                wrong_packed, _ = native_bridge._guided_packed_prediction(
                    self.diffusion,
                    self.transformer,
                    wrong_pack,
                    timestep=timestep_value,
                    cond_embeds=self._action_condition,
                    uncond_embeds=self._unconditional_condition,
                    adapter=None,
                    sequence_parallel_rank=0,
                    sequence_parallel_size=1,
                    sigma_schedule_index=schedule_index,
                    adapter_enabled=False,
                    no_grad=True,
                )
                noop_packed, _ = native_bridge._guided_packed_prediction(
                    self.diffusion,
                    self.transformer,
                    correct_pack,
                    timestep=timestep_value,
                    cond_embeds=self._noop_condition,
                    uncond_embeds=self._unconditional_condition,
                    adapter=None,
                    sequence_parallel_rank=0,
                    sequence_parallel_size=1,
                    sigma_schedule_index=schedule_index,
                    adapter_enabled=False,
                    no_grad=True,
                )
                video_action = native.forward_native_target_branch(
                    self.diffusion,
                    correct_pack.video,
                    timestep=timestep_value,
                    cond_embeds=self._action_condition,
                )
            except (
                native.NativeRefContrastiveV3Error,
                native_bridge.PairV5NativeBridgeError,
            ) as error:
                raise PairV5SourceIdentityEnergyError(str(error)) from error

            none_uncond = correct_components["none_uncond"]
            video_uncond = correct_components["V_uncond"]
            reference_off_packed = (
                none_uncond
                + guidance.OMEGA_VIDEO * (video_uncond - none_uncond)
                + guidance.OMEGA_TEXT * (video_action - video_uncond)
            )
            packed_by_cell = {
                "correct": correct_packed,
                "wrong_source": wrong_packed,
                "reference_off": reference_off_packed,
                "noop": noop_packed,
            }
            velocities = {
                name: native_bridge._unpack_spatial_velocity(
                    packed.float(), video_shape=state.shape
                )
                .detach()
                .contiguous()
                for name, packed in packed_by_cell.items()
            }

        if tuple(velocities) != CELL_ORDER:
            raise PairV5SourceIdentityEnergyError("identity cell order differs")
        for name, value in velocities.items():
            _exact81(value, label=f"{name} frozen velocity")
            if tuple(value.shape) != tuple(state.shape) or value.device != state.device:
                raise PairV5SourceIdentityEnergyError(
                    f"{name} velocity and candidate geometry/device differ"
                )
        value = {
            "schema_version": SCORER_SCHEMA,
            "contract_digest": contract_receipt()["digest"],
            "cell_order": list(CELL_ORDER),
            "candidate_state_shape": [int(item) for item in state.shape],
            "native_exact40_schedule_index": schedule_index,
            "sigma_float32_bits_hex": _fp32_bits(sigma_value, label="sigma"),
            "timestep_float32_bits_hex": _fp32_bits(
                timestep_value, label="timestep"
            ),
            "frozen_model_receipt_digest": self._frozen_model_receipt_digest,
            "condition_registry_digest": self.condition_registry_digest,
            "correct_source_provenance_digest": (
                correct_source.provenance.receipt_digest
            ),
            "wrong_source_provenance_digest": wrong_source.provenance.receipt_digest,
            "native_rv2v4_reference_contract_digest": (
                native.native_rv2v4_reference_contract()["digest"]
            ),
            "guidance_receipt_digest": guidance.guidance_receipt()["digest"],
            "correct_and_wrong_use_native_rv2v4": True,
            "reference_off_uses_native_video_branch": True,
            "reference_off_uses_synthetic_reference": False,
            "same_candidate_state_object_for_both_native_packs": True,
            "typed_content_bound_source_bundles": True,
            "correct_wrong_source_content_and_origin_disjoint": True,
            "same_sigma_and_timestep_for_every_cell": True,
            "all_velocities_detached_fp32": True,
            "postvideo_scores_consumed": False,
            "student_or_adapter_consumed": False,
            "proposal_or_paired_target_consumed": False,
            "mask_flow_pose_track_trajectory_consumed": False,
        }
        self._last_receipt = {**value, "digest": object_sha256(value)}
        return FrozenIdentityCellPredictions(velocities, self._last_receipt)


@dataclass(frozen=True)
class SafeParetoSourceBindingPacket:
    """Unit-interval preservation proxy plus closed evaluator provenance."""

    unit_interval_score: float
    source_binding_pass: bool
    selection_authorized: bool
    evaluator_provenance: Mapping[str, Any]
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class CandidateSourceBindingEnergy:
    """Detached full-source binding result; not a pure actor-identity score."""

    x_sigma: torch.Tensor
    target_velocity: torch.Tensor
    cell_velocities: Mapping[str, torch.Tensor]
    cell_energies: Mapping[str, torch.Tensor]
    counterfactual_gaps: Mapping[str, torch.Tensor]
    source_binding_margin: torch.Tensor
    source_binding_pass: bool
    source_binding_score: torch.Tensor
    postvideo_scores: Mapping[str, torch.Tensor]
    safe_pareto_packet: SafeParetoSourceBindingPacket
    receipt: Mapping[str, Any]


def evaluate_candidate_source_binding_energy(
    clean_candidate: torch.Tensor,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    timestep: torch.Tensor,
    correct_source: SourceConditionBundle,
    wrong_source: SourceConditionBundle,
    scorer: FrozenBerniniRV2V4IdentityScorer,
    preregistration: SourceBindingPreregistration,
    postvideo_camera_score: torch.Tensor,
    postvideo_background_score: torch.Tensor,
    postvideo_quality_score: torch.Tensor,
    *,
    candidate_receipt_digest: str,
    registered_preregistration_digest: str,
    postvideo_evaluator_receipt_digest: str,
) -> CandidateSourceBindingEnergy:
    """Evaluate deploy-available source binding with strict null rejection."""

    if not isinstance(scorer, FrozenBerniniRV2V4IdentityScorer):
        raise PairV5SourceIdentityEnergyError(
            "scorer must be FrozenBerniniRV2V4IdentityScorer"
        )
    if not isinstance(preregistration, SourceBindingPreregistration):
        raise PairV5SourceIdentityEnergyError(
            "source-binding threshold requires typed preregistration"
        )
    preregistration.validate()
    registered_policy = _require_sha256(
        registered_preregistration_digest,
        label="registered_preregistration_digest",
    )
    if registered_policy != preregistration.receipt_digest:
        raise PairV5SourceIdentityEnergyError(
            "runtime source-binding policy differs from preregistered digest"
        )
    if not isinstance(correct_source, SourceConditionBundle) or not isinstance(
        wrong_source, SourceConditionBundle
    ):
        raise PairV5SourceIdentityEnergyError(
            "candidate evaluator requires typed correct/wrong source bundles"
        )
    _validate_disjoint_source_bundles(correct_source, wrong_source)

    clean = _exact81(clean_candidate, label="clean RV2V candidate")
    noise = _exact81(epsilon, label="shared epsilon")
    if tuple(clean.shape) != tuple(noise.shape) or clean.device != noise.device:
        raise PairV5SourceIdentityEnergyError(
            "clean candidate and epsilon geometry/device differ"
        )
    source_tensors = (
        correct_source.video,
        *correct_source.references,
        wrong_source.video,
        *wrong_source.references,
    )
    if any(
        tensor.device != clean.device
        or (
            tensor.ndim == 5
            and int(tensor.shape[2]) == LATENT_PHASES
            and tuple(tensor.shape) != tuple(clean.shape)
        )
        for tensor in source_tensors
    ):
        raise PairV5SourceIdentityEnergyError(
            "candidate and source condition geometry/device differ"
        )
    candidate_tensor_digest = _tensor_sha256(clean)
    if candidate_tensor_digest == correct_source.provenance.full_video_tensor_sha256:
        raise PairV5SourceIdentityEnergyError(
            "exact null-copy candidate is forbidden by source-binding preregistration"
        )
    candidate_storage = _storage_ptr(clean)
    if any(candidate_storage == _storage_ptr(item) for item in source_tensors):
        raise PairV5SourceIdentityEnergyError(
            "candidate may not alias correct/wrong source condition storage"
        )
    sigma_value, timestep_value, schedule_index = _validate_coordinate(
        sigma, timestep, device=clean.device
    )
    candidate_digest = _require_sha256(
        candidate_receipt_digest, label="candidate_receipt_digest"
    )
    postvideo_evaluator_digest = _require_sha256(
        postvideo_evaluator_receipt_digest,
        label="postvideo_evaluator_receipt_digest",
    )
    threshold = torch.tensor(
        preregistration.minimum_source_binding_margin,
        dtype=torch.float32,
        device=clean.device,
    )
    saturation = torch.tensor(
        preregistration.unit_score_saturation_margin,
        dtype=torch.float32,
        device=clean.device,
    )

    with torch.no_grad():
        sigma_spatial = sigma_value.reshape(1, 1, 1, 1, 1)
        x_sigma = ((1.0 - sigma_spatial) * clean + sigma_spatial * noise).contiguous()
        target_velocity = (noise - clean).contiguous()
        predictions = scorer(
            x_sigma,
            sigma_value,
            timestep_value,
            correct_source,
            wrong_source,
        )
        cell_energies = {
            name: (predictions.velocities[name] - target_velocity)
            .float()
            .square()
            .mean()
            .detach()
            for name in CELL_ORDER
        }
        gaps = {
            name: (cell_energies[name] - cell_energies["correct"]).detach()
            for name in COUNTERFACTUAL_CELL_ORDER
        }
        binding_margin = torch.stack(
            [gaps[name] for name in COUNTERFACTUAL_CELL_ORDER]
        ).amin().detach()
        binding_pass = all(
            bool((gaps[name] > threshold).item())
            for name in COUNTERFACTUAL_CELL_ORDER
        )
        unit_score = (
            (binding_margin - threshold) / (saturation - threshold)
        ).clamp(0.0, 1.0).detach()

    postvideo_scores = {
        "camera": _fp32_scalar(
            postvideo_camera_score,
            label="postvideo_camera_score",
            unit_interval=True,
        ),
        "background": _fp32_scalar(
            postvideo_background_score,
            label="postvideo_background_score",
            unit_interval=True,
        ),
        "quality": _fp32_scalar(
            postvideo_quality_score,
            label="postvideo_quality_score",
            unit_interval=True,
        ),
    }
    scorer_receipt = predictions.receipt
    if not isinstance(scorer_receipt, Mapping) or not isinstance(
        scorer_receipt.get("digest"), str
    ):
        raise PairV5SourceIdentityEnergyError("frozen scorer receipt is unavailable")

    evaluator_value = {
        "schema_version": "bernini-pair-v5-source-binding-evaluator-provenance-v1",
        "contract_digest": contract_receipt()["digest"],
        "frozen_scorer_receipt_digest": scorer_receipt["digest"],
        "source_binding_preregistration_digest": preregistration.receipt_digest,
        "correct_source_provenance_digest": (
            correct_source.provenance.receipt_digest
        ),
        "wrong_source_provenance_digest": wrong_source.provenance.receipt_digest,
        "candidate_receipt_digest": candidate_digest,
        "postvideo_evaluator_receipt_digest": postvideo_evaluator_digest,
        "score_semantics": SOURCE_BINDING_SCORE_SEMANTICS,
        "safe_pareto_axis": "identity",
        "pure_actor_identity_claim": False,
        "full_source_identity_background_camera_old_motion_entangled": True,
        "exact_null_copy_rejected": True,
        "strictly_positive_preregistered_margin": True,
    }
    evaluator_provenance = {
        **evaluator_value,
        "provenance_digest": object_sha256(evaluator_value),
    }

    value = {
        "schema_version": RECEIPT_SCHEMA,
        "contract_digest": contract_receipt()["digest"],
        "candidate_receipt_digest": candidate_digest,
        "candidate_tensor_digest": candidate_tensor_digest,
        "epsilon_tensor_digest": _tensor_sha256(noise),
        "candidate_shape": [int(item) for item in clean.shape],
        "frame_count": FRAME_COUNT,
        "native_exact40_schedule_index": schedule_index,
        "sigma_float32_bits_hex": _fp32_bits(sigma_value, label="sigma"),
        "timestep_float32_bits_hex": _fp32_bits(timestep_value, label="timestep"),
        "correct_source_provenance": correct_source.provenance.payload(),
        "correct_source_provenance_digest": correct_source.provenance.receipt_digest,
        "wrong_source_provenance": wrong_source.provenance.payload(),
        "wrong_source_provenance_digest": wrong_source.provenance.receipt_digest,
        "source_disjointness": {
            "source_keys_distinct": True,
            "source_media_artifact_hashes_distinct": True,
            "source_receipts_distinct": True,
            "full_video_tensor_content_distinct": True,
            "all_cross_bundle_condition_content_hashes_disjoint": True,
            "condition_bundle_storage_disjoint": True,
        },
        "candidate_exact_null_copy_rejected": True,
        "candidate_source_storage_disjoint": True,
        "frozen_scorer_receipt_digest": scorer_receipt["digest"],
        "cell_order": list(CELL_ORDER),
        "target_velocity_definition": "epsilon-clean_candidate",
        "cell_energy_definition": "mean_fp32_squared_velocity_error",
        "cell_energy_float32_bits_hex": {
            name: _fp32_bits(cell_energies[name], label=f"{name} energy")
            for name in CELL_ORDER
        },
        "counterfactual_gap_definition": "counterfactual_energy-correct_energy",
        "counterfactual_gap_float32_bits_hex": {
            name: _fp32_bits(gaps[name], label=f"{name} gap")
            for name in COUNTERFACTUAL_CELL_ORDER
        },
        "source_binding_margin_definition": "min_wrong_source_reference_off_noop_gap",
        "source_binding_margin_float32_bits_hex": _fp32_bits(
            binding_margin, label="source-binding margin"
        ),
        "minimum_source_binding_margin_float32_bits_hex": _fp32_bits(
            threshold, label="minimum source-binding margin"
        ),
        "unit_score_saturation_margin_float32_bits_hex": _fp32_bits(
            saturation, label="unit score saturation margin"
        ),
        "source_binding_pass_comparison": "all_gaps_strictly_greater_than_threshold",
        "source_binding_pass": binding_pass,
        "source_binding_unit_interval_score_float32_bits_hex": _fp32_bits(
            unit_score, label="source-binding score"
        ),
        "source_binding_score_semantics": SOURCE_BINDING_SCORE_SEMANTICS,
        "pure_actor_identity_claim": False,
        "evaluator_provenance_digest": evaluator_provenance["provenance_digest"],
        "postvideo_score_order": list(POSTVIDEO_SCORE_ORDER),
        "postvideo_score_float32_bits_hex": {
            name: _fp32_bits(postvideo_scores[name], label=f"{name} score")
            for name in POSTVIDEO_SCORE_ORDER
        },
        "same_clean_candidate_epsilon_sigma_for_every_cell": True,
        "candidate_source_epsilon_velocities_and_energies_detached_fp32": True,
        "postvideo_scores_used_as_model_conditions": False,
        "postvideo_scores_modify_source_binding_energy": False,
        "score_packet_role": "safe_pareto_preservation_proxy_only",
        "score_packet_is_student_input": False,
        "proposal_or_paired_target_consumed": False,
        "mask_flow_pose_track_trajectory_consumed": False,
        "scientific_action_editing_claim": False,
    }
    receipt = {**value, "digest": object_sha256(value)}
    unit_score_float = float(unit_score.item())
    packet_value = {
        "schema_version": SAFE_PARETO_PACKET_SCHEMA,
        "safe_pareto_axis": "identity",
        "safe_pareto_candidate_score_field": "identity_score",
        "identity_score": unit_score_float,
        "unit_interval_score": unit_score_float,
        "score_semantics": SOURCE_BINDING_SCORE_SEMANTICS,
        "source_binding_pass": binding_pass,
        "selection_authorized": binding_pass,
        "pure_actor_identity_claim": False,
        "candidate_source_binding_receipt_digest": receipt["digest"],
        "evaluator_provenance_digest": evaluator_provenance["provenance_digest"],
    }
    packet_receipt = {**packet_value, "packet_digest": object_sha256(packet_value)}
    packet = SafeParetoSourceBindingPacket(
        unit_interval_score=unit_score_float,
        source_binding_pass=binding_pass,
        selection_authorized=binding_pass,
        evaluator_provenance=evaluator_provenance,
        receipt=packet_receipt,
    )
    return CandidateSourceBindingEnergy(
        x_sigma=x_sigma.detach(),
        target_velocity=target_velocity.detach(),
        cell_velocities={
            name: predictions.velocities[name].detach() for name in CELL_ORDER
        },
        cell_energies=cell_energies,
        counterfactual_gaps=gaps,
        source_binding_margin=binding_margin,
        source_binding_pass=binding_pass,
        source_binding_score=unit_score,
        postvideo_scores=postvideo_scores,
        safe_pareto_packet=packet,
        receipt=receipt,
    )


# Compatibility names retain import stability while exposing source-binding
# semantics and the new typed signature.
CandidateSourceIdentityEnergy = CandidateSourceBindingEnergy
evaluate_candidate_source_identity_energy = evaluate_candidate_source_binding_energy


def contract_receipt() -> Mapping[str, Any]:
    """Return and re-check the closed public information-flow contract."""

    signatures = {
        "FrozenBerniniRV2V4IdentityScorer.__init__": set(
            inspect.signature(FrozenBerniniRV2V4IdentityScorer.__init__).parameters
        ),
        "FrozenBerniniRV2V4IdentityScorer.forward": set(
            inspect.signature(FrozenBerniniRV2V4IdentityScorer.forward).parameters
        ),
        "evaluate_candidate_source_binding_energy": set(
            inspect.signature(evaluate_candidate_source_binding_energy).parameters
        ),
    }
    offending = {
        name: sorted(parameters & FORBIDDEN_PUBLIC_INPUT_NAMES)
        for name, parameters in signatures.items()
        if parameters & FORBIDDEN_PUBLIC_INPUT_NAMES
    }
    if offending:
        raise PairV5SourceIdentityEnergyError(
            f"public source-binding API exposes forbidden inputs: {offending}"
        )
    value = {
        "schema_version": SCHEMA_VERSION,
        "frame_count": FRAME_COUNT,
        "latent_channels": LATENT_CHANNELS,
        "latent_phases": LATENT_PHASES,
        "reference_count_per_source": REFERENCE_COUNT,
        "reference_frame_indices": list(REFERENCE_FRAME_INDICES),
        "cell_order": list(CELL_ORDER),
        "counterfactual_cell_order": list(COUNTERFACTUAL_CELL_ORDER),
        "postvideo_score_order": list(POSTVIDEO_SCORE_ORDER),
        "native_rv2v4_reference_contract_digest": (
            native.native_rv2v4_reference_contract()["digest"]
        ),
        "native_exact40_schedule_digest": (
            native.native_unipc40_schedule_receipt()["digest"]
        ),
        "native_bridge_contract_digest": native_bridge.bridge_contract_receipt()[
            "digest"
        ],
        "guidance_receipt_digest": guidance.guidance_receipt()["digest"],
        "correct_and_wrong_cells": "native_rv2v4_action_condition",
        "reference_off_cell": "native_video_only_action_condition",
        "noop_cell": "native_rv2v4_noop_condition",
        "state_equation": "x_sigma=(1-sigma)*clean_candidate+sigma*epsilon",
        "velocity_label": "epsilon-clean_candidate",
        "source_binding_margin": (
            "min(counterfactual_mse-correct_source_mse)"
        ),
        "source_provenance_schema": SOURCE_PROVENANCE_SCHEMA,
        "source_provenance_fields": [
            "source_key",
            "source_media_artifact_sha256",
            "source_media_receipt_digest",
            "full_video_tensor_sha256",
            "full_video_encoding_receipt_digest",
            "reference_frame_indices",
            "reference_tensor_sha256",
            "reference_extraction_receipt_digest",
            "receipt_digest",
        ],
        "typed_source_bundles_required": True,
        "source_origin_content_and_storage_disjointness_required": True,
        "strictly_positive_preregistered_margin_required": True,
        "counterfactual_gap_comparison": "strict_greater_than",
        "exact_null_copy_rejected": True,
        "candidate_source_storage_alias_rejected": True,
        "unit_interval_score_semantics": SOURCE_BINDING_SCORE_SEMANTICS,
        "safe_pareto_packet_schema": SAFE_PARETO_PACKET_SCHEMA,
        "safe_pareto_axis": "identity",
        "pure_actor_identity_claim": False,
        "full_source_identity_background_camera_old_motion_entangled": True,
        "candidate_source_epsilon_velocities_and_energies_detached_fp32": True,
        "native_internal_compute_dtype_may_follow_frozen_model": True,
        "postvideo_scores_are_unit_interval_detached_fp32": True,
        "postvideo_scores_are_never_model_conditions": True,
        "student_input_fields_added": [],
        "forbidden_public_input_names": sorted(FORBIDDEN_PUBLIC_INPUT_NAMES),
        "score_packet_role": "safe_pareto_preservation_proxy_only",
        "proposal_or_paired_target_consumed": False,
        "mask_flow_pose_track_trajectory_consumed": False,
        "scientific_action_editing_claim": False,
    }
    return {**value, "digest": object_sha256(value)}


__all__ = [
    "CELL_ORDER",
    "COUNTERFACTUAL_CELL_ORDER",
    "CandidateSourceBindingEnergy",
    "CandidateSourceIdentityEnergy",
    "FORBIDDEN_PUBLIC_INPUT_NAMES",
    "FRAME_COUNT",
    "FrozenBerniniRV2V4IdentityScorer",
    "FrozenIdentityCellPredictions",
    "LATENT_CHANNELS",
    "LATENT_PHASES",
    "POSTVIDEO_SCORE_ORDER",
    "PairV5SourceIdentityEnergyError",
    "REFERENCE_COUNT",
    "REFERENCE_FRAME_INDICES",
    "RECEIPT_SCHEMA",
    "SAFE_PARETO_PACKET_SCHEMA",
    "SCHEMA_VERSION",
    "SCORER_SCHEMA",
    "SOURCE_BINDING_SCORE_SEMANTICS",
    "SOURCE_POLICY_SCHEMA",
    "SOURCE_PROVENANCE_SCHEMA",
    "SafeParetoSourceBindingPacket",
    "SourceBindingPreregistration",
    "SourceConditionBundle",
    "SourceConditionProvenance",
    "contract_receipt",
    "evaluate_candidate_source_binding_energy",
    "evaluate_candidate_source_identity_energy",
    "make_source_condition_bundle",
    "object_sha256",
    "seal_source_binding_preregistration",
    "seal_source_condition_provenance",
]
