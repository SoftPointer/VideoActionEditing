#!/usr/bin/env python3
"""Fail-closed BRAID Stage-A source-reconstruction primitives.

This module is deliberately smaller than a trainer.  It binds Stage A to the
pinned Bernini-R native RV2V-4 pack and exact-40, flow-shift=5 UniPC
coordinates already audited in :mod:`source_self_native_ref_contrastive_v3`.
It provides:

* exact81 ``source -> same source`` rectified-flow states;
* native correct/source-dropped/wrong-source query packs sharing one noisy
  target state, timestep, text condition, and Gaussian;
* the teacher-forced FM plus non-compensating source-dependence margins; and
* a structural Stage-0 authorization gate for constructing a graph-connected
  Stage-A objective.

It does not construct an optimizer, call backward, mutate a parameter, or
apply an update.  In particular, the old fixed-sigma donor-repaint and
appearance-orbit supervision are not reused.  A future launcher must still
close the three live Stage-0 canaries, the I_source module/sigma scope, native
WORLD8=DP2xSP4 gradient routing, and copy-on-write rollback before it may own
an optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Optional, Sequence

import torch

import source_self_native_ref_contrastive_v3 as native
import source_self_runtime as source_runtime


SCHEMA_VERSION = "bernini-braid-stage-a-source-reconstruction-v1"
STAGE0_AUTHORIZATION_SCHEMA_VERSION = "bernini-braid-stage0-authorization-v1"
FRAME_COUNT = 81
FPS = 25
LATENT_PHASES = native.LATENT_PHASES
LATENT_CHANNELS = native.LATENT_CHANNELS
REFERENCE_COUNT = native.REFERENCE_COUNT
QUERY_NAMES = ("correct", "drop", "wrong")
REQUIRED_STAGE0_CANARIES = (
    "two_branch_native_apg_parity",
    "co_state_reset_world4_sp4_oracle",
    "old_motion_action_capacity_oracle",
)
PINNED_CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
PINNED_BERNINI_REVISION = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
PINNED_VEOMNI_REVISION = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class BRAIDStageAError(RuntimeError):
    """Raised before an ambiguous Stage-A query or training objective exists."""


def object_sha256(value: Any) -> str:
    """Use the pinned native module's canonical finite-JSON digest."""

    try:
        return native.object_sha256(value)
    except native.NativeRefContrastiveV3Error as error:
        raise BRAIDStageAError(str(error)) from error


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise BRAIDStageAError(f"{label} must be one lowercase SHA-256")
    return value


def _revision(value: Any, *, label: str) -> str:
    if type(value) is not str or _REVISION.fullmatch(value) is None:
        raise BRAIDStageAError(f"{label} must be one lowercase 40-hex revision")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(value) is None:
        raise BRAIDStageAError(f"{label} is not a closed safe identifier")
    return value


def _exact81_latent(value: Any, *, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.dtype != torch.float32
        or value.requires_grad
        or value.grad_fn is not None
        or value.ndim != 5
        or tuple(int(item) for item in value.shape[:3])
        != (1, LATENT_CHANNELS, LATENT_PHASES)
        or int(value.shape[3]) <= 0
        or int(value.shape[4]) <= 0
        or int(value.shape[3]) % 2
        or int(value.shape[4]) % 2
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise BRAIDStageAError(
            f"{label} must be detached contiguous finite FP32 "
            f"[1,{LATENT_CHANNELS},{LATENT_PHASES},evenH,evenW]"
        )
    return value


def _reference_latents(
    values: Sequence[torch.Tensor],
    *,
    spatial_shape: tuple[int, int],
    device: torch.device,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = tuple(values)
    if len(rows) != REFERENCE_COUNT:
        raise BRAIDStageAError(f"{label} must contain exactly four native references")
    for index, value in enumerate(rows):
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.device.type == "meta"
            or value.dtype != torch.float32
            or value.requires_grad
            or value.grad_fn is not None
            or value.ndim != 5
            or tuple(int(item) for item in value.shape[:3])
            != (1, LATENT_CHANNELS, 1)
            or tuple(int(item) for item in value.shape[3:]) != spatial_shape
            or value.device != device
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all().item())
        ):
            raise BRAIDStageAError(
                f"{label}[{index}] must be detached contiguous finite FP32 "
                f"[1,{LATENT_CHANNELS},1,H,W] with source geometry"
            )
    return rows  # type: ignore[return-value]


@dataclass(frozen=True)
class TeacherForcedStageABatch:
    """One raw source, one Gaussian, and one exact-40 sigma stratum."""

    clean_source: torch.Tensor
    epsilon: torch.Tensor
    states: native.MultiSigmaStates
    source_video_sha256: str
    noop_caption_utf8_sha256: str

    def __post_init__(self) -> None:
        clean = _exact81_latent(self.clean_source, label="clean source")
        epsilon = _exact81_latent(self.epsilon, label="official Gaussian")
        if clean.shape != epsilon.shape or clean.device != epsilon.device:
            raise BRAIDStageAError("clean source and Gaussian geometry/device differ")
        _sha256(self.source_video_sha256, label="source video")
        _sha256(self.noop_caption_utf8_sha256, label="semantic-noop caption")
        if not isinstance(self.states, native.MultiSigmaStates):
            raise BRAIDStageAError("states must use the pinned native exact40 type")
        if (
            self.states.noisy.shape
            != (len(self.states.indices),) + tuple(clean.shape)
            or self.states.target_velocity.shape != self.states.noisy.shape
            or self.states.noisy.device != clean.device
        ):
            raise BRAIDStageAError("teacher-forced exact40 state geometry differs")

    def receipt(self) -> Mapping[str, Any]:
        schedule = native.native_unipc40_schedule_receipt()
        if schedule["digest"] != native.PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST:
            raise BRAIDStageAError("native exact40 schedule digest changed")
        value = {
            "schema_version": SCHEMA_VERSION,
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "latent_phases": LATENT_PHASES,
            "source_video_sha256": self.source_video_sha256,
            "noop_caption_utf8_sha256": self.noop_caption_utf8_sha256,
            "source_latent_sha256": source_runtime.tensor_sha256(self.clean_source),
            "gaussian_sha256": source_runtime.tensor_sha256(self.epsilon),
            "state_receipt": self.states.receipt(),
            "native_schedule_digest": schedule["digest"],
            "flow_shift": schedule["flow_shift"],
            "prediction_type": schedule["prediction_type"],
            "training_target_role": "same_raw_source_latent",
            "separate_edited_target_consumed": False,
            "same_state_and_gaussian_for_all_condition_queries": True,
        }
        return {**value, "digest": object_sha256(value)}


def prepare_teacher_forced_source_batch(
    clean_source: torch.Tensor,
    epsilon: torch.Tensor,
    *,
    indices: Sequence[int],
    source_video_sha256: str,
    noop_caption_utf8_sha256: str,
) -> TeacherForcedStageABatch:
    """Construct ``x_sigma=(1-sigma)S+sigma*epsilon`` and ``epsilon-S``.

    There is intentionally no second clean-video argument: the raw source is
    both the native visual condition and the exact reconstruction endpoint.
    """

    clean = _exact81_latent(clean_source, label="clean source")
    noise = _exact81_latent(epsilon, label="official Gaussian")
    if clean.shape != noise.shape or clean.device != noise.device:
        raise BRAIDStageAError("clean source and Gaussian geometry/device differ")
    try:
        states = native.build_multi_sigma_states(clean, noise, indices=indices)
    except native.NativeRefContrastiveV3Error as error:
        raise BRAIDStageAError(str(error)) from error
    return TeacherForcedStageABatch(
        clean,
        noise,
        states,
        _sha256(source_video_sha256, label="source video"),
        _sha256(noop_caption_utf8_sha256, label="semantic-noop caption"),
    )


def pack_exact81_velocity(value: torch.Tensor) -> torch.Tensor:
    """Pack one latent velocity in Bernini's official output-channel order."""

    latent = _exact81_latent(value, label="target velocity").squeeze(0)
    channels, phases, height, width = (int(item) for item in latent.shape)
    patches = (
        latent.reshape(channels, phases, height // 2, 2, width // 2, 2)
        .permute(1, 2, 4, 0, 3, 5)
        .reshape(phases * (height // 2) * (width // 2), channels, 1, 2, 2)
        .contiguous()
    )
    return source_runtime.packed_output_field(patches)


def predicted_clean_from_velocity(
    states: native.MultiSigmaStates, prediction: torch.Tensor
) -> torch.Tensor:
    """Apply the pinned flow-prediction identity ``z0=x_sigma-sigma*v``."""

    if (
        not isinstance(states, native.MultiSigmaStates)
        or not isinstance(prediction, torch.Tensor)
        or prediction.shape != states.noisy.shape
        or prediction.device != states.noisy.device
        or not prediction.is_floating_point()
        or not bool(torch.isfinite(prediction).all().item())
    ):
        raise BRAIDStageAError("predicted velocity must match the exact40 noisy states")
    sigma_shape = (len(states.indices),) + (1,) * (states.noisy.ndim - 1)
    sigma = states.sigmas.float().reshape(sigma_shape)
    return (states.noisy.float() - sigma * prediction.float()).contiguous()


@dataclass(frozen=True)
class WrongSourceAdmission:
    """Hash-bound data admission for an optional matched wrong source.

    The core can validate distinct identity/content and a shared declared
    semantic/camera-framing bucket.  It cannot establish those semantic facts;
    ``selection_evidence_sha256`` must therefore bind an external frozen data
    audit before this row is used for a wrong-source margin.
    """

    correct_source_video_sha256: str
    wrong_source_video_sha256: str
    correct_source_latent_sha256: str
    wrong_source_latent_sha256: str
    correct_reference_latent_sha256s: tuple[str, str, str, str]
    wrong_reference_latent_sha256s: tuple[str, str, str, str]
    correct_identity_group: str
    wrong_identity_group: str
    semantic_class_id: str
    scene_camera_bucket_id: str
    selection_evidence_sha256: str

    def __post_init__(self) -> None:
        correct = _sha256(
            self.correct_source_video_sha256, label="correct source video"
        )
        wrong = _sha256(self.wrong_source_video_sha256, label="wrong source video")
        if correct == wrong:
            raise BRAIDStageAError("correct and wrong source videos must differ")
        correct_latent = _sha256(
            self.correct_source_latent_sha256, label="correct source latent"
        )
        wrong_latent = _sha256(
            self.wrong_source_latent_sha256, label="wrong source latent"
        )
        if correct_latent == wrong_latent:
            raise BRAIDStageAError("correct and wrong source latents must differ")
        for label, values in (
            ("correct reference latent", self.correct_reference_latent_sha256s),
            ("wrong reference latent", self.wrong_reference_latent_sha256s),
        ):
            if not isinstance(values, tuple) or len(values) != REFERENCE_COUNT:
                raise BRAIDStageAError(f"{label} hashes require exactly four entries")
            for index, value in enumerate(values):
                _sha256(value, label=f"{label}[{index}]")
        correct_group = _safe_id(
            self.correct_identity_group, label="correct identity group"
        )
        wrong_group = _safe_id(self.wrong_identity_group, label="wrong identity group")
        if correct_group == wrong_group:
            raise BRAIDStageAError("wrong source must belong to a different identity group")
        _safe_id(self.semantic_class_id, label="semantic class")
        _safe_id(self.scene_camera_bucket_id, label="scene/camera-framing bucket")
        _sha256(self.selection_evidence_sha256, label="wrong-source selection evidence")

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "correct_source_video_sha256": self.correct_source_video_sha256,
            "wrong_source_video_sha256": self.wrong_source_video_sha256,
            "correct_source_latent_sha256": self.correct_source_latent_sha256,
            "wrong_source_latent_sha256": self.wrong_source_latent_sha256,
            "correct_reference_latent_sha256s": list(
                self.correct_reference_latent_sha256s
            ),
            "wrong_reference_latent_sha256s": list(
                self.wrong_reference_latent_sha256s
            ),
            "correct_identity_group": self.correct_identity_group,
            "wrong_identity_group": self.wrong_identity_group,
            "semantic_class_id": self.semantic_class_id,
            "scene_camera_bucket_id": self.scene_camera_bucket_id,
            "selection_evidence_sha256": self.selection_evidence_sha256,
            "same_declared_semantic_class": True,
            "same_declared_scene_camera_framing_bucket": True,
            "semantic_match_is_external_evidence_not_tensor_inference": True,
        }
        return {**value, "digest": object_sha256(value)}


@dataclass(frozen=True)
class NativeStageAQuerySet:
    correct: native.NativeRV2VBranch
    drop: native.NativeRV2VBranch
    wrong: Optional[native.NativeRV2VBranch]
    schedule_index: int
    timestep: int
    sigma: float
    batch_receipt_digest: str
    correct_pack_receipt_digest: str
    wrong_pack_receipt_digest: Optional[str]
    wrong_source_admission_digest: Optional[str]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.correct, native.NativeRV2VBranch)
            or not isinstance(self.drop, native.NativeRV2VBranch)
            or self.correct.name != "VI"
            or self.drop.name != "none"
            or (self.wrong is not None and self.wrong.name != "VI")
        ):
            raise BRAIDStageAError("Stage-A branches must be native VI/none/VI")
        if (
            isinstance(self.schedule_index, bool)
            or not isinstance(self.schedule_index, int)
            or not 0 <= self.schedule_index < 40
            or self.timestep != native.NATIVE_UNIPC40_TIMESTEPS[self.schedule_index]
            or self.sigma != native.NATIVE_UNIPC40_SIGMAS[self.schedule_index]
        ):
            raise BRAIDStageAError("Stage-A query coordinate is not pinned exact40")
        _sha256(self.batch_receipt_digest, label="teacher batch receipt")
        _sha256(self.correct_pack_receipt_digest, label="correct native pack receipt")
        if (self.wrong is None) != (self.wrong_pack_receipt_digest is None):
            raise BRAIDStageAError("wrong branch and native pack receipt differ")
        if (self.wrong is None) != (self.wrong_source_admission_digest is None):
            raise BRAIDStageAError("wrong branch and admission receipt differ")
        if self.wrong_pack_receipt_digest is not None:
            _sha256(self.wrong_pack_receipt_digest, label="wrong native pack receipt")
            _sha256(
                self.wrong_source_admission_digest,
                label="wrong-source admission receipt",
            )
        self._assert_shared_target_payload()

    @staticmethod
    def _target_latents(branch: native.NativeRV2VBranch) -> torch.Tensor:
        return branch.latents[:, branch.condition_tokens :, :]

    @staticmethod
    def _target_rotary(branch: native.NativeRV2VBranch) -> torch.Tensor:
        if int(branch.rotary.shape[2]) != branch.total_tokens:
            raise BRAIDStageAError("native rotary token dimension differs")
        return branch.rotary.narrow(
            2, branch.condition_tokens, branch.total_tokens - branch.condition_tokens
        )

    def _assert_shared_target_payload(self) -> None:
        reference_latent = self._target_latents(self.correct)
        reference_rotary = self._target_rotary(self.correct)
        for name, branch in self.branches().items():
            if (
                not torch.equal(self._target_latents(branch), reference_latent)
                or not torch.equal(self._target_rotary(branch), reference_rotary)
                or int(branch.target_mask.sum().item()) != int(reference_latent.shape[1])
            ):
                raise BRAIDStageAError(
                    f"{name} condition intervention changed the shared target payload"
                )

    def branches(self) -> Mapping[str, native.NativeRV2VBranch]:
        value: dict[str, native.NativeRV2VBranch] = {
            "correct": self.correct,
            "drop": self.drop,
        }
        if self.wrong is not None:
            value["wrong"] = self.wrong
        return value

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": SCHEMA_VERSION,
            "query_order": list(self.branches()),
            "native_branch_by_query": {
                "correct": "VI",
                "drop": "none",
                "wrong": None if self.wrong is None else "VI",
            },
            "schedule_index": self.schedule_index,
            "timestep": self.timestep,
            "sigma_float64_hex": float(self.sigma).hex(),
            "native_schedule_digest": native.PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST,
            "teacher_batch_receipt_digest": self.batch_receipt_digest,
            "correct_pack_receipt_digest": self.correct_pack_receipt_digest,
            "wrong_pack_receipt_digest": self.wrong_pack_receipt_digest,
            "wrong_source_admission_digest": self.wrong_source_admission_digest,
            "target_patch_and_rotary_shared_exactly": True,
            "same_text_condition_required_by_forward_api": True,
            "wrong_margin_present": self.wrong is not None,
        }
        return {**value, "digest": object_sha256(value)}


def build_native_stage_a_queries(
    transformer: Any,
    batch: TeacherForcedStageABatch,
    *,
    sigma_position: int,
    correct_references: Sequence[torch.Tensor],
    wrong_source: Optional[torch.Tensor] = None,
    wrong_references: Optional[Sequence[torch.Tensor]] = None,
    wrong_source_admission: Optional[WrongSourceAdmission] = None,
) -> NativeStageAQuerySet:
    """Build correct/drop/wrong queries through Bernini's native pack only."""

    if not isinstance(batch, TeacherForcedStageABatch):
        raise BRAIDStageAError("batch must be an exact Stage-A teacher batch")
    if (
        isinstance(sigma_position, bool)
        or not isinstance(sigma_position, int)
        or not 0 <= sigma_position < len(batch.states.indices)
    ):
        raise BRAIDStageAError("sigma position lies outside the teacher stratum")
    spatial = tuple(int(item) for item in batch.clean_source.shape[3:])
    correct_refs = _reference_latents(
        correct_references,
        spatial_shape=spatial,
        device=batch.clean_source.device,
        label="correct references",
    )
    noisy_target = batch.states.noisy[sigma_position].detach().contiguous()
    try:
        correct_pack = native.build_native_rv2v_pack(
            transformer,
            donor_video=batch.clean_source,
            image_references=correct_refs,
            noisy_target=noisy_target,
        )
    except native.NativeRefContrastiveV3Error as error:
        raise BRAIDStageAError(str(error)) from error

    wrong_inputs = (wrong_source, wrong_references, wrong_source_admission)
    if any(value is not None for value in wrong_inputs) and not all(
        value is not None for value in wrong_inputs
    ):
        raise BRAIDStageAError("wrong-source tensor, references, and admission are atomic")

    wrong_branch: Optional[native.NativeRV2VBranch] = None
    wrong_pack_digest: Optional[str] = None
    wrong_admission_digest: Optional[str] = None
    if wrong_source is not None:
        assert wrong_references is not None
        assert wrong_source_admission is not None
        wrong_clean = _exact81_latent(wrong_source, label="wrong source")
        if wrong_clean.shape != batch.clean_source.shape or wrong_clean.device != batch.clean_source.device:
            raise BRAIDStageAError("wrong source geometry/device differs from correct source")
        if source_runtime.tensor_sha256(wrong_clean) == source_runtime.tensor_sha256(
            batch.clean_source
        ):
            raise BRAIDStageAError("wrong-source latent aliases the correct source")
        if wrong_source_admission.correct_source_video_sha256 != batch.source_video_sha256:
            raise BRAIDStageAError("wrong-source admission names a different correct source")
        correct_latent_sha = source_runtime.tensor_sha256(batch.clean_source)
        wrong_latent_sha = source_runtime.tensor_sha256(wrong_clean)
        if (
            wrong_source_admission.correct_source_latent_sha256 != correct_latent_sha
            or wrong_source_admission.wrong_source_latent_sha256 != wrong_latent_sha
        ):
            raise BRAIDStageAError("wrong-source admission latent hashes differ")
        wrong_refs = _reference_latents(
            wrong_references,
            spatial_shape=spatial,
            device=batch.clean_source.device,
            label="wrong references",
        )
        correct_ref_shas = tuple(source_runtime.tensor_sha256(item) for item in correct_refs)
        wrong_ref_shas = tuple(source_runtime.tensor_sha256(item) for item in wrong_refs)
        if (
            wrong_source_admission.correct_reference_latent_sha256s
            != correct_ref_shas
            or wrong_source_admission.wrong_reference_latent_sha256s != wrong_ref_shas
        ):
            raise BRAIDStageAError("wrong-source admission reference hashes differ")
        try:
            wrong_pack = native.build_native_rv2v_pack(
                transformer,
                donor_video=wrong_clean,
                image_references=wrong_refs,
                noisy_target=noisy_target,
            )
        except native.NativeRefContrastiveV3Error as error:
            raise BRAIDStageAError(str(error)) from error
        wrong_branch = wrong_pack.video_image
        wrong_pack_digest = str(wrong_pack.receipt()["digest"])
        wrong_admission_digest = str(wrong_source_admission.receipt()["digest"])

    index = batch.states.indices[sigma_position]
    return NativeStageAQuerySet(
        correct=correct_pack.video_image,
        drop=correct_pack.none,
        wrong=wrong_branch,
        schedule_index=index,
        timestep=native.NATIVE_UNIPC40_TIMESTEPS[index],
        sigma=native.NATIVE_UNIPC40_SIGMAS[index],
        batch_receipt_digest=str(batch.receipt()["digest"]),
        correct_pack_receipt_digest=str(correct_pack.receipt()["digest"]),
        wrong_pack_receipt_digest=wrong_pack_digest,
        wrong_source_admission_digest=wrong_admission_digest,
    )


def forward_native_stage_a_queries(
    wan_diffusion: Any,
    queries: NativeStageAQuerySet,
    *,
    noop_cond_embeds: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Execute the fixed query order with one shared frozen no-op condition."""

    if not isinstance(queries, NativeStageAQuerySet):
        raise BRAIDStageAError("queries must be one closed native Stage-A set")
    device = queries.correct.latents.device
    timestep = torch.tensor([queries.timestep], dtype=torch.float32, device=device)
    predictions: dict[str, torch.Tensor] = {}
    try:
        for name, branch in queries.branches().items():
            predictions[name] = native.forward_native_target_branch(
                wan_diffusion,
                branch,
                timestep=timestep,
                cond_embeds=noop_cond_embeds,
            )
    except native.NativeRefContrastiveV3Error as error:
        raise BRAIDStageAError(str(error)) from error
    return predictions


@dataclass(frozen=True)
class Stage0AuthorizationToken:
    authorization_receipt_sha256: str
    authorized_shadow_updates: int
    method_source_revision: str
    exact81_registry_sha256: str


def verify_stage0_authorization(value: Any) -> Stage0AuthorizationToken:
    """Verify an explicit, sealed, all8 Stage-0 PASS receipt.

    This is a structural boundary, not a substitute for implementing the
    referenced oracle validators.  No positive receipt currently exists in
    this module or is synthesized from booleans here.
    """

    required = {
        "schema_version",
        "decision",
        "checkpoint_content_manifest_sha256",
        "bernini_revision",
        "veomni_revision",
        "method_source_revision",
        "exact81_registry_sha256",
        "native_schedule_digest",
        "parallel_contract",
        "canary_evidence",
        "optimizer_created",
        "parameter_updates_executed",
        "stage_a_shadow_updates_authorized",
        "authorization_receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise BRAIDStageAError("Stage-0 authorization has a non-closed schema")
    row = dict(value)
    seal = row.pop("authorization_receipt_sha256")
    if _sha256(seal, label="Stage-0 authorization receipt") != object_sha256(row):
        raise BRAIDStageAError("Stage-0 authorization seal differs")
    if (
        row["schema_version"] != STAGE0_AUTHORIZATION_SCHEMA_VERSION
        or row["decision"] != "AUTHORIZED_ONE_STAGE_A_SHADOW_UPDATE"
        or row["checkpoint_content_manifest_sha256"]
        != PINNED_CHECKPOINT_CONTENT_MANIFEST_SHA256
        or row["bernini_revision"] != PINNED_BERNINI_REVISION
        or row["veomni_revision"] != PINNED_VEOMNI_REVISION
        or row["native_schedule_digest"]
        != native.PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST
        or row["optimizer_created"] is not False
        or type(row["parameter_updates_executed"]) is not int
        or row["parameter_updates_executed"] != 0
        or type(row["stage_a_shadow_updates_authorized"]) is not int
        or row["stage_a_shadow_updates_authorized"] != 1
    ):
        raise BRAIDStageAError("Stage-0 authorization does not permit one clean shadow update")
    method_revision = _revision(row["method_source_revision"], label="method source")
    registry_sha = _sha256(row["exact81_registry_sha256"], label="exact81 registry")
    parallel = row["parallel_contract"]
    expected_parallel = {
        "node_count": 1,
        "visible_gpu_count": 8,
        "world_size": 8,
        "sequence_parallel_size": 4,
        "data_parallel_size": 2,
    }
    if (
        not isinstance(parallel, Mapping)
        or set(parallel) != set(expected_parallel)
        or any(type(parallel[key]) is not int for key in expected_parallel)
        or dict(parallel) != expected_parallel
    ):
        raise BRAIDStageAError("Stage-0 authorization is not native all8 DP2xSP4")
    evidence = row["canary_evidence"]
    if not isinstance(evidence, Mapping) or set(evidence) != set(REQUIRED_STAGE0_CANARIES):
        raise BRAIDStageAError("Stage-0 authorization lacks one of three required canaries")
    for name in REQUIRED_STAGE0_CANARIES:
        item = evidence[name]
        if not isinstance(item, Mapping) or set(item) != {
            "status",
            "job_id",
            "evidence_sha256",
            "checkpoint_content_manifest_sha256",
            "method_source_revision",
            "exact81_registry_sha256",
            "native_schedule_digest",
            "parameter_updates_executed",
        }:
            raise BRAIDStageAError(f"{name} evidence schema differs")
        if (
            item["status"] != "PASS"
            or isinstance(item["job_id"], bool)
            or not isinstance(item["job_id"], int)
            or item["job_id"] <= 0
            or item["checkpoint_content_manifest_sha256"]
            != PINNED_CHECKPOINT_CONTENT_MANIFEST_SHA256
            or item["method_source_revision"] != method_revision
            or item["exact81_registry_sha256"] != registry_sha
            or item["native_schedule_digest"]
            != native.PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST
            or type(item["parameter_updates_executed"]) is not int
            or item["parameter_updates_executed"] != 0
        ):
            raise BRAIDStageAError(f"{name} did not pass without updates")
        _sha256(item["evidence_sha256"], label=f"{name} evidence")
    return Stage0AuthorizationToken(str(seal), 1, method_revision, registry_sha)


@dataclass(frozen=True)
class StageASourceObjective:
    loss: torch.Tensor
    correct_error: torch.Tensor
    error_by_query: Mapping[str, torch.Tensor]
    gap_by_counterfactual: Mapping[str, torch.Tensor]
    hinge_by_counterfactual: Mapping[str, torch.Tensor]
    authorization_receipt_sha256: str
    wrong_source_margin_used: bool


def build_authorized_stage_a_objective(
    predictions: Mapping[str, torch.Tensor],
    target_velocity: torch.Tensor,
    *,
    sigma_weights: torch.Tensor,
    stage0_authorization: Mapping[str, Any],
    drop_margin: float = 0.05,
    wrong_margin: float = 0.05,
    fm_weight: float = 1.0,
    drop_weight: float = 1.0,
    wrong_weight: float = 1.0,
) -> StageASourceObjective:
    """Build FM + correct/drop/(optional wrong) loss after Stage-0 authorization."""

    token = verify_stage0_authorization(stage0_authorization)
    names = set(predictions)
    if names not in ({"correct", "drop"}, {"correct", "drop", "wrong"}):
        raise BRAIDStageAError("predictions require correct/drop and optional admitted wrong")
    if (
        not isinstance(target_velocity, torch.Tensor)
        or target_velocity.ndim < 2
        or not target_velocity.is_floating_point()
        or target_velocity.requires_grad
        or not bool(torch.isfinite(target_velocity).all().item())
    ):
        raise BRAIDStageAError("target velocity must be detached finite [S,...]")
    for name in names:
        prediction = predictions[name]
        if (
            not isinstance(prediction, torch.Tensor)
            or prediction.shape != target_velocity.shape
            or prediction.device != target_velocity.device
            or not prediction.is_floating_point()
            or not prediction.requires_grad
            or not bool(torch.isfinite(prediction).all().item())
        ):
            raise BRAIDStageAError(
                f"{name} prediction must be graph-connected with target geometry"
            )
    if (
        not isinstance(sigma_weights, torch.Tensor)
        or sigma_weights.ndim != 1
        or int(sigma_weights.numel()) != int(target_velocity.shape[0])
        or sigma_weights.device != target_velocity.device
        or not sigma_weights.is_floating_point()
        or bool((sigma_weights < 0).any().item())
        or not bool(torch.isfinite(sigma_weights).all().item())
        or float(sigma_weights.sum().item()) <= 0.0
    ):
        raise BRAIDStageAError("sigma weights must be finite nonnegative shared [S]")
    scalars = {
        "drop_margin": drop_margin,
        "wrong_margin": wrong_margin,
        "fm_weight": fm_weight,
        "drop_weight": drop_weight,
        "wrong_weight": wrong_weight,
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in scalars.values()
    ):
        raise BRAIDStageAError("Stage-A margins and weights must be finite positive scalars")

    reduce_dims = tuple(range(1, target_velocity.ndim))
    normalized = sigma_weights.float() / sigma_weights.float().sum()
    per_sigma = {
        name: (predictions[name].float() - target_velocity.float())
        .square()
        .mean(dim=reduce_dims)
        for name in predictions
    }
    errors = {name: (values * normalized).sum() for name, values in per_sigma.items()}
    correct = errors["correct"]
    gaps = {
        name: errors[name] - correct for name in ("drop", "wrong") if name in errors
    }
    margins = {"drop": float(drop_margin), "wrong": float(wrong_margin)}
    hinges = {
        name: torch.relu(correct.new_tensor(margins[name]) - gap)
        for name, gap in gaps.items()
    }
    loss = float(fm_weight) * correct + float(drop_weight) * hinges["drop"]
    if "wrong" in hinges:
        loss = loss + float(wrong_weight) * hinges["wrong"]
    if not loss.requires_grad or not bool(torch.isfinite(loss).item()):
        raise BRAIDStageAError("authorized Stage-A objective is detached or non-finite")
    return StageASourceObjective(
        loss=loss,
        correct_error=correct,
        error_by_query=errors,
        gap_by_counterfactual=gaps,
        hinge_by_counterfactual=hinges,
        authorization_receipt_sha256=token.authorization_receipt_sha256,
        wrong_source_margin_used="wrong" in errors,
    )


__all__ = [
    "BRAIDStageAError",
    "FRAME_COUNT",
    "FPS",
    "LATENT_CHANNELS",
    "LATENT_PHASES",
    "NativeStageAQuerySet",
    "PINNED_BERNINI_REVISION",
    "PINNED_CHECKPOINT_CONTENT_MANIFEST_SHA256",
    "PINNED_VEOMNI_REVISION",
    "QUERY_NAMES",
    "REQUIRED_STAGE0_CANARIES",
    "SCHEMA_VERSION",
    "STAGE0_AUTHORIZATION_SCHEMA_VERSION",
    "Stage0AuthorizationToken",
    "StageASourceObjective",
    "TeacherForcedStageABatch",
    "WrongSourceAdmission",
    "build_authorized_stage_a_objective",
    "build_native_stage_a_queries",
    "forward_native_stage_a_queries",
    "object_sha256",
    "pack_exact81_velocity",
    "predicted_clean_from_velocity",
    "prepare_teacher_forced_source_batch",
    "verify_stage0_authorization",
]
