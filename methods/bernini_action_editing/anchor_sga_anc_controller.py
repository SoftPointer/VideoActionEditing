"""Source-state SGA/ANC sampler with online pure-T2V action branches.

The outer edit state follows DynaEdit's clean-state algebra.  Depending on the
registered transport, the pure-T2V action source is consumed either per SGA
candidate by a block/target-state query, or once per active solver step by two
native action/no-op APG+UniPC trajectories shared across candidates.  The
trace records those scopes separately; positive ``transport_steps`` must never
be interpreted as proof that a particular block or per-candidate route ran.
"""

from __future__ import annotations

import copy
import contextlib
from dataclasses import dataclass
import math
from typing import Any, Optional

import torch
import torch.nn.functional as F

import anchor_cross_attention_transport as cross_transport
import anchor_qk_transport as qk_transport
import differential_sampler as cdf
import guided_source_aligned_controller as guided
import source_aligned_controller as source_aligned
import source_kv_replay as replay_runtime


ARMS = ("AQK_IID1", "AQK_ANC1", "AQK_AVG5", "AQK_SGA5")
FIELD_GUIDANCES = ("apg", "raw_conditional", "raw_cfg")
ANCHOR_CFG_SCOPES = ("shared", "target_conditional_only")
ANCHOR_CONTRAST_MODES = (
    "caption_noop_same_video",
    "dynamic_static_same_caption",
)
ANCHOR_SIGMA_CAPS = (0.6, 0.8, 1.0)
PRESERVATION_MODES = (
    "none",
    "source_motion_support",
    "source_motion_support_snapshot_residual",
    "source_motion_support_event01_object1",
    "source_motion_support_event01_actor_object",
)
SGA_SCORE_MODES = (
    "global_source_cosine",
    "background_source_cosine",
    "background_plus_anchor_action_002",
    "background_trust_anchor_action_003",
    "background_plus_anchor_envelope_005",
    "background_trust_anchor_envelope_003",
)
ANCHOR_CANDIDATE_MODES = ("single_shared", "bank_per_candidate")
ANCHOR_SPATIAL_ALIGNMENTS = ("none", "motion_support_affine")
PRESERVATION_KEEP_FRACTIONS = (0.10, 0.20, 0.30)
PRESERVATION_OUTSIDE_SCALES = (0.0, 0.05)
PRESERVATION_DILATIONS = (1, 2)
PRESERVATION_RESIDUAL_FRACTIONS = (0.0, 0.005, 0.01, 0.0125, 0.015, 0.02)
PRESERVATION_OBJECT_IDENTITY_STRENGTHS = (0.0, 0.025, 0.075)
MODEL_TIMESTEP_SCALE = 1000.0
FIELD_MODELS = (
    "source_conditioned_rv2v",
    "first_phase_source_rv2v",
    "first_phase_caption_i2v",
    "source_free_t2v",
)
FIELD_VELOCITY_RESIDUAL = "temporal_residual_velocity"
FIELD_CONTRAST_VELOCITY = "temporal_contrast_velocity"
FIELD_TARGET_CONTRAST_VELOCITY = "target_state_temporal_contrast_velocity"
FIELD_NATIVE_GATED_TARGET_CONTRAST_VELOCITY = (
    "native_gated_target_state_temporal_contrast_velocity"
)
FIELD_NATIVE_T2V_TARGET_VELOCITY_REPLACEMENT = (
    "native_t2v_target_velocity_replacement"
)
FIELD_NATIVE_T2V_DELTA_VELOCITY_REPLACEMENT = (
    "native_t2v_delta_velocity_replacement"
)
FIELD_NATIVE_T2V_TEMPORAL_DELTA_REPLACEMENT = (
    "native_t2v_temporal_delta_replacement"
)
FIELD_NATIVE_T2V_SPARSE25_TEMPORAL_DELTA_REPLACEMENT = (
    "native_t2v_sparse25_temporal_delta_replacement"
)
FIELD_NATIVE_TARGETSTATE_TEMPORAL_DELTA_REPLACEMENT = (
    "native_targetstate_temporal_delta_replacement"
)
FIELD_NATIVE_TARGETSTATE_SPARSE25_TEMPORAL_DELTA_REPLACEMENT = (
    "native_targetstate_sparse25_temporal_delta_replacement"
)
FIELD_TARGETSTATE_RAW_DELTA_REPLACEMENT = (
    "targetstate_raw_delta_replacement"
)
FIELD_TARGETSTATE_SPARSE25_RAW_DELTA_REPLACEMENT = (
    "targetstate_sparse25_raw_delta_replacement"
)
FIELD_NATIVE_TARGETSTATE_RAW_DELTA_REPLACEMENT = (
    "native_targetstate_raw_delta_replacement"
)
FIELD_NATIVE_TARGETSTATE_SPARSE25_RAW_DELTA_REPLACEMENT = (
    "native_targetstate_sparse25_raw_delta_replacement"
)
FIELD_NATIVE_ROLEWARP_TEMPORAL_DELTA_REPLACEMENT = (
    "native_rolewarp_temporal_delta_replacement"
)
FIELD_NATIVE_ROLEWARP_SPARSE25_TEMPORAL_DELTA_REPLACEMENT = (
    "native_rolewarp_sparse25_temporal_delta_replacement"
)
NATIVE_T2V_REPLACEMENT_TRANSPORTS = (
    FIELD_NATIVE_T2V_TARGET_VELOCITY_REPLACEMENT,
    FIELD_NATIVE_T2V_DELTA_VELOCITY_REPLACEMENT,
    FIELD_NATIVE_T2V_TEMPORAL_DELTA_REPLACEMENT,
    FIELD_NATIVE_T2V_SPARSE25_TEMPORAL_DELTA_REPLACEMENT,
)
TARGETSTATE_TEMPORAL_REPLACEMENT_TRANSPORTS = (
    FIELD_NATIVE_TARGETSTATE_TEMPORAL_DELTA_REPLACEMENT,
    FIELD_NATIVE_TARGETSTATE_SPARSE25_TEMPORAL_DELTA_REPLACEMENT,
)
TARGETSTATE_RAW_REPLACEMENT_TRANSPORTS = (
    FIELD_TARGETSTATE_RAW_DELTA_REPLACEMENT,
    FIELD_TARGETSTATE_SPARSE25_RAW_DELTA_REPLACEMENT,
    FIELD_NATIVE_TARGETSTATE_RAW_DELTA_REPLACEMENT,
    FIELD_NATIVE_TARGETSTATE_SPARSE25_RAW_DELTA_REPLACEMENT,
)
TARGETSTATE_HARD_REPLACEMENT_TRANSPORTS = (
    *TARGETSTATE_TEMPORAL_REPLACEMENT_TRANSPORTS,
    *TARGETSTATE_RAW_REPLACEMENT_TRANSPORTS,
)
ROLEWARP_REPLACEMENT_TRANSPORTS = (
    FIELD_NATIVE_ROLEWARP_TEMPORAL_DELTA_REPLACEMENT,
    FIELD_NATIVE_ROLEWARP_SPARSE25_TEMPORAL_DELTA_REPLACEMENT,
)
TARGET_STATE_FIELD_TRANSPORTS = (
    FIELD_TARGET_CONTRAST_VELOCITY,
    FIELD_NATIVE_GATED_TARGET_CONTRAST_VELOCITY,
    *NATIVE_T2V_REPLACEMENT_TRANSPORTS,
    *TARGETSTATE_HARD_REPLACEMENT_TRANSPORTS,
)
TRANSPORTS = qk_transport.TRANSPORTS + (
    cross_transport.TEMPORAL_CONTRAST_CROSS_ATTN_OUTPUT,
    FIELD_VELOCITY_RESIDUAL,
    FIELD_CONTRAST_VELOCITY,
    *TARGET_STATE_FIELD_TRANSPORTS,
    *ROLEWARP_REPLACEMENT_TRANSPORTS,
)
DYNAEDIT_SGA_TEMPERATURE = 0.01
SUPPORTED_SGA_TEMPERATURES = (0.001, DYNAEDIT_SGA_TEMPERATURE)
EARLY_CANDIDATE_STEPS = 3
EARLY_CANDIDATES = 5
SUPPORTED_EARLY_CANDIDATES = (5, 8, 12)
INITIAL_NOISE_PROPOSAL_MODES = (
    "keyed_only",
    "anchor_candidate0",
    "anchor_candidate0_forced",
)
ANCHOR_STATE_MODES = (
    "clean_noised",
    "native_t2v_trajectory",
)
ANCHOR_ACTION_KEEP_FRACTION = 0.25
ANCHOR_ACTION_ADDITIVE_WEIGHT = 0.02
ANCHOR_ACTION_BACKGROUND_TRUST = 0.003
ANCHOR_ENVELOPE_ADDITIVE_WEIGHT = 0.05
ANCHOR_ENVELOPE_CANONICAL_SIZE = 16


class AnchorSGAANCError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnchorSGAANCConfig:
    arm: str
    transport: str = qk_transport.HARD_QK
    selected_block_indices: tuple[int, ...] = qk_transport.DEFAULT_BLOCKS
    transport_strength: float = 1.0
    transport_steps: int = guided.EXPECTED_STEPS
    initial_phase_clamp: bool = True
    field_guidance: str = "apg"
    field_model: str = "source_conditioned_rv2v"
    source_cfg_scale: float = 1.0
    target_cfg_scale: float = 1.0
    anchor_cfg_scope: str = "shared"
    anchor_contrast_mode: str = "caption_noop_same_video"
    anchor_sigma_cap: float = 1.0
    preservation_mode: str = "none"
    preservation_keep_fraction: float = 0.20
    preservation_outside_scale: float = 0.0
    preservation_dilation: int = 1
    preservation_residual_fraction: float = 0.0
    preservation_object_identity_strength: float = 0.0
    preservation_start_step: int = 0
    preservation_ramp_steps: int = 1
    sga_score_mode: str = "global_source_cosine"
    anchor_candidate_mode: str = "single_shared"
    anchor_spatial_alignment: str = "none"
    seed: int = guided.EXPECTED_SEED
    num_inference_steps: int = guided.EXPECTED_STEPS
    flow_shift: float = guided.EXPECTED_FLOW_SHIFT
    sga_temperature: float = DYNAEDIT_SGA_TEMPERATURE
    early_candidate_count: int = EARLY_CANDIDATES
    initial_noise_proposal_mode: str = "keyed_only"
    anchor_state_mode: str = "clean_noised"
    anc_lock_sigma: float = guided.ANC_LOCK_SIGMA
    event01_forced_role_proposal_index: int = -1

    def validate(self) -> "AnchorSGAANCConfig":
        if self.arm not in ARMS:
            raise AnchorSGAANCError(f"arm must be one of {ARMS}")
        if self.transport not in TRANSPORTS:
            raise AnchorSGAANCError("unknown anchor attention transport")
        if (
            isinstance(self.event01_forced_role_proposal_index, bool)
            or not isinstance(self.event01_forced_role_proposal_index, int)
            or not -1
            <= self.event01_forced_role_proposal_index
            < qk_transport.EVENT01_ROLE_PROPOSALS
        ):
            raise AnchorSGAANCError(
                "event01_forced_role_proposal_index must be -1 or a valid proposal"
            )
        if (
            self.event01_forced_role_proposal_index >= 0
            and self.transport not in qk_transport.EVENT01_ROLE_GRAPH_TRANSPORTS
        ):
            raise AnchorSGAANCError(
                "a forced Event01 proposal requires an Event01 role transport"
            )
        if self.seed != guided.EXPECTED_SEED:
            raise AnchorSGAANCError("first anchor SGA/ANC canary uses the pinned seed")
        if self.num_inference_steps != guided.EXPECTED_STEPS:
            raise AnchorSGAANCError("first anchor SGA/ANC canary is exact40")
        if self.flow_shift != guided.EXPECTED_FLOW_SHIFT:
            raise AnchorSGAANCError("flow shift differs from the pinned scheduler")
        if self.sga_temperature not in SUPPORTED_SGA_TEMPERATURES:
            raise AnchorSGAANCError(
                "SGA temperature must be one of the preregistered DynaEdit values"
            )
        if self.early_candidate_count not in SUPPORTED_EARLY_CANDIDATES:
            raise AnchorSGAANCError(
                "early_candidate_count must be one of the preregistered pool sizes"
            )
        if self.arm not in ("AQK_AVG5", "AQK_SGA5") and self.early_candidate_count != 5:
            raise AnchorSGAANCError(
                "only multi-candidate arms may expand the early proposal pool"
            )
        if self.initial_noise_proposal_mode not in INITIAL_NOISE_PROPOSAL_MODES:
            raise AnchorSGAANCError("unknown initial-noise proposal mode")
        if self.anchor_state_mode not in ANCHOR_STATE_MODES:
            raise AnchorSGAANCError("unknown anchor-state mode")
        if (
            self.initial_noise_proposal_mode != "keyed_only"
            and self.arm != "AQK_SGA5"
        ):
            raise AnchorSGAANCError(
                "anchor-seeded noise proposal requires the SGA arm"
            )
        if self.anchor_state_mode == "native_t2v_trajectory":
            if self.anchor_candidate_mode != "single_shared":
                raise AnchorSGAANCError(
                    "native T2V trajectory replay requires one shared anchor"
                )
            if self.anchor_contrast_mode != "caption_noop_same_video":
                raise AnchorSGAANCError(
                    "native T2V trajectory replay requires action/no-op trajectory contrast"
                )
            if float(self.anchor_sigma_cap) != 1.0:
                raise AnchorSGAANCError(
                    "native T2V trajectory replay follows the exact outer schedule"
                )
        if (
            self.transport
            in (*TARGET_STATE_FIELD_TRANSPORTS, *ROLEWARP_REPLACEMENT_TRANSPORTS)
            and self.anchor_state_mode != "native_t2v_trajectory"
        ):
            raise AnchorSGAANCError(
                "target-state T2V contrast requires an audited native trajectory"
            )
        if (
            self.transport in NATIVE_T2V_REPLACEMENT_TRANSPORTS
            and self.initial_noise_proposal_mode == "keyed_only"
        ):
            raise AnchorSGAANCError(
                "native T2V replacement requires the audited anchor Gaussian in candidate 0"
            )
        if self.anc_lock_sigma != guided.ANC_LOCK_SIGMA:
            raise AnchorSGAANCError("ANC lock sigma differs from the matched control")
        qk_transport.AnchorQKCacheBank(self.selected_block_indices)
        if (
            isinstance(self.transport_strength, bool)
            or not isinstance(self.transport_strength, (int, float))
            or not math.isfinite(float(self.transport_strength))
            or not 0.0 < float(self.transport_strength) <= 1.0
        ):
            raise AnchorSGAANCError("transport_strength must be in (0,1]")
        if (
            self.transport
            in (
                *NATIVE_T2V_REPLACEMENT_TRANSPORTS,
                *TARGETSTATE_HARD_REPLACEMENT_TRANSPORTS,
                *ROLEWARP_REPLACEMENT_TRANSPORTS,
            )
            and float(self.transport_strength) != 1.0
        ):
            raise AnchorSGAANCError(
                "native T2V hard replacement requires strength 1"
            )
        if (
            self.transport in ROLEWARP_REPLACEMENT_TRANSPORTS
            and self.early_candidate_count != 5
        ):
            raise AnchorSGAANCError(
                "Event01 role-warp canary requires exactly five early proposals"
            )
        if (
            self.transport in qk_transport.EVENT01_ROLE_GRAPH_TRANSPORTS
            and (
                self.anchor_state_mode != "native_t2v_trajectory"
                or self.anchor_contrast_mode != "caption_noop_same_video"
                or self.early_candidate_count != 5
            )
        ):
            raise AnchorSGAANCError(
                "Event01 role graph requires native action/no-op trajectories and five proposals"
            )
        if (
            isinstance(self.transport_steps, bool)
            or not isinstance(self.transport_steps, int)
            or not 0 <= self.transport_steps <= self.num_inference_steps
        ):
            raise AnchorSGAANCError(
                "transport_steps must be an integer in [0,num_inference_steps]"
            )
        if not isinstance(self.initial_phase_clamp, bool):
            raise AnchorSGAANCError("initial_phase_clamp must be boolean")
        if self.field_guidance not in FIELD_GUIDANCES:
            raise AnchorSGAANCError(
                f"field_guidance must be one of {FIELD_GUIDANCES}"
            )
        if self.anchor_cfg_scope not in ANCHOR_CFG_SCOPES:
            raise AnchorSGAANCError(
                f"anchor_cfg_scope must be one of {ANCHOR_CFG_SCOPES}"
            )
        if (
            self.anchor_cfg_scope == "target_conditional_only"
            and self.field_guidance not in ("apg", "raw_cfg")
        ):
            raise AnchorSGAANCError(
                "target-conditional-only anchor routing requires a two-branch guidance"
            )
        if self.anchor_contrast_mode not in ANCHOR_CONTRAST_MODES:
            raise AnchorSGAANCError(
                f"anchor_contrast_mode must be one of {ANCHOR_CONTRAST_MODES}"
            )
        if (
            self.anchor_contrast_mode == "dynamic_static_same_caption"
            and self.transport
            not in (
                FIELD_CONTRAST_VELOCITY,
                qk_transport.TEMPORAL_CONTRAST_QK,
                qk_transport.TEMPORAL_CONTRAST_ATTN_OUTPUT,
                qk_transport.TEMPORAL_CORRESPONDENCE_CONTRAST_QK,
                qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK,
                qk_transport.TEMPORAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
                qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_ATTN_OUTPUT,
                qk_transport.TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_QK,
                qk_transport.TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
                qk_transport.HARD_PHASE_MEAN_CONTRAST_QK,
                qk_transport.HARD_PHASE_MEAN_CONTRAST_ATTN_OUTPUT,
                qk_transport.HARD_PREROPE_PHASE_MEAN_CONTRAST_QK,
                qk_transport.TEMPORAL_KERNEL_CONTRAST_ATTN_OUTPUT,
                *qk_transport.TARGET_OWNED_QK_TRANSPORTS_V14R2,
                *qk_transport.TARGET_GATED_HARD_KERNEL_TRANSPORTS,
                *qk_transport.EVENT01_ROLE_GRAPH_TRANSPORTS,
                cross_transport.TEMPORAL_CONTRAST_CROSS_ATTN_OUTPUT,
            )
        ):
            raise AnchorSGAANCError(
                "dynamic/static anchor contrast requires a contrast transport"
            )
        correspondence_contrast_transports = (
            qk_transport.TEMPORAL_CORRESPONDENCE_CONTRAST_QK,
            qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK,
            qk_transport.TEMPORAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
            qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_ATTN_OUTPUT,
            qk_transport.TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_QK,
            qk_transport.TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
        )
        if (
            self.transport in correspondence_contrast_transports
            and self.anchor_contrast_mode != "dynamic_static_same_caption"
        ):
            raise AnchorSGAANCError(
                "correspondence contrast requires a dynamic/static same-caption anchor"
            )
        if self.transport in (
            qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK,
            qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_ATTN_OUTPUT,
            qk_transport.HARD_PHASE_MEAN_CONTRAST_QK,
            qk_transport.HARD_PHASE_MEAN_CONTRAST_ATTN_OUTPUT,
            qk_transport.HARD_PREROPE_PHASE_MEAN_CONTRAST_QK,
        ) and float(self.transport_strength) != 1.0:
            raise AnchorSGAANCError(
                "hard temporal trajectory replacement requires strength 1"
            )
        if self.anchor_sigma_cap not in ANCHOR_SIGMA_CAPS:
            raise AnchorSGAANCError(
                f"anchor_sigma_cap must be one of {ANCHOR_SIGMA_CAPS}"
            )
        if self.preservation_mode not in PRESERVATION_MODES:
            raise AnchorSGAANCError(
                f"preservation_mode must be one of {PRESERVATION_MODES}"
            )
        if self.preservation_keep_fraction not in PRESERVATION_KEEP_FRACTIONS:
            raise AnchorSGAANCError(
                "preservation_keep_fraction is outside the preregistered values"
            )
        if self.preservation_outside_scale not in PRESERVATION_OUTSIDE_SCALES:
            raise AnchorSGAANCError(
                "preservation_outside_scale is outside the preregistered values"
            )
        if self.preservation_dilation not in PRESERVATION_DILATIONS:
            raise AnchorSGAANCError(
                "preservation_dilation is outside the preregistered values"
            )
        if self.preservation_residual_fraction not in PRESERVATION_RESIDUAL_FRACTIONS:
            raise AnchorSGAANCError(
                "preservation_residual_fraction is outside the preregistered values"
            )
        if (
            self.preservation_object_identity_strength
            not in PRESERVATION_OBJECT_IDENTITY_STRENGTHS
        ):
            raise AnchorSGAANCError(
                "preservation_object_identity_strength is outside the "
                "preregistered values"
            )
        if (
            self.preservation_object_identity_strength > 0.0
            and self.preservation_mode
            not in (
                "source_motion_support_event01_object1",
                "source_motion_support_event01_actor_object",
            )
        ):
            raise AnchorSGAANCError(
                "object identity projection requires the Event01 object corridor"
            )
        for label, value in (
            ("preservation_start_step", self.preservation_start_step),
            ("preservation_ramp_steps", self.preservation_ramp_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise AnchorSGAANCError(f"{label} must be an integer")
        if not 0 <= self.preservation_start_step < self.num_inference_steps:
            raise AnchorSGAANCError(
                "preservation_start_step must be in [0,num_inference_steps)"
            )
        if not 1 <= self.preservation_ramp_steps <= self.num_inference_steps:
            raise AnchorSGAANCError(
                "preservation_ramp_steps must be in [1,num_inference_steps]"
            )
        if self.sga_score_mode not in SGA_SCORE_MODES:
            raise AnchorSGAANCError(
                f"sga_score_mode must be one of {SGA_SCORE_MODES}"
            )
        if (
            self.sga_score_mode
            in (
                "background_source_cosine",
                "background_plus_anchor_action_002",
                "background_trust_anchor_action_003",
                "background_plus_anchor_envelope_005",
                "background_trust_anchor_envelope_003",
            )
            and self.preservation_mode
            not in (
                "source_motion_support",
                "source_motion_support_snapshot_residual",
                "source_motion_support_event01_object1",
                "source_motion_support_event01_actor_object",
            )
        ):
            raise AnchorSGAANCError(
                "background-masked SGA requires source-motion preservation support"
            )
        if self.anchor_candidate_mode not in ANCHOR_CANDIDATE_MODES:
            raise AnchorSGAANCError(
                f"anchor_candidate_mode must be one of {ANCHOR_CANDIDATE_MODES}"
            )
        if (
            self.sga_score_mode
            in (
                "background_plus_anchor_action_002",
                "background_trust_anchor_action_003",
                "background_plus_anchor_envelope_005",
                "background_trust_anchor_envelope_003",
            )
            and self.anchor_candidate_mode != "single_shared"
        ):
            raise AnchorSGAANCError(
                "anchor-action SGA reward requires one fixed action anchor"
            )
        if self.anchor_candidate_mode == "bank_per_candidate" and self.arm not in (
            "AQK_AVG5",
            "AQK_SGA5",
        ):
            raise AnchorSGAANCError(
                "an anchor candidate bank requires a multi-candidate SGA/AVG arm"
            )
        if self.anchor_spatial_alignment not in ANCHOR_SPATIAL_ALIGNMENTS:
            raise AnchorSGAANCError(
                f"anchor_spatial_alignment must be one of {ANCHOR_SPATIAL_ALIGNMENTS}"
            )
        spatial_alignment_contrast_is_valid = (
            self.anchor_contrast_mode == "dynamic_static_same_caption"
            or (
                self.anchor_state_mode == "native_t2v_trajectory"
                and self.anchor_contrast_mode == "caption_noop_same_video"
            )
        )
        if self.anchor_spatial_alignment != "none" and (
            self.transport != FIELD_CONTRAST_VELOCITY
            or not spatial_alignment_contrast_is_valid
            or self.preservation_mode != "source_motion_support"
        ):
            raise AnchorSGAANCError(
                "anchor spatial alignment requires an explicit high-dimensional "
                "action contrast and source-motion support"
            )
        if self.field_model not in FIELD_MODELS:
            raise AnchorSGAANCError(f"field_model must be one of {FIELD_MODELS}")
        if self.transport in qk_transport.DUAL_SOURCE_KV_TRANSPORTS:
            if self.field_model == "source_free_t2v":
                raise AnchorSGAANCError(
                    "dual source K/V replay requires a paired source/target field"
                )
            if self.selected_block_indices != tuple(range(4, 30)):
                raise AnchorSGAANCError(
                    "dual route requires the audited contiguous block seam 4..29"
                )
        if self.field_model in (
            "first_phase_caption_i2v",
            "source_free_t2v",
        ) and self.field_guidance not in ("raw_conditional", "raw_cfg"):
            raise AnchorSGAANCError(
                "caption generation fields require raw conditional or raw CFG guidance"
            )
        for label, value in (
            ("source_cfg_scale", self.source_cfg_scale),
            ("target_cfg_scale", self.target_cfg_scale),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 < float(value) <= 20.0
            ):
                raise AnchorSGAANCError(f"{label} must be in (0,20]")
        if self.field_guidance == "raw_cfg" and (
            self.source_cfg_scale <= 1.0 or self.target_cfg_scale <= 1.0
        ):
            raise AnchorSGAANCError("raw CFG scales must both be greater than one")
        return self

    @property
    def uses_anc(self) -> bool:
        return self.arm != "AQK_IID1"

    @property
    def aggregation(self) -> str:
        if self.arm == "AQK_AVG5":
            return "uniform"
        if self.arm == "AQK_SGA5":
            return "sga_cosine"
        return "single"

    def candidate_count(self, step_index: int) -> int:
        if self.arm in ("AQK_AVG5", "AQK_SGA5") and step_index < EARLY_CANDIDATE_STEPS:
            return self.early_candidate_count
        return 1

    @property
    def uses_rv2v_condition(self) -> bool:
        return self.field_model in (
            "source_conditioned_rv2v",
            "first_phase_source_rv2v",
            "first_phase_caption_i2v",
        )

    @property
    def uses_source_target_captions(self) -> bool:
        return self.field_model in (
            "first_phase_caption_i2v",
            "source_free_t2v",
        )


def _parallel_identity() -> tuple[int, int]:
    try:
        from bernini.parallel import get_parallel_state

        state = get_parallel_state()
    except (ImportError, AttributeError):
        return 0, 1
    _, rank, size = replay_runtime.parallel_identity(state)
    return rank, size


def _frozen_anchor_adapter_context(adapter_controller: Any) -> Any:
    """Disable a loaded editor LoRA only while querying the T2V anchor teacher.

    A plain frozen Bernini renderer has no ``disable_adapter`` method and needs
    no special context.  Inference with a trained PEFT editor keeps the adapter
    unmerged, so every source-free action/no-op teacher call can recover the
    exact frozen base without changing the target/source editor calls.
    """

    if adapter_controller is None:
        return contextlib.nullcontext()
    disabled = getattr(adapter_controller, "disable_adapter", None)
    return disabled() if callable(disabled) else contextlib.nullcontext()


def _capture_anchor_qk(
    *,
    diffusion: Any,
    transformer: Any,
    anchor_state: torch.Tensor,
    anchor_prompt_embeds: torch.Tensor,
    timestep: torch.Tensor,
    cache_bank: qk_transport.AnchorQKCacheBank,
    step_index: int,
    candidate_index: int,
    transport: str,
    transport_strength: float,
    replay_uses: int,
    replay_scope: str,
    adapter_controller: Any,
    slot: str = qk_transport.ACTION_SLOT,
    role_proposal_index: int = 0,
) -> torch.Tensor:
    compute_dtype = cdf._module_dtype(transformer, anchor_state.dtype)
    tokens, rotary = transformer.patch_vae_latent(
        anchor_state.to(dtype=compute_dtype), source_id=cdf.QUERY_ID
    )
    text = anchor_prompt_embeds.to(
        device=anchor_state.device, dtype=compute_dtype
    )
    rank, size = _parallel_identity()
    invocation = qk_transport.AnchorQKInvocation(
        qk_transport.CAPTURE,
        cache_bank,
        step_index=step_index,
        candidate_index=candidate_index,
        rank=rank,
        ulysses_size=size,
        transport=transport,
        transport_strength=transport_strength,
        replay_uses=replay_uses,
        replay_scope=replay_scope,
        slot=slot,
        role_proposal_index=role_proposal_index,
    )
    with _frozen_anchor_adapter_context(adapter_controller), qk_transport.anchor_qk_invocation(invocation):
        prediction = diffusion.shared_step(
            model_id="transformer_1",
            noisy_latents=tokens,
            timesteps=timestep.expand(1),
            cond_embeds=text,
            rotary_embs=rotary,
            batch_vae_seqlen=[int(tokens.shape[1])],
            batch_text_seqlen=[int(text.shape[1])],
        )
    expected = (1, cdf.validate_latent_shape(tuple(anchor_state.shape)).tokens, 64)
    if tuple(int(item) for item in prediction.shape) != expected:
        raise AnchorSGAANCError("pure-T2V anchor prediction geometry differs")
    return prediction.float()


def _capture_anchor_cross_attention(
    *,
    diffusion: Any,
    transformer: Any,
    anchor_state: torch.Tensor,
    anchor_prompt_embeds: torch.Tensor,
    timestep: torch.Tensor,
    cache_bank: cross_transport.AnchorCrossAttentionCache,
    step_index: int,
    candidate_index: int,
    transport_strength: float,
    replay_uses: int,
    replay_scope: str,
    adapter_controller: Any,
    slot: str = cross_transport.ACTION_SLOT,
) -> torch.Tensor:
    compute_dtype = cdf._module_dtype(transformer, anchor_state.dtype)
    tokens, rotary = transformer.patch_vae_latent(
        anchor_state.to(dtype=compute_dtype), source_id=cdf.QUERY_ID
    )
    text = anchor_prompt_embeds.to(
        device=anchor_state.device, dtype=compute_dtype
    )
    rank, size = _parallel_identity()
    invocation = cross_transport.AnchorCrossAttentionInvocation(
        cross_transport.CAPTURE,
        cache_bank,
        step_index=step_index,
        candidate_index=candidate_index,
        rank=rank,
        ulysses_size=size,
        transport_strength=transport_strength,
        replay_uses=replay_uses,
        replay_scope=replay_scope,
        slot=slot,
    )
    with _frozen_anchor_adapter_context(adapter_controller), cross_transport.anchor_cross_attention_invocation(invocation):
        prediction = diffusion.shared_step(
            model_id="transformer_1",
            noisy_latents=tokens,
            timesteps=timestep.expand(1),
            cond_embeds=text,
            rotary_embs=rotary,
            batch_vae_seqlen=[int(tokens.shape[1])],
            batch_text_seqlen=[int(text.shape[1])],
        )
    expected = (1, cdf.validate_latent_shape(tuple(anchor_state.shape)).tokens, 64)
    if tuple(int(item) for item in prediction.shape) != expected:
        raise AnchorSGAANCError("pure-T2V anchor cross-attention prediction geometry differs")
    return prediction.float()


def _predict_source_free_velocity(
    *,
    diffusion: Any,
    transformer: Any,
    query_state: torch.Tensor,
    prompt_embeds: torch.Tensor,
    timestep: torch.Tensor,
    adapter_controller: Any = None,
) -> torch.Tensor:
    """Query Bernini's generation field without an RV2V source prefix."""

    compute_dtype = cdf._module_dtype(transformer, query_state.dtype)
    tokens, rotary = transformer.patch_vae_latent(
        query_state.to(dtype=compute_dtype), source_id=cdf.QUERY_ID
    )
    text = prompt_embeds.to(device=query_state.device, dtype=compute_dtype)
    with _frozen_anchor_adapter_context(adapter_controller):
        prediction = diffusion.shared_step(
            model_id="transformer_1",
            noisy_latents=tokens,
            timesteps=timestep.expand(1),
            cond_embeds=text,
            rotary_embs=rotary,
            batch_vae_seqlen=[int(tokens.shape[1])],
            batch_text_seqlen=[int(text.shape[1])],
        )
    expected = (1, cdf.validate_latent_shape(tuple(query_state.shape)).tokens, 64)
    if tuple(int(item) for item in prediction.shape) != expected:
        raise AnchorSGAANCError("source-free T2V field geometry differs")
    return prediction.float()


def _guided_source_free_apg_velocity(
    *,
    diffusion: Any,
    transformer: Any,
    query_state: torch.Tensor,
    condition_prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    timestep: torch.Tensor,
    sigma: torch.Tensor,
    branch: str,
    adapter_controller: Any,
) -> torch.Tensor:
    """Replay Bernini's native momentum-zero ``t2v_apg`` velocity exactly.

    This is deliberately separate from the edited target field.  It evolves
    the self-generated action/no-op anchor trajectories with their own UniPC
    histories, so block capture observes the actual generation-time states
    rather than a clean endpoint mixed with an unrelated FlowEdit noise.
    """

    if branch not in (
        "anchor_action_trajectory",
        "anchor_noop_trajectory",
        "anchor_reverse_trajectory",
        "anchor_static_trajectory",
    ):
        raise AnchorSGAANCError("unknown native T2V trajectory branch")
    if not isinstance(query_state, torch.Tensor) or query_state.dtype != torch.float32:
        raise AnchorSGAANCError("native T2V trajectory state must be FP32")
    raw_negative = _predict_source_free_velocity(
        diffusion=diffusion,
        transformer=transformer,
        query_state=query_state,
        prompt_embeds=negative_prompt_embeds,
        timestep=timestep,
        adapter_controller=adapter_controller,
    )
    raw_condition = _predict_source_free_velocity(
        diffusion=diffusion,
        transformer=transformer,
        query_state=query_state,
        prompt_embeds=condition_prompt_embeds,
        timestep=timestep,
        adapter_controller=adapter_controller,
    )
    layout = cdf.validate_latent_shape(tuple(int(value) for value in query_state.shape))
    negative_velocity = cdf._unpack_spatial_latent(raw_negative, layout)
    condition_velocity = cdf._unpack_spatial_latent(raw_condition, layout)
    sigma_tensor = guided._validate_sigma_cpu_fp32(sigma)
    negative_clean = query_state - sigma_tensor * negative_velocity
    condition_clean = query_state - sigma_tensor * condition_velocity
    momentum = guided.tri._MomentumBuffer(guided.APG_MOMENTUM, branch=branch)
    guided_clean = guided.tri._normalized_guidance(
        condition_clean,
        negative_clean,
        guided.APG_GUIDANCE_SCALE,
        momentum,
        guided.APG_ETA,
        guided.APG_NORM_THRESHOLD,
    )
    if momentum.update_count != 1 or momentum.momentum != 0.0:
        raise AnchorSGAANCError("native T2V APG must remain momentum-zero")
    guided_velocity = (query_state - guided_clean) / sigma_tensor
    packed = cdf._pack_spatial_latent(guided_velocity.float(), layout)
    if packed.dtype != torch.float32 or not bool(torch.isfinite(packed).all()):
        raise AnchorSGAANCError("native T2V APG velocity is invalid")
    return packed


def _native_unipc_step(
    scheduler: Any,
    *,
    velocity_packed: torch.Tensor,
    timestep: torch.Tensor,
    state_packed: torch.Tensor,
) -> torch.Tensor:
    result = scheduler.step(
        velocity_packed,
        timestep,
        state_packed,
        return_dict=False,
    )
    if (
        not isinstance(result, (tuple, list))
        or not result
        or not isinstance(result[0], torch.Tensor)
    ):
        raise AnchorSGAANCError("native T2V UniPC return ABI differs")
    state = result[0].detach().float().contiguous()
    if tuple(state.shape) != tuple(state_packed.shape) or not bool(
        torch.isfinite(state).all()
    ):
        raise AnchorSGAANCError("native T2V UniPC state is invalid")
    return state


def _model_timestep_for_anchor_sigma(
    *,
    outer_timestep: torch.Tensor,
    outer_sigma: float,
    anchor_sigma: float,
    num_train_timesteps: int,
) -> torch.Tensor:
    """Keep the teacher's noisy latent and model timestep on the same FM state.

    Bernini's pinned flow-matching model time has 1000 training steps.  The
    live UniPC schedule stores integer outer timesteps, so an uncapped teacher
    reuses that exact tensor.  A preregistered cap (.8/.6) maps exactly to
    timestep 800/600.  Capping only the latent mixture while retaining the
    outer timestep would query the teacher off-manifold.
    """

    if (
        not isinstance(outer_timestep, torch.Tensor)
        or outer_timestep.ndim != 0
        or outer_timestep.dtype == torch.bool
        or not (outer_timestep.dtype.is_floating_point or outer_timestep.dtype in (torch.int32, torch.int64))
        or not bool(torch.isfinite(outer_timestep))
    ):
        raise AnchorSGAANCError("outer model timestep must be one finite numeric scalar")
    for label, value in (("outer_sigma", outer_sigma), ("anchor_sigma", anchor_sigma)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise AnchorSGAANCError(f"{label} must be a finite scalar in [0,1]")
    if float(anchor_sigma) > float(outer_sigma) + 1.0e-7:
        raise AnchorSGAANCError("anchor sigma cannot exceed the outer sigma")
    if (
        isinstance(num_train_timesteps, bool)
        or not isinstance(num_train_timesteps, int)
        or num_train_timesteps != int(MODEL_TIMESTEP_SCALE)
    ):
        raise AnchorSGAANCError("teacher requires the pinned 1000-step FM model time")
    if math.isclose(
        float(anchor_sigma), float(outer_sigma), rel_tol=0.0, abs_tol=0.0
    ):
        return outer_timestep
    model_time = float(anchor_sigma) * float(num_train_timesteps)
    if not model_time.is_integer():
        raise AnchorSGAANCError("anchor sigma cap must map to an exact model timestep")
    return outer_timestep.new_tensor(int(model_time))


def _predict_field_velocity(
    *,
    diffusion: Any,
    transformer: Any,
    source_condition: Any,
    query_state: torch.Tensor,
    prompt_embeds: torch.Tensor,
    timestep: torch.Tensor,
    uses_rv2v_condition: bool,
) -> torch.Tensor:
    if uses_rv2v_condition:
        return cdf._predict_source_conditioned_velocity(
            diffusion=diffusion,
            transformer=transformer,
            source_condition=source_condition,
            query_latent=query_state,
            prompt_embeds=prompt_embeds,
            timestep=timestep,
        ).float()
    return _predict_source_free_velocity(
        diffusion=diffusion,
        transformer=transformer,
        query_state=query_state,
        prompt_embeds=prompt_embeds,
        timestep=timestep,
    )


def _clamp_initial_latent_phase(
    edit_packed: torch.Tensor,
    source_packed: torch.Tensor,
    layout: cdf.LatentLayout,
) -> None:
    """Keep the complete first VAE phase byte-equal to the source after updates.

    Bernini packs latent video tokens in frame-major order.  Clamping the first
    spatial token slab therefore protects the actual initial video state, not a
    scalar statistic or an appearance reward.  The operation is in-place so it
    is applied after every edit-ODE update before the next model query.
    """

    if (
        not isinstance(edit_packed, torch.Tensor)
        or not isinstance(source_packed, torch.Tensor)
        or tuple(edit_packed.shape) != tuple(source_packed.shape)
        or edit_packed.ndim != 3
        or layout.frames <= 0
        or layout.tokens % layout.frames
    ):
        raise AnchorSGAANCError("initial-phase clamp geometry differs")
    spatial_tokens = layout.tokens // layout.frames
    edit_packed[:, :spatial_tokens].copy_(source_packed[:, :spatial_tokens])


def _source_motion_support(
    source_clean: torch.Tensor,
    *,
    keep_fraction: float,
    dilation: int,
) -> torch.Tensor:
    """Return a dense spatial support for source subject motion.

    The mask is computed from every source latent channel and phase.  It is not
    an endpoint score or a compressed motion code: the maximum temporal-change
    energy selects source pixels over the whole clip, then a spatial dilation
    includes contact neighborhoods around the moving subject.
    """

    if (
        not isinstance(source_clean, torch.Tensor)
        or source_clean.ndim != 5
        or int(source_clean.shape[0]) != 1
        or int(source_clean.shape[2]) != qk_transport.LATENT_PHASES
    ):
        raise AnchorSGAANCError("source motion support requires one 21-phase latent")
    delta = source_clean[:, :, 1:].float() - source_clean[:, :, :-1].float()
    energy = delta.square().mean(dim=1).amax(dim=1)
    flat = energy.flatten(1)
    keep = max(1, math.ceil(int(flat.shape[1]) * float(keep_fraction)))
    top = torch.topk(flat, k=keep, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, top, True)
    mask = mask.reshape_as(energy).float().unsqueeze(1)
    kernel = 2 * int(dilation) + 1
    mask = torch.nn.functional.max_pool2d(
        mask,
        kernel_size=kernel,
        stride=1,
        padding=int(dilation),
    )
    return mask.to(dtype=source_clean.dtype).unsqueeze(2)


def _effective_source_edit_support(
    edit: torch.Tensor,
    source_clean: torch.Tensor,
    support: torch.Tensor,
    *,
    residual_fraction: float = 0.0,
) -> torch.Tensor:
    """Union source motion with a sparse per-phase candidate edit support."""

    if (
        tuple(edit.shape) != tuple(source_clean.shape)
        or support.ndim != 5
        or tuple(support.shape[:2]) != (int(edit.shape[0]), 1)
        or int(support.shape[2]) not in (1, int(edit.shape[2]))
        or tuple(support.shape[-2:]) != tuple(edit.shape[-2:])
    ):
        raise AnchorSGAANCError("source preservation support geometry differs")
    effective_support = support
    if float(residual_fraction) > 0.0:
        residual_energy = (edit.float() - source_clean.float()).square().mean(
            dim=1, keepdim=True
        )
        outside_energy = residual_energy * (1.0 - support.float())
        batch, _, phases, height, width = outside_energy.shape
        flat = outside_energy.permute(0, 2, 1, 3, 4).reshape(
            batch * phases, height * width
        )
        keep = max(1, math.ceil(height * width * float(residual_fraction)))
        top = torch.topk(flat, k=keep, dim=1, largest=True, sorted=False).indices
        adaptive = torch.zeros_like(flat, dtype=torch.bool)
        adaptive.scatter_(1, top, True)
        adaptive = adaptive.reshape(batch, phases, 1, height, width).permute(
            0, 2, 1, 3, 4
        )
        effective_support = torch.maximum(
            support,
            adaptive.to(dtype=support.dtype),
        )
    return effective_support


def _apply_source_motion_preservation(
    edit_packed: torch.Tensor,
    source_clean: torch.Tensor,
    support: torch.Tensor,
    *,
    layout: cdf.LatentLayout,
    outside_scale: float,
    residual_fraction: float = 0.0,
) -> torch.Tensor:
    """Keep source outside motion support plus a sparse per-phase edit residual.

    The optional residual support lets a small initially-static contacted
    object become dynamic. It is selected independently in every latent phase
    from the full edit displacement outside the source-motion support, so it
    cannot expand into an unrestricted global target mask.
    """

    edit = cdf._unpack_spatial_latent(edit_packed, layout)
    effective_support = _effective_source_edit_support(
        edit,
        source_clean,
        support,
        residual_fraction=residual_fraction,
    )
    scale = effective_support + (1.0 - effective_support) * float(outside_scale)
    preserved = source_clean + scale * (edit - source_clean)
    preserved[:, :, 0].copy_(source_clean[:, :, 0])
    return cdf._pack_spatial_latent(preserved, layout)


def _event01_object1_phasewise_preservation_support(
    source_support: torch.Tensor,
) -> torch.Tensor:
    """Add a narrow phase-wise source-stone-to-hand interaction corridor."""

    if (
        not isinstance(source_support, torch.Tensor)
        or source_support.ndim != 5
        or int(source_support.shape[1]) != 1
        or int(source_support.shape[2]) != 1
    ):
        raise AnchorSGAANCError("Event01 preservation base support differs")
    batch, _, _, height, width = tuple(int(item) for item in source_support.shape)
    if height <= 0 or width <= 0:
        raise AnchorSGAANCError("Event01 preservation grid is empty")
    centers = qk_transport._event01_dynamic_target_centers(1)
    if len(centers) != qk_transport.LATENT_PHASES:
        raise AnchorSGAANCError("Event01 object corridor phase count differs")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=source_support.device, dtype=torch.float32),
        torch.arange(width, device=source_support.device, dtype=torch.float32),
        indexing="ij",
    )
    scale_x = float(width) / float(qk_transport.EVENT01_SPATIAL_WIDTH)
    scale_y = float(height) / float(qk_transport.EVENT01_SPATIAL_HEIGHT)
    radius_x = 2.0 * scale_x
    radius_y = 1.75 * scale_y
    phase_support = source_support.repeat(1, 1, qk_transport.LATENT_PHASES, 1, 1)
    source_object = qk_transport.EVENT01_SOURCE_OBJECT_PROPOSALS_XY[1]
    source_distance = (
        ((xx - float(source_object[0]) * scale_x) / radius_x).square()
        + ((yy - float(source_object[1]) * scale_y) / radius_y).square()
    ).sqrt()
    # Keep the vacated source site editable after pickup. Otherwise the hard
    # source projection restores a second stationary copy of the stone.
    vacancy = (2.0 - source_distance).clamp(0.0, 1.0).reshape(
        1, 1, 1, height, width
    )
    phase_support = torch.maximum(
        phase_support,
        vacancy.to(source_support.dtype),
    )
    for phase_index, (_actor_xy, object_xy) in enumerate(centers):
        distance = (
            ((xx - float(object_xy[0]) * scale_x) / radius_x).square()
            + ((yy - float(object_xy[1]) * scale_y) / radius_y).square()
        ).sqrt()
        corridor = (2.0 - distance).clamp(0.0, 1.0)
        phase_support[:, :, phase_index] = torch.maximum(
            phase_support[:, :, phase_index],
            corridor.reshape(1, 1, height, width).to(source_support.dtype),
        )
    return phase_support


def _expected_event01_early_role_proposals(
    runtime: AnchorSGAANCConfig,
) -> list[int]:
    """Receipt closure for automatic five-way or one forced role proposal."""

    if runtime.event01_forced_role_proposal_index >= 0:
        return [runtime.event01_forced_role_proposal_index] * (
            qk_transport.EVENT01_ROLE_PROPOSALS * EARLY_CANDIDATE_STEPS
        )
    return (
        list(range(qk_transport.EVENT01_ROLE_PROPOSALS))
        * EARLY_CANDIDATE_STEPS
    )


def _event01_actor_object_phasewise_preservation_support(
    source_support: torch.Tensor,
) -> torch.Tensor:
    """Open the full new-action actor region plus the audited object tube."""

    support = _event01_object1_phasewise_preservation_support(source_support)
    batch, _, phases, height, width = (int(item) for item in support.shape)
    centers = qk_transport._event01_dynamic_target_centers(1)
    if phases != len(centers):
        raise AnchorSGAANCError("Event01 actor/object support phase count differs")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=support.device, dtype=torch.float32),
        torch.arange(width, device=support.device, dtype=torch.float32),
        indexing="ij",
    )
    scale_x = float(width) / float(qk_transport.EVENT01_SPATIAL_WIDTH)
    scale_y = float(height) / float(qk_transport.EVENT01_SPATIAL_HEIGHT)
    radius_x = 5.0 * scale_x
    radius_y = 10.0 * scale_y
    for phase_index, (actor_xy, _object_xy) in enumerate(centers):
        distance = (
            ((xx - float(actor_xy[0]) * scale_x) / radius_x).square()
            + ((yy - float(actor_xy[1]) * scale_y) / radius_y).square()
        ).sqrt()
        actor_support = (1.25 - distance).clamp(0.0, 1.0)
        support[:, :, phase_index] = torch.maximum(
            support[:, :, phase_index],
            actor_support.reshape(1, 1, height, width).to(support.dtype),
        )
    return support


def _event01_object1_phasewise_source_reference(
    source_clean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate only the audited phase-0 source-stone latent along its path."""

    if (
        not isinstance(source_clean, torch.Tensor)
        or source_clean.ndim != 5
        or int(source_clean.shape[2]) != qk_transport.LATENT_PHASES
    ):
        raise AnchorSGAANCError("Event01 source reference geometry differs")
    batch, channels, phases, height, width = (
        int(item) for item in source_clean.shape
    )
    if height < 2 or width < 2:
        raise AnchorSGAANCError("Event01 source reference grid is too small")
    centers = qk_transport._event01_dynamic_target_centers(1)
    scale_x = float(width) / float(qk_transport.EVENT01_SPATIAL_WIDTH)
    scale_y = float(height) / float(qk_transport.EVENT01_SPATIAL_HEIGHT)
    source_xy = qk_transport.EVENT01_SOURCE_OBJECT_PROPOSALS_XY[1]
    source_x = float(source_xy[0]) * scale_x
    source_y = float(source_xy[1]) * scale_y
    yy, xx = torch.meshgrid(
        torch.arange(height, device=source_clean.device, dtype=torch.float32),
        torch.arange(width, device=source_clean.device, dtype=torch.float32),
        indexing="ij",
    )
    phase0 = source_clean[:, :, 0].float()
    references = []
    masks = []
    radius_x = 3.0 * scale_x
    radius_y = 1.25 * scale_y
    for _actor_xy, object_xy in centers:
        target_x = float(object_xy[0]) * scale_x
        target_y = float(object_xy[1]) * scale_y
        sample_x = xx - (target_x - source_x)
        sample_y = yy - (target_y - source_y)
        grid = torch.stack(
            (
                2.0 * sample_x / float(width - 1) - 1.0,
                2.0 * sample_y / float(height - 1) - 1.0,
            ),
            dim=-1,
        ).reshape(1, height, width, 2).repeat(batch, 1, 1, 1)
        references.append(
            F.grid_sample(
                phase0,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        )
        distance = (
            ((xx - target_x) / radius_x).square()
            + ((yy - target_y) / radius_y).square()
        ).sqrt()
        masks.append((1.5 - distance).clamp(0.0, 1.0))
    reference = torch.stack(references, dim=2).to(source_clean.dtype)
    mask = torch.stack(masks, dim=0).reshape(1, 1, phases, height, width)
    mask = mask.repeat(batch, 1, 1, 1, 1).to(source_clean.dtype)
    if tuple(reference.shape) != (batch, channels, phases, height, width):
        raise AnchorSGAANCError("Event01 translated object reference differs")
    return reference, mask


def _apply_event01_object1_identity_projection(
    edit_packed: torch.Tensor,
    source_clean: torch.Tensor,
    *,
    layout: cdf.LatentLayout,
    strength: float,
) -> torch.Tensor:
    """Weakly project the moving object onto source-owned phase-0 identity."""

    if not 0.0 <= float(strength) <= 1.0:
        raise AnchorSGAANCError("object identity strength must be in [0,1]")
    if float(strength) == 0.0:
        return edit_packed
    edit = cdf._unpack_spatial_latent(edit_packed, layout)
    reference, mask = _event01_object1_phasewise_source_reference(source_clean)
    projected = edit + float(strength) * mask * (reference - edit)
    projected[:, :, 0].copy_(source_clean[:, :, 0])
    return cdf._pack_spatial_latent(projected, layout)


def _apply_event01_object1_sparse_signature_projection(
    edit_packed: torch.Tensor,
    source_clean: torch.Tensor,
    *,
    layout: cdf.LatentLayout,
    strength: float,
) -> torch.Tensor:
    """Composite a compact source-owned entity signature and explicit vacancy.

    Unlike the historical full-crop grid sample, this route never copies the
    whole rectangular source neighborhood.  It preserves an ordered 4x4
    latent patch relative to a local background ring, translates that compact
    signature along the anchor-derived object path, and removes the same
    signature from the vacated origin once lifting begins.
    """

    if not 0.0 <= float(strength) <= 1.0:
        raise AnchorSGAANCError("sparse object identity strength must be in [0,1]")
    if float(strength) == 0.0:
        return edit_packed
    edit = cdf._unpack_spatial_latent(edit_packed, layout)
    if (
        edit.ndim != 5
        or tuple(edit.shape) != tuple(source_clean.shape)
        or int(edit.shape[2]) != qk_transport.LATENT_PHASES
    ):
        raise AnchorSGAANCError("sparse Event01 entity geometry differs")
    batch, _channels, phases, height, width = (
        int(item) for item in edit.shape
    )
    centers = qk_transport._event01_dynamic_target_centers(1)
    source_xy = qk_transport.EVENT01_SOURCE_OBJECT_PROPOSALS_XY[1]
    scale_x = float(width) / float(qk_transport.EVENT01_SPATIAL_WIDTH)
    scale_y = float(height) / float(qk_transport.EVENT01_SPATIAL_HEIGHT)
    source_x = float(source_xy[0]) * scale_x
    source_y = float(source_xy[1]) * scale_y
    yy, xx = torch.meshgrid(
        torch.arange(height, device=edit.device, dtype=torch.float32),
        torch.arange(width, device=edit.device, dtype=torch.float32),
        indexing="ij",
    )
    patch_distance = (
        ((xx - source_x) / (1.0 * scale_x)).square()
        + ((yy - source_y) / (0.75 * scale_y)).square()
    )
    source_flat = torch.topk(
        patch_distance.flatten(), k=16, largest=False, sorted=True
    ).indices
    source_yx = torch.stack(
        (
            torch.div(source_flat, width, rounding_mode="floor"),
            source_flat.remainder(width),
        ),
        dim=1,
    ).float()
    source_ring_distance = (
        ((xx - source_x) / (2.5 * scale_x)).square()
        + ((yy - source_y) / (1.75 * scale_y)).square()
    )
    source_ring_mask = (
        (source_ring_distance > 0.45) & (source_ring_distance <= 1.0)
    ).flatten()
    if not bool(source_ring_mask.any()):
        raise AnchorSGAANCError("sparse Event01 source ring is empty")
    phase0 = source_clean[:, :, 0].float().flatten(2).transpose(1, 2)
    source_patch = phase0.index_select(1, source_flat)
    source_ring = phase0[:, source_ring_mask].mean(dim=1, keepdim=True)
    source_signature = source_patch - source_ring
    offsets_y = source_yx[:, 0] - source_y
    offsets_x = source_yx[:, 1] - source_x

    routed = edit.float().clone()
    for phase_index, (_actor_xy, object_xy) in enumerate(centers):
        if phase_index == 0:
            continue
        target_x = float(object_xy[0]) * scale_x
        target_y = float(object_xy[1]) * scale_y
        target_yy = torch.floor(offsets_y + target_y + 0.5).long()
        target_xx = torch.floor(offsets_x + target_x + 0.5).long()
        if not bool(
            (
                (target_yy >= 0)
                & (target_yy < height)
                & (target_xx >= 0)
                & (target_xx < width)
            ).all()
        ):
            raise AnchorSGAANCError("sparse Event01 target patch leaves grid")
        target_flat = target_yy * width + target_xx
        if int(target_flat.unique().numel()) != 16:
            raise AnchorSGAANCError("sparse Event01 target patch duplicates cells")
        target_ring_distance = (
            ((xx - target_x) / (2.5 * scale_x)).square()
            + ((yy - target_y) / (1.75 * scale_y)).square()
        )
        target_ring_mask = (
            (target_ring_distance > 0.45) & (target_ring_distance <= 1.0)
        ).flatten()
        current = edit[:, :, phase_index].float().flatten(2).transpose(1, 2)
        moved = current.clone()
        target_ring = current[:, target_ring_mask].mean(dim=1, keepdim=True)
        if float(qk_transport.EVENT01_TARGET_OBJECT_LIFT_PROGRESS[phase_index]) > 0.0:
            origin_ring = current[:, source_ring_mask].mean(dim=1, keepdim=True)
            moved[:, source_flat] = origin_ring
        moved[:, target_flat] = target_ring + source_signature
        blended = current + float(strength) * (moved - current)
        routed[:, :, phase_index] = blended.transpose(1, 2).reshape(
            batch, -1, height, width
        )
    routed[:, :, 0].copy_(source_clean[:, :, 0])
    return cdf._pack_spatial_latent(routed.to(edit.dtype), layout)


def _aggregate_candidates_background_source_cosine(
    *,
    source_packed: torch.Tensor,
    source_clean: torch.Tensor,
    edit_packed: torch.Tensor,
    candidate_deltas: torch.Tensor,
    sigma: float,
    temperature: float,
    layout: cdf.LatentLayout,
    source_support: torch.Tensor,
    residual_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply DynaEdit SGA on candidate-specific unedited background only.

    Global source cosine rewards candidates for leaving the actor and contacted
    object unchanged. Here each projected candidate first declares the same
    bounded editable support used by preservation; cosine is then evaluated on
    its complement. The full projected latent is still aggregated and becomes
    the Euler endpoint, so this changes candidate selection rather than
    replacing the generated action with a compressed representation.
    """

    if (
        candidate_deltas.ndim != source_packed.ndim + 1
        or tuple(candidate_deltas.shape[1:]) != tuple(source_packed.shape)
        or int(candidate_deltas.shape[0]) < 2
    ):
        raise AnchorSGAANCError("background SGA candidate geometry differs")
    sigma_value = float(sigma)
    temperature_value = float(temperature)
    if not 0.0 < sigma_value <= 1.0 or not 0.0 < temperature_value:
        raise AnchorSGAANCError("background SGA sigma/temperature is invalid")
    projected = edit_packed.unsqueeze(0) - sigma_value * candidate_deltas
    scores = []
    for projected_packed in projected.unbind(0):
        projected_clean = cdf._unpack_spatial_latent(projected_packed, layout)
        editable = _effective_source_edit_support(
            projected_clean,
            source_clean,
            source_support,
            residual_fraction=residual_fraction,
        )
        background = 1.0 - editable.float()
        source_flat = (source_clean.float() * background).flatten(1)
        candidate_flat = (projected_clean.float() * background).flatten(1)
        source_norm = source_flat.square().sum(dim=-1).sqrt()
        candidate_norm = candidate_flat.square().sum(dim=-1).sqrt()
        if bool((source_norm <= 1.0e-12).any()) or bool(
            (candidate_norm <= 1.0e-12).any()
        ):
            raise AnchorSGAANCError("background SGA cosine has zero norm")
        scores.append(
            (source_flat * candidate_flat).sum(dim=-1)
            / (source_norm * candidate_norm)
        )
    score_tensor = torch.stack(scores, dim=0)
    weights = torch.softmax(score_tensor / temperature_value, dim=0)
    broadcast = weights.reshape(
        int(weights.shape[0]),
        int(weights.shape[1]),
        *([1] * (source_packed.ndim - 1)),
    )
    aggregate_projection = (
        broadcast.to(projected.dtype) * projected
    ).sum(dim=0)
    aggregate_delta = (edit_packed - aggregate_projection) / sigma_value
    if aggregate_delta.dtype != torch.float32:
        raise AnchorSGAANCError("background SGA aggregate must remain fp32")
    return aggregate_delta, weights, score_tensor


def _action_temporal_signature(effect: torch.Tensor) -> torch.Tensor:
    """Return a coordinate-free dense temporal signature of a video effect.

    The signature is not an edit target or a feature injected into the model.
    It is used only to rank complete SGA candidate directions.  Every selected
    spatial site's full channel trajectory contributes to a local T-by-T
    self-similarity matrix; source and anchor coordinates are never compared.
    """

    if (
        not isinstance(effect, torch.Tensor)
        or effect.ndim != 5
        or int(effect.shape[2]) != qk_transport.LATENT_PHASES
    ):
        raise AnchorSGAANCError(
            "action temporal signature expects [B,C,21,H,W]"
        )
    effect = effect.float()
    effect = effect - effect[:, :, :1]
    batch, channels, phases, height, width = tuple(int(v) for v in effect.shape)
    spatial = height * width
    tokens = effect.permute(0, 2, 3, 4, 1).reshape(
        batch, phases, spatial, channels
    )
    activity = tokens.square().mean(dim=(1, 3))
    keep = max(1, math.ceil(spatial * ANCHOR_ACTION_KEEP_FRACTION))
    active = torch.topk(
        activity, k=keep, dim=1, largest=True, sorted=False
    ).indices
    gather_index = active[:, None, :, None].expand(
        batch, phases, keep, channels
    )
    selected = torch.gather(tokens, 2, gather_index)
    normalized = torch.nn.functional.normalize(selected, dim=-1, eps=1.0e-6)
    local_gram = torch.einsum(
        "btkc,bukc->bktu", normalized, normalized
    ).mean(dim=1)
    motion_energy = selected.square().mean(dim=(2, 3)).sqrt()
    difference_energy = (
        selected[:, 1:] - selected[:, :-1]
    ).square().mean(dim=(2, 3)).sqrt()

    components = []
    for value in (local_gram, motion_energy, difference_energy):
        flat = value.flatten(1)
        components.append(
            torch.nn.functional.normalize(flat, dim=-1, eps=1.0e-6)
        )
    signature = torch.cat(components, dim=-1) / math.sqrt(len(components))
    if not bool(torch.isfinite(signature).all()):
        raise AnchorSGAANCError("action temporal signature is non-finite")
    return signature


def _aggregate_candidates_anchor_action_reward(
    *,
    source_packed: torch.Tensor,
    source_clean: torch.Tensor,
    edit_packed: torch.Tensor,
    candidate_deltas: torch.Tensor,
    sigma: float,
    temperature: float,
    layout: cdf.LatentLayout,
    source_support: torch.Tensor,
    residual_fraction: float,
    anchor_clean: torch.Tensor,
    anchor_static: torch.Tensor,
    mode: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Use the action anchor as an SGA reward, never as candidate content."""

    if mode not in (
        "background_plus_anchor_action_002",
        "background_trust_anchor_action_003",
    ):
        raise AnchorSGAANCError("unknown anchor-action SGA reward mode")
    _background_aggregate, _background_weights, background_scores = (
        _aggregate_candidates_background_source_cosine(
            source_packed=source_packed,
            source_clean=source_clean,
            edit_packed=edit_packed,
            candidate_deltas=candidate_deltas,
            sigma=sigma,
            temperature=temperature,
            layout=layout,
            source_support=source_support,
            residual_fraction=residual_fraction,
        )
    )
    sigma_value = float(sigma)
    projected = edit_packed.unsqueeze(0) - sigma_value * candidate_deltas
    anchor_signature = _action_temporal_signature(anchor_clean - anchor_static)
    action_scores = []
    for projected_packed in projected.unbind(0):
        projected_clean = cdf._unpack_spatial_latent(projected_packed, layout)
        candidate_signature = _action_temporal_signature(
            projected_clean - source_clean
        )
        action_scores.append((candidate_signature * anchor_signature).sum(dim=-1))
    action_score_tensor = torch.stack(action_scores, dim=0)
    if mode == "background_plus_anchor_action_002":
        score_tensor = (
            background_scores
            + ANCHOR_ACTION_ADDITIVE_WEIGHT * action_score_tensor
        )
    else:
        best_background = background_scores.max(dim=0, keepdim=True).values
        eligible = background_scores >= (
            best_background - ANCHOR_ACTION_BACKGROUND_TRUST
        )
        rejected_floor = action_score_tensor.min(dim=0, keepdim=True).values - 2.0
        score_tensor = torch.where(
            eligible, action_score_tensor, rejected_floor.expand_as(action_score_tensor)
        )
    weights = torch.softmax(score_tensor / float(temperature), dim=0)
    broadcast = weights.reshape(
        int(weights.shape[0]),
        int(weights.shape[1]),
        *([1] * (source_packed.ndim - 1)),
    )
    aggregate_projection = (
        broadcast.to(projected.dtype) * projected
    ).sum(dim=0)
    aggregate_delta = (edit_packed - aggregate_projection) / sigma_value
    if (
        aggregate_delta.dtype != torch.float32
        or not bool(torch.isfinite(aggregate_delta).all())
    ):
        raise AnchorSGAANCError("anchor-action SGA aggregate must remain finite fp32")
    return (
        aggregate_delta,
        weights,
        score_tensor,
        background_scores,
        action_score_tensor,
    )


def _action_motion_envelope_signature(effect: torch.Tensor) -> torch.Tensor:
    """Canonicalize dense temporal-derivative energy without anchor/source matching."""

    if (
        not isinstance(effect, torch.Tensor)
        or effect.ndim != 5
        or int(effect.shape[2]) != qk_transport.LATENT_PHASES
    ):
        raise AnchorSGAANCError(
            "action motion envelope expects [B,C,21,H,W]"
        )
    effect = effect.float()
    effect = effect - effect[:, :, :1]
    derivative = torch.cat(
        (torch.zeros_like(effect[:, :, :1]), effect[:, :, 1:] - effect[:, :, :-1]),
        dim=2,
    )
    energy = derivative.square().mean(dim=1).sqrt()
    batch, phases, height, width = tuple(int(v) for v in energy.shape)
    aggregate = energy.sum(dim=1)
    aggregate_mass = aggregate.sum(dim=(1, 2), keepdim=True).clamp_min(1.0e-8)
    y_axis = torch.linspace(
        -1.0, 1.0, height, device=energy.device, dtype=energy.dtype
    ).reshape(1, height, 1)
    x_axis = torch.linspace(
        -1.0, 1.0, width, device=energy.device, dtype=energy.dtype
    ).reshape(1, 1, width)
    center_y = (aggregate * y_axis).sum(dim=(1, 2), keepdim=True) / aggregate_mass
    center_x = (aggregate * x_axis).sum(dim=(1, 2), keepdim=True) / aggregate_mass
    scale_y = (
        (aggregate * (y_axis - center_y).square()).sum(dim=(1, 2), keepdim=True)
        / aggregate_mass
    ).sqrt().clamp_min(2.0 / max(2, height))
    scale_x = (
        (aggregate * (x_axis - center_x).square()).sum(dim=(1, 2), keepdim=True)
        / aggregate_mass
    ).sqrt().clamp_min(2.0 / max(2, width))

    canonical_axis = torch.linspace(
        -1.0,
        1.0,
        ANCHOR_ENVELOPE_CANONICAL_SIZE,
        device=energy.device,
        dtype=energy.dtype,
    )
    grid_y = center_y.reshape(batch, 1, 1) + (
        2.5 * scale_y.reshape(batch, 1, 1) * canonical_axis.reshape(1, -1, 1)
    )
    grid_x = center_x.reshape(batch, 1, 1) + (
        2.5 * scale_x.reshape(batch, 1, 1) * canonical_axis.reshape(1, 1, -1)
    )
    grid = torch.stack(
        (
            grid_x.expand(batch, ANCHOR_ENVELOPE_CANONICAL_SIZE, -1),
            grid_y.expand(batch, -1, ANCHOR_ENVELOPE_CANONICAL_SIZE),
        ),
        dim=-1,
    ).clamp(-1.0, 1.0)
    phase_grid = grid[:, None].expand(
        batch,
        phases,
        ANCHOR_ENVELOPE_CANONICAL_SIZE,
        ANCHOR_ENVELOPE_CANONICAL_SIZE,
        2,
    ).reshape(
        batch * phases,
        ANCHOR_ENVELOPE_CANONICAL_SIZE,
        ANCHOR_ENVELOPE_CANONICAL_SIZE,
        2,
    )
    canonical = torch.nn.functional.grid_sample(
        energy.reshape(batch * phases, 1, height, width),
        phase_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(
        batch,
        phases,
        ANCHOR_ENVELOPE_CANONICAL_SIZE,
        ANCHOR_ENVELOPE_CANONICAL_SIZE,
    )

    phase_mass = energy.sum(dim=(2, 3)).clamp_min(1.0e-8)
    phase_center_y = (energy * y_axis[:, None]).sum(dim=(2, 3)) / phase_mass
    phase_center_x = (energy * x_axis[:, None]).sum(dim=(2, 3)) / phase_mass
    relative_center = torch.stack(
        (
            (phase_center_y - center_y.reshape(batch, 1))
            / scale_y.reshape(batch, 1),
            (phase_center_x - center_x.reshape(batch, 1))
            / scale_x.reshape(batch, 1),
        ),
        dim=-1,
    )
    # Phase zero has no derivative; assign the aggregate center rather than a
    # meaningless zero-mass coordinate.
    relative_center[:, 0].zero_()
    components = []
    for value in (canonical, phase_mass, relative_center):
        components.append(
            torch.nn.functional.normalize(value.flatten(1), dim=-1, eps=1.0e-6)
        )
    signature = torch.cat(components, dim=-1) / math.sqrt(len(components))
    if not bool(torch.isfinite(signature).all()):
        raise AnchorSGAANCError("action motion envelope signature is non-finite")
    return signature


def _aggregate_candidates_anchor_envelope_reward(
    *,
    source_packed: torch.Tensor,
    source_clean: torch.Tensor,
    edit_packed: torch.Tensor,
    candidate_deltas: torch.Tensor,
    sigma: float,
    temperature: float,
    layout: cdf.LatentLayout,
    source_support: torch.Tensor,
    residual_fraction: float,
    anchor_clean: torch.Tensor,
    anchor_static: torch.Tensor,
    mode: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if mode not in (
        "background_plus_anchor_envelope_005",
        "background_trust_anchor_envelope_003",
    ):
        raise AnchorSGAANCError("unknown anchor-envelope SGA reward mode")
    _background_aggregate, _background_weights, background_scores = (
        _aggregate_candidates_background_source_cosine(
            source_packed=source_packed,
            source_clean=source_clean,
            edit_packed=edit_packed,
            candidate_deltas=candidate_deltas,
            sigma=sigma,
            temperature=temperature,
            layout=layout,
            source_support=source_support,
            residual_fraction=residual_fraction,
        )
    )
    sigma_value = float(sigma)
    projected = edit_packed.unsqueeze(0) - sigma_value * candidate_deltas
    anchor_signature = _action_motion_envelope_signature(anchor_clean - anchor_static)
    envelope_scores = []
    for projected_packed in projected.unbind(0):
        projected_clean = cdf._unpack_spatial_latent(projected_packed, layout)
        candidate_signature = _action_motion_envelope_signature(
            projected_clean - source_clean
        )
        envelope_scores.append((candidate_signature * anchor_signature).sum(dim=-1))
    envelope_score_tensor = torch.stack(envelope_scores, dim=0)
    if mode == "background_plus_anchor_envelope_005":
        score_tensor = (
            background_scores
            + ANCHOR_ENVELOPE_ADDITIVE_WEIGHT * envelope_score_tensor
        )
    else:
        best_background = background_scores.max(dim=0, keepdim=True).values
        eligible = background_scores >= (
            best_background - ANCHOR_ACTION_BACKGROUND_TRUST
        )
        rejected_floor = envelope_score_tensor.min(dim=0, keepdim=True).values - 2.0
        score_tensor = torch.where(
            eligible,
            envelope_score_tensor,
            rejected_floor.expand_as(envelope_score_tensor),
        )
    weights = torch.softmax(score_tensor / float(temperature), dim=0)
    broadcast = weights.reshape(
        int(weights.shape[0]),
        int(weights.shape[1]),
        *([1] * (source_packed.ndim - 1)),
    )
    aggregate_projection = (
        broadcast.to(projected.dtype) * projected
    ).sum(dim=0)
    aggregate_delta = (edit_packed - aggregate_projection) / sigma_value
    if (
        aggregate_delta.dtype != torch.float32
        or not bool(torch.isfinite(aggregate_delta).all())
    ):
        raise AnchorSGAANCError(
            "anchor-envelope SGA aggregate must remain finite fp32"
        )
    return (
        aggregate_delta,
        weights,
        score_tensor,
        background_scores,
        envelope_score_tensor,
    )


def _sparse_packed_temporal_residual(
    current: torch.Tensor,
    anchor: torch.Tensor,
    *,
    strength: float,
) -> torch.Tensor:
    """Transport a full-network velocity trajectory without its static basis."""

    if (
        not isinstance(current, torch.Tensor)
        or not isinstance(anchor, torch.Tensor)
        or current.ndim != 3
        or tuple(current.shape) != tuple(anchor.shape)
        or int(current.shape[1]) % qk_transport.LATENT_PHASES
    ):
        raise AnchorSGAANCError("velocity residual requires matched packed 21-phase fields")
    routed = qk_transport._sparse_frame0_temporal_residual(
        current.unsqueeze(2),
        anchor.unsqueeze(2),
        strength=strength,
    ).squeeze(2)
    if routed.dtype != current.dtype or routed.device != current.device:
        raise AnchorSGAANCError("velocity residual dtype/device differs")
    return routed


def _sparse_packed_action_contrast(
    action_velocity: torch.Tensor,
    noop_velocity: torch.Tensor,
    *,
    strength: float,
    keep_fraction: float = 0.25,
) -> torch.Tensor:
    """Extract an action-minus-noop direction from the same anchor state."""

    if (
        not isinstance(action_velocity, torch.Tensor)
        or not isinstance(noop_velocity, torch.Tensor)
        or action_velocity.ndim != 3
        or tuple(action_velocity.shape) != tuple(noop_velocity.shape)
        or int(action_velocity.shape[1]) % guided.EXPECTED_LATENT_PHASES
        or not 0.0 < float(keep_fraction) <= 1.0
    ):
        raise AnchorSGAANCError(
            "anchor contrast requires matched packed 21-phase velocity fields"
        )
    phases = guided.EXPECTED_LATENT_PHASES
    spatial = int(action_velocity.shape[1]) // phases
    contrast = (action_velocity - noop_velocity).reshape(
        int(action_velocity.shape[0]),
        phases,
        spatial,
        int(action_velocity.shape[2]),
    )
    temporal = contrast - contrast[:, :1]
    temporal[:, 0] = 0
    importance = temporal.float().square().sum(dim=-1)
    keep = max(1, math.ceil(spatial * keep_fraction))
    topk = importance.topk(keep, dim=-1).indices
    mask = torch.zeros_like(importance, dtype=torch.bool).scatter_(-1, topk, True)
    mask[:, 0] = False
    routed = float(strength) * temporal * mask.unsqueeze(-1)
    return routed.reshape_as(action_velocity)


def _sparse_packed_raw_action_contrast(
    action_velocity: torch.Tensor,
    noop_velocity: torch.Tensor,
    *,
    strength: float,
    keep_fraction: float = 1.0,
) -> torch.Tensor:
    """Keep the direct action-minus-noop field, including absolute phases.

    Unlike ``_sparse_packed_action_contrast``, this control deliberately does
    not subtract the phase-0 contrast.  The outer sampler still clamps latent
    phase zero to the source after every update.  Comparing the two operators
    therefore tests whether phase-0 quotienting removed a necessary action
    component rather than merely suppressing anchor appearance.
    """

    if (
        not isinstance(action_velocity, torch.Tensor)
        or not isinstance(noop_velocity, torch.Tensor)
        or action_velocity.ndim != 3
        or tuple(action_velocity.shape) != tuple(noop_velocity.shape)
        or int(action_velocity.shape[1]) % guided.EXPECTED_LATENT_PHASES
        or not 0.0 < float(keep_fraction) <= 1.0
    ):
        raise AnchorSGAANCError(
            "raw action contrast requires matched packed 21-phase velocity fields"
        )
    phases = guided.EXPECTED_LATENT_PHASES
    spatial = int(action_velocity.shape[1]) // phases
    contrast = (action_velocity - noop_velocity).reshape(
        int(action_velocity.shape[0]),
        phases,
        spatial,
        int(action_velocity.shape[2]),
    )
    importance = contrast.float().square().sum(dim=-1)
    keep = max(1, math.ceil(spatial * keep_fraction))
    topk = importance.topk(keep, dim=-1).indices
    mask = torch.zeros_like(importance, dtype=torch.bool).scatter_(-1, topk, True)
    routed = float(strength) * contrast * mask.unsqueeze(-1)
    return routed.reshape_as(action_velocity)


EVENT01_ROLE_ANCHOR_ACTOR_XY = (38, 31)
EVENT01_ROLE_ANCHOR_OBJECT_XY = (14, 43)
EVENT01_ROLE_SOURCE_ACTOR_XY = (24, 50)
EVENT01_ROLE_SOURCE_OBJECT_PROPOSALS_XY = (
    (19, 58),
    (35, 61),
    (41, 57),
    (22, 64),
    (14, 69),
)


def _shift_spatial_zero_pad(
    value: torch.Tensor,
    *,
    shift_y: int,
    shift_x: int,
) -> torch.Tensor:
    """Translate ``[..., H, W]`` without circular wraparound."""

    if value.ndim < 2:
        raise AnchorSGAANCError("spatial shift requires at least two dimensions")
    height, width = int(value.shape[-2]), int(value.shape[-1])
    if abs(int(shift_y)) >= height or abs(int(shift_x)) >= width:
        raise AnchorSGAANCError("role shift leaves the complete spatial canvas")
    source_y0 = max(0, -int(shift_y))
    source_y1 = min(height, height - int(shift_y))
    source_x0 = max(0, -int(shift_x))
    source_x1 = min(width, width - int(shift_x))
    target_y0 = source_y0 + int(shift_y)
    target_y1 = source_y1 + int(shift_y)
    target_x0 = source_x0 + int(shift_x)
    target_x1 = source_x1 + int(shift_x)
    shifted = torch.zeros_like(value)
    shifted[..., target_y0:target_y1, target_x0:target_x1] = value[
        ..., source_y0:source_y1, source_x0:source_x1
    ]
    return shifted


def _event01_role_warp_native_route(
    route_packed: torch.Tensor,
    *,
    layout: cdf.LatentLayout,
    proposal_index: int,
    keep_fraction: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Rebind native actor/contact fields to one explicit source-stone proposal.

    This is deliberately an Event01 geometry canary, not an automatic object
    detector.  The native route is split smoothly around preregistered anchor
    actor/object centers.  The two components are translated independently to
    the source child and one source stone, avoiding the single global affine
    that previously mapped both roles to the same wrong topology.
    """

    if (
        not isinstance(route_packed, torch.Tensor)
        or route_packed.dtype != torch.float32
        or route_packed.ndim != 3
        or layout.height != 72
        or layout.width != 52
        or not isinstance(proposal_index, int)
        or not 0 <= proposal_index < len(EVENT01_ROLE_SOURCE_OBJECT_PROPOSALS_XY)
        or not 0.0 < float(keep_fraction) <= 1.0
    ):
        raise AnchorSGAANCError("Event01 role-warp contract differs")
    route = cdf._unpack_spatial_latent(route_packed, layout)
    yy, xx = torch.meshgrid(
        torch.arange(layout.height, device=route.device, dtype=torch.float32),
        torch.arange(layout.width, device=route.device, dtype=torch.float32),
        indexing="ij",
    )
    anchor_actor_x, anchor_actor_y = EVENT01_ROLE_ANCHOR_ACTOR_XY
    anchor_object_x, anchor_object_y = EVENT01_ROLE_ANCHOR_OBJECT_XY
    actor_cost = (
        ((xx - float(anchor_actor_x)) / 15.0).square()
        + ((yy - float(anchor_actor_y)) / 20.0).square()
    )
    object_cost = (
        ((xx - float(anchor_object_x)) / 12.0).square()
        + ((yy - float(anchor_object_y)) / 10.0).square()
    )
    assignment = torch.softmax(
        torch.stack((-actor_cost, -object_cost), dim=0), dim=0
    )
    actor_component = route * assignment[0].reshape(1, 1, 1, layout.height, layout.width)
    object_component = route * assignment[1].reshape(1, 1, 1, layout.height, layout.width)
    source_actor_x, source_actor_y = EVENT01_ROLE_SOURCE_ACTOR_XY
    source_object_x, source_object_y = EVENT01_ROLE_SOURCE_OBJECT_PROPOSALS_XY[
        proposal_index
    ]
    actor_shift = (source_actor_y - anchor_actor_y, source_actor_x - anchor_actor_x)
    object_shift = (
        source_object_y - anchor_object_y,
        source_object_x - anchor_object_x,
    )
    warped = _shift_spatial_zero_pad(
        actor_component, shift_y=actor_shift[0], shift_x=actor_shift[1]
    ) + _shift_spatial_zero_pad(
        object_component, shift_y=object_shift[0], shift_x=object_shift[1]
    )
    if float(keep_fraction) < 1.0:
        batch, _, phases, height, width = warped.shape
        energy = warped.float().square().mean(dim=1)
        flat = energy.reshape(batch * phases, height * width)
        keep = max(1, math.ceil(height * width * float(keep_fraction)))
        top = flat.topk(keep, dim=-1, largest=True, sorted=False).indices
        mask = torch.zeros_like(flat, dtype=torch.bool).scatter_(-1, top, True)
        mask = mask.reshape(batch, phases, height, width).unsqueeze(1)
        warped = warped * mask.to(warped.dtype)
    warped[:, :, 0].zero_()
    packed = cdf._pack_spatial_latent(warped.contiguous(), layout)
    if packed.dtype != torch.float32 or not bool(torch.isfinite(packed).all()):
        raise AnchorSGAANCError("Event01 role-warp route is invalid")
    return packed, {
        "proposal_index": proposal_index,
        "anchor_actor_xy": list(EVENT01_ROLE_ANCHOR_ACTOR_XY),
        "anchor_object_xy": list(EVENT01_ROLE_ANCHOR_OBJECT_XY),
        "source_actor_xy": list(EVENT01_ROLE_SOURCE_ACTOR_XY),
        "source_object_xy": list(EVENT01_ROLE_SOURCE_OBJECT_PROPOSALS_XY[proposal_index]),
        "actor_shift_yx": list(actor_shift),
        "object_shift_yx": list(object_shift),
        "keep_fraction": float(keep_fraction),
    }


def _apply_native_t2v_hard_replacement(
    *,
    target_velocity: torch.Tensor,
    source_velocity: torch.Tensor,
    action_velocity: torch.Tensor,
    noop_velocity: torch.Tensor,
    transport: str,
) -> torch.Tensor:
    """Replace the target field or complete edit delta without attenuation."""

    tensors = (target_velocity, source_velocity, action_velocity, noop_velocity)
    if (
        transport not in NATIVE_T2V_REPLACEMENT_TRANSPORTS
        or any(item.shape != target_velocity.shape for item in tensors[1:])
        or any(item.dtype != target_velocity.dtype for item in tensors[1:])
        or any(item.device != target_velocity.device for item in tensors[1:])
        or any(not bool(torch.isfinite(item).all()) for item in tensors)
    ):
        raise AnchorSGAANCError("native T2V hard-replacement tensors differ")
    if transport == FIELD_NATIVE_T2V_TARGET_VELOCITY_REPLACEMENT:
        return action_velocity
    if transport in (
        FIELD_NATIVE_T2V_TEMPORAL_DELTA_REPLACEMENT,
        FIELD_NATIVE_T2V_SPARSE25_TEMPORAL_DELTA_REPLACEMENT,
    ):
        keep_fraction = (
            1.0
            if transport == FIELD_NATIVE_T2V_TEMPORAL_DELTA_REPLACEMENT
            else 0.25
        )
        temporal_quotient = _sparse_packed_action_contrast(
            action_velocity,
            noop_velocity,
            strength=1.0,
            keep_fraction=keep_fraction,
        )
        return source_velocity + temporal_quotient
    return source_velocity + action_velocity - noop_velocity


def _apply_native_phase_envelope(
    target_route: torch.Tensor,
    native_action_velocity: torch.Tensor,
    native_noop_velocity: torch.Tensor,
) -> tuple[torch.Tensor, list[float]]:
    """Gate a source-coordinate route by the anchor trajectory's phase energy.

    The transferred action remains the full target-state T2V action/no-op
    velocity tensor.  The independent self-generated trajectory contributes
    only its 21-phase event envelope, so no anchor spatial coordinate,
    appearance value, clean latent, or RGB pixel is copied into the edit.
    """

    if (
        not isinstance(target_route, torch.Tensor)
        or not isinstance(native_action_velocity, torch.Tensor)
        or not isinstance(native_noop_velocity, torch.Tensor)
        or target_route.ndim != 3
        or tuple(target_route.shape) != tuple(native_action_velocity.shape)
        or tuple(target_route.shape) != tuple(native_noop_velocity.shape)
        or int(target_route.shape[1]) % guided.EXPECTED_LATENT_PHASES
    ):
        raise AnchorSGAANCError(
            "native phase envelope requires matched packed 21-phase fields"
        )
    phases = guided.EXPECTED_LATENT_PHASES
    spatial = int(target_route.shape[1]) // phases
    native_contrast = (native_action_velocity - native_noop_velocity).reshape(
        int(target_route.shape[0]),
        phases,
        spatial,
        int(target_route.shape[2]),
    )
    native_temporal = native_contrast - native_contrast[:, :1]
    native_temporal[:, 0].zero_()
    energy = native_temporal.float().square().mean(dim=(-2, -1)).sqrt()
    noninitial_mean = energy[:, 1:].mean(dim=1, keepdim=True).clamp_min(1e-6)
    envelope = (energy / noninitial_mean).clamp(0.25, 4.0)
    envelope[:, 0].zero_()
    routed = target_route.reshape(
        int(target_route.shape[0]),
        phases,
        spatial,
        int(target_route.shape[2]),
    ) * envelope.to(dtype=target_route.dtype).unsqueeze(-1).unsqueeze(-1)
    routed = routed.reshape_as(target_route).contiguous()
    if not bool(torch.isfinite(routed).all()):
        raise AnchorSGAANCError("native phase-gated target route is invalid")
    return routed, [float(value) for value in envelope[0].detach().cpu()]


def _align_packed_route_to_source_motion(
    route_packed: torch.Tensor,
    source_support: torch.Tensor,
    *,
    layout: cdf.LatentLayout,
    keep_fraction: float = 0.10,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Affine-align a dense anchor motion route to the source actor support.

    The transform is estimated from full spatial-temporal energy, not from a
    low-dimensional action code.  It only changes where the already isolated
    dynamic-minus-static velocity route acts: the source video remains the
    identity/content authority and phase zero remains exactly zero.
    """

    if (
        not isinstance(route_packed, torch.Tensor)
        or route_packed.dtype != torch.float32
        or source_support.ndim != 5
        or tuple(source_support.shape)
        != (layout.batch, 1, 1, layout.height, layout.width)
        or not 0.0 < float(keep_fraction) <= 1.0
    ):
        raise AnchorSGAANCError("anchor/source motion alignment geometry differs")
    route = cdf._unpack_spatial_latent(route_packed, layout)
    energy = route.square().mean(dim=1).amax(dim=1)
    flat = energy.flatten(1)
    keep = max(1, math.ceil(int(flat.shape[1]) * float(keep_fraction)))
    top = flat.topk(keep, dim=1, largest=True, sorted=False).indices
    anchor_mask = torch.zeros_like(flat, dtype=torch.bool)
    anchor_mask.scatter_(1, top, True)
    anchor_weights = anchor_mask.reshape_as(energy).float()
    source_weights = source_support[:, 0, 0].float()

    ys = (
        (torch.arange(layout.height, device=route.device, dtype=torch.float32) + 0.5)
        * (2.0 / float(layout.height))
        - 1.0
    )
    xs = (
        (torch.arange(layout.width, device=route.device, dtype=torch.float32) + 0.5)
        * (2.0 / float(layout.width))
        - 1.0
    )
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    def moments(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mass = weights.sum(dim=(-2, -1)).clamp_min(1.0)
        center_x = (weights * grid_x).sum(dim=(-2, -1)) / mass
        center_y = (weights * grid_y).sum(dim=(-2, -1)) / mass
        std_x = (
            (weights * (grid_x - center_x[:, None, None]).square()).sum(
                dim=(-2, -1)
            )
            / mass
        ).sqrt().clamp_min(0.05)
        std_y = (
            (weights * (grid_y - center_y[:, None, None]).square()).sum(
                dim=(-2, -1)
            )
            / mass
        ).sqrt().clamp_min(0.05)
        return torch.stack((center_x, center_y), dim=-1), torch.stack(
            (std_x, std_y), dim=-1
        )

    anchor_center, anchor_std = moments(anchor_weights)
    source_center, source_std = moments(source_weights)
    scale = (source_std / anchor_std).clamp(0.5, 2.0)
    inverse_scale = scale.reciprocal()
    theta = torch.zeros(
        layout.batch, 2, 3, device=route.device, dtype=torch.float32
    )
    theta[:, 0, 0] = inverse_scale[:, 0]
    theta[:, 1, 1] = inverse_scale[:, 1]
    theta[:, 0, 2] = anchor_center[:, 0] - inverse_scale[:, 0] * source_center[:, 0]
    theta[:, 1, 2] = anchor_center[:, 1] - inverse_scale[:, 1] * source_center[:, 1]

    phases = int(route.shape[2])
    frames = route.permute(0, 2, 1, 3, 4).reshape(
        layout.batch * phases, layout.channels, layout.height, layout.width
    )
    frame_theta = theta[:, None].expand(-1, phases, -1, -1).reshape(
        layout.batch * phases, 2, 3
    )
    sample_grid = torch.nn.functional.affine_grid(
        frame_theta,
        frames.shape,
        align_corners=False,
    )
    aligned_frames = torch.nn.functional.grid_sample(
        frames,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    aligned = aligned_frames.reshape(
        layout.batch,
        phases,
        layout.channels,
        layout.height,
        layout.width,
    ).permute(0, 2, 1, 3, 4).contiguous()
    aligned[:, :, 0].zero_()
    aligned_packed = cdf._pack_spatial_latent(aligned, layout)
    if aligned_packed.dtype != torch.float32 or not bool(
        torch.isfinite(aligned_packed).all()
    ):
        raise AnchorSGAANCError("aligned anchor motion route is invalid")
    audit = {
        "anchor_center_xy": [float(item) for item in anchor_center[0].cpu()],
        "source_center_xy": [float(item) for item in source_center[0].cpu()],
        "scale_xy": [float(item) for item in scale[0].cpu()],
        "anchor_support_fraction": float(anchor_weights.mean().cpu().item()),
        "source_support_fraction": float(source_weights.mean().cpu().item()),
    }
    return aligned_packed, audit


def _validate_target_owned_qk_route_closure(
    *,
    transport: str,
    transport_steps: int,
    expected_anchor_cells: int,
    selected_block_count: int,
    field_guidance: str,
    anchor_cfg_scope: str,
    trace: Mapping[str, Any],
    cache_receipt: Mapping[str, Any],
) -> None:
    """Prove that a v14r2 route actually captured and replayed donor Q/K."""

    if transport not in qk_transport.TARGET_OWNED_QK_TRANSPORTS_V14R2:
        return
    route_on = transport_steps > 0
    expected_capture = (
        2 * expected_anchor_cells * selected_block_count if route_on else 0
    )
    replay_multiplier = (
        2 if field_guidance in ("apg", "raw_cfg") and anchor_cfg_scope == "shared" else 1
    )
    expected_replay = replay_multiplier * expected_capture
    if (
        isinstance(expected_anchor_cells, bool)
        or expected_anchor_cells < 0
        or isinstance(selected_block_count, bool)
        or selected_block_count < 1
        or cache_receipt.get("capture_count") != expected_capture
        or cache_receipt.get("qk_only_capture_count") != expected_capture
        or cache_receipt.get("replay_count") != expected_replay
        or cache_receipt.get("qk_only_replay_count") != expected_replay
        or cache_receipt.get("pending_entries") != 0
        or cache_receipt.get("qk_only_cached_fields") != ["query", "key"]
        or trace.get("target_owned_qk_route_v14r2") is not route_on
        or trace.get("anchor_donor_cached_fields")
        != (["query", "key"] if route_on else None)
        or trace.get("anchor_donor_value_hidden_output_or_coordinate_used")
        is not (False if route_on else None)
    ):
        raise AnchorSGAANCError("v14r2 target-owned QK route closure differs")


def sample_anchor_sga_anc(
    renderer_or_diffusion: Any,
    *,
    source_latent: torch.Tensor,
    anchor_latent: torch.Tensor,
    anchor_initial_gaussian: Optional[torch.Tensor] = None,
    source_rgb_frames: int,
    action_prompt_embeds: torch.Tensor,
    anchor_prompt_embeds: torch.Tensor,
    anchor_noop_prompt_embeds: torch.Tensor,
    source_t2v_prompt_embeds: torch.Tensor,
    target_t2v_prompt_embeds: torch.Tensor,
    noop_prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    config: AnchorSGAANCConfig,
    return_trace: bool = False,
) -> Any:
    runtime = config.validate()
    if source_rgb_frames != guided.EXPECTED_RGB_FRAMES:
        raise AnchorSGAANCError("source must contain exactly 81 RGB frames")
    if (
        not isinstance(source_latent, torch.Tensor)
        or not isinstance(anchor_latent, torch.Tensor)
        or source_latent.dtype != torch.float32
        or anchor_latent.dtype != torch.float32
        or source_latent.ndim != 5
        or int(source_latent.shape[2]) != guided.EXPECTED_LATENT_PHASES
    ):
        raise AnchorSGAANCError("source/anchor must be FP32 21-phase latents")
    if anchor_latent.ndim == source_latent.ndim:
        anchor_clean_bank = anchor_latent.unsqueeze(0)
    elif anchor_latent.ndim == source_latent.ndim + 1:
        anchor_clean_bank = anchor_latent
    else:
        raise AnchorSGAANCError("anchor bank must be [K,B,C,T,H,W]")
    anchor_bank_size = int(anchor_clean_bank.shape[0])
    if (
        tuple(anchor_clean_bank.shape[1:]) != tuple(source_latent.shape)
        or anchor_bank_size < 1
        or anchor_bank_size > EARLY_CANDIDATES
        or any(torch.equal(source_latent, item) for item in anchor_clean_bank.unbind(0))
    ):
        raise AnchorSGAANCError("source/anchor bank latent contract differs")
    if runtime.anchor_candidate_mode == "single_shared" and anchor_bank_size != 1:
        raise AnchorSGAANCError("single_shared requires exactly one pure-T2V anchor")
    if runtime.anchor_candidate_mode == "bank_per_candidate" and anchor_bank_size < 2:
        raise AnchorSGAANCError("bank_per_candidate requires at least two anchors")

    diffusion = cdf.resolve_diffusion_core(renderer_or_diffusion)
    layout, transformer = cdf._validate_runtime_inputs(
        diffusion, source_latent, action_prompt_embeds, noop_prompt_embeds
    )
    cdf._validate_runtime_inputs(
        diffusion, source_latent, anchor_prompt_embeds, negative_prompt_embeds
    )
    cdf._validate_runtime_inputs(
        diffusion, source_latent, anchor_noop_prompt_embeds, negative_prompt_embeds
    )
    if cdf.prompts_are_exactly_identical(
        anchor_prompt_embeds, anchor_noop_prompt_embeds
    ):
        raise AnchorSGAANCError("anchor action/no-op prompt embeddings must differ")
    cdf._validate_runtime_inputs(
        diffusion, source_latent, source_t2v_prompt_embeds, negative_prompt_embeds
    )
    cdf._validate_runtime_inputs(
        diffusion, source_latent, target_t2v_prompt_embeds, negative_prompt_embeds
    )
    intervals_config = cdf.DifferentialFlowConfig(
        num_inference_steps=runtime.num_inference_steps,
        flow_shift=runtime.flow_shift,
        seed=runtime.seed,
    )
    timesteps, raw_intervals = cdf._set_scheduler_timesteps(
        diffusion, intervals_config, source_latent.device
    )
    intervals = guided.validate_pinned_sigma_intervals(raw_intervals)
    scheduler_sigmas, scheduler_digest = guided.capture_pinned_scheduler_sigma_scalars(
        diffusion, intervals
    )
    if scheduler_digest != guided.PINNED_UNIPC_SIGMA_FP32_DIGEST:
        raise AnchorSGAANCError("pinned scheduler sigma digest differs")
    timestep_values = tuple(
        float(value.detach().to(device="cpu", dtype=torch.float64).item())
        for value in timesteps
    )
    sigma_values = tuple(intervals[0][:1]) + tuple(pair[1] for pair in intervals)
    schedule_digest = guided._object_sha256(
        {
            "timesteps": list(timestep_values),
            "sigmas": list(sigma_values),
            "flow_shift": runtime.flow_shift,
            "steps": runtime.num_inference_steps,
        }
    )
    if schedule_digest != guided.PINNED_UNIPC_SCHEDULE_DIGEST:
        raise AnchorSGAANCError("pinned UniPC full schedule digest differs")
    scheduler_config = getattr(getattr(diffusion, "scheduler", None), "config", None)
    num_train_timesteps = getattr(scheduler_config, "num_train_timesteps", None)
    if num_train_timesteps is None and isinstance(scheduler_config, dict):
        num_train_timesteps = scheduler_config.get("num_train_timesteps")
    if (
        isinstance(num_train_timesteps, bool)
        or not isinstance(num_train_timesteps, int)
        or num_train_timesteps != int(MODEL_TIMESTEP_SCALE)
    ):
        raise AnchorSGAANCError("pinned scheduler must expose 1000 train timesteps")

    source_clean = source_latent.detach().float()
    anchor_clean_bank = anchor_clean_bank.detach().float()
    needs_anchor_initial_gaussian = (
        runtime.initial_noise_proposal_mode != "keyed_only"
        or runtime.anchor_state_mode == "native_t2v_trajectory"
    )
    if not needs_anchor_initial_gaussian:
        if anchor_initial_gaussian is not None:
            raise AnchorSGAANCError(
                "inactive anchor Gaussian must not be supplied"
            )
        anchor_initial_packed = None
    else:
        if (
            not isinstance(anchor_initial_gaussian, torch.Tensor)
            or anchor_initial_gaussian.dtype != torch.float32
            or tuple(anchor_initial_gaussian.shape) != tuple(source_latent.shape)
            or not bool(torch.isfinite(anchor_initial_gaussian).all())
        ):
            raise AnchorSGAANCError(
                "anchor initial Gaussian must be finite FP32 source geometry"
            )
        anchor_initial_packed = cdf._pack_spatial_latent(
            anchor_initial_gaussian.detach().float(), layout
        )
    source_packed = cdf._pack_spatial_latent(source_clean, layout)
    anchor_packed_bank = torch.stack(
        [cdf._pack_spatial_latent(item, layout) for item in anchor_clean_bank.unbind(0)]
    )
    anchor_static_clean_bank = torch.stack(
        [
            item[:, :, :1].repeat(1, 1, int(item.shape[2]), 1, 1)
            for item in anchor_clean_bank.unbind(0)
        ]
    )
    anchor_static_packed_bank = torch.stack(
        [
            cdf._pack_spatial_latent(item, layout)
            for item in anchor_static_clean_bank.unbind(0)
        ]
    )
    native_anchor_action_packed: Optional[torch.Tensor] = None
    native_anchor_noop_packed: Optional[torch.Tensor] = None
    native_anchor_action_scheduler: Any = None
    native_anchor_noop_scheduler: Any = None
    if runtime.anchor_state_mode == "native_t2v_trajectory":
        if anchor_initial_packed is None:
            raise AnchorSGAANCError("native T2V trajectory requires its initial Gaussian")
        native_anchor_action_packed = anchor_initial_packed.clone()
        native_anchor_noop_packed = anchor_initial_packed.clone()
        native_anchor_action_scheduler = copy.deepcopy(diffusion.scheduler)
        native_anchor_noop_scheduler = copy.deepcopy(diffusion.scheduler)
        native_anchor_action_scheduler.set_timesteps(runtime.num_inference_steps)
        native_anchor_noop_scheduler.set_timesteps(runtime.num_inference_steps)
    collapsed_anchor_packed: Optional[torch.Tensor] = None
    collapsed_anchor_static_packed: Optional[torch.Tensor] = None
    selected_role_proposal_index = (
        runtime.event01_forced_role_proposal_index
        if runtime.event01_forced_role_proposal_index >= 0
        else 0
    )
    edit_packed = source_packed.clone()
    sga_source_support = (
        _source_motion_support(
            source_clean,
            keep_fraction=runtime.preservation_keep_fraction,
            dilation=runtime.preservation_dilation,
        )
        if runtime.preservation_mode
        in (
            "source_motion_support",
            "source_motion_support_snapshot_residual",
            "source_motion_support_event01_object1",
            "source_motion_support_event01_actor_object",
        )
        else None
    )
    preservation_support = (
        _event01_actor_object_phasewise_preservation_support(sga_source_support)
        if runtime.preservation_mode
        == "source_motion_support_event01_actor_object"
        else _event01_object1_phasewise_preservation_support(sga_source_support)
        if runtime.preservation_mode == "source_motion_support_event01_object1"
        else sga_source_support
    )
    snapshot_residual_support: Optional[torch.Tensor] = None
    early_candidate_count = (
        anchor_bank_size
        if runtime.anchor_candidate_mode == "bank_per_candidate"
        else runtime.candidate_count(0)
    )
    previous_noises = [torch.zeros_like(source_packed) for _ in range(early_candidate_count)]
    uses_cross_attention_transport = (
        runtime.transport
        == cross_transport.TEMPORAL_CONTRAST_CROSS_ATTN_OUTPUT
    )
    anchor_enabled = runtime.transport_steps > 0
    cache_bank: Any = (
        cross_transport.AnchorCrossAttentionCache(runtime.selected_block_indices)
        if uses_cross_attention_transport
        else qk_transport.AnchorQKCacheBank(runtime.selected_block_indices)
    )
    trace: dict[str, Any] = {
        "method": "bernini-anchor-attention-source-state-sga-anc-v46",
        "arm": runtime.arm,
        "transport": runtime.transport,
        "transport_strength": float(runtime.transport_strength),
        "transport_steps": runtime.transport_steps,
        "anchor_teacher_uses_unadapted_base": True,
        "anchor_teacher_disable_adapter_context_available": callable(
            getattr(renderer_or_diffusion, "disable_adapter", None)
        ),
        "target_source_editor_calls_leave_adapter_enabled": True,
        "field_guidance": runtime.field_guidance,
        "field_model": runtime.field_model,
        "source_cfg_scale": float(runtime.source_cfg_scale),
        "target_cfg_scale": float(runtime.target_cfg_scale),
        "anchor_cfg_scope": runtime.anchor_cfg_scope,
        "anchor_contrast_mode": runtime.anchor_contrast_mode,
        "anchor_sigma_cap": float(runtime.anchor_sigma_cap),
        "preservation_mode": runtime.preservation_mode,
        "preservation_keep_fraction": float(runtime.preservation_keep_fraction),
        "preservation_outside_scale": float(runtime.preservation_outside_scale),
        "preservation_dilation": runtime.preservation_dilation,
        "preservation_residual_fraction": float(
            runtime.preservation_residual_fraction
        ),
        "preservation_object_identity_strength": float(
            runtime.preservation_object_identity_strength
        ),
        "preservation_start_step": runtime.preservation_start_step,
        "preservation_ramp_steps": runtime.preservation_ramp_steps,
        "preservation_applied_schedule": [],
        "preservation_snapshot_residual_support_step": None,
        "preservation_snapshot_residual_support_fraction": None,
        "sga_score_mode": runtime.sga_score_mode,
        "anchor_candidate_mode": runtime.anchor_candidate_mode,
        "anchor_bank_size": anchor_bank_size,
        "anchor_bank_collapse_weights": None,
        "anchor_spatial_alignment": runtime.anchor_spatial_alignment,
        "anchor_spatial_alignment_audits": [],
        "anchor_native_phase_envelopes": [],
        "preservation_actual_spatial_fraction": (
            float(preservation_support.float().mean().detach().cpu().item())
            if preservation_support is not None
            else 1.0
        ),
        "preservation_event01_object1_phasewise_corridor": (
            runtime.preservation_mode
            in (
                "source_motion_support_event01_object1",
                "source_motion_support_event01_actor_object",
            )
        ),
        "preservation_event01_actor_object_phasewise_corridor": (
            runtime.preservation_mode
            == "source_motion_support_event01_actor_object"
        ),
        "event01_forced_role_proposal_index": (
            runtime.event01_forced_role_proposal_index
        ),
        "anchor_video_information_present_at_outer_sigma_one": (
            anchor_enabled and runtime.anchor_sigma_cap < 1.0
        ),
        "anchor_active_schedule": [],
        "anchor_candidate_cells": 0,
        "capped_teacher_coordinate_exact": True,
        "uncapped_teacher_reuses_outer_timestep": True,
        "outer_schedule_digest": schedule_digest,
        "anchor_reference_is_static_phase0_video": (
            runtime.anchor_contrast_mode == "dynamic_static_same_caption"
        ),
        "source_condition_role": {
            "source_conditioned_rv2v": "full_source_video_visual_prefix",
            "first_phase_source_rv2v": "source_initial_phase_repeated_static_visual_prefix",
            "first_phase_caption_i2v": "source_initial_phase_repeated_static_visual_prefix_with_source_target_caption_field",
            "source_free_t2v": "no_visual_prefix",
        }[runtime.field_model],
        "selected_block_indices": list(runtime.selected_block_indices),
        "candidate_counts": [],
        "configured_early_candidate_count": runtime.early_candidate_count,
        "initial_noise_proposal_mode": runtime.initial_noise_proposal_mode,
        "anchor_state_mode": runtime.anchor_state_mode,
        "anchor_initial_gaussian_used_at_step0_candidate0": False,
        "anchor_initial_gaussian_used_for_native_t2v_trajectory": (
            runtime.anchor_state_mode == "native_t2v_trajectory"
        ),
        "anchor_native_trajectory_model_forwards": 0,
        "anchor_native_trajectory_unipc_steps": 0,
        "anchor_native_action_terminal_mse_to_saved_clean": None,
        "anchor_native_action_terminal_max_abs_to_saved_clean": None,
        "native_target_replacement_step0_candidate0_action_mse_to_anchor_path": None,
        "native_target_replacement_step0_candidate0_noop_mse_to_anchor_path": None,
        "native_target_prebind_step0_candidate0_action_mse_to_anchor_path": None,
        "native_target_prebind_step0_candidate0_noop_mse_to_anchor_path": None,
        "native_path_velocity_bound_to_candidate0_schedule": [],
        "sga_weights_forced_to_anchor_candidate0": (
            runtime.initial_noise_proposal_mode == "anchor_candidate0_forced"
        ),
        "sga_scores": [],
        "sga_weights": [],
        "sga_background_scores": [],
        "sga_anchor_action_scores": [],
        "sga_combined_scores": [],
        "sga_temperature": float(runtime.sga_temperature),
        "anchor_action_reward_used_for_sga": runtime.sga_score_mode
        in (
            "background_plus_anchor_action_002",
            "background_trust_anchor_action_003",
            "background_plus_anchor_envelope_005",
            "background_trust_anchor_envelope_003",
        ),
        "anchor_action_reward_is_model_injection": False,
        "anchor_action_reward_signature": {
            "background_plus_anchor_action_002": "top25-local-latent-channel-trajectory-gram21x21-plus-energy-curves",
            "background_trust_anchor_action_003": "top25-local-latent-channel-trajectory-gram21x21-plus-energy-curves",
            "background_plus_anchor_envelope_005": "canonical-21x16x16-temporal-derivative-motion-envelope-plus-center-trajectory",
            "background_trust_anchor_envelope_003": "canonical-21x16x16-temporal-derivative-motion-envelope-plus-center-trajectory",
        }.get(runtime.sga_score_mode),
        "noise_chain_collapse": "variance_normalized_weighted",
        "anchor_model_forwards": 0,
        "target_apg_forwards": 0,
        "source_apg_forwards": 0,
        "target_raw_cfg_forwards": 0,
        "source_raw_cfg_forwards": 0,
        "target_model_forwards": 0,
        "source_model_forwards": 0,
        "anchor_present_in_every_active_target_candidate": anchor_enabled,
        "anchor_present_after_active_interval": False,
        "anchor_action_noop_attention_observed_without_transport": (
            runtime.transport == qk_transport.ACTION_NOOP_OBSERVER_ATTN_OUTPUT
            and anchor_enabled
        ),
        "anchor_value_stream_copied": False,
        "anchor_temporal_value_residual_transported": runtime.transport in (
            qk_transport.TEMPORAL_RESIDUAL_QKV,
            qk_transport.TEMPORAL_RESIDUAL_V,
        ) and anchor_enabled,
        "anchor_attention_output_residual_transported": runtime.transport
        in (
            qk_transport.TEMPORAL_RESIDUAL_ATTN_OUTPUT,
            qk_transport.TEMPORAL_CORRESPONDENCE_ATTN_OUTPUT,
            qk_transport.TEMPORAL_CONTRAST_ATTN_OUTPUT,
            qk_transport.HARD_PHASE_MEAN_CONTRAST_ATTN_OUTPUT,
            qk_transport.TEMPORAL_KERNEL_CONTRAST_ATTN_OUTPUT,
            *qk_transport.TARGET_OWNED_QK_TRANSPORTS_V14R2,
            *qk_transport.TARGET_GATED_HARD_KERNEL_TRANSPORTS,
            *qk_transport.EVENT01_ROLE_GRAPH_TRANSPORTS,
            cross_transport.TEMPORAL_CONTRAST_CROSS_ATTN_OUTPUT,
        ) and anchor_enabled,
        "anchor_phase0_semantic_correspondence": runtime.transport
        == qk_transport.TEMPORAL_CORRESPONDENCE_ATTN_OUTPUT and anchor_enabled,
        "anchor_field_velocity_residual_transported": runtime.transport
        == FIELD_VELOCITY_RESIDUAL and anchor_enabled,
        "anchor_action_noop_velocity_contrast_transported": runtime.transport
        in (
            FIELD_CONTRAST_VELOCITY,
            *TARGET_STATE_FIELD_TRANSPORTS,
            *ROLEWARP_REPLACEMENT_TRANSPORTS,
        )
        and anchor_enabled,
        "anchor_target_state_action_noop_velocity_contrast_transported": (
            runtime.transport in TARGET_STATE_FIELD_TRANSPORTS and anchor_enabled
        ),
        "native_t2v_target_velocity_hard_replacement": runtime.transport
        == FIELD_NATIVE_T2V_TARGET_VELOCITY_REPLACEMENT
        and anchor_enabled,
        "native_t2v_delta_velocity_hard_replacement": runtime.transport
        == FIELD_NATIVE_T2V_DELTA_VELOCITY_REPLACEMENT
        and anchor_enabled,
        "native_t2v_temporal_delta_hard_replacement": runtime.transport
        == FIELD_NATIVE_T2V_TEMPORAL_DELTA_REPLACEMENT
        and anchor_enabled,
        "native_t2v_sparse25_temporal_delta_hard_replacement": runtime.transport
        == FIELD_NATIVE_T2V_SPARSE25_TEMPORAL_DELTA_REPLACEMENT
        and anchor_enabled,
        "native_targetstate_temporal_delta_hard_replacement": runtime.transport
        == FIELD_NATIVE_TARGETSTATE_TEMPORAL_DELTA_REPLACEMENT
        and anchor_enabled,
        "native_targetstate_sparse25_temporal_delta_hard_replacement": runtime.transport
        == FIELD_NATIVE_TARGETSTATE_SPARSE25_TEMPORAL_DELTA_REPLACEMENT
        and anchor_enabled,
        "targetstate_raw_delta_hard_replacement": runtime.transport
        == FIELD_TARGETSTATE_RAW_DELTA_REPLACEMENT
        and anchor_enabled,
        "targetstate_sparse25_raw_delta_hard_replacement": runtime.transport
        == FIELD_TARGETSTATE_SPARSE25_RAW_DELTA_REPLACEMENT
        and anchor_enabled,
        "native_targetstate_raw_delta_hard_replacement": runtime.transport
        == FIELD_NATIVE_TARGETSTATE_RAW_DELTA_REPLACEMENT
        and anchor_enabled,
        "native_targetstate_sparse25_raw_delta_hard_replacement": runtime.transport
        == FIELD_NATIVE_TARGETSTATE_SPARSE25_RAW_DELTA_REPLACEMENT
        and anchor_enabled,
        "native_rolewarp_temporal_delta_hard_replacement": runtime.transport
        == FIELD_NATIVE_ROLEWARP_TEMPORAL_DELTA_REPLACEMENT
        and anchor_enabled,
        "native_rolewarp_sparse25_temporal_delta_hard_replacement": runtime.transport
        == FIELD_NATIVE_ROLEWARP_SPARSE25_TEMPORAL_DELTA_REPLACEMENT
        and anchor_enabled,
        "rolewarp_candidate_proposals_enabled": runtime.transport
        in ROLEWARP_REPLACEMENT_TRANSPORTS
        and anchor_enabled,
        "rolewarp_route_schedule": [],
        "rolewarp_selected_proposal_index": None,
        "event01_role_graph_attention_enabled": runtime.transport
        in qk_transport.EVENT01_ROLE_GRAPH_TRANSPORTS
        and anchor_enabled,
        "event01_role_graph_additive_logit_bias": runtime.transport
        in (
            qk_transport.EVENT01_ROLE_GRAPH_LOGIT_BIAS_ATTN_OUTPUT,
            qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_LOGIT_BIAS_ATTN_OUTPUT,
            qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_OBJECT_ATTN_OUTPUT,
            qk_transport.EVENT01_DYNAMIC_SOURCE_OBJECT_VALUE_ATTN_OUTPUT,
            qk_transport.EVENT01_DYNAMIC_SOURCE_OBJECT_OUTPUT_ATTN_OUTPUT,
            qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_MOVE_ATTN_OUTPUT,
            qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_ATTN_OUTPUT,
            qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_SIDE_ALIGNED_ATTN_OUTPUT,
        )
        and anchor_enabled,
        "event01_dynamic_role_trajectories": runtime.transport
        in (
            qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_LOGIT_BIAS_ATTN_OUTPUT,
            qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_OBJECT_ATTN_OUTPUT,
            qk_transport.EVENT01_DYNAMIC_SOURCE_OBJECT_VALUE_ATTN_OUTPUT,
            qk_transport.EVENT01_DYNAMIC_SOURCE_OBJECT_OUTPUT_ATTN_OUTPUT,
            qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_MOVE_ATTN_OUTPUT,
            qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_ATTN_OUTPUT,
            qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_SIDE_ALIGNED_ATTN_OUTPUT,
        )
        and anchor_enabled,
        "event01_source_object_phase0_value_carried": runtime.transport
        in (
            qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_OBJECT_ATTN_OUTPUT,
            qk_transport.EVENT01_DYNAMIC_SOURCE_OBJECT_VALUE_ATTN_OUTPUT,
            qk_transport.EVENT01_DYNAMIC_SOURCE_OBJECT_OUTPUT_ATTN_OUTPUT,
            qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_MOVE_ATTN_OUTPUT,
            qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_ATTN_OUTPUT,
            qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_SIDE_ALIGNED_ATTN_OUTPUT,
        )
        and anchor_enabled,
        "event01_source_object_value_hard_routed": runtime.transport
        == qk_transport.EVENT01_DYNAMIC_SOURCE_OBJECT_VALUE_ATTN_OUTPUT
        and anchor_enabled,
        "event01_source_object_attention_output_hard_routed": runtime.transport
        == qk_transport.EVENT01_DYNAMIC_SOURCE_OBJECT_OUTPUT_ATTN_OUTPUT
        and anchor_enabled,
        "event01_source_object_spatial_patch_value_hard_routed": (
            runtime.transport
            in (
                qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_ATTN_OUTPUT,
                qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_SIDE_ALIGNED_ATTN_OUTPUT,
            )
            and anchor_enabled
        ),
        "event01_source_object_spatial_patch_output_hard_routed": runtime.transport
        == qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_MOVE_ATTN_OUTPUT
        and anchor_enabled,
        "event01_source_object_explicit_source_branch_patch_carried": (
            runtime.transport
            in (
                qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_MOVE_ATTN_OUTPUT,
                qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_ATTN_OUTPUT,
                qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_SIDE_ALIGNED_ATTN_OUTPUT,
            )
            and anchor_enabled
        ),
        "event01_source_object_relation_source_side_aligned": runtime.transport
        == qk_transport.EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_SIDE_ALIGNED_ATTN_OUTPUT
        and anchor_enabled,
        "event01_role_graph_route_schedule": [],
        "event01_role_graph_selected_proposal_index": None,
        "anchor_native_phase_envelope_gated_target_state_velocity": (
            runtime.transport == FIELD_NATIVE_GATED_TARGET_CONTRAST_VELOCITY
            and anchor_enabled
        ),
        "anchor_action_noop_attention_contrast_transported": runtime.transport
        in (
            qk_transport.TEMPORAL_CONTRAST_QK,
            qk_transport.TEMPORAL_CONTRAST_ATTN_OUTPUT,
            qk_transport.TEMPORAL_CORRESPONDENCE_CONTRAST_QK,
            qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK,
            qk_transport.TEMPORAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
            qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_ATTN_OUTPUT,
            qk_transport.TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_QK,
            qk_transport.TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
            qk_transport.HARD_PHASE_MEAN_CONTRAST_QK,
            qk_transport.HARD_PHASE_MEAN_CONTRAST_ATTN_OUTPUT,
            qk_transport.HARD_PREROPE_PHASE_MEAN_CONTRAST_QK,
            qk_transport.TEMPORAL_KERNEL_CONTRAST_ATTN_OUTPUT,
            *qk_transport.TARGET_OWNED_QK_TRANSPORTS_V14R2,
            *qk_transport.TARGET_GATED_HARD_KERNEL_TRANSPORTS,
            *qk_transport.EVENT01_ROLE_GRAPH_TRANSPORTS,
            cross_transport.TEMPORAL_CONTRAST_CROSS_ATTN_OUTPUT,
        ) and anchor_enabled,
        "anchor_phase0_correspondence_contrast_transported": runtime.transport
        in (
            qk_transport.TEMPORAL_CORRESPONDENCE_CONTRAST_QK,
            qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK,
            qk_transport.TEMPORAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
            qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_ATTN_OUTPUT,
            qk_transport.TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_QK,
            qk_transport.TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
        )
        and anchor_enabled,
        "anchor_hard_temporal_trajectory_replacement": runtime.transport
        in (
            qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK,
            qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_ATTN_OUTPUT,
            qk_transport.HARD_PHASE_MEAN_CONTRAST_QK,
            qk_transport.HARD_PHASE_MEAN_CONTRAST_ATTN_OUTPUT,
            qk_transport.HARD_PREROPE_PHASE_MEAN_CONTRAST_QK,
            *qk_transport.TARGET_GATED_HARD_KERNEL_TRANSPORTS,
            qk_transport.EVENT01_ROLE_GRAPH_HARD_ATTN_OUTPUT,
        )
        and anchor_enabled,
        "anchor_absolute_qk_or_k_replacement": runtime.transport
        in (qk_transport.HARD_QK, qk_transport.HARD_K)
        and anchor_enabled,
        "anchor_absolute_q_replacement_early_blocks": runtime.transport
        in qk_transport.DUAL_SOURCE_KV_TRANSPORTS
        and anchor_enabled,
        "source_target_kv_replay_late_blocks": runtime.transport
        in qk_transport.DUAL_SOURCE_KV_TRANSPORTS
        and anchor_enabled,
        "source_target_kv_replay_uses_target_rope_reprojection": runtime.transport
        in qk_transport.DUAL_SOURCE_KV_TRANSPORTS
        and anchor_enabled,
        "source_target_kv_replay_static_fraction": (
            0.75
            if runtime.transport
            == qk_transport.DUAL_HARD_Q_EARLY_SOURCE_KV_LATE_STATIC75
            else 1.0
            if runtime.transport
            == qk_transport.DUAL_HARD_Q_EARLY_SOURCE_KV_LATE_ALL
            else 0.0
        ),
        "anchor_coordinate_free_phase_mean_contrast": runtime.transport
        in (
            qk_transport.HARD_PHASE_MEAN_CONTRAST_QK,
            qk_transport.HARD_PHASE_MEAN_CONTRAST_ATTN_OUTPUT,
            qk_transport.HARD_PREROPE_PHASE_MEAN_CONTRAST_QK,
        )
        and anchor_enabled,
        "anchor_prerope_hidden_phase_mean_contrast": runtime.transport
        == qk_transport.HARD_PREROPE_PHASE_MEAN_CONTRAST_QK
        and anchor_enabled,
        "anchor_temporal_attention_kernel_contrast": runtime.transport
        in (
            qk_transport.TEMPORAL_KERNEL_CONTRAST_ATTN_OUTPUT,
            *qk_transport.TARGET_OWNED_QK_TRANSPORTS_V14R2,
            *qk_transport.TARGET_GATED_HARD_KERNEL_TRANSPORTS,
        )
        and anchor_enabled,
        "anchor_temporal_kernel_applied_to_target_value_only": runtime.transport
        in (
            qk_transport.TEMPORAL_KERNEL_CONTRAST_ATTN_OUTPUT,
            *qk_transport.TARGET_OWNED_QK_TRANSPORTS_V14R2,
            *qk_transport.TARGET_GATED_HARD_KERNEL_TRANSPORTS,
        )
        and anchor_enabled,
        "anchor_target_activity_gated_hard_kernel": runtime.transport
        in (
            *qk_transport.TARGET_GATED_HARD_KERNEL_TRANSPORTS,
            qk_transport.TARGET_OWNED_ACTIVITY_KERNEL_TOP10_ATTN_OUTPUT_V14R2,
            qk_transport.TARGET_OWNED_ACTIVITY_KERNEL_TOP25_ATTN_OUTPUT_V14R2,
        )
        and anchor_enabled,
        "target_owned_qk_route_v14r2": runtime.transport
        in qk_transport.TARGET_OWNED_QK_TRANSPORTS_V14R2
        and anchor_enabled,
        "anchor_donor_cached_fields": (
            ["query", "key"]
            if runtime.transport in qk_transport.TARGET_OWNED_QK_TRANSPORTS_V14R2
            and anchor_enabled
            else None
        ),
        "anchor_donor_value_hidden_output_or_coordinate_used": False
        if runtime.transport in qk_transport.TARGET_OWNED_QK_TRANSPORTS_V14R2
        and anchor_enabled
        else None,
        "anchor_to_target_appearance_correspondence_used": False
        if runtime.transport in qk_transport.TARGET_OWNED_QK_TRANSPORTS_V14R2
        and anchor_enabled
        else None,
        "anchor_mutual_correspondence_gate": runtime.transport
        in (
            qk_transport.TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_QK,
            qk_transport.TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
        )
        and anchor_enabled,
        "anchor_cross_attention_output_contrast_transported": (
            uses_cross_attention_transport and anchor_enabled
        ),
        "source_value_stream_retained": runtime.transport
        not in qk_transport.DUAL_SOURCE_KV_TRANSPORTS,
        "anchor_route_shared_by_target_negative_and_condition": (
            runtime.field_guidance in ("apg", "raw_cfg")
            and runtime.anchor_cfg_scope == "shared"
        ),
        "anchor_route_target_conditional_only": (
            runtime.anchor_cfg_scope == "target_conditional_only"
        ),
        "initial_latent_phase_clamped_after_every_update": runtime.initial_phase_clamp,
    }

    patch_context = (
        cross_transport.install_anchor_cross_attention_transport(
            transformer, selected_block_indices=runtime.selected_block_indices
        )
        if uses_cross_attention_transport
        else qk_transport.install_anchor_qk_transport(
            transformer, selected_block_indices=runtime.selected_block_indices
        )
    )
    with patch_context as patch_handle:
        # Use the handle-owned bank so processors and invocations share identity.
        cache_bank = patch_handle.cache_bank
        with torch.no_grad():
            if runtime.field_model == "source_conditioned_rv2v":
                source_condition_latent = source_clean
            elif runtime.field_model in (
                "first_phase_source_rv2v",
                "first_phase_caption_i2v",
            ):
                source_condition_latent = source_clean[:, :, :1].repeat(
                    1, 1, int(source_clean.shape[2]), 1, 1
                )
            else:
                source_condition_latent = None
            source_condition = (
                cdf._patch_source_condition(transformer, source_condition_latent)
                if source_condition_latent is not None
                else None
            )
            for step_index, (sigma_value, next_sigma_value) in enumerate(intervals):
                count = (
                    early_candidate_count
                    if step_index < EARLY_CANDIDATE_STEPS
                    else 1
                )
                if len(previous_noises) != count:
                    raise AnchorSGAANCError("candidate-zero continuation differs")
                retained = (
                    source_aligned.anc_retained_variance(
                        float(sigma_value), lock_sigma=runtime.anc_lock_sigma
                    )
                    if runtime.uses_anc
                    else 0.0
                )
                noises = []
                deltas = []
                anchor_active = step_index < runtime.transport_steps
                native_action_state: Optional[torch.Tensor] = None
                native_noop_state: Optional[torch.Tensor] = None
                native_action_path_velocity: Optional[torch.Tensor] = None
                native_noop_path_velocity: Optional[torch.Tensor] = None
                if (
                    anchor_active
                    and runtime.anchor_state_mode == "native_t2v_trajectory"
                ):
                    if (
                        native_anchor_action_packed is None
                        or native_anchor_noop_packed is None
                    ):
                        raise AnchorSGAANCError("native T2V trajectory state is absent")
                    native_action_state = cdf._unpack_spatial_latent(
                        native_anchor_action_packed, layout
                    )
                    native_noop_state = cdf._unpack_spatial_latent(
                        native_anchor_noop_packed, layout
                    )
                anchor_sigma_value = min(
                    float(sigma_value), float(runtime.anchor_sigma_cap)
                )
                anchor_timestep = timestep = timesteps[step_index]
                if anchor_active:
                    anchor_timestep = _model_timestep_for_anchor_sigma(
                        outer_timestep=timestep,
                        outer_sigma=float(sigma_value),
                        anchor_sigma=anchor_sigma_value,
                        num_train_timesteps=num_train_timesteps,
                    )
                    trace["anchor_active_schedule"].append(
                        {
                            "step_index": step_index,
                            "candidate_count": count,
                            "outer_sigma": float(sigma_value),
                            "outer_timestep": float(timestep.detach().cpu().item()),
                            "anchor_sigma": anchor_sigma_value,
                            "anchor_timestep": float(
                                anchor_timestep.detach().cpu().item()
                            ),
                            "cap_applied": anchor_sigma_value < float(sigma_value),
                        }
                    )
                    trace["anchor_candidate_cells"] += count
                if (
                    anchor_active
                    and runtime.anchor_state_mode == "native_t2v_trajectory"
                ):
                    if native_action_state is None or native_noop_state is None:
                        raise AnchorSGAANCError(
                            "native T2V trajectory state was not materialized"
                        )
                    native_action_path_velocity = _guided_source_free_apg_velocity(
                        diffusion=diffusion,
                        transformer=transformer,
                        query_state=native_action_state,
                        condition_prompt_embeds=anchor_prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        timestep=timestep,
                        sigma=scheduler_sigmas[step_index],
                        branch="anchor_action_trajectory",
                        adapter_controller=renderer_or_diffusion,
                    )
                    native_noop_path_velocity = _guided_source_free_apg_velocity(
                        diffusion=diffusion,
                        transformer=transformer,
                        query_state=native_noop_state,
                        condition_prompt_embeds=anchor_noop_prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        timestep=timestep,
                        sigma=scheduler_sigmas[step_index],
                        branch="anchor_noop_trajectory",
                        adapter_controller=renderer_or_diffusion,
                    )
                    trace["anchor_native_trajectory_model_forwards"] += 4
                for candidate_index in range(count):
                    fresh, _ = guided._draw_keyed_packed_noise(
                        source_latent=source_clean,
                        layout=layout,
                        seed=runtime.seed,
                        step=step_index,
                        candidate=candidate_index,
                    )
                    if (
                        step_index == 0
                        and candidate_index == 0
                        and runtime.initial_noise_proposal_mode != "keyed_only"
                        and anchor_initial_packed is not None
                    ):
                        fresh = anchor_initial_packed.clone()
                        trace[
                            "anchor_initial_gaussian_used_at_step0_candidate0"
                        ] = True
                    noise = (
                        source_aligned.advance_anc_noise(
                            previous_noises[candidate_index],
                            fresh,
                            retained_variance=retained,
                        )
                        if runtime.uses_anc
                        else fresh
                    )
                    noises.append(noise)
                    source_state_packed, target_state_packed = (
                        source_aligned.flowedit_source_target_states(
                            source_packed,
                            edit_packed,
                            noise,
                            sigma=float(sigma_value),
                        )
                    )
                    source_state = cdf._unpack_spatial_latent(source_state_packed, layout)
                    target_state = cdf._unpack_spatial_latent(target_state_packed, layout)
                    role_proposal_index = (
                        runtime.event01_forced_role_proposal_index
                        if runtime.event01_forced_role_proposal_index >= 0
                        else candidate_index
                        if (
                            runtime.transport
                            in qk_transport.EVENT01_ROLE_GRAPH_TRANSPORTS
                            and step_index < EARLY_CANDIDATE_STEPS
                        )
                        else selected_role_proposal_index
                        if runtime.transport
                        in qk_transport.EVENT01_ROLE_GRAPH_TRANSPORTS
                        else 0
                    )
                    replay_uses = (
                        2
                        if runtime.field_guidance in ("apg", "raw_cfg")
                        and runtime.anchor_cfg_scope == "shared"
                        else 1
                    )
                    replay_scope = (
                        qk_transport.PAIRED_SUFFIX
                        if runtime.uses_rv2v_condition
                        else qk_transport.FULL_SEQUENCE
                    )
                    negative_context = None
                    condition_context = None
                    anchor_velocity = None
                    anchor_noop_velocity = None
                    target_state_anchor_velocity = None
                    target_state_anchor_noop_velocity = None
                    if anchor_active:
                        if (
                            runtime.transport
                            in qk_transport.EVENT01_ROLE_GRAPH_TRANSPORTS
                        ):
                            trace["event01_role_graph_route_schedule"].append(
                                {
                                    "step_index": step_index,
                                    "candidate_index": candidate_index,
                                    "proposal_index": role_proposal_index,
                                }
                            )
                        if runtime.anchor_candidate_mode == "bank_per_candidate":
                            if step_index < EARLY_CANDIDATE_STEPS:
                                active_anchor_packed = anchor_packed_bank[candidate_index]
                                active_anchor_static_packed = anchor_static_packed_bank[
                                    candidate_index
                                ]
                            else:
                                if (
                                    collapsed_anchor_packed is None
                                    or collapsed_anchor_static_packed is None
                                ):
                                    raise AnchorSGAANCError(
                                        "anchor bank was not collapsed after early SGA"
                                    )
                                active_anchor_packed = collapsed_anchor_packed
                                active_anchor_static_packed = collapsed_anchor_static_packed
                        else:
                            active_anchor_packed = anchor_packed_bank[0]
                            active_anchor_static_packed = anchor_static_packed_bank[0]
                        if runtime.anchor_state_mode == "native_t2v_trajectory":
                            if native_action_state is None or native_noop_state is None:
                                raise AnchorSGAANCError(
                                    "native T2V trajectory state was not materialized"
                                )
                            anchor_state = native_action_state
                            anchor_reference_state = native_noop_state
                            anchor_reference_prompt_embeds = anchor_noop_prompt_embeds
                        else:
                            anchor_state_packed = (
                                (1.0 - anchor_sigma_value) * active_anchor_packed
                                + anchor_sigma_value * noise
                            )
                            anchor_state = cdf._unpack_spatial_latent(
                                anchor_state_packed, layout
                            )
                            anchor_reference_state = anchor_state
                            anchor_reference_prompt_embeds = anchor_noop_prompt_embeds
                            if (
                                runtime.anchor_contrast_mode
                                == "dynamic_static_same_caption"
                            ):
                                anchor_reference_state_packed = (
                                    (1.0 - anchor_sigma_value)
                                    * active_anchor_static_packed
                                    + anchor_sigma_value * noise
                                )
                                anchor_reference_state = cdf._unpack_spatial_latent(
                                    anchor_reference_state_packed, layout
                                )
                                anchor_reference_prompt_embeds = anchor_prompt_embeds
                        if runtime.transport in (
                            FIELD_VELOCITY_RESIDUAL,
                            FIELD_CONTRAST_VELOCITY,
                        ):
                            if runtime.anchor_state_mode == "native_t2v_trajectory":
                                if native_action_path_velocity is None:
                                    raise AnchorSGAANCError(
                                        "native action trajectory velocity is absent"
                                    )
                                anchor_velocity = native_action_path_velocity
                                if runtime.transport == FIELD_CONTRAST_VELOCITY:
                                    if native_noop_path_velocity is None:
                                        raise AnchorSGAANCError(
                                            "native no-op trajectory velocity is absent"
                                        )
                                    anchor_noop_velocity = native_noop_path_velocity
                            else:
                                anchor_velocity = _predict_source_free_velocity(
                                    diffusion=diffusion,
                                    transformer=transformer,
                                    query_state=anchor_state,
                                    prompt_embeds=anchor_prompt_embeds,
                                    timestep=anchor_timestep,
                                    adapter_controller=renderer_or_diffusion,
                                )
                                if runtime.transport == FIELD_CONTRAST_VELOCITY:
                                    anchor_noop_velocity = _predict_source_free_velocity(
                                        diffusion=diffusion,
                                        transformer=transformer,
                                        query_state=anchor_reference_state,
                                        prompt_embeds=anchor_reference_prompt_embeds,
                                        timestep=anchor_timestep,
                                        adapter_controller=renderer_or_diffusion,
                                    )
                        elif runtime.transport in (
                            *TARGET_STATE_FIELD_TRANSPORTS,
                            *ROLEWARP_REPLACEMENT_TRANSPORTS,
                        ):
                            # These velocity transports bypass block capture.
                            # Target-state routes are queried below per
                            # candidate; role-warp routes instead consume the
                            # shared exact native trajectory computed above.
                            pass
                        elif uses_cross_attention_transport:
                            _capture_anchor_cross_attention(
                                diffusion=diffusion,
                                transformer=transformer,
                                anchor_state=anchor_state,
                                anchor_prompt_embeds=anchor_prompt_embeds,
                                timestep=anchor_timestep,
                                cache_bank=cache_bank,
                                step_index=step_index,
                                candidate_index=candidate_index,
                                transport_strength=runtime.transport_strength,
                                replay_uses=replay_uses,
                                replay_scope=replay_scope,
                                adapter_controller=renderer_or_diffusion,
                            )
                            _capture_anchor_cross_attention(
                                diffusion=diffusion,
                                transformer=transformer,
                                anchor_state=anchor_reference_state,
                                anchor_prompt_embeds=anchor_reference_prompt_embeds,
                                timestep=anchor_timestep,
                                cache_bank=cache_bank,
                                step_index=step_index,
                                candidate_index=candidate_index,
                                transport_strength=runtime.transport_strength,
                                replay_uses=replay_uses,
                                replay_scope=replay_scope,
                                slot=cross_transport.NOOP_SLOT,
                                adapter_controller=renderer_or_diffusion,
                            )
                        else:
                            _capture_anchor_qk(
                                diffusion=diffusion,
                                transformer=transformer,
                                anchor_state=anchor_state,
                                anchor_prompt_embeds=anchor_prompt_embeds,
                                timestep=anchor_timestep,
                                cache_bank=cache_bank,
                                step_index=step_index,
                                candidate_index=candidate_index,
                                transport=runtime.transport,
                                transport_strength=runtime.transport_strength,
                                replay_uses=replay_uses,
                                replay_scope=replay_scope,
                                role_proposal_index=role_proposal_index,
                                adapter_controller=renderer_or_diffusion,
                            )
                            if (
                                runtime.transport
                                in (
                                    qk_transport.TEMPORAL_CONTRAST_QK,
                                    qk_transport.TEMPORAL_CONTRAST_ATTN_OUTPUT,
                                    qk_transport.TEMPORAL_CORRESPONDENCE_CONTRAST_QK,
                                    qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK,
                                    qk_transport.TEMPORAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
                                    qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_ATTN_OUTPUT,
                                    qk_transport.TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_QK,
                                    qk_transport.TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
                                    qk_transport.HARD_PHASE_MEAN_CONTRAST_QK,
                                    qk_transport.HARD_PHASE_MEAN_CONTRAST_ATTN_OUTPUT,
                                    qk_transport.HARD_PREROPE_PHASE_MEAN_CONTRAST_QK,
                                    qk_transport.TEMPORAL_KERNEL_CONTRAST_ATTN_OUTPUT,
                                    *qk_transport.TARGET_OWNED_QK_TRANSPORTS_V14R2,
                                    qk_transport.ACTION_NOOP_OBSERVER_ATTN_OUTPUT,
                                    *qk_transport.TARGET_GATED_HARD_KERNEL_TRANSPORTS,
                                    *qk_transport.EVENT01_ROLE_GRAPH_TRANSPORTS,
                                )
                            ):
                                _capture_anchor_qk(
                                    diffusion=diffusion,
                                    transformer=transformer,
                                    anchor_state=anchor_reference_state,
                                    anchor_prompt_embeds=anchor_reference_prompt_embeds,
                                    timestep=anchor_timestep,
                                    cache_bank=cache_bank,
                                    step_index=step_index,
                                    candidate_index=candidate_index,
                                    transport=runtime.transport,
                                    transport_strength=runtime.transport_strength,
                                    replay_uses=replay_uses,
                                    replay_scope=replay_scope,
                                    slot=qk_transport.NOOP_SLOT,
                                    role_proposal_index=role_proposal_index,
                                    adapter_controller=renderer_or_diffusion,
                                )
                        trace["anchor_model_forwards"] += (
                            0
                            if runtime.transport
                            in (
                                *TARGET_STATE_FIELD_TRANSPORTS,
                                *ROLEWARP_REPLACEMENT_TRANSPORTS,
                            )
                            or runtime.anchor_state_mode == "native_t2v_trajectory"
                            and runtime.transport
                            in (FIELD_VELOCITY_RESIDUAL, FIELD_CONTRAST_VELOCITY)
                            else 2
                            if runtime.transport in (
                                FIELD_CONTRAST_VELOCITY,
                                qk_transport.TEMPORAL_CONTRAST_QK,
                                qk_transport.TEMPORAL_CONTRAST_ATTN_OUTPUT,
                                qk_transport.TEMPORAL_CORRESPONDENCE_CONTRAST_QK,
                                qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK,
                                qk_transport.TEMPORAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
                                qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_ATTN_OUTPUT,
                                qk_transport.TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_QK,
                                qk_transport.TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
                                qk_transport.HARD_PHASE_MEAN_CONTRAST_QK,
                                qk_transport.HARD_PHASE_MEAN_CONTRAST_ATTN_OUTPUT,
                                qk_transport.HARD_PREROPE_PHASE_MEAN_CONTRAST_QK,
                                qk_transport.TEMPORAL_KERNEL_CONTRAST_ATTN_OUTPUT,
                                *qk_transport.TARGET_OWNED_QK_TRANSPORTS_V14R2,
                                qk_transport.ACTION_NOOP_OBSERVER_ATTN_OUTPUT,
                                *qk_transport.TARGET_GATED_HARD_KERNEL_TRANSPORTS,
                                *qk_transport.EVENT01_ROLE_GRAPH_TRANSPORTS,
                                cross_transport.TEMPORAL_CONTRAST_CROSS_ATTN_OUTPUT,
                            )
                            else 1
                        )
                        if runtime.transport not in (
                            FIELD_VELOCITY_RESIDUAL,
                            FIELD_CONTRAST_VELOCITY,
                            *TARGET_STATE_FIELD_TRANSPORTS,
                            *ROLEWARP_REPLACEMENT_TRANSPORTS,
                        ):
                            rank, size = _parallel_identity()
                            if uses_cross_attention_transport:
                                replay_condition = (
                                    cross_transport.AnchorCrossAttentionInvocation(
                                        cross_transport.REPLAY,
                                        cache_bank,
                                        step_index=step_index,
                                        candidate_index=candidate_index,
                                        rank=rank,
                                        ulysses_size=size,
                                        transport_strength=runtime.transport_strength,
                                        replay_uses=replay_uses,
                                        replay_scope=replay_scope,
                                    )
                                )
                                condition_context = (
                                    cross_transport.anchor_cross_attention_invocation(
                                        replay_condition
                                    )
                                )
                            else:
                                replay_condition = qk_transport.AnchorQKInvocation(
                                    qk_transport.REPLAY,
                                    cache_bank,
                                    step_index=step_index,
                                    candidate_index=candidate_index,
                                    rank=rank,
                                    ulysses_size=size,
                                    transport=runtime.transport,
                                    transport_strength=runtime.transport_strength,
                                    replay_uses=replay_uses,
                                    replay_scope=replay_scope,
                                    role_proposal_index=role_proposal_index,
                                )
                                condition_context = qk_transport.anchor_qk_invocation(
                                    replay_condition
                                )
                            if (
                                runtime.field_guidance in ("apg", "raw_cfg")
                                and runtime.anchor_cfg_scope == "shared"
                            ):
                                if uses_cross_attention_transport:
                                    replay_negative = (
                                        cross_transport.AnchorCrossAttentionInvocation(
                                            cross_transport.REPLAY,
                                            cache_bank,
                                            step_index=step_index,
                                            candidate_index=candidate_index,
                                            rank=rank,
                                            ulysses_size=size,
                                            transport_strength=runtime.transport_strength,
                                            replay_uses=replay_uses,
                                            replay_scope=replay_scope,
                                        )
                                    )
                                    negative_context = (
                                        cross_transport.anchor_cross_attention_invocation(
                                            replay_negative
                                        )
                                    )
                                else:
                                    replay_negative = qk_transport.AnchorQKInvocation(
                                        qk_transport.REPLAY,
                                        cache_bank,
                                        step_index=step_index,
                                        candidate_index=candidate_index,
                                        rank=rank,
                                        ulysses_size=size,
                                        transport=runtime.transport,
                                        transport_strength=runtime.transport_strength,
                                        replay_uses=replay_uses,
                                        replay_scope=replay_scope,
                                        role_proposal_index=role_proposal_index,
                                    )
                                    negative_context = qk_transport.anchor_qk_invocation(
                                        replay_negative
                                    )
                    if runtime.field_guidance == "apg":
                        target_result = guided._guided_apg_velocity(
                            diffusion=diffusion,
                            transformer=transformer,
                            source_condition=source_condition,
                            query_latent=target_state,
                            condition_prompt_embeds=action_prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            timestep=timestep,
                            sigma=scheduler_sigmas[step_index],
                            branch="target_action",
                            negative_context=negative_context,
                            condition_context=condition_context,
                        )
                        source_result = guided._guided_apg_velocity(
                            diffusion=diffusion,
                            transformer=transformer,
                            source_condition=source_condition,
                            query_latent=source_state,
                            condition_prompt_embeds=noop_prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            timestep=timestep,
                            sigma=scheduler_sigmas[step_index],
                            branch="source_noop",
                        )
                        target_velocity = target_result.velocity_packed_fp32
                        source_velocity = source_result.velocity_packed_fp32
                        trace["target_apg_forwards"] += 2
                        trace["source_apg_forwards"] += 2
                        trace["target_model_forwards"] += 2
                        trace["source_model_forwards"] += 2
                    else:
                        target_field_prompt = (
                            target_t2v_prompt_embeds
                            if runtime.uses_source_target_captions
                            else action_prompt_embeds
                        )
                        source_field_prompt = (
                            source_t2v_prompt_embeds
                            if runtime.uses_source_target_captions
                            else noop_prompt_embeds
                        )
                        active_context = (
                            contextlib.nullcontext()
                            if condition_context is None
                            else condition_context
                        )
                        with active_context:
                            target_condition_velocity = _predict_field_velocity(
                                diffusion=diffusion,
                                transformer=transformer,
                                source_condition=source_condition,
                                query_state=target_state,
                                prompt_embeds=target_field_prompt,
                                timestep=timestep,
                                uses_rv2v_condition=runtime.uses_rv2v_condition,
                            )
                        source_condition_velocity = _predict_field_velocity(
                            diffusion=diffusion,
                            transformer=transformer,
                            source_condition=source_condition,
                            query_state=source_state,
                            prompt_embeds=source_field_prompt,
                            timestep=timestep,
                            uses_rv2v_condition=runtime.uses_rv2v_condition,
                        )
                        if runtime.field_guidance == "raw_cfg":
                            target_negative_scope = (
                                contextlib.nullcontext()
                                if negative_context is None
                                else negative_context
                            )
                            with target_negative_scope:
                                target_negative_velocity = _predict_field_velocity(
                                    diffusion=diffusion,
                                    transformer=transformer,
                                    source_condition=source_condition,
                                    query_state=target_state,
                                    prompt_embeds=negative_prompt_embeds,
                                    timestep=timestep,
                                    uses_rv2v_condition=runtime.uses_rv2v_condition,
                                )
                            source_negative_velocity = _predict_field_velocity(
                                diffusion=diffusion,
                                transformer=transformer,
                                source_condition=source_condition,
                                query_state=source_state,
                                prompt_embeds=negative_prompt_embeds,
                                timestep=timestep,
                                uses_rv2v_condition=runtime.uses_rv2v_condition,
                            )
                            target_velocity = target_negative_velocity + float(
                                runtime.target_cfg_scale
                            ) * (target_condition_velocity - target_negative_velocity)
                            source_velocity = source_negative_velocity + float(
                                runtime.source_cfg_scale
                            ) * (source_condition_velocity - source_negative_velocity)
                            trace["target_raw_cfg_forwards"] += 2
                            trace["source_raw_cfg_forwards"] += 2
                            trace["target_model_forwards"] += 2
                            trace["source_model_forwards"] += 2
                        else:
                            target_velocity = target_condition_velocity
                            source_velocity = source_condition_velocity
                            trace["target_model_forwards"] += 1
                            trace["source_model_forwards"] += 1
                    if anchor_active and runtime.transport in TARGET_STATE_FIELD_TRANSPORTS:
                        target_state_anchor_velocity = _guided_source_free_apg_velocity(
                            diffusion=diffusion,
                            transformer=transformer,
                            query_state=target_state,
                            condition_prompt_embeds=anchor_prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            timestep=timestep,
                            sigma=scheduler_sigmas[step_index],
                            branch="anchor_action_trajectory",
                            adapter_controller=renderer_or_diffusion,
                        )
                        target_state_anchor_noop_velocity = (
                            _guided_source_free_apg_velocity(
                                diffusion=diffusion,
                                transformer=transformer,
                                query_state=target_state,
                                condition_prompt_embeds=anchor_noop_prompt_embeds,
                                negative_prompt_embeds=negative_prompt_embeds,
                                timestep=timestep,
                                sigma=scheduler_sigmas[step_index],
                                branch="anchor_noop_trajectory",
                                adapter_controller=renderer_or_diffusion,
                            )
                        )
                        trace["anchor_model_forwards"] += 4
                        if (
                            candidate_index == 0
                            and runtime.transport
                            in NATIVE_T2V_REPLACEMENT_TRANSPORTS
                        ):
                            if (
                                native_action_path_velocity is None
                                or native_noop_path_velocity is None
                            ):
                                raise AnchorSGAANCError(
                                    "native replacement lacks its exact trajectory control"
                                )
                            if step_index == 0:
                                trace[
                                    "native_target_prebind_step0_candidate0_action_mse_to_anchor_path"
                                ] = float(
                                    (
                                        target_state_anchor_velocity
                                        - native_action_path_velocity
                                    )
                                    .float()
                                    .square()
                                    .mean()
                                    .detach()
                                    .cpu()
                                    .item()
                                )
                                trace[
                                    "native_target_prebind_step0_candidate0_noop_mse_to_anchor_path"
                                ] = float(
                                    (
                                        target_state_anchor_noop_velocity
                                        - native_noop_path_velocity
                                    )
                                    .float()
                                    .square()
                                    .mean()
                                    .detach()
                                    .cpu()
                                    .item()
                                )
                            target_state_anchor_velocity = native_action_path_velocity
                            target_state_anchor_noop_velocity = native_noop_path_velocity
                            trace[
                                "native_path_velocity_bound_to_candidate0_schedule"
                            ].append(step_index)
                        if (
                            step_index == 0
                            and candidate_index == 0
                            and runtime.transport
                            in NATIVE_T2V_REPLACEMENT_TRANSPORTS
                        ):
                            trace[
                                "native_target_replacement_step0_candidate0_action_mse_to_anchor_path"
                            ] = float(
                                (
                                    target_state_anchor_velocity
                                    - native_action_path_velocity
                                )
                                .float()
                                .square()
                                .mean()
                                .detach()
                                .cpu()
                                .item()
                            )
                            trace[
                                "native_target_replacement_step0_candidate0_noop_mse_to_anchor_path"
                            ] = float(
                                (
                                    target_state_anchor_noop_velocity
                                    - native_noop_path_velocity
                                )
                                .float()
                                .square()
                                .mean()
                                .detach()
                                .cpu()
                                .item()
                            )
                    if (
                        anchor_active
                        and runtime.transport in ROLEWARP_REPLACEMENT_TRANSPORTS
                    ):
                        if (
                            native_action_path_velocity is None
                            or native_noop_path_velocity is None
                        ):
                            raise AnchorSGAANCError(
                                "role-warp replacement lacks native action/no-op path"
                            )
                        native_route = _sparse_packed_action_contrast(
                            native_action_path_velocity,
                            native_noop_path_velocity,
                            strength=1.0,
                            keep_fraction=1.0,
                        )
                        proposal_index = (
                            candidate_index
                            if step_index < EARLY_CANDIDATE_STEPS
                            else selected_role_proposal_index
                        )
                        keep_fraction = (
                            1.0
                            if runtime.transport
                            == FIELD_NATIVE_ROLEWARP_TEMPORAL_DELTA_REPLACEMENT
                            else 0.25
                        )
                        native_route, role_audit = _event01_role_warp_native_route(
                            native_route,
                            layout=layout,
                            proposal_index=proposal_index,
                            keep_fraction=keep_fraction,
                        )
                        trace["rolewarp_route_schedule"].append(
                            {
                                "step_index": step_index,
                                "candidate_index": candidate_index,
                                **role_audit,
                            }
                        )
                        target_velocity = source_velocity + native_route
                    if anchor_velocity is not None:
                        if anchor_noop_velocity is None:
                            target_velocity = _sparse_packed_temporal_residual(
                                target_velocity,
                                anchor_velocity,
                                strength=runtime.transport_strength,
                            )
                        else:
                            action_route = _sparse_packed_action_contrast(
                                anchor_velocity,
                                anchor_noop_velocity,
                                strength=runtime.transport_strength,
                            )
                            if runtime.anchor_spatial_alignment == "motion_support_affine":
                                if sga_source_support is None:
                                    raise AnchorSGAANCError(
                                        "source support is absent for anchor alignment"
                                    )
                                action_route, alignment_audit = (
                                    _align_packed_route_to_source_motion(
                                        action_route,
                                        sga_source_support,
                                        layout=layout,
                                    )
                                )
                                if step_index < EARLY_CANDIDATE_STEPS:
                                    trace["anchor_spatial_alignment_audits"].append(
                                        {
                                            "step_index": step_index,
                                            "candidate_index": candidate_index,
                                            **alignment_audit,
                                        }
                                )
                            target_velocity = target_velocity + action_route
                    if target_state_anchor_velocity is not None:
                        if target_state_anchor_noop_velocity is None:
                            raise AnchorSGAANCError(
                                "target-state T2V no-op velocity is absent"
                            )
                        if runtime.transport in NATIVE_T2V_REPLACEMENT_TRANSPORTS:
                            target_velocity = _apply_native_t2v_hard_replacement(
                                target_velocity=target_velocity,
                                source_velocity=source_velocity,
                                action_velocity=target_state_anchor_velocity,
                                noop_velocity=target_state_anchor_noop_velocity,
                                transport=runtime.transport,
                            )
                        elif runtime.transport in TARGETSTATE_TEMPORAL_REPLACEMENT_TRANSPORTS:
                            keep_fraction = (
                                1.0
                                if runtime.transport
                                == FIELD_NATIVE_TARGETSTATE_TEMPORAL_DELTA_REPLACEMENT
                                else 0.25
                            )
                            action_route = _sparse_packed_action_contrast(
                                target_state_anchor_velocity,
                                target_state_anchor_noop_velocity,
                                strength=1.0,
                                keep_fraction=keep_fraction,
                            )
                            if (
                                native_action_path_velocity is None
                                or native_noop_path_velocity is None
                            ):
                                raise AnchorSGAANCError(
                                    "target-state temporal replacement lacks native timing"
                                )
                            action_route, phase_envelope = _apply_native_phase_envelope(
                                action_route,
                                native_action_path_velocity,
                                native_noop_path_velocity,
                            )
                            if candidate_index == 0:
                                trace["anchor_native_phase_envelopes"].append(
                                    {
                                        "step_index": step_index,
                                        "values": phase_envelope,
                                    }
                                )
                            target_velocity = source_velocity + action_route
                        elif runtime.transport in TARGETSTATE_RAW_REPLACEMENT_TRANSPORTS:
                            keep_fraction = (
                                1.0
                                if runtime.transport
                                in (
                                    FIELD_TARGETSTATE_RAW_DELTA_REPLACEMENT,
                                    FIELD_NATIVE_TARGETSTATE_RAW_DELTA_REPLACEMENT,
                                )
                                else 0.25
                            )
                            action_route = _sparse_packed_raw_action_contrast(
                                target_state_anchor_velocity,
                                target_state_anchor_noop_velocity,
                                strength=1.0,
                                keep_fraction=keep_fraction,
                            )
                            if runtime.transport in (
                                FIELD_NATIVE_TARGETSTATE_RAW_DELTA_REPLACEMENT,
                                FIELD_NATIVE_TARGETSTATE_SPARSE25_RAW_DELTA_REPLACEMENT,
                            ):
                                if (
                                    native_action_path_velocity is None
                                    or native_noop_path_velocity is None
                                ):
                                    raise AnchorSGAANCError(
                                        "target-state raw replacement lacks native timing"
                                    )
                                action_route, phase_envelope = (
                                    _apply_native_phase_envelope(
                                        action_route,
                                        native_action_path_velocity,
                                        native_noop_path_velocity,
                                    )
                                )
                                if candidate_index == 0:
                                    trace["anchor_native_phase_envelopes"].append(
                                        {
                                            "step_index": step_index,
                                            "values": phase_envelope,
                                        }
                                    )
                            target_velocity = source_velocity + action_route
                        else:
                            action_route = _sparse_packed_action_contrast(
                                target_state_anchor_velocity,
                                target_state_anchor_noop_velocity,
                                strength=runtime.transport_strength,
                            )
                            if (
                                runtime.transport
                                == FIELD_NATIVE_GATED_TARGET_CONTRAST_VELOCITY
                            ):
                                if (
                                    native_action_path_velocity is None
                                    or native_noop_path_velocity is None
                                ):
                                    raise AnchorSGAANCError(
                                        "native trajectory envelope velocity is absent"
                                    )
                                action_route, phase_envelope = _apply_native_phase_envelope(
                                    action_route,
                                    native_action_path_velocity,
                                    native_noop_path_velocity,
                                )
                                if candidate_index == 0:
                                    trace["anchor_native_phase_envelopes"].append(
                                        {
                                            "step_index": step_index,
                                            "values": phase_envelope,
                                        }
                                    )
                            target_velocity = target_velocity + action_route
                    cache_bank.assert_empty()
                    delta = target_velocity - source_velocity
                    if delta.dtype != torch.float32 or not bool(torch.isfinite(delta).all()):
                        raise AnchorSGAANCError("anchor-conditioned field delta is invalid")
                    deltas.append(delta)

                if (
                    anchor_active
                    and runtime.anchor_state_mode == "native_t2v_trajectory"
                ):
                    if (
                        native_action_state is None
                        or native_noop_state is None
                        or native_anchor_action_scheduler is None
                        or native_anchor_noop_scheduler is None
                        or native_anchor_action_packed is None
                        or native_anchor_noop_packed is None
                    ):
                        raise AnchorSGAANCError(
                            "native T2V trajectory runtime is incomplete"
                        )
                    if (
                        native_action_path_velocity is None
                        or native_noop_path_velocity is None
                    ):
                        raise AnchorSGAANCError(
                            "native T2V trajectory velocity was not evaluated"
                        )
                    native_anchor_action_packed = _native_unipc_step(
                        native_anchor_action_scheduler,
                        velocity_packed=native_action_path_velocity,
                        timestep=timestep,
                        state_packed=native_anchor_action_packed,
                    )
                    native_anchor_noop_packed = _native_unipc_step(
                        native_anchor_noop_scheduler,
                        velocity_packed=native_noop_path_velocity,
                        timestep=timestep,
                        state_packed=native_anchor_noop_packed,
                    )
                    trace["anchor_native_trajectory_unipc_steps"] += 1

                noise_bank = torch.stack(noises)
                delta_bank = torch.stack(deltas)
                if count == 1:
                    aggregate = delta_bank[0]
                    scores: tuple[float, ...] = ()
                    weights: tuple[float, ...] = (1.0,)
                    background_scores: tuple[float, ...] = ()
                    anchor_action_scores: tuple[float, ...] = ()
                    combined_scores: tuple[float, ...] = ()
                else:
                    if runtime.aggregation == "uniform":
                        aggregate, weight_tensor, score_tensor = guided._aggregate_candidates(
                            source=source_packed,
                            edit=edit_packed,
                            candidate_deltas=delta_bank,
                            sigma=float(sigma_value),
                            mode=runtime.aggregation,
                            temperature=runtime.sga_temperature,
                        )
                        background_score_tensor = torch.empty(
                            0, device=score_tensor.device, dtype=score_tensor.dtype
                        )
                        anchor_action_score_tensor = torch.empty(
                            0, device=score_tensor.device, dtype=score_tensor.dtype
                        )
                    elif runtime.sga_score_mode == "background_source_cosine":
                        if sga_source_support is None:
                            raise AnchorSGAANCError(
                                "background SGA support is unexpectedly absent"
                            )
                        aggregate, weight_tensor, score_tensor = (
                            _aggregate_candidates_background_source_cosine(
                                source_packed=source_packed,
                                source_clean=source_clean,
                                edit_packed=edit_packed,
                                candidate_deltas=delta_bank,
                                sigma=float(sigma_value),
                                temperature=runtime.sga_temperature,
                                layout=layout,
                                source_support=sga_source_support,
                                residual_fraction=runtime.preservation_residual_fraction,
                            )
                        )
                        background_score_tensor = score_tensor
                        anchor_action_score_tensor = torch.empty(
                            0, device=score_tensor.device, dtype=score_tensor.dtype
                        )
                    elif runtime.sga_score_mode in (
                        "background_plus_anchor_action_002",
                        "background_trust_anchor_action_003",
                    ):
                        if sga_source_support is None:
                            raise AnchorSGAANCError(
                                "anchor-action SGA support is unexpectedly absent"
                            )
                        (
                            aggregate,
                            weight_tensor,
                            score_tensor,
                            background_score_tensor,
                            anchor_action_score_tensor,
                        ) = _aggregate_candidates_anchor_action_reward(
                            source_packed=source_packed,
                            source_clean=source_clean,
                            edit_packed=edit_packed,
                            candidate_deltas=delta_bank,
                            sigma=float(sigma_value),
                            temperature=runtime.sga_temperature,
                            layout=layout,
                            source_support=sga_source_support,
                            residual_fraction=runtime.preservation_residual_fraction,
                            anchor_clean=anchor_clean_bank[0],
                            anchor_static=anchor_static_clean_bank[0],
                            mode=runtime.sga_score_mode,
                        )
                    elif runtime.sga_score_mode in (
                        "background_plus_anchor_envelope_005",
                        "background_trust_anchor_envelope_003",
                    ):
                        if sga_source_support is None:
                            raise AnchorSGAANCError(
                                "anchor-envelope SGA support is unexpectedly absent"
                            )
                        (
                            aggregate,
                            weight_tensor,
                            score_tensor,
                            background_score_tensor,
                            anchor_action_score_tensor,
                        ) = _aggregate_candidates_anchor_envelope_reward(
                            source_packed=source_packed,
                            source_clean=source_clean,
                            edit_packed=edit_packed,
                            candidate_deltas=delta_bank,
                            sigma=float(sigma_value),
                            temperature=runtime.sga_temperature,
                            layout=layout,
                            source_support=sga_source_support,
                            residual_fraction=runtime.preservation_residual_fraction,
                            anchor_clean=anchor_clean_bank[0],
                            anchor_static=anchor_static_clean_bank[0],
                            mode=runtime.sga_score_mode,
                        )
                    else:
                        aggregate, weight_tensor, score_tensor = guided._aggregate_candidates(
                            source=source_packed,
                            edit=edit_packed,
                            candidate_deltas=delta_bank,
                            sigma=float(sigma_value),
                            mode=runtime.aggregation,
                            temperature=runtime.sga_temperature,
                        )
                        background_score_tensor = torch.empty(
                            0, device=score_tensor.device, dtype=score_tensor.dtype
                        )
                        anchor_action_score_tensor = torch.empty(
                            0, device=score_tensor.device, dtype=score_tensor.dtype
                        )
                    if (
                        runtime.initial_noise_proposal_mode
                        == "anchor_candidate0_forced"
                    ):
                        aggregate = delta_bank[0]
                        weight_tensor = torch.zeros_like(weight_tensor)
                        weight_tensor[0] = 1.0
                    scores = tuple(float(item) for item in score_tensor[:, 0].cpu())
                    weights = tuple(float(item) for item in weight_tensor[:, 0].cpu())
                    background_scores = tuple(
                        float(item) for item in background_score_tensor[:, 0].cpu()
                    ) if background_score_tensor.numel() else ()
                    anchor_action_scores = tuple(
                        float(item) for item in anchor_action_score_tensor[:, 0].cpu()
                    ) if anchor_action_score_tensor.numel() else ()
                    combined_scores = scores if anchor_action_scores else ()
                trace["candidate_counts"].append(count)
                trace["sga_scores"].append(scores)
                trace["sga_weights"].append(weights)
                trace["sga_background_scores"].append(background_scores)
                trace["sga_anchor_action_scores"].append(anchor_action_scores)
                trace["sga_combined_scores"].append(combined_scores)
                if count > 1 and step_index == EARLY_CANDIDATE_STEPS - 1:
                    if runtime.transport in ROLEWARP_REPLACEMENT_TRANSPORTS:
                        selected_role_proposal_index = int(
                            weight_tensor[:, 0].argmax().detach().cpu().item()
                        )
                        trace["rolewarp_selected_proposal_index"] = (
                            selected_role_proposal_index
                        )
                    elif (
                        runtime.transport
                        in qk_transport.EVENT01_ROLE_GRAPH_TRANSPORTS
                    ):
                        if runtime.event01_forced_role_proposal_index < 0:
                            selected_role_proposal_index = int(
                                weight_tensor[:, 0].argmax().detach().cpu().item()
                            )
                        trace["event01_role_graph_selected_proposal_index"] = (
                            selected_role_proposal_index
                        )
                    previous_noises = [
                        source_aligned.collapse_sga_noise_chains(
                            noise_bank, weight_tensor
                        )
                    ]
                    if runtime.anchor_candidate_mode == "bank_per_candidate":
                        anchor_weights = weight_tensor[:, 0].to(
                            dtype=anchor_packed_bank.dtype,
                            device=anchor_packed_bank.device,
                        )
                        broadcast = anchor_weights.reshape(
                            anchor_bank_size,
                            *([1] * (anchor_packed_bank.ndim - 1)),
                        )
                        collapsed_anchor_packed = (
                            broadcast * anchor_packed_bank
                        ).sum(dim=0)
                        collapsed_anchor_static_packed = (
                            broadcast * anchor_static_packed_bank
                        ).sum(dim=0)
                        trace["anchor_bank_collapse_weights"] = [
                            float(item) for item in anchor_weights.detach().cpu()
                        ]
                else:
                    previous_noises = list(noise_bank.unbind(0))
                edit_packed = edit_packed + float(next_sigma_value - sigma_value) * aggregate
                if (
                    preservation_support is not None
                    and step_index >= runtime.preservation_start_step
                ):
                    ramp_index = step_index - runtime.preservation_start_step + 1
                    ramp_progress = min(
                        1.0,
                        float(ramp_index) / float(runtime.preservation_ramp_steps),
                    )
                    effective_outside_scale = 1.0 - ramp_progress * (
                        1.0 - float(runtime.preservation_outside_scale)
                    )
                    active_support = preservation_support
                    active_residual_fraction = runtime.preservation_residual_fraction
                    if (
                        runtime.preservation_mode
                        == "source_motion_support_snapshot_residual"
                    ):
                        if snapshot_residual_support is None:
                            snapshot_residual_support = _effective_source_edit_support(
                                cdf._unpack_spatial_latent(edit_packed, layout),
                                source_clean,
                                preservation_support,
                                residual_fraction=runtime.preservation_residual_fraction,
                            ).detach()
                            trace["preservation_snapshot_residual_support_step"] = (
                                step_index
                            )
                            trace[
                                "preservation_snapshot_residual_support_fraction"
                            ] = float(
                                snapshot_residual_support.float()
                                .mean()
                                .detach()
                                .cpu()
                                .item()
                            )
                        active_support = snapshot_residual_support
                        active_residual_fraction = 0.0
                    edit_packed = _apply_source_motion_preservation(
                        edit_packed,
                        source_clean,
                        active_support,
                        layout=layout,
                        outside_scale=effective_outside_scale,
                        residual_fraction=active_residual_fraction,
                    )
                    if runtime.preservation_object_identity_strength > 0.0:
                        edit_packed = (
                            _apply_event01_object1_sparse_signature_projection(
                                edit_packed,
                                source_clean,
                                layout=layout,
                                strength=runtime.preservation_object_identity_strength,
                            )
                            if runtime.preservation_mode
                            == "source_motion_support_event01_actor_object"
                            else _apply_event01_object1_identity_projection(
                                edit_packed,
                                source_clean,
                                layout=layout,
                                strength=runtime.preservation_object_identity_strength,
                            )
                        )
                    trace["preservation_applied_schedule"].append(
                        {
                            "step_index": step_index,
                            "ramp_progress": ramp_progress,
                            "effective_outside_scale": effective_outside_scale,
                        }
                    )
                elif runtime.initial_phase_clamp:
                    _clamp_initial_latent_phase(edit_packed, source_packed, layout)
                if not bool(torch.isfinite(edit_packed).all()):
                    raise AnchorSGAANCError("clean edit state left the finite domain")

    def candidate_count_for_step(index: int) -> int:
        return early_candidate_count if index < EARLY_CANDIDATE_STEPS else 1

    expected_candidates = sum(
        candidate_count_for_step(index) for index in range(runtime.num_inference_steps)
    )
    expected_anchor_cells = sum(
        candidate_count_for_step(index) for index in range(runtime.transport_steps)
    )
    expected_anchor_candidates = expected_anchor_cells
    if runtime.transport in (
        FIELD_CONTRAST_VELOCITY,
        qk_transport.TEMPORAL_CONTRAST_QK,
        qk_transport.TEMPORAL_CONTRAST_ATTN_OUTPUT,
        qk_transport.TEMPORAL_CORRESPONDENCE_CONTRAST_QK,
        qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK,
        qk_transport.TEMPORAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
        qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_ATTN_OUTPUT,
        qk_transport.TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_QK,
        qk_transport.TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
        qk_transport.HARD_PHASE_MEAN_CONTRAST_QK,
        qk_transport.HARD_PHASE_MEAN_CONTRAST_ATTN_OUTPUT,
        qk_transport.HARD_PREROPE_PHASE_MEAN_CONTRAST_QK,
        qk_transport.TEMPORAL_KERNEL_CONTRAST_ATTN_OUTPUT,
        *qk_transport.TARGET_OWNED_QK_TRANSPORTS_V14R2,
        qk_transport.ACTION_NOOP_OBSERVER_ATTN_OUTPUT,
        *qk_transport.TARGET_GATED_HARD_KERNEL_TRANSPORTS,
        *qk_transport.EVENT01_ROLE_GRAPH_TRANSPORTS,
        cross_transport.TEMPORAL_CONTRAST_CROSS_ATTN_OUTPUT,
    ):
        expected_anchor_candidates *= 2
    if (
        runtime.anchor_state_mode == "native_t2v_trajectory"
        and runtime.transport in (FIELD_VELOCITY_RESIDUAL, FIELD_CONTRAST_VELOCITY)
    ):
        # The four exact APG action/no-op forwards are counted once per solver
        # step in ``anchor_native_trajectory_model_forwards`` and their dense
        # velocity contrast is shared by all SGA candidates.  No proxy anchor
        # endpoint forward is made per candidate in this mode.
        expected_anchor_candidates = 0
    elif runtime.transport in TARGET_STATE_FIELD_TRANSPORTS:
        # Action and no-op are each native APG queries (negative + conditional)
        # on every live target candidate: four raw model forwards per cell.
        expected_anchor_candidates = 4 * expected_anchor_cells
    elif runtime.transport in ROLEWARP_REPLACEMENT_TRANSPORTS:
        # Role-warp consumes the once-per-step exact native action/no-op paths;
        # it deliberately performs no proxy anchor forward per SGA candidate.
        expected_anchor_candidates = 0
    forwards_per_field = (
        2 if runtime.field_guidance in ("apg", "raw_cfg") else 1
    )
    expected_native_trajectory_forwards = (
        4 * runtime.transport_steps
        if runtime.anchor_state_mode == "native_t2v_trajectory"
        else 0
    )
    expected_native_trajectory_steps = (
        runtime.transport_steps
        if runtime.anchor_state_mode == "native_t2v_trajectory"
        else 0
    )
    if (
        runtime.anchor_state_mode == "native_t2v_trajectory"
        and runtime.transport_steps == runtime.num_inference_steps
    ):
        if native_anchor_action_packed is None:
            raise AnchorSGAANCError("native action terminal state is absent")
        terminal_error = native_anchor_action_packed.float() - anchor_packed_bank[0].float()
        trace["anchor_native_action_terminal_mse_to_saved_clean"] = float(
            terminal_error.square().mean().detach().cpu().item()
        )
        trace["anchor_native_action_terminal_max_abs_to_saved_clean"] = float(
            terminal_error.abs().amax().detach().cpu().item()
        )
    if (
        trace["anchor_model_forwards"] != expected_anchor_candidates
        or trace["anchor_native_trajectory_model_forwards"]
        != expected_native_trajectory_forwards
        or trace["anchor_native_trajectory_unipc_steps"]
        != expected_native_trajectory_steps
        or len(trace["anchor_active_schedule"]) != runtime.transport_steps
        or trace["anchor_candidate_cells"] != expected_anchor_cells
        or trace["target_model_forwards"] != forwards_per_field * expected_candidates
        or trace["source_model_forwards"] != forwards_per_field * expected_candidates
        or trace["target_apg_forwards"]
        != (2 * expected_candidates if runtime.field_guidance == "apg" else 0)
        or trace["source_apg_forwards"]
        != (2 * expected_candidates if runtime.field_guidance == "apg" else 0)
        or trace["target_raw_cfg_forwards"]
        != (2 * expected_candidates if runtime.field_guidance == "raw_cfg" else 0)
        or trace["source_raw_cfg_forwards"]
        != (2 * expected_candidates if runtime.field_guidance == "raw_cfg" else 0)
    ):
        raise AnchorSGAANCError("online anchor/target/source forward closure differs")
    if runtime.transport in NATIVE_T2V_REPLACEMENT_TRANSPORTS and (
        trace["native_path_velocity_bound_to_candidate0_schedule"]
        != list(range(runtime.transport_steps))
        or trace[
            "native_target_replacement_step0_candidate0_action_mse_to_anchor_path"
        ]
        != 0.0
        or trace[
            "native_target_replacement_step0_candidate0_noop_mse_to_anchor_path"
        ]
        != 0.0
    ):
        raise AnchorSGAANCError(
            "native generation path was not exactly bound to candidate 0"
        )
    if runtime.transport in qk_transport.EVENT01_ROLE_GRAPH_TRANSPORTS:
        role_schedule = trace["event01_role_graph_route_schedule"]
        early_proposals = [
            item["proposal_index"]
            for item in role_schedule
            if item["step_index"] < EARLY_CANDIDATE_STEPS
        ]
        if (
            len(role_schedule) != expected_anchor_cells
            or early_proposals != _expected_event01_early_role_proposals(runtime)
            or trace["event01_role_graph_selected_proposal_index"] is None
        ):
            raise AnchorSGAANCError("Event01 role-graph proposal closure differs")
    trace["attention_cache"] = cache_bank.receipt()
    _validate_target_owned_qk_route_closure(
        transport=runtime.transport,
        transport_steps=runtime.transport_steps,
        expected_anchor_cells=expected_anchor_cells,
        selected_block_count=len(runtime.selected_block_indices),
        field_guidance=runtime.field_guidance,
        anchor_cfg_scope=runtime.anchor_cfg_scope,
        trace=trace,
        cache_receipt=trace["attention_cache"],
    )
    result = cdf._unpack_spatial_latent(edit_packed, layout)
    return (result, trace) if return_trace else result


__all__ = [
    "ARMS",
    "ANCHOR_CFG_SCOPES",
    "ANCHOR_CONTRAST_MODES",
    "ANCHOR_SIGMA_CAPS",
    "PRESERVATION_MODES",
    "SGA_SCORE_MODES",
    "ANCHOR_CANDIDATE_MODES",
    "ANCHOR_SPATIAL_ALIGNMENTS",
    "ANCHOR_STATE_MODES",
    "PRESERVATION_KEEP_FRACTIONS",
    "PRESERVATION_OUTSIDE_SCALES",
    "PRESERVATION_DILATIONS",
    "PRESERVATION_RESIDUAL_FRACTIONS",
    "PRESERVATION_OBJECT_IDENTITY_STRENGTHS",
    "FIELD_GUIDANCES",
    "FIELD_MODELS",
    "FIELD_VELOCITY_RESIDUAL",
    "FIELD_CONTRAST_VELOCITY",
    "FIELD_TARGET_CONTRAST_VELOCITY",
    "FIELD_NATIVE_GATED_TARGET_CONTRAST_VELOCITY",
    "FIELD_NATIVE_T2V_TARGET_VELOCITY_REPLACEMENT",
    "FIELD_NATIVE_T2V_DELTA_VELOCITY_REPLACEMENT",
    "FIELD_NATIVE_T2V_TEMPORAL_DELTA_REPLACEMENT",
    "FIELD_NATIVE_T2V_SPARSE25_TEMPORAL_DELTA_REPLACEMENT",
    "FIELD_NATIVE_TARGETSTATE_TEMPORAL_DELTA_REPLACEMENT",
    "FIELD_NATIVE_TARGETSTATE_SPARSE25_TEMPORAL_DELTA_REPLACEMENT",
    "FIELD_TARGETSTATE_RAW_DELTA_REPLACEMENT",
    "FIELD_TARGETSTATE_SPARSE25_RAW_DELTA_REPLACEMENT",
    "FIELD_NATIVE_TARGETSTATE_RAW_DELTA_REPLACEMENT",
    "FIELD_NATIVE_TARGETSTATE_SPARSE25_RAW_DELTA_REPLACEMENT",
    "FIELD_NATIVE_ROLEWARP_TEMPORAL_DELTA_REPLACEMENT",
    "FIELD_NATIVE_ROLEWARP_SPARSE25_TEMPORAL_DELTA_REPLACEMENT",
    "NATIVE_T2V_REPLACEMENT_TRANSPORTS",
    "TARGETSTATE_TEMPORAL_REPLACEMENT_TRANSPORTS",
    "TARGETSTATE_RAW_REPLACEMENT_TRANSPORTS",
    "TARGETSTATE_HARD_REPLACEMENT_TRANSPORTS",
    "ROLEWARP_REPLACEMENT_TRANSPORTS",
    "TARGET_STATE_FIELD_TRANSPORTS",
    "DYNAEDIT_SGA_TEMPERATURE",
    "TRANSPORTS",
    "AnchorSGAANCConfig",
    "AnchorSGAANCError",
    "_clamp_initial_latent_phase",
    "_source_motion_support",
    "_effective_source_edit_support",
    "_apply_source_motion_preservation",
    "_event01_object1_phasewise_preservation_support",
    "_event01_actor_object_phasewise_preservation_support",
    "_expected_event01_early_role_proposals",
    "_event01_object1_phasewise_source_reference",
    "_apply_event01_object1_identity_projection",
    "_apply_event01_object1_sparse_signature_projection",
    "_aggregate_candidates_background_source_cosine",
    "_action_temporal_signature",
    "_aggregate_candidates_anchor_action_reward",
    "_action_motion_envelope_signature",
    "_aggregate_candidates_anchor_envelope_reward",
    "_sparse_packed_temporal_residual",
    "_sparse_packed_raw_action_contrast",
    "_event01_role_warp_native_route",
    "_sparse_packed_action_contrast",
    "_apply_native_t2v_hard_replacement",
    "_apply_native_phase_envelope",
    "_align_packed_route_to_source_motion",
    "sample_anchor_sga_anc",
]
