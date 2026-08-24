#!/usr/bin/env python3
"""Guided source-aligned FlowEdit controls for frozen Bernini-R 1.3B.

This is the scientifically auditable successor to ``source_aligned_controller``.
The older module remains a raw-conditional diagnostic.  Every field queried by
this module is reconstructed with the pinned Bernini ``v2v_apg`` numerical
program before the source/target velocity difference is formed.

The public inference condition remains exactly a clean 81-frame source video
and an edit instruction.  The semantic no-op and Bernini negative prompt are
fixed internal controls.  No target video, mask, flow, pose, track, trajectory,
swept tube, or first-frame oracle is accepted.

For every FlowEdit candidate the transformer call order is fixed to

``target negative -> target action -> source negative -> source no-op``.

Thus a one-candidate 40-step arm executes 160 transformer forwards and a
three-step five-candidate arm executes 208.  APG is deliberately restricted to
the pinned momentum-zero setting.  A nonzero momentum would require separate
state for every candidate and an otherwise undefined K-to-one state collapse.

The fresh-noise generator is counter based: each draw is keyed by
``(master_seed, step, candidate)``.  Consequently the IID and ANC arms share
candidate zero exactly, while uniform and SGA arms share all five early
candidates.  When five chains become one, both aggregation arms continue
candidate zero.  There is no weight-dependent noise-chain collapse.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any, Optional

import differential_sampler as cdf
import source_aligned_controller as raw_controller
import tri_branch_unipc as tri


EXPECTED_RGB_FRAMES = 81
EXPECTED_LATENT_PHASES = 21
EXPECTED_STEPS = 40
EXPECTED_SEED = 2027
EXPECTED_FLOW_SHIFT = 5.0
EARLY_CANDIDATES = 5
EARLY_CANDIDATE_STEPS = 3
ANC_LOCK_SIGMA = 0.25
# Historical guided-controller matched-control temperature.  DynaEdit-style
# callers pass their own lower temperature explicitly rather than changing the
# semantics of this frozen comparison protocol.
SGA_TEMPERATURE = 1.0

# Measured from the pinned Diffusers UniPC flow schedule on AUH.  It is one
# float32 ulp-range beyond the old 1e-6 check and must not be rounded to 1.0 in
# receipts or in the first ANC coefficient.
PINNED_UNIPC_START_SIGMA = 0.9999989867210388
PINNED_UNIPC_START_SIGMA_ATOL = 2.0e-7
PINNED_UNIPC_SCHEDULE_DIGEST = (
    "43cd53329945280dccea5c1a1aa3b5da05337a7f10cfec0ab5a727592ea77d25"
)
PINNED_UNIPC_SIGMA_FP32_DIGEST = (
    "b0d8af444d64bd51d638973de86578ad6685681521acad453c7b422f2c628dcd"
)

APG_GUIDANCE_MODE = "v2v_apg"
APG_GUIDANCE_SCALE = 4.0
APG_ETA = 0.5
APG_NORM_THRESHOLD = 50.0
APG_MOMENTUM = 0.0

BRANCH_ORDER = (
    "target_negative",
    "target_action",
    "source_negative",
    "source_noop",
)
GUIDED_ARMS = ("FIID1G", "FANC1G", "FAVG5G", "FSGA5G")


class GuidedSourceAlignedControllerError(RuntimeError):
    """Raised whenever the guided V2 scientific contract cannot be proven."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise GuidedSourceAlignedControllerError(
            f"value is not canonical JSON: {error}"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class GuidedSourceAlignedConfig:
    """One of four matched guided FlowEdit mechanism arms."""

    arm: str
    num_inference_steps: int = EXPECTED_STEPS
    flow_shift: float = EXPECTED_FLOW_SHIFT
    seed: int = EXPECTED_SEED
    motion_scale: float = 1.0
    sga_temperature: float = SGA_TEMPERATURE
    anc_lock_sigma: float = ANC_LOCK_SIGMA
    apg_guidance_scale: float = APG_GUIDANCE_SCALE
    apg_eta: float = APG_ETA
    apg_norm_threshold: float = APG_NORM_THRESHOLD
    apg_momentum: float = APG_MOMENTUM

    def validate(self) -> "GuidedSourceAlignedConfig":
        if self.arm not in GUIDED_ARMS:
            raise GuidedSourceAlignedControllerError(
                f"guided arm must be one of {GUIDED_ARMS}, got {self.arm!r}"
            )
        exact = {
            "num_inference_steps": (self.num_inference_steps, EXPECTED_STEPS),
            "seed": (self.seed, EXPECTED_SEED),
        }
        for label, (actual, expected) in exact.items():
            if type(actual) is not int or actual != expected:
                raise GuidedSourceAlignedControllerError(
                    f"{label} must equal the pinned value {expected}"
                )
        scalar_exact = {
            "flow_shift": (self.flow_shift, EXPECTED_FLOW_SHIFT),
            "motion_scale": (self.motion_scale, 1.0),
            "sga_temperature": (self.sga_temperature, SGA_TEMPERATURE),
            "anc_lock_sigma": (self.anc_lock_sigma, ANC_LOCK_SIGMA),
            "apg_guidance_scale": (
                self.apg_guidance_scale,
                APG_GUIDANCE_SCALE,
            ),
            "apg_eta": (self.apg_eta, APG_ETA),
            "apg_norm_threshold": (
                self.apg_norm_threshold,
                APG_NORM_THRESHOLD,
            ),
            "apg_momentum": (self.apg_momentum, APG_MOMENTUM),
        }
        for label, (actual, expected) in scalar_exact.items():
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isfinite(float(actual))
                or not math.isclose(
                    float(actual), float(expected), rel_tol=0.0, abs_tol=0.0
                )
            ):
                raise GuidedSourceAlignedControllerError(
                    f"{label} must equal the pinned value {expected}"
                )
        return self

    @property
    def uses_anc(self) -> bool:
        return self.arm != "FIID1G"

    @property
    def aggregation(self) -> str:
        if self.arm == "FAVG5G":
            return "uniform"
        if self.arm == "FSGA5G":
            return "sga_cosine"
        return "single"

    def candidate_count(self, step: int) -> int:
        if self.arm in ("FAVG5G", "FSGA5G") and step < EARLY_CANDIDATE_STEPS:
            return EARLY_CANDIDATES
        return 1

    @property
    def expected_candidate_evaluations(self) -> int:
        return sum(self.candidate_count(step) for step in range(EXPECTED_STEPS))

    @property
    def expected_shared_step_calls(self) -> int:
        return 4 * self.expected_candidate_evaluations


@dataclass(frozen=True)
class GuidedAPGResult:
    """One condition's pinned APG velocity plus dtype/parity evidence."""

    velocity_packed_fp32: Any
    raw_negative_dtype: str
    raw_condition_dtype: str
    negative_clean_dtype: str
    condition_clean_dtype: str
    guided_clean_dtype: str
    guided_velocity_fp32_dtype: str
    paired_query_object: bool


@dataclass(frozen=True)
class GuidedSourceAlignedTrace:
    """Tensor-free evidence emitted identically by every Ulysses rank."""

    arm: str
    sigmas: tuple[float, ...]
    timesteps: tuple[float, ...]
    schedule_digest: str
    scheduler_sigma_fp32_digest: str
    scheduler_sigma_dtype: str
    scheduler_sigma_device: str
    scheduler_sigma_direct_views: bool
    candidate_counts: tuple[int, ...]
    anc_retained_variance: tuple[float, ...]
    anc_nominal_correlation: tuple[float, ...]
    sga_scores: tuple[tuple[float, ...], ...]
    sga_weights: tuple[tuple[float, ...], ...]
    sga_entropy: tuple[float, ...]
    sga_top1_margin: tuple[float, ...]
    delta_rms: tuple[float, ...]
    update_rms: tuple[float, ...]
    noise_state_change_rms: tuple[float, ...]
    fresh_noise_draws: int
    used_noise_key_digest: str
    used_fresh_noise_content_digest: str
    candidate0_fresh_noise_content_digest: str
    full_noise_bank_digest: str
    candidate0_noise_bank_digest: str
    branch_order: tuple[str, ...]
    branch_counts: tuple[int, ...]
    total_shared_step_calls: int
    apg_parameters: tuple[tuple[str, Any], ...]
    target_branch_query_parity: bool
    source_branch_query_parity: bool
    raw_velocity_dtype: str
    guided_velocity_dtype: str
    apg_clean_dtype: str
    delta_dtype: str
    edit_state_dtype: str
    candidate_continuation: str
    weighted_noise_collapse_used: bool
    anc_initial_predecessor_policy: str


def guided_controller_contract() -> dict[str, Any]:
    """Return the conservative guided V2 inference contract."""

    return {
        "method": "bernini_guided_source_aligned_flowedit_v2",
        "status": "guided_bernini_adaptation_not_official_dynaedit_reproduction",
        "raw_v1_status": "historical_diagnostic_not_formal_baseline",
        "clip_geometry": {
            "rgb_frames": EXPECTED_RGB_FRAMES,
            "wan_vae_phases": EXPECTED_LATENT_PHASES,
        },
        "user_inputs": ["source_video", "edit_instruction"],
        "internal_fixed_conditions": [
            "clean_source_vae_latent",
            "semantic_noop_instruction",
            "verbatim_bernini_negative_prompt",
        ],
        "forbidden_conditions": [
            "target_video",
            "mask",
            "track",
            "swept_tube",
            "pose",
            "trajectory",
            "optical_flow",
            "first_frame_anchor",
        ],
        "guided_flowedit_field": (
            "APG(action,z_tar|source)-APG(noop,z_src|source)"
        ),
        "per_candidate_branch_order": list(BRANCH_ORDER),
        "per_candidate_transformer_forwards": 4,
        "apg": {
            "guidance_mode": APG_GUIDANCE_MODE,
            "guidance_scale": APG_GUIDANCE_SCALE,
            "eta": APG_ETA,
            "norm_threshold": APG_NORM_THRESHOLD,
            "momentum": APG_MOMENTUM,
            "momentum_nonzero_supported": False,
            "numerical_implementation": "tri_branch_unipc_helpers",
            "dtype_order": (
                "CPU_fp32_sigma_times_GPU_bf16_velocity_then_fp32_subtract"
            ),
        },
        "noise_bank": noise_bank_pairing_contract(
            seed=EXPECTED_SEED,
            steps=EXPECTED_STEPS,
            candidates=EARLY_CANDIDATES,
        ),
        "runtime_noise_evidence": [
            "used_fresh_noise_content_digest",
            "candidate0_fresh_noise_content_digest",
        ],
        "aggregation": {
            "early_steps": EARLY_CANDIDATE_STEPS,
            "early_candidates": EARLY_CANDIDATES,
            "sga_temperature": SGA_TEMPERATURE,
            "k_to_one": "continue_candidate_0_for_uniform_and_sga",
            "weighted_noise_collapse": False,
        },
        "pinned_unipc_start_sigma": PINNED_UNIPC_START_SIGMA,
        "pinned_unipc_schedule_digest": PINNED_UNIPC_SCHEDULE_DIGEST,
        "pinned_unipc_sigma_fp32_digest": PINNED_UNIPC_SIGMA_FP32_DIGEST,
        "anc_initial_predecessor_policy": "zero_initialized_per_dynaedit_pseudocode",
        "sequence_parallel_owner": "official_bernini_transformer",
    }


def keyed_noise_seed(master_seed: int, step: int, candidate: int) -> int:
    """Derive an arm-independent CPU generator seed for one noise cell."""

    for label, value in (
        ("master_seed", master_seed),
        ("step", step),
        ("candidate", candidate),
    ):
        if type(value) is not int or value < 0:
            raise GuidedSourceAlignedControllerError(
                f"{label} must be a non-negative integer"
            )
    payload = (
        f"bernini-guided-sac-v2\0{master_seed}\0{step}\0{candidate}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def noise_bank_pairing_contract(
    *, seed: int, steps: int = EXPECTED_STEPS, candidates: int = EARLY_CANDIDATES
) -> dict[str, Any]:
    """Describe the full keyed bank and its shared candidate-zero sub-bank."""

    if type(steps) is not int or steps <= 0:
        raise GuidedSourceAlignedControllerError("noise-bank steps must be positive")
    if type(candidates) is not int or candidates <= 0:
        raise GuidedSourceAlignedControllerError(
            "noise-bank candidates must be positive"
        )
    rows = [
        {
            "step": step,
            "candidate": candidate,
            "derived_seed": keyed_noise_seed(seed, step, candidate),
        }
        for step in range(steps)
        for candidate in range(candidates)
    ]
    candidate_zero = [row for row in rows if row["candidate"] == 0]
    return {
        "scheme": "sha256_keyed_cpu_torch_generator_v1",
        "master_seed": seed,
        "steps": steps,
        "candidates": candidates,
        "full_bank_digest": _object_sha256(rows),
        "candidate0_bank_digest": _object_sha256(candidate_zero),
        "iid_anc_candidate0_exact_pairing": True,
        "avg_sga_full_early_bank_exact_pairing": True,
    }


def used_noise_key_digest(config: GuidedSourceAlignedConfig) -> str:
    """Hash the exact keyed cells consumed by one active arm."""

    runtime = config.validate()
    rows = [
        {
            "step": step,
            "candidate": candidate,
            "derived_seed": keyed_noise_seed(runtime.seed, step, candidate),
        }
        for step in range(runtime.num_inference_steps)
        for candidate in range(runtime.candidate_count(step))
    ]
    return _object_sha256(rows)


def validate_pinned_sigma_intervals(intervals: Any) -> tuple[tuple[float, float], ...]:
    """Accept the measured UniPC start without silently rounding it to one."""

    try:
        values = tuple((float(left), float(right)) for left, right in intervals)
    except (TypeError, ValueError) as error:
        raise GuidedSourceAlignedControllerError("sigma intervals are not numeric") from error
    if len(values) != EXPECTED_STEPS:
        raise GuidedSourceAlignedControllerError(
            f"guided controller requires {EXPECTED_STEPS} intervals"
        )
    first = values[0][0]
    if not math.isclose(
        first,
        PINNED_UNIPC_START_SIGMA,
        rel_tol=0.0,
        abs_tol=PINNED_UNIPC_START_SIGMA_ATOL,
    ):
        raise GuidedSourceAlignedControllerError(
            "UniPC start sigma differs from the measured pinned schedule: "
            f"{first!r} != {PINNED_UNIPC_START_SIGMA!r}"
        )
    if not math.isclose(values[-1][1], 0.0, rel_tol=0.0, abs_tol=1.0e-7):
        raise GuidedSourceAlignedControllerError("guided schedule must terminate at zero")
    if any(
        not math.isfinite(value)
        or value < -1.0e-7
        or value > 1.0 + 1.0e-7
        for pair in values
        for value in pair
    ):
        raise GuidedSourceAlignedControllerError("sigma schedule leaves finite [0,1]")
    if any(right > left + 1.0e-7 for left, right in values):
        raise GuidedSourceAlignedControllerError("sigma schedule is not descending")
    return values


def _require_torch() -> Any:
    try:
        import torch
    except Exception as error:  # pragma: no cover - exercised on AUH
        raise GuidedSourceAlignedControllerError(
            "PyTorch is required by the guided controller"
        ) from error
    return torch


def _draw_keyed_packed_noise(
    *, source_latent: Any, layout: cdf.LatentLayout, seed: int, step: int, candidate: int
) -> tuple[Any, str]:
    torch = _require_torch()
    derived = keyed_noise_seed(seed, step, candidate)
    generator = torch.Generator(device="cpu").manual_seed(derived)
    fresh_cpu = torch.randn(
        tuple(int(value) for value in source_latent.shape),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    content_digest = hashlib.sha256(
        fresh_cpu.contiguous().numpy().tobytes(order="C")
    ).hexdigest()
    fresh = fresh_cpu.to(device=source_latent.device)
    if fresh.dtype != torch.float32:
        raise GuidedSourceAlignedControllerError("fresh noise must remain fp32")
    return cdf._pack_spatial_latent(fresh, layout), content_digest


def _validate_sigma_cpu_fp32(sigma: Any) -> Any:
    torch = _require_torch()
    if (
        not isinstance(sigma, torch.Tensor)
        or sigma.ndim != 0
        or sigma.device.type != "cpu"
        or sigma.dtype != torch.float32
        or not bool(torch.isfinite(sigma))
        or not bool(sigma > 0)
    ):
        raise GuidedSourceAlignedControllerError(
            "APG sigma must be one positive CPU fp32 scalar"
        )
    return sigma


def capture_pinned_scheduler_sigma_scalars(
    diffusion: Any, intervals: Any
) -> tuple[tuple[Any, ...], str]:
    """Capture direct CPU-fp32 scalar views before any float reconstruction.

    The returned objects are the actual ``scheduler.sigmas[index]`` views.  The
    float interval copy remains useful for FlowEdit coefficients and receipts,
    but it is never sent through the APG numerical path.
    """

    torch = _require_torch()
    raw = getattr(getattr(diffusion, "scheduler", None), "sigmas", None)
    if (
        not isinstance(raw, torch.Tensor)
        or raw.ndim != 1
        or raw.device.type != "cpu"
        or raw.dtype != torch.float32
        or int(raw.numel()) != EXPECTED_STEPS + 1
        or not bool(torch.isfinite(raw).all())
    ):
        raise GuidedSourceAlignedControllerError(
            "pinned scheduler.sigmas must be a finite CPU fp32 vector"
        )
    if struct.pack(">f", float(raw[-1].item())) != b"\x00\x00\x00\x00":
        raise GuidedSourceAlignedControllerError(
            "pinned UniPC scheduler must expose a bit-exact +0 terminal sigma"
        )
    values = validate_pinned_sigma_intervals(intervals)
    scalars = tuple(raw[index] for index in range(EXPECTED_STEPS))
    for index, (scalar, pair) in enumerate(zip(scalars, values)):
        _validate_sigma_cpu_fp32(scalar)
        if float(scalar.item()) != pair[0]:
            raise GuidedSourceAlignedControllerError(
                f"scheduler sigma scalar/interval bit value differs at step {index}"
            )
        if scalar.untyped_storage().data_ptr() != raw.untyped_storage().data_ptr():
            raise GuidedSourceAlignedControllerError(
                "scheduler sigma capture reconstructed rather than viewed storage"
            )
    payload = b"".join(struct.pack(">f", float(value.item())) for value in scalars)
    return scalars, hashlib.sha256(payload).hexdigest()


def _guided_apg_velocity(
    *,
    diffusion: Any,
    transformer: Any,
    source_condition: tuple[Any, Any, Any],
    query_latent: Any,
    condition_prompt_embeds: Any,
    negative_prompt_embeds: Any,
    timestep: Any,
    sigma: Any,
    branch: str,
    negative_context: Any = None,
    condition_context: Any = None,
) -> GuidedAPGResult:
    """Evaluate one negative+condition pair with exact pinned APG arithmetic."""

    torch = _require_torch()
    if branch not in ("target_action", "source_noop"):
        raise GuidedSourceAlignedControllerError("unknown guided APG branch")
    if not isinstance(query_latent, torch.Tensor) or query_latent.dtype != torch.float32:
        raise GuidedSourceAlignedControllerError("APG query state must be GPU/CPU fp32")

    # Passing this exact object to both calls is the function-level state-pair
    # certificate.  The official patcher may create fresh token tensors, but
    # both originate from identical query bytes and the same source condition.
    negative_query = query_latent
    condition_query = query_latent
    active_negative_context = (
        contextlib.nullcontext()
        if negative_context is None
        else negative_context
    )
    with active_negative_context:
        raw_negative = cdf._predict_source_conditioned_velocity(
            diffusion=diffusion,
            transformer=transformer,
            source_condition=source_condition,
            query_latent=negative_query,
            prompt_embeds=negative_prompt_embeds,
            timestep=timestep,
        )
    active_context = (
        contextlib.nullcontext()
        if condition_context is None
        else condition_context
    )
    with active_context:
        raw_condition = cdf._predict_source_conditioned_velocity(
            diffusion=diffusion,
            transformer=transformer,
            source_condition=source_condition,
            query_latent=condition_query,
            prompt_embeds=condition_prompt_embeds,
            timestep=timestep,
        )
    if raw_negative.dtype != torch.bfloat16 or raw_condition.dtype != torch.bfloat16:
        raise GuidedSourceAlignedControllerError(
            "pinned Bernini raw APG velocities must both be bfloat16"
        )
    if tuple(raw_negative.shape) != tuple(raw_condition.shape):
        raise GuidedSourceAlignedControllerError("negative/condition velocity shapes differ")

    layout = cdf.validate_latent_shape(tuple(int(value) for value in query_latent.shape))
    negative_v = cdf._unpack_spatial_latent(raw_negative, layout)
    condition_v = cdf._unpack_spatial_latent(raw_condition, layout)
    # Do not rebuild this scalar from the float-valued FlowEdit schedule.  The
    # pinned Bernini APG path consumes scheduler.sigmas[index] as a CPU fp32
    # 0-d tensor, which has a distinct wrapped-scalar promotion path on ROCm.
    sigma_tensor = _validate_sigma_cpu_fp32(sigma)
    try:
        negative_clean = tri.pinned_raw_condition_clean(
            query_latent, negative_v, sigma_tensor
        )
        condition_clean = tri.pinned_raw_condition_clean(
            query_latent, condition_v, sigma_tensor
        )
        momentum = tri._MomentumBuffer(APG_MOMENTUM, branch=branch)
        guided_clean = tri._normalized_guidance(
            condition_clean,
            negative_clean,
            APG_GUIDANCE_SCALE,
            momentum,
            APG_ETA,
            APG_NORM_THRESHOLD,
        )
    except tri.TriBranchHookError as error:
        raise GuidedSourceAlignedControllerError(str(error)) from error
    if momentum.update_count != 1 or momentum.momentum != 0.0:
        raise GuidedSourceAlignedControllerError("APG must be stateless momentum-zero")
    for label, value in (
        ("negative clean", negative_clean),
        ("condition clean", condition_clean),
        ("guided clean", guided_clean),
    ):
        if value.dtype != torch.float32 or not bool(torch.isfinite(value).all().item()):
            raise GuidedSourceAlignedControllerError(f"{label} must be finite fp32")
    guided_velocity_fp32 = (query_latent - guided_clean) / sigma_tensor
    if guided_velocity_fp32.dtype != torch.float32:
        raise GuidedSourceAlignedControllerError("guided APG velocity must form in fp32")
    guided_velocity_packed = cdf._pack_spatial_latent(
        guided_velocity_fp32, layout
    )
    if guided_velocity_packed.dtype != torch.float32:
        raise GuidedSourceAlignedControllerError(
            "packed guided APG velocity must remain fp32"
        )
    if not bool(torch.isfinite(guided_velocity_packed).all().item()):
        raise GuidedSourceAlignedControllerError("guided APG velocity is non-finite")
    return GuidedAPGResult(
        velocity_packed_fp32=guided_velocity_packed,
        raw_negative_dtype=str(raw_negative.dtype),
        raw_condition_dtype=str(raw_condition.dtype),
        negative_clean_dtype=str(negative_clean.dtype),
        condition_clean_dtype=str(condition_clean.dtype),
        guided_clean_dtype=str(guided_clean.dtype),
        guided_velocity_fp32_dtype=str(guided_velocity_fp32.dtype),
        paired_query_object=negative_query is condition_query,
    )


def _aggregate_candidates(
    *,
    source: Any,
    edit: Any,
    candidate_deltas: Any,
    sigma: float,
    mode: str,
    temperature: float = SGA_TEMPERATURE,
) -> tuple[Any, Any, Any]:
    """Use identical projections/scores while changing only AVG/SGA weights."""

    torch = _require_torch()
    if mode not in ("uniform", "sga_cosine"):
        raise GuidedSourceAlignedControllerError("multi-candidate mode is invalid")
    _, sga_weights, scores, projected = raw_controller.similarity_guided_aggregate(
        source=source,
        edit=edit,
        candidate_deltas=candidate_deltas,
        sigma=sigma,
        temperature=temperature,
    )
    if mode == "uniform":
        weights = torch.full_like(sga_weights, 1.0 / int(sga_weights.shape[0]))
    else:
        weights = sga_weights
    broadcast = weights.reshape(
        int(weights.shape[0]),
        int(weights.shape[1]),
        *([1] * (source.ndim - 1)),
    )
    aggregate_projection = (
        broadcast.to(dtype=projected.dtype) * projected
    ).sum(dim=0)
    aggregate_delta = (edit - aggregate_projection) / float(sigma)
    if aggregate_delta.dtype != torch.float32:
        raise GuidedSourceAlignedControllerError(
            "aggregated guided FlowEdit delta must remain fp32"
        )
    return aggregate_delta, weights, scores


def sample_guided_source_aligned_controller(
    renderer_or_diffusion: Any,
    *,
    source_latent: Any,
    source_rgb_frames: int,
    action_prompt_embeds: Any,
    noop_prompt_embeds: Any,
    negative_prompt_embeds: Any,
    config: GuidedSourceAlignedConfig,
    return_trace: bool = False,
) -> Any:
    """Run one active guided FlowEdit arm on exact 81-frame Bernini input."""

    runtime = config.validate()
    torch = _require_torch()
    if type(source_rgb_frames) is not int or source_rgb_frames != EXPECTED_RGB_FRAMES:
        raise GuidedSourceAlignedControllerError(
            f"source_rgb_frames must equal {EXPECTED_RGB_FRAMES}"
        )
    if not isinstance(source_latent, torch.Tensor) or source_latent.dtype != torch.float32:
        raise GuidedSourceAlignedControllerError("source latent must be fp32")
    if source_latent.ndim != 5 or int(source_latent.shape[2]) != EXPECTED_LATENT_PHASES:
        raise GuidedSourceAlignedControllerError(
            "source latent must expose exactly 21 temporal phases"
        )
    if cdf.prompts_are_exactly_identical(action_prompt_embeds, noop_prompt_embeds):
        raise GuidedSourceAlignedControllerError(
            "active guided arm requires distinct action and no-op embeddings"
        )
    if cdf.prompts_are_exactly_identical(action_prompt_embeds, negative_prompt_embeds):
        raise GuidedSourceAlignedControllerError(
            "action and negative embeddings unexpectedly match"
        )
    if cdf.prompts_are_exactly_identical(noop_prompt_embeds, negative_prompt_embeds):
        raise GuidedSourceAlignedControllerError(
            "no-op and negative embeddings unexpectedly match"
        )

    diffusion = cdf.resolve_diffusion_core(renderer_or_diffusion)
    layout, transformer = cdf._validate_runtime_inputs(
        diffusion, source_latent, action_prompt_embeds, noop_prompt_embeds
    )
    # Validate the negative embedding through the same model-facing checks.
    cdf._validate_runtime_inputs(
        diffusion, source_latent, negative_prompt_embeds, noop_prompt_embeds
    )
    raw_controller._validate_exact_geometry(
        layout, source_rgb_frames=source_rgb_frames
    )
    cdf_runtime = cdf.DifferentialFlowConfig(
        num_inference_steps=runtime.num_inference_steps,
        flow_shift=runtime.flow_shift,
        seed=runtime.seed,
        motion_scale=runtime.motion_scale,
    )
    timesteps, raw_intervals = cdf._set_scheduler_timesteps(
        diffusion, cdf_runtime, source_latent.device
    )
    intervals = validate_pinned_sigma_intervals(raw_intervals)
    scheduler_sigma_scalars, scheduler_sigma_fp32_digest = (
        capture_pinned_scheduler_sigma_scalars(diffusion, intervals)
    )
    if scheduler_sigma_fp32_digest != PINNED_UNIPC_SIGMA_FP32_DIGEST:
        raise GuidedSourceAlignedControllerError(
            "pinned UniPC CPU-fp32 sigma bit digest differs before first forward"
        )
    timestep_values = tuple(
        float(value.detach().to(device="cpu", dtype=torch.float64).item())
        for value in timesteps
    )
    sigma_values = tuple(intervals[0][:1]) + tuple(pair[1] for pair in intervals)
    schedule_digest = _object_sha256(
        {
            "timesteps": list(timestep_values),
            "sigmas": list(sigma_values),
            "flow_shift": runtime.flow_shift,
            "steps": runtime.num_inference_steps,
        }
    )
    if schedule_digest != PINNED_UNIPC_SCHEDULE_DIGEST:
        raise GuidedSourceAlignedControllerError(
            "pinned UniPC full schedule digest differs before first transformer forward"
        )

    source_clean = source_latent.detach().to(dtype=torch.float32)
    source_packed = cdf._pack_spatial_latent(source_clean, layout)
    if source_packed.dtype != torch.float32:
        raise GuidedSourceAlignedControllerError("packed source must remain fp32")
    edit_packed = source_packed.clone()
    initial_count = runtime.candidate_count(0)
    previous_noises = [torch.zeros_like(source_packed) for _ in range(initial_count)]

    candidate_counts: list[int] = []
    retention_trace: list[float] = []
    correlation_trace: list[float] = []
    score_trace: list[tuple[float, ...]] = []
    weight_trace: list[tuple[float, ...]] = []
    entropy_trace: list[float] = []
    top1_margin_trace: list[float] = []
    delta_trace: list[float] = []
    update_trace: list[float] = []
    noise_change_trace: list[float] = []
    used_noise_keys: list[dict[str, int]] = []
    used_noise_content: list[dict[str, Any]] = []
    target_parity = True
    source_parity = True
    raw_dtypes: set[str] = set()
    clean_dtypes: set[str] = set()
    branch_counts = {name: 0 for name in BRANCH_ORDER}

    with torch.no_grad():
        source_condition = cdf._patch_source_condition(transformer, source_clean)
        for index, (sigma_value, next_sigma_value) in enumerate(intervals):
            candidate_count = runtime.candidate_count(index)
            if len(previous_noises) != candidate_count:
                raise GuidedSourceAlignedControllerError(
                    "candidate count changed without candidate-zero continuation"
                )
            retained = (
                raw_controller.anc_retained_variance(
                    float(sigma_value), lock_sigma=runtime.anc_lock_sigma
                )
                if runtime.uses_anc
                else 0.0
            )
            candidate_noises = []
            candidate_deltas = []
            per_candidate_change = []
            for candidate_index in range(candidate_count):
                used_noise_keys.append(
                    {
                        "step": index,
                        "candidate": candidate_index,
                        "derived_seed": keyed_noise_seed(
                            runtime.seed, index, candidate_index
                        ),
                    }
                )
                fresh, fresh_content_digest = _draw_keyed_packed_noise(
                    source_latent=source_clean,
                    layout=layout,
                    seed=runtime.seed,
                    step=index,
                    candidate=candidate_index,
                )
                used_noise_content.append(
                    {
                        "step": index,
                        "candidate": candidate_index,
                        "sha256": fresh_content_digest,
                    }
                )
                if runtime.uses_anc:
                    correlated = raw_controller.advance_anc_noise(
                        previous_noises[candidate_index],
                        fresh,
                        retained_variance=retained,
                    )
                else:
                    correlated = fresh
                per_candidate_change.append(
                    (correlated - previous_noises[candidate_index])
                    .square()
                    .mean()
                    .sqrt()
                )
                candidate_noises.append(correlated)
                source_state_packed, target_state_packed = (
                    raw_controller.flowedit_source_target_states(
                        source_packed,
                        edit_packed,
                        correlated,
                        sigma=float(sigma_value),
                    )
                )
                source_state = cdf._unpack_spatial_latent(source_state_packed, layout)
                target_state = cdf._unpack_spatial_latent(target_state_packed, layout)
                if source_state.dtype != torch.float32 or target_state.dtype != torch.float32:
                    raise GuidedSourceAlignedControllerError(
                        "FlowEdit query states must remain fp32"
                    )
                timestep = timesteps[index]

                target_result = _guided_apg_velocity(
                    diffusion=diffusion,
                    transformer=transformer,
                    source_condition=source_condition,
                    query_latent=target_state,
                    condition_prompt_embeds=action_prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    timestep=timestep,
                    sigma=scheduler_sigma_scalars[index],
                    branch="target_action",
                )
                branch_counts["target_negative"] += 1
                branch_counts["target_action"] += 1
                source_result = _guided_apg_velocity(
                    diffusion=diffusion,
                    transformer=transformer,
                    source_condition=source_condition,
                    query_latent=source_state,
                    condition_prompt_embeds=noop_prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    timestep=timestep,
                    sigma=scheduler_sigma_scalars[index],
                    branch="source_noop",
                )
                branch_counts["source_negative"] += 1
                branch_counts["source_noop"] += 1
                target_parity = target_parity and target_result.paired_query_object
                source_parity = source_parity and source_result.paired_query_object
                raw_dtypes.update(
                    (
                        target_result.raw_negative_dtype,
                        target_result.raw_condition_dtype,
                        source_result.raw_negative_dtype,
                        source_result.raw_condition_dtype,
                    )
                )
                clean_dtypes.update(
                    (
                        target_result.negative_clean_dtype,
                        target_result.condition_clean_dtype,
                        target_result.guided_clean_dtype,
                        target_result.guided_velocity_fp32_dtype,
                        source_result.negative_clean_dtype,
                        source_result.condition_clean_dtype,
                        source_result.guided_clean_dtype,
                        source_result.guided_velocity_fp32_dtype,
                    )
                )
                delta = (
                    target_result.velocity_packed_fp32
                    - source_result.velocity_packed_fp32
                )
                if delta.dtype != torch.float32 or not bool(torch.isfinite(delta).all()):
                    raise GuidedSourceAlignedControllerError(
                        "guided FlowEdit delta must be finite fp32"
                    )
                candidate_deltas.append(delta)

            noise_bank = torch.stack(candidate_noises, dim=0)
            delta_bank = torch.stack(candidate_deltas, dim=0)
            if candidate_count == 1:
                aggregate_delta = delta_bank[0]
                score_trace.append(())
                weight_trace.append((1.0,))
                entropy_trace.append(0.0)
                top1_margin_trace.append(1.0)
            else:
                aggregate_delta, weights, scores = _aggregate_candidates(
                    source=source_packed,
                    edit=edit_packed,
                    candidate_deltas=delta_bank,
                    sigma=float(sigma_value),
                    mode=runtime.aggregation,
                )
                score_trace.append(
                    tuple(float(value) for value in scores[:, 0].cpu().tolist())
                )
                weight_trace.append(
                    tuple(float(value) for value in weights[:, 0].cpu().tolist())
                )
                first_batch_weights = weights[:, 0].float()
                entropy_trace.append(
                    float(
                        (-(first_batch_weights * first_batch_weights.clamp_min(1.0e-30).log()).sum())
                        .cpu()
                        .item()
                    )
                )
                ordered_weights = first_batch_weights.sort(descending=True).values
                top1_margin_trace.append(
                    float((ordered_weights[0] - ordered_weights[1]).cpu().item())
                )

            # Both AVG and SGA deliberately retain candidate zero.  The edit
            # states differ only because of their aggregation weights.
            if candidate_count > 1 and index == EARLY_CANDIDATE_STEPS - 1:
                previous_noises = [noise_bank[0]]
            else:
                previous_noises = list(noise_bank.unbind(dim=0))

            delta_sigma = float(next_sigma_value - sigma_value)
            update = delta_sigma * aggregate_delta
            if update.dtype != torch.float32 or not bool(torch.isfinite(update).all()):
                raise GuidedSourceAlignedControllerError(
                    "Euler update must remain finite fp32"
                )
            edit_packed = edit_packed + update
            if edit_packed.dtype != torch.float32 or not bool(
                torch.isfinite(edit_packed).all()
            ):
                raise GuidedSourceAlignedControllerError(
                    "edit state must remain finite fp32"
                )
            candidate_counts.append(candidate_count)
            retention_trace.append(float(retained))
            correlation_trace.append(math.sqrt(float(retained)))
            delta_trace.append(float(aggregate_delta.square().mean().sqrt().cpu().item()))
            update_trace.append(float(update.square().mean().sqrt().cpu().item()))
            noise_change_trace.append(
                float(torch.stack(per_candidate_change).mean().cpu().item())
            )

    expected_evaluations = runtime.expected_candidate_evaluations
    if any(branch_counts[name] != expected_evaluations for name in BRANCH_ORDER):
        raise GuidedSourceAlignedControllerError("guided branch counts differ")
    if raw_dtypes != {"torch.bfloat16"} or clean_dtypes != {"torch.float32"}:
        raise GuidedSourceAlignedControllerError("guided APG dtype closure differs")
    total_calls = sum(branch_counts.values())
    if total_calls != runtime.expected_shared_step_calls:
        raise GuidedSourceAlignedControllerError("guided transformer call count differs")
    bank = noise_bank_pairing_contract(seed=runtime.seed)
    result = cdf._unpack_spatial_latent(edit_packed, layout)
    if not return_trace:
        return result
    observed_used_noise_digest = _object_sha256(used_noise_keys)
    if observed_used_noise_digest != used_noise_key_digest(runtime):
        raise GuidedSourceAlignedControllerError("consumed keyed noise bank differs")
    used_noise_content_digest = _object_sha256(used_noise_content)
    candidate0_noise_content_digest = _object_sha256(
        [row for row in used_noise_content if row["candidate"] == 0]
    )
    trace = GuidedSourceAlignedTrace(
        arm=runtime.arm,
        sigmas=sigma_values,
        timesteps=timestep_values,
        schedule_digest=schedule_digest,
        scheduler_sigma_fp32_digest=scheduler_sigma_fp32_digest,
        scheduler_sigma_dtype="torch.float32",
        scheduler_sigma_device="cpu",
        scheduler_sigma_direct_views=True,
        candidate_counts=tuple(candidate_counts),
        anc_retained_variance=tuple(retention_trace),
        anc_nominal_correlation=tuple(correlation_trace),
        sga_scores=tuple(score_trace),
        sga_weights=tuple(weight_trace),
        sga_entropy=tuple(entropy_trace),
        sga_top1_margin=tuple(top1_margin_trace),
        delta_rms=tuple(delta_trace),
        update_rms=tuple(update_trace),
        noise_state_change_rms=tuple(noise_change_trace),
        fresh_noise_draws=len(used_noise_keys),
        used_noise_key_digest=observed_used_noise_digest,
        used_fresh_noise_content_digest=used_noise_content_digest,
        candidate0_fresh_noise_content_digest=candidate0_noise_content_digest,
        full_noise_bank_digest=bank["full_bank_digest"],
        candidate0_noise_bank_digest=bank["candidate0_bank_digest"],
        branch_order=BRANCH_ORDER,
        branch_counts=tuple(branch_counts[name] for name in BRANCH_ORDER),
        total_shared_step_calls=total_calls,
        apg_parameters=(
            ("guidance_mode", APG_GUIDANCE_MODE),
            ("guidance_scale", APG_GUIDANCE_SCALE),
            ("eta", APG_ETA),
            ("norm_threshold", APG_NORM_THRESHOLD),
            ("momentum", APG_MOMENTUM),
        ),
        target_branch_query_parity=target_parity,
        source_branch_query_parity=source_parity,
        raw_velocity_dtype="torch.bfloat16",
        guided_velocity_dtype="torch.float32",
        apg_clean_dtype="torch.float32",
        delta_dtype="torch.float32",
        edit_state_dtype="torch.float32",
        candidate_continuation="candidate_0",
        weighted_noise_collapse_used=False,
        anc_initial_predecessor_policy="zero_initialized_per_dynaedit_pseudocode",
    )
    return result, trace


__all__ = [
    "ANC_LOCK_SIGMA",
    "APG_ETA",
    "APG_GUIDANCE_MODE",
    "APG_GUIDANCE_SCALE",
    "APG_MOMENTUM",
    "APG_NORM_THRESHOLD",
    "BRANCH_ORDER",
    "EARLY_CANDIDATES",
    "EARLY_CANDIDATE_STEPS",
    "EXPECTED_FLOW_SHIFT",
    "EXPECTED_LATENT_PHASES",
    "EXPECTED_RGB_FRAMES",
    "EXPECTED_SEED",
    "EXPECTED_STEPS",
    "GUIDED_ARMS",
    "GuidedAPGResult",
    "GuidedSourceAlignedConfig",
    "GuidedSourceAlignedControllerError",
    "GuidedSourceAlignedTrace",
    "PINNED_UNIPC_START_SIGMA",
    "PINNED_UNIPC_START_SIGMA_ATOL",
    "SGA_TEMPERATURE",
    "guided_controller_contract",
    "keyed_noise_seed",
    "noise_bank_pairing_contract",
    "sample_guided_source_aligned_controller",
    "validate_pinned_sigma_intervals",
]
