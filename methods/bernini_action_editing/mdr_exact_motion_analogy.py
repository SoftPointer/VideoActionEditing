#!/usr/bin/env python3
"""Exact source/donor motion-analogy pretexts for Bernini MDR-Edit.

The training primitive is deliberately cross-identity and relative::

    source=A, donor_packet=(B,T(B)), target=T(A), with A != B.

``target`` is constructed inside this module from the source and a registered
temporal program.  It is never supplied by an action-pair dataset.  The paired
donor makes ``T`` observable relative to ``B``; a lone ``T(B)`` generally does
not reveal whether it is original speed/phase or a transformed one.  Donor
appearance still cannot be the correct regression answer.  A generic
instruction may be shared by all
programs; correct/wrong-program donor swaps can therefore test whether a model
actually reads donor timing instead of solving the task from text alone.

This tensor core neither extracts nor accepts masks, tracks, flow, pose, boxes,
trajectories, target videos, generated action proposals, or edited keyframes.
It constructs exact 81-frame RGB-space pretexts only.  Encoding the returned
videos with the pinned Bernini VAE is a separate, auditable materialization
step; directly permuting the 21 causal VAE phases is not equivalent and is not
authorized here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any, Mapping, Optional, Sequence

import torch


SCHEMA_VERSION = "bernini-mdr-exact-motion-analogy-v1"
FRAME_COUNT = 81
GENERIC_DONOR_INSTRUCTION = (
    "Apply to the source video the temporal program demonstrated by the "
    "change between the two motion-donor videos."
)

PROGRAM_KINDS = (
    "identity",
    "reverse",
    "speed_up",
    "slow_down",
    "pause_then_catch_up",
    "cyclic_phase",
)

FORBIDDEN_EXTERNAL_INPUT_NAMES = frozenset(
    {
        "target",
        "target_video",
        "target_latent",
        "action_proposal",
        "generated_action_video",
        "mask",
        "motion_mask",
        "flow",
        "optical_flow",
        "pose",
        "track",
        "tracks",
        "box",
        "boxes",
        "trajectory",
        "trajectories",
        "edited_first_frame",
    }
)


class MDRMotionAnalogyError(ValueError):
    """The exact motion-analogy construction is ambiguous or invalid."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MDRMotionAnalogyError(f"contract is not canonical JSON: {error}") from error


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MDRMotionAnalogyError(f"{label} must be a lowercase SHA-256")
    return value


def _finite_real(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MDRMotionAnalogyError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise MDRMotionAnalogyError(f"{label} must be finite")
    return result


def _exact_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MDRMotionAnalogyError(f"{label} must be an exact integer")
    return value


@dataclass(frozen=True)
class TemporalProgram:
    """One registered 81-frame output-to-input temporal resampling program."""

    kind: str
    parameter: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in PROGRAM_KINDS:
            raise MDRMotionAnalogyError(
                f"program kind must be one of {PROGRAM_KINDS}, got {self.kind!r}"
            )
        parameter = _finite_real(self.parameter, label="program parameter")
        if self.kind in {"identity", "reverse"} and parameter != 0.0:
            raise MDRMotionAnalogyError(f"{self.kind} requires parameter=0")
        if self.kind == "speed_up" and not 0.2 <= parameter <= 0.8:
            raise MDRMotionAnalogyError("speed_up exponent must be in [0.2,0.8]")
        if self.kind == "slow_down" and not 1.25 <= parameter <= 4.0:
            raise MDRMotionAnalogyError("slow_down exponent must be in [1.25,4]")
        if self.kind == "pause_then_catch_up" and not 0.1 <= parameter <= 0.5:
            raise MDRMotionAnalogyError("pause fraction must be in [0.1,0.5]")
        if self.kind == "cyclic_phase":
            phase = int(parameter)
            if float(phase) != parameter or not 1 <= phase < FRAME_COUNT:
                raise MDRMotionAnalogyError(
                    f"cyclic phase must be an integer in [1,{FRAME_COUNT - 1}]"
                )

    @property
    def output_to_input(self) -> torch.Tensor:
        """Return detached FP64 source coordinates for all 81 output frames."""

        end = float(FRAME_COUNT - 1)
        position = torch.linspace(0.0, 1.0, FRAME_COUNT, dtype=torch.float64)
        if self.kind == "identity":
            coordinate = position * end
        elif self.kind == "reverse":
            coordinate = (1.0 - position) * end
        elif self.kind in {"speed_up", "slow_down"}:
            coordinate = position.pow(float(self.parameter)) * end
        elif self.kind == "pause_then_catch_up":
            pause = float(self.parameter)
            # Hold the first frame for ``pause`` of the output, then traverse
            # the complete original interval.  Both endpoints remain exact.
            coordinate = ((position - pause).clamp_min(0.0) / (1.0 - pause)) * end
        else:
            coordinate = torch.remainder(position * end + float(self.parameter), FRAME_COUNT)
        return coordinate.detach().contiguous()

    @property
    def interpolation_matrix(self) -> torch.Tensor:
        """Return the exact FP64 linear-resampling matrix ``[81,81]``."""

        coordinate = self.output_to_input
        lower = coordinate.floor().to(torch.int64)
        upper = torch.where(
            lower == FRAME_COUNT - 1,
            lower,
            lower + 1,
        )
        fraction = coordinate - lower.to(torch.float64)
        matrix = torch.zeros(FRAME_COUNT, FRAME_COUNT, dtype=torch.float64)
        rows = torch.arange(FRAME_COUNT, dtype=torch.int64)
        matrix[rows, lower] += 1.0 - fraction
        matrix[rows, upper] += fraction
        if not torch.equal(matrix.sum(dim=1), torch.ones(FRAME_COUNT, dtype=torch.float64)):
            raise MDRMotionAnalogyError("temporal interpolation rows do not sum exactly to one")
        return matrix.detach().contiguous()

    def as_dict(self) -> dict[str, Any]:
        coordinates = self.output_to_input
        return {
            "kind": self.kind,
            "parameter_hex": float(self.parameter).hex(),
            "frame_count": FRAME_COUNT,
            "coordinate_dtype": str(coordinates.dtype),
            "coordinate_sha256": _sha256(coordinates.numpy().tobytes(order="C")),
            "rgb_resampling": "linear_output_to_input",
            "vae_phase_permutation_authorized": False,
        }

    @property
    def digest(self) -> str:
        return _sha256(_canonical_json(self.as_dict()))


def _validate_video(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise MDRMotionAnalogyError(f"{name} must be a torch.Tensor")
    if value.layout != torch.strided or value.device.type == "meta":
        raise MDRMotionAnalogyError(f"{name} must be a dense materialized tensor")
    if value.ndim != 5 or tuple(int(item) for item in value.shape[:3]) != (
        1,
        3,
        FRAME_COUNT,
    ):
        raise MDRMotionAnalogyError(
            f"{name} must have exact [1,3,{FRAME_COUNT},H,W] RGB layout"
        )
    if int(value.shape[3]) <= 0 or int(value.shape[4]) <= 0:
        raise MDRMotionAnalogyError(f"{name} spatial dimensions must be positive")
    if value.dtype != torch.float32:
        raise MDRMotionAnalogyError(f"{name} must use detached FP32")
    if value.requires_grad or value.grad_fn is not None:
        raise MDRMotionAnalogyError(f"{name} must be detached")
    if not value.is_contiguous():
        raise MDRMotionAnalogyError(f"{name} must be contiguous")
    if not bool(torch.isfinite(value).all().item()):
        raise MDRMotionAnalogyError(f"{name} contains NaN or infinity")
    if bool(((value < -1.0) | (value > 1.0)).any().item()):
        raise MDRMotionAnalogyError(f"{name} must use normalized RGB range [-1,1]")
    return value


def apply_temporal_program(video: Any, program: TemporalProgram) -> torch.Tensor:
    """Apply ``program`` to normalized RGB, returning detached contiguous FP32."""

    source = _validate_video("video", video)
    if not isinstance(program, TemporalProgram):
        raise MDRMotionAnalogyError("program must be a TemporalProgram")
    matrix = program.interpolation_matrix.to(device=source.device)
    # Accumulate the exact registered interpolation in FP64 and round once.
    transformed = torch.einsum("ot,bcthw->bcohw", matrix, source.to(torch.float64))
    result = transformed.to(torch.float32).detach().contiguous()
    _validate_video("transformed video", result)
    return result


@dataclass(frozen=True)
class MotionAnalogyExample:
    source_identity_video: torch.Tensor
    motion_donor_before_video: torch.Tensor
    motion_donor_after_video: torch.Tensor
    regression_target_video: torch.Tensor
    program: TemporalProgram
    source_identity_sha256: str
    donor_identity_sha256: str
    instruction: str
    receipt: Mapping[str, Any]


def build_motion_analogy_example(
    source_identity_video: Any,
    donor_base_video: Any,
    program: TemporalProgram,
    *,
    source_identity_sha256: str,
    donor_identity_sha256: str,
    instruction: str = GENERIC_DONOR_INSTRUCTION,
) -> MotionAnalogyExample:
    """Construct ``A, (B,T(B)), T(A)`` without an external target."""

    source = _validate_video("source_identity_video", source_identity_video)
    donor_base = _validate_video("donor_base_video", donor_base_video)
    if tuple(source.shape) != tuple(donor_base.shape):
        raise MDRMotionAnalogyError("source and donor videos must share exact RGB geometry")
    source_digest = _require_sha256(
        source_identity_sha256, label="source identity digest"
    )
    donor_digest = _require_sha256(
        donor_identity_sha256, label="donor identity digest"
    )
    if source_digest == donor_digest:
        raise MDRMotionAnalogyError("motion analogy requires source identity A != donor identity B")
    if not isinstance(program, TemporalProgram):
        raise MDRMotionAnalogyError("program must be a TemporalProgram")
    if (
        not isinstance(instruction, str)
        or not instruction
        or instruction != instruction.strip()
        or "\x00" in instruction
    ):
        raise MDRMotionAnalogyError("instruction must be canonical non-empty text")

    donor = apply_temporal_program(donor_base, program)
    target = apply_temporal_program(source, program)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "construction": "source=A,donor_packet=(B,T(B)),target=T(A)",
        "source_identity_sha256": source_digest,
        "donor_identity_sha256": donor_digest,
        "source_and_donor_identity_distinct": True,
        "program": program.as_dict(),
        "program_digest": program.digest,
        "instruction_sha256": _sha256(instruction.encode("utf-8")),
        "instruction_is_generic_donor_follow": instruction == GENERIC_DONOR_INSTRUCTION,
        "target_origin": "deterministic_RGB_transform_of_source_inside_builder",
        "external_target_accepted": False,
        "donor_appearance_is_correct_target": False,
        "relative_donor_program_observable": True,
        "single_after_only_donor_is_main_training_input": False,
        "paired_action_dataset_used": False,
        "mask_flow_pose_track_trajectory_used": False,
        "frame_count": FRAME_COUNT,
        "latent_frame_count_after_pinned_Wan_VAE": 21,
        "direct_21_phase_permutation_authorized": False,
    }
    receipt["receipt_digest"] = _sha256(_canonical_json(receipt))
    return MotionAnalogyExample(
        source_identity_video=source.detach().clone().contiguous(),
        motion_donor_before_video=donor_base.detach().clone().contiguous(),
        motion_donor_after_video=donor,
        regression_target_video=target,
        program=program,
        source_identity_sha256=source_digest,
        donor_identity_sha256=donor_digest,
        instruction=instruction,
        receipt=receipt,
    )


def registered_program_grid() -> tuple[TemporalProgram, ...]:
    """Return the preregistered small canary grid; it is not sample-tuned."""

    return (
        TemporalProgram("identity"),
        TemporalProgram("reverse"),
        TemporalProgram("speed_up", 0.5),
        TemporalProgram("slow_down", 2.0),
        TemporalProgram("pause_then_catch_up", 0.25),
        TemporalProgram("cyclic_phase", 20.0),
    )


def motion_analogy_contract() -> dict[str, Any]:
    """Return the static scientific limits of this self-supervised pretext."""

    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pretext": "cross_identity_motion_analogy",
        "equation": "source=A; donor_packet=(B,T(B)); target=T(A); A!=B",
        "programs": [program.as_dict() for program in registered_program_grid()],
        "generic_instruction": GENERIC_DONOR_INSTRUCTION,
        "identification_controls": [
            "same_text_correct_vs_wrong_program_donor",
            "same_text_correct_vs_identity_donor",
            "relative_packet_vs_after_only_ablation",
            "condition_order_swap",
            "source_donor_identity_disjoint",
            "heldout_source_and_donor",
        ],
        "action_pair_target_used": False,
        "generated_action_donor_used_by_this_pretext": False,
        "single_after_only_donor_authorized_as_main": False,
        "natural_semantic_action_learned_by_this_pretext": False,
        "mask_flow_pose_track_trajectory_used": False,
        "training_claim": (
            "learns whether Bernini can bind source appearance to a donor temporal program; "
            "does not by itself establish open-vocabulary action editing"
        ),
    }
    value["contract_digest"] = _sha256(_canonical_json(value))
    return value


__all__ = [
    "FORBIDDEN_EXTERNAL_INPUT_NAMES",
    "FRAME_COUNT",
    "GENERIC_DONOR_INSTRUCTION",
    "MDRMotionAnalogyError",
    "MotionAnalogyExample",
    "PROGRAM_KINDS",
    "SCHEMA_VERSION",
    "TemporalProgram",
    "apply_temporal_program",
    "build_motion_analogy_example",
    "motion_analogy_contract",
    "registered_program_grid",
]
