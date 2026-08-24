"""Motive-inspired utilities for motion/action data curation.

The package deliberately separates two kinds of representation:

* geometry descriptors, which are cheap and checkpoint independent; and
* motion-masked gradient fingerprints, which are queryable but tied to a
  particular generator checkpoint, loss, parameter subset, timestep, and
  shared noise realization.
"""

from .descriptor import (
    DescriptorConfig,
    encode_action_descriptor,
    encode_factorized_action_delta,
)
from .geometry import (
    MotionAnalysis,
    MotionConfig,
    MotionMetrics,
    analyze_video,
    classify_motion,
    delta_motion_mask,
    normalize_motion_magnitude,
)
from .selection import SelectionResult, majority_vote_select, rank_by_query
from .semantics import InstructionSemantics, classify_instruction

__all__ = [
    "DescriptorConfig",
    "InstructionSemantics",
    "MotionAnalysis",
    "MotionConfig",
    "MotionMetrics",
    "SelectionResult",
    "analyze_video",
    "classify_motion",
    "classify_instruction",
    "delta_motion_mask",
    "encode_action_descriptor",
    "encode_factorized_action_delta",
    "majority_vote_select",
    "normalize_motion_magnitude",
    "rank_by_query",
]
