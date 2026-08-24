"""Transparent instruction heuristics for separating action from endpoint edits.

This is a high-recall triage layer, not a substitute for an instruction LLM or
human labels.  Every decision includes matched terms so thresholds remain
auditable.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class InstructionSemantics:
    label: str
    matched_motion_terms: tuple[str, ...]
    matched_static_cues: tuple[str, ...]
    matched_endpoint_terms: tuple[str, ...]
    matched_appearance_terms: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


MOTION_SUPPRESSION_PATTERNS = (
    r"\bstop(?:ping|ped)?\b",
    r"\bcome to (?:a )?rest\b",
    r"\bstand(?:ing)? still\b",
    r"\bremain completely still\b",
    r"\bfreeze\b",
)

STATIC_POSE_CUES = (
    r"\bappear(?:s|ing)? to be\b",
    r"\bas if\b",
    r"\bpoised to\b",
    r"\bsuspended\b",
    r"\bready to\b",
    r"\bpose[ds]?\b",
    r"\bmid[- ](?:jump|air|stride|motion)\b",
    r"\bjust above\b",
    r"\binto (?:a |the )?.{0,30}\bposition\b",
)

CONTINUOUS_ACTION_PATTERNS = (
    r"\bwalk(?:s|ing|ed)?\b",
    r"\brun(?:s|ning|ran)?\b",
    r"\bjump(?:s|ing|ed)?\b",
    r"\bleap(?:s|ing|ed)?\b",
    r"\bdanc(?:e|es|ing|ed)\b",
    r"\bfly(?:ing|ies|flew)?\b",
    r"\bswim(?:s|ming|swam)?\b",
    r"\bcrawl(?:s|ing|ed)?\b",
    r"\bclimb(?:s|ing|ed)?\b",
    r"\bthrow(?:s|ing|threw|n)?\b",
    r"\bcatch(?:es|ing|caught)?\b",
    r"\bkick(?:s|ing|ed)?\b",
    r"\bpunch(?:es|ing|ed)?\b",
    r"\bbounc(?:e|es|ing|ed)\b",
    r"\broll(?:s|ing|ed)?\b",
    r"\bspin(?:s|ning|spun)?\b",
    r"\bswing(?:s|ing|swung)?\b",
    r"\bfall(?:s|ing|fell|en)?\b",
    r"\bpour(?:s|ing|ed)?\b",
    r"\bsplash(?:es|ing|ed)?\b",
    r"\bexplod(?:e|es|ing|ed)\b",
    r"\bburst(?:s|ing)?\b",
    r"\bpick(?:s|ing|ed)?\b",
    r"\breach(?:es|ing|ed)?\b",
    r"\bflow(?:s|ing|ed)?\b",
    r"\bride(?:s|riding|rode|ridden)?\b",
    r"\bdrive(?:s|driving|drove|driven)?\b",
    r"\bland(?:s|ing|ed)?\b",
    r"\bcontinuously\b",
    r"\brepeated(?:ly)?\b",
    r"\bmultiple times\b",
)

ENDPOINT_POSE_PATTERNS = (
    r"\bstand(?:s|ing)?(?: up)?\b",
    r"\bsit(?:s|ting)?(?: down)?\b",
    r"\bkneel(?:s|ing|ed)?\b",
    r"\bcrouch(?:es|ing|ed)?\b",
    r"\bbend(?:s|ing|bent)?\b",
    r"\bface(?:s|ing)? (?:the|toward|forward|backward|left|right)\b",
    r"\bturn (?:his|her|their|its|the)? ?(?:head|torso|body|face)\b",
    r"\b(?:raise|lower|lift|extend|spread|fold) (?:his |her |their |its |the )?"
    r"(?:arm|arms|hand|hands|leg|legs|head|wing|wings|branch|branches)\b",
    r"\bhold(?:s|ing)?\b",
)

RIGID_TRANSFORM_PATTERNS = (
    r"\bmove\b",
    r"\bshift\b",
    r"\breposition\b",
    r"\brotate\b",
    r"\btilt\b",
    r"\balign\b",
    r"\barrange\b",
    r"\bplace\b",
    r"\braise\b",
    r"\blower\b",
    r"\btranslate\b",
    r"\bopen\b",
    r"\bclose[ds]?\b",
)

APPEARANCE_PATTERNS = (
    r"\bcolor\b",
    r"\btexture\b",
    r"\bmaterial\b",
    r"\bstyle\b",
    r"\breshape\b",
    r"\benlarge\b",
    r"\bshrink\b",
    r"\bthick(?:er|en)?\b",
    r"\bthin(?:ner)?\b",
    r"\bwider\b",
    r"\bnarrower\b",
    r"\blengthen\b",
    r"\bshorten\b",
    r"\bdecay(?:ed|ing)?\b",
    r"\bripe\b",
    r"\bwrinkl(?:e|ed)\b",
    r"\bremove\b",
    r"\badd\b",
    r"\breplace\b",
)

FALSE_MOTION_CONTEXTS = (
    r"\bride height\b",
    r"\btrain ride\b",
    r"\bamusement (?:park )?ride\b",
    r"\broll[- ]down door\b",
    r"\brunning decks?\b",
    r"\bcontinuous ring\b",
    r"\bswing set\b",
    r"\bsimulat(?:e|ing) a swing\b",
    r"\broll up\b",
    r"\ballow(?:ing)? .{0,40}\bfall naturally\b",
    r"\bfac(?:e|ing) the direction .{0,30}\bwalk(?:ing|s)?\b",
    r"\bfor someone to (?:walk|pass)\b",
    r"\blook(?:ing)? at .{0,30}\bfalling\b",
    r"\bthomas ride\b",
    r"\bwalk space\b",
    r"\bon (?:a|the) swing\b",
    r"\bfireworks? bursts?\b",
)

ENVIRONMENTAL_CONTEXTS = (
    r"\bwaterfall\b",
    r"\briver\b",
    r"\bwave\b",
    r"\baurora\b",
    r"\bclouds?\b",
    r"\bsmoke\b",
    r"\bflames?\b",
    r"\bfire\b",
    r"\bsnow\b",
    r"\brain\b",
    r"\bfoliage\b",
    r"\bbranches?\b",
)

ACTOR_CONTEXTS = (
    r"\bperson\b",
    r"\bman\b",
    r"\bwoman\b",
    r"\bchild\b",
    r"\bbaby\b",
    r"\brunner\b",
    r"\brider\b",
    r"\bclimber\b",
    r"\bdog\b",
    r"\bcat\b",
    r"\bkitten\b",
    r"\bbird\b",
    r"\bmonkey\b",
    r"\bhorse\b",
    r"\bjet ski\b",
)


def _matches(text: str, patterns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(pattern for pattern in patterns if re.search(pattern, text))


def classify_instruction(instruction: str) -> InstructionSemantics:
    text = re.sub(r"\s+", " ", instruction.strip().lower())
    suppression = _matches(text, MOTION_SUPPRESSION_PATTERNS)
    static_cues = _matches(text, STATIC_POSE_CUES)
    motion = _matches(text, CONTINUOUS_ACTION_PATTERNS)
    endpoint = _matches(text, ENDPOINT_POSE_PATTERNS)
    rigid = _matches(text, RIGID_TRANSFORM_PATTERNS)
    appearance = _matches(text, APPEARANCE_PATTERNS)
    false_motion_contexts = _matches(text, FALSE_MOTION_CONTEXTS)
    environmental_contexts = _matches(text, ENVIRONMENTAL_CONTEXTS)
    actor_contexts = _matches(text, ACTOR_CONTEXTS)
    if false_motion_contexts:
        motion = ()

    if suppression:
        label = "motion_suppression"
        motion = tuple(dict.fromkeys((*motion, *suppression)))
    elif appearance:
        label = "shape_appearance"
    elif motion and not static_cues:
        label = (
            "environmental_motion"
            if environmental_contexts and not actor_contexts
            else "continuous_action"
        )
    elif endpoint or (motion and static_cues):
        label = "endpoint_pose"
    elif rigid:
        label = "rigid_transform"
    else:
        label = "ambiguous"
    return InstructionSemantics(
        label=label,
        matched_motion_terms=motion,
        matched_static_cues=static_cues,
        matched_endpoint_terms=tuple(dict.fromkeys((*endpoint, *rigid))),
        matched_appearance_terms=appearance,
    )
