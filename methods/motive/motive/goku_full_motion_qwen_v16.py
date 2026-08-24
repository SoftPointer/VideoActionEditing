"""Lightweight two-stage Qwen3-VL annotation for full-motion I2V edits.

The source video is evidence, but Wan only receives its exact first frame.
Consequently every independently moving source subject must receive a complete,
self-contained target trajectory and the camera trajectory must be explicit.

This lineage intentionally does *two* normal-path model calls per row:

1. one visual source-motion census; and
2. one visual target-plan proposal conditioned on that census.

There is no repeated free-text census, critic pass, cross-row gate, or global
pass-rate threshold.  Model JSON is allowed a small deterministic mechanical
repair (fence removal, redundant schema/ID insertion, aliases, and ordering),
but missing subjects, missing evidence, unresolved crowds, or semantic no-ops
are never repaired.  Each row publishes an immutable result and terminal
receipt.  A successful row additionally publishes a one-row passed JSONL file
so downstream generation can consume it without waiting for a batch.

Single-row execution remains available for smoke tests.  Full-scale runs use
deterministic strided persistent workers: each four-GPU process constructs one
Qwen backend, then annotates all indices it owns without reloading model
weights or reparsing the manifest for every row.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

from .goku_action_anchor_qwen import (
    _reject_backend_cpu_or_disk_offload,
    validate_input_row,
    verify_exact_i0_binding,
)
from .goku_full_motion_contract import CAMERA_MOTION_CLASSES, ENTITY_TYPES
from .goku_full_motion_qwen import _build_visuals
from .qwen_filter import LocalQwenBackend, _file_digest


SOURCE_CENSUS_SCHEMA = "motive-goku-full-motion-v16-source-census-v1"
SOURCE_SUBJECT_SCHEMA = "motive-goku-full-motion-v16-source-subject-v1"
SOURCE_CAMERA_SCHEMA = "motive-goku-full-motion-v16-source-camera-v1"
MOTION_EVIDENCE_SCHEMA = "motive-goku-full-motion-v16-evidence-v1"
TARGET_PLAN_SCHEMA = "motive-goku-full-motion-v16-target-plan-v1"
TARGET_SUBJECT_SCHEMA = "motive-goku-full-motion-v16-target-subject-v1"
TARGET_CAMERA_SCHEMA = "motive-goku-full-motion-v16-target-camera-v1"
TARGET_COVERAGE_SCHEMA = "motive-goku-full-motion-v16-target-coverage-v1"
COMPILED_INSTRUCTION_SCHEMA = (
    "motive-goku-full-motion-v16-compiled-instruction-v1"
)
INSTRUCTION_CLAUSE_SCHEMA = "motive-goku-full-motion-v16-clause-v1"
RECORD_SCHEMA = "motive-goku-full-motion-qwen-v16-record-v1"
PASSED_SCHEMA = "motive-goku-full-motion-qwen-v16-passed-v1"
ROW_RECEIPT_SCHEMA = "motive-goku-full-motion-qwen-v16-row-receipt-v1"
MECHANICAL_REPAIR_SCHEMA = "motive-goku-full-motion-qwen-v16-repair-v1"

DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_NFRAMES = 16
DEFAULT_MAX_PIXELS = 2_359_296
DEFAULT_TILE_WIDTH = 512
DEFAULT_MOSAIC_COLUMNS = 4
MAX_DYNAMIC_SUBJECTS = 6
FRAME_COUNT = 81
FPS = 25.0
MAX_TIMELINE_FRAME_INDEX = FRAME_COUNT - 1
TIMELINE_SPAN_SECONDS = MAX_TIMELINE_FRAME_INDEX / FPS

_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SUBJECT_ID_RE = re.compile(r"subject_(\d{2})\Z")
_SIGNATURE_RE = re.compile(r"[a-z][a-z0-9_]{1,95}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_UNSAFE_TARGET_RE = re.compile(
    r"\b(?:same|original|source|existing|previous|prior|earlier)\s+"
    r"(?:[a-z][a-z'-]*\s+){0,3}"
    r"(?:motion|action|trajectory|movement|gesture|gait|behavior|behaviour|pace)\b"
    r"|\b(?:as|like)\s+in\s+(?:the\s+)?(?:source|original)(?:\s+video)?\b"
    r"|\b(?:keeps?|continues?|maintains?)\s+(?:on\s+)?(?:doing\s+)?what\s+"
    r"(?:it|he|she|they)\s+(?:was|were)\s+doing\b"
    r"|\b(?:keeps?|continues?|maintains?)\b(?:\s+[a-z][a-z'-]*){0,3}\s+"
    r"(?:as\s+before|as\s+previously)\b"
    r"|\b(?:source|original)\s+"
    r"(?:video|clip|footage|sequence|frames?)\b"
    r"|\b(?:follows?|continues?|keeps?|resumes?|repeats?)\s+"
    r"(?:what|whatever)\s+(?:happens?|occurs?|is\s+shown)\s+"
    r"(?:next|later)\b"
    r"|\b(?:motion|action|trajectory|movement|gesture|gait|behavior|behaviour)\s+"
    r"(?:visible|shown|seen|depicted)\s+"
    r"(?:in|after|beyond)\s+(?:the\s+)?"
    r"(?:later|subsequent|following|future|remaining|first|initial|i0)"
    r"(?:\s+frames?)?\b"
    r"|\b(?:the\s+)?(?:rest|remainder)\s+of\s+"
    r"(?:(?:the|this|that|his|her|its|their)\s+)?"
    r"(?:clip|video|footage|sequence|recorded\s+action|action|motion)\b"
    r"|\b(?:later|remaining|subsequent|following|future)\s+"
    r"(?:frames?|footage|clip|video|action|motion|sequence)\b"
    r"|\b(?:frames?|footage)\s+that\s+"
    r"(?:follow|follows|come|comes)\s+(?:after\s+)?"
    r"(?:i0|the\s+(?:first|initial)\s+frame)\b"
    r"|\b(?:what|whatever)\s+(?:it|he|she|they)\s+"
    r"(?:does|do)\s+after\s+(?:i0|the\s+(?:first|initial)\s+frame)\b"
    r"|\b(?:what|whatever)\s+(?:happens?|occurs?)\s+"
    r"(?:after|beyond)\s+(?:i0|frame\s+(?:zero|0)|"
    r"the\s+(?:first|initial)\s+frame)\b"
    r"|\b(?:what|whatever)\s+(?:(?:is|was|were)\s+)?"
    r"(?:shown|seen|depicted|recorded)\s+(?:after|beyond)\s+"
    r"(?:i0|frame\s+(?:zero|0)|the\s+(?:first|initial)\s+frame)\b"
    r"|\b(?:as|just\s+as)\s+(?:shown|seen|depicted|recorded)\s+"
    r"(?:in\s+)?(?:the\s+)?(?:later|remaining|subsequent|following|future)\b",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(
    r"(?:<[^<>]+>|\b(?:todo|tbd|unknown|unclear|something|some action)\b)",
    re.IGNORECASE,
)
# These labels identify sparse Qwen visual samples, not executable output
# frames.  They must never leak into a target trajectory or edit instruction.
_SAMPLING_FRAME_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[SF](?:[-_]\s*|\s+)?\d+|"
    r"C(?:[-_]\s*|\s+)?0|C[-_]?[MF])(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_TEMPORAL_NUMBER_RE = r"(?:\d+(?:\.\d+)?|\.\d+)"
_TIME_UNIT_RE = (
    r"(?:milliseconds?|msecs?|ms|seconds?|secs?|sec|s|minutes?|mins?|min|"
    r"hours?|hrs?|hr)"
)
_TIME_AMOUNT_RE = re.compile(
    rf"(?<![A-Za-z0-9_.])(?P<value>{_TEMPORAL_NUMBER_RE})"
    rf"(?P<join>\s*[-\u2013\u2014]?\s*)"
    rf"(?:(?:full|whole|entire)\s+)?"
    rf"(?P<unit>{_TIME_UNIT_RE})\b",
    re.IGNORECASE,
)
_FRAME_DURATION_RE = re.compile(
    rf"(?<![A-Za-z0-9_.])(?P<value>{_TEMPORAL_NUMBER_RE})"
    r"(?P<join>\s*[-\u2013\u2014]?\s*)"
    r"(?:(?:full|whole|entire)\s+)?"
    r"(?P<unit>frames?|fr)\b",
    re.IGNORECASE,
)
_FRAME_REFERENCE_RE = re.compile(
    rf"\bframes?\s*(?:(?:number|index|indices|no\.?)\s*)?(?:#|:)?\s*"
    rf"(?P<value>{_TEMPORAL_NUMBER_RE})\b",
    re.IGNORECASE,
)
_FRAME_RANGE_RE = re.compile(
    rf"\bframes?\s*(?:(?:number|index|indices|no\.?)\s*)?(?:#|:)?\s*"
    rf"(?P<start>{_TEMPORAL_NUMBER_RE})\s*"
    rf"(?:-|\u2013|\u2014|to|through|and)\s*"
    rf"(?:frames?\s*)?(?P<end>{_TEMPORAL_NUMBER_RE})\b",
    re.IGNORECASE,
)
_ORDINAL_FRAME_REFERENCE_RE = re.compile(
    rf"\b(?P<value>{_TEMPORAL_NUMBER_RE})(?:st|nd|rd|th)\s+frames?\b",
    re.IGNORECASE,
)
_FRAME_INDEX_LIST_RE = re.compile(
    r"\bframes?\s+(?:indices|indexes|numbers)\s+(?P<values>"
    r"\d+(?:\s*,\s*\d+)*(?:\s*,?\s*and\s*\d+)?)\b",
    re.IGNORECASE,
)
_SMALL_NUMBER_WORDS = {
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
    "thirteen": 13.0,
    "fourteen": 14.0,
    "fifteen": 15.0,
    "sixteen": 16.0,
    "seventeen": 17.0,
    "eighteen": 18.0,
    "nineteen": 19.0,
    "twenty": 20.0,
    "thirty": 30.0,
    "forty": 40.0,
    "fifty": 50.0,
    "sixty": 60.0,
    "seventy": 70.0,
    "eighty": 80.0,
    "ninety": 90.0,
}
_NUMBER_WORD_ATOM_RE = "(?:" + "|".join(
    sorted(_SMALL_NUMBER_WORDS, key=len, reverse=True)
) + r"|hundred)"
_WORD_NUMBER_VALUE_RE = (
    r"(?:a|an|half(?:\s+a)?|"
    + _NUMBER_WORD_ATOM_RE
    + r"(?:(?:\s+and\s+|\s+|[-\u2013\u2014])"
    + r"(?:a\s+half|"
    + _NUMBER_WORD_ATOM_RE
    + r"))*)"
)
_WORD_AMOUNT_RE = re.compile(
    r"\b(?P<value>"
    + _WORD_NUMBER_VALUE_RE
    + r")"
    r"(?P<join>\s*[-\u2013\u2014]\s*|\s+)"
    r"(?:(?:full|whole|entire)\s+)?"
    rf"(?P<unit>{_TIME_UNIT_RE}|frames?|fr)\b",
    re.IGNORECASE,
)
_ORDINAL_WORD_VALUES = {
    "first": 1.0,
    "second": 2.0,
    "third": 3.0,
    "fourth": 4.0,
    "fifth": 5.0,
    "sixth": 6.0,
    "seventh": 7.0,
    "eighth": 8.0,
    "ninth": 9.0,
    "tenth": 10.0,
    "eleventh": 11.0,
    "twelfth": 12.0,
    "thirteenth": 13.0,
    "fourteenth": 14.0,
    "fifteenth": 15.0,
    "sixteenth": 16.0,
    "seventeenth": 17.0,
    "eighteenth": 18.0,
    "nineteenth": 19.0,
    "twentieth": 20.0,
    "thirtieth": 30.0,
    "fortieth": 40.0,
    "fiftieth": 50.0,
    "sixtieth": 60.0,
    "seventieth": 70.0,
    "eightieth": 80.0,
    "ninetieth": 90.0,
}
for _ordinal_tens_word, _ordinal_tens_value in (
    ("twenty", 20),
    ("thirty", 30),
    ("forty", 40),
    ("fifty", 50),
    ("sixty", 60),
    ("seventy", 70),
    ("eighty", 80),
    ("ninety", 90),
):
    for _ordinal_unit_word, _ordinal_unit_value in tuple(
        _ORDINAL_WORD_VALUES.items()
    )[:9]:
        for _ordinal_join in ("-", " "):
            _ORDINAL_WORD_VALUES[
                _ordinal_tens_word + _ordinal_join + _ordinal_unit_word
            ] = float(_ordinal_tens_value) + _ordinal_unit_value
_WORD_ORDINAL_FRAME_REFERENCE_RE = re.compile(
    r"\b(?:the\s+)?(?P<value>"
    + "|".join(
        re.escape(item)
        for item in sorted(_ORDINAL_WORD_VALUES, key=len, reverse=True)
    )
    + r")\s+frames?\b",
    re.IGNORECASE,
)
_ORDINAL_WORD_PATTERN = "(?:" + "|".join(
    re.escape(item)
    for item in sorted(_ORDINAL_WORD_VALUES, key=len, reverse=True)
) + ")"
_WORD_FRAME_REFERENCE_RE = re.compile(
    r"\bframes?\s*(?:(?:number|index)\s*)?#?\s*"
    r"(?P<value>" + _WORD_NUMBER_VALUE_RE + r")\b",
    re.IGNORECASE,
)
_WORD_FRAME_ORDINAL_REFERENCE_RE = re.compile(
    r"\bframes?\s*(?:(?:number|index|indices|no\.?)\s*)?(?:#|:)?\s*"
    r"(?P<value>" + _ORDINAL_WORD_PATTERN + r")\b",
    re.IGNORECASE,
)
_FRAME_MARK_REFERENCE_RE = re.compile(
    r"\b(?P<value>"
    + _TEMPORAL_NUMBER_RE
    + r"|"
    + _WORD_NUMBER_VALUE_RE
    + r")\s*[- ]\s*frames?\s+(?:mark|index|point)\b",
    re.IGNORECASE,
)
_TIMESTEP_REFERENCE_RE = re.compile(
    r"\b(?:time\s*steps?|timesteps?)\s*(?:(?:number|index|no\.?)\s*)?"
    r"(?:#|:)?\s*(?P<value>"
    + _TEMPORAL_NUMBER_RE
    + r"|"
    + _WORD_NUMBER_VALUE_RE
    + r"|"
    + _ORDINAL_WORD_PATTERN
    + r")\b",
    re.IGNORECASE,
)
_DURATION_AFTER_FRAME_RE = re.compile(
    r"\b(?P<amount>"
    + _TEMPORAL_NUMBER_RE
    + r"|"
    + _WORD_NUMBER_VALUE_RE
    + r")\s*[-\u2013\u2014]?\s*"
    r"(?:(?:full|whole|entire)\s+)?"
    rf"(?P<unit>{_TIME_UNIT_RE}|frames?|fr)\s+after\s+(?:the\s+)?"
    r"(?:frames?|time\s*steps?|timesteps?)\s*"
    r"(?:(?:number|index|no\.?)\s*)?(?:#|:)?\s*"
    r"(?P<frame>"
    + _TEMPORAL_NUMBER_RE
    + r"|"
    + _WORD_NUMBER_VALUE_RE
    + r"|"
    + _ORDINAL_WORD_PATTERN
    + r")(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
_AMBIGUOUS_TEMPORAL_AMOUNT_RE = re.compile(
    r"\b(?:about|approximately|around|roughly|nearly|almost|at\s+least|"
    r"more\s+than|less\s+than|no\s+more\s+than|at\s+most|several|many|"
    r"multiple|some|a\s+few|a\s+couple\s+of)\s+"
    r"(?:(?:\d+(?:\.\d+)?)|"
    + _WORD_NUMBER_VALUE_RE
    + r")?\s*(?:(?:full|whole|entire)\s+)?"
    rf"(?:{_TIME_UNIT_RE}|frames?|fr)\b",
    re.IGNORECASE,
)
_NEGATIVE_TEMPORAL_AMOUNT_RE = re.compile(
    rf"(?:"
    rf"(?<![A-Za-z0-9_])-\s*{_TEMPORAL_NUMBER_RE}\s*"
    rf"(?:{_TIME_UNIT_RE}|frames?|fr)\b|"
    rf"\bminus\s+(?:{_TEMPORAL_NUMBER_RE}|{_WORD_NUMBER_VALUE_RE})\s*"
    rf"(?:{_TIME_UNIT_RE}|frames?|fr)\b|"
    rf"\bframes?\s*(?:(?:number|index)\s*)?#?\s*(?:-|minus\s+)\s*"
    rf"(?:{_TEMPORAL_NUMBER_RE}|{_WORD_NUMBER_VALUE_RE})\b"
    rf")",
    re.IGNORECASE,
)
_UNSUPPORTED_TEMPORAL_NOTATION_RE = re.compile(
    rf"(?:"
    rf"\b\d+\s+and\s+\d+\s*/\s*\d+|"
    rf"\b\d+\s+and\s+(?:a\s+)?(?:half|quarter)|"
    rf"\b(?:\d+|{_WORD_NUMBER_VALUE_RE})\s+and\s+"
    rf"(?:a|one|two|three)(?:\s+|[-\u2013\u2014])"
    rf"(?:half|quarters?)|"
    rf"\b(?:a|one|two|three)(?:\s+|[-\u2013\u2014])"
    rf"(?:half|quarters?)|"
    rf"\b(?:(?:{_NUMBER_WORD_ATOM_RE}|and|a|an)(?:\s+|[-\u2013\u2014]))*"
    rf"(?:thousand|million|billion)"
    rf"(?:(?:\s+|[-\u2013\u2014]|\s*,\s*)"
    rf"(?:{_NUMBER_WORD_ATOM_RE}|and|a|an))*|"
    rf"\b(?:\d+|{_WORD_NUMBER_VALUE_RE})\s+point\s+"
    rf"(?:\d+|{_WORD_NUMBER_VALUE_RE})"
    rf"(?:(?:\s+|[-\u2013\u2014])(?:\d+|{_WORD_NUMBER_VALUE_RE}))*|"
    rf"\b(?:\d+|{_WORD_NUMBER_VALUE_RE})\s+and\s+"
    rf"(?:a|an|one|two|three|four|five|six|seven|eight|nine)"
    rf"(?:\s+|[-\u2013\u2014])"
    rf"(?:thirds?|fourths?|fifths?|sixths?|sevenths?|eighths?|ninths?|tenths?)|"
    rf"\b(?:a|an|one|two|three|four|five|six|seven|eight|nine)"
    rf"(?:\s+|[-\u2013\u2014])"
    rf"(?:thirds?|fourths?|fifths?|sixths?|sevenths?|eighths?|ninths?|tenths?)|"
    rf"\b\d+(?:\.\d+)?\s*[kmb]|"
    rf"\b\d+\s*/\s*\d+|"
    rf"\b\d*[¼½¾]|"
    rf"\b\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|"
    rf"\b\d+(?:\.\d+)?e[+-]?\d+"
    rf")\s*(?:[-\u2013\u2014]\s*)?(?:{_TIME_UNIT_RE}|frames?|fr)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_TRAILING_FRACTION_RE = re.compile(
    rf"\b(?:{_TIME_UNIT_RE}|frames?|fr)\s+and\s+"
    rf"(?:a\s+)?(?:half|quarter)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_FRAME_REFERENCE_RE = re.compile(
    rf"\bframes?\s*(?:(?:number|index|indices|no\.?)\s*)?"
    rf"(?:#|:)?\s*\d+(?:\.\d+)?e[+-]?\d+\b",
    re.IGNORECASE,
)
_RELATIVE_DELAY_RE = re.compile(
    rf"(?:"
    rf"{_TEMPORAL_NUMBER_RE}\s*[-\u2013\u2014]?\s*"
    rf"(?:(?:full|whole|entire)\s+)?"
    rf"(?:{_TIME_UNIT_RE}|frames?|fr)|"
    rf"{_WORD_NUMBER_VALUE_RE}\s+"
    rf"(?:(?:full|whole|entire)\s+)?"
    rf"(?:{_TIME_UNIT_RE}|frames?|fr)"
    rf")\s+later\b",
    re.IGNORECASE,
)
_SEQUENTIAL_STAGE_RE = re.compile(
    r"\b(?:then|after\s+(?:that|which)|afterwards?|thereafter|"
    r"followed(?:\s+\S+){0,5}\s+(?:by|with)|prior\s+to|next|once|"
    r"subsequently|finally|"
    r"(?:before|after)(?=\s+(?:[A-Za-z]+ing|he\b|she\b|they\b|it\b)))\b",
    re.IGNORECASE,
)
_ORDERED_STAGE_CUE_RE = re.compile(
    r"\b(?:First|Second|Third|Fourth|firstly|secondly|thirdly|fourthly)\s*[,:]",
)
_LEADING_POSTPOSED_STAGE_RE = re.compile(
    r"^\s*(?:(?:immediately|right|only)\s+)?(?:after|before|once)\b"
    r"[^,;]+[,;]",
    re.IGNORECASE,
)
_LEADING_SEQUENTIAL_MODIFIER_RE = re.compile(
    r"^\s*(?:(?:immediately|right|only)\s+)?(?:after|before|once)\b",
    re.IGNORECASE,
)
_REPETITION_RE = re.compile(
    r"\b(?P<value>\d+(?:\.\d+)?|"
    + "|".join(_SMALL_NUMBER_WORDS)
    + r")\s+(?:times|repetitions?|cycles?)\b",
    re.IGNORECASE,
)
_CAMERA_REPLACE_RELATION_ALIAS_RE = re.compile(
    r"replace_"
    r"(?:(?:source_)?(?:camera_)?"
    r"(?:motion|trajectory|zoom|pan|tilt|dolly|truck|orbit))"
    r"(?:_with_(?:static|fixed|stationary|locked|locked_off|no_camera_motion))?"
    r"\Z"
)
_LOCKED_CAMERA_ASSERTION_RE = re.compile(
    r"\b(?:locked(?:\s+off)?|fixed|static|stationary)\b"
    r"|\bno\s+(?:(?:camera|viewpoint)\s+)(?:motion|movement)\b",
    re.IGNORECASE,
)
_LOCKED_CAMERA_CLOSED_TEMPLATE_RE = re.compile(
    r"(?:the\s+)?camera\s+"
    r"(?:remains?|stays?|is|keeps?)\s+"
    r"(?:completely\s+|fully\s+)?"
    r"(?:fixed|static|stationary|locked(?:\s+off)?)"
    r"(?:\s+at\s+(?:the\s+)?(?:initial|i0|first[- ]frame)\s+framing)?"
    r"(?:\s+with\s+no\s+(?:camera\s+(?:motion|movement)|"
    r"zoom\s+or\s+pan|pan\s+or\s+zoom))?"
    r"(?:\s+(?:throughout|for\s+the\s+entire\s+target\s+clip))?"
    r"(?:,\s+no\s+reframing\s+occurs)?"
    r"[.]?",
    re.IGNORECASE,
)
_CAMERA_MOTION_WORD_RE = (
    r"(?:pan(?:s|ned|ning)?|zoom(?:s|ed|ing)?|tilt(?:s|ed|ing)?|"
    r"doll(?:y|ies|ied|ying)|truck(?:s|ed|ing)?|"
    r"orbit(?:s|ed|ing)?|track(?:s|ed|ing)?|"
    r"crane(?:s|d|ing)?|drift(?:s|ed|ing)?|"
    r"shake(?:s|n|ing)?|shak(?:y|ily)|"
    r"refram(?:e|es|ed|ing)|roll(?:s|ed|ing)?|"
    r"push(?:es|ed|ing)?|pull(?:s|ed|ing)?|"
    r"slide(?:s|d)?|sliding|slid|sweep(?:s|ing)?|swept|"
    r"translat(?:e|es|ed|ing)|move(?:s|d)?|moving|"
    r"travel(?:s|ed|ing)?|travelling|shift(?:s|ed|ing)?|"
    r"arc(?:s|ed|ing)?|circl(?:e|es|ed|ing)|"
    r"rotat(?:e|es|ed|ing)|rise|rises|rose|rising|"
    r"descend(?:s|ed|ing)?|pedestal(?:s|ed|ing)?|handheld)"
)
_CAMERA_MOTION_TOKEN_RE = re.compile(
    rf"\b{_CAMERA_MOTION_WORD_RE}\b",
    re.IGNORECASE,
)
_NEGATED_CAMERA_MOTION_SERIES_RE = re.compile(
    r"\b(?:no|zero|without(?:\s+any)?)\s+"
    r"(?:(?:camera|viewpoint)\s+)?"
    rf"(?:motion|movement|{_CAMERA_MOTION_WORD_RE})"
    rf"(?:\s*(?:,|/|\band\b|\bor\b)\s*"
    rf"{_CAMERA_MOTION_WORD_RE})*",
    re.IGNORECASE,
)
_NEGATED_CAMERA_VERB_SERIES_RE = re.compile(
    r"\b(?:the\s+)?(?:camera|viewpoint|view)\s+"
    r"(?:does\s+not|doesn't|doesnt|never)\s+"
    r"(?:pan|zoom|tilt|dolly|truck|orbit|track|crane|drift|shake|"
    r"reframe|roll|push|pull|slide|sweep|translate|move|travel|shift|"
    r"arc|circle|rotate|rise|descend|pedestal)"
    r"(?:\s*(?:,|/|\band\b|\bor\b)\s*"
    r"(?:pan|zoom|tilt|dolly|truck|orbit|track|crane|drift|shake|"
    r"reframe|roll|push|pull|slide|sweep|translate|move|travel|shift|"
    r"arc|circle|rotate|rise|descend|pedestal))*",
    re.IGNORECASE,
)


class GokuFullMotionQwenV16Error(ValueError):
    """A row, model response, media binding, or publication is invalid."""


class GokuFullMotionQwenV16StageError(GokuFullMotionQwenV16Error):
    """A stage exhausted its sole local schema retry."""

    def __init__(self, stage: str, attempts: Sequence[Mapping[str, Any]]) -> None:
        self.stage = stage
        self.attempts = [dict(item) for item in attempts]
        message = str(self.attempts[-1].get("error") or "stage failed")
        super().__init__(f"{stage} failed: {message}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def object_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _digest_object_with_field(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    # Self-digests bind the closed schema with the digest slot explicitly
    # nulled.  Keeping the key makes the convention unambiguous to adapters.
    payload[field] = None
    return object_sha256(payload)


def _reject_constant(value: str) -> None:
    raise GokuFullMotionQwenV16Error(f"non-finite JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GokuFullMotionQwenV16Error(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _loads_object(raw: str, *, stage: str) -> tuple[dict[str, Any], list[str]]:
    """Parse one object while repairing only presentation-level JSON noise."""

    if not isinstance(raw, str) or not raw.strip():
        raise GokuFullMotionQwenV16Error(f"{stage} returned no text")
    text = raw.lstrip("\ufeff").strip()
    repairs: list[str] = []
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            raise GokuFullMotionQwenV16Error(
                f"{stage} has an unterminated JSON fence"
            )
        text = "\n".join(lines[1:-1]).strip()
        if text.casefold().startswith("json"):
            text = text[4:].lstrip()
        repairs.append("removed_markdown_json_fence")
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, TypeError) as error:
        raise GokuFullMotionQwenV16Error(
            f"{stage} is not a single strict JSON object: {error}"
        ) from error
    if not isinstance(value, dict):
        raise GokuFullMotionQwenV16Error(f"{stage} must be a JSON object")
    return value, repairs


def _require_closed(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    if set(value) != expected:
        raise GokuFullMotionQwenV16Error(
            f"{context} keys differ from closed schema: "
            f"{sorted(set(value) ^ expected)}"
        )


def _text(
    value: Any,
    *,
    context: str,
    minimum: int = 2,
    maximum: int = 700,
    allow_semicolon: bool = False,
) -> str:
    if not isinstance(value, str):
        raise GokuFullMotionQwenV16Error(f"{context} must be a string")
    normalized = " ".join(value.split())
    if not minimum <= len(normalized) <= maximum:
        raise GokuFullMotionQwenV16Error(
            f"{context} length must be in [{minimum}, {maximum}]"
        )
    if "\n" in value or (not allow_semicolon and ";" in normalized):
        raise GokuFullMotionQwenV16Error(
            f"{context} contains a clause delimiter"
        )
    if _PLACEHOLDER_RE.search(normalized):
        raise GokuFullMotionQwenV16Error(f"{context} contains placeholder text")
    return normalized


def _reject_source_sample_markers(value: str, *, context: str) -> None:
    for marker in _SAMPLING_FRAME_TOKEN_RE.finditer(value):
        compact = re.sub(r"[-_\s]", "", marker.group(0)).casefold()
        if compact == "f1" and re.match(
            r"\s+(?:race|racing|car|vehicle|driver|team)\b",
            value[marker.end() :],
            re.IGNORECASE,
        ) is not None:
            # F1 is a real entity label in racing footage, not a sparse-view
            # frame code when immediately qualified as such.
            continue
        raise GokuFullMotionQwenV16Error(
            f"{context} contains source-view sample marker {marker.group(0)!r}; "
            "sparse-view labels cannot appear in an executable target instruction"
        )


def _time_value_seconds(value: float, unit: str) -> float:
    normalized = unit.casefold()
    if normalized.startswith(("millisecond", "msec")) or normalized == "ms":
        return value / 1000.0
    if normalized.startswith(("minute", "min")):
        return value * 60.0
    if normalized.startswith(("hour", "hr")):
        return value * 3600.0
    return value


def _number_text_value(value: str) -> float:
    normalized = re.sub(r"[-\u2013\u2014]", " ", value.casefold()).strip()
    if re.fullmatch(_TEMPORAL_NUMBER_RE, normalized) is not None:
        return float(normalized)
    ordinal_key = "-".join(normalized.split())
    if ordinal_key in _ORDINAL_WORD_VALUES:
        return _ORDINAL_WORD_VALUES[ordinal_key]
    tokens = [
        token
        for token in normalized.split()
        if token not in {"and", "a", "an"}
    ]
    if not tokens:
        return 1.0
    value = 0.0
    for token in tokens:
        if token == "half":
            value += 0.5
        elif token == "hundred":
            value = max(value, 1.0) * 100.0
        else:
            value += _SMALL_NUMBER_WORDS[token]
    return value


def _word_amount_value(match: re.Match[str]) -> float:
    return _number_text_value(match.group("value"))


def _is_absolute_time_anchor(text: str, start: int) -> bool:
    """Return whether an amount is an absolute time/deadline, not a duration."""

    prefix = text[max(0, start - 48) : start]
    return re.search(
        r"(?:\b(?:at|by|before|until|within|from|to)\s+"
        r"(?:(?:exactly|the)\s+){0,2}|"
        r"\b(?:no|not)\s+later\s+than\s+(?:the\s+)?)$",
        prefix,
        re.IGNORECASE,
    ) is not None


def _is_explicit_frame_duration(text: str, start: int) -> bool:
    if _is_absolute_time_anchor(text, start):
        return False
    prefix = text[max(0, start - 32) : start]
    if re.search(
        r"\b(?:for|over|during|after|in|lasting|lasts|takes|taking|"
        r"spanning|spans)\s+$",
        prefix,
        re.IGNORECASE,
    ) is not None:
        return True
    # The amount match itself is passed by start only, so inspect a short
    # suffix from the first unit boundary for "N frames later".
    suffix = text[start : start + 96]
    return re.search(r"\b(?:frames?|fr)\s+later\b", suffix, re.IGNORECASE) is not None


def _temporal_stage_index(
    boundaries: Sequence[int], position: int
) -> int:
    return sum(boundary <= position for boundary in boundaries)


def _validate_target_timeline_text(
    value: str, *, context: str, check_sequence: bool = True
) -> None:
    """Mechanically prove that one target trajectory fits frames 0..80.

    Subjects and camera are checked independently because they execute
    concurrently.  Inside one trajectory, only explicit sequential connectors
    make stages cumulative; simultaneous amounts in one stage use the longest
    duration.  Ambiguous quantities fail instead of being guessed or shortened.
    This validator never rewrites ``value``.
    """

    _reject_source_sample_markers(value, context=context)
    negative = _NEGATIVE_TEMPORAL_AMOUNT_RE.search(value)
    if negative is not None:
        raise GokuFullMotionQwenV16Error(
            f"{context} contains invalid negative temporal amount "
            f"{negative.group(0)!r}"
        )
    unsupported = (
        _UNSUPPORTED_TEMPORAL_NOTATION_RE.search(value)
        or _UNSUPPORTED_TRAILING_FRACTION_RE.search(value)
        or _UNSUPPORTED_FRAME_REFERENCE_RE.search(value)
    )
    if unsupported is not None:
        raise GokuFullMotionQwenV16Error(
            f"{context} contains unsupported temporal notation "
            f"{unsupported.group(0)!r}; it cannot be safely normalized"
        )
    ambiguous = _AMBIGUOUS_TEMPORAL_AMOUNT_RE.search(value)
    if ambiguous is not None:
        raise GokuFullMotionQwenV16Error(
            f"{context} contains ambiguous temporal amount "
            f"{ambiguous.group(0)!r}; target timing must be mechanically bounded"
        )

    # (character position, seconds, is_absolute_anchor).  Whole compiled
    # instructions disable sequence accounting because subject clauses run in
    # parallel; every closed subject/camera clause is checked separately.
    events: list[tuple[int, float, bool]] = []
    time_matches: list[tuple[re.Match[str], float]] = [
        (match, float(match.group("value")))
        for match in _TIME_AMOUNT_RE.finditer(value)
    ]
    time_matches.extend(
        (match, _word_amount_value(match))
        for match in _WORD_AMOUNT_RE.finditer(value)
        if not match.group("unit").casefold().startswith("frame")
    )
    time_matches.sort(key=lambda item: item[0].start())
    duration_records: list[tuple[re.Match[str], float]] = []
    for match, amount in time_matches:
        seconds = _time_value_seconds(amount, match.group("unit"))
        if not math.isfinite(seconds) or seconds < 0:
            raise GokuFullMotionQwenV16Error(
                f"{context} contains invalid temporal amount {match.group(0)!r}"
            )
        if seconds > TIMELINE_SPAN_SECONDS + 1e-9:
            raise GokuFullMotionQwenV16Error(
                f"{context} temporal amount {match.group(0)!r} exceeds the "
                f"{TIMELINE_SPAN_SECONDS:g}-second target timeline"
            )
        is_anchor = _is_absolute_time_anchor(value, match.start())
        events.append((match.start(), seconds, is_anchor))
        if not is_anchor:
            duration_records.append((match, seconds))

    additive_clusters: list[list[tuple[re.Match[str], float]]] = []
    for record in duration_records:
        if not additive_clusters:
            additive_clusters.append([record])
            continue
        previous = additive_clusters[-1][-1][0]
        if re.fullmatch(
            r"\s*,?\s*and\s+",
            value[previous.end() : record[0].start()],
            re.IGNORECASE,
        ) is not None:
            additive_clusters[-1].append(record)
        else:
            additive_clusters.append([record])
    for cluster in additive_clusters:
        if len(cluster) < 2:
            continue
        seconds = sum(record[1] for record in cluster)
        if seconds > TIMELINE_SPAN_SECONDS + 1e-9:
            rendered = value[cluster[0][0].start() : cluster[-1][0].end()]
            raise GokuFullMotionQwenV16Error(
                f"{context} additive duration {rendered!r} exceeds the "
                f"{TIMELINE_SPAN_SECONDS:g}-second target timeline"
            )
        events.append((cluster[0][0].start(), seconds, False))

    frame_amount_matches: list[tuple[re.Match[str], float]] = [
        (match, float(match.group("value")))
        for match in _FRAME_DURATION_RE.finditer(value)
    ]
    frame_amount_matches.extend(
        (match, _word_amount_value(match))
        for match in _WORD_AMOUNT_RE.finditer(value)
        if match.group("unit").casefold().startswith("frame")
    )
    for match, amount in frame_amount_matches:
        if not amount.is_integer() or amount <= 0:
            raise GokuFullMotionQwenV16Error(
                f"{context} frame amount {match.group(0)!r} must be a positive integer"
            )
        if amount > FRAME_COUNT:
            raise GokuFullMotionQwenV16Error(
                f"{context} frame amount {match.group(0)!r} exceeds the "
                f"{FRAME_COUNT}-frame clip"
            )
        if _is_explicit_frame_duration(value, match.start()):
            seconds = amount / FPS
            if seconds > TIMELINE_SPAN_SECONDS + 1e-9:
                raise GokuFullMotionQwenV16Error(
                    f"{context} frame duration {match.group(0)!r} exceeds the "
                    f"{MAX_TIMELINE_FRAME_INDEX}-frame target timeline "
                    f"({TIMELINE_SPAN_SECONDS:g} seconds)"
                )
            events.append((match.start(), seconds, False))

    frame_references: list[tuple[str, float, int]] = [
        (match.group(0), float(match.group("value")), match.start())
        for match in _FRAME_REFERENCE_RE.finditer(value)
    ]
    frame_references.extend(
        (match.group(0), float(match.group("value")), match.start())
        for match in _ORDINAL_FRAME_REFERENCE_RE.finditer(value)
    )
    frame_references.extend(
        (
            match.group(0),
            _ORDINAL_WORD_VALUES[
                "-".join(match.group("value").casefold().split())
            ],
            match.start(),
        )
        for match in _WORD_ORDINAL_FRAME_REFERENCE_RE.finditer(value)
    )
    frame_references.extend(
        (match.group(0), _word_amount_value(match), match.start())
        for match in _WORD_FRAME_REFERENCE_RE.finditer(value)
    )
    frame_references.extend(
        (match.group(0), _word_amount_value(match), match.start())
        for match in _WORD_FRAME_ORDINAL_REFERENCE_RE.finditer(value)
    )
    for match in _FRAME_MARK_REFERENCE_RE.finditer(value):
        raw_frame = match.group("value")
        frame = (
            float(raw_frame)
            if re.fullmatch(_TEMPORAL_NUMBER_RE, raw_frame) is not None
            else _word_amount_value(match)
        )
        frame_references.append((match.group(0), frame, match.start()))
    for match in _FRAME_INDEX_LIST_RE.finditer(value):
        frame_references.extend(
            (match.group(0), float(rendered), match.start())
            for rendered in re.findall(r"\d+", match.group("values"))
        )
    frame_references.extend(
        (
            match.group(0),
            _number_text_value(match.group("value")),
            match.start(),
        )
        for match in _TIMESTEP_REFERENCE_RE.finditer(value)
    )
    for match in _FRAME_RANGE_RE.finditer(value):
        start_frame = float(match.group("start"))
        end_frame = float(match.group("end"))
        if start_frame > end_frame:
            raise GokuFullMotionQwenV16Error(
                f"{context} frame range {match.group(0)!r} is reversed"
            )
        frame_references.extend(
            (
                (match.group(0), start_frame, match.start()),
                (match.group(0), end_frame, match.start()),
            )
        )
    for rendered, frame, position in frame_references:
        if not frame.is_integer() or not 0 <= frame <= MAX_TIMELINE_FRAME_INDEX:
            raise GokuFullMotionQwenV16Error(
                f"{context} frame reference {rendered!r} is outside integer "
                f"frame indices 0..{MAX_TIMELINE_FRAME_INDEX}"
            )
        events.append((position, frame / FPS, True))

    for delayed in _DURATION_AFTER_FRAME_RE.finditer(value):
        amount = _number_text_value(delayed.group("amount"))
        frame = _number_text_value(delayed.group("frame"))
        unit = delayed.group("unit")
        duration_seconds = (
            amount / FPS
            if unit.casefold().startswith(("frame", "fr"))
            else _time_value_seconds(amount, unit)
        )
        if (
            amount <= 0
            or not frame.is_integer()
            or not 0 <= frame <= MAX_TIMELINE_FRAME_INDEX
            or frame / FPS + duration_seconds
            > TIMELINE_SPAN_SECONDS + 1e-9
        ):
            raise GokuFullMotionQwenV16Error(
                f"{context} delayed motion {delayed.group(0)!r} exceeds the "
                f"{TIMELINE_SPAN_SECONDS:g}-second target timeline"
            )

    for repetition in _REPETITION_RE.finditer(value):
        token = repetition.group("value").casefold()
        amount = (
            float(token)
            if re.fullmatch(_TEMPORAL_NUMBER_RE, token) is not None
            else _SMALL_NUMBER_WORDS[token]
        )
        if not amount.is_integer() or amount > 6:
            raise GokuFullMotionQwenV16Error(
                f"{context} repetition count {repetition.group(0)!r} is not "
                "moderate enough for the 3.2-second target timeline"
            )

    if not check_sequence:
        return
    raw_connectors = list(_SEQUENTIAL_STAGE_RE.finditer(value))
    connectors: list[tuple[int, int]] = []
    for connector in raw_connectors:
        span = (connector.start(), connector.end())
        if connectors and re.fullmatch(
            r"[\s,]*(?:and[\s,]*)?",
            value[connectors[-1][1] : span[0]],
            re.IGNORECASE,
        ) is not None:
            # "then finally" is one transition, not an empty extra stage.
            connectors[-1] = (connectors[-1][0], span[1])
        else:
            connectors.append(span)

    relative_delays = list(_RELATIVE_DELAY_RE.finditer(value))
    wrapped_connector_ends: set[int] = set()
    for delay in relative_delays:
        # A connector such as "followed two seconds later by" wraps the delay;
        # replace that one boundary with the delay stage's start and end.
        wrapped_connector_ends.update(
            end
            for start, end in connectors
            if start <= delay.start() and end >= delay.end()
        )
    boundaries = [
        end for _, end in connectors if end not in wrapped_connector_ends
    ]
    for delay in relative_delays:
        boundaries.extend((delay.start(), delay.end()))
    ordered_cues = list(_ORDERED_STAGE_CUE_RE.finditer(value))
    if ordered_cues:
        first_nonspace = len(value) - len(value.lstrip())
        boundaries.extend(
            cue.end()
            for cue_index, cue in enumerate(ordered_cues)
            if not (cue_index == 0 and cue.start() == first_nonspace)
        )
    boundaries = sorted(set(boundaries))
    leading_postposed = _LEADING_POSTPOSED_STAGE_RE.match(value)
    if leading_postposed is not None:
        # In "After waving ..., clap ...", the comma—not the leading word
        # "after"—separates the two chronological stages.
        if connectors and connectors[0][0] < leading_postposed.end():
            if connectors[0][1] in boundaries:
                boundaries.remove(connectors[0][1])
        boundaries.append(leading_postposed.end())
        boundaries = sorted(set(boundaries))
    elif _LEADING_SEQUENTIAL_MODIFIER_RE.match(value) is not None:
        # Fail closed for an unpunctuated leading subordinate clause.  Its
        # second explicit temporal event is necessarily in the main clause.
        if connectors and connectors[0][0] < 32:
            if connectors[0][1] in boundaries:
                boundaries.remove(connectors[0][1])
        event_positions = sorted(set(position for position, _, _ in events))
        if len(event_positions) >= 2:
            boundaries.append(event_positions[1])
            boundaries = sorted(set(boundaries))
    if len(boundaries) >= 3:
        raise GokuFullMotionQwenV16Error(
            f"{context} has too many sequential stages for a moderate "
            "3.2-second target motion"
        )
    if not boundaries:
        return

    stage_durations = [0.0] * (len(boundaries) + 1)
    duration_events: list[list[tuple[int, float]]] = [
        [] for _ in range(len(boundaries) + 1)
    ]
    anchors: list[tuple[int, int, float]] = []
    for position, seconds, is_anchor in events:
        stage = _temporal_stage_index(boundaries, position)
        if is_anchor:
            anchors.append((stage, position, seconds))
        else:
            # Multiple timed motions in one stage may be simultaneous (for
            # example, waving while smiling).  Explicit stages remain serial.
            stage_durations[stage] = max(stage_durations[stage], seconds)
            duration_events[stage].append((position, seconds))

    required_seconds = sum(stage_durations)
    if anchors:
        required_seconds = max(required_seconds, max(item[2] for item in anchors))
        required_seconds = max(
            required_seconds,
            max(
                anchor_seconds
                + max(
                    (
                        seconds
                        for position, seconds in duration_events[anchor_stage]
                        if position > anchor_position
                    ),
                    default=0.0,
                )
                + sum(stage_durations[anchor_stage + 1 :])
                for anchor_stage, anchor_position, anchor_seconds in anchors
            ),
        )
    if required_seconds > TIMELINE_SPAN_SECONDS + 1e-9:
        raise GokuFullMotionQwenV16Error(
            f"{context} explicit stages require at least "
            f"{required_seconds:g} seconds, exceeding the "
            f"{TIMELINE_SPAN_SECONDS:g}-second target timeline; cumulative "
            "sequence is infeasible"
        )


def _signature(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SIGNATURE_RE.fullmatch(value) is None:
        raise GokuFullMotionQwenV16Error(
            f"{context} must be lower_snake_case"
        )
    return value


def _slug(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    value = "_".join(tokens[:10]).strip("_")
    if not value or not value[0].isalpha():
        value = "motion_" + value
    if len(value) < 2:
        value += "_action"
    return value[:96].rstrip("_")


def _repair_alias(
    value: dict[str, Any], old: str, new: str, repairs: list[str], *, context: str
) -> None:
    if old not in value:
        return
    if new in value:
        raise GokuFullMotionQwenV16Error(
            f"{context} supplies both {old!r} and {new!r}"
        )
    value[new] = value.pop(old)
    repairs.append(f"renamed_{context}.{old}_to_{new}")


def _insert(
    value: dict[str, Any], key: str, item: Any, repairs: list[str], *, context: str
) -> None:
    if key not in value:
        value[key] = item
        repairs.append(f"inserted_{context}.{key}")


def _normalize_natural_language_leaf(
    value: dict[str, Any],
    key: str,
    repairs: list[str],
    *,
    context: str,
) -> None:
    """Replace model-authored clause separators without changing semantics."""

    raw = value.get(key)
    if not isinstance(raw, str):
        return
    normalized = re.sub(r"[;；]+", ",", raw)
    if normalized != raw:
        value[key] = normalized
        repairs.append(f"normalized_{context}.{key}_semicolon_to_comma")


def _normalize_camera_class(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "static": "locked_off",
        "fixed": "locked_off",
        "stationary": "locked_off",
        "no_camera_motion": "locked_off",
        "locked": "locked_off",
    }
    return aliases.get(normalized, normalized)


def _has_explicit_locked_camera_assertion(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = " ".join(value.split())
    negated = re.search(
        r"\b(?:not|never|isn't|isnt)\s+"
        r"(?:completely\s+|fully\s+)?"
        r"(?:fixed|static|stationary|locked(?:\s+off)?)\b",
        text,
        re.IGNORECASE,
    )
    return negated is None and _LOCKED_CAMERA_ASSERTION_RE.search(text) is not None


def _has_affirmative_camera_motion(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = " ".join(value.split())
    scrubbed = _NEGATED_CAMERA_MOTION_SERIES_RE.sub(" ", text)
    scrubbed = _NEGATED_CAMERA_VERB_SERIES_RE.sub(" ", scrubbed)
    return _CAMERA_MOTION_TOKEN_RE.search(scrubbed) is not None


def _explicit_locked_camera_target(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = " ".join(value.split())
    return _LOCKED_CAMERA_CLOSED_TEMPLATE_RE.fullmatch(text) is not None


def _canonicalize_camera_relation(
    camera: dict[str, Any],
    *,
    source_census: Mapping[str, Any],
    repairs: list[str],
) -> None:
    raw_relation = camera.get("relation")
    if isinstance(raw_relation, str):
        token = raw_relation.strip().casefold().replace("-", "_").replace(" ", "_")
        if token in {"preserve_static", "replace_motion"}:
            if token != raw_relation:
                camera["relation"] = token
                repairs.append("normalized_camera.relation_enum_spelling")
        elif _CAMERA_REPLACE_RELATION_ALIAS_RE.fullmatch(token) is not None:
            camera["relation"] = "replace_motion"
            repairs.append(
                "normalized_camera.relation_replace_alias_to_replace_motion"
            )

    source_camera = source_census.get("camera")
    source_class = (
        source_camera.get("motion_class")
        if isinstance(source_camera, Mapping)
        else None
    )
    if (
        source_class != "locked_off"
        and camera.get("relation") == "preserve_static"
        and camera.get("motion_class") == "locked_off"
        and _explicit_locked_camera_target(camera.get("target_motion"))
    ):
        # The target class and literal target prose already specify the
        # replacement.  Correct only the contradictory enum; never invent or
        # rewrite a camera trajectory here.
        camera["relation"] = "replace_motion"
        repairs.append(
            "normalized_camera.relation_preserve_static_to_replace_motion_"
            "from_explicit_locked_target"
        )


def _normalize_entity_type(value: Any) -> Any:
    """Map literal species/object labels to the closed ontology parent.

    Qwen often emits an accurate visual noun (``dog``, ``bird``, ``boat``)
    where the contract asks for its semantic parent (``animal``, ``vehicle``).
    This is an ontology-only canonicalization: the subject, reference, bbox,
    motion, and evidence are unchanged.
    """

    if not isinstance(value, str):
        return value
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "human": "person",
        "man": "person",
        "woman": "person",
        "boy": "person",
        "girl": "person",
        "adult": "person",
        "child": "person",
        "dog": "animal",
        "puppy": "animal",
        "cat": "animal",
        "kitten": "animal",
        "bird": "animal",
        "bear": "animal",
        "horse": "animal",
        "car": "vehicle",
        "truck": "vehicle",
        "bus": "vehicle",
        "boat": "vehicle",
        "ship": "vehicle",
        "airplane": "vehicle",
        "plane": "vehicle",
        "bicycle": "vehicle",
        "motorcycle": "vehicle",
        "water": "fluid_or_emitter",
        "wave": "fluid_or_emitter",
        "waves": "fluid_or_emitter",
        "wake": "fluid_or_emitter",
        "cloud": "fluid_or_emitter",
        "clouds": "fluid_or_emitter",
        "smoke": "fluid_or_emitter",
        "steam": "fluid_or_emitter",
        "fire": "fluid_or_emitter",
        "flame": "fluid_or_emitter",
        "flames": "fluid_or_emitter",
    }
    return aliases.get(normalized, normalized)


def canonicalize_source_census(
    raw_value: Mapping[str, Any], *, expected_iid: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Insert only redundant structure; never invent a subject or evidence."""

    value = copy.deepcopy(dict(raw_value))
    repairs: list[str] = []
    _repair_alias(value, "subjects", "dynamic_subjects", repairs, context="census")
    _repair_alias(value, "camera_motion", "camera", repairs, context="census")
    _insert(value, "schema_version", SOURCE_CENSUS_SCHEMA, repairs, context="census")
    _insert(value, "iid", expected_iid, repairs, context="census")
    subjects = value.get("dynamic_subjects")
    if isinstance(subjects, list):
        for index, item in enumerate(subjects, start=1):
            if not isinstance(item, dict):
                continue
            context = f"subject_{index:02d}"
            _repair_alias(item, "id", "subject_id", repairs, context=context)
            _repair_alias(
                item, "bbox", "i0_bbox_xyxy_1000", repairs, context=context
            )
            _repair_alias(
                item, "action_signature", "source_action_signature", repairs,
                context=context,
            )
            _repair_alias(
                item, "motion", "source_motion", repairs, context=context
            )
            _repair_alias(
                item, "evidence", "motion_evidence", repairs, context=context
            )
            _insert(
                item, "schema_version", SOURCE_SUBJECT_SCHEMA, repairs,
                context=context,
            )
            if "entity_type" in item:
                normalized_entity_type = _normalize_entity_type(item["entity_type"])
                if normalized_entity_type != item["entity_type"]:
                    item["entity_type"] = normalized_entity_type
                    repairs.append(f"normalized_{context}.entity_type")
            _insert(
                item, "subject_id", f"subject_{index:02d}", repairs,
                context=context,
            )
            _insert(item, "dynamic", True, repairs, context=context)
            for leaf in ("stable_reference", "i0_state", "source_motion"):
                _normalize_natural_language_leaf(
                    item, leaf, repairs, context=context
                )
            if "source_action_signature" not in item and isinstance(
                item.get("source_motion"), str
            ):
                _insert(
                    item,
                    "source_action_signature",
                    _slug(item["source_motion"]),
                    repairs,
                    context=context,
                )
            evidence = item.get("motion_evidence")
            if isinstance(evidence, list):
                for evidence_index, evidence_item in enumerate(evidence, start=1):
                    if isinstance(evidence_item, dict):
                        _normalize_natural_language_leaf(
                            evidence_item,
                            "description",
                            repairs,
                            context=f"{context}.evidence_{evidence_index:02d}",
                        )
                        _insert(
                            evidence_item,
                            "schema_version",
                            MOTION_EVIDENCE_SCHEMA,
                            repairs,
                            context=f"{context}.evidence_{evidence_index:02d}",
                        )
    camera = value.get("camera")
    if isinstance(camera, dict):
        _repair_alias(
            camera, "description", "source_motion", repairs, context="camera"
        )
        _repair_alias(camera, "evidence", "motion_evidence", repairs, context="camera")
        _insert(camera, "schema_version", SOURCE_CAMERA_SCHEMA, repairs, context="camera")
        _normalize_natural_language_leaf(
            camera, "source_motion", repairs, context="camera"
        )
        if "motion_class" in camera:
            normalized = _normalize_camera_class(camera["motion_class"])
            if normalized != camera["motion_class"]:
                camera["motion_class"] = normalized
                repairs.append("normalized_camera.motion_class")
        evidence = camera.get("motion_evidence")
        if isinstance(evidence, list):
            for index, item in enumerate(evidence, start=1):
                if isinstance(item, dict):
                    _normalize_natural_language_leaf(
                        item,
                        "description",
                        repairs,
                        context=f"camera.evidence_{index:02d}",
                    )
                    _insert(
                        item,
                        "schema_version",
                        MOTION_EVIDENCE_SCHEMA,
                        repairs,
                        context=f"camera.evidence_{index:02d}",
                    )
    # These two fields are semantic assertions, not redundant formatting.
    # Their absence must fail validation rather than being repaired to a pass.
    receipt = {
        "schema_version": MECHANICAL_REPAIR_SCHEMA,
        "stage": "source_census",
        "operations": repairs,
        "semantic_fields_invented": False,
    }
    return value, receipt


def _validate_evidence(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GokuFullMotionQwenV16Error(f"{context} must be an object")
    item = dict(value)
    _require_closed(
        item,
        {"schema_version", "start_frame", "end_frame", "description"},
        context=context,
    )
    if item["schema_version"] != MOTION_EVIDENCE_SCHEMA:
        raise GokuFullMotionQwenV16Error(f"{context} schema differs")
    start, end = item["start_frame"], item["end_frame"]
    if type(start) is not int or type(end) is not int or not (
        0 <= start < end < FRAME_COUNT
    ):
        raise GokuFullMotionQwenV16Error(f"{context} frame span is invalid")
    item["description"] = _text(
        item["description"], context=f"{context}.description", maximum=500
    )
    return item


def validate_source_census(
    value: Mapping[str, Any], *, expected_iid: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GokuFullMotionQwenV16Error("source census must be an object")
    census = copy.deepcopy(dict(value))
    _require_closed(
        census,
        {
            "schema_version",
            "iid",
            "dynamic_subjects",
            "camera",
            "all_dynamic_subjects_enumerated",
            "crowd_or_unresolved_motion",
            "confidence",
        },
        context="source_census",
    )
    if census["schema_version"] != SOURCE_CENSUS_SCHEMA:
        raise GokuFullMotionQwenV16Error("source census schema differs")
    if census["iid"] != expected_iid:
        raise GokuFullMotionQwenV16Error("source census IID differs")
    subjects = census["dynamic_subjects"]
    if not isinstance(subjects, list) or not 1 <= len(subjects) <= MAX_DYNAMIC_SUBJECTS:
        raise GokuFullMotionQwenV16Error(
            f"source census requires 1..{MAX_DYNAMIC_SUBJECTS} dynamic subjects"
        )
    references: set[str] = set()
    for index, raw_subject in enumerate(subjects, start=1):
        context = f"source_census.dynamic_subjects[{index - 1}]"
        if not isinstance(raw_subject, Mapping):
            raise GokuFullMotionQwenV16Error(f"{context} must be an object")
        subject = dict(raw_subject)
        _require_closed(
            subject,
            {
                "schema_version",
                "subject_id",
                "entity_type",
                "stable_reference",
                "i0_bbox_xyxy_1000",
                "i0_state",
                "source_action_signature",
                "source_motion",
                "motion_evidence",
                "dynamic",
            },
            context=context,
        )
        expected_id = f"subject_{index:02d}"
        if subject["schema_version"] != SOURCE_SUBJECT_SCHEMA:
            raise GokuFullMotionQwenV16Error(f"{context} schema differs")
        if subject["subject_id"] != expected_id:
            raise GokuFullMotionQwenV16Error(
                f"{context}.subject_id must be {expected_id}"
            )
        if subject["entity_type"] not in ENTITY_TYPES - {"coherent_group"}:
            raise GokuFullMotionQwenV16Error(f"{context}.entity_type is invalid")
        reference = _text(
            subject["stable_reference"],
            context=f"{context}.stable_reference",
            maximum=256,
        )
        reference_key = reference.casefold()
        if reference_key in references:
            raise GokuFullMotionQwenV16Error(
                "source census stable references are not unique"
            )
        references.add(reference_key)
        subject["stable_reference"] = reference
        bbox = subject["i0_bbox_xyxy_1000"]
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(type(item) is not int for item in bbox)
            or not (0 <= bbox[0] < bbox[2] <= 1000)
            or not (0 <= bbox[1] < bbox[3] <= 1000)
        ):
            raise GokuFullMotionQwenV16Error(f"{context}.bbox is invalid")
        subject["i0_state"] = _text(
            subject["i0_state"], context=f"{context}.i0_state", maximum=500
        )
        subject["source_action_signature"] = _signature(
            subject["source_action_signature"],
            context=f"{context}.source_action_signature",
        )
        subject["source_motion"] = _text(
            subject["source_motion"], context=f"{context}.source_motion"
        )
        evidence = subject["motion_evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise GokuFullMotionQwenV16Error(
                f"{context}.motion_evidence must be non-empty"
            )
        subject["motion_evidence"] = [
            _validate_evidence(item, context=f"{context}.motion_evidence[{j}]")
            for j, item in enumerate(evidence)
        ]
        if subject["dynamic"] is not True:
            raise GokuFullMotionQwenV16Error(f"{context} is not dynamic")
        subjects[index - 1] = subject
    if census["all_dynamic_subjects_enumerated"] is not True:
        raise GokuFullMotionQwenV16Error(
            "source census does not assert complete dynamic-subject coverage"
        )
    if census["crowd_or_unresolved_motion"] is not False:
        raise GokuFullMotionQwenV16Error(
            "source census contains crowd or unresolved independent motion"
        )
    camera_raw = census["camera"]
    if not isinstance(camera_raw, Mapping):
        raise GokuFullMotionQwenV16Error("source camera must be an object")
    camera = dict(camera_raw)
    _require_closed(
        camera,
        {"schema_version", "motion_class", "source_motion", "motion_evidence"},
        context="source_census.camera",
    )
    if camera["schema_version"] != SOURCE_CAMERA_SCHEMA:
        raise GokuFullMotionQwenV16Error("source camera schema differs")
    if camera["motion_class"] not in CAMERA_MOTION_CLASSES:
        raise GokuFullMotionQwenV16Error("source camera motion class is invalid")
    camera["source_motion"] = _text(
        camera["source_motion"], context="source_census.camera.source_motion"
    )
    camera_evidence = camera["motion_evidence"]
    if not isinstance(camera_evidence, list) or not camera_evidence:
        raise GokuFullMotionQwenV16Error("source camera evidence must be non-empty")
    camera["motion_evidence"] = [
        _validate_evidence(item, context=f"source_census.camera.evidence[{j}]")
        for j, item in enumerate(camera_evidence)
    ]
    if census["confidence"] not in {"high", "medium"}:
        raise GokuFullMotionQwenV16Error("source census confidence is too low")
    census["dynamic_subjects"] = subjects
    census["camera"] = camera
    return census


def canonicalize_target_plan(
    raw_value: Mapping[str, Any],
    *,
    expected_iid: str,
    source_census: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = copy.deepcopy(dict(raw_value))
    repairs: list[str] = []
    _repair_alias(
        value, "targets", "dynamic_subject_targets", repairs, context="plan"
    )
    _repair_alias(value, "camera", "camera_target", repairs, context="plan")
    _insert(value, "schema_version", TARGET_PLAN_SCHEMA, repairs, context="plan")
    _insert(value, "iid", expected_iid, repairs, context="plan")
    targets = value.get("dynamic_subject_targets")
    source_ids = [item["subject_id"] for item in source_census["dynamic_subjects"]]
    if isinstance(targets, list):
        for index, item in enumerate(targets, start=1):
            if not isinstance(item, dict):
                continue
            context = f"target_{index:02d}"
            _repair_alias(item, "id", "subject_id", repairs, context=context)
            _repair_alias(
                item, "action_signature", "target_action_signature", repairs,
                context=context,
            )
            _repair_alias(item, "motion", "target_motion", repairs, context=context)
            _insert(
                item, "schema_version", TARGET_SUBJECT_SCHEMA, repairs,
                context=context,
            )
            _normalize_natural_language_leaf(
                item, "target_motion", repairs, context=context
            )
            if len(targets) == len(source_ids):
                _insert(
                    item, "subject_id", source_ids[index - 1], repairs,
                    context=context,
                )
            # ``substantive_change`` is a semantic assertion and is required
            # verbatim; it is never filled by the mechanical repair layer.
            if "target_action_signature" not in item and isinstance(
                item.get("target_motion"), str
            ):
                _insert(
                    item,
                    "target_action_signature",
                    _slug(item["target_motion"]),
                    repairs,
                    context=context,
                )
        if all(isinstance(item, Mapping) and "subject_id" in item for item in targets):
            by_id = {str(item["subject_id"]): item for item in targets}
            if len(by_id) == len(targets) and set(by_id) == set(source_ids):
                ordered = [by_id[subject_id] for subject_id in source_ids]
                if ordered != targets:
                    value["dynamic_subject_targets"] = ordered
                    targets = ordered
                    repairs.append("ordered_plan.dynamic_subject_targets_by_source")
    camera = value.get("camera_target")
    if isinstance(camera, dict):
        _repair_alias(camera, "description", "target_motion", repairs, context="camera")
        _insert(camera, "schema_version", TARGET_CAMERA_SCHEMA, repairs, context="camera")
        _normalize_natural_language_leaf(
            camera, "target_motion", repairs, context="camera"
        )
        if "motion_class" in camera:
            normalized = _normalize_camera_class(camera["motion_class"])
            if normalized != camera["motion_class"]:
                camera["motion_class"] = normalized
                repairs.append("normalized_camera_target.motion_class")
        _canonicalize_camera_relation(
            camera,
            source_census=source_census,
            repairs=repairs,
        )
        if "relation" not in camera and "motion_class" in camera:
            source_class = source_census["camera"]["motion_class"]
            relation = (
                "preserve_static"
                if source_class == "locked_off" and camera["motion_class"] == "locked_off"
                else "replace_motion"
            )
            _insert(camera, "relation", relation, repairs, context="camera")
    if "coverage" not in value:
        value["coverage"] = {
            "schema_version": TARGET_COVERAGE_SCHEMA,
            "dynamic_subject_ids": source_ids,
            "camera_covered": camera is not None,
        }
        repairs.append("inserted_plan.coverage_from_explicit_targets")
    elif isinstance(value["coverage"], dict):
        _insert(
            value["coverage"], "schema_version", TARGET_COVERAGE_SCHEMA, repairs,
            context="coverage",
        )
    receipt = {
        "schema_version": MECHANICAL_REPAIR_SCHEMA,
        "stage": "target_plan",
        "operations": repairs,
        "semantic_fields_invented": False,
    }
    return value, receipt


def _normalize_motion(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def validate_target_plan(
    value: Mapping[str, Any],
    *,
    expected_iid: str,
    source_census: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GokuFullMotionQwenV16Error("target plan must be an object")
    plan = copy.deepcopy(dict(value))
    _require_closed(
        plan,
        {
            "schema_version",
            "iid",
            "dynamic_subject_targets",
            "camera_target",
            "coverage",
            "confidence",
        },
        context="target_plan",
    )
    if plan["schema_version"] != TARGET_PLAN_SCHEMA or plan["iid"] != expected_iid:
        raise GokuFullMotionQwenV16Error("target plan identity differs")
    targets = plan["dynamic_subject_targets"]
    sources = source_census["dynamic_subjects"]
    if not isinstance(targets, list) or len(targets) != len(sources):
        raise GokuFullMotionQwenV16Error(
            "target plan must have exactly one target per dynamic subject"
        )
    for index, (raw_target, source) in enumerate(zip(targets, sources, strict=True)):
        context = f"target_plan.dynamic_subject_targets[{index}]"
        if not isinstance(raw_target, Mapping):
            raise GokuFullMotionQwenV16Error(f"{context} must be an object")
        target = dict(raw_target)
        _require_closed(
            target,
            {
                "schema_version",
                "subject_id",
                "target_action_signature",
                "target_motion",
                "substantive_change",
            },
            context=context,
        )
        if target["schema_version"] != TARGET_SUBJECT_SCHEMA:
            raise GokuFullMotionQwenV16Error(f"{context} schema differs")
        if target["subject_id"] != source["subject_id"]:
            raise GokuFullMotionQwenV16Error(
                f"{context} does not cover its source subject"
            )
        target["target_action_signature"] = _signature(
            target["target_action_signature"],
            context=f"{context}.target_action_signature",
        )
        _validate_target_timeline_text(
            target["target_action_signature"].replace("_", " "),
            context=f"{context}.target_action_signature",
        )
        target["target_motion"] = _text(
            target["target_motion"], context=f"{context}.target_motion"
        )
        _validate_target_timeline_text(
            target["target_motion"], context=f"{context}.target_motion"
        )
        if _UNSAFE_TARGET_RE.search(target["target_motion"]):
            raise GokuFullMotionQwenV16Error(
                f"{context}.target_motion refers to unavailable source future"
            )
        if target["substantive_change"] is not True:
            raise GokuFullMotionQwenV16Error(
                f"{context} is not a substantive action change"
            )
        if (
            target["target_action_signature"] == source["source_action_signature"]
            or _normalize_motion(target["target_motion"])
            == _normalize_motion(source["source_motion"])
        ):
            raise GokuFullMotionQwenV16Error(
                f"{context} repeats the source action"
            )
        targets[index] = target
    camera_raw = plan["camera_target"]
    if not isinstance(camera_raw, Mapping):
        raise GokuFullMotionQwenV16Error("target camera must be an object")
    camera = dict(camera_raw)
    _require_closed(
        camera,
        {"schema_version", "relation", "motion_class", "target_motion"},
        context="target_plan.camera_target",
    )
    if camera["schema_version"] != TARGET_CAMERA_SCHEMA:
        raise GokuFullMotionQwenV16Error("target camera schema differs")
    if camera["relation"] not in {"preserve_static", "replace_motion"}:
        raise GokuFullMotionQwenV16Error("target camera relation is invalid")
    if camera["motion_class"] not in CAMERA_MOTION_CLASSES:
        raise GokuFullMotionQwenV16Error("target camera class is invalid")
    camera["target_motion"] = _text(
        camera["target_motion"], context="target_plan.camera_target.target_motion"
    )
    _validate_target_timeline_text(
        camera["target_motion"],
        context="target_plan.camera_target.target_motion",
    )
    source_camera = source_census["camera"]
    if source_camera["motion_class"] != "locked_off":
        if camera["relation"] != "replace_motion":
            raise GokuFullMotionQwenV16Error(
                "a moving source camera requires an explicit replacement trajectory"
            )
        if (
            camera["motion_class"] == source_camera["motion_class"]
            and _normalize_motion(camera["target_motion"])
            == _normalize_motion(source_camera["source_motion"])
        ):
            raise GokuFullMotionQwenV16Error(
                "target camera repeats the moving source camera trajectory"
            )
    elif camera["relation"] == "preserve_static" and camera["motion_class"] != "locked_off":
        raise GokuFullMotionQwenV16Error(
            "preserve_static camera relation requires locked_off"
        )
    locked_assertion = _has_explicit_locked_camera_assertion(camera["target_motion"])
    explicitly_locked = _explicit_locked_camera_target(camera["target_motion"])
    affirmative_camera_motion = _has_affirmative_camera_motion(
        camera["target_motion"]
    )
    if camera["motion_class"] == "locked_off" and (
        not explicitly_locked or affirmative_camera_motion
    ):
        raise GokuFullMotionQwenV16Error(
            "locked_off target camera requires explicit, non-contradictory "
            "locked-camera prose"
        )
    if camera["motion_class"] != "locked_off" and locked_assertion:
        raise GokuFullMotionQwenV16Error(
            "moving target camera class contradicts locked-camera prose"
        )
    if camera["motion_class"] != "locked_off" and not affirmative_camera_motion:
        raise GokuFullMotionQwenV16Error(
            "moving target camera class requires explicit camera-motion prose"
        )
    coverage_raw = plan["coverage"]
    if not isinstance(coverage_raw, Mapping):
        raise GokuFullMotionQwenV16Error("target coverage must be an object")
    coverage = dict(coverage_raw)
    _require_closed(
        coverage,
        {"schema_version", "dynamic_subject_ids", "camera_covered"},
        context="target_plan.coverage",
    )
    source_ids = [item["subject_id"] for item in sources]
    if (
        coverage["schema_version"] != TARGET_COVERAGE_SCHEMA
        or coverage["dynamic_subject_ids"] != source_ids
        or coverage["camera_covered"] is not True
    ):
        raise GokuFullMotionQwenV16Error(
            "target coverage does not close all dynamic subjects and camera"
        )
    if plan["confidence"] not in {"high", "medium"}:
        raise GokuFullMotionQwenV16Error("target plan confidence is too low")
    plan["dynamic_subject_targets"] = targets
    plan["camera_target"] = camera
    plan["coverage"] = coverage
    return plan


def _render_compiled_instruction(
    source_census: Mapping[str, Any], target_plan: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    """Render the only instruction surface authorized by the signed plan."""

    clauses: list[dict[str, Any]] = []
    rendered: list[str] = []
    for source_subject, target in zip(
        source_census["dynamic_subjects"],
        target_plan["dynamic_subject_targets"],
        strict=True,
    ):
        text = (
            f"Have {source_subject['stable_reference']} perform this complete "
            f"target motion: {target['target_motion'].rstrip('.!?;:')}"
        )
        clauses.append(
            {
                "schema_version": INSTRUCTION_CLAUSE_SCHEMA,
                "kind": "dynamic_subject",
                "subject_id": source_subject["subject_id"],
                "text": text,
            }
        )
        rendered.append(text)
    preservation_text = (
        "Preserve every subject's identity and appearance and preserve all "
        "visible scene content except changes physically required by the "
        "specified target motions"
    )
    clauses.append(
        {
            "schema_version": INSTRUCTION_CLAUSE_SCHEMA,
            "kind": "preservation",
            "subject_id": None,
            "text": preservation_text,
        }
    )
    rendered.append(preservation_text)
    camera = target_plan["camera_target"]
    camera_text = (
        "Keep the camera locked off"
        if camera["motion_class"] == "locked_off"
        else "Set the camera trajectory to "
        + camera["target_motion"].rstrip(".!?;:")
    )
    clauses.append(
        {
            "schema_version": INSTRUCTION_CLAUSE_SCHEMA,
            "kind": "camera",
            "subject_id": None,
            "text": camera_text,
        }
    )
    rendered.append(camera_text)
    instruction = "Starting from the exact first frame: " + "; ".join(rendered) + "."
    return clauses, instruction


def compile_instruction(
    source_census: Mapping[str, Any], target_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Render one actor-bound clause per dynamic subject plus camera."""

    source = validate_source_census(
        source_census, expected_iid=str(source_census.get("iid") or "")
    )
    plan = validate_target_plan(
        target_plan,
        expected_iid=str(source["iid"]),
        source_census=source,
    )
    clauses, instruction = _render_compiled_instruction(source, plan)
    result = {
        "schema_version": COMPILED_INSTRUCTION_SCHEMA,
        "iid": source["iid"],
        "instruction": instruction,
        "instruction_sha256": _sha256_bytes(instruction.encode("utf-8")),
        "source_census_sha256": object_sha256(source),
        "target_plan_sha256": object_sha256(plan),
        "clauses": clauses,
        "covered_dynamic_subject_ids": [
            item["subject_id"] for item in source["dynamic_subjects"]
        ],
        "camera_covered": True,
    }
    return validate_compiled_instruction(result, source_census=source, target_plan=plan)


def validate_compiled_instruction(
    value: Mapping[str, Any],
    *,
    source_census: Mapping[str, Any],
    target_plan: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GokuFullMotionQwenV16Error("compiled instruction must be an object")
    compiled = copy.deepcopy(dict(value))
    _require_closed(
        compiled,
        {
            "schema_version",
            "iid",
            "instruction",
            "instruction_sha256",
            "source_census_sha256",
            "target_plan_sha256",
            "clauses",
            "covered_dynamic_subject_ids",
            "camera_covered",
        },
        context="compiled_instruction",
    )
    if compiled["schema_version"] != COMPILED_INSTRUCTION_SCHEMA:
        raise GokuFullMotionQwenV16Error("compiled instruction schema differs")
    if compiled["iid"] != source_census["iid"]:
        raise GokuFullMotionQwenV16Error("compiled instruction IID differs")
    instruction = _text(
        compiled["instruction"],
        context="compiled_instruction.instruction",
        minimum=30,
        maximum=5000,
        allow_semicolon=True,
    )
    _validate_target_timeline_text(
        instruction,
        context="compiled_instruction.instruction",
        check_sequence=False,
    )
    if compiled["instruction_sha256"] != _sha256_bytes(instruction.encode("utf-8")):
        raise GokuFullMotionQwenV16Error("compiled instruction hash differs")
    if compiled["source_census_sha256"] != object_sha256(source_census):
        raise GokuFullMotionQwenV16Error("compiled source census hash differs")
    if compiled["target_plan_sha256"] != object_sha256(target_plan):
        raise GokuFullMotionQwenV16Error("compiled target plan hash differs")
    expected_ids = [
        item["subject_id"] for item in source_census["dynamic_subjects"]
    ]
    if compiled["covered_dynamic_subject_ids"] != expected_ids:
        raise GokuFullMotionQwenV16Error("compiled subject coverage differs")
    if compiled["camera_covered"] is not True:
        raise GokuFullMotionQwenV16Error("compiled camera coverage is missing")
    clauses = compiled["clauses"]
    if not isinstance(clauses, list) or len(clauses) != len(expected_ids) + 2:
        raise GokuFullMotionQwenV16Error("compiled clause count differs")
    for index, clause_raw in enumerate(clauses):
        if not isinstance(clause_raw, Mapping):
            raise GokuFullMotionQwenV16Error("compiled clause must be an object")
        clause = dict(clause_raw)
        _require_closed(
            clause,
            {"schema_version", "kind", "subject_id", "text"},
            context=f"compiled_instruction.clauses[{index}]",
        )
        if clause["schema_version"] != INSTRUCTION_CLAUSE_SCHEMA:
            raise GokuFullMotionQwenV16Error("compiled clause schema differs")
        if index < len(expected_ids):
            if clause["kind"] != "dynamic_subject" or clause["subject_id"] != expected_ids[index]:
                raise GokuFullMotionQwenV16Error("compiled dynamic clause order differs")
        elif index == len(expected_ids):
            if clause["kind"] != "preservation" or clause["subject_id"] is not None:
                raise GokuFullMotionQwenV16Error(
                    "compiled preservation clause differs"
                )
        elif clause["kind"] != "camera" or clause["subject_id"] is not None:
            raise GokuFullMotionQwenV16Error("compiled final clause is not camera")
        clause_text = _text(
            clause["text"], context="compiled clause text", maximum=1200
        )
        if clause["kind"] in {"dynamic_subject", "camera"}:
            _validate_target_timeline_text(
                clause_text,
                context=f"compiled_instruction.clauses[{index}].text",
            )
        else:
            _reject_source_sample_markers(
                clause_text,
                context=f"compiled_instruction.clauses[{index}].text",
            )
        if clause_text not in instruction:
            raise GokuFullMotionQwenV16Error("compiled clause text is not rendered")
    expected_clauses, expected_instruction = _render_compiled_instruction(
        source_census, target_plan
    )
    if clauses != expected_clauses:
        raise GokuFullMotionQwenV16Error(
            "compiled clauses differ from the signed target plan"
        )
    if instruction != expected_instruction:
        raise GokuFullMotionQwenV16Error(
            "compiled instruction differs from the deterministic clause rendering"
        )
    return compiled


SOURCE_CENSUS_SYSTEM = """You are a precise video-motion annotator. Inspect the exact first frame and every chronological source-video view. Return only one JSON object. Enumerate every independently moving, I0-identifiable subject exactly once, including people, animals, vehicles, articulated objects, and emitters. Do not merge two actors. Exclude passive background pixels. Explicitly classify camera motion. If a crowd or motion cannot be resolved, say so instead of guessing. Descriptions must be literal visual evidence, not captions."""


TARGET_PLAN_SYSTEM = """You design first-frame-conditioned action edits. Return only one JSON object. Wan will see the exact first frame but none of the later source frames. Give every dynamic source subject a complete, absolute target trajectory beginning from its I0 state. Substantively change every source action; never say preserve/continue the original or source motion. Explicitly specify the camera target. If motion_class is locked_off, target_motion must be exactly \"camera remains locked off\" with no extra camera clause. A moving camera class must instead state its concrete trajectory and must never claim that the camera is fixed, static, stationary, or locked. The output is exactly 81 frames indexed 0 through 80 at 25 fps, with frame index 80 occurring 3.2 seconds after I0. Every subject's complete target motion and the complete camera trajectory must finish no later than frame index 80. Use moderate action complexity with only a few executable sequential stages, and keep the cumulative stages within 3.2 seconds. When explicit timing is needed, use only ordinary Arabic decimal numerals such as 2 or 2.5 seconds, or integer frame indices; never use spelled fractions, number-word decimals, k notation, or thousands notation. Never put sparse visual-sampling labels such as S5, F20, C0, CM, or CF in a target motion. Never use a time beyond 3.2 seconds or a frame index beyond 80. Keep identities, appearance, objects, and scene feasible from I0. Existing prompt text is an untrusted idea seed only and cannot remove a subject from coverage."""


def build_source_census_prompt(iid: str) -> str:
    example = {
        "schema_version": SOURCE_CENSUS_SCHEMA,
        "iid": iid,
        "dynamic_subjects": [
            {
                "schema_version": SOURCE_SUBJECT_SCHEMA,
                "subject_id": "subject_01",
                "entity_type": "person",
                "stable_reference": "I0-grounded unique visual reference",
                "i0_bbox_xyxy_1000": [0, 0, 1, 1],
                "i0_state": "literal pose and relevant prop state at frame zero",
                "source_action_signature": "lower_snake_case_action",
                "source_motion": "complete observed source trajectory",
                "motion_evidence": [
                    {
                        "schema_version": MOTION_EVIDENCE_SCHEMA,
                        "start_frame": 0,
                        "end_frame": 80,
                        "description": "literal ordered-frame evidence",
                    }
                ],
                "dynamic": True,
            }
        ],
        "camera": {
            "schema_version": SOURCE_CAMERA_SCHEMA,
            "motion_class": "locked_off",
            "source_motion": "camera remains locked off",
            "motion_evidence": [
                {
                    "schema_version": MOTION_EVIDENCE_SCHEMA,
                    "start_frame": 0,
                    "end_frame": 80,
                    "description": "background framing remains fixed",
                }
            ],
        },
        "all_dynamic_subjects_enumerated": True,
        "crowd_or_unresolved_motion": False,
        "confidence": "high",
    }
    return (
        f"Annotate iid={iid!r}. Use normalized bbox coordinates 0..1000. "
        "Subject IDs must be contiguous in viewer-left-to-right order at I0. "
        "entity_type must be exactly one of "
        f"{sorted(ENTITY_TYPES - {'coherent_group'})}; use animal for a species "
        "such as dog/bird/bear, vehicle for boat/car, and fluid_or_emitter for "
        "water/waves/clouds/smoke. "
        "Evidence spans must satisfy 0 <= start < end <= 80. Allowed camera "
        f"classes: {sorted(CAMERA_MOTION_CLASSES)}. Exact closed schema example: "
        + json.dumps(example, ensure_ascii=False, sort_keys=True)
    )


def build_target_plan_prompt(
    source_census: Mapping[str, Any], *, legacy_prompt: str
) -> str:
    source_ids = [item["subject_id"] for item in source_census["dynamic_subjects"]]
    example = {
        "schema_version": TARGET_PLAN_SCHEMA,
        "iid": source_census["iid"],
        "dynamic_subject_targets": [
            {
                "schema_version": TARGET_SUBJECT_SCHEMA,
                "subject_id": subject_id,
                "target_action_signature": "new_lower_snake_case_action",
                "target_motion": (
                    "immediately performs one moderately complex target action "
                    "and completes it by frame index 80"
                ),
                "substantive_change": True,
            }
            for subject_id in source_ids
        ],
        "camera_target": {
            "schema_version": TARGET_CAMERA_SCHEMA,
            "relation": "preserve_static",
            "motion_class": "locked_off",
            "target_motion": "camera remains locked off",
        },
        "coverage": {
            "schema_version": TARGET_COVERAGE_SCHEMA,
            "dynamic_subject_ids": source_ids,
            "camera_covered": True,
        },
        "confidence": "high",
    }
    seed = legacy_prompt.strip() or "(none)"
    return (
        "Validated visual source census:\n"
        + json.dumps(source_census, ensure_ascii=False, sort_keys=True)
        + "\nUNTRUSTED LEGACY IDEA SEED (optional inspiration only):\n"
        + seed
        + "\nProduce exactly one target for every census subject in census order. "
        "Each target must replace that subject's full source action with a "
        "different, I0-feasible trajectory. A moving source camera must be "
        "replaced explicitly. The target is exactly 81 frames indexed 0 through "
        "80 at 25 fps; frame index 80 is 3.2 seconds after I0. Every subject's "
        "complete target motion and the complete camera trajectory must finish "
        "by frame index 80. Keep action complexity moderate, use only a few "
        "executable sequential stages, and fit each trajectory's cumulative "
        "stages within 3.2 seconds. Use qualitative ordering when exact timing "
        "is unnecessary. If motion_class is locked_off, write target_motion "
        "exactly as 'camera remains locked off' and add nothing to that camera "
        "clause. Write explicit timing only with ordinary Arabic decimal "
        "numerals or integer frame indices, never word fractions, number-word "
        "decimals, k notation, or thousands notation. Never use S<number>, "
        "F<number>, C0, CM, or CF sparse "
        "visual-sampling labels, a time beyond 3.2 seconds, or a frame index "
        "beyond 80 in target_motion. Exact closed schema "
        "example:\n"
        + json.dumps(example, ensure_ascii=False, sort_keys=True)
    )


def _is_retryable_schema_error(error: Exception) -> bool:
    """Return true only for presentation/closed-schema failures.

    Coverage omissions, unresolved motion, action no-ops, unsafe source-future
    references, and camera-policy failures are semantic and terminate the row
    immediately.  Retrying those would turn the model into a hidden critic
    loop and recreate the v15 throughput problem.
    """

    message = str(error).casefold()
    mechanical_markers = (
        "not a single strict json object",
        "must be a json object",
        "must be an object",
        "keys differ from closed schema",
        "schema differs",
        "identity differs",
        "must be lower_snake_case",
        "subject_id must be",
        "clause delimiter",
        "length must be in",
        "must be a string",
    )
    semantic_markers = (
        "exactly one target per dynamic subject",
        "crowd or unresolved",
        "complete dynamic-subject coverage",
        "motion_evidence must be non-empty",
        "camera evidence must be non-empty",
        "unavailable source future",
        "not a substantive action change",
        "repeats the source action",
        "source-view sample marker",
        "exceeds the 3.2-second target timeline",
        "exceeds the 80-frame target timeline",
        "outside integer frame indices 0..80",
        "explicit stages require at least",
        "ambiguous temporal amount",
        "replacement trajectory",
        "repeats the moving source camera",
        "target coverage does not close",
        "confidence is too low",
    )
    return not any(marker in message for marker in semantic_markers) and any(
        marker in message for marker in mechanical_markers
    )


def _schema_retry_prompt(base_prompt: str, *, stage: str, error: Exception) -> str:
    return (
        base_prompt
        + "\nMECHANICAL RETRY (one attempt only): Your previous response failed "
        + stage
        + " closed-schema validation with this error: "
        + str(error)[:700]
        + ". Re-inspect the same visual evidence and emit one bare JSON object "
        "matching the exact schema. Preserve the visual semantics; only repair "
        "JSON structure, types, IDs, and required fields."
    )


def _attempt_transcript(
    *, attempt: int, prompt: str, raw: str, error: Exception | None
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "raw": raw,
        "raw_sha256": _sha256_bytes(raw.encode("utf-8")),
        "error_type": None if error is None else type(error).__name__,
        "error": None if error is None else str(error),
    }


def _generate_visual_stage(
    backend: Any,
    *,
    custom_method: str,
    system: str,
    prompt: str,
    source_path: Path,
    anchor_path: Path,
    visuals: Sequence[Any],
    visual_input_digest: str,
    runtime: Mapping[str, int],
) -> str:
    custom = getattr(backend, custom_method, None)
    if callable(custom):
        result = custom(
            source_path=str(source_path),
            anchor_path=str(anchor_path),
            system=system,
            user=prompt,
            expected_visual_input_digest=visual_input_digest,
            **runtime,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise GokuFullMotionQwenV16Error(
                f"backend {custom_method} must return (raw, visual_digest)"
            )
        raw, digest = result
        if digest != visual_input_digest:
            raise GokuFullMotionQwenV16Error(
                f"backend {custom_method} visual digest differs"
            )
        return str(raw)
    if getattr(backend, "mode", None) != "visual":
        raise GokuFullMotionQwenV16Error("v16 requires a visual backend")
    processor = getattr(backend, "processor", None)
    if processor is None:
        raise GokuFullMotionQwenV16Error("visual backend has no processor")
    labels = (
        "EXACT LOSSLESS INITIAL FRAME I0",
        "SOURCE CHRONOLOGICAL MOSAIC",
        "SOURCE FULL-FRAME I0/MID/FINAL",
        "SOURCE LEFT/RIGHT TEMPORAL ZOOMS",
        "DETERMINISTIC PIXEL-CHANGE ATTENTION AID",
    )
    content: list[dict[str, Any]] = []
    for label, image in zip(labels, visuals, strict=True):
        content.extend(
            ({"type": "text", "text": label}, {"type": "image", "image": image})
        )
    content.append({"type": "text", "text": prompt})
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[rendered],
        images=list(visuals),
        videos=None,
        padding=True,
        return_tensors="pt",
    ).to(backend.model.device)
    with backend.torch.inference_mode():
        generated = backend.model.generate(
            **inputs,
            max_new_tokens=backend.max_new_tokens,
            do_sample=False,
        )
    return backend._decode(inputs, generated, processor)


def annotate_prepared_row(
    row: Mapping[str, Any],
    *,
    backend: Any,
    source_path: Path,
    anchor_path: Path,
    media_verification: Mapping[str, Any],
    visuals: Sequence[Any],
    visual_input_digest: str,
    runtime: Mapping[str, int],
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run two normal calls, with one local retry for mechanical errors only."""

    trace_out = trace if trace is not None else {}
    iid = str(row["iid"])
    trace_out["media_verification"] = dict(media_verification)
    trace_out["visual_input_digest"] = visual_input_digest
    census_base_prompt = build_source_census_prompt(iid)
    census_attempts: list[dict[str, Any]] = []
    census: dict[str, Any] | None = None
    census_repair: dict[str, Any] | None = None
    census_prompt = census_base_prompt
    for attempt in (1, 2):
        census_raw = _generate_visual_stage(
            backend,
            custom_method="generate_source_motion_census_v16",
            system=SOURCE_CENSUS_SYSTEM,
            prompt=census_prompt,
            source_path=source_path,
            anchor_path=anchor_path,
            visuals=visuals,
            visual_input_digest=visual_input_digest,
            runtime=runtime,
        )
        try:
            census_value, census_parse_repairs = _loads_object(
                census_raw, stage="source census"
            )
            census_canonical, census_repair = canonicalize_source_census(
                census_value, expected_iid=iid
            )
            census_repair["operations"] = (
                census_parse_repairs + census_repair["operations"]
            )
            census = validate_source_census(census_canonical, expected_iid=iid)
        except Exception as error:
            census_attempts.append(
                _attempt_transcript(
                    attempt=attempt, prompt=census_prompt, raw=census_raw, error=error
                )
            )
            trace_out["source_stage"] = {
                "attempts": census_attempts,
                "selected_attempt": None,
                "mechanical_repair": None,
            }
            if attempt == 1 and _is_retryable_schema_error(error):
                census_prompt = _schema_retry_prompt(
                    census_base_prompt, stage="source census", error=error
                )
                continue
            raise GokuFullMotionQwenV16StageError(
                "source_census", census_attempts
            ) from error
        census_attempts.append(
            _attempt_transcript(
                attempt=attempt, prompt=census_prompt, raw=census_raw, error=None
            )
        )
        trace_out["source_stage"] = {
            "attempts": census_attempts,
            "selected_attempt": attempt,
            "mechanical_repair": census_repair,
        }
        break
    assert census is not None
    trace_out["source_census"] = census

    target_base_prompt = build_target_plan_prompt(
        census, legacy_prompt=str(row.get("prompt") or "")
    )
    target_attempts: list[dict[str, Any]] = []
    target_plan: dict[str, Any] | None = None
    target_repair: dict[str, Any] | None = None
    target_prompt = target_base_prompt
    for attempt in (1, 2):
        target_raw = _generate_visual_stage(
            backend,
            custom_method="generate_target_plan_v16",
            system=TARGET_PLAN_SYSTEM,
            prompt=target_prompt,
            source_path=source_path,
            anchor_path=anchor_path,
            visuals=visuals,
            visual_input_digest=visual_input_digest,
            runtime=runtime,
        )
        try:
            target_value, target_parse_repairs = _loads_object(
                target_raw, stage="target plan"
            )
            target_canonical, target_repair = canonicalize_target_plan(
                target_value,
                expected_iid=iid,
                source_census=census,
            )
            target_repair["operations"] = (
                target_parse_repairs + target_repair["operations"]
            )
            target_plan = validate_target_plan(
                target_canonical,
                expected_iid=iid,
                source_census=census,
            )
        except Exception as error:
            target_attempts.append(
                _attempt_transcript(
                    attempt=attempt, prompt=target_prompt, raw=target_raw, error=error
                )
            )
            trace_out["target_stage"] = {
                "attempts": target_attempts,
                "selected_attempt": None,
                "mechanical_repair": None,
            }
            if attempt == 1 and _is_retryable_schema_error(error):
                target_prompt = _schema_retry_prompt(
                    target_base_prompt, stage="target plan", error=error
                )
                continue
            raise GokuFullMotionQwenV16StageError(
                "target_plan", target_attempts
            ) from error
        target_attempts.append(
            _attempt_transcript(
                attempt=attempt, prompt=target_prompt, raw=target_raw, error=None
            )
        )
        trace_out["target_stage"] = {
            "attempts": target_attempts,
            "selected_attempt": attempt,
            "mechanical_repair": target_repair,
        }
        break
    assert target_plan is not None
    trace_out["target_plan"] = target_plan
    compiled = compile_instruction(census, target_plan)
    payload = {
        "media_verification": dict(media_verification),
        "visual_input_digest": visual_input_digest,
        "source_stage": trace_out["source_stage"],
        "target_stage": trace_out["target_stage"],
        "source_census": census,
        "target_plan": target_plan,
        "compiled_instruction": compiled,
    }
    trace_out.update(payload)
    return payload


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=True)


def _video_geometry(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise GokuFullMotionQwenV16Error(f"cannot open source video: {path}")
    try:
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    if frame_count != FRAME_COUNT or not math.isclose(fps, FPS, abs_tol=1e-3):
        raise GokuFullMotionQwenV16Error(
            f"source geometry must be {FRAME_COUNT} frames at {FPS:g} fps"
        )
    if width <= 0 or height <= 0:
        raise GokuFullMotionQwenV16Error("source dimensions are invalid")
    return {
        "frame_count": frame_count,
        "fps": "25/1",
        "timeline_span_seconds": (frame_count - 1) / fps,
        "width": width,
        "height": height,
    }


def prepare_row(
    row: Mapping[str, Any], *, root: Path, runtime: Mapping[str, int]
) -> tuple[Path, Path, dict[str, Any], tuple[Any, ...], str]:
    selected = validate_input_row(dict(row))
    source_path = _resolve_path(str(selected["resolved_src_video"]), root)
    anchor_path = _resolve_path(str(selected["resolved_anchor_image"]), root)
    verification = verify_exact_i0_binding(
        source_path=source_path,
        anchor_path=anchor_path,
        source_sha256=str(selected["source_video_sha256"]),
        anchor_sha256=str(selected["anchor_sha256"]),
    )
    verification["temporal_geometry"] = _video_geometry(source_path)
    visuals_with_digest = _build_visuals(
        source_path=source_path,
        anchor_path=anchor_path,
        nframes=runtime["nframes"],
        max_pixels=runtime["max_pixels"],
        tile_width=runtime["tile_width"],
        mosaic_columns=runtime["mosaic_columns"],
    )
    return (
        source_path,
        anchor_path,
        verification,
        tuple(visuals_with_digest[:-1]),
        str(visuals_with_digest[-1]),
    )


def _model_identity(backend: Any) -> dict[str, str]:
    return {
        "model_path": str(getattr(backend, "model_path", "test-backend")),
        "model_revision": str(getattr(backend, "model_revision", "unknown")),
        "transformers_version": str(
            getattr(backend, "transformers_version", "unknown")
        ),
    }


def _new_record(
    row: Mapping[str, Any], *, backend: Any, runtime: Mapping[str, int]
) -> dict[str, Any]:
    return {
        "schema_version": RECORD_SCHEMA,
        "iid": str(row["iid"]),
        "status": "running",
        "input_digest": object_sha256(row),
        "input_row": copy.deepcopy(dict(row)),
        "model": _model_identity(backend),
        "runtime": dict(runtime),
        "media_verification": None,
        "visual_input_digest": None,
        "source_stage": None,
        "target_stage": None,
        "source_census": None,
        "target_plan": None,
        "compiled_instruction": None,
        "error": None,
        "record_digest": None,
    }


def _passed_row(record: Mapping[str, Any], *, source_path: Path, anchor_path: Path) -> dict[str, Any]:
    row = record["input_row"]
    compiled = record["compiled_instruction"]
    return {
        "schema_version": PASSED_SCHEMA,
        "iid": record["iid"],
        "group_id": row["group_id"],
        "family": row["family"],
        "source_video": row["src_video"],
        "resolved_source_video": str(source_path),
        "anchor_image": row["anchor_image"],
        "resolved_anchor_image": str(anchor_path),
        "source_video_sha256": row["source_video_sha256"],
        "anchor_sha256": row["anchor_sha256"],
        "strict_temporal_geometry": record["media_verification"]["temporal_geometry"],
        "edit_instruction": compiled["instruction"],
        "edit_instruction_sha256": compiled["instruction_sha256"],
        "source_census": record["source_census"],
        "target_plan": record["target_plan"],
        "compiled_instruction": compiled,
        "qwen_record_digest": record["record_digest"],
        "action_change_substantive": True,
        "all_dynamic_subjects_covered": True,
        "camera_covered": True,
        "human_review_status": "pending",
        "generation_authorized": False,
        "production_eligible": False,
    }


def validate_passed_row(value: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-validate the non-production row consumed by the Wan v16 adapter."""

    if not isinstance(value, Mapping):
        raise GokuFullMotionQwenV16Error("passed row must be an object")
    row = copy.deepcopy(dict(value))
    _require_closed(
        row,
        {
            "schema_version",
            "iid",
            "group_id",
            "family",
            "source_video",
            "resolved_source_video",
            "anchor_image",
            "resolved_anchor_image",
            "source_video_sha256",
            "anchor_sha256",
            "strict_temporal_geometry",
            "edit_instruction",
            "edit_instruction_sha256",
            "source_census",
            "target_plan",
            "compiled_instruction",
            "qwen_record_digest",
            "action_change_substantive",
            "all_dynamic_subjects_covered",
            "camera_covered",
            "human_review_status",
            "generation_authorized",
            "production_eligible",
        },
        context="passed_row",
    )
    if row["schema_version"] != PASSED_SCHEMA:
        raise GokuFullMotionQwenV16Error("passed row schema differs")
    iid = row["iid"]
    if not isinstance(iid, str) or _IID_RE.fullmatch(iid) is None:
        raise GokuFullMotionQwenV16Error("passed row IID is invalid")
    for field in ("group_id", "family", "source_video", "anchor_image"):
        row[field] = _text(
            row[field], context=f"passed_row.{field}", maximum=2000
        )
    for field in ("resolved_source_video", "resolved_anchor_image"):
        path_value = row[field]
        if (
            not isinstance(path_value, str)
            or not path_value
            or "\n" in path_value
            or not Path(path_value).is_absolute()
        ):
            raise GokuFullMotionQwenV16Error(
                f"passed_row.{field} must be an absolute path"
            )
    for field in (
        "source_video_sha256",
        "anchor_sha256",
        "edit_instruction_sha256",
        "qwen_record_digest",
    ):
        if not isinstance(row[field], str) or _SHA256_RE.fullmatch(row[field]) is None:
            raise GokuFullMotionQwenV16Error(f"passed_row.{field} is invalid")
    geometry_raw = row["strict_temporal_geometry"]
    if not isinstance(geometry_raw, Mapping):
        raise GokuFullMotionQwenV16Error("passed temporal geometry must be an object")
    geometry = dict(geometry_raw)
    _require_closed(
        geometry,
        {"frame_count", "fps", "timeline_span_seconds", "width", "height"},
        context="passed_row.strict_temporal_geometry",
    )
    if (
        geometry["frame_count"] != FRAME_COUNT
        or geometry["fps"] != "25/1"
        or isinstance(geometry["timeline_span_seconds"], bool)
        or not isinstance(geometry["timeline_span_seconds"], (int, float))
        or not math.isclose(
            float(geometry["timeline_span_seconds"]), 3.2, abs_tol=1e-9
        )
        or type(geometry["width"]) is not int
        or type(geometry["height"]) is not int
        or geometry["width"] <= 0
        or geometry["height"] <= 0
    ):
        raise GokuFullMotionQwenV16Error("passed temporal geometry differs")
    census = validate_source_census(row["source_census"], expected_iid=iid)
    plan = validate_target_plan(
        row["target_plan"], expected_iid=iid, source_census=census
    )
    compiled = validate_compiled_instruction(
        row["compiled_instruction"],
        source_census=census,
        target_plan=plan,
    )
    if row["edit_instruction"] != compiled["instruction"]:
        raise GokuFullMotionQwenV16Error(
            "passed edit instruction differs from compiled instruction"
        )
    if row["edit_instruction_sha256"] != _sha256_bytes(
        str(row["edit_instruction"]).encode("utf-8")
    ):
        raise GokuFullMotionQwenV16Error("passed edit instruction hash differs")
    for field in (
        "action_change_substantive",
        "all_dynamic_subjects_covered",
        "camera_covered",
    ):
        if row[field] is not True:
            raise GokuFullMotionQwenV16Error(f"passed row {field} must be true")
    if row["human_review_status"] != "pending":
        raise GokuFullMotionQwenV16Error("passed review status must be pending")
    if row["generation_authorized"] is not False:
        raise GokuFullMotionQwenV16Error(
            "passed row must not claim generation authorization"
        )
    if row["production_eligible"] is not False:
        raise GokuFullMotionQwenV16Error(
            "passed row must not claim production eligibility"
        )
    row["strict_temporal_geometry"] = geometry
    row["source_census"] = census
    row["target_plan"] = plan
    row["compiled_instruction"] = compiled
    return row


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise GokuFullMotionQwenV16Error(f"unsafe output directory: {path}")


def _publish_create_only(path: Path, payload: bytes) -> None:
    """Publish atomically without ever replacing an existing path."""

    _ensure_directory(path.parent)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"create-only output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_read_object(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    value, repairs = _loads_object(raw, stage=str(path))
    if repairs:
        raise GokuFullMotionQwenV16Error(f"published JSON is not canonical: {path}")
    return value


def _validate_terminal_receipt(
    receipt_path: Path, *, output_root: Path, iid: str, input_digest: str
) -> dict[str, Any]:
    receipt = _strict_read_object(receipt_path)
    _require_closed(
        receipt,
        {
            "schema_version",
            "iid",
            "status",
            "input_digest",
            "result_path",
            "result_sha256",
            "passed_path",
            "passed_sha256",
            "receipt_digest",
        },
        context="terminal_receipt",
    )
    if (
        receipt["schema_version"] != ROW_RECEIPT_SCHEMA
        or receipt["iid"] != iid
        or receipt["input_digest"] != input_digest
        or receipt["status"] not in {"ok", "error"}
    ):
        raise GokuFullMotionQwenV16Error("terminal receipt identity differs")
    result_path = output_root / "rows" / iid / "result.json"
    if receipt["result_path"] != str(result_path.resolve()) or not result_path.is_file():
        raise GokuFullMotionQwenV16Error("terminal receipt result path differs")
    if receipt["result_sha256"] != _file_digest(result_path):
        raise GokuFullMotionQwenV16Error("terminal receipt result hash differs")
    passed_path = output_root / "passed" / f"{iid}.jsonl"
    if receipt["status"] == "ok":
        if (
            receipt["passed_path"] != str(passed_path.resolve())
            or not passed_path.is_file()
            or receipt["passed_sha256"] != _file_digest(passed_path)
        ):
            raise GokuFullMotionQwenV16Error("terminal receipt passed binding differs")
    elif receipt["passed_path"] is not None or receipt["passed_sha256"] is not None:
        raise GokuFullMotionQwenV16Error("error receipt unexpectedly binds passed row")
    if receipt["receipt_digest"] != _digest_object_with_field(
        receipt, "receipt_digest"
    ):
        raise GokuFullMotionQwenV16Error("terminal receipt digest differs")
    return receipt


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise GokuFullMotionQwenV16Error("input JSONL must be non-empty and newline terminated")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise GokuFullMotionQwenV16Error(f"input line {index} is not UTF-8") from error
        value, repairs = _loads_object(text, stage=f"input line {index}")
        if repairs:
            raise GokuFullMotionQwenV16Error(f"input line {index} is fenced")
        rows.append(value)
    iids = [str(row.get("iid") or "") for row in rows]
    if any(_IID_RE.fullmatch(iid) is None for iid in iids) or len(set(iids)) != len(iids):
        raise GokuFullMotionQwenV16Error("input IIDs are invalid or duplicated")
    return rows


def run_one(
    args: argparse.Namespace,
    *,
    backend_factory: Callable[..., Any] | None = None,
    backend: Any | None = None,
    prepare: Callable[..., tuple[Path, Path, dict[str, Any], tuple[Any, ...], str]] = prepare_row,
    loaded_rows: Sequence[Mapping[str, Any]] | None = None,
) -> int:
    input_path = args.input.expanduser().resolve(strict=True)
    rows = (
        _iter_jsonl(input_path)
        if loaded_rows is None
        else [dict(row) for row in loaded_rows]
    )
    if args.num_rows != len(rows):
        raise GokuFullMotionQwenV16Error(
            f"--num-rows={args.num_rows} differs from input rows={len(rows)}"
        )
    if args.row_index is None or not 0 <= args.row_index < args.num_rows:
        raise GokuFullMotionQwenV16Error("row index is out of range")
    row = rows[args.row_index]
    validate_input_row(dict(row))
    iid = str(row["iid"])
    output_root = args.output_root.expanduser().resolve()
    _ensure_directory(output_root)
    result_path = output_root / "rows" / iid / "result.json"
    passed_path = output_root / "passed" / f"{iid}.jsonl"
    receipt_path = output_root / "terminal" / f"{iid}.receipt.json"
    input_digest = object_sha256(row)
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = _validate_terminal_receipt(
            receipt_path,
            output_root=output_root,
            iid=iid,
            input_digest=input_digest,
        )
        return 0 if receipt["status"] == "ok" or args.allow_errors else 2
    if result_path.exists() or passed_path.exists():
        raise GokuFullMotionQwenV16Error(
            f"partial create-only output exists for iid={iid}; use a fresh output root"
        )
    runtime = {
        "nframes": args.nframes,
        "max_pixels": args.max_pixels,
        "tile_width": args.tile_width,
        "mosaic_columns": args.mosaic_columns,
    }
    if backend is None:
        factory = backend_factory or LocalQwenBackend
        backend = factory(
            model_path=args.model,
            mode="visual",
            attn_implementation=args.attn_implementation,
            allow_download=args.allow_download,
            max_new_tokens=args.max_new_tokens,
        )
        _reject_backend_cpu_or_disk_offload(backend)
    record = _new_record(row, backend=backend, runtime=runtime)
    source_path: Path | None = None
    anchor_path: Path | None = None
    annotation_trace: dict[str, Any] = {}
    try:
        source_path, anchor_path, media, visuals, visual_digest = prepare(
            row, root=args.root.expanduser().resolve(strict=True), runtime=runtime
        )
        payload = annotate_prepared_row(
            row,
            backend=backend,
            source_path=source_path,
            anchor_path=anchor_path,
            media_verification=media,
            visuals=visuals,
            visual_input_digest=visual_digest,
            runtime=runtime,
            trace=annotation_trace,
        )
        record.update(payload)
        record["status"] = "ok"
    except GokuFullMotionQwenV16StageError as error:
        for field in (
            "media_verification",
            "visual_input_digest",
            "source_stage",
            "target_stage",
            "source_census",
            "target_plan",
            "compiled_instruction",
        ):
            if field in annotation_trace:
                record[field] = annotation_trace[field]
        record["status"] = "error"
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    record["record_digest"] = _digest_object_with_field(record, "record_digest")
    result_bytes = _pretty_bytes(record)
    _publish_create_only(result_path, result_bytes)
    passed_sha: str | None = None
    if record["status"] == "ok":
        assert source_path is not None and anchor_path is not None
        passed = validate_passed_row(
            _passed_row(record, source_path=source_path, anchor_path=anchor_path)
        )
        passed_bytes = _canonical_bytes(passed) + b"\n"
        _publish_create_only(passed_path, passed_bytes)
        passed_sha = _sha256_bytes(passed_bytes)
    receipt: dict[str, Any] = {
        "schema_version": ROW_RECEIPT_SCHEMA,
        "iid": iid,
        "status": record["status"],
        "input_digest": input_digest,
        "result_path": str(result_path.resolve()),
        "result_sha256": _sha256_bytes(result_bytes),
        "passed_path": str(passed_path.resolve()) if passed_sha is not None else None,
        "passed_sha256": passed_sha,
        "receipt_digest": None,
    }
    receipt["receipt_digest"] = _digest_object_with_field(receipt, "receipt_digest")
    _publish_create_only(receipt_path, _pretty_bytes(receipt))
    print(
        f"[goku-full-motion-qwen-v16] iid={iid} status={record['status']} "
        f"result={result_path}",
        flush=True,
    )
    return 0 if record["status"] == "ok" or args.allow_errors else 2


def run_worker(
    args: argparse.Namespace,
    *,
    backend_factory: Callable[..., Any] | None = None,
    prepare: Callable[..., tuple[Path, Path, dict[str, Any], tuple[Any, ...], str]] = prepare_row,
) -> int:
    """Process one deterministic strided shard with one persistent backend."""

    input_path = args.input.expanduser().resolve(strict=True)
    rows = _iter_jsonl(input_path)
    if args.num_rows != len(rows):
        raise GokuFullMotionQwenV16Error(
            f"--num-rows={args.num_rows} differs from input rows={len(rows)}"
        )
    worker_index = getattr(args, "worker_index", None)
    num_workers = getattr(args, "num_workers", None)
    if type(worker_index) is not int or type(num_workers) is not int:
        raise GokuFullMotionQwenV16Error(
            "worker mode requires integer --worker-index and --num-workers"
        )
    if not 1 <= num_workers <= args.num_rows:
        raise GokuFullMotionQwenV16Error(
            "num_workers must be in [1, num_rows]"
        )
    if not 0 <= worker_index < num_workers:
        raise GokuFullMotionQwenV16Error(
            "worker_index must be in [0, num_workers)"
        )
    assigned_indices = list(range(worker_index, args.num_rows, num_workers))
    if not assigned_indices:
        raise GokuFullMotionQwenV16Error("worker owns no input rows")

    original_factory = backend_factory or LocalQwenBackend
    shared_backend: Any | None = None

    def persistent_factory(**kwargs: Any) -> Any:
        nonlocal shared_backend
        if shared_backend is None:
            shared_backend = original_factory(**kwargs)
            _reject_backend_cpu_or_disk_offload(shared_backend)
        return shared_backend

    processed = 0
    for row_index in assigned_indices:
        row_args = argparse.Namespace(**vars(args))
        row_args.row_index = row_index
        status = run_one(
            row_args,
            backend_factory=persistent_factory,
            prepare=prepare,
            loaded_rows=rows,
        )
        processed += 1
        if status != 0:
            print(
                "[goku-full-motion-qwen-v16-worker] "
                f"worker={worker_index}/{num_workers} stopped_at={row_index} "
                f"processed={processed}/{len(assigned_indices)} status={status}",
                flush=True,
            )
            return status
    print(
        "[goku-full-motion-qwen-v16-worker] "
        f"worker={worker_index}/{num_workers} indices={assigned_indices} "
        f"processed={processed} backend_loaded={shared_backend is not None}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", type=Path, required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--row-index", type=int)
    selector.add_argument("--worker-index", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--num-rows", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--nframes", type=int, default=DEFAULT_NFRAMES)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--tile-width", type=int, default=DEFAULT_TILE_WIDTH)
    parser.add_argument("--mosaic-columns", type=int, default=DEFAULT_MOSAIC_COLUMNS)
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "sdpa", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker_index is not None:
        return run_worker(args)
    if args.num_workers is not None:
        raise GokuFullMotionQwenV16Error(
            "--num-workers is only valid with --worker-index"
        )
    return run_one(args)


if __name__ == "__main__":
    raise SystemExit(main())
