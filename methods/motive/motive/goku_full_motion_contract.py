"""Closed, fail-closed contracts for first-frame full-motion editing.

The Wan target generator sees the exact initial frame, not the future source
trajectory.  A usable pseudo-edit must therefore spell out the target motion
for every independently moving source unit and for the camera.  This module
contains no model code.  It validates the immutable source census and target
plan that later Qwen, release, generation, and post-generation stages bind.

The contract deliberately rejects crowds and unresolved motion.  It supports
one to three independently referable dynamic units, optional salient static
people or animals, and exactly one explicit camera record.  Every dynamic unit
is changed substantively.  A source base motion may coexist with a novel action
only when that base is written literally.  References to ``the original/source
motion`` or ``the original path`` remain forbidden shortcuts.  Under a
``replace`` relation, a locomotion continuation is valid only after the same
target prose has first established that new locomotion; otherwise it is a
shared source base and must use the explicit shared-base structure.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Mapping


SOURCE_CENSUS_SCHEMA = "motive-goku-full-motion-source-census-v2"
CLIP_SCHEMA = "motive-goku-full-motion-clip-v1"
SOURCE_I0_ENTITY_SCHEMA = "motive-goku-full-motion-i0-entity-v2"
SOURCE_MOTION_COMPONENT_SCHEMA = "motive-goku-full-motion-source-component-v2"
SOURCE_INVENTORY_ALIGNMENT_SCHEMA = (
    "motive-goku-full-motion-source-inventory-alignment-v4"
)
SOURCE_DYNAMIC_UNIT_SCHEMA = "motive-goku-full-motion-source-unit-v2"
SOURCE_STATIC_ENTITY_SCHEMA = (
    "motive-goku-full-motion-source-static-entity-v2"
)
# Compatibility names retain the original field/API spelling.  In this v2
# contract ``static_salient_people`` and ``static_person_targets`` cover both
# salient people *and animals*.  ``entity_type`` disambiguates them, and the
# legacy ``static_person_NN`` ID namespace is only an opaque stable identifier.
SOURCE_STATIC_PERSON_SCHEMA = SOURCE_STATIC_ENTITY_SCHEMA
SOURCE_CAMERA_SCHEMA = "motive-goku-full-motion-source-camera-v1"
MOTION_EVIDENCE_SCHEMA = "motive-goku-full-motion-evidence-v1"

TARGET_PLAN_SCHEMA = "motive-goku-full-motion-target-plan-v2"
TARGET_COMPONENT_DISPOSITION_SCHEMA = (
    "motive-goku-full-motion-target-component-disposition-v2"
)
TARGET_DYNAMIC_UNIT_SCHEMA = "motive-goku-full-motion-target-unit-v2"
TARGET_STATIC_ENTITY_SCHEMA = (
    "motive-goku-full-motion-target-static-entity-v2"
)
TARGET_STATIC_PERSON_SCHEMA = TARGET_STATIC_ENTITY_SCHEMA
TARGET_CAMERA_SCHEMA = "motive-goku-full-motion-target-camera-v2"
TARGET_PRESERVATION_SCHEMA = "motive-goku-full-motion-preservation-v1"
TARGET_COVERAGE_SCHEMA = "motive-goku-full-motion-coverage-v1"
COVERAGE_CRITIC_SCHEMA = "motive-goku-full-motion-coverage-critic-v1"

CONTRACT_SCHEMA = "motive-goku-full-motion-contract-v2"
CONTRACT_POLICY = (
    "dual-source-inventory-all-components-i0-registry-explicit-camera-v2"
)

# Model output canonicalization is intentionally narrower than semantic repair.
# It may only re-derive fields that are redundant with already-present,
# authoritative structure.  In particular, it never invents an entity, unit,
# motion component, target, or camera trajectory.
MODEL_OUTPUT_CANONICALIZATION_RECEIPT_SCHEMA = (
    "motive-goku-full-motion-model-output-canonicalization-receipt-v1"
)
MODEL_OUTPUT_CANONICALIZATION_POLICY = (
    "safe-redundancy-only-no-semantic-repair-v6"
)

FRAME_COUNT = 81
FPS = "25/1"
TIMELINE_SPAN_SECONDS = 3.2
MAX_DYNAMIC_UNITS = 3
MAX_STATIC_SALIENT_PEOPLE = 3
MAX_STATIC_SALIENT_ENTITIES = MAX_STATIC_SALIENT_PEOPLE
# Stable references are authoritative identity text and are copied byte for
# byte through source census, target planning, release, and generation.  Do
# not truncate them during canonicalization: allow enough room for two similar
# actors to remain uniquely I0-grounded while retaining a finite hard bound.
MAX_STABLE_REFERENCE_CHARS = 256

ENTITY_TYPES = frozenset(
    {
        "person",
        "animal",
        "vehicle",
        "rigid_object",
        "rider_vehicle_system",
        "articulated_object",
        "machine",
        "fluid_or_emitter",
        "coherent_group",
    }
)
I0_ENTITY_ROLES = frozenset(
    {"dynamic_subject", "static_salient", "passive_interaction_object"}
)
VIEWER_REGIONS = frozenset(
    {
        "upper_left",
        "upper_center",
        "upper_right",
        "center_left",
        "center",
        "center_right",
        "lower_left",
        "lower_center",
        "lower_right",
    }
)
MOTION_COMPONENT_TYPES = frozenset(
    {
        "locomotion",
        "body_pose",
        "gesture",
        "head_or_gaze",
        "object_interaction",
        "vehicle_motion",
        "articulation",
        "emission_or_fluid",
        "other_visible_motion",
    }
)
COMPONENT_DISPOSITIONS = frozenset({"suppress", "explicit_shared_base"})
CAMERA_MOTION_CLASSES = frozenset(
    {
        "locked_off",
        "pan_left",
        "pan_right",
        "tilt_up",
        "tilt_down",
        "zoom_in",
        "zoom_out",
        "dolly_in",
        "dolly_out",
        "truck_left",
        "truck_right",
        "orbit_left",
        "orbit_right",
        "compound_motion",
    }
)
MOTION_RELATIONS = frozenset(
    {"replace", "explicit_shared_base_with_novel_action"}
)
CAMERA_RELATIONS = frozenset({"preserve_static", "replace_motion"})

_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SIGNATURE_RE = re.compile(r"[a-z][a-z0-9_]{1,95}\Z")
_DYNAMIC_ID_RE = re.compile(r"unit_(\d{2})\Z")
_STATIC_ID_RE = re.compile(r"static_person_(\d{2})\Z")
_ENTITY_ID_RE = re.compile(r"entity_(\d{2})\Z")
_COMPONENT_ID_RE = re.compile(r"component_(\d{2})\Z")
_ENTITY_MARKER_RE = re.compile(r"\[\[(entity_\d{2})\]\]")
_INTERACTION_VERB_RE = re.compile(
    r"\b(?:pick(?:s|ed|ing)?\s+up|grab(?:s|bed|bing)?|grasp(?:s|ed|ing)?|"
    r"carry|carries|carried|carrying|catch(?:es|ing)?|caught|"
    r"throw(?:s|ing)?|threw|kick(?:s|ed|ing)?|"
    r"push(?:es|ed|ing)?|pull(?:s|ed|ing)?|drag(?:s|ged|ging)?|"
    r"set(?:s|ting)?\s+down|hand(?:s|ed|ing)?\s+over|"
    r"take(?:s|n|ing)?|took|drop(?:s|ped|ping)?)\b",
    re.IGNORECASE,
)

# Executable target prose must describe an absolute, self-contained target
# trajectory.  Reject only explicit shortcuts to unavailable source-future
# motion.  I0-grounded wording such as ``current pose`` and ordinary target
# constraints such as ``keep the torso upright`` remain valid.
_FORBIDDEN_TARGET_PATTERNS = (
    # An I2V target cannot refer to an unseen source-video continuation through
    # generic anaphora.  These shortcuts used to slip past the narrower
    # ``same/original/source motion`` checks (for example, ``keep doing what he
    # does while nodding``), even though they do not specify an executable
    # absolute trajectory from I0.
    re.compile(
        r"\b(?:keep(?:s|ing)?|kept|continu(?:e|es|ed|ing)|carry|carries|"
        r"carried|carrying)\s+(?:on\s+)?(?:doing\s+)?"
        r"(?:what(?:ever)?\s+(?:he|she|it|they|the\s+(?:person|animal|"
        r"subject|actor))\s+(?:does|do|did|is\s+doing|are\s+doing)|"
        r"it|that|this|the\s+same\s+thing)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:keep(?:s|ing)?|kept)\s+(?:it|that|this)\s+up\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:ongoing|current|existing|previous)\s+"
        r"(?:action|motion|movement|trajectory|gesture|activity|behavior)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:do|does|did|doing)\s+(?:so|it|that|the\s+same)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:continu(?:e|es|ed|ing)|retain(?:s|ed|ing)?)\b\s+"
        r"(?:it|that|this|them|the\s+(?:(?:motion|action|trajectory)|"
        r"(?:same|original|source|existing|previous)\s+"
        r"(?:[a-z][a-z'-]*\s+){0,2}(?:motion|action|trajectory)))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:same|original|source|existing|previous)\s+"
        r"(?:[a-z][a-z'-]*\s+){0,2}"
        r"(?:motion|action|trajectory)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:as|like)\s+in\s+(?:the\s+)?(?:source|original)"
        r"(?:\s+video)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:original|source|existing|previous)\s+"
        r"(?:[a-z][a-z'-]*\s+){0,2}"
        r"(?:path|route|course|direction|gait|pace|speed|movement|"
        r"locomotion|travel|track|line)\b",
        re.IGNORECASE,
    ),
)

# ``replace`` means that no source locomotion base survives.  These patterns
# identify prose that instead preserves an already-running walking, riding, or
# other translational base.  A later continuation is still legal when the
# prefix of the *same target prose* explicitly starts or absolutely establishes
# the new locomotion first (for example, "begin trotting right, then continue
# trotting right").  This keeps target-internal chronology while rejecting
# first-use shortcuts such as "the bike continues moving leftward".
_LOCOMOTION_PREDICATE = (
    r"(?:accelerat(?:e|es|ed|ing)|advanc(?:e|es|ed|ing)|"
    r"ascend(?:s|ed|ing)?|canter(?:s|ed|ing)?|climb(?:s|ed|ing)?|"
    r"crawl(?:s|ed|ing)?|cycl(?:e|es|ed|ing)|descend(?:s|ed|ing)?|"
    r"driv(?:e|es|en|ing)|fly|flies|flying|gallop(?:s|ed|ing)?|"
    r"glid(?:e|es|ed|ing)|jog(?:s|ged|ging)?|mov(?:e|es|ed|ing)|"
    r"pedal(?:s|ed|ing)?|proceed(?:s|ed|ing)?|"
    r"rid(?:e|es|ing|den)|roll(?:s|ed|ing)?|run|runs|running|"
    r"sail(?:s|ed|ing)?|skate(?:s|d|ing)?|ski(?:s|ed|ing)?|"
    r"slid(?:e|es|ing)|swim|swims|swimming|travel(?:s|ed|ing|led|ling)?|"
    r"trot(?:s|ted|ting)?|walk(?:s|ed|ing)?)"
)
_CONTINUING_LOCOMOTION_PATTERNS = (
    re.compile(
        rf"\b(?:continu(?:e|es|ed|ing)|keep(?:s|ing)?|kept)\s+"
        rf"(?:(?:on|to)\s+)?{_LOCOMOTION_PREDICATE}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:continu(?:e|es|ed|ing)|maintain(?:s|ed|ing)?)\s+"
        r"(?:its|their|his|her|the|a|an)\s+"
        r"(?:[a-z][a-z'-]*\s+){0,3}"
        r"(?:path|route|course|trajectory|gait|pace|movement|motion|"
        r"locomotion|ride|walk|travel)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:keep(?:s|ing)?|kept)\s+"
        r"(?:the|a|an|their|his|her|its)\s+"
        r"(?:[a-z0-9][a-z0-9'-]*\s+){0,5}"
        r"(?:bike|bicycle|motorcycle|vehicle|car|truck|horse|dog|"
        r"animal|person|rider|runner|skater|boat|aircraft|subject|actor)\s+"
        rf"{_LOCOMOTION_PREDICATE}\b",
        re.IGNORECASE,
    ),
)
_TARGET_LOCOMOTION_ESTABLISHMENT_PATTERNS = (
    re.compile(
        rf"\b(?:begin|begins|began|beginning|start|starts|started|starting)\s+"
        rf"(?:to\s+)?{_LOCOMOTION_PREDICATE}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_LOCOMOTION_PREDICATE}\b"
        r"(?:\s+[a-z][a-z'-]*){0,4}\s+"
        r"(?:leftward|rightward|forwards?|backwards?|upward|downward|"
        r"toward|towards|along|across|around|through|into|onto|away|past)\b",
        re.IGNORECASE,
    ),
)
_DIRECTIONAL_LOCOMOTION_IN_PLACE_RE = re.compile(
    rf"\b{_LOCOMOTION_PREDICATE}\b"
    r"(?:\s+[a-z][a-z'-]*){0,4}\s+"
    r"(?:forwards?|backwards?|leftward|rightward)"
    r"(?:\s+(?:steadily|slowly|quickly|rapidly|continuously)){0,2}\s+"
    r"(?:(?:while|but|yet|and)\s+"
    r"(?:remain(?:s|ed|ing)?|stay(?:s|ed|ing)?)\s+"
    r"(?:(?:completely|entirely|fully|stationary)\s+)?)?"
    r"in\s+place\b",
    re.IGNORECASE,
)
_UNCERTAIN_TEXT_RE = re.compile(
    r"\b(?:unclear|unknown|possibly|perhaps|maybe|might|could be)\b",
    re.IGNORECASE,
)
_PURE_NEGATIVE_MOTION_RE = re.compile(
    r"\b(?:"
    r"(?:no|without)\s+"
    r"(?:(?:visible|significant|substantive|independent)\s+)?"
    r"(?:(?:head|gaze|hand|arm|finger|torso|body|pose|gesture|vehicle|wheel)\s+)?"
    r"(?:motion|movement|action|gestures?(?:\s+changes?)?|change|"
    r"articulation|turn(?:s|ing)?|rotation|reorientation|adjustment|deviation)"
    r"|(?:remain|remains|remained|remaining|stay|stays|stayed|staying)"
    r"\s+(?:completely\s+|entirely\s+|fully\s+)?"
    r"(?:still|static|stationary|motionless|unchanged)"
    r"|(?:is|are|was|were)\s+(?:"
    r"(?:completely|entirely|fully)\s+still"
    r"|(?:completely\s+|entirely\s+|fully\s+)?"
    r"(?:static|stationary|motionless|unchanged)"
    r")"
    r")\b",
    re.IGNORECASE,
)
_POSITIVE_MOTION_CUE_RE = re.compile(
    r"\b(?:move|moves|moving|moved|walk|walks|walking|run|runs|running|"
    r"raise|raises|raising|lower|lowers|lowering|turn|turns|turning|"
    r"wave|waves|waving|gesturing|"
    r"shift|shifts|shifting|bend|bends|bending|extend|extends|extending|"
    r"retract|retracts|retracting|open|opens|opening|close|closes|closing|"
    r"nod|nods|nodding|tilt|tilts|tilting|rotate|rotates|rotating|"
    r"jump|jumps|jumping|pour|pours|pouring|ride|rides|riding|drive|drives|"
    r"driving|travel|travels|traveling|approach|approaches|approaching|"
    r"depart|departs|departing|swing|swings|swinging|step|steps|stepping|"
    r"lift|lifts|lifting|drop|drops|dropping|reach|reaches|reaching|"
    r"grab|grabs|grabbing|trot|trots|trotting|gallop|gallops|galloping)\b",
    re.IGNORECASE,
)
_NEGATIVE_MOTION_SIGNATURE_RE = re.compile(
    r"(?:^|_)(?:"
    r"no_(?:visible_)?(?:motion|movement|action|gesture_change|change|articulation)"
    r"|(?:remain|remains|stay|stays)(?:_[a-z0-9]+){0,3}_"
    r"(?:still|static|stationary|motionless|unchanged)"
    r"|(?:still|static|stationary|motionless|unchanged)"
    r"|(?:stable|steady|fixed)(?:_[a-z0-9]+){0,2}_"
    r"(?:posture|pose|gaze|head|grip|hold|orientation|position|stance)"
    r"|(?:posture|pose|gaze|head|grip|hold|orientation|position|stance)_"
    r"(?:stable|steady|fixed)(?:_[a-z0-9]+){0,2}"
    r")(?:_|$)"
)

# A deliberately closed, conservative semantic core derived from executable
# prose.  It is not intended to recognize every conceivable action: unknown
# actions fail closed during curation.  Its purpose is to prevent a planner
# from bypassing novelty checks by changing a free-form signature and
# paraphrasing the same source action (``raises a gloved hand into a hand
# sign`` -> ``moves the glove upward and shapes a hand sign``).
_MOTION_PRIMITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "limb_raise",
        re.compile(
            r"\b(?:rais(?:e|es|ed|ing)|lift(?:s|ed|ing)?|hoist(?:s|ed|ing)?)\b"
            r"(?:\s+[a-z0-9][a-z0-9'-]*){0,5}\s+"
            r"\b(?:hand|arm|glove|paw|forelimb|leg|foot)\b|"
            r"\b(?:move|moves|moved|moving)\b"
            r"(?:\s+[a-z0-9][a-z0-9'-]*){0,5}\s+"
            r"\b(?:hand|arm|glove|paw|forelimb|leg|foot)\b\s+upward\b|"
            r"\b(?:hand|arm|glove|paw|forelimb|leg|foot)\b"
            r"(?:\s+\([a-z][a-z -]*\))?"
            r"(?:\s+[a-z0-9][a-z0-9'-]*){0,3}\s+"
            r"\b(?:rais(?:e|es|ed|ing)|lift(?:s|ed|ing)?|"
            r"hoist(?:s|ed|ing)?|rises|rose|rising|moves?\s+upward)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "limb_lower",
        re.compile(
            r"\b(?:lower(?:s|ed|ing)?|drop(?:s|ped|ping)?)\b"
            r"(?:\s+[a-z0-9][a-z0-9'-]*){0,5}\s+"
            r"\b(?:hand|arm|glove|paw|forelimb|leg|foot|fingers?)\b|"
            r"\b(?:hand|arm|glove|paw|forelimb|leg|foot)\b"
            r"(?:\s+[a-z0-9][a-z0-9'-]*){0,3}\s+"
            r"\b(?:descends|moves?\s+downward)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "open_hand",
        re.compile(
            r"\b(?:open(?:s|ed|ing)?|unfurl(?:s|ed|ing)?|spread(?:s|ing)?)\b"
            r"(?:\s+[a-z0-9][a-z0-9'-]*){0,4}\s+"
            r"\b(?:hand|palm|fingers?|paw)\b|\bopen[- ]palm\b",
            re.IGNORECASE,
        ),
    ),
    (
        "close_hand",
        re.compile(
            r"\b(?:clos(?:e|es|ed|ing)|clench(?:es|ed|ing)?|curl(?:s|ed|ing)?)\b"
            r"(?:\s+[a-z0-9][a-z0-9'-]*){0,4}\s+"
            r"\b(?:hand|palm|fingers?|fist|paw)\b|\bmake(?:s|d|ing)?\s+a\s+fist\b",
            re.IGNORECASE,
        ),
    ),
    ("wave", re.compile(r"\bwave(?:s|d|ing)?\b|\bside[- ]to[- ]side\s+waves?\b", re.IGNORECASE)),
    ("peace_sign", re.compile(r"\b(?:peace|v)[- ]sign\b|\btwo[- ]finger(?:ed)?\s+(?:sign|gesture)\b", re.IGNORECASE)),
    ("hand_sign", re.compile(r"\bhand[- ]sign\b|\bsign\s+gesture\b|\bshap(?:e|es|ed|ing)\b(?:\s+[a-z0-9][a-z0-9'-]*){0,4}\s+\b(?:hand[- ]?)?sign\b", re.IGNORECASE)),
    ("point", re.compile(r"\bpoint(?:s|ed|ing)?\b", re.IGNORECASE)),
    ("clap", re.compile(r"\bclap(?:s|ped|ping)?\b", re.IGNORECASE)),
    ("reach", re.compile(r"\breach(?:es|ed|ing)?\b", re.IGNORECASE)),
    ("head_nod", re.compile(r"\bnod(?:s|ded|ding)?\b", re.IGNORECASE)),
    ("head_shake", re.compile(r"\bshake(?:s|n|shook|shaking)?\b(?:\s+[a-z0-9][a-z0-9'-]*){0,3}\s+\bhead\b|\bhead\b(?:\s+[a-z0-9][a-z0-9'-]*){0,3}\s+\bshakes?\b", re.IGNORECASE)),
    ("head_turn", re.compile(r"\bturn(?:s|ed|ing)?\b(?:\s+[a-z0-9][a-z0-9'-]*){0,3}\s+\b(?:head|face|gaze)\b|\b(?:head|face|gaze)\b(?:\s+[a-z0-9][a-z0-9'-]*){0,3}\s+\bturns?\b", re.IGNORECASE)),
    ("body_pivot", re.compile(r"\b(?:pivot|rotate|spin|turn)(?:s|ed|ing)?\b", re.IGNORECASE)),
    ("bow", re.compile(r"\bbow(?:s|ed|ing)?\b", re.IGNORECASE)),
    ("sit", re.compile(r"\b(?:sit|sits|sat|sitting)\b", re.IGNORECASE)),
    ("stand", re.compile(r"\b(?:stand|stands|stood|standing|rise|rises|rose|rising)\b", re.IGNORECASE)),
    ("crouch", re.compile(r"\b(?:crouch|squat)(?:es|ed|ing|s)?\b", re.IGNORECASE)),
    ("lean", re.compile(r"\blean(?:s|ed|ing)?\b", re.IGNORECASE)),
    ("jump", re.compile(r"\b(?:jump|hop|leap)(?:s|ped|ping|ed|ing)?\b", re.IGNORECASE)),
    ("dance", re.compile(r"\bdanc(?:e|es|ed|ing)\b|\bcartwheel(?:s|ed|ing)?\b", re.IGNORECASE)),
    ("walk", re.compile(r"\bwalk(?:s|ed|ing)?\b", re.IGNORECASE)),
    ("run", re.compile(r"\b(?:run|runs|ran|running|jog|jogs|jogged|jogging)\b", re.IGNORECASE)),
    ("trot", re.compile(r"\btrot(?:s|ted|ting)?\b", re.IGNORECASE)),
    ("gallop", re.compile(r"\bgallop(?:s|ed|ing)?\b", re.IGNORECASE)),
    ("crawl", re.compile(r"\bcrawl(?:s|ed|ing)?\b", re.IGNORECASE)),
    ("roll", re.compile(r"\broll(?:s|ed|ing)?\b", re.IGNORECASE)),
    ("slide", re.compile(r"\bslid(?:e|es|ing)\b", re.IGNORECASE)),
    ("kick", re.compile(r"\bkick(?:s|ed|ing)?\b", re.IGNORECASE)),
    (
        "mount",
        re.compile(
            r"\bmount(?:s|ed|ing)?\b|"
            r"\b(?:jump|leap)(?:s|ed|ing|ped|ping)?\s+onto\b",
            re.IGNORECASE,
        ),
    ),
    (
        "dribble",
        re.compile(r"\bdribbl(?:e|es|ed|ing)\b", re.IGNORECASE),
    ),
    (
        "ball_strike",
        re.compile(
            r"\b(?:kick(?:s|ed|ing)?|strik(?:e|es|ing)|struck|"
            r"hit(?:s|ting)?|tap(?:s|ped|ping)?)\b"
            r"(?:\s+[a-z0-9][a-z0-9'-]*){0,5}\s+"
            r"\b(?:ball|puck)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "tackle",
        re.compile(
            r"\b(?:tackl(?:e|es|ed|ing)|intercept(?:s|ed|ing)?|"
            r"dispossess(?:es|ed|ing)?)\b",
            re.IGNORECASE,
        ),
    ),
    ("pick_up", re.compile(r"\b(?:pick(?:s|ed|ing)?\s+up|grab(?:s|bed|bing)?|grasp(?:s|ed|ing)?)\b", re.IGNORECASE)),
    ("throw", re.compile(r"\b(?:throw|throws|threw|throwing)\b", re.IGNORECASE)),
    ("catch", re.compile(r"\b(?:catch|catches|caught|catching)\b", re.IGNORECASE)),
    ("drop_object", re.compile(r"\bdrop(?:s|ped|ping)?\b", re.IGNORECASE)),
    ("carry", re.compile(r"\b(?:carry|carries|carried|carrying)\b", re.IGNORECASE)),
    ("push", re.compile(r"\bpush(?:es|ed|ing)?\b", re.IGNORECASE)),
    ("pull", re.compile(r"\bpull(?:s|ed|ing)?\b", re.IGNORECASE)),
    ("pour", re.compile(r"\bpour(?:s|ed|ing)?\b", re.IGNORECASE)),
    ("eat", re.compile(r"\b(?:eat|eats|ate|eating|bite|bites|biting)\b", re.IGNORECASE)),
    ("drink", re.compile(r"\b(?:drink|drinks|drank|drinking|sip|sips|sipped|sipping)\b", re.IGNORECASE)),
    ("swim", re.compile(r"\b(?:swim|swims|swam|swimming)\b", re.IGNORECASE)),
    ("fly", re.compile(r"\b(?:fly|flies|flew|flying)\b", re.IGNORECASE)),
    ("ride", re.compile(r"\b(?:ride|rides|rode|riding)\b", re.IGNORECASE)),
    ("drive", re.compile(r"\b(?:drive|drives|drove|driving)\b", re.IGNORECASE)),
    ("steer", re.compile(r"\bsteer(?:s|ed|ing)?\b", re.IGNORECASE)),
    ("articulation_open", re.compile(r"\bopen(?:s|ed|ing)?\b(?:\s+[a-z0-9][a-z0-9'-]*){0,3}\s+\b(?:door|lid|gate|drawer|mouth|beak)\b", re.IGNORECASE)),
    ("articulation_close", re.compile(r"\bclos(?:e|es|ed|ing)\b(?:\s+[a-z0-9][a-z0-9'-]*){0,3}\s+\b(?:door|lid|gate|drawer|mouth|beak)\b", re.IGNORECASE)),
    ("emit", re.compile(r"\b(?:emit|spray|spout|release|kick\s+up)(?:s|ted|ting|ed|ing)?\b(?:\s+[a-z0-9][a-z0-9'-]*){0,3}\s+\b(?:smoke|dust|water|liquid|steam|sparks?)\b", re.IGNORECASE)),
    ("direction_left", re.compile(r"\b(?:leftward|to\s+the\s+left)\b", re.IGNORECASE)),
    ("direction_right", re.compile(r"\b(?:rightward|to\s+the\s+right)\b", re.IGNORECASE)),
    ("direction_forward", re.compile(r"\bforwards?\b", re.IGNORECASE)),
    ("direction_backward", re.compile(r"\bbackwards?\b", re.IGNORECASE)),
    (
        "direction_up",
        re.compile(
            r"\b(?:upwards?|rais(?:e|es|ed|ing)|lift(?:s|ed|ing)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "direction_down",
        re.compile(
            r"\b(?:downwards?|lower(?:s|ed|ing)?|drop(?:s|ped|ping)?)\b",
            re.IGNORECASE,
        ),
    ),
    ("clockwise", re.compile(r"\bclockwise\b", re.IGNORECASE)),
    ("counterclockwise", re.compile(r"\b(?:counterclockwise|anticlockwise)\b", re.IGNORECASE)),
)

# Source-inventory alignment compares two independent descriptions of the same
# pixels.  The general novelty vocabulary above is intentionally sensitive to
# every executable target verb, but that sensitivity is wrong for census
# alignment: a locomotion description may mention a lowered head, a forward
# reaching gait leg, or a forward-facing gaze without changing the actor's
# trajectory.  Keep this normalization private to alignment so target novelty
# remains fail-closed.
_ALIGNMENT_DIRECTION_PRIMITIVES = frozenset(
    {
        "direction_left",
        "direction_right",
        "direction_forward",
        "direction_backward",
        "direction_up",
        "direction_down",
    }
)
_ALIGNMENT_LOCOMOTION_PRIMITIVES = frozenset(
    {
        "body_pivot",
        "jump",
        "dance",
        "walk",
        "run",
        "trot",
        "gallop",
        "crawl",
        "roll",
        "slide",
        "swim",
        "fly",
        "ride",
        "drive",
        "clockwise",
        "counterclockwise",
    }
)
_ALIGNMENT_VEHICLE_MOTION_PRIMITIVES = frozenset(
    {
        "body_pivot",
        "jump",
        "roll",
        "drive",
        "steer",
        "clockwise",
        "counterclockwise",
    }
)
_ALIGNMENT_COMPONENT_PRIMITIVES = {
    "locomotion": _ALIGNMENT_LOCOMOTION_PRIMITIVES,
    "body_pose": frozenset(
        {
            "body_pivot",
            "bow",
            "sit",
            "stand",
            "crouch",
            "lean",
            "jump",
            "dance",
            "kick",
            "clockwise",
            "counterclockwise",
        }
    ),
    "gesture": frozenset(
        {
            "limb_raise",
            "limb_lower",
            "open_hand",
            "close_hand",
            "wave",
            "peace_sign",
            "hand_sign",
            "point",
            "clap",
            "reach",
            "kick",
        }
    ),
    "head_or_gaze": frozenset(
        {"head_nod", "head_shake", "head_turn", "bow"}
    ),
    "object_interaction": frozenset(
        {
            "reach",
            "kick",
            "mount",
            "dribble",
            "ball_strike",
            "tackle",
            "pick_up",
            "throw",
            "catch",
            "drop_object",
            "carry",
            "push",
            "pull",
            "pour",
            "eat",
            "drink",
            "articulation_open",
            "articulation_close",
        }
    ),
    "vehicle_motion": _ALIGNMENT_VEHICLE_MOTION_PRIMITIVES,
    "articulation": frozenset(
        {
            "limb_raise",
            "limb_lower",
            "open_hand",
            "close_hand",
            "body_pivot",
            "roll",
            "kick",
            "articulation_open",
            "articulation_close",
            "clockwise",
            "counterclockwise",
        }
    ),
    "emission_or_fluid": frozenset({"emit", "pour"}),
    "other_visible_motion": frozenset(
        name for name, _ in _MOTION_PRIMITIVE_PATTERNS
    )
    - _ALIGNMENT_DIRECTION_PRIMITIVES,
}
_ALIGNMENT_TRANSLATION_VERB = (
    r"(?:mov(?:e|es|ed|ing)|travel(?:s|ed|ing)?|walk(?:s|ed|ing)?|"
    r"run(?:s|ning)?|ran|jog(?:s|ged|ging)?|trot(?:s|ted|ting)?|"
    r"gallop(?:s|ed|ing)?|crawl(?:s|ed|ing)?|roll(?:s|ed|ing)?|"
    r"slid(?:e|es|ing)|swim(?:s|ming)?|swam|fly|flies|flew|flying|"
    r"ride|rides|rode|riding|drive|drives|drove|driving)"
)
_ALIGNMENT_TRAJECTORY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "trajectory_right",
        re.compile(
            rf"\b(?:rightward|from\s+(?:the\s+)?left\s+to\s+(?:the\s+)?right|"
            rf"{_ALIGNMENT_TRANSLATION_VERB}(?:\s+[a-z0-9][a-z0-9'-]*){{0,4}}"
            rf"\s+to\s+(?:the\s+)?right)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "trajectory_left",
        re.compile(
            rf"\b(?:leftward|from\s+(?:the\s+)?right\s+to\s+(?:the\s+)?left|"
            rf"{_ALIGNMENT_TRANSLATION_VERB}(?:\s+[a-z0-9][a-z0-9'-]*){{0,4}}"
            rf"\s+to\s+(?:the\s+)?left)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "trajectory_forward",
        re.compile(
            rf"\b(?:{_ALIGNMENT_TRANSLATION_VERB}"
            rf"(?:\s+[a-z0-9][a-z0-9'-]*){{0,4}}\s+forwards?|"
            rf"forwards?\s+{_ALIGNMENT_TRANSLATION_VERB})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "trajectory_backward",
        re.compile(
            rf"\b(?:{_ALIGNMENT_TRANSLATION_VERB}"
            rf"(?:\s+[a-z0-9][a-z0-9'-]*){{0,4}}\s+backwards?|"
            rf"backwards?\s+{_ALIGNMENT_TRANSLATION_VERB})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "trajectory_up",
        re.compile(
            rf"\b(?:{_ALIGNMENT_TRANSLATION_VERB}"
            rf"(?:\s+[a-z0-9][a-z0-9'-]*){{0,4}}\s+upwards?|"
            rf"upwards?\s+{_ALIGNMENT_TRANSLATION_VERB})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "trajectory_down",
        re.compile(
            rf"\b(?:{_ALIGNMENT_TRANSLATION_VERB}"
            rf"(?:\s+[a-z0-9][a-z0-9'-]*){{0,4}}\s+downwards?|"
            rf"downwards?\s+{_ALIGNMENT_TRANSLATION_VERB})\b",
            re.IGNORECASE,
        ),
    ),
)


class GokuFullMotionContractError(ValueError):
    """A source census or target plan violates the frozen contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the sole canonical JSON encoding used by contract digests."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise GokuFullMotionContractError(
            "contract value is not finite canonical JSON"
        ) from error


def sha256_text(value: str) -> str:
    if type(value) is not str:
        raise GokuFullMotionContractError("SHA-256 text input must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    if set(value) != expected:
        raise GokuFullMotionContractError(
            f"{context} keys differ from closed schema: "
            f"{sorted(set(value) ^ expected)}"
        )


def _closed_keys_with_optional_redundancy(
    value: Mapping[str, Any],
    expected: set[str],
    optional_redundancy: set[str],
    *,
    context: str,
) -> None:
    """Check a closed object before whitelisted redundant fields are filled."""

    actual = set(value)
    extra = actual - expected
    missing_nonredundant = (expected - actual) - optional_redundancy
    if extra or missing_nonredundant:
        raise GokuFullMotionContractError(
            f"{context} keys differ from pre-canonicalization closed schema: "
            f"extra={sorted(extra)!r} "
            f"missing_nonredundant={sorted(missing_nonredundant)!r}"
        )


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GokuFullMotionContractError(f"{context} must be an object")
    return value


def _list(value: Any, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise GokuFullMotionContractError(f"{context} must be an array")
    return value


def _text(
    value: Any,
    *,
    context: str,
    max_length: int = 1000,
) -> str:
    if type(value) is not str or not value:
        raise GokuFullMotionContractError(f"{context} must be a non-empty string")
    if value != value.strip() or "\n" in value or "\r" in value or "\t" in value:
        raise GokuFullMotionContractError(
            f"{context} must be trimmed single-line text"
        )
    if len(value) > max_length:
        raise GokuFullMotionContractError(f"{context} is too long")
    if "  " in value:
        raise GokuFullMotionContractError(
            f"{context} contains non-canonical repeated whitespace"
        )
    return value


def _target_text(
    value: Any,
    *,
    context: str,
) -> str:
    result = _text(value, context=context)
    for pattern in _FORBIDDEN_TARGET_PATTERNS:
        if pattern.search(result):
            raise GokuFullMotionContractError(
                f"{context} depends on ambiguous source-future motion"
            )
    if _DIRECTIONAL_LOCOMOTION_IN_PLACE_RE.search(result):
        raise GokuFullMotionContractError(
            f"{context} contains contradictory directional locomotion "
            "in place"
        )
    return result


def _reject_unestablished_replace_locomotion(
    value: str,
    *,
    context: str,
) -> None:
    """Reject a shared locomotion base hidden inside ``replace`` prose."""

    for continuation_pattern in _CONTINUING_LOCOMOTION_PATTERNS:
        for match in continuation_pattern.finditer(value):
            prefix = value[: match.start()]
            if any(
                pattern.search(prefix)
                for pattern in _TARGET_LOCOMOTION_ESTABLISHMENT_PATTERNS
            ):
                continue
            raise GokuFullMotionContractError(
                f"{context} hides continuing locomotion under replace; use "
                "explicit_shared_base_with_novel_action and "
                "explicit_shared_base_motion"
            )


def _prose_comparison_key(value: str) -> str:
    """Compare prose without treating ordinary sentence endings as motion."""

    return value.rstrip(".!?;:").rstrip().casefold()


def _motion_semantic_primitives(*values: str) -> frozenset[str]:
    """Map executable prose to the closed conservative motion vocabulary."""

    text = ". ".join(values)
    return frozenset(
        name
        for name, pattern in _MOTION_PRIMITIVE_PATTERNS
        if pattern.search(text)
    )


def _alignment_trajectory_primitives(*values: str) -> frozenset[str]:
    """Normalize an actor's principal translation without local-body noise.

    Signatures use underscores while prose commonly uses hyphens or spaces;
    normalize those separators before matching.  The patterns deliberately do
    not treat a bare ``forward``/``lowered`` token as trajectory.  Such words
    routinely describe a gait leg, head pose, or gaze inside an otherwise
    rightward locomotion component.
    """

    text = " ".join(values).replace("_", " ").replace("-", " ")
    return frozenset(
        name
        for name, pattern in _ALIGNMENT_TRAJECTORY_PATTERNS
        if pattern.search(text)
    )


def _source_alignment_component_primitives(
    component: Mapping[str, Any],
) -> frozenset[str]:
    """Return component-conditioned primitives for blind-census agreement.

    Component types are already compared exactly.  This second semantic layer
    distinguishes real action opposites (for example raise/lower or walk/run)
    while excluding incidental pose and gait wording.  In particular,
    locomotion descriptions often mention limb mechanics that are explicitly
    required to remain folded into the locomotion component by the prompt.
    """

    component_type = str(component["component_type"])
    description = str(component["motion_description"])
    signature = str(component["motion_signature"])
    primitives = set(_motion_semantic_primitives(description))
    primitives.difference_update(_ALIGNMENT_DIRECTION_PRIMITIVES)
    primitives.intersection_update(
        _ALIGNMENT_COMPONENT_PRIMITIVES[component_type]
    )
    if component_type == "locomotion":
        primitives.update(
            _alignment_trajectory_primitives(signature, description)
        )
    elif component_type == "vehicle_motion":
        primitives.update(
            _alignment_trajectory_primitives(signature, description)
        )
    return frozenset(primitives)


def _bool(value: Any, *, context: str) -> bool:
    if type(value) is not bool:
        raise GokuFullMotionContractError(f"{context} must be a JSON boolean")
    return value


def _true(value: Any, *, context: str) -> None:
    if _bool(value, context=context) is not True:
        raise GokuFullMotionContractError(f"{context} must be exactly true")


def _false(value: Any, *, context: str) -> None:
    if _bool(value, context=context) is not False:
        raise GokuFullMotionContractError(f"{context} must be exactly false")


def _enum(value: Any, allowed: frozenset[str], *, context: str) -> str:
    result = _text(value, context=context, max_length=128)
    if result not in allowed:
        raise GokuFullMotionContractError(
            f"{context} is outside the closed enum: {result!r}"
        )
    return result


def _signature(value: Any, *, context: str) -> str:
    result = _text(value, context=context, max_length=96)
    if _SIGNATURE_RE.fullmatch(result) is None:
        raise GokuFullMotionContractError(
            f"{context} must be a lower snake-case motion signature"
        )
    if (
        result in {"unclear", "unknown", "remain_still"}
        or _NEGATIVE_MOTION_SIGNATURE_RE.search(result)
    ):
        raise GokuFullMotionContractError(f"{context} is not a dynamic action")
    return result


def _reject_pure_negative_motion_assertion(
    values: list[str], *, context: str
) -> None:
    """Reject evidence that asserts only absence of motion.

    A dynamic component may mention a locally still body part while another
    part moves (for example, ``torso remains still while the hand waves``).
    The positive-motion cue keeps that useful contrast valid.  What is not
    valid is labelling ``no visible gesture change`` as a gesture component.
    """

    joined = " ".join(values)
    negative = _PURE_NEGATIVE_MOTION_RE.search(joined)
    # Do not let a verb inside a negated span (for example ``no significant
    # turning``) satisfy the positive-motion cue.  Mixed descriptions such as
    # ``the torso remains still while the hand waves`` retain the independent
    # positive clause after the negative span is removed.
    positive_domain = _PURE_NEGATIVE_MOTION_RE.sub(" ", joined)
    if negative and not _POSITIVE_MOTION_CUE_RE.search(positive_domain):
        raise GokuFullMotionContractError(
            f"{context} asserts no positive dynamic motion"
        )


def _ordered_unique_strings(
    value: Any,
    *,
    context: str,
    allow_empty: bool,
) -> list[str]:
    raw = _list(value, context=context)
    if not allow_empty and not raw:
        raise GokuFullMotionContractError(f"{context} must not be empty")
    output = [
        _text(item, context=f"{context}[{index}]", max_length=256)
        for index, item in enumerate(raw)
    ]
    if len(set(output)) != len(output):
        raise GokuFullMotionContractError(f"{context} contains duplicates")
    return output


def _positive_seconds(value: Any, *, context: str) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(float(value))
        or not 0.0 < float(value) <= TIMELINE_SPAN_SECONDS
    ):
        raise GokuFullMotionContractError(
            f"{context} must be in (0, {TIMELINE_SPAN_SECONDS}]"
        )
    return float(value)


def _validate_evidence(value: Any, *, context: str) -> dict[str, Any]:
    item = _mapping(value, context=context)
    _exact_keys(
        item,
        {"schema_version", "start_frame", "end_frame", "description"},
        context=context,
    )
    if item.get("schema_version") != MOTION_EVIDENCE_SCHEMA:
        raise GokuFullMotionContractError(f"{context} schema differs")
    start = item.get("start_frame")
    end = item.get("end_frame")
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start <= end < FRAME_COUNT
    ):
        raise GokuFullMotionContractError(
            f"{context} frame interval must lie in [0, {FRAME_COUNT - 1}]"
        )
    description = _text(item.get("description"), context=f"{context}.description")
    if _UNCERTAIN_TEXT_RE.search(description):
        raise GokuFullMotionContractError(
            f"{context}.description is semantically uncertain"
        )
    return dict(item)


def _validate_evidence_list(value: Any, *, context: str) -> list[dict[str, Any]]:
    items = _list(value, context=context)
    if not items:
        raise GokuFullMotionContractError(f"{context} must not be empty")
    validated = [
        _validate_evidence(item, context=f"{context}[{index}]")
        for index, item in enumerate(items)
    ]
    intervals = [
        (int(item["start_frame"]), int(item["end_frame"]))
        for item in validated
    ]
    if intervals != sorted(intervals):
        raise GokuFullMotionContractError(
            f"{context} must be chronologically ordered"
        )
    return validated


def _expected_ids(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_{index:02d}" for index in range(1, count + 1)]


def _i0_bbox(value: Any, *, context: str) -> list[int]:
    bbox = _list(value, context=context)
    if (
        len(bbox) != 4
        or any(type(coordinate) is not int for coordinate in bbox)
        or not (0 <= bbox[0] < bbox[2] <= 1000)
        or not (0 <= bbox[1] < bbox[3] <= 1000)
    ):
        raise GokuFullMotionContractError(
            f"{context} must be integer normalized xyxy in [0, 1000]"
        )
    return list(bbox)


def _viewer_region_for_bbox(bbox: list[int]) -> str:
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    horizontal = "left" if center_x < 1000 / 3 else (
        "right" if center_x >= 2000 / 3 else "center"
    )
    vertical = "upper" if center_y < 1000 / 3 else (
        "lower" if center_y >= 2000 / 3 else "center"
    )
    return "center" if vertical == horizontal == "center" else f"{vertical}_{horizontal}"


def _validate_i0_entity(
    value: Any, *, expected_id: str, context: str
) -> dict[str, Any]:
    entity = _mapping(value, context=context)
    _exact_keys(
        entity,
        {
            "schema_version",
            "entity_id",
            "entity_type",
            "stable_reference",
            "i0_bbox_xyxy_1000",
            "viewer_region",
            "region_ordinal",
            "role",
            "visible_at_i0",
            "reachable_at_i0",
            "confidence",
        },
        context=context,
    )
    if entity.get("schema_version") != SOURCE_I0_ENTITY_SCHEMA:
        raise GokuFullMotionContractError(f"{context} schema differs")
    entity_id = _text(entity.get("entity_id"), context=f"{context}.entity_id")
    if entity_id != expected_id or _ENTITY_ID_RE.fullmatch(entity_id) is None:
        raise GokuFullMotionContractError(
            f"{context}.entity_id must be contiguous and equal {expected_id!r}"
        )
    entity_type = _enum(
        entity.get("entity_type"), ENTITY_TYPES, context=f"{context}.entity_type"
    )
    _text(
        entity.get("stable_reference"),
        context=f"{context}.stable_reference",
        max_length=MAX_STABLE_REFERENCE_CHARS,
    )
    bbox = _i0_bbox(
        entity.get("i0_bbox_xyxy_1000"),
        context=f"{context}.i0_bbox_xyxy_1000",
    )
    viewer_region = _enum(
        entity.get("viewer_region"),
        VIEWER_REGIONS,
        context=f"{context}.viewer_region",
    )
    if viewer_region != _viewer_region_for_bbox(bbox):
        raise GokuFullMotionContractError(
            f"{context}.viewer_region disagrees with normalized I0 bbox center"
        )
    ordinal = entity.get("region_ordinal")
    if type(ordinal) is not int or not 1 <= ordinal <= 24:
        raise GokuFullMotionContractError(
            f"{context}.region_ordinal must be an integer in [1, 24]"
        )
    role = _enum(entity.get("role"), I0_ENTITY_ROLES, context=f"{context}.role")
    _true(entity.get("visible_at_i0"), context=f"{context}.visible_at_i0")
    reachable = _bool(
        entity.get("reachable_at_i0"), context=f"{context}.reachable_at_i0"
    )
    if role == "static_salient" and entity_type not in {"person", "animal"}:
        raise GokuFullMotionContractError(
            f"{context} static_salient entries must be people or animals"
        )
    if role == "passive_interaction_object":
        if entity_type not in {
            "vehicle",
            "rigid_object",
            "articulated_object",
            "machine",
        }:
            raise GokuFullMotionContractError(
                f"{context} passive interaction entry has an invalid entity type"
            )
        if reachable is not True:
            raise GokuFullMotionContractError(
                f"{context} passive interaction object must be reachable at I0"
            )
    if entity.get("confidence") != "high":
        raise GokuFullMotionContractError(
            f"{context}.confidence must be exactly 'high'"
        )
    return dict(entity)


def _validate_motion_component(
    value: Any,
    *,
    expected_id: str,
    owner_entity_id: str,
    registry_by_id: Mapping[str, Mapping[str, Any]],
    context: str,
) -> dict[str, Any]:
    component = _mapping(value, context=context)
    _exact_keys(
        component,
        {
            "schema_version",
            "component_id",
            "component_type",
            "motion_signature",
            "motion_description",
            "dependent_entity_ids",
            "motion_evidence",
        },
        context=context,
    )
    if component.get("schema_version") != SOURCE_MOTION_COMPONENT_SCHEMA:
        raise GokuFullMotionContractError(f"{context} schema differs")
    component_id = _text(
        component.get("component_id"), context=f"{context}.component_id"
    )
    if component_id != expected_id or _COMPONENT_ID_RE.fullmatch(component_id) is None:
        raise GokuFullMotionContractError(
            f"{context}.component_id must be contiguous and equal {expected_id!r}"
        )
    component_type = _enum(
        component.get("component_type"),
        MOTION_COMPONENT_TYPES,
        context=f"{context}.component_type",
    )
    _signature(
        component.get("motion_signature"),
        context=f"{context}.motion_signature",
    )
    description = _text(
        component.get("motion_description"),
        context=f"{context}.motion_description",
    )
    if _UNCERTAIN_TEXT_RE.search(description):
        raise GokuFullMotionContractError(
            f"{context}.motion_description is semantically uncertain"
        )
    dependent = _ordered_unique_strings(
        component.get("dependent_entity_ids"),
        context=f"{context}.dependent_entity_ids",
        allow_empty=True,
    )
    unknown = [entity_id for entity_id in dependent if entity_id not in registry_by_id]
    if unknown:
        raise GokuFullMotionContractError(
            f"{context} contains unknown dependent I0 entity IDs: {unknown!r}"
        )
    if owner_entity_id in dependent:
        raise GokuFullMotionContractError(
            f"{context} cannot list its owner as a dependent entity"
        )
    if component_type == "object_interaction" and not dependent:
        raise GokuFullMotionContractError(
            f"{context} object_interaction must bind a dependent I0 entity"
        )
    validated_evidence = _validate_evidence_list(
        component.get("motion_evidence"), context=f"{context}.motion_evidence"
    )
    _reject_pure_negative_motion_assertion(
        [description], context=f"{context}.motion_description"
    )
    _reject_pure_negative_motion_assertion(
        [str(item["description"]) for item in validated_evidence],
        context=f"{context}.motion_evidence",
    )
    return dict(component)


def _validate_clip(value: Any) -> dict[str, Any]:
    context = "source_census.clip"
    clip = _mapping(value, context=context)
    _exact_keys(
        clip,
        {
            "schema_version",
            "frame_count",
            "fps",
            "timeline_span_seconds",
            "single_continuous_shot",
        },
        context=context,
    )
    if clip.get("schema_version") != CLIP_SCHEMA:
        raise GokuFullMotionContractError(f"{context} schema differs")
    if clip.get("frame_count") != FRAME_COUNT:
        raise GokuFullMotionContractError(
            f"{context}.frame_count must be exactly {FRAME_COUNT}"
        )
    if clip.get("fps") != FPS:
        raise GokuFullMotionContractError(
            f"{context}.fps must be exactly {FPS!r}"
        )
    span = clip.get("timeline_span_seconds")
    if type(span) not in (int, float) or not math.isclose(
        float(span), TIMELINE_SPAN_SECONDS, rel_tol=0.0, abs_tol=1e-9
    ):
        raise GokuFullMotionContractError(
            f"{context}.timeline_span_seconds must be exactly "
            f"{TIMELINE_SPAN_SECONDS}"
        )
    _true(
        clip.get("single_continuous_shot"),
        context=f"{context}.single_continuous_shot",
    )
    return dict(clip)


def _validate_source_dynamic_unit(
    value: Any,
    *,
    expected_id: str,
    registry_by_id: Mapping[str, Mapping[str, Any]],
    context: str,
) -> dict[str, Any]:
    unit = _mapping(value, context=context)
    _exact_keys(
        unit,
        {
            "schema_version",
            "unit_id",
            "entity_id",
            "entity_type",
            "stable_reference",
            "visible_at_i0",
            "independent_motion",
            "i0_state",
            "source_action_signature",
            "source_motion",
            "source_motion_components",
            "motion_evidence",
            "confidence",
        },
        context=context,
    )
    if unit.get("schema_version") != SOURCE_DYNAMIC_UNIT_SCHEMA:
        raise GokuFullMotionContractError(f"{context} schema differs")
    unit_id = _text(unit.get("unit_id"), context=f"{context}.unit_id")
    if unit_id != expected_id or _DYNAMIC_ID_RE.fullmatch(unit_id) is None:
        raise GokuFullMotionContractError(
            f"{context}.unit_id must be contiguous and equal {expected_id!r}"
        )
    entity_id = _text(unit.get("entity_id"), context=f"{context}.entity_id")
    registry_entity = registry_by_id.get(entity_id)
    if registry_entity is None or registry_entity.get("role") != "dynamic_subject":
        raise GokuFullMotionContractError(
            f"{context}.entity_id must bind one dynamic-subject I0 registry entry"
        )
    entity_type = _enum(
        unit.get("entity_type"), ENTITY_TYPES, context=f"{context}.entity_type"
    )
    reference = _text(
        unit.get("stable_reference"),
        context=f"{context}.stable_reference",
        max_length=MAX_STABLE_REFERENCE_CHARS,
    )
    if (
        entity_type != registry_entity.get("entity_type")
        or reference != registry_entity.get("stable_reference")
    ):
        raise GokuFullMotionContractError(
            f"{context} entity type/reference differs from the I0 registry"
        )
    _true(unit.get("visible_at_i0"), context=f"{context}.visible_at_i0")
    _true(
        unit.get("independent_motion"),
        context=f"{context}.independent_motion",
    )
    _text(unit.get("i0_state"), context=f"{context}.i0_state")
    _signature(
        unit.get("source_action_signature"),
        context=f"{context}.source_action_signature",
    )
    source_motion = _text(
        unit.get("source_motion"), context=f"{context}.source_motion"
    )
    if _UNCERTAIN_TEXT_RE.search(source_motion):
        raise GokuFullMotionContractError(
            f"{context}.source_motion is semantically uncertain"
        )
    components = _list(
        unit.get("source_motion_components"),
        context=f"{context}.source_motion_components",
    )
    if not 1 <= len(components) <= 6:
        raise GokuFullMotionContractError(
            f"{context}.source_motion_components must contain one to six components"
        )
    component_ids = _expected_ids("component", len(components))
    validated_components = [
        _validate_motion_component(
            item,
            expected_id=component_ids[index],
            owner_entity_id=entity_id,
            registry_by_id=registry_by_id,
            context=f"{context}.source_motion_components[{index}]",
        )
        for index, item in enumerate(components)
    ]
    signatures = [str(item["motion_signature"]) for item in validated_components]
    if len(set(signatures)) != len(signatures):
        raise GokuFullMotionContractError(
            f"{context}.source_motion_components contain duplicate signatures"
        )
    component_types = [str(item["component_type"]) for item in validated_components]
    if len(set(component_types)) != len(component_types):
        raise GokuFullMotionContractError(
            f"{context}.source_motion_components must use unique component types"
        )
    validated_evidence = _validate_evidence_list(
        unit.get("motion_evidence"), context=f"{context}.motion_evidence"
    )
    _reject_pure_negative_motion_assertion(
        [source_motion], context=f"{context}.source_motion"
    )
    _reject_pure_negative_motion_assertion(
        [str(item["description"]) for item in validated_evidence],
        context=f"{context}.motion_evidence",
    )
    if unit.get("confidence") != "high":
        raise GokuFullMotionContractError(
            f"{context}.confidence must be exactly 'high'"
        )
    return dict(unit)


def _validate_source_static_person(
    value: Any,
    *,
    expected_id: str,
    registry_by_id: Mapping[str, Mapping[str, Any]],
    context: str,
) -> dict[str, Any]:
    person = _mapping(value, context=context)
    _exact_keys(
        person,
        {
            "schema_version",
            "unit_id",
            "entity_id",
            "entity_type",
            "stable_reference",
            "visible_at_i0",
            "i0_state",
            "source_state",
            "motion_evidence",
            "confidence",
        },
        context=context,
    )
    if person.get("schema_version") != SOURCE_STATIC_PERSON_SCHEMA:
        raise GokuFullMotionContractError(f"{context} schema differs")
    unit_id = _text(person.get("unit_id"), context=f"{context}.unit_id")
    if unit_id != expected_id or _STATIC_ID_RE.fullmatch(unit_id) is None:
        raise GokuFullMotionContractError(
            f"{context}.unit_id must be contiguous and equal {expected_id!r}"
        )
    entity_id = _text(person.get("entity_id"), context=f"{context}.entity_id")
    registry_entity = registry_by_id.get(entity_id)
    if registry_entity is None or registry_entity.get("role") != "static_salient":
        raise GokuFullMotionContractError(
            f"{context}.entity_id must bind one static-salient I0 registry entry"
        )
    if person.get("entity_type") not in {"person", "animal"}:
        raise GokuFullMotionContractError(
            f"{context}.entity_type must be exactly 'person' or 'animal'"
        )
    reference = _text(
        person.get("stable_reference"),
        context=f"{context}.stable_reference",
        max_length=MAX_STABLE_REFERENCE_CHARS,
    )
    if (
        person.get("entity_type") != registry_entity.get("entity_type")
        or reference != registry_entity.get("stable_reference")
    ):
        raise GokuFullMotionContractError(
            f"{context} entity type/reference differs from the I0 registry"
        )
    _true(person.get("visible_at_i0"), context=f"{context}.visible_at_i0")
    _text(person.get("i0_state"), context=f"{context}.i0_state")
    if person.get("source_state") != "remain_still":
        raise GokuFullMotionContractError(
            f"{context}.source_state must be exactly 'remain_still'"
        )
    _validate_evidence_list(
        person.get("motion_evidence"), context=f"{context}.motion_evidence"
    )
    if person.get("confidence") != "high":
        raise GokuFullMotionContractError(
            f"{context}.confidence must be exactly 'high'"
        )
    return dict(person)


def _validate_source_camera(value: Any) -> dict[str, Any]:
    context = "source_census.camera"
    camera = _mapping(value, context=context)
    _exact_keys(
        camera,
        {
            "schema_version",
            "camera_id",
            "motion_class",
            "motion_signature",
            "motion_description",
            "dynamic",
            "motion_evidence",
            "confidence",
        },
        context=context,
    )
    if camera.get("schema_version") != SOURCE_CAMERA_SCHEMA:
        raise GokuFullMotionContractError(f"{context} schema differs")
    if camera.get("camera_id") != "camera":
        raise GokuFullMotionContractError(
            f"{context}.camera_id must be exactly 'camera'"
        )
    motion_class = _enum(
        camera.get("motion_class"),
        CAMERA_MOTION_CLASSES,
        context=f"{context}.motion_class",
    )
    signature = _text(
        camera.get("motion_signature"),
        context=f"{context}.motion_signature",
        max_length=96,
    )
    if _SIGNATURE_RE.fullmatch(signature) is None:
        raise GokuFullMotionContractError(
            f"{context}.motion_signature must be lower snake-case"
        )
    description = _text(
        camera.get("motion_description"),
        context=f"{context}.motion_description",
    )
    if _UNCERTAIN_TEXT_RE.search(description):
        raise GokuFullMotionContractError(
            f"{context}.motion_description is semantically uncertain"
        )
    dynamic = _bool(camera.get("dynamic"), context=f"{context}.dynamic")
    locked = motion_class == "locked_off"
    if locked != (signature == "locked_off") or dynamic == locked:
        raise GokuFullMotionContractError(
            f"{context} locked-off class/signature/dynamic fields disagree"
        )
    _validate_evidence_list(
        camera.get("motion_evidence"), context=f"{context}.motion_evidence"
    )
    if camera.get("confidence") != "high":
        raise GokuFullMotionContractError(
            f"{context}.confidence must be exactly 'high'"
        )
    return dict(camera)


def validate_source_census(value: Any) -> dict[str, Any]:
    """Validate and return a defensive copy of one complete source census."""

    census = _mapping(value, context="source_census")
    _exact_keys(
        census,
        {
            "schema_version",
            "iid",
            "clip",
            "source_quality",
            "scene_description",
            "i0_visible_entities",
            "i0_entity_registry",
            "motion_inventory_complete",
            "crowd_or_unresolved_motion",
            "diffuse_unresolved_motion",
            "dynamic_units",
            "static_salient_people",
            "camera",
            "uncertainty_codes",
            "confidence",
        },
        context="source_census",
    )
    if census.get("schema_version") != SOURCE_CENSUS_SCHEMA:
        raise GokuFullMotionContractError("source_census schema differs")
    iid = _text(census.get("iid"), context="source_census.iid", max_length=128)
    if _IID_RE.fullmatch(iid) is None:
        raise GokuFullMotionContractError("source_census.iid is unsafe")
    _validate_clip(census.get("clip"))
    if census.get("source_quality") != "high":
        raise GokuFullMotionContractError(
            "source_census.source_quality must be exactly 'high'"
        )
    scene = _text(
        census.get("scene_description"), context="source_census.scene_description"
    )
    if _UNCERTAIN_TEXT_RE.search(scene):
        raise GokuFullMotionContractError(
            "source_census.scene_description is semantically uncertain"
        )
    visible_entities = _ordered_unique_strings(
        census.get("i0_visible_entities"),
        context="source_census.i0_visible_entities",
        allow_empty=False,
    )
    if len(visible_entities) > 24:
        raise GokuFullMotionContractError(
            "source_census.i0_visible_entities is too large for strict grounding"
        )
    registry = _list(
        census.get("i0_entity_registry"), context="source_census.i0_entity_registry"
    )
    if not 1 <= len(registry) <= 24:
        raise GokuFullMotionContractError(
            "source_census.i0_entity_registry must contain one to 24 entries"
        )
    entity_ids = _expected_ids("entity", len(registry))
    validated_registry = [
        _validate_i0_entity(
            item,
            expected_id=entity_ids[index],
            context=f"source_census.i0_entity_registry[{index}]",
        )
        for index, item in enumerate(registry)
    ]
    registry_by_id = {
        str(item["entity_id"]): item for item in validated_registry
    }
    registry_references = [
        str(item["stable_reference"]).casefold() for item in validated_registry
    ]
    if len(set(registry_references)) != len(registry_references):
        raise GokuFullMotionContractError(
            "source_census I0 registry stable references must be unique"
        )
    exact_registry_references = [
        str(item["stable_reference"]) for item in validated_registry
    ]
    if visible_entities != exact_registry_references:
        raise GokuFullMotionContractError(
            "source_census.i0_visible_entities must exactly equal the I0 "
            "registry stable references in registry order"
        )
    grounding_keys = [
        (
            str(item["entity_type"]),
            str(item["viewer_region"]),
            int(item["region_ordinal"]),
        )
        for item in validated_registry
    ]
    if len(set(grounding_keys)) != len(grounding_keys):
        raise GokuFullMotionContractError(
            "source_census I0 registry grounding keys must be unique"
        )
    _true(
        census.get("motion_inventory_complete"),
        context="source_census.motion_inventory_complete",
    )
    _false(
        census.get("crowd_or_unresolved_motion"),
        context="source_census.crowd_or_unresolved_motion",
    )
    _false(
        census.get("diffuse_unresolved_motion"),
        context="source_census.diffuse_unresolved_motion",
    )

    dynamic = _list(census.get("dynamic_units"), context="source_census.dynamic_units")
    if not 1 <= len(dynamic) <= MAX_DYNAMIC_UNITS:
        raise GokuFullMotionContractError(
            "source_census.dynamic_units must contain one to three units"
        )
    dynamic_ids = _expected_ids("unit", len(dynamic))
    validated_dynamic = [
        _validate_source_dynamic_unit(
            item,
            expected_id=dynamic_ids[index],
            registry_by_id=registry_by_id,
            context=f"source_census.dynamic_units[{index}]",
        )
        for index, item in enumerate(dynamic)
    ]

    static = _list(
        census.get("static_salient_people"),
        context="source_census.static_salient_people",
    )
    if len(static) > MAX_STATIC_SALIENT_PEOPLE:
        raise GokuFullMotionContractError(
            "source_census has too many salient static entities"
        )
    static_ids = _expected_ids("static_person", len(static))
    validated_static = [
        _validate_source_static_person(
            item,
            expected_id=static_ids[index],
            registry_by_id=registry_by_id,
            context=f"source_census.static_salient_people[{index}]",
        )
        for index, item in enumerate(static)
    ]

    references = [
        str(item["stable_reference"]).casefold()
        for item in (*validated_dynamic, *validated_static)
    ]
    if len(set(references)) != len(references):
        raise GokuFullMotionContractError(
            "source_census stable references must be unique"
        )
    linked_dynamic_entities = [str(item["entity_id"]) for item in validated_dynamic]
    linked_static_entities = [str(item["entity_id"]) for item in validated_static]
    if len(set(linked_dynamic_entities)) != len(linked_dynamic_entities):
        raise GokuFullMotionContractError(
            "source_census dynamic units reuse an I0 registry entity"
        )
    if len(set(linked_static_entities)) != len(linked_static_entities):
        raise GokuFullMotionContractError(
            "source_census static units reuse an I0 registry entity"
        )
    expected_dynamic_entities = [
        str(item["entity_id"])
        for item in validated_registry
        if item["role"] == "dynamic_subject"
    ]
    expected_static_entities = [
        str(item["entity_id"])
        for item in validated_registry
        if item["role"] == "static_salient"
    ]
    if linked_dynamic_entities != expected_dynamic_entities:
        raise GokuFullMotionContractError(
            "source_census dynamic units do not close over dynamic I0 entities"
        )
    if linked_static_entities != expected_static_entities:
        raise GokuFullMotionContractError(
            "source_census static units do not close over static I0 entities"
        )
    _validate_source_camera(census.get("camera"))
    if census.get("uncertainty_codes") != []:
        raise GokuFullMotionContractError(
            "source_census.uncertainty_codes must be exactly []"
        )
    if census.get("confidence") != "high":
        raise GokuFullMotionContractError(
            "source_census.confidence must be exactly 'high'"
        )
    canonical_json_bytes(census)
    return copy.deepcopy(dict(census))


def _source_alignment_dynamic_projection(
    unit: Mapping[str, Any],
) -> dict[str, Any]:
    component_projections: list[dict[str, Any]] = []
    unit_primitives: set[str] = set()
    for component in unit["source_motion_components"]:
        primitives = sorted(
            _source_alignment_component_primitives(component)
        )
        if not primitives:
            raise GokuFullMotionContractError(
                "independent source motion component has no recognized "
                "semantic primitives: "
                f"{unit['unit_id']}.{component['component_id']} "
                f"({component['component_type']})"
            )
        unit_primitives.update(primitives)
        component_projections.append(
            {
                "component_id": component["component_id"],
                "component_type": component["component_type"],
                "semantic_primitives": primitives,
                "dependent_entity_ids": component["dependent_entity_ids"],
            }
        )
    return {
        "unit_id": unit["unit_id"],
        "entity_id": unit["entity_id"],
        "entity_type": unit["entity_type"],
        # The component list is the closed positive-motion inventory.  Derive
        # unit semantics from that list instead of free-form source_motion,
        # which is also allowed to carry stable pose/context descriptions.
        "semantic_primitives": sorted(unit_primitives),
        "components": component_projections,
    }


def source_inventory_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact structured inventory used to align two blind censuses."""

    census = validate_source_census(value)
    return {
        "iid": census["iid"],
        "entities": [
            {
                "entity_id": item["entity_id"],
                "entity_type": item["entity_type"],
                "i0_bbox_xyxy_1000": item["i0_bbox_xyxy_1000"],
                "viewer_region": item["viewer_region"],
                "region_ordinal": item["region_ordinal"],
                "role": item["role"],
            }
            for item in census["i0_entity_registry"]
        ],
        "dynamic_units": [
            _source_alignment_dynamic_projection(item)
            for item in census["dynamic_units"]
        ],
        "static_entities": [
            {
                "unit_id": item["unit_id"],
                "entity_id": item["entity_id"],
                "entity_type": item["entity_type"],
            }
            for item in census["static_salient_people"]
        ],
        "camera": {
            "dynamic": census["camera"]["dynamic"],
            "motion_class": census["camera"]["motion_class"],
        },
    }


def _bbox_iou_milli(first: list[int], second: list[int]) -> int:
    intersection_width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return int(round(1000 * intersection / union)) if union else 0


def _bbox_center_linf(first: list[int], second: list[int]) -> int:
    first_center = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
    second_center = (
        (second[0] + second[2]) / 2.0,
        (second[1] + second[3]) / 2.0,
    )
    return int(round(max(abs(first_center[0] - second_center[0]), abs(first_center[1] - second_center[1]))))


def _align_i0_entities(
    primary_entities: list[Mapping[str, Any]],
    secondary_entities: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(primary_entities) != len(secondary_entities):
        raise GokuFullMotionContractError(
            "independent source inventory projections differ"
        )
    primary_kinds = sorted(
        (str(item["role"]), str(item["entity_type"]))
        for item in primary_entities
    )
    secondary_kinds = sorted(
        (str(item["role"]), str(item["entity_type"]))
        for item in secondary_entities
    )
    if primary_kinds != secondary_kinds:
        raise GokuFullMotionContractError(
            "independent source inventory projections differ"
        )

    matches: list[dict[str, Any]] = []
    used_secondary: set[str] = set()
    for primary_entity in primary_entities:
        candidates: list[tuple[int, int, Mapping[str, Any]]] = []
        first_bbox = list(primary_entity["i0_bbox_xyxy_1000"])
        for secondary_entity in secondary_entities:
            if (
                secondary_entity["role"] != primary_entity["role"]
                or secondary_entity["entity_type"]
                != primary_entity["entity_type"]
            ):
                continue
            second_bbox = list(secondary_entity["i0_bbox_xyxy_1000"])
            iou_milli = _bbox_iou_milli(first_bbox, second_bbox)
            center_distance = _bbox_center_linf(first_bbox, second_bbox)
            if iou_milli >= 250 or center_distance <= 80:
                candidates.append((iou_milli, center_distance, secondary_entity))
        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1],
                str(item[2]["entity_id"]),
            )
        )
        if not candidates:
            raise GokuFullMotionContractError(
                "independent source inventory I0 boxes do not align"
            )
        if len(candidates) > 1 and candidates[0][:2] == candidates[1][:2]:
            raise GokuFullMotionContractError(
                "independent source inventory I0 alignment is ambiguous"
            )
        iou_milli, center_distance, secondary_entity = candidates[0]
        secondary_id = str(secondary_entity["entity_id"])
        if secondary_id in used_secondary:
            raise GokuFullMotionContractError(
                "independent source inventory I0 alignment is not one-to-one"
            )
        used_secondary.add(secondary_id)
        matches.append(
            {
                "primary_entity_id": primary_entity["entity_id"],
                "secondary_entity_id": secondary_id,
                "entity_type": primary_entity["entity_type"],
                "role": primary_entity["role"],
                "bbox_iou_milli": iou_milli,
                "center_linf_distance_1000": center_distance,
            }
        )
    if len(used_secondary) != len(secondary_entities):
        raise GokuFullMotionContractError(
            "independent source inventory I0 alignment leaves unmatched entities"
        )
    return matches


def build_source_inventory_alignment(
    *, primary: Mapping[str, Any], secondary: Mapping[str, Any]
) -> dict[str, Any]:
    """Require two independently generated censuses to agree structurally."""

    primary_valid = validate_source_census(primary)
    secondary_valid = validate_source_census(secondary)
    if primary_valid["iid"] != secondary_valid["iid"]:
        raise GokuFullMotionContractError(
            "independent source inventory projections differ"
        )
    primary_projection = source_inventory_projection(primary_valid)
    secondary_projection = source_inventory_projection(secondary_valid)
    if primary_projection["camera"] != secondary_projection["camera"]:
        raise GokuFullMotionContractError(
            "independent source inventory projections differ"
        )
    entity_matches = _align_i0_entities(
        list(primary_projection["entities"]),
        list(secondary_projection["entities"]),
    )
    secondary_to_primary = {
        str(item["secondary_entity_id"]): str(item["primary_entity_id"])
        for item in entity_matches
    }

    primary_dynamic_by_entity = {
        str(item["entity_id"]): item for item in primary_projection["dynamic_units"]
    }
    secondary_dynamic_by_primary_entity: dict[str, Mapping[str, Any]] = {}
    for item in secondary_projection["dynamic_units"]:
        mapped_entity = secondary_to_primary.get(str(item["entity_id"]))
        if mapped_entity is None or mapped_entity in secondary_dynamic_by_primary_entity:
            raise GokuFullMotionContractError(
                "independent source dynamic-unit alignment differs"
            )
        secondary_dynamic_by_primary_entity[mapped_entity] = item
    if set(primary_dynamic_by_entity) != set(secondary_dynamic_by_primary_entity):
        raise GokuFullMotionContractError(
            "independent source inventory projections differ"
        )
    dynamic_matches: list[dict[str, Any]] = []
    for primary_unit in primary_projection["dynamic_units"]:
        primary_entity_id = str(primary_unit["entity_id"])
        secondary_unit = secondary_dynamic_by_primary_entity[primary_entity_id]
        primary_components = {
            str(item["component_type"]): item for item in primary_unit["components"]
        }
        secondary_components = {
            str(item["component_type"]): item
            for item in secondary_unit["components"]
        }
        if set(primary_components) != set(secondary_components):
            raise GokuFullMotionContractError(
                "independent source motion-component inventories differ"
            )
        if (
            primary_unit["semantic_primitives"]
            != secondary_unit["semantic_primitives"]
        ):
            raise GokuFullMotionContractError(
                "independent source motion semantic primitives differ"
            )
        for component_type, primary_component in primary_components.items():
            secondary_component = secondary_components[component_type]
            if (
                primary_component["semantic_primitives"]
                != secondary_component["semantic_primitives"]
            ):
                raise GokuFullMotionContractError(
                    "independent source motion-component semantic primitives "
                    "differ"
                )
            mapped_dependencies = sorted(
                secondary_to_primary.get(str(entity_id), "")
                for entity_id in secondary_component["dependent_entity_ids"]
            )
            if "" in mapped_dependencies or mapped_dependencies != sorted(
                str(entity_id)
                for entity_id in primary_component["dependent_entity_ids"]
            ):
                raise GokuFullMotionContractError(
                    "independent source component dependencies differ"
                )
        dynamic_matches.append(
            {
                "primary_unit_id": primary_unit["unit_id"],
                "secondary_unit_id": secondary_unit["unit_id"],
                "primary_entity_id": primary_entity_id,
                "secondary_entity_id": secondary_unit["entity_id"],
                "component_types": sorted(primary_components),
                "semantic_primitives": primary_unit["semantic_primitives"],
            }
        )

    primary_static_by_entity = {
        str(item["entity_id"]): item for item in primary_projection["static_entities"]
    }
    secondary_static_by_primary_entity: dict[str, Mapping[str, Any]] = {}
    for item in secondary_projection["static_entities"]:
        mapped_entity = secondary_to_primary.get(str(item["entity_id"]))
        if mapped_entity is None or mapped_entity in secondary_static_by_primary_entity:
            raise GokuFullMotionContractError(
                "independent source static-unit alignment differs"
            )
        secondary_static_by_primary_entity[mapped_entity] = item
    if set(primary_static_by_entity) != set(secondary_static_by_primary_entity):
        raise GokuFullMotionContractError(
            "independent source inventory projections differ"
        )
    static_matches = [
        {
            "primary_unit_id": primary_static_by_entity[entity_id]["unit_id"],
            "secondary_unit_id": secondary_static_by_primary_entity[entity_id][
                "unit_id"
            ],
            "primary_entity_id": entity_id,
            "secondary_entity_id": secondary_static_by_primary_entity[entity_id][
                "entity_id"
            ],
        }
        for entity_id in primary_static_by_entity
    ]
    result = {
        "schema_version": SOURCE_INVENTORY_ALIGNMENT_SCHEMA,
        "iid": primary_valid["iid"],
        "primary_source_census_sha256": object_sha256(primary_valid),
        "secondary_source_census_sha256": object_sha256(secondary_valid),
        "primary_projection_sha256": object_sha256(primary_projection),
        "secondary_projection_sha256": object_sha256(secondary_projection),
        "entity_matches": entity_matches,
        "dynamic_unit_matches": dynamic_matches,
        "static_unit_matches": static_matches,
        "aligned_entity_ids": [
            str(item["entity_id"]) for item in primary_projection["entities"]
        ],
        "aligned_dynamic_unit_ids": [
            str(item["unit_id"]) for item in primary_projection["dynamic_units"]
        ],
        "aligned_static_unit_ids": [
            str(item["unit_id"]) for item in primary_projection["static_entities"]
        ],
        "camera_aligned": True,
        "projections_equal": True,
    }
    canonical_json_bytes(result)
    return result


def validate_source_inventory_alignment(
    value: Any,
    *,
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
) -> dict[str, Any]:
    alignment = _mapping(value, context="source_inventory_alignment")
    expected = build_source_inventory_alignment(
        primary=primary, secondary=secondary
    )
    _exact_keys(alignment, set(expected), context="source_inventory_alignment")
    if dict(alignment) != expected:
        raise GokuFullMotionContractError("source inventory alignment differs")
    return copy.deepcopy(dict(alignment))


def _validate_component_disposition(
    value: Any,
    *,
    expected_component: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    disposition = _mapping(value, context=context)
    _exact_keys(
        disposition,
        {
            "schema_version",
            "component_id",
            "disposition",
            "explicit_target_motion",
        },
        context=context,
    )
    if disposition.get("schema_version") != TARGET_COMPONENT_DISPOSITION_SCHEMA:
        raise GokuFullMotionContractError(f"{context} schema differs")
    if disposition.get("component_id") != expected_component["component_id"]:
        raise GokuFullMotionContractError(
            f"{context}.component_id differs from the source component"
        )
    kind = _enum(
        disposition.get("disposition"),
        COMPONENT_DISPOSITIONS,
        context=f"{context}.disposition",
    )
    motion = disposition.get("explicit_target_motion")
    if kind == "suppress":
        if motion is not None:
            raise GokuFullMotionContractError(
                f"{context} suppressed component must have null target motion"
            )
    else:
        _target_text(motion, context=f"{context}.explicit_target_motion")
    return dict(disposition)


def _ordered_entity_markers(value: str, *, context: str) -> list[str]:
    markers = _ENTITY_MARKER_RE.findall(value)
    if "[[" in _ENTITY_MARKER_RE.sub("", value) or "]]" in _ENTITY_MARKER_RE.sub(
        "", value
    ):
        raise GokuFullMotionContractError(f"{context} contains a malformed entity marker")
    ordered: list[str] = []
    for marker in markers:
        if marker not in ordered:
            ordered.append(marker)
    return ordered


def _validate_target_dynamic_unit(
    value: Any,
    *,
    source: Mapping[str, Any],
    registry_by_id: Mapping[str, Mapping[str, Any]],
    expected_id: str,
    context: str,
) -> dict[str, Any]:
    target = _mapping(value, context=context)
    _exact_keys(
        target,
        {
            "schema_version",
            "unit_id",
            "entity_id",
            "stable_reference",
            "target_action_signature",
            "motion_relation",
            "source_motion_suppressed",
            "explicit_shared_base_motion",
            "source_component_dispositions",
            "novel_target_motion",
            "target_clause",
            "substantive_change",
            "starts_at_i0",
            "i0_executable",
            "complete_within_clip",
            "completion_time_seconds",
            "ordered_stages",
            "interaction_entity_ids",
            "required_i0_entity_ids",
        },
        context=context,
    )
    if target.get("schema_version") != TARGET_DYNAMIC_UNIT_SCHEMA:
        raise GokuFullMotionContractError(f"{context} schema differs")
    if target.get("unit_id") != expected_id:
        raise GokuFullMotionContractError(
            f"{context}.unit_id must equal {expected_id!r}"
        )
    entity_id = _text(target.get("entity_id"), context=f"{context}.entity_id")
    if entity_id != source["entity_id"]:
        raise GokuFullMotionContractError(
            f"{context}.entity_id differs from the source dynamic unit"
        )
    reference = _text(
        target.get("stable_reference"),
        context=f"{context}.stable_reference",
        max_length=MAX_STABLE_REFERENCE_CHARS,
    )
    if reference != source["stable_reference"]:
        raise GokuFullMotionContractError(
            f"{context}.stable_reference differs from source census"
        )
    target_signature = _signature(
        target.get("target_action_signature"),
        context=f"{context}.target_action_signature",
    )
    if target_signature == source["source_action_signature"]:
        raise GokuFullMotionContractError(
            f"{context} target action is identical to the source action"
        )
    relation = _enum(
        target.get("motion_relation"),
        MOTION_RELATIONS,
        context=f"{context}.motion_relation",
    )
    suppressed = _bool(
        target.get("source_motion_suppressed"),
        context=f"{context}.source_motion_suppressed",
    )
    shared_value = target.get("explicit_shared_base_motion")
    shared: str | None
    if relation == "replace":
        if suppressed is not True or shared_value is not None:
            raise GokuFullMotionContractError(
                f"{context} replace relation must suppress source motion and "
                "have null explicit_shared_base_motion"
            )
        shared = None
    else:
        if suppressed is not False:
            raise GokuFullMotionContractError(
                f"{context} shared-base relation cannot suppress its base"
            )
        shared = _target_text(
            shared_value,
            context=f"{context}.explicit_shared_base_motion",
        )
    source_components = list(source["source_motion_components"])
    raw_dispositions = _list(
        target.get("source_component_dispositions"),
        context=f"{context}.source_component_dispositions",
    )
    if len(raw_dispositions) != len(source_components):
        raise GokuFullMotionContractError(
            f"{context} must dispose every source motion component exactly once"
        )
    dispositions = [
        _validate_component_disposition(
            item,
            expected_component=source_components[index],
            context=f"{context}.source_component_dispositions[{index}]",
        )
        for index, item in enumerate(raw_dispositions)
    ]
    shared_dispositions = [
        item for item in dispositions if item["disposition"] == "explicit_shared_base"
    ]
    if relation == "replace" and shared_dispositions:
        raise GokuFullMotionContractError(
            f"{context} replace relation must suppress every source component"
        )
    if relation == "explicit_shared_base_with_novel_action" and not shared_dispositions:
        raise GokuFullMotionContractError(
            f"{context} shared-base relation must explicitly retain a source component"
        )
    if shared is not None:
        missing_shared = [
            str(item["component_id"])
            for item in shared_dispositions
            if _prose_comparison_key(str(item["explicit_target_motion"]))
            not in _prose_comparison_key(shared)
        ]
        if missing_shared:
            raise GokuFullMotionContractError(
                f"{context}.explicit_shared_base_motion omits component motions: "
                f"{missing_shared!r}"
            )
    novel = _target_text(
        target.get("novel_target_motion"),
        context=f"{context}.novel_target_motion",
    )
    if relation == "replace":
        _reject_unestablished_replace_locomotion(
            novel,
            context=f"{context}.novel_target_motion",
        )
    if _prose_comparison_key(novel) == _prose_comparison_key(
        str(source["source_motion"])
    ):
        raise GokuFullMotionContractError(
            f"{context}.novel_target_motion restates source motion"
        )
    _text(
        target.get("target_clause"),
        context=f"{context}.target_clause",
    )
    # target_clause is a non-executable cross-check.  The sole executable
    # instruction is deterministically built from stable_reference and the
    # structured novel/shared motion fields.  This redundant model-authored
    # prose may paraphrase the actor as well as the motion, so it must not be
    # required to reproduce stable_reference literally.
    # For a shared-base relation, ``shared`` is mandatory above and is the
    # executable structured value used by the compiler.  Do not impose byte
    # equality on this redundant model-authored prose cross-check.
    _true(
        target.get("substantive_change"),
        context=f"{context}.substantive_change",
    )
    _true(target.get("starts_at_i0"), context=f"{context}.starts_at_i0")
    _true(target.get("i0_executable"), context=f"{context}.i0_executable")
    _true(
        target.get("complete_within_clip"),
        context=f"{context}.complete_within_clip",
    )
    _positive_seconds(
        target.get("completion_time_seconds"),
        context=f"{context}.completion_time_seconds",
    )
    stages = _ordered_unique_strings(
        target.get("ordered_stages"),
        context=f"{context}.ordered_stages",
        allow_empty=False,
    )
    if len(stages) > 6:
        raise GokuFullMotionContractError(f"{context}.ordered_stages is too long")
    for index, stage in enumerate(stages):
        _target_text(stage, context=f"{context}.ordered_stages[{index}]")
    if relation == "replace":
        _reject_unestablished_replace_locomotion(
            ". ".join(stages),
            context=f"{context}.ordered_stages",
        )
    source_primitives = _motion_semantic_primitives(
        str(source["source_motion"]),
        *(
            str(component["motion_description"])
            for component in source_components
        ),
    )
    # Only prose rendered by the deterministic compiler may establish target
    # novelty.  ``ordered_stages`` is a non-executable chronology cross-check;
    # counting a primitive that exists only there lets a semantic restatement
    # pass while the actual Wan prompt still requests the source action.  A
    # shared base is executable too, so include it in the rendered-fragment
    # inventory, but never let non-rendered stages manufacture novelty.
    executable_fragments = [novel]
    if shared is not None:
        executable_fragments.append(shared)
    target_primitives = _motion_semantic_primitives(*executable_fragments)
    if not source_primitives:
        raise GokuFullMotionContractError(
            f"{context} source motion has no supported closed semantic "
            "primitive; reject this clip rather than guessing novelty"
        )
    if not target_primitives:
        raise GokuFullMotionContractError(
            f"{context} target motion has no supported closed semantic "
            "primitive"
        )
    if not (target_primitives - source_primitives):
        raise GokuFullMotionContractError(
            f"{context}.novel_target_motion is only a semantic restatement "
            "of source motion"
        )
    interaction_ids = _ordered_unique_strings(
        target.get("interaction_entity_ids"),
        context=f"{context}.interaction_entity_ids",
        allow_empty=True,
    )
    for interaction_id in interaction_ids:
        interaction = registry_by_id.get(interaction_id)
        if interaction is None or interaction_id == entity_id:
            raise GokuFullMotionContractError(
                f"{context} interaction entity {interaction_id!r} is not a distinct I0 entity"
            )
        if interaction.get("reachable_at_i0") is not True:
            raise GokuFullMotionContractError(
                f"{context} interaction entity {interaction_id!r} is not reachable at I0"
            )
        if interaction.get("role") == "dynamic_subject":
            raise GokuFullMotionContractError(
                f"{context} cannot borrow novelty from another dynamic "
                "subject; describe each dynamic subject's own new action in "
                "its corresponding unit"
            )
    # Free prose cannot reliably attribute a newly detected primitive across
    # two independently moving agents.  Dynamic-to-dynamic dependencies are
    # therefore fail-closed above, and the executable fragments may not spell
    # out another dynamic registry subject as an unmarked subordinate actor.
    # Passive I0 interaction objects remain legal and continue to require an
    # exact [[entity_NN]] marker below.
    executable_text = " ".join(executable_fragments).casefold()
    for other_entity_id, other_entity in registry_by_id.items():
        if (
            other_entity_id != entity_id
            and other_entity.get("role") == "dynamic_subject"
            and str(other_entity["stable_reference"]).casefold()
            in executable_text
        ):
            raise GokuFullMotionContractError(
                f"{context} executable motion names another dynamic subject; "
                "each unit must describe only its owner's new primitives"
            )
    # Only the fields consumed by the deterministic compiler may establish an
    # interaction binding.  ``ordered_stages`` is a semantic cross-check and
    # cannot smuggle a marker that is absent from the rendered instruction.
    marker_ids: list[str] = []
    for index, fragment in enumerate(executable_fragments):
        for marker_id in _ordered_entity_markers(
            fragment, context=f"{context}.executable_fragment[{index}]"
        ):
            if marker_id not in marker_ids:
                marker_ids.append(marker_id)
    if marker_ids != interaction_ids:
        raise GokuFullMotionContractError(
            f"{context} executable entity markers must exactly equal interaction_entity_ids"
        )
    for index, fragment in enumerate(executable_fragments):
        fragment_markers = _ordered_entity_markers(
            fragment, context=f"{context}.rendered_fragment[{index}]"
        )
        if _INTERACTION_VERB_RE.search(fragment) and not fragment_markers:
            raise GokuFullMotionContractError(
                f"{context} interaction motion must bind its I0 entity in the "
                "same rendered fragment"
            )
    for index, stage in enumerate(stages):
        stage_markers = _ordered_entity_markers(
            stage, context=f"{context}.ordered_stages[{index}]"
        )
        unknown_stage_markers = [
            marker for marker in stage_markers if marker not in interaction_ids
        ]
        if unknown_stage_markers:
            raise GokuFullMotionContractError(
                f"{context}.ordered_stages contains unbound entity markers: "
                f"{unknown_stage_markers!r}"
            )
        if _INTERACTION_VERB_RE.search(stage) and not stage_markers:
            raise GokuFullMotionContractError(
                f"{context}.ordered_stages interaction must use a bound I0 marker"
            )
    required_ids = _ordered_unique_strings(
        target.get("required_i0_entity_ids"),
        context=f"{context}.required_i0_entity_ids",
        allow_empty=False,
    )
    expected_required = [entity_id, *interaction_ids]
    if required_ids != expected_required:
        raise GokuFullMotionContractError(
            f"{context}.required_i0_entity_ids must equal subject plus interactions"
        )
    return dict(target)


def _validate_target_static_person(
    value: Any,
    *,
    source: Mapping[str, Any],
    expected_id: str,
    context: str,
) -> dict[str, Any]:
    target = _mapping(value, context=context)
    _exact_keys(
        target,
        {
            "schema_version",
            "unit_id",
            "entity_id",
            "entity_type",
            "stable_reference",
            "target_state",
            "target_clause",
        },
        context=context,
    )
    if target.get("schema_version") != TARGET_STATIC_PERSON_SCHEMA:
        raise GokuFullMotionContractError(f"{context} schema differs")
    if target.get("unit_id") != expected_id:
        raise GokuFullMotionContractError(
            f"{context}.unit_id must equal {expected_id!r}"
        )
    if target.get("entity_id") != source["entity_id"]:
        raise GokuFullMotionContractError(
            f"{context}.entity_id differs from source census"
        )
    if target.get("entity_type") != source["entity_type"]:
        raise GokuFullMotionContractError(
            f"{context}.entity_type differs from source census"
        )
    reference = _text(
        target.get("stable_reference"),
        context=f"{context}.stable_reference",
        max_length=MAX_STABLE_REFERENCE_CHARS,
    )
    if reference != source["stable_reference"]:
        raise GokuFullMotionContractError(
            f"{context}.stable_reference differs from source census"
        )
    if target.get("target_state") != "remain_still":
        raise GokuFullMotionContractError(
            f"{context}.target_state must be exactly 'remain_still'"
        )
    _text(
        target.get("target_clause"),
        context=f"{context}.target_clause",
    )
    # target_clause is non-executable model prose.  ``target_state`` above is
    # the authoritative static constraint consumed by the compiler, so a
    # semantically valid paraphrase must not fail a literal wording check.
    return dict(target)


def _validate_target_camera(
    value: Any, *, source: Mapping[str, Any]
) -> dict[str, Any]:
    context = "target_plan.camera_target"
    target = _mapping(value, context=context)
    _exact_keys(
        target,
        {
            "schema_version",
            "camera_id",
            "motion_relation",
            "target_motion_class",
            "target_motion_signature",
            "target_motion_description",
            "target_clause",
            "source_motion_suppressed",
            "substantive_change",
            "starts_at_i0",
            "i0_executable",
            "complete_within_clip",
            "completion_time_seconds",
            "ordered_stages",
        },
        context=context,
    )
    if target.get("schema_version") != TARGET_CAMERA_SCHEMA:
        raise GokuFullMotionContractError(f"{context} schema differs")
    if target.get("camera_id") != "camera":
        raise GokuFullMotionContractError(
            f"{context}.camera_id must be exactly 'camera'"
        )
    relation = _enum(
        target.get("motion_relation"),
        CAMERA_RELATIONS,
        context=f"{context}.motion_relation",
    )
    motion_class = _enum(
        target.get("target_motion_class"),
        CAMERA_MOTION_CLASSES,
        context=f"{context}.target_motion_class",
    )
    signature = _text(
        target.get("target_motion_signature"),
        context=f"{context}.target_motion_signature",
        max_length=96,
    )
    if _SIGNATURE_RE.fullmatch(signature) is None:
        raise GokuFullMotionContractError(
            f"{context}.target_motion_signature must be lower snake-case"
        )
    description = _target_text(
        target.get("target_motion_description"),
        context=f"{context}.target_motion_description",
    )
    _text(
        target.get("target_clause"),
        context=f"{context}.target_clause",
    )
    # target_clause is only a non-executable semantic cross-check.  The
    # structured relation, class, signature, and description remain the
    # authoritative camera plan and the compiler never renders this prose.
    source_dynamic = bool(source["dynamic"])
    suppressed = _bool(
        target.get("source_motion_suppressed"),
        context=f"{context}.source_motion_suppressed",
    )
    substantive = _bool(
        target.get("substantive_change"),
        context=f"{context}.substantive_change",
    )
    if not source_dynamic:
        if (
            relation != "preserve_static"
            or motion_class != "locked_off"
            or signature != "locked_off"
            or suppressed is not False
            or substantive is not False
        ):
            raise GokuFullMotionContractError(
                f"{context} must explicitly preserve a static camera as locked off"
            )
    else:
        if (
            relation != "replace_motion"
            or motion_class == source["motion_class"]
            or signature == source["motion_signature"]
            or _prose_comparison_key(description)
            == _prose_comparison_key(str(source["motion_description"]))
            or suppressed is not True
            or substantive is not True
        ):
            raise GokuFullMotionContractError(
                f"{context} must replace dynamic source camera motion with a "
                "different explicit trajectory"
            )
    locked = motion_class == "locked_off"
    if locked != (signature == "locked_off"):
        raise GokuFullMotionContractError(
            f"{context} locked-off class and signature disagree"
        )
    _true(target.get("starts_at_i0"), context=f"{context}.starts_at_i0")
    _true(target.get("i0_executable"), context=f"{context}.i0_executable")
    _true(
        target.get("complete_within_clip"),
        context=f"{context}.complete_within_clip",
    )
    _positive_seconds(
        target.get("completion_time_seconds"),
        context=f"{context}.completion_time_seconds",
    )
    stages = _ordered_unique_strings(
        target.get("ordered_stages"),
        context=f"{context}.ordered_stages",
        allow_empty=False,
    )
    if len(stages) > 4:
        raise GokuFullMotionContractError(f"{context}.ordered_stages is too long")
    for index, stage in enumerate(stages):
        _target_text(stage, context=f"{context}.ordered_stages[{index}]")
    return dict(target)


def _validate_preservation(value: Any) -> dict[str, Any]:
    context = "target_plan.preservation"
    preservation = _mapping(value, context=context)
    _exact_keys(
        preservation,
        {
            "schema_version",
            "preserve_identity",
            "preserve_appearance",
            "preserve_scene",
            "allow_new_entities",
            "allow_removed_entities",
        },
        context=context,
    )
    if preservation.get("schema_version") != TARGET_PRESERVATION_SCHEMA:
        raise GokuFullMotionContractError(f"{context} schema differs")
    for field in ("preserve_identity", "preserve_appearance", "preserve_scene"):
        _true(preservation.get(field), context=f"{context}.{field}")
    for field in ("allow_new_entities", "allow_removed_entities"):
        _false(preservation.get(field), context=f"{context}.{field}")
    return dict(preservation)


def _validate_coverage(
    value: Any,
    *,
    dynamic_ids: list[str],
    static_ids: list[str],
) -> dict[str, Any]:
    context = "target_plan.coverage"
    coverage = _mapping(value, context=context)
    _exact_keys(
        coverage,
        {
            "schema_version",
            "required_dynamic_unit_ids",
            "planned_changed_unit_ids",
            "missing_unit_ids",
            "extra_unit_ids",
            "required_static_person_ids",
            "constrained_static_person_ids",
            "camera_clause_present",
        },
        context=context,
    )
    if coverage.get("schema_version") != TARGET_COVERAGE_SCHEMA:
        raise GokuFullMotionContractError(f"{context} schema differs")
    list_fields = {
        "required_dynamic_unit_ids": dynamic_ids,
        "planned_changed_unit_ids": dynamic_ids,
        "missing_unit_ids": [],
        "extra_unit_ids": [],
        "required_static_person_ids": static_ids,
        "constrained_static_person_ids": static_ids,
    }
    for field, expected in list_fields.items():
        actual = _ordered_unique_strings(
            coverage.get(field), context=f"{context}.{field}", allow_empty=True
        )
        if actual != expected:
            raise GokuFullMotionContractError(
                f"{context}.{field} differs: expected={expected!r} actual={actual!r}"
            )
    _true(
        coverage.get("camera_clause_present"),
        context=f"{context}.camera_clause_present",
    )
    return dict(coverage)


def validate_target_plan(
    value: Any, *, source_census: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a complete all-unit target plan against its source census."""

    source = validate_source_census(source_census)
    plan = _mapping(value, context="target_plan")
    _exact_keys(
        plan,
        {
            "schema_version",
            "iid",
            "source_census_sha256",
            "dynamic_unit_targets",
            "static_person_targets",
            "camera_target",
            "preservation",
            "coverage",
            "i0_executable",
            "no_new_prerequisites",
            "uncertainty_codes",
            "confidence",
        },
        context="target_plan",
    )
    if plan.get("schema_version") != TARGET_PLAN_SCHEMA:
        raise GokuFullMotionContractError("target_plan schema differs")
    if plan.get("iid") != source["iid"]:
        raise GokuFullMotionContractError("target_plan.iid differs from source")
    source_digest = object_sha256(source)
    if plan.get("source_census_sha256") != source_digest:
        raise GokuFullMotionContractError(
            "target_plan.source_census_sha256 differs"
        )

    dynamic_sources = list(source["dynamic_units"])
    registry_by_id = {
        str(item["entity_id"]): item for item in source["i0_entity_registry"]
    }
    dynamic_ids = [str(item["unit_id"]) for item in dynamic_sources]
    dynamic_targets = _list(
        plan.get("dynamic_unit_targets"),
        context="target_plan.dynamic_unit_targets",
    )
    if len(dynamic_targets) != len(dynamic_sources):
        raise GokuFullMotionContractError(
            "target_plan must contain exactly one target for every dynamic unit"
        )
    validated_dynamic = [
        _validate_target_dynamic_unit(
            item,
            source=dynamic_sources[index],
            registry_by_id=registry_by_id,
            expected_id=dynamic_ids[index],
            context=f"target_plan.dynamic_unit_targets[{index}]",
        )
        for index, item in enumerate(dynamic_targets)
    ]
    for target in validated_dynamic:
        missing = [
            entity_id
            for entity_id in target["required_i0_entity_ids"]
            if entity_id not in registry_by_id
        ]
        if missing:
            raise GokuFullMotionContractError(
                f"target unit {target['unit_id']} requires registry entities "
                f"absent at I0: {missing!r}"
            )

    static_sources = list(source["static_salient_people"])
    static_ids = [str(item["unit_id"]) for item in static_sources]
    static_targets = _list(
        plan.get("static_person_targets"),
        context="target_plan.static_person_targets",
    )
    if len(static_targets) != len(static_sources):
        raise GokuFullMotionContractError(
            "target_plan must constrain every salient static entity"
        )
    validated_static = [
        _validate_target_static_person(
            item,
            source=static_sources[index],
            expected_id=static_ids[index],
            context=f"target_plan.static_person_targets[{index}]",
        )
        for index, item in enumerate(static_targets)
    ]

    _validate_target_camera(plan.get("camera_target"), source=source["camera"])
    _validate_preservation(plan.get("preservation"))
    _validate_coverage(
        plan.get("coverage"), dynamic_ids=dynamic_ids, static_ids=static_ids
    )
    _true(plan.get("i0_executable"), context="target_plan.i0_executable")
    _true(
        plan.get("no_new_prerequisites"),
        context="target_plan.no_new_prerequisites",
    )
    if plan.get("uncertainty_codes") != []:
        raise GokuFullMotionContractError(
            "target_plan.uncertainty_codes must be exactly []"
        )
    if plan.get("confidence") != "high":
        raise GokuFullMotionContractError(
            "target_plan.confidence must be exactly 'high'"
        )
    canonical_json_bytes(plan)
    return copy.deepcopy(dict(plan))


_SOURCE_CENSUS_KEYS = {
    "schema_version",
    "iid",
    "clip",
    "source_quality",
    "scene_description",
    "i0_visible_entities",
    "i0_entity_registry",
    "motion_inventory_complete",
    "crowd_or_unresolved_motion",
    "diffuse_unresolved_motion",
    "dynamic_units",
    "static_salient_people",
    "camera",
    "uncertainty_codes",
    "confidence",
}
_SOURCE_REGISTRY_KEYS = {
    "schema_version",
    "entity_id",
    "entity_type",
    "stable_reference",
    "i0_bbox_xyxy_1000",
    "viewer_region",
    "region_ordinal",
    "role",
    "visible_at_i0",
    "reachable_at_i0",
    "confidence",
}
_SOURCE_DYNAMIC_KEYS = {
    "schema_version",
    "unit_id",
    "entity_id",
    "entity_type",
    "stable_reference",
    "visible_at_i0",
    "independent_motion",
    "i0_state",
    "source_action_signature",
    "source_motion",
    "source_motion_components",
    "motion_evidence",
    "confidence",
}
_SOURCE_STATIC_KEYS = {
    "schema_version",
    "unit_id",
    "entity_id",
    "entity_type",
    "stable_reference",
    "visible_at_i0",
    "i0_state",
    "source_state",
    "motion_evidence",
    "confidence",
}
_TARGET_PLAN_KEYS = {
    "schema_version",
    "iid",
    "source_census_sha256",
    "dynamic_unit_targets",
    "static_person_targets",
    "camera_target",
    "preservation",
    "coverage",
    "i0_executable",
    "no_new_prerequisites",
    "uncertainty_codes",
    "confidence",
}
_TARGET_DYNAMIC_KEYS = {
    "schema_version",
    "unit_id",
    "entity_id",
    "stable_reference",
    "target_action_signature",
    "motion_relation",
    "source_motion_suppressed",
    "explicit_shared_base_motion",
    "source_component_dispositions",
    "novel_target_motion",
    "target_clause",
    "substantive_change",
    "starts_at_i0",
    "i0_executable",
    "complete_within_clip",
    "completion_time_seconds",
    "ordered_stages",
    "interaction_entity_ids",
    "required_i0_entity_ids",
}
_TARGET_STATIC_KEYS = {
    "schema_version",
    "unit_id",
    "entity_id",
    "entity_type",
    "stable_reference",
    "target_state",
    "target_clause",
}
_CANONICALIZATION_RECEIPT_KEYS = {
    "schema_version",
    "artifact_kind",
    "policy",
    "semantic_repair",
    "context",
    "raw_sha256",
    "canonical_sha256",
    "normalized_field_paths",
    "changed_field_paths",
    "receipt_sha256",
}
_MISSING = object()


def _set_redundant_field(
    container: dict[str, Any],
    field: str,
    value: Any,
    *,
    path: str,
    normalized_paths: list[str],
    changed_paths: list[str],
    reject_present_conflict: bool,
) -> None:
    """Set one explicitly whitelisted redundancy and record the decision.

    Identity and semantic redundancies may be filled when absent, but a
    model-supplied conflicting value is evidence of an actor-binding error and
    must fail closed.  Only mechanically derived geometry fields opt into
    normalization of a present conflict.
    """

    normalized_paths.append(path)
    previous = container.get(field, _MISSING)
    differs = previous is _MISSING or canonical_json_bytes(
        previous
    ) != canonical_json_bytes(value)
    if previous is not _MISSING and differs and reject_present_conflict:
        raise GokuFullMotionContractError(
            f"{path} conflicts with its authoritative value before "
            "canonicalization"
        )
    if differs:
        changed_paths.append(path)
    container[field] = copy.deepcopy(value)


def _build_canonicalization_receipt(
    *,
    artifact_kind: str,
    context: Mapping[str, Any],
    raw: Mapping[str, Any],
    canonical: Mapping[str, Any],
    normalized_paths: list[str],
    changed_paths: list[str],
) -> dict[str, Any]:
    if artifact_kind not in {"source_census", "target_plan"}:
        raise GokuFullMotionContractError(
            "canonicalization receipt artifact kind differs"
        )
    if len(set(normalized_paths)) != len(normalized_paths):
        raise GokuFullMotionContractError(
            "canonicalization normalized field paths contain duplicates"
        )
    if any(path not in set(normalized_paths) for path in changed_paths):
        raise GokuFullMotionContractError(
            "canonicalization changed paths are outside normalized paths"
        )
    receipt_without_digest: dict[str, Any] = {
        "schema_version": MODEL_OUTPUT_CANONICALIZATION_RECEIPT_SCHEMA,
        "artifact_kind": artifact_kind,
        "policy": MODEL_OUTPUT_CANONICALIZATION_POLICY,
        "semantic_repair": False,
        "context": copy.deepcopy(dict(context)),
        "raw_sha256": object_sha256(raw),
        "canonical_sha256": object_sha256(canonical),
        "normalized_field_paths": list(normalized_paths),
        "changed_field_paths": list(changed_paths),
    }
    receipt = {
        **receipt_without_digest,
        "receipt_sha256": object_sha256(receipt_without_digest),
    }
    _exact_keys(
        receipt,
        _CANONICALIZATION_RECEIPT_KEYS,
        context="model_output_canonicalization_receipt",
    )
    canonical_json_bytes(receipt)
    return receipt


def canonicalize_source_census_model_output(
    raw: Any, expected_iid: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive only safe redundant source fields from authoritative structure.

    ``raw`` must already be one directly parsed JSON object.  Text extraction,
    malformed JSON repair, entity invention, unit reordering, component
    merging, and camera repair are deliberately outside this function.
    """

    expected = _text(
        expected_iid,
        context="source canonicalization expected_iid",
        max_length=128,
    )
    if _IID_RE.fullmatch(expected) is None:
        raise GokuFullMotionContractError(
            "source canonicalization expected_iid is unsafe"
        )
    raw_mapping = _mapping(raw, context="raw source census model output")
    raw_object = copy.deepcopy(dict(raw_mapping))
    canonical_json_bytes(raw_object)
    if raw_object.get("iid") != expected:
        raise GokuFullMotionContractError(
            "raw source census iid differs from expected_iid"
        )
    _closed_keys_with_optional_redundancy(
        raw_object,
        _SOURCE_CENSUS_KEYS,
        {"i0_visible_entities"},
        context="raw source census",
    )
    candidate = copy.deepcopy(raw_object)
    normalized_paths: list[str] = []
    changed_paths: list[str] = []

    registry_raw = _list(
        candidate.get("i0_entity_registry"),
        context="raw source census.i0_entity_registry",
    )
    if not 1 <= len(registry_raw) <= 24:
        raise GokuFullMotionContractError(
            "raw source census I0 registry must contain one to 24 entries"
        )
    registry: list[dict[str, Any]] = []
    grounding_counts: dict[tuple[str, str], int] = {}
    for index, raw_entity in enumerate(registry_raw):
        context = f"raw source census.i0_entity_registry[{index}]"
        entity_mapping = _mapping(raw_entity, context=context)
        _closed_keys_with_optional_redundancy(
            entity_mapping,
            _SOURCE_REGISTRY_KEYS,
            {"viewer_region", "region_ordinal"},
            context=context,
        )
        entity = copy.deepcopy(dict(entity_mapping))
        bbox = _i0_bbox(
            entity.get("i0_bbox_xyxy_1000"),
            context=f"{context}.i0_bbox_xyxy_1000",
        )
        entity_type = _enum(
            entity.get("entity_type"),
            ENTITY_TYPES,
            context=f"{context}.entity_type",
        )
        _text(
            entity.get("stable_reference"),
            context=f"{context}.stable_reference",
            max_length=MAX_STABLE_REFERENCE_CHARS,
        )
        _enum(entity.get("role"), I0_ENTITY_ROLES, context=f"{context}.role")
        region = _viewer_region_for_bbox(bbox)
        grounding_key = (entity_type, region)
        ordinal = grounding_counts.get(grounding_key, 0) + 1
        grounding_counts[grounding_key] = ordinal
        _set_redundant_field(
            entity,
            "viewer_region",
            region,
            path=f"i0_entity_registry[{index}].viewer_region",
            normalized_paths=normalized_paths,
            changed_paths=changed_paths,
            reject_present_conflict=False,
        )
        _set_redundant_field(
            entity,
            "region_ordinal",
            ordinal,
            path=f"i0_entity_registry[{index}].region_ordinal",
            normalized_paths=normalized_paths,
            changed_paths=changed_paths,
            reject_present_conflict=False,
        )
        registry.append(entity)
    candidate["i0_entity_registry"] = registry

    registry_by_id: dict[str, dict[str, Any]] = {}
    for index, entity in enumerate(registry):
        entity_id = _text(
            entity.get("entity_id"),
            context=(
                f"raw source census.i0_entity_registry[{index}].entity_id"
            ),
        )
        if entity_id in registry_by_id:
            raise GokuFullMotionContractError(
                "raw source census I0 registry entity IDs contain duplicates"
            )
        registry_by_id[entity_id] = entity

    # ``i0_visible_entities`` is review-only prose.  The closed schema and
    # both census prompts explicitly make the structured registry
    # authoritative, so model-written aggregate prose here cannot override
    # identity or grounding.  Rebuild it deterministically even when present.
    # This is normalization, not semantic repair: no entity is added, removed,
    # reordered, or renamed beyond the already supplied authoritative rows.
    visible_references = [str(item["stable_reference"]) for item in registry]
    _set_redundant_field(
        candidate,
        "i0_visible_entities",
        visible_references,
        path="i0_visible_entities",
        normalized_paths=normalized_paths,
        changed_paths=changed_paths,
        reject_present_conflict=False,
    )

    dynamic_raw = _list(
        candidate.get("dynamic_units"),
        context="raw source census.dynamic_units",
    )
    dynamic: list[dict[str, Any]] = []
    for index, raw_unit in enumerate(dynamic_raw):
        context = f"raw source census.dynamic_units[{index}]"
        unit_mapping = _mapping(raw_unit, context=context)
        _closed_keys_with_optional_redundancy(
            unit_mapping,
            _SOURCE_DYNAMIC_KEYS,
            {"entity_type", "stable_reference", "visible_at_i0"},
            context=context,
        )
        unit = copy.deepcopy(dict(unit_mapping))
        entity_id = _text(unit.get("entity_id"), context=f"{context}.entity_id")
        entity = registry_by_id.get(entity_id)
        if entity is None or entity.get("role") != "dynamic_subject":
            raise GokuFullMotionContractError(
                f"{context}.entity_id must already bind a dynamic-subject "
                "registry entity before canonicalization"
            )
        for field in ("entity_type", "stable_reference", "visible_at_i0"):
            _set_redundant_field(
                unit,
                field,
                entity[field],
                path=f"dynamic_units[{index}].{field}",
                normalized_paths=normalized_paths,
                changed_paths=changed_paths,
                reject_present_conflict=True,
            )
        dynamic.append(unit)
    candidate["dynamic_units"] = dynamic

    static_raw = _list(
        candidate.get("static_salient_people"),
        context="raw source census.static_salient_people",
    )
    static: list[dict[str, Any]] = []
    for index, raw_unit in enumerate(static_raw):
        context = f"raw source census.static_salient_people[{index}]"
        unit_mapping = _mapping(raw_unit, context=context)
        _closed_keys_with_optional_redundancy(
            unit_mapping,
            _SOURCE_STATIC_KEYS,
            {"entity_type", "stable_reference", "visible_at_i0"},
            context=context,
        )
        unit = copy.deepcopy(dict(unit_mapping))
        entity_id = _text(unit.get("entity_id"), context=f"{context}.entity_id")
        entity = registry_by_id.get(entity_id)
        if entity is None or entity.get("role") != "static_salient":
            raise GokuFullMotionContractError(
                f"{context}.entity_id must already bind a static-salient "
                "registry entity before canonicalization"
            )
        for field in ("entity_type", "stable_reference", "visible_at_i0"):
            _set_redundant_field(
                unit,
                field,
                entity[field],
                path=f"static_salient_people[{index}].{field}",
                normalized_paths=normalized_paths,
                changed_paths=changed_paths,
                reject_present_conflict=True,
            )
        static.append(unit)
    candidate["static_salient_people"] = static

    canonical = validate_source_census(candidate)
    receipt = _build_canonicalization_receipt(
        artifact_kind="source_census",
        context={"expected_iid": expected},
        raw=raw_object,
        canonical=canonical,
        normalized_paths=normalized_paths,
        changed_paths=changed_paths,
    )
    return canonical, receipt


def validate_source_census_canonicalization(
    raw: Any,
    canonical: Any,
    receipt: Any,
    expected_iid: str,
) -> dict[str, Any]:
    """Recompute and exactly verify a source canonicalization receipt."""

    expected_canonical, expected_receipt = (
        canonicalize_source_census_model_output(raw, expected_iid)
    )
    supplied_canonical = _mapping(
        canonical, context="canonical source census model output"
    )
    if dict(supplied_canonical) != expected_canonical:
        raise GokuFullMotionContractError(
            "canonical source census differs from deterministic reconstruction"
        )
    supplied_receipt = _mapping(
        receipt, context="source census canonicalization receipt"
    )
    _exact_keys(
        supplied_receipt,
        _CANONICALIZATION_RECEIPT_KEYS,
        context="source census canonicalization receipt",
    )
    if dict(supplied_receipt) != expected_receipt:
        raise GokuFullMotionContractError(
            "source census canonicalization receipt differs from reconstruction"
        )
    return copy.deepcopy(expected_receipt)


def canonicalize_target_plan_model_output(
    raw: Any, source_census: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive only target identity redundancies after exact list closure."""

    source = validate_source_census(source_census)
    raw_mapping = _mapping(raw, context="raw target plan model output")
    raw_object = copy.deepcopy(dict(raw_mapping))
    canonical_json_bytes(raw_object)
    _closed_keys_with_optional_redundancy(
        raw_object,
        _TARGET_PLAN_KEYS,
        set(),
        context="raw target plan",
    )
    candidate = copy.deepcopy(raw_object)
    normalized_paths: list[str] = []
    changed_paths: list[str] = []

    dynamic_sources = list(source["dynamic_units"])
    dynamic_raw = _list(
        candidate.get("dynamic_unit_targets"),
        context="raw target plan.dynamic_unit_targets",
    )
    if len(dynamic_raw) != len(dynamic_sources):
        raise GokuFullMotionContractError(
            "raw target plan must contain exactly one target for every "
            "dynamic unit before canonicalization"
        )
    dynamic: list[dict[str, Any]] = []
    for index, (raw_target, source_unit) in enumerate(
        zip(dynamic_raw, dynamic_sources, strict=True)
    ):
        context = f"raw target plan.dynamic_unit_targets[{index}]"
        target_mapping = _mapping(raw_target, context=context)
        _closed_keys_with_optional_redundancy(
            target_mapping,
            _TARGET_DYNAMIC_KEYS,
            {"entity_id", "stable_reference"},
            context=context,
        )
        target = copy.deepcopy(dict(target_mapping))
        expected_unit_id = str(source_unit["unit_id"])
        if target.get("unit_id") != expected_unit_id:
            raise GokuFullMotionContractError(
                f"{context}.unit_id must already equal {expected_unit_id!r} "
                "before canonicalization"
            )
        for field in ("entity_id", "stable_reference"):
            _set_redundant_field(
                target,
                field,
                source_unit[field],
                path=f"dynamic_unit_targets[{index}].{field}",
                normalized_paths=normalized_paths,
                changed_paths=changed_paths,
                reject_present_conflict=True,
            )
        dynamic.append(target)
    candidate["dynamic_unit_targets"] = dynamic

    static_sources = list(source["static_salient_people"])
    static_raw = _list(
        candidate.get("static_person_targets"),
        context="raw target plan.static_person_targets",
    )
    if len(static_raw) != len(static_sources):
        raise GokuFullMotionContractError(
            "raw target plan must constrain every salient static entity "
            "before canonicalization"
        )
    static: list[dict[str, Any]] = []
    for index, (raw_target, source_unit) in enumerate(
        zip(static_raw, static_sources, strict=True)
    ):
        context = f"raw target plan.static_person_targets[{index}]"
        target_mapping = _mapping(raw_target, context=context)
        _closed_keys_with_optional_redundancy(
            target_mapping,
            _TARGET_STATIC_KEYS,
            {"entity_id", "entity_type", "stable_reference"},
            context=context,
        )
        target = copy.deepcopy(dict(target_mapping))
        expected_unit_id = str(source_unit["unit_id"])
        if target.get("unit_id") != expected_unit_id:
            raise GokuFullMotionContractError(
                f"{context}.unit_id must already equal {expected_unit_id!r} "
                "before canonicalization"
            )
        for field in ("entity_id", "entity_type", "stable_reference"):
            _set_redundant_field(
                target,
                field,
                source_unit[field],
                path=f"static_person_targets[{index}].{field}",
                normalized_paths=normalized_paths,
                changed_paths=changed_paths,
                reject_present_conflict=True,
            )
        static.append(target)
    candidate["static_person_targets"] = static

    canonical = validate_target_plan(candidate, source_census=source)
    receipt = _build_canonicalization_receipt(
        artifact_kind="target_plan",
        context={
            "iid": source["iid"],
            "source_census_sha256": object_sha256(source),
        },
        raw=raw_object,
        canonical=canonical,
        normalized_paths=normalized_paths,
        changed_paths=changed_paths,
    )
    return canonical, receipt


def validate_target_plan_canonicalization(
    raw: Any,
    canonical: Any,
    receipt: Any,
    source_census: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute and exactly verify a target canonicalization receipt."""

    expected_canonical, expected_receipt = canonicalize_target_plan_model_output(
        raw, source_census
    )
    supplied_canonical = _mapping(
        canonical, context="canonical target plan model output"
    )
    if dict(supplied_canonical) != expected_canonical:
        raise GokuFullMotionContractError(
            "canonical target plan differs from deterministic reconstruction"
        )
    supplied_receipt = _mapping(
        receipt, context="target plan canonicalization receipt"
    )
    _exact_keys(
        supplied_receipt,
        _CANONICALIZATION_RECEIPT_KEYS,
        context="target plan canonicalization receipt",
    )
    if dict(supplied_receipt) != expected_receipt:
        raise GokuFullMotionContractError(
            "target plan canonicalization receipt differs from reconstruction"
        )
    return copy.deepcopy(expected_receipt)


def validate_coverage_critic(
    value: Any,
    *,
    source_census: Mapping[str, Any],
    target_plan: Mapping[str, Any],
    compiled_instruction: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an independent all-unit/instruction coverage decision.

    This validator intentionally does not import the compiler module (which
    itself imports this contract).  It binds the compiler's public fields and
    requires exact ID closure; callers should additionally invoke
    ``validate_compiled_instruction`` before accepting a critic record.
    """

    source = validate_source_census(source_census)
    plan = validate_target_plan(target_plan, source_census=source)
    compiled = _mapping(compiled_instruction, context="compiled_instruction")
    critic = _mapping(value, context="coverage_critic")
    _exact_keys(
        critic,
        {
            "schema_version",
            "iid",
            "source_census_sha256",
            "target_plan_sha256",
            "instruction_sha256",
            "required_dynamic_unit_ids",
            "plan_covered_dynamic_unit_ids",
            "instruction_covered_dynamic_unit_ids",
            "missing_unit_ids",
            "extra_unit_ids",
            "ambiguous_unit_ids",
            "per_unit_substantive_change",
            "source_future_suppressed_or_explicit",
            "camera_clause_present",
            "camera_target_valid",
            "required_static_person_ids",
            "static_people_preserved",
            "i0_executable",
            "no_new_prerequisites",
            "no_unrequested_action",
            "verdict",
            "uncertainty_codes",
            "confidence",
        },
        context="coverage_critic",
    )
    if critic.get("schema_version") != COVERAGE_CRITIC_SCHEMA:
        raise GokuFullMotionContractError("coverage_critic schema differs")
    if critic.get("iid") != source["iid"]:
        raise GokuFullMotionContractError("coverage_critic.iid differs")
    expected_source_sha = object_sha256(source)
    expected_plan_sha = object_sha256(plan)
    if critic.get("source_census_sha256") != expected_source_sha:
        raise GokuFullMotionContractError(
            "coverage_critic.source_census_sha256 differs"
        )
    if critic.get("target_plan_sha256") != expected_plan_sha:
        raise GokuFullMotionContractError(
            "coverage_critic.target_plan_sha256 differs"
        )
    compiled_source_sha = compiled.get("source_census_sha256")
    compiled_plan_sha = compiled.get("target_plan_sha256")
    compiled_instruction_sha = compiled.get("instruction_sha256")
    instruction = compiled.get("edit_instruction")
    if (
        compiled_source_sha != expected_source_sha
        or compiled_plan_sha != expected_plan_sha
        or type(instruction) is not str
        or compiled_instruction_sha != sha256_text(instruction)
        or critic.get("instruction_sha256") != compiled_instruction_sha
    ):
        raise GokuFullMotionContractError(
            "coverage_critic/compiler digest binding differs"
        )

    dynamic_ids = [str(item["unit_id"]) for item in source["dynamic_units"]]
    static_ids = [
        str(item["unit_id"]) for item in source["static_salient_people"]
    ]
    list_fields = {
        "required_dynamic_unit_ids": dynamic_ids,
        "plan_covered_dynamic_unit_ids": dynamic_ids,
        "instruction_covered_dynamic_unit_ids": dynamic_ids,
        "missing_unit_ids": [],
        "extra_unit_ids": [],
        "ambiguous_unit_ids": [],
        "required_static_person_ids": static_ids,
    }
    for field, expected in list_fields.items():
        actual = _ordered_unique_strings(
            critic.get(field),
            context=f"coverage_critic.{field}",
            allow_empty=True,
        )
        if actual != expected:
            raise GokuFullMotionContractError(
                f"coverage_critic.{field} differs: "
                f"expected={expected!r} actual={actual!r}"
            )

    for field in (
        "per_unit_substantive_change",
        "source_future_suppressed_or_explicit",
    ):
        record = _mapping(critic.get(field), context=f"coverage_critic.{field}")
        _exact_keys(record, set(dynamic_ids), context=f"coverage_critic.{field}")
        for unit_id in dynamic_ids:
            _true(record.get(unit_id), context=f"coverage_critic.{field}.{unit_id}")
    static_record = _mapping(
        critic.get("static_people_preserved"),
        context="coverage_critic.static_people_preserved",
    )
    _exact_keys(
        static_record,
        set(static_ids),
        context="coverage_critic.static_people_preserved",
    )
    for unit_id in static_ids:
        _true(
            static_record.get(unit_id),
            context=f"coverage_critic.static_people_preserved.{unit_id}",
        )

    entity_clauses = _mapping(
        compiled.get("entity_clauses"),
        context="compiled_instruction.entity_clauses",
    )
    if set(entity_clauses) != set((*dynamic_ids, *static_ids)):
        raise GokuFullMotionContractError(
            "compiled_instruction.entity_clauses does not exactly cover all units"
        )
    if type(compiled.get("camera_clause")) is not str or not str(
        compiled["camera_clause"]
    ):
        raise GokuFullMotionContractError(
            "compiled_instruction.camera_clause is missing"
        )
    for field in (
        "camera_clause_present",
        "camera_target_valid",
        "i0_executable",
        "no_new_prerequisites",
        "no_unrequested_action",
    ):
        _true(critic.get(field), context=f"coverage_critic.{field}")
    if critic.get("verdict") != "pass":
        raise GokuFullMotionContractError(
            "coverage_critic.verdict must be exactly 'pass'"
        )
    if critic.get("uncertainty_codes") != []:
        raise GokuFullMotionContractError(
            "coverage_critic.uncertainty_codes must be exactly []"
        )
    if critic.get("confidence") != "high":
        raise GokuFullMotionContractError(
            "coverage_critic.confidence must be exactly 'high'"
        )
    canonical_json_bytes(critic)
    return copy.deepcopy(dict(critic))


def build_contract(
    *, source_census: Mapping[str, Any], target_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Build a small immutable binding after validating both closed records."""

    source = validate_source_census(source_census)
    target = validate_target_plan(target_plan, source_census=source)
    return {
        "schema_version": CONTRACT_SCHEMA,
        "policy": CONTRACT_POLICY,
        "iid": source["iid"],
        "source_census_sha256": object_sha256(source),
        "target_plan_sha256": object_sha256(target),
        "dynamic_unit_ids": [
            str(item["unit_id"]) for item in source["dynamic_units"]
        ],
        "static_person_ids": [
            str(item["unit_id"]) for item in source["static_salient_people"]
        ],
        "camera_id": "camera",
        "all_dynamic_units_changed": True,
        "camera_explicit": True,
    }


def validate_contract_binding(
    value: Any,
    *,
    source_census: Mapping[str, Any],
    target_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify that ``value`` is byte-semantically identical to a rebuilt binding."""

    binding = _mapping(value, context="full_motion_contract")
    expected = build_contract(source_census=source_census, target_plan=target_plan)
    _exact_keys(binding, set(expected), context="full_motion_contract")
    if dict(binding) != expected:
        raise GokuFullMotionContractError("full_motion_contract binding differs")
    return copy.deepcopy(dict(binding))


__all__ = [
    "CAMERA_MOTION_CLASSES",
    "CAMERA_RELATIONS",
    "CLIP_SCHEMA",
    "CONTRACT_POLICY",
    "CONTRACT_SCHEMA",
    "COVERAGE_CRITIC_SCHEMA",
    "ENTITY_TYPES",
    "I0_ENTITY_ROLES",
    "FPS",
    "FRAME_COUNT",
    "GokuFullMotionContractError",
    "MAX_DYNAMIC_UNITS",
    "MAX_STATIC_SALIENT_ENTITIES",
    "MAX_STATIC_SALIENT_PEOPLE",
    "MODEL_OUTPUT_CANONICALIZATION_POLICY",
    "MODEL_OUTPUT_CANONICALIZATION_RECEIPT_SCHEMA",
    "MOTION_EVIDENCE_SCHEMA",
    "MOTION_COMPONENT_TYPES",
    "MOTION_RELATIONS",
    "COMPONENT_DISPOSITIONS",
    "SOURCE_CAMERA_SCHEMA",
    "SOURCE_CENSUS_SCHEMA",
    "SOURCE_DYNAMIC_UNIT_SCHEMA",
    "SOURCE_I0_ENTITY_SCHEMA",
    "SOURCE_INVENTORY_ALIGNMENT_SCHEMA",
    "SOURCE_MOTION_COMPONENT_SCHEMA",
    "SOURCE_STATIC_PERSON_SCHEMA",
    "SOURCE_STATIC_ENTITY_SCHEMA",
    "TARGET_CAMERA_SCHEMA",
    "TARGET_COMPONENT_DISPOSITION_SCHEMA",
    "TARGET_COVERAGE_SCHEMA",
    "TARGET_DYNAMIC_UNIT_SCHEMA",
    "TARGET_PLAN_SCHEMA",
    "TARGET_PRESERVATION_SCHEMA",
    "TARGET_STATIC_PERSON_SCHEMA",
    "TARGET_STATIC_ENTITY_SCHEMA",
    "TIMELINE_SPAN_SECONDS",
    "VIEWER_REGIONS",
    "build_contract",
    "build_source_inventory_alignment",
    "canonicalize_source_census_model_output",
    "canonicalize_target_plan_model_output",
    "canonical_json_bytes",
    "object_sha256",
    "sha256_text",
    "source_inventory_projection",
    "validate_contract_binding",
    "validate_coverage_critic",
    "validate_source_census",
    "validate_source_census_canonicalization",
    "validate_source_inventory_alignment",
    "validate_target_plan",
    "validate_target_plan_canonicalization",
]
