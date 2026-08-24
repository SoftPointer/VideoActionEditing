"""High-recall, auditable rules for temporal action-edit candidate mining."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Iterable

from .semantics import classify_instruction


RULE_VERSION = "goku-action-rules-v3"


@dataclass(frozen=True)
class RuleDecision:
    label: str
    action_families: tuple[str, ...]
    actors: tuple[str, ...]
    score: float
    tier: str
    positive_cues: tuple[str, ...]
    negative_cues: tuple[str, ...]
    reasons: tuple[str, ...]
    version: str = RULE_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ACTION_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "walk",
        (
            r"\bwalk(?:s|ing|ed)?\b",
            r"\bstep(?:ping|ped)\b",
            r"\b(?:take|takes|taking|took) (?:a )?step\b",
            r"\bsteps? (?:forward|backward|left|right|towards?|away)\b",
        ),
    ),
    (
        "run",
        (
            r"\b(?:run|runs|running|ran)\b",
            r"\b(?:jog|jogs|jogging|jogged)\b",
        ),
    ),
    ("jump", (r"\bjump(?:s|ing|ed)?\b", r"\bleap(?:s|ing|ed)?\b", r"\bhop(?:s|ping|ped)?\b")),
    ("dance", (r"\bdanc(?:e|es|ing|ed)\b",)),
    (
        "fly",
        (
            r"\b(?:fly|flies|flying|flew|flown)\b",
            r"\b(?:take|takes|taking|took|taken) off\b",
        ),
    ),
    (
        "swim",
        (
            r"\b(?:swim|swims|swimming|swam|swum)\b",
            r"\b(?:dive|dives|dived|diving|dove)\b",
        ),
    ),
    ("crawl", (r"\bcrawl(?:s|ing|ed)?\b",)),
    ("climb", (r"\bclimb(?:s|ing|ed)?\b",)),
    (
        "throw",
        (
            r"\b(?:throw|throws|throwing|threw|thrown)\b",
            r"\btoss(?:es|ing|ed)?\b",
        ),
    ),
    ("catch", (r"\b(?:catch|catches|catching|caught)\b",)),
    ("kick", (r"\bkick(?:s|ing|ed)?\b",)),
    ("punch", (r"\bpunch(?:es|ing|ed)?\b",)),
    ("bounce", (r"\bbounc(?:e|es|ing|ed)\b",)),
    ("roll", (r"\broll(?:s|ing|ed)?\b",)),
    (
        "spin",
        (
            r"\b(?:spin|spins|spinning|spun)\b",
            r"\btwirl(?:s|ing|ed)?\b",
        ),
    ),
    ("swing", (r"\b(?:swing|swings|swinging|swung)\b",)),
    (
        "fall",
        (
            r"\b(?:fall|falls|falling|fell|fallen)\b",
            r"\btrip(?:s|ping|ped)?\b",
        ),
    ),
    ("pour", (r"\bpour(?:s|ing|ed)?\b", r"\bdrip(?:s|ping|ped)?\b")),
    ("splash", (r"\bsplash(?:es|ing|ed)?\b",)),
    ("explode", (r"\bexplod(?:e|es|ing|ed)\b", r"\bburst(?:s|ing)?\b")),
    ("pick", (r"\bpick(?:s|ing|ed)? up\b", r"\bpick(?:s|ing|ed)?\b")),
    ("reach", (r"\breach(?:es|ing|ed)?\b",)),
    ("ride", (r"\b(?:ride|rides|riding|rode|ridden)\b",)),
    ("drive", (r"\b(?:drive|drives|driving|drove|driven)\b",)),
    ("land", (r"\bland(?:s|ing|ed)?\b",)),
    ("wave", (r"\b(?:wave|waves|waved|waving)\b",)),
    ("nod", (r"\bnod(?:s|ding|ded)?\b",)),
    ("shake", (r"\b(?:shake|shakes|shaking|shook|shaken)\b",)),
    ("grab", (r"\bgrab(?:s|bing|bed)?\b", r"\bgrasp(?:s|ing|ed)?\b")),
    ("release", (r"\breleas(?:e|es|ing|ed)\b", r"\blet go\b")),
    ("drop", (r"\bdrop(?:s|ping|ped)?\b",)),
    ("push", (r"\bpush(?:es|ing|ed)?\b",)),
    ("pull", (r"\bpull(?:s|ing|ed)?\b",)),
    ("enter", (r"\benter(?:s|ing|ed)?\b", r"\bwalk .{0,25}\binto\b")),
    (
        "exit",
        (
            r"\bexit(?:s|ing|ed)?\b",
            r"\b(?:leave|leaves|leaving|left) (?:the )?(?:scene|frame|room|building|house|area|vehicle|car|stage)\b",
        ),
    ),
    ("approach", (r"\bapproach(?:es|ing|ed)?\b", r"\bmove .{0,20}\btowards?\b")),
    ("turn", (r"\bturn(?:s|ing|ed)? around\b", r"\bturn .{0,30}\band (?:walk|run|move)\b")),
    (
        "stand_up",
        (
            r"\bstand(?:s|ing)? up\b",
            r"\b(?:get|gets|getting|got|gotten) up\b",
        ),
    ),
    ("sit_down", (r"\bsit(?:s|ting)? down\b",)),
    ("kneel_down", (r"\bkneel(?:s|ing|ed)? down\b",)),
    (
        "lie_down",
        (
            r"\b(?:lie|lies|lying|lay|lain) down\b",
        ),
    ),
    (
        "open_close",
        (
            r"\bopen(?:s|ing|ed)? (?:the|a|an|his|her|their|its|both|this|that|front|rear|left|right)\b",
            r"\bclos(?:e|es|ing|ed) (?:the|a|an|his|her|their|its|both|this|that|front|rear|left|right)\b",
            r"\b(?:door|gate|lid|eyes?|mouth|drawer|cabinet|umbrella|flower|petals?) (?:open|opens|opening|close|closes|closing)\b",
            r"\bunlatch(?:es|ing|ed)?\b",
        ),
    ),
)

SUPPRESSION_PATTERNS = (
    r"\bstop(?:s|ping|ped)?\b",
    r"\bstand(?:ing)? still\b",
    r"\bcome to (?:a )?rest\b",
    r"\bfreeze\b",
    r"\bcease(?:s|d)?\b",
)

TEMPORAL_CUES = (
    r"\bcontinuously\b",
    r"\brepeated(?:ly)?\b",
    r"\bmultiple times\b",
    r"\bthen\b",
    r"\bwhile\b",
    r"\bfrom .{0,50}\bto\b",
    r"\bbegin(?:s|ning)?\b",
    r"\bstart(?:s|ing|ed)?\b",
    r"\bgradually\b",
    r"\bacross\b",
    r"\btowards?\b",
    r"\baway from\b",
)

ENDPOINT_NEGATIVE_PATTERNS = (
    r"\bappear(?:s|ing)? to (?:be|have)\b",
    r"\bas if\b",
    r"\bpoised to\b",
    r"\bsuspended\b",
    r"\bready to\b",
    r"\bpose[ds]?\b",
    r"\bmid[- ](?:jump|air|stride|motion)\b",
    r"\blook(?:s|ing)? like\b",
)

APPEARANCE_NEGATIVE_PATTERNS = (
    r"\b(?:change|make).{0,30}\b(?:color|texture|material|style)\b",
    r"\breshape\b",
    r"\benlarge\b",
    r"\bshrink\b",
    r"\bthick(?:er|en)?\b",
    r"\bthin(?:ner)?\b",
    r"\bremove\b",
    r"\breplace\b",
)

ENVIRONMENT_PATTERNS = (
    r"\bwaterfall\b",
    r"\briver\b",
    r"\bwaves?\b",
    r"\bclouds?\b",
    r"\bsmoke\b",
    r"\bflames?\b",
    r"\bfire\b",
    r"\bsnow\b",
    r"\brain\b",
    r"\bfoliage\b",
    r"\bbranches?\b",
)

ACTOR_PATTERNS: tuple[tuple[str, str], ...] = (
    ("person", r"\b(?:person|man|woman|child|boy|girl|runner|rider|climber|skier|surfer)\b"),
    ("animal", r"\b(?:dog|cat|kitten|bird|monkey|horse|flamingo|animal)\b"),
    ("vehicle", r"\b(?:car|truck|motorcycle|bike|bicycle|jet ski|boat|vehicle)\b"),
    ("object", r"\b(?:ball|bottle|door|clasp|object|fruit|flower|box|toy)\b"),
)

KNOWN_FALSE_CONTEXTS = (
    r"\bride height\b",
    r"\btrain ride\b",
    r"\bamusement (?:park )?ride\b",
    r"\bswing set\b",
    r"\broll[- ]down door\b",
    r"\brunning decks?\b",
    r"\bfor someone to (?:walk|pass)\b",
    r"\bwalk space\b",
)


def _matched(patterns: Iterable[str], text: str) -> tuple[str, ...]:
    return tuple(pattern for pattern in patterns if re.search(pattern, text))


def _action_families(text: str) -> tuple[str, ...]:
    return tuple(
        family
        for family, patterns in ACTION_PATTERNS
        if any(re.search(pattern, text) for pattern in patterns)
    )


def stable_group_split(
    *,
    source_video: str,
    source_caption: str = "",
    source_group_key: str | None = None,
    seed: int = 260108828,
) -> tuple[str, str]:
    """Return a stable group digest and 80/10/10 split.

    ``source_group_key`` should be a content-derived source fingerprint when
    available.  Caption/path grouping remains a rule-stage fallback only.
    """

    normalized_caption = re.sub(r"\s+", " ", source_caption.strip().lower())
    if source_group_key is not None:
        normalized_group_key = source_group_key.strip().lower()
        if not normalized_group_key:
            raise ValueError("source_group_key must be non-empty when provided")
        group_material = f"source_content\0{normalized_group_key}"
    else:
        group_material = normalized_caption or source_video
    digest = hashlib.sha256(
        f"{seed}\0{group_material}".encode("utf-8")
    ).hexdigest()
    bucket = int(digest[:8], 16) % 100
    split = "train" if bucket < 80 else ("validation" if bucket < 90 else "test")
    return digest, split


def score_action_rule(
    instruction: str,
    *,
    source_caption: str = "",
    edited_caption: str = "",
    use_edited_caption: bool = False,
) -> RuleDecision:
    text = re.sub(r"\s+", " ", instruction.strip().lower())
    source_text = re.sub(r"\s+", " ", source_caption.strip().lower())
    target_text = re.sub(r"\s+", " ", edited_caption.strip().lower())
    # edited_caption may have been generated from the edited video (or copied
    # from the requested answer).  Keep it out of the default decision path so
    # curation cannot silently turn into target-label leakage.
    evidence_text = (
        f"{text} {source_text} {target_text}"
        if use_edited_caption
        else f"{text} {source_text}"
    ).strip()

    action_families = _action_families(text)
    caption_actions = _action_families(target_text) if use_edited_caption else ()
    suppression = _matched(SUPPRESSION_PATTERNS, text)
    temporal = _matched(TEMPORAL_CUES, text)
    endpoint = _matched(ENDPOINT_NEGATIVE_PATTERNS, text)
    appearance = _matched(APPEARANCE_NEGATIVE_PATTERNS, text)
    environmental = _matched(ENVIRONMENT_PATTERNS, evidence_text)
    false_contexts = _matched(KNOWN_FALSE_CONTEXTS, text)
    actors = tuple(
        name for name, pattern in ACTOR_PATTERNS if re.search(pattern, evidence_text)
    )
    source_actions = _action_families(source_text)

    score = 0.05
    reasons: list[str] = []
    positive: list[str] = []
    negative: list[str] = []
    if action_families:
        score += 0.48 + min(0.12, 0.04 * (len(action_families) - 1))
        positive.extend(f"action:{family}" for family in action_families)
        reasons.append("instruction names a temporal action")
    if suppression:
        score += 0.48
        positive.extend(f"suppression:{cue}" for cue in suppression)
        reasons.append("instruction explicitly changes motion to rest")
    if temporal:
        score += min(0.18, 0.08 + 0.03 * len(temporal))
        positive.extend(f"temporal:{cue}" for cue in temporal)
        reasons.append("instruction contains temporal extent/phase cues")
    if actors:
        score += 0.08
        positive.extend(f"actor:{actor}" for actor in actors)
    novel_caption_actions = tuple(
        family for family in caption_actions if family not in source_actions
    )
    if novel_caption_actions:
        score += 0.12
        positive.extend(f"target_caption_action:{family}" for family in novel_caption_actions)
        reasons.append("edited caption adds an action absent from source caption")
    if endpoint:
        score -= 0.38
        negative.extend(f"endpoint:{cue}" for cue in endpoint)
        reasons.append("endpoint/as-if wording may describe a static pose")
    if appearance:
        score -= 0.42
        negative.extend(f"appearance:{cue}" for cue in appearance)
    if environmental and not actors:
        score -= 0.25
        negative.extend(f"environment:{cue}" for cue in environmental)
    if false_contexts:
        score -= 0.55
        negative.extend(f"false_context:{cue}" for cue in false_contexts)

    score = min(max(score, 0.0), 1.0)
    legacy_label = classify_instruction(instruction).label
    if suppression:
        label = "motion_suppression"
    elif action_families and endpoint:
        label = "endpoint_risk"
    elif action_families:
        label = "temporal_action"
    elif legacy_label == "endpoint_pose":
        label = "endpoint_pose"
    elif environmental and not actors:
        label = "environmental_motion"
    else:
        label = legacy_label
    tier = "high" if score >= 0.70 else ("possible" if score >= 0.40 else "reject")
    all_families = tuple(dict.fromkeys((*action_families, *caption_actions)))
    return RuleDecision(
        label=label,
        action_families=all_families,
        actors=actors,
        score=round(score, 6),
        tier=tier,
        positive_cues=tuple(positive),
        negative_cues=tuple(negative),
        reasons=tuple(reasons),
    )
