#!/usr/bin/env python3
"""Frozen CPU-only identities for the V8 LOAO action-carrier diagnostic.

This module contains prompt/identity authority only.  It cannot load a model,
launch a device job, train, decode, render, or route a representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence


METHOD = "bernini-self-generated-loao-action-tube-graph-probe-v8"
SCHEMA_VERSION = "bernini-self-generated-loao-action-registry-v8"
APPEARANCE_IDS = ("appearance_0", "appearance_1", "appearance_2")
ACTION_IDS = ("transfer", "lift_pause_return", "clockwise_orbit_return")
ARMS = (
    "primary",
    "paraphrase",
    "reverse",
    "lexical_placebo",
    "null_a",
    "null_b",
    "noop",
    "neutral",
)
BLOCKS = (6, 12, 18, 24)
SIGMA_CELL_INDICES = MappingProxyType({"high": 18, "mid": 32, "low": 38})
SEED_IDS = (2026082308, 2026082309)
PHASES, PATCH_HEIGHT, PATCH_WIDTH = 21, 37, 25
PREREG_PATH = (
    Path(__file__).absolute().parent
    / "assets"
    / "self_generated_loao_action_tube_graph_prereg_v8.json"
)
NEUTRAL_CAPTION = (
    "A locked-camera scene with an unspecified foreground and a still "
    "background. Nothing is named or assigned a semantic role."
)


class RegistryV8Error(ValueError):
    """A frozen prompt, embedding, state, runtime, or count identity differs."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise RegistryV8Error("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise RegistryV8Error("text digest requires a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hex_digest(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RegistryV8Error(f"{label} is not one SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise RegistryV8Error(f"{label} is not hexadecimal") from error
    return value


@dataclass(frozen=True)
class AppearanceV8:
    appearance_id: str
    actor: str
    object: str
    source: str
    destination: str

    def __post_init__(self) -> None:
        if self.appearance_id not in APPEARANCE_IDS:
            raise RegistryV8Error("appearance id differs")
        values = (self.actor, self.object, self.source, self.destination)
        if any(not isinstance(item, str) or item != item.strip() or not item for item in values):
            raise RegistryV8Error("appearance noun is malformed")
        if len(set(values)) != 4:
            raise RegistryV8Error("appearance nouns must be distinct")

    def body(self) -> Mapping[str, Any]:
        return {
            "appearance_id": self.appearance_id,
            "actor": self.actor,
            "object": self.object,
            "source": self.source,
            "destination": self.destination,
        }

    @property
    def state_sha256(self) -> str:
        return object_sha256(self.body())


APPEARANCES = (
    AppearanceV8(
        "appearance_0",
        "a young woman in a green jacket",
        "a red ceramic mug",
        "a dark wooden table on the left",
        "a white metal shelf on the right",
    ),
    AppearanceV8(
        "appearance_1",
        "an older man in a blue sweater",
        "a yellow rubber ball",
        "a gray stone bench on the left",
        "a blue plastic bin on the right",
    ),
    AppearanceV8(
        "appearance_2",
        "a small silver robot",
        "a purple wooden block",
        "a black steel platform on the left",
        "an orange tray on the right",
    ),
)
APPEARANCE_BY_ID = MappingProxyType({row.appearance_id: row for row in APPEARANCES})
if len({row.state_sha256 for row in APPEARANCES}) != len(APPEARANCES):
    raise RuntimeError("V8 state SHA-256 identities are not pairwise distinct")


@dataclass(frozen=True)
class ActionProgramV8:
    action_id: str
    dynamics: str

    def __post_init__(self) -> None:
        if self.action_id not in ACTION_IDS or not self.dynamics:
            raise RegistryV8Error("action program differs")

    @property
    def action_sha256(self) -> str:
        return object_sha256({"action_id": self.action_id, "dynamics": self.dynamics})


ACTIONS = (
    ActionProgramV8("transfer", "source-to-destination horizontal transfer"),
    ActionProgramV8("lift_pause_return", "vertical lift, overhead pause, return to source"),
    ActionProgramV8("clockwise_orbit_return", "clockwise closed orbit, return to source"),
)
ACTION_BY_ID = MappingProxyType({row.action_id: row for row in ACTIONS})


def _scene(state: AppearanceV8) -> str:
    return (
        f"A locked-camera scene contains {state.actor}, {state.object}, "
        f"{state.source}, and {state.destination}. "
    )


def _caption(state: AppearanceV8, action_id: str, arm: str) -> str:
    if action_id not in ACTION_IDS or arm not in ARMS:
        raise RegistryV8Error("prompt action or arm differs")
    scene = _scene(state)
    still = " The camera and background remain completely still."
    actor, obj, source, destination = (
        state.actor,
        state.object,
        state.source,
        state.destination,
    )
    if arm == "neutral":
        return NEUTRAL_CAPTION
    if arm == "noop":
        return (
            scene
            + f"{actor} remains motionless; "
            f"{obj} stays unchanged on {source}, and {destination} remains unchanged."
            + still
        )
    if arm == "null_a":
        return (
            scene
            + "No physical motion occurs: "
            f"{actor}, {obj}, {source}, and {destination} keep exactly the same state."
            + still
        )
    if arm == "null_b":
        return (
            scene
            + f"The scene is observed without an event; neither {actor} "
            f"nor {obj} changes position relative to {source} or {destination}."
            + still
        )

    if action_id == "transfer":
        primary = f"{actor} picks up {obj} from {source}, carries it horizontally, and places it on {destination}."
        paraphrase = f"Beginning at {source}, {obj} is lifted by {actor}, moved laterally through the shot, then released at {destination}."
        reverse = f"{actor} picks up {obj} from {destination}, carries it horizontally backward, and places it on {source}."
        words = "pick up, carry horizontally, and place"
    elif action_id == "lift_pause_return":
        primary = f"{actor} lifts {obj} vertically from {source}, pauses with it overhead, then lowers it back onto {source}; {destination} is untouched."
        paraphrase = f"From {source}, {obj} rises straight upward in {actor}'s grasp, waits overhead, and descends to the same {source}; {destination} is untouched."
        reverse = f"Starting overhead, {actor} lowers {obj} vertically to {source}, pauses, then raises it overhead again; {destination} is untouched."
        words = "lift vertically, pause overhead, and return"
    else:
        primary = f"{actor} carries {obj} clockwise in one closed circle around the foreground and returns it to {source}; {destination} is untouched."
        paraphrase = f"Holding {obj}, {actor} traces a clockwise loop through the scene and finishes with it back on {source}; {destination} is untouched."
        reverse = f"{actor} carries {obj} counter-clockwise in one closed circle and returns it to {source}; {destination} is untouched."
        words = "circle clockwise and return"
    if arm == "lexical_placebo":
        return (
            scene
            + f"The words {words} are printed on a stationary card, but {actor}, "
            f"{obj}, {source}, and {destination} remain physically motionless."
            + still
        )
    return scene + {"primary": primary, "paraphrase": paraphrase, "reverse": reverse}[arm] + still


@dataclass(frozen=True)
class PromptV8:
    state_id: str
    action_id: str
    arm: str
    caption: str

    def __post_init__(self) -> None:
        if self.state_id not in APPEARANCE_BY_ID:
            raise RegistryV8Error("prompt state differs")
        expected = _caption(APPEARANCE_BY_ID[self.state_id], self.action_id, self.arm)
        if self.caption != expected or self.caption != self.caption.strip() or "\x00" in self.caption:
            raise RegistryV8Error("prompt caption is not the state-local frozen rendering")
        if self.arm != "neutral":
            own = APPEARANCE_BY_ID[self.state_id]
            own_nouns = (own.actor, own.object, own.source, own.destination)
            if any(noun not in self.caption for noun in own_nouns):
                raise RegistryV8Error("state-local caption omits a required noun")
            foreign = [
                noun
                for row in APPEARANCES
                if row.appearance_id != self.state_id
                for noun in (row.actor, row.object, row.source, row.destination)
            ]
            if any(noun in self.caption for noun in foreign):
                raise RegistryV8Error("caption contains a foreign appearance noun")

    @property
    def prompt_id(self) -> str:
        return f"{self.state_id}:{self.action_id}:{self.arm}"

    @property
    def prompt_sha256(self) -> str:
        return text_sha256(self.caption)


PROMPTS = tuple(
    PromptV8(state_id, action_id, arm, _caption(APPEARANCE_BY_ID[state_id], action_id, arm))
    for state_id in APPEARANCE_IDS
    for action_id in ACTION_IDS
    for arm in ARMS
)
PROMPT_BY_KEY = MappingProxyType({(row.state_id, row.action_id, row.arm): row for row in PROMPTS})
if len(PROMPT_BY_KEY) != 72:
    raise RuntimeError("V8 prompt carrier matrix is not exact 3x3x8")


@dataclass(frozen=True)
class RuntimeIdentityV8:
    runtime_source_sha256: str
    prompt_encoder_sha256: str
    nontext_encoder_sha256: str
    projection_sha256: str
    frozen_state_before_sha256: str
    frozen_state_after_sha256: str
    observer_forward_invocations: int = 432
    projected_block_capture_rows: int = 1728
    b0_observer_absent_forward_invocations: int = 54
    trajectory_forward_invocations: int = 1440
    total_frozen_forward_invocations: int = 1926
    generator_parameter_updates: int = 0
    decoder_forward_invocations: int = 0
    training_steps: int = 0
    route_injection_invocations: int = 0
    causal_editing_renderer_forward_invocations: int = 0

    def __post_init__(self) -> None:
        for name in (
            "runtime_source_sha256", "prompt_encoder_sha256", "nontext_encoder_sha256",
            "projection_sha256", "frozen_state_before_sha256", "frozen_state_after_sha256",
        ):
            _hex_digest(getattr(self, name), name)
        if self.frozen_state_before_sha256 != self.frozen_state_after_sha256:
            raise RegistryV8Error("frozen base is not bit-exact before/after")
        counts = (
            self.observer_forward_invocations,
            self.projected_block_capture_rows,
            self.b0_observer_absent_forward_invocations,
            self.trajectory_forward_invocations,
            self.total_frozen_forward_invocations,
            self.generator_parameter_updates,
            self.decoder_forward_invocations,
            self.training_steps,
            self.route_injection_invocations,
            self.causal_editing_renderer_forward_invocations,
        )
        if counts != (432, 1728, 54, 1440, 1926, 0, 0, 0, 0, 0):
            raise RegistryV8Error("frozen forward/update count contract differs")
        if 432 + 54 + 1440 != self.total_frozen_forward_invocations:
            raise RegistryV8Error("total frozen forward count is not recomputable")

    def body(self) -> Mapping[str, Any]:
        return dict(self.__dict__)

    @property
    def digest(self) -> str:
        return object_sha256(self.body())


@dataclass(frozen=True)
class CaptureBindingV8:
    state_id: str
    action_id: str
    arm: str
    seed_id: int
    sigma_name: str
    sigma_cell_index: int
    block: int
    prompt_id: str
    prompt_sha256: str
    prompt_embedding_sha256: str
    action_sha256: str
    action_embedding_sha256: str
    state_sha256: str
    state_embedding_sha256: str
    middle_nontext_embedding_sha256: str
    neutral_embedding_sha256: str
    carrier_identity_sha256: str
    carrier_state_sha256: str
    timestep_identity_sha256: str
    rotary_identity_sha256: str
    projection_sha256: str
    projected_tensor_sha256: str
    forward_event_sha256: str
    four_block_invocation_sha256: str
    runtime_identity_digest: str

    def __post_init__(self) -> None:
        if (self.state_id, self.action_id, self.arm) not in PROMPT_BY_KEY:
            raise RegistryV8Error("capture prompt key differs")
        if self.seed_id not in SEED_IDS or self.sigma_name not in SIGMA_CELL_INDICES:
            raise RegistryV8Error("capture seed or sigma differs")
        if self.sigma_cell_index != SIGMA_CELL_INDICES[self.sigma_name] or self.block not in BLOCKS:
            raise RegistryV8Error("capture sigma cell or block differs")
        expected_prompt = PROMPT_BY_KEY[(self.state_id, self.action_id, self.arm)]
        expected_action = ACTION_BY_ID[self.action_id]
        expected_state = APPEARANCE_BY_ID[self.state_id]
        if self.prompt_id != expected_prompt.prompt_id or self.prompt_sha256 != expected_prompt.prompt_sha256:
            raise RegistryV8Error("capture prompt identity differs")
        if self.action_sha256 != expected_action.action_sha256:
            raise RegistryV8Error("capture action identity differs")
        if self.state_sha256 != expected_state.state_sha256:
            raise RegistryV8Error("capture state identity differs")
        for name in (
            "prompt_embedding_sha256", "action_embedding_sha256", "state_embedding_sha256",
            "middle_nontext_embedding_sha256", "neutral_embedding_sha256", "carrier_identity_sha256",
            "carrier_state_sha256",
            "timestep_identity_sha256", "rotary_identity_sha256", "projection_sha256",
            "projected_tensor_sha256",
            "forward_event_sha256", "four_block_invocation_sha256",
            "runtime_identity_digest",
        ):
            _hex_digest(getattr(self, name), name)

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.seed_id, self.sigma_name, self.state_id, self.action_id, self.arm, self.block
        )


class FrozenEmbeddingRecomputerV8(Protocol):
    """Required live CPU ABI for recomputing, rather than trusting, embeddings."""

    prompt_encoder_sha256: str
    nontext_encoder_sha256: str

    def prompt_embedding_sha256(self, caption: str) -> str: ...
    def action_embedding_sha256(self, noun_free_dynamics: str) -> str: ...
    def state_embedding_sha256(self, state_body: Mapping[str, Any]) -> str: ...
    def middle_embedding_sha256(self, actual_middle_tensor: Any) -> str: ...


@dataclass(frozen=True)
class B0BindingV8:
    state_id: str
    action_id: str
    seed_id: int
    sigma_name: str
    sigma_cell_index: int
    carrier_state_sha256: str
    observer_absent_output_sha256: str
    matching_primary_output_sha256: str
    observer_module_forward_count: int
    runtime_identity_digest: str

    def __post_init__(self) -> None:
        if self.state_id not in APPEARANCE_IDS or self.action_id not in ACTION_IDS:
            raise RegistryV8Error("B0 action-state key differs")
        if self.seed_id not in SEED_IDS or self.sigma_name not in SIGMA_CELL_INDICES:
            raise RegistryV8Error("B0 seed/sigma key differs")
        if self.sigma_cell_index != SIGMA_CELL_INDICES[self.sigma_name]:
            raise RegistryV8Error("B0 sigma cell differs")
        for name in (
            "carrier_state_sha256", "observer_absent_output_sha256",
            "matching_primary_output_sha256", "runtime_identity_digest",
        ):
            _hex_digest(getattr(self, name), name)
        if self.observer_module_forward_count != 0:
            raise RegistryV8Error("B0 observer module was not absent")
        if self.observer_absent_output_sha256 != self.matching_primary_output_sha256:
            raise RegistryV8Error("B0 output is not bit-exact to matching primary replay")

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.seed_id, self.sigma_name, self.state_id, self.action_id)


def validate_capture_bindings_v8(
    rows: Sequence[CaptureBindingV8],
    runtime: RuntimeIdentityV8,
    embedding_recomputer: FrozenEmbeddingRecomputerV8,
) -> Mapping[str, Any]:
    expected_count = len(SEED_IDS) * len(SIGMA_CELL_INDICES) * len(PROMPTS) * len(BLOCKS)
    if len(rows) != expected_count or expected_count != 1728:
        raise RegistryV8Error("capture binding row count differs")
    if len({row.key for row in rows}) != len(rows):
        raise RegistryV8Error("capture binding key is duplicated")
    expected_keys = {
        (seed, sigma, prompt.state_id, prompt.action_id, prompt.arm, block)
        for seed in SEED_IDS
        for sigma in SIGMA_CELL_INDICES
        for prompt in PROMPTS
        for block in BLOCKS
    }
    if {row.key for row in rows} != expected_keys:
        raise RegistryV8Error("capture binding matrix is incomplete")
    if any(row.runtime_identity_digest != runtime.digest for row in rows):
        raise RegistryV8Error("capture runtime identity differs")
    if any(row.projection_sha256 != runtime.projection_sha256 for row in rows):
        raise RegistryV8Error("capture projection identity differs")
    if embedding_recomputer.prompt_encoder_sha256 != runtime.prompt_encoder_sha256:
        raise RegistryV8Error("live embedding recomputer source identity differs")
    if embedding_recomputer.nontext_encoder_sha256 != runtime.nontext_encoder_sha256:
        raise RegistryV8Error("live nontext recomputer source identity differs")
    for prompt in PROMPTS:
        expected = embedding_recomputer.prompt_embedding_sha256(prompt.caption)
        _hex_digest(expected, "recomputed prompt embedding")
        observed = {
            row.prompt_embedding_sha256
            for row in rows
            if (row.state_id, row.action_id, row.arm)
            == (prompt.state_id, prompt.action_id, prompt.arm)
        }
        if observed != {expected}:
            raise RegistryV8Error("prompt embedding was not live-recomputed")
    for action in ACTIONS:
        expected = embedding_recomputer.action_embedding_sha256(action.dynamics)
        _hex_digest(expected, "recomputed noun-free action embedding")
        observed = {row.action_embedding_sha256 for row in rows if row.action_id == action.action_id}
        if observed != {expected}:
            raise RegistryV8Error("noun-free action embedding was not live-recomputed")
    for state in APPEARANCES:
        expected = embedding_recomputer.state_embedding_sha256(state.body())
        _hex_digest(expected, "recomputed state embedding")
        observed = {row.state_embedding_sha256 for row in rows if row.state_id == state.appearance_id}
        if observed != {expected}:
            raise RegistryV8Error("state embedding was not live-recomputed")

    def consistent(key, value, label: str) -> Mapping[Any, str]:
        groups: dict[Any, set[str]] = {}
        for row in rows:
            groups.setdefault(key(row), set()).add(value(row))
        if any(len(items) != 1 for items in groups.values()):
            raise RegistryV8Error(f"{label} is inconsistent")
        return {group: next(iter(items)) for group, items in groups.items()}

    action_embeddings = consistent(lambda r: r.action_id, lambda r: r.action_embedding_sha256, "action embedding")
    if len(set(action_embeddings.values())) != len(ACTION_IDS):
        raise RegistryV8Error("different actions have duplicate canonical embeddings")
    state_embeddings = consistent(lambda r: r.state_id, lambda r: r.state_embedding_sha256, "state embedding")
    if len(set(state_embeddings.values())) != len(APPEARANCE_IDS):
        raise RegistryV8Error("state embedding is duplicated across appearances")
    consistent(lambda r: r.state_id, lambda r: r.state_sha256, "state SHA")
    if len({row.state_sha256 for row in rows}) != len(APPEARANCE_IDS):
        raise RegistryV8Error("state SHA is not pairwise distinct")

    prompt_embeddings = consistent(
        lambda r: (r.state_id, r.action_id, r.arm),
        lambda r: r.prompt_embedding_sha256,
        "prompt embedding",
    )
    caption_groups: dict[str, set[tuple[str, str, str]]] = {}
    for prompt in PROMPTS:
        caption_groups.setdefault(prompt.prompt_sha256, set()).add((prompt.state_id, prompt.action_id, prompt.arm))
    reverse_prompt: dict[str, set[str]] = {}
    for key, digest in prompt_embeddings.items():
        reverse_prompt.setdefault(digest, set()).add(PROMPT_BY_KEY[key].prompt_sha256)
    if any(len(caption_hashes) != 1 for caption_hashes in reverse_prompt.values()):
        raise RegistryV8Error("different captions have a duplicate prompt embedding")
    neutral_prompt_embeddings = {
        row.prompt_embedding_sha256 for row in rows if row.arm == "neutral"
    }
    neutral_declared_embeddings = {row.neutral_embedding_sha256 for row in rows}
    if len(neutral_prompt_embeddings) != 1 or neutral_declared_embeddings != neutral_prompt_embeddings:
        raise RegistryV8Error("neutral embedding is not identical and explicitly bound")

    # Every carrier owns exactly one nontext/timestep/rotary identity shared by
    # all eight arms and all four blocks at a fixed sigma.  No identity is
    # allowed to cross an action carrier.
    carrier_identity = consistent(
        lambda r: (r.seed_id, r.state_id, r.action_id),
        lambda r: r.carrier_identity_sha256,
        "cross-sigma carrier identity",
    )
    middle_identity = consistent(
        lambda r: (r.seed_id, r.sigma_name, r.state_id, r.action_id),
        lambda r: r.middle_nontext_embedding_sha256,
        "within-cell middle nontext identity",
    )
    carrier_states = consistent(
        lambda r: (r.seed_id, r.sigma_name, r.state_id, r.action_id),
        lambda r: r.carrier_state_sha256,
        "within-cell carrier state",
    )
    if len(set(carrier_states.values())) != 54:
        raise RegistryV8Error("self-generated carrier-state SHA is shared across cells")
    if len(set(carrier_identity.values())) != 18 or len(set(middle_identity.values())) != 54:
        raise RegistryV8Error("action carrier/nontext identity is shared")
    timestep = consistent(lambda r: r.sigma_name, lambda r: r.timestep_identity_sha256, "sigma timestep")
    if len(set(timestep.values())) != len(SIGMA_CELL_INDICES):
        raise RegistryV8Error("sigma timestep identity is duplicated")
    rotary = {row.rotary_identity_sha256 for row in rows}
    if len(rotary) != 1:
        raise RegistryV8Error("37x25 rotary identity is not globally exact")
    forward_events=consistent(
        lambda r:(r.seed_id,r.sigma_name,r.state_id,r.action_id,r.arm),
        lambda r:r.forward_event_sha256,
        "four-block forward event",
    )
    invocations=consistent(
        lambda r:(r.seed_id,r.sigma_name,r.state_id,r.action_id,r.arm),
        lambda r:r.four_block_invocation_sha256,
        "four-block invocation",
    )
    if len(set(forward_events.values()))!=432 or len(set(invocations.values()))!=432:
        raise RegistryV8Error("observer forward event/invocation is shared")
    value = {
        "verified": True,
        "row_count": len(rows),
        "observer_forward_invocations": runtime.observer_forward_invocations,
        "projected_block_capture_rows": runtime.projected_block_capture_rows,
        "state_sha_pairwise_distinct": True,
        "neutral_embedding_identical": True,
        "same_action_embedding_across_appearances": True,
        "same_state_nontext_within_carrier": True,
        "carrier_identity_not_shared_across_actions": True,
        "carrier_state_sha_pairwise_distinct": True,
        "prompt_action_state_embeddings_live_recomputed": True,
        "middle_tensor_embedding_recompute_pending_bound_reducer": True,
        "four_blocks_share_one_unique_forward_event": True,
        "official_hook_execution_pending": True,
    }
    return {**value, "digest": object_sha256(value)}


def validate_b0_bindings_v8(
    rows: Sequence[B0BindingV8],
    captures: Sequence[CaptureBindingV8],
    runtime: RuntimeIdentityV8,
) -> Mapping[str, Any]:
    expected_keys = {
        (seed, sigma, state, action)
        for seed in SEED_IDS
        for sigma in SIGMA_CELL_INDICES
        for state in APPEARANCE_IDS
        for action in ACTION_IDS
    }
    if len(rows) != 54 or {row.key for row in rows} != expected_keys:
        raise RegistryV8Error("B0 exact 54-row matrix differs")
    if len({row.key for row in rows}) != len(rows):
        raise RegistryV8Error("B0 row is duplicated")
    primary_state = {
        (row.seed_id, row.sigma_name, row.state_id, row.action_id): row.carrier_state_sha256
        for row in captures
        if row.arm == "primary"
    }
    if set(primary_state) != expected_keys:
        raise RegistryV8Error("B0 primary capture pairing matrix differs")
    for row in rows:
        if row.runtime_identity_digest != runtime.digest:
            raise RegistryV8Error("B0 runtime identity differs")
        if primary_state[row.key] != row.carrier_state_sha256:
            raise RegistryV8Error("B0 is not paired to the matching primary carrier state")
    value = {
        "verified": True,
        "row_count": 54,
        "observer_module_forward_count": 0,
        "schema_rows_claim_bit_exact_output_digests": True,
        "all_primary_carrier_states_bound": True,
        "execution_proven": False,
        "runner_and_external_completion_receipt_pending": True,
    }
    return {**value, "digest": object_sha256(value)}


def _plain_prereg_path() -> Path:
    original = PREREG_PATH.absolute()
    if not original.is_file() or original.is_symlink() or original.resolve(strict=True) != original:
        raise RegistryV8Error("V8 preregistration is not one canonical plain file")
    return original


def load_preregistration() -> Mapping[str, Any]:
    path = _plain_prereg_path()
    def reject_pairs(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
        output: dict[str, Any] = {}
        for key, item in pairs:
            if key in output:
                raise RegistryV8Error("V8 preregistration has a duplicate key")
            output[key] = item
        return output
    def reject_constant(value: str) -> Any:
        raise RegistryV8Error(f"V8 preregistration has nonfinite constant {value}")
    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RegistryV8Error("V8 preregistration cannot be parsed") from error
    if not isinstance(value, dict):
        raise RegistryV8Error("V8 preregistration is not one object")
    claim = value.get("prereg_self_sha256")
    body = dict(value); body.pop("prereg_self_sha256", None)
    if claim != object_sha256(body):
        raise RegistryV8Error("V8 preregistration self hash differs")
    if value.get("method") != METHOD or value.get("frozen_cpu_only") is not True:
        raise RegistryV8Error("V8 preregistration identity differs")
    cross = value.get("cross_fit", {})
    for branch in ("A_to_B", "B_to_A"):
        proposal = tuple(cross.get("proposal_phases_by_branch", {}).get(branch, ()))
        evaluation = tuple(cross.get("evaluation_phases_by_branch", {}).get(branch, ()))
        shuffle = tuple(cross.get("proposal_phase_shuffle_by_branch", {}).get(branch, ()))
        if len(proposal) != 10 or set(shuffle) != set(proposal) or len(set(shuffle)) != 10 or set(proposal) & set(evaluation):
            raise RegistryV8Error("branch proposal-only phase shuffle closure differs")
    frozen = value.get("frozen_base", {})
    exact = (
        frozen.get("expected_observer_forward_invocations"),
        frozen.get("expected_projected_block_capture_rows"),
        frozen.get("expected_B0_observer_absent_forward_invocations"),
        frozen.get("expected_trajectory_forward_invocations"),
        frozen.get("expected_total_frozen_forward_invocations"),
    )
    if exact != (432, 1728, 54, 1440, 1926):
        raise RegistryV8Error("V8 frozen forward count preregistration differs")
    claims = value.get("claims", {})
    if any(
        (
            claims.get("representation_admission_hard_false") is not True,
            claims.get("scientific_claim_authorized") is not False,
            claims.get("training_authorized") is not False,
            claims.get("renderer_or_decoder_authorized") is not False,
            claims.get("route_or_injection_authorized") is not False,
            claims.get("gpu_launch_authorized") is not False,
        )
    ):
        raise RegistryV8Error("V8 hard-false claims differ")
    return value


def registry_receipt_v8() -> Mapping[str, Any]:
    prereg = load_preregistration()
    value = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "appearance_count": len(APPEARANCES),
        "action_program_count": len(ACTIONS),
        "arm_count_per_carrier": len(ARMS),
        "prompt_identity_count_per_seed_sigma": len(PROMPTS),
        "state_sha256": {row.appearance_id: row.state_sha256 for row in APPEARANCES},
        "action_sha256": {row.action_id: row.action_sha256 for row in ACTIONS},
        "prompt_sha256": {row.prompt_id: row.prompt_sha256 for row in PROMPTS},
        "prereg_self_sha256": prereg["prereg_self_sha256"],
        "cpu_only": True,
        "gpu_launch_authorized": False,
        "representation_admission_hard_false": True,
    }
    return {**value, "digest": object_sha256(value)}


__all__ = [
    "ACTION_BY_ID", "ACTION_IDS", "ACTIONS", "APPEARANCE_BY_ID", "APPEARANCE_IDS",
    "APPEARANCES", "ARMS", "B0BindingV8", "BLOCKS", "CaptureBindingV8",
    "FrozenEmbeddingRecomputerV8", "METHOD", "NEUTRAL_CAPTION",
    "PATCH_HEIGHT", "PATCH_WIDTH", "PHASES", "PROMPTS", "PROMPT_BY_KEY", "PREREG_PATH",
    "RegistryV8Error", "RuntimeIdentityV8", "SEED_IDS", "SIGMA_CELL_INDICES",
    "canonical_json_bytes", "load_preregistration", "object_sha256", "registry_receipt_v8",
    "text_sha256", "validate_b0_bindings_v8", "validate_capture_bindings_v8",
]
