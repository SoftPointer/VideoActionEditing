"""Deterministically compile a full-motion target plan into one instruction.

There is intentionally no free-form writer field in this module.  The final
instruction consists only of canonical clauses derived from the validated
stable references and structured target-motion fields, followed by salient
static-person and camera clauses.  Model-authored ``target_clause`` prose is a
non-executable cross-check, not an instruction source.  Character and UTF-8
byte spans make exact clause-level post-generation auditing possible.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

from .goku_full_motion_contract import (
    GokuFullMotionContractError,
    canonical_json_bytes,
    object_sha256,
    sha256_text,
    validate_source_census,
    validate_target_plan,
)


INSTRUCTION_RENDER_SCHEMA = "motive-goku-full-motion-instruction-render-v2"
INSTRUCTION_CLAUSE_SCHEMA = "motive-goku-full-motion-instruction-clause-v2"
INSTRUCTION_COMPILER_CONTRACT_SCHEMA = (
    "motive-goku-full-motion-instruction-compiler-contract-v2"
)
INSTRUCTION_POLICY = (
    "canonical-structured-motion-i0-registry-static-camera-clauses-v5"
)

PREFIX = "Starting from the exact first frame: "
SEPARATOR = "; "
SUFFIX = "."
TERMINAL_PUNCTUATION = ".!?;:"
_ENTITY_MARKER_RE = re.compile(r"\[\[(entity_\d{2})\]\]")


class GokuFullMotionInstructionError(ValueError):
    """A compiled instruction or its exact span/hash binding is invalid."""


def _implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).resolve(strict=True).read_bytes()).hexdigest()


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    if set(value) != expected:
        raise GokuFullMotionInstructionError(
            f"{context} keys differ from closed schema: "
            f"{sorted(set(value) ^ expected)}"
        )


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GokuFullMotionInstructionError(f"{context} must be an object")
    return value


def _compiler_contract() -> dict[str, Any]:
    contract: dict[str, Any] = {
        "schema_version": INSTRUCTION_COMPILER_CONTRACT_SCHEMA,
        "policy": INSTRUCTION_POLICY,
        "prefix": PREFIX,
        "separator": SEPARATOR,
        "suffix": SUFFIX,
        "clause_order": ["dynamic_units", "static_people", "camera"],
        "free_form_text_allowed": False,
        "clause_text_transformed": True,
        "clause_source": (
            "structured_stable_reference_registry_bound_dynamic_motion_"
            "static_state_camera_motion"
        ),
        "entity_marker_rendering": "resolve_exact_i0_registry_stable_reference",
        "model_target_clause_executable": False,
        "span_semantics": "zero_based_half_open_character_and_utf8_byte",
        "implementation_sha256": _implementation_sha256(),
    }
    contract["contract_sha256"] = object_sha256(contract)
    return contract


def _motion_fragment_for_render(value: Any, *, context: str) -> str:
    """Remove only sentence-ending punctuation from a validated fragment.

    The raw target plan remains byte-for-byte untouched and continues to bind
    ``target_plan_sha256``.  Rendering owns punctuation at clause boundaries,
    so model-authored sentence endings must not produce ``.;`` or duplicate
    final punctuation in the deterministic instruction.
    """

    raw = str(value)
    rendered = raw.rstrip(TERMINAL_PUNCTUATION).rstrip()
    if not rendered:
        raise GokuFullMotionInstructionError(
            f"{context} has no text before terminal punctuation"
        )
    return rendered


def _resolve_entity_markers(
    value: Any,
    *,
    registry_by_id: Mapping[str, Mapping[str, Any]],
    context: str,
) -> str:
    """Resolve validated ``[[entity_NN]]`` tokens from the I0 registry only."""

    rendered = _motion_fragment_for_render(value, context=context)

    def replace(match: re.Match[str]) -> str:
        entity_id = match.group(1)
        entity = registry_by_id.get(entity_id)
        if entity is None:
            raise GokuFullMotionInstructionError(
                f"{context} references unknown I0 entity {entity_id!r}"
            )
        return str(entity["stable_reference"])

    resolved = _ENTITY_MARKER_RE.sub(replace, rendered)
    if "[[" in resolved or "]]" in resolved:
        raise GokuFullMotionInstructionError(
            f"{context} contains an unresolved I0 entity marker"
        )
    return resolved


def _registry_reference(
    *,
    registry_by_id: Mapping[str, Mapping[str, Any]],
    entity_id: Any,
    context: str,
) -> str:
    """Return an actor reference only by resolving its exact registry ID."""

    normalized_id = str(entity_id)
    entity = registry_by_id.get(normalized_id)
    if entity is None:
        raise GokuFullMotionInstructionError(
            f"{context} references unknown I0 entity {normalized_id!r}"
        )
    reference = str(entity["stable_reference"])
    if not reference:
        raise GokuFullMotionInstructionError(
            f"{context} registry stable_reference is empty"
        )
    return reference


def _clause_sources(
    source: Mapping[str, Any], plan: Mapping[str, Any]
) -> list[tuple[str, str, str, str]]:
    clauses: list[tuple[str, str, str, str]] = []
    registry_by_id = {
        str(item["entity_id"]): item for item in source["i0_entity_registry"]
    }
    for source_unit, target in zip(
        source["dynamic_units"], plan["dynamic_unit_targets"], strict=True
    ):
        unit_id = str(source_unit["unit_id"])
        reference = _registry_reference(
            registry_by_id=registry_by_id,
            entity_id=source_unit["entity_id"],
            context=f"source_census.dynamic_units[{unit_id}]",
        )
        shared_raw = target["explicit_shared_base_motion"]
        shared = (
            None
            if shared_raw is None
            else _resolve_entity_markers(
                shared_raw,
                registry_by_id=registry_by_id,
                context=(
                    f"target_plan.dynamic_unit_targets[{unit_id}]."
                    "explicit_shared_base_motion"
                ),
            )
        )
        novel = _resolve_entity_markers(
            target["novel_target_motion"],
            registry_by_id=registry_by_id,
            context=(
                f"target_plan.dynamic_unit_targets[{unit_id}]."
                "novel_target_motion"
            ),
        )
        if shared is None:
            # ``novel_target_motion`` is deliberately complete standalone
            # prose, not a verb phrase.  A labelled colon therefore remains
            # grammatical whether the structured value starts with a verb,
            # an actor noun phrase, or a temporal anchor such as ``from I0``.
            text = (
                f"Have {reference} perform this complete target motion: "
                f"{novel}"
            )
        else:
            # Spell out the only permitted shared base and the novel action
            # as separate, actor-bound components.  Nothing here asks the
            # generator to infer or preserve an unobserved future trajectory.
            text = (
                f"Have {reference} perform this explicitly specified base "
                f"motion: {shared}. Concurrently, have {reference} perform "
                f"this complete novel action: {novel}"
            )
        clauses.append(
            (
                f"dynamic:{unit_id}",
                "dynamic_unit",
                unit_id,
                text,
            )
        )
    for source_person, target in zip(
        source["static_salient_people"],
        plan["static_person_targets"],
        strict=True,
    ):
        unit_id = str(source_person["unit_id"])
        reference = _registry_reference(
            registry_by_id=registry_by_id,
            entity_id=source_person["entity_id"],
            context=f"source_census.static_salient_people[{unit_id}]",
        )
        clauses.append(
            (
                f"static:{unit_id}",
                "static_person",
                unit_id,
                f"Have {reference} remain still",
            )
        )
    camera = plan["camera_target"]
    if camera["target_motion_class"] == "locked_off":
        camera_text = "Keep the camera locked off"
    else:
        description = _motion_fragment_for_render(
            camera["target_motion_description"],
            context="target_plan.camera_target.target_motion_description",
        )
        camera_text = (
            "Set the camera trajectory to "
            f"{description}"
        )
    clauses.append(
        (
            "camera:camera",
            "camera",
            "camera",
            camera_text,
        )
    )
    return clauses


def _compile_validated(
    source: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    sources = _clause_sources(source, plan)
    clause_texts = [item[3] for item in sources]
    instruction = PREFIX + SEPARATOR.join(clause_texts) + SUFFIX

    records: list[dict[str, Any]] = []
    char_cursor = len(PREFIX)
    byte_cursor = len(PREFIX.encode("utf-8"))
    separator_bytes = len(SEPARATOR.encode("utf-8"))
    for order, (clause_id, kind, subject_id, text) in enumerate(sources):
        if order:
            char_cursor += len(SEPARATOR)
            byte_cursor += separator_bytes
        encoded = text.encode("utf-8")
        record = {
            "schema_version": INSTRUCTION_CLAUSE_SCHEMA,
            "order": order,
            "clause_id": clause_id,
            "clause_kind": kind,
            "subject_id": subject_id,
            "text": text,
            "text_sha256": sha256_text(text),
            "start_char": char_cursor,
            "end_char": char_cursor + len(text),
            "start_byte": byte_cursor,
            "end_byte": byte_cursor + len(encoded),
        }
        records.append(record)
        char_cursor = int(record["end_char"])
        byte_cursor = int(record["end_byte"])

    entity_clauses = {
        record["subject_id"]: record["text"]
        for record in records
        if record["clause_kind"] in {"dynamic_unit", "static_person"}
    }
    camera_clause = records[-1]["text"]
    compiler = _compiler_contract()
    result: dict[str, Any] = {
        "schema_version": INSTRUCTION_RENDER_SCHEMA,
        "policy": INSTRUCTION_POLICY,
        "iid": source["iid"],
        "source_census_sha256": object_sha256(source),
        "target_plan_sha256": object_sha256(plan),
        "compiler_contract": compiler,
        "compiler_contract_sha256": object_sha256(compiler),
        "ordered_clause_ids": [record["clause_id"] for record in records],
        "clauses": records,
        "entity_clauses": entity_clauses,
        "camera_clause": camera_clause,
        "edit_instruction": instruction,
        "instruction_sha256": sha256_text(instruction),
    }
    # Force a finite, serializable result now; downstream callers may bind the
    # returned object without importing any compiler implementation details.
    canonical_json_bytes(result)
    return result


def compile_full_motion_instruction(
    source_census: Mapping[str, Any], target_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate both inputs and return their sole permitted instruction."""

    try:
        source = validate_source_census(source_census)
        plan = validate_target_plan(target_plan, source_census=source)
    except GokuFullMotionContractError:
        raise
    return _compile_validated(source, plan)


def render_edit_instruction(
    source_census: Mapping[str, Any], target_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Backward-readable alias for :func:`compile_full_motion_instruction`."""

    return compile_full_motion_instruction(source_census, target_plan)


def validate_compiled_instruction(
    value: Any,
    *,
    source_census: Mapping[str, Any],
    target_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompile and require exact object equality, including every span/hash."""

    compiled = _mapping(value, context="compiled_instruction")
    expected = compile_full_motion_instruction(source_census, target_plan)
    _exact_keys(compiled, set(expected), context="compiled_instruction")
    if dict(compiled) != expected:
        raise GokuFullMotionInstructionError(
            "compiled instruction differs from deterministic recompilation"
        )

    instruction = str(compiled["edit_instruction"])
    encoded = instruction.encode("utf-8")
    clauses = compiled["clauses"]
    if not isinstance(clauses, list):
        raise GokuFullMotionInstructionError(
            "compiled_instruction.clauses must be an array"
        )
    for index, raw_record in enumerate(clauses):
        record = _mapping(raw_record, context=f"compiled_instruction.clauses[{index}]")
        _exact_keys(
            record,
            {
                "schema_version",
                "order",
                "clause_id",
                "clause_kind",
                "subject_id",
                "text",
                "text_sha256",
                "start_char",
                "end_char",
                "start_byte",
                "end_byte",
            },
            context=f"compiled_instruction.clauses[{index}]",
        )
        text = str(record["text"])
        if (
            instruction[int(record["start_char"]) : int(record["end_char"])]
            != text
            or encoded[int(record["start_byte"]) : int(record["end_byte"])]
            != text.encode("utf-8")
            or record["text_sha256"] != sha256_text(text)
        ):
            raise GokuFullMotionInstructionError(
                f"compiled instruction clause {index} span/hash differs"
            )
    if compiled["instruction_sha256"] != sha256_text(instruction):
        raise GokuFullMotionInstructionError(
            "compiled instruction final SHA-256 differs"
        )
    return copy.deepcopy(dict(compiled))


__all__ = [
    "GokuFullMotionInstructionError",
    "INSTRUCTION_CLAUSE_SCHEMA",
    "INSTRUCTION_COMPILER_CONTRACT_SCHEMA",
    "INSTRUCTION_POLICY",
    "INSTRUCTION_RENDER_SCHEMA",
    "PREFIX",
    "SEPARATOR",
    "SUFFIX",
    "compile_full_motion_instruction",
    "render_edit_instruction",
    "validate_compiled_instruction",
]
