#!/usr/bin/env python3
"""Validate the draft E00 three-vessel clean matched-route specification.

This module is deliberately data/validation-only.  It neither imports Torch
nor calls a renderer, controller, scheduler, optimizer, or cluster launcher.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-e00-three-vessel-clean-matched-route-probe-spec-v1"
METHOD_ROOT = Path(__file__).resolve().parent
DEFAULT_SPEC = METHOD_ROOT / "assets/e00_three_vessel_clean_matched_route_probe_v1.json"
SOURCE_SHA256 = "888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de"
ANCHOR_SHA256 = "e485b2963d5ba191d8fc98158b7ab100bd1bd7584735221077c014ff0c3a19aa"
CHECKPOINT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
PROMPT_FIELDS = (
    "source_caption",
    "target_caption",
    "source_noop_caption",
    "editing_instruction",
)
ALLOWED_ARM_DIFFERENCES = (
    "arm_id",
    "route.enabled",
    "route.replay_to_target",
    "route.strength",
)


class ThreeVesselSpecError(ValueError):
    pass


def _expect(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ThreeVesselSpecError(
            f"{label}: expected {expected!r}, got {actual!r}"
        )


def _closed(value: Any, expected: Sequence[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ThreeVesselSpecError(
            f"{label} keys differ: expected={sorted(expected)!r}, actual={actual!r}"
        )
    return value


def text_sha256(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ThreeVesselSpecError("hash-bound prompt must be non-empty UTF-8 text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def keyed_noise_seed(master_seed: int, step: int, candidate: int = 0) -> int:
    """Mirror the registered arm-independent keyed-noise seed derivation."""

    if any(type(value) is not int or value < 0 for value in (master_seed, step, candidate)):
        raise ThreeVesselSpecError("keyed-noise coordinates must be non-negative integers")
    payload = (
        f"bernini-guided-sac-v2\0{master_seed}\0{step}\0{candidate}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def keyed_noise_bank_digest(master_seed: int, steps: int = 40) -> str:
    if type(steps) is not int or steps <= 0:
        raise ThreeVesselSpecError("keyed-noise step count must be positive")
    rows = [
        {"step": step, "candidate": 0, "derived_seed": keyed_noise_seed(master_seed, step)}
        for step in range(steps)
    ]
    return canonical_sha256(rows)


def load_spec(path: Path | str = DEFAULT_SPEC) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ThreeVesselSpecError(f"missing plain spec file: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ThreeVesselSpecError(f"invalid spec JSON: {source}: {error}") from error
    if not isinstance(value, dict):
        raise ThreeVesselSpecError("spec root must be an object")
    validate_spec(value)
    return value


def materialize_arm(spec: Mapping[str, Any], arm_id: str) -> dict[str, Any]:
    common = copy.deepcopy(spec["common_arm_contract"])
    rows = [row for row in spec["arms"] if row.get("arm_id") == arm_id]
    if len(rows) != 1:
        raise ThreeVesselSpecError(f"arm lookup differs for {arm_id!r}")
    common["arm_id"] = arm_id
    common["route"] = copy.deepcopy(rows[0]["route"])
    return common


def _leaf_paths(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_leaf_paths(value[key], child))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            result.update(_leaf_paths(item, child))
        return result
    return {prefix: value}


def arm_difference_paths(spec: Mapping[str, Any]) -> tuple[str, ...]:
    route_off = _leaf_paths(materialize_arm(spec, "clean_route_off"))
    route_on = _leaf_paths(materialize_arm(spec, "clean_route_on"))
    if set(route_off) != set(route_on):
        raise ThreeVesselSpecError("materialized arm field closure differs")
    return tuple(key for key in sorted(route_off) if route_off[key] != route_on[key])


def validate_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    _closed(
        spec,
        (
            "schema_version",
            "status",
            "event",
            "prompt_contract",
            "frozen_model_contract",
            "common_arm_contract",
            "arms",
            "matched_pair_contract",
            "human_admission_gates",
        ),
        "spec",
    )
    _expect(spec["schema_version"], SCHEMA_VERSION, "schema_version")

    status = spec["status"]
    for field, expected in {
        "draft_only": True,
        "execution_authorized": False,
        "gpu_run_started": False,
        "training_performed": False,
    }.items():
        _expect(status.get(field), expected, f"status.{field}")

    event = spec["event"]
    _expect(event.get("event_ordinal"), 0, "event ordinal")
    _expect(event.get("source_video_sha256"), SOURCE_SHA256, "source SHA-256")
    _expect((event.get("frames"), event.get("fps")), (81, 25), "source geometry")
    objects = event.get("source_objects")
    if not isinstance(objects, list) or len(objects) != 3:
        raise ThreeVesselSpecError("E00 must bind exactly three source vessels")
    _expect([row.get("object_id") for row in objects], [1, 2, 3], "object IDs")
    _expect(
        [row.get("material_color") for row in objects],
        ["white ceramic", "transparent glass", "white ceramic"],
        "object material/color order",
    )
    _expect(
        [row.get("target_role") for row in objects],
        [
            "protected inactive vessel after its original pour is stopped",
            "desired pouring actor",
            "desired recipient of object #2",
        ],
        "target object roles",
    )
    _expect(
        event.get("frame0_relation"),
        "#1 -> #2 is the ongoing source pour trajectory; #3 is idle",
        "frame-0 relation",
    )
    _expect(
        event.get("desired_relation"),
        "after frame 0: stop #1 -> #2, then execute #2 -> #3",
        "desired relation",
    )

    prompts = spec["prompt_contract"]
    for field in PROMPT_FIELDS:
        expected_hash = text_sha256(prompts.get(field))
        _expect(
            prompts.get(f"{field}_utf8_sha256"),
            expected_hash,
            f"prompt hash {field}",
        )
    instruction = prompts["editing_instruction"]
    ordered_fragments = (
        "Frame 0 must exactly retain the source state",
        "stop any ongoing pour from #1 into #2",
        "lift that same original transparent handled glass vessel #2",
        "continuous amber stream from #2 into #3",
        "return #2 upright",
        "all three original vessels",
        "do not turn #2 into a white or opaque vessel",
        "do not copy any person, vessel, clothing, table, or background appearance from the action anchor",
    )
    positions = [instruction.find(fragment) for fragment in ordered_fragments]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise ThreeVesselSpecError("editing instruction loses ordered three-vessel transition")

    frozen = spec["frozen_model_contract"]
    _expect(
        frozen.get("checkpoint_manifest_sha256"),
        CHECKPOINT_MANIFEST_SHA256,
        "checkpoint manifest SHA-256",
    )
    for field, expected in {
        "base_frozen": True,
        "optimization_steps": 0,
        "trained_checkpoint_loaded": False,
        "adapter_present": False,
        "trainable_parameter_elements": 0,
    }.items():
        _expect(frozen.get(field), expected, f"frozen_model_contract.{field}")

    common = spec["common_arm_contract"]
    _expect(common.get("num_inference_steps"), 40, "inference steps")
    _expect(common.get("flow_shift"), 5.0, "flow shift")
    _expect(
        common.get("candidate_count_by_step"), [1] * 40, "single-candidate schedule"
    )
    _expect(
        common.get("frame0_source_latent_clamp_after_every_update"),
        True,
        "frame-0 clamp",
    )
    noise = common.get("noise")
    for field, expected in {
        "mode": "keyed_only",
        "scheme": "sha256_keyed_cpu_torch_generator_v1",
        "master_seed": 2027,
        "candidate_index": 0,
        "fresh_noise_per_solver_step": True,
        "same_exact_noise_tensor_across_arms": True,
        "anchor_generation_gaussian_path_read": False,
        "anchor_generation_gaussian_used": False,
        "previous_step_noise_correlation": False,
        "runtime_raw_noise_hash_required_for_every_step": True,
    }.items():
        _expect(noise.get(field), expected, f"noise.{field}")
    _expect(
        noise.get("keyed_bank_digest"),
        keyed_noise_bank_digest(2027, 40),
        "keyed noise bank digest",
    )
    _expect(common.get("sga"), {"enabled": False, "aggregation": "none", "temperature": None}, "SGA disabled")
    _expect(common.get("anc"), {"enabled": False, "retained_variance": 0.0}, "ANC disabled")
    _expect(
        common.get("preservation"),
        {
            "mode": "frame0_hard_clamp_only",
            "soft_reward": False,
            "spatial_mask": False,
            "object_identity_residual": False,
        },
        "shared preservation",
    )
    observer = common.get("anchor_observer")
    _expect(observer.get("video_sha256"), ANCHOR_SHA256, "anchor observer SHA-256")
    for field, expected in {
        "execute_identical_capture_in_both_arms": True,
        "same_caption_for_dynamic_and_static": True,
        "same_model_timestep_for_dynamic_and_static": True,
        "same_exact_keyed_noise_for_dynamic_and_static": True,
        "donor_value_hidden_state_attention_output_rgb_or_latent_used": False,
        "anchor_pixels_or_clean_latent_copied_to_target": False,
        "anchor_generation_gaussian_read": False,
    }.items():
        _expect(observer.get(field), expected, f"anchor_observer.{field}")
    _expect(observer.get("cached_fields"), ["query", "key"], "Q/K-only capture")

    arms = spec["arms"]
    if not isinstance(arms, list) or len(arms) != 2:
        raise ThreeVesselSpecError("matched probe requires exactly two arms")
    for row in arms:
        _closed(row, ("arm_id", "route"), "arm")
        _closed(row["route"], ("enabled", "strength", "replay_to_target"), "route")
    _expect([row["arm_id"] for row in arms], ["clean_route_off", "clean_route_on"], "arm order")
    _expect(
        arms[0]["route"],
        {"enabled": False, "strength": 0.0, "replay_to_target": False},
        "route-off",
    )
    _expect(
        arms[1]["route"],
        {"enabled": True, "strength": 1.0, "replay_to_target": True},
        "route-on",
    )
    matched = spec["matched_pair_contract"]
    _expect(
        matched.get("only_allowed_arm_differences"),
        list(ALLOWED_ARM_DIFFERENCES),
        "allowed arm differences",
    )
    _expect(arm_difference_paths(spec), ALLOWED_ARM_DIFFERENCES, "actual arm differences")
    forbidden = matched.get("forbidden_shared_inputs")
    if not isinstance(forbidden, list) or not any("anchor generation Gaussian" in row for row in forbidden):
        raise ThreeVesselSpecError("matched pair must forbid the anchor generation Gaussian")
    return {
        "schema_version": SCHEMA_VERSION,
        "spec_sha256": canonical_sha256(spec),
        "source_sha256": SOURCE_SHA256,
        "instruction_sha256": prompts["editing_instruction_utf8_sha256"],
        "arm_difference_paths": list(ALLOWED_ARM_DIFFERENCES),
    }


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    receipt = validate_spec(load_spec())
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
