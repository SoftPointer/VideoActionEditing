"""Validate and inspect the instruction-only video-editor experiment ladder.

This registry is deliberately a scheduling boundary, not a model wrapper.
The representation search may inspect availability, but no renderer probe is
eligible until a separately committed representation decision passes.  No
entry may require a user mask, pose, trajectory, keyframe, or second prompt.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "motive-instruction-video-editor-registry-v1"
ALLOWED_INTERFACE = "source_video_plus_instruction"
REQUIRED_USER_INPUTS = ("source_video", "instruction")
FORBIDDEN_USER_INPUTS = (
    "mask",
    "bbox",
    "pose",
    "trajectory",
    "reference_image",
    "edited_keyframe",
    "target_video",
    "source_prompt",
    "target_prompt",
)


class InstructionModelRegistryError(ValueError):
    """The model inventory violates the instruction-only scope."""


def _object(value: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InstructionModelRegistryError(f"{where} is not an object")
    return value


def load_registry(path: Path) -> dict[str, Any]:
    """Load and fail closed on any scope or registry inconsistency."""

    unresolved = path.expanduser()
    if unresolved.is_symlink() or not unresolved.is_file():
        raise FileNotFoundError(unresolved)
    payload = json.loads(unresolved.read_text(encoding="utf-8"))
    root = _object(payload, where="registry")
    if root.get("schema_version") != SCHEMA:
        raise InstructionModelRegistryError("registry schema differs")

    scope = _object(root.get("scope"), where="scope")
    if tuple(scope.get("required_user_inputs", ())) != REQUIRED_USER_INPUTS:
        raise InstructionModelRegistryError(
            "required user inputs are not source_video + instruction"
        )
    if tuple(scope.get("forbidden_user_inputs", ())) != FORBIDDEN_USER_INPUTS:
        raise InstructionModelRegistryError(
            "forbidden user-input contract differs"
        )
    if scope.get("internal_automatic_analysis_allowed") is not True:
        raise InstructionModelRegistryError(
            "internal automatic analysis must remain allowed"
        )
    if scope.get("internal_analysis_is_never_a_required_user_input") is not True:
        raise InstructionModelRegistryError(
            "internal analysis may not become a required user input"
        )

    promotion = _object(
        root.get("promotion_policy"),
        where="promotion_policy",
    )
    required_true = (
        "representation_gate_required_before_renderer_probe",
        "renderer_probe_required_before_training",
        "cross_content_transfer_required",
        "same_representation_payload_required_across_contents",
    )
    if any(promotion.get(name) is not True for name in required_true):
        raise InstructionModelRegistryError(
            "promotion policy weakens an ordering or transfer gate"
        )
    if int(promotion.get("minimum_distinct_renderers_for_training", 0)) < 2:
        raise InstructionModelRegistryError(
            "training requires at least two distinct renderers"
        )

    raw_models = root.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise InstructionModelRegistryError("models must be a nonempty list")
    ids: set[str] = set()
    primary_families: set[str] = set()
    for index, raw in enumerate(raw_models):
        model = _object(raw, where=f"models[{index}]")
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id or model_id in ids:
            raise InstructionModelRegistryError(
                f"models[{index}] has an invalid/duplicate id"
            )
        ids.add(model_id)
        if model.get("interface") != ALLOWED_INTERFACE:
            raise InstructionModelRegistryError(
                f"{model_id} is not source-video + instruction"
            )
        primary = model.get("primary_eligible")
        if not isinstance(primary, bool):
            raise InstructionModelRegistryError(
                f"{model_id} primary_eligible is not boolean"
            )
        family = model.get("architecture_family")
        if not isinstance(family, str) or not family:
            raise InstructionModelRegistryError(
                f"{model_id} architecture family is invalid"
            )
        if primary:
            primary_families.add(family)
        probe = _object(
            model.get("availability_probe"),
            where=f"{model_id}.availability_probe",
        )
        kind = probe.get("kind")
        if kind == "all_paths":
            values = probe.get("paths")
        elif kind == "any_glob":
            values = probe.get("patterns")
        else:
            raise InstructionModelRegistryError(
                f"{model_id} availability probe kind differs"
            )
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(item, str) and item for item in values)
        ):
            raise InstructionModelRegistryError(
                f"{model_id} availability probe values are invalid"
            )
    if len(primary_families) < 2:
        raise InstructionModelRegistryError(
            "primary ladder lacks architecture diversity"
        )

    excluded = root.get("explicitly_excluded")
    if not isinstance(excluded, list):
        raise InstructionModelRegistryError(
            "explicitly_excluded must be a list"
        )
    excluded_ids = {
        str(_object(item, where="explicitly_excluded item").get("id"))
        for item in excluded
    }
    if "vace" not in excluded_ids:
        raise InstructionModelRegistryError("VACE must be explicitly excluded")
    return dict(root)


def availability_report(
    registry: Mapping[str, Any],
    *,
    workspace: Path,
    representation_gate_passed: bool,
) -> dict[str, Any]:
    """Resolve local artifacts without downloading or mutating anything."""

    root = workspace.expanduser().resolve(strict=True)
    rows: list[dict[str, Any]] = []
    for raw in registry["models"]:
        model = dict(raw)
        probe = model["availability_probe"]
        matches: list[str] = []
        if probe["kind"] == "all_paths":
            candidates = [root / value for value in probe["paths"]]
            available = all(path.exists() for path in candidates)
            matches = [str(path) for path in candidates if path.exists()]
        else:
            for pattern in probe["patterns"]:
                matches.extend(
                    sorted(
                        glob.glob(str(root / pattern)),
                    )
                )
            matches = sorted(set(matches))
            available = bool(matches)
        primary = bool(model["primary_eligible"])
        if not representation_gate_passed:
            scheduling_status = "blocked_by_representation_gate"
        elif not primary:
            scheduling_status = "control_only"
        elif available:
            scheduling_status = "ready_for_renderer_probe"
        else:
            scheduling_status = "requires_installation_or_weights"
        rows.append(
            {
                "id": model["id"],
                "display_name": model["display_name"],
                "architecture_family": model["architecture_family"],
                "role": model["role"],
                "primary_eligible": primary,
                "available": available,
                "matches": matches,
                "scheduling_status": scheduling_status,
            }
        )
    return {
        "schema_version": "motive-instruction-model-availability-v1",
        "representation_gate_passed": bool(representation_gate_passed),
        "mutations_performed": False,
        "downloads_performed": False,
        "models": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument(
        "--representation-gate-passed",
        action="store_true",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registry = load_registry(args.registry)
    report = availability_report(
        registry,
        workspace=args.workspace,
        representation_gate_passed=bool(
            args.representation_gate_passed
        ),
    )
    text = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
