#!/usr/bin/env python3
"""Diagnose cross-anchor consensus in detached action-teacher caches.

The tool is intentionally read-only.  It compares several temporal views of
the cached ``[1, 21, 32]`` teacher code and reports whether a direction from
one actor/scene agrees with a consensus built only from another actor/scene
performing the same action.  It never evaluates a training checkpoint and it
does not turn an in-sample centroid into a claimed target.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "bernini-action-teacher-consensus-diagnostic-v1"


class ConsensusDiagnosticError(RuntimeError):
    pass


def _unit(vector: Any) -> Any:
    import torch

    flat = vector.float().reshape(-1)
    norm = torch.linalg.vector_norm(flat)
    if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 1.0e-12:
        raise ConsensusDiagnosticError("teacher representation has zero/non-finite norm")
    return flat / norm


def _representations() -> Mapping[str, Callable[[Any], Any]]:
    return {
        "raw": lambda value: value,
        "temporal_centered": lambda value: value - value.mean(dim=1, keepdim=True),
        "temporal_delta": lambda value: value[:, 1:, :] - value[:, :-1, :],
        "temporal_acceleration": lambda value: (
            value[:, 2:, :] - 2.0 * value[:, 1:-1, :] + value[:, :-2, :]
        ),
        "endpoint_delta": lambda value: (
            value[:, -3:, :].mean(dim=1) - value[:, :3, :].mean(dim=1)
        ),
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ConsensusDiagnosticError("cannot summarize an empty metric")
    return float(sum(values) / len(values))


def _cosine(left: Any, right: Any) -> float:
    value = float((left * right).sum().item())
    if not math.isfinite(value):
        raise ConsensusDiagnosticError("non-finite cosine")
    return value


def _centroid(vectors: Sequence[Any]) -> Any:
    import torch

    return _unit(torch.stack(list(vectors), dim=0).mean(dim=0))


def _validate_cells(cache: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    import torch

    cells = cache.get("cells")
    if not isinstance(cells, list) or len(cells) != 16 or cache.get("slots") != 4:
        raise ConsensusDiagnosticError("expected the sealed 4-row x 4-slot cache")
    keys: set[tuple[int, int]] = set()
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ConsensusDiagnosticError("cache cell must be a mapping")
        key = (cell.get("row_index"), cell.get("slot"))
        direction = cell.get("teacher_unit")
        if (
            key[0] not in range(4)
            or key[1] not in range(4)
            or key in keys
            or not isinstance(direction, torch.Tensor)
            or tuple(int(item) for item in direction.shape) != (1, 21, 32)
        ):
            raise ConsensusDiagnosticError("teacher cache grid or tensor shape differs")
        keys.add(key)
    if keys != {(row, slot) for row in range(4) for slot in range(4)}:
        raise ConsensusDiagnosticError("teacher cache grid is incomplete")
    return sorted(cells, key=lambda item: (item["row_index"], item["slot"]))


def analyze(cache: Mapping[str, Any]) -> Mapping[str, Any]:
    cells = _validate_cells(cache)
    action_pairs = ((0, 1), (2, 3))
    results: dict[str, Any] = {}
    for name, transform in _representations().items():
        vectors = {
            (cell["row_index"], cell["slot"]): _unit(transform(cell["teacher_unit"]))
            for cell in cells
        }
        same_action_all: list[float] = []
        same_action_slot: list[float] = []
        nearest_sigma: list[float] = []
        own_consensus: list[float] = []
        other_action_consensus: list[float] = []
        separations: list[float] = []
        admissions = 0
        for left_row, right_row in action_pairs:
            for left_slot in range(4):
                for right_slot in range(4):
                    same_action_all.append(
                        _cosine(vectors[(left_row, left_slot)], vectors[(right_row, right_slot)])
                    )
                same_action_slot.append(
                    _cosine(vectors[(left_row, left_slot)], vectors[(right_row, left_slot)])
                )
                left_cell = cells[left_row * 4 + left_slot]
                closest = min(
                    range(4),
                    key=lambda slot: abs(
                        float(left_cell["sigma"])
                        - float(cells[right_row * 4 + slot]["sigma"])
                    ),
                )
                nearest_sigma.append(
                    _cosine(vectors[(left_row, left_slot)], vectors[(right_row, closest)])
                )

            for row, peer in ((left_row, right_row), (right_row, left_row)):
                own = _centroid([vectors[(peer, slot)] for slot in range(4)])
                other_rows = (2, 3) if row in (0, 1) else (0, 1)
                other = _centroid(
                    [vectors[(other_row, slot)] for other_row in other_rows for slot in range(4)]
                )
                for slot in range(4):
                    own_value = _cosine(vectors[(row, slot)], own)
                    other_value = _cosine(vectors[(row, slot)], other)
                    own_consensus.append(own_value)
                    other_action_consensus.append(other_value)
                    separations.append(own_value - other_value)
                    admissions += int(own_value > 0.0 and own_value > other_value)

        results[name] = {
            "same_action_cross_anchor_all_mean": _mean(same_action_all),
            "same_action_same_slot_mean": _mean(same_action_slot),
            "same_action_nearest_sigma_mean": _mean(nearest_sigma),
            "held_anchor_own_consensus_mean": _mean(own_consensus),
            "other_action_consensus_mean": _mean(other_action_consensus),
            "own_minus_other_consensus_mean": _mean(separations),
            "positive_discriminative_admission_fraction": admissions / len(own_consensus),
        }
    return {
        "schema_version": SCHEMA,
        "cache_schema_version": cache.get("schema_version"),
        "manifest_digest": cache.get("manifest_digest"),
        "rows": [cells[row * 4]["iid"] for row in range(4)],
        "action_row_pairs": [list(pair) for pair in action_pairs],
        "cell_count": len(cells),
        "representations": results,
        "claim_boundary": (
            "Mechanism diagnostic only; two anchors per action are insufficient for a "
            "production action-invariant claim."
        ),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--cache", required=True, type=Path)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    import torch

    args = parser().parse_args(argv)
    cache = torch.load(args.cache.resolve(strict=True), map_location="cpu", weights_only=False)
    if not isinstance(cache, Mapping):
        raise ConsensusDiagnosticError("cache root must be a mapping")
    print(json.dumps(analyze(cache), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
