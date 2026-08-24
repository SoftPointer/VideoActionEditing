#!/usr/bin/env python3
"""Post-confirmation fit-LOO representation postmortem for source8 hidden data.

This diagnostic is deliberately exploratory.  It was authored after the
preregistered centered-sketch candidate failed source8 confirmation.  It uses
only the two fit sources per family to rank a fixed registry of prior temporal
views and small centered-sketch hybrids, then reports the already-opened
confirmation behavior of the fit-LOO winner.  It can generate hypotheses for
a new population, but cannot rescue the failed gate or authorize an editor,
training, or an optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import diagnose_saic_source8_hidden_quotient_confirmation_v1 as confirmation  # noqa: E402
import diagnose_starc_core4_hidden_temporal_quotient_v1 as prior  # noqa: E402
import materialize_saic_source8_hidden_quotient_v1 as materializer  # noqa: E402


SCHEMA_VERSION = "bernini-saic-source8-hidden-fit-loo-postmortem-v1"
BASE = "centered_sketch_self_similarity"
SINGLE_VIEWS = (
    "raw_hidden",
    "centered_hidden",
    "temporal_velocity",
    "endpoint_arrow",
    "phase_energy",
    "velocity_energy",
    "temporal_singular_values",
    "global_temporal_self_similarity",
    "sketch_temporal_self_similarity",
    "centered_sketch_self_similarity",
    "centered_phase_mean",
)
CANDIDATES = (
    *((name,) for name in SINGLE_VIEWS),
    *((BASE, name) for name in SINGLE_VIEWS if name != BASE),
    (BASE, "endpoint_arrow", "temporal_velocity"),
    (BASE, "endpoint_arrow", "centered_phase_mean"),
    (BASE, "temporal_velocity", "centered_phase_mean"),
)


class Source8FitLOOPostmortemError(RuntimeError):
    """A sealed input or fixed exploratory topology failed closed."""


def candidate_id(components: Sequence[str]) -> str:
    if not components or len(set(components)) != len(components):
        raise Source8FitLOOPostmortemError("candidate components differ")
    return "+".join(components)


def compose_feature(views: Mapping[str, Any], components: Sequence[str]) -> Any:
    import torch

    if not components or any(name not in views for name in components):
        raise Source8FitLOOPostmortemError("candidate view is absent")
    parts = [confirmation._unit(views[name]) for name in components]
    result = confirmation._unit(torch.cat(parts))
    if not bool(torch.isfinite(result).all().item()):
        raise Source8FitLOOPostmortemError("composed feature is non-finite")
    return result


def _fit_one_source_direction(source: Mapping[str, Any], components: Sequence[str]) -> Any:
    import torch

    positive = compose_feature(source["views"]["forward"], components)
    contrasts = [
        confirmation._unit(
            positive - compose_feature(source["views"][negative], components)
        )
        for negative in ("noop", "reverse")
    ]
    direction = confirmation._unit(torch.stack(contrasts).mean(dim=0))
    if sum(float(torch.dot(direction, row).item()) for row in contrasts) < 0.0:
        direction = -direction
    return direction


def fit_loo_candidate(
    fit_by_family: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    components: Sequence[str],
) -> dict[str, Any]:
    import torch

    rows = []
    for actor_family in materializer.ACTOR_FAMILIES:
        sources = fit_by_family.get(actor_family)
        if not isinstance(sources, Sequence) or len(sources) != 2:
            raise Source8FitLOOPostmortemError("fit-LOO source topology differs")
        for train, test in ((sources[0], sources[1]), (sources[1], sources[0])):
            direction = _fit_one_source_direction(train, components)
            positive = compose_feature(test["views"]["forward"], components)
            for negative in ("noop", "reverse"):
                margin = float(
                    torch.dot(
                        direction,
                        positive - compose_feature(test["views"][negative], components),
                    ).item()
                )
                if not math.isfinite(margin):
                    raise Source8FitLOOPostmortemError("fit-LOO margin is non-finite")
                rows.append(
                    {
                        "actor_family": actor_family,
                        "train_iid": train["iid"],
                        "test_iid": test["iid"],
                        "negative_branch": negative,
                        "margin": margin,
                        "positive": margin > 0.0,
                    }
                )
    margins = [row["margin"] for row in rows]
    return {
        "candidate_id": candidate_id(components),
        "components": list(components),
        "positive_count": sum(row["positive"] for row in rows),
        "count": len(rows),
        "all_positive": all(row["positive"] for row in rows),
        "mean_margin": sum(margins) / len(margins),
        "minimum_margin": min(margins),
        "rows": rows,
    }


def _selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -row["positive_count"],
        -row["minimum_margin"],
        -row["mean_margin"],
        row["candidate_id"],
    )


def run(master_path: Path) -> dict[str, Any]:
    import torch

    master, sources = confirmation._load_population(master_path)
    prepared = []
    for source in sources.values():
        prepared.append(
            {
                **{key: value for key, value in source.items() if key != "branches"},
                "views": {
                    branch: prior.temporal_representations(row["tensor"])
                    for branch, row in source["branches"].items()
                },
            }
        )
    observed_views = tuple(prepared[0]["views"]["forward"])
    if observed_views != SINGLE_VIEWS or len(CANDIDATES) != 24:
        raise Source8FitLOOPostmortemError("candidate registry differs")
    fit_by_family = {
        family: [
            source
            for source in prepared
            if source["actor_family"] == family
            and source["analysis_split"] == "fit"
        ]
        for family in materializer.ACTOR_FAMILIES
    }
    candidate_results = sorted(
        (
            fit_loo_candidate(fit_by_family, components=components)
            for components in CANDIDATES
        ),
        key=_selection_key,
    )
    selected = candidate_results[0]
    selected_components = tuple(selected["components"])

    family_results = {}
    aggregate_rows = {branch: [] for branch in materializer.BRANCH_ORDER}
    for family in materializer.ACTOR_FAMILIES:
        family_sources = [source for source in prepared if source["actor_family"] == family]
        fit = [source for source in family_sources if source["analysis_split"] == "fit"]
        held = [
            source
            for source in family_sources
            if source["analysis_split"] == "confirmation"
        ]

        def selected_features(source: Mapping[str, Any]) -> dict[str, Any]:
            return {
                **source,
                "features": {
                    branch: compose_feature(source["views"][branch], selected_components)
                    for branch in materializer.BRANCH_ORDER
                },
            }

        fit_selected = [selected_features(source) for source in fit]
        held_selected = [selected_features(source) for source in held]
        by_positive = {
            branch: confirmation.evaluate_positive_branch(
                fit_sources=fit_selected,
                confirmation_sources=held_selected,
                positive_branch=branch,
            )
            for branch in materializer.BRANCH_ORDER
        }
        for branch in materializer.BRANCH_ORDER:
            aggregate_rows[branch].extend(by_positive[branch]["confirmation_rows"])
        family_results[family] = {
            "fit_iids": [source["iid"] for source in fit],
            "confirmation_iids": [source["iid"] for source in held],
            "by_positive_branch": by_positive,
        }
    aggregate = {}
    for branch, rows in aggregate_rows.items():
        margins = [row["margin"] for row in rows]
        aggregate[branch] = {
            "positive_count": sum(row["positive"] for row in rows),
            "count": len(rows),
            "all_positive": all(row["positive"] for row in rows),
            "mean_margin": sum(margins) / len(margins),
            "minimum_margin": min(margins),
            "rows": rows,
        }

    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "status": "post_confirmation_exploratory_fit_loo_no_authority",
        "master_binding": {
            "path": str(master_path),
            "file_sha256": confirmation.file_sha256(master_path),
            "receipt_digest": master["receipt_digest"],
        },
        "candidate_registry": {
            "single_views": list(SINGLE_VIEWS),
            "candidate_count": len(CANDIDATES),
            "composition": "unit-normalize each view, concatenate, unit-normalize",
            "selection_key": (
                "descending fit-LOO positive_count, minimum_margin, mean_margin; "
                "then lexical candidate_id"
            ),
        },
        "fit_only_selection": {
            "sources_per_family": 2,
            "folds_per_family": 2,
            "margins_per_candidate": 8,
            "candidate_results": candidate_results,
            "selected_candidate": selected,
            "confirmation_tensor_used_for_selection": False,
        },
        "already_opened_confirmation_of_selected_candidate": {
            "family_results": family_results,
            "aggregate_by_positive_branch": aggregate,
            "independent_confirmation": False,
            "can_rescue_preregistered_failure": False,
        },
        "interpretation": {
            "fit_loo_can_be_perfect_with_two_sources_per_family": selected[
                "all_positive"
            ],
            "selected_candidate_confirmation_forward_all_positive": aggregate[
                "forward"
            ]["all_positive"],
            "two_fit_sources_per_family_are_insufficient_for_representation_selection": (
                selected["all_positive"] and not aggregate["forward"]["all_positive"]
            ),
        },
        "limitations": {
            "authored_after_preregistered_confirmation_failure": True,
            "confirmation_population_already_opened": True,
            "candidate_registry_is_post_hoc": True,
            "new_scene_and_seed_confirmation_required": True,
        },
        "runtime_binding": {
            "source_sha256": confirmation.file_sha256(Path(__file__).resolve()),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "device": "cpu",
            "optimizer_constructed": False,
            "editor_forward_performed": False,
        },
        "authority": {
            "data_selection": False,
            "representation_selection": False,
            "editor_feature_target": False,
            "training": False,
            "optimizer": False,
            "editor_update": False,
            "scientific_claim": False,
        },
    }
    return {**unsigned, "receipt_digest": confirmation.object_sha256(unsigned)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.master.resolve(strict=True))
    confirmation._write_create_only(args.output.resolve(strict=False), result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "receipt_digest": result["receipt_digest"],
                "selected_candidate": result["fit_only_selection"][
                    "selected_candidate"
                ]["candidate_id"],
                "fit_loo": result["fit_only_selection"]["selected_candidate"][
                    "positive_count"
                ],
                "confirmation_forward": result[
                    "already_opened_confirmation_of_selected_candidate"
                ]["aggregate_by_positive_branch"]["forward"]["positive_count"],
                "authority": result["authority"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
