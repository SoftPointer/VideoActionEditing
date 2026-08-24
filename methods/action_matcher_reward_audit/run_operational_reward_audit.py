#!/usr/bin/env python3
"""Replay the operational group-relative reward on frozen audit features."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from operational_reward import (
    RewardConfig,
    pareto_preference_pairs,
    score_candidate_pool,
    select_training_pair,
)
from run_sequence_audit import (
    controlled_variants,
    load_records,
    object_sha256,
    write_json,
)


SCHEMA_VERSION = "action-editing-operational-reward-audit-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def m3(record: Mapping[str, Any]) -> torch.Tensor:
    components = record["components"].float()
    if components.ndim != 2 or components.shape[0] < 3:
        raise ValueError(f"invalid SemanticMoments components for {record['item_id']}")
    return components[2]


def candidate(record: Mapping[str, Any]) -> dict[str, Any]:
    identifier = record.get("metadata", {}).get("candidate_id", record["item_id"])
    return {
        "candidate_id": identifier,
        "m3": m3(record),
        "frame_sequence": record["frame_sequence"].float(),
    }


def count_rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "count": int(numerator),
        "total": int(denominator),
        "rate": float(numerator / denominator) if denominator else 0.0,
    }


def result_summary(rows: Sequence[Mapping[str, Any]], expected_key: str) -> dict[str, Any]:
    count = len(rows)
    diagnostic_top_expected = sum(row["diagnostic_top_expected"] for row in rows)
    abstained = sum(row["abstain_required"] for row in rows)
    authorized_correct = sum(
        row["diagnostic_top_expected"] and row["selection_authorized"]
        for row in rows
    )
    authorized_wrong = sum(
        (not row["diagnostic_top_expected"])
        and row["selection_authorized"]
        for row in rows
    )
    training_pair_available = sum(row["training_pair"] is not None for row in rows)
    return {
        "expected_role": expected_key,
        "population": count,
        "diagnostic_top_expected": count_rate(diagnostic_top_expected, count),
        "abstain_required": count_rate(abstained, count),
        "authorized_correct": count_rate(authorized_correct, count),
        "authorized_wrong": count_rate(authorized_wrong, count),
        "training_pair_available": count_rate(training_pair_available, count),
        "rows": list(rows),
    }


def audit_project_contract(
    records: Sequence[Mapping[str, Any]], contract: str
) -> dict[str, Any]:
    rows = []
    by_id = {
        record["metadata"]["candidate_id"]: record for record in records
    }
    for reference in records:
        metadata = reference["metadata"]
        if metadata["branch"] != "forward":
            continue
        siblings = [
            row
            for row in records
            if row["metadata"]["iid"] == metadata["iid"]
            and row["metadata"]["seed"] != metadata["seed"]
        ]
        if not siblings:
            continue
        result = score_candidate_pool(
            reference_id=metadata["candidate_id"],
            reference_m3=m3(reference),
            reference_sequence=reference["frame_sequence"].float(),
            candidates=[candidate(row) for row in siblings],
            config=RewardConfig(contract=contract),
            valid_candidate_prior=True,
        )
        diagnostic_top = result["diagnostic_top_candidate_id"]
        diagnostic_top_branch = (
            by_id[diagnostic_top]["metadata"]["branch"] if diagnostic_top else None
        )
        rows.append(
            {
                "reference_id": metadata["candidate_id"],
                "iid": metadata["iid"],
                "pool_size": len(siblings),
                "diagnostic_top_candidate_id": diagnostic_top,
                "diagnostic_top_branch": diagnostic_top_branch,
                "diagnostic_top_expected": diagnostic_top_branch == "forward",
                "selected_candidate_id": result["selected_candidate_id"],
                "abstain_required": result["abstain_required"],
                "abstain_reasons": result["abstain_reasons"],
                "selection_authorized": result["selection_authorized"],
                "pareto_pair_count": len(pareto_preference_pairs(result)),
                "training_pair": select_training_pair(
                    result, minimum_event_gain=0.20
                ),
                "reward_result": result,
            }
        )
    summary = result_summary(rows, "generation-contract forward branch")
    summary["diagnostic_top_branch_counts"] = dict(
        sorted(Counter(row["diagnostic_top_branch"] for row in rows).items(), key=str)
    )
    summary["label_authority"] = (
        "generation branch contract only; no human correctness truth"
    )
    return summary


def audit_project(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        contract: audit_project_contract(records, contract)
        for contract in ("generic_ordered", "directional_endpoint")
    }


def audit_simmotion(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["metadata"]["example_id"]].append(record)
    output = {}
    for contract in ("generic_ordered", "cyclic"):
        rows = []
        for example_id, example_rows in sorted(grouped.items()):
            by_role = {row["metadata"]["role"]: row for row in example_rows}
            reference = by_role["ref"]
            candidates = [by_role["positive"], by_role["negative"]]
            result = score_candidate_pool(
                reference_id=reference["item_id"],
                reference_m3=m3(reference),
                reference_sequence=reference["frame_sequence"].float(),
                candidates=[candidate(row) for row in candidates],
                config=RewardConfig(contract=contract),
                valid_candidate_prior=True,
            )
            diagnostic_top = result["diagnostic_top_candidate_id"]
            diagnostic_top_role = next(
                (
                    row["metadata"]["role"]
                    for row in candidates
                    if row["item_id"] == diagnostic_top
                ),
                None,
            )
            rows.append(
                {
                    "example_id": example_id,
                    "diagnostic_top_candidate_id": diagnostic_top,
                    "diagnostic_top_role": diagnostic_top_role,
                    "diagnostic_top_expected": diagnostic_top_role == "positive",
                    "selected_candidate_id": result["selected_candidate_id"],
                    "abstain_required": result["abstain_required"],
                    "abstain_reasons": result["abstain_reasons"],
                    "selection_authorized": result["selection_authorized"],
                    "pareto_pair_count": len(pareto_preference_pairs(result)),
                    "training_pair": select_training_pair(
                        result, minimum_event_gain=0.20
                    ),
                    "reward_result": result,
                }
            )
        output[contract] = result_summary(rows, "dataset-designated positive")
        output[contract]["label_authority"] = (
            "dataset-designated retrieval role only; no human correctness truth"
        )
    return output


def audit_controlled_stress(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stress abstention with objective counterfactuals from the same video.

    M3 is deliberately held equal for every variant.  This is conservative for
    detecting whether order/activity evidence alone can prevent false
    authorization when the valid-candidate prior is accidentally misasserted.
    """

    rows = []
    variants_to_use = (
        "reverse",
        "random_shuffle",
        "noop_first_frame",
        "incomplete_tail_hold",
    )
    for reference in records:
        sequence = reference["frame_sequence"].float()
        variants = controlled_variants(reference["item_id"], sequence)
        candidates = [
            {
                "candidate_id": f"{reference['item_id']}::{name}",
                "m3": m3(reference),
                "frame_sequence": variants[name],
            }
            for name in variants_to_use
        ]
        diagnostic = score_candidate_pool(
            reference_id=reference["item_id"],
            reference_m3=m3(reference),
            reference_sequence=sequence,
            candidates=candidates,
            config=RewardConfig(contract="generic_ordered"),
            valid_candidate_prior=True,
        )
        protected = score_candidate_pool(
            reference_id=reference["item_id"],
            reference_m3=m3(reference),
            reference_sequence=sequence,
            candidates=candidates,
            config=RewardConfig(contract="generic_ordered"),
            valid_candidate_prior=False,
        )
        rows.append(
            {
                "reference_id": reference["item_id"],
                "misasserted_prior_would_authorize": diagnostic[
                    "selection_authorized"
                ],
                "unasserted_prior_abstains": protected["abstain_required"],
                "diagnostic_result": diagnostic,
            }
        )
    population = len(rows)
    false_authorized = sum(row["misasserted_prior_would_authorize"] for row in rows)
    protected = sum(row["unasserted_prior_abstains"] for row in rows)
    return {
        "population": population,
        "candidate_variants": list(variants_to_use),
        "m3_stress_condition": "all candidate M3 vectors equal the reference M3",
        "misasserted_valid_prior_false_authorization": count_rate(
            false_authorized, population
        ),
        "unasserted_valid_prior_abstention": count_rate(protected, population),
        "rows": rows,
    }


def audit(args: argparse.Namespace) -> int:
    started_at = time.perf_counter()
    records, receipt = load_records(args.feature_root)
    simmotion = [row for row in records if row["group"] == "simmotion_real"]
    project = [row for row in records if row["group"] == "project_saic_bank"]
    if (len(simmotion), len(project)) != (120, 60):
        raise ValueError(
            f"unexpected frozen population: simmotion={len(simmotion)}, project={len(project)}"
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "authority": {
            "reward_authorized": False,
            "automatic_acceptance_authorized": False,
            "preference_data_authorized": False,
            "optimizer_update_authorized": False,
        },
        "implementation_contract": {
            "reward_type": "group-relative Best-of-N reranker with abstention",
            "not_supported": "candidate-independent absolute same-action verification",
            "semantic_moments_role": "coarse motion-set axis only",
            "abstention_action": "drop training group; keep source or resample at inference",
            "slow_model_fallback": "none",
        },
        "feature_receipt": receipt,
        "project_branch_contract": audit_project(project),
        "simmotion_designated": audit_simmotion(simmotion),
        "controlled_negative_only_stress": audit_controlled_stress(records),
    }
    result["reward_head_wall_seconds"] = time.perf_counter() - started_at
    result["result_digest"] = object_sha256(result)
    write_json(args.output, result)
    compact = {
        "output": str(Path(args.output).resolve()),
        "digest": result["result_digest"],
        "project": {
            contract: {key: value for key, value in summary.items() if key != "rows"}
            for contract, summary in result["project_branch_contract"].items()
        },
        "simmotion": {
            contract: {
                key: value
                for key, value in summary.items()
                if key not in {"rows"}
            }
            for contract, summary in result["simmotion_designated"].items()
        },
        "negative_only": {
            key: value
            for key, value in result["controlled_negative_only_stress"].items()
            if key not in {"rows"}
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    return audit(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
