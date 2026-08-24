from __future__ import annotations

from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operational_reward import (  # noqa: E402
    RewardConfig,
    average_tie_percentiles,
    pareto_preference_pairs,
    score_candidate_pool,
    select_training_pair,
)


def _sequence() -> torch.Tensor:
    phase = torch.linspace(-1.0, 1.0, 16)
    return torch.stack(
        [phase, phase.square(), torch.sin(phase * 2.3), torch.cos(phase * 1.7)],
        dim=1,
    )


def _candidate(identifier: str, sequence: torch.Tensor) -> dict[str, object]:
    return {
        "candidate_id": identifier,
        "m3": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "frame_sequence": sequence,
    }


def test_average_tie_percentiles() -> None:
    assert average_tie_percentiles([2.0, 2.0, 5.0]) == [0.25, 0.25, 1.0]
    assert average_tie_percentiles([4.0]) == [1.0]


def test_directional_contract_rejects_exact_reverse_and_noop() -> None:
    reference = _sequence()
    result = score_candidate_pool(
        reference_id="anchor",
        reference_m3=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        reference_sequence=reference,
        candidates=[
            _candidate("forward", reference.clone()),
            _candidate("reverse", torch.flip(reference, dims=(0,))),
            _candidate("noop", reference[0:1].repeat(len(reference), 1)),
        ],
        config=RewardConfig(minimum_top_gap=0.0),
        valid_candidate_prior=True,
    )
    by_id = {row["candidate_id"]: row for row in result["candidates"]}
    assert result["selected_candidate_id"] == "forward"
    assert by_id["forward"]["eligible"]
    assert "reverse_explains_candidate_at_least_as_well" in by_id["reverse"][
        "gate_reasons"
    ]
    assert "activity_below_reference_ratio_floor" in by_id["noop"][
        "gate_reasons"
    ]
    assert by_id["forward"]["event_score"] == min(
        by_id["forward"]["pool_percentiles"].values()
    )


def test_unasserted_valid_candidate_prior_always_abstains() -> None:
    reference = _sequence()
    result = score_candidate_pool(
        reference_id="anchor",
        reference_m3=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        reference_sequence=reference,
        candidates=[_candidate("only", reference)],
    )
    assert result["abstain_required"]
    assert not result["selection_authorized"]
    assert result["selected_candidate_id"] is None
    assert result["diagnostic_top_candidate_id"] == "only"
    assert "valid_candidate_prior_not_asserted" in result["abstain_reasons"]


def test_pareto_pairs_do_not_allow_cross_axis_compensation() -> None:
    result = {
        "required_axes": ["a", "b"],
        "candidates": [
            {
                "candidate_id": "dominant",
                "eligible": True,
                "event_score": 0.8,
                "raw_scores": {"a": 2.0, "b": 2.0},
            },
            {
                "candidate_id": "dominated",
                "eligible": True,
                "event_score": 0.2,
                "raw_scores": {"a": 1.0, "b": 1.0},
            },
            {
                "candidate_id": "tradeoff",
                "eligible": True,
                "event_score": 0.1,
                "raw_scores": {"a": 3.0, "b": 0.5},
            },
        ],
    }
    pairs = pareto_preference_pairs(result)
    identities = {
        (row["chosen_candidate_id"], row["rejected_candidate_id"])
        for row in pairs
    }
    assert ("dominant", "dominated") in identities
    assert ("dominant", "tradeoff") not in identities
    assert ("tradeoff", "dominant") not in identities


def test_training_pair_is_zero_update_when_pool_abstains() -> None:
    assert select_training_pair({"abstain_required": True}) is None


def test_training_pair_can_reject_a_hard_gated_reverse() -> None:
    reference = _sequence()
    result = score_candidate_pool(
        reference_id="anchor",
        reference_m3=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        reference_sequence=reference,
        candidates=[
            _candidate("forward", reference.clone()),
            _candidate("reverse", torch.flip(reference, dims=(0,))),
            _candidate("noop", reference[0:1].repeat(len(reference), 1)),
        ],
        config=RewardConfig(minimum_top_gap=0.0),
        valid_candidate_prior=True,
    )
    pair = select_training_pair(result, minimum_event_gain=0.20)
    assert pair is not None
    assert pair["chosen_candidate_id"] == "forward"
    assert pair["rejected_candidate_id"] in {"reverse", "noop"}


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} operational reward tests passed")
