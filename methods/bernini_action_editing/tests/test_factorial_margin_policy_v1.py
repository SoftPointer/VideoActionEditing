from __future__ import annotations

import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import factorial_margin_policy_v1 as policy


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def branch(action: float, preservation: float) -> dict[str, float]:
    return {
        "action": action,
        "identity": preservation,
        "camera": preservation,
        "background": preservation,
        "owner": preservation,
        "quality": preservation,
    }


def candidate(*, forward: float, noop: float, reverse: float, preservation: float = 0.95) -> dict:
    return {
        "forward": branch(forward, preservation),
        "noop": branch(noop, preservation - 0.01),
        "reverse": branch(reverse, preservation - 0.01),
        "incomplete": branch(noop + 0.02, preservation - 0.01),
        "camera_only": branch(noop - 0.01, preservation - 0.01),
        "appearance_only": branch(noop - 0.02, preservation - 0.01),
        "wrong_actor_or_object": branch(noop - 0.03, preservation - 0.01),
    }


def cell(label: str, source: str, seed: int, scores: dict) -> dict:
    return {
        "cell_id": label,
        "source_id": source,
        "source_media_sha256": digest(f"media:{source}"),
        "seed": seed,
        "action_family": "sit",
        "candidate_scores": scores,
    }


def population() -> dict:
    score_contract = {
        "schema_version": policy.SCORE_CONTRACT_SCHEMA,
        "evaluator_sha256": digest("evaluator"),
        "action_score_semantics": "decoded_forward_event_order_unit_interval",
        "preservation_score_semantics": "source_relative_axis_similarity_unit_interval",
        "score_range": [0.0, 1.0],
        "higher_is_better": True,
        "branch_order": list(policy.BRANCHES),
        "preservation_axis_order": list(policy.PRESERVATION_AXES),
    }
    fit_scores_a = {
        "rank1": candidate(forward=0.82, noop=0.38, reverse=0.31),
        "rank2": candidate(forward=0.86, noop=0.34, reverse=0.27),
    }
    fit_scores_b = {
        "rank1": candidate(forward=0.78, noop=0.40, reverse=0.32),
        "rank2": candidate(forward=0.84, noop=0.35, reverse=0.26),
    }
    calibration_scores = {
        "rank1": candidate(forward=0.76, noop=0.41, reverse=0.34),
        "rank2": candidate(forward=0.81, noop=0.39, reverse=0.30),
    }
    return {
        "schema_version": policy.POPULATION_SCHEMA,
        "population_id": "factorial-canary-v1",
        "created_utc": "2026-08-13T13:00:00Z",
        "score_contract": score_contract,
        "confirmation_registry_sha256": digest("sealed-confirmation-registry"),
        "fit_cells": [
            cell("fit-a", "source-fit-a", 11, fit_scores_a),
            cell("fit-b", "source-fit-b", 12, fit_scores_b),
        ],
        "calibration_cells": [
            cell("cal-a", "source-cal-a", 21, calibration_scores),
        ],
    }


class FactorialMarginPolicyTests(unittest.TestCase):
    def test_policy_selects_fit_winner_and_freezes_calibration_gates(self) -> None:
        result = policy.author_policy(population())
        self.assertEqual(result["selected_candidate_id"], "rank2")
        self.assertTrue(result["optimizer_step_allowed"])
        self.assertFalse(result["confirmation_scores_consumed"])
        self.assertAlmostEqual(
            result["calibration_gate_policy"]["action_margin_minimums"]["noop"],
            0.42,
        )
        policy.validate_policy(result)

    def test_reverse_or_pseudo_positive_is_not_compensated(self) -> None:
        value = population()
        for cell_value in value["fit_cells"]:
            bad = cell_value["candidate_scores"]["rank2"]
            bad["camera_only"]["action"] = 0.99
        result = policy.author_policy(value)
        rank2 = next(
            row for row in result["fit_candidate_summaries"]
            if row["candidate_id"] == "rank2"
        )
        self.assertFalse(rank2["fit_eligible"])
        self.assertEqual(result["selected_candidate_id"], "rank1")

    def test_preservation_failure_yields_zero_update(self) -> None:
        value = population()
        for candidate_id in ("rank1", "rank2"):
            forward = value["calibration_cells"][0]["candidate_scores"][candidate_id]["forward"]
            forward["camera"] = 0.2
        result = policy.author_policy(value)
        self.assertFalse(result["optimizer_step_allowed"])
        self.assertEqual(result["status"], "zero_update_calibration_failed")

    def test_calibration_cannot_reselect_fit_winner(self) -> None:
        value = population()
        calibration = value["calibration_cells"][0]["candidate_scores"]
        calibration["rank1"] = candidate(forward=0.99, noop=0.05, reverse=0.04)
        calibration["rank2"] = candidate(forward=0.70, noop=0.40, reverse=0.35)
        result = policy.author_policy(value)
        self.assertEqual(result["selected_candidate_id"], "rank2")

    def test_confirmation_failure_does_not_change_policy(self) -> None:
        frozen = policy.author_policy(population())
        original_digest = frozen["policy_digest"]
        scores = {
            "rank1": candidate(forward=0.70, noop=0.30, reverse=0.25),
            "rank2": candidate(forward=0.60, noop=0.59, reverse=0.58),
        }
        confirmation = {
            "schema_version": policy.CONFIRMATION_SCHEMA,
            "policy_digest": original_digest,
            "score_contract": frozen["score_contract"],
            "confirmation_registry_sha256": frozen["confirmation_registry_sha256"],
            "cells": [cell("confirmation-fail", "source-confirm-fail", 32, scores)],
        }
        receipt = policy.evaluate_confirmation(frozen, confirmation)
        self.assertFalse(receipt["all_confirmation_cells_pass"])
        self.assertEqual(frozen["policy_digest"], original_digest)
        self.assertFalse(receipt["optimizer_step_allowed"])

    def test_recomputed_digest_cannot_make_negative_gate_deployable(self) -> None:
        frozen = policy.author_policy(population())
        forged = copy.deepcopy(frozen)
        forged["calibration_gate_policy"]["action_margin_minimums"]["noop"] = -1.0
        unsigned = {key: value for key, value in forged.items() if key != "policy_digest"}
        forged["policy_digest"] = policy.object_sha256(unsigned)
        with self.assertRaises(policy.FactorialMarginError):
            policy.validate_policy(forged)

    def test_fit_calibration_source_leakage_is_rejected(self) -> None:
        value = population()
        value["calibration_cells"][0]["source_id"] = "source-fit-a"
        with self.assertRaises(policy.FactorialMarginError):
            policy.author_policy(value)

    def test_confirmation_uses_frozen_thresholds_and_disjoint_sources(self) -> None:
        frozen = policy.author_policy(population())
        scores = {
            "rank1": candidate(forward=0.70, noop=0.30, reverse=0.25),
            "rank2": candidate(forward=0.90, noop=0.30, reverse=0.20),
        }
        confirmation = {
            "schema_version": policy.CONFIRMATION_SCHEMA,
            "policy_digest": frozen["policy_digest"],
            "score_contract": frozen["score_contract"],
            "confirmation_registry_sha256": frozen["confirmation_registry_sha256"],
            "cells": [cell("confirmation-a", "source-confirm-a", 31, scores)],
        }
        receipt = policy.evaluate_confirmation(frozen, confirmation)
        self.assertTrue(receipt["thresholds_frozen_before_confirmation"])
        self.assertTrue(receipt["all_confirmation_cells_pass"])
        self.assertFalse(receipt["optimizer_step_allowed"])
        self.assertFalse(receipt["method_success_claimed"])

    def test_confirmation_cannot_reuse_fit_media(self) -> None:
        frozen = policy.author_policy(population())
        scores = {
            "rank1": candidate(forward=0.70, noop=0.30, reverse=0.25),
            "rank2": candidate(forward=0.90, noop=0.30, reverse=0.20),
        }
        leaked = cell("confirmation-a", "source-confirm-a", 31, scores)
        leaked["source_media_sha256"] = frozen["fit_source_media_sha256s"][0]
        confirmation = {
            "schema_version": policy.CONFIRMATION_SCHEMA,
            "policy_digest": frozen["policy_digest"],
            "score_contract": frozen["score_contract"],
            "confirmation_registry_sha256": frozen["confirmation_registry_sha256"],
            "cells": [leaked],
        }
        with self.assertRaises(policy.FactorialMarginError):
            policy.evaluate_confirmation(frozen, confirmation)

    def test_cli_writes_create_only_read_only_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pop_path = root / "population.json"
            output = root / "policy.json"
            pop_path.write_text(json.dumps(population()), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    policy.main(
                        [
                            "author-policy",
                            "--population", str(pop_path),
                            "--output", str(output),
                        ]
                    ),
                    0,
                )
            self.assertEqual(output.stat().st_mode & 0o777, 0o444)
            with self.assertRaises(policy.FactorialMarginError):
                with redirect_stdout(io.StringIO()):
                    policy.main(
                        [
                            "author-policy",
                            "--population", str(pop_path),
                            "--output", str(output),
                        ]
                    )


if __name__ == "__main__":
    unittest.main()
