from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_action_energy_calibration as calibration  # noqa: E402


GENERATOR_DIGEST = "a" * 64
SCORER_DIGEST = "b" * 64
FAMILIES = ("articulated", "interaction")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row(
    split: str,
    family: str,
    branch: str,
    score: float,
    *,
    suffix: str = "0",
    event_qualified: bool | None = None,
):
    if event_qualified is None:
        event_qualified = branch == calibration.ACTION_BRANCH
    return calibration.make_score_row(
        f"{split}-{family}-{branch}-{suffix}",
        split=split,
        action_family=family,
        prompt_group=f"{split}-{family}-prompt",
        action_family_group=f"{split}-{family}-instance",
        branch=branch,
        raw_phase_conjunctive_score=score,
        event_qualified=event_qualified,
        frozen_generator_receipt_digest=GENERATOR_DIGEST,
        frozen_scorer_receipt_digest=SCORER_DIGEST,
        event_qualification_receipt_digest=_digest(
            f"event:{split}:{family}:{branch}:{suffix}"
        ),
    )


def _rows(
    *,
    articulated_fit=(0.90, 0.10),
    articulated_confirmation=(0.85, 0.15),
    interaction_fit=(9.0, 1.0),
    interaction_confirmation=(8.5, 1.5),
):
    score_pairs = {
        ("articulated", "fit"): articulated_fit,
        ("articulated", "confirmation"): articulated_confirmation,
        ("interaction", "fit"): interaction_fit,
        ("interaction", "confirmation"): interaction_confirmation,
    }
    result = []
    for family in FAMILIES:
        for split in calibration.SPLITS:
            positive, negative = score_pairs[(family, split)]
            result.append(_row(split, family, calibration.ACTION_BRANCH, positive))
            result.extend(
                _row(split, family, branch, negative)
                for branch in calibration.NEGATIVE_BRANCHES
            )
    return result


def _policy(**updates):
    defaults = {
        "action_families": FAMILIES,
        "fit_positive_lower_quantile": 0.10,
        "fit_negative_upper_quantile": 0.90,
        "minimum_fit_anchor_gap": 0.05,
        "decision_threshold": 0.50,
        "minimum_confirmation_auroc": 0.90,
        "minimum_confirmation_positive_recall": 0.80,
        "minimum_confirmation_negative_specificity": 0.80,
    }
    defaults.update(updates)
    return calibration.make_preregistration("pair-v5-self-action-v2", **defaults)


def _calibrate(rows=None, policy=None):
    rows = _rows() if rows is None else rows
    policy = _policy() if policy is None else policy
    return calibration.calibrate_action_energy(
        rows,
        policy,
        registered_preregistration_digest=policy["preregistration_digest"],
    )


def _reseal_row(row, **changes):
    unsigned = dict(row)
    unsigned.pop("row_digest")
    unsigned.update(changes)
    unsigned["row_digest"] = calibration.object_sha256(unsigned)
    return unsigned


class PassingCalibrationTests(unittest.TestCase):
    def test_exact_mace_branch_closure(self):
        self.assertEqual(
            calibration.BRANCHES,
            (
                "action",
                "noop",
                "incomplete",
                "reverse",
                "shuffle",
                "wrong_actor",
                "wrong_object",
                "camera_only",
                "appearance_only",
                "generic_wrong_motion",
            ),
        )

    def test_per_family_fit_and_disjoint_confirmation_authorize_optimizer(self):
        receipt = _calibrate()
        self.assertTrue(receipt["optimizer_authorized"])
        self.assertEqual(receipt["failure_reasons"], [])
        self.assertEqual(set(receipt["mapping_by_family"]), set(FAMILIES))
        self.assertAlmostEqual(
            receipt["mapping_by_family"]["articulated"]["lower_raw_anchor"], 0.10
        )
        self.assertAlmostEqual(
            receipt["mapping_by_family"]["articulated"]["upper_raw_anchor"], 0.90
        )
        self.assertAlmostEqual(
            receipt["mapping_by_family"]["interaction"]["lower_raw_anchor"], 1.0
        )
        self.assertAlmostEqual(
            receipt["mapping_by_family"]["interaction"]["upper_raw_anchor"], 9.0
        )
        self.assertEqual(receipt["confirmation_metrics"]["overall"]["auroc"], 1.0)
        self.assertTrue(all(receipt["gates"]["confirmation_by_family"].values()))
        self.assertEqual(
            receipt["positive_definition"],
            "branch==action AND event_qualified==true",
        )
        closure = receipt["input_closure"]
        self.assertTrue(closure["t2v_self_generations_are_calibration_only"])
        self.assertFalse(closure["proposal_media_consumed"])
        self.assertFalse(closure["proposal_latent_consumed"])
        self.assertFalse(closure["proposal_noise_consumed"])

    def test_apply_uses_family_scale_and_rv2v_score_is_scalar_only(self):
        receipt = _calibrate()
        articulated = calibration.apply_calibrator(
            0.50,
            "articulated",
            receipt,
            registered_calibration_receipt_digest=receipt["receipt_digest"],
        )
        interaction = calibration.apply_calibrator(
            5.0,
            "interaction",
            receipt,
            registered_calibration_receipt_digest=receipt["receipt_digest"],
        )
        self.assertAlmostEqual(articulated, 0.5)
        self.assertAlmostEqual(interaction, 0.5)
        scored = calibration.score_rv2v_candidate(
            "rv2v-candidate-17",
            action_family="articulated",
            raw_candidate_own_score=0.50,
            candidate_evaluator_receipt_digest="c" * 64,
            calibration_receipt=receipt,
            registered_calibration_receipt_digest=receipt["receipt_digest"],
        )
        self.assertAlmostEqual(scored["calibrated_action_score"], 0.5)
        self.assertEqual(scored["score_coordinate"], "candidate_own_exact81")
        self.assertFalse(scored["proposal_visual_data_consumed"])
        self.assertFalse(scored["privileged_visual_inputs_consumed"])

    def test_confirmation_cannot_move_fit_anchors(self):
        first = _calibrate()
        second = _calibrate(
            _rows(
                articulated_confirmation=(0.95, 0.05),
                interaction_confirmation=(9.5, 0.5),
            )
        )
        self.assertEqual(first["mapping_by_family"], second["mapping_by_family"])

    def test_provenance_binds_each_family_mapping(self):
        receipt = _calibrate()
        provenance = calibration.make_calibrator_provenance(
            receipt,
            registered_calibration_receipt_digest=receipt["receipt_digest"],
        )
        self.assertTrue(provenance["optimizer_authorized"])
        self.assertEqual(set(provenance["mapping_digest_by_family"]), set(FAMILIES))
        self.assertFalse(provenance["proposal_visual_data_consumed"])


class FailClosedCalibrationTests(unittest.TestCase):
    def test_unqualified_action_is_not_a_positive(self):
        row = _row(
            "fit",
            "articulated",
            calibration.ACTION_BRANCH,
            0.89,
            suffix="unqualified",
            event_qualified=False,
        )
        self.assertFalse(calibration.event_qualified_positive(row))
        receipt = _calibrate([*_rows(), row])
        self.assertGreater(
            receipt["mapping_by_family"]["articulated"]["lower_raw_anchor"],
            0.10,
        )

    def test_missing_one_negative_returns_sealed_unauthorized_receipt(self):
        rows = [
            row
            for row in _rows()
            if not (
                row["split"] == "confirmation"
                and row["action_family"] == "interaction"
                and row["branch"] == "shuffle"
            )
        ]
        receipt = _calibrate(rows)
        self.assertFalse(receipt["optimizer_authorized"])
        self.assertFalse(receipt["gates"]["coverage_complete"])
        self.assertIsNone(receipt["mapping_by_family"])
        self.assertIsNone(receipt["confirmation_metrics"])
        self.assertIn("coverage:confirmation:interaction:shuffle", receipt["failure_reasons"])
        calibration.validate_calibration_receipt(receipt)
        with self.assertRaisesRegex(calibration.PairV5CalibrationError, "no usable RV2V"):
            calibration.apply_calibrator(
                8.0,
                "interaction",
                receipt,
                registered_calibration_receipt_digest=receipt["receipt_digest"],
            )

    def test_empty_confirmation_split_is_a_sealed_failure_not_an_optimizer_input(self):
        receipt = _calibrate([row for row in _rows() if row["split"] == "fit"])
        self.assertEqual(receipt["confirmation_row_count"], 0)
        self.assertFalse(receipt["optimizer_authorized"])
        self.assertFalse(receipt["gates"]["coverage_complete"])
        self.assertIsNone(receipt["mapping_by_family"])
        calibration.validate_calibration_receipt(receipt)

    def test_prompt_and_action_family_group_leakage_return_no_mapping(self):
        for axis in calibration.GROUP_AXES:
            with self.subTest(axis=axis):
                rows = _rows()
                fit_value = next(row[axis] for row in rows if row["split"] == "fit")
                replaced = [
                    _reseal_row(row, **{axis: fit_value})
                    if row["split"] == "confirmation"
                    else row
                    for row in rows
                ]
                receipt = _calibrate(replaced)
                self.assertFalse(receipt["optimizer_authorized"])
                self.assertFalse(receipt["gates"]["group_disjoint"])
                self.assertIsNone(receipt["mapping_by_family"])
                self.assertTrue(
                    any(reason.startswith(f"group_leakage:{axis}:") for reason in receipt["failure_reasons"])
                )

    def test_one_family_anchor_overlap_disables_every_family(self):
        receipt = _calibrate(_rows(interaction_fit=(1.04, 1.0)))
        self.assertFalse(receipt["optimizer_authorized"])
        self.assertFalse(
            receipt["gates"]["fit_anchor_separation_by_family"]["interaction"]
        )
        self.assertIsNone(receipt["mapping_by_family"])
        self.assertIn(
            "fit_anchor_gap_below_minimum:interaction", receipt["failure_reasons"]
        )

    def test_one_family_confirmation_failure_disables_optimizer(self):
        rows = _rows()
        replaced = []
        for row in rows:
            if (
                row["split"] == "confirmation"
                and row["action_family"] == "interaction"
                and row["branch"] == "wrong_actor"
            ):
                replaced.append(
                    _reseal_row(row, raw_phase_conjunctive_score=8.8)
                )
            else:
                replaced.append(row)
        receipt = _calibrate(replaced)
        self.assertFalse(receipt["optimizer_authorized"])
        self.assertFalse(receipt["gates"]["confirmation_by_family"]["interaction"])
        self.assertIn("confirmation_metrics_failed:interaction", receipt["failure_reasons"])
        with self.assertRaisesRegex(calibration.PairV5CalibrationError, "no usable RV2V"):
            calibration.score_rv2v_candidate(
                "candidate-1",
                action_family="interaction",
                raw_candidate_own_score=8.0,
                candidate_evaluator_receipt_digest="c" * 64,
                calibration_receipt=receipt,
                registered_calibration_receipt_digest=receipt["receipt_digest"],
            )

    def test_resealed_preregistration_cannot_replace_external_commitment(self):
        policy = _policy()
        resealed = deepcopy(policy)
        resealed.pop("preregistration_digest")
        resealed["minimum_confirmation_auroc"] = 0.10
        resealed["preregistration_digest"] = calibration.object_sha256(resealed)
        calibration.validate_preregistration(resealed)
        with self.assertRaisesRegex(calibration.PairV5CalibrationError, "externally registered"):
            calibration.calibrate_action_energy(
                _rows(),
                resealed,
                registered_preregistration_digest=policy["preregistration_digest"],
            )

    def test_tampered_or_resealed_receipt_cannot_replace_pinned_receipt(self):
        receipt = _calibrate()
        tampered = deepcopy(receipt)
        tampered["mapping_by_family"]["articulated"]["upper_raw_anchor"] = 100.0
        with self.assertRaisesRegex(calibration.PairV5CalibrationError, "embedded digest mismatch"):
            calibration.apply_calibrator(
                0.8,
                "articulated",
                tampered,
                registered_calibration_receipt_digest=receipt["receipt_digest"],
            )

        resealed = deepcopy(receipt)
        resealed.pop("receipt_digest")
        mapping = dict(resealed["mapping_by_family"]["articulated"])
        mapping.pop("mapping_digest")
        mapping["upper_raw_anchor"] = 100.0
        mapping["mapping_digest"] = calibration.object_sha256(mapping)
        resealed["mapping_by_family"]["articulated"] = mapping
        resealed["receipt_digest"] = calibration.object_sha256(resealed)
        calibration.validate_calibration_receipt(resealed)
        with self.assertRaisesRegex(calibration.PairV5CalibrationError, "externally registered"):
            calibration.apply_calibrator(
                0.8,
                "articulated",
                resealed,
                registered_calibration_receipt_digest=receipt["receipt_digest"],
            )

    def test_rows_must_be_frozen_t2v_candidate_own_exact81(self):
        row = _row("fit", "articulated", "action", 0.9)
        for field, value, pattern in (
            ("generation_mode", "native_rv2v_candidate", "not frozen Bernini T2V"),
            ("score_coordinate", "proposal_coordinate", "not candidate-own exact81"),
            ("frame_count", 41, "exactly 81"),
        ):
            with self.subTest(field=field):
                changed = _reseal_row(row, **{field: value})
                with self.assertRaisesRegex(calibration.PairV5CalibrationError, pattern):
                    calibration.validate_score_row(changed)

    def test_api_has_no_media_tensor_or_privileged_slots(self):
        forbidden = {
            "video",
            "latent",
            "noise",
            "proposal",
            "source",
            "target",
            "donor",
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
        }
        for function in (
            calibration.calibrate_action_energy,
            calibration.apply_calibrator,
            calibration.score_rv2v_candidate,
        ):
            parameters = set(inspect.signature(function).parameters)
            self.assertTrue(forbidden.isdisjoint(parameters))


if __name__ == "__main__":
    unittest.main()
