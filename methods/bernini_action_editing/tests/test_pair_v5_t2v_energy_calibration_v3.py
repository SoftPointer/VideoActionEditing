from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_t2v_energy_calibration_v3 as v3  # noqa: E402


FAMILIES = ("dog-sit-facing-camera", "human-rise-to-stand")


def _artifacts():
    rows = []
    audits = []
    counter = 0
    for family_index, family in enumerate(FAMILIES):
        for split_index, split in enumerate(v3.ANALYSIS_SPLITS):
            cell = f"{family}-{split}"
            for branch_index, branch in enumerate(v3.BRANCH_ORDER):
                candidate = f"candidate-{family_index}-{split_index}-{branch_index}"
                audit = v3.seal_event_audit_receipt(
                    candidate_id=candidate,
                    analysis_split=split,
                    action_family_id=family,
                    calibration_group_id=cell,
                    actor_group_id=f"actor-{family_index}-{split}",
                    scene_group_id=f"scene-{family_index}-{split}",
                    action_group_id=f"action-instance-{family_index}-{split}",
                    semantic_branch=branch,
                    generation_receipt_digest=f"{counter + 1:064x}",
                    audit_source_kind="vlm_detached",
                    external_audit_artifact_sha256=f"{counter + 1000:064x}",
                    complete_target_transition_observed=branch == "action",
                    terminal_hold_observed=branch == "action",
                    full_target_action_observed=branch == "action",
                    full_target_action_false_confirmed=branch != "action",
                )
                score = 4.0 + family_index if branch == "action" else -2.0 - branch_index / 10
                row = v3.make_score_row(
                    row_id=f"row-{candidate}",
                    candidate_id=candidate,
                    analysis_split=split,
                    action_family_id=family,
                    calibration_group_id=cell,
                    actor_group_id=f"actor-{family_index}-{split}",
                    scene_group_id=f"scene-{family_index}-{split}",
                    action_group_id=f"action-instance-{family_index}-{split}",
                    semantic_branch=branch,
                    raw_global_action_energy_score=float(score),
                    generation_receipt_digest=f"{counter + 1:064x}",
                    frozen_scorer_receipt_digest=f"{counter + 2000:064x}",
                    event_audit_receipt_digest=audit["receipt_digest"],
                )
                rows.append(row)
                audits.append(audit)
                counter += 1
    prereg = v3.make_preregistration("core4-index33-v3", FAMILIES)
    return rows, audits, prereg


def _calibrate(rows, audits, prereg):
    return v3.calibrate_global_action_energy(
        rows,
        audits,
        prereg,
        source_bank_spec_sha256="a" * 64,
        source_bank_receipt_digest="b" * 64,
    )


class GlobalEnergyCalibrationV3Tests(unittest.TestCase):
    def test_full_event_qualified_disjoint_bank_passes(self) -> None:
        rows, audits, prereg = _artifacts()
        receipt = _calibrate(rows, audits, prereg)
        self.assertTrue(receipt["optimizer_authorized"])
        self.assertEqual(receipt["failure_reasons"], [])
        self.assertEqual(receipt["score_field"], "raw_global_action_energy_score")
        self.assertFalse(receipt["phase_conjunctive_score_used_for_calibration"])
        self.assertEqual(receipt["source_bank_spec_sha256"], "a" * 64)
        self.assertEqual(receipt["source_bank_receipt_digest"], "b" * 64)
        self.assertEqual(set(receipt["raw_score_evidence_by_family"]), set(FAMILIES))
        for family in FAMILIES:
            for split in v3.ANALYSIS_SPLITS:
                evidence = receipt["raw_score_evidence_by_family"][family][split]
                self.assertEqual(len(evidence), 10)
                self.assertEqual(
                    [item["semantic_branch"] for item in evidence],
                    list(v3.BRANCH_ORDER),
                )
                self.assertTrue(
                    all("event_audit_receipt_digest" in item for item in evidence)
                )

    def test_action_requires_complete_transition_and_terminal_hold(self) -> None:
        rows, audits, prereg = _artifacts()
        target = next(
            index
            for index, audit in enumerate(audits)
            if audit["analysis_split"] == "confirmation"
            and audit["semantic_branch"] == "action"
        )
        old = audits[target]
        audits[target] = v3.seal_event_audit_receipt(
            candidate_id=old["candidate_id"],
            analysis_split=old["analysis_split"],
            action_family_id=old["action_family_id"],
            calibration_group_id=old["calibration_group_id"],
            actor_group_id=old["actor_group_id"],
            scene_group_id=old["scene_group_id"],
            action_group_id=old["action_group_id"],
            semantic_branch="action",
            generation_receipt_digest=old["generation_receipt_digest"],
            audit_source_kind=old["audit_source_kind"],
            external_audit_artifact_sha256=old["external_audit_artifact_sha256"],
            complete_target_transition_observed=True,
            terminal_hold_observed=False,
            full_target_action_observed=True,
            full_target_action_false_confirmed=False,
        )
        source = rows[target]
        rows[target] = v3.make_score_row(
            row_id=source["row_id"],
            candidate_id=source["candidate_id"],
            analysis_split=source["analysis_split"],
            action_family_id=source["action_family_id"],
            calibration_group_id=source["calibration_group_id"],
            actor_group_id=source["actor_group_id"],
            scene_group_id=source["scene_group_id"],
            action_group_id=source["action_group_id"],
            semantic_branch=source["semantic_branch"],
            raw_global_action_energy_score=source["raw_global_action_energy_score"],
            generation_receipt_digest=source["generation_receipt_digest"],
            frozen_scorer_receipt_digest=source["frozen_scorer_receipt_digest"],
            event_audit_receipt_digest=audits[target]["receipt_digest"],
        )
        receipt = _calibrate(rows, audits, prereg)
        self.assertFalse(receipt["optimizer_authorized"])
        self.assertTrue(any(reason.startswith("event_audit:") for reason in receipt["failure_reasons"]))

    def test_every_negative_must_be_full_target_action_false(self) -> None:
        rows, audits, prereg = _artifacts()
        target = next(
            index
            for index, audit in enumerate(audits)
            if audit["semantic_branch"] == "camera_only"
        )
        old = audits[target]
        audits[target] = v3.seal_event_audit_receipt(
            candidate_id=old["candidate_id"],
            analysis_split=old["analysis_split"],
            action_family_id=old["action_family_id"],
            calibration_group_id=old["calibration_group_id"],
            actor_group_id=old["actor_group_id"],
            scene_group_id=old["scene_group_id"],
            action_group_id=old["action_group_id"],
            semantic_branch=old["semantic_branch"],
            generation_receipt_digest=old["generation_receipt_digest"],
            audit_source_kind=old["audit_source_kind"],
            external_audit_artifact_sha256=old["external_audit_artifact_sha256"],
            complete_target_transition_observed=True,
            terminal_hold_observed=True,
            full_target_action_observed=True,
            full_target_action_false_confirmed=False,
        )
        source = rows[target]
        rows[target] = v3.make_score_row(
            row_id=source["row_id"],
            candidate_id=source["candidate_id"],
            analysis_split=source["analysis_split"],
            action_family_id=source["action_family_id"],
            calibration_group_id=source["calibration_group_id"],
            actor_group_id=source["actor_group_id"],
            scene_group_id=source["scene_group_id"],
            action_group_id=source["action_group_id"],
            semantic_branch=source["semantic_branch"],
            raw_global_action_energy_score=source["raw_global_action_energy_score"],
            generation_receipt_digest=source["generation_receipt_digest"],
            frozen_scorer_receipt_digest=source["frozen_scorer_receipt_digest"],
            event_audit_receipt_digest=audits[target]["receipt_digest"],
        )
        self.assertFalse(_calibrate(rows, audits, prereg)["optimizer_authorized"])

    def test_ambiguous_negative_is_rejected_not_silently_treated_as_false(self) -> None:
        rows, audits, prereg = _artifacts()
        target = next(
            index
            for index, audit in enumerate(audits)
            if audit["analysis_split"] == "confirmation"
            and audit["semantic_branch"] == "shuffle"
        )
        old = audits[target]
        audits[target] = v3.seal_event_audit_receipt(
            candidate_id=old["candidate_id"],
            analysis_split=old["analysis_split"],
            action_family_id=old["action_family_id"],
            calibration_group_id=old["calibration_group_id"],
            actor_group_id=old["actor_group_id"],
            scene_group_id=old["scene_group_id"],
            action_group_id=old["action_group_id"],
            semantic_branch=old["semantic_branch"],
            generation_receipt_digest=old["generation_receipt_digest"],
            audit_source_kind=old["audit_source_kind"],
            external_audit_artifact_sha256=old[
                "external_audit_artifact_sha256"
            ],
            complete_target_transition_observed=False,
            terminal_hold_observed=False,
            full_target_action_observed=False,
            full_target_action_false_confirmed=False,
        )
        source = rows[target]
        rows[target] = v3.make_score_row(
            row_id=source["row_id"],
            candidate_id=source["candidate_id"],
            analysis_split=source["analysis_split"],
            action_family_id=source["action_family_id"],
            calibration_group_id=source["calibration_group_id"],
            actor_group_id=source["actor_group_id"],
            scene_group_id=source["scene_group_id"],
            action_group_id=source["action_group_id"],
            semantic_branch=source["semantic_branch"],
            raw_global_action_energy_score=source[
                "raw_global_action_energy_score"
            ],
            generation_receipt_digest=source["generation_receipt_digest"],
            frozen_scorer_receipt_digest=source[
                "frozen_scorer_receipt_digest"
            ],
            event_audit_receipt_digest=audits[target]["receipt_digest"],
        )
        receipt = _calibrate(rows, audits, prereg)
        self.assertFalse(receipt["optimizer_authorized"])
        self.assertIn(
            f"event_audit:{old['candidate_id']}:shuffle",
            receipt["failure_reasons"],
        )

        with self.assertRaises(v3.PairV5EnergyCalibrationV3Error):
            v3.seal_event_audit_receipt(
                candidate_id="contradictory-negative",
                analysis_split="confirmation",
                action_family_id=FAMILIES[0],
                calibration_group_id="contradictory-cell",
                actor_group_id="contradictory-actor",
                scene_group_id="contradictory-scene",
                action_group_id="contradictory-action",
                semantic_branch="shuffle",
                generation_receipt_digest="c" * 64,
                audit_source_kind="manual_detached",
                external_audit_artifact_sha256="d" * 64,
                complete_target_transition_observed=True,
                terminal_hold_observed=True,
                full_target_action_observed=True,
                full_target_action_false_confirmed=True,
            )

    def test_group_leakage_and_missing_branch_fail_closed(self) -> None:
        rows, audits, prereg = _artifacts()
        leaked_rows = deepcopy(rows)
        leaked_audits = deepcopy(audits)
        fit_actor = next(row["actor_group_id"] for row in leaked_rows if row["analysis_split"] == "fit")
        target = next(index for index, row in enumerate(leaked_rows) if row["analysis_split"] == "confirmation")
        # Changing identity requires coherent re-sealing of both artifacts.
        audit = leaked_audits[target]
        leaked_audits[target] = v3.seal_event_audit_receipt(
            candidate_id=audit["candidate_id"], analysis_split=audit["analysis_split"],
            action_family_id=audit["action_family_id"], calibration_group_id=audit["calibration_group_id"],
            actor_group_id=fit_actor, scene_group_id=audit["scene_group_id"], action_group_id=audit["action_group_id"],
            semantic_branch=audit["semantic_branch"], generation_receipt_digest=audit["generation_receipt_digest"],
            audit_source_kind=audit["audit_source_kind"], external_audit_artifact_sha256=audit["external_audit_artifact_sha256"],
            complete_target_transition_observed=audit["complete_target_transition_observed"],
            terminal_hold_observed=audit["terminal_hold_observed"], full_target_action_observed=audit["full_target_action_observed"],
            full_target_action_false_confirmed=audit["full_target_action_false_confirmed"],
        )
        row = leaked_rows[target]
        leaked_rows[target] = v3.make_score_row(
            row_id=row["row_id"], candidate_id=row["candidate_id"], analysis_split=row["analysis_split"],
            action_family_id=row["action_family_id"], calibration_group_id=row["calibration_group_id"],
            actor_group_id=fit_actor, scene_group_id=row["scene_group_id"], action_group_id=row["action_group_id"],
            semantic_branch=row["semantic_branch"], raw_global_action_energy_score=row["raw_global_action_energy_score"],
            generation_receipt_digest=row["generation_receipt_digest"], frozen_scorer_receipt_digest=row["frozen_scorer_receipt_digest"],
            event_audit_receipt_digest=leaked_audits[target]["receipt_digest"],
        )
        receipt = _calibrate(leaked_rows, leaked_audits, prereg)
        self.assertFalse(receipt["optimizer_authorized"])
        self.assertIn("fit_confirmation_overlap:actor_group_id", receipt["failure_reasons"])

        rows, audits, prereg = _artifacts()
        removed = next(index for index, row in enumerate(rows) if row["semantic_branch"] == "shuffle")
        rows.pop(removed); audits.pop(removed)
        receipt = _calibrate(rows, audits, prereg)
        self.assertFalse(receipt["optimizer_authorized"])
        self.assertTrue(any(":shuffle" in reason for reason in receipt["failure_reasons"]))

    def test_row_schema_rejects_phase_alias_and_media_fields(self) -> None:
        rows, _, _ = _artifacts()
        for key, value in (
            ("raw_phase_conjunctive_score", 9.0),
            ("mp4_path", "/tmp/video.mp4"),
            ("source_video", "/tmp/source.mp4"),
        ):
            mutated = dict(rows[0])
            mutated[key] = value
            with self.subTest(key=key), self.assertRaises(v3.PairV5EnergyCalibrationV3Error):
                v3.validate_score_row(mutated)


if __name__ == "__main__":
    unittest.main()
