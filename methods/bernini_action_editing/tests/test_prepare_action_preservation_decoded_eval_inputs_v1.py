from __future__ import annotations

import pathlib
import json
import io
import sys
import tempfile
import unittest
from contextlib import ExitStack
from contextlib import redirect_stdout
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import prepare_action_preservation_decoded_eval_inputs_v1 as prepare


class PostTrainingEvalInputPreparationTests(unittest.TestCase):
    def test_cli_exposes_status_but_no_p0_override(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = prepare.main(["authority-status"])
        self.assertEqual(status, 2)
        value = json.loads(output.getvalue())
        self.assertEqual(value["status"], "BLOCKED_MODEL_CONSUMPTION_AUTHORITY_P0")
        self.assertFalse(value["cli_override_available"])
        phase_b = prepare.build_parser().parse_args(["phase-b"])
        self.assertFalse(hasattr(phase_b, "allow_unsafe_model_authority"))

    def test_exact14_model_consumption_authority_is_an_unbypassable_no_go(self) -> None:
        self.assertIs(
            prepare.MODEL_CONSUMPTION_AUTHORITY_ENFORCED_BY_PRODUCTION,
            False,
        )
        with self.assertRaisesRegex(
            prepare.EvalInputPreparationError,
            "P0: exact14 does not enforce same-run physical authority",
        ):
            prepare.require_production_model_consumption_authority()

    def test_check_cannot_report_ready_when_other_authorities_validate(self) -> None:
        completion = {"path": "/fixture/TRAINING_COMPLETE.json", "sha256": "1" * 64}
        audit = {"path": "/fixture/training-audit.json", "sha256": "2" * 64}
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                prepare,
                "validate_training_authority",
                return_value={
                    "completion_file": completion,
                    "training_audit_file": audit,
                    "checkpoint_count": 32,
                    "checkpoint_paths": [
                        {"root": f"/fixture/checkpoint-{index}"}
                        for index in range(32)
                    ],
                },
            ))
            stack.enter_context(mock.patch.object(
                prepare,
                "validate_static_source_authority",
                return_value=[
                    {"iid": iid, "seed": 2026081801 + index}
                    for index, iid in enumerate(prepare.FITTED_IIDS)
                ],
            ))
            stack.enter_context(mock.patch.object(
                prepare,
                "validate_static_runtime",
                return_value={"model": {"manifest": {"sha256": "3" * 64}}},
            ))
            stack.enter_context(mock.patch.object(
                prepare,
                "validate_eval_release_artifacts",
                return_value={"manifest": {"sha256": "4" * 64}},
            ))
            with self.assertRaisesRegex(
                prepare.EvalInputPreparationError,
                "production decoder",
            ):
                prepare.check_all("0" * 64)

    def test_phase_a_no_go_creates_no_request_or_release_root(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        work = pathlib.Path(temporary.name).resolve()
        release = work / "fresh-exact14"
        with mock.patch.object(
            prepare,
            "validate_training_authority",
            return_value={
                "completion_file": {},
                "training_audit_file": {},
                "checkpoint_count": 32,
                "checkpoint_paths": [],
            },
        ), mock.patch.object(
            prepare, "validate_static_source_authority", return_value=[]
        ), mock.patch.object(
            prepare,
            "validate_static_runtime",
            return_value={"model": {"manifest": {}}},
        ), mock.patch.object(
            prepare,
            "validate_eval_release_artifacts",
            return_value={"manifest": {}},
        ):
            with self.assertRaisesRegex(
                prepare.EvalInputPreparationError,
                "P0: exact14",
            ):
                prepare.publish_deployment_request(
                    expected_completion_sha256="0" * 64,
                    work_root=work,
                    materialized_release_root=release,
                )
        self.assertFalse((work / "deployment-request.json").exists())
        self.assertFalse(release.exists())

    def test_action_windows_are_preregistered_acceptance_contracts(self) -> None:
        contracts = [prepare.action_review_contract(row) for row in prepare.SOURCE_ROWS]
        self.assertEqual([row["iid"] for row in prepare.SOURCE_ROWS], list(prepare.FITTED_IIDS))
        for contract in contracts:
            self.assertEqual(contract["expected_onset_frame_min"], 4)
            self.assertEqual(contract["expected_onset_frame_max"], 20)
            self.assertEqual(contract["terminal_hold_start_frame_min"], 65)
            self.assertEqual(contract["terminal_hold_end_frame"], 80)
            self.assertEqual(contract["full_video_frame_count"], 81)
            self.assertEqual((contract["fps_num"], contract["fps_den"]), (25, 1))
            unsigned = dict(contract)
            declared = unsigned.pop("contract_digest")
            self.assertEqual(declared, prepare.object_sha256(unsigned))


if __name__ == "__main__":
    unittest.main()
