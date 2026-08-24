from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import oceg_z0_admission_v1 as admission


FIXTURE = METHOD_ROOT / "assets" / "oceg_z0_synthetic_smoke_v1.json"
SCHEMA = METHOD_ROOT / "assets" / "oceg_z0_canonical_observations_schema_v1.json"
CONFIG = (
    METHOD_ROOT.parents[1]
    / "md"
    / "action_editing"
    / "20260822_object_centric_interaction_graph_reward"
    / "oceg_r1_experiment_config.json"
)


def fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def graph(value: dict) -> dict:
    return value["cases"][0]["graph_registry"]


class OCEGZ0AdmissionTests(unittest.TestCase):
    def test_synthetic_smoke_is_only_mechanically_admitted(self) -> None:
        value = fixture()
        receipt = admission.evaluate_bundle(value)

        self.assertEqual(receipt["status"], "MECHANICALLY_ADMITTED")
        self.assertTrue(receipt["summary"]["mechanical_admission_passed"])
        self.assertTrue(receipt["case_results"][0]["graph_registry"]["passed"])
        self.assertEqual(
            receipt["case_results"][0]["hard_gates"],
            {
                "identity": True,
                "contact": True,
                "terminal": True,
                "uncertainty": True,
                "passed": True,
            },
        )
        self.assertEqual(
            receipt["summary"]["hard_gate_axis_pass_case_counts"],
            {"identity": 1, "contact": 1, "terminal": 1, "uncertainty": 1},
        )
        self.assertFalse(receipt["claim_limits"]["scientific_claim_authorized"])
        self.assertFalse(receipt["claim_limits"]["renderer_effectiveness_claimed"])
        self.assertFalse(
            receipt["claim_limits"]["stable_transferable_action_representation_claimed"]
        )
        self.assertFalse(receipt["claim_limits"]["validator_ran_sam2"])
        self.assertFalse(receipt["claim_limits"]["validator_ran_cotracker"])

        frozen = receipt["frozen_base_registry"][0]
        self.assertFalse(frozen["graph_success_claimed"])
        self.assertFalse(frozen["used_as_graph_positive"])
        for row in frozen["records"]:
            self.assertIsNone(row["graph_success"])
            self.assertFalse(row["graph_observation_supplied"])
            self.assertFalse(row["used_as_graph_positive"])

        unsigned = dict(receipt)
        receipt_digest = unsigned.pop("receipt_sha256")
        self.assertEqual(receipt_digest, admission.object_sha256(unsigned))

    def test_minimal_schema_is_json_and_matches_runtime_contract(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["properties"]["schema_version"]["const"], admission.INPUT_SCHEMA
        )
        self.assertEqual(
            schema["properties"]["input_kind"]["const"], admission.INPUT_KIND
        )
        self.assertIn("graph_registry", schema["$defs"])
        self.assertIn("frozen_base", schema["$defs"])
        admission.validate_bundle(fixture())

    def test_synthetic_case_cannot_be_relabelled_as_development_or_confirmation(self) -> None:
        for profile in ("development_draft", "formal_confirmation_draft"):
            with self.subTest(profile=profile):
                value = fixture()
                value["profile"] = profile
                value["observer_execution"]["origin"] = "external_observer"
                with self.assertRaises(admission.OCEGZ0AdmissionError):
                    admission.evaluate_bundle(value)

    def test_any_uncertain_measure_fails_the_complete_bundle(self) -> None:
        value = fixture()
        measure = value["cases"][0]["target_observations"]["reverse"]["scores"][
            "event_order"
        ]
        measure["status"] = "uncertain"
        measure["value"] = None

        receipt = admission.evaluate_bundle(value)
        result = receipt["case_results"][0]
        self.assertEqual(receipt["status"], "REJECTED")
        self.assertTrue(result["any_uncertain"])
        self.assertFalse(result["target_controls"]["reverse"]["passed"])
        self.assertFalse(receipt["summary"]["global_gates"]["no_uncertainty_anywhere"])

    def test_target_forward_must_beat_reverse_shuffle_and_noop(self) -> None:
        value = fixture()
        value["cases"][0]["target_observations"]["forward"]["scores"][
            "event_order"
        ]["value"] = 0.20

        receipt = admission.evaluate_bundle(value)
        result = receipt["case_results"][0]
        self.assertEqual(receipt["status"], "REJECTED")
        self.assertFalse(result["target_controls"]["reverse"]["passed"])
        self.assertFalse(result["target_all_controls_passed"])

    def test_missing_multiappearance_pair_is_malformed_not_a_partial_pass(self) -> None:
        value = fixture()
        value["cases"][0]["multiappearance_consensus"].pop()
        with self.assertRaises(admission.OCEGZ0AdmissionError):
            admission.evaluate_bundle(value)

    def test_frozen_base_cannot_supply_or_claim_graph_success(self) -> None:
        for key, forged in (
            ("graph_observation_supplied", True),
            ("graph_success", True),
            ("used_as_graph_positive", True),
        ):
            with self.subTest(key=key):
                value = fixture()
                value["cases"][0]["frozen_base_records"][0][key] = forged
                with self.assertRaises(admission.OCEGZ0AdmissionError):
                    admission.evaluate_bundle(value)

    def test_instruction_introduced_node_is_not_source_matched_and_can_pass(self) -> None:
        value = fixture()
        registry = graph(value)
        registry["nodes"].append(
            {
                "node_id": "phone",
                "role": "moving_object",
                "ownership": "instruction_introduced",
                "introduction_authority": "instruction_self_anchor_generic_node",
                "source_node_id": None,
                "first_reliable_phase": 3,
                "preappearance_state": "unresolved_until_first_reliable_appearance",
                "source_identity_match_required": False,
                "postappearance_persistence": "pass",
                "morph_or_split_from_source": "not_detected",
                "source_noninterference": "not_applicable",
            }
        )
        registry["terminal_visibility"].append(
            {
                "node_id": "phone",
                "mode": "visible_at_terminal",
                "preexit_approach_contact_phase_count": None,
                "hand_release": "not_applicable",
                "trajectory_to_known_support_or_frame_boundary": "not_applicable",
                "bounded_support_state": "not_applicable",
            }
        )

        receipt = admission.evaluate_bundle(value)
        phone = next(
            row
            for row in receipt["case_results"][0]["graph_registry"]["nodes"]
            if row["node_id"] == "phone"
        )
        self.assertEqual(receipt["status"], "MECHANICALLY_ADMITTED")
        self.assertFalse(phone["source_identity_match_required"])
        self.assertEqual(phone["first_reliable_phase"], 3)

        value["cases"][0]["graph_registry"]["nodes"][-1][
            "introduction_authority"
        ] = "target_teacher"
        with self.assertRaises(admission.OCEGZ0AdmissionError):
            admission.evaluate_bundle(value)

    def test_instruction_object_morph_or_split_and_source_interference_fail(self) -> None:
        value = fixture()
        graph(value)["nodes"][1]["morph_or_split_from_source"] = "detected"
        receipt = admission.evaluate_bundle(value)
        self.assertEqual(receipt["status"], "REJECTED")
        self.assertFalse(receipt["case_results"][0]["hard_gates"]["identity"])

        value = fixture()
        graph(value)["nodes"][2]["source_noninterference"] = "fail"
        receipt = admission.evaluate_bundle(value)
        self.assertEqual(receipt["status"], "REJECTED")
        self.assertFalse(receipt["case_results"][0]["hard_gates"]["identity"])

    def test_offscreen_effector_has_no_fabricated_mask_identity_or_contact(self) -> None:
        value = fixture()
        effector = graph(value)["effector"]
        effector.update(
            {
                "mode": "exogenous_or_offscreen_effector",
                "observed_node_id": None,
                "mask_identity_claimed": False,
                "contact_truth_observed": False,
                "latent_action_evidence": {
                    "support_edge_off_on": "pass",
                    "relative_height_reversal": "pass",
                    "terminal_supported_hold": "pass",
                },
            }
        )
        receipt = admission.evaluate_bundle(value)
        result = receipt["case_results"][0]["graph_registry"]["effector"]
        self.assertEqual(receipt["status"], "MECHANICALLY_ADMITTED")
        self.assertFalse(result["mask_identity_claimed"])
        self.assertFalse(result["contact_truth_observed"])

        value = copy.deepcopy(value)
        graph(value)["effector"]["mask_identity_claimed"] = True
        with self.assertRaises(admission.OCEGZ0AdmissionError):
            admission.evaluate_bundle(value)

    def test_uncertain_offscreen_effector_evidence_fails_closed(self) -> None:
        value = fixture()
        effector = graph(value)["effector"]
        effector.update(
            {
                "mode": "exogenous_or_offscreen_effector",
                "observed_node_id": None,
                "mask_identity_claimed": False,
                "contact_truth_observed": False,
                "latent_action_evidence": {
                    "support_edge_off_on": "pass",
                    "relative_height_reversal": "uncertain",
                    "terminal_supported_hold": "pass",
                },
            }
        )
        receipt = admission.evaluate_bundle(value)
        self.assertEqual(receipt["status"], "REJECTED")
        self.assertTrue(receipt["case_results"][0]["any_uncertain"])

    def test_out_of_frame_terminal_requires_two_phases_release_and_bounded_exit(self) -> None:
        value = fixture()
        terminal = graph(value)["terminal_visibility"][0]
        terminal.update(
            {
                "mode": "out_of_frame_after_confirmed_support_release",
                "preexit_approach_contact_phase_count": 2,
                "hand_release": "pass",
                "trajectory_to_known_support_or_frame_boundary": "pass",
                "bounded_support_state": "pass",
            }
        )
        receipt = admission.evaluate_bundle(value)
        self.assertEqual(receipt["status"], "MECHANICALLY_ADMITTED")

        for key, failure_value in (
            ("preexit_approach_contact_phase_count", 1),
            ("hand_release", "fail"),
            ("trajectory_to_known_support_or_frame_boundary", "fail"),
            ("bounded_support_state", "fail"),
        ):
            with self.subTest(key=key):
                failed = copy.deepcopy(value)
                graph(failed)["terminal_visibility"][0][key] = failure_value
                failed_receipt = admission.evaluate_bundle(failed)
                self.assertEqual(failed_receipt["status"], "REJECTED")
                self.assertFalse(
                    failed_receipt["case_results"][0]["hard_gates"]["terminal"]
                )

    def test_unresolved_disappearance_is_uncertain_and_fails(self) -> None:
        value = fixture()
        graph(value)["terminal_visibility"][0].update(
            {
                "mode": "unresolved_disappearance",
                "preexit_approach_contact_phase_count": None,
                "hand_release": "uncertain",
                "trajectory_to_known_support_or_frame_boundary": "uncertain",
                "bounded_support_state": "uncertain",
            }
        )
        receipt = admission.evaluate_bundle(value)
        self.assertEqual(receipt["status"], "REJECTED")
        self.assertTrue(receipt["case_results"][0]["any_uncertain"])
        self.assertFalse(receipt["case_results"][0]["hard_gates"]["terminal"])

    def test_cli_writes_receipt_exclusively_and_will_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    admission.main(["--input", str(FIXTURE), "--output", str(output)]),
                    0,
                )
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "MECHANICALLY_ADMITTED")
            expected = json.loads(CONFIG.read_text(encoding="utf-8"))[
                "z0_offline_admission"
            ]["synthetic_expected_receipt"]
            self.assertEqual(receipt["input"]["sha256"], expected["input_file_sha256"])
            self.assertEqual(receipt["receipt_sha256"], expected["receipt_sha256"])
            self.assertEqual(receipt["status"], expected["status"])
            self.assertEqual(
                receipt["claim_limits"]["scientific_claim_authorized"],
                expected["scientific_claim_authorized"],
            )
            self.assertEqual(
                receipt["claim_limits"]["frozen_base_graph_success_claimed"],
                expected["frozen_base_graph_success_claimed"],
            )
            with self.assertRaises(admission.OCEGZ0AdmissionError):
                with redirect_stdout(io.StringIO()):
                    admission.main(
                        ["--input", str(FIXTURE), "--output", str(output)]
                    )


if __name__ == "__main__":
    unittest.main()
