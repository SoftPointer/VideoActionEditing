import importlib.util
from html.parser import HTMLParser
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[3]
REVIEW_ROOT = (
    REPO
    / "md"
    / "action_editing"
    / "20260822_crosscase_target_graph_teacher_sam2_r3_review"
)
BUILDER = REVIEW_ROOT / "build_review.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("graph_teacher_r3_review", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _LocalReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        for attribute in ("src", "href"):
            value = values.get(attribute)
            if value and "://" not in value and not value.startswith("#"):
                self.paths.append(value)


class GraphTeacherR3ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_builder()
        cls.manifest, cls.summary = cls.builder.verify_bundle()
        cls.cases = {
            case["case_id"]: case for case in cls.summary["cases"]
        }

    def test_all_receipt_listed_artifacts_are_hash_verified(self):
        self.assertEqual(self.manifest["artifact_count"], 24)
        self.assertTrue(self.manifest["all_remote_receipt_hashes_verified"])
        self.assertTrue(
            all(
                item["verified_against_remote_receipt"]
                for item in self.manifest["artifacts"]
            )
        )

    def test_fail_closed_case_diagnostics(self):
        grill = self.cases["8b05aaf463db"]
        self.assertEqual(grill["visual_qa_verdict"], "partial_relative_trajectory")
        self.assertEqual(
            grill["node_phase_counts"]["basket_handle"]["target_observed_phases"],
            7,
        )
        self.assertAlmostEqual(
            grill["machine_diagnostics"]["return_fraction_from_peak_toward_baseline"],
            0.514,
            places=3,
        )

        phone = self.cases["40712e1341dc"]
        self.assertEqual(
            phone["machine_diagnostics"]["support_observation_mode"],
            "persistent_unknown",
        )
        self.assertEqual(phone["machine_diagnostics"]["support_observed_phases"], 0)
        self.assertEqual(
            phone["machine_diagnostics"]["support_edge_unresolved_phases"], 21
        )

        glasses = self.cases["5e83a9279951"]
        self.assertEqual(
            glasses["visual_qa_verdict"], "no_go_cross_phase_effector"
        )
        event_states = {
            event["event_id"]: event["status"]
            for event in glasses["post_run_event_qualification"]
        }
        self.assertEqual(event_states["pick_up_phone"], "unresolved_same_effector")
        self.assertEqual(event_states["terminal_phone_in_hand"], "unresolved_hand_edge")

        grill_event_states = {
            event["event_id"]: event["status"]
            for event in grill["post_run_event_qualification"]
        }
        self.assertEqual(
            grill_event_states["terminal_support_restored"],
            "unresolved_exact_restoration",
        )

    def test_global_claim_boundary(self):
        self.assertEqual(
            self.summary["overall_verdict"],
            "sealed_r3_complete_but_no_case_has_full_typed_interaction_graph",
        )
        self.assertEqual(
            self.summary["method_class"],
            "review_only_teacher_observation_scaffold_not_oceg",
        )
        for key in (
            "target_graph_authorized_for_generator",
            "target_graph_authorized_for_renderer",
            "target_graph_authorized_for_training",
            "target_graph_authorized_for_selection",
        ):
            self.assertFalse(self.summary[key])

    def test_built_page_exposes_claim_boundary_and_has_no_broken_local_links(self):
        page = (REVIEW_ROOT / "index.html").read_text(encoding="utf-8")
        for required in (
            "sealed r3",
            "Frozen-SAM2",
            "target→generator = forbidden",
            "Fail-closed machine diagnostics",
            "Post-run event qualification",
            "8b05aaf463db",
            "40712e1341dc",
            "5e83a9279951",
        ):
            self.assertIn(required, page)

        parser = _LocalReferences()
        parser.feed(page)
        missing = [
            value
            for value in parser.paths
            if not (REVIEW_ROOT / value).resolve().exists()
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
