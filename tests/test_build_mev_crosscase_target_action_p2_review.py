from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BUILDER_PATH = (
    REPO
    / "md"
    / "action_editing"
    / "20260822_mev_crosscase_target_action_p2_review"
    / "build_review.py"
)


def load_builder():
    spec = importlib.util.spec_from_file_location("crosscase_review_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CrosscaseReviewBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load_builder()

    def test_missing_slots_render_placeholders_and_metrics_skeleton(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            input_root = base / "pulled"
            output_dir = base / "review"

            self.builder.build(input_root, None, output_dir)

            html = (output_dir / "index.html").read_text(encoding="utf-8")
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

            self.assertIn("Real target 只作为人工 review reference", html)
            self.assertIn("重新运行 builder 后会自动替换此占位", html)
            self.assertEqual(manifest["counts"]["candidate_slots"], 18)
            self.assertEqual(manifest["counts"]["reference_slots"], 6)
            self.assertEqual(manifest["counts"]["present_media"], 0)
            self.assertEqual(manifest["counts"]["missing_media"], 24)
            self.assertEqual(len(metrics["reviews"]), 18)
            self.assertEqual(
                set(metrics["reviews"][0]["fields"]),
                {"action_coarse", "object_identity", "contact_terminal", "quality"},
            )
            self.assertTrue(all(value is None for value in metrics["reviews"][0]["fields"].values()))

    def test_present_media_get_video_attributes_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            input_root = base / "pulled"
            reference_root = base / "references"
            output_dir = base / "review"
            case_id = "8b05aaf463db"

            candidate = input_root / case_id / "seed2028" / "p2.mp4"
            source = reference_root / case_id / "source.mp4"
            target = reference_root / case_id / "real_target.mp4"
            for path, payload in (
                (candidate, b"candidate-video"),
                (source, b"source-video"),
                (target, b"target-video"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)

            self.builder.build(input_root, reference_root, output_dir)

            html = (output_dir / "index.html").read_text(encoding="utf-8")
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            private_mapping = json.loads(
                (output_dir / "private" / "arm_mapping.json").read_text(encoding="utf-8")
            )
            first_case = manifest["cases"][0]
            p2_review_id = next(
                row["review_id"]
                for row in private_mapping["rows"]
                if row["case_id"] == case_id
                and row["seed"] == 2028
                and row["arm"] == "p2"
            )
            p2 = next(
                candidate
                for seed in first_case["seeds"]
                if seed["seed"] == 2028
                for candidate in seed["candidates"]
                if candidate["review_id"] == p2_review_id
            )

            self.assertIn("controls loop muted playsinline", html)
            self.assertNotIn("/seed2028/p2.mp4", html)
            self.assertEqual(manifest["counts"]["present_media"], 3)
            self.assertTrue(first_case["references"]["source"]["present"])
            self.assertTrue(first_case["references"]["real_target"]["present"])
            self.assertTrue(p2["media"]["present"])
            self.assertEqual(len(p2["media"]["sha256"]), 64)
            self.assertIn("candidate_", p2["media"]["local_src"])
            self.assertNotIn("arm", p2)

    def test_contract_is_axis_exact_and_public_packet_is_arm_blind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output_dir = base / "review"
            self.builder.build(base / "pulled", None, output_dir)

            html = (output_dir / "index.html").read_text(encoding="utf-8")
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            contract = json.loads((output_dir / "review_contract.json").read_text(encoding="utf-8"))

            self.assertFalse(manifest["review_contract"]["arm_labels_visible"])
            self.assertFalse(metrics["review_mode"]["arm_labels_visible"])
            self.assertTrue(all("arm" not in row for row in metrics["reviews"]))
            self.assertIn("Candidate A", html)
            self.assertNotIn("/seed2028/p0a.mp4", html)
            self.assertNotIn("/seed2028/p1.mp4", html)
            self.assertNotIn("/seed2028/p2.mp4", html)
            self.assertEqual(len(contract["cases"]), 3)
            self.assertEqual(
                set(contract["cases"][0]["criteria"]),
                {"action_coarse", "object_identity", "contact_terminal", "quality"},
            )
            self.assertIn(
                "crosscase_p2_signal",
                {row["claim"] for row in contract["unblinded_p2_decision_table"]},
            )

    def test_compatible_scored_metrics_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            input_root = base / "pulled"
            output_dir = base / "review"

            self.builder.build(input_root, None, output_dir)
            metrics_path = output_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["reviews"][0]["fields"]["action_coarse"] = "pass"
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

            self.builder.build(input_root, None, output_dir)
            rebuilt = json.loads(metrics_path.read_text(encoding="utf-8"))
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(rebuilt["reviews"][0]["fields"]["action_coarse"], "pass")
            self.assertTrue(
                manifest["outputs"]["metrics_skeleton"]["preserved_existing_compatible_file"]
            )


if __name__ == "__main__":
    unittest.main()
