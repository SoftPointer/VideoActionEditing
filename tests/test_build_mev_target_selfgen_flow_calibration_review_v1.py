from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BUILDER_PATH = (
    REPO
    / "methods"
    / "bernini_action_editing"
    / "build_mev_target_selfgen_flow_calibration_review_v1.py"
)


def load_builder():
    spec = importlib.util.spec_from_file_location("mev_flow_calibration_review", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve the defining module through sys.modules.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class MevTargetSelfgenFlowCalibrationReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load_builder()

    def make_fixture(self, root: Path, *, case_ids: tuple[str, ...] = ("case01",)) -> tuple[Path, Path]:
        inputs = root / "inputs"
        inference = root / "inference"
        cases = []
        for case_id in case_ids:
            specs = {}
            for key in ("source", "real_forward", "self_generated", "frozen"):
                payload = f"{case_id}-{key}-video".encode()
                path = inputs / case_id / f"{key}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                specs[key] = {"path": str(path), "sha256": digest(payload)}

            for route in self.builder.ROUTES:
                route_dir = inference / case_id / route
                route_dir.mkdir(parents=True, exist_ok=True)
                output_payload = f"{case_id}-{route}-carrier-output".encode()
                output = route_dir / "output.mp4"
                output.write_bytes(output_payload)
                receipt = {
                    "schema_version": self.builder.EXPECTED_RECEIPT_SCHEMA,
                    "experiment_id": "fixture-experiment",
                    "case_id": case_id,
                    "route_kind": route,
                    "anchor_kind": route,
                    "historical_carrier_global_step": 32,
                    "current_experiment_optimization_steps": 0,
                    "parameter_updates_in_current_experiment": False,
                    "carrier_checkpoint": "/fixture/checkpoint-00000032",
                    "carrier_receipt_sha256": "1" * 64,
                    "flow_bundle": f"/fixture/{case_id}/{route}.safetensors",
                    "flow_bundle_sha256": "2" * 64,
                    "output": str(output.resolve()),
                    "output_sha256": digest(output_payload),
                    "information_firewall": {
                        "target_video_accessed_by_extractor": route != "self_generated",
                        "target_video_accessed_by_trainer": False,
                        "target_video_accessed_by_renderer": False,
                        "target_rgb_or_vae_target_used": False,
                        "anchor_role": "detached_dense_flow_representation_only",
                    },
                    "claim_boundary": self.builder.CLAIM_BOUNDARY,
                }
                (route_dir / "calibration_receipt.json").write_text(
                    json.dumps(receipt), encoding="utf-8"
                )

            cases.append(
                {
                    "case_id": case_id,
                    "pair_id": f"{case_id}-full-pair-id",
                    "split": "fit" if case_id == case_ids[0] else "heldout",
                    "action_family": f"family-{case_id}",
                    "seed": 2026082401,
                    "instruction": f"Edit the action for <{case_id}> & keep identity.",
                    **specs,
                }
            )

        manifest = {
            "schema_version": self.builder.EXPECTED_MANIFEST_SCHEMA,
            "experiment_id": "fixture-experiment",
            "frame_count": 81,
            "fps": 25,
            "current_experiment_optimization_steps": 0,
            "historical_carrier_global_step": 32,
            "flow_roles": list(self.builder.ROUTES),
            "cases": cases,
        }
        manifest_path = root / "experiment_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, inference

    def test_builds_fixed_eight_columns_and_synchronized_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, inference = self.make_fixture(root, case_ids=("case01", "case02"))
            output = root / "review"

            result = self.builder.build(manifest, inference, output)

            html = (output / "index.html").read_text(encoding="utf-8")
            review = json.loads((output / "review_manifest.json").read_text(encoding="utf-8"))
            readme = (output / "README.md").read_text(encoding="utf-8")
            self.assertEqual(result["case_count"], 2)
            self.assertEqual(review["case_count"], 2)
            self.assertEqual(
                [column["label"] for column in review["columns"]],
                [label for _, label in self.builder.COLUMNS],
            )
            for label in (
                "Source",
                "Real target reference",
                "Self-generated anchor",
                "Frozen",
                "real-forward carrier",
                "shuffle carrier",
                "reverse carrier",
                "selfgen carrier",
            ):
                self.assertIn(label, html)
            self.assertEqual(html.count('data-controls-scope="case01"'), 1)
            self.assertEqual(html.count('data-controls-scope="case02"'), 1)
            self.assertEqual(html.count('data-controls-scope="all"'), 1)
            self.assertIn('data-command="play" data-scope="case01"', html)
            self.assertIn('data-command="pause" data-scope="case01"', html)
            self.assertIn('data-command="restart" data-scope="case01"', html)
            self.assertIn("video.currentTime =", html)
            self.assertIn("video.playbackRate =", html)
            self.assertIn("window.setInterval", html)
            self.assertIn('controls muted loop playsinline preload="metadata"', html)
            self.assertNotIn(" autoplay", html)
            self.assertIn("NOT TRAINED “OURS” RESULTS", html)
            self.assertIn("CURRENT OPTIMIZATION STEPS = 0", html)
            self.assertIn("not a trained Ours result", readme)

            first = review["cases"][0]
            self.assertEqual(set(first["media"]), {key for key, _ in self.builder.COLUMNS})
            for role, media in first["media"].items():
                published = output / media["published_path"]
                self.assertTrue(published.is_file(), role)
                self.assertEqual(digest(published.read_bytes()), media["sha256"])
            self.assertEqual(set(first["receipts"]), set(self.builder.ROUTES))
            for receipt in first["receipts"].values():
                self.assertTrue((output / receipt["published_path"]).is_file())
                self.assertEqual(receipt["current_experiment_optimization_steps"], 0)
                self.assertEqual(receipt["historical_carrier_global_step"], 32)

    def test_refuses_nonempty_output_without_touching_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, inference = self.make_fixture(root)
            output = root / "review"
            output.mkdir()
            sentinel = output / "user-file.txt"
            sentinel.write_text("keep me", encoding="utf-8")

            with self.assertRaisesRegex(self.builder.BuildError, "refusing to overwrite"):
                self.builder.build(manifest, inference, output)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep me")
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_receipt_gates_route_and_step_provenance(self) -> None:
        mutations = (
            ("route_kind", "reverse", "route_kind"),
            ("current_experiment_optimization_steps", 1, "current_experiment_optimization_steps"),
            ("historical_carrier_global_step", 31, "historical_carrier_global_step"),
            ("parameter_updates_in_current_experiment", True, "parameter_updates"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, inference = self.make_fixture(root)
                receipt_path = inference / "case01" / "real_forward" / "calibration_receipt.json"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                receipt[field] = value
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

                with self.assertRaisesRegex(self.builder.BuildError, message):
                    self.builder.build(manifest, inference, root / "review")
                self.assertFalse((root / "review").exists())

    def test_missing_media_fails_without_publishing_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, inference = self.make_fixture(root)
            (inference / "case01" / "self_generated" / "output.mp4").unlink()

            with self.assertRaisesRegex(self.builder.BuildError, "does not resolve to a file"):
                self.builder.build(manifest, inference, root / "review")
            self.assertFalse((root / "review").exists())

    def test_symlink_mode_publishes_media_but_copies_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, inference = self.make_fixture(root)
            output = root / "review"

            self.builder.build(manifest, inference, output, copy_mode="symlink")

            self.assertTrue((output / "media" / "case01" / "source.mp4").is_symlink())
            self.assertFalse(
                (
                    output
                    / "receipts"
                    / "case01"
                    / "real_forward"
                    / "calibration_receipt.json"
                ).is_symlink()
            )


if __name__ == "__main__":
    unittest.main()
