from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import prospective_factorial_branch_manifest_v1 as author  # noqa: E402
import run_prospective_factorial_branch_shard_v1 as runner  # noqa: E402


REGISTRY = METHOD_ROOT / "assets" / "factorial_source_population_scout_v1.json"
DESCRIPTORS = METHOD_ROOT / "assets" / "factorial_branch_prompt_descriptors_v1.json"
MANIFEST = METHOD_ROOT / "assets" / "prospective_factorial_branch_manifest_v1.json"


class ProspectiveFactorialBranchManifestTests(unittest.TestCase):
    def test_author_closes_balanced_two_seed_population(self) -> None:
        manifest = author.author(REGISTRY, DESCRIPTORS)

        rows = author.validate_manifest(manifest)
        self.assertEqual(len(rows), 252)
        self.assertEqual(
            {split: sum(row["analysis_split"] == split for row in rows) for split in author.SPLITS},
            {"fit": 84, "calibration": 84, "confirmation": 84},
        )
        self.assertFalse(manifest["authority"]["optimizer_step_authorized"])
        self.assertFalse(manifest["inference_contract"]["old_synthetic_target_accessed"])

    def test_noop_is_copy_and_other_branches_are_frozen_base(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        rows = author.validate_manifest(manifest)

        for row in rows:
            expected = "exact_source_copy" if row["branch"] == "noop" else "frozen_bernini"
            self.assertEqual(row["executor"], expected)

    def test_camera_prompt_has_no_locked_camera_contradiction(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        camera_rows = [row for row in manifest["entries"] if row["branch"] == "camera_only"]

        self.assertEqual(len(camera_rows), 36)
        self.assertTrue(all("locked camera" not in row["instruction"] for row in camera_rows))

    def test_descriptor_registry_binding_is_fail_closed(self) -> None:
        descriptors = json.loads(DESCRIPTORS.read_text(encoding="utf-8"))
        descriptors["source_registry_digest"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "descriptors.json"
            path.write_text(json.dumps(descriptors), encoding="utf-8")
            with self.assertRaisesRegex(author.FactorialBranchManifestError, "registry digest"):
                author.author(REGISTRY, path)

    def test_manifest_digest_detects_prompt_mutation(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        mutated = copy.deepcopy(manifest)
        mutated["entries"][0]["instruction"] += " mutation"

        with self.assertRaisesRegex(author.FactorialBranchManifestError, "manifest digest"):
            author.validate_manifest(mutated)

    def test_runtime_releases_only_complete_fit_cells(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cells = runner._cells(
            ["0b2fc177202e4d08:2026082511", "1367d5595ed641ae:2026082601"]
        )

        rows = runner._released_entries(manifest, cells, "fit")

        self.assertEqual(len(rows), 14)
        for cell in cells:
            self.assertEqual(
                {
                    row["branch"]
                    for row in rows
                    if (row["source_id"], row["seed"]) == cell
                },
                set(author.BRANCHES),
            )

    def test_runtime_refuses_calibration_release(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

        with self.assertRaisesRegex(runner.FactorialBranchRunError, "fit only"):
            runner._released_entries(
                manifest, [("059ab2ce4dec4b71", 2026082611)], "calibration"
            )


if __name__ == "__main__":
    unittest.main()
