#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_caper_native_kseed_population_v1 as population  # noqa: E402


class CaperNativeKSeedPopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry_path = METHOD_ROOT / population.CANONICAL_REGISTRY_RELATIVE
        cls.registry = json.loads(cls.registry_path.read_text(encoding="utf-8"))

    def test_registry_is_sealed_complete_cartesian_population(self) -> None:
        self.assertEqual(
            population._file_sha256(self.registry_path),
            population.CANONICAL_REGISTRY_SHA256,
        )
        population._validate_registry(self.registry)
        self.assertEqual(len(population.FIT_CELL_ORDER), 16)
        self.assertEqual(len(population.LOCKBOX_CELL_ORDER), 4)
        fit = self.registry["population_design"]["fit"]
        lockbox = self.registry["population_design"]["lockbox"]
        self.assertEqual(fit["cell_order"], list(population.FIT_CELL_ORDER))
        self.assertEqual(lockbox["cell_order"], list(population.LOCKBOX_CELL_ORDER))
        self.assertEqual(fit["seeds"], list(population.FIT_SEEDS))
        self.assertEqual(lockbox["seeds"], list(population.LOCKBOX_SEEDS))
        self.assertFalse(fit["seed_filtering_or_best_of_k_authorized"])
        self.assertFalse(lockbox["seed_filtering_or_best_of_k_authorized"])
        self.assertFalse(
            self.registry["population_design"]["posthoc_seed_selection_authorized"]
        )

    def test_all_cells_resolve_only_inside_the_requested_phase(self) -> None:
        fit_rows = [
            population._registry_cell(
                self.registry, cell_id=cell_id, population_phase="fit"
            )
            for cell_id in population.FIT_CELL_ORDER
        ]
        lockbox_rows = [
            population._registry_cell(
                self.registry, cell_id=cell_id, population_phase="lockbox"
            )
            for cell_id in population.LOCKBOX_CELL_ORDER
        ]
        self.assertEqual(
            {row["source_id"] for row in fit_rows}, set(population.FIT_SOURCE_IDS)
        )
        self.assertEqual(
            {row["seed"] for row in fit_rows}, set(population.FIT_SEEDS)
        )
        self.assertEqual(
            {row["source_id"] for row in lockbox_rows},
            set(population.LOCKBOX_SOURCE_IDS),
        )
        self.assertEqual(
            {row["seed"] for row in lockbox_rows}, set(population.LOCKBOX_SEEDS)
        )
        with self.assertRaises(population.CaperNativeKSeedPopulationError):
            population._registry_cell(
                self.registry,
                cell_id=population.LOCKBOX_CELL_ORDER[0],
                population_phase="fit",
            )

    def test_source_hash_geometry_and_three_phase_captions_are_pinned(self) -> None:
        expected = {
            "7b88a1ca1f804f41": (
                "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed",
                [480, 496], [1, 16, 21, 60, 62],
            ),
            "841b5e0080a1441d": (
                "5f354b6b0f5cf49bf14d57a359bad03e90263d1a3965a57b1b89ce1a707f492a",
                [480, 496], [1, 16, 21, 60, 62],
            ),
            "402059390cdb4f50": (
                "210732166ae851bae57135d55798d79ac24b83503b81fadf47e3285b182abdce",
                [576, 416], [1, 16, 21, 72, 52],
            ),
            "3be4072a63144b8f": (
                "792c506404fba2cc2e88239f52bbb2e88c16af9be9842ec6194234806147b28f",
                [480, 496], [1, 16, 21, 60, 62],
            ),
            "6a7ebea80ba64f18": (
                "2bcf9bc195167c5450ac36026bac0477104814a602bf0caf2a241d302deebcf7",
                [400, 592], [1, 16, 21, 50, 74],
            ),
            "ac87fea937864bd5": (
                "12f766286bfe1409c7ee3a7bff5a88e6c135c44f20a551919a7801a31dd10ebe",
                [480, 496], [1, 16, 21, 60, 62],
            ),
        }
        for row in self.registry["sources"]:
            sha, bucket, latent = expected[row["source_id"]]
            self.assertEqual(row["source_video_sha256"], sha)
            self.assertEqual(row["bucket_hw"], bucket)
            self.assertEqual(row["latent_shape"], latent)
            self.assertEqual(
                population._sha256_text(row["target_action_caption"]),
                row["target_action_caption_sha256"],
            )
            for phrase in (
                "From frames 0 through 20",
                "From frames 20 through 40",
                "From frames 40 through 80",
                "locked-off camera",
            ):
                self.assertIn(phrase, row["target_action_caption"])

    def test_registry_mutation_fails_closed(self) -> None:
        changed = copy.deepcopy(self.registry)
        changed["population_design"]["fit"]["cell_order"].pop()
        with self.assertRaises(population.CaperNativeKSeedPopulationError):
            population._registry_cell(
                changed, cell_id=population.FIT_CELL_ORDER[0], population_phase="fit"
            )
        changed = copy.deepcopy(self.registry)
        changed["sources"][0]["target_action_caption"] += " changed"
        with self.assertRaises(population.CaperNativeKSeedPopulationError):
            population._validate_registry(changed)
        changed = copy.deepcopy(self.registry)
        changed["native_v2v_contract"]["mask_count"] = 1
        with self.assertRaises(population.CaperNativeKSeedPopulationError):
            population._validate_registry(changed)

    def test_fit_cli_refuses_every_lockbox_authority_argument(self) -> None:
        common = {
            "expected_registry_sha256": population.CANONICAL_REGISTRY_SHA256,
            "runtime_source_archive_sha256": "1" * 64,
            "runtime_source_closure_sha256": "2" * 64,
            "launcher_source_sha256": "3" * 64,
            "expected_checkpoint_tree_sha256": population.CHECKPOINT_TREE_SHA256,
            "runtime_source_revision": "4" * 40,
            "expected_bernini_commit": population.BERNINI_OFFICIAL_COMMIT,
            "expected_veomni_commit": population.VEOMNI_TESTED_COMMIT,
            "population_phase": "fit",
            "cell_id": population.FIT_CELL_ORDER[0],
            "lockbox_second_stage_enabled": False,
            "threshold_freeze_receipt": None,
            "expected_threshold_freeze_sha256": None,
        }
        population._validate_cli(argparse.Namespace(**common))
        for key, value in (
            ("lockbox_second_stage_enabled", True),
            ("threshold_freeze_receipt", "/tmp/freeze.json"),
            ("expected_threshold_freeze_sha256", "5" * 64),
        ):
            changed = dict(common)
            changed[key] = value
            with self.assertRaises(population.CaperNativeKSeedPopulationError):
                population._validate_cli(argparse.Namespace(**changed))

    def test_lockbox_requires_valid_sealed_threshold_freeze(self) -> None:
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            path = root / "threshold-freeze.json"
            receipt = {
                "schema_version": population.THRESHOLD_FREEZE_SCHEMA_VERSION,
                "registry_sha256": population.CANONICAL_REGISTRY_SHA256,
                "action_family_id": population.ACTION_FAMILY_ID,
                "action_taxonomy_sha256": population.ACTION_TAXONOMY_SHA256,
                "runtime_source_revision": revision,
                "fit_population_complete": True,
                "fit_population_receipt_sha256": "1" * 64,
                "fit_population_cell_ids_sha256": population.FIT_CELL_IDS_SHA256,
                "reward_thresholds_sha256": "2" * 64,
                "evaluation_protocol_sha256": "3" * 64,
                "method_and_thresholds_frozen_before_lockbox": True,
                "lockbox_media_opened_before_freeze_is_false": True,
                "lockbox_generation_authorized": True,
            }
            receipt["receipt_digest"] = population._object_sha256(receipt)
            path.write_text(
                json.dumps(
                    receipt,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="ascii",
            )
            file_sha = population._file_sha256(path)
            summary = population._validate_threshold_freeze_receipt(
                path, expected_sha256=file_sha, runtime_source_revision=revision
            )
            self.assertEqual(summary["artifact_sha256"], file_sha)
            self.assertTrue(summary["method_and_thresholds_frozen_before_lockbox"])
            changed = json.loads(path.read_text(encoding="utf-8"))
            changed["lockbox_generation_authorized"] = False
            changed["receipt_digest"] = population._object_sha256(
                {key: value for key, value in changed.items() if key != "receipt_digest"}
            )
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(population.CaperNativeKSeedPopulationError):
                population._validate_threshold_freeze_receipt(
                    path,
                    expected_sha256=population._file_sha256(path),
                    runtime_source_revision=revision,
                )

    def test_a66e_is_posthoc_only_and_not_a_runnable_cell(self) -> None:
        negative = self.registry["posthoc_negative_control"]
        self.assertEqual(negative["source_id"], "a66e6818e4144928")
        self.assertEqual(negative["role"], "posthoc_wrong_family_audit_only")
        self.assertFalse(negative["generation_by_this_launcher_authorized"])
        self.assertFalse(negative["optimizer_materialization_authorized"])
        self.assertFalse(any("a66e" in cell for cell in population.CELL_ORDER))

    def test_actual_render_path_reuses_only_the_base_native_v2v_arm(self) -> None:
        source = Path(population.__file__).read_text(encoding="utf-8")
        self.assertIn("infer_t2v_v2v_branch_homotopy_canary as implementation", source)
        self.assertIn(
            "base.conditions_for_arm(NATIVE_ARM, source_latent=source_latent)",
            source,
        )
        self.assertIn("base.sampling_contract(NATIVE_ARM", source)
        self.assertIn('"guidance_mode": "v2v_apg"', source)
        self.assertNotIn('build_mode_native_prompt(\n        "pure-t2v"', source)
        self.assertNotIn("T2VV2VBranchHomotopyRuntimePatch(", source)
        self.assertNotIn("optimizer.step", source)


if __name__ == "__main__":
    unittest.main()
