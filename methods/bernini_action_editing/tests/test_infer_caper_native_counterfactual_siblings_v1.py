#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_caper_native_counterfactual_siblings_v1 as siblings  # noqa: E402


class CaperNativeCounterfactualSiblingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry_path = METHOD_ROOT / siblings.CANONICAL_REGISTRY_RELATIVE
        cls.registry = json.loads(cls.registry_path.read_text(encoding="utf-8"))
        cls.runner_source = Path(siblings.__file__).read_text(encoding="utf-8")

    def test_registry_is_sealed_public_fit_source_k4_by_four_arms(self) -> None:
        self.assertEqual(
            siblings._file_sha256(self.registry_path),
            siblings.CANONICAL_REGISTRY_SHA256,
        )
        source = siblings._validate_registry(self.registry)
        self.assertEqual(source["source_id"], "7b88a1ca1f804f41")
        self.assertEqual(len(siblings.CELL_ORDER), 4)
        self.assertEqual(len(siblings.ARM_ORDER), 4)
        population = self.registry["population_design"]
        self.assertEqual(population["k"], 4)
        self.assertEqual(population["expected_rollout_count"], 16)
        self.assertEqual(population["cell_order"], list(siblings.CELL_ORDER))
        self.assertEqual(population["seeds"], list(siblings.SEEDS))
        self.assertFalse(population["seed_filtering_or_best_of_k_authorized"])
        self.assertFalse(population["replacement_seed_authorized"])
        self.assertFalse(population["lockbox_source_or_media_access_authorized"])

    def test_all_four_cells_resolve_without_free_seed_or_arm_selection(self) -> None:
        rows = [
            siblings._registry_cell(self.registry, cell_id=cell_id)
            for cell_id in siblings.CELL_ORDER
        ]
        self.assertEqual({row["source_id"] for row in rows}, {siblings.SOURCE_ID})
        self.assertEqual({row["seed"] for row in rows}, set(siblings.SEEDS))
        self.assertTrue(all(row["arm_order"] == list(siblings.ARM_ORDER) for row in rows))
        with self.assertRaises(siblings.CaperNativeSiblingError):
            siblings._registry_cell(
                self.registry, cell_id="fit-7b88a1ca1f804f41-s2026081899"
            )

    def test_captions_are_complete_source_scene_and_hash_bound(self) -> None:
        source = self.registry["source"]
        captions = source["captions"]
        self.assertEqual(tuple(captions), siblings.ARM_ORDER)
        for arm in siblings.ARM_ORDER:
            caption = captions[arm]
            self.assertEqual(
                siblings._sha256_text(caption), source["caption_sha256"][arm]
            )
            for phrase in (
                "A locked-off camera shows the same single grey French bulldog",
                "From frames 0 through 20",
                "From frames 20 through 40",
                "From frames 40 through 80",
                "Preserve the source dog",
            ):
                self.assertIn(phrase, caption)
        phase = captions["phase-order-violation"]
        self.assertIn("remains in its source standing pose", phase)
        self.assertIn("first turns its head", phase)
        self.assertIn("rises back to a source-consistent standing pose", phase)
        semantics = self.registry["counterfactual_semantics"]
        self.assertFalse(semantics["literal_time_reversal_used"])
        self.assertTrue(semantics["all_arms_share_source_initial_precondition"])
        self.assertNotIn("reverse", siblings.ARM_ORDER)

    def test_registry_mutations_fail_closed(self) -> None:
        mutations = []
        changed = copy.deepcopy(self.registry)
        changed["arm_order"].reverse()
        mutations.append(changed)
        changed = copy.deepcopy(self.registry)
        changed["population_design"]["seeds"].pop()
        mutations.append(changed)
        changed = copy.deepcopy(self.registry)
        changed["native_v2v_contract"]["shared_high_sigma_prefix"] = True
        mutations.append(changed)
        changed = copy.deepcopy(self.registry)
        changed["source"]["captions"]["noop"] += " changed"
        mutations.append(changed)
        for value in mutations:
            with self.subTest(value=value):
                with self.assertRaises(siblings.CaperNativeSiblingError):
                    siblings._validate_registry(value)

    def test_cli_is_hash_revision_and_cell_bound(self) -> None:
        values = {
            "expected_registry_sha256": siblings.CANONICAL_REGISTRY_SHA256,
            "runtime_source_archive_sha256": "1" * 64,
            "runtime_source_closure_sha256": "2" * 64,
            "launcher_source_sha256": "3" * 64,
            "expected_checkpoint_tree_sha256": siblings.CHECKPOINT_TREE_SHA256,
            "runtime_source_revision": "4" * 40,
            "expected_bernini_commit": siblings.BERNINI_OFFICIAL_COMMIT,
            "expected_veomni_commit": siblings.VEOMNI_TESTED_COMMIT,
            "cell_id": siblings.CELL_ORDER[0],
        }
        siblings._validate_cli(argparse.Namespace(**values))
        changed = dict(values)
        changed["cell_id"] = "fit-7b88a1ca1f804f41-s1"
        with self.assertRaises(siblings.CaperNativeSiblingError):
            siblings._validate_cli(argparse.Namespace(**changed))
        changed = dict(values)
        changed["expected_registry_sha256"] = "f" * 64
        with self.assertRaises(siblings.CaperNativeSiblingError):
            siblings._validate_cli(argparse.Namespace(**changed))

    def test_runner_uses_four_independent_stock_v2v_trajectories(self) -> None:
        source = self.runner_source
        self.assertIn(
            'base.build_mode_native_prompt(\n            "source-mv2v"', source
        )
        self.assertIn(
            "base.conditions_for_arm(\n        NATIVE_ARM, source_latent=source_latent",
            source,
        )
        self.assertIn("common_sampling = base.sampling_contract(NATIVE_ARM", source)
        self.assertIn("for arm in ARM_ORDER:", source)
        self.assertIn("diffusion.sample(**kwargs)", source)
        self.assertIn("_sample_with_native_initial_noise_observer", source)
        self.assertIn('"denoiser_or_scheduler_field_hook_installed": False', source)
        self.assertIn('"initial_noise_observer_installed": True', source)
        self.assertIn('"initial_noise_observer_read_only": True', source)
        self.assertIn(
            '"initial_noise_observer_replaces_or_injects_noise": False', source
        )
        self.assertIn('"shared_high_sigma_prefix": False', source)
        self.assertIn('"shared_prefix_steps": 0', source)
        self.assertIn('"independent_complete_stock_trajectory_per_arm": True', source)
        self.assertNotIn("T2VV2VBranchHomotopyRuntimePatch(", source)
        self.assertNotIn("install_tri_branch", source)
        self.assertNotIn("clean_field_callback=", source)
        self.assertNotIn("optimizer.step", source)

    def test_read_only_live_schedule_audit_requires_completed_exact40(self) -> None:
        class BranchBase:
            NATIVE_UNIPC40_TIMESTEPS = tuple(range(40))
            NATIVE_UNIPC40_SIGMAS = tuple((40 - index) / 40.0 for index in range(40))

            @staticmethod
            def pinned_exact40_schedule_receipt() -> dict[str, str]:
                return {"digest": siblings.SCHEDULE_SHA256}

        class Base:
            branch_base = BranchBase

        base = Base()

        class Values:
            def __init__(self, values: list[float | int]) -> None:
                self.values = values

            def detach(self) -> "Values":
                return self

            def cpu(self) -> "Values":
                return self

            def tolist(self) -> list[float | int]:
                return list(self.values)

        class Scheduler:
            timesteps = Values(list(BranchBase.NATIVE_UNIPC40_TIMESTEPS))
            sigmas = Values([*BranchBase.NATIVE_UNIPC40_SIGMAS, 0.0])
            step_index = 40

        receipt = siblings._live_exact40_schedule_receipt(Scheduler(), base=base)
        self.assertEqual(receipt["pinned_schedule_sha256"], siblings.SCHEDULE_SHA256)
        self.assertEqual(receipt["completed_step_index"], 40)
        changed = Scheduler()
        changed.step_index = 39
        with self.assertRaises(siblings.CaperNativeSiblingError):
            siblings._live_exact40_schedule_receipt(changed, base=base)

    def test_each_arm_and_cell_have_sealed_media_receipts(self) -> None:
        source = self.runner_source
        for suffix in (
            ".official-initial-gaussian.safetensors",
            ".receipt.json",
        ):
            self.assertIn(suffix, source)
        self.assertIn("base.native._save_outputs(", source)
        self.assertIn("source.normalized-clean-latent.safetensors", source)
        self.assertIn('output_dir / "receipt.json"', source)
        self.assertIn('"all_four_sibling_arms_complete": True', source)
        self.assertIn('"same_official_gaussian_value_all_arms": True', source)
        self.assertIn('"training_performed": False', source)
        self.assertIn('"optimizer_created": False', source)
        self.assertIn('"parameter_update": False', source)


if __name__ == "__main__":
    unittest.main()
