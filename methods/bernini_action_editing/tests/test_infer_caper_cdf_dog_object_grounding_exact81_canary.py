#!/usr/bin/env python3

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

import infer_caper_cdf_dog_object_grounding_exact81_canary as canary  # noqa: E402


class CaperCDFDogObjectGroundingCanaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry_path = (
            METHOD_ROOT / canary.CANONICAL_REGISTRY_RELATIVE
        )
        cls.registry = json.loads(cls.registry_path.read_text(encoding="utf-8"))

    def test_registry_is_sealed_same_source_two_seed_preregistration(self) -> None:
        self.assertEqual(
            canary.base.native.legacy.file_sha256(self.registry_path),
            canary.CANONICAL_REGISTRY_SHA256,
        )
        rows = [
            canary._registry_cell(self.registry, cell_id=cell_id)
            for cell_id in canary.CELL_ORDER
        ]
        self.assertEqual([row["seed"] for row in rows], [2027, 2026081701])
        self.assertEqual(
            {row["source_video"] for row in rows}, {canary.SOURCE_VIDEO}
        )
        self.assertEqual(
            {row["source_video_sha256"] for row in rows},
            {canary.SOURCE_VIDEO_SHA256},
        )
        self.assertEqual(
            {row["target_action_caption"] for row in rows},
            {canary.TARGET_ACTION_CAPTION},
        )
        self.assertEqual(
            [row["bucket_hw"] for row in rows], [[496, 480], [496, 480]]
        )
        self.assertEqual(
            [row["latent_shape"] for row in rows],
            [[1, 16, 21, 62, 60], [1, 16, 21, 62, 60]],
        )

    def test_registry_mutation_fails_closed(self) -> None:
        changes = (
            ("cells", 0, "seed", 2028),
            ("cells", 1, "source_video_sha256", "0" * 64),
        )
        for root, index, key, value in changes:
            changed = copy.deepcopy(self.registry)
            changed[root][index][key] = value
            with self.assertRaises(canary.CaperCDFDogObjectGroundingCanaryError):
                canary._registry_cell(changed, cell_id=canary.CELL_ORDER[index])
        changed = copy.deepcopy(self.registry)
        changed["contract"]["homotopy"]["low_sigma"] = 0.74
        with self.assertRaises(canary.CaperCDFDogObjectGroundingCanaryError):
            canary._registry_cell(changed, cell_id=canary.CELL_ORDER[0])
        changed = copy.deepcopy(self.registry)
        changed["object_grounding_evaluation_contract"][
            "ordered_object_event_gate"
        ]["required_stage_order"] = ["approach", "grip", "lift", "hold"]
        with self.assertRaises(canary.CaperCDFDogObjectGroundingCanaryError):
            canary._registry_cell(changed, cell_id=canary.CELL_ORDER[0])

    def test_object_event_and_source_correspondence_gates_are_explicit(self) -> None:
        evaluation = canary._validate_object_grounding_contract(self.registry)
        event = evaluation["ordered_object_event_gate"]
        correspondence = evaluation["source_correspondence_gate"]
        self.assertEqual(
            event["required_stage_order"],
            ["approach", "contact", "grip", "lift", "hold"],
        )
        self.assertEqual(
            correspondence["required_correspondences"],
            [
                "source_dog_identity",
                "source_bone_identity",
                "source_dog_mouth_anatomical_identity",
            ],
        )
        self.assertFalse(event["automatic_adjudication_performed"])
        self.assertFalse(event["automatic_success_claim_authorized"])
        self.assertFalse(correspondence["automatic_success_claim_authorized"])
        self.assertFalse(evaluation["single_example_success_claim_authorized"])

    def test_same_source_cells_are_not_independent_identities(self) -> None:
        population = self.registry["population_design"]
        self.assertTrue(
            population[
                "same_source_cells_are_seed_replicates_not_independent_identities"
            ]
        )
        self.assertFalse(
            population["aggregate_as_independent_identities_authorized"]
        )
        self.assertFalse(population["single_example_conclusion_authorized"])
        self.assertFalse(
            population["seed_preregistration"][
                "seed_search_or_posthoc_selection_authorized"
            ]
        )

    def test_base_specialization_is_scoped_and_restorable(self) -> None:
        original = {
            "METHOD": canary.base.METHOD,
            "CELL_ORDER": canary.base.CELL_ORDER,
            "_registry_cell": canary.base._registry_cell,
        }
        previous = canary._specialize_base()
        try:
            self.assertEqual(canary.base.METHOD, canary.METHOD)
            self.assertEqual(canary.base.CELL_ORDER, canary.CELL_ORDER)
            self.assertIs(canary.base._registry_cell, canary._registry_cell)
        finally:
            canary._restore_base(previous)
        self.assertEqual(canary.base.METHOD, original["METHOD"])
        self.assertEqual(canary.base.CELL_ORDER, original["CELL_ORDER"])
        self.assertIs(canary.base._registry_cell, original["_registry_cell"])

    def test_rank0_receipt_augmentation_records_contract_not_outcome(self) -> None:
        cell = canary._registry_cell(
            self.registry, cell_id=canary.CELL_ORDER[0]
        )
        with tempfile.TemporaryDirectory() as directory:
            # Match ``main``: the receipt trust boundary receives a canonical
            # output directory.  On macOS ``/var`` is a symlink to
            # ``/private/var`` even though the receipt itself is not a symlink.
            output = Path(directory).resolve(strict=True)
            receipt = {
                "schema_version": canary.SCHEMA_VERSION,
                "method": canary.METHOD,
                "cell_id": cell["cell_id"],
                "input": {
                    "source_video_sha256": canary.SOURCE_VIDEO_SHA256,
                    "target_action_caption": canary.TARGET_ACTION_CAPTION,
                },
                "sampling": {"seed": cell["seed"]},
                "training_performed": False,
                "optimizer_created": False,
                "parameter_update": False,
            }
            receipt["receipt_digest"] = canary.base._object_sha256(receipt)
            (output / "receipt.json").write_bytes(
                canary.base.native.legacy.canonical_json_bytes(receipt) + b"\n"
            )
            canary._augment_rank0_receipt(output_dir=output, cell=cell)
            observed = json.loads(
                (output / "receipt.json").read_text(encoding="utf-8")
            )
            unsigned = dict(observed)
            declared = unsigned.pop("receipt_digest")
            self.assertEqual(canary.base._object_sha256(unsigned), declared)
            gate = observed["object_grounding_evaluation"]
            self.assertEqual(gate["gate_status"], "not_automatically_adjudicated")
            self.assertFalse(gate["observed_outcome_recorded"])
            self.assertFalse(gate["automatic_success_claim"])
            self.assertFalse(
                observed[
                    "same_source_replication_identity_aggregation_authorized"
                ]
            )
            self.assertFalse(observed["single_example_conclusion_authorized"])

    def test_runner_reuses_committed_frozen_exact81_implementation(self) -> None:
        source = Path(canary.__file__).read_text(encoding="utf-8")
        base_source = Path(canary.base.__file__).read_text(encoding="utf-8")
        self.assertIn("result = base.main(argv_list)", source)
        self.assertIn("infer_t2v_v2v_branch_homotopy_canary as base", source)
        self.assertIn("sampler_contract.validate_runtime_source_identity", base_source)
        self.assertIn("freeze_after != freeze_before", base_source)
        self.assertIn('"training_performed": False', base_source)
        self.assertIn('"optimizer_created": False', base_source)
        self.assertIn('"parameter_update": False', base_source)
        self.assertNotIn("optimizer.step", base_source)


if __name__ == "__main__":
    unittest.main()
