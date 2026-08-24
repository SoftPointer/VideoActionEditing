import argparse
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_self_generated_action_quotient_v1 as trainer


SEED = 20260817
SHA = "a" * 64
REVISION = "b" * 40


def args() -> argparse.Namespace:
    return argparse.Namespace(
        arm="action_only",
        max_steps=160,
        method_source_archive_sha256=SHA,
        method_source_revision=REVISION,
        seed=SEED,
        slots=4,
        source_manifest_sha256="c" * 64,
    )


def cache_metadata(seed: int = SEED) -> dict:
    return {
        "seed": seed,
        "initialization_seed": seed,
        "teacher_cache_seed": seed,
    }


class SeedTwoContractTest(unittest.TestCase):
    def test_replication_seed_is_required_and_exact(self):
        seed_action = next(action for action in trainer.parser()._actions if action.dest == "seed")
        self.assertTrue(seed_action.required)
        trainer.require_replication_seed(SEED)
        with self.assertRaisesRegex(trainer.QuotientTrainingError, "20260817"):
            trainer.require_replication_seed(20260816)

    def test_legacy_missing_and_wrong_teacher_cache_seeds_are_rejected(self):
        with self.assertRaisesRegex(
            trainer.QuotientTrainingError, "legacy teacher cache seed 20260816"
        ):
            trainer.validate_teacher_cache_seed({"seed": 20260816}, SEED)
        with self.assertRaisesRegex(trainer.QuotientTrainingError, "lacks an explicit"):
            trainer.validate_teacher_cache_seed({"seed": SEED}, SEED)
        wrong = cache_metadata()
        wrong["teacher_cache_seed"] = SEED + 1
        with self.assertRaisesRegex(trainer.QuotientTrainingError, "differs"):
            trainer.validate_teacher_cache_seed(wrong, SEED)
        self.assertEqual(trainer.validate_teacher_cache_seed(cache_metadata(), SEED), SEED)

    def test_expected_cache_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.pt"
            cache.write_bytes(b"fresh-seed-two-cache")
            with self.assertRaisesRegex(
                trainer.QuotientTrainingError, "teacher cache SHA-256 differs"
            ):
                trainer.validate_file_sha(cache, "0" * 64, "teacher cache")

    def test_cache_payload_and_receipt_bind_seed_and_source_authority(self):
        namespace = args()
        manifest = {"manifest_digest": "d" * 64}
        payload = trainer.teacher_cache_payload(args=namespace, manifest=manifest, cells=[])
        receipt = trainer.teacher_cache_receipt(
            args=namespace, manifest=manifest, cell_count=0, cache_sha256="e" * 64
        )
        for value in (payload, receipt):
            self.assertEqual(value["initialization_seed"], SEED)
            self.assertEqual(value["teacher_cache_seed"], SEED)
            self.assertEqual(value["method_source_revision"], REVISION)
            self.assertEqual(value["method_source_archive_sha256"], SHA)
            self.assertEqual(value["source_manifest_sha256"], namespace.source_manifest_sha256)
        self.assertEqual(payload["seed"], SEED)
        self.assertEqual(receipt["seed"], SEED)
        self.assertEqual(receipt["cache_sha256"], "e" * 64)
        self.assertEqual(
            receipt["receipt_digest"],
            trainer.object_sha(
                {key: value for key, value in receipt.items() if key != "receipt_digest"}
            ),
        )

    def test_cache_cells_are_the_exact_seeded_cartesian_product(self):
        cells = [
            {
                "row_index": row_index,
                "slot": slot,
                "seed": trainer.legacy.step_seed(SEED, slot, row_index),
            }
            for row_index in range(4)
            for slot in range(2)
        ]
        cache = {**cache_metadata(), "slots": 2, "cells": cells}
        _, by_key = trainer.validate_teacher_cache_cells(
            cache, slots=2, expected_seed=SEED
        )
        self.assertEqual(
            set(by_key), {(row_index, slot) for row_index in range(4) for slot in range(2)}
        )
        cells[0]["seed"] += 1
        with self.assertRaisesRegex(trainer.QuotientTrainingError, "cell seed differs"):
            trainer.validate_teacher_cache_cells(cache, slots=2, expected_seed=SEED)

    def test_checkpoint_receipt_records_both_seeds_cache_and_new_archive(self):
        namespace = args()
        receipt = trainer.checkpoint_receipt(
            args=namespace,
            manifest={"manifest_digest": "d" * 64},
            step=10,
            loss=1.0,
            grad_norm=2.0,
            target_modules=("q_proj",),
            trainable_count=3,
            bernini_revision="1" * 40,
            veomni_revision="2" * 40,
            transformers_version="test",
            initial_digest="3" * 64,
            teacher_cache_seed=SEED,
            teacher_cache_sha256="e" * 64,
        )
        self.assertEqual(receipt["initialization_seed"], SEED)
        self.assertEqual(receipt["teacher_cache_seed"], SEED)
        self.assertEqual(receipt["teacher_cache_sha256"], "e" * 64)
        self.assertEqual(receipt["method_source_revision"], REVISION)
        self.assertEqual(receipt["method_source_archive_sha256"], SHA)
        self.assertEqual(receipt["source_manifest_sha256"], namespace.source_manifest_sha256)

    def test_runner_requires_cache_sha_only_in_train_mode_and_has_no_old_seed(self):
        runner = (
            METHOD_ROOT / "scripts/auh_run_self_generated_action_quotient_v1.sh"
        ).read_text()
        train_start = runner.index('if [[ "${mode}" == train ]]; then')
        train_end = runner.index("\nfi", train_start)
        cache_sha_use = runner.index("ACTION_QUOTIENT_EXPECTED_CACHE_SHA256")
        self.assertLess(train_start, cache_sha_use)
        self.assertLess(cache_sha_use, train_end)
        self.assertEqual(runner.count("ACTION_QUOTIENT_EXPECTED_CACHE_SHA256"), 1)
        self.assertIn('seed="${ACTION_QUOTIENT_SEED:?set ACTION_QUOTIENT_SEED}"', runner)
        self.assertIn('"${seed}" == 20260817', runner)
        self.assertNotIn("20260816", runner)
        self.assertIn("ACTION_QUOTIENT_SOURCE_MANIFEST_SHA256", runner)
        self.assertIn('--expected-cache-sha256 "${expected_cache_sha}"', runner)


if __name__ == "__main__":
    unittest.main()
