from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_lora as trainer  # noqa: E402
from tools import materialize_reward_selected_synthetic_targets as materializer  # noqa: E402


class RewardSelectedDatasetContractTests(unittest.TestCase):
    def test_reward_result_schema_matches_frozen_ablation(self) -> None:
        self.assertEqual(
            materializer.REWARD_RESULT_SCHEMA,
            "action-editing-reward-ablation-result-v1",
        )

    def test_explicit_flag_is_required_and_valid_summary_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shards = root / "shards"
            shards.mkdir()
            files = []
            index_rows = []
            for number in range(4):
                iid = f"{number:016x}"
                shard = shards / f"{iid}.parquet"
                shard.write_bytes(f"shard-{number}".encode())
                files.append(shard.resolve())
                index_rows.append(
                    {
                        "schema_version": trainer.VAE_DATASET_INDEX_ROW_SCHEMA,
                        "iid": iid,
                        "parquet_path": str(shard),
                        "parquet_sha256": trainer.file_sha256(shard),
                    }
                )
            index = root / "dataset_index.jsonl"
            index.write_bytes(
                b"".join(trainer.canonical_json_bytes(row) + b"\n" for row in index_rows)
            )
            summary = {
                "schema_version": trainer.REWARD_SELECTED_DATASET_SUMMARY_SCHEMA,
                "complete": True,
                "preview_only": True,
                "training_authorized": False,
                "training_use_forbidden": True,
                "experimental_training_acknowledged": True,
                "production_claim_forbidden": True,
                "scientific_claim_authorized": False,
                "reward_selected_synthetic_target": True,
                "same_source_instruction_rows_across_arms": True,
                "arm": "action_only",
                "expected_sample_count": 4,
                "materialized_sample_count": 4,
                "missing_sample_count": 0,
                "frame_count": 81,
                "fps": 25.0,
                "latent_frame_count": 21,
                "bucket_counts": {"60x62-latent": 4},
                "shards_directory": str(shards),
                "index_path": str(index),
                "index_sha256": trainer.file_sha256(index),
            }
            summary["summary_digest"] = trainer.object_sha256(summary)
            summary_path = root / "dataset_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            class Dataset:
                def __init__(self):
                    self.root = shards.resolve()
                    self.files = tuple(files)

                @staticmethod
                def __len__():
                    return 4

            dataset = Dataset()

            with self.assertRaisesRegex(trainer.TrainingContractError, "explicit"):
                trainer.validate_preprocessed_dataset_summary(
                    summary_path, dataset, allow_incomplete=False
                )
            identity = trainer.validate_preprocessed_dataset_summary(
                summary_path,
                dataset,
                allow_incomplete=False,
                allow_reward_selected_synthetic_targets=True,
            )
            self.assertTrue(identity["reward_selected_synthetic_targets"])
            self.assertEqual(identity["arm"], "action_only")


if __name__ == "__main__":
    unittest.main()
