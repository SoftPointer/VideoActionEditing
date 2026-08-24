from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "materialize_saic_t2v_cross_run_equivalence_v1.py"
)
SPEC = importlib.util.spec_from_file_location("saic_cross_run", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SaicCrossRunEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference"
        self.fresh = self.root / "fresh"
        self.records = self.root / "records.jsonl"
        self.summary = self.root / "summary.json"
        self.output = self.root / "equivalence.json"
        qwen_rows = []
        for index in range(MODULE.EXPECTED_COUNT):
            candidate_id = f"candidate-{index:02d}"
            candidate = {
                "candidate_id": candidate_id,
                "event_verified": False,
                "branch": "incomplete",
                "analysis_split": "fit" if index < 30 else "confirmation",
                "seed": index,
            }
            for run in (self.reference, self.fresh):
                attempt = run / candidate_id
                attempt.mkdir(parents=True)
                artifacts = {}
                for kind, name in (
                    ("mp4", "t2v.mp4"),
                    ("normalized_clean_latent", "latent.safetensors"),
                    ("official_initial_gaussian", "gaussian.safetensors"),
                ):
                    path = attempt / name
                    path.write_bytes(f"{candidate_id}:{kind}".encode("ascii"))
                    artifacts[kind] = {"path": str(path), "sha256": sha(path)}
                gaussian_tensor_digest = hashlib.sha256(
                    f"{candidate_id}:gaussian-value".encode("ascii")
                ).hexdigest()
                artifacts["official_initial_gaussian"].update({
                    "tensor_value_sha256": gaussian_tensor_digest,
                    "raw_value_sha256": gaussian_tensor_digest,
                    "all_rank_identity": {
                        "identity": {
                            "raw_storage_sha256": gaussian_tensor_digest
                        }
                    },
                })
                receipt = {
                    "candidate": candidate,
                    "artifacts": {
                        "mp4": {
                            **artifacts["mp4"],
                            "normalized_clean_latent": artifacts[
                                "normalized_clean_latent"
                            ],
                        },
                        "official_initial_gaussian": artifacts[
                            "official_initial_gaussian"
                        ],
                    },
                }
                (attempt / MODULE.GENERATION_RECEIPT).write_text(
                    json.dumps(receipt), encoding="ascii"
                )
            authority = {
                "human_review": False,
                "data_selection": False,
                "training": False,
                "optimizer": False,
                "scientific_claim": False,
            }
            unsigned = {
                "schema_version": MODULE.RECORD_SCHEMA,
                "candidate_id": candidate_id,
                "video_sha256": sha(self.reference / candidate_id / "t2v.mp4"),
                "authority": authority,
            }
            qwen_rows.append({
                **unsigned, "receipt_digest": MODULE.object_sha256(unsigned)
            })
        self.records.write_text(
            "".join(json.dumps(row) + "\n" for row in qwen_rows),
            encoding="ascii",
        )
        summary = {
            "schema_version": MODULE.SUMMARY_SCHEMA,
            "record_count": MODULE.EXPECTED_COUNT,
            "output_jsonl_sha256": sha(self.records),
            "authority": qwen_rows[0]["authority"],
            "receipt_digest": "a" * 64,
        }
        self.summary.write_text(json.dumps(summary), encoding="ascii")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def args(self) -> list[str]:
        return [
            "program",
            "--reference-attempts", str(self.reference),
            "--fresh-attempts", str(self.fresh),
            "--qwen-records", str(self.records),
            "--qwen-summary", str(self.summary),
            "--output", str(self.output),
        ]

    def test_exact_runs_bind_qwen_by_video_identity(self) -> None:
        with mock.patch.object(sys, "argv", self.args()):
            self.assertEqual(MODULE.main(), 0)
        receipt = json.loads(self.output.read_text(encoding="ascii"))
        self.assertTrue(receipt["candidate_specs_all_equal"])
        self.assertTrue(all(receipt["artifact_kinds_all_byte_equal"].values()))
        self.assertTrue(receipt["observation_reuse_by_bitwise_video_identity"])
        self.assertTrue(
            receipt["official_initial_gaussian_tensor_values_all_equal"]
        )
        self.assertEqual(receipt["authority"], MODULE.FALSE_AUTHORITY)

    def test_changed_fresh_video_fails_equivalence_without_rejection(self) -> None:
        changed = self.fresh / "candidate-00" / "t2v.mp4"
        changed.write_bytes(b"changed")
        receipt_path = changed.parent / MODULE.GENERATION_RECEIPT
        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
        receipt["artifacts"]["mp4"]["sha256"] = sha(changed)
        receipt_path.write_text(json.dumps(receipt), encoding="ascii")
        with mock.patch.object(sys, "argv", self.args()):
            self.assertEqual(MODULE.main(), 0)
        output = json.loads(self.output.read_text(encoding="ascii"))
        self.assertFalse(output["artifact_kinds_all_byte_equal"]["mp4"])
        self.assertFalse(output["observation_reuse_by_bitwise_video_identity"])


if __name__ == "__main__":
    unittest.main()
