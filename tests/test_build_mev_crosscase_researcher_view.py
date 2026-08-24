from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BUILDER_PATH = (
    REPO / "md" / "action_editing" / "20260822_mev_crosscase_target_action_p2_review"
    / "build_researcher_view.py"
)


def load_builder():
    spec = importlib.util.spec_from_file_location("crosscase_researcher_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def receipt(case_id: str, seed: int, video_hashes: dict[str, str]) -> dict:
    replay_hash = f"replay-{case_id}-{seed}"
    return {
        "case_id": case_id,
        "receipt_digest": f"digest-{case_id}-{seed}",
        "freeze_certificate": {
            "base_frozen": True,
            "lora_module_count": 0,
            "trainable_parameter_elements": 0,
            "trainable_parameter_tensors": 0,
        },
        "interpretation": {
            "training_performed": False,
            "target_media_or_action_json_read_by_generator": False,
        },
        "input": {
            "target_video": False,
            "target_action_json": False,
            "target_rgb_mask_box_xy_flow_feature_embedding_latent_qkv_gaussian": False,
        },
        "sampling": {"p0a": {"seed": seed}},
        "outputs": {arm: {"sha256": digest} for arm, digest in video_hashes.items()},
        "generated_identities": {
            "p0a": {"identity": {"raw_storage_sha256": replay_hash}},
            "p0b": {"identity": {"raw_storage_sha256": replay_hash}},
        },
        "paired_same_process_contract": {
            "target_media_or_action_json_read": False,
            "p0_replay": {
                "generated_latent_bit_exact": True,
                "positive_tokens_and_embedding_bit_exact": True,
            },
        },
    }


class CrosscaseResearcherViewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load_builder()

    def materialize_packet(self, root: Path) -> tuple[Path, Path]:
        input_root = root / "runs"
        reference_root = root / "references"
        for case in self.builder.CASES:
            case_id = case["case_id"]
            for filename in ("source.mp4", "real_target.mp4"):
                path = reference_root / case_id / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{case_id}-{filename}".encode())
            for seed in self.builder.SEEDS:
                run = input_root / case_id / f"seed{seed}"
                run.mkdir(parents=True, exist_ok=True)
                hashes = {}
                for arm, _, _ in self.builder.ARMS:
                    path = run / f"{arm}.mp4"
                    path.write_bytes(f"{case_id}-{seed}-{arm}".encode())
                    hashes[arm] = self.builder.sha256_file(path)
                (run / "receipt.json").write_text(
                    json.dumps(receipt(case_id, seed, hashes)), encoding="utf-8"
                )
        return input_root, reference_root

    def test_explicit_columns_and_receipt_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root, reference_root = self.materialize_packet(root)
            output = root / "researcher_view.html"
            manifest_path = root / "researcher_view_manifest.json"
            self.builder.build(input_root, reference_root, output, manifest_path)

            page = output.read_text(encoding="utf-8")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("Frozen Base P0", page)
            self.assertIn("Event-order P1", page)
            self.assertIn("Relation P2", page)
            self.assertIn("Real Target (review-only)", page)
            self.assertIn("base_frozen=true", page)
            self.assertIn("LoRA=0", page)
            self.assertIn("trainable=0", page)
            self.assertIn("target read=false", page)
            self.assertIn("P0 replay exact=true", page)
            self.assertEqual(manifest["counts"]["generated_videos"], 18)
            self.assertTrue(all(manifest["aggregate_audit"].values()))

    def test_fails_closed_on_non_frozen_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root, reference_root = self.materialize_packet(root)
            bad = input_root / self.builder.CASES[0]["case_id"] / "seed2028" / "receipt.json"
            value = json.loads(bad.read_text(encoding="utf-8"))
            value["freeze_certificate"]["base_frozen"] = False
            bad.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "receipt audit failed"):
                self.builder.build(
                    input_root, reference_root, root / "page.html", root / "manifest.json"
                )


if __name__ == "__main__":
    unittest.main()
