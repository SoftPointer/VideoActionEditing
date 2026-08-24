from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
if str(METHOD_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT / "tools"))

import render_source_owned_role_locator_v15b_review as review  # noqa: E402


ARTIFACT_ROOT = (
    REPO_ROOT
    / "md/action_editing/20260815_reward/action_quotient_140846"
    / "v15b_e00_source_role_observer_review"
)
SOURCE = (
    REPO_ROOT
    / "md/action_editing/20260815_reward/action_quotient_140846"
    / "v14r3d2_gradgeom_decode_review/media/e00-source.mp4"
)
RECEIPT = ARTIFACT_ROOT / "provenance/e00_v15b_probe_receipt.json"
TENSORS = ARTIFACT_ROOT / "provenance/e00_v15b_affinity.safetensors"
R6_RECEIPT = (
    ARTIFACT_ROOT
    / "r6_null64/provenance/e00_v15b_r6_probe_receipt.json"
)
R6_TENSORS = (
    ARTIFACT_ROOT
    / "r6_null64/provenance/e00_v15b_r6_affinity.safetensors"
)


class FrozenV15BReviewBuilderTests(unittest.TestCase):
    def test_authenticated_builder_reproduces_exact_150_image_five_column_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review"
            result = review.render(
                argparse.Namespace(
                    receipt=RECEIPT,
                    tensors=TENSORS,
                    source=SOURCE,
                    output=output,
                )
            )
            self.assertEqual(result["image_count"], 150)
            self.assertEqual(len(tuple((output / "media").glob("*.jpg"))), 150)
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["roi_usage"], "human_audit_not_algorithm")
            self.assertEqual(
                metrics["render_contract"],
                {
                    "grid_columns_max": 5,
                    "grid_count": 30,
                    "images_per_grid": 5,
                    "rendered_image_count": 150,
                },
            )
            page = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn("grid-template-columns:repeat(5", page)
            self.assertEqual(page.count("<div class='grid'>"), 30)
            self.assertIn("human_audit_not_algorithm", page)

    def test_authenticated_r6_null64_builder_is_five_column_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review"
            result = review.render(
                argparse.Namespace(
                    receipt=R6_RECEIPT,
                    tensors=R6_TENSORS,
                    source=SOURCE,
                    output=output,
                )
            )
            self.assertEqual(result["image_count"], 150)
            self.assertEqual(len(tuple((output / "media").glob("*.jpg"))), 150)
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["review_profile"], review.R6_PROFILE)
            self.assertEqual(metrics["null_bank_span_count"], 64)
            self.assertEqual(metrics["roi_usage"], "human_audit_not_algorithm")
            self.assertEqual(
                metrics["strict_aggregate_mask_basis"],
                "role_specific_block_weights_from_null_percentile_times_temporal_coherence",
            )
            self.assertFalse(metrics["route_authorized"])
            self.assertEqual(
                metrics["block_role_summaries"]["aggregate"]["old_actor"][
                    "strict_mask_pixels"
                ],
                0,
            )
            page = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn("NO-GO for routing/training", page)
            self.assertIn("64 preregistered", page)
            self.assertIn("human_audit_not_algorithm", page)
            self.assertEqual(page.count("<div class='grid'>"), 30)

    def test_receipt_file_mutation_fails_before_parse_or_render(self):
        with tempfile.TemporaryDirectory() as directory:
            mutated = Path(directory) / "receipt.json"
            shutil.copyfile(RECEIPT, mutated)
            mutated.write_bytes(mutated.read_bytes() + b"\n")
            with self.assertRaisesRegex(review.V15BReviewError, "receipt file SHA"):
                review._validate_inputs(mutated, TENSORS, SOURCE)

    def test_tensor_file_mutation_fails_before_load_or_render(self):
        with tempfile.TemporaryDirectory() as directory:
            mutated = Path(directory) / "affinity.safetensors"
            shutil.copyfile(TENSORS, mutated)
            payload = bytearray(mutated.read_bytes())
            payload[-1] ^= 1
            mutated.write_bytes(payload)
            with self.assertRaisesRegex(review.V15BReviewError, "tensor file SHA"):
                review._validate_inputs(RECEIPT, mutated, SOURCE)

    def test_r6_receipt_and_tensor_mutations_fail_before_render(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            shutil.copyfile(R6_RECEIPT, receipt)
            receipt.write_bytes(receipt.read_bytes() + b"\n")
            with self.assertRaisesRegex(review.V15BReviewError, "receipt file SHA"):
                review._validate_inputs(receipt, R6_TENSORS, SOURCE)

            tensor = Path(directory) / "affinity.safetensors"
            shutil.copyfile(R6_TENSORS, tensor)
            payload = bytearray(tensor.read_bytes())
            payload[-1] ^= 1
            tensor.write_bytes(payload)
            with self.assertRaisesRegex(review.V15BReviewError, "tensor file SHA"):
                review._validate_inputs(R6_RECEIPT, tensor, SOURCE)


if __name__ == "__main__":
    unittest.main()
