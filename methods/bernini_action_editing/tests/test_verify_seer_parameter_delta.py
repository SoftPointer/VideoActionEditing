from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from tools import verify_seer_parameter_delta as verifier  # noqa: E402


SHA_A = "a" * 64
SHA_B = "b" * 64


class VerifySeerParameterDeltaTest(unittest.TestCase):
    def make_checkpoint(
        self,
        root: Path,
        *,
        initial: str = SHA_A,
        final: str = SHA_B,
        gradient: float = 0.5,
        steps: int = 4,
    ) -> Path:
        checkpoint = root / "checkpoint-00000004"
        adapter = checkpoint / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"real-delta")
        receipt = {
            "schema_version": "test",
            "global_step": steps,
            "last_metrics": {"preclip_gradient_norm": gradient},
            "immutable_contract": {
                "expected_seer_manifest_sha256": SHA_A,
                "method_source_archive_sha256": SHA_B,
            },
            "adapter": {
                "initialization_digest": initial,
                "checkpoint_parameter_digest": final,
            },
        }
        receipt["receipt_digest"] = verifier.object_sha256(receipt)
        (checkpoint / "receipt.json").write_text(
            json.dumps(receipt, sort_keys=True), encoding="utf-8"
        )
        return checkpoint

    def test_accepts_real_four_step_delta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.make_checkpoint(Path(directory))
            result = verifier.verify_checkpoint(
                checkpoint=checkpoint,
                expected_steps=4,
                expected_manifest_sha256=SHA_A,
                expected_source_archive_sha256=SHA_B,
            )
        self.assertTrue(result["parameter_digest_changed"])
        self.assertTrue(result["engineering_execution_success"])
        self.assertFalse(result["method_success"])

    def test_rejects_unchanged_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.make_checkpoint(Path(directory), final=SHA_A)
            with self.assertRaisesRegex(verifier.VerificationError, "equal initialization"):
                verifier.verify_checkpoint(
                    checkpoint=checkpoint,
                    expected_steps=4,
                expected_manifest_sha256=SHA_A,
                    expected_source_archive_sha256=SHA_B,
                )

    def test_rejects_zero_gradient(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.make_checkpoint(Path(directory), gradient=0.0)
            with self.assertRaisesRegex(verifier.VerificationError, "gradient norm"):
                verifier.verify_checkpoint(
                    checkpoint=checkpoint,
                    expected_steps=4,
                expected_manifest_sha256=SHA_A,
                    expected_source_archive_sha256=SHA_B,
                )

    def test_main_writes_create_only_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.make_checkpoint(root)
            output = root / "delta.json"
            argv = [
                "--checkpoint", str(checkpoint),
                "--expected-steps", "4",
                "--expected-seer-manifest-sha256", SHA_A,
                "--expected-source-archive-sha256", SHA_B,
                "--output", str(output),
            ]
            self.assertEqual(verifier.main(argv), 0)
            with self.assertRaisesRegex(
                verifier.VerificationError, "create-only verification output exists"
            ):
                verifier.main(argv)

    def test_accepts_exact_b0_receipt_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-00000004"
            adapter = checkpoint / "adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"b0-delta")
            receipt = {
                "schema_version": verifier.B0_TRAINING_RECEIPT_SCHEMA,
                "global_step": 4,
                "last_preclip_gradient_norm": 0.125,
                "method_source_archive_sha256": SHA_B,
                "seer": {
                    "owner_spec_sha256": SHA_A,
                    "training_completion_is_method_success": False,
                    "heldout_decoded_review_required": True,
                },
                "parameter_update_evidence": {
                    "initial_trainable_parameter_digest": SHA_A,
                    "final_trainable_parameter_digest": SHA_B,
                    "exact_parameter_bytes_changed": True,
                    "method_success_claimed": False,
                },
            }
            receipt["receipt_digest"] = verifier.object_sha256(receipt)
            (checkpoint / "receipt.json").write_text(
                json.dumps(receipt, sort_keys=True), encoding="utf-8"
            )
            result = verifier.verify_checkpoint(
                checkpoint=checkpoint,
                expected_steps=4,
                expected_owner_spec_sha256=SHA_A,
                expected_source_archive_sha256=SHA_B,
            )
        self.assertEqual(result["trainer_receipt_layout"], "b0_train_lora_specialization")
        self.assertEqual(result["seer_binding_kind"], "owner_spec")

    def test_rejects_b0_receipt_as_dataset_manifest_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-00000004"
            adapter = checkpoint / "adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"b0-delta")
            receipt = {
                "schema_version": verifier.B0_TRAINING_RECEIPT_SCHEMA,
                "global_step": 4,
                "last_preclip_gradient_norm": 0.125,
                "method_source_archive_sha256": SHA_B,
                "seer": {
                    "owner_spec_sha256": SHA_A,
                    "training_completion_is_method_success": False,
                    "heldout_decoded_review_required": True,
                },
                "parameter_update_evidence": {
                    "initial_trainable_parameter_digest": SHA_A,
                    "final_trainable_parameter_digest": SHA_B,
                    "exact_parameter_bytes_changed": True,
                    "method_success_claimed": False,
                },
            }
            receipt["receipt_digest"] = verifier.object_sha256(receipt)
            (checkpoint / "receipt.json").write_text(
                json.dumps(receipt, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(verifier.VerificationError, "not an admitted"):
                verifier.verify_checkpoint(
                    checkpoint=checkpoint,
                    expected_steps=4,
                    expected_manifest_sha256=SHA_A,
                    expected_source_archive_sha256=SHA_B,
                )


if __name__ == "__main__":
    unittest.main()
