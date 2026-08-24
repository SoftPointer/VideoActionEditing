#!/usr/bin/env python3
"""Tests for the fail-closed RS-FQT v8 post-save finalizer."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import finalize_feasible_quotient_checkpoint as finalizer  # noqa: E402


SHA1 = "1" * 40
ARCHIVE_SHA = "2" * 64
PARAMETER_SHA = "3" * 64
BERNINI_SHA = finalizer.legacy.BERNINI_OFFICIAL_COMMIT
VEOMNI_SHA = finalizer.legacy.VEOMNI_TESTED_COMMIT
TREE_SHA = finalizer.legacy.CHECKPOINT_TREE_SHA256


def _resign(receipt):
    receipt.pop("receipt_digest", None)
    receipt["receipt_digest"] = finalizer.legacy.object_sha256(receipt)
    return receipt


def _pending_receipt(base_checkpoint: Path):
    parity = finalizer.v8_loader._expected_parity_contract()
    immutable_value = {
        "method": finalizer.v8_loader.METHOD_NAME,
        "schema_version": finalizer.v8_loader.TRAINING_RECEIPT_SCHEMA,
        "teacher_mode": "paired_displacement_only",
        "pilot_scope": "exact40_fixed_lr_falsification",
        "method_source_revision": SHA1,
        "method_source_archive_sha256": ARCHIVE_SHA,
        "bernini_commit": BERNINI_SHA,
        "veomni_commit": VEOMNI_SHA,
        "checkpoint_path": str(base_checkpoint),
        "checkpoint_tree_sha256": TREE_SHA,
        "checkpoint_content_manifest_sha256": (
            finalizer.v8_loader.v8_train.CHECKPOINT_CONTENT_MANIFEST_SHA256
        ),
        "checkpoint_content_file_count": (
            finalizer.v8_loader.v8_train.CHECKPOINT_CONTENT_FILE_COUNT
        ),
        "inference_loader_parity": parity,
    }
    steps = [
        {
            "optimizer_step": index + 1,
            "sigma_schedule_index": index,
            "teacher_mode": "paired_displacement_only",
        }
        for index in range(40)
    ]
    receipt = {
        "schema_version": finalizer.v8_loader.TRAINING_RECEIPT_SCHEMA,
        "method": finalizer.v8_loader.METHOD_NAME,
        "global_step": 40,
        "max_steps": 40,
        "formal_40_sigma_cycle_complete": True,
        "accepted_sigma_schedule_indices": list(range(40)),
        "step_audit": steps,
        "step_audit_sha256": finalizer.legacy.object_sha256(steps),
        "immutable_contract": {
            "value": immutable_value,
            "digest": finalizer.legacy.object_sha256(immutable_value),
        },
        "bernini_commit": BERNINI_SHA,
        "veomni_commit": VEOMNI_SHA,
        "checkpoint": {
            "path": str(base_checkpoint),
            "tree_sha256": TREE_SHA,
        },
        "adapter": {"checkpoint_parameter_digest": PARAMETER_SHA},
        "inference_loader_parity_pending": True,
        "inference_loader_parity": parity,
        "artifact_validation": {
            "schema_version": finalizer.ARTIFACT_VALIDATION_SCHEMA,
            "verified": False,
            "status": finalizer.PENDING_ARTIFACT_STATUS,
        },
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    return _resign(receipt)


def _write_json(path: Path, value):
    path.write_bytes(finalizer.legacy.canonical_json_bytes(value) + b"\n")


class PendingV8ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.base = Path("/base/Bernini-R-1.3B")

    def test_pending_exact40_v8_is_accepted_but_v7_and_ready_are_not(self):
        pending = _pending_receipt(self.base)
        result = finalizer.validate_pending_receipt(pending)
        self.assertEqual(result["receipt_digest"], pending["receipt_digest"])

        v7 = copy.deepcopy(pending)
        v7["schema_version"] = finalizer.v8_loader.rmc.TRAINING_RECEIPT_SCHEMA
        v7["method"] = finalizer.v8_loader.rmc.METHOD_NAME
        _resign(v7)
        ready = copy.deepcopy(pending)
        ready["inference_loader_parity_pending"] = False
        _resign(ready)
        for value in (v7, ready):
            with self.assertRaises(finalizer.FeasibleQuotientFinalizationError):
                finalizer.validate_pending_receipt(value)

    def test_ready_candidate_binds_exact_pending_transition(self):
        pending = _pending_receipt(self.base)
        artifact = finalizer.build_ready_artifact_validation(
            receipt=pending,
            adapter_config_sha256="4" * 64,
            adapter_model_sha256="5" * 64,
            method_source_revision=SHA1,
            method_source_archive_sha256=ARCHIVE_SHA,
            bernini_revision=BERNINI_SHA,
            veomni_revision=VEOMNI_SHA,
            expected_checkpoint_tree_sha256=TREE_SHA,
        )
        artifact_without_digest = dict(artifact)
        digest = artifact_without_digest.pop("digest")
        self.assertEqual(
            digest, finalizer.legacy.object_sha256(artifact_without_digest)
        )
        self.assertEqual(
            artifact["pending_receipt_digest"], pending["receipt_digest"]
        )
        candidate = finalizer.build_ready_receipt_candidate(
            pending, artifact_validation=artifact
        )
        self.assertFalse(candidate["inference_loader_parity_pending"])
        reconstructed = copy.deepcopy(candidate)
        reconstructed.pop("receipt_digest")
        reconstructed["inference_loader_parity_pending"] = True
        reconstructed["artifact_validation"] = {
            "schema_version": finalizer.ARTIFACT_VALIDATION_SCHEMA,
            "verified": False,
            "status": finalizer.PENDING_ARTIFACT_STATUS,
        }
        self.assertEqual(
            finalizer.legacy.object_sha256(reconstructed),
            pending["receipt_digest"],
        )


class FinalizeV8CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.formal = self.root / finalizer.FORMAL_CHECKPOINT_NAME
        self.adapter = self.formal / "adapter"
        self.adapter.mkdir(parents=True)
        (self.formal / "optimizer.pt").write_bytes(b"optimizer")
        self.config = self.adapter / "adapter_config.json"
        self.model = self.adapter / "adapter_model.safetensors"
        _write_json(self.config, {"peft_type": "LORA"})
        self.model.write_bytes(b"exact-v8-92-tensor-fixture")
        self.base = self.root / "base_checkpoint"
        self.bernini = self.root / "bernini"
        self.veomni = self.root / "veomni"
        self.pending = _pending_receipt(self.base)
        self.receipt_path = self.formal / "receipt.json"
        self.latest_path = self.root / "latest.json"
        _write_json(self.receipt_path, self.pending)
        self.pending_latest = {
            "checkpoint": str(self.formal),
            "global_step": 40,
            "receipt_digest": self.pending["receipt_digest"],
        }
        _write_json(self.latest_path, self.pending_latest)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _identity(receipt):
        artifact = receipt["artifact_validation"]
        return {
            "checkpoint_parameter_digest": PARAMETER_SHA,
            "adapter_config_sha256": artifact["adapter_config_sha256"],
            "adapter_model_sha256": artifact["adapter_model_sha256"],
            "artifact_validation_digest": artifact["digest"],
        }

    def _patch_runtime(self, *, strict_side_effect=None):
        fresh_base = object()

        def validate_contract(config, receipt, **kwargs):
            del config, kwargs
            return self._identity(receipt)

        def strict_load(**kwargs):
            if strict_side_effect is not None:
                return strict_side_effect(**kwargs)
            return object(), 92, 46, self._identity(kwargs["receipt"])

        return (
            fresh_base,
            mock.patch.object(
                finalizer.legacy,
                "validate_source_trees",
                return_value=(self.bernini, self.veomni, BERNINI_SHA, VEOMNI_SHA),
            ),
            mock.patch.object(
                finalizer.legacy,
                "validate_checkpoint",
                return_value=(self.base, {"num_attention_heads": 12}),
            ),
            mock.patch.object(finalizer.legacy, "activate_source_trees"),
            mock.patch.object(
                finalizer, "_build_fresh_bernini_base", return_value=fresh_base
            ),
            mock.patch.object(
                finalizer.v8_loader,
                "validate_training_adapter_contract",
                side_effect=validate_contract,
            ),
            mock.patch.object(
                finalizer.v8_loader,
                "strict_load_adapter",
                side_effect=strict_load,
            ),
        )

    def _finalize(self, *, strict_side_effect=None):
        (
            fresh_base,
            source_patch,
            checkpoint_patch,
            activate_patch,
            base_patch,
            validate_patch,
            strict_patch,
        ) = self._patch_runtime(strict_side_effect=strict_side_effect)
        with source_patch, checkpoint_patch, activate_patch, base_patch, validate_patch as validator, strict_patch as strict:
            result = finalizer.finalize_checkpoint(
                self.formal,
                bernini_root=self.bernini,
                veomni_root=self.veomni,
                base_checkpoint=self.base,
                method_source_revision=SHA1,
                method_source_archive_sha256=ARCHIVE_SHA,
            )
        return result, fresh_base, validator, strict

    def test_success_strict_load_then_atomically_publishes_ready_state(self):
        result, fresh_base, validator, strict = self._finalize()
        published = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        latest = json.loads(self.latest_path.read_text(encoding="utf-8"))
        self.assertEqual(result, published)
        self.assertFalse(published["inference_loader_parity_pending"])
        artifact = published["artifact_validation"]
        self.assertTrue(artifact["verified"])
        self.assertEqual(artifact["adapter_tensor_count"], 92)
        self.assertEqual(artifact["active_lora_module_count"], 46)
        self.assertEqual(
            artifact["adapter_config_sha256"],
            finalizer.legacy.file_sha256(self.config),
        )
        self.assertEqual(
            artifact["adapter_model_sha256"],
            finalizer.legacy.file_sha256(self.model),
        )
        self.assertEqual(latest["receipt_digest"], published["receipt_digest"])
        self.assertFalse(latest["inference_loader_parity_pending"])
        self.assertEqual(latest["artifact_validation_digest"], artifact["digest"])
        self.assertIs(strict.call_args.kwargs["base_model"], fresh_base)
        validator.assert_called_once()
        strict.assert_called_once()
        self.assertFalse(
            (self.formal / ".finalize-feasible-quotient.lock").exists()
        )

    def test_strict_failure_or_artifact_drift_keeps_pending_bytes(self):
        receipt_before = self.receipt_path.read_bytes()
        latest_before = self.latest_path.read_bytes()

        def fail(**kwargs):
            del kwargs
            raise finalizer.v8_loader.FeasibleQuotientInferenceError(
                "injected v8 strict failure"
            )

        with self.assertRaisesRegex(
            finalizer.FeasibleQuotientFinalizationError, "injected"
        ):
            self._finalize(strict_side_effect=fail)
        self.assertEqual(self.receipt_path.read_bytes(), receipt_before)
        self.assertEqual(self.latest_path.read_bytes(), latest_before)

        def mutate(**kwargs):
            self.model.write_bytes(b"changed-during-finalization")
            return object(), 92, 46, self._identity(kwargs["receipt"])

        with self.assertRaisesRegex(
            finalizer.FeasibleQuotientFinalizationError,
            "changed during strict finalization",
        ):
            self._finalize(strict_side_effect=mutate)
        self.assertEqual(self.receipt_path.read_bytes(), receipt_before)
        self.assertEqual(self.latest_path.read_bytes(), latest_before)

    def test_latest_first_interruption_is_recoverable(self):
        config_sha = finalizer.legacy.file_sha256(self.config)
        model_sha = finalizer.legacy.file_sha256(self.model)
        artifact = finalizer.build_ready_artifact_validation(
            receipt=self.pending,
            adapter_config_sha256=config_sha,
            adapter_model_sha256=model_sha,
            method_source_revision=SHA1,
            method_source_archive_sha256=ARCHIVE_SHA,
            bernini_revision=BERNINI_SHA,
            veomni_revision=VEOMNI_SHA,
            expected_checkpoint_tree_sha256=TREE_SHA,
        )
        candidate = finalizer.build_ready_receipt_candidate(
            self.pending, artifact_validation=artifact
        )
        interrupted_latest = {
            **self.pending_latest,
            "receipt_digest": candidate["receipt_digest"],
            "inference_loader_parity_pending": False,
            "artifact_validation_digest": artifact["digest"],
        }
        _write_json(self.latest_path, interrupted_latest)
        result, *_ = self._finalize()
        self.assertEqual(result, candidate)
        self.assertFalse(
            json.loads(self.receipt_path.read_text())[
                "inference_loader_parity_pending"
            ]
        )

    def test_publication_failure_restores_prior_latest_and_pending_receipt(self):
        ready = copy.deepcopy(self.pending)
        ready["inference_loader_parity_pending"] = False
        ready["artifact_validation"] = {"digest": "9" * 64}
        _resign(ready)
        real_writer = finalizer._atomic_write_json
        calls = []

        def fail_second(path, value):
            calls.append(path)
            if len(calls) == 2:
                raise OSError("injected publication failure")
            return real_writer(path, value)

        receipt_before = self.receipt_path.read_bytes()
        latest_before = self.latest_path.read_bytes()
        with mock.patch.object(
            finalizer, "_atomic_write_json", side_effect=fail_second
        ):
            with self.assertRaisesRegex(
                finalizer.FeasibleQuotientFinalizationError,
                "prior latest.json was restored",
            ):
                finalizer._publish_ready_receipt(
                    receipt_path=self.receipt_path,
                    ready_receipt=ready,
                    latest_path=self.latest_path,
                    pending_latest=self.pending_latest,
                )
        self.assertEqual(self.receipt_path.read_bytes(), receipt_before)
        self.assertEqual(self.latest_path.read_bytes(), latest_before)

    def test_cli_exposes_no_video_or_sampling_surface(self):
        destinations = {action.dest for action in finalizer.build_parser()._actions}
        self.assertIn("formal_checkpoint_root", destinations)
        for forbidden in (
            "source_video",
            "target_video",
            "instruction",
            "output",
            "num_inference_steps",
        ):
            self.assertNotIn(forbidden, destinations)


if __name__ == "__main__":
    unittest.main()
