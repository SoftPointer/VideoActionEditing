#!/usr/bin/env python3
"""Tests for the fail-closed RMC v7 post-save checkpoint finalizer."""

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

import finalize_relational_motion_commutator_checkpoint as finalizer  # noqa: E402


SHA1 = "1" * 40
ARCHIVE_SHA = "2" * 64
BERNINI_SHA = finalizer.legacy.BERNINI_OFFICIAL_COMMIT
VEOMNI_SHA = finalizer.legacy.VEOMNI_TESTED_COMMIT
TREE_SHA = finalizer.legacy.CHECKPOINT_TREE_SHA256
PARAMETER_SHA = "3" * 64


def _resign_receipt(receipt):
    receipt.pop("receipt_digest", None)
    receipt["receipt_digest"] = finalizer.legacy.object_sha256(receipt)
    return receipt


def _pending_receipt(base_checkpoint: Path):
    parity = {
        "verified": True,
        "verification_stage": "immutable_launcher_preflight_before_model_load",
        "loader_module": finalizer.v7_train.INFERENCE_LOADER_MODULE,
        "runner_module": finalizer.v7_train.INFERENCE_RUNNER_MODULE,
        "finalizer_module": finalizer.v7_train.INFERENCE_FINALIZER_MODULE,
        "training_receipt_schema": finalizer.rmc.TRAINING_RECEIPT_SCHEMA,
        "inference_receipt_schema": finalizer.rmc.INFERENCE_RECEIPT_SCHEMA,
        "contract_tests": list(finalizer.v7_train.INFERENCE_PARITY_TESTS),
        "source_revision_and_archive_bound": True,
        "strict_loader_rejects_pending_canary_and_incomplete_cycle": True,
    }
    immutable_value = {
        "method": finalizer.rmc.METHOD_NAME,
        "schema_version": finalizer.rmc.TRAINING_RECEIPT_SCHEMA,
        "teacher_mode": "target_only",
        "method_source_revision": SHA1,
        "method_source_archive_sha256": ARCHIVE_SHA,
        "bernini_commit": BERNINI_SHA,
        "veomni_commit": VEOMNI_SHA,
        "checkpoint_path": str(base_checkpoint),
        "checkpoint_tree_sha256": TREE_SHA,
        "inference_loader_parity": parity,
    }
    steps = [
        {
            "optimizer_step": index + 1,
            "sigma_schedule_index": index,
            "teacher_mode": "target_only",
        }
        for index in range(40)
    ]
    receipt = {
        "schema_version": finalizer.rmc.TRAINING_RECEIPT_SCHEMA,
        "method": finalizer.rmc.METHOD_NAME,
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
        "adapter": {
            "checkpoint_parameter_digest": PARAMETER_SHA,
        },
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
    return _resign_receipt(receipt)


def _write_json(path: Path, value):
    path.write_bytes(finalizer.legacy.canonical_json_bytes(value) + b"\n")


class PendingReceiptTests(unittest.TestCase):
    def setUp(self):
        self.base = Path("/base/Bernini-R-1.3B")

    def test_pending_exact40_target_only_is_accepted(self):
        receipt = _pending_receipt(self.base)
        result = finalizer.validate_pending_receipt(receipt)
        self.assertEqual(result["receipt_digest"], receipt["receipt_digest"])
        self.assertEqual(result["immutable_value"]["teacher_mode"], "target_only")

    def test_canary_non40_and_non_target_only_are_rejected(self):
        cases = []
        canary = _pending_receipt(self.base)
        canary.update(
            {
                "global_step": 1,
                "max_steps": 1,
                "formal_40_sigma_cycle_complete": False,
                "accepted_sigma_schedule_indices": [0],
                "step_audit": canary["step_audit"][:1],
            }
        )
        canary["step_audit_sha256"] = finalizer.legacy.object_sha256(
            canary["step_audit"]
        )
        cases.append(_resign_receipt(canary))

        non40 = _pending_receipt(self.base)
        non40["max_steps"] = 41
        cases.append(_resign_receipt(non40))

        relational = _pending_receipt(self.base)
        relational["step_audit"][7]["teacher_mode"] = "relational_auxiliary"
        relational["step_audit_sha256"] = finalizer.legacy.object_sha256(
            relational["step_audit"]
        )
        cases.append(_resign_receipt(relational))

        for candidate in cases:
            with self.subTest(global_step=candidate["global_step"]):
                with self.assertRaises(
                    finalizer.RelationalMotionCommutatorFinalizationError
                ):
                    finalizer.validate_pending_receipt(candidate)

    def test_ready_or_non_pristine_artifact_is_rejected(self):
        ready = _pending_receipt(self.base)
        ready["inference_loader_parity_pending"] = False
        _resign_receipt(ready)
        with self.assertRaisesRegex(
            finalizer.RelationalMotionCommutatorFinalizationError, "pending"
        ):
            finalizer.validate_pending_receipt(ready)

        altered = _pending_receipt(self.base)
        altered["artifact_validation"]["attempted"] = True
        _resign_receipt(altered)
        with self.assertRaisesRegex(
            finalizer.RelationalMotionCommutatorFinalizationError, "pristine"
        ):
            finalizer.validate_pending_receipt(altered)

    def test_ready_candidate_binds_artifact_and_both_digests(self):
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
        artifact_digest = artifact_without_digest.pop("digest")
        self.assertEqual(
            artifact_digest,
            finalizer.legacy.object_sha256(artifact_without_digest),
        )
        self.assertEqual(artifact["checkpoint_parameter_digest"], PARAMETER_SHA)
        self.assertEqual(
            artifact["pending_receipt_digest"], pending["receipt_digest"]
        )
        candidate = finalizer.build_ready_receipt_candidate(
            pending, artifact_validation=artifact
        )
        self.assertFalse(candidate["inference_loader_parity_pending"])
        self.assertTrue(candidate["artifact_validation"]["verified"])
        receipt_without_digest = dict(candidate)
        receipt_digest = receipt_without_digest.pop("receipt_digest")
        self.assertEqual(
            receipt_digest, finalizer.legacy.object_sha256(receipt_without_digest)
        )
        self.assertTrue(pending["inference_loader_parity_pending"])


class FinalizeCheckpointTests(unittest.TestCase):
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
        self.model.write_bytes(b"exact-92-tensor-fixture")
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
            return (
                object(),
                92,
                46,
                self._identity(kwargs["receipt"]),
            )

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
                finalizer.rmc,
                "validate_training_adapter_contract",
                side_effect=validate_contract,
            ),
            mock.patch.object(
                finalizer.rmc, "strict_load_adapter", side_effect=strict_load
            ),
        )

    def _finalize_with_patches(self, *, strict_side_effect=None):
        (
            fresh_base,
            source_patch,
            checkpoint_patch,
            activate_patch,
            base_patch,
            validate_patch,
            strict_patch,
        ) = self._patch_runtime(strict_side_effect=strict_side_effect)
        with source_patch, checkpoint_patch, activate_patch, base_patch, validate_patch as validator, strict_patch as loader:
            result = finalizer.finalize_checkpoint(
                self.formal,
                bernini_root=self.bernini,
                veomni_root=self.veomni,
                base_checkpoint=self.base,
                method_source_revision=SHA1,
                method_source_archive_sha256=ARCHIVE_SHA,
            )
        return result, fresh_base, validator, loader

    def test_success_actual_strict_reload_then_publishes_receipt_and_latest(self):
        result, fresh_base, validator, loader = self._finalize_with_patches()
        published = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        latest = json.loads(self.latest_path.read_text(encoding="utf-8"))
        self.assertEqual(published, result)
        self.assertFalse(published["inference_loader_parity_pending"])
        artifact = published["artifact_validation"]
        self.assertTrue(artifact["verified"])
        self.assertEqual(artifact["serialized_target_pattern_count"], 17)
        self.assertEqual(artifact["expanded_target_module_count"], 46)
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
        self.assertFalse(
            (self.formal / ".finalize-relational-motion-commutator.lock").exists()
        )
        validator.assert_called_once()
        loader.assert_called_once()
        self.assertIs(loader.call_args.kwargs["base_model"], fresh_base)
        self.assertEqual(loader.call_args.kwargs["receipt"], published)

    def test_strict_reload_failure_keeps_pending_files_byte_exact(self):
        receipt_before = self.receipt_path.read_bytes()
        latest_before = self.latest_path.read_bytes()

        def fail_strict(**kwargs):
            del kwargs
            raise finalizer.rmc.RelationalMotionCommutatorInferenceError(
                "injected strict reload failure"
            )

        with self.assertRaisesRegex(
            finalizer.RelationalMotionCommutatorFinalizationError,
            "injected strict reload failure",
        ):
            self._finalize_with_patches(strict_side_effect=fail_strict)
        self.assertEqual(self.receipt_path.read_bytes(), receipt_before)
        self.assertEqual(self.latest_path.read_bytes(), latest_before)
        self.assertTrue(
            json.loads(self.receipt_path.read_text())["inference_loader_parity_pending"]
        )
        self.assertFalse(
            (self.formal / ".finalize-relational-motion-commutator.lock").exists()
        )

    def test_artifact_drift_during_reload_keeps_receipt_pending(self):
        receipt_before = self.receipt_path.read_bytes()
        latest_before = self.latest_path.read_bytes()

        def mutate_strict(**kwargs):
            self.model.write_bytes(b"changed-during-load")
            return object(), 92, 46, self._identity(kwargs["receipt"])

        with self.assertRaisesRegex(
            finalizer.RelationalMotionCommutatorFinalizationError,
            "changed during strict finalization",
        ):
            self._finalize_with_patches(strict_side_effect=mutate_strict)
        self.assertEqual(self.receipt_path.read_bytes(), receipt_before)
        self.assertEqual(self.latest_path.read_bytes(), latest_before)

    def test_nonformal_receipt_is_rejected_before_model_or_loader(self):
        invalid = copy.deepcopy(self.pending)
        invalid["max_steps"] = 41
        _resign_receipt(invalid)
        _write_json(self.receipt_path, invalid)
        self.pending_latest["receipt_digest"] = invalid["receipt_digest"]
        _write_json(self.latest_path, self.pending_latest)
        with mock.patch.object(
            finalizer, "_build_fresh_bernini_base"
        ) as base_builder, mock.patch.object(
            finalizer.rmc, "strict_load_adapter"
        ) as loader:
            with self.assertRaises(
                finalizer.RelationalMotionCommutatorFinalizationError
            ):
                finalizer.finalize_checkpoint(
                    self.formal,
                    bernini_root=self.bernini,
                    veomni_root=self.veomni,
                    base_checkpoint=self.base,
                    method_source_revision=SHA1,
                    method_source_archive_sha256=ARCHIVE_SHA,
                )
        base_builder.assert_not_called()
        loader.assert_not_called()

    def test_publication_failure_restores_pending_latest_and_receipt(self):
        ready_receipt = copy.deepcopy(self.pending)
        ready_receipt["inference_loader_parity_pending"] = False
        ready_receipt["artifact_validation"] = {"digest": "9" * 64}
        _resign_receipt(ready_receipt)
        real_writer = finalizer._atomic_write_json
        calls = []

        def fail_second(path, value):
            calls.append(path)
            if len(calls) == 2:
                raise OSError("injected receipt publication failure")
            return real_writer(path, value)

        receipt_before = self.receipt_path.read_bytes()
        latest_before = self.latest_path.read_bytes()
        with mock.patch.object(finalizer, "_atomic_write_json", side_effect=fail_second):
            with self.assertRaisesRegex(
                finalizer.RelationalMotionCommutatorFinalizationError,
                "pending latest.json was restored",
            ):
                finalizer._publish_ready_receipt(
                    receipt_path=self.receipt_path,
                    ready_receipt=ready_receipt,
                    latest_path=self.latest_path,
                    pending_latest=self.pending_latest,
                )
        self.assertEqual(self.receipt_path.read_bytes(), receipt_before)
        self.assertEqual(self.latest_path.read_bytes(), latest_before)

    def test_cli_has_no_video_or_sampling_arguments(self):
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
