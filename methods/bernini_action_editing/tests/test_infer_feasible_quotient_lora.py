#!/usr/bin/env python3
"""Contract tests for the finalized RS-FQT v8 exact46/92 loader."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_feasible_quotient_lora as loader  # noqa: E402


SHA1 = "1" * 40
ARCHIVE_SHA = "2" * 64
SUMMARY_SHA = "3" * 64
INDEX_SHA = "4" * 64
ROUTING_DIGEST = "5" * 64
INIT_SHA = "6" * 64
PARAMETER_SHA = "7" * 64
OPTIMIZER_SHA = "8" * 64
CONFIG_SHA = "9" * 64
MODEL_SHA = "a" * 64
TREE_SHA = loader.legacy.CHECKPOINT_TREE_SHA256
BERNINI_SHA = loader.legacy.BERNINI_OFFICIAL_COMMIT
VEOMNI_SHA = loader.legacy.VEOMNI_TESTED_COMMIT


def _resign(receipt):
    receipt.pop("receipt_digest", None)
    receipt["receipt_digest"] = loader.legacy.object_sha256(receipt)
    return receipt


def _adapter_config():
    return {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "bias": "none",
        "modules_to_save": None,
        "use_dora": False,
        "use_rslora": False,
        "target_modules": list(
            reversed(loader.expected_serialized_target_patterns())
        ),
    }


def _pending_receipt():
    targets = loader.expected_lora_targets()
    args = SimpleNamespace(
        method_source_revision=SHA1,
        method_source_archive_sha256=ARCHIVE_SHA,
        expected_bernini_commit=BERNINI_SHA,
        expected_veomni_commit=VEOMNI_SHA,
        expected_checkpoint_tree_sha256=TREE_SHA,
        seed=17,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )
    dataset = SimpleNamespace(signature="strict-dataset-signature")
    router = SimpleNamespace(
        digest=ROUTING_DIGEST,
        file_sha256=loader.v8_train.v5.STRICT_ROUTING_SHA256,
    )
    route = SimpleNamespace(
        iid="iid", tier="motion_only", full_target_weight=0.0
    )
    immutable = loader.v8_train._immutable_contract(
        args=args,
        dataset=dataset,
        dataset_summary={"sha256": SUMMARY_SHA, "index_sha256": INDEX_SHA},
        router=router,
        eligible_routes=[(index, route) for index in range(359)],
        target_modules=targets,
        checkpoint=Path("/base/Bernini-R-1.3B"),
        loss_config=loader.objective.FeasibleQuotientLossConfig(),
    )
    steps = [
        {
            "optimizer_step": index + 1,
            "row_index": index,
            "iid": f"iid-{index}",
            "seed": index + 100,
            "sigma_schedule_index": index,
            "sigma_timestep": loader.sigma_strata.PINNED_TIMESTEPS[index],
            "teacher_mode": "paired_displacement_only",
            "metrics_timing": loader.v8_train.METRICS_TIMING,
            "rho": loader.commutator.release_rho(index),
        }
        for index in range(40)
    ]
    parameter_names = [f"adapter.parameter.{index:03d}" for index in range(92)]
    receipt = {
        "schema_version": loader.TRAINING_RECEIPT_SCHEMA,
        "method": loader.METHOD_NAME,
        "global_step": 40,
        "max_steps": 40,
        "formal_40_sigma_cycle_complete": True,
        "accepted_sigma_schedule_indices": list(range(40)),
        "step_audit": steps,
        "step_audit_sha256": loader.legacy.object_sha256(steps),
        "last_metrics": {"loss_total": 0.5},
        "metrics_timing": loader.v8_train.METRICS_TIMING,
        "immutable_contract": immutable,
        "bernini_commit": BERNINI_SHA,
        "veomni_commit": VEOMNI_SHA,
        "checkpoint": {
            "path": "/base/Bernini-R-1.3B",
            "tree_sha256": TREE_SHA,
        },
        "dataset": {
            "path": "/data/parquet",
            "rows": 644,
            "signature": dataset.signature,
            "summary": {"sha256": SUMMARY_SHA, "index_sha256": INDEX_SHA},
            "routing": {
                "default_tier": "reject",
                "explicit_route_counts": {
                    "full_pair": 0,
                    "motion_only": 359,
                    "reject": 285,
                },
                "file_sha256": router.file_sha256,
                "routing_digest": ROUTING_DIGEST,
            },
        },
        "adapter": {
            "rank": 8,
            "alpha": 8,
            "scope": loader.REQUIRED_LORA_SCOPE,
            "target_module_count": 46,
            "target_modules": targets,
            "target_modules_sha256": loader.legacy.object_sha256(targets),
            "trainable_parameter_count": 4096,
            "parameter_names_sha256": loader.legacy.object_sha256(
                parameter_names
            ),
            "initialization_digest": INIT_SHA,
            "checkpoint_parameter_digest": PARAMETER_SHA,
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": loader.v8_train.LEARNING_RATE,
            "weight_decay": 0.0,
            "max_gradient_norm": 1.0,
            "parameter_names": parameter_names,
            "checkpoint_state_digest": OPTIMIZER_SHA,
            "zero_release_moment_reset": {
                "first_zero_release_schedule_index": 31,
                "reset_before_optimizer_step": 32,
                "completed_optimizer_steps": 40,
                "reset_count": 1,
                "state_step_after_reset_suffix": 9,
                "state_step_values": [9],
                "state_parameter_count": len(parameter_names),
                "weight_decay": 0.0,
            },
        },
        "distributed": {
            "world_size": 4,
            "ulysses_size": 4,
            "backend": "nccl/rccl",
            "same_pair_all_ranks": True,
            "explicit_lora_gradient_all_reduce": True,
        },
        "transformers_version": "test-transformers",
        "inference_conditions": ["source_video", "action_instruction"],
        "training_only_generator_and_target": False,
        "training_only_paired_target": True,
        "training_generator_forwards": 0,
        "inference_generator_forwards": 0,
        "teacher_mode": "paired_displacement_only",
        "pilot_scope": "exact40_fixed_lr_falsification",
        "external_mask_track_flow_pose_trajectory": False,
        "first_frame_anchor": False,
        "experimental_training": True,
        "dataset_post_video_acceptance": "pending",
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "resume_integrated": False,
        "inference_loader_parity_pending": True,
        "inference_loader_parity": immutable["value"][
            "inference_loader_parity"
        ],
        "artifact_validation": {
            "schema_version": loader.ARTIFACT_VALIDATION_SCHEMA,
            "verified": False,
            "status": loader.PENDING_ARTIFACT_STATUS,
        },
    }
    return _resign(receipt)


def _ready_receipt(*, config_sha=CONFIG_SHA, model_sha=MODEL_SHA):
    pending = _pending_receipt()
    artifact = {
        "schema_version": loader.ARTIFACT_VALIDATION_SCHEMA,
        "verified": True,
        "status": loader.READY_ARTIFACT_STATUS,
        "adapter_config_sha256": config_sha,
        "adapter_model_sha256": model_sha,
        "serialized_target_pattern_count": 17,
        "expanded_target_module_count": 46,
        "adapter_tensor_count": 92,
        "active_lora_module_count": 46,
        "strict_tensor_reload_equal": True,
        "parameter_digest_verified_after_safetensors_reload": True,
        "checkpoint_parameter_digest": PARAMETER_SHA,
        "pending_receipt_digest": pending["receipt_digest"],
        "validator_method_source_revision": SHA1,
        "validator_method_source_archive_sha256": ARCHIVE_SHA,
        "bernini_commit": BERNINI_SHA,
        "veomni_commit": VEOMNI_SHA,
        "checkpoint_tree_sha256": TREE_SHA,
        "checkpoint_content_manifest_sha256": (
            loader.v8_train.CHECKPOINT_CONTENT_MANIFEST_SHA256
        ),
        "checkpoint_content_file_count": (
            loader.v8_train.CHECKPOINT_CONTENT_FILE_COUNT
        ),
        "loader_module": loader.LOADER_MODULE,
        "finalizer_module": loader.FINALIZER_MODULE,
    }
    artifact["digest"] = loader.legacy.object_sha256(artifact)
    pending["artifact_validation"] = artifact
    pending["inference_loader_parity_pending"] = False
    return _resign(pending)


class FeasibleQuotientReceiptContractTests(unittest.TestCase):
    def test_valid_ready_v8_exact40_exact46_contract(self):
        identity = loader.validate_training_adapter_contract(
            _adapter_config(), _ready_receipt()
        )
        self.assertEqual(identity["global_step"], 40)
        self.assertEqual(len(identity["targets"]), 46)
        self.assertEqual(len(identity["serialized_target_modules"]), 17)
        self.assertEqual(identity["checkpoint_parameter_digest"], PARAMETER_SHA)

    def test_pending_v7_canary_and_transition_drift_fail_closed(self):
        pending = _pending_receipt()
        v7 = _ready_receipt()
        v7["schema_version"] = loader.rmc.TRAINING_RECEIPT_SCHEMA
        v7["method"] = loader.rmc.METHOD_NAME
        _resign(v7)
        canary = _ready_receipt()
        canary["global_step"] = 1
        _resign(canary)
        drift = _ready_receipt()
        drift["last_metrics"] = {"loss_total": 999.0}
        _resign(drift)
        for value in (pending, v7, canary, drift):
            with self.subTest(schema=value["schema_version"]):
                with self.assertRaises(loader.FeasibleQuotientInferenceError):
                    loader.validate_training_adapter_contract(
                        _adapter_config(), value
                    )

    def test_artifact_extra_field_and_nonexact17_are_rejected(self):
        receipt = _ready_receipt()
        receipt["artifact_validation"]["unaudited"] = True
        artifact = receipt["artifact_validation"]
        artifact.pop("digest")
        artifact["digest"] = loader.legacy.object_sha256(artifact)
        _resign(receipt)
        with self.assertRaisesRegex(
            loader.FeasibleQuotientInferenceError, "unaudited"
        ):
            loader.validate_training_adapter_contract(_adapter_config(), receipt)

        config = _adapter_config()
        config["target_modules"] = config["target_modules"][:-1]
        with self.assertRaisesRegex(
            loader.FeasibleQuotientInferenceError, "exact17"
        ):
            loader.validate_training_adapter_contract(
                config, _ready_receipt()
            )


class FeasibleQuotientStrictLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.config_path = root / "adapter_config.json"
        self.model_path = root / "adapter_model.safetensors"
        self.config = _adapter_config()
        self.config_path.write_text(
            json.dumps(self.config, sort_keys=True), encoding="utf-8"
        )
        self.model_path.write_bytes(b"exact-v8-adapter-fixture")
        self.config_sha = loader.legacy.file_sha256(self.config_path)
        self.model_sha = loader.legacy.file_sha256(self.model_path)
        self.receipt = _ready_receipt(
            config_sha=self.config_sha, model_sha=self.model_sha
        )
        self.bundle = SimpleNamespace(
            adapter_dir=self.config_path.parent,
            adapter_config_path=self.config_path,
            adapter_model_path=self.model_path,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_strict_loader_delegates_generic_exact_target_loader(self):
        sentinel = object()
        with mock.patch.object(
            loader.rmc.v5,
            "_strict_load_adapter",
            return_value=(sentinel, 92, 46),
        ) as delegated:
            result = loader.strict_load_adapter(
                base_model=object(),
                bundle=self.bundle,
                adapter_config=self.config,
                receipt=self.receipt,
            )
        self.assertIs(result[0], sentinel)
        self.assertEqual(result[1:3], (92, 46))
        self.assertEqual(result[3]["adapter_model_sha256"], self.model_sha)
        self.assertEqual(
            delegated.call_args.kwargs["identity"]["targets"],
            loader.expected_lora_targets(),
        )

    def test_hash_drift_before_or_during_reload_is_rejected(self):
        self.model_path.write_bytes(b"changed-before-load")
        with mock.patch.object(loader.rmc.v5, "_strict_load_adapter") as delegated:
            with self.assertRaisesRegex(
                loader.FeasibleQuotientInferenceError, "finalized artifact hashes"
            ):
                loader.strict_load_adapter(
                    base_model=object(),
                    bundle=self.bundle,
                    adapter_config=self.config,
                    receipt=self.receipt,
                )
            delegated.assert_not_called()

        self.model_path.write_bytes(b"exact-v8-adapter-fixture")

        def mutate(**kwargs):
            del kwargs
            self.model_path.write_bytes(b"changed-during-load")
            return object(), 92, 46

        with mock.patch.object(
            loader.rmc.v5, "_strict_load_adapter", side_effect=mutate
        ):
            with self.assertRaisesRegex(
                loader.FeasibleQuotientInferenceError, "changed during"
            ):
                loader.strict_load_adapter(
                    base_model=object(),
                    bundle=self.bundle,
                    adapter_config=self.config,
                    receipt=self.receipt,
                )

    def test_wrong_physical_counts_are_rejected(self):
        with mock.patch.object(
            loader.rmc.v5,
            "_strict_load_adapter",
            return_value=(object(), 90, 45),
        ):
            with self.assertRaisesRegex(
                loader.FeasibleQuotientInferenceError, "exact-46/92"
            ):
                loader.strict_load_adapter(
                    base_model=object(),
                    bundle=self.bundle,
                    adapter_config=self.config,
                    receipt=self.receipt,
                )


if __name__ == "__main__":
    unittest.main()
