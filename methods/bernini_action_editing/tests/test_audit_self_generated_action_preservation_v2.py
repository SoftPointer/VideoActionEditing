import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import audit_self_generated_action_preservation_v2 as audit
import train_self_generated_action_quotient_v1 as trainer


def require_artifact_stack():
    try:
        import torch
        from safetensors.torch import save_file
    except ModuleNotFoundError as error:
        raise unittest.SkipTest("torch+safetensors are unavailable") from error
    return torch, save_file


def adapter_config():
    return {
        "alora_invocation_tokens": None,
        "alpha_pattern": {},
        "arrow_config": None,
        "auto_mapping": {
            "base_model_class": "BerniniRendererModel",
            "parent_library": "bernini.models.renderer",
        },
        "base_model_name_or_path": "",
        "bias": "none",
        "corda_config": None,
        "ensure_weight_tying": False,
        "eva_config": None,
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": 8,
        "lora_bias": False,
        "lora_dropout": 0.0,
        "lora_ga_config": None,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "peft_version": "0.19.1",
        "qalora_group_size": 16,
        "r": 8,
        "rank_pattern": {},
        "revision": None,
        "target_modules": ["attn2.to_out.0", "attn2.to_q"],
        "target_parameters": None,
        "task_type": None,
        "trainable_token_indices": None,
        "use_bdlora": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }


class PhysicalArtifactAuditTests(unittest.TestCase):
    def test_exact_scope_config_sha_authorities_match_canonical_bytes(self):
        for scope, expected_sha256 in sorted(
            audit.EXPECTED_ADAPTER_CONFIG_SHA256_BY_SCOPE.items()
        ):
            config = adapter_config()
            config["target_modules"] = list(
                trainer.expected_peft_serialized_target_modules(scope)
            )
            self.assertEqual(
                hashlib.sha256(trainer.canonical(config) + b"\n").hexdigest(),
                expected_sha256,
            )

    def make_adapter(self, root, *, trained=True, config_mutator=None):
        torch, save_file = require_artifact_stack()
        adapter = root / "adapter"
        adapter.mkdir()
        targets = [
            "diff_dec.transformer.blocks.0.attn2.to_out.0",
            "diff_dec.transformer.blocks.0.attn2.to_q",
        ]
        tensors = {}
        for target in targets:
            tensors[f"base_model.model.{target}.lora_A.weight"] = torch.ones(8, 4)
            tensors[f"base_model.model.{target}.lora_B.weight"] = (
                torch.ones(4, 8) if trained else torch.zeros(4, 8)
            )
        save_file(
            tensors,
            str(adapter / "adapter_model.safetensors"),
            metadata={"format": "pt"},
        )
        config = adapter_config()
        if config_mutator is not None:
            config_mutator(config)
        (adapter / "adapter_config.json").write_bytes(
            trainer.canonical(config) + b"\n"
        )
        os.chmod(adapter / "adapter_model.safetensors", 0o444)
        os.chmod(adapter / "adapter_config.json", 0o444)
        os.chmod(adapter, 0o555)
        return adapter, targets

    def make_optimizer(self, root, *, step, tensor_shapes):
        torch, _ = require_artifact_stack()
        params = list(range(len(tensor_shapes)))
        state = {}
        if step:
            for parameter_id, shape in zip(params, tensor_shapes):
                state[parameter_id] = {
                    "step": torch.tensor(float(step), dtype=torch.float32),
                    "exp_avg": torch.ones(shape, dtype=torch.float32),
                    "exp_avg_sq": torch.ones(shape, dtype=torch.float32),
                }
        payload = {
            "global_step": step,
            "optimizer": {
                "state": state,
                "param_groups": [
                    {
                        "lr": 1.0e-4,
                        "betas": (0.9, 0.999),
                        "eps": 1.0e-8,
                        "weight_decay": trainer.V2_WEIGHT_DECAY,
                        "amsgrad": False,
                        "maximize": False,
                        "foreach": None,
                        "capturable": False,
                        "differentiable": False,
                        "fused": None,
                        "decoupled_weight_decay": True,
                        "params": params,
                    }
                ],
            },
        }
        path = root / "optimizer.pt"
        torch.save(payload, path)
        os.chmod(path, 0o444)
        return path

    def test_physical_adapter_and_optimizer_positive_path(self):
        for step, trained in ((0, False), (5, True)):
            with self.subTest(step=step), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                adapter, targets = self.make_adapter(root, trained=trained)
                try:
                    with mock.patch.object(audit, "LORA_WIDTH", 4):
                        evidence = audit.validate_adapter_artifacts(
                            adapter=adapter,
                            targets=targets,
                            route_scope="cross_attn2_qo",
                            step=step,
                        )
                        self.assertEqual(
                            evidence["config_sha256"],
                            audit.EXPECTED_ADAPTER_CONFIG_SHA256_BY_SCOPE[
                                "cross_attn2_qo"
                            ],
                        )
                        self.assertEqual(
                            evidence["config_sha256"],
                            hashlib.sha256(
                                trainer.canonical(adapter_config()) + b"\n"
                            ).hexdigest(),
                        )
                        optimizer = self.make_optimizer(
                            root,
                            step=step,
                            tensor_shapes=evidence["tensor_shapes"],
                        )
                        digest = audit.validate_optimizer_artifact(
                            path=optimizer,
                            step=step,
                            arm="v2_func025_cross_qo",
                            tensor_count=evidence["tensor_count"],
                            tensor_shapes=evidence["tensor_shapes"],
                        )
                        self.assertEqual(len(digest), 64)
                finally:
                    os.chmod(adapter, 0o755)

    def test_auditor_rejects_target_duplicate_missing_unknown_and_order_tamper(self):
        cases = {
            "duplicate": ["attn2.to_out.0", "attn2.to_out.0"],
            "missing": ["attn2.to_q"],
            "unknown": ["attn2.to_out.0", "attn2.to_k"],
            "order": ["attn2.to_q", "attn2.to_out.0"],
        }
        for label, serialized_targets in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                adapter, targets = self.make_adapter(
                    root,
                    trained=True,
                    config_mutator=lambda value, serialized_targets=serialized_targets: value.update(
                        {"target_modules": serialized_targets}
                    ),
                )
                try:
                    with mock.patch.object(
                        audit, "LORA_WIDTH", 4
                    ), self.assertRaisesRegex(
                        audit.PreservationAuditError,
                        "serialized target scope/order differs",
                    ):
                        audit.validate_adapter_artifacts(
                            adapter=adapter,
                            targets=targets,
                            route_scope="cross_attn2_qo",
                            step=5,
                        )
                finally:
                    os.chmod(adapter, 0o755)

    def test_auditor_requires_exact_canonical_config_bytes_and_scope_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter, targets = self.make_adapter(root, trained=True)
            config_path = adapter / "adapter_config.json"
            os.chmod(adapter, 0o755)
            os.chmod(config_path, 0o600)
            config_path.write_text(
                json.dumps(adapter_config(), indent=2) + "\n", encoding="utf-8"
            )
            os.chmod(config_path, 0o444)
            os.chmod(adapter, 0o555)
            try:
                with mock.patch.object(
                    audit, "LORA_WIDTH", 4
                ), self.assertRaisesRegex(
                    audit.PreservationAuditError,
                    "canonical byte closure differs",
                ):
                    audit.validate_adapter_artifacts(
                        adapter=adapter,
                        targets=targets,
                        route_scope="cross_attn2_qo",
                        step=5,
                    )
            finally:
                os.chmod(adapter, 0o755)

    def test_auditor_rejects_surviving_peft_readme_and_unknown_extra(self):
        for extra in ("README.md", "unknown.bin"):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                adapter = Path(directory) / "adapter"
                adapter.mkdir()
                (adapter / "adapter_config.json").write_bytes(b"{}")
                (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
                (adapter / extra).write_bytes(b"must not survive publication")
                os.chmod(adapter, 0o555)
                try:
                    with self.assertRaisesRegex(
                        audit.PreservationAuditError, "adapter entry closure"
                    ):
                        audit.validate_adapter_artifacts(
                            adapter=adapter,
                            targets=(),
                            route_scope="cross_attn2_qo",
                            step=0,
                        )
                finally:
                    os.chmod(adapter, 0o755)

    def test_adapter_semantic_drift_and_zero_trained_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter, targets = self.make_adapter(
                root,
                trained=True,
                config_mutator=lambda value: value.update(
                    {"rank_pattern": {"to_q": 64}}
                ),
            )
            try:
                with mock.patch.object(audit, "LORA_WIDTH", 4), self.assertRaisesRegex(
                    audit.PreservationAuditError, "semantic closure"
                ):
                    audit.validate_adapter_artifacts(
                        adapter=adapter,
                        targets=targets,
                        route_scope="cross_attn2_qo",
                        step=5,
                    )
            finally:
                os.chmod(adapter, 0o755)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter, targets = self.make_adapter(root, trained=False)
            try:
                with mock.patch.object(audit, "LORA_WIDTH", 4), self.assertRaisesRegex(
                    audit.PreservationAuditError, "did not update"
                ):
                    audit.validate_adapter_artifacts(
                        adapter=adapter,
                        targets=targets,
                        route_scope="cross_attn2_qo",
                        step=5,
                    )
            finally:
                os.chmod(adapter, 0o755)

    def test_negative_second_moment_and_wrong_counter_dtype_are_rejected(self):
        torch, _ = require_artifact_stack()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_optimizer(root, step=5, tensor_shapes=[(2, 2)])
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1.0)
            os.chmod(path, 0o600)
            torch.save(payload, path)
            os.chmod(path, 0o444)
            with self.assertRaisesRegex(audit.PreservationAuditError, "moment values"):
                audit.validate_optimizer_artifact(
                    path=path,
                    step=5,
                    arm="v2_func025_cross_qo",
                    tensor_count=1,
                    tensor_shapes=[(2, 2)],
                )

            payload["optimizer"]["state"][0]["exp_avg_sq"].fill_(1.0)
            payload["optimizer"]["state"][0]["step"] = torch.tensor(
                5, dtype=torch.int64
            )
            os.chmod(path, 0o600)
            torch.save(payload, path)
            os.chmod(path, 0o444)
            with self.assertRaisesRegex(audit.PreservationAuditError, "step tensor"):
                audit.validate_optimizer_artifact(
                    path=path,
                    step=5,
                    arm="v2_func025_cross_qo",
                    tensor_count=1,
                    tensor_shapes=[(2, 2)],
                )

if __name__ == "__main__":
    unittest.main()
