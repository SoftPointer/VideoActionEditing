import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import self_generated_action_preservation_v2 as preservation
import audit_self_generated_action_preservation_v2 as preservation_audit
import train_self_generated_action_quotient_v1 as trainer


SEED = trainer.V2_CANARY_SEED
SHA = "a" * 64
REVISION = "b" * 40


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise unittest.SkipTest("torch is unavailable") from error
    return torch


def fake_checkpoint_torch():
    """Exercise the real publication filesystem path without a local torch wheel."""

    torch = types.ModuleType("torch")
    distributed = types.ModuleType("torch.distributed")
    distributed.is_initialized = lambda: False
    torch.distributed = distributed

    def save(_value, path):
        Path(path).write_bytes(b"deterministic optimizer fixture\n")

    torch.save = save
    return mock.patch.dict(
        sys.modules,
        {"torch": torch, "torch.distributed": distributed},
    )


def v2_publication_receipt(*, step=0, route_scope="cross_attn2_qo"):
    return {
        "schema_version": trainer.RECEIPT_SCHEMA_V2,
        "objective_family": "preservation_v2",
        "global_step": step,
        "training_contract": {"lora_route_scope": route_scope},
    }


def write_peft_config(path, target_modules):
    Path(path).write_text(
        json.dumps(
            {
                "producer_fixture": "peft-0.19.1",
                "target_modules": list(target_modules),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def args(**overrides):
    value = dict(
        objective_family="preservation_v2",
        arm="v2_func025_cross_qo",
        max_steps=20,
        method_source_archive_sha256=SHA,
        method_source_revision=REVISION,
        seed=SEED,
        slots=5,
        limit_cells=0,
        source_manifest_sha256="c" * 64,
    )
    value.update(overrides)
    return argparse.Namespace(**value)


def cache_fixture():
    torch = require_torch()
    sigma_values = (0.10, 0.30, 0.50, 0.70, 0.90)
    cells = []
    for row_index in range(4):
        for slot, sigma in enumerate(sigma_values):
            trial = slot + 10 * row_index
            velocity = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
            velocity[:, 3, 1] = 1.0
            source_noop = torch.zeros(1, 21, 32)
            teacher = torch.zeros(1, 21, 32)
            camera = torch.zeros(1, 21, 32)
            appearance = torch.zeros(1, 21, 32)
            teacher[:, 1, 0] = 1.0
            camera[:, 1, 1] = 1.0
            appearance[:, 1, 2] = 1.0
            floor = torch.ones(1)
            cells.append(
                {
                    "iid": f"iid-{row_index}",
                    "row_index": row_index,
                    "slot": slot,
                    "seed": trainer.legacy.step_seed(SEED, trial, row_index),
                    "seed_trial": trial,
                    "sigma_bin": slot,
                    "sigma": sigma,
                    "source_state_digest": "d" * 64 + ":0.5",
                    "source_noop_raw": source_noop.clone(),
                    "teacher_unit": teacher.clone(),
                    "camera_unit": camera.clone(),
                    "appearance_unit": appearance.clone(),
                    "amplitude_floor": floor,
                    "source_amplitude": 1.0,
                    "teacher_amplitude": 2.0,
                    "frozen_source_action_velocity": velocity,
                    "frozen_source_action_velocity_sha256": trainer.tensor_sha(velocity),
                }
            )
    return {
        "slots": 5,
        "sigma_bins": [list(item) for item in trainer.V2_SIGMA_BINS],
        "cells": cells,
    }


class ContractTests(unittest.TestCase):
    def test_v2_contract_is_fresh_exact_and_does_not_change_legacy_seed(self):
        self.assertEqual(trainer.require_objective_contract(args()), "preservation_v2")
        for field, value, message in (
            ("seed", SEED + 1, "initialization/cache seed"),
            ("slots", 4, "five stratified"),
            ("max_steps", 40, "20 optimizer"),
            ("limit_cells", 1, "partial"),
            ("arm", "action_only", "v2 objective arm"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                trainer.QuotientTrainingError, message
            ):
                trainer.require_objective_contract(args(**{field: value}))
        trainer.require_replication_seed(20260817)

    def test_sigma_bins_are_exact_and_finite(self):
        for index, value in enumerate((0.0, 0.2, 0.4, 0.6, 0.8)):
            self.assertEqual(trainer.sigma_bin_index(value), index)
        self.assertEqual(trainer.sigma_bin_index(1.0), 4)
        with self.assertRaisesRegex(trainer.QuotientTrainingError, "non-finite"):
            trainer.sigma_bin_index(float("nan"))

    def test_v2_cache_is_exact_stratified_and_sha_bound(self):
        torch = require_torch()
        cache = cache_fixture()
        cells, by_key = trainer.validate_teacher_cache_cells_v2(
            cache, expected_seed=SEED
        )
        self.assertEqual(len(cells), 20)
        self.assertEqual(
            set(by_key),
            {(row_index, slot) for row_index in range(4) for slot in range(5)},
        )
        tampered = copy.deepcopy(cache)
        tampered["cells"][0]["frozen_source_action_velocity"].view(-1)[0] += 1
        with self.assertRaisesRegex(
            trainer.QuotientTrainingError, "velocity authority differs"
        ):
            trainer.validate_teacher_cache_cells_v2(tampered, expected_seed=SEED)
        resigned_semantic_tamper = copy.deepcopy(cache)
        resigned_semantic_tamper["cells"][0]["source_amplitude"] = 123.0
        resigned_semantic_tamper["cells"][0]["amplitude_floor"] = torch.tensor(
            [123.0], dtype=torch.float32
        )
        with self.assertRaisesRegex(
            trainer.QuotientTrainingError, "source amplitude differs"
        ):
            trainer.validate_teacher_cache_cells_v2(
                resigned_semantic_tamper, expected_seed=SEED
            )

    def test_v2_payload_and_receipt_do_not_claim_decoded_preservation(self):
        namespace = args()
        manifest = {"manifest_digest": "d" * 64}
        payload = trainer.teacher_cache_payload(
            args=namespace, manifest=manifest, cells=[]
        )
        receipt = trainer.teacher_cache_receipt(
            args=namespace,
            manifest=manifest,
            cell_count=20,
            cache_sha256="e" * 64,
        )
        self.assertEqual(payload["schema_version"], trainer.CACHE_SCHEMA_V2)
        self.assertEqual(receipt["schema_version"], trainer.CACHE_SCHEMA_V2)
        self.assertEqual(payload["sigma_bins"], [list(x) for x in trainer.V2_SIGMA_BINS])
        self.assertFalse(payload["decoded_identity_background_camera_claim_authorized"])
        self.assertFalse(receipt["decoded_identity_background_camera_claim_authorized"])

    def test_v2_checkpoint_binds_route_losses_and_human_gate(self):
        components = {
            "action": 1.0,
            "onset": 2.0,
            "nuisance": 3.0,
            "noop": 4.0,
            "functional_code": 5.0,
            "functional_temporal_dc": 6.0,
            "functional_total": 11.0,
        }
        targets = (
            "diff_dec.transformer.blocks.0.attn2.to_out.0",
            "diff_dec.transformer.blocks.0.attn2.to_q",
        )
        receipt = trainer.checkpoint_receipt(
            args=args(),
            manifest={"manifest_digest": "d" * 64},
            step=5,
            loss=9.0,
            grad_norm=2.0,
            target_modules=targets,
            trainable_count=3,
            bernini_revision="1" * 40,
            veomni_revision="2" * 40,
            transformers_version="test",
            initial_digest="3" * 64,
            teacher_cache_seed=SEED,
            teacher_cache_sha256="e" * 64,
            loss_components=components,
        )
        self.assertEqual(receipt["schema_version"], trainer.RECEIPT_SCHEMA_V2)
        self.assertEqual(receipt["target_modules"], list(targets))
        self.assertEqual(receipt["last_loss_components"], components)
        contract = receipt["training_contract"]
        self.assertEqual(contract["lora_route_scope"], "cross_attn2_qo")
        self.assertFalse(contract["decoded_identity_background_camera_claim_authorized"])
        self.assertTrue(contract["blind_full_video_review_required_for_promotion"])
        self.assertFalse(receipt["automatic_scientific_promotion_authorized"])
        self.assertEqual(
            receipt["optimizer"]["weight_decay"], trainer.V2_WEIGHT_DECAY
        )

    def test_auditor_rejects_fully_resigned_checkpoint_overclaim_and_zero_update(self):
        targets = preservation_audit.expected_targets("cross_attn2_qo")
        components = {
            "action": 1.0,
            "onset": 2.0,
            "nuisance": 3.0,
            "noop": 4.0,
            "functional_code": 5.0,
            "functional_temporal_dc": 6.0,
            "functional_total": 11.0,
        }
        receipt = trainer.checkpoint_receipt(
            args=args(),
            manifest={"manifest_digest": preservation_audit.SOURCE_MANIFEST_DIGEST},
            step=5,
            loss=4.55,
            grad_norm=2.0,
            target_modules=targets,
            trainable_count=len(targets) * 2 * preservation_audit.LORA_RANK
            * preservation_audit.LORA_WIDTH,
            bernini_revision=preservation_audit.BERNINI_COMMIT,
            veomni_revision=preservation_audit.VEOMNI_COMMIT,
            transformers_version=preservation_audit.TRANSFORMERS_VERSION,
            initial_digest="3" * 64,
            teacher_cache_seed=SEED,
            teacher_cache_sha256="e" * 64,
            loss_components=components,
        )
        preservation_audit.validate_digest(receipt, label="fixture")
        preservation_audit.validate_checkpoint_receipt(
            receipt,
            arm="v2_func025_cross_qo",
            step=5,
            cache_sha256="e" * 64,
            source_manifest_sha256="c" * 64,
            method_source_revision=REVISION,
            method_source_archive_sha256=SHA,
            targets=targets,
        )

        overclaim = copy.deepcopy(receipt)
        overclaim["scientific_claim_authorized"] = True
        overclaim.pop("receipt_digest")
        overclaim["receipt_digest"] = trainer.legacy.object_sha256(overclaim)
        with self.assertRaisesRegex(
            preservation_audit.PreservationAuditError, "scientific overclaim"
        ):
            preservation_audit.validate_checkpoint_receipt(
                overclaim,
                arm="v2_func025_cross_qo",
                step=5,
                cache_sha256="e" * 64,
                source_manifest_sha256="c" * 64,
                method_source_revision=REVISION,
                method_source_archive_sha256=SHA,
                targets=targets,
            )

        zero_update = copy.deepcopy(receipt)
        zero_update["last_loss"] = 0.0
        zero_update["last_preclip_gradient_norm"] = 0.0
        zero_update["last_loss_components"] = {
            key: 0.0 for key in zero_update["last_loss_components"]
        }
        zero_update.pop("receipt_digest")
        zero_update["receipt_digest"] = trainer.legacy.object_sha256(zero_update)
        with self.assertRaisesRegex(
            preservation_audit.PreservationAuditError, "positive update evidence"
        ):
            preservation_audit.validate_checkpoint_receipt(
                zero_update,
                arm="v2_func025_cross_qo",
                step=5,
                cache_sha256="e" * 64,
                source_manifest_sha256="c" * 64,
                method_source_revision=REVISION,
                method_source_archive_sha256=SHA,
                targets=targets,
            )

    def test_v2_optimizer_contract_does_not_use_adamw_default_decay(self):
        self.assertEqual(trainer.V2_WEIGHT_DECAY, 0.0)
        source = Path(trainer.__file__).read_text(encoding="utf-8")
        self.assertIn(
            'weight_decay=V2_WEIGHT_DECAY if family == "preservation_v2" else 0.01',
            source,
        )

    def test_finite_scalar_gate_precedes_mutation_contract(self):
        torch = require_torch()
        trainer.require_finite_scalar_all_ranks(torch.tensor(1.0), label="objective")
        with self.assertRaisesRegex(trainer.QuotientTrainingError, "non-finite"):
            trainer.require_finite_scalar_all_ranks(
                torch.tensor(float("nan")), label="objective"
            )

    def test_checkpoint0_and_trained_publication_remove_readme_and_seal_exact_tree(self):
        class Model:
            @staticmethod
            def save_pretrained(path, *, safe_serialization):
                self.assertTrue(safe_serialization)
                path = Path(path)
                path.mkdir()
                write_peft_config(
                    path / "adapter_config.json",
                    ["attn2.to_q", "attn2.to_out.0"],
                )
                (path / "adapter_model.safetensors").write_bytes(b"safetensors")
                (path / "README.md").write_text(
                    "PEFT generated model card fixture\n", encoding="utf-8"
                )

        class Optimizer:
            @staticmethod
            def state_dict():
                return {"state": {}, "param_groups": []}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "arm"
            output.mkdir(mode=0o700)
            checkpoints = [
                output / "checkpoint-00000000",
                output / "checkpoint-00000005",
            ]
            try:
                with fake_checkpoint_torch():
                    for step in (0, 5):
                        trainer.save_checkpoint(
                            model=Model(),
                            optimizer=Optimizer(),
                            output=output,
                            step=step,
                            receipt=v2_publication_receipt(step=step),
                            rank=0,
                        )
                for checkpoint in checkpoints:
                    self.assertEqual(
                        {path.name for path in checkpoint.iterdir()},
                        trainer.CHECKPOINT_ENTRY_NAMES,
                    )
                    adapter = checkpoint / "adapter"
                    self.assertEqual(
                        {path.name for path in adapter.iterdir()},
                        trainer.CHECKPOINT_ADAPTER_ENTRY_NAMES,
                    )
                    self.assertNotIn("README.md", {path.name for path in adapter.iterdir()})
                    self.assertEqual(
                        (adapter / "adapter_config.json").read_bytes(),
                        trainer.canonical(
                            {
                                "producer_fixture": "peft-0.19.1",
                                "target_modules": [
                                    "attn2.to_out.0",
                                    "attn2.to_q",
                                ],
                            }
                        )
                        + b"\n",
                    )
                    self.assertEqual(os.stat(checkpoint).st_mode & 0o777, 0o555)
                    self.assertEqual(os.stat(adapter).st_mode & 0o777, 0o555)
                    for path in (
                        checkpoint / "optimizer.pt",
                        checkpoint / "receipt.json",
                        adapter / "adapter_config.json",
                        adapter / "adapter_model.safetensors",
                    ):
                        self.assertEqual(os.stat(path).st_mode & 0o777, 0o444)
                        self.assertGreater(path.stat().st_size, 0)
            finally:
                for checkpoint in checkpoints:
                    if checkpoint.exists():
                        os.chmod(checkpoint, 0o755)
                        adapter = checkpoint / "adapter"
                        if adapter.exists():
                            os.chmod(adapter, 0o755)

    def test_checkpoint_publication_with_real_torch_serializer_accepts_peft_shape(self):
        require_torch()

        class Model:
            @staticmethod
            def save_pretrained(path, *, safe_serialization):
                self.assertTrue(safe_serialization)
                path = Path(path)
                path.mkdir()
                write_peft_config(
                    path / "adapter_config.json",
                    ["attn2.to_q", "attn2.to_out.0"],
                )
                (path / "adapter_model.safetensors").write_bytes(b"safetensors")
                (path / "README.md").write_text(
                    "PEFT generated model card fixture\n", encoding="utf-8"
                )

        class Optimizer:
            @staticmethod
            def state_dict():
                return {"state": {}, "param_groups": []}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "arm"
            output.mkdir(mode=0o700)
            checkpoint = output / "checkpoint-00000000"
            try:
                trainer.save_checkpoint(
                    model=Model(), optimizer=Optimizer(), output=output,
                    step=0, receipt=v2_publication_receipt(step=0), rank=0,
                )
                self.assertEqual(
                    {path.name for path in checkpoint.iterdir()},
                    trainer.CHECKPOINT_ENTRY_NAMES,
                )
                self.assertEqual(
                    {path.name for path in (checkpoint / "adapter").iterdir()},
                    trainer.CHECKPOINT_ADAPTER_ENTRY_NAMES,
                )
            finally:
                if checkpoint.exists():
                    os.chmod(checkpoint, 0o755)
                    adapter = checkpoint / "adapter"
                    if adapter.exists():
                        os.chmod(adapter, 0o755)

    def test_peft_target_canonicalization_is_process_hash_order_independent(self):
        script = """
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import train_self_generated_action_quotient_v1 as trainer
adapter = Path(sys.argv[2])
adapter.mkdir()
targets = list({"to_q", "to_k", "to_v", "to_out.0"})
(adapter / "adapter_config.json").write_text(
    json.dumps({"producer_fixture": "peft-0.19.1", "target_modules": targets}, indent=2) + "\\n",
    encoding="utf-8",
)
trainer._canonicalize_peft_adapter_config(adapter, "all_attention")
print(json.dumps({"input": targets, "output": (adapter / "adapter_config.json").read_text(encoding="utf-8")}))
"""
        rows = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in range(1, 9):
                environment = dict(os.environ)
                environment["PYTHONHASHSEED"] = str(seed)
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-c",
                        script,
                        str(METHOD_ROOT),
                        str(root / f"adapter-{seed}"),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                rows.append(json.loads(completed.stdout))
        self.assertGreater(len({tuple(row["input"]) for row in rows}), 1)
        self.assertEqual(len({row["output"] for row in rows}), 1)
        self.assertEqual(
            rows[0]["output"].encode("utf-8"),
            trainer.canonical(
                {
                    "producer_fixture": "peft-0.19.1",
                    "target_modules": ["to_k", "to_out.0", "to_q", "to_v"],
                }
            )
            + b"\n",
        )

    def test_peft_target_canonicalization_rejects_duplicate_missing_and_unknown(self):
        cases = {
            "duplicate": ["attn2.to_out.0", "attn2.to_out.0"],
            "missing": ["attn2.to_q"],
            "unknown": ["attn2.to_out.0", "attn2.to_k"],
        }
        for label, targets in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                adapter = Path(directory) / "adapter"
                adapter.mkdir()
                write_peft_config(adapter / "adapter_config.json", targets)
                with self.assertRaisesRegex(
                    trainer.QuotientTrainingError,
                    "serialized target scope differs",
                ):
                    trainer._canonicalize_peft_adapter_config(
                        adapter, "cross_attn2_qo"
                    )

    def test_peft_config_accepts_vast_nlink_lag_without_open_unlink_silly_rename(self):
        class NlinkView:
            def __init__(self, original, nlink):
                self._original = original
                self.st_nlink = nlink

            def __getattr__(self, name):
                return getattr(self._original, name)

        state = {"unlink_stage": 0, "silly_renames": 0}
        open_inodes = {}
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            adapter.mkdir()
            config_path = adapter / "adapter_config.json"
            write_peft_config(
                config_path,
                ["attn2.to_q", "attn2.to_out.0"],
            )
            original_inode = config_path.stat().st_ino
            original_fstat = os.fstat
            original_stat = os.stat
            original_open = os.open
            original_close = os.close
            original_unlink = os.unlink

            def lagged(value):
                if value.st_ino != original_inode:
                    return value
                if state["unlink_stage"] == 1:
                    return NlinkView(value, 2)
                if state["unlink_stage"] == 2:
                    return NlinkView(value, 1)
                return value

            def lagged_fstat(descriptor):
                return lagged(original_fstat(descriptor))

            def lagged_stat(path, *arguments, **kwargs):
                return lagged(original_stat(path, *arguments, **kwargs))

            def tracked_open(path, flags, *arguments, **kwargs):
                descriptor = original_open(path, flags, *arguments, **kwargs)
                open_inodes[descriptor] = original_fstat(descriptor).st_ino
                return descriptor

            def tracked_close(descriptor):
                open_inodes.pop(descriptor, None)
                return original_close(descriptor)

            def tracked_unlink(path, *arguments, **kwargs):
                target = original_stat(path, *arguments, **kwargs)
                if target.st_ino in open_inodes.values():
                    state["silly_renames"] += 1
                    directory_descriptor = kwargs.get("dir_fd")
                    os.rename(
                        path,
                        f".nfs-hostile-{state['silly_renames']}",
                        src_dir_fd=directory_descriptor,
                        dst_dir_fd=directory_descriptor,
                    )
                    return None
                result = original_unlink(path, *arguments, **kwargs)
                if path == "adapter_config.json":
                    state["unlink_stage"] = 1
                elif str(path).startswith(".adapter_config.json.original-"):
                    state["unlink_stage"] = 2
                return result

            with mock.patch.object(
                trainer.os, "fstat", side_effect=lagged_fstat
            ), mock.patch.object(
                trainer.os, "stat", side_effect=lagged_stat
            ), mock.patch.object(
                trainer.os, "open", side_effect=tracked_open
            ), mock.patch.object(
                trainer.os, "close", side_effect=tracked_close
            ), mock.patch.object(
                trainer.os, "unlink", side_effect=tracked_unlink
            ):
                trainer._canonicalize_peft_adapter_config(
                    adapter, "cross_attn2_qo"
                )
            self.assertEqual(state["unlink_stage"], 2)
            self.assertEqual(state["silly_renames"], 0)
            self.assertEqual(
                config_path.read_bytes(),
                trainer.canonical(
                    {
                        "producer_fixture": "peft-0.19.1",
                        "target_modules": [
                            "attn2.to_out.0",
                            "attn2.to_q",
                        ],
                    }
                )
                + b"\n",
            )
            self.assertEqual({path.name for path in adapter.iterdir()}, {
                "adapter_config.json"
            })

    def test_peft_config_rejects_symlink_hardlink_and_path_replacement(self):
        for topology in ("symlink", "hardlink"):
            with self.subTest(topology=topology), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                adapter = root / "adapter"
                adapter.mkdir()
                sentinel = root / "sentinel.json"
                write_peft_config(
                    sentinel, ["attn2.to_out.0", "attn2.to_q"]
                )
                if topology == "symlink":
                    os.symlink(sentinel, adapter / "adapter_config.json")
                else:
                    os.link(sentinel, adapter / "adapter_config.json")
                with self.assertRaisesRegex(
                    trainer.QuotientTrainingError,
                    "adapter config topology differs",
                ):
                    trainer._canonicalize_peft_adapter_config(
                        adapter, "cross_attn2_qo"
                    )
                self.assertTrue(sentinel.is_file())

        state = {"raced": False}
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            adapter.mkdir()
            write_peft_config(
                adapter / "adapter_config.json",
                ["attn2.to_q", "attn2.to_out.0"],
            )
            original_open = os.open

            def racing_open(path, flags, *arguments, **kwargs):
                if (
                    path == "adapter_config.json"
                    and flags & os.O_CREAT
                    and flags & os.O_EXCL
                    and not state["raced"]
                ):
                    state["raced"] = True
                    descriptor = original_open(
                        "adapter_config.json",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=kwargs["dir_fd"],
                    )
                    try:
                        os.write(descriptor, b'{"target_modules":[]}\n')
                    finally:
                        os.close(descriptor)
                return original_open(path, flags, *arguments, **kwargs)

            with mock.patch.object(
                trainer.os, "open", side_effect=racing_open
            ), self.assertRaisesRegex(
                trainer.QuotientTrainingError,
                "target is not fresh",
            ):
                trainer._canonicalize_peft_adapter_config(
                    adapter, "cross_attn2_qo"
                )
            self.assertTrue(state["raced"])
            self.assertEqual(
                (adapter / "adapter_config.json").read_bytes(),
                b'{"target_modules":[]}\n',
            )

    def test_peft_config_rejects_rename_during_same_fd_capture(self):
        state = {"raced": False}
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            adapter.mkdir()
            write_peft_config(
                adapter / "adapter_config.json",
                ["attn2.to_q", "attn2.to_out.0"],
            )
            original_read = os.read

            def racing_read(descriptor, size):
                if not state["raced"]:
                    state["raced"] = True
                    os.rename(
                        adapter / "adapter_config.json",
                        adapter / "adapter_config.moved",
                    )
                    write_peft_config(
                        adapter / "adapter_config.json",
                        ["attn2.to_out.0", "attn2.to_q"],
                    )
                return original_read(descriptor, size)

            with mock.patch.object(
                trainer.os, "read", side_effect=racing_read
            ), self.assertRaisesRegex(
                trainer.QuotientTrainingError,
                "changed during stable capture",
            ):
                trainer._canonicalize_peft_adapter_config(
                    adapter, "cross_attn2_qo"
                )
            self.assertTrue(state["raced"])

    def test_checkpoint_rejects_symlink_and_hardlink_peft_readme(self):
        class Optimizer:
            @staticmethod
            def state_dict():
                return {"state": {}, "param_groups": []}

        for topology in ("symlink", "hardlink"):
            with self.subTest(topology=topology), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "arm"
                output.mkdir(mode=0o700)
                sentinel = root / "sentinel.txt"
                sentinel.write_text("must survive\n", encoding="utf-8")

                class Model:
                    @staticmethod
                    def save_pretrained(path, *, safe_serialization):
                        self.assertTrue(safe_serialization)
                        path = Path(path)
                        path.mkdir()
                        write_peft_config(
                            path / "adapter_config.json",
                            ["attn2.to_q", "attn2.to_out.0"],
                        )
                        (path / "adapter_model.safetensors").write_bytes(
                            b"safetensors"
                        )
                        if topology == "symlink":
                            os.symlink(sentinel, path / "README.md")
                        else:
                            os.link(sentinel, path / "README.md")

                with fake_checkpoint_torch(), self.assertRaisesRegex(
                    trainer.QuotientTrainingError, "PEFT README topology"
                ):
                    trainer.save_checkpoint(
                        model=Model(),
                        optimizer=Optimizer(),
                        output=output,
                        step=0,
                        receipt=v2_publication_receipt(step=0),
                        rank=0,
                    )
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "must survive\n")
                self.assertFalse((output / "checkpoint-00000000").exists())

    def test_checkpoint_rejects_unknown_extra_without_silently_removing_it(self):
        class Model:
            @staticmethod
            def save_pretrained(path, *, safe_serialization):
                path = Path(path)
                path.mkdir()
                write_peft_config(
                    path / "adapter_config.json",
                    ["attn2.to_q", "attn2.to_out.0"],
                )
                (path / "adapter_model.safetensors").write_bytes(b"safetensors")
                (path / "README.md").write_text("generated\n", encoding="utf-8")
                (path / "unknown.bin").write_bytes(b"not authorized")

        class Optimizer:
            @staticmethod
            def state_dict():
                return {}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "arm"
            output.mkdir()
            with fake_checkpoint_torch(), self.assertRaisesRegex(
                trainer.QuotientTrainingError,
                "entry closure differs before README removal",
            ):
                trainer.save_checkpoint(
                    model=Model(), optimizer=Optimizer(), output=output,
                    step=0, receipt=v2_publication_receipt(step=0), rank=0,
                )
            staging = output / f".checkpoint-00000000.tmp-{os.getpid()}" / "adapter"
            self.assertTrue((staging / "README.md").is_file())
            self.assertTrue((staging / "unknown.bin").is_file())

    def test_checkpoint_rejects_readme_rename_during_same_fd_capture(self):
        state = {"adapter": None, "raced": False}

        class Model:
            @staticmethod
            def save_pretrained(path, *, safe_serialization):
                path = Path(path)
                path.mkdir()
                state["adapter"] = path
                (path / "adapter_config.json").write_text("{}", encoding="utf-8")
                (path / "adapter_model.safetensors").write_bytes(b"safetensors")
                (path / "README.md").write_text("captured\n", encoding="utf-8")

        class Optimizer:
            @staticmethod
            def state_dict():
                return {}

        original_read = os.read

        def racing_read(descriptor, size):
            if not state["raced"]:
                state["raced"] = True
                adapter = state["adapter"]
                assert adapter is not None
                os.rename(adapter / "README.md", adapter / "README.moved")
                (adapter / "README.md").write_text("replacement\n", encoding="utf-8")
            return original_read(descriptor, size)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "arm"
            output.mkdir()
            with fake_checkpoint_torch(), mock.patch.object(
                trainer.os, "read", side_effect=racing_read
            ), self.assertRaisesRegex(
                trainer.QuotientTrainingError, "README changed during stable capture"
            ):
                trainer.save_checkpoint(
                    model=Model(), optimizer=Optimizer(), output=output,
                    step=0, receipt={"receipt": "fixture"}, rank=0,
                )
            self.assertTrue(state["raced"])
            self.assertFalse((output / "checkpoint-00000000").exists())


if __name__ == "__main__":
    unittest.main()
