from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

try:
    import torch
except ImportError as error:  # pragma: no cover
    raise unittest.SkipTest("torch unavailable") from error


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import probe_source_owned_role_locator_v15_sp4 as probe  # noqa: E402


class FrozenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.eval()


class Adapter:
    def __init__(self, **overrides):
        self.model = FrozenModel()
        self._overrides = overrides

    def observer_contract(self):
        value = {
            "schema_version": probe.ADAPTER_SCHEMA_VERSION,
            "checkpoint_sha256": "1" * 64,
            "source_manifest_sha256": "2" * 64,
            "source_is_real_video": True,
            "frozen_base": True,
            "eval_mode": True,
            "adapters_disabled": True,
            "ulysses_group_is_world": True,
            "world_size": 4,
            "selected_block_indices": [4, 9, 14, 19, 24],
            "observer_only": True,
            "training_authorized": False,
            "route_authorized": False,
        }
        value.update(self._overrides)
        return value

    def materialize_source(self, **_kwargs):
        raise AssertionError("unit contract test does not materialize")

    def prepare_inputs_for_sp(self, **_kwargs):
        raise AssertionError("unit contract test does not prepare")

    def run_frozen_forward(self, **_kwargs):
        raise AssertionError("unit contract test does not forward")


class SP4ProbeContractTests(unittest.TestCase):
    def test_adapter_contract_requires_frozen_observer_only_sp4(self):
        blocks = (4, 9, 14, 19, 24)
        accepted = probe.validate_adapter_contract(Adapter(), blocks)
        self.assertTrue(accepted["observer_only"])
        for bad in (
            {"world_size": 8},
            {"adapters_disabled": False},
            {"source_is_real_video": False},
            {"training_authorized": True},
            {"route_authorized": True},
            {"selected_block_indices": [4]},
        ):
            with self.assertRaises(probe.SourceRoleSP4ProbeError):
                probe.validate_adapter_contract(Adapter(**bad), blocks)

    def test_adapter_contract_rejects_trainable_or_training_model(self):
        trainable = Adapter()
        trainable.model.weight.requires_grad_(True)
        with self.assertRaises(probe.SourceRoleSP4ProbeError):
            probe.validate_adapter_contract(trainable, (4, 9, 14, 19, 24))
        training = Adapter()
        training.model.train()
        with self.assertRaises(probe.SourceRoleSP4ProbeError):
            probe.validate_adapter_contract(training, (4, 9, 14, 19, 24))

    def test_source_materialization_is_exact_and_source_receipted(self):
        geometry = probe.locator.SourceVisualGeometry(height=1, width=2)
        value = {
            "tokenizer": object(),
            "tokenizer_dir": Path("/tmp/tokenizer"),
            "raw_source_text_hidden_states": torch.zeros((1, 512, 4)),
            "derive_conditioned_source_text": lambda tensor: tensor,
            "renderer_text_length": 512,
            "geometry": geometry,
            "source_receipt_sha256": "3" * 64,
        }
        materialized = probe.SourceMaterialization.from_adapter(value)
        self.assertEqual(materialized.geometry, geometry)
        with self.assertRaises(probe.SourceRoleSP4ProbeError):
            probe.SourceMaterialization.from_adapter({**value, "action_anchor": object()})
        with self.assertRaises(probe.SourceRoleSP4ProbeError):
            probe.SourceMaterialization.from_adapter(
                {**value, "source_receipt_sha256": "not-a-sha"}
            )

    def test_parser_requires_adapter_and_create_only_output(self):
        parser = probe.build_parser()
        args = parser.parse_args(
            ["--runtime-adapter", "runtime:factory", "--output", "/tmp/v15.json"]
        )
        self.assertEqual(args.block_indices, [4, 9, 14, 19, 24])
        self.assertEqual(args.event_id, "pour-liquid-into-cup")
        with self.assertRaises(probe.SourceRoleSP4ProbeError):
            probe._load_factory("missing-colon")

    def test_probe_source_has_no_training_route_or_decoder_execution(self):
        source_path = METHOD_ROOT / "probe_source_owned_role_locator_v15_sp4.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn("backward", called_attributes)
        self.assertNotIn("step", called_attributes)
        self.assertNotIn("zero_grad", called_attributes)
        self.assertNotIn("decode", called_attributes)
        self.assertNotIn("optimizer", source.lower())
        self.assertIn("with torch.inference_mode(), context:", source)
        self.assertLess(
            source.index("with torch.inference_mode(), context:"),
            source.index("adapter.prepare_inputs_for_sp("),
        )

    def test_distributed_preflight_fails_without_initialized_torchrun(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.skipTest("test process already owns a process group")
        with self.assertRaises(probe.SourceRoleSP4ProbeError):
            probe._distributed_preflight()


if __name__ == "__main__":
    unittest.main()
