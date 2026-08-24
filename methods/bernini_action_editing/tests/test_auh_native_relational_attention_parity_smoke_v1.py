#!/usr/bin/env python3

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import auh_native_relational_attention_parity_smoke_v1 as smoke  # noqa: E402
import auh_source_owned_role_locator_v15_adapter as site  # noqa: E402
import source_owned_role_locator_v15b_e00_asset as role_asset  # noqa: E402


class AUHNativeRelationalParityStaticTests(unittest.TestCase):
    def test_gather_second_allocation_fault_scrubs_payload_and_first_flat(self) -> None:
        qk_local = torch.ones((2, 1, 8, 3, 128), dtype=torch.float32)
        proxy_local = torch.ones((1, 2, 3), dtype=torch.float32)

        class Shard:
            block_index = 6

            def collective_payload_and_zeroize(self):
                return qk_local, proxy_local, {"digest": "test"}

        real_empty = torch.empty
        allocations = []

        def fail_second_empty(*args, **kwargs):
            if allocations:
                raise RuntimeError("second allocation fault")
            value = real_empty(*args, **kwargs)
            value.fill_(1.0)
            allocations.append(value)
            return value

        with mock.patch.object(smoke.torch, "empty", side_effect=fail_second_empty):
            with self.assertRaisesRegex(RuntimeError, "second allocation fault"):
                smoke._gather_one_block_after_forward(
                    Shard(),
                    invocation=None,
                    role_partition=None,
                )
        self.assertEqual(int(torch.count_nonzero(qk_local)), 0)
        self.assertEqual(int(torch.count_nonzero(proxy_local)), 0)
        self.assertEqual(len(allocations), 1)
        self.assertEqual(int(torch.count_nonzero(allocations[0])), 0)

    def test_contract_is_exact_world4_frozen_observer_only(self) -> None:
        value = smoke.parity_smoke_contract()
        self.assertEqual(value["world_size"], 4)
        self.assertEqual(value["blocks"], [6, 12, 18, 24])
        self.assertEqual(value["forward_order"], ["observer_off", "observer_on"])
        self.assertTrue(value["same_prepared_tensor_objects"])
        self.assertTrue(value["output_bit_exact_required"])
        self.assertEqual(
            value["added_collective_location"],
            "after_observer_on_transformer_forward_returned",
        )
        self.assertEqual(value["added_collectives_inside_attention"], 0)
        self.assertFalse(value["backend_attention_weights_observed"])
        self.assertFalse(value["persistent_tensor_artifact_authorized"])
        self.assertFalse(value["target_inputs_consumed"])
        self.assertFalse(value["training_authorized"])
        self.assertFalse(value["gpu_launch_authorized_by_contract_print"])
        self.assertFalse(value["scientific_claim_authorized"])
        self.assertEqual(value["checkpoint_tree_sha256"], site.CHECKPOINT_TREE_SHA256)

    def test_e00_partition_is_exhaustive_with_explicit_null(self) -> None:
        event, _raw = role_asset.load_e00_v15b_asset()
        partition = smoke._role_partition_from_e00_event(event)
        self.assertEqual(
            partition.role_names,
            (*role_asset.ROLE_NAMES, smoke.NULL_ROLE),
        )
        self.assertEqual(len(partition.token_to_role), site.RENDERER_TEXT_LENGTH)
        self.assertEqual(
            set(partition.token_to_role), set(range(len(partition.role_names)))
        )
        null_index = partition.role_names.index(smoke.NULL_ROLE)
        self.assertGreater(partition.token_to_role.count(null_index), 0)

    def test_module_state_marker_detects_in_place_change(self) -> None:
        module = torch.nn.Linear(3, 2, bias=False)
        module.requires_grad_(False)
        module.eval()
        before = smoke._module_state_version_receipt(module)
        with torch.no_grad():
            module.weight.add_(1.0)
        after = smoke._module_state_version_receipt(module)
        self.assertNotEqual(before["digest"], after["digest"])

    def test_scalar_prepared_kwarg_digest_is_supported_without_value_change(self) -> None:
        scalar = torch.tensor(19425, dtype=torch.int64)
        before = scalar.clone()
        digest = smoke._tensor_digest(scalar, label="max sequence length")
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertTrue(torch.equal(scalar, before))
        self.assertEqual(scalar.ndim, 0)

    def test_print_contract_is_cpu_static_and_never_initializes_world4(self) -> None:
        stream = io.StringIO()
        with mock.patch.object(smoke, "_initialize_world4") as initialize:
            with redirect_stdout(stream):
                result = smoke.main(["--print-contract"])
        self.assertEqual(result, 0)
        initialize.assert_not_called()
        self.assertIn("backend_attention_weights_observed", stream.getvalue())
        self.assertIn("launch_executed", stream.getvalue())

    def test_source_has_two_forwards_then_external_gathers_and_create_only_json(self) -> None:
        source_path = METHOD_ROOT / "auh_native_relational_attention_parity_smoke_v1.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertEqual(calls.count("run_frozen_forward"), 2)
        self.assertIn("all_gather_into_tensor", calls)
        self.assertIn("all_gather_object", calls)
        self.assertIn("install_native_relational_attention_hook", calls)
        self.assertNotIn("backward", calls)
        self.assertNotIn("zero_grad", calls)
        self.assertNotIn("step", calls)
        self.assertNotIn("decode", calls)
        self.assertIn('output.open("x", encoding="utf-8")', source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("PeftModel", source)
        self.assertNotIn("route_strength", source)
        self.assertNotIn("target_video=", source)
        run_source = ast.get_source_segment(
            source,
            next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "run_real_world4_parity_smoke"
            ),
        )
        self.assertLess(
            run_source.index("observer_on = runtime.run_frozen_forward"),
            run_source.index("_gather_one_block_after_forward"),
        )

    def test_launch_template_is_nonexecuting(self) -> None:
        value = smoke.remote_launch_template()
        self.assertEqual(value["nproc_per_node"], 4)
        self.assertFalse(value["launch_executed"])
        self.assertFalse(value["gpu_launch_authorized"])
        self.assertEqual(value["arguments"][0], "--run")


if __name__ == "__main__":
    unittest.main()
