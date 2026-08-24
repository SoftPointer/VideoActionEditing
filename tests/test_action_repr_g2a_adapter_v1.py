#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


try:
    import torch
    from torch import nn
except ModuleNotFoundError as error:  # pragma: no cover - host dependent
    raise unittest.SkipTest("PyTorch unavailable") from error


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBJECT_PATH = (
    REPO_ROOT
    / "methods"
    / "bernini_action_editing"
    / "action_repr_g2a_adapter_v1.py"
)


def _load_subject():
    spec = importlib.util.spec_from_file_location(
        "action_repr_g2a_adapter_v1_test", SUBJECT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


g2a = _load_subject()


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.native = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.native.weight.copy_(torch.eye(width) * 0.03125)

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        return hidden_states + self.native(hidden_states)


class _Transformer(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(width) for _ in range(30)])

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class G2AAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260824)
        self.width = 8
        self.middle_width = 5
        self.layout = g2a.TokenLayout(total_tokens=12, source_tokens=6, phase_count=3)
        self.hidden = torch.randn((1, 12, self.width), dtype=torch.float32)
        self.activity = torch.zeros((1, 12, 1), dtype=torch.bool)
        self.activity[:, 6:] = True

    def active_route(self, kind, *, step=0, trace=None, offset=0.0):
        flow = torch.randn((1, 12, 12), dtype=torch.float32) + float(offset)
        middle = {
            index: torch.randn((1, 12, self.middle_width), dtype=torch.float32)
            for index in g2a.DEFAULT_BLOCK_INDICES
        }
        return g2a.ActionRepresentationRoute(
            kind=kind,
            optimizer_step=step,
            layout=self.layout,
            flow=flow.detach(),
            activity=self.activity,
            middle_by_block=middle,
            representation_origin=(
                "real_target_frozen_extractor"
                if kind == "correct"
                else "counterfactual_control"
            ),
            representation_cache_sha256=(str(len(kind) % 10) * 64),
            middle_value_kind="post_attention_residual",
            matched_noise_timestep_rotary=True,
            trace=trace,
        )

    def bypass_route(self, kind, *, step=0, trace=None):
        return g2a.ActionRepresentationRoute(
            kind=kind,
            optimizer_step=step,
            layout=self.layout,
            trace=trace,
        )

    def install(self, *, source_copy=False):
        transformer = _Transformer(self.width)
        handle = g2a.install_action_repr_g2a_adapter(
            transformer,
            hidden_width=self.width,
            bottleneck_width=4,
            middle_width=self.middle_width,
            enable_source_copy_adapter=source_copy,
        )
        return transformer, handle

    def test_all_six_step0_routes_are_bit_exact_and_first_backward_is_live(self):
        transformer, handle = self.install(source_copy=True)
        native = transformer(self.hidden)
        outputs = {}
        routes = {
            "correct": self.active_route("correct", offset=0.0),
            "zero": self.bypass_route("zero"),
            "temporal_shuffle": self.active_route("temporal_shuffle", offset=2.0),
            "reverse": self.active_route("reverse", offset=3.0),
            "incomplete": self.active_route("incomplete", offset=4.0),
            "wrong_action": self.active_route("wrong_action", offset=5.0),
        }
        self.assertEqual(tuple(routes), g2a.STEP0_REQUIRED_ROUTES)
        for kind, route in routes.items():
            with g2a.action_representation_route(route):
                outputs[kind] = transformer(self.hidden)
            self.assertTrue(g2a.tensor_bits_equal(native, outputs[kind]))
        outputs["correct"].sum().backward()
        output_grads = [
            parameter.grad
            for name, parameter in handle.trainable_named_parameters()
            if name.endswith("output.weight")
        ]
        self.assertTrue(output_grads)
        self.assertTrue(all(gradient is not None for gradient in output_grads))
        self.assertTrue(any(torch.count_nonzero(gradient) for gradient in output_grads))
        self.assertTrue(handle.output_gates_are_byte_zero())

    def test_receipt_closes_allowlist_base_digest_and_create_only_publication(self):
        transformer, handle = self.install(source_copy=True)
        native = transformer(self.hidden)
        outputs = {}
        for index, kind in enumerate(g2a.STEP0_REQUIRED_ROUTES):
            route = (
                self.bypass_route(kind)
                if kind == "zero"
                else self.active_route(kind, offset=float(index))
            )
            with g2a.action_representation_route(route):
                outputs[kind] = transformer(self.hidden)
        receipt = handle.build_g2a_receipt(
            native_output=native,
            routed_outputs=outputs,
            matched_input_sha256="a" * 64,
            forward_scope="toy_full_transformer_forward",
        )
        self.assertTrue(receipt["passed"])
        self.assertFalse(receipt["optimizer_created"])
        self.assertFalse(receipt["optimizer_authorized_by_this_receipt"])
        self.assertEqual(
            set(receipt["parameter_audit"]["roles"]), set(g2a.TRAINABLE_ROLES)
        )
        self.assertEqual(
            receipt["parameter_audit"]["base_digest_at_install"],
            receipt["parameter_audit"]["base_digest_current"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "g2a.json"
            g2a.write_receipt_create_only(path.resolve(), receipt)
            self.assertEqual(json.loads(path.read_text()), receipt)
            with self.assertRaisesRegex(g2a.G2AAdapterError, "create-only"):
                g2a.write_receipt_create_only(path.resolve(), receipt)

    def test_trace_is_per_block_bpnd_and_keeps_step0_gradient(self):
        transformer, handle = self.install()
        trace = g2a.ResidualTraceCollector(g2a.DEFAULT_BLOCK_INDICES)
        route = self.active_route("correct", trace=trace)
        with g2a.action_representation_route(route):
            observed = transformer(self.hidden)
        expected = transformer(self.hidden)
        self.assertTrue(g2a.tensor_bits_equal(observed, expected))
        values = trace.require_complete()
        self.assertEqual(set(values), set(g2a.DEFAULT_BLOCK_INDICES))
        for index in g2a.DEFAULT_BLOCK_INDICES:
            self.assertEqual(tuple(trace.for_block(index).shape), (1, 3, 2, 8))
            self.assertEqual(
                tuple(trace.activity_for_block(index).shape), (1, 3, 2, 1)
            )
            self.assertTrue(trace.for_block(index).requires_grad)
            self.assertEqual(torch.count_nonzero(trace.for_block(index)), 0)
        trace.for_block(24).sum().backward()
        self.assertIsNotNone(handle.bundles[-1].motion_adapter.output.weight.grad)

    def test_after_update_active_route_changes_but_zero_and_off_are_native(self):
        transformer, handle = self.install()
        native = transformer(self.hidden)
        with torch.no_grad():
            for bundle in handle.bundles:
                bundle.motion_adapter.output.weight.fill_(0.05)
        with g2a.action_representation_route(self.active_route("correct", step=1)):
            active = transformer(self.hidden)
        self.assertFalse(g2a.tensor_bits_equal(native, active))
        for kind in ("zero", "route_off"):
            with g2a.action_representation_route(self.bypass_route(kind, step=1)):
                bypass = transformer(self.hidden)
            self.assertTrue(g2a.tensor_bits_equal(native, bypass))
        with self.assertRaisesRegex(g2a.G2AAdapterError, "nonzero output gate"):
            with g2a.action_representation_route(self.active_route("correct", step=0)):
                transformer(self.hidden)

    def test_parameter_audit_fails_on_base_mutation_or_trainable_escape(self):
        transformer, handle = self.install()
        base = transformer.blocks[0].native.weight
        with torch.no_grad():
            base.add_(1.0)
        with self.assertRaisesRegex(g2a.G2AAdapterError, "bytes changed"):
            handle.audit_parameters(deep_base_bytes=True)

        transformer2, handle2 = self.install()
        transformer2.blocks[0].native.weight.requires_grad_(True)
        with self.assertRaisesRegex(g2a.G2AAdapterError, "allowlist"):
            handle2.audit_parameters(deep_base_bytes=False)

    def test_route_contract_rejects_attached_or_incomplete_middle_payload(self):
        transformer, _ = self.install()
        route = self.active_route("correct")
        route.flow.requires_grad_(True)
        with self.assertRaisesRegex(g2a.G2AAdapterError, "detached"):
            route.validate_basic()
        incomplete = self.active_route("correct")
        incomplete.middle_by_block.pop(24)
        with self.assertRaisesRegex(g2a.G2AAdapterError, "block closure"):
            with g2a.action_representation_route(incomplete):
                transformer(self.hidden)

    def test_receipt_tamper_and_nonmatching_output_fail_closed(self):
        transformer, handle = self.install()
        native = transformer(self.hidden)
        outputs = {kind: native.clone() for kind in g2a.STEP0_REQUIRED_ROUTES}
        bad = dict(outputs)
        bad["wrong_action"] = bad["wrong_action"].clone()
        bad["wrong_action"][0, 0, 0] += 1.0
        with self.assertRaisesRegex(g2a.G2AAdapterError, "not exact"):
            handle.build_g2a_receipt(
                native_output=native,
                routed_outputs=bad,
                matched_input_sha256="b" * 64,
                forward_scope="toy",
            )
        receipt = handle.build_g2a_receipt(
            native_output=native,
            routed_outputs=outputs,
            matched_input_sha256="b" * 64,
            forward_scope="toy",
        )
        tampered = copy.deepcopy(receipt)
        tampered["optimizer_created"] = True
        with self.assertRaises(g2a.G2AAdapterError):
            g2a.validate_g2a_receipt(tampered)

    def test_restore_and_two_handles_do_not_share_installation_cell(self):
        first, first_handle = self.install()
        second, second_handle = self.install()
        first_native = first(self.hidden)
        second_native = second(self.hidden)
        with g2a.action_representation_route(self.active_route("correct")):
            self.assertTrue(g2a.tensor_bits_equal(first(self.hidden), first_native))
        with g2a.action_representation_route(self.active_route("correct")):
            self.assertTrue(g2a.tensor_bits_equal(second(self.hidden), second_native))
        first_handle.restore()
        self.assertTrue(g2a.tensor_bits_equal(first(self.hidden), first_native))
        self.assertFalse(second_handle.restored)


if __name__ == "__main__":
    unittest.main()
