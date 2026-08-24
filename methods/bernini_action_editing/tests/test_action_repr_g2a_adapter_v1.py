from __future__ import annotations

from pathlib import Path
import sys
import unittest


try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - runtime dependent
    raise unittest.SkipTest("PyTorch unavailable") from error


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import action_repr_g2a_adapter_v1 as g2a  # noqa: E402


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


class G2ASixRouteGateTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260824)
        self.layout = g2a.TokenLayout(
            total_tokens=12, source_tokens=6, phase_count=3
        )
        self.hidden = torch.randn((1, 12, 8), dtype=torch.float32)
        self.activity = torch.zeros((1, 12, 1), dtype=torch.bool)
        self.activity[:, 6:] = True

    def active(
        self,
        kind: str,
        offset: float,
        *,
        trace: g2a.ResidualTraceCollector | None = None,
    ):
        return g2a.ActionRepresentationRoute(
            kind=kind,
            optimizer_step=0,
            layout=self.layout,
            flow=(torch.randn((1, 12, 12)) + offset).detach(),
            activity=self.activity,
            middle_by_block={
                index: torch.randn((1, 12, 5)).detach()
                for index in g2a.DEFAULT_BLOCK_INDICES
            },
            representation_origin=(
                "real_target_frozen_extractor"
                if kind == "correct"
                else "counterfactual_control"
            ),
            representation_cache_sha256=(str(int(offset) + 1) * 64),
            middle_value_kind="post_attention_residual",
            matched_noise_timestep_rotary=True,
            trace=trace,
        )

    def test_six_routes_are_bit_exact_and_receipt_is_non_authorizing(self):
        model = _Transformer().eval()
        handle = g2a.install_action_repr_g2a_adapter(
            model,
            hidden_width=8,
            bottleneck_width=4,
            middle_width=5,
            enable_source_copy_adapter=True,
        )
        native = model(self.hidden)
        routes = {
            "correct": self.active("correct", 0.0),
            "zero": g2a.ActionRepresentationRoute(
                kind="zero", optimizer_step=0, layout=self.layout
            ),
            "temporal_shuffle": self.active("temporal_shuffle", 1.0),
            "reverse": self.active("reverse", 2.0),
            "incomplete": self.active("incomplete", 3.0),
            "wrong_action": self.active("wrong_action", 4.0),
        }
        self.assertEqual(tuple(routes), g2a.STEP0_REQUIRED_ROUTES)
        outputs = {}
        for kind, route in routes.items():
            with g2a.action_representation_route(route):
                outputs[kind] = model(self.hidden)
            self.assertTrue(g2a.tensor_bits_equal(native, outputs[kind]))
        receipt = handle.build_g2a_receipt(
            native_output=native,
            routed_outputs=outputs,
            matched_input_sha256="a" * 64,
            forward_scope="cpu_toy_full_transformer",
        )
        self.assertEqual(
            receipt["step0_noop_audit"]["required_routes"],
            list(g2a.STEP0_REQUIRED_ROUTES),
        )
        self.assertEqual(len(receipt["step0_noop_audit"]["route_outputs"]), 6)
        self.assertTrue(receipt["step0_noop_audit"]["all_routes_exact_native_bits"])
        self.assertFalse(receipt["optimizer_created"])
        self.assertFalse(receipt["optimizer_authorized_by_this_receipt"])
        self.assertTrue(receipt["parameter_audit"]["base_bytes_unchanged"])

    def test_receipt_rejects_missing_reverse_or_incomplete(self):
        model = _Transformer().eval()
        handle = g2a.install_action_repr_g2a_adapter(
            model, hidden_width=8, bottleneck_width=4, middle_width=5
        )
        native = model(self.hidden)
        complete = {
            kind: native.clone() for kind in g2a.STEP0_REQUIRED_ROUTES
        }
        for missing in ("reverse", "incomplete"):
            outputs = dict(complete)
            outputs.pop(missing)
            with self.subTest(missing=missing), self.assertRaisesRegex(
                g2a.G2AAdapterError, "six required routes"
            ):
                handle.build_g2a_receipt(
                    native_output=native,
                    routed_outputs=outputs,
                    matched_input_sha256="b" * 64,
                    forward_scope="cpu_toy_full_transformer",
                )

    def test_fixed_projection_emits_256d_trace_and_keeps_output_gate_gradient(self):
        model = _Transformer().eval()
        handle = g2a.install_action_repr_g2a_adapter(
            model, hidden_width=8, bottleneck_width=4, middle_width=5
        )
        caller_projection = torch.randn((8, 256), dtype=torch.float32).detach()
        trace = g2a.ResidualTraceCollector(
            g2a.DEFAULT_BLOCK_INDICES,
            feature_projection=caller_projection,
        )
        self.assertIsNot(trace.feature_projection, caller_projection)
        self.assertFalse(trace.feature_projection.requires_grad)
        route = self.active("correct", 0.0, trace=trace)
        native = model(self.hidden)
        with g2a.action_representation_route(route):
            routed = model(self.hidden)
        self.assertTrue(g2a.tensor_bits_equal(native, routed))
        for index in g2a.DEFAULT_BLOCK_INDICES:
            value = trace.for_block(index)
            self.assertEqual(tuple(value.shape), (1, 3, 2, 256))
            self.assertTrue(value.requires_grad)
            self.assertEqual(int(torch.count_nonzero(value).item()), 0)
        trace.for_block(g2a.DEFAULT_BLOCK_INDICES[-1]).sum().backward()
        output_gradient = handle.bundles[-1].motion_adapter.output.weight.grad
        self.assertIsNotNone(output_gradient)
        self.assertGreater(int(torch.count_nonzero(output_gradient).item()), 0)

    def test_feature_projection_rejects_attached_nonfinite_and_wrong_shape(self):
        invalid = (
            torch.randn((8, 256), requires_grad=True),
            torch.full((8, 256), float("nan")),
            torch.randn((1, 8, 256)),
            torch.empty((8, 0)),
            torch.ones((8, 256), dtype=torch.int64),
        )
        for projection in invalid:
            with self.subTest(shape=tuple(projection.shape), dtype=projection.dtype):
                with self.assertRaisesRegex(
                    g2a.G2AAdapterError, "finite floating rank-2|detached"
                ):
                    g2a.ResidualTraceCollector(
                        g2a.DEFAULT_BLOCK_INDICES,
                        feature_projection=projection,
                    )

        model = _Transformer().eval()
        g2a.install_action_repr_g2a_adapter(
            model, hidden_width=8, bottleneck_width=4, middle_width=5
        )
        wrong_width = g2a.ResidualTraceCollector(
            g2a.DEFAULT_BLOCK_INDICES,
            feature_projection=torch.randn((7, 256)).detach(),
        )
        with self.assertRaisesRegex(g2a.G2AAdapterError, "input width differs"):
            with g2a.action_representation_route(
                self.active("correct", 0.0, trace=wrong_width)
            ):
                model(self.hidden)


if __name__ == "__main__":
    unittest.main()
