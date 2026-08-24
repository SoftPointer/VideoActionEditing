from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATH = (
    METHOD_ROOT
    / "scripts/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh"
)
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_anchor_sga_anc_trained_editor_decode_v1 as audited  # noqa: E402


class _Parameter:
    def __init__(
        self,
        *,
        numel: int = 8,
        shape: tuple[int, ...] = (2, 4),
        requires_grad: bool = False,
    ) -> None:
        self._numel = numel
        self.shape = shape
        self.requires_grad = requires_grad
        self.dtype = "torch.bfloat16"

    def numel(self) -> int:
        return self._numel

    def requires_grad_(self, value: bool):
        self.requires_grad = value
        return self


class _LoraLayer:
    def __init__(self) -> None:
        self.lora_A = {"default": object()}
        self.lora_B = {"default": object()}
        self.active_adapters = ["default"]
        self.disable_adapters = False
        self.merged = False
        self.merged_adapters: list[str] = []


class _PlainModel:
    def named_modules(self):
        return [("", self)]

    def named_parameters(self):
        return [("weight", _Parameter())]


class _PeftModel:
    def __init__(self) -> None:
        self.peft_config = {"default": object()}
        self.layers = [_LoraLayer() for _ in range(audited.EXPECTED_LORA_LAYER_COUNT)]
        self.base_parameter = _Parameter(numel=32, shape=(8, 4))
        self.adapter_parameters = [
            (_Parameter(), _Parameter())
            for _ in range(audited.EXPECTED_LORA_LAYER_COUNT)
        ]
        self.break_disable = False
        self.break_restore = False

    def named_modules(self):
        yield "", self
        for index, layer in enumerate(self.layers):
            yield f"base.blocks.{index}.to_q", layer

    def named_parameters(self):
        yield "base.weight", self.base_parameter
        for index, (parameter_a, parameter_b) in enumerate(self.adapter_parameters):
            yield f"base.blocks.{index}.to_q.lora_A.default.weight", parameter_a
            yield f"base.blocks.{index}.to_q.lora_B.default.weight", parameter_b

    @contextmanager
    def disable_adapter(self):
        if not self.break_disable:
            for layer in self.layers:
                layer.disable_adapters = True
        try:
            yield
        finally:
            if not self.break_restore:
                for layer in self.layers:
                    layer.disable_adapters = False
            # Match PEFT 0.19.1: enable_adapter_layers()->set_adapter()
            # restores active adapter parameters to trainable state.
            for parameter_a, parameter_b in self.adapter_parameters:
                parameter_a.requires_grad = True
                parameter_b.requires_grad = True


class TrainedEditorFreezeCertificateTests(unittest.TestCase):
    def _certificate(self, strict=None):
        if strict is None:
            strict = lambda model: {
                "base_frozen": True,
                "adapter_modules_absent": True,
            }
        return audited.build_model_freeze_certificate(
            strict,
            error_type=ValueError,
            peft_model_type=_PeftModel,
        )

    def test_plain_base_keeps_original_strict_no_adapter_gate(self) -> None:
        calls = []

        def strict(model):
            calls.append(model)
            return {"base_frozen": True, "adapter_modules_absent": True}

        model = _PlainModel()
        result = self._certificate(strict)(model)
        self.assertEqual(calls, [model])
        self.assertEqual(
            result, {"base_frozen": True, "adapter_modules_absent": True}
        )

    def test_frozen_unmerged_enabled_peft_editor_is_certified(self) -> None:
        model = _PeftModel()
        first = self._certificate()(model)
        second = self._certificate()(model)
        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], audited.CERTIFICATE_SCHEMA)
        self.assertTrue(first["base_and_adapter_frozen"])
        self.assertTrue(first["adapter_enabled_for_editor_calls"])
        self.assertTrue(first["adapter_disable_context_reversible"])
        self.assertTrue(first["adapter_disable_context_refreezes_parameters"])
        self.assertEqual(first["lora_layer_count"], 240)
        self.assertEqual(first["lora_parameter_tensors"], 480)
        self.assertEqual(len(first["lora_layer_inventory_sha256"]), 64)
        self.assertEqual(len(first["lora_parameter_inventory_sha256"]), 64)
        self.assertFalse(any(layer.disable_adapters for layer in model.layers))
        self.assertFalse(
            any(
                parameter.requires_grad
                for pair in model.adapter_parameters
                for parameter in pair
            )
        )
        bridge = BRIDGE_PATH.read_text(encoding="utf-8")
        exec_tail = bridge[bridge.index('exec "$python_bin"') :]
        self.assertIn(
            '"$dev/infer_anchor_sga_anc_trained_editor_decode_v1.py"', exec_tail
        )
        self.assertNotIn('"$dev/infer_anchor_sga_anc_event_v1.py"', exec_tail)

    def test_trainable_parameter_is_rejected(self) -> None:
        model = _PeftModel()
        model.base_parameter.requires_grad = True
        with self.assertRaisesRegex(ValueError, "contains trainable parameters"):
            self._certificate()(model)

    def test_disabled_editor_is_rejected(self) -> None:
        model = _PeftModel()
        model.layers[0].disable_adapters = True
        with self.assertRaisesRegex(ValueError, "disabled outside"):
            self._certificate()(model)

    def test_merged_editor_is_rejected(self) -> None:
        model = _PeftModel()
        model.layers[0].merged = True
        model.layers[0].merged_adapters = ["default"]
        with self.assertRaisesRegex(ValueError, "merged into"):
            self._certificate()(model)

    def test_layer_or_parameter_inventory_drift_is_rejected(self) -> None:
        model = _PeftModel()
        model.layers.pop()
        with self.assertRaisesRegex(ValueError, "layer count differs"):
            self._certificate()(model)

        class MissingParameterModel(_PeftModel):
            def named_parameters(self):
                rows = list(super().named_parameters())
                yield from rows[:-1]

        with self.assertRaisesRegex(ValueError, "parameter tensor count differs"):
            audited.build_model_freeze_certificate(
                lambda model: {},
                error_type=ValueError,
                peft_model_type=MissingParameterModel,
            )(MissingParameterModel())

    def test_disable_context_must_transition_and_restore_every_layer(self) -> None:
        model = _PeftModel()
        model.break_disable = True
        with self.assertRaisesRegex(ValueError, "did not disable every"):
            self._certificate()(model)

        model = _PeftModel()
        model.break_restore = True
        with self.assertRaisesRegex(ValueError, "did not restore"):
            self._certificate()(model)

    def test_partial_or_non_peft_adapter_surface_is_rejected(self) -> None:
        class Partial(_PlainModel):
            peft_config = {"default": object()}

        with self.assertRaisesRegex(ValueError, "partial or unauthenticated"):
            self._certificate()(Partial())

        class ExpectedPeftModel:
            pass

        with self.assertRaisesRegex(ValueError, "not a PEFT PeftModel"):
            audited.build_model_freeze_certificate(
                lambda model: {},
                error_type=ValueError,
                peft_model_type=ExpectedPeftModel,
            )(_PeftModel())


if __name__ == "__main__":
    unittest.main()
