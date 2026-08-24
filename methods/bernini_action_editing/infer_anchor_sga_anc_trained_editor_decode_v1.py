#!/usr/bin/env python3
"""Decode entrypoint with a strict, adapter-aware inference freeze certificate.

The shared SGA/ANC runner predates trained-editor decoding and therefore
correctly rejects every model containing a LoRA module.  A trained decode has
two different roles, however: target/source calls must use the frozen LoRA
editor, while pure-T2V teacher calls temporarily enter PEFT's
``disable_adapter()`` context.  This entrypoint changes only the audit
function used by that runner; it does not modify its sampler or attention
controller.

Plain Bernini models still pass through the original no-adapter certificate.
PEFT models are accepted only when the entire model is frozen, the exact
all-attention adapter inventory is present, selected and enabled, no layer is
merged, and the reversible PEFT disable context exists.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import hashlib
import json
from typing import Any, Optional


CERTIFICATE_SCHEMA = "bernini-frozen-peft-editor-certificate-v1"
EXPECTED_LORA_LAYER_COUNT = 240
EXPECTED_LORA_PARAMETER_TENSORS = 2 * EXPECTED_LORA_LAYER_COUNT


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _active_adapter_names(value: Any, *, fail: Callable[[str], None]) -> tuple[str, ...]:
    if isinstance(value, str):
        names = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        names = tuple(str(item) for item in value)
    else:
        fail("PEFT LoRA layer has no explicit active adapter selection")
        raise AssertionError("unreachable")
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        fail("PEFT LoRA layer active adapter selection is invalid")
    return tuple(sorted(names))


def build_model_freeze_certificate(
    strict_plain_base_certificate: Callable[[Any], Mapping[str, Any]],
    *,
    error_type: type[Exception] = RuntimeError,
    peft_model_type: Optional[type[Any]] = None,
) -> Callable[[Any], dict[str, Any]]:
    """Return a certificate function that keeps the original plain-base gate.

    ``peft_model_type`` is injectable only for dependency-free unit tests.  A
    production call resolves the class from the pinned PEFT installation.
    """

    def fail(message: str) -> None:
        raise error_type(message)

    def certificate(model: Any) -> dict[str, Any]:
        named_modules = list(model.named_modules())
        lora_layers = [
            (name, module)
            for name, module in named_modules
            if hasattr(module, "lora_A") and hasattr(module, "lora_B")
        ]
        peft_config = getattr(model, "peft_config", None)
        disable_adapter = getattr(model, "disable_adapter", None)
        adapter_signals = (
            bool(lora_layers),
            isinstance(peft_config, Mapping) and bool(peft_config),
            callable(disable_adapter),
        )
        if not any(adapter_signals):
            # Preserve the old fail-closed rule for every plain frozen model:
            # no trainable parameter and no hidden/accidental adapter module.
            return dict(strict_plain_base_certificate(model))
        if not all(adapter_signals):
            fail("partial or unauthenticated PEFT editor surface")

        resolved_peft_type = peft_model_type
        if resolved_peft_type is None:
            try:
                from peft import PeftModel
            except ImportError as error:
                raise error_type("cannot authenticate PEFT editor class") from error
            resolved_peft_type = PeftModel
        if not isinstance(model, resolved_peft_type):
            fail("adapter-bearing editor is not a PEFT PeftModel")

        parameters = list(model.named_parameters())
        trainable = [
            (name, int(parameter.numel()))
            for name, parameter in parameters
            if bool(parameter.requires_grad)
        ]
        if trainable:
            fail(
                "frozen PEFT editor contains trainable parameters: "
                f"{len(trainable)} tensors/{sum(item[1] for item in trainable)} elements"
            )

        # PEFT 0.19.1's LoRA disable context calls set_adapter() on exit,
        # which restores adapter requires_grad flags even when the checkpoint
        # was loaded with is_trainable=False.  The shared controller invokes
        # that public context repeatedly.  Install one instance-local guard
        # that preserves PEFT's enable/disable behavior but immediately
        # restores the inference-only freeze after every teacher call.
        guard_attribute = "_bernini_frozen_disable_adapter_guard_v1"
        guard_state = getattr(model, guard_attribute, None)
        if guard_state is None:
            original_disable_adapter = disable_adapter

            @contextmanager
            def guarded_disable_adapter():
                try:
                    with original_disable_adapter():
                        yield
                finally:
                    for _, parameter in model.named_parameters():
                        parameter.requires_grad_(False)

            guard_state = {
                "original": original_disable_adapter,
                "guarded": guarded_disable_adapter,
            }
            object.__setattr__(model, "disable_adapter", guarded_disable_adapter)
            object.__setattr__(model, guard_attribute, guard_state)
            disable_adapter = guarded_disable_adapter
        elif (
            not isinstance(guard_state, Mapping)
            or set(guard_state) != {"original", "guarded"}
            or not callable(guard_state.get("original"))
            or not callable(guard_state.get("guarded"))
            or disable_adapter is not guard_state.get("guarded")
        ):
            fail("trained editor disable-adapter freeze guard changed")
        if len(lora_layers) != EXPECTED_LORA_LAYER_COUNT:
            fail(
                "trained editor LoRA layer count differs: "
                f"{len(lora_layers)} != {EXPECTED_LORA_LAYER_COUNT}"
            )

        config_names = tuple(sorted(str(name) for name in peft_config))
        if len(config_names) != 1:
            fail("trained editor must expose exactly one PEFT adapter config")

        layer_rows: list[dict[str, Any]] = []
        for name, layer in sorted(lora_layers, key=lambda item: item[0]):
            try:
                a_names = tuple(sorted(str(item) for item in layer.lora_A.keys()))
                b_names = tuple(sorted(str(item) for item in layer.lora_B.keys()))
            except (AttributeError, TypeError) as error:
                raise error_type("PEFT LoRA layer adapter maps are unreadable") from error
            active_names = _active_adapter_names(
                getattr(layer, "active_adapters", None), fail=fail
            )
            if (
                a_names != b_names
                or a_names != config_names
                or active_names != config_names
            ):
                fail("PEFT LoRA layer adapter keys/active selection differ")
            if bool(getattr(layer, "disable_adapters", True)):
                fail("trained editor adapter is disabled outside pure-T2V teacher calls")
            if bool(getattr(layer, "merged", False)) or tuple(
                getattr(layer, "merged_adapters", ())
            ):
                fail("trained editor adapter is merged into the frozen base")
            layer_rows.append(
                {
                    "name": name,
                    "adapter_names": list(a_names),
                    "active_adapter_names": list(active_names),
                }
            )

        # Prove the exact API used by the controller is reversible without
        # executing a model forward.  This is stronger than merely checking
        # that a same-named method exists: every LoRA layer must transition to
        # base-only and then return to editor-enabled state.
        try:
            with disable_adapter():
                if any(
                    not bool(getattr(layer, "disable_adapters", False))
                    for _, layer in lora_layers
                ):
                    fail("PEFT disable_adapter did not disable every LoRA layer")
        except error_type:
            raise
        except Exception as error:
            raise error_type("PEFT disable_adapter context failed") from error
        if any(
            bool(getattr(layer, "disable_adapters", True))
            for _, layer in lora_layers
        ):
            fail("PEFT disable_adapter did not restore the trained editor")
        if any(bool(parameter.requires_grad) for _, parameter in model.named_parameters()):
            fail("PEFT disable_adapter did not preserve the inference-only freeze")

        adapter_rows = []
        unexpected_adapter_parameters = []
        for name, parameter in parameters:
            lowered = name.lower()
            is_adapter_parameter = "lora_" in lowered or ".lora" in lowered
            is_expected = ".lora_a." in lowered or ".lora_b." in lowered
            if is_adapter_parameter and not is_expected:
                unexpected_adapter_parameters.append(name)
            if is_expected:
                adapter_rows.append(
                    {
                        "name": name,
                        "shape": [int(item) for item in parameter.shape],
                        "numel": int(parameter.numel()),
                        "dtype": str(parameter.dtype),
                    }
                )
        adapter_rows.sort(key=lambda row: row["name"])
        if unexpected_adapter_parameters:
            fail("trained editor contains an unexpected LoRA parameter family")
        if len(adapter_rows) != EXPECTED_LORA_PARAMETER_TENSORS:
            fail(
                "trained editor LoRA parameter tensor count differs: "
                f"{len(adapter_rows)} != {EXPECTED_LORA_PARAMETER_TENSORS}"
            )
        adapter_elements = sum(int(row["numel"]) for row in adapter_rows)
        if adapter_elements <= 0:
            fail("trained editor LoRA parameter inventory is empty")

        return {
            "schema_version": CERTIFICATE_SCHEMA,
            "base_and_adapter_frozen": True,
            "base_frozen": True,
            "trainable_parameter_tensors": 0,
            "trainable_parameter_elements": 0,
            "peft_model_authenticated": True,
            "adapter_disable_context_available": True,
            "adapter_disable_context_reversible": True,
            "adapter_disable_context_refreezes_parameters": True,
            "adapter_kept_unmerged": True,
            "adapter_enabled_for_editor_calls": True,
            "pure_t2v_teacher_policy": "temporary_disable_adapter_context",
            "adapter_config_names": list(config_names),
            "lora_layer_count": len(layer_rows),
            "lora_parameter_tensors": len(adapter_rows),
            "lora_parameter_elements": adapter_elements,
            "lora_layer_inventory_sha256": _canonical_sha256(layer_rows),
            "lora_parameter_inventory_sha256": _canonical_sha256(adapter_rows),
        }

    certificate.__name__ = "trained_editor_model_freeze_certificate"
    return certificate


def main(argv: Optional[Sequence[str]] = None) -> int:
    import infer_anchor_sga_anc_event_v1 as runner

    original = runner.source_audit.model_freeze_certificate
    replacement = build_model_freeze_certificate(
        original,
        error_type=runner.AnchorEventInferenceError,
    )
    runner.source_audit.model_freeze_certificate = replacement
    try:
        return runner.main(argv)
    finally:
        runner.source_audit.model_freeze_certificate = original


if __name__ == "__main__":
    raise SystemExit(main())
