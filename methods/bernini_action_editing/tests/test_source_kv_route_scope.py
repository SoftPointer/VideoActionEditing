from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_kv_route_scope as scope  # noqa: E402


class _Tensor:
    def __init__(self, shape):
        self.shape = shape


class _Affine:
    def __init__(self, shape=(1536, 1536)):
        self.weight = _Tensor(shape)


def _runtime_inventory() -> dict[str, object]:
    inventory: dict[str, object] = {
        name: _Affine() for name in scope.canonical_target_modules()
    }
    # These real Bernini modules exist but are intentionally frozen by V9.
    for block in tuple(range(7)) + tuple(range(23, 30)):
        for projection in ("to_q", "to_out.0"):
            inventory[
                f"diff_dec.transformer.blocks.{block}.attn1.{projection}"
            ] = _Affine()
    inventory["diff_dec.transformer.blocks.0.attn2.to_k"] = _Affine()
    inventory["t5_text_encoder.encoder.block.0.layer.0"] = object()
    return inventory


def _adapter_state() -> dict[str, _Tensor]:
    return {
        key: _Tensor(shape)
        for key, shape in scope.expected_adapter_shapes().items()
    }


class CanonicalScopeTests(unittest.TestCase):
    def test_exact_92_scope_and_locked_digest(self) -> None:
        targets = scope.canonical_target_modules()
        self.assertEqual(targets, sorted(targets))
        self.assertEqual(len(targets), 92)
        self.assertEqual(len(set(targets)), 92)
        self.assertEqual(
            scope.object_sha256(targets),
            "16e5dc87ca134419841e2e9af6d26091141aa473aa4cc11ae53d2e4e28e0e4b5",
        )
        cross = [name for name in targets if ".attn2." in name]
        middle_self = [name for name in targets if ".attn1." in name]
        self.assertEqual(len(cross), 60)
        self.assertEqual(len(middle_self), 32)
        for block in range(30):
            for projection in ("to_q", "to_out.0"):
                self.assertIn(
                    f"diff_dec.transformer.blocks.{block}.attn2.{projection}",
                    targets,
                )
        for block in range(7, 23):
            for projection in ("to_q", "to_out.0"):
                self.assertIn(
                    f"diff_dec.transformer.blocks.{block}.attn1.{projection}",
                    targets,
                )

    def test_missing_duplicate_or_extra_target_name_fails_closed(self) -> None:
        targets = scope.canonical_target_modules()
        variants = (
            targets[:-1],
            targets + [targets[0]],
            targets + ["diff_dec.transformer.blocks.30.attn2.to_q"],
        )
        for values in variants:
            with self.subTest(count=len(values)), self.assertRaises(
                scope.SourceKVRouteScopeError
            ):
                scope.validate_target_module_names(values)


class RuntimeInventoryTests(unittest.TestCase):
    def test_runtime_inventory_selects_exact_scope_and_weight_geometry(self) -> None:
        selected = scope.validate_runtime_target_modules(_runtime_inventory())
        self.assertEqual(selected, scope.canonical_target_modules())

    def test_missing_or_out_of_range_runtime_module_fails_closed(self) -> None:
        missing = _runtime_inventory()
        missing.pop(scope.canonical_target_modules()[0])
        extra = _runtime_inventory()
        extra["diff_dec.transformer.blocks.30.attn2.to_q"] = _Affine()
        for inventory in (missing, extra):
            with self.assertRaises(scope.SourceKVRouteScopeError):
                scope.validate_runtime_target_modules(inventory)

    def test_wrong_runtime_weight_shape_fails_closed(self) -> None:
        inventory = _runtime_inventory()
        inventory[scope.canonical_target_modules()[11]] = _Affine((1536, 1024))
        with self.assertRaisesRegex(
            scope.SourceKVRouteScopeError, "weight shape differs"
        ):
            scope.validate_runtime_target_modules(inventory)


class AdapterContractTests(unittest.TestCase):
    def test_exact_rank_alpha_tensor_count_shapes_and_parameter_count(self) -> None:
        hyperparameters = scope.validate_lora_hyperparameters(
            rank=8, alpha=8
        )
        self.assertEqual(hyperparameters["hidden_size"], 1536)
        evidence = scope.validate_adapter_state(_adapter_state())
        self.assertTrue(evidence["validated"])
        self.assertEqual(evidence["adapter_tensor_count"], 184)
        self.assertEqual(evidence["trainable_parameter_count"], 2_260_992)
        self.assertEqual(len(scope.canonical_adapter_state_keys()), 184)

    def test_wrong_rank_or_alpha_fails_closed(self) -> None:
        with self.assertRaises(scope.SourceKVRouteScopeError):
            scope.validate_lora_hyperparameters(rank=4, alpha=8)
        with self.assertRaises(scope.SourceKVRouteScopeError):
            scope.validate_lora_hyperparameters(rank=8, alpha=16)
        with self.assertRaises(scope.SourceKVRouteScopeError):
            scope.validate_adapter_state(_adapter_state(), rank=4)

    def test_missing_or_extra_adapter_tensor_fails_closed(self) -> None:
        state = _adapter_state()
        state.pop(next(iter(state)))
        with self.assertRaisesRegex(
            scope.SourceKVRouteScopeError, "adapter tensor scope differs"
        ):
            scope.validate_adapter_state(state)
        state = _adapter_state()
        state["unexpected.lora_A.weight"] = _Tensor((8, 1536))
        with self.assertRaisesRegex(
            scope.SourceKVRouteScopeError, "adapter tensor scope differs"
        ):
            scope.validate_adapter_state(state)

    def test_wrong_adapter_tensor_shape_fails_closed(self) -> None:
        state = _adapter_state()
        key = next(key for key in state if key.endswith("lora_A.weight"))
        state[key] = _Tensor((4, 1536))
        with self.assertRaisesRegex(
            scope.SourceKVRouteScopeError, "adapter tensor shape differs"
        ):
            scope.validate_adapter_state(state)


class FreshInitializationAndReceiptTests(unittest.TestCase):
    def test_main_accepts_only_exact_fresh_initialization(self) -> None:
        declaration = scope.fresh_initialization_declaration()
        self.assertEqual(
            scope.validate_fresh_initialization(declaration), declaration
        )
        for change in (
            {"v8_warm_start": True},
            {"warm_start": True, "warm_start_method": "v8"},
            {"adapter_checkpoint_loaded": True},
            {"warm_start_adapter_sha256": "a" * 64},
            {"initialization_source": "v8_checkpoint"},
        ):
            value = copy.deepcopy(declaration)
            value.update(change)
            with self.subTest(change=change), self.assertRaises(
                scope.SourceKVRouteScopeError
            ):
                scope.validate_fresh_initialization(value)

    def test_receipt_manifest_is_complete_digest_bound_and_stable(self) -> None:
        kwargs = {
            "runtime_module_inventory": _runtime_inventory(),
            "adapter_state": _adapter_state(),
            "initialization": scope.fresh_initialization_declaration(),
        }
        first = scope.build_receipt_manifest(**kwargs)
        second = scope.build_receipt_manifest(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], scope.RECEIPT_MANIFEST_SCHEMA)
        self.assertEqual(first["lora"]["target_module_count"], 92)
        self.assertEqual(first["lora"]["adapter_tensor_count"], 184)
        self.assertEqual(
            first["lora"]["trainable_parameter_count"], 2_260_992
        )
        self.assertEqual(
            first["lora"]["target_modules_sha256"],
            scope.EXPECTED_TARGET_MODULES_SHA256,
        )
        self.assertTrue(
            first["validation"]["v8_warm_start_forbidden_for_main"]
        )
        digest = first.pop("manifest_digest")
        self.assertEqual(digest, scope.object_sha256(first))


if __name__ == "__main__":
    unittest.main()
