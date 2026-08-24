from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import sys
import unittest

try:
    import torch
except ImportError as error:  # pragma: no cover
    raise unittest.SkipTest("torch unavailable") from error


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_owned_role_locator_v15 as locator  # noqa: E402
import source_owned_role_locator_v15b_e00_asset as role_asset  # noqa: E402
import source_owned_role_null_bank_observer_v15b_r6 as observer  # noqa: E402
import source_owned_role_null_registry_v15b_r6 as registry_module  # noqa: E402


class _ExactTokenizer:
    def __init__(self, registry):
        self.registry = registry

    def __call__(self, _text, **_kwargs):
        return {
            "input_ids": list(self.registry.active_token_ids),
            "attention_mask": [1] * len(self.registry.active_token_ids),
            "offset_mapping": [list(item) for item in self.registry.active_token_offsets],
        }


class V15BR6NullRegistryAndABITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = registry_module.load_null_registry_v15b_r6()
        cls.spec, _raw = role_asset.load_e00_v15b_asset()

    def test_static_registry_is_exact64_unclaimed_and_runtime_reproducible(self):
        registry = self.registry
        self.assertEqual(registry.registry_sha256, registry_module.REGISTRY_SHA256)
        self.assertEqual(len(registry.spans), 64)
        occupied = set(registry.occupied_token_indices)
        for span in registry.spans:
            self.assertFalse(occupied.intersection(range(span.token_start, span.token_end)))
            self.assertGreater(
                registry.active_token_offsets[span.token_start][1],
                registry.active_token_offsets[span.token_start][0],
            )
        receipt = registry_module.validate_runtime_null_registry_v15b_r6(
            _ExactTokenizer(registry), registry
        )
        self.assertTrue(receipt["runtime_exact"])
        self.assertFalse(receipt["anchor_consumed"])
        self.assertFalse(receipt["route_authorized"])

    def test_runtime_token_mutation_fails_closed(self):
        registry = self.registry

        class Mutated(_ExactTokenizer):
            def __call__(self, text, **kwargs):
                value = super().__call__(text, **kwargs)
                value["input_ids"][0] += 1
                return value

        with self.assertRaisesRegex(
            registry_module.NullRegistryV15BR6Error, "differs from preregistration"
        ):
            registry_module.validate_runtime_null_registry_v15b_r6(
                Mutated(registry), registry
            )

    def _make_shard(self, rank: int, geometry: locator.SourceVisualGeometry):
        layout = locator.UlyssesVisualShard(geometry=geometry, rank=rank, size=4)
        valid = layout.valid_local_tokens
        start = layout.global_start
        real = torch.arange(start, start + valid, dtype=torch.float32).repeat(5, 1)
        legacy = torch.arange(start, start + valid, dtype=torch.float32) + 1000
        shuffled = real + 2000
        nulls = torch.arange(start, start + valid, dtype=torch.float32).repeat(64, 1)
        nulls += torch.arange(64, dtype=torch.float32).view(64, 1) * 100
        return observer.NullBankAffinityShardV15BR6(
            event_id="pour-liquid-into-cup",
            source_text_provenance_sha256="1" * 64,
            null_registry_sha256=registry_module.REGISTRY_SHA256,
            step_index=0,
            block_index=9,
            role_names=role_asset.ROLE_NAMES,
            layout=layout,
            affinity=real.contiguous(),
            legacy_null_affinity=legacy.contiguous(),
            shuffled_affinity=shuffled.contiguous(),
            null_span_affinity=nulls.contiguous(),
        )

    def test_sp4_75_channel_roundtrip_and_global_assembly(self):
        geometry = locator.SourceVisualGeometry(height=1, width=2)
        original = [self._make_shard(rank, geometry) for rank in range(4)]
        rebuilt = [
            observer.NullBankAffinityShardV15BR6.from_collective(
                item.padded_collective_tensor(), item.collective_metadata()
            )
            for item in original
        ]
        self.assertTrue(all(item.padded_collective_tensor().shape[0] == 75 for item in rebuilt))
        global_value = observer.assemble_global_null_bank_affinity_v15b_r6(rebuilt)
        self.assertEqual(tuple(global_value.affinity.shape), (5, 21, 1, 2))
        self.assertEqual(tuple(global_value.null_span_affinity.shape), (64, 21, 1, 2))
        self.assertTrue(
            torch.equal(
                global_value.null_span_affinity[7].reshape(-1),
                torch.arange(42, dtype=torch.float32) + 700,
            )
        )

    def test_collective_metadata_is_field_closed_and_hashed(self):
        shard = self._make_shard(0, locator.SourceVisualGeometry(height=1, width=2))
        tensor = shard.padded_collective_tensor()
        for mutation in ("missing", "extra", "channel"):
            metadata = deepcopy(dict(shard.collective_metadata()))
            if mutation == "missing":
                metadata.pop("null_span_count")
            elif mutation == "extra":
                metadata["unexpected"] = True
            else:
                metadata["collective_channels"]["null_spans"][1] -= 1
            with self.assertRaises(observer.NullBankObserverV15BR6Error):
                observer.NullBankAffinityShardV15BR6.from_collective(tensor, metadata)

    def test_explicit_64_maps_are_bit_repeat_deterministic(self):
        generator = torch.Generator().manual_seed(23)
        query = torch.randn((1, 17, 2, 8), generator=generator)
        key = torch.randn((1, 56, 2, 8), generator=generator)
        args = dict(
            query=query,
            key=key,
            roles=self.spec.roles,
            registry=self.registry,
            valid_local_tokens=17,
            active_source_tokens=56,
        )
        first = observer.source_role_null_bank_affinity_v15b_r6(**args)
        second = observer.source_role_null_bank_affinity_v15b_r6(**args)
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(first, second)))
        self.assertEqual(tuple(first[3].shape), (64, 17))

    def test_harness_has_no_training_route_or_decode_call(self):
        path = METHOD_ROOT / "probe_source_owned_role_locator_v15b_r6_sp4.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        for forbidden in ("backward", "step", "zero_grad", "decode"):
            self.assertNotIn(forbidden, calls)
        self.assertNotIn("optimizer", source.lower())
        self.assertIn("observer_only", source)
        self.assertIn("route_authorized", source)

    def test_auh_launcher_sets_every_adapter_pre_model_environment_gate(self):
        launcher = (
            METHOD_ROOT
            / "scripts/auh_probe_source_owned_role_locator_v15b_r6_sp4.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('export MODELING_BACKEND="hf"', launcher)
        self.assertIn('export HIP_VISIBLE_DEVICES="0,1,2,3"', launcher)
        self.assertIn('export V15B_MIOPEN_CACHE_ROOT=', launcher)
        self.assertIn("--nproc_per_node=4", launcher)
        self.assertNotIn("scancel", launcher)


if __name__ == "__main__":
    unittest.main()
