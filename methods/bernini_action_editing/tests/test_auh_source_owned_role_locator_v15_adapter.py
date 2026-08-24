from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest
from unittest import mock

try:
    import torch
except ImportError as error:  # pragma: no cover
    raise unittest.SkipTest("torch unavailable") from error


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import auh_source_owned_role_locator_v15_adapter as adapter  # noqa: E402
import source_owned_role_locator_v15 as locator  # noqa: E402
import source_owned_role_locator_v15b_e00_asset as role_asset_v15b  # noqa: E402


class FrozenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.eval()


def source_manifest_fixture():
    return {
        "schema_version": "bernini-native-identity-generation-canary-v1",
        "bernini_commit": adapter.BERNINI_REVISION,
        "veomni_commit": adapter.VEOMNI_REVISION,
        "checkpoint": {"tree_sha256": adapter.CHECKPOINT_TREE_SHA256},
        "input": {
            "source_video_path": str(adapter.SOURCE_VIDEO),
            "source_video_sha256": adapter.SOURCE_VIDEO_SHA256,
            "target_video": False,
        },
        "source_condition_artifact": {
            "path": str(adapter.CLEAN_LATENT),
            "sha256": adapter.CLEAN_LATENT_FILE_SHA256,
            "tensor_key": "normalized_clean_latent",
            "shape": list(adapter.LATENT_SHAPE),
            "source_video_vae_encode_before_any_decode": True,
            "roundtrip_byte_exact_fp32": True,
        },
        "initial_noise_artifacts": {
            "rv2v": {
                "path": str(adapter.NOISE),
                "sha256": adapter.NOISE_FILE_SHA256,
                "tensor_key": "official_initial_gaussian",
                "shape": list(adapter.LATENT_SHAPE),
                "source_or_target_derived": False,
                "captured_from_native_sampler": True,
                "roundtrip_raw_value_exact": True,
            }
        },
        "runtime_versions": {
            "torch": adapter.PINNED_VERSIONS["torch"],
            "transformers": adapter.PINNED_VERSIONS["transformers"],
            "diffusers": adapter.PINNED_VERSIONS["diffusers"],
        },
    }


class AUHAdapterStaticTests(unittest.TestCase):
    def test_config_is_closed_and_pins_one_registered_block_set(self):
        normalized = adapter._normalize_config({})
        self.assertEqual(normalized.selected_block_indices, (4, 9, 14, 19, 24))
        self.assertEqual(
            adapter._normalize_config(
                {"selected_block_indices": [4, 9, 14, 19, 24]}
            ),
            normalized,
        )
        for invalid in (
            {"selected_block_indices": [4]},
            {"checkpoint": "/tmp/substitute"},
            {"selected_block_indices": "4,9,14,19,24"},
        ):
            with self.assertRaises(adapter.AUHSourceRoleAdapterError):
                adapter._normalize_config(invalid)

    def test_binding_registry_covers_static_authorities_and_forbids_route(self):
        receipt = adapter.pinned_e00_binding_registry()
        self.assertEqual(receipt["checkpoint_tree_sha256"], "6be0d0db" + receipt["checkpoint_tree_sha256"][8:])
        for key in (
            "source_video_sha256",
            "source_manifest_sha256",
            "clean_latent_file_sha256",
            "clean_latent_tensor_sha256",
            "noise_file_sha256",
            "noise_tensor_sha256",
            "timestep_tensor_sha256",
            "schedule_sigma_tensor_sha256",
            "noisy_source_tensor_sha256",
            "token_input_ids_sha256",
            "token_attention_mask_sha256",
            "model_text_sha256",
            "tokenizer_tree_sha256",
        ):
            self.assertRegex(receipt[key], r"^[0-9a-f]{64}$")
        self.assertEqual(
            receipt["official_schedule"],
            {
                "scheduler": "diffusers.UniPCMultistepScheduler",
                "flow_shift": 5.0,
                "steps": 40,
                "index": 37,
                "timestep": 291,
                "sigma": 0.2911904454231262,
            },
        )
        self.assertEqual(receipt["source_geometry"], [21, 37, 25])
        self.assertEqual(receipt["role_asset_sha256"], role_asset_v15b.ASSET_SHA256)
        self.assertEqual(receipt["role_names"], list(role_asset_v15b.ROLE_NAMES))
        self.assertEqual(
            receipt["vessel_competition_group"],
            ["old_actor", "new_actor", "recipient"],
        )
        self.assertEqual(receipt["independent_roles"], ["agent", "support"])
        self.assertFalse(receipt["action_success_authorized"])
        self.assertTrue(receipt["adapters_disabled"])
        self.assertFalse(receipt["route_authorized"])
        self.assertFalse(receipt["training_authorized"])
        self.assertFalse(receipt["decode_authorized"])

    def test_v15b_asset_binds_instance_roles_and_only_vessels_compete(self):
        spec, raw = role_asset_v15b.load_e00_v15b_asset()
        self.assertEqual(spec.event_sha256, role_asset_v15b.EVENT_SHA256)
        self.assertEqual(spec.role_names, role_asset_v15b.ROLE_NAMES)
        self.assertEqual(
            tuple(item.kind for item in spec.roles),
            (
                "agent_instance",
                "vessel_instance",
                "vessel_instance",
                "vessel_instance",
                "support",
            ),
        )
        self.assertEqual(
            raw["competition_groups"]["vessel_instances"],
            ["old_actor", "new_actor", "recipient"],
        )
        self.assertEqual(raw["independent_roles"], ["agent", "support"])
        self.assertTrue(
            raw["semantic_contract"]
            ["labels_are_source_instance_descriptors_not_action_ground_truth"]
        )
        self.assertFalse(
            raw["semantic_contract"]["prompt_action_success_authorized"]
        )

    def test_source_manifest_requires_real_source_clean_and_independent_noise(self):
        valid = source_manifest_fixture()
        adapter._validate_source_manifest(valid)
        mutations = (
            ("input", "source_video_sha256", "0" * 64),
            ("source_condition_artifact", "source_video_vae_encode_before_any_decode", False),
            ("initial_noise_artifacts.rv2v", "source_or_target_derived", True),
            ("runtime_versions", "transformers", "5.5.3"),
        )
        for section, key, value in mutations:
            changed = source_manifest_fixture()
            target = changed
            for part in section.split("."):
                target = target[part]
            target[key] = value
            with self.assertRaises(adapter.AUHSourceRoleAdapterError):
                adapter._validate_source_manifest(changed)

    def test_observer_contract_is_exact_frozen_adapter_off_world4(self):
        value = object.__new__(adapter.AUHBerniniSourceRoleRuntimeAdapter)
        value.selected_block_indices = adapter.SELECTED_BLOCKS
        value.model = FrozenModel()
        contract = value.observer_contract()
        self.assertEqual(
            set(contract),
            {
                "schema_version",
                "checkpoint_sha256",
                "source_manifest_sha256",
                "source_is_real_video",
                "frozen_base",
                "eval_mode",
                "adapters_disabled",
                "ulysses_group_is_world",
                "world_size",
                "selected_block_indices",
                "observer_only",
                "training_authorized",
                "route_authorized",
            },
        )
        self.assertEqual(contract["schema_version"], adapter.ADAPTER_SCHEMA_VERSION)
        self.assertTrue(contract["adapters_disabled"])
        self.assertFalse(contract["training_authorized"])
        self.assertFalse(contract["route_authorized"])

    def test_binding_receipt_is_defensive_copy(self):
        value = object.__new__(adapter.AUHBerniniSourceRoleRuntimeAdapter)
        value._binding_receipt = {"nested": {"sha256": "1" * 64}}
        first = value.binding_receipt()
        first["nested"]["sha256"] = "2" * 64
        self.assertEqual(value.binding_receipt()["nested"]["sha256"], "1" * 64)

    def test_factory_is_the_module_factory_and_remains_lazy_under_patch(self):
        sentinel = object()
        with mock.patch.object(
            adapter, "AUHBerniniSourceRoleRuntimeAdapter", return_value=sentinel
        ) as constructor:
            result = adapter.create_auh_bernini_source_role_adapter({})
        self.assertIs(result, sentinel)
        constructor.assert_called_once()
        self.assertEqual(
            constructor.call_args.args[0].selected_block_indices,
            adapter.SELECTED_BLOCKS,
        )

    def test_remote_import_preflight_is_read_only_then_sp4(self):
        plan = adapter.remote_import_preflight_plan()
        self.assertEqual(plan["schema_version"], adapter.PREFLIGHT_SCHEMA_VERSION)
        self.assertEqual(
            plan["module_factory"],
            "auh_source_owned_role_locator_v15_adapter:create_auh_bernini_source_role_adapter",
        )
        self.assertIn("no_model_construction_no_GPU_no_output_write", plan["checks"])
        self.assertRegex(plan["plan_sha256"], r"^[0-9a-f]{64}$")

    def test_source_has_real_prepare_full_forward_and_no_training_decode_route(self):
        source_path = METHOD_ROOT / "auh_source_owned_role_locator_v15_adapter.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("prepare_inputs_for_sp", calls)
        self.assertIn("condition_embedder", calls)
        self.assertIn("gather_outputs", source)
        self.assertNotIn("backward", calls)
        self.assertNotIn("step", calls)
        self.assertNotIn("zero_grad", calls)
        self.assertNotIn("decode", calls)
        self.assertNotIn("optimizer", source.lower())
        self.assertNotIn("PeftModel", source)
        self.assertNotIn("action_anchor", source)
        self.assertNotIn("appearance_donor", source)
        # Bernini imports are factory-time only; module import itself stays CPU/static.
        top_imports = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        self.assertFalse(
            any(
                (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("bernini"))
                or (
                    isinstance(node, ast.Import)
                    and any(alias.name.startswith("bernini") for alias in node.names)
                )
                for node in top_imports
            )
        )


if __name__ == "__main__":
    unittest.main()
