from __future__ import annotations

import ast
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

import probe_source_owned_role_locator_v15b_sp4 as probe  # noqa: E402
import source_owned_role_locator_v15b_e00_asset as asset  # noqa: E402


class V15BAssetAndProbeTests(unittest.TestCase):
    def test_asset_is_exact_instance_level_and_not_action_truth(self):
        spec, raw = asset.load_e00_v15b_asset()
        self.assertEqual(spec.role_names, asset.ROLE_NAMES)
        self.assertEqual(
            tuple(item.substring for item in spec.roles),
            (
                "An East Asian woman",
                "a white bowl",
                "a clear glass pitcher",
                "a small white cup",
                "a wooden tea table",
            ),
        )
        self.assertEqual(
            raw["competition_groups"],
            {"vessel_instances": ["old_actor", "new_actor", "recipient"]},
        )
        self.assertEqual(raw["independent_roles"], ["agent", "support"])
        self.assertFalse(raw["semantic_contract"]["prompt_action_success_authorized"])
        self.assertFalse(raw["route_authorized"])
        self.assertFalse(raw["training_authorized"])
        self.assertFalse(raw["decode_authorized"])

    def test_group_masks_never_cross_compete_agent_or_support(self):
        scores = torch.zeros((5, 21, 37, 25), dtype=torch.float32)
        null = torch.zeros((21, 37, 25), dtype=torch.float32)
        shuffled = torch.zeros_like(scores)
        # All five roles deliberately peak at one site.  Independent roles may
        # overlap it, while only one of the three vessel peers may win it.
        scores[:, :, 10, 10] = torch.tensor([0.8, 0.7, 0.9, 0.6, 0.85]).view(5, 1)
        scores[:, :, 0, 0] = 0.1  # nonzero spatial variance and control margin
        masks, receipt = probe._diagnostic_group_masks(
            scores,
            null,
            shuffled,
            role_names=asset.ROLE_NAMES,
            minimum_spatial_std=0.001,
            minimum_absolute_affinity=0.05,
            minimum_null_margin=0.01,
            minimum_shuffled_margin=0.01,
            minimum_peer_margin=0.01,
        )
        agent = asset.ROLE_NAMES.index("agent")
        old = asset.ROLE_NAMES.index("old_actor")
        new = asset.ROLE_NAMES.index("new_actor")
        recipient = asset.ROLE_NAMES.index("recipient")
        support = asset.ROLE_NAMES.index("support")
        self.assertTrue(bool(masks[agent, :, 10, 10].all()))
        self.assertTrue(bool(masks[support, :, 10, 10].all()))
        self.assertFalse(bool(masks[old, :, 10, 10].any()))
        self.assertTrue(bool(masks[new, :, 10, 10].all()))
        self.assertFalse(bool(masks[recipient, :, 10, 10].any()))
        self.assertFalse(receipt["policy"]["cross_layer_winner_take_all"])
        self.assertFalse(receipt["semantic_localization_certified"])
        self.assertFalse(receipt["action_success_certified"])

    def test_no_forced_nonempty_fails_closed(self):
        scores = torch.zeros((5, 21, 37, 25), dtype=torch.float32)
        masks, receipt = probe._diagnostic_group_masks(
            scores,
            torch.zeros((21, 37, 25), dtype=torch.float32),
            torch.zeros_like(scores),
            role_names=asset.ROLE_NAMES,
        )
        self.assertFalse(bool(masks.any()))
        self.assertFalse(receipt["mechanical_candidate_qualified"])
        self.assertEqual(
            receipt["status"], "requires_heatmap_overlay_and_instance_ROI_audit"
        )

    def test_harness_contains_no_train_decode_or_route_calls(self):
        path = METHOD_ROOT / "probe_source_owned_role_locator_v15b_sp4.py"
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
        self.assertIn("route_authorized", source)
        self.assertIn("training_authorized", source)
        self.assertIn("decode_authorized", source)


if __name__ == "__main__":
    unittest.main()
