from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from action.config import (  # noqa: E402
    ACTION_CONFIG_FORMAT,
    ActionConfig,
    ActionConfigError,
    load_action_config,
)


def action_config() -> dict:
    return {
        "format": ACTION_CONFIG_FORMAT,
        "seed": 20260805,
        "data": {
            "video_num_frames": 81,
            "video_fps": 25.0,
            "video_height": 480,
            "video_width": 832,
            "temporal_mode": "full_81_25fps",
            "spatial_profile": "full_480p",
            "allow_transpose": True,
            "smoke_only": False,
            "require_materialization_metadata": True,
        },
        "model": {
            "max_context_len": 9216,
            "checkpoint_contract_id": "omnivideo2-1.3b-adcee0a4-f269fe8c-72129ce9-v1",
            "context_padding_mode": "fixed_budget",
            "expected_special_token_rows": 26,
            "visual_patch_size": [1, 4, 4],
            "wan_patch_size": [1, 2, 2],
            "require_special_tokens": True,
            "require_uncompressed_source": True,
            "gradient_checkpointing": True,
        },
        "flow": {"shift": 5.0, "num_train_timesteps": 1000},
        "lora": {
            "scope": "cross_qo",
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.05,
        },
        "planner": {
            "num_tokens": 8,
            "input_dim": 2048,
            "hidden_dim": 256,
            "depth": 2,
            "weight": 0.25,
        },
        "optimizer": {
            "learning_rate": 1e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "eps": 1e-8,
        },
        "training": {
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_steps": 0,
            "num_workers": 0,
            "mixed_precision": "bf16",
            "log_every": 1,
            "save_every": 100,
            "allow_preview": False,
            "allowed_task_types": [
                "action_edit",
                "identity_reconstruction",
                "native_replay",
                "native_isolation_probe",
            ],
        },
    }


class ActionConfigTest(unittest.TestCase):
    def test_round_trip_and_nested_types(self) -> None:
        raw = action_config()
        checked = ActionConfig.from_mapping(raw)
        self.assertEqual(checked.to_dict(), raw)
        self.assertEqual(checked.planner.input_dim, 2048)
        self.assertEqual(checked.model.visual_patch_size, (1, 4, 4))
        self.assertEqual(
            checked.data.expected_latent_shapes,
            ((16, 21, 60, 104), (16, 21, 104, 60)),
        )
        self.assertFalse(checked.training.allow_preview)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            self.assertEqual(load_action_config(path), checked)

    def test_top_level_and_nested_schemas_are_closed(self) -> None:
        extra = action_config()
        extra["legacy_router"] = {}
        with self.assertRaisesRegex(ActionConfigError, "unknown"):
            ActionConfig.from_mapping(extra)

        missing = action_config()
        del missing["planner"]["weight"]
        with self.assertRaisesRegex(ActionConfigError, "missing"):
            ActionConfig.from_mapping(missing)

        nested_extra = action_config()
        nested_extra["flow"]["sampling"] = "continuous"
        with self.assertRaisesRegex(ActionConfigError, "unknown"):
            ActionConfig.from_mapping(nested_extra)

    def test_fixed_dimensions_and_numeric_domains_fail_closed(self) -> None:
        cases = []
        wrong_input = action_config()
        wrong_input["planner"]["input_dim"] = 1024
        cases.append(("input", "exactly 2048", wrong_input))

        bad_hidden = action_config()
        bad_hidden["planner"]["hidden_dim"] = 258
        cases.append(("hidden", "divisible", bad_hidden))

        bool_seed = action_config()
        bool_seed["seed"] = True
        cases.append(("seed", "integer", bool_seed))

        bad_dropout = action_config()
        bad_dropout["lora"]["dropout"] = 1.0
        cases.append(("dropout", "smaller than", bad_dropout))

        wrong_preview = action_config()
        wrong_preview["training"]["allow_preview"] = 0
        cases.append(("preview", "bool", wrong_preview))

        wrong_scope = action_config()
        wrong_scope["lora"]["scope"] = "everything"
        cases.append(("scope", "one of", wrong_scope))

        for label, message, value in cases:
            with self.subTest(label=label), self.assertRaisesRegex(
                ActionConfigError, message
            ):
                ActionConfig.from_mapping(value)

    def test_temporal_geometry_and_padding_policy_fail_closed(self) -> None:
        legacy = action_config()
        legacy["format"] = "marp-omnivideo2-action-training-v1"
        with self.assertRaisesRegex(ActionConfigError, "format"):
            ActionConfig.from_mapping(legacy)

        unlabelled_41 = action_config()
        unlabelled_41["data"].update(
            {"video_num_frames": 41, "video_fps": 12.5}
        )
        with self.assertRaisesRegex(ActionConfigError, "temporal_mode"):
            ActionConfig.from_mapping(unlabelled_41)

        bad_frames = action_config()
        bad_frames["data"]["video_num_frames"] = 80
        with self.assertRaisesRegex(ActionConfigError, "4n\+1"):
            ActionConfig.from_mapping(bad_frames)

        batch_exact = action_config()
        batch_exact["model"]["context_padding_mode"] = "batch_exact"
        batch_exact["training"]["batch_size"] = 2
        with self.assertRaisesRegex(ActionConfigError, "batch_exact"):
            ActionConfig.from_mapping(batch_exact)

        fixed_batch = action_config()
        fixed_batch["training"]["batch_size"] = 2
        self.assertEqual(ActionConfig.from_mapping(fixed_batch).training.batch_size, 2)

        wrong_official_rows = action_config()
        wrong_official_rows["model"]["expected_special_token_rows"] = 4
        with self.assertRaisesRegex(ActionConfigError, "pinned official"):
            ActionConfig.from_mapping(wrong_official_rows)

        wrong_contract = action_config()
        wrong_contract["model"]["checkpoint_contract_id"] = "another-checkpoint"
        with self.assertRaisesRegex(ActionConfigError, "checkpoint_contract_id"):
            ActionConfig.from_mapping(wrong_contract)

        missing_special_tokens = action_config()
        missing_special_tokens["model"]["require_special_tokens"] = False
        with self.assertRaisesRegex(ActionConfigError, "require_special_tokens"):
            ActionConfig.from_mapping(missing_special_tokens)

    def test_checked_in_81_frame_profiles_are_closed(self) -> None:
        config_root = ROOT / "configs"
        full = load_action_config(config_root / "marp_1_3b.json")
        low = load_action_config(
            config_root / "marp_1_3b_81f_640x384_ctx6144.json"
        )
        full_step = load_action_config(
            config_root / "marp_81f_fullres_real_one_step.json"
        )
        low_step = load_action_config(
            config_root / "marp_81f_motion384_real_one_step.json"
        )
        smoke = load_action_config(
            config_root / "marp_smoke_41f_real_one_step.json"
        )
        self.assertEqual(full.data.video_num_frames, 81)
        self.assertEqual(full.model.max_context_len, 9216)
        self.assertEqual(full.model.context_padding_mode, "fixed_budget")
        self.assertEqual(full.model.expected_special_token_rows, 26)
        self.assertEqual(low.data.expected_latent_shape, (16, 21, 48, 80))
        self.assertEqual(low.model.max_context_len, 6144)
        self.assertEqual(low.model.expected_special_token_rows, 26)
        self.assertFalse(low.data.smoke_only)
        self.assertEqual(full_step.data.expected_latent_shape, (16, 21, 60, 104))
        self.assertEqual(full_step.model.max_context_len, 9216)
        self.assertEqual(low_step.data.expected_latent_shape, (16, 21, 48, 80))
        self.assertEqual(low_step.model.max_context_len, 6144)
        self.assertEqual(smoke.data.video_num_frames, 41)
        self.assertEqual(smoke.data.expected_latent_shape, (16, 11, 60, 104))
        self.assertTrue(smoke.data.smoke_only)


if __name__ == "__main__":
    unittest.main()
