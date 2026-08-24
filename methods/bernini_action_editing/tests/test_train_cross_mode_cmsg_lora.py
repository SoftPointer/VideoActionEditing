from __future__ import annotations

import ast
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_cross_mode_cmsg_lora as trainer


class CrossModeCMSGTrainingPureTests(unittest.TestCase):
    def test_method_lora_and_condition_contracts_are_explicit(self) -> None:
        self.assertEqual(trainer.NUM_FRAMES, 81)
        self.assertEqual(trainer.LATENT_PHASES, 21)
        self.assertEqual(trainer.LORA_RANK, 8)
        self.assertEqual(trainer.LORA_ALPHA, 8)
        self.assertEqual(trainer.EXPECTED_LORA_MODULES, 46)
        self.assertEqual(
            trainer.INFERENCE_CONDITIONS,
            ("source_video", "action_instruction"),
        )
        self.assertIn("mask", trainer.FORBIDDEN_INFERENCE_CONDITIONS)
        self.assertIn("optical_flow", trainer.FORBIDDEN_INFERENCE_CONDITIONS)
        self.assertEqual(
            trainer.GENERATOR_ACTION_TEXT_CONTRACT,
            "official_t2v_system_prompt_plus_action_instruction",
        )

    def test_exact_cross_plus_middle_self_q_scope(self) -> None:
        available = trainer.canonical_attention_modules()
        selected = trainer.select_cmsg_lora_targets(available)
        self.assertEqual(len(selected), 46)
        cross = [name for name in selected if ".attn2." in name]
        self_q = [name for name in selected if ".attn1." in name]
        self.assertEqual(len(cross), 30)
        self.assertEqual(len(self_q), 16)
        self.assertTrue(all(name.endswith(".to_q") for name in selected))
        self.assertEqual(
            {
                int(name.split(".blocks.")[1].split(".")[0])
                for name in self_q
            },
            set(range(7, 23)),
        )
        contract = trainer.lora_contract(available)
        self.assertEqual(contract["rank"], 8)
        self.assertEqual(contract["alpha"], 8)
        self.assertEqual(contract["target_module_count"], 46)

    def test_scope_fails_closed_on_missing_or_duplicate_modules(self) -> None:
        available = trainer.canonical_attention_modules()
        missing = [
            name
            for name in available
            if name != "diff_dec.transformer.blocks.12.attn1.to_q"
        ]
        with self.assertRaisesRegex(
            trainer.CrossModeCMSGTrainingError, "all 30 cross"
        ):
            trainer.select_cmsg_lora_targets(missing)
        with self.assertRaisesRegex(
            trainer.CrossModeCMSGTrainingError, "duplicates"
        ):
            trainer.select_cmsg_lora_targets(available + [available[0]])

    def test_parser_exposes_only_truthful_preflight(self) -> None:
        destinations = {action.dest for action in trainer.build_parser()._actions}
        self.assertEqual(destinations, {"help", "preflight_only"})
        for forbidden in trainer.FORBIDDEN_INFERENCE_CONDITIONS:
            self.assertNotIn(forbidden, destinations)
        with self.assertRaisesRegex(
            trainer.CrossModeCMSGTrainingError, "not yet integrated"
        ):
            trainer.main([])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(trainer.main(["--preflight-only"]), 0)
        receipt = json.loads(output.getvalue())
        self.assertTrue(receipt["core_api_ready"])
        self.assertFalse(receipt["full_bernini_training_integrated"])
        self.assertFalse(receipt["optimizer_updates_authorized"])
        self.assertEqual(receipt["lora"]["target_module_count"], 46)
        self.assertEqual(
            receipt["generator_action_text_contract"],
            "official_t2v_system_prompt_plus_action_instruction",
        )
        self.assertEqual(
            receipt["inference_conditions"],
            ["source_video", "action_instruction"],
        )
        self.assertIn(
            "frozen_target_only_generator_teacher",
            receipt["training_only_conditions"],
        )

    def test_torch_is_not_an_import_time_dependency(self) -> None:
        tree = ast.parse(Path(trainer.__file__).read_text(encoding="utf-8"))
        eager = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                eager.extend(
                    alias.name for alias in node.names if alias.name == "torch"
                )
            elif isinstance(node, ast.ImportFrom) and node.module == "torch":
                eager.append(node.module)
        self.assertEqual(eager, [])

    def test_loss_configuration_rejects_invalid_values(self) -> None:
        invalid = (
            {"editor_direction_weight": -1.0},
            {"charbonnier_scale": 0.0},
            {"active_relative_floor": 0.0},
            {"gate_min_active_phases": 21},
            {"gate_min_mean_cosine": 2.0},
            {"gate_min_coverage": 1.1},
            {"gate_max_normalized_rmse": float("nan")},
            {"enforce_frozen_prior_gate": 1},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(trainer.CrossModeCMSGTrainingError):
                    trainer.CMSGTrainingLossConfig(**values).validate()


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class CrossModeCMSGTrainingTensorTests(unittest.TestCase):
    @staticmethod
    def _editor_batch(target_tokens: int = 5):
        total = 2 * target_tokens
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "t5_input_lens": torch.tensor([[3]], dtype=torch.long),
            "input_vae_latents": torch.arange(
                total * 8, dtype=torch.bfloat16
            ).reshape(total, 2, 1, 2, 2),
            "input_vae_rope": torch.arange(
                total * 12, dtype=torch.float32
            ).reshape(total, 3, 4),
            "vae_latents_mask": torch.tensor(
                [[False] * target_tokens + [True] * target_tokens]
            ),
            "vae_seqlen": torch.tensor([[total]], dtype=torch.long),
            "timesteps": torch.tensor([[750]], dtype=torch.long),
            "target_velocity": torch.arange(
                target_tokens * 8, dtype=torch.bfloat16
            ).reshape(target_tokens, 2, 1, 2, 2),
            "target_lens": torch.tensor([[target_tokens]], dtype=torch.long),
            "vlm_seqlen": torch.tensor([[3]], dtype=torch.long),
            "num_tokens": torch.tensor([[total + 3]], dtype=torch.long),
            "provenance": {"iid": "fixture"},
        }
        generator_action = {
            "input_ids": torch.tensor([[4, 5, 6, 7]], dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "t5_input_lens": torch.tensor([[4]], dtype=torch.long),
        }
        generator_negative = {
            "input_ids": torch.tensor([[9, 8]], dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "t5_input_lens": torch.tensor([[2]], dtype=torch.long),
        }
        return batch, generator_action, generator_negative

    @staticmethod
    def _direction(*, scale: float = 1.0):
        phase = torch.arange(21, dtype=torch.float32).reshape(1, 21, 1, 1)
        pattern = torch.tensor(
            [
                [[1.0, -0.5, 0.25], [0.5, -1.0, 0.75]],
                [[-0.25, 0.5, 1.0], [1.25, -0.75, 0.5]],
            ],
            dtype=torch.float32,
        ).reshape(1, 1, 4, 3)
        return scale * phase * pattern / 20.0

    def _loss_fields(self, *, adapted_scale: float = 1.05, generator_scale: float = 2.0):
        direction = self._direction()
        editor_noop = torch.zeros_like(direction)
        frozen_editor = direction.clone()
        adapted_editor = (adapted_scale * direction).clone().requires_grad_(True)
        generator_uncond = torch.zeros_like(direction)
        frozen_generator = generator_scale * direction
        target = direction.clone()
        return {
            "adapted_editor_action_field": adapted_editor,
            "frozen_editor_action_field": frozen_editor,
            "editor_noop_field": editor_noop,
            "frozen_generator_action_field": frozen_generator,
            "generator_uncond_field": generator_uncond,
            "target_motion_field": target,
        }

    def test_training_branches_are_direct_target_tail_views(self) -> None:
        editor, generator_text, negative = self._editor_batch()
        branches = trainer.build_training_branches(
            editor, generator_text, negative
        )
        generator = branches.generator_action
        unconditional = branches.generator_negative
        target_tokens = int(editor["target_lens"].item())
        self.assertEqual(
            generator["input_vae_latents"].untyped_storage().data_ptr(),
            editor["input_vae_latents"].untyped_storage().data_ptr(),
        )
        self.assertEqual(
            generator["input_vae_rope"].untyped_storage().data_ptr(),
            editor["input_vae_rope"].untyped_storage().data_ptr(),
        )
        self.assertTrue(
            torch.equal(
                generator["input_vae_latents"],
                editor["input_vae_latents"][target_tokens:],
            )
        )
        self.assertIs(generator["timesteps"], editor["timesteps"])
        self.assertIs(unconditional["timesteps"], editor["timesteps"])
        self.assertIs(
            unconditional["input_vae_latents"],
            generator["input_vae_latents"],
        )
        for field in trainer.cross_mode.TEXT_FIELDS:
            self.assertIs(generator[field], generator_text[field])
            self.assertIs(unconditional[field], negative[field])
            self.assertFalse(torch.equal(generator[field], editor[field]))
        self.assertEqual(int(generator["vlm_seqlen"].item()), 4)
        self.assertEqual(int(generator["num_tokens"].item()), target_tokens + 4)
        self.assertEqual(int(unconditional["vlm_seqlen"].item()), 2)
        self.assertEqual(
            int(unconditional["num_tokens"].item()), target_tokens + 2
        )

    def test_editor_mv2v_text_cannot_be_reused_as_generator_t2v_text(self) -> None:
        editor, _, negative = self._editor_batch()
        reused_editor_text = {
            field: editor[field] for field in trainer.cross_mode.TEXT_FIELDS
        }
        with self.assertRaisesRegex(
            trainer.CrossModeCMSGTrainingError,
            "T2V|editor|distinct|differ",
        ):
            trainer.build_training_branches(
                editor, reused_editor_text, negative
            )

    def test_frozen_prior_gate_accepts_aligned_and_rejects_opposite_motion(self) -> None:
        direction = self._direction()
        accepted = trainer.compute_frozen_prior_gate(direction, direction)
        self.assertTrue(bool(accepted.passed.all()))
        self.assertEqual(int(accepted.active_phase_count.item()), 20)
        self.assertGreater(float(accepted.mean_direction_cosine.item()), 0.99)
        self.assertLess(float(accepted.log_amplitude_mae.item()), 1e-4)
        self.assertGreater(float(accepted.covered_phase_fraction.item()), 0.99)
        rejected = trainer.compute_frozen_prior_gate(direction, -direction)
        self.assertFalse(bool(rejected.passed.any()))
        self.assertLess(float(rejected.mean_direction_cosine.item()), -0.99)

    def test_all_five_losses_are_finite_and_lora_gradients_flow(self) -> None:
        fields = self._loss_fields()
        result = trainer.compute_cmsg_lora_loss(
            **fields,
            step_index=0,
            spatial_hw=(2, 2),
        )
        self.assertEqual(result.rho, 1.0)
        self.assertTrue(bool(result.frozen_prior_gate.passed.all()))
        components = (
            result.total,
            result.editor_direction,
            result.log_amplitude,
            result.generator_spectral_consistency,
            result.high_frequency_detail,
            result.late_frozen_replay,
        )
        for value in components:
            self.assertEqual(value.ndim, 0)
            self.assertTrue(bool(torch.isfinite(value)))
            self.assertGreaterEqual(float(value.detach()), 0.0)
        result.total.backward()
        self.assertIsNotNone(fields["adapted_editor_action_field"].grad)
        self.assertTrue(
            bool(torch.isfinite(fields["adapted_editor_action_field"].grad).all())
        )
        self.assertGreater(
            float(fields["adapted_editor_action_field"].grad.abs().sum()), 0.0
        )

    def test_gate_rejection_happens_before_an_update(self) -> None:
        fields = self._loss_fields()
        fields["target_motion_field"] = -fields["target_motion_field"]
        with self.assertRaisesRegex(
            trainer.FrozenPriorGateRejected, "failed direction"
        ):
            trainer.compute_cmsg_lora_loss(
                **fields,
                step_index=0,
                spatial_hw=(2, 2),
            )

    def test_late_zero_release_is_exact_frozen_object_and_only_replay_trains(self) -> None:
        fields = self._loss_fields(adapted_scale=1.2)
        # A deliberately ineligible teacher must remain diagnostic-only once
        # rho is exactly zero; late preservation replay cannot be starved by
        # an unused cross-mode prior.
        fields["target_motion_field"] = -fields["target_motion_field"]
        config = trainer.CMSGTrainingLossConfig()
        result = trainer.compute_cmsg_lora_loss(
            **fields,
            step_index=31,
            spatial_hw=(2, 2),
            loss_config=config,
        )
        self.assertEqual(result.rho, 0.0)
        self.assertIs(
            result.student_execution.executed_field,
            result.frozen_editor_direction,
        )
        self.assertFalse(bool(result.frozen_prior_gate.passed.all().item()))
        expected = config.late_frozen_replay_weight * result.late_frozen_replay
        self.assertTrue(torch.equal(result.total, expected))
        result.total.backward()
        self.assertGreater(
            float(fields["adapted_editor_action_field"].grad.abs().sum()), 0.0
        )

    def test_direction_and_log_amplitude_are_not_conflated(self) -> None:
        fields = self._loss_fields(adapted_scale=2.0)
        # Disable the gate only to isolate the student component numerically;
        # the frozen prior itself remains aligned and would pass either way.
        result = trainer.compute_cmsg_lora_loss(
            **fields,
            step_index=0,
            spatial_hw=(2, 2),
        )
        self.assertLess(float(result.editor_direction.detach()), 1e-4)
        self.assertGreater(float(result.log_amplitude.detach()), 0.1)

    def test_generator_spectrum_and_high_frequency_losses_detect_drift(self) -> None:
        base = self._loss_fields(adapted_scale=1.0, generator_scale=1.0)
        unchanged = trainer.compute_cmsg_lora_loss(
            **base,
            step_index=0,
            spatial_hw=(2, 2),
        )
        self.assertLess(
            float(unchanged.generator_spectral_consistency.detach()), 2e-3
        )

        drifted = self._loss_fields(adapted_scale=1.0, generator_scale=1.0)
        checker = torch.tensor(
            [[1.0, -1.0, 1.0, -1.0]], dtype=torch.float32
        ).reshape(1, 1, 4, 1)
        temporal_ramp = torch.arange(21, dtype=torch.float32).reshape(
            1, 21, 1, 1
        ) / 20.0
        drifted_editor = (
            drifted["adapted_editor_action_field"].detach()
            + 0.2 * temporal_ramp * checker
        ).requires_grad_(True)
        drifted["adapted_editor_action_field"] = drifted_editor
        drifted["frozen_generator_action_field"] = (
            1.5 * drifted["frozen_generator_action_field"]
        )
        changed = trainer.compute_cmsg_lora_loss(
            **drifted,
            step_index=0,
            spatial_hw=(2, 2),
        )
        self.assertGreater(
            float(changed.generator_spectral_consistency.detach()), 1e-4
        )
        self.assertGreater(float(changed.high_frequency_detail.detach()), 1e-4)

    def test_frozen_fields_cannot_secretly_carry_gradients(self) -> None:
        fields = self._loss_fields()
        fields["frozen_generator_action_field"] = fields[
            "frozen_generator_action_field"
        ].requires_grad_(True)
        with self.assertRaisesRegex(
            trainer.CrossModeCMSGTrainingError, "graph-free"
        ):
            trainer.compute_cmsg_lora_loss(
                **fields,
                step_index=0,
                spatial_hw=(2, 2),
            )


if __name__ == "__main__":
    unittest.main()
