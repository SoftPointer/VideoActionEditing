from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import cross_mode_branches as cross_mode


class CrossModePureContractTests(unittest.TestCase):
    def test_torch_is_not_an_import_time_dependency(self) -> None:
        tree = ast.parse(Path(cross_mode.__file__).read_text(encoding="utf-8"))
        eager = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                eager.extend(alias.name for alias in node.names if alias.name == "torch")
            elif isinstance(node, ast.ImportFrom) and node.module == "torch":
                eager.append(node.module)
        self.assertEqual(eager, [])
        self.assertEqual(
            cross_mode.TEXT_FIELDS,
            ("input_ids", "attention_mask", "t5_input_lens"),
        )


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class CrossModeTensorTests(unittest.TestCase):
    def _fixture(self, target_tokens: int = 5):
        total = target_tokens * 2
        action = {
            "input_ids": torch.tensor([[11, 12, 13]], dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "t5_input_lens": torch.tensor([[3]], dtype=torch.long),
            "input_vae_latents": torch.arange(
                total * 8, dtype=torch.bfloat16
            ).reshape(total, 2, 1, 2, 2),
            "input_vae_rope": torch.arange(
                total * 24, dtype=torch.float32
            ).reshape(total, 4, 6),
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
            "provenance": {"row": 17},
        }
        generator_action_text = {
            "input_ids": torch.tensor([[41, 42, 43, 44]], dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "t5_input_lens": torch.tensor([[4]], dtype=torch.long),
        }
        negative = {
            "input_ids": torch.tensor([[91, 92]], dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "t5_input_lens": torch.tensor([[2]], dtype=torch.long),
        }
        return action, generator_action_text, negative

    def _build(self):
        action, generator_action_text, negative = self._fixture()
        return (
            action,
            generator_action_text,
            negative,
            cross_mode.build_generator_branches(
                action, generator_action_text, negative
            ),
        )

    def test_target_tail_is_a_view_and_all_diffusion_state_is_shared(self) -> None:
        action, generator_action_text, negative, branches = self._build()
        generator = branches.generator_action
        uncond = branches.generator_negative
        target_tokens = int(action["target_lens"].item())

        self.assertIs(branches.editor_action, action)
        self.assertTrue(
            torch.equal(
                generator["input_vae_latents"],
                action["input_vae_latents"][target_tokens:],
            )
        )
        self.assertTrue(
            torch.equal(
                generator["input_vae_rope"],
                action["input_vae_rope"][target_tokens:],
            )
        )
        self.assertEqual(
            generator["input_vae_latents"].untyped_storage().data_ptr(),
            action["input_vae_latents"].untyped_storage().data_ptr(),
        )
        self.assertEqual(
            generator["input_vae_rope"].untyped_storage().data_ptr(),
            action["input_vae_rope"].untyped_storage().data_ptr(),
        )
        self.assertEqual(tuple(generator["vae_latents_mask"].shape), (1, 5))
        self.assertTrue(bool(generator["vae_latents_mask"].all()))
        self.assertEqual(int(generator["vae_seqlen"].item()), 5)
        self.assertEqual(int(generator["vlm_seqlen"].item()), 4)
        self.assertEqual(int(generator["num_tokens"].item()), 9)
        self.assertEqual(int(uncond["vlm_seqlen"].item()), 2)
        self.assertEqual(int(uncond["num_tokens"].item()), 7)

        for field in ("timesteps", "target_velocity", "target_lens"):
            self.assertIs(generator[field], action[field])
            self.assertIs(uncond[field], generator[field])
        for field in cross_mode.TEXT_FIELDS:
            self.assertIs(generator[field], generator_action_text[field])
            self.assertIs(uncond[field], negative[field])
        for field in generator:
            if field not in (
                *cross_mode.TEXT_FIELDS,
                "vlm_seqlen",
                "num_tokens",
            ):
                self.assertIs(uncond[field], generator[field])

    def test_builder_does_not_advance_torch_rng_state(self) -> None:
        action, generator_action_text, negative = self._fixture()
        before = torch.random.get_rng_state().clone()
        cross_mode.build_generator_branches(
            action, generator_action_text, negative
        )
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(after, before))

    def test_equal_clone_is_rejected_because_tail_must_be_a_storage_view(self) -> None:
        action, generator_action_text, negative, branches = self._build()
        bad_action = dict(branches.generator_action)
        bad_action["input_vae_latents"] = bad_action[
            "input_vae_latents"
        ].clone()
        self.assertTrue(
            torch.equal(
                bad_action["input_vae_latents"],
                branches.generator_action["input_vae_latents"],
            )
        )
        with self.assertRaisesRegex(
            cross_mode.CrossModeBranchError, "direct storage view"
        ):
            cross_mode.validate_cross_mode_branches(
                action,
                bad_action,
                branches.generator_negative,
                generator_action_text_fields=generator_action_text,
                generator_negative_text_fields=negative,
            )

    def test_builder_does_not_mutate_full_editor_batch(self) -> None:
        action, generator_action_text, negative = self._fixture()
        original = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in action.items()
        }
        cross_mode.build_generator_branches(
            action, generator_action_text, negative
        )
        for key, expected in original.items():
            if isinstance(expected, torch.Tensor):
                self.assertTrue(torch.equal(action[key], expected), key)
            else:
                self.assertIs(action[key], expected)
        self.assertEqual(tuple(action["input_vae_latents"].shape), (10, 2, 1, 2, 2))
        self.assertEqual(int(action["vae_seqlen"].item()), 10)

    def test_validator_intentionally_accepts_cross_mode_geometry(self) -> None:
        action, generator_action_text, negative, branches = self._build()
        self.assertNotEqual(
            tuple(action["input_vae_latents"].shape),
            tuple(branches.generator_action["input_vae_latents"].shape),
        )
        cross_mode.validate_cross_mode_branches(
            action,
            branches.generator_action,
            branches.generator_negative,
            generator_action_text_fields=generator_action_text,
            generator_negative_text_fields=negative,
        )

    def test_each_editor_target_state_mismatch_fails_closed(self) -> None:
        cases = {
            "input_vae_latents": "noisy state differs",
            "input_vae_rope": "rope differs",
            "timesteps": "timesteps differs",
            "target_velocity": "target_velocity differs",
            "target_lens": "target_lens differs",
        }
        for field, message in cases.items():
            with self.subTest(field=field):
                action, generator_action_text, negative, branches = self._build()
                bad_action = dict(branches.generator_action)
                bad_action[field] = bad_action[field].clone()
                bad_action[field].reshape(-1)[0] += 1
                with self.assertRaisesRegex(cross_mode.CrossModeBranchError, message):
                    cross_mode.validate_cross_mode_branches(
                        action,
                        bad_action,
                        branches.generator_negative,
                        generator_action_text_fields=generator_action_text,
                        generator_negative_text_fields=negative,
                    )

    def test_negative_branch_state_and_nontext_mutations_fail_closed(self) -> None:
        action, generator_action_text, negative, branches = self._build()
        bad_negative = dict(branches.generator_negative)
        bad_negative["input_vae_latents"] = bad_negative[
            "input_vae_latents"
        ].clone()
        bad_negative["input_vae_latents"].reshape(-1)[-1] += 1
        with self.assertRaisesRegex(
            cross_mode.CrossModeBranchError, "noisy state differs"
        ):
            cross_mode.validate_cross_mode_branches(
                action,
                branches.generator_action,
                bad_negative,
                generator_action_text_fields=generator_action_text,
                generator_negative_text_fields=negative,
            )

        bad_negative = dict(branches.generator_negative)
        bad_negative["provenance"] = {"row": 18}
        with self.assertRaisesRegex(
            cross_mode.CrossModeBranchError, "changed non-text field provenance"
        ):
            cross_mode.validate_cross_mode_branches(
                action,
                branches.generator_action,
                bad_negative,
                generator_action_text_fields=generator_action_text,
                generator_negative_text_fields=negative,
            )

    def test_generator_action_must_use_supplied_t2v_not_editor_text(self) -> None:
        action, generator_action_text, negative, branches = self._build()
        bad_action = dict(branches.generator_action)
        bad_action["input_ids"] = bad_action["input_ids"].clone()
        bad_action["input_ids"][0, 0] += 1
        with self.assertRaisesRegex(
            cross_mode.CrossModeBranchError, "differs from supplied text"
        ):
            cross_mode.validate_cross_mode_branches(
                action,
                bad_action,
                branches.generator_negative,
                generator_action_text_fields=generator_action_text,
                generator_negative_text_fields=negative,
            )

        with self.assertRaisesRegex(
            cross_mode.CrossModeBranchError, "distinct official T2V text"
        ):
            cross_mode.build_generator_branches(
                action,
                {field: action[field] for field in cross_mode.TEXT_FIELDS},
                negative,
            )

    def test_generator_action_cannot_change_unrelated_fields(self) -> None:
        action, generator_action_text, negative, branches = self._build()
        bad_action = dict(branches.generator_action)
        bad_action["provenance"] = {"row": 99}
        with self.assertRaisesRegex(
            cross_mode.CrossModeBranchError,
            "changed non-geometry field provenance",
        ):
            cross_mode.validate_cross_mode_branches(
                action,
                bad_action,
                branches.generator_negative,
                generator_action_text_fields=generator_action_text,
                generator_negative_text_fields=negative,
            )

    def test_num_tokens_and_vlm_seqlen_are_branch_specific_and_strict(self) -> None:
        action, generator_action_text, negative, branches = self._build()
        self.assertEqual(
            int(branches.generator_action["num_tokens"].item()),
            int(branches.generator_action["vae_seqlen"].item())
            + int(branches.generator_action["vlm_seqlen"].item()),
        )
        self.assertEqual(
            int(branches.generator_negative["num_tokens"].item()),
            int(branches.generator_negative["vae_seqlen"].item())
            + int(branches.generator_negative["vlm_seqlen"].item()),
        )

        for branch_name, field, message in (
            ("generator_action", "vlm_seqlen", "vlm_seqlen"),
            ("generator_action", "num_tokens", "num_tokens"),
            ("generator_negative", "vlm_seqlen", "vlm_seqlen"),
            ("generator_negative", "num_tokens", "num_tokens"),
        ):
            with self.subTest(branch=branch_name, field=field):
                bad_action = dict(branches.generator_action)
                bad_negative = dict(branches.generator_negative)
                target = bad_action if branch_name == "generator_action" else bad_negative
                target[field] = target[field].clone()
                target[field].reshape(-1)[0] += 1
                with self.assertRaisesRegex(
                    cross_mode.CrossModeBranchError, message
                ):
                    cross_mode.validate_cross_mode_branches(
                        action,
                        bad_action,
                        bad_negative,
                        generator_action_text_fields=generator_action_text,
                        generator_negative_text_fields=negative,
                    )

    def test_negative_text_must_be_exactly_the_supplied_three_fields(self) -> None:
        action, generator_action_text, negative, branches = self._build()
        wrong = dict(negative)
        wrong["input_ids"] = wrong["input_ids"].clone()
        wrong["input_ids"][0, 0] += 1
        with self.assertRaisesRegex(
            cross_mode.CrossModeBranchError, "differs from supplied text"
        ):
            cross_mode.validate_cross_mode_branches(
                action,
                branches.generator_action,
                branches.generator_negative,
                generator_action_text_fields=generator_action_text,
                generator_negative_text_fields=wrong,
            )

        with self.assertRaisesRegex(
            cross_mode.CrossModeBranchError, "exactly the three T5 text fields"
        ):
            cross_mode.build_generator_branches(
                action,
                generator_action_text,
                {**negative, "unexpected": torch.tensor([1])},
            )

    def test_malformed_full_source_batch_is_rejected(self) -> None:
        action, generator_action_text, negative = self._fixture()
        action["vae_latents_mask"] = torch.tensor(
            [[False, True, False, False, False, True, True, True, True, True]]
        )
        with self.assertRaisesRegex(
            cross_mode.CrossModeBranchError, "contiguous target tail"
        ):
            cross_mode.build_generator_branches(
                action, generator_action_text, negative
            )

        action, generator_action_text, negative = self._fixture()
        action["vae_seqlen"] = action["vae_seqlen"].float()
        with self.assertRaisesRegex(
            cross_mode.CrossModeBranchError, "must have an integer dtype"
        ):
            cross_mode.build_generator_branches(
                action, generator_action_text, negative
            )


if __name__ == "__main__":
    unittest.main()
