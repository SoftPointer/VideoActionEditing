from __future__ import annotations

from pathlib import Path
import hashlib
import unittest
from unittest import mock

from methods.bernini_action_editing import (
    infer_mev840_native_rv2v_paired_prompt_matrix_v1 as runner,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "assets" / "mev840_native_rv2v_same_process_prompt_matrix_v2.json"


def _identity(tag: str) -> dict[str, object]:
    digest = hashlib.sha256(tag.encode("ascii")).hexdigest()
    return {
        "shape": [2],
        "dtype": "torch.float32",
        "numel": 2,
        "byte_count": 8,
        "content_sha256": digest,
        "raw_storage_sha256": digest,
    }


def _scheduler_state(*, lower: int, timestep_list: list[int | None]) -> dict[str, object]:
    return {
        "label": "ignored",
        "class": "diffusers.UniPCMultistepScheduler",
        "config_sha256": "a" * 64,
        "solver_order": 2,
        "num_inference_steps": 2,
        "timesteps": _identity("timesteps"),
        "sigmas": _identity("sigmas"),
        "model_outputs_none": [True, True] if lower == 0 else [True, False],
        "timestep_list": timestep_list,
        "lower_order_nums": lower,
        "last_sample_present": False,
        "step_index": None if lower == 0 else lower,
        "begin_index": None,
    }


def _scheduler_matrix() -> dict[str, object]:
    rows = {}
    for index, cell in enumerate(runner.EXECUTION_ORDER):
        stale = [900 + index, 901 + index]
        reset = _scheduler_state(lower=0, timestep_list=stale)
        first_before = _scheduler_state(lower=0, timestep_list=stale)
        first_after = _scheduler_state(lower=1, timestep_list=[stale[-1], 10])
        second_before = _scheduler_state(lower=1, timestep_list=[stale[-1], 10])
        second_after = _scheduler_state(lower=1, timestep_list=[10, 0])
        rows[cell] = {
            "set_timesteps": [reset],
            "steps": [
                {"timestep": 10, "before": first_before, "after": first_after},
                {"timestep": 0, "before": second_before, "after": second_after},
            ],
        }
    return rows


class PairedPromptMatrixRunnerTests(unittest.TestCase):
    def test_authority_and_cli_are_two_or_forty_step_fail_closed(self) -> None:
        digest = hashlib.sha256(AUTHORITY.read_bytes()).hexdigest()
        parser = runner.build_parser()
        common = [
            "--bernini-root", "/tmp/bernini",
            "--veomni-root", "/tmp/veomni",
            "--checkpoint", "/tmp/checkpoint",
            "--checkpoint-content-manifest", "/tmp/checkpoint.manifest",
            "--source-video", "/tmp/source.mp4",
            "--expected-source-sha256", "0" * 64,
            "--prompt-matrix-authority", str(AUTHORITY),
            "--expected-prompt-matrix-authority-sha256", digest,
            "--output-dir", "/tmp/mev840-paired-canary-test-output",
            "--seed", "2028",
            "--method-source-revision", "0" * 40,
            "--method-source-archive-sha256", "1" * 64,
        ]
        args = parser.parse_args([*common, "--num-inference-steps", "2", "--skip-video-decode"])
        bundle = runner.validate_cli(args)
        self.assertEqual(bundle["authority_sha256"], digest)
        self.assertEqual(bundle["prompts"]["P0"], bundle["prompts"]["P0"])
        bad = parser.parse_args([*common, "--num-inference-steps", "40", "--skip-video-decode"])
        with self.assertRaisesRegex(runner.NativeIdentityCanaryError, "restricted"):
            runner.validate_cli(bad)
        bad = parser.parse_args([*common, "--num-inference-steps", "3"])
        with self.assertRaisesRegex(runner.NativeIdentityCanaryError, "only 2 or 40"):
            runner.validate_cli(bad)

    def test_effective_scheduler_reset_accepts_stale_lists_but_not_state_leakage(self) -> None:
        rows = _scheduler_matrix()
        receipt = runner._validate_scheduler_observations(rows, expected_steps=2)
        self.assertIs(receipt["effective_reset_fields_exact_across_calls"], True)
        self.assertIs(receipt["stale_timestep_list_recorded"], True)
        self.assertIs(receipt["stale_timestep_list_inactive_on_first_order_step"], True)
        self.assertIs(receipt["fresh_predecessor_present_before_order2_step"], True)
        self.assertEqual(receipt["step_count_per_call"], 2)
        rows["p2"]["set_timesteps"][0]["lower_order_nums"] = 1
        with self.assertRaisesRegex(runner.NativeIdentityCanaryError, "effective reset"):
            runner._validate_scheduler_observations(rows, expected_steps=2)

    def test_encode_prompt_observer_forwards_exact_return_and_restores_method(self) -> None:
        class FakeTensor:
            def __init__(self, tag: str) -> None:
                self.tag = tag

            def gt(self, _: int) -> "FakeTensor":
                return self

            def sum(self, *, dim: int) -> "FakeTensor":
                self.dim = dim
                return self

            def detach(self) -> "FakeTensor":
                return self

            def cpu(self) -> "FakeTensor":
                return self

            def tolist(self) -> list[int]:
                return [2]

        class FakeModel:
            def __init__(self) -> None:
                self.returned = FakeTensor("returned")

            def encode_prompt(self, input_ids: FakeTensor, attention_mask: FakeTensor) -> FakeTensor:
                del input_ids, attention_mask
                return self.returned

        model = FakeModel()
        ids = FakeTensor("ids")
        mask = FakeTensor("mask")

        def identity(value: FakeTensor, *, label: str) -> dict[str, object]:
            return {**_identity(value.tag), "label": label}

        with mock.patch.object(runner.value_audit, "tensor_identity", side_effect=identity):
            with runner._observe_encode_prompt(model, cell="p0a") as calls:
                returned = model.encode_prompt(ids, mask)
        self.assertIs(returned, model.returned)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0]["input_ids"]["raw_storage_sha256"],
            _identity("ids")["raw_storage_sha256"],
        )
        self.assertIs(model.encode_prompt(ids, mask), model.returned)

    def test_entrypoint_is_same_process_native_observer_only(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn("for cell in EXECUTION_ORDER:", source)
        self.assertIn("model.sample(", source)
        self.assertIn("_sample_with_native_initial_noise_observer", source)
        self.assertIn("_observe_native_scheduler(scheduler)", source)
        self.assertIn("_observe_encode_prompt(model, cell=cell)", source)
        self.assertIn("P0 native replay latent is not bit exact", source)
        self.assertIn("global torch RNG state changed", source)
        self.assertIn(runner.UNIPC_SOURCE_SHA256, source)
        self.assertIn("mechanical Slurm job/node/step/WORLD4 differs", source)
        self.assertIn("oom_oom_kill_oom_group_kill_delta_zero", source)
        self.assertIn('"observed_module_names": ["transformer_1.rope"]', source)
        self.assertIn('"p1_p2_p0b_full_state_exact_to_p0a": True', source)
        self.assertNotIn("def _fresh_native_scheduler", source)
        self.assertNotIn("def _reset_native_model_runtime_state", source)
        self.assertNotIn("_restore_torch_rng_state", source)
        self.assertNotIn("target_action_oracle", source)


if __name__ == "__main__":
    unittest.main()
