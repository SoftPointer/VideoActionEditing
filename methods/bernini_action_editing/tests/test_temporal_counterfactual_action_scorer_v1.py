#!/usr/bin/env python3

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import temporal_counterfactual_action_scorer_v1 as scorer  # noqa: E402
import temporal_counterfactual_contract_v1 as contract  # noqa: E402


LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_temporal_counterfactual_action_scorer_v1_dual4.sbatch"
)


class TemporalCounterfactualScorerContractTests(unittest.TestCase):
    def test_frozen_runtime_is_a_separate_exact_d541801_source_artifact(self) -> None:
        frozen = METHOD_ROOT / "score_pair_v5_t2v_energy_bank_frozen_d541801.py"
        self.assertTrue(frozen.is_file())
        self.assertEqual(
            contract.file_sha256(frozen),
            contract.REQUIRED_D541801_SCORER_SHA256,
        )
        source = inspect.getsource(scorer._frozen_d541801_runtime)
        self.assertIn("score_pair_v5_t2v_energy_bank_frozen_d541801.py", source)
        self.assertIn(
            '"score_pair_v5_t2v_energy_bank_frozen_d541801"', source
        )
        self.assertNotIn('import_module("score_pair_v5_t2v_energy_bank_v3")', source)

    def test_model_boundary_has_no_source_donor_or_label_argument(self) -> None:
        names = set(inspect.signature(scorer.forward_native_prompt_pair).parameters)
        forbidden = {
            "source",
            "source_video",
            "source_latent",
            "target_video",
            "target_latent",
            "proposal",
            "donor",
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
            "event_label",
        }
        self.assertFalse(names & forbidden)
        self.assertEqual(
            names,
            {
                "diffusion",
                "transformer",
                "x_sigma",
                "native_schedule_index",
                "action_condition",
                "noop_condition",
            },
        )

    def test_real_insertion_point_is_patch_once_then_shared_step_pair(self) -> None:
        source = inspect.getsource(scorer.forward_native_prompt_pair)
        self.assertIn("transformer.patch_vae_latent", source)
        self.assertIn("runtime_contract.build_t2v_target_branch", source)
        self.assertIn("diffusion.shared_step", source)
        self.assertIn("native_scheduler_timestep", contract.make_sigma_coordinate_receipt()["coordinates"][0])
        self.assertIn(
            "zip(PROMPT_ORDER, (action_condition, noop_condition))", source
        )
        self.assertIn("timesteps=timestep", source)
        self.assertEqual(scorer.PROMPT_ORDER, ("target_action", "noop"))
        self.assertEqual(scorer.MODEL_FORWARDS_PER_CANDIDATE, 42)

    def test_prompt_pair_uses_cell_action_and_scene_matched_noop_rows(self) -> None:
        source = Path(scorer.__file__).read_text(encoding="utf-8")
        self.assertIn('semantic_branch"] == contract.ACTION_BRANCH', source)
        self.assertIn('semantic_branch"] == "noop"', source)
        self.assertIn('noop_candidate["full_t2v_caption"]', source)
        self.assertNotIn("NOOP_RAW_CAPTION", source)

    def test_all_temporal_arms_keep_the_same_official_gaussian(self) -> None:
        source = Path(scorer.__file__).read_text(encoding="utf-8")
        self.assertIn("contract.fixed_official_gaussian_tensor", source)
        self.assertNotIn(
            "apply_temporal_transform_tensor(epsilon", source
        )

    def test_same_state_audit_never_reads_inference_tensor_version(self) -> None:
        source = inspect.getsource(scorer.forward_native_prompt_pair)
        self.assertNotIn("._version", source)
        try:
            import torch
        except ImportError:
            self.skipTest("Torch unavailable in CPU contract environment")
        with torch.inference_mode():
            inference_tensor = torch.arange(8, dtype=torch.float32)
        copied = (
            inference_tensor.detach()
            .to(device="cpu")
            .contiguous()
            .clone()
            .view(torch.uint8)
        )
        self.assertGreater(copied.numel(), 0)

    def test_runtime_rejects_later_v4_scorer_under_same_module_name(self) -> None:
        fake_v4 = SimpleNamespace(
            SCORE_RECEIPT_SCHEMA="bernini-pair-v5-frozen-t2v-global-energy-score-v4",
            GROUP_RECEIPT_SCHEMA="bernini-pair-v5-frozen-t2v-global-energy-group-v4",
        )
        with mock.patch.object(
            scorer.contract,
            "file_sha256",
            return_value=contract.REQUIRED_D541801_SCORER_SHA256,
        ), mock.patch.object(
            scorer.importlib, "import_module", return_value=fake_v4
        ), self.assertRaisesRegex(
            scorer.TemporalCounterfactualScoringError,
            "not the formal d541801/v3 authority",
        ):
            scorer._frozen_d541801_runtime()

    def test_three_native_coordinates_are_revalidated_against_runtime(self) -> None:
        sigmas = [0.0] * 40
        timesteps = [0] * 40
        for index, sigma, timestep in contract.NATIVE_SIGMA_COORDINATES:
            sigmas[index] = sigma
            timesteps[index] = timestep
        fake = SimpleNamespace(
            native_schedule=SimpleNamespace(
                PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST=contract.NATIVE_SCHEDULE_DIGEST,
                NATIVE_UNIPC40_SIGMAS=tuple(sigmas),
                NATIVE_UNIPC40_TIMESTEPS=tuple(timesteps),
            )
        )
        scorer.validate_native_coordinate_runtime(fake)
        fake.native_schedule.NATIVE_UNIPC40_TIMESTEPS = tuple(
            999 if index == 33 else value for index, value in enumerate(timesteps)
        )
        with self.assertRaisesRegex(
            scorer.TemporalCounterfactualScoringError, "coordinate 33 differs"
        ):
            scorer.validate_native_coordinate_runtime(fake)

    def test_cli_requires_explicit_calibration_only_acknowledgement(self) -> None:
        args = scorer.build_parser().parse_args(
            [
                "--root-spec", "/x/spec.json",
                "--expected-root-spec-sha256", "1" * 64,
                "--bank-output-dir", "/x/bank",
                "--bank-receipt", "/x/bank/receipt.json",
                "--expected-bank-receipt-sha256", "2" * 64,
                "--group-id", "sp4-a",
                "--bernini-root", "/x/bernini",
                "--veomni-root", "/x/veomni",
                "--checkpoint", "/x/checkpoint",
                "--checkpoint-content-manifest", "/x/checkpoint.json",
                "--output-dir", "/x/output",
                "--expected-bernini-commit", "3" * 40,
                "--expected-veomni-commit", "4" * 40,
                "--method-source-revision", "5" * 40,
                "--method-source-archive-sha256", "6" * 64,
                "--expected-scorer-source-sha256", "7" * 64,
                "--expected-contract-source-sha256", "8" * 64,
            ]
        )
        self.assertFalse(args.ack_t2v_calibration_only_never_rv2v_input)

    def test_dual4_launcher_is_unique_hash_bound_and_not_self_launching(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:mi210:8", text)
        self.assertIn("--nproc_per_node=4", text)
        self.assertIn('run_group sp4-a "0,1,2,3"', text)
        self.assertIn('run_group sp4-b "4,5,6,7"', text)
        self.assertIn("TEMPORAL_CF_D541801_SOURCE_ARCHIVE", text)
        self.assertIn("TEMPORAL_CF_OVERLAY_SOURCE_ARCHIVE", text)
        self.assertIn('methods/bernini_action_editing\n', text)
        self.assertIn('normalized.startswith(selected_prefix)', text)
        self.assertIn(contract.REQUIRED_D541801_SCORER_REVISION, text)
        self.assertIn(contract.REQUIRED_D541801_SCORER_SHA256, text)
        self.assertIn(
            contract.REQUIRED_CHECKPOINT_CONTENT_MANIFEST_SHA256, text
        )
        self.assertIn(contract.REQUIRED_BERNINI_REVISION, text)
        self.assertIn(contract.REQUIRED_VEOMNI_REVISION, text)
        self.assertIn(
            "a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95",
            text,
        )
        self.assertIn("temporal_counterfactual_action_scorer_v1.py", text)
        self.assertIn("temporal_counterfactual_contract_v1.py", text)
        self.assertIn("--ack-t2v-calibration-only-never-rv2v-input", text)
        self.assertIn("transforms=7 sigmas=3 prompts=2", text)
        self.assertNotIn("global-energy-score-v4", text)
        self.assertNotIn("global-energy-group-v4", text)
        self.assertNotIn("sbatch ", text)


if __name__ == "__main__":
    unittest.main()
