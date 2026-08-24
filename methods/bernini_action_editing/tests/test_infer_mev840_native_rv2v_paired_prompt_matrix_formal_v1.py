from __future__ import annotations

from pathlib import Path
import hashlib
import os
import unittest
from unittest import mock

from methods.bernini_action_editing import (
    infer_mev840_native_rv2v_paired_prompt_matrix_formal_v1 as runner,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "assets" / "mev840_native_rv2v_same_process_formal_v1.json"


def _args(seed: int = 2027, *extra: str):
    digest = hashlib.sha256(AUTHORITY.read_bytes()).hexdigest()
    parser = runner.build_parser()
    return parser.parse_args(
        [
            "--bernini-root",
            "/tmp/bernini",
            "--veomni-root",
            "/tmp/veomni",
            "--checkpoint",
            "/tmp/checkpoint",
            "--checkpoint-content-manifest",
            "/tmp/checkpoint.manifest",
            "--source-video",
            "/tmp/source.mp4",
            "--expected-source-sha256",
            "0" * 64,
            "--prompt-matrix-authority",
            str(AUTHORITY),
            "--expected-prompt-matrix-authority-sha256",
            digest,
            "--output-dir",
            f"/tmp/mev840-formal-{seed}",
            "--num-inference-steps",
            "40",
            "--seed",
            str(seed),
            "--method-source-revision",
            "0" * 40,
            "--method-source-archive-sha256",
            "1" * 64,
            *extra,
        ]
    )


class FormalPairedPromptMatrixRunnerTests(unittest.TestCase):
    def test_authority_and_cli_are_formal_only(self) -> None:
        for seed in (2027, 2028):
            bundle = runner.validate_cli(_args(seed))
            self.assertEqual(bundle["authority"]["execution_mode"]["seeds"], [2027, 2028])
        wrong_steps = _args(2027)
        wrong_steps.num_inference_steps = 2
        with self.assertRaisesRegex(runner.NativeIdentityCanaryError, "exactly 40"):
            runner.validate_cli(wrong_steps)
        wrong_seed = _args(2027)
        wrong_seed.seed = 2029
        with self.assertRaisesRegex(runner.NativeIdentityCanaryError, "2027 or 2028"):
            runner.validate_cli(wrong_seed)
        skip = _args(2027, "--skip-video-decode")
        with self.assertRaisesRegex(runner.NativeIdentityCanaryError, "requires P0a/P1/P2"):
            runner.validate_cli(skip)

    def test_formal_slurm_context_is_seed_bound(self) -> None:
        bundle = runner.validate_cli(_args(2027))
        environment = {
            "SLURM_JOB_ID": "143808",
            "SLURM_STEP_ID": "42",
            "WORLD_SIZE": "4",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            with mock.patch.object(runner.socket, "gethostname", return_value="auh7-1b-gpu-292"):
                row = runner._formal_slurm_context(_args(2027), bundle)
        self.assertEqual(
            row,
            {
                "job_id": "143808",
                "step_id": "42",
                "job_step_id": "143808.42",
                "node": "auh7-1b-gpu-292",
                "world_size": 4,
            },
        )
        environment["SLURM_JOB_ID"] = "147873"
        with mock.patch.dict(os.environ, environment, clear=True):
            with mock.patch.object(runner.socket, "gethostname", return_value="auh7-1b-gpu-292"):
                with self.assertRaisesRegex(runner.NativeIdentityCanaryError, "job/node"):
                    runner._formal_slurm_context(_args(2027), bundle)

    def test_exact13_and_prompt_design_are_sealed(self) -> None:
        authority = runner.load_prompt_matrix_authority(
            AUTHORITY,
            expected_sha256=hashlib.sha256(AUTHORITY.read_bytes()).hexdigest(),
        )["authority"]
        self.assertEqual(authority["execution_mode"]["decode_cells"], ["p0a", "p1", "p2"])
        self.assertEqual(authority["execution_mode"]["latent_only_replay_cells"], ["p0b"])
        self.assertEqual(authority["execution_mode"]["exact_regular_file_count_per_seed"], 13)
        self.assertEqual(
            authority["runtime_authority"]["formal_slurm_by_seed"],
            {
                "2027": {"job_id": "143808", "node": "auh7-1b-gpu-292", "world_size": 4},
                "2028": {"job_id": "147873", "node": "auh7-1b-gpu-284", "world_size": 4},
            },
        )

    def test_main_retires_aliases_and_tensors_before_and_after_decode(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        main = source[source.index("def main(") :]
        alias_retirement = main.index("del model, scheduler, rope_pristine")
        decode_load = main.index("decode_vae = AutoencoderKLWan.from_pretrained")
        final_retirement = main.index("del generated", decode_load)
        final_gate = main.index("formal_terminal_memory_after_decode_and_tensor_retirement")
        self.assertLess(alias_retirement, decode_load)
        self.assertLess(decode_load, final_retirement)
        self.assertLess(final_retirement, final_gate)
        self.assertIn("source_conditions_and_noise_captures_retired_before_rank_zero_decode_vae_load", main)
        self.assertIn("rank_zero_decode_vae_cpu_materialization_count_after_decode", main)
        paired_output = source[
            source.index("def _save_paired_outputs") : source.index("def _build_paired_receipt")
        ]
        self.assertNotIn('vae.to("cpu")', paired_output)
        self.assertIn('if cell == "p0b":', paired_output)
        self.assertNotIn("target_action_oracle", source)


if __name__ == "__main__":
    unittest.main()
