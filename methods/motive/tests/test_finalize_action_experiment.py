from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from motive.finalize_action_experiment import finalize, parse_training_log


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _complete_run(root: Path) -> Path:
    run_root = root / "run"
    _json(run_root / "prep" / "summary.json", {"selected": 4})
    for seed in (2026, 2027, 2028):
        directory = run_root / "prep" / f"repr_seed_{seed}"
        _json(
            directory / "metrics.json",
            {
                "initial_loss": 2.0,
                "final_loss": 1.0,
                "loss_history": [2.0, 1.0],
                "train": {},
                "validation": {},
                "test": {},
                "shortcut_baselines": {},
            },
        )
        (directory / "prompt_action_encoder.pt").write_bytes(b"checkpoint")
        _json(directory / "prompt_action_encoder.pt.json", {"schema": "test"})
    for arm in (
        "e1_plain_lora",
        "e2_fixed_random",
        "e2_random_router",
        "e3_motive_frozen",
    ):
        output = run_root / "lucy" / arm / "seed_2026"
        _json(
            output / "run_config.json",
            {"num_processes": 2, "effective_global_batch_size": 8},
        )
        (output / "checkpoint_step_000010.pt").write_bytes(b"checkpoint")
        for rank in range(2):
            (
                output / f"checkpoint_step_000010.rng_rank_{rank:03d}.pt"
            ).write_bytes(b"rng")
        output.with_suffix(".log").write_text(
            "step=5 loss=2.0 diffusion=1.0 grad=1.0 samples_per_s=0.1\n"
            "step=10 loss=1.0 diffusion=0.5 grad=0.5 samples_per_s=0.2\n",
            encoding="utf-8",
        )
        _json(
            run_root / "status" / f"{arm}_seed_2026.json",
            {"state": "succeeded", "exit_code": 0},
        )
        _json(
            output / "training_validation.json",
            {"complete": True},
        )
    return run_root


class FinalizeActionExperimentTests(unittest.TestCase):
    def test_parse_training_log_deduplicates_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "train.log"
            log.write_text(
                "step=5 loss=2.0 diffusion=1.5 grad=0.9 samples_per_s=0.1\n"
                "noise\n"
                "step=5 loss=1.9 diffusion=1.4 grad=0.8 samples_per_s=0.2\n"
                "step=10 loss=1.0 diffusion=0.7 grad=0.4 samples_per_s=0.3\n",
                encoding="utf-8",
            )
            rows = parse_training_log(log)
            self.assertEqual([row["step"] for row in rows], [5, 10])
            self.assertAlmostEqual(float(rows[0]["loss"]), 1.9)

    def test_finalize_complete_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = _complete_run(Path(temporary))
            payload = finalize(
                run_root,
                seed=2026,
                expected_step=10,
                expected_processes=2,
                expected_global_batch=8,
                allow_incomplete=False,
                overwrite=False,
            )
            self.assertTrue(payload["complete"])
            self.assertTrue(
                (
                    run_root
                    / "analysis"
                    / "e3_motive_frozen_losses.jsonl"
                ).is_file()
            )

    def test_finalize_incomplete_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = _complete_run(Path(temporary))
            (
                run_root
                / "lucy"
                / "e2_random_router"
                / "seed_2026"
                / "checkpoint_step_000010.pt"
            ).unlink()
            payload = finalize(
                run_root,
                seed=2026,
                expected_step=10,
                expected_processes=2,
                expected_global_batch=8,
                allow_incomplete=True,
                overwrite=False,
            )
            self.assertFalse(payload["complete"])
            self.assertTrue(
                (run_root / "analysis" / "final_summary.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
