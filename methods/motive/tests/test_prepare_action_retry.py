from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from motive.prepare_action_retry import prepare_retry


class PrepareActionRetryTests(unittest.TestCase):
    def test_prepares_runtime_links_and_reused_e1_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            (snapshot / "lucy").mkdir(parents=True)
            (snapshot / "methods").mkdir()
            digest = "a" * 64
            (snapshot / "SOURCE_FILES.jsonl").write_text("{}\n", encoding="utf-8")
            (snapshot / "SOURCE_PROVENANCE.json").write_text(
                json.dumps({"source_tree_sha256": digest}),
                encoding="utf-8",
            )
            live = root / "live"
            (live / "data").mkdir(parents=True)
            (live / "checkpoints").mkdir()
            base = root / "base"
            prep = base / "prep"
            (prep / "repr_seed_2026").mkdir(parents=True)
            (prep / "lucy_train_manifest.jsonl").write_text(
                "{}\n",
                encoding="utf-8",
            )
            (prep / "repr_seed_2026" / "prompt_action_encoder.pt").write_bytes(
                b"action"
            )
            output = base / "lucy" / "e1_plain_lora" / "seed_2026"
            output.mkdir(parents=True)
            (output / "run_config.json").write_text("{}\n", encoding="utf-8")
            (output / "checkpoint_step_000100.pt").write_bytes(b"checkpoint")
            for rank in range(8):
                (
                    output
                    / f"checkpoint_step_000100.rng_rank_{rank:03d}.pt"
                ).write_bytes(b"rng")
            output.with_suffix(".log").write_text(
                "step=100 loss=0.1\n",
                encoding="utf-8",
            )
            status = base / "status" / "e1_plain_lora_seed_2026.json"
            status.parent.mkdir(parents=True)
            status.write_text('{"state":"succeeded"}\n', encoding="utf-8")

            run_root = root / "retry"
            runtime = root / "runtime"
            contract = prepare_retry(
                run_id="retry-test",
                run_root=run_root,
                runtime_repo=runtime,
                source_snapshot=snapshot,
                source_tree_sha256=digest,
                live_repo=live,
                base_run=base,
            )

            self.assertEqual(contract["schema"], "motive-action-retry-contract-v1")
            self.assertTrue((runtime / "lucy").is_symlink())
            self.assertTrue((run_root / "prep").is_symlink())
            self.assertTrue(
                (
                    run_root
                    / "lucy"
                    / "e1_plain_lora"
                    / "seed_2026"
                    / "checkpoint_step_000100.pt"
                ).is_symlink()
            )
            self.assertTrue(
                (run_root / "contracts" / "retry_contract.json").is_file()
            )

            with self.assertRaisesRegex(FileExistsError, "retry run root"):
                prepare_retry(
                    run_id="retry-test",
                    run_root=run_root,
                    runtime_repo=root / "runtime-2",
                    source_snapshot=snapshot,
                    source_tree_sha256=digest,
                    live_repo=live,
                    base_run=base,
                )


if __name__ == "__main__":
    unittest.main()
