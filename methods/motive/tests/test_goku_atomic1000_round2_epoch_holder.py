from __future__ import annotations

import re
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
HOLDER = ROOT / "tmp" / "goku_atomic1000_round2_epoch_holder.sbatch"


def embedded(text: str, marker: str) -> str:
    opening = f"<<'{marker}'"
    start = text.index("\n", text.index(opening)) + 1
    end = text.index(f"\n{marker}\n", start)
    return text[start:end] + "\n"


class Round2EpochHolderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = HOLDER.read_text(encoding="utf-8")

    def test_shell_and_all_embedded_python_compile(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(HOLDER)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        markers = re.findall(r"<<'([A-Z][A-Z0-9_]*)'", self.text)
        self.assertEqual(
            markers,
            ["PY_EXIT", "PY_EPOCH_PREFLIGHT", "PY_READY", "PY_RELEASE_FINAL"],
        )
        for marker in markers:
            compile(embedded(self.text, marker), marker, "exec")

    def test_exact_four_node_three_day_geometry(self) -> None:
        for contract in (
            "#SBATCH --nodes=4",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=128",
            "#SBATCH --gres=gpu:mi210:8",
            "#SBATCH --mem=2T",
            "#SBATCH --time=3-00:00:00",
            '"${SLURM_JOB_NUM_NODES:-}" == 4',
            "'NumNodes=4' 'gres/gpu:mi210=32'",
        ):
            self.assertIn(contract, self.text)

    def test_epoch_inputs_and_exact_targets_are_explicit(self) -> None:
        for binding in (
            "epoch_index=${EPOCH_INDEX:-}",
            "epoch_target=${EPOCH_TARGET:-}",
            "epoch_selected_sha=${EPOCH_SELECTED_SHA256:-}",
            "epoch_done_sha=${EPOCH_DONE_SHA256:-}",
            '"${epoch_index}" =~ ^[2-8]$',
            "epochs 2--7 have the immutable target 128",
            "epoch 8 has the immutable final target 104",
            "expected_rows=2000",
        ):
            self.assertIn(binding, self.text)
        preflight = embedded(self.text, "PY_EPOCH_PREFLIGHT")
        for closure in (
            'done["output_sha256"] != selected_sha',
            'binding.get("epoch_index") != epoch_index',
            'binding.get("parent_selected_sha256")',
            'source_done.get("artifacts", {}).get("selected.jsonl")',
            'epochs_done.get("artifacts", {}).get("epoch_done_sha256", {}).get(',
            '!= selected_raw',
        ):
            self.assertIn(closure, preflight)

    def test_frozen_software_and_pipeline_bindings(self) -> None:
        for binding in (
            "goku-atomic-round2-source-v3-20260813T070242Z",
            "681251a969eddca71eaa25402e9fce9f3ee19a4484f3a8dea83162cf7bcc4e06",
            "launch_goku_atomic1000_round2_4node_pipeline.sh",
            "650a4e6d155de2bf0f3da8dbaa92f81afbab53ce83a23b8fba9faf672869dc6d",
            "launch_fullmotion_v16_wan_stream_round2_4node.sh",
            "1857a1e0b29a29889141d55195e9cde842e0e385b9e923340de4b4317bf3ddcb",
            'MOTIVE_ATOMIC_LAUNCHER_SHA256="${pipeline_sha}"',
            'MOTIVE_ATOMIC_WAN_LAUNCHER_SHA256="${wan_launcher_sha}"',
        ):
            self.assertIn(binding, self.text)

    def test_integrated_ready_launch_release_and_exit_contract(self) -> None:
        for contract in (
            "motive-goku-atomic1000-round2-allocation-holder-v1",
            "motive-goku-atomic1000-round2-allocation-release-v1",
            "motive-goku-atomic1000-round2-epoch-holder-exit-v1",
            'MOTIVE_ATOMIC_RESUME=0',
            'MOTIVE_ATOMIC_SMOKE_BATCH_ROWS=64',
            'MOTIVE_ATOMIC_FULL_BATCH_ROWS=128',
            'MOTIVE_ATOMIC_JOB_ID="${SLURM_JOB_ID}"',
            'MOTIVE_ATOMIC_JOB_NAME="${SLURM_JOB_NAME}"',
            'MOTIVE_ATOMIC_MINIMUM_FINAL_SUCCESS="${epoch_target}"',
            'release_holder_${SLURM_JOB_ID}.json',
            'flock -n "${epoch_lock_fd}"',
            'controller exited without a release receipt',
            'write_exit_receipt "${holder_status}" 0',
        ):
            self.assertIn(contract, self.text)
        release = embedded(self.text, "PY_RELEASE_FINAL")
        self.assertIn("release_value != expected_release", release)
        self.assertIn("len(lines) != target", release)
        self.assertIn('summary["manifest_sha256"] != sha(manifest_raw)', release)

    def test_holder_cannot_submit_cancel_delete_or_resume(self) -> None:
        self.assertNotRegex(
            self.text,
            r"(?m)^\s*(?:scancel|sbatch|salloc|rm|unlink)(?:\s|$)",
        )
        self.assertNotIn("MOTIVE_ATOMIC_RESUME=1", self.text)
        self.assertIn('[[ ! -e "${run_root}" && ! -L "${run_root}" ]]', self.text)
        self.assertIn("os.O_EXCL", embedded(self.text, "PY_READY"))
        self.assertIn("os.O_EXCL", embedded(self.text, "PY_EXIT"))


if __name__ == "__main__":
    unittest.main()
