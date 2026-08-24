from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_train_seer_same_state_fm_160step_single4_v1.sbatch"
)


class AUHSEERFM160Single4LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = LAUNCHER.read_text(encoding="utf-8")

    def test_bash_syntax_and_embedded_python(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", self.text, re.DOTALL)
        self.assertEqual(len(blocks), 1)
        ast.parse(blocks[0])

    def test_exact_single_node_world4_resources(self) -> None:
        for fragment in (
            "#SBATCH --partition=faculty",
            "#SBATCH --qos=bgqos",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=32",
            "#SBATCH --mem=256G",
            "#SBATCH --gres=gpu:mi210:4",
            "#SBATCH --exclusive",
            "#SBATCH --time=04:00:00",
            "--nnodes=1",
            "--nproc-per-node=4",
        ):
            self.assertIn(fragment, self.text)
        self.assertNotRegex(self.text, r"(?m)^\s*srun(?:\s|$)")

    def test_fresh_160_step_schedule_is_fixed(self) -> None:
        for fragment in (
            'OUTPUT="${EXP}/runs/seer-same-state-fm-160step-r1"',
            'test ! -e "${OUTPUT}"',
            "--max-steps 160",
            "--save-every 40",
            "--learning-rate 1e-6",
            "for step in 00000040 00000080 00000120 00000160",
        ):
            self.assertIn(fragment, self.text)
        self.assertNotIn("--resume", self.text)
        self.assertNotRegex(self.text, r"(?m)^\s*sbatch(?:\s|$)")

    def test_sealed_method_and_data_are_exactly_hash_bound(self) -> None:
        expected = (
            "ec4064949b1cc8f4dba5d3c15fab375c9392a1407f9610ad01cee24c173bf822",
            "4afea87301a05cbd44bfe906b404efc4c8ce355f4c16459fff104bd80c01379e",
            "5d2e38e790620f500a251a0febe620f98abe4e8591cb453a55438359dcb7e738",
            "42e0421404db59c198c6881ea4866344061e3b1ed03edb3c79f61efca68bb1f1",
            "6c69cb748e6776d105f7c6c15caf3bd13179f4b09d9819e744de82aa1af77847",
            "69b4587abd78459fdabf6fa41c4ed5e3f9c6f291b7516db9faf2dbd981e9f4a2",
            "5ae25aefff8f2b3583954be6010e696bae3c21df7aa9281a35ae88123c3e9aaa",
            "6bae2f9a70fb851fd1bf87f5a01a1064a8fe8a6a",
            "2d2b4591ac053ec25c6371b01a5a6746679e5793",
            "f90b3dc6fbb0ce693745223cc7a94064123dbf4d",
            "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca",
        )
        for digest in expected:
            self.assertIn(digest, self.text)
        for fragment in (
            'METHOD_ARCHIVE="${EXP}/staging/seer-event-erasure-method-r1.tar"',
            'METHOD_ROOT="${EXP}/staging/seer-event-erasure-b0-r1/methods/bernini_action_editing"',
            'TRAINER="${METHOD_ROOT}/train_seer_event_erasure_fm.py"',
            'RAW="${EXP}/data/event-erasure-raw-r2"',
            'VAE="${EXP}/data/event-erasure-vae-r1"',
            'FINAL="${EXP}/data/event-erasure-final-r1"',
            'require_sha256 "${METHOD_ARCHIVE}"',
            'require_sha256 "${TRAINER}"',
            'require_sha256 "${RAW}/full_pair.routing.jsonl"',
            'require_sha256 "${FINAL}/vae.summary.json"',
            'require_sha256 "${FINAL}/seer_dataset_manifest.json"',
            '--expected-routing-jsonl-sha256 "${ROUTING_SHA256}"',
            '--expected-bernini-commit "${BERNINI_COMMIT}"',
            '--expected-veomni-commit "${VEOMNI_COMMIT}"',
            '--expected-checkpoint-tree-sha256 "${CHECKPOINT_TREE_SHA256}"',
            '--expected-seer-owner-spec-sha256 "${OWNER_SHA256}"',
            '--expected-seer-manifest-sha256 "${MANIFEST_SHA256}"',
            '--method-source-revision "${METHOD_REVISION}"',
            '--method-source-archive-sha256 "${METHOD_ARCHIVE_SHA256}"',
        ):
            self.assertIn(fragment, self.text)

    def test_rank_caches_reuse_exact_successful_r3_wrapper(self) -> None:
        for fragment in (
            'WRAPPER="${EXP}/staging/seer-train-runner-r1/rank-wrapper.sh"',
            'require_sha256 "${WRAPPER}" "${WRAPPER_SHA256}" "successful-r3 rank wrapper"',
            'cache_base="${SLURM_TMPDIR:-/tmp}/seer-fm160-single4-${SLURM_JOB_ID}"',
            'export SEER_CACHE_ROOT="${cache_base}"',
            'export SEER_PYTHON="${PYTHON}"',
            "--no-python",
            '"${WRAPPER}"',
            'trap cleanup EXIT',
        ):
            self.assertIn(fragment, self.text)

    def test_postflight_enforces_update_without_method_claim(self) -> None:
        for fragment in (
            'receipt.get("global_step") == 160 and receipt.get("max_steps") == 160',
            'distributed.get("world_size") == 4 and distributed.get("ulysses_size") == 4',
            'evidence.get("engineering_execution_success") is True',
            'evidence.get("exact_parameter_bytes_changed") is True',
            'evidence.get("method_success_claimed") is False',
            'seer.get("training_completion_is_method_success") is False',
            'receipt.get("production_claim_forbidden") is True',
            'declared_digest == hashlib.sha256(canonical).hexdigest()',
        ):
            self.assertIn(fragment, self.text)


if __name__ == "__main__":
    unittest.main()
