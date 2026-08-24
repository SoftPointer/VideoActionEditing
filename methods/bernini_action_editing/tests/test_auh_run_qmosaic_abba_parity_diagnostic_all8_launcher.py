#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts/auh_run_qmosaic_abba_parity_diagnostic_all8.sbatch"
)


class QMosaicABBAAll8LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = LAUNCHER.read_text(encoding="utf-8")

    def test_shell_requests_one_all8_node_as_two_concurrent_world4_groups(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.text)
        self.assertIn("#SBATCH --nodes=1", self.text)
        self.assertNotRegex(self.text, r"(?m)^#SBATCH\s+--qos(?:=|\s)")
        self.assertIn('launch_cell "${phase}" dog "${dog_seed}" 0,1,2,3', self.text)
        self.assertIn('launch_cell "${phase}" human "${human_seed}" 4,5,6,7', self.text)
        self.assertIn("& dog_pid=$!", self.text)
        self.assertIn("& human_pid=$!", self.text)
        self.assertEqual(self.text.count("--nproc_per_node=4"), 2)
        self.assertNotIn("HIP_VISIBLE_DEVICES=", self.text)
        self.assertIn('ROCR_VISIBLE_DEVICES="${devices}"', self.text)

    def test_fixed_cells_and_seeds_have_no_selection_loop(self) -> None:
        self.assertIn("readonly dog_seed=2026081502", self.text)
        self.assertIn("readonly human_seed=2026081505", self.text)
        self.assertIn("QMOSAIC_ABBA_ACK_FIXED_DOG_HUMAN_SEEDS", self.text)
        self.assertNotRegex(self.text, r"for\s+(?:seed|query_seed)\s+in")

    def test_only_packet_and_diagnostic_phases_execute(self) -> None:
        self.assertIn('run_pair packet "${base_port}"', self.text)
        self.assertIn('run_pair diagnostic "$((base_port + 2))"', self.text)
        self.assertIn('"${diagnostic}" run-world4', self.text)
        self.assertIn("--diagnostic-only", self.text)
        self.assertNotIn("direction_runner=", self.text)
        self.assertNotIn("--no-lora-vjp", self.text)
        self.assertNotIn("postflight=", self.text)
        self.assertNotRegex(self.text, r'"\$\{python_bin\}"[^\n]*--(?:train|optimizer|update)')

    def test_source_archive_closes_new_runtime_launcher_and_tests(self) -> None:
        for name in (
            "run_qmosaic_abba_parity_diagnostic_sp4_v1.py",
            "auh_run_qmosaic_abba_parity_diagnostic_all8.sbatch",
            "test_run_qmosaic_abba_parity_diagnostic_sp4_v1.py",
            "test_auh_run_qmosaic_abba_parity_diagnostic_all8_launcher.py",
        ):
            self.assertIn(name, self.text)
        self.assertIn("source archive contains an unsafe or duplicate member", self.text)
        self.assertIn("member.issym()", self.text)
        self.assertIn("member.islnk()", self.text)
        self.assertIn("running launcher differs from authenticated source archive", self.text)
        self.assertIn(
            'find "${task_scratch}/source-tree" -type f -exec chmod a-w', self.text
        )

    def test_cpu_contracts_precede_any_packet_or_diagnostic_gpu_phase(self) -> None:
        cpu = self.text.index("# CPU contracts precede")
        packet = self.text.index('run_pair packet "${base_port}"')
        diagnostic = self.text.index('run_pair diagnostic "$((base_port + 2))"')
        self.assertLess(cpu, packet)
        self.assertLess(packet, diagnostic)

    def test_editor_authority_keypair_is_checked_before_gpu_work(self) -> None:
        mode = self.text.index("editor private key mode differs")
        pair = self.text.index("editor authority private/public keys do not match")
        packet = self.text.index('run_pair packet "${base_port}"')
        self.assertLess(mode, pair)
        self.assertLess(pair, packet)
        self.assertIn("Ed25519PrivateKey", self.text)
        self.assertIn("Ed25519PublicKey", self.text)
        self.assertIn("private.public_key().public_bytes", self.text)

    def test_manifest_is_last_fail_closed_publication_boundary(self) -> None:
        preclosure = self.text.index("pre-manifest diagnostic closure differs")
        terminal = self.text.index("source_archive_terminal")
        aggregate = self.text.rindex('"${diagnostic}" aggregate-all8')
        chmod_manifest = self.text.index(
            'chmod a-w -- "${output_root}/all8.manifest.json"'
        )
        publish = self.text.rindex("published=true")
        self.assertLess(preclosure, terminal)
        self.assertLess(terminal, aggregate)
        self.assertLess(aggregate, chmod_manifest)
        self.assertLess(chmod_manifest, publish)
        after_publish = self.text[publish + len("published=true") :].strip()
        self.assertEqual(after_publish, "")
        cleanup = self.text[
            self.text.index("cleanup() {") : self.text.index("trap cleanup EXIT")
        ]
        self.assertIn('if [[ "${published}" != true', cleanup)
        self.assertIn('rm -rf -- "${output_root}"', cleanup)
        self.assertNotIn("all8.manifest.json", cleanup)

    def test_published_tree_contains_json_receipts_only(self) -> None:
        self.assertIn('expected_files = set()', self.text)
        self.assertIn('expected_files.add(f"{cell}/world4.receipt.json")', self.text)
        self.assertIn('expected_files.add(f"{cell}/rank-{rank}.receipt.json")', self.text)
        self.assertIn("Scratch packets/logs never publish", self.text)
        self.assertNotRegex(
            self.text,
            re.compile(r'expected_files\.add\([^\n]*\.(?:mp4|pt|safetensors)'),
        )


if __name__ == "__main__":
    unittest.main()
