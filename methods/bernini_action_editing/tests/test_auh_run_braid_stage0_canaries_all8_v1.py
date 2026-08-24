#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts/auh_run_braid_stage0_canaries_all8_v1.sbatch"


class BraidStage0All8LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = LAUNCHER.read_text(encoding="utf-8")

    def test_one_all8_node_is_two_independent_concurrent_world4_groups(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        self.assertIn("#SBATCH --nodes=1", self.text)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.text)
        self.assertNotRegex(self.text, r"(?m)^#SBATCH\s+--qos(?:=|\s)")
        self.assertIn('launch_cell dog "${dog_seed}" 0,1,2,3', self.text)
        self.assertIn('launch_cell human "${human_seed}" 4,5,6,7', self.text)
        self.assertIn("& dog_pid=$!", self.text)
        self.assertIn("& human_pid=$!", self.text)
        self.assertEqual(self.text.count("--nproc_per_node=4"), 1)
        self.assertNotIn("--nproc_per_node=8", self.text)
        self.assertIn('ROCR_VISIBLE_DEVICES="${devices}"', self.text)
        self.assertNotIn("HIP_VISIBLE_DEVICES=", self.text)
        self.assertIn("env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL", self.text)

    def test_fixed_six_arms_and_fixed_seeds_have_no_selection(self) -> None:
        self.assertIn("readonly dog_seed=2026081502", self.text)
        self.assertIn("readonly human_seed=2026081505", self.text)
        expected = (
            "parity-reset-off-reference-4f-a",
            "parity-reset-off-reference-4f-b",
            "parity-reset-off-shared-negative-3f",
            "reset-on-reference-4f",
            "capacity-source-bias-off-reference-4f",
            "capacity-source-bias-on-reference-4f",
        )
        arm_block = self.text[
            self.text.index("readonly full_arms=(") : self.text.index(
                "readonly partial_arms=("
            )
        ]
        for arm in expected:
            self.assertEqual(arm_block.count(arm), 1)
        self.assertNotRegex(self.text, r"for\s+(?:seed|query_seed)\s+in")
        self.assertIn("BRAID_STAGE0_ACK_FIXED_DOG_HUMAN_ARMS", self.text)

    def test_each_arm_is_a_new_torchrun_and_not_one_world8_process(self) -> None:
        self.assertIn('for index in "${!arms[@]}"; do', self.text)
        self.assertIn('run_pair "${arms[$index]}"', self.text)
        self.assertIn('"${runner}" run-world4', self.text)
        self.assertIn("fresh_process=true", self.text)
        self.assertIn("distinct torchrun invocation", self.text)
        self.assertNotIn("torch.distributed.run --nproc_per_node=8", self.text)
        self.assertNotIn("--world-size=8", self.text)

    def test_world4_implementation_closure_is_an_explicit_pre_gpu_blocker(self) -> None:
        required = self.text.index('"methods/bernini_action_editing/run_braid_stage0_world4_v1.py"')
        extracted = self.text.index('runner="${method_root}/run_braid_stage0_world4_v1.py"')
        cpu_tests = self.text.index("# CPU contracts and implemented-arm closure")
        closure = self.text.index('implemented = set(module.stage0.IMPLEMENTED_WORLD4_ARM_IDS)')
        output = self.text.index('mkdir -- "${output_root}"')
        gpu = self.text.index('base_port=$((25000 +')
        self.assertLess(required, extracted)
        self.assertLess(extracted, cpu_tests)
        self.assertLess(cpu_tests, closure)
        self.assertLess(closure, output)
        self.assertLess(output, gpu)
        self.assertIn("source archive lacks BRAID Stage-0 closure", self.text)
        self.assertIn("test_run_braid_stage0_world4_v1.py", self.text)
        self.assertIn(
            "full all8 launcher is intentionally blocked before GPU model load",
            self.text,
        )

    def test_authenticated_runtime_plan_and_signed_editor_packets_are_bound(self) -> None:
        for value in (
            "braid_dual_native_apg_runtime_v1.py",
            "braid_stage0_all8_orchestrator_v1.py",
            "bernini_braid_exact81_arms_v1.json",
            "self_imagined_motion_cotangent_core2_v1.json",
        ):
            self.assertIn(value, self.text)
        self.assertIn('runtime_sha256="$(hash_file "${runtime}")"', self.text)
        self.assertIn("--expected-dual-runtime-source-sha256", self.text)
        self.assertIn("--expected-plan-file-sha256", self.text)
        self.assertIn("--expected-editor-receipt-sha256", self.text)
        self.assertIn("--expected-editor-public-key-sha256", self.text)
        self.assertIn("--dog-editor-receipt-file-sha256", self.text)
        self.assertIn("--human-editor-receipt-file-sha256", self.text)
        self.assertIn("pinned_editor_public_key_sha256=b1357fcf", self.text)
        self.assertIn("--expected-execution-public-key-sha256", self.text)
        self.assertIn("--expected-checkpoint-tree-sha256", self.text)
        self.assertIn("--expected-owner-master-receipt-sha256", self.text)
        self.assertIn("--expected-owner-cell-receipt-sha256", self.text)
        self.assertIn("--expected-owner-audit-sidecar-sha256", self.text)
        self.assertIn("--expected-owner-audit-public-key-sha256", self.text)
        self.assertIn("running launcher differs from authenticated source archive", self.text)

    def test_ephemeral_execution_key_is_private_and_publication_is_plan_bound(self) -> None:
        generated = self.text.index("Ed25519PrivateKey.generate()")
        plan = self.text.index('"${orchestrator}" write-plan')
        torchrun = self.text.index('"${python_bin}" -B -m torch.distributed.run')
        self.assertLess(generated, plan)
        self.assertLess(plan, torchrun)
        self.assertIn("serialization.PrivateFormat.PKCS8", self.text)
        self.assertIn("(private_path, private_bytes, 0o600)", self.text)
        self.assertIn("--execution-private-key", self.text)
        self.assertIn("--execution-public-key", self.text)
        self.assertIn("--execution-public-key-file-sha256", self.text)
        self.assertIn(
            'execution_public_key="${output_root}/stage0-execution-ed25519-public.pem"',
            self.text,
        )
        self.assertNotIn("execution-ed25519-private.pem\" \"${output_root}", self.text)

    def test_forward_only_ack_and_receipts_forbid_training_or_decode_authority(self) -> None:
        self.assertIn(
            "BRAID_STAGE0_ACK_NO_DECODE_BACKWARD_OPTIMIZER_UPDATE", self.text
        )
        self.assertIn(
            "--ack-forward-only-no-decode-backward-optimizer-update", self.text
        )
        runner_call = self.text[
            self.text.index('"${runner}" run-world4') : self.text.index("run_pair() {")
        ]
        self.assertNotRegex(runner_call, r"--(?:train|optimizer|update|decode)(?:\s|=)")
        self.assertNotRegex(
            self.text, r'"\$\{python_bin\}"[^\n]*(?:\.backward\(|optimizer\.step)'
        )
        self.assertIn("stage0.plan.json", self.text)
        self.assertIn("validate-world4", self.text)

    def test_all_receipts_reopen_before_terminal_manifest_publication(self) -> None:
        validate = self.text.index('"${orchestrator}" validate-world4')
        terminal = self.text.index("source_archive_terminal")
        aggregate = self.text.index('"${orchestrator}" aggregate-all8')
        chmod = self.text.index('chmod a-w -- "${publication}"')
        publish = self.text.rindex("published=true")
        self.assertLess(validate, terminal)
        self.assertLess(terminal, aggregate)
        self.assertLess(aggregate, chmod)
        self.assertLess(chmod, publish)
        self.assertEqual(
            self.text[publish + len("published=true") :].strip(), ""
        )
        cleanup = self.text[
            self.text.index("cleanup() {") : self.text.index("trap cleanup EXIT")
        ]
        self.assertIn('if [[ "${published}" != true', cleanup)
        self.assertIn('rm -rf -- "${output_root}"', cleanup)

    def test_archive_extraction_and_source_identity_are_fail_closed(self) -> None:
        for value in (
            "member.issym()",
            "member.islnk()",
            "member.isdev()",
            "source archive contains an unsafe or duplicate member",
            "validate_source_trees",
            "authenticated source revisions differ",
        ):
            self.assertIn(value, self.text)
        self.assertIn(
            'find "${task_scratch}/source-tree" -type f -exec chmod a-w',
            self.text,
        )
        self.assertNotRegex(
            self.text,
            re.compile(r"(?m)^\s*git\s+(?:add|commit|push|reset|clean)\b"),
        )


if __name__ == "__main__":
    unittest.main()
