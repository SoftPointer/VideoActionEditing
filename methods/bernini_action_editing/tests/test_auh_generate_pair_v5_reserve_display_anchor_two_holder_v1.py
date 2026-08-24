from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = METHOD_ROOT / "scripts/auh_generate_pair_v5_reserve_display_anchor_two_holder_v1.sh"
OVERLAY = METHOD_ROOT / "infer_pair_v5_reserve_display_anchor_overlay_v1.py"
STAGE_B_CONTROLLER = METHOD_ROOT / "scripts/auh_infer_source_noised_carrier_stage_b_two_holder_v5.sh"
STAGE_B_RUNTIME = METHOD_ROOT / "infer_source_noised_carrier_stage_b_v1.py"
STAGE_B_RELEASE = METHOD_ROOT / "releases/source_noised_carrier_stage_b_inference_r3"


class PairV5ReserveDisplayAnchorTwoHolderControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = CONTROLLER.read_text(encoding="utf-8")

    def test_bash_syntax_and_usage_fail_closed(self) -> None:
        subprocess.run(["bash", "-n", str(CONTROLLER)], check=True)
        env = dict(os.environ)
        env.update(
            BERNINI_PAIR_V5_ANCHOR_WORK_JOB0="135412",
            BERNINI_PAIR_V5_ANCHOR_WORK_JOB1="135407",
        )
        result = subprocess.run(
            [str(CONTROLLER)], env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)

    def test_exact_selectable_holder_pair_and_retained_third(self) -> None:
        for first, second in (
            ("135407", "135411"), ("135411", "135407"),
            ("135407", "135412"), ("135412", "135407"),
            ("135411", "135412"), ("135412", "135411"),
        ):
            env = dict(os.environ)
            env.update(
                BERNINI_PAIR_V5_ANCHOR_WORK_JOB0=first,
                BERNINI_PAIR_V5_ANCHOR_WORK_JOB1=second,
            )
            result = subprocess.run(
                [str(CONTROLLER)], env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("usage:", result.stderr)
        self.assertIn("135407) printf '%s\\n' auh7-1b-gpu-260", self.source)
        self.assertIn("135411) printf '%s\\n' auh7-1b-gpu-214", self.source)
        self.assertIn("135412) printf '%s\\n' auh7-1b-gpu-293", self.source)
        self.assertIn('assert_parent_running "${retained_job}"', self.source)
        self.assertNotIn('launch_node "${retained_job}"', self.source)
        self.assertNotIn('assert_remote_idle_once "${retained_job}"', self.source)

    def test_exact_archive_authoring_generator_and_overlay_pins(self) -> None:
        pins = (
            "f9360fcef6bdcb9e37345515fb85d18e4c444fd2b100de35aeb0c1a55a98ac55",
            "17cc2c73d774e14cdd10bd2ceea4afbaf4b0be26",
            "a60c37591c40206c6130185f1a2d2a7a8e473f5af4425205e268ae4a8b58f334",
            "e19e353d7e83ce7a7fe37bc958dd67e58ae6ae772fafaba8cc40bfb2097e3db6",
            "a4baa1aea27f6497ca2dd615cc09b2b90eee37173f506e60ae7d630c41886be6",
            "204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c",
            "b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1",
        )
        for pin in pins:
            self.assertIn(pin, self.source)
        overlay_sha = hashlib.sha256(OVERLAY.read_bytes()).hexdigest()
        self.assertEqual(
            overlay_sha,
            "e4d2f998ce61c007ca61c7d66278cbcbc0576bf152010d35d543162cf264064d",
        )
        self.assertIn(overlay_sha, self.source)
        self.assertIn('git get-tar-commit-id <"${source_archive}"', self.source)

    def test_action_only_is_default_and_family4_is_explicit_serial_extension(self) -> None:
        self.assertIn('readonly profile="${BERNINI_PAIR_V5_ANCHOR_PROFILE:-action-only}"', self.source)
        action = self.source.index('run_branch action "$(( first_port + 0 ))"')
        family_gate = self.source.index('if [[ "${profile}" == family4 ]]')
        noop = self.source.index('run_branch noop "$(( first_port + 1 ))"')
        incomplete = self.source.index('run_branch incomplete "$(( first_port + 2 ))"')
        reverse = self.source.index('run_branch reverse "$(( first_port + 3 ))"')
        self.assertLess(action, family_gate)
        self.assertEqual([noop, incomplete, reverse], sorted((noop, incomplete, reverse)))
        self.assertLess(family_gate, noop)
        self.assertIn('--profile "${profile}"', self.source)
        self.assertIn('"action_canary_first":True', self.source)

    def test_world4_two_by_two_and_strict_memory_contract(self) -> None:
        self.assertIn("--nnodes=2 --nproc_per_node=2", self.source)
        self.assertIn('--node_rank="${rank}"', self.source)
        self.assertIn("--ntasks=1 --cpus-per-task=16 --mem=56G --gres=gpu:mi210:2", self.source)
        self.assertIn("readonly memory_peak_limit_bytes=55834574848", self.source)
        self.assertIn("sampled<int(limit)", self.source)
        self.assertIn("sacct<int(limit)", self.source)
        self.assertIn("torch.cuda.device_count() == 2", self.source)
        self.assertIn('assert_idle_twice "pre-${branch}"', self.source)
        self.assertIn("assert_idle_twice final", self.source)

    def test_only_identity_bound_direct_child_srun_can_be_signaled(self) -> None:
        self.assertIn('ppid="$(proc_field "${pid}" 4)"', self.source)
        self.assertIn('[[ "${ppid}" == "$$"', self.source)
        self.assertIn('"$(basename -- "${exe}")" == srun', self.source)
        self.assertIn("pid_cmd_sha", self.source)
        forbidden = ("s" + "cancel", "scontrol " + "release", "scontrol " + "requeue", "p" + "kill", "kill" + "all")
        for item in forbidden:
            self.assertNotIn(item, self.source)
        direct = [line.strip() for line in self.source.splitlines() if "kill -" in line and "kill -0" not in line]
        self.assertEqual(
            direct,
            ['signal_owned_pid() { if pid_identity_matches "$1"; then kill -"$2" "$1" 2>/dev/null || true; elif [[ -e "/proc/$1" ]]; then echo "REFUSE_SIGNAL pid=$1" >&2; fi; }'],
        )

    def test_display_only_path_never_invokes_stage_b_or_old_bank_audit(self) -> None:
        self.assertNotIn("infer_source_noised_carrier_stage_b", self.source)
        self.assertNotIn("verify-pair", self.source)
        self.assertNotIn("--audit-bank", self.source)
        self.assertNotIn("--anchor-action-video", self.source)
        self.assertIn('"display_only":True', self.source)
        self.assertIn('"stage_b_condition":False', self.source)
        self.assertIn('"old40_bank_audit_claimed":False', self.source)
        self.assertIn('"action_success_claimed":False', self.source)
        self.assertIn('"scientific_claim_authorized":False', self.source)

    def test_real_run_branch_reaches_idle_gate_under_nounset(self) -> None:
        start = self.source.index("run_branch() {")
        end = self.source.index('\nrun_branch action "$(( first_port + 0 ))"', start)
        real_function = self.source[start:end]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "outputs").mkdir()
            (root / "topology").mkdir()
            harness = "\n".join(
                (
                    "set -u",
                    f"run_root={root}",
                    "fail() { printf 'FAIL:%s\\n' \"$*\" >&2; exit 2; }",
                    "assert_idle_twice() { printf 'REACHED_ANCHOR_GATE:%s\\n' \"$1\"; exit 73; }",
                    real_function,
                    "run_branch action 29961",
                )
            )
            result = subprocess.run(
                ["bash", "-c", harness], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertEqual(result.stdout.strip(), "REACHED_ANCHOR_GATE:pre-action")
        self.assertNotIn("unbound variable", result.stderr)

    def test_no_local_builtin_self_reference(self) -> None:
        unsafe = []
        assignment = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=")
        for line_number, line in enumerate(self.source.splitlines(), 1):
            command = line.split(";", 1)[0]
            if not command.lstrip().startswith("local "):
                continue
            declarations = list(assignment.finditer(command))
            for item in declarations:
                if f"${{{item.group(1)}}}" in command[item.end():]:
                    unsafe.append((line_number, item.group(1), command.strip()))
        self.assertEqual(unsafe, [])

    def test_frozen_stage_b_bytes_remain_unchanged(self) -> None:
        self.assertEqual(
            hashlib.sha256(STAGE_B_CONTROLLER.read_bytes()).hexdigest(),
            "5aa68c97c52cba9f2a2171b9ff98f6fc865c67ab641c11a07799369715e71f02",
        )
        self.assertEqual(
            hashlib.sha256(STAGE_B_RUNTIME.read_bytes()).hexdigest(),
            "7e6cdba95c62d2ae9bbe81cfa123ac208c2ca890f134cfe6d0538cefea68db50",
        )
        self.assertEqual(
            hashlib.sha256((STAGE_B_RELEASE / "source.tar").read_bytes()).hexdigest(),
            "e3880934c3e6cfcb0dfe56aa34a03f3ffbb2cb192a262fdb8ae1734a02f183ca",
        )
        self.assertEqual(
            hashlib.sha256((STAGE_B_RELEASE / "source.manifest.json").read_bytes()).hexdigest(),
            "6849ed11ad214e4c49f72731e4beb88948f2abf26e79f0ff5cf8c4e2814e62a3",
        )


if __name__ == "__main__":
    unittest.main()
