from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = (
    METHOD_ROOT
    / "scripts"
    / "auh_eval_seer_full160_core4_two_holder_v1.sh"
)


class SeerFull160TwoHolderControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = CONTROLLER.read_text(encoding="utf-8")

    def test_shell_is_valid_and_no_argument_path_is_read_only(self) -> None:
        subprocess.run(["bash", "-n", str(CONTROLLER)], check=True)
        result = subprocess.run(
            ["bash", str(CONTROLLER)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)

    def test_holder_allowlist_and_exact_nodes_are_closed(self) -> None:
        for job, node in (
            ("135407", "auh7-1b-gpu-260"),
            ("135411", "auh7-1b-gpu-214"),
            ("135412", "auh7-1b-gpu-293"),
        ):
            self.assertIn(f"{job}) printf '%s\\n' {node}", self.text)
        allowlist = re.search(
            r"is_allowed_holder\(\).*?case \"\$1\" in(.*?)esac",
            self.text,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(allowlist)
        self.assertIn("135407|135411|135412", allowlist.group(1))
        self.assertIn("assert_all_parents_running", self.text)
        self.assertGreaterEqual(self.text.count("assert_all_parents_running"), 4)

    def test_parent_mutation_commands_are_absent(self) -> None:
        lowered = self.text.lower()
        forbidden = (
            "scan" + "cel",
            "scontrol " + "release",
            "p" + "kill",
            "kill" + "all",
        )
        for token in forbidden:
            self.assertNotIn(token, lowered)
        self.assertNotRegex(lowered, r"kill[^\n]*(135407|135411|135412)")
        self.assertNotRegex(lowered, r"kill[^\n]*(batch|extern|sleep infinity)")

    def test_only_registered_background_srun_pids_are_signalled(self) -> None:
        self.assertIn('pid=$!', self.text)
        self.assertIn('register_child_pid "${pid}" "${job}" "${node}"', self.text)
        self.assertIn('registered_child_pids+=("${pid}")', self.text)
        self.assertIn('current_arm_pids+=("${pid}")', self.text)
        self.assertIn("CHILD_SRUN_REGISTERED", self.text)
        self.assertIn("unregister_child_pid", self.text)
        self.assertIn('unregister_child_pid "${p0}"', self.text)
        self.assertIn('unregister_child_pid "${p1}"', self.text)
        self.assertIn("child PID registry was not empty before arm launch", self.text)
        self.assertIn("child PID registry was not empty after arm reap", self.text)
        self.assertEqual(self.text.count('kill -"${signal}" "${pid}"'), 1)
        self.assertNotRegex(self.text, r"kill\s+-(?:TERM|KILL)\s+")
        for token in (
            'safe_signal_child "${pid}" TERM',
            'safe_signal_child "${pid}" KILL',
            'safe_signal_child "${p0}" TERM',
            'safe_signal_child "${p1}" TERM',
            'safe_signal_child "${p0}" KILL',
            'safe_signal_child "${p1}" KILL',
        ):
            self.assertIn(token, self.text)
        self.assertIn('wait "${pid}"', self.text)
        self.assertIn("wait_for_steps_gone", self.text)

    def test_pid_identity_and_atomic_signal_window_are_closed(self) -> None:
        for token in (
            'proc_field "${pid}" 4',
            'proc_field "${pid}" 22',
            'readlink -f -- "/proc/${pid}/exe"',
            'sha256_file "/proc/${pid}/cmdline"',
            '[[ "${ppid}" == "$$" ]]',
            '"$(basename -- "${exe}")" == srun',
            '" --jobid=${job} "',
            '" --nodelist=${node} "',
            "launch_critical=1",
            "launch_critical=0",
            '[[ -z "${pending_signal}" ]] || exit 130',
        ):
            self.assertIn(token, self.text)
        background = self.text.index('>"${log}" 2>&1 &')
        register = self.text.index('register_child_pid "${pid}" "${job}" "${node}"')
        release = self.text.index("launch_critical=0", register)
        pending = self.text.index('[[ -z "${pending_signal}" ]] || exit 130', release)
        self.assertLess(background, register)
        self.assertLess(register, release)
        self.assertLess(release, pending)
        self.assertIn('(( launch_critical == 1 )) && return 0', self.text)
        self.assertIn("ARM_TIMEOUT", self.text)

    def test_hidden_direct_process_and_gpu_checks_are_double_gated(self) -> None:
        self.assertIn("remote_process_snapshot", self.text)
        self.assertIn("ps -u ${holder_user} -ww", self.text)
        self.assertIn("assert_remote_process_idle", self.text)
        self.assertIn("assert_remote_gpu_idle", self.text)
        self.assertIn("assert_pair_idle_twice startup", self.text)
        self.assertIn('assert_pair_idle_twice "pre-${iid}-${arm}"', self.text)
        body = re.search(
            r"assert_pair_idle_twice\(\) \{(.*?)\n\}",
            self.text,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(body)
        self.assertEqual(body.group(1).count("assert_remote_idle_once"), 4)
        self.assertIn("sleep 2", body.group(1))

    def test_child_visible_gpu_preflight_runs_before_inference(self) -> None:
        self.assertIn("__node_preflight_exec", self.text)
        self.assertIn(
            'assert_rocm_snapshot_idle "${snapshot}" "child-visible preflight" child-visible',
            self.text,
        )
        self.assertIn("assert torch.cuda.device_count() == 2", self.text)
        self.assertIn("assert all(torch.cuda.memory_allocated(i) == 0", self.text)
        launch = self.text.index('"${controller_source}" __node_preflight_exec')
        runner = self.text.index('"${python_bin}" -B "${runner}"', launch)
        self.assertLess(launch, runner)
        self.assertIn("controller must be invoked by absolute path", self.text)

    def test_two_node_world4_and_rank_local_cache_are_exact(self) -> None:
        for token in (
            "--mem=56G",
            "--gres=gpu:mi210:2",
            "--torchrun-nnodes 2",
            "--torchrun-nproc-per-node 2",
            '--torchrun-node-rank "${node_rank}"',
            '--torchrun-master-addr "${node0}"',
            '--torchrun-worker-prefix "${rank_cache_exec}"',
            "NCCL_SOCKET_IFNAME=bond0",
            "GLOO_SOCKET_IFNAME=bond0",
            "NCCL_IB_DISABLE=1",
            "BERNINI_HELDOUT_RANK_CACHE_TOKEN",
        ):
            self.assertIn(token, self.text)
        self.assertIn(
            "f416f8f687a61788d4f687e500dae04af677072ec205fb560016cb5249b4f9c5",
            self.text,
        )
        self.assertLess(
            self.text.index('run_arm_pair "${iid}" frozen_base'),
            self.text.index('run_arm_pair "${iid}" trained_adapter'),
        )

    def test_full160_v2_stage_recovery_and_checkpoint_are_pinned(self) -> None:
        pins = (
            "16d9429ab1bc456a0d3faac5310efc1f0301678f1058dee4de74912f17ab0c19",
            "da277a895cd86d09697d0e7e2db1e8952c1aaff46b214cc537c61f5a94429453",
            "ec4064949b1cc8f4dba5d3c15fab375c9392a1407f9610ad01cee24c173bf822",
            "6bae2f9a70fb851fd1bf87f5a01a1064a8fe8a6a",
            "2eaa2f38b7a2cb220a3a4ecafe1ead9c53667a9108d20c9d8cc22dabbdb2c2f4",
            "001000a79ca69d7f6addb482424a7466e261e7f1b9f6f9d02f5ab7b2edac83b4",
            "c8aacbb931cb0b1fa8ffa1067ceb2ba908973a5496b47da1e50133725faee537",
            "acafda85f45db31ba299b89db597f4bdb4d1def1c23c90c06c785ba4407c0ab3",
            "3dadbd4a1f2551c34942c52bcae2694bb5a695e88b9a6d471f2720f4fc074c5d",
        )
        for pin in pins:
            self.assertIn(pin, self.text)
        self.assertIn("seer-full160-eval-overlay-v2", self.text)
        self.assertIn('r["eval_job_submitted"] is False', self.text)
        self.assertIn('r["global_step"] == r["max_steps"] == 160', self.text)

    def test_each_case_binds_overlay_then_runs_and_finalizes(self) -> None:
        ordered = (
            'tar --delay-directory-restore',
            'tar --no-same-owner --no-same-permissions -xf "${overlay_archive}"',
            '"${source_binder}" bind',
            '"${source_binder}" verify-receipt',
            'run_arm_pair "${iid}" frozen_base',
            'run_arm_pair "${iid}" trained_adapter',
            'verify-pair',
            '"${source_binder}" finalize-case',
            '"${source_binder}" verify-case',
        )
        positions = [self.text.index(token) for token in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertIn(
            "--trained-infer-runner infer_seer_same_state_full160_lora.py",
            self.text,
        )
        self.assertIn('--method-source-archive-sha256 "${training_archive_sha}"', self.text)

    def test_canary_and_full_core4_cli_are_separate(self) -> None:
        self.assertIn("canary <heldout-iid>", self.text)
        self.assertIn("core4", self.text)
        self.assertIn('iids=("$1")', self.text)
        self.assertIn(
            "iids=(99cde432839f4240 6ea45d35943742bb 311c82f83eca4a7f 6d346c38cf504493)",
            self.text,
        )
        self.assertIn('if [[ "${mode}" == core4 ]]', self.text)
        self.assertIn("verify-core4", self.text)

    def test_current_file_identity_is_reportable(self) -> None:
        digest = hashlib.sha256(CONTROLLER.read_bytes()).hexdigest()
        self.assertRegex(digest, r"[0-9a-f]{64}")


if __name__ == "__main__":
    unittest.main()
