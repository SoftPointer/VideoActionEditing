from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE = (
    REPO_ROOT
    / "tmp"
    / "launch_fullmotion_v14_smoke_20260801T211000Z.sh"
)
SUPERVISOR = (
    REPO_ROOT
    / "tmp"
    / "launch_wait_fullmotion_v14_job118150_envclosed_v2.sh"
)
SNAPSHOT_TEST_RUNNER = REPO_ROOT / "tmp" / "run_fullmotion_v14_snapshot_tests.sh"

RUN_ID = "fullmotion128_v14_20260801T211000Z"
SNAPSHOT_NAME = "goku-full-motion128-source-v14-20260801T211000Z"
TREE_PLACEHOLDER = "__V14_TREE_SHA__"
TREE_SHA = "4393e64d94eea22c1346cb0a55db81ca9c01863028e924687c24581ce345f6c8"
RELEASE_PLACEHOLDER = "__V14_RELEASE_CHALLENGE__"
RELEASE_CHALLENGE = (
    "25d184535d9cbbabad3b5c0140a38bb5be74ae04ed6c267074717bc26b6b8923"
)
SMOKE_INPUT_SHA = (
    "d433acc0ccc74a9ddbff3a701a7cf86dc27e2cad24cdae0bfe7fcfc499982896"
)
FULL_INPUT_SHA = (
    "e4536937d1eb3a065907eb5f6db16b910bea75ff1ec2cdaa17c414ee943c4e42"
)
SNAPSHOT_TESTS = (
    "methods/motive/tests/test_goku_full_motion_contract.py",
    "methods/motive/tests/test_goku_full_motion_instruction.py",
    "methods/motive/tests/test_goku_full_motion_qwen.py",
    "methods/motive/tests/test_goku_full_motion_finalize.py",
    "methods/motive/tests/test_wan22_full_motion_signed_release.py",
    "methods/motive/tests/test_goku_full_motion_postcheck.py",
    "methods/motive/tests/test_goku_full_motion_select128.py",
    "methods/motive/tests/test_goku_full_motion_smoke_gate.py",
    "methods/motive/tests/test_auh_full_motion_pipeline_existing_job.py",
    "methods/motive/tests/test_auh_full_motion_qwen_distributed_existing_job.py",
    "methods/motive/tests/test_auh_full_motion_qwen_runtime_contract.py",
    "methods/motive/tests/test_auh_full_motion_finalize_release_watcher.py",
    "methods/motive/tests/test_auh_full_motion_wan_dispatch_existing_job.py",
    "methods/motive/tests/test_full_motion_postcheck_dispatch_existing_job.py",
    "methods/motive/tests/test_full_motion_select128_controller.py",
)


class FullMotionV14LauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.smoke = SMOKE.read_text(encoding="utf-8")
        self.supervisor = SUPERVISOR.read_text(encoding="utf-8")
        self.snapshot_test_runner = SNAPSHOT_TEST_RUNNER.read_text(
            encoding="utf-8"
        )

    @staticmethod
    def _receipt_verifier(text: str) -> str:
        function_start = text.index("verify_snapshot_test_receipt() {")
        program_start = text.index("<<'PY'\n", function_start) + len("<<'PY'\n")
        program_end = text.index("\nPY\n}", program_start)
        return text[program_start:program_end]

    @staticmethod
    def _canonical_receipt(
        *, snapshot: Path, runner: Path, log: Path, tree_sha: str
    ) -> dict[str, object]:
        return {
            "exit_code": 0,
            "log_path": str(log),
            "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
            "runner_path": str(runner),
            "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
            "schema_version": (
                "motive-fullmotion128-snapshot-tests-receipt-v1"
            ),
            "snapshot_path": str(snapshot),
            "source_tree_sha256": tree_sha,
            "status": "pass",
            "test_paths": list(SNAPSHOT_TESTS),
        }

    @staticmethod
    def _write_receipt(path: Path, payload: dict[str, object]) -> None:
        path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        path.chmod(0o400)

    def test_launchers_have_valid_bash_syntax(self) -> None:
        for script in (SNAPSHOT_TEST_RUNNER, SMOKE, SUPERVISOR):
            completed = subprocess.run(
                ["bash", "-n", str(script)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"{script}: {completed.stderr}",
            )

    def test_supervisor_and_pipeline_use_absolute_clean_shell_environments(self) -> None:
        self.assertTrue(self.supervisor.startswith("#!/bin/bash\n"))
        self.assertIn("trusted_path=/usr/bin:/bin", self.supervisor)
        self.assertIn("DYLD_INSERT_LIBRARIES", self.supervisor)
        self.assertIn("LD_PRELOAD", self.supervisor)
        self.assertIn("PYTHONPATH", self.supervisor)
        self.assertIn(
            "exec /usr/bin/nohup /usr/bin/env -i", self.supervisor
        )
        self.assertIn(
            '/bin/bash --noprofile --norc "${pipeline_controller}"',
            self.supervisor,
        )
        self.assertNotIn("exec nohup env", self.supervisor)

        hostile_env = dict(os.environ)
        hostile_env.update(
            {
                "BASH_ENV": "/tmp/should-never-be-read",
                "PYTHONPATH": "/tmp/poison",
                "LD_PRELOAD": "/tmp/poison.dylib",
                "PATH": "/tmp/poison-bin",
            }
        )
        completed = subprocess.run(
            [
                "/usr/bin/env",
                "-i",
                "HOME=/tmp/trusted-home",
                "USER=trusted",
                "LOGNAME=trusted",
                "PATH=/usr/bin:/bin",
                "SHELL=/bin/bash",
                "LC_ALL=C",
                "LANG=C",
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                "[[ -z ${BASH_ENV+x} && -z ${PYTHONPATH+x} "
                "&& -z ${LD_PRELOAD+x} && $PATH == /usr/bin:/bin ]]",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=hostile_env,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_fixed_names_placeholders_hashes_and_job_ids(self) -> None:
        for text in (self.smoke, self.supervisor):
            self.assertIn(RUN_ID, text)
            self.assertIn(SNAPSHOT_NAME, text)
            self.assertNotIn("fullmotion128_v9_", text)
            self.assertNotIn("source-v9-", text)
            self.assertNotIn("20260801T142000Z", text)
            self.assertNotIn(
                "ed4a905ef008ee72347d47564ed8a719d628359ea045a4bf16770ea44e24e237",
                text,
            )
            self.assertNotIn("fullmotion128_v10_", text)
            self.assertNotIn("source-v10-", text)
            self.assertNotIn("20260801T154500Z", text)
            self.assertNotIn(
                "6d6782453c1d3c847b0a2fd505d99561b7eb8aa3d5309e96b95f41d373d07300",
                text,
            )
            self.assertNotIn("fullmotion128_v11_", text)
            self.assertNotIn("source-v11-", text)
            self.assertNotIn("20260801T164500Z", text)
            self.assertNotIn("fullmotion128_v12_", text)
            self.assertNotIn("source-v12-", text)
            self.assertNotIn("20260801T180000Z", text)
            self.assertNotIn("fullmotion128_v13_", text)
            self.assertNotIn("source-v13-", text)
            self.assertNotIn("20260801T195500Z", text)
        for field, text in (
            ("tree_sha", self.smoke),
            ("source_tree_sha", self.supervisor),
        ):
            self.assertRegex(
                text,
                rf"(?m)^{field}=(?:{TREE_PLACEHOLDER}|[0-9a-f]{{64}})$",
            )
        self.assertIn("job_id=118150", self.smoke)
        self.assertIn("job_id=118150", self.supervisor)
        for text in (self.smoke, self.supervisor):
            self.assertIn("job_name=fm128-v14-g8", text)
            self.assertIn("job_account=test-acc", text)
            self.assertIn("job_partition=faculty", text)
            self.assertIn(
                "control_dir=${base}/allocations/fullmotion128_g8_20260802T111000Z",
                text,
            )
        self.assertIn(f"smoke_input_sha={SMOKE_INPUT_SHA}", self.smoke)
        self.assertIn(f"smoke_input_sha={SMOKE_INPUT_SHA}", self.supervisor)
        self.assertIn(f"tree_sha={TREE_SHA}", self.smoke)
        self.assertIn(f"source_tree_sha={TREE_SHA}", self.supervisor)
        self.assertIn(f"tree_sha={TREE_SHA}", self.snapshot_test_runner)
        for text in (self.smoke, self.supervisor):
            self.assertIn("prepare_smoke8_uniform/candidates.jsonl", text)
        self.assertIn(f"full_input_sha={FULL_INPUT_SHA}", self.supervisor)
        self.assertIn("release_id=goku-full-motion128-v14-release", self.supervisor)
        for field in ("release_challenge", "claim_owner_token"):
            self.assertRegex(
                self.supervisor,
                rf"(?m)^{field}=(?:{RELEASE_PLACEHOLDER}|[0-9a-f]{{64}})$",
            )
            self.assertIn(f"{field}={RELEASE_CHALLENGE}", self.supervisor)
        self.assertNotIn(
            "d181358590b9cbdaa5de9968e1ba3bc798dc983bcdca79ef28768f8c3bcae591",
            self.supervisor,
        )
        self.assertNotIn(
            "c0a13e10ec1a685148542b7ed0f897328a93959b61569b5a9cf510e09c13a375",
            self.supervisor,
        )
        self.assertIn(RUN_ID, self.snapshot_test_runner)
        self.assertIn(SNAPSHOT_NAME, self.snapshot_test_runner)
        self.assertRegex(
            self.snapshot_test_runner,
            rf"(?m)^tree_sha=(?:{TREE_PLACEHOLDER}|[0-9a-f]{{64}})$",
        )

    def test_placeholders_are_explicitly_fail_closed(self) -> None:
        self.assertIn(
            'unbound_tree_sha_sentinel=__V14_"TREE"_SHA__', self.smoke
        )
        self.assertIn(
            '[[ "${tree_sha}" != "${unbound_tree_sha_sentinel}" ]]',
            self.smoke,
        )
        self.assertIn(
            'unbound_tree_sha_sentinel=__V14_"TREE"_SHA__',
            self.supervisor,
        )
        self.assertIn(
            'unbound_release_challenge_sentinel=__V14_"RELEASE"_CHALLENGE__',
            self.supervisor,
        )
        self.assertIn(
            '[[ "${source_tree_sha}" != "${unbound_tree_sha_sentinel}" ]]',
            self.supervisor,
        )
        self.assertIn(
            'unbound_tree_sha_sentinel=__V14_"TREE"_SHA__',
            self.snapshot_test_runner,
        )
        self.assertIn(
            '[[ "${tree_sha}" != "${unbound_tree_sha_sentinel}" ]]',
            self.snapshot_test_runner,
        )
        self.assertIn(
            '"${release_challenge}" != "${unbound_release_challenge_sentinel}"',
            self.supervisor,
        )
        self.assertIn(
            '"${claim_owner_token}" != "${unbound_release_challenge_sentinel}"',
            self.supervisor,
        )

    def test_unbound_tree_templates_stop_before_launch_preflight(self) -> None:
        unbound_smoke = self.smoke.replace(
            f"tree_sha={TREE_SHA}", f"tree_sha={TREE_PLACEHOLDER}", 1
        )
        with tempfile.TemporaryDirectory() as root:
            script = Path(root) / "unbound-v14-smoke.sh"
            script.write_text(unbound_smoke, encoding="utf-8")
            completed = subprocess.run(
                ["/bin/bash", str(script)],
                check=False,
                capture_output=True,
                text=True,
                env={
                    "HOME": "/tmp",
                    "PATH": "/usr/bin:/bin",
                    "LC_ALL": "C",
                    "LANG": "C",
                },
                timeout=5,
            )
        self.assertEqual(completed.returncode, 2, SMOKE)
        self.assertIn(
            "replace the v14 tree SHA placeholder", completed.stderr
        )
        runner_guard = self.snapshot_test_runner.index(
            '[[ "${tree_sha}" != "${unbound_tree_sha_sentinel}" ]]'
        )
        self.assertLess(
            runner_guard,
            self.snapshot_test_runner.index("for command in awk chmod"),
        )

    def test_global_placeholder_binding_does_not_poison_sentinels(self) -> None:
        tree_sha = "a" * 64
        release_challenge = "b" * 64
        for text in (
            self.smoke,
            self.supervisor,
            self.snapshot_test_runner,
        ):
            self.assertLessEqual(text.count(TREE_PLACEHOLDER), 1)
        self.assertLessEqual(self.supervisor.count(RELEASE_PLACEHOLDER), 2)

        def bind_field(text: str, field: str, value: str) -> str:
            bound, substitutions = re.subn(
                rf"(?m)^{field}=(?:__[A-Z0-9_]+__|[0-9a-f]{{64}})$",
                f"{field}={value}",
                text,
                count=1,
            )
            self.assertEqual(substitutions, 1, field)
            return bound

        bound_smoke = bind_field(self.smoke, "tree_sha", tree_sha)
        bound_runner = bind_field(
            self.snapshot_test_runner, "tree_sha", tree_sha
        )
        bound_supervisor = bind_field(
            self.supervisor, "source_tree_sha", tree_sha
        )
        bound_supervisor = bind_field(
            bound_supervisor, "release_challenge", release_challenge
        )
        bound_supervisor = bind_field(
            bound_supervisor, "claim_owner_token", release_challenge
        )
        self.assertNotIn(TREE_PLACEHOLDER, bound_smoke)
        self.assertNotIn(TREE_PLACEHOLDER, bound_runner)
        self.assertNotIn(TREE_PLACEHOLDER, bound_supervisor)
        self.assertNotIn(RELEASE_PLACEHOLDER, bound_supervisor)
        self.assertIn(f"tree_sha={tree_sha}", bound_smoke)
        self.assertIn(f"tree_sha={tree_sha}", bound_runner)
        self.assertIn(f"source_tree_sha={tree_sha}", bound_supervisor)
        self.assertIn(
            f"release_challenge={release_challenge}", bound_supervisor
        )
        self.assertIn(
            f"claim_owner_token={release_challenge}", bound_supervisor
        )
        self.assertIn(
            "unbound_tree_sha_sentinel=__V14_\"TREE\"_SHA__",
            bound_smoke,
        )
        self.assertIn(
            "unbound_tree_sha_sentinel=__V14_\"TREE\"_SHA__",
            bound_runner,
        )
        self.assertIn(
            "unbound_tree_sha_sentinel=__V14_\"TREE\"_SHA__",
            bound_supervisor,
        )
        self.assertIn(
            "unbound_release_challenge_sentinel="
            '__V14_"RELEASE"_CHALLENGE__',
            bound_supervisor,
        )
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            for name, text in (
                ("bound-snapshot-tests.sh", bound_runner),
                ("bound-smoke.sh", bound_smoke),
                ("bound-supervisor.sh", bound_supervisor),
            ):
                script = root_path / name
                script.write_text(text, encoding="utf-8")
                completed = subprocess.run(
                    ["bash", "-n", str(script)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    f"{name}: {completed.stderr}",
                )

    def test_snapshot_runner_has_exact_sealed_create_only_proof_chain(self) -> None:
        runner = self.snapshot_test_runner
        self.assertEqual(len(SNAPSHOT_TESTS), 15)
        for test_path in SNAPSHOT_TESTS:
            self.assertEqual(runner.count(test_path), 1)
        for marker in (
            "snapshot_critical_tests.receipt.json",
            "motive-fullmotion128-snapshot-tests-receipt-v1",
            '(( ${#tests[@]} == 15 ))',
            'chmod 0500 "${runner}"',
            'exec 3>"${log_temp}"',
            'ln -T -- "${log_temp}" "${log}"',
            "os.O_WRONLY | os.O_CREAT | os.O_EXCL",
            'ln -T -- "${receipt_temp}" "${receipt}"',
            'chmod 0400 "${receipt_temp}"',
            'separators=(",", ":")',
            "sort_keys=True",
            '"exit_code": 0',
            '"status": "pass"',
        ):
            self.assertIn(marker, runner)
        before = runner.index(
            "frozen snapshot verification failed before tests"
        )
        pytest_run = runner.index('pytest.main(["-q", *sys.argv[1:]])')
        after = runner.index(
            "frozen snapshot verification failed after tests"
        )
        receipt_build = runner.index('"schema_version": ')
        self.assertLess(before, pytest_run)
        self.assertLess(pytest_run, after)
        self.assertLess(after, receipt_build)
        self.assertLess(
            runner.index('(( test_status == 0 ))'), receipt_build
        )
        self.assertLess(
            runner.index('(( post_verify_status == 0 ))'), receipt_build
        )
        for forbidden in (
            '>"${log}"',
            '>"${receipt}"',
            "os.replace(",
            "snapshot_critical_tests.status",
        ):
            self.assertNotIn(forbidden, runner)

    def test_snapshot_receipt_verifiers_accept_exact_proof_and_reject_tamper(
        self,
    ) -> None:
        verifier_programs = (
            ("smoke", self._receipt_verifier(self.smoke)),
            ("supervisor", self._receipt_verifier(self.supervisor)),
        )
        self.assertEqual(verifier_programs[0][1], verifier_programs[1][1])
        tree_sha = "a" * 64
        for launcher, program in verifier_programs:
            for mutation in (
                "valid",
                "log_bytes",
                "runner_bytes",
                "test_list",
                "tree_sha",
                "noncanonical",
                "receipt_mode",
            ):
                with self.subTest(launcher=launcher, mutation=mutation), \
                    tempfile.TemporaryDirectory() as root:
                    root_path = Path(root)
                    snapshot = root_path / "snapshot"
                    snapshot.mkdir()
                    runner = root_path / "runner.sh"
                    runner.write_bytes(b"#!/bin/sh\nexit 0\n")
                    runner.chmod(0o500)
                    log = root_path / "tests.log"
                    log.write_bytes(b"15 passed\n")
                    log.chmod(0o400)
                    receipt = root_path / "receipt.json"
                    payload = self._canonical_receipt(
                        snapshot=snapshot,
                        runner=runner,
                        log=log,
                        tree_sha=tree_sha,
                    )
                    if mutation == "test_list":
                        payload["test_paths"] = list(SNAPSHOT_TESTS[:-1])
                    elif mutation == "tree_sha":
                        payload["source_tree_sha256"] = "b" * 64
                    self._write_receipt(receipt, payload)
                    if mutation == "log_bytes":
                        log.chmod(0o600)
                        log.write_bytes(b"tampered log\n")
                        log.chmod(0o400)
                    elif mutation == "runner_bytes":
                        runner.chmod(0o700)
                        runner.write_bytes(b"#!/bin/sh\nexit 1\n")
                        runner.chmod(0o500)
                    elif mutation == "noncanonical":
                        receipt.chmod(0o600)
                        receipt.write_text(
                            json.dumps(payload, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        receipt.chmod(0o400)
                    elif mutation == "receipt_mode":
                        receipt.chmod(0o600)
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-",
                            str(receipt),
                            str(snapshot),
                            tree_sha,
                            str(runner),
                            str(log),
                        ],
                        input=program,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    if mutation == "valid":
                        self.assertEqual(
                            completed.returncode,
                            0,
                            completed.stderr,
                        )
                    else:
                        self.assertNotEqual(
                            completed.returncode,
                            0,
                            f"accepted tamper={mutation}",
                        )

    def test_snapshot_receipt_gates_precede_smoke_pid_claim_and_launch(self) -> None:
        smoke_initial = self.smoke.index(
            "sealed v14 snapshot-test receipt preflight failed"
        )
        smoke_job_lookup = self.smoke.index('job_record="$(scontrol show job')
        smoke_final = self.smoke.index(
            "sealed v14 snapshot-test receipt changed before smoke launch"
        )
        smoke_root_create = self.smoke.index('mkdir "${smoke_root}"')
        self.assertLess(smoke_initial, smoke_job_lookup)
        self.assertLess(smoke_final, smoke_root_create)

        supervisor_initial = self.supervisor.index(
            "sealed v14 snapshot-test receipt preflight failed"
        )
        supervisor_pid = self.supervisor.index(
            'atomic_publish_lines "${supervisor_pid_file}"'
        )
        supervisor_final = self.supervisor.index(
            "sealed v14 snapshot-test receipt changed before allocation claim"
        )
        claim = self.supervisor.index(
            'atomic_publish_lines "${holder_claim}"'
        )
        pipeline_exec = self.supervisor.index(
            "exec /usr/bin/nohup /usr/bin/env -i"
        )
        self.assertLess(supervisor_initial, supervisor_pid)
        self.assertLess(supervisor_final, claim)
        self.assertLess(supervisor_final, pipeline_exec)

    def test_all_run_scoped_outputs_and_release_id_are_v14(self) -> None:
        output_markers = (
            "qwen_smoke8_v14",
            "qwen_smoke8_gate_v14.json",
            "qwen_full768_v14",
            "final_pool_v14",
            "wan_generation_v14",
            "postcheck_v14",
            "exact128_v14",
            "finalize_release_watcher_v14",
            "pipeline_v14",
            "goku-full-motion128-v14-release",
        )
        for marker in output_markers:
            self.assertIn(marker, self.supervisor)
        for forbidden in (
            "qwen_smoke8_v12",
            "qwen_smoke8_gate_v12.json",
            "qwen_full768_v12",
            "final_pool_v12",
            "wan_generation_v12",
            "postcheck_v12",
            "exact128_v12",
            "finalize_release_watcher_v12",
            "pipeline_v12",
            "goku-full-motion128-v12-release",
            "qwen_smoke8_v11",
            "qwen_smoke8_gate_v11.json",
            "qwen_full768_v11",
            "final_pool_v11",
            "wan_generation_v11",
            "postcheck_v11",
            "exact128_v11",
            "finalize_release_watcher_v11",
            "pipeline_v11",
            "goku-full-motion128-v11-release",
        ):
            self.assertNotIn(forbidden, self.supervisor)

    def test_smoke_idle_probe_overlaps_holder_memory(self) -> None:
        function = self.smoke.split("check_idle_node() {", 1)[1].split(
            "\n}\n", 1
        )[0]
        self.assertIn("--overlap", function)
        self.assertIn("--mem=0", function)
        self.assertIn("--exact", function)
        self.assertNotIn("--exclusive", function)
        self.assertNotIn("--mem=1G", function)
        for marker in (
            "rocm-smi --showuse --showmemuse --showmeminfo vram --csv",
            "END { exit !(seen == 8 && !bad) }",
            "rocm-smi --showpids --csv",
            "(used_value+0) > 1073741824",
            "(gpu_flag+0) != 0 || (vram+0) != 0",
        ):
            self.assertIn(marker, function)

    def test_smoke_is_eight_node_preclaim_distributed_and_create_only(self) -> None:
        for marker in (
            "NumNodes=8",
            "gres/gpu:mi210=64",
            "(( ${#nodes[@]} == 8 ))",
            'local node="${nodes[${shard_index}]}"',
            "for shard_index in 0 1 2 3 4 5 6 7; do",
            "--gpus-per-task=4 --gpu-bind=none",
            "--shard-index \"${shard_index}\" --num-shards 8",
            "--overlap",
            "--mem=0",
            '[[ ! -e "${output}" && ! -L "${output}"',
            "if len(rows) != 1:",
            "if any(len(assigned) != 1 for assigned in assigned_by_shard)",
            "if observed_iid != assigned_by_shard[shard_index][0]",
            "set(observed_iids) != set(expected_iids)",
            '[[ ! -e "${holder_claim}" && ! -L "${holder_claim}"',
            'ln -T -- "${smoke_gate_stage}" "${smoke_gate}"',
            'chmod 0400 "${smoke_gate_stage}"',
            'wait_for_shard_publications "${smoke_root}" 61 1',
            "absence after the bounded settle still fails",
        ):
            self.assertIn(marker, self.smoke)
        self.assertNotIn("--max-samples", self.smoke)
        topology_fail = self.smoke.index(
            'fail "one or more smoke shards or topology checks failed"'
        )
        gate_run = self.smoke.index(
            '"${python_bin}" -m motive.goku_full_motion_smoke_gate'
        )
        gate_publish = self.smoke.index(
            'ln -T -- "${smoke_gate_stage}" "${smoke_gate}"'
        )
        self.assertLess(topology_fail, gate_run)
        self.assertLess(gate_run, gate_publish)
        for forbidden in (
            'atomic_publish_lines "${holder_claim}"',
            '>"${holder_claim}"',
            '> "${holder_claim}"',
            'ln -T -- "${smoke_gate_stage}" "${holder_claim}"',
            'touch "${holder_claim}"',
        ):
            self.assertNotIn(forbidden, self.smoke)

    def test_smoke_shard_publication_settle_is_bounded_and_fail_closed(
        self,
    ) -> None:
        start = self.smoke.index("  wait_for_shard_publications() {")
        end = self.smoke.index("\n  }\n", start) + len("\n  }\n")
        function = self.smoke[start:end]
        harness = f"""\
set -Eeuo pipefail
root=$1
mode=$2
{function}
publish_all() {{
  local index output receipt
  for index in 0 1 2 3 4 5 6 7; do
    output=${{root}}/qwen_shard_$(printf '%03d' "${{index}}").jsonl
    receipt=${{output%.jsonl}}.receipt.json
    : >"${{output}}"
    : >"${{receipt}}"
  done
}}
case "${{mode}}" in
  delayed_visibility)
    (sleep 0.05; publish_all) &
    publisher=$!
    wait_for_shard_publications "${{root}}" 100 0.01
    wait "${{publisher}}"
    ;;
  permanent_absence)
    publish_all
    rm "${{root}}/qwen_shard_007.receipt.json"
    if wait_for_shard_publications "${{root}}" 2 0.01; then
      exit 91
    fi
    ;;
  symlink_is_not_plain)
    publish_all
    rm "${{root}}/qwen_shard_007.receipt.json"
    ln -s "${{root}}/qwen_shard_006.receipt.json" \
      "${{root}}/qwen_shard_007.receipt.json"
    if wait_for_shard_publications "${{root}}" 2 0.01; then
      exit 92
    fi
    ;;
  *) exit 99 ;;
esac
"""
        for mode in (
            "delayed_visibility",
            "permanent_absence",
            "symlink_is_not_plain",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                completed = subprocess.run(
                    ["bash", "-c", harness, "settle-test", root, mode],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    completed.stderr,
                )

    def test_v6_gate_is_sealed_before_supervisor_claim(self) -> None:
        for text in (self.smoke, self.supervisor):
            self.assertIn(
                "motive-goku-full-motion-qwen-smoke-gate-v6", text
            )
            self.assertNotIn(
                "motive-goku-full-motion-qwen-smoke-gate-v4", text
            )
            self.assertNotIn(
                "motive-goku-full-motion-qwen-smoke-gate-v5", text
            )
        for marker in (
            "motive-goku-full-motion-qwen-smoke-gate-failure-v1",
            "object_sha256(payload)",
            "authorizes_full_run",
            'chmod 0400 "${smoke_gate_stage}"',
            '"$(stat -c \'%a\' "${smoke_gate}")" == 400',
        ):
            self.assertIn(marker, self.smoke)
        gate_validation = self.supervisor.index(
            "strict v14 smoke-gate preflight failed"
        )
        claim_publication = self.supervisor.index(
            'atomic_publish_lines "${holder_claim}"'
        )
        self.assertLess(gate_validation, claim_publication)

    def test_supervisor_parent_child_launch_handshake_is_fail_closed(self) -> None:
        for marker in (
            "pipeline_v14_supervisor.pid",
            "pipeline_v14_launch_cancel_request.txt",
            "pipeline_v14_launch_token.txt",
            "motive-fullmotion128-pipeline-supervisor-pid-v1",
            "motive-fullmotion128-pipeline-launch-cancel-request-v1",
            "motive-fullmotion128-pipeline-launch-token-v1",
            'publish_launch_cancel_request "${postclaim_cancel_source}"',
            'child_resolve_launch_decision "${child_pipeline_pid}"',
            'publish_launch_token cancel "${validated_launch_cancel_source}"',
            'publish_launch_token go none "${pipeline_pid}"',
            '"pipeline_pid=${expected_pipeline_pid}"',
            "Rechecking after go closes that",
        ):
            self.assertIn(marker, self.supervisor)
        launch_block = self.supervisor.split(
            "# The child cannot exec the pipeline", 1
        )[1].split("# A caught signal may interrupt wait(1)", 1)[0]
        self.assertLess(
            launch_block.index("child_resolve_launch_decision"),
            launch_block.index("exec /usr/bin/nohup /usr/bin/env -i"),
        )
        parent_after_fork = launch_block.split("pipeline_pid=$!", 1)[1]
        self.assertIn("publish_launch_token", parent_after_fork)
        self.assertIn("atomic_publish_lines \"${pipeline_pid_file}\"", parent_after_fork)
        handler = self.supervisor.split("handle_signal() {", 1)[1].split(
            "\n}\n", 1
        )[0]
        self.assertIn(
            '[[ ! -e "${launch_token}" && ! -L "${launch_token}" ]]',
            handler,
        )
        self.assertIn("publish_launch_cancel_request", handler)

    def test_supervisor_launch_handshake_race_simulation(self) -> None:
        start = self.supervisor.index("validate_cancel_sentinel() {")
        end = self.supervisor.index("\npublish_supervisor_receipt() {")
        functions = self.supervisor[start:end]
        harness = f"""\
set -Eeuo pipefail
umask 077
root=$1
mode=$2
job_id=118150
run_id={RUN_ID}
source_tree_sha={'a' * 64}
claim_owner_token={'b' * 64}
smoke_input_sha={'c' * 64}
full_input_sha={'d' * 64}
supervisor_pid=$$
cancel_sentinel=${{root}}/cancel_sentinel
postclaim_cancel_marker=${{root}}/postclaim_cancel_marker
launch_cancel_request=${{root}}/launch_cancel_request
launch_token=${{root}}/launch_token
postclaim_cancel_requested=0
postclaim_cancel_source=""
validated_launch_cancel_source=""
validated_launch_token_status=""
validated_launch_token_source=""
validated_postclaim_cancel_source=""
claimed=1
claim_transition=0
pipeline_pid=4242
signal_count=0
terminal_status=active
pipeline_exit_code=125
fail() {{ printf 'FAIL: %s\\n' "$*" >&2; exit 2; }}
stat() {{
  [[ "$1" == -c ]] || return 97
  case "$2" in
    %a) printf '400\\n' ;;
    %U) id -un ;;
    *) return 98 ;;
  esac
}}
inject_before_link=none
ln() {{
  case "${{inject_before_link}}" in
    hup)
      inject_before_link=none
      kill -HUP "$$"
      ;;
    sentinel)
      inject_before_link=none
      printf '%s\\n' \\
        'schema=motive-fullmotion128-preclaim-cancel-v1' \\
        'status=cancel' \\
        "slurm_job_id=${{job_id}}" \\
        "run_id=${{run_id}}" \\
        "source_tree_sha256=${{source_tree_sha}}" \\
        "smoke_input_sha256=${{smoke_input_sha}}" \\
        "full_input_sha256=${{full_input_sha}}" \\
        >"${{cancel_sentinel}}"
      chmod 0400 "${{cancel_sentinel}}"
      ;;
  esac
  command ln "$@"
}}
{functions}
case "${{mode}}" in
  signal_during_go)
    # This is the state produced when a signal interrupts the parent's go
    # publication before its create-only link becomes visible.
    publish_launch_cancel_request signal_HUP
    publish_launch_token go none 4242
    decision="$(child_resolve_launch_decision 4242)"
    [[ "${{decision}}" == $'cancel\\tsignal_HUP' ]]
    ;;
  actual_signal_interrupts_go_link)
    trap 'handle_signal HUP 129' HUP
    inject_before_link=hup
    publish_launch_token go none 4242
    decision="$(child_resolve_launch_decision 4242)"
    [[ "${{decision}}" == $'cancel\\tsignal_HUP' ]]
    ;;
  sentinel_interrupts_go_link)
    inject_before_link=sentinel
    publish_launch_token go none 4242
    decision="$(child_resolve_launch_decision 4242)"
    [[ "${{decision}}" == $'cancel\\tvalidated_sentinel' ]]
    ;;
  signal_after_go)
    publish_launch_token go none 4242
    publish_launch_cancel_request signal_TERM
    [[ ! -e "${{launch_cancel_request}}" ]]
    decision="$(child_resolve_launch_decision 4242)"
    [[ "${{decision}}" == $'go\\tnone' ]]
    ;;
  parent_cancel)
    publish_launch_cancel_request validated_sentinel
    publish_launch_token cancel validated_sentinel 4242
    decision="$(child_resolve_launch_decision 4242)"
    [[ "${{decision}}" == $'cancel\\tvalidated_sentinel' ]]
    ;;
  actual_fork_pid_binding)
    (
      child_pid=${{BASHPID}}
      decision="$(child_resolve_launch_decision "${{child_pid}}")"
      [[ "${{decision}}" == $'go\\tnone' ]]
    ) &
    child_pid=$!
    publish_launch_token go none "${{child_pid}}"
    wait "${{child_pid}}"
    ;;
  *) exit 99 ;;
esac
"""
        for mode in (
            "signal_during_go",
            "actual_signal_interrupts_go_link",
            "sentinel_interrupts_go_link",
            "signal_after_go",
            "parent_cancel",
            "actual_fork_pid_binding",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                completed = subprocess.run(
                    ["bash", "-c", harness, "handshake-test", root, mode],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    f"stdout={completed.stdout}\nstderr={completed.stderr}",
                )


if __name__ == "__main__":
    unittest.main()
