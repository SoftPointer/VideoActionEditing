from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TRANSITION = (
    REPO_ROOT / "tmp" / "run_fullmotion_v14_after_smoke_transition.sh"
)
SUPERVISOR = (
    REPO_ROOT
    / "tmp"
    / "launch_wait_fullmotion_v14_job118150_envclosed_v2.sh"
)
SIGNER = REPO_ROOT / "tmp" / "fullmotion_v14_offline_sign_upload.sh"
WATCHER = REPO_ROOT / "tmp" / "run_fullmotion_v14_signer_watch.sh"
SCREENRC = REPO_ROOT / "tmp" / "fullmotion_v14_signer.screenrc"
TRANSITION_SCREENRC = (
    REPO_ROOT / "tmp" / "fullmotion_v14_after_smoke_transition.screenrc"
)

TREE_SHA = "4393e64d94eea22c1346cb0a55db81ca9c01863028e924687c24581ce345f6c8"
INPUT_SHA = "d433acc0ccc74a9ddbff3a701a7cf86dc27e2cad24cdae0bfe7fcfc499982896"
CANARY = "1dbe39537c984690"
EXPECTED_HASHES = {
    SUPERVISOR: "5212c53ebf71d64e6cb2159ae212f6f48fd8184a9fb6e15029b96cdd8e0be264",
    SIGNER: "72e4b1bd94519dcdeb68754f84d1ab9e894d8e944bc97e8036c0104db65d4ab5",
    WATCHER: "99615a5db9207947478c7bdcfacfc91d6dc0fef32767309e858c3c38297d7316",
}


class FullMotionV14AfterSmokeTransitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.script = TRANSITION.read_text(encoding="utf-8")

    def test_bash_syntax_and_private_mode(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(TRANSITION)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(stat.S_IMODE(TRANSITION.stat().st_mode), 0o500)
        self.assertFalse(TRANSITION_SCREENRC.is_symlink())
        self.assertEqual(
            stat.S_IMODE(TRANSITION_SCREENRC.stat().st_mode), 0o600
        )
        self.assertFalse(SCREENRC.is_symlink())
        self.assertEqual(stat.S_IMODE(SCREENRC.stat().st_mode), 0o600)
        screenrc = TRANSITION_SCREENRC.read_text(encoding="utf-8")
        self.assertIn("deflog on", screenrc)
        self.assertIn("fullmotion_v14_after_smoke_transition.log", screenrc)
        self.assertIn("logfile flush 1", screenrc)
        self.assertIn(
            "/usr/bin/env -i ... /bin/bash --noprofile --norc", self.script
        )
        self.assertIn(
            "fullmotion_v14_after_smoke_transition.launchd.out", self.script
        )
        self.assertIn(
            "fullmotion_v14_after_smoke_transition.launchd.err", self.script
        )

    def test_all_bound_local_inputs_have_exact_hashes(self) -> None:
        for path, expected in EXPECTED_HASHES.items():
            self.assertFalse(path.is_symlink())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
            self.assertIn(expected, self.script)
        self.assertIn(f"source_tree_sha={TREE_SHA}", self.script)
        self.assertIn(
            'unbound_tree_sha_sentinel=__V14_"TREE"_SHA__', self.script
        )
        tree_guard = self.script.index(
            '[[ "${source_tree_sha}" != "${unbound_tree_sha_sentinel}" ]]'
        )
        remote_probe = self.script.index("probe_remote_gate()")
        self.assertLess(tree_guard, remote_probe)
        self.assertLess(
            tree_guard,
            self.script.index('mkdir -m 0700 -- "${transition_lock}"'),
        )
        self.assertLess(
            tree_guard,
            self.script.index('launchctl submit -l "${signer_label}"'),
        )
        self.assertIn(f"smoke_input_sha={INPUT_SHA}", self.script)
        self.assertNotIn("fullmotion128_v12_", self.script)
        self.assertNotIn("source-v12-", self.script)
        self.assertNotIn("qwen_smoke8_v12", self.script)

    def test_ssh_policy_is_exact_and_no_key_can_be_transferred(self) -> None:
        options = self.script.split("ssh_options=(\n", 1)[1].split("\n)", 1)[0]
        self.assertEqual(
            [line.strip() for line in options.splitlines() if line.strip()],
            [
                "-o BatchMode=yes",
                "-o ControlMaster=no",
                "-o ConnectTimeout=10",
                "-o StrictHostKeyChecking=yes",
            ],
        )
        for forbidden in (
            "scp ",
            "rsync ",
            "sftp ",
            "id_ed25519",
            "motive-fullmotion-release.llVNJy",
            "pipeline_claim_118150",
            "scontrol ",
            "sbatch ",
            "scancel ",
        ):
            self.assertNotIn(forbidden, self.script)

    def test_gate_replay_strictly_precedes_signer_and_supervisor(self) -> None:
        replay = self.script.index(
            '"${python_bin}" -P -m motive.goku_full_motion_smoke_gate'
        )
        byte_compare = self.script.index(
            'cmp -s -- "${gate}" "${replay_gate}"'
        )
        authorization = self.script.index(
            "strict gate replay passed sha256=%s"
        )
        signer = self.script.index(
            'launchctl submit -l "${signer_label}"'
        )
        watcher_recheck = self.script.index(
            "launchd signer watcher stopped or changed before supervisor launch"
        )
        launch = self.script.index('launch_output="$(launch_remote_supervisor)"')
        self.assertLess(replay, byte_compare)
        self.assertLess(byte_compare, authorization)
        self.assertLess(authorization, signer)
        self.assertLess(signer, watcher_recheck)
        self.assertLess(watcher_recheck, launch)
        for marker in (
            "--expected-tree-sha256 \"${tree_sha}\"",
            "--minimum-hard-passes 3",
            "--minimum-canary-dynamic-units 2",
            "SCHEMA_VERSION",
            'gate.get("status") != "pass"',
            'len(dynamic_ids) != 2',
            'oracle_ids != {"left_person", "right_person"}',
            '"camera locked off" not in camera_clause',
        ):
            self.assertIn(marker, self.script)

    def test_remote_python_cannot_import_from_login_working_directory(self) -> None:
        self.assertIn(
            '"${python_bin}" -I "${snapshot_tool}" verify', self.script
        )
        self.assertIn(
            '"${python_bin}" -P - "${gate}" "${smoke_input}"', self.script
        )
        self.assertIn(
            '"${python_bin}" -P -m motive.goku_full_motion_smoke_gate',
            self.script,
        )
        self.assertIn(
            '"${python_bin}" -I - \\\n    "${pid_receipt}"', self.script
        )

    def test_complete_transition_has_atomic_local_lock(self) -> None:
        acquire = self.script.index(
            'mkdir -m 0700 -- "${transition_lock}"'
        )
        gate = self.script.index("probe_remote_gate()")
        signer = self.script.index(
            'launchctl submit -l "${signer_label}"'
        )
        launch = self.script.index('launch_output="$(launch_remote_supervisor)"')
        self.assertLess(acquire, gate)
        self.assertLess(acquire, signer)
        self.assertLess(acquire, launch)
        for marker in (
            ".fullmotion_v14_after_smoke_transition.lock",
            "transition_lock_acquired=0",
            "trap cleanup_transition_lock EXIT",
            'rmdir -- "${transition_lock}"',
            "another transition is active or a stale lock requires inspection",
        ):
            self.assertIn(marker, self.script)

    def test_terminal_failure_and_invalid_gate_never_continue(self) -> None:
        failure_case = self.script.index("10)\n      fail \"v14 Qwen smoke")
        signer = self.script.index(
            'launchctl submit -l "${signer_label}"'
        )
        self.assertLess(failure_case, signer)
        self.assertIn("75 means not published yet", self.script)
        self.assertIn("76 means an unsafe or invalid artifact", self.script)
        self.assertIn(
            'fail "v14 smoke gate is terminal but unsafe or invalid',
            self.script,
        )

    def test_signer_launchd_job_is_clean_fresh_exact_and_verified(self) -> None:
        for marker in (
            "transition_label=com.siriuschu.fullmotion.v14.transition",
            "signer_label=com.siriuschu.fullmotion.v14.signer",
            'signer_domain=gui/$(id -u)',
            'launchctl print "${signer_service}"',
            'launchctl submit -l "${signer_label}"',
            '-o "${signer_stdout}" -e "${signer_stderr}" --',
            'program = /usr/bin/env',
            '"${signer_launch_command[@]}"',
            '/bin/bash --noprofile --norc "${local_signer_watcher}"',
            'MOTIVE_FULL_MOTION_SIGNER_WATCHER_SHA256=',
            "fullmotion_v14_signer.launchd.out",
            "fullmotion_v14_signer.launchd.err",
            'trimmed_line_count "type = Submitted"',
            'trimmed_line_count "state = running"',
            'trimmed_line_count "program = /usr/bin/env"',
            '"${listed_arguments}" == "${expected_signer_arguments}"',
            'trimmed_line_count "stdout path = ${signer_stdout}"',
            'trimmed_line_count "stderr path = ${signer_stderr}"',
            '"${command_line}" == "/bin/bash --noprofile --norc ${local_signer_watcher}"',
            "refusing to reuse a pre-existing launchd signer authorization",
            "refusing fresh launchd signer with a pre-existing log",
            "bound launchd signer watcher identity changed",
            "fullmotion_v14_signer_authorization_window_envclosed_v2.txt",
            "signer_window_seconds=345600",
        ):
            self.assertIn(marker, self.script)
        for forbidden in (
            "screen -ls",
            "screen -c",
            "signer_session=",
            'touch "${signer_stdout}"',
            'touch "${signer_stderr}"',
            '>"${signer_stdout}"',
            '>"${signer_stderr}"',
        ):
            self.assertNotIn(forbidden, self.script)

        start = self.script.index("trimmed_line_count() {")
        end = self.script.index('\nfor signer_log in "', start)
        functions = self.script[start:end]
        harness = f"""\
set -Eeuo pipefail
mode=$1
signer_service=gui/501/com.siriuschu.fullmotion.v14.signer
local_signer_watcher=/workspace/tmp/run_fullmotion_v14_signer_watch.sh
signer_stdout=/workspace/tmp/fullmotion_v14_signer.launchd.out
signer_stderr=/workspace/tmp/fullmotion_v14_signer.launchd.err
watcher_sha={'a' * 64}
expected_signer_arguments="/usr/bin/env
-i
HOME=/Users/siriuschu
USER=siriuschu
LOGNAME=siriuschu
PATH=/usr/bin:/bin:/usr/sbin:/sbin
SHELL=/bin/bash
LC_ALL=C
LANG=C
MOTIVE_FULL_MOTION_SIGNER_WATCHER_SHA256=${{watcher_sha}}
/bin/bash
--noprofile
--norc
${{local_signer_watcher}}"
launchctl() {{
  [[ "$1" == print && "$2" == "${{signer_service}}" ]]
  printf '%s\n' \
    "${{signer_service}} = {{" \
    "type = Submitted" \
    "state = ${{fixture_state}}" \
    "program = ${{fixture_program}}" \
    "arguments = {{"
  printf '%s\n' "${{fixture_arguments}}"
  printf '%s\n' \
    "}}" \
    "stdout path = ${{fixture_stdout}}" \
    "stderr path = ${{fixture_stderr}}" \
    "${{fixture_poison}}" \
    "pid = 4242" \
    "${{fixture_extra_pid}}" \
    "}}"
}}
kill() {{ [[ "$1" == -0 && "$2" == 4242 ]]; }}
ps() {{ printf '%s\n' "${{fixture_command}}"; }}
{functions}
fixture_state=running
fixture_program=/usr/bin/env
fixture_arguments=${{expected_signer_arguments}}
fixture_stdout=${{signer_stdout}}
fixture_stderr=${{signer_stderr}}
fixture_extra_pid=
fixture_poison=
fixture_command="/bin/bash --noprofile --norc ${{local_signer_watcher}}"
case "${{mode}}" in
  valid) verify_signer_launchd_job >/dev/null ;;
  stopped) fixture_state=exited ;;
  wrong_program) fixture_program=/bin/bash ;;
  extra_argument) fixture_arguments="${{fixture_arguments}}
/tmp/extra" ;;
  poison_environment) fixture_poison='BASH_ENV => /tmp/evil' ;;
  wrong_stdout) fixture_stdout=/tmp/wrong.out ;;
  duplicate_pid) fixture_extra_pid='pid = 4343' ;;
  wrong_command) fixture_command='/bin/bash /tmp/wrong' ;;
  *) exit 99 ;;
esac
if [[ "${{mode}}" != valid ]] && verify_signer_launchd_job >/dev/null 2>&1; then
  exit 90
fi
"""
        for mode in (
            "valid",
            "stopped",
            "wrong_program",
            "extra_argument",
            "poison_environment",
            "wrong_stdout",
            "duplicate_pid",
            "wrong_command",
        ):
            with self.subTest(mode=mode):
                completed = subprocess.run(
                    ["bash", "-c", harness, "launchd-verifier", mode],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_remote_supervisor_start_is_create_only_and_process_bound(self) -> None:
        for marker in (
            '[[ ! -e "${path}" && ! -L "${path}" ]] || exit 76',
            'ln -T -- "${log_stage}" "${outer_log}"',
            '/usr/bin/nohup /usr/bin/setsid /usr/bin/env -i',
            '/bin/bash --noprofile --norc "${supervisor}"',
            "schema=motive-fullmotion128-pipeline-supervisor-pid-v1",
            '"source_tree_sha256=" + tree_sha',
            "os.kill(pid, 0)",
            'Path(f"/proc/{pid}/cmdline")',
            'argv != ["/bin/bash", "--noprofile", "--norc", supervisor]',
            'sleep 3\n[[ ! -e "${terminal_receipt}"',
            '[[ "$(verify_pid_receipt)" == "${supervisor_pid}" ]]',
        ):
            self.assertIn(marker, self.script)
        self.assertEqual(
            self.script.count('/usr/bin/nohup /usr/bin/setsid /usr/bin/env -i'),
            1,
        )

    def test_failed_remote_launch_terminates_its_unverified_session(self) -> None:
        start = self.script.index(
            '/usr/bin/nohup /usr/bin/setsid /usr/bin/env -i'
        )
        verified = self.script.index("launch_verified=1", start)
        proof = self.script.index("printf 'STARTED\\t%s\\t%s\\t%s\\n'", start)
        self.assertLess(start, verified)
        self.assertLess(verified, proof)
        for marker in (
            "launch_verified=0",
            "cleanup_unverified_launch()",
            "trap cleanup_unverified_launch EXIT",
            'kill -TERM -- "-${outer_pid}"',
            'kill -TERM -- "${outer_pid}"',
        ):
            self.assertIn(marker, self.script)

    def test_hashes_are_rechecked_after_gate_and_before_remote_launch(self) -> None:
        gate = self.script.index("strict gate replay passed sha256=%s")
        submit = self.script.index('launchctl submit -l "${signer_label}"')
        remote_launch = self.script.index(
            'launch_output="$(launch_remote_supervisor)"'
        )
        signer_check = (
            'verify_local_file "${local_signer}" "${local_signer_sha}" yes'
        )
        watcher_check = (
            'verify_local_file "${local_signer_watcher}" \\\n'
            '  "${local_signer_watcher_sha}" yes'
        )
        self.assertGreaterEqual(self.script.count(signer_check), 3)
        self.assertGreaterEqual(self.script.count(watcher_check), 3)
        self.assertLess(gate, self.script.index(signer_check, gate))
        self.assertLess(self.script.rindex(signer_check, 0, submit), submit)
        self.assertLess(
            self.script.rindex(signer_check, 0, remote_launch), remote_launch
        )

    def test_success_is_sealed_then_enters_inert_non_relaunching_state(self) -> None:
        publish = self.script.index(
            'atomic_publish_local_lines "${transition_terminal}"'
        )
        launch = self.script.index(
            'launch_output="$(launch_remote_supervisor)"'
        )
        inert = self.script.index(
            'inert_forever "supervisor launch is sealed; relaunch is disabled"'
        )
        self.assertLess(launch, publish)
        self.assertLess(publish, inert)
        for marker in (
            "fullmotion_v14_after_smoke_transition_terminal_envclosed_v2.txt",
            "motive-fullmotion-v14-after-smoke-transition-terminal-v2",
            "status=supervisor_started",
            'if [[ -e "${transition_terminal}" || -L "${transition_terminal}" ]]',
            'inert_forever "validated supervisor-started receipt"',
        ):
            self.assertIn(marker, self.script)

    def test_every_remote_bash_is_started_through_clean_absolute_chain(self) -> None:
        self.assertNotIn('"${remote_host}" bash -s --', self.script)
        self.assertEqual(
            self.script.count(
                '"${remote_host}" "${remote_clean_bash[@]}"'
            ),
            3,
        )
        for marker in (
            "remote_clean_bash=(",
            "/usr/bin/env -i",
            "/bin/bash --noprofile --norc -s --",
            "trusted_local_path=/usr/bin:/bin:/usr/sbin:/sbin",
            "remote_path=/usr/bin:/bin",
            "DYLD_INSERT_LIBRARIES",
            "LD_PRELOAD",
            "PYTHONPATH",
        ):
            self.assertIn(marker, self.script)

    @staticmethod
    def _gate_validator(script: str) -> str:
        start = script.index("<<'PY_GATE'\n") + len("<<'PY_GATE'\n")
        end = script.index("\nPY_GATE\n", start)
        return script[start:end]

    @staticmethod
    def _write_canonical(path: Path, value: dict[str, object]) -> None:
        if path.exists():
            path.chmod(0o600)
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        path.chmod(0o400)

    def test_embedded_gate_validator_accepts_only_canonical_bound_pass(self) -> None:
        validator = self._gate_validator(self.script)
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            # Keep this focused verifier test independent of optional video
            # dependencies (numpy/cv2).  Tiny fake modules expose only the
            # frozen constants and canonical digest primitive consumed by the
            # embedded verifier.  Production still imports the tree-verified
            # real modules and then performs a complete real gate replay.
            fake_root = root_path / "fake_modules"
            fake_motive = fake_root / "motive"
            fake_motive.mkdir(parents=True)
            (fake_motive / "__init__.py").write_text("", encoding="utf-8")
            (fake_motive / "goku_full_motion_contract.py").write_text(
                "import hashlib, json\n"
                "def object_sha256(value):\n"
                "    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,"
                "separators=(',',':'),allow_nan=False).encode('utf-8')\n"
                "    return hashlib.sha256(raw).hexdigest()\n",
                encoding="utf-8",
            )
            qwen_constants = {
                "CHANGE_REGION_PROPOSALS_SCHEMA": (
                    "motive-goku-full-motion-change-region-proposals-v1"
                ),
                "COVERAGE_AUTHORITY_SCHEMA": (
                    "motive-goku-full-motion-coverage-authority-v2"
                ),
                "COVERAGE_AUTHORITY_INVENTORY_SCHEMA": (
                    "motive-goku-full-motion-coverage-authority-inventory-v1"
                ),
                "COVERAGE_AUTHORITY_ASSIGNMENTS_SCHEMA": (
                    "motive-goku-full-motion-coverage-authority-assignments-v1"
                ),
                "COVERAGE_AUTHORITY_ALLOWED_OWNER_MAP_SCHEMA": (
                    "motive-goku-full-motion-coverage-authority-allowed-owner-map-v1"
                ),
                "COVERAGE_AUTHORITY_ALIGNMENT_SCHEMA": (
                    "motive-goku-full-motion-coverage-authority-alignment-v2"
                ),
            }
            (fake_motive / "goku_full_motion_qwen.py").write_text(
                "\n".join(
                    f"{key}={value!r}" for key, value in qwen_constants.items()
                )
                + "\n",
                encoding="utf-8",
            )
            smoke_constants = {
                "FAILURE_SCHEMA_VERSION": (
                    "motive-goku-full-motion-qwen-smoke-gate-failure-v1"
                ),
                "REQUIRED_HARD_GATE_SCHEMA": "goku-full-motion-hard-gate-v6",
                "REQUIRED_PROVENANCE_SCHEMA": (
                    "goku-full-motion-qwen-provenance-v6"
                ),
                "REQUIRED_RECORD_SCHEMA": "goku-full-motion-qwen-record-v6",
                "REQUIRED_SOURCE_ALIGNMENT_SCHEMA": (
                    "motive-goku-full-motion-source-inventory-alignment-v4"
                ),
                "SCHEMA_VERSION": "motive-goku-full-motion-qwen-smoke-gate-v6",
            }
            (fake_motive / "goku_full_motion_smoke_gate.py").write_text(
                "\n".join(
                    f"{key}={value!r}" for key, value in smoke_constants.items()
                )
                + "\n",
                encoding="utf-8",
            )
            env = dict(os.environ)
            env["PYTHONPATH"] = str(fake_root)

            def object_sha256(value: object) -> str:
                raw = json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                return hashlib.sha256(raw).hexdigest()

            input_path = root_path / "candidates.jsonl"
            input_path.write_bytes(b"{}\n" * 8)
            input_path.chmod(0o400)
            actual_input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
            qwen_root = root_path / "qwen_smoke8_v14"
            qwen_root.mkdir()
            gate_path = root_path / "qwen_smoke8_gate_v14.json"

            actor_matches = [
                {"oracle_id": "left_person", "unit_id": "unit_01"},
                {"oracle_id": "right_person", "unit_id": "unit_02"},
            ]
            value: dict[str, object] = {
                "schema_version": smoke_constants["SCHEMA_VERSION"],
                "status": "pass",
                "input": {
                    "path": str(input_path.resolve(strict=True)),
                    "rows": 8,
                    "sha256": actual_input_sha,
                },
                "qwen_root": str(qwen_root),
                "qwen_lineage": {
                    "record": smoke_constants["REQUIRED_RECORD_SCHEMA"],
                    "hard_gate": smoke_constants["REQUIRED_HARD_GATE_SCHEMA"],
                    "provenance": smoke_constants["REQUIRED_PROVENANCE_SCHEMA"],
                    "source_inventory_alignment": smoke_constants[
                        "REQUIRED_SOURCE_ALIGNMENT_SCHEMA"
                    ],
                    "change_region_proposals": qwen_constants[
                        "CHANGE_REGION_PROPOSALS_SCHEMA"
                    ],
                    "coverage_authority": qwen_constants[
                        "COVERAGE_AUTHORITY_SCHEMA"
                    ],
                    "coverage_authority_inventory": qwen_constants[
                        "COVERAGE_AUTHORITY_INVENTORY_SCHEMA"
                    ],
                    "coverage_authority_assignments": qwen_constants[
                        "COVERAGE_AUTHORITY_ASSIGNMENTS_SCHEMA"
                    ],
                    "coverage_authority_allowed_owner_map": (
                        qwen_constants[
                            "COVERAGE_AUTHORITY_ALLOWED_OWNER_MAP_SCHEMA"
                        ]
                    ),
                    "coverage_authority_alignment": (
                        qwen_constants["COVERAGE_AUTHORITY_ALIGNMENT_SCHEMA"]
                    ),
                },
                "qwen_runtime": {},
                "receipt_bindings": [],
                "hard_passes": 3,
                "hard_pass_iids": [CANARY, "sample_a", "sample_b"],
                "hard_pass_bindings": [{}, {}, {}],
                "canary": {
                    "iid": CANARY,
                    "dynamic_unit_ids": ["unit_01", "unit_02"],
                    "primary_actor_matches": actor_matches,
                    "secondary_actor_matches": actor_matches,
                    "camera_clause": "Keep the camera locked off throughout.",
                    "compiled_instruction_sha256": "a" * 64,
                },
            }
            value["gate_digest"] = object_sha256(value)
            self._write_canonical(gate_path, value)

            command = [
                sys.executable,
                "-c",
                validator,
                str(gate_path),
                str(input_path),
                actual_input_sha,
                str(qwen_root),
                CANARY,
            ]
            passed = subprocess.run(
                command, check=False, capture_output=True, text=True, env=env
            )
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertEqual(
                passed.stdout.strip(),
                hashlib.sha256(gate_path.read_bytes()).hexdigest(),
            )

            # A fully re-digested pass-looking gate still fails when one of
            # the two independently moving canary actors is omitted.
            bad_canary = json.loads(json.dumps(value))
            bad_canary["canary"]["dynamic_unit_ids"] = ["unit_01"]
            bad_canary.pop("gate_digest")
            bad_canary["gate_digest"] = object_sha256(bad_canary)
            self._write_canonical(gate_path, bad_canary)
            rejected = subprocess.run(
                command, check=False, capture_output=True, text=True, env=env
            )
            self.assertNotEqual(rejected.returncode, 0)

            # A well-formed failure receipt is recognized as terminal but
            # returns the dedicated non-authorizing status 10.
            failure: dict[str, object] = {
                "schema_version": smoke_constants["FAILURE_SCHEMA_VERSION"],
                "status": "fail",
                "authorizes_full_run": False,
                "input_path": str(input_path),
                "qwen_root": str(qwen_root),
                "canary_iid": CANARY,
                "error_type": "FullMotionSmokeGateError",
                "error": "test terminal failure",
            }
            failure["failure_digest"] = object_sha256(failure)
            self._write_canonical(gate_path, failure)
            terminal_failure = subprocess.run(
                command, check=False, capture_output=True, text=True, env=env
            )
            self.assertEqual(terminal_failure.returncode, 10)

            # Semantically valid but non-canonical bytes are also rejected.
            self._write_canonical(gate_path, value)
            gate_path.chmod(0o600)
            gate_path.write_text(
                json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
            )
            gate_path.chmod(0o400)
            noncanonical = subprocess.run(
                command, check=False, capture_output=True, text=True, env=env
            )
            self.assertNotEqual(noncanonical.returncode, 0)


if __name__ == "__main__":
    unittest.main()
