from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
METHOD = REPO / "methods/bernini_action_editing"
AUDITS = METHOD / "audits"
SCRIPTS = METHOD / "scripts"

BOOTSTRAP = METHOD / "action_edit_level_b_p2_00435_bootstrap_0817_v4.py"
CAPACITY = METHOD / "action_edit_level_b_p2_00435_capacity_0817_v4.py"
RANK = SCRIPTS / "auh_action_edit_level_b_p2_00435_rank_exec_v4.sh"
STEP = SCRIPTS / "auh_action_edit_level_b_p2_00435_step_v4.sh"
CONTROLLER = SCRIPTS / "auh_launch_action_edit_level_b_p2_00435_job140846_v4.sh"
CORE = AUDITS / "fresh_world8_level_b_p2_00435_v4_LAUNCH_AUTHORITY_CORE.json"
PINS = AUDITS / "fresh_world8_level_b_p2_00435_v4_DEPLOYMENT_PINS.json"
OUTER = AUDITS / "auh_action_edit_level_b_p2_00435_outer_handoff_v4.sh"
KNOWN_HOSTS = AUDITS / "fresh_world8_level_b_p2_00435_v4_NODE279_KNOWN_HOSTS"
HARNESS = AUDITS / "calibrate_fresh_world8_level_b_p2_00435_v4_static_preflight.py"

EXPECTED = {
    BOOTSTRAP: ("1d72a1594ab52e258f0fbac5410ea1d27e5c557a12d76e88b806b3ac99794391", 57405),
    CAPACITY: ("87fc10c580070eef660fdfeaecf18ddd997d031a009508edbcc34a263cd6c4dc", 55392),
    RANK: ("64fc0df647ab28d950d81b6735aead559d1e91216416a8e44e8ac0c3707620c8", 9470),
    STEP: ("74f4ad83a198447246031a5b68ec6d812455dae0b6adee944e42c452bad0f0cc", 21720),
    CONTROLLER: ("e932a95815204438ecc4b8568c27d631d815e7fe85a799f5306891cba5ec92b8", 60584),
    CORE: ("166cb80170763562c8041d80b3ed771bb1088890261b71821d807df0f998c92a", 18110),
    KNOWN_HOSTS: ("376ed12f9662eba4fe41396853713c9e2ad30bc3069698016f295853ce3e4454", 142),
    HARNESS: ("9938ca3ec9fa5dfb682fb0f24d5b560293df7cae563fa5e720a3424b162f5e9e", 11628),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class LevelBV4LaunchboundTests(unittest.TestCase):
    maxDiff = None

    def _run_instrumented_outer(
        self,
        *,
        failure_label: str = "",
        extra_environment=None,
    ) -> dict:
        """Run the real outer control flow against path-relocated mock tools.

        Only remote identities, immutable hashes, and GPU probes are replaced;
        ordering, fail exits, root publication calls, and the sole screen call
        remain the outer script's own main program.
        """

        def replace_once(source: str, old: str, new: str) -> str:
            self.assertEqual(source.count(old), 1, old)
            return source.replace(old, new, 1)

        with tempfile.TemporaryDirectory(
            prefix="level-b-v4-outer-test-", dir="/private/tmp"
        ) as temporary_directory:
            temporary = Path(temporary_directory)
            experiment = temporary / "experiment"
            launch = experiment / "launchers" / "fresh-world8-level-b-p2-00435-v4"
            (experiment / "attempts").mkdir(parents=True)
            (experiment / "runs").mkdir(parents=True)
            launch.mkdir(parents=True)
            launch_names = {
                "LAUNCH_AUTHORITY_CORE.json",
                "action_edit_level_b_p2_00435_bootstrap_0817_v4.py",
                "action_edit_level_b_p2_00435_capacity_0817_v4.py",
                "auh_action_edit_level_b_p2_00435_rank_exec_v4.sh",
                "auh_action_edit_level_b_p2_00435_step_v4.sh",
                "auh_launch_action_edit_level_b_p2_00435_job140846_v4.sh",
                "node279_known_hosts",
            }
            for name in launch_names:
                member = launch / name
                member.write_bytes((name + "\n").encode())
                member.chmod(0o555 if name.endswith(".sh") else 0o444)
            launch.chmod(0o555)

            events = temporary / "events"
            mock_sha = temporary / "sha256sum"
            mock_sha.write_text(
                "#!" + sys.executable + "\n"
                "import hashlib,pathlib,sys\n"
                "if len(sys.argv)>1:\n"
                " data=pathlib.Path(sys.argv[1]).read_bytes(); label=sys.argv[1]\n"
                "else:\n"
                " data=sys.stdin.buffer.read(); label='-'\n"
                "print(hashlib.sha256(data).hexdigest()+'  '+label)\n"
            )
            mock_sha.chmod(0o755)
            mock_find = temporary / "find"
            mock_find.write_text(
                "#!" + sys.executable + "\n"
                "import pathlib,sys\n"
                "root=pathlib.Path(sys.argv[1])\n"
                "if '-printf' in sys.argv:\n"
                " print('\\n'.join(sorted(p.name for p in root.iterdir())))\n"
                "else:\n"
                " rows=sorted(root.rglob('*')) if root.exists() else []\n"
                " print(str(rows[0]) if rows else '')\n"
            )
            mock_find.chmod(0o755)
            mock_squeue = temporary / "squeue"
            mock_squeue.write_text(
                "#!" + sys.executable + "\n"
                "import sys\n"
                "if '--steps' not in sys.argv:\n"
                " print('RUNNING|auh7-1b-gpu-[246-248,279]|gres/gpu:mi210:8')\n"
            )
            mock_squeue.chmod(0o755)
            mock_screen = temporary / "screen"
            mock_screen.write_text(
                "#!/bin/bash\n"
                + "/usr/bin/printf 'screen:%s\\n' \"$*\" >> "
                + shlex.quote(str(events))
                + "\n"
            )
            mock_screen.chmod(0o755)

            static_raw = b'{"fixture":"static"}'
            static_transport = base64.b64encode(static_raw).decode()
            challenge = "1" * 64
            receipt_raw = canonical(
                {"sample_challenge": challenge, "sample_phase": "foreground"}
            )
            receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
            receipt_base64 = base64.b64encode(receipt_raw).decode()
            challenge_transport = base64.b64encode(challenge.encode()).decode()
            remote_transport = base64.b64encode(receipt_base64.encode()).decode()
            validation_transport = base64.b64encode(receipt_raw).decode()
            publication_transport = base64.b64encode(receipt_sha.encode()).decode()

            source = text(OUTER)
            source = replace_once(
                source,
                "readonly experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_action_editing_0817",
                "readonly experiment_root=" + shlex.quote(str(experiment)),
            )
            source = replace_once(
                source,
                "readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12",
                "readonly python_bin=" + shlex.quote(sys.executable),
            )
            source = replace_once(
                source,
                "readonly capacity_python=/usr/bin/python3.10",
                "readonly capacity_python=" + shlex.quote(sys.executable),
            )
            source = replace_once(
                source,
                "readonly screen_bin=/usr/bin/screen",
                "readonly screen_bin=" + shlex.quote(str(mock_screen)),
            )
            source = source.replace("/usr/bin/sha256sum", str(mock_sha))
            source = source.replace("/usr/bin/find", str(mock_find))
            source = source.replace("/usr/bin/squeue", str(mock_squeue))
            source = source.replace("/usr/bin/mkdir", "/bin/mkdir")
            source = replace_once(
                source,
                "readonly static_raw_sha=68edb1c3d5925d5ef26a2601b989f91777da3be4ad03d772aa5c2c7f1dde7998",
                "readonly static_raw_sha=" + hashlib.sha256(static_raw).hexdigest(),
            )
            source = replace_once(
                source,
                "readonly static_raw_size=29205",
                "readonly static_raw_size=" + str(len(static_raw)),
            )
            source = replace_once(
                source,
                "readonly static_b64_sha=e69abf4f8829c8748b271d94ceae717c014c181f9a39207f5fb17846af5b6f59",
                "readonly static_b64_sha="
                + hashlib.sha256(static_transport.encode()).hexdigest(),
            )
            source = replace_once(
                source,
                "readonly static_b64_size=38940",
                "readonly static_b64_size=" + str(len(static_transport)),
            )

            injected = f"""
# Test-only substitutions for external identities and probes.  The main outer
# program below is retained verbatim.
require_sha() {{ :; }}
require_stat() {{ :; }}
require_no_children() {{
  /usr/bin/printf 'children:%s\\n' "$1" >> {shlex.quote(str(events))}
}}
capture_framed_base64() {{
  local output_name="$1" label="$2"
  /usr/bin/printf 'frame:%s\\n' "${{label}}" >> {shlex.quote(str(events))}
  [[ {shlex.quote(failure_label)} != "${{label}}" ]] || fail "fixture ${{label}}"
  builtin printf -v "${{output_name}}" '%s' {shlex.quote(static_transport)}
}}
decode_base64() {{
  local output_name="$1" payload="$2" raw
  case "${{payload}}" in
    {shlex.quote(static_transport)}) raw={shlex.quote(static_raw.decode())} ;;
    {shlex.quote(challenge_transport)}) raw={shlex.quote(challenge)} ;;
    {shlex.quote(remote_transport)}) raw={shlex.quote(receipt_base64)} ;;
    {shlex.quote(receipt_base64)}) raw={shlex.quote(receipt_raw.decode())} ;;
    {shlex.quote(validation_transport)}) raw={shlex.quote(receipt_raw.decode())} ;;
    {shlex.quote(publication_transport)}) raw={shlex.quote(receipt_sha)} ;;
    *) fail "fixture decode differs" ;;
  esac
  builtin printf -v "${{output_name}}" '%s' "${{raw}}"
}}
capture_capacity() {{
  local output_name="$1" label="$2"
  shift 2
  /usr/bin/printf 'capacity:%s:%s\\n' "${{label}}" "${{1:-}}" >> {shlex.quote(str(events))}
  [[ {shlex.quote(failure_label)} != "${{label}}" ]] || fail "fixture ${{label}}"
  case "${{label}}" in
    'foreground challenge') builtin printf -v "${{output_name}}" '%s' {shlex.quote(challenge_transport)} ;;
    'foreground direct-node capacity') builtin printf -v "${{output_name}}" '%s' {shlex.quote(remote_transport)} ;;
    'foreground receipt validation') builtin printf -v "${{output_name}}" '%s' {shlex.quote(validation_transport)} ;;
    'foreground receipt publication')
      builtin printf '%s' {shlex.quote(receipt_raw.decode())} > "${{6}}"
      /bin/chmod 0444 "${{6}}"
      builtin printf -v "${{output_name}}" '%s' {shlex.quote(publication_transport)}
      ;;
    *) fail "fixture capacity label differs: ${{label}}" ;;
  esac
}}
create_runtime_root() {{
  local parent="$1" final="$2" stage="$3" label="$4"
  /usr/bin/printf 'root:%s\\n' "${{label}}" >> {shlex.quote(str(events))}
  [[ {shlex.quote(failure_label)} != "${{label}}" ]] || fail "fixture ${{label}}"
  [[ -d "${{parent}}" && ! -e "${{final}}" && ! -L "${{final}}" ]] || fail "fixture root precondition"
  /bin/mkdir -m 0700 -- "${{final}}"
}}
"""
            marker = "\nfor digest in "
            self.assertEqual(source.count(marker), 1)
            source = source.replace(marker, "\n" + injected + marker, 1)

            environment = {
                "PATH": "/usr/bin:/bin",
                "LC_ALL": "C",
                "LANG": "C",
                "HOME": "/vast/users/guangyi.chen",
                "BASH_ENV": "/dev/null",
                "LEVEL_B_V4_EXTERNAL_LAUNCH_AUTHORITY": "LEVEL_B_P2_00435_V4_REVIEWED_ONE_SHOT",
            }
            if extra_environment:
                environment.update(extra_environment)
            process = subprocess.run(
                ["/bin/bash", "--noprofile", "--norc", "-s"],
                input=source.encode(),
                cwd="/",
                env=environment,
                capture_output=True,
            )
            attempt = experiment / "attempts" / "fresh-world8-level-b-p2-00435-v4"
            run = experiment / "runs" / "fresh-world8-level-b-p2-00435-v4"
            screen = experiment / "screen-logs" / "fresh-world8-level-b-p2-00435-v4"
            return {
                "returncode": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "attempt_exists": attempt.exists(),
                "attempt_entries": sorted(p.name for p in attempt.iterdir())
                if attempt.exists()
                else [],
                "run_exists": run.exists(),
                "screen_exists": screen.exists(),
                "screen_entries": sorted(p.name for p in screen.iterdir())
                if screen.exists()
                else [],
                "events": events.read_text().splitlines() if events.exists() else [],
            }

    def test_exact_upstream_bytes(self) -> None:
        for path, (expected_sha, expected_size) in EXPECTED.items():
            raw = path.read_bytes()
            self.assertEqual(len(raw), expected_size, path)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected_sha, path)

    def test_core_and_pins_are_canonical_without_lf(self) -> None:
        for path in (CORE, PINS):
            raw = path.read_bytes()
            self.assertFalse(raw.endswith(b"\n"), path)
            value = json.loads(raw)
            self.assertEqual(raw, canonical(value), path)
        core = json.loads(CORE.read_bytes())
        pins = json.loads(PINS.read_bytes())
        self.assertEqual(core["schema_version"], "bernini-action-edit-level-b-p2-launch-authority-core-v4")
        self.assertEqual(pins["schema_version"], "bernini-action-edit-level-b-p2-deployment-pins-v4")
        self.assertFalse(pins["authorization"]["launch_authorized_by_artifact"])
        self.assertFalse(pins["remote_writes_performed_by_local_builder_as_of_pins_creation"])
        self.assertTrue(pins["status_is_historical_snapshot_not_runtime_authority"])

    def test_exact_seven_inventory_and_hash_chain(self) -> None:
        pins = json.loads(PINS.read_bytes())
        inventory = pins["file_inventory"]
        self.assertEqual(len(inventory), 7)
        self.assertEqual(
            {row["final_name"] for row in inventory},
            {
                "LAUNCH_AUTHORITY_CORE.json",
                "action_edit_level_b_p2_00435_bootstrap_0817_v4.py",
                "action_edit_level_b_p2_00435_capacity_0817_v4.py",
                "auh_action_edit_level_b_p2_00435_rank_exec_v4.sh",
                "auh_action_edit_level_b_p2_00435_step_v4.sh",
                "auh_launch_action_edit_level_b_p2_00435_job140846_v4.sh",
                "node279_known_hosts",
            },
        )
        for row in inventory:
            path = REPO / row["local_path"]
            self.assertEqual(len(path.read_bytes()), row["size"], path)
            self.assertEqual(digest(path), row["sha256"], path)
            self.assertEqual(row["nlink"], 1)
            self.assertIn(row["mode"], (0o444, 0o555))
        chain = pins["hash_chain"]
        self.assertEqual(chain["core_sha256"], digest(CORE))
        self.assertEqual(chain["controller_sha256"], digest(CONTROLLER))
        self.assertEqual(chain["outer_sha256"], digest(OUTER))

    def test_all_shells_absolute_and_parse(self) -> None:
        for path in (RANK, STEP, CONTROLLER, OUTER):
            raw = path.read_bytes()
            self.assertTrue(raw.startswith(b"#!/bin/bash\n"), path)
            self.assertNotIn(b"#!/usr/bin/env bash", raw)
            subprocess.run(["/bin/bash", "-n", str(path)], check=True)

    def test_no_masked_command_substitution_authority(self) -> None:
        forbidden = (
            re.compile(r"\b(?:readonly|local)\b[^\n]*\$\("),
            re.compile(r"\[\[[^\n]*\$\("),
        )
        for path in (RANK, STEP, CONTROLLER, OUTER):
            source = text(path)
            for pattern in forbidden:
                self.assertIsNone(pattern.search(source), (path, pattern.pattern))
            for line_number, line in enumerate(source.splitlines(), 1):
                if "$(" in line:
                    self.assertIn("if !", line, (path, line_number, line))

    def test_outer_function_output_propagation_smoke(self) -> None:
        source = text(OUTER)
        cap = re.search(r"(?ms)^capture_sha\(\) \{.*?^\}\n\nrequire_sha\(\) \{.*?^\}\n", source)
        self.assertIsNotNone(cap)
        functions = cap.group(0).replace(
            "/usr/bin/sha256sum", "/usr/bin/shasum -a 256"
        )
        script = (
            "set -Eeuo pipefail\n"
            "fail(){ exit 99; }\n"
            + functions
            + "require_sha /dev/null e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 smoke\n"
            + "result=; capture_sha /dev/null result smoke; [[ $result == e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 ]]\n"
        )
        subprocess.run(["/bin/bash", "-c", script], check=True)
        self.assertIn("local path=\"$1\" expected=\"$2\" label=\"$3\" require_sha_observed", functions)
        self.assertIn("builtin printf -v", functions)

    def test_outer_instrumented_failure_semantics_and_single_screen(self) -> None:
        for label in ("foreground static preflight", "foreground direct-node capacity"):
            with self.subTest(label=label):
                result = self._run_instrumented_outer(failure_label=label)
                self.assertEqual(result["returncode"], 96, result["stderr"])
                self.assertFalse(result["attempt_exists"])
                self.assertFalse(result["run_exists"])
                self.assertFalse(result["screen_exists"])
                self.assertFalse(any(row.startswith("screen:") for row in result["events"]))

        deployment_failure = self._run_instrumented_outer(failure_label="run root")
        self.assertEqual(deployment_failure["returncode"], 96, deployment_failure["stderr"])
        self.assertTrue(deployment_failure["attempt_exists"])
        self.assertEqual(
            deployment_failure["attempt_entries"],
            ["foreground-capacity-receipt.json"],
        )
        self.assertFalse(deployment_failure["run_exists"])
        self.assertFalse(deployment_failure["screen_exists"])
        self.assertFalse(
            any(row.startswith("screen:") for row in deployment_failure["events"])
        )

        success = self._run_instrumented_outer()
        self.assertEqual(success["returncode"], 0, success["stderr"])
        self.assertEqual(success["attempt_entries"], ["foreground-capacity-receipt.json"])
        self.assertTrue(success["run_exists"])
        self.assertTrue(success["screen_exists"])
        self.assertEqual(success["screen_entries"], ["controller.screen.log"])
        remote_samples = [
            row
            for row in success["events"]
            if row == "capacity:foreground direct-node capacity:remote-probe-base64"
        ]
        screen_calls = [row for row in success["events"] if row.startswith("screen:")]
        self.assertEqual(len(remote_samples), 1)
        self.assertEqual(len(screen_calls), 1)
        self.assertIn("-dmS bernini0817-level-b-p2-00435-v4 -c /dev/null", screen_calls[0])
        self.assertIn(
            "auh_launch_action_edit_level_b_p2_00435_job140846_v4.sh",
            screen_calls[0],
        )

    def test_outer_imported_function_and_shellopts_hostiles_fail_before_tools(self) -> None:
        imported_function = self._run_instrumented_outer(
            extra_environment={"BASH_FUNC_preloaded%%": "() { :;}"}
        )
        self.assertEqual(imported_function["returncode"], 96)
        self.assertIn(b"preloaded Bash function exists", imported_function["stderr"])
        self.assertEqual(imported_function["events"], [])
        self.assertFalse(imported_function["attempt_exists"])

        hostile_options = self._run_instrumented_outer(
            extra_environment={
                "SHELLOPTS": "braceexpand:hashall:interactive-comments:xtrace"
            }
        )
        self.assertEqual(hostile_options["returncode"], 96)
        self.assertIn(b"Bash option state differs", hostile_options["stderr"])
        self.assertEqual(hostile_options["events"], [])
        self.assertFalse(hostile_options["attempt_exists"])

    def test_outer_foreground_gate_precedes_every_runtime_write(self) -> None:
        source = text(OUTER)
        order = [
            "foreground static preflight'",
            "foreground direct-node capacity'",
            "foreground receipt validation'",
            'create_runtime_root "${experiment_root}/attempts"',
            "foreground receipt publication'",
            'create_runtime_root "${experiment_root}/runs"',
            'create_runtime_root "${screen_log_parent}"',
            "screen log create-only publication failed",
            '"${screen_bin}" -dmS',
        ]
        offsets = [source.index(token) for token in order]
        self.assertEqual(offsets, sorted(offsets))
        gate_end = source.index('create_runtime_root "${experiment_root}/attempts"')
        # Ignore helper definitions: the authority property is that the main
        # program does not *call* a mutating helper before both foreground
        # gates have passed.
        prefix = source[source.index("for digest"):gate_end]
        self.assertNotIn("/usr/bin/mkdir -m 0700", prefix)
        self.assertNotIn('"${screen_bin}" -dmS', prefix)
        self.assertEqual(source.count("remote-probe-base64 foreground"), 1)
        self.assertNotIn("mkdir -p", source)

    def test_outer_frames_stderr_and_isolates_screen_config(self) -> None:
        source = text(OUTER)
        self.assertIn('"$@" 2>&1 | "${base64_bin}" -w0', source)
        self.assertIn('statuses=("${PIPESTATUS[@]}")', source)
        self.assertIn("SCREENRC=/dev/null", source)
        self.assertIn("SYSSCREENRC=/dev/null SYSTEM_SCREENRC=/dev/null", source)
        self.assertIn('"${screen_bin}" -dmS "${session}" -c /dev/null -L', source)
        self.assertEqual(source.count('"${screen_bin}" -dmS'), 1)
        self.assertIn("precreated_log_is_regular_mode_0600_nlink_one", PINS.read_text())

    def test_outer_entry_and_external_authority_are_fail_closed(self) -> None:
        source = text(OUTER)
        authority = source.index("LEVEL_B_V4_EXTERNAL_LAUNCH_AUTHORITY")
        first_remote_sensitive = source.index("for digest")
        self.assertLess(authority, first_remote_sensitive)
        self.assertIn('[[ "$0" == /bin/bash ]]', source)
        self.assertIn("builtin declare -F | /usr/bin/grep . >/dev/null", source)
        self.assertIn('[[ "${PATH:-}" == /usr/bin:/bin', source)
        self.assertIn('"${BASH_ENV:-}" == /dev/null', source)
        self.assertIn('-z "${LD_PRELOAD:-}"', source)
        pins = json.loads(PINS.read_bytes())
        self.assertFalse(pins["authorization"]["launch_authorized_by_artifact"])
        entry = pins["deployment_recipe"]["outer_handoff"]
        self.assertFalse(entry["additional_environment_keys_authorized"])
        self.assertEqual(
            entry["required_interpreter_argv"],
            ["/bin/bash", "--noprofile", "--norc", "-s"],
        )
        self.assertEqual(entry["required_env_tool_argv_prefix"], ["/usr/bin/env", "-i"])
        self.assertTrue(entry["preloaded_bash_functions_forbidden"])

    def test_controller_has_independent_direct_gate_immediately_before_started(self) -> None:
        source = text(CONTROLLER)
        controller_probe = source.index("remote-probe-base64 controller")
        controller_validate = source.index('validate-base64 \\\n  "${controller_capacity_base64}"')
        started = source.index('readonly started="${attempt_root}/STARTED"')
        srun = source.index("/usr/bin/srun")
        self.assertLess(controller_probe, controller_validate)
        self.assertLess(controller_validate, started)
        self.assertLess(started, srun)
        self.assertNotIn("/usr/bin/sleep", source[controller_probe:srun])
        self.assertEqual(source.count("remote-probe-base64 controller"), 1)
        self.assertIn('[[ "${controller_capacity_challenge}" != "${foreground_capacity_challenge}" ]]', source)

    def test_step_has_fresh_local_gate_before_torch_and_retains_physical_gate(self) -> None:
        source = text(STEP)
        controller_validate = source.index("controller capacity receipt validation")
        step_probe = source.index("probe-base64 step")
        step_publish = source.index("publish-base64")
        torch_import = source.index("import os, pathlib, torch")
        torchrun = source.index("-m torch.distributed.run")
        self.assertLess(controller_validate, step_probe)
        self.assertLess(step_probe, step_publish)
        self.assertLess(step_publish, torch_import)
        self.assertLess(torch_import, torchrun)
        self.assertEqual(source.count("probe-base64 step"), 1)
        self.assertIn("free * 100 >= total * 95", source)
        self.assertIn('"${step_capacity_challenge}" != "${controller_capacity_challenge}"', source)

    def test_rank_read_only_validates_and_unsets_capacity_authority(self) -> None:
        source = text(RANK)
        self.assertIn("validate-file", source)
        self.assertNotIn("publish-base64", source)
        self.assertIn("unset LEVEL_B_STEP_CAPACITY_RECEIPT", source)
        self.assertIn("unset BASH_ENV ENV", source)
        self.assertLess(source.index("validate-file"), source.index('exec "${python_bin}"'))

    def test_controller_failure_status_and_archival_semantics(self) -> None:
        source = text(CONTROLLER)
        self.assertIn('"state":"present_invalid"', source)
        self.assertIn('"state":"present_parseable_identity"', source)
        self.assertNotIn('"state":"valid"', source)
        self.assertIn('status_parent_after_observation', source)
        self.assertIn('{"observed","query_failed"}', source)
        self.assertIn("validate-file-archival", source)
        self.assertEqual(source.count("validate-file-archival"), 1)
        archival = source.index("validate-file-archival")
        srun = source.index("/usr/bin/srun")
        self.assertGreater(archival, srun)
        self.assertIn(
            '"step_capacity_post_child_archival_revalidation_is_non_admission_evidence_only":True',
            source,
        )
        self.assertIn(
            '"post_child_archival_revalidation_admission_authority":False',
            source,
        )

    def test_terminal_rechecks_all_three_receipts_and_final_parent_gate(self) -> None:
        source = text(CONTROLLER)
        self.assertIn("def capacity_identity", source)
        self.assertIn('stat.S_IMODE(info.st_mode)==0o444', source)
        self.assertIn('"mode":stat.S_IMODE(info.st_mode)', source)
        self.assertIn(
            '"three_independent_capacity_gates_verified":len({sys.argv[12],sys.argv[15],sys.argv[18]})==3',
            source,
        )
        self.assertIn('assert out["three_independent_capacity_gates_verified"] is True', source)
        terminal = source.index('terminal_raw=')
        final_parent = source.index('terminal_parent_state=')
        success = source.index('readonly success="${attempt_root}/SUCCESS"')
        self.assertLess(final_parent, terminal)
        self.assertLess(terminal, success)
        parent_gate_start = source.index(
            'require_no_node_children "numeric child appeared before terminal seal"'
        )
        self.assertLess(parent_gate_start, final_parent)

    def test_dynamic_intent_is_not_a_frozen_content_hash(self) -> None:
        core = json.loads(CORE.read_bytes())
        intent = core["launchers"]["dynamic_intent"]
        self.assertTrue(intent["canonical_json_required"])
        self.assertTrue(intent["content_sha256_computed_at_runtime"])
        self.assertFalse(intent["fixed_content_sha256_authorized"])
        source = text(CONTROLLER)
        self.assertIn("attempt_intent_sha=", source)
        self.assertIn(
            '"foreground":{"path":sys.argv[15],"sha256":sys.argv[16],"challenge":sys.argv[17],"phase":"foreground"}',
            source,
        )
        self.assertIn(
            '"controller":{"path":sys.argv[18],"sha256":sys.argv[19],"challenge":sys.argv[20],"phase":"controller"}',
            source,
        )

    def test_release_reuse_is_the_only_active_old_tag_exception(self) -> None:
        forbidden = (
            "/attempts/fresh-world8-level-b-p2-00435-v3",
            "/runs/fresh-world8-level-b-p2-00435-v3",
            "/launchers/fresh-world8-level-b-p2-00435-v3",
            "_v3.mp4",
        )
        old_release_references = []
        for path in (BOOTSTRAP, RANK, STEP, CONTROLLER, OUTER):
            source = text(path)
            for token in forbidden:
                self.assertNotIn(token, source, path)
            for line_number, line in enumerate(source.splitlines(), 1):
                if "fresh-world8-level-b-p2-00435-v3" in line:
                    self.assertIn("release", line.lower(), (path, line_number, line))
                    old_release_references.append((path, line_number))
        self.assertEqual(
            [path for path, _ in old_release_references],
            [BOOTSTRAP, STEP, CONTROLLER],
        )
        core = json.loads(CORE.read_bytes())
        self.assertTrue(core["release"]["release_reused_from_v3"])
        self.assertTrue(core["release"]["release_bytes_unchanged"])
        self.assertEqual(core["release"]["renderer_sha256"], "8e34d976481ed81e3b8b285253878f0c02bbfbe177ea608aa51b0f4b594bf1c6")

    def test_v3_372_supersession_is_truthful(self) -> None:
        core = json.loads(CORE.read_bytes())
        self.assertFalse(core["claims"]["launch_authorized_by_artifact"])
        replacement = core["supersession"]["replacement"]
        self.assertTrue(replacement["user_requested_fresh_v4_design_and_replacement"])
        self.assertTrue(replacement["user_request_does_not_authorize_launch_without_final_review"])
        old = core["supersession"]["superseded_v3_attempt"]
        self.assertEqual(old["child_step_id"], "140846.372")
        self.assertEqual(old["first_failed_card"]["index"], 4)
        self.assertEqual(old["first_failed_card"]["free_basis_points"], 7023)
        self.assertEqual(old["external_same_user_login_process_count"], 8)
        self.assertTrue(old["gpus_released_not_claimed"])
        self.assertTrue(old["gpu_context_absence_not_claimed"])
        encoded = canonical(old)
        self.assertNotIn(b'"gpus_released":true', encoded)
        self.assertNotIn(b'"no_gpu_context":true', encoded)

    def test_static_preflight_v4_values_and_inverse_literals(self) -> None:
        source = BOOTSTRAP.read_bytes()
        self.assertEqual(source.count(b'STATIC_PREFLIGHT_STDOUT_SHA256 = "68edb1c3d5925d5ef26a2601b989f91777da3be4ad03d772aa5c2c7f1dde7998"'), 1)
        self.assertEqual(source.count(b"STATIC_PREFLIGHT_STDOUT_SIZE = 29205"), 1)
        pending = source.replace(
            b'STATIC_PREFLIGHT_STDOUT_SHA256 = "68edb1c3d5925d5ef26a2601b989f91777da3be4ad03d772aa5c2c7f1dde7998"',
            b'STATIC_PREFLIGHT_STDOUT_SHA256 = "PENDING_FINAL_STATIC_PREFLIGHT_SHA256"',
        ).replace(b"STATIC_PREFLIGHT_STDOUT_SIZE = 29205", b"STATIC_PREFLIGHT_STDOUT_SIZE = 0")
        harness_tree = ast.parse(HARNESS.read_text())
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in harness_tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"PENDING_BOOTSTRAP_SHA256", "PENDING_BOOTSTRAP_SIZE"}
        }
        self.assertEqual(len(pending), assignments["PENDING_BOOTSTRAP_SIZE"])
        self.assertEqual(hashlib.sha256(pending).hexdigest(), assignments["PENDING_BOOTSTRAP_SHA256"])

    def test_capacity_threshold_and_archival_authority_in_core(self) -> None:
        core = json.loads(CORE.read_bytes())
        cap = core["capacity"]
        self.assertEqual(cap["gpu_use_percent_required"], 0)
        self.assertEqual(cap["vram_total_bytes_required"], 68702699520)
        self.assertEqual(cap["minimum_free_basis_points"], 9500)
        self.assertEqual(cap["independent_fresh_samples"], ["foreground", "controller", "step"])
        self.assertFalse(cap["archival_revalidation"]["admission_authority"])
        self.assertTrue(cap["archival_revalidation"]["future_skew_still_enforced"])
        semantics = cap["failure_semantics"]
        self.assertIn("foreground_capacity_gate_failure", semantics)
        self.assertIn("controller_capacity_gate_failure", semantics)
        self.assertIn("post_controller_gate_pre_srun_publication_failure", semantics)
        self.assertIn("step_management_capacity_gate_failure", semantics)


if __name__ == "__main__":
    unittest.main()
