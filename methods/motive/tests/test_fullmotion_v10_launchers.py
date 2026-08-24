from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CANARY = (
    REPO_ROOT
    / "tmp"
    / "launch_fullmotion_v10_canary_20260801T154500Z.sh"
)
SMOKE = (
    REPO_ROOT
    / "tmp"
    / "launch_fullmotion_v10_smoke_20260801T154500Z.sh"
)
SUPERVISOR = REPO_ROOT / "tmp" / "launch_wait_fullmotion_v10_job116234.sh"

RUN_ID = "fullmotion128_v10_20260801T154500Z"
SNAPSHOT_NAME = "goku-full-motion128-source-v10-20260801T154500Z"
TREE_PLACEHOLDER = "__V10_TREE_SHA__"
RELEASE_PLACEHOLDER = "__V10_RELEASE_CHALLENGE__"
SMOKE_INPUT_SHA = (
    "d433acc0ccc74a9ddbff3a701a7cf86dc27e2cad24cdae0bfe7fcfc499982896"
)
FULL_INPUT_SHA = (
    "e4536937d1eb3a065907eb5f6db16b910bea75ff1ec2cdaa17c414ee943c4e42"
)


class FullMotionV10LauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canary = CANARY.read_text(encoding="utf-8")
        self.smoke = SMOKE.read_text(encoding="utf-8")
        self.supervisor = SUPERVISOR.read_text(encoding="utf-8")

    def test_launchers_have_valid_bash_syntax(self) -> None:
        for script in (CANARY, SMOKE, SUPERVISOR):
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

    def test_fixed_names_placeholders_hashes_and_job_ids(self) -> None:
        for text in (self.canary, self.smoke, self.supervisor):
            self.assertIn(RUN_ID, text)
            self.assertIn(SNAPSHOT_NAME, text)
            self.assertIn(TREE_PLACEHOLDER, text)
            self.assertNotIn("fullmotion128_v9_", text)
            self.assertNotIn("source-v9-", text)
            self.assertNotIn("20260801T142000Z", text)
            self.assertNotIn(
                "ed4a905ef008ee72347d47564ed8a719d628359ea045a4bf16770ea44e24e237",
                text,
            )
        for field, text in (
            ("tree_sha", self.canary),
            ("tree_sha", self.smoke),
            ("source_tree_sha", self.supervisor),
        ):
            self.assertRegex(
                text,
                rf"(?m)^{field}=(?:{TREE_PLACEHOLDER}|[0-9a-f]{{64}})$",
            )
        self.assertIn("job_id=113864", self.canary)
        self.assertIn("job_id=116234", self.smoke)
        self.assertIn("job_id=116234", self.supervisor)
        self.assertIn(f"input_sha={SMOKE_INPUT_SHA}", self.canary)
        self.assertIn(f"smoke_input_sha={SMOKE_INPUT_SHA}", self.smoke)
        self.assertIn(f"smoke_input_sha={SMOKE_INPUT_SHA}", self.supervisor)
        for text in (self.canary, self.smoke, self.supervisor):
            self.assertIn("prepare_smoke8_uniform/candidates.jsonl", text)
        self.assertIn(f"full_input_sha={FULL_INPUT_SHA}", self.supervisor)
        self.assertIn("release_id=goku-full-motion128-v10-release", self.supervisor)
        for field in ("release_challenge", "claim_owner_token"):
            self.assertRegex(
                self.supervisor,
                rf"(?m)^{field}=(?:{RELEASE_PLACEHOLDER}|[0-9a-f]{{64}})$",
            )
        self.assertNotIn(
            "d181358590b9cbdaa5de9968e1ba3bc798dc983bcdca79ef28768f8c3bcae591",
            self.supervisor,
        )

    def test_placeholders_are_explicitly_fail_closed(self) -> None:
        for text in (self.canary, self.smoke):
            self.assertIn(
                '[[ "${tree_sha}" != __V10_TREE_SHA__ ]]', text
            )
            self.assertIn("replace __V10_TREE_SHA__", text)
        self.assertIn(
            '[[ "${source_tree_sha}" != __V10_TREE_SHA__ ]]',
            self.supervisor,
        )
        self.assertIn(
            '"${release_challenge}" != __V10_RELEASE_CHALLENGE__',
            self.supervisor,
        )
        self.assertIn(
            '"${claim_owner_token}" != __V10_RELEASE_CHALLENGE__',
            self.supervisor,
        )

    def test_canary_and_smoke_idle_probes_overlap_holder_memory(self) -> None:
        for name, text in (("canary", self.canary), ("smoke", self.smoke)):
            function = text.split("check_idle_node() {", 1)[1].split(
                "\n}\n", 1
            )[0]
            with self.subTest(name=name):
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

    def test_v5_gate_is_sealed_before_supervisor_claim(self) -> None:
        for text in (self.smoke, self.supervisor):
            self.assertIn(
                "motive-goku-full-motion-qwen-smoke-gate-v5", text
            )
            self.assertNotIn(
                "motive-goku-full-motion-qwen-smoke-gate-v4", text
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
            "strict v10 smoke-gate preflight failed"
        )
        claim_publication = self.supervisor.index(
            'atomic_publish_lines "${holder_claim}"'
        )
        self.assertLess(gate_validation, claim_publication)

    def test_supervisor_parent_child_launch_handshake_is_fail_closed(self) -> None:
        for marker in (
            "pipeline_v10_supervisor.pid",
            "pipeline_v10_launch_cancel_request.txt",
            "pipeline_v10_launch_token.txt",
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
            launch_block.index("exec nohup env"),
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
job_id=116234
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
