from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
WATCHER = REPO_ROOT / "tmp" / "launch_fullmotion_v16_wan_stream.sh"


class FullMotionV16WanStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = WATCHER.read_text(encoding="utf-8")

    def test_bash_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(WATCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_embedded_python_helpers_compile(self) -> None:
        lines = self.text.splitlines()
        blocks: list[str] = []
        index = 0
        while index < len(lines):
            if "<<'PY'" not in lines[index]:
                index += 1
                continue
            end = index + 1
            while end < len(lines) and lines[end] != "PY":
                end += 1
            self.assertLess(end, len(lines), f"unclosed heredoc at {index + 1}")
            blocks.append("\n".join(lines[index + 1 : end]) + "\n")
            index = end + 1
        self.assertEqual(len(blocks), 13)
        for number, source in enumerate(blocks, 1):
            with self.subTest(block=number):
                compile(source, f"wan-stream-helper-{number}", "exec")

    def test_fixed_holder_snapshot_and_reserved_node_geometry(self) -> None:
        self.assertIn("job_id=${MOTIVE_FULL_MOTION_JOB_ID:-118150}", self.text)
        self.assertIn("expected_job_name=${MOTIVE_FULL_MOTION_JOB_NAME:-fm128-v14-g8}", self.text)
        self.assertIn("NumNodes=8", self.text)
        self.assertIn("gres/gpu:mi210=64", self.text)
        self.assertIn("qwen_nodes=(", self.text)
        self.assertIn('"${allocated_nodes[0]}" "${allocated_nodes[1]}"', self.text)
        self.assertIn('"${allocated_nodes[2]}" "${allocated_nodes[3]}"', self.text)
        self.assertIn("wan_nodes=(", self.text)
        self.assertIn('"${allocated_nodes[4]}" "${allocated_nodes[5]}"', self.text)
        self.assertIn('"${allocated_nodes[6]}" "${allocated_nodes[7]}"', self.text)
        self.assertIn('"max_concurrent": 4', self.text)
        self.assertIn("action_source_snapshot.py", self.text)
        self.assertIn("--expected-tree-sha256", self.text)

    def test_each_pass_is_streamed_without_a_batch_gate(self) -> None:
        self.assertIn("qwen_root=${MOTIVE_FULL_MOTION_QWEN_ROOT", self.text)
        self.assertIn("terminal/${iid}.receipt.json", self.text)
        self.assertIn("passed", self.text)
        self.assertIn("inspect_qwen_terminal", self.text)
        self.assertIn("preflight_fragment", self.text)
        self.assertIn("load_non_production_preview_manifest", self.text)
        self.assertIn("queue+=(\"${iid}\")", self.text)
        self.assertIn("run_wan_sample \"${iid}\"", self.text)
        self.assertIn("pass dispatched independently", self.text)
        for forbidden in (
            "minimum-hard-passes",
            "minimum_hard_passes",
            "goku_full_motion_smoke_gate",
            "3/8",
            "all rows pass",
        ):
            self.assertNotIn(forbidden, self.text)

    def test_input_closure_accepts_any_nonempty_unique_subset(self) -> None:
        self.assertIn("MOTIVE_FULL_MOTION_QWEN_INPUT", self.text)
        self.assertIn("MOTIVE_FULL_MOTION_QWEN_INPUT_SHA", self.text)
        self.assertIn("if not iids or len(set(iids)) != len(iids):", self.text)
        self.assertIn("(( ${#expected_iids[@]} >= 1 ))", self.text)
        self.assertIn("expected_iid_count=${#expected_iids[@]}", self.text)
        self.assertIn('"expected_terminal_rows": int(expected_count)', self.text)
        self.assertNotIn("len(iids) != 8", self.text)
        self.assertNotIn("${#expected_iids[@]} == 8", self.text)

    def test_input_closure_helper_runs_for_one_and_many_rows(self) -> None:
        lines = self.text.splitlines()
        blocks: list[str] = []
        index = 0
        while index < len(lines):
            if "<<'PY'" not in lines[index]:
                index += 1
                continue
            end = index + 1
            while end < len(lines) and lines[end] != "PY":
                end += 1
            blocks.append("\n".join(lines[index + 1 : end]) + "\n")
            index = end + 1
        closure = next(
            block
            for block in blocks
            if "v16 Wan stream requires one or more unique input IIDs" in block
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for count in (1, 3, 17):
                path = root / f"subset_{count}.jsonl"
                path.write_text(
                    "".join(
                        json.dumps({"iid": f"retry-{index:03d}"}) + "\n"
                        for index in range(count)
                    ),
                    encoding="utf-8",
                )
                result = subprocess.run(
                    [sys.executable, "-c", closure, str(path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    result.stdout.splitlines(),
                    [f"retry-{index:03d}" for index in range(count)],
                )

            duplicate = root / "duplicate.jsonl"
            duplicate.write_text('{"iid":"same"}\n{"iid":"same"}\n')
            rejected = subprocess.run(
                [sys.executable, "-c", closure, str(duplicate)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)

    def test_custom_subset_requires_an_independent_create_only_wan_root(self) -> None:
        self.assertIn("MOTIVE_FULL_MOTION_WAN_ROOT", self.text)
        self.assertIn("custom retry/subset input requires an independent", self.text)
        self.assertIn("Qwen and Wan roots must be independent non-nested", self.text)
        self.assertIn("create-only Wan stream root exists", self.text)

    def test_optional_expansion_is_default_off_and_strictly_gated(self) -> None:
        self.assertIn(
            "MOTIVE_FULL_MOTION_WAN_EXPAND_AFTER_QWEN_TERMINAL:-0",
            self.text,
        )
        self.assertIn('if expand_after_terminal == "1":', self.text)
        self.assertIn('"expand_after_qwen_terminal": True', self.text)
        self.assertIn('dispatch_nodes=("${wan_nodes[@]}")', self.text)
        self.assertIn("qwen_step_nodes_are_clear", self.text)
        self.assertIn("expansion_idle_audit", self.text)
        self.assertIn('for node in "${qwen_nodes[@]}"; do', self.text)
        self.assertIn('dispatch_nodes=("${wan_nodes[@]}" "${qwen_nodes[@]}")', self.text)
        self.assertIn("pool_expansion.json", self.text)
        self.assertIn("all_qwen_terminal_steps_exited_double_gpu_idle", self.text)
        self.assertIn("(expand_after_qwen_terminal == 0 || expanded == 1)", self.text)
        gate = self.text.index("if (( all_qwen_terminal == 1")
        call = self.text.index("if ! try_expand_dispatch_pool", gate)
        dispatch = self.text.index(
            'for node in "${dispatch_nodes[@]}"; do', call
        )
        self.assertLess(gate, call)
        self.assertLess(call, dispatch)

    def test_expansion_terminal_evidence_helper_executes_and_fails_closed(self) -> None:
        lines = self.text.splitlines()
        blocks: list[str] = []
        index = 0
        while index < len(lines):
            if "<<'PY'" not in lines[index]:
                index += 1
                continue
            end = index + 1
            while end < len(lines) and lines[end] != "PY":
                end += 1
            blocks.append("\n".join(lines[index + 1 : end]) + "\n")
            index = end + 1
        helper = next(
            block
            for block in blocks
            if "motive-full-motion-v16-expansion-terminal-evidence-v1" in block
        )

        def canonical(value: object) -> bytes:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_sha = "c" * 64
            iids = ["retry-001", "retry-002", "retry-003"]
            manifest = root / "input.jsonl"
            manifest.write_text(
                "".join(json.dumps({"iid": iid}) + "\n" for iid in iids),
                encoding="utf-8",
            )
            manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            for number, iid in enumerate(iids):
                observation = {
                    "schema_version": "motive-full-motion-v16-qwen-observation-v1",
                    "iid": iid,
                    "status": "ok" if number != 1 else "error",
                    "qwen_terminal_receipt": f"/qwen/terminal/{iid}.json",
                    "qwen_receipt_digest": "d" * 64,
                    "passed_fragment": (
                        f"/qwen/passed/{iid}.jsonl" if number != 1 else None
                    ),
                    "watch_contract_sha256": contract_sha,
                }
                observation["observation_digest"] = hashlib.sha256(
                    canonical(observation)
                ).hexdigest()
                (root / f"{iid}.json").write_text(
                    json.dumps(observation, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    helper,
                    str(root),
                    str(manifest),
                    manifest_sha,
                    contract_sha,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            evidence = json.loads(result.stdout)
            self.assertEqual(evidence["expected_iids"], iids)
            self.assertEqual(evidence["terminal_count"], 3)
            self.assertEqual(
                evidence["schema_version"],
                "motive-full-motion-v16-expansion-terminal-evidence-v1",
            )

            (root / f"{iids[-1]}.json").unlink()
            missing = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    helper,
                    str(root),
                    str(manifest),
                    manifest_sha,
                    contract_sha,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(missing.returncode, 0)

    def test_expansion_receipt_streams_payload_larger_than_arg_max(self) -> None:
        lines = self.text.splitlines()
        blocks: list[str] = []
        index = 0
        while index < len(lines):
            if "<<'PY'" not in lines[index]:
                index += 1
                continue
            end = index + 1
            while end < len(lines) and lines[end] != "PY":
                end += 1
            blocks.append("\n".join(lines[index + 1 : end]) + "\n")
            index = end + 1
        publisher = next(
            block
            for block in blocks
            if "motive-full-motion-v16-wan-pool-expansion-v1" in block
        )

        def canonical(value: object) -> bytes:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()

        contract_sha = "c" * 64
        argument_limit = int(os.sysconf("SC_ARG_MAX"))
        count = max(1000, argument_limit // 160)
        while True:
            iids = [f"large-{number:06d}" for number in range(count)]
            records = [
                {
                    "iid": iid,
                    "status": "ok",
                    "observation_sha256": "a" * 64,
                    "qwen_receipt_digest": "b" * 64,
                }
                for iid in iids
            ]
            evidence = {
                "schema_version": (
                    "motive-full-motion-v16-expansion-terminal-evidence-v1"
                ),
                "expected_iids": iids,
                "terminal_count": count,
                "records": records,
                "watch_contract_sha256": contract_sha,
            }
            evidence["evidence_digest"] = hashlib.sha256(
                canonical(evidence)
            ).hexdigest()
            evidence_bytes = canonical(evidence) + b"\n"
            if len(evidence_bytes) > argument_limit + 65536:
                break
            count *= 2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper_path = root / "publisher.py"
            evidence_path = root / "evidence.json"
            target = root / "pool_expansion.json"
            helper_path.write_text(publisher, encoding="utf-8")
            evidence_path.write_bytes(evidence_bytes)
            args = [
                sys.executable,
                str(helper_path),
                str(target),
                contract_sha,
                "w0,w1,w2,w3",
                "q0,q1,q2,q3",
                "w0,w1,w2,w3,q0,q1,q2,q3",
            ]

            def publish() -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [
                        "bash",
                        "-c",
                        'evidence=$1; shift; exec "$@" 3<"${evidence}"',
                        "bash",
                        str(evidence_path),
                        *args,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )

            created = publish()
            self.assertEqual(created.returncode, 0, created.stderr)
            self.assertRegex(created.stdout.strip(), r"^[0-9a-f]{64}$")
            receipt_bytes = target.read_bytes()
            receipt = json.loads(receipt_bytes)
            self.assertEqual(
                receipt["terminal_evidence"]["terminal_count"], count
            )
            self.assertGreater(len(evidence_bytes), argument_limit)

            # Same evidence is resume-idempotent and preserves exact bytes.
            resumed = publish()
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(target.read_bytes(), receipt_bytes)
            self.assertEqual(resumed.stdout, created.stdout)

            # Changed evidence cannot overwrite the create-only receipt.
            evidence["records"][0]["status"] = "error"
            evidence.pop("evidence_digest")
            evidence["evidence_digest"] = hashlib.sha256(
                canonical(evidence)
            ).hexdigest()
            evidence_path.write_bytes(canonical(evidence) + b"\n")
            changed = publish()
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn("existing pool expansion receipt differs", changed.stderr)
            self.assertEqual(target.read_bytes(), receipt_bytes)

    def test_contract_helper_preserves_default_and_binds_enabled_expansion(self) -> None:
        lines = self.text.splitlines()
        blocks: list[str] = []
        index = 0
        while index < len(lines):
            if "<<'PY'" not in lines[index]:
                index += 1
                continue
            end = index + 1
            while end < len(lines) and lines[end] != "PY":
                end += 1
            blocks.append("\n".join(lines[index + 1 : end]) + "\n")
            index = end + 1
        helper = next(block for block in blocks if "wan_contract = {" in block)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contracts = {}
            for enabled in ("0", "1"):
                path = root / f"contract_{enabled}.json"
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        helper,
                        str(path),
                        "/run",
                        "/snapshot",
                        "a" * 64,
                        "118150",
                        "fm128-v14-g8",
                        "/input.jsonl",
                        "b" * 64,
                        "/qwen",
                        "motive.goku_full_motion_qwen_v16",
                        "q0,q1,q2,q3",
                        "w0,w1,w2,w3",
                        "/python",
                        "/wan-code",
                        "/checkpoint",
                        "/compat",
                        "/ffprobe",
                        "2",
                        "0",
                        "17",
                        enabled,
                        "w0,w1,w2,w3,q0,q1,q2,q3",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                contracts[enabled] = json.loads(path.read_text())

            self.assertEqual(
                contracts["0"]["qwen"]["terminal_adapter_module"],
                "motive.goku_full_motion_qwen_v16",
            )

            disabled_wan = contracts["0"]["wan"]
            self.assertEqual(disabled_wan["nodes"], ["w0", "w1", "w2", "w3"])
            self.assertEqual(disabled_wan["max_concurrent"], 4)
            self.assertNotIn("expand_after_qwen_terminal", disabled_wan)
            enabled_wan = contracts["1"]["wan"]
            self.assertTrue(enabled_wan["expand_after_qwen_terminal"])
            self.assertEqual(enabled_wan["max_concurrent_after_expansion"], 8)
            self.assertEqual(
                enabled_wan["expanded_nodes"],
                ["w0", "w1", "w2", "w3", "q0", "q1", "q2", "q3"],
            )

    def test_expansion_step_gate_executes_for_clear_busy_and_foreign_steps(self) -> None:
        start = self.text.index("qwen_step_nodes_are_clear() {")
        end = self.text.index("\n}\n\ncheck_idle_node()", start) + 2
        function = self.text[start:end]

        def run_case(step_nodes: dict[str, str]) -> subprocess.CompletedProcess[str]:
            step_lines = ["118150.batch", "118150.extern", *step_nodes]
            squeue_body = " ".join(f"'{value}'" for value in step_lines)
            step_cases = "\n".join(
                f"    {step}) printf '%s\\n' 'StepId={step} Nodes={node}' ;;"
                for step, node in step_nodes.items()
            )
            harness = f"""
set -Eeuo pipefail
job_id=118150
qwen_nodes=(q0 q1 q2 q3)
wan_nodes=(w0 w1 w2 w3)
declare -A node_pids=()
node_pids[w0]=$BASHPID
fail() {{ printf '%s\\n' "$*" >&2; exit 2; }}
in_list() {{
  local needle=$1; shift
  local item
  for item in "$@"; do [[ "$item" == "$needle" ]] && return 0; done
  return 1
}}
squeue() {{ printf '%s\\n' {squeue_body}; }}
scontrol() {{
  if [[ "$1" == show && "$2" == hostnames ]]; then
    printf '%s\\n' "$3"
    return 0
  fi
  local target="${{@: -1}}"
  case "$target" in
{step_cases}
    *) return 2 ;;
  esac
}}
{function}
qwen_step_nodes_are_clear
"""
            return subprocess.run(
                ["bash", "-c", harness],
                check=False,
                capture_output=True,
                text=True,
            )

        clear = run_case({})
        self.assertEqual(clear.returncode, 0, clear.stderr)
        qwen_busy = run_case({"118150.0": "q0"})
        self.assertEqual(qwen_busy.returncode, 1, qwen_busy.stderr)
        own_wan = run_case({"118150.1": "w0"})
        self.assertEqual(own_wan.returncode, 0, own_wan.stderr)
        foreign_wan = run_case({"118150.2": "w1"})
        self.assertEqual(foreign_wan.returncode, 2)
        self.assertIn("unexpected step during expansion audit", foreign_wan.stderr)

    def test_wan_command_matches_working_v3_preview_runtime(self) -> None:
        required = (
            "/anaconda3/envs/vace/bin/python3.12",
            "wan22-vace-rocm63/lib/python3.12/site-packages",
            "transformers.__version__ == \"4.51.3\"",
            "tokenizers.__version__ == \"0.21.4\"",
            "torch.__version__ == \"2.7.1+rocm6.3\"",
            "numpy.__version__ == \"1.26.4\"",
            "ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
            "HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
            "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
            "NCCL_IB_DISABLE=1",
            "NCCL_SOCKET_IFNAME=bond0",
            "GLOO_SOCKET_IFNAME=bond0",
            "--nnodes=1 --nproc_per_node=8",
            "-m motive.wan22_i2v_batch",
            "--non-production-preview",
            "--max-new-samples 1",
            "--size '1280*720' --frame-num 81",
            "--sample-steps 40 --sample-shift 5.0 --sample-solver unipc",
            "--sample-guide-scale-low 3.5 --sample-guide-scale-high 3.5",
            "--base-seed 260730",
            "--expected-world-size 8 --require-rocm",
            "--expected-gpu-name-substring MI210",
            "--video-codec libx264 --video-quality 8",
        )
        for item in required:
            self.assertIn(item, self.text)
        self.assertNotIn("--signed-release", self.text)

    def test_every_dispatch_is_gpu_idle_audited_and_step_scoped(self) -> None:
        self.assertIn("assert_expected_steps", self.text)
        self.assertIn('"${job_id}.batch"', self.text)
        self.assertIn('"${job_id}.extern"', self.text)
        self.assertIn("scontrol show step -o", self.text)
        self.assertIn("unexpected active Slurm step", self.text)
        self.assertIn("double_idle_audit \"${node}\"", self.text)
        self.assertIn("for audit in 1 2", self.text)
        self.assertIn("rocm-smi --showuse --showmemuse", self.text)
        self.assertIn("rocm-smi --showpids --csv", self.text)
        self.assertIn("seen==8", self.text)
        self.assertIn("1073741824", self.text)
        self.assertIn("--gpus-per-task=8", self.text)
        self.assertNotIn("scancel", self.text)

    def test_create_only_atomic_receipts_and_resume_validation(self) -> None:
        self.assertIn("MOTIVE_FULL_MOTION_WAN_RESUME", self.text)
        self.assertIn("create-only Wan stream root exists", self.text)
        self.assertIn("another v16 Wan watcher is active", self.text)
        self.assertIn("flock -n", self.text)
        self.assertIn("os.link(temporary, target)", self.text)
        self.assertIn("motive-full-motion-v16-wan-dispatch-claim-v1", self.text)
        self.assertIn("motive-full-motion-v16-wan-dispatch-status-v1", self.text)
        self.assertIn("motive-full-motion-v16-wan-stream-terminal-v1", self.text)
        self.assertIn("attempt_state", self.text)
        self.assertIn("claim fragment binding differs", self.text)
        self.assertIn("status log binding differs", self.text)
        self.assertIn("successful completion binding differs", self.text)
        self.assertIn("trap '' HUP", self.text)
        self.assertIn('exec >"${watch_log}" 2>&1 </dev/null', self.text)
        self.assertIn("disown", self.text)

    def test_qwen_errors_are_terminal_but_do_not_block_other_passes(self) -> None:
        self.assertIn('[[ "${status}" == ok || "${status}" == error ]]', self.text)
        self.assertIn("qwen_terminal", self.text)
        self.assertIn("if [[ \"${status}\" == ok ]]", self.text)
        self.assertIn("qwen_error_iids", self.text)
        self.assertIn("wan_success_iids", self.text)
        self.assertIn("wan_error_iids", self.text)
        self.assertIn('"expected_terminal_rows": int(expected_count)', self.text)

    def test_unbound_snapshot_fails_before_cluster_access(self) -> None:
        result = subprocess.run(
            ["/bin/bash", str(WATCHER)],
            check=False,
            capture_output=True,
            text=True,
            env={
                "HOME": "/tmp",
                "PATH": "/usr/bin:/bin",
                "LC_ALL": "C",
                "LANG": "C",
                "MOTIVE_FULL_MOTION_TREE_SHA": "",
            },
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("bind MOTIVE_FULL_MOTION_TREE_SHA", result.stderr)
        self.assertNotIn("scontrol", result.stderr)


if __name__ == "__main__":
    unittest.main()
