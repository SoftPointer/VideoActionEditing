from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "auh_full_motion_finalize_release_watcher.sh"
)
NODES = [f"auh{index:03d}" for index in range(1, 9)]
QWEN_NODES = NODES[:4]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object_sha(value: object) -> str:
    return _sha(_canonical(value))


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


FAKE_PYTHON = r'''#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import sys


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_bytes(canonical(value) + b"\n")


def metadata(raw, rows):
    return {"sha256": sha(raw), "bytes": len(raw), "rows": rows}


args = sys.argv[1:]
with Path(os.environ["FAKE_PYTHON_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write(" ".join(args[:3]) + "\n")

if args and args[0] == os.environ["MOTIVE_FULL_MOTION_SNAPSHOT_TOOL"]:
    if args[1:2] != ["verify"]:
        raise SystemExit(81)
    raise SystemExit(0)

if args[:2] == ["-m", "motive.goku_full_motion_finalize"]:
    def after(flag):
        return args[args.index(flag) + 1]
    output = Path(after("--output-dir"))
    output.mkdir()
    primary = b"".join(
        canonical({"iid": f"iid-{index:03d}"}) + b"\n"
        for index in range(256)
    )
    reserve = b"".join(
        canonical({"iid": f"reserve-{index:03d}"}) + b"\n"
        for index in range(64)
    )
    review = b""
    summary = canonical({"status": "complete"}) + b"\n"
    raw_outputs = {
        "primary_256.jsonl": (primary, 256),
        "reserve_64.jsonl": (reserve, 64),
        "review_candidates.jsonl": (review, 0),
        "summary.json": (summary, 1),
    }
    for name, (raw, _rows) in raw_outputs.items():
        (output / name).write_bytes(raw)
    candidate = Path(after("--candidate-manifest")).resolve()
    qwen = Path(after("--qwen-dir")).resolve()
    artifacts = {
        name: metadata(raw, rows)
        for name, (raw, rows) in raw_outputs.items()
    }
    inputs = {
        "candidate_manifest": {
            "path": str(candidate),
            "sha256": sha(candidate.read_bytes()),
            "bytes": candidate.stat().st_size,
            "rows": 768,
        },
        "qwen_shards": [
            {
                "output_path": str(qwen / f"qwen_shard_{index:03d}.jsonl"),
                "receipt_path": str(
                    qwen / f"qwen_shard_{index:03d}.receipt.json"
                ),
            }
            for index in range(8)
        ],
    }
    done = {
        "schema_version": "motive-goku-full-motion-finalize-done-v1",
        "status": "complete",
        "inputs": inputs,
        "artifacts": artifacts,
    }
    done["done_digest"] = sha(canonical(done))
    write_json(output / "done.json", done)
    raise SystemExit(0)

if args[:2] == ["-m", "motive.goku_full_motion_shard_manifest"]:
    def after(flag):
        return args[args.index(flag) + 1]
    final = Path(after("--finalizer-dir"))
    output = Path(after("--output-dir"))
    leaf = output / "shards"
    leaf.mkdir(parents=True)
    lines = (final / "primary_256.jsonl").read_bytes().splitlines(keepends=True)
    artifacts = {}
    job_rows = []
    job_columns = [
        "shard_index", "shard_id", "manifest_relpath",
        "root_row_start_zero_based", "root_row_end_exclusive", "row_count",
        "manifest_sha256", "manifest_bytes", "ordered_iids_sha256",
        "ordered_row_sha256", "ordered_iids_json",
    ]
    for index in range(32):
        name = f"shard_{index:03d}.jsonl"
        raw = b"".join(lines[index * 8:(index + 1) * 8])
        (leaf / name).write_bytes(raw)
        artifacts[f"shards/{name}"] = metadata(raw, 8)
        rows = [json.loads(line) for line in raw.splitlines()]
        iids = [row["iid"] for row in rows]
        ordered_iids = sha(b"".join(iid.encode() + b"\n" for iid in iids))
        row_digests = [sha(canonical(row)) for row in rows]
        ordered_rows = sha(
            b"".join(value.encode() + b"\n" for value in row_digests)
        )
        job_rows.append([
            str(index), f"shard_{index:03d}", f"shards/{name}",
            str(index * 8), str((index + 1) * 8), "8", sha(raw), str(len(raw)),
            ordered_iids, ordered_rows,
            json.dumps(iids, ensure_ascii=False, separators=(",", ":")),
        ])
    if os.environ.get("FAKE_BAD_JOBS") == "1":
        jobs = ("\t".join(job_columns) + "\n").encode()
    else:
        jobs = (
            "\n".join(
                ["\t".join(job_columns), *("\t".join(row) for row in job_rows)]
            )
            + "\n"
        ).encode()
    summary = canonical({"status": "complete"}) + b"\n"
    (output / "jobs.tsv").write_bytes(jobs)
    (output / "summary.json").write_bytes(summary)
    artifacts["jobs.tsv"] = metadata(jobs, 32)
    artifacts["summary.json"] = metadata(summary, 1)
    done = {
        "schema_version": "motive-goku-full-motion-shard-manifest-done-v1",
        "status": "complete",
        "artifacts": artifacts,
    }
    done["done_digest"] = sha(canonical(done))
    write_json(output / "done.json", done)
    raise SystemExit(0)

if args[:3] == ["-m", "motive.wan22_full_motion_signed_release", "prepare"]:
    def after(flag):
        return args[args.index(flag) + 1]
    primary = Path(after("--root-manifest"))
    raw = primary.read_bytes()
    signed = {
        "schema_version": "motive-wan22-full-motion-root-release-payload-v3",
        "release_id": after("--release-id"),
        "issued_at_utc": after("--issued-at-utc"),
        "root_manifest": {
            "sha256": sha(raw),
            "bytes": len(raw),
            "rows": 256,
            "contiguous_shard_rows": 8,
        },
    }
    request = {
        "schema_version": "motive-wan22-full-motion-release-request-v3",
        "challenge_sha256": after("--challenge"),
        "builder": {"fixture": "frozen"},
        "signed": signed,
    }
    request["request_digest"] = sha(canonical(request))
    output = Path(after("--request"))
    write_json(output, request)
    output.chmod(0o400)
    raise SystemExit(0)

os.execv(sys.executable, [sys.executable, *args])
'''


FAKE_SRUN = r'''#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

args = sys.argv[1:]
with Path(os.environ["FAKE_SRUN_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write(" ".join(args) + "\n")
if "--overlap" not in args or "--mem=0" not in args or "--exact" not in args:
    raise SystemExit(90)
if "--exclusive" in args or any(value.startswith("--kill-on-bad-exit") for value in args):
    raise SystemExit(91)
if f"--nodelist={os.environ['MOTIVE_FULL_MOTION_FINALIZE_NODE']}" not in args:
    raise SystemExit(92)
if os.environ.get("FAKE_BLOCK_SRUN") == "1":
    Path(os.environ["FAKE_BLOCK_MARKER"]).write_text("active\n")
    while True:
        import time
        time.sleep(1)
index = args.index("env")
command = ["/usr/bin/env", *args[index + 1:]]
raise SystemExit(subprocess.run(command, env=os.environ, check=False).returncode)
'''


FAKE_SLEEP = r'''#!/usr/bin/env python3
import hashlib
import os
from pathlib import Path

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

done = Path(os.environ["MOTIVE_FULL_MOTION_FULL_QWEN_DONE"])
if not done.exists():
    root = Path(os.environ["MOTIVE_FULL_MOTION_FULL_QWEN_ROOT"])
    root.mkdir(exist_ok=True)
    for index in range(8):
        (root / f"qwen_shard_{index:03d}.jsonl").write_text("{}\n")
        (root / f"qwen_shard_{index:03d}.receipt.json").write_text("{}\n")
    input_path = Path(os.environ["MOTIVE_FULL_MOTION_FULL_INPUT"])
    lines = [
        "schema=motive-goku-full-motion-qwen-controller-v1",
        "status=complete",
        "input=" + str(input_path),
        "input_sha256=" + sha(input_path.read_bytes()),
        "output_root=" + str(root),
        "slurm_job_id=" + os.environ["MOTIVE_EXISTING_SLURM_JOB_ID"],
    ]
    if os.environ.get("FAKE_OLD_QWEN_RECEIPT") != "1":
        lines.extend([
            "nodes=" + os.environ["MOTIVE_FULL_MOTION_QWEN_NODES"],
            "completed_at_utc=2026-08-01T12:34:56Z",
        ])
    done.write_text("\n".join(lines) + "\n")
'''


class FullMotionFinalizeReleaseWatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run_root = self.root / "run"
        self.run_root.mkdir()
        self.production = self.run_root / "production"
        self.release_dir = self.production / "release"
        self.release_dir.mkdir(parents=True)

        self.snapshot = self.root / "snapshot"
        motive = self.snapshot / "methods" / "motive" / "motive"
        motive.mkdir(parents=True)
        (self.snapshot / "SOURCE_FILES.jsonl").write_text("{}\n")
        for name in (
            "goku_full_motion_finalize.py",
            "goku_full_motion_shard_manifest.py",
            "wan22_full_motion_signed_release.py",
        ):
            path = motive / name
            path.write_text("# frozen fixture\n")
            path.chmod(0o444)
        self.snapshot_tool = self.snapshot / "verify_snapshot.py"
        self.snapshot_tool.write_text("# frozen verifier fixture\n")
        self.snapshot_tool.chmod(0o444)

        self.full_input = self.run_root / "full_input.jsonl"
        self.full_input.write_bytes(b"{}\n" * 768)
        self.qwen_root = self.run_root / "qwen"
        self.qwen_done = self.run_root / "qwen.done"
        self.final_pool = self.run_root / "final_pool"
        self.shard_root = self.production / "generation_shards"
        self.request = self.release_dir / "release_request.json"
        self.ready = self.run_root / "watcher.ready.json"
        self.terminal = self.run_root / "watcher.done.json"

        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        self.fake_python = self.root / "python"
        _write_executable(self.fake_python, FAKE_PYTHON)
        self.python_log = self.root / "python.log"
        self.python_log.write_text("")
        self.srun_log = self.root / "srun.log"
        self.srun_log.write_text("")
        _write_executable(self.fake_bin / "srun", FAKE_SRUN)
        _write_executable(self.fake_bin / "sleep", FAKE_SLEEP)
        _write_executable(
            self.fake_bin / "squeue",
            "#!/usr/bin/env bash\nprintf 'auh[001-008]\\n'\n",
        )
        _write_executable(
            self.fake_bin / "scontrol",
            """#!/usr/bin/env bash
if [[ "${1:-}" == show && "${2:-}" == job ]]; then
  printf 'JobId=123 UserId=%s(1000) JobState=RUNNING NumNodes=8 NodeList=auh[001-008] gres/gpu:mi210=64\\n' "$(id -un)"
elif [[ "${1:-}" == show && "${2:-}" == hostnames ]]; then
  printf 'auh001\\nauh002\\nauh003\\nauh004\\nauh005\\nauh006\\nauh007\\nauh008\\n'
else
  exit 2
fi
""",
        )

    def _environment(self, **updates: str) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update(
            {
                "PATH": f"{self.fake_bin}:{environment['PATH']}",
                "FAKE_PYTHON_LOG": str(self.python_log),
                "FAKE_SRUN_LOG": str(self.srun_log),
                "MOTIVE_EXISTING_SLURM_JOB_ID": "123",
                "MOTIVE_FULL_MOTION_PIPELINE_NODES": ",".join(NODES),
                "MOTIVE_FULL_MOTION_QWEN_NODES": ",".join(QWEN_NODES),
                "MOTIVE_FULL_MOTION_FINALIZE_NODE": NODES[0],
                "MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT": str(self.snapshot),
                "MOTIVE_FULL_MOTION_SOURCE_TREE_SHA256": "1" * 64,
                "MOTIVE_FULL_MOTION_SNAPSHOT_TOOL": str(self.snapshot_tool),
                "MOTIVE_FULL_MOTION_FULL_INPUT": str(self.full_input),
                "MOTIVE_FULL_MOTION_FULL_INPUT_SHA256": _sha(
                    self.full_input.read_bytes()
                ),
                "MOTIVE_FULL_MOTION_FULL_QWEN_ROOT": str(self.qwen_root),
                "MOTIVE_FULL_MOTION_FULL_QWEN_DONE": str(self.qwen_done),
                "MOTIVE_FULL_MOTION_FINAL_POOL": str(self.final_pool),
                "MOTIVE_FULL_MOTION_PRODUCTION_ROOT": str(self.production),
                "MOTIVE_FULL_MOTION_SHARD_MANIFEST_DIR": str(self.shard_root),
                "MOTIVE_FULL_MOTION_RELEASE_DIR": str(self.release_dir),
                "MOTIVE_FULL_MOTION_RELEASE_REQUEST": str(self.request),
                "MOTIVE_FULL_MOTION_RELEASE_ID": "fixture-release-v1",
                "MOTIVE_FULL_MOTION_RELEASE_CHALLENGE": "a" * 64,
                "MOTIVE_FULL_MOTION_QWEN_PYTHON": str(self.fake_python),
                "MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WAIT_SECONDS": "5",
                "MOTIVE_FULL_MOTION_FINALIZE_RELEASE_POLL_SECONDS": "1",
                "MOTIVE_FULL_MOTION_FINALIZE_CPUS": "4",
                "MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WATCHER_READY": str(
                    self.ready
                ),
                "MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WATCHER_RECEIPT": str(
                    self.terminal
                ),
                "LC_ALL": "C",
            }
        )
        environment.update(updates)
        return environment

    def _run(self, **updates: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT)],
            env=self._environment(**updates),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )

    def _assert_receipt_digest(self, path: Path) -> dict[str, object]:
        value = json.loads(path.read_text(encoding="utf-8"))
        stored = value.pop("receipt_digest")
        self.assertEqual(stored, _object_sha(value))
        return value

    def test_success_publishes_ready_and_terminal_after_three_safe_steps(self) -> None:
        completed = self._run()
        self.assertEqual(completed.returncode, 0, completed.stderr)

        ready = self._assert_receipt_digest(self.ready)
        self.assertEqual(
            ready,
            {
                "schema_version": "motive-goku-full-motion-finalize-release-watcher-ready-v1",
                "status": "ready",
                "slurm_job_id": "123",
                "nodes": NODES,
                "qwen_nodes": QWEN_NODES,
                "finalize_node": NODES[0],
            },
        )
        terminal = self._assert_receipt_digest(self.terminal)
        self.assertEqual(
            terminal["schema_version"],
            "motive-goku-full-motion-finalize-release-watcher-v1",
        )
        self.assertEqual(terminal["status"], "complete")
        self.assertEqual(terminal["nodes"], NODES)
        self.assertEqual(terminal["finalize_node"], NODES[0])
        self.assertEqual(terminal["full_input"]["sha256"], _sha(self.full_input.read_bytes()))
        self.assertEqual(terminal["release_request"]["path"], str(self.request))
        self.assertEqual(stat.S_IMODE(self.ready.stat().st_mode), 0o400)
        self.assertEqual(stat.S_IMODE(self.terminal.stat().st_mode), 0o400)
        self.assertEqual(stat.S_IMODE(self.request.stat().st_mode), 0o400)

        calls = self.srun_log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(calls), 3)
        self.assertIn("motive.goku_full_motion_finalize", calls[0])
        self.assertIn("motive.goku_full_motion_shard_manifest", calls[1])
        self.assertIn("motive.wan22_full_motion_signed_release prepare", calls[2])
        for call in calls:
            self.assertIn("--overlap", call)
            self.assertIn("--mem=0", call)
            self.assertIn("--exact", call)
            self.assertNotIn("--exclusive", call)
            self.assertNotIn("--kill-on-bad-exit", call)
        self.assertTrue(self.final_pool.is_dir())
        self.assertEqual(len(list((self.shard_root / "shards").glob("*.jsonl"))), 32)
        self.assertFalse((self.release_dir / "root_signed_release.json").exists())

    def test_old_six_line_qwen_receipt_is_rejected_before_any_srun(self) -> None:
        completed = self._run(FAKE_OLD_QWEN_RECEIPT="1")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("exactly eight lines", completed.stderr)
        self.assertTrue(self.ready.is_file())
        self.assertFalse(self.terminal.exists())
        self.assertFalse(self.final_pool.exists())
        self.assertEqual(self.srun_log.read_text(encoding="utf-8"), "")

    def test_existing_terminal_receipt_fails_before_readiness_or_steps(self) -> None:
        self.terminal.write_text("do not replace\n", encoding="utf-8")
        completed = self._run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("create-only watcher terminal receipt already exists", completed.stderr)
        self.assertEqual(self.terminal.read_text(encoding="utf-8"), "do not replace\n")
        self.assertFalse(self.ready.exists())
        self.assertEqual(self.srun_log.read_text(encoding="utf-8"), "")

    def test_readiness_receipt_cannot_be_published_inside_release_root(self) -> None:
        unsafe = self.release_dir / "watcher.ready.json"
        completed = self._run(
            MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WATCHER_READY=str(unsafe)
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("outside stage output roots", completed.stderr)
        self.assertFalse(unsafe.exists())
        self.assertFalse(self.terminal.exists())
        self.assertEqual(self.srun_log.read_text(encoding="utf-8"), "")

    def test_forged_jobs_row_count_is_rejected_without_terminal_receipt(self) -> None:
        completed = self._run(FAKE_BAD_JOBS="1")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("exact header and 32 data rows", completed.stderr)
        self.assertTrue(self.ready.is_file())
        self.assertFalse(self.terminal.exists())

    def test_lexical_snapshot_escape_for_verifier_is_rejected(self) -> None:
        outside = self.root / "outside_snapshot_verifier.py"
        outside.write_text("# not frozen\n", encoding="utf-8")
        escaped = self.snapshot / ".." / outside.name
        completed = self._run(MOTIVE_FULL_MOTION_SNAPSHOT_TOOL=str(escaped))
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("resolve inside the frozen snapshot", completed.stderr)
        self.assertFalse(self.ready.exists())
        self.assertEqual(self.srun_log.read_text(encoding="utf-8"), "")

    def test_term_kills_active_srun_and_never_publishes_terminal_receipt(self) -> None:
        marker = self.root / "blocked-srun.active"
        process = subprocess.Popen(
            ["bash", str(SCRIPT)],
            env=self._environment(
                FAKE_BLOCK_SRUN="1", FAKE_BLOCK_MARKER=str(marker)
            ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 10
            while not marker.exists() and process.poll() is None:
                if time.monotonic() >= deadline:
                    self.fail("watcher did not enter the fake srun step")
                time.sleep(0.05)
            self.assertIsNone(process.poll())
            process.terminate()
            stdout, stderr = process.communicate(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
        self.assertNotEqual(process.returncode, 0, stdout + stderr)
        self.assertTrue(self.ready.is_file())
        self.assertFalse(self.terminal.exists())
        self.assertFalse(self.final_pool.exists())

    def test_script_has_signal_cleanup_and_no_sign_or_wan_execution(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for marker in (
            "trap 'abort_signal 143' TERM",
            "trap 'abort_signal 130' INT",
            "kill -TERM",
            "wait \"${active_pid}\"",
            "srun --overlap",
            "--mem=0",
            "motive.goku_full_motion_signed_release prepare",
        ):
            if marker == "motive.goku_full_motion_signed_release prepare":
                self.assertNotIn(marker, text)
            else:
                self.assertIn(marker, text)
        self.assertNotIn(" signed_release sign", text)
        self.assertNotIn("Wan", "\n".join(
            line for line in text.splitlines() if line.lstrip().startswith("srun")
        ))


if __name__ == "__main__":
    unittest.main()
