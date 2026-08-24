from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "auh_full_motion_qwen_distributed_existing_job.sh"
)
NODES = [f"auh{index:03d}" for index in range(1, 5)]


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


FAKE_SRUN = r'''#!/usr/bin/env python3
import os
from pathlib import Path
import sys
import time

args = sys.argv[1:]
log = Path(os.environ["FAKE_QWEN_SRUN_LOG"])
node = next(
    (arg.split("=", 1)[1] for arg in args if arg.startswith("--nodelist=")),
    "",
)
def append(value):
    with log.open("a", encoding="utf-8") as handle:
        handle.write(value + "\n")
if "--overlap" not in args:
    append("unsafe\tmissing_overlap")
    raise SystemExit(90)
if "--exclusive" in args or any(
    arg.startswith("--kill-on-bad-exit") for arg in args
):
    append("unsafe\tstep_control")
    raise SystemExit(91)
joined = " ".join(args)
if "rocm-smi" in joined:
    append("idle\t" + node)
    if os.environ.get("FAKE_IDLE_FAIL_NODE") == node:
        raise SystemExit(19)
    raise SystemExit(0)
probe_slot_arg = next(
    (arg for arg in args if arg.startswith("MOTIVE_DUAL4_PROBE_SLOT=")),
    "",
)
if probe_slot_arg:
    slot = int(probe_slot_arg.split("=", 1)[1])
    expected_gpus = "0,1,2,3" if slot == 0 else "4,5,6,7"
    required = {
        "--overlap",
        "--nodes=1",
        "--exact",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--mem=0",
        "--gpus-per-task=4",
        "--gpu-bind=none",
    }
    if node != "auh001" or not required.issubset(args):
        append(f"unsafe\tprobe_geometry_{slot}")
        raise SystemExit(94)
    for name in (
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        if f"{name}={expected_gpus}" not in args:
            append(f"unsafe\tprobe_visibility_{slot}_{name}")
            raise SystemExit(95)
    if not any(
        "torch.cuda.device_count()" in arg
        and "torch.cuda.synchronize(logical_device)" in arg
        and "torch.ones(" in arg
        and "(256,)" in arg
        and 'device=f"cuda:{logical_device}"' in arg
        for arg in args
    ):
        append(f"unsafe\tprobe_payload_{slot}")
        raise SystemExit(96)
    state = Path(os.environ["FAKE_QWEN_PROBE_STATE_DIR"])
    state.mkdir(parents=True, exist_ok=True)
    (state / f"started_{slot}").write_text("started\n", encoding="utf-8")
    append(f"probe_start\t{node}\t{slot}\t{expected_gpus}")
    peer = state / f"started_{1 - slot}"
    deadline = time.monotonic() + 3.0
    while not peer.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not peer.exists():
        append(f"unsafe\tprobe_not_concurrent_{slot}")
        raise SystemExit(97)
    append(f"probe_concurrent\t{node}\t{slot}")
    if os.environ.get("FAKE_PROBE_FAIL_SLOT") == str(slot):
        append(f"probe_fail\t{node}\t{slot}")
        raise SystemExit(31)
    append(f"probe_done\t{node}\t{slot}")
    raise SystemExit(0)
if "motive.goku_full_motion_qwen" not in args:
    append("cache\t" + node)
    raise SystemExit(0)
def after(flag):
    return args[args.index(flag) + 1]
shard = int(after("--shard-index"))
output = Path(after("--output"))
expected_node = f"auh{shard // 2 + 1:03d}"
expected_gpus = "0,1,2,3" if shard % 2 == 0 else "4,5,6,7"
if node != expected_node:
    raise SystemExit(92)
for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
    if f"{name}={expected_gpus}" not in args:
        raise SystemExit(93)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("{}\n", encoding="utf-8")
output.with_name(output.stem + ".receipt.json").write_text(
    "{}\n", encoding="utf-8"
)
append(f"run\t{node}\t{shard}\t{expected_gpus}")
'''


class FullMotionQwenDistributedExistingJobTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.snapshot = self.root / "snapshot"
        code = self.snapshot / "methods" / "motive" / "motive"
        code.mkdir(parents=True)
        (code / "goku_full_motion_qwen.py").write_text(
            "# frozen fixture\n", encoding="utf-8"
        )
        self.input = self.root / "full_input.jsonl"
        self.input.write_text("{}\n" * 768, encoding="utf-8")
        self.model = self.root / "Qwen3-VL-32B-Instruct"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}\n", encoding="utf-8")
        self.smoke = self.root / "smoke.json"
        self.smoke.write_text('{"status":"pass"}\n', encoding="utf-8")
        self.output = self.root / "qwen"
        self.done = self.output / "controller.done"
        self.log = self.root / "srun.tsv"
        self.log.write_text("", encoding="utf-8")
        self.probe_state = self.root / "probe-state"
        self.temp_root = self.root / "tmp"
        self.temp_root.mkdir()
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        _write_executable(self.fake_bin / "srun", FAKE_SRUN)
        _write_executable(
            self.fake_bin / "scontrol",
            """#!/usr/bin/env bash
if [[ "${1:-}" == show && "${2:-}" == job ]]; then
  echo 'JobId=123 JobState=RUNNING NodeList=auh[001-008] gres/gpu:mi210=64'
elif [[ "${1:-}" == show && "${2:-}" == hostnames ]]; then
  printf 'auh001\nauh002\nauh003\nauh004\nauh005\nauh006\nauh007\nauh008\n'
else
  exit 2
fi
""",
        )
        _write_executable(
            self.fake_bin / "squeue",
            """#!/usr/bin/env bash
case " $* " in
  *' -s '*) echo '123.0' ;;
  *) echo 'auh[001-008]' ;;
esac
""",
        )
        _write_executable(
            self.fake_bin / "sleep", "#!/usr/bin/env bash\nexit 0\n"
        )
        self.python = self.root / "python"
        _write_executable(
            self.python,
            """#!/usr/bin/env bash
if [[ "${1:-}" == '-c' && "${2:-}" == *'atomic create-only publication'* ]]; then
  if [[ -n "${FAKE_RACE_PUBLICATION:-}" && "${4:-}" == "${FAKE_RACE_PUBLICATION}" ]]; then
    printf 'competitor\n' > "${FAKE_RACE_PUBLICATION}"
  fi
  exec "${REAL_PYTHON:?}" "$@"
fi
if [[ "${1:-}" == '-c' && "${2:-}" == *'Qwen controller input changed while hashing'* ]]; then
  exec "${REAL_PYTHON:?}" "$@"
fi
if [[ "${1:-}" == '-c' ]]; then
  echo pass
  exit 0
fi
exit 99
""",
        )

    def _run(self, **updates: str) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment.update(
            {
                "PATH": f"{self.fake_bin}:{environment['PATH']}",
                "FAKE_QWEN_SRUN_LOG": str(self.log),
                "FAKE_QWEN_PROBE_STATE_DIR": str(self.probe_state),
                "TMPDIR": str(self.temp_root),
                "MOTIVE_EXISTING_SLURM_JOB_ID": "123",
                "MOTIVE_FULL_MOTION_NODES": ",".join(NODES),
                "MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT": str(self.snapshot),
                "MOTIVE_FULL_MOTION_SMOKE_GATE": str(self.smoke),
                "MOTIVE_FULL_MOTION_FULL_INPUT": str(self.input),
                "MOTIVE_FULL_MOTION_FULL_QWEN_ROOT": str(self.output),
                "MOTIVE_FULL_MOTION_FULL_QWEN_DONE": str(self.done),
                "MOTIVE_FULL_MOTION_QWEN_MODEL": str(self.model),
                "MOTIVE_FULL_MOTION_QWEN_PYTHON": str(self.python),
                "REAL_PYTHON": str(Path(sys.executable).resolve(strict=True)),
                "LC_ALL": "C",
            }
        )
        environment.update(updates)
        return subprocess.run(
            ["bash", str(SCRIPT)],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )

    def _calls(self, kind: str) -> list[list[str]]:
        return [
            line.split("\t")
            for line in self.log.read_text(encoding="utf-8").splitlines()
            if line.startswith(kind + "\t")
        ]

    def test_holder_step_is_preserved_and_all_eight_shards_run(self) -> None:
        completed = self._run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("preserving existing steps: 123.0", completed.stdout)
        idle = self._calls("idle")
        self.assertEqual(len(idle), 8)
        self.assertEqual(
            {node: sum(row[1] == node for row in idle) for node in NODES},
            {node: 2 for node in NODES},
        )
        runs = self._calls("run")
        self.assertEqual(len(runs), 8)
        self.assertEqual({int(row[2]) for row in runs}, set(range(8)))
        for shard in range(8):
            row = next(value for value in runs if int(value[2]) == shard)
            self.assertEqual(row[1], NODES[shard // 2])
            self.assertEqual(
                row[3], "0,1,2,3" if shard % 2 == 0 else "4,5,6,7"
            )
        self.assertTrue(self.done.is_file())
        self.assertEqual(self._calls("unsafe"), [])
        self.assertEqual(list(self.temp_root.iterdir()), [])
        starts = self._calls("probe_start")
        concurrent = self._calls("probe_concurrent")
        completed_probes = self._calls("probe_done")
        self.assertEqual({int(row[2]) for row in starts}, {0, 1})
        self.assertEqual({int(row[2]) for row in concurrent}, {0, 1})
        self.assertEqual({int(row[2]) for row in completed_probes}, {0, 1})
        lines = self.log.read_text(encoding="utf-8").splitlines()
        last_probe = max(
            index for index, line in enumerate(lines) if line.startswith("probe_done\t")
        )
        first_product = min(
            index
            for index, line in enumerate(lines)
            if line.startswith(("cache\t", "run\t"))
        )
        self.assertLess(last_probe, first_product)

        self.assertEqual(stat.S_IMODE(self.done.stat().st_mode), 0o400)
        self.assertEqual(
            list(self.done.parent.glob(self.done.name + ".tmp.*")), []
        )

    def test_preexisting_done_is_rejected_before_any_slurm_step(self) -> None:
        self.output.mkdir()
        self.done.write_text("competitor\n", encoding="utf-8")
        completed = self._run()
        self.assertEqual(completed.returncode, 2)
        self.assertIn("done output already exists", completed.stderr)
        self.assertEqual(self.done.read_text(encoding="utf-8"), "competitor\n")
        self.assertEqual(self.log.read_text(encoding="utf-8"), "")

    def test_competing_done_publication_is_never_overwritten(self) -> None:
        completed = self._run(FAKE_RACE_PUBLICATION=str(self.done))
        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "create-only done output publication failed", completed.stderr
        )
        self.assertEqual(self.done.read_text(encoding="utf-8"), "competitor\n")
        self.assertEqual(len(self._calls("run")), 8)
        self.assertEqual(
            list(self.done.parent.glob(self.done.name + ".tmp.*")), []
        )

    def test_idle_failure_stops_before_cache_or_qwen_workers(self) -> None:
        completed = self._run(FAKE_IDLE_FAIL_NODE=NODES[2])
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(self._calls("run"), [])
        self.assertEqual(self._calls("cache"), [])
        self.assertEqual(self._calls("probe_start"), [])
        self.assertFalse(self.done.exists())

    def test_dual4_probe_failure_is_fail_closed_before_product_workers(self) -> None:
        completed = self._run(FAKE_PROBE_FAIL_SLOT="1")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "concurrent dual4 Qwen admission probe failed", completed.stderr
        )
        self.assertEqual(
            {int(row[2]) for row in self._calls("probe_start")}, {0, 1}
        )
        self.assertEqual(
            {int(row[2]) for row in self._calls("probe_concurrent")}, {0, 1}
        )
        self.assertEqual(self._calls("run"), [])
        self.assertEqual(self._calls("cache"), [])
        self.assertFalse(self.done.exists())
        self.assertEqual(list(self.temp_root.iterdir()), [])

    def test_overlap_policy_is_non_destructive_and_exact_zero(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for marker in (
            "preserving existing steps",
            "for audit in 1 2",
            'label == "GPU use (%)"',
            'label == "GPU Memory Allocated (VRAM%)"',
            'label == "VRAM Total Used Memory (B)"',
            "seen == 8",
            "1073741824",
            "rocm-smi --showpids --csv",
            "(gpu_flag+0) != 0",
            "(vram+0) != 0",
            "srun --overlap",
            "--mem=0",
            'gpu_devices="0,1,2,3"',
            'gpu_devices="4,5,6,7"',
            "run_dual4_probe 0 \"0,1,2,3\"",
            "run_dual4_probe 1 \"4,5,6,7\"",
            "--cpus-per-task=1 --mem=0",
            "--gpus-per-task=4 --gpu-bind=none",
            "torch.cuda.device_count()",
            "torch.ones(",
            "(256,)",
            'device=f"cuda:{logical_device}"',
            "torch.cuda.synchronize(logical_device)",
            'wait "${dual4_probe_pid_0}"',
            'wait "${dual4_probe_pid_1}"',
            "concurrent dual4 Qwen admission probe failed",
            "--nframes 16",
            "--tile-width 512",
            "--mosaic-columns 4",
            "--max-pixels 2359296",
            "--max-new-tokens 6144",
            "os.link(source, target, follow_symlinks=False)",
            "atomic create-only publication lost a race",
            "Qwen controller input changed while hashing",
        ):
            self.assertIn(marker, text)
        self.assertNotIn('mv "${temporary}" "${done_output}"', text)
        self.assertNotIn("sha256sum", text)
        self.assertNotIn("--exclusive", text)
        self.assertNotIn("--kill-on-bad-exit", text)
        self.assertNotIn("scancel", text)

        probe_start = text.index('run_dual4_probe 0 "0,1,2,3"')
        product_start = text.index("for shard_index in 0 1 2 3 4 5 6 7")
        self.assertLess(probe_start, product_start)


if __name__ == "__main__":
    unittest.main()
