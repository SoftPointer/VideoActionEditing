from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    REPO_ROOT
    / "methods"
    / "motive"
    / "scripts"
    / "auh_full_motion_postcheck_dispatch_existing_job.sh"
)


class FullMotionPostcheckDispatchTests(unittest.TestCase):
    def test_frozen_inputs_topology_and_postcheck_command(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for variable in (
            "MOTIVE_EXISTING_SLURM_JOB_ID",
            "MOTIVE_FULL_MOTION_POSTCHECK_NODES",
            "MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT",
            "MOTIVE_FULL_MOTION_GENERATION_SHARD_DIR",
            "MOTIVE_FULL_MOTION_WAN_SHARDS_ROOT",
            "MOTIVE_FULL_MOTION_POSTCHECK_MODEL",
            "MOTIVE_FULL_MOTION_POSTCHECK_PYTHON",
            "MOTIVE_FULL_MOTION_POSTCHECK_FFPROBE",
            "MOTIVE_FULL_MOTION_POSTCHECK_FFMPEG",
            "MOTIVE_FULL_MOTION_POSTCHECK_OUTPUT_ROOT",
        ):
            self.assertIn(f"${{{variable}:?", text)
        for marker in (
            "exactly four nodes are required",
            "for wave in 0 1 2 3",
            "for slot in 0 1 2 3 4 5 6 7",
            'shard_index="$(( wave * 8 + slot ))"',
            'node="${nodes[$(( slot / 2 ))]}"',
            "--gpus-per-task=4",
            "--cpus-per-task=\"${worker_cpus}\"",
            "--mem=0",
            "srun --overlap",
            "ROCR_VISIBLE_DEVICES=\"${gpu_devices}\"",
            "HIP_VISIBLE_DEVICES=\"${gpu_devices}\"",
            "CUDA_VISIBLE_DEVICES=\"${gpu_devices}\"",
            "-m motive.goku_full_motion_postcheck",
            '--manifest "${manifest}"',
            '--generation-root "${generation_root}"',
            '--output "${output}"',
            '--model "${model}"',
            '--ffprobe "${ffprobe_bin}"',
            '--ffmpeg "${ffmpeg_bin}"',
            "--nframes 24",
            "--max-pixels 1179648",
            "--max-new-tokens 4096",
            "--resume",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("--allow-download", text)
        self.assertNotIn("--exclusive", text)
        self.assertNotIn("--kill-on-bad-exit", text)
        self.assertNotIn("scancel", text)

    def test_two_strict_idle_audits_and_failures_are_isolated(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for marker in (
            "check_idle_node",
            "for audit in 1 2",
            "rocm-smi --showuse --showmemuse --showmeminfo vram --csv",
            'if (label == "GPU use (%)") use_index=column_number',
            'if (label == "GPU Memory Allocated (VRAM%)") percent_index=column_number',
            'if (label == "VRAM Total Used Memory (B)") used_index=column_number',
            "(use_value+0) != 0",
            "(percent_value+0) != 0",
            "(used_value+0) > 1073741824",
            "rocm-smi --showpids --csv",
            "(gpu_flag+0) != 0",
            "(vram+0) != 0",
            'sleep "${idle_recheck_seconds}"',
            "preflight_wan_closure",
            "p._validate_run_contract",
            "p._validate_generated_manifest",
            'wait "${pid}" || exit_code=$?',
            "preflight_failed",
            "postcheck_failed",
            "dispatcher_status.tsv",
            "dispatcher_receipt.json",
            "motive-goku-full-motion-postcheck-dispatch-receipt-v2",
            "os.link(source, target, follow_symlinks=False)",
            "atomic create-only publication lost a race",
            '(( failure_count == 0 )) || exit 1',
        ):
            self.assertIn(marker, text)
        self.assertIn('>"${log_root}/${shard_id}.out"', text)
        self.assertIn('2>"${log_root}/${shard_id}.err"', text)
        self.assertNotIn("worker_mem", text)
        self.assertNotIn("--mem=1G", text)
        self.assertNotIn("--mem=110G", text)
        self.assertNotIn('mv "${status_tmp}" "${status_path}"', text)
        self.assertNotIn('mv "${receipt_tmp}" "${controller_receipt}"', text)

    def test_script_has_valid_bash_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def _fake_run(
        self,
        *,
        failed_shard: str | None,
        omit_media: str | None = None,
        ffprobe_mode: str = "valid",
        ffmpeg_mode: str = "valid",
        preexisting_publication: str | None = None,
        race_publication: str | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        snapshot = root / "snapshot"
        code = snapshot / "methods" / "motive" / "motive"
        manifests = root / "manifests"
        wan = root / "wan_shards"
        model = root / "Qwen3-VL-32B-Instruct"
        output = root / "postcheck"
        fake_bin = root / "fake-bin"
        for directory in (code, manifests, wan, model, fake_bin):
            directory.mkdir(parents=True, exist_ok=True)
        (snapshot / "SOURCE_FILES.jsonl").write_text("{}\n", encoding="utf-8")
        (code / "goku_full_motion_postcheck.py").write_text(
            "# frozen fixture\n", encoding="utf-8"
        )
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        media_paths: dict[str, Path] = {}
        for label, mode in (
            ("ffprobe", ffprobe_mode),
            ("ffmpeg", ffmpeg_mode),
        ):
            target = root / f"{label}-real"
            target.write_text(
                f"#!/usr/bin/env bash\necho fake-{label}\n", encoding="utf-8"
            )
            target.chmod(target.stat().st_mode | stat.S_IXUSR)
            if mode == "valid":
                path = target
            elif mode == "symlink":
                path = root / f"{label}-symlink"
                path.symlink_to(target)
            elif mode == "unexecutable":
                target.chmod(target.stat().st_mode & ~0o111)
                path = target
            else:
                raise ValueError(f"unsupported {label} fixture mode: {mode}")
            media_paths[label] = path
        for index in range(32):
            shard_id = f"shard_{index:03d}"
            (manifests / f"{shard_id}.jsonl").write_text(
                "{}\n" * 8, encoding="utf-8"
            )
            (wan / shard_id).mkdir()

        python_log = root / "python.log"
        fake_python = root / "fake-python"
        fake_python.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "${FAKE_PYTHON_LOG:?}"
if [[ "${1:-}" == "-c" && "${2:-}" == *'atomic create-only publication'* ]]; then
  if [[ -n "${FAKE_RACE_PUBLICATION:-}" && "${4:-}" == "${FAKE_RACE_PUBLICATION}" ]]; then
    printf 'competitor\n' > "${FAKE_RACE_PUBLICATION}"
  fi
  exec "${REAL_PYTHON:?}" "$@"
fi
if [[ "${1:-}" == "-c" && "${2:-}" == *'media executable changed while hashing'* ]]; then
  exec "${REAL_PYTHON:?}" "$@"
fi
if [[ "${1:-}" == "-c" && "${2:-}" == *'dispatcher status does not contain 32 shards'* ]]; then
  exec "${REAL_PYTHON:?}" "$@"
fi
exit 0
""",
            encoding="utf-8",
        )
        fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)

        (fake_bin / "scontrol").write_text(
            """#!/usr/bin/env bash
if [[ "${1:-}" == "show" && "${2:-}" == "job" ]]; then
  echo 'JobId=123 JobState=RUNNING NodeList=auh1,auh2,auh3,auh4'
elif [[ "${1:-}" == "show" && "${2:-}" == "hostnames" ]]; then
  printf 'auh1\nauh2\nauh3\nauh4\n'
else
  exit 2
fi
""",
            encoding="utf-8",
        )
        (fake_bin / "squeue").write_text(
            """#!/usr/bin/env bash
case " $* " in
  *' -s '*) echo '123.0' ;;
  *) echo 'auh1,auh2,auh3,auh4' ;;
esac
""",
            encoding="utf-8",
        )
        (fake_bin / "sleep").write_text(
            "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
        )
        (fake_bin / "srun").write_text(
            """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "${FAKE_SRUN_LOG:?}"
case " $* " in
  *' --overlap '*) ;;
  *) exit 90 ;;
esac
case " $* " in
  *' --exclusive '*|*' --kill-on-bad-exit'*) exit 91 ;;
esac
output=''
previous=''
for argument in "$@"; do
  if [[ "${previous}" == '--output' ]]; then output="${argument}"; fi
  previous="${argument}"
done
if [[ -n "${output}" ]]; then
  if [[ -n "${FAKE_FAIL_SHARD:-}" && "${output}" == *"${FAKE_FAIL_SHARD}"* ]]; then
    exit 7
  fi
  mkdir -p "${output%/*}"
  printf '{}\n' > "${output}"
  receipt="${output%.jsonl}.receipt.json"
  printf '{}\n' > "${receipt}"
fi
exit 0
""",
            encoding="utf-8",
        )
        for name in ("scontrol", "squeue", "sleep", "srun"):
            path = fake_bin / name
            path.chmod(path.stat().st_mode | stat.S_IXUSR)

        srun_log = root / "srun.log"
        publication_paths = {
            "status": output / "dispatcher_status.tsv",
            "receipt": output / "dispatcher_receipt.json",
        }
        if preexisting_publication is not None:
            publication = publication_paths[preexisting_publication]
            publication.parent.mkdir(parents=True, exist_ok=True)
            publication.write_text("competitor\n", encoding="utf-8")
        environment = dict(os.environ)
        environment.update(
            {
                "PATH": f"{fake_bin}:{environment['PATH']}",
                "MOTIVE_EXISTING_SLURM_JOB_ID": "123",
                "MOTIVE_FULL_MOTION_POSTCHECK_NODES": "auh1,auh2,auh3,auh4",
                "MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT": str(snapshot),
                "MOTIVE_FULL_MOTION_GENERATION_SHARD_DIR": str(manifests),
                "MOTIVE_FULL_MOTION_WAN_SHARDS_ROOT": str(wan),
                "MOTIVE_FULL_MOTION_POSTCHECK_MODEL": str(model),
                "MOTIVE_FULL_MOTION_POSTCHECK_PYTHON": str(fake_python),
                "MOTIVE_FULL_MOTION_POSTCHECK_FFPROBE": str(
                    media_paths["ffprobe"]
                ),
                "MOTIVE_FULL_MOTION_POSTCHECK_FFMPEG": str(
                    media_paths["ffmpeg"]
                ),
                "MOTIVE_FULL_MOTION_POSTCHECK_OUTPUT_ROOT": str(output),
                "MOTIVE_FULL_MOTION_POSTCHECK_IDLE_RECHECK_SECONDS": "1",
                "FAKE_PYTHON_LOG": str(python_log),
                "FAKE_SRUN_LOG": str(srun_log),
                "FAKE_FAIL_SHARD": failed_shard or "",
                "FAKE_RACE_PUBLICATION": (
                    str(publication_paths[race_publication])
                    if race_publication is not None
                    else ""
                ),
                "REAL_PYTHON": str(Path(sys.executable).resolve(strict=True)),
            }
        )
        if omit_media is not None:
            environment.pop(f"MOTIVE_FULL_MOTION_POSTCHECK_{omit_media.upper()}")
        completed = subprocess.run(
            ["bash", str(SCRIPT)],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        return completed, root

    def test_missing_media_binary_bindings_fail_before_dispatch(self) -> None:
        for label in ("ffprobe", "ffmpeg"):
            with self.subTest(label=label):
                completed, root = self._fake_run(
                    failed_shard=None, omit_media=label
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(
                    f"MOTIVE_FULL_MOTION_POSTCHECK_{label.upper()}",
                    completed.stderr,
                )
                self.assertFalse((root / "postcheck").exists())

    def test_symlink_and_unexecutable_media_binaries_are_rejected(self) -> None:
        for label, mode in (
            ("ffprobe", "symlink"),
            ("ffmpeg", "unexecutable"),
        ):
            with self.subTest(label=label, mode=mode):
                completed, root = self._fake_run(
                    failed_shard=None,
                    ffprobe_mode=mode if label == "ffprobe" else "valid",
                    ffmpeg_mode=mode if label == "ffmpeg" else "valid",
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(
                    f"{label} must be an absolute regular non-symlink executable",
                    completed.stderr,
                )
                self.assertFalse((root / "postcheck").exists())

    def test_fake_success_runs_all_32_shards_on_expected_dual_slots(self) -> None:
        completed, root = self._fake_run(failed_shard=None)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("preserving existing steps: 123.0", completed.stdout)
        srun_lines = (root / "srun.log").read_text(encoding="utf-8").splitlines()
        postchecks = [line for line in srun_lines if "--output " in line]
        self.assertEqual(len(postchecks), 32)
        srun_commands = [line for line in srun_lines if "--jobid=" in line]
        self.assertEqual(len(srun_commands), 48)
        self.assertTrue(all("--mem=0" in line for line in srun_commands))
        for index in range(32):
            shard_id = f"shard_{index:03d}"
            line = next(item for item in postchecks if shard_id in item)
            expected_node = f"auh{(index % 8) // 2 + 1}"
            expected_gpus = "0,1,2,3" if index % 2 == 0 else "4,5,6,7"
            self.assertIn(f"--nodelist={expected_node}", line)
            self.assertIn("--gpus-per-task=4", line)
            self.assertIn(f"ROCR_VISIBLE_DEVICES={expected_gpus}", line)
            self.assertIn(f"--ffprobe {root / 'ffprobe-real'}", line)
            self.assertIn(f"--ffmpeg {root / 'ffmpeg-real'}", line)
        status = (root / "postcheck" / "dispatcher_status.tsv").read_text()
        self.assertEqual(status.count("\tcomplete\t"), 32)
        self.assertTrue(
            (root / "postcheck" / "dispatcher_receipt.json").is_file()
        )
        receipt = json.loads(
            (root / "postcheck" / "dispatcher_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            receipt["schema_version"],
            "motive-goku-full-motion-postcheck-dispatch-receipt-v2",
        )
        for label in ("ffprobe", "ffmpeg"):
            executable = root / f"{label}-real"
            self.assertEqual(
                receipt["media_tools"][label],
                {
                    "path": str(executable),
                    "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                },
            )
        receipt_payload = dict(receipt)
        receipt_digest = receipt_payload.pop("receipt_digest")
        canonical = json.dumps(
            receipt_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self.assertEqual(receipt_digest, hashlib.sha256(canonical).hexdigest())

    def test_fake_single_failure_does_not_block_other_shards(self) -> None:
        completed, root = self._fake_run(failed_shard="shard_005")
        self.assertEqual(completed.returncode, 1, completed.stderr)
        srun_lines = (root / "srun.log").read_text(encoding="utf-8").splitlines()
        self.assertEqual(sum("--output " in line for line in srun_lines), 32)
        status = (root / "postcheck" / "dispatcher_status.tsv").read_text()
        self.assertIn("shard_005\t0\t5\tauh3\tpostcheck_failed\t7", status)
        self.assertEqual(status.count("\tcomplete\t"), 31)
        self.assertTrue(
            (root / "postcheck" / "dispatcher_receipt.json").is_file()
        )

    def test_preexisting_terminal_publications_are_rejected_unchanged(self) -> None:
        for label in ("status", "receipt"):
            with self.subTest(label=label):
                completed, root = self._fake_run(
                    failed_shard=None, preexisting_publication=label
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(
                    "create-only publication already exists", completed.stderr
                )
                publication = root / "postcheck" / (
                    "dispatcher_status.tsv"
                    if label == "status"
                    else "dispatcher_receipt.json"
                )
                self.assertEqual(
                    publication.read_text(encoding="utf-8"), "competitor\n"
                )
                srun_log = root / "srun.log"
                self.assertFalse(srun_log.exists() and srun_log.read_text())

    def test_competing_terminal_publications_are_never_overwritten(self) -> None:
        for label in ("status", "receipt"):
            with self.subTest(label=label):
                completed, root = self._fake_run(
                    failed_shard=None, race_publication=label
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(
                    f"create-only dispatcher {label} publication failed",
                    completed.stderr,
                )
                publication = root / "postcheck" / (
                    "dispatcher_status.tsv"
                    if label == "status"
                    else "dispatcher_receipt.json"
                )
                self.assertEqual(
                    publication.read_text(encoding="utf-8"), "competitor\n"
                )
                self.assertEqual(
                    list((root / "postcheck").glob("dispatcher_*.tmp.*")), []
                )
                self.assertEqual(
                    list((root / "postcheck").glob("dispatcher_*.unsorted.*")),
                    [],
                )


if __name__ == "__main__":
    unittest.main()
