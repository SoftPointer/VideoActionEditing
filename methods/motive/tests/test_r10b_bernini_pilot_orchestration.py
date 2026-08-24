from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import re
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
    / "auh_r10b_bernini_pilot.sbatch"
)


class R10BBerniniPilotOrchestrationTests(unittest.TestCase):
    def test_resources_and_closed_pilot_contract_are_explicit(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for directive in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=16",
            "#SBATCH --mem=128G",
            "#SBATCH --gres=gpu:mi210:1",
            "#SBATCH --time=03:00:00",
            "#SBATCH --job-name=motive-r10b-bernini-pilot",
        ):
            self.assertIn(directive, text)

        required = (
            "MOTIVE_R10B_SOURCE_SNAPSHOT",
            "MOTIVE_R10B_SOURCE_TREE_SHA256",
            "MOTIVE_R10B_MANIFEST",
            "MOTIVE_R10B_MANIFEST_SHA256",
            "MOTIVE_R10B_TRACK_CACHE",
            "MOTIVE_R10B_TRACK_CACHE_SHA256",
            "MOTIVE_R10B_MODEL_PATH",
            "MOTIVE_R10B_MODEL_REVISION",
            "MOTIVE_R10B_MODEL_TREE_SHA256",
            "MOTIVE_R10B_BERNINI_REPO",
            "MOTIVE_R10B_BERNINI_SOURCE_COMMIT",
            "MOTIVE_R10B_BERNINI_SOURCE_BUNDLE_SHA256",
            "MOTIVE_R10B_OUTPUT_DIR",
            'artifact_kind="controlled_retrieval_pilot"',
            'resize_mode="aspect_preserving_center_crop"',
            'scheduler_index="${MOTIVE_R10B_SCHEDULER_INDEX:-25}"',
            'noise_mode="${MOTIVE_R10B_NOISE_MODE:-iid_spatiotemporal}"',
            'width="${MOTIVE_R10B_WIDTH:-256}"',
            'height="${MOTIVE_R10B_HEIGHT:-256}"',
            'projection_dim="${MOTIVE_R10B_PROJECTION_DIM:-2048}"',
            'num_frames="17"',
            'projection_seed_1="260108851"',
            'projection_seed_2="260108852"',
            "--artifact-kind",
            "--resize-mode",
            "--scheduler-index",
            "--noise-mode",
            "--projection-dim",
            "--projection-seeds",
            "--validate-only",
            'verify_bound_inputs "before"',
            'verify_bound_inputs "after"',
            "HF_HUB_OFFLINE=1",
            "HF_DATASETS_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "WANDB_DISABLED=true",
            "videos_rendered",
            "videos_copied",
            "optimizer_steps",
            "representation_gate_passed",
            "renderer_probe_authorized",
            "editor_training_authorized",
        )
        for marker in required:
            self.assertIn(marker, text)

        self.assertEqual(
            text.count("-m motive.r10b_bernini_tangent_extract"),
            2,
        )
        self.assertNotIn("ffmpeg", text)
        self.assertNotIn("rsync", text)
        self.assertNotIn("torchrun", text)
        self.assertNotIn("deepspeed", text)
        self.assertNotRegex(text, r"(?m)^\s*(?:cp|scp)\s")

    def test_max_samples_is_only_forwarded_when_explicitly_set(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"${MOTIVE_R10B_MAX_SAMPLES+x}" == "x"', text)
        self.assertIn(
            'max_samples_args=(--max-samples "${MOTIVE_R10B_MAX_SAMPLES}")',
            text,
        )
        self.assertNotIn("--max-samples 2", text)

    def test_script_has_valid_bash_and_embedded_python_syntax(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        blocks = re.findall(
            r"<<'PY'\n(.*?)\nPY",
            SCRIPT.read_text(encoding="utf-8"),
            flags=re.DOTALL,
        )
        self.assertEqual(len(blocks), 2)
        for block in blocks:
            ast.parse(block)

    def test_fake_run_forwards_defaults_and_optional_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source_snapshot = root / "snapshot"
            manifest = root / "manifest.jsonl"
            track_cache = root / "tracks.npz"
            model_path = root / "model"
            bernini_repo = root / "bernini"
            fake_python = root / "fake-python"
            fake_bin = root / "bin"
            fake_sha256sum = fake_bin / "sha256sum"
            log = root / "python.log"
            scratch = root / "scratch"
            source_snapshot.mkdir()
            model_path.mkdir()
            bernini_repo.mkdir()
            fake_bin.mkdir()
            scratch.mkdir()
            manifest.write_bytes(b'{"schema_version":"test"}\n')
            track_cache.write_bytes(b"frozen-track-cache")
            fake_python.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${FAKE_PYTHON_LOG:?}"
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  exit 0
fi
if [[ "${1:-}" == "-m" \
  && "${2:-}" == "motive.r10b_bernini_tangent_extract" ]]; then
  output=""
  validate="false"
  previous=""
  for argument in "$@"; do
    if [[ "${previous}" == "--output-dir" ]]; then
      output="${argument}"
    fi
    if [[ "${argument}" == "--validate-only" ]]; then
      validate="true"
    fi
    previous="${argument}"
  done
  if [[ "${validate}" == "false" ]]; then
    mkdir -p "${output}"
  fi
fi
""",
                encoding="utf-8",
            )
            fake_python.chmod(
                fake_python.stat().st_mode
                | stat.S_IXUSR
                | stat.S_IXGRP
                | stat.S_IXOTH
            )
            fake_sha256sum.write_text(
                f"""#!{sys.executable}
from pathlib import Path
import hashlib
import sys

for raw in sys.argv[1:]:
    path = Path(raw)
    print(f"{{hashlib.sha256(path.read_bytes()).hexdigest()}}  {{path}}")
""",
                encoding="utf-8",
            )
            fake_sha256sum.chmod(
                fake_sha256sum.stat().st_mode
                | stat.S_IXUSR
                | stat.S_IXGRP
                | stat.S_IXOTH
            )

            def sha256(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            environment = dict(os.environ)
            environment.update(
                {
                    "MOTIVE_R10B_SOURCE_SNAPSHOT": str(source_snapshot),
                    "MOTIVE_R10B_SOURCE_TREE_SHA256": "1" * 64,
                    "MOTIVE_R10B_MANIFEST": str(manifest),
                    "MOTIVE_R10B_MANIFEST_SHA256": sha256(manifest),
                    "MOTIVE_R10B_TRACK_CACHE": str(track_cache),
                    "MOTIVE_R10B_TRACK_CACHE_SHA256": sha256(track_cache),
                    "MOTIVE_R10B_MODEL_PATH": str(model_path),
                    "MOTIVE_R10B_MODEL_REVISION": (
                        "ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce"
                    ),
                    "MOTIVE_R10B_MODEL_TREE_SHA256": "2" * 64,
                    "MOTIVE_R10B_BERNINI_REPO": str(bernini_repo),
                    "MOTIVE_R10B_BERNINI_SOURCE_COMMIT": (
                        "2d2b4591ac053ec25c6371b01a5a6746679e5793"
                    ),
                    "MOTIVE_R10B_BERNINI_SOURCE_BUNDLE_SHA256": "3" * 64,
                    "MOTIVE_R10B_PYTHON_BIN": str(fake_python),
                    "FAKE_PYTHON_LOG": str(log),
                    "SLURM_TMPDIR": str(scratch),
                    "SLURM_JOB_ID": "123",
                    "PATH": f"{fake_bin}:{environment.get('PATH', '')}",
                }
            )
            environment.pop("MOTIVE_R10B_MAX_SAMPLES", None)

            output_default = root / "pilot-default"
            environment["MOTIVE_R10B_OUTPUT_DIR"] = str(output_default)
            first = subprocess.run(
                ["bash", str(SCRIPT)],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            calls = log.read_text(encoding="utf-8").splitlines()
            extracts = [
                call
                for call in calls
                if call.startswith(
                    "-m motive.r10b_bernini_tangent_extract "
                )
                and "--validate-only" not in call
            ]
            self.assertEqual(len(extracts), 1)
            extract = extracts[0]
            self.assertIn(
                "--artifact-kind controlled_retrieval_pilot",
                extract,
            )
            self.assertIn(
                "--resize-mode aspect_preserving_center_crop",
                extract,
            )
            self.assertIn("--scheduler-index 25", extract)
            self.assertIn("--noise-mode iid_spatiotemporal", extract)
            self.assertIn("--width 256 --height 256 --num-frames 17", extract)
            self.assertIn("--projection-dim 2048", extract)
            self.assertIn(
                "--projection-seeds 260108851 260108852",
                extract,
            )
            self.assertNotIn("--max-samples", extract)

            log.write_text("", encoding="utf-8")
            output_capped = root / "pilot-capped"
            environment["MOTIVE_R10B_OUTPUT_DIR"] = str(output_capped)
            environment["MOTIVE_R10B_MAX_SAMPLES"] = "7"
            second = subprocess.run(
                ["bash", str(SCRIPT)],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            calls = log.read_text(encoding="utf-8").splitlines()
            extracts = [
                call
                for call in calls
                if call.startswith(
                    "-m motive.r10b_bernini_tangent_extract "
                )
                and "--validate-only" not in call
            ]
            self.assertEqual(len(extracts), 1)
            self.assertIn("--max-samples 7", extracts[0])

            repeat = subprocess.run(
                ["bash", str(SCRIPT)],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(repeat.returncode, 2)
            self.assertIn("refusing to reuse", repeat.stderr)


if __name__ == "__main__":
    unittest.main()
