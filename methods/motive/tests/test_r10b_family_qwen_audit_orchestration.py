from __future__ import annotations

import ast
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
    / "auh_r10b_family_qwen_audit.sbatch"
)


class R10BFamilyQwenAuditOrchestrationTests(unittest.TestCase):
    def test_resources_and_closed_audit_contract_are_explicit(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for directive in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=16",
            "#SBATCH --mem=128G",
            "#SBATCH --gres=gpu:mi210:1",
            "#SBATCH --time=02:00:00",
            "#SBATCH --job-name=motive-r10b-qwen-audit",
        ):
            self.assertIn(directive, text)

        for marker in (
            "MOTIVE_R10B_SOURCE_SNAPSHOT",
            "MOTIVE_R10B_SOURCE_TREE_SHA256",
            "MOTIVE_R10B_QUEUE_DIR",
            "MOTIVE_R10B_DATA_ROOT",
            "MOTIVE_R10B_QWEN_MODEL",
            "MOTIVE_R10B_OUTPUT_DIR",
            "MOTIVE_R10B_PYTHON_BIN",
            'nframes="12"',
            'max_pixels="589824"',
            'max_new_tokens="512"',
            'attn_implementation="sdpa"',
            'visual_input="mosaic"',
            "HF_HUB_OFFLINE=1",
            "HF_DATASETS_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "WANDB_DISABLED=true",
            'verify_bound_inputs "before"',
            'verify_bound_inputs "after"',
            "--queue-dir",
            "--data-root",
            "--model",
            "--output-dir",
            "--nframes",
            "--max-pixels",
            "--max-new-tokens",
            "--attn-implementation",
            "--validate-only",
            "videos_copied",
            "videos_rendered",
            "optimizer_steps",
            "representation_gate_passed",
            "renderer_probe_authorized",
            "generation_authorized",
            "training_authorized",
        ):
            self.assertIn(marker, text)

        self.assertEqual(
            text.count("-m motive.r10b_family_qwen_audit"),
            2,
        )
        for forbidden in (
            "ffmpeg",
            "rsync",
            "torchrun",
            "deepspeed",
            "srun",
            "sbatch",
        ):
            self.assertNotIn(forbidden, text)
        self.assertNotRegex(text, r"(?m)^\s*(?:cp|scp)\s")
        self.assertNotRegex(text, r"(?i)\b(?:train|optimizer)\w*\s*\(")

    def test_all_external_paths_and_source_digest_are_mandatory(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        required = (
            "MOTIVE_R10B_SOURCE_SNAPSHOT",
            "MOTIVE_R10B_SOURCE_TREE_SHA256",
            "MOTIVE_R10B_QUEUE_DIR",
            "MOTIVE_R10B_DATA_ROOT",
            "MOTIVE_R10B_QWEN_MODEL",
            "MOTIVE_R10B_OUTPUT_DIR",
            "MOTIVE_R10B_PYTHON_BIN",
        )
        for name in required:
            self.assertRegex(
                text,
                rf'\$\{{{name}:\?set {name}\}}',
            )
        self.assertNotIn("MOTIVE_R10B_PYTHON_BIN:-", text)
        self.assertIn("action_source_snapshot.py", text)
        self.assertIn("--expected-tree-sha256", text)
        self.assertIn("_load_queue_commit", text)
        self.assertIn("_resolve_media", text)
        self.assertIn("PROMPT_CONTRACT_SHA256", text)
        self.assertIn("model.name != revision", text)
        self.assertIn('[[ -e "${output_dir}" || -L "${output_dir}" ]]', text)

    def test_script_and_embedded_python_have_valid_syntax(self) -> None:
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

    def test_fake_run_forwards_fixed_contract_and_refuses_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            snapshot = root / "snapshot"
            queue = root / "queue"
            data = root / "data"
            model = root / ("c" * 40)
            output = root / "audit"
            scratch = root / "scratch"
            fake_python = root / "fake-python"
            log = root / "python.log"
            for path in (snapshot, queue, data, model, scratch):
                path.mkdir()
            fake_python.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${FAKE_PYTHON_LOG:?}"
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  exit 0
fi
if [[ "${1:-}" == "-m" \
  && "${2:-}" == "motive.r10b_family_qwen_audit" ]]; then
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
            environment = dict(os.environ)
            environment.update(
                {
                    "MOTIVE_R10B_SOURCE_SNAPSHOT": str(snapshot),
                    "MOTIVE_R10B_SOURCE_TREE_SHA256": "a" * 64,
                    "MOTIVE_R10B_QUEUE_DIR": str(queue),
                    "MOTIVE_R10B_DATA_ROOT": str(data),
                    "MOTIVE_R10B_QWEN_MODEL": str(model),
                    "MOTIVE_R10B_OUTPUT_DIR": str(output),
                    "MOTIVE_R10B_PYTHON_BIN": str(fake_python),
                    "FAKE_PYTHON_LOG": str(log),
                    "SLURM_TMPDIR": str(scratch),
                    "SLURM_JOB_ID": "456",
                }
            )
            completed = subprocess.run(
                ["bash", str(SCRIPT)],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            calls = log.read_text(encoding="utf-8").splitlines()
            audits = [
                call
                for call in calls
                if call.startswith("-m motive.r10b_family_qwen_audit ")
            ]
            self.assertEqual(len(audits), 2)
            run = next(call for call in audits if "--validate-only" not in call)
            validate = next(
                call for call in audits if "--validate-only" in call
            )
            self.assertIn(f"--queue-dir {queue}", run)
            self.assertIn(f"--data-root {data}", run)
            self.assertIn(f"--model {model}", run)
            self.assertIn(f"--output-dir {output}", run)
            self.assertIn("--nframes 12", run)
            self.assertIn("--max-pixels 589824", run)
            self.assertIn("--max-new-tokens 512", run)
            self.assertIn("--attn-implementation sdpa", run)
            self.assertEqual(
                validate,
                (
                    "-m motive.r10b_family_qwen_audit "
                    f"--output-dir {output} --validate-only"
                ),
            )

            repeated = subprocess.run(
                ["bash", str(SCRIPT)],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(repeated.returncode, 2)
            self.assertIn("refusing to reuse", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
