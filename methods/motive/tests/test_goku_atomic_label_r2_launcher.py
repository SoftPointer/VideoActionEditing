from __future__ import annotations

import re
from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "tmp" / "launch_goku_atomic_label_r2_smoke_existing_job.sh"


def embedded_python_blocks(text: str) -> dict[str, str]:
    """Return every single-quoted PY_* heredoc in a shell launcher."""

    lines = text.splitlines()
    blocks: dict[str, str] = {}
    for index, line in enumerate(lines):
        match = re.search(r"<<'(?P<marker>PY_[A-Z0-9_]+)'", line)
        if match is None:
            continue
        marker = match.group("marker")
        try:
            end = lines.index(marker, index + 1)
        except ValueError as error:  # pragma: no cover - bash -n also catches it
            raise AssertionError(f"unterminated heredoc {marker}") from error
        if marker in blocks:
            raise AssertionError(f"duplicate heredoc marker {marker}")
        blocks[marker] = "\n".join(lines[index + 1 : end]) + "\n"
    return blocks


def block_containing(blocks: dict[str, str], *needles: str) -> tuple[str, str]:
    matches = [
        (marker, source)
        for marker, source in blocks.items()
        if all(needle in source for needle in needles)
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one embedded Python block containing {needles!r}; "
            f"found {[marker for marker, _ in matches]}"
        )
    return matches[0]


class GokuAtomicLabelR2LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = LAUNCHER.read_text(encoding="utf-8")
        cls.blocks = embedded_python_blocks(cls.text)

    def test_shell_syntax_and_every_embedded_python_block_compile(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.blocks, "launcher has no auditable embedded helpers")
        for marker, source in self.blocks.items():
            with self.subTest(marker=marker):
                compile(source, f"atomic-label-r2-{marker}", "exec")

    def test_missing_explicit_bindings_fail_before_slurm_or_filesystem_work(self) -> None:
        result = subprocess.run(
            ["/bin/bash", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing explicit binding", result.stderr)
        for forbidden in ("scontrol", "squeue", "srun", "mkdir"):
            self.assertNotIn(forbidden, result.stderr)

    def test_source_planner_contract_is_frozen_and_separate_from_new_run(self) -> None:
        for marker in (
            "MOTIVE_ATOMIC_LABEL_R2_RUN_ROOT",
            "MOTIVE_ATOMIC_LABEL_R2_PLANNER_ROOT",
            "MOTIVE_ATOMIC_LABEL_R2_CANDIDATES",
            "MOTIVE_ATOMIC_LABEL_R2_CANDIDATES_SHA256",
            "MOTIVE_ATOMIC_LABEL_R2_EXPECTED_CANDIDATES:-1235",
            "MOTIVE_ATOMIC_LABEL_R2_EXPECTED_PLANNER_TERMINALS",
        ):
            self.assertIn(marker, self.text)
        # A lexical path comparison is not enough: the launcher must resolve the
        # two roots and reject equal/nested destinations before creating output.
        source_guard = "\n".join(self.blocks.values())
        self.assertRegex(source_guard, r"resolve\(")
        self.assertRegex(source_guard.lower(), r"(overlap|inside|nested|ancestor|separate)")
        self.assertIn("planner_root", source_guard)
        self.assertIn("run_root", source_guard)

    def test_explicit_planner_terminal_prefix_is_replayed_in_parent_order(self) -> None:
        marker, closure = block_containing(
            self.blocks, "_validate_terminal_receipt", "object_sha256"
        )
        # The old planner run is intentionally incomplete.  1,235 freezes the
        # parent candidate pool only; a separate explicit N freezes the closed
        # terminal prefix that may be reused.
        self.assertIn("1235", self.text)
        self.assertRegex(
            self.text,
            r"expected_planner_terminals.*(?:-ge|>=|<)\s*1|"
            r"planner_terminal.*positive",
        )
        self.assertIn("validate_input_row", closure)
        self.assertRegex(closure, r"rows\s*\[:\s*expected.*terminal")
        self.assertRegex(closure, r"len\(rows\).*expected.*candidate")
        self.assertIn("input_digest=", closure)
        self.assertIn("planner.object_sha256", closure)
        self.assertIn("planner._validate_terminal_receipt", closure)
        # _validate_terminal_receipt is the planner's cryptographic replay: it
        # validates the result file hash, optional passed hash, and receipt digest.
        # The launcher must retain these bindings in its own immutable closure.
        for field in (
            "result_sha256",
            "passed_sha256",
            "receipt_digest",
            "input_digest",
        ):
            with self.subTest(marker=marker, field=field):
                self.assertIn(field, closure)
        self.assertRegex(closure, r"records\.append|records\s*=|terminals\.append")
        self.assertRegex(closure, r"status.*ok")
        # A missing terminal inside the prefix is fatal, while rows after N are
        # neither required nor eligible.  The closure itself binds the boundary.
        self.assertRegex(closure, r"expected.*terminal")
        self.assertRegex(closure, r"prefix|boundary|terminal_count")
        self.assertNotRegex(closure, r"for\s+.*\s+in\s+rows\s*:\s*$")

    def test_planner_pass_materialization_is_all_passes_not_a_smoke_prefix(self) -> None:
        # Materialization happens only after the explicit N-row prefix replay
        # and consumes every OK planner pass within that prefix.  exact8
        # selection is later; rows after N are ineligible even if files exist.
        closure_call = self.text.index("_validate_terminal_receipt")
        materialize = self.text.index(" materialize ", closure_call)
        atomic_run = self.text.index('run --input', materialize)
        verify = self.text.index('verify --input', atomic_run)
        exact8 = min(
            position
            for token in ("exact8", "EXACT8")
            if (position := self.text.find(token, verify)) >= 0
        )
        self.assertLess(closure_call, materialize)
        self.assertLess(materialize, atomic_run)
        self.assertLess(atomic_run, verify)
        self.assertLess(verify, exact8)
        materialization_region = self.text[materialize:atomic_run]
        self.assertIn("--expected-passed", materialization_region)
        self.assertRegex(materialization_region, r"planner.*ok|ok.*planner")
        self.assertNotIn("[:8]", materialization_region)
        self.assertNotIn("smoke8.jsonl", self.text[atomic_run:verify])

    def test_fixed_atomic_label_uses_exactly_two_rocr_only_four_gpu_workers(self) -> None:
        for marker in (
            "motive.goku_atomic_motion_instruction",
            '--ntasks=2',
            '--ntasks-per-node=2',
            '--gpus-per-task=4',
            '--gpu-bind=none',
            'visible=0,1,2,3',
            'visible=4,5,6,7',
            'unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES',
            'export ROCR_VISIBLE_DEVICES="${visible}"',
            '--worker-index "${SLURM_LOCALID}"',
            '--num-workers 2',
            "dual4 ROCm probe failed",
            "Qwen3-VL-32B-Instruct",
        ):
            self.assertIn(marker, self.text)
        self.assertNotIn(
            'export HIP_VISIBLE_DEVICES="${visible}" ROCR_VISIBLE_DEVICES="${visible}"',
            self.text,
        )
        self.assertNotRegex(self.text, r"export\s+CUDA_VISIBLE_DEVICES=.*visible")

    def test_atomic_verify_precedes_parent_order_exact8_publication(self) -> None:
        verify_position = self.text.index('verify --input')
        publish_position = min(
            position
            for token in ("exact8_manifest", "exact8-manifest", "atomic_smoke8_manifest")
            if (position := self.text.find(token, verify_position)) >= 0
        )
        self.assertLess(verify_position, publish_position)
        publication = self.text[publish_position:]
        self.assertRegex(publication, r"(first|\[:8\]|target_ok.*8|exact.*8)")
        self.assertRegex(publication, r"(parent|original_candidate_index)")
        self.assertRegex(publication, r"manifest.*sha256|sha256.*manifest")
        self.assertRegex(publication, r"selected.*iid|iid.*selected")
        self.assertRegex(publication, r"len\(.*\).*8|==\s*8|target.*8")

    def test_pool_exhaustion_publishes_an_explicit_insufficient_receipt(self) -> None:
        combined = "\n".join(self.blocks.values())
        self.assertRegex(combined.lower(), r"insufficient")
        self.assertRegex(combined, r"ok_rows|available.*passes|observed.*passes")
        self.assertRegex(combined, r"required.*8|target.*8")
        self.assertRegex(combined, r"status")
        self.assertRegex(
            self.text,
            r"insufficient[^\n]*(receipt|gate)|(?:receipt|gate)[^\n]*insufficient",
        )
        # Insufficient data must be terminal evidence, never a short success gate.
        self.assertNotRegex(combined, r"(?:rows|passes)\s*\[:\s*min\(8")

    def test_new_outputs_are_create_only_and_resume_is_fail_closed(self) -> None:
        for marker in (
            "MOTIVE_ATOMIC_LABEL_R2_RESUME",
            "create-only",
            "controller.lock",
            "flock -n",
            "another",
            "existing",
            "differs",
            "launches",
            ".pid",
            "trap '' HUP",
            "</dev/null",
        ):
            self.assertIn(marker, self.text)
        self.assertRegex(self.text, r"resume.*==\s*0|resume.*== 0")
        self.assertRegex(self.text, r"! -e .*run_root|run_root.*exists")
        self.assertNotIn('mkdir -p "${run_root}"', self.text)
        self.assertNotRegex(self.text, r'>\s*"?\$\{?atomic_manifest')
        self.assertNotRegex(self.text, r'>\s*"?\$\{?smoke_gate')

    def test_old_planner_tree_has_before_after_read_only_closure(self) -> None:
        combined = "\n".join(self.blocks.values())
        # Resume safety alone is insufficient: source evidence is checked again
        # after the Qwen run, so an accidental/concurrent mutation fails closed.
        self.assertRegex(combined.lower(), r"planner.*(before|preflight)")
        self.assertRegex(combined.lower(), r"planner.*(after|postflight|unchanged)")
        self.assertRegex(combined, r"sha256")
        self.assertRegex(self.text.lower(), r"planner.*changed|planner.*differs")

    def test_launcher_has_no_wan_remote_or_allocation_mutation_path(self) -> None:
        lowered = self.text.lower()
        for forbidden in (
            "wan22",
            "wan_generation",
            "preview.mp4",
            "frame-num",
            "sbatch",
            "scancel",
            "scontrol cancel",
            "ssh ",
            "scp ",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
