from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "auh_r7_expansion_dino_edges.sbatch"
)
SUBMIT = (
    Path(__file__).parents[1]
    / "scripts"
    / "auh_submit_r7_expansion_dino_edges.sh"
)
MODULE = (
    Path(__file__).parents[1]
    / "motive"
    / "r7_expansion_dino_edges.py"
)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


class _SubmitFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.snapshot = root / "snapshot"
        self.graph = root / "graph"
        self.output = root / "edges"
        self.logs = root / "logs"
        self.sbatch_log = root / "sbatch.log"
        for relative in (
            "SOURCE_FILES.jsonl",
            "methods/motive/motive/r7_expansion_dino_edges.py",
            "methods/motive/motive/r7_dino_quotient_calibration.py",
            "methods/motive/motive/r7_visual_graph_input.py",
            "methods/motive/scripts/action_source_snapshot.py",
            "methods/motive/scripts/auh_r7_expansion_dino_edges.sbatch",
        ):
            path = self.snapshot / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
        self.graph.mkdir()
        for name in (
            "manifest.jsonl",
            "features.npz",
            "summary.json",
            "done.json",
        ):
            (self.graph / name).write_text("fixture\n", encoding="utf-8")
        self.python = root / "python"
        _write_executable(
            self.python,
            f"""#!/usr/bin/env bash
set -Eeuo pipefail
if [[ -n "${{FAKE_PYTHON_CWD_LOG:-}}" ]]; then
  printf '%s\\n' "${{PWD}}" >> "${{FAKE_PYTHON_CWD_LOG}}"
fi
if [[ "${{1:-}}" == "-c" ]]; then
  printf '%s\\n' "{SHA_C}"
fi
exit 0
""",
        )
        self.bin_directory = root / "bin"
        self.bin_directory.mkdir()
        self.sbatch = self.bin_directory / "sbatch"
        _write_executable(
            self.sbatch,
            """#!/usr/bin/env bash
set -Eeuo pipefail
printf '%s\\n' "$*" >> "${FAKE_SBATCH_LOG:?}"
printf '%s\\n' "424242"
""",
        )

    def environment(self) -> dict[str, str]:
        inherited = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith("SBATCH_")
        }
        return {
            **inherited,
            "PATH": (
                f"{self.bin_directory}:"
                f"{inherited.get('PATH', '')}"
            ),
            "MOTIVE_SOURCE_SNAPSHOT": str(self.snapshot),
            "MOTIVE_SOURCE_TREE_SHA256": SHA_A,
            "MOTIVE_R7_GRAPH_INPUT": str(self.graph),
            "MOTIVE_R7_GRAPH_INPUT_DIGEST": SHA_B,
            "MOTIVE_R7_DINO_EDGE_OUTPUT": str(self.output),
            "MOTIVE_R7_DINO_LOG_DIR": str(self.logs),
            "MOTIVE_R7_DINO_BLOCK_SIZE": "256",
            "MOTIVE_R7_DINO_AUDIT_TOP_K": "20",
            "MOTIVE_R7_DINO_CALIBRATION_PER_STRATUM": "256",
            "PYTHON_BIN": str(self.python),
            "FAKE_SBATCH_LOG": str(self.sbatch_log),
        }

    def run_submit(
        self,
        *,
        overrides: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = self.environment()
        if overrides:
            environment.update(overrides)
        return subprocess.run(
            [str(SUBMIT)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            cwd=cwd,
        )

    def prepare_compute_reservation(self) -> dict[str, str]:
        reservation = (
            self.output / ".r7-dino-edge-submission.lock"
        )
        reservation.mkdir(parents=True)
        (reservation / "token").write_text(
            SHA_C + "\n",
            encoding="utf-8",
        )
        invocation_log = self.root / "python-invoked"
        _write_executable(
            self.python,
            """#!/usr/bin/env bash
set -Eeuo pipefail
printf 'invoked\\n' >> "${FAKE_PYTHON_LOG:?}"
exit 97
""",
        )
        slurm_tmp = self.root / "slurm-tmp"
        slurm_tmp.mkdir()
        return {
            **self.environment(),
            "MOTIVE_SOURCE_SNAPSHOT":
                str(self.snapshot.resolve()),
            "MOTIVE_R7_GRAPH_INPUT":
                str(self.graph.resolve()),
            "MOTIVE_R7_DINO_EDGE_OUTPUT":
                str(self.output.resolve()),
            "PYTHON_BIN": str(self.python.resolve()),
            "MOTIVE_R7_DINO_SUBMISSION_TOKEN": SHA_C,
            "SLURM_JOB_ID": "99",
            "SLURM_TMPDIR": str(slurm_tmp.resolve()),
            "FAKE_PYTHON_LOG": str(invocation_log),
        }


class R7ExpansionDinoOrchestrationTests(unittest.TestCase):
    def test_script_has_valid_bash_syntax(self) -> None:
        for path in (SCRIPT, SUBMIT):
            with self.subTest(path=path.name):
                completed = subprocess.run(
                    ["bash", "-n", str(path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    completed.stderr,
                )

    def test_script_is_exact_eight_gpu_fresh_only_and_hash_bound(
        self,
    ) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        submit_text = SUBMIT.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:mi210:8", text)
        self.assertIn("--nproc_per_node=8", text)
        self.assertIn("expected 8 visible GPUs", text)
        self.assertIn("MOTIVE_R7_GRAPH_INPUT_DIGEST", text)
        self.assertIn("validate_graph_input", text)
        self.assertIn("MOTIVE_R7_DINO_SUBMISSION_TOKEN", text)
        self.assertIn('[[ -e "${final_path}" || -L "${final_path}" ]]', text)
        self.assertIn("-mindepth 1", text)
        self.assertIn("shards directory is not empty", text)
        self.assertNotIn("-name 'rank-?????-of-00008'", text)
        self.assertNotIn("--resume", text)
        self.assertIn("observed_compared_pairs", text)
        self.assertIn("complete_upper_triangle", text)
        self.assertIn(
            "MOTIVE_R7_DINO_CALIBRATION_PER_STRATUM",
            text,
        )
        self.assertIn("--calibration-per-stratum", text)
        self.assertIn("calibration_population_complete", text)
        self.assertIn(
            "r7_dino_quotient_calibration.py",
            text,
        )
        self.assertIn("quotient_partials_per_iid_pair", text)
        self.assertIn("iid_pair_maxima_rows", text)
        self.assertNotIn("DISABLE_QUOTIENT", text)
        module_text = MODULE.read_text(encoding="utf-8")
        self.assertIn("PROGRESS_OWNED_ASSET_INTERVAL = 128", module_text)
        self.assertIn("[r7-dino-edge-progress]", module_text)
        self.assertIn("action_source_snapshot.py", text)
        self.assertNotIn("r7_preflight_extract", text)
        self.assertNotIn("CoTracker", text)
        for resource in (
            '--partition="${slurm_partition}"',
            '--account="${slurm_account}"',
            '--qos="${slurm_qos}"',
            '--nodes="${slurm_nodes}"',
            '--ntasks="${slurm_ntasks}"',
            '--cpus-per-task="${slurm_cpus_per_task}"',
            '--mem="${slurm_memory}"',
            '--gres="${slurm_gres}"',
            '--time="${slurm_time_limit}"',
            '--exclude="${slurm_exclude}"',
        ):
            self.assertIn(resource, submit_text)
        self.assertIn(
            "reject_exported_sbatch_controls",
            submit_text,
        )
        self.assertNotIn("${SBATCH_BIN", submit_text)
        for script_text in (submit_text, text):
            self.assertIn(
                "actual_module = "
                "Path(module.__file__).resolve(strict=True)",
                script_text,
            )
            self.assertIn(
                'PYTHONPATH="${source_snapshot}/methods/motive"',
                script_text,
            )
            self.assertNotIn(
                'PYTHONPATH="${source_snapshot}/methods/motive:',
                script_text,
            )
        self.assertIn(
            'login_cwd_parent="$(realpath /tmp)"',
            submit_text,
        )
        self.assertIn(
            "motive-r7-dino-submit.XXXXXX",
            submit_text,
        )
        self.assertLess(
            text.index('cd "${cache_root}"'),
            text.index(
                '"${python_bin}" \\\n'
                '  "${source_snapshot}/methods/motive/scripts/'
                'action_source_snapshot.py"'
            ),
        )
        self.assertIn(
            "trap 'forward_signal TERM 143' TERM",
            text,
        )
        self.assertIn(
            'run_checked "${python_bin}" -m '
            "torch.distributed.run",
            text,
        )

    def test_submit_wrapper_preflights_then_reserves_and_submits_once(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _SubmitFixture(Path(temporary))
            completed = fixture.run_submit()
            self.assertEqual(
                completed.returncode,
                0,
                completed.stderr,
            )
            self.assertIn("job_id=424242", completed.stdout)
            calls = fixture.sbatch_log.read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(calls), 1)
            self.assertIn("--parsable", calls[0])
            self.assertIn(
                f"MOTIVE_R7_DINO_SUBMISSION_TOKEN={SHA_C}",
                calls[0],
            )
            self.assertIn(
                "MOTIVE_R7_DINO_CALIBRATION_PER_STRATUM=256",
                calls[0],
            )
            for resource in (
                "--partition=faculty",
                "--account=test-acc",
                "--qos=bgqos",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=32",
                "--mem=256G",
                "--gres=gpu:mi210:8",
                "--time=01:00:00",
                "--exclude=auh7-1b-gpu-185,"
                "auh7-1b-gpu-195,"
                "auh7-1b-gpu-233,"
                "auh7-1b-gpu-318",
            ):
                self.assertIn(resource, calls[0])
            reservation = (
                fixture.output
                / ".r7-dino-edge-submission.lock"
            )
            self.assertEqual(
                (reservation / "token").read_text(
                    encoding="utf-8"
                ),
                SHA_C + "\n",
            )
            self.assertEqual(
                (reservation / "job_id").read_text(
                    encoding="utf-8"
                ),
                "424242\n",
            )
            self.assertEqual(
                (reservation / "token").stat().st_mode & 0o222,
                0,
            )
            self.assertEqual(
                (reservation / "job_id").stat().st_mode & 0o222,
                0,
            )
            self.assertEqual(
                [
                    entry.name
                    for entry in reservation.iterdir()
                    if ".tmp." in entry.name
                ],
                [],
            )

            repeated = fixture.run_submit()
            self.assertEqual(repeated.returncode, 2)
            self.assertIn("already reserved", repeated.stderr)
            self.assertEqual(
                len(
                    fixture.sbatch_log.read_text(
                        encoding="utf-8"
                    ).splitlines()
                ),
                1,
            )

    def test_submit_rejects_every_exported_sbatch_control(
        self,
    ) -> None:
        variables = (
            "SBATCH_PARTITION",
            "SBATCH_ACCOUNT",
            "SBATCH_QOS",
            "SBATCH_GRES",
            "SBATCH_MEM_PER_NODE",
            "SBATCH_EXCLUDE",
            "SBATCH_BIN",
        )
        for variable in variables:
            with (
                self.subTest(variable=variable),
                tempfile.TemporaryDirectory() as temporary,
            ):
                fixture = _SubmitFixture(Path(temporary))
                completed = fixture.run_submit(
                    overrides={variable: "attacker-controlled"},
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(
                    f"exported {variable} is forbidden",
                    completed.stderr,
                )
                self.assertFalse(fixture.sbatch_log.exists())
                self.assertFalse(fixture.output.exists())

    def test_submit_rejects_normalized_mutable_path_overlaps(
        self,
    ) -> None:
        def case_paths(
            fixture: _SubmitFixture,
            case: str,
        ) -> dict[str, str]:
            if case == "log_equals_output":
                return {
                    "MOTIVE_R7_DINO_LOG_DIR":
                        str(fixture.output),
                }
            if case == "log_inside_shards":
                return {
                    "MOTIVE_R7_DINO_LOG_DIR":
                        str(fixture.output / "shards" / "logs"),
                }
            if case == "log_inside_final":
                return {
                    "MOTIVE_R7_DINO_LOG_DIR":
                        str(fixture.output / "final" / "logs"),
                }
            if case == "output_inside_log":
                return {
                    "MOTIVE_R7_DINO_EDGE_OUTPUT":
                        str(fixture.logs / "edges"),
                    "MOTIVE_R7_DINO_LOG_DIR":
                        str(fixture.logs),
                }
            if case == "output_inside_snapshot":
                return {
                    "MOTIVE_R7_DINO_EDGE_OUTPUT":
                        str(fixture.snapshot / "edges"),
                }
            if case == "log_inside_graph":
                return {
                    "MOTIVE_R7_DINO_LOG_DIR":
                        str(fixture.graph / "logs"),
                }
            if case == "output_contains_source":
                return {
                    "MOTIVE_R7_DINO_EDGE_OUTPUT":
                        str(fixture.root),
                    "MOTIVE_R7_DINO_LOG_DIR":
                        str(
                            fixture.root.parent
                            / f"{fixture.root.name}-external-logs"
                        ),
                }
            if case == "output_contains_python":
                container = fixture.root / "python-container"
                container.mkdir()
                nested_python = container / "python"
                _write_executable(
                    nested_python,
                    "#!/usr/bin/env bash\nexit 0\n",
                )
                return {
                    "MOTIVE_R7_DINO_EDGE_OUTPUT":
                        str(container),
                    "PYTHON_BIN": str(nested_python),
                }
            if case == "output_via_snapshot_symlink":
                alias = fixture.root / "snapshot-alias"
                alias.symlink_to(
                    fixture.snapshot,
                    target_is_directory=True,
                )
                return {
                    "MOTIVE_R7_DINO_EDGE_OUTPUT":
                        str(alias / "edges"),
                }
            raise AssertionError(case)

        cases = (
            "log_equals_output",
            "log_inside_shards",
            "log_inside_final",
            "output_inside_log",
            "output_inside_snapshot",
            "log_inside_graph",
            "output_contains_source",
            "output_contains_python",
            "output_via_snapshot_symlink",
        )
        for case in cases:
            with (
                self.subTest(case=case),
                tempfile.TemporaryDirectory() as temporary,
            ):
                fixture = _SubmitFixture(Path(temporary))
                completed = fixture.run_submit(
                    overrides=case_paths(fixture, case),
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(
                    "overlap after normalization",
                    completed.stderr,
                )
                self.assertFalse(fixture.sbatch_log.exists())

    def test_submit_python_runs_only_from_external_isolated_cwd(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _SubmitFixture(Path(temporary))
            shadow = fixture.root / "attacker-cwd"
            (shadow / "motive").mkdir(parents=True)
            (
                shadow / "motive" / "r7_expansion_dino_edges.py"
            ).write_text(
                "raise RuntimeError('cwd shadow imported')\n",
                encoding="utf-8",
            )
            cwd_log = fixture.root / "login-python-cwds.log"
            completed = fixture.run_submit(
                overrides={
                    "FAKE_PYTHON_CWD_LOG": str(cwd_log),
                },
                cwd=shadow,
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stderr,
            )
            observed = cwd_log.read_text(
                encoding="utf-8",
            ).splitlines()
            self.assertGreaterEqual(len(observed), 3)
            self.assertEqual(len(set(observed)), 1)
            isolated = Path(observed[0]).resolve()
            self.assertNotEqual(isolated, shadow.resolve())
            self.assertNotIn(
                fixture.snapshot.resolve(),
                isolated.parents,
            )
            self.assertFalse(isolated.exists())

    def test_submit_wrapper_rejects_dangling_final_and_any_shard_entry(
        self,
    ) -> None:
        cases = ("dangling_final", "noncanonical_file", "shards_symlink")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                fixture = _SubmitFixture(Path(temporary))
                fixture.output.mkdir()
                if case == "dangling_final":
                    (fixture.output / "final").symlink_to(
                        fixture.root / "missing-final",
                    )
                elif case == "noncanonical_file":
                    shards = fixture.output / "shards"
                    shards.mkdir()
                    (shards / "junk.txt").write_text(
                        "stale\n",
                        encoding="utf-8",
                    )
                else:
                    target = fixture.root / "external-shards"
                    target.mkdir()
                    (fixture.output / "shards").symlink_to(target)
                completed = fixture.run_submit()
                self.assertEqual(completed.returncode, 2)
                self.assertFalse(fixture.sbatch_log.exists())

    def test_compute_script_rejects_closed_set_violations_before_python(
        self,
    ) -> None:
        cases = ("dangling_final", "noncanonical_file", "shards_symlink")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                fixture = _SubmitFixture(Path(temporary))
                environment = fixture.prepare_compute_reservation()
                if case == "dangling_final":
                    (fixture.output / "final").symlink_to(
                        fixture.root / "missing-final",
                    )
                elif case == "noncanonical_file":
                    shards = fixture.output / "shards"
                    shards.mkdir()
                    (shards / "rank-00000-of-00008").mkdir()
                else:
                    target = fixture.root / "external-shards"
                    target.mkdir()
                    (fixture.output / "shards").symlink_to(target)
                completed = subprocess.run(
                    ["bash", str(SCRIPT)],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertFalse(
                    Path(environment["FAKE_PYTHON_LOG"]).exists()
                )

    def test_compute_rejects_cache_inside_snapshot_before_python(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _SubmitFixture(Path(temporary))
            environment = fixture.prepare_compute_reservation()
            environment["SLURM_TMPDIR"] = str(fixture.snapshot)
            completed = subprocess.run(
                ["bash", str(SCRIPT)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn(
                "compute cache and source snapshot overlap",
                completed.stderr,
            )
            self.assertFalse(
                Path(environment["FAKE_PYTHON_LOG"]).exists()
            )
            self.assertFalse(
                (
                    fixture.snapshot
                    / "motive-r7-dino-edges-99"
                ).exists()
            )

    def test_compute_isolated_cwd_and_torchrun_failure_propagate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _SubmitFixture(Path(temporary))
            environment = fixture.prepare_compute_reservation()
            cwd_log = fixture.root / "compute-python-cwds.log"
            finalize_log = fixture.root / "finalize-called.log"
            _write_executable(
                fixture.python,
                f"""#!/usr/bin/env bash
set -Eeuo pipefail
printf '%s\\n' "${{PWD}}" >> "${{FAKE_PYTHON_CWD_LOG:?}}"
if [[ "${{1:-}}" == *action_source_snapshot.py ]]; then
  exit 0
fi
if [[ "${{1:-}}" == "-" ]]; then
  printf '%s\\n' "{SHA_B}"
  exit 0
fi
if [[ "${{1:-}}" == "-c" ]]; then
  printf '%s\\n' "8"
  exit 0
fi
if [[
  "${{1:-}}" == "-m" &&
  "${{2:-}}" == "torch.distributed.run"
]]; then
  exit 23
fi
if [[
  "${{1:-}}" == "-m" &&
  "${{2:-}}" == "motive.r7_expansion_dino_edges"
]]; then
  printf 'called\\n' >> "${{FAKE_FINALIZE_LOG:?}}"
  exit 0
fi
exit 91
""",
            )
            environment["FAKE_PYTHON_CWD_LOG"] = str(cwd_log)
            environment["FAKE_FINALIZE_LOG"] = str(finalize_log)
            completed = subprocess.run(
                ["bash", str(SCRIPT)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                cwd=fixture.snapshot,
            )
            self.assertEqual(
                completed.returncode,
                23,
                completed.stderr,
            )
            self.assertFalse(finalize_log.exists())
            self.assertFalse((fixture.output / "final").exists())
            observed = cwd_log.read_text(
                encoding="utf-8",
            ).splitlines()
            self.assertGreaterEqual(len(observed), 4)
            self.assertEqual(len(set(observed)), 1)
            expected_cwd = (
                Path(environment["SLURM_TMPDIR"])
                / "motive-r7-dino-edges-99"
            ).resolve()
            self.assertEqual(
                Path(observed[0]).resolve(),
                expected_cwd,
            )
            self.assertNotIn(
                fixture.snapshot.resolve(),
                expected_cwd.parents,
            )


if __name__ == "__main__":
    unittest.main()
