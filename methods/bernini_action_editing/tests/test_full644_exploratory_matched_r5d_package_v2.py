from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = MODULE_ROOT / "tools"
SCRIPTS_ROOT = MODULE_ROOT / "scripts"
for entry in (MODULE_ROOT, TOOLS_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import full644_exploratory_matched_spooled_launcher_auh_r5d as launcher
import materialize_full644_exploratory_matched_r5d_case00_package_v1 as retired
import materialize_full644_exploratory_matched_r5d_case00_package_v2 as candidate


MATERIALIZER = TOOLS_ROOT / (
    "materialize_full644_exploratory_matched_r5d_case00_package_v2.py"
)
CONTROLLER = SCRIPTS_ROOT / (
    "auh_materialize_full644_exploratory_matched_r5d_"
    "job143812_node293_once_v2.sh"
)
RETIRED_CONTROLLER = SCRIPTS_ROOT / (
    "auh_materialize_full644_exploratory_matched_r5d_"
    "job143812_node293_once_v1.sh"
)
EXPECTED_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/"
    "bernini_full644_exploratory_matched_eval_auh_r5d_"
    "job143812_node293_case00_847b91a2_c91de7eb_85ccc17b_r2"
)
RETIRED_ROOT = EXPECTED_ROOT[:-1] + "1"
MATERIALIZER_SHA256 = (
    "79a3fc988bbdcd74abf13823a7046ffba9c984f660e57b0c0678b4412cfa96e6"
)
MATERIALIZER_SIZE = 36169
CONTROLLER_SHA256 = (
    "399c33e2882de6f1f5d05fdeff1db905a2418d39e51213c7a71240c924500171"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class R5DPackageGenerationV2Tests(unittest.TestCase):
    def test_retired_generation_is_unchanged_and_not_reused(self) -> None:
        self.assertEqual(
            sha(TOOLS_ROOT / retired.__file__.rsplit("/", 1)[-1]),
            "896ded9bf9351cc5f5af1eb52b06aa4bf0d9e8d322ffeefef04a908cd92275af",
        )
        self.assertEqual(
            sha(RETIRED_CONTROLLER),
            "318dbe2d01bb41b2e5f1d644753f8d3fcdcd2e1f9d392448246f5205590bae3d",
        )
        key = ("143812", "auh7-1b-gpu-293")
        self.assertEqual(str(retired.TARGETS[key]), RETIRED_ROOT)
        self.assertEqual(set(candidate.TARGETS), {key})
        self.assertEqual(str(candidate.TARGETS[key]), EXPECTED_ROOT)
        self.assertNotIn(RETIRED_ROOT, {str(path) for path in candidate.TARGETS.values()})
        for burned_key in (
            ("143811", "auh7-1b-gpu-306"),
            ("143808", "auh7-1b-gpu-315"),
        ):
            with mock.patch.object(candidate, "mkdir_fresh") as mkdir:
                with self.assertRaisesRegex(
                    candidate.R5DMaterializationError,
                    "unsupported r5d binding target",
                ):
                    candidate._materialize(
                        types.SimpleNamespace(
                            job_id=burned_key[0],
                            node=burned_key[1],
                            source_root=str(MODULE_ROOT.parents[1]),
                        )
                    )
                mkdir.assert_not_called()

    def test_v2_materializer_is_exact_and_preserves_frozen_inputs(self) -> None:
        self.assertEqual(sha(MATERIALIZER), MATERIALIZER_SHA256)
        self.assertEqual(MATERIALIZER.stat().st_size, MATERIALIZER_SIZE)
        self.assertEqual(candidate.RELEASE_FILES, retired.RELEASE_FILES)
        self.assertEqual(
            candidate.DIAGNOSTIC_SOURCE_PINS,
            retired.DIAGNOSTIC_SOURCE_PINS,
        )
        frozen = {
            "full644_exploratory_matched_infer_adapter_auh_r5d.py":
                "5794e1f0e5ecb84ffdb37f618fe63696ee4f87176952ac083c8c91792a9d192a",
            "full644_exploratory_matched_spooled_launcher_auh_r5d.py":
                "85ccc17b30d97a7bf048702cd8a8ed10c3421e01721902fea7db6242eac45753",
            "full644_exploratory_matched_r5d_root_bootstrap_probe_runner_v1.py":
                "e4890e5d45c6a3982bab03f311711effc87efd29718ff8d5726ad4580b8a3845",
            "full644_exploratory_matched_r5d_static_nomodel_probe_v1.py":
                "4b17a1919a6ef928d572f769d6713a4764b0d31dc6da77eb7498cc5152c6de6c",
            "full644_exploratory_matched_r5d_cpu_consumption_probe_v1.py":
                "5c7f5caf5ad73aecacedda618e941308e4fc1b94218b71cdc44e88afc3d3f0ea",
        }
        for name, digest in frozen.items():
            self.assertEqual(sha(MODULE_ROOT / name), digest)

    def test_physical17_identity16_and_exact2_are_distinct_closures(self) -> None:
        self.assertEqual(len(candidate.RELEASE_FILES), 17)
        identity_roles = set(launcher.EXPECTED_STATIC_SHA256) | {
            "python", "ffmpeg", "plan",
        }
        self.assertEqual(len(identity_roles), 16)
        self.assertEqual(
            candidate.SELECTED_TASK_IDS,
            ("shared8-00-base", "shared8-00-full644"),
        )
        self.assertEqual(len(candidate.SELECTED_TASK_IDS), 2)
        self.assertEqual(candidate.CAMPAIGN, launcher.CASE00_CANARY_CAMPAIGN)

    def test_existing_target_refuses_before_source_read_or_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary).resolve() / "already-consumed-r2"
            target.mkdir()
            key = ("143812", "auh7-1b-gpu-293")
            with mock.patch.dict(candidate.TARGETS, {key: target}, clear=True), \
                 mock.patch.object(candidate.os, "geteuid", return_value=2012), \
                 mock.patch.object(candidate.os, "getegid", return_value=2000), \
                 mock.patch.object(candidate, "stable_file") as stable, \
                 mock.patch.object(candidate, "mkdir_fresh") as mkdir:
                with self.assertRaisesRegex(
                    candidate.R5DMaterializationError,
                    "fresh r5d root exists",
                ):
                    candidate._materialize(
                        types.SimpleNamespace(
                            job_id=key[0],
                            node=key[1],
                            source_root=str(MODULE_ROOT.parents[1]),
                        )
                    )
                stable.assert_not_called()
                mkdir.assert_not_called()

    def test_controller_is_held_fd_bound_to_v2_only(self) -> None:
        raw = CONTROLLER.read_bytes()
        source = raw.decode("utf-8", "strict")
        self.assertEqual(hashlib.sha256(raw).hexdigest(), CONTROLLER_SHA256)
        self.assertIn(
            "readonly R5D_TARGET_ROOT=" + EXPECTED_ROOT,
            source,
        )
        self.assertNotIn("readonly R5D_TARGET_ROOT=" + RETIRED_ROOT, source)
        self.assertIn(
            "materialize_full644_exploratory_matched_r5d_case00_package_v2.py",
            source,
        )
        self.assertNotIn(
            "materialize_full644_exploratory_matched_r5d_case00_package_v1.py",
            source,
        )
        self.assertEqual(source.count(MATERIALIZER_SHA256), 2)
        self.assertIn(f",0o644,False,{MATERIALIZER_SIZE})", source)
        self.assertIn('exec {R5D_PYTHON_FD}<"${R5D_PYTHON}"', source)
        self.assertIn('exec {R5D_MATERIALIZER_FD}<"${R5D_MATERIALIZER}"', source)
        self.assertIn('exec -c "/proc/self/fd/${R5D_PYTHON_FD}" -I -S -B -c', source)

    def test_controller_checks_absence_before_and_after_complete_preflight(self) -> None:
        source = CONTROLLER.read_text(encoding="utf-8")
        shell_absent = source.index('[[ ! -e "${R5D_TARGET_ROOT}"')
        first_open = source.index('exec {R5D_PYTHON_FD}<')
        self.assertLess(shell_absent, first_open)
        first = source.index("if os.path.lexists(target_root):")
        specs = source.index("source_specs=(")
        materializer = source.index("materializer_raw=stable_fd(")
        second = source.index("if os.path.lexists(target_root):", first + 1)
        execve = source.index('os.execve("/proc/self/fd/"')
        self.assertLess(first, specs)
        self.assertLess(materializer, second)
        self.assertLess(second, execve)
        self.assertEqual(source.count("if os.path.lexists(target_root):"), 2)

    def test_controller_bash_and_embedded_python_compile_normal_and_optimized(self) -> None:
        completed = subprocess.run(
            ["/bin/bash", "-n", str(CONTROLLER)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        source = CONTROLLER.read_text(encoding="utf-8")
        prefix = "R5D_BOOTSTRAP='"
        suffix = "'\nreadonly R5D_BOOTSTRAP"
        start = source.index(prefix) + len(prefix)
        end = source.index(suffix, start)
        bootstrap = source[start:end]
        for optimize in (0, 2):
            compile(
                bootstrap,
                "<r5d-r2-materialization-bootstrap>",
                "exec",
                dont_inherit=True,
                optimize=optimize,
            )
            compile(
                MATERIALIZER.read_text(encoding="utf-8"),
                str(MATERIALIZER),
                "exec",
                dont_inherit=True,
                optimize=optimize,
            )


if __name__ == "__main__":
    unittest.main()
