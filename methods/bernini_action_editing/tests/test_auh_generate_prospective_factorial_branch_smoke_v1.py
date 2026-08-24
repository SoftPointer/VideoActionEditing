from __future__ import annotations

from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "auh_generate_prospective_factorial_branch_smoke_v1.sbatch"
)


class ProspectiveFactorialBranchSmokeLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_default_python_is_the_plain_environment_executable(self) -> None:
        self.assertIn(
            "anaconda3/envs/vace/bin/python3.12}",
            self.source,
        )
        self.assertNotIn("anaconda3/envs/vace/bin/python}", self.source)

    def test_launcher_rejects_a_python_symlink_before_creating_output(self) -> None:
        validation = (
            '[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]]'
        )
        self.assertIn(validation, self.source)
        self.assertLess(
            self.source.index(validation),
            self.source.index('mkdir "${output_root}"'),
        )


if __name__ == "__main__":
    unittest.main()
