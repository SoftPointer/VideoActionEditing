from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    REPO_ROOT
    / "methods"
    / "motive"
    / "scripts"
    / "watch_auh_r9_jobs.sh"
)


class R9WatcherScriptTests(unittest.TestCase):
    def test_script_has_valid_bash_syntax(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_watcher_checks_processes_and_commits(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for marker in (
            "sacct -j",
            "srun --overlap",
            "validate_published_search",
            "[motive-r9-representation-search]",
            "[motive-r7-candidate-temporal-screen]",
            "[r9-controller] completed",
            "editor_training_authorized",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("scp ", text)
        self.assertNotIn("rsync ", text)


if __name__ == "__main__":
    unittest.main()
