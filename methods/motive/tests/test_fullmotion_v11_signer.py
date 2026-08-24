from __future__ import annotations

import stat
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SIGNER = REPO_ROOT / "tmp" / "fullmotion_v11_offline_sign_upload.sh"
WATCHER = REPO_ROOT / "tmp" / "run_fullmotion_v11_signer_watch.sh"
SCREENRC = REPO_ROOT / "tmp" / "fullmotion_v11_signer.screenrc"
CHALLENGE_FILE = REPO_ROOT / "tmp" / "fullmotion_v11_release_challenge.txt"
TREE_SHA = "097789726e19c6942e1f2038ec7f6ea541b7aca6f079e4015afa2d2562ec13db"
CHALLENGE = "69832060de16bd655a94c293e32c63e6c757cbedd88f6a210e133fe8775130f4"
RUN_ID = "fullmotion128_v11_20260801T164500Z"
SNAPSHOT = "goku-full-motion128-source-v11-20260801T164500Z"


class FullMotionV11SignerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.signer = SIGNER.read_text(encoding="utf-8")
        self.watcher = WATCHER.read_text(encoding="utf-8")
        self.screenrc = SCREENRC.read_text(encoding="utf-8")

    def test_shell_files_have_valid_bash_syntax(self) -> None:
        for script in (SIGNER, WATCHER):
            completed = subprocess.run(
                ["bash", "-n", str(script)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_challenge_is_exact_and_private(self) -> None:
        self.assertFalse(CHALLENGE_FILE.is_symlink())
        self.assertEqual(
            CHALLENGE_FILE.read_bytes(), (CHALLENGE + "\n").encode("ascii")
        )
        self.assertEqual(stat.S_IMODE(CHALLENGE_FILE.stat().st_mode), 0o600)

    def test_signer_is_fully_bound_to_v11(self) -> None:
        for marker in (
            f'bound_tree_sha="{TREE_SHA}"',
            f'bound_release_challenge="{CHALLENGE}"',
            f'local_challenge="${{repo_root}}/tmp/{CHALLENGE_FILE.name}"',
            RUN_ID,
            SNAPSHOT,
            'remote_pool="${remote_run}/final_pool_v11"',
            'release_id="goku-full-motion128-v11-release"',
            'remote_primary="${remote_pool}/primary_256.jsonl"',
            'remote_shards="${remote_run}/production/generation_shards"',
            'remote_release_dir="${remote_run}/production/release"',
        ):
            self.assertIn(marker, self.signer)
        for stale in (
            "fullmotion128_v9_",
            "source-v9-",
            "final_pool_v9",
            "goku-full-motion128-v9-release",
            "ed4a905ef008ee72347d47564ed8a719d628359ea045a4bf16770ea44e24e237",
            "d181358590b9cbdaa5de9968e1ba3bc798dc983bcdca79ef28768f8c3bcae591",
        ):
            self.assertNotIn(stale, self.signer)
            self.assertNotIn(stale, self.watcher)

    def test_ssh_disables_control_master_and_has_short_connect_timeout(self) -> None:
        options = self.signer.split("ssh_options=(", 1)[1].split("\n)", 1)[0]
        self.assertIn("-o ControlMaster=no", options)
        self.assertIn("-o ConnectTimeout=10", options)
        self.assertNotIn("ConnectTimeout=15", options)

    def test_signer_keeps_fail_closed_create_only_publication(self) -> None:
        for marker in (
            '"$(stat -c \'%a\' "${directory}")" == 700',
            '[[ ! -e "${signed}" && ! -L "${signed}" ]]',
            '[[ ! -e "${target}" && ! -L "${target}" ]]',
            'ln -T -- "${stage}" "${target}"',
            "verify_signed_release",
            "verify_media=True",
            "if b\"\".join(raw_parts) != primary_raw:",
        ):
            self.assertIn(marker, self.signer)

    def test_watcher_and_screen_log_are_v11_bound(self) -> None:
        self.assertIn("fullmotion_v11_offline_sign_upload.sh", self.watcher)
        self.assertIn(f"tree_sha={TREE_SHA}", self.watcher)
        self.assertIn("--expected-tree-sha256 \"${tree_sha}\" --watch", self.watcher)
        self.assertIn("fullmotion_v11_signer_retry.log", self.screenrc)
        self.assertNotIn("fullmotion_v9", self.screenrc)


if __name__ == "__main__":
    unittest.main()
