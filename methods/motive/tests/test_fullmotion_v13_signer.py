from __future__ import annotations

import hashlib
import stat
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SIGNER = REPO_ROOT / "tmp" / "fullmotion_v13_offline_sign_upload.sh"
WATCHER = REPO_ROOT / "tmp" / "run_fullmotion_v13_signer_watch.sh"
SCREENRC = REPO_ROOT / "tmp" / "fullmotion_v13_signer.screenrc"
CHALLENGE_FILE = REPO_ROOT / "tmp" / "fullmotion_v13_release_challenge.txt"
TREE_SHA = "384c98ae656a2896e27f2b2bcef946bf5431f889f9a9d0d403304f3e056e3d77"
CHALLENGE = "109fe68c2ee6deff9d16053d2542f7fffdb839cff045dcaac1148101dc861262"
RUN_ID = "fullmotion128_v13_20260801T195500Z"
SNAPSHOT = "goku-full-motion128-source-v13-20260801T195500Z"
KEY_DIR = "/tmp/motive-fullmotion-release.llVNJy"
KEY_PATH = f"{KEY_DIR}/id_ed25519"
KEY_FINGERPRINT = "SHA256:A6zKKVBr6MSG29PO5J7A91aJYKcORNOkidofuI+jf6Y"
SIGNER_PUBLIC = (
    "ssh-ed25519 "
    "AAAAC3NzaC1lZDI1NTE5AAAAIKe6Q+9i1y9DZE5n6PZNXFJw/YQBEtojl3ClolirGDlO"
)


class FullMotionV13SignerTests(unittest.TestCase):
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

    def test_local_driver_permissions_are_private(self) -> None:
        self.assertEqual(stat.S_IMODE(SIGNER.stat().st_mode), 0o500)
        self.assertEqual(stat.S_IMODE(WATCHER.stat().st_mode), 0o500)
        self.assertEqual(stat.S_IMODE(SCREENRC.stat().st_mode), 0o600)

    def test_signer_is_fully_bound_to_v13(self) -> None:
        for marker in (
            f'bound_tree_sha="{TREE_SHA}"',
            f'bound_release_challenge="{CHALLENGE}"',
            f'local_challenge="${{repo_root}}/tmp/{CHALLENGE_FILE.name}"',
            RUN_ID,
            SNAPSHOT,
            'remote_pool="${remote_run}/final_pool_v13"',
            'release_id="goku-full-motion128-v13-release"',
            'remote_primary="${remote_pool}/primary_256.jsonl"',
            'remote_shards="${remote_run}/production/generation_shards"',
            'remote_release_dir="${remote_run}/production/release"',
            f'local_key_dir="{KEY_DIR}"',
            'local_key="${local_key_dir}/id_ed25519"',
            f'signer_fingerprint="{KEY_FINGERPRINT}"',
            f'signer_public="{SIGNER_PUBLIC}"',
            'expected_qwen_model_metadata_sha="1377f3b975293cd6bcd26c9275826359e9ddd2ab8c982fa9c088c88dc5deda82"',
            'expected_full_input_sha="e4536937d1eb3a065907eb5f6db16b910bea75ff1ec2cdaa17c414ee943c4e42"',
        ):
            self.assertIn(marker, self.signer)
        for stale in (
            "fullmotion128_v10_",
            "source-v10-",
            "final_pool_v10",
            "goku-full-motion128-v10-release",
            "fullmotion128_v11_",
            "source-v11-",
            "final_pool_v11",
            "goku-full-motion128-v11-release",
            "fullmotion128_v12_",
            "source-v12-",
            "final_pool_v12",
            "goku-full-motion128-v12-release",
            "fullmotion128_v9_",
            "source-v9-",
            "final_pool_v9",
            "goku-full-motion128-v9-release",
            "ed4a905ef008ee72347d47564ed8a719d628359ea045a4bf16770ea44e24e237",
            "d181358590b9cbdaa5de9968e1ba3bc798dc983bcdca79ef28768f8c3bcae591",
        ):
            self.assertNotIn(stale, self.signer)
            self.assertNotIn(stale, self.watcher)

    def test_private_key_is_local_only(self) -> None:
        self.assertIn(f'local_key_dir="{KEY_DIR}"', self.signer)
        self.assertIn('local_key="${local_key_dir}/id_ed25519"', self.signer)
        for line in self.signer.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("scp ") or stripped.startswith("ssh "):
                self.assertNotIn("local_key", line)
                self.assertNotIn(KEY_DIR, line)
                self.assertNotIn("id_ed25519", line)
        self.assertNotIn(KEY_PATH, self.watcher)
        self.assertNotIn(KEY_PATH, self.screenrc)

    def test_signing_python_is_isolated_from_launcher_working_directory(self) -> None:
        self.assertIn(
            '"${local_python}" -I -S - "${local_request}" "${local_signed}"',
            self.signer,
        )
        self.assertIn('package = types.ModuleType("motive")', self.signer)
        self.assertIn('package.__path__ = [str(package_root)]', self.signer)
        self.assertIn('sys.modules["motive"] = package', self.signer)
        self.assertNotIn(
            'PYTHONPATH="${local_code_root}" \\\n+  "${local_python}" - ',
            self.signer,
        )
        for marker in (
            '"${remote_python}" -P - ',
            '"${remote_python}" -I "${remote_snapshot_tool}" verify',
            'python3 -I - "${target}"',
        ):
            self.assertIn(marker, self.signer)

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

    def test_watcher_and_screen_log_are_v13_bound(self) -> None:
        self.assertIn("fullmotion_v13_offline_sign_upload.sh", self.watcher)
        self.assertIn(f"tree_sha={TREE_SHA}", self.watcher)
        self.assertIn("--expected-tree-sha256 \"${tree_sha}\" --watch", self.watcher)
        self.assertIn("fullmotion_v13_signer_retry.log", self.screenrc)
        self.assertNotIn("fullmotion_v11", self.screenrc)
        self.assertNotIn("fullmotion_v9", self.screenrc)

    def test_watcher_rehashes_exact_signer_before_every_invocation(self) -> None:
        signer_sha = hashlib.sha256(SIGNER.read_bytes()).hexdigest()
        self.assertIn(f"signer_sha={signer_sha}", self.watcher)
        invocation = self.watcher.index('"${signer_command[@]}" &')
        immediate_check = self.watcher.rindex(
            "if ! verify_signer; then", 0, invocation
        )
        self.assertLess(immediate_check, invocation)
        for marker in (
            '"$(local_digest "${signer}")" == "${signer_sha}"',
            '"$(/usr/bin/stat -f \'%Lp\' "${signer}")" == 500',
            "running watcher bytes differ from the launch authorization",
            "MOTIVE_FULL_MOTION_SIGNER_WATCHER_SHA256",
        ):
            self.assertIn(marker, self.watcher)

    def test_signing_authorization_is_one_persisted_bounded_window(self) -> None:
        for marker in (
            "window_seconds=345600",
            "fullmotion_v13_signer_authorization_window_envclosed_v2.txt",
            "fullmotion_v13_signer_terminal_envclosed_v2.txt",
            "motive-fullmotion-v13-signer-authorization-window-v2",
            "motive-fullmotion-v13-signer-terminal-v2",
            "deadline_epoch=${window_deadline_epoch}",
            "MOTIVE_FULL_MOTION_SIGNING_DEADLINE_EPOCH=",
            'inert_forever "validated terminal=${validated_terminal_status}"',
            'inert_forever "signature published"',
            'inert_forever "authorization deadline expired"',
        ):
            self.assertIn(marker, self.watcher)
        self.assertNotIn("overall_deadline=", self.watcher)
        self.assertIn("check_authorization_deadline", self.signer)
        self.assertGreaterEqual(
            self.signer.count("check_authorization_deadline"), 4
        )
        remote_publish = self.signer.rsplit("<<'REMOTE'", 1)[1]
        self.assertIn('authorization_deadline_epoch="$5"', remote_publish)
        deadline_check = remote_publish.index('"$(date +%s)" -lt')
        hard_link = remote_publish.index('ln -T -- "${stage}" "${target}"')
        self.assertLess(deadline_check, hard_link)

    def test_clean_environment_and_wrong_watcher_hash_fail_before_network(self) -> None:
        clean_env = {
            "HOME": "/Users/siriuschu",
            "USER": "siriuschu",
            "LOGNAME": "siriuschu",
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "SHELL": "/bin/bash",
            "LC_ALL": "C",
            "LANG": "C",
            "MOTIVE_FULL_MOTION_SIGNER_WATCHER_SHA256": "0" * 64,
        }
        wrong_hash = subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", str(WATCHER)],
            check=False,
            capture_output=True,
            text=True,
            env=clean_env,
            timeout=5,
        )
        self.assertEqual(wrong_hash.returncode, 2)
        self.assertIn("watcher bytes differ", wrong_hash.stderr)

        hostile_env = dict(clean_env)
        hostile_env["PYTHONPATH"] = "/tmp/poison"
        hostile = subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", str(WATCHER)],
            check=False,
            capture_output=True,
            text=True,
            env=hostile_env,
            timeout=5,
        )
        self.assertEqual(hostile.returncode, 2)
        self.assertIn("inherited environment is not closed", hostile.stderr)

        expired_env = dict(clean_env)
        expired_env.pop("MOTIVE_FULL_MOTION_SIGNER_WATCHER_SHA256")
        expired_env["MOTIVE_FULL_MOTION_SIGNING_DEADLINE_EPOCH"] = "1"
        expired = subprocess.run(
            [
                "/bin/bash",
                "--noprofile",
                "--norc",
                str(SIGNER),
                "--expected-tree-sha256",
                TREE_SHA,
                "--watch",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=expired_env,
            timeout=5,
        )
        self.assertEqual(expired.returncode, 2)
        self.assertIn("authorization deadline expired", expired.stderr)


if __name__ == "__main__":
    unittest.main()
