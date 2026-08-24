from __future__ import annotations

import subprocess
import stat
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CANARY = REPO_ROOT / "tmp" / "launch_fullmotion_v9_canary_20260801T142000Z.sh"
SMOKE = REPO_ROOT / "tmp" / "launch_fullmotion_v9_smoke_20260801T142000Z.sh"
SUPERVISOR = REPO_ROOT / "tmp" / "launch_wait_fullmotion_v9_job116234.sh"
SIGNER = REPO_ROOT / "tmp" / "fullmotion_v9_offline_sign_upload.sh"
SIGNER_WATCH = REPO_ROOT / "tmp" / "run_fullmotion_v9_signer_watch.sh"
CHALLENGE = REPO_ROOT / "tmp" / "fullmotion_v9_release_challenge.txt"
TREE_SHA = "ed4a905ef008ee72347d47564ed8a719d628359ea045a4bf16770ea44e24e237"
RELEASE_CHALLENGE = "d181358590b9cbdaa5de9968e1ba3bc798dc983bcdca79ef28768f8c3bcae591"


class FullMotionV9LauncherTests(unittest.TestCase):
    def test_launchers_have_valid_bash_syntax(self) -> None:
        for script in (CANARY, SMOKE, SUPERVISOR, SIGNER, SIGNER_WATCH):
            completed = subprocess.run(
                ["bash", "-n", str(script)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"{script}: {completed.stderr}",
            )

    def test_holder_idle_probe_overlaps_without_memory_tres(self) -> None:
        text = SUPERVISOR.read_text(encoding="utf-8")
        function = text.split("check_idle_node() {", 1)[1].split(
            "\n}\n\nfor audit", 1
        )[0]
        self.assertIn("--overlap", function)
        self.assertIn("--mem=0", function)
        self.assertIn("--exact", function)
        self.assertNotIn("--exclusive", function)
        self.assertNotIn("--mem=1G", function)

    def test_smoke_launcher_validates_and_seals_raw_gate(self) -> None:
        text = SMOKE.read_text(encoding="utf-8")
        for marker in (
            "motive-full-motion-v9-smoke-sealer",
            "PYTHONOPTIMIZE=",
            "motive-goku-full-motion-qwen-smoke-gate-v4",
            "motive-goku-full-motion-qwen-smoke-gate-failure-v1",
            "object_sha256(payload)",
            "authorizes_full_run",
            'chmod 0400 "${gate}"',
            '"$(stat -c "%a" "${gate}")" == 400',
            'sealed_sha=$(sha256sum "${gate}")',
            'sealed_sha=${sealed_sha%% *}',
        ):
            self.assertIn(marker, text)

    def test_supervisor_round_trips_postclaim_prelaunch_cancellation(self) -> None:
        text = SUPERVISOR.read_text(encoding="utf-8")
        for marker in (
            "postclaim_cancel_marker=${run}/pipeline_v9_postclaim_prelaunch_cancelled.txt",
            "postclaim_cancel_requested=0",
            "postclaim_cancel_source=\"\"",
            "publish_postclaim_cancel_marker()",
            "validate_postclaim_cancel_marker()",
            "schema=motive-fullmotion128-postclaim-prelaunch-cancel-v1",
            '"claim_owner_token=${claim_owner_token}"',
            "postclaim_cancel_requested=1",
            'postclaim_cancel_source="signal_${signal_name}"',
            '"${supervisor_receipt}" "${postclaim_cancel_marker}"; do',
            "(( pipeline_exit_code == 143 ))",
            "terminal_status=cancelled_after_claim_prelaunch",
            "publish_holder_release",
            '[[ ! -L "${holder_claim}" && -f "${holder_claim}"',
        ):
            self.assertIn(marker, text)
        self.assertLess(
            text.index("validate_postclaim_cancel_marker", text.index("done\n\nif [[ -e")),
            text.index("if (( pipeline_exit_code != 0 ))"),
        )

    def test_offline_signer_handoff_is_bound_and_create_only(self) -> None:
        signer = SIGNER.read_text(encoding="utf-8")
        watcher = SIGNER_WATCH.read_text(encoding="utf-8")
        self.assertEqual(CHALLENGE.read_bytes(), (RELEASE_CHALLENGE + "\n").encode())
        self.assertEqual(stat.S_IMODE(CHALLENGE.stat().st_mode), 0o600)
        for marker in (
            f'bound_tree_sha="{TREE_SHA}"',
            f'bound_release_challenge="{RELEASE_CHALLENGE}"',
            "fullmotion128_v9_20260801T142000Z",
            "goku-full-motion128-source-v9-20260801T142000Z",
            'release_id="goku-full-motion128-v9-release"',
            'remote_primary="${remote_pool}/primary_256.jsonl"',
            "verify_signed_release",
            "verify_media=True",
            '[[ ! -e "${target}" && ! -L "${target}" ]]',
            'ln -T -- "${stage}" "${target}"',
        ):
            self.assertIn(marker, signer)
        self.assertIn(f"tree_sha={TREE_SHA}", watcher)
        for stale in ("fullmotion128_v7_", "source-v7-", "v8r2"):
            self.assertNotIn(stale, signer)
            self.assertNotIn(stale, watcher)


if __name__ == "__main__":
    unittest.main()
