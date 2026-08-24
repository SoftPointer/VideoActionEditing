import hashlib
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
METHOD = ROOT / "methods" / "bernini_action_editing"
LAUNCHER = METHOD / "scripts" / "auh_launch_mev840_legacy_oracle_lowstrength_sweep_v1.sh"
FINALIZER = METHOD / "scripts" / "auh_finalize_mev840_legacy_oracle_lowstrength_sweep_v1.sh"
MANIFEST = METHOD / "assets" / "mev840_legacy_oracle_lowstrength_sweep_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MEV840LegacyOracleLowStrengthSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text())

    def test_shell_syntax_and_pins(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        subprocess.run(["bash", "-n", str(FINALIZER)], check=True)
        self.assertEqual(self.manifest["pins"]["launcher"], sha256(LAUNCHER))
        self.assertEqual(self.manifest["pins"]["login_finalizer"], sha256(FINALIZER))

    def test_exact_factorial_and_node_mapping(self) -> None:
        arms = self.manifest["arms"]
        self.assertEqual(len(arms), 6)
        self.assertEqual(
            {(arm["strength"], arm["transport_steps"]) for arm in arms},
            {(0.05, 5), (0.10, 5), (0.25, 5), (0.05, 10), (0.10, 10), (0.25, 10)},
        )
        self.assertEqual(len({arm["node"] for arm in arms}), 6)
        self.assertEqual(
            {arm["node"] for arm in arms},
            {
                "auh7-1b-gpu-213", "auh7-1b-gpu-284", "auh7-1b-gpu-232",
                "auh7-1b-gpu-268", "auh7-1b-gpu-315", "auh7-1b-gpu-233",
            },
        )

    def test_oracle_zero_update_contract(self) -> None:
        manifest = self.manifest
        self.assertTrue(manifest["authority"]["oracle_anchor"]["real_target_read"])
        self.assertEqual(manifest["authority"]["oracle_anchor"]["role"], "oracle_only")
        self.assertEqual(
            manifest["authority"]["oracle_anchor"]["sha256"],
            "355535f4f5ff83581c2286dfb70a64c7f5131f5ae81d76fbb6351b2aa972baf0",
        )
        invariant = manifest["invariants"]
        self.assertTrue(invariant["zero_update"])
        self.assertFalse(invariant["training_performed"])
        self.assertEqual(invariant["optimization_steps"], 0)
        self.assertFalse(invariant["checkpoint_or_adapter_loaded"])
        self.assertEqual(invariant["initial_noise_proposal_mode"], "keyed_only")
        self.assertFalse(invariant["anchor_gaussian_supplied"])
        self.assertEqual(invariant["activity_keep_fraction"], 0.25)
        self.assertEqual(invariant["source_cfg_scale"], invariant["target_cfg_scale"])

    def test_fail_closed_postflight_split(self) -> None:
        execution = self.manifest["fresh_execution"]
        self.assertFalse(execution["overwrite_allowed"])
        self.assertFalse(execution["parent_job_cancel_allowed"])
        self.assertFalse(execution["compute_postflight_allowed"])
        self.assertTrue(execution["login_postflight_required"])
        launcher_text = LAUNCHER.read_text()
        worker_text = launcher_text.split('if [ "$mode" = launch ]', 1)[1]
        self.assertNotIn("ffprobe", worker_text.split("# Compute nodes deliberately stop before ffprobe", 1)[1])
        self.assertIn("ffprobe", FINALIZER.read_text())


if __name__ == "__main__":
    unittest.main()
