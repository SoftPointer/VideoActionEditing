#!/usr/bin/env python3

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "auh_materialize_starc_core4_hidden_v1_dual4.sbatch"


class AUHSTARCHiddenV1LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_bash_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_eight_gpus_are_two_concurrent_sp4_groups(self) -> None:
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.source)
        self.assertIn('run_group sp4-a "0,1,2,3"', self.source)
        self.assertIn('run_group sp4-b "4,5,6,7"', self.source)
        self.assertEqual(self.source.count("--nproc_per_node=4"), 1)
        self.assertIn('>"${output_root}/logs/sp4-a.log" 2>&1 &', self.source)
        self.assertIn('>"${output_root}/logs/sp4-b.log" 2>&1 &', self.source)

    def test_only_new_dynamic_geometry_materializer_is_called(self) -> None:
        self.assertIn("materialize_starc_core4_hidden_v1.py", self.source)
        self.assertNotIn("materialize_latent_temporal_event_critic_core4.py", self.source)
        self.assertIn("author_pair_v5_core4_event_labels_d541801_v3.py", self.source)
        self.assertNotIn('tools/author_pair_v5_core4_event_labels_v3.py"', self.source)
        self.assertIn("P=930x2,928x1,918x1", self.source)
        self.assertIn("positions != [918, 928, 930, 930]", self.source)
        self.assertIn("aggregate-master", self.source)

    def test_archive_and_all_authorities_are_hash_bound(self) -> None:
        for token in (
            "git get-tar-commit-id",
            "running launcher differs from source archive",
            "d541801a162796aacde34c2bfc2b1f0472d954d2",
            "535aba9b5445e2a9b06cf3da267325c49d247ab2ef9f4a9dd129a51fdbb008c7",
            "3d7ce459ddb9a014873acd6384c7c4030b4e3aca9004c1b8486ebbc1f0f5d32e",
            "a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95",
            "8c4f77bdd24fa14786f3dff28a4044d819f444c0338484a2fa6df9588100cb59",
            "9246504e97e1ee46c2cdcf7dfac0f41364dca40f26e5c26f28f0968d0443808d",
            "a71854673f64e027bd673cf4c74673bcd7de74dca6f5b7b3b2c429467055f215",
            "c24e0193b29c7a8fa05cf9a25035ac01816fe54f3b80820b8e8de47418b90457",
            "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
        ):
            self.assertIn(token, self.source)
        self.assertIn('normalized.startswith("methods/bernini_action_editing/")', self.source)
        self.assertIn("if not selected:", self.source)
        self.assertIn("methods/bernini_action_editing \\", self.source)

    def test_editor_and_generated_media_denials_are_explicit(self) -> None:
        for token in (
            "--ack-generated-t2v-hidden-critic-only",
            "--ack-no-generated-media-editor-use",
            "--ack-no-optimizer-or-editor-update",
            "editor_target=false",
            "mask=false",
            "flow=false",
            "pose=false",
            "track=false",
        ):
            self.assertIn(token, self.source)


if __name__ == "__main__":
    unittest.main()
