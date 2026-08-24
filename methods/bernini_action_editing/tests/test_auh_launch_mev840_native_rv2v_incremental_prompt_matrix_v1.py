from __future__ import annotations

from pathlib import Path
import hashlib
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "auh_launch_mev840_native_rv2v_incremental_prompt_matrix_v1.sh"
AUTHORITY = ROOT / "assets" / "mev840_native_rv2v_incremental_prompt_matrix_v1.json"
P1 = ROOT / "assets" / "mev840_action_only_p1_event_order_v1.txt"
P2 = ROOT / "assets" / "mev840_action_only_p2_relation_contact_v1.txt"


class NativeIncrementalPromptMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = LAUNCHER.read_text(encoding="utf-8")
        cls.authority = json.loads(AUTHORITY.read_text(encoding="ascii"))

    def test_three_prompts_are_exact_incremental_factorization(self) -> None:
        common = self.authority["common"]
        prefix = common["source_context_utf8"]
        base = common["base_action_utf8"]
        suffix = common["preservation_suffix_utf8"]
        payloads = {
            "P0": "",
            "P1": " Follow this event order: " + P1.read_text(encoding="ascii")[:-1],
            "P2": " Follow these contact relations: " + P2.read_text(encoding="ascii")[:-1],
        }
        for label, addition in payloads.items():
            full = (prefix + base + addition + suffix).encode("utf-8")
            row = self.authority["prompts"][label]
            self.assertEqual(full.decode("utf-8"), row["full_prompt_utf8"])
            self.assertEqual(len(full), row["full_prompt_utf8_bytes"])
            self.assertEqual(hashlib.sha256(full).hexdigest(), row["full_prompt_utf8_sha256"])

    def test_same_seed_node_and_sequential_wave_contract(self) -> None:
        self.assertIn("launch-p0) [[ $# == 1 ]] || usage; launch_wave P0", self.script)
        self.assertIn("launch-p1) [[ $# == 1 ]] || usage; launch_wave P1", self.script)
        self.assertIn("launch-p2) [[ $# == 1 ]] || usage; launch_wave P2", self.script)
        self.assertIn("2027) readonly expected_job=143808 expected_node=auh7-1b-gpu-292", self.script)
        self.assertIn("2028) readonly expected_job=147873 expected_node=auh7-1b-gpu-284", self.script)
        self.assertNotIn("auh7-1b-gpu-268", self.script)
        self.assertNotIn("auh7-1b-gpu-315", self.script)

    def test_exact_scratch_native_and_comparison_gates(self) -> None:
        for digest in (
            "46ae7529d640a197006ab8d7d17c23ac81925dabd7fa1caf4b0bb261197e8115",
            "e104031526236f16e94a4753c31ad8048b1a65345b1913212c35e421fcad48ae",
            "bf402cd65257121d1ebedcc83c2c59965b37305a36b0b5a6327241e74d7b4f42",
            "2a334405d892434b8855d1a652c577c6caedf9bf63e1e0698ee4cd1973dd994b",
            "e22733bd003e77b0a914ce8a3a15f3b850285f7d721a04a8a51d81c1920e3f34",
        ):
            self.assertIn(digest, self.script)
        self.assertIn('actual != set(expected)', self.script)
        self.assertIn('cfile=sys.argv[2]', self.script)
        self.assertIn('--arms rv2v --num-inference-steps 40', self.script)
        self.assertIn('len(set(noises[seed]))!=1', self.script)
        self.assertIn('len(set(sources[seed]))!=1', self.script)
        self.assertIn('for index in ("0","27","53","80")', self.script)
        self.assertIn('q["raw_storage_sha256"]', self.script)
        self.assertIn('generator_target_action_json_read":False', self.script)
        self.assertNotIn("target_action_oracle", self.script)
        self.assertNotIn("activity25", self.script)
        self.assertNotIn("anchor_qk", self.script)


if __name__ == "__main__":
    unittest.main()
