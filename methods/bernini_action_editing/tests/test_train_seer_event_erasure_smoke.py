from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_seer_event_erasure_smoke as seer
import train_lora as core


class SeerSmokeContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec_path = METHOD_ROOT / "assets" / "seer_owner_core2_v1.json"
        self.spec_sha = hashlib.sha256(self.spec_path.read_bytes()).hexdigest()

    def test_frozen_owner_spec(self) -> None:
        value = seer._load_owner_spec(self.spec_path.resolve(), self.spec_sha)
        self.assertEqual(len(value["rows"]), 2)
        self.assertEqual(
            {row["actor_family"] for row in value["rows"]}, {"dog", "human"}
        )
        self.assertFalse(
            value["fresh_experiment_authority"][
                "method_success_claim_authorized_by_training_completion"
            ]
        )

    def test_owner_spec_hash_is_fail_closed(self) -> None:
        with self.assertRaises(seer.SeerSmokeError):
            seer._load_owner_spec(self.spec_path.resolve(), "0" * 64)

    def test_instruction_mutation_is_rejected(self) -> None:
        value = json.loads(self.spec_path.read_text(encoding="utf-8"))
        value["rows"][0]["instruction"] += " changed"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "owner.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(seer.SeerSmokeError):
                seer._load_owner_spec(path.resolve(), digest)

    def test_cross_q_out_scope_is_exact(self) -> None:
        accepted = [
            f"renderer.transformer.blocks.{block}.attn2.{projection}"
            for block in range(30)
            for projection in ("to_q", "to_out.0")
        ]
        rejected = [
            "renderer.transformer.blocks.0.attn1.to_q",
            "renderer.transformer.blocks.0.attn2.to_k",
            "renderer.transformer.blocks.0.attn2.to_v",
        ]
        self.assertEqual(
            sum(bool(seer._CROSS_Q_OUT.fullmatch(name)) for name in accepted), 60
        )
        self.assertFalse(any(seer._CROSS_Q_OUT.fullmatch(name) for name in rejected))

    def test_specialization_pins_strict_single_actor_inclusion(self) -> None:
        owner = seer._load_owner_spec(self.spec_path.resolve(), self.spec_sha)
        original = (
            core.EXPECTED_DATASET_ROWS,
            core.EXPECTED_STRICT_ROWS,
            core.EXPECTED_NON_STRICT_ROWS,
            core.EXPECTED_INCLUSION_POLICY,
            core.EXPECTED_LORA_TARGET_MODULES,
            core.select_attention_projection_names,
            core.build_receipt,
            core.save_training_checkpoint,
        )
        try:
            seer._install_specialization(owner, self.spec_path.resolve())
            self.assertEqual(core.EXPECTED_DATASET_ROWS, 2)
            self.assertEqual(core.EXPECTED_STRICT_ROWS, 2)
            self.assertEqual(core.EXPECTED_NON_STRICT_ROWS, 0)
            self.assertEqual(core.EXPECTED_INCLUSION_POLICY, "strict_single_actor")
        finally:
            (
                core.EXPECTED_DATASET_ROWS,
                core.EXPECTED_STRICT_ROWS,
                core.EXPECTED_NON_STRICT_ROWS,
                core.EXPECTED_INCLUSION_POLICY,
                core.EXPECTED_LORA_TARGET_MODULES,
                core.select_attention_projection_names,
                core.build_receipt,
                core.save_training_checkpoint,
            ) = original


if __name__ == "__main__":
    unittest.main()
