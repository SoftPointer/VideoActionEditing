from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import validate_pair_v5_rollout_manifest as validator  # noqa: E402


MANIFEST_PATH = (
    METHOD_ROOT / "assets" / "pair_v5_action_preference_rollout_train_v1.json"
)


def _payload() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


class PairV5RolloutManifestTests(unittest.TestCase):
    def test_sealed_manifest_passes_and_has_exact_grid(self) -> None:
        result = validator.load_manifest(MANIFEST_PATH)
        self.assertEqual(result["rollout_count"], 80)
        self.assertEqual(
            result["source_iids"],
            ["0159124a76ff4e09", "a9e969c99361494e"],
        )
        self.assertFalse(result["evidence_files_verified"])
        self.assertRegex(result["manifest_digest"], r"^[0-9a-f]{64}$")

    def test_closed_top_level_rejects_extra_key(self) -> None:
        payload = _payload()
        payload["notes"] = "open schemas are forbidden"
        with self.assertRaisesRegex(
            validator.PairV5RolloutManifestError, "extra=.*notes"
        ):
            validator.validate_manifest(payload)

    def test_shared8_iid_cannot_become_training_source(self) -> None:
        payload = _payload()
        payload["sources"][0]["iid"] = validator.EXPECTED_SHARED8_IIDS[5]
        with self.assertRaisesRegex(
            validator.PairV5RolloutManifestError, "leaks shared8"
        ):
            validator.validate_manifest(payload)

    def test_prompt_mutation_requires_matching_seal_and_still_changes_digest(self) -> None:
        original = _payload()
        payload = deepcopy(original)
        prompt = payload["sources"][0]["prompts"]["action"]
        prompt["text"] += " Altered."
        with self.assertRaisesRegex(
            validator.PairV5RolloutManifestError, "prompt SHA-256 differs"
        ):
            validator.validate_manifest(payload)

        prompt["sha256"] = hashlib.sha256(prompt["text"].encode("utf-8")).hexdigest()
        changed = validator.validate_manifest(payload)
        baseline = validator.validate_manifest(original)
        self.assertNotEqual(changed["manifest_digest"], baseline["manifest_digest"])

    def test_target_donor_mask_flow_and_custom_noise_are_fail_closed(self) -> None:
        for key in validator.FORBIDDEN_INPUT_KEYS:
            with self.subTest(key=key):
                payload = _payload()
                payload["rendering_contract"]["forbidden_conditions"][key] = True
                with self.assertRaisesRegex(
                    validator.PairV5RolloutManifestError, f"{key} must be exactly false"
                ):
                    validator.validate_manifest(payload)

    def test_grid_requires_four_unique_seeds_and_exact_count(self) -> None:
        payload = _payload()
        payload["candidate_grid"]["seeds"] = [7, 7, 8, 9]
        with self.assertRaisesRegex(
            validator.PairV5RolloutManifestError, "four unique"
        ):
            validator.validate_manifest(payload)

        payload = _payload()
        payload["candidate_grid"]["expected_rollout_count"] = 79
        with self.assertRaisesRegex(
            validator.PairV5RolloutManifestError, "rollout count differs"
        ):
            validator.validate_manifest(payload)

    def test_exact81_rv2v4_contract_cannot_drift(self) -> None:
        mutations = {
            "frame_count": 41,
            "source_reference_frame_indices": [0, 20, 40, 60, 80],
            "native_arm": "t2v",
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                payload = _payload()
                payload["rendering_contract"][key] = value
                with self.assertRaisesRegex(
                    validator.PairV5RolloutManifestError,
                    f"rendering_contract.{key} differs",
                ):
                    validator.validate_manifest(payload)

    def test_optional_file_verifier_recomputes_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.bin"
            artifact.write_bytes(b"sealed-evidence")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            validator._verify_file(str(artifact), digest, "fixture")
            with self.assertRaisesRegex(
                validator.PairV5RolloutManifestError, "SHA-256 differs"
            ):
                validator._verify_file(str(artifact), "0" * 64, "fixture")


if __name__ == "__main__":
    unittest.main()
