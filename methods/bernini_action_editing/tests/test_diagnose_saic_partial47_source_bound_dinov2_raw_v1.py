from __future__ import annotations

from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import diagnose_saic_partial47_source_bound_dinov2_raw_v1 as diagnostic


class Exact47SourceBoundRawContractTest(unittest.TestCase):
    def test_partition_is_exactly_once_and_balanced(self) -> None:
        partitions = [diagnostic.partition_indices(47, rank, 8) for rank in range(8)]
        self.assertEqual([len(row) for row in partitions], [6, 6, 6, 6, 6, 6, 6, 5])
        flattened = [index for row in partitions for index in row]
        self.assertEqual(sorted(flattened), list(range(47)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_manifest_and_wrong_source_policy_are_pinned(self) -> None:
        self.assertEqual(
            diagnostic.EXPECTED_SOURCE_MANIFEST_SHA256,
            "899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9",
        )
        self.assertEqual(
            diagnostic.WRONG_SOURCE_POLICY,
            "same_actor_family_iid_lexical_cyclic_next_v1",
        )
        self.assertEqual(
            diagnostic._PINNED["diagnose_saic_partial28_dinov2_temporal_v1.py"],
            "213e408295610b5a0dd2e1eeb54f406c19a1985fb1ff290f89522fd38b4aaf4d",
        )
        self.assertEqual(diagnostic.EXPECTED_EVALUATOR_SPEC_SHA256, "6b18b9bc10589325ee2c09af339ef43a3eff507bcc754a2a6984cb70f0afd736")
        self.assertEqual(diagnostic.EXPECTED_VISUAL_SCORER_SHA256, "9e86ee8128841f624db92b99914235a37fee4d7b92aeda2e62104ab57e531b39")
        self.assertEqual(diagnostic.EXPECTED_VISUAL_CONTRACT_SHA256, "183eaafaebef426f888aa3abe91632a884f827d39ae16db576d57da401a8533a")

    def test_schema_coverage_and_authority_are_source_bound_raw_only(self) -> None:
        source = Path(diagnostic.__file__).read_text("utf-8")
        self.assertIn("partial47-source-bound-dinov2-raw", diagnostic.SCHEMA_VERSION)
        self.assertIn('"coverage": "exactly_once_complete_partial47_source_bound_raw"', source)
        self.assertFalse(diagnostic.AUTHORITY_CLOSURE["absolute_preservation_authority"])
        self.assertFalse(diagnostic.AUTHORITY_CLOSURE["source_bound_proxy_authority"])
        self.assertFalse(diagnostic.AUTHORITY_CLOSURE["identity_authority"])
        self.assertFalse(diagnostic.AUTHORITY_CLOSURE["event_authority"])
        self.assertFalse(diagnostic.AUTHORITY_CLOSURE["scientific_claim_authorized"])
        self.assertFalse(diagnostic.AUTHORITY_CLOSURE["ranking_authorized"])
        self.assertFalse(diagnostic.AUTHORITY_CLOSURE["selection_authorized"])
        self.assertFalse(diagnostic.AUTHORITY_CLOSURE["training_target_authorized"])

    def test_raw_metrics_expose_no_authority(self) -> None:
        import torch
        candidate_global = torch.nn.functional.normalize(torch.arange(24, dtype=torch.float32).reshape(3, 8) + 1, dim=-1)
        correct_global = candidate_global.clone()
        wrong_global = torch.flip(candidate_global, dims=(0,))
        candidate_dense = candidate_global[:, None, :].repeat(1, 4, 1)
        correct_dense = candidate_dense.clone()
        wrong_dense = torch.flip(candidate_dense, dims=(0,))
        result = diagnostic.raw_metrics(candidate_global, candidate_dense, correct_global, correct_dense, wrong_global, wrong_dense)
        self.assertEqual(result["global_source_self_upper_bound"], 1.0)
        self.assertEqual(result["dense_source_self_upper_bound"], 1.0)
        self.assertIsNone(result["thresholds"])
        for field in (
            "absolute_preservation_authority", "identity_authority", "event_authority",
            "scientific_claim_authorized", "ranking_authorized", "selection_authorized",
            "training_target_authorized",
        ):
            self.assertFalse(result[field])

    def test_launcher_contract(self) -> None:
        launcher = (METHOD_ROOT / "scripts" / "auh_diagnose_saic_partial47_source_bound_dinov2_raw_v1.sh").read_text("utf-8")
        self.assertIn('if [[ "$#" -ne 18 ]]', launcher)
        self.assertIn("--source-manifest", launcher)
        self.assertIn("--expected-source-manifest-sha256", launcher)
        self.assertIn("for rank in 0 1 2 3 4 5 6 7", launcher)
        self.assertIn('ROCR_VISIBLE_DEVICES="$rank"', launcher)
        self.assertIn("env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL", launcher)
        self.assertNotIn('HIP_VISIBLE_DEVICES="$rank"', launcher)
        self.assertNotIn('CUDA_VISIBLE_DEVICES="$rank"', launcher)
        self.assertIn("aggregate is forbidden", launcher)


if __name__ == "__main__":
    unittest.main()
