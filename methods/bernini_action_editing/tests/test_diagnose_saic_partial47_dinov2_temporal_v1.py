from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import diagnose_saic_partial47_dinov2_temporal_v1 as diagnostic


class Partial47DINOContractTest(unittest.TestCase):
    def test_exact47_partition_is_balanced_and_exactly_once(self) -> None:
        partitions = [diagnostic.partition_indices(47, rank, 8) for rank in range(8)]
        self.assertEqual(
            [len(row) for row in partitions],
            [6, 6, 6, 6, 6, 6, 6, 5],
        )
        flattened = [index for row in partitions for index in row]
        self.assertEqual(sorted(flattened), list(range(47)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_schema_and_coverage_namespace_are_exact47(self) -> None:
        self.assertEqual(diagnostic.EXPECTED_ATTEMPT_COUNT, 47)
        self.assertEqual(diagnostic.EXPECTED_WORLD_SIZE, 8)
        self.assertIn("partial47", diagnostic.INPUT_SCHEMA)
        self.assertIn("partial47", diagnostic.SHARD_SCHEMA)
        self.assertIn("partial47", diagnostic.AGGREGATE_SCHEMA)
        source = Path(diagnostic.__file__).read_text("utf-8")
        self.assertIn('"coverage": "exactly_once_complete_partial47"', source)
        self.assertNotIn('"coverage": "exactly_once_complete_partial28"', source)

    def test_pinned_base_dependency_matches_declared_sha256(self) -> None:
        base = METHOD_ROOT / diagnostic._BASE_BASENAME
        self.assertEqual(hashlib.sha256(base.read_bytes()).hexdigest(), diagnostic._BASE_SHA256)
        self.assertEqual(
            diagnostic._BASE_SHA256,
            "213e408295610b5a0dd2e1eeb54f406c19a1985fb1ff290f89522fd38b4aaf4d",
        )

    def test_authority_is_fail_closed(self) -> None:
        self.assertEqual(
            diagnostic.AUTHORITY_CLOSURE,
            {
                "diagnostic_only": True,
                "raw_proxy_evidence_only": True,
                "identity_authority": False,
                "identity_preservation_verified": False,
                "event_authority": False,
                "event_verified": False,
                "scientific_claim_authorized": False,
                "selection_authorized": False,
                "ranking_authorized": False,
                "training_target_authorized": False,
                "optimizer_or_parameter_update_authorized": False,
            },
        )

    def test_launcher_is_exact47_and_maps_only_rocr_devices(self) -> None:
        launcher = (
            METHOD_ROOT / "scripts" / "auh_diagnose_saic_partial47_dinov2_temporal_v1.sh"
        ).read_text("utf-8")
        self.assertIn("for rank in 0 1 2 3 4 5 6 7", launcher)
        self.assertIn('ROCR_VISIBLE_DEVICES="$rank"', launcher)
        self.assertIn(
            "env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL",
            launcher,
        )
        self.assertNotIn('HIP_VISIBLE_DEVICES="$rank"', launcher)
        self.assertNotIn('CUDA_VISIBLE_DEVICES="$rank"', launcher)
        self.assertIn("aggregate is forbidden", launcher)
        self.assertIn("partial47 DINO", launcher)
        self.assertNotIn("partial28 DINO", launcher)


if __name__ == "__main__":
    unittest.main()
