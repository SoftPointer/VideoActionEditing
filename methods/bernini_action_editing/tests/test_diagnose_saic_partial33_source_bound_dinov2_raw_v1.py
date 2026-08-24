from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import diagnose_saic_partial33_source_bound_dinov2_raw_v1 as diagnostic


class Exact33SourceBoundRawContractTest(unittest.TestCase):
    def test_partition_is_exactly_once_and_balanced(self) -> None:
        partitions = [diagnostic.partition_indices(33, rank, 8) for rank in range(8)]
        self.assertEqual([len(row) for row in partitions], [5, 4, 4, 4, 4, 4, 4, 4])
        flattened = [index for row in partitions for index in row]
        self.assertEqual(sorted(flattened), list(range(33)))
        self.assertEqual(len(flattened), len(set(flattened)))
        with self.assertRaises(diagnostic.SourceBoundRaw33Error):
            diagnostic.partition_indices(47, 0, 8)
        with self.assertRaises(diagnostic.SourceBoundRaw33Error):
            diagnostic.partition_indices(33, 0, 4)
        with self.assertRaises(diagnostic.SourceBoundRaw33Error):
            diagnostic.partition_indices(33, 8, 8)

    def test_exact47_base_and_registered_inputs_are_pinned(self) -> None:
        self.assertEqual(diagnostic._BASE_SHA256, "ffbc9ba149d1ddadf704dd8258678a8893235e328da4c7601e98d63ba37aa7a2")
        self.assertEqual(diagnostic.core.EXPECTED_SOURCE_MANIFEST_SHA256, "899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9")
        self.assertEqual(diagnostic.core.EXPECTED_EVALUATOR_SPEC_SHA256, "6b18b9bc10589325ee2c09af339ef43a3eff507bcc754a2a6984cb70f0afd736")
        self.assertEqual(diagnostic.core.EXPECTED_VISUAL_SCORER_SHA256, "9e86ee8128841f624db92b99914235a37fee4d7b92aeda2e62104ab57e531b39")
        self.assertEqual(diagnostic.core.EXPECTED_VISUAL_CONTRACT_SHA256, "183eaafaebef426f888aa3abe91632a884f827d39ae16db576d57da401a8533a")
        self.assertEqual(diagnostic.core.WRONG_SOURCE_POLICY, "same_actor_family_iid_lexical_cyclic_next_v1")

    def test_nested_core_self_identity_and_build_manifest_dispatch(self) -> None:
        source = Path(diagnostic.__file__).resolve()
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        diagnostic._configure_core()
        self.assertEqual(Path(diagnostic.core.core.__file__).resolve(), source)
        self.assertEqual(diagnostic.core.core._verify_self(source_sha), source_sha)
        with mock.patch.object(diagnostic.core, "build_manifest", return_value=17) as build:
            result = diagnostic.main([
                "build-manifest",
                "--attempts-root", "/fixture/attempts",
                "--expected-root-spec-sha256", "a" * 64,
                "--source-manifest", "/fixture/source.json",
                "--expected-source-manifest-sha256", diagnostic.core.EXPECTED_SOURCE_MANIFEST_SHA256,
                "--expected-source-sha256", source_sha,
                "--output-root", "/fixture/output",
            ])
        self.assertEqual(result, 17)
        build.assert_called_once()
        dispatched = build.call_args.args[0]
        self.assertEqual(dispatched.command, "build-manifest")
        self.assertEqual(dispatched.expected_source_sha256, source_sha)

    def test_schema_coverage_and_authority_are_exact33_raw_only(self) -> None:
        source = Path(diagnostic.__file__).read_text("utf-8")
        self.assertIn("partial33-source-bound-dinov2-raw", diagnostic.SCHEMA_VERSION)
        self.assertIn('"coverage": "exactly_once_complete_partial33_source_bound_raw"', source)
        for field in (
            "absolute_preservation_authority", "source_bound_proxy_authority",
            "identity_authority", "event_authority", "scientific_claim_authorized",
            "ranking_authorized", "selection_authorized", "training_target_authorized",
        ):
            self.assertFalse(diagnostic.AUTHORITY_CLOSURE[field])

    def test_launcher_has_18_args_and_only_rocr_rank_mapping(self) -> None:
        launcher = (METHOD_ROOT / "scripts" / "auh_diagnose_saic_partial33_source_bound_dinov2_raw_v1.sh").read_text("utf-8")
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
