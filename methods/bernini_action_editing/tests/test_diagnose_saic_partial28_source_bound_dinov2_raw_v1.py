from __future__ import annotations

from pathlib import Path
import hashlib
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import diagnose_saic_partial28_source_bound_dinov2_raw_v1 as diagnostic


class R6Exact28SourceBoundRawContractTest(unittest.TestCase):
    def test_partition_is_exactly_once_and_balanced(self) -> None:
        partitions = [diagnostic.partition_indices(28, rank, 8) for rank in range(8)]
        self.assertEqual([len(row) for row in partitions], [4, 4, 4, 4, 3, 3, 3, 3])
        flattened = [index for row in partitions for index in row]
        self.assertEqual(sorted(flattened), list(range(28)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_nested_self_verification_targets_the_exact28_source(self) -> None:
        source = Path(diagnostic.__file__)
        source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        diagnostic._install_specialization()
        self.assertEqual(Path(diagnostic.core.__file__), source)
        self.assertEqual(Path(diagnostic.core.core.__file__), source)
        self.assertEqual(diagnostic.core.core._verify_self(source_sha256), source_sha256)
        self.assertEqual(diagnostic.core.core.EXPECTED_ATTEMPT_COUNT, 28)
        self.assertEqual(diagnostic.core.core.EXPECTED_WORLD_SIZE, 8)
        self.assertIs(diagnostic.core.partition_indices, diagnostic.partition_indices)
        self.assertIs(diagnostic.core._base_partition_indices, diagnostic.partition_indices)
        self.assertIs(diagnostic.core.core.partition_indices, diagnostic.partition_indices)
        with self.assertRaises(diagnostic.SourceBoundRaw28Error):
            diagnostic.partition_indices(47, 0, 8)

    def test_exact47_base_and_r6_bank_are_pinned(self) -> None:
        self.assertEqual(
            diagnostic._BASE_SHA256,
            "ffbc9ba149d1ddadf704dd8258678a8893235e328da4c7601e98d63ba37aa7a2",
        )
        self.assertEqual(
            diagnostic.EXPECTED_ATTEMPTS_ROOT,
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/runs/t2v-events-topup-r6-umaskfix-72f3a40-r1/attempts",
        )
        self.assertEqual(
            diagnostic.EXPECTED_ROOT_SPEC_SHA256,
            "d693d0784530f007888e2825d15db3db808fdf4f1d111b5d080d968c894ff145",
        )
        self.assertEqual(
            diagnostic.EXPECTED_SOURCE_MANIFEST_SHA256,
            "899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9",
        )
        self.assertEqual(
            diagnostic.EXPECTED_RECEIPT_BINDING_SHA256,
            "f3e1717fb86298a5e0995d6a70322709c4b9df0614c22e7eb63fe927e35dcb92",
        )

    def test_registered_visual_inputs_and_wrong_policy_are_unchanged(self) -> None:
        self.assertEqual(diagnostic.core.EXPECTED_EVALUATOR_SPEC_SHA256, "6b18b9bc10589325ee2c09af339ef43a3eff507bcc754a2a6984cb70f0afd736")
        self.assertEqual(diagnostic.core.EXPECTED_VISUAL_SCORER_SHA256, "9e86ee8128841f624db92b99914235a37fee4d7b92aeda2e62104ab57e531b39")
        self.assertEqual(diagnostic.core.EXPECTED_VISUAL_CONTRACT_SHA256, "183eaafaebef426f888aa3abe91632a884f827d39ae16db576d57da401a8533a")
        self.assertEqual(diagnostic.core.WRONG_SOURCE_POLICY, "same_actor_family_iid_lexical_cyclic_next_v1")
        self.assertEqual(
            diagnostic.EXPECTED_WRONG_IID_BY_IID,
            {
                "311c82f83eca4a7f": "31c34509415745ca",
                "31c34509415745ca": "6d346c38cf504493",
                "6d346c38cf504493": "a35b590961d24694",
                "a35b590961d24694": "311c82f83eca4a7f",
                "6ea45d35943742bb": "7b88a1ca1f804f41",
                "7b88a1ca1f804f41": "841b5e0080a1441d",
                "841b5e0080a1441d": "99cde432839f4240",
                "99cde432839f4240": "6ea45d35943742bb",
            },
        )

    def test_schema_coverage_and_authority_are_r6_exact28_raw_only(self) -> None:
        source = Path(diagnostic.__file__).read_text("utf-8")
        self.assertIn("r6-partial28-source-bound-dinov2-raw", diagnostic.SCHEMA_VERSION)
        self.assertIn('"coverage": "exactly_once_complete_r6_partial28_source_bound_raw"', source)
        for field in (
            "absolute_preservation_authority", "source_bound_proxy_authority",
            "identity_authority", "event_authority", "scientific_claim_authorized",
            "ranking_authorized", "selection_authorized", "training_target_authorized",
        ):
            self.assertFalse(diagnostic.AUTHORITY_CLOSURE[field])

    def test_launcher_has_18_args_fixed_runtime_and_only_rocr_mapping(self) -> None:
        launcher = (METHOD_ROOT / "scripts" / "auh_diagnose_saic_partial28_source_bound_dinov2_raw_v1.sh").read_text("utf-8")
        self.assertIn('if [[ "$#" -ne 18 ]]', launcher)
        self.assertIn("runtime/venv-transformers-4.53.2/bin/python", launcher)
        self.assertIn("ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime", launcher)
        self.assertIn("356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5", launcher)
        self.assertIn('export PATH="$portable_ffprobe_dir:${PATH:-/usr/bin:/bin}"', launcher)
        self.assertNotIn("readlink", launcher)
        self.assertNotIn("realpath", launcher)
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
