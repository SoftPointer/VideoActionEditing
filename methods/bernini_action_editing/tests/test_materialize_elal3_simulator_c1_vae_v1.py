from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import materialize_elal3_simulator_c1_vae_v1 as subject


class ELAL3SimulatorC1VAETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.packet = (
            REPO_ROOT
            / "md/action_editing/20260817_box/simulator_gt_canary_v1"
        ).resolve(strict=True)
        cls.authority = (
            REPO_ROOT
            / "md/action_editing/20260817_box/evidence/"
            "elal3_c1_simulator_optimizer_diagnostic_authority_v1.json"
        ).resolve(strict=True)

    def test_exact_packet_closes_c1_media(self) -> None:
        manifest, row, bindings = subject.validate_packet(
            self.packet, subject.PACKET_MANIFEST_SHA256
        )
        self.assertEqual(manifest["sha256"], subject.PACKET_MANIFEST_SHA256)
        self.assertEqual(row["row_id"], subject.ROW_ID)
        self.assertEqual(tuple(bindings), subject.MEDIA_ORDER)
        self.assertTrue(all(value["nlink"] == 1 for value in bindings.values()))

    def test_wrong_manifest_literal_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            subject.ELAL3SimulatorVAEError, "manifest literal differs"
        ):
            subject.validate_packet(self.packet, "0" * 64)

    def test_external_derivative_authority_is_literal_bound(self) -> None:
        value, binding = subject.validate_derivative_authority(
            self.authority, subject.DERIVATIVE_AUTHORITY_SHA256
        )
        self.assertEqual(value["authorized_row_id"], subject.ROW_ID)
        self.assertEqual(binding["sha256"], subject.DERIVATIVE_AUTHORITY_SHA256)
        with self.assertRaisesRegex(
            subject.ELAL3SimulatorVAEError, "literal SHA-256 differs"
        ):
            subject.validate_derivative_authority(self.authority, "0" * 64)

    def test_real_model_authority_literal_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            subject.ELAL3SimulatorVAEError,
            "real-model authority literal SHA-256 differs",
        ):
            subject.validate_model_authority(
                Path("/abs/unopened-model-authority.json"),
                "0" * 64,
                bernini_root=Path("/abs/bernini"),
                checkpoint_root=Path("/abs/checkpoint"),
            )

    def test_acknowledgement_is_required_before_gpu_import(self) -> None:
        with self.assertRaisesRegex(
            subject.ELAL3SimulatorVAEError, "acknowledgement is required"
        ):
            subject.main(
                [
                    "--bernini-root",
                    "/abs/bernini",
                    "--veomni-root",
                    "/abs/veomni",
                    "--checkpoint",
                    "/abs/checkpoint",
                    "--packet-root",
                    str(self.packet),
                    "--packet-manifest-sha256",
                    subject.PACKET_MANIFEST_SHA256,
                    "--derivative-authority",
                    str(self.authority),
                    "--derivative-authority-sha256",
                    subject.DERIVATIVE_AUTHORITY_SHA256,
                    "--model-authority",
                    "/abs/model-authority.json",
                    "--model-authority-sha256",
                    subject.MODEL_AUTHORITY_SHA256,
                    "--output",
                    "/abs/fresh-output",
                ]
            )


if __name__ == "__main__":
    unittest.main()
