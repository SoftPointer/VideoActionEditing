from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch

    import fewshot_proposal_motion_carrier as carrier
except ImportError as error:  # pragma: no cover - local environments without torch
    raise unittest.SkipTest("PyTorch is unavailable") from error


class FewShotProposalMotionCarrierTests(unittest.TestCase):
    def test_patch_embedding_dtype_is_not_inferred_from_first_transformer_parameter(
        self,
    ) -> None:
        class RecordingPatchEmbedding:
            def __init__(self) -> None:
                self.weight = torch.zeros(1, dtype=torch.bfloat16)
                self.bias = torch.zeros(1, dtype=torch.bfloat16)
                self.observed = []

            def __call__(self, value):
                self.observed.append(value.dtype)
                return torch.empty(
                    1, 1536, 21, 30, 31, dtype=torch.bfloat16, device="meta"
                )

        patch_embedding = RecordingPatchEmbedding()
        transformer = types.SimpleNamespace(
            patch_embedding=patch_embedding,
            parameters=lambda: iter((torch.nn.Parameter(torch.zeros(1)),)),
        )
        latent = torch.zeros(1, 16, 21, 60, 62, dtype=torch.float32)
        original = carrier.build_fewshot_proposal_carrier
        observed = {}

        def record(action, noop, *, expected_patch_grid):
            observed["action_dtype"] = action.dtype
            observed["noop_dtype"] = noop.dtype
            observed["grid"] = expected_patch_grid
            return "carrier"

        carrier.build_fewshot_proposal_carrier = record
        try:
            result = carrier.build_carrier_from_proposal_latents(
                transformer,
                latent,
                latent,
                expected_patch_grid=(30, 31),
            )
        finally:
            carrier.build_fewshot_proposal_carrier = original

        self.assertEqual(result, "carrier")
        self.assertEqual(patch_embedding.observed, [torch.bfloat16, torch.bfloat16])
        self.assertEqual(observed["action_dtype"], torch.bfloat16)
        self.assertEqual(observed["noop_dtype"], torch.bfloat16)
        self.assertEqual(observed["grid"], (30, 31))

    def test_geometry_is_closed_and_orientation_is_not_transposed(self) -> None:
        self.assertEqual(carrier.validate_patch_grid((30, 31)), (30, 31))
        self.assertEqual(carrier.validate_patch_grid((31, 30)), (31, 30))
        for bad in ((30, 30), (62, 15), (31.0, 30), True):
            with self.subTest(bad=bad):
                with self.assertRaises(carrier.FewShotCarrierContractError):
                    carrier.validate_patch_grid(bad)

    def test_pooled_normalization_preserves_phase_zero_and_caps_tokens(self) -> None:
        value = torch.zeros(1, 21, 8, 8, 1536, dtype=torch.float32)
        value[:, 1, 0, 0, :] = 100.0
        value[:, 2:] = 0.25
        result = carrier.normalize_pooled_motion_content(value)
        self.assertEqual(tuple(result.activity.shape), (1, 21))
        self.assertFalse(bool(result.activity[0, 0]))
        self.assertTrue(bool(result.activity[0, 1:].all()))
        self.assertEqual(
            torch.count_nonzero(result.clipped_content_fp32[:, 0]).item(), 0
        )
        self.assertFalse(
            torch.signbit(result.clipped_content_fp32[:, 0]).any().item()
        )
        rms = result.clipped_content_fp32.square().mean(dim=-1).sqrt()
        self.assertLessEqual(float(rms.max()), carrier.TOKEN_RMS_CAP + 1.0e-5)

    def test_true_30_by_31_grid_builds_without_transpose(self) -> None:
        phase = torch.zeros(1, 21, 1, 1, 1, dtype=torch.float32)
        phase[:, 7:] = 1.0
        # Expanded views avoid allocating two 120 MiB input tensors; the
        # implementation still materializes and audits its own FP32 difference.
        action = phase.expand(1, 21, 30, 31, 1536)
        noop = torch.zeros(1, 1, 1, 1, 1, dtype=torch.float32).expand_as(action)
        result = carrier.build_fewshot_proposal_carrier(
            action, noop, expected_patch_grid=(30, 31)
        )
        self.assertEqual(tuple(result.carrier_fp32.shape), (1, 21, 8, 8, 1536))
        self.assertEqual(result.patch_grid, (30, 31))
        self.assertEqual(result.activity[0].nonzero().flatten().tolist(), [7])
        receipt = result.audit_receipt()
        self.assertEqual(receipt["patch_grid_yx"], [30, 31])
        self.assertIs(receipt["patch_grid_transposed"], False)
        self.assertIs(receipt["target_or_support_input"], False)
        self.assertEqual(len(receipt["receipt_sha256"]), 64)

    def test_bad_phase_zero_and_wrong_orientation_fail_closed(self) -> None:
        pooled = torch.zeros(1, 21, 8, 8, 1536, dtype=torch.float32)
        pooled[:, 0, 0, 0, 0] = 1.0
        with self.assertRaisesRegex(
            carrier.FewShotCarrierContractError, "phase 0"
        ):
            carrier.normalize_pooled_motion_content(pooled)

        wrong = torch.zeros(1, 21, 31, 30, 1536, dtype=torch.float32)
        with self.assertRaisesRegex(
            carrier.FewShotCarrierContractError, "exact shape"
        ):
            carrier.build_fewshot_proposal_carrier(
                wrong, wrong, expected_patch_grid=(30, 31)
            )


if __name__ == "__main__":
    unittest.main()
