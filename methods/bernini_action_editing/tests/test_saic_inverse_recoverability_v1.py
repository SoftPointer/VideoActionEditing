from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch  # noqa: E402
except ImportError:
    torch = None

if torch is not None:
    import saic_inverse_recoverability_v1 as inverse  # noqa: E402
else:  # pragma: no cover
    inverse = None


def _sha(character: str) -> str:
    return character * 64


def _codec_receipt() -> dict[str, object]:
    body = {
        "schema_version": inverse.CODEC_RECEIPT_SCHEMA_VERSION,
        "candidate_id": "dog-chosen",
        "input_output_media_sha256": _sha("a"),
        "decoded_rgb24_sha256": _sha("b"),
        "codec_name": "h264-crf18",
        "codec_bitstream_sha256": _sha("c"),
        "codec_decoded_rgb24_sha256": _sha("d"),
        "vae_id": "bernini-vae",
        "vae_weights_sha256": _sha("e"),
        "reencoded_latent_sha256": _sha("f"),
        "frame_count": 81,
        "fps_numerator": 25,
        "fps_denominator": 1,
        "endpoint_detached": True,
    }
    return {**body, "receipt_digest": inverse.object_sha256(body)}


def _authorize(*, event=0.9, verified=True, chosen_error=0.1, baseline_error=0.5):
    return inverse.authorize_inverse_flow_matching(
        midpoint_codec_receipt=_codec_receipt(),
        inverse_conditioning_source_sha256=_sha("f"),
        terminal_event_verified=verified,
        forward_event_score=torch.tensor(event, dtype=torch.float32),
        absolute_event_floor=0.8,
        chosen_reconstruction_error=torch.tensor(chosen_error, dtype=torch.float32),
        baseline_reconstruction_error=torch.tensor(
            baseline_error, dtype=torch.float32
        ),
        recoverability_floor=0.8,
        minimum_recoverability_gain=0.1,
    )


@unittest.skipIf(torch is None, "torch unavailable")
class SAICInverseRecoverabilityTests(unittest.TestCase):
    def test_recoverability_axis_is_detached_and_monotone(self) -> None:
        better = inverse.recoverability_score(torch.tensor(0.1, dtype=torch.float32))
        worse = inverse.recoverability_score(torch.tensor(1.0, dtype=torch.float32))
        self.assertFalse(better.requires_grad)
        self.assertIsNone(better.grad_fn)
        self.assertGreater(float(better.item()), float(worse.item()))
        with self.assertRaisesRegex(inverse.SAICInverseRecoverabilityError, "detached"):
            inverse.recoverability_score(
                torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
            )

    def test_absolute_event_gate_precedes_inverse_authorization(self) -> None:
        result = _authorize(event=0.79, verified=True)
        self.assertFalse(result.authorized)
        self.assertEqual(result.zero_update_reason, "absolute_forward_event_not_verified")
        self.assertTrue(result.receipt["recoverability_floor_pass"])
        self.assertTrue(result.receipt["recoverability_rank_pass"])
        with self.assertRaisesRegex(
            inverse.SAICInverseRecoverabilityError, "not authorized"
        ):
            inverse.validate_authorization(result.receipt)
        result = _authorize(event=0.99, verified=False)
        self.assertFalse(result.authorized)

    def test_all_detached_gates_authorize_unique_midpoint_condition(self) -> None:
        result = _authorize()
        self.assertTrue(result.authorized)
        receipt = inverse.validate_authorization(result.receipt)
        self.assertEqual(
            receipt["midpoint_reencoded_latent_sha256"],
            receipt["inverse_conditioning_source_sha256"],
        )
        self.assertTrue(
            receipt["inverse_conditioning_uses_midpoint_as_unique_visual_source"]
        )
        self.assertFalse(receipt["pure_t2v_visual_condition_used"])

    def test_midpoint_codec_binding_and_seals_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            inverse.SAICInverseRecoverabilityError, "not the codec-reencoded midpoint"
        ):
            inverse.authorize_inverse_flow_matching(
                midpoint_codec_receipt=_codec_receipt(),
                inverse_conditioning_source_sha256=_sha("9"),
                terminal_event_verified=True,
                forward_event_score=torch.tensor(0.9),
                absolute_event_floor=0.8,
                chosen_reconstruction_error=torch.tensor(0.1),
                baseline_reconstruction_error=torch.tensor(0.5),
                recoverability_floor=0.8,
                minimum_recoverability_gain=0.1,
            )
        receipt = deepcopy(_authorize().receipt)
        receipt["forward_event_score"] = 0.1
        with self.assertRaisesRegex(inverse.SAICInverseRecoverabilityError, "digest differs"):
            inverse.validate_authorization(receipt)
        # Recomputing the unkeyed integrity digest cannot turn inconsistent
        # gate claims into an authorization.
        receipt["authorization_digest"] = inverse.object_sha256(
            {key: value for key, value in receipt.items() if key != "authorization_digest"}
        )
        with self.assertRaisesRegex(
            inverse.SAICInverseRecoverabilityError, "gates are inconsistent"
        ):
            inverse.validate_authorization(receipt)

    def test_authorized_inverse_fm_targets_real_source_and_backpropagates(self) -> None:
        authorization = _authorize()
        source = torch.zeros(1, 16, 21, 2, 2, dtype=torch.float32)
        epsilon = torch.full_like(source, 2.0)
        target = epsilon - source
        leaf = (target + 0.1).clone().requires_grad_(True)
        prediction = leaf * 1.0
        prediction.retain_grad()
        result = inverse.authorized_inverse_flow_matching(
            source,
            epsilon,
            torch.tensor(0.25, dtype=torch.float32),
            prediction,
            exact40_index=33,
            midpoint_condition_sha256=_sha("f"),
            authorization_receipt=authorization.receipt,
        )
        self.assertTrue(torch.equal(result.state, 0.75 * source + 0.25 * epsilon))
        self.assertTrue(torch.equal(result.velocity_target, target))
        self.assertAlmostEqual(float(result.loss.item()), 0.01, places=6)
        result.loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertGreater(float(prediction.grad.abs().sum().item()), 0.0)

    def test_inverse_fm_rejects_wrong_midpoint_and_forbidden_indices(self) -> None:
        authorization = _authorize()
        source = torch.zeros(1, 16, 21, 2, 2, dtype=torch.float32)
        epsilon = torch.ones_like(source)
        prediction = ((epsilon - source) + 0.1).requires_grad_(True) * 1.0
        kwargs = {
            "source_clean": source,
            "epsilon": epsilon,
            "sigma": torch.tensor(0.5),
            "inverse_prediction": prediction,
            "exact40_index": 20,
            "midpoint_condition_sha256": _sha("9"),
            "authorization_receipt": authorization.receipt,
        }
        with self.assertRaisesRegex(inverse.SAICInverseRecoverabilityError, "midpoint"):
            inverse.authorized_inverse_flow_matching(**kwargs)
        kwargs["midpoint_condition_sha256"] = _sha("f")
        for index in (38, 39):
            kwargs["exact40_index"] = index
            with self.assertRaisesRegex(inverse.SAICInverseRecoverabilityError, "38/39"):
                inverse.authorized_inverse_flow_matching(**kwargs)


if __name__ == "__main__":
    unittest.main()
