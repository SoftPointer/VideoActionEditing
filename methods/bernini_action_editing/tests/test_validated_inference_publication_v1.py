from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import validated_inference_publication_v1 as publication


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


DECODER_AUTHORITY = {
    "authority_id": "decoder-release-registry",
    "authority_version": "decoder-v7.2",
    "implementation_sha256": sha("decoder-implementation-v7.2"),
    "profile_sha256": sha("decoder-profile-exact81-rgb"),
}
SEMANTIC_AUTHORITY = {
    "authority_id": "action-preservation-review-board",
    "authority_version": "semantic-v3",
    "implementation_sha256": sha("semantic-evaluator-v3"),
    "profile_sha256": sha("action-and-preservation-profile-v3"),
}
QUALIFICATION_AUTHORITY = {
    "authority_id": "independent-lkg-qualification-board",
    "authority_version": "lkg-v2",
    "implementation_sha256": sha("lkg-qualification-v2"),
    "profile_sha256": sha("lkg-full81-profile-v2"),
}
INSTRUCTION_SHA = sha("raise the left arm and hold")


def clean_moving_video(*, offset: float = 0.0) -> np.ndarray:
    height, width = 32, 40
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx /= width - 1
    yy /= height - 1
    frames = []
    for index in range(81):
        phase = index / 80.0
        red = np.clip(0.18 + 0.55 * xx + 0.08 * np.sin(phase * np.pi) + offset, 0, 1)
        green = np.clip(0.15 + 0.60 * yy + 0.10 * phase + offset, 0, 1)
        blue = np.clip(0.22 + 0.28 * xx + 0.20 * yy - 0.08 * phase + offset, 0, 1)
        frames.append(np.stack((red, green, blue), axis=-1))
    return np.stack(frames).astype(np.float32)


def failed_video() -> np.ndarray:
    frames = clean_moving_video()
    frames[1:] = 0.0
    return frames


def seal(body: dict) -> dict:
    receipt = copy.deepcopy(body)
    receipt["receipt_sha256"] = publication._canonical_digest(receipt)
    return receipt


def evidence_bundle(
    identifier: str,
    role: str,
    frames: np.ndarray,
    *,
    semantic: str = "pass",
    qualification: str = "go",
) -> dict:
    container_sha = sha("container-bytes:" + identifier)
    binding = publication._decoded_binding(frames)
    decode = seal({
        "schema_version": publication.DECODE_RECEIPT_SCHEMA,
        "artifact_id": identifier,
        "artifact_role": role,
        "container_sha256": container_sha,
        "decoder_authority": copy.deepcopy(DECODER_AUTHORITY),
        "decoded_tensor": {
            "sha256": binding["decoded_tensor_sha256"],
            "dtype": binding["dtype"],
            "shape": binding["shape"],
            "frame_count": binding["frame_count"],
            "fps": 16.0,
        },
    })

    semantic_receipt = None
    if semantic != "missing":
        passed = semantic == "pass"
        semantic_receipt = seal({
            "schema_version": publication.SEMANTIC_VERDICT_SCHEMA,
            "artifact_id": identifier,
            "artifact_role": role,
            "container_sha256": container_sha,
            "decoded_tensor_sha256": binding["decoded_tensor_sha256"],
            "action_instruction_sha256": INSTRUCTION_SHA,
            "evaluation_authority": copy.deepcopy(SEMANTIC_AUTHORITY),
            "full_trajectory_reviewed": True,
            "action_success": passed,
            "preservation_success": passed,
            "verdict": "PASS" if passed else "FAIL",
        })

    qualification_receipt = None
    if role == publication.FROZEN_LAST_KNOWN_GOOD:
        qualified = qualification == "go"
        qualification_receipt = seal({
            "schema_version": publication.FROZEN_QUALIFICATION_SCHEMA,
            "artifact_id": identifier,
            "artifact_role": role,
            "container_sha256": container_sha,
            "decoded_tensor_sha256": binding["decoded_tensor_sha256"],
            "qualification_authority": copy.deepcopy(QUALIFICATION_AUTHORITY),
            "full_trajectory_qualified": qualified,
            "qualified": qualified,
            "verdict": "GO" if qualified else "NO_GO",
        })

    artifact = {
        "artifact_id": identifier,
        "role": role,
        "container_sha256": container_sha,
        "decode_receipt_sha256": decode["receipt_sha256"],
        "semantic_verdict_receipt_sha256": (
            None if semantic_receipt is None else semantic_receipt["receipt_sha256"]
        ),
    }
    if role == publication.FROZEN_LAST_KNOWN_GOOD:
        artifact["qualification_receipt_sha256"] = qualification_receipt["receipt_sha256"]
    return {
        "artifact": artifact,
        "decode": decode,
        "semantic": semantic_receipt,
        "qualification": qualification_receipt,
    }


def serving_kwargs(bundle: dict) -> dict:
    result = {
        "candidate_decode_receipt": bundle["decode"],
        "trusted_decoder_authority": DECODER_AUTHORITY,
        "action_instruction_sha256": INSTRUCTION_SHA,
    }
    if bundle["semantic"] is not None:
        result["candidate_semantic_verdict"] = bundle["semantic"]
        result["trusted_candidate_semantic_authority"] = SEMANTIC_AUTHORITY
    return result


def fallback_kwargs(bundle: dict, frames: np.ndarray) -> dict:
    result = {
        "fallback_policy": publication.FROZEN_LAST_KNOWN_GOOD,
        "frozen_frames": frames,
        "frozen_artifact": bundle["artifact"],
        "frozen_decode_receipt": bundle["decode"],
        "frozen_qualification_receipt": bundle["qualification"],
        "trusted_frozen_qualification_authority": QUALIFICATION_AUTHORITY,
    }
    if bundle["semantic"] is not None:
        result["frozen_semantic_verdict"] = bundle["semantic"]
        result["trusted_frozen_semantic_authority"] = SEMANTIC_AUTHORITY
    return result


class ValidatedInferencePublicationTest(unittest.TestCase):
    def test_fully_bound_candidate_can_receive_overall_authorization(self) -> None:
        frames = clean_moving_video()
        bundle = evidence_bundle("candidate.mp4", "candidate", frames)
        receipt = publication.decide_serving_publication(
            frames, bundle["artifact"], **serving_kwargs(bundle)
        )
        self.assertEqual(
            receipt["decision"], "authorize_candidate_for_action_editor_serving"
        )
        self.assertTrue(receipt["visual_media_publishable"])
        self.assertTrue(receipt["action_editor_serving_authorized"])
        self.assertEqual(receipt["selected_artifact"], bundle["artifact"])
        self.assertNotIn("serving_publishable", receipt)
        self.assertEqual(
            receipt["candidate"]["decode_receipt"]["container_sha256"],
            bundle["artifact"]["container_sha256"],
        )
        json.dumps(receipt, allow_nan=False)

    def test_visual_noop_without_semantic_verdict_has_no_overall_authorization(self) -> None:
        frames = clean_moving_video()
        bundle = evidence_bundle(
            "visually-valid-noop.mp4", "candidate", frames, semantic="missing"
        )
        receipt = publication.decide_serving_publication(
            frames, bundle["artifact"], **serving_kwargs(bundle)
        )
        self.assertTrue(receipt["visual_media_publishable"])
        self.assertFalse(receipt["action_editor_serving_authorized"])
        self.assertIsNone(receipt["selected_artifact"])
        self.assertIn("candidate_semantic_verdict_missing", receipt["authorization_blockers"])

    def test_semantic_action_or_preservation_failure_blocks_serving(self) -> None:
        frames = clean_moving_video()
        bundle = evidence_bundle("wrong-action.mp4", "candidate", frames, semantic="fail")
        receipt = publication.decide_serving_publication(
            frames, bundle["artifact"], **serving_kwargs(bundle)
        )
        self.assertTrue(receipt["visual_media_publishable"])
        self.assertFalse(receipt["action_editor_serving_authorized"])
        self.assertIn("candidate_action_or_preservation_failed", receipt["authorization_blockers"])

    def test_clean_frames_cannot_be_paired_with_wrong_artifact_sha(self) -> None:
        frames = clean_moving_video()
        bundle = evidence_bundle("candidate.mp4", "candidate", frames)
        wrong = copy.deepcopy(bundle["artifact"])
        wrong["container_sha256"] = sha("unrelated-noisy-container")
        with self.assertRaisesRegex(publication.PublicationInputError, "container_sha256"):
            publication.decide_serving_publication(
                frames, wrong, **serving_kwargs(bundle)
            )

    def test_clean_frames_cannot_use_decode_receipt_for_other_frame_bytes(self) -> None:
        clean = clean_moving_video()
        noisy = np.random.default_rng(7).random(clean.shape).astype(np.float32)
        bundle = evidence_bundle("candidate.mp4", "candidate", noisy)
        with self.assertRaisesRegex(publication.PublicationInputError, "actual decoded"):
            publication.decide_serving_publication(
                clean, bundle["artifact"], **serving_kwargs(bundle)
            )

    def test_forged_decode_receipt_fails_canonical_and_external_pins(self) -> None:
        frames = clean_moving_video()
        bundle = evidence_bundle("candidate.mp4", "candidate", frames)
        forged = copy.deepcopy(bundle["decode"])
        forged["decoded_tensor"]["fps"] = 30.0
        kwargs = serving_kwargs(bundle)
        kwargs["candidate_decode_receipt"] = forged
        with self.assertRaisesRegex(publication.PublicationInputError, "canonical receipt"):
            publication.decide_serving_publication(frames, bundle["artifact"], **kwargs)

    def test_untrusted_decoder_authority_is_rejected(self) -> None:
        frames = clean_moving_video()
        bundle = evidence_bundle("candidate.mp4", "candidate", frames)
        untrusted = copy.deepcopy(DECODER_AUTHORITY)
        untrusted["authority_version"] = "untrusted-v1"
        kwargs = serving_kwargs(bundle)
        kwargs["trusted_decoder_authority"] = untrusted
        with self.assertRaisesRegex(publication.PublicationInputError, "trusted authority"):
            publication.decide_serving_publication(frames, bundle["artifact"], **kwargs)

    def test_failed_candidate_can_use_fully_qualified_semantic_lkg(self) -> None:
        candidate_frames = failed_video()
        candidate = evidence_bundle("candidate.mp4", "candidate", candidate_frames)
        frozen_frames = clean_moving_video(offset=0.01)
        frozen = evidence_bundle(
            "frozen-lkg.mp4", publication.FROZEN_LAST_KNOWN_GOOD, frozen_frames
        )
        kwargs = serving_kwargs(candidate)
        kwargs.update(fallback_kwargs(frozen, frozen_frames))
        receipt = publication.decide_serving_publication(
            candidate_frames, candidate["artifact"], **kwargs
        )
        self.assertEqual(
            receipt["decision"],
            "authorize_frozen_last_known_good_for_action_editor_serving",
        )
        self.assertTrue(receipt["action_editor_serving_authorized"])
        self.assertEqual(receipt["selected_artifact"], frozen["artifact"])

    def test_lkg_role_string_without_qualification_receipt_is_rejected(self) -> None:
        candidate_frames = failed_video()
        candidate = evidence_bundle("candidate.mp4", "candidate", candidate_frames)
        frozen_frames = clean_moving_video(offset=0.01)
        frozen = evidence_bundle(
            "frozen-lkg.mp4", publication.FROZEN_LAST_KNOWN_GOOD, frozen_frames
        )
        kwargs = serving_kwargs(candidate)
        kwargs.update(fallback_kwargs(frozen, frozen_frames))
        kwargs["frozen_qualification_receipt"] = None
        with self.assertRaisesRegex(publication.PublicationInputError, "qualification"):
            publication.decide_serving_publication(
                candidate_frames, candidate["artifact"], **kwargs
            )

    def test_no_go_lkg_is_not_visually_selected_or_served(self) -> None:
        candidate_frames = failed_video()
        candidate = evidence_bundle("candidate.mp4", "candidate", candidate_frames)
        frozen_frames = clean_moving_video(offset=0.01)
        frozen = evidence_bundle(
            "frozen-lkg.mp4", publication.FROZEN_LAST_KNOWN_GOOD, frozen_frames,
            qualification="no-go",
        )
        kwargs = serving_kwargs(candidate)
        kwargs.update(fallback_kwargs(frozen, frozen_frames))
        receipt = publication.decide_serving_publication(
            candidate_frames, candidate["artifact"], **kwargs
        )
        self.assertFalse(receipt["visual_media_publishable"])
        self.assertFalse(receipt["action_editor_serving_authorized"])
        self.assertIn("fallback_not_independently_qualified", receipt["authorization_blockers"])

    def test_lkg_qualification_authority_must_be_independent_of_decoder(self) -> None:
        candidate_frames = failed_video()
        candidate = evidence_bundle("candidate.mp4", "candidate", candidate_frames)
        frozen_frames = clean_moving_video(offset=0.01)
        frozen = evidence_bundle(
            "frozen-lkg.mp4", publication.FROZEN_LAST_KNOWN_GOOD, frozen_frames
        )
        forged = copy.deepcopy(frozen["qualification"])
        forged["qualification_authority"] = copy.deepcopy(DECODER_AUTHORITY)
        forged = seal({key: value for key, value in forged.items() if key != "receipt_sha256"})
        frozen["artifact"]["qualification_receipt_sha256"] = forged["receipt_sha256"]
        kwargs = serving_kwargs(candidate)
        kwargs.update(fallback_kwargs(frozen, frozen_frames))
        kwargs["frozen_qualification_receipt"] = forged
        kwargs["trusted_frozen_qualification_authority"] = DECODER_AUTHORITY
        with self.assertRaisesRegex(publication.PublicationInputError, "not independent"):
            publication.decide_serving_publication(
                candidate_frames, candidate["artifact"], **kwargs
            )

    def test_scientific_failure_displays_candidate_label_and_never_fallback(self) -> None:
        frames = failed_video()
        bundle = evidence_bundle("failed-candidate.mp4", "candidate", frames)
        receipt = publication.build_scientific_review_receipt(
            frames,
            bundle["artifact"],
            candidate_decode_receipt=bundle["decode"],
            trusted_decoder_authority=DECODER_AUTHORITY,
        )
        self.assertEqual(receipt["decision"], "display_candidate_with_failure_label")
        self.assertFalse(receipt["action_editor_serving_authorized"])
        self.assertTrue(receipt["review_failure_label_required"])
        self.assertEqual(receipt["review_display_artifact"], bundle["artifact"])
        self.assertIsNone(receipt["fallback"])
        self.assertFalse(receipt["contract"]["fallback_substitution_allowed"])

    def test_scientific_visual_pass_never_authorizes_serving(self) -> None:
        frames = clean_moving_video()
        bundle = evidence_bundle("candidate.mp4", "candidate", frames)
        receipt = publication.build_scientific_review_receipt(
            frames,
            bundle["artifact"],
            candidate_decode_receipt=bundle["decode"],
            trusted_decoder_authority=DECODER_AUTHORITY,
        )
        self.assertTrue(receipt["visual_media_publishable"])
        self.assertFalse(receipt["action_editor_serving_authorized"])

    def test_closed_descriptors_and_exact81_receipt_are_enforced(self) -> None:
        frames = clean_moving_video()
        bundle = evidence_bundle("candidate.mp4", "candidate", frames)
        extra = copy.deepcopy(bundle["artifact"])
        extra["path"] = "/untrusted/path"
        with self.assertRaises(publication.PublicationInputError):
            publication.decide_serving_publication(
                frames, extra, **serving_kwargs(bundle)
            )

        short = frames[:-1]
        short_bundle = evidence_bundle("short.mp4", "candidate", short)
        with self.assertRaisesRegex(publication.PublicationInputError, "\[81,H,W,3\]"):
            publication.decide_serving_publication(
                short, short_bundle["artifact"], **serving_kwargs(short_bundle)
            )

    def test_final_receipt_digest_detects_mutation(self) -> None:
        frames = clean_moving_video()
        bundle = evidence_bundle("candidate.mp4", "candidate", frames)
        receipt = publication.decide_serving_publication(
            frames, bundle["artifact"], **serving_kwargs(bundle)
        )
        digest = receipt.pop("receipt_sha256")
        self.assertEqual(digest, publication._canonical_digest(receipt))
        receipt["decision"] = "forged"
        self.assertNotEqual(digest, publication._canonical_digest(receipt))


if __name__ == "__main__":
    unittest.main()
