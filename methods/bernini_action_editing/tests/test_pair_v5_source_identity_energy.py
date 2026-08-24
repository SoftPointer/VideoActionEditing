from __future__ import annotations

import inspect
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
    import pair_v5_source_identity_energy as binding  # noqa: E402
    import source_self_native_ref_contrastive_v3 as native  # noqa: E402
else:  # pragma: no cover - dependency-light environments
    binding = None
    native = None


if torch is not None:
    def _pack_spatial(value: torch.Tensor) -> torch.Tensor:
        batch, channels, phases, height, width = value.shape
        return (
            value.reshape(
                batch,
                channels,
                phases,
                1,
                height // 2,
                2,
                width // 2,
                2,
            )
            .permute(0, 2, 4, 6, 3, 5, 7, 1)
            .reshape(batch, phases * (height // 2) * (width // 2), 64)
        )


    class _Transformer(torch.nn.Module):
        dtype = torch.float32

        def __init__(self) -> None:
            super().__init__()
            self.patch_calls: list[dict[str, object]] = []

        def patch_vae_latent(self, value, *, source_id):
            self.patch_calls.append(
                {
                    "input_id": id(value),
                    "source_id": float(source_id),
                    "shape": tuple(value.shape),
                }
            )
            packed = _pack_spatial(value)
            padding = torch.zeros(
                int(packed.shape[0]),
                int(packed.shape[1]),
                1536 - 64,
                dtype=packed.dtype,
                device=packed.device,
            )
            tokens = torch.cat((packed, padding), dim=2)
            count = int(tokens.shape[1])
            real = torch.arange(count * 64, dtype=torch.float64).reshape(
                1, 1, count, 64
            )
            imag = torch.full_like(real, float(source_id))
            rotary = torch.complex(real, imag).to(device=value.device)
            return tokens, rotary


    class _FrozenDiffusion(torch.nn.Module):
        def __init__(self, transformer: _Transformer, *, ignore_conditions: bool = False) -> None:
            super().__init__()
            self.transformer = transformer
            self.ignore_conditions = ignore_conditions
            self.calls: list[dict[str, object]] = []

        def shared_step(self, **kwargs):
            tokens = kwargs["noisy_latents"]
            target_tokens = 21
            condition_tokens = int(tokens.shape[1]) - target_tokens
            if self.ignore_conditions:
                scalar = torch.zeros((), dtype=torch.float32, device=tokens.device)
            else:
                visual = (
                    tokens[:, :condition_tokens, :64].float().mean()
                    if condition_tokens
                    else torch.zeros((), dtype=torch.float32, device=tokens.device)
                )
                text_code = kwargs["cond_embeds"][0, 0, 0].float()
                scalar = 0.2 * visual + 0.2 * text_code
            self.calls.append(
                {
                    "tokens_id": id(tokens),
                    "timestep_id": id(kwargs["timesteps"]),
                    "total_tokens": int(tokens.shape[1]),
                    "condition_tokens": condition_tokens,
                }
            )
            return torch.ones(
                1,
                int(tokens.shape[1]),
                64,
                dtype=torch.float32,
                device=tokens.device,
            ) * scalar


    def _video(value: float) -> torch.Tensor:
        return torch.full((1, 16, 21, 2, 2), value, dtype=torch.float32)


    def _refs(value: float) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.full((1, 16, 1, 2, 2), value + 0.01 * index, dtype=torch.float32)
            for index in range(4)
        )


    def _condition(value: float) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float32).reshape(1, 1, 1).expand(
            1, 512, 4096
        )


    def _native_coordinate(index: int = 20):
        return (
            torch.tensor([native.NATIVE_UNIPC40_SIGMAS[index]], dtype=torch.float32),
            torch.tensor([native.NATIVE_UNIPC40_TIMESTEPS[index]], dtype=torch.float32),
        )


    def _score(value: float) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float32)


@unittest.skipIf(torch is None, "torch is unavailable")
class PairV5SourceBindingEnergyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transformer = _Transformer()
        self.diffusion = _FrozenDiffusion(self.transformer)
        self.diffusion.eval()
        self.transformer.eval()
        self.scorer = binding.FrozenBerniniRV2V4IdentityScorer(
            self.diffusion,
            self.transformer,
            _condition(1.0),
            _condition(0.0),
            _condition(-1.0),
            frozen_model_receipt_digest="a" * 64,
        )
        self.clean = _video(0.1)
        self.epsilon = _video(1.0)
        self.correct_video = _video(0.0)
        self.correct_refs = _refs(0.4)
        self.wrong_video = _video(2.0)
        self.wrong_refs = _refs(2.0)
        self.correct = self.bundle(
            "correct-iid", self.correct_video, self.correct_refs, "1"
        )
        self.wrong = self.bundle("wrong-iid", self.wrong_video, self.wrong_refs, "2")
        self.policy = binding.seal_source_binding_preregistration(
            policy_id="source-binding-policy-v1",
            minimum_source_binding_margin=0.001,
            unit_score_saturation_margin=1.0,
        )
        self.sigma, self.timestep = _native_coordinate()

    def bundle(self, key, video, refs, digit):
        provenance = binding.seal_source_condition_provenance(
            source_key=key,
            source_media_artifact_sha256=digit * 64,
            source_media_receipt_digest=hex((int(digit, 16) + 2) % 16)[2:] * 64,
            source_video=video,
            source_references=refs,
            full_video_encoding_receipt_digest=hex((int(digit, 16) + 4) % 16)[2:] * 64,
            reference_extraction_receipt_digest=hex((int(digit, 16) + 6) % 16)[2:] * 64,
        )
        return binding.make_source_condition_bundle(video, refs, provenance)

    def evaluate(self, **updates):
        kwargs = {
            "clean_candidate": self.clean,
            "epsilon": self.epsilon,
            "sigma": self.sigma,
            "timestep": self.timestep,
            "correct_source": self.correct,
            "wrong_source": self.wrong,
            "scorer": self.scorer,
            "preregistration": self.policy,
            "postvideo_camera_score": _score(0.91),
            "postvideo_background_score": _score(0.82),
            "postvideo_quality_score": _score(0.73),
            "candidate_receipt_digest": "b" * 64,
            "registered_preregistration_digest": self.policy.receipt_digest,
            "postvideo_evaluator_receipt_digest": "e" * 64,
        }
        kwargs.update(updates)
        return binding.evaluate_candidate_source_binding_energy(**kwargs)

    def test_correct_source_wins_and_emits_unit_safe_pareto_packet(self) -> None:
        result = self.evaluate()

        self.assertEqual(tuple(result.cell_velocities), binding.CELL_ORDER)
        self.assertEqual(tuple(result.counterfactual_gaps), binding.COUNTERFACTUAL_CELL_ORDER)
        self.assertTrue(result.source_binding_pass)
        self.assertGreater(float(result.source_binding_margin), 0.001)
        self.assertGreaterEqual(float(result.source_binding_score), 0.0)
        self.assertLessEqual(float(result.source_binding_score), 1.0)
        self.assertTrue(result.safe_pareto_packet.selection_authorized)
        self.assertEqual(
            result.safe_pareto_packet.receipt["safe_pareto_axis"], "identity"
        )
        self.assertEqual(
            result.safe_pareto_packet.receipt["identity_score"],
            result.safe_pareto_packet.unit_interval_score,
        )
        self.assertEqual(
            result.safe_pareto_packet.receipt["score_semantics"],
            binding.SOURCE_BINDING_SCORE_SEMANTICS,
        )
        self.assertFalse(
            result.safe_pareto_packet.evaluator_provenance["pure_actor_identity_claim"]
        )
        self.assertTrue(
            result.safe_pareto_packet.evaluator_provenance[
                "full_source_identity_background_camera_old_motion_entangled"
            ]
        )
        self.assertEqual(len(self.diffusion.calls), 13)
        target_patch_ids = [
            row["input_id"]
            for row in self.transformer.patch_calls
            if row["source_id"] == 0.0
        ]
        self.assertEqual(len(target_patch_ids), 2)
        self.assertEqual(len(set(target_patch_ids)), 1)

        receipt = dict(result.receipt)
        digest = receipt.pop("digest")
        self.assertEqual(digest, binding.object_sha256(receipt))
        self.assertTrue(receipt["candidate_exact_null_copy_rejected"])
        self.assertEqual(
            receipt["source_binding_score_semantics"],
            binding.SOURCE_BINDING_SCORE_SEMANTICS,
        )
        self.assertFalse(receipt["pure_actor_identity_claim"])

    def test_tie_and_exact_threshold_are_not_authorized(self) -> None:
        transformer = _Transformer()
        diffusion = _FrozenDiffusion(transformer, ignore_conditions=True)
        transformer.eval()
        diffusion.eval()
        tie_scorer = binding.FrozenBerniniRV2V4IdentityScorer(
            diffusion,
            transformer,
            _condition(1.0),
            _condition(0.0),
            _condition(-1.0),
            frozen_model_receipt_digest="f" * 64,
        )
        tie = self.evaluate(scorer=tie_scorer)
        self.assertFalse(tie.source_binding_pass)
        self.assertEqual(float(tie.source_binding_margin), 0.0)
        self.assertEqual(float(tie.source_binding_score), 0.0)
        self.assertFalse(tie.safe_pareto_packet.selection_authorized)

        first = self.evaluate()
        exact = float(first.source_binding_margin.item())
        exact_policy = binding.seal_source_binding_preregistration(
            policy_id="exact-threshold-policy",
            minimum_source_binding_margin=exact,
            unit_score_saturation_margin=exact + 1.0,
        )
        second = self.evaluate(
            preregistration=exact_policy,
            registered_preregistration_digest=exact_policy.receipt_digest,
        )
        self.assertFalse(second.source_binding_pass)
        self.assertEqual(float(second.source_binding_score), 0.0)

    def test_policy_must_be_strict_positive_and_externally_registered(self) -> None:
        with self.assertRaisesRegex(
            binding.PairV5SourceIdentityEnergyError, "strictly positive"
        ):
            binding.seal_source_binding_preregistration(
                policy_id="zero-policy",
                minimum_source_binding_margin=0.0,
                unit_score_saturation_margin=1.0,
            )
        with self.assertRaisesRegex(
            binding.PairV5SourceIdentityEnergyError, "strictly positive"
        ):
            binding.seal_source_binding_preregistration(
                policy_id="underflow-policy",
                minimum_source_binding_margin=1.0e-50,
                unit_score_saturation_margin=1.0,
            )
        with self.assertRaisesRegex(
            binding.PairV5SourceIdentityEnergyError, "preregistered digest"
        ):
            self.evaluate(registered_preregistration_digest="f" * 64)
        with self.assertRaisesRegex(
            binding.PairV5SourceIdentityEnergyError, "typed preregistration"
        ):
            self.evaluate(preregistration={"minimum": 0.001})

    def test_exact_null_copy_and_candidate_source_alias_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            binding.PairV5SourceIdentityEnergyError, "exact null-copy"
        ):
            self.evaluate(clean_candidate=self.correct_video.clone())
        aliased = self.correct_video.view_as(self.correct_video)
        with self.assertRaisesRegex(
            binding.PairV5SourceIdentityEnergyError, "null-copy|alias"
        ):
            self.evaluate(clean_candidate=aliased)

    def test_clone_references_are_content_not_source_disjoint(self) -> None:
        cloned_refs = tuple(item.clone() for item in self.correct_refs)
        wrong = self.bundle("wrong-cloned-refs", self.wrong_video, cloned_refs, "9")
        with self.assertRaisesRegex(
            binding.PairV5SourceIdentityEnergyError, "condition content hashes"
        ):
            self.evaluate(wrong_source=wrong)

        cloned_video = self.bundle(
            "wrong-cloned-video", self.correct_video.clone(), self.wrong_refs, "8"
        )
        with self.assertRaisesRegex(
            binding.PairV5SourceIdentityEnergyError, "not source-disjoint"
        ):
            self.evaluate(wrong_source=cloned_video)

        same_origin = self.bundle(
            "correct-iid", self.wrong_video, _refs(3.0), "7"
        )
        with self.assertRaisesRegex(
            binding.PairV5SourceIdentityEnergyError, "not source-disjoint"
        ):
            self.evaluate(wrong_source=same_origin)

    def test_cross_swapped_media_and_provenance_fail_before_model_call(self) -> None:
        before = len(self.diffusion.calls)
        cross = binding.SourceConditionBundle(
            self.wrong_video,
            self.correct_refs,
            self.correct.provenance,
        )
        with self.assertRaisesRegex(
            binding.PairV5SourceIdentityEnergyError, "full video differs"
        ):
            self.evaluate(correct_source=cross)
        self.assertEqual(len(self.diffusion.calls), before)

        cross_refs = binding.SourceConditionBundle(
            self.correct_video,
            self.wrong_refs,
            self.correct.provenance,
        )
        with self.assertRaisesRegex(
            binding.PairV5SourceIdentityEnergyError, "references differ"
        ):
            self.evaluate(correct_source=cross_refs)
        self.assertEqual(len(self.diffusion.calls), before)

    def test_opaque_source_digest_cannot_replace_typed_provenance(self) -> None:
        with self.assertRaisesRegex(
            binding.PairV5SourceIdentityEnergyError, "typed.*opaque digest"
        ):
            binding.make_source_condition_bundle(
                self.correct_video,
                self.correct_refs,
                "a" * 64,
            )
        with self.assertRaisesRegex(
            binding.PairV5SourceIdentityEnergyError, "typed correct/wrong"
        ):
            self.evaluate(correct_source="a" * 64)

    def test_fixed_reference_indices_and_content_hashes_are_bound(self) -> None:
        self.assertEqual(
            self.correct.provenance.reference_frame_indices,
            binding.REFERENCE_FRAME_INDICES,
        )
        tampered = binding.SourceConditionProvenance(
            **{
                **self.correct.provenance.__dict__,
                "reference_frame_indices": (0, 1, 2, 3),
            }
        )
        with self.assertRaisesRegex(
            binding.PairV5SourceIdentityEnergyError, r"exactly \[0,27,53,80\]"
        ):
            tampered.validate()
        changed_refs = list(self.correct_refs)
        changed_refs[2] = changed_refs[2].clone()
        changed_refs[2].reshape(-1)[0] += 0.25
        with self.assertRaisesRegex(
            binding.PairV5SourceIdentityEnergyError, "references differ"
        ):
            binding.make_source_condition_bundle(
                self.correct_video,
                changed_refs,
                self.correct.provenance,
            )

    def test_postvideo_scores_cannot_change_model_energy(self) -> None:
        first = self.evaluate()
        first_energies = {name: value.clone() for name, value in first.cell_energies.items()}
        call_count = len(self.diffusion.calls)
        second = self.evaluate(
            postvideo_camera_score=_score(0.01),
            postvideo_background_score=_score(0.02),
            postvideo_quality_score=_score(0.03),
        )
        self.assertEqual(len(self.diffusion.calls) - call_count, 13)
        for name in binding.CELL_ORDER:
            torch.testing.assert_close(second.cell_energies[name], first_energies[name])
        self.assertFalse(second.receipt["postvideo_scores_modify_source_binding_energy"])

    def test_geometry_coordinate_and_score_types_fail_closed(self) -> None:
        with self.assertRaisesRegex(binding.PairV5SourceIdentityEnergyError, "exact81"):
            self.evaluate(clean_candidate=self.clean.double())
        with self.assertRaisesRegex(
            binding.PairV5SourceIdentityEnergyError, "outside native exact40"
        ):
            self.evaluate(timestep=torch.tensor([500.0], dtype=torch.float32))
        with self.assertRaisesRegex(
            binding.PairV5SourceIdentityEnergyError, "detached finite FP32"
        ):
            self.evaluate(postvideo_quality_score=torch.tensor(0.7, dtype=torch.float64))
        with self.assertRaisesRegex(binding.PairV5SourceIdentityEnergyError, r"\[0,1\]"):
            self.evaluate(postvideo_quality_score=_score(1.1))

    def test_non_explanatory_correct_source_is_rejected(self) -> None:
        result = self.evaluate(correct_source=self.wrong, wrong_source=self.correct)
        self.assertFalse(result.source_binding_pass)
        self.assertLess(float(result.counterfactual_gaps["wrong_source"]), 0.0)
        self.assertFalse(result.safe_pareto_packet.selection_authorized)

    def test_public_contract_uses_typed_source_binding_semantics(self) -> None:
        signatures = (
            inspect.signature(binding.FrozenBerniniRV2V4IdentityScorer.__init__),
            inspect.signature(binding.FrozenBerniniRV2V4IdentityScorer.forward),
            inspect.signature(binding.evaluate_candidate_source_binding_energy),
        )
        for signature in signatures:
            self.assertFalse(
                set(signature.parameters) & binding.FORBIDDEN_PUBLIC_INPUT_NAMES
            )
        contract = binding.contract_receipt()
        unsigned = dict(contract)
        digest = unsigned.pop("digest")
        self.assertEqual(digest, binding.object_sha256(unsigned))
        self.assertEqual(
            contract["reference_frame_indices"], list(binding.REFERENCE_FRAME_INDICES)
        )
        self.assertTrue(contract["typed_source_bundles_required"])
        self.assertTrue(contract["strictly_positive_preregistered_margin_required"])
        self.assertTrue(contract["exact_null_copy_rejected"])
        self.assertFalse(contract["pure_actor_identity_claim"])
        self.assertEqual(
            contract["unit_interval_score_semantics"],
            binding.SOURCE_BINDING_SCORE_SEMANTICS,
        )


if __name__ == "__main__":
    unittest.main()
