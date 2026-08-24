from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if _TORCH_AVAILABLE:
    import torch
    import braid_stage_b_model_as_data_plan_v1 as stage_b
else:
    torch = None  # type: ignore[assignment]
    stage_b = None  # type: ignore[assignment]


@unittest.skipUnless(_TORCH_AVAILABLE, "torch required")
class BraidStageBMathOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def _metadata(self) -> dict[str, tuple]:
        rows = []
        for family in ("sit", "turn"):
            for identity in ("identity-0", "identity-1"):
                for scene in ("scene-0", "scene-1"):
                    for seed in (0, 1):
                        rows.append((family, identity, scene, seed, "fit"))
        rows.extend(
            (
                ("sit", "identity-2", "scene-0", 0, "identity_holdout"),
                ("sit", "identity-0", "scene-2", 0, "scene_holdout"),
                ("turn", "identity-0", "scene-0", 9, "seed_holdout"),
                ("jump", "identity-0", "scene-0", 0, "action_family_holdout"),
            )
        )
        return {
            "sample_ids": tuple(f"sample-{index}" for index in range(len(rows))),
            "action_family_ids": tuple(row[0] for row in rows),
            "identity_group_ids": tuple(row[1] for row in rows),
            "scene_group_ids": tuple(row[2] for row in rows),
            "sealed_seeds": tuple(row[3] for row in rows),
            "split_by_sample": tuple(row[4] for row in rows),
        }

    def _phase_tensor(self) -> torch.Tensor:
        metadata = self._metadata()
        value = torch.zeros(20, 4, 4, 32, dtype=torch.float32)
        offsets = {"sit": 0, "turn": 10, "jump": 20}
        fit_deltas = (-0.31, -0.23, -0.15, -0.07, 0.04, 0.12, 0.21, 0.30)
        family_seen = {"sit": 0, "turn": 0, "jump": 0}
        for sample, family in enumerate(metadata["action_family_ids"]):
            offset = offsets[family]
            within = family_seen[family]
            family_seen[family] += 1
            delta = fit_deltas[within] if sample < 16 else 0.09 + 0.01 * sample
            for stage in range(4):
                value[sample, 0, stage, offset + stage] = 1.0
                value[sample, 2, stage, offset + 6 + stage] = 1.0
                value[sample, 0, stage, 30] = delta * (1.0 + 0.11 * stage)
                value[sample, 0, stage, 31] = (delta * delta + 0.03 * delta) * (
                    1.0 + 0.07 * stage
                )
                value[sample, 2, stage, 30] = -0.83 * delta * (1.0 + 0.09 * stage)
                value[sample, 2, stage, 31] = (0.61 * delta * delta - 0.02 * delta) * (
                    1.0 + 0.05 * stage
                )
            for stage in range(2):
                value[sample, 3, stage, offset + 4 + stage] = 1.0
                value[sample, 3, stage, 30] = 0.57 * delta * (1.0 + 0.13 * stage)
                value[sample, 3, stage, 31] = (-0.47 * delta * delta + 0.04 * delta) * (
                    1.0 + 0.08 * stage
                )
        return value

    def _population(self, phase: torch.Tensor | None = None, metadata=None):
        metadata = metadata or self._metadata()
        return stage_b.build_untrusted_math_population(
            self._phase_tensor() if phase is None else phase,
            **metadata,
        )

    def _fixture(self, *, seed: int = 7):
        population = self._population()
        binding = stage_b.bind_pinned_canonical_registry(population)
        torch.manual_seed(seed)
        head = stage_b.BraidTextToPlanHead(
            stage_b.BraidTextToPlanConfig(token_width=8, hidden_width=32)
        )
        bundle = stage_b.compute_stage_b_objective(head, population, binding)
        return population, binding, head, bundle

    def test_public_contract_is_structurally_non_authorizing(self):
        contract = stage_b.public_input_contract()
        self.assertEqual(contract["purpose"], "non-authorizing mathematical and gradient preflight only")
        self.assertFalse(contract["raw_mapping_artifact_input"])
        self.assertFalse(contract["caller_media_score_input"])
        self.assertFalse(contract["caller_video_digest_input"])
        self.assertFalse(contract["caller_canonical_payload_input"])
        self.assertFalse(contract["semantic_evidence_authentication_available"])
        self.assertFalse(contract["hard_gate_available"])
        self.assertFalse(contract["freeze_authorization_available"])
        self.assertTrue(contract["freeze_assertion_structurally_raises"])
        self.assertFalse(hasattr(stage_b, "authenticate_pure_t2v_phase_evidence"))
        self.assertFalse(hasattr(stage_b, "authenticate_stage_b_hard_gate_audit"))
        self.assertFalse(hasattr(stage_b, "build_stage_b_freeze_receipt"))

    def test_untrusted_math_population_has_no_media_or_score_authority(self):
        population = self._population()
        receipt = population.receipt()
        self.assertEqual(receipt["source_trust"], "caller_supplied_detached_math_only")
        self.assertFalse(receipt["real_media_bytes_reopened"])
        self.assertFalse(receipt["independent_materializer_present"])
        self.assertFalse(receipt["independent_semantic_scorer_present"])
        self.assertFalse(receipt["semantic_evidence_authenticated"])
        self.assertFalse(receipt["freeze_authority"])
        serialized = stage_b.canonical_json_bytes(receipt).decode("ascii")
        self.assertNotIn("yield_score", serialized)
        self.assertNotIn("false_positive", serialized)
        self.assertNotIn("video_sha", serialized)
        with self.assertRaises(stage_b.BraidStageBError):
            stage_b.build_untrusted_math_population(  # type: ignore[arg-type]
                {"phase": self._phase_tensor()}, **self._metadata()
            )
        with self.assertRaises(TypeError):
            stage_b.build_untrusted_math_population(
                self._phase_tensor(),
                **self._metadata(),
                action_yield_score=1.0,
            )

    def test_registry_is_code_owned_sealed_and_caller_payloads_are_impossible(self):
        population = self._population()
        binding = stage_b.bind_pinned_canonical_registry(population)
        receipt = binding.receipt()
        self.assertEqual(receipt["registry_digest"], stage_b.PINNED_CANONICAL_REGISTRY_DIGEST)
        self.assertTrue(receipt["registry_is_code_owned_and_caller_payloads_are_rejected"])
        self.assertTrue(receipt["field_axis_preserved"])
        self.assertTrue(receipt["byte_position_axis_preserved"])
        self.assertFalse(receipt["truncation_allowed"])
        signature = str(inspect.signature(stage_b.bind_pinned_canonical_registry))
        self.assertNotIn("payload", signature)
        self.assertNotIn("registry", signature.split("(", 1)[1])
        with self.assertRaises(TypeError):
            stage_b.bind_pinned_canonical_registry(
                population, registry={"sit": "forged"}
            )
        with self.assertRaises(AttributeError):
            binding._token_ids = torch.zeros_like(binding._token_ids)
        object.__setattr__(binding, "_token_ids", torch.zeros_like(binding._token_ids))
        with self.assertRaisesRegex(stage_b.BraidStageBError, "live replay"):
            binding.assert_live()

    def test_field_position_encoder_breaks_bag_of_bytes_anagram_collision(self):
        torch.manual_seed(5)
        head = stage_b.BraidTextToPlanHead(
            stage_b.BraidTextToPlanConfig(token_width=8, hidden_width=32)
        )
        token_ids, mask = stage_b._anagram_canary_token_grids()
        left_bag = sorted(token_ids[0][mask[0]].tolist())
        right_bag = sorted(token_ids[1][mask[1]].tolist())
        self.assertEqual(left_bag, right_bag)

        token_only = head.token_embedding(token_ids)
        old_left_mean = token_only[0][mask[0]].mean(dim=0)
        old_right_mean = token_only[1][mask[1]].mean(dim=0)
        self.assertTrue(torch.allclose(old_left_mean, old_right_mean, atol=1.0e-7, rtol=0.0))

        ordered = head._ordered_features(token_ids, mask)
        self.assertFalse(torch.equal(ordered[0], ordered[1]))
        self.assertGreater(
            float((ordered[0] - ordered[1]).abs().max().item()),
            stage_b.PINNED_MIN_HEAD_GRADIENT_MAX_ABS,
        )

    def test_exact_head_type_rejects_override_forward_oracle(self):
        with self.assertRaisesRegex(TypeError, "final and cannot be subclassed"):
            class CheatingHead(stage_b.BraidTextToPlanHead):
                def forward(self, tokens, *, role="action"):
                    return torch.ones(tokens.batch_size, 4, 32, requires_grad=True)

    def test_design_rejects_fractional_factorial_and_unmatched_holdout(self):
        metadata = {key: list(value) for key, value in self._metadata().items()}
        parity_cells = (
            ("identity-0", "scene-0", 0),
            ("identity-0", "scene-0", 0),
            ("identity-0", "scene-1", 1),
            ("identity-0", "scene-1", 1),
            ("identity-1", "scene-0", 1),
            ("identity-1", "scene-0", 1),
            ("identity-1", "scene-1", 0),
            ("identity-1", "scene-1", 0),
        )
        for index, (identity, scene, seed) in enumerate(parity_cells):
            metadata["identity_group_ids"][index] = identity
            metadata["scene_group_ids"][index] = scene
            metadata["sealed_seeds"][index] = seed
        with self.assertRaisesRegex(stage_b.BraidStageBError, "complete balanced nuisance factorial"):
            self._population(metadata=metadata)

        unmatched = {key: list(value) for key, value in self._metadata().items()}
        unmatched["identity_group_ids"][17] = "identity-unmatched"
        with self.assertRaisesRegex(stage_b.BraidStageBError, "strict single-variable"):
            self._population(metadata=unmatched)

    def test_degenerate_population_and_same_stage_alias_fail_closed(self):
        degenerate = self._phase_tensor()
        families = self._metadata()["action_family_ids"]
        for family in ("sit", "turn"):
            members = [index for index in range(16) if families[index] == family]
            for index in members[1:]:
                degenerate[index] = degenerate[members[0]]
        with self.assertRaisesRegex(stage_b.BraidStageBError, "degenerate phase variation"):
            self._population(phase=degenerate)

        aliased = self._phase_tensor()
        aliased[0, 0, 1] = aliased[0, 0, 0]
        with self.assertRaisesRegex(stage_b.BraidStageBError, "wrong-stage distinction"):
            self._population(phase=aliased)

    def test_every_required_loss_is_nondegenerate_and_reaches_real_head(self):
        population, binding, head, bundle = self._fixture()
        self.assertEqual(tuple(bundle.action_plan.shape), (20, 4, 32))
        self.assertEqual(tuple(bundle.fit_margin_vector.shape), (16, 10))
        self.assertGreater(float(bundle.sample_phase_loss.item()), 0.0)
        self.assertGreater(float(bundle.robust_centroid_loss.item()), 0.0)
        self.assertGreater(float(bundle.margin_dispersion_loss.item()), 0.0)
        for name in ("sample_phase", "robust_centroid", "margin_dispersion"):
            self.assertGreater(
                bundle.gradient_max_abs_by_loss[name],
                stage_b.PINNED_MIN_HEAD_GRADIENT_MAX_ABS,
            )
        self.assertTrue(torch.equal(bundle.action_plan[0], bundle.action_plan[7]))
        self.assertTrue(torch.equal(bundle.action_plan[8], bundle.action_plan[15]))
        self.assertEqual(int(torch.count_nonzero(bundle.plan_by_role["noop"]).item()), 0)
        self.assertEqual(
            int(torch.count_nonzero(bundle.plan_by_role["incomplete"][:, 2:]).item()), 0
        )
        receipt = bundle.receipt()
        self.assertFalse(receipt["caller_math_is_semantic_evidence"])
        self.assertFalse(receipt["hard_gate_available"])
        self.assertFalse(receipt["freeze_authority"])
        self.assertFalse(receipt["semantic_action_editing_success_claim"])
        self.assertTrue(receipt["same_role_wrong_stage_negatives_used"])
        self.assertFalse(population.tensor(device=torch.device("cpu")).requires_grad)
        self.assertTrue(any(parameter.requires_grad for parameter in head.parameters()))

    def test_objective_bundle_replays_and_rejects_margin_or_head_mutation(self):
        _, _, head, bundle = self._fixture()
        with self.assertRaises((AttributeError, TypeError)):
            bundle.phase_margin_by_role_stage = {}
        with torch.no_grad():
            bundle.phase_margin_by_role_stage["action"].fill_(99.0)
        with self.assertRaisesRegex(stage_b.BraidStageBError, "live replay"):
            bundle.assert_live()

        _, _, other_head, other_bundle = self._fixture(seed=9)
        with torch.no_grad():
            next(other_head.parameters()).add_(0.01)
        with self.assertRaisesRegex(stage_b.BraidStageBError, "changed after graph build"):
            other_bundle.assert_live()

    def test_same_role_wrong_stage_is_in_the_margin_closure(self):
        evidence = self._phase_tensor()
        plan_by_role = {
            "action": evidence[:, 0].clone().requires_grad_(),
            "noop": torch.zeros_like(evidence[:, 1]),
            "reverse": evidence[:, 2].clone().requires_grad_(),
            "incomplete": evidence[:, 3].clone().requires_grad_(),
        }
        correct = stage_b._phase_margin_tensors(plan_by_role, evidence)
        permuted = evidence.clone()
        permuted[:, 0] = permuted[:, 0].roll(shifts=1, dims=1)
        wrong = stage_b._phase_margin_tensors(plan_by_role, permuted)
        self.assertGreater(
            float(correct["action"].mean().item()),
            float(wrong["action"].mean().item()) + 1.0,
        )

    def test_freeze_always_raises_even_for_valid_bundle_or_forged_receipt(self):
        *_, bundle = self._fixture()
        for candidate in (
            bundle,
            {"checkpoint_freeze_authorized": True},
            {"all_hard_gates_passed": True, "digest": "0" * 64},
            None,
        ):
            with self.assertRaisesRegex(
                stage_b.BraidStageBNotAuthorizingError,
                "structurally unavailable",
            ):
                stage_b.assert_stage_b_freeze_authorized(candidate)

    def test_module_has_no_optimizer_update_checkpoint_or_media_io(self):
        source = Path(stage_b.__file__).read_text(encoding="utf-8")
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("torch.save", source)
        self.assertNotIn("open(", source)
        contract = stage_b.public_input_contract()
        self.assertFalse(contract["optimizer_constructed"])
        self.assertFalse(contract["backward_called"])
        self.assertTrue(contract["autograd_gradient_canary_used"])
        self.assertFalse(contract["parameter_update_executed"])
        self.assertFalse(contract["checkpoint_io"])
        self.assertFalse(contract["media_io"])


if __name__ == "__main__":
    unittest.main()
