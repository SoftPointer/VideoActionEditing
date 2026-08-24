from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

try:
    import torch  # noqa: E402
except ImportError:
    torch = None

if torch is not None:
    import dclr_counterfactual_bank as counterfactual_bank  # noqa: E402
    import dclr_preference_batch as preference_batch  # noqa: E402
    import dclr_preference_objective as objective  # noqa: E402
    import dclr_preference_training_step as training_step  # noqa: E402
    import dclr_runtime_contract as runtime_contract  # noqa: E402
    import test_dclr_counterfactual_bank as bank_fixtures  # noqa: E402
else:  # pragma: no cover - dependency-light environments
    counterfactual_bank = None
    preference_batch = None
    objective = None
    training_step = None
    runtime_contract = None
    bank_fixtures = None


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DependencyLightSourceGuards(unittest.TestCase):
    def test_public_step_accepts_no_prediction_energy_or_loss(self) -> None:
        path = METHOD_ROOT / "dclr_preference_training_step.py"
        source = path.read_text(encoding="utf-8")
        self.assertIn("optimizer.step()", source)
        self.assertNotIn(".numpy(", source)
        if training_step is not None:
            parameters = inspect.signature(
                training_step.run_preference_training_step
            ).parameters
            for forbidden in (
                "prediction",
                "current_prediction",
                "reference_prediction",
                "energy",
                "loss",
                "reward",
                "route_evidence",
            ):
                self.assertNotIn(forbidden, parameters)


if torch is not None:

    class _PatchTransformer:
        dtype = torch.float32

        def patch_vae_latent(self, latent, source_id=None):
            _, _, phases, height, width = latent.shape
            tokens = phases * (height // 2) * (width // 2)
            flat = latent.float().mean(dim=1).reshape(1, -1)
            summary = flat[:, :tokens].unsqueeze(-1)
            packed = summary.expand(
                1, tokens, runtime_contract.PINNED_INNER_DIM
            ).contiguous()
            rope = torch.full(
                (1, 1, tokens, runtime_contract.PINNED_ROPE_DIM),
                complex(float(source_id), 1.0),
                dtype=torch.complex128,
            )
            return packed, rope


    class _CountingSGD(torch.optim.SGD):
        def __init__(self, params, *, lr=0.01):
            super().__init__(params, lr=lr)
            self.step_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)


    class _FakeSharedStepModel:
        def __init__(
            self,
            *,
            route_attention: str,
            trainable: bool,
            checkpoint_digest: str,
            leaf_output: bool = False,
            disconnect_index: int | None = None,
            zero_index: int | None = None,
            mutate_query: bool = False,
        ) -> None:
            self.training = trainable
            self.dclr_checkpoint_digest = checkpoint_digest
            self.calls: list[dict[str, object]] = []
            self.leaf_output = leaf_output
            self.disconnect_index = disconnect_index
            self.zero_index = zero_index
            self.mutate_query = mutate_query
            names = (
                f"diff_dec.transformer.blocks.0.{route_attention}.to_q."
                "lora_A.default.weight",
                f"diff_dec.transformer.blocks.0.{route_attention}.to_q."
                "lora_B.default.weight",
                f"diff_dec.transformer.blocks.0.{route_attention}.to_out.0."
                "lora_A.default.weight",
                f"diff_dec.transformer.blocks.0.{route_attention}.to_out.0."
                "lora_B.default.weight",
            )
            self._named: list[tuple[str, torch.nn.Parameter]] = []
            for index, name in enumerate(names):
                parameter = torch.nn.Parameter(
                    torch.tensor([0.05 + 0.01 * index], dtype=torch.float32),
                    requires_grad=trainable,
                )
                self._named.append((name, parameter))
            self._named.append(
                (
                    "diff_dec.transformer.base.weight",
                    torch.nn.Parameter(
                        torch.tensor([0.125], dtype=torch.float32),
                        requires_grad=False,
                    ),
                )
            )
            self.last_prediction = None

        def named_parameters(self):
            return iter(self._named)

        def shared_step(
            self,
            *,
            model_id,
            noisy_latents,
            timesteps,
            cond_embeds,
            rotary_embs,
            batch_vae_seqlen,
            batch_text_seqlen,
        ):
            self.calls.append(
                {
                    "model_id": model_id,
                    "noisy_latents": noisy_latents,
                    "timesteps": timesteps,
                    "cond_embeds": cond_embeds,
                    "rotary_embs": rotary_embs,
                    "batch_vae_seqlen": tuple(batch_vae_seqlen),
                    "batch_text_seqlen": tuple(batch_text_seqlen),
                }
            )
            if self.mutate_query:
                cond_embeds.add_(0.25)
            total = int(noisy_latents.shape[1])
            base = noisy_latents[:, :, : runtime_contract.PINNED_PATCH_DIM]
            positions = torch.linspace(
                -1.0,
                1.0,
                total,
                dtype=base.dtype,
                device=base.device,
            ).reshape(1, total, 1)
            prediction = 0.01 * base
            if self.leaf_output:
                return torch.zeros_like(prediction, requires_grad=True)
            for index, (_, parameter) in enumerate(self._named[:-1]):
                if index == self.disconnect_index:
                    continue
                multiplier = 0.0 if index == self.zero_index else 1.0
                pattern = torch.cos((index + 1.0) * positions) + 0.2 * index
                prediction = prediction + multiplier * parameter * pattern
            prediction = prediction + self._named[-1][1] * positions.square()
            self.last_prediction = prediction.detach().clone()
            return prediction


    class _FakeDiffusionCore:
        """Match official GEN_Wanx22: core owns shared_step, transformer params."""

        def __init__(self, transformer) -> None:
            self.transformer = transformer

        def shared_step(self, **kwargs):
            return self.transformer.shared_step(**kwargs)


@unittest.skipIf(torch is None, "torch is unavailable")
class PreferenceTrainingStepFixture(unittest.TestCase):
    def setUp(self) -> None:
        shape = (1, 16, 21, 2, 2)
        elements = 1
        for item in shape:
            elements *= item
        source = torch.linspace(-0.25, 0.75, elements).reshape(shape).float()
        winner = torch.linspace(-1.0, 0.5, elements).reshape(shape).float()
        loser = torch.linspace(0.75, -0.75, elements).reshape(shape).float()
        epsilon = torch.linspace(0.5, -0.5, elements).reshape(shape).float()
        sigma = torch.tensor([0.375], dtype=torch.float32)
        flow = objective.build_shared_pair_flow_state(
            winner, loser, epsilon, sigma
        )
        self.batch = preference_batch.build_packed_preference_batch(
            _PatchTransformer(), normalized_source=source, flow_state=flow
        )
        self.cond = torch.linspace(-0.2, 0.2, 10 * 32).reshape(1, 10, 32)
        self.text_lengths = (5, 5)
        self.model_id = "transformer_1"
        self.checkpoint = _sha("bernini-base-checkpoint")
        raw_sources = [
            bank_fixtures._source("source-a", "identity-a"),
            bank_fixtures._source("source-b", "identity-b"),
            bank_fixtures._source("source-c", "identity-c"),
            *[
                bank_fixtures._source(
                    f"reward-cal-step-{index:02d}",
                    f"identity-cal-step-{index:02d}",
                    split="reward_cal",
                    action_ontology_id=f"step-cal-action-{index:02d}",
                )
                for index in range(
                    counterfactual_bank.MIN_REWARD_CAL_SAMPLES
                )
            ],
        ]
        self.sources = counterfactual_bank.validate_source_action_records(
            raw_sources
        )
        self.manifest_sha = _sha("training-step-source-manifest")
        self.counterfactual_bank = bank_fixtures._bank(
            self.sources, self.manifest_sha
        )
        self.counterfactual_context = bank_fixtures._evidence_context(
            self.sources, self.manifest_sha
        )
        rollout_kwargs = {
            "action_axes": {"actor": True, "order": True, "contact": True},
            "preservation_axes": {"identity": True, "camera": True},
        }
        self.action_winner = bank_fixtures._rollout(
            self.sources["source-a"],
            self.counterfactual_bank,
            self.counterfactual_context,
            receipt_id="step-action-winner",
            seed=401,
            episode_id="step-action-episode",
            candidate_slot=0,
            **rollout_kwargs,
        )
        self.action_loser = bank_fixtures._rollout(
            self.sources["source-a"],
            self.counterfactual_bank,
            self.counterfactual_context,
            receipt_id="step-action-loser",
            seed=402,
            episode_id="step-action-episode",
            candidate_slot=1,
            action_axes={"actor": True, "order": False, "contact": True},
            preservation_axes={"identity": True, "camera": True},
        )
        self.preservation_winner = bank_fixtures._rollout(
            self.sources["source-a"],
            self.counterfactual_bank,
            self.counterfactual_context,
            receipt_id="step-preservation-winner",
            seed=403,
            episode_id="step-preservation-episode",
            candidate_slot=0,
            **rollout_kwargs,
        )
        self.preservation_loser = bank_fixtures._rollout(
            self.sources["source-a"],
            self.counterfactual_bank,
            self.counterfactual_context,
            receipt_id="step-preservation-loser",
            seed=404,
            episode_id="step-preservation-episode",
            candidate_slot=1,
            action_axes={"actor": True, "order": True, "contact": True},
            preservation_axes={"identity": False, "camera": True},
        )
        self.receipts = counterfactual_bank.validate_rollout_receipts(
            [
                self.action_winner,
                self.action_loser,
                self.preservation_winner,
                self.preservation_loser,
            ],
            self.sources,
            self.counterfactual_bank,
            split_ledger=self.counterfactual_context["ledger"],
            source_manifest_sha256=self.manifest_sha,
            artifacts_by_digest=self.counterfactual_context["artifacts"],
        )
        self.pair = bank_fixtures._pair(
            self.action_winner,
            self.action_loser,
            pair_type=objective.ACTION_NEARMISS,
        )
        self.preservation_pair = bank_fixtures._pair(
            self.preservation_winner,
            self.preservation_loser,
            pair_type=objective.PRESERVATION_NEARMISS,
        )
        self.pair_digest = self.pair["pair_digest"]
        self.binding = training_step.build_shared_step_binding(
            checkpoint_digest=self.checkpoint,
            pair_digest=self.pair_digest,
            batch=self.batch,
            cond_embeds=self.cond,
            batch_text_seqlen=self.text_lengths,
            model_id=self.model_id,
        )

    def make_models(
        self,
        *,
        route_attention="attn2",
        current_kwargs=None,
        reference_kwargs=None,
    ):
        current_kwargs = {} if current_kwargs is None else current_kwargs
        reference_kwargs = {} if reference_kwargs is None else reference_kwargs
        current = _FakeSharedStepModel(
            route_attention=route_attention,
            trainable=True,
            checkpoint_digest=self.checkpoint,
            **current_kwargs,
        )
        reference = _FakeSharedStepModel(
            route_attention=route_attention,
            trainable=False,
            checkpoint_digest=self.checkpoint,
            **reference_kwargs,
        )
        optimizer = _CountingSGD(
            [parameter for _, parameter in current._named if parameter.requires_grad]
        )
        return current, reference, optimizer

    def run_step(self, current, reference, optimizer, **overrides):
        values = {
            "current_model": current,
            "collection_reference_model": reference,
            "optimizer": optimizer,
            "batch": self.batch,
            "cond_embeds": self.cond,
            "batch_text_seqlen": self.text_lengths,
            "model_id": self.model_id,
            "pair": self.pair,
            "receipts_by_digest": self.receipts,
            "sources": self.sources,
            "counterfactual_bank_document": self.counterfactual_bank,
            "split_ledger": self.counterfactual_context["ledger"],
            "source_manifest_sha256": self.manifest_sha,
            "artifacts_by_digest": self.counterfactual_context["artifacts"],
            "current_binding": self.binding,
            "reference_binding": replace(self.binding),
            "beta": 0.5,
        }
        values.update(overrides)
        return training_step.run_preference_training_step(**values)


class ValidTrainingStepTests(PreferenceTrainingStepFixture):
    def test_tensor_digest_is_numpy_free_and_logical_layout_stable(self) -> None:
        value = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        noncontiguous = value.transpose(0, 1)
        copied = noncontiguous.contiguous().clone()
        self.assertEqual(
            training_step.tensor_content_sha256(
                noncontiguous, label="noncontiguous"
            ),
            training_step.tensor_content_sha256(copied, label="copied"),
        )

    def test_real_dual_forward_exact_split_backward_audit_and_step(self) -> None:
        current, reference, optimizer = self.make_models()
        before = [
            parameter.detach().clone()
            for _, parameter in current._named
            if parameter.requires_grad
        ]
        result = self.run_step(current, reference, optimizer)

        self.assertEqual(len(current.calls), 1)
        self.assertEqual(len(reference.calls), 1)
        self.assertEqual(optimizer.step_calls, 1)
        self.assertTrue(result.optimizer_step_performed)
        self.assertEqual(result.current_shared_step_calls, 1)
        self.assertEqual(result.reference_shared_step_calls, 1)
        self.assertEqual(result.binding, self.binding)
        self.assertEqual(result.route.active_adapter, objective.ACTION_ADAPTER)
        self.assertEqual(result.gradient_audit.route_attention, "attn2")
        self.assertEqual(
            set(result.gradient_audit.trainable_parameter_names),
            set(result.gradient_audit.finite_nonzero_gradient_names),
        )
        self.assertTrue(all(value > 0 for value in result.gradient_audit.gradient_l2_norms))
        self.assertTrue(all(parameter.grad is None for _, parameter in reference._named))
        after = [
            parameter.detach()
            for _, parameter in current._named
            if parameter.requires_grad
        ]
        self.assertTrue(all(not torch.equal(left, right) for left, right in zip(before, after)))

        n = self.batch.source_token_count
        selector = self.batch.candidate_target_selector
        manual_winner = objective.candidate_current_reference_target_tail_mse(
            current.last_prediction[:, : 2 * n].clone().requires_grad_(),
            reference.last_prediction[:, : 2 * n],
            self.batch.target_true_velocity[:, :n],
            selector,
        )
        manual_loser = objective.candidate_current_reference_target_tail_mse(
            current.last_prediction[:, 2 * n :].clone().requires_grad_(),
            reference.last_prediction[:, 2 * n :],
            self.batch.target_true_velocity[:, n:],
            selector,
        )
        self.assertAlmostEqual(
            result.winner_current_energy,
            float(manual_winner.current.item()),
            places=6,
        )
        self.assertAlmostEqual(
            result.loser_reference_energy,
            float(manual_loser.reference.item()),
            places=6,
        )
        self.assertEqual(
            current.calls[0]["batch_vae_seqlen"],
            (2 * n, 2 * n),
        )
        self.assertEqual(current.calls[0]["batch_text_seqlen"], (5, 5))

    def test_official_core_transformer_parameter_ownership_is_supported(self) -> None:
        current, reference, optimizer = self.make_models()
        result = self.run_step(
            _FakeDiffusionCore(current),
            _FakeDiffusionCore(reference),
            optimizer,
        )
        self.assertTrue(result.optimizer_step_performed)
        self.assertEqual(len(current.calls), 1)
        self.assertEqual(len(reference.calls), 1)

    def test_preservation_route_allows_only_attn1_qo_lora(self) -> None:
        binding = training_step.build_shared_step_binding(
            checkpoint_digest=self.checkpoint,
            pair_digest=self.preservation_pair["pair_digest"],
            batch=self.batch,
            cond_embeds=self.cond,
            batch_text_seqlen=self.text_lengths,
            model_id=self.model_id,
        )
        current, reference, optimizer = self.make_models(route_attention="attn1")
        result = self.run_step(
            current,
            reference,
            optimizer,
            pair=self.preservation_pair,
            current_binding=binding,
            reference_binding=replace(binding),
        )
        self.assertEqual(result.route.active_adapter, objective.IDENTITY_ADAPTER)
        self.assertEqual(result.gradient_audit.route_attention, "attn1")
        self.assertTrue(
            all(".attn1." in name for name in result.gradient_audit.trainable_parameter_names)
        )


class FailClosedTrainingStepTests(PreferenceTrainingStepFixture):
    def assert_failed_without_step(self, message, current, reference, optimizer, **kwargs):
        with self.assertRaisesRegex(
            training_step.DCLRPreferenceTrainingStepError, message
        ):
            self.run_step(current, reference, optimizer, **kwargs)
        self.assertEqual(optimizer.step_calls, 0)

    def test_every_forward_binding_axis_is_equal_and_recomputable(self) -> None:
        for field in (
            "checkpoint_digest",
            "query_digest",
            "pair_digest",
            "sigma_digest",
            "epsilon_digest",
            "rope_digest",
        ):
            with self.subTest(field=field):
                current, reference, optimizer = self.make_models()
                forged = replace(self.binding, **{field: _sha(f"forged:{field}")})
                self.assert_failed_without_step(
                    "bindings differ",
                    current,
                    reference,
                    optimizer,
                    reference_binding=forged,
                )

        current, reference, optimizer = self.make_models()
        forged = replace(self.binding, query_digest=_sha("same forged query"))
        self.assert_failed_without_step(
            "cannot be recomputed",
            current,
            reference,
            optimizer,
            current_binding=forged,
            reference_binding=replace(forged),
        )

    def test_model_checkpoint_and_actual_pair_are_bound(self) -> None:
        current, reference, optimizer = self.make_models()
        reference.dclr_checkpoint_digest = _sha("another base checkpoint")
        self.assert_failed_without_step(
            "base checkpoint binding",
            current,
            reference,
            optimizer,
        )

        current, reference, optimizer = self.make_models()
        self.assert_failed_without_step(
            "pair digest differs",
            current,
            reference,
            optimizer,
            pair=self.preservation_pair,
        )

        current, reference, optimizer = self.make_models()
        self.assert_failed_without_step(
            "beta",
            current,
            reference,
            optimizer,
            beta=0.0,
        )

    def test_forged_pair_label_or_route_axes_cannot_reach_forward(self) -> None:
        current, reference, optimizer = self.make_models()
        forged_label = deepcopy(self.pair)
        forged_label["pair_type"] = objective.PRESERVATION_NEARMISS
        bank_fixtures._seal(forged_label, "pair_digest")
        self.assert_failed_without_step(
            "preservation_nearmiss loser",
            current,
            reference,
            optimizer,
            pair=forged_label,
        )
        self.assertEqual(len(current.calls), 0)
        self.assertEqual(len(reference.calls), 0)

        current, reference, optimizer = self.make_models()
        forged_loser = deepcopy(self.action_loser)
        forged_loser["action_axis_pass"]["order"] = True
        forged_loser["action_pass"] = True
        forged_loser["joint_pass"] = True
        forged_loser["reward_evidence"][
            "action_axis_calibrated_margins"
        ]["order"] = 1.0
        bank_fixtures._reseal_rollout(forged_loser)
        receipts = dict(self.receipts)
        receipts[forged_loser["receipt_digest"]] = forged_loser
        forged_axes = deepcopy(self.pair)
        forged_axes["loser_receipt_digest"] = forged_loser["receipt_digest"]
        bank_fixtures._seal(forged_axes, "pair_digest")
        self.assert_failed_without_step(
            "cannot be recomputed",
            current,
            reference,
            optimizer,
            pair=forged_axes,
            receipts_by_digest=receipts,
        )
        self.assertEqual(len(current.calls), 0)
        self.assertEqual(len(reference.calls), 0)

    def test_rejects_leaf_fake_prediction_and_query_mutation(self) -> None:
        current, reference, optimizer = self.make_models(
            current_kwargs={"leaf_output": True}
        )
        self.assert_failed_without_step(
            "non-leaf result of the model graph",
            current,
            reference,
            optimizer,
        )

        current, reference, optimizer = self.make_models(
            current_kwargs={"mutate_query": True}
        )
        self.assert_failed_without_step(
            "mutated the bound query",
            current,
            reference,
            optimizer,
        )

    def test_reference_must_be_distinct_eval_frozen_and_grad_none(self) -> None:
        current, _, optimizer = self.make_models()
        self.assert_failed_without_step(
            "distinct objects", current, current, optimizer
        )

        current, reference, optimizer = self.make_models()
        reference.training = True
        self.assert_failed_without_step(
            "eval mode", current, reference, optimizer
        )

        current, reference, optimizer = self.make_models()
        reference._named[0][1].requires_grad_(True)
        self.assert_failed_without_step(
            "fully frozen", current, reference, optimizer
        )

        current, reference, optimizer = self.make_models()
        reference._named[0][1].grad = torch.ones_like(reference._named[0][1])
        self.assert_failed_without_step(
            "grad=None", current, reference, optimizer
        )

    def test_model_wide_route_and_optimizer_allowlists_fail_closed(self) -> None:
        current, reference, optimizer = self.make_models(route_attention="attn1")
        self.assert_failed_without_step(
            "trainability leak.*attn2",
            current,
            reference,
            optimizer,
        )

        current, reference, optimizer = self.make_models()
        current._named[-1][1].requires_grad_(True)
        optimizer = _CountingSGD(
            [parameter for _, parameter in current._named if parameter.requires_grad]
        )
        self.assert_failed_without_step(
            "trainability leak", current, reference, optimizer
        )

        current, reference, _ = self.make_models()
        optimizer = _CountingSGD(
            [
                *[
                    parameter
                    for _, parameter in current._named
                    if parameter.requires_grad
                ],
                current._named[-1][1],
            ]
        )
        self.assert_failed_without_step(
            "optimizer parameters differ", current, reference, optimizer
        )

    def test_none_zero_and_nonfinite_gradients_abort_before_step(self) -> None:
        current, reference, optimizer = self.make_models(
            current_kwargs={"disconnect_index": 0}
        )
        self.assert_failed_without_step(
            "grad=None", current, reference, optimizer
        )

        current, reference, optimizer = self.make_models(
            current_kwargs={"zero_index": 1}
        )
        self.assert_failed_without_step(
            "zero gradient", current, reference, optimizer
        )

        current, reference, optimizer = self.make_models()
        current._named[2][1].register_hook(
            lambda gradient: torch.full_like(gradient, float("nan"))
        )
        self.assert_failed_without_step(
            "non-finite/malformed gradient",
            current,
            reference,
            optimizer,
        )


if __name__ == "__main__":
    unittest.main()
