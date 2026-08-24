from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
for root in (METHOD_ROOT, TEST_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import caper_phase_action_quotient_probe as paq  # noqa: E402
import caper_sigma_gated_target_row_lora as lora  # noqa: E402
import caper_train_authorization_v1 as auth  # noqa: E402
from test_caper_phase_action_quotient_probe import (  # noqa: E402
    fake_decode_exact81,
    good_trials,
    make_bundle,
)
from test_caper_sigma_gated_target_row_lora import _Renderer  # noqa: E402
from test_caper_stage1_preference_admission_v1 import (  # noqa: E402
    CAPERManifestFixture,
)


class _Authority:
    def __init__(self, state: lora.CAPERParallelState) -> None:
        self.state = state

    def snapshot(self) -> lora.CAPERParallelState:
        return self.state


class CAPERTrainAuthorizationTests(CAPERManifestFixture):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(9017)
        self.renderer = _Renderer(hidden=8)
        self.renderer.requires_grad_(False)
        self.handle = lora.install_caper_capacity_probe(
            self.renderer, expected_hidden_size=8
        )
        self.live_state = lora.CAPERParallelState(
            world_size=4,
            world_rank=0,
            sequence_parallel_group_ranks=(0, 1, 2, 3),
            sequence_parallel_rank=0,
            authority_id="torch.distributed+bernini.parallel.get_parallel_state",
            test_only=False,
        )
        self.live_patch = mock.patch.object(
            lora, "snapshot_live_bernini_parallel_state", return_value=self.live_state
        )
        self.live_snapshot = self.live_patch.start()

    def tearDown(self) -> None:
        self.live_patch.stop()
        if not self.handle.restored:
            self.handle.restore()
        super().tearDown()

    def _admitted_paq(
        self,
        *,
        source_revision: str | None = None,
        checkpoint: str | None = None,
        policy: str | None = None,
        exposure: str | None = None,
        action_family: str = "sit-down",
    ):
        manifest, hidden, states = make_bundle()
        checkpoint = self.policy["checkpoint_tree_sha256"] if checkpoint is None else checkpoint
        policy = self.policy["inference_contract_sha256"] if policy is None else policy
        exposure = (
            self.manifest["exposure_ledger"]["sha256"]
            if exposure is None
            else exposure
        )
        records = [dict(row) for row in manifest["records"]]
        for row in records:
            row["checkpoint_sha256"] = checkpoint
            row["policy_sha256"] = policy
            row["source_exposure_registry_sha256"] = exposure
            row["action_family_id"] = action_family
            row["requested_action_id"] = "sit-down-and-hold"
        manifest = paq.seal_observation_manifest(
            records,
            probe_id="auth-unit-paq",
            checkpoint_sha256=checkpoint,
            policy_sha256=policy,
            source_revision_sha256=(
                self.source_revision_sha256
                if source_revision is None
                else source_revision
            ),
            source_exposure_registry_sha256=exposure,
            intervention_scale=manifest["intervention_scale"],
        )
        audit = paq.observational_audit(manifest, hidden, states)
        with tempfile.TemporaryDirectory() as directory:
            trials = good_trials(
                manifest,
                audit.candidate_code_sha256,
                Path(directory).resolve(),
            )
            with mock.patch.object(
                paq, "_decode_exact81_media", side_effect=fake_decode_exact81
            ):
                decision = paq.decide_phase_action_quotient(
                    manifest, hidden, states, causal_trials=trials
                )
        self.assertTrue(decision.admitted_code)
        return decision, manifest

    @staticmethod
    def _factory_spy():
        calls: list[tuple[torch.nn.Parameter, ...]] = []

        def factory(parameters):
            owned = tuple(parameters)
            calls.append(owned)
            return torch.optim.SGD(owned, lr=1.0e-4)

        return calls, factory

    def _dry_run_forward(self):
        hidden = torch.linspace(-0.75, 0.85, 24, dtype=torch.float32).reshape(1, 3, 8)
        outputs = []
        for _, wrapper in self.handle.q_wrappers + self.handle.o_wrappers:
            outputs.append(wrapper(hidden).square().mean())
        return torch.stack(outputs).sum()

    def _native_sp4_route(self):
        state = lora.CAPERParallelState(
            world_size=4,
            world_rank=0,
            sequence_parallel_group_ranks=(0, 1, 2, 3),
            sequence_parallel_rank=0,
            authority_id="unit-native-sp4-authority",
            test_only=False,
        )
        segments = lora.preference_pack_segments(
            source_tokens=2, target_tokens=3
        )
        route = lora.CAPERRoute.from_runtime_sigma(
            global_target_selector=lora.preference_pack_target_selector(
                source_tokens=2, target_tokens=3
            ),
            pack_segments=segments,
            parallel_state_authority=_Authority(state),
            sigma_schedule_index=0,
            sigma=lora.sigma_strata.PINNED_POSITIVE_SIGMAS[0],
        )
        return route

    def test_no_fit_pair_never_calls_factory_and_checksums_absent_state(self) -> None:
        empty = deepcopy(self.manifest)
        empty["splits"]["fit"] = []
        self._sync_ledger_to_manifest(empty)
        stage = self.materialize(empty)
        before = self.handle.trainable_parameter_values_sha256()
        calls, factory = self._factory_spy()
        result = auth.authorize_caper_training_and_create_optimizer(
            paq_decision=None,
            paq_manifest={},
            stage1_materialization=stage,
            caper_handle=self.handle,
            caper_route=None,
            optimizer_factory=factory,
        )
        self.assertFalse(result.authorized)
        self.assertIsNone(result.optimizer)
        self.assertEqual(calls, [])
        self.assertFalse(result.receipt["optimizer_created"])
        self.assertEqual(
            result.receipt["optimizer_state_before_sha256"],
            result.receipt["optimizer_state_after_sha256"],
        )
        self.assertEqual(
            result.receipt["caper_adapter_before_sha256"], before
        )
        self.assertEqual(
            result.receipt["caper_adapter_before_sha256"],
            result.receipt["caper_adapter_after_sha256"],
        )
        auth.verify_authorization_receipt(result.receipt)
        unsigned = dict(result.receipt)
        seal = unsigned.pop("authorization_receipt_sha256")
        self.assertEqual(seal, auth._object_sha256(unsigned))

    def test_tampered_paq_decision_cannot_reach_optimizer_factory(self) -> None:
        stage = self.materialize()
        decision, manifest = self._admitted_paq()
        attacked = replace(decision, training_updates_authorized=0)
        calls, factory = self._factory_spy()
        with self.assertRaisesRegex(
            auth.CAPERTrainAuthorizationError, "decision receipt seal"
        ):
            auth.authorize_caper_training_and_create_optimizer(
                paq_decision=attacked,
                paq_manifest=manifest,
                stage1_materialization=stage,
                caper_handle=self.handle,
                caper_route=self._native_sp4_route(),
                optimizer_factory=factory,
            )
        self.assertEqual(calls, [])

    def test_common_source_ledger_mismatch_blocks_before_factory(self) -> None:
        stage = self.materialize()
        decision, manifest = self._admitted_paq(source_revision="9" * 64)
        calls, factory = self._factory_spy()
        with self.assertRaisesRegex(
            auth.CAPERTrainAuthorizationError,
            "common authority mismatch: source_revision_sha256",
        ):
            auth.authorize_caper_training_and_create_optimizer(
                paq_decision=decision,
                paq_manifest=manifest,
                stage1_materialization=stage,
                caper_handle=self.handle,
                caper_route=self._native_sp4_route(),
                optimizer_factory=factory,
            )
        self.assertEqual(calls, [])

    def test_checkpoint_exposure_and_action_mismatches_each_block_factory(self) -> None:
        attacks = (
            ({"checkpoint": "7" * 64}, "checkpoint_tree_sha256"),
            ({"policy": "8" * 64}, "inference_contract_sha256"),
            ({"exposure": "6" * 64}, "exposure_ledger_artifact_sha256"),
            ({"action_family": "stand-up"}, "fit action families"),
        )
        for kwargs, reason in attacks:
            with self.subTest(reason=reason):
                stage = self.materialize()
                decision, manifest = self._admitted_paq(**kwargs)
                calls, factory = self._factory_spy()
                with self.assertRaisesRegex(
                    auth.CAPERTrainAuthorizationError, reason
                ):
                    auth.authorize_caper_training_and_create_optimizer(
                        paq_decision=decision,
                        paq_manifest=manifest,
                        stage1_materialization=stage,
                        caper_handle=self.handle,
                        caper_route=self._native_sp4_route(),
                        optimizer_factory=factory,
                    )
                self.assertEqual(calls, [])

    def test_caller_sp1_route_does_not_override_live_sp4_authority(self) -> None:
        stage = self.materialize()
        decision, manifest = self._admitted_paq()
        state = lora.CAPERParallelState(
            world_size=1,
            world_rank=0,
            sequence_parallel_group_ranks=(0,),
            sequence_parallel_rank=0,
            authority_id=lora.SP1_TEST_AUTHORITY_ID,
            test_only=True,
        )
        route = lora.CAPERRoute.from_runtime_sigma(
            global_target_selector=lora.preference_pack_target_selector(
                source_tokens=2, target_tokens=3
            ),
            pack_segments=lora.preference_pack_segments(
                source_tokens=2, target_tokens=3
            ),
            parallel_state_authority=_Authority(state),
            sigma_schedule_index=0,
            sigma=lora.sigma_strata.PINNED_POSITIVE_SIGMAS[0],
        )
        result = auth.authorize_caper_training_and_create_optimizer(
            paq_decision=decision,
            paq_manifest=manifest,
            stage1_materialization=stage,
            caper_handle=self.handle,
            caper_route=route,
            dry_run_forward=self._dry_run_forward,
        )
        self.assertTrue(result.authorized)
        self.assertEqual(self.live_snapshot.call_count, 1)
        self.assertEqual(
            result.receipt["live_parallel_state_receipt"]["sequence_parallel_size"],
            4,
        )

    def test_forged_route_mapping_cannot_replace_authority_constructed_route(self) -> None:
        stage = self.materialize()
        decision, manifest = self._admitted_paq()
        forged = dict(self._native_sp4_route().receipt())
        calls, factory = self._factory_spy()
        with self.assertRaisesRegex(
            auth.CAPERTrainAuthorizationError, "exact authority-constructed"
        ):
            auth.authorize_caper_training_and_create_optimizer(
                paq_decision=decision,
                paq_manifest=manifest,
                stage1_materialization=stage,
                caper_handle=self.handle,
                caper_route=forged,
                optimizer_factory=factory,
            )
        self.assertEqual(calls, [])

    def test_all_authorities_run_fixed_adamw_and_sealed_real_step(self) -> None:
        stage = self.materialize()
        decision, manifest = self._admitted_paq()
        result = auth.authorize_caper_training_and_create_optimizer(
            paq_decision=decision,
            paq_manifest=manifest,
            stage1_materialization=stage,
            caper_handle=self.handle,
            caper_route=self._native_sp4_route(),
            dry_run_forward=self._dry_run_forward,
        )
        self.assertTrue(result.authorized)
        self.assertIsNotNone(result.optimizer)
        self.assertIs(type(result.optimizer), torch.optim.AdamW)
        self.assertEqual(
            {
                id(item)
                for group in result.optimizer.param_groups
                for item in group["params"]
            },
            {
                id(parameter)
                for _, parameter in self.handle.trainable_named_parameters()
            },
        )
        common = result.receipt["common_authority_ledger"]
        unsigned = dict(common)
        seal = unsigned.pop("ledger_sha256")
        self.assertEqual(seal, auth._object_sha256(unsigned))
        self.assertEqual(
            common["exposure_ledger_artifact_sha256"],
            self.manifest["exposure_ledger"]["sha256"],
        )
        self.assertEqual(common["action_family_id"], "sit-down")
        dry = result.receipt["dry_run_receipt"]
        self.assertTrue(dry["only_lora_B_changed"])
        self.assertTrue(dry["participating_parameter_names"])
        self.assertTrue(dry["gradient_receipts"])
        self.assertTrue(dry["changed_parameter_names"])
        self.assertTrue(
            all(name.endswith(".caper_lora_B.weight") for name in dry["changed_parameter_names"])
        )
        self.assertEqual(dry["sp4_shard_route"]["parallel_state_receipt"]["world_size"], 4)
        self.assertEqual(result.receipt["optimizer_contract"]["weight_decay"], 0.0)
        auth.verify_authorization_receipt(result.receipt)
        tampered = deepcopy(result.receipt)
        tampered["optimizer_created"] = False
        with self.assertRaisesRegex(
            auth.CAPERTrainAuthorizationError, "receipt seal"
        ):
            auth.verify_authorization_receipt(tampered)

    def test_external_optimizer_factory_is_rejected_without_call(self) -> None:
        stage = self.materialize()
        decision, manifest = self._admitted_paq()
        calls, factory = self._factory_spy()
        with self.assertRaisesRegex(auth.CAPERTrainAuthorizationError, "external optimizer"):
            auth.authorize_caper_training_and_create_optimizer(
                paq_decision=decision,
                paq_manifest=manifest,
                stage1_materialization=stage,
                caper_handle=self.handle,
                caper_route=self._native_sp4_route(),
                optimizer_factory=factory,
                dry_run_forward=self._dry_run_forward,
            )
        self.assertEqual(calls, [])

    def test_illegal_a_gradient_blocks_step_and_leaves_all_ab_initial(self) -> None:
        stage = self.materialize()
        decision, manifest = self._admitted_paq()
        before = self.handle.trainable_parameter_values_sha256()

        def attacked_forward():
            legitimate = self._dry_run_forward()
            first_a = self.handle.q_wrappers[0][1].caper_lora_A.weight
            return legitimate + first_a.sum()

        with self.assertRaisesRegex(auth.CAPERTrainAuthorizationError, "illegal LoRA A gradient"):
            auth.authorize_caper_training_and_create_optimizer(
                paq_decision=decision,
                paq_manifest=manifest,
                stage1_materialization=stage,
                caper_handle=self.handle,
                caper_route=self._native_sp4_route(),
                dry_run_forward=attacked_forward,
            )
        self.assertEqual(self.handle.trainable_parameter_values_sha256(), before)
        self.assertTrue(
            all(parameter.grad is None for _, parameter in self.handle.trainable_named_parameters())
        )

    def test_preset_gradient_blocks_optimizer_construction(self) -> None:
        stage = self.materialize()
        decision, manifest = self._admitted_paq()
        parameter = self.handle.q_wrappers[0][1].caper_lora_B.weight
        parameter.grad = torch.zeros_like(parameter)
        with mock.patch.object(auth, "_fixed_adamw", wraps=auth._fixed_adamw) as constructor:
            with self.assertRaisesRegex(auth.CAPERTrainAuthorizationError, "clean-gradient"):
                auth.authorize_caper_training_and_create_optimizer(
                    paq_decision=decision,
                    paq_manifest=manifest,
                    stage1_materialization=stage,
                    caper_handle=self.handle,
                    caper_route=self._native_sp4_route(),
                    dry_run_forward=self._dry_run_forward,
                )
            constructor.assert_not_called()
        parameter.grad = None


class CAPERTrainStepCoreTests(unittest.TestCase):
    """Core optimizer/step tests intentionally independent of Stage-1 fixtures."""

    def setUp(self) -> None:
        torch.manual_seed(8123)
        self.renderer = _Renderer(hidden=8)
        self.renderer.requires_grad_(False)
        self.handle = lora.install_caper_capacity_probe(
            self.renderer, expected_hidden_size=8
        )
        state = lora.CAPERParallelState(
            world_size=4,
            world_rank=0,
            sequence_parallel_group_ranks=(0, 1, 2, 3),
            sequence_parallel_rank=0,
            authority_id="torch.distributed+bernini.parallel.get_parallel_state",
            test_only=False,
        )
        self.live_patch = mock.patch.object(
            lora, "snapshot_live_bernini_parallel_state", return_value=state
        )
        self.live_patch.start()
        self.route_spec = lora.CAPERRoute.from_runtime_sigma(
            global_target_selector=lora.preference_pack_target_selector(
                source_tokens=2, target_tokens=3
            ),
            pack_segments=lora.preference_pack_segments(
                source_tokens=2, target_tokens=3
            ),
            parallel_state_authority=_Authority(state),
            sigma_schedule_index=0,
            sigma=lora.sigma_strata.PINNED_POSITIVE_SIGMAS[0],
        )

    def tearDown(self) -> None:
        self.live_patch.stop()
        if not self.handle.restored:
            self.handle.restore()

    def _forward(self):
        hidden = torch.linspace(-0.75, 0.85, 24).reshape(1, 3, 8)
        return torch.stack(
            [
                wrapper(hidden).square().mean()
                for _, wrapper in self.handle.q_wrappers + self.handle.o_wrappers
            ]
        ).sum()

    def test_fixed_adamw_real_step_changes_only_b_and_receipts_all_gradients(self) -> None:
        route = auth._live_route_from_spec(self.route_spec)
        rows = auth._parameter_inventory(self.handle)
        optimizer = auth._fixed_adamw(tuple(parameter for _, parameter in rows))
        self.assertIs(type(optimizer), torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]["lr"], auth.ADAMW_LR)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)
        base = self.handle.freeze_checksum_certificate()[
            "frozen_transformer_current_sha256"
        ]
        receipt = auth._run_one_step_dry_run(
            handle=self.handle,
            route=route,
            optimizer=optimizer,
            forward=self._forward,
            base_before=base,
        )
        self.assertEqual(len(receipt["participating_parameter_names"]), 120)
        self.assertEqual(len(receipt["gradient_receipts"]), 120)
        self.assertEqual(len(receipt["changed_parameter_names"]), 60)
        self.assertTrue(
            all(
                name.endswith(".caper_lora_B.weight")
                for name in receipt["changed_parameter_names"]
            )
        )
        self.assertEqual(
            receipt["sp4_shard_route"]["parallel_state_receipt"]
            ["sequence_parallel_group_ranks"],
            [0, 1, 2, 3],
        )

    def test_optimizer_state_hash_accepts_scalar_adamw_step_tensor(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1.0e-4)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        self.assertEqual(optimizer.state[parameter]["step"].ndim, 0)
        first = auth.optimizer_state_sha256(optimizer)
        second = auth.optimizer_state_sha256(optimizer)
        self.assertRegex(first, r"^[0-9a-f]{64}$")
        self.assertEqual(first, second)

    def test_illegal_a_gradient_is_rejected_before_step(self) -> None:
        route = auth._live_route_from_spec(self.route_spec)
        rows = auth._parameter_inventory(self.handle)
        optimizer = auth._fixed_adamw(tuple(parameter for _, parameter in rows))
        before = self.handle.trainable_parameter_values_sha256()

        def attacked():
            return self._forward() + rows[0][1].sum()

        with self.assertRaisesRegex(auth.CAPERTrainAuthorizationError, "illegal LoRA A"):
            auth._run_one_step_dry_run(
                handle=self.handle,
                route=route,
                optimizer=optimizer,
                forward=attacked,
                base_before=self.handle.freeze_checksum_certificate()[
                    "frozen_transformer_current_sha256"
                ],
            )
        self.assertEqual(self.handle.trainable_parameter_values_sha256(), before)
        self.assertEqual(optimizer.state, {})


if __name__ == "__main__":
    unittest.main()
