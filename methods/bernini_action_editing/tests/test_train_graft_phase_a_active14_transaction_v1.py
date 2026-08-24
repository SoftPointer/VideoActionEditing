#!/usr/bin/env python3

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch

    import graft_phase_a_native_training_closure_v1 as native_v1
    import graft_phase_a_native_training_closure_v2 as native_v2
    import train_graft_phase_a_a_lite_short_v1 as short_trainer
    import train_graft_phase_a_active14_transaction_v1 as active14

    TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


class _MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value):
        self.running_average = (
            update_value
            + self.momentum * self.running_average
        )


def _normalized_guidance(
    pred_cond,
    pred_uncond,
    guidance_scale,
    momentum_buffer=None,
    eta=1.0,
    norm_threshold=0.0,
):
    import torch.nn.functional as functional

    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -4], keepdim=True)
        diff = diff * torch.minimum(ones, norm_threshold / diff_norm)
    projected, base = diff.double(), pred_cond.double()
    base = functional.normalize(base, dim=[-1, -2, -4])
    parallel = (projected * base).sum(
        dim=[-1, -2, -4], keepdim=True
    ) * base
    orthogonal = projected - parallel
    normalized = orthogonal.to(diff.dtype) + eta * parallel.to(diff.dtype)
    return pred_uncond + guidance_scale * normalized


if TORCH_AVAILABLE:

    class _FakeAtlas(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Parameter(torch.tensor(0.19))


    class _FakeTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.query = torch.nn.Parameter(torch.tensor(0.23))
            self.key = torch.nn.Parameter(torch.tensor(-0.31))
            self.value = torch.nn.Parameter(torch.tensor(0.41))
            # Field14 follows the short bootstrap, so output is already nonzero.
            self.output = torch.nn.Parameter(torch.tensor([0.01, -0.02]))
            self.frozen_base = torch.nn.Parameter(
                torch.tensor(1.25), requires_grad=False
            )
            self.dtype = torch.bfloat16
            self.gradient_checkpointing = False

        def patch_vae_latent(self, hidden_states, source_id=None):
            batch, channels, phases, height, width = hidden_states.shape
            patches = (
                hidden_states.reshape(
                    batch, channels, phases, height // 2, 2, width // 2, 2
                )
                .permute(0, 2, 3, 5, 4, 6, 1)
                .reshape(batch, phases * (height // 2) * (width // 2), 64)
            )
            seed = patches.mean(dim=-1, keepdim=True)
            tokens = seed.expand(batch, seed.shape[1], 1536).contiguous()
            rotary = torch.full(
                (batch, 1, seed.shape[1], 8),
                float(source_id),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            return tokens, rotary


    class _FakeDiffusion(torch.nn.Module):
        def __init__(self, transformer, atlas):
            super().__init__()
            self.transformer = transformer
            self.transformer_2 = None
            self.atlas = atlas

        def shared_step(
            self,
            model_id,
            noisy_latents,
            timesteps,
            cond_embeds,
            rotary_embs,
            batch_vae_seqlen=None,
            batch_text_seqlen=None,
            **kwargs,
        ):
            del model_id, timesteps, rotary_embs, batch_vae_seqlen
            del batch_text_seqlen, kwargs
            base = noisy_latents[..., :64].float()
            text = cond_embeds.float().mean().reshape(1, 1, 1)
            feature = (
                self.transformer.query * (base + 0.17)
                + self.transformer.key * (text + 0.29)
                + self.transformer.value * (base * text + 0.37)
                + self.atlas.proj * (base.square() + text + 0.43)
            )
            raw = (
                base * (1.0 + 0.03125 * text)
                + 0.0078125 * self.transformer.frozen_base
                + self.transformer.output.mean() * feature
            )
            return raw.to(torch.bfloat16)


@unittest.skipUnless(TORCH_AVAILABLE, "torch runtime is required")
class Active14TransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.atlas = _FakeAtlas().eval()
        self.transformer = _FakeTransformer().eval()
        self.diffusion = _FakeDiffusion(self.transformer, self.atlas).eval()

        @contextmanager
        def route(*, request):
            rank = 2
            local_rows = (request.total_tokens + 3) // 4
            padded = local_rows * 4
            selector = torch.cat(
                (
                    torch.zeros(request.condition_tokens, dtype=torch.bool),
                    torch.ones(request.target_tokens, dtype=torch.bool),
                    torch.zeros(padded - request.total_tokens, dtype=torch.bool),
                )
            )[rank * local_rows : (rank + 1) * local_rows].contiguous()
            targets = int(torch.count_nonzero(selector).item())
            yield native_v1.build_native_forward_context_observation(
                request=request,
                sequence_parallel_rank=rank,
                sequence_parallel_size=4,
                local_target_selector=selector,
                route_gate=1.0,
                adapter_graph_bearing=(request.phase == "replay" and targets > 0),
            )

        self.names = (
            ("atlas_encoder.proj.weight", self.atlas.proj),
            (
                "blocks.8.attn1.to_out.0.identity_rebinder.query.weight",
                self.transformer.query,
            ),
            (
                "blocks.8.attn1.to_out.0.identity_rebinder.key.weight",
                self.transformer.key,
            ),
            (
                "blocks.8.attn1.to_out.0.identity_rebinder.value.weight",
                self.transformer.value,
            ),
            (
                "blocks.8.attn1.to_out.0.identity_rebinder.output.weight",
                self.transformer.output,
            ),
        )
        self.bindings = native_v2.authenticate_cpu_test_fakes(
            diffusion=self.diffusion,
            transformer=self.transformer,
            vendor_normalized_guidance=_normalized_guidance,
            momentum_buffer_factory=_MomentumBuffer,
            named_trainable_parameters=self.names,
            external_trainable_owner_modules={"atlas_encoder": self.atlas},
            test_name="cpu_fake:phase_a_active14_transaction",
            forward_context_factory=route,
        )
        self.backend = short_trainer.authenticate_cpu_test_collectives(rank=2)
        generator = torch.Generator(device="cpu").manual_seed(20260811)
        self.source = torch.randn(
            (1, 16, 21, 2, 4), generator=generator, dtype=torch.float32
        )
        self.noisy = torch.randn(
            (1, 16, 21, 2, 4), generator=generator, dtype=torch.float32
        )
        self.negative = torch.full((1, 2, 4), -1.0, dtype=torch.bfloat16)
        self.positive = torch.full((1, 2, 4), 2.0, dtype=torch.bfloat16)
        self.short_digest = "1" * 64
        self.row_iid = "fit-dog"
        self.row_sha = "2" * 64
        self.qualification = active14.seal_mapping(
            {
                "schema_version": (
                    "bernini-graft-phase-a-active14-upstream-qualification-v1"
                ),
                "field14_job_id": "123456",
                "field14_parent_receipt_digest": "3" * 64,
                "field14_runner_result_digest": "4" * 64,
                "field14_world8_result_digest": "5" * 64,
                "slurm_afterok_is_queue_gate_only": True,
                "weights_inherited_from_dependency_job": False,
                "optimizer_state_inherited_from_dependency_job": False,
                "checkpoint_consumed_from_dependency_job": False,
                "checkpoint_written": False,
                "publication_performed": False,
                **{name: False for name in active14.AUTHORITY_FIELDS},
            }
        )
        self.local_field = active14.seal_mapping(
            {
                "schema_version": active14.LOCAL_FIELD14_SCHEMA_VERSION,
                "status": "completed_in_memory_exact40_no_grad_no_checkpoint",
                "family": "dog",
                "wrong_owner_iid": self.row_iid,
                "short_result_digest": self.short_digest,
                "schedule_indices": list(range(40)),
                "inactive_indices": list(range(26)),
                "active_indices": list(range(26, 40)),
                "exact40_official_order": True,
                "ambient_torch_no_grad": True,
                "one_index_admitted_hashed_and_released_before_next": True,
                "cross_index_tensor_retention": False,
                "cross_index_compensation_used": False,
                "cross_index_selection_used": False,
                "rows": [
                    {
                        "schedule_index": index,
                        "all_field_tensor_objects_released": True,
                    }
                    for index in range(40)
                ],
                "checkpoint_written": False,
                "publication_performed": False,
                **{name: False for name in active14.AUTHORITY_FIELDS},
            }
        )

    def _cell(self, schedule_index):
        sigma = torch.tensor(
            native_v1.sigma_strata.PINNED_POSITIVE_SIGMAS[schedule_index],
            dtype=torch.float32,
        )
        timestep = torch.tensor(
            [native_v1.sigma_strata.PINNED_TIMESTEPS[schedule_index]],
            dtype=torch.int64,
        )
        return native_v2.PhaseANativeTrainingClosure(
            bindings=self.bindings,
            source_video=self.source,
            noisy_target=self.noisy,
            negative_condition=self.negative,
            positive_condition=self.positive,
            schedule_index=schedule_index,
            sigma=sigma,
            timestep=timestep,
        )

    def _source_only_native_result(self):
        sp_rank = 0
        backend = short_trainer.authenticate_cpu_test_collectives(rank=sp_rank)
        schedule_index = active14.ACTIVE_INDICES[0]
        plan_body = {
            "schema_version": active14.PLAN_SCHEMA_VERSION,
            "update_number": 1,
            "schedule_index": schedule_index,
            "training_regime": active14.TRAINING_REGIME,
            "dp_arm": backend.dp_arm,
            "row_iid": self.row_iid,
            "row_source_sha256": self.row_sha,
            "official_active14_order": True,
        }
        plan = active14.Active14CellPlan._mint(  # noqa: SLF001
            update_number=1,
            schedule_index=schedule_index,
            training_regime=active14.TRAINING_REGIME,
            dp_arm=backend.dp_arm,
            row_iid=self.row_iid,
            row_source_sha256=self.row_sha,
            plan_digest=active14.object_sha256(plan_body),
        )
        cell = self._cell(schedule_index)
        cell.measure()
        cell.derive_phase_a_flow_matching_vjp()
        target_owner_result = cell.replay_and_backward()
        receipt = dict(target_owner_result.receipt)
        receipt.pop("digest")
        receipt.update(
            {
                "local_sequence_parallel_rank": sp_rank,
                "local_target_rows": 0,
                "local_adapter_graph_bearing": False,
                "bootstrap_output_only_gate_verified": False,
                "post_bootstrap_five_category_local_gate_verified": False,
                "source_only_sp_all_five_categories_exact_zero_verified": True,
            }
        )
        source_only_result = replace(
            target_owner_result,
            receipt=active14.seal_mapping(receipt),
        )
        return self.bindings, backend, plan, source_only_result

    @staticmethod
    def _route(plan, _admission):
        return active14.seal_mapping(
            {
                "schema_version": "cpu-active14-route-v1",
                "update_number": plan.update_number,
                "schedule_index": plan.schedule_index,
                "row_iid": plan.row_iid,
                "exact_four_native_forwards": True,
                "fit_row_only": True,
                "checkpoint_written": False,
                "publication_performed": False,
                **{name: False for name in active14.AUTHORITY_FIELDS},
            }
        )

    @staticmethod
    def _prepare(_preliminary):
        return active14.seal_mapping(
            {
                "schema_version": "cpu-active14-preparation-v1",
                "preparation_completed": True,
                "kind": "standalone-noop",
                "published": False,
                "checkpoint_written": False,
                "publication_performed": False,
                **{name: False for name in active14.AUTHORITY_FIELDS},
            }
        )

    @staticmethod
    def _finalize(active14_commit, preparation):
        return active14.seal_mapping(
            {
                "schema_version": "cpu-active14-finalize-v1",
                "finalize_completed": True,
                "active14_commit_receipt_digest": active14_commit["digest"],
                "preparation_receipt_digest": preparation["digest"],
                "checkpoint_written": False,
                "publication_performed": False,
                **{name: False for name in active14.AUTHORITY_FIELDS},
            }
        )

    def _services(self, make_cell=None):
        return active14.authenticate_cpu_test_services(
            test_name="cpu_fake_active14",
            make_update_cell=(
                (lambda plan: self._cell(plan.schedule_index))
                if make_cell is None
                else make_cell
            ),
            after_update=self._route,
        )

    def test_exact14_success_has_no_checkpoint_or_authority(self) -> None:
        before = {name: value.detach().clone() for name, value in self.names}
        result = active14.execute_active14_transaction(
            upstream_qualification=self.qualification,
            local_field14_receipt=self.local_field,
            short_result_digest=self.short_digest,
            family="dog",
            row_iid=self.row_iid,
            row_source_sha256=self.row_sha,
            bindings=self.bindings,
            backend=self.backend,
            services=self._services(),
            prepare=self._prepare,
            finalize=self._finalize,
        )
        receipt = result.active14_commit_receipt
        self.assertEqual(receipt["optimizer_contract"]["steps"], 14)
        self.assertEqual(
            receipt["optimizer_contract"]["schedule_indices"], list(range(26, 40))
        )
        self.assertTrue(receipt["all_fourteen_updates_completed"])
        self.assertTrue(receipt["transaction_committed_in_memory"])
        self.assertFalse(receipt["checkpoint_written"])
        self.assertFalse(receipt["training_authority"])
        self.assertNotEqual(
            receipt["initial_trainable_digest"], receipt["final_trainable_digest"]
        )
        self.assertTrue(
            any(not torch.equal(before[name], value) for name, value in self.names)
        )
        self.assertTrue(torch.equal(self.transformer.frozen_base, torch.tensor(1.25)))

    def test_source_only_sp_rank_truth_table_and_hostile_mixed_claims(self) -> None:
        bindings, backend, plan, result = self._source_only_native_result()
        receipt = result.receipt
        self.assertEqual(receipt["local_target_rows"], 0)
        self.assertFalse(receipt["local_adapter_graph_bearing"])
        self.assertFalse(receipt["bootstrap_output_only_gate_verified"])
        self.assertFalse(
            receipt["post_bootstrap_five_category_local_gate_verified"]
        )
        self.assertTrue(
            receipt["source_only_sp_all_five_categories_exact_zero_verified"]
        )
        admission = active14._validate_native_result(  # noqa: SLF001
            result=result,
            bindings=bindings,
            backend=backend,
            plan=plan,
        )
        self.assertEqual(
            admission["schema_version"],
            "bernini-graft-phase-a-active14-native-admission-v2",
        )
        self.assertFalse(admission["local_target_owner"])
        self.assertEqual(admission["local_target_rows"], 0)
        self.assertEqual(
            admission["local_gradient_gate"],
            "source_only_sp_rank_all_five_categories_exact_zero",
        )

        hostile_changes = (
            {"post_bootstrap_five_category_local_gate_verified": True},
            {"local_target_rows": 1},
            {"local_adapter_graph_bearing": True},
        )
        for changes in hostile_changes:
            with self.subTest(changes=changes):
                attacked_receipt = dict(receipt)
                attacked_receipt.pop("digest")
                attacked_receipt.update(changes)
                attacked = replace(
                    result,
                    receipt=active14.seal_mapping(attacked_receipt),
                )
                with self.assertRaises(active14.Active14TransactionError):
                    active14._validate_native_result(  # noqa: SLF001
                        result=attacked,
                        bindings=bindings,
                        backend=backend,
                        plan=plan,
                    )

    def test_cell_failure_restores_entire_transaction_snapshot(self) -> None:
        before = {name: value.detach().clone() for name, value in self.names}

        def fail_at_30(plan):
            if plan.schedule_index == 30:
                raise RuntimeError("injected-cell-failure")
            return self._cell(plan.schedule_index)

        with self.assertRaises(active14.Active14TransactionError) as raised:
            active14.execute_active14_transaction(
                upstream_qualification=self.qualification,
                local_field14_receipt=self.local_field,
                short_result_digest=self.short_digest,
                family="dog",
                row_iid=self.row_iid,
                row_source_sha256=self.row_sha,
                bindings=self.bindings,
                backend=self.backend,
                services=self._services(fail_at_30),
                prepare=self._prepare,
                finalize=self._finalize,
            )
        failure = raised.exception.failure_receipt
        self.assertEqual(failure["status"], "failed_rolled_back_no_checkpoint")
        self.assertEqual(failure["completed_schedule_indices"], [26, 27, 28, 29])
        self.assertTrue(
            failure["trainable_parameters_restored_to_transaction_snapshot"]
        )
        self.assertFalse(failure["checkpoint_written"])
        for name, value in self.names:
            self.assertTrue(torch.equal(before[name], value), name)
            self.assertIsNone(value.grad)

    def test_preparation_failure_also_rolls_back(self) -> None:
        before = {name: value.detach().clone() for name, value in self.names}

        def fail_preparation(_preliminary):
            raise RuntimeError("downstream-failed")

        with self.assertRaises(active14.Active14TransactionError) as raised:
            active14.execute_active14_transaction(
                upstream_qualification=self.qualification,
                local_field14_receipt=self.local_field,
                short_result_digest=self.short_digest,
                family="dog",
                row_iid=self.row_iid,
                row_source_sha256=self.row_sha,
                bindings=self.bindings,
                backend=self.backend,
                services=self._services(),
                prepare=fail_preparation,
                finalize=self._finalize,
            )
        self.assertEqual(
            raised.exception.failure_receipt["failure_phase"],
            "downstream_prepare",
        )
        for name, value in self.names:
            self.assertTrue(torch.equal(before[name], value), name)

    def test_upstream_parent_validator_rejects_job_or_authority_drift(self) -> None:
        world8 = active14.seal_mapping(
            {
                "all_eight_exact40_completed": True,
                "both_sp4_arms_exact_field_hash_and_metric_consensus": True,
                "all_eight_trainable_bytes_unchanged_during_sweep": True,
                "all_eight_base_bytes_unchanged_entire_process": True,
                "authority": {name: False for name in active14.AUTHORITY_FIELDS},
            }
        )
        runner = active14.seal_mapping(
            {
                "status": "completed_in_memory_short_then_exact40_no_checkpoint",
                "world8": world8,
                "checkpoint_written": False,
                "publication_performed": False,
                "authority": {name: False for name in active14.AUTHORITY_FIELDS},
            }
        )
        core = {
            "schema_version": active14.UPSTREAM_SCHEMA_VERSION,
            "status": "completed_diagnostic_no_checkpoint",
            "complete": True,
            "pass": True,
            "job_id": "123456",
            "runner_result_digest": runner["digest"],
            "runtime": {},
            "validated": {
                "same_process_short_then_exact40": True,
                "exact40_official_order": True,
                "inactive_0_25_zero_gate_preinstall_parity_producer_validated": True,
                "active_26_39_finite_nonzero_gate_producer_validated": True,
                "all_eight_full_field14_receipts_deeply_validated": True,
                "both_sp4_per_index_field_hash_and_metric_consensus_recomputed": True,
                "rank_log_root_identity_retained_from_before_torchrun": True,
                "rank_stdout_opened_relative_to_retained_root_with_O_NOFOLLOW": True,
                "output_root_identity_retained_from_creation": True,
                "checkpoint_content_full_rehash_pre_and_post": True,
                "per_index_hash_then_release": True,
                "cross_index_compensation_or_selection": False,
                "semantic_metrics_authoritative": False,
            },
            "runner_result": runner,
            "checkpoint_written": False,
            "checkpoint_payload_returned": False,
            "publication_performed": False,
            "authority": {name: False for name in active14.AUTHORITY_FIELDS},
            "receipt_publication": {
                "create_only_O_EXCL": True,
                "provisional_mode": "0600",
                "final_success_mode": "0444",
                "mode_0444_is_terminal_success_transition": True,
                "canonical_ascii_json_newline": True,
                "output_root_identity_retained_from_creation": True,
            },
        }
        parent = {**core, "receipt_digest": active14.object_sha256(core)}
        admitted = active14.validate_upstream_field14_parent(
            parent, expected_job_id="123456"
        )
        self.assertFalse(admitted["weights_inherited_from_dependency_job"])
        with self.assertRaises(active14.Active14TransactionError):
            active14.validate_upstream_field14_parent(
                parent, expected_job_id="654321"
            )
        attacked = dict(parent)
        attacked["authority"] = dict(attacked["authority"])
        attacked["authority"]["training_authority"] = True
        attacked_core = dict(attacked)
        attacked_core.pop("receipt_digest")
        attacked["receipt_digest"] = active14.object_sha256(attacked_core)
        with self.assertRaises(active14.Active14TransactionError):
            active14.validate_upstream_field14_parent(
                attacked, expected_job_id="123456"
            )

    def test_source_contains_no_checkpoint_writer_or_free_schedule(self) -> None:
        source = Path(active14.__file__).read_text(encoding="utf-8")
        self.assertNotIn("torch.save", source)
        self.assertNotIn("--schedule", source)
        self.assertEqual(active14.ACTIVE_INDICES, tuple(range(26, 40)))
        self.assertEqual(
            hashlib.sha256(Path(active14.__file__).read_bytes()).hexdigest(),
            hashlib.sha256(source.encode("utf-8")).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
