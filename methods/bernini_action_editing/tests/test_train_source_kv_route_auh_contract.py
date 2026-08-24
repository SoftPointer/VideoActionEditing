from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_kv_replay as replay
import source_kv_route_batches as route_batches
import source_kv_route_scope as route_scope
import train_lora as legacy
import train_source_kv_route_auh as trainer


try:
    import torch
except ImportError:  # pragma: no cover - local lightweight environment
    torch = None


class _Object:
    def __init__(self, **values):
        self.__dict__.update(values)


def _valid_args(*extra: str):
    return trainer.build_parser().parse_args(
        [
            "--bernini-root",
            "/tmp/bernini",
            "--veomni-root",
            "/tmp/veomni",
            "--checkpoint",
            "/tmp/checkpoint",
            "--output",
            "/tmp/output",
            "--method-source-revision",
            "a" * 40,
            "--method-source-archive-sha256",
            "b" * 64,
            *extra,
        ]
    )


def _empty_counter_receipt(selected_blocks):
    return {
        "identity": None,
        "selected_blocks": list(selected_blocks),
        "captured_blocks": [],
        "complete": False,
        "capture_calls": 0,
        "replay_lookups": 0,
        "replay_branch_counts": {},
        "replay_phase_counts": {
            replay.EAGER_EXECUTION: 0,
            replay.CHECKPOINT_FORWARD: 0,
            replay.CHECKPOINT_RECOMPUTE: 0,
        },
        "checkpoint_context_counts": {
            replay.CHECKPOINT_FORWARD: 0,
            replay.CHECKPOINT_RECOMPUTE: 0,
        },
        "retired_identity_count": 0,
        "entries": [],
    }


class StaticTrainerContractTests(unittest.TestCase):
    def test_main_cli_is_exact40_all30_and_ablation_is_explicit(self) -> None:
        args = _valid_args()
        trainer.validate_cli(args)
        self.assertEqual(args.block_selection, "all")
        self.assertEqual(args.max_steps, 40)
        self.assertEqual(args.num_frames, 81)

        with self.assertRaises(trainer.SourceKVRouteAUHError):
            trainer.validate_cli(_valid_args("--block-selection", "mid"))
        trainer.validate_cli(
            _valid_args(
                "--block-selection", "mid", "--experimental-block-ablation"
            )
        )
        with self.assertRaises(trainer.SourceKVRouteAUHError):
            trainer.validate_cli(_valid_args("--experimental-block-ablation"))

    def test_checkpoint_configuration_binds_nonreentrant_context_fn(self) -> None:
        class FakeModel:
            def gradient_checkpointing_enable(self, **kwargs):
                self.kwargs = kwargs

        model = FakeModel()
        returned = trainer.configure_source_kv_gradient_checkpointing(model)
        self.assertIs(returned["context_fn"], replay.source_kv_replay_checkpoint_context_fn)
        self.assertFalse(returned["use_reentrant"])
        self.assertEqual(model.kwargs, {"gradient_checkpointing_kwargs": returned})

    def test_runner_owns_backward_before_clear_without_v8_monkeypatch(self) -> None:
        source = (METHOD_ROOT / "train_source_kv_route_auh.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_install_strategy", source)
        self.assertNotIn("_run_five_forward_cell", source)
        backward = source.index("cell.loss_result.total.backward()")
        clear = source.index("cache_bank.clear()", backward)
        optimizer = source.index("optimizer.step()", clear)
        self.assertLess(backward, clear)
        self.assertLess(clear, optimizer)
        self.assertIn("fields, sigma=sigma, config=loss_config", source)

    def test_launcher_pins_formal_inputs_all30_and_full_receipt_gate(self) -> None:
        launcher = (
            METHOD_ROOT / "scripts/auh_train_source_kv_route_v9_pilot.sbatch"
        ).read_text(encoding="utf-8")
        for literal in (
            trainer.PINNED_DATASET_SHARDS,
            trainer.PINNED_DATASET_SUMMARY,
            trainer.PINNED_DATASET_INDEX_SHA256,
            trainer.PINNED_ROUTING_JSONL,
            trainer.PINNED_ROUTING_SHA256,
            "--block-selection all",
            ".cache_after_backward_audit.replay_lookups_delta == 210",
            ".cache_after_backward_audit.replay_phase_count_delta.eager == 90",
            ".cache_after_backward_audit.replay_phase_count_delta.checkpoint_forward == 60",
            ".cache_after_backward_audit.replay_phase_count_delta.checkpoint_recompute == 60",
            ".gradient_audit.positive_global_l2_norm == true",
            ".query_state_policy.query_state_train_test_matched == false",
            ".artifact_validation.fresh_optimizer_load_state_dict_verified == true",
            ".artifact_validation.runtime_adapter_loader_verified == false",
            ".dataset.input_integrity.access_count == 40",
            "canonical receipt digest mismatch",
            "optimizer SHA differs from receipt",
        ):
            self.assertIn(literal, launcher)
        self.assertNotIn("BERNINI_ACTION_PARQUET_DIR", launcher)
        self.assertNotIn("BERNINI_ACTION_DATASET_SUMMARY", launcher)

    def test_query_endpoint_is_isolated_beta0_and_never_silent_beta1(self) -> None:
        source_payload = ({}, {}, {}, {"bridge_fraction": 0.0})
        target_payload = ({}, {}, {}, {"bridge_fraction": 1.0})
        endpoints = {"source": source_payload, "target": target_payload}
        self.assertIs(
            trainer.select_training_query_endpoint(endpoints), source_payload
        )
        with self.assertRaises(trainer.SourceKVRouteAUHError):
            trainer.select_training_query_endpoint(
                endpoints, query_state_policy="target_beta1"
            )
        broken = {"source": target_payload, "target": target_payload}
        with self.assertRaises(trainer.SourceKVRouteAUHError):
            trainer.select_training_query_endpoint(broken)

    def test_immutable_contract_pins_data_scope_counts_and_exposure_gap(self) -> None:
        args = _valid_args()
        dataset = _Object(signature="dataset-signature")
        summary = {"summary_digest": "summary-digest"}
        router = _Object(digest="router-digest")
        value = trainer._immutable_contract(
            args=args,
            dataset=dataset,
            dataset_summary=summary,
            router=router,
            eligible_routes=[],
            scope_manifest={"exact92": True},
            checkpoint=Path("/tmp/checkpoint"),
            loss_config=trainer.loss_config_from_args(args),
        )["value"]
        self.assertEqual(value["dataset"]["shards_path"], trainer.PINNED_DATASET_SHARDS)
        self.assertEqual(
            value["dataset"]["summary_file_sha256"],
            trainer.PINNED_DATASET_SUMMARY_FILE_SHA256,
        )
        self.assertEqual(
            value["dataset"]["index_sha256"], trainer.PINNED_DATASET_INDEX_SHA256
        )
        self.assertEqual(value["dataset"]["rows"], 644)
        self.assertEqual(value["dataset"]["eligible_rows"], 359)
        self.assertEqual(value["carrier"]["selected_block_count"], 30)
        self.assertEqual(value["forwards_per_candidate"], 6)
        self.assertEqual(
            value["expected_all30_counts"],
            {
                "capture_calls": 30,
                "forward_replay_lookups": 150,
                "backward_recompute_replay_lookups": 60,
                "total_replay_lookups": 210,
            },
        )
        self.assertEqual(value["optimizer_contract"]["trainable_tensor_count"], 184)
        self.assertEqual(
            value["optimizer_contract"]["trainable_parameter_count"], 2_260_992
        )
        policy = value["query_state_policy"]
        self.assertEqual(policy["name"], trainer.QUERY_STATE_POLICY)
        self.assertEqual(policy["training_tail_formula"], trainer.QUERY_STATE_TRAIN_FORMULA)
        self.assertFalse(policy["query_state_train_test_matched"])
        self.assertFalse(policy["paired_target_tail_used"])
        self.assertEqual(policy["future_ablation"], trainer.QUERY_STATE_FOLLOWUP_ARM)
        self.assertTrue(value["production_claim_forbidden"])
        self.assertGreaterEqual(len(value["limitations"]), 2)

    def test_mid_ablation_separates_selected_replays_from_all30_contexts(self) -> None:
        selected = replay.resolve_block_indices(30, "mid")
        count = len(selected)
        before = _empty_counter_receipt(selected)
        after = _empty_counter_receipt(selected)
        after.update(
            {
                "identity": {
                    "generation": 0,
                    "step_index": 0,
                    "timestep_token": "t=999,sigma=0.9",
                    "rank": 0,
                    "ulysses_size": 4,
                },
                "captured_blocks": list(selected),
                "complete": True,
                "capture_calls": count,
                "replay_lookups": 7 * count,
                "replay_branch_counts": {
                    "frozen_negative": count,
                    "frozen_noop": count,
                    "frozen_action": count,
                    "adapted_noop": 2 * count,
                    "adapted_action": 2 * count,
                },
                "replay_phase_counts": {
                    replay.EAGER_EXECUTION: 3 * count,
                    replay.CHECKPOINT_FORWARD: 2 * count,
                    replay.CHECKPOINT_RECOMPUTE: 2 * count,
                },
                "checkpoint_context_counts": {
                    replay.CHECKPOINT_FORWARD: 60,
                    replay.CHECKPOINT_RECOMPUTE: 60,
                },
                "entries": [
                    {"block_index": block, "detached": True}
                    for block in selected
                ],
            }
        )
        audit = trainer.audit_cache_after_backward(
            before=before, after=after, selected_blocks=selected
        )
        self.assertEqual(audit["replay_lookups_delta"], 7 * count)
        self.assertEqual(
            audit["checkpoint_context_count_delta"],
            {
                replay.CHECKPOINT_FORWARD: 60,
                replay.CHECKPOINT_RECOMPUTE: 60,
            },
        )
        self.assertEqual(audit["checkpointed_transformer_block_count"], 30)

    def test_exact40_gate_requires_positive_finite_184_gradient_contract(self) -> None:
        blocks = list(range(30))
        records = []
        for index in range(40):
            records.append(
                {
                    "sigma_schedule_index": index,
                    "forward_order": list(trainer.FORWARD_ORDER),
                    "forwards_per_candidate": 6,
                    "graph_forwards_per_candidate": 2,
                    "paired_target_model_forward_access": False,
                    "gradient_audit": {
                        "trainable_tensor_count": 184,
                        "all_gradients_finite": True,
                        "positive_global_l2_norm": True,
                        "global_l2_norm": 0.25,
                    },
                    "optimizer_audit": {
                        "state_parameter_count": 184,
                        "state_step_values": [index + 1],
                        "no_moment_reset": True,
                    },
                    "target_energy_retention": 1.0,
                    "target_clipped_fraction": 0.0,
                    "cache_after_backward_audit": {
                        "selected_blocks": blocks,
                        "capture_calls_delta": 30,
                        "replay_lookups_delta": 210,
                        "backward_recompute_observed": True,
                    },
                    "cache_after_clear_audit": {
                        "cleared_after_backward": True,
                        "identity_after_clear": None,
                    },
                    "row_index": index,
                    "input_shard_integrity": {
                        "access_ordinal": index,
                        "row_index": index,
                        "expected_sha256": "d" * 64,
                        "before_read_sha256": "d" * 64,
                        "after_read_sha256": "d" * 64,
                        "cache_invalidated_before_read": True,
                        "hash_closed_read": True,
                    },
                    "fresh_parity_checked": index == 0,
                }
            )
        result = trainer.validate_exact40_step_audit(records, block_selection="all")
        self.assertTrue(result["validated"])
        self.assertTrue(result["all30_main"])

        records[17]["gradient_audit"]["global_l2_norm"] = 0.0
        records[17]["gradient_audit"]["positive_global_l2_norm"] = False
        with self.assertRaises(trainer.SourceKVRouteAUHError):
            trainer.validate_exact40_step_audit(records, block_selection="all")

    def test_accessed_shard_tracker_hash_closes_each_uncached_read(self) -> None:
        class FakeDataset:
            def __init__(self, root, shard):
                self.root = root
                self.files = (shard,)
                self._length = 1
                self._groups = [(0, 1, shard, 0)]
                self._ends = [1]
                self._cached_key = (shard, 0)
                self._cached_rows = [{"iid": "stale"}]

            def __len__(self):
                return self._length

            def __getitem__(self, index):
                if self._cached_key is None:
                    self._cached_rows = [{"iid": self.files[0].read_text()}]
                    self._cached_key = (self.files[0], 0)
                return self._cached_rows[index]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "row.parquet"
            shard.write_text("fresh", encoding="utf-8")
            digest = trainer._sha256_file(shard)
            dataset = FakeDataset(root, shard)
            tracker = trainer.AccessedShardIntegrityTracker(
                dataset=dataset,
                expected_shard_sha256={shard: digest},
                index_path=root / "index.jsonl",
            )
            row, audit = tracker.read(0, access_ordinal=0)
            self.assertEqual(row["iid"], "fresh")
            self.assertTrue(audit["cache_invalidated_before_read"])
            self.assertEqual(audit["before_read_sha256"], digest)
            self.assertEqual(audit["after_read_sha256"], digest)

            shard.write_text("mutated", encoding="utf-8")
            tracker_2 = trainer.AccessedShardIntegrityTracker(
                dataset=dataset,
                expected_shard_sha256={shard: digest},
                index_path=root / "index.jsonl",
            )
            with self.assertRaises(trainer.SourceKVRouteAUHError):
                tracker_2.read(0, access_ordinal=0)


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class TensorTrainerIntegrationTests(unittest.TestCase):
    class Controller:
        @contextmanager
        def disable_adapter(self):
            yield

    def _candidate(self):
        tokens = 21
        total = 2 * tokens
        selector = torch.zeros((1, total), dtype=torch.bool)
        selector[:, tokens:] = True
        state = {
            "input_vae_latents": torch.zeros((total, 1), dtype=torch.bfloat16),
            "input_vae_rope": torch.zeros((total, 1, 2), dtype=torch.float32),
            "vae_latents_mask": selector,
            "vae_seqlen": torch.tensor([[total]], dtype=torch.long),
            "timesteps": torch.tensor([[500]], dtype=torch.long),
        }

        def branch(token_id):
            return {
                "input_ids": torch.tensor([[token_id, token_id + 1]], dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
                "t5_input_lens": torch.tensor([[2]], dtype=torch.long),
                **{name: value.clone() for name, value in state.items()},
            }

        negative = branch(10)
        noop = branch(20)
        action = branch(30)
        carrier = route_batches.build_source_only_carrier_batch(
            action_pair_batch=action,
            noop_pair_batch=noop,
            noop_instruction=route_batches.EXACT_NOOP_INSTRUCTION,
        )
        phase = torch.arange(tokens, dtype=torch.float32).reshape(1, tokens, 1)
        source_clean = torch.zeros((1, tokens, 4), dtype=torch.float32)
        target_clean = (0.035 * phase).expand(1, tokens, 4).contiguous()
        return trainer.MovedSourceKVRouteCandidate(
            editor_negative=negative,
            editor_noop=noop,
            editor_action=action,
            carrier=carrier,
            auxiliary={
                "shared_noisy": torch.zeros((1, tokens, 4), dtype=torch.float32),
                "sigma": torch.tensor(0.5, dtype=torch.float32),
                "source_clean": source_clean,
                "target_clean": target_clean,
                "bridge_fraction": 0.0,
            },
            spatial_hw=(1, 1),
            instruction_sha256="c" * 64,
        )

    def _parameters(self):
        named = []
        for name, shape in route_scope.expected_adapter_shapes().items():
            named.append((name, torch.nn.Parameter(torch.zeros(shape))))
        self.assertEqual(len(named), 184)
        self.assertEqual(sum(parameter.numel() for _, parameter in named), 2_260_992)
        return named

    def test_one_owned_step_keeps_cache_through_backward_and_builds_184_states(self) -> None:
        from torch.utils.checkpoint import checkpoint

        candidate = self._candidate()
        named = self._parameters()
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in named], lr=1.0e-5, weight_decay=0.0
        )
        bank = replay.SourceKVCacheBank(tuple(range(30)))
        events = []
        observed_model_fields = []

        def parameter_signal():
            return torch.stack([parameter.mean() for _, parameter in named]).sum()

        def full_velocity(_renderer, batch):
            invocation = replay.current_source_kv_invocation()
            events.append(invocation.branch_tag)
            observed_model_fields.append(set(batch))
            token_count = int(batch["input_vae_latents"].shape[0])
            if invocation.mode == replay.CAPTURE_MODE:
                key = torch.zeros(
                    (1, token_count, 1, 2), dtype=torch.bfloat16
                )
                for block_index in bank.selected_block_indices:
                    bank.capture(
                        invocation=invocation,
                        block_index=block_index,
                        key=key,
                        value=key,
                    )
                return torch.zeros(
                    (1, token_count, 4), dtype=torch.bfloat16
                )

            source_tokens = token_count // 2
            current_kv = torch.zeros(
                (1, token_count, 1, 2), dtype=torch.bfloat16
            )
            branch = invocation.branch_tag.replace("adapted_", "").replace(
                "frozen_", ""
            )
            phase = torch.arange(token_count, dtype=torch.float32).reshape(
                1, token_count, 1
            )
            if branch == "negative":
                base = torch.full((1, token_count, 4), -0.015)
                coefficient = 0.0
            elif branch == "noop":
                base = torch.full((1, token_count, 4), 0.005)
                coefficient = 0.35
            else:
                base = (0.004 * phase).expand(1, token_count, 4).clone()
                coefficient = 0.85
            signal = (
                parameter_signal()
                if invocation.branch_tag.startswith("adapted_")
                else torch.zeros((), dtype=torch.float32)
            )
            value = base

            for block_index in range(30):
                def block(current, adapter_signal, index=block_index):
                    active = replay.current_source_kv_invocation()
                    if index in bank.selected_block_indices:
                        bank._lookup(
                            invocation=active,
                            block_index=index,
                            current_key=current_kv,
                            current_value=current_kv,
                            source_tokens=source_tokens,
                        )
                    return current + 0.001 * torch.sin(
                        current + coefficient * adapter_signal + index * 0.01
                    )

                if invocation.branch_tag.startswith("adapted_"):
                    value = checkpoint(
                        block,
                        value,
                        signal,
                        use_reentrant=False,
                        context_fn=replay.source_kv_replay_checkpoint_context_fn,
                    )
                else:
                    value = block(value, signal)
            return value.to(dtype=torch.bfloat16)

        def local_gradient_norm(parameters):
            squared = sum(
                float(parameter.grad.detach().double().square().sum())
                for _, parameter in parameters
            )
            return math.sqrt(squared)

        result = trainer.execute_source_kv_route_optimizer_step(
            renderer=object(),
            adapter_controller=self.Controller(),
            candidate=candidate,
            cache_bank=bank,
            named_trainable=named,
            optimizer=optimizer,
            generation=0,
            step_index=0,
            timestep_token="t=999,sigma=0.5",
            rank=0,
            ulysses_size=1,
            loss_config=trainer.loss_config_from_args(_valid_args()),
            require_fresh_parity=True,
            full_velocity_fn=full_velocity,
            gradient_reduce_fn=local_gradient_norm,
        )
        self.assertEqual(
            events,
            [
                replay.CAPTURE_BRANCH_TAG,
                "frozen_negative",
                "frozen_noop",
                "frozen_action",
                "adapted_noop",
                "adapted_action",
            ],
        )
        self.assertTrue(
            all(fields == set(route_batches.CARRIER_MODEL_FIELDS) for fields in observed_model_fields)
        )
        record = result.record
        cache = record["cache_after_backward_audit"]
        self.assertEqual(cache["capture_calls_delta"], 30)
        self.assertEqual(cache["replay_lookups_delta"], 210)
        self.assertEqual(
            cache["replay_phase_count_delta"],
            {
                replay.EAGER_EXECUTION: 90,
                replay.CHECKPOINT_FORWARD: 60,
                replay.CHECKPOINT_RECOMPUTE: 60,
            },
        )
        self.assertEqual(
            cache["checkpoint_context_count_delta"],
            {
                replay.CHECKPOINT_FORWARD: 60,
                replay.CHECKPOINT_RECOMPUTE: 60,
            },
        )
        self.assertTrue(record["cache_after_clear_audit"]["cleared_after_backward"])
        self.assertIsNone(bank.identity)
        self.assertEqual(record["gradient_audit"]["trainable_tensor_count"], 184)
        self.assertTrue(record["gradient_audit"]["all_gradients_present"])
        self.assertTrue(record["gradient_audit"]["all_gradients_finite"])
        self.assertTrue(record["gradient_audit"]["positive_global_l2_norm"])
        self.assertEqual(record["optimizer_audit"]["state_parameter_count"], 184)
        self.assertEqual(record["optimizer_audit"]["state_step_values"], [1])
        self.assertTrue(result.cell.fresh_noop_exact)
        self.assertTrue(result.cell.fresh_action_exact)
        self.assertEqual(record["target_clipped_fraction"], 0.0)
        self.assertEqual(record["target_energy_retention"], 1.0)

        payload = {
            "schema_version": trainer.OPTIMIZER_SCHEMA,
            "global_step": 1,
            "optimizer": optimizer.state_dict(),
            "parameter_names": [name for name, _ in named],
            "optimizer_audit": record["optimizer_audit"],
            "resume_integrated": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            optimizer_path = Path(temporary) / "optimizer.pt"
            torch.save(payload, optimizer_path)
            roundtrip = trainer.validate_optimizer_artifact_roundtrip(
                optimizer_path=optimizer_path,
                optimizer_payload=payload,
                named_trainable=named,
                expected_step=1,
            )
        self.assertTrue(roundtrip["torch_deserialize_verified"])
        self.assertTrue(roundtrip["fresh_optimizer_load_state_dict_verified"])
        self.assertTrue(roundtrip["optimizer_state_logical_equality_verified"])
        self.assertEqual(roundtrip["state_parameter_count"], 184)
        self.assertEqual(roundtrip["state_step_values"], [1])


if __name__ == "__main__":
    unittest.main()
