#!/usr/bin/env python3

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import ast
import hashlib
import inspect
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch

    import graft_phase_a_native_training_closure_v1 as native_v1
    import graft_phase_a_native_training_closure_v2 as native_v2
    import identity_rebinder_v1 as rebinder
    import infer_lora as legacy
    import run_graft_phase_a_a_lite_short_gpu_v1 as runner
    import train_graft_phase_a_a_lite_short_v1 as trainer

    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None  # type: ignore[assignment]
    native_v1 = None  # type: ignore[assignment]
    native_v2 = None  # type: ignore[assignment]
    rebinder = None  # type: ignore[assignment]
    legacy = None  # type: ignore[assignment]
    runner = None  # type: ignore[assignment]
    trainer = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _walk_boolean_authority(test: unittest.TestCase, value) -> None:
    if isinstance(value, dict) or hasattr(value, "items"):
        for key, item in value.items():
            if (
                key in trainer.AUTHORITY_FIELDS
                or key.endswith("_authorized")
                or "authority" in key
            ) and isinstance(item, bool):
                test.assertFalse(item, key)
            _walk_boolean_authority(test, item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _walk_boolean_authority(test, item)


if _TORCH_AVAILABLE:
    class _MomentumBuffer:
        def __init__(self, momentum):
            self.momentum = momentum
            self.running_average = 0

        def update(self, update_value):
            self.running_average = (
                update_value + self.momentum * self.running_average
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
            self.output = torch.nn.Parameter(torch.zeros(2))
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


    class _ExplodingScheduler:
        def __getattribute__(self, name):
            if name.startswith("__"):
                return object.__getattribute__(self, name)
            raise AssertionError("the v2 cell must not access a scheduler")


    class _FakeDiffusion(torch.nn.Module):
        def __init__(self, transformer, atlas):
            super().__init__()
            self.transformer = transformer
            self.transformer_2 = None
            self.atlas = atlas
            self.call_count = 0
            self.scheduler = _ExplodingScheduler()

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
            self.call_count += 1
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


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class GraftPhaseAShortGPUContractTests(unittest.TestCase):
    def test_dependency_hashes_and_fixed_protocol_are_exact(self) -> None:
        runner.assert_pinned_dependencies()
        self.assertEqual(
            runner.PINNED_CONSUMER_SOURCE_SHA256,
            runner.file_sha256(runner.source_consumer.__file__),
        )
        self.assertEqual(
            runner.PINNED_NATIVE_V2_SOURCE_SHA256,
            runner.file_sha256(native_v2.__file__),
        )
        self.assertEqual(
            runner.PINNED_SHORT_TRAINER_SOURCE_SHA256,
            runner.file_sha256(trainer.__file__),
        )
        self.assertEqual(
            runner.PINNED_SHORT_TRAINER_EXECUTION_RUNTIME_SHA256,
            trainer.PINNED_TRAINER_EXECUTION_RUNTIME_SHA256,
        )
        with mock.patch.object(
            trainer,
            "PINNED_TRAINER_EXECUTION_RUNTIME_SHA256",
            "0" * 64,
        ):
            with self.assertRaises(runner.GraftPhaseAShortGPUError):
                runner.assert_pinned_dependencies()
        self.assertEqual((runner.WORLD_SIZE, runner.DP_SIZE, runner.SP_SIZE), (8, 2, 4))
        self.assertEqual(runner.UPDATE_INDICES, (29, 38))
        self.assertEqual(runner.CONFIRMATION_INDICES, (29, 38))
        self.assertEqual(runner.ADAPTER_OFF_PARITY_INDICES, (0, 25))
        topology_source = inspect.getsource(runner._initialize_world8_dp2sp4)  # noqa: SLF001
        self.assertNotIn("inference_distributed_contract", topology_source)
        self.assertIn('getattr(state, "dp_group", None)', topology_source)
        drop_source = inspect.getsource(runner._route_for_pack)  # noqa: SLF001
        self.assertIn('branch_name, enabled, atlas = "V", False, None', drop_source)

    def test_live_equal_gather_canonicalizes_unpickleable_sealed_mapping(self) -> None:
        value = runner.seal_mapping(
            {
                "schema_version": "mappingproxy-collective-regression-v1",
                "nested": (
                    "tuple-member",
                    runner.seal_mapping({"rows": [1, 2, 3]}),
                ),
            }
        )
        with self.assertRaises(TypeError):
            pickle.dumps(value, protocol=5)
        expected = runner.canonical_json_bytes(value)
        self.assertEqual(
            expected,
            runner.canonical_json_bytes(
                {
                    "digest": value["digest"],
                    "nested": [
                        "tuple-member",
                        dict(value["nested"][1]),
                    ],
                    "schema_version": "mappingproxy-collective-regression-v1",
                }
            ),
        )
        group = object()

        def equal_collective(rows, payload, *, group):
            self.assertIs(group, group_token)
            self.assertIs(type(payload), bytes)
            self.assertEqual(payload, expected)
            transported = pickle.loads(pickle.dumps(payload, protocol=5))
            rows[:] = [transported, transported]

        group_token = group
        with mock.patch.object(
            torch.distributed,
            "all_gather_object",
            side_effect=equal_collective,
        ):
            rows = runner._gather_equal(  # noqa: SLF001
                value,
                group=group,
                count=2,
                label="sealed mapping",
            )
        self.assertEqual(rows, [expected, expected])

        def unequal_collective(rows, payload, *, group):
            self.assertIs(group, group_token)
            rows[:] = [payload, b"{}"]

        with mock.patch.object(
            torch.distributed,
            "all_gather_object",
            side_effect=unequal_collective,
        ):
            with self.assertRaisesRegex(
                runner.GraftPhaseAShortGPUError,
                "sealed mapping differs across ranks",
            ):
                runner._gather_equal(  # noqa: SLF001
                    value,
                    group=group,
                    count=2,
                    label="sealed mapping",
                )

        def nonbytes_collective(rows, payload, *, group):
            self.assertIs(group, group_token)
            rows[:] = [payload, bytearray(payload)]

        with mock.patch.object(
            torch.distributed,
            "all_gather_object",
            side_effect=nonbytes_collective,
        ):
            with self.assertRaisesRegex(
                runner.GraftPhaseAShortGPUError,
                "sealed mapping differs across ranks",
            ):
                runner._gather_equal(  # noqa: SLF001
                    value,
                    group=group,
                    count=2,
                    label="sealed mapping",
                )

    def test_chunked_base_digest_is_v1_exact_and_runtime_guarded(self) -> None:
        tensors = (
            torch.arange(24, dtype=torch.float32).reshape(4, 6).t(),
            torch.arange(17, dtype=torch.int64),
            torch.tensor([True, False, True], dtype=torch.bool),
            torch.empty((0, 7), dtype=torch.bfloat16),
            torch.zeros(
                runner.MAX_CTYPES_DIGEST_CHUNK_BYTES + 17,
                dtype=torch.uint8,
            ),
        )
        for tensor in tensors:
            self.assertEqual(
                runner.short_chunked_tensor_identity(tensor),
                runner.native_runner_v1.tensor_identity(tensor),
            )
        rows = tuple(
            (f"base.{index}", torch.nn.Parameter(tensor, requires_grad=False))
            for index, tensor in enumerate(tensors[:-1])
        )
        self.assertEqual(
            runner.short_chunked_parameter_registry_digest(rows),
            runner.native_runner_v1.parameter_registry_digest(rows),
        )
        with mock.patch.object(runner.ctypes, "string_at", lambda *_args: b""):
            with self.assertRaises(runner.GraftPhaseAShortGPUError):
                runner.assert_pinned_dependencies()
        with mock.patch.object(
            runner,
            "short_chunked_tensor_identity",
            lambda _value: {},
        ):
            with self.assertRaises(runner.GraftPhaseAShortGPUError):
                runner.assert_pinned_dependencies()

    def test_cli_has_only_source_model_and_diagnostic_inputs(self) -> None:
        parser = runner.build_parser()
        destinations = {
            action.dest for action in parser._actions if action.dest != "help"
        }
        for fragment in (
            "proposal",
            "retelling",
            "caption",
            "t2v",
            "target_video",
            "generated_video",
            "output",
            "checkpoint_output",
        ):
            self.assertFalse(
                any(fragment in destination.lower() for destination in destinations),
                fragment,
            )
        args = parser.parse_args(self._valid_cli_argv())
        self.assertIs(runner.validate_cli(args), args)

    def test_source_consumer_then_training_route_is_the_only_production_path(self) -> None:
        source = inspect.getsource(runner.consume_authenticated_source_routing)
        tree = ast.parse(source)
        names = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        self.assertEqual(names.count("_PINNED_CONSUME"), 1)
        self.assertEqual(names.count("_PINNED_VALIDATE_FOR_TRAINING"), 1)
        self.assertLess(
            source.index("_PINNED_CONSUME("),
            source.index("_PINNED_VALIDATE_FOR_TRAINING("),
        )
        self.assertNotIn("open(", source)
        self.assertNotIn("source_bytes=", source)

    def test_dp_arms_route_dog_and_human_fit_confirmation_rows(self) -> None:
        routing = trainer.authenticate_cpu_test_routing(
            test_name="cpu_fake:short_gpu_source_routing"
        )
        dog = runner.route_local_family(routing, dp_arm=0)
        human = runner.route_local_family(routing, dp_arm=1)
        self.assertEqual(
            (dog.family, dog.fit_iid, dog.confirmation_iid),
            ("dog", "7b88a1ca1f804f41", "841b5e0080a1441d"),
        )
        self.assertEqual(
            (human.family, human.fit_iid, human.confirmation_iid),
            ("human", "a35b590961d24694", "a66e6818e4144928"),
        )
        self.assertIs(dog.fit_row, routing.update_rows[0])
        self.assertIs(dog.confirmation_row, routing.confirmation_rows[0])
        self.assertIs(human.fit_row, routing.update_rows[1])
        self.assertIs(human.confirmation_row, routing.confirmation_rows[1])

    def test_source_has_no_checkpoint_publication_or_phase_b_import(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("graft_action_first_source_guided_aggregation_v1", imported)
        self.assertNotIn("graft_source_conditioned_proposal_selector_v1", imported)
        forbidden_calls = {"save", "save_file", "atomic_torch_save", "save_pretrained"}
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in forbidden_calls
                for node in ast.walk(tree)
            )
        )

    def test_condition_seam_consumes_only_authenticated_a_lite_row_noop(self) -> None:
        routing = trainer.authenticate_cpu_test_routing(
            test_name="cpu_fake:short_gpu_condition_seam"
        )
        local = runner.route_local_family(routing, dp_arm=0)

        class RecordingTokenizer:
            def __init__(self):
                self.calls = []

            def __call__(self, text, **kwargs):
                self.calls.append((text, dict(kwargs)))
                marker = len(self.calls)
                length = 512 if kwargs.get("padding") == "max_length" else 3
                return SimpleNamespace(
                    input_ids=torch.full(
                        (1, length), marker, dtype=torch.int64
                    ),
                    attention_mask=torch.ones(
                        (1, length), dtype=torch.int64
                    ),
                )

        class RecordingRenderer:
            @staticmethod
            def encode_prompt(input_ids, _attention_mask):
                marker = float(input_ids[0, 0].item())
                return torch.full(
                    (1, 512, 4096), marker, dtype=torch.bfloat16
                )

        tokenizer = RecordingTokenizer()
        renderer = RecordingRenderer()
        old_v1_noop = (
            runner.native_runner_v1.route_batches.EXACT_NOOP_INSTRUCTION
        )
        self.assertNotEqual(
            old_v1_noop, runner.source_consumer.NOOP_INSTRUCTION
        )
        deliberately_different_old_v1 = (
            "OLD V1 NOOP MUST NEVER ENTER THE A-LITE CONDITION SEAM."
        )
        with mock.patch.object(
            runner.native_runner_v1.route_batches,
            "EXACT_NOOP_INSTRUCTION",
            deliberately_different_old_v1,
        ), mock.patch.object(
            runner.native_runner_v1,
            "canonical_noop_prompt_contract",
            side_effect=AssertionError("old v1 no-op helper was consumed"),
        ):
            negative, noop, action, receipt = runner._encode_conditions(  # noqa: SLF001
                tokenizer=tokenizer,
                renderer=renderer,
                prompt_cleaner=lambda text: text,
                device=torch.device("cpu"),
                local=local,
            )

        expected_noop_prompt = (
            legacy.MV2V_SYSTEM_PROMPT
            + runner.source_consumer.NOOP_INSTRUCTION
        )
        self.assertEqual(
            [text for text, _ in tokenizer.calls],
            [
                expected_noop_prompt,
                legacy.MV2V_SYSTEM_PROMPT
                + runner.ACTION_INSTRUCTION_BY_DP_ARM[0],
                legacy.DEFAULT_NEGATIVE_PROMPT,
            ],
        )
        self.assertTrue(
            all(
                text != deliberately_different_old_v1
                for text, _kwargs in tokenizer.calls
            )
        )
        self.assertTrue(
            all(value.dtype == torch.bfloat16 for value in (negative, noop, action))
        )
        noop_sha = hashlib.sha256(
            runner.source_consumer.NOOP_INSTRUCTION.encode("utf-8")
        ).hexdigest()
        self.assertEqual(receipt["fit_iid"], local.fit_iid)
        self.assertEqual(receipt["confirmation_iid"], local.confirmation_iid)
        self.assertEqual(
            receipt["fit_row_noop_instruction_utf8_sha256"], noop_sha
        )
        self.assertEqual(
            receipt["confirmation_row_noop_instruction_utf8_sha256"],
            noop_sha,
        )
        self.assertEqual(
            receipt["built_noop_prompt_utf8_sha256"],
            hashlib.sha256(expected_noop_prompt.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            receipt["negative_prompt_utf8_sha256"],
            hashlib.sha256(
                legacy.DEFAULT_NEGATIVE_PROMPT.encode("utf-8")
            ).hexdigest(),
        )
        self.assertEqual(
            receipt["authenticated_routing_digest"], local.routing_digest
        )
        self.assertEqual(receipt["fit_source_sha256"], local.fit_row.source_sha256)
        self.assertEqual(
            receipt["confirmation_source_sha256"],
            local.confirmation_row.source_sha256,
        )
        self.assertFalse(receipt["legacy_v1_canonical_noop_helper_consumed"])

        wrong_confirmation = replace(
            local,
            confirmation_row=routing.confirmation_rows[1],
        )
        old_fit = replace(local.fit_row, noop_instruction=old_v1_noop)
        old_confirmation = replace(
            local.confirmation_row, noop_instruction=old_v1_noop
        )
        old_text_local = replace(
            local, fit_row=old_fit, confirmation_row=old_confirmation
        )
        rewritten = "Keep this clip unchanged."
        rewritten_local = replace(
            local,
            fit_row=replace(local.fit_row, noop_instruction=rewritten),
            confirmation_row=replace(
                local.confirmation_row, noop_instruction=rewritten
            ),
        )
        mismatch_local = replace(local, fit_row=old_fit)
        for attacked in (
            wrong_confirmation,
            old_text_local,
            rewritten_local,
            mismatch_local,
        ):
            with self.assertRaisesRegex(
                runner.GraftPhaseAShortGPUError,
                "no-op routing differs",
            ):
                runner._encode_conditions(  # noqa: SLF001
                    tokenizer=RecordingTokenizer(),
                    renderer=renderer,
                    prompt_cleaner=lambda text: text,
                    device=torch.device("cpu"),
                    local=attacked,
                )
        with self.assertRaisesRegex(
            runner.GraftPhaseAShortGPUError, "was rewritten"
        ):
            runner._encode_conditions(  # noqa: SLF001
                tokenizer=RecordingTokenizer(),
                renderer=renderer,
                prompt_cleaner=lambda _text: rewritten,
                device=torch.device("cpu"),
                local=local,
            )

    def test_world8_full_result_assembly_exposes_human_and_rejects_tamper(self) -> None:
        def local_result(
            rank, *, hard_gate=True, authority=False, bad_iid=False, bad_arm=False,
            nested_tamper=False, forged_consensus=False, swap_admissions=False,
            tamper_confirmation=False, tamper_route=False
        ):
            arm = rank // 4
            metrics = {}
            for index in (29, 38):
                metric = runner.seal_mapping(
                    {
                        "schema_version": "bernini-graft-phase-a-confirmation-metrics-v1",
                        "schedule_index": index,
                        "noncompensating_gates": {
                            "correct_vs_wrong_noop_relative_gain": hard_gate,
                            "correct_vs_drop_noop_relative_gain": True,
                            "action_delta_correct_drop_norm_ratio": True,
                            "action_delta_correct_drop_cosine": True,
                        },
                        "noncompensating_all_pass": hard_gate,
                        **{name: False for name in trainer.AUTHORITY_FIELDS},
                    }
                )
                metrics[str(index)] = metric
            if nested_tamper:
                attacked_metric = dict(metrics["29"])
                attacked_metric["noncompensating_all_pass"] = False
                attacked_metric["digest"] = "0" * 64
                metrics["29"] = attacked_metric
            consensus = {}
            admissions = []
            provenances = []
            for index in (29, 38):
                record = {
                    "row_iid": runner.CONFIRMATION_IID_BY_DP_ARM[arm],
                    "wrong_owner_iid": runner.FIT_IID_BY_DP_ARM[arm],
                    "schedule_index": index,
                    "metrics_digest": metrics[str(index)]["digest"],
                    "parameter_digest": "1" * 64,
                    "base_digest": "2" * 64,
                    "optimizer_digest": "3" * 64,
                }
                consensus[str(index)] = (
                    metrics[str(index)]["digest"]
                    if forged_consensus else runner.object_sha256(record)
                )
                admissions.append(runner.seal_mapping({
                    "schema_version": "bernini-graft-phase-a-confirmation-field-admission-v1",
                    **record, "metrics": dict(metrics[str(index)]),
                    "sp4_consensus_digest": consensus[str(index)],
                    "checkpoint_written": False,
                    **{name: False for name in trainer.AUTHORITY_FIELDS},
                }))
                provenances.append(runner.seal_mapping({
                    "schedule_index": index,
                    "confirmation_iid": runner.CONFIRMATION_IID_BY_DP_ARM[arm],
                    "wrong_owner_iid": runner.FIT_IID_BY_DP_ARM[arm],
                    **{name: True for name in runner._CONFIRMATION_TRUE_FLAGS},  # noqa: SLF001
                    **{name: False for name in runner._CONFIRMATION_FALSE_FLAGS},  # noqa: SLF001
                    "same_state_tensor_identities_recomputed_byte_equal": True,
                    "wrong_route_receipts_differ_only_in_atlas_memory": True,
                    "drop_route_receipts_retain_v_branch_disable_only_rebinder": True,
                    "action_noop_route_receipts_equal_with_negative_raw_reuse": True,
                    **{name: False for name in trainer.AUTHORITY_FIELDS},
                }))
            if swap_admissions:
                admissions.reverse()
            short = runner.seal_mapping(
                {
                    "schema_version": trainer.SCHEMA_VERSION,
                    "status": "completed_in_memory_orchestration",
                    "topology": {
                        "rank": rank,
                        "dp_arm": arm,
                        "sp_rank": rank % 4,
                    },
                    "source_routing": {
                        "local_update_iid": runner.FIT_IID_BY_DP_ARM[arm],
                        "local_confirmation_iid": runner.CONFIRMATION_IID_BY_DP_ARM[arm],
                    },
                    "confirmation": {
                        "per_index_metrics": {
                            key: dict(value) for key, value in metrics.items()
                        },
                        "sp4_consensus_digest": consensus,
                        "all_indices_noncompensating_hard_gate_passed": hard_gate,
                    },
                    "checkpoint_written": False,
                    "publication_performed": False,
                    **{name: False for name in trainer.AUTHORITY_FIELDS},
                }
            )
            return runner.seal_mapping(
                {
                    "schema_version": runner.SCHEMA_VERSION,
                    "status": "completed_in_memory_diagnostic_no_checkpoint",
                    "complete": True,
                    "topology": {
                        "rank": rank,
                        "dp_arm": (1 - arm) if bad_arm else arm,
                        "sp_rank": rank % 4,
                        "family": runner.FAMILY_BY_DP_ARM[arm],
                    },
                    "source_routing": {
                        "fit_iid": (
                            "bad" if bad_iid else runner.FIT_IID_BY_DP_ARM[arm]
                        ),
                        "confirmation_iid": runner.CONFIRMATION_IID_BY_DP_ARM[arm],
                    },
                    "confirmation": {
                        "schedule_indices": [29, 38],
                        "field_roles": list(trainer.CONFIRMATION_FIELD_ROLES),
                        "provenance": [dict(row) for row in provenances],
                        "admissions": [dict(row) for row in admissions],
                        "exact_six_fields_per_index": not tamper_confirmation,
                        "same_state_interventions_verified": True,
                        "wrong_atlas_same_family_fit_verified": True,
                        "drop_disables_only_identity_rebinder_memory_verified": True,
                    },
                    "adapter_off_parity": dict(
                        GraftPhaseAShortGPUStateMachineTests._parity_receipt()
                    ),
                    "update_route_receipts": [
                        dict(runner.seal_mapping({
                            "update_number": (2 if tamper_route and index == 29 else (1 if index == 29 else 2)),
                            "schedule_index": index,
                            "row_iid": runner.FIT_IID_BY_DP_ARM[arm],
                            "fit_row_only": True,
                            "exact_four_native_forwards": True,
                            "forward_order": [
                                ["measurement", "negative"],
                                ["measurement", "positive"],
                                ["replay", "negative"],
                                ["replay", "positive"],
                            ],
                            "checkpoint_written": False,
                            **{name: False for name in trainer.AUTHORITY_FIELDS},
                        })) for index in (29, 38)
                    ],
                    "training_updates_executed_for_diagnostic": 2,
                    "full_sampler_used": False,
                    "decoded_media_output_created": False,
                    "target_video_used": False,
                    "generated_proposal_used": False,
                    "t2v_branch_used": False,
                    "source_retelling_used": False,
                    "selector_used": False,
                    "short_trainer_receipt": dict(short),
                    "checkpoint_written": False,
                    "publication_performed": False,
                    "action_authority": authority,
                    **{
                        name: False
                        for name in trainer.AUTHORITY_FIELDS
                        if name != "action_authority"
                    },
                }
            )

        def packets(overrides=None):
            overrides = {} if overrides is None else overrides
            values = []
            for rank in range(8):
                result = local_result(rank, **overrides.get(rank, {}))
                values.append(
                    {
                        "global_rank": rank,
                        "result_digest": result["digest"],
                        "local_result": dict(result),
                    }
                )
            return values

        assembled = runner.assemble_world8_local_results(packets())
        self.assertEqual(len(assembled["all_eight_full_local_receipts"]), 8)
        self.assertEqual(
            [row["representative_global_rank"] for row in assembled["arm_representatives"]],
            [0, 4],
        )
        human = assembled["arm_representatives"][1]["representative_full_receipt"]
        self.assertEqual(human["topology"]["family"], "human")
        self.assertTrue(
            human["short_trainer_receipt"]["confirmation"][
                "all_indices_noncompensating_hard_gate_passed"
            ]
        )
        attacks = [
            packets({4: {"hard_gate": False}}),
            packets({4: {"authority": True}}),
            packets({4: {"bad_iid": True}}),
            packets({4: {"bad_arm": True}}),
            packets({4: {"nested_tamper": True}}),
            packets({4: {"forged_consensus": True}}),
            packets({4: {"swap_admissions": True}}),
            packets({4: {"tamper_confirmation": True}}),
            packets({4: {"tamper_route": True}}),
        ]
        missing = packets()
        missing.pop(0)
        attacks.append(missing)
        swapped = packets()
        swapped[4]["result_digest"] = swapped[0]["result_digest"]
        attacks.append(swapped)
        wrong_rank = packets()
        wrong_rank[4]["global_rank"] = 5
        attacks.append(wrong_rank)
        for attacked in attacks:
            with self.assertRaises(runner.GraftPhaseAShortGPUError):
                runner.assemble_world8_local_results(attacked)

    def _valid_cli_argv(self):
        sha = "a" * 64
        return [
            "--bernini-root", "/tmp/bernini",
            "--veomni-root", "/tmp/veomni",
            "--checkpoint", "/tmp/checkpoint",
            "--checkpoint-content-manifest", "/tmp/checkpoint-manifest.json",
            "--manifest-path", "/tmp/manifest.json",
            "--producer-receipt-path", "/tmp/producer.json",
            "--execution-receipt-path", "/tmp/execution.json",
            "--submission-receipt-path", "/tmp/submission.json",
            "--terminal-admission-path", "/tmp/terminal.json",
            "--manifest-sha256", sha,
            "--producer-receipt-sha256", sha,
            "--execution-receipt-sha256", sha,
            "--submission-receipt-sha256", sha,
            "--terminal-admission-sha256", sha,
            "--terminal-materializer-implementation-sha256", sha,
            "--terminal-materializer-runtime-sha256", sha,
            "--expected-checkpoint-content-manifest-sha256",
            runner.EXPECTED_CHECKPOINT_MANIFEST_SHA256,
            "--expected-runner-sha256", runner.file_sha256(runner.__file__),
            "--expected-identity-rebinder-sha256",
            runner.file_sha256(rebinder.__file__),
            "--expected-bernini-commit", rebinder.PINNED_BERNINI_SOURCE_COMMIT,
            "--expected-veomni-commit", legacy.trainer.VEOMNI_TESTED_COMMIT,
            "--expected-checkpoint-tree-sha256",
            legacy.trainer.CHECKPOINT_TREE_SHA256,
            "--ack-two-update-diagnostic-no-checkpoint-no-scientific-claim",
        ]


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class GraftPhaseAShortGPUStateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.events = []
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
            test_name="cpu_fake:short_gpu_runner",
            forward_context_factory=route,
        )
        generator = torch.Generator(device="cpu").manual_seed(20260810)
        self.source = torch.randn(
            (1, 16, 21, 2, 4), generator=generator, dtype=torch.float32
        )
        self.noisy = torch.randn(
            (1, 16, 21, 2, 4), generator=generator, dtype=torch.float32
        )
        self.negative = torch.full((1, 2, 4), -1.0, dtype=torch.bfloat16)
        self.positive = torch.full((1, 2, 4), 2.0, dtype=torch.bfloat16)
        self.routing = trainer.authenticate_cpu_test_routing(
            test_name="cpu_fake:short_gpu_runner"
        )
        self.backend = trainer.authenticate_cpu_test_collectives(rank=2)

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

    @staticmethod
    def _field_tensors(scale=1.0):
        return {
            "source_noop_target_velocity": torch.tensor(
                [0.0, 0.0, 0.0, 0.0], dtype=torch.float32
            ),
            "correct_atlas_noop_velocity": torch.tensor(
                [0.1 * scale, 0.0, 0.0, 0.0], dtype=torch.float32
            ),
            "wrong_atlas_noop_velocity": torch.tensor(
                [0.3 * scale, 0.0, 0.0, 0.0], dtype=torch.float32
            ),
            "dropped_atlas_noop_velocity": torch.tensor(
                [0.2 * scale, 0.0, 0.0, 0.0], dtype=torch.float32
            ),
            "correct_atlas_action_velocity": torch.tensor(
                [0.3 * scale, 0.0, 0.0, 0.0], dtype=torch.float32
            ),
            "dropped_atlas_action_velocity": torch.tensor(
                [0.4 * scale, 0.0, 0.0, 0.0], dtype=torch.float32
            ),
        }

    @staticmethod
    def _parity_receipt(*, mismatch=False, missing_index=False):
        rows = []
        indices = (0,) if missing_index else runner.ADAPTER_OFF_PARITY_INDICES
        for schedule_index in indices:
            for branch_role in runner.PARITY_BRANCH_ROLES:
                raw_sha = hashlib.sha256(
                    f"{schedule_index}:{branch_role}".encode("ascii")
                ).hexdigest()
                rows.append(
                    {
                        "schedule_index": schedule_index,
                        "branch_role": branch_role,
                        "adapter_route_gate_float64_hex": 0.0.hex(),
                        "adapter_off_raw_sha256": raw_sha,
                        "installed_zero_gate_raw_sha256": (
                            "f" * 64 if mismatch else raw_sha
                        ),
                        "raw_storage_byte_exact": not mismatch,
                        "native_full_source_v_pack_bytes_unchanged": True,
                        "noisy_target_bytes_unchanged": True,
                        "epsilon_bytes_unchanged": True,
                        "sigma_timestep_unchanged": True,
                        "condition_bytes_unchanged": True,
                        "target_video_used": False,
                    }
                )
        return runner.seal_mapping(
            {
                "schema_version": runner.ADAPTER_OFF_PARITY_SCHEMA_VERSION,
                "schedule_indices": list(runner.ADAPTER_OFF_PARITY_INDICES),
                "branch_roles": list(runner.PARITY_BRANCH_ROLES),
                "rows": rows,
                "baseline_captured_before_adapter_install": True,
                "comparison_executed_after_two_updates_and_confirmation": True,
                "all_installed_zero_gate_raw_bytes_equal_adapter_off": not mismatch,
                "raw_dtype": "torch.bfloat16",
                "checkpoint_written": False,
                **{name: False for name in trainer.AUTHORITY_FIELDS},
            }
        )

    def _services(self, *, parity_mode="valid", provenance_attack=None):
        def make_update_cell(*, plan):
            self.events.append(("make_update_cell", plan.schedule_index))
            return self._cell(plan.schedule_index)

        def after_update(*, plan, update_receipt):
            self.events.append(("after_update", plan.schedule_index))
            return runner.seal_mapping(
                {
                    "schema_version": "cpu-fake-short-update-route-v1",
                    "update_number": plan.update_number,
                    "schedule_index": plan.schedule_index,
                    "row_iid": plan.row_iid,
                    "update_receipt_digest": update_receipt["digest"],
                    "exact_four_native_forwards": True,
                    "forward_order": [
                        ["measurement", "negative"],
                        ["measurement", "positive"],
                        ["replay", "negative"],
                        ["replay", "positive"],
                    ],
                    "fit_row_only": True,
                    "checkpoint_written": False,
                    **{name: False for name in trainer.AUTHORITY_FIELDS},
                }
            )

        def make_confirmation_fields(*, plan, schedule_index):
            self.events.append(("make_confirmation_fields", schedule_index))
            scale = 1.0 if schedule_index == 29 else 1.25
            fields = self._field_tensors(scale)
            provenance = runner.build_confirmation_provenance(
                plan=plan,
                schedule_index=schedule_index,
                fields=fields,
                runtime_evidence={
                    "test_only_native_field_source": True,
                    "same_state_identity": {
                        "confirmation_source_zs_sha256": "1" * 64,
                        "native_full_source_v_pack_sha256": "2" * 64,
                        "noisy_target_sha256": "3" * 64,
                        "epsilon_sha256": "4" * 64,
                        "negative_condition_sha256": "5" * 64,
                        "sigma_timestep_coordinate_sha256": "6" * 64,
                    },
                },
            )
            if provenance_attack is not None and schedule_index == 29:
                plain = dict(provenance)
                plain.pop("digest")
                plain[provenance_attack] = False
                provenance = runner.seal_mapping(plain)
            return runner.ConfirmationFieldSet(
                **fields,
                provenance=provenance,
            )

        def adapter_off_parity(*, schedule_indices):
            self.events.append(("adapter_off_parity", tuple(schedule_indices)))
            if parity_mode == "mismatch":
                return self._parity_receipt(mismatch=True)
            if parity_mode == "missing_index":
                return self._parity_receipt(missing_index=True)
            return self._parity_receipt()

        return runner.authenticate_cpu_test_services(
            test_name="cpu_fake:short_gpu_runner_services",
            make_update_cell=make_update_cell,
            after_update=after_update,
            make_confirmation_fields=make_confirmation_fields,
            adapter_off_parity=adapter_off_parity,
        )

    def test_exact_end_to_end_order_six_fields_parity_and_no_authority(self) -> None:
        initial_base = self.transformer.frozen_base.detach().clone()
        services = self._services()
        result = runner.execute_authenticated_short_run(
            routing=self.routing,
            bindings=self.bindings,
            collectives=self.backend,
            services=services,
        )
        self.assertEqual(
            self.events,
            [
                ("make_update_cell", 29),
                ("after_update", 29),
                ("make_update_cell", 38),
                ("after_update", 38),
                ("make_confirmation_fields", 29),
                ("make_confirmation_fields", 38),
                ("adapter_off_parity", (0, 25)),
            ],
        )
        receipt = result.receipt
        full_packet = {
            "global_rank": self.backend.rank,
            "result_digest": receipt["digest"],
            "local_result": dict(receipt),
        }
        serialized = pickle.dumps(full_packet, protocol=5)
        self.assertLess(len(serialized), runner.MAX_FULL_LOCAL_RESULT_PACKET_BYTES)
        self.assertEqual(pickle.loads(serialized), full_packet)
        operations = [row["operation"] for row in receipt["execution_trace"]]
        self.assertEqual(
            operations,
            [
                "open_short_training",
                "next_update_plan",
                "make_native_v2_cell",
                "run_update",
                "admit_update_route_evidence",
                "next_update_plan",
                "make_native_v2_cell",
                "run_update",
                "admit_update_route_evidence",
                "confirmation_plan",
                "measure_six_confirmation_fields",
                "admit_confirmation_fields",
                "measure_six_confirmation_fields",
                "admit_confirmation_fields",
                "admit_adapter_off_bf16_raw_parity",
                "finish_in_memory_short_core",
            ],
        )
        self.assertEqual(self.diffusion.call_count, 8)
        self.assertTrue(torch.equal(self.transformer.frozen_base, initial_base))
        self.assertIsNone(result.checkpoint_payload)
        self.assertIsNone(result.publication_payload)
        self.assertFalse(receipt["checkpoint_written"])
        self.assertFalse(receipt["publication_performed"])
        self.assertEqual(
            receipt["confirmation"]["schedule_indices"], [29, 38]
        )
        self.assertEqual(len(receipt["confirmation"]["provenance"]), 2)
        for provenance in receipt["confirmation"]["provenance"]:
            self.assertEqual(len(provenance["field_roles"]), 6)
            self.assertEqual(len(provenance["field_tensor_identities"]), 6)
            for flag in runner._CONFIRMATION_TRUE_FLAGS:  # noqa: SLF001
                self.assertTrue(provenance[flag], flag)
            for flag in runner._CONFIRMATION_FALSE_FLAGS:  # noqa: SLF001
                self.assertFalse(provenance[flag], flag)
        parity = receipt["adapter_off_parity"]
        self.assertEqual(
            [(row["schedule_index"], row["branch_role"]) for row in parity["rows"]],
            [
                (index, role)
                for index in (0, 25)
                for role in ("negative", "noop_positive", "action_positive")
            ],
        )
        self.assertTrue(
            all(
                row["adapter_off_raw_sha256"]
                == row["installed_zero_gate_raw_sha256"]
                for row in parity["rows"]
            )
        )
        _walk_boolean_authority(self, receipt)
        _walk_boolean_authority(self, services.receipt())

    def test_six_field_validator_rejects_alias_and_missing_identity_flag(self) -> None:
        local = runner.route_local_family(self.routing, dp_arm=0)
        plan = SimpleNamespace(
            row_iid=local.confirmation_iid,
            row=local.confirmation_row,
            wrong_owner_iid=local.fit_iid,
            wrong_owner_row=local.fit_row,
        )
        fields = self._field_tensors()
        provenance = runner.build_confirmation_provenance(
            plan=plan,
            schedule_index=29,
            fields=fields,
            runtime_evidence={"test_only": True},
        )
        valid = runner.ConfirmationFieldSet(**fields, provenance=provenance)
        admitted, _ = runner.validate_confirmation_field_set(
            valid, plan=plan, schedule_index=29
        )
        self.assertEqual(tuple(admitted), trainer.CONFIRMATION_FIELD_ROLES)

        alias_fields = dict(fields)
        alias_fields["wrong_atlas_noop_velocity"] = alias_fields[
            "correct_atlas_noop_velocity"
        ]
        alias_provenance = runner.build_confirmation_provenance(
            plan=plan,
            schedule_index=29,
            fields=alias_fields,
            runtime_evidence={"test_only": True},
        )
        with self.assertRaisesRegex(runner.GraftPhaseAShortGPUError, "alias"):
            runner.validate_confirmation_field_set(
                runner.ConfirmationFieldSet(
                    **alias_fields, provenance=alias_provenance
                ),
                plan=plan,
                schedule_index=29,
            )

        attacked = dict(provenance)
        attacked.pop("digest")
        attacked["drop_retains_native_full_source_v_pack"] = False
        attacked_provenance = runner.seal_mapping(attacked)
        with self.assertRaisesRegex(
            runner.GraftPhaseAShortGPUError, "same-state provenance"
        ):
            runner.validate_confirmation_field_set(
                replace(valid, provenance=attacked_provenance),
                plan=plan,
                schedule_index=29,
            )

    def test_confirmation_provenance_attack_rolls_back_both_updates(self) -> None:
        initial = {name: parameter.detach().clone() for name, parameter in self.names}
        with self.assertRaises(runner.GraftPhaseAShortGPUError) as caught:
            runner.execute_authenticated_short_run(
                routing=self.routing,
                bindings=self.bindings,
                collectives=self.backend,
                services=self._services(
                    provenance_attack=(
                        "wrong_intervention_changes_only_identity_atlas_memory"
                    )
                ),
            )
        diagnostic = caught.exception.diagnostic_receipt
        self.assertEqual(diagnostic["status"], "failed_rolled_back_no_checkpoint")
        self.assertTrue(diagnostic["trainable_parameters_rolled_back"])
        self.assertFalse(diagnostic["checkpoint_written"])
        self.assertFalse(diagnostic["publication_performed"])
        for name, parameter in self.names:
            self.assertTrue(torch.equal(parameter, initial[name]), name)
            self.assertIsNone(parameter.grad)
        _walk_boolean_authority(self, diagnostic)

    def test_index_zero_twenty_five_parity_failure_rolls_back_before_finish(self) -> None:
        initial = {name: parameter.detach().clone() for name, parameter in self.names}
        with self.assertRaises(runner.GraftPhaseAShortGPUError) as caught:
            runner.execute_authenticated_short_run(
                routing=self.routing,
                bindings=self.bindings,
                collectives=self.backend,
                services=self._services(parity_mode="mismatch"),
            )
        diagnostic = caught.exception.diagnostic_receipt
        self.assertEqual(diagnostic["status"], "failed_rolled_back_no_checkpoint")
        self.assertTrue(diagnostic["trainable_parameters_rolled_back"])
        self.assertNotIn(
            "finish_in_memory_short_core",
            [row["operation"] for row in diagnostic["trace"]],
        )
        for name, parameter in self.names:
            self.assertTrue(torch.equal(parameter, initial[name]), name)
        _walk_boolean_authority(self, diagnostic)

    def test_parity_inventory_requires_both_indices_and_three_roles(self) -> None:
        runner.validate_adapter_off_parity(self._parity_receipt())
        with self.assertRaisesRegex(
            runner.GraftPhaseAShortGPUError, "parity receipt"
        ):
            runner.validate_adapter_off_parity(
                self._parity_receipt(missing_index=True)
            )


if __name__ == "__main__":
    unittest.main()
