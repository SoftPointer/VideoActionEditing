#!/usr/bin/env python3

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD_ROOT))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    import full30_action_runtime_v1 as runtime
    import packed_preservation_lora_v2 as packed_core
else:  # pragma: no cover - exercised only on dependency-light hosts
    torch = None
    runtime = None
    packed_core = None


def _fake_unpack(packed, *, spatial_shape):
    """CPU helper fake with the same public graph-preserving contract."""

    batch, channels, phases, height, width = spatial_shape
    patches = packed.reshape(
        batch, phases, height // 2, width // 2, 1, 2, 2, channels
    )
    return (
        patches.permute(0, 7, 1, 4, 2, 5, 3, 6)
        .reshape(batch, channels, phases, height, width)
        .contiguous()
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable")
class Full30ActionRuntimeTests(unittest.TestCase):
    class SpyNative(torch.nn.Conv3d if TORCH_AVAILABLE else object):
        def __init__(self):
            super().__init__(
                16,
                1536,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2),
                bias=True,
            )
            self.input_ids = []

        def forward(self, value):
            self.input_ids.append(id(value))
            return super().forward(value)

    class AdapterController(torch.nn.Module if TORCH_AVAILABLE else object):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.375, dtype=torch.float32))
            self.disabled = False
            self.disable_entries = 0

        @contextmanager
        def disable_adapter(self):
            if self.disabled:
                raise RuntimeError("nested disable_adapter")
            self.disabled = True
            self.disable_entries += 1
            try:
                yield
            finally:
                self.disabled = False

    class SharedStepFake:
        def __init__(self, adapter, mode="target"):
            self.adapter = adapter
            self.mode = mode
            self.calls = []

        def __call__(
            self,
            renderer,
            *,
            model_id,
            noisy_latents,
            rotary_embs,
            target_tokens,
            target_mask,
            timestep,
            condition,
        ):
            self.calls.append(
                {
                    "rotary_id": id(rotary_embs),
                    "timestep_id": id(timestep),
                    "disabled": self.adapter.disabled,
                    "training": self.adapter.training,
                    "inference_mode": torch.is_inference_mode_enabled(),
                    "condition": float(condition),
                    "target_mask": target_mask.detach().clone(),
                }
            )
            base = noisy_latents[..., :64]
            if self.mode == "raise-frozen" and self.adapter.disabled:
                raise RuntimeError("injected frozen failure")
            adapter_term = 0.0 if self.adapter.disabled else self.adapter.scale
            full = base + adapter_term + float(condition)
            if self.mode == "prehead":
                return noisy_latents[:, target_mask, :]
            if self.mode == "full-source-and-target":
                return full
            target = full[:, target_mask, :].contiguous()
            if self.mode == "detached-trainable":
                return target.detach()
            return target

    def _fixture(self, *, mode="target", arm="action+retain", seed=17):
        torch.manual_seed(seed)
        native = self.SpyNative()
        wrapped = packed_core.TypedPackedPatchEmbeddingV2(native)
        with torch.no_grad():
            wrapped.source_delta.weight.fill_(0.002)
            wrapped.target_delta.weight.fill_(-0.003)
            wrapped.source_delta.bias.fill_(0.01)
            wrapped.target_delta.bias.fill_(-0.02)
            wrapped.role_embedding[0].fill_(0.04)
            wrapped.role_embedding[1].fill_(-0.05)
        transformer = SimpleNamespace(patch_embedding=wrapped)
        adapter = self.AdapterController()
        helper = self.SharedStepFake(adapter, mode=mode)
        core = runtime.Full30ActionRuntimeV1(
            renderer=SimpleNamespace(diff_dec=SimpleNamespace()),
            transformer=transformer,
            adapter_controller=adapter,
            shared_step_helper=helper,
            unpack_helper=_fake_unpack,
            test_only_injected_helpers=True,
        )
        tokens = 21
        source = torch.randn(tokens, 16, 1, 2, 2, dtype=torch.float32).contiguous()
        noisy = torch.randn(tokens, 16, 1, 2, 2, dtype=torch.float32).contiguous()
        real = torch.randn(1, 1, 2 * tokens, 64, dtype=torch.float64)
        imag = torch.randn(1, 1, 2 * tokens, 64, dtype=torch.float64)
        rotary = torch.complex(real, imag).to(torch.complex128).contiguous()
        timestep = torch.tensor([500.0], dtype=torch.float32)
        record = runtime.Full30ActionRecordV1(
            row_id="source-001--action",
            source_iid="source-001",
            branch="action",
            source_patches=source,
            noisy_target_patches=noisy,
            rotary_embs=rotary,
            timestep=timestep,
            spatial_shape=(1, 16, 21, 2, 2),
            branch_condition=runtime.ConditionBindingV1(
                role="branch", authority_sha256="a" * 64, condition=2.0
            ),
            noop_condition=runtime.ConditionBindingV1(
                role="noop", authority_sha256="b" * 64, condition=-1.0
            ),
        )
        return core, record, wrapped, native, adapter, helper, arm

    def test_action_only_and_main_have_exact_24_and_32_physical_plans(self):
        action_only = runtime.build_physical_evaluation_plan_v1("action-only")
        main = runtime.build_physical_evaluation_plan_v1("action+retain")
        self.assertEqual(len(action_only), 24)
        self.assertEqual(len(main), 32)
        self.assertEqual(
            [row.branch_slot for row in action_only[:3]],
            ["trainable_branch", "frozen_noop", "frozen_branch"],
        )
        self.assertEqual(
            [row.branch_slot for row in main[:4]],
            [
                "trainable_branch",
                "frozen_noop",
                "frozen_branch",
                "trainable_noop",
            ],
        )
        self.assertEqual(
            runtime.physical_evaluation_plan_receipt_v1("action-only")[
                "physical_evaluation_count"
            ],
            24,
        )
        self.assertEqual(
            runtime.physical_evaluation_plan_receipt_v1("action+retain")[
                "physical_evaluation_count"
            ],
            32,
        )

    def test_batch_unpack_preserves_graph_and_rejects_prehead_width(self):
        packed = torch.randn(2, 21, 64, dtype=torch.float32, requires_grad=True)
        calls = []

        def observed_unpack(value, *, spatial_shape):
            calls.append((id(value), tuple(spatial_shape)))
            return _fake_unpack(value, spatial_shape=spatial_shape)

        spatial = runtime.unpack_post_head_target_velocity_v1(
            packed,
            spatial_shape=(2, 16, 21, 2, 2),
            unpack_helper=observed_unpack,
        )
        self.assertEqual(tuple(spatial.shape), (2, 16, 21, 2, 2))
        self.assertTrue(spatial.requires_grad)
        self.assertIsNotNone(spatial.grad_fn)
        self.assertEqual([shape for _, shape in calls], [(1, 16, 21, 2, 2)] * 2)
        spatial.square().sum().backward()
        self.assertIsNotNone(packed.grad)
        self.assertGreater(float(packed.grad.abs().sum().item()), 0.0)

        with self.assertRaisesRegex(runtime.Full30ActionRuntimeError, "target rows"):
            runtime.unpack_post_head_target_velocity_v1(
                torch.zeros(2, 21, 1536),
                spatial_shape=(2, 16, 21, 2, 2),
                unpack_helper=observed_unpack,
            )
        with self.assertRaisesRegex(runtime.Full30ActionRuntimeError, "spatial shape"):
            runtime.unpack_post_head_target_velocity_v1(
                torch.zeros(2, 21, 64),
                spatial_shape=(2, 16, 20, 2, 2),
                unpack_helper=observed_unpack,
            )

    def test_routes_reuse_exact_objects_freeze_official_and_preserve_trainable_graph(self):
        core, record, wrapped, native, adapter, helper, arm = self._fixture()
        outputs = core.execute_record(arm=arm, record=record)
        self.assertEqual(len(helper.calls), 4)
        self.assertEqual([row["disabled"] for row in helper.calls], [False, True, True, False])
        self.assertEqual(
            [row["training"] for row in helper.calls],
            [True, False, False, True],
        )
        self.assertEqual(
            [row["inference_mode"] for row in helper.calls],
            [False, True, True, False],
        )
        self.assertTrue(adapter.training)
        self.assertEqual(adapter.disable_entries, 2)
        self.assertEqual({row["rotary_id"] for row in helper.calls}, {id(record.rotary_embs)})
        self.assertEqual({row["timestep_id"] for row in helper.calls}, {id(record.timestep)})
        # Compatibility execute_record composes the public strict phases.  The
        # packed bytes/authority are identical, while the noop phase is a real
        # replay and therefore owns a fresh packed tensor object.
        self.assertEqual(len(set(native.input_ids)), 2)
        self.assertEqual(
            [row["condition"] for row in helper.calls], [2.0, -1.0, 2.0, -1.0]
        )
        self.assertTrue(outputs.trainable_branch_velocity.requires_grad)
        self.assertIsNotNone(outputs.trainable_branch_velocity.grad_fn)
        self.assertTrue(outputs.trainable_noop_velocity.requires_grad)
        self.assertFalse(outputs.frozen_noop_velocity.requires_grad)
        self.assertIsNone(outputs.frozen_noop_velocity.grad_fn)
        self.assertFalse(outputs.frozen_branch_velocity.requires_grad)
        loss = outputs.trainable_branch_velocity.square().mean()
        loss = loss + outputs.trainable_noop_velocity.square().mean()
        loss.backward()
        self.assertIsNotNone(adapter.scale.grad)
        self.assertGreater(float(adapter.scale.grad.abs().item()), 0.0)
        self.assertIsNotNone(wrapped.target_delta.weight.grad)
        self.assertGreater(float(wrapped.target_delta.weight.grad.abs().sum().item()), 0.0)
        self.assertEqual(
            outputs.receipt["output_contract"]["stage"], runtime.POST_HEAD_STAGE
        )
        self.assertEqual(
            outputs.receipt["formal_update_evaluation_plan"][
                "physical_evaluation_count"
            ],
            32,
        )
        self.assertTrue(outputs.receipt["frozen_route_contract"]["peft_disable_adapter"])
        self.assertTrue(outputs.receipt["frozen_route_contract"]["model_eval"])
        self.assertTrue(
            outputs.receipt["frozen_route_contract"]["torch_inference_mode"]
        )
        self.assertTrue(
            outputs.receipt["frozen_route_contract"]["official_frozen_native_only"]
        )

    def test_public_action_then_noop_phases_are_strict_replay_with_24_plus_8_plan(self):
        core, record, wrapped, _native, adapter, helper, _arm = self._fixture()
        action = core.execute_action_phase(record=record)
        self.assertEqual([row["disabled"] for row in helper.calls], [False, True, True])
        self.assertEqual(action.receipt["phase"], "action")
        self.assertEqual(
            action.receipt["phase_evaluation_plan"][
                "global_physical_evaluation_count"
            ],
            24,
        )
        action.trainable_branch_velocity.square().mean().backward()
        self.assertIsNotNone(adapter.scale.grad)
        adapter.zero_grad(set_to_none=True)
        wrapped.zero_grad(set_to_none=True)

        noop = core.execute_noop_phase(record=record)
        self.assertEqual(
            [row["disabled"] for row in helper.calls], [False, True, True, False]
        )
        self.assertEqual(noop.receipt["phase"], "noop")
        self.assertEqual(
            noop.receipt["phase_evaluation_plan"][
                "global_physical_evaluation_count"
            ],
            8,
        )
        self.assertEqual(
            action.receipt["input_binding_digest"],
            noop.receipt["input_binding_digest"],
        )
        noop.trainable_noop_velocity.square().mean().backward()
        self.assertIsNotNone(adapter.scale.grad)
        self.assertGreater(float(adapter.scale.grad.abs().item()), 0.0)
        self.assertEqual(
            len(action.receipt["record_branch_trace"])
            + len(noop.receipt["record_branch_trace"]),
            4,
        )

    def test_source_rows_prehead_and_detached_trainable_results_fail_closed(self):
        for mode, pattern in (
            ("full-source-and-target", "post-head target rows"),
            ("prehead", "post-head target rows"),
            ("detached-trainable", "lost its graph"),
        ):
            with self.subTest(mode=mode):
                core, record, *_ = self._fixture(mode=mode)
                with self.assertRaisesRegex(runtime.Full30ActionRuntimeError, pattern):
                    core.execute_record(arm="action-only", record=record)

        core, record, *_ = self._fixture()
        aliased = replace(record, noisy_target_patches=record.source_patches)
        with self.assertRaisesRegex(runtime.Full30ActionRuntimeError, "distinct matching"):
            core.execute_record(arm="action-only", record=aliased)
        wrong_shape = replace(record, spatial_shape=(1, 16, 21, 4, 2))
        with self.assertRaisesRegex(runtime.Full30ActionRuntimeError, "geometry"):
            core.execute_record(arm="action-only", record=wrong_shape)

    def test_receipt_and_outputs_are_deterministic_and_canonical(self):
        first_core, first_record, *_ = self._fixture(seed=101)
        second_core, second_record, *_ = self._fixture(seed=101)
        first = first_core.execute_record(arm="action-only", record=first_record)
        second = second_core.execute_record(arm="action-only", record=second_record)
        self.assertEqual(first.receipt, second.receipt)
        self.assertTrue(
            torch.equal(first.trainable_branch_velocity, second.trainable_branch_velocity)
        )
        self.assertTrue(torch.equal(first.frozen_noop_velocity, second.frozen_noop_velocity))
        encoded = runtime.canonical_receipt_bytes(first.receipt)
        self.assertEqual(encoded, runtime.canonical_json_bytes(first.receipt))
        self.assertEqual(json.loads(encoded.decode("ascii")), first.receipt)

        hostile = copy.deepcopy(first.receipt)
        hostile["input_contract"]["source_rows_selected"] = True
        with self.assertRaisesRegex(runtime.Full30ActionRuntimeError, "digest"):
            runtime.canonical_receipt_bytes(hostile)

    def test_frozen_failure_restores_model_mode_and_adapter_state(self):
        core, record, _wrapped, _native, adapter, _helper, _arm = self._fixture(
            mode="raise-frozen"
        )
        self.assertTrue(adapter.training)
        with self.assertRaisesRegex(
            runtime.Full30ActionRuntimeError,
            "official shared_step target-tail helper failed",
        ):
            core.execute_action_phase(record=record)
        self.assertTrue(adapter.training)
        self.assertFalse(adapter.disabled)


if __name__ == "__main__":
    unittest.main()
