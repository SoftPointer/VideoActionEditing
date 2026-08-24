#!/usr/bin/env python3

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import ast
import hashlib
import inspect
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch

    import graft_phase_a_native_training_closure_v1 as native_v1
    import graft_phase_a_native_training_closure_v2 as native_v2
    import train_graft_phase_a_a_lite_short_v1 as trainer

    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None  # type: ignore[assignment]
    native_v1 = None  # type: ignore[assignment]
    native_v2 = None  # type: ignore[assignment]
    trainer = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class PhaseAShortContractTests(unittest.TestCase):
    def test_dependency_pins_topology_optimizer_and_cells_are_exact(self) -> None:
        trainer._assert_pinned_dependencies()  # noqa: SLF001
        trainer._assert_pinned_ctypes_runtime()  # noqa: SLF001
        self.assertEqual((trainer.WORLD_SIZE, trainer.DP_SIZE, trainer.SP_SIZE), (8, 2, 4))
        self.assertEqual(trainer.UPDATE_SCHEDULE_INDICES, (29, 38))
        self.assertEqual(trainer.UPDATE_REGIMES, ("bootstrap", "post_bootstrap"))
        self.assertEqual(trainer.OPTIMIZER_LEARNING_RATE, 1.0e-3)
        self.assertEqual(trainer.OPTIMIZER_WEIGHT_DECAY, 0.0)
        self.assertEqual(trainer.MAX_GRAD_NORM, 1.0)
        self.assertGreater(trainer.RAW_DIGEST_CHUNK_SIZE_BYTES, 0)
        self.assertLess(trainer.RAW_DIGEST_CHUNK_SIZE_BYTES, 2**31)
        self.assertEqual(
            trainer.PINNED_CONSUMER_SOURCE_SHA256,
            hashlib.sha256(
                Path(trainer.source_consumer.__file__).read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(
            trainer.PINNED_NATIVE_V2_SOURCE_SHA256,
            hashlib.sha256(Path(native_v2.__file__).read_bytes()).hexdigest(),
        )
        self.assertEqual(
            trainer.PINNED_CRITICAL_RUNTIME_SURFACE_SHA256,
            trainer._critical_runtime_surface_sha256(),  # noqa: SLF001
        )
        self.assertEqual(
            trainer.PINNED_TRAINER_EXECUTION_RUNTIME_SHA256,
            trainer._trainer_execution_runtime_sha256(),  # noqa: SLF001
        )
        trainer._assert_pinned_trainer_execution_runtime()  # noqa: SLF001

    def test_preimport_dependency_replacement_does_not_mint_a_new_baseline(self) -> None:
        probe = f"""
import sys
sys.path.insert(0, {str(METHOD_ROOT)!r})
import graft_a_lite_source_release_consumer_v1 as consumer

class ReplacementRouting:
    pass

ReplacementRouting.__module__ = consumer.__name__
ReplacementRouting.__qualname__ = "TrainerRouting"
consumer.TrainerRouting = ReplacementRouting

import train_graft_phase_a_a_lite_short_v1 as trainer
try:
    trainer._assert_pinned_dependencies()
except trainer.GraftPhaseAShortTrainingError:
    print("REJECTED")
else:
    raise SystemExit("pre-import replacement was accepted")
"""
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "REJECTED")

    def test_preimport_adamw_method_replacement_does_not_mint_a_baseline(self) -> None:
        probe = f"""
import sys
sys.path.insert(0, {str(METHOD_ROOT)!r})
import torch
torch.optim.AdamW.step = lambda self, closure=None: None

import train_graft_phase_a_a_lite_short_v1 as trainer
try:
    trainer._assert_pinned_dependencies()
except trainer.GraftPhaseAShortTrainingError:
    print("REJECTED")
else:
    raise SystemExit("pre-import AdamW.step replacement was accepted")
"""
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "REJECTED")

    def test_preimport_ctypes_string_at_replacement_is_rejected_without_call(
        self,
    ) -> None:
        for attribute in ("string_at", "_string_at"):
            with self.subTest(attribute=attribute):
                probe = f"""
import sys
sys.path.insert(0, {str(METHOD_ROOT)!r})
import ctypes

calls = []
def replacement(*_args, **_kwargs):
    calls.append("called")
    return b""
setattr(ctypes, {attribute!r}, replacement)

import train_graft_phase_a_a_lite_short_v1 as trainer
try:
    trainer._assert_pinned_trainer_execution_runtime()
except trainer.GraftPhaseAShortTrainingError:
    print("REJECTED", len(calls))
else:
    raise SystemExit("pre-import ctypes.string_at replacement was accepted")
"""
                completed = subprocess.run(
                    [sys.executable, "-c", probe],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(completed.stdout.strip(), "REJECTED 0")

    def test_ctypes_source_pin_python_312_simulation_is_exact(self) -> None:
        python_312_sha256 = (
            "30e5adafeec1e9a07cd4848465995c49f0124ddedd31ab3ee627b218086839eb"
        )
        trainer._assert_pinned_ctypes_string_at_source_sha256(  # noqa: SLF001
            python_minor=(3, 12),
            observed_sha256=python_312_sha256,
        )
        self.assertEqual(
            trainer._expected_ctypes_string_at_source_sha256(  # noqa: SLF001
                (3, 12)
            ),
            python_312_sha256,
        )
        self.assertEqual(
            set(
                trainer._PINNED_CTYPES_STRING_AT_SOURCE_SHA256_BY_PYTHON_MINOR
            ),
            {(3, 8), (3, 10), (3, 12)},
        )

    def test_ctypes_source_pin_unsupported_minor_fails_closed(self) -> None:
        for python_minor in ((3, 9), (3, 11), (3, 13), (4, 0)):
            with self.subTest(python_minor=python_minor), self.assertRaisesRegex(
                trainer.GraftPhaseAShortTrainingError,
                "Python minor is unsupported",
            ):
                trainer._expected_ctypes_string_at_source_sha256(  # noqa: SLF001
                    python_minor
                )

    def test_adamw_step_runtime_pins_cover_only_exact_supported_variants(
        self,
    ) -> None:
        expected = {
            "2.2.1": (
                "torch.optim.adamw",
                "AdamW.step",
                ("optim", "adamw.py"),
                "0a974170c9297c6d1d3a65d472e245012cc3dca7342b78cd373a2712588daaad",
            ),
            "2.4.0": (
                "torch.optim.adamw",
                "AdamW.step",
                ("optim", "adamw.py"),
                "bad2ab4efb8a3ffd8e383d46dd73f3cacd4136d64bfb3b190e4549d79ed3671a",
            ),
            "2.7.1+rocm6.3": (
                "torch.optim.adam",
                "Adam.step",
                ("optim", "adam.py"),
                "a30c184cf93b5e1b06bf6abf2ef3642eb1e6c2a03992e4914f538239fd28a5f2",
            ),
        }
        self.assertEqual(
            dict(trainer._PINNED_ADAMW_STEP_RUNTIME_BY_TORCH_VERSION),  # noqa: SLF001
            expected,
        )
        for torch_version, runtime in expected.items():
            with self.subTest(torch_version=torch_version):
                self.assertEqual(
                    trainer._expected_adamw_step_runtime(torch_version),  # noqa: SLF001
                    runtime,
                )
                trainer._assert_pinned_adamw_step_runtime_contract(  # noqa: SLF001
                    torch_version=torch_version,
                    observed_module=runtime[0],
                    observed_qualname=runtime[1],
                    observed_source_path_suffix=runtime[2],
                    observed_source_sha256=runtime[3],
                )

    def test_adamw_step_runtime_unsupported_version_fails_closed(self) -> None:
        for torch_version in (
            "",
            "2.2.1+cpu",
            "2.4.0+cu121",
            "2.7.1",
            "2.7.1+rocm6.2",
            "2.8.0+rocm6.3",
        ):
            with self.subTest(torch_version=torch_version), self.assertRaisesRegex(
                trainer.GraftPhaseAShortTrainingError,
                "torch version is unsupported",
            ):
                trainer._expected_adamw_step_runtime(torch_version)  # noqa: SLF001

    def test_adamw_step_runtime_field_mismatch_fails_closed(self) -> None:
        runtime = trainer._expected_adamw_step_runtime(  # noqa: SLF001
            "2.7.1+rocm6.3"
        )
        observed = {
            "torch_version": "2.7.1+rocm6.3",
            "observed_module": runtime[0],
            "observed_qualname": runtime[1],
            "observed_source_path_suffix": runtime[2],
            "observed_source_sha256": runtime[3],
        }
        mutations = {
            "module": {"observed_module": "torch.optim.adamw"},
            "qualname": {"observed_qualname": "AdamW.step"},
            "path": {
                "observed_source_path_suffix": ("optim", "adamw.py")
            },
            "source_sha256": {"observed_source_sha256": "0" * 64},
        }
        for label, mutation in mutations.items():
            with self.subTest(field=label), self.assertRaisesRegex(
                trainer.GraftPhaseAShortTrainingError,
                "runtime provenance differs",
            ):
                trainer._assert_pinned_adamw_step_runtime_contract(  # noqa: SLF001
                    **{**observed, **mutation}
                )

    def test_live_optimizer_and_clipper_namespace_replacement_is_rejected(self) -> None:
        original = torch.nn.utils.clip_grad_norm_
        try:
            torch.nn.utils.clip_grad_norm_ = lambda *_args, **_kwargs: 0.0
            with self.assertRaisesRegex(
                trainer.GraftPhaseAShortTrainingError,
                "live dependency namespace",
            ):
                trainer._assert_pinned_dependencies()  # noqa: SLF001
        finally:
            torch.nn.utils.clip_grad_norm_ = original

    def test_live_v2_explicit_dunder_replacement_is_rejected(self) -> None:
        original = native_v2.PhaseANativeTrainingClosure.__getattribute__
        try:
            native_v2.PhaseANativeTrainingClosure.__getattribute__ = (
                object.__getattribute__
            )
            with self.assertRaisesRegex(
                trainer.GraftPhaseAShortTrainingError,
                "critical consumer/native runtime surface",
            ):
                trainer._assert_pinned_dependencies()  # noqa: SLF001
        finally:
            native_v2.PhaseANativeTrainingClosure.__getattribute__ = original

    def test_public_orchestration_api_has_no_phase_b_or_endpoint_inputs(self) -> None:
        forbidden = {
            "proposal",
            "asga",
            "selector",
            "retelling",
            "caption",
            "target_video",
            "generated_video",
        }
        callables = (
            trainer.open_authenticated_short_training,
            trainer.authenticate_torch_distributed_world8_dp2sp4,
            trainer.PhaseAShortTrainingSession.run_update,
            trainer.PhaseAShortTrainingSession.record_confirmation_fields,
        )
        for function in callables:
            fields = {name.lower() for name in inspect.signature(function).parameters}
            for fragment in forbidden:
                self.assertFalse(
                    any(fragment in field for field in fields),
                    (function.__qualname__, fragment, fields),
                )
        source = Path(trainer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("graft_action_first_source_guided_aggregation_v1", imported)
        self.assertNotIn("graft_source_conditioned_proposal_selector_v1", imported)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(
            any(
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"save", "save_file", "atomic_torch_save"}
                for node in calls
            )
        )
        self.assertNotIn("argparse", imported)

    def test_path_free_test_routing_is_exact_two_plus_two(self) -> None:
        routing = trainer.authenticate_cpu_test_routing()
        self.assertEqual(len(routing.update_rows), 2)
        self.assertEqual(len(routing.confirmation_rows), 2)
        self.assertTrue(all(row.optimizer_update_allowed for row in routing.update_rows))
        self.assertTrue(
            all(row.optimizer_confirmation_only for row in routing.confirmation_rows)
        )
        for row in routing.update_rows + routing.confirmation_rows:
            self.assertFalse(hasattr(row, "source_path"))
            self.assertIs(type(row.source_bytes), bytes)
            self.assertEqual(hashlib.sha256(row.source_bytes).hexdigest(), row.source_sha256)

    def test_none_gradients_materialize_before_sp_then_dp_average(self) -> None:
        events = []

        def reducer(value, axis, name, update_number):
            events.append((axis, name, update_number, float(value.item())))
            if name.endswith("output.weight"):
                value.fill_(8.0 if axis == "sp" else 6.0)
            else:
                value.zero_()

        backend = trainer.authenticate_cpu_test_collectives(
            rank=0, gradient_reducer=reducer
        )
        names = (
            "atlas_encoder.proj.weight",
            "blocks.8.attn1.to_out.0.identity_rebinder.query.weight",
            "blocks.8.attn1.to_out.0.identity_rebinder.key.weight",
            "blocks.8.attn1.to_out.0.identity_rebinder.value.weight",
            "blocks.8.attn1.to_out.0.identity_rebinder.output.weight",
        )
        named = tuple((name, torch.nn.Parameter(torch.tensor(0.0))) for name in names)
        categories = {
            category: tuple(
                row for row in named if trainer._gradient_category(row[0]) == category  # noqa: SLF001
            )
            for category in native_v2.GRADIENT_CATEGORIES
        }
        receipt = trainer._synchronize_dp2_sp4_gradients(  # noqa: SLF001
            named=named,
            categories=categories,
            backend=backend,
            update_number=1,
            expected_regime="bootstrap",
        )
        self.assertEqual(receipt["none_materialized_as_true_zero_count"], 5)
        self.assertEqual(receipt["gate"], "world8_bootstrap_output_projection_only_nonzero")
        for name, parameter in named[:-1]:
            self.assertEqual(parameter.grad.item(), 0.0, name)
        self.assertEqual(named[-1][1].grad.item(), 3.0)
        output_events = [row for row in events if row[1].endswith("output.weight")]
        self.assertEqual(output_events, [("sp", names[-1], 1, 0.0), ("dp", names[-1], 1, 2.0)])

    def test_ctypes_digest_matches_old_storage_bytes_for_supported_dtypes(
        self,
    ) -> None:
        dtype_names = (
            "uint8",
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "bfloat16",
            "float32",
            "float64",
            "bool",
            "complex32",
            "complex64",
            "complex128",
            "float8_e4m3fn",
            "float8_e4m3fnuz",
            "float8_e5m2",
            "float8_e5m2fnuz",
            "uint16",
            "uint32",
            "uint64",
            "bits1x8",
            "bits2x4",
            "bits4x2",
            "bits8",
            "bits16",
        )
        cases = []
        for name in dtype_names:
            dtype = getattr(torch, name, None)
            if dtype is None:
                continue
            value = torch.empty((3, 4), dtype=dtype)
            try:
                value.detach().cpu().contiguous().clone()
            except RuntimeError:
                continue
            cases.append((name, value))
        for name, dtype, zero_point in (
            ("qint8", torch.qint8, 0),
            ("quint8", torch.quint8, 3),
            ("qint32", torch.qint32, 0),
        ):
            cases.append(
                (
                    name,
                    torch.quantize_per_tensor(
                        torch.tensor([-1.25, 0.0, 2.5, 7.0]),
                        scale=0.25,
                        zero_point=zero_point,
                        dtype=dtype,
                    ),
                )
            )
        cases.extend(
            (
                (
                    "noncontiguous_offset_int64",
                    torch.arange(48, dtype=torch.int64)
                    .reshape(6, 8)
                    .t()[1:, 1::2],
                ),
                (
                    "noncontiguous_offset_float32",
                    torch.arange(48, dtype=torch.float32)
                    .reshape(6, 8)
                    .t()[1:, 1::2],
                ),
                (
                    "float32_nan_payload_and_signed_zero",
                    torch.tensor(
                        [0x00000000, -0x80000000, 0x7FC00001, -0x003FFFFE],
                        dtype=torch.int32,
                    ).view(torch.float32),
                ),
            )
        )
        self.assertGreaterEqual(len(cases), 20)
        for label, value in cases:
            with self.subTest(label=label, dtype=str(value.dtype)):
                owned = value.detach().cpu().contiguous().clone()
                storage = owned.untyped_storage()
                expected_raw = bytes(storage)
                self.assertEqual(
                    len(expected_raw),
                    int(owned.numel()) * int(owned.element_size()),
                )
                self.assertEqual(
                    trainer._tensor_bytes_sha256(value),  # noqa: SLF001
                    hashlib.sha256(expected_raw).hexdigest(),
                )

    def test_ctypes_digest_zero_and_nonzero_pointer_paths_are_exact(self) -> None:
        empty = torch.empty((0, 3), dtype=torch.float32)
        zero_calls = []

        def forbidden_zero_read(*_args):
            zero_calls.append("called")
            raise AssertionError("zero storage pointer must not be read")

        with mock.patch.object(
            trainer,
            "_PINNED_CTYPES_STRING_AT",
            new=forbidden_zero_read,
        ):
            observed_empty = trainer._tensor_bytes_sha256(empty)  # noqa: SLF001
        self.assertEqual(zero_calls, [])
        self.assertEqual(observed_empty, hashlib.sha256(b"").hexdigest())

        value = torch.arange(17, dtype=torch.int16)
        owned = value.detach().cpu().contiguous().clone()
        expected = hashlib.sha256(bytes(owned.untyped_storage())).hexdigest()
        original = trainer._PINNED_CTYPES_STRING_AT  # noqa: SLF001
        reads = []

        def observe(pointer, size):
            reads.append((pointer, size))
            return original(pointer, size)

        with mock.patch.object(
            trainer, "_PINNED_CTYPES_STRING_AT", new=observe
        ), mock.patch.object(trainer, "RAW_DIGEST_CHUNK_SIZE_BYTES", new=7):
            observed = trainer._tensor_bytes_sha256(value)  # noqa: SLF001
        self.assertEqual(observed, expected)
        self.assertGreater(len(reads), 1)
        self.assertTrue(all(pointer > 0 and 0 < size <= 7 for pointer, size in reads))
        self.assertEqual(sum(size for _pointer, size in reads), value.numel() * value.element_size())

    def test_digest_consensus_failure_is_not_silently_accepted(self) -> None:
        backend = trainer.authenticate_cpu_test_collectives(
            rank=2,
            digest_consensus=lambda _value, _scope, _label: "0" * 64,
        )
        with self.assertRaisesRegex(
            trainer.GraftPhaseAShortTrainingError, "consensus"
        ):
            backend.consensus("1" * 64, scope="world", label="attack")

    def test_test_backend_cannot_be_promoted_or_acquire_old_callback_slots(self) -> None:
        backend = trainer.authenticate_cpu_test_collectives(rank=2)
        with self.assertRaises(AttributeError):
            object.__setattr__(backend, "_reduce", lambda *_args: None)
        object.__setattr__(backend, "_kind", "torch_distributed")
        object.__setattr__(backend, "_test_only", False)
        with self.assertRaisesRegex(
            trainer.GraftPhaseAShortTrainingError,
            "collective backend",
        ):
            backend.assert_live()


if _TORCH_AVAILABLE:
    class _MomentumBuffer:
        def __init__(self, momentum):
            self.momentum = momentum
            self.running_average = 0

        def update(self, update_value):
            self.running_average = update_value + self.momentum * self.running_average


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
            self.route_active = True

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
            raise AssertionError("the short core/native v2 cell must not access a scheduler")


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
            if not self.transformer.route_active:
                raw = (
                    base * (1.0 + 0.03125 * text)
                    + 0.0078125 * self.transformer.frozen_base
                )
                return raw.to(torch.bfloat16)
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
class PhaseAShortStateMachineTests(unittest.TestCase):
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
            test_name="cpu_fake:phase_a_short_state_machine",
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
            test_name="cpu_fake:phase_a_short_state_machine"
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

    def _open(self):
        return trainer.open_authenticated_short_training(
            routing=self.routing,
            bindings=self.bindings,
            collectives=self.backend,
        )

    def _run_two_updates(self, session):
        first_plan = session.next_update_plan()
        self.assertEqual((first_plan.schedule_index, first_plan.expected_regime), (29, "bootstrap"))
        first = session.run_update(plan=first_plan, cell=self._cell(29))
        self.assertEqual(session.phase, "update_2_pending")
        self.assertEqual(first["gradient_sync"]["gate"], "world8_bootstrap_output_projection_only_nonzero")
        second_plan = session.next_update_plan()
        self.assertEqual((second_plan.schedule_index, second_plan.expected_regime), (38, "post_bootstrap"))
        second = session.run_update(plan=second_plan, cell=self._cell(38))
        self.assertEqual(session.phase, "confirmation_pending")
        self.assertEqual(second["gradient_sync"]["gate"], "world8_post_bootstrap_all_five_categories_nonzero")
        return first, second

    def _source_only_result(self, *, world_rank, expected_regime):
        sp_rank = world_rank % trainer.SP_SIZE
        self.transformer.route_active = False

        @contextmanager
        def route(*, request):
            local_rows = (request.total_tokens + 3) // 4
            padded = local_rows * 4
            selector = torch.cat(
                (
                    torch.zeros(request.condition_tokens, dtype=torch.bool),
                    torch.ones(request.target_tokens, dtype=torch.bool),
                    torch.zeros(padded - request.total_tokens, dtype=torch.bool),
                )
            )[
                sp_rank * local_rows : (sp_rank + 1) * local_rows
            ].contiguous()
            targets = int(torch.count_nonzero(selector).item())
            yield native_v1.build_native_forward_context_observation(
                request=request,
                sequence_parallel_rank=sp_rank,
                sequence_parallel_size=trainer.SP_SIZE,
                local_target_selector=selector,
                route_gate=1.0,
                adapter_graph_bearing=(
                    request.phase == "replay" and targets > 0
                ),
            )

        bindings = native_v2.authenticate_cpu_test_fakes(
            diffusion=self.diffusion,
            transformer=self.transformer,
            vendor_normalized_guidance=_normalized_guidance,
            momentum_buffer_factory=_MomentumBuffer,
            named_trainable_parameters=self.names,
            external_trainable_owner_modules={"atlas_encoder": self.atlas},
            test_name=f"cpu_fake:source_only_world_rank_{world_rank}",
            forward_context_factory=route,
        )
        def target_owner_collective(
            value, axis, parameter_name, update_number
        ):
            if update_number != 1:
                raise AssertionError("only the bootstrap collective is simulated")
            if axis == "sp":
                if parameter_name.endswith("output.weight"):
                    value.fill_(float(trainer.SP_SIZE))
                else:
                    value.zero_()
            else:
                value.mul_(float(trainer.DP_SIZE))

        backend = trainer.authenticate_cpu_test_collectives(
            rank=world_rank,
            gradient_reducer=target_owner_collective,
        )
        session = trainer.open_authenticated_short_training(
            routing=self.routing,
            bindings=bindings,
            collectives=backend,
        )
        def make_cell(schedule_index):
            sigma = torch.tensor(
                native_v1.sigma_strata.PINNED_POSITIVE_SIGMAS[schedule_index],
                dtype=torch.float32,
            )
            timestep = torch.tensor(
                [native_v1.sigma_strata.PINNED_TIMESTEPS[schedule_index]],
                dtype=torch.int64,
            )
            return native_v2.PhaseANativeTrainingClosure(
                bindings=bindings,
                source_video=self.source,
                noisy_target=self.noisy,
                negative_condition=self.negative,
                positive_condition=self.positive,
                schedule_index=schedule_index,
                sigma=sigma,
                timestep=timestep,
            )

        plan = session.next_update_plan()
        if expected_regime == "post_bootstrap":
            session.run_update(
                plan=plan,
                cell=make_cell(plan.schedule_index),
            )
            plan = session.next_update_plan()
        self.assertEqual(plan.expected_regime, expected_regime)
        cell = make_cell(plan.schedule_index)
        cell.measure()
        cell.derive_phase_a_flow_matching_vjp()
        return bindings, backend, plan, cell.replay_and_backward()

    def test_bootstrap_source_only_ranks_use_exact_zero_gate_and_reject_mixed_gate(
        self,
    ) -> None:
        for world_rank in (0, 1, 4, 5):
            with self.subTest(world_rank=world_rank):
                bindings, backend, plan, result = (
                    self._source_only_result(
                        world_rank=world_rank,
                        expected_regime="bootstrap",
                    )
                )
                self.assertEqual(result.receipt["local_target_rows"], 0)
                self.assertFalse(
                    result.receipt["bootstrap_output_only_gate_verified"]
                )
                self.assertFalse(
                    result.receipt[
                        "post_bootstrap_five_category_local_gate_verified"
                    ]
                )
                self.assertTrue(
                    result.receipt[
                        "source_only_sp_all_five_categories_exact_zero_verified"
                    ]
                )
                admission = trainer._validate_native_result(  # noqa: SLF001
                    result=result,
                    bindings=bindings,
                    backend=backend,
                    plan=plan,
                )
                self.assertEqual(
                    admission["schema_version"],
                    "bernini-graft-native-v2-cell-admission-v2",
                )
                self.assertFalse(admission["local_target_owner"])
                self.assertEqual(
                    admission["local_gradient_gate"],
                    "source_only_sp_rank_all_five_categories_exact_zero",
                )

                attacked_receipt = dict(result.receipt)
                attacked_receipt.pop("digest")
                attacked_receipt["bootstrap_output_only_gate_verified"] = True
                attacked = replace(
                    result,
                    receipt=native_v1._seal(attacked_receipt),  # noqa: SLF001
                )
                with self.assertRaisesRegex(
                    trainer.GraftPhaseAShortTrainingError,
                    "bootstrap v2 gate differs",
                ):
                    trainer._validate_native_result(  # noqa: SLF001
                        result=attacked,
                        bindings=bindings,
                        backend=backend,
                        plan=plan,
                    )

    def test_post_bootstrap_source_only_gate_rejects_mixed_regime_claim(
        self,
    ) -> None:
        for world_rank in (0, 1, 4, 5):
            with self.subTest(world_rank=world_rank):
                with torch.no_grad():
                    self.transformer.output.zero_()
                for _, parameter in self.names:
                    parameter.grad = None
                bindings, backend, plan, result = self._source_only_result(
                    world_rank=world_rank,
                    expected_regime="post_bootstrap",
                )
                self.assertEqual(result.receipt["local_target_rows"], 0)
                self.assertFalse(
                    result.receipt["bootstrap_output_only_gate_verified"]
                )
                self.assertFalse(
                    result.receipt[
                        "post_bootstrap_five_category_local_gate_verified"
                    ]
                )
                self.assertTrue(
                    result.receipt[
                        "source_only_sp_all_five_categories_exact_zero_verified"
                    ]
                )
                admission = trainer._validate_native_result(  # noqa: SLF001
                    result=result,
                    bindings=bindings,
                    backend=backend,
                    plan=plan,
                )
                self.assertEqual(
                    admission["local_gradient_gate"],
                    "source_only_sp_rank_all_five_categories_exact_zero",
                )

                attacked_receipt = dict(result.receipt)
                attacked_receipt.pop("digest")
                attacked_receipt[
                    "post_bootstrap_five_category_local_gate_verified"
                ] = True
                attacked = replace(
                    result,
                    receipt=native_v1._seal(attacked_receipt),  # noqa: SLF001
                )
                with self.assertRaisesRegex(
                    trainer.GraftPhaseAShortTrainingError,
                    "post-bootstrap v2 gate differs",
                ):
                    trainer._validate_native_result(  # noqa: SLF001
                        result=attacked,
                        bindings=bindings,
                        backend=backend,
                        plan=plan,
                    )

    @staticmethod
    def _passing_confirmation_fields(scale=1.0):
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

    def test_two_step_state_machine_and_no_grad_confirmation(self) -> None:
        base_before = self.transformer.frozen_base.detach().clone()
        session = self._open()
        first, second = self._run_two_updates(session)
        self.assertEqual(session.optimizer_steps, 2)
        self.assertTrue(torch.equal(self.transformer.frozen_base, base_before))
        self.assertTrue(all(parameter.grad is None for _, parameter in self.names))
        self.assertEqual(first["optimizer"]["learning_rate"], 1.0e-3)
        self.assertEqual(second["optimizer"]["weight_decay"], 0.0)
        self.assertEqual(second["max_grad_norm"], 1.0)

        plan = session.confirmation_plan()
        self.assertTrue(plan.row.optimizer_confirmation_only)
        self.assertTrue(plan.wrong_owner_row.optimizer_update_allowed)
        self.assertEqual(plan.wrong_owner_iid, self.routing.update_rows[0].iid)
        with torch.no_grad():
            for schedule_index, scale in ((29, 1.0), (38, 1.25)):
                observation = session.record_confirmation_fields(
                    plan=plan,
                    schedule_index=schedule_index,
                    **self._passing_confirmation_fields(scale),
                )
                self.assertTrue(observation["no_grad"])
                self.assertFalse(observation["optimizer_update_performed"])
                self.assertTrue(
                    observation["metrics"]["noncompensating_all_pass"]
                )
        result = session.finish()
        self.assertIsNone(result.checkpoint_payload)
        self.assertIsNone(result.publication_payload)
        receipt = result.receipt
        self.assertEqual(receipt["status"], "completed_in_memory_orchestration")
        self.assertEqual(receipt["optimizer_contract"]["steps"], 2)
        self.assertEqual(receipt["optimizer_contract"]["schedule_indices"], [29, 38])
        self.assertFalse(receipt["source_routing"]["confirmation_rows_consumed_by_optimizer"])
        self.assertTrue(
            receipt["confirmation"]["all_indices_noncompensating_hard_gate_passed"]
        )
        self.assertEqual(
            set(receipt["confirmation"]["per_index_metrics"]), {"29", "38"}
        )
        self.assertEqual(
            receipt["confirmation"]["runner_adapter_off_parity_indices"],
            [0, 25],
        )
        self.assertFalse(
            receipt["confirmation"]["runner_adapter_off_parity_verified_by_this_core"]
        )
        self.assertFalse(receipt["checkpoint_written"])
        self.assertFalse(receipt["publication_performed"])
        self.assertTrue(receipt["trainer_execution_runtime_live_verified"])
        self.assertFalse(
            receipt["same_process_formal_security_proven_by_this_core"]
        )
        self.assertEqual(receipt["initial_frozen_base_digest"], receipt["final_frozen_base_digest"])
        self.assertNotEqual(receipt["initial_parameter_digest"], receipt["final_parameter_digest"])
        self.assertEqual(session.phase, "closed")

        def walk(value):
            if isinstance(value, dict) or hasattr(value, "items"):
                for key, item in value.items():
                    if (
                        key in trainer.AUTHORITY_FIELDS
                        or key.endswith("_authorized")
                        or "authority" in key
                    ) and isinstance(item, bool):
                        self.assertFalse(item, key)
                    walk(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    walk(item)

        walk(receipt)

    def test_confirmation_grad_mode_failure_rolls_back_and_has_no_checkpoint(self) -> None:
        initial = {name: parameter.detach().clone() for name, parameter in self.names}
        session = self._open()
        self._run_two_updates(session)
        plan = session.confirmation_plan()
        self.assertTrue(torch.is_grad_enabled())
        with self.assertRaisesRegex(
            trainer.GraftPhaseAShortTrainingError, "torch.no_grad"
        ):
            session.record_confirmation_fields(
                plan=plan,
                schedule_index=29,
                **self._passing_confirmation_fields(),
            )
        self.assertEqual(session.phase, "failed")
        failure = session.failure_receipt()
        self.assertEqual(failure["status"], "failed_rolled_back_no_checkpoint")
        self.assertTrue(failure["trainable_parameters_restored_to_initial_snapshot"])
        self.assertFalse(failure["checkpoint_written"])
        self.assertFalse(failure["checkpoint_payload_returned"])
        self.assertFalse(
            failure["same_process_formal_security_proven_by_this_core"]
        )
        for name, parameter in self.names:
            self.assertTrue(torch.equal(parameter, initial[name]), name)
            self.assertIsNone(parameter.grad)

    def test_confirmation_split_cannot_be_promoted_or_forged(self) -> None:
        invalid_update = replace(
            self.routing.update_rows[0],
            split="optimizer_confirmation",
            optimizer_update_allowed=False,
            optimizer_confirmation_only=True,
        )
        attacked = replace(
            self.routing,
            update_rows=(invalid_update, self.routing.update_rows[1]),
        )
        with self.assertRaisesRegex(
            trainer.GraftPhaseAShortTrainingError, "routed source row"
        ):
            trainer.open_authenticated_short_training(
                routing=attacked,
                bindings=self.bindings,
                collectives=self.backend,
            )

        session = self._open()
        real = session.next_update_plan()
        forged = object.__new__(trainer.UpdateCellPlan)
        for name in trainer.UpdateCellPlan.__dataclass_fields__:
            object.__setattr__(forged, name, getattr(real, name))
        object.__setattr__(forged, "row", self.routing.confirmation_rows[0])
        with self.assertRaisesRegex(
            trainer.GraftPhaseAShortTrainingError, "wrong split"
        ):
            session.run_update(plan=forged, cell=self._cell(29))
        self.assertEqual(session.phase, "failed")
        self.assertFalse(session.failure_receipt()["checkpoint_written"])

    def test_routing_bytes_or_digest_cannot_change_after_open(self) -> None:
        session = self._open()
        object.__setattr__(
            self.routing.update_rows[0],
            "source_bytes",
            b"cpu fake routing mutation",
        )
        object.__setattr__(self.routing, "routing_digest", "f" * 64)
        with self.assertRaisesRegex(
            trainer.GraftPhaseAShortTrainingError,
            "routed source row|routing",
        ):
            session.next_update_plan()

    def test_session_slots_reject_instance_execution_method_shadow(self) -> None:
        session = self._open()
        with self.assertRaises(AttributeError):
            object.__setattr__(session, "_assert_live", lambda: None)
        with self.assertRaises(TypeError):
            vars(session)

    def test_optimizer_live_method_replacement_fails_before_call_and_rolls_back(
        self,
    ) -> None:
        initial = {
            name: parameter.detach().clone() for name, parameter in self.names
        }
        session = self._open()
        first_plan = session.next_update_plan()
        session.run_update(plan=first_plan, cell=self._cell(29))
        second_plan = session.next_update_plan()
        second_cell = self._cell(38)
        calls = []

        def substitute(_session):
            calls.append("called")

        with mock.patch.object(
            trainer.PhaseAShortTrainingSession,
            "_assert_optimizer_live",
            new=substitute,
        ):
            session._optimizer.param_groups[0]["lr"] = 0.5  # noqa: SLF001
            with self.assertRaisesRegex(
                trainer.GraftPhaseAShortTrainingError,
                "live short-trainer execution runtime",
            ):
                session.run_update(plan=second_plan, cell=second_cell)

        self.assertEqual(calls, [])
        self.assertEqual(session.phase, "failed")
        failure = session.failure_receipt()
        self.assertTrue(
            failure["trainable_parameters_restored_to_initial_snapshot"]
        )
        self.assertFalse(failure["checkpoint_written"])
        for name, parameter in self.names:
            self.assertTrue(torch.equal(parameter, initial[name]), name)

    def test_routing_live_helper_replacement_fails_before_call_and_rolls_back(
        self,
    ) -> None:
        initial = {
            name: parameter.detach().clone() for name, parameter in self.names
        }
        session = self._open()
        first_plan = session.next_update_plan()
        session.run_update(plan=first_plan, cell=self._cell(29))
        object.__setattr__(
            self.routing.update_rows[0],
            "source_bytes",
            b"cpu fake routing mutation hidden by replacement",
        )
        object.__setattr__(self.routing, "routing_digest", "f" * 64)
        calls = []

        def substitute(_routing):
            calls.append("called")
            return session._opened_routing_live_fingerprint  # noqa: SLF001

        with mock.patch.object(
            trainer, "_routing_live_fingerprint", new=substitute
        ):
            with self.assertRaisesRegex(
                trainer.GraftPhaseAShortTrainingError,
                "live short-trainer execution runtime",
            ):
                session.next_update_plan()

        self.assertEqual(calls, [])
        self.assertEqual(session.phase, "failed")
        failure = session.failure_receipt()
        self.assertTrue(
            failure["trainable_parameters_restored_to_initial_snapshot"]
        )
        self.assertFalse(failure["checkpoint_written"])
        for name, parameter in self.names:
            self.assertTrue(torch.equal(parameter, initial[name]), name)

    def test_confirmation_helper_replacement_fails_before_call_and_rolls_back(
        self,
    ) -> None:
        initial = {
            name: parameter.detach().clone() for name, parameter in self.names
        }
        session = self._open()
        self._run_two_updates(session)
        plan = session.confirmation_plan()
        calls = []

        def substitute(**_fields):
            calls.append("called")
            return {
                "schedule_index": 29,
                "noncompensating_all_pass": True,
                "digest": "0" * 64,
            }

        same_non_tensor = object()
        with torch.no_grad(), mock.patch.object(
            trainer, "_confirmation_field_metrics", new=substitute
        ):
            with self.assertRaisesRegex(
                trainer.GraftPhaseAShortTrainingError,
                "live short-trainer execution runtime",
            ):
                session.record_confirmation_fields(
                    plan=plan,
                    schedule_index=29,
                    source_noop_target_velocity=same_non_tensor,
                    correct_atlas_noop_velocity=same_non_tensor,
                    wrong_atlas_noop_velocity=same_non_tensor,
                    dropped_atlas_noop_velocity=same_non_tensor,
                    correct_atlas_action_velocity=same_non_tensor,
                    dropped_atlas_action_velocity=same_non_tensor,
                )

        self.assertEqual(calls, [])
        self.assertEqual(session.phase, "failed")
        failure = session.failure_receipt()
        self.assertTrue(
            failure["trainable_parameters_restored_to_initial_snapshot"]
        )
        self.assertFalse(failure["checkpoint_written"])
        for name, parameter in self.names:
            self.assertTrue(torch.equal(parameter, initial[name]), name)

    def test_live_ctypes_reader_replacement_fails_before_call_and_rolls_back(
        self,
    ) -> None:
        for attribute in ("string_at", "_string_at"):
            with self.subTest(attribute=attribute):
                initial = {
                    name: parameter.detach().clone()
                    for name, parameter in self.names
                }
                session = self._open()
                first_plan = session.next_update_plan()
                session.run_update(plan=first_plan, cell=self._cell(29))
                second_plan = session.next_update_plan()
                second_cell = self._cell(38)
                calls = []

                def substitute(*_args, **_kwargs):
                    calls.append("called")
                    return b""

                with mock.patch.object(
                    trainer.ctypes,
                    attribute,
                    new=substitute,
                ):
                    with self.assertRaisesRegex(
                        trainer.GraftPhaseAShortTrainingError,
                        "live ctypes.string_at runtime",
                    ):
                        session.run_update(
                            plan=second_plan,
                            cell=second_cell,
                        )

                self.assertEqual(calls, [])
                self.assertEqual(session.phase, "failed")
                failure = session.failure_receipt()
                self.assertTrue(
                    failure["trainable_parameters_restored_to_initial_snapshot"]
                )
                self.assertFalse(failure["checkpoint_written"])
                for name, parameter in self.names:
                    self.assertTrue(torch.equal(parameter, initial[name]), name)

    def test_wrong_ctypes_source_fails_before_reader_call_and_rolls_back(
        self,
    ) -> None:
        initial = {
            name: parameter.detach().clone() for name, parameter in self.names
        }
        session = self._open()
        first_plan = session.next_update_plan()
        session.run_update(plan=first_plan, cell=self._cell(29))
        second_plan = session.next_update_plan()
        second_cell = self._cell(38)
        original_getsource = trainer.inspect.getsource
        original_reader = trainer.ctypes.string_at
        reader_calls = []

        def wrong_ctypes_source(value):
            if value is trainer._PINNED_CTYPES_STRING_AT:  # noqa: SLF001
                return "def string_at(ptr, size=-1):\n    return b'wrong'\n"
            return original_getsource(value)

        def reader_observer(*args, **kwargs):
            reader_calls.append((args, kwargs))
            return original_reader(*args, **kwargs)

        with mock.patch.object(
            trainer.inspect,
            "getsource",
            new=wrong_ctypes_source,
        ), mock.patch.object(
            trainer.ctypes,
            "string_at",
            new=reader_observer,
        ):
            with self.assertRaisesRegex(
                trainer.GraftPhaseAShortTrainingError,
                "live ctypes.string_at source differs",
            ):
                session.run_update(plan=second_plan, cell=second_cell)

        self.assertEqual(reader_calls, [])
        self.assertEqual(session.phase, "failed")
        failure = session.failure_receipt()
        self.assertTrue(
            failure["trainable_parameters_restored_to_initial_snapshot"]
        )
        self.assertFalse(failure["checkpoint_written"])
        for name, parameter in self.names:
            self.assertTrue(torch.equal(parameter, initial[name]), name)

    def test_tensor_digest_helper_replacement_fails_before_call_and_rolls_back(
        self,
    ) -> None:
        initial = {
            name: parameter.detach().clone() for name, parameter in self.names
        }
        session = self._open()
        first_plan = session.next_update_plan()
        session.run_update(plan=first_plan, cell=self._cell(29))
        second_plan = session.next_update_plan()
        second_cell = self._cell(38)
        calls = []

        def substitute(_value):
            calls.append("called")
            return "0" * 64

        with mock.patch.object(
            trainer,
            "_tensor_bytes_sha256",
            new=substitute,
        ):
            with self.assertRaisesRegex(
                trainer.GraftPhaseAShortTrainingError,
                "live short-trainer execution runtime",
            ):
                session.run_update(plan=second_plan, cell=second_cell)

        self.assertEqual(calls, [])
        self.assertEqual(session.phase, "failed")
        failure = session.failure_receipt()
        self.assertTrue(
            failure["trainable_parameters_restored_to_initial_snapshot"]
        )
        self.assertFalse(failure["checkpoint_written"])
        for name, parameter in self.names:
            self.assertTrue(torch.equal(parameter, initial[name]), name)

    def test_live_adamw_hyperparameter_mutation_fails_and_rolls_back(self) -> None:
        initial = {
            name: parameter.detach().clone() for name, parameter in self.names
        }
        session = self._open()
        plan = session.next_update_plan()
        session._optimizer.param_groups[0]["lr"] = 0.5  # noqa: SLF001
        with self.assertRaisesRegex(
            trainer.GraftPhaseAShortTrainingError,
            "AdamW parameter group",
        ):
            session.run_update(plan=plan, cell=self._cell(29))
        self.assertEqual(session.phase, "failed")
        failure = session.failure_receipt()
        self.assertFalse(failure["checkpoint_written"])
        self.assertTrue(
            failure["trainable_parameters_restored_to_initial_snapshot"]
        )
        for name, parameter in self.names:
            self.assertTrue(torch.equal(parameter, initial[name]), name)

    def test_live_adamw_state_mutation_fails_and_rolls_back(self) -> None:
        initial = {
            name: parameter.detach().clone() for name, parameter in self.names
        }
        session = self._open()
        first_plan = session.next_update_plan()
        session.run_update(plan=first_plan, cell=self._cell(29))
        second_plan = session.next_update_plan()
        first_parameter = self.names[0][1]
        session._optimizer.state[first_parameter]["exp_avg"].fill_(  # noqa: SLF001
            float("nan")
        )
        with self.assertRaisesRegex(
            trainer.GraftPhaseAShortTrainingError,
            "AdamW exp_avg state",
        ):
            session.run_update(plan=second_plan, cell=self._cell(38))
        self.assertEqual(session.phase, "failed")
        failure = session.failure_receipt()
        self.assertFalse(failure["checkpoint_written"])
        self.assertTrue(
            failure["trainable_parameters_restored_to_initial_snapshot"]
        )
        for name, parameter in self.names:
            self.assertTrue(torch.equal(parameter, initial[name]), name)

    def test_nested_update_receipt_mutation_is_detected_before_confirmation(self) -> None:
        session = self._open()
        first, _second = self._run_two_updates(session)
        plan = session.confirmation_plan()
        first["optimizer"]["learning_rate"] = 0.5
        with torch.no_grad(), self.assertRaisesRegex(
            trainer.GraftPhaseAShortTrainingError,
            "stored update receipt",
        ):
            session.record_confirmation_fields(
                plan=plan,
                schedule_index=29,
                **self._passing_confirmation_fields(),
            )
        self.assertEqual(session.phase, "failed")
        self.assertFalse(session.failure_receipt()["checkpoint_written"])

    def test_noncompensating_confirmation_gate_blocks_bad_relative_gain(self) -> None:
        session = self._open()
        self._run_two_updates(session)
        plan = session.confirmation_plan()
        fields = self._passing_confirmation_fields()
        fields["correct_atlas_noop_velocity"] = torch.tensor(
            [0.5, 0.0, 0.0, 0.0], dtype=torch.float32
        )
        with torch.no_grad(), self.assertRaisesRegex(
            trainer.GraftPhaseAShortTrainingError, "noncompensating gate"
        ):
            session.record_confirmation_fields(
                plan=plan, schedule_index=29, **fields
            )
        failure = session.failure_receipt()
        self.assertFalse(failure["checkpoint_written"])
        metrics = failure["failed_confirmation_metrics"]
        self.assertIsNotNone(metrics)
        self.assertFalse(metrics["noncompensating_all_pass"])
        self.assertFalse(
            metrics["noncompensating_gates"][
                "correct_vs_wrong_noop_relative_gain"
            ]
        )

    def test_noncompensating_confirmation_gate_blocks_negative_delta_cosine(self) -> None:
        session = self._open()
        self._run_two_updates(session)
        plan = session.confirmation_plan()
        fields = self._passing_confirmation_fields()
        fields["dropped_atlas_action_velocity"] = torch.tensor(
            [0.0, 0.0, 0.0, 0.0], dtype=torch.float32
        )
        with torch.no_grad(), self.assertRaisesRegex(
            trainer.GraftPhaseAShortTrainingError,
            "noncompensating gate",
        ):
            session.record_confirmation_fields(
                plan=plan, schedule_index=29, **fields
            )
        metrics = session.failure_receipt()["failed_confirmation_metrics"]
        self.assertFalse(
            metrics["noncompensating_gates"][
                "action_delta_correct_drop_cosine"
            ]
        )
        self.assertFalse(session.failure_receipt()["checkpoint_written"])

    def test_confirmation_norm_ratio_is_one_way_correct_over_drop(self) -> None:
        session = self._open()
        self._run_two_updates(session)
        plan = session.confirmation_plan()
        fields = self._passing_confirmation_fields()
        fields["correct_atlas_action_velocity"] = torch.tensor(
            [0.15, 0.0, 0.0, 0.0], dtype=torch.float32
        )
        with torch.no_grad(), self.assertRaisesRegex(
            trainer.GraftPhaseAShortTrainingError,
            "noncompensating gate",
        ):
            session.record_confirmation_fields(
                plan=plan, schedule_index=29, **fields
            )
        metrics = session.failure_receipt()["failed_confirmation_metrics"]
        ratio = float.fromhex(
            metrics["action_delta_correct_drop_norm_ratio_float64_hex"]
        )
        self.assertLess(ratio, 0.5)
        self.assertFalse(
            metrics["noncompensating_gates"][
                "action_delta_correct_drop_norm_ratio"
            ]
        )

    def test_confirmation_fields_require_detached_fp32_distinct_storage(self) -> None:
        session = self._open()
        self._run_two_updates(session)
        plan = session.confirmation_plan()
        fields = self._passing_confirmation_fields()
        fields["correct_atlas_action_velocity"] = torch.tensor(
            [0.3, 0.0, 0.0, 0.0],
            dtype=torch.float32,
            requires_grad=True,
        )
        with torch.no_grad(), self.assertRaisesRegex(
            trainer.GraftPhaseAShortTrainingError,
            "confirmation field contract",
        ):
            session.record_confirmation_fields(
                plan=plan, schedule_index=29, **fields
            )
        self.assertEqual(session.phase, "failed")
        self.assertFalse(session.failure_receipt()["checkpoint_written"])

    def test_dp_arm_one_wrong_owner_is_same_family_fit_row(self) -> None:
        self.backend = trainer.authenticate_cpu_test_collectives(rank=6)
        session = self._open()
        self._run_two_updates(session)
        plan = session.confirmation_plan()
        self.assertEqual(plan.row_iid, self.routing.confirmation_rows[1].iid)
        self.assertEqual(plan.wrong_owner_iid, self.routing.update_rows[1].iid)


if __name__ == "__main__":
    unittest.main()
