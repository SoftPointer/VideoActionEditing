from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import inspect
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
    from torch import nn

    import caper_sigma_gated_target_row_lora as caper
    import train_lora as legacy_train

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    caper = None  # type: ignore[assignment]
    legacy_train = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class _Attention(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.to_q = nn.Linear(hidden, hidden, bias=True)
            self.to_k = nn.Linear(hidden, hidden, bias=True)
            self.to_v = nn.Linear(hidden, hidden, bias=True)
            self.to_out = nn.ModuleList(
                [nn.Linear(hidden, hidden, bias=True), nn.Identity()]
            )


    class _Block(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.attn1 = _Attention(hidden)
            self.attn2 = _Attention(hidden)
            self.ffn = nn.Sequential(
                nn.Linear(hidden, hidden * 2),
                nn.SiLU(),
                nn.Linear(hidden * 2, hidden),
            )


    class _Transformer(nn.Module):
        def __init__(self, hidden: int = 8) -> None:
            super().__init__()
            self.patch_embedding = nn.Conv3d(
                16, hidden, kernel_size=(1, 2, 2)
            )
            self.blocks = nn.ModuleList(
                [_Block(hidden) for _ in range(caper.TOTAL_BLOCKS_1P3B)]
            )
            self.proj_out = nn.Linear(hidden, 16)

        def patch_vae_latent(self, value: torch.Tensor, source_id: float = 0.0):
            del source_id
            return value, value


    class _Diffusion(nn.Module):
        def __init__(self, transformer: nn.Module) -> None:
            super().__init__()
            self.transformer = transformer
            self.transformer_2 = None


    class _Renderer(nn.Module):
        def __init__(self, hidden: int = 8) -> None:
            super().__init__()
            self.diff_dec = _Diffusion(_Transformer(hidden))
            # Trainable-leakage auditing must cover parameters outside diff_dec.
            self.text_stub = nn.Linear(hidden, hidden)


def _sigma(index: int) -> float:
    return caper.sigma_strata.PINNED_POSITIVE_SIGMAS[index]


class _ParallelAuthority:
    def __init__(self, state: "caper.CAPERParallelState") -> None:
        self.state = state
        self.calls = 0

    def snapshot(self) -> "caper.CAPERParallelState":
        self.calls += 1
        return self.state


def _authority(*, rank: int = 0, size: int = 1) -> _ParallelAuthority:
    if size == 1:
        state = caper.CAPERParallelState(
            world_size=1,
            world_rank=0,
            sequence_parallel_group_ranks=(0,),
            sequence_parallel_rank=0,
            authority_id=caper.SP1_TEST_AUTHORITY_ID,
            test_only=True,
        )
    elif size == 4:
        state = caper.CAPERParallelState(
            world_size=4,
            world_rank=rank,
            sequence_parallel_group_ranks=(0, 1, 2, 3),
            sequence_parallel_rank=rank,
            authority_id="caper-world4-sp4-logic-oracle",
            test_only=True,
        )
    else:  # tests may exercise the constructor's rejection directly
        raise AssertionError("test helper supports only SP1/SP4")
    return _ParallelAuthority(state)


def _route(
    index: int,
    *,
    rank: int = 0,
    size: int = 1,
    source_tokens: int = 2,
    target_tokens: int = 3,
    enabled: bool = True,
) -> "caper.CAPERRoute":
    selector = caper.preference_pack_target_selector(
        source_tokens=source_tokens, target_tokens=target_tokens
    )
    segments = caper.preference_pack_segments(
        source_tokens=source_tokens, target_tokens=target_tokens
    )
    return caper.CAPERRoute.from_runtime_sigma(
        global_target_selector=selector,
        pack_segments=segments,
        parallel_state_authority=_authority(rank=rank, size=size),
        sigma_schedule_index=index,
        sigma=_sigma(index),
        enabled=enabled,
    )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class CAPERRouteTests(unittest.TestCase):
    def test_live_snapshot_reads_torch_distributed_and_bernini_state(self) -> None:
        state_type = type("ParallelState", (), {"__module__": "bernini.parallel"})
        group = object()
        state = state_type()
        state.ulysses_enabled = True
        state.ulysses_group = group
        state.ulysses_size = 4
        state.ulysses_rank = 2
        state.world_size = 8
        state.rank = 6
        bernini_parallel = SimpleNamespace(get_parallel_state=lambda: state)

        def gather(rows, value, *, group):
            self.assertEqual(value, 6)
            rows[:] = [4, 5, 6, 7]

        with mock.patch.object(
            caper.importlib, "import_module", return_value=bernini_parallel
        ), mock.patch.object(
            torch.distributed, "is_available", return_value=True
        ), mock.patch.object(
            torch.distributed, "is_initialized", return_value=True
        ), mock.patch.object(
                torch.distributed,
                "get_world_size",
                side_effect=lambda selected=None: 8 if selected is None else 4,
        ), mock.patch.object(
                torch.distributed,
                "get_rank",
                side_effect=lambda selected=None: 6 if selected is None else 2,
        ), mock.patch.object(
            torch.distributed, "all_gather_object", side_effect=gather
        ):
            snapshot = caper.snapshot_live_bernini_parallel_state()
        self.assertEqual(snapshot.world_size, 8)
        self.assertEqual(snapshot.world_rank, 6)
        self.assertEqual(snapshot.sequence_parallel_group_ranks, (4, 5, 6, 7))
        self.assertEqual(snapshot.sequence_parallel_rank, 2)
        self.assertFalse(snapshot.test_only)

    def test_exact_runtime_sigma_is_bound_to_explicit_gate(self) -> None:
        self.assertEqual(_route(0).gate_name, "high")
        self.assertEqual(_route(0).gate_weight, 1.0)
        self.assertEqual(_route(33).gate_name, "mid")
        self.assertEqual(_route(33).gate_weight, 0.5)
        self.assertEqual(_route(38).gate_name, "low_base_only")
        self.assertEqual(_route(38).gate_weight, 0.0)
        with self.assertRaisesRegex(caper.CAPERContractError, "pinned schedule"):
            caper.CAPERRoute.from_runtime_sigma(
                global_target_selector=caper.preference_pack_target_selector(
                    source_tokens=2, target_tokens=3
                ),
                pack_segments=caper.preference_pack_segments(
                    source_tokens=2, target_tokens=3
                ),
                parallel_state_authority=_authority(),
                sigma_schedule_index=0,
                sigma=_sigma(1),
            )

    def test_real_preference_pack_selector_and_receipts_are_bound(self) -> None:
        route = _route(0)
        expected = (False, False, True, True, True, False, False, True, True, True)
        self.assertEqual(route.global_target_selector, expected)
        receipt = dict(route.receipt())
        digest = receipt.pop("digest")
        self.assertEqual(digest, caper.object_sha256(receipt))
        pack = dict(receipt["preference_pack_receipt"])
        pack_digest = pack.pop("digest")
        self.assertEqual(pack_digest, caper.object_sha256(pack))
        self.assertEqual(pack["layout"], "[S,y+,S,y-]")
        self.assertEqual(pack["target_intervals"], 2)
        self.assertEqual(
            pack["target_selector_sha256"],
            caper.target_selector_sha256(expected),
        )
        self.assertEqual(
            receipt["target_selector_sha256"],
            pack["target_selector_sha256"],
        )
        parallel = dict(receipt["parallel_state_receipt"])
        parallel_digest = parallel.pop("digest")
        self.assertEqual(parallel_digest, caper.object_sha256(parallel))
        self.assertEqual(parallel["sequence_parallel_group_ranks"], [0])
        self.assertEqual(parallel["authority_id"], caper.SP1_TEST_AUTHORITY_ID)

    def test_sp4_selector_appends_false_padding_and_slices_contiguously(self) -> None:
        selectors = [
            _route(0, rank=rank, size=4).local_target_selector(
                device=torch.device("cpu")
            )
            for rank in range(4)
        ]
        joined = torch.cat(selectors)
        self.assertEqual(joined.numel(), 12)
        self.assertTrue(
            torch.equal(
                joined[:10],
                torch.tensor(
                    [False, False, True, True, True, False, False, True, True, True]
                ),
            )
        )
        self.assertFalse(bool(joined[10:].any()))
        self.assertTrue(
            torch.equal(
                selectors[0], torch.tensor([False, False, True])
            )
        )
        self.assertTrue(torch.equal(selectors[1], torch.tensor([True, True, False])))
        self.assertTrue(torch.equal(selectors[2], torch.tensor([False, True, True])))
        self.assertTrue(torch.equal(selectors[3], torch.tensor([True, False, False])))

    def test_route_requires_authority_and_has_no_raw_rank_size_api(self) -> None:
        parameters = inspect.signature(
            caper.CAPERRoute.from_runtime_sigma
        ).parameters
        self.assertNotIn("sequence_parallel_rank", parameters)
        self.assertNotIn("sequence_parallel_size", parameters)
        selector = caper.preference_pack_target_selector(
            source_tokens=2, target_tokens=3
        )
        segments = caper.preference_pack_segments(
            source_tokens=2, target_tokens=3
        )
        with self.assertRaisesRegex(caper.CAPERContractError, "authority is required"):
            caper.CAPERRoute.from_runtime_sigma(
                global_target_selector=selector,
                pack_segments=segments,
                sigma_schedule_index=0,
                sigma=_sigma(0),
            )

    def test_sp1_requires_explicit_unit_test_authority(self) -> None:
        with self.assertRaisesRegex(caper.CAPERContractError, "SP1"):
            caper.CAPERParallelState(
                world_size=1,
                world_rank=0,
                sequence_parallel_group_ranks=(0,),
                sequence_parallel_rank=0,
                authority_id="caller-self-report",
                test_only=False,
            )

    def test_parallel_snapshot_validates_group_size_and_rank_membership(self) -> None:
        with self.assertRaisesRegex(caper.CAPERContractError, "SP rank"):
            caper.CAPERParallelState(
                world_size=4,
                world_rank=1,
                sequence_parallel_group_ranks=(0, 1, 2, 3),
                sequence_parallel_rank=0,
                authority_id="bad-membership-oracle",
                test_only=True,
            )
        with self.assertRaisesRegex(caper.CAPERContractError, "SP1.*SP4"):
            caper.CAPERParallelState(
                world_size=4,
                world_rank=0,
                sequence_parallel_group_ranks=(0, 1),
                sequence_parallel_rank=0,
                authority_id="bad-size-oracle",
                test_only=True,
            )

    def test_authority_snapshot_is_typed_and_called_once(self) -> None:
        authority = _authority(rank=2, size=4)
        selector = caper.preference_pack_target_selector(
            source_tokens=2, target_tokens=3
        )
        route = caper.CAPERRoute.from_runtime_sigma(
            global_target_selector=selector,
            pack_segments=caper.preference_pack_segments(
                source_tokens=2, target_tokens=3
            ),
            parallel_state_authority=authority,
            sigma_schedule_index=0,
            sigma=_sigma(0),
        )
        self.assertEqual(authority.calls, 1)
        self.assertEqual(route.sequence_parallel_rank, 2)
        self.assertEqual(route.sequence_parallel_size, 4)
        with self.assertRaisesRegex(caper.CAPERContractError, "untyped"):
            caper.CAPERRoute.from_runtime_sigma(
                global_target_selector=selector,
                pack_segments=caper.preference_pack_segments(
                    source_tokens=2, target_tokens=3
                ),
                parallel_state_authority=lambda: {
                    "sequence_parallel_rank": 0,
                    "sequence_parallel_size": 4,
                },
                sigma_schedule_index=0,
                sigma=_sigma(0),
            )

    def test_selector_must_match_canonical_pack_receipt(self) -> None:
        selector = list(
            caper.preference_pack_target_selector(
                source_tokens=2, target_tokens=3
            )
        )
        selector[5] = True
        with self.assertRaisesRegex(caper.CAPERContractError, "not bound"):
            caper.CAPERRoute.from_runtime_sigma(
                global_target_selector=selector,
                pack_segments=caper.preference_pack_segments(
                    source_tokens=2, target_tokens=3
                ),
                parallel_state_authority=_authority(),
                sigma_schedule_index=0,
                sigma=_sigma(0),
            )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class CAPERAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2209)
        self.renderer = _Renderer(hidden=8)
        self.renderer.requires_grad_(False)
        self.transformer = self.renderer.diff_dec.transformer
        self.original_patch = self.transformer.patch_embedding
        self.original_proj_out = self.transformer.proj_out
        self.original_attn1 = tuple(block.attn1 for block in self.transformer.blocks)
        self.original_k = tuple(block.attn2.to_k for block in self.transformer.blocks)
        self.original_v = tuple(block.attn2.to_v for block in self.transformer.blocks)
        self.original_ffn = tuple(block.ffn for block in self.transformer.blocks)
        self.original_q = tuple(block.attn2.to_q for block in self.transformer.blocks)
        self.original_o = tuple(
            block.attn2.to_out[0] for block in self.transformer.blocks
        )
        self.handle = caper.install_caper_capacity_probe(
            self.renderer, expected_hidden_size=8
        )

    def tearDown(self) -> None:
        if not self.handle.restored:
            # A deliberate leakage/mutation test may make assert_scope fail.
            for index, original in enumerate(self.original_q):
                self.transformer.blocks[index].attn2.to_q = original
            for index, original in enumerate(self.original_o):
                self.transformer.blocks[index].attn2.to_out[0] = original
            self.handle.restored = True

    @staticmethod
    def _make_nonzero(wrapper: "caper.CAPERTargetRowLoRA") -> None:
        with torch.no_grad():
            wrapper.caper_lora_A.weight.fill_(0.25)
            wrapper.caper_lora_B.weight.fill_(0.5)

    def test_source_rows_are_byte_exact_and_only_targets_change(self) -> None:
        route = _route(0)
        hidden = torch.ones(1, route.local_length, 8)
        selector = route.local_target_selector(device=torch.device("cpu"))
        for wrapper, original in (
            (self.handle.q_wrappers[0][1], self.original_q[0]),
            (self.handle.o_wrappers[0][1], self.original_o[0]),
        ):
            self._make_nonzero(wrapper)
            expected = original(hidden)
            with self.handle.route(_route(0)):
                actual = wrapper(hidden)
            self.assertTrue(torch.equal(actual[:, ~selector], expected[:, ~selector]))
            self.assertGreater(
                float((actual[:, selector] - expected[:, selector]).abs().sum()),
                0.0,
            )

    def test_world4_sp4_projection_oracle_covers_source_target_and_padding(self) -> None:
        target_rows = 0
        exact_non_target_rows = 0
        padding_rows = 0
        for wrapper, original in (
            (self.handle.q_wrappers[0][1], self.original_q[0]),
            (self.handle.o_wrappers[0][1], self.original_o[0]),
        ):
            self._make_nonzero(wrapper)
            for rank in range(4):
                route = _route(0, rank=rank, size=4)
                selector = route.local_target_selector(device=torch.device("cpu"))
                hidden = torch.full(
                    (1, route.local_length, 8), float(rank + 1)
                )
                expected = original(hidden)
                with self.handle.route(route):
                    actual = wrapper(hidden)
                start = rank * route.local_length
                for local_index, is_target in enumerate(selector.tolist()):
                    global_index = start + local_index
                    if is_target:
                        target_rows += 1
                        self.assertGreater(
                            float(
                                (
                                    actual[:, local_index, :]
                                    - expected[:, local_index, :]
                                )
                                .abs()
                                .sum()
                            ),
                            0.0,
                        )
                    else:
                        exact_non_target_rows += 1
                        self.assertTrue(
                            torch.equal(
                                actual[:, local_index, :],
                                expected[:, local_index, :],
                            )
                        )
                        if global_index >= route.total_tokens:
                            padding_rows += 1
        # Two wrapped projection families, each with six targets, four source
        # rows, and two append-padding rows in the WORLD4/SP4 pack.
        self.assertEqual(target_rows, 12)
        self.assertEqual(exact_non_target_rows, 12)
        self.assertEqual(padding_rows, 4)

    def test_low_sigma_is_direct_exact_base_and_has_no_lora_graph(self) -> None:
        wrapper = self.handle.q_wrappers[0][1]
        # NaNs prove this is a direct return, not delta * 0.
        with torch.no_grad():
            wrapper.caper_lora_A.weight.fill_(float("nan"))
            wrapper.caper_lora_B.weight.fill_(float("nan"))
        hidden = torch.randn(1, _route(38).local_length, 8, requires_grad=True)
        expected = self.original_q[0](hidden)
        with self.handle.route(_route(38)):
            actual = wrapper(hidden)
        self.assertTrue(torch.equal(actual, expected))
        actual.sum().backward()
        self.assertIsNone(wrapper.caper_lora_A.weight.grad)
        self.assertIsNone(wrapper.caper_lora_B.weight.grad)

    def test_gradients_and_requires_grad_are_only_on_allowed_a_b(self) -> None:
        route = _route(0)
        hidden = torch.ones(1, route.local_length, 8)
        selector = route.local_target_selector(device=torch.device("cpu"))
        for _, wrapper in (*self.handle.q_wrappers, *self.handle.o_wrappers):
            self._make_nonzero(wrapper)
        with self.handle.route(_route(0)):
            loss = sum(
                wrapper(hidden)[:, selector].sum()
                for _, wrapper in (*self.handle.q_wrappers, *self.handle.o_wrappers)
            )
        loss.backward()

        allowed = dict(self.handle.trainable_named_parameters())
        observed_trainable = {
            name: parameter
            for name, parameter in self.renderer.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(set(observed_trainable), set(allowed))
        self.assertTrue(
            all(
                name.endswith(".caper_lora_A.weight")
                or name.endswith(".caper_lora_B.weight")
                for name in allowed
            )
        )
        self.assertTrue(
            all(
                parameter.grad is not None
                and float(parameter.grad.abs().sum()) > 0.0
                for parameter in allowed.values()
            )
        )
        allowed_ids = {id(parameter) for parameter in allowed.values()}
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in self.renderer.parameters()
                if id(parameter) not in allowed_ids
            )
        )

    def test_allowlist_freeze_and_checksum_certificate(self) -> None:
        certificate = dict(self.handle.freeze_checksum_certificate())
        digest = certificate.pop("digest")
        self.assertEqual(digest, caper.object_sha256(certificate))
        self.assertEqual(certificate["target_module_count"], 60)
        self.assertEqual(
            certificate["target_modules"], list(caper.CAPER_TARGET_MODULES)
        )
        self.assertEqual(
            certificate["target_modules_sha256"],
            caper.CAPER_TARGET_MODULES_SHA256,
        )
        self.assertEqual(certificate["trainable_parameter_tensor_count"], 120)
        self.assertEqual(certificate["trainable_parameter_elements"], 60 * 2 * 8 * 8)
        self.assertTrue(certificate["trainable_parameter_values_hashed"])
        self.assertEqual(
            certificate["trainable_parameter_initial_values_sha256"],
            certificate["trainable_parameter_current_values_sha256"],
        )
        self.assertTrue(certificate["default_all_30_blocks"])
        self.assertTrue(certificate["frozen_transformer_byte_exact"])
        self.assertTrue(
            certificate["wrapped_projection_source_and_padding_rows_byte_exact"]
        )
        self.assertEqual(
            certificate["source_exactness_scope"],
            "wrapped_attn2_q_and_o_projection_output_only",
        )
        self.assertFalse(
            certificate["global_source_activation_byte_exact_after_joint_attn1"]
        )
        self.assertIn(
            "joint attn1", certificate["global_source_activation_disclaimer"]
        )
        self.assertFalse(certificate["key_value_trainable"])
        self.assertFalse(certificate["attn1_trainable"])
        self.assertFalse(certificate["ffn_trainable"])
        self.assertFalse(certificate["patch_embedding_trainable"])
        self.assertFalse(certificate["proj_out_trainable"])
        self.assertTrue(certificate["gradient_checkpointing_must_be_disabled"])
        self.assertTrue(certificate["route_context_must_cover_forward_and_backward"])
        self.assertTrue(
            certificate["every_unregistered_transformer_module_identity_exact"]
        )

        self.assertIs(self.transformer.patch_embedding, self.original_patch)
        self.assertIs(self.transformer.proj_out, self.original_proj_out)
        for index, block in enumerate(self.transformer.blocks):
            self.assertIs(block.attn1, self.original_attn1[index])
            self.assertIs(block.attn2.to_k, self.original_k[index])
            self.assertIs(block.attn2.to_v, self.original_v[index])
            self.assertIs(block.ffn, self.original_ffn[index])

    def test_certificate_hashes_a_b_values_not_only_names(self) -> None:
        before = self.handle.freeze_checksum_certificate()
        self.assertEqual(
            self.handle.trainable_parameter_values_sha256(),
            before["trainable_parameter_current_values_sha256"],
        )
        with torch.no_grad():
            self.handle.q_wrappers[0][1].caper_lora_B.weight[0, 0].add_(1.0)
        after = self.handle.freeze_checksum_certificate()
        self.assertEqual(
            self.handle.trainable_parameter_values_sha256(),
            after["trainable_parameter_current_values_sha256"],
        )
        self.assertEqual(
            before["trainable_parameter_names_sha256"],
            after["trainable_parameter_names_sha256"],
        )
        self.assertEqual(
            before["trainable_parameter_initial_values_sha256"],
            after["trainable_parameter_initial_values_sha256"],
        )
        self.assertNotEqual(
            before["trainable_parameter_current_values_sha256"],
            after["trainable_parameter_current_values_sha256"],
        )
        self.assertEqual(
            before["frozen_transformer_current_sha256"],
            after["frozen_transformer_current_sha256"],
        )

    def test_certificate_detects_frozen_base_mutation(self) -> None:
        with torch.no_grad():
            self.original_k[0].weight.add_(1.0)
        with self.assertRaisesRegex(caper.CAPERContractError, "checksum changed"):
            self.handle.freeze_checksum_certificate()

    def test_scope_detects_trainable_leakage(self) -> None:
        self.original_k[0].weight.requires_grad_(True)
        with self.assertRaisesRegex(caper.CAPERContractError, "became trainable"):
            self.handle.assert_scope()

    def test_scope_detects_parameterless_forbidden_module_replacement(self) -> None:
        self.transformer.blocks[0].ffn[1] = nn.ReLU()
        with self.assertRaisesRegex(caper.CAPERContractError, "protected frozen module"):
            self.handle.assert_scope()

    def test_restore_recovers_exact_original_q_o_modules(self) -> None:
        self.handle.restore()
        for index, block in enumerate(self.transformer.blocks):
            self.assertIs(block.attn2.to_q, self.original_q[index])
            self.assertIs(block.attn2.to_out[0], self.original_o[index])


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class CAPERFailClosedInstallTests(unittest.TestCase):
    def test_rejects_unfrozen_renderer(self) -> None:
        with self.assertRaisesRegex(caper.CAPERContractError, "freeze"):
            caper.install_caper_capacity_probe(
                _Renderer(hidden=8), expected_hidden_size=8
            )

    def test_rejects_target_module_alias(self) -> None:
        renderer = _Renderer(hidden=8)
        shared = renderer.diff_dec.transformer.blocks[0].attn2.to_q
        renderer.diff_dec.transformer.blocks[1].attn2.to_q = shared
        renderer.requires_grad_(False)
        with self.assertRaisesRegex(caper.CAPERContractError, "alias"):
            caper.install_caper_capacity_probe(
                renderer, expected_hidden_size=8
            )

    def test_rejects_unknown_target_wrapper_or_shape(self) -> None:
        renderer = _Renderer(hidden=8)
        renderer.diff_dec.transformer.blocks[0].attn2.to_q = nn.Sequential(
            nn.Linear(8, 8)
        )
        renderer.requires_grad_(False)
        with self.assertRaisesRegex(caper.CAPERContractError, "structure differs"):
            caper.install_caper_capacity_probe(
                renderer, expected_hidden_size=8
            )

    def test_rejects_gradient_checkpointing(self) -> None:
        renderer = _Renderer(hidden=8)
        renderer.diff_dec.transformer.gradient_checkpointing = True
        renderer.requires_grad_(False)
        with self.assertRaisesRegex(caper.CAPERContractError, "checkpointing"):
            caper.install_caper_capacity_probe(
                renderer, expected_hidden_size=8
            )

    def test_default_allowlist_excludes_every_forbidden_family(self) -> None:
        names = caper.canonical_target_module_names()
        self.assertEqual(len(names), 60)
        self.assertTrue(all(".attn2." in name for name in names))
        self.assertTrue(
            all(name.endswith(".to_q") or name.endswith(".to_out.0") for name in names)
        )
        forbidden = (".to_k", ".to_v", ".attn1.", ".ffn", "patch_embedding", "proj_out")
        self.assertTrue(all(not any(token in name for token in forbidden) for name in names))

        # Bind CAPER's namespace to the same fully-qualified discovery contract
        # used by the existing Bernini train_lora harness.
        discovered = legacy_train.select_attention_projection_names(
            _Renderer(hidden=8)
        )
        legacy_cross_qo = {
            name
            for name in discovered
            if ".attn2." in name
            and (name.endswith(".to_q") or name.endswith(".to_out.0"))
        }
        self.assertEqual(legacy_cross_qo, set(names))


if __name__ == "__main__":
    unittest.main()
