from __future__ import annotations

from contextlib import contextmanager
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
    import mace_candidate_action_energy as mace  # noqa: E402
    import pair_v5_action_adapter as action_adapter  # noqa: E402
    import pair_v5_native_bridge as bridge  # noqa: E402
    import pair_v5_phase_conjunctive_energy as phase_energy  # noqa: E402
    import source_self_native_ref_contrastive_v3 as native  # noqa: E402
else:  # pragma: no cover - dependency-light environments
    mace = None
    action_adapter = None
    bridge = None
    phase_energy = None
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
            self.patch_source_ids: list[float] = []

        def patch_vae_latent(self, value, *, source_id):
            self.patch_source_ids.append(float(source_id))
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


    class _Diffusion(torch.nn.Module):
        def __init__(
            self,
            transformer: _Transformer,
            *,
            trainable: bool,
            initial_gain: float = 1.0,
            observe_action_routes: bool = False,
        ) -> None:
            super().__init__()
            self.transformer = transformer
            self.gain = torch.nn.Parameter(
                torch.tensor(initial_gain, dtype=torch.float32),
                requires_grad=trainable,
            )
            self.calls: list[dict[str, object]] = []
            self.observe_action_routes = observe_action_routes

        def shared_step(self, **kwargs):
            route = (
                action_adapter.active_route()
                if self.observe_action_routes
                else None
            )
            self.calls.append(
                {
                    "tokens_id": id(kwargs["noisy_latents"]),
                    "timestep_id": id(kwargs["timesteps"]),
                    "timestep": float(kwargs["timesteps"].item()),
                    "batch_vae_seqlen": tuple(kwargs["batch_vae_seqlen"]),
                    "schedule_index": (
                        None if route is None else route.sigma_schedule_index
                    ),
                    "route_enabled": None if route is None else route.enabled,
                    "branch_name": None if route is None else route.branch_name,
                }
            )
            total = int(kwargs["noisy_latents"].shape[1])
            scalar = kwargs["cond_embeds"][0, 0, 0].float() * self.gain
            return torch.ones(
                1,
                total,
                64,
                dtype=torch.float32,
                device=kwargs["noisy_latents"].device,
            ) * scalar


    def _prompts():
        return {branch: f"registered {branch} prompt" for branch in mace.BRANCH_ORDER}


    def _condition(value: float) -> torch.Tensor:
        # A stride-zero view keeps this focused CPU test small while retaining
        # the real pinned [1,512,4096] Bernini text geometry.
        return torch.tensor(value, dtype=torch.float32).reshape(1, 1, 1).expand(
            1, 512, 4096
        )


    def _conditions():
        return {
            branch: _condition(float(index + 1))
            for index, branch in enumerate(mace.BRANCH_ORDER)
        }


    def _phase_commitment():
        weights = {
            milestone: [1.0 / 21.0] * 21
            for milestone in phase_energy.MILESTONE_ORDER
        }
        return phase_energy.make_phase_weight_commitment(weights)


@unittest.skipIf(torch is None, "torch is unavailable")
class FrozenT2VBridgeTests(unittest.TestCase):
    def _scorer(self):
        transformer = _Transformer()
        diffusion = _Diffusion(transformer, trainable=False)
        diffusion.eval()
        scorer = bridge.FrozenBerniniT2VScorer(
            diffusion,
            transformer,
            _prompts(),
            _conditions(),
            frozen_model_receipt_digest="a" * 64,
        )
        return scorer, transformer, diffusion

    def test_one_internal_state_ten_prompts_then_phase_conjunction(self) -> None:
        scorer, transformer, diffusion = self._scorer()
        clean = torch.zeros(1, 16, 21, 2, 2, dtype=torch.float32)
        epsilon = torch.ones_like(clean)
        sigma = torch.tensor(0.35, dtype=torch.float32)
        commitment = _phase_commitment()
        result = bridge.score_frozen_t2v_action_energy(
            clean,
            epsilon,
            sigma,
            _prompts(),
            scorer,
            commitment,
            registered_phase_weight_digest=commitment["registration_digest"],
        )

        self.assertEqual(
            tuple(result.branch_velocities.shape),
            (len(mace.BRANCH_ORDER), 1, 16, 21, 2, 2),
        )
        self.assertEqual(transformer.patch_source_ids, [0.0])
        self.assertEqual(len(diffusion.calls), len(mace.BRANCH_ORDER))
        self.assertEqual(len({row["tokens_id"] for row in diffusion.calls}), 1)
        self.assertEqual(len({row["timestep_id"] for row in diffusion.calls}), 1)
        self.assertEqual({row["timestep"] for row in diffusion.calls}, {350.0})
        self.assertEqual(
            {row["batch_vae_seqlen"] for row in diffusion.calls}, {(21,)}
        )
        torch.testing.assert_close(
            result.branch_velocities[0], torch.ones_like(clean)
        )
        self.assertGreater(float(result.energy.reward.item()), 0.0)
        self.assertGreater(float(result.phase_energy.reward.item()), 0.0)
        self.assertTrue(
            torch.equal(result.energy.x_sigma, result.phase_energy.x_sigma)
        )
        self.assertTrue(result.receipt["phase_conjunction_applied_inside_native_bridge"])
        self.assertTrue(
            result.receipt["velocity_energy_serial_closure_bit_exact"]
        )
        self.assertTrue(
            result.receipt["velocity_energy_batched_closure_verified"]
        )
        self.assertTrue(
            result.receipt["phase_global_energy_closure_verified"]
        )
        self.assertLessEqual(
            result.receipt["velocity_energy_closure_max_abs_error"],
            result.receipt["velocity_energy_closure_atol"],
        )
        self.assertLessEqual(
            result.receipt["phase_global_energy_closure_max_abs_error"],
            result.receipt["phase_global_energy_closure_atol"],
        )
        self.assertEqual(
            result.receipt["velocity_energy_closure_rtol"],
            bridge.VELOCITY_ENERGY_CLOSURE_RTOL,
        )
        unsigned = dict(result.receipt)
        digest = unsigned.pop("digest")
        self.assertEqual(digest, bridge.object_sha256(unsigned))
        packet = dict(scorer.last_packet_receipt)
        packet_digest = packet.pop("digest")
        self.assertEqual(packet_digest, bridge.object_sha256(packet))

    def test_prompt_and_candidate_state_substitution_fail_closed(self) -> None:
        scorer, _, _ = self._scorer()
        clean = torch.zeros(1, 16, 21, 2, 2, dtype=torch.float32)
        epsilon = torch.ones_like(clean)
        sigma = torch.tensor([0.35], dtype=torch.float32)
        bad_prompts = _prompts()
        bad_prompts["reverse"] = "a different reverse prompt"
        commitment = _phase_commitment()
        with self.assertRaisesRegex(
            bridge.PairV5NativeBridgeError, "registry differs"
        ):
            bridge.score_frozen_t2v_action_energy(
                clean,
                epsilon,
                sigma,
                bad_prompts,
                scorer,
                commitment,
                registered_phase_weight_digest=commitment["registration_digest"],
            )

        x_sigma = (1.0 - sigma) * clean + sigma * epsilon
        scorer(x_sigma, sigma, _prompts()["action"])
        with self.assertRaisesRegex(
            bridge.PairV5NativeBridgeError, "same x_sigma/sigma objects"
        ):
            scorer(x_sigma.clone(), sigma, _prompts()["noop"])
        scorer.abort_packet()

        with self.assertRaisesRegex(
            bridge.PairV5NativeBridgeError, "pre-registered digest"
        ):
            bridge.score_frozen_t2v_action_energy(
                clean,
                epsilon,
                sigma,
                _prompts(),
                scorer,
                commitment,
                registered_phase_weight_digest="0" * 64,
            )

    def test_frozen_scorer_rejects_trainable_model_and_extra_branch(self) -> None:
        transformer = _Transformer()
        diffusion = _Diffusion(transformer, trainable=True)
        with self.assertRaisesRegex(
            bridge.PairV5NativeBridgeError, "trainable parameters"
        ):
            bridge.FrozenBerniniT2VScorer(
                diffusion,
                transformer,
                _prompts(),
                _conditions(),
                frozen_model_receipt_digest="a" * 64,
            )
        prompts = _prompts()
        prompts["proposal"] = "forbidden proposal prompt"
        frozen = _Diffusion(transformer, trainable=False)
        with self.assertRaisesRegex(
            bridge.PairV5NativeBridgeError, "closure differs"
        ):
            bridge.FrozenBerniniT2VScorer(
                frozen,
                transformer,
                prompts,
                _conditions(),
                frozen_model_receipt_digest="a" * 64,
            )

    def test_text_condition_registry_mutation_fails_before_model_call(self) -> None:
        scorer, _, diffusion = self._scorer()
        clean = torch.zeros(1, 16, 21, 2, 2, dtype=torch.float32)
        epsilon = torch.ones_like(clean)
        sigma = torch.tensor([0.35], dtype=torch.float32)
        with torch.no_grad():
            scorer._condition_0.fill_(77.0)
        with self.assertRaisesRegex(
            bridge.PairV5NativeBridgeError, "registry differs from its seal"
        ):
            bridge.score_frozen_t2v_action_energy(
                clean,
                epsilon,
                sigma,
                _prompts(),
                scorer,
                _phase_commitment(),
                registered_phase_weight_digest=_phase_commitment()[
                    "registration_digest"
                ],
            )
        self.assertEqual(diffusion.calls, [])


@unittest.skipIf(torch is None, "torch is unavailable")
class NativeRV2V4PolicyPairTests(unittest.TestCase):
    def _inputs(self):
        video = torch.zeros(1, 16, 21, 2, 2, dtype=torch.float32)
        refs = tuple(
            torch.full((1, 16, 1, 2, 2), float(index), dtype=torch.float32)
            for index in range(4)
        )
        state = torch.full_like(video, 0.5)
        index = 20
        sigma = torch.tensor(native.NATIVE_UNIPC40_SIGMAS[index], dtype=torch.float32)
        timestep = torch.tensor(
            float(native.NATIVE_UNIPC40_TIMESTEPS[index]), dtype=torch.float32
        )
        return video, refs, state, index, sigma, timestep

    def test_separate_student_reference_exact_rv2v4_velocities(self) -> None:
        student_transformer = _Transformer()
        reference_transformer = _Transformer()
        student = _Diffusion(student_transformer, trainable=True)
        reference = _Diffusion(
            reference_transformer, trainable=False, initial_gain=1.5
        )
        reference.eval()
        reference_transformer.eval()
        video, refs, state, index, sigma, timestep = self._inputs()
        result = bridge.forward_native_rv2v4_policy_pair(
            student,
            student_transformer,
            reference,
            reference_transformer,
            video,
            refs,
            state,
            sigma,
            timestep,
            _condition(1.0),
            _condition(0.25),
            sequence_parallel_rank=0,
            sequence_parallel_size=1,
        )
        self.assertEqual(tuple(result.student_velocity.shape), tuple(state.shape))
        self.assertEqual(tuple(result.reference_velocity.shape), tuple(state.shape))
        self.assertTrue(result.student_velocity.requires_grad)
        self.assertFalse(result.reference_velocity.requires_grad)
        self.assertEqual(
            student_transformer.patch_source_ids,
            list(native.PATCH_CALL_SOURCE_IDS),
        )
        self.assertEqual(
            reference_transformer.patch_source_ids,
            list(native.PATCH_CALL_SOURCE_IDS),
        )
        self.assertEqual(result.receipt["reference_count"], 4)
        self.assertEqual(result.receipt["sigma_schedule_index"], index)
        self.assertEqual(
            result.receipt["expanded_guidance_coefficients_hex"],
            {
                name: float(value).hex()
                for name, value in bridge.EXPANDED_GUIDANCE_COEFFICIENTS.items()
            },
        )
        unsigned = dict(result.receipt)
        digest = unsigned.pop("digest")
        self.assertEqual(digest, bridge.object_sha256(unsigned))
        result.student_velocity.mean().backward()
        self.assertIsNotNone(student.gain.grad)

    def test_shared_action_adapter_routes_one_schedule_index_and_disables_reference(self) -> None:
        transformer = _Transformer()
        diffusion = _Diffusion(
            transformer, trainable=True, observe_action_routes=True
        )
        handle = object.__new__(action_adapter.PairV5ActionAdapterHandle)
        handle.transformer = transformer
        handle.q_wrappers = ()
        handle.o_wrappers = ()
        handle.original_q = ()
        handle.original_o = ()
        handle.original_patch_embedding_id = id(transformer.patch_vae_latent)
        handle.original_self_attention_ids = ()
        handle.restored = False
        video, refs, state, index, sigma, timestep = self._inputs()
        result = bridge.forward_native_rv2v4_policy_pair(
            diffusion,
            transformer,
            diffusion,
            transformer,
            video,
            refs,
            state,
            sigma,
            timestep,
            _condition(1.0),
            _condition(0.25),
            student_adapter=handle,
            reference_adapter=handle,
            sequence_parallel_rank=0,
            sequence_parallel_size=1,
        )
        self.assertEqual(len(diffusion.calls), 8)
        self.assertEqual(
            {row["schedule_index"] for row in diffusion.calls}, {index}
        )
        self.assertEqual(
            [row["route_enabled"] for row in diffusion.calls],
            [True] * 4 + [False] * 4,
        )
        self.assertEqual(
            [row["branch_name"] for row in diffusion.calls],
            ["none", "V", "VI", "VI", "none", "V", "VI", "VI"],
        )
        self.assertTrue(result.receipt["student_adapter_route_enabled"])
        self.assertTrue(
            result.receipt["reference_adapter_route_explicitly_disabled"]
        )

    def test_low_sigma_action_adapter_returns_intentional_frozen_base(self) -> None:
        transformer = _Transformer()
        diffusion = _Diffusion(
            transformer, trainable=False, observe_action_routes=True
        )
        diffusion.eval()
        transformer.eval()
        handle = object.__new__(action_adapter.PairV5ActionAdapterHandle)
        handle.transformer = transformer
        handle.q_wrappers = ()
        handle.o_wrappers = ()
        handle.original_q = ()
        handle.original_o = ()
        handle.original_patch_embedding_id = id(transformer.patch_vae_latent)
        handle.original_self_attention_ids = ()
        handle.restored = False
        video, refs, state, _, _, _ = self._inputs()
        index = 38
        sigma = torch.tensor(
            native.NATIVE_UNIPC40_SIGMAS[index], dtype=torch.float32
        )
        timestep = torch.tensor(
            float(native.NATIVE_UNIPC40_TIMESTEPS[index]), dtype=torch.float32
        )
        result = bridge.forward_native_rv2v4_policy_pair(
            diffusion,
            transformer,
            diffusion,
            transformer,
            video,
            refs,
            state,
            sigma,
            timestep,
            _condition(1.0),
            _condition(0.25),
            student_adapter=handle,
            reference_adapter=handle,
            sequence_parallel_rank=0,
            sequence_parallel_size=1,
        )

        self.assertFalse(result.student_velocity.requires_grad)
        self.assertFalse(result.reference_velocity.requires_grad)
        torch.testing.assert_close(
            result.student_velocity, result.reference_velocity
        )
        self.assertEqual(
            {row["schedule_index"] for row in diffusion.calls}, {index}
        )
        self.assertEqual(
            [row["route_enabled"] for row in diffusion.calls],
            [True] * 4 + [False] * 4,
        )
        self.assertEqual(result.receipt["student_action_adapter_gate"], "low_base_only")
        self.assertEqual(
            result.receipt["student_action_adapter_gate_weight_hex"],
            float(0.0).hex(),
        )
        self.assertFalse(result.receipt["student_adapter_route_enabled"])
        self.assertFalse(result.receipt["student_prediction_trainable"])

    def test_schedule_pair_four_refs_and_shared_policy_fail_closed(self) -> None:
        student_transformer = _Transformer()
        reference_transformer = _Transformer()
        student = _Diffusion(student_transformer, trainable=True)
        reference = _Diffusion(reference_transformer, trainable=False)
        reference.eval()
        video, refs, state, _, sigma, timestep = self._inputs()
        with self.assertRaisesRegex(
            bridge.PairV5NativeBridgeError, "same native exact40 coordinate"
        ):
            bridge.forward_native_rv2v4_policy_pair(
                student,
                student_transformer,
                reference,
                reference_transformer,
                video,
                refs,
                state,
                torch.tensor(0.5, dtype=torch.float32),
                timestep,
                _condition(1.0),
                _condition(0.25),
                sequence_parallel_rank=0,
                sequence_parallel_size=1,
            )
        with self.assertRaisesRegex(
            bridge.PairV5NativeBridgeError, "exactly four"
        ):
            bridge.forward_native_rv2v4_policy_pair(
                student,
                student_transformer,
                reference,
                reference_transformer,
                video,
                refs[:3],
                state,
                sigma,
                timestep,
                _condition(1.0),
                _condition(0.25),
                sequence_parallel_rank=0,
                sequence_parallel_size=1,
            )
        with self.assertRaisesRegex(
            bridge.PairV5NativeBridgeError, "explicit adapter handle"
        ):
            bridge.forward_native_rv2v4_policy_pair(
                student,
                student_transformer,
                student,
                student_transformer,
                video,
                refs,
                state,
                sigma,
                timestep,
                _condition(1.0),
                _condition(0.25),
                sequence_parallel_rank=0,
                sequence_parallel_size=1,
            )


@unittest.skipIf(torch is None, "torch is unavailable")
class ReceiptAndAPITests(unittest.TestCase):
    def test_static_contract_digest_geometry_and_no_privileged_scorer_api(self) -> None:
        receipt = dict(bridge.bridge_contract_receipt())
        digest = receipt.pop("digest")
        self.assertEqual(digest, bridge.object_sha256(receipt))
        self.assertEqual(receipt["frame_count"], 81)
        self.assertEqual(receipt["rv2v_reference_count"], 4)
        self.assertEqual(receipt["rv2v_reference_frame_indices"], [0, 27, 53, 80])
        self.assertEqual(receipt["mace_branch_order"], list(mace.BRANCH_ORDER))
        self.assertFalse(receipt["proposal_visual_data_consumed"])
        self.assertTrue(
            receipt["t2v_phase_energy_handoff"].startswith("inside_bridge")
        )
        for callable_value in (
            bridge.FrozenBerniniT2VScorer.__init__,
            bridge.FrozenBerniniT2VScorer.forward,
            bridge.score_frozen_t2v_action_energy,
        ):
            parameters = set(inspect.signature(callable_value).parameters)
            self.assertTrue(
                parameters.isdisjoint(bridge.FORBIDDEN_SCORER_INPUT_NAMES)
            )


if __name__ == "__main__":
    unittest.main()
