from __future__ import annotations

from dataclasses import replace
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
    import dclr_preference_batch as preference_batch  # noqa: E402
    import dclr_preference_objective as objective  # noqa: E402
    import dclr_runtime_contract as runtime_contract  # noqa: E402
else:  # pragma: no cover - dependency-light environments
    preference_batch = None
    objective = None
    runtime_contract = None


class DependencyLightSourceGuards(unittest.TestCase):
    def test_module_has_no_forward_optimizer_or_distributed_runtime(self) -> None:
        module_path = (
            Path(preference_batch.__file__)
            if preference_batch is not None
            else METHOD_ROOT / "dclr_preference_batch.py"
        )
        source = module_path.read_text(encoding="utf-8")
        self.assertNotIn("torch.distributed", source)
        self.assertNotIn("all_reduce(", source)
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn(".shared_step(", source)


if torch is not None:

    class _FakeTransformer:
        def __init__(self, dtype=torch.float32) -> None:
            self.dtype = dtype
            self.calls: list[dict[str, object]] = []

        def patch_vae_latent(self, latent, source_id=None):
            batch, channels, phases, height, width = latent.shape
            assert batch == 1 and channels == 16
            tokens = phases * (height // 2) * (width // 2)
            call_index = len(self.calls)
            # Distinct values make the exact [S,y+,S,y-] chunk order visible.
            token_value = float(call_index + 1)
            packed = torch.full(
                (1, tokens, runtime_contract.PINNED_INNER_DIM),
                token_value,
                dtype=self.dtype,
                device=latent.device,
            )
            # source_id=0 is the identity phase.  source_id=1 must differ.
            rope_value = complex(float(source_id), 1.0)
            rope = torch.full(
                (1, 1, tokens, runtime_contract.PINNED_ROPE_DIM),
                rope_value,
                dtype=torch.complex128,
                device=latent.device,
            )
            self.calls.append(
                {
                    "latent": latent,
                    "source_id": source_id,
                    "tokens": packed,
                    "rotary": rope,
                }
            )
            return packed, rope


@unittest.skipIf(torch is None, "torch is unavailable")
class PackedPreferenceBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        shape = (1, 16, 21, 4, 6)
        elements = 1
        for item in shape:
            elements *= item
        self.source = torch.linspace(
            -0.5, 0.5, elements, dtype=torch.float32
        ).reshape(shape)
        winner = torch.linspace(
            -1.0, 1.0, elements, dtype=torch.float32
        ).reshape(shape)
        loser = winner.flip(-1).contiguous() + 0.125
        epsilon = torch.linspace(
            1.0, -1.0, elements, dtype=torch.float32
        ).reshape(shape)
        sigma = torch.tensor([0.375], dtype=torch.float32)
        self.state = objective.build_shared_pair_flow_state(
            winner, loser, epsilon, sigma
        )

    def build(self, *, dtype=None):
        if dtype is None:
            dtype = torch.float32
        transformer = _FakeTransformer(dtype=dtype)
        batch = preference_batch.build_packed_preference_batch(
            transformer,
            normalized_source=self.source,
            flow_state=self.state,
        )
        return transformer, batch

    def test_exact_official_physical_and_logical_packing(self) -> None:
        transformer, batch = self.build(dtype=torch.bfloat16)
        n = 21 * 2 * 3

        self.assertEqual([call["source_id"] for call in transformer.calls], [1, 0, 0])
        self.assertEqual(len(transformer.calls), 3)
        for call in transformer.calls:
            self.assertEqual(call["latent"].dtype, torch.bfloat16)
        self.assertEqual(
            tuple(batch.noisy_latents.shape),
            (1, 4 * n, runtime_contract.PINNED_INNER_DIM),
        )
        self.assertEqual(batch.noisy_latents.dtype, torch.bfloat16)
        self.assertEqual(
            tuple(batch.rotary_embs.shape),
            (1, 1, 4 * n, runtime_contract.PINNED_ROPE_DIM),
        )
        self.assertEqual(batch.rotary_embs.dtype, torch.complex128)
        self.assertEqual(batch.batch_vae_seqlen, (2 * n, 2 * n))
        self.assertEqual(batch.timesteps.shape, (2,))
        self.assertTrue(torch.equal(batch.timesteps[:1], self.state.timestep))
        self.assertTrue(torch.equal(batch.timesteps[1:], self.state.timestep))
        self.assertEqual(
            batch.logical_self_attention_cu_seqlens.tolist(),
            [0, 2 * n, 4 * n],
        )
        self.assertEqual(
            batch.logical_self_attention_cu_seqlens.dtype, torch.int32
        )

        source_tokens = transformer.calls[0]["tokens"]
        winner_tokens = transformer.calls[1]["tokens"]
        loser_tokens = transformer.calls[2]["tokens"]
        self.assertTrue(torch.equal(batch.noisy_latents[:, :n], source_tokens))
        self.assertTrue(
            torch.equal(batch.noisy_latents[:, n : 2 * n], winner_tokens)
        )
        self.assertTrue(
            torch.equal(batch.noisy_latents[:, 2 * n : 3 * n], source_tokens)
        )
        self.assertTrue(
            torch.equal(batch.noisy_latents[:, 3 * n :], loser_tokens)
        )
        self.assertEqual(
            batch.target_selector.tolist(),
            ([False] * n + [True] * n) * 2,
        )
        self.assertEqual(
            batch.candidate_target_selector.tolist(),
            [False] * n + [True] * n,
        )
        self.assertEqual(
            tuple(batch.target_true_velocity.shape),
            (1, 2 * n, runtime_contract.PINNED_PATCH_DIM),
        )
        self.assertEqual(batch.target_true_velocity.dtype, torch.float32)
        expected_velocity = torch.cat(
            (
                preference_batch.pack_spatial_velocity(
                    self.state.winner_true_velocity
                ),
                preference_batch.pack_spatial_velocity(
                    self.state.loser_true_velocity
                ),
            ),
            dim=1,
        )
        self.assertTrue(torch.equal(batch.target_true_velocity, expected_velocity))

    def test_one_literal_epsilon_and_exact_sigma_timestep_evidence(self) -> None:
        _, batch = self.build()
        self.assertIs(batch.winner_epsilon, self.state.epsilon)
        self.assertIs(batch.loser_epsilon, self.state.epsilon)
        self.assertIs(batch.winner_epsilon, batch.loser_epsilon)
        self.assertEqual(batch.sigma_float32_bits_hex, "3ec00000")
        self.assertEqual(batch.timestep_float32_bits_hex, "43bb8000")
        kwargs = preference_batch.shared_step_visual_kwargs(batch)
        self.assertIs(kwargs["noisy_latents"], batch.noisy_latents)
        self.assertIs(kwargs["timesteps"], batch.timesteps)
        self.assertIs(kwargs["rotary_embs"], batch.rotary_embs)
        self.assertEqual(kwargs["batch_vae_seqlen"], [252, 252])
        self.assertNotIn("cu_seqlens_q_cache", kwargs)

    def test_velocity_patch_order_is_pt_ph_pw_then_channels(self) -> None:
        value = torch.arange(
            16 * 21 * 2 * 2, dtype=torch.float32
        ).reshape(1, 16, 21, 2, 2)
        packed = preference_batch.pack_spatial_velocity(value)
        self.assertEqual(tuple(packed.shape), (1, 21, 64))
        expected_first = value[0, :, 0].permute(1, 2, 0).reshape(-1)
        self.assertTrue(torch.equal(packed[0, 0], expected_first))

    def test_rejects_bad_source_and_forged_flow_state(self) -> None:
        transformer = _FakeTransformer()
        with self.assertRaisesRegex(
            preference_batch.DCLRPreferenceBatchError, "exact FP32"
        ):
            preference_batch.build_packed_preference_batch(
                transformer,
                normalized_source=self.source.bfloat16(),
                flow_state=self.state,
            )
        with self.assertRaisesRegex(
            preference_batch.DCLRPreferenceBatchError, "positive even H/W"
        ):
            preference_batch.build_packed_preference_batch(
                transformer,
                normalized_source=self.source[:, :, :, :, :5],
                flow_state=self.state,
            )
        forged_time = replace(
            self.state,
            timestep=self.state.timestep + torch.tensor([1.0]),
        )
        with self.assertRaisesRegex(
            preference_batch.DCLRPreferenceBatchError, r"1000\*sigma"
        ):
            preference_batch.build_packed_preference_batch(
                transformer,
                normalized_source=self.source,
                flow_state=forged_time,
            )
        forged_noise_use = replace(
            self.state,
            loser_x_sigma=self.state.loser_x_sigma + 0.25,
        )
        with self.assertRaisesRegex(
            preference_batch.DCLRPreferenceBatchError, "shared epsilon/sigma"
        ):
            preference_batch.build_packed_preference_batch(
                transformer,
                normalized_source=self.source,
                flow_state=forged_noise_use,
            )

    def test_rejects_transformer_and_patch_contract_drift(self) -> None:
        class NoPatch:
            dtype = torch.float32

        with self.assertRaisesRegex(
            preference_batch.DCLRPreferenceBatchError, "patch_vae_latent"
        ):
            preference_batch.build_packed_preference_batch(
                NoPatch(),
                normalized_source=self.source,
                flow_state=self.state,
            )

        class WrongRope(_FakeTransformer):
            def patch_vae_latent(self, latent, source_id=None):
                tokens, rope = super().patch_vae_latent(latent, source_id)
                return tokens, rope.to(dtype=torch.complex64)

        with self.assertRaisesRegex(
            preference_batch.DCLRPreferenceBatchError, "complex128"
        ):
            preference_batch.build_packed_preference_batch(
                WrongRope(),
                normalized_source=self.source,
                flow_state=self.state,
            )

        class NoSourceIdRope(_FakeTransformer):
            def patch_vae_latent(self, latent, source_id=None):
                tokens, rope = super().patch_vae_latent(latent, source_id)
                return tokens, torch.ones_like(rope)

        with self.assertRaisesRegex(
            preference_batch.DCLRPreferenceBatchError, "source-id rotary"
        ):
            preference_batch.build_packed_preference_batch(
                NoSourceIdRope(),
                normalized_source=self.source,
                flow_state=self.state,
            )

        class TrainablePatch(_FakeTransformer):
            def patch_vae_latent(self, latent, source_id=None):
                tokens, rope = super().patch_vae_latent(latent, source_id)
                return tokens.requires_grad_(), rope

        with self.assertRaisesRegex(
            preference_batch.DCLRPreferenceBatchError, "detached transformer-dtype"
        ):
            preference_batch.build_packed_preference_batch(
                TrainablePatch(),
                normalized_source=self.source,
                flow_state=self.state,
            )

    def test_revalidation_fails_closed_on_packing_tamper(self) -> None:
        _, batch = self.build()
        n = batch.source_token_count
        cases = (
            (
                replace(batch, batch_vae_seqlen=(n, n)),
                r"\[2N,2N\]",
            ),
            (
                replace(batch, timesteps=batch.timesteps + torch.tensor([0.0, 1.0])),
                "same timestep bits",
            ),
            (
                replace(batch, target_selector=torch.ones_like(batch.target_selector)),
                r"0N\+1N\+0N\+1N",
            ),
            (
                replace(
                    batch,
                    logical_self_attention_cu_seqlens=torch.tensor(
                        [0, 4 * n], dtype=torch.int32
                    ),
                ),
                "isolate",
            ),
            (
                replace(
                    batch,
                    noisy_latents=torch.cat(
                        (
                            batch.noisy_latents[:, : 2 * n],
                            torch.zeros_like(batch.noisy_latents[:, 2 * n : 3 * n]),
                            batch.noisy_latents[:, 3 * n :],
                        ),
                        dim=1,
                    ),
                ),
                "same source-token prefix",
            ),
            (
                replace(batch, sigma_float32_bits_hex="00000000"),
                "bit evidence",
            ),
        )
        for corrupted, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                    preference_batch.DCLRPreferenceBatchError, message
                ):
                    preference_batch.validate_packed_preference_batch(corrupted)


if __name__ == "__main__":
    unittest.main()
