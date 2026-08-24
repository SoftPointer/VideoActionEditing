from __future__ import annotations

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
    import dclr_runtime_contract as contract  # noqa: E402
else:  # pragma: no cover - dependency-light environments
    contract = None


class DependencyLightSourceGuards(unittest.TestCase):
    def test_runtime_contract_contains_no_distributed_collective(self) -> None:
        source = (METHOD_ROOT / "dclr_runtime_contract.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("import torch.distributed", source)
        self.assertNotIn("all_reduce(", source)
        self.assertNotIn("reduce_scatter(", source)


@unittest.skipIf(torch is None, "torch is unavailable")
class SigmaAndGeometryTests(unittest.TestCase):
    def _tokens(self, value: float, count: int = 3) -> torch.Tensor:
        return torch.full(
            (1, count, contract.PINNED_INNER_DIM),
            value,
            dtype=torch.float32,
        )

    def _rope(self, count: int = 3) -> torch.Tensor:
        return torch.ones(
            (1, 1, count, contract.PINNED_ROPE_DIM),
            dtype=torch.complex128,
        )

    def test_fp32_sigma_maps_directly_without_shift_or_grid_snap(self) -> None:
        sigma = torch.tensor([0.0, 0.1234, 0.5, 1.0], dtype=torch.float32)
        timestep = contract.fp32_sigma_to_timestep(sigma)
        self.assertEqual(timestep.dtype, torch.float32)
        self.assertTrue(torch.equal(timestep, sigma * 1000.0))

        with self.assertRaisesRegex(
            contract.DCLRRuntimeContractError, "FP32"
        ):
            contract.fp32_sigma_to_timestep(sigma.double())
        with self.assertRaisesRegex(
            contract.DCLRRuntimeContractError, r"\[0, 1\]"
        ):
            contract.fp32_sigma_to_timestep(torch.tensor([1.01]))
        with self.assertRaisesRegex(
            contract.DCLRRuntimeContractError, "detached"
        ):
            contract.fp32_sigma_to_timestep(
                torch.tensor([0.5], requires_grad=True)
            )

    def test_t2v_and_mv2v_use_target_only_and_source_target_geometry(self) -> None:
        target = self._tokens(7.0)
        target_rope = self._rope()
        source = self._tokens(2.0)
        source_rope = self._rope()
        t2v = contract.build_t2v_target_branch(target, target_rope)
        mv2v = contract.build_mv2v_target_tail_branch(
            source, source_rope, target, target_rope
        )

        self.assertIs(t2v.noisy_latents, target)
        self.assertEqual(t2v.batch_vae_seqlen, (3,))
        self.assertTrue(t2v.target_selector.all().item())
        self.assertEqual(mv2v.batch_vae_seqlen, (6,))
        self.assertFalse(mv2v.target_selector[:3].any().item())
        self.assertTrue(mv2v.target_selector[3:].all().item())
        self.assertTrue(torch.equal(mv2v.noisy_latents[:, 3:], target))
        self.assertTrue(
            torch.equal(mv2v.rotary_embs[:, :, 3:], target_rope)
        )
        kwargs = contract.shared_step_visual_kwargs(mv2v)
        self.assertEqual(kwargs["batch_vae_seqlen"], [6])
        self.assertNotEqual(kwargs["batch_vae_seqlen"], [3, 3])

        timestep = contract.fp32_sigma_to_timestep(
            torch.tensor([0.5], dtype=torch.float32)
        )
        self.assertEqual(
            contract.validate_cross_mode_target_tail(
                t2v,
                mv2v,
                t2v_timestep=timestep,
                mv2v_timestep=timestep.clone(),
            ),
            3,
        )

    def test_cross_mode_parity_rejects_changed_target_or_timestep(self) -> None:
        target = self._tokens(7.0)
        rope = self._rope()
        source = self._tokens(2.0)
        t2v = contract.build_t2v_target_branch(target, rope)
        changed_target = target.clone()
        changed_target[0, 0, 0] += 1.0
        mv2v = contract.build_mv2v_target_tail_branch(
            source, rope, changed_target, rope
        )
        timestep = torch.tensor([500.0], dtype=torch.float32)
        with self.assertRaisesRegex(
            contract.DCLRRuntimeContractError, "target tail"
        ):
            contract.validate_cross_mode_target_tail(
                t2v,
                mv2v,
                t2v_timestep=timestep,
                mv2v_timestep=timestep,
            )

        valid_mv2v = contract.build_mv2v_target_tail_branch(
            source, rope, target, rope
        )
        with self.assertRaisesRegex(
            contract.DCLRRuntimeContractError, "physical-sigma timestep"
        ):
            contract.validate_cross_mode_target_tail(
                t2v,
                valid_mv2v,
                t2v_timestep=timestep,
                mv2v_timestep=torch.tensor([501.0], dtype=torch.float32),
            )

    def test_mv2v_builder_rejects_nonmatching_source_geometry_and_ids(self) -> None:
        target = self._tokens(7.0)
        rope = self._rope()
        with self.assertRaisesRegex(
            contract.DCLRRuntimeContractError, "equal source and target"
        ):
            contract.build_mv2v_target_tail_branch(
                self._tokens(2.0, count=2), self._rope(count=2), target, rope
            )
        with self.assertRaisesRegex(
            contract.DCLRRuntimeContractError, "source_id=1"
        ):
            contract.build_mv2v_target_tail_branch(
                self._tokens(2.0), rope, target, rope, source_id=2
            )


@unittest.skipIf(torch is None, "torch is unavailable")
class CorrectDecoyAndEnergyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.count = 2
        self.target = torch.full(
            (1, self.count, contract.PINNED_INNER_DIM),
            5.0,
            dtype=torch.float32,
        )
        self.rope = torch.ones(
            (1, 1, self.count, contract.PINNED_ROPE_DIM),
            dtype=torch.complex128,
        )
        self.correct = contract.build_mv2v_target_tail_branch(
            torch.zeros_like(self.target),
            self.rope,
            self.target,
            self.rope,
        )
        self.decoy = contract.build_mv2v_target_tail_branch(
            torch.ones_like(self.target),
            self.rope,
            self.target,
            self.rope,
        )
        self.timestep = torch.tensor([375.0], dtype=torch.float32)
        self.condition = torch.zeros(
            (
                1,
                contract.PINNED_TEXT_TOKENS,
                contract.PINNED_TEXT_DIM,
            ),
            dtype=torch.float16,
        )

    def _validate(self, **overrides: object) -> int:
        values: dict[str, object] = {
            "correct": self.correct,
            "decoy": self.decoy,
            "correct_timestep": self.timestep,
            "decoy_timestep": self.timestep.clone(),
            "correct_cond_embeds": self.condition,
            "decoy_cond_embeds": self.condition,
            "correct_text_seqlen": [contract.PINNED_TEXT_TOKENS],
            "decoy_text_seqlen": (contract.PINNED_TEXT_TOKENS,),
        }
        values.update(overrides)
        return contract.validate_correct_decoy_same_state(**values)

    def test_correct_decoy_changes_only_source_prefix_content(self) -> None:
        self.assertEqual(self._validate(), self.count)

    def test_correct_decoy_rejects_same_source_changed_target_rope_or_text(self) -> None:
        with self.assertRaisesRegex(
            contract.DCLRRuntimeContractError, "identical"
        ):
            self._validate(decoy=self.correct)

        changed_rope = self.rope.clone()
        changed_rope[:, :, 0, 0] *= 1j
        changed_decoy = contract.build_mv2v_target_tail_branch(
            torch.ones_like(self.target),
            changed_rope,
            self.target,
            self.rope,
        )
        with self.assertRaisesRegex(
            contract.DCLRRuntimeContractError, "rotary differs"
        ):
            self._validate(decoy=changed_decoy)

        changed_condition = self.condition.clone()
        changed_condition[0, 0, 0] = 1.0
        with self.assertRaisesRegex(
            contract.DCLRRuntimeContractError, "action condition differs"
        ):
            self._validate(decoy_cond_embeds=changed_condition)

    def test_target_only_fp32_mse_ignores_source_prefix(self) -> None:
        full_prediction = torch.empty(
            (1, 2 * self.count, contract.PINNED_PATCH_DIM),
            dtype=torch.float16,
        )
        full_prediction[:, : self.count, :] = 1000.0
        full_prediction[:, self.count :, :] = 2.0
        target = torch.ones(
            (1, self.count, contract.PINNED_PATCH_DIM),
            dtype=torch.float32,
        )
        energy = contract.target_only_fp32_mse(
            full_prediction, target, self.correct.target_selector
        )
        self.assertEqual(energy.dtype, torch.float32)
        self.assertEqual(energy.ndim, 0)
        self.assertEqual(float(energy.item()), 1.0)

    def test_target_only_mse_rejects_non_tail_and_non_fp32_target(self) -> None:
        prediction = torch.zeros(
            (1, 4, contract.PINNED_PATCH_DIM), dtype=torch.float32
        )
        target = torch.zeros(
            (1, 2, contract.PINNED_PATCH_DIM), dtype=torch.float32
        )
        non_tail = torch.tensor([False, True, False, True])
        with self.assertRaisesRegex(
            contract.DCLRRuntimeContractError, "contiguous target tail"
        ):
            contract.target_only_fp32_mse(prediction, target, non_tail)
        with self.assertRaisesRegex(
            contract.DCLRRuntimeContractError, "must be FP32"
        ):
            contract.target_only_fp32_mse(
                prediction,
                target.half(),
                torch.tensor([False, False, True, True]),
            )


@unittest.skipIf(torch is None, "torch is unavailable")
class SequenceParallelReplicationTests(unittest.TestCase):
    def test_exact_replicas_return_the_original_local_scalar_without_reduction(self) -> None:
        local = torch.tensor(1.25, dtype=torch.float32)
        gathered = tuple(local.clone() for _ in range(4))
        result = contract.assert_sp_replicated_scalar(
            local, gathered, expected_world_size=4
        )
        self.assertIs(result, local)
        tensor_result = contract.assert_sp_replicated_scalar(
            local,
            torch.tensor([1.25, 1.25, 1.25, 1.25], dtype=torch.float32),
            expected_world_size=4,
        )
        self.assertIs(tensor_result, local)

    def test_replication_rejects_bit_mismatch_wrong_count_and_wrong_dtype(self) -> None:
        local = torch.tensor(1.25, dtype=torch.float32)
        with self.assertRaisesRegex(
            contract.DCLRRuntimeContractError, "exact replica"
        ):
            contract.assert_sp_replicated_scalar(
                local,
                [
                    torch.tensor(1.25, dtype=torch.float32),
                    torch.tensor(1.2501, dtype=torch.float32),
                ],
            )
        with self.assertRaisesRegex(
            contract.DCLRRuntimeContractError, "world size"
        ):
            contract.assert_sp_replicated_scalar(
                local, [local.clone(), local.clone()], expected_world_size=4
            )
        with self.assertRaisesRegex(
            contract.DCLRRuntimeContractError, "FP32 scalar"
        ):
            contract.assert_sp_replicated_scalar(
                local.double(), [local.double()]
            )


if __name__ == "__main__":
    unittest.main()
