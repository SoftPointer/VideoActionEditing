from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
    from torch import nn
    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    import cage_branch_locked_action_adapter as cage
    import cage_source_boundary_lock as boundary
else:
    cage = None  # type: ignore[assignment]
    boundary = None  # type: ignore[assignment]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _key(
    label: str, *, sigma_index: int = 0
) -> "boundary.CAGESameStateActionKey":
    return boundary.CAGESameStateActionKey(
        state_sha256=_sha("state:" + label),
        source_sha256=_sha("source:" + label),
        action_prompt_sha256=_sha("action:" + label),
        sigma_schedule_index=sigma_index,
    )


def _route(
    *,
    rank: int = 0,
    size: int = 1,
    total: int = 7,
    condition: int = 3,
    guidance_row: str = "VI_cond",
    sigma_index: int = 0,
) -> "cage.CAGEBranchLockedRoute":
    return cage.CAGEBranchLockedRoute(
        total_tokens=total,
        condition_tokens=condition,
        sequence_parallel_rank=rank,
        sequence_parallel_size=size,
        guidance_row=guidance_row,
        sigma_schedule_index=sigma_index,
    )


if _TORCH_AVAILABLE:
    class _ToyBlock(nn.Module):
        """Row-local block plus route-only leakage into every row."""

        def __init__(self, hidden: int, index: int) -> None:
            super().__init__()
            self.scale = nn.Parameter(
                torch.full((hidden,), 1.05 + 0.05 * index)
            )
            self.action_delta = nn.Parameter(
                torch.full((hidden,), 0.20 + 0.03 * index)
            )
            self.register_buffer(
                "base_offset", torch.full((hidden,), 0.01 * (index + 1))
            )

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            result = hidden * self.scale + self.base_offset
            if cage.active_route() is not None:
                # This deliberately simulates action information mixed into
                # source/padding rows by a joint transformer block.
                result = result + self.action_delta
            return result


    class _ToyTransformer(nn.Module):
        def __init__(self, *, blocks: int = 3, hidden: int = 4) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [_ToyBlock(hidden, index) for index in range(blocks)]
            )

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                hidden = block(hidden)
            return hidden


def _cleanup(handle: "boundary.CAGESourceBoundaryLockHandle") -> None:
    if handle.restored:
        return
    if handle.cache_bank.poisoned or not handle.cache_bank.cache_released:
        handle.discard_cache()
    handle.restore()


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class CAGESourceBoundaryLockTests(unittest.TestCase):
    def test_source_and_padding_are_base_exact_but_target_keeps_graph(self) -> None:
        model = _ToyTransformer()
        handle = boundary.install_cage_source_boundary_lock(
            model, selected_block_indices=(1, 2)
        )
        route = _route()
        key = _key("sp1")
        state = torch.arange(28, dtype=torch.float32).reshape(1, 7, 4) / 17
        selector = route.local_target_selector(device=state.device)
        try:
            with handle.capture_base(key=key, route=route):
                with torch.no_grad():
                    base = model(state)

            with torch.no_grad(), cage.activate_route(route):
                unlocked = model(state)
            self.assertFalse(
                cage.tensors_byte_exact(
                    unlocked[:, ~selector, :], base[:, ~selector, :]
                )
            )

            student_state = state.detach().clone().requires_grad_(True)
            with handle.lock_student(key=key, route=route):
                with cage.activate_route(route):
                    locked = model(student_state)

            self.assertTrue(
                cage.tensors_byte_exact(
                    locked[:, ~selector, :], base[:, ~selector, :]
                )
            )
            self.assertTrue(
                cage.tensors_byte_exact(
                    locked[:, selector, :], unlocked[:, selector, :]
                )
            )
            locked.sum().backward()
            self.assertTrue(
                torch.equal(
                    student_state.grad[:, ~selector, :],
                    torch.zeros_like(student_state.grad[:, ~selector, :]),
                )
            )
            self.assertTrue(
                bool(torch.all(student_state.grad[:, selector, :] != 0))
            )
            for block in model.blocks:
                self.assertIsNotNone(block.action_delta.grad)
                self.assertTrue(bool(torch.all(block.action_delta.grad != 0)))

            self.assertTrue(handle.cache_bank.cache_released)
            self.assertEqual(handle.cache_bank.completed_pair_count, 1)
            receipt = handle.receipt()
            self.assertFalse(
                receipt["integration"]["global_source_memory_lock_claim"]
            )
            forward_contract = receipt["training_inference_forward_contract"]
            self.assertEqual(forward_contract["native_guidance_forward_count"], 4)
            self.assertEqual(forward_contract["total_denoiser_forward_count"], 5)
            self.assertEqual(
                forward_contract["denoiser_forward_compute_multiplier"], 1.25
            )
            unsigned = dict(receipt)
            digest = unsigned.pop("digest")
            self.assertEqual(digest, boundary.object_sha256(unsigned))
        finally:
            _cleanup(handle)
        self.assertTrue(handle.restored)
        self.assertTrue(all(not block._forward_hooks for block in model.blocks))

    def test_native_sp4_selector_locks_source_and_append_padding_on_each_rank(
        self,
    ) -> None:
        for rank in range(4):
            with self.subTest(rank=rank):
                model = _ToyTransformer(blocks=1)
                handle = boundary.install_cage_source_boundary_lock(
                    model, selected_block_indices=(0,)
                )
                route = _route(
                    rank=rank, size=4, total=13, condition=5
                )
                key = _key("sp4-rank-" + str(rank))
                state = (
                    torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
                    + 100 * rank
                ) / 19
                selector = route.local_target_selector(device=state.device)
                try:
                    with handle.capture_base(key=key, route=route):
                        with torch.no_grad():
                            base = model(state)
                    with torch.no_grad(), cage.activate_route(route):
                        unlocked = model(state)
                    student_state = state.detach().clone().requires_grad_(True)
                    with handle.lock_student(key=key, route=route):
                        with cage.activate_route(route):
                            locked = model(student_state)

                    self.assertTrue(
                        cage.tensors_byte_exact(
                            locked[:, ~selector, :], base[:, ~selector, :]
                        )
                    )
                    self.assertTrue(
                        cage.tensors_byte_exact(
                            locked[:, selector, :], unlocked[:, selector, :]
                        )
                    )
                    locked.sum().backward()
                    self.assertTrue(
                        torch.equal(
                            student_state.grad[:, ~selector, :],
                            torch.zeros_like(student_state.grad[:, ~selector, :]),
                        )
                    )
                    if bool(selector.any()):
                        self.assertTrue(
                            bool(torch.all(student_state.grad[:, selector, :] != 0))
                        )
                    else:
                        self.assertTrue(
                            torch.equal(
                                student_state.grad,
                                torch.zeros_like(student_state.grad),
                            )
                        )
                    if rank == 3:
                        # Global token 12 is target; 13..15 are append padding.
                        self.assertEqual(selector.tolist(), [True, False, False, False])
                finally:
                    _cleanup(handle)

    def test_cache_is_complete_create_once_consume_once_and_fail_closed(self) -> None:
        route = _route()
        state = torch.ones(1, route.local_length, 4)

        model = _ToyTransformer(blocks=2)
        handle = boundary.install_cage_source_boundary_lock(
            model, selected_block_indices=(0, 1)
        )
        try:
            with self.assertRaisesRegex(
                boundary.CAGESourceBoundaryLockError, "missed selected blocks"
            ):
                with handle.capture_base(key=_key("missing"), route=route):
                    with torch.no_grad():
                        model.blocks[0](state)
            self.assertTrue(handle.cache_bank.poisoned)
            self.assertTrue(handle.cache_bank.cache_released)
            self.assertIsNone(boundary.active_invocation())
        finally:
            _cleanup(handle)

        model = _ToyTransformer(blocks=1)
        handle = boundary.install_cage_source_boundary_lock(
            model, selected_block_indices=(0,)
        )
        try:
            with self.assertRaisesRegex(
                boundary.CAGESourceBoundaryLockError, "captured twice"
            ):
                with handle.capture_base(key=_key("duplicate"), route=route):
                    with torch.no_grad():
                        model(state)
                        model(state)
            self.assertTrue(handle.cache_bank.poisoned)
            self.assertTrue(handle.cache_bank.cache_released)
        finally:
            _cleanup(handle)

        model = _ToyTransformer(blocks=1)
        handle = boundary.install_cage_source_boundary_lock(
            model, selected_block_indices=(0,)
        )
        key = _key("one-use")
        try:
            with handle.capture_base(key=key, route=route):
                with torch.no_grad():
                    model(state)
            with self.assertRaisesRegex(
                boundary.CAGESourceBoundaryLockError,
                "cross-state/source/action/sigma",
            ):
                with handle.lock_student(key=_key("wrong"), route=route):
                    pass
            self.assertEqual(handle.cache_bank.phase, "captured")
            with handle.lock_student(key=key, route=route):
                with cage.activate_route(route):
                    model(state.clone().requires_grad_(True))
            self.assertTrue(handle.cache_bank.cache_released)
            with self.assertRaisesRegex(
                boundary.CAGESourceBoundaryLockError, "already consumed"
            ):
                with handle.capture_base(key=key, route=route):
                    pass
        finally:
            _cleanup(handle)

    def test_exact_route_object_vi_cond_and_sigma_are_required(self) -> None:
        state = torch.ones(1, 7, 4)
        model = _ToyTransformer(blocks=1)
        handle = boundary.install_cage_source_boundary_lock(
            model, selected_block_indices=(0,)
        )
        try:
            with self.assertRaisesRegex(
                boundary.CAGESourceBoundaryLockError, "VI_cond-only"
            ):
                with handle.capture_base(
                    key=_key("row"),
                    route=_route(guidance_row="VI_uncond"),
                ):
                    pass
            with self.assertRaisesRegex(
                boundary.CAGESourceBoundaryLockError, "route sigma differs"
            ):
                with handle.capture_base(
                    key=_key("sigma", sigma_index=1), route=_route(sigma_index=0)
                ):
                    pass
        finally:
            _cleanup(handle)

        model = _ToyTransformer(blocks=1)
        handle = boundary.install_cage_source_boundary_lock(
            model, selected_block_indices=(0,)
        )
        route = _route()
        equal_but_distinct_route = _route()
        key = _key("route-object")
        try:
            with handle.capture_base(key=key, route=route):
                with torch.no_grad():
                    model(state)
            with self.assertRaisesRegex(
                boundary.CAGESourceBoundaryLockError, "exact active CAGE route object"
            ):
                with handle.lock_student(key=key, route=route):
                    with cage.activate_route(equal_but_distinct_route):
                        model(state)
            self.assertTrue(handle.cache_bank.poisoned)
            self.assertTrue(handle.cache_bank.cache_released)
        finally:
            _cleanup(handle)

    def test_hook_registry_is_audited_and_restore_is_exact(self) -> None:
        model = _ToyTransformer(blocks=2)
        handle = boundary.install_cage_source_boundary_lock(
            model, selected_block_indices=(0, 1)
        )
        extra = model.blocks[0].register_forward_hook(
            lambda module, inputs, output: None
        )
        try:
            with self.assertRaisesRegex(
                boundary.CAGESourceBoundaryLockError, "registry/order differs"
            ):
                handle.assert_scope()
        finally:
            extra.remove()
        handle.restore()
        self.assertTrue(all(not block._forward_hooks for block in model.blocks))
        with self.assertRaisesRegex(
            boundary.CAGESourceBoundaryLockError, "hooks are restored"
        ):
            handle.assert_scope()

        occupied = _ToyTransformer(blocks=1)
        old = occupied.blocks[0].register_forward_hook(
            lambda module, inputs, output: None
        )
        try:
            with self.assertRaisesRegex(
                boundary.CAGESourceBoundaryLockError,
                "already has forward hooks",
            ):
                boundary.install_cage_source_boundary_lock(
                    occupied, selected_block_indices=(0,)
                )
        finally:
            old.remove()

    def test_key_receipt_is_closed_and_replayable(self) -> None:
        key = _key("receipt", sigma_index=37)
        receipt = key.receipt()
        self.assertEqual(
            boundary.validate_same_state_action_key_receipt(receipt), receipt
        )
        changed = dict(receipt)
        changed["source_sha256"] = _sha("other-source")
        with self.assertRaisesRegex(
            boundary.CAGESourceBoundaryLockError, "digest differs"
        ):
            boundary.validate_same_state_action_key_receipt(changed)
        extra = dict(receipt)
        extra["unknown"] = True
        with self.assertRaisesRegex(
            boundary.CAGESourceBoundaryLockError, "closure differs"
        ):
            boundary.validate_same_state_action_key_receipt(extra)


if __name__ == "__main__":
    unittest.main()
