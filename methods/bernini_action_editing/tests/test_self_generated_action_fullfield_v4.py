from __future__ import annotations

import torch

import self_generated_action_fullfield_v4 as v4


def field(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((1, 16, 21, 4, 6), generator=generator)


def test_fullfield_loss_uses_all_phases_and_has_zero_exact_match() -> None:
    teacher = v4.anchor_action_trajectory(field())
    exact = v4.fullfield_action_loss(teacher.clone().requires_grad_(), teacher)
    wrong = teacher.clone()
    wrong[:, :, 10, 1, 2] += 2.0
    changed = v4.fullfield_action_loss(wrong.requires_grad_(), teacher)
    assert exact.total.item() == 0.0
    assert changed.total.item() > 0.0
    changed.total.backward()
    assert wrong.grad is not None


def test_source_carrier_preserves_phase_zero_exactly() -> None:
    source = field(1)
    anchor = field(2)
    target = v4.source_carrier_target(source, anchor)
    assert torch.equal(target[:, :, 0], source[:, :, 0])
    assert not torch.equal(target[:, :, 1:], source[:, :, 1:])


def test_action_first_projection_removes_conflict_and_caps_preservation() -> None:
    action = [torch.tensor([1.0, 0.0])]
    preservation = [torch.tensor([-2.0, 4.0])]
    combined, metrics = v4.project_and_cap_preservation_gradients(
        action, preservation, cap_ratio=0.25
    )
    injected = combined[0] - action[0]
    assert metrics["conflict_projected"] is True
    assert torch.dot(injected, action[0]).abs().item() < 1.0e-6
    assert torch.linalg.vector_norm(injected).item() <= 0.250001
