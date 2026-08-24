import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import self_generated_action_quotient_v1 as quotient


def test_arm_factorial_is_frozen():
    assert len(quotient.ARM_NAMES) == 8
    assert quotient.arm_spec("action_only").learning_rate == pytest.approx(1e-4)
    assert quotient.arm_spec("action_only_lowlr").learning_rate == pytest.approx(5e-5)
    assert quotient.arm_spec("action_start_nuisance_noop").noop_weight > 0
    assert quotient.arm_spec("action_start_nuisance_border").border_weight > 0


def test_nuisance_loss_has_gradient_only_for_student_code():
    torch = pytest.importorskip("torch")
    raw = torch.randn(1, 21, 32, requires_grad=True)
    camera = torch.randn(1, 21, 32)
    camera = camera / camera.norm()
    appearance = torch.randn(1, 21, 32)
    appearance = appearance - (appearance * camera).sum() * camera
    appearance = appearance / appearance.norm()
    value = quotient.nuisance_coefficient_loss(raw, camera.detach(), appearance.detach())
    value.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert camera.grad is None and appearance.grad is None


def test_start_loss_is_not_hidden_by_later_frames():
    torch = pytest.importorskip("torch")
    source = torch.zeros(1, 16, 21, 12, 12)
    predicted = source.clone()
    predicted[:, :, 0] = 2
    losses = quotient.preservation_losses(
        predicted_clean=predicted, source_clean=source, border_width=2
    )
    assert losses["start"].item() == pytest.approx(4.0)
    assert losses["border"].item() > 0


def test_weighted_total_respects_zero_weight_controls():
    torch = pytest.importorskip("torch")
    scalars = [torch.tensor(float(x)) for x in (1, 2, 3, 4, 5)]
    total = quotient.weighted_total(
        spec=quotient.arm_spec("action_only"),
        action=scalars[0], noop=scalars[1], start=scalars[2],
        nuisance=scalars[3], border=scalars[4],
    )
    assert total.item() == pytest.approx(1.0)
