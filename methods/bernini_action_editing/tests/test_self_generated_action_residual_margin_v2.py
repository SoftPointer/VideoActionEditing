from __future__ import annotations

import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import self_generated_action_residual_margin_v2 as residual


def _teacher(torch):
    value = torch.zeros(1, 21, 32, dtype=torch.float32)
    value[0, 4, 7] = 1.0
    return value


def test_eight_arm_factorial_is_frozen():
    assert len(residual.ARM_NAMES) == 8
    assert residual.arm_spec("margin_005").margin_scale == pytest.approx(0.05)
    assert residual.arm_spec("margin_010_perp_100").perpendicular_weight == pytest.approx(1.0)
    assert residual.arm_spec("margin_010_perp_100_onset_400").onset_weight == pytest.approx(4.0)
    assert residual.arm_spec("margin_010_perp_100_onset_400_noop_020").noop_weight == pytest.approx(0.2)


def test_zero_residual_needs_gain_but_has_no_perpendicular_change():
    torch = pytest.importorskip("torch")
    frozen = torch.randn(1, 21, 32)
    teacher = _teacher(torch)
    loss = residual.residual_margin_loss(
        student_raw=frozen.clone().requires_grad_(True),
        frozen_raw=frozen,
        detached_teacher_unit=teacher,
        detached_teacher_amplitude=torch.tensor([2.0]),
        margin_scale=0.1,
    )
    assert loss.action.item() == pytest.approx(1.0)
    assert loss.perpendicular.item() == pytest.approx(0.0)
    assert loss.gain_mean.item() == pytest.approx(0.0)


def test_exact_margin_changes_only_action_direction():
    torch = pytest.importorskip("torch")
    frozen = torch.randn(1, 21, 32)
    teacher = _teacher(torch)
    student = (frozen + 0.2 * teacher).requires_grad_(True)
    loss = residual.residual_margin_loss(
        student_raw=student,
        frozen_raw=frozen,
        detached_teacher_unit=teacher,
        detached_teacher_amplitude=torch.tensor([2.0]),
        margin_scale=0.1,
    )
    assert loss.action.item() == pytest.approx(0.0, abs=1e-10)
    assert loss.perpendicular.item() == pytest.approx(0.0, abs=1e-10)
    assert loss.gain_mean.item() == pytest.approx(0.2)


def test_orthogonal_adapter_change_is_penalized_and_teacher_is_detached():
    torch = pytest.importorskip("torch")
    frozen = torch.zeros(1, 21, 32)
    teacher = _teacher(torch)
    student = torch.zeros_like(frozen)
    student[0, 5, 9] = 1.0
    student.requires_grad_(True)
    loss = residual.residual_margin_loss(
        student_raw=student,
        frozen_raw=frozen,
        detached_teacher_unit=teacher,
        detached_teacher_amplitude=torch.tensor([2.0]),
        margin_scale=0.1,
    )
    total = loss.action + loss.perpendicular
    total.backward()
    assert loss.action.item() == pytest.approx(1.0)
    assert loss.perpendicular.item() == pytest.approx(0.25)
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None


def test_onset_envelope_weights_three_phases_only():
    torch = pytest.importorskip("torch")
    source = torch.zeros(1, 2, 21, 2, 2)
    predicted = source.clone()
    predicted[:, :, 0] = 1.0
    predicted[:, :, 1] = 2.0
    predicted[:, :, 2] = 4.0
    predicted[:, :, 3] = 100.0
    loss = residual.onset_preservation_loss(
        predicted_clean=predicted,
        source_clean=source,
        onset_frames=3,
    )
    expected = (1.0 * 1.0 + 4.0 * 0.5 + 16.0 * 0.25) / 1.75
    assert loss.item() == pytest.approx(expected)
