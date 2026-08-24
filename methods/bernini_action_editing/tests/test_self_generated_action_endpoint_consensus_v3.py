from __future__ import annotations

import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import self_generated_action_endpoint_consensus_v3 as endpoint


def _raw_teacher(torch, row: int, slot: int):
    value = torch.zeros(1, 21, 32, dtype=torch.float32)
    value[:, -3:, row] = 1.0 + 0.05 * slot
    value[:, :3, row] = -1.0
    return value


def _cells(torch):
    cells = []
    for row in range(4):
        action_channel = 0 if row < 2 else 1
        for slot in range(4):
            raw = _raw_teacher(torch, action_channel, slot)
            amplitude = torch.linalg.vector_norm(raw).item()
            cells.append(
                {
                    "row_index": row,
                    "slot": slot,
                    "teacher_unit": raw / amplitude,
                    "teacher_amplitude": amplitude,
                }
            )
    return cells


def test_arm_factorial_is_frozen():
    assert endpoint.ARM_NAMES == (
        "endpoint_cell_band",
        "endpoint_consensus_band",
        "endpoint_consensus_trust_001",
        "endpoint_consensus_trust_010",
    )
    assert endpoint.arm_spec("endpoint_cell_band").teacher_mode == "cell"
    assert endpoint.arm_spec("endpoint_consensus_trust_010").full_trust_weight == pytest.approx(0.1)


def test_endpoint_displacement_removes_time_constant_bias():
    torch = pytest.importorskip("torch")
    raw = torch.randn(1, 21, 32)
    bias = torch.randn(1, 1, 32)
    assert torch.allclose(
        endpoint.endpoint_displacement(raw),
        endpoint.endpoint_displacement(raw + bias),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_consensus_authority_uses_family_channel_and_robust_amplitude():
    torch = pytest.importorskip("torch")
    authority = endpoint.build_endpoint_authority(_cells(torch))
    assert set(authority) == {(row, slot) for row in range(4) for slot in range(4)}
    assert authority[(0, 0)].peer_consensus_cosine == pytest.approx(1.0)
    assert authority[(0, 0)].consensus_unit[0, 0].item() == pytest.approx(1.0)
    assert authority[(2, 0)].consensus_unit[0, 1].item() == pytest.approx(1.0)
    assert authority[(0, 0)].robust_amplitude > 0


def test_two_sided_band_penalizes_below_and_above_but_not_inside():
    torch = pytest.importorskip("torch")
    frozen = torch.zeros(1, 21, 32)
    teacher = torch.zeros(1, 32)
    teacher[0, 3] = 1.0
    amplitude = torch.tensor([2.0], dtype=torch.float32)

    def student_with_endpoint_gain(gain: float):
        value = frozen.clone()
        value[:, -3:, 3] = gain / 2.0
        value[:, :3, 3] = -gain / 2.0
        return value.requires_grad_(True)

    inside = endpoint.endpoint_band_loss(
        student_raw=student_with_endpoint_gain(0.2),
        frozen_raw=frozen,
        detached_teacher_unit=teacher,
        detached_teacher_amplitude=amplitude,
        lower_scale=0.05,
        upper_scale=0.15,
    )
    below = endpoint.endpoint_band_loss(
        student_raw=student_with_endpoint_gain(0.0),
        frozen_raw=frozen,
        detached_teacher_unit=teacher,
        detached_teacher_amplitude=amplitude,
        lower_scale=0.05,
        upper_scale=0.15,
    )
    above = endpoint.endpoint_band_loss(
        student_raw=student_with_endpoint_gain(0.4),
        frozen_raw=frozen,
        detached_teacher_unit=teacher,
        detached_teacher_amplitude=amplitude,
        lower_scale=0.05,
        upper_scale=0.15,
    )
    assert inside.action.item() == pytest.approx(0.0, abs=1.0e-9)
    assert below.action.item() == pytest.approx(1.0)
    assert above.action.item() == pytest.approx(1.0)


def test_full_functional_trust_is_zero_only_at_frozen_output():
    torch = pytest.importorskip("torch")
    frozen = torch.ones(1, 2, 3, 4, 5)
    same = frozen.clone().requires_grad_(True)
    changed = (frozen + 0.5).requires_grad_(True)
    assert endpoint.full_functional_trust(
        student_velocity=same, frozen_velocity=frozen
    ).item() == pytest.approx(0.0)
    assert endpoint.full_functional_trust(
        student_velocity=changed, frozen_velocity=frozen
    ).item() == pytest.approx(0.25)
