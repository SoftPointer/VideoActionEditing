from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "probe_factorial_axis_controls_v3_dinov2.py"
LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "scripts/auh_probe_factorial_axis_controls_v3_dinov2_on_135411.sh"
)
SPEC = importlib.util.spec_from_file_location("axis_v3_probe", SOURCE)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _arc(end: float) -> np.ndarray:
    theta = np.linspace(0.0, end, len(probe.base.FRAME_INDICES))
    return np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)


def test_phasewarp_reduces_hold_static_shortcut() -> None:
    forward = _arc(0.8)
    noop = np.repeat([[1.0, 0.0, 0.0]], len(probe.base.FRAME_INDICES), axis=0)
    hold = forward.copy()
    hold[3:] = hold[2]
    warp = _arc(0.3)
    values = {
        "normalized_noop": noop,
        "normalized_forward": forward,
        "reverse_from_forward": forward[::-1].copy(),
        "incomplete_hold": hold,
        "incomplete_phasewarp": warp,
        "camera_right_push": _arc(0.05),
        "camera_center_push": _arc(-0.04),
        "camera_vertical_push": _arc(0.03),
        "camera_center_pull": _arc(-0.02),
        "appearance_hue_ramp": _arc(0.01),
    }
    result = probe.analyze_family(values, family="dog")
    incomplete = result["incomplete_comparison"]
    assert incomplete["incomplete_hold"]["near_static_interval_fraction"] > 0.8
    assert incomplete["incomplete_phasewarp"]["near_static_interval_fraction"] < 0.1
    assert result["reverse_integrity"]["mean_forward_vs_time_reversed_reverse_alignment"] == 1.0


def test_authority_and_branch_closure() -> None:
    assert len(probe.BRANCHES) == 10
    assert len(probe.NUISANCE_BRANCHES) == 5
    assert probe.AUTHORITY["diagnostic_only"] is True
    assert probe.AUTHORITY["representation_selection_authorized"] is False
    assert probe.AUTHORITY["training_target_authorized"] is False
    assert probe.AUTHORITY["optimizer_or_parameter_update_authorized"] is False
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "readonly holder_job=135411" in launcher
    assert "holder_retained=%s" in launcher
    assert "scancel" not in launcher
    assert "scontrol release" not in launcher
