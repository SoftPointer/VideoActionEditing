from __future__ import annotations

import importlib.util
from pathlib import Path
import hashlib

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "probe_factorial_axis_controls_v2_dinov2.py"
LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "scripts/auh_probe_factorial_axis_controls_v2_dinov2_on_135407.sh"
)
SPEC = importlib.util.spec_from_file_location("axis_probe", SOURCE)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _arc(end: float) -> np.ndarray:
    theta = np.linspace(0.0, end, len(probe.FRAME_INDICES))
    return np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)


def test_exact_reverse_and_hold_shortcut_are_exposed() -> None:
    forward = _arc(0.8)
    incomplete = forward.copy()
    incomplete[3:] = incomplete[2]
    noop = np.repeat([[1.0, 0.0, 0.0]], len(probe.FRAME_INDICES), axis=0)
    phi = np.linspace(0.0, 0.3, len(probe.FRAME_INDICES))
    camera = np.stack([np.cos(phi), np.zeros_like(phi), np.sin(phi)], axis=1)
    result = probe.analyze_family(
        {
            "normalized_noop": noop,
            "normalized_forward": forward,
            "reverse_from_forward": forward[::-1].copy(),
            "incomplete_from_forward": incomplete,
            "camera_from_noop": camera,
        },
        family="dog", cut_frame=10,
    )
    assert result["reverse_integrity"]["mean_alignment"] == 1.0
    assert result["reverse_integrity"]["temporal_self_similarity_reversal_rmse"] < 1e-12
    assert result["endpoint_arrow"]["cosine_to_forward_axis"]["reverse_from_forward"] < -0.999999
    assert result["incomplete_hold"]["prefix_mean_alignment"] == 1.0
    assert result["incomplete_hold"]["static_tail_shortcut_severity"] > 0.999999


def test_authority_is_diagnostic_only() -> None:
    assert probe.AUTHORITY["diagnostic_only"] is True
    assert probe.AUTHORITY["representation_selection_authorized"] is False
    assert probe.AUTHORITY["training_target_authorized"] is False
    assert probe.AUTHORITY["optimizer_or_parameter_update_authorized"] is False
    text = SOURCE.read_text(encoding="utf-8")
    assert "scancel" not in text
    assert "scontrol release" not in text


def test_holder_launcher_retains_allocation() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "readonly holder_job=135407" in text
    assert "holder_retained=%s" in text
    assert "ROCR_VISIBLE_DEVICES=0" in text
    assert "scancel" not in text
    assert "scontrol release" not in text


def test_media_manifest_uses_family_branch_suffix(tmp_path: Path) -> None:
    root = tmp_path / "axis-controls-v2-a3"
    rows = []
    for family in probe.FAMILIES:
        (root / family).mkdir(parents=True, exist_ok=True)
        for branch in probe.BRANCHES:
            video = root / family / f"{branch}.mp4"
            video.write_bytes(f"{family}:{branch}".encode("ascii"))
            digest = hashlib.sha256(video.read_bytes()).hexdigest()
            rows.append(f"{digest}  /sealed/attempt/axis-controls-v2-a3/{family}/{branch}.mp4")
    manifest = root / "media.sha256"
    manifest.write_text("\n".join(rows) + "\n", encoding="ascii")
    sealed = probe.sealed_media(root, probe.file_sha256(manifest))
    assert set(sealed) == set(probe.FAMILIES)
    assert all(set(sealed[family]) == set(probe.BRANCHES) for family in probe.FAMILIES)
