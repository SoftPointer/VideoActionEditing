from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import probe_source_local_anchor_quotient_v1 as probe  # noqa: E402


def cell(scale: float = 1.0) -> dict:
    noop = np.zeros((17, 4), dtype=np.float64)
    forward = noop.copy(); forward[:, 0] = np.linspace(0, scale, 17)
    reverse = forward[::-1].copy()
    hold16 = noop.copy(); hold16[:4, 0] = np.linspace(0, 0.25 * scale, 4); hold16[4:, 0] = 0.25 * scale
    hold24 = noop.copy(); hold24[:6, 0] = np.linspace(0, 0.4 * scale, 6); hold24[6:, 0] = 0.4 * scale
    nuisance = noop.copy(); nuisance[:, 1] = np.linspace(0, 0.8, 17)
    return {
        "normalized_noop": noop, "normalized_forward": forward,
        "reverse_from_forward": reverse, "incomplete_hold16": hold16,
        "incomplete_hold24": hold24, "camera_right_push": nuisance,
        "camera_center_push": nuisance * .9, "camera_vertical_push": nuisance * .8,
        "camera_center_pull": nuisance * -.7, "appearance_hue_ramp": nuisance * .6,
    }


def bank(prefix: str) -> dict:
    return {f"{family}:{prefix}-{family}-{index}": cell(1 + .1 * index) for family in ("dog", "human") for index in range(3)}


def test_signed_anchor_coefficient_separates_direction_completion_and_nuisance() -> None:
    result = probe.cell_scores(cell(), representation="velocity_trajectory", nuisance_rank=2)
    assert result["scores"]["normalized_forward"] == 1.0
    assert result["scores"]["reverse_from_forward"] < 0.0
    assert result["scores"]["incomplete_hold16"] < 1.0
    assert all(result["passes"].values())


def test_selection_is_fit_only_and_replay_is_disclosed() -> None:
    result = probe.select_and_replay(bank("fit"), bank("replay"))
    assert result["candidate_count"] == 18
    assert result["fit"]["all_five_margins_all_six_cells"] is True
    assert result["development_calibration_replay"]["all_five_margins_all_six_cells"] is True
    assert result["independent_calibration_claimed"] is False
