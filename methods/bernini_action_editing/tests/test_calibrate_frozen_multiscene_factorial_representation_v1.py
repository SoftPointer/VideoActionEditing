from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import calibrate_frozen_multiscene_factorial_representation_v1 as calibration  # noqa: E402


def frozen() -> dict:
    return {
        "head_a": {
            "candidate_id": "velocity_trajectory-ar1-nr2",
            "representation": "velocity_trajectory", "action_rank": 1, "nuisance_rank": 2,
        },
        "head_b": {
            "candidate_id": "centered_trajectory-ar1-nr2",
            "representation": "centered_trajectory", "action_rank": 1, "nuisance_rank": 2,
        },
        "weight_a": 0.6, "weight_b": 0.4,
    }


def bank(prefix: str, scale: float = 1.0) -> dict:
    result = {}
    for family_index, family in enumerate(("dog", "human")):
        for index in range(3):
            noop = np.zeros((17, 4), dtype=np.float64)
            forward = noop.copy()
            forward[:, 0] = np.linspace(0.0, scale * (1.0 + 0.1 * index), 17)
            reverse = forward[::-1].copy()
            incomplete = noop.copy()
            incomplete[:9, 0] = np.linspace(0.0, scale * 0.4, 9)
            incomplete[9:, 0] = scale * 0.4
            nuisance = noop.copy()
            nuisance[:, 1] = np.linspace(0.0, 0.2 + 0.05 * family_index, 17)
            result[f"{family}:{prefix}-{family}-{index}"] = {
                "normalized_noop": noop,
                "normalized_forward": forward,
                "reverse_from_forward": reverse,
                "incomplete_phasewarp": incomplete,
                "camera_right_push": nuisance,
                "camera_center_push": nuisance * 0.9,
                "camera_vertical_push": nuisance * 0.8,
                "camera_center_pull": nuisance * -0.7,
                "appearance_hue_ramp": nuisance * 0.6,
            }
    return result


def test_calibration_uses_zero_ordinal_boundary_and_disjoint_cells() -> None:
    result, models = calibration.calibrate(bank("fit"), bank("cal", 1.1), frozen())
    assert result["cell_count"] == 6
    assert set(result["ordinal_thresholds"].values()) == {0.0}
    assert len(models) == 4
    assert result["pass_counts"]["forward_gt_noop"] == 6
    assert result["confirmation_evaluation_authorized"] is True


def test_calibration_rejects_fit_calibration_overlap() -> None:
    value = bank("same")
    with pytest.raises(calibration.FrozenCalibrationError, match="overlap"):
        calibration.calibrate(value, value, frozen())


def test_frozen_receipt_rejects_adaptive_calibration() -> None:
    receipt = {
        "schema_version": calibration.TWO_HEAD_SCHEMA,
        "authority": {
            "fit_only_representation_selection": True,
            "calibration_accessed": True,
            "confirmation_accessed": False,
            "calibration_may_not_reselect_heads_or_weights": True,
            "optimizer_step_authorized": False,
        },
        "selected": {
            "head_a": "speed_profile-ar1-nr4",
            "head_b": "temporal_self_similarity-ar2-nr2",
            "weight_a": 0.61, "weight_b": 0.39,
            "all_four_margins_all_six_folds": True,
        },
    }
    with pytest.raises(calibration.FrozenCalibrationError, match="receipt differs"):
        calibration.frozen_selection(receipt)
