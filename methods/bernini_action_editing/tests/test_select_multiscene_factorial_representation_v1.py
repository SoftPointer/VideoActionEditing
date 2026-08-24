from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import select_multiscene_factorial_representation_v1 as selection  # noqa: E402


def test_representations_are_finite_vectors() -> None:
    features = np.arange(17 * 6, dtype=np.float64).reshape(17, 6)
    for name in selection.REPRESENTATIONS:
        value = selection.representation(features, name)
        assert value.ndim == 1
        assert np.isfinite(value).all()


def test_nullspace_projection_removes_observed_nuisance() -> None:
    nuisance = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    basis = selection._row_basis(nuisance, 2)
    projected = selection._project_null(np.asarray([3.0, 4.0, 5.0]), basis)
    assert np.allclose(projected, [0.0, 0.0, 5.0], atol=1.0e-10)


def test_candidate_grid_selects_without_absolute_threshold() -> None:
    bank = {}
    for family_index, family in enumerate(selection.FAMILIES):
        for cell_index in range(3):
            key = f"{family}:{cell_index}"
            noop = np.zeros((17, 4), dtype=np.float64)
            forward = noop.copy()
            forward[:, 0] = np.linspace(0.0, 1.0 + 0.1 * cell_index, 17)
            reverse = forward[::-1].copy()
            incomplete = noop.copy()
            incomplete[:, 0] = np.linspace(0.0, 0.4, 17)
            nuisance = noop.copy()
            nuisance[:, 1] = np.linspace(0.0, 1.0 + family_index, 17)
            bank[key] = {
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
    result = selection.select_candidates(bank)
    assert result["candidate_count"] == 36
    assert result["selection_rule"].endswith("no_absolute_threshold")
    assert result["selected_candidate_id"]
