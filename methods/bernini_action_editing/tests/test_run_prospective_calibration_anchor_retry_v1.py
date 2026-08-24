from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import run_prospective_calibration_anchor_retry_v1 as retry  # noqa: E402


MANIFEST = METHOD_ROOT / "assets" / "prospective_factorial_branch_manifest_v1.json"


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_retry_is_calibration_only() -> None:
    row, registered = retry.retry_row(manifest(), "8780e5b895774e8a", 2026082413)
    assert registered == [2026082411, 2026082412]
    assert row["analysis_split"] == "calibration"
    assert row["seed"] == 2026082413


def test_retry_rejects_fit_source() -> None:
    with pytest.raises(retry.CalibrationAnchorRetryError, match="calibration"):
        retry.retry_row(manifest(), "0b2fc177202e4d08", 2026082513)


def test_receipt_prevents_reselection_and_optimizer() -> None:
    row, registered = retry.retry_row(manifest(), "8780e5b895774e8a", 2026082413)
    value = retry.retry_receipt(
        manifest_sha256="a" * 64,
        manifest_digest="b" * 64,
        row=row,
        registered_seeds=registered,
        entry_receipt={"receipt_digest": "c" * 64},
    )
    assert value["authority"]["representation_reselection_authorized"] is False
    assert value["authority"]["threshold_freezing_authorized"] is False
    assert value["authority"]["optimizer_step_authorized"] is False
