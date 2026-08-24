from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import run_prospective_calibration_noop_retry_v1 as retry  # noqa: E402


MANIFEST = METHOD_ROOT / "assets" / "prospective_factorial_branch_manifest_v1.json"


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_retry_noop_is_calibration_same_seed_counterpart() -> None:
    row, registered = retry.retry_row(manifest(), "1a6fdd7d9c19413b", 2026082534)
    assert registered == [2026082531, 2026082532]
    assert row["analysis_split"] == "calibration"
    assert row["branch"] == "noop"
    assert row["seed"] == 2026082534
    assert row["entry_id"] == "1a6fdd7d9c19413b-cal-retry-s2026082534-noop"


def test_retry_noop_rejects_registered_seed() -> None:
    with pytest.raises(retry.CalibrationNoopRetryError, match="aliases"):
        retry.retry_row(manifest(), "8780e5b895774e8a", 2026082411)


def test_retry_noop_rejects_prompt_drift() -> None:
    value = copy.deepcopy(manifest())
    for row in value["entries"]:
        if (
            row["source_id"] == "8780e5b895774e8a"
            and row["branch"] == "noop"
            and row["seed"] == 2026082412
        ):
            row["instruction"] += " drift"
    with pytest.raises(retry.CalibrationNoopRetryError, match="differ beyond seed"):
        retry.retry_row(value, "8780e5b895774e8a", 2026082413)


def test_retry_noop_receipt_stays_fail_closed() -> None:
    value = manifest()
    row, registered = retry.retry_row(value, "8780e5b895774e8a", 2026082413)
    receipt = retry.retry_receipt(
        manifest_sha256="a" * 64,
        manifest_digest=value["manifest_digest"],
        row=row,
        registered_seeds=registered,
        entry_receipt={"receipt_digest": "b" * 64},
    )
    assert receipt["branch"] == "noop"
    assert receipt["authority"]["representation_reselection_authorized"] is False
    assert receipt["authority"]["optimizer_step_authorized"] is False
    unsigned = dict(receipt)
    digest = unsigned.pop("receipt_digest")
    assert digest == retry.base.branch_manifest.object_sha256(unsigned)
