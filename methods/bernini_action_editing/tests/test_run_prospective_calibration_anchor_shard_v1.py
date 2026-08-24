from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import run_prospective_calibration_anchor_shard_v1 as runner  # noqa: E402


MANIFEST = METHOD_ROOT / "assets" / "prospective_factorial_branch_manifest_v1.json"


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_releases_only_calibration_forward_and_noop() -> None:
    rows = runner.released_rows(manifest(), [("1a6fdd7d9c19413b", 2026082531)])
    assert [row["branch"] for row in rows] == ["noop", "forward"]
    assert {row["analysis_split"] for row in rows} == {"calibration"}


def test_rejects_fit_cell() -> None:
    with pytest.raises(runner.CalibrationAnchorShardError, match="closure"):
        runner.released_rows(manifest(), [("0b2fc177202e4d08", 2026082511)])


def test_receipt_keeps_reselection_and_optimizer_closed() -> None:
    value = runner.receipt(
        manifest_sha256="a" * 64,
        manifest_digest="b" * 64,
        cells=[("1a6fdd7d9c19413b", 2026082531)],
        entry_receipts=[{"receipt_digest": "c" * 64}],
    )
    assert value["analysis_split"] == "calibration"
    assert value["authority"]["representation_reselection_authorized"] is False
    assert value["authority"]["optimizer_step_authorized"] is False
