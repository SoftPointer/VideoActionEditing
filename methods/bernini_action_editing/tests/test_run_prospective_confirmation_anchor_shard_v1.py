from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import run_prospective_confirmation_anchor_shard_v1 as shard  # noqa: E402


MANIFEST = METHOD_ROOT / "assets" / "prospective_factorial_branch_manifest_v1.json"


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_released_rows_are_confirmation_forward_noop_only() -> None:
    cells = [("07a4fa39808a4f02", 2026082501), ("07a4fa39808a4f02", 2026082502)]
    rows = shard.released_rows(manifest(), cells)
    assert len(rows) == 4
    assert {row["branch"] for row in rows} == {"noop", "forward"}
    assert {row["analysis_split"] for row in rows} == {"confirmation"}


def test_released_rows_reject_calibration_cell() -> None:
    with pytest.raises(shard.ConfirmationAnchorShardError, match="confirmation"):
        shard.released_rows(manifest(), [("2af820dd10324328", 2026082401)])


def test_receipt_prevents_reselection_recalibration_and_optimizer() -> None:
    value = manifest()
    cells = [("2d98b7c91dbc4d98", 2026082621)]
    receipt = shard.receipt(
        manifest_sha256="a" * 64,
        manifest_digest=value["manifest_digest"],
        cells=cells,
        entry_receipts=[{"receipt_digest": "b" * 64}, {"receipt_digest": "c" * 64}],
    )
    assert receipt["analysis_split"] == "confirmation"
    assert receipt["authority"]["representation_reselection_authorized"] is False
    assert receipt["authority"]["threshold_recalibration_authorized"] is False
    assert receipt["authority"]["optimizer_step_authorized"] is False
    unsigned = dict(receipt)
    digest = unsigned.pop("receipt_digest")
    assert digest == shard.base.branch_manifest.object_sha256(unsigned)
