from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import run_prospective_forward_anchor_retry_v1 as retry  # noqa: E402


MANIFEST = METHOD_ROOT / "assets" / "prospective_factorial_branch_manifest_v1.json"


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_retry_row_keeps_frozen_source_and_instruction() -> None:
    value = manifest()
    row, registered = retry.retry_row(value, "173371bf8fa74785", 2026082523)
    originals = [
        item for item in value["entries"]
        if item["source_id"] == "173371bf8fa74785" and item["branch"] == "forward"
    ]
    assert registered == [2026082521, 2026082522]
    assert row["seed"] == 2026082523
    assert row["entry_id"] == "173371bf8fa74785-retry-s2026082523-forward"
    assert row["instruction"] == originals[0]["instruction"]
    assert row["source_video_sha256"] == originals[0]["source_video_sha256"]


def test_retry_rejects_registered_seed() -> None:
    with pytest.raises(retry.ForwardAnchorRetryError, match="aliases"):
        retry.retry_row(manifest(), "173371bf8fa74785", 2026082521)


def test_retry_rejects_non_fit_or_nonclosed_source() -> None:
    value = manifest()
    value["entries"] = [
        row for row in value["entries"]
        if not (
            row["source_id"] == "173371bf8fa74785"
            and row["branch"] == "forward"
            and row["seed"] == 2026082522
        )
    ]
    with pytest.raises(retry.ForwardAnchorRetryError, match="two registered"):
        retry.retry_row(value, "173371bf8fa74785", 2026082523)


def test_retry_rejects_prompt_drift_between_registered_rows() -> None:
    value = copy.deepcopy(manifest())
    for row in value["entries"]:
        if (
            row["source_id"] == "173371bf8fa74785"
            and row["branch"] == "forward"
            and row["seed"] == 2026082522
        ):
            row["instruction"] += " drift"
    with pytest.raises(retry.ForwardAnchorRetryError, match="differ beyond seed"):
        retry.retry_row(value, "173371bf8fa74785", 2026082523)


def test_retry_receipt_is_fail_closed() -> None:
    value = manifest()
    row, registered = retry.retry_row(value, "31fcd6205efb4b84", 2026082633)
    receipt = retry.retry_receipt(
        manifest_sha256="a" * 64,
        manifest_digest=value["manifest_digest"],
        row=row,
        registered_seeds=registered,
        entry_receipt={"receipt_digest": "b" * 64},
    )
    assert receipt["authority"]["post_review_seed_expansion_disclosed"] is True
    assert receipt["authority"]["representation_selection_authorized"] is False
    assert receipt["authority"]["optimizer_step_authorized"] is False
    unsigned = dict(receipt)
    digest = unsigned.pop("receipt_digest")
    assert digest == retry.base.branch_manifest.object_sha256(unsigned)
