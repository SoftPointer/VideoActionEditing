from __future__ import annotations

import importlib.util
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "tools/build_prospective_forward_anchor_review_v1.py"
)
SPEC = importlib.util.spec_from_file_location("forward_review", SOURCE)
assert SPEC is not None and SPEC.loader is not None
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def test_population_is_exact_fit12() -> None:
    assert len(review.OLD_CELLS) == 2
    assert len(review.NEW_CELLS) == 10
    assert len(review.ALL_CELLS) == 12
    assert len(set(review.ALL_CELLS)) == 12
    assert {source for source, _ in review.ALL_CELLS} == {
        "0b2fc177202e4d08", "1367d5595ed641ae", "173371bf8fa74785",
        "31fcd6205efb4b84", "5f4ba4fb4c6441e0", "7421728d949d40dd",
    }


def test_authority_and_html_copy_boundary() -> None:
    assert review.AUTHORITY["decoded_review_only"] is True
    assert review.AUTHORITY["training_target_authorized"] is False
    assert review.AUTHORITY["representation_selection_authorized"] is False
    assert review.AUTHORITY["optimizer_step_authorized"] is False
    text = SOURCE.read_text(encoding="utf-8")
    assert "symlink_to" not in text
    assert "scancel" not in text
    assert "scontrol release" not in text
