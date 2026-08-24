from __future__ import annotations

import importlib.util
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "run_prospective_forward_anchor_shard_v1.py"
SPEC = importlib.util.spec_from_file_location("forward_anchor_shard", SOURCE)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _manifest() -> dict:
    return {
        "entries": [
            {
                "source_id": source, "seed": seed, "branch": branch,
                "analysis_split": split, "entry_id": f"{source}-{seed}-{branch}",
            }
            for source, seed, split in (
                ("0" * 16, 11, "fit"), ("1" * 16, 22, "fit"),
            )
            for branch in (
                "appearance_only", "camera_only", "forward", "incomplete",
                "noop", "reverse", "wrong_actor_or_object",
            )
        ]
    }


def test_only_forward_and_noop_are_released() -> None:
    rows = runner._released_rows(_manifest(), [("0" * 16, 11), ("1" * 16, 22)])
    assert [row["branch"] for row in rows] == ["noop", "forward", "noop", "forward"]
    assert all(row["analysis_split"] == "fit" for row in rows)


def test_authority_is_closed() -> None:
    assert runner.RELEASE_BRANCHES == ("noop", "forward")
    assert runner.AUTHORITY["decoded_review_required"] is True
    assert runner.AUTHORITY["representation_selection_authorized"] is False
    assert runner.AUTHORITY["training_target_authorized"] is False
    assert runner.AUTHORITY["optimizer_step_authorized"] is False
    text = SOURCE.read_text(encoding="utf-8")
    assert "scancel" not in text
    assert "scontrol release" not in text
