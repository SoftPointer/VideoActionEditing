from __future__ import annotations

import importlib.util
from pathlib import Path


TOOL = Path(__file__).resolve().parents[1] / "tools" / "build_frozen_multiscene_calibration_review_v1.py"
SPEC = importlib.util.spec_from_file_location("frozen_calibration_html", TOOL)
assert SPEC and SPEC.loader
html_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(html_tool)


def test_metric_card_marks_failed_margin() -> None:
    value = html_tool.metric_card("forward_gt_reverse", 4, -0.14)
    assert 'class="metric fail"' in value
    assert "4/6" in value
    assert "-0.140000" in value


def test_html_escape_is_applied() -> None:
    assert html_tool.esc('<script src="x">') == "&lt;script src=&quot;x&quot;&gt;"
