from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from action.omni import (  # noqa: E402
    SourceContextBudgetError,
    full_source_context_budget,
    require_full_source_context,
    set_exact_omni_context_length,
)


class _FakeWan:
    def __init__(self, text_len: int) -> None:
        self.text_len = text_len


class _FakeOmni(nn.Module):
    def __init__(self, max_context_len: int) -> None:
        super().__init__()
        self.max_context_len = max_context_len
        self.wan_model = _FakeWan(max_context_len)

    def reset_wan_text_len(self, value: int) -> None:
        self.wan_model.text_len = value


class ActionContextBudgetTest(unittest.TestCase):
    def test_81_full_and_motion_384_token_counts(self) -> None:
        full = full_source_context_budget(
            torch.zeros(1, 16, 21, 60, 104),
            max_context_len=9216,
            nonvisual_tokens=420,
        )
        self.assertEqual(full.visual_tokens, 8190)
        self.assertEqual(full.total_tokens, 8610)
        self.assertEqual(full.fixed_budget_padding_tokens, 606)
        self.assertTrue(full.fits)

        low = full_source_context_budget(
            torch.zeros(1, 16, 21, 48, 80),
            max_context_len=6144,
            nonvisual_tokens=420,
        )
        self.assertEqual(low.visual_tokens, 5040)
        self.assertEqual(low.total_tokens, 5460)
        self.assertEqual(low.fixed_budget_padding_tokens, 684)
        self.assertTrue(low.fits)

    def test_over_budget_error_reports_every_required_count(self) -> None:
        with self.assertRaises(SourceContextBudgetError) as caught:
            require_full_source_context(
                torch.zeros(1, 16, 21, 60, 104),
                max_context_len=6144,
                nonvisual_tokens=420,
                sample_id="row-81",
                task_type="action_edit",
            )
        message = str(caught.exception)
        for fragment in (
            "sample_id='row-81'",
            "nonvisual=420",
            "visual=8190",
            "total=8610",
            "budget=6144",
        ):
            self.assertIn(fragment, message)

    def test_batch_exact_length_change_is_explicit_helper_only(self) -> None:
        model = _FakeOmni(9216)
        self.assertEqual(model.wan_model.text_len, 9216)
        self.assertEqual(
            set_exact_omni_context_length(
                model, exact_context_len=8610, max_context_len=9216
            ),
            8610,
        )
        self.assertEqual(model.max_context_len, 9216)
        self.assertEqual(model.wan_model.text_len, 8610)


if __name__ == "__main__":
    unittest.main()
