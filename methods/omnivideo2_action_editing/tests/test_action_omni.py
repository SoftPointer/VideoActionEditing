from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from action.checkpoint_contract import (  # noqa: E402
    OMNIVIDEO2_1_3B_ACTIVE_SPECIAL_TOKEN_ROWS,
    OMNIVIDEO2_1_3B_SERIALIZED_SPECIAL_TOKEN_ROWS,
)
from action.omni import _load_special_token_payload  # noqa: E402


def _official_layout() -> dict[str, torch.Tensor]:
    return {
        "<img_st>": torch.zeros(6, 4096, dtype=torch.bfloat16),
        "<img_ed>": torch.zeros(6, 4096, dtype=torch.bfloat16),
        "<ipl_st>": torch.zeros(7, 4096, dtype=torch.bfloat16),
        "<ipl_ed>": torch.zeros(7, 4096, dtype=torch.bfloat16),
        "<prp_st>": torch.zeros(7, 4096, dtype=torch.bfloat16),
        "<prp_ed>": torch.zeros(7, 4096, dtype=torch.bfloat16),
    }


def _encode(value: dict[str, torch.Tensor]) -> tuple[bytes, str]:
    payload = pickle.dumps(value)
    return payload, hashlib.sha256(payload).hexdigest()


class ActionOmniCheckpointContractTest(unittest.TestCase):
    def test_exact_six_entry_layout_yields_four_active_entries_and_26_rows(self) -> None:
        payload, digest = _encode(_official_layout())
        result, rows, actual_digest = _load_special_token_payload(
            payload,
            expected_sha256=digest,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        self.assertEqual(
            tuple(result), ("<img_st>", "<img_ed>", "<prp_st>", "<prp_ed>")
        )
        self.assertEqual(rows, OMNIVIDEO2_1_3B_ACTIVE_SPECIAL_TOKEN_ROWS)
        self.assertEqual(rows, 26)
        self.assertEqual(OMNIVIDEO2_1_3B_SERIALIZED_SPECIAL_TOKEN_ROWS, 40)
        self.assertEqual(actual_digest, digest)

    def test_digest_is_checked_before_unpickler_is_called(self) -> None:
        called = False

        def forbidden_unpickler(_payload):
            nonlocal called
            called = True
            raise AssertionError("unpickler must not run")

        with self.assertRaisesRegex(ValueError, "digest differs"):
            _load_special_token_payload(
                b"not the pinned pickle",
                expected_sha256="0" * 64,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
                unpickler=forbidden_unpickler,
            )
        self.assertFalse(called)

    def test_same_total_but_different_active_layout_is_rejected(self) -> None:
        value = _official_layout()
        value["<img_st>"] = torch.zeros(5, 4096, dtype=torch.bfloat16)
        value["<img_ed>"] = torch.zeros(7, 4096, dtype=torch.bfloat16)
        payload, digest = _encode(value)
        with self.assertRaisesRegex(ValueError, "<img_st>.*CPU BF16"):
            _load_special_token_payload(
                payload,
                expected_sha256=digest,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )

    def test_wrong_dtype_and_nonfinite_tensor_are_rejected(self) -> None:
        wrong_dtype = _official_layout()
        wrong_dtype["<prp_ed>"] = torch.zeros(7, 4096, dtype=torch.float32)
        payload, digest = _encode(wrong_dtype)
        with self.assertRaisesRegex(ValueError, "<prp_ed>.*CPU BF16"):
            _load_special_token_payload(
                payload,
                expected_sha256=digest,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )

        nonfinite = _official_layout()
        nonfinite["<ipl_st>"][0, 0] = float("nan")
        payload, digest = _encode(nonfinite)
        with self.assertRaisesRegex(ValueError, "<ipl_st>.*NaN/Inf"):
            _load_special_token_payload(
                payload,
                expected_sha256=digest,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
