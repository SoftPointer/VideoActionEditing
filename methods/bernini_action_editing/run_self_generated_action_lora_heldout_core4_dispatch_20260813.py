#!/usr/bin/env python3
"""Evaluation-only dispatcher admitting the two sealed SEER receipt readers."""

from __future__ import annotations

from pathlib import Path
import sys


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import run_self_generated_action_lora_heldout_core4_v1 as heldout


heldout.ADMITTED_TRAINED_INFER_RUNNERS = frozenset(
    {
        "infer_seer_scoped_lora.py",
        "infer_seer_same_state_lora.py",
    }
)


if __name__ == "__main__":
    raise SystemExit(heldout.main())
