#!/usr/bin/env python3
"""Run the fixed nuisance-projected P-Q-MOSAIC exact81 WORLD4 canary.

This versioned entrypoint has no direction-variant, seed-selection, dose,
sign, arm, callback, or semantic-selection option.  It delegates all owner,
editor-packet, checkpoint, WORLD4, zero-route, VJP, terminal-seal, decode, and
publication work to the audited Q-MOSAIC runner and fixes the P-Q profile in
code before argument parsing.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import run_qmosaic_editor_direction_sp4_v1 as _runtime  # noqa: E402
import p_qmosaic_direction_envelope_v1 as _profile  # noqa: E402


METHOD_NAME = _profile.METHOD_NAME
RUN_RECEIPT_SCHEMA = _profile.RUN_RECEIPT_SCHEMA
DIRECTION_VARIANT_ID = _profile.VARIANT_ID


def main(argv: Optional[Sequence[str]] = None) -> int:
    return _runtime.main_p_qmosaic(argv)


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "DIRECTION_VARIANT_ID",
    "METHOD_NAME",
    "RUN_RECEIPT_SCHEMA",
    "main",
]
