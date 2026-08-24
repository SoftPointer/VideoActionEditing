#!/usr/bin/env python3
"""Postflight the entrypoint-fixed P-Q-MOSAIC exact81 artifacts."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import p_qmosaic_direction_envelope_v1 as profile  # noqa: E402
import postflight_qmosaic_editor_direction_v1 as shared  # noqa: E402


POSTFLIGHT_SCHEMA = profile.POSTFLIGHT_SCHEMA


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = shared.build_parser().parse_args(argv)
    validated = shared.validate_run_artifacts(
        run_receipt_path=args.run_receipt,
        expected_run_receipt_file_sha256=args.expected_run_receipt_sha256,
        artifact_root=args.artifact_root,
        _p_qmosaic=True,
    )
    receipt = shared.build_postflight_receipt(validated, _p_qmosaic=True)
    if (
        receipt.get("schema_version") != profile.POSTFLIGHT_SCHEMA
        or receipt.get("method_name") != profile.METHOD_NAME
        or receipt.get("direction_variant") != profile.variant_lock()
    ):
        raise shared.QMosaicDirectionPostflightError(
            "P-Q postflight profile lock differs"
        )
    output = Path(args.output)
    if (
        output.name != shared.POSTFLIGHT_FILENAME
        or output.parent.resolve(strict=True) != Path(validated["artifact_root"])
    ):
        raise shared.QMosaicDirectionPostflightError("postflight output path differs")
    try:
        shared.runtime._write_create_only_json(output, receipt)  # noqa: SLF001
    except shared.runtime.QMosaicEditorDirectionError as error:
        raise shared.QMosaicDirectionPostflightError(str(error)) from error
    print(  # noqa: SLF001
        shared.runtime._canonical_json_bytes(receipt).decode("ascii"), flush=True
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = ["POSTFLIGHT_SCHEMA", "main"]
