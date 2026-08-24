#!/usr/bin/env python3
"""Run retry7 static checks that remain meaningful after old authorities are revoked."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


EXCLUDED = {
    "test_retry5_sealed_bytes_remain_unchanged",
    "test_retry6_sealed_bytes_remain_unchanged",
}


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_retry7_static_after_revocation_v1.py TEST_FILE")
    path = Path(sys.argv[1]).resolve(strict=True)
    spec = importlib.util.spec_from_file_location("retry7_static", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load retry7 static test")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        module.StageBT0SingleUpdateStaticContractTest
    )
    kept = unittest.TestSuite(
        test for test in suite if test._testMethodName not in EXCLUDED
    )
    print(
        "STATIC_HISTORICAL_RAW_AUTHORITY_TESTS_EXCLUDED_BY_RUNTIME_REVOCATION="
        + ",".join(sorted(EXCLUDED))
    )
    result = unittest.TextTestRunner(verbosity=1).run(kept)
    if not result.wasSuccessful() or result.testsRun != 15:
        return 1
    print("RETRY7_AUH_STATIC15_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
