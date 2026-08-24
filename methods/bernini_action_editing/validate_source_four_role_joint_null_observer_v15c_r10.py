#!/usr/bin/env python3
"""Validate the v15c-r10 registry and, optionally, one local tensor artifact.

With no artifact arguments this validates only the preregistered plan and
prints its explicit NO-GO status.  It never launches the model, a route, a
decoder, an optimizer, or a remote job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    from . import source_four_role_joint_null_observer_v15c_r10 as core
except ImportError:  # pragma: no cover - flat local invocation
    import source_four_role_joint_null_observer_v15c_r10 as core


class ValidateFourRoleJointNullObserverV15CR10Error(RuntimeError):
    """CLI arguments or the local r10 observer artifact differ."""


def validate(
    *,
    registry_path: Path = core.DEFAULT_REGISTRY_ASSET,
    artifact_directory: Path | None = None,
    receipt_path: Path | None = None,
    expected_capture_channel_value_binding_sha256: str | None = None,
):
    registry = core.load_joint_null_registry_v15c_r10(registry_path)
    if (artifact_directory is None) != (receipt_path is None):
        raise ValidateFourRoleJointNullObserverV15CR10Error(
            "artifact directory and receipt must be supplied together"
        )
    if artifact_directory is None:
        return core.current_no_tensor_status_v15c_r10()
    artifact = core.load_joint_null_artifact_v15c_r10(
        artifact_directory,
        receipt_path,
        registry=registry,
    )
    result = core.validate_loaded_joint_null_artifact_v15c_r10(
        artifact,
        registry=registry,
        expected_capture_channel_value_binding_sha256=(
            expected_capture_channel_value_binding_sha256
        ),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate source-only v15c-r10 joint-null evidence"
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=core.DEFAULT_REGISTRY_ASSET,
    )
    parser.add_argument("--artifact-directory", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--expected-capture-channel-value-binding-sha256")
    args = parser.parse_args(argv)
    result = validate(
        registry_path=args.registry,
        artifact_directory=args.artifact_directory,
        receipt_path=args.receipt,
        expected_capture_channel_value_binding_sha256=(
            args.expected_capture_channel_value_binding_sha256
        ),
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
