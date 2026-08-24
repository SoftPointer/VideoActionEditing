#!/usr/bin/env python3
"""Pre-registered T1/IAVG/I1/I1A exact81 SSFT mechanism screen.

This is a narrow, versioned arm registry over the sealed-coordinate runner in
``infer_saic_source_state_flow_transport_v2``.  It restores exactly the four
unrun arms registered in ``bernini_saic_source_state_flow_transport_v3.md``;
it does not introduce a new scientific treatment.

Every invocation consumes the immutable source-clean coordinate produced by
Slurm job 132387 and one fresh, immutable source-RGB-frame-0 coordinate shared
by every arm for that source.  The delegated runner performs zero VAE encodes.
The output is a frozen diagnostic only: no selection, training, optimizer,
checkpoint, semantic, identity, quality, or production authority is granted.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import infer_saic_source_state_flow_transport_v2 as _impl


SCHEMA_VERSION = "bernini-saic-ssft-preregistered-mechanism-screen-v1"
METHOD = "frozen-bernini-saic-ssft-preregistered-mechanism-screen"

ArmSpec = _impl.ArmSpec
SAICInferenceError = _impl.SAICInferenceError

_ARM_SPECS = (
    ArmSpec(
        "T1",
        "t2v_apg",
        "t2v",
        "t2v_apg",
        _impl.K5_EARLY_SCHEDULE,
        True,
        "source_similarity_softmax",
        0.01,
        False,
    ),
    ArmSpec(
        "IAVG",
        "r2v_apg_source_i0",
        "r2v",
        "r2v_apg",
        _impl.K5_EARLY_SCHEDULE,
        True,
        "uniform",
        None,
        False,
    ),
    ArmSpec(
        "I1",
        "r2v_apg_source_i0",
        "r2v",
        "r2v_apg",
        _impl.K5_EARLY_SCHEDULE,
        True,
        "source_similarity_softmax",
        0.01,
        False,
    ),
    ArmSpec(
        "I1A",
        "r2v_apg_source_i0",
        "r2v",
        "r2v_apg",
        _impl.K5_EARLY_SCHEDULE,
        True,
        "source_similarity_softmax",
        0.01,
        True,
    ),
)
_REGISTERED_ARM_SPECS: Mapping[str, ArmSpec] = MappingProxyType(
    {item.arm: item for item in _ARM_SPECS}
)
ARM_NAMES = tuple(item.arm for item in _ARM_SPECS)
ARM_SPECS = _REGISTERED_ARM_SPECS


def arm_spec(name: str) -> ArmSpec:
    """Resolve only a pre-registered mechanism-screen arm."""

    try:
        return _REGISTERED_ARM_SPECS[name]
    except (KeyError, TypeError) as error:
        raise SAICInferenceError(
            f"arm must be one of {ARM_NAMES}, got {name!r}"
        ) from error


def _configure_delegate() -> None:
    """Install this immutable registry into the sealed-coordinate runtime."""

    runtime_files = tuple(
        item
        for item in _impl.RUNTIME_METHOD_FILES
        if item != "infer_saic_ssft_mechanism_screen_v1.py"
    ) + ("infer_saic_ssft_mechanism_screen_v1.py",)
    _impl.SCHEMA_VERSION = SCHEMA_VERSION
    _impl.METHOD = METHOD
    _impl._ARM_SPECS = _ARM_SPECS
    _impl._REGISTERED_ARM_SPECS = _REGISTERED_ARM_SPECS
    _impl._REGISTERED_ARM_NAMES = ARM_NAMES
    _impl.ARM_SPECS = ARM_SPECS
    _impl.ARM_NAMES = ARM_NAMES
    _impl.arm_spec = arm_spec
    _impl.RUNTIME_METHOD_FILES = runtime_files
    _impl.RUNTIME_ARCHIVE_MEMBERS = tuple(
        f"methods/bernini_action_editing/{relative}"
        for relative in runtime_files
    )


_configure_delegate()


def build_parser() -> Any:
    return _impl.build_parser()


def validate_cli(args: Any) -> ArmSpec:
    return _impl.validate_cli(args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_delegate()
    return _impl.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_NAMES",
    "ARM_SPECS",
    "ArmSpec",
    "METHOD",
    "SAICInferenceError",
    "SCHEMA_VERSION",
    "arm_spec",
    "build_parser",
    "main",
    "validate_cli",
]
