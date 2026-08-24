#!/usr/bin/env python3
"""Decode a sealed v16r3 checkpoint with the real online-anchor controller.

The shared event runner already owns the sampling-time target-owned Q/K route,
SGA/ANC, and its capture/replay closure checks.  This entrypoint replaces only
its historical checkpoint schema validator and its plain-model freeze audit.
It deliberately leaves the runner's CLI, sampler, output schema, and route-off
causal-control option unchanged.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_anchor_sga_anc_trained_editor_decode_v1 as trained_editor  # noqa: E402
import v16r3_checkpoint_contract as contract  # noqa: E402


_ACTIVE_CHECKPOINT: dict[str, Any] | None = None
_ACTIVE_RUNNER: Any = None


def _runner_module() -> Any:
    global _ACTIVE_RUNNER

    if _ACTIVE_RUNNER is None:
        # The event runner imports Torch at module load.  Keep this wrapper
        # importable on login/test machines and resolve it only for execution.
        import infer_anchor_sga_anc_event_v1 as resolved

        _ACTIVE_RUNNER = resolved
    return _ACTIVE_RUNNER


def _fail(message: str) -> None:
    runner = _runner_module()
    raise runner.AnchorEventInferenceError(message)


def _attention_lora_checkpoint(
    value: str,
    *,
    expected_global_step: int = 0,
    expected_training_objective: str = "",
    expected_route_operator: str = "",
    expected_adapter_model_sha256: str = "",
    expected_adapter_config_sha256: str = "",
    expected_receipt_sha256: str = "",
) -> Optional[dict[str, Any]]:
    """Authenticate a v16r3 adapter for the existing unmerged PEFT path."""

    global _ACTIVE_CHECKPOINT

    runner = _runner_module()

    expectations = (
        expected_global_step,
        expected_training_objective,
        expected_route_operator,
        expected_adapter_model_sha256,
        expected_adapter_config_sha256,
        expected_receipt_sha256,
    )
    if not value:
        if any(expectations):
            _fail("trained attention expectations require a checkpoint")
        _ACTIVE_CHECKPOINT = None
        return None
    if expected_training_objective != contract.TRAINING_OBJECTIVE:
        _fail("v16r3 trained attention objective differs")
    if expected_route_operator != contract.ROUTE_OPERATOR:
        _fail("v16r3 trained attention route operator differs")
    try:
        contract.validate_runtime_versions()
        authenticated = contract.authenticate_checkpoint(
            value,
            expected_global_step=expected_global_step,
            expected_adapter_config_sha256=expected_adapter_config_sha256,
            expected_adapter_model_sha256=expected_adapter_model_sha256,
            expected_training_receipt_sha256=expected_receipt_sha256,
            include_model=True,
        )
    except contract.V16R3CheckpointContractError as error:
        raise runner.AnchorEventInferenceError(str(error)) from error

    bundle = authenticated["bundle"]
    # Match the runner's pre-existing binding vocabulary so downstream
    # postflight can compare old v14r2 and v16r3 decodes field-for-field.
    binding = {
        "receipt_sha256": authenticated["training_receipt_sha256"],
        "adapter_config_sha256": authenticated["adapter_config_sha256"],
        "adapter_model_sha256": authenticated["adapter_model_sha256"],
        "global_step": authenticated["global_step"],
        "training_objective": authenticated["training_objective"],
        "route_operator": authenticated["route_operator"],
        "required_decode_transport": authenticated[
            "required_decode_transport"
        ],
    }
    binding_sha256 = hashlib.sha256(runner._canonical_json(binding)).hexdigest()
    _ACTIVE_CHECKPOINT = authenticated
    return {
        "root": bundle.checkpoint_root,
        "adapter_dir": bundle.adapter_dir,
        "receipt": authenticated["training_receipt"],
        "receipt_sha256": authenticated["training_receipt_sha256"],
        "adapter_config_sha256": authenticated["adapter_config_sha256"],
        "model_sha256": authenticated["adapter_model_sha256"],
        "target_modules_sha256": authenticated["target_modules_sha256"],
        "global_step": authenticated["global_step"],
        "schema_version": authenticated["training_schema_version"],
        "training_objective": authenticated["training_objective"],
        "route_operator": authenticated["route_operator"],
        "required_decode_transport": authenticated[
            "required_decode_transport"
        ],
        "binding": binding,
        "binding_sha256": binding_sha256,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _ACTIVE_CHECKPOINT, _ACTIVE_RUNNER

    runner = _runner_module()
    original_checkpoint = runner._attention_lora_checkpoint
    original_certificate = runner.source_audit.model_freeze_certificate
    replacement_certificate = trained_editor.build_model_freeze_certificate(
        original_certificate,
        error_type=runner.AnchorEventInferenceError,
    )
    _ACTIVE_CHECKPOINT = None
    runner._attention_lora_checkpoint = _attention_lora_checkpoint
    runner.source_audit.model_freeze_certificate = replacement_certificate
    try:
        return runner.main(argv)
    finally:
        runner.source_audit.model_freeze_certificate = original_certificate
        runner._attention_lora_checkpoint = original_checkpoint
        _ACTIVE_CHECKPOINT = None
        _ACTIVE_RUNNER = None


if __name__ == "__main__":
    raise SystemExit(main())
