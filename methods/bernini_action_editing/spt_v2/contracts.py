"""Auditable module and checkpoint contracts for SPT-v2."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .phase_transport import PhaseTransportConfig, PhaseTransportError


SPT_LORA_SCOPES = ("cross_q_out", "cross_mid_q_out", "q_out")
PLANNER_SCHEMA = "bernini-spt-v2-planner-v1"
_MODULE = re.compile(
    r"^(?P<prefix>.+\.blocks\.(?P<block>\d+)\.attn(?P<attention>[12]))\."
    r"(?P<projection>to_q|to_k|to_v|to_out\.0)$"
)


def select_spt_lora_scope(
    available: Sequence[str],
    scope: str,
    *,
    middle_blocks: tuple[int, int] = (7, 22),
) -> list[str]:
    """Select the conservative v2 LoRA scope without changing CDF-v1."""

    if scope not in SPT_LORA_SCOPES:
        raise PhaseTransportError(f"unknown SPT LoRA scope: {scope!r}")
    start, end = middle_blocks
    if not 0 <= start <= end < 30:
        raise PhaseTransportError("middle block range must lie in [0,29]")
    selected = []
    for name in available:
        match = _MODULE.fullmatch(name)
        if match is None:
            raise PhaseTransportError(f"unexpected attention module: {name}")
        attention = int(match.group("attention"))
        block = int(match.group("block"))
        projection = match.group("projection")
        q_out = projection in ("to_q", "to_out.0")
        keep = {
            "cross_q_out": attention == 2 and q_out,
            "cross_mid_q_out": q_out
            and (attention == 2 or (attention == 1 and start <= block <= end)),
            "q_out": q_out,
        }[scope]
        if keep:
            selected.append(name)
    expected = {
        "cross_q_out": 60,
        "cross_mid_q_out": 60 + 2 * (end - start + 1),
        "q_out": 120,
    }[scope]
    selected = sorted(selected)
    if len(selected) != expected:
        raise PhaseTransportError(
            f"scope {scope} selected {len(selected)} modules, expected {expected}"
        )
    return selected


def planner_config_digest(config: PhaseTransportConfig) -> str:
    config.validate()
    payload = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def joint_trainable_parameters(peft_model: Any, planner: Any) -> list[tuple[str, Any]]:
    """Return exactly LoRA and planner parameters; reject base-model leakage."""

    model_named = [
        (f"lora::{name}", parameter)
        for name, parameter in peft_model.named_parameters()
        if parameter.requires_grad
    ]
    leaked = [name for name, _ in model_named if "lora_A" not in name and "lora_B" not in name]
    if leaked:
        raise PhaseTransportError(f"non-LoRA Bernini parameters are trainable: {leaked[:4]}")
    planner_named = [
        (f"planner::{name}", parameter)
        for name, parameter in planner.named_parameters()
        if parameter.requires_grad
    ]
    if not model_named or not planner_named:
        raise PhaseTransportError("joint training requires both LoRA and planner parameters")
    names = [name for name, _ in model_named + planner_named]
    if len(names) != len(set(names)):
        raise PhaseTransportError("joint trainable parameter names are not unique")
    return model_named + planner_named


def save_planner(
    planner: Any,
    output: Path,
    *,
    config: PhaseTransportConfig,
    global_step: int,
) -> dict[str, Any]:
    """Save planner separately from PEFT, preserving exact state-key scope."""

    from safetensors.torch import save_file

    if type(global_step) is not int or global_step <= 0:
        raise PhaseTransportError("planner global_step must be positive")
    output.mkdir(parents=True, exist_ok=False)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in planner.state_dict().items()}
    if not state:
        raise PhaseTransportError("planner state is empty")
    weights = output / "planner.safetensors"
    save_file(state, str(weights))
    receipt = {
        "schema_version": PLANNER_SCHEMA,
        "global_step": global_step,
        "config": asdict(config),
        "config_sha256": planner_config_digest(config),
        "state_keys": sorted(state),
        "state_keys_sha256": hashlib.sha256(
            json.dumps(sorted(state), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    (output / "planner_config.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def validate_planner_receipt(value: Mapping[str, Any]) -> PhaseTransportConfig:
    if value.get("schema_version") != PLANNER_SCHEMA:
        raise PhaseTransportError("planner receipt schema differs")
    raw = value.get("config")
    if not isinstance(raw, dict):
        raise PhaseTransportError("planner receipt lacks config")
    # JSON converts tuples to lists; restore the two finite candidate sets.
    raw = dict(raw)
    for key in ("teacher_temporal_offsets", "teacher_spatial_offsets"):
        if isinstance(raw.get(key), list):
            raw[key] = tuple(raw[key])
    config = PhaseTransportConfig(**raw)
    if value.get("config_sha256") != planner_config_digest(config):
        raise PhaseTransportError("planner config digest differs")
    keys = value.get("state_keys")
    if not isinstance(keys, list) or not keys or keys != sorted(set(keys)):
        raise PhaseTransportError("planner state-key list is invalid")
    digest = hashlib.sha256(
        json.dumps(keys, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if value.get("state_keys_sha256") != digest:
        raise PhaseTransportError("planner state-key digest differs")
    return config
